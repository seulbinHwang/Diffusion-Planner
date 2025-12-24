import os
from typing import Optional, Tuple
# 128 MiB 단위로 메모리 청크를 잘라서 할당하도록 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import args_util
# DDP 디버깅을 위해 사용되지 않은 파라미터 정보를 상세히 출력
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
import torch
from typing import Any, Callable, Dict, List
import numpy as np
# deprecated 키는 사용 금지
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
from collections import defaultdict
from datetime import datetime
from typing import Optional, Dict, Any
import io
from nuplan_extent.planning.training.preprocessing.utils.near_agents import add_near_agents_info_inplace

# TensorFloat-32(TF32) 연산을 허용하여
#   - Ampere(A100 등) GPU에서 matmul/cuDNN 연산을 FP32보다 빠르게 처리하고
#   - 눈에 띄는 정밀도 손실 없이 학습·추론 속도를 높이기 위한 설정입니다.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # PyTorch>=2.0
import sys, os, faulthandler, traceback

faulthandler.enable(all_threads=True)
from timm.optim.optim_factory import param_groups_weight_decay

import argparse
import shutil
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from requests.exceptions import HTTPError
from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import (
    build_pytorch_warmup_cosine_scheduler,
    build_pytorch_warmup_constant_scheduler,
)

from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp
import time
from diffusion_planner.train_epoch import train_epoch
import os

import math

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass
import _hashlib
import hashlib
from typing import Iterable
from wandb.sdk.lib import hashutil as wandb_hashutil


def md5_file_hasher_no_mmap(*paths: str) -> _hashlib.HASH:
    """Ceph 환경에서 mmap(SIGBUS) 피하기 위한 W&B용 MD5 해시 함수."""
    md5_hash: _hashlib.HASH = wandb_hashutil._md5()

    for path in sorted(map(str, paths)):
        with open(path, "rb") as file_obj:
            while True:
                chunk: bytes = file_obj.read(wandb_hashutil._CHUNKSIZE)
                if not chunk:
                    break
                md5_hash.update(chunk)

    return md5_hash


# W&B 내부 해시 함수 갈아끼우기 (mmap 사용 금지)
wandb_hashutil._md5_file_hasher = md5_file_hasher_no_mmap  # type: ignore[attr-defined]


def build_deepspeed_config(args: argparse.Namespace,
                           current_global_batch: int) -> Dict[str, Any]:
    """args 값을 조합해서 DeepSpeed ZeRO-2 설정 dict를 만든다.

    처리 흐름:
      1) WORLD_SIZE(프로세스 수)를 읽어서, GPU 한 장당 미니배치 크기를 정한다.
      2) args.grad_accum_steps, args.max_grad_norm 같은 값을 읽어서
         한 번에 얼마나 gradient를 모을지, 얼마나 잘라낼지 결정한다.
      3) ZeRO-2 단계(stage=2)를 켜고,
         · gradient와 옵티마 상태를 GPU끼리 나눠 들도록 설정한다.
      4) 필요하면 zero_offload_optimizer 를 써서
         옵티마 상태 일부를 CPU 메모리로 옮겨 GPU 메모리를 더 아낀다.

    이렇게 만든 ds_config dict는 train_predictor.py에서
    deepspeed.initialize(..., config=ds_config) 에 그대로 넘겨서 사용한다.
    """
    # --- 1) world_size 계산 (torchrun이 넘겨준 값 우선 사용) ---
    world_size_str = os.environ.get("WORLD_SIZE")
    if world_size_str is not None:
        try:
            world_size = max(1, int(world_size_str))
        except ValueError:
            world_size = 1
    else:
        world_size = int(getattr(args, "world_size", 1))
        if world_size <= 0:
            world_size = 1
    micro_batch_size = max(1, current_global_batch // world_size)

    # --- 3) args에서 세부 설정값 읽기 ---
    """ grad_accum_steps
    실질 배치 크기 = micro_batch_size * world_size * grad_accum_steps 
    
    grad_accum_steps 커지면 좋은 점 (배치 사이즈를 키우는 것과 같은 효과)
    
    batch size를 키우는 것에 비해, grad_accum_steps를 키우는 것의 단점
      - 학습 속도가 느리다.
    """
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))

    zero_offload_optimizer = bool(getattr(args, "zero_offload_optimizer",
                                          False))
    """
    allgather_bucket_size : 
        weight 조각들을 다른 GPU에서 가져올 때
        한 번에 얼마나 큰 묶음으로 모아서 보내고 받을지 크기.
    reduce_bucket_size
        기울기 값을 여러 GPU에서 합칠 때 한 번에 묶는 크기.
        
    둘 다 크게 설정할수록
        통신 횟수 ↓
        한 번에 처리하는 메모리 ↑ (피크 메모리 조금 증가 가능)
    """
    allgather_bucket_size = float(getattr(args, "ds_allgather_bucket_size",
                                          2e8))
    reduce_bucket_size = float(getattr(args, "ds_reduce_bucket_size", 2e8))

    steps_per_print = int(getattr(args, "ds_steps_per_print", 100))

    # --- 4) ZeRO-2 중심 DeepSpeed 설정 dict ---
    """        
        stage: 2 -> ZeRO 단계 2.
            기울기와 옵티마 상태를 GPU 사이에 나눠 들고,
            weight 자체는 여전히 GPU마다 풀 복사본을 유지하는 단계.
            
        allgather_partitions: True
            “기울기/옵티마 나누고 모으는 방식”을 미세하게 조절하는 옵션
            True
            False
                “나눠진 기울기/옵티마를 꽤 자주 전체 형태로 모아서 쓰는 쪽”
        
        allgather_bucket_size: 
            weight 조각들을 다른 GPU에서 가져올 때
            한 번에 얼마나 큰 묶음으로 모아서 보내고 받을지 크기.
        
        reduce_scatter: True
        
            기울기를 각 GPU에서 계산한 뒤
            합치고 다시 나누는 과정을 한 번에 처리하는 모드.
            기울기 통신 비용을 줄이기 위한 방식이라고 보면 된다.
        
        reduce_bucket_size: reduce_bucket_size
                    기울기 값을 여러 GPU에서 합칠 때 한 번에 묶는 크기.
        
        overlap_comm: True
        
            통신(다른 GPU와 데이터 주고받기)과
            연산(다음 레이어 계산)을 가능한 한 동시에 진행하도록 시도한다.
        
        속도 향상을 노리는 옵션.
        
        contiguous_gradients: True
        
            기울기 텐서들을 메모리 상에서 연속된 큰 덩어리로 재배치한다.
            
            이렇게 하면 통신이나 클리핑 같은 작업에서
            여러 텐서를 따로따로 다루지 않아도 되어서 효율이 좋아진다.
    """
    ds_config: Dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_clipping": max_grad_norm,
        "torch_autocast": {
            "enabled": True,
            # 둘 중 하나 선택:
            "dtype": "bfloat16",  # 또는 "float16"
            # (선택) 이 목록을 안 주면 DeepSpeed 기본 목록을 씀 :contentReference[oaicite:1]{index=1}
            # "lower_precision_safe_modules": ["torch.nn.Linear", "torch.nn.Conv2d"],
        },
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 2,  # ZeRO-2
            "allgather_partitions": True,
            "allgather_bucket_size": allgather_bucket_size,
            "reduce_scatter": True,
            "reduce_bucket_size": reduce_bucket_size,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "steps_per_print": steps_per_print,
    }

    # 옵티마 상태를 CPU로 일부 넘겨서 GPU 메모리를 더 줄이고 싶을 때
    if zero_offload_optimizer:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return ds_config


# === [NEW] LR auto-scaling (root scale for Adam/AdamW) ======================
def _effective_global_batch(batch_size: int, world_size: int) -> int:
    """DataLoader가 사용하는 실제 글로벌 배치(DDP 정합)"""
    world_size = max(1, int(world_size))
    per_rank = max(1, batch_size // world_size)
    return per_rank * world_size  # drop_last 정합 반영


# === [NEW] Epoch auto-scaling by global batch ===============================
def _auto_scale_train_epochs(
    base_global_batch: int,  # B_0 = 2048
    base_epochs: int,  # E_0 = 500
    current_global_batch: int,  # B = 실제 글로벌 배치
    beta: float = 0.4,
    clamp_min: Optional[int] = 1,
    clamp_max: Optional[int] = None,
) -> int:
    """
    E = base_epochs * (current_global_batch / base_global_batch) ** beta
    - beta=0.4 (AdamW에서 '빠르게' 쪽으로 살짝 치우친 값)
    - clamp_min/clamp_max: 필요 없는 경우 None
    """
    if current_global_batch <= 0 or base_global_batch <= 0:
        raise ValueError("Global batch sizes must be positive integers.")
    k = float(current_global_batch) / float(base_global_batch)  # B / B_0
    scaled_epochs = int(round(float(base_epochs) *
                              (k**float(beta))))  # E_0 * (B / B_0)^β
    if clamp_min is not None:
        scaled_epochs = max(int(clamp_min), scaled_epochs)
    if clamp_max is not None:
        scaled_epochs = min(int(clamp_max), scaled_epochs)
    return scaled_epochs


def pnn_schedule(
    epoch: int,
    total_epochs: int,
    *,
    base: int = 32,  # 시작 Pnn
    cap: int,  # 최종 Pnn (보통 args.predicted_neighbor_num)
    step: int = 32,  # Pnn 증분 단위(캐시/모델 구조와 맞추기)
    ramp_fraction: float = 0.30  # 총 학습의 몇 % 지점에서 cap 도달할지
) -> int:
    """
    커리큘럼: [0 .. ramp_end-1] 구간에서 base→cap을 'step' 단위로 균등 분할해서 올리고,
             ramp_end 이후엔 cap으로 고정한다.
    - epoch: 0-indexed
    - total_epochs: 전체 학습 epoch 수(예: 500)
    - base/cap/step: (base + k*step)이 cap을 넘지 않도록 설계
    - ramp_fraction: cap에 도달할 시점(비율). 예: 0.30 → 500×0.30≈150epoch에 cap 도달

    장점:
    - 학습 초기 충분한 '쉬운 구간'을 확보 (안정·속도 유리)
    - 전체의 60~80%는 풀 난이도에서 수렴 (성능 유지)
    """
    cap = int(cap)
    base = int(base)
    step = int(step)
    total_epochs = int(total_epochs)
    ramp_end = max(1, int(round(
        total_epochs * ramp_fraction)))  # cap 도달 epoch (0-index 기준 ramp_end-1)

    if cap <= base:
        return cap  # 방어: 이미 최대 이하라면 그대로

    steps_needed = math.ceil((cap - base) / step)  # 몇 번 올려야 cap에 도달하는지
    # 0..(ramp_end-1) 구간을 steps_needed 칸으로 균등 분할하여 인덱스 계산
    # epoch=ramp_end-1일 때 idx=steps_needed가 되도록 설계
    idx = 0 if ramp_end == 0 else min(
        steps_needed, math.floor(((epoch + 1) / ramp_end) * steps_needed))

    pnn = base + idx * step
    if pnn > cap:
        pnn = cap
    return int(pnn)


# === [ADD] Pnn 기반 배치 크기 스케줄러 ==================================
def bs_schedule_by_pnn(pnn_curr: int,
                       pnn_base: int,
                       bs_base: int,
                       *,
                       world_size: int = 1,
                       alpha: float = 1.5,
                       min_per_rank: int = 1,
                       max_global: int = None):
    """
    메모리 예산을 대략 일정하게 유지하기 위해, Pnn(=시퀀스 길이)↑ 시 global batch를 줄입니다.
      - pnn_base: pnn_schedule의 시작값(예: 32)과 일치시킬 것
      - bs_base : 기준 '글로벌' 배치 크기(args.batch_size)
      - alpha   : 메모리 증가 근사 지수(1~2 권장; self-attn 지배면 2에 가까움)
    반환:
      - (bs_global_now, bs_per_rank_now)
    """
    if pnn_curr <= 0:
        pnn_curr = pnn_base
    scale = (float(pnn_base) / float(pnn_curr))**float(alpha)
    bs_global = int(max(world_size, round(bs_base * scale)))
    if max_global is not None:
        bs_global = min(bs_global, int(max_global))
    # DDP 정합: world_size 배수로 내림
    bs_global = (bs_global // world_size) * world_size
    bs_per_rank = max(min_per_rank, bs_global // world_size)
    return bs_global, bs_per_rank


# =======================================================================


def print_parameter_index_mapping(model):
    print("Parameter index mapping:")
    for idx, (name, _) in enumerate(model.named_parameters()):
        print(f"{idx}: {name}")


def safe_get_artifacts(api, type_name, path):
    try:
        return list(api.artifacts(type_name, path))
    except (wandb.errors.CommError, HTTPError) as e:
        # 404 또는 권한 오류 → 컬렉션이 아직 없다고 판단
        print(f"[SKIP] '{path}' 컬렉션 없음/권한 문제: {e}")
        return []


def purge_collection(api, entity, project, coll_name):
    path = f"{entity}/{project}/{coll_name}"
    versions = safe_get_artifacts(api, "model", path)  # 신규 API 사용
    if not versions:
        print(f"[PURGE] {coll_name}: 삭제할 버전이 없습니다.")
        return

    deleted_count = 0
    failed_count = 0

    for art in versions:
        try:
            # alias가 있는 artifact의 경우 alias를 먼저 제거
            if hasattr(art, 'aliases') and art.aliases:
                print(
                    f"[PURGE] {coll_name}: Artifact {art.id}에 alias가 있어 alias를 먼저 제거합니다: {art.aliases}"
                )
                for alias in art.aliases:
                    try:
                        art.delete_alias(alias)
                        print(f"[PURGE] {coll_name}: Alias '{alias}' 제거 완료")
                    except Exception as alias_err:
                        print(
                            f"[PURGE] {coll_name}: Alias '{alias}' 제거 실패: {alias_err}"
                        )

            # artifact 삭제 시도
            art.delete()
            deleted_count += 1
            print(f"[PURGE] {coll_name}: Artifact {art.id} 삭제 완료")

        except Exception as e:
            failed_count += 1
            print(f"[PURGE] {coll_name}: Artifact {art.id} 삭제 실패 - {str(e)}")
            # alias가 있는 경우의 오류는 경고로만 처리하고 계속 진행
            if "due to existing alias" in str(e):
                print(
                    f"[PURGE] {coll_name}: Alias로 인한 삭제 실패는 정상적인 상황입니다. 계속 진행합니다."
                )

    print(f"[PURGE] {coll_name}: 삭제 완료 {deleted_count}개, 실패 {failed_count}개")


def _prune_old_wandb_artifact_versions(
    collection_name: str,
    alias: str,
) -> None:
    """지정한 모델 묶음에서 현재 버전을 제외하고 나머지 W&B 아티팩트를 지운다.

    학습 도중에 너무 많은 버전이 쌓이지 않도록,
    방금 올린 버전(alias 기준)만 남기고 나머지는 삭제한다.

    Args:
        collection_name: W&B에서 모델을 묶을 때 쓰는 이름.
            예: f"{args.name}_latest-model", f"{args.name}_best-model".
        alias: 남기고 싶은 버전에 붙인 별칭.
            예: "latest", "best".

    Returns:
        None: 삭제 작업만 수행하고 값을 돌려주지 않는다.
    """
    # 아직 run이 없으면(초기화 전이거나 offline 특수 상황 등) 아무 것도 하지 않는다.
    if not wandb.run:
        return

    try:
        api = wandb.Api()
        entity = wandb.run.entity
        project = wandb.run.project
        coll_path = f"{entity}/{project}/{collection_name}"

        # 현재 alias가 가리키는 버전
        current = api.artifact(f"{coll_path}:{alias}")

        # 이 컬렉션 안의 모든 버전 목록
        versions = safe_get_artifacts(api, "model", coll_path)
        if not versions:
            return

        for art in versions:
            if art.id == current.id:
                continue
            try:
                art.delete()
                print(
                    f"[PRUNE] {collection_name}: 이전 버전 {art.id} 삭제 완료 (alias={alias})"
                )
            except Exception as e:
                print(
                    f"[PRUNE] {collection_name}: 이전 버전 {art.id} 삭제 실패 - {str(e)}"
                )
    except Exception as e:
        print(f"[PRUNE] {collection_name}: 버전 정리 중 오류 발생 - {str(e)}")


# --- put this in a utils file or near your optimizer build code ---
from typing import List, Set
import torch
import torch.nn as nn
from timm.optim.optim_factory import param_groups_weight_decay
import bitsandbytes as bnb

# 토큰/포지션 계열에서 자주 쓰는 속성 이름들(모듈의 attribute로 존재하는 nn.Parameter)
_TOKEN_POS_ATTRS = (
    # 일반
    "cls_token",
    "class_token",
    "dist_token",
    # 위치/상대위치
    "cls_pos",
    "pos_embed",
    "absolute_pos_embed",
    "rel_pos",
    "rel_pos_bias",
    "relative_position_bias_table",
    # RoPE/ALiBi 류
    "rope",
    "alibi",
    "rotary_emb",
    "rotary_embedding",
)


def discover_extra_no_weight_decay_names(
    model: nn.Module,
    include_seed_params: bool = True,
) -> List[str]:
    """
    모델을 실제로 순회하여 '토큰/포지션'류 파라미터 이름만 수집합니다.
    - nn.Parameter 이면서 차원>=2 인 것만 채택 (행렬 아닌 토큰류)
    - Linear/Conv 등 submodule의 .weight/.bias는 제외
    - (옵션) seeds/ego_fut_seeds 같은 학습 쿼리 토큰도 포함
    """
    extra: Set[str] = set()

    # (1) 모듈 attribute로 붙은 nn.Parameter 후보 탐색
    for mod_name, mod in model.named_modules():
        for attr in _TOKEN_POS_ATTRS:
            if hasattr(mod, attr):
                p = getattr(mod, attr)
                if isinstance(p,
                              nn.Parameter) and p.requires_grad and p.ndim >= 2:
                    full_name = f"{mod_name}.{attr}" if mod_name else attr
                    extra.add(full_name)

        # (선택) 학습 쿼리 토큰(seeds 류)
        if include_seed_params:
            for attr in ("seeds", "ego_fut_seeds"):
                if hasattr(mod, attr):
                    p = getattr(mod, attr)
                    # 보통 형태: (1, K, H) → 첫 축이 batch-like
                    if isinstance(
                            p, nn.Parameter
                    ) and p.requires_grad and p.ndim >= 2 and p.shape[0] == 1:
                        full_name = f"{mod_name}.{attr}" if mod_name else attr
                        extra.add(full_name)

    # (2) timm 모델이 no_weight_decay() 제공하면 합치기 (관례)
    if hasattr(model, "no_weight_decay") and callable(
            getattr(model, "no_weight_decay")):
        try:
            extra.update(set(model.no_weight_decay()))
        except Exception:
            pass

    # DDP/랩핑 전개 여부와 무관하게 named_parameters() 기준의 풀네임과 일치시켜야 합니다.
    return sorted(extra)


# 바뀜
def _set_requires_grad_for_encoder_local(
    model: nn.Module,
    requires_grad: bool,
) -> None:
    """encoder_local(Group A) 파라미터들의 requires_grad 값을 일괄 변경한다.

    Args:
        model: Diffusion_Planner 또는 래핑된 모델(DDP/DeepSpeed 상관 없음).
        requires_grad: True 이면 학습 가능, False 이면 고정.
    """
    if not hasattr(model, "iter_group_encoder_local_parameters"):
        return

    for param in model.iter_group_encoder_local_parameters():
        if isinstance(param, nn.Parameter):
            param.requires_grad_(requires_grad)


def _build_param_role_mapping(model: nn.Module,) -> Dict[int, str]:
    """모델 파라미터 id → 역할 이름(encoder_local/global/decoder/default) 매핑을 만든다.

    Diffusion_Planner 처럼
      - iter_group_encoder_local_parameters()
      - iter_group_encoder_global_parameters()
      - iter_group_decoder_parameters()
    메서드를 제공하는 모델을 가정하고, 각 이터레이터에서 반환되는 파라미터에
    역할 이름을 붙인다.

    Args:
        model (nn.Module):
            - Diffusion_Planner 또는 동일한 인터페이스를 가진 모델.
            - 각 이터레이터는 nn.Parameter 텐서를 yield 한다.
              예: (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW).

    Returns:
        Dict[int, str]:
            - key: 파라미터 객체의 id(id(param)).
            - value: "encoder_local" / "encoder_global" / "decoder" / "default" 중 하나.
              "default"는 이 함수 내부에서는 직접 채우지 않고,
              이후 단계에서 역할을 못 찾은 파라미터에 대해 사용된다.
    """
    param_int_id_to_role_name_str: Dict[int, str] = {}

    # Group A: encoder_local
    if hasattr(model, "iter_group_encoder_local_parameters"):
        for p in model.iter_group_encoder_local_parameters():
            if isinstance(p, nn.Parameter):
                param_int_id_to_role_name_str[id(p)] = "encoder_local"

    # Group B: encoder_global
    if hasattr(model, "iter_group_encoder_global_parameters"):
        for p in model.iter_group_encoder_global_parameters():
            if isinstance(p, nn.Parameter):
                param_int_id_to_role_name_str[id(p)] = "encoder_global"

    # Group C: decoder
    if hasattr(model, "iter_group_decoder_parameters"):
        for p in model.iter_group_decoder_parameters():
            if isinstance(p, nn.Parameter):
                param_int_id_to_role_name_str[id(p)] = "decoder"

    return param_int_id_to_role_name_str


def _build_param_groups_with_roles(
    base_param_groups: List[Dict[str, Any]],
    param_int_id_to_role_name_str: Dict[int, str],
    lr: float,
    role_to_lr_scale: Dict[str, float],
) -> List[Dict[str, Any]]:
    """역할별 lr 배율을 적용한 AdamW param_group 리스트를 만든다.

    timm.param_groups_weight_decay 로 만든 기본 그룹(base_param_groups)을 받아서,
    각 그룹 안의 파라미터를 역할별로 다시 나누고 역할별 lr_scale 을 곱한다.

    Args:
        base_param_groups (List[Dict[str, Any]]):
            - timm.param_groups_weight_decay 가 만든 기본 그룹들.
            - 각 원소 g 에 대해:
                g["params"]: List[nn.Parameter]
                    · Linear weight: (out_dim, in_dim)
                    · bias/Norm weight: (dim,)
                    · Conv weight: (C_out, C_in, kH, kW)
                g["weight_decay"]: float
        param_int_id_to_role_name_str (Dict[int, str]):
            - _build_param_role_mapping 에서 만든 파라미터 id → 역할 이름 매핑.
        lr (float):
            - Stage 기준 학습률(Decoder/기본 그룹 기준).
        weight_decay (float):
            - 가중치 감쇠 값. decay 그룹에만 적용된다.
        role_to_lr_scale (Dict[str, float]):
            - 역할별 lr 배율.
            - key: "encoder_local" / "encoder_global" / "decoder" / "default"
            - value: lr에 곱해줄 배율.

    Returns:
        List[Dict[str, Any]]:
            - 최종 AdamW param_group 리스트.
            - 각 그룹 딕셔너리에는
                · "params": List[nn.Parameter]
                · "weight_decay": float
                · "lr": float
                · "lr_max": float
                · "wd_max": float
              필드가 포함된다.
    """
    final_param_groups: List[Dict[str, Any]] = []

    default_scale: float = float(role_to_lr_scale.get("default", 1.0))

    for base_pg in base_param_groups:
        wd_value: float = float(base_pg.get("weight_decay", 0.0))
        # params_in_base: [nn.Parameter], 각 텐서 shape는 레이어 종류에 따라 다름.
        params_in_base: List[nn.Parameter] = [
            p for p in base_pg["params"] if p.requires_grad
        ]
        if not params_in_base:
            continue

        # 역할별로 파라미터를 다시 나눈다.
        buckets: Dict[str, List[nn.Parameter]] = defaultdict(list)
        for p in params_in_base:
            role = param_int_id_to_role_name_str.get(id(p), "default")
            buckets[role].append(p)

        for role, params_in_role in buckets.items():
            if not params_in_role:
                continue
            scale: float = float(role_to_lr_scale.get(role, default_scale))
            lr_role: float = lr * scale
            pg: Dict[str, Any] = {
                "params": params_in_role,
                "weight_decay": wd_value,
                "lr": lr_role,
                "lr_max": float(lr_role),
                "wd_max": float(wd_value),
            }
            final_param_groups.append(pg)

    return final_param_groups


def _build_fallback_param_groups(
    base_param_groups: List[Dict[str, Any]],
    lr: float,
) -> List[Dict[str, Any]]:
    """역할 정보를 찾지 못한 경우 timm 기본 그룹을 그대로 사용하는 fallback을 만든다.

    역할 매핑이 비어 있거나 iter_group_* 인터페이스가 없는 모델에서도
    학습이 가능하도록, timm.param_groups_weight_decay 결과를
    그대로 AdamW param_group 형식으로 옮긴다.

    Args:
        base_param_groups (List[Dict[str, Any]]):
            - timm.param_groups_weight_decay 가 만든 그룹들.
        lr (float):
            - 기준 학습률.

    Returns:
        List[Dict[str, Any]]:
            - "lr_max" / "wd_max" 필드가 채워진 param_group 리스트.
    """
    final_param_groups: List[Dict[str, Any]] = []

    for base_pg in base_param_groups:
        wd_value: float = float(base_pg.get("weight_decay", 0.0))
        params_in_base: List[nn.Parameter] = [
            p for p in base_pg["params"] if p.requires_grad
        ]
        if not params_in_base:
            continue
        pg: Dict[str, Any] = {
            "params": params_in_base,
            "weight_decay": wd_value,
            "lr": lr,
            "lr_max": float(lr),
            "wd_max": float(wd_value),
        }
        final_param_groups.append(pg)

    return final_param_groups


# 변경
def build_adamw_with_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    include_seed_params: bool = True,
    use_8bit_optimizer: bool = False,
    role_to_lr_scale: Optional[Dict[str, float]] = None,
) -> Tuple[optim.Optimizer, List[str]]:
    """역할별 lr 배율이 반영된 AdamW(또는 8bit AdamW) 옵티마이저를 생성한다.

    전체 흐름은 다음과 같다.

    1) discover_extra_no_weight_decay_names 로
       토큰/포지션 계열 파라미터 이름(extra_nwd)을 찾아 weight decay에서 제외한다.
    2) timm.param_groups_weight_decay 를 호출해,
       - weight_decay를 적용할 그룹
       - 적용하지 않을 그룹
       으로 1차 그룹(base_param_groups)을 만든다.
       이때 각 그룹의 "params" 리스트 안에는
         · (out_dim, in_dim) shaped Linear weight
         · (dim,) shaped bias / Norm weight
         · (C_out, C_in, kH, kW) shaped Conv weight
       와 같은 텐서들이 들어 있다.
    3) Diffusion_Planner 가 제공하는
       iter_group_encoder_local_parameters / iter_group_encoder_global_parameters /
       iter_group_decoder_parameters 를 사용해
       각 파라미터를 역할 이름("encoder_local"/"encoder_global"/"decoder"/"default")
       으로 매핑한다.
    4) role_to_lr_scale 에 따라 역할별 lr 배율을 곱해
       최종 param_group 리스트를 만든다.
       - 각 그룹에는 "lr", "lr_max", "weight_decay", "wd_max" 가 포함된다.
    5) use_8bit_optimizer 플래그에 따라
       - bnb.optim.AdamW8bit 또는
       - torch.optim.AdamW
       인스턴스를 생성한다. 전역 weight_decay 는 0.0 로 두고,
       그룹별 weight_decay 만 사용한다.

    Args:
        model (nn.Module):
            - 학습 대상 모델.
            - Diffusion_Planner 처럼 encoder_local/global/decoder 그룹 이터레이터를
              제공하는 경우 역할별 lr 제어가 적용된다.
        lr (float):
            - 기준 학습률.
            - 역할별 lr_scale 은 이 값에 곱해진다.
        weight_decay (float):
            - 가중치 감쇠 값.
        include_seed_params (bool):
            - seeds/ego_fut_seeds 같은 학습 쿼리 토큰을
              no_weight_decay 목록에 포함할지 여부.
        use_8bit_optimizer (bool):
            - True 이면 bitsandbytes의 AdamW8bit 를 사용한다.
        role_to_lr_scale (Optional[Dict[str, float]]):
            - 역할별 lr 배율.
            - key: "encoder_local", "encoder_global", "decoder", "default"
            - value: lr 배율 (예: 0.1, 1.0 등).

    Returns:
        Tuple[optim.Optimizer, List[str]]:
            - optimizer:
                · torch.optim.AdamW 또는 bnb.optim.AdamW8bit 인스턴스.
                · optimizer.param_groups[i]["params"] 리스트 안에는
                  다양한 shape 의 nn.Parameter 텐서들이 들어 있다.
            - extra_nwd:
                · discover_extra_no_weight_decay_names 로 찾은
                  no_weight_decay 파라미터 이름 목록.
    """
    # 1) weight decay 에서 제외할 파라미터 이름 목록
    extra_nwd = discover_extra_no_weight_decay_names(
        model,
        include_seed_params=include_seed_params,
    )

    # 2) timm 기본 규칙으로 1차 그룹 생성
    # List[Dict[str]]
    base_param_groups = param_groups_weight_decay(
        model,
        weight_decay=weight_decay,
        no_weight_decay_list=extra_nwd,
    )

    role_to_lr_scale = role_to_lr_scale or {}

    # 3) 파라미터 id → 역할 매핑
    param_int_id_to_role_name_str: Dict[int,
                                        str] = _build_param_role_mapping(model)

    # 4) 역할별 lr 배율이 반영된 최종 그룹 생성
    final_param_groups = _build_param_groups_with_roles(
        base_param_groups=base_param_groups,
        param_int_id_to_role_name_str=param_int_id_to_role_name_str,
        lr=lr,
        role_to_lr_scale=role_to_lr_scale,
    )

    # 5) 역할 정보를 하나도 못 찾은 경우에는 timm 기본 그룹 그대로 사용
    if not final_param_groups:
        final_param_groups = _build_fallback_param_groups(
            base_param_groups=base_param_groups,
            lr=lr,
        )

    # 6) 옵티마이저 생성
    if use_8bit_optimizer:
        # 8비트 AdamW: 모멘트 텐서를 8bit 로 저장하지만,
        #   파라미터 텐서 shape는 (out_dim, in_dim), (dim,) 등 그대로 유지된다.
        optim_obj = bnb.optim.AdamW8bit(  # type: ignore[attr-defined]
            final_param_groups,
            lr=lr,
            weight_decay=0.0,
        )
    else:
        # 전역 WD는 0.0 으로 두고, 그룹별 WD만 사용
        try:
            optim_obj = torch.optim.AdamW(
                final_param_groups,
                fused=True,
                weight_decay=0.0,
            )
        except (TypeError, RuntimeError):
            optim_obj = torch.optim.AdamW(
                final_param_groups,
                weight_decay=0.0,
            )

    return optim_obj, extra_nwd



class DiffusionPlannerCollate:
    """DiffusionPlannerData 샘플들을 배치 텐서로 묶는 collate_fn.

    각 샘플은 agent / lane / route / static 개수에 상한이 걸려 있고,
    이 collate_fn 은
      1) (선택) 중심 기준 거리 크로핑
      2) 배치 내 최대 길이 계산
      3) 그 길이에 맞춰 0 패딩
    순서로 고정 shape 배치를 만든다.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """collate 설정을 초기화한다.

        Args:
            args (argparse.Namespace):
                - caching_max_agent_num (int)
                - caching_max_lane_num (int)
                - caching_max_static_num (int)
                - center_crop_radius_m (float, 선택)
                - center_crop_mode (str, 'npc' / 'ego' / 'none')
        """
        # 중심 기준 크로핑 옵션
        # - center_crop_radius_m <= 0: 크로핑 사용 안 함
        # - center_crop_mode: "npc" → 임의 NPC 1대를 기준, "ego" → ego 기준
        self.args = args
        self.center_crop_radius_m: float = float(
            getattr(args, "center_crop_radius_m", 0.0))
        self.center_crop_mode: str = str(
            getattr(args, "center_crop_mode", "none")).lower()

    def _infer_max_sizes_for_agent_route_lane_order(
            self,
            batch: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """agent_route_lane_order를 패딩하기 위한 (최대 agent 수, 최대 lane 수)를 계산합니다.

        왜 이 함수가 필요한가
        --------------------
        agent_route_lane_order는 (행=agent, 열=lane) 관계를 담고 있습니다.
        현재 로직에서는 agent_route_lane_order의 "행"이
        neighbor 전체가 아니라 near agent 기준으로 잘려서 들어올 수 있습니다.

        그래서 이 key를 배치로 패딩할 때,
        agent 축 최대값(max_agent_num)은 neighbor_agents_past 같은 다른 텐서가 아니라
        **agent_route_lane_order 자체의 행 개수**만 보고 정하는 것이 안전합니다.
        (그래야 agent_route_lane_order_is_valid 같은 agent 축 마스크와도 길이가 맞습니다.)

        lane 축(max_lane_num)은 lanes.shape[0]와 agent_route_lane_order.shape[1] 중
        더 큰 값을 사용해, lane 텐서 패딩과도 잘 맞도록 합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            Tuple[int, int]:
                - max_agent_num: shape (), int
                    agent_route_lane_order의 행 수(=agent 수) 최대값.
                - max_lane_num: shape (), int
                    lanes의 lane 개수 또는 agent_route_lane_order의 열 수 최대값.
        """
        max_agent_num: int = 0
        max_lane_num: int = 0

        for sample in batch:
            aro = sample.get("agent_route_lane_order", None)
            if aro is not None:
                aro_arr = np.asarray(aro)
                # aro_arr: (A_i, L_i)
                if aro_arr.ndim == 2:
                    max_agent_num = max(max_agent_num, int(aro_arr.shape[0]))
                    max_lane_num = max(max_lane_num, int(aro_arr.shape[1]))

            lanes = sample.get("lanes", None)
            if lanes is not None:
                lanes_arr = np.asarray(lanes)
                # lanes_arr: (L_i, lane_len, feat) 또는 최소 (L_i, ...)
                if lanes_arr.ndim >= 1:
                    max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))

        return int(max_agent_num), int(max_lane_num)

    def _get_special_padding_category_spec(
        self,
        key: str,
    ) -> Optional[Tuple[int, int]]:
        """특정 key에 대해, '패딩을 어떤 카테고리로 채울지' 규칙을 돌려줍니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / left_line_type / right_line_type 같은 값은
        보통 (카테고리 개수)만큼의 길이를 가진 벡터(대부분 one-hot)로 들어옵니다.

        그런데 패딩을 전부 0으로 채우면,
        나중에 argmax 같은 방식으로 카테고리를 뽑을 때
        "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.

        그래서 아래 key들만은 패딩을 0이 아니라
        '모름/미정'에 해당하는 카테고리로 채우도록 규칙을 따로 둡니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()

        Returns:
            Optional[Tuple[int, int]]:
                - (unknown_category_index, num_categories)
                - lane_type: (3, 4)  -> (L, 4)에서 마지막(3번)을 1로 채움
                - left/right_line_type: (11, 13) -> (L, 13)에서 11번을 1로 채움
                - 그 외 key는 None
        """
        key_str = str(key)
        if key_str == "lane_type":
            return 3, 4
        if key_str in ("left_line_type", "right_line_type"):
            return 11, 13
        return None

    def _infer_max_lane_num_from_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> int:
        """배치에서 'lane 개수의 최대값'을 안전하게 구합니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / line_type 같은 값이 어떤 샘플에서는 None일 수 있습니다.
        그런데 그 샘플의 lanes 개수는 클 수 있어서,
        단순히 non-None 값들만 보고 max 길이를 정하면
        lanes 텐서와 lane_type 텐서의 lane 차원이 달라질 수 있습니다.

        그래서 lane 관련 key들은 lane 차원을 항상 lanes 기준으로 맞추기 위해
        batch 전체의 lanes.shape[0] 최대값을 구합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            int:
                - max_lane_num: 배치 내 lanes의 첫 번째 축 최대값. shape: ()
        """
        max_lane_num: int = 0
        for sample in batch:
            lanes_value = sample.get("lanes", None)
            if lanes_value is None:
                continue
            lanes_arr = np.asarray(lanes_value)
            if lanes_arr.ndim >= 1:
                max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))
        return int(max_lane_num)

    def _build_default_padded_tensor_for_key(
        self,
        key: str,
        out_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """key 성격에 맞는 '기본 패딩 텐서'를 만듭니다.

        기본 규칙
        --------
        - 대부분의 key: 0으로 채운 텐서를 만듭니다.
        - lane_type: (…, 4) 형태라면 마지막 축에서 3번 인덱스만 1로 채웁니다.
            즉, [0,0,0,1] 형태의 "미정" one-hot 패딩입니다.
        - left_line_type / right_line_type: (…, 13) 형태라면 마지막 축에서 11번만 1로 채웁니다.
            즉, "선은 있으나 타입 모름" one-hot 패딩입니다.

        주의
        ----
        - 만약 어떤 이유로 one-hot 형태가 아니라 (…, ) 라벨 형태로 들어온다면,
          그 경우는 텐서 전체를 unknown 카테고리 숫자(예: 3, 11)로 채웁니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()
            out_shape (Tuple[int, ...]):
                출력 텐서 shape.
                예:
                  - lane_type: (B, L_max, 4)
                  - left_line_type: (B, L_max, 13)
            dtype (torch.dtype):
                출력 텐서 dtype.

        Returns:
            torch.Tensor:
                out: out_shape 그대로의 텐서.
        """
        spec = self._get_special_padding_category_spec(key)
        if spec is None:
            return torch.zeros(out_shape, dtype=dtype)

        unknown_index, num_categories = int(spec[0]), int(spec[1])

        # out_shape가 충분히 길고, 마지막 축이 카테고리 개수와 맞으면 one-hot 패딩
        if len(out_shape) >= 2 and int(out_shape[-1]) == int(num_categories):
            out = torch.zeros(out_shape, dtype=dtype)
            if 0 <= unknown_index < num_categories:
                out[..., unknown_index] = torch.as_tensor(1, dtype=dtype)
            return out

        # one-hot이 아니라 라벨 형태(예: (B, L_max))로 들어오는 경우를 대비한 fallback
        if len(out_shape) >= 2:
            return torch.full(out_shape, fill_value=unknown_index, dtype=dtype)

        # 기대 형태가 너무 이상하면 안전하게 0으로
        return torch.zeros(out_shape, dtype=dtype)

    # ------------------------------------------------------------------
    # 0) 크로핑 여부 판단
    # ------------------------------------------------------------------
    def _should_center_crop(self) -> bool:
        """중심 기준 크로핑을 쓸지 여부를 반환한다.

        Returns:
            bool: True면 크로핑을 적용한다.
        """
        return (self.center_crop_radius_m > 0.0 and
                self.center_crop_mode in ("npc", "ego"))

    # ------------------------------------------------------------------
    # 0-1) 샘플별 개수 통계 계산
    # ------------------------------------------------------------------
    def _compute_object_lengths(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """각 샘플의 agent / lane / route_lane / static 개수를 모은다.

        Args:
            batch: 길이 B 리스트, 각 원소는 단일 샘플 dict.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - agent_len_arr: (B,) 각 샘플의 agent 수
                - lane_len_arr: (B,) 각 샘플의 lane 수
                - route_lane_len_arr: (B,) 각 샘플의 route lane 수
                - static_len_arr: (B,) 각 샘플의 static 수
        """
        agent_len_arr = np.array(
            [sample["neighbor_agents_past"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        lane_len_arr = np.array(
            [sample["lanes"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        route_lane_len_arr = np.array(
            [sample["route_lanes"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        static_len_arr = np.array(
            [sample["static_objects"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        return agent_len_arr, lane_len_arr, route_lane_len_arr, static_len_arr

    # ------------------------------------------------------------------
    # 0-2) 최대 개수 및 lane_len 계산
    # ------------------------------------------------------------------
    def _compute_max_counts_and_lane_len(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
    ) -> Tuple[int, int, int, int, int, int]:
        """배치 안에서 agent / lane / route_lane / static 최대 개수와 길이를 계산한다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 개수.
            lane_len_arr: (B,) lane 개수.
            route_lane_len_arr: (B,) route lane 개수.
            static_len_arr: (B,) static 개수.

        Returns:
            Tuple[int, int, int, int, int, int]:
                - max_agent_num: 배치 내 최대 agent 수
                - max_lane_num: 배치 내 최대 lane 수
                - max_route_lane_num: 배치 내 최대 route lane 수
                - caching_max_static_num: 배치 내 최대 static 수
                - lane_len: lane 한 개당 점 개수
                - route_lane_len: route lane 한 개당 점 개수
        """
        batch_size: int = len(batch)
        max_agent_num: int = int(agent_len_arr.max()) if batch_size > 0 else 0
        max_lane_num: int = int(lane_len_arr.max()) if batch_size > 0 else 0
        max_route_lane_num: int = (int(route_lane_len_arr.max())
                                   if batch_size > 0 else 0)
        max_static_num: int = (int(static_len_arr.max())
                               if batch_size > 0 else 0)

        lane_len: int = (batch[0]["lanes"].shape[1] if max_lane_num > 0 else 0)
        route_lane_len: int = (batch[0]["route_lanes"].shape[1]
                               if max_route_lane_num > 0 else 0)

        return (
            max_agent_num,
            max_lane_num,
            max_route_lane_num,
            max_static_num,
            lane_len,
            route_lane_len,
        )

    # ------------------------------------------------------------------
    # 0-3) ego / 중심 좌표 계산
    # ------------------------------------------------------------------
    def _build_ego_and_center_xy(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ego 현재 위치와 초기 중심 좌표를 만든다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - ego_xy: (B, 2) ego 현재 위치
                - center_xy: (B, 2) 초기 중심(ego와 동일)
        """
        ego_xy = np.stack(
            [sample["ego_agent_past"][-1, 0:2] for sample in batch],
            axis=0,
        ).astype(np.float32)  # (B, 2)
        center_xy = ego_xy.copy()  # (B, 2)
        return ego_xy, center_xy

    def _compute_keep_route_lane_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        route_lanes_xy: np.
        ndarray,  # (B, max_route_lane_num, route_lane_len, 2)
        route_lane_len_arr: np.ndarray,  # (B,)
        max_route_lane_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 route lane 을 남길지 여부를 계산한다.

        lane 과 동일한 규칙이지만, route_lanes 만 따로 본다.

        Args:
            center_xy: (B, 2) 중심 좌표.
            route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2) route lane 좌표.
            route_lane_len_arr: (B,) 각 샘플의 route lane 개수.
            max_route_lane_num: 배치 내 최대 route lane 수.

        Returns:
            np.ndarray:
                keep_route_lane_mask: (B, max_route_lane_num)
        """
        batch_size: int = route_lane_len_arr.shape[0]
        if max_route_lane_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        # route_lane_valid_mask: (B, max_route_lane_num)
        route_lane_valid_mask = (np.arange(max_route_lane_num)[None, :]
                                 < route_lane_len_arr[:, None])

        # diff_route_lane: (B, max_route_lane_num, route_lane_len, 2)
        diff_route_lane = route_lanes_xy - center_xy[:, None, None, :]
        # dist2_route_lane: (B, max_route_lane_num, route_lane_len)
        dist2_route_lane = (diff_route_lane**2).sum(axis=-1)
        # min_dist2_route_lane: (B, max_route_lane_num)
        min_dist2_route_lane = dist2_route_lane.min(axis=-1)

        keep_route_lane_mask = route_lane_valid_mask & (min_dist2_route_lane
                                                        <= radius_sq)
        return keep_route_lane_mask

    # ------------------------------------------------------------------
    # 0-4) 큰 좌표 배열 만들기 (batch 연산용)
    # ------------------------------------------------------------------
    def _build_positions_arrays(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
        max_agent_num: int,
        max_lane_num: int,
        max_route_lane_num: int,
        max_static_num: int,
        lane_len: int,
        route_lane_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """agent / lane / route_lane / static 좌표를 큰 배열로 모은다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 개수.
            lane_len_arr: (B,) lane 개수.
            route_lane_len_arr: (B,) route lane 개수.
            static_len_arr: (B,) static 개수.
            max_agent_num: 배치 내 최대 agent 수.
            max_lane_num: 배치 내 최대 lane 수.
            max_route_lane_num: 배치 내 최대 route lane 수.
            max_static_num: 배치 내 최대 static 수.
            lane_len: lane 한 개당 점 개수.
            route_lane_len: route lane 한 개당 점 개수.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - agents_xy: (B, max_agent_num, 2)
                - lanes_xy: (B, max_lane_num, lane_len, 2)
                - route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2)
                - static_xy: (B, max_static_num, 2)
        """
        batch_size: int = len(batch)

        # agents_xy: (B, max_agent_num, 2)
        agents_xy = np.zeros(
            (batch_size, max_agent_num, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_agent_num = int(agent_len_arr[b_idx])
            if real_agent_num > 0:
                agents_xy[b_idx, :real_agent_num, :] = sample[
                    "neighbor_agents_past"][real_agent_num * 0:real_agent_num,
                                            -1, 0:2]

        # lanes_xy: (B, max_lane_num, lane_len, 2)
        lanes_xy = np.zeros(
            (batch_size, max_lane_num, lane_len, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_lane_num = int(lane_len_arr[b_idx])
            if real_lane_num > 0:
                lanes_xy[b_idx, :real_lane_num, :, :] = sample[
                    "lanes"][:real_lane_num, :, 0:2]

        # route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2)
        route_lanes_xy = np.zeros(
            (batch_size, max_route_lane_num, route_lane_len, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_route_lane_num = int(route_lane_len_arr[b_idx])
            if real_route_lane_num > 0:
                route_lanes_xy[b_idx, :real_route_lane_num, :, :] = sample[
                    "route_lanes"][:real_route_lane_num, :, 0:2]

        # static_xy: (B, max_static_num, 2)
        static_xy = np.zeros(
            (batch_size, max_static_num, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_static_num = int(static_len_arr[b_idx])
            if real_static_num > 0:
                static_xy[b_idx, :real_static_num, :] = sample[
                    "static_objects"][real_static_num * 0:real_static_num, 0:2]

        return agents_xy, lanes_xy, route_lanes_xy, static_xy

    # ------------------------------------------------------------------
    # 0-5) agent 유효 마스크
    # ------------------------------------------------------------------
    def _build_agent_valid_mask(
        self,
        agent_len_arr: np.ndarray,
        max_agent_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """실제 agent 위치가 있는 칸을 표시하는 마스크를 만든다.

        Args:
            agent_len_arr (np.ndarray): (B,) 각 샘플 agent 수.
            max_agent_num (int): 배치 내 최대 agent 수.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - agent_valid_mask: (B, max_agent_num) True=해당 칸에 agent 있음
                - has_any_agent_mask: (B,) True=해당 샘플에 agent 하나 이상
        """
        batch_size: int = agent_len_arr.shape[0]
        if max_agent_num > 0:
            idx = np.arange(max_agent_num)[None, :]  # (1, max_agent_num)
            agent_valid_mask = idx < agent_len_arr[:,
                                                   None]  # (B, max_agent_num)
        else:
            agent_valid_mask = np.zeros((batch_size, 0), dtype=bool)
        has_any_agent_mask = agent_len_arr > 0  # (B,)
        return agent_valid_mask, has_any_agent_mask

    # ------------------------------------------------------------------
    # 0-6) 중심을 NPC 기준으로 바꿀지 결정
    # ------------------------------------------------------------------
    def _maybe_update_center_by_npc(
        self,
        center_xy: np.ndarray,  # (B, 2)
        agents_xy: np.ndarray,  # (B, max_agent_num, 2)
        agent_len_arr: np.ndarray,  # (B,)
        agent_valid_mask: np.ndarray,  # (B, max_agent_num)
        max_agent_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """mode가 'npc'일 때, 랜덤 NPC 한 대를 중심으로 바꾼다.

        Args:
            center_xy (np.ndarray): (B, 2) 기존 중심 좌표.
            agents_xy (np.ndarray): (B, max_agent_num, 2) agent 현재 위치.
            agent_len_arr (np.ndarray): (B,) 각 샘플 agent 수.
            agent_valid_mask (np.ndarray): (B, max_agent_num) True=유효 위치.
            max_agent_num (int): 배치 내 최대 agent 수.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - center_xy_new: (B, 2) 새 중심 좌표.
                - center_agent_idx: (B,) 샘플마다 선택된 중심 agent 인덱스.
        """
        batch_size: int = agent_len_arr.shape[0]
        has_any_agent_mask = agent_len_arr > 0  # (B,)
        center_agent_idx = np.zeros(batch_size, dtype=np.int64)

        if self.center_crop_mode != "npc" or max_agent_num <= 0:
            return center_xy, center_agent_idx

        rand_vals = np.random.rand(batch_size, max_agent_num).astype(
            np.float32)  # (B, max_agent_num)
        rand_vals[~agent_valid_mask] = -1.0

        center_agent_idx = rand_vals.argmax(axis=1)  # (B,)
        center_npc_xy = agents_xy[
            np.arange(batch_size),
            np.clip(center_agent_idx, 0, max_agent_num - 1),
        ]  # (B, 2)

        center_xy_new = center_xy.copy()
        center_xy_new[has_any_agent_mask, :] = center_npc_xy[
            has_any_agent_mask, :]
        return center_xy_new, center_agent_idx

    # ------------------------------------------------------------------
    # 0-7-1) agent keep 마스크 계산
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 0-7-1) agent keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_agent_mask(
            self,
            center_xy: np.ndarray,  # (B, 2)
            agents_xy: np.ndarray,  # (B, max_agent_num, 2)
            agent_len_arr: np.ndarray,  # (B,)
            max_agent_num: int,
            agent_valid_mask: np.ndarray,  # (B, max_agent_num)
    ) -> np.ndarray:
        """중심 기준 거리로 agent 를 남길지 여부를 계산한다.

        동작 요약:
          1) 각 agent 의 현재 위치와 중심 좌표(center_xy) 사이의 거리^2 을 계산한다.
          2) 거리^2 <= (반경 K)^2 인 agent 만 True 로 표시한다.
          3) agent_valid_mask 가 False 인 자리는 무조건 False 로 둔다.

        즉, “중심에서 K m 안에 실제로 존재하는 agent만 남긴다”는 규칙을
        배치 단위로 한 번에 적용하는 함수이다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점(ego 또는 임의 NPC)의 x,y.
            agents_xy (np.ndarray):
                - shape: (B, max_agent_num, 2)
                - 각 배치에서 agent 현재 위치의 x,y.
                - 실제 agent 수보다 큰 부분은 0 으로 채워져 있음.
            agent_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 agent 수.
            max_agent_num (int):
                - 배치 전체에서 가장 많은 agent 개수.
            agent_valid_mask (np.ndarray):
                - shape: (B, max_agent_num)
                - True 이면 해당 자리에는 실제 agent 가 있고,
                  False 이면 패딩(없는 자리).

        Returns:
            np.ndarray:
                - keep_agent_mask
                - shape: (B, max_agent_num)
                - True 인 위치의 agent 만 이후에 남기고,
                  False 인 agent 는 잘라낸다.
        """
        batch_size: int = agent_len_arr.shape[0]
        if max_agent_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        # diff_agent: (B, max_agent_num, 2)
        diff_agent = agents_xy - center_xy[:, None, :]
        # dist2_agent: (B, max_agent_num)
        dist2_agent = (diff_agent**2).sum(axis=-1)

        # 유효한 자리 + 반경 안에 있는 agent 만 남김
        keep_agent_mask = agent_valid_mask & (dist2_agent <= radius_sq)
        return keep_agent_mask

    # ------------------------------------------------------------------
    # 0-7-2) lane keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_lane_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        lanes_xy: np.ndarray,  # (B, max_lane_num, lane_len, 2)
        lane_len_arr: np.ndarray,  # (B,)
        max_lane_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 lane 을 남길지 여부를 계산한다.

        동작 요약:
          1) lane 하나는 여러 점으로 이루어진 선이다.
             · lanes_xy[b, l, p, :] 가 "l번째 lane 의 p번째 점" 좌표.
          2) 각 lane 에 대해, 모든 점과 중심 사이 거리^2 을 구한다.
          3) 그 중 "가장 가까운 점의 거리^2" 이 (반경 K)^2 이하이면
             그 lane 을 keep=True 로 표시한다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점(ego 또는 NPC)의 x,y.
            lanes_xy (np.ndarray):
                - shape: (B, max_lane_num, lane_len, 2)
                - lane 의 각 점 위치 x,y.
                - 실제 lane 수보다 큰 부분은 0 으로 채워져 있음.
            lane_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 lane 수.
            max_lane_num (int):
                - 배치 전체에서 가장 많은 lane 개수.

        Returns:
            np.ndarray:
                - keep_lane_mask
                - shape: (B, max_lane_num)
                - True 인 lane 은 그대로 유지하고,
                  False 인 lane 은 이후 슬라이스에서 제거된다.
        """
        batch_size: int = lane_len_arr.shape[0]
        if max_lane_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        lane_valid_mask = (np.arange(max_lane_num)[None, :]
                           < lane_len_arr[:, None])  # (B, max_lane_num)
        diff_lane = lanes_xy - center_xy[:, None,
                                         None, :]  # (B, max_lane_num, lane_len, 2)
        dist2_lane = (diff_lane**2).sum(axis=-1)  # (B, max_lane_num, lane_len)
        min_dist2_lane = dist2_lane.min(axis=-1)  # (B, max_lane_num)
        keep_lane_mask = lane_valid_mask & (min_dist2_lane <= radius_sq)
        return keep_lane_mask

    # ------------------------------------------------------------------
    # 0-7-3) static keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_static_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        static_xy: np.ndarray,  # (B, max_static_num, 2)
        static_len_arr: np.ndarray,  # (B,)
        max_static_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 static object 를 남길지 여부를 계산한다.

        동작 요약:
          1) 각 static object 의 위치와 중심 좌표 사이의 거리^2 을 계산한다.
          2) 거리^2 <= (반경 K)^2 인 static 만 keep=True 로 표시한다.
          3) 실제 static 이 없는 칸은 False 로 남는다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점의 x,y.
            static_xy (np.ndarray):
                - shape: (B, max_static_num, 2)
                - 각 static 의 위치 x,y.
                - 실제 개수보다 큰 부분은 0 으로 채워져 있음.
            static_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 static 수.
            max_static_num (int):
                - 배치 전체에서 가장 많은 static 개수.

        Returns:
            np.ndarray:
                - keep_static_mask
                - shape: (B, max_static_num)
                - True 인 static 만 이후에 남기고,
                  False 인 static 은 잘라낸다.
        """
        batch_size: int = static_len_arr.shape[0]
        if max_static_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        static_valid_mask = (np.arange(max_static_num)[None, :]
                             < static_len_arr[:, None])  # (B, max_static_num)
        diff_static = static_xy - center_xy[:,
                                            None, :]  # (B, max_static_num, 2)
        dist2_static = (diff_static**2).sum(axis=-1)  # (B, max_static_num)
        keep_static_mask = static_valid_mask & (dist2_static <= radius_sq)
        return keep_static_mask

    # ------------------------------------------------------------------
    # 0-8) keep 마스크를 샘플 dict에 적용 (얇은 for 루프)
    # ------------------------------------------------------------------
    def _apply_keep_masks_to_batch(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
        keep_agent_mask: np.ndarray,
        keep_lane_mask: np.ndarray,
        keep_route_lane_mask: np.ndarray,
        keep_static_mask: np.ndarray,
    ) -> None:
        """계산된 keep 마스크를 이용해 각 샘플 dict를 잘라낸다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 수.
            lane_len_arr: (B,) lane 수.
            route_lane_len_arr: (B,) route lane 수.
            static_len_arr: (B,) static 수.
            keep_agent_mask: (B, max_agent_num) agent keep 마스크.
            keep_lane_mask: (B, max_lane_num) lane keep 마스크.
            keep_route_lane_mask: (B, max_route_lane_num) route lane keep 마스크.
            keep_static_mask: (B, max_static_num) static keep 마스크.
        """
        batch_size: int = len(batch)
        for b_idx in range(batch_size):
            sample = batch[b_idx]

            # 1) agent / future
            if int(agent_len_arr[b_idx]) > 0:
                keep_agent_idx = np.nonzero(keep_agent_mask[b_idx])[0]
                sample["neighbor_agents_past"] = sample["neighbor_agents_past"][
                    keep_agent_idx]
                sample["neighbor_future_gt_3_dim"] = sample[
                    "neighbor_future_gt_3_dim"][keep_agent_idx]

            # 2) lane 관련 (superset)
            if int(lane_len_arr[b_idx]) > 0:
                keep_lane_idx = np.nonzero(keep_lane_mask[b_idx])[0]
                sample["lanes"] = sample["lanes"][keep_lane_idx]
                sample["lanes_speed_limit"] = sample["lanes_speed_limit"][
                    keep_lane_idx]
                sample["lanes_has_speed_limit"] = sample[
                    "lanes_has_speed_limit"][keep_lane_idx]

            # 2-1) route_lanes 관련 (lanes 와 독립 마스크)
            if int(route_lane_len_arr[b_idx]) > 0:
                keep_route_lane_idx = np.nonzero(keep_route_lane_mask[b_idx])[0]

                if "route_lanes" in sample:
                    sample["route_lanes"] = sample["route_lanes"][
                        keep_route_lane_idx]
                if "route_lanes_speed_limit" in sample:
                    sample["route_lanes_speed_limit"] = sample[
                        "route_lanes_speed_limit"][keep_route_lane_idx]
                if "route_lanes_has_speed_limit" in sample:
                    sample["route_lanes_has_speed_limit"] = sample[
                        "route_lanes_has_speed_limit"][keep_route_lane_idx]

            # 3) agent_route_lane_order (행: agent, 열: lane → lanes 기준)
            if int(agent_len_arr[b_idx]) > 0 and int(lane_len_arr[b_idx]) > 0:
                keep_agent_idx = np.nonzero(keep_agent_mask[b_idx])[0]
                keep_lane_idx = np.nonzero(keep_lane_mask[b_idx])[0]
                aro = sample["agent_route_lane_order"]  # (A_i, L_i)
                sample["agent_route_lane_order"] = aro[np.ix_(
                    keep_agent_idx, keep_lane_idx)]

            # 4) static_objects
            if int(static_len_arr[b_idx]) > 0:
                keep_static_idx = np.nonzero(keep_static_mask[b_idx])[0]
                sample["static_objects"] = sample["static_objects"][
                    keep_static_idx]

    # ------------------------------------------------------------------
    # 0) 중심 기준 배치 크로핑 (배치 연산 + 얇은 for 루프)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 1) 고정 shape 텐서들 스택
    # ------------------------------------------------------------------
    def _stack_fixed(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """모든 샘플에서 shape가 같은 키를 (B, ...) 텐서로 쌓는다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.
            key (str): 예: "ego_agent_past".
            dtype (torch.dtype): 출력 dtype.

        Returns:
            torch.Tensor: shape (B, *sample_shape)
        """
        return torch.stack(
            [torch.as_tensor(sample[key], dtype=dtype) for sample in batch],
            dim=0,
        )

    # ------------------------------------------------------------------
    # 2) (N_i, ...) → (B, target_len, ...) 패딩
    # ------------------------------------------------------------------
    def _pad_first_dim_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """샘플마다 길이가 다른 1D 축을 (B, target_len, ...)으로 0 패딩한다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.
            key (str): 예: "neighbor_agents_past".
            target_len (int): 두 번째 축 길이.
            dtype (torch.dtype): 출력 dtype.

        Returns:
            torch.Tensor: shape (B, target_len, *rest_shape)
        """
        batch_size: int = len(batch)
        if batch_size == 0:
            return torch.zeros((0, target_len), dtype=dtype)

        first_arr = batch[0][key]
        rest_shape = first_arr.shape[1:]  # 예: (time_len, 11)
        out_shape = (batch_size, target_len, *rest_shape)
        padded = torch.zeros(out_shape, dtype=dtype)

        for b_idx, sample in enumerate(batch):
            arr = sample[key]
            n_i = min(arr.shape[0], target_len)
            if n_i > 0:
                padded[b_idx, :n_i, ...] = torch.as_tensor(arr[:n_i],
                                                           dtype=dtype)

        return padded

    # ------------------------------------------------------------------
    # 3) (N_i, M_i) → (B, target0, target1) 패딩
    # ------------------------------------------------------------------
    def _pad_two_dims_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len0: int,
        target_len1: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """샘플마다 (행,열) 길이가 다른 배열을 (B, target0, target1)으로 패딩한다.

        현재는 agent_route_lane_order에 사용하며,
        패딩 구간은 -1로 채워서 "연결 없음"을 뜻하게 한다.
        """
        batch_size: int = len(batch)
        out = torch.full(
            (batch_size, target_len0, target_len1),
            fill_value=-1,
            dtype=dtype,
        )

        for b_idx, sample in enumerate(batch):
            arr = sample[key]
            n0 = min(arr.shape[0], target_len0)
            n1 = min(arr.shape[1], target_len1)
            if n0 > 0 and n1 > 0:
                out[b_idx, :n0, :n1] = torch.as_tensor(arr[:n0, :n1],
                                                       dtype=dtype)

        return out

    _FIXED_STACK_KEYS: Tuple[str, ...] = (
        "ego_agent_past",
        "ego_future_gt_3_dim",
        "planner_future_11_dim",
        "ego_agent_past_is_valid",
        "ego_future_gt_is_valid",
    )

    def _is_collatable_value(self, value: Any) -> bool:
        """이 값이 collate 대상(텐서로 묶을 수 있는 값)인지 빠르게 판별합니다.

        이 함수의 목적
        ------------
        collate 단계에서는 "배치 텐서로 묶을 수 있는 값"만 모아야 안전합니다.
        예를 들어 문자열(str), dict 같은 값까지 같이 묶으려 하면 학습 코드에서
        `.to(device)` 같은 처리가 깨질 수 있습니다.

        허용하는 값의 형태
        ----------------
        - None (데이터가 없는 경우. 이 경우는 0 padding으로 처리할 수 있습니다)
        - torch.Tensor
        - numpy.ndarray
        - 숫자 스칼라(int/float/bool, numpy scalar 포함)
        - 숫자 리스트/튜플(예: [1,2,3] 같은 값)  # np.asarray로 변환 가능하다고 가정

        Args:
            value: 샘플 dict 안의 어떤 값.

        Returns:
            bool:
                - True: 텐서로 묶어서 반환 가능한 값
                - False: collate 결과에 넣지 않는 값(예: 문자열 등)
        """
        if value is None:
            return True
        if isinstance(value, torch.Tensor):
            return True
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (bool, int, float, np.number)):
            return True
        if isinstance(value, (list, tuple)):
            # 숫자 리스트/튜플일 가능성이 높으므로 허용
            return True
        return False

    def _get_value_shape(self, value: Any) -> Tuple[int, ...]:
        """입력 값의 shape를 튜플로 얻습니다.

        Args:
            value:
                - torch.Tensor 또는 numpy.ndarray 또는 숫자 리스트/튜플/스칼라.

        Returns:
            Tuple[int, ...]:
                - 예: (A, T, 11), (L, P, 2), (A, L), (T,), ()(스칼라)
        """
        if isinstance(value, torch.Tensor):
            return tuple(int(x) for x in value.shape)
        if isinstance(value, np.ndarray):
            return tuple(int(x) for x in value.shape)

        # 리스트/튜플/스칼라 등: numpy로 shape만 확인
        arr = np.asarray(value)
        return tuple(int(x) for x in arr.shape)

    def _choose_output_dtype(self, value: Any) -> torch.dtype:
        """배치 텐서를 만들 때 사용할 dtype을 결정합니다.

        규칙(속도/안정성 목적)
        -------------------
        - 입력이 float 계열이면: torch.float32 로 통일 (float64 방지)
        - bool은: torch.bool 유지
        - 정수는: torch dtype 그대로 유지(보통 int64)

        Args:
            value:
                - None이 아닌 샘플 값(배치 내 최소 1개는 실제 값이 있어야 dtype을 정할 수 있음)

        Returns:
            torch.dtype:
                - 출력 텐서의 dtype
        """
        t = torch.as_tensor(value)
        if t.is_floating_point():
            return torch.float32
        return t.dtype

    def _collect_batch_keys(self, batch: List[Dict[str, Any]]) -> List[str]:
        """배치 안에 등장한 key를 모아 collate 대상으로 삼을 key 목록을 만듭니다.

        중요한 점
        --------
        - 특정 key 이름을 나열해서 하드코딩하지 않습니다.
        - 배치 dict에 들어온 key를 전부 대상으로 삼습니다.
        - 다만, 텐서로 묶기 어려운 값(예: str)은 제외합니다.
        - 어떤 샘플에서는 None이고 다른 샘플에서는 실제 배열인 key도 있을 수 있어서,
          None도 일단 포함시킨 뒤 "배치 전체가 None인 key"만 최종적으로 제외합니다.

        Args:
            batch:
                - 길이 B인 샘플 dict 리스트.

        Returns:
            List[str]:
                - collate 대상으로 삼을 key 목록(중복 제거, 등장 순서 최대한 유지)
        """
        seen = set()
        keys: List[str] = []

        for sample in batch:
            for k, v in sample.items():
                if k in seen:
                    continue
                if self._is_collatable_value(v):
                    seen.add(k)
                    keys.append(k)

        # 5개 고정 key는 "배치 첫 샘플에 없더라도" 우선순위로 앞에 두고 싶으면,
        # 아래 로직이 도움이 됩니다. (데이터에 실제로 없으면 최종 단계에서 자동 제외됨)
        fixed_front: List[str] = []
        for k in self._FIXED_STACK_KEYS:
            if k in seen and k in keys:
                fixed_front.append(k)

        # fixed_front가 이미 keys 안에 있으니, 순서만 fixed가 앞에 오도록 재정렬
        if fixed_front:
            rest = [k for k in keys if k not in fixed_front]
            return fixed_front + rest

        return keys

    def _stack_fixed_key_for_named_key(
        self,
        key: str,
        values: List[Any],
    ) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다. validity면 bool로 강제합니다.

        동작
        ----
        - ref_shape: 첫 번째 non-None 샘플 shape를 기준으로 (B, *ref_shape) 텐서를 만듭니다.
        - None인 샘플은 0/False로 남습니다.
        - key가 "*_is_valid"면 dtype을 torch.bool로 강제합니다.
          (입력이 int(0/1)이어도 자동으로 False/True로 변환됩니다)

        Args:
            key (str): key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)

        # 고정 key이므로 shape가 다르면 에러
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"key={key}, ref_shape={ref_shape}, got={self._get_value_shape(v)}"
                )

        # out: shape (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _stack_fixed_key(self, values: List[Any]) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다.

        이 함수가 담당하는 경우
        ----------------------
        - ego_agent_past
        - ego_future_gt_3_dim
        - planner_future_11_dim
        - ego_agent_past_is_valid
        - ego_future_gt_is_valid

        위 5개는 "모든 샘플에서 길이가 항상 동일"하다는 전제이므로
        max length를 찾거나 0 padding으로 늘릴 필요가 없습니다.

        단, 값이 None일 수도 있으므로:
        - 배치 안에 실제 값이 하나라도 있으면 그 shape를 기준으로
          None인 샘플은 0으로 채운 텐서를 만들어 stack합니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S) 텐서
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)  # 예: (T_past, 11)

        # non-None 값들의 shape가 동일한지 검사 (고정 key이므로 같아야 정상)
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"ref_shape={ref_shape}, got={self._get_value_shape(v)}")

        # (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        # 값이 있는 샘플만 복사 (None은 0 그대로)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _pad_and_stack_agent_route_lane_order(
        self,
        batch: List[Dict[str, Any]],
    ) -> torch.Tensor:
        """agent_route_lane_order를 (B, A_max, L_max) 텐서로 만들고 -1로 패딩합니다.

        핵심 규칙(중요)
        -------------
        - A_max(최대 agent 수)는 **agent_route_lane_order의 행 개수만** 보고 결정합니다.

        - L_max(최대 lane 수)는
          lanes.shape[0] 와 agent_route_lane_order.shape[1] 중 더 큰 값을 사용합니다.
          (lane 텐서 패딩과 같이 쓰기 쉬운 형태를 유지)

        패딩 값
        -------
        - 없는 값은 전부 -1로 채웁니다. (연결/매칭 없음 의미)

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            torch.Tensor:
                - out: shape (B, A_max, L_max)
                - dtype: torch.int64
                - padding: -1
        """
        batch_size: int = int(len(batch))

        max_agent_num, max_lane_num = self._infer_max_sizes_for_agent_route_lane_order(
            batch)
        out = torch.full(
            (batch_size, max_agent_num, max_lane_num),
            fill_value=-1,
            dtype=torch.int64,
        )

        for b_idx, sample in enumerate(batch):
            aro = sample.get("agent_route_lane_order", None)
            if aro is None:
                continue

            aro_arr = np.asarray(aro)
            # aro_arr: (A_i, L_i)
            if aro_arr.ndim != 2:
                continue

            a_i: int = min(int(aro_arr.shape[0]), max_agent_num)
            l_i: int = min(int(aro_arr.shape[1]), max_lane_num)

            if a_i <= 0 or l_i <= 0:
                continue

            out[b_idx, :a_i, :l_i] = torch.as_tensor(
                aro_arr[:a_i, :l_i],
                dtype=torch.int64,
            )

        return out

    def _pad_and_stack_variable_key_for_named_key(
        self,
        key: str,
        values: List[Any],
        batch: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape 기준으로 padding 후 쌓습니다. validity면 bool로 강제합니다.

        추가로 반영된 규칙
        ----------------
        1) lane_type (shape: (lane_num, 4))
           - 패딩을 0으로 하면 "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 (0,0,0,1)로 둡니다. (index 3이 1인 one-hot)

        2) left_line_type / right_line_type (shape: (lane_num, 13))
           - 패딩을 0으로 하면 특정 선 타입(0번)으로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 index 11이 1인 one-hot로 둡니다.
             (선은 있으나 타입을 모름)

        3) lane 관련 key는 lanes의 최대 lane_num에 맞춰 첫 번째 축 길이를 보정합니다.
           - 어떤 샘플에서 lane_type이 None이라도,
             lanes 개수는 클 수 있어서 lane 차원이 어긋나지 않게 보강합니다.

        Args:
            key (str):
                key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None
            batch (List[Dict[str, Any]]):
                - 원본 샘플 dict 리스트(길이 B).
                - lanes 개수 등을 참고하기 위해 사용합니다.

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)
        ndim: int = int(len(ref_shape))

        max_shape = list(ref_shape)
        all_same_shape: bool = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)

            if len(shape_i) != ndim:
                raise ValueError(
                    f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. key={key}, "
                    f"ref_ndim={ndim}, got_ndim={len(shape_i)}")

            if shape_i != ref_shape:
                all_same_shape = False

            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # ✅ lane_type / line_type 계열은 lanes 최대 lane_num에 맞춰 첫 축을 보정
        special_spec = self._get_special_padding_category_spec(key)
        if special_spec is not None and ndim >= 1:
            max_lane_num_from_lanes: int = self._infer_max_lane_num_from_batch(
                batch)
            if max_lane_num_from_lanes > int(max_shape[0]):
                max_shape[0] = int(max_lane_num_from_lanes)

        # 빠른 경로: 패딩이 전혀 필요 없으면 stack
        # (special_spec가 있어도 max_shape가 ref_shape와 같으면 패딩 영역이 없으므로 stack 가능)
        if all_same_shape and (len(non_none_idx)
                               == batch_size) and (tuple(max_shape)
                                                   == tuple(ref_shape)):
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values],
                dim=0,
            )

        # out: shape (B, *max_shape)
        out_shape: Tuple[int,
                         ...] = (batch_size, *tuple(int(x) for x in max_shape))
        out: torch.Tensor = self._build_default_padded_tensor_for_key(
            key=key,
            out_shape=out_shape,
            dtype=out_dtype,
        )

        # 값 복사 (있는 샘플만 앞쪽부터 채움)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                out[b_idx] = t
                continue

            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *slices)] = t[slices]

        return out

    def _pad_and_stack_variable_key(
            self, values: List[Any]) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape를 기준으로 0 padding 배치 텐서를 만듭니다.

        처리 방식(핵심)
        -------------
        - key 이름을 보고 분기하지 않습니다.
        - 값이 ndarray/tensor 라면 그 shape를 보고,
          배치 내에서 각 차원별 최대값(max)을 구해 큰 텐서를 한 번만 만들고,
          샘플 값은 "앞쪽부터" 그대로 복사합니다.
        - padding 영역은 전부 0 입니다.
          (bool이면 False, float이면 0.0, int면 0)

        이 로직이 자연스럽게 커버하는 예시 shape
        ----------------------------------------
        - neighbor_agents_past:        (A_i, T_past, 11)  -> (B, A_max, T_past, 11)
        - neighbor_future_gt_3_dim:        (A_i, T_fut, 3)    -> (B, A_max, T_fut, 3)
        - lanes:                       (L_i, P_lane, 12) -> (B, L_max, P_max, 12)
        - lanes_len_is_valid:          (L_i, P_lane)      -> (B, L_max, P_max)
        - agent_route_lane_order:      (A_i, L_i)         -> (B, A_max, L_max)
        - road_edge:                   (E_i, P_edge, 2)   -> (B, E_max, P_max, 2)
        - stop_sign_points:            (S_i, P, 2)        -> (B, S_max, P_max, 2)
        - 그리고 그 외 "첫 번째 축이 개수"인 모든 값들

        None 처리
        --------
        - 어떤 샘플에서 None이면 그 샘플은 전부 0으로 남습니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)
        ndim = int(len(ref_shape))

        # max_shape 계산: 배치 내 각 차원별 최대 크기
        max_shape = list(ref_shape)
        all_same_shape = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)
            if len(shape_i) != ndim:
                raise ValueError(f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. "
                                 f"ref_ndim={ndim}, got_ndim={len(shape_i)}")
            if shape_i != tuple(ref_shape):
                all_same_shape = False
            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # (빠른 경로) None도 없고, 모든 shape가 동일하면 stack이 제일 빠름
        if all_same_shape and (len(non_none_idx) == batch_size):
            # shape: (B, *ref_shape)
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values], dim=0)

        # 일반 경로: 0 padding 텐서 만들고 값 복사
        # out shape: (B, *max_shape)
        out = torch.zeros((batch_size, *max_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                # 스칼라: out[b_idx]에 바로 대입
                out[b_idx] = t
                continue

            # 각 차원별로 실제 길이만큼만 복사
            # 예: t.shape == (A_i, T, 11) 이면 slice(0,A_i), slice(0,T), slice(0,11)
            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *([slice(None)] * 0))]  # (형태 힌트용; 실제론 아래 라인만으로 충분)

            out[(b_idx, *slices)] = t[slices]

        return out

    def _build_collated_batch_tensors(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """배치(dict 리스트)를 최종 배치 텐서(dict)로 변환합니다.

        추가 규칙
        --------
        - 어떤 key가 배치 내에서 전부 None이면:
            · 기존: key 자체를 batch_out에서 제거
            · 변경: batch_out[key] = None 으로 유지
        - "*_is_valid" 류 key는 입력이 int(0/1) 이 섞여 있어도 출력 dtype을 torch.bool로 강제합니다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 dict 리스트

        Returns:
            Dict[str, Optional[torch.Tensor]]:
                - 텐서로 만들 수 있으면 torch.Tensor
                - 배치 전체가 None이면 해당 key의 value는 None
        """
        batch_size: int = int(len(batch))
        if batch_size <= 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        keys: List[str] = self._collect_batch_keys(batch)
        batch_out: Dict[str, Optional[torch.Tensor]] = {}

        # 1) 고정 5개 key
        for k in self._FIXED_STACK_KEYS:
            if k not in keys:
                continue
            values = [sample.get(k, None) for sample in batch]
            t = self._stack_fixed_key_for_named_key(k, values)
            # ✅ 전부 None이면 key를 빼지 말고 None으로 유지
            batch_out[k] = t  # t: torch.Tensor 또는 None

        # 2) 나머지 key
        fixed_set = set(self._FIXED_STACK_KEYS)
        for k in keys:
            if k in fixed_set:
                continue

            values = [sample.get(k, None) for sample in batch]

            # ✅ 전부 None이면 key를 빼지 말고 None으로 유지
            if all(v is None for v in values):
                batch_out[k] = None
                continue

            # agent_route_lane_order는 -1 padding 전용 처리
            if k == "agent_route_lane_order":
                batch_out[k] = self._pad_and_stack_agent_route_lane_order(batch)
            else:
                t = self._pad_and_stack_variable_key_for_named_key(k, values, batch)
                batch_out[k] = t  # 정상 케이스에서는 torch.Tensor가 들어옴
            agent_route_lane_order = batch_out.get("agent_route_lane_order", None)
            neighbor_agents_past = batch_out.get("neighbor_agents_past", None)
            agent_route_lane_order_agent_num = agent_route_lane_order.shape[1] if agent_route_lane_order is not None else 0
            neighbor_agents_past_agent_num = neighbor_agents_past.shape[1] if neighbor_agents_past is not None else 0
            if agent_route_lane_order_agent_num != neighbor_agents_past_agent_num:
                raise ValueError(
                    f"[Collate] agent_route_lane_order의 agent 수와 neighbor_agents_past의 agent 수가 일치하지 않습니다. "
                    f"agent_route_lane_order agent num: {agent_route_lane_order_agent_num}, "
                    f"neighbor_agents_past agent num: {neighbor_agents_past_agent_num}"
                )
        return batch_out

    def _is_validity_key_name(self, key: str) -> bool:
        """key 이름이 validity 마스크인지 빠르게 판별합니다.

        규칙
        ----
        - key가 "_is_valid" 로 끝나면 validity 마스크로 봅니다.
          예) ego_agent_past_is_valid, lanes_len_is_valid, lanes_is_valid, ...

        Args:
            key (str): sample dict의 key 이름. shape: ()

        Returns:
            bool:
                - True: validity 마스크 key
                - False: 그 외 key
        """
        return str(key).endswith("_is_valid")

    def _choose_output_dtype_for_key(self, key: str, value: Any) -> torch.dtype:
        """특정 key에 대해, collate 출력 dtype을 결정합니다.

        핵심 규칙
        --------
        - "*_is_valid" 류 key는 입력이 int(0/1) 이 섞여 있어도 **항상 torch.bool**로 강제합니다.
        - 그 외 key는 기존 규칙을 그대로 사용합니다.
          (float -> float32, 나머지 -> 입력 dtype 유지)

        Args:
            key (str): sample dict의 key 이름. shape: ()
            value (Any): None이 아닌 샘플 값(배치에서 dtype 기준으로 삼을 값)

        Returns:
            torch.dtype: 출력 텐서 dtype
        """
        if self._is_validity_key_name(key):
            return torch.bool
        return self._choose_output_dtype(value)

    def _as_bool_1d(
        self,
        mask: Any,
        expected_len: int,
    ) -> Optional[np.ndarray]:
        """입력 마스크를 (expected_len,) bool 배열로 안전 변환합니다.

        이 함수가 필요한 이유
        --------------------
        validity key는 데이터가 없으면 None일 수 있고,
        또 어떤 실수로 shape가 어긋난 값이 들어올 수도 있습니다.
        그럴 때 바로 예외를 내기보다, "안전하게 None 처리"해서
        나머지 crop 로직은 계속 돌도록 만드는 목적입니다.

        Args:
            mask: 원본 마스크 값. None 또는 numpy 배열/리스트 형태.
            expected_len: 기대하는 길이. shape: (expected_len,)

        Returns:
            Optional[np.ndarray]:
                - 정상 변환 시: np.ndarray(dtype=bool), shape (expected_len,)
                - 변환 불가/None이면: None
        """
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 1:
            return None
        if int(arr.shape[0]) != int(expected_len):
            return None
        return arr.astype(bool)

    def _as_bool_2d(
        self,
        mask: Any,
        expected_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """입력 마스크를 expected_shape=(N,P) bool 배열로 안전 변환합니다.

        Args:
            mask: 원본 마스크 값. None 또는 numpy 배열/리스트 형태.
            expected_shape: (N, P)
                - N: 객체 개수
                - P: 한 객체를 이루는 점 개수

        Returns:
            Optional[np.ndarray]:
                - 정상 변환 시: np.ndarray(dtype=bool), shape (N, P)
                - 변환 불가/None이면: None
        """
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 2:
            return None
        if (int(arr.shape[0]) != int(expected_shape[0])) or (int(
                arr.shape[1]) != int(expected_shape[1])):
            return None
        return arr.astype(bool)

    def _slice_first_dim_inplace(
            self,
            sample: Dict[str, Any],
            keys: List[str],
            keep_idx: np.ndarray,  # shape: (K,)
    ) -> None:
        """sample의 여러 key를 첫 번째 축 기준으로 같은 인덱스로 같이 자릅니다.

        Args:
            sample: 단일 샘플 dict.
            keys: 같이 잘라야 하는 key 목록.
            keep_idx: 남길 인덱스. shape: (K,)
        """
        for k in keys:
            if k not in sample:
                continue
            v = sample.get(k, None)
            if v is None:
                continue
            sample[k] = np.asarray(v)[keep_idx]

    def _sort_indices_by_distance(
            self,
            keep_idx: np.ndarray,  # shape: (K,)
            dist2_all: np.ndarray,  # shape: (N,)
    ) -> np.ndarray:
        """keep_idx를 '가까운 순'으로 재정렬합니다.

        Args:
            keep_idx: 남길 인덱스. shape: (K,)
            dist2_all: 전체 객체의 거리^2 배열. shape: (N,)

        Returns:
            sorted_keep_idx: 가까운 순으로 정렬된 keep_idx. shape: (K,)
        """
        if keep_idx.size <= 1:
            return keep_idx
        d = dist2_all[keep_idx]  # (K,)
        order = np.argsort(d, kind="mergesort")
        return keep_idx[order].astype(np.int64)

    def _keep_indices_from_points_xy(
            self,
            points_xy: np.ndarray,  # shape: (N, 2)
            center_xy: np.ndarray,  # shape: (2,)
            radius_sq: float,
            invalid_fill: np.float32,
            object_valid_1d: Optional[np.ndarray],  # shape: (N,)
    ) -> np.ndarray:
        """(N,2) 점 기준으로 반경 안에 들어오는 객체 인덱스를 만듭니다.

        규칙
        ----
        - 각 점의 (x,y)와 center_xy 사이 거리^2 <= radius_sq 이면 keep
        - object_valid_1d가 있으면 False인 객체는 무조건 제외
        - 반환 인덱스는 "가까운 순"으로 정렬해서 돌려줍니다.

        Args:
            points_xy: shape (N,2)
            center_xy: shape (2,)
            radius_sq: 반경^2 (스칼라)
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            object_valid_1d: shape (N,) 또는 None

        Returns:
            keep_idx: shape (K,), dtype int64
        """
        n = int(points_xy.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)

        pts = np.asarray(points_xy, dtype=np.float32)  # (N,2)
        diff = pts - center_xy[None, :]  # (N,2)
        dist2 = (diff * diff).sum(axis=-1)  # (N,)

        if object_valid_1d is not None:
            dist2 = np.where(object_valid_1d, dist2, invalid_fill)

        keep_mask = dist2 <= np.float32(radius_sq)
        keep_idx = np.nonzero(keep_mask)[0].astype(np.int64)
        return self._sort_indices_by_distance(keep_idx, dist2)

    def _keep_indices_from_polyline_xy(
            self,
            poly_xy: np.ndarray,  # shape: (N, P, 2)
            center_xy: np.ndarray,  # shape: (2,)
            radius_sq: float,
            invalid_fill: np.float32,
            point_valid_2d: Optional[np.ndarray],  # shape: (N, P)
            object_valid_1d: Optional[np.ndarray],  # shape: (N,)
    ) -> np.ndarray:
        """(N,P,2) 선/영역 형태에서 반경 안에 들어오는 객체 인덱스를 만듭니다.

        규칙
        ----
        - 한 객체는 P개의 점으로 이루어져 있다고 보고,
          그 중 center_xy에 가장 가까운 점의 거리^2이 radius_sq 이하이면 keep
        - point_valid_2d가 있으면 False인 점은 거리 계산에서 제외(매우 멀리 취급)
        - point_valid_2d가 없으면, (x,y) 둘 다 0인 점을 "없는 점"으로 보고 제외
        - object_valid_1d가 있으면 False인 객체는 무조건 제외
        - 반환 인덱스는 "가까운 순"으로 정렬해서 돌려줍니다.

        Args:
            poly_xy: shape (N,P,2)
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            point_valid_2d: shape (N,P) 또는 None
            object_valid_1d: shape (N,) 또는 None

        Returns:
            keep_idx: shape (K,), dtype int64
        """
        n = int(poly_xy.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        p = int(poly_xy.shape[1])
        if p <= 0:
            return np.zeros((0,), dtype=np.int64)

        pts = np.asarray(poly_xy, dtype=np.float32)  # (N,P,2)
        diff = pts - center_xy[None, None, :]  # (N,P,2)
        dist2_pts = (diff * diff).sum(axis=-1)  # (N,P)

        if point_valid_2d is None:
            fallback_valid = np.any(np.abs(pts) > 0.0, axis=-1)  # (N,P)
            dist2_pts = np.where(fallback_valid, dist2_pts, invalid_fill)
        else:
            dist2_pts = np.where(point_valid_2d, dist2_pts, invalid_fill)

        min_dist2 = dist2_pts.min(axis=-1)  # (N,)

        if object_valid_1d is not None:
            min_dist2 = np.where(object_valid_1d, min_dist2, invalid_fill)

        keep_mask = min_dist2 <= np.float32(radius_sq)
        keep_idx = np.nonzero(keep_mask)[0].astype(np.int64)
        return self._sort_indices_by_distance(keep_idx, min_dist2)

    def _compute_center_xy_for_sample(
        self,
        sample: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """샘플에서 중심점(center_xy)을 계산합니다.

        규칙
        ----
        - 기본은 ego 기준:
            center_xy = ego_agent_past[-1, 0:2]  # shape (2,)
        - center_crop_mode == "npc" 이면:
            neighbor_agents_is_valid==True인 agent들 중 하나를 랜덤 선택해
            center_xy = neighbor_agents_past[selected, -1, 0:2]
          유효 agent가 없으면 ego 기준으로 fallback 합니다.

        Args:
            sample: 단일 샘플 dict.

        Returns:
            Optional[np.ndarray]:
                - 성공 시: center_xy, shape (2,), dtype float32
                - ego 좌표를 뽑을 수 없으면: None
        """
        ego_past = sample.get("ego_agent_past", None)
        if ego_past is None:
            return None

        ego_past_arr = np.asarray(ego_past)
        if ego_past_arr.ndim < 2 or int(ego_past_arr.shape[0]) <= 0:
            return None

        ego_xy = np.asarray(ego_past_arr[-1, 0:2], dtype=np.float32)  # (2,)
        center_xy = ego_xy

        if self.center_crop_mode != "npc":
            return center_xy

        neighbor_past = sample.get("neighbor_agents_past", None)
        if neighbor_past is None:
            return center_xy

        neighbor_past_arr = np.asarray(neighbor_past)
        if neighbor_past_arr.ndim < 3 or int(neighbor_past_arr.shape[0]) <= 0:
            return center_xy

        a = int(neighbor_past_arr.shape[0])
        neighbor_valid_1d = self._as_bool_1d(sample.get(
            "neighbor_agents_is_valid", None),
                                             expected_len=a)
        if neighbor_valid_1d is None:
            candidate_idx = np.arange(a, dtype=np.int64)
        else:
            candidate_idx = np.nonzero(neighbor_valid_1d)[0].astype(np.int64)

        if int(candidate_idx.shape[0]) <= 0:
            return center_xy

        chosen = int(candidate_idx[np.random.randint(int(
            candidate_idx.shape[0]))])
        center_xy = np.asarray(neighbor_past_arr[chosen, -1, 0:2],
                               dtype=np.float32)
        return center_xy

    def _crop_poly_object_group_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape (2,)
        radius_sq: float,
        invalid_fill: np.float32,
        points_key: str,
        is_valid_key: str,
    ) -> None:
        """(N,P,2) + (N,) 형태인 지도/영역 요소를 반경 기준으로 crop 합니다.

        예:
          - stop_sign_points / stop_sign_is_valid
          - crosswalk_points / crosswalk_is_valid
          - speed_bump_points / speed_bump_is_valid
          - driveway_points / driveway_is_valid

        Args:
            sample: 단일 샘플 dict.
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            points_key: 예) "crosswalk_points"
            is_valid_key: 예) "crosswalk_is_valid"
        """
        pts = sample.get(points_key, None)
        if pts is None:
            return
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 3:
            return
        n = int(pts_arr.shape[0])
        if n <= 0:
            return

        obj_valid_1d = self._as_bool_1d(sample.get(is_valid_key, None),
                                        expected_len=n)
        keep_idx = self._keep_indices_from_polyline_xy(
            poly_xy=np.asarray(pts_arr, dtype=np.float32),
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=None,
            object_valid_1d=obj_valid_1d,
        )
        self._slice_first_dim_inplace(sample,
                                      keys=[points_key, is_valid_key],
                                      keep_idx=keep_idx)

    def _crop_road_edge_group_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """road_edge 관련 key들을 반경 기준으로 함께 crop 합니다.

        함께 자르는 key
        --------------
        - road_edge: shape (E, safety_len, 2)
        - road_edge_type: shape (E, 3)
        - road_edge_is_valid: shape (E,) (존재하면 같이)

        Args:
            sample: 단일 샘플 dict.
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
        """
        road_edge = sample.get("road_edge", None)
        if road_edge is None:
            return

        road_edge_arr = np.asarray(road_edge)
        if road_edge_arr.ndim != 3:
            return
        e = int(road_edge_arr.shape[0])
        if e <= 0:
            return

        road_edge_valid_1d = self._as_bool_1d(sample.get(
            "road_edge_is_valid", None),
                                              expected_len=e)

        keep_idx = self._keep_indices_from_polyline_xy(
            poly_xy=np.asarray(road_edge_arr, dtype=np.float32),
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=None,
            object_valid_1d=road_edge_valid_1d,
        )
        self._slice_first_dim_inplace(
            sample,
            keys=["road_edge", "road_edge_type", "road_edge_is_valid"],
            keep_idx=keep_idx,
        )

    def _crop_agent_route_lane_order_inplace(
        self,
        sample: Dict[str, Any],
        keep_agent_idx: Optional[np.ndarray],
        keep_lane_idx: Optional[np.ndarray],
    ) -> None:
        """agent_route_lane_order를 (행=agent, 열=lane) 기준으로 동시에 crop 합니다.

        Args:
            sample: 단일 샘플 dict.
            keep_agent_idx: shape (K_a,) 또는 None (None이면 agent는 유지)
            keep_lane_idx: shape (K_l,) 또는 None (None이면 lane은 유지)
        """
        aro = sample.get("agent_route_lane_order", None)
        if aro is None:
            return

        aro_arr = np.asarray(aro)
        if aro_arr.ndim != 2:
            return

        if keep_agent_idx is None:
            keep_agent_idx_eff = np.arange(int(aro_arr.shape[0]),
                                           dtype=np.int64)
        else:
            keep_agent_idx_eff = keep_agent_idx.astype(np.int64)

        if keep_lane_idx is None:
            keep_lane_idx_eff = np.arange(int(aro_arr.shape[1]), dtype=np.int64)
        else:
            keep_lane_idx_eff = keep_lane_idx.astype(np.int64)

        sample["agent_route_lane_order"] = aro_arr[np.ix_(
            keep_agent_idx_eff, keep_lane_idx_eff)]

        # agent_route_lane_order_is_valid 는 (agent 축)만 맞추면 됨
        aro_valid = sample.get("agent_route_lane_order_is_valid", None)
        if aro_valid is not None:
            aro_valid_arr = np.asarray(aro_valid)
            if aro_valid_arr.ndim == 1 and int(aro_valid_arr.shape[0]) == int(
                    keep_agent_idx_eff.shape[0]):
                # 이미 앞에서 agent 기준으로 잘려져 들어온 경우라면 그대로 둔다.
                return
            if aro_valid_arr.ndim == 1 and int(aro_valid_arr.shape[0]) >= int(
                    keep_agent_idx_eff.max(initial=-1) + 1):
                sample["agent_route_lane_order_is_valid"] = aro_valid_arr[
                    keep_agent_idx_eff]

    def _build_center_crop_constants(self) -> Tuple[float, np.float32]:
        """center crop에 필요한 상수들을 한 번에 만든다.

        Args:
            없음

        Returns:
            Tuple[float, np.float32]:
                - radius_sq: float, 반경의 제곱(= (center_crop_radius_m)^2). shape: ()
                - invalid_fill: np.float32, invalid를 멀리 보내기 위한 매우 큰 값. shape: ()
        """
        radius_sq: float = float(self.center_crop_radius_m)**2
        invalid_fill: np.float32 = np.float32(1.0e12)
        return radius_sq, invalid_fill

    def _crop_agents_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> Optional[np.ndarray]:
        """Agent 기준으로 crop 하고, agent 관련 의존 key들도 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: neighbor_agents_past (현재 위치: neighbor_agents_past[:, -1, 0:2])
        - 같이 자름:
            neighbor_agents_past
            neighbor_future_gt_3_dim
            neighbor_agents_past_is_valid
            neighbor_agents_is_valid
            neighbor_future_gt_is_valid
            agent_route_lane_order_is_valid  (agent 축과 길이 맞추기)

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()

        Returns:
            Optional[np.ndarray]:
                - keep_agent_idx: np.ndarray(int64), shape (K_a,)
                - agent 정보를 못 자르면 None
        """
        neighbor_past = sample.get("neighbor_agents_past", None)
        if neighbor_past is None:
            return None

        neighbor_past_arr = np.asarray(neighbor_past)
        if neighbor_past_arr.ndim < 3:
            return None

        a = int(neighbor_past_arr.shape[0])
        if a <= 0:
            return np.zeros((0,), dtype=np.int64)

        agents_xy = np.asarray(neighbor_past_arr[:, -1, 0:2],
                               dtype=np.float32)  # (A,2)
        neighbor_valid_1d = self._as_bool_1d(
            sample.get("neighbor_agents_is_valid", None),
            expected_len=a,
        )

        keep_agent_idx = self._keep_indices_from_points_xy(
            points_xy=agents_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            object_valid_1d=neighbor_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "neighbor_agents_past",
                "neighbor_future_gt_3_dim",
                "neighbor_agents_past_is_valid",
                "neighbor_agents_is_valid",
                "neighbor_future_gt_is_valid",
                "agent_route_lane_order_is_valid",
            ],
            keep_idx=keep_agent_idx,
        )
        return keep_agent_idx

    def _crop_lanes_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> Optional[np.ndarray]:
        """Lane 기준으로 crop 하고, lane 관련 의존 key들도 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: lanes (좌표: lanes[:, :, 0:2])
        - 같이 자름:
            lanes
            lanes_speed_limit
            lanes_has_speed_limit
            lane_type
            left_line_type
            right_line_type
            lanes_len_is_valid
            lanes_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()

        Returns:
            Optional[np.ndarray]:
                - keep_lane_idx: np.ndarray(int64), shape (K_l,)
                - lanes를 못 자르면 None
        """
        lanes = sample.get("lanes", None)
        if lanes is None:
            return None

        lanes_arr = np.asarray(lanes)
        if lanes_arr.ndim < 3:
            return None

        l = int(lanes_arr.shape[0])
        if l <= 0:
            return np.zeros((0,), dtype=np.int64)

        lane_len = int(lanes_arr.shape[1])
        lanes_xy = np.asarray(lanes_arr[:, :, 0:2],
                              dtype=np.float32)  # (L, lane_len, 2)

        lanes_len_is_valid_2d = self._as_bool_2d(
            sample.get("lanes_len_is_valid", None),
            expected_shape=(l, lane_len),
        )
        lanes_is_valid_1d = self._as_bool_1d(
            sample.get("lanes_is_valid", None),
            expected_len=l,
        )

        keep_lane_idx = self._keep_indices_from_polyline_xy(
            poly_xy=lanes_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=lanes_len_is_valid_2d,
            object_valid_1d=lanes_is_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "lanes",
                "lanes_speed_limit",
                "lanes_has_speed_limit",
                "lane_type",
                "left_line_type",
                "right_line_type",
                "lanes_len_is_valid",
                "lanes_is_valid",
            ],
            keep_idx=keep_lane_idx,
        )
        return keep_lane_idx

    def _crop_route_lanes_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Route lane 기준으로 crop 하고, route lane 관련 의존 key들을 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: route_lanes (좌표: route_lanes[:, :, 0:2])
        - 같이 자름:
            route_lanes
            route_lanes_speed_limit
            route_lanes_has_speed_limit
            route_lanes_len_is_valid
            route_lanes_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        route_lanes = sample.get("route_lanes", None)
        if route_lanes is None:
            return

        route_arr = np.asarray(route_lanes)
        if route_arr.ndim < 3:
            return

        r = int(route_arr.shape[0])
        if r <= 0:
            return

        route_len = int(route_arr.shape[1])
        route_xy = np.asarray(route_arr[:, :, 0:2],
                              dtype=np.float32)  # (R, route_len, 2)

        route_len_is_valid_2d = self._as_bool_2d(
            sample.get("route_lanes_len_is_valid", None),
            expected_shape=(r, route_len),
        )
        route_is_valid_1d = self._as_bool_1d(
            sample.get("route_lanes_is_valid", None),
            expected_len=r,
        )

        keep_route_idx = self._keep_indices_from_polyline_xy(
            poly_xy=route_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=route_len_is_valid_2d,
            object_valid_1d=route_is_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "route_lanes",
                "route_lanes_speed_limit",
                "route_lanes_has_speed_limit",
                "route_lanes_len_is_valid",
                "route_lanes_is_valid",
            ],
            keep_idx=keep_route_idx,
        )

    def _crop_static_objects_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Static object 기준으로 crop 하고, static 관련 의존 key들을 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: static_objects (좌표: static_objects[:, 0:2])
        - 같이 자름:
            static_objects
            static_objects_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        static_objects = sample.get("static_objects", None)
        if static_objects is None:
            return

        static_arr = np.asarray(static_objects)
        if static_arr.ndim < 2:
            return

        s = int(static_arr.shape[0])
        if s <= 0:
            return

        static_xy = np.asarray(static_arr[:, 0:2], dtype=np.float32)  # (S,2)
        static_valid_1d = self._as_bool_1d(
            sample.get("static_objects_is_valid", None),
            expected_len=s,
        )

        keep_static_idx = self._keep_indices_from_points_xy(
            points_xy=static_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            object_valid_1d=static_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=["static_objects", "static_objects_is_valid"],
            keep_idx=keep_static_idx,
        )

    def _crop_safety_and_map_objects_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Stop/Crosswalk/SpeedBump/Driveway/RoadEdge 그룹을 한 번에 crop 한다.

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        # (N,P,2) + (N,) 쌍들
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="stop_sign_points",
            is_valid_key="stop_sign_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="crosswalk_points",
            is_valid_key="crosswalk_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="speed_bump_points",
            is_valid_key="speed_bump_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="driveway_points",
            is_valid_key="driveway_is_valid",
        )

        # road_edge (+ type + (있으면) road_edge_is_valid)
        self._crop_road_edge_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

    def _center_crop_one_sample_inplace(
        self,
        sample: Dict[str, Any],
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """샘플 1개를 center 기준 거리로 crop 하는 전체 절차를 수행한다.

        Args:
            sample: 단일 샘플 dict
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        center_xy = self._compute_center_xy_for_sample(sample)
        if center_xy is None:
            return

        keep_agent_idx = self._crop_agents_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        keep_lane_idx = self._crop_lanes_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

        self._crop_route_lanes_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        self._crop_static_objects_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        self._crop_safety_and_map_objects_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

        # (행=agent, 열=lane) 동시 crop
        self._crop_agent_route_lane_order_inplace(
            sample=sample,
            keep_agent_idx=keep_agent_idx,
            keep_lane_idx=keep_lane_idx,
        )

    def _center_crop_batch_batched(
        self,
        batch: List[Dict[str, Any]],
    ) -> None:
        """배치 샘플들을 중심점(center_xy) 기준 거리로 crop 합니다.

        - batch는 샘플마다 (agent/lane/route/static 개수)가 달라 완전 벡터화가 이득이 크지 않아,
          샘플 단위 루프는 유지하되, 각 샘플 안에서 계산은 numpy 연산으로 처리합니다.
        """
        if not batch:
            return

        radius_sq, invalid_fill = self._build_center_crop_constants()

        for sample in batch:
            self._center_crop_one_sample_inplace(
                sample=sample,
                radius_sq=radius_sq,
                invalid_fill=invalid_fill,
            )

    # ------------------------------------------------------------------
    # 4) collate 본체
    # ------------------------------------------------------------------
    def __call__(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """단일 샘플 dict 리스트를 (B, ·) 텐서 dict로 변환한다.

        Returns:
            Dict[str, Optional[torch.Tensor]]:
                - 텐서로 만들 수 있는 key는 torch.Tensor
                - 배치 전체가 None인 key는 None
        """
        if len(batch) == 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        if self._should_center_crop():
            self._center_crop_batch_batched(batch)
        add_near_agents_info_inplace(
            batch,
            predicted_neighbor_num=self.args.predicted_neighbor_num,
        )
        return self._build_collated_batch_tensors(batch)


def _init_distributed(args: argparse.Namespace,) -> Tuple[int, int, int, bool]:
    """여러 GPU를 쓸 때 필요한 분산 학습 환경을 준비하고, 내 위치 정보를 정리한다.

    DeepSpeed를 쓰는 경우에도 torch.distributed를 미리 초기화해 두어서
    _get_save_path() 안의 broadcast가 항상 동작하도록 만든다.
    """
    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    if use_deepspeed:
        # torchrun이 설정한 환경변수만 사용해서 rank/world_size를 계산한다.
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # 단일 프로세스 fallback
            global_rank = 0
            world_size = 1
            rank = 0

        if args.device.startswith("cuda"):
            torch.cuda.set_device(rank)

        # ✅ DeepSpeed에서도 torch.distributed를 미리 초기화
        #    → _get_save_path의 broadcast_object_list가 항상 동작
        if (world_size > 1 and torch.distributed.is_available() and
                not torch.distributed.is_initialized()):
            backend = "nccl" if args.device.startswith("cuda") else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=global_rank,
            )
    else:
        # 기존 DDP 초기화 흐름과 동일
        global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
        world_size = ddp.get_world_size()

    return global_rank, rank, world_size, use_deepspeed


def _scale_learning_rate_and_epochs(
    args: argparse.Namespace,
    world_size: int,
) -> Tuple[int, int]:
    """글로벌 배치 크기에 맞춰, 학습률과 학습 epoch 수를 자동 조정한다.

    한 번의 업데이트에서 사용하는 샘플 수(글로벌 배치 크기)가 커지면,
    · 한 번 업데이트가 더 안정적으로 되지만
    · 전체 업데이트 횟수(스텝 수)는 줄어든다.
    이 함수는 이런 변화에 맞게 두 가지를 같이 조정한다.

      1) 학습률(lr)을 글로벌 배치 비율에 따라 키운다.
         배치가 커질수록 한 번에 더 많은 정보를 보니,
         한 스텝에서 조금 더 크게 움직여도 안정적이라는 가정이다.
      2) 전체 epoch 수를 줄이거나 늘려서,
         “전체 업데이트 스텝 수”가 기준 세팅과 비슷하도록 맞춘다.
         이렇게 하면 배치 크기를 바꿔도 학습 양이 너무 많아지거나
         너무 적어지지 않도록 균형을 맞출 수 있다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        world_size: 전체 프로세스 개수. (GPU 개수 기준)

    Returns:
        BASE_GLOBAL_BATCH: 기준이 되는 글로벌 배치 크기(예: 2048).
        current_global_batch: 현재 설정에서 실제로 사용되는 글로벌 배치 크기.
    """
    ################ effective warm up epoch ################
    warm_up_epoch: int = min(int(args.warm_up_epoch), args.train_epochs)
    BASE_GLOBAL_BATCH = 2048  # 기준 글로벌 배치
    base_lr: float = float(args.learning_rate)
    base_min_lr = getattr(args, "min_learning_rate", None)

    # DataLoader가 실제로 사용할 글로벌 배치(정수 배수)로 계산
    current_global_batch = _effective_global_batch(args.batch_size, world_size)

    # 배치가 커질수록 lr를 키우되, 작아지는 경우는 그대로 둔다.
    scale_factor = math.sqrt(current_global_batch / float(BASE_GLOBAL_BATCH))
    scale_factor = max(1.0, scale_factor)

    args.learning_rate = base_lr * scale_factor
    if base_min_lr is not None:
        args.min_learning_rate = float(base_min_lr) * scale_factor

    # 전체 학습 epoch 수를 글로벌 배치에 맞게 조정
    EPOCH_BETA = 0.4
    base_epochs_anchor: int = int(args.train_epochs)

    scaled_epochs = _auto_scale_train_epochs(
        base_global_batch=BASE_GLOBAL_BATCH,
        base_epochs=base_epochs_anchor,
        current_global_batch=current_global_batch,
        beta=EPOCH_BETA,
        clamp_min=1,
        clamp_max=None,
    )
    scale_epoch_factor = scaled_epochs / float(args.train_epochs)
    args.train_epochs = int(scaled_epochs)
    args.warm_up_epoch = int(warm_up_epoch * scale_epoch_factor)

    return BASE_GLOBAL_BATCH, current_global_batch


def _dump_args(
    args: argparse.Namespace,
    global_rank: int,
) -> None:
    """체크포인트/로그 디렉터리에 args.json을 기록한다.

    주의:
    - args는 학습 중에 lr/epoch 등이 스케일링되거나,
      resume 과정에서 train_epochs가 보정될 수 있다.
    - 따라서 '최종 args'가 결정된 뒤에 호출하는 것이
      실제 학습 재현/디버깅에 더 정확하다.

    Args:
        args: 학습 설정 Namespace.
        global_rank: 전역 rank. 0에서만 파일을 쓴다.

    Returns:
        None
    """
    if global_rank != 0:
        return

    save_path = getattr(args, "save_path", None)
    if not save_path:
        return

    os.makedirs(save_path, exist_ok=True)

    args_dict = vars(args)
    args_dict = {
        k: (v if not isinstance(v, (StateNormalizer, ObservationNormalizer))
            else v.to_dict()) for k, v in args_dict.items()
    }

    from mmengine.fileio import dump
    dump(
        args_dict,
        os.path.join(save_path, "args.json"),
        file_format="json",
        indent=4,
    )


def _build_augmentation(args: argparse.Namespace,) -> Optional[object]:
    """ego / npc 궤적에 사용할 데이터 증가(augmentation) 객체를 만든다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.

    Returns:
        aug: StatePerturbation 또는 NPCStatePerturbation, 사용 안 하면 None.
    """
    if args.use_ego_data_augment and args.use_npc_data_augment:
        raise ValueError(
            "You cannot use both ego and npc data augmentation at the same time. "
        )

    if args.use_ego_data_augment:
        aug = StatePerturbation(
            augment_prob=args.augment_prob,
            device=args.device,
        )
    elif args.use_npc_data_augment:
        aug = NPCStatePerturbation(
            augment_prob=args.augment_prob,
            device=args.device,
        )
    else:
        aug = None

    return aug


def _build_dataset_and_sampler(
    args: argparse.Namespace,
    world_size: int,
    global_rank: int,
) -> Tuple[DiffusionPlannerData, DistributedSampler]:
    """학습용 Dataset과 DistributedSampler를 만든다.

    DiffusionPlannerData 는
      - 학습에 쓸 파일 목록(JSON)을 읽고
      - 각 항목에 대응하는 npz 파일을 열어서
      - 하나의 샘플을 이름 기반 dict로 돌려준다.
        예: "ego_agent_past": (time_len, 11),
            "neighbor_agents_past": (agent_num, time_len, 11),
            "lanes": (lane_num, lane_len, 12) 등.

    DistributedSampler 는 전체 샘플 인덱스를
      - world_size 개만큼 나누고
      - 각 rank(global_rank)에 서로 다른 구간을 배정해서
        여러 GPU가 겹치지 않는 데이터를 보게 만든다.
    이렇게 하면 모든 GPU가 동시에 다른 샘플을 보면서도,
    한 epoch 안에서 전체 데이터가 고르게 사용되도록 맞출 수 있다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        world_size: 전체 프로세스 개수.
        global_rank: 전체 프로세스 기준 번호.

    Returns:
        train_set: DiffusionPlannerData 인스턴스.
        train_sampler: 각 rank에 샘플을 나눠주는 DistributedSampler.
    """
    # train_set 내부에서 개별 샘플은 npz를 읽어 (agent, lane 등) 배열을
    #   모델 입력 shape에 맞는 Tensor로 바꿔준다. (예: (time_len, 11), (A, T, 11) 등)
    train_set = DiffusionPlannerData(
        args.train_set,  # 예: "/mnt/nuplan/dataset/processed"
        args.train_set_list,  # 예: diffusion_planner_training.json
    )

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
    )

    return train_set, train_sampler


def _compute_warmup_steps(
    args: argparse.Namespace,
    BASE_GLOBAL_BATCH: int,
    current_global_batch: int,
    total_data_num: int,
) -> Tuple[int, int]:
    """워밍업(epoch 수와 배치 크기)에 맞춰 워밍업 step 수를 계산한다.

    여기서는 이미 `_scale_learning_rate_and_epochs`에서
    args.warm_up_epoch 가 현재 설정에 맞게 스케일된 상태라고 가정한다.
    따라서, warm_up_epoch "만큼의 epoch" 동안 워밍업이 유지되도록
    step 수를 직접 계산한다.
    """
    use_warmup: bool = bool(getattr(args, "use_lr_warmup", True))
    warm_up_epoch: int = int(getattr(args, "warm_up_epoch", 0))

    if (not use_warmup) or warm_up_epoch <= 0:
        return 0, 0

    # 기준 배치 B0에서의 epoch당 step 수 (로그용)
    steps_per_epoch_at_B0 = math.ceil(total_data_num / float(BASE_GLOBAL_BATCH))
    warmup_steps_at_B0 = steps_per_epoch_at_B0 * warm_up_epoch

    # 현재 글로벌 배치에서의 epoch당 step 수
    steps_per_epoch_current = math.ceil(total_data_num /
                                        float(current_global_batch))

    # warm_up_epoch 만큼의 epoch 동안 워밍업이 지속되도록 step 수를 정의
    warmup_steps = steps_per_epoch_current * warm_up_epoch

    return warmup_steps_at_B0, warmup_steps


def _build_train_loader(
    args: argparse.Namespace,
    train_set: DiffusionPlannerData,
    train_sampler: DistributedSampler,
    batch_size: int,
    world_size: int,
) -> DataLoader:
    """분산 학습에 맞는 DataLoader를 만든다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        train_set: 학습용 DiffusionPlannerData.
        train_sampler: rank별로 인덱스를 나누는 DistributedSampler.
        batch_size: 전체 글로벌 배치 크기.
        world_size: 전체 프로세스 개수.

    Returns:
        train_loader: 각 step마다 배치 dict를 반환하는 DataLoader.
                      각 배치의 텐서는 (B, ·) shape를 가진다.
    """
    # 각 rank가 보는 per-rank 배치 크기
    batch_size_per_rank = batch_size // world_size

    train_loader = DataLoader(
        train_set,  # DiffusionPlannerData
        sampler=train_sampler,  # DistributedSampler
        batch_size=batch_size_per_rank,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_mem,
        persistent_workers=True,  # 에폭이 바뀌어도 워커 유지
        drop_last=True,
        collate_fn=DiffusionPlannerCollate(args),  # 배치 텐서 shape 맞춤
    )
    return train_loader


def _compute_schedule_info(
    args: argparse.Namespace,
    train_loader: DataLoader,
    world_size: int,
) -> Tuple[int, int, int, int]:
    """step 수와 글로벌 배치 크기, 에폭당 샘플 수를 계산한다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        train_loader: 학습용 DataLoader.
        world_size: 전체 프로세스 개수.

    Returns:
        total_step_of_this_epoch: 한 epoch에서의 step 수.
        total_step_of_all_epoch: 전체 학습 동안의 총 step 수.
        global_batch_size: 실제 글로벌 배치 크기.
        data_num_in_a_epoch: 한 epoch 동안 처리되는 샘플 개수.
    """
    total_step_of_this_epoch = len(train_loader)
    total_step_of_all_epoch = args.train_epochs * total_step_of_this_epoch

    bs_per_rank = args.batch_size // world_size
    global_batch_size = bs_per_rank * world_size
    data_num_in_a_epoch = total_step_of_this_epoch * global_batch_size

    return (
        total_step_of_this_epoch,
        total_step_of_all_epoch,
        global_batch_size,
        data_num_in_a_epoch,
    )


def _print_schedule_summary(
    global_rank: int,
    train_set_len: int,
    BASE_GLOBAL_BATCH: int,
    current_global_batch: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    total_step_of_all_epoch: int,
    warmup_steps_at_B0: int,
    warmup_steps: int,
) -> None:
    """rank 0에서 데이터/스케줄 정보를 한 번 출력한다.

    Args:
        global_rank: 전체 프로세스 기준 번호.
        train_set_len: 학습 샘플 개수.
        BASE_GLOBAL_BATCH: 기준 글로벌 배치 크기.
        current_global_batch: 현재 글로벌 배치 크기.
        args: 학습 설정이 들어 있는 argparse.Namespace.
        train_loader: 학습용 DataLoader.
        total_step_of_all_epoch: 전체 학습 동안의 총 step 수.
        warmup_steps_at_B0: 기준 배치에서의 워밍업 step 수.
        warmup_steps: 현재 배치에서의 워밍업 step 수.
    """
    if global_rank == 0:
        print("==========[Learning HYPERPARAMETER INFO]===============")
        print(f"Dataset Prepared: {train_set_len} train data\n")
        print(f"[Schedule] B0={BASE_GLOBAL_BATCH}, B={current_global_batch}, "
              f"eta_max={args.learning_rate:.3e}, "
              f"E(B)={args.train_epochs}, "
              f"steps/epoch={len(train_loader)}, "
              f"T={total_step_of_all_epoch}, "
              f"Tw0={warmup_steps_at_B0}, Tw(B)={warmup_steps}")
        print("=====================================================")


def _create_diffusion_planner_and_ema(
    args: argparse.Namespace,
    rank: int,
    use_deepspeed: bool,
) -> Tuple[nn.Module, Optional[ModelEma], nn.Module]:
    """Diffusion_Planner, EMA, base_model(DDP unwrap)을 한 번에 만든다.

    처리 순서:
      1) Diffusion_Planner(args) 로 원본 모델을 만들고,
         rank 또는 args.device 로 to() 호출한다.
      2) DDP 모드이면 torch.nn.parallel.DistributedDataParallel 로 래핑한다.
      3) use_ema=True 이면 timm.utils.ModelEma 로 EMA 복사본을 만든다.
      4) ddp.get_model(...) 로 DDP/DeepSpeed 래퍼를 벗긴 실제 모듈(base_model)을 얻는다.

    Args:
        args (argparse.Namespace):
            - device(str): "cuda" 또는 "cpu" 등.
            - ddp(bool): DDP 사용 여부.
            - use_ema(bool): EMA 사용 여부.
        rank (int):
            - 이 프로세스에서 사용할 GPU 인덱스.  # shape: ()
        use_deepspeed (bool):
            - DeepSpeed 모드 사용 여부. True 이면 DDP 래퍼를 만들지 않는다.

    Returns:
        Tuple[nn.Module, Optional[ModelEma], nn.Module]:
            - diffusion_planner:
                · 원본 모델 또는 DDP/DeepSpeed 래핑된 모델.
            - model_ema:
                · EMA 래퍼 또는 None.
            - base_model:
                · ddp.get_model 으로 래퍼를 벗긴 실제 nn.Module.
                  base_model.parameters() 텐서들의 shape 는
                  레이어 종류에 따라 (out_dim, in_dim), (dim,), (C_out, C_in, kH, kW) 등이다.
    """
    diffusion_planner = Diffusion_Planner(args)
    device = rank if args.device == 'cuda' else args.device
    diffusion_planner = diffusion_planner.to(device)

    # DeepSpeed를 쓸 때는 torch.nn.parallel.DDP 래퍼는 사용하지 않는다.
    if args.ddp and (not use_deepspeed):
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])

    model_ema: Optional[ModelEma] = None
    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    # DDP를 쓴 경우에만 래퍼를 벗겨서 실제 모듈 기준으로 param_group을 만든다.
    base_model = ddp.get_model(
        diffusion_planner,
        args.ddp and (not use_deepspeed),
    )
    return diffusion_planner, model_ema, base_model


def _build_optimizer_with_roles_from_args(
    base_model: nn.Module,
    args: argparse.Namespace,
) -> optim.Optimizer:
    """Group A/B/C lr 배율과 freeze 설정을 반영한 AdamW 옵티마이저를 만든다.

    처리 순서:
      1) freeze_encoder_local 이 True 이면
         Group A(encoder_local) 파라미터들의 requires_grad 를 False 로 바꿔
         옵티마이저에서 완전히 제외한다.
      2) encoder_local_lr_scale / encoder_global_lr_scale / decoder_lr_scale / default
         값을 읽어 role_to_lr_scale 딕셔너리를 구성한다.
      3) build_adamw_with_param_groups(...) 를 호출해
         역할별 lr 배율이 반영된 AdamW(또는 8bit AdamW)를 만든다.
      4) 각 param_group 에
         - "lr_max": 기준 학습률
         - "wd_max": 기준 weight_decay
         필드를 채워 이후 WD warmdown 에서 사용할 수 있도록 한다.

    Args:
        base_model (nn.Module):
            - ddp.get_model 으로 래퍼가 벗겨진 실제 모델.
            - iter_group_encoder_local_parameters / iter_group_encoder_global_parameters /
              iter_group_decoder_parameters 메서드를 제공해야 Group A/B/C 제어가 적용된다.
        args (argparse.Namespace):
            - learning_rate (float)
            - weight_decay (float)
            - use_8bit_optimizer (bool)
            - freeze_encoder_local (bool)
            - encoder_local_lr_scale (float)
            - encoder_global_lr_scale (float)
            - decoder_lr_scale (float)

    Returns:
        optim.Optimizer:
            - torch.optim.AdamW 또는 bnb.optim.AdamW8bit 옵티마이저.
            - optimizer.param_groups[i]["params"] 리스트 안에는
              다양한 shape의 nn.Parameter 텐서들이 들어 있다.
    """
    # 1) Group A freeze (Stage2 에서 encoder_local 고정)
    if args.freeze_encoder_local:
        _set_requires_grad_for_encoder_local(
            model=base_model,
            requires_grad=False,
        )

    # 2) 역할별 lr 스케일 설정
    role_to_lr_scale: Dict[str, float] = {
        "encoder_local": float(args.encoder_local_lr_scale),
        "encoder_global": float(args.encoder_global_lr_scale),
        "decoder": float(args.decoder_lr_scale),
        "default": 1.0,
    }
    # 변경
    optimizer, _ = build_adamw_with_param_groups(
        model=base_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,  # 예: 1e-2
        include_seed_params=True,
        use_8bit_optimizer=args.use_8bit_optimizer,
        role_to_lr_scale=role_to_lr_scale,
    )

    # 각 param_group 에 기준 lr, wd 기록
    for pg in optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))  # 기준 LR (역할별로 다를 수 있음)
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))  # 기준 WD

    return optimizer


def _build_scheduler_from_args(
    args: argparse.Namespace,
    optimizer: optim.Optimizer,
    total_step_of_all_epoch: int,
    warmup_steps: int,
) -> Any:
    """lr_scheduler_type / use_lr_warmup 설정에 맞는 스케줄러를 생성한다.

    처리 규칙:
      - lr_scheduler_type == "cosine":
          · build_pytorch_warmup_cosine_scheduler 사용
          · eta_min 은
            · min_learning_rate 가 설정되어 있으면 그 값,
            · 아니면 0.2 * learning_rate 를 사용한다.
      - lr_scheduler_type == "uniform":
          · build_pytorch_warmup_constant_scheduler 사용
          · 워밍업 이후에는 lr 을 계속 유지한다.
      - use_lr_warmup == False 이면 warmup_steps 를 0 으로 보고 동작한다.

    Args:
        args (argparse.Namespace):
            - lr_scheduler_type (str): "cosine" 또는 "uniform".
            - use_lr_warmup (bool): 워밍업 사용 여부.
            - min_learning_rate (Optional[float]): cosine 스케줄에서 최소 lr.
            - learning_rate (float): 기준 lr.
        optimizer (optim.Optimizer):
            - param_groups[*] 안에 "lr" 필드가 있는 옵티마이저.
        total_step_of_all_epoch (int):
            - 전체 학습 동안 scheduler.step() 이 호출될 총 step 수.  # shape: ()
        warmup_steps (int):
            - 기준 배치에서 계산된 워밍업 step 수.  # shape: ()

    Returns:
        Any:
            - torch.optim.lr_scheduler._LRScheduler 또는 DeepSpeed가 감싼 스케줄러.
    """
    scheduler_type: str = str(getattr(args, "lr_scheduler_type",
                                      "cosine")).lower()
    use_lr_warmup: bool = bool(getattr(args, "use_lr_warmup", True))
    warmup_steps_for_scheduler: int = warmup_steps if use_lr_warmup else 0

    if scheduler_type == "uniform":
        # 선형 워밍업 후, lr 고정
        scheduler = build_pytorch_warmup_constant_scheduler(
            optimizer=optimizer,
            total_update_steps=total_step_of_all_epoch,
            warmup_steps=warmup_steps_for_scheduler,
        )
    elif scheduler_type == "cosine":
        # 선형 워밍업 + 코사인 디케이 + 꼬리 hold
        min_lr_cfg = getattr(args, "min_learning_rate", None)
        if min_lr_cfg is None:
            eta_min = 0.2 * float(args.learning_rate)
        else:
            eta_min = float(min_lr_cfg)

        scheduler = build_pytorch_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_update_steps=total_step_of_all_epoch,
            warmup_steps=warmup_steps_for_scheduler,
            eta_min=eta_min,
        )
    else:
        raise ValueError(f"지원하지 않는 lr_scheduler_type 입니다: {scheduler_type}")

    return scheduler


def _distributed_barrier() -> None:
    """분산 학습 상태라면 모든 프로세스가 같은 지점에서 만나도록 기다린다.

    왜 필요한가?
    - 여러 프로세스(rank)가 동시에 체크포인트를 저장할 때는,
      어떤 프로세스는 파일 저장이 끝났고 다른 프로세스는 아직 저장 중일 수 있다.
    - 이 상태에서 누군가가 'latest.pth 같은 작은 메타 파일'을 먼저 써버리면,
      겉으로는 저장이 끝난 것처럼 보이지만 실제 체크포인트 폴더는 아직 완성되지 않아
      재개(resume)나 업로드가 불안정해질 수 있다.

    Returns:
        None
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _torch_save_to_bytes(obj: Any) -> bytes:
    """torch.save 결과를 메모리(bytes)로 만든다.

    Args:
        obj: torch.save로 저장 가능한 객체(예: dict[str, scalar]).
            - 예) {"epoch": 10, "loss": 1.23, ...}

    Returns:
        bytes: torch.save 결과 바이트열.
    """
    with io.BytesIO() as buff:
        torch.save(obj, buff)
        return buff.getvalue()


def _atomic_write_bytes(file_path: str, data_bytes: bytes) -> None:
    """작은 바이너리 데이터를 '깨질 확률을 줄이는 방식'으로 파일에 저장한다.

    구현 방식:
    - 같은 디렉터리에 임시 파일(tmp)을 먼저 만든다.
    - tmp에 전체 바이트를 쓰고 flush + fsync 한다.
    - 마지막에 os.replace로 목표 파일을 한 번에 교체한다.
      (중간에 죽어도 기존 파일이 반쯤 깨진 상태로 남는 일을 줄인다)

    Args:
        file_path: 최종 저장 경로. shape: ()
        data_bytes: 저장할 바이트열. shape: ()

    Returns:
        None
    """
    dir_path = os.path.dirname(file_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    tmp_path = f"{file_path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        f.write(data_bytes)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            # 일부 파일시스템/환경에서 fsync가 실패할 수 있어도,
            # replace 기반 원자 교체는 계속 진행한다.
            pass

    os.replace(tmp_path, file_path)


def _write_deepspeed_meta_checkpoint_files(
    save_path: Optional[str],
    epoch: int,
    train_total_loss: float,
    wandb_run_id: Optional[str],
    save_best: bool,
) -> None:
    """DeepSpeed 모드용 얇은 메타 체크포인트(latest/best)를 저장한다.

    이 파일은 '실제 모델 가중치/옵티마 상태'가 아니라,
    재개할 때 참고할 작은 정보만 담는다.

    저장 내용(모두 스칼라):
      - epoch (int): 다음에 시작할 epoch 인덱스. shape: ()
      - loss (float): 해당 epoch 손실 값. shape: ()
      - wandb_id (Optional[str]): 이어서 쓸 W&B run id. shape: ()
      - is_deepspeed (bool): DeepSpeed 메타 파일임을 표시. shape: ()

    주의:
    - 이 함수는 "rank 0만 호출"되는 것을 전제로 설계하는 것이 안전하다.
      (모든 rank가 동시에 같은 파일(latest.pth)을 쓰면 파일이 깨질 수 있다)

    Args:
        save_path: 체크포인트 루트 디렉터리 경로.
        epoch: 현재 epoch(0부터 시작).
        train_total_loss: 이번 epoch 손실 스칼라.
        wandb_run_id: W&B run id 또는 None.
        save_best: True면 best.pth도 갱신한다.

    Returns:
        None
    """
    if save_path is None:
        raise ValueError(
            "_write_deepspeed_meta_checkpoint_files 에는 유효한 save_path 가 필요합니다.")

    meta_dict: Dict[str, Any] = {
        "epoch": int(epoch + 1),
        "loss": float(train_total_loss),
        "wandb_id": wandb_run_id,
        "is_deepspeed": True,
    }

    os.makedirs(save_path, exist_ok=True)

    data_bytes = _torch_save_to_bytes(meta_dict)

    # latest.pth 메타 저장
    latest_path = os.path.join(save_path, "latest.pth")
    _atomic_write_bytes(latest_path, data_bytes)

    # best 갱신 시 best.pth도 동일 내용으로 덮어쓴다.
    if save_best:
        best_path = os.path.join(save_path, "best.pth")
        _atomic_write_bytes(best_path, data_bytes)


def _save_deepspeed_checkpoint_for_epoch(
    diffusion_planner: torch.nn.Module,
    save_path: Optional[str],
    epoch: int,
    train_total_loss: float,
    wandb_run_id: Optional[str],
    model_ema: Optional[ModelEma],
    use_deepspeed: bool,
    save_best: bool,
    global_rank: int,
) -> Tuple[str, str]:
    """DeepSpeed 엔진일 때 한 epoch 끝에서 전체 학습 상태를 체크포인트로 저장한다.

    핵심 규칙
    --------
    - DeepSpeedEngine.save_checkpoint(...) 는 각 rank가 자기 파티션을 저장해야 하므로
      모든 rank가 반드시 호출해야 한다.
    - 반면, latest.pth / best.pth 같은 작은 메타 파일은
      모든 rank가 동시에 쓰면 파일이 깨질 수 있으므로 rank 0만 쓴다.
    - 체크포인트 폴더 저장이 다 끝난 뒤에 meta를 쓰도록 barrier로 순서를 맞춘다.

    Args:
        diffusion_planner: deepspeed.DeepSpeedEngine 인스턴스를 기대.
        save_path: 체크포인트 루트 디렉터리.
        epoch: 현재 epoch 인덱스(0부터).
        train_total_loss: 이번 epoch 손실 스칼라.
        wandb_run_id: W&B run id 또는 None.
        model_ema: EMA 래퍼 또는 None.
        use_deepspeed: True여야 동작.
        save_best: best 체크포인트도 저장할지 여부.
        global_rank: 전역 rank. meta 파일은 global_rank==0에서만 작성.

    Returns:
        Tuple[str, str]:
            - tag_latest: 이번 epoch의 latest 태그 문자열
            - tag_best: 이번 epoch의 best 태그 문자열(문자열은 항상 만들지만, save_best=False면 폴더는 안 생길 수 있음)
    """
    if (not use_deepspeed) or save_path is None:
        raise ValueError(
            "_save_deepspeed_checkpoint_for_epoch 는 use_deepspeed=True 및 유효한 save_path 가 필요합니다."
        )
    if not hasattr(diffusion_planner, "save_checkpoint"):
        raise ValueError(
            "diffusion_planner 는 deepspeed.DeepSpeedEngine 인스턴스여야 합니다.")

    # EMA 상태 dict 준비 (없으면 None)
    ema_state_dict: Optional[Dict[str, Any]] = None
    if model_ema is not None:
        try:
            ema_state_dict = model_ema.ema.state_dict()
        except Exception:
            ema_state_dict = model_ema.state_dict()

    client_state: Dict[str, Any] = {
        "epoch": int(epoch + 1),
        "loss": float(train_total_loss),
        "wandb_id": wandb_run_id,
        "ema_state_dict": ema_state_dict,
    }

    tag_latest = f"latest_epoch-{epoch + 1:06d}"
    tag_best = f"best_epoch-{epoch + 1:06d}"

    # 1) latest 저장: 모든 rank가 호출
    diffusion_planner.save_checkpoint(
        save_dir=save_path,
        tag=tag_latest,
        client_state=client_state,
    )

    # 모든 rank의 latest 저장 완료까지 대기
    _distributed_barrier()

    # 2) best 저장: save_best인 경우에만(단, save_best는 모든 rank에서 동일하다고 가정)
    if save_best:
        diffusion_planner.save_checkpoint(
            save_dir=save_path,
            tag=tag_best,
            client_state=client_state,
        )

    # best 저장 여부와 관계없이 한 번 더 동기화(순서 안정화)
    _distributed_barrier()

    # 3) meta 파일(latest.pth / best.pth)은 rank 0만 작성
    if global_rank == 0:
        _write_deepspeed_meta_checkpoint_files(
            save_path=save_path,
            epoch=epoch,
            train_total_loss=train_total_loss,
            wandb_run_id=wandb_run_id,
            save_best=save_best,
        )

    # meta 파일 작성 완료까지 대기
    _distributed_barrier()

    return tag_latest, tag_best


def _is_deepspeed_checkpoint_dir(save_path: str) -> bool:
    """주어진 경로가 DeepSpeed checkpoint 디렉터리인지 대략 판별한다.

    규칙(간단한 휴리스틱):
      - save_path 하위 폴더들 중 이름에 "latest" 또는 "best" 가 들어가는
        디렉터리를 찾고
      - 그 디렉터리 안에 "mp_rank_" 로 시작하는 하위 파일/디렉터리가 있으면
        DeepSpeed 가 저장한 구조라고 본다.

    Args:
        save_path:
            체크포인트 루트 디렉터리 경로.

    Returns:
        bool:
            True  → DeepSpeed 형식으로 저장된 디렉터리라고 판단.
            False → PyTorch 단일 .pth 형식이라고 보고 처리한다.
    """
    if not os.path.isdir(save_path):
        return False

    try:
        # save_path 하위의 1차 자식 디렉터리 목록
        top_children = os.listdir(save_path)
    except OSError:
        return False

    # 이름에 "latest" 또는 "best" 가 들어가는 하위 디렉터리만 후보로 사용
    candidate_tag_dirs = []
    for name in sorted(top_children):
        if ("latest" in name) or ("best" in name):
            full = os.path.join(save_path, name)
            if os.path.isdir(full):
                candidate_tag_dirs.append(full)

    # 후보 tag 디렉터리들 중 하나라도 mp_rank_* 를 포함하면 DeepSpeed 형식으로 판단
    for tag_dir in candidate_tag_dirs:
        try:
            children = os.listdir(tag_dir)
        except OSError:
            continue
        for name in children:
            # 예: mp_rank_00_model_states.pt, mp_rank_00/
            if name.startswith("mp_rank_"):
                return True

    return False


def _update_deepspeed_optimizer_param_group_meta(
    diffusion_planner: nn.Module,) -> None:
    """DeepSpeed 엔진 내부 옵티마이저 param_group 메타 정보를 보정한다.

    DeepSpeed 모드에서는
      - diffusion_planner.optimizer.param_groups[*]["lr"]
      - diffusion_planner.optimizer.param_groups[*]["weight_decay"]
    값들만 채워져 있는 경우가 있어, 이후 WD warmdown 등의 처리를 위해
    각 그룹에
      - "lr_max": 기준 학습률 (shape: ())
      - "wd_max": 기준 weight decay (shape: ())
    필드를 추가로 넣어 준다.

    Args:
        diffusion_planner (nn.Module):
            - deepspeed.DeepSpeedEngine 인스턴스를 기대한다.
            - diffusion_planner.optimizer.param_groups 는
              각 그룹별로 "params": List[nn.Parameter] 를 가진다.
              각 파라미터 텐서 shape 는 (out_dim, in_dim), (dim,),
              (C_out, C_in, kH, kW) 등 계층 종류에 따라 다양하다.
    """
    if not hasattr(diffusion_planner, "optimizer"):
        return

    for pg in diffusion_planner.optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))


def _select_latest_like_tag(save_path: str) -> str:
    """DeepSpeed 체크포인트 루트에서 '가장 최신 latest 계열 태그'를 고른다.

    동작 규칙:
    1) save_path 아래에 'latest'라는 디렉터리가 정확히 있으면 그걸 최우선으로 사용한다.
       (사람이 심볼릭 링크/복사로 만들어둔 경우를 고려)
    2) 그 외에는 이름에 "latest"가 들어간 디렉터리들(예: latest_epoch-000010) 중
       정렬 기준 가장 마지막(=가장 큰 숫자) 것을 선택한다.
       - epoch를 6자리 0패딩으로 쓰고 있으므로 문자열 정렬로도 최신 선택이 가능하다.

    Args:
        save_path: 체크포인트 루트 디렉터리.

    Returns:
        str: load_checkpoint에 넣을 tag 문자열.
    """
    tag: str = "latest"
    if not os.path.isdir(save_path):
        return tag

    try:
        children = [
            d for d in os.listdir(save_path)
            if os.path.isdir(os.path.join(save_path, d))
        ]
    except OSError as e:
        raise RuntimeError(
            f"[DeepSpeed] 체크포인트 디렉터리 목록을 읽는 중 오류 발생: {save_path}, {e}") from e

    # 1) 정확히 "latest" 디렉터리가 있으면 우선 사용
    if "latest" in children:
        return "latest"

    # 2) "latest"가 포함된 태그 디렉터리들 중 가장 최신(정렬 마지막) 선택
    candidate_tags = [d for d in children if "latest" in d]
    if candidate_tags:
        return sorted(candidate_tags)[-1]

    return tag


def _restore_ema_and_model_from_client_state_for_deepspeed(
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    client_state: Dict[str, Any],
    global_rank: int,
    load_ema_for_model: bool,
    load_ema_for_ema_model: bool,
) -> Optional[ModelEma]:
    """client_state 정보를 이용해 EMA와 (필요하다면) 모델 가중치를 복원한다.

    Args:
        diffusion_planner (nn.Module): DeepSpeed 엔진 객체.
        model_ema (Optional[ModelEma]): EMA 래퍼. 없으면 None.
        client_state (Dict[str, Any]): 체크포인트에 저장된 작은 상태 dict.
            - client_state.get("ema_state_dict"): Dict[str, Tensor]
              각 텐서의 shape 는 원본 모델 파라미터와 동일
              (예: (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등).

            save_checkpoint 때 넘겨준 client_state 그대로 돌아온 dict.
            epoch, loss, wandb_id, ema_state_dict 가 그 안에 들어있음
            ( client_state 내용을 포함한 메타 정보를 별도의
            체크포인트 파일(예: mp_rank_00_model_states.pt 안의 딕셔너리 키)로 같이 저장해 둔다. )

        global_rank (int): 전역 rank. 0일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]: EMA 상태가 반영된 ModelEma 또는 None.
    """
    ema_state_dict: Optional[Dict[str, Any]] = client_state.get(
        "ema_state_dict", None)

    # 1) EMA 객체가 있고, ema_state_dict 가 있으면 EMA 가중치 복원
    if load_ema_for_ema_model:
        assert (model_ema is not None) and (ema_state_dict is not None), \
            "[DeepSpeed LOAD MODEL PARAM] Restoring an EMA model requires both `model_ema` and `ema_state_dict`."
        try:
            ema_model: nn.Module = getattr(model_ema, "ema", model_ema)
            # ema_model.state_dict() 의 각 텐서 shape:
            #   (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW) 등
            ema_model.load_state_dict(ema_state_dict, strict=False)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)
            if global_rank == 0:
                print("[DeepSpeed LOAD MODEL PARAM]EMA state load done")
        except Exception as e:
            if global_rank == 0:
                print(f"[DeepSpeed LOAD MODEL PARAM]EMA state load FAIL: {e}")

    if load_ema_for_model:
        assert ema_state_dict is not None, \
            "[DeepSpeed LOAD MODEL PARAM] Initializing EMA → base_model requires an EMA `state_dict`."
        try:
            base_model: nn.Module = getattr(diffusion_planner, "module",
                                            diffusion_planner)
            incompatible = base_model.load_state_dict(ema_state_dict,
                                                      strict=False)
            if global_rank == 0:
                if getattr(incompatible, "missing_keys", None):
                    print(
                        f"[DeepSpeed LOAD MODEL PARAM] model-only(EMA) missing_keys: {incompatible.missing_keys}"
                    )
                if getattr(incompatible, "unexpected_keys", None):
                    print(
                        f"[DeepSpeed LOAD MODEL PARAM] model-only(EMA) unexpected_keys: {incompatible.unexpected_keys}"
                    )
                print(
                    "[DeepSpeed LOAD MODEL PARAM] model-only resume: base model initialized from EMA weights"
                )
        except Exception as e:
            if global_rank == 0:
                print(f"... 실패: {e}")
            raise RuntimeError(
                "[DeepSpeed LOAD MODEL PARAM] Error occurred during EMA → base_model initialization in model-only mode."
            ) from e
            if global_rank == 0:
                print(
                    f"[DeepSpeed LOAD MODEL PARAM] EMA→base_model 초기화 실패: {e}")

    return model_ema


def _resume_from_deepspeed_checkpoint_format(
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    save_path: str,
    global_rank: int,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    load_ema_for_model: bool = True,
    load_ema_for_ema_model: bool = True,
) -> Tuple[int, Optional[str], Optional[ModelEma]]:
    """DeepSpeed 형식으로 저장된 체크포인트를 읽어와 학습을 이어갈 준비를 한다.

    Args:
        diffusion_planner (nn.Module): DeepSpeed 엔진 객체.
        model_ema (Optional[ModelEma]): EMA 래퍼. 없으면 None.
        save_path (str): 체크포인트 루트 디렉터리 경로.
        global_rank (int): 전역 rank. 0일 때만 로그를 출력한다.
        load_optimizer_states (bool): 옵티마 상태를 함께 불러올지 여부.
        load_lr_scheduler_states (bool): 스케줄러 상태를 함께 불러올지 여부.

    Returns:
        Tuple[int, Optional[str], Optional[ModelEma]]:
            - init_epoch (int): 재개할 epoch 인덱스(0 기반).
            - wandb_id (Optional[str]): 이어서 사용할 W&B run id 또는 None.
            - model_ema (Optional[ModelEma]): EMA 상태가 반영된 EMA 래퍼 또는 None.
    """
    # 0) tag 자동 선택: save_path 하위 폴더 중 이름에 "latest"가 포함된 디렉터리들 중
    #    정렬 기준 첫 번째 것을 tag로 사용. 없으면 기본값 "latest".
    tag: str = _select_latest_like_tag(save_path)

    if global_rank == 0:
        print(
            f"[DeepSpeed LOAD MODEL PARAM] load_checkpoint: save_path={save_path}, tag='{tag}'"
        )

        # 1) DeepSpeedEngine.load_checkpoint 호출 (신/구 버전 둘 다 지원)
        """
load_path: str
    실제로 로드에 사용된 체크포인트 디렉터리 경로.
    보통 save_path/tag 에 해당하는 문자열 
    (예: /.../training_log/.../latest_epoch-000010)

client_state: Optional[Dict[str, Any]]
    save_checkpoint 때 넘겨준 client_state 그대로 돌아온 dict.
    epoch, loss, wandb_id, ema_state_dict 가 그 안에 들어있음
    ( client_state 내용을 포함한 메타 정보를 별도의 
    체크포인트 파일(예: mp_rank_00_model_states.pt 안의 딕셔너리 키)로 같이 저장해 둔다. )
        """
    load_path, client_state = diffusion_planner.load_checkpoint(
        save_path,
        tag=tag,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )

    if global_rank == 0:
        print(
            f"[DeepSpeed LOAD MODEL PARAM] load_checkpoint returned load_path={load_path}, "
            f"client_state keys={list((client_state or {}).keys())}")

    # client_state 가 None 인 경우에도 이후 로직이 동일하게 동작하도록 빈 dict 로 대체
    client_state = client_state or {}
    init_epoch: int = int(client_state.get("epoch", 0))
    wandb_id: Optional[str] = client_state.get("wandb_id", None)

    # EMA 및 (필요 시) base model 복원
    model_ema = _restore_ema_and_model_from_client_state_for_deepspeed(
        diffusion_planner=diffusion_planner,
        model_ema=model_ema,
        client_state=client_state,
        global_rank=global_rank,
        load_ema_for_model=load_ema_for_model,
        load_ema_for_ema_model=load_ema_for_ema_model,
    )

    # DeepSpeed 엔진 내부 옵티마 param_group 메타 보정
    _update_deepspeed_optimizer_param_group_meta(diffusion_planner)

    return init_epoch, wandb_id, model_ema


def _resume_from_pytorch_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str]]:
    """기존 PyTorch/DDP 형식 latest.pth 를 그대로 불러와서 학습을 재개한다.

    처리 흐름:
      1) diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema =
         resume_model(save_path, ...) 를 호출해
         - 모델 파라미터
         - 옵티마이저 상태
         - 스케줄러 상태
         - EMA 상태
         를 한 번에 복원한다.
      2) optimizer.param_groups[*] 에 대해
         - "lr_max": 기준 학습률 (shape: ())
         - "wd_max": 기준 weight_decay (shape: ())
         를 채워 이후 WD warmdown 에서 사용할 수 있게 한다.

    Args:
        save_path (str):
            - latest.pth 가 들어 있는 PyTorch 체크포인트 디렉터리.
        diffusion_planner (nn.Module):
            - nn.Module 또는 DDP 래핑된 모델.
        optimizer (optim.Optimizer):
            - AdamW/AdamW8bit 옵티마이저.
        scheduler (Any):
            - 학습률 스케줄러 객체.
        model_ema (Optional[ModelEma]):
            - EMA 래퍼 또는 None.
        device (str):
            - "cuda" / "cpu" 등. resume_model 내부에서 map_location 에 사용된다.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 복원된 모델.
            - optimizer: 복원된 옵티마이저.
            - scheduler: 복원된 스케줄러.
            - model_ema: 복원된 EMA 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스(0 기반). shape: ()
            - wandb_id: W&B run id 또는 None.
    """
    (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id,
     model_ema) = resume_model(
         save_path,
         diffusion_planner,
         optimizer,
         scheduler,
         model_ema,
     )

    # resume 후에도 wd_max / lr_max 기본값이 유지되도록 보강
    for pg in optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))

    return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id


def _select_pytorch_checkpoint_path_for_model_only(ckpt_dir: str) -> str:
    """model-only 로딩에 사용할 latest.pth 또는 best.pth 경로를 고른다.

    우선순위:
      1) ckpt_dir/latest.pth 가 있으면 그 경로를 사용한다.
      2) 없으면 ckpt_dir/best.pth 를 시도한다.
      3) 둘 다 없으면 FileNotFoundError 를 발생시킨다.

    Args:
        ckpt_dir (str):
            PyTorch 체크포인트가 들어 있는 디렉터리 경로.

    Returns:
        str:
            실제로 torch.load 에 사용할 체크포인트 파일 경로.
    """
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    if os.path.exists(ckpt_path):
        return ckpt_path

    alt_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(alt_ckpt_path):
        return alt_ckpt_path

    raise FileNotFoundError(
        f"PyTorch 체크포인트 latest.pth / best.pth 를 찾을 수 없습니다: {ckpt_dir}")


def _load_pytorch_checkpoint_dict_for_model_only(
    ckpt_path: str,
    device: str,
) -> Dict[str, Any]:
    """model-only 로딩용 PyTorch 체크포인트 파일을 dict 로 읽어온다.

    Args:
        ckpt_path (str):
            latest.pth 또는 best.pth 전체 경로.
        device (str):
            "cuda" / "cuda:0" / "cpu" 등. torch.load map_location 에 사용된다.

    Returns:
        Dict[str, Any]:
            torch.load 로 읽은 체크포인트 dict.
            - ckpt["model"]: 모델 state_dict (각 value 텐서 shape는
              (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW) 등).
            - ckpt["ema_state_dict"] (선택): EMA state_dict.
    """
    map_location = "cpu" if device.startswith("cuda") else device
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=map_location)
    return ckpt


def _choose_state_dict_for_model_only(
    ckpt: Dict[str, Any],) -> Tuple[Dict[str, torch.Tensor], str]:
    """EMA state 가 있으면 EMA를, 없으면 model state 를 선택한다.

    Args:
        ckpt (Dict[str, Any]):
            torch.load 로 읽어온 체크포인트 dict.

    Returns:
        Tuple[Dict[str, torch.Tensor], str]:
            - state_like:
                모델에 실제로 넣을 state_dict.
            - source_name:
                "EMA" 또는 "model" 문자열. 로그용.
    """
    raw_model_state: Dict[str, torch.Tensor] = ckpt.get("model", ckpt)
    raw_ema_state: Optional[Dict[str, torch.Tensor]] = ckpt.get(
        "ema_state_dict", None)

    if raw_ema_state is not None and isinstance(
            raw_ema_state, dict) and len(raw_ema_state) > 0:
        return raw_ema_state, "EMA"
    else:
        return raw_model_state, "model"


def _strip_or_add_module_prefix_for_model_only(
    src_state: Dict[str, torch.Tensor],
    want_module_prefix: bool,
) -> Dict[str, torch.Tensor]:
    """state_dict 키에 "module." 접두사를 붙이거나 떼는 작은 도우미 함수.

    DDP/DeepSpeed 래핑 여부에 따라
    - 모델 쪽 키는 "module.xxx" 또는 "xxx" 형태일 수 있고,
    - 체크포인트 쪽 키도 두 형태 중 하나일 수 있다.
    이 함수는 원하는 형태(want_module_prefix)에 맞춰
    키 문자열만 정리해 새 dict를 만들어 준다.

    Args:
        src_state (Dict[str, torch.Tensor]):
            체크포인트에서 가져온 state_dict.
            각 value 텐서의 shape 는 레이어 종류에 따라
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등 다양할 수 있다.
        want_module_prefix (bool):
            True  이면 모든 키가 "module." 으로 시작하도록 맞춘다.
            False 이면 "module." 으로 시작하는 키는 접두사를 떼고,
            나머지는 그대로 둔다.

    Returns:
        Dict[str, torch.Tensor]:
            키 문자열만 정리된 새 state_dict.
    """
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        if want_module_prefix:
            # 이미 "module." 로 시작하면 그대로, 아니면 앞에 붙인다.
            new_k = k if k.startswith("module.") else f"module.{k}"
        else:
            # "module." 로 시작하면 떼고, 아니면 그대로 둔다.
            new_k = k[7:] if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state


def _align_state_dict_keys_to_base_model_for_model_only(
    state_like: Dict[str, torch.Tensor],
    base_model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """base_model.state_dict() 와 키 형태(module. prefix 유무)를 맞춘 state_dict 를 만든다.

    DeepSpeed/DDP 래핑 여부에 따라
      - 현재 모델 키는 "module.xxx" 또는 "xxx" 형태일 수 있고,
      - ckpt 키도 "module.xxx" 또는 "xxx" 형태일 수 있다.
    이 함수는 두 쪽의 키 패턴을 비교해서,
    필요한 경우 "module." 접두사를 붙이거나 떼서
    base_model.load_state_dict(...) 에 바로 넣을 수 있는 state_dict 로 맞춰준다.

    Args:
        state_like (Dict[str, torch.Tensor]):
            ckpt 에서 선택한 state_dict.
            각 value 텐서는 레이어에 따라
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등의 shape 를 가진다.
        base_model (nn.Module):
            실제 파라미터를 가진 nn.Module (DDP 래퍼를 벗긴 모델).

    Returns:
        Dict[str, torch.Tensor]:
            base_model.load_state_dict(...) 에 바로 사용할 수 있는 state_dict.
    """
    current_keys = list(base_model.state_dict().keys())
    ckpt_keys = list(state_like.keys())

    has_module_in_ckpt = all(k.startswith("module.") for k in ckpt_keys)
    has_module_in_model = all(k.startswith("module.") for k in current_keys)

    # ckpt: module.* / 모델: 평범한 키 → prefix 제거
    if has_module_in_ckpt and (not has_module_in_model):
        return _strip_or_add_module_prefix_for_model_only(
            state_like,
            want_module_prefix=False,
        )
    # ckpt: 평범한 키 / 모델: module.* → prefix 추가
    elif (not has_module_in_ckpt) and has_module_in_model:
        return _strip_or_add_module_prefix_for_model_only(
            state_like,
            want_module_prefix=True,
        )
    else:
        # 둘 다 module.* 이거나, 둘 다 평범한 키인 경우 그대로 사용
        return state_like


def _load_state_dict_into_base_and_ema_model_only(
    state_dict: Dict[str, torch.Tensor],
    source_name: str,
    base_model: nn.Module,
    model_ema: Optional[ModelEma],
    ckpt_path: str,
    global_rank: int,
) -> Optional[ModelEma]:
    """선택된 state_dict 를 base_model 과 EMA 모델에 주입하고 로그를 남긴다.

    처리 순서:
      1) base_model.load_state_dict(..., strict=False) 로 파라미터 로드.
      2) global_rank==0 인 경우 missing_keys / unexpected_keys 를 출력.
      3) model_ema 가 있으면 base_model.state_dict() 로 EMA 모델도 초기화하고,
         eval 모드 + requires_grad=False 로 고정한다.

    Args:
        state_dict (Dict[str, torch.Tensor]):
            base_model.load_state_dict 에 넣을 state_dict.
        source_name (str):
            "EMA" 또는 "model". 어떤 state 를 사용했는지 로그용.
        base_model (nn.Module):
            실제 파라미터를 가진 nn.Module.
            base_model.state_dict() 의 각 텐서는
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등 shape 를 가진다.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        ckpt_path (str):
            사용한 체크포인트 파일 경로. 로그용.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]:
            EMA 상태가 덮어써진 model_ema (또는 원래 None).
    """
    incompatible = base_model.load_state_dict(state_dict, strict=False)
    if global_rank == 0:
        print(
            f"[ModelOnly<Pytorch>] {source_name} state_dict 로 LOAD MODEL PARAM 완료 "
            f"(ckpt_path={ckpt_path})")
        if getattr(incompatible, "missing_keys", None):
            print(
                f"[ModelOnly<Pytorch>] missing_keys: {incompatible.missing_keys}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            print(
                f"[ModelOnly<Pytorch>] unexpected_keys: {incompatible.unexpected_keys}"
            )

    # EMA 객체도 있으면 base_model 과 동일한 파라미터로 맞춰준다.
    if model_ema is not None:
        try:
            ema_model: nn.Module = getattr(model_ema, "ema", model_ema)
            ema_incompatible = ema_model.load_state_dict(
                base_model.state_dict(),
                strict=False,
            )
            ema_model.eval()
            for p in ema_model.parameters():
                # 각 파라미터 텐서 shape: (out_dim, in_dim), (hidden_dim,), ...
                p.requires_grad_(False)
            if global_rank == 0:
                if getattr(ema_incompatible, "missing_keys", None):
                    print(
                        f"[ModelOnly<Pytorch>] EMA missing_keys: {ema_incompatible.missing_keys}"
                    )
                if getattr(ema_incompatible, "unexpected_keys", None):
                    print(
                        f"[ModelOnly<Pytorch>] EMA unexpected_keys: {ema_incompatible.unexpected_keys}"
                    )
                print("[ModelOnly<Pytorch>] EMA state 동기화 완료")
        except Exception as e:
            if global_rank == 0:
                print(f"[ModelOnly<Pytorch>] EMA state 동기화 실패: {e}")

    return model_ema


def _load_model_and_ema_state_only_from_pytorch_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    device: str,
    global_rank: int,
) -> Optional[ModelEma]:
    """PyTorch latest.pth 에서 모델/EMA 파라미터만 불러와 새 학습을 시작할 때 사용한다.

    전체 흐름:
      1) save_path 안에서 latest.pth 또는 best.pth 경로를 고른다.
      2) 선택한 파일을 torch.load 로 읽어 체크포인트 dict 를 얻는다.
      3) EMA state_dict 가 있으면 EMA를, 없으면 model state_dict 를 선택한다.
      4) base_model(state_dict) 의 키 형태와 맞도록 module. prefix 를 조정한다.
      5) base_model.load_state_dict(...) 로 파라미터를 주입하고,
         model_ema 가 있으면 base_model.state_dict() 로 EMA도 초기화한다.

    Args:
        save_path (str):
            latest.pth / best.pth 가 들어 있는 디렉터리 경로.
        diffusion_planner (nn.Module):
            현재 학습에 사용할 모델 또는 DDP 래퍼.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        device (str):
            "cuda" / "cpu" 등. torch.load 의 map_location 에 사용된다.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]:
            EMA 상태가 덮어써진 model_ema (또는 원래 None).
    """
    ckpt_path: str = _select_pytorch_checkpoint_path_for_model_only(save_path)
    ckpt: Dict[str, Any] = _load_pytorch_checkpoint_dict_for_model_only(
        ckpt_path=ckpt_path,
        device=device,
    )

    # base_model: 실제 nn.Module (DDP 래퍼 벗겨진 상태)
    base_model: nn.Module = getattr(diffusion_planner, "module",
                                    diffusion_planner)

    state_like, source_name = _choose_state_dict_for_model_only(ckpt)
    state_dict = _align_state_dict_keys_to_base_model_for_model_only(
        state_like=state_like,
        base_model=base_model,
    )

    model_ema = _load_state_dict_into_base_and_ema_model_only(
        state_dict=state_dict,
        source_name=source_name,
        base_model=base_model,
        model_ema=model_ema,
        ckpt_path=ckpt_path,
        global_rank=global_rank,
    )
    return model_ema


def _load_model_and_ema_state_only_from_deepspeed_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    global_rank: int,
) -> Optional[ModelEma]:
    """DeepSpeed 폴더에서 **모델/EMA 가중치만** 읽어오는 함수.

    이 함수는 optimizer나 스케줄러 상태는 전혀 건드리지 않고,
    디스크에 저장된 DeepSpeed 체크포인트 폴더에서
    순수하게 모델 값(가중치)만 불러와 현재 모델에 채워 넣는다.

    사용 의도:
        * stage1에서 이미 학습한 모델 값을 가져오되,
          stage2에서는 새로운 학습 설정(optimizer, 학습률 등)을 그대로 쓰고 싶을 때.
        * encoder_local을 고정(freeze)해서 메모리를 아끼고 싶지만,
          DeepSpeed의 복잡한 재개(load) 로직은 피하고 싶을 때.

    동작 요약:
        1) save_path 안에서 "latest"가 들어간 하위 폴더 이름을 고른다.
           예: latest_epoch-000001
        2) 그 안의 mp_rank_XX_model_states.pt 파일을 열어
           저장된 모델 값들을 읽어온다.
        3) 읽어온 모델 값을 현재 모델(diffusion_planner)에 넣는다.
        4) EMA 모델이 있으면, 현재 모델 값으로 EMA 모델도 맞춰 준다.

    Args:
        save_path (str):
            DeepSpeed 체크포인트가 들어 있는 루트 폴더 경로.
            예: "./training_log/실험이름/2025-12-10-12:52:12/"
        diffusion_planner (nn.Module):
            DeepSpeed로 감싸진 주행 계획 모델 또는 그와 같은 구조의 모델.
        model_ema (Optional[ModelEma]):
            EMA(지수 이동 평균)를 추적하는 보조 모델 래퍼.
            없으면 None.
        global_rank (int):
            분산 학습에서 이 프로세스의 전체 순번.
            주로 로그 출력에만 사용된다.

    Returns:
        Optional[ModelEma]:
            모델 값이 동기화된 EMA 래퍼를 돌려준다.
            EMA를 쓰지 않는 경우에는 원래대로 None을 돌려준다.
    """
    tag = _select_latest_like_tag(save_path)
    tag_dir = os.path.join(save_path, tag)

    # tag 디렉터리 안에서 *_model_states.pt 중 첫 번째를 찾는 방식
    candidates = [
        f for f in os.listdir(tag_dir) if f.endswith("_model_states.pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_model_states.pt found in {tag_dir}")

    mp_rank_str = sorted(candidates)[0]  # 보통 mp_rank_00_model_states.pt
    ckpt_path = os.path.join(tag_dir, mp_rank_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_like = ckpt.get("ema_state_dict", ckpt["module"])

    base_model: nn.Module = getattr(diffusion_planner, "module",
                                    diffusion_planner)
    incompatible = base_model.load_state_dict(state_like, strict=False)

    if global_rank == 0:
        print(f"[ModelOnly<DeepSpeed>] LOAD MODEL PARAM from {ckpt_path}")
        if getattr(incompatible, "missing_keys", None):
            print(
                f"[ModelOnly<DeepSpeed>] missing_keys: {incompatible.missing_keys}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            print(
                f"[ModelOnly<DeepSpeed>] unexpected_keys: {incompatible.unexpected_keys}"
            )

    if model_ema is not None:
        ema_model = getattr(model_ema, "ema", model_ema)
        ema_model.load_state_dict(base_model.state_dict(), strict=False)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    return model_ema


def _resume_from_checkpoint_with_deepspeed(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    global_rank: int,
    resume_model_only: bool,
) -> Tuple[nn.Module, Optional[ModelEma], int, Optional[str]]:
    """DeepSpeed 모드에서 args.save_path 기준으로 모델/EMA를 불러온다.

    처리 규칙:
      - args.save_path 이 DeepSpeed 전용 디렉터리 구조이면
        _resume_from_deepspeed_checkpoint_format(...) 을 사용한다.
        · resume_model_only=False → optimizer/scheduler 상태까지 함께 복원.
        · resume_model_only=True  → optimizer/scheduler 는 건드리지 않고,
          모델/EMA만 로드한다.
      - args.save_path 이 PyTorch latest.pth 형식이면
        _resume_deepspeed_from_pytorch_checkpoint(...) 를 사용해
        모델/EMA를 불러온다. (optimizer/scheduler는 그대로)

      - resume_model_only=True 인 경우:
        · init_epoch=0, wandb_id=None 으로 돌려준다.
      - resume_model_only=False 인 경우:
        · ckpt 에서 읽은 init_epoch / wandb_id 를 그대로 사용한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace. device 정보 등을 포함한다.
        diffusion_planner (nn.Module):
            deepspeed.DeepSpeedEngine 또는 DDP 래퍼가 씌워진 모델.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 상세 로그를 출력한다.
        resume_model_only (bool):
            True 이면 모델/EMA 파라미터만 불러오는 모드.

    Returns:
        Tuple[nn.Module, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 모델/EMA 로딩이 반영된 모델.
            - model_ema: EMA 상태가 복원된 ModelEma 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스. resume_model_only=True 이면 0.
            - wandb_id: 이어서 사용할 W&B run id 또는 None.
    """
    # ✅ 1. model-only + DeepSpeed인 경우: DeepSpeedEngine.load_checkpoint 안 쓴다
    if resume_model_only:
        model_ema = _load_model_and_ema_state_only_from_deepspeed_checkpoint(
            save_path=args.save_path,
            diffusion_planner=diffusion_planner,
            model_ema=model_ema,
            global_rank=global_rank,
        )
        init_epoch = 0
        wandb_id = None
        return diffusion_planner, model_ema, init_epoch, wandb_id
    elif _is_deepspeed_checkpoint_dir(args.save_path):
        init_epoch_loaded, wandb_id_loaded, model_ema = \
            _resume_from_deepspeed_checkpoint_format(
                diffusion_planner=diffusion_planner,
                model_ema=model_ema,
                save_path=args.save_path,
                global_rank=global_rank,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
                load_ema_for_model=args.load_ema_for_model,
                load_ema_for_ema_model=args.load_ema_for_ema_model,
            )
    else:
        raise ValueError(
            "The save_path does not contain a valid DeepSpeed checkpoint directory."
        )

    if resume_model_only:
        init_epoch = 0
        wandb_id = None
    else:
        init_epoch = init_epoch_loaded
        wandb_id = wandb_id_loaded

    return diffusion_planner, model_ema, init_epoch, wandb_id


def _resume_from_checkpoint_with_pytorch(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    save_path: str,
    global_rank: int,
    resume_model_only: bool,
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str]]:
    """일반 PyTorch/DDP 모드에서 save_path 기준으로 모델/옵티마를 불러온다.

    처리 규칙:
      - resume_model_only=True:
          · _load_model_and_ema_state_only_from_pytorch_checkpoint(...) 를 사용해
            모델/EMA 파라미터만 불러온다.
          · optimizer / scheduler 상태는 그대로 둔다.
          · init_epoch=0, wandb_id=None 으로 돌려준다.
      - resume_model_only=False:
          · _resume_from_pytorch_checkpoint(...) 로
            모델/optimizer/scheduler/EMA/epoch/wandb_id 를 모두 복원한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace. device 정보 포함.
        diffusion_planner (nn.Module):
            현재 학습 중인 모델 또는 DDP 래퍼.
        optimizer (optim.Optimizer):
            AdamW 또는 8bit AdamW 옵티마이저.
        scheduler (Any):
            학습률 스케줄러 객체.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        save_path (str):
            PyTorch latest.pth / best.pth 가 들어 있는 디렉터리 경로.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 상세 로그 출력.
        resume_model_only (bool):
            True 이면 모델/EMA 파라미터만 불러오는 모드.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 복원된 모델.
            - optimizer: 복원된(또는 기존) 옵티마이저.
            - scheduler: 복원된(또는 기존) 스케줄러.
            - model_ema: EMA 상태가 복원된 ModelEma 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스. model-only 이면 0.
            - wandb_id: 이어서 사용할 W&B run id 또는 None.
    """
    if resume_model_only:
        # 모델(또는 EMA) weight만 불러오고 optimizer/scheduler 는 그대로 유지
        model_ema = _load_model_and_ema_state_only_from_pytorch_checkpoint(
            save_path=save_path,
            diffusion_planner=diffusion_planner,
            model_ema=model_ema,
            device=args.device,
            global_rank=global_rank,
        )
        init_epoch = 0
        wandb_id = None
        return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id

    # 전체 상태 복원 (기존 PyTorch 경로)
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch,
     wandb_id) = _resume_from_pytorch_checkpoint(
         save_path=save_path,
         diffusion_planner=diffusion_planner,
         optimizer=optimizer,
         scheduler=scheduler,
         model_ema=model_ema,
     )
    return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id


def _finalize_train_epochs_and_wandb_after_resume(
    args: argparse.Namespace,
    init_epoch: int,
    resume_model_only: bool,
    global_rank: int,
) -> Tuple[int, bool]:
    """체크포인트 로드 이후 train_epochs 와 allow_val_change 값을 최종 결정한다.

    처리 규칙:
      - resume_model_only=True:
          · epoch 정보는 모두 무시하고,
            args.train_epochs 를 그대로 사용한다.
          · allow_val_change=False 로 둔다 (새 run 개념).
      - resume_model_only=False:
          · init_epoch 가 이미 지난 epoch 수이므로,
            args.train_epochs <= init_epoch 이면 init_epoch+1 로 늘린다.
          · args.resume_wandb_model_name 가 설정되어 있으면
            allow_val_change=True 로 W&B config 변경을 허용한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace.
        init_epoch (int):
            체크포인트에서 읽어온 재개 시작 epoch 인덱스(0 기반).
        resume_model_only (bool):
            True 이면 model-only 재시작 모드.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Tuple[int, bool]:
            - train_epochs: 최종 사용할 train_epochs 값.
            - allow_val_change: W&B config 변경 허용 여부.
    """
    allow_val_change: bool = False

    if resume_model_only:
        # model-only 재시작: epoch 정보는 모두 무시하고 config 값 그대로 사용
        train_epochs = int(args.train_epochs)
        return train_epochs, allow_val_change

    # 기존 full-resume 경로: init_epoch 를 기준으로 train_epochs 보정
    if args.train_epochs <= init_epoch:
        old = int(args.train_epochs)
        args.train_epochs = int(init_epoch) + 1
        if global_rank == 0:
            print(
                f"[ADJUST] scaled train_epochs ({old}) <= init_epoch ({init_epoch}). "
                f"Bumping to {args.train_epochs}.")
    train_epochs = int(args.train_epochs)

    # W&B 아티팩트에서 재개하는 경우 config 값 변경 허용
    if args.resume_wandb_model_name:
        allow_val_change = True

    return train_epochs, allow_val_change


def _maybe_resume_from_checkpoint(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    global_rank: int,
    use_deepspeed: bool,
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str], int, bool]:
    """local 또는 W&B에서 지정한 체크포인트가 있으면 학습 상태를 복원한다.

    이 함수는 "재개할 체크포인트가 있는지"를 먼저 확인한 뒤,
    상황에 따라 다음과 같이 동작한다.

      * resume_model_only=False (기존 resume 모드):
        - DeepSpeed 모드 + DeepSpeed 형식 디렉터리:
            · 모델/옵티마이저/스케줄러/EMA/epoch/wandb_id 를 모두 복원.
        - DeepSpeed 모드 + PyTorch latest.pth:
            · 모델/EMA 를 불러오고, epoch/wandb_id 를 사용해 계속 진행
              (옵티마이저/스케줄러는 새 상태로 시작).
        - 일반 PyTorch/DDP 모드:
            · resume_model(...) 을 통해 모델/옵티마이저/스케줄러/EMA/epoch/wandb_id 복원.

      * resume_model_only=True (model-only 재시작 모드):
        - 어떤 형식이든 모델/EMA 파라미터만 불러오고,
          optimizer/scheduler/epoch/wandb_id 는 모두 새로 시작한다.
          · DeepSpeed: load_optimizer_states=False, load_lr_scheduler_states=False 로 로드.
          · PyTorch: _load_model_and_ema_state_only_from_pytorch_checkpoint(...) 사용.
        - init_epoch=0, train_epochs 는 config 값 그대로 사용.
        - wandb_id=None, allow_val_change=False 로 처리한다.

    Args:
        args (argparse.Namespace):
            학습 설정 및 resume 관련 인자를 담고 있는 Namespace.
        diffusion_planner (nn.Module):
            현재 학습 중인 모델 또는 DeepSpeed/ DDP 래퍼.
        optimizer (optim.Optimizer):
            AdamW 또는 8bit AdamW 옵티마이저.
        scheduler (Any):
            학습률 스케줄러 인스턴스.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 주요 로그를 출력한다.
        use_deepspeed (bool):
            DeepSpeed 모드 사용 여부.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
              Optional[str], int, bool]:
            - diffusion_planner:
                체크포인트를 반영한 모델(DeepSpeed/ DDP 래퍼 포함 가능).
            - optimizer:
                재개된 옵티마이저 또는 기존 옵티마이저.
            - scheduler:
                재개된 스케줄러 또는 기존 스케줄러.
            - model_ema:
                EMA 상태가 복원된 ModelEma 또는 None.
            - init_epoch:
                재개 시작 epoch 인덱스(0 기반). model-only 모드에서는 0.
            - wandb_id:
                이어서 사용할 W&B run id 또는 None.
            - train_epochs:
                최종 train_epochs 값.
            - allow_val_change:
                W&B config 값을 바꿔도 되는지 여부.
    """

    if args.resume_wandb_model_name is not None:
        print(f"[LOAD MODEL PARAM] Model loaded from {args.save_path}")

        # -------- DeepSpeed 경로 --------
        if use_deepspeed and hasattr(diffusion_planner, "load_checkpoint"):
            diffusion_planner, model_ema, init_epoch, wandb_id = \
                _resume_from_checkpoint_with_deepspeed(
                    args=args,
                    diffusion_planner=diffusion_planner,
                    model_ema=model_ema,
                    global_rank=global_rank,
                    resume_model_only=args.resume_model_only,
                )

        # -------- 일반 PyTorch / DDP 경로 --------
        else:
            (diffusion_planner, optimizer, scheduler, model_ema, init_epoch,
             wandb_id) = _resume_from_checkpoint_with_pytorch(
                 args=args,
                 diffusion_planner=diffusion_planner,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 model_ema=model_ema,
                 save_path=args.save_path,
                 global_rank=global_rank,
                 resume_model_only=args.resume_model_only,
             )

        # -------- train_epochs / allow_val_change 최종 결정 --------
        train_epochs, allow_val_change = _finalize_train_epochs_and_wandb_after_resume(
            args=args,
            init_epoch=init_epoch,
            resume_model_only=args.resume_model_only,
            global_rank=global_rank,
        )

    else:
        # 체크포인트 지정이 전혀 없는 경우: 완전 새 학습
        init_epoch = 0
        wandb_id = None
        train_epochs = int(args.train_epochs)
        allow_val_change = False

    return (
        diffusion_planner,
        optimizer,
        scheduler,
        model_ema,
        init_epoch,
        wandb_id,
        train_epochs,
        allow_val_change,
    )


def _setup_logger_and_purge(
    args: argparse.Namespace,
    global_rank: int,
    wandb_id: Optional[str],
    allow_val_change: bool,
) -> Logger:
    """TensorBoard / W&B 로거를 만들고, 기존 아티팩트를 정리한다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        global_rank: 전체 프로세스 기준 번호.
        wandb_id: 재개 시 사용할 wandb run id.
        allow_val_change: wandb config 값 변경 허용 여부.

    Returns:
        wandb_logger: TensorBoardLogger 래퍼.
    """
    wandb_logger = Logger(
        args.name,
        args.notes,
        args,
        wandb_resume_id=wandb_id,
        save_path=args.save_path,
        rank=global_rank,
        allow_val_change=allow_val_change,
    )

    if global_rank == 0 and args.remove_existing_wb_weight:
        api = wandb.Api()
        entity = wandb.run.entity
        project = wandb.run.project

        purge_collection(api, entity, project, f"{args.name}_latest-model")
        purge_collection(api, entity, project, f"{args.name}_best-model")

    if args.ddp:
        torch.distributed.barrier()

    return wandb_logger


def _train_one_epoch(
    epoch: int,
    train_epochs: int,
    train_loader: DataLoader,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    aug: Optional[object],
) -> Tuple[Dict[str, float], float, float]:
    """하나의 epoch 동안 학습을 실행하고, 손실과 걸린 시간을 계산한다.

    Args:
        epoch: 현재 epoch 인덱스(0 기반).
        train_epochs: 전체 epoch 수.
        train_loader: 학습용 DataLoader. 배치 텐서는 (B, ·) shape.
        diffusion_planner: 학습 중인 모델(nn.Module 또는 DDP/DeepSpeed 래퍼).
        optimizer: 옵티마이저.
        scheduler: 학습률 스케줄러.
        args: 학습 설정.
        model_ema: EMA 래퍼 또는 None.
        aug: augmentation 객체(StatePerturbation/NPCStatePerturbation 등) 또는 None.

    Returns:
        train_loss: key별 epoch 평균 손실 딕셔너리. 각 값은 scalar float.
        train_total_loss: 최종 합 손실 scalar float.
        epoch_elapsed_time_sec: 해당 epoch에 걸린 시간(초).
    """
    # train_epoch 안에서 각 step 배치 텐서는 대략
    #   ego_agent_past: (B, T_past, 11)
    #   neighbor_agents_past: (B, A, T_past, 11)
    #   lanes: (B, L, lane_len, 12)
    #   static_objects: (B, S, 10)
    #   neighbor_future_gt_3_dim: (B, A, T_future, 3)
    # 와 같은 shape로 사용된다.
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{train_epochs}")

    epoch_t0 = time.perf_counter()
    """ train_loss: Dict[str, float]
    <diffusion_loss_func 가 출력해주는 loss_dict>
        - neighbor_prediction_loss : 이웃 예측 손실 텐서
        - integration_loss : 통합 손실 텐서
        - constraint_loss : 제약 손실 텐서

        - neighbor_prediction_loss_xy / integration_loss_xy
        - neighbor_prediction_loss_yaw / integration_loss_yaw
        - neighbor_prediction_loss_xy_early / integration_loss_xy_early
        - neighbor_prediction_loss_yaw_early / integration_loss_yaw_early
        - constraint_diff_vx_b / constraint_diff_vy_b / constraint_diff_yaw_rate

    <diffusion_loss_func 출력 후 _compute_loss_dict 가 추가하는 항목>
    - learn_progress / direct_loss_weight / int_loss_weight / const_loss_weight :
        진행도 및 가중치 기록용 텐서
    - loss_dict : 최종 합 손실 텐서.
    """

    train_loss, train_total_loss = train_epoch(
        train_loader,
        diffusion_planner,
        optimizer,
        args,
        model_ema,
        scheduler,
        aug,
    )

    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()

    if args.ddp:
        torch.cuda.synchronize()
        torch.distributed.barrier()

    epoch_elapsed_time_sec = time.perf_counter() - epoch_t0
    return train_loss, train_total_loss, epoch_elapsed_time_sec


TEMP_WANDB_ROOT_DIR: str = "/mnt/temp_wandb"


def _copy_file_to_temp_wandb(
    src_file_path: str,
    temp_root_dir: str,
    dst_file_name: Optional[str] = None,
) -> Optional[str]:
    """W&B 업로드 전에 원본 파일을 로컬 임시 디렉터리로 복사한다.

    Ceph(/mnt/nuplan) 경로에 있는 파일이라도
    /mnt/temp_wandb 아래로 복사한 뒤,
    W&B에는 이 로컬 경로만 넘기기 위한 용도다.

    Args:
        src_file_path (str):
            원본 파일 경로.
        temp_root_dir (str):
            로컬 임시 루트 디렉터리 경로. 예: "/mnt/temp_wandb/latest".
        dst_file_name (Optional[str]):
            임시 디렉터리 안에서 사용할 파일 이름.
            None이면 src_file_path의 basename을 그대로 사용한다.

    Returns:
        Optional[str]:
            복사된 로컬 파일의 전체 경로.
            복사 중 오류가 나면 None을 반환하고, 호출 측에서 건너뛰도록 한다.
    """
    if not os.path.exists(src_file_path):
        print(f"[W&B TEMP] source file not found, skip copy: {src_file_path}")
        return None

    os.makedirs(temp_root_dir, exist_ok=True)

    if dst_file_name is None:
        dst_file_name = os.path.basename(src_file_path)

    dst_file_path: str = os.path.join(temp_root_dir, dst_file_name)

    try:
        shutil.copy2(src_file_path, dst_file_path)
        return dst_file_path
    except Exception as copy_err:
        print(
            f"[W&B TEMP] file copy failed: {src_file_path} -> {dst_file_path}, "
            f"error={copy_err}")
        return None


def _copy_dir_to_temp_wandb(
    src_dir_path: str,
    temp_root_dir: str,
    dst_dir_name: Optional[str] = None,
) -> Optional[str]:
    """W&B 업로드 전에 원본 디렉터리를 로컬 임시 디렉터리로 복사한다.

    DeepSpeed 체크포인트 디렉터리 같이
    여러 파일로 이루어진 폴더를 그대로 W&B에 올리고 싶지만,
    Ceph 경로를 직접 넘기고 싶지 않을 때 사용한다.

    Args:
        src_dir_path (str):
            원본 디렉터리 경로.
        temp_root_dir (str):
            로컬 임시 루트 디렉터리 경로. 예: "/mnt/temp_wandb/latest".
        dst_dir_name (Optional[str]):
            임시 디렉터리 안에서 사용할 디렉터리 이름.
            None이면 src_dir_path의 basename을 그대로 사용한다.

    Returns:
        Optional[str]:
            복사된 로컬 디렉터리의 전체 경로.
            복사 중 오류가 나면 None을 반환하고, 호출 측에서 건너뛰도록 한다.
    """
    if not os.path.isdir(src_dir_path):
        print(f"[W&B TEMP] source dir not found, skip copy: {src_dir_path}")
        return None

    os.makedirs(temp_root_dir, exist_ok=True)

    if dst_dir_name is None:
        dst_dir_name = os.path.basename(src_dir_path)

    dst_dir_path: str = os.path.join(temp_root_dir, dst_dir_name)

    try:
        # 이미 있으면 깨끗하게 지우고 다시 만든다.
        if os.path.exists(dst_dir_path):
            shutil.rmtree(dst_dir_path)
        shutil.copytree(src_dir_path, dst_dir_path)
        return dst_dir_path
    except Exception as copy_err:
        print(f"[W&B TEMP] dir copy failed: {src_dir_path} -> {dst_dir_path}, "
              f"error={copy_err}")
        return None


def _log_wandb_checkpoint_artifacts(
    args: argparse.Namespace,
    epoch: int,
    train_total_loss: float,
    save_best: bool,
    use_deepspeed: bool,
    tag_latest: Optional[str],
    tag_best: Optional[str],
) -> None:
    """한 epoch가 끝난 뒤 local 체크포인트를 W&B 아티팩트로 올린다.

    - PyTorch/DDP 학습: latest.pth / best.pth 파일만 업로드
    - DeepSpeed 학습: latest.pth / best.pth + DeepSpeed tag 디렉터리도 함께 업로드

    주의:
    - alias는 DeepSpeed 여부와 무관하게 항상 ["latest"] / ["best"] 로만 붙인다.
    - epoch 정보는 metadata에 넣으면 충분하므로 alias에 epoch를 포함하지 않는다.
    """
    if args.save_path is None:
        return
    if not getattr(args, "use_wandb", False):
        return
    if not wandb.run:
        return

    temp_root_dir: str = TEMP_WANDB_ROOT_DIR
    try:
        if os.path.isdir(temp_root_dir):
            shutil.rmtree(temp_root_dir, ignore_errors=True)
        os.makedirs(temp_root_dir, exist_ok=True)
    except Exception as e:
        print(f"[W&B TEMP] failed to prepare temp root dir '{temp_root_dir}', "
              f"fallback to direct upload. error={e}")
        temp_root_dir = ""

    base_dir_name = os.path.basename(os.path.normpath(args.save_path))
    time_str_meta = base_dir_name.replace(":", "-")

    try:
        # ── latest-model ─────────────────────────────────────────
        latest_coll = f"{args.name}_latest-model"
        latest_art = wandb.Artifact(
            name=latest_coll,
            type="model",
            metadata={
                "time_str": time_str_meta,
                "epoch": epoch + 1,
                "loss": float(train_total_loss),
            },
        )

        temp_latest_root_dir: Optional[str] = (os.path.join(
            temp_root_dir, "latest") if temp_root_dir else None)

        latest_pth = os.path.join(args.save_path, "latest.pth")
        if os.path.exists(latest_pth):
            if temp_latest_root_dir is not None:
                local_latest_pth = _copy_file_to_temp_wandb(
                    src_file_path=latest_pth,
                    temp_root_dir=temp_latest_root_dir,
                    dst_file_name="latest.pth",
                )
                if local_latest_pth is not None:
                    latest_art.add_file(local_latest_pth)
            else:
                latest_art.add_file(latest_pth)

        if use_deepspeed and tag_latest is not None:
            latest_tag_dir = os.path.join(args.save_path, tag_latest)
            if os.path.isdir(latest_tag_dir):
                if temp_latest_root_dir is not None:
                    local_latest_tag_dir = _copy_dir_to_temp_wandb(
                        src_dir_path=latest_tag_dir,
                        temp_root_dir=temp_latest_root_dir,
                        dst_dir_name=tag_latest,
                    )
                    if local_latest_tag_dir is not None:
                        latest_art.add_dir(local_latest_tag_dir,
                                           name=tag_latest)
                else:
                    latest_art.add_dir(latest_tag_dir, name=tag_latest)

        # ✅ alias는 항상 "latest" 하나만
        wandb.log_artifact(latest_art, aliases=["latest"])
        latest_art.wait()

        if getattr(args, "delete_wb_weight_when_running", False):
            _prune_old_wandb_artifact_versions(
                collection_name=latest_coll,
                alias="latest",
            )

        # ── best-model ───────────────────────────────────────────
        if not save_best:
            return

        best_coll = f"{args.name}_best-model"
        best_art = wandb.Artifact(
            name=best_coll,
            type="model",
            metadata={
                "time_str": time_str_meta,
                "epoch": epoch + 1,
                "loss": float(train_total_loss),
            },
        )

        temp_best_root_dir: Optional[str] = (os.path.join(
            temp_root_dir, "best") if temp_root_dir else None)

        best_pth = os.path.join(args.save_path, "best.pth")
        if os.path.exists(best_pth):
            if temp_best_root_dir is not None:
                local_best_pth = _copy_file_to_temp_wandb(
                    src_file_path=best_pth,
                    temp_root_dir=temp_best_root_dir,
                    dst_file_name="best.pth",
                )
                if local_best_pth is not None:
                    best_art.add_file(local_best_pth)
            else:
                best_art.add_file(best_pth)

        if use_deepspeed and tag_best is not None:
            best_tag_dir = os.path.join(args.save_path, tag_best)
            if os.path.isdir(best_tag_dir):
                if temp_best_root_dir is not None:
                    local_best_tag_dir = _copy_dir_to_temp_wandb(
                        src_dir_path=best_tag_dir,
                        temp_root_dir=temp_best_root_dir,
                        dst_dir_name=tag_best,
                    )
                    if local_best_tag_dir is not None:
                        best_art.add_dir(local_best_tag_dir, name=tag_best)
                else:
                    best_art.add_dir(best_tag_dir, name=tag_best)

        # ✅ alias는 항상 "best" 하나만
        wandb.log_artifact(best_art, aliases=["best"])
        best_art.wait()

        if getattr(args, "delete_wb_weight_when_running", False):
            _prune_old_wandb_artifact_versions(
                collection_name=best_coll,
                alias="best",
            )

    finally:
        if temp_root_dir:
            try:
                shutil.rmtree(temp_root_dir, ignore_errors=True)
                print(f"[W&B TEMP] removed temp dir: {temp_root_dir}")
            except Exception as e:
                print(
                    f"[W&B TEMP] failed to remove temp dir '{temp_root_dir}': {e}"
                )


def _log_and_save(
    epoch: int,
    args: argparse.Namespace,
    train_total_loss: float,
    metrics: Dict[str, float],
    wandb_logger: Logger,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    best_loss: float,
    global_rank: int,
) -> float:
    """rank 0 프로세스에서만 로그 기록과 체크포인트 저장을 담당한다.

    전체 학습 루프(_run_training_loop) 안에서 한 epoch가 끝날 때마다 호출되며,
    다음과 같은 일을 **오직 global_rank == 0 일 때만** 수행한다.

    1) metric 로깅
       - metrics 딕셔너리의 값들을 wandb_logger.log_metrics(...) 로 한 번에 기록한다.
         · metrics[key] 값은 모두 스칼라(float)라고 가정한다.
         · 주요 값: "loss_dict/loss", "info_dict/lr", "speed_info/data_process_per_sec" 등.

    2) 저장 주기 판단
       - args.save_utd (예: 1, 5) 를 읽어서
         `(epoch + 1) % save_utd == 0` 인 epoch 에서만 체크포인트를 저장한다.
         · save_utd = 1 이면 매 epoch 저장.
         · save_utd = 5 이면 5 epoch마다 한 번 저장.

    3) best loss 갱신 여부 판단
       - train_total_loss 가 현재 best_loss 보다 작으면
         · best_loss 를 새 값으로 교체하고,
         · save_best=True 로 표시해 best.pth 도 함께 갱신하도록 만든다.

    4) local 체크포인트 저장
       - use_deepspeed=True 이면:
         · _save_deepspeed_checkpoint_for_epoch(...) 을 호출해
           DeepSpeed 포맷(latest / best tag + meta 파일)으로 저장한다.
       - use_deepspeed=False 이면:
         · diffusion_planner, optimizer, scheduler, EMA 를
           diffusion_planner.utils.train_utils.save_model(...) 로
           PyTorch latest.pth / best.pth 로 저장한다.

    5) W&B 아티팩트 업로드
       - _log_wandb_checkpoint_artifacts(...) 를 호출해
         방금 저장한 latest.pth / best.pth (및 DeepSpeed 폴더)를
         W&B 아티팩트로 올린다.

    Args:
        epoch (int):
            - 현재 epoch 인덱스(0 기반).  # shape: ()
        args (argparse.Namespace):
            - 학습 설정과 save_utd, use_deepspeed, use_wandb 등을 포함한 Namespace.
        train_total_loss (float):
            - 이번 epoch 의 전체 train loss 스칼라 값.  # shape: ()
        metrics (Dict[str, float]):
            - "info_dict/*", "loss_dict/*", "speed_info/*" 등 prefix 가 붙은
              로그용 값들을 담은 딕셔너리. 각 value 는 float 스칼라.
        wandb_logger (Logger):
            - W&B + TensorBoard 로 metrics 를 기록하는 로거.
        diffusion_planner (nn.Module):
            - 학습 중인 모델 또는 DeepSpeed/ DDP 래퍼.
        optimizer (optim.Optimizer):
            - AdamW/AdamW8bit 옵티마이저.
        scheduler (Any):
            - 학습률 스케줄러 객체. state_dict() 호출을 지원한다고 가정.
        model_ema (Optional[ModelEma]):
            - EMA 래퍼. PyTorch 체크포인트 저장 시 ema.ema.state_dict() 도 같이 저장한다.
        best_loss (float):
            - 지금까지의 최소 train_total_loss 값.  # shape: ()
        global_rank (int):
            - 전체 프로세스 기준 rank. 0 이 아닐 경우 이 함수는 아무 것도 하지 않고
              best_loss 를 그대로 돌려준다.

    Returns:
        float:
            - 업데이트된 best_loss 값.
              · 이번 epoch 손실이 더 작으면 새 값으로 교체된 값.
              · 그렇지 않으면 입력과 동일한 값.
    """
    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    if (not use_deepspeed) and (global_rank != 0):
        print(f"[Rank {global_rank}] Skipping logging and saving.\n")
        return best_loss

    # 1) 메트릭 로그
    if global_rank == 0:
        wandb_logger.log_metrics(metrics, step=epoch + 1)

    # 2) 저장 주기 확인 (DeepSpeed / PyTorch 공통)
    save_interval: int = max(1, int(getattr(args, "save_utd", 1)))
    is_first_epoch: bool = (epoch == 0)

    # 첫 epoch(=epoch 0)은 무조건 저장, 그 이후에는 save_utd 주기로 저장
    if (not is_first_epoch) and ((epoch + 1) % save_interval != 0):
        print("Skipping checkpoint save for this epoch.\n")
        return best_loss

    # 3) best 갱신 여부
    save_best = False
    if train_total_loss < best_loss:
        best_loss = train_total_loss
        save_best = True

    # 4) local 체크포인트 저장
    if use_deepspeed:
        tag_latest, tag_best = _save_deepspeed_checkpoint_for_epoch(
            diffusion_planner=diffusion_planner,
            save_path=args.save_path,
            epoch=epoch,
            train_total_loss=train_total_loss,
            wandb_run_id=wandb_logger.id,
            model_ema=model_ema,
            use_deepspeed=use_deepspeed,
            save_best=save_best,
            global_rank=global_rank,
        )
        print(f"[DeepSpeed] Checkpoint saved in {args.save_path}\n")
    else:
        tag_latest = tag_best = None
        save_model(
            diffusion_planner,
            optimizer,
            scheduler,
            args.save_path,
            epoch,
            train_total_loss,
            wandb_logger.id,
            model_ema.ema if model_ema is not None else None,
            save_best,
        )
        if global_rank == 0:
            print(f"Model saved in {args.save_path}\n")
    # 5) W&B 아티팩트 업로드
    if global_rank == 0:
        _log_wandb_checkpoint_artifacts(args=args,
                                        epoch=epoch,
                                        train_total_loss=train_total_loss,
                                        save_best=save_best,
                                        use_deepspeed=use_deepspeed,
                                        tag_latest=tag_latest,
                                        tag_best=tag_best)

    return best_loss


def _build_epoch_info_dict_for_logging(
    optimizer: optim.Optimizer,
    train_epochs: int,
    train_loader: DataLoader,
    global_batch_size: int,
) -> Dict[str, float]:
    """한 epoch에 대한 기본 정보(info_dict)를 만든다.

    포함되는 값:
      - lr: 현재 학습률. optimizer.param_groups[0]["lr"]  # shape: ()
      - total_train_epochs: 전체 epoch 수.  # shape: ()
      - batch_num_in_epoch: 이번 epoch의 배치 개수(len(train_loader)).  # shape: ()
      - total_batch_num_of_all_epochs: 전체 학습 동안의 총 배치 수.  # shape: ()
      - global_batch_size: 한 step에서 처리되는 샘플 수.  # shape: ()

    Args:
        optimizer (optim.Optimizer):
            - AdamW 또는 8bit AdamW 옵티마이저.
        train_epochs (int):
            - 전체 학습 epoch 수.  # shape: ()
        train_loader (DataLoader):
            - 한 epoch 동안 순회할 DataLoader. len(train_loader)는 배치 수.  # shape: ()
        global_batch_size (int):
            - 실제 글로벌 배치 크기.  # shape: ()

    Returns:
        Dict[str, float]:
            - info_dict용 key/value 쌍. 값들은 모두 스칼라(float로 해석 가능한 숫자).
    """
    info_dict: Dict[str, float] = {
        "lr": optimizer.param_groups[0]["lr"],
        "total_train_epochs": train_epochs,
        "batch_num_in_epoch": len(train_loader),
        "total_batch_num_of_all_epochs": train_epochs * len(train_loader),
        "global_batch_size": global_batch_size,
    }
    return info_dict


def _split_train_loss_for_logging(
    train_loss: Dict[str, float],
    info_dict: Dict[str, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[
        str, float], Dict[str, float]]:
    """train_loss dict를 로그용 하위 딕셔너리들로 나눈다.

    입력:
      - train_loss:
          diffusion_loss_func + _compute_loss_dict 에서 나온 손실 항목 모음.
          각 value는 스칼라 텐서 .item() 결과 또는 float 값이다.  # shape: ()

    처리 규칙:
      - "learn_progress" 항목은 info_dict 에 그대로 추가.
      - 가중치 관련 키는 weight_dict 로,
      - direct loss / integration loss / constraint loss / 전체 loss 계열은
        각각 별도 dict로 분리한다.

    Args:
        train_loss (Dict[str, float]):
            - 손실 항목 이름 → 값(스칼라) 딕셔너리.
        info_dict (Dict[str, float]):
            - 기본 info_dict. "learn_progress" 키가 있으면 여기에 추가된다.

    Returns:
        Tuple[Dict[str, float], Dict[str, float], Dict[str, float],
              Dict[str, float], Dict[str, float]]:
            - weight_dict: direct/int/const loss weight 항목들.
            - direct_loss_dict: neighbor_prediction_* direct loss 항목들.
            - integration_loss_dict: integration_* loss 항목들.
            - constraint_loss_dict: constraint_* loss 항목들.
            - loss_dict: neighbor/integration/constraint/loss_dict 합산용 항목들.
    """
    weight_dict: Dict[str, float] = {}
    direct_loss_dict: Dict[str, float] = {}
    integration_loss_dict: Dict[str, float] = {}
    constraint_loss_dict: Dict[str, float] = {}
    loss_dict: Dict[str, float] = {}

    for k, v in train_loss.items():
        if k == "learn_progress":
            info_dict[k] = v
        elif k in ("direct_loss_weight", "int_loss_weight",
                   "const_loss_weight"):
            weight_dict[k] = v
        elif k in (
                "neighbor_prediction_loss_xy",
                "neighbor_prediction_loss_yaw",
                "neighbor_prediction_loss_xy_early",
                "neighbor_prediction_loss_yaw_early",
        ):
            direct_loss_dict[k] = v
        elif k in (
                "integration_loss_xy",
                "integration_loss_yaw",
                "integration_loss_xy_early",
                "integration_loss_yaw_early",
        ):
            integration_loss_dict[k] = v
        elif k in (
                "constraint_diff_vx_b",
                "constraint_diff_vy_b",
                "constraint_diff_yaw_rate",
        ):
            constraint_loss_dict[k] = v
        elif k in (
                "loss",
                "neighbor_prediction_loss",
                "integration_loss",
                "constraint_loss",
        ):
            loss_dict[k] = v

    return weight_dict, direct_loss_dict, integration_loss_dict, constraint_loss_dict, loss_dict


def _build_speed_info_for_logging(
    epoch_elapsed_time_sec: float,
    data_num_in_a_epoch: int,
    elapsed_training_time_hour: float,
) -> Dict[str, float]:
    """속도/시간 관련 로그용 딕셔너리를 만든다.

    계산 내용:
      - epoch_elapsed_time_sec: 이번 epoch에 걸린 시간(초).  # shape: ()
      - data_process_per_sec: 초당 처리 샘플 수
            = data_num_in_a_epoch / max(epoch_elapsed_time_sec, 1e-9).  # shape: ()
      - elapsed_training_time_hour: 지금까지 누적 학습 시간(시간 단위).  # shape: ()

    Args:
        epoch_elapsed_time_sec (float):
            - 이번 epoch 에 걸린 시간(초).  # shape: ()
        data_num_in_a_epoch (int):
            - 이번 epoch에서 처리된 샘플 수.  # shape: ()
        elapsed_training_time_hour (float):
            - 이전 epoch까지의 누적 학습 시간(시간 단위).  # shape: ()

    Returns:
        Dict[str, float]:
            - speed_info 딕셔너리. 각 값은 스칼라 float.
    """
    data_process_per_sec: float = data_num_in_a_epoch / max(
        epoch_elapsed_time_sec, 1e-9)
    speed_info: Dict[str, float] = {
        "epoch_elapsed_time_sec": epoch_elapsed_time_sec,
        "data_process_per_sec": data_process_per_sec,
        "elapsed_training_time_hour": elapsed_training_time_hour,
    }
    return speed_info


def _build_flat_metrics_dict_for_logging(
    info_dict: Dict[str, float],
    weight_dict: Dict[str, float],
    direct_loss_dict: Dict[str, float],
    integration_loss_dict: Dict[str, float],
    constraint_loss_dict: Dict[str, float],
    loss_dict: Dict[str, float],
    speed_info: Dict[str, float],
) -> Dict[str, float]:
    """여러 그룹의 딕셔너리를 prefix를 붙인 단일 metrics dict로 합친다.

    입력 딕셔너리들의 값은 모두 스칼라(float)라고 가정하고,
    key 앞에는 아래와 같은 prefix를 붙여 평탄한(flat) 형태로 만든다.

      - "info_dict/"       + key
      - "weight_dict/"     + key
      - "direct_loss_dict/"+ key
      - "integration_loss_dict/" + key
      - "constraint_loss_dict/"  + key
      - "loss_dict/"       + key
      - "speed_info/"      + key

    Args:
        info_dict (Dict[str, float]):
            - lr, epoch 정보, 진행도 등 정보성 값.
        weight_dict (Dict[str, float]):
            - direct/int/const loss weight 들.
        direct_loss_dict (Dict[str, float]):
            - neighbor_prediction_* 계열 direct loss 값들.
        integration_loss_dict (Dict[str, float]):
            - integration_* 계열 loss 값들.
        constraint_loss_dict (Dict[str, float]):
            - constraint_* 계열 loss 값들.
        loss_dict (Dict[str, float]):
            - neighbor_prediction_loss / integration_loss / constraint_loss / loss_dict 등.
        speed_info (Dict[str, float]):
            - epoch_elapsed_time_sec / data_process_per_sec / elapsed_training_time_hour.

    Returns:
        Dict[str, float]:
            - W&B / TensorBoard 에 바로 쓸 수 있는 flat metrics dict.
              각 value는 스칼라 float.
    """
    metrics: Dict[str, float] = {}

    # add "info_dict/" prefix
    metrics.update({f"info_dict/{k}": v for k, v in info_dict.items()})
    # add "weight_dict/" prefix
    metrics.update({f"weight_dict/{k}": v for k, v in weight_dict.items()})
    # add "direct_loss_dict/" prefix
    metrics.update({
        f"direct_loss_dict/{k}": v for k, v in direct_loss_dict.items()
    })
    # add "integration_loss_dict/" prefix
    metrics.update({
        f"integration_loss_dict/{k}": v
        for k, v in integration_loss_dict.items()
    })
    # add "constraint_loss_dict/" prefix
    metrics.update({
        f"constraint_loss_dict/{k}": v for k, v in constraint_loss_dict.items()
    })
    # add "loss_dict/" prefix
    metrics.update({f"loss_dict/{k}": v for k, v in loss_dict.items()})
    # add "speed_info/" prefix
    metrics.update({f"speed_info/{k}": v for k, v in speed_info.items()})

    return metrics


def _run_training_loop(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    wandb_logger: Logger,
    best_loss: float,
    global_rank: int,
    train_epochs: int,
    init_epoch: int,
    data_num_in_a_epoch: int,
    global_batch_size: int,
    aug: Optional[object],
) -> float:
    """전체 epoch 루프를 돌면서 학습, 속도 측정, 로깅, 체크포인트 저장을 수행한다."""
    elapsed_training_time_hour: float = 0.0

    for epoch in range(init_epoch, train_epochs):
        # ✅ (중요) epoch 시작 전에 sampler epoch를 먼저 세팅
        # - resume(init_epoch>0) 시에도 첫 epoch부터 올바른 shuffle이 나오도록 함
        train_sampler.set_epoch(epoch + args.sampler_epoch_offset)

        train_loss, train_total_loss, epoch_elapsed_time_sec = _train_one_epoch(
            epoch=epoch,
            train_epochs=train_epochs,
            train_loader=train_loader,
            diffusion_planner=diffusion_planner,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            model_ema=model_ema,
            aug=aug,
        )

        elapsed_training_time_hour += epoch_elapsed_time_sec / 3600.0

        speed_info: Dict[str, float] = _build_speed_info_for_logging(
            epoch_elapsed_time_sec=epoch_elapsed_time_sec,
            data_num_in_a_epoch=data_num_in_a_epoch,
            elapsed_training_time_hour=elapsed_training_time_hour,
        )

        info_dict: Dict[str, float] = _build_epoch_info_dict_for_logging(
            optimizer=optimizer,
            train_epochs=train_epochs,
            train_loader=train_loader,
            global_batch_size=global_batch_size,
        )

        (
            weight_dict,
            direct_loss_dict,
            integration_loss_dict,
            constraint_loss_dict,
            loss_dict,
        ) = _split_train_loss_for_logging(
            train_loss=train_loss,
            info_dict=info_dict,
        )

        metrics: Dict[str, float] = _build_flat_metrics_dict_for_logging(
            info_dict=info_dict,
            weight_dict=weight_dict,
            direct_loss_dict=direct_loss_dict,
            integration_loss_dict=integration_loss_dict,
            constraint_loss_dict=constraint_loss_dict,
            loss_dict=loss_dict,
            speed_info=speed_info,
        )

        best_loss = _log_and_save(
            epoch=epoch,
            args=args,
            train_total_loss=train_total_loss,
            metrics=metrics,
            wandb_logger=wandb_logger,
            diffusion_planner=diffusion_planner,
            optimizer=optimizer,
            scheduler=scheduler,
            model_ema=model_ema,
            best_loss=best_loss,
            global_rank=global_rank,
        )

    return best_loss


def _finalize_training_cleanup(
    args: argparse.Namespace,
    global_rank: int,
    wandb_logger: Logger,
) -> None:
    """학습이 끝난 뒤, 분산 동기화와 wandb/TensorBoard/local 파일 정리를 수행한다."""
    # 1) 분산 학습일 때만 barrier 호출
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # 2) W&B / TensorBoard 종료
    if args.use_wandb and wandb.run is not None:
        wandb.finish()
    if global_rank == 0:
        wandb_logger.finish()

    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # 3) local 체크포인트/로그 삭제는 옵션으로만 수행
    if (global_rank == 0 and args.save_path and
            getattr(args, "cleanup_local_artifacts_after_train", False)):
        print("[CLEANUP] 훈련 종료 후 local 체크포인트 및 로그 정리 시작")
        for f in ["latest.pth", "best.pth", "args.json"]:
            p = os.path.join(args.save_path, f)
            try:
                os.remove(p)
                print(f"[CLEANUP] 파일 삭제: {p}")
            except FileNotFoundError:
                print(f"[CLEANUP] 파일 없음 (이미 삭제됨): {p}")
            except Exception as e:
                print(f"[CLEANUP] 파일 삭제 오류: {p}, {e}")

        tb_dir = os.path.join(args.save_path, "tb")
        try:
            shutil.rmtree(tb_dir)
            print(f"[CLEANUP] 디렉터리 삭제: {tb_dir}")
        except FileNotFoundError:
            print(f"[CLEANUP] 디렉터리 없음 (이미 삭제됨): {tb_dir}")
        except Exception as e:
            print(f"[CLEANUP] 디렉터리 삭제 오류: {tb_dir}, {e}")


def _init_or_restore_global_update_step(
    args: argparse.Namespace,
    init_epoch: int,
    total_step_of_this_epoch: int,
) -> None:
    """전역 스텝 카운터(_global_update_step)를 초기화하거나 재설정한다.

    w_dir / w_int / w_const 같은 가중치 스케줄은
    args._global_update_step 값을 기준으로 진행도를 계산한다.
    이 함수는
      - 처음 학습을 시작할 때는 0에서 시작하고,
      - 체크포인트에서 재개할 때는 이미 끝낸 epoch 수만큼
        스텝을 미리 더해 둔다.

    Args:
        args: 학습 설정/상태 Namespace. 내부에 `_global_update_step` 속성을 추가한다.
        init_epoch: 이번 run 이 시작될 epoch 인덱스(0 기준).
        total_step_of_this_epoch: 한 epoch 안의 배치 수. 스칼라 정수.  # shape: ()

    Returns:
        None
    """
    # 이미 외부에서 명시적으로 세팅한 값이 있으면 건드리지 않는다.
    if hasattr(args, "_global_update_step"):
        return

    if init_epoch <= 0:
        # 처음 학습 시작일 때
        args._global_update_step = 0
    else:
        # 예: init_epoch=5, total_step=100 이면 500 step 만큼 이미 지난 것으로 본다.
        args._global_update_step = int(init_epoch) * int(
            total_step_of_this_epoch)


def model_training(
    args: argparse.Namespace,
    global_rank: int,
    rank: int,
    world_size: int,
    use_deepspeed: bool,
) -> None:
    """전체 학습 파이프라인을 실행하는 상위 함수."""
    best_loss: float = float("inf")
    torch.cuda.empty_cache()

    # 1) 글로벌 배치 크기에 따라 학습률 / epoch 수 스케일링
    BASE_GLOBAL_BATCH, current_global_batch = _scale_learning_rate_and_epochs(
        args,
        world_size,
    )

    # 2) seed 고정
    set_seed(args.seed + global_rank)

    # 3) augmentation, Dataset, Sampler
    batch_size = args.batch_size
    aug = _build_augmentation(args)
    _ = aug

    train_set, train_sampler = _build_dataset_and_sampler(
        args,
        world_size,
        global_rank,
    )

    # 4) warmup step 계산
    warmup_steps_at_B0, warmup_steps = _compute_warmup_steps(
        args=args,
        BASE_GLOBAL_BATCH=BASE_GLOBAL_BATCH,
        current_global_batch=current_global_batch,
        total_data_num=len(train_set),
    )

    # 5) DataLoader 및 step/샘플 수 정보
    train_loader = _build_train_loader(
        args,
        train_set,
        train_sampler,
        batch_size,
        world_size,
    )
    (total_step_of_this_epoch, total_step_of_all_epoch, global_batch_size,
     data_num_in_a_epoch) = _compute_schedule_info(
         args=args,
         train_loader=train_loader,
         world_size=world_size,
     )

    _print_schedule_summary(
        global_rank=global_rank,
        train_set_len=len(train_set),
        BASE_GLOBAL_BATCH=BASE_GLOBAL_BATCH,
        current_global_batch=current_global_batch,
        args=args,
        train_loader=train_loader,
        total_step_of_all_epoch=total_step_of_all_epoch,
        warmup_steps_at_B0=warmup_steps_at_B0,
        warmup_steps=warmup_steps,
    )

    if args.ddp and not use_deepspeed:
        torch.distributed.barrier()

    diffusion_planner, model_ema, base_model = _create_diffusion_planner_and_ema(
        args=args,
        rank=rank,
        use_deepspeed=use_deepspeed,
    )

    optimizer = _build_optimizer_with_roles_from_args(
        base_model=base_model,
        args=args,
    )

    scheduler = _build_scheduler_from_args(
        args=args,
        optimizer=optimizer,
        total_step_of_all_epoch=total_step_of_all_epoch,
        warmup_steps=warmup_steps,
    )

    if use_deepspeed:
        import deepspeed
        ds_config = build_deepspeed_config(args, current_global_batch)
        diffusion_planner, optimizer, _, scheduler = deepspeed.initialize(
            model=diffusion_planner,
            model_parameters=base_model.parameters(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )

    # 6) 체크포인트 재개
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id,
     train_epochs, allow_val_change) = _maybe_resume_from_checkpoint(
         args=args,
         diffusion_planner=diffusion_planner,
         optimizer=optimizer,
         scheduler=scheduler,
         model_ema=model_ema,
         global_rank=global_rank,
         use_deepspeed=use_deepspeed,
     )

    # ✅ (중요) 스케일링 + resume 처리까지 끝난 "최종 args"를 args.json으로 저장
    _dump_args(args, global_rank)

    _init_or_restore_global_update_step(
        args=args,
        init_epoch=init_epoch,
        total_step_of_this_epoch=total_step_of_this_epoch,
    )

    wandb_logger = _setup_logger_and_purge(
        args=args,
        global_rank=global_rank,
        wandb_id=wandb_id,
        allow_val_change=allow_val_change,
    )

    best_loss = _run_training_loop(
        args=args,
        diffusion_planner=diffusion_planner,
        optimizer=optimizer,
        scheduler=scheduler,
        model_ema=model_ema,
        train_loader=train_loader,
        train_sampler=train_sampler,
        wandb_logger=wandb_logger,
        best_loss=best_loss,
        global_rank=global_rank,
        train_epochs=train_epochs,
        init_epoch=init_epoch,
        data_num_in_a_epoch=data_num_in_a_epoch,
        global_batch_size=global_batch_size,
        aug=aug,
    )

    _finalize_training_cleanup(
        args=args,
        global_rank=global_rank,
        wandb_logger=wandb_logger,
    )


def _validate_wandb_resume_args(args: argparse.Namespace) -> None:
    """W&B에서 체크포인트를 받아올 때 필요한 기본 인자들을 점검한다.

    처리 내용:
      - args.resume_wandb_model_name 가 'latest' 인지 확인한다.
      - args.past_name 이 비어 있지 않은지 확인한다.
    둘 중 하나라도 조건을 만족하지 않으면 바로 ValueError 를 던져서
    잘못된 설정으로 내려받기를 시도하지 않도록 막는다.
    """
    # resume_wandb_model_name 가 "latest"가 아니면 에러를 발생시킵니다.
    if args.resume_wandb_model_name not in ['latest']:
        raise ValueError("args.resume_wandb_model_name must be 'latest.")
    if not args.past_name:
        raise ValueError(
            "args.past_name must be provided to resume from a wandb artifact.")


def _determine_wandb_artifact_config(
    args: argparse.Namespace,) -> Tuple[str, str, str]:
    """재개하려는 W&B 아티팩트의 이름과 파일 이름을 결정한다.

    Args:
        args: argparse.Namespace
            - args.resume_wandb_model_name (str): 'latest' 또는 'best' 등을 기대.
            - args.past_name (str): 실험 이름. 컬렉션 이름의 접두사로 사용된다.

    Returns:
        resume_alias: str
            - 실제로 사용할 별칭. 예: 'latest', 'best'.
        collection_name: str
            - W&B 상에서 모델 묶음 이름. 예: f"{args.past_name}_latest-model".
        checkpoint_filename: str
            - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    resume_alias = args.resume_wandb_model_name
    if resume_alias == 'best':
        collection_name = f"{args.past_name}_best-model"
        checkpoint_filename = "best.pth"
    else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
        collection_name = f"{args.past_name}_latest-model"
        checkpoint_filename = "latest.pth"
    return resume_alias, collection_name, checkpoint_filename


def _download_wandb_checkpoint_to_local(
    args: argparse.Namespace,
    api: wandb.Api,
    entity: str,
    project: str,
    collection_name: str,
    resume_alias: str,
    checkpoint_filename: str,
) -> None:
    """지정한 W&B 아티팩트에서 체크포인트 파일을 내려받고, 최종 디렉터리 경로를 돌려준다."""
    # rank 0만 별도 run을 만들어서 use_artifact 호출
    rank_str = os.environ.get("RANK", "0")
    try:
        rank = int(rank_str)
    except ValueError:
        rank = 0
    """
    resume_alias : str
        - 실제로 사용할 별칭. 예: 'latest', 'best'.
    collection_name : str
        - W&B 상에서 모델 묶음 이름. 예: f"{args.past_name}_latest-model".
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    """
    entity: str 예: 'jksg01019-naver-labs'
    project: str 예: 'Diffusion-Planner'
    """
    artifact_wandb_path = f"{entity}/{project}/{collection_name}:{resume_alias}"
    print(
        f"[WANDB->local] loading from artifact_wandb_path: {artifact_wandb_path}"
    )
    artifact = api.artifact(artifact_wandb_path, type='model')

    source_run = artifact.logged_by()
    if 'save_path' not in source_run.config:
        raise ValueError("NO save path in WANDB run config")

    # "./training_log/feasible_full_time_use_vel_gpu_2_exp_A/2025-12-06-06:56:58/"
    past_save_path = source_run.config['save_path']
    print(
        f"[WANDB->local] find save_path in WANDB run config past_save_path: {past_save_path}"
    )
    if args.save_path is None:  #
        experiment_is_same = args.name == args.past_name
        assert experiment_is_same, (
            "args.save_path None -> , args.name과 args.past_name should be same."
        )
        assert args.resume_wandb_model_name is not None, (
            "args.resume_wandb_model_name must be set when resuming the same experiment."
        )
        args.save_path = past_save_path

    # target_local_ckpt_path: ./training_log/.../2025-12-06-06:56:58/latest.pth
    target_local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
    if rank == 0:
        os.makedirs(past_save_path, exist_ok=True)
        artifacts_root = os.path.join(past_save_path, "artifacts")
        artifact_dir_past = artifact.download(root=artifacts_root)
        print(
            f"[WANDB->local] {artifact_wandb_path} -> {artifact_dir_past} [download]. "
            f"FYI, past_save_path: {past_save_path}")

        # 1) checkpoint 파일을 past_save_path 루트로 복사
        """
        checkpoint_filename: 'latest.pth' 또는 'best.pth'
        src_ckpt_path_past: ./training_log/.../2025-12-06-06:56:58/artifacts/latest.pth
        target_local_ckpt_path: ./training_log/.../2025-12-06-06:56:58/latest.pth
        """

        if os.path.exists(target_local_ckpt_path):
            print(
                f"[local->local] target_local_ckpt_path '{target_local_ckpt_path}' already exist so delete and re download ."
            )
            os.remove(target_local_ckpt_path)
        src_ckpt_path_past = os.path.join(artifact_dir_past,
                                          checkpoint_filename)
        if os.path.exists(src_ckpt_path_past):
            shutil.copy2(src_ckpt_path_past, target_local_ckpt_path)
            print(
                f"[local->local] copy checkpoint: src_ckpt_path_past {src_ckpt_path_past} -> target_local_ckpt_path {target_local_ckpt_path}"
            )
        else:
            raise FileNotFoundError(
                f"[local->local] cannot find checkpoint in downloaded artifact src_ckpt_path_past: {src_ckpt_path_past}"
            )

        # 2) DeepSpeed용 latest / best 디렉터리도 있으면 같이 복사
        for tag in ("latest", "best"):
            """
            artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            artifact_tag_dir_past : ./training_log/.../2025-12-06-06:56:58/artifacts/latest/
            save_path_tag_dir : ./training_log/.../2025-12-06-06:56:58/latest/
            """
            """
            artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            안에서, 이름에 tag가 포함된 서브 폴더를 찾는다.
            예: latest_epoch-000001, best_epoch-000010 등
            """
            if not os.path.isdir(artifact_dir_past):
                continue

            # tag 문자열이 포함된 서브 디렉터리들 후보
            candidate_dirs = [
                d for d in os.listdir(artifact_dir_past) if
                os.path.isdir(os.path.join(artifact_dir_past, d)) and tag in d
            ]
            if not candidate_dirs:
                continue

            # 첫 번째(정렬 기준) 후보를 최종 tag 디렉터리 이름으로 사용
            chosen_dir_name = sorted(candidate_dirs)[0]

            artifact_tag_dir_past = os.path.join(artifact_dir_past,
                                                 chosen_dir_name)
            save_path_tag_dir = os.path.join(args.save_path, chosen_dir_name)

            if os.path.isdir(save_path_tag_dir):
                shutil.rmtree(save_path_tag_dir)
            shutil.copytree(artifact_tag_dir_past, save_path_tag_dir)
            print(
                "[local->local] COpy DeepSpeed checkpoint directory: "
                f"artifact_tag_dir_past   {artifact_tag_dir_past} -> save_path_tag_dir  {save_path_tag_dir}"
            )
        if not os.path.exists(target_local_ckpt_path):
            downloaded_files = []
            # artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            if os.path.isdir(artifact_dir_past):
                for root, dirs, files in os.walk(artifact_dir_past):
                    for name in files:
                        rel = os.path.relpath(os.path.join(root, name),
                                              artifact_dir_past)
                        downloaded_files.append(rel)
            raise FileNotFoundError(
                f"[local->local] cannot find checkpoint_filename   '{checkpoint_filename}' from downloaded artifact: "
                f"artifact_dir_past={artifact_dir_past}. example files: {downloaded_files[:10]}"
            )
    else:
        # rank 0이 다운로드해서 파일이 생길 때까지 대기
        print(
            f"rank {rank} is waiting for creating checkpoint: target_local_ckpt_path {target_local_ckpt_path}"
        )
        waited_seconds = 0
        max_wait_seconds = 600  # 10분 정도 대기 (필요하면 조정)

        while not os.path.exists(target_local_ckpt_path):
            if waited_seconds >= max_wait_seconds:
                raise TimeoutError(
                    f"rank 0이 {max_wait_seconds}second fail: target_local_ckpt_path {target_local_ckpt_path}"
                )
            time.sleep(1)
            waited_seconds += 1

        print(
            f"rank {rank} already checked downloaded checkpoint! : target_local_ckpt_path {target_local_ckpt_path}"
        )


def _get_save_path(args: argparse.Namespace, global_rank: int) -> None:
    """실험 이름과 past_name을 기준으로 save_path를 만들고, 모든 rank에 공유한다.

    동작 요약:
      * (stage1 처럼) 새로운 실험(name != past_name)인 경우
        - rank 0에서만 새로운 디렉터리를 만들고
          예: {args.save_dir}/training_log/{args.name}/{YYYY-MM-DD-HH:MM:SS}/
        - 그 경로를 torch.distributed.broadcast_object_list 로 모든 rank에 전파한다.
      * (같은 실험을 이어서 학습하는) name == past_name 인 경우
        - 여기서는 save_path를 정하지 않고(None 유지)
        - 이후 W&B 아티팩트에서 가져온 past_save_path를 args.save_path에 채운다.
      * 분산이 초기화되어 있지 않은(single GPU / single process) 경우
        - rank 0 로직만 실행되고, broadcast는 생략된다.

    Args:
        args (argparse.Namespace):
            학습 및 실험 설정이 들어 있는 argparse 인자 모음.
            - name (str): 현재 stage에서 사용할 실험 이름.
            - past_name (str): 이전 stage 실험 이름. 다른 stage의 weight를 재사용할 때 사용.
            - save_dir (str): training_log 루트 디렉터리 상위 경로.
            - save_path (Optional[str]): 이 함수 안에서 최종 로그/체크포인트 경로로 설정된다.
        global_rank (int):
            DDP에서 현재 프로세스의 global rank.
            - 0: 메인 프로세스. 디렉터리 생성 및 초기 save_path 결정 담당.
            - >0: rank 0이 만든 save_path를 broadcast로 전달받기만 한다.

    Returns:
        None:
            args.save_path를 in-place로 수정하고, 모든 rank에서 동일한 값을 갖도록 만든다.
    """
    past_name: str = getattr(args, "past_name", "")
    experiment_is_same: bool = bool(past_name) and (args.name == past_name)

    save_path: Optional[str] = None
    if global_rank == 0:
        if experiment_is_same:
            # 동일 실험 계속 학습: save_path는 나중에 W&B에서 past_save_path로 채움
            save_path = None
        else:
            time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_dir}/training_log/{args.name}/{time_str}/"
            os.makedirs(save_path, exist_ok=True)

    # DDP / 멀티 GPU에서 rank0가 만든 save_path를 모든 rank에 공유
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        obj_list = [save_path]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        save_path = obj_list[0]

    args.save_path = save_path


def _prepare_wandb_resume(args: argparse.Namespace,) -> None:
    """명령행 인자를 읽고, 필요하다면 W&B 아티팩트에서 체크포인트를 내려받아 args를 보정한다.

    처리 흐름:
      2) args.resume_wandb_model_name 가 설정되지 않았다면 그대로 반환한다.
      3) 설정된 경우
         - _validate_wandb_resume_args() 로 기본 인자 검사를 하고
         - wandb.Api() 를 만든 뒤
         - _determine_wandb_artifact_config() 으로 컬렉션/파일 이름을 정하고
         - _get_wandb_entity_and_project_for_resume() 로 entity, project 를 얻고
         - _download_wandb_checkpoint_to_local() 로 체크포인트를 내려받는다.
      4) 내려받은 디렉터리 경로를 args.save_path 에 저장해
         이후 model_training() 의 resume 로직이 그대로 동작하도록 만든다.

    Returns:
        args: argparse.Namespace
            - 체크포인트를 쓰지 않을 때는 원본 get_args() 결과와 동일.
            - W&B에서 내려받기를 한 경우에는
              args.save_path 가 실제 디렉터리 경로로 채워진 상태.
    """

    if not args.resume_wandb_model_name:
        return

    _validate_wandb_resume_args(args)
    print(
        f"Resuming from wandb artifact: {args.past_name}:{args.resume_wandb_model_name}"
    )
    """
    resume_alias : str
        - 실제로 사용할 별칭. 예: 'latest', 'best'.
    collection_name : str
        - W&B 상에서 모델 묶음 이름. 예: f"{args.name}_latest-model".
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    api = wandb.Api()
    resume_alias, collection_name, checkpoint_filename = \
        _determine_wandb_artifact_config(args)
    if getattr(args, "save_path", None) is not None:
        local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
        if os.path.exists(local_ckpt_path):
            print(
                f"[WANDB->local] found existing local checkpoint: {local_ckpt_path} "
                f"(alias={resume_alias}), skip wandb download.")
            return
    """
    entity: str 예: 'jksg01019-naver-labs'
    project: str 예: 'Diffusion-Planner'
    """
    try:
        entity = args.entity
        project = args.project
        _download_wandb_checkpoint_to_local(
            args,
            api=api,
            entity=entity,
            project=project,
            collection_name=collection_name,
            resume_alias=resume_alias,
            checkpoint_filename=checkpoint_filename,
        )

    except Exception as e:
        raise RuntimeError(f"W&B 아티팩트에서 체크포인트를 내려받는 중 오류가 발생했습니다: {e}") from e


def _set_distributed_flag_from_env(args: argparse.Namespace) -> None:
    """환경변수 WORLD_SIZE를 읽어 args.distributed 플래그를 설정한다.

    WORLD_SIZE 가 존재하면
      - WORLD_SIZE > 1 이면 여러 프로세스를 쓰는 분산 학습으로 보고 True,
      - 그렇지 않으면 False 로 둔다.
    WORLD_SIZE 가 없으면 단일 프로세스 학습으로 간주하고 False 를 넣는다.
    """
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False


def main() -> None:
    """train_predictor 진입점.

    처리 순서:
      1) 명령행 인자를 읽고(args_util.get_args), 필요 시 W&B 아티팩트에서
         체크포인트 파일(latest.pth 등)을 내려받아 args.save_path 를 채운다.
      2) 환경변수 WORLD_SIZE 를 기준으로 args.distributed 를 설정해,
         이후 ddp 설정이 올바르게 동작하도록 만든다.
      3) model_training(args) 를 호출해 전체 학습 파이프라인을 수행하고,
         예외가 발생하면 rank 정보를 찍고 전체 스택을 출력한다.
    """
    # 1) 분산 초기화 및 rank 정보
    args = args_util.get_args()
    global_rank, rank, world_size, use_deepspeed = _init_distributed(args)
    _get_save_path(
        args=args,
        global_rank=global_rank,
    )
    _prepare_wandb_resume(args)
    _set_distributed_flag_from_env(args)

    # Run
    try:
        model_training(args, global_rank, rank, world_size, use_deepspeed)
    except BaseException:
        rank = int(os.environ.get("RANK", -1))
        print(
            f"\n[rank{rank}] Unhandled exception (printing full traceback):",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()  # <-- 표준에러로 자세한 스택
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
