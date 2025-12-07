import os
from typing import Optional, Tuple
# 128 MiB 단위로 메모리 청크를 잘라서 할당하도록 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import args_util
# DDP 디버깅을 위해 사용되지 않은 파라미터 정보를 상세히 출력
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
import torch
from typing import Any, Callable, Dict

# deprecated 키는 사용 금지
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

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
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts, build_pytorch_warmup_cosine_scheduler
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
except Exception:
    pass


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
    beta: float = 1.,
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


def build_adamw_with_param_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    include_seed_params: bool = True,
    use_8bit_optimizer: bool = False,
) -> Tuple[optim.Optimizer, List[str]]:
    """AdamW 옵티마이저와 파라미터 그룹을 만드는 헬퍼 함수.

    이 함수는 모델을 훑어서
      1) 감쇠를 적용할 가중치
      2) 감쇠를 적용하지 않을 가중치(편향, 정규화 계층, 토큰/위치 임베딩 등)
    를 자동으로 나눈 뒤, 한 번에 AdamW 옵티마이저를 만들어 준다.

    Args:
        model: 학습할 모델 객체(nn.Module). 내부 파라미터 텐서의 예시는
            - 선형 계층 weight: (out_dim, in_dim)
            - 선형 계층 bias / LayerNorm weight: (dim,)
            - 컨볼루션 weight: (C_out, C_in, kH, kW) 와 같은 형태.
        lr: 기준 학습률. 각 파라미터 그룹의 "lr"와 "lr_max"에 동일하게 들어간다.
        weight_decay: 가중치 감쇠 값. 감쇠를 적용할 그룹에만 사용된다.
        include_seed_params: seeds/ego_fut_seeds 같은 학습용 토큰 파라미터를
            감쇠 제외 목록에 포함할지 여부.
        use_8bit_optimizer: True이면 8비트 AdamW(bnb.AdamW8bit)를 사용한다.

    Returns:
        Tuple[optim.Optimizer, List[str]]:
            - optim: AdamW 또는 8비트 AdamW 옵티마이저 인스턴스.
              내부 파라미터 그룹의 각 원소는
                · "params": List[Tensor]  (각 텐서 shape 예: (out_dim, in_dim), (dim,) 등)
                · "weight_decay": float
                · "lr": float
                · "lr_max": float
                · "wd_max": float
              와 같은 키를 가진다.
            - extra_nwd: 감쇠를 적용하지 않을 파라미터 이름 목록.
    """
    # extra_nwd: List[str], 감쇠 제외로 추가할 파라미터 이름들
    extra_nwd = discover_extra_no_weight_decay_names(
        model, include_seed_params=include_seed_params)

    # param_groups: List[Dict[str, Any]]
    #   각 그룹은 "params" 리스트 안에 다양한 shape의 파라미터 텐서를 가진다.
    #   예) Linear weight: (out_dim, in_dim), LayerNorm weight: (dim,), conv weight: (C_out, C_in, kH, kW)
    param_groups = param_groups_weight_decay(
        model,
        weight_decay=weight_decay,
        no_weight_decay_list=extra_nwd,
    )

    # 공통 LR / 기준값 부여
    for g in param_groups:
        g["lr"] = lr  # η_max(B): 현재 배치에서의 최대 학습률
        g["lr_max"] = float(lr)  # 스케줄 시작 시 기준 학습률
        g["wd_max"] = float(g.get("weight_decay", 0.0))  # 그룹별 기준 감쇠 값

    if use_8bit_optimizer:
        # 8비트 AdamW: 내부 모멘트 텐서는 8비트 정수로 저장되지만,
        #   파라미터 텐서 shape는 그대로 유지된다.
        optim = bnb.optim.AdamW8bit(  # type: ignore[attr-defined]
            param_groups,
            lr=lr,
            weight_decay=0.0,
        )
    else:
        # 최종 옵티마이저 (전역 WD는 0.0로 두고, 그룹별 WD만 사용)
        try:
            optim = torch.optim.AdamW(
                param_groups,
                fused=True,
                weight_decay=0.0,
            )
        except (TypeError, RuntimeError):
            optim = torch.optim.AdamW(param_groups, weight_decay=0.0)

    return optim, extra_nwd


from typing import Any, Dict, List
import torch


class DiffusionPlannerCollate:
    """DiffusionPlannerData 샘플들을 배치 텐서로 묶는 collate_fn.

    각 샘플마다 아래 상한이 적용된 상태로 .npz에 저장되어 있다.
      - agent 수   ≤ caching_max_agent_num
      - lane 수    ≤ lane_num
      - route 수   ≤ route_num
      - static 수  ≤ max_static_num

    하지만 샘플마다 실제 개수는 다를 수 있으므로, 한 배치 안에서는

        - data_max_agent_num  = max_i neighbor_agents_past_i.shape[0]
        - data_max_lane_num   = max_i lanes_i.shape[0]
        - data_max_route_num  = max_i route_lanes_i.shape[0]
        - data_max_static_num = max_i static_objects_i.shape[0]

    으로 실제 최대 길이를 구한 뒤, 그 길이에 맞춰 0으로 패딩해서 고정 shape 배치를 만든다.

    최종 shape 예:
      - neighbor_agents_past         : (B, data_max_agent_num, time_len, 11)
      - near_future_gt_3_dim         : (B, data_max_agent_num, future_len, 3)
      - static_objects               : (B, data_max_static_num, 10)
      - lanes / lanes_*              : (B, data_max_lane_num, lane_len, ·)
      - route_lanes / route_lanes_*  : (B, data_max_route_num, route_len, ·)
      - agent_route_lane_order       : (B, data_max_agent_num, data_max_lane_num)

    이때, real_* 값이 config 상한
      - caching_max_agent_num / lane_num / route_num / max_static_num
    을 초과하면 바로 예외를 발생시켜, 캐싱/학습 설정 불일치를 빨리 발견할 수 있게 한다.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.caching_max_agent_num: int = int(args.caching_max_agent_num)
        self.lane_num: int = int(args.lane_num)
        self.route_num: int = int(args.route_num)
        self.max_static_num: int = int(args.max_static_num)

    # ------------------------------------------------------------------
    # 1) 고정 shape 텐서들(이미 모든 샘플에서 shape 동일) 스택하는 도우미
    # ------------------------------------------------------------------
    def _stack_fixed(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """모든 샘플에서 shape가 같은 키를 단순히 (B, ...)로 쌓는다.

        Args:
            batch: 샘플 dict 리스트. 길이 = B.
            key:   예: "ego_agent_past".
            dtype: torch.float32 / torch.int64 / torch.bool 등.

        Returns:
            stacked: 텐서, shape = (B, *sample_shape)
        """
        # sample_shape: (T, D) 같은 뒷부분. B 차원만 앞에 새로 붙는다.
        return torch.stack(
            [torch.as_tensor(sample[key], dtype=dtype) for sample in batch],
            dim=0,
        )

    # ------------------------------------------------------------------
    # 2) (N_i, ...) 형태를 (B, target_len, ...)으로 패딩하는 도우미
    # ------------------------------------------------------------------
    def _pad_first_dim_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """(N_i, ...) 배열들을 (B, target_len, ...) 텐서로 0 패딩한다.

        Args:
            batch:
                샘플 dict 리스트. 길이 = B.
            key:
                예: "neighbor_agents_past", "lanes", "static_objects".
            target_len:
                배치 차원 뒤 첫 번째 축 길이. 예: data_max_agent_num.
            dtype:
                출력 텐서 dtype.

        Returns:
            padded:
                shape = (B, target_len, *rest_shape)
                · rest_shape 는 첫 샘플의 arr.shape[1:] 과 동일.
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
            if n_i <= 0:
                continue
            padded[b_idx, :n_i, ...] = torch.as_tensor(arr[:n_i], dtype=dtype)

        return padded

    # ------------------------------------------------------------------
    # 3) (N_i, M_i) 형태를 (B, target0, target1)으로 패딩하는 도우미
    # ------------------------------------------------------------------
    def _pad_two_dims_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len0: int,
        target_len1: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """(N_i, M_i) 배열들을 (B, target_len0, target_len1)로 패딩.

        현재는 agent_route_lane_order에만 사용됨.
        - npz 안에서는: -1 = not in route, 0.. = rank
        - 여기서 새로 생기는 패딩도 -1로 맞춰줘야 Encoder 로직과 일관됨.
        """
        batch_size: int = len(batch)
        # 🔴 기존: out = torch.zeros(...)
        # ✅ 수정: -1로 채워서 패딩 = "route 없음"으로 명시
        out = torch.full(
            (batch_size, target_len0, target_len1),
            fill_value=-1,
            dtype=dtype,
        )

        for b_idx, sample in enumerate(batch):
            arr = sample[key]
            n0 = min(arr.shape[0], target_len0)
            n1 = min(arr.shape[1], target_len1)
            if n0 <= 0 or n1 <= 0:
                continue
            out[b_idx, :n0, :n1] = torch.as_tensor(arr[:n0, :n1], dtype=dtype)

        return out

    # ------------------------------------------------------------------
    # 4) collate 본체
    # ------------------------------------------------------------------
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """단일 샘플 dict 리스트를 고정 길이 배치 텐서 dict로 변환한다.

        Args:
            batch:
                DiffusionPlannerData 에서 나온 샘플 dict 리스트. 길이 = B.

        Returns:
            Dict[str, torch.Tensor]:
                key 별로 배치 차원(B)이 앞에 붙은 텐서 묶음.
        """
        batch_size: int = len(batch)
        if batch_size == 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        # === (1) 실제 배치에서의 최대 길이(real_*) 계산 ====================
        data_max_agent_num = max(
            sample["neighbor_agents_past"].shape[0] for sample in batch)
        data_max_lane_num = max(sample["lanes"].shape[0] for sample in batch)
        data_max_route_num = max(
            sample["route_lanes"].shape[0] for sample in batch)
        data_max_static_num = max(
            sample["static_objects"].shape[0] for sample in batch)

        # === (2) config 상한과 일치하는지 빠르게 체크 ======================
        if data_max_agent_num > self.caching_max_agent_num:
            raise ValueError(
                f"배치 내 agent 수(data_max_agent_num={data_max_agent_num})가 "
                f"caching_max_agent_num={self.caching_max_agent_num} 를 초과했습니다.")
        if data_max_lane_num > self.lane_num:
            raise ValueError(
                f"배치 내 lane 수(data_max_lane_num={data_max_lane_num})가 "
                f"lane_num={self.lane_num} 를 초과했습니다.")
        if data_max_route_num > self.route_num:
            raise ValueError(
                f"배치 내 route 수(data_max_route_num={data_max_route_num})가 "
                f"route_num={self.route_num} 를 초과했습니다.")
        if data_max_static_num > self.max_static_num:
            raise ValueError(
                f"배치 내 static 수(data_max_static_num={data_max_static_num})가 "
                f"max_static_num={self.max_static_num} 를 초과했습니다.")

        # === (3) 고정 길이 텐서들 쌓기 =====================================
        ego_agent_past = self._stack_fixed(
            batch,
            key="ego_agent_past",
            dtype=torch.float32,  # (B, time_len, 11)
        )
        ego_future_gt_3_dim = self._stack_fixed(
            batch,
            key="ego_future_gt_3_dim",
            dtype=torch.float32,  # (B, future_len, 3)
        )
        planner_future_11_dim = self._stack_fixed(
            batch,
            key="planner_future_11_dim",
            dtype=torch.float32,  # (B, future_len, 11)
        )

        # === (4) agent / static / lane / route 축 패딩 ======================
        neighbor_agents_past = self._pad_first_dim_to(
            batch,
            key="neighbor_agents_past",
            target_len=data_max_agent_num,
            dtype=torch.float32,  # (B, data_max_agent_num, time_len, 11)
        )
        near_future_gt_3_dim = self._pad_first_dim_to(
            batch,
            key="near_future_gt_3_dim",
            target_len=data_max_agent_num,
            dtype=torch.float32,  # (B, data_max_agent_num, future_len, 3)
        )
        static_objects = self._pad_first_dim_to(
            batch,
            key="static_objects",
            target_len=data_max_static_num,
            dtype=torch.float32,  # (B, data_max_static_num, 10)
        )

        lanes = self._pad_first_dim_to(
            batch,
            key="lanes",
            target_len=data_max_lane_num,
            dtype=torch.float32,  # (B, data_max_lane_num, lane_len, 12)
        )
        lanes_speed_limit = self._pad_first_dim_to(
            batch,
            key="lanes_speed_limit",
            target_len=data_max_lane_num,
            dtype=torch.float32,  # (B, data_max_lane_num, 1)
        )
        lanes_has_speed_limit = self._pad_first_dim_to(
            batch,
            key="lanes_has_speed_limit",
            target_len=data_max_lane_num,
            dtype=torch.bool,  # (B, data_max_lane_num, 1)
        )

        route_lanes = self._pad_first_dim_to(
            batch,
            key="route_lanes",
            target_len=data_max_route_num,
            dtype=torch.float32,  # (B, data_max_route_num, route_len, 12)
        )
        route_lanes_speed_limit = self._pad_first_dim_to(
            batch,
            key="route_lanes_speed_limit",
            target_len=data_max_route_num,
            dtype=torch.float32,  # (B, data_max_route_num, 1)
        )
        route_lanes_has_speed_limit = self._pad_first_dim_to(
            batch,
            key="route_lanes_has_speed_limit",
            target_len=data_max_route_num,
            dtype=torch.bool,  # (B, data_max_route_num, 1)
        )

        agent_route_lane_order = self._pad_two_dims_to(
            batch,
            key="agent_route_lane_order",
            target_len0=data_max_agent_num,
            target_len1=data_max_lane_num,
            dtype=torch.int64,  # (B, data_max_agent_num, data_max_lane_num)
        )

        # === (5) dict 구성 (for문으로 깔끔하게) ============================
        batch_out: Dict[str, torch.Tensor] = {}
        for k, v in [
            ("ego_agent_past", ego_agent_past),
            ("ego_future_gt_3_dim", ego_future_gt_3_dim),
            ("neighbor_agents_past", neighbor_agents_past),
            ("lanes", lanes),
            ("lanes_speed_limit", lanes_speed_limit),
            ("lanes_has_speed_limit", lanes_has_speed_limit),
            ("route_lanes", route_lanes),
            ("route_lanes_speed_limit", route_lanes_speed_limit),
            ("route_lanes_has_speed_limit", route_lanes_has_speed_limit),
            ("static_objects", static_objects),
            ("near_future_gt_3_dim", near_future_gt_3_dim),
            ("planner_future_11_dim", planner_future_11_dim),
            ("agent_route_lane_order", agent_route_lane_order),
        ]:
            batch_out[k] = v

        return batch_out


def _init_distributed(args: argparse.Namespace,) -> Tuple[int, int, int, bool]:
    """여러 GPU를 쓸 때 필요한 분산 학습 환경을 준비하고, 내 위치 정보를 정리한다.

    이 함수는 두 가지 모드를 지원한다.
      - use_deepspeed=True  인 경우:
        torchrun 이 미리 넣어준 환경변수(RANK, WORLD_SIZE, LOCAL_RANK)를 그대로 읽어서
        · global_rank : 전체 학습 프로세스 중에서 내 번호
        · world_size  : 전체 프로세스 개수
        · rank        : 이 학습 노드(프로세스)에서 내가 사용할 GPU 번호
        를 정하고, 해당 GPU 로 장치를 고정한다.
      - use_deepspeed=False 인 경우:
        기존 ddp.ddp_setup_universal() 을 호출해서 위와 같은 값을 계산하고,
        PyTorch 기본 분산 통신 설정까지 한 번에 마친다.
    """
    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    if use_deepspeed:
        # torchrun이 설정한 환경변수만 사용해서 rank/world_size를 계산한다.
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # 단일 프로세스 fallback
            global_rank = 0
            world_size = 1
            rank = 0
        if args.device.startswith('cuda'):
            torch.cuda.set_device(rank)
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
    BASE_GLOBAL_BATCH = 2048  # 기존 기준 글로벌 배치
    BASE_LR = args.learning_rate  # 기존 기준 LR (Adam/AdamW)

    # DataLoader가 실제로 사용할 글로벌 배치(정수 배수)로 계산
    current_global_batch = _effective_global_batch(args.batch_size, world_size)
    scale = math.sqrt(current_global_batch / float(BASE_GLOBAL_BATCH))
    args.learning_rate = BASE_LR * scale

    # 전체 학습 epoch 수를 글로벌 배치에 맞게 조정
    EPOCH_BETA = 1.0
    base_epochs_anchor = int(args.train_epochs)  # 예: 360

    scaled_epochs = _auto_scale_train_epochs(
        base_global_batch=BASE_GLOBAL_BATCH,  # B_0 = 2048
        base_epochs=base_epochs_anchor,  # E_0 = 360
        current_global_batch=current_global_batch,  # B = 실제 글로벌 배치
        beta=EPOCH_BETA,
        clamp_min=max(1, args.warm_up_epoch + 1),
        clamp_max=None,
    )
    args.train_epochs = int(scaled_epochs)

    return BASE_GLOBAL_BATCH, current_global_batch


def _prepare_save_path_and_dump_args(
    args: argparse.Namespace,
    global_rank: int,
) -> Tuple[Optional[str], Optional[str]]:
    """체크포인트/로그를 저장할 경로를 만들고, args.json을 기록한다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        global_rank: 전체 프로세스 기준 번호.

    Returns:
        save_path: rank 0에서 만든 저장 경로. 나머지 rank는 None.
        time_str: 디렉터리 이름에 사용한 시간 문자열. rank 0에서만 유효.
    """
    if global_rank == 0:
        if args.resume_local_path_model_path is not None:
            # resume_local_path_model_path: ./training_log/.../2025-08-06-07:23:04/
            save_path = args.resume_local_path_model_path
            # 폴더명 자체가 타임스탬프이므로 그대로 재사용
            time_str = os.path.basename(
                os.path.normpath(args.resume_local_path_model_path))
        else:
            from datetime import datetime
            time_ = datetime.now()
            time_ = time_.strftime("%Y-%m-%d-%H:%M:%S")
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")
            save_path = f"{args.save_dir}/training_log/{args.name}/{time_}/"
            os.makedirs(save_path, exist_ok=True)

        args.save_path = save_path

        # StateNormalizer / ObservationNormalizer는 dict로 변환해서 저장
        args_dict = vars(args)
        args_dict = {
            k: (v if not isinstance(v, (StateNormalizer, ObservationNormalizer))
                else v.to_dict()) for k, v in args_dict.items()
        }

        from mmengine.fileio import dump
        dump(
            args_dict,
            os.path.join(save_path, 'args.json'),
            file_format='json',
            indent=4,
        )
    else:
        save_path = None
        args.save_path = None
        time_str = None

    return save_path, time_str


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
        args.future_len,
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

    여기서 워밍업은 “처음 몇 step 동안은 학습률을 서서히 올리다가,
    어느 시점 이후부터 본래 학습률로 쓰자”는 개념이다.
    너무 이른 단계부터 큰 학습률을 쓰면 값이 튀거나 발산하기 쉬워서,
    초반에는 조금씩 적응시키는 완충 구간을 둔다.

    계산 순서는 다음과 같다.
      1) 기준 배치 크기(BASE_GLOBAL_BATCH)에서
         · 한 epoch에 몇 step이 나오는지
         · 워밍업 epoch 수(args.warm_up_epoch)를 곱해
           “기준 환경에서의 총 워밍업 step 수(warmup_steps_at_B0)”를 정한다.
      2) 현재 글로벌 배치(current_global_batch)가 기준보다 크거나 작을 수 있으므로,
         배치 비율에 따라 워밍업 step 수도 함께 스케일링한다.
         배치가 커지면 한 epoch당 step이 줄어들기 때문에,
         워밍업을 너무 길게 또는 너무 짧게 가져가지 않도록 조정하는 목적이다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        BASE_GLOBAL_BATCH: 기준 글로벌 배치 크기.
        current_global_batch: 현재 글로벌 배치 크기.
        total_data_num: 전체 학습 샘플 수.

    Returns:
        warmup_steps_at_B0: 기준 배치 크기에서의 워밍업 step 수.
        warmup_steps: 현재 배치 크기에서의 워밍업 step 수.
    """
    # 기준 배치 B_0에서의 epoch당 step 수
    steps_per_epoch_at_B0 = math.ceil(total_data_num / float(BASE_GLOBAL_BATCH))
    warmup_steps_at_B0 = steps_per_epoch_at_B0 * args.warm_up_epoch

    batch_ratio = current_global_batch / float(BASE_GLOBAL_BATCH)
    WARMUP_SCALE_EXP = 0.5
    warmup_steps = math.ceil(warmup_steps_at_B0 * batch_ratio**WARMUP_SCALE_EXP)

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


def _build_model_optimizer_scheduler(
    args: argparse.Namespace,
    rank: int,
    use_deepspeed: bool,
    total_step_of_all_epoch: int,
    warmup_steps: int,
    current_global_batch: int,
) -> Tuple[nn.Module, Optional[ModelEma], optim.Optimizer, Any]:
    """모델, EMA, 옵티마이저, 스케줄러를 한 번에 준비한다.

    처리 흐름은 다음과 같다.
      1) Diffusion_Planner 모델을 만들고, rank 에 해당하는 GPU 또는 CPU 로 옮긴다.
      2) use_deepspeed 가 False 이고 args.ddp 가 True 이면,
         모델을 PyTorch 기본 분산 래퍼(DDP)로 감싸서
         여러 GPU가 같은 모델 복사본을 들고 gradient를 각자 가지도록 만든다.
      3) EMA를 켜면, 모델과 똑같은 구조의 복사본을 하나 더 두고
         매 step마다 천천히 따라가게 해서, 나중에 더 안정적인 결과를 평가할 수 있게 한다.
      4) build_adamw_with_param_groups 로 가중치 묶음을 나누고
         AdamW 계열 옵티마이저를 만든다. 파라미터 그룹 안의 "params" 리스트에는
         다양한 shape의 텐서들이 들어간다.
      5) build_pytorch_warmup_cosine_scheduler 로
         전체 step 수(total_step_of_all_epoch)와 warmup_steps 에 맞는
         “처음에는 천천히 올리고, 이후에는 서서히 줄이는” 학습률 스케줄을 만든다.
      6) use_deepspeed 가 True 이면,
         위에서 만든 모델·옵티마이저·스케줄러를 DeepSpeed 엔진으로 한 번 더 감싼다.
         이때 DeepSpeed는 설정 파일(ds_config)에 따라
         · 옵티마 상태를 여러 GPU에 나누어 들거나
         · 일부를 CPU 쪽으로 옮기는 등의 메모리 최적화를 대신 처리한다.
         DDP 모드에서는 이런 분할 없이, 각 GPU가 모델과 옵티마 상태를 그대로 들고 간다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        rank: 이 노드에서 사용할 GPU 인덱스.
        use_deepspeed: DeepSpeed 모드 사용 여부.
        total_step_of_all_epoch: 전체 학습 동안의 총 step 수.
        warmup_steps: 워밍업 step 수.
        current_global_batch : 현재 글로벌 배치 크기.

    Returns:
        diffusion_planner: Diffusion_Planner 또는 DeepSpeed/DDP 래핑된 모델.
        model_ema: EMA 추적용 모델 래퍼. 사용하지 않으면 None.
        optimizer: AdamW 또는 8bit AdamW 옵티마이저.
        scheduler: PyTorch warmup + cosine + hold 스케줄러.
    """
    # diffusion_planner.parameters(): 각 파라미터 텐서는
    #   (out_dim, in_dim) 또는 (dim,) 등 모델 구조에 따라 다양한 shape를 가진다.
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device ==
                                             'cuda' else args.device)

    # DeepSpeed를 쓸 때는 torch.nn.parallel.DDP 래퍼는 사용하지 않는다.
    if args.ddp and (not use_deepspeed):
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])

    model_ema: Optional[ModelEma] = None
    if args.use_ema:
        # model_ema.ema.parameters(): 원본 모델 파라미터와 같은 shape
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    # DDP를 쓴 경우에만 래퍼를 벗겨서 실제 모듈 기준으로 param_group을 만든다.
    base_model = ddp.get_model(diffusion_planner, args.ddp and
                               (not use_deepspeed))

    optimizer, extra_nwd = build_adamw_with_param_groups(
        model=base_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,  # 예: 1e-2
        include_seed_params=True,
        use_8bit_optimizer=getattr(args, "use_8bit_optimizer", False),
    )

    # param_groups[i]["params"] 리스트 안에는 다양한 shape의 파라미터 텐서들이 들어 있다.
    for pg in optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))  # 기준 LR
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))  # 기준 WD

    scheduler = build_pytorch_warmup_cosine_scheduler(
        optimizer,
        total_step_of_all_epoch,
        warmup_steps,
        eta_min=0.2 * args.learning_rate,
    )

    # ➕ [ADD] DeepSpeed 엔진으로 래핑 (ZeRO-2 설정은 build_deepspeed_config 로 생성)
    if use_deepspeed:
        import deepspeed  # use_deepspeed=True일 때만 import
        ds_config = build_deepspeed_config(args, current_global_batch)
        diffusion_planner, optimizer, _, scheduler = deepspeed.initialize(
            model=diffusion_planner,
            model_parameters=base_model.parameters(),
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=ds_config,
        )

    return diffusion_planner, model_ema, optimizer, scheduler


from typing import Optional, Dict, Any
import io


def _write_deepspeed_meta_checkpoint_files(
    save_path: Optional[str],
    epoch: int,
    train_total_loss: float,
    wandb_run_id: Optional[str],
    save_best: bool,
) -> None:
    """DeepSpeed 모드용 얇은 메타 체크포인트(latest/best)를 저장한다.

    이 파일은 실제 모델 파라미터/옵티마가 아니라,
    재개할 때 참고할 작은 정보만 담는다.

    저장 내용:
      - epoch (int): 다음 학습에 사용할 epoch 인덱스.  # shape: ()
      - loss (float): 해당 epoch 전체 손실 값.         # shape: ()
      - wandb_id (str 또는 None): 이어서 쓸 W&B run id.
      - is_deepspeed (bool): DeepSpeed에서 만든 메타 파일임을 표시.

    DeepSpeed 엔진의 실제 파라미터/옵티마/스케줄러 상태는
    engine.save_checkpoint(..., tag="latest"/"best") 에 의해 따로 저장된다.
    """
    if save_path is None:
        return

    meta_dict: Dict[str, Any] = {
        "epoch": int(epoch + 1),
        "loss": float(train_total_loss),
        "wandb_id": wandb_run_id,
        "is_deepspeed": True,
    }

    os.makedirs(save_path, exist_ok=True)

    # latest.pth 메타 저장 (dict[str, scalar])
    latest_path = os.path.join(save_path, "latest.pth")
    with io.BytesIO() as buff:
        torch.save(meta_dict, buff)
        data_bytes = buff.getvalue()
    with open(latest_path, "wb") as f:
        f.write(data_bytes)

    # best 갱신 시 best.pth도 동일 내용으로 덮어쓴다.
    if save_best:
        best_path = os.path.join(save_path, "best.pth")
        with open(best_path, "wb") as f:
            f.write(data_bytes)


def _save_deepspeed_checkpoint_for_epoch(
    diffusion_planner: torch.nn.Module,
    save_path: Optional[str],
    epoch: int,
    train_total_loss: float,
    wandb_run_id: Optional[str],
    model_ema: Optional[ModelEma],
    use_deepspeed: bool,
    save_best: bool,
) -> None:
    """DeepSpeed 엔진일 때 한 epoch 단위로 checkpoint를 저장한다.

    Args:
        diffusion_planner:
            deepspeed.DeepSpeedEngine 인스턴스를 기대한다.
            내부 파라미터 텐서 shape 는
              - 선형 weight: (out_dim, in_dim)
              - LayerNorm weight: (dim,)
              - conv weight: (C_out, C_in, kH, kW)
            등 모델 구조에 따라 다양하다.
        save_path:
            체크포인트 디렉터리 경로. None 이면 아무 것도 하지 않는다.
        epoch:
            현재 epoch 인덱스(0 기반).  # shape: ()
        train_total_loss:
            해당 epoch 전체 손실 값.   # shape: ()
        wandb_run_id:
            현재 W&B run ID. 없으면 None.
        model_ema:
            EMA 래퍼(ModelEma 등). 없으면 None.
        use_deepspeed:
            DeepSpeed 모드 사용 여부.
        save_best:
            이번 epoch 가 지금까지의 best 인지 여부.
            True 이면 tag="best" 로도 한 번 더 저장한다.
    """
    if (not use_deepspeed) or save_path is None:
        return
    if not hasattr(diffusion_planner, "save_checkpoint"):
        # DeepSpeedEngine 이 아니면 그냥 패스
        return

    # EMA 상태 dict 준비 (없으면 None)
    ema_state_dict: Optional[Dict[str, Any]] = None
    if model_ema is not None:
        try:
            # timm.ModelEma 는 .ema 가 실제 모델(nn.Module)이다.
            ema_state_dict = model_ema.ema.state_dict()
        except Exception:
            # 혹시 .ema 가 없는 커스텀 EMA 일 경우를 위한 fallback
            ema_state_dict = model_ema.state_dict()

    client_state: Dict[str, Any] = {
        "epoch": int(epoch + 1),
        "loss": float(train_total_loss),
        "wandb_id": wandb_run_id,
        "ema_state_dict": ema_state_dict,
    }

    # ✅ 실제 파라미터/옵티마/스케줄러 상태 저장 (latest 태그)
    diffusion_planner.save_checkpoint(
        save_dir=save_path,
        tag="latest",
        client_state=client_state,
    )

    # ✅ best 인 경우 best 태그도 별도로 저장해 둔다.
    if save_best:
        diffusion_planner.save_checkpoint(
            save_dir=save_path,
            tag="best",
            client_state=client_state,
        )

    # ✅ W&B / 사람이 보는 용도의 얇은 latest.pth / best.pth 메타 파일도 같이 작성
    _write_deepspeed_meta_checkpoint_files(
        save_path=save_path,
        epoch=epoch,
        train_total_loss=train_total_loss,
        wandb_run_id=wandb_run_id,
        save_best=save_best,
    )


def _is_deepspeed_checkpoint_dir(ckpt_dir: str) -> bool:
    """주어진 경로가 DeepSpeed checkpoint 디렉터리인지 대략 판별한다.

    규칙(간단한 휴리스틱):
      - ckpt_dir 안에 tag 디렉터리("latest" 또는 "best")가 있고
      - 그 안에 "mp_rank_" 로 시작하는 하위 디렉터리가 있으면
        DeepSpeed 가 저장한 구조라고 본다.

    Args:
        ckpt_dir:
            체크포인트 루트 디렉터리 경로.

    Returns:
        bool:
            True  → DeepSpeed 형식으로 저장된 디렉터리라고 판단.
            False → PyTorch 단일 .pth 형식이라고 보고 처리한다.
    """
    latest_tag_dir = os.path.join(ckpt_dir, "latest")
    best_tag_dir = os.path.join(ckpt_dir, "best")

    for tag_dir in (latest_tag_dir, best_tag_dir):
        if not os.path.isdir(tag_dir):
            continue
        try:
            children = os.listdir(tag_dir)
        except OSError:
            continue
        for name in children:
            # 예: mp_rank_00_model_states.pt, mp_rank_00/
            if name.startswith("mp_rank_"):
                return True
    return False


def _load_pytorch_checkpoint_into_deepspeed(
    ckpt_dir: str,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    device: str,
    global_rank: int,
) -> Tuple[int, Optional[str]]:
    """PyTorch 형식 latest.pth 를 불러 DeepSpeedEngine 내부 모듈에 주입한다.

    사용 상황:
      - 이전에는 DDP/단일 GPU(PyTorch) 로 학습해서 latest.pth 를 만들었고,
      - 지금은 DeepSpeed 모드로 전환해서 그 가중치부터 이어서 학습하고 싶을 때.

    처리 흐름:
      1) ckpt_dir/latest.pth 를 torch.load 로 읽는다.
      2) ckpt['model'] (또는 ckpt 전체)을 base_model(state_dict 구조와 맞게)로 맞춘다.
         - DDP 에서 저장된 "module.*" prefix 가 있으면 자동으로 제거한다.
      3) base_model.load_state_dict(..., strict=False) 로 파라미터를 채운다.
      4) EMA 가 있으면 ckpt['ema_state_dict'] 를 사용해 EMA 도 복원 시도한다.
      5) epoch / wandb_id 를 꺼내서 반환한다.

    Args:
        ckpt_dir:
            PyTorch latest.pth 가 들어 있는 디렉터리 경로.
        diffusion_planner:
            DeepSpeedEngine 인스턴스. 내부에 .module 로 실제 모델이 있다.
        model_ema:
            ModelEma 래퍼 또는 None.
        device:
            "cuda" / "cuda:0" / "cpu" 등. torch.load map_location 용.
        global_rank:
            전체 프로세스 기준 rank. 0 일 때만 로그를 찍는다.

    Returns:
        Tuple[int, Optional[str]]:
            - init_epoch: 재개 시작 epoch 인덱스.
            - wandb_id : ckpt 에 저장된 W&B run id (없으면 None).
    """
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"PyTorch latest.pth 를 찾을 수 없습니다: {ckpt_path}")

    # CPU 로 먼저 읽고, 이후 load_state_dict 가 device 에 맞게 옮기도록 둔다.
    map_location = "cpu" if device.startswith("cuda") else device
    ckpt = torch.load(ckpt_path, map_location=map_location)

    state_like = ckpt.get("model", ckpt)  # 'model' 키가 없으면 전체를 그대로 사용

    # DeepSpeedEngine 의 실제 모델(nn.Module)을 꺼낸다.
    base_model: nn.Module = getattr(diffusion_planner, "module",
                                    diffusion_planner)

    # 현재 모델 키와 ckpt 키를 비교해서 DDP prefix("module.") 정리
    current_keys = list(base_model.state_dict().keys())
    ckpt_keys = list(state_like.keys())

    def _strip_or_add_module_prefix(
        src_state: Dict[str, torch.Tensor],
        want_module_prefix: bool,
    ) -> Dict[str, torch.Tensor]:
        """DDP prefix(module.) 를 붙이거나 떼는 간단한 도우미."""

        new_state: Dict[str, torch.Tensor] = {}
        for k, v in src_state.items():
            if want_module_prefix:
                # 이미 module. 로 시작하면 그대로, 아니면 앞에 붙인다.
                new_k = k if k.startswith("module.") else f"module.{k}"
            else:
                # module. 로 시작하면 떼고, 아니면 그대로 둔다.
                new_k = k[7:] if k.startswith("module.") else k
            new_state[new_k] = v
        return new_state

    has_module_in_ckpt = all(k.startswith("module.") for k in ckpt_keys)
    has_module_in_model = all(k.startswith("module.") for k in current_keys)

    # ckpt: module.* / 모델: 평범한 키 → prefix 제거
    if has_module_in_ckpt and (not has_module_in_model):
        state_dict = _strip_or_add_module_prefix(
            state_like,
            want_module_prefix=False,
        )
    # ckpt: 평범한 키 / 모델: module.* → prefix 추가
    elif (not has_module_in_ckpt) and has_module_in_model:
        state_dict = _strip_or_add_module_prefix(
            state_like,
            want_module_prefix=True,
        )
    else:
        state_dict = state_like

    incompatible = base_model.load_state_dict(state_dict, strict=False)
    if global_rank == 0:
        if getattr(incompatible, "missing_keys", None):
            print(
                f"[DeepSpeed<PytorchResume>] missing_keys: {incompatible.missing_keys}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            print(
                f"[DeepSpeed<PytorchResume>] unexpected_keys: {incompatible.unexpected_keys}"
            )

    # EMA 복원 (있을 때만)
    if model_ema is not None:
        ema_state = ckpt.get("ema_state_dict", None)
        if ema_state is not None:
            try:
                ema_model: nn.Module = getattr(model_ema, "ema", model_ema)
                ema_incompatible = ema_model.load_state_dict(ema_state,
                                                             strict=False)
                ema_model.eval()
                for p in ema_model.parameters():
                    p.requires_grad_(False)

                if global_rank == 0:
                    if getattr(ema_incompatible, "missing_keys", None):
                        print(
                            f"[DeepSpeed<PytorchResume>] EMA missing_keys: {ema_incompatible.missing_keys}"
                        )
                    if getattr(ema_incompatible, "unexpected_keys", None):
                        print(
                            f"[DeepSpeed<PytorchResume>] EMA unexpected_keys: {ema_incompatible.unexpected_keys}"
                        )
                    print("[DeepSpeed<PytorchResume>] EMA state load done")
            except Exception as e:
                if global_rank == 0:
                    print(f"[DeepSpeed<PytorchResume>] EMA load 실패: {e}")

    init_epoch = int(ckpt.get("epoch", 0))
    wandb_id = ckpt.get("wandb_id", None)

    if global_rank == 0:
        print(f"[DeepSpeed<PytorchResume>] PyTorch latest.pth 로부터 로드 완료 "
              f"(epoch={init_epoch}, wandb_id={wandb_id})")

    return init_epoch, wandb_id


    # ✅ W&B / 사람이 보는 용도의 얇은 latest.pth / best.pth도 같이 작성
    # best 여부는 이 함수 안에서는 판단할 수 없으니,
    # 바깥에서 save_best를 같이 넘겨서 호출하도록 설계할 수도 있지만
    # 여기서는 "메타 latest.pth"만 쓰고, best는 PyTorch 경로에서만 관리해도 된다.
def _maybe_resume_from_checkpoint(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    global_rank: int,
    train_epochs: int,
    use_deepspeed: bool,
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str], int, bool]:
    """체크포인트가 지정된 경우 모델/옵티마/스케줄러/EMA 상태를 불러와서 이어 학습한다.

    분기:
      - use_deepspeed=False:
        · 기존 resume_model() 로 PyTorch latest.pth 를 그대로 불러온다.
      - use_deepspeed=True:
        · 먼저 ckpt_dir 가 DeepSpeed 형식인지 확인한다.
          * DeepSpeed 형식이면 engine.load_checkpoint(...) 를 사용.
          * 아니라면 PyTorch latest.pth 로 보고
            _load_pytorch_checkpoint_into_deepspeed(...) 로
            내부 모듈(및 EMA)만 채운 뒤, 옵티마/스케줄러는 새로 시작한다.

    EMA:
      - DeepSpeed load_checkpoint 를 쓸 때는 client_state['ema_state_dict'] 에
        EMA 상태를 같이 넣어두고, 여기서 model_ema 에 다시 복원한다.
      - PyTorch ckpt 를 DeepSpeed 로 읽을 때도 ckpt['ema_state_dict'] 를 사용해
        EMA 를 가능한 한 그대로 살린다.
    """
    allow_val_change = False

    if args.resume_local_path_model_path is not None:
        ckpt_dir = args.resume_local_path_model_path
        print(f"Model loaded from {ckpt_dir}")

        if use_deepspeed and hasattr(diffusion_planner, "load_checkpoint"):
            # ✅ 1) 먼저 DeepSpeed 형식인지 확인
            if _is_deepspeed_checkpoint_dir(ckpt_dir):
                # ---- DeepSpeed 전용 로드 경로 ----
                load_path, client_state = diffusion_planner.load_checkpoint(
                    ckpt_dir,
                    tag="latest",
                )
                if global_rank == 0:
                    print(
                        f"[DeepSpeed] load_checkpoint returned path={load_path}, "
                        f"client_state keys={list((client_state or {}).keys())}"
                    )

                client_state = client_state or {}
                init_epoch = int(client_state.get("epoch", 0))
                wandb_id = client_state.get("wandb_id", None)

                # EMA 복원
                ema_state_dict = client_state.get("ema_state_dict", None)
                if (model_ema is not None) and (ema_state_dict is not None):
                    try:
                        ema_model: nn.Module = getattr(model_ema, "ema",
                                                       model_ema)
                        ema_model.load_state_dict(ema_state_dict, strict=False)
                        ema_model.eval()
                        for p in ema_model.parameters():
                            p.requires_grad_(False)
                        if global_rank == 0:
                            print("[DeepSpeed] EMA state load done")
                    except Exception as e:
                        if global_rank == 0:
                            print(f"[DeepSpeed] EMA state load 실패: {e}")

                # DeepSpeed 엔진 내부 옵티마이저 그룹에도 lr_max / wd_max 보정
                if hasattr(diffusion_planner, "optimizer"):
                    for pg in diffusion_planner.optimizer.param_groups:
                        pg.setdefault("lr_max", float(pg["lr"]))
                        pg.setdefault("wd_max",
                                      float(pg.get("weight_decay", 0.0)))
            else:
                # ---- PyTorch ckpt 를 DeepSpeed 엔진으로 불러오는 경로 ----
                init_epoch, wandb_id = _load_pytorch_checkpoint_into_deepspeed(
                    ckpt_dir=ckpt_dir,
                    diffusion_planner=diffusion_planner,
                    model_ema=model_ema,
                    device=args.device,
                    global_rank=global_rank,
                )

                # DeepSpeed 옵티마이저 param_group 기본값 보정
                if hasattr(diffusion_planner, "optimizer"):
                    for pg in diffusion_planner.optimizer.param_groups:
                        pg.setdefault("lr_max", float(pg["lr"]))
                        pg.setdefault("wd_max",
                                      float(pg.get("weight_decay", 0.0)))
        else:
            # ✅ 기존 PyTorch / DDP 복원 경로
            (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id,
             model_ema) = resume_model(
                 ckpt_dir,
                 diffusion_planner,
                 optimizer,
                 scheduler,
                 model_ema,
                 args.device,
             )

            # resume 후에도 wd_max / lr_max 기본값이 유지되도록 보강
            for pg in optimizer.param_groups:
                pg.setdefault("lr_max", float(pg["lr"]))
                pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))

        # 재개 시 총 epoch 수가 init_epoch보다 작지 않도록 보정
        if args.train_epochs <= init_epoch:
            old = int(args.train_epochs)
            args.train_epochs = int(init_epoch) + 1
            if global_rank == 0:
                print(
                    f"[ADJUST] scaled train_epochs ({old}) <= init_epoch ({init_epoch}). "
                    f"Bumping to {args.train_epochs}.")

        train_epochs = int(args.train_epochs)

        if args.resume_model_from_wandb:
            allow_val_change = True
    else:
        init_epoch = 0
        wandb_id = None

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
    save_path: Optional[str],
    global_rank: int,
    wandb_id: Optional[str],
    allow_val_change: bool,
) -> Logger:
    """TensorBoard / W&B 로거를 만들고, 기존 아티팩트를 정리한다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        save_path: 체크포인트 저장 경로.
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
        save_path=save_path,
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
    #   near_future_gt_3_dim: (B, A, T_future, 3)
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


def _log_wandb_checkpoint_artifacts(
    args: argparse.Namespace,
    save_path: Optional[str],
    epoch: int,
    train_total_loss: float,
    save_best: bool,
    use_deepspeed: bool,
) -> None:
    """한 epoch가 끝난 뒤 로컬 체크포인트를 W&B 아티팩트로 올린다.

    - PyTorch/DDP 학습:
      latest.pth / best.pth 파일만 아티팩트에 담는다.
    - DeepSpeed 학습:
      latest.pth / best.pth 파일과 함께
      DeepSpeed가 만든 latest/ best 폴더(파라미터, 옵티마 포함)도 같이 올린다.

    Args:
        args: 학습 설정이 담긴 Namespace.
        save_path: 체크포인트가 저장된 로컬 폴더 경로. None이면 아무 것도 하지 않는다.
        epoch: 현재 epoch 인덱스(0부터 시작). 기록용으로만 사용한다.
        train_total_loss: 이번 epoch의 전체 손실 값.
        save_best: best.pth를 새로 갱신했는지 여부.
        use_deepspeed: DeepSpeed 모드 사용 여부.

    Returns:
        None: 파일 업로드만 수행한다.
    """
    if save_path is None:
        return
    if not getattr(args, "use_wandb", False):
        return
    if not wandb.run:
        # W&B 런이 없으면 업로드 불가
        return

    # 디렉터리 이름에서 대략적인 시간 문자열을 뽑아서 메타데이터에 남긴다.
    base_dir_name = os.path.basename(os.path.normpath(save_path))
    time_str_meta = base_dir_name.replace(":", "-")

    # ── latest-model 아티팩트 (매번 덮어쓰기) ──
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

    latest_pth = os.path.join(save_path, "latest.pth")
    if os.path.exists(latest_pth):
        latest_art.add_file(latest_pth)

    # DeepSpeed라면 latest 태그 디렉터리도 같이 넣어준다.
    if use_deepspeed:
        latest_tag_dir = os.path.join(save_path, "latest")
        if os.path.isdir(latest_tag_dir):
            # 아티팩트 안에서도 "latest/" 이름 그대로 보이도록 고정
            latest_art.add_dir(latest_tag_dir, name="latest")

    wandb.log_artifact(latest_art, aliases=["latest"])
    latest_art.wait()  # 업로드 완료 보장

    # 학습 도중 이전 버전들을 지우고 싶을 때
    if getattr(args, "delete_wb_weight_when_running", False):
        _prune_old_wandb_artifact_versions(
            collection_name=latest_coll,
            alias="latest",
        )

    # ── best-model 아티팩트 (새 best일 때만 갱신) ──
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

    best_pth = os.path.join(save_path, "best.pth")
    if os.path.exists(best_pth):
        best_art.add_file(best_pth)

    if use_deepspeed:
        best_tag_dir = os.path.join(save_path, "best")
        if os.path.isdir(best_tag_dir):
            best_art.add_dir(best_tag_dir, name="best")

    wandb.log_artifact(best_art, aliases=["best"])
    best_art.wait()

    if getattr(args, "delete_wb_weight_when_running", False):
        _prune_old_wandb_artifact_versions(
            collection_name=best_coll,
            alias="best",
        )


def _log_and_save_on_rank0(
    epoch: int,
    args: argparse.Namespace,
    train_total_loss: float,
    metrics: Dict[str, float],
    wandb_logger: Logger,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    save_path: Optional[str],
    best_loss: float,
    global_rank: int,
) -> float:
    """rank 0에서 epoch 로그를 기록하고, 필요 시 체크포인트와 아티팩트를 저장한다.

    DeepSpeed 모드와 일반(PyTorch/DDP) 모드를 나눠서 처리한다.
      - use_deepspeed=True:
        · DeepSpeed 엔진의 save_checkpoint(...) 를 호출해서
          ZeRO-2 상태까지 포함한 checkpoint 를 저장하고,
        · 같은 디렉터리에 latest.pth / best.pth 메타 파일도 함께 만든다.
        · 그런 뒤, 해당 폴더/파일을 W&B 아티팩트로 업로드한다.
      - use_deepspeed=False:
        · 기존 PyTorch save_model() 로 latest.pth / best.pth 를 저장하고,
        · 동일 파일을 W&B 아티팩트(latest / best)로 업로드한다.
    """
    if global_rank != 0:
        return best_loss

    # 1) W&B / TensorBoard 로그
    wandb_logger.log_metrics(metrics, step=epoch + 1)

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    # 2) 저장 주기 확인 (DeepSpeed / PyTorch 공통)
    if (epoch + 1) % int(args.save_utd) != 1:
        return best_loss

    # 3) best 갱신 여부 판단
    save_best = False
    if train_total_loss < best_loss:
        best_loss = train_total_loss
        save_best = True

    # 4) 모드별로 실제 체크포인트 저장 (로컬)
    if use_deepspeed:
        # DeepSpeedEngine 에게 직접 저장을 맡긴다.
        _save_deepspeed_checkpoint_for_epoch(
            diffusion_planner=diffusion_planner,
            save_path=save_path,
            epoch=epoch,
            train_total_loss=train_total_loss,
            wandb_run_id=wandb_logger.id,
            model_ema=model_ema,
            use_deepspeed=use_deepspeed,
            save_best=save_best,
        )
        print(f"[DeepSpeed] Checkpoint saved in {save_path}\n")
    else:
        # PyTorch / DDP 기존 경로
        save_model(
            diffusion_planner,
            optimizer,
            scheduler,
            save_path,
            epoch,
            train_total_loss,
            wandb_logger.id,
            model_ema.ema if model_ema is not None else None,
            save_best,
        )
        print(f"Model saved in {save_path}\n")

    # 5) 로컬 저장 후 W&B 아티팩트 업로드 (rank 0 전용)
    _log_wandb_checkpoint_artifacts(
        args=args,
        save_path=save_path,
        epoch=epoch,
        train_total_loss=train_total_loss,
        save_best=save_best,
        use_deepspeed=use_deepspeed,
    )

    return best_loss


def _run_training_loop(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    train_loader: DataLoader,
    train_sampler: DistributedSampler,
    wandb_logger: Logger,
    save_path: Optional[str],
    best_loss: float,
    global_rank: int,
    train_epochs: int,
    init_epoch: int,
    data_num_in_a_epoch: int,
    global_batch_size: int,
    aug: Optional[object],
) -> float:
    """전체 epoch 루프를 돌면서 학습, 로깅, 저장을 수행한다.

    Args:
        args: 학습 설정.
        diffusion_planner: 학습 중인 모델.
        optimizer: 옵티마이저.
        scheduler: 학습률 스케줄러.
        model_ema: EMA 래퍼 또는 None.
        train_loader: 학습용 DataLoader. 배치 텐서는 (B, ·) shape.
        train_sampler: DistributedSampler.
        wandb_logger: TensorBoardLogger 래퍼.
        save_path: 체크포인트 저장 경로.
        time_str: run 식별용 시간 문자열.
        best_loss: 현재까지의 best loss 값.
        global_rank: 전체 프로세스 기준 번호.
        train_epochs: 전체 epoch 수.
        init_epoch: 재개 시작 epoch 인덱스.
        data_num_in_a_epoch: 한 epoch당 처리 샘플 수.
        global_batch_size: 실제 글로벌 배치 크기.
        aug: augmentation 객체 또는 None.

    Returns:
        best_loss: 학습 종료 시의 best loss 값.
    """
    elapsed_training_time_hour = 0.0
    for epoch in range(init_epoch, train_epochs):
        # 1) 한 epoch 학습
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
        # 2) epoch당 처리 속도 계산
        data_process_per_sec = data_num_in_a_epoch / max(
            epoch_elapsed_time_sec, 1e-9)
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
        # 3) lr_dict / metrics 구성
        info_dict: Dict[str, float] = {
            'lr': optimizer.param_groups[0]['lr'],
            "total_train_epochs": train_epochs,
            "batch_num_in_epoch": len(train_loader),
            "total_batch_num_of_all_epochs": train_epochs * len(train_loader),
            "global_batch_size": global_batch_size,
        }
        weight_dict = {}
        direct_loss_dict = {}
        integration_loss_dict = {}
        constraint_loss_dict = {}
        loss_dict = {}
        for k, v in train_loss:
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
            elif k in ("integration_loss_xy", "integration_loss_yaw",
                       "integration_loss_xy_early",
                       "integration_loss_yaw_early"):
                integration_loss_dict[k] = v
            elif k in (
                    "constraint_diff_vx_b",
                    "constraint_diff_vy_b",
                    "constraint_diff_yaw_rate",
            ):
                constraint_loss_dict[k] = v
            elif k in ("loss_dict", "neighbor_prediction_loss",
                       "integration_loss", "constraint_loss"):
                loss_dict[k] = v
        speed_info = {
            "epoch_elapsed_time_sec": epoch_elapsed_time_sec,
            "data_process_per_sec": data_process_per_sec,
            "elapsed_training_time_hour": elapsed_training_time_hour,
        }
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
            f"constraint_loss_dict/{k}": v
            for k, v in constraint_loss_dict.items()
        })
        # add "loss_dict/" prefix
        metrics.update({f"loss_dict/{k}": v for k, v in loss_dict.items()})

        # add "speed_info/" prefix
        metrics.update({f"speed_info/{k}": v for k, v in speed_info.items()})

        # 4) rank 0에서 로그 및 체크포인트/아티팩트 저장
        best_loss = _log_and_save_on_rank0(
            epoch=epoch,
            args=args,
            train_total_loss=train_total_loss,
            metrics=metrics,  # Dict[str, float]
            wandb_logger=wandb_logger,
            diffusion_planner=diffusion_planner,
            optimizer=optimizer,
            scheduler=scheduler,
            model_ema=model_ema,
            save_path=save_path,
            best_loss=best_loss,
            global_rank=global_rank,
        )

        # 5) 다음 epoch 를 위한 sampler seed 변경
        train_sampler.set_epoch(epoch + 1 + args.sampler_epoch_offset)

    return best_loss


def _finalize_training_cleanup(
    args: argparse.Namespace,
    global_rank: int,
    wandb_logger: Logger,
) -> None:
    """학습이 끝난 뒤, 분산 동기화와 wandb/TensorBoard/로컬 파일 정리를 수행한다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        global_rank: 전체 프로세스 기준 번호.
        wandb_logger: TensorBoardLogger 래퍼.
    """
    # 모든 rank에서 학습 루프가 끝날 때까지 대기
    torch.distributed.barrier()

    if args.use_wandb:
        wandb.finish()
        if global_rank == 0:
            wandb_logger.finish()

    torch.distributed.barrier()

    if global_rank == 0 and args.save_path:
        print("[CLEANUP] 훈련 종료 후 로컬 체크포인트 및 로그 정리 시작")
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


def model_training(args: argparse.Namespace) -> None:
    """전체 학습 파이프라인을 실행하는 상위 함수.

    - 분산 설정/학습률 스케일링
    - Dataset / DataLoader 준비
    - 모델 / 옵티마이저 / 스케줄러 / EMA 준비
    - 체크포인트 재개 및 로깅 설정
    - epoch 루프 실행과 최종 정리

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
    """
    best_loss: float = float('inf')
    torch.cuda.empty_cache()

    # 1) 분산 초기화 및 rank 정보
    global_rank, rank, world_size, use_deepspeed = _init_distributed(args)

    # 2) 글로벌 배치 크기에 따라 학습률 / epoch 수 스케일링
    BASE_GLOBAL_BATCH, current_global_batch = _scale_learning_rate_and_epochs(
        args,
        world_size,
    )
    train_epochs: int = args.train_epochs

    # 3) save_path / args.json 준비
    save_path, time_str = _prepare_save_path_and_dump_args(args, global_rank)

    # 4) seed 고정
    set_seed(args.seed + global_rank)

    # 5) augmentation, Dataset, Sampler
    batch_size = args.batch_size
    aug = _build_augmentation(args)
    # train_epoch 내부에서 args를 통해 augmentation이 사용되므로 aug는 실제로는
    #   train_epoch 인자로만 전달되고, 배치 텐서 shape는 (B, ·)로 유지된다.
    _ = aug  # 형식상 참조 (실제 로직은 기존과 동일하게 train_epoch에서 사용)

    train_set, train_sampler = _build_dataset_and_sampler(
        args,
        world_size,
        global_rank,
    )

    # 6) warmup step 계산
    warmup_steps_at_B0, warmup_steps = _compute_warmup_steps(
        args=args,
        BASE_GLOBAL_BATCH=BASE_GLOBAL_BATCH,
        current_global_batch=current_global_batch,
        total_data_num=len(train_set),
    )

    # 7) DataLoader 및 step/샘플 수 정보
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

    # 8) rank 0에서 스케줄 요약 출력
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

    # 9) DDP인 경우, 모델 생성 전 barrier
    if args.ddp and not use_deepspeed:
        torch.distributed.barrier()

    # 10) 모델 / EMA / 옵티마이저 / 스케줄러 준비
    (diffusion_planner, model_ema, optimizer,
     scheduler) = _build_model_optimizer_scheduler(
         args=args,
         rank=rank,
         use_deepspeed=use_deepspeed,
         total_step_of_all_epoch=total_step_of_all_epoch,
         warmup_steps=warmup_steps,
         current_global_batch=current_global_batch,
     )

    # 11) 체크포인트 재개
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id,
     train_epochs, allow_val_change) = _maybe_resume_from_checkpoint(
         args=args,
         diffusion_planner=diffusion_planner,
         optimizer=optimizer,
         scheduler=scheduler,
         model_ema=model_ema,
         global_rank=global_rank,
         train_epochs=train_epochs,
         use_deepspeed=use_deepspeed,  # ✅ 추가
     )
    # 11-1) 재개 시 global step 복원 (w_dir / w_int / w_const 스케줄 연속성 보장)
    _init_or_restore_global_update_step(
        args=args,
        init_epoch=init_epoch,
        total_step_of_this_epoch=total_step_of_this_epoch,
    )

    # 12) 로거 설정 및 이전 아티팩트 정리
    wandb_logger = _setup_logger_and_purge(
        args=args,
        save_path=save_path,
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
        save_path=save_path,
        best_loss=best_loss,
        global_rank=global_rank,
        train_epochs=train_epochs,
        init_epoch=init_epoch,
        data_num_in_a_epoch=data_num_in_a_epoch,
        global_batch_size=global_batch_size,
        aug=aug,
    )

    # 14) 학습 종료 후 정리
    _finalize_training_cleanup(
        args=args,
        global_rank=global_rank,
        wandb_logger=wandb_logger,
    )


def _validate_wandb_resume_args(args: argparse.Namespace) -> None:
    """W&B에서 체크포인트를 받아올 때 필요한 기본 인자들을 점검한다.

    처리 내용:
      - args.resume_model_from_wandb 가 'latest' 인지 확인한다.
      - args.name 이 비어 있지 않은지 확인한다.
    둘 중 하나라도 조건을 만족하지 않으면 바로 ValueError 를 던져서
    잘못된 설정으로 내려받기를 시도하지 않도록 막는다.
    """
    # resume_model_from_wandb 가 "latest"가 아니면 에러를 발생시킵니다.
    if args.resume_model_from_wandb not in ['latest']:
        raise ValueError("args.resume_model_from_wandb must be 'latest.")
    if not args.name:
        raise ValueError(
            "args.name must be provided to resume from a wandb artifact.")


def _resolve_wandb_artifact_config(
    args: argparse.Namespace,) -> Tuple[str, str, str]:
    """재개하려는 W&B 아티팩트의 이름과 파일 이름을 결정한다.

    Args:
        args: argparse.Namespace
            - args.resume_model_from_wandb (str): 'latest' 또는 'best' 등을 기대.
            - args.name (str): 실험 이름. 컬렉션 이름의 접두사로 사용된다.

    Returns:
        resume_alias: str
            - 실제로 사용할 별칭. 예: 'latest', 'best'.
        collection_name: str
            - W&B 상에서 모델 묶음 이름. 예: f"{args.name}_latest-model".
        checkpoint_filename: str
            - 로컬에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    resume_alias = args.resume_model_from_wandb
    if resume_alias == 'best':
        collection_name = f"{args.name}_best-model"
        checkpoint_filename = "best.pth"
    else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
        collection_name = f"{args.name}_latest-model"
        checkpoint_filename = "latest.pth"
    return resume_alias, collection_name, checkpoint_filename


def _get_wandb_entity_and_project_for_resume(
    args: argparse.Namespace,) -> Tuple[str, str]:
    """W&B에서 아티팩트를 가져오기 위해 entity / project 정보를 얻는다.

    처리 내용:
      - 임시 W&B Run 을 하나 만든 뒤,
        그 Run 에서 entity, project 값을 읽고 바로 종료한다.
      - 이때 args.name 은 임시 Run 이름에만 사용되며,
        모델 학습용 Run 과는 별개이다.

    Returns:
        entity: str
            - 예: 'jksg01019-naver-labs'
        project: str
            - 예: 'Diffusion-Planner'
    """
    temp_run_for_context = wandb.init(
        project="Diffusion-Planner",
        name=f"temp_api_run_{args.name}",
        job_type="api_access",
    )
    entity = temp_run_for_context.entity
    project = temp_run_for_context.project
    temp_run_for_context.finish()
    return entity, project


def _download_wandb_checkpoint_to_local(
    api: wandb.Api,
    entity: str,
    project: str,
    collection_name: str,
    resume_alias: str,
    checkpoint_filename: str,
) -> str:
    """지정한 W&B 아티팩트에서 체크포인트 파일을 내려받고, 최종 디렉터리 경로를 돌려준다.

    처리 흐름:
      1) f"{entity}/{project}/{collection_name}:{resume_alias}" 경로로 아티팩트를 찾는다.
      2) 원본 Run 의 config 에서 'save_path' 를 읽어, 그 경로를 기준 디렉터리로 사용한다.
      3) 동일한 이름의 파일(checkpoint_filename)이 이미 있으면 지우고,
         다시 다운로드 받아 덮어쓴다.
      4) 다운로드가 끝나면, save_path 안에서 checkpoint_filename 이 실제로 존재하는지 확인한다.
         - 없다면 os.listdir(save_path) 로 파일 목록(list[str])을 찍어주고
           FileNotFoundError 를 던진다.
      5) 마지막으로 model_path 의 상위 디렉터리(체크포인트가 들어 있는 폴더 절대경로)를
         반환한다.

    Args:
        api: wandb.Api 인스턴스.
        entity: W&B entity 이름. 예: 'jksg01019-naver-labs'.
        project: W&B project 이름. 예: 'Diffusion-Planner'.
        collection_name: 모델 컬렉션 이름. 예: f"{args.name}_latest-model".
        resume_alias: 사용할 아티팩트 별칭. 예: 'latest', 'best'.
        checkpoint_filename: 내려받을 파일 이름. 예: 'latest.pth'.

    Returns:
        save_path: str
            - latest.pth / best.pth 가 들어 있는 디렉터리의 절대경로.
    """
    artifact_path = f"{entity}/{project}/{collection_name}:{resume_alias}"
    print(f"아티팩트 경로에서 가져오는 중: {artifact_path}")
    # Public API를 통해 아티팩트 객체를 가져옵니다.
    artifact = api.artifact(artifact_path, type='model')

    # 이 아티팩트를 생성한 원본 Run을 가져옵니다.
    source_run = artifact.logged_by()

    # 원본 Run의 config에서 save_path를 가져옵니다.
    if 'save_path' not in source_run.config:
        raise ValueError("원본 Run의 config에 'save_path'가 없습니다.")

    save_path = source_run.config['save_path']
    print(f"원본 Run의 config에서 save_path를 찾았습니다: {save_path}")
    os.makedirs(save_path, exist_ok=True)

    # 다운로드될 체크포인트 파일의 전체 경로를 지정합니다.
    target_file_path = os.path.join(save_path, checkpoint_filename)

    # 만약 해당 경로에 파일이 이미 존재하면, 덮어쓰기를 위해 삭제합니다.
    if os.path.exists(target_file_path):
        print(f"기존 파일 '{target_file_path}'가 존재하여 삭제하고 새로 다운로드합니다.")
        os.remove(target_file_path)

    # 아티팩트 파일을 다운로드하기 위해 임시 Run이 필요합니다.
    download_run = wandb.init(
        project=project,
        name=f"resume_run_download_{collection_name}",
        resume="allow",
    )
    # 현재 Run에서 사용할 아티팩트를 지정합니다.
    artifact_for_download = download_run.use_artifact(artifact,
                                                      aliases=[resume_alias])
    # 원본 save_path에 아티팩트 파일을 다운로드합니다.
    artifact_for_download.download(root=save_path)
    download_run.finish()
    print(f"아티팩트 '{artifact_path}'을(를) {save_path}에 다운로드했습니다.")

    # 다운로드된 체크포인트의 전체 경로를 설정합니다.
    model_path = os.path.join(save_path, checkpoint_filename)
    if not os.path.exists(model_path):
        downloaded_files = os.listdir(save_path)  # shape: (N,) list[str]
        raise FileNotFoundError(
            f"다운로드된 아티팩트 디렉터리에서 '{checkpoint_filename}'을(를) 찾을 수 없습니다: {save_path}. "
            f"사용 가능한 파일: {downloaded_files}")

    # save_path: latest.pth / best.pth 가 들어 있는 디렉터리
    save_path = os.path.dirname(os.path.abspath(model_path))
    return save_path


def _prepare_args_and_wandb_resume() -> argparse.Namespace:
    """명령행 인자를 읽고, 필요하다면 W&B 아티팩트에서 체크포인트를 내려받아 args를 보정한다.

    처리 흐름:
      2) args.resume_model_from_wandb 가 설정되지 않았다면 그대로 반환한다.
      3) 설정된 경우
         - _validate_wandb_resume_args() 로 기본 인자 검사를 하고
         - wandb.Api() 를 만든 뒤
         - _resolve_wandb_artifact_config() 으로 컬렉션/파일 이름을 정하고
         - _get_wandb_entity_and_project_for_resume() 로 entity, project 를 얻고
         - _download_wandb_checkpoint_to_local() 로 체크포인트를 내려받는다.
      4) 내려받은 디렉터리 경로를 args.resume_local_path_model_path 에 저장해
         이후 model_training() 의 resume 로직이 그대로 동작하도록 만든다.

    Returns:
        args: argparse.Namespace
            - 체크포인트를 쓰지 않을 때는 원본 get_args() 결과와 동일.
            - W&B에서 내려받기를 한 경우에는
              args.resume_local_path_model_path 가 실제 디렉터리 경로로 채워진 상태.
    """
    args = args_util.get_args()

    if not args.resume_model_from_wandb:
        return args

    _validate_wandb_resume_args(args)
    print(
        f"Resuming from wandb artifact: {args.name}:{args.resume_model_from_wandb}"
    )

    api = wandb.Api()

    try:
        """
        resume_alias : str
            - 실제로 사용할 별칭. 예: 'latest', 'best'.
        collection_name : str
            - W&B 상에서 모델 묶음 이름. 예: f"{args.name}_latest-model".
        checkpoint_filename : str
            - 로컬에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
        """
        resume_alias, collection_name, checkpoint_filename = \
            _resolve_wandb_artifact_config(args)
        """
        entity: str 예: 'jksg01019-naver-labs'
        project: str 예: 'Diffusion-Planner'
        """
        entity, project = _get_wandb_entity_and_project_for_resume(args)

        save_path = _download_wandb_checkpoint_to_local(
            api=api,
            entity=entity,
            project=project,
            collection_name=collection_name,
            resume_alias=resume_alias,
            checkpoint_filename=checkpoint_filename,
        )
        args.resume_local_path_model_path = save_path

        print(
            f"아티팩트를 {save_path}에 다운로드했습니다. 체크포인트에서 학습을 재개합니다: {args.resume_local_path_model_path}"
        )
    except Exception as e:
        print(f"W&B에서 재개하는 동안 오류 발생: {e}")
        # 오류 발생 시 임시 Run이 종료되도록 보장합니다.
        if wandb.run:
            wandb.finish()
        raise e

    return args


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
         체크포인트 파일(latest.pth 등)을 내려받아 args.resume_local_path_model_path 를 채운다.
      2) 환경변수 WORLD_SIZE 를 기준으로 args.distributed 를 설정해,
         이후 ddp 설정이 올바르게 동작하도록 만든다.
      3) model_training(args) 를 호출해 전체 학습 파이프라인을 수행하고,
         예외가 발생하면 rank 정보를 찍고 전체 스택을 출력한다.
    """
    args = _prepare_args_and_wandb_resume()
    _set_distributed_flag_from_env(args)

    # Run
    try:
        model_training(args)
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
