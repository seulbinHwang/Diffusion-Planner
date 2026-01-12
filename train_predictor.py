import os
from typing import Tuple
# 128 MiB 단위로 메모리 청크를 잘라서 할당하도록 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from diffusion_planner.utils.data_augmentation import StatePerturbation
# DDP 디버깅을 위해 사용되지 않은 파라미터 정보를 상세히 출력
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
import torch
from diffusion_planner.utils.lr_schedule import (
    build_pytorch_warmup_cosine_scheduler,
    build_pytorch_warmup_constant_scheduler,
)
# deprecated 키는 사용 금지
os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
from collections import defaultdict
from typing import Optional, Dict
import io

# TensorFloat-32(TF32) 연산을 허용하여
#   - Ampere(A100 등) GPU에서 matmul/cuDNN 연산을 FP32보다 빠르게 처리하고
#   - 눈에 띄는 정밀도 손실 없이 학습·추론 속도를 높이기 위한 설정입니다.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # PyTorch>=2.0
import sys, faulthandler, traceback

faulthandler.enable(all_threads=True)

import argparse
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader
import wandb
from diffusion_planner.utils.train_utils import set_seed, save_model
from diffusion_planner.utils.normalizer import ObservationNormalizer
from tools.predictor_utils import (
    build_dataset_and_sampler,
    build_data_loader,
    create_diffusion_planner_and_ema,
    set_save_path,
    prepare_wandb_resume,
    set_distributed_flag_from_env,
    build_deepspeed_config,
    effective_global_batch,
    init_distributed,
    maybe_resume_from_checkpoint,
    safe_get_artifacts,
)
from tools.predictor_utils import (build_dataset_and_sampler, build_data_loader,
                                   create_diffusion_planner_and_ema,
                                   setup_logger_and_purge)

from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.train_epoch import train_epoch
import os
from eval_predictor import *
import math

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass
import _hashlib
from wandb.sdk.lib import hashutil as wandb_hashutil
from src.smart.metrics import WOSACMetrics
from src.smart.metrics import minADE


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

# === [NEW] LR auto-scaling (root scale for Adam/AdamW) ======================


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
    current_global_batch = effective_global_batch(args.batch_size, world_size)

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
    batch_num_in_all_epoch: int,
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
        batch_num_in_all_epoch,
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
    use_deepspeed = args.use_deepspeed

    if (not use_deepspeed) and (global_rank != 0):
        print(f"[Rank {global_rank}] Skipping logging and saving.\n")
        return best_loss

    # 1) 메트릭 로그
    if global_rank == 0:
        wandb_logger.log_metrics(metrics, step=epoch + 1)

    # 2) 저장 주기 확인 (DeepSpeed / PyTorch 공통)
    save_interval: int = max(1, int(args.save_utd))
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


def _init_global_step_and_total_updates(
    args: argparse.Namespace,
    batch_num_in_epoch: int,
) -> int:
    """전역 스텝 상태를 초기화하고, 전체 업데이트 스텝 수를 계산한다.

    Args:
        args:
            학습 설정/상태를 담고 있는 Namespace.
            - train_epochs: 전체 학습 epoch 수.
            - _global_update_step: 없으면 0으로 새로 만든다.
        batch_num_in_epoch:
            현재 epoch에서 배치 개수. len(data_loader).

    Returns:
        int:
            batch_num_in_all_epoch = train_epochs * batch_num_in_epoch (최소 1).
    """
    if not hasattr(args, "_global_update_step"):
        args._global_update_step = 0
    batch_num_in_all_epoch: int = max(
        1,
        int(args.train_epochs) * int(batch_num_in_epoch))
    return batch_num_in_all_epoch


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
    # 전체 업데이트 스텝 수 설정 및 global step 초기화 보장
    batch_num_in_all_epoch: int = _init_global_step_and_total_updates(
        args,
        batch_num_in_epoch=len(train_loader),
    )
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
            batch_num_in_all_epoch=batch_num_in_all_epoch,
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
            weight_dict,  # Dict[str, float]
            direct_loss_dict,  # Dict[str, float]
            integration_loss_dict,  # Dict[str, float]
            constraint_loss_dict,  # Dict[str, float]
            loss_dict,  # Dict[str, float]
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
    if global_rank == 0:
        wandb_logger.finish()
    # 2) W&B / TensorBoard 종료
    if args.use_wandb and wandb.run is not None:
        wandb.finish()


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
    """
    train_set: DiffusionPlannerData
    train_sampler: DistributedSampler
    """
    train_set, train_sampler = build_dataset_and_sampler(
        args,
        args.train_set,
        args.train_set_list,
        "train",
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
    train_loader = build_data_loader(
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

    diffusion_planner, model_ema, base_model = create_diffusion_planner_and_ema(
        args=args,
        rank=rank,
        use_deepspeed=use_deepspeed,
    )

    # ✅ (핵심) DeepSpeed + model-only 재시작인 경우:
    #    checkpoint(모델/EMA) 로드를 DeepSpeed initialize 이전에 수행한다.
    resume_model_only_pre_ds_init: bool = bool(
        use_deepspeed and bool(getattr(args, "resume_model_only", False)) and
        (getattr(args, "resume_wandb_model_name", None) is not None)
    )

    init_epoch: int = 0
    wandb_id: Optional[str] = None
    train_epochs: int = int(args.train_epochs)
    allow_val_change: bool = False

    if resume_model_only_pre_ds_init:
        if global_rank == 0:
            print("[DeepSpeed][ModelOnly] checkpoint를 DeepSpeed initialize 이전에 로드합니다.")
        (
            diffusion_planner,
            _,
            _,
            model_ema,
            init_epoch,
            wandb_id,
            train_epochs,
            allow_val_change,
        ) = maybe_resume_from_checkpoint(
            args=args,
            diffusion_planner=diffusion_planner,
            optimizer=None,
            scheduler=None,
            model_ema=model_ema,
            global_rank=global_rank,
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
    # - model-only + DeepSpeed는 위에서 이미 로드했으므로 여기서는 다시 로드하지 않는다.
    if not resume_model_only_pre_ds_init:
        (
            diffusion_planner,
            optimizer,
            scheduler,
            model_ema,
            init_epoch,
            wandb_id,
            train_epochs,
            allow_val_change,
        ) = maybe_resume_from_checkpoint(
            args=args,
            diffusion_planner=diffusion_planner,
            optimizer=optimizer,
            scheduler=scheduler,
            model_ema=model_ema,
            global_rank=global_rank,
            use_deepspeed=use_deepspeed,
        )


    _dump_args(args, global_rank)

    _init_or_restore_global_update_step(
        args=args,
        init_epoch=init_epoch,
        total_step_of_this_epoch=total_step_of_this_epoch,
    )

    wandb_logger = setup_logger_and_purge(
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
    global_rank, rank, world_size, use_deepspeed = init_distributed(args)
    set_save_path(
        args=args,
        global_rank=global_rank,
    )
    should_finish = prepare_wandb_resume(args)
    set_distributed_flag_from_env(args)

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
