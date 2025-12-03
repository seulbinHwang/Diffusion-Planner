import os
from typing import Optional, Tuple
# 128 MiB 단위로 메모리 청크를 잘라서 할당하도록 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import args_util
# DDP 디버깅을 위해 사용되지 않은 파라미터 정보를 상세히 출력
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
import torch
from typing import Any, Callable, Dict

# --- sanity check: env vars that must be ints ---
# def _fix_int_env(name: str, default: Optional[int] = None):
#     v = os.environ.get(name)
#     if v is None:
#         return
#     try:
#         int(str(v).strip())
#     except Exception:
#         msg = f"[WARN] invalid {name}={v!r}"
#         if default is None:
#             os.environ.pop(name, None)
#             print(msg + " -> unset")
#         else:
#             os.environ[name] = str(int(default))
#             print(msg + f" -> set {name}={default}")
#
# for _k,_d in [
#     ("CUDA_DEVICE_MAX_CONNECTIONS", 32),
#     ("TORCH_NCCL_ASYNC_ERROR_HANDLING", 1),
#     ("OMP_NUM_THREADS", 8),
# ]:
#     _fix_int_env(_k, _d)

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

import math

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass


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


# --- put this in a utils file or near your optimizer build code ---
from typing import List, Set
import torch
import torch.nn as nn
from timm.optim.optim_factory import param_groups_weight_decay

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
):
    """
    - timm의 param_groups_weight_decay를 사용해 bias/Norm/1D 자동 no-decay
    - 위 discover 함수로 찾은 토큰/포지션 파라미터 추가 no-decay
    - Optimizer에는 전역 WD=0.0 (그룹에 이미 들어감)
    """
    # 토큰/포지션 추가 no-decay 수집
    extra_nwd = discover_extra_no_weight_decay_names(
        model, include_seed_params=include_seed_params)

    # 그룹 생성 (timm 헬퍼)
    param_groups = param_groups_weight_decay(
        model,
        weight_decay=weight_decay,
        no_weight_decay_list=extra_nwd,
    )
    # 공통 LR 부여
    for g in param_groups:
        g["lr"] = lr  # η_max(B)
        g["lr_max"] = float(lr)  # 기준 LR(스케줄 시작 시점)
        g["wd_max"] = float(g.get("weight_decay", 0.0))  # 그룹별 기준 WD

    # 최종 옵티마이저 (전역 WD는 0.0로 중복 방지)
    try:
        optim = torch.optim.AdamW(param_groups, fused=True, weight_decay=0.0)
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


def model_training(args):
    best_loss = float('inf')
    torch.cuda.empty_cache()
    # init ddp
    """
    global_rank (int) : 훈련 전체 프로세스에서 내 등번호 (샘플러 분배, 로깅 분기 등에 사용)
    rank (int) : 이 노드에서 내가 사용할 CUDA 디바이스 인덱스 (DDP device_ids=[rank]에 그대로 들어감)
        - global_rank와 rank는 우리의 경우 같음.
    """
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    world_size = ddp.get_world_size()

    ########  [LR (1) ] 학습률 선형 스케일링 =========================
    BASE_GLOBAL_BATCH = 2048  # 기존 기준 글로벌 배치
    BASE_LR = args.learning_rate  # 기존 기준 LR (Adam/AdamW)

    # DataLoader가 실제로 사용할 글로벌 배치(정수 배수)로 계산
    current_global_batch = _effective_global_batch(args.batch_size, world_size)
    scale = math.sqrt(current_global_batch / float(BASE_GLOBAL_BATCH))
    args.learning_rate = BASE_LR * scale
    ################################################################
    ########  [LR (2) ] epoch scaling (keep total steps constant) ########
    # - 앵커 에폭은 "현재 args.train_epochs"를 기준(기본 500)
    EPOCH_BETA = 1
    base_epochs_anchor = int(args.train_epochs)  # 500

    scaled_epochs = _auto_scale_train_epochs(
        base_global_batch=BASE_GLOBAL_BATCH,  # B_0 = 2048
        base_epochs=base_epochs_anchor,  # E_0 = 500
        current_global_batch=current_global_batch,  # B = 실제 글로벌 배치
        beta=EPOCH_BETA,  # 1
        clamp_min=max(1, args.warm_up_epoch + 1),  # warmup 안전
        clamp_max=None,  # 필요하면 예: 2000 등
    )
    args.train_epochs = int(scaled_epochs)
    train_epochs = args.train_epochs

    if global_rank == 0:
        """
    •	rank 0 프로세스에서만, 재개 경로가 있으면 그 경로를 save_path로 쓰고, 없으면 save_dir/training_log/name/현재시각/ 폴더를 새로 만들고 save_path로 설정합니다.
	•	args를 딕셔너리로 변환하되 StateNormalizer, ObservationNormalizer는 .to_dict()로 바꿔 넣습니다.
	•	그 딕셔너리를 save_path/args.json에 JSON으로 저장합니다.
        """
        if args.resume_local_path_model_path is not None:
            # resume_local_path_model_path: ./training_log/npc_aug_n_ego_past/2025-08-06-07:23:04/
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
        # Save args
        args_dict = vars(args)
        args_dict = {
            k:
                v if not isinstance(v, (StateNormalizer, ObservationNormalizer))
                else v.to_dict() for k, v in args_dict.items()
        }

        from mmengine.fileio import dump
        dump(args_dict,
             os.path.join(save_path, 'args.json'),
             file_format='json',
             indent=4)
    else:
        save_path = None
        args.save_path = None

    # set seed
    set_seed(args.seed + global_rank)
    # Enable anomaly detection to trace NaN/Inf origins during training
    # torch.autograd.set_detect_anomaly(True)

    # training parameters
    batch_size = args.batch_size

    # set up data loaders
    if args.use_ego_data_augment and args.use_npc_data_augment:
        raise ValueError(
            "You cannot use both ego and npc data augmentation at the same time. "
        )
    if args.use_ego_data_augment:
        aug = StatePerturbation(augment_prob=args.augment_prob,
                                device=args.device)
    elif args.use_npc_data_augment:
        aug = NPCStatePerturbation(augment_prob=args.augment_prob,
                                   device=args.device)
    else:
        aug = None
    """ DiffusionPlannerData
	•	학습에 쓸 샘플 목록을 읽고( JSON 경로 ), 
	    **데이터 루트 디렉터리( npz 위치 )**와 함께 
	    인덱스→파일 경로 매핑을 제공
	•	__len__은 샘플 개수(목록 길이) 를, 
	    __getitem__(idx)는 idx번째 npz를 로드해서 
	        max_agent_num, predicted_neighbor_num, future_len에 맞게 필요 채널만 잘라 
	        모델이 기대하는 튜플(batch 항목 순서 고정) 로 반환
	•	즉, DataLoader/DistributedSampler가 순회할 단일 샘플 로더를 정의해 주는 역할이며, 
	    셔플·분산 분배는 Sampler가 담당
    """
    train_set = DiffusionPlannerData(
        args.train_set,  # "/mnt/nuplan/dataset/processed"
        args.
        train_set_list,  # "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
        args.future_len)
    """ DistributedSampler
	•	전체 데이터 인덱스를 전역으로 섞고(shuffle=True),
	•	월드 크기(num_replicas=전체 프로세스 수)만큼 균등 분할한 뒤,
	•	현재 프로세스(rank=global_rank)에 해당하는 부분만 DataLoader에 넘겨줍니다.
	    - rank * num_samples : (rank+1)*num_samples 슬라이스 반환 규칙이 있다.
	•	그래서 GPU마다 겹치지 않는 샘플만 보게 되고, 
	        set_epoch() 호출 시 에폭마다 다시 전역 셔플되어 분배가 바뀝니다.
    """
    train_sampler = DistributedSampler(train_set,
                                       num_replicas=ddp.get_world_size(),
                                       rank=global_rank,
                                       shuffle=True,
                                       seed=args.seed)
    ################################################################
    ######## [LR (3) ] T_w,0  (warmup_steps_at_B0) 구하기 = 배치 B_0에서의 워밍업 스텝 수 ##############
    # (배치 B_0에서의 에폭당 스텝 수 * 워밍업 에폭 수(args.warm_up_epoch))
    total_data_num = len(train_set)
    steps_per_epoch_at_B0 = math.ceil(
        total_data_num / float(BASE_GLOBAL_BATCH))  # B_0=2048에서의 epoch당 스텝 수
    warmup_steps_at_B0 = steps_per_epoch_at_B0 * args.warm_up_epoch
    batch_ratio = current_global_batch / float(BASE_GLOBAL_BATCH)
    ####### T_w(B) (warmup_steps) 구하기 : 배치 B에서의 워밍업 스텝 수 ##############
    WARMUP_SCALE_EXP = 0.5
    # : warmup_steps
    warmup_steps = math.ceil(warmup_steps_at_B0 *
                             batch_ratio**WARMUP_SCALE_EXP)  # B에서의 워밍업 스텝 수
    ###########################################################
    """ DataLoader
•	무엇을 하냐: 
        DistributedSampler가 넘겨준 인덱스 순서대로
            DiffusionPlannerData에서 데이터를 읽어와, 
        이 랭크의 배치 크기(batch_size // world_size)로 배치들을 만들어 GPU로 전달
•	어떻게 빠르게 읽냐: 
        num_workers 만큼 백그라운드 워커로 병렬 로딩, 
        prefetch_factor로 미리 읽기, 
        pin_memory=True로 고정 메모리 사용, 
        persistent_workers=True로 에폭 간 워커 유지.
•	배치 정합: 
        drop_last=True라서 마지막 불완전 배치는 버려져, 모든 랭크가 동일 스텝 수를 가집니다.

    """
    train_loader = DataLoader(
        train_set,  # DiffusionPlannerData
        sampler=train_sampler,  # DistributedSampler
        batch_size=batch_size // ddp.get_world_size(),
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_mem,
        persistent_workers=True,  # 에폭이 바뀌어도 워커 유지
        drop_last=True,
        collate_fn=DiffusionPlannerCollate(args),  # ← 변동 길이 패딩
    )
    ############## [LR (2) ] T: 총 업데이트 스텝 수 (유지 대상) 구하기 ####################
    """ total_step_of_this_epoch
    floor( (샘플러가 이 랭크에 준 샘플 수) / (batch_size_per_rank) ).
    """
    total_step_of_this_epoch = len(train_loader)
    total_step_of_all_epoch = args.train_epochs * total_step_of_this_epoch
    ################################################################################
    # 실제 글로벌 배치 크기(정확)
    world_size = ddp.get_world_size()
    bs_per_rank = args.batch_size // world_size
    global_batch_size = bs_per_rank * world_size
    # 에폭당 처리 샘플 수 (drop_last이므로 len(loader)*global_batch_size)
    samples_this_epoch = total_step_of_this_epoch * global_batch_size
    if global_rank == 0:
        print("==========[Learning HYPERPARAMETER INFO]===============")
        print("Dataset Prepared: {} train data\n".format(len(train_set)))
        print(f"[Schedule] B0={BASE_GLOBAL_BATCH}, B={current_global_batch}, "
              f"eta_max={args.learning_rate:.3e}, "
              f"E(B)={args.train_epochs}, "
              f"steps/epoch={len(train_loader)}, "
              f"T={total_step_of_all_epoch}, "
              f"Tw0={warmup_steps_at_B0}, Tw(B)={warmup_steps}")
        print("=====================================================")

    if args.ddp:
        """
•	모든 DDP 프로세스를 이 지점에서 멈춰 세우고, 모든 맴버이 도착할 때까지 기다립니다.
•	그래서 앞 단계 작업(로깅, 저장, 초기화 등)이 전 프로세스에서 끝난 뒤 다음 코드로 함께 진행하게 합니다.
        """
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device ==
                                             'cuda' else args.device)

    if args.ddp:
        """
•	이 줄은 현재 모델을 “분산 학습용 모델”로 바꿉니다.
    •	이 프로세스가 사용할 GPU 번호를 지정하고, 그 GPU에 모델을 올립니다.
•	backward() 이후 
        모든 프로세스의 그라디언트를 자동으로 합/평균(ALL-REDUCE) 해서 
            각 프로세스가 같은 그라디언트를 갖게 합니다.
•	optimizer.step()은 각 프로세스에서 동일하게 실행되지만, 
        파라미터가 항상 동일하게 유지됩니다(초기 파라미터 방송, 버퍼/파라미터 동기화 포함).
•	코드에서 원본 모델에 접근할 땐 ddp.get_model(...).module 형태로 씁니다.

요약: 한 프로세스-한 GPU로 모델을 배치하고, 역전파 때 자동 동기화까지 처리하는 래퍼입니다.
        """
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])
    model_ema = None
    if args.use_ema:
        " 현재 모델을 깊은 복사해서 EMA용 모델(model_ema.ema)을 만듭니다. "
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
    # --- build param groups with correct no-decay (timm helper) ---
    base_model = ddp.get_model(diffusion_planner, args.ddp)
    """ build_adamw_with_param_groups
•	무엇을 하냐: 모델 파라미터를 두 묶음으로 나눠 AdamW 옵티마이저를 만듭니다.
        1.	weight decay(가중치 감쇠) 적용 묶음(일반 가중치)
        2.	감쇠 제외 묶음
            (bias, LayerNorm/Norm 계열, 토큰·포지션 파라미터(seeds/ego_fut_seeds 포함))
•	어떻게 세팅하냐: 각 묶음에 같은 학습률을 넣되, 
        감쇠 적용 여부만 다르게 해서 옵티마이저를 만들고,
        함께 감쇠 제외 목록(extra_nwd) 도 돌려줍니다.
•	왜 이렇게 하냐: 
        weight decay는 행렬 가중치엔 유익하지만, 
        bias/Norm/임베딩(토큰·포지션) 같은 값에 걸면 표현력이 줄거나 수렴이 흔들릴 수 있어요. 
        그래서 **“감쇠로 과적합 억제”**와 **“중요 파라미터 보존”**을 동시에 잡기 위해
        이런 파라미터 그룹화로 옵티마이저를 만드는 거예요.
    """
    optimizer, extra_nwd = build_adamw_with_param_groups(
        model=base_model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,  # 1e-2
        include_seed_params=True,
        # ← seeds/ego_fut_seeds도 no-decay에 포함(원치 않으면 False)
    )
    # ↓↓↓ 여기 추가 (스케줄러 생성/step 이전) ↓↓↓
    for pg in optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))  # η_max(B)
        pg.setdefault("wd_max", float(pg.get("weight_decay",
                                             0.0)))  # 그룹별 WD 기준값
    # ↑↑↑
    ############## [LR (4) ] 선형 워밍업 -> 코사인 디케이 (스케쥴) ##########################
    # pseudo_total_update_steps = 110000
    scheduler = build_pytorch_warmup_cosine_scheduler(optimizer,
                                                      total_step_of_all_epoch,
                                                      warmup_steps,
                                                      eta_min=0.2 *
                                                      args.learning_rate)
    # if warmup_steps > 0:
    #     scheduler.step()  # 초기 LR을 warmup 첫 값으로 세팅
    allow_val_change = False
    if args.resume_local_path_model_path is not None:
        print(f"Model loaded from {args.resume_local_path_model_path}")
        """
•	체크포인트 재개: 
        resume_model(<path>, ...)을 호출해 
        <path>/latest.pth에서 모델 가중치, 옵티마이저 상태, 스케줄러 상태, 
        시작 에폭, W&B id, EMA 가중치를 불러옵니다.
•	EMA 세팅: 
        불러온 EMA 가중치를 eval()로 두고 requires_grad=False로 고정합니다.
•	반환/대입: 
        복구된 객체들을 (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema)
        에 덮어써 이후 학습을 바로 이어갑니다.
        """
        (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id,
         model_ema) = resume_model(args.resume_local_path_model_path,
                                   diffusion_planner, optimizer, scheduler,
                                   model_ema, args.device)
        # ↓↓↓ 여기 추가 (resume이 optimizer를 교체했으므로 보강) ↓↓↓
        for pg in optimizer.param_groups:
            pg.setdefault("lr_max", float(pg["lr"]))  # η_max(B)
            pg.setdefault("wd_max", float(pg.get("weight_decay",
                                                 0.0)))  # 그룹별 WD 기준값
        # ↑↑↑
        # --- [NEW] If resumed, ensure total epochs > init_epoch ------------------
        # 재개 시 총 에폭이 초기 에폭보다 작거나 같으면 최소 1epoch 더 돌도록 보정
        if args.train_epochs <= init_epoch:
            old = int(args.train_epochs)
            args.train_epochs = int(init_epoch) + 1
            if global_rank == 0:
                print(
                    f"[ADJUST] scaled train_epochs ({old}) <= init_epoch ({init_epoch}). "
                    f"Bumping to {args.train_epochs}.")

        # (로컬 변수도 갱신하여 아래 스케줄러/루프에서 동일 값 사용)
        train_epochs = int(args.train_epochs)
        if args.resume_model_from_wandb:
            allow_val_change = True
    else:
        init_epoch = 0
        wandb_id = None

    # logger
    wandb_logger = Logger(args.name,
                          args.notes,
                          args,
                          wandb_resume_id=wandb_id,
                          save_path=save_path,
                          rank=global_rank,
                          allow_val_change=allow_val_change)
    if global_rank == 0 and args.remove_existing_wb_weight:
        api = wandb.Api()

        # 현재 run의 entity / project
        entity = wandb.run.entity
        project = wandb.run.project

        # 우리가 새로 만들 컬렉션 이름 (time_str는 앞에서 한 번 계산됨)
        purge_collection(api, entity, project, f"{args.name}_latest-model")
        purge_collection(api, entity, project, f"{args.name}_best-model")
    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        # scheduler.step()
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        epoch_t0 = time.perf_counter()
        train_loss, train_total_loss = train_epoch(
            train_loader,  # DataLoader
            diffusion_planner,
            optimizer,
            args,
            model_ema,
            scheduler,
            aug)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        # === [추가] 에폭 종료 시간 & 에폭 속도 계산 ===
        if args.ddp:
            torch.cuda.synchronize()
            torch.distributed.barrier()
        epoch_time_sec = time.perf_counter() - epoch_t0

        epoch_sps = samples_this_epoch / max(epoch_time_sec, 1e-9)
        ###########################
        if global_rank == 0:
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}

            # <추가하자> 에폭 평균으로 집계된 feasible weight/progress를 lr/* 아래에 함께 기록
            for k in ("feasible_progress", "feasible_w_dir", "feasible_w_int",
                      "feasible_w_const"):
                if k in train_loss:  # train_loss는 get_epoch_mean_loss() 결과
                    lr_dict[k] = train_loss[k]  # 예: lr/feasible_w_dir 로 기록됨

            metrics = {
                **{
                    f"train_loss/{k}": v for k, v in train_loss.items()
                },
                **{
                    f"lr/{k}": v for k, v in lr_dict.items()
                },
                "speed/epoch_time_sec": epoch_time_sec,
                "speed/epoch_samples_per_sec": epoch_sps,
                "speed/epoch_batches": len(train_loader),
                "speed/global_batch_size": global_batch_size,
            }
            wandb_logger.log_metrics(metrics, step=epoch + 1)

            if (epoch + 1) % args.save_utd == 1:
                save_best = False
                if train_total_loss < best_loss:
                    best_loss = train_total_loss
                    save_best = True
                # save model at the end of epoch
                save_model(diffusion_planner, optimizer, scheduler, save_path,
                           epoch, train_total_loss, wandb_logger.id,
                           model_ema.ema if model_ema is not None else None,
                           save_best)
                print(f"Model saved in {save_path}\n")
                # ── latest-model 아티팩트 (매번 덮어쓰기) ──
                # save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
                # f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth'
                latest_coll = f"{args.name}_latest-model"
                latest_art = wandb.Artifact(
                    name=latest_coll,  # latest-model 라는 이름의 데이터 묶음
                    type="model",  # "dataset" 이 될수도 있음
                    metadata={
                        "time_str": time_str,
                        "epoch": epoch + 1,
                        "loss": train_total_loss
                    })
                # 로컬에 저장된 latest.pth 파일을 담아 넣는 동작
                latest_art.add_file(os.path.join(save_path, "latest.pth"))
                """
이 상자(Artifact)를 W&B 서버로 전송하고, ["latest"]라는 별명을 붙여 줘요.
별명을 쓰면, 다음에 다시 “latest”라는 이름으로 덮어써 가며 하나의 모델만 관리할 수 있습니다.
                """
                wandb.log_artifact(latest_art, aliases=["latest"])
                latest_art.wait()  # 업로드 완료 보장

                if args.delete_wb_weight_when_running:
                    # 이전 버전 삭제
                    api = wandb.Api()
                    # wandb.run.entity: jksg01019-naver-labs
                    # wandb.run.project: Diffusion-Planner
                    entity = wandb.run.entity
                    project = wandb.run.project
                    # ':latest' alias로 가져오면 방금 올린 버전이 리턴됩니다

                    current = api.artifact(
                        f"{entity}/{project}/{latest_coll}:latest")

                    # 3) 모든 버전 목록 중, 이 버전이 아닌 나머지를 삭제
                    for v in api.artifacts("model",
                                           f"{entity}/{project}/{latest_coll}"):
                        if v.id != current.id:
                            v.delete()
                # ── best-model 아티팩트 (조건부 덮어쓰기) ──
                if save_best:
                    best_coll = f"{args.name}_best-model"
                    best_art = wandb.Artifact(name=best_coll,
                                              type="model",
                                              metadata={
                                                  "time_str": time_str,
                                                  "epoch": epoch + 1,
                                                  "loss": train_total_loss
                                              })
                    best_art.add_file(os.path.join(save_path, "best.pth"))
                    wandb.log_artifact(best_art, aliases=["best"])
                    best_art.wait()  # 업로드 완료 보장

                    # 이전 버전 삭제
                    if args.delete_wb_weight_when_running:
                        entity = wandb.run.entity
                        project = wandb.run.project
                        # ':latest' alias로 가져오면 방금 올린 버전이 리턴됩니다
                        current = api.artifact(
                            f"{entity}/{project}/{best_coll}:best")

                        # 3) 모든 버전 목록 중, 이 버전이 아닌 나머지를 삭제
                        for v in api.artifacts(
                                "model", f"{entity}/{project}/{best_coll}"):
                            if v.id != current.id:
                                v.delete()

        # scheduler.step()
        train_sampler.set_epoch(epoch + 1 + args.sampler_epoch_offset)

    # ── 모든 훈련 종료 후 정리 ─
    torch.distributed.barrier()  # ① 모든 rank의 학습 루프 종료 동기화

    # ② 모든 rank에서 wandb 종료 (사용 시)
    if args.use_wandb:
        wandb.finish()
        if global_rank == 0:
            wandb_logger.finish()  # TensorBoard Logger의 writer.close() 호출

    torch.distributed.barrier()  # ③ 모든 rank의 wandb 종료 동기화

    # ④ Rank 0에서만 로컬 파일 정리
    if global_rank == 0 and args.save_path:
        print("[CLEANUP] 훈련 종료 후 로컬 체크포인트 및 로그 정리 시작")
        # TensorBoard 로그·체크포인트 일괄 삭제
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


if __name__ == "__main__":

    args = args_util.get_args()

    if args.resume_model_from_wandb:
        # resume_model_from_wandb 가 "latest"가 아니면 에러를 발생시킵니다.
        if args.resume_model_from_wandb not in ['latest']:
            raise ValueError("args.resume_model_from_wandb must be 'latest.")
        if not args.name:
            raise ValueError(
                "args.name must be provided to resume from a wandb artifact.")

        print(
            f"Resuming from wandb artifact: {args.name}:{args.resume_model_from_wandb}"
        )

        # W&B Public API를 사용하여 아티팩트와 원본 Run의 config를 가져옵니다.
        api = wandb.Api()

        try:
            # resume_model_from_wandb 값('best', 'latest' 등)에 따라 컬렉션과 파일 이름을 결정합니다.
            resume_alias = args.resume_model_from_wandb
            if resume_alias == 'best':
                collection_name = f"{args.name}_best-model"
                checkpoint_filename = "best.pth"
            else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
                collection_name = f"{args.name}_latest-model"
                checkpoint_filename = "latest.pth"

            # Public API를 사용해 아티팩트를 가져오려면 entity와 project 정보가 필요합니다.
            # 임시 Run을 생성하여 컨텍스트(entity, project)를 얻어옵니다.
            temp_run_for_context = wandb.init(project="Diffusion-Planner",
                                              name=f"temp_api_run_{args.name}",
                                              job_type="api_access")
            # wandb.run.entity: jksg01019-naver-labs
            # wandb.run.project: Diffusion-Planner
            entity = temp_run_for_context.entity
            project = temp_run_for_context.project
            temp_run_for_context.finish()

            artifact_path = f"{entity}/{project}/{collection_name}:{resume_alias}"
            print(f"아티팩트 경로에서 가져오는 중: {artifact_path}")
            # Public API를 통해 아티팩트 객체를 가져옵니다.
            artifact = api.artifact(artifact_path, type='model')

            # 이 아티팩트를 생성한 원본 Run을 가져옵니다.
            source_run = artifact.logged_by()

            # 원본 Run의 config에서 save_path를 가져옵니다.
            if 'save_path' in source_run.config:
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
                    name=f"resume_run_download_{args.name}",
                    resume="allow")
                # 현재 Run에서 사용할 아티팩트를 지정합니다.
                artifact_for_download = download_run.use_artifact(
                    artifact, aliases=[resume_alias])
                # 원본 save_path에 아티팩트 파일을 다운로드합니다.
                artifact_for_download.download(root=save_path)
                download_run.finish()
                print(f"아티팩트 '{artifact_path}'을(를) {save_path}에 다운로드했습니다.")

                # 다운로드된 체크포인트의 전체 경로를 설정합니다.
                model_path = os.path.join(save_path, checkpoint_filename)
                if not os.path.exists(model_path):
                    downloaded_files = os.listdir(save_path)
                    raise FileNotFoundError(
                        f"다운로드된 아티팩트 디렉터리에서 '{checkpoint_filename}'을(를) 찾을 수 없습니다: {save_path}. "
                        f"사용 가능한 파일: {downloaded_files}")
                save_path = os.path.dirname(os.path.abspath(model_path))
                args.resume_local_path_model_path = save_path

                print(
                    f"아티팩트를 {save_path}에 다운로드했습니다. 체크포인트에서 학습을 재개합니다: {args.resume_local_path_model_path}"
                )
            else:
                raise ValueError("원본 Run의 config에 'save_path'가 없습니다.")

        except Exception as e:
            print(f"W&B에서 재개하는 동안 오류 발생: {e}")
            # 오류 발생 시 임시 Run이 종료되도록 보장합니다.
            if wandb.run:
                wandb.finish()
            raise e

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    # Run
    try:
        model_training(args)
    except BaseException:
        rank = int(os.environ.get("RANK", -1))
        print(f"\n[rank{rank}] Unhandled exception (printing full traceback):",
              file=sys.stderr,
              flush=True)
        traceback.print_exc()  # <-- 표준에러로 자세한 스택
        sys.stderr.flush()
        raise
