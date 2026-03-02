import os
import numpy as np
from torch import nn
from typing import Tuple
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.model.module.feasible import FeasibleProjector
# =====================================================================
from typing import Dict, Tuple, Optional, List, Any
import argparse
from dataclasses import dataclass
import math
import torch
from torch import nn
from typing import Tuple
from diffusion_planner.model.module.feasible import FeasibleProjector
from typing import Dict, List
import argparse
import torch
# =====================================================================
import time
from contextlib import contextmanager
from typing import Iterator

import time
from contextlib import contextmanager
from typing import Dict, Iterator
from typing import Any, List, Optional
import torch


def _get_cuda_inflight_step_limit(args: Any) -> int:
    """args에서 GPU in-flight step 제한 값을 읽어 안전한 정수로 만든다.

    목적:
        - args.cuda_inflight_step_limit 값을 읽어옵니다.
        - 0 이하이면 기능을 끕니다.
        - 1 이상이면 "동시에 GPU에서 처리 중인 step"의 최대 개수로 사용합니다.

    Args:
        args (Any):
            학습 설정 객체.
            - cuda_inflight_step_limit (int)가 있을 수 있습니다.

    Returns:
        int:
            - 0 이하: 비활성화
            - 1 이상: 최대 in-flight step 개수
    """
    try:
        return int(getattr(args, "cuda_inflight_step_limit", 0))
    except Exception:
        return 0


class _CudaInFlightStepLimiter:
    """GPU가 처리 중인 step이 너무 많이 쌓이지 않게 제한한다.

    배경:
        - 배치를 GPU로 옮길 때 non_blocking=True이면, CPU는 "복사 요청"만 걸고 바로 다음으로 넘어갈 수 있습니다.
        - CPU가 너무 빨리 다음 배치로 넘어가면, 아직 복사가 끝나지 않은 pinned 메모리(배치 버퍼)가
          여러 step 동안 동시에 남아 있을 수 있습니다.
        - 이 상태가 반복되면 pinned 메모리 사용량이 계속 올라갈 수 있습니다.

    해결 방식:
        - 매 step 끝에 CUDA 이벤트를 하나 기록합니다.
        - 이벤트가 max_inflight_steps 보다 많이 쌓이면,
          가장 오래된 이벤트 1개만 synchronize() 해서 GPU가 따라올 시간을 줍니다.
        - 즉, "매 step마다 전부 기다리는 것"이 아니라,
          "너무 앞서가기 시작할 때만 조금 기다리는" 방식입니다.

    Notes:
        - 이 로직은 학습 결과(손실/업데이트 값)를 바꾸지 않습니다.
        - CUDA가 아니거나, limit이 0 이하면 아무 것도 하지 않습니다.
    """

    def __init__(self, max_inflight_steps: int) -> None:
        """제한기 생성.

        Args:
            max_inflight_steps (int):
                동시에 GPU에서 처리 중인 step 최대 개수.
                - 1 이상이어야 의미가 있습니다.
        """
        self._max_inflight_steps: int = max(1, int(max_inflight_steps))
        self._events: List[torch.cuda.Event] = []

    def record_step_end_and_maybe_wait(self) -> None:
        """현재 step 끝에 이벤트를 기록하고, 필요하면 오래된 step 1개만 기다린다.

        Returns:
            None
        """
        if not torch.cuda.is_available():
            return

        evt = torch.cuda.Event(enable_timing=False)
        # 현재 CUDA stream에 "여기까지의 작업이 끝났는지"를 표시하는 이벤트를 기록합니다.
        evt.record()
        self._events.append(evt)

        # 너무 많이 쌓였으면, 가장 오래된 1개만 기다립니다.
        if len(self._events) > self._max_inflight_steps:
            oldest = self._events.pop(0)
            oldest.synchronize()

    def flush(self) -> None:
        """남아 있는 이벤트를 모두 기다려서 정리한다.

        Returns:
            None
        """
        if not torch.cuda.is_available():
            self._events.clear()
            return

        while self._events:
            self._events.pop(0).synchronize()


from typing import Callable, Iterable


def _should_use_cuda_prefetch(args: argparse.Namespace) -> bool:
    """CUDA prefetch를 사용할지 결정합니다.

    Args:
        args (argparse.Namespace):
            - args.device: "cuda" 또는 "cuda:0" 등
            - args.use_cuda_prefetch: (선택) True/False
            - args.pin_mem: (선택) pin_memory 사용 여부

    Returns:
        bool:
            - True: CUDA prefetch 사용
            - False: 기존 방식 사용
    """
    use_flag = bool(getattr(args, "use_cuda_prefetch", False))
    if not use_flag:
        return False

    dev_str = str(getattr(args, "device", "cpu")).lower()
    if not dev_str.startswith("cuda"):
        return False

    if not torch.cuda.is_available():
        return False

    # pin_memory가 꺼져 있으면 non_blocking이 기대만큼 안 먹을 수 있어서,
    # 그래도 동작은 하지만 효과가 약할 가능성이 큼
    return True


def _record_stream_for_nested(obj: Any, stream: "torch.cuda.Stream") -> None:
    """중첩 구조(dict/list/tuple) 안의 CUDA 텐서들에 record_stream을 적용합니다.

    목적:
        - prefetch stream에서 만들어진 CUDA 텐서가
          메인 stream에서 사용될 예정임을 알려서,
          stream 간 메모리 재사용 꼬임을 방지합니다.

    Args:
        obj (Any):
            torch.Tensor 또는 dict/list/tuple로 중첩된 구조
        stream (torch.cuda.Stream):
            메인에서 사용할 stream (보통 torch.cuda.current_stream())

    Returns:
        None
    """
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            obj.record_stream(stream)
        return

    if isinstance(obj, dict):
        for v in obj.values():
            _record_stream_for_nested(v, stream)
        return

    if isinstance(obj, (list, tuple)):
        for v in obj:
            _record_stream_for_nested(v, stream)
        return

    return


class CudaPreparedBatchPrefetcher:
    """다음 배치를 별도 CUDA stream에서 미리 GPU로 옮기는 프리패처.

    동작 방식:
        - DataLoader에서 다음 CPU 배치를 하나 미리 가져옵니다.
        - prefetch 전용 CUDA stream에서 _prepare_batch_for_device(...)를 실행해
          (inputs, outputs)를 GPU로 올려둡니다.
        - 학습 루프는 다음 step 시작 시점에 wait_stream만 걸고
          이미 준비된 (inputs, outputs)를 바로 사용합니다.

    메모리/성능 관점:
        - CPU pinned 메모리는 "복사가 끝날 때까지" 잡히는데,
          복사를 더 일찍 시작하면 해제 시점이 앞당겨질 수 있습니다.
        - 보통 step 시간은 유지되거나 개선될 가능성이 있습니다.
        - 대신 GPU에는 "다음 배치 1개"가 추가로 올라가므로 GPU 메모리는 증가합니다.
    """

    def __init__(
        self,
        data_loader: Iterable[Dict[str, Any]],
        device: str,
        prepare_fn: Callable[[Dict[str, Any], str],
                             Tuple[Dict[str, Any], Dict[str, torch.Tensor]]],
    ) -> None:
        self._data_loader = data_loader
        self._iter = iter(data_loader)
        self._device = str(device)
        self._prepare_fn = prepare_fn

        # 현재 device 기준 stream (DDP면 각 rank가 자기 device로 이미 set_device 되어 있어야 함)
        self._prefetch_stream: torch.cuda.Stream = torch.cuda.Stream()

        self._next_prepared: Optional[Tuple[Dict[str, Any],
                                            Dict[str, torch.Tensor]]] = None
        self._next_cpu_batch_ref: Optional[Dict[str, Any]] = None

        self._preload()

    def __iter__(self) -> "CudaPreparedBatchPrefetcher":
        return self

    def _preload(self) -> None:
        """다음 배치를 prefetch stream에서 GPU로 미리 올립니다."""
        try:
            cpu_batch = next(self._iter)
        except StopIteration:
            self._next_prepared = None
            self._next_cpu_batch_ref = None
            return

        # CPU batch 참조를 들고 있어야(조기 해제 방지) 복사가 안전합니다.
        self._next_cpu_batch_ref = cpu_batch

        with torch.cuda.stream(self._prefetch_stream):
            self._next_prepared = self._prepare_fn(cpu_batch, self._device)

    def __next__(self) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
        if self._next_prepared is None:
            raise StopIteration

        # 메인 stream이 prefetch stream의 작업이 끝날 때까지 기다립니다.
        torch.cuda.current_stream().wait_stream(self._prefetch_stream)

        prepared = self._next_prepared

        # 이 텐서들이 메인 stream에서 사용될 예정임을 기록(안전장치)
        _record_stream_for_nested(prepared, torch.cuda.current_stream())

        # CPU batch 참조 해제(다음 preload에서 갱신됨)
        self._next_cpu_batch_ref = None

        # 다음 배치를 다시 미리 올립니다.
        self._preload()
        return prepared


# name -> 호출 횟수 / 누적 시간(ms)
_PROFILE_CALL_COUNT: Dict[str, int] = {}
_PROFILE_TOTAL_MS: Dict[str, float] = {}


def _is_main_process() -> bool:
    """로그를 출력할 프로세스인지 확인합니다.

    Returns:
        bool:
            - True: rank 0(메인 프로세스)
            - False: 그 외
    """
    try:
        return int(ddp.get_rank()) == 0
    except Exception:
        # 단일 프로세스/분산 미초기화 등
        return True


# name -> 호출 횟수 / 누적 시간(ms)
# - total_* : 전체 호출 기준(워밍업 포함)
# - avg_*   : 평균 계산에 포함된 호출(워밍업 제외)
_PROFILE_TOTAL_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_TOTAL_MS: Dict[str, float] = {}

# 각 블록(name)별로 처음 N번 호출은 avg에서 제외
_PROFILE_WARMUP_CALLS: int = 30


@contextmanager
def profile_block(
    name: str,
    enabled: bool = True,
    device_type: str = "cuda",
    *,
    print_rank0_only: bool = True,
) -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력합니다(누적 평균 포함).

    변경점:
        - 각 name별로 최초 _PROFILE_WARMUP_CALLS번 호출은 avg 계산에서 제외합니다.
    """
    if not enabled:
        yield
        return

    is_cuda: bool = isinstance(device_type,
                               str) and device_type.startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0: float = time.perf_counter()
    yield

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms: float = (time.perf_counter() - t0) * 1000.0

    # (1) 전체 호출 카운트(워밍업 포함)
    total_prev: int = _PROFILE_TOTAL_CALL_COUNT.get(name, 0)
    total_now: int = total_prev + 1
    _PROFILE_TOTAL_CALL_COUNT[name] = total_now

    # (2) avg에 포함할지 결정: (name별) 처음 30번은 제외
    avg_cnt_prev: int = _PROFILE_AVG_CALL_COUNT.get(name, 0)
    avg_sum_prev: float = _PROFILE_AVG_TOTAL_MS.get(name, 0.0)

    if total_now > int(_PROFILE_WARMUP_CALLS):
        avg_cnt_now: int = avg_cnt_prev + 1
        avg_sum_now: float = avg_sum_prev + float(elapsed_ms)
        _PROFILE_AVG_CALL_COUNT[name] = avg_cnt_now
        _PROFILE_AVG_TOTAL_MS[name] = avg_sum_now
    else:
        avg_cnt_now = avg_cnt_prev
        avg_sum_now = avg_sum_prev

    avg_ms: float = (avg_sum_now /
                     float(avg_cnt_now)) if avg_cnt_now > 0 else 0.0

    if (not print_rank0_only) or _is_main_process():
        warmup_left = max(0, int(_PROFILE_WARMUP_CALLS) - total_now)
        print(
            f"[PROFILE] {name}: {elapsed_ms:.3f} ms | avg {avg_ms:.3f} ms | n={avg_cnt_now} | warmup_left={warmup_left}"
        )


def _run_backward_and_step_deepspeed(
    model: nn.Module,
    loss_tensor: torch.Tensor,
    *,
    enable_profile: bool,
    device_type: str,
) -> None:
    """DeepSpeed 경로에서 역전파+업데이트를 수행하고, 구간별 시간을 출력합니다.

    Args:
        model (nn.Module):
            DeepSpeedEngine(모델이면서 .backward/.step을 가진 객체)를 기대합니다.
        loss_tensor (torch.Tensor):
            최종 손실 스칼라 텐서.
            - shape: ()  (스칼라)
        enable_profile (bool):
            True면 구간별 시간을 출력합니다.
        device_type (str):
            "cuda" 또는 "cpu"
    """
    with profile_block(
            "train_epoch._backward_and_step.deepspeed.backward",
            enabled=enable_profile,
            device_type=device_type,
    ):
        model.backward(loss_tensor)

    with profile_block(
            "train_epoch._backward_and_step.deepspeed.step",
            enabled=enable_profile,
            device_type=device_type,
    ):
        model.step()


def _run_backward_and_step_pytorch(
    model: nn.Module,
    loss_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    max_grad_norm: float,
    enable_profile: bool,
    device_type: str,
) -> None:
    """일반(PyTorch/DDP) 경로에서 역전파+클리핑+업데이트를 수행하고 구간별 시간을 출력합니다.

    Args:
        model (nn.Module): 학습 중인 모델.
        loss_tensor (torch.Tensor):
            최종 손실 스칼라 텐서.
            - shape: ()  (스칼라)
        optimizer (torch.optim.Optimizer): 옵티마이저.
        scheduler (Any): 스케줄러(있으면 step() 호출).
        max_grad_norm (float): 0보다 크면 기울기 크기 제한을 수행합니다.
        enable_profile (bool): True면 구간별 시간을 출력합니다.
        device_type (str): "cuda" 또는 "cpu"
    """
    with profile_block(
            "train_epoch._backward_and_step.pytorch.backward",
            enabled=enable_profile,
            device_type=device_type,
    ):
        loss_tensor.backward()

    if float(max_grad_norm) > 0.0:
        with profile_block(
                "train_epoch._backward_and_step.pytorch.clip_grad_norm",
                enabled=enable_profile,
                device_type=device_type,
        ):
            nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))

    with profile_block(
            "train_epoch._backward_and_step.pytorch.optimizer_step",
            enabled=enable_profile,
            device_type=device_type,
    ):
        optimizer.step()

    if scheduler is not None:
        with profile_block(
                "train_epoch._backward_and_step.pytorch.scheduler_step",
                enabled=enable_profile,
                device_type=device_type,
        ):
            scheduler.step()


def _maybe_run_torch_profiler_for_backward_step(
    args: Any,
    *,
    use_deepspeed: bool,
    model: nn.Module,
    loss_tensor: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    max_grad_norm: float,
) -> bool:
    """(선택) torch.profiler로 backward/step 내부를 더 자세히 캡처합니다.

    이 기능은 “연산”과 “GPU간 통신”이 무엇 때문에 오래 걸리는지(예: NCCL 통신, 특정 backward 연산 등)
    더 깊게 보고 싶을 때 사용합니다.

    켜는 방법(예시)
        args.profile_backward_torch = True
        args.profile_backward_torch_steps = 2   # 처음 2 step만 캡처
        args.profile_backward_torch_row_limit = 30

    Returns:
        bool:
            - True: 이번 호출에서 profiler를 실제로 실행함
            - False: 실행 안 함
    """
    enabled: bool = bool(getattr(args, "profile_backward_torch", False))
    if not enabled:
        return False

    step_idx: int = int(getattr(args, "_global_update_step", 0))
    max_steps: int = int(getattr(args, "profile_backward_torch_steps", 1))
    if step_idx >= max_steps:
        return False

    # torch.profiler는 모든 rank에서 동일하게 도는 편이 안전합니다(속도는 느려져도 “멈춤” 위험을 줄임).
    from torch.profiler import ProfilerActivity, profile

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
            activities=activities,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
    ) as prof:
        if use_deepspeed:
            model.backward(loss_tensor)
            model.step()
        else:
            loss_tensor.backward()
            if float(max_grad_norm) > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(),
                                         float(max_grad_norm))
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    # 출력은 rank 0만
    if _is_main_process():
        row_limit: int = int(
            getattr(args, "profile_backward_torch_row_limit", 30))
        print(
            "\n==================== [torch.profiler] backward/step top CUDA ===================="
        )
        try:
            print(prof.key_averages().table(sort_by="self_cuda_time_total",
                                            row_limit=row_limit))
        except Exception:
            # CUDA가 없거나 버전에 따라 컬럼명이 다를 수 있음
            print(prof.key_averages().table(sort_by="cpu_time_total",
                                            row_limit=row_limit))
        print(
            "=================================================================================\n"
        )

    return True


def _move_batch_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """배치 dict에서 torch.Tensor만 device로 옮기고, 나머지(None 포함)는 그대로 둔다."""
    agent_route_lane_order_agent_num = None
    neighbor_agents_agent_num = None

    batch_on_device: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch_on_device[key] = value.to(device, non_blocking=True)
        else:
            batch_on_device[key] = value

        if key == "agent_route_lane_order" and isinstance(
                batch_on_device[key], torch.Tensor):
            agent_route_lane_order_agent_num = int(
                batch_on_device[key].shape[1])
        elif key == "neighbor_agents_past" and isinstance(
                batch_on_device[key], torch.Tensor):
            neighbor_agents_agent_num = int(batch_on_device[key].shape[1])

    if agent_route_lane_order_agent_num is not None and neighbor_agents_agent_num is not None:
        assert agent_route_lane_order_agent_num == neighbor_agents_agent_num, \
            f"agent_route_lane_order agent num ({agent_route_lane_order_agent_num}) " \
            f"!= neighbor_agents_past agent num ({neighbor_agents_agent_num})"

    return batch_on_device


def assert_cur_future_valid_mask_np(
    valid_bpt: np.ndarray,
    context: str = "savgol_filter_for_control",
) -> None:
    """
    유효 마스크가 각 (b,p) 행마다 True*False* (단조 감소)인지 검증 (NumPy 버전).

    Args:
        valid_bpt (np.ndarray):
            shape: (B, Pnn, T1)
            dtype: bool 권장 (또는 0/1 int/float 등도 허용)
            - True(또는 1): 유효
            - False(또는 0): 무효
        context (str):
            에러 메시지에 표시할 호출 위치 문자열.

    Raises:
        ValueError:
            0→1 전이(False→True)가 하나라도 발견되면 발생합니다.
            (예: True, False, True ... / False, True ... 형태는 금지)
    """
    v0 = np.asarray(valid_bpt)

    if v0.ndim != 3:
        raise AssertionError("valid_bpt는 (B,Pnn,T1) 여야 합니다.", v0.shape)

    B, Pnn, T1 = v0.shape
    if Pnn == 0 or T1 <= 1:
        # 검사할 행이 없거나, 시간축이 1 이하이면 0→1 전이를 정의하기 어려우므로 통과
        return

    # bool이 아니면 0/비0 기준으로 bool 변환
    v_bool = v0 if v0.dtype == np.bool_ else (v0 != 0)

    # (B*Pnn, T1) int8로 변환
    v = v_bool.reshape(-1, T1).astype(np.int8)

    # d[t] = v[t+1] - v[t], 0→1이면 +1
    d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
    has_01 = (d > 0).any(axis=1)  # (B*Pnn,)

    if np.any(has_01):
        bad_idx = np.nonzero(has_01)[0]  # (N_bad,)
        max_show = min(int(bad_idx.size), 8)
        bad_idx_sample = bad_idx[:max_show]

        b_list = (bad_idx_sample // Pnn).tolist()
        p_list = (bad_idx_sample % Pnn).tolist()
        print("valid_bpt:", valid_bpt)
        raise ValueError(
            f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*). \n"
            f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.size)},  \n"
            f"example (b,p)={list(zip(b_list, p_list))}.  \n"
            f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
        )


def _prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """배치 dict를 GPU/CPU로 옮기고, 모델 상한에 맞게 축을 잘라 입력/정답을 나눈다.

    전제: collate_fn(DiffusionPlannerCollate)에서 이미
      - neighbor_agents_past         : (B, data_max_agent_num, time_len, 11)
      - near_future_gt_3_dim         : (B, data_max_agent_num, future_len, 3)
      - static_objects               : (B, data_max_static_num, 10)
      - lanes / lanes_*              : (B, data_max_lane_num, lane_len, ·)
      - route_lanes / route_lanes_*  : (B, data_max_route_num, route_len, ·)
      - agent_route_lane_order       : (B, data_max_agent_num, data_max_lane_num)
    형태로 패딩이 끝난 상태라고 가정한다.

    여기서는 (do_not_clip=False 인 경우에만)
      - neighbor_agents_past / near_future_gt_3_dim: agent 축 → max_agent_num
      - lanes / lanes_*                            : lane  축 → max_lane_num
      - route_lanes / route_lanes_*                : lane  축 → max_lane_num
      - agent_route_lane_order                     : (agent, lane) 축 → (, max_lane_num)
      - static_objects                              : 그대로 유지

    를 수행한다.
    """
    batch_on_device: Dict[str, Any] = _move_batch_to_device(batch, device)
    aro = batch_on_device.get("agent_route_lane_order", None)
    if isinstance(aro, torch.Tensor) and aro.dtype != torch.long:
        batch_on_device["agent_route_lane_order"] = aro.to(torch.long)

    # outputs 분리 (정답은 반드시 Tensor여야 함)
    target_keys = {
        "ego_future_gt_4_dim",  # output 에서만 꺼내도록
        "near_future_gt_4_dim",  # output 에서만 꺼내도록
        "future_seg_control_gt_3_dim",  # output 에서만 꺼내도록
        "ego_future_gt_is_valid",  # output 에서만 꺼내도록
        "near_future_gt_is_valid",  # output 에서만 꺼내도록
        "future_seg_control_is_valid",  # output 에서만 꺼내도록
        # "ego_future_gt_11_dim", # (B, future_len, 11)
        # "neighbor_future_gt_11_dim", # (B, Pnn, future_len, 11)
    }
    outputs: Dict[str, torch.Tensor] = {}
    for key in list(batch_on_device.keys()):
        if key in target_keys:
            # pop으로 꺼내자.
            value = batch_on_device.pop(key, None)
            if value is None:
                continue
            # value = batch_on_device.pop(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"target '{key}' must be torch.Tensor, got {type(value)}")
            outputs[key] = value
    inputs: Dict[str, Any] = batch_on_device
    return inputs, outputs


# =====================================================================
# 아래부터 train_epoch 내부를 역할별 함수로 분리
# =====================================================================


def _as_bool_mask(mask: torch.Tensor) -> torch.Tensor:
    """마스크 텐서를 bool로 통일한다.

    Args:
        mask: 마스크 텐서. shape: (...,)

    Returns:
        bool 마스크. shape: mask.shape
    """
    if mask.dtype == torch.bool:
        return mask
    return mask > 0


def _restore_padding_values_inplace(
    inputs: Dict[str, torch.Tensor],
    pad_masks: Dict[str, torch.Tensor],
) -> None:
    """pad_masks로 지정된 위치를 다시 0으로 되돌린다.

    Args:
        inputs: 입력 dict. (in-place 수정)
        pad_masks: feature_key -> pad_mask(bool)

    Returns:
        None
    """
    for feature_key, pad_mask in pad_masks.items():
        if feature_key not in inputs:
            continue
        t = inputs[feature_key]
        if not isinstance(t, torch.Tensor):
            continue

        if t.dtype == torch.bool:
            inputs[feature_key] = t.masked_fill(pad_mask, False)
        elif torch.is_floating_point(t):
            inputs[feature_key] = t.masked_fill(pad_mask, 0.0)
        else:
            inputs[feature_key] = t.masked_fill(pad_mask, 0)


def _build_near_future_4dim_and_mask(
    near_future_gt_3_dim: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """yaw를 cos/sin으로 확장하고, 비어 있는 구간 마스크를 만든다.

    처리 내용:
      1) (x, y, yaw) 궤적에서 yaw를 cos/sin 두 값으로 나눠
         (x, y, cos, sin) 4차원 궤적을 만든다.
      2) (x, y, yaw)가 모두 0인 프레임을 "비어 있는 프레임"으로 보고
         near_future_mask 를 True/False로 만든다.
      3) 비어 있는 위치의 4차원 값은 0으로 덮어쓴다.

    Args:
        near_future_gt_3_dim:
            - shape: (B, agent_num, future_len, 3)
            - [..., 0:2] = (x, y), [..., 2] = yaw.

    Returns:
        near_future_gt_4_dim:
            - shape: (B, agent_num, future_len, 4)
            - [x, y, cos(yaw), sin(yaw)].
        near_future_mask:
            - shape: (B, agent_num, future_len)
            - True 이면 해당 프레임은 완전히 비어 있는 프레임.
    """
    # near_future_gt_3_dim: (B, A, Tf, 3)
    near_future_mask: torch.Tensor = torch.sum(
        torch.ne(near_future_gt_3_dim[..., :3], 0),
        dim=-1,
    ) == 0  # (B, A, Tf)

    # yaw: (B, A, Tf)
    yaw: torch.Tensor = near_future_gt_3_dim[..., 2]

    # near_future_gt_4_dim: (B, A, Tf, 4)
    near_future_gt_4_dim: torch.Tensor = torch.cat(
        [
            near_future_gt_3_dim[..., :2],  # (B, A, Tf, 2)
            torch.stack(
                [
                    yaw.cos(),  # (B, A, Tf)
                    yaw.sin(),  # (B, A, Tf)
                ],
                dim=-1,  # -> (B, A, Tf, 2)
            ),
        ],
        dim=-1,
    )

    near_future_gt_4_dim[near_future_mask] = 0.0

    if not torch.isfinite(near_future_gt_4_dim).all():
        raise ValueError("Non-finite values detected in near_future_gt_4_dim")

    return near_future_gt_4_dim, near_future_mask


def _compute_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    batch_num_in_all_epoch: int,
) -> Dict[str, torch.Tensor]:
    """diffusion 손실과 feasible 가중합까지 포함한 loss_dict dict 를 계산한다.

    Args:
        args:
            - _global_update_step: 현재까지 진행된 전체 스텝 수.
            - train_epochs 등은 batch_num_in_all_epoch 계산에 이미 반영되어 있음.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        norm_inputs:
            관측값 정규화가 적용된 입력 dict. 각 텐서 shape: (B, ...).
        near_future_gt_4_dim:
            - shape: (B, agent_num, future_len, 4)
            - [x, y, cos(yaw), sin(yaw)].
        near_future_mask:
            - shape: (B, agent_num, future_len)
            - True 인 위치는 완전히 비어 있는 프레임.

    Returns:
        loss_dict:
            - 다양한 부분 손실과 최종 합(loss_dict["loss"])를 담은 dict.
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

    # 진행도 0~1 계산
    progress: float = min(
        1.0,
        args._global_update_step / float(max(1, batch_num_in_all_epoch - 1)),
    )
    w_dir, w_int, w_const = FeasibleProjector.loss_weights_by_progress(
        progress, args)
    # 진행도/가중치 기록(평균 로그용)
    loss_dict["learn_progress"] = torch.tensor(float(progress),
                                               device=next(
                                                   model.parameters()).device)
    loss_dict["direct_loss_weight"] = torch.tensor(
        float(w_dir), device=next(model.parameters()).device)
    loss_dict["int_loss_weight"] = torch.tensor(float(w_int),
                                                device=next(
                                                    model.parameters()).device)
    loss_dict["const_loss_weight"] = torch.tensor(
        float(w_const), device=next(model.parameters()).device)

    # 개별 손실이 없을 수도 있으니 기본값 0 텐서로 처리
    device = norm_inputs["ego_agent_past"].device
    l_dir = loss_dict.get("neighbor_prediction_loss",
                          torch.tensor(0.0, device=device))
    l_int = loss_dict.get("integration_loss", torch.tensor(0.0, device=device))
    l_con = loss_dict.get("constraint_loss", torch.tensor(0.0, device=device))

    # 최종 손실 합성
    loss_dict["loss"] = w_dir * l_dir + w_int * l_int + w_const * l_con

    return loss_dict


def _backward_and_step(
    loss_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    args: Any,
) -> None:
    """역전파/업데이트를 수행합니다. (로그용 CPU loss 변환은 하지 않습니다)

    핵심 변경:
        - 기존에는 여기서 loss를 매 step마다 CPU float로 만들었습니다(.cpu().item()).
        - 이제는 학습은 그대로 수행하고, "로그용 숫자 변환"은 epoch 끝에 한 번만 합니다.

    Args:
        loss_dict (Dict[str, torch.Tensor]):
            - loss_dict["loss"] 는 스칼라 텐서여야 합니다.
              shape: ()
        model (nn.Module):
            - DeepSpeed 사용 시: .backward/.step을 가진 엔진
            - 그 외: 일반 nn.Module
        optimizer (torch.optim.Optimizer):
            - 일반 PyTorch 경로에서 사용
        scheduler (Any):
            - 일반 PyTorch 경로에서 step() 호출
        args (Any):
            - 학습 설정/상태

    Returns:
        None
    """
    loss_tensor: torch.Tensor = loss_dict["loss"]  # shape: ()
    device_type: str = "cuda" if loss_tensor.is_cuda else "cpu"

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False)) \
        and hasattr(model, "backward") and hasattr(model, "step")

    enable_profile: bool = bool(getattr(args, "profile_backward_detail", False))
    max_grad_norm: float = float(getattr(args, "max_grad_norm", 0.0))

    ran_torch_prof: bool = _maybe_run_torch_profiler_for_backward_step(
        args,
        use_deepspeed=use_deepspeed,
        model=model,
        loss_tensor=loss_tensor,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=max_grad_norm,
    )

    if not ran_torch_prof:
        if use_deepspeed:
            _run_backward_and_step_deepspeed(
                model=model,
                loss_tensor=loss_tensor,
                enable_profile=enable_profile,
                device_type=device_type,
            )
        else:
            _run_backward_and_step_pytorch(
                model=model,
                loss_tensor=loss_tensor,
                optimizer=optimizer,
                scheduler=scheduler,
                max_grad_norm=max_grad_norm,
                enable_profile=enable_profile,
                device_type=device_type,
            )


def _apply_weight_decay_warmdown(optimizer: torch.optim.Optimizer) -> None:
    """각 파라미터 그룹의 lr 비율에 맞춰 weight_decay 를 선형으로 조정한다.

    Args:
        optimizer:
            AdamW 옵티마이저. 각 param_group 에 "wd_max", "lr_max" 키가 설정되어 있어야 한다.
    """
    for pg in optimizer.param_groups:
        wd_max = pg.get("wd_max", None)
        lr_max = pg.get("lr_max", None)
        if wd_max is None or lr_max is None:
            continue
        if wd_max > 0.0 and lr_max > 0.0:
            pg["weight_decay"] = wd_max * (pg["lr"] / lr_max)


def _update_ema_if_needed(ema: Optional[object], model: nn.Module) -> None:
    """EMA 객체가 있으면 한 스텝 업데이트한다.

    Args:
        ema:
            timm.utils.ModelEma 와 같은 EMA 래퍼 또는 None.
        model:
            현재 학습 중인 모델. ema.update(model) 의 인자로 사용된다.
    """
    if ema is not None:
        # DDP / DeepSpeed 래퍼가 씌워져 있으면 .module을 사용해 실제 모듈 기준으로 EMA를 업데이트한다.
        src_model: nn.Module = getattr(model, "module", model)
        ema.update(src_model)


def _to_float_scalar_tensor(
    value: Any,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """값을 float32 스칼라 텐서로 바꿉니다.

    목적:
        - loss_dict 안의 값을 "누적합(sum)"에 더하기 쉬운 형태로 통일합니다.
        - CPU로 가져오지 않고, 같은 device(GPU/CPU) 위에서만 처리합니다.

    Args:
        value (Any):
            - 보통 torch.Tensor 를 기대합니다.
            - int/float 도 허용합니다.
        device (torch.device):
            - 결과 텐서를 둘 device.
            - 예: torch.device("cuda") 또는 torch.device("cpu")

    Returns:
        Optional[torch.Tensor]:
            - shape: ()  (스칼라)
            - dtype: torch.float32
            - 변환 불가능한 타입이면 None

    Note:
        - value가 텐서인데 원소가 1개가 아니면(mean으로) 스칼라로 만듭니다.
          (예: shape이 (N,) 이면 mean() 후 shape () )
    """
    if isinstance(value, torch.Tensor):
        t: torch.Tensor = value.detach()
        if t.numel() != 1:
            # 여러 값이면 평균으로 스칼라 하나로 만듭니다.
            t = t.mean()
        return t.to(device=device, dtype=torch.float32)

    if isinstance(value, (int, float)):
        return torch.tensor(float(value), device=device, dtype=torch.float32)

    return None


def _accumulate_epoch_loss_sums_inplace(
    loss_sums: Dict[str, torch.Tensor],
    loss_counts: Dict[str, torch.Tensor],
    loss_dict: Dict[str, Any],
    *,
    device: torch.device,
) -> None:
    """step 단위 loss_dict를 에폭 누적합(sum)과 횟수(count)에 더합니다. (in-place)

    Args:
        loss_sums (Dict[str, torch.Tensor]):
            - key -> 누적합 텐서
            - 각 텐서 shape: () (스칼라), dtype: float32
        loss_counts (Dict[str, torch.Tensor]):
            - key -> 해당 key가 등장한 횟수
            - 각 텐서 shape: () (스칼라), dtype: float32
        loss_dict (Dict[str, Any]):
            - 이번 step에서 계산된 loss 들
            - 보통 value는 torch.Tensor(스칼라)입니다.
        device (torch.device):
            - 누적을 수행할 device

    Returns:
        None

    Note:
        - 어떤 key가 어떤 step에 없으면, 그 step은 그 key 평균 계산에 포함되지 않습니다.
          (기존 get_epoch_mean_loss 동작과 같은 의미)
    """
    with torch.no_grad():
        for key, value in loss_dict.items():
            scalar = _to_float_scalar_tensor(value, device=device)
            if scalar is None:
                continue

            if key not in loss_sums:
                loss_sums[key] = torch.zeros((),
                                             device=device,
                                             dtype=torch.float32)
                loss_counts[key] = torch.zeros((),
                                               device=device,
                                               dtype=torch.float32)

            loss_sums[key] += scalar
            loss_counts[key] += 1.0


def _ddp_get_union_keys(local_keys: List[str]) -> List[str]:
    """DDP에서 rank 전체의 key 목록 합집합(union)을 만듭니다.

    Args:
        local_keys (List[str]):
            - 현재 rank에서 관측된 key 목록

    Returns:
        List[str]:
            - 전체 rank에서의 key 합집합(정렬됨)
    """
    keys_sorted: List[str] = sorted(local_keys)

    if not ddp.is_dist_avail_and_initialized():
        return keys_sorted

    import torch.distributed as dist

    # 가능하면 all_gather_object로 각 rank의 key 리스트를 모아서 union을 만듭니다.
    if hasattr(dist, "all_gather_object"):
        gathered: List[List[str]] = [[] for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, keys_sorted)

        union_set: set = set()
        for ks in gathered:
            union_set.update(ks)
        return sorted(union_set)

    # fallback: object gather가 없으면 "키가 모든 rank에서 동일"하다고 가정합니다.
    return keys_sorted


def _finalize_epoch_mean_loss(
    loss_sums: Dict[str, torch.Tensor],
    loss_counts: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> Dict[str, float]:
    """에폭 누적합/횟수로 평균을 만들고, (DDP면) rank 전체 평균으로 맞춘 뒤 CPU float로 반환합니다.

    Args:
        loss_sums (Dict[str, torch.Tensor]):
            - key -> sum 텐서 (shape: ())
        loss_counts (Dict[str, torch.Tensor]):
            - key -> count 텐서 (shape: ())
        args (argparse.Namespace):
            - args.ddp 가 True이면 DDP all_reduce 수행
        device (torch.device):
            - 통신/계산에 사용할 device

    Returns:
        Dict[str, float]:
            - key -> epoch 평균(파이썬 float)

    Note:
        - DDP일 때:
            sum들과 count들을 SUM으로 합친 뒤, mean = sum / count 로 만듭니다.
        - CPU로의 복사는 epoch 끝에 "한 번"만 일어나도록 벡터로 묶어서 가져옵니다.
    """
    if len(loss_sums) == 0:
        return {}

    union_keys: List[str] = _ddp_get_union_keys(list(loss_sums.keys()))

    # 모든 rank에서 같은 key 순서를 가지도록, 없는 키는 0으로 채웁니다.
    for k in union_keys:
        if k not in loss_sums:
            loss_sums[k] = torch.zeros((), device=device, dtype=torch.float32)
        if k not in loss_counts:
            loss_counts[k] = torch.zeros((), device=device, dtype=torch.float32)

    # 벡터로 묶어서 통신 횟수를 줄입니다.
    # sum_vec: (K,), count_vec: (K,)
    sum_vec: torch.Tensor = torch.stack([loss_sums[k] for k in union_keys],
                                        dim=0).to(device)
    count_vec: torch.Tensor = torch.stack([loss_counts[k] for k in union_keys],
                                          dim=0).to(device)

    if bool(getattr(args, "ddp",
                    False)) and ddp.is_dist_avail_and_initialized():
        import torch.distributed as dist
        dist.all_reduce(sum_vec, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_vec, op=dist.ReduceOp.SUM)

    safe_count: torch.Tensor = torch.clamp(count_vec, min=1.0)  # (K,)
    mean_vec: torch.Tensor = sum_vec / safe_count  # (K,)

    # CPU로 한 번에 가져옵니다. (2, K)
    packed_cpu: torch.Tensor = torch.stack([mean_vec, count_vec],
                                           dim=0).detach().cpu()
    mean_list: List[float] = packed_cpu[0].tolist()
    count_list: List[float] = packed_cpu[1].tolist()

    epoch_mean_loss: Dict[str, float] = {}
    for k, m, c in zip(union_keys, mean_list, count_list):
        # count가 0이면(어느 rank에서도 한 번도 안 나온 key) 결과에서 제외
        if float(c) <= 0.0:
            continue
        epoch_mean_loss[k] = float(m)

    return epoch_mean_loss


def train_epoch(
    data_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ema: Optional[object],
    scheduler,
    batch_num_in_all_epoch: int,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행합니다.

    변경점:
        - cuda_inflight_step_limit 기능 제거
        - use_cuda_prefetch(CUDA prefetch) 기능 제거
        - 항상 "기존 방식"으로 배치를 받아서 학습합니다.

    Returns:
        Tuple[Dict[str, float], float]:
            - epoch_mean_loss: 손실 항목별 epoch 평균 dict
            - epoch_mean_loss.get("loss"): 최종 손실 값 (없으면 NaN)
    """
    model.train()


    stat_device: torch.device = torch.device(args.device)
    epoch_loss_sums: Dict[str, torch.Tensor] = {}
    epoch_loss_counts: Dict[str, torch.Tensor] = {}

    # ✅ (추가) NPC/ego current state 증강기 1회 생성(매 step 새로 만들지 않음)
    npc_state_augmenter: Optional[NPCStatePerturbation] = None
    use_npc_data_augment = bool(getattr(args, "use_npc_data_augment", False))
    if use_npc_data_augment:
        npc_state_augmenter = NPCStatePerturbation(
            dt=float(getattr(args, "feasible_stride_dt", 0.1)),
            time_len=int(getattr(args, "time_len", 21)),
            future_len=int(getattr(args, "future_len", 80)),
        )

    for batch in data_loader:
        inputs, outputs = _prepare_batch_for_device(batch, device=args.device)
        # ✅ (추가) 정규화 전에 증강 적용
        # ✅ (추가) 정규화 전에 증강 적용
        if npc_state_augmenter is not None:
            # ---- 시각화 on/off ----
            enable_vis: bool = False  # train_epoch에서만 결정

            if enable_vis and _is_main_process():
                debug_vis_dir: Optional[str] = os.environ.get(
                    "DP_NPC_AUG_VIS_DIR", "").strip()
                if debug_vis_dir == "":
                    debug_vis_dir = os.path.join(
                        str(getattr(args, "save_path", ".")), "npc_aug_vis")

                debug_step: Optional[int] = int(
                    getattr(args, "_global_update_step", 0))
                debug_max_scenes: int = int(
                    os.environ.get("DP_NPC_AUG_VIS_MAX_SCENES", "20"))
                debug_every_n_steps: int = int(
                    os.environ.get("DP_NPC_AUG_VIS_EVERY", "1"))
                debug_past_stride: int = int(
                    os.environ.get("DP_NPC_AUG_VIS_PAST_STRIDE", "1"))
                debug_future_stride: int = int(
                    os.environ.get("DP_NPC_AUG_VIS_FUTURE_STRIDE", "1"))
                debug_vel_arrow_len_m: float = float(
                    os.environ.get("DP_NPC_AUG_VIS_VEL_ARROW_LEN_M", "1.0"))
            else:
                # ✅ 완전 OFF: apply_inplace 내부에서 debug 관련 분기 자체가 절대 실행되지 않게
                debug_vis_dir = None
                debug_step = None
                debug_max_scenes = 0
                debug_every_n_steps = 1
                debug_past_stride = 1
                debug_future_stride = 1
                debug_vel_arrow_len_m = 1.0

            npc_state_augmenter.apply_inplace(
                inputs=inputs,
                outputs=outputs,
                augment_prob=float(getattr(args, "augment_prob", 0.5)),
                agent_prob=float(getattr(args, "augment_prob", 0.5)),
                use_body_vel=bool(getattr(args, "use_body_vel", True)),

                # ---- debug vis args ----
                debug_vis_dir=debug_vis_dir,
                debug_step=debug_step,
                debug_max_scenes=debug_max_scenes,
                debug_every_n_steps=debug_every_n_steps,
                debug_past_stride=debug_past_stride,
                debug_future_stride=debug_future_stride,
                debug_vel_arrow_len_m=debug_vel_arrow_len_m,
            )
        # 1) 관측 정규화
        norm_inputs: Dict[str,
                          torch.Tensor] = args.observation_normalizer(inputs)

        outputs["ego_future_gt_4_dim"] = args.state_normalizer(
            data=outputs["ego_future_gt_4_dim"],
            valid_mask=outputs["ego_future_gt_is_valid"],
        )
        outputs["near_future_gt_4_dim"] = args.state_normalizer(
            data=outputs["near_future_gt_4_dim"],
            valid_mask=outputs["near_future_gt_is_valid"],
        )
        outputs["future_seg_control_gt_3_dim"] = args.state_normalizer(
            data=outputs["future_seg_control_gt_3_dim"],
            valid_mask=outputs["future_seg_control_is_valid"],
        )

        # 2) grad 초기화
        if args.use_deepspeed and hasattr(model, "zero_grad"):
            model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)

        base_model = ddp.get_model(model, args.ddp)
        sde_marginal_prob = base_model.sde.marginal_prob

        # 3) loss
        raw_loss_dict: Dict[str, torch.Tensor] = {}
        raw_loss_dict, _ = diffusion_loss_func(
            args=args,
            model=model,
            norm_inputs=norm_inputs,
            norm_outputs=outputs,
            marginal_prob=sde_marginal_prob,
            state_normalizer=args.state_normalizer,
            loss_dict=raw_loss_dict,
            model_type=args.diffusion_model_type,
        )

        loss_dict: Dict[str, torch.Tensor] = _compute_loss_dict(
            loss_dict=raw_loss_dict,
            args=args,
            model=model,
            norm_inputs=norm_inputs,
            batch_num_in_all_epoch=batch_num_in_all_epoch,
        )

        # 4) 통계 누적
        _accumulate_epoch_loss_sums_inplace(
            loss_sums=epoch_loss_sums,
            loss_counts=epoch_loss_counts,
            loss_dict=loss_dict,
            device=stat_device,
        )

        # 5) backward + step
        enable_profile: bool = bool(getattr(args, "profile_feasible", False))
        if enable_profile and _is_main_process():
            print("===============[PROFILE train_epoch ENABLED]===============")

        device_type: str = ("cuda" if "cuda" in str(args.device) else "cpu")
        with profile_block(
                "train_epoch.train_epoch._backward_and_step",
                enabled=enable_profile,
                device_type=device_type,
        ):
            _backward_and_step(
                loss_dict=loss_dict,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )

        _apply_weight_decay_warmdown(optimizer)
        _update_ema_if_needed(ema, model)
        if args.ddp:
            torch.cuda.synchronize()
        args._global_update_step += 1

    epoch_mean_loss: Dict[str, float] = _finalize_epoch_mean_loss(
        loss_sums=epoch_loss_sums,
        loss_counts=epoch_loss_counts,
        args=args,
        device=stat_device,
    )

    if ddp.get_rank() == 0 and "loss" in epoch_mean_loss:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss.get("loss", float("nan"))
