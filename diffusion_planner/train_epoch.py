from tqdm import tqdm
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

    is_cuda: bool = isinstance(device_type, str) and device_type.startswith(
        "cuda")
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

    avg_ms: float = (
                avg_sum_now / float(avg_cnt_now)) if avg_cnt_now > 0 else 0.0

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
            batch_on_device[key] = value  # None / metadata 등 유지
        if key == "agent_route_lane_order":
            agent_route_lane_order_agent_num = int(
                batch_on_device[key].shape[1])
        elif key == "neighbor_agents_past":
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
    if isinstance(aro, torch.Tensor):
        batch_on_device["agent_route_lane_order"] = aro.long()

    # outputs 분리 (정답은 반드시 Tensor여야 함)
    target_keys = {
        "ego_future_gt_4_dim",  # output 에서만 꺼내도록
        "near_future_gt_4_dim",  # output 에서만 꺼내도록
        "ego_future_gt_is_valid",  # input / output 둘다
        "near_future_gt_is_valid",  # input / output 둘다
    }
    outputs: Dict[str, torch.Tensor] = {}
    for key in list(batch_on_device.keys()):
        if key in target_keys:
            # TODO: 나중에 pop으로 바꾸기
            value = batch_on_device[key]
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
        near_future_gt_3_dim: torch.Tensor, ) -> Tuple[
    torch.Tensor, torch.Tensor]:
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

    if not args.use_direct_loss:
        w_dir = 0.0
        w_int = 1.0

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
) -> float:
    """역전파/업데이트를 수행하고, (선택) 내부 시간을 더 잘게 출력합니다.

    Args:
        loss_dict (Dict[str, torch.Tensor]):
            - loss_dict["loss"]는 스칼라 텐서여야 합니다.
              shape: ()
        model (nn.Module):
            - DeepSpeed 사용 시: .backward/.step을 가진 엔진
            - 그 외: 일반 nn.Module
        optimizer (torch.optim.Optimizer): 옵티마이저(DeepSpeed면 내부에서 사용될 수 있음)
        scheduler (Any): 스케줄러(DeepSpeed면 내부에서 처리, 일반이면 여기서 step)
        args (Any): 학습 설정/상태

    Returns:
        float: loss 값(로그용)
    """
    loss_tensor: torch.Tensor = loss_dict["loss"]  # shape: ()
    device_type: str = "cuda" if loss_tensor.is_cuda else "cpu"

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False)) \
                          and hasattr(model, "backward") and hasattr(model,
                                                                     "step")

    # ✅ 내부 구간 프로파일링 on/off (기본은 False)
    enable_profile: bool = bool(getattr(args, "profile_backward_detail", False))
    enable_profile: True

    max_grad_norm: float = float(getattr(args, "max_grad_norm", 0.0))

    # (선택) torch.profiler: 더 깊게 보고 싶을 때만
    ran_torch_prof: bool = _maybe_run_torch_profiler_for_backward_step(
        args,
        use_deepspeed=use_deepspeed,
        model=model,
        loss_tensor=loss_tensor,
        optimizer=optimizer,
        scheduler=scheduler,
        max_grad_norm=max_grad_norm,
    )

    # torch.profiler를 안 돌린 경우에만, 우리가 쪼갠 profile_block 계측 실행
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

    # 로그용 loss 값(스칼라). CPU로 가져오는 순간 동기화가 일어날 수 있습니다.
    total_loss: float = float(loss_tensor.detach().float().cpu().item())
    return total_loss


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


# =====================================================================


def train_epoch(
        data_loader,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        args: argparse.Namespace,
        ema: Optional[object],
        scheduler,
        batch_num_in_all_epoch: int,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행한다.

    처리 순서:
      1) 모델을 train 모드로 두고, DDP 사용 시 CUDA 동기화.
      2) 전체 업데이트 스텝 수(batch_num_in_all_epoch)를 계산해 진행도(progress) 기준을 잡는다.
      3) 각 배치에 대해
         - device 로 이동 및 상한 클리핑(_prepare_batch_for_device)
         - augmentation, near future mask/ near future 4차원 궤적 생성
         - 관측 normalization
         - diffusion 손실 / feasible 손실 합성
         - 역전파, optimizer/scheduler step, EMA 업데이트
         - 배치별 loss 를 epoch_loss_dict_list 리스트에 모은다.
      4) epoch 종료 후, 배치별 loss 를 평균(get_epoch_mean_loss).
      5) DDP 사용 시 rank 0 기준으로 평균 후 출력/반환.

    Args:
        data_loader:
            PyTorch DataLoader. 각 요소는 collate_fn 이 만든 배치 dict.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        optimizer:
            torch.optim.Optimizer 인스턴스.
        args:
            학습 설정/상태 Namespace.
        ema:
            EMA 래퍼(ModelEma 등) 또는 None.
        scheduler:
            학습률 스케줄러. 배치마다 step() 이 호출된다.
        aug:
            StatePerturbation 또는 NPCStatePerturbation, 또는 None.

    Returns:
        - epoch_mean_loss: 손실 항목별 평균 dict. ( Dict[str, float] )
        - epoch_mean_loss["loss"]: 최종 스칼라 손실 값. float
    """
    epoch_loss_dict_list: List[Dict[str, torch.Tensor]] = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:

            # 1) device 이동 + 상한 클리핑 + 정답 분리
            """ outputs
        "ego_future_gt_4_dim",
        "near_future_gt_4_dim",
        "ego_future_gt_is_valid",
        "near_future_gt_is_valid",
            """
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
            )
            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)
            # normed_ego_future_gt_4_dim: (B, Tf, 4)
            outputs["ego_future_gt_4_dim"] = args.state_normalizer(
                data=outputs["ego_future_gt_4_dim"],
                valid_mask=outputs["ego_future_gt_is_valid"])
            # normed_near_future_gt_4_dim: (B, A, Tf, 4)
            outputs["near_future_gt_4_dim"] = args.state_normalizer(
                data=outputs["near_future_gt_4_dim"],
                valid_mask=outputs["near_future_gt_is_valid"])

            # 5) loss 계산 + 역전파 + optimizer/scheduler step
            """
            이번 배치(batch)를 학습하기 전에, 이전 배치에서 남아있는 기울기(gradient) 값을 깨끗이 지우는 작업

            (DeepSpeed를 쓰면 optimizer가 모델 내부에 묶여 동작하는 경우가 많아서, “모델에게” 초기화를 맡기는 방식이 맞습니다.)

            set_to_none=True는 기울기 값을 “0으로 채우기”보다 **아예 비워(None으로 만들기)**에 가까워서, 보통 메모리/속도 면에서 조금 더 유리할 수 있습니다.
            """
            if args.use_deepspeed and hasattr(model, "zero_grad"):
                model.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            # base_model: DDP/DeepSpeed 래퍼 벗긴 실제 모델
            base_model = ddp.get_model(model, args.ddp)
            sde_marginal_prob = base_model.sde.marginal_prob

            # 5-1) diffusion 기본 loss 계산 (neighbor / integration / constraint 등)
            raw_loss_dict: Dict[str, torch.Tensor] = {}
            raw_loss_dict, _ = diffusion_loss_func(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                norm_outputs=outputs,
                marginal_prob=sde_marginal_prob,
                state_normalizer=args.state_normalizer,
                loss_dict=raw_loss_dict,
                model_type=args.diffusion_model_type,  # 보통 "x_start" 또는 "score"
                observation_normalizer=args.observation_normalizer,
            )

            # 5-2) feasible weight 로 최종 loss 합성
            loss_dict: Dict[str, torch.Tensor] = _compute_loss_dict(
                loss_dict=raw_loss_dict,
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                batch_num_in_all_epoch=batch_num_in_all_epoch,
            )

            enable_profile: bool = bool(getattr(args, "profile_feasible",
                                                False))
            if enable_profile:
                print(
                    "===============[PROFILE train_epoch ENABLED]==============="
                )
            device_type: str = ("cuda" if "cuda" in str(args.device) else "cpu")

            with profile_block(
                    "train_epoch.train_epoch._backward_and_step",
                    enabled=enable_profile,
                    device_type=device_type,
            ):
                total_loss: float = _backward_and_step(
                    loss_dict=loss_dict,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    args=args,
                )
            # 6) WD warmdown, EMA 업데이트
            _apply_weight_decay_warmdown(optimizer)
            _update_ema_if_needed(ema, model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            loss_dict_detach_cpu: Dict[str, torch.Tensor] = {
                k: v.detach().cpu() for k, v in loss_dict.items()
            }
            epoch_loss_dict_list.append(loss_dict_detach_cpu)

            # 전역 스텝 누적 (진행도 계산에 사용)
            args._global_update_step += 1

    # --- 에폭 평균 손실 계산 ---
    """ epoch_mean_loss (각 GPU마다 계산해 둠)
    {"loss": 0.42, "neighbor_prediction_loss": 0.3, ...}
    """
    epoch_mean_loss: Dict[str,
    float] = get_epoch_mean_loss(epoch_loss_dict_list)

    if args.ddp:
        """ ddp.reduce_and_average_losses
            두 GPU가 서로 값을 더해서 합을 만든 뒤(world_size로 나눔)
            → 모든 GPU가 동일한 평균 값을 가지게 함
        통신하는 내용
            스칼라 몇 개밖에 안 되는 작은 숫자
        """
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss,
            torch.device(args.device),
        )
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
