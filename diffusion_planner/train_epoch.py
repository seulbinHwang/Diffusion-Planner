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


def _move_batch_to_device(
    batch: Dict[str, Any],
    device: str,
) -> Dict[str, Any]:
    """배치 dict에서 torch.Tensor만 device로 옮기고, 나머지(None 포함)는 그대로 둔다."""
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
    assert agent_route_lane_order_agent_num == neighbor_agents_agent_num, \
        f"agent_route_lane_order agent num ({agent_route_lane_order_agent_num}) " \
        f"!= neighbor_agents_past agent num ({neighbor_agents_agent_num})"
    return batch_on_device


def _clip_axis1_for_key_inplace(
    batch_on_device: Dict[str, torch.Tensor],
    key: str,
    keep_len: int,
) -> None:
    """배치 텐서에서 '두 번째 축(axis=1)' 길이를 keep_len으로 안전하게 줄입니다.

    왜 필요한가
    ----------
    어떤 그룹의 "개수"를 줄일 때(예: agent 수, lane 수),
    그 그룹과 짝으로 움직여야 하는 *_is_valid 텐서도 같은 개수로 줄지 않으면
    이후 연산에서 shape가 안 맞아서 오류가 나거나, 마스크가 엉뚱한 객체를 가리킬 수 있습니다.

    동작 규칙
    --------
    - batch_on_device에 key가 없으면 아무 것도 하지 않습니다.
    - keep_len <= 0이면 아무 것도 하지 않습니다.
    - 텐서 차원이 2 미만이면(axis=1이 없으면) 아무 것도 하지 않습니다.
    - 이미 axis=1 길이가 keep_len 이하이면 아무 것도 하지 않습니다.
    - 위 조건을 통과하면, 해당 텐서를 [:, :keep_len, ...] 형태로 잘라서 다시 넣습니다.

    Args:
        batch_on_device (Dict[str, torch.Tensor]):
            배치 dict. 각 텐서는 보통 (B, N, ...) 형태입니다.
            예:
              - neighbor_agents_is_valid: (B, A)
              - lanes_is_valid: (B, L)
              - lanes_len_is_valid: (B, L, P)
        key (str):
            자를 대상 key 이름.
        keep_len (int):
            axis=1에서 남길 길이.
            예: agent를 A'개만 남기면 keep_len=A'

    Returns:
        None
    """
    if keep_len <= 0:
        return
    if key not in batch_on_device:
        return

    t = batch_on_device[key]
    if not isinstance(t, torch.Tensor):
        return
    if t.dim() < 2:
        return

    current_len = int(t.shape[1])
    if keep_len >= current_len:
        return

    # t: (B, N, ...) -> (B, keep_len, ...)
    batch_on_device[key] = t[:, :keep_len, ...]


def _clip_axis1_for_keys_inplace(
    batch_on_device: Dict[str, torch.Tensor],
    keys: List[str],
    keep_len: int,
) -> None:
    """여러 key에 대해 axis=1 길이를 동일하게 맞춰 안전하게 줄입니다.

    Args:
        batch_on_device (Dict[str, torch.Tensor]):
            배치 dict.
        keys (List[str]):
            함께 길이를 맞춰 잘라야 하는 key 목록.
        keep_len (int):
            axis=1에서 남길 길이.

    Returns:
        None
    """
    for k in keys:
        _clip_axis1_for_key_inplace(batch_on_device=batch_on_device,
                                    key=k,
                                    keep_len=keep_len)


def _clip_input_axes_by_args(
    batch_on_device: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    if getattr(args, "do_not_clip", False):
        return

    # 1) neighbor_agents_past
    max_agent_num = int(getattr(args, "max_agent_num", 0))
    neighbor_agents_past = batch_on_device.get("neighbor_agents_past", None)
    if isinstance(neighbor_agents_past, torch.Tensor
                 ) and max_agent_num > 0 and neighbor_agents_past.dim() == 4:
        keep_agents = min(max_agent_num, int(neighbor_agents_past.shape[1]))
        _clip_axis1_for_key_inplace(batch_on_device, "neighbor_agents_past",
                                    keep_agents)
        _clip_axis1_for_keys_inplace(
            batch_on_device=batch_on_device,
            keys=[
                "neighbor_agents_past_is_valid", "neighbor_agents_is_valid",
                "neighbor_future_gt_is_valid"
            ],
            keep_len=keep_agents,
        )

    # 2) lanes
    max_lane_num = int(getattr(args, "max_lane_num", 0))
    lanes = batch_on_device.get("lanes", None)
    if isinstance(lanes,
                  torch.Tensor) and max_lane_num > 0 and lanes.dim() == 4:
        keep_lanes = min(max_lane_num, int(lanes.shape[1]))
        if keep_lanes < int(lanes.shape[1]):
            batch_on_device["lanes"] = lanes[:, :keep_lanes, :, :]
            _clip_axis1_for_key_inplace(batch_on_device, "lanes_speed_limit",
                                        keep_lanes)
            _clip_axis1_for_key_inplace(batch_on_device,
                                        "lanes_has_speed_limit", keep_lanes)
            _clip_axis1_for_keys_inplace(
                batch_on_device=batch_on_device,
                keys=["lanes_len_is_valid", "lanes_is_valid"],
                keep_len=keep_lanes,
            )

    # 3) route_lanes
    route_lanes = batch_on_device.get("route_lanes", None)
    if isinstance(route_lanes,
                  torch.Tensor) and max_lane_num > 0 and route_lanes.dim() == 4:
        keep_route_lanes = min(max_lane_num, int(route_lanes.shape[1]))
        if keep_route_lanes < int(route_lanes.shape[1]):
            batch_on_device[
                "route_lanes"] = route_lanes[:, :keep_route_lanes, :, :]
            _clip_axis1_for_key_inplace(batch_on_device,
                                        "route_lanes_speed_limit",
                                        keep_route_lanes)
            _clip_axis1_for_key_inplace(batch_on_device,
                                        "route_lanes_has_speed_limit",
                                        keep_route_lanes)
            _clip_axis1_for_keys_inplace(
                batch_on_device=batch_on_device,
                keys=["route_lanes_len_is_valid", "route_lanes_is_valid"],
                keep_len=keep_route_lanes,
            )

    # 4) agent_route_lane_order
    arl = batch_on_device.get("agent_route_lane_order", None)
    if isinstance(arl, torch.Tensor) and arl.dim() == 3:
        _, c_agent, c_lane = arl.shape
        keep_agents_for_route = int(c_agent)

        lane_dim_input = int(c_lane)
        lanes_now = batch_on_device.get("lanes", None)
        if isinstance(lanes_now, torch.Tensor) and lanes_now.dim() >= 2:
            lane_dim_input = int(lanes_now.shape[1])

        lane_cap = max_lane_num if max_lane_num > 0 else lane_dim_input
        keep_lanes_for_route = min(int(lane_cap), int(lane_dim_input))

        batch_on_device[
            "agent_route_lane_order"] = arl[:, :keep_agents_for_route, :
                                            keep_lanes_for_route]
        _clip_axis1_for_key_inplace(batch_on_device,
                                    "agent_route_lane_order_is_valid",
                                    keep_agents_for_route)


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
    args: Optional[argparse.Namespace] = None,
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

    if args is not None:
        _clip_input_axes_by_args(batch_on_device, args)

    # outputs 분리 (정답은 반드시 Tensor여야 함)
    target_keys = {"ego_future_gt_3_dim", "near_future_gt_3_dim"}
    outputs: Dict[str, torch.Tensor] = {}
    for key in list(batch_on_device.keys()):
        if key in target_keys:
            value = batch_on_device.pop(key)
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"target '{key}' must be torch.Tensor, got {type(value)}")
            outputs[key] = value
            if key == "ego_future_gt_3_dim":
                # ego_future_gt_3_dim: (B, future_len, 3)
                ego_future_gt_3_dim: torch.Tensor = value
                ego_future_gt_4_dim = torch.cat(
                    [
                        ego_future_gt_3_dim[..., :2],  # (B, future_len, 2)
                        torch.stack(
                            [
                                ego_future_gt_3_dim[..., 2].cos(),
                                ego_future_gt_3_dim[..., 2].sin(),
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=-1,
                )  # (B, future_len, 4)
                # ego_future_mask: (B, future_len)
                ego_future_gt_11_dim = batch_on_device.get("planner_future_11_dim", None)
                ego_future_mask: torch.Tensor = torch.sum(
                    torch.ne(ego_future_gt_11_dim[..., :8], 0),
                    dim=-1,
                ) == 0
                ego_future_gt_4_dim[ego_future_mask] = 0.0

                # DEBUG
                ego_future_valid = ~ego_future_mask  # (B, future_len)  True=유효
                ego_future_valid_np = ego_future_valid.cpu().numpy()

                assert_cur_future_valid_mask_np(
                    ego_future_valid_np[None, ...],
                    context="_prepare_batch_for_device")

                ego_future_len = ego_future_gt_3_dim.shape[1]
                assert ego_future_len == args.future_len, \
                    f"ego future len mismatch: {ego_future_len} vs {args.future_len}"
                outputs["ego_future_gt_4_dim"] = ego_future_gt_4_dim
            elif key == "near_future_gt_3_dim":
                # 3) near future 4차원 궤적 + mask 생성
                # near_future_gt_4_dim: (B, Pnn, future_len, 4)
                # near_future_mask:    (B, Pnn, future_len)
                near_future_gt_4_dim, near_future_mask = \
                    _build_near_future_4dim_and_mask(value)
                outputs["near_future_gt_4_dim"] = near_future_gt_4_dim
                outputs["near_future_mask"] = near_future_mask

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


def _build_broadcastable_pad_mask(
    feature: torch.Tensor,
    is_valid: torch.Tensor,
) -> torch.Tensor:
    """feature에 바로 적용 가능한 '패딩 위치 마스크'를 만든다.

    Args:
        feature: 원본 feature 텐서. shape: (B, ..., C)
        is_valid: 유효 여부 텐서. 보통 feature의 마지막 차원(C)을 뺀 모양.
            예:
              - feature: (B, A, T, 11) / is_valid: (B, A, T)
              - feature: (B, N, S, 2)  / is_valid: (B, N)

    Returns:
        pad_mask: bool 텐서. feature에 masked_fill로 바로 쓸 수 있는 모양.
            True인 곳이 "패딩(0으로 유지해야 하는 곳)".
    """
    valid_bool = _as_bool_mask(is_valid)
    pad_mask = ~valid_bool
    while pad_mask.dim() < feature.dim():
        pad_mask = pad_mask.unsqueeze(-1)
    return pad_mask


def _collect_padding_masks_before_augmentation(
    inputs: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
    """augmentation 전에 패딩 위치를 기억해 둔다.

    Args:
        inputs: 모델 입력 dict. 각 value는 torch.Tensor.

    Returns:
        feature_key -> pad_mask(bool) dict.
        pad_mask는 feature에 바로 masked_fill 할 수 있는 모양입니다.
    """
    pairs: List[Tuple[str, str]] = [
        ("ego_agent_past", "ego_agent_past_is_valid"),  # (B,T,11) / (B,T)
        ("planner_future_11_dim",
         "ego_future_gt_is_valid"),  # (B,Tf,11) / (B,Tf)
        ("neighbor_agents_past",
         "neighbor_agents_past_is_valid"),  # (B,A,T,11) / (B,A,T)
        ("lanes", "lanes_len_is_valid"),  # (B,L,S,12) / (B,L,S)
        ("route_lanes", "route_lanes_len_is_valid"),  # (B,R,S,12) / (B,R,S)
        ("static_objects", "static_objects_is_valid"),  # (B,N,10) / (B,N)
        ("stop_sign_points", "stop_sign_is_valid"),  # (B,N,S,2) / (B,N)
        ("crosswalk_points", "crosswalk_is_valid"),  # (B,N,S,2) / (B,N)
        ("speed_bump_points", "speed_bump_is_valid"),  # (B,N,S,2) / (B,N)
        ("driveway_points", "driveway_is_valid"),  # (B,N,S,2) / (B,N)
        ("road_edge", "road_edge_is_valid"),  # (B,N,S,2) / (B,N)
    ]

    pad_masks: Dict[str, torch.Tensor] = {}
    for feature_key, valid_key in pairs:
        if feature_key not in inputs:
            continue
        feature = inputs[feature_key]
        if not isinstance(feature, torch.Tensor):
            continue

        if valid_key in inputs and isinstance(inputs[valid_key], torch.Tensor):
            pad_masks[feature_key] = _build_broadcastable_pad_mask(
                feature, inputs[valid_key])
        else:
            # is_valid가 없으면 "마지막 차원이 전부 0"인 곳을 패딩으로 보수적으로 간주
            pad_masks[feature_key] = (feature == 0).all(dim=-1, keepdim=True)

    return pad_masks


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


def _validate_batch_shapes_for_loss(
    inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """손실 계산 전에 꼭 맞아야 하는 모양(shape)을 점검한다.

    Args:
        inputs:
            - neighbor_agents_past: (B, A_in, T_past, 11)
        outputs:
            - ego_future_gt_3_dim: (B, Tf, 3)
            - near_future_gt_3_dim: (B, A_pred, Tf, 3)
        args:
            - future_len 사용

    Raises:
        ValueError: 모양이 맞지 않으면 즉시 에러로 중단합니다.
    """
    if "ego_future_gt_3_dim" not in outputs or "near_future_gt_3_dim" not in outputs:
        raise ValueError(
            "outputs에 'ego_future_gt_3_dim' 또는 'near_future_gt_3_dim'이 없습니다.")

    ego_fut = outputs["ego_future_gt_3_dim"]
    near_fut = outputs["near_future_gt_3_dim"]

    if ego_fut.dim() != 3 or ego_fut.size(-1) != 3:
        raise ValueError(
            f"ego_future_gt_3_dim shape expected (B,Tf,3), got {tuple(ego_fut.shape)}"
        )
    if near_fut.dim() != 4 or near_fut.size(-1) != 3:
        raise ValueError(
            f"near_future_gt_3_dim shape expected (B,A,Tf,3), got {tuple(near_fut.shape)}"
        )

    if int(ego_fut.shape[1]) != int(args.future_len) or int(
            near_fut.shape[2]) != int(args.future_len):
        raise ValueError(
            f"future_len mismatch: args.future_len={int(args.future_len)}, "
            f"ego_future_len={int(ego_fut.shape[1])}, near_future_len={int(near_fut.shape[2])}"
        )

    if "neighbor_agents_past" not in inputs:
        raise ValueError(
            "inputs에 'neighbor_agents_past'가 없습니다. loss에서 현재 상태를 만들 수 없습니다.")

    neigh_past = inputs["neighbor_agents_past"]
    if neigh_past.dim() != 4:
        raise ValueError(
            f"neighbor_agents_past must be (B,A,T,11), got {tuple(neigh_past.shape)}"
        )

    a_in = int(neigh_past.shape[1])
    a_pred = int(near_fut.shape[1])
    if a_in < a_pred:
        raise ValueError(
            f"neighbor_agents_past agent 수({a_in}) < near_future_gt_3_dim agent 수({a_pred}). "
            "보통 /max_agent_num 설정 또는 collate padding 크기 문제입니다.")


def _apply_augmentation(
    inputs: Dict[str, torch.Tensor],
    ego_future_gt_3_dim: torch.Tensor,  #(B, future_len, 3)
    near_future_gt_3_dim: torch.Tensor,  # (B, Pnn, future_len, 3)
    ego_future_gt_mask: torch.Tensor,  # (B, future_len)
    near_future_mask: torch.Tensor,  # (B, Pnn, future_len)
    aug: Optional[StatePerturbation],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """ego / 이웃 궤적에 대해 augmentation 을 적용한다.

    핵심 안전장치(중요)
    ------------------
    좌표를 회전/이동시키는 augmentation을 하면,
    원래 0으로 채워져 있던 패딩 구간까지 값이 바뀌어
    존재하지 않는 객체가 존재하는 것처럼 보일 수 있습니다.

    그래서 augmentation 전에 "원래 패딩이었던 위치"를 기억해두고,
    augmentation 후에 그 위치를 다시 전부 0으로 되돌립니다.

    Args:
        inputs:
            모델 입력용 dict. 각 텐서는 (B, ...) 모양.
        ego_future_gt_3_dim:
            (B, Tf, 3)
        near_future_gt_3_dim:
            (B, A, Tf, 3)
        aug:
            StatePerturbation 또는 NPCStatePerturbation 또는 None
        args:
            NPCStatePerturbation에서 필요할 수 있는 설정

    Returns:
        inputs, ego_future_gt_3_dim, near_future_gt_3_dim (augmentation 반영 + padding 복원)
    """
    # (1) augmentation 이전: 입력/정답에서 "원래 패딩 위치"를 기억
    input_pad_masks: Dict[
        str, torch.Tensor] = _collect_padding_masks_before_augmentation(inputs)

    ego_future_pad_mask = ego_future_gt_mask.unsqueeze(-1)  # (B,Tf,1)
    near_future_pad_mask = near_future_mask.unsqueeze(-1)  # (B,A,Tf,1)
    assert isinstance(aug,
                      NPCStatePerturbation), "현재 NPCStatePerturbation 만 지원합니다."
    if args.do_ego_predict:
        target_future_gt_3_dim = torch.cat(
            [ego_future_gt_3_dim.unsqueeze(1), near_future_gt_3_dim],
            dim=1)  # (B, 1 + A, Tf, 3)
    else:
        target_future_gt_3_dim = near_future_gt_3_dim  # (B, A, Tf, 3)
    inputs, target_future_gt_3_dim = aug(inputs, target_future_gt_3_dim, args)
    if args.do_ego_predict:
        ego_future_gt_3_dim = target_future_gt_3_dim[:, 0, :, :]  # (B, Tf, 3)
        # near_future_gt_3_dim = target_future_gt_3_dim[:, 1:, :, :]  # (B, A, Tf, 3)
    else:
        near_future_gt_3_dim = target_future_gt_3_dim  # (B, A, Tf, 3)

    # (3) augmentation 이후: 원래 패딩이었던 위치는 다시 0으로 복원
    _restore_padding_values_inplace(inputs, input_pad_masks)

    ego_future_gt_3_dim = ego_future_gt_3_dim.masked_fill(
        ego_future_pad_mask, 0.0)
    near_future_gt_3_dim = near_future_gt_3_dim.masked_fill(
        near_future_pad_mask, 0.0)

    return inputs, ego_future_gt_3_dim, near_future_gt_3_dim


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
    scheduler,
    args: argparse.Namespace,
) -> float:
    """역전파/그래디언트 클리핑/스케줄러/옵티마이저 업데이트를 한 번 수행한다.

    Args:
        loss_dict:
            - "loss" 키에 최종 scalar 손실 텐서가 들어 있는 dict.
              · loss_dict["loss"]의 shape: ()  스칼라 텐서.
        model:
            학습 중인 모델.
            - 일반 모드: nn.Module 또는 DDP 래퍼.
            - ZeRO-2 모드: deepspeed.DeepSpeedEngine.
        optimizer:
            torch.optim.Optimizer 또는 DeepSpeed가 감싼 Optimizer.
        scheduler:
            학습률 스케줄러. 일반 모드에서만 직접 step()을 호출한다.
        args:
            학습 설정/상태 Namespace.
            - args.use_deepspeed: True이면 DeepSpeed 엔진을 사용한다.
            - args.max_grad_norm: 기울기 클리핑 기준값. 0 이하이면 클리핑 안 함.

    Returns:
        float:
            loss_dict["loss"].item() 값 (logging 용).
    """
    # total_loss는 scalar float 값
    total_loss: float = float(loss_dict["loss"].item())
    loss_tensor: torch.Tensor = loss_dict["loss"]

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False)) \
        and hasattr(model, "backward") and hasattr(model, "step")

    # 한 곳에서만 클리핑 기준을 정한다.
    max_grad_norm: float = float(getattr(args, "max_grad_norm", 0.0))

    if use_deepspeed:
        # DeepSpeed 엔진을 사용하는 경우:
        # gradient_clipping 값은 build_deepspeed_config()에서
        # ds_config["gradient_clipping"] = max_grad_norm 으로 넘어간다.
        # 여기서는 따로 clip_grad_norm을 호출하지 않는다.
        """ model.backward(loss_tensor) # deepseed 전용 역전파
        각 GPU가 자기 배치에 대한 기울기를 먼저 계산한다.
            GPU0: g0(W1), g0(W2), g0(W3), g0(W4)
            GPU1: g1(W1), g1(W2), g1(W3), g1(W4)
        여기까진 “각자 전체 기울기를 한 번씩 계산했다”고 보면 된다.
            (이 단계는 ZeRO-2라도 어쩔 수 없이 한 번 거치는 단계)

        2. 그 다음 “기울기를 나누고 합치는 통신 단계”가 들어간다.
        
           * GPU0와 GPU1이 서로 기울기 조각을 주고받아서:
        
             * GPU0는 W1, W2에 대한
               **(g0 + g1)의 합**만 남기고 W3, W4에 대한 기울기는 버린다.
             * GPU1은 W3, W4에 대한
               **(g0 + g1)의 합**만 남기고 W1, W2에 대한 기울기는 버린다.
        
           즉, **최종적으로**:
        
           * GPU0: (합쳐진 기울기) g(W1), g(W2) 만 보관
           * GPU1: (합쳐진 기울기) g(W3), g(W4) 만 보관
        
           → 이게 “기울기를 GPU 사이에 나눠서 가진다”는 뜻이다.
           (옵티마 상태도 비슷하게 “나눠서 저장”한다.)

# algather_partitions = False 일 때
    1. GPU0, GPU1이 각각 **자기 배치에 대해 전체 기울기**를 계산 (g0, g1).
    
    2. GPU0와 GPU1이 통신해서:
       * **먼저 전체 기울기를 서로 합친다.**
         * 결과적으로 GPU0, GPU1 둘 다
           * g_sum(W1), g_sum(W2), g_sum(W3), g_sum(W4)
             를 잠깐씩 다 들고 있을 수 있다.
       * 그 다음,
         * “나는 W1,W2만 쥐고 있을게” / “나는 W3,W4만 쥐고 있을게” 식으로
           기울기와 옵티마 상태를 다시 나누고 정리.
    
       → 즉, **중간에 “모든 기울기 합본을 한 번씩 다 들고 있는 순간”이 있을 수 있다.**
       그래서 메모리 사용량 관점에서 조금 덜 효율적인 쪽.

# allgather_partitions = True 일 때
    2. 그런데 여기서는, **“전체 합본을 두 군데 다 오래 들고 있게 만들지 않고”**
       바로 “나눠진 형태” 위주로 유지하려고 한다.
    
       예를 들어 개념적으로는:
    
       * 단계 1: GPU0/1이 서로 기울기를 교환하면서,
    
         * GPU0는 W1,W2 부분에 대한 `g0+g1`만 남기고,
         * GPU1은 W3,W4 부분에 대한 `g0+g1`만 남긴다.
       * 이때 “모든 파라미터에 대한 합본”을 각 GPU가 오래 들고 있는 순간을 줄이고,
         * 바로 “나눠진 합본”만 남기는 쪽으로 통신을 설계하는 것.
    
       → 실제 구현은 더 복잡하지만,
       **“합친 전체 기울기를 각 GPU가 길게 들고 있지 않는다”**는 방향으로 생각하면 된다.
        """
        model.backward(loss_tensor)
        """ model.step()
3. 옵티마 단계에서:
   * GPU0는 자신이 가진 파라미터에 대해서만 갱신
     * W1, W2 를 g(W1), g(W2) 와 자기 쪽 옵티마 상태를 써서 업데이트
   * GPU1은 W3, W4 를 자기 쪽 기울기/옵티마로 업데이트

4. 업데이트가 끝나면,
   * GPU0와 GPU1이 서로 **업데이트된 W1~W4 전체를 다시 맞춘다**.
     (브로드캐스트 혹은 비슷한 방식으로 동기화)
   * 그래서 **step이 끝난 후에는** 다시
     * GPU0: W1~W4 전체 최신 버전
     * GPU1: W1~W4 전체 최신 버전

        """
        model.step()
    else:
        # 일반 PyTorch / DDP 모드
        """ loss_tensor.backward()
이때 DDP가 **파라미터별 gradient가 만들어지는 순간마다 후크를 걸어** 다음을 수행:

* GPU0는 자기 gradient `g0`를 가지고 있음.
* GPU1는 자기 gradient `g1`를 가지고 있음.
* PyTorch DDP 내부에서:
  * 각 파라미터마다
    `g_avg = (g0 + g1) / 2` 를 만들기 위해
    GPU0와 GPU1이 서로 값을 주고받고, 더하고, 나눔.
* 그 결과:
  * GPU0의 해당 파라미터 gradient = `g_avg`
  * GPU1의 해당 파라미터 gradient = `g_avg`
즉, **backward가 끝났을 때, 두 GPU의 gradient는 완전히 동일**하게 맞춰져 있음.
이 과정이 이 설정에서 **가장 큰 통신 비용**이야.        
        """
        loss_tensor.backward()
        """
        아래 3줄 코드에서는 GPU끼리 통신하지 않음.
        """
        # max_grad_norm > 0 일 때만 클리핑 수행
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 스케줄러 / 옵티마이저 스텝
        scheduler.step()
        optimizer.step()

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
    aug: Optional[StatePerturbation] = None,
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
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )
            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)

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
                marginal_prob=sde_marginal_prob,
                ego_future_gt_4_dim=outputs["ego_future_gt_4_dim"],
                near_future_gt_4_dim=outputs["near_future_gt_4_dim"],
                near_future_mask=outputs["near_future_mask"],
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
            epoch_loss_dict_list.append(loss_dict)

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
