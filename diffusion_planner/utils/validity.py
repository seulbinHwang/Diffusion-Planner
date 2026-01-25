from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Union, Literal

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]
MissingPolicy = Literal["none", "skip"]


def _get_first_non_none_value(
    sample: Mapping[str, Any],
    key_candidates: Sequence[str],
) -> Any:
    """여러 key 후보 중, 값이 None이 아닌 첫 번째 값을 반환합니다.

    같은 의미의 값이 "키 이름만 바뀌어" 들어올 수 있어,
    후보 키들을 순서대로 확인해서 None이 아닌 값을 하나 고르는 용도입니다.

    Args:
        sample: 입력 딕셔너리.
        key_candidates: 후보 키 이름들.

    Returns:
        None이 아닌 첫 값. 전부 없거나 None이면 None.
    """
    for key in key_candidates:
        value = sample.get(str(key), None)
        if value is not None:
            return value
    return None


def _is_array_like(value: Any) -> bool:
    """입력이 NumPy 배열 또는 torch 텐서인지 확인합니다.

    Args:
        value: 임의 타입 입력.

    Returns:
        bool: np.ndarray 또는 torch.Tensor면 True.
    """
    return isinstance(value, (np.ndarray, torch.Tensor))


def _compute_valid_mask_from_prefix_nonzero(
    features: ArrayLike,
    prefix_dim: int,
) -> ArrayLike:
    """마지막 축의 앞쪽(prefix) 값이 전부 0인지로 유효(True/False)를 계산합니다.

    규칙:
      - 마지막 축(feature_dim) 중 앞 prefix_dim개가 전부 0이면 False(무효)
      - 하나라도 0이 아니면 True(유효)

    예시:
      - ego_agent_past: shape (time_len, 11) -> (time_len,)
      - neighbor_agents_past: shape (A, time_len, 11) -> (A, time_len)
      - lanes: shape (L, lane_len, 12) -> (L, lane_len)

    Args:
        features: np.ndarray 또는 torch.Tensor. shape: (..., feature_dim)
        prefix_dim: 마지막 축에서 검사할 앞쪽 길이.

    Returns:
        True/False 배열. shape: features.shape[:-1]
        - 입력이 np.ndarray면 np.ndarray(bool)
        - 입력이 torch.Tensor면 torch.Tensor(bool)
    """
    if getattr(features, "ndim", 0) < 1:
        raise ValueError(
            f"features must have at least 1 dim. got shape={getattr(features, 'shape', None)}"
        )

    feature_dim = int(features.shape[-1])
    k = int(prefix_dim)
    if k <= 0 or k > feature_dim:
        raise ValueError(
            f"prefix_dim must be in [1, feature_dim]. got prefix_dim={k}, feature_dim={feature_dim}"
        )

    if isinstance(features, torch.Tensor):
        # prefix: shape (..., k)
        prefix = features[..., :k]
        # abs_sum: shape (...,)
        abs_sum = torch.sum(torch.abs(prefix), dim=-1)
        valid_mask = abs_sum > 0.0
        return valid_mask.to(dtype=torch.bool)

    # NumPy
    prefix = features[..., :k]  # shape (..., k)
    abs_sum = np.sum(np.abs(prefix), axis=-1)  # shape (...,)
    valid_mask = abs_sum > 0.0
    return valid_mask.astype(bool)


def _reduce_any_along_last_axis(mask_2d: ArrayLike) -> ArrayLike:
    """(N, M) True/False 배열을 (N,)로 줄입니다.

    규칙:
      - 마지막 축(M)에서 True가 하나라도 있으면 True
      - 전부 False면 False

    Args:
        mask_2d: np.ndarray(bool) 또는 torch.Tensor(bool). shape: (N, M)

    Returns:
        reduced: shape: (N,)
    """
    if int(getattr(mask_2d, "ndim", 0)) != 2:
        raise ValueError(
            f"mask_2d must be 2D. got shape={getattr(mask_2d, 'shape', None)}")

    if isinstance(mask_2d, torch.Tensor):
        return torch.any(mask_2d, dim=-1).to(dtype=torch.bool)

    return np.any(mask_2d, axis=-1).astype(bool)


def _make_all_true_mask_from_first_dim(array: ArrayLike) -> ArrayLike:
    """배열의 첫 번째 길이만큼 전부 True인 마스크를 만듭니다.

    예) stop_sign_points처럼 "있으면 전부 유효"로 보고 싶을 때 씁니다.

    Args:
        array: np.ndarray 또는 torch.Tensor. shape: (N, ...)

    Returns:
        True/False 배열. shape: (N,)
    """
    if int(getattr(array, "ndim", 0)) < 1:
        # shape를 알 수 없으면 (0,)로 통일
        if isinstance(array, torch.Tensor):
            return torch.ones((0,), dtype=torch.bool, device=array.device)
        return np.ones((0,), dtype=bool)

    n = int(array.shape[0])

    if isinstance(array, torch.Tensor):
        return torch.ones((n,), dtype=torch.bool, device=array.device)

    return np.ones((n,), dtype=bool)


def build_validity_key_dict(
    sample: Mapping[str, Any],
    *,
    missing_policy: MissingPolicy = "none",
) -> Dict[str, Any]:
    """입력 dict에서 validity key들을 계산해 별도 dict로 반환합니다.

    핵심 목표
    --------
    - `DiffusionPlannerData.__getitem__` (NumPy)와
      `WorldModelFeatureBuilder.get_features_from_simulation` (torch)
      두 경로가 "완전히 같은 규칙"으로 validity를 만들게 합니다.
    - 수정이 필요해지면 이 함수만 고치면 두 군데에 같이 반영됩니다.

    missing_policy 동작
    -------------------
    - "none": source가 없으면 validity key를 만들어서 값은 None으로 둡니다.
    - "skip": source가 없으면 그 validity key 자체를 만들지 않습니다.

    Args:
        sample: 입력 딕셔너리.
        missing_policy: 누락된 source 처리 방식.

    Returns:
        validity_dict: validity key들만 모은 dict.
    """

    def _set_or_skip(out: Dict[str, Any], key: str, value: Any) -> None:
        if value is None and missing_policy == "skip":
            return
        out[key] = value

    out: Dict[str, Any] = {}

    # ---------- a: ego_agent_past_is_valid ----------
    ego_agent_past = sample.get("ego_agent_past", None)
    if ego_agent_past is None or (not _is_array_like(ego_agent_past)):
        _set_or_skip(out, "ego_agent_past_is_valid", None)
    else:
        # ego_agent_past: shape (time_len, 11)
        _set_or_skip(
            out,
            "ego_agent_past_is_valid",
            _compute_valid_mask_from_prefix_nonzero(
                ego_agent_past, prefix_dim=8),  # shape (time_len,)
        )

    # ---------- b: ego_future_gt_is_valid ----------
    ego_future_11 = _get_first_non_none_value(
        sample, ["planner_future_11_dim", "ego_future_gt_11_dim"])
    if ego_future_11 is None or (not _is_array_like(ego_future_11)):
        _set_or_skip(out, "ego_future_gt_is_valid", None)
    else:
        # ego_future_11: shape (future_len, 11)
        _set_or_skip(
            out,
            "ego_future_gt_is_valid",
            _compute_valid_mask_from_prefix_nonzero(
                ego_future_11, prefix_dim=8),  # shape (future_len,)
        )

    # ---------- c: neighbor_agents_past_is_valid ----------
    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if neighbor_agents_past is None or (
            not _is_array_like(neighbor_agents_past)):
        _set_or_skip(out, "neighbor_agents_past_is_valid", None)
    else:
        # neighbor_agents_past: shape (A, time_len, 11)
        _set_or_skip(
            out,
            "neighbor_agents_past_is_valid",
            _compute_valid_mask_from_prefix_nonzero(
                neighbor_agents_past, prefix_dim=8),  # shape (A, time_len)
        )

    # ---------- d: neighbor_agents_is_valid ----------
    if neighbor_agents_past is None or (
            not _is_array_like(neighbor_agents_past)):
        _set_or_skip(out, "neighbor_agents_is_valid", None)
    else:
        # last_step: shape (A, 11)
        last_step = neighbor_agents_past[:, -1, :]
        _set_or_skip(
            out,
            "neighbor_agents_is_valid",
            _compute_valid_mask_from_prefix_nonzero(last_step,
                                                    prefix_dim=8),  # shape (A,)
        )

    # ---------- e: neighbor_future_gt_is_valid ----------
    neighbor_future_11 = _get_first_non_none_value(
        sample,
        ["neighbor_future_gt_11_dim"],
    )
    print("neighbor_future_11.type", type(neighbor_future_11))
    if neighbor_future_11 is None or (not _is_array_like(neighbor_future_11)):
        print("here A")
        _set_or_skip(out, "neighbor_future_gt_is_valid", None)
        print("out['neighbor_future_gt_is_valid']", out.get("neighbor_future_gt_is_valid", None))
    else:
        # neighbor_future_11: shape (A, future_len, 11)
        print("here B")
        print("neighbor_future_11.shape", neighbor_future_11.shape)
        _set_or_skip(
            out,
            "neighbor_future_gt_is_valid",
            _compute_valid_mask_from_prefix_nonzero(
                neighbor_future_11,
                prefix_dim=8,  # 마지막 11차원 중 앞 8개 값이 전부 0이면 False(무효)
            ),  # shape (A, future_len)
        )
        print("out['neighbor_future_gt_is_valid'].shape",
              out["neighbor_future_gt_is_valid"].shape)

    # ---------- f: stop_sign_is_valid ----------
    stop_sign_points = sample.get("stop_sign_points", None)
    if stop_sign_points is None or (not _is_array_like(stop_sign_points)):
        _set_or_skip(out, "stop_sign_is_valid", None)
    else:
        # stop_sign_points: shape (N_stop, safety_len, 2)
        _set_or_skip(
            out, "stop_sign_is_valid",
            _make_all_true_mask_from_first_dim(stop_sign_points))  # (N_stop,)

    # ---------- g: crosswalk_is_valid ----------
    crosswalk_points = sample.get("crosswalk_points", None)
    if crosswalk_points is None or (not _is_array_like(crosswalk_points)):
        _set_or_skip(out, "crosswalk_is_valid", None)
    else:
        # crosswalk_points: shape (N_cross, safety_len, 2)
        _set_or_skip(
            out, "crosswalk_is_valid",
            _make_all_true_mask_from_first_dim(crosswalk_points))  # (N_cross,)

    # ---------- h: lanes_len_is_valid / lanes_is_valid ----------
    lanes = sample.get("lanes", None)
    if lanes is None or (not _is_array_like(lanes)):
        _set_or_skip(out, "lanes_len_is_valid", None)
        _set_or_skip(out, "lanes_is_valid", None)
    else:
        # lanes: shape (L, lane_len, 12)
        lanes_len_is_valid = _compute_valid_mask_from_prefix_nonzero(
            lanes, prefix_dim=8)  # (L, lane_len)
        _set_or_skip(out, "lanes_len_is_valid", lanes_len_is_valid)
        _set_or_skip(out, "lanes_is_valid",
                     _reduce_any_along_last_axis(lanes_len_is_valid))  # (L,)

    # ---------- static_objects_is_valid ----------
    static_objects = sample.get("static_objects", None)
    if static_objects is None or (not _is_array_like(static_objects)):
        _set_or_skip(out, "static_objects_is_valid", None)
    else:
        # static_objects: shape (N_static, 10)
        _set_or_skip(
            out, "static_objects_is_valid",
            _make_all_true_mask_from_first_dim(static_objects))  # (N_static,)

    # ---------- route_lanes_len_is_valid / route_lanes_is_valid ----------
    route_lanes = sample.get("route_lanes", None)
    if route_lanes is None or (not _is_array_like(route_lanes)):
        _set_or_skip(out, "route_lanes_len_is_valid", None)
        _set_or_skip(out, "route_lanes_is_valid", None)
    else:
        # route_lanes: shape (R, route_len, 12)
        route_lanes_len_is_valid = _compute_valid_mask_from_prefix_nonzero(
            route_lanes, prefix_dim=8)  # (R, route_len)
        _set_or_skip(out, "route_lanes_len_is_valid", route_lanes_len_is_valid)
        _set_or_skip(
            out, "route_lanes_is_valid",
            _reduce_any_along_last_axis(route_lanes_len_is_valid))  # (R,)

    # ---------- agent_route_lane_order_is_valid ----------
    agent_route_lane_order = sample.get("agent_route_lane_order", None)
    if agent_route_lane_order is None or (
            not _is_array_like(agent_route_lane_order)):
        _set_or_skip(out, "agent_route_lane_order_is_valid", None)
    else:
        # agent_route_lane_order: shape (A, ?)
        _set_or_skip(
            out, "agent_route_lane_order_is_valid",
            _make_all_true_mask_from_first_dim(agent_route_lane_order))  # (A,)

    # ---------- WOMD 전용: speed_bump / driveway / road_edge ----------
    speed_bump_points = sample.get("speed_bump_points", None)
    if speed_bump_points is None or (not _is_array_like(speed_bump_points)):
        _set_or_skip(out, "speed_bump_is_valid", None)
    else:
        _set_or_skip(out, "speed_bump_is_valid",
                     _make_all_true_mask_from_first_dim(speed_bump_points))

    driveway_points = _get_first_non_none_value(sample,
                                                ["driveway_points", "driveway"])
    if driveway_points is None or (not _is_array_like(driveway_points)):
        _set_or_skip(out, "driveway_is_valid", None)
    else:
        _set_or_skip(out, "driveway_is_valid",
                     _make_all_true_mask_from_first_dim(driveway_points))

    road_edge = sample.get("road_edge", None)
    if road_edge is None or (not _is_array_like(road_edge)):
        _set_or_skip(out, "road_edge_is_valid", None)
    else:
        _set_or_skip(out, "road_edge_is_valid",
                     _make_all_true_mask_from_first_dim(road_edge))

    return out


def add_validity_keys_inplace(
    sample: Dict[str, Any],
    *,
    missing_policy: MissingPolicy = "none",
) -> None:
    """입력 dict에 validity key들을 in-place로 추가합니다.

    Args:
        sample: 수정 대상 dict.
        missing_policy: 누락 source 처리 방식.
            - "none": validity key를 추가하되 값은 None
            - "skip": 누락된 것은 key 자체를 추가하지 않음
    """
    validity = build_validity_key_dict(sample, missing_policy=missing_policy)
    sample.update(validity)
