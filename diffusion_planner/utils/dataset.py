import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata


def _get_first_non_none_value(
    sample: Dict[str, Any],
    key_candidates: Sequence[str],
) -> Any:
    """여러 key 후보 중, 값이 None이 아닌 첫 번째 값을 가져옵니다.

    rename 등으로 같은 의미의 값이 key 이름만 다른 경우가 있습니다.
    이 함수는 후보 key들을 순서대로 확인해서,
    값이 None이 아닌 첫 값을 반환합니다.

    Args:
        sample: __getitem__이 만들고 있는 샘플 dict.
        key_candidates: 같은 의미의 값을 가질 수 있는 key 이름 후보들.

    Returns:
        None이 아닌 첫 값. 전부 None이거나 key가 없으면 None.
    """
    for key in key_candidates:
        value = sample.get(str(key), None)
        if value is not None:
            return value
    return None


def _compute_valid_mask_from_prefix_nonzero(
    features: np.ndarray,
    prefix_dim: int,
) -> np.ndarray:
    """마지막 차원의 앞쪽(prefix) 값이 '전부 0인지'로 유효 마스크를 만듭니다.

    규칙:
      - 마지막 차원(feature_dim) 중 앞 prefix_dim개가 전부 0이면 invalid(False)
      - 하나라도 0이 아닌 값이 있으면 valid(True)

    예시:
      - ego_agent_past: (time_len, 11) -> (time_len,) bool
      - neighbor_agents_past: (A, time_len, 11) -> (A, time_len) bool
      - lanes: (L, lane_len, 12) -> (L, lane_len) bool

    Args:
        features: numpy 배열. shape: (..., feature_dim)
            - 마지막 차원이 feature_dim이어야 합니다.
        prefix_dim: 마지막 차원에서 앞쪽으로 검사할 길이.
            - agent feature(11)에서는 보통 8
            - lane/route feature(12)에서도 보통 8
            - (x,y,heading) 3차원에서는 3

    Returns:
        valid_mask: bool 배열. shape: features.shape[:-1]
    """
    # features: np.ndarray, shape (..., feature_dim)
    if features.ndim < 1:
        raise ValueError(
            f"features must have at least 1 dim. got shape={features.shape}")

    feature_dim = int(features.shape[-1])
    k = int(prefix_dim)
    if k <= 0 or k > feature_dim:
        raise ValueError(
            f"prefix_dim must be in [1, feature_dim]. got prefix_dim={k}, feature_dim={feature_dim}"
        )

    prefix = features[..., :k]  # shape (..., k)
    # abs_sum: shape (...,)
    abs_sum = np.sum(np.abs(prefix), axis=-1)
    valid_mask = abs_sum > 0.0
    return valid_mask.astype(bool)


def _reduce_any_along_last_axis(mask_2d: np.ndarray) -> np.ndarray:
    """(N, M) bool 마스크를 (N,)으로 줄입니다.

    규칙:
      - 마지막 축(M)에서 True가 하나라도 있으면 True
      - 전부 False면 False

    Args:
        mask_2d: bool 배열. shape: (N, M)

    Returns:
        reduced: bool 배열. shape: (N,)
    """
    # mask_2d: np.ndarray, shape (N, M)
    if mask_2d.ndim != 2:
        raise ValueError(f"mask_2d must be 2D. got shape={mask_2d.shape}")
    return np.any(mask_2d, axis=-1).astype(bool)


def _make_all_true_mask_from_first_dim(array: np.ndarray,) -> np.ndarray:
    """배열의 첫 번째 길이만큼 True로 채운 마스크를 만듭니다.

    stop_sign_points처럼 "있으면 전부 유효(True)"로 두고 싶은 경우에 사용합니다.

    Args:
        array: numpy 배열. shape: (N, ...)

    Returns:
        mask: bool 배열. shape: (N,)
    """
    # array: np.ndarray, shape (N, ...)
    if array.ndim < 1:
        return np.ones((0,), dtype=bool)
    n = int(array.shape[0])
    return np.ones((n,), dtype=bool)


def _compute_agent_level_valid_from_future_gt_3(
    future_gt_3: np.ndarray,) -> np.ndarray:
    """(A, future_len, 3) future GT에서 agent별 유효 여부(A,)를 만듭니다.

    규칙:
      - 한 agent의 (future_len, 3) 전체가 전부 0이면 invalid(False)
      - 어디든 하나라도 값이 있으면 valid(True)

    Args:
        future_gt_3: numpy 배열. shape: (A, future_len, 3)

    Returns:
        agent_valid: bool 배열. shape: (A,)
    """
    # future_gt_3: np.ndarray, shape (A, future_len, 3)
    if future_gt_3.ndim != 3 or int(future_gt_3.shape[-1]) != 3:
        raise ValueError(
            f"future_gt_3 must have shape (A, future_len, 3). got shape={future_gt_3.shape}"
        )

    # per_step_valid: shape (A, future_len)
    per_step_valid = _compute_valid_mask_from_prefix_nonzero(future_gt_3,
                                                             prefix_dim=3)
    # agent_valid: shape (A,)
    agent_valid = np.any(per_step_valid, axis=-1)
    return agent_valid.astype(bool)


def _add_validity_keys_inplace(sample: Dict[str, Any]) -> None:
    """sample dict에 a~n validity key들을 추가합니다.

    공통 규칙
    ----------
    - source 값이 None이면 새 key도 None으로 둡니다.
    - source 값이 있으면 요청한 규칙대로 bool 마스크를 만들어 넣습니다.

    이 함수가 추가하는 key 목록과 타입/shape
    --------------------------------------
    아래에서 `time_len`, `future_len`, `lane_len`, `route_len`, `safety_len`은
    각 샘플에 들어있는 원본 배열의 shape에서 그대로 따라옵니다.

    1) ego
      - ego_agent_past_is_valid:
          · source: "ego_agent_past" (shape: (time_len, 11))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (time_len,)
      - ego_future_gt_is_valid:
          · source: "planner_future_11_dim" 또는 "ego_future_gt_11_dim" (shape: (future_len, 11))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (future_len,)

    2) neighbor agents
      - neighbor_agents_past_is_valid:
          · source: "neighbor_agents_past" (shape: (chosen_agent_num, time_len, 11))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_agent_num, time_len)
      - neighbor_agents_is_valid:
          · source: "neighbor_agents_past"[:, -1, :] (shape: (chosen_agent_num, 11))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_agent_num,)
      - neighbor_future_gt_is_valid:
          · source: "near_future_gt_3_dim" 또는 "neighbor_future_gt_3_dim" (shape: (chosen_agent_num, future_len, 3))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_agent_num,)

    3) map / static objects / road safety
      - stop_sign_is_valid:
          · source: "stop_sign_points" (shape: (stop_sign_num, safety_len, 2))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (stop_sign_num,)
          · 규칙: 전부 True
      - crosswalk_is_valid:
          · source: "crosswalk_points" (shape: (crosswalk_num, safety_len, 2))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (crosswalk_num,)
          · 규칙: 전부 True
      - lanes_len_is_valid:
          · source: "lanes" (shape: (chosen_lane_num, lane_len, 12))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_lane_num, lane_len)
      - lanes_is_valid:
          · source: lanes_len_is_valid (shape: (chosen_lane_num, lane_len))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_lane_num,)
          · 규칙: lane_len 중 True가 하나라도 있으면 True
      - static_objects_is_valid:
          · source: "static_objects" (shape: (chosen_static_num, 10))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_static_num,)
          · 규칙: 전부 True

    4) route lanes (nuPlan 전용)
      - route_lanes_len_is_valid:
          · source: "route_lanes" (shape: (chosen_route_lane_num, route_len, 12))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_route_lane_num, route_len)
      - route_lanes_is_valid:
          · source: route_lanes_len_is_valid (shape: (chosen_route_lane_num, route_len))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_route_lane_num,)
          · 규칙: route_len 중 True가 하나라도 있으면 True
      - agent_route_lane_order_is_valid:
          · source: "agent_route_lane_order" (shape: (chosen_agent_num, chosen_lane_num))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_agent_num,)
          · 규칙: 전부 True

    5) WOMD 전용(추가 지도 요소)
      - speed_bump_is_valid:
          · source: "speed_bump_points" (shape: (speed_bump_num, safety_len, 2))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (speed_bump_num,)
          · 규칙: 전부 True
      - driveway_is_valid:
          · source: "driveway_points" 또는 "driveway" (shape: (driveway_num, safety_len, 2))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (driveway_num,)
          · 규칙: 전부 True
      - road_edge_is_valid:
          · source: "road_edge" (shape: (chosen_edge_num, safety_len, 2))
          · type: np.ndarray(dtype=bool) 또는 None
          · shape: (chosen_edge_num,)
          · 규칙: 전부 True

    Args:
        sample: __getitem__이 만들고 있는 샘플 dict.
            - 이 함수는 sample을 **in-place**로 수정해서 위 key들을 추가합니다.
    """
    # ---------- a ----------
    ego_agent_past = sample.get("ego_agent_past", None)
    if ego_agent_past is None:
        sample["ego_agent_past_is_valid"] = None
    else:
        # ego_agent_past: shape (time_len, 11)
        sample[
            "ego_agent_past_is_valid"] = _compute_valid_mask_from_prefix_nonzero(
                ego_agent_past, prefix_dim=8)  # shape (time_len,)

    # ---------- b ----------
    # rename 대응: ego_future_gt_11_dim -> planner_future_11_dim
    ego_future_11 = _get_first_non_none_value(
        sample, ["planner_future_11_dim", "ego_future_gt_11_dim"])
    if ego_future_11 is None:
        sample["ego_future_gt_is_valid"] = None
    else:
        # ego_future_11: shape (future_len, 11)
        sample[
            "ego_future_gt_is_valid"] = _compute_valid_mask_from_prefix_nonzero(
                ego_future_11, prefix_dim=8)  # shape (future_len,)

    # ---------- c ----------
    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if neighbor_agents_past is None:
        sample["neighbor_agents_past_is_valid"] = None
    else:
        # neighbor_agents_past: shape (chosen_agent_num, time_len, 11)
        sample[
            "neighbor_agents_past_is_valid"] = _compute_valid_mask_from_prefix_nonzero(
                neighbor_agents_past,
                prefix_dim=8)  # shape (chosen_agent_num, time_len)

    # ---------- d ----------
    if neighbor_agents_past is None:
        sample["neighbor_agents_is_valid"] = None
    else:
        # last_step: shape (chosen_agent_num, 11)
        last_step = neighbor_agents_past[:, -1, :]
        sample[
            "neighbor_agents_is_valid"] = _compute_valid_mask_from_prefix_nonzero(
                last_step, prefix_dim=8)  # shape (chosen_agent_num,)

    # ---------- e ----------
    # rename 대응: neighbor_future_gt_3_dim -> near_future_gt_3_dim
    neighbor_future_3 = _get_first_non_none_value(
        sample, ["near_future_gt_3_dim", "neighbor_future_gt_3_dim"])
    if neighbor_future_3 is None:
        sample["neighbor_future_gt_is_valid"] = None
    else:
        # neighbor_future_3: shape (chosen_agent_num, future_len, 3)
        sample[
            "neighbor_future_gt_is_valid"] = _compute_agent_level_valid_from_future_gt_3(
                neighbor_future_3)  # shape (chosen_agent_num,)

    # ---------- f ----------
    stop_sign_points = sample.get("stop_sign_points", None)
    if stop_sign_points is None:
        sample["stop_sign_is_valid"] = None
    else:
        # stop_sign_points: shape (stop_sign_num, safety_len, 2)
        sample["stop_sign_is_valid"] = _make_all_true_mask_from_first_dim(
            stop_sign_points)  # shape (stop_sign_num,)

    # ---------- g ----------
    crosswalk_points = sample.get("crosswalk_points", None)
    if crosswalk_points is None:
        sample["crosswalk_is_valid"] = None
    else:
        # crosswalk_points: shape (crosswalk_num, safety_len, 2)
        sample["crosswalk_is_valid"] = _make_all_true_mask_from_first_dim(
            crosswalk_points)  # shape (crosswalk_num,)

    # ---------- h ----------
    lanes = sample.get("lanes", None)
    if lanes is None:
        sample["lanes_len_is_valid"] = None
        sample["lanes_is_valid"] = None
    else:
        # lanes: shape (chosen_lane_num, lane_len, 12)
        lanes_len_is_valid = _compute_valid_mask_from_prefix_nonzero(
            lanes, prefix_dim=8)  # (chosen_lane_num, lane_len)
        sample["lanes_len_is_valid"] = lanes_len_is_valid
        sample["lanes_is_valid"] = _reduce_any_along_last_axis(
            lanes_len_is_valid)  # (chosen_lane_num,)

    # ---------- (요청문에선 h로 또 표기됨) static_objects_is_valid ----------
    static_objects = sample.get("static_objects", None)
    if static_objects is None:
        sample["static_objects_is_valid"] = None
    else:
        # static_objects: shape (chosen_static_num, 10)
        sample["static_objects_is_valid"] = _make_all_true_mask_from_first_dim(
            static_objects)  # shape (chosen_static_num,)

    # ---------- i / j ----------
    route_lanes = sample.get("route_lanes", None)
    if route_lanes is None:
        sample["route_lanes_len_is_valid"] = None
        sample["route_lanes_is_valid"] = None
    else:
        # route_lanes: shape (chosen_route_lane_num, route_len, 12)
        route_lanes_len_is_valid = _compute_valid_mask_from_prefix_nonzero(
            route_lanes, prefix_dim=8)  # (R, route_len)
        sample["route_lanes_len_is_valid"] = route_lanes_len_is_valid
        sample["route_lanes_is_valid"] = _reduce_any_along_last_axis(
            route_lanes_len_is_valid)  # (R,)

    # ---------- k ----------
    agent_route_lane_order = sample.get("agent_route_lane_order", None)
    if agent_route_lane_order is None:
        sample["agent_route_lane_order_is_valid"] = None
    else:
        # agent_route_lane_order: shape (chosen_agent_num, chosen_lane_num)
        sample[
            "agent_route_lane_order_is_valid"] = _make_all_true_mask_from_first_dim(
                agent_route_lane_order)  # (chosen_agent_num,)

    # ---------- l ----------
    speed_bump_points = sample.get("speed_bump_points", None)
    if speed_bump_points is None:
        sample["speed_bump_is_valid"] = None
    else:
        # speed_bump_points: shape (speed_bump_num, safety_len, 2)
        sample["speed_bump_is_valid"] = _make_all_true_mask_from_first_dim(
            speed_bump_points)  # (speed_bump_num,)

    # ---------- m ----------
    driveway_points = _get_first_non_none_value(sample,
                                                ["driveway_points", "driveway"])
    if driveway_points is None:
        sample["driveway_is_valid"] = None
    else:
        # driveway_points: shape (driveway_num, safety_len, 2)
        sample["driveway_is_valid"] = _make_all_true_mask_from_first_dim(
            driveway_points)  # (driveway_num,)

    # ---------- n ----------
    road_edge = sample.get("road_edge", None)
    if road_edge is None:
        sample["road_edge_is_valid"] = None
    else:
        # road_edge: shape (chosen_edge_num, safety_len, 2)
        sample["road_edge_is_valid"] = _make_all_true_mask_from_first_dim(
            road_edge)  # (chosen_edge_num,)


class DiffusionPlannerData(Dataset):

    def __init__(self, data_dir, data_list):
        """
        data_dir: "/mnt/nuplan/dataset/processed"
        data_list: "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
        """
        self.data_dir = data_dir
        self.data_list = openjson(data_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """한 샘플을 이름 기반 dict로 반환한다.
        각 key별 기본 shape는 다음과 같다 (B는 배치에서 묶일 때 앞에 붙는다).

        # 참고
            - chosen_agent_num <= caching_max_agent_num
            - chosen_lane_num <= lane_num
            - chosen_route_lane_num <= route_num
            - chosen_static_num <= max_static_num
        """
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        if data is None:
            raise IndexError(f"Corrupted sample at index {idx}")

        both_keys: List[str] = [
            "ego_agent_past",  # (time_len, 11) # nuplan # womd
            "ego_future_gt_3_dim",  # (future_len, 3) # nuplan  # womd
            "ego_future_gt_11_dim",  # (future_len, 11) # nuplan # womd
            "neighbor_agents_past",  # (chosen_agent_num, time_len, 11) # nuplan  # womd
            "neighbor_future_gt_3_dim",  # (chosen_agent_num, future_len, 3) # nuplan # womd
            "stop_sign_points",  # (stop_sign_num, safety_len, 2) # nuplan  # womd # TODO
            "crosswalk_points",  # (crosswalk_num, safety_len, 2) # nuplan  # womd # TODO
            "lanes",  # (chosen_lane_num, lane_len, 12) # nuplan # womd
            "lanes_speed_limit",  # (chosen_lane_num, 1) # nuplan # womd
            "lanes_has_speed_limit",  # (chosen_lane_num, 1) # nuplan # womd
        ]

        nuplan_only_keys: List[str] = [
            "static_objects",  # (chosen_static_num, 10) # nuplan
            "route_lanes",  # (chosen_route_lane_num, route_len, 12) # nuplan
            "route_lanes_speed_limit",  # (chosen_route_lane_num, 1) # nuplan
            "route_lanes_has_speed_limit",  # (chosen_route_lane_num, 1) # nuplan
            "agent_route_lane_order",  # (chosen_agent_num, chosen_lane_num) # nuplan
        ]

        womd_only_keys: List[str] = [
            "speed_bump_points",  # (speed_bump_num, safety_len, 2) # womd  # TODO
            "driveway_points",  # (driveway_num, safety_len, 2) # womd (환경에 따라 driveway라는 이름일 수도 있음) # TODO
            "lane_type",  # (chosen_lane_num, 4) # womd
            "left_line_type",  # (chosen_lane_num, 13) # womd
            "right_line_type",  # (chosen_lane_num, 13) # womd
            "road_edge",  # (chosen_edge_num, safety_len, 2) # womd # TODO
            "road_edge_type",  # (chosen_edge_num, 3) # womd
        ]

        npz_keys: List[str] = both_keys + nuplan_only_keys + womd_only_keys

        npz_key_to_new_key: Dict[str, str] = {
            "ego_future_gt_11_dim": "planner_future_11_dim",
            "neighbor_future_gt_3_dim": "near_future_gt_3_dim",
        }

        sample: Dict[str, Any] = {}

        for npz_key in npz_keys:
            value = data.get(npz_key, None)
            if value is not None and npz_key == "agent_route_lane_order":
                value = value.astype("int64")
            out_key = npz_key_to_new_key.get(npz_key, npz_key)
            sample[out_key] = value

        # a~n validity key 추가
        _add_validity_keys_inplace(sample)

        return sample
