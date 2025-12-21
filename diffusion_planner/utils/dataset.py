import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset
from diffusion_planner.utils.validity import add_validity_keys_inplace
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
            "stop_sign_points",  # (stop_sign_num, safety_len, 2) # nuplan  # womd
            "crosswalk_points",  # (crosswalk_num, safety_len, 2) # nuplan  # womd
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
            "speed_bump_points",  # (speed_bump_num, safety_len, 2) # womd
            "driveway_points",  # (driveway_num, safety_len, 2) # womd (환경에 따라 driveway라는 이름일 수도 있음)
            "lane_type",  # (chosen_lane_num, 4) # womd
            "left_line_type",  # (chosen_lane_num, 13) # womd
            "right_line_type",  # (chosen_lane_num, 13) # womd
            "road_edge",  # (chosen_edge_num, safety_len, 2) # womd
            "road_edge_type",  # (chosen_edge_num, 3) # womd
        ]

        npz_keys: List[str] = both_keys + nuplan_only_keys + womd_only_keys

        npz_key_to_new_key: Dict[str, str] = {
            "ego_future_gt_11_dim": "planner_future_11_dim",
        }

        sample: Dict[str, Any] = {}
        try:
            for npz_key in npz_keys:
                value = data.get(npz_key, None)
                if value is not None and npz_key == "agent_route_lane_order":
                    value = value.astype("int64")
                out_key = npz_key_to_new_key.get(npz_key, npz_key)
                sample[out_key] = value
        finally:
            # opendata가 np.load(...) 결과(NpzFile)를 반환하므로 닫아주는 게 안전
            if hasattr(data, "close"):
                try:
                    data.close()
                except Exception:
                    pass
        # a~n validity key 추가 (공통 유틸)
        add_validity_keys_inplace(sample, missing_policy="none")
        return sample
