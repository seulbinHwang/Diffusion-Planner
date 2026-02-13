from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import time
from torch.utils.data import Dataset
from nuplan_extent.planning.training.preprocessing.utils.near_agents import add_near_agents_info_inplace
from diffusion_planner.utils.validity import add_validity_keys_inplace
from diffusion_planner.utils.train_utils import openjson, opendata

from numpy.typing import NDArray
from typing import Any, List, Sequence, Tuple
import os
import numpy as np
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# --- add_control_to_npz.py로 미리 저장될 수 있는 key들 ---
_PRECOMPUTED_VALIDITY_KEYS: List[str] = [
    "ego_agent_past_is_valid",
    "ego_future_gt_is_valid",
    "neighbor_agents_past_is_valid",
    "neighbor_agents_is_valid",
    "neighbor_future_gt_is_valid",
    "stop_sign_is_valid",
    "crosswalk_is_valid",
    "lanes_len_is_valid",
    "lanes_is_valid",
    "static_objects_is_valid",
    "route_lanes_len_is_valid",
    "route_lanes_is_valid",
    "agent_route_lane_order_is_valid",
    "speed_bump_is_valid",
    "driveway_is_valid",
    "road_edge_is_valid",
]

_PRECOMPUTED_NEAR_KEYS: List[str] = [
    "near_agents_past",
    "non_near_agents_past",
    "near_future_gt_3_dim",
    "near_agents_past_is_valid",
    "non_near_agents_past_is_valid",
    "near_agents_is_valid",
    "non_near_agents_is_valid",
    "near_future_gt_is_valid",
    "non_near_future_gt_is_valid",
]

_PRECOMPUTED_GT4_KEYS: List[str] = [
    "ego_future_gt_4_dim",
    "near_future_gt_4_dim",
]

_PRECOMPUTED_CONTROL_KEYS: List[str] = [
    "past_future_seg_control_gt_3_dim",
]


def _unique_keep_order(items: Sequence[str]) -> List[str]:
    """중복을 제거하되, 처음 등장한 순서는 유지합니다."""
    seen = set()
    out: List[str] = []
    for x in items:
        k = str(x)
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def _is_np_array(value: Any) -> bool:
    """value가 numpy 배열인지 확인합니다."""
    return isinstance(value, np.ndarray)


def _infer_agent_dim_from_neighbor_agents_past(neighbor_agents_past: np.ndarray) -> Optional[int]:
    """neighbor_agents_past에서 agent 축이 몇 번째인지 추정합니다.

    Args:
        neighbor_agents_past (np.ndarray):
            - (A, T, F) 또는 (B, A, T, F) 형태를 기대합니다.

    Returns:
        Optional[int]:
            - (A, T, F)면 0
            - (B, A, T, F)면 1
            - 그 외면 None
    """
    if not _is_np_array(neighbor_agents_past):
        return None
    if int(neighbor_agents_past.ndim) == 3:
        return 0
    if int(neighbor_agents_past.ndim) == 4:
        return 1
    return None


def _slice_np_along_dim(value: np.ndarray, dim: int, end: int) -> np.ndarray:
    """numpy 배열을 특정 축 기준으로 [:end] 슬라이스합니다."""
    if int(value.ndim) <= int(dim):
        return value
    slices = [slice(None)] * int(value.ndim)
    slices[int(dim)] = slice(0, int(end))
    return value[tuple(slices)]


def _ensure_validity_keys_if_needed_inplace(sample: Dict[str, Any]) -> None:
    """validity key를 '이미 있으면 그대로 쓰고', 없으면 계산해서 채웁니다.

    규칙
    ----
    - key가 이미 있고 np.ndarray면: 그대로 둡니다.
    - key가 없거나 None인데, 원본 입력도 없으면: 그 key는 None으로 채워둡니다.
    - key가 없거나 None인데, 원본 입력은 있으면: add_validity_keys_inplace로 계산합니다.
    """
    # 어떤 validity는 입력이 없으면 None이 정상인데, npz에는 None을 저장하지 않았을 수 있음.
    # 그래서 "입력도 없으면 None으로만 채움"을 먼저 하고, 정말 계산이 필요한 경우만 계산합니다.

    def _has_source_for(key: str) -> bool:
        if key == "ego_future_gt_is_valid":
            # planner_future_11_dim(=ego_future_gt_11_dim rename)이 있으면 계산 가능
            return sample.get("planner_future_11_dim", None) is not None or sample.get("ego_future_gt_11_dim", None) is not None
        if key == "driveway_is_valid":
            return sample.get("driveway_points", None) is not None or sample.get("driveway", None) is not None
        if key in ("lanes_len_is_valid", "lanes_is_valid"):
            return sample.get("lanes", None) is not None
        if key in ("route_lanes_len_is_valid", "route_lanes_is_valid"):
            return sample.get("route_lanes", None) is not None
        if key == "static_objects_is_valid":
            return sample.get("static_objects", None) is not None
        if key == "stop_sign_is_valid":
            return sample.get("stop_sign_points", None) is not None
        if key == "crosswalk_is_valid":
            return sample.get("crosswalk_points", None) is not None
        if key == "speed_bump_is_valid":
            return sample.get("speed_bump_points", None) is not None
        if key == "road_edge_is_valid":
            return sample.get("road_edge", None) is not None
        if key == "agent_route_lane_order_is_valid":
            return sample.get("agent_route_lane_order", None) is not None
        if key == "ego_agent_past_is_valid":
            return sample.get("ego_agent_past", None) is not None
        if key in ("neighbor_agents_past_is_valid", "neighbor_agents_is_valid"):
            return sample.get("neighbor_agents_past", None) is not None
        if key == "neighbor_future_gt_is_valid":
            return sample.get("neighbor_future_gt_11_dim", None) is not None
        return False

    need_compute = False
    for k in _PRECOMPUTED_VALIDITY_KEYS:
        v = sample.get(k, None)
        if _is_np_array(v):
            continue

        # v가 None/미존재인데, source도 없으면 None으로 채워서 "키 존재"만 맞춤
        if not _has_source_for(k):
            sample[k] = None
            continue

        # source는 있는데 값이 없으면 계산 필요
        need_compute = True

    if need_compute:
        # 기존 공통 유틸 그대로 사용 (규칙 통일)
        add_validity_keys_inplace(sample, missing_policy="none")


def _expected_near_num_from_sample(
    sample: Dict[str, Any],
    *,
    predicted_neighbor_num: int,
    agent_dim: int,
    has_batch_dim: bool,
    use_agent_route_lane_order: bool,
) -> int:
    """현재 sample 내용과 설정값으로 'near로 남아야 하는 agent 수'를 계산합니다."""
    requested = max(0, int(predicted_neighbor_num))

    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if not _is_np_array(neighbor_agents_past):
        return 0

    near_num = min(requested, int(neighbor_agents_past.shape[int(agent_dim)]))

    neighbor_future = sample.get("neighbor_future_gt_3_dim", None)
    if _is_np_array(neighbor_future):
        expected_ndim = 4 if has_batch_dim else 3
        if int(neighbor_future.ndim) == int(expected_ndim):
            near_num = min(near_num, int(neighbor_future.shape[int(agent_dim)]))

    if use_agent_route_lane_order:
        aro = sample.get("agent_route_lane_order", None)
        if _is_np_array(aro):
            expected_ndim = 3 if has_batch_dim else 2
            if int(aro.ndim) == int(expected_ndim):
                near_num = min(near_num, int(aro.shape[int(agent_dim)]))

        aro_valid = sample.get("agent_route_lane_order_is_valid", None)
        if _is_np_array(aro_valid):
            expected_ndim = 2 if has_batch_dim else 1
            if int(aro_valid.ndim) == int(expected_ndim):
                near_num = min(near_num, int(aro_valid.shape[int(agent_dim)]))

    return int(near_num)


def _near_keys_are_ready(
    sample: Dict[str, Any],
    *,
    near_num: int,
    agent_dim: int,
    has_batch_dim: bool,
) -> bool:
    """near 관련 key들이 이미 있고, near_num과 shape도 맞는지 확인합니다."""
    near_agents_past = sample.get("near_agents_past", None)
    non_near_agents_past = sample.get("non_near_agents_past", None)
    if (not _is_np_array(near_agents_past)) or (not _is_np_array(non_near_agents_past)):
        return False

    # neighbor_agents_past와 같은 ndim(3 또는 4)이어야 함
    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if not _is_np_array(neighbor_agents_past):
        return False

    if int(near_agents_past.ndim) != int(neighbor_agents_past.ndim):
        return False
    if int(near_agents_past.shape[int(agent_dim)]) != int(near_num):
        return False

    # neighbor_future가 있으면 near_future_gt_3_dim도 있어야 함
    neighbor_future = sample.get("neighbor_future_gt_3_dim", None)
    if _is_np_array(neighbor_future):
        expected_ndim = 4 if has_batch_dim else 3
        if int(neighbor_future.ndim) == int(expected_ndim):
            near_future = sample.get("near_future_gt_3_dim", None)
            if not _is_np_array(near_future):
                return False
            if int(near_future.shape[int(agent_dim)]) != int(near_num):
                return False

    # neighbor_*_is_valid가 "배열"로 있으면 near_*_is_valid도 있어야 함
    neighbor_agents_past_is_valid = sample.get("neighbor_agents_past_is_valid", None)
    if _is_np_array(neighbor_agents_past_is_valid):
        expected_ndim = 3 if has_batch_dim else 2
        if int(neighbor_agents_past_is_valid.ndim) == int(expected_ndim):
            if not _is_np_array(sample.get("near_agents_past_is_valid", None)):
                return False
            if not _is_np_array(sample.get("non_near_agents_past_is_valid", None)):
                return False

    neighbor_agents_is_valid = sample.get("neighbor_agents_is_valid", None)
    if _is_np_array(neighbor_agents_is_valid):
        expected_ndim = 2 if has_batch_dim else 1
        if int(neighbor_agents_is_valid.ndim) == int(expected_ndim):
            if not _is_np_array(sample.get("near_agents_is_valid", None)):
                return False
            if not _is_np_array(sample.get("non_near_agents_is_valid", None)):
                return False

    neighbor_future_gt_is_valid = sample.get("neighbor_future_gt_is_valid", None)
    if _is_np_array(neighbor_future_gt_is_valid):
        expected_ndim = 3 if has_batch_dim else 2
        if int(neighbor_future_gt_is_valid.ndim) == int(expected_ndim):
            if not _is_np_array(sample.get("near_future_gt_is_valid", None)):
                return False
            if not _is_np_array(sample.get("non_near_future_gt_is_valid", None)):
                return False

    return True


def _ensure_near_keys_if_needed_inplace(
    sample: Dict[str, Any],
    *,
    predicted_neighbor_num: int,
    use_agent_route_lane_order: bool,
) -> None:
    """near 관련 key를 '이미 있으면 그대로 쓰고', 없거나 안 맞으면 다시 만듭니다."""
    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if not _is_np_array(neighbor_agents_past):
        return

    agent_dim = _infer_agent_dim_from_neighbor_agents_past(neighbor_agents_past)
    if agent_dim is None:
        # shape가 예상과 다르면 기존 유틸이 알아서 스킵할 수도 있으니 호출만 해봄
        add_near_agents_info_inplace(sample, predicted_neighbor_num=int(predicted_neighbor_num))
        return

    has_batch_dim = (int(agent_dim) == 1)

    near_num = _expected_near_num_from_sample(
        sample,
        predicted_neighbor_num=int(predicted_neighbor_num),
        agent_dim=int(agent_dim),
        has_batch_dim=bool(has_batch_dim),
        use_agent_route_lane_order=bool(use_agent_route_lane_order),
    )

    if _near_keys_are_ready(sample, near_num=int(near_num), agent_dim=int(agent_dim), has_batch_dim=bool(has_batch_dim)):
        # add_control_to_npz는 원본 key를 덮어쓰지 않게 했으므로,
        # sample dict에서는 agent_route_lane_order만 near 길이로 맞춰줍니다(필요한 경우).
        if use_agent_route_lane_order:
            aro = sample.get("agent_route_lane_order", None)
            if _is_np_array(aro):
                sample["agent_route_lane_order"] = _slice_np_along_dim(aro, dim=int(agent_dim), end=int(near_num))

            aro_valid = sample.get("agent_route_lane_order_is_valid", None)
            if _is_np_array(aro_valid):
                sample["agent_route_lane_order_is_valid"] = _slice_np_along_dim(aro_valid, dim=int(agent_dim), end=int(near_num))
        return

    # 없거나 shape가 안 맞으면 원래 유틸로 재생성
    add_near_agents_info_inplace(sample, predicted_neighbor_num=int(predicted_neighbor_num))


def _normalize_use_data_percent(use_data_percent: Any) -> float:
    """사용할 비율(%) 값을 0~100 범위로 안전하게 정리합니다."""
    if use_data_percent is None:
        return 100.0

    try:
        percent = float(use_data_percent)
    except (TypeError, ValueError):
        return 100.0

    # NaN/Inf 방지
    if not bool(np.isfinite(percent)):
        return 100.0

    # ✅ 음수(예: args 기본값 -100.0)는 "미설정"으로 보고 100%로 처리
    if percent < 0.0:
        return 100.0

    # 0~100 clamp
    percent = max(0.0, min(100.0, percent))
    return float(percent)


def _compute_keep_count_from_percent(
    total_count: int,
    use_data_percent: float,
) -> int:
    """전체 개수에서 '앞쪽 N%'만 쓸 때 실제로 남길 개수를 계산합니다.

    규칙
    ----
    - 리스트를 섞지 않고 앞에서부터 잘라 쓰는 방식이므로,
      keep_count개만 남기면 됩니다.
    - total_count > 0 인데,
      비율이 너무 작아서 계산 결과가 0이 되면(예: 0.1%),
      파이프라인이 아예 비는 것을 막기 위해 최소 1개는 남깁니다.

    Args:
        total_count (int):
            전체 항목 수. shape: ()
        use_data_percent (float):
            사용할 비율(0~100). shape: ()

    Returns:
        int:
            남길 항목 수. shape: ()
            - 0 <= keep_count <= total_count
            - total_count > 0 이면 keep_count는 최소 1
    """
    total = int(total_count)
    if total <= 0:
        return 0

    percent = float(use_data_percent)

    # percent가 0이어도 "완전 빈 데이터"로 들어가면 validation 로직이 깨질 가능성이 큼
    # (metric compute 등). 그래서 최소 1개는 남기도록 처리합니다.
    if percent <= 0.0:
        return 1

    keep = int(total * percent / 100.0)

    # 너무 작아서 0이 된 경우 최소 1
    keep = max(1, keep)

    # 최대는 total
    keep = min(total, keep)
    return int(keep)


def _select_first_n_percent_items(
    data_list: Sequence[str],
    use_data_percent: Any,
) -> Tuple[List[str], float, int, int]:
    """리스트를 섞지 않고 '앞쪽 N%'만 남깁니다.

    Args:
        data_list (Sequence[str]):
            파일 이름 리스트. length=total_count
        use_data_percent (Any):
            사용할 비율(%). shape: ()

    Returns:
        Tuple[List[str], float, int, int]:
            (selected_list, normalized_percent, total_count, keep_count)
            - selected_list: 앞쪽 keep_count개만 남긴 리스트
            - normalized_percent: 0~100으로 정리된 percent
            - total_count: 원래 전체 개수
            - keep_count: 실제로 남긴 개수
    """
    total_count = int(len(data_list))
    normalized_percent = _normalize_use_data_percent(use_data_percent)
    keep_count = _compute_keep_count_from_percent(total_count,
                                                  normalized_percent)

    selected_list = list(data_list[:keep_count])
    return selected_list, float(normalized_percent), total_count, int(
        keep_count)


def _should_print_dataset_subset_info() -> bool:
    """여러 프로세스로 도는 경우에도 로그를 한 번만 찍도록 판단합니다.

    - torchrun을 쓰면 RANK 환경변수가 들어오는 경우가 많습니다.
    - 보통 rank 0만 출력하면 로그가 깔끔합니다.

    Returns:
        bool:
            - True면 출력
            - False면 출력 생략
    """
    rank_str = os.environ.get("RANK", "0")
    try:
        rank = int(rank_str)
    except ValueError:
        rank = 0
    return rank == 0


def _print_dataset_subset_info(
    *,
    eval_method: str,
    data_list_path: Any,
    total_count: int,
    keep_count: int,
    use_data_percent: float,
) -> None:
    """데이터를 얼마나 줄여서 쓸지 한 줄로 출력합니다.

    Args:
        eval_method (str):
            "train" / "validation" 등. shape: ()
        data_list_path (Any):
            json 경로(문자열) 또는 그와 비슷한 값. shape: ()
        total_count (int):
            전체 항목 수. shape: ()
        keep_count (int):
            실제로 사용할 항목 수. shape: ()
        use_data_percent (float):
            사용할 비율(정리된 값). shape: ()
    """
    if not _should_print_dataset_subset_info():
        return

    try:
        list_name = os.path.basename(str(data_list_path))
    except Exception:
        list_name = str(data_list_path)

    print(
        f"[DiffusionPlannerData] eval_method='{eval_method}' | data_list='{list_name}' | "
        f"total={int(total_count)} | use_percent={float(use_data_percent):.2f}% | "
        f"using_first={int(keep_count)}")


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

    def __init__(
        self,
        data_dir,
        data_list,
        predicted_neighbor_num,
        eval_method: str = "train",
        use_data_percent: float = 100.0,
        use_agent_route_lane_order: bool = False,
    ):
        """
        data_dir: "/mnt/nuplan/dataset/processed"
                "${WOMD_PATH}/processed_womd_final/validation"
        data_list: "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"

        use_data_percent:
            - 전체 data_list 중 앞쪽 N%만 사용할지 결정하는 값입니다. shape: ()
            - 리스트를 섞지 않고, 시작 지점부터 앞에서부터 잘라서 씁니다.
            - 예: 10.0 -> 앞 10%만 사용
        """
        self.data_dir = data_dir
        self.data_tfrecords_dir = None
        self.use_agent_route_lane_order = bool(use_agent_route_lane_order)


        # 1) json에서 파일 리스트 로드 (순서 유지)
        loaded_list = openjson(data_list)
        if not isinstance(loaded_list, list):
            raise TypeError(
                f"openjson(data_list) 결과는 list여야 합니다. got type={type(loaded_list)}"
            )

        # 2) 앞쪽 N%만 남기기 (섞지 않음, 고정)
        selected_list, normalized_percent, total_count, keep_count = \
            _select_first_n_percent_items(
                data_list=loaded_list,
                use_data_percent=use_data_percent,
            )
        print(
            f"[DiffusionPlannerData] Loaded {len(loaded_list)} items from '{data_list}'"
        )
        print(
            f"[DiffusionPlannerData] Using first {keep_count} items ({normalized_percent:.2f}%)"
        )
        self.data_list = selected_list
        self.predicted_neighbor_num = predicted_neighbor_num
        self.eval_method = eval_method

        # 3) 어떤 비율로 얼마나 쓰는지 출력
        _print_dataset_subset_info(
            eval_method=str(self.eval_method),
            data_list_path=data_list,
            total_count=int(total_count),
            keep_count=int(keep_count),
            use_data_percent=float(normalized_percent),
        )

        # 4) validation일 때 tfrecords dir 설정 (기존 로직 유지)
        if self.eval_method == "validation":
            # self.data_dir: "/mnt/nuplan/dataset/processed"
            parent_dir = os.path.dirname(self.data_dir)  # "/mnt/nuplan/dataset"
            last_dir_name = os.path.basename(self.data_dir)  # "processed"
            # "/mnt/nuplan/dataset/processed_tfrecords_splitted"
            self.data_tfrecords_dir = os.path.join(
                parent_dir, f"{last_dir_name}_tfrecords_splitted")

    def __len__(self):
        return len(self.data_list)

    @staticmethod
    def assert_cur_future_valid_mask_np(
        valid_bpt: NDArray[np.bool_],
        *,
        context: str = "savgol_filter_for_control",
    ) -> None:
        """유효 마스크가 행마다 True*False* (단조 감소)인지 검사합니다(NumPy 버전).

        의미
        ----
        시간축을 왼쪽→오른쪽으로 볼 때,
        한 번 False(무효)가 된 이후에는 다시 True(유효)로 돌아오면 안 됩니다.
        즉, 각 (b, p) 행이 아래 패턴만 허용됩니다.

          - 허용: [True, True, True, False, False]
          - 금지: [True, False, True, False]  (중간에 구멍)
          - 금지: [False, True, True, ...]   (무효였다가 다시 유효)

        Args:
            valid_bpt (np.ndarray):
                유효 마스크.
                shape: (B, Pnn, T1)
                dtype: bool (또는 0/1 같은 값이면 bool로 해석됨)
                - True: 유효
                - False: 무효
            context (str):
                에러 메시지에 표시할 호출 위치 문자열.

        Raises:
            ValueError:
                0→1 전이(False→True)가 하나라도 발견되면 발생합니다.
        """
        v0 = np.asarray(valid_bpt)
        if v0.ndim != 3:
            raise ValueError(
                f"valid_bpt must be (B,Pnn,T1). got shape={v0.shape}")

        B, Pnn, T1 = v0.shape
        v = v0.astype(np.bool_).reshape(-1, T1).astype(np.int8)  # (B*Pnn, T1)

        # d[t] = v[t+1] - v[t]  ->  (0→1) 이면 +1
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
        has_01 = (d > 0).any(axis=1)  # (B*Pnn,)

        if np.any(has_01):
            bad_idx = np.nonzero(has_01)[0]  # (N_bad,)
            max_show = min(int(bad_idx.size), 8)
            bad_idx_sample = bad_idx[:max_show]

            b_list = (bad_idx_sample // Pnn).tolist()
            p_list = (bad_idx_sample % Pnn).tolist()

            raise ValueError(
                f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*).\n"
                f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.size)},\n"
                f"example (b,p)={list(zip(b_list, p_list))}.\n"
                f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
            )

    @staticmethod
    def _build_future_gt_4_dim_from_3_dim(
        future_gt_3_dim: NDArray[np.floating],
        future_gt_is_valid: NDArray[np.bool_],
    ) -> NDArray[np.floating]:
        """(…, 3) 미래 GT를 (…, 4)로 바꿉니다.

        변환 규칙
        --------
        - 입력 마지막 축 3개는 (x, y, 방향각)이라고 가정합니다.
        - 출력 마지막 축 4개는 (x, y, cos(방향각), sin(방향각)) 입니다.
        - future_gt_is_valid가 False인 위치는 출력 값을 0.0으로 만듭니다.

        Args:
            future_gt_3_dim:
                미래 GT. shape: (..., 3)
                예:
                  - ego: (future_len, 3)
                  - near: (predicted_neighbor_num, future_len, 3)
            future_gt_is_valid:
                유효 마스크. shape: (...)  (마지막 차원(3)은 제외한 shape)
                예:
                  - ego: (future_len,)
                  - near: (predicted_neighbor_num, future_len)

        Returns:
            future_gt_4_dim:
                변환된 미래 GT. shape: (..., 4)
        """
        # future_gt_3_dim: (..., 3)
        heading = future_gt_3_dim[..., 2:3]  # (..., 1)
        cos_heading = np.cos(heading)  # (..., 1)
        sin_heading = np.sin(heading)  # (..., 1)

        future_gt_4_dim = np.concatenate(
            [future_gt_3_dim[..., :2], cos_heading, sin_heading],
            axis=-1,
        )  # (..., 4)

        # future_gt_is_valid: (...)  -> future_gt_4_dim: (..., 4)
        future_gt_4_dim[~future_gt_is_valid] = 0.0
        return future_gt_4_dim

    def _add_future_gt_4_dim_keys_inplace(self, sample: Dict[str, Any]) -> None:
        """sample dict에 ego/near의 *_future_gt_4_dim 키를 필요할 때만 추가합니다.

        규칙
        ----
        - key가 이미 있고 shape가 맞으면 그대로 둡니다.
        - 없거나 shape가 안 맞으면 계산해서 채웁니다.
        """
        # ego
        ego_gt3 = sample.get("ego_future_gt_3_dim", None)          # (Tf, 3)
        ego_valid = sample.get("ego_future_gt_is_valid", None)     # (Tf,)
        ego_gt4 = sample.get("ego_future_gt_4_dim", None)          # (Tf, 4) 예상

        if _is_np_array(ego_gt3) and _is_np_array(ego_valid):
            need_ego = True
            if _is_np_array(ego_gt4):
                if int(ego_gt4.ndim) == 2 and int(ego_gt4.shape[-1]) == 4 and ego_gt4.shape[0] == ego_gt3.shape[0]:
                    need_ego = False
            if need_ego:
                sample["ego_future_gt_4_dim"] = self._build_future_gt_4_dim_from_3_dim(
                    ego_gt3,
                    ego_valid,
                )

        # near
        near_gt3 = sample.get("near_future_gt_3_dim", None)        # (Pnn, Tf, 3)
        near_valid = sample.get("near_future_gt_is_valid", None)   # (Pnn, Tf)
        near_gt4 = sample.get("near_future_gt_4_dim", None)        # (Pnn, Tf, 4) 예상

        if _is_np_array(near_gt3) and _is_np_array(near_valid):
            need_near = True
            if _is_np_array(near_gt4):
                if int(near_gt4.ndim) == 3 and int(near_gt4.shape[-1]) == 4 and near_gt4.shape[:2] == near_gt3.shape[:2]:
                    need_near = False
            if need_near:
                sample["near_future_gt_4_dim"] = self._build_future_gt_4_dim_from_3_dim(
                    near_gt3,
                    near_valid,
                )


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_name = self.data_list[idx]
        data = opendata(os.path.join(self.data_dir, file_name))
        if data is None:
            raise IndexError(f"Corrupted sample at index {idx}")

        both_keys: List[str] = [
            "origin_world_pose",
            "ego_agent_past",
            "ego_future_gt_3_dim",
            "ego_future_gt_11_dim",
            "neighbor_agents_past",
            "neighbor_future_gt_3_dim",
            "neighbor_future_gt_11_dim",
            "stop_sign_points",
            "crosswalk_points",
            "lanes",
            "lanes_speed_limit",
            "lanes_has_speed_limit",
        ]

        # ✅ 전처리로 추가된 control 키도 읽어오도록 포함(없으면 None)
        both_keys += _PRECOMPUTED_CONTROL_KEYS

        nuplan_only_keys: List[str] = [
            "static_objects",
            "route_lanes",
            "route_lanes_speed_limit",
            "route_lanes_has_speed_limit",
        ]
        if self.use_agent_route_lane_order:
            nuplan_only_keys.append("agent_route_lane_order")

        womd_only_keys: List[str] = [
            "speed_bump_points",
            "driveway_points",
            "lane_type",
            "left_line_type",
            "right_line_type",
            "road_edge",
            "road_edge_type",
        ]

        wosac_only_keys: List[str] = []
        if self.eval_method in ("validation", "test"):
            wosac_only_keys = [
                "target_id",
                "target_z",
            ]

        # ✅ 전처리로 저장된 파생 key들도 "읽을 수 있으면" 읽습니다.
        precomputed_extra_keys: List[str] = (
            ["scenario_id"]
            + _PRECOMPUTED_VALIDITY_KEYS
            + _PRECOMPUTED_NEAR_KEYS
            + _PRECOMPUTED_GT4_KEYS
        )

        npz_keys: List[str] = _unique_keep_order(
            both_keys + nuplan_only_keys + womd_only_keys + wosac_only_keys + precomputed_extra_keys
        )

        npz_key_to_new_key: Dict[str, str] = {
            "ego_future_gt_11_dim": "planner_future_11_dim",
            "driveway": "driveway_points",
        }

        sample: Dict[str, Any] = {}
        try:
            for npz_key in npz_keys:
                value = data.get(npz_key, None)

                if value is not None and npz_key == "agent_route_lane_order":
                    try:
                        value = np.asarray(value).astype("int64")
                    except Exception:
                        pass

                out_key = npz_key_to_new_key.get(npz_key, npz_key)
                sample[out_key] = value
        finally:
            if hasattr(data, "close"):
                try:
                    data.close()
                except Exception:
                    pass

        # ✅ scenario_id는 항상 "파이썬 str"로 통일 (npz에 저장된 값 타입 차이 방지)
        scenario_id = str(os.path.splitext(file_name)[0])
        sample["scenario_id"] = scenario_id

        # (1) validity: 이미 있으면 skip, 없으면 계산
        _ensure_validity_keys_if_needed_inplace(sample)

        # (2) near split: 이미 있으면 skip, 없거나 predicted_neighbor_num이 달라서 안 맞으면 계산
        _ensure_near_keys_if_needed_inplace(
            sample,
            predicted_neighbor_num=int(self.predicted_neighbor_num),
            use_agent_route_lane_order=bool(self.use_agent_route_lane_order),
        )

        # (3) future_gt_4_dim: 이미 있으면 skip, 없으면 계산
        self._add_future_gt_4_dim_keys_inplace(sample)

        if self.eval_method == "validation":
            tfrecord_file_name = file_name.replace(".npz", ".tfrecords")
            tfrecord_path = os.path.join(self.data_tfrecords_dir, tfrecord_file_name)
            sample["tfrecord_path"] = tfrecord_path

        return sample
