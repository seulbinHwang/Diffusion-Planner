from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import time
from torch.utils.data import Dataset
from nuplan_extent.planning.training.preprocessing.utils.near_agents import add_near_agents_info_inplace
from diffusion_planner.utils.validity import add_validity_keys_inplace
from diffusion_planner.utils.train_utils import openjson, opendata

import numpy as np
from numpy.typing import NDArray
from typing import Any, List, Sequence, Tuple
import os
import numpy as np


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
        """sample dict에 ego/near의 *_future_gt_4_dim 키를 추가합니다.

        - sample을 새로 만들지 않고, 들어온 sample dict를 그대로 수정합니다.
        - 필요한 입력 키:
          - ego: "ego_future_gt_3_dim", "ego_future_gt_is_valid"
          - near: "near_future_gt_3_dim", "near_future_gt_is_valid"

        Args:
            sample:
                __getitem__에서 만드는 샘플 dict.
        """
        # ego
        ego_future_gt_3_dim = sample["ego_future_gt_3_dim"]  # (future_len, 3)
        ego_future_gt_is_valid = sample[
            "ego_future_gt_is_valid"]  # (future_len,)
        sample["ego_future_gt_4_dim"] = self._build_future_gt_4_dim_from_3_dim(
            ego_future_gt_3_dim,
            ego_future_gt_is_valid,
        )  # (future_len, 4)

        # near
        near_future_gt_3_dim = sample[
            "near_future_gt_3_dim"]  # (Pnn, future_len, 3)
        near_future_gt_is_valid = sample[
            "near_future_gt_is_valid"]  # (Pnn, future_len)
        sample["near_future_gt_4_dim"] = self._build_future_gt_4_dim_from_3_dim(
            near_future_gt_3_dim,
            near_future_gt_is_valid,
        )  # (Pnn, future_len, 4)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """한 샘플을 이름 기반 dict로 반환한다.
        각 key별 기본 shape는 다음과 같다 (B는 배치에서 묶일 때 앞에 붙는다).

        # 참고
            - chosen_agent_num <= caching_max_agent_num
            - chosen_lane_num <= lane_num
            - chosen_route_lane_num <= route_num
            - chosen_static_num <= max_static_num

        womd 추가 : scenario_id /

        """
        file_name = self.data_list[idx]
        data = opendata(os.path.join(self.data_dir, file_name))
        if data is None:
            raise IndexError(f"Corrupted sample at index {idx}")
        both_keys: List[str] = [
            "origin_world_pose",  # (4,)  # nuplan  # womd
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
            "static_objects",  #check # (chosen_static_num, 10) # nuplan
            "route_lanes",  # (chosen_route_lane_num, route_len, 12) # nuplan
            "route_lanes_speed_limit",  # (chosen_route_lane_num, 1) # nuplan
            "route_lanes_has_speed_limit",  # (chosen_route_lane_num, 1) # nuplan
            "agent_route_lane_order",  # (chosen_agent_num, chosen_lane_num) # nuplan
        ]

        womd_only_keys: List[str] = [
            "speed_bump_points",  # (speed_bump_num, safety_len, 2) # womd
            "driveway_points",  # (driveway_num, safety_len, 2) # womd (환경에 따라 driveway라는 이름일 수도 있음)
            "driveway",  # (driveway_num, safety_len, 2)
            "lane_type",  # (chosen_lane_num, 4) # womd
            "left_line_type",  # (chosen_lane_num, 13) # womd
            "right_line_type",  # (chosen_lane_num, 13) # womd
            "road_edge",  # (chosen_edge_num, safety_len, 2) # womd
            "road_edge_type",  # (chosen_edge_num, 3) # womd
        ]
        wosac_only_keys: List[str] = []
        if self.eval_method in ("validation", "test"):
            wosac_only_keys = [
                "target_id",
                "target_z",
            ]

        npz_keys: List[
            str] = both_keys + nuplan_only_keys + womd_only_keys + wosac_only_keys

        npz_key_to_new_key: Dict[str, str] = {
            "ego_future_gt_11_dim": "planner_future_11_dim",
            "driveway": "driveway_points",
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
        add_near_agents_info_inplace(
            sample,
            predicted_neighbor_num=self.predicted_neighbor_num,
        )
        scenario_id = str(os.path.splitext(file_name)[0])
        sample["scenario_id"] = scenario_id

        # ✅ 여기만 한 줄로 정리
        self._add_future_gt_4_dim_keys_inplace(sample)

        if self.eval_method == "validation":
            tfrecord_file_name = file_name.replace(".npz", ".tfrecords")
            tfrecord_path = os.path.join(self.data_tfrecords_dir,
                                         tfrecord_file_name)
            sample["tfrecord_path"] = tfrecord_path
            # TODO: 다시 복구해야함
            # if not os.path.exists(tfrecord_path):

            # raise FileNotFoundError(
            #     f"TFRecords file not found: {tfrecord_path}")
            # sample["tfrecord_path"] = tfrecord_path

        return sample
