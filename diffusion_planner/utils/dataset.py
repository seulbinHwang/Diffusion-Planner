import os
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata
from typing import Dict, Any, List  # ← 추가


def _fix_legacy_neighbor_future_len_bug(
    data: Dict[str, Any],
    expected_future_len: int,
    key: str = "neighbor_future_gt_3_dim",
) -> None:
    """구버전 캐시의 neighbor_future_gt_3_dim 길이 버그를 보정하는 임시 함수.

    DataProcessor의 옛 버전에서 neighbor_future_gt_3_dim 이
    (agent_num, future_len+1, 3) 형태로 저장되는 버그가 있었다.
    (0번째 time 스텝이 현재 프레임, 나머지 1..future_len 이 미래 프레임)

    이 함수는 npz를 다시 캐싱하지 않고도,
    런타임에서 아래와 같이 shape를 맞춰준다.

    - 배열의 shape 가 (N, expected_future_len + 1, D) 이면
        → 0번째 time 스텝을 잘라내고 (N, expected_future_len, D) 로 바꾼다.
    - 이미 (N, expected_future_len, D) 이면 아무 것도 하지 않는다.
    - 그 외 길이는 건드리지 않는다. (다른 문제이므로 그대로 예외가 나게 둠)

    Args:
        data: opendata(...) 로부터 읽은 npz dict.
        expected_future_len: 학습에서 기대하는 future_len (예: 80).
        key: 시간 길이 버그가 있었던 npz 상의 키 이름.
    """
    if key not in data:
        return

    arr = data[key]
    # numpy.ndarray / torch.Tensor 가 아니더라도, shape/ndim 만 있으면 동작
    if not hasattr(arr, "shape") or not hasattr(arr, "ndim"):
        return
    if arr.ndim != 3:
        # (agent_num, T, D) 형태가 아니면 스킵
        return

    _, time_len, _ = arr.shape

    # 이미 기대 길이면 그대로 사용
    if time_len == expected_future_len:
        return

    # 정확히 future_len+1 인 경우만 "옛날 버그"로 보고 앞 프레임 제거
    if time_len == expected_future_len + 1:
        # 0번째 time 스텝(현재 프레임)을 버리고, 미래 future_len 프레임만 사용
        data[key] = arr[:, 1:, :]
        return

    # 그 외 길이는 예상치 못한 경우이므로, 여기서는 건드리지 않는다.
    # (문제가 있으면 이후 collate/train 단계에서 그대로 shape mismatch 에러가 나게 둠)
    return

class DiffusionPlannerData(Dataset):

    def __init__(self, data_dir, data_list, future_len):
        """
        data_dir: "/mnt/nuplan/dataset/processed"
        data_list: "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
        """
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """한 샘플을 이름 기반 dict로 반환한다.

        각 key별 기본 shape는 다음과 같다 (B는 배치에서 묶일 때 앞에 붙는다).

        - "ego_agent_past"           : (time_len, 11)
        - "ego_future_gt_3_dim"      : (future_len, 3)
        - "neighbor_agents_past"     : (chosen_agent_num, time_len, 11)
        - "lanes"                    : (chosen_lane_num, lane_len, 12)
        - "lanes_speed_limit"        : (chosen_lane_num, 1)
        - "lanes_has_speed_limit"    : (chosen_lane_num, 1)
        - "route_lanes"              : (chosen_route_lane_num, route_len, 12)
        - "route_lanes_speed_limit"  : (chosen_route_lane_num, 1)
        - "route_lanes_has_speed_limit": (chosen_route_lane_num, 1)
        - "static_objects"           : (chosen_static_num, 10)
        - "near_future_gt_3_dim"     : (chosen_agent_num, future_len, 3)
        - "planner_future_11_dim"    : (future_len, 11)
        - "agent_route_lane_order"   : (chosen_agent_num, chosen_lane_num)

        # 참고
            - chosen_agent_num <= caching_max_agent_num
            - chosen_lane_num <= lane_num
            - chosen_route_lane_num <= route_num
            - chosen_static_num <= max_static_num
        """
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        if data is None:
            # 이 샘플은 건너뛰고, DataLoader가 다시 뽑도록 예외를 던지거나
            raise IndexError(f"Corrupted sample at index {idx}")
        # 🔧 [임시 버그 패치] 구버전 캐시의 neighbor_future_gt_3_dim off-by-one 보정
        _fix_legacy_neighbor_future_len_bug(
            data=data,
            expected_future_len=self._future_len,
            key="neighbor_future_gt_3_dim",
        )

        # 최종으로 내보낼 key 목록
        output_keys: List[str] = [
            "ego_agent_past",
            "ego_future_gt_3_dim",
            "neighbor_agents_past",
            "lanes",
            "lanes_speed_limit",
            "lanes_has_speed_limit",
            "route_lanes",
            "route_lanes_speed_limit",
            "route_lanes_has_speed_limit",
            "static_objects",
            "near_future_gt_3_dim",
            "planner_future_11_dim",
            "agent_route_lane_order",
        ]

        # 이름이 다른 경우에만 source_key를 명시
        rename_source_map: Dict[str, str] = {
            "near_future_gt_3_dim": "neighbor_future_gt_3_dim",
            "planner_future_11_dim": "ego_future_gt_11_dim",
            # "agent_route_lane_order"는 이름은 같고 dtype만 바꾸면 되므로 여기엔 안 넣음
        }

        sample: Dict[str, Any] = {}

        for out_key in output_keys:
            src_key = rename_source_map.get(out_key, out_key)
            value = data[src_key]

            # dtype 보정이 필요한 key만 별도 처리
            if out_key == "agent_route_lane_order":
                value = value.astype("int64")

            sample[out_key] = value

        return sample