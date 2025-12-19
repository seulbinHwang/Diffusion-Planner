import os
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata
from typing import Dict, Any, List  # ← 추가


def _fix_legacy_neighbor_future_len_bug(
    arr,
    expected_future_len: int,
):
    """구버전 캐시의 (N, future_len+1, 3) → (N, future_len, 3) 보정용 임시 함수."""
    if not hasattr(arr, "shape") or not hasattr(arr, "ndim"):
        return arr
    if arr.ndim != 3:
        return arr

    _, time_len, _ = arr.shape

    # 이미 정상 길이면 그대로
    if time_len == expected_future_len:
        return arr

    # 옛날 버그: 현재+미래 81프레임(= future_len+1)으로 저장된 경우
    if time_len == expected_future_len + 1:
        return arr[:, 1:, :]  # 0번째(현재) frame 버리고 미래 future_len개만 사용

    # 그 외 이상한 길이는 건드리지 않음 (문제 있으면 그대로 에러 나게 둠)
    return arr


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

        rename_source_map: Dict[str, str] = {
            "near_future_gt_3_dim": "neighbor_future_gt_3_dim",
            "planner_future_11_dim": "ego_future_gt_11_dim",
        }

        sample: Dict[str, Any] = {}

        for out_key in output_keys:
            src_key = rename_source_map.get(out_key, out_key)
            value = data[src_key]

            # 🔧 [임시 버그 패치] neighbor_future_gt_3_dim 길이 보정
            # if out_key == "near_future_gt_3_dim":
            #     value = _fix_legacy_neighbor_future_len_bug(
            #         value, self._future_len)

            if out_key == "agent_route_lane_order":
                value = value.astype("int64")

            sample[out_key] = value

        return sample
