import os
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata
from typing import Dict, Any  # ← 추가


class DiffusionPlannerData(Dataset):

    def __init__(self, data_dir, data_list, max_agent_num,
                 predicted_neighbor_num, future_len):
        """
        data_dir: "/mnt/nuplan/dataset/processed"
        data_list: "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
        """
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._max_agent_num = max_agent_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """한 샘플을 이름 기반 dict로 반환한다.

        각 key별 기본 shape는 다음과 같다(B는 배치에서 묶일 때 앞에 붙는다).

        - "ego_agent_past"           : (time_len, 11)
        - "ego_future_gt_3_dim"      : (1 + future_len, 3)
        - "neighbor_agents_past"     : (max_agent_num, time_len, 11)
        - "lanes"                    : (lane_num, lane_len, 12)
        - "lanes_speed_limit"        : (lane_num, 1)
        - "lanes_has_speed_limit"    : (lane_num, 1)
        - "route_lanes"              : (route_lane_num, route_len, 12)
        - "route_lanes_speed_limit"  : (route_lane_num, 1)
        - "route_lanes_has_speed_limit": (route_lane_num, 1)
        - "static_objects"           : (max_static_num, 10)
        - "near_future_gt_3_dim"     : (predicted_neighbor_num, future_len, 3)
        - "planner_future_11_dim"    : (future_len, 11)
        - "agent_route_lane_order"   : (predicted_neighbor_num, lane_num)
        """
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))

        neighbor_agents_past = data["neighbor_agents_past"][:self.
                                                            _max_agent_num]
        near_future_gt_3_dim = data[
            "neighbor_future_gt_3_dim"][:self._predicted_neighbor_num]
        agent_route_lane_order = data["agent_route_lane_order"].astype(
            "int64")[:self._predicted_neighbor_num]

        sample: Dict[str, Any] = {
            "ego_agent_past": data["ego_agent_past"],
            "ego_future_gt_3_dim": data["ego_future_gt_3_dim"],
            "neighbor_agents_past": neighbor_agents_past,
            "lanes": data["lanes"],
            "lanes_speed_limit": data["lanes_speed_limit"],
            "lanes_has_speed_limit": data["lanes_has_speed_limit"],
            "route_lanes": data["route_lanes"],
            "route_lanes_speed_limit": data["route_lanes_speed_limit"],
            "route_lanes_has_speed_limit": data["route_lanes_has_speed_limit"],
            "static_objects": data["static_objects"],
            "near_future_gt_3_dim": near_future_gt_3_dim,
            # 유일하게 key 이름이 다른 ego future
            "planner_future_11_dim": data["ego_future_gt_11_dim"],
            "agent_route_lane_order": agent_route_lane_order,
        }

        return sample
