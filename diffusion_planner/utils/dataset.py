import os
from torch.utils.data import Dataset

from diffusion_planner.utils.train_utils import openjson, opendata


class DiffusionPlannerData(Dataset):

    def __init__(self, data_dir, data_list, past_neighbor_num,
                 predicted_neighbor_num, future_len):
        """
        data_dir: "/mnt/nuplan/dataset/processed"
        data_list: "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
        """
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        # TODO: revive
        # ego_future_gt_11_dim = data["ego_future_gt_11_dim"]

        neighbor_agents_past = data["neighbor_agents_past"][:self.
                                                            _past_neighbor_num]
        # (num_agents, future_len, 3) -> (predicted_neighbor_num, future_len, 3)
        # TODO: revive
        # near_future_gt_3_dim = data[
        #     "neighbor_future_gt_3_dim"][:self._predicted_neighbor_num]
        # TODO: remove
        neighbor_agents_future = data[
            "neighbor_agents_future"][:self._predicted_neighbor_num]
        ############
        lanes = data["lanes"]
        lanes_speed_limit = data["lanes_speed_limit"]
        lanes_has_speed_limit = data["lanes_has_speed_limit"]

        route_lanes = data["route_lanes"]
        route_lanes_speed_limit = data["route_lanes_speed_limit"]
        route_lanes_has_speed_limit = data["route_lanes_has_speed_limit"]
        agent_route_lane_order = data["agent_route_lane_order"].astype(
            "int64"
        )[:self._predicted_neighbor_num]  # (predicted_neighbor_num, lane_num)

        static_objects = data["static_objects"]

        data = {
            "ego_agent_past": data["ego_agent_past"],  # 0
            "ego_current_state": data["ego_current_state"],  # 1
            ###
            "ego_future_gt_3_dim": data["ego_agent_future"],  # 2 ###
            "neighbor_agents_past": neighbor_agents_past,  # 3
            "lanes": lanes,  # 4
            "lanes_speed_limit": lanes_speed_limit,  # 5
            "lanes_has_speed_limit": lanes_has_speed_limit,  # 6
            "route_lanes": route_lanes,  # 7
            "route_lanes_speed_limit": route_lanes_speed_limit,  # 8
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,  # 9
            "static_objects": static_objects,  # 10
            ###
            "near_future_gt_3_dim": neighbor_agents_future,  # 11 ###
            ### 유일하게 key 이름이 다름.
            "planner_future_11_dim":  data["ego_agent_future_11_dim"],  # 12 ###
            "agent_route_lane_order": agent_route_lane_order,  # 13
        }

        return tuple(data.values())
    # TODO: 수정 필요
    #
    #
    # def __getitem__(self, idx):
    #     data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
    #
    #     ego_future_gt_11_dim = data["ego_future_gt_11_dim"]
    #
    #     neighbor_agents_past = data["neighbor_agents_past"][:self.
    #                                                         _past_neighbor_num]
    #     # (num_agents, future_len, 3) -> (predicted_neighbor_num, future_len, 3)
    #     near_future_gt_3_dim = data[
    #         "neighbor_future_gt_3_dim"][:self._predicted_neighbor_num]
    #
    #     lanes = data["lanes"]
    #     lanes_speed_limit = data["lanes_speed_limit"]
    #     lanes_has_speed_limit = data["lanes_has_speed_limit"]
    #
    #     route_lanes = data["route_lanes"]
    #     route_lanes_speed_limit = data["route_lanes_speed_limit"]
    #     route_lanes_has_speed_limit = data["route_lanes_has_speed_limit"]
    #     agent_route_lane_order = data["agent_route_lane_order"].astype(
    #         "int64"
    #     )[:self._predicted_neighbor_num]  # (predicted_neighbor_num, lane_num)
    #
    #     static_objects = data["static_objects"]
    #
    #     data = {
    #         "ego_agent_past": data["ego_agent_past"],  # 0
    #         "ego_current_state": data["ego_current_state"],  # 1
    #         ###
    #         "ego_future_gt_3_dim": data["ego_future_gt_3_dim"],  # 2
    #         "neighbor_agents_past": neighbor_agents_past,  # 3
    #         "lanes": lanes,  # 4
    #         "lanes_speed_limit": lanes_speed_limit,  # 5
    #         "lanes_has_speed_limit": lanes_has_speed_limit,  # 6
    #         "route_lanes": route_lanes,  # 7
    #         "route_lanes_speed_limit": route_lanes_speed_limit,  # 8
    #         "route_lanes_has_speed_limit": route_lanes_has_speed_limit,  # 9
    #         "static_objects": static_objects,  # 10
    #         ###
    #         "near_future_gt_3_dim": near_future_gt_3_dim,  # 11
    #         ### 유일하게 key 이름이 다름.
    #         "planner_future_11_dim": ego_future_gt_11_dim,  # 12
    #         "agent_route_lane_order": agent_route_lane_order,  # 13
    #     }
    #
    #     return tuple(data.values())
