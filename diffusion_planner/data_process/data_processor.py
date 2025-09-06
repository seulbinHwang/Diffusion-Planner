import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # GUI 백엔드 사용 안함 (메모리 절약)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import wandb
import os
import torch
from typing import Dict, Tuple, Union, List, Optional
from nuplan.common.actor_state.state_representation import Point2D
import draw_machine
# matplotlib 설정 추가
plt.rcParams['figure.max_open_warning'] = 0  # 경고 메시지 비활성화
matplotlib.rcParams['figure.max_open_warning'] = 0

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_ego_objects_to_array_list, sampled_static_objects_to_array_list,
    agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs, get_npc_route_roadblock_ids


class DataProcessor(object):

    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)
        self.config = config
        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100  # [m] query radius scope relative to the current pose.

        self._map_features = [
            'LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'
        ]  # name of map features to be extracted.
        self._max_elements = {
            'LANE': config.lane_num,
            'LEFT_BOUNDARY': config.lane_num,
            'RIGHT_BOUNDARY': config.lane_num,
            'ROUTE_LANES': config.route_num
        }  # maximum number of elements to extract per feature layer.
        self._max_points = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.route_len
        }  # maximum number of points per feature to extract per feature layer.
        # wandb 사용 여부를 한 곳에서만 판단
        if getattr(config, "use_wandb", True) and wandb.run is None:
            # 컨트롤러 run id를 같은 group 으로 묶어 두면 대시보드가 깔끔
            wandb.init(
                project=getattr(config, "wandb_project", "Diffusion-Planner"),
                entity=getattr(config, "wandb_entity", None),
                group=getattr(config, "wandb_group", None),  # ← 컨트롤러 ID
                job_type="preprocess-worker",
                name=f"{config.name}-worker-{os.getpid()}",
                mode=getattr(config, "wandb_mode", "online"),
                reinit=True,  # fork 안전
                settings=wandb.Settings(start_method="fork"),
            )
        self._wandb_enabled = wandb.run is not None

    def _filter_agents_within_radius(
        self,
        neighbor_agents_past: np.ndarray,
        neighbor_agents_future: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """ego 중심 정사각형 영역(가로·세로 2*radius)으로 에이전트 클리핑.

        원래는 원형 반경(r) 내부 여부를 L2 거리로 판정했지만,
        이제는 정사각형의 내부 여부를 다음 조건으로 판정합니다:
            |x| <= r  AND  |y| <= r
        (여기서 (x, y)는 상대좌표계의 마지막 시점 위치)

        Args:
            neighbor_agents_past (np.ndarray): (N, Tp, 11)
                상대 좌표계 과거 에이전트 시퀀스.
            neighbor_agents_future (Optional[np.ndarray], optional): (N, Tf, 3)
                상대 좌표계 미래 에이전트 시퀀스. 기본값 None.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]:
                - filtered_neighbor_agents_past:   (N, Tp, 11)
                - filtered_neighbor_agents_future: (N, Tf, 3) 또는 None
              정사각형 바깥 에이전트는 전체 시퀀스를 0으로 채운 상태로 유지됩니다(개수 고정).
        """
        # 마지막 시점 상대 좌표 (N, 2)  ← ego 기준이므로 ego는 정중앙(0,0)
        cur_xy = neighbor_agents_past[:, -1, :2]  # (N, 2)

        # 정사각형 내부 판정: |x| <= r AND |y| <= r  → (N,)
        mask_x = np.abs(cur_xy[:, 0]) <= self._radius  # (N,)
        mask_y = np.abs(cur_xy[:, 1]) <= self._radius  # (N,)
        mask = mask_x & mask_y  # (N,), True=정사각형 내부(유효)

        # 브로드캐스팅을 위한 차원 확장: (N, 1, 1)
        mask_expanded = mask[:, None, None]

        # 정사각형 바깥 에이전트는 전체 시퀀스를 0으로 만듦(개수는 고정)
        # filtered_neighbor_agents_past: (N, Tp, 11)
        filtered_neighbor_agents_past = neighbor_agents_past * mask_expanded

        filtered_neighbor_agents_future = None
        if neighbor_agents_future is not None:
            # filtered_neighbor_agents_future: (N, Tf, 3)
            filtered_neighbor_agents_future = neighbor_agents_future * mask_expanded

        return filtered_neighbor_agents_past, filtered_neighbor_agents_future

    # Use for inference
    def observation_adapter(self,
                            history_buffer,
                            traffic_light_data,
                            map_api,
                            route_roadblock_ids,
                            device='cpu',
                            squeeze=False) -> Dict[str, torch.Tensor]:
        '''
        ego
        '''
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.rear_axle.heading
        ],
                                    dtype=np.float64)
        '''
        neighbor
        '''
        ego_state_buffer = history_buffer.ego_state_buffer
        # all_frame_ego_feature: np.ndarray: (num_frames, 10)
        # 가장 과거 -> 가장 최근 순서
        all_frame_ego_feature = sampled_ego_objects_to_array_list(
            ego_state_buffer)

        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        # all_frame_agents_feature: List[np.ndarray], (frame_agents_num, 8) # frame_agents_num 길이가 가변적
        # all_frame_agents_types:  List[List[TrackedObjectType]]
        (all_frame_agents_feature, all_frame_agents_types
        ) = sampled_tracked_objects_to_array_list(observation_buffer)

        # present_static_feature: np.ndarray, (len(static_obj), 5)
        # static_objects_types: List[TrackedObjectType]
        (present_static_feature,
         static_objects_types) = sampled_static_objects_to_array_list(
             observation_buffer[-1])
        # neighbor_agents_past: (agent_num, num_frames, 11)
        # static_objects: (num_static, 10)
        """
    # ego_agent_past: (num_frames, 11)
    # neighbor_agents_past: (agent_num, num_frames, 11)
    # sorted_cur_neighbor_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
    # static_objects: (num_static, 10)
        """
        (ego_agent_past, neighbor_agents_past, _,
         static_objects, final_veh_num) = agent_past_process(
             all_frame_ego_feature, all_frame_agents_feature,
             all_frame_agents_types, self.num_agents, present_static_feature,
             static_objects_types, self.num_static, self.max_ped_bike,
             anchor_ego_state)
        neighbor_agents_past, _ = self._filter_agents_within_radius(
            neighbor_agents_past)
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids: List[str] = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids)
        (coords, traffic_light_data, speed_limit,
         lane_route) = get_neighbor_vector_set_map(map_api, self._map_features,
                                                   ego_coords, self._radius,
                                                   traffic_light_data)
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords,
                                 traffic_light_data, speed_limit, lane_route,
                                 self._map_features, self._max_elements,
                                 self._max_points)

        data = {
            "ego_agent_past": ego_agent_past[-21:],  # (time_len, 11)
            "neighbor_agents_past":
                neighbor_agents_past[:, -21:],  # (agent_num, time_len, 11)
            "static_objects": static_objects
        }
        # data: Dict[str, np.ndarray]
        data.update(vector_map)
        # data: Dict[str, torch.Tensor]
        data = convert_to_model_inputs(data, device, squeeze)

        return data

    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api
            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([
                ego_state.rear_axle.x, ego_state.rear_axle.y,
                ego_state.rear_axle.heading
            ],
                                        dtype=np.float64)  # shape (3,)
            # all_frame_ego_feature: np (21, 10) # x, y, theta, vx, vy, width, length
            all_frame_ego_feature, time_stamps_past = get_ego_past_array_from_scenario(
                scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses)
            ]
            sampled_past_observations = past_tracked_objects + [
                present_tracked_objects
            ]
            # all_frame_agents_feature: List[np.ndarray], (frame_agents_num, 8) # frame_agents_num 길이가 가변적
            # all_frame_agents_types:  List[List[TrackedObjectType]]
            all_frame_agents_feature, all_frame_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            # present_static_feature: np.ndarray, (len(static_obj), 5)
            # static_objects_types: List[TrackedObjectType]
            (present_static_feature, static_objects_types
            ) = sampled_static_objects_to_array_list(present_tracked_objects)

            # : ego_agent_past: (num_frames, 11)
            (ego_agent_past, neighbor_agents_past, neighbor_indices,
             static_objects, final_veh_num) = agent_past_process(
                 all_frame_ego_feature, all_frame_agents_feature,
                 all_frame_agents_types, self.num_agents,
                 present_static_feature, static_objects_types, self.num_static,
                 self.max_ped_bike, anchor_ego_state)
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids)
            token_to_route_roadblock_ids: Dict[
                str, Optional[List[str]]] = get_npc_route_roadblock_ids(
                    scenario, self._radius, final_veh_num)

            (coords, traffic_light_data, speed_limit,
             lane_route) = get_neighbor_vector_set_map(map_api,
                                                       self._map_features,
                                                       ego_coords, self._radius,
                                                       traffic_light_data)

            vector_map = map_process(route_roadblock_ids, anchor_ego_state,
                                     coords, traffic_light_data, speed_limit,
                                     lane_route, self._map_features,
                                     self._max_elements, self._max_points)
            '''
            ego & agents future
            ego_agent_future : rear axle x,y, ~~~
            ego_agent_future_11_dim : center x,y, ~~~
            '''
            (ego_agent_future,
             ego_agent_future_11_dim) = get_ego_future_array_from_scenario(
                 scenario, ego_state, self.num_future_poses,
                 self.future_time_horizon)

            Tf, Df = ego_agent_future_11_dim.shape
            assert Tf == self.num_future_poses, (
                "Ego agent future states should have T time steps")
            assert Df == 11, (
                "Ego agent future states should have 11 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, agent type)"
            )

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0,
                    time_horizon=self.future_time_horizon,
                    num_samples=self.num_future_poses)
            ]

            sampled_future_observations = [present_tracked_objects
                                          ] + future_tracked_objects
            (future_tracked_objects_array_list,
             _) = sampled_tracked_objects_to_array_list(
                 sampled_future_observations)
            # neighbor_agents_future: (num_agents, future_len, 3)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state, future_tracked_objects_array_list,
                self.num_agents, neighbor_indices)
            neighbor_agents_past, neighbor_agents_future = \
                self._filter_agents_within_radius(neighbor_agents_past,
                                                 neighbor_agents_future)
            '''
            ego current
            
            
            '''
            # ego_agent_past: (T, 11)
            _, ego_current_state = calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)
            T, D = ego_agent_past.shape
            assert T == self.num_past_poses + 1, "Ego agent past states should have T+1 time steps"
            assert D == 11, "Ego agent past states should have 8 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length)"

            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_agent_past": ego_agent_past,  # (time_len, 11)
                "ego_current_state": ego_current_state,  # (10,)
                # TODO: ego_agent_future 의 shape이 (0,) 인 경우가 있음. (왜 그런지는 모르겠음)
                "ego_agent_future":
                    ego_agent_future,  # rear_axle x,y # (future_len, 3)
                "ego_agent_future_11_dim":
                    ego_agent_future_11_dim,  # center x,y # (future_len, 11)
                "neighbor_agents_past":
                    neighbor_agents_past,  # (num_agents, time_len, 11)
                "neighbor_agents_future":
                    neighbor_agents_future,  # (num_agents, future_len, 3)
                "static_objects": static_objects  # (num_static, 5)
            }
            data.update(vector_map)

            # 디버깅용 그림 그리기
            if self._wandb_enabled or self.config.save_image:
                print("Visualizing scenario:", map_name, token)
                draw_machine.draw_world_model_to_png(
                    data,
                    token_to_future_traj_wrt_ego=None,
                    save_path=self._save_dir)
