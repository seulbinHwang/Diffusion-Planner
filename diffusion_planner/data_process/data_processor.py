from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list, agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs


class DataProcessor(object):

    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 200  #100  # [m] query radius scope relative to the current pose.

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
        self.predicted_neighbor_num = config.predicted_neighbor_num

    # Use for inference
    def observation_adapter(self,
                            history_buffer,
                            traffic_light_data,
                            map_api,
                            route_roadblock_ids,
                            npc_route_roadblock_ids,
                            device='cpu'):
        '''
        ego
        '''
        ego_agent_past = None  # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_x = ego_state.rear_axle.x
        ego_y = ego_state.rear_axle.y
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.rear_axle.heading
        ],
                                    dtype=np.float64)
        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(
            observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(
            observation_buffer[-1])
        (_, neighbor_agents_past, _,
         static_objects, agents_track_token) = agent_past_process(
             ego_agent_past, neighbor_agents_past, neighbor_agents_types,
             self.num_agents, static_objects, static_objects_types,
             self.num_static, self.max_ped_bike, anchor_ego_state)
        neighbor_agents_current = neighbor_agents_past[:, -1,] # (32, 11)
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids)
        near_token_to_route_roadblock_ids: Dict[str, Optional[List[str]]] = {}
        near_agents_current = []
        near_agent_count = 0
        for idx, token in enumerate(agents_track_token):  # List[Optional[str]]
            if near_agent_count >= self.predicted_neighbor_num:
                break
            # 1) 토큰에 해당하는 객체 찾기
            last_obs = observation_buffer[-1]  # DetectionsTracks
            obj = next(
                (o for o in last_obs.tracked_objects if o.track_token == token),
                None)
            if obj is None:
                raise ValueError("Object not found in the last observation.")
            if abs(obj.center.x -
                   ego_x) > self._radius / 2 or abs(obj.center.y -
                                                    ego_y) > self._radius / 2:
                continue  # 사각형 밖이면 결과에 포함하지 않음
            # 2) npc_state 생성 ― ego_state 역할을 대신할 간단한 객체
            #    rear_axle 에 Point2D(x, y)를 담고, heading 속성도 붙여줍니다.
            point = StateSE2(obj.center.x, obj.center.y, obj.center.heading)
            npc_state = SimpleNamespace(rear_axle=point)

            # 3) 기존 npc_route_roadblock_ids[token] 에 보정 함수 적용
            # npc_route_roadblock_ids: Dict[str, List[str]]
            a_npc_route_roadblock_ids = npc_route_roadblock_ids.get(token)
            if a_npc_route_roadblock_ids is None:
                near_token_to_route_roadblock_ids[token] = None
            else:
                near_token_to_route_roadblock_ids[
                    token] = route_roadblock_correction(
                        npc_state, map_api, a_npc_route_roadblock_ids)
                # neighbor_agents_current: (32, 11)
                near_agents_current.append(
                    neighbor_agents_current[idx]
                )
            near_agent_count += 1
        # near_agents_current : (near_agent_count, 11)
        near_agents_current = np.array(near_agents_current)  # (near_agent_count, 11)

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius,
            traffic_light_data)
        vector_map = map_process(route_roadblock_ids,
                                 near_token_to_route_roadblock_ids,
                                 near_agents_current,
                                 anchor_ego_state, coords, traffic_light_data,
                                 speed_limit, lane_route, self._map_features,
                                 self._max_elements, self._max_points)

        data = {
            "neighbor_agents_past":
                neighbor_agents_past[:, -21:],
            "ego_current_state":
                np.array(
                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32
                ),  # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
            "static_objects":
                static_objects
        }
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

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
                                        dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
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
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)

            static_objects, static_objects_types = sampled_static_objects_to_array_list(
                present_tracked_objects)

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects, _ = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids)

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius,
                traffic_light_data)

            vector_map = map_process(route_roadblock_ids, anchor_ego_state,
                                     coords, traffic_light_data, speed_limit,
                                     lane_route, self._map_features,
                                     self._max_elements, self._max_points)
            '''
            ego & agents future
            '''
            ego_agent_future = get_ego_future_array_from_scenario(
                scenario, ego_state, self.num_future_poses,
                self.future_time_horizon)

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
            future_tracked_objects_array_list, _ = sampled_tracked_objects_to_array_list(
                sampled_future_observations)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state, future_tracked_objects_array_list,
                self.num_agents, neighbor_indices)
            '''
            ego current
            '''
            ego_current_state = calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)

            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
                "static_objects": static_objects
            }
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)
