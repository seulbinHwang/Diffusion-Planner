from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type
import time
import numpy as np
import torch
import sys
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType
import torch
import matplotlib.pyplot as plt
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
import numpy as np
from tqdm import tqdm

from nuplan.common.actor_state.state_representation import Point2D
from torch.utils.data.dataloader import default_collate

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list, agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List
import torch
from torch import Tensor

from nuplan.planning.training.preprocessing.features.abstract_model_feature import \
    AbstractModelFeature


@dataclasses.dataclass
class ModelInputFeature(AbstractModelFeature):
    """
    Extended feature class that also includes neighbor_agents_token_str (List[List[str]]).
    The outer list dimension (batch dimension) is typically = 1 for single-sample usage,
    but can be >1 if collate is used in a DataLoader scenario.
    """
    neighbor_agents_past: Tensor  # shape: (1, N, T, 11) etc.
    neighbor_agents_original: Tensor  # shape: (1, N, T, 2) etc.
    neighbor_agents_id: Tensor  # shape: (N,) or (1, N)
    ego_current_state: Tensor  # shape: (1, 4)
    static_objects: Tensor  # shape: (1, 5, 10)
    lanes: Tensor  # shape: (1, 70, 20, 12)
    lanes_speed_limit: Tensor  # shape: (1, 70, 1)
    lanes_has_speed_limit: Tensor  # shape: (1, 70, 1)
    route_lanes: Tensor  # shape: (1, 25, 20, 12)
    route_lanes_speed_limit: Tensor  # shape: (1, 25, 1)
    route_lanes_has_speed_limit: Tensor  # shape: (1, 25, 1)
    # 새로 추가: agent 토큰 문자열 정보 (List[List[str]]: batch dimension x agent dimension)
    neighbor_agents_token_str: List[List[str]] = field(default_factory=list)

    @classmethod
    def collate(cls, batch: List[ModelInputFeature]) -> ModelInputFeature:
        return batch[0]

    def to_feature_tensor(self) -> ModelInputFeature:
        """
        Convert any underlying data (if numpy) to torch.Tensor.
        (이미 torch.Tensor 형태라고 가정하므로, 여기서는 그대로 self를 반환)
        """
        return self

    def to_device(self, device: torch.device) -> ModelInputFeature:
        """
        장치(GPU/CPU) 이동을 일괄 처리.
        neighbor_agents_token_str는 Python list이므로 그대로 둡니다.
        """
        return ModelInputFeature(
            neighbor_agents_past=self.neighbor_agents_past.to(device),
            neighbor_agents_original=self.neighbor_agents_original.to(device),
            neighbor_agents_id=self.neighbor_agents_id.to(device),
            ego_current_state=self.ego_current_state.to(device),
            static_objects=self.static_objects.to(device),
            lanes=self.lanes.to(device),
            lanes_speed_limit=self.lanes_speed_limit.to(device),
            lanes_has_speed_limit=self.lanes_has_speed_limit.to(device),
            route_lanes=self.route_lanes.to(device),
            route_lanes_speed_limit=self.route_lanes_speed_limit.to(device),
            route_lanes_has_speed_limit=self.route_lanes_has_speed_limit.to(
                device),
            neighbor_agents_token_str=self.neighbor_agents_token_str,
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert internal fields back to a Dict[str, Any],
        replicating the original model_inputs style + token_str.
        """
        return {
            "neighbor_agents_past": self.neighbor_agents_past,
            "neighbor_agents_original": self.neighbor_agents_original,
            "neighbor_agents_id": self.neighbor_agents_id,
            "ego_current_state": self.ego_current_state,
            "static_objects": self.static_objects,
            "lanes": self.lanes,
            "lanes_speed_limit": self.lanes_speed_limit,
            "lanes_has_speed_limit": self.lanes_has_speed_limit,
            "route_lanes": self.route_lanes,
            "route_lanes_speed_limit": self.route_lanes_speed_limit,
            "route_lanes_has_speed_limit": self.route_lanes_has_speed_limit,
            "neighbor_agents_token_str": self.neighbor_agents_token_str,
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> ModelInputFeature:
        """
        Dict -> ModelInputFeature 복원
        """
        return cls(**data)

    def unpack(self) -> List[ModelInputFeature]:
        """
        배치 차원을 (batch_size, ...)에서 각 sample 단위로 나누어 List로 반환.

        neighbor_agents_token_str는 List[List[str]] 형태이므로,
        이 batch_size만큼 상응하게 쪼갤 수 있습니다.
        (batch_size=1인 경우는 그냥 동일하게 한 원소를 반환)
        """
        batch_size = self.neighbor_agents_past.shape[0]

        features_list = []
        for i in range(batch_size):
            # i번째 batch slice
            # neighbor_agents_token_str도 i번째 것을 선택
            if batch_size == len(self.neighbor_agents_token_str):
                per_sample_token_str = [self.neighbor_agents_token_str[i]]
            else:
                # 혹은 batch size와 토큰 스트링 리스트 개수가 다르면 로직 수정 필요
                per_sample_token_str = [[]]

            features_list.append(
                ModelInputFeature(
                    neighbor_agents_past=self.neighbor_agents_past[i:i + 1],
                    neighbor_agents_original=self.neighbor_agents_original[i:i +
                                                                           1],
                    neighbor_agents_id=self.neighbor_agents_id[i:i + 1],
                    ego_current_state=self.ego_current_state[i:i + 1],
                    static_objects=self.static_objects[i:i + 1],
                    lanes=self.lanes[i:i + 1],
                    lanes_speed_limit=self.lanes_speed_limit[i:i + 1],
                    lanes_has_speed_limit=self.lanes_has_speed_limit[i:i + 1],
                    route_lanes=self.route_lanes[i:i + 1],
                    route_lanes_speed_limit=self.route_lanes_speed_limit[i:i +
                                                                         1],
                    route_lanes_has_speed_limit=self.
                    route_lanes_has_speed_limit[i:i + 1],
                    neighbor_agents_token_str=per_sample_token_str,
                ))
        return features_list


class DataProcessor(AbstractFeatureBuilder):
    """Dummy vector map feature builder used in testing."""

    def __init__(self, config):
        self._save_dir = getattr(config, "save_path", None)
        self.observation_normalizer = config.observation_normalizer
        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num  # 32
        self.num_static = config.static_objects_num  # 5
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

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "processed_data"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return ModelInputFeature

    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: PlannerInitialization) -> AbstractModelFeature:
        # neighbor_agents_token_str: List[List[str]]
        model_inputs, neighbor_agents_token_str = self.observation_adapter(
            history_buffer=current_input.history,  # SimulationHistoryBuffer
            traffic_light_data=list(current_input.traffic_light_data),
            map_api=initialization.map_api,
            route_roadblock_ids=initialization.route_roadblock_ids)
        neighbor_agents_original = model_inputs["neighbor_agents_past"].detach(
        ).clone()  # [1, 32, 21, 11]
        neighbor_agents_original = neighbor_agents_original[:, :, :, :
                                                            6]  # [1, 32, 21, 2]
        model_inputs = self.observation_normalizer(
            model_inputs)  # Dict[str, torch.Tensor]
        my_feature = ModelInputFeature(
            neighbor_agents_past=model_inputs[
                "neighbor_agents_past"],  # [1, 32, 21, 11]
            neighbor_agents_original=neighbor_agents_original,  # [1, 32, 21, 2]
            neighbor_agents_id=model_inputs["neighbor_agents_id"],  # [1, 20]
            ego_current_state=model_inputs["ego_current_state"],  # [1, 4]
            static_objects=model_inputs["static_objects"],  # [1, 5, 10]
            lanes=model_inputs["lanes"],  # [1, 70, 20, 12]
            lanes_speed_limit=model_inputs["lanes_speed_limit"],  # [1, 70, 1]
            lanes_has_speed_limit=model_inputs[
                "lanes_has_speed_limit"],  # [1, 70, 1]
            route_lanes=model_inputs["route_lanes"],  # [1, 25, 20, 12]
            route_lanes_speed_limit=model_inputs[
                "route_lanes_speed_limit"],  # [1, 25, 1]
            route_lanes_has_speed_limit=model_inputs[
                "route_lanes_has_speed_limit"],  # [1, 25, 1]
            neighbor_agents_token_str=
            neighbor_agents_token_str,  # List[List[str]]
        )
        return my_feature

    def observation_adapter(
            self,
            history_buffer,  # SimulationHistoryBuffer
            traffic_light_data,  # List
            map_api,
            route_roadblock_ids,
            device='cpu'):
        '''
        ego
        '''
        ego_agent_past = None  # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]  # EgoState
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.rear_axle.heading
        ],
                                    dtype=np.float64)  # shape: (3,)
        '''
        neighbor
        '''
        # observation_buffer: Deque[Observation]
        # Past observations including the current
        observation_buffer = history_buffer.observation_buffer
        """
        neighbor_agents_past : List[ np.ndarray (N, 8) ]: N: number of agents 
            - len(neighbor_agents_past) = 총 시간 스텝 수
        neighbor_agents_types: List[List[TrackedObjectType]]
        track_token_ids: Dict[str, int]
        """
        (neighbor_agents_past, neighbor_agents_types, track_token_ids
        ) = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(
            observation_buffer[-1])

        (_, neighbor_agents_past, selected_indices, static_objects,
         neighbor_agents_id, neighbor_agents_token_str) = agent_past_process(
             ego_agent_past, neighbor_agents_past, neighbor_agents_types,
             track_token_ids, self.num_agents, static_objects,
             static_objects_types, self.num_static, self.max_ped_bike,
             anchor_ego_state)
        # anchor_ego_state:  [6.64917136e+05 3.99964917e+06 2.76081205e+00]
        """
        neighbor_agents_past: (num_agents, T_past, 11)
        selected_indices : ???
        static_objects: (B, N_static, 10) 
            - 마지막 4 차원은 one-hot # (CZONE_SIGN, BARRIER, TRAFFIC_CONE, ELSE)
        neighbor_agents_id:  (shrinked_num_agents,)
        neighbor_agents_token_str : List[str] # len = shrinked_num_agents
            - ego 와 가까운 순서대로 정렬된 agent token 문자열
        """
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        # route_roadblock_ids: List[str] # Roadblock ids comprising goal route
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids)
        (coords, traffic_light_data, speed_limit,
         lane_route) = get_neighbor_vector_set_map(
             map_api,
             self._map_features,
             ego_coords,
             self._radius,  # 100
             traffic_light_data)
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords,
                                 traffic_light_data, speed_limit, lane_route,
                                 self._map_features, self._max_elements,
                                 self._max_points)

        data = {
            "neighbor_agents_past": neighbor_agents_past[:, -21:],
            "neighbor_agents_id": neighbor_agents_id,
            "ego_current_state": np.array(
                [0., 0., 1., 0.],
                dtype=np.float32),  # ego centric x, y, cos, sin
            "static_objects": static_objects
        }
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)
        neighbor_agents_token_str = [neighbor_agents_token_str
                                    ]  # for batch size 1 # TODO: ???
        # neighbor_agents_token_str: List[List[str]]
        return data, neighbor_agents_token_str

    def get_features_from_scenario(
            self, scenario: AbstractScenario) -> AbstractModelFeature:
        """Inherited, see superclass."""
        return ModelInputFeature(
            data1=[np.zeros((10, 10, 10))],
            data2=[np.zeros((10, 10, 10))],
            data3=[{
                "test": np.zeros((10, 10, 10))
            }],
        )
