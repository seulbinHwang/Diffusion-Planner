from __future__ import annotations

from typing import Dict, Type, Optional
import numpy as np

import torch

from nuplan_extent.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.data_process.utils import convert_to_model_inputs

class WorldModelFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self, config: Config) -> None:
        """
        Initializes the RasterFeatureBuilder class.
        """
        self._config = config
        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer
        self.unnormalized_features: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "world_model_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return WorldModelFeature  # type: ignore

    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: HorizonPlannerInitialization) -> WorldModelFeature:
        history_buffer = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        model_inputs: Dict[
            str, torch.Tensor] = self.data_processor.observation_adapter(
                history_buffer,
                traffic_light_data,
                initialization.map_api,
                initialization.route_roadblock_ids,
                scenario=initialization.scenario,
                squeeze=True)
        # (interpol_num, 11)
        model_inputs[
            "ego_agent_next_11_dim"] = current_input.ego_agent_next_11_dim
        # (future_len, 11)
        model_inputs[
            "ego_future_gt_11_dim"] = current_input.ego_agent_future_11_dim
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_inputs = convert_to_model_inputs(model_inputs, device, squeeze=True)
        self.unnormalized_features = model_inputs.copy()
        for key in self.unnormalized_features:
            # torch -> numpy
            self.unnormalized_features[key] = self.unnormalized_features[
                key].cpu().numpy()

        model_inputs = self.observation_normalizer(model_inputs)
        """
    world_model_feature: Dict[str, numpy.ndarray] 가 아래와 같이 구성되어 있고, 이게 함수의 input으로 쓰일거야.
        ego_agent_past: (time_len, 11)
            - time_len: (2.0초 과거, 1.9초 과거 , ... 0.1초 과거, 현재) 21개
            - 11: x , y, cos(yaw), sin(yaw), "vx, vy", length, width, one-hot (car, pedestrian, bicycle)
                - current ego 차량 뒷축 좌표계  기준 값들임. 단위는 m, rad, m/s 
                - 직사각형(+ 사각형의 중점에서 나오는 heading 방향선도 그리기) 그리기 
                    - x, y, cos(yaw), sin(yaw) , length, width 로 
                    - "fill_color": "#FFFFFF", "line_color": "#808080", "line_width": 2
                    - 현재 위치만 속을 칠하고(fill_alpha=1.0) , 과거 위치는 속을 안채우기
                - 실선 화살표 (속도): (vx, vy) # 크기와 상관없이 길이는 2으로 + 방향만 잘 그리기 
                  - "line_color": "#808080", "line_width": 2 
        neighbor_agents_past: (agent_num, time_len, 11)
            - time_len: (2.0초 과거, 1.9초 과거 , ... 0.1초 과거, 현재) 21개
            - 11: x, y, cos(yaw), sin(yaw), "vx, vy", length, width, one-hot (car, pedestrian, bicycle)
                - current ego 차량 뒷축 좌표계  기준 값들임. 단위는 m, rad, m/s 
                - 직사각형(+ 사각형의 중점에서 나오는 heading 방향선도 그리기) 그리기 
                    - x, y, cos(yaw), sin(yaw) , length, width 로 
                        "vehicles": {"fill_color": "#84E573", "fill_alpha": 0.5, "line_color": "#84E573", "line_width": 1},
                        "pedestrians": {"fill_color": "#4D83E1", "fill_alpha": 0.5, "line_color": "#4D83E1", "line_width": 1},
                        "bicycles": {"fill_color": "#FF4D4D", "fill_alpha": 0.5, "line_color": "#FF4D4D", "line_width": 1},
                        - 현재 위치만 속을 칠하고(fill_alpha=0.5) , 과거 위치는 속을 안채우기
                - 실선 화살표 (속도): (vx, vy) # 크기와 상관없이 길이는 2으로 + 방향만 잘 그리기 
                        "vehicles": {"line_color": "#84E573", "line_width": 1},
                        "pedestrians": {"line_color": "#4D83E1", "line_width": 1},
                        "bicycles": {"line_color": "#FF4D4D", "line_width": 1},
        ego_agent_next_11_dim: (interpol_num, 11)
            - interpol_num: (0.1초 후, ... ) 0.1초 간격으로 interpol_num개
            - 11: x, y, cos(yaw), sin(yaw), "vx, vy", length, width, one-hot (car, pedestrian, bicycle)
            - current ego 차량 뒷축 좌표계  기준 값들임. 단위는 m, rad, m/s 
                - 직사각형(+ 사각형의 중점에서 나오는 heading 방향선도 그리기) 그리기 
                    - x, y, cos(yaw), sin(yaw) , length, width 로 
                    - "fill_color": "#808080", "line_color": "#FFFFFF", "line_width": 2
                    - 미래 위치는 속을 안채우기
                - 실선 화살표 (속도): (vx, vy) # 크기와 상관없이 길이는 2으로 + 방향만 잘 그리기 
                  - "line_color": "#00C8C8", "line_alpha": 0.8, "line_width": 2
        ego_future_gt_11_dim: (future_len, 11)
            - future_len: (0.1초 후, ... 8.0초 후) 0.1초 간격으로 80개
            - 11: x, y, cos(yaw), sin(yaw), "vx, vy", length, width, one-hot (car, pedestrian, bicycle)
            - current ego 차량 뒷축 좌표계  기준 값들임. 단위는 m, rad, m/s 
                - 직사각형(+ 사각형의 중점에서 나오는 heading 방향선도 그리기) 그리기 
                    - x, y, cos(yaw), sin(yaw) , length, width 로 
                    - "fill_color": "#808080", "line_color": "#FFFFFF", "line_width": 1
                    - 미래 위치는 속을 안채우기
                - 실선 화살표 (속도): (vx, vy) # 크기와 상관없이 길이는 2으로 + 방향만 잘 그리기 
                  - "line_color": "#00C8C8", "line_alpha": 0.8, "line_width": 2            
        lanes: (lane_num, lane_len, 12)
            - lane_num: 70개 (차선 개수)
            - lane_len: 차선을 등간격으로 나눈 갯수
            - 12: 
            - LANE (차선의 실선): "line_color": "#2d3ea7" 실선
            - BASELINE_PATHS(차선 실선과 차선 실선 사이의 중심 경로(centerline) ): "line_color": "#CBCBCB" 점선
        """
        world_model_feature = WorldModelFeature(
            ego_agent_past=model_inputs["ego_agent_past"],  # (time_len, 11)
            neighbor_agents_past=model_inputs[
                "neighbor_agents_past"],  # (agent_num, time_len, 11)
            static_objects=model_inputs[
                "static_objects"],  # (static_objects_num, 10)
            lanes=model_inputs["lanes"],  # (lane_num, lane_len, 12)
            lanes_speed_limit=model_inputs[
                "lanes_speed_limit"],  # (lane_num, 1)
            lanes_has_speed_limit=model_inputs[
                "lanes_has_speed_limit"],  # (lane_num, 1)
            route_lanes=model_inputs[
                "route_lanes"],  # (route_num, lane_len, 12)
            route_lanes_speed_limit=model_inputs[
                "route_lanes_speed_limit"],  # (route_num, 1)
            route_lanes_has_speed_limit=model_inputs[
                "route_lanes_has_speed_limit"],  # (route_num, 1)
            ego_agent_next_11_dim=model_inputs[
                "ego_agent_next_11_dim"],  # (interpol_num, 11)
            ego_future_gt_11_dim=model_inputs["ego_future_gt_11_dim"]
        )  # (future_len, 11)
        return world_model_feature

    def get_features_from_scenario(
            self, scenario: "AbstractScenario") -> WorldModelFeature:  # 추가
        """시뮬레이션 전용 빌더입니다. 학습/오프라인 전처리 경로에서 호출되면 에러를 냅니다."""  # 추가
        raise NotImplementedError(  # 추가
            "[WorldModelFeatureBuilder] get_features_from_scenario is not implemented. "
            "This builder is intended for simulation-only. "
            "If you need scenario-based preprocessing, implement this method to "
            "extract features directly from the scenario.")  # 추가
