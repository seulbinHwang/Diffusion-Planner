from __future__ import annotations

from typing import Dict, Type, Optional, List
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
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer


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

    def _post_process_unnormalized_features(
            self,
            neighbor_track_token: List[Optional[str]],  # len: agents_num
            target_agents_mask: np.ndarray,  # len: agents_num
    ) -> None:
        neighbor_future_gt_3_dim = self.unnormalized_features.get(
            "neighbor_future_gt_3_dim", None)  # (agent_num, future_len, 3)

        token_to_future_gt_3_dim: Dict[str, np.ndarray] = {}
        if neighbor_future_gt_3_dim is not None:
            for idx, token in enumerate(neighbor_track_token):
                if token is not None:
                    token_to_future_gt_3_dim[token] = neighbor_future_gt_3_dim[
                        idx]  # (future_len, 3)

        neighbor_future_all_gt_3_dim = self.unnormalized_features.get(
            "neighbor_future_all_gt_3_dim",
            None)  # (agent_num, future_all_len, 3)
        token_to_future_all_gt_3_dim: Dict[str, np.ndarray] = {}
        # near_future_all_gt_3_dim : (Pnn, future_all_len, 3)
        if neighbor_future_all_gt_3_dim is not None:
            near_future_all_gt_3_dim = neighbor_future_all_gt_3_dim[
                target_agents_mask]
            for idx, token in enumerate(neighbor_track_token):
                if token is not None:
                    token_to_future_all_gt_3_dim[
                        token] = neighbor_future_all_gt_3_dim[
                            idx]  # (future_all_len, 3)
        else:
            near_future_all_gt_3_dim = None
        self.unnormalized_features[
            "token_to_future_gt_3_dim"] = token_to_future_gt_3_dim  # Dict[str, np.ndarray] # len : valid_agent_num
        self.unnormalized_features[
            "near_future_all_gt_3_dim"] = near_future_all_gt_3_dim  # (Pnn, future_all_len, 3) or None
        self.unnormalized_features[
            "token_to_future_all_gt_3_dim"] = token_to_future_all_gt_3_dim  # Dict[str, np.ndarray] # len : valid_agent_num

    def get_features_from_simulation(
            self, current_input: PlannerInput,
            initialization: HorizonPlannerInitialization) -> WorldModelFeature:
        history_buffer: SimulationHistoryBuffer = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        model_inputs: Dict[
            str, torch.Tensor] = self.data_processor.observation_adapter(
                current_input.iteration.index,
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
            "planner_future_11_dim"] = current_input.planner_future_11_dim
        # # List[Optional[str]], (agent_num,)
        neighbor_track_token = model_inputs["neighbor_track_token"]
        model_inputs.pop("neighbor_track_token")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_inputs: Dict[str,
                           torch.Tensor] = convert_to_model_inputs(model_inputs,
                                                                   device,
                                                                   squeeze=True)
        self.unnormalized_features = model_inputs.copy()
        for key in self.unnormalized_features:
            # torch -> numpy
            self.unnormalized_features[key] = self.unnormalized_features[
                key].cpu().numpy()
        self.unnormalized_features[
            "neighbor_track_token"] = neighbor_track_token
        model_inputs = self.observation_normalizer(model_inputs)
        """
        input
            - neighbor_track_token :  List[Optional[str]], (agent_num,)
            - diffusion_agents_tokens: List[str], (valid_agent_num) maxlen=Pnn
        output
            - target_agents_mask: np.ndarray, (agent_num,) bool
        """
        target_agents_mask = self._get_target_agents_mask(
            neighbor_track_token, current_input.diffusion_agents_tokens)
        self._post_process_unnormalized_features(neighbor_track_token,
                                                 target_agents_mask)

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
            agent_route_lane_order=model_inputs[
                "agent_route_lane_order"],  # (agent_num, 1)
            target_agents_mask=target_agents_mask,  # (agent_num,) bool
            ego_agent_next_11_dim=model_inputs[
                "ego_agent_next_11_dim"],  # (interpol_num, 11)
            planner_future_11_dim=model_inputs["planner_future_11_dim"]
        )  # (future_len, 11)
        return world_model_feature

    def _get_target_agents_mask(
            self, neighbor_track_token: List[Optional[str]],
            diffusion_agents_tokens: List[str]) -> np.ndarray:
        """
        input
            - neighbor_track_token :  List[Optional[str]], (agent_num,)
            - diffusion_agents_tokens: List[str], (valid_agent_num) maxlen=Pnn
        output
            - target_agents_mask: np.ndarray, (agent_num,) bool
        """
        assert isinstance(neighbor_track_token, list)
        agent_num = len(neighbor_track_token)
        target_agents_mask = np.zeros((agent_num,), dtype=bool)
        for idx in range(agent_num):
            if neighbor_track_token[idx] in diffusion_agents_tokens:
                target_agents_mask[idx] = True
        return target_agents_mask

    def get_features_from_scenario(
            self, scenario: "AbstractScenario") -> WorldModelFeature:  # 추가
        """시뮬레이션 전용 빌더입니다. 학습/오프라인 전처리 경로에서 호출되면 에러를 냅니다."""  # 추가
        raise NotImplementedError(  # 추가
            "[WorldModelFeatureBuilder] get_features_from_scenario is not implemented. "
            "This builder is intended for simulation-only. "
            "If you need scenario-based preprocessing, implement this method to "
            "extract features directly from the scenario.")  # 추가
