from __future__ import annotations

from typing import Dict, Type

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


class WorldModelFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self, config: Config) -> None:
        """
        Initializes the RasterFeatureBuilder class.
        """
        self._config = config
        self.data_processor = DataProcessor(config)

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
                do_unsqueeze=False)
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
            near_route_lanes=None,
            near_route_lanes_speed_limit=None,
            near_route_lanes_has_speed_limit=None,
            ego_agent_next_11_dim=current_input.ego_agent_next_11_dim,
            ego_future_gt_11_dim=current_input.ego_agent_future_11_dim)
        return world_model_feature
