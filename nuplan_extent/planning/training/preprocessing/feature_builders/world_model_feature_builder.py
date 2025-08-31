from __future__ import annotations

from typing import Dict, Type

import numpy as np
import numpy.typing as npt

from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from nuplan.planning.training.preprocessing.features.raster_utils import (
    get_agents_raster,
    get_baseline_paths_raster,
    get_ego_raster,
    get_roadmap_raster,
)


class WorldModelFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self,) -> None:
        """
        Initializes the RasterFeatureBuilder class.
        """
        # TODO
        super().__init__()

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
            initialization: PlannerInitialization) -> WorldModelFeature:
        # TODO
        """Inherited, see superclass."""
        history = current_input.history
        ego_state = history.ego_states[-1]
        observation = history.observations[-1]

        if isinstance(observation, DetectionsTracks):
            return self._compute_feature(ego_state, observation,
                                         initialization.map_api)
        else:
            raise TypeError(
                f"Observation was type {observation.detection_type()}. Expected DetectionsTracks"
            )
