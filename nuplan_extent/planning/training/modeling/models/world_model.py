from typing import List

import timm
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


# nuplan/planning/training/preprocessing/feature_builders/raster_feature_builder.py
# nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder.GenericAgentsFeatureBuilder
class WorldModel(TorchModuleWrapper):

    def __init__(
        self,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
    ):
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        :return: The results of the inference as a TargetsType.
        """
        pass
