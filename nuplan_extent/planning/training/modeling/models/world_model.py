from typing import List, Dict

import timm
import torch

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner


# nuplan/planning/training/preprocessing/feature_builders/raster_feature_builder.py
# nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder.GenericAgentsFeatureBuilder
class WorldModel(TorchModuleWrapper):

    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
    ):
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self._planner = Diffusion_Planner(config)
        self._ckpt_path = ckpt_path
        self._ema_enabled = enable_ema

        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path,
                                          map_location=self._device)

            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {
                k[len("module."):]: v
                for k, v in state_dict.items()
                if k.startswith("module.")
            }
            self._planner.load_state_dict(model_state_dict)
        else:
            raise RuntimeError("No checkpoint path provided")

    def forward(self, features: WorldModelFeature) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        :return: The results of the inference as a TargetsType.
        """
        inputs = features.to_tensor_dict()
        _, outputs = self._planner(inputs)
        """
        outputs: Dict[str, torch.Tensor]
            "prediction" : (B, Pnn, T, 4)
        """
