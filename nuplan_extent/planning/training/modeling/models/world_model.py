from typing import List, Dict, Deque, Optional

import timm
import torch
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from diffusion_planner.utils.config import Config
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.utils.amp import amp_context_for_infer
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


# nuplan/planning/script/config/common/model/raster_model.yaml
class WorldModel(TorchModuleWrapper):

    def __init__(
        self,
        config: Config,  #
        ckpt_path: str,  #
        feature_builders: List[AbstractFeatureBuilder],  #
        target_builders: List[AbstractTargetBuilder],  # 안씀
        future_trajectory_sampling: TrajectorySampling,  # 씀.
        enable_ema: bool = True,
    ):
        super().__init__(
            future_trajectory_sampling=future_trajectory_sampling,
            feature_builders=feature_builders,
            target_builders=target_builders,
        )
        self.config = config
        self._planner = Diffusion_Planner(config)
        self._ckpt_path = ckpt_path
        self._ema_enabled = enable_ema
        self._step_interval = 0.1  # [s]
        self._future_horizon = self.config.future_len * self._step_interval  # [s]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path, map_location=device)

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

    def forward(self, features: WorldModelFeature) -> torch.Tensor:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        """
        inputs: Dict[str, Optional[torch.Tensor]] = features.to_tensor_dict()
        with torch.inference_mode():
            with amp_context_for_infer():
                _, outputs = self._planner(inputs)
        """
        outputs: Dict[str, torch.Tensor]
            "prediction" : (B, Pnn, 1+T, 4)
        """
        npc_future_trajectories = outputs['prediction']  # (B, Pnn, 1+T, 4)
        assert npc_future_trajectories.shape == (
            1,
            self.config.predicted_neighbor_num,
            1 + self.config.future_len,
            4,
        )
        npc_future_trajectories = npc_future_trajectories.squeeze(
            0)  # (Pnn, T, 4)
        return npc_future_trajectories
