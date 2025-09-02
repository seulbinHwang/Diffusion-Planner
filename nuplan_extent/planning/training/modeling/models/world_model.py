from typing import List, Dict, Deque

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
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states


# nuplan/planning/training/preprocessing/feature_builders/raster_feature_builder.py
# nuplan_extent.planning.training.preprocessing.feature_builders.horizon_vector_feature_builder.GenericAgentsFeatureBuilder
class WorldModel(TorchModuleWrapper):

    def __init__(
        self,
        config: Config,  #
        ckpt_path: str,  #
        feature_builders: List[AbstractFeatureBuilder],  #
        target_builders: List[AbstractTargetBuilder],  # 안씀
        future_trajectory_sampling: TrajectorySampling,  # 안씀
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

    def outputs_to_trajectory(
            self, outputs: Dict[str, torch.Tensor],
            ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(
            np.float64)  # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1)

        states = transform_predictions_to_states(predictions, ego_state_history,
                                                 self._future_horizon,
                                                 self._step_interval)

        return states

    def forward(self, features: WorldModelFeature) -> torch.Tensor:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
        """
        inputs = features.to_tensor_dict()
        _, outputs = self._planner(inputs)
        """
        outputs: Dict[str, torch.Tensor]
            "prediction" : (B, Pnn, T, 4)
        """
        npc_future_trajectories = outputs['prediction']  # (B, Pnn, T, 4)
        assert npc_future_trajectories.shape == (
            1,
            self.config.predicted_neighbor_num,
            self.config.future_len,
            4,
        )
        npc_future_trajectories = npc_future_trajectories.squeeze(
            0)  # (Pnn, T, 4)
        return npc_future_trajectories
        """
        TODO: npc_future_trajectories 는 x, y, cos(yaw), sin(yaw) 로 되어있음.
        그런데, 각 차량의 중심에 대한 x, y, yaw 값임. (ego 좌표계 기준)
        나는 npc_future_trajectories를, 각 챠량의 rear_axle 좌표계 기준으로 바꾸고 싶음.
        """
        for agent_idx in range(npc_future_trajectories.shape[0]):
            npc_future_trajectory = npc_future_trajectories[agent_idx]  # (T, 4)
            """
            TODO:
            npc_future_trajectory 의 값에 대해
                ego 뒷축 좌표계 → vehicle 뒷축 좌표계 로 일괄 변환
            """
            future_trajectory = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(npc_future_trajectory,
                                                      self_history))
