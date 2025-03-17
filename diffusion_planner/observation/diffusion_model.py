from __future__ import annotations

import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type, Tuple, Optional

import timm

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.raster import Raster
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder

import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder

import dataclasses
from typing import List, Any, Dict
import torch
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature

import torch
import math

from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


def angle_difference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Safely compute the difference (a - b) in angle, wrapping to [-pi, pi].
    :param a: [*] (각 배치 차원을 포함한 임의 차원)
    :param b: [*] (각 배치 차원을 포함한 임의 차원)
    :return: [*]와 동일한 shape, angle wrap [-pi, pi]
    """
    diff = a - b
    # wrap to [-pi, pi]
    diff = (diff + math.pi) % (2 * math.pi) - math.pi
    return diff


def convert_decoder_outputs_to_agents_trajectories(
    predictions: torch.Tensor,
    dt: float = 0.1,
) -> AgentsTrajectories:
    """
    Convert decoder_outputs['prediction'] of shape (B, P, V_future, 4)
    into an AgentsTrajectories with shape (V_future, P, 6) per batch.

    * predictions.shape = (B, P, V_future, 4)
      - B: batch_size
      - P: number of agents or multi-predictions (depending on usage)
      - V_future: number of future time steps
      - last dim: [x, y, cosθ, sinθ]

    We'll compute:
      heading = atan2(sinθ, cosθ)
      vx, vy, yaw_rate with finite differences
      shape => (V_future, P, 6):
        (x, y, heading, vx, vy, yaw_rate)
    :param predictions: decoder output tensor
    :param dt: time interval (sec) between consecutive frames
    :return: AgentsTrajectories with data of shape (V_future, P, 6) for each batch
    """
    # predictions.shape = (B, P, V_future, 4)
    B = predictions.shape[0]  # batch size
    V_future = predictions.shape[2]  # number of future frames

    data_list = []

    for b_idx in range(B):
        # sample.shape = (P, V_future, 4)
        sample = predictions[b_idx]

        # permute => (V_future, P, 4)
        #   time dimension is first now
        sample = sample.permute(1, 0, 2)

        # Decompose sample
        # sample[..., 0]: x => shape (V_future, P)
        # sample[..., 1]: y => shape (V_future, P)
        # sample[..., 2]: cosθ => shape (V_future, P)
        # sample[..., 3]: sinθ => shape (V_future, P)
        x = sample[..., 0]
        y = sample[..., 1]
        cos_theta = sample[..., 2]
        sin_theta = sample[..., 3]

        # shape (V_future, P)
        heading = torch.atan2(sin_theta, cos_theta)

        # Allocate arrays for velocity, yaw_rate
        # shape = (V_future, P)
        vx = torch.zeros_like(x)  # (V_future, P)
        vy = torch.zeros_like(x)  # (V_future, P)
        yaw_rate = torch.zeros_like(x)  # (V_future, P)

        # Finite difference
        # for i=1..V_future-1, compute velocity, yaw_rate
        for point_idx in range(1, V_future):
            vx[point_idx] = (x[point_idx] -
                             x[point_idx - 1]) / dt  # shape => (P,)
            vy[point_idx] = (y[point_idx] -
                             y[point_idx - 1]) / dt  # shape => (P,)
            dtheta = angle_difference(heading[point_idx],
                                      heading[point_idx - 1])  # shape => (P,)
            yaw_rate[point_idx] = dtheta / dt

        # stack => shape (V_future, P, 6)
        # last dim: [x, y, heading, vx, vy, yaw_rate]
        final_sample = torch.stack([x, y, heading, vx, vy, yaw_rate], dim=-1)

        # accumulate in data_list
        data_list.append(final_sample)

    # data_list: length B
    #   each element shape = (V_future, P, 6)
    agent_trajectories = AgentsTrajectories(data=data_list)

    return agent_trajectories


def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(TorchModuleWrapper):

    def __init__(self,
                 config: Config,
                 ckpt_path: str,
                 future_trajectory_sampling: TrajectorySampling,
                 feature_builders: List[AbstractFeatureBuilder],
                 target_builders: Optional[List[AbstractTargetBuilder]] = None,
                 enable_ema: bool = True):
        if target_builders is None:
            target_builders = []
        super().__init__(
            feature_builders=feature_builders,
            target_builders=target_builders,
            future_trajectory_sampling=future_trajectory_sampling,
        )
        self._ema_enabled = enable_ema
        self._config = config
        self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)
        self._ckpt_path = ckpt_path
        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path)

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
            self.load_state_dict(model_state_dict)
        else:
            print("load random model")

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        The main inference call for the model.
        :param features: A dictionary of the required features.
            FeaturesType = Dict[str, ModelInputFeature]
            {"processed_data": ModelInputFeature}
        :return: The results of the inference as a TargetsType.
            decoder_outputs: TargetsType = Dict[str, AbstractModelFeature]
{
    ...
    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
    ...
}
        """
        features = features["processed_data"].to_dict()
        encoder_outputs = self.encoder(features)
        decoder_outputs = self.decoder(encoder_outputs, features)
        decoder_prediction_outputs = decoder_outputs[
            "prediction"][:, 1:]  # [B, P, V_future, 4]
        neighbor_selected_agents_token_str = decoder_outputs[
            "neighbor_selected_agents_token_str"]  # List[List[str]]
        time_gap = self.future_trajectory_sampling.step_time
        agents_trajectories = convert_decoder_outputs_to_agents_trajectories(
            decoder_prediction_outputs, time_gap)
        returns = {
            "agents_trajectory":
                agents_trajectories,
            "neighbor_selected_agents_token_str":
                neighbor_selected_agents_token_str
        }
        return returns


class Diffusion_Planner_Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight,
                        std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs


class Diffusion_Planner_Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(
            self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(
            self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return decoder_outputs
