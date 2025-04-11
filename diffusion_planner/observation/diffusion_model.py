from __future__ import annotations

import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type, Tuple, Optional

import timm
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

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
import itertools

import torch
import math
from matplotlib.ticker import MultipleLocator
from nuplan.planning.simulation.controller.tracker.tracker_utils import get_velocity_curvature_profiles_with_derivatives_from_poses

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
        sample = sample.permute(1, 0, 2)  #

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
        velocity_profile = []
        agents_number = sample.shape[1]
        for agent_idx in range(agents_number):
            poses_horizon = torch.stack([x[:, agent_idx], y[:, agent_idx], heading[:, agent_idx]], dim=-1)
            poses_horizon_np = poses_horizon.detach().cpu().numpy()
            a_velocity_profile, _, _, _ = get_velocity_curvature_profiles_with_derivatives_from_poses(
                discretization_time=dt,
                poses=poses_horizon_np,  # (num_frames, 3)
                jerk_penalty=1e-4,
                curvature_rate_penalty=1e-2,
            )
            # a_velocity_profile (num_frames-1, ) -> (num_frames, )
            a_velocity_profile = np.concatenate(
                (a_velocity_profile, a_velocity_profile[-1:]))
            a_velocity_profile = torch.tensor(a_velocity_profile, dtype=torch.float32).to(x.device)
            velocity_profile.append(a_velocity_profile)
        velocity_profile = torch.stack(velocity_profile, dim=1)  # (V_future, P)
        vx = velocity_profile * cos_theta # shape => (V_future, P)
        vy = velocity_profile * sin_theta # shape => (V_future, P)
        yaw_rate = torch.zeros_like(x)  # (V_future, P)

        # Finite difference
        # for i=1..V_future-1, compute velocity, yaw_rate

        # for point_idx in range(1, V_future):
        #     vx[point_idx] = (x[point_idx] -
        #                      x[point_idx - 1]) / dt  # shape => (P,)
        #     vy[point_idx] = (y[point_idx] -
        #                      y[point_idx - 1]) / dt  # shape => (P,)
        #     dtheta = angle_difference(heading[point_idx],
        #                               heading[point_idx - 1])  # shape => (P,)
        #     yaw_rate[point_idx] = dtheta / dt

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
        self.count = 0
        # 토큰(string) -> color 매핑을 저장해둘 딕셔너리
        self.token2color = {}

        # color cycle 준비 (matplotlib 기본 color, 무한반복)
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_iterator = itertools.cycle(color_cycle)

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

    def plot_agents_trajectories(
        self,
        neighbor_agents_past,  # (num_agents_past, past_points, 2)
        past_agent_tokens,  # List[str], 길이 <= num_agents_past
        npc_decoder_prediction_outputs,
        # (num_agents_future, future_points, 2),
        ego_decoder_prediction_outputs,  # (future_points, 2)
        future_agent_tokens,  # List[str], 길이 <= num_agents_future
        save_path="neighbor_agents_future_plot.png"):
        """
        - neighbor_agents_past : (num_agents_past, V_past, 2)
        - past_agent_tokens    : List[str] (길이 ≤ num_agents_past)
        - npc_decoder_prediction_outputs : (num_agents_future, V_future, 2)
        - future_agent_tokens           : List[str] (길이 ≤ num_agents_future)
        - save_path : 저장 경로
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # ---------- 과거 궤적(큰 원) 그리기 ----------
        num_past_agents = neighbor_agents_past.shape[0]
        for i in range(num_past_agents):
            traj = neighbor_agents_past[i]
            if isinstance(traj, torch.Tensor):
                traj = traj.detach().cpu().numpy()

            x = traj[:, 0]
            y = traj[:, 1]

            # past 토큰 가져오기
            token_str = None
            if i < len(past_agent_tokens):
                token_str = past_agent_tokens[i]
            else:
                break
            # 새 토큰이면 새 color 할당, 기존이면 기존 color 사용
            color = self.get_color_for_token(token_str)

            # 큰 원 (marker='o', markersize=10)
            ax.plot(x,
                    y,
                    marker='o',
                    markersize=1,
                    color=color,
                    label=f"agent_past_{i}")

            # 마지막 점에 토큰 표시
            if token_str is not None:
                ax.text(x[-1],
                        y[-1],
                        "past",
                        fontsize=8,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        color=color)

        # ---------- 미래 궤적(작은 X) 그리기 ----------
        ego_traj = ego_decoder_prediction_outputs
        if isinstance(ego_traj, torch.Tensor):
            ego_traj = ego_traj.detach().cpu().numpy()

        x = ego_traj[:, 0]
        y = ego_traj[:, 1]

        # future 토큰 가져오기

        # 작은 X (marker='x', markersize=6), 선은 점선(linestyle='--')
        ax.plot(
            x,
            y,
            marker='x',
            markersize=2,
            # linestyle='--',
            color='black',
            label=f"ego_future")

        num_future_agents = npc_decoder_prediction_outputs.shape[0]
        for i in range(num_future_agents):
            traj = npc_decoder_prediction_outputs[i]
            if isinstance(traj, torch.Tensor):
                traj = traj.detach().cpu().numpy()

            x = traj[:, 0]
            y = traj[:, 1]

            # future 토큰 가져오기
            token_str = None
            if i < len(future_agent_tokens):
                token_str = future_agent_tokens[i]

            # 새 토큰이면 새 color 할당, 기존이면 기존 color 사용
            color = self.get_color_for_token(token_str)

            # 작은 X (marker='x', markersize=6), 선은 점선(linestyle='--')
            ax.plot(
                x,
                y,
                marker='x',
                markersize=1,
                # linestyle='--',
                color=color,
                label=f"agent_future_{i}")

            # 마지막 점에 토큰 표시
            if token_str is not None:
                ax.text(x[-1],
                        y[-1],
                        "future",
                        fontsize=8,
                        verticalalignment='bottom',
                        horizontalalignment='right',
                        color=color)

        # ---- 공통 축 설정 (눈금, 범위, 비율 등) ----
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Neighbors Past & Future Trajectories')

        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.set_xlim(-80, 80)
        ax.set_ylim(-80, 80)

        ax.set_aspect('equal', adjustable='box')
        ax.autoscale(False)
        ax.grid(True, which='major', linestyle='--', color='gray')

        # 범례 표시 여부 (필요하다면)
        # ax.legend(loc='upper right')

        plt.tight_layout()

        # 그림 저장 & 종료
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def get_color_for_token(self, token_str):
        """
        토큰에 해당하는 색깔을 반환.
        - token_str가 None이거나 빈 문자열인 경우, 기본 'gray' 사용
        - 이미 등록된 토큰이면 self.token2color[token_str] 반환
        - 새 토큰이면 color_iterator에서 새 color를 꺼내와 등록 후 반환
        """
        if not token_str:  # None 또는 빈 문자열
            return 'gray'

        if token_str not in self.token2color:
            # 새로운 토큰 → 새 색상 할당
            self.token2color[token_str] = next(self.color_iterator)

        return self.token2color[token_str]

    def _save_debug_graph(
            self,
            positions: np.ndarray,
            headings_rad: np.ndarray,
            velocities_mps: np.ndarray,
            output_path: str
    ) -> None:
        """
        positions : 임의 위치(가령 m 단위), shape (K,)
        headings_rad : 라디안 단위 heading, shape (N,)
        velocities_mps : m/s 단위 velocity, shape (M,)
        output_path : 결과 그래프를 저장할 파일 경로

        - 1행 3열 subplot 구조로 그림:
          (1) Heading subplot: 라디안 -> deg 변환
          (2) Velocity subplot: m/s -> km/h 변환
          (3) Position subplot: positions 그대로

        모든 subplot에서 x축은 index (0 ~ len-1)
        """
        import matplotlib.pyplot as plt
        import numpy as np




        # (1) Position: 그대로 사용
        x_position = np.arange(len(positions))  # 0..K-1

        # (2) Heading: rad -> deg
        headings_deg = np.degrees(headings_rad)
        x_heading = np.arange(len(headings_deg))  # 0..N-1

        # (3) Velocity: m/s -> km/h
        velocities_kmh = velocities_mps * 3.6
        x_velocity = np.arange(len(velocities_kmh))  # 0..M-1

        # 서브플롯 구성: 1행 3열
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), tight_layout=True)



        # --- Subplot (0): Position ---
        axes[0].plot(x_position, positions, marker='o', linestyle='-',
                     color='green')
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Position (m)")  # 필요에 따라 단위 라벨 수정
        axes[0].set_title("Position Profile")

        # --- Subplot (1): Heading (deg) ---
        axes[1].plot(x_heading, headings_deg, marker='o', linestyle='-')
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Heading (deg)")
        axes[1].set_title("Heading Profile")
        # --- Subplot (3): Velocity (km/h) ---
        axes[2].plot(x_velocity, velocities_kmh, marker='o', linestyle='-',
                     color='orange')
        axes[2].set_xlabel("Index")
        axes[2].set_ylabel("Velocity (km/h)")
        axes[2].set_title("Velocity Profile")

        # 저장
        plt.savefig(output_path, dpi=150)
        plt.close(fig)


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

        all_decoder_prediction_outputs = decoder_outputs[
            "prediction"]  # [B, P+1, V_future, 4] # Exclude current state
        neighbor_selected_agents_token_str = decoder_outputs[
            "neighbor_selected_agents_token_str"]  # List[List[str]]
        a_all_decoder_prediction_outputs_four = all_decoder_prediction_outputs.clone(
        ).detach()[0]  # [P+1, V_future, 4]
        a_all_decoder_prediction_outputs = a_all_decoder_prediction_outputs_four[:, :, :2]  # [P+1, V_future, 2]
        ego_decoder_prediction_outputs = a_all_decoder_prediction_outputs[
            0]  # [V_future, 2]
        npc_decoder_prediction_outputs = a_all_decoder_prediction_outputs[
            1:]  # [P, V_future, 2]
        npc_neighbor_selected_agents_token_str = neighbor_selected_agents_token_str[
            0]  # List[str]
        ######### DRAW ##########
        neighbor_agents_past = features["neighbor_agents_original"][
            0][..., :2]  # [P, V_past, 2]
        neighbor_agents_token_str_small = features["neighbor_agents_token_str"][
            0]  # List[str]
        # if self.count < 50:
        #     self.plot_agents_trajectories(
        #         neighbor_agents_past,
        #         neighbor_agents_token_str_small,
        #         npc_decoder_prediction_outputs,
        #         ego_decoder_prediction_outputs,
        #         npc_neighbor_selected_agents_token_str,
        #         save_path=f"[{self.count}]draw_{self.count}.png")
        neighbor_agents_future_np = a_all_decoder_prediction_outputs_four.cpu().numpy()[1:] # [P, V_future, 4]
        neighbor_agents_past_np_original = features["neighbor_agents_original"][
            0].cpu().numpy() # [P, V_past, 6]
        neighbor_agents_past_np = features["neighbor_agents_original"][
            0][..., :4].cpu().numpy() # [P, V_past, 4]
        neighbor_agents_past_np_vxy = features["neighbor_agents_original"][
            0][..., 4:6].cpu().numpy() # [P, V_past, 2]
        ##########################
        # Exclude the ego agent's prediction
        decoder_prediction_outputs = decoder_outputs[
            "prediction"][:, 1:]  # [B, P, V_future, 4]
        neighbor_selected_agents_token_str = decoder_outputs[
            "neighbor_selected_agents_token_str"]  # List[List[str]]
        time_gap = self.future_trajectory_sampling.step_time
        assert time_gap == 0.1, f"Time gap should be 0.1, but got {time_gap}"
        agents_trajectories = convert_decoder_outputs_to_agents_trajectories(
            decoder_prediction_outputs, time_gap)
        agents_trajectories_torch = agents_trajectories.data[0]  # [V_future, P, 6]
        agents_trajectories_np = agents_trajectories_torch.cpu().numpy()  # [V_future, P, 6]
        all_agents_num = neighbor_agents_future_np.shape[0] # [P, V_future, 4]
        past_agents_trajectory = []
        for i in range(all_agents_num):
            token_str = npc_neighbor_selected_agents_token_str[i] # str
            past_token_idx = neighbor_agents_token_str_small.index(token_str)
            a_past_trajectory = neighbor_agents_past_np_original[
                past_token_idx]  # [V_past, 6]
            past_agents_trajectory.append(a_past_trajectory)
        past_agents_trajectory = np.stack(past_agents_trajectory, axis=0) # [P, V_past, 6]
        returns = {
            "agents_trajectory":
                agents_trajectories,
            "neighbor_selected_agents_token_str":
                neighbor_selected_agents_token_str,
            "past_agents_trajectory": past_agents_trajectory # [P, V_past, 6]
        }
        ###############DEBUG###########################
        all_agents_num = neighbor_agents_future_np.shape[0] # [P, V_future, 4]
        for i in range(all_agents_num):
            save_path = f"[{self.count}]{i}_neighbor_agents_pose_plot.png"
            a_future_trajectory = neighbor_agents_future_np[i] # [V, 4]
            token_str = npc_neighbor_selected_agents_token_str[i] # str
            past_token_idx = neighbor_agents_token_str_small.index(token_str)
            a_past_trajectory = neighbor_agents_past_np[past_token_idx] # [V_past, 4]
            a_past_future_trajectory = np.concatenate([a_past_trajectory, a_future_trajectory], axis=0) # [V_past + V_future, 4]
            a_past_future_trajectory_xy = a_past_future_trajectory[:, :2] # [V, 2]
            a_past_future_trajectory_xy_sum = np.sum(a_past_future_trajectory_xy, axis=1) # [V]
            cos_ = a_past_future_trajectory[:, 2]
            sin_ = a_past_future_trajectory[:, 3]
            a_past_future_trajectory_heading_rad = np.arctan2(sin_, cos_) # [V_future]
            ##############################
            agent_trajectories_np = agents_trajectories_np[:, i] # [V_future, 6]
            agent_trajectories_np_vxvy = agent_trajectories_np[:, 3:5] # [V_future, 2]
            neighbor_agents_past_np_vxy_i = neighbor_agents_past_np_vxy[past_token_idx] # [V_past, 2]
            a_past_future_trajectory_vxy = np.concatenate([neighbor_agents_past_np_vxy_i, agent_trajectories_np_vxvy], axis=0) # [V_past + V_future, 2]
            a_past_future_trajectory_vxy_norm = np.linalg.norm(a_past_future_trajectory_vxy, axis=1) # [V_past + V_future]
            ###############DEBUG###########################
            # self._save_debug_graph(
            #     a_past_future_trajectory_xy_sum, # [V]
            #     a_past_future_trajectory_heading_rad, # [V]
            #     a_past_future_trajectory_vxy_norm, # [V]
            #     save_path
            # )
            ###############DEBUG###########################
        self.count += 1
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
