import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath
from typing import Optional, Dict, Tuple
from flash_attn.bert_padding import unpad_input, pad_input
from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, \
    StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock
from diffusion_planner.loss import _require_finite
# decoder.py 상단 import 섹션에 추가
from diffusion_planner.model.module.pram_v2 import (
    PRAMV2Composer,
    PRAMV2TimeModulator,
    PRAMV2BlockPathScalars,
    PRAMV2StateTokenEncoder,
    compute_pram_v2_modulations_for_block,
    apply_pram_v2_final_layer,  # ← 9단계 마무리 보정 호출용(스켈레톤이어도 OK)
)
from typing import Tuple, Optional
from diffusion_planner.model.module.feasible import FeasibleProjector


def _cast_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """ref 텐서의 dtype/device로 x를 캐스팅합니다."""
    return x.to(dtype=ref.dtype, device=ref.device)


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len: int = config.future_len
        self._sde = VPSDE_linear()
        self._cond_last_prob: float = getattr(config, "cond_last_prob",
                                              0.0)  # 20%

        if self.config.use_current_input:
            output_dim = (config.future_len + 1) * 4  # x, y, cos, sin
        else:
            output_dim = (config.future_len) * 4  # x, y, cos, sin

        self.dit = DiT(
            config=config,
            sde=self._sde,
            # route_encoder=RouteEncoder(
            #     config.route_num,
            #     config.lane_len,
            #     drop_path_rate=config.encoder_drop_path_rate,
            #     hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=output_dim,  # x, y, cos, sin
            hidden_dim=config.hidden_dim,
            heads=config.num_heads,
            dropout=dpr,
            model_type=config.diffusion_model_type)

        self._state_normalizer: StateNormalizer = config.state_normalizer
        self._observation_normalizer: ObservationNormalizer = config.observation_normalizer

        # self._guidance_fn = config.guidance_fn
        self._guidance_fn = getattr(config, 'guidance_fn', None)

    @property
    def sde(self):
        return self._sde

    # TODO: 점검하기
    def _maybe_apply_last_pos_condition_training(
        self,
        near_cur_future_norm_xT: torch.Tensor,
        # (B, Pnn, (T) * 4)  # 정규화 상태
        near_current_mask: torch.Tensor,  # (B, Pnn)  True=무효 에이전트
        cond_last_pos_norm: torch.Tensor,  # (B, Pnn, 4)  # 정규화 목표점
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """

        훈련 시, **배치 단위**로 cond_last_prob 확률(기본 20%)에 따라
        마지막 스텝(목표점)을 관측(노이즈 0)으로 **전원 적용/전원 미적용**한다.

        배치 토글:
            - 토글=True (약 20%): 이번 배치의 모든 유효 Pnn에 목표를 주입
            - 토글=False (약 80%): 이번 배치에는 전혀 목표를 주지 않음

        Args:
            near_cur_future_norm_xT: (B, Pnn, T * 4), 정규화된 입력 시퀀스.
            near_current_mask: (B, Pnn), True=무효 이웃 에이전트.
            cond_last_pos_norm: (B, Pnn, 4), 정규화된 목표점(마지막 프레임 값).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - x_t_out: (B, Pnn, T * 4)
                    * 배치 토글이 True일 때, 모든 유효 에이전트의 마지막 프레임을
                      cond_last_pos_norm으로 치환(=노이즈 0).
                    * 배치 토글이 False면 입력을 그대로 반환.
                - apply_mask: (B, Pnn) 또는 None
                    * 이번 배치에서 실제로 목표를 적용한 에이전트 마스크(로깅/평가용).
                    * 미적용 시 None.
        """
        B, Pnn, flat = near_cur_future_norm_xT.shape
        assert flat % 4 == 0, "near_cur_future_norm_xT의 마지막 차원은 4의 배수여야 합니다."
        T_plus1: int = flat // 4
        assert T_plus1 >= 2, "미래 길이 T는 최소 1 이상이어야 합니다."
        assert cond_last_pos_norm.shape == (B, Pnn,
                                            4), "cond_last_pos_norm shape 불일치"
        cond_last_pos_norm = _cast_like(cond_last_pos_norm,
                                        near_cur_future_norm_xT)
        # cond_last_prob==0 이면 빠르게 종료
        if self._cond_last_prob <= 0.0:
            return near_cur_future_norm_xT, None

        # (1) 적용 가능 위치: 유효 에이전트 & has_goal(목표값이 전부 0 이 아닌)
        has_goal: torch.Tensor = torch.isfinite(cond_last_pos_norm).all(
            dim=-1) & (cond_last_pos_norm.ne(0).sum(dim=-1) > 0)  # (B, Pnn)
        can_apply: torch.Tensor = (~near_current_mask) & has_goal  # (B, Pnn)

        # (2) 배치 단일 베르누이 샘플: 약 20% 확률로 전체 적용
        batch_toggle: bool = (torch.rand(
            (), device=near_cur_future_norm_xT.device).item()
                              < float(self._cond_last_prob))

        if not batch_toggle:
            # 이번 배치는 전원 미적용
            return near_cur_future_norm_xT, None

        # (3) 전체 적용: 유효한 위치(can_apply)에만 주입
        if not can_apply.any().item():
            # 유효한 에이전트가 하나도 없으면 변경 없음
            return near_cur_future_norm_xT, None

        x_seq = near_cur_future_norm_xT.view(B, Pnn, T_plus1,
                                             4).clone()  # (B, Pnn, 1 + T, 4)

        # 마지막 프레임만 뽑아서 (B, Pnn, 4) 모양으로 맞춘 뒤 where로 바꿔끼우기
        last = x_seq[:, :, -1, :]  # (B, Pnn, 4)
        apply_cond = can_apply.unsqueeze(-1)  # (B, Pnn, 1) → 4에 브로드캐스트
        # cond_last_pos_norm은 이미 _cast_like로 dtype/device 맞춤
        last = torch.where(apply_cond, cond_last_pos_norm, last)  # (B, Pnn, 4)
        x_seq[:, :, -1, :] = last

        # x_out: (B, Pnn, T * 4)
        # can_apply: (B, Pnn)
        x_out = x_seq.view(B, Pnn, flat)
        return x_out, can_apply

    def _get_near_current_infos(
        self,
        target_agents_mask: Optional[torch.Tensor],  # [B, agent_num] bool
        neighbor_agents_past: torch.Tensor,  # [B, agent_num, time_len, 11]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            near_current_xyyaw: [B, pnn, 4]  (x, y, cos(yaw), sin(yaw))
            near_current_mask: [B, pnn]  True=빈 슬롯(무효 에이전트)
            near_class_one_hot: [B, pnn, 3]  one-hot class vector
        """
        # neighbor_agents_past_xyyaw: [B, agent_num, time_len, 4]
        neighbor_agents_past_xyyaw = neighbor_agents_past[..., :4]
        neighbor_agents_class_one_hot = neighbor_agents_past[
            ..., 0, 8:11]  # (B, agent_num, 3)
        # 마지막 타임스텝만 추출: [B, agent_num, 4]
        neighbor_current_xyyaw = neighbor_agents_past_xyyaw[:, :, -1, :]

        if target_agents_mask is None:
            near_current_xyyaw = neighbor_current_xyyaw[:, :self.
                                                        _predicted_neighbor_num, :]  # [B, pnn, 4]
            near_class_one_hot = neighbor_agents_class_one_hot[:, :self.
                                                               _predicted_neighbor_num, :]  # [B, pnn, 3]
            # [B, pnn]
            near_current_mask = torch.sum(torch.ne(near_current_xyyaw, 0),
                                          dim=-1) == 0

        else:
            B, agent_num, time_len, D4 = neighbor_agents_past_xyyaw.shape
            # 출력 버퍼(초기 0): [B, agent_num, 4]
            near_current_xyyaw = torch.zeros(
                (B, agent_num, 4),
                dtype=neighbor_current_xyyaw.dtype,
                device=neighbor_current_xyyaw.device,
            )
            near_class_one_hot = torch.zeros(
                (B, agent_num, 3),
                dtype=neighbor_agents_class_one_hot.dtype,
                device=neighbor_agents_class_one_hot.device,
            )

            # 마스크가 True인 “그 자리”에 값 대입 (슬롯 유지)
            # target_agents_mask: [B, agent_num] bool
            near_current_xyyaw[target_agents_mask] = neighbor_current_xyyaw[
                target_agents_mask]
            near_class_one_hot[
                target_agents_mask] = neighbor_agents_class_one_hot[
                    target_agents_mask]  #
            # TODO: 임시 -> _predicted_neighbor_num 에 대한 의존성 타파?
            near_current_xyyaw = near_current_xyyaw[:, :self.
                                                    _predicted_neighbor_num, :]  # [B, pnn, 4]
            near_class_one_hot = near_class_one_hot[:, :self.
                                                    _predicted_neighbor_num, :]  # [B, pnn, 3]
            # [B, pnn]  True=빈 슬롯(무효 에이전트)
            near_current_mask = (near_current_xyyaw.ne(0).sum(dim=-1) == 0)
        return near_current_xyyaw, near_current_mask, near_class_one_hot

    def _get_near_cur_future_valid(
        self,
        near_current_mask: torch.Tensor,  # [B, pnn] bool
        near_future_valid: Optional[
            torch.Tensor] = None,  # [B, pnn, future_len] bool
    ) -> torch.Tensor:  # [B, pnn, 1 + future_len] bool
        near_current_valid = ~near_current_mask  # [B, pnn]  True=유효 에이전트
        if near_future_valid is None:
            near_future_valid = near_current_valid.unsqueeze(-1).expand(
                -1, -1, self._future_len)  # [B, pnn, future_len] bool
        near_cur_future_valid = torch.cat(
            [near_current_valid.unsqueeze(-1), near_future_valid],
            dim=-1)  # [B, pnn, 1 + future_len] bool
        return near_cur_future_valid

    def forward(self, encoder_outputs, inputs):
        """
        Diffusion decoder process.

        Args:
            encoder_outputs: Dict
                {
                    ...
                    "encoding": agents, static objects and lanes context encoding
                    ...
                }
            inputs: Dict
                {
                    ...
                    "ego_agent_past": past and current ego states,
                    "ego_current_state": current ego states,
                    "neighbor_agent_past": past and current neighbor states,

                    [training-only] "near_cur_future_norm_xT": sampled current-future ego & neighbor states,        [B, P, 1 + future_len, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + future_len, 4]
                    [inference-only] "score": Predicted future states, [B, P, 1 + future_len, 4]
                    ...
                }

        """

        # near_current_xyyaw: [B, pnn, 4]  (x, y, cos(yaw), sin(yaw))
        # near_current_mask: [B, pnn]  True=빈 슬롯(무효 에이전트)
        (near_current_xyyaw, near_current_mask,
         near_class_one_hot) = self._get_near_current_infos(
             target_agents_mask=inputs.get("target_agents_mask",
                                           None),  # [B, agent_num] bool
             neighbor_agents_past=inputs[
                 "neighbor_agents_past"],  # [B, agent_num, time_len, 11]
         )
        inputs["near_current_mask"] = near_current_mask
        near_cur_future_valid = self._get_near_cur_future_valid(
            near_current_mask,
            near_future_valid=inputs.get("near_future_valid",
                                         None)  # [B, pnn, future_len] bool
        )

        return_ = {}
        B, Pnn, _ = near_current_xyyaw.shape

        if "cond_last_pos_norm" in inputs:
            # 길이(pnn_dyn)에 맞춰 잘라서 정합 보장
            cond_last_pos_norm = inputs[
                "cond_last_pos_norm"]  # [:, :Pnn, :]  # [B, Pnn, 4]
        else:
            # NaN으로 채워서 'isfinite' 검사에 의해 자동 미적용되게 만든다.
            cond_last_pos_norm = near_current_xyyaw.new_full((B, Pnn, 4),
                                                             float('nan'))

        # ★ FIX: 이후 모든 사용을 안전하게 만들기 위해 dtype/device를 near_current에 정렬
        cond_last_pos_norm = _cast_like(cond_last_pos_norm, near_current_xyyaw)

        # Extract context encoding
        scene_encoding_token = encoder_outputs[
            'encoding']  # (B, token_num, hidden_dim)
        scene_encoding_token_mask = encoder_outputs[
            'encoding_mask']  # (B, token_num) bool
        ego_fut_global = encoder_outputs["ego_fut_global"]  # (B, hidden_dim)
        near_agents_route_lane_emb = encoder_outputs[
            "near_agents_route_lane_emb"]  # (B, Pnn, hidden_dim)
        route_known_mask = encoder_outputs[
            "route_known_mask"]  # (B, Pnn) bool # True=해당 에이전트가 유효 route
        assert ego_fut_global.shape == (B, scene_encoding_token.shape[-1])

        if self.training:
            near_cur_future_norm_xT = inputs[
                "near_cur_future_norm_xT"]  # [B, Pnn, 1 + T, 4]
            near_cur_norm_xT = near_cur_future_norm_xT[:, :,
                                                       0, :]  # [B, Pnn, 4]
            near_future_norm_xT = near_cur_future_norm_xT[:, :,
                                                          1:, :]  # [B, Pnn, T, 4]
            if self.config.use_current_input:
                xT_input = near_cur_future_norm_xT
            else:
                xT_input = near_future_norm_xT

            xT_input = xT_input.reshape(B, Pnn, -1)  # [B, Pnn, T * 4]

            # 🔹 20% 확률로 마지막 프레임(목표) 주입 — Conditioned Generation 학습 신호
            # near_cur_future_norm_xT: [B, Pnn, T * 4]
            # near_current_mask [B, pnn]
            # cond_last_pos_norm: [B, Pnn, 4]
            xT_input, _ = self._maybe_apply_last_pos_condition_training(
                xT_input, near_current_mask, cond_last_pos_norm)
            diffusion_time = inputs['diffusion_time']
            # (B, Pnn, T , 4) or (B, Pnn, (1+T) , 4)
            score = self.dit(
                xT_input,  # ( B, Pnn, T* 4 ) or  (B, Pnn, (1+T)*4)
                diffusion_time,  # (B)
                scene_encoding_token,  # (B, token_num, hidden_dim)
                ego_fut_global,  # (B, hidden_dim)
                near_agents_route_lane_emb,  # (B, Pnn, hidden_dim)
                near_cur_future_valid,  # [B, pnn, 1 + future_len] bool
                scene_encoding_token_mask,  # (B, token_num) bool
                route_known_mask,  # (B, Pnn) bool # True=해당 에이전트가 유효 route
                near_class_one_hot,  # (B, Pnn, 3)
                near_current_xyyaw=near_cur_norm_xT,  # (B, Pnn, 4)
            )
            _require_finite("decoder_dit_output", score)
            if self.config.use_current_input:
                score = score.reshape(B, Pnn, 1 + self._future_len,
                                      4)  # (B,Pnn,T,4)
            else:
                score = score.reshape(B, Pnn, self._future_len,
                                      4)  # (B,Pnn,T,4)
                # TODO: near_current_xyyaw 를 detach() 해서 붙이는 게 맞는지 점검
                score = torch.cat([near_current_xyyaw.unsqueeze(2), score],
                                  dim=2)  # (B,Pnn,1+T,4)
            return_["score"] = score  # (B, Pnn, (1 + T) , 4)

            if self.config.use_feasible:
                integrated_trajectory = self.dit.dit_returns.integrated_trajectory  # (B, Pnn, T, 4)
                integrated_trajectory = torch.cat(
                    [near_current_xyyaw.unsqueeze(2), integrated_trajectory],
                    dim=2)  # (B,Pnn,1+T,4)
                return_[
                    "integrated_trajectory"] = integrated_trajectory  # (B, Pnn, (1 + T) , 4)
                return_[
                    "control_constraint_diff"] = self.dit.dit_returns.control_constraint_diff  # (B,Pnn,T,3)
            return return_
        else:
            noise = near_current_xyyaw.new_empty(
                (B, Pnn, self._future_len, 4)).normal_(0.0,
                                                       0.5)  # (B, Pnn, T, 4)
            if self.config.use_current_input:
                # xT: (B, Pnn, (1+T)*4)
                xT = torch.cat(
                    [
                        near_current_xyyaw[:, :, None, :],  # (B, Pnn, 1, 4)
                        noise  # (B, Pnn, T, 4)
                    ],
                    dim=2).reshape(B, Pnn, -1)
            else:
                xT = noise.reshape(B, Pnn, -1)  # (B, Pnn, T*4)

            # cond_last_pos_norm: [B, Pnn, 4] (이미 near_current와 dtype/device 일치)
            cond_last_pos = None
            if torch.isfinite(cond_last_pos_norm).any():
                cond_last_pos = cond_last_pos_norm

            if cond_last_pos is not None:
                cond_last_mask = torch.isfinite(cond_last_pos).all(
                    dim=-1)  # [B, Pnn]
            else:
                # (B, Pnn) # True 이면 last pose 정보가 있다는 뜻
                cond_last_mask = torch.zeros(B,
                                             Pnn,
                                             dtype=torch.bool,
                                             device=xT.device)

            def initial_state_constraint(xt, t, step):
                if self.config.use_current_input:
                    xt = xt.reshape(B, Pnn, 1 + self._future_len, 4)
                    xt[:, :, 0, :] = near_current_xyyaw
                else:
                    xt = xt.reshape(B, Pnn, self._future_len, 4)

                # cond-last injection (있을 때만)
                if torch.isfinite(cond_last_pos_norm).any():
                    cond_last_mask = torch.isfinite(cond_last_pos_norm).all(
                        dim=-1)  # (B,Pnn)
                    if cond_last_mask.any().item():
                        last = xt[:, :, -1, :]  # (B,Pnn,4)
                        src = _cast_like(cond_last_pos_norm, xt)
                        last = torch.where(cond_last_mask.unsqueeze(-1), src,
                                           last)
                        xt[:, :, -1, :] = last

                # --- add: unit‑circle projection for future frames only ---
                # yaw 단위원 투영 (미래 전 프레임)
                yaw = xt[:, :, :, 2:4]  # (B, Pnn, T, 2)
                norm = torch.linalg.norm(yaw, dim=-1, keepdim=True).clamp_min(
                    1e-6)  # (B, Pnn, T, 1)
                xt[:, :, :, 2:4] = yaw / norm  # (B, Pnn, T, 2)
                # ---------------------------------------------------------

                return xt.reshape(B, Pnn, -1)

            x0 = dpm_sampler(
                self.dit,
                xT.float(),
                other_model_params={
                    "cross_c": scene_encoding_token,
                    "ego_fut_global": ego_fut_global,
                    "near_agents_route_lane_emb": near_agents_route_lane_emb,
                    "near_cur_future_valid": near_cur_future_valid,
                    "cross_mask": scene_encoding_token_mask,
                    "route_known_mask": route_known_mask,
                    "near_class_one_hot": near_class_one_hot,
                    "near_current_xyyaw": near_current_xyyaw,
                },
                dpm_solver_params={
                    "correcting_xt_fn": initial_state_constraint,
                },
                model_wrapper_params={
                    "classifier_fn":
                        self._guidance_fn,
                    "classifier_kwargs": {
                        "model": self.dit,
                        "model_condition": {
                            "cross_c":
                                scene_encoding_token,
                            "ego_fut_global":
                                ego_fut_global,
                            "near_agents_route_lane_emb":
                                near_agents_route_lane_emb,
                            "near_cur_future_valid":
                                near_cur_future_valid,
                            "cross_mask":
                                scene_encoding_token_mask,
                            "route_known_mask":
                                route_known_mask,
                            "near_class_one_hot":
                                near_class_one_hot,
                        },
                        "inputs": inputs,
                        "observation_normalizer": self._observation_normalizer,
                        "state_normalizer": self._state_normalizer
                    },
                    "guidance_scale":
                        0.5,
                    "guidance_type":
                        "classifier"
                        if self._guidance_fn is not None else "uncond"
                },
            )
            x0 = x0.to(xT.dtype)
            if self.config.use_current_input:
                assert x0.shape == (B, Pnn, (1 + self._future_len) * 4)
                x0 = x0.reshape(B, Pnn, -1, 4)  # (B,Pnn,1+T,4)
            else:
                assert x0.shape == (B, Pnn, self._future_len * 4)
                # concat near_current_xyyaw to x0.
                x0 = torch.cat([
                    near_current_xyyaw.unsqueeze(2),
                    x0.reshape(B, Pnn, -1, 4)
                ],
                               dim=2)  # (B,Pnn,1+T,4)
            unnorm_x0 = self._state_normalizer.inverse(x0)  # (B,Pnn,1+T,4)
            if self.config.use_feasible:
                integrated_trajectory = self.dit.dit_returns.integrated_trajectory  # (B, Pnn, T, 4)
                integrated_trajectory = torch.cat(
                    [near_current_xyyaw.unsqueeze(2), integrated_trajectory],
                    dim=2)  # (B,Pnn,1+T,4)
                unnorm_integrated_trajectory = self._state_normalizer.inverse(
                    integrated_trajectory)  # (B,Pnn,1+T,4)
                return_[
                    "integrated_trajectory"] = unnorm_integrated_trajectory  # (B, Pnn, (1 + T) , 4)
            return_["score"] = unnorm_x0  # (B, Pnn, (1 + T) , 4)
            return return_


class RouteEncoder(nn.Module):

    def __init__(self,
                 route_num,
                 lane_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 tokens_mlp_dim=32,
                 channels_mlp_dim=64):
        super().__init__()

        self._channel = channels_mlp_dim

        self.channel_pre_project = Mlp(in_features=4,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=route_num * lane_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.Mixer = MixerBlock(tokens_mlp_dim, channels_mlp_dim,
                                drop_path_rate)

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, V, D # (B, P=25, V=20, D=12)
        '''
        # only x and x->x' vector, no boundary, no speed limit, no traffic light
        x = x[..., :4]  # (B, P, V, 4)

        B, P, V, _ = x.shape
        """
        mask_v: (B, P, V) -> True if all 4 values are 0 # (점이 없는 경우)
        mask_p: (B, P) -> True if all V values are 0 # (차선이 없는 경우)
        mask_b: (B) -> True if all P values are 0 # (route lanes가 없는 경우)
        """
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1)  # (B, P * V, 4)

        valid_indices = ~mask_b.view(-1)  # (B)
        x = x[valid_indices]  # (B`, P * V, 4)
        """
        token
            - P (route lane 차선 수) * V(차선 당 점의 수) = 25 * 20 = 500 
            - channel_pre_project: -> (B`, P * V, C) where C is channels_mlp_dim
        channel
            - 4 (x, y, dx, dy)
            - token_pre_project: -> (B`, C, T) where T is tokens_mlp_dim
        """
        x = self.channel_pre_project(
            x)  # (B`, P * V, C) where C is channels_mlp_dim
        x = x.permute(0, 2, 1)  # (B`, C, P=25 * V=20)
        x = self.token_pre_project(x)  # (B`, C, T) where T is tokens_mlp_dim
        x = x.permute(0, 2, 1)  # (B`, T, C) # (8, 32, 64)
        x = self.Mixer(x)
        # x.shape: (B`, T, C) # (8, 32, 64)

        x = x.float().mean(dim=1).to(x.dtype)
        # x.shape: (B`, C) # (8, 64)

        x = self.emb_project(self.norm(x))
        # x.shape: (B`, D=192)

        # ★ FIX: 결과 버퍼를 x(dtype/device)에 맞춰 생성
        x_result = torch.zeros((B, x.shape[-1]), device=x.device, dtype=x.dtype)
        x_result[valid_indices] = x  # dtype 충돌 없이 안전
        return_ = x_result.view(B, -1)
        # return_.shape: (B, D=192)
        return return_


from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class DiTReturns:
    integrated_trajectory: torch.Tensor  # (B, Pnn, T, 4)
    control_constraint_diff: torch.Tensor  # (B, Pnn, T, 3)


class DiT(nn.Module):

    def __init__(self,
                 config,
                 sde: SDE,
                 depth,
                 output_dim,
                 hidden_dim=192,
                 heads=6,
                 dropout=0.1,
                 mlp_ratio=4.0,
                 model_type="x_start"):
        super().__init__()
        self.dit_returns = None
        self.config = config

        assert model_type in ["score",
                              "x_start"], f"Unknown model type: {model_type}"
        self.final_hidden_tokens = None
        if self.config.use_feasible:
            self.feasible_projector = FeasibleProjector(hidden_dim, self.config.use_feasible_train, self.config.use_feasible_filter)

        self._model_type = model_type
        self.preproj = Mlp(in_features=output_dim,
                           hidden_features=512,
                           out_features=hidden_dim,
                           act_layer=nn.GELU,
                           drop=0.)
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, heads, dropout, mlp_ratio)
            for i in range(depth)
        ])
        ##################
        # ----- PRAM‑v2 구성요소 추가 -----
        # S/E/R를 저차원으로 정리(RMSNorm 포함) → 쌍곱(SE/ER/RS) → 혼합 MLP → base 모듈레이션(Δs,b,logit g)을 산출.
        self.pram_v2_composer = PRAMV2Composer(
            hidden_dim=hidden_dim,
            adapter_hidden_dim=128,
            composed_hidden_dim=128,
            activation="gelu",
            gate_init_bias=-3.0,
        )
        # 확산 시간(t) 임베딩으로부터 전역(time) 모듈레이션을 생성.
        self.pram_v2_time_mod = PRAMV2TimeModulator(hidden_dim=hidden_dim)

        self.pram_v2_state_token_encoder = PRAMV2StateTokenEncoder(
            hidden_dim=hidden_dim)
        # 블록×경로 토글 스칼라 및 게이트 편향.
        self.pram_v2_block_path_scalars = PRAMV2BlockPathScalars(depth=depth)

        # 9단계: v2 전용 최종 LN/Linear(출력 투영)
        self.pram_v2_final_norm = nn.LayerNorm(hidden_dim)
        self.pram_v2_out_proj = nn.Sequential(
            # nn.LayerNorm(hidden_size),
            nn.Linear(hidden_dim, hidden_dim * 4, bias=True),
            nn.GELU(approximate="tanh"),
            # nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_dim * 4, output_dim, bias=True))
        nn.init.zeros_(
            self.pram_v2_out_proj[-1].weight)  # pram_v2_out_proj의 마지막 Linear
        nn.init.zeros_(self.pram_v2_out_proj[-1].bias)

        # [추가] TimestepEmbedder MLP 초기화 (여기서 1회만)
        #  - 구조: Sequential[ Linear(256→H), SiLU, Linear(H→H) ]
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        if self.t_embedder.mlp[0].bias is not None:
            nn.init.constant_(self.t_embedder.mlp[0].bias, 0.0)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.t_embedder.mlp[2].bias is not None:
            nn.init.constant_(self.t_embedder.mlp[2].bias, 0.0)

        # [선택] 가독성 차원에서 명시(기본값과 동일)
        nn.init.constant_(self.pram_v2_final_norm.weight, 1.0)
        nn.init.constant_(self.pram_v2_final_norm.bias, 0.0)

        # 9단계 마무리 보정용 스칼라 (k^{final}_s, k^{final}_{sh})
        # 실제 보정 함수는 스켈레톤 상태여도 호출부만 준비해 둡니다.
        self.pram_v2_final_scale_scalar = nn.Parameter(torch.tensor(1.0))
        self.pram_v2_final_shift_scalar = nn.Parameter(torch.tensor(1.0))
        #################
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    def preproj_varlen(
            self,
            near_cur_future_norm_xT: torch.Tensor,  # (B, Pnn, F=(1+T)*4)
            near_current_mask: torch.Tensor,  # (B, Pnn) True=pad(무효 에이전트)
    ) -> torch.Tensor:
        """pre‑proj(Mlp)를 **유효 에이전트 토큰**에만 적용하는 varlen 전처리.

        기존 pre‑proj는 (B,Pnn,F) 전체에 적용되어 K≪Pnn일 때 불필요한 연산이 발생합니다.
        이 함수는 마스크 기반 **unpad→pre‑proj→pad‑back**으로 pre‑proj도 K에 비례로 줄입니다.

        Args:
            near_cur_future_norm_xT: 정규화된 (현재+미래) 입력.
                **shape:** (B, Pnn, F)  (F=(1+T)*4, 예: 81*4=324)
            near_current_mask: 키 패딩 마스크(True=pad=무효 에이전트).
                **shape:** (B, Pnn)

        Returns:
            x: pre‑proj 결과.
                **shape:** (B, Pnn, D)  (D=hidden_dim, 예: 192)
                무효 위치는 0으로 채워져 있습니다.
        """
        B, Pnn, F = near_cur_future_norm_xT.shape
        # 유효 토큰만 추출
        # unpad_input은 'True=유효' 마스크를 기대 → 반전 필요
        attention_mask = (~near_current_mask).to(
            torch.bool)  # (B, Pnn), True=유효

        res = unpad_input(near_cur_future_norm_xT, attention_mask)
        # x_unpad: (T, F), indices: (T,), cu: (B+1,), max_seqlen: int
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        # 모든 토큰이 pad인 극단 케이스 방어
        if x_unpad.numel() == 0:
            D_out = self.preproj.fc2.out_features  # timm Mlp의 최종 out_features
            zeros = near_cur_future_norm_xT.new_zeros((B, Pnn, D_out))
            zeros = near_cur_future_norm_xT.new_zeros((B, Pnn, D_out))
            # timm Mlp(fc1/fc2)의 파라미터를 0-스케일로 터치(그래프 포함)
            touch = (self.preproj.fc1.weight.view(-1)[:1].sum() +
                     (self.preproj.fc1.bias.view(-1)[:1].sum()
                      if self.preproj.fc1.bias is not None else 0) +
                     self.preproj.fc2.weight.view(-1)[:1].sum() +
                     (self.preproj.fc2.bias.view(-1)[:1].sum()
                      if self.preproj.fc2.bias is not None else 0)) * 0.0
            return zeros + touch

        # 유효 토큰만 pre‑proj 수행  (T, F) -> (T, D)
        x_unpad = self.preproj(x_unpad)  # (T, D)

        # 배치 모양으로 복원 (pad 위치는 0)
        x = pad_input(x_unpad, indices, B, Pnn)  # (B, Pnn, D)
        return x

    @property
    def model_type(self):
        return self._model_type

    def forward(
        self,
        near_future_norm_xT: torch.Tensor,  # (B, Pnn, T*4)
        diffusion_time: torch.Tensor,  # (B,)
        cross_c: torch.Tensor,  # (B, token_num, D)
        ego_fut_global: torch.Tensor,  # (B, D)
        near_agents_route_lane_emb: torch.Tensor,  # (B, Pnn, D)
        near_cur_future_valid: torch.Tensor,
        # [B, pnn, 1 + future_len] bool
        cross_mask: torch.Tensor,  # (B, token_num) True=pad
        route_known_mask: torch.Tensor,  # (B, Pnn) True=known
        near_class_one_hot: torch.Tensor,
        # (B, Pnn, 3) # 0: 차량, 1: 보행자, 2: 자전거
        near_current_xyyaw: torch.Tensor  # ★ 추가: (B, Pnn, 4)
    ) -> torch.Tensor:
        """
        Forward pass of DiT.
        near_future_norm_xT:  [B, Pnn, T * 4] # (80*4 = 324)
        diffusion_time:  [B,]                 -> Diffusion time uniformly sampled in [eps, 1]
        cross_c: [B, N = token_num, D = 192]
        ego_fut_global: [B, D]   -> Global encoding of the future trajectory of the ego agent.
        near_cur_future_valid: torch.Tensor,  # [B, pnn, 1 + future_len] bool
        near_agents_route_lane_emb, # (B, Pnn, D)
        cross_mask: (B, token_num)
        """
        near_current_valid = near_cur_future_valid[:, :,
                                                   0]  # [B, Pnn] True=유효 에이전트
        near_current_mask = ~near_current_valid  # [B, Pnn] True=무효 에이전트
        B, Pnn, _ = near_future_norm_xT.shape
        # (B, Pnn, 324) -> (B, Pnn, D=192)
        # x = self.preproj(near_future_norm_xT)
        x = self.preproj_varlen(near_future_norm_xT, near_current_mask)

        x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)  # ← 무효 토큰 0 클램프

        # diffusion_time: [B,]
        # t_embedding: (B, D=192)
        t_embedding = self.t_embedder(diffusion_time)
        t_embedding = t_embedding.to(x.dtype)
        ego_fut_global = ego_fut_global.to(x.dtype)  # 방어적 정렬
        ############################
        # ★ 추가: state_token_in 생성 (현재 프레임만 사용)
        B, Pnn, _ = near_future_norm_xT.shape
        # state_token_in: [B,Pnn,D]
        state_token_in = self.pram_v2_state_token_encoder(
            # PRAMV2StateTokenEncoder
            near_cur_norm=near_current_xyyaw.to(x.dtype),  # [B,Pnn,4]
            near_current_mask=near_current_mask  # [B,Pnn]
        )
        # [V2 - START]  ✅ Composer/Time 1회 계산 → 블록별 합성 → v2 전용 포워드
        """
        1) “입력 요약” 만들기 (한 번만 계산 → 모든 블록 재사용)
        2) 쌍곱(상호작용) 특징 만들기 — (SE, ER, RS)
        3) 쌍곱 포함해 한 덩어리로 묶고 z 만들기 — LN → 작은 MLP
        5) (z→) 에이전트별 “base” 모듈레이션 (선형 헤드 3개 + 안전 초기화)
        """
        # composer_out: ComposerOutputs
        #   Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
        composer_out = self.pram_v2_composer(
            state_token_in=state_token_in,  # [B, Pnn, D]
            ego_fut_global=ego_fut_global,  # [B, D] # (배치 단위 0벡터 처리)
            near_agents_route_lane_emb=near_agents_route_lane_emb,
            # [B, Pnn, D]
            route_known_mask=route_known_mask  # [B, Pnn] True=known
        )
        """
        7) 확산 시간 t의 글로벌 모듈레이션

        time_out : TimeModulationOutputs 
            delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
        """
        time_out = self.pram_v2_time_mod(
            t_embedding)  # [B,1,H] 3종 # PRAMV2TimeModulator

        for block_index, block in enumerate(self.blocks):
            """ pram_mods
            Dict[PathName, ModulationTriplet]: 
                {"SA": ModulationTriplet, "FFN":ModulationTriplet, "CA":ModulationTriplet}
            """
            pram_mods = compute_pram_v2_modulations_for_block(
                # Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
                composer_out=composer_out,
                # delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
                time_out=time_out,
                path_scalars=self.
                pram_v2_block_path_scalars,  # PRAMV2BlockPathScalars
                block_index=block_index,
                batch_size=B,
                predicted_neighbor_num=Pnn,
                hidden_dim=x.shape[-1],
            )  # -> {"SA": ModulationTriplet, "FFN": ..., "CA": ...}

            # ★ DiTBlock에 추가한 v2 전용 진입점(스켈레톤; 구현은 이후 단계)
            x = block(
                x=x,  # [B, Pnn, H]
                cross_c=cross_c,  # [B, N_c, H]
                pram_v2_modulations=
                pram_mods,  # Dict[PathName, ModulationTriplet]
                near_current_mask=near_current_mask,  # [B, Pnn]
                cross_mask=cross_mask  # [B, N_c]
            )
            x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)
        self.final_hidden_tokens = x.detach().clone()  # (B, Pnn, H)
        # [V2 - END]
        # --- ✅ PRAM‑v2: 9단계 최종 보정 + 최종 투영(= FinalLayer 완전 대체) ---
        x = apply_pram_v2_final_layer(
            x=x,  # [B, Pnn, H]
            # Δscale_base/shift_base/logit_gate_base (모두 [B, Pnn, H])
            composer_out=composer_out,
            # delta_scale_time/shift_time/logit_gate_time (모두 [B, 1, H])
            time_out=time_out,
            final_norm=self.pram_v2_final_norm,  # LN(H)
            out_proj=self.pram_v2_out_proj,  # Linear(H -> (T)*4)
            final_scalars=(
                self.pram_v2_final_scale_scalar,  # k_final_s
                self.pram_v2_final_shift_scalar,  # k_final_sh
            ),
        )  # -> [B, Pnn, (T)*4]

        # 마스크(무효 토큰) 0 클램프 유지
        x = x.masked_fill(near_current_mask.unsqueeze(-1), 0.0)

        if self._model_type == "score":
            std = self.marginal_prob_std(diffusion_time).float()[:, None,
                                                                 None]  # FP32
            out = (x.float() / (std + 1e-6)).to(
                x.dtype)  # 계산은 FP32, 최종만 원래 dtype
            return out
        elif self._model_type == "x_start":
            # CURRENT DEFAULT OPTION: "x_start"
            # x: (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
            if self.config.use_feasible:
                x_for_feasible_detach: torch.Tensor = x.detach()
                near_current_xyyaw_detach: torch.Tensor = near_current_xyyaw.detach()
                near_cur_future_valid_detach: torch.Tensor = near_cur_future_valid.detach()
                # DiT.forward (model_type == "x_start" 분기 내부)
                if getattr(self.config, "use_current_input", False):
                    # x: (B, Pnn, (1+T)*4) → 이미 현재 프레임 포함
                    diffusion_trajectory = x_for_feasible_detach.reshape(B, Pnn, -1,
                                                     4).contiguous()  # (B,Pnn,1+T,4)
                else:
                    # x: (B, Pnn, T*4) → 현재 프레임을 앞에 붙여서 1+T로 맞춤
                    diffusion_trajectory = torch.cat([
                        near_current_xyyaw_detach.unsqueeze(2),
                        x_for_feasible_detach.reshape(B, Pnn, -1, 4)
                    ],
                                                     dim=2).contiguous()  # (B,Pnn,1+T,4)
                self._feasible_projection(diffusion_trajectory,
                                          near_class_one_hot,
                                          near_cur_future_valid_detach)
            return x  # (B, Pnn, T * 4) or (B, Pnn, (1+T) * 4)
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def _feasible_projection(
        self,
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+T, 4)
        near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
        near_cur_future_valid: torch.Tensor  # (B, Pnn, 1+T) bool
    ):
        # diffusion_trajectory: (B, Pnn, 1+T, 4)
        # (B, Pnn, 1+T, 4)
        unnorm_diffusion_trajectory = self.config.state_normalizer.inverse(
            diffusion_trajectory)
        unnorm_near_current_state = unnorm_diffusion_trajectory[:, :,
                                                                0, :]  # (B, Pnn, 4)

        unnorm_cur_future_control = self.feasible_projector.savgol_filter_for_control(
            unnorm_diffusion_trajectory,  # (B, Pnn, 1+T, 4),
            near_cur_future_valid,  # (B, Pnn, 1+T) bool
        )  # (B, Pnn, 1+T, 3)
        # cur_future_seg_body_control: (B, Pnn, T, 3) #  v_x^b, v_y^b, ω # 각 시점 몸체 좌표계 기준 속도 + 세계 좌표계 기준 요레이트
        unnorm_cur_future_seg_body_control = self.feasible_projector.compute_midpoint_controls(
            unnorm_diffusion_trajectory, unnorm_cur_future_control,
            near_cur_future_valid)

        temp_dict = {
            "cur_future_seg_body_control": unnorm_cur_future_seg_body_control
        }
        temp_dict = self.config.observation_normalizer(temp_dict)
        cur_future_seg_body_control = temp_dict[
            "cur_future_seg_body_control"]  # (B, Pnn, T, 3)
        # cur_future_seg_body_control: (B, Pnn, T, 3)
        cur_future_seg_body_control = self.feasible_projector(
            near_cur_future_valid,  # (B, Pnn, 1+T)
            diffusion_trajectory,  # (B, Pnn, 1+T, 4)
            cur_future_seg_body_control,  # (B, Pnn, T, 3)
            self.final_hidden_tokens,  # (B, Pnn, H)
        )
        temp_dict = {"cur_future_seg_body_control": cur_future_seg_body_control}
        temp_dict = self.config.observation_normalizer.inverse(temp_dict)
        unnorm_cur_future_seg_body_control = temp_dict[
            "cur_future_seg_body_control"]  # (B, Pnn, T, 3)
        (unnorm_integrated_trajectory, unnorm_control_constraint_diff
        ) = self.feasible_projector.filter_and_integrate(
            unnorm_near_current_state,  # (B, Pnn, 4)
            near_cur_future_valid,  # (B, Pnn, 1+T) bool  ← 시점별 마스크
            unnorm_cur_future_seg_body_control,  # (B, Pnn, T, 3)
            near_class_one_hot,
            # (B, Pnn, 3) # 0: vehicle, 1: pedestrian, 2: bicycle
        )  # (B, Pnn, T, 4)
        integrated_trajectory = self.config.state_normalizer(
            unnorm_integrated_trajectory)  # (B, Pnn, T, 4)
        temp_dict = {"control_constraint_diff": unnorm_control_constraint_diff}
        temp_dict = self.config.observation_normalizer(temp_dict)
        control_constraint_diff = temp_dict[
            "control_constraint_diff"]  # (B, Pnn, T, 3)
        self.dit_returns = DiTReturns(
            integrated_trajectory=integrated_trajectory,
            control_constraint_diff=control_constraint_diff)
