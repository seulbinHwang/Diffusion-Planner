import math
import torch
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.diffusion_utils.sampling import dpm_sampler
from diffusion_planner.model.diffusion_utils.sde import SDE, VPSDE_linear
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.model.module.mixer import MixerBlock
from diffusion_planner.model.module.dit import TimestepEmbedder, DiTBlock, FinalLayer
from diffusion_planner.loss import _require_finite


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            # route_encoder=RouteEncoder(
            #     config.route_num,
            #     config.lane_len,
            #     drop_path_rate=config.encoder_drop_path_rate,
            #     hidden_dim=config.hidden_dim),
            depth=config.decoder_depth,
            output_dim=(config.future_len + 1) * 4,  # x, y, cos, sin
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
                    [inference-only] "prediction": Predicted future states, [B, P, future_len, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        near_current = inputs["neighbor_agents_past"][:, :self.
                                                      _predicted_neighbor_num,
                                                      -1, :4]  # [B, pnn, 4]
        near_current_mask = torch.sum(torch.ne(near_current[..., :4], 0),
                                      dim=-1) == 0  # [B, pnn]
        inputs["near_current_mask"] = near_current_mask

        B, Pnn, _ = near_current.shape
        assert Pnn == (self._predicted_neighbor_num)
        assert near_current_mask.shape[1] == Pnn

        # Extract context encoding
        scene_encoding_token = encoder_outputs[
            'encoding']  #  (B, token_num, hidden_dim)
        scene_encoding_token_mask = encoder_outputs[
            'encoding_mask']  # (B, token_num) bool
        ego_fut_global = encoder_outputs["ego_fut_global"]  # (B, hidden_dim)
        assert ego_fut_global.shape == (B, scene_encoding_token.shape[-1])

        if self.training:
            near_cur_future_norm_xT = inputs['near_cur_future_norm_xT'].reshape(
                B, Pnn, -1)  # [B, Pnn, 1 + T, 4] -> [B, Pnn, (1 + T) * 4]
            diffusion_time = inputs['diffusion_time']
            # (B, Pnn, (1 + T) , 4)
            score = self.dit(
                near_cur_future_norm_xT,  # ( B, Pnn, (1 + T) * 4 )
                diffusion_time,  # (B)
                scene_encoding_token,  # (B, token_num, hidden_dim)
                ego_fut_global,  # (B, hidden_dim)
                near_current_mask,  # (B, Pnn),
                scene_encoding_token_mask  # (B, token_num) bool
            )
            _require_finite("decoder_dit_output", score)
            return {"score": score.reshape(B, Pnn, -1, 4)}  #  (B, Pnn, (1 + T) , 4)
        else:
            # [B, Pnn, (1 + future_len) * 4]
            xT = torch.cat(
                [
                    near_current[:, :, None],  # (B, Pnn, 1, 4)
                    torch.randn(B, Pnn, self._future_len,
                                4).to(  # (B, Pnn, T, 4)
                                    near_current.device) * 0.5
                ],
                dim=2).reshape(B, Pnn, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, Pnn, -1, 4)
                xt[:, :, 0, :] = near_current
                return xt.reshape(B, Pnn, -1)

            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": scene_encoding_token,
                    "ego_fut_global": ego_fut_global,
                    "near_current_mask": near_current_mask,
                    "cross_mask": scene_encoding_token_mask,
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
                            "cross_c": scene_encoding_token,
                            "ego_fut_global": ego_fut_global,
                            "near_current_mask": near_current_mask,
                            "cross_mask": scene_encoding_token_mask,
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
            x0 = self._state_normalizer.inverse(x0.reshape(
                B, Pnn, -1, 4))  # (B, Pnn, 1 + T, 4)
            x0 = x0[:, :, 1:]  # (B, Pnn, T, 4)

            return {"prediction": x0}


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

        x = torch.mean(x, dim=1)
        # x.shape: (B`, C) # (8, 64)

        x = self.emb_project(self.norm(x))
        # x.shape: (B`, D=192)

        x_result = torch.zeros((B, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        return_ = x_result.view(B, -1)
        # return_.shape: (B, D=192)
        return return_


class DiT(nn.Module):

    def __init__(self,
                 sde: SDE,
                 depth,
                 output_dim,
                 hidden_dim=192,
                 heads=6,
                 dropout=0.1,
                 mlp_ratio=4.0,
                 model_type="x_start"):
        super().__init__()

        assert model_type in ["score",
                              "x_start"], f"Unknown model type: {model_type}"
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
        self.final_layer = FinalLayer(hidden_dim, output_dim)
        self._sde = sde
        self.marginal_prob_std = self._sde.marginal_prob_std

    @property
    def model_type(self):
        return self._model_type

    def forward(self, near_cur_future_norm_xT, diffusion_time, cross_c,
                ego_fut_global, near_current_mask, cross_mask):
        """
        Forward pass of DiT.
        near_cur_future_norm_xT:  [B, Pnn, (1 + T) * 4] # (81*4 = 324)
        diffusion_time:  [B,]                 -> Diffusion time uniformly sampled in [eps, 1]
        cross_c: [B, N = token_num, D = 192]
        ego_fut_global: [B, D]   -> Global encoding of the future trajectory of the ego agent.
        near_current_mask: [B, Pnn]
        cross_mask: (B, token_num)
        """
        B, Pnn, in_dim = near_cur_future_norm_xT.shape  # in_dim = (1 + T) * 4
        _, N, D = cross_c.shape  # D = hidden_dim

        # diffusion_time: [B,] -> t_embedding: (B, D=192)
        # y: (B, D=192) = ego_fut_global + t_embedding
        t_embedding = self.t_embedder(diffusion_time)
        y = ego_fut_global + t_embedding

        # ----------------------------------------------------------------------
        # [핵심] 배치별로 유효 토큰(False)이 앞으로 오도록 정렬하여, 유효 길이까지만 잘라 연산
        #  - near 쿼리(Pnn 축): 유효 길이 Lp
        #  - cross K/V(N 축):  버킷 내 최대 유효 길이 Lc_max
        # ----------------------------------------------------------------------

        # (1) near(이웃) 토큰 정렬: 유효(False)=0, 무효(True)=1 이므로 argsort로 유효를 앞쪽으로
        # near_order: (B, Pnn), mask_near_sorted: (B, Pnn)
        near_order = torch.argsort(near_current_mask.to(torch.int), dim=1)  # (B, Pnn)
        idx_feat_near = near_order.unsqueeze(-1).expand(B, Pnn, in_dim)     # (B, Pnn, in_dim)
        near_sorted = torch.gather(near_cur_future_norm_xT, dim=1, index=idx_feat_near)  # (B, Pnn, in_dim)
        mask_near_sorted = torch.gather(near_current_mask, dim=1, index=near_order)     # (B, Pnn)

        # (2) cross(장면 컨텍스트) 토큰 정렬: 동일하게 유효를 앞쪽으로
        # cross_order: (B, N), mask_cross_sorted: (B, N)
        cross_order = torch.argsort(cross_mask.to(torch.int), dim=1)              # (B, N)
        idx_feat_cross = cross_order.unsqueeze(-1).expand(B, N, D)                # (B, N, D)
        cross_sorted = torch.gather(cross_c, dim=1, index=idx_feat_cross)         # (B, N, D)
        mask_cross_sorted = torch.gather(cross_mask, dim=1, index=cross_order)    # (B, N)

        # (3) 배치별 유효 길이 계산
        # Lp: (B,) near 유효 길이,  Lc: (B,) cross 유효 길이
        Lp = (~mask_near_sorted).sum(dim=1)  # (B,)
        Lc = (~mask_cross_sorted).sum(dim=1) # (B,)

        # (4) 정렬 상태의 출력 버퍼 준비 (초기값 0)
        # x_sorted_out: (B, Pnn, out_dim)  # out_dim = (1 + T) * 4
        out_dim = in_dim
        x_sorted_out = near_sorted.new_zeros(B, Pnn, out_dim)

        # (5) near 유효 길이(Lp)가 같은 샘플끼리 버킷으로 묶어 처리 → Self-Attn 길이 단축
        # unique_Lp: (K,), inv_Lp: (B,)
        unique_Lp, inv_Lp = torch.unique(Lp, sorted=True, return_inverse=True)
        for k in range(unique_Lp.numel()):
            Lp_val = int(unique_Lp[k].item())
            if Lp_val == 0:
                # 이 버킷의 샘플들은 near 유효 토큰이 0개 → 연산 스킵
                continue

            # 이 버킷에 속한 배치 인덱스
            # batch_idx: (B_k,)
            batch_mask = (inv_Lp == k)
            if not batch_mask.any():
                continue
            batch_idx = torch.nonzero(batch_mask, as_tuple=False).squeeze(-1)  # (B_k,)
            Bk = batch_idx.numel()

            # (5-1) near 쿼리 유효 구간만 슬라이스
            # near_chunk: (B_k, Lp_val, in_dim)
            near_chunk = near_sorted[batch_idx, :Lp_val, :]

            # (5-2) cross K/V도 버킷 내 최대 유효 길이 Lc_max까지만 슬라이스
            # cross_chunk: (B_k, Lc_max, D) (Lc_max=0이면 (B_k, 0, D))
            # cross_mask_chunk: (B_k, Lc_max)
            Lc_batch = Lc[batch_idx]
            Lc_max = int(Lc_batch.max().item())
            if Lc_max > 0:
                cross_chunk = cross_sorted[batch_idx, :Lc_max, :]
                cross_mask_chunk = mask_cross_sorted[batch_idx, :Lc_max]
            else:
                cross_chunk = cross_sorted.new_zeros(Bk, 0, D)
                cross_mask_chunk = mask_cross_sorted.new_ones(Bk, 0)

            # (5-3) 사전 투영: (B_k, Lp_val, in_dim) -> (B_k, Lp_val, D)
            x = self.preproj(near_chunk)

            # (5-4) 블록 스택 통과
            # attn_mask_near_all_false: (B_k, Lp_val)  # 모두 False(패딩 없음)
            attn_mask_near_all_false = torch.zeros(Bk, Lp_val, dtype=torch.bool, device=x.device)
            for block in self.blocks:
                """
                Input shapes:
                x: (B_k, Lp_val, D)
                cross_chunk: (B_k, Lc_max, D)
                y[batch_idx]: (B_k, D)
                attn_mask_near_all_false: (B_k, Lp_val)
                cross_mask_chunk: (B_k, Lc_max)
                """
                x = block(x, cross_chunk, y[batch_idx], attn_mask_near_all_false, cross_mask_chunk)

            # (5-5) 최종 투영: (B_k, Lp_val, D) -> (B_k, Lp_val, out_dim)
            x = self.final_layer(x, y[batch_idx])

            # (5-6) 정렬 상태 출력 버퍼의 앞쪽 Lp_val 위치에만 써넣기
            x_sorted_out[batch_idx, :Lp_val, :] = x  # (나머지 패딩 위치는 0 유지)

        # (6) 정렬을 되돌려 원래 P 슬롯 순서로 복원
        # inv_near_order: (B, Pnn), idx_feat_inv: (B, Pnn, out_dim)
        inv_near_order = torch.empty_like(near_order)
        arange_P = torch.arange(Pnn, device=near_order.device).unsqueeze(0).expand(B, Pnn)
        inv_near_order.scatter_(1, near_order, arange_P)  # 역순열 구축
        idx_feat_inv = inv_near_order.unsqueeze(-1).expand(B, Pnn, out_dim)
        x_out_original_order = torch.gather(x_sorted_out, dim=1, index=idx_feat_inv)  # (B, Pnn, out_dim)

        # (7) 안전하게 무효 토큰은 최종 출력에서도 0으로 보장
        x_out_original_order = x_out_original_order.masked_fill(near_current_mask.unsqueeze(-1), 0.0)

        # (8) 타입에 따른 반환 (원본과 동일)
        if self._model_type == "score":
            return x_out_original_order / (self.marginal_prob_std(diffusion_time)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            # CURRENT DEFAULT OPTION: "x_start"
            # x: (B, Pnn, (1 + T) * 4)
            return x_out_original_order
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
