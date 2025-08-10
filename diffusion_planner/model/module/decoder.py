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


class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        dpr = config.decoder_drop_path_rate
        self._predicted_neighbor_num = config.predicted_neighbor_num
        self._future_len = config.future_len
        self._sde = VPSDE_linear()

        self.dit = DiT(
            sde=self._sde,
            route_encoder=RouteEncoder(
                config.route_num,
                config.lane_len,
                drop_path_rate=config.encoder_drop_path_rate,
                hidden_dim=config.hidden_dim),
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

                    [training-only] "sampled_trajectories": sampled current-future ego & neighbor states,        [B, P, 1 + V_future, 4]
                    [training-only] "diffusion_time": timestep of diffusion process $t \in [0, 1]$,              [B]
                    ...
                }

        Returns:
            decoder_outputs: Dict
                {
                    ...
                    [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
                    [inference-only] "prediction": Predicted future states, [B, P, V_future, 4]
                    ...
                }

        """
        # Extract ego & neighbor current states
        ego_current = inputs['ego_current_state'][:, None, :4]  # [B, 1, 4]
        neighbors_current = inputs[
            "neighbor_agents_past"][:, :self._predicted_neighbor_num,
                                    -1, :4]  # [B, pnn, 4]
        neighbor_current_mask = torch.sum(
            torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0  # [B, pnn]
        inputs["neighbor_current_mask"] = neighbor_current_mask

        current_states = torch.cat([ego_current, neighbors_current],
                                   dim=1)  # [B, 1+Pnn, 4]

        B, one_P, _ = current_states.shape
        assert one_P == (1 + self._predicted_neighbor_num)

        # Extract context encoding
        ego_neighbor_encoding = encoder_outputs['encoding']  #  (B, 107 + future_len, 192)
        ego_future_global = encoder_outputs.get('ego_future_global')
        route_lanes = inputs['route_lanes']  # (B, 25, 20, 12)

        if self.training:
            sampled_trajectories = inputs['sampled_trajectories'].reshape(
                B, one_P,
                -1)  # [B, 1+ Pnn, 1 + T, 4] -> [B, one_P, (1 + T) * 4]
            diffusion_time = inputs['diffusion_time']

            return {
                "score":
                    self.dit(sampled_trajectories, diffusion_time,
                             ego_neighbor_encoding, route_lanes,
                             neighbor_current_mask, ego_future_global).reshape(B, one_P, -1, 4)
            }
        else:
            # [B, 1 + predicted_neighbor_num, (1 + V_future) * 4]
            xT = torch.cat([
                current_states[:, :, None],
                torch.randn(B, one_P, self._future_len, 4).to(
                    current_states.device) * 0.5
            ],
                           dim=2).reshape(B, one_P, -1)

            def initial_state_constraint(xt, t, step):
                xt = xt.reshape(B, one_P, -1, 4)
                xt[:, :, 0, :] = current_states
                return xt.reshape(B, one_P, -1)

            x0 = dpm_sampler(
                self.dit,
                xT,
                other_model_params={
                    "cross_c": ego_neighbor_encoding,
                    "route_lanes": route_lanes,
                    "neighbor_current_mask": neighbor_current_mask,
                    "global_cond": ego_future_global
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
                            "cross_c": ego_neighbor_encoding,
                            "route_lanes": route_lanes,
                            "neighbor_current_mask": neighbor_current_mask,
                            "global_cond": ego_future_global
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
            x0 = self._state_normalizer.inverse(x0.reshape(B, one_P, -1,
                                                           4))[:, :, 1:]

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
        x = x[..., :4] # (B, P, V, 4)

        B, P, V, _ = x.shape
        """
        mask_v: (B, P, V) -> True if all 4 values are 0 # (점이 없는 경우)
        mask_p: (B, P) -> True if all V values are 0 # (차선이 없는 경우)
        mask_b: (B) -> True if all P values are 0 # (route lanes가 없는 경우)
        """
        mask_v = torch.sum(torch.ne(x[..., :4], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        mask_b = torch.sum(~mask_p, dim=-1) == 0
        x = x.view(B, P * V, -1) # (B, P * V, 4)

        valid_indices = ~mask_b.view(-1) # (B)
        x = x[valid_indices] # (B`, P * V, 4)
        """
        token
            - P (route lane 차선 수) * V(차선 당 점의 수) = 25 * 20 = 500 
            - channel_pre_project: -> (B`, P * V, C) where C is channels_mlp_dim
        channel
            - 4 (x, y, dx, dy)
            - token_pre_project: -> (B`, C, T) where T is tokens_mlp_dim
        """
        x = self.channel_pre_project(x) # (B`, P * V, C) where C is channels_mlp_dim
        x = x.permute(0, 2, 1)# (B`, C, P=25 * V=20)
        x = self.token_pre_project(x) # (B`, C, T) where T is tokens_mlp_dim
        x = x.permute(0, 2, 1) # (B`, T, C) # (8, 32, 64)
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
                 route_encoder: nn.Module,
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
        self.route_encoder = route_encoder
        self.agent_embedding = nn.Embedding(2, hidden_dim)
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

    def forward(self, x, t, cross_c, route_lanes, neighbor_current_mask, global_cond=None):
        """
        Forward pass of DiT.
        x:  [B, 1+ Pnn, (1 + T) * 4] # (81*4 = 324)
        t:  [B,]                 -> Diffusion time uniformly sampled in [eps, 1]
        cross_c: [B, one_Pnn, D] = [B, N = 107 + future_len, D = 192]
        route_lanes: (B, 25, 20, 12)
        neighbor_current_mask: [B, Pnn]
        global_cond: (B, D)
        """
        B, one_Pnn, _ = x.shape
        # (B, 11, 324) -> (B, 11, D=192)
        x = self.preproj(x)

        a = self.agent_embedding.weight[0][None, :] # (1, D = 192)
        b = self.agent_embedding.weight[1][None, :] # (1, D)
        b_expanded = b.expand(one_Pnn - 1, -1) # (Pnn, D)

        x_embedding = torch.cat([
            a,
            b_expanded,
        ], dim=0)  # (one_Pnn, D)
        x_embedding = x_embedding[None, :, :].expand(B, -1,
                                                     -1)  # (B, one_Pnn, D)
        # [B, one_Pnn, D] + (B, one_Pnn, D)
        x = x + x_embedding
        # route_lanes: (B, 25, 20, 12)
        # route_encoding: (B, D=192)
        route_encoding = self.route_encoder(route_lanes)
        y = route_encoding
        t_embedding = self.t_embedder(t)
        y = y + t_embedding
        if global_cond is not None:
            y = y + global_cond

        all_current_mask_for_attn = torch.zeros((B, one_Pnn), dtype=torch.bool, device=x.device)
        all_current_mask_for_attn[:, 1:] = neighbor_current_mask

        for block in self.blocks:
            """
            Input shapes:
            x: (B, one_Pnn, D=192)
            cross_c: (B, N=107, D=192)
            y: (B, D=192)
            all_current_mask_for_attn: (B, one_Pnn)
            """
            x = block(x, cross_c, y, all_current_mask_for_attn)
        # output: x: (B, one_Pnn, D=192)
        # y: (B, D=192)
        x = self.final_layer(x, y)
        # x.shape: (B, one_Pnn, (1 + T) * 4)

        if self._model_type == "score":
            return x / (self.marginal_prob_std(t)[:, None, None] + 1e-6)
        elif self._model_type == "x_start":
            # x: (B, one_Pnn, (1 + T) * 4)
            return x
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
