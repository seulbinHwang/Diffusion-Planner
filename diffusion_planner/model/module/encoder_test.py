import torch
import torch.nn as nn
from typing import Tuple
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.module.mixer import MixerBlock

import torch
from typing import Optional


def timegrid_past_3d(dt: float,
                     total_steps: int,
                     B: int,
                     agents_num: int,
                     *,
                     device: Optional[torch.device] = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    반환 모양: (B, agents_num, total_steps)
    마지막 축(시간축): [-dt*(N-1), ..., -dt*1, 0.0]
    항상 독립 메모리 텐서(복사본)를 반환합니다.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1")
    if B <= 0 or agents_num <= 0:
        raise ValueError("B and agents_num must be >= 1")

    # (N,) = [N-1, ..., 0]
    idx = torch.arange(total_steps - 1, -1, -1, device=device, dtype=dtype)
    base = -float(dt) * idx
    base[-1] = torch.tensor(0.0, dtype=dtype, device=device)  # -0.0 → 0.0

    # (1,1,N) → (B,A,N) 확장 후 clone()으로 복사본 보장
    return base.view(1, 1, total_steps).expand(B, agents_num,
                                               total_steps).clone()


def timegrid_future_2d(dt: float,
                       total_steps: int,
                       B: int,
                       *,
                       device: Optional[torch.device] = None,
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    반환 모양: (B, agents_num, total_steps)
    마지막 축(시간축): [dt, 2*dt, ..., N*dt]
    항상 독립 메모리 텐서(복사본)를 반환합니다.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1")
    if B <= 0:
        raise ValueError("B and agents_num must be >= 1")

    # (N,) = [1, 2, ..., N]
    idx = torch.arange(1, total_steps + 1, device=device, dtype=dtype)
    base = float(dt) * idx  # [dt, 2dt, ..., N*dt]

    # (1,N) → (B, N) 확장 후 clone()으로 복사본 보장
    return base.view(1, total_steps).expand(B, total_steps).clone()


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.token_num = 1 + config.agent_num + config.static_objects_num + config.lane_num
        self.agents_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth)
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim)
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth)

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device)

        # position embedding encode x, y, cos, sin, type (ego, neighbor, static, lane)
        self.pos_emb = nn.Linear(8, config.hidden_dim)

    def forward(self, inputs):
        encoder_outputs = {}
        # ego
        ego_past = inputs["ego_agent_past"]  # (B, V=21, D=11) -> (B, 1, V, D)
        ego_past = ego_past.unsqueeze(1)  # Add a dimension for P
        # agents
        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]
        encoding_neighbors, neighbors_mask, neighbor_pos = self.agents_encoder(
            ego_past, neighbors)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit)

        encoding_input = torch.cat(
            [encoding_neighbors, encoding_static, encoding_lanes], dim=1)
        encoding_pos = torch.cat([neighbor_pos, static_pos, lane_pos],
                                 dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([neighbors_mask, static_mask, lanes_mask],
                                  dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim),
                                          device=encoding_pos.device)
        encoding_pos_result[
            ~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(
            B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num))

        return encoder_outputs


class SelfAttentionBlock(nn.Module):

    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU,
                       drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(
            self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):

    def __init__(self,
                 time_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128):
        super().__init__()
        self.time_gap = 0.1
        self.time_min = -2.0
        self.time_max = 8.0
        self.num_fourier_frequencies = 4
        num_fourier_dim = 2 * self.num_fourier_frequencies + 1  # 2K + 1

        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8 + num_fourier_dim + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=time_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def _get_agents_past_cur_mask(
            self, agents_past_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: agents_past_current (B, agents_num, time_len, 8 + 2K + 1)

        Output:
            agents_past_cur_off_p_mask: (B, agents_num, time_len)
            agents_past_cur_off_mask: (B, agents_num)
        """
        agents_past_current_is_not_zero = torch.ne(
            agents_past_current[..., :8], 0)  # (B, agents_num, time_len, 8)
        agents_past_current_not_zero_num = torch.sum(
            agents_past_current_is_not_zero,
            dim=-1).to(agents_past_current.device)  # (B, agents_num, time_len)
        # (B, agents_num, time_len)
        agents_past_cur_off_p_mask = agents_past_current_not_zero_num == 0
        agents_past_cur_off_mask = torch.sum(~agents_past_cur_off_p_mask,
                                             dim=-1) == 0  # (B, agents_num)

        return agents_past_cur_off_p_mask, agents_past_cur_off_mask

    def _reverse_agents_past_cur_mask(
        self, agents_past_cur_off_p_mask: torch.Tensor,
        agents_past_cur_off_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
            agents_past_cur_off_p_mask : (B, agents_num, time_len)
            agents_past_cur_off_mask: (B, agents_num)
        Output:
            agents_past_cur_on_p_mask: (B, agents_num, time_len, 1)
            agents_past_cur_on_mask: (B * agents_num)

        """
        agents_past_cur_on_p_mask = (
            ~agents_past_cur_off_p_mask).float().unsqueeze(
                -1)  # (B, agents_num, time_len, 1)
        agents_past_cur_on_mask = ~agents_past_cur_off_mask.view(
            -1)  # (B * agents_num)
        return agents_past_cur_on_p_mask, agents_past_cur_on_mask

    def _get_ego_future_mask(
            self,
            ego_future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: ego_future (B, future_len,  8 + 2K + 1)

        Output:
            ego_future_off_p_mask: (B, future_len)
            ego_future_off_mask: (B)
        """
        ego_future_is_not_zero = torch.ne(ego_future[..., :8],
                                          0)  # (B, future_len, 8)
        ego_future_not_zero_num = torch.sum(ego_future_is_not_zero, dim=-1).to(
            ego_future.device)  # (B, future_len)
        # (B, future_len)
        ego_future_off_p_mask = ego_future_not_zero_num == 0  # (B , future_len)
        ego_future_off_mask = torch.sum(~ego_future_off_p_mask,
                                        dim=-1) == 0  # (B)
        return ego_future_off_p_mask, ego_future_off_mask

    def _get_on_agents_past_cur(
        self,
        agents_past_current: torch.Tensor,  # (B, agents_num, time_len, 8 + 2K + 1 )
        agents_past_cur_on_p_mask: torch.
        Tensor,  # (B, agents_num, time_len, 1) # float
        agents_past_cur_on_mask: torch.Tensor  # (B * agents_num)
    ) -> torch.Tensor:  # (agents_past_cur_on_num, time_len, 8 + 2K + 1 "+1")
        B, agents_num, time_len, _ = agents_past_current.shape
        # agents_past_current: (B , agents_num, time_len, 8 + 2K + 1 "+1")
        agents_past_current = torch.cat(
            [agents_past_current, agents_past_cur_on_p_mask], dim=-1)
        # agents_past_current: (B * agents_num, time_len, 8 + 2K + 1 "+1")
        agents_past_current = agents_past_current.view(B * agents_num, time_len,
                                                       -1)
        """
        agents_past_cur_on_mask 에서, True의 개수 = 
        (B * agents_num) 개 중에서 agents_past_cur_on_num 개
        """
        # on_agents_past_cur: (agents_past_cur_on_num, time_len, 8 + 2K + 1 "+1")
        on_agents_past_cur = agents_past_current[agents_past_cur_on_mask]
        return on_agents_past_cur

    def _get_on_ego_future(
        self,
        ego_future: torch.Tensor,  # (B, future_len,  8 + 2K + 1)
        ego_future_on_p_mask: torch.Tensor,  # (B, future_len, 1)
        ego_future_on_mask: torch.Tensor  # (B)
    ) -> torch.Tensor:
        ego_future = torch.cat([ego_future, ego_future_on_p_mask],
                               dim=-1)  # (B, future_len,  8 + 2K + 1 + "1")
        # (ego_future_on_num, future_len, 10 + 2k)
        on_ego_future = ego_future[ego_future_on_mask]
        return on_ego_future  # (ego_future_on_num, future_len, 10 + 2k)

    def _concat_past_cur_and_future(
        self,
        on_agents_past_cur: torch.
        Tensor,  # (agents_past_cur_on_num, time_len, 10 + 2k)
        on_ego_future: torch.Tensor  # (ego_future_on_num, future_len, 10 + 2k)
    ) -> torch.Tensor:  # (on_all_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len, 10 + 2k)
        # (agents_past_cur_on_num * time_len, 10+2k)
        on_agents_past_cur = on_agents_past_cur.view(
            -1, on_agents_past_cur.shape[-1])
        # (ego_future_on_num * future_len, 10 + 2k)
        on_ego_future = on_ego_future.view(-1, on_ego_future.shape[-1])
        # (on_all_num = agents_past_cur_on_num * time_len + on_ego_future_on_num * future_len, 10 + 2k)
        on_all = torch.cat([on_agents_past_cur, on_ego_future], dim=0)
        return on_all

    def _encode_time_with_fourier_features(self,
                                           t_sec: torch.Tensor) -> torch.Tensor:
        """시간 값을 Fourier(2K) + 중심화된 시간 스칼라(1)로 인코딩한다.

        Args:
            t_sec (torch.Tensor):
                - shape: (...,)   # 초 단위 시간

        Returns:
            torch.Tensor:
                - shape: (..., 2K + 1)
                - 구성: [cos(·), sin(·),  t_hat]
                    * 앞쪽 2K채널: Fourier 성분
                    * 마지막 1채널: 중심화 시간 스칼라
        """
        # 정규화된 시간: (...,)
        t_norm = (t_sec - self.time_min) / (self.time_max - self.time_min)
        t_norm = t_norm.clamp(0.0, 1.0)

        # 주파수 인덱스: (K,)
        k_indices = torch.arange(1,
                                 self.num_fourier_frequencies + 1,
                                 device=t_sec.device,
                                 dtype=t_sec.dtype)  # (K,)

        # 각 주파수 각도: (..., K)
        angles = t_norm.unsqueeze(-1) * k_indices * torch.pi  # (..., K)

        # Fourier 성분: (..., 2K)
        fourier = torch.cat(
            [torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., 2K)

        # 중심화 시간 스칼라: (..., 1),  t_hat = 2 * t_norm - 1
        t_scalar = (2.0 * t_norm - 1.0).unsqueeze(-1)  # (..., 1)

        # 최종: (..., 2K + 1)  [cos | sin | t_hat]
        return torch.cat([fourier, t_scalar], dim=-1)

    def forward(self, ego_past_current, npc_past_current, ego_future):
        '''
        ego_past_current: B, 1, time_len, D_11 (x, y, cos, sin, vx, vy, w, l, type(3))
        npc_past_current: B, agent_num, time_len, D_11 (x, y, cos, sin, vx, vy, w, l, type(3))
        ego_future: B, future_len=80, D_11
        '''
        # (B, agents_num=1+agent_num, time_len, D)
        agents_past_current = torch.cat([ego_past_current, npc_past_current],
                                        dim=1)
        B, agents_num, time_len, _ = agents_past_current.shape
        future_len = ego_future.shape[1]
        agents_type = agents_past_current[:, :, -1, 8:]  # (B, agents_num, 3)
        ego_type = agents_type[:, 0, :].clone()  # (B, 3)
        # (B, agents_num, time_len, d_8)
        agents_past_current = agents_past_current[..., :8]
        ### add timestep
        # agents_past_current_timestep: (B, agents_num, time_len)
        # [-2.0, -1.9, ..., 0.0]
        agents_past_current_timestep = timegrid_past_3d(
            dt=self.time_gap,
            total_steps=time_len,
            B=B,
            agents_num=agents_num,
            device=agents_past_current.device,
            dtype=agents_past_current.dtype)
        agent_past_current_time_fourier = self._encode_time_with_fourier_features(
            agents_past_current_timestep)  # (B, agents_num, time_len, 2K + 1)
        agents_past_current = torch.cat(
            [agents_past_current, agent_past_current_time_fourier],
            dim=-1)  # (B, agents_num, time_len, 8 + 2K + 1)
        ############
        # TODO: pos 구하기
        pos = agents_past_current[:, :, -1, :8].clone(
        )  # x, y, cos, sin # (B, agents_num, 8)
        # neighbor: [0,1,0,0]
        pos[..., -4:] = 0.0
        pos[..., -3] = 1.0
        # ego: [1, 0, 0, 0]
        pos[:, 0, -4:] = 0.0
        pos[:, 0, -4] = 1.0
        ############
        """
        agents_past_cur_off_p_mask: (B, agents_num, time_len)
        agents_past_cur_off_mask: (B, agents_num)
        -----------
        agents_past_cur_on_p_mask: (B, agents_num, time_len, 1) # float
        agents_past_cur_on_mask: (B * agents_num)
        """
        (agents_past_cur_off_p_mask, agents_past_cur_off_mask
        ) = self._get_agents_past_cur_mask(agents_past_current)
        (agents_past_cur_on_p_mask,
         agents_past_cur_on_mask) = self._reverse_agents_past_cur_mask(
             agents_past_cur_off_p_mask, agents_past_cur_off_mask)

        # on_agents_past_cur: (agents_past_cur_on_num, time_len, 8 + 2K + 1 "+ 1" )
        on_agents_past_cur = self._get_on_agents_past_cur(
            agents_past_current, agents_past_cur_on_p_mask,
            agents_past_cur_on_mask)

        #################################
        # ego_future: (B, future_len, 8)
        ego_future = ego_future[..., :8]
        # ego_future_timestep: (B, future_len)
        ego_future_timestep = timegrid_future_2d(
            dt=self.time_gap,
            total_steps=ego_future.shape[1],  # future_len
            B=B,
            device=ego_future.device,
            dtype=ego_future.dtype)
        ego_future_time_fourier = self._encode_time_with_fourier_features(
            ego_future_timestep)  # (B, future_len, 2K + 1)
        ego_future = torch.cat([ego_future, ego_future_time_fourier],
                               dim=-1)  # (B, future_len, 8 + 2K + 1)
        # TODO: pos ego future 구현하기
        """
        ego_future_off_p_mask: (B, future_len)
        ego_future_off_mask: (B)
        --------
        ego_future_on_p_mask: (B, future_len, 1)
        ego_future_on_mask: (B)
        """
        ego_future_off_p_mask, ego_future_off_mask = self._get_ego_future_mask(
            ego_future)  # (B, future_len)
        ego_future_on_p_mask = ~ego_future_off_p_mask.unsqueeze(
            -1)  # (B, future_len, 1)
        ego_future_on_mask = ~ego_future_off_mask  # (B)
        # (ego_future_on_num, future_len, 10 + 2k)
        on_ego_future = self._get_on_ego_future(ego_future,
                                                ego_future_on_p_mask,
                                                ego_future_on_mask)
        # on_all_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len
        agents_past_cur_on_num = on_agents_past_cur.shape[0]
        on_agents_past_cur_num = agents_past_cur_on_num * time_len
        ego_future_on_num = on_ego_future.shape[0]

        on_all = self._concat_past_cur_and_future(on_agents_past_cur,
                                                  on_ego_future)

        # on_all output: (on_all_num, channels_mlp_dim)
        on_all = self.channel_pre_project(on_all)
        # on_agents_past_cur: (agents_past_cur_on_num, time_len, channels_mlp_dim)
        on_agents_past_cur = on_all[:on_agents_past_cur_num, :].view(
            agents_past_cur_on_num, time_len, -1)
        # on_ego_future: (ego_future_on_num, future_len, channels_mlp_dim)
        on_ego_future = on_all[on_agents_past_cur_num:, :].view(
            ego_future_on_num, future_len, -1)
        # TODO: channel_pre_project 구현하기

        on_agents_past_cur = on_agents_past_cur.permute(0, 2, 1)

        agents_past_current = self.token_pre_project(agents_past_current)
        agents_past_current = agents_past_current.permute(0, 2, 1)
        for block in self.blocks:
            agents_past_current = block(agents_past_current)

            # pooling
        agents_past_current = torch.mean(agents_past_current, dim=1)

        agents_type = agents_type.view(B * agents_num,
                                       -1)  # (B * agents_num, 3)
        agents_type = agents_type[
            agents_past_cur_on_mask]  # (on_agents_past_cur.sum(), 3)
        type_embedding = self.type_emb(
            agents_type)  # Type embedding for valid data
        agents_past_current = agents_past_current + type_embedding

        agents_past_current = self.emb_project(self.norm(agents_past_current))

        x_result = torch.zeros((B * agents_num, agents_past_current.shape[-1]),
                               device=agents_past_current.device)
        x_result[
            agents_past_cur_on_mask] = agents_past_current  # Fill in valid parts

        return x_result.view(B,
                             agents_num, -1), agents_past_cur_off_mask.reshape(
                                 B, -1), pos.view(B, agents_num, -1)


class StaticFusionEncoder(nn.Module):

    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim,
                              hidden_features=hidden_dim,
                              out_features=hidden_dim,
                              act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        '''
        B, P, _ = x.shape

        pos = x[:, :, :8].clone()  # x, y, cos, sin
        # static: [0, 0,1,0]
        pos[..., -4:] = 0.0
        pos[..., -2] = 1.0

        x_result = torch.zeros((B * P, self._hidden_dim), device=x.device)

        mask_p = torch.sum(torch.ne(x[..., :10], 0), dim=-1).to(x.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            x = x.view(B * P, -1)
            x = x[valid_indices]
            x = self.projection(x)
            x_result[valid_indices] = x

        return x_result.view(B, P, -1), mask_p.view(B, P), pos.view(B, P, -1)


class LaneFusionEncoder(nn.Module):

    def __init__(self,
                 lane_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU,
                                       drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU,
                                     drop=0.)

        self.blocks = nn.ModuleList([
            MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim,
                               out_features=hidden_dim,
                               act_layer=nn.GELU,
                               drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :8].clone()  # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0, 0,0,1]
        pos[..., -4:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        agents_past_cur_off_p_mask = torch.sum(torch.ne(x[..., :8], 0),
                                               dim=-1).to(x.device) == 0
        mask_p = torch.sum(~agents_past_cur_off_p_mask, dim=-1) == 0
        x = x.view(B * P, V, -1)

        valid_indices = ~mask_p.view(-1)
        x = x[valid_indices]

        x = self.channel_pre_project(x)
        x = x.permute(0, 2, 1)
        x = self.token_pre_project(x)
        x = x.permute(0, 2, 1)
        for block in self.blocks:
            x = block(x)

        x = torch.mean(x, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * P, 1)
        has_speed_limit = has_speed_limit.view(B * P, 1)
        traffic = traffic.view(B * P, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros(
            (speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1)
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(
            traffic)  # Traffic light embedding for valid data

        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts

        return x_result.view(B, P, -1), mask_p.reshape(B,
                                                       -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=192,
                 num_heads=6,
                 drop_path_rate=0.3,
                 depth=3,
                 device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):
        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)
