import torch
import torch.nn as nn
from typing import Tuple
from timm.models.layers import Mlp
from timm.layers import DropPath
import torch.nn.functional as F

from diffusion_planner.model.module.mixer import MixerBlock

import torch
from typing import Optional
from typing import Tuple
import torch
import torch.nn as nn


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
        self.chunk_length = 10
        num_fourier_dim = 2 * self.num_fourier_frequencies + 1  # 2K + 1

        self._hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim
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
        agents_past_current: torch.
        Tensor,  # (B, agents_num, time_len, 8 + 2K + 1 )
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

    @staticmethod
    def _compute_equal_chunks(
            seq_len: int, chunk_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """AdaptiveAvgPool1d와 동일한 **균등 분할 경계**를 계산한다.

        분할 포인트 b_m = ceil(m * V / chunk_num)를 이용해 [b_{m-1}, b_m-1]를 구간으로 정한다.

        Args:
            seq_len (int): 시점 길이 V.
            chunk_num (int): 구간 수 chunk_num.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - starts: (chunk_num,) 각 구간 시작 인덱스(포함)
                - ends:   (chunk_num,) 각 구간 종료 인덱스(포함)
        """
        boundaries = torch.div(torch.arange(0, chunk_num + 1) * seq_len,
                               chunk_num,
                               rounding_mode="ceil")
        starts = boundaries[:-1]  # (chunk_num,)
        ends = boundaries[1:] - 1  # (chunk_num,)
        return starts, ends

    def _gated_attentive_pool(
            self,
            chunk_values: torch.Tensor,  # (N, L, C)
            chunk_off_points_mask: torch.Tensor,  # (N, L)
            chunk_is_invalid: torch.Tensor  # (N)
    ) -> torch.Tensor:
        """게이트드 어텐션 풀링을 **tokens_mlp_dim개 쿼리**로 일반화하여 (N, tokens_mlp_dim, C) 출력을 반환한다.
        Args:
            chunk_values (torch.Tensor):
                - shape: (N, L, C)
                - 의미: 구간 내부 L개 시점의 채널 임베딩(이미 per-timestep proj를 통과한 값)
            chunk_off_points_mask (torch.Tensor):
                - shape: (N, L)   # True=무효(해당 프레임 제외)
            chunk_is_invalid: shape: (N,) # True=해당 구간에 유효 시점 0개

        Returns:
            torch.Tensor:
                - shape: (N, tokens_mlp_dim, C)
                - 의미: tokens_mlp_dim개의 쿼리별로 풀링된 구간 대표 벡터

        """
        # ----- 1) 게이트 전처리: h_t = GELU(W z_t) -----
        # chunk_values: (N, L, C) → gated_hidden: (N, L, C)
        gated_hidden = F.gelu(self.gate_linear_W(chunk_values))

        # ----- 2) Q개 점수 산출: logits_{t,q} -----
        # (N, L, C) → (N, L, tokens_mlp_dim)
        logits = self.gate_linear_V(gated_hidden)

        # chunk_off_points_mask.unsqueeze(-1): (N, L, 1)
        # 없는 시간의 점의 가중치를 -1e9로 설정하여 softmax에서 무시.
        logits = logits.masked_fill(chunk_off_points_mask.unsqueeze(-1), -1e9)

        # ----- 3) L축 softmax (쿼리별로 시점 가중치) -----
        # (N, L, tokens_mlp_dim)
        attn = torch.softmax(logits, dim=1)

        # 모든 시점이 무효인 경우(분모 0) softmax NaN 방지 → 0으로 설정
        # chunk_off_points_mask: (N, L)
        # chunk_valid_points_num: (N, 1)
        if chunk_is_invalid.any():
            attn[chunk_is_invalid] = 0.0  # (해당 샘플은 0 가중치)

        # ----- 4) 값 변환 및 가중합 -----
        # value_linear: (C→C)
        # values_proj: (N, L, C)
        values_proj = self.value_linear(chunk_values)

        # einsum으로 Σ_t a_{t,q} * v_t  → (N, tokens_mlp_dim, C)
        # attn: (N, L, tokens_mlp_dim), values_proj: (N, L, C)
        chunk_token = torch.einsum("nlq,nlc->nqc", attn,
                                   values_proj)  # (N, tokens_mlp_dim, C)

        return chunk_token

    def traj_to_chunk_token(
        self,
        valid_traj_token: torch.Tensor,
        valid_traj_off_p_mask: torch.Tensor,
        chunk_starts: torch.Tensor,
        chunk_ends: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """균등 분할된 각 구간에 대해 **집계 토큰**과 **구간 마스크**를 만든다.

        Args:
            valid_traj_token (torch.Tensor):
                - shape: (N, V, C)   # 유효 에이전트 수 N, 시점 V, 채널 C
            valid_traj_off_p_mask (torch.Tensor):
                - shape: (N, V)      # True = 무효(시점)
            chunk_starts (torch.Tensor):
                - shape: (chunk_num,)        # 구간 시작 인덱스(포함)
            chunk_ends (torch.Tensor):
                - shape: (chunk_num,)        # 구간 종료 인덱스(포함)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - chunks: (N, chunk_num, tokens_mlp_dim, C)
                - invalid_chunk_mask: (N, chunk_num)  # True = 무효(해당 구간에 유효 시점 0개)
        """
        num_agents_valid, _, channel_dim = valid_traj_token.shape
        chunks_num = chunk_starts.numel()

        chunks = valid_traj_token.new_zeros(
            num_agents_valid, chunks_num, self.tokens_mlp_dim,
            channel_dim)  # (N, chunk_num, tokens_mlp_dim, C)
        invalid_chunk_mask = torch.zeros(
            num_agents_valid,
            chunks_num,
            dtype=torch.bool,
            device=valid_traj_token.device)  # (N, chunk_num)

        for chunk_idx in range(chunks_num):
            start, end = int(chunk_starts[chunk_idx].item()), int(
                chunk_ends[chunk_idx].item())
            chunk_values = valid_traj_token[:, start:end + 1, :]  # (N, L, C)
            # (N, L)
            chunk_off_points_mask = valid_traj_off_p_mask[:, start:end + 1]
            # (N,)
            chunk_is_invalid = (chunk_off_points_mask == False).sum(dim=1) == 0
            invalid_chunk_mask[:, chunk_idx] = chunk_is_invalid
            # chunk_token: (N, tokens_mlp_dim, C)
            chunk_token = self._gated_attentive_pool(
                chunk_values, chunk_off_points_mask,
                chunk_is_invalid)  # (N, tokens_mlp_dim, C)
            # (N, chunk_num, tokens_mlp_dim, C)
            chunks[:, chunk_idx, :, :] = chunk_token

        return chunks, invalid_chunk_mask  # (N, chunk_num, tokens_mlp_dim, C), (N, chunk_num)

    def _get_type_embedding(
        self,
        agents_type: torch.Tensor,  # (B, agents_num, 3)
        ego_fut_type: torch.Tensor,  # (B, 3)
        agents_past_cur_on_mask: torch.Tensor,  # (B * agents_num)
        ego_future_on_mask: torch.Tensor,  # (B)
        on_agents_past_cur_on_chunk_mask: torch.
        Tensor,  # (agents_past_cur_on_num * past_cur_chunk_num)
        on_ego_fut_on_chunk_mask: torch.
        Tensor,  # (ego_future_on_num * future_chunk_num)
        past_cur_chunk_num: int,
        future_chunk_num: int,
    ) -> torch.Tensor:
        ######## Add Type Embedding ########
        # (B * agents_num, 3)
        B, agents_num = agents_type.shape[:2]
        agents_type = agents_type.view(B * agents_num, -1)
        # (agents_past_cur_on_num, 3)
        agents_type = agents_type[agents_past_cur_on_mask]
        agents_past_cur_on_num = agents_type.shape[0]
        # (agents_past_cur_on_num, 3) -> (agents_past_cur_on_num, 1, 3) -> (agents_past_cur_on_num, past_cur_chunk_num, 3)
        agents_type = agents_type.unsqueeze(1).expand(-1, past_cur_chunk_num,
                                                      -1)
        # (agents_past_cur_on_num, past_cur_chunk_num, 3) -> (agents_past_cur_on_num * past_cur_chunk_num, 3)
        agents_type = agents_type.view(
            agents_past_cur_on_num * past_cur_chunk_num, -1)
        # (agents_past_cur_on_num * past_cur_chunk_num, 3) -> (on_past_cur_on_chunk_num, 3)
        agents_type = agents_type[on_agents_past_cur_on_chunk_mask]
        ##############################
        # ego_fut_type: (B, 3) -> (ego_future_on_num, 3)
        ego_fut_type = ego_fut_type[ego_future_on_mask]
        ego_future_on_num = ego_fut_type.shape[0]
        # (ego_future_on_num, 3) -> (ego_future_on_num, 1, 3) -> (ego_future_on_num, future_chunk_num, 3)
        ego_fut_type = ego_fut_type.unsqueeze(1).expand(-1, future_chunk_num,
                                                        -1)
        # (ego_future_on_num, future_chunk_num, 3) -> (ego_future_on_num * future_chunk_num, 3)
        ego_fut_type = ego_fut_type.view(ego_future_on_num * future_chunk_num,
                                         -1)
        # (ego_future_on_num * future_chunk_num, 3) -> (on_ego_fut_on_chunk_num, 3)
        ego_fut_type = ego_fut_type[on_ego_fut_on_chunk_mask]
        # (on_all_on_chunk_num, 3)
        agents_ego_fut_type = torch.cat([agents_type, ego_fut_type], dim=0)

        agents_ego_fut_type_emb = self.type_emb(agents_ego_fut_type)
        return agents_ego_fut_type_emb

    def _fill_on_chunk_to_on_agent(
            self,
            on_all_on_chunk: torch.Tensor,  # (on_all_on_chunk_num, C)
            on_agents_past_cur_on_chunk_mask: torch.
        Tensor,  # (agents_past_cur_on_num * past_cur_chunk_num)
            on_ego_fut_on_chunk_mask: torch.
        Tensor,  # (ego_future_on_num * future_chunk_num)
            on_past_cur_on_chunk_num: int,
            agents_past_cur_on_num: int,
            ego_future_on_num: int,
            past_cur_chunk_num: int,
            future_chunk_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # (on_past_cur_on_chunk_num, C)
        on_agents_past_cur_on_chunk = on_all_on_chunk[:
                                                      on_past_cur_on_chunk_num, :]
        on_agents_past_cur_chunk = torch.zeros(
            (agents_past_cur_on_num * past_cur_chunk_num, self._hidden_dim),
            device=on_all_on_chunk.device)
        # on_agents_past_cur_on_chunk_mask: (agents_past_cur_on_num * past_cur_chunk_num)
        on_agents_past_cur_chunk[
            on_agents_past_cur_on_chunk_mask] = on_agents_past_cur_on_chunk

        # (on_ego_fut_on_chunk_num, C)
        on_ego_fut_on_chunk = on_all_on_chunk[on_past_cur_on_chunk_num:, :]
        on_ego_fut_chunk = torch.zeros(
            (ego_future_on_num * future_chunk_num, self._hidden_dim),
            device=on_all_on_chunk.device)
        # on_ego_fut_on_chunk_mask: (ego_future_on_num * future_chunk_num)
        on_ego_fut_chunk[on_ego_fut_on_chunk_mask] = on_ego_fut_on_chunk
        return on_agents_past_cur_chunk, on_ego_fut_chunk

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
        ego_fut_type = ego_future[:, 0, 8:].clone()  # (B, 3)
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
        """
        token_pre_project = hard split + gated attentional pooling
        """
        # ---------- 8) 균등 분할 경계 ----------
        past_cur_chunk_num = time_len // self.chunk_length
        past_chunk_start_idx, past_chunk_end_idx = self._compute_equal_chunks(
            seq_len=time_len, chunk_num=past_cur_chunk_num
        )  # (past_cur_chunk_num,), (past_cur_chunk_num,)
        future_chunk_num = future_len // self.chunk_length
        fut_chunk_start_idx, fut_chunk_end_idx = self._compute_equal_chunks(
            seq_len=future_len, chunk_num=future_chunk_num
        )  # (future_chunk_num,), (future_chunk_num,)
        """
        # agents_past_cur_off_p_mask: (B, agents_num, time_len)
        # agents_past_cur_on_mask: (B * agents_num)
        
        # on_agents_past_cur_off_p_mask: (agents_past_cur_on_num, time_len)
        """
        agents_past_cur_off_p_mask = agents_past_cur_off_p_mask.view(
            B * agents_num, time_len)  # (B * agents_num, time_len)
        on_agents_past_cur_off_p_mask = agents_past_cur_off_p_mask[
            agents_past_cur_on_mask]  # (agents_past_cur_on_num, time_len)

        # on_agents_past_cur_chunk: (agents_past_cur_on_num, past_cur_chunk_num, tokens_mlp_dim, C)
        # on_agents_past_cur_off_chunk_mask: (agents_past_cur_on_num, past_cur_chunk_num)

        (on_agents_past_cur_chunk, on_agents_past_cur_off_chunk_mask
        ) = self.traj_to_chunk_token(
            on_agents_past_cur,  # (agents_past_cur_on_num, time_len, channels_mlp_dim)
            on_agents_past_cur_off_p_mask,  # (agents_past_cur_on_num, time_len)
            past_chunk_start_idx,  # (past_cur_chunk_num,)
            past_chunk_end_idx,  # (past_cur_chunk_num,)
        )
        """
        ego_future_off_p_mask: (B, future_len)
        ego_future_on_mask: (B)
        
        on_ego_future_off_p_mask: (ego_future_on_num, future_len)
        """
        on_ego_future_off_p_mask = ego_future_off_p_mask[ego_future_on_mask]

        # on_ego_fut_chunk: (ego_future_on_num, future_chunk_num, tokens_mlp_dim, C)
        # on_ego_fut_off_chunk_mask: (ego_future_on_num, future_chunk_num)

        (on_ego_fut_chunk,
         on_ego_fut_off_chunk_mask) = self.traj_to_chunk_token(
             on_ego_future,  # (ego_future_on_num, future_len, channels_mlp_dim)
             on_ego_future_off_p_mask,  # (ego_future_on_num, future_len)
             fut_chunk_start_idx,  # (future_chunk_num,)
             fut_chunk_end_idx,  # (future_chunk_num,)
         )

        #################
        """
        on_agents_past_cur_chunk: (agents_past_cur_on_num * past_cur_chunk_num, tokens_mlp_dim, C)
        on_agents_past_cur_on_chunk_mask: (agents_past_cur_on_num * past_cur_chunk_num)
        
        on_agents_past_cur_on_chunk: (on_past_cur_on_chunk_num, tokens_mlp_dim, C)
            on_past_cur_on_chunk_num: 
                agents_past_cur_on_num * past_cur_chunk_num 중, True인 개수
        """
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.view(
            agents_past_cur_on_num * past_cur_chunk_num, self.tokens_mlp_dim,
            -1)
        on_agents_past_cur_on_chunk_mask = ~on_agents_past_cur_off_chunk_mask.view(
            agents_past_cur_on_num * past_cur_chunk_num)
        on_agents_past_cur_on_chunk = on_agents_past_cur_chunk[
            on_agents_past_cur_on_chunk_mask]
        on_past_cur_on_chunk_num = on_agents_past_cur_on_chunk.shape[0]
        """
        on_ego_fut_chunk: (ego_future_on_num, future_chunk_num * tokens_mlp_dim, C)
        on_ego_fut_on_chunk_mask: (ego_future_on_num * future_chunk_num)
        
        on_ego_fut_on_chunk: (on_ego_fut_on_chunk_num, tokens_mlp_dim, C)
            on_ego_fut_on_chunk_num:
                ego_future_on_num * future_chunk_num 중, True인 개수
        """
        on_ego_fut_chunk = on_ego_fut_chunk.view(
            ego_future_on_num * future_chunk_num, self.tokens_mlp_dim, -1)
        on_ego_fut_on_chunk_mask = ~on_ego_fut_off_chunk_mask.view(
            ego_future_on_num * future_chunk_num)
        on_ego_fut_on_chunk = on_ego_fut_chunk[on_ego_fut_on_chunk_mask]
        on_ego_fut_on_chunk_num = on_ego_fut_on_chunk.shape[0]
        """
        on_all_on_chunk: (on_all_on_chunk_num, tokens_mlp_dim, C)
            on_all_on_chunk_num = on_past_cur_on_chunk_num + on_ego_fut_on_chunk_num
        """
        on_all_on_chunk = torch.cat(
            [on_agents_past_cur_on_chunk, on_ego_fut_on_chunk], dim=0)

        ############
        for block in self.blocks:
            on_all_on_chunk = block(on_all_on_chunk)
        # pooling
        # on_all_on_chunk: (on_all_on_chunk_num, C)
        on_all_on_chunk = torch.mean(on_all_on_chunk, dim=1)

        # TODO: 여기서부터!
        agents_ego_fut_type_emb = self._get_type_embedding(
            agents_type, ego_fut_type, agents_past_cur_on_mask,
            ego_future_on_mask, on_agents_past_cur_on_chunk_mask,
            on_ego_fut_on_chunk_mask, past_cur_chunk_num, future_chunk_num)

        # on_all_on_chunk: (on_all_on_chunk_num, C)
        on_all_on_chunk = on_all_on_chunk + agents_ego_fut_type_emb

        on_all_on_chunk = self.emb_project(self.norm(on_all_on_chunk))

        ########################
        # on_agents_past_cur_chunk: (agents_past_cur_on_num * past_cur_chunk_num, C)
        # on_ego_fut_chunk: (ego_future_on_num * future_chunk_num, C)
        on_agents_past_cur_chunk, on_ego_fut_chunk = self._fill_on_chunk_to_on_agent(
            on_all_on_chunk, on_agents_past_cur_on_chunk_mask,
            on_ego_fut_on_chunk_mask, on_past_cur_on_chunk_num,
            agents_past_cur_on_num, ego_future_on_num, past_cur_chunk_num,
            future_chunk_num)
        # _fill_on_agent_to_agent
        agents_past_cur_chunk = torch.zeros(
            (B * agents_num, past_cur_chunk_num, self._channel),
            device=on_agents_past_cur_chunk.device)
        # agents_past_cur_on_mask: (B * agents_num) -> agents_past_cur_on_num
        agents_past_cur_chunk[
            agents_past_cur_on_mask] = on_agents_past_cur_chunk
        # (B * agents_num, past_cur_chunk_num, self._channel)
        # -> (B,  agents_num * past_cur_chunk_num , self._channel)
        agents_past_cur_chunk = agents_past_cur_chunk.view(
            B, agents_num * past_cur_chunk_num, self._channel)

        ego_fut_chunk = torch.zeros((B, future_chunk_num, self._channel),
                                    device=on_ego_fut_chunk.device)
        # ego_future_on_mask: (B)
        # -> ego_future_on_num
        ego_fut_chunk[ego_future_on_mask] = on_ego_fut_chunk
        # concat: (B, agents_num * past_cur_chunk_num + future_chunk_num, self._channel)
        # all_chunk = torch.cat([agents_past_cur_chunk, ego_fut_chunk], dim=1)

        all_chunk, all_chunk_off_mask = self._concat_chunks_and_build_mask(
            agents_past_cur_chunk=agents_past_cur_chunk,
            # (B, agents_num*past_cur_chunk_num, C)
            ego_fut_chunk=ego_fut_chunk,  # (B, future_chunk_num, C)
            past_cur_chunk_num=past_cur_chunk_num,
            future_chunk_num=future_chunk_num,
            agents_past_cur_on_mask=agents_past_cur_on_mask,
            # (B*agents_num,)  True=유효
            on_agents_past_cur_off_chunk_mask=on_agents_past_cur_off_chunk_mask,
            # (agents_past_cur_on_num, past_cur_chunk_num) True=무효
            ego_future_on_mask=ego_future_on_mask,  # (B,) True=유효
            on_ego_fut_off_chunk_mask=on_ego_fut_off_chunk_mask,
            # (ego_future_on_num, future_chunk_num) True=무효
        )

        # all_chunk_off_mask: (B, agents_num * past_cur_chunk_num + future_chunk_num)
        return all_chunk, all_chunk_off_mask

    def _concat_chunks_and_build_mask(
        self,
        agents_past_cur_chunk: torch.Tensor,
        ego_fut_chunk: torch.Tensor,
        past_cur_chunk_num: int,
        future_chunk_num: int,
        agents_past_cur_on_mask: torch.Tensor,
        on_agents_past_cur_off_chunk_mask: torch.Tensor,
        ego_future_on_mask: torch.Tensor,
        on_ego_fut_off_chunk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """에이전트 과거/현재 구간과 ego 미래 구간의 **토큰/마스크**를 결합한다.

        개요:
            - 과거/현재(agents)와 미래(ego)의 구간별 토큰을 concat하여 `all_chunk`를 만들고,
            - 각각의 구간 무효(오프) 마스크를 적절히 브로드캐스트/리쉐이프/대입하여
              배치 단위 `(B, agents_num * past_cur_chunk_num + future_chunk_num)`의
              **최종 무효 마스크** `all_chunk_off_mask`를 만든다.

        Args:
            agents_past_cur_chunk (torch.Tensor):
                - shape: (B, agents_num * past_cur_chunk_num, C)
                - 의미: 에이전트별 과거/현재 구간 토큰을 (에이전트,구간)축을 펼쳐 concat한 토큰.
            ego_fut_chunk (torch.Tensor):
                - shape: (B, future_chunk_num, C)
                - 의미: ego 미래 구간 토큰.
            past_cur_chunk_num (int):
                - 에이전트당 과거/현재 구간 수.
            future_chunk_num (int):
                - ego 미래 구간 수.
            agents_past_cur_on_mask (torch.Tensor):
                - shape: (B * agents_num,)
                - dtype: torch.bool
                - 의미: 에이전트 **유효(True)** 마스크. (주의: True=유효, False=무효)
                  * 기존 코드 컨벤션과 동일하게 사용합니다.
            on_agents_past_cur_off_chunk_mask (torch.Tensor):
                - shape: (agents_past_cur_on_num, past_cur_chunk_num)
                - dtype: torch.bool
                - 의미: **구간 무효(True=무효)** 마스크(유효 에이전트만 추려진 상태).
            ego_future_on_mask (torch.Tensor):
                - shape: (B,)
                - dtype: torch.bool
                - 의미: 배치별 ego 미래 시퀀스 **유효(True)** 마스크.
            on_ego_fut_off_chunk_mask (torch.Tensor):
                - shape: (ego_future_on_num, future_chunk_num)
                - dtype: torch.bool
                - 의미: **구간 무효(True=무효)** 마스크(유효 배치만 추려진 상태).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - all_chunk:
                    * shape: (B, agents_num * past_cur_chunk_num + future_chunk_num, C)
                    * 의미: 과거/현재(에이전트) + 미래(ego)의 구간 토큰을 열 방향으로 이어붙인 결과
                - all_chunk_off_mask:
                    * shape: (B, agents_num * past_cur_chunk_num + future_chunk_num)
                    * dtype: torch.bool
                    * 의미: **True=무효(패딩)** 키 패딩 마스크. 위 all_chunk의 각 토큰과 1:1 대응.

        Note:
            - `agents_num`은 `agents_past_cur_chunk.shape[1] // past_cur_chunk_num` 로 유추합니다.
            - 디바이스/ dtype은 입력 텐서들에서 자동으로 맞춥니다.
        """
        # -------------------- 기본 치수/디바이스 유추 --------------------
        B: int = agents_past_cur_chunk.size(
            0)  # (B, agents_num * past_cur_chunk_num, C)
        # (B, A*past_cur_chunk_num, C) → agents_num 유추
        agents_num: int = agents_past_cur_chunk.size(1) // past_cur_chunk_num
        device = agents_past_cur_chunk.device

        # -------------------- 1) 에이전트 과거/현재 구간 무효 마스크 구성 --------------------
        # (B*agents_num, past_cur_chunk_num), 기본 True=무효로 초기화
        agents_past_cur_off_chunk_mask_all = torch.ones(
            B * agents_num,
            past_cur_chunk_num,
            dtype=torch.bool,
            device=device,
        )  # True=무효(기본)

        # 유효 에이전트 위치에 대해 실제 구간 무효 마스크를 덮어쓰기
        # agents_past_cur_on_mask: (B * agents_num), True=유효
        # on_agents_past_cur_off_chunk_mask: (agents_past_cur_on_num, past_cur_chunk_num), True=무효
        agents_past_cur_off_chunk_mask_all[
            agents_past_cur_on_mask] = on_agents_past_cur_off_chunk_mask

        # (B, agents_num * past_cur_chunk_num)로 펴기
        agents_past_cur_off_chunk_mask_all = agents_past_cur_off_chunk_mask_all.view(
            B, agents_num, past_cur_chunk_num).reshape(
                B, agents_num * past_cur_chunk_num)  # (B, A*M_past)

        # -------------------- 2) ego 미래 구간 무효 마스크 구성 --------------------
        # (B, future_chunk_num), 기본 True=무효로 초기화
        ego_fut_off_chunk_mask_all = torch.ones(
            B,
            future_chunk_num,
            dtype=torch.bool,
            device=device,
        )  # (B, M_future), True=무효(기본)

        # 유효 배치(ego_future_on_mask=True)에 대해 실제 구간 무효 마스크를 덮어쓰기
        # on_ego_fut_off_chunk_mask: (ego_future_on_num, future_chunk_num), True=무효
        ego_fut_off_chunk_mask_all[
            ego_future_on_mask] = on_ego_fut_off_chunk_mask  # (B, M_future)

        # -------------------- 3) 최종 마스크 결합 --------------------
        all_chunk_off_mask = torch.cat(
            [agents_past_cur_off_chunk_mask_all, ego_fut_off_chunk_mask_all],
            dim=1)  # (B, A*M_past + M_future)

        # -------------------- 4) 토큰 결합 --------------------
        # agents_past_cur_chunk: (B, A*M_past, C)
        # ego_fut_chunk        : (B, M_future, C)
        all_chunk = torch.cat([agents_past_cur_chunk, ego_fut_chunk],
                              dim=1)  # (B, A*M_past + M_future, C)

        return all_chunk, all_chunk_off_mask


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
