import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Tuple
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.module.mixer import MixerBlock


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hidden_dim = config.hidden_dim

        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )
        self.token_num = (
            (config.agent_num + 1) * self.neighbor_encoder.tokens_per_agent
            + config.static_objects_num
            + config.lane_num
        )
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
        )

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device,
        )

        # position embedding encode x, y, cos, sin, type, time
        self.pos_emb = nn.Linear(
            8 + 2 * self.neighbor_encoder.num_fourier_frequencies,
            config.hidden_dim,
        )

    def forward(self, inputs):

        encoder_outputs = {}

        # agents
        ego_past = inputs['ego_agent_past']
        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]

        encoding_agents, agents_mask, agent_pos = self.neighbor_encoder(
            ego_past.unsqueeze(1), neighbors
        )
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit
        )

        pos_dim = agent_pos.shape[-1]
        if static_pos.shape[-1] < pos_dim:
            pad = static_pos.new_zeros(B, static_pos.shape[1], pos_dim - static_pos.shape[-1])
            static_pos = torch.cat([static_pos, pad], dim=-1)
        if lane_pos.shape[-1] < pos_dim:
            pad = lane_pos.new_zeros(B, lane_pos.shape[1], pos_dim - lane_pos.shape[-1])
            lane_pos = torch.cat([lane_pos, pad], dim=-1)

        encoding_input = torch.cat([encoding_agents, encoding_static, encoding_lanes], dim=1)

        encoding_pos = torch.cat([agent_pos, static_pos, lane_pos], dim=1).view(
            B * self.token_num, -1
        )
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask], dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros(
            (B * self.token_num, self.hidden_dim), device=encoding_pos.device
        )
        encoding_pos_result[~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + encoding_pos_result.view(B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num)
        )

        return encoder_outputs


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim=192, heads=6, dropout=0.1, mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)

        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x, mask):
        x = x + self.drop_path(self.attn(self.norm1(x), x, x, key_padding_mask=mask)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    """에이전트(ego + 이웃) 시계열을 **에이전트당 M개 토큰**으로 인코딩하는 모듈.

    개요:
        - 입력으로 ego 1대와 P명의 이웃 에이전트의 시계열 벡터를 받는다.
        - 시간 길이 V를 **균등 분할**하여 각 에이전트별로 M개 구간을 만든다.
        - 각 구간 내부는
          - *mean*: 마스크 평균(AdaptiveAvgPool1d와 동일한 경계) 또는
          - *gated*: 게이트드 어텐션 풀링
          으로 집계하여 구간 대표 토큰을 만든다.
        - 구간 토큰은 **MLP‑Mixer**를 통과(토큰 길이 M 유지)하고,
          에이전트 **type 임베딩**을 M축으로 브로드캐스트하여 더한다.
        - 마지막으로 LN+MLP로 모델 히든 차원 d로 사상한다.
        - pos는 구간 **중앙 시점**의 위치/방향 + type(4) + **시간 Fourier feature(2K)** 로 구성된다.

    Shape 약속:
        - B: 배치 크기
        - A: 에이전트 수 (ego 1 + 이웃 P → A = 1+P)
        - V: 시간 길이 (예: 21)
        - M: 에이전트당 출력 토큰 수 (기본 2)
        - D_raw: 원시 피처 차원
        - C: 채널 임베딩 차원 (기본 128)
        - d: 최종 임베딩 차원 (기본 192)
        - K: 시간 Fourier 주파수 수 (기본 4)

    Args:
        time_len (int): 시간 길이 V.
        hidden_dim (int): 최종 임베딩 차원 d. 기본 192.
        depth (int): MixerBlock 반복 횟수.
        channels_mlp_dim (int): 채널 임베딩 차원 C. 기본 128.
        tokens_per_agent (int): 에이전트당 토큰 수 M. 기본 2.
        pooling_mode (Literal["mean","gated"]): 구간 집계 방식.
            - "mean": 마스크 평균(AdaptiveAvgPool 경계).
            - "gated": 게이트드 어텐션 풀링.
        drop_path_rate (float): DropPath 비율.
        num_fourier_frequencies (int): 시간 Fourier feature의 주파수 수 K. 기본 4.
        t_min (float): 전역 시간축 최소값(초). 기본 -2.0.
        t_max (float): 전역 시간축 최대값(초). 기본 +8.0.
        use_value_proj (bool): 게이트드 풀링의 값 변환 W_v 사용 여부.

    Note:
        - `token_pre_project`는 제거되었으며, Mixer의 토큰 길이는 **항상 M**으로 유지됨.
        - 상위 `Encoder`의 `pos_emb` 입력 차원은 **8 + 2*K**가 되어야 한다.
        - 상위 `Encoder`의 `token_num`은 에이전트 토큰을 **A → A*M**으로 반영해야 한다.
    """

    def __init__(
        self,
        time_len: int,
        drop_path_rate: float = 0.3,
        hidden_dim: int = 192,
        depth: int = 3,
        tokens_per_agent: int = 2,
        pooling_mode: Literal["mean", "gated"] = "mean",
        channels_mlp_dim: int = 128,
        num_fourier_frequencies: int = 4,
        t_min: float = -2.0,
        t_max: float = 8.0,
        use_value_proj: bool = False,
    ) -> None:
        super().__init__()

        # -------------------- 하이퍼/상수 --------------------
        self.time_len: int = time_len                              # V
        self.tokens_per_agent: int = tokens_per_agent              # M
        self.channel_dim: int = channels_mlp_dim                   # C
        self.hidden_dim: int = hidden_dim                          # d
        self.pooling_mode: str = pooling_mode
        self.num_fourier_frequencies: int = num_fourier_frequencies  # K
        self.time_min: float = t_min
        self.time_max: float = t_max
        # 과거 구간 [-2, 0]에서 프레임 간 시간 간격(초) 추정치: 2s / (V-1)
        self.dt_sec: float = 2.0 / max(1, time_len - 1)

        # -------------------- 타입/채널 임베딩 --------------------
        # [shape] (B, A, 3) one-hot → (B, A, C)
        self.type_embedding = nn.Linear(3, channels_mlp_dim)

        # per‑timestep 채널 임베딩: (8D + valid) → C
        # [입력 shape] (B, A, V, 9) → [출력 shape] (B, A, V, C)
        self.per_timestep_proj = Mlp(
            in_features=8 + 1,
            hidden_features=channels_mlp_dim,
            out_features=channels_mlp_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # token_pre_project 제거!

        # -------------------- MLP‑Mixer --------------------
        # MixerBlock의 첫 인자가 "토큰 축 MLP 차원"인 구현이라 가정하고 M으로 설정
        # [입력/출력 shape] (N, M, C) → (N, M, C)
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(self.tokens_per_agent, channels_mlp_dim, drop_path_rate) for _ in range(depth)]
        )

        self.pre_out_norm = nn.LayerNorm(channels_mlp_dim)  # [shape] (N, M, C) → (N, M, C)
        self.token_out_proj = Mlp(                          # [shape] (N, M, C) → (N, M, d)
            in_features=channels_mlp_dim,
            hidden_features=hidden_dim,
            out_features=hidden_dim,
            act_layer=nn.GELU,
            drop=drop_path_rate,
        )

        # -------------------- 게이트드 풀링 파라미터(옵션) --------------------
        # [입력 shape] (N, L, C)
        if self.pooling_mode == "gated":
            self.gate_linear_W = nn.Linear(channels_mlp_dim, channels_mlp_dim)  # (C → C)
            self.gate_linear_v = nn.Linear(channels_mlp_dim, 1, bias=False)     # (C → 1)
            # 값 변환(선택): (C → C)
            self.value_linear = (
                nn.Linear(channels_mlp_dim, channels_mlp_dim) if use_value_proj else nn.Identity()
            )

    # ==================== 내부 유틸 ====================

    @staticmethod
    def _compute_invalid_timestep_mask_from_8d(spatial_features_8d: torch.Tensor) -> torch.Tensor:
        """시점별 **유효/무효 마스크**를 계산한다.

        시점의 8D 공간 피처(좌표/방향/크기 등)가 **전부 0**이면 무효(True)로 간주한다.

        Args:
            spatial_features_8d (torch.Tensor):
                - shape: (B, A, V, 8)

        Returns:
            torch.Tensor:
                - shape: (B, A, V)
                - 의미: True = 무효(전부 0), False = 유효
        """
        # [연산 shape] (B, A, V, 8) → (B, A, V)
        return torch.sum(torch.ne(spatial_features_8d, 0), dim=-1) == 0

    @staticmethod
    def _compute_invalid_agent_mask(invalid_timestep_mask: torch.Tensor) -> torch.Tensor:
        """에이전트 단위의 **무효 여부**를 계산한다.

        한 에이전트의 모든 시점이 무효(True)라면, 해당 에이전트를 무효(True)로 처리한다.

        Args:
            invalid_timestep_mask (torch.Tensor):
                - shape: (B, A, V)
                - 의미: True = 무효(시점)

        Returns:
            torch.Tensor:
                - shape: (B, A)
                - 의미: True = 무효(에이전트)
        """
        # [연산 shape] (B, A, V) → (B, A)
        return torch.sum(~invalid_timestep_mask, dim=-1) == 0

    @staticmethod
    def _compute_equal_segments(seq_len: int, num_segments: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """AdaptiveAvgPool1d와 동일한 **균등 분할 경계**를 계산한다.

        분할 포인트 b_m = floor(m * V / M)를 이용해 [b_{m-1}, b_m-1]를 구간으로 정한다.

        Args:
            seq_len (int): 시점 길이 V.
            num_segments (int): 구간 수 M.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - starts: (M,) 각 구간 시작 인덱스(포함)
                - ends:   (M,) 각 구간 종료 인덱스(포함)
        """
        # [연산 shape] (M+1,) 경계 인덱스 생성
        boundaries = torch.div(torch.arange(0, num_segments + 1) * seq_len, num_segments, rounding_mode="floor")
        starts = boundaries[:-1]               # (M,)
        ends = boundaries[1:] - 1              # (M,)
        return starts, ends

    @staticmethod
    def _segment_centers(starts: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """각 구간의 **중앙 시점 인덱스**를 계산한다.

        Args:
            starts (torch.Tensor): (M,) 구간 시작(포함)
            ends (torch.Tensor):   (M,) 구간 종료(포함)

        Returns:
            torch.Tensor:
                - shape: (M,)
                - dtype: long
                - 의미: 각 구간의 중앙 인덱스
        """
        # [연산 shape] (M,) → (M,)
        return ((starts + ends) // 2).to(torch.long)

    @staticmethod
    def _masked_mean_pool(values: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """구간 내부의 **마스크 평균 풀링**.

        Args:
            values (torch.Tensor):
                - shape: (N, L, C)  # N: 유효 에이전트 수, L: 구간 길이, C: 채널
            invalid_mask (torch.Tensor):
                - shape: (N, L)     # True = 무효(해당 프레임 제외)

        Returns:
            torch.Tensor:
                - shape: (N, C)     # 구간 대표 벡터
        """
        # [연산 shape] (N, L) → (N, L, 1)
        valid_weight = (~invalid_mask).float().unsqueeze(-1)
        # [연산 shape] (N, L, 1) * (N, L, C) → (N, L, C) → (N, C)
        numerator = (valid_weight * values).sum(dim=1)
        # [연산 shape] (N, L, 1) → (N, 1)
        denominator = valid_weight.sum(dim=1).clamp_min(1e-6)
        return numerator / denominator

    def _gated_attentive_pool(self, values: torch.Tensor, invalid_mask: torch.Tensor) -> torch.Tensor:
        """구간 내부의 **게이트드 어텐션 풀링**.

        수식 개요:
            - 점수:  score_t = v^T GELU(W z_t)
            - 가중치: a_t = softmax(score_t) (무효 프레임은 -∞ 마스킹)
            - 집계:  H = Σ_t a_t * (W_v z_t)   (W_v는 선택적)

        Args:
            values (torch.Tensor):
                - shape: (N, L, C)  # 시점 임베딩
            invalid_mask (torch.Tensor):
                - shape: (N, L)     # True = 무효

        Returns:
            torch.Tensor:
                - shape: (N, C)     # 구간 대표 벡터
        """
        # [연산] (N, L, C) → (N, L, C)
        gated_hidden = F.gelu(self.gate_linear_W(values))
        # [연산] (N, L, C) → (N, L)
        logits = self.gate_linear_v(gated_hidden).squeeze(-1)
        # 무효 프레임은 큰 음수로 마스킹
        logits = logits.masked_fill(invalid_mask, -1e9)
        # [연산] (N, L) → (N, L, 1)
        attn = logits.softmax(dim=1).unsqueeze(-1)
        # [연산] (N, L, C)
        pooled_values = self.value_linear(values)
        # [연산] Σ_t a_t * v_t → (N, C)
        return (attn * pooled_values).sum(dim=1)

    def _aggregate_segments(
        self,
        timestep_emb: torch.Tensor,
        invalid_timestep_mask: torch.Tensor,
        seg_starts: torch.Tensor,
        seg_ends: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """균등 분할된 각 구간에 대해 **집계 토큰**과 **구간 마스크**를 만든다.

        Args:
            timestep_emb (torch.Tensor):
                - shape: (N, V, C)   # 유효 에이전트 수 N, 시점 V, 채널 C
            invalid_timestep_mask (torch.Tensor):
                - shape: (N, V)      # True = 무효(시점)
            seg_starts (torch.Tensor):
                - shape: (M,)        # 구간 시작 인덱스(포함)
            seg_ends (torch.Tensor):
                - shape: (M,)        # 구간 종료 인덱스(포함)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - segment_tokens: (N, M, C)  # 구간별 대표 토큰
                - invalid_segment_mask: (N, M)  # True = 무효(해당 구간에 유효 시점 0개)
        """
        num_agents_valid, seq_len, channel_dim = timestep_emb.shape
        num_segments = seg_starts.numel()

        # [초기화 shape] (N, M, C), (N, M)
        segment_tokens = timestep_emb.new_zeros(num_agents_valid, num_segments, channel_dim)
        invalid_segment_mask = torch.zeros(
            num_agents_valid, num_segments, dtype=torch.bool, device=timestep_emb.device
        )

        # 각 구간에 대해 슬라이스 → 집계
        for m_idx in range(num_segments):
            start, end = int(seg_starts[m_idx].item()), int(seg_ends[m_idx].item())

            # [슬라이스 shape] (N, L, C) / (N, L)
            slice_values = timestep_emb[:, start : end + 1, :]
            slice_invalid = invalid_timestep_mask[:, start : end + 1]

            # [무효 구간 여부] (N,)  — 해당 구간에 유효 프레임이 하나도 없으면 True
            no_valid_in_slice = (slice_invalid == False).sum(dim=1) == 0
            invalid_segment_mask[:, m_idx] = no_valid_in_slice

            # [집계] (N, L, C) → (N, C)
            if self.pooling_mode == "gated":
                pooled = self._gated_attentive_pool(slice_values, slice_invalid)
            else:
                pooled = self._masked_mean_pool(slice_values, slice_invalid)

            # [쓰기] (N, C) → (N, M, C)
            segment_tokens[:, m_idx, :] = pooled

        return segment_tokens, invalid_segment_mask

    def _encode_time_with_fourier_features(self, t_sec: torch.Tensor) -> torch.Tensor:
        """시간 값을 **Fourier feature**로 인코딩한다.

        정규화:
            \( \tilde{t} = \frac{t - t_{\min}}{t_{\max} - t_{\min}} \in [0,1] \)

        생성:
            \( [\cos(k\pi\tilde{t}),\ \sin(k\pi\tilde{t})]_{k=1..K} \Rightarrow 2K\text{차원} \)

        Args:
            t_sec (torch.Tensor):
                - shape: (...,)   # 초 단위 시간

        Returns:
            torch.Tensor:
                - shape: (..., 2K)
        """
        # [연산] 정규화 (...,) → (...,)
        t_norm = (t_sec - self.time_min) / (self.time_max - self.time_min)
        t_norm = t_norm.clamp(0.0, 1.0)

        # [연산] (K,)
        k_indices = torch.arange(1, self.num_fourier_frequencies + 1, device=t_sec.device).float()
        # [연산] (...,1) * (K,) → (..., K)
        angles = t_norm.unsqueeze(-1) * k_indices * torch.pi
        # [연산] (..., K) → (..., 2K)
        return torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)

    # ==================== forward ====================

    def forward(self, x_ego: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """에이전트 시계열을 **에이전트당 M개 토큰**으로 인코딩한다.

        Args:
            x_ego (torch.Tensor):
                - shape: (B, 1, V, D_raw)  # ego 1대
            x (torch.Tensor):
                - shape: (B, P, V, D_raw)  # 이웃 P대

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - enc:
                    * shape: (B, (1+P)*M, d)
                    * 의미: 에이전트당 M개 토큰의 최종 임베딩
                - mask:
                    * shape: (B, (1+P)*M)
                    * 의미: True=무효(키 패딩 마스크)
                - pos:
                    * shape: (B, (1+P)*M, 8 + 2K)
                    * 의미: [x, y, cos, sin, type(4), Fourier time(2K)]
        """
        # ---------- 1) ego + neighbors 결합 ----------
        # [입력 shape] x_ego (B, 1, V, D_raw), x (B, P, V, D_raw)
        agents_raw = torch.cat([x_ego, x], dim=1)                                 # (B, A, V, D_raw)
        batch_size, num_agents, seq_len, _ = agents_raw.shape
        assert seq_len == self.time_len, f"time_len(V) 불일치: got {seq_len}, expected {self.time_len}"

        # ---------- 2) 타입(one-hot 3) 추출 ----------
        # [shape] (B, A, 3)
        agent_type_onehot = agents_raw[:, :, -1, 8:]

        # ---------- 3) 공간 8D / 유효 플래그 ----------
        # [shape] (B, A, V, 8)
        spatial_features_8d = agents_raw[..., :8]
        # [shape] (B, A, V)  True=무효
        invalid_timestep_mask = self._compute_invalid_timestep_mask_from_8d(spatial_features_8d)
        # [shape] (B, A, V, 1)  (1=유효)
        timestep_valid_float = (~invalid_timestep_mask).float().unsqueeze(-1)

        # ---------- 4) 시점별 채널 임베딩 (8D+valid → C) ----------
        # [입력 shape] (B, A, V, 9) → [출력 shape] (B, A, V, C)
        per_timestep_inputs = torch.cat([spatial_features_8d, timestep_valid_float], dim=-1)
        timestep_channel_embeddings = self.per_timestep_proj(per_timestep_inputs)

        # ---------- 5) 에이전트 무효 마스크 ----------
        # [shape] (B, A)  True=무효(해당 에이전트의 모든 시점이 무효)
        invalid_agent_mask = self._compute_invalid_agent_mask(invalid_timestep_mask)

        # ---------- 6) (B, A, ...) → (BA, ...) 평탄화 ----------
        batch_agents = batch_size * num_agents
        # [shape] (BA, V, C)
        timestep_channel_embeddings = timestep_channel_embeddings.view(batch_agents, seq_len, self.channel_dim)
        # [shape] (BA, V)
        invalid_timestep_mask = invalid_timestep_mask.view(batch_agents, seq_len)
        # [shape] (BA, 3)
        agent_type_onehot_flat = agent_type_onehot.view(batch_agents, -1)
        # [shape] (BA, V, 8)
        spatial_features_8d_flat = spatial_features_8d.view(batch_agents, seq_len, 8)

        # ---------- 7) 유효 에이전트만 인덱싱 ----------
        # [shape] (BA,) → (N,) 인덱스
        valid_agent_indices = (~invalid_agent_mask.view(-1)).nonzero(as_tuple=False).squeeze(-1)
        # [shape] (N, V, C), (N, V), (N, 3), (N, V, 8)
        emb_valid = timestep_channel_embeddings[valid_agent_indices]
        invalid_timestep_mask_valid = invalid_timestep_mask[valid_agent_indices]
        type_onehot_valid = agent_type_onehot_flat[valid_agent_indices]
        spatial_features_8d_valid = spatial_features_8d_flat[valid_agent_indices]
        num_valid_agents = emb_valid.size(0)

        # ---------- 8) 균등 분할 경계(AdaptiveAvgPool 스타일) ----------
        # [shape] (M,), (M,)
        seg_starts, seg_ends = self._compute_equal_segments(seq_len=self.time_len, num_segments=self.tokens_per_agent)

        # ---------- 9) 구간 집계(평균/게이트드) ----------
        # [입력] (N, V, C) → [출력] (N, M, C), (N, M)
        segment_tokens, invalid_segment_mask = self._aggregate_segments(
            emb_valid, invalid_timestep_mask_valid, seg_starts, seg_ends
        )

        # ---------- 10) MLP‑Mixer (토큰 길이 M 유지) ----------
        # [shape] (N, M, C) → (N, M, C)
        for mixer in self.mixer_blocks:
            segment_tokens = mixer(segment_tokens)

        # ---------- 11) 타입 임베딩을 M축 브로드캐스트 ----------
        # [shape] (N, 3) → (N, C)
        per_agent_type_emb = self.type_embedding(type_onehot_valid)
        # [연산] (N, C) → (N, 1, C) → 브로드캐스트 더하기
        segment_tokens = segment_tokens + per_agent_type_emb.unsqueeze(1)  # 최종 (N, M, C)

        # ---------- 12) 토큰 투영 (C → d) ----------
        # [shape] (N, M, C) → (N, M, d)
        segment_tokens = self.token_out_proj(self.pre_out_norm(segment_tokens))

        # ---------- 13) (BA, M, d) 버퍼에 복구 ----------
        # [초기화] (BA, M, d)
        encoded_tokens_all = timestep_channel_embeddings.new_zeros(
            batch_agents, self.tokens_per_agent, self.hidden_dim
        )
        # [쓰기] 유효 에이전트 위치에만 기록
        encoded_tokens_all[valid_agent_indices] = segment_tokens  # (BA, M, d)

        # ---------- 14) 구간 마스크(무효) ----------
        # [초기화] (BA, M) True로 채우고 → 유효 에이전트에 대해 실제 seg 마스크로 대체
        invalid_segment_mask_all = torch.ones(
            batch_agents, self.tokens_per_agent, dtype=torch.bool, device=timestep_channel_embeddings.device
        )
        invalid_segment_mask_all[valid_agent_indices] = invalid_segment_mask  # (BA, M)
        # [shape] (BA, M) → (B, A*M)
        mask_out = invalid_segment_mask_all.view(batch_size, num_agents * self.tokens_per_agent)

        # ---------- 15) pos: 중앙 시점의 [x,y,cos,sin] + type(4) + 시간 Fourier(2K) ----------
        # [shape] (M,)
        segment_center_indices = self._segment_centers(seg_starts, seg_ends).to(spatial_features_8d_valid.device)
        # 안전 인덱싱: index_select
        # [shape] (N, M, 8)
        central_spatial_8d = spatial_features_8d_valid.index_select(1, segment_center_indices)
        # [shape] (N, M, 4)
        pos_xycos_sin = central_spatial_8d[:, :, :4]

        # [shape] (N, M) ego 여부(샘플 내 0번째가 ego)
        agent_index_within_sample = (valid_agent_indices % num_agents)
        is_ego_mask = (agent_index_within_sample == 0).view(num_valid_agents, 1).expand(
            num_valid_agents, self.tokens_per_agent
        )
        # [shape] (N, M, 4)  type one-hot: ego->[1,0,0,0], neighbor->[0,1,0,0]
        pos_type_onehot = torch.zeros(
            num_valid_agents, self.tokens_per_agent, 4, device=timestep_channel_embeddings.device
        )
        pos_type_onehot[..., 0] = is_ego_mask.float()
        pos_type_onehot[..., 1] = (~is_ego_mask).float()

        # 중앙 시점의 실제 시간(초): t = -2.0 + center_idx * dt_sec  → Fourier(2K)
        # [shape] (N, M)
        center_idx_broadcast = segment_center_indices.view(1, self.tokens_per_agent).expand(
            num_valid_agents, self.tokens_per_agent
        )
        t_center_seconds = -2.0 + center_idx_broadcast.float() * self.dt_sec
        # [shape] (N, M, 2K)
        time_fourier_features = self._encode_time_with_fourier_features(t_center_seconds)

        # [shape] (N, M, 8 + 2K)
        pos_features_valid_agents = torch.cat([pos_xycos_sin, pos_type_onehot, time_fourier_features], dim=-1)

        # (BA, M, 8+2K) 버퍼에 복구
        pos_features_all_agents = spatial_features_8d_flat.new_zeros(
            batch_agents, self.tokens_per_agent, 8 + 2 * self.num_fourier_frequencies
        )
        pos_features_all_agents[valid_agent_indices] = pos_features_valid_agents
        # [shape] (BA, M, 8+2K) → (B, A*M, 8+2K)
        pos_out = pos_features_all_agents.view(batch_size, num_agents * self.tokens_per_agent, -1)

        # ---------- 16) (BA, M, d) → (B, A*M, d) ----------
        enc_out = encoded_tokens_all.view(batch_size, num_agents * self.tokens_per_agent, self.hidden_dim)

        return enc_out, mask_out, pos_out

    
class StaticFusionEncoder(nn.Module):
    def __init__(self, dim, drop_path_rate=0.3, hidden_dim=192, device='cuda'):
        super().__init__()

        self._hidden_dim = hidden_dim

        self.projection = Mlp(in_features=dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x):
        '''
        x: B, P, D (x, y, cos, sin, w, l, type(4))
        ''' 
        B, P, _ = x.shape

        pos = x[:, :, :7].clone() # x, y, cos, sin
        # static: [0,1,0]
        pos[..., -3:] = 0.0
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
    def __init__(self, lane_len, drop_path_rate=0.3, hidden_dim=192, depth=3, tokens_mlp_dim=64, channels_mlp_dim=128):
        super().__init__()

        self._lane_len = lane_len
        self._channel = channels_mlp_dim

        self.speed_limit_emb = nn.Linear(1, channels_mlp_dim)
        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8, hidden_features=channels_mlp_dim, out_features=channels_mlp_dim, act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=lane_len, hidden_features=tokens_mlp_dim, out_features=tokens_mlp_dim, act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate) for i in range(depth)])

        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim, hidden_features=hidden_dim, out_features=hidden_dim, act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x, speed_limit, has_speed_limit):
        '''
        x: B, P, V, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, P, 1
        has_speed_limit: B, P, 1
        '''
        traffic = x[:, :, 0, 8:]
        x = x[..., :8]

        pos = x[:, :, int(self._lane_len / 2), :7].clone() # x, y, x'-x, y'-y
        heading = torch.atan2(pos[..., 3], pos[..., 2])
        pos[..., 2] = torch.cos(heading)
        pos[..., 3] = torch.sin(heading)
        # lane: [0,0,1]
        pos[..., -3:] = 0.0
        pos[..., -1] = 1.0

        B, P, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
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
        speed_limit_embedding = torch.zeros((speed_limit.shape[0], self._channel), device=x.device)

        if has_speed_limit.sum() > 0:
            speed_limit_with_limit = self.speed_limit_emb(speed_limit[has_speed_limit].unsqueeze(-1))
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1
            )
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(traffic)  # Traffic light embedding for valid data


        x = x + speed_limit_embedding + traffic_light_embedding
        x = self.emb_project(self.norm(x))

        x_result = torch.zeros((B * P, x.shape[-1]), device=x.device)
        x_result[valid_indices] = x  # Fill in valid parts
        
        return x_result.view(B, P, -1) , mask_p.reshape(B, -1), pos.view(B, P, -1)


class FusionEncoder(nn.Module):
    def __init__(self, hidden_dim=192, num_heads=6, drop_path_rate=0.3, depth=3, device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        self.blocks = nn.ModuleList(
            [SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr) for i in range(depth)]
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, mask):

        mask[:, 0] = False

        for b in self.blocks:
            x = b(x, mask)

        return self.norm(x)