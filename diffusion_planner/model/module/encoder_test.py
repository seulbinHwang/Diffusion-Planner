""" 아래 전체 클래스, 내가 직접 설계해서 구현해본거야. 구현 상에 버그가 있는지? 내 의도대로 동작하지 않게 잘못 구현된 부분이 있는지? 매우 냉철하고 비판적으로 검토해줘! """

from timm.models.layers import Mlp
from timm.layers import DropPath
import torch.nn.functional as F

from diffusion_planner.model.module.mixer import MixerBlock

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn


def encode_time_with_fourier_features(
    t_sec: torch.Tensor,
    time_min: float,
    time_max: float,
    num_fourier_frequencies: int,
) -> torch.Tensor:
    """시간 값을 Fourier(2K) + 중심화 시간 스칼라(1)로 인코딩합니다(AMP 안전).

    Args:
        t_sec (torch.Tensor): (...,) 형태의 초 단위 시간 텐서.
        time_min (float): 시간 최소값.
        time_max (float): 시간 최대값.
        num_fourier_frequencies (int): K, 사용할 주파수 개수.

    Returns:
        torch.Tensor: (..., 2K + 1) 형태의 인코딩 텐서. 채널 구성은
            [cos(·), sin(·), t_hat]이며, 앞의 2K채널이 Fourier 성분,
            마지막 1채널이 중심화된 시간 스칼라(t_hat)입니다.

    Note:
        - 삼각함수 계산은 fp32에서 수행 후, 최종적으로 입력 텐서 dtype으로 캐스팅합니다.
        - 0으로 나누기 방지를 위해 작은 epsilon을 분모에 더합니다.
    """
    eps: float = 1e-6

    # 정규화된 시간: (...,)
    t_norm: torch.Tensor = (t_sec - time_min) / (time_max - time_min + eps
                                                )  # (...,)
    t_norm: torch.Tensor = t_norm.clamp(0.0, 1.0).to(torch.float32)  # (...,)

    # 주파수 인덱스: (K,)
    k_indices: torch.Tensor = torch.arange(1,
                                           num_fourier_frequencies + 1,
                                           device=t_sec.device,
                                           dtype=torch.float32)  # (K,)

    # 각도: (..., K)
    angles: torch.Tensor = t_norm.unsqueeze(
        -1) * k_indices * torch.pi  # (..., K)

    # Fourier 성분: (..., 2K)
    fourier: torch.Tensor = torch.cat(
        [torch.cos(angles), torch.sin(angles)], dim=-1)  # (..., 2K)

    # 중심화 시간 스칼라: (..., 1),  t_hat = 2 * t_norm - 1
    t_scalar: torch.Tensor = (2.0 * t_norm - 1.0).unsqueeze(-1)  # (..., 1)

    # 최종 인코딩: (..., 2K + 1)  [cos | sin | t_hat]
    out: torch.Tensor = torch.cat([fourier, t_scalar],
                                  dim=-1).to(t_sec.dtype)  # (..., 2K + 1)
    return out


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
    # (total_steps,) = [total_steps-1, ..., 1, 0]
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
    반환 모양: (B, total_steps)
    마지막 축(시간축): [dt, 2*dt, ..., N*dt]
    항상 독립 메모리 텐서(복사본)를 반환합니다.
    """
    if total_steps <= 0:
        raise ValueError("total_steps must be >= 1")
    if B <= 0:
        raise ValueError("B must be >= 1")

    # (N,) = [1, 2, ..., N]
    idx = torch.arange(1, total_steps + 1, device=device, dtype=dtype)
    base = float(dt) * idx  # [dt, 2dt, ..., N*dt]

    # (1,N) → (B, N) 확장 후 clone()으로 복사본 보장
    return base.view(1, total_steps).expand(B, total_steps).clone()


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.chunk_length = 10
        self.num_fourier_frequencies = 4
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.time_gap = 0.1
        self.time_min = -(config.time_len - 1) * self.time_gap
        self.time_max = config.future_len * self.time_gap
        self.agents_encoder = AgentFusionEncoder(
            config.time_len,
            config.future_len,
            chunk_length=self.chunk_length,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        self.static_encoder = StaticFusionEncoder(
            config.static_objects_state_dim,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        self.token_num = (1 * self.agents_encoder.future_chunk_num) + (
            (1 + config.agent_num) * self.agents_encoder.past_cur_chunk_num
        ) + config.static_objects_num + config.lane_num

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device)

        # position embedding encode
        # x, y, cos, sin,
        # time : 2 * num_fourier_frequencies + 1
        # diff to current time position : 4
        # type (ego, neighbor, static, lane)
        self.pos_emb = nn.Linear(
            4 + 2 * self.num_fourier_frequencies + 1 + 4 + 4, config.hidden_dim)

    def _sample_uniform_prefix_lengths(self, batch_size: int,
                                       max_future_len: int,
                                       device: torch.device) -> torch.Tensor:
        """무작위 길이 M_i를 각 배치별로 균일 분포에서 샘플링합니다.

        Args:
            batch_size (int): 배치 크기 B.
            max_future_len (int): 전체 미래 길이 N(예: 80).
            device (torch.device): 결과 텐서를 올릴 디바이스.

        Returns:
            torch.Tensor: [B] 형태의 정수 텐서. 각 값은 M_i ∈ {0,1,...,N}.
        """
        # shape: (B,)
        # 낮은 값 포함(0), 높은 값은 제외 → high = N+1 로 설정해 {0..N} 범위
        prefix_lengths: torch.Tensor = torch.randint(low=0,
                                                     high=max_future_len + 1,
                                                     size=(batch_size,),
                                                     device=device)
        return prefix_lengths

    def _build_known_mask_from_lengths(self, prefix_lengths: torch.Tensor,
                                       max_future_len: int) -> torch.Tensor:
        """길이 M_i로부터 '알려진 구간(조건 제공)' 마스크를 만듭니다.

        정의:
            known_mask[b, t] = True  ⇔  t < M_b
            (여기서 t는 0..N-1 인덱스이며, 시간은 0.1s,1.0s 등과 매핑 가능)

        Args:
            prefix_lengths (torch.Tensor): [B] 각 배치의 M_i.
            max_future_len (int): 전체 미래 길이 N.

        Returns:
            torch.Tensor: [B, N]의 bool 텐서. True=알려진(조건), False=미제공.
        """
        # time_index: [1, N] → [0..N-1] 인덱스
        time_index = torch.arange(max_future_len,
                                  device=prefix_lengths.device)  # (N,)
        # 브로드캐스트 비교 → (B, N)
        known_mask = time_index.unsqueeze(0) < prefix_lengths.unsqueeze(1)
        return known_mask  # (B, N), bool

    def _truncate_and_pad_ego_future_for_encoder(
            self, ego_future_full: torch.Tensor,
            known_mask: torch.Tensor) -> torch.Tensor:
        """미래 궤적을 [처음 M_i 스텝 유지 + 나머지 0패딩]으로 변환합니다.

        주의:
        - 에이전트 인코더의 assert(길이 고정)를 만족시키기 위해 **길이는 그대로 N 유지**합니다.
        - 마스크 판단은 에이전서 내부에서 첫 8채널이 0인지로 이루어지므로,
          M_i 이후 프레임은 **앞 8채널을 0**으로 채웁니다.
        - 마지막 3채널(type)은 항상 ego를 의미하도록 **[1,0,0]**(또는 입력의 ego one-hot)을 유지합니다.

        Args:
            ego_future_full (torch.Tensor):
                형태: [B, N, 11]
                채널: [x, y, cos, sin, vx, vy, w, l, type(3)]
            known_mask (torch.Tensor):
                형태: [B, N], bool
                True=조건 제공(유지), False=미제공(패딩)

        Returns:
            torch.Tensor:
                형태: [B, N, 11]
                처음 M_i는 원본 유지, 나머지는 앞 8채널 0, type 3채널은 ego one-hot 유지.
        """
        B, N, D = ego_future_full.shape  # (B, N, 11)
        assert D == 11, "ego_future_full의 마지막 차원은 11이어야 합니다."

        # (B, 3) ego 타입 벡터를 첫 프레임에서 추출(이미 one-hot이라 가정)
        ego_type: torch.Tensor = ego_future_full[:, 0, 8:11].clone()  # (B, 3)

        # 기본 패딩 텐서: 앞 8채널=0, type 3채널=ego one-hot 반복
        padded_default = ego_future_full.new_zeros(B, N, D)  # (B, N, 11)
        padded_default[:, :, 8:11] = ego_type.unsqueeze(1).expand(-1, N, -1)

        # keep: (B, N, 1)  True=원본 유지
        keep = known_mask.unsqueeze(-1)

        # 최종: 알려진 구간은 원본, 나머지는 기본 패딩
        ego_future_masked = torch.where(keep, ego_future_full,
                                        padded_default)  # (B, N, 11)
        return ego_future_masked

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인코더 전방 패스(무작위 길이 M로 ego 미래를 잘라 조건 제공).

        입력 딕셔너리 키와 텐서 형태:
            - ego_agent_past:           [B, V=21, 11]
            - ego_future_gt_11_dim:     [B, N=80, 11]
            - neighbor_agents_past:     [B, A, V=21, 11]
            - static_objects:           [B, P, D_static]
            - lanes:                    [B, L, lane_len, D_lane]
            - lanes_speed_limit:        [B, L, 1]
            - lanes_has_speed_limit:    [B, L, 1]

        처리 개요:
            1) 배치별로 M_i ~ Uniform{0..N} 샘플.
            2) ego 미래를 처음 M_i만 남기고, 나머지 스텝은 앞 8채널 0으로 패딩.
               (길이는 N을 유지하여 에이전트 인코더와 호환)
            3) 나머지 인코딩/융합은 기존과 동일.

        Returns:
            Dict[str, torch.Tensor]:
                - 'encoding':        [B, T_token, H]
                - 'ego_fut_global':  [B, H]
                - 'ego_plan_known_mask': [B, N]  (True=조건 제공)
                - 'ego_plan_prefix_lengths': [B] (각 배치의 M_i)
        """
        encoder_outputs = {}
        # ego
        ego_past = inputs["ego_agent_past"]  # (B, V=21, D=11) -> (B, 1, V, D)
        ego_past = ego_past.unsqueeze(1)  # Add a dimension for P

        ego_future_full = inputs[
            "ego_future_gt_11_dim"]  # (B ,future_len= 80, 11)
        # agents
        neighbors = inputs['neighbor_agents_past']

        # static objects
        static = inputs['static_objects']

        # vector maps
        lanes = inputs['lanes']
        lanes_speed_limit = inputs['lanes_speed_limit']
        lanes_has_speed_limit = inputs['lanes_has_speed_limit']

        B = neighbors.shape[0]
        future_len: int = ego_future_full.shape[1]

        # ---------------------- 1) M_i 샘플링 ---------------------- #
        prefix_lengths = self._sample_uniform_prefix_lengths(
            batch_size=B,
            max_future_len=future_len,
            device=ego_future_full.device,
        )  # (B,)
        known_mask = self._build_known_mask_from_lengths(
            prefix_lengths,
            max_future_len=future_len)  # (B, future_len) True=조건 제공

        # ---------------------- 2) 잘라 + 패딩 ---------------------- #
        ego_future_masked = self._truncate_and_pad_ego_future_for_encoder(
            ego_future_full,
            known_mask)  # (B, future_len, 11)  길이 유지, 마스크는 내부에서 활용됨

        # ---------------------- 3) 인코딩 ---------------------- #
        # agents_mask: (B, agents_num * past_cur_chunk_num + future_chunk_num)
        # ego_fut_global: (B, hidden_dim)
        (encoding_agents, agents_mask, agents_pos,
         ego_fut_global) = self.agents_encoder(ego_past, neighbors,
                                               ego_future_masked)
        encoding_static, static_mask, static_pos = self.static_encoder(static)
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit)

        # ---------------------- 4) 포지션 임베딩 결합 ---------------------- #
        encoding_input = torch.cat(
            [encoding_agents, encoding_static, encoding_lanes], dim=1)
        encoding_pos = torch.cat([agents_pos, static_pos, lane_pos],
                                 dim=1).view(B * self.token_num, -1)
        encoding_mask = torch.cat([agents_mask, static_mask, lanes_mask],
                                  dim=1).view(-1)
        encoding_pos = self.pos_emb(encoding_pos[~encoding_mask])
        encoding_pos_result = torch.zeros((B * self.token_num, self.hidden_dim),
                                          device=encoding_pos.device)
        encoding_pos_result[
            ~encoding_mask] = encoding_pos  # Fill in valid parts

        encoding_input = encoding_input + self.pos_scale * encoding_pos_result.view(
            B, self.token_num, -1)

        encoder_outputs['encoding'] = self.fusion(
            encoding_input, encoding_mask.view(B, self.token_num))
        encoder_outputs["ego_fut_global"] = ego_fut_global

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
        x_norm = self.norm1(x)
        x = x + self.drop_path(
            self.attn(x_norm,
                      x_norm,
                      x_norm,
                      key_padding_mask=mask,
                      need_weights=False)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):

    def __init__(self,
                 time_len,
                 future_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128,
                 chunk_length=10,
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self.future_len = future_len
        self.chunk_length = chunk_length
        self.past_cur_chunk_num = max(1, time_len // self.chunk_length)
        self.future_chunk_num = max(1, future_len // self.chunk_length)
        self.num_fourier_frequencies = num_fourier_frequencies
        num_fourier_dim = 2 * self.num_fourier_frequencies + 1  # 2K + 1

        # 게이트드 풀링에 필요한 선형층 정의
        self.gate_linear_W = nn.Linear(channels_mlp_dim, channels_mlp_dim)
        self.gate_linear_V = nn.Linear(channels_mlp_dim, tokens_mlp_dim)
        self.value_linear = nn.Linear(channels_mlp_dim, channels_mlp_dim)

        self._hidden_dim = hidden_dim
        self.tokens_mlp_dim = tokens_mlp_dim

        self.type_scale = nn.Parameter(torch.tensor(0.01))  # 스케일 조정용
        self.type_emb = nn.Linear(3, channels_mlp_dim)

        self.channel_pre_project = Mlp(in_features=8 + num_fourier_dim + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
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

        # 각 미래 chunk를 스칼라 로짓으로
        self.ego_fut_pool_q = nn.Linear(hidden_dim, 1)
        # 잔차 스케일: 0으로 시작(초기엔 평균만), 학습되며 켜짐
        self.ego_fut_pool_scale = nn.Parameter(torch.tensor(0.0))

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

    def _filter_on_agents_past_cur(
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

    def _filter_on_ego_future(
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
    ) -> torch.Tensor:  # (on_all_time_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len, 10 + 2k)
        # (agents_past_cur_on_num * time_len, 10+2k)
        on_agents_past_cur = on_agents_past_cur.view(
            -1, on_agents_past_cur.shape[-1])
        # (ego_future_on_num * future_len, 10 + 2k)
        on_ego_future = on_ego_future.view(-1, on_ego_future.shape[-1])
        # (on_all_time_num = agents_past_cur_on_num * time_len + on_ego_future_on_num * future_len, 10 + 2k)
        on_all = torch.cat([on_agents_past_cur, on_ego_future], dim=0)
        return on_all

    @staticmethod
    def _compute_equal_chunks(seq_len: int, chunk_num: int,
                              device) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return starts.to(device), ends.to(device)

    def _gated_attentive_pool(
            self,
            chunk_values: torch.Tensor,  # (N, L, C)
            chunk_off_points_mask: torch.Tensor,  # (N, L)
            chunk_is_off: torch.Tensor  # (N)
    ) -> torch.Tensor:
        """게이트드 어텐션 풀링을 **tokens_mlp_dim개 쿼리**로 일반화하여 (N, tokens_mlp_dim, C) 출력을 반환한다.
        Args:
            chunk_values (torch.Tensor):
                - shape: (N, L, C)
                - 의미: 구간 내부 L개 시점의 채널 임베딩(이미 per-timestep proj를 통과한 값)
            chunk_off_points_mask (torch.Tensor):
                - shape: (N, L)   # True=무효(해당 프레임 제외)
            chunk_is_off: shape: (N,) # True=해당 구간에 유효 시점 0개

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

        ##############3
        # 3) FP16/FP32 안전 마스킹(softmax 전)
        logits = logits.float()  # 안정성 위해 fp32로
        logits = logits.masked_fill((chunk_off_points_mask).unsqueeze(-1),
                                    float('-inf'))  # 유효 아님 → -inf
        if chunk_is_off.any():
            # 전부 -inf이면 softmax NaN → 해당 행은 0으로 세팅해 NaN 회피(균등분포가 됨)
            logits[chunk_is_off] = 0.0

        # 4) 소프트맥스 & 전부 마스크된 행은 0으로 고정(gradient도 0)
        attn = F.softmax(logits, dim=1).to(chunk_values.dtype)  # (N,L,Q) fp32
        attn = attn.masked_fill(chunk_is_off.view(-1, 1, 1), 0.0)
        # if chunk_is_off.any():
        #     attn[chunk_is_off] = 0.0
        ###############

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
        on_trajs: torch.Tensor,
        on_trajs_off_p_mask: torch.Tensor,
        chunk_starts: torch.Tensor,
        chunk_ends: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """균등 분할된 각 구간에 대해 **집계 토큰**과 **구간 마스크**를 만든다.

        Args:
            on_trajs (torch.Tensor):
                - shape: (N, V, C)   # 유효 에이전트 수 N, 시점 V, 채널 C
            on_trajs_off_p_mask (torch.Tensor):
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
        trajs_on_num, _, channel_mlp_dim = on_trajs.shape
        chunks_num = chunk_starts.numel()

        chunks = on_trajs.new_zeros(
            trajs_on_num, chunks_num, self.tokens_mlp_dim,
            channel_mlp_dim)  # (N, chunk_num, tokens_mlp_dim, channel_mlp_dim)
        invalid_chunk_mask = torch.zeros(
            trajs_on_num, chunks_num, dtype=torch.bool,
            device=on_trajs.device)  # (N, chunk_num)

        for chunk_idx in range(chunks_num):
            start, end = int(chunk_starts[chunk_idx].item()), int(
                chunk_ends[chunk_idx].item())
            # chunk_values: (N, L, channel_mlp_dim)
            chunk_values = on_trajs[:, start:end + 1, :]
            # chunk_off_points_mask: (N, L)
            chunk_off_points_mask = on_trajs_off_p_mask[:, start:end + 1]
            # chunk_is_off: (N,)
            chunk_is_off = (chunk_off_points_mask == False).sum(dim=1) == 0
            invalid_chunk_mask[:, chunk_idx] = chunk_is_off
            # chunk_token: (N, tokens_mlp_dim, channel_mlp_dim)
            chunk_token = self._gated_attentive_pool(
                chunk_values, chunk_off_points_mask,
                chunk_is_off)  # (N, tokens_mlp_dim, channel_mlp_dim)
            # (N, chunk_num, tokens_mlp_dim, channel_mlp_dim)
            chunks[:, chunk_idx, :, :] = chunk_token
        # (N, chunk_num, tokens_mlp_dim, channel_mlp_dim), (N, chunk_num)
        return chunks, invalid_chunk_mask

    def _get_on_past_cur_on_chunk_num_sized_agents_type(
        self,
        agents_type: torch.Tensor,  # (B, agents_num, 3)
        agents_past_cur_on_mask: torch.Tensor,  # (B * agents_num)
        on_agents_past_cur_on_chunk_mask: torch.
        Tensor,  # (agents_past_cur_on_num * past_cur_chunk_num)
        past_cur_chunk_num: int
    ) -> torch.Tensor:  # (on_past_cur_on_chunk_num, 3)
        # (B * agents_num, 3)
        B, agents_num = agents_type.shape[:2]
        agents_type = agents_type.view(B * agents_num, -1)
        # (agents_past_cur_on_num, 3)
        agents_type = agents_type[agents_past_cur_on_mask]
        agents_past_cur_on_num = agents_type.shape[0]
        # (agents_past_cur_on_num, 3) -> (agents_past_cur_on_num, 1, 3)
        # -> (agents_past_cur_on_num, past_cur_chunk_num, 3)
        agents_type = agents_type.unsqueeze(1).expand(-1, past_cur_chunk_num,
                                                      -1)
        # (agents_past_cur_on_num, past_cur_chunk_num, 3)
        # -> (agents_past_cur_on_num * past_cur_chunk_num, 3)
        agents_type = agents_type.view(
            agents_past_cur_on_num * past_cur_chunk_num, -1)
        # (agents_past_cur_on_num * past_cur_chunk_num, 3) -> (on_past_cur_on_chunk_num, 3)
        agents_type = agents_type[on_agents_past_cur_on_chunk_mask]
        return agents_type

    def _get_on_ego_fut_on_chunk_num_sized_ego_fut_type(
        self,
        ego_fut_type: torch.Tensor,  # (B, 3)
        ego_future_on_mask: torch.Tensor,  # (B)
        on_ego_fut_on_chunk_mask: torch.
        Tensor,  # (on_ego_fut_on_num * future_chunk_num)
        future_chunk_num: int
    ) -> torch.Tensor:
        # ego_fut_type: (B, 3) -> (ego_future_on_num, 3)
        ego_fut_type = ego_fut_type[ego_future_on_mask]
        ego_future_on_num = ego_fut_type.shape[0]
        # (ego_future_on_num, 3) -> (ego_future_on_num, 1, 3)
        # -> (ego_future_on_num, future_chunk_num, 3)
        ego_fut_type = ego_fut_type.unsqueeze(1).expand(-1, future_chunk_num,
                                                        -1)
        # (ego_future_on_num, future_chunk_num, 3)
        # -> (ego_future_on_num * future_chunk_num, 3)
        ego_fut_type = ego_fut_type.view(ego_future_on_num * future_chunk_num,
                                         -1)
        # (ego_future_on_num * future_chunk_num, 3)
        # -> (on_ego_fut_on_chunk_num, 3)
        ego_fut_type = ego_fut_type[on_ego_fut_on_chunk_mask]
        return ego_fut_type

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
        # agents_type: (on_past_cur_on_chunk_num, 3)
        agents_type = self._get_on_past_cur_on_chunk_num_sized_agents_type(
            agents_type, agents_past_cur_on_mask,
            on_agents_past_cur_on_chunk_mask, past_cur_chunk_num)
        # ego_fut_type: (on_ego_fut_on_chunk_num, 3)
        ego_fut_type = self._get_on_ego_fut_on_chunk_num_sized_ego_fut_type(
            ego_fut_type, ego_future_on_mask, on_ego_fut_on_chunk_mask,
            future_chunk_num)
        # (on_all_on_chunk_num, 3)
        agents_ego_fut_type = torch.cat([agents_type, ego_fut_type], dim=0)

        agents_ego_fut_type_emb = self.type_emb(agents_ego_fut_type)
        return agents_ego_fut_type_emb

    def _fill_on_chunk_to_on_agent(
            self,
            on_all_on_chunk: torch.Tensor,  # (on_all_on_chunk_num, hidden_dim)
            on_agents_past_cur_on_chunk_mask: torch.
        Tensor,  # (agents_past_cur_on_num * past_cur_chunk_num)
            on_ego_fut_on_chunk_mask: torch.
        Tensor,  # (ego_future_on_num * future_chunk_num)
            on_past_cur_on_chunk_num: int,
            agents_past_cur_on_num: int,
            ego_future_on_num: int,
            past_cur_chunk_num: int,
            future_chunk_num: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # (on_past_cur_on_chunk_num, hidden_dim)
        on_agents_past_cur_on_chunk = on_all_on_chunk[:
                                                      on_past_cur_on_chunk_num, :]
        on_agents_past_cur_chunk = torch.zeros(
            (agents_past_cur_on_num * past_cur_chunk_num, self._hidden_dim),
            device=on_all_on_chunk.device)
        # on_agents_past_cur_on_chunk_mask: (agents_past_cur_on_num * past_cur_chunk_num)
        on_agents_past_cur_chunk[
            on_agents_past_cur_on_chunk_mask] = on_agents_past_cur_on_chunk
        # (agents_past_cur_on_num, past_cur_chunk_num, hidden_dim)
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.view(
            agents_past_cur_on_num, past_cur_chunk_num, self._hidden_dim)

        # (on_ego_fut_on_chunk_num, hidden_dim)
        on_ego_fut_on_chunk = on_all_on_chunk[on_past_cur_on_chunk_num:, :]
        on_ego_fut_chunk = torch.zeros(
            (ego_future_on_num * future_chunk_num, self._hidden_dim),
            device=on_all_on_chunk.device)
        # on_ego_fut_on_chunk_mask: (ego_future_on_num * future_chunk_num)
        on_ego_fut_chunk[on_ego_fut_on_chunk_mask] = on_ego_fut_on_chunk
        # (ego_future_on_num, future_chunk_num, hidden_dim)
        on_ego_fut_chunk = on_ego_fut_chunk.view(ego_future_on_num,
                                                 future_chunk_num,
                                                 self._hidden_dim)
        return on_agents_past_cur_chunk, on_ego_fut_chunk

    def _get_all_off_chunk_mask(
        self,
        agents_past_cur_on_mask: torch.Tensor,
        on_agents_past_cur_off_chunk_mask: torch.Tensor,
        ego_future_on_mask: torch.Tensor,
        on_ego_fut_off_chunk_mask: torch.Tensor,
    ) -> torch.Tensor:
        """과거/현재(agents)와 미래(ego)의 구간 무효(True) 마스크를 (B, A*M_past + M_future)로 결합해 반환.

        개요:
            - 각각의 구간 무효(오프) 마스크를 적절히 브로드캐스트/리쉐이프/대입하여
              배치 단위 `(B, agents_num * past_cur_chunk_num + future_chunk_num)`의
              **최종 무효 마스크** `all_off_chunk_mask`를 만든다.

        Args:
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
            torch.Tensor
                - all_off_chunk_mask:
                    * shape: (B, agents_num * past_cur_chunk_num + future_chunk_num)
                    * dtype: torch.bool
                    * 의미: **True=무효(패딩)** 키 패딩 마스크. 위 all_chunk의 각 토큰과 1:1 대응.

        Note:
            - `agents_num`은 `agents_past_cur_chunk.shape[1] // past_cur_chunk_num` 로 유추합니다.
            - 디바이스/ dtype은 입력 텐서들에서 자동으로 맞춥니다.
        """
        # -------------------- 기본 치수/디바이스 유추 --------------------
        B = ego_future_on_mask.shape[0]
        B_agents_num = agents_past_cur_on_mask.shape[0]
        agents_num = B_agents_num // B  # 에이전트 수
        past_cur_chunk_num = on_agents_past_cur_off_chunk_mask.shape[1]
        future_chunk_num = on_ego_fut_off_chunk_mask.shape[1]
        device = agents_past_cur_on_mask.device

        # -------------------- 1) 에이전트 과거/현재 구간 무효 마스크 구성 --------------------
        # (B*agents_num, past_cur_chunk_num), 기본 True=무효로 초기화로 하고, 유효한 chunk 찾아서 False로 덮어쓰기
        agents_past_cur_off_chunk_mask = torch.ones(
            B_agents_num,
            past_cur_chunk_num,
            dtype=torch.bool,
            device=device,
        )  # True=무효(기본)

        # 유효 에이전트 위치에 대해 실제 구간 무효 마스크를 덮어쓰기
        # agents_past_cur_on_mask: (B * agents_num), True=유효
        # on_agents_past_cur_off_chunk_mask: (agents_past_cur_on_num, past_cur_chunk_num), True=무효
        agents_past_cur_off_chunk_mask[
            agents_past_cur_on_mask] = on_agents_past_cur_off_chunk_mask

        # (B, agents_num * past_cur_chunk_num)로 펴기
        agents_past_cur_off_chunk_mask = agents_past_cur_off_chunk_mask.view(
            B, agents_num, past_cur_chunk_num).reshape(
                B, agents_num * past_cur_chunk_num)  # (B, A*M_past)

        # -------------------- 2) ego 미래 구간 무효 마스크 구성 --------------------
        # (B, future_chunk_num), 기본 True=무효로 초기화하고, 유효한 chunk 찾아서 False로 덮어쓰기
        ego_fut_off_chunk_mask = torch.ones(
            B,
            future_chunk_num,
            dtype=torch.bool,
            device=device,
        )  # (B, M_future), True=무효(기본)

        # 유효 배치(ego_future_on_mask=True)에 대해 실제 구간 무효 마스크를 덮어쓰기
        # on_ego_fut_off_chunk_mask: (ego_future_on_num, future_chunk_num), True=무효
        ego_fut_off_chunk_mask[
            ego_future_on_mask] = on_ego_fut_off_chunk_mask  # (B, M_future)

        # -------------------- 3) 최종 마스크 결합 --------------------
        all_off_chunk_mask = torch.cat(
            [agents_past_cur_off_chunk_mask, ego_fut_off_chunk_mask],
            dim=1)  # (B, agents_num * past_cur_chunk_num + future_chunk_num)

        return all_off_chunk_mask

    def _get_agent_past_cur_chunks_delta(
        self,
        agents_past_cur_chunks_pos_time: torch.
        Tensor,  # (B, agents_num, past_cur_chunk_num, _)
        agent_cur_xyyaw: torch.Tensor  # (B, agents_num, 4 )
    ) -> torch.Tensor:  # (B, agents_num, past_cur_chunk_num, 4)
        # (B, agents_num, past_cur_chunk_num, 2)
        agents_past_cur_center_xy = agents_past_cur_chunks_pos_time[:, :, :, :2]
        # (B, agents_num, past_cur_chunk_num)
        agents_past_cur_center_cos = agents_past_cur_chunks_pos_time[:, :, :, 2]
        agents_past_cur_center_sin = agents_past_cur_chunks_pos_time[:, :, :, 3]

        # (B, agents_num, 4) # x, y, cos(yaw), sin(yaw)
        # (B, agents_num, 2) # x, y
        agent_cur_xy = agent_cur_xyyaw[:, :, :2]
        # (B, agents_num)
        agent_cur_cos = agent_cur_xyyaw[:, :, 2]
        agent_cur_sin = agent_cur_xyyaw[:, :, 3]

        # ---------- 4) Δyaw의 cos/sin (항등식 직계산) ----------
        # cos(Δ) =  agents_past_cur_center_cos*agent_cur_cos + agents_past_cur_center_sin*agent_cur_sin
        # sin(Δ) =  agents_past_cur_center_sin*agent_cur_cos - agents_past_cur_center_cos*agent_cur_sin
        # cos_d: (B, agents_num, past_cur_chunk_num)
        # sin_d: (B, agents_num, past_cur_chunk_num)
        cos_d = agents_past_cur_center_cos * agent_cur_cos.unsqueeze(
            2) + agents_past_cur_center_sin * agent_cur_sin.unsqueeze(2)
        sin_d = agents_past_cur_center_sin * agent_cur_cos.unsqueeze(
            2) - agents_past_cur_center_cos * agent_cur_sin.unsqueeze(2)
        agents_past_cur_center_delta_cos_sin = torch.cat(
            [cos_d.unsqueeze(-1), sin_d.unsqueeze(-1)],
            dim=-1)  # (B, agents_num, past_cur_chunk_num, 2)

        agents_past_cur_center_delta_xy = agents_past_cur_center_xy - agent_cur_xy.unsqueeze(
            2)  # (B, agents_num, past_cur_chunk_num, 2)
        # (B, agents_num, past_cur_chunk_num, 2)

        agent_past_cur_chunks_delta = torch.cat(
            [
                agents_past_cur_center_delta_xy,
                agents_past_cur_center_delta_cos_sin
            ],
            dim=-1)  # (B, agents_num, past_cur_chunk_num, 4)

        return agent_past_cur_chunks_delta  # (B, agents_num, past_cur_chunk_num, 4)

    def _get_agents_past_cur_chunks_type(
            self, agents_past_cur_chunks_feature: torch.Tensor) -> torch.Tensor:
        # (B, agents_num, past_cur_chunk_num, 4 + 2k + 1 + 4)
        B, agents_num, past_cur_chunk_num = agents_past_cur_chunks_feature.shape[:
                                                                                 3]
        agents_past_cur_chunks_type = torch.zeros(
            (B, agents_num, past_cur_chunk_num, 4),
            device=agents_past_cur_chunks_feature.device,
            dtype=agents_past_cur_chunks_feature.dtype,
        )
        agents_past_cur_chunks_type[:, 0, :, 0] = 1.0  # ego
        agents_past_cur_chunks_type[:, 1:, :, 1] = 1.0  # neighbor
        return agents_past_cur_chunks_type  # (B, agents_num, past_cur_chunk_num, 4)

    def _get_agents_past_cur_chunks_pos_time(
        self,
        past_chunk_start_idx: torch.Tensor,  # (past_cur_chunk_num,)
        past_chunk_end_idx: torch.Tensor,  # (past_cur_chunk_num,)
        agents_past_cur_off_p_mask: torch.Tensor,  # (B, agents_num, time_len)
        agents_past_cur_xyyaw_time: torch.
        Tensor  # (B, agents_num, time_len, 4 + 2K + 1
    ) -> torch.Tensor:  # (B, agents_num, past_cur_chunk_num, 4 + 2K + 1)
        B, agents_num, time_len, feat_dim = agents_past_cur_xyyaw_time.shape
        past_cur_chunk_num: int = past_chunk_start_idx.numel()
        device = agents_past_cur_xyyaw_time.device
        dtype_idx = torch.long

        # ----- 1) 각 chunk별 center 인덱스 계산 (유효 구간 중앙) -----
        # center_idx: (B, agents_num, past_cur_chunk_num) [long]
        center_idx = torch.empty(B,
                                 agents_num,
                                 past_cur_chunk_num,
                                 dtype=dtype_idx,
                                 device=device)

        for chunk_idx in range(past_cur_chunk_num):
            start_idx = int(past_chunk_start_idx[chunk_idx].item())
            end_idx = int(past_chunk_end_idx[chunk_idx].item())
            L = end_idx - start_idx + 1

            # valid: (B, agents_num, L)   True=유효
            valid = ~agents_past_cur_off_p_mask[:, :, start_idx:end_idx + 1]

            # 상대 인덱스 t: (B, agents_num, L)
            t = torch.arange(L, device=device,
                             dtype=dtype_idx).view(1, 1,
                                                   L).expand(B, agents_num, L)

            # 유효한 곳의 첫/마지막 인덱스 찾기
            big = (L + 1)
            first_true = torch.where(valid, t, torch.full_like(t, big)).amin(
                dim=-1)  # (B, agents_num)
            last_true = torch.where(valid, t, torch.full_like(t, -1)).amax(
                dim=-1)  # (B, agents_num)

            has_valid = last_true >= 0  # (B, agents_num)
            center_rel = (first_true + last_true) // 2  # (B, agents_num)
            center_abs = center_rel + start_idx  # (B, agents_num)

            # 유효 프레임이 하나도 없는 경우 fallback: 산술 중앙
            default_center = torch.full_like(center_abs,
                                             (start_idx + end_idx) // 2)
            center_abs = torch.where(has_valid, center_abs,
                                     default_center)  # (B, agents_num)

            center_idx[:, :, chunk_idx] = center_abs

        # ----- 2) center 인덱스로 feature gather -----
        # gather용 인덱스: (B, agents_num, M, feat_dim)
        gather_idx = center_idx.unsqueeze(-1).expand(B, agents_num,
                                                     past_cur_chunk_num,
                                                     feat_dim)
        # centers: (B, agents_num, M, 4 + 2K + 1)
        agents_past_cur_chunks_pos_time = torch.gather(
            agents_past_cur_xyyaw_time, dim=2, index=gather_idx)

        return agents_past_cur_chunks_pos_time  # (B, agents_num, past_cur_chunk_num, 4 + 2K + 1)

    def _get_agents_past_cur_chunks_feature(
        self,
        past_chunk_start_idx: torch.Tensor,  # (past_cur_chunk_num,)
        past_chunk_end_idx: torch.Tensor,  # (past_cur_chunk_num,)
        agents_past_cur_off_p_mask: torch.Tensor,  # (B, agents_num, time_len)
        agents_past_cur_xyyaw_time: torch.
        Tensor  # (B, agents_num, time_len, 4 + 2K + 1)
    ) -> torch.Tensor:
        B, agents_num, time_len, _ = agents_past_cur_xyyaw_time.shape
        past_cur_chunk_num = past_chunk_start_idx.shape[0]
        ##########
        # (B, agents_num, past_cur_chunk_num, 4 + 2K + 1)
        agents_past_cur_chunks_pos_time = self._get_agents_past_cur_chunks_pos_time(
            past_chunk_start_idx, past_chunk_end_idx,
            agents_past_cur_off_p_mask, agents_past_cur_xyyaw_time)

        # (B, agents_num, 4)
        agent_cur_xyyaw = agents_past_cur_xyyaw_time[:, :, -1, :4].clone()
        # agent_past_cur_chunks_delta: # (B, agents_num, past_cur_chunk_num, 4)
        agent_past_cur_chunks_delta = self._get_agent_past_cur_chunks_delta(
            agents_past_cur_chunks_pos_time, agent_cur_xyyaw)
        # (B, agents_num, past_cur_chunk_num, 4 + 2K + 1 + 4)
        agents_past_cur_chunks_feature = torch.cat(
            [agents_past_cur_chunks_pos_time, agent_past_cur_chunks_delta],
            dim=-1)
        # (B, agents_num, past_cur_chunk_num, 4)
        agents_past_cur_chunks_type = self._get_agents_past_cur_chunks_type(
            agents_past_cur_chunks_feature)
        # (B, agents_num, past_cur_chunk_num, 4 + 2K + 1 + 4 + 4)
        agents_past_cur_chunks_feature = torch.cat(
            [agents_past_cur_chunks_feature, agents_past_cur_chunks_type],
            dim=-1)
        # (B, agents_num * past_cur_chunk_num, 4 + 2K + 1 + 4 + 4)
        agents_past_cur_chunks_feature = agents_past_cur_chunks_feature.view(
            B, agents_num * past_cur_chunk_num, -1)
        return agents_past_cur_chunks_feature

    def _get_ego_fut_chunks_pos_time(
        self,
        fut_chunk_start_idx: torch.Tensor,  # (future_chunk_num,)
        fut_chunk_end_idx: torch.Tensor,  # (future_chunk_num,)
        ego_future_off_p_mask: torch.Tensor,  # (B, future_len)
        ego_future_xyyaw_time: torch.Tensor  # (B, future_len, 4 + 2K + 1)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4 + 2K + 1)
        B, future_len, feat_dim = ego_future_xyyaw_time.shape
        future_chunk_num: int = fut_chunk_start_idx.numel()
        device = ego_future_xyyaw_time.device
        dtype_idx = torch.long

        # center_idx: (B, future_chunk_num) [long]
        center_idx = torch.empty(B,
                                 future_chunk_num,
                                 dtype=dtype_idx,
                                 device=device)

        for chunk_idx in range(future_chunk_num):
            start_idx = int(fut_chunk_start_idx[chunk_idx].item())
            end_idx = int(fut_chunk_end_idx[chunk_idx].item())
            L = end_idx - start_idx + 1

            # valid: (B, L)   True=유효
            valid = ~ego_future_off_p_mask[:, start_idx:end_idx + 1]

            # 상대 인덱스 t: (B, L)
            t = torch.arange(L, device=device,
                             dtype=dtype_idx).view(1, L).expand(B, L)

            # 유효한 곳의 첫/마지막 인덱스
            big = (L + 1)
            first_true = torch.where(valid, t,
                                     torch.full_like(t,
                                                     big)).amin(dim=-1)  # (B,)
            last_true = torch.where(valid, t,
                                    torch.full_like(t, -1)).amax(dim=-1)  # (B,)

            has_valid = last_true >= 0  # (B,)
            center_rel = (first_true + last_true) // 2  # (B,)
            center_abs = center_rel + start_idx  # (B,)

            # 유효 프레임이 하나도 없으면 산술 중앙으로 fallback
            default_center = torch.full_like(center_abs,
                                             (start_idx + end_idx) // 2)
            center_abs = torch.where(has_valid, center_abs,
                                     default_center)  # (B,)

            center_idx[:, chunk_idx] = center_abs

        # gather용 인덱스: (B, future_chunk_num, feat_dim)
        gather_idx = center_idx.unsqueeze(-1).expand(B, future_chunk_num,
                                                     feat_dim)
        # (B, future_chunk_num, 4 + 2K + 1)
        ego_fut_chunks_pos_time = torch.gather(ego_future_xyyaw_time,
                                               dim=1,
                                               index=gather_idx)

        return ego_fut_chunks_pos_time

    def _get_ego_fut_chunks_delta(
        self,
        ego_fut_center_pos_time: torch.Tensor,
        # (B, future_chunk_num, 4 + 2K + 1)
        ego_cur_xyyaw: torch.Tensor  # (B, 4)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4)

        # (B, future_chunk_num, 2)
        ego_fut_center_xy = ego_fut_center_pos_time[:, :, :2]
        # (B, future_chunk_num)
        ego_fut_center_cos = ego_fut_center_pos_time[:, :, 2]
        ego_fut_center_sin = ego_fut_center_pos_time[:, :, 3]

        # 현재 ego: (B, 2) / (B,)
        ego_cur_xy = ego_cur_xyyaw[:, :2]
        ego_cur_cos = ego_cur_xyyaw[:, 2]
        ego_cur_sin = ego_cur_xyyaw[:, 3]

        # Δyaw = yaw_center - yaw_cur 의 cos/sin 직계산
        # cos(Δ) = cos_c*cos_0 + sin_c*sin_0
        # sin(Δ) = sin_c*cos_0 - cos_c*sin_0
        cos_d = ego_fut_center_cos * ego_cur_cos.unsqueeze(
            1) + ego_fut_center_sin * ego_cur_sin.unsqueeze(1)
        sin_d = ego_fut_center_sin * ego_cur_cos.unsqueeze(
            1) - ego_fut_center_cos * ego_cur_sin.unsqueeze(1)

        # Δxy
        delta_xy = ego_fut_center_xy - ego_cur_xy.unsqueeze(
            1)  # (B, future_chunk_num, 2)

        # concat: (B, future_chunk_num, 4)
        ego_fut_chunks_delta = torch.cat(
            [delta_xy, cos_d.unsqueeze(-1),
             sin_d.unsqueeze(-1)], dim=-1)
        return ego_fut_chunks_delta

    def _get_ego_fut_chunks_type(
        self, ego_future_chunks_feature: torch.Tensor
        # (B, future_chunk_num, 4 + 2K + 1 + 4)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4)
        B, future_chunk_num = ego_future_chunks_feature.shape[:2]
        ego_fut_chunks_type = torch.zeros(
            (B, future_chunk_num, 4),
            device=ego_future_chunks_feature.device,
            dtype=ego_future_chunks_feature.dtype,
        )
        # ego one-hot
        ego_fut_chunks_type[:, :, 0] = 1.0
        return ego_fut_chunks_type

    def _get_ego_future_chunks_feature(
        self,
        fut_chunk_start_idx: torch.Tensor,  # (future_chunk_num)
        fut_chunk_end_idx: torch.Tensor,  # (future_chunk_num)
        ego_future_off_p_mask: torch.Tensor,  # (B, future_len)
        ego_future_xyyaw_time: torch.Tensor,  # (B, future_len, 4 + 2K + 1)
        ego_cur_xyyaw: torch.Tensor  # (B, 4)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4 + 2K + 1 + 4 + 4)
        B, future_len, _ = ego_future_xyyaw_time.shape
        future_chunk_num = fut_chunk_start_idx.shape[0]
        # (B, future_chunk_num, 4 + 2K + 1)
        ego_fut_chunks_pos_time = self._get_ego_fut_chunks_pos_time(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw_time)
        # ego_fut_chunks_delta: # (B, future_chunk_num, 4)
        ego_fut_chunks_delta = self._get_ego_fut_chunks_delta(
            ego_fut_chunks_pos_time, ego_cur_xyyaw)
        # (B, future_chunk_num, 4 + 2K + 1 + 4)
        ego_future_chunks_feature = torch.cat(
            [ego_fut_chunks_pos_time, ego_fut_chunks_delta], dim=-1)
        # (B, future_chunk_num, 4)
        ego_fut_chunks_type = self._get_ego_fut_chunks_type(
            ego_future_chunks_feature)
        # (B, future_chunk_num, 4 + 2K + 1 + 4 + 4)
        ego_future_chunks_feature = torch.cat(
            [ego_future_chunks_feature, ego_fut_chunks_type], dim=-1)
        return ego_future_chunks_feature

    def _add_timestep_to_agents_past_cur(
            self, agents_past_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # agents_past_current: (B, agents_num, time_len, d_8)
        B, agents_num, time_len, _ = agents_past_current.shape
        # agents_past_current_timestep: (B, agents_num, time_len)
        # [-2.0, -1.9, ..., 0.0]
        agents_past_current_timestep = timegrid_past_3d(
            dt=self.time_gap,
            total_steps=time_len,
            B=B,
            agents_num=agents_num,
            device=agents_past_current.device,
            dtype=agents_past_current.dtype)
        agent_past_current_time_fourier = encode_time_with_fourier_features(
            agents_past_current_timestep, self.time_min, self.time_max,
            self.num_fourier_frequencies)  # (B, agents_num, time_len, 2K + 1)
        agents_past_current = torch.cat(
            [agents_past_current, agent_past_current_time_fourier],
            dim=-1)  # (B, agents_num, time_len, 8 + 2K + 1)

        ######### FOR POSITIONAL EMBEDDING #########
        agents_past_cur_xyyaw = agents_past_current[:, :, :, :4].clone(
        )  # (B, agents_num, time_len, 4)
        agents_past_cur_xyyaw_time = torch.cat(
            [agents_past_cur_xyyaw, agent_past_current_time_fourier], dim=-1
        )  # agents_past_cur_xyyaw_time: (B, agents_num, time_len, 4 + 2K + 1)

        # (B, agents_num, time_len, 8 + 2K + 1)
        # (B, agents_num, time_len, 4 + 2K + 1)
        return agents_past_current, agents_past_cur_xyyaw_time

    def _add_timestep_to_ego_fut(
            self,
            ego_future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # ego_future: (B, future_len, d_8)
        B, future_len, _ = ego_future.shape
        # ego_future_timestep: (B, future_len)
        ego_future_timestep = timegrid_future_2d(
            dt=self.time_gap,  # 0.1
            total_steps=future_len,  # 80
            B=B,
            device=ego_future.device,
            dtype=ego_future.dtype)
        ego_future_time_fourier = encode_time_with_fourier_features(
            ego_future_timestep, self.time_min, self.time_max,
            self.num_fourier_frequencies)  # (B, future_len, 2K + 1)
        ego_future = torch.cat([ego_future, ego_future_time_fourier],
                               dim=-1)  # (B, future_len, 8 + 2K + 1)
        ######### FOR POSITIONAL EMBEDDING #########
        # (B, future_len, 4)
        ego_future_xyyaw = ego_future[:, :, :4].clone()
        # (B, future_len, 4 + 2K + 1)
        ego_future_xyyaw_time = torch.cat(
            [ego_future_xyyaw, ego_future_time_fourier], dim=-1)
        # ego_future: (B, future_len, 8 + 2K + 1)
        # ego_future_xyyaw_time: (B, future_len, 4 + 2K + 1)
        return ego_future, ego_future_xyyaw_time

    def _get_on_agents_past_cur_on_chunk(
        self, on_agents_past_cur_chunk: torch.Tensor,
        on_agents_past_cur_off_chunk_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        on_agents_past_cur_chunk: (agents_past_cur_on_num, past_cur_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        on_agents_past_cur_off_chunk_mask: (agents_past_cur_on_num, past_cur_chunk_num)

        on_agents_past_cur_on_chunk: (on_past_cur_on_chunk_num, tokens_mlp_dim, channels_mlp_dim)
            on_past_cur_on_chunk_num:
                agents_past_cur_on_num * past_cur_chunk_num 중, True인 개수
        on_agents_past_cur_on_chunk_mask: (agents_past_cur_on_num * past_cur_chunk_num)
        """
        agents_past_cur_on_num, past_cur_chunk_num = on_agents_past_cur_chunk.shape[:
                                                                                    2]
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.view(
            agents_past_cur_on_num * past_cur_chunk_num, self.tokens_mlp_dim,
            -1)
        on_agents_past_cur_on_chunk_mask = ~on_agents_past_cur_off_chunk_mask.view(
            agents_past_cur_on_num * past_cur_chunk_num)
        on_agents_past_cur_on_chunk = on_agents_past_cur_chunk[
            on_agents_past_cur_on_chunk_mask]
        return on_agents_past_cur_on_chunk, on_agents_past_cur_on_chunk_mask

    def _get_on_ego_fut_on_chunk(
        self,
        on_ego_fut_chunk: torch.Tensor,
        on_ego_fut_off_chunk_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        on_ego_fut_chunk: (ego_future_on_num, future_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        on_ego_fut_off_chunk_mask: (ego_future_on_num, future_chunk_num)

        on_ego_fut_on_chunk: (on_ego_fut_on_chunk_num, tokens_mlp_dim, channels_mlp_dim)
            on_ego_fut_on_chunk_num:
                ego_future_on_num * future_chunk_num 중, True인 개수
        on_ego_fut_on_chunk_mask: (ego_future_on_num * future_chunk_num)
        """
        ego_future_on_num, future_chunk_num = on_ego_fut_chunk.shape[:2]
        on_ego_fut_chunk = on_ego_fut_chunk.view(
            ego_future_on_num * future_chunk_num, self.tokens_mlp_dim, -1)
        on_ego_fut_on_chunk_mask = ~on_ego_fut_off_chunk_mask.view(
            ego_future_on_num * future_chunk_num)
        on_ego_fut_on_chunk = on_ego_fut_chunk[on_ego_fut_on_chunk_mask]
        return on_ego_fut_on_chunk, on_ego_fut_on_chunk_mask

    def _fill_on_agent_to_agent(
            self,
            on_agents_past_cur_chunk: torch.
        Tensor,  # (agents_past_cur_on_num,  past_cur_chunk_num, hidden_dim)
            on_ego_fut_chunk: torch.
        Tensor,  # (ego_future_on_num, future_chunk_num, hidden_dim)
            agents_past_cur_on_mask: torch.Tensor,  # (B * agents_num)
            ego_future_on_mask: torch.Tensor,  # (B)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        past_cur_chunk_num = on_agents_past_cur_chunk.shape[1]
        future_chunk_num = on_ego_fut_chunk.shape[1]
        B_agents_num = agents_past_cur_on_mask.shape[0]  # B * agents_num
        B = ego_future_on_mask.shape[0]  # 배치 크기
        agents_num = B_agents_num // B  # agents_num = B * agents_num / B
        agents_past_cur_chunk = torch.zeros(
            (B_agents_num, past_cur_chunk_num, self._hidden_dim),
            device=on_agents_past_cur_chunk.device)
        # agents_past_cur_on_mask: (B * agents_num) -> agents_past_cur_on_num
        agents_past_cur_chunk[
            agents_past_cur_on_mask] = on_agents_past_cur_chunk
        # (B * agents_num, past_cur_chunk_num, hidden_dim)
        # -> (B,  agents_num * past_cur_chunk_num , hidden_dim)
        agents_past_cur_chunk = agents_past_cur_chunk.view(
            B, agents_num * past_cur_chunk_num, self._hidden_dim)

        ego_fut_chunk = torch.zeros((B, future_chunk_num, self._hidden_dim),
                                    device=on_ego_fut_chunk.device)
        # ego_future_on_mask: (B)
        # -> ego_future_on_num
        ego_fut_chunk[ego_future_on_mask] = on_ego_fut_chunk
        # (B, agents_num * past_cur_chunk_num , hidden_dim)
        # (B, future_chunk_num, hidden_dim)
        return agents_past_cur_chunk, ego_fut_chunk

    def forward(self, ego_past_current, npc_past_current, ego_future):
        '''
        ego_past_current: (B, 1, time_len, 11)
        npc_past_current: (B, agent_num, time_len, 11)
        ego_future: (B, future_len, 11)

        (x, y, cos, sin, vx, vy, w, l, type(3)
        '''
        assert self.future_len == ego_future.shape[1], \
            f"ego_future.shape[1] should be {self.future_len}, but got {ego_future.shape[1]}"
        # (B, agents_num=1+agent_num, time_len, D)
        agents_past_current = torch.cat([ego_past_current, npc_past_current],
                                        dim=1)
        B, agents_num, time_len, _ = agents_past_current.shape
        device = agents_past_current.device
        ############
        agents_type = agents_past_current[:, :, -1, 8:]  # (B, agents_num, 3)
        ############
        # (B, agents_num, time_len, d_8)
        agents_past_current = agents_past_current[..., :8]
        ### add timestep
        # agents_past_current: (B, agents_num, time_len, 8 "+ 2K + 1")
        # agents_past_cur_xyyaw_time: (B, agents_num, time_len, 4 + 2K + 1)
        (agents_past_current, agents_past_cur_xyyaw_time
        ) = self._add_timestep_to_agents_past_cur(agents_past_current)

        ############
        """
        agents_past_cur_off_p_mask: (B, agents_num, time_len)
        agents_past_cur_off_mask: (B, agents_num)
        -----------
        agents_past_cur_on_p_mask: (B, agents_num, time_len, 1) # float
        agents_past_cur_on_mask: (B * agents_num)
        
        _get_agents_past_cur_mask
        _reverse_agents_past_cur_mask
        """
        (agents_past_cur_off_p_mask, agents_past_cur_off_mask
        ) = self._get_agents_past_cur_mask(agents_past_current)
        (agents_past_cur_on_p_mask,
         agents_past_cur_on_mask) = self._reverse_agents_past_cur_mask(
             agents_past_cur_off_p_mask, agents_past_cur_off_mask)

        # on_agents_past_cur: (agents_past_cur_on_num, time_len, 8 + 2K + 1 "+ 1" )
        on_agents_past_cur = self._filter_on_agents_past_cur(
            agents_past_current, agents_past_cur_on_p_mask,
            agents_past_cur_on_mask)

        #################################
        future_len = ego_future.shape[1]
        ego_fut_type = ego_future[:, 0, 8:].clone()  # (B, 3)

        # ego_fut_type : (B, 3)
        # ego_future: (B, future_len, 8)
        ego_future = ego_future[..., :8]
        # ego_future_timestep: (B, future_len)

        ### add timestep
        # ego_future: (B, future_len, 8 "+ 2K + 1")
        # ego_future_xyyaw_time: (B, future_len, 4 + 2K + 1)
        (ego_future,
         ego_future_xyyaw_time) = self._add_timestep_to_ego_fut(ego_future)
        ############
        """
        ego_future_off_p_mask: (B, future_len)
        ego_future_off_mask: (B)
        --------
        ego_future_on_p_mask: (B, future_len, 1)
        ego_future_on_mask: (B)
        """
        ego_future_off_p_mask, ego_future_off_mask = self._get_ego_future_mask(
            ego_future)  # (B, future_len)
        ego_future_on_p_mask = (~ego_future_off_p_mask).float().unsqueeze(
            -1)  # (B, future_len, 1)
        ego_future_on_mask = ~ego_future_off_mask  # (B)
        ###########
        # on_ego_future:
        # (ego_future_on_num, future_len, 8 + 2K + 1 "+ 1")
        on_ego_future = self._filter_on_ego_future(ego_future,
                                                   ego_future_on_p_mask,
                                                   ego_future_on_mask)
        # on_all_time_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len
        agents_past_cur_on_num = on_agents_past_cur.shape[0]
        on_agents_past_cur_num = agents_past_cur_on_num * time_len
        ego_future_on_num = on_ego_future.shape[0]

        # on_all (on_all_time_num, 10 + 2k)
        # on_all_time_num =
        # agents_past_cur_on_num * time_len + on_ego_future_on_num * future_len
        on_all = self._concat_past_cur_and_future(on_agents_past_cur,
                                                  on_ego_future)

        # on_all: (on_all_time_num, channels_mlp_dim)
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

        past_chunk_start_idx, past_chunk_end_idx = self._compute_equal_chunks(
            seq_len=time_len, chunk_num=self.past_cur_chunk_num,
            device=device)  # (past_cur_chunk_num,), (past_cur_chunk_num,)
        #################
        # (B, agents_num * past_cur_chunk_num, 4 + 2K + 1 + 4 + 4)
        agents_past_cur_chunks_feature = self._get_agents_past_cur_chunks_feature(
            past_chunk_start_idx, past_chunk_end_idx,
            agents_past_cur_off_p_mask, agents_past_cur_xyyaw_time)

        #################
        fut_chunk_start_idx, fut_chunk_end_idx = self._compute_equal_chunks(
            seq_len=future_len, chunk_num=self.future_chunk_num,
            device=device)  # (future_chunk_num,), (future_chunk_num,)
        #################
        # ego_future_xyyaw_time: (B, future_len, 4 + 2K + 1)
        # ego_future_off_p_mask: (B, future_len)

        # agents_past_cur_xyyaw_time: (B, agents_num, time_len, 4 + 2K + 1)
        # ego_cur_xyyaw: (B, 4)
        ego_cur_xyyaw = agents_past_cur_xyyaw_time[:, 0, -1, :4].clone()

        # ego_future_chunks_feature: (B, future_chunk_num, 4 + 2K + 1 + 4 + 4)
        ego_future_chunks_feature = self._get_ego_future_chunks_feature(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw_time, ego_cur_xyyaw)
        #################
        all_chunk_pos_feature = torch.cat(
            [agents_past_cur_chunks_feature, ego_future_chunks_feature], dim=1
        )  # (B, agents_num * past_cur_chunk_num + future_chunk_num, 4 + 2K + 1 + 4 + 4)
        #################
        """
        # agents_past_cur_off_p_mask: (B, agents_num, time_len)
        # agents_past_cur_on_mask: (B * agents_num)
        
        # on_agents_past_cur_off_p_mask: (agents_past_cur_on_num, time_len)
        """
        agents_past_cur_off_p_mask = agents_past_cur_off_p_mask.view(
            B * agents_num, time_len)  # (B * agents_num, time_len)
        on_agents_past_cur_off_p_mask = agents_past_cur_off_p_mask[
            agents_past_cur_on_mask]  # (agents_past_cur_on_num, time_len)

        # on_agents_past_cur_chunk:
        #   (agents_past_cur_on_num, past_cur_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        # on_agents_past_cur_off_chunk_mask:
        #   (agents_past_cur_on_num, past_cur_chunk_num)

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

        # on_ego_fut_chunk:
        #   (ego_future_on_num, future_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        # on_ego_fut_off_chunk_mask:
        #   (ego_future_on_num, future_chunk_num)

        (on_ego_fut_chunk,
         on_ego_fut_off_chunk_mask) = self.traj_to_chunk_token(
             on_ego_future,  # (ego_future_on_num, future_len, channels_mlp_dim)
             on_ego_future_off_p_mask,  # (ego_future_on_num, future_len)
             fut_chunk_start_idx,  # (future_chunk_num,)
             fut_chunk_end_idx,  # (future_chunk_num,)
         )
        # on_agents_past_cur_on_chunk: (on_past_cur_on_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        # on_agents_past_cur_on_chunk_mask: (agents_past_cur_on_num * past_cur_chunk_num)
        (on_agents_past_cur_on_chunk, on_agents_past_cur_on_chunk_mask
        ) = self._get_on_agents_past_cur_on_chunk(
            on_agents_past_cur_chunk, on_agents_past_cur_off_chunk_mask)
        on_past_cur_on_chunk_num = on_agents_past_cur_on_chunk.shape[0]

        # on_ego_fut_on_chunk: (on_ego_fut_on_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        # on_ego_fut_on_chunk_mask: (ego_future_on_num * future_chunk_num)
        (on_ego_fut_on_chunk,
         on_ego_fut_on_chunk_mask) = self._get_on_ego_fut_on_chunk(
             on_ego_fut_chunk, on_ego_fut_off_chunk_mask)
        """
        on_all_on_chunk: (on_all_on_chunk_num, tokens_mlp_dim, channels_mlp_dim)
            on_all_on_chunk_num = on_past_cur_on_chunk_num + on_ego_fut_on_chunk_num
        """
        on_all_on_chunk = torch.cat(
            [on_agents_past_cur_on_chunk, on_ego_fut_on_chunk], dim=0)

        ############
        for block in self.blocks:
            on_all_on_chunk = block(on_all_on_chunk)
        # pooling
        # on_all_on_chunk: (on_all_on_chunk_num, channels_mlp_dim)
        on_all_on_chunk = torch.mean(on_all_on_chunk, dim=1)
        # agents_ego_fut_type_emb: (on_all_on_chunk_num, channels_mlp_dim)
        agents_ego_fut_type_emb = self._get_type_embedding(
            agents_type, ego_fut_type, agents_past_cur_on_mask,
            ego_future_on_mask, on_agents_past_cur_on_chunk_mask,
            on_ego_fut_on_chunk_mask, self.past_cur_chunk_num,
            self.future_chunk_num)

        # on_all_on_chunk: (on_all_on_chunk_num, channels_mlp_dim)
        on_all_on_chunk += self.type_scale * agents_ego_fut_type_emb
        # on_all_on_chunk: (on_all_on_chunk_num, hidden_dim)
        on_all_on_chunk = self.emb_project(self.norm(on_all_on_chunk))
        ########################
        # on_agents_past_cur_chunk: (agents_past_cur_on_num, past_cur_chunk_num, hidden_dim)
        # on_ego_fut_chunk: (ego_future_on_num, future_chunk_num, hidden_dim)
        (on_agents_past_cur_chunk,
         on_ego_fut_chunk) = self._fill_on_chunk_to_on_agent(
             on_all_on_chunk, on_agents_past_cur_on_chunk_mask,
             on_ego_fut_on_chunk_mask, on_past_cur_on_chunk_num,
             agents_past_cur_on_num, ego_future_on_num, self.past_cur_chunk_num,
             self.future_chunk_num)
        # agents_past_cur_chunk: (B, agents_num * past_cur_chunk_num , hidden_dim)
        # ego_fut_chunk: (B, future_chunk_num, hidden_dim)
        (agents_past_cur_chunk, ego_fut_chunk) = self._fill_on_agent_to_agent(
            on_agents_past_cur_chunk, on_ego_fut_chunk, agents_past_cur_on_mask,
            ego_future_on_mask)
        # concat: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
        # all_chunk = torch.cat([agents_past_cur_chunk, ego_fut_chunk], dim=1)

        # -------------------- 4) 토큰 결합 --------------------
        # all_chunk: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
        all_chunk = torch.cat([agents_past_cur_chunk, ego_fut_chunk], dim=1)

        all_off_chunk_mask = self._get_all_off_chunk_mask(
            agents_past_cur_on_mask,
            # (B*agents_num,)  True=유효
            on_agents_past_cur_off_chunk_mask,
            # (agents_past_cur_on_num, past_cur_chunk_num) True=무효
            ego_future_on_mask,  # (B,) True=유효
            on_ego_fut_off_chunk_mask,
            # (ego_future_on_num, future_chunk_num) True=무효
        )

        # all_chunk: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
        # all_off_chunk_mask: (B, agents_num * past_cur_chunk_num + future_chunk_num)
        # all_chunk_pos : (B, agents_num * past_cur_chunk_num + future_chunk_num, 4 + (2k + 1) + 4 + 4)
        # ego_fut_global: (B, hidden_dim)

        ego_fut_global = self._get_ego_fut_global(ego_fut_chunk,
                                                  ego_future_on_mask,
                                                  on_ego_fut_off_chunk_mask)
        return all_chunk, all_off_chunk_mask, all_chunk_pos_feature, ego_fut_global

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor,
                     dim: int) -> torch.Tensor:
        """
        x:    (B, M, H) # (B, future_chunk_num, hidden_dim)
        mask: (B, M)  # True=무효 # (B, future_chunk_num)
        return: (B, H) # (B, hidden_dim)
        """
        # 고정 전제
        assert dim == 1, "dim은 1이어야 합니다."
        assert x.dim() == 3 and mask.dim() == 2, "x:(B,M,H), mask:(B,M) 필요"
        assert mask.dtype == torch.bool, "mask는 bool이어야 합니다."

        B, M, H = x.shape
        assert mask.shape == (B, M)

        valid = ~mask  # (B, M)
        denom = valid.sum(dim=dim, keepdim=True)  # (B, 1)
        denom = denom.clamp(min=1)  # (B, 1)  # 0 분모 방지

        valid_exp = valid.unsqueeze(-1)  # (B, M, 1)
        x_masked = x * valid_exp  # (B, M, H)  # 브로드캐스트

        numer = x_masked.sum(dim=dim)  # (B, H)

        out = numer / denom  # (B, H)  # (B,H)/(B,1) 브로드캐스트
        assert out.shape == (B, H)

        return out

    def _get_ego_fut_global(
        self,
        ego_fut_chunk: torch.Tensor,  # (B, future_chunk_num, hidden_dim)
        ego_future_on_mask: torch.Tensor,  # (B)
        on_ego_fut_off_chunk_mask: torch.
        Tensor,  # (ego_future_on_num, future_chunk_num)
    ) -> torch.Tensor:  # (B, hidden_dim):
        # --- (A) 배치 크기로 확장된 미래 chunk 마스크 만들기: (B, future_chunk_num) ---
        B, future_chunk_num = ego_fut_chunk.shape[:2]
        # --- (A) 배치 크기 마스크로 복원: (B, future_chunk_num), True=무효 ---
        ego_fut_off_chunk_mask_full = torch.ones((B, future_chunk_num),
                                                 dtype=torch.bool,
                                                 device=ego_fut_chunk.device)
        # 유효한 배치 위치에만 on_... 마스크 주입
        ego_fut_off_chunk_mask_full[
            ego_future_on_mask] = on_ego_fut_off_chunk_mask

        # --- (B) 안정적 기본값: 마스크드 평균 ---
        ego_fut_global_mean = self._masked_mean(
            ego_fut_chunk, mask=ego_fut_off_chunk_mask_full,
            dim=1)  # (B, hidden_dim)

        # --- (C) 학습형 주의 풀링(가중합) + NaN 방지 ---
        # 로짓 계산 (AMP 안전을 위해 fp32로)
        logits = self.ego_fut_pool_q(
            ego_fut_chunk).float()  # (B, future_chunk_num, 1)
        logits = logits.masked_fill(ego_fut_off_chunk_mask_full.unsqueeze(-1),
                                    float('-inf'))
        all_off = ego_fut_off_chunk_mask_full.all(dim=1)  # (B,)
        if all_off.any():
            logits[all_off] = 0.0  # softmax NaN 방지

        weights = F.softmax(logits, dim=1).to(ego_fut_chunk.dtype)  # (B, M, 1)
        if all_off.any():
            weights[all_off] = 0.0

        ego_fut_global_attn = (weights * ego_fut_chunk).sum(
            dim=1)  # (B, hidden_dim)

        # --- (D) 최종 대표 토큰: 평균 + (학습형 풀링 잔차) ---
        ego_fut_global = ego_fut_global_mean + (
            self.ego_fut_pool_scale * ego_fut_global_attn)  # (B, hidden_dim)

        # 기존 반환값 + ego_fut_global 추가
        return ego_fut_global


class StaticFusionEncoder(nn.Module):

    def __init__(self,
                 dim,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 device='cuda',
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self._hidden_dim = hidden_dim
        self.num_fourier_frequencies = num_fourier_frequencies
        self.projection = Mlp(in_features=dim,
                              hidden_features=hidden_dim,
                              out_features=hidden_dim,
                              act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, static_info):
        '''
        static_info: B, P, D (x, y, cos, sin, w, l, type(4))
        '''
        B, P, _ = static_info.shape

        # static_xyyaw_time: (B, static_objects_num, 4 + 2K + 1)
        static_xyyaw_time = self._get_static_xyyaw_time(static_info)
        # static_feature: (B, static_objects_num, 4 + 2K + 1 + 4 + 4)
        static_feature = self._get_static_feature(static_xyyaw_time)

        static_encoding = torch.zeros((B * P, self._hidden_dim),
                                      device=static_info.device)

        mask_p = torch.sum(torch.ne(static_info[..., :10], 0),
                           dim=-1).to(static_info.device) == 0

        valid_indices = ~mask_p.view(-1)

        if valid_indices.sum() > 0:
            static_info = static_info.view(B * P, -1)
            static_info = static_info[valid_indices]
            static_info = self.projection(static_info)
            static_encoding[valid_indices] = static_info
        static_encoding = static_encoding.view(B, P, -1)  # (B, P, hidden_dim)
        mask_p = mask_p.view(B, P)  # (B, P)
        return static_encoding, mask_p, static_feature

    def _get_static_xyyaw_time(self, static_info: torch.Tensor) -> torch.Tensor:
        """정적 객체의 (x,y,cos,sin) + '현재시점' 시간채널(전부 0) 결합.

        정적 객체는 과거/미래 시계열이 없고, 현재 시점만 의미가 있다.
        시간 채널(2K+1)은 모두 0으로 채워 동일 차원을 유지한다.

        Args:
            static_info (torch.Tensor):
                모양 [B, P, D]
                - B: 배치 크기
                - P: 정적 객체 개수
                - D: 피처 차원(앞 4개는 x,y,cos,sin)

        Returns:
            torch.Tensor:
                모양 [B, P, 4 + (2K+1)]
                - 앞 4: (x, y, cos, sin)
                - 뒤 2K+1: 시간 채널(전부 0, 현재 시점 표현)
                - K = self.num_fourier_frequencies
        """
        B, P, _ = static_info.shape  # B,P,_
        static_xyyaw = static_info[:, :, :4].clone()  # [B,P,4]
        K: int = self.num_fourier_frequencies
        zeros_time = static_info.new_zeros(  # [B,P,2K+1]
            B, P, 2 * K + 1)
        static_xyyaw_time = torch.cat(  # [B,P,4+(2K+1)]
            [static_xyyaw, zeros_time], dim=-1)
        return static_xyyaw_time

    def _get_static_feature(
        self,
        static_xyyaw_time: torch.Tensor  # (B, static_objects_num, 4 + 2K + 1)
    ) -> torch.Tensor:
        # First: add (x, y, cos, sin) delta -> (0., 0., 1., 0.)
        B, static_objects_num, _ = static_xyyaw_time.shape
        delta_feature = torch.zeros(
            (B, static_objects_num, 4),
            device=static_xyyaw_time.device,
            dtype=static_xyyaw_time.dtype,
        )
        delta_feature[:, :, 2] = 1.0  # cos
        # (B, static_objects_num, 4 + 2K + 1 + 4)
        static_xyyaw_time = torch.cat([static_xyyaw_time, delta_feature],
                                      dim=-1)
        # static_type: (B, static_objects_num, 4) # 4:
        static_type = torch.zeros(
            (B, static_objects_num, 4),
            device=static_xyyaw_time.device,
            dtype=static_xyyaw_time.dtype,
        )
        static_type[:, :, -2] = 1.0  # type
        # static_feature: (B, static_objects_num, 4 + 2K + 1 + 4 + 4)
        static_feature = torch.cat([static_xyyaw_time, static_type], dim=-1)
        return static_feature


class LaneFusionEncoder(nn.Module):

    def __init__(self,
                 lane_len,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 depth=3,
                 tokens_mlp_dim=64,
                 channels_mlp_dim=128,
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self._lane_len = lane_len
        self.num_fourier_frequencies = num_fourier_frequencies
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

    def _get_lane_xyyaw_time(self, lane_pos: torch.Tensor) -> torch.Tensor:
        """차선 중심 포인트의 (x,y,cos,sin) + '현재시점' 시간채널(전부 0) 결합.

        차선도 현재 프레임에서 추출된 벡터 표현만 사용한다.
        시간 채널(2K+1)은 모두 0으로 채워 동일 차원을 유지한다.

        Args:
            lane_pos (torch.Tensor):
                모양 [B, L, 4]
                - B: 배치 크기
                - L: 차선(폴리라인) 수
                - 4: (x, y, cos, sin)

        Returns:
            torch.Tensor:
                모양 [B, L, 4 + (2K+1)]
                - 앞 4: (x, y, cos, sin)
                - 뒤 2K+1: 시간 채널(전부 0, 현재 시점 표현)
                - K = self.num_fourier_frequencies
        """
        B, L, _ = lane_pos.shape  # B,L,4
        K: int = self.num_fourier_frequencies
        zeros_time: torch.Tensor = lane_pos.new_zeros(  # [B,L,2K+1]
            B, L, 2 * K + 1)
        lane_xyyaw_time: torch.Tensor = torch.cat(  # [B,L,4+(2K+1)]
            [lane_pos, zeros_time], dim=-1)
        return lane_xyyaw_time

    def _get_lane_feature(
        self, lane_xyyaw_time: torch.Tensor
        # (B, lane_num, 4 + 2K + 1)
    ) -> torch.Tensor:
        # First: add (x, y, cos, sin) delta -> (0., 0., 1., 0.)
        B, lane_num, _ = lane_xyyaw_time.shape
        delta_feature = torch.zeros(
            (B, lane_num, 4),
            device=lane_xyyaw_time.device,
            dtype=lane_xyyaw_time.dtype,
        )
        delta_feature[:, :, 2] = 1.0  # cos
        # (B, lane_num, 4 + 2K + 1 + 4)
        lane_xyyaw_time = torch.cat([lane_xyyaw_time, delta_feature], dim=-1)
        # lane_type: (B, lane_num, 4) # 4:
        lane_type = torch.zeros(
            (B, lane_num, 4),
            device=lane_xyyaw_time.device,
            dtype=lane_xyyaw_time.dtype,
        )
        lane_type[:, :, -1] = 1.0  # type
        # static_feature: (B, lane_num, 4 + 2K + 1 + 4 + 4)
        lane_feature = torch.cat([lane_xyyaw_time, lane_type], dim=-1)
        return lane_feature

    def forward(self, lane_info, speed_limit, has_speed_limit):
        '''
        lane_info: B, lane_num, lane_len, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, lane_num, 1
        has_speed_limit: B, lane_num, 1
        '''

        traffic = lane_info[:, :, 0, 8:]
        lane_info = lane_info[..., :8]

        lane_pos = lane_info[:, :, int(self._lane_len /
                                       2), :4].clone()  # (B, lane_num, 4)
        heading = torch.atan2(lane_pos[..., 3], lane_pos[..., 2])
        lane_pos[..., 2] = torch.cos(heading)
        lane_pos[..., 3] = torch.sin(heading)
        # lane_pos: (B, lane_num, 4)
        # lane_xyyaw_time: (B, lane_num, 4 + 2K + 1)
        lane_xyyaw_time = self._get_lane_xyyaw_time(lane_pos)
        # lane_feature: (B, lane_num, 4 + 2K + 1 + 4 + 4)
        lane_feature = self._get_lane_feature(lane_xyyaw_time)

        B, lane_num, lane_len, _ = lane_info.shape
        mask_v = torch.sum(torch.ne(lane_info[..., :8], 0),
                           dim=-1).to(lane_info.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        lane_info = lane_info.view(B * lane_num, lane_len, -1)

        valid_indices = ~mask_p.view(-1)
        lane_info = lane_info[valid_indices]

        lane_info = self.channel_pre_project(lane_info)
        lane_info = lane_info.permute(0, 2, 1)
        lane_info = self.token_pre_project(lane_info)
        lane_info = lane_info.permute(0, 2, 1)
        for block in self.blocks:
            lane_info = block(lane_info)

        lane_info = torch.mean(lane_info, dim=1)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.view(B * lane_num, 1)
        has_speed_limit = has_speed_limit.to(torch.bool).view(B * lane_num, 1)
        traffic = traffic.view(B * lane_num, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)
        speed_limit_embedding = torch.zeros(
            (speed_limit.shape[0], self._channel), device=lane_info.device)

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

        lane_info = lane_info + speed_limit_embedding + traffic_light_embedding
        lane_info = self.emb_project(self.norm(lane_info))

        lane_embedding = torch.zeros((B * lane_num, lane_info.shape[-1]),
                                     device=lane_info.device)
        lane_embedding[valid_indices] = lane_info  # Fill in valid parts

        return lane_embedding.view(B, lane_num,
                                   -1), mask_p.reshape(B, -1), lane_feature


class FusionEncoder(nn.Module):

    def __init__(self,
                 hidden_dim=192,
                 num_heads=6,
                 drop_path_rate=0.3,
                 depth=3,
                 device='cuda'):
        super().__init__()

        dpr = drop_path_rate

        # 1) CLS/scene 토큰과 그 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout=dpr)
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """장면 융합 전용 포워드(배치별 전부 패딩 샘플은 건너뜀).

        모든 토큰이 패딩(True)인 배치는 연산을 생략하고 0을 반환한다.
        유효 토큰이 하나라도 있는 배치만 CLS 토큰을 붙여 블록을 통과시킨 뒤,
        최종 출력에서 CLS를 제거하고 원래 위치에 복원한다.

        Args:
            x (torch.Tensor): 입력 토큰 시퀀스.
                모양: [B, T, H]
                - B: 배치 크기
                - T: 토큰 길이
                - H: 히든 차원
            mask (torch.Tensor): 키 패딩 마스크(True=패딩으로 무시).
                모양: [B, T]

        Returns:
            torch.Tensor: CLS 제외 최종 시퀀스 임베딩.
                모양: [B, T, H]
        """
        # 입력 별칭(가독성)
        seq_tokens = x  # [B, T, H]
        seq_key_pad_mask = mask  # [B, T], True=pad

        B, T, H = seq_tokens.shape  # B, T, H 스칼라

        # 1) 전체 패딩 배치 식별 및 결과 버퍼 준비
        is_all_pad_per_batch = seq_key_pad_mask.all(dim=1)  # [B]
        will_process_mask = ~is_all_pad_per_batch  # [B]
        out_tokens = seq_tokens.new_zeros(B, T, H)  # [B, T, H]

        # 2) 유효 배치만 선택
        if will_process_mask.any():
            kept_tokens = seq_tokens[will_process_mask]  # [B_keep, T, H]
            kept_key_pad = seq_key_pad_mask[will_process_mask]  # [B_keep, T]

            B_keep: int = kept_tokens.size(0)

            # 3) CLS 부착 및 CLS 위치 임베딩 추가
            cls_tokens = self.cls_token.expand(B_keep, 1, H)  # [B_keep, 1, H]
            kept_with_cls = torch.cat([cls_tokens, kept_tokens],
                                      dim=1)  # [B_keep, T+1, H]
            kept_with_cls[:, 0:
                          1, :] = kept_with_cls[:, 0:
                                                1, :] + self.cls_pos  # [B_keep, 1, H] += pos

            # 4) 마스크에 CLS(False) 추가
            cls_false = torch.zeros(B_keep,
                                    1,
                                    dtype=torch.bool,
                                    device=kept_key_pad.device)  # [B_keep,1]
            kept_mask_with_cls = torch.cat([cls_false, kept_key_pad],
                                           dim=1)  # [B_keep, T+1]

            # 5) 블록 통과
            fused = kept_with_cls  # [B_keep, T+1, H]
            for block in self.blocks:
                fused = block(fused, kept_mask_with_cls)  # [B_keep, T+1, H]
            fused = self.norm(fused)  # [B_keep, T+1, H]

            # 6) CLS 제거 후 원래 배치 위치에 복원
            fused_wo_cls = fused[:, 1:, :]  # [B_keep, T, H]
            out_tokens[will_process_mask] = fused_wo_cls  # [B, T, H]

        # 전부 패딩 배치는 out_tokens의 0 유지
        return out_tokens  # [B, T, H]
