import math
from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn
from timm.models.layers import Mlp
from timm.layers import DropPath

from diffusion_planner.model.module.mixer import MixerBlock


# =============================================================================
# 공용 유틸 함수들
# =============================================================================

def time_embed_seconds(t: Tensor, d_fourier: int = 12, Tmax: float = 8.0) -> Tensor:
    """연속 시간을 멀티-주파수 벡터로 임베딩.

    시간 t(초)를 여러 사인/코사인과 스칼라 두 개([t/Tmax, sign(t)])로 확장한다.

    Args:
      t (Tensor): (...,) 모양의 실수 초 텐서. 과거<0, 미래>0 허용.
      d_fourier (int): 사인/코사인 주파수 개수 K.
      Tmax (float): 정규화 기준 시간(초). 보통 예측 지평(예: 8.0).

    Returns:
      Tensor: (..., 2*K+2) 임베딩. [sin(w_k t), cos(w_k t)]_k ⊕ [t/Tmax, sign(t)]
    """
    device, dtype = t.device, t.dtype
    k = torch.arange(d_fourier, device=device, dtype=dtype)          # (K,)
    w = (2.0 ** k) * (math.pi / Tmax)                                 # (K,)
    wt = t.unsqueeze(-1) * w                                          # (..., K)
    fourier = torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)       # (..., 2K)
    t_norm = (t / Tmax).unsqueeze(-1)                                 # (..., 1)
    t_sign = torch.sign(t).unsqueeze(-1)                              # (..., 1)
    return torch.cat([fourier, t_norm, t_sign], dim=-1)               # (..., 2K+2)


def make_past_time_vector(V: int, dt: float, device, dtype) -> Tensor:
    """과거 시간 벡터 생성.

    Args:
      V (int): 프레임 수(예: 21).
      dt (float): 샘플 간격(초).
      device: torch device.
      dtype: torch dtype.

    Returns:
      Tensor: (V,) = [-dt*(V-1), ..., 0.0]
    """
    return torch.linspace(-dt * (V - 1), 0.0, V, device=device, dtype=dtype)


def make_future_time_vector(F: int, dt: float, device, dtype) -> Tensor:
    """미래 시간 벡터 생성.

    Args:
      F (int): 프레임 수(예: 80).
      dt (float): 샘플 간격(초).
      device: torch device.
      dtype: torch dtype.

    Returns:
      Tensor: (F,) = [dt, 2dt, ..., F*dt]
    """
    idx = torch.arange(1, F + 1, device=device, dtype=dtype)
    return idx * dt


def masked_mean_lastdim(x: Tensor, mask: Tensor) -> Tensor:
    """마스크 적용 평균(마지막 차원 기준).

    Args:
      x (Tensor): (..., T, D) 텐서.  (Mv, V, tstem)
      mask (Tensor): (..., T) bool 이나 {0,1} 텐서. True/1 = 유효, False/0 = 무효가 아님에 주의!
                     이 함수는 **유효=1** 가정이므로, 일반적 pad-mask(True=무효)를 받으면
                     먼저 (~mask)로 뒤집어 전달해야 함.
                     (Mv, V)

    Returns:
      Tensor: (..., D) 유효 위치 평균.
    """
    valid = mask.to(x.dtype).unsqueeze(-1)            # (..., T, 1) # (Mv, V, 1)
    denom = valid.sum(dim=-2).clamp_min(1.0)          # (..., 1) # (Mv, 1)
    return (x * valid).sum(dim=-2) / denom            # (..., D) # (Mv, tstem)


def build_segment_centers_and_weights(base_time: Tensor,
                                      K: int,
                                      sigma_scale: float) -> Tuple[Tensor, Tensor, float]:
    """균등 K-세그먼트의 중심과 가우시안 가중치 템플릿 계산.

    Args:
      base_time (Tensor): (V,) 과거 시간 벡터(예: [-2.0, ..., 0.0]).
      K (int): 세그먼트 개수.
      sigma_scale (float): 각 세그먼트 폭을 (세그 길이 * sigma_scale) 로 설정.

    Returns:
      Tuple[Tensor, Tensor, float]:
        - mu (Tensor): (K,) 세그먼트 중심(초).
        - w_template (Tensor): (K, V) 가우시안 가중치(정규화 전).
        - sigma (float): 가우시안 표준편차(초).
    """
    V = base_time.numel()
    T = base_time[-1] - base_time[0]  # end - start (음수일 수 있음, 과거→현재면 T>0로 보정)
    T = abs(T.item())
    start_t, end_t = -T, 0.0
    mu = torch.linspace(start_t + (T / (2 * K)),
                        end_t - (T / (2 * K)),
                        K, device=base_time.device, dtype=base_time.dtype)     # (K,)
    sigma = (T / K) * sigma_scale + 1e-6
    t_vec = base_time.view(1, -1)                                            # (1,V)
    w_template = torch.exp(-(t_vec - mu.view(-1, 1)) ** 2 / (2.0 * sigma ** 2))  # (K,V)
    return mu, w_template, sigma


def segment_pool_gaussian(h: Tensor,
                          valid_v: Tensor,
                          w_template: Tensor) -> Tensor:
    """가우시안 가중합으로 시간 축을 K-세그먼트로 요약.

    Args:
      h (Tensor): (Mv, V, C) 시간-채널 피처.
      valid_v (Tensor): (Mv, V) 유효 시점(1) / 무효(0) 마스크.
      w_template (Tensor): (K, V) 가우시안 가중치(정규화 전).

    Returns:
      Tensor: (Mv, K, C) 세그먼트별 요약 피처.
    """
    Mv, V, C = h.shape
    K = w_template.shape[0]
    w = w_template.unsqueeze(0).expand(Mv, -1, -1)        # (Mv,K,V)
    w = w * valid_v.unsqueeze(1)                           # (Mv,K,V), 무효 시점 0
    w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)   # 정규화
    # (Mv,K,V) x (Mv,V,C) -> (Mv,K,C)
    return torch.einsum('mkv,mvc->mkc', w, h)


def add_time_bypass(h: Tensor, z_bar: Tensor, proj: nn.Linear) -> Tensor:
    """시간 stem 요약(z_bar)을 토큰 축 전체에 더하는 바이패스.

    Args:
      h (Tensor): (..., T, C) 토큰-채널 피처. # (Mv, V, C')
      z_bar (Tensor): (..., D_tstem) 시간 stem 평균. # (Mv, tstem)
      proj (nn.Linear): D_tstem -> C 사상 선형층.

    Returns:
      Tensor: (..., T, C) 바이패스가 더해진 피처.
    """
    bypass = proj(z_bar).unsqueeze(-2)              # (..., 1, C) # (Mv, 1, C)
    return h + bypass                               # 브로드캐스팅으로 (..., T, C)


def inject_pos_emb(encoding_pos: Tensor,
                   encoding_mask: Tensor,
                   pos_emb: nn.Linear,
                   hidden_dim: int) -> Tensor:
    """pos 임베딩(기하+타입) 주입을 유효 토큰에만 적용.

    Args:
      encoding_pos (Tensor): (B, L, 8) pos 텐서(앞4=기하, 뒤4=타입 one-hot).
      encoding_mask (Tensor): (B, L) bool, True=무효 토큰.
      pos_emb (nn.Linear): 8 -> hidden_dim 사상층.
      hidden_dim (int): 히든 차원.

    Returns:
      Tensor: (B, L, hidden_dim) pos 임베딩 결과(무효 위치는 0).
    """
    B, L, _ = encoding_pos.shape
    flat_pos = encoding_pos.view(B * L, -1)                 # (B*L, 8)
    flat_mask = encoding_mask.view(-1)                      # (B*L,)
    out = torch.zeros((B * L, hidden_dim), device=encoding_pos.device)
    out[~flat_mask] = pos_emb(flat_pos[~flat_mask])         # 유효 토큰만
    return out.view(B, L, hidden_dim)


def build_time_attn_bias(time_anchors: Optional[Tensor],
                         beta: float,
                         dtype,
                         device) -> Optional[Tensor]:
    """시간 어텐션 바이어스 행렬 생성.

    Args:
      time_anchors (Optional[Tensor]): (B, L) 각 토큰의 대표 시각(초). None이면 미사용.
      beta (float): 바이어스 강도(가까운 시간 우대).
      dtype: torch dtype.
      device: torch device.

    Returns:
      Optional[Tensor]: (L, L) additive bias. None이면 바이어스 미사용.
    """
    if time_anchors is None:
        return None
    # 배치마다 동일한 앵커 배열이라고 가정(마스크는 key_padding_mask로 처리)
    t = time_anchors[0].to(dtype=dtype, device=device)  # (L,)
    return -beta * (t[:, None] - t[None, :]).abs()      # (L,L)


def scatter_segments_to_batch(h_seg: Tensor,
                              valid_indices: Tensor,
                              B: int,
                              M: int,
                              K: int) -> Tensor:
    """세그먼트 요약 결과를 (B, M*K, H)로 복원(scatter).

    Args:
      h_seg (Tensor): (Mv, K, H) 유효 토큰들의 세그먼트 요약.
      valid_indices (Tensor): (B*M,) bool. True=유효 토큰.
      B (int): 배치 크기.
      M (int): 배치당 에이전트 수(ego 포함).
      K (int): 세그먼트 수.

    Returns:
      Tensor: (B, M*K, H) 복원된 텐서(무효=0).
    """
    H = h_seg.shape[-1]
    out = torch.zeros((B * M * K, H), device=h_seg.device)
    flat_valid_idx = valid_indices.nonzero(as_tuple=False).squeeze(-1)   # (Mv,)
    for j, idx in enumerate(flat_valid_idx):
        start = (idx.item() * K)
        out[start:start + K] = h_seg[j]                                  # (K,H)
    return out.view(B, M * K, H)


# =============================================================================
# 모듈들
# =============================================================================

class TimeStem(nn.Module):
    """시간 임베딩을 공통 저차원 공간으로 투영하는 공유 MLP.

    Args:
      in_dim (int): 입력 차원(=2*d_fourier+2).
      out_dim (int): 출력 차원(공유 시간 공간 차원).

    Shape:
      - 입력: (..., in_dim)
      - 출력: (..., out_dim)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.out_dim = out_dim

    def forward(self, te: Tensor) -> Tensor:
        return self.net(te)


class SelfAttentionBlock(nn.Module):
    """시간 바이어스(attn_mask) 가능 Self-Attention 블록 + MLP.

    Args:
      dim (int): 히든 차원.
      heads (int): 어텐션 헤드 수.
      dropout (float): 드롭아웃 확률.
      mlp_ratio (float): MLP 확장 비율.

    Shape:
      - 입력 x: (B, L, D)
      - mask: (B, L) bool, True=무효
      - attn_bias: (L, L) or None
      - 출력: (B, L, D)
    """
    def __init__(self, dim: int = 192, heads: int = 6, dropout: float = 0.1, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout, batch_first=True)
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

    def forward(self, x: Tensor, mask: Tensor, attn_bias: Optional[Tensor] = None) -> Tensor:
        q = self.norm1(x)
        x = x + self.drop_path(self.attn(q, q, q, key_padding_mask=mask, attn_mask=attn_bias)[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class AgentFusionEncoder(nn.Module):
    """과거 ego+이웃 에이전트 시계열 인코더 (K-세그먼트 요약 + 공통 TimeStem 바이패스).

    Note:
      - K-세그먼트 풀링을 위해 토큰 길이 = 원래 시간 길이(V)로 유지해야 한다.
        내부에서 `tokens_mlp_dim`을 `time_len`으로 강제 설정한다.

    Args:
      time_len (int): 과거 시간 길이 V.
      drop_path_rate (float): 드롭패스.
      hidden_dim (int): 출력 히든 차원 H.
      depth (int): MixerBlock 개수.
      tokens_mlp_dim (int|None): 토큰 MLP 차원(무시; time_len으로 강제).
      channels_mlp_dim (int): 채널 MLP 차원.
      time_fourier_dim (int): 시간 임베딩 K.
      time_max_sec (float): Tmax.
      dt (float): 샘플 간격(초).
      segments_K (int): 세그먼트 수 K.
      segment_sigma_scale (float): 세그 폭 스케일.
      time_stem (TimeStem): 공유 타임 스템.

    Returns:
      Tuple[Tensor, Tensor, Tensor, Tensor]:
        - tokens: (B, M*K, H)
        - mask:   (B, M*K) bool
        - pos:    (B, M*K, 8)
        - time_anchor: (B, M*K) 각 세그먼트 중심 시각(초, 음수)
    """
    def __init__(self,
                 time_len: int,
                 drop_path_rate: float = 0.3,
                 hidden_dim: int = 192,
                 depth: int = 3,
                 tokens_mlp_dim: Optional[int] = None,
                 channels_mlp_dim: int = 128,
                 time_fourier_dim: int = 12,
                 time_max_sec: float = 8.0,
                 dt: float = 0.1,
                 segments_K: int = 2,
                 segment_sigma_scale: float = 0.5,
                 time_stem: Optional[TimeStem] = None):
        super().__init__()
        assert time_stem is not None, "AgentFusionEncoder requires a shared TimeStem."
        self._hidden_dim = hidden_dim
        self._channel = channels_mlp_dim

        # 시간/세그먼트 설정
        self.time_fourier_dim = time_fourier_dim
        self.time_max_sec = time_max_sec
        self.dt = dt
        self.K = segments_K
        self.segment_sigma_scale = segment_sigma_scale
        self.time_stem = time_stem
        time_stem_dim = self.time_stem.out_dim

        time_feat_dim = 2 * self.time_fourier_dim + 2  # [sin,cos]*K + [t/Tmax, sign]

        self.type_emb = nn.Linear(3, channels_mlp_dim)

        # 입력 -> 채널 임베딩
        self.channel_pre_project = Mlp(in_features=8 + time_feat_dim + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU, drop=0.)

        # 토큰 길이를 V로 유지(세그 풀링 정합성)
        if tokens_mlp_dim is None or tokens_mlp_dim != time_len:
            tokens_mlp_dim = time_len
        self.token_pre_project = Mlp(in_features=time_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU, drop=0.)

        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                                     for _ in range(depth)])

        self.time_bypass = nn.Linear(time_stem_dim, channels_mlp_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim, out_features=hidden_dim,
                               act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, x_ego: Tensor, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        # 입력 정리
        x = torch.cat([x_ego, x], dim=1)               # (B, M=P+1, V, D)
        neighbor_type = x[:, :, -1, 8:]                # (B, M, 3)
        x = x[..., :8]                                 # (B, M, V, 8)

        # pos (마지막 프레임 기준) + 타입 원핫
        pos = x[:, :, -1, :8].clone()                  # (B, M, 8)
        pos[..., -4:] = 0.0
        pos[..., -3] = 1.0                             # neighbor
        pos[:, 0, -3] = 0.0; pos[:, 0, -4] = 1.0       # ego

        B, M, V, _ = x.shape
        mask_v = torch.sum(torch.ne(x[..., :8], 0), dim=-1).to(x.device) == 0  # (B,M,V)
        mask_p = torch.sum(~mask_v, dim=-1) == 0                               # (B,M)

        # 시간 임베딩 (과거)
        base_time = make_past_time_vector(V, self.dt, x.device, x.dtype)       # (V,)
        te = time_embed_seconds(base_time, self.time_fourier_dim, self.time_max_sec)  # (V, D_te)
        te_stem = self.time_stem(te)                                           # (V, tstem)
        te_stem = te_stem.view(1, 1, V, -1).expand(B, M, -1, -1)               # (B,M,V,tstem)

        # 입력 채널 결합: [x y cos sin] + [time] + [valid]
        time_feat = te.view(1, 1, V, -1).expand(B, M, -1, -1)                  # (B,M,V,D_te)
        x_cat = torch.cat([x, time_feat, (~mask_v).float().unsqueeze(-1)], dim=-1)     # (B,M,V,Cin)
        x_cat = x_cat.view(B * M, V, -1)                                       # (B*M, V, Cin)

        valid_indices = ~mask_p.view(-1)                                       # (B*M,)
        x_cat = x_cat[valid_indices]                                           # (Mv, V, Cin)
        te_stem_v = te_stem.view(B * M, V, -1)[valid_indices]                  # (Mv, V, tstem)

        # 전처리 + 토큰 믹서
        h = self.channel_pre_project(x_cat)                                    # (Mv, V, C)
        h = h.permute(0, 2, 1)                                                 # (Mv, C, V)
        h = self.token_pre_project(h)                                          # (Mv, C', V=V)
        h = h.permute(0, 2, 1)                                                 # (Mv, V, C')

        for block in self.blocks:
            h = block(h)                                                       # (Mv, V, C')

        # 시간 stem 바이패스
        valid_v = (~mask_v.view(B * M, V))[valid_indices].float()              # (Mv, V)
        # te_stem_v: (Mv, V, tstem) # valid_v: (Mv, V)
        z_bar = masked_mean_lastdim(te_stem_v, valid_v)                        # (Mv, tstem)
        h = add_time_bypass(h, z_bar, self.time_bypass)                        # (Mv, V, C')

        # K-세그먼트 가우시안 풀링
        mu, w_template, _ = build_segment_centers_and_weights(base_time, self.K, self.segment_sigma_scale)
        h_seg = segment_pool_gaussian(h, valid_v, w_template)                  # (Mv, K, C')

        # 타입 임베딩 추가 + 투영
        neighbor_type_v = neighbor_type.view(B * M, -1)[valid_indices]         # (Mv,3)
        type_emb = self.type_emb(neighbor_type_v).unsqueeze(1)                 # (Mv,1,C')
        h_seg = h_seg + type_emb                                               # (Mv,K,C')

        h_seg = self.norm(h_seg)
        h_seg = self.emb_project(h_seg.reshape(h_seg.shape[0] * self.K, -1))   # (Mv*K, H)
        h_seg = h_seg.view(-1, self.K, h_seg.shape[-1])                        # (Mv,K,H)

        # 배치 스캐터
        tokens_result = scatter_segments_to_batch(h_seg, valid_indices, B, M, self.K)  # (B, M*K, H)

        # 마스크/pos/time_anchor
        mask_seg = mask_p.unsqueeze(-1).expand(B, M, self.K).reshape(B, M * self.K)
        pos_seg = pos.unsqueeze(2).expand(B, M, self.K, 8).reshape(B, M * self.K, 8)
        time_anchor_seg = mu.view(1, 1, self.K).expand(B, M, -1).reshape(B, M * self.K)

        return tokens_result, mask_seg, pos_seg, time_anchor_seg


class EgoFutureEncoder(nn.Module):
    """미래 ego 플랜 시계열 인코더 (공통 TimeStem 바이패스 + 시각 앵커).

    Args:
      future_len (int): 미래 길이 F.
      drop_path_rate (float): 드롭패스.
      hidden_dim (int): 출력 히든 차원 H.
      depth (int): MixerBlock 개수.
      tokens_mlp_dim (int|None): 토큰 MLP 차원(기본=F).
      channels_mlp_dim (int): 채널 MLP 차원.
      time_fourier_dim (int): 시간 임베딩 K.
      time_max_sec (float): Tmax.
      dt (float): 샘플 간격(초).
      time_stem (TimeStem): 공유 타임 스템.

    Returns:
      Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        - tokens: (B, F, H)
        - mask:   (B, F) bool
        - pos:    (B, F, 8)
        - global_vec: (B, H)
        - time_anchor: (B, F) 각 스텝 시각(초)
    """
    def __init__(self,
                 future_len: int,
                 drop_path_rate: float = 0.3,
                 hidden_dim: int = 192,
                 depth: int = 3,
                 tokens_mlp_dim: Optional[int] = None,
                 channels_mlp_dim: int = 128,
                 time_fourier_dim: int = 12,
                 time_max_sec: float = 8.0,
                 dt: float = 0.1,
                 time_stem: Optional[TimeStem] = None):
        super().__init__()
        assert time_stem is not None, "EgoFutureEncoder requires a shared TimeStem."

        if tokens_mlp_dim is None:
            tokens_mlp_dim = future_len

        self._future_len = future_len
        self.time_fourier_dim = time_fourier_dim
        self.time_max_sec = time_max_sec
        self.dt = dt
        self.time_stem = time_stem
        time_stem_dim = self.time_stem.out_dim

        time_feat_dim = 2 * self.time_fourier_dim + 2

        self.channel_pre_project = Mlp(in_features=4 + time_feat_dim + 1,
                                       hidden_features=channels_mlp_dim,
                                       out_features=channels_mlp_dim,
                                       act_layer=nn.GELU, drop=0.)
        self.token_pre_project = Mlp(in_features=future_len,
                                     hidden_features=tokens_mlp_dim,
                                     out_features=tokens_mlp_dim,
                                     act_layer=nn.GELU, drop=0.)
        self.blocks = nn.ModuleList([MixerBlock(tokens_mlp_dim, channels_mlp_dim, drop_path_rate)
                                     for _ in range(depth)])
        self.time_bypass = nn.Linear(time_stem_dim, channels_mlp_dim)
        self.norm = nn.LayerNorm(channels_mlp_dim)
        self.emb_project = Mlp(in_features=channels_mlp_dim,
                               hidden_features=hidden_dim, out_features=hidden_dim,
                               act_layer=nn.GELU, drop=drop_path_rate)

    def forward(self, ego_future: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        B, F, _ = ego_future.shape
        mask = torch.sum(torch.ne(ego_future, 0), dim=-1) == 0                 # (B,F)

        # 시간 임베딩 (미래)
        t = make_future_time_vector(F, self.dt, ego_future.device, ego_future.dtype)  # (F,)
        te = time_embed_seconds(t, self.time_fourier_dim, self.time_max_sec)          # (F, D_te)
        te_stem = self.time_stem(te).unsqueeze(0).expand(B, -1, -1)                   # (B,F,tstem)
        time_feat = te.unsqueeze(0).expand(B, -1, -1)                                  # (B,F,D_te)

        # 입력 채널 결합
        x = torch.cat([ego_future, time_feat, (~mask).float().unsqueeze(-1)], dim=-1)  # (B,F,Cin)
        h = self.channel_pre_project(x)
        h = h.permute(0, 2, 1)
        h = self.token_pre_project(h)
        h = h.permute(0, 2, 1)

        for block in self.blocks:
            h = block(h)                                                               # (B,F,C')

        # 시간 stem 바이패스
        valid_f = (~mask).float()                                                      # (B,F)
        z_bar = masked_mean_lastdim(te_stem, valid_f)                                  # (B,tstem)
        h = add_time_bypass(h, z_bar, self.time_bypass)                                # (B,F,C')

        # 투영/요약
        h = self.norm(h)
        tokens = self.emb_project(h.reshape(B * F, -1)).view(B, F, -1)                 # (B,F,H)
        global_vec = self.emb_project(h.mean(dim=1))                                    # (B,H)

        # pos(앞4=기하, 뒤4=타입 one-hot)
        pos = torch.zeros((B, F, 8), device=ego_future.device, dtype=ego_future.dtype)
        pos[..., :4] = ego_future
        pos[..., -4:] = 0.0
        pos[..., -4] = 1.0
        pos[mask] = 0.0

        time_anchor = t.view(1, F).expand(B, -1)                                       # (B,F)
        return tokens, mask, pos, global_vec, time_anchor


class FusionEncoder(nn.Module):
    """시나리오 토큰 통합용 셀프-어텐션 인코더 (시간 바이어스 지원).

    Args:
      hidden_dim (int): 히든 차원 D.
      num_heads (int): 어텐션 헤드 수.
      drop_path_rate (float): 드롭패스.
      depth (int): 블록 수.
      device (str): 디바이스 문자열.
      time_bias_beta (float): 시간 바이어스 강도 β.

    Shape:
      - 입력 x: (B, L, D), mask: (B, L) bool, time_anchors: (B, L) or None
      - 출력: (B, L, D)
    """
    def __init__(self,
                 hidden_dim: int = 192,
                 num_heads: int = 6,
                 drop_path_rate: float = 0.3,
                 depth: int = 3,
                 device: str = 'cuda',
                 time_bias_beta: float = 0.7):
        super().__init__()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(hidden_dim, num_heads, dropout=drop_path_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.time_bias_beta = time_bias_beta

    def forward(self, x: Tensor, mask: Tensor, time_anchors: Optional[Tensor] = None) -> Tensor:
        # (선택) 시간 바이어스 생성
        attn_bias = build_time_attn_bias(time_anchors, self.time_bias_beta, x.dtype, x.device)
        # 원래 로직 유지
        mask[:, 0] = False
        for b in self.blocks:
            x = b(x, mask, attn_bias=attn_bias)
        return self.norm(x)


class Encoder(nn.Module):
    """전체 인코더 오케스트레이션.

    Note:
      - StaticFusionEncoder, LaneFusionEncoder는 기존 구현을 사용한다고 가정.

    Args:
      config: 설정 객체. 다음 속성을 사용:
        - hidden_dim, encoder_drop_path_rate, encoder_depth, num_heads, device
        - future_len, time_len, lane_len, static_objects_state_dim
        - dt(=0.1), time_fourier_dim(=12), time_max_sec(=8.0), time_stem_dim(=64)
        - agent_segments_K(=2), agent_segment_sigma_scale(=0.5), time_bias_beta(=0.7)

    Returns:
      Dict[str, Tensor]:
        - 'encoding': (B, L, H) 통합 토큰
        - 'ego_future_global': (B, H) ego 전역 요약
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim

        # 공유 TimeStem
        d_fourier = getattr(config, "time_fourier_dim", 12)
        Tmax = getattr(config, "time_max_sec", 8.0)
        time_stem_dim = getattr(config, "time_stem_dim", 64)
        self.time_stem = TimeStem(in_dim=2 * d_fourier + 2, out_dim=time_stem_dim)

        # 서브 인코더들
        self.ego_future_encoder = EgoFutureEncoder(
            config.future_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            time_fourier_dim=d_fourier,
            time_max_sec=Tmax,
            dt=getattr(config, "dt", 0.1),
            time_stem=self.time_stem
        )
        self.neighbor_encoder = AgentFusionEncoder(
            config.time_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            time_fourier_dim=d_fourier,
            time_max_sec=Tmax,
            dt=getattr(config, "dt", 0.1),
            segments_K=getattr(config, "agent_segments_K", 2),
            segment_sigma_scale=getattr(config, "agent_segment_sigma_scale", 0.5),
            time_stem=self.time_stem
        )
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
            device=config.device,
            time_bias_beta=getattr(config, "time_bias_beta", 0.7)
        )

        # pos 임베딩(기존 포맷 유지: 8 -> H)
        self.pos_emb = nn.Linear(8, config.hidden_dim)

    def forward(self, inputs) -> dict:
        encoder_outputs = {}

        # 입력 파싱
        ego_past = inputs["ego_agent_past"].unsqueeze(1)  # (B, 1, V, D)
        neighbors = inputs["neighbor_agents_past"]        # (B, P, V, D)
        static = inputs["static_objects"]                 # (B, Ps, D)
        ego_future = inputs["ego_future"]                 # (B, F, 4)
        lanes = inputs["lanes"]                           # (B, Pl, V_l, D_l)
        lanes_speed_limit = inputs["lanes_speed_limit"]   # (B, Pl, 1)
        lanes_has_speed_limit = inputs["lanes_has_speed_limit"]  # (B, Pl, 1)

        B = neighbors.shape[0]

        # 서브 인코더
        enc_future, future_mask, future_pos, future_global, future_time = \
            self.ego_future_encoder(ego_future)  # (B,F,H), (B,F), (B,F,8), (B,H), (B,F)

        enc_neighbors, neighbors_mask, neighbor_pos, neighbors_time = \
            self.neighbor_encoder(ego_past, neighbors)  # (B,M*K,H), (B,M*K), (B,M*K,8), (B,M*K)

        enc_static, static_mask, static_pos = self.static_encoder(static)  # (B,Ps,H), (B,Ps), (B,Ps,8)
        enc_lanes, lanes_mask, lane_pos = self.lane_encoder(lanes, lanes_speed_limit, lanes_has_speed_limit)

        # 시간 앵커(정적/차선=0초로 둠)
        static_time = torch.zeros((B, enc_static.shape[1]), device=enc_static.device, dtype=enc_static.dtype)
        lanes_time = torch.zeros((B, enc_lanes.shape[1]), device=enc_lanes.device, dtype=enc_lanes.dtype)

        # 컨캣
        enc_cat = torch.cat([enc_future, enc_neighbors, enc_static, enc_lanes], dim=1)  # (B,L,H)
        pos_cat = torch.cat([future_pos, neighbor_pos, static_pos, lane_pos], dim=1)    # (B,L,8)
        mask_cat = torch.cat([future_mask, neighbors_mask, static_mask, lanes_mask], dim=1)  # (B,L)
        time_cat = torch.cat([future_time, neighbors_time, static_time, lanes_time], dim=1)  # (B,L)

        # pos 임베딩 주입(유효 토큰만)
        pos_out = inject_pos_emb(pos_cat, mask_cat, self.pos_emb, self.hidden_dim)      # (B,L,H)
        enc_cat = enc_cat + pos_out

        # Fusion
        encoder_outputs['encoding'] = self.fusion(enc_cat, mask_cat, time_anchors=time_cat)  # (B,L,H)
        encoder_outputs['ego_future_global'] = future_global                                 # (B,H)
        return encoder_outputs
