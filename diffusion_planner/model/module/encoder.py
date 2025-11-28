from timm.models.layers import Mlp
from timm.layers import DropPath
import torch.nn.functional as F
import math
from diffusion_planner.model.module.mixer import MixerBlock
from flash_attn.bert_padding import unpad_input, pad_input

# ==== (encoder.py 상단 import 근처에 추가) ====
from typing import Tuple  # 이미 있으면 중복 추가 불필요

import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def profile_block(name: str,
                  enabled: bool = True,
                  device_type: str = "cuda") -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력하는 간단한 프로파일러.

    Args:
        name: 출력에 사용할 블록 이름.
        enabled: False 이면 아무 것도 하지 않고 그냥 통과.
        device_type: "cuda" 인 경우, GPU 연산 정합을 위해 앞/뒤에 synchronize 호출.
    """
    if not enabled:
        # 아무 것도 하지 않고 블록만 실행
        yield
        return

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time: float = time.perf_counter()

    yield  # 실제 코드 실행

    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms: float = (time.perf_counter() - start_time) * 1000.0
    print(f"[PROFILE] {name}: {elapsed_ms:.3f} ms")


# ===== FlashAttention-2 varlen import (2.x 표준 경로 + 백업 경로) =====
try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func

    try:
        # 일부 버전(옛 코드) 표기
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_q_kvpacked_func as flash_attn_varlen_cross_func)
    except ImportError:
        # flash-attn 2.8.x의 정식 이름
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_kvpacked_func as flash_attn_varlen_cross_func)
    _FA2_AVAILABLE = True
    _FA2_IMPORT_ERR = None
except Exception as _e1:
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func

        try:
            from flash_attn import (flash_attn_varlen_q_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        except ImportError:
            from flash_attn import (flash_attn_varlen_kvpacked_func as
                                    flash_attn_varlen_cross_func)
        _FA2_AVAILABLE = True
        _FA2_IMPORT_ERR = None
    except Exception as _e2:
        _FA2_AVAILABLE = False
        _FA2_IMPORT_ERR = Exception(
            f"interface import err: {_e1}; top-level err: {_e2}")
        flash_attn_varlen_cross_func = None
# ===========================================================

# ===========================================================

# ============================================

from typing import Tuple, Dict, Optional
import torch
import torch.nn as nn
from typing import Iterable


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
    return base.reshape(1, 1, total_steps).expand(B, agents_num,
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
    return base.reshape(1, total_steps).expand(B, total_steps).clone()


class NearAgentsRouteLaneEncoder(nn.Module):
    """
    (route_num, D) → (D) 경량 Set-Encoder.
    입력 route_lanes는 이미 LaneEncoder/Fusion에서 위치/신호 등 정보가 반영된 토큰들이라고 가정.

    구성:
      1) LayerNorm
      2) Seeded attentional pooling (PMA-lite, 순열 invariant)
      3) DeepSets(mean/max) 잔차 (스칼라 게이트 0에서 시작 → 학습되며 켜짐)
      4) all-off(모든 route가 pad)인 에이전트는 0 벡터 반환

    입력:
      route_lanes:       (B, Pnn, route_num, D)
      route_lanes_mask:  (B, Pnn, route_num)   # True=pad(무효)

    출력:
      near_agents_route_lane_emb: (B, Pnn, D)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int = 4,
        attn_drop_p: float = 0.0,
        out_drop_p: float = 0.0,
        ffn_ratio: float = 2.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_seeds = int(num_seeds)

        # 1) 입력 정규화 + 1층 Self-Attention (batch_first=True)
        self.in_norm = nn.LayerNorm(hidden_dim)
        # Self-Attn 뒤 소형 FFN(잔차 게이트 0 시작)
        ffn_hidden = int(hidden_dim * ffn_ratio)

        # 2) Seeded attentional pooling (학습 쿼리 k개)
        self.seeds = nn.Parameter(torch.zeros(1, self.num_seeds, hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)

        # seed 요약 후 소형 FFN(게이트 0 시작)
        self.seed_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, hidden_dim, bias=True),
        )
        self.seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # seed 가중합 → 단일 벡터
        self.seed_gate = nn.Linear(hidden_dim, 1, bias=True)

        # 3) DeepSets(mean/max) 잔차(스칼라 게이트 0 시작)
        self.ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        # 4) 출력 정규화/드롭아웃
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_drop = nn.Dropout(
            out_drop_p) if out_drop_p > 0 else nn.Identity()

        self.attn_drop_p = float(attn_drop_p)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B_Pnn, route_num, D), mask: (B_Pnn, route_num) True=pad → (B_Pnn, D)
        """
        valid = (~mask).to(x.dtype)  # (B_Pnn,route_num)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: (B_Pnn, route_num, D), mask: (B_Pnn, route_num) True=pad → (B_Pnn, D)
        전부 마스크(N행)는 0 벡터로 반환.
        """
        very_neg = torch.finfo(
            x.dtype).min if x.dtype in (torch.float16,
                                        torch.bfloat16) else -1e30
        x_masked = x.masked_fill(mask.unsqueeze(-1), very_neg)
        mx = x_masked.amax(dim=1)  # (B_Pnn,D)
        all_off = mask.all(dim=1)  # (B_Pnn,)
        if all_off.any().item():
            mx = torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)
        return mx

    def forward(
            self,
            route_lanes: torch.Tensor,  # (B, Pnn, route_num, D)
            route_lanes_mask: torch.Tensor,  # (B, Pnn, route_num) True=pad
    ) -> torch.Tensor:  # (B, Pnn, D)
        assert route_lanes.dim() == 4 and route_lanes_mask.dim() == 3
        B, Pnn, route_num, D = route_lanes.shape
        assert D == self.hidden_dim, f"hidden_dim mismatch: got {D}, expected {self.hidden_dim}"
        assert route_lanes_mask.shape == (B, Pnn, route_num)

        # ---- 준비: dtype/device & 모양 변환 ----
        route_lanes = self.in_norm(route_lanes)  # (B,Pnn,route_num,D)
        route_lanes_mask = route_lanes_mask.to(torch.bool)  # (B,Pnn,route_num)
        route_lanes = route_lanes.masked_fill(route_lanes_mask.unsqueeze(-1),
                                              0.0)  # 패딩은 0 고정

        B_Pnn = B * Pnn
        route_lanes_3 = route_lanes.reshape(B_Pnn, route_num,
                                            D)  # (B_Pnn,route_num,D)
        route_lanes_mask_2 = route_lanes_mask.reshape(
            B_Pnn, route_num)  # (B_Pnn,route_num)
        no_route_agent_mask = route_lanes_mask_2.all(dim=1)  # (B_Pnn,)

        # ---- 2) Seeded attentional pooling (PMA-lite) ----
        # (1, num_seeds, D) → (B_Pnn, num_seeds, D)
        seeds = self.seeds.to(route_lanes_3.dtype).expand(
            B_Pnn, self.num_seeds, D)
        # 점수: (B_Pnn,num_seeds,route_num) = (B_Pnn,num_seeds,D) @ (B_Pnn,D, route_num) / sqrt(D)
        logits = torch.einsum("nkd,nrd->nkr", seeds, route_lanes_3) / math.sqrt(
            max(1.0, float(D)))
        # 마스크: pad(True) → -inf
        if route_lanes_mask_2.any().item():  # (B_Pnn,route_num)
            logits = logits.masked_fill(route_lanes_mask_2.unsqueeze(1),
                                        float("-inf"))
        # 전부 마스크 행 softmax NaN 방지
        if no_route_agent_mask.any().item():  # (B_Pnn,)
            logits = logits.clone()
            logits[no_route_agent_mask] = 0.0

        attn = F.softmax(logits, dim=-1)
        if self.attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop_p)

        # (B_Pnn,num_seeds,route_num) @ (B_Pnn,route_num,D) = (B_Pnn,num_seeds,D)
        y_seed = torch.einsum("nkr,nrd->nkd", attn,
                              route_lanes_3)  # (B_Pnn,num_seeds,D)
        # 소형 FFN 잔차(게이트 0 시작)
        y_seed = y_seed + self.seed_ffn_alpha.to(
            y_seed.dtype) * self.seed_ffn(y_seed)

        # seed 가중합 → (B_Pnn,D)
        w = F.softmax(self.seed_gate(y_seed).squeeze(-1),
                      dim=1).unsqueeze(-1)  # (B_Pnn,num_seeds,1)
        y_pool = (w * y_seed).sum(dim=1)  # (B_Pnn,D)

        # ---- 3) DeepSets(mean/max) 잔차 ----
        ds_mean = self._masked_mean(route_lanes_3,
                                    route_lanes_mask_2)  # (B_Pnn,D)
        ds_max = self._masked_max(route_lanes_3,
                                  route_lanes_mask_2)  # (B_Pnn,D)
        out = y_pool \
              + self.ds_mean_alpha.to(y_pool.dtype) * ds_mean \
              + self.ds_max_alpha.to(y_pool.dtype) * ds_max  # (B_Pnn,D)

        # all-off 는 0 보장
        if no_route_agent_mask.any().item():
            out = torch.where(no_route_agent_mask.unsqueeze(-1),
                              torch.zeros_like(out), out)

        # ---- 4) 정규화/드롭아웃 & 모양 복원 ----
        out = self.out_drop(self.out_norm(out))  # (B_Pnn,D)
        return out.view(B, Pnn, D)  # (B,Pnn,D)


class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.route_order_drop_prob: float = getattr(config,
                                                    "route_order_drop_prob",
                                                    0.5)

        self.chunk_length = 20
        self.num_fourier_frequencies = 4
        self.pos_scale = nn.Parameter(torch.tensor(1.0))
        self.time_gap = 0.1
        self.time_min = -(config.time_len - 1) * self.time_gap
        self.time_max = config.future_len * self.time_gap
        self.config = config
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
        self.npc_route_encoder = NearAgentsRouteLaneEncoder(
            hidden_dim=config.hidden_dim, attn_drop_p=0.0)
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
        # type (ego, neighbor, static, lane)
        self.pos_emb = nn.Linear(8, config.hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _zero_with_touch(self, ref: torch.Tensor,
                         params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        """ref와 같은 shape의 0 텐서를 반환하되, 주어진 파라미터들을 0계수로 터치해
        autograd 그래프를 연결(grad는 0)."""
        # 스칼라 누산(기기/dtype 맞춤)
        touch = ref.new_zeros(())
        for p in params:
            # 첫 원소만 살짝 참조 → 비용 최소화
            touch = touch + p.view(-1)[:1].sum()
        return ref.new_zeros(ref.shape) + touch * 0.0  # broadcast OK

    def _ego_fut_params(self) -> Iterable[torch.nn.Parameter]:
        for n, p in self.agents_encoder.named_parameters():
            if n.startswith("ego_fut_"):
                yield p

    @staticmethod
    def _mask_all_routes_like(
        route_lanes: torch.Tensor,  # (B, Pnn, route_num, H)
        route_lanes_mask: torch.Tensor,  # (B, Pnn, route_num)
        drop_mask_b: torch.Tensor,  # (B,) bool, False인 샘플을 "경로 없음"으로 강제
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 마스크에 따라 해당 샘플의 모든 route 토큰을 패딩 처리합니다.

        Args:
            route_lanes: (B, Pnn, route_num, H)
            route_lanes_mask: (B, Pnn, route_num)  # True=pad
            drop_mask_b: (B,) bool         # True=유지(keep), False=드롭(경로 없음)

        Returns:
            (route_lanes, route_lanes_mask)  # same shapes
        """
        if drop_mask_b.dtype != torch.bool:
            drop_mask_b = drop_mask_b.to(torch.bool)

        if (~drop_mask_b).any().item():
            # (B,) -> (B,1,1) for broadcast
            b_drop = (~drop_mask_b).view(-1, 1, 1)  # True = 드롭 대상 배치
            # 값/포지션: float 복사 마스크로 곱셈(새 텐서 반환)
            keep_f_lanes = (~b_drop).unsqueeze(-1).to(
                route_lanes.dtype)  # (B,1,1,1)
            route_lanes = route_lanes * keep_f_lanes

            # 마스크: out‑of‑place 결합(원본 텐서 저장소를 수정하지 않음)
            route_lanes_mask = route_lanes_mask | b_drop.expand_as(
                route_lanes_mask)
        return route_lanes, route_lanes_mask

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
        ego_type = ego_future_full[:, 0, 8:11].clone()  # (B, 3)

        # 기본 패딩 텐서: 앞 8채널=0, type 3채널=ego one-hot 반복
        padded_default = ego_future_full.new_zeros(B, N, D)  # (B, N, 11)
        padded_default[:, :, 8:11] = ego_type.unsqueeze(1).expand(-1, N, -1)

        # keep: (B, N, 1)  True=원본 유지
        keep = known_mask.unsqueeze(-1)

        # 최종: 알려진 구간은 원본, 나머지는 기본 패딩
        truncated_ego_future = torch.where(keep, ego_future_full,
                                           padded_default)  # (B, N, 11)
        return truncated_ego_future

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인코더 전방 패스(무작위 길이 M으로 ego 미래를 잘라 조건 제공).

        입력 딕셔너리 키와 텐서 형태:
            - ego_agent_past:           [B, V=21, 11]
            - planner_future_11_dim:     [B, N=80, 11]
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
        """
        # ego
        device_type = inputs["ego_agent_past"].device.type
        with profile_block(
                "encoder.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            ego_past = inputs[
                "ego_agent_past"]  # (B, V=21, D=11) -> (B, 1, V, D)
            if not self.config.use_vel_input:
                ego_past[:, :, 4:6] = 0.0  # vx, vy 0으로 설정
            ego_past = ego_past.unsqueeze(1)  # Add a dimension for P
            # (B ,future_len= 80, 11)
            planner_future_11_dim = inputs["planner_future_11_dim"]
            # agents
            # (B, A, V=21, D=11)
            neighbors = inputs["neighbor_agents_past"]
            if not self.config.use_vel_input:
                neighbors[:, :, :, 4:6] = 0.0  # vx, vy 0으로 설정

            # static objects
            # (B, P, D_static)
            static = inputs["static_objects"]

            # vector maps
            # (B, L, lane_len, D_lane)
            lanes = inputs["lanes"]
            # ( B, L, 1) 속도 제한
            lanes_speed_limit = inputs["lanes_speed_limit"]
            # (B, L, 1) 속도 제한 여부
            lanes_has_speed_limit = inputs["lanes_has_speed_limit"]
            # (B, Pnn, lane_num) # -1: route가 아님
            agent_route_lane_order = inputs["agent_route_lane_order"]

            B = neighbors.shape[0]
            # TODO : 임시 테스트
            self.neglect_route_encoding = False
            if not self.training and self.neglect_route_encoding:
                agent_route_lane_order = torch.full_like(agent_route_lane_order,
                                                         fill_value=-1)

            future_len: int = planner_future_11_dim.shape[1]
            if self.training:
                # ---------------------- 1) M_i 샘플링 ---------------------- #
                prefix_lengths = self._sample_uniform_prefix_lengths(
                    batch_size=B,
                    max_future_len=future_len,
                    device=planner_future_11_dim.device,
                )  # (B,)
                known_mask = self._build_known_mask_from_lengths(
                    prefix_lengths,
                    max_future_len=future_len)  # (B, future_len) True=조건 제공

                # ---------------------- 2) 잘라 + 패딩 ---------------------- #
                ego_future_trajectory = self._truncate_and_pad_ego_future_for_encoder(
                    planner_future_11_dim,
                    known_mask)  # (B, future_len, 11)  길이 유지, 마스크는 내부에서 활용됨
            else:
                # TODO: "using" ego_agent_next_11_dim 도 해보자.
                ego_future_trajectory = planner_future_11_dim
                if ego_future_trajectory is None:
                    ego_future_trajectory = torch.zeros((B, future_len, 11),
                                                        device=ego_past.device,
                                                        dtype=ego_past.dtype)
            if not self.config.use_vel_input:
                ego_future_trajectory[:, :, 4:6] = 0.0  # vx, vy 0으로 설정
            # ---------------------- 3) 인코딩 ---------------------- #
            # ego_fut_global: (B, hidden_dim)
            """
            # encoding_agents_chunk: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
            # agents_chunk_mask: (B, agents_num * past_cur_chunk_num + future_chunk_num)
            # agents_chunk_pos : (B, agents_num * past_cur_chunk_num + future_chunk_num, 8)
            # ego_fut_global: (B, hidden_dim)
            """
            (encoding_agents_chunk, agents_chunk_mask, agents_chunk_pos,
             ego_fut_global) = self.agents_encoder(ego_past, neighbors,
                                                   ego_future_trajectory)
            if not self.config.use_pram:
                ego_fut_global = self._zero_with_touch(ego_fut_global,
                                                       self._ego_fut_params())
            """
            encoding_static: (B, static_objects_num, hidden_dim)
            static_mask: (B, static_objects_num)
            static_pos: (B, static_objects_num, 8)
            """
            encoding_static, static_mask, static_pos = self.static_encoder(
                static)
            """
            encoding_lanes: (B, lane_num, hidden_dim)
            lanes_mask: (B, lane_num)
            lane_pos: (B, lane_num, 8)
            """
            encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
                lanes, lanes_speed_limit, lanes_has_speed_limit)

            # ---------------------- 4) 포지션 임베딩 결합 ---------------------- #
            """
    token_num = (agents_num * past_cur_chunk_num + future_chunk_num) + static_objects_num + lane_num
            encoding_input: (B, token_num, hidden_dim)
            encoding_pos: (B * token_num, 8)
            encoding_mask: (B * token_num)
    
            """
            encoding_input = torch.cat(
                [encoding_agents_chunk, encoding_static, encoding_lanes], dim=1)
            encoding_mask = torch.cat(
                [agents_chunk_mask, static_mask, lanes_mask], dim=1).reshape(-1)
            encoding_pos = torch.cat([agents_chunk_pos, static_pos, lane_pos],
                                     dim=1).reshape(B * self.token_num, -1)

            # 결합 후 실제 길이
            token_num_actual = encoding_agents_chunk.size(
                1) + encoding_static.size(1) + encoding_lanes.size(1)
            # (선택) 방어적 체크
            assert token_num_actual == self.token_num, \
                f"token_num mismatch: expected {self.token_num}, got {token_num_actual}"
            # encoding_pos: (on_token_num, hidden_dim)
            pos_valid = self.pos_emb(encoding_pos[~encoding_mask])
            pos_valid = pos_valid.to(dtype=encoding_input.dtype,
                                     device=encoding_input.device)
            scale = self.pos_scale.to(dtype=encoding_input.dtype,
                                      device=encoding_input.device)
            encoding_pos = scale * pos_valid
            encoding_pos_result = torch.zeros(
                (B * self.token_num, self.hidden_dim),
                device=encoding_input.device,
                dtype=encoding_input.dtype)
            # encoding_pos_result: (B * token_num, hidden_dim)
            encoding_pos_result[
                ~encoding_mask] = encoding_pos  # Fill in valid parts

            # (B, token_num, hidden_dim)
            encoding_input = encoding_input + encoding_pos_result.reshape(
                B, self.token_num, -1)
            # get encoding_input_lanes from encoding_input
            # (B, lane_num, hidden_dim)
            encoder_outputs = {}
            encoding_tokens, encoding_mask = self.fusion(
                encoding_input, encoding_mask.reshape(B, self.token_num))
            encoder_outputs[
                "encoding"] = encoding_tokens  # (B, token_num, hidden_dim)
            encoder_outputs["encoding_mask"] = encoding_mask  # (B, token_num)
            encoder_outputs["ego_fut_global"] = ego_fut_global
            encoding_lanes = encoding_input[:, -encoding_lanes.size(
                1):, :]  # (B, lane_num, hidden_dim)

            (near_agents_route_lane_emb,
             route_known_mask) = self._get_near_agents_route_lane_emb(
                 encoding_lanes, lanes_mask, agent_route_lane_order)
            if not self.config.use_pram:
                near_agents_route_lane_emb = self._zero_with_touch(
                    near_agents_route_lane_emb,
                    self.npc_route_encoder.parameters())
                route_known_mask = torch.zeros_like(route_known_mask).bool()

            encoder_outputs[
                "near_agents_route_lane_emb"] = near_agents_route_lane_emb  # (B, Pnn, hidden_dim)
            encoder_outputs[
                "route_known_mask"] = route_known_mask  # (B, Pnn) True=해당 에이전트가 유효 route
            return encoder_outputs

    def _get_near_agents_route_lane_emb(
        self,
        encoding_lanes: torch.Tensor,  # (B, lane_num, hidden_dim)
        lanes_mask: torch.Tensor,  # (B, lane_num)  True=pad
        agent_route_lane_order: torch.
        Tensor,  # (B, Pnn, lane_num)  -1=not in route
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # agent_route_lane_order: (B, Pnn, lane_num)
        # -1: route가 아님 # agent당 유효한 route의 최대 수는 최대 route_num 개임. (즉, route_num 이하임) # route_num <= lane_num 항상

        (route_lanes,
         route_lanes_mask) = self.build_route_lane_tensors_from_order(
             encoding_lanes, lanes_mask, agent_route_lane_order,
             self.config.route_num)
        """
            route_lanes:       (B, Pnn, route_num, H)
            route_lanes_mask:  (B, Pnn, route_num)  # True=pad(무효)
        Returns:
            near_agents_route_lane_emb: (B, Pnn, H)
            route_known_mask : (B, Pnn) True=해당 에이전트가 유효 route
        """
        B = encoding_lanes.shape[0]
        if self.training and self.route_order_drop_prob > 0.0:
            # route_keep_mask[b] = True면 주어진 order를 사용, False면 "경로 없음"
            route_keep_mask: torch.Tensor = (torch.rand(
                (B,), device=encoding_lanes.device) >= float(
                    self.route_order_drop_prob))  # (B,) bool

            # 샘플 단위 드롭을 실제 텐서에 반영 (전부 패딩 처리)
            (route_lanes, route_lanes_mask) = self._mask_all_routes_like(
                route_lanes, route_lanes_mask, route_keep_mask)
        # (B, Pnn, hidden_dim)
        near_agents_route_lane_emb = self.npc_route_encoder(
            route_lanes, route_lanes_mask)
        # (B, Pnn) True=해당 에이전트가 유효 route를 가짐
        route_known_mask = (~route_lanes_mask).any(
            dim=-1)  # (B, Pnn) True=known
        return near_agents_route_lane_emb, route_known_mask

    @staticmethod
    def build_route_lane_tensors_from_order(
            encoding_lanes: torch.Tensor,  # (B, lane_num, hidden_dim)
            lanes_mask: torch.Tensor,  # (B, lane_num)  # True=pad
            agent_route_lane_order: torch.
        Tensor,  # (B, Pnn, lane_num)  # -1=not in route, 0..=rank(가까운 순)
            route_num: int,  # 선택할 최대 route 개수(≤ lane_num)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """에이전트-별 랭크(agent_route_lane_order)에 따라, 가까운 순으로 최대 route_num개의
        차선을 선택해 (임베딩, 마스크)을 에이전트 축으로 확장해 반환합니다.

        Returns:
            route_lanes:       (B, Pnn, route_num, hidden_dim)
            route_lanes_mask:  (B, Pnn, route_num)  # True=pad(무효)
        """
        B, lane_num, hidden_dim = encoding_lanes.shape
        Pnn = agent_route_lane_order.shape[1]
        # === [ADD] torch.topk(k=0) 방지: route_num을 최소 1로 보정 ===
        route_num = int(max(1, route_num))
        # (B, Pnn, lane_num)  # -1=not in route → 큰 값으로 바꿔서 '최소값 top-k'에서 탈락시키기
        route_lane_order = agent_route_lane_order.to(torch.long)
        # lanes_mask: (B, lane_num) -> (B, 1, lane_num) -> (B, Pnn, lane_num)
        lanes_mask_exp = lanes_mask.unsqueeze(1).expand(
            B, Pnn, lane_num)  # (B, Pnn, lane_num)
        # valid_route_mask: (B, Pnn, lane_num)  # 내 route이면서 lane 유효
        valid_route_mask = (route_lane_order >= 0) & (~lanes_mask_exp)

        BIG = 2**30  # 충분히 큰 값(랭크보다 큼)
        # order_for_sort: (B, Pnn, lane_num)  # 무효 위치는 BIG으로 채움
        order_for_sort = torch.where(valid_route_mask, route_lane_order,
                                     torch.full_like(route_lane_order, BIG))

        # 에이전트별 '랭크가 작은 순'으로 최대 route_num개 선택
        # vals: (B, Pnn, route_num), valid_lane_order_idx: (B, Pnn, route_num)
        vals, valid_lane_order_idx = torch.topk(order_for_sort,
                                                k=route_num,
                                                dim=-1,
                                                largest=False)
        route_lane_valid_mask = vals != BIG  # (B, Pnn, route_num)

        # --- gather 준비 ---
        # 임베딩: (B, 1, lane_num, hidden_dim) -> (B, Pnn, lane_num, hidden_dim)
        encoding_lanes_ = encoding_lanes.unsqueeze(1).expand(
            B, Pnn, lane_num, hidden_dim)

        # 인덱스 확장
        gather_idx_H = valid_lane_order_idx.unsqueeze(-1).expand(
            B, Pnn, route_num, hidden_dim)  # (B,Pnn,route_num,hidden_dim)

        # --- gather ---
        # (B, Pnn, route_num, hidden_dim)
        route_lanes = torch.gather(encoding_lanes_, 2, gather_idx_H)

        # 마스크: lanes_mask에서 같은 인덱스 gather 후, sel_valid로 보강
        # (B, lane_num) -> (B, 1, lane_num) -> (B, Pnn, lane_num)
        lanes_mask_ = lanes_mask.unsqueeze(1).expand(B, Pnn, lane_num)
        # (B,Pnn,route_num)
        gathered_lane_mask = torch.gather(lanes_mask_, 2, valid_lane_order_idx)
        # route_lane_valid_mask: (B,Pnn,route_num)  # True=유효
        # route_lanes_mask: (B,Pnn,route_num)  # True=pad/무효
        route_lanes_mask = gathered_lane_mask | (~route_lane_valid_mask
                                                )  # True=pad(무효)

        # 무효 위치는 0으로 채우기 (곱셈으로 처리하여 마스크 alias 문제 제거)
        # (B,Pnn,R,1) float copy
        mask_f = route_lanes_mask.unsqueeze(-1).to(route_lanes.dtype)
        route_lanes = route_lanes * (1 - mask_f)  # (B,Pnn,R,hidden_dim)

        return route_lanes, route_lanes_mask


class SelfAttentionBlock(nn.Module):

    def __init__(
            self,
            dim=192,
            heads=8,
            attn_drop_p: float = 0.0,  # 어텐션 드롭아웃
            ffn_drop_p: float = 0.0,  # FFN 드롭아웃
            drop_path_p: float = 0.0,  # Stochastic Depth
            mlp_ratio=4.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        # 원본의 self.attn(nn.MultiheadAttention)은 폴백 경로에서만 사용
        self.use_fallback_mha = not _FA2_AVAILABLE
        if self.use_fallback_mha:
            self.attn = nn.MultiheadAttention(dim,
                                              heads,
                                              attn_drop_p,
                                              batch_first=True)
        else:
            self.attn = None
        self.attn_out_drop = nn.Dropout(attn_drop_p)
        self.drop_path = DropPath(
            drop_path_p) if drop_path_p > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=nn.GELU,
                       drop=ffn_drop_p)

        # === FlashAttention‑2용 QKV/출력 프로젝션 ===
        self.num_heads = heads
        self.head_dim = dim // heads
        assert dim % heads == 0, f"dim({dim}) must be divisible by heads({heads})"
        if (self.head_dim % 8) != 0:
            # 권장: 8 배수
            print(f"[Warning] head_dim={self.head_dim} (not multiple of 8). "
                  "FlashAttention‑2 성능이 저하될 수 있습니다.")

        # 유효 토큰에만 적용하는 선형 레이어
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        # FlashAttention dropout 확률(학습 시에만 사용)
        self._attn_dropout_p = attn_drop_p

    # ------------------------------------------------------------------
    # 아래 유틸리티 함수들은 varlen 커널 구동을 위한 핵심 로직입니다.
    # 모두 타입 힌트와 Google Style 한글 docstring, 자주 쓰이는 shape 주석 포함.
    # ------------------------------------------------------------------

    @staticmethod
    def _get_compute_dtype(x: torch.Tensor) -> torch.dtype:
        """연산 dtype을 선택합니다(BF16/FP16 우선).

        1) autocast가 켜져 있으면 해당 dtype을 우선 사용합니다.
        2) 입력 텐서 `x`가 FP16/BF16이면 그대로 사용합니다.
        3) 그 외에는 GPU 아키텍처에 따라 BF16(>=SM80) 또는 FP16을 반환합니다.

        Args:
            x (torch.Tensor): 임의 텐서. (shape 무관)

        Returns:
            torch.dtype: torch.float16 또는 torch.bfloat16
        """
        if torch.is_autocast_enabled():
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                pass

        if x.dtype in (torch.float16, torch.bfloat16):
            return x.dtype

        if x.is_cuda and torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(x.device)
            return torch.bfloat16 if major >= 8 else torch.float16

        return torch.float16

    @staticmethod
    def _unpad_from_mask(
        x: torch.Tensor,  # (B, L, D)
        mask: torch.Tensor  # (B, L)  True=pad(무효)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """패딩 마스크로부터 유효 토큰만 추출(unpad)

        FlashAttention‑2 varlen 커널은 (B, L, D)를 직접 받지 않고, **유효 토큰을 연결한 2D 버퍼**와
        **배치별 누적 길이(cu_seqlens)**를 필요로 합니다. 또한 pad back(복원)을 위한 인덱스도 반환합니다.

        Args:
            x (torch.Tensor): 입력 시퀀스, 모양 (B, L, D).
            mask (torch.Tensor): 키 패딩 마스크(True=pad), 모양 (B, L).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
                x_unpad: (T, D) 유효 토큰만 이어붙인 텐서. T = sum(seqlens).
                indices: (T,)  원래 (B*L) 평탄화 인덱스에서 유효 토큰의 위치.
                cu_seqlens: (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0.
                max_seqlen: int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1).
                seqlens: (B,) int32  배치별 유효 토큰 개수.

        Note:
            - 유효 토큰이 전혀 없는 경우 T=0이 되어, 호출 측에서 안전하게 영 텐서를 반환하도록 합니다.
        """
        B, L, D = x.shape
        valid = (~mask).to(torch.bool)  # (B, L)
        seqlens = valid.sum(dim=1).to(torch.int32)  # (B,)
        cu_seqlens = torch.nn.functional.pad(seqlens.cumsum(dim=0),
                                             pad=(1, 0))  # [학습 input] # (B+1,)
        flat_valid = valid.reshape(B * L)  # (B*L,)
        # torch.nonzero: True인 원소들의 **인덱스(좌표)**를 반환. # as_tuple=False: (N,1) 반환.
        # long = int64
        indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1).to(
            torch.long)  # (T,) # [pad back용]
        x_unpad = x.reshape(B * L, D).index_select(0, indices)  # (T, D)
        max_seqlen = int(seqlens.max().item()) if B > 0 else 0  # [학습 input]
        """
        x_unpad: (T, D) 유효 토큰만 이어붙인 텐서. T = sum(seqlens). [학습 input]
        indices: (T,)  원래 (B*L) 평탄화 인덱스에서 유효 토큰의 위치. [pad back용]
        cu_seqlens: (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0. [학습 input]
        max_len= max_seqlen: int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1). [학습 input]
        seqlens = seqlens: (B,) int32  배치별 유효 토큰 개수.
        """
        return x_unpad, indices, cu_seqlens, max_seqlen

    @staticmethod
    def _pad_to_batch(
        y_unpad: torch.Tensor,
        indices: torch.Tensor,
        B: int,
        L: int,
        D: int,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if dtype is None:
            dtype = y_unpad.dtype
        out = torch.zeros(B * L, D, device=device, dtype=dtype)
        if y_unpad.numel() > 0:
            if y_unpad.dtype != dtype:
                y_unpad = y_unpad.to(dtype)
            out.index_copy_(0, indices, y_unpad)
        return out.view(B, L, D)

    def _self_attn_flash_varlen(
            self,
            x: torch.Tensor,  # (B, L, D)
            mask: torch.Tensor,  # (B, L) True=pad
    ) -> torch.Tensor:
        """FlashAttention‑2(varlen) Self‑Attention을 수행합니다.

        1) `mask`를 이용해 유효 토큰만 언패드 → 2) QKV/어텐션/출력 프로젝션을 유효 토큰에만 수행
        → 3) pad back으로 (B, L, D) 복원합니다. 패딩 토큰은 연산 경로에 참여하지 않으므로
        연산량과 메모리 사용량이 유효 길이에 비례합니다.

        Args:
            x (torch.Tensor): 입력(쿼리=키=값), 모양 (B, L, D).
            mask (torch.Tensor): 키 패딩 마스크(True=pad), 모양 (B, L).

        Returns:
            torch.Tensor: Self‑Attention 출력, 모양 (B, L, D).
        """
        if not _FA2_AVAILABLE:
            raise RuntimeError("FlashAttention‑2(varlen) 모듈을 불러오지 못했습니다. "
                               "pip install flash-attn>=2.3 등으로 설치 후 다시 시도하세요. "
                               f"(원인: {_FA2_IMPORT_ERR})")

            # TODO: cpu 만 사용가능할 때, PyTorch SDPA(패딩 포함) 사용하는 옵션 추가 (아래 주석 해제)
            # # FlashAttention‑2가 없으면 원래 경로로 폴백
            # y = self.attn(x, x, x, key_padding_mask=mask,
            #               need_weights=False)[0]  # (B, L, D)
            return y

        B, L, D = x.shape

        # (1) 언패드
        """
        x_unpad: (T, D) 유효 토큰만 이어붙인 텐서. T = sum(seqlens). [학습 input]
        indices: (T,)  원래 (B*L) 평탄화 인덱스에서 유효 토큰의 위치. [pad back용]
        cu_seqlens: (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0. [학습 input]
        max_len= max_seqlen: int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1). [학습 input]
        seqlens = seqlens: (B,) int32  배치별 유효 토큰 개수.
        """
        # x_unpad, indices, cu_seqlens, max_seqlen = self._unpad_from_mask(
        #     x, mask)  # x_unpad: (T, D)
        attention_mask = (~mask).to(torch.bool)  # True=유효

        res = unpad_input(x, attention_mask)

        # v2.7 이하: 4개 / v2.8.x: 5개
        if len(res) == 4:
            x_unpad, indices, cu_seqlens, max_seqlen = res
            # seqlens가 필요하면 cu_seqlens로부터 복원 가능
            seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        elif len(res) == 5:
            x_unpad, indices, cu_seqlens, max_seqlen, seqlens = res

        T = x_unpad.shape[0]
        if T == 0 or max_seqlen == 0:
            touch = (self.qkv_proj.weight.view(-1)[:1].sum() +
                     (self.qkv_proj.bias.view(-1)[:1].sum()
                      if self.qkv_proj.bias is not None else 0) +
                     self.out_proj.weight.view(-1)[:1].sum() +
                     (self.out_proj.bias.view(-1)[:1].sum()
                      if self.out_proj.bias is not None else 0)) * 0.0
            return torch.zeros_like(x) + touch

        # (2) QKV 프로젝션 (유효 토큰만)
        qkv = self.qkv_proj(x_unpad)  # (T, 3*D)
        qkv = qkv.view(T, 3, self.num_heads, self.head_dim)  # (T, 3, H, Hd)
        comp_dtype = self._get_compute_dtype(qkv)
        qkv = qkv.to(comp_dtype)

        # (3) FlashAttention‑2 varlen
        """
        # 원래 input x: (B, L, D) → 언패드 x_unpad: (T, D) → QKV: (T, 3, H, Hd)
        batch 내에서, 각 L 토큰들끼리만 서로 어텐션을 수행합니다.
        """
        # out: (T, H, Hd)
        out = flash_attn_varlen_qkvpacked_func(
            qkv,  # (T, 3, H, Hd)
            cu_seqlens=cu_seqlens.to(
                torch.int32),  # (B+1,) int32  배치별 누적 길이(prefix sum), 첫 원소는 0.
            max_seqlen=max_seqlen,  # int  배치 내 최대 유효 길이(≥1일 수 있음; CLS만 유효해도 1).
            dropout_p=self._attn_dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=False,
        )  # (T, H, Hd)

        # (4) 출력 프로젝션 + pad back
        out = out.reshape(T, self.num_heads * self.head_dim)  # (T, D)
        out = self.out_proj(out.to(x.dtype))  # (T, D) -> in dtype
        out = out.to(x.dtype)
        # out = self._pad_to_batch(out, indices, B, L, D, x.device)  # (B, L, D)
        out = pad_input(out, indices, B, L)  # (B, L, D)
        return out

    # ------------------------------------------------------------------

    def forward(self, x, mask):
        """
        x:  [on_B, 1 + token_num, H]
        mask: [on_B, 1 + token_num]  # True=pad

        Note:
            - 입력을 먼저 마스크로 0 클램프(원본 유지)한 뒤 LN → varlen Self‑Attention.
            - pad back된 출력은 마스크 위치가 0이며, 이후 MLP/DropPath를 통과하면서도
              마스크는 다시 0으로 클램프합니다(원본 동작과 정합).
        """
        # (on_B, L, H)
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)

        x_norm = self.norm1(x)  # (on_B, L, H)

        # === FlashAttention‑2(varlen) 또는 폴백 ===
        y = self._self_attn_flash_varlen(x_norm, mask)  # (on_B, L, H)

        y = self.attn_out_drop(y)
        x = x + self.drop_path(y)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.masked_fill(mask.unsqueeze(-1), 0.0)
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
        nn.init.normal_(self.type_emb.weight, std=0.02)

        # 8 = x, y, cos, sin, vx, vy, w, l,
        # 4 = delta x, delta y, cos (delta), sin (delta),
        # num_fourier_dim = 2K + 1,
        # 1 = validity
        self.channel_pre_project = Mlp(in_features=12 + num_fourier_dim + 1,
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

        ###################
        # --- Ego future global pooling: PMA-lite + DeepSets ---
        self.ego_fut_in_norm = nn.LayerNorm(hidden_dim)

        self.ego_fut_num_seeds = 4  # future_chunk_num이 작으므로 2~4 권장
        self.ego_fut_seeds = nn.Parameter(
            torch.zeros(1, self.ego_fut_num_seeds, hidden_dim))
        nn.init.trunc_normal_(self.ego_fut_seeds, std=0.02)

        ffn_hidden = int(hidden_dim * 2.0)  # 소형 FFN
        self.ego_fut_seed_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, hidden_dim, bias=True),
        )
        # 잔차 게이트(0에서 시작 → 학습되며 켜짐)
        self.ego_fut_seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # 여러 시드의 출력을 (B, hidden_dim) 하나로 합치는 게이트
        self.ego_fut_seed_gate = nn.Linear(hidden_dim, 1, bias=True)

        # DeepSets(mean/max) 잔차 게이트(0에서 시작)
        self.ego_fut_ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ego_fut_ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        # 출력 정규화/드롭
        self.ego_fut_out_norm = nn.LayerNorm(hidden_dim)
        self.ego_fut_out_drop = nn.Identity()  # 필요 시 nn.Dropout(p)로 교체 가능

        # 어텐션 드롭아웃(선택): 0~0.1 권장. 기본 0.
        self.ego_fut_attn_drop_p = 0.0

    @staticmethod
    def _masked_max_ego_chunks(ego_fut_chunk: torch.Tensor,
                               ego_fut_off_chunk_mask_full: torch.Tensor,
                               dim: int) -> torch.Tensor:
        """
        ego_fut_chunk: (B, future_chunk_num, hidden_dim)
        ego_fut_off_chunk_mask_full: (B, future_chunk_num)  True=무효
        return: (B, hidden_dim)
        """
        assert dim == 1, "dim은 1이어야 합니다."
        assert ego_fut_off_chunk_mask_full.dtype == torch.bool
        # FP16/BF16에서도 안전한 아주 작은 값
        very_neg = (torch.finfo(ego_fut_chunk.dtype).min if ego_fut_chunk.dtype
                    in (torch.float16, torch.bfloat16) else -1e30)
        x = ego_fut_chunk.masked_fill(ego_fut_off_chunk_mask_full.unsqueeze(-1),
                                      very_neg)
        mx = x.amax(dim=dim)  # (B, hidden_dim)
        all_off = ego_fut_off_chunk_mask_full.all(dim=dim)  # (B,)
        if all_off.any().item():
            mx = torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)
        return mx

    def _get_agents_past_cur_mask(
            self, agents_past_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: agents_past_current (B, agents_num, time_len, 12 + 2K + 1)

        Output:
            agents_past_cur_off_p_mask: (B, agents_num, time_len)
            agents_past_cur_off_mask: (B, agents_num)
        """
        agents_past_current_is_not_zero = torch.ne(
            agents_past_current[..., :8],
            0)  # (B, agents_num, time_len, 8) # 0이 아니면 True
        agents_past_current_not_zero_num = torch.sum(
            agents_past_current_is_not_zero,
            dim=-1).to(agents_past_current.device)  # (B, agents_num, time_len)
        # (B, agents_num, time_len)
        agents_past_cur_off_p_mask = agents_past_current_not_zero_num == 0  # 점의 정보가 없으면 True
        agents_past_cur_on_p_mask = ~agents_past_cur_off_p_mask  # (B, agents_num, time_len)
        agents_past_cur_off_mask = torch.sum(agents_past_cur_on_p_mask,
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
        agents_past_cur_on_mask = ~agents_past_cur_off_mask.reshape(
            -1)  # (B * agents_num)
        return agents_past_cur_on_p_mask, agents_past_cur_on_mask

    def _get_ego_future_mask(
            self,
            ego_future: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input: ego_future (B, future_len,  12 + 2K + 1)

        Output:
            ego_future_off_p_mask: (B, future_len)
            ego_future_off_mask: (B)
        """
        ego_future_is_not_zero = torch.ne(ego_future[..., :8],
                                          0)  # (B, future_len, 8) # 0이 아니면 True
        ego_future_not_zero_num = torch.sum(ego_future_is_not_zero, dim=-1).to(
            ego_future.device)  # (B, future_len)
        # (B, future_len)
        ego_future_off_p_mask = ego_future_not_zero_num == 0  # (B , future_len)
        ego_future_on_p_mask = ~ego_future_off_p_mask  # (B, future_len)
        ego_future_off_mask = torch.sum(ego_future_on_p_mask,
                                        dim=-1) == 0  # (B)
        return ego_future_off_p_mask, ego_future_off_mask

    def _filter_on_agents_past_cur(
        self,
        agents_past_current: torch.
        Tensor,  # (B, agents_num, time_len, 12 + 2K + 1 )
        agents_past_cur_on_p_mask: torch.
        Tensor,  # (B, agents_num, time_len, 1) # float
        agents_past_cur_on_mask: torch.Tensor  # (B * agents_num)
    ) -> torch.Tensor:  # (agents_past_cur_on_num, time_len, 12 + 2K + 1 "+1")
        B, agents_num, time_len, _ = agents_past_current.shape
        # agents_past_current: (B , agents_num, time_len, 12 + 2K + 1 "+1")
        agents_past_current = torch.cat(
            [agents_past_current, agents_past_cur_on_p_mask], dim=-1)
        # agents_past_current: (B * agents_num, time_len, 12 + 2K + 1 "+1")
        agents_past_current = agents_past_current.reshape(
            B * agents_num, time_len, -1)
        """
        agents_past_cur_on_mask 에서, True의 개수 = 
        (B * agents_num) 개 중에서 agents_past_cur_on_num 개
        """
        # on_agents_past_cur: (agents_past_cur_on_num, time_len, 12 + 2K + 1 "+1")
        on_agents_past_cur = agents_past_current[agents_past_cur_on_mask]
        assert on_agents_past_cur.shape[
                   -1] == 12 + 2 * self.num_fourier_frequencies + 1 + 1, \
            f"on_agents_past_cur shape mismatch: {on_agents_past_cur.shape}"
        return on_agents_past_cur

    def _filter_on_ego_future(
        self,
        ego_future: torch.Tensor,  # (B, future_len,  12 + 2K + 1)
        ego_future_on_p_mask: torch.Tensor,  # (B, future_len, 1)
        ego_future_on_mask: torch.Tensor  # (B)
    ) -> torch.Tensor:
        ego_future = torch.cat(
            [ego_future, ego_future_on_p_mask.to(ego_future.dtype)],
            dim=-1)  # (B, future_len,  12 + 2K + 1 + "1")
        # (ego_future_on_num, future_len, 10 + 2k)
        on_ego_future = ego_future[ego_future_on_mask]
        assert on_ego_future.shape[
                   -1] == 12 + 2 * self.num_fourier_frequencies + 1 + 1, \
            f"on_ego_future shape mismatch: {on_ego_future.shape}"
        return on_ego_future  # (ego_future_on_num, future_len, 14 + 2k)

    def _concat_past_cur_and_future(
        self,
        on_agents_past_cur: torch.
        Tensor,  # (agents_past_cur_on_num, time_len, 14 + 2k)
        on_ego_future: torch.Tensor
        # (ego_future_on_num, future_len, 14 + 2k)
    ) -> torch.Tensor:  # (on_all_time_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len, 14 + 2k)
        # (agents_past_cur_on_num * time_len, 10+2k)
        on_agents_past_cur = on_agents_past_cur.reshape(
            -1, on_agents_past_cur.shape[-1])
        # (ego_future_on_num * future_len, 14 + 2k)
        on_ego_future = on_ego_future.reshape(-1, on_ego_future.shape[-1])
        # (on_all_time_num = agents_past_cur_on_num * time_len + on_ego_future_on_num * future_len, 14 + 2k)
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
        idx = torch.arange(0, chunk_num + 1, device=device,
                           dtype=torch.long)  # (M+1,)
        boundaries = (idx * seq_len + chunk_num - 1) // chunk_num  # (M+1,)
        starts = boundaries[:-1]  # (chunk_num,)
        ends = boundaries[1:] - 1  # (chunk_num,)
        assert starts.shape == ends.shape == (chunk_num,)  # (chunk_num,)
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
                                    float("-inf"))  # 유효 아님 → -inf
        if chunk_is_off.any().item():
            # 전부 -inf이면 softmax NaN → 해당 행은 0으로 세팅해 NaN 회피(균등분포가 됨)
            logits = logits.clone()
            logits[chunk_is_off] = 0.0

        # 4) 소프트맥스 & 전부 마스크된 행은 0으로 고정(gradient도 0)
        attn = F.softmax(logits, dim=1).to(chunk_values.dtype)  # (N,L,Q) fp32
        attn = attn.masked_fill(chunk_is_off.reshape(-1, 1, 1), 0.0)
        # if chunk_is_off.any():
        #     attn[chunk_is_off] = 0.0
        ###############

        # ----- 4) 값 변환 및 가중합 -----
        # value_linear: (C→C)
        # einsum으로 Σ_t a_{t,q} * v_t  → (N, tokens_mlp_dim, C)
        # attn: (N, L, tokens_mlp_dim), values_proj: (N, L, C)
        values_proj = self.value_linear(chunk_values).float()  # (N, L, C)
        attn_f = attn.float()  # (N, L, Q)
        chunk_token = torch.einsum("nlq,nlc->nqc", attn_f,
                                   values_proj).to(chunk_values.dtype)

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
        assert on_trajs_off_p_mask.dtype == torch.bool
        on_agents_num, _, channel_mlp_dim = on_trajs.shape
        chunks_num = chunk_starts.numel()

        chunks = on_trajs.new_zeros(
            on_agents_num, chunks_num, self.tokens_mlp_dim,
            channel_mlp_dim)  # (N, chunk_num, tokens_mlp_dim, channel_mlp_dim)
        invalid_chunk_mask = torch.zeros(
            on_agents_num, chunks_num, dtype=torch.bool,
            device=on_trajs.device)  # (N, chunk_num)

        for chunk_idx in range(chunks_num):
            start, end = int(chunk_starts[chunk_idx].item()), int(
                chunk_ends[chunk_idx].item())
            # chunk_values: (N, L, channel_mlp_dim)
            chunk_values = on_trajs[:, start:end + 1, :]
            # chunk_off_points_mask: (N, L)
            chunk_off_points_mask = on_trajs_off_p_mask[:, start:end + 1]
            # chunk_is_off: (N,)
            chunk_on_points_mask = ~chunk_off_points_mask
            chunk_is_off = (chunk_on_points_mask).sum(dim=1) == 0
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
        agents_type = agents_type.reshape(B * agents_num,
                                          -1)  # (B * agents_num, 3)
        # (agents_past_cur_on_num, 3)
        agents_type = agents_type[agents_past_cur_on_mask]
        agents_past_cur_on_num = agents_type.shape[0]
        # (agents_past_cur_on_num, 3) -> (agents_past_cur_on_num, 1, 3)
        # -> (agents_past_cur_on_num, past_cur_chunk_num, 3)
        agents_type = agents_type.unsqueeze(1).expand(-1, past_cur_chunk_num,
                                                      -1)
        # (agents_past_cur_on_num, past_cur_chunk_num, 3)
        # -> (agents_past_cur_on_num * past_cur_chunk_num, 3)
        agents_type = agents_type.reshape(
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
        ego_fut_type = ego_fut_type.reshape(
            ego_future_on_num * future_chunk_num, -1)
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
    ) -> torch.Tensor:  # (on_all_on_chunk_num, channels_mlp_dim)
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

        agents_ego_fut_type = agents_ego_fut_type.to(
            dtype=self.type_emb.weight.dtype)
        # (on_all_on_chunk_num, 3) → (on_all_on_chunk_num, channels_mlp_dim)
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
        dtype = on_all_on_chunk.dtype
        device = on_all_on_chunk.device
        ######## PAST/CURRENT PART ########
        # (on_past_cur_on_chunk_num, hidden_dim)
        on_agents_past_cur_on_chunk = on_all_on_chunk[:
                                                      on_past_cur_on_chunk_num, :]
        on_agents_past_cur_chunk = torch.zeros(
            (agents_past_cur_on_num * past_cur_chunk_num, self._hidden_dim),
            dtype=dtype,
            device=device)
        # mask_agent: (on_agents_past_cur_on_num * past_cur_chunk_num, hidden_dim)
        mask_agent = on_agents_past_cur_on_chunk_mask.unsqueeze(-1).expand(
            -1, self._hidden_dim)
        assert mask_agent.sum().item() == on_agents_past_cur_on_chunk.numel()
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.masked_scatter(
            mask_agent, on_agents_past_cur_on_chunk)
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.reshape(
            agents_past_cur_on_num, past_cur_chunk_num, self._hidden_dim)
        ######## FUTURE PART ########
        # (on_ego_fut_on_chunk_num, hidden_dim)
        on_ego_fut_on_chunk = on_all_on_chunk[on_past_cur_on_chunk_num:, :]
        on_ego_fut_chunk = torch.zeros(
            (ego_future_on_num * future_chunk_num, self._hidden_dim),
            dtype=dtype,
            device=device)
        mask_ego = on_ego_fut_on_chunk_mask.unsqueeze(-1).expand(
            -1, self._hidden_dim)
        assert mask_ego.sum().item() == on_ego_fut_on_chunk.numel()
        on_ego_fut_chunk = on_ego_fut_chunk.masked_scatter(
            mask_ego, on_ego_fut_on_chunk)
        # (ego_future_on_num, future_chunk_num, hidden_dim)
        on_ego_fut_chunk = on_ego_fut_chunk.reshape(ego_future_on_num,
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
        assert on_agents_past_cur_off_chunk_mask.shape[0] == int(
            agents_past_cur_on_mask.sum().item())
        assert on_ego_fut_off_chunk_mask.shape[0] == int(
            ego_future_on_mask.sum().item())

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
        agents_past_cur_off_chunk_mask = agents_past_cur_off_chunk_mask.reshape(
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

    def _get_agents_past_cur_chunks_type(
            self, agents_past_cur_chunks_xyyaw: torch.Tensor) -> torch.Tensor:
        # (B, agents_num, past_cur_chunk_num, 4 )
        B, agents_num, past_cur_chunk_num = agents_past_cur_chunks_xyyaw.shape[:
                                                                               3]
        agents_past_cur_chunks_type = torch.zeros(
            (B, agents_num, past_cur_chunk_num, 4),
            device=agents_past_cur_chunks_xyyaw.device,
            dtype=agents_past_cur_chunks_xyyaw.dtype,
        )
        agents_past_cur_chunks_type[:, 0, :, 0] = 1.0  # ego
        agents_past_cur_chunks_type[:, 1:, :, 1] = 1.0  # neighbor
        return agents_past_cur_chunks_type  # (B, agents_num, past_cur_chunk_num, 4)

    def _get_agents_past_cur_chunks_xyyaw(
        self,
        past_chunk_start_idx: torch.Tensor,  # (past_cur_chunk_num,)
        past_chunk_end_idx: torch.Tensor,  # (past_cur_chunk_num,)
        agents_past_cur_off_p_mask: torch.Tensor,
        # (B, agents_num, time_len)
        agents_past_cur_xyyaw: torch.Tensor  # (B, agents_num, time_len, 4)
    ) -> torch.Tensor:  # (B, agents_num, past_cur_chunk_num, 4)
        B, agents_num, time_len, feat_dim = agents_past_cur_xyyaw.shape
        past_cur_chunk_num: int = past_chunk_start_idx.numel()
        device = agents_past_cur_xyyaw.device
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
            t = torch.arange(L, device=device, dtype=dtype_idx).reshape(
                1, 1, L).expand(B, agents_num, L)

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
        # centers: (B, agents_num, M, 4 +)
        agents_past_cur_chunks_xyyaw = torch.gather(agents_past_cur_xyyaw,
                                                    dim=2,
                                                    index=gather_idx)

        return agents_past_cur_chunks_xyyaw  # (B, agents_num, past_cur_chunk_num, 4)

    def _get_agents_past_cur_chunks_feature(
        self,
        past_chunk_start_idx: torch.Tensor,  # (past_cur_chunk_num,)
        past_chunk_end_idx: torch.Tensor,  # (past_cur_chunk_num,)
        agents_past_cur_off_p_mask: torch.Tensor,
        # (B, agents_num, time_len)
        agents_past_cur_xyyaw: torch.Tensor  # (B, agents_num, time_len, 4)
    ) -> torch.Tensor:  # (B, agents_num * past_cur_chunk_num, 8)
        B, agents_num, time_len, _ = agents_past_cur_xyyaw.shape
        past_cur_chunk_num = past_chunk_start_idx.shape[0]
        ##########
        # agents_past_cur_chunks_xyyaw: (B, agents_num, past_cur_chunk_num, 4)
        agents_past_cur_chunks_xyyaw = self._get_agents_past_cur_chunks_xyyaw(
            past_chunk_start_idx, past_chunk_end_idx,
            agents_past_cur_off_p_mask, agents_past_cur_xyyaw)

        # (B, agents_num, past_cur_chunk_num, 4)
        agents_past_cur_chunks_type = self._get_agents_past_cur_chunks_type(
            agents_past_cur_chunks_xyyaw)
        # (B, agents_num, past_cur_chunk_num, 4 + 4)
        agents_past_cur_chunks_feature = torch.cat(
            [agents_past_cur_chunks_xyyaw, agents_past_cur_chunks_type], dim=-1)
        # (B, agents_num * past_cur_chunk_num, 4 + 4)
        agents_past_cur_chunks_feature = agents_past_cur_chunks_feature.reshape(
            B, agents_num * past_cur_chunk_num, -1)
        return agents_past_cur_chunks_feature

    def _get_ego_fut_chunks_pose(
        self,
        fut_chunk_start_idx: torch.Tensor,  # (future_chunk_num,)
        fut_chunk_end_idx: torch.Tensor,  # (future_chunk_num,)
        ego_future_off_p_mask: torch.Tensor,  # (B, future_len)
        ego_future_xyyaw: torch.Tensor  # (B, future_len, 4)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4)
        B, future_len, feat_dim = ego_future_xyyaw.shape
        future_chunk_num: int = fut_chunk_start_idx.numel()
        device = ego_future_xyyaw.device
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
                             dtype=dtype_idx).reshape(1, L).expand(B, L)

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
        # (B, future_chunk_num, 4)
        ego_fut_chunks_xyyaw = torch.gather(ego_future_xyyaw,
                                            dim=1,
                                            index=gather_idx)

        return ego_fut_chunks_xyyaw

    def _get_ego_fut_chunks_type(
        self, ego_future_chunks_feature: torch.Tensor
        # (B, future_chunk_num, 4)
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
            ego_future_xyyaw: torch.Tensor,  # (B, future_len, 4)
    ) -> torch.Tensor:  # (B, future_chunk_num, 4 +  4)
        B, future_len, _ = ego_future_xyyaw.shape
        # (B, future_chunk_num, 4)
        ego_fut_chunks_xyyaw = self._get_ego_fut_chunks_pose(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw)
        # (B, future_chunk_num, 4)
        ego_fut_chunks_type = self._get_ego_fut_chunks_type(
            ego_fut_chunks_xyyaw)
        # (B, future_chunk_num, 4 + 4)
        ego_future_chunks_feature = torch.cat(
            [ego_fut_chunks_xyyaw, ego_fut_chunks_type], dim=-1)
        return ego_future_chunks_feature

    def _add_timestep_to_agents_past_cur(
            self, agents_past_current: torch.Tensor) -> torch.Tensor:
        # agents_past_current: (B, agents_num, time_len, 12)
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
            dim=-1)  # (B, agents_num, time_len, 12 + 2K + 1)
        # (B, agents_num, time_len, 12 + 2K + 1)
        assert agents_past_current.shape[
            -1] == 12 + 2 * self.num_fourier_frequencies + 1
        return agents_past_current

    def _add_timestep_to_ego_fut(self,
                                 ego_future: torch.Tensor) -> torch.Tensor:
        # ego_future: (B, future_len, 12)
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
                               dim=-1)  # (B, future_len, 12 + 2K + 1)
        assert ego_future.shape[-1] == 12 + 2 * self.num_fourier_frequencies + 1
        return ego_future

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
        assert on_agents_past_cur_off_chunk_mask.dtype == torch.bool
        agents_past_cur_on_num, past_cur_chunk_num = on_agents_past_cur_chunk.shape[:
                                                                                    2]
        on_agents_past_cur_chunk = on_agents_past_cur_chunk.reshape(
            agents_past_cur_on_num * past_cur_chunk_num, self.tokens_mlp_dim,
            -1)
        on_agents_past_cur_on_chunk_mask = ~on_agents_past_cur_off_chunk_mask.reshape(
            -1)
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
        assert on_ego_fut_off_chunk_mask.dtype == torch.bool
        ego_future_on_num, future_chunk_num = on_ego_fut_chunk.shape[:2]
        on_ego_fut_chunk = on_ego_fut_chunk.reshape(
            ego_future_on_num * future_chunk_num, self.tokens_mlp_dim, -1)
        on_ego_fut_on_chunk_mask = ~on_ego_fut_off_chunk_mask.reshape(-1)
        on_ego_fut_on_chunk = on_ego_fut_chunk[on_ego_fut_on_chunk_mask]
        return on_ego_fut_on_chunk, on_ego_fut_on_chunk_mask

    def _fill_on_agent_to_agent(
            self,
            on_agents_past_cur_chunk: torch.Tensor,
            # (agents_past_cur_on_num, past_cur_chunk_num, hidden_dim)
            on_ego_fut_chunk: torch.Tensor,
            # (ego_future_on_num, future_chunk_num, hidden_dim)
            agents_past_cur_on_mask: torch.Tensor,  # (B * agents_num), True=유효
            ego_future_on_mask: torch.Tensor,  # (B), True=유효
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            # agents_past_cur_chunk: (B, agents_num * past_cur_chunk_num , hidden_dim)
            # ego_fut_chunk: (B, future_chunk_num, hidden_dim)
        """
        past_cur_chunk_num = on_agents_past_cur_chunk.shape[1]
        future_chunk_num = on_ego_fut_chunk.shape[1]
        B_agents_num = agents_past_cur_on_mask.shape[0]  # B * agents_num
        B = ego_future_on_mask.shape[0]  # 배치 크기
        agents_num = B_agents_num // B  # agents_num = (B*agents_num)/B
        dtype = on_agents_past_cur_chunk.dtype
        device = on_agents_past_cur_chunk.device
        H = self._hidden_dim

        # 방어적 체크(디버깅 내구성 ↑)
        assert agents_past_cur_on_mask.dtype == torch.bool
        assert ego_future_on_mask.dtype == torch.bool
        assert int(agents_past_cur_on_mask.sum().item()) == \
               on_agents_past_cur_chunk.shape[0], \
            "agents_past_cur_on_mask True 개수와 on_agents_past_cur_chunk 행 수가 다릅니다."
        assert int(ego_future_on_mask.sum().item()) == on_ego_fut_chunk.shape[
            0], \
            "ego_future_on_mask True 개수와 on_ego_fut_chunk 행 수가 다릅니다."

        # ---------- AGENTS (past/current) ----------
        # 타깃 버퍼 (B*A, past_cur_chunk_num, H)
        agents_past_cur_chunk = torch.zeros(
            (B_agents_num, past_cur_chunk_num, H), dtype=dtype, device=device)
        # 마스크: 유효 에이전트 행들 전체를 채움
        # (B * agents_num) -> (B * agents_num, 1, 1) -> (B * agents_num, past_cur_chunk_num, H)
        mask_agent = agents_past_cur_on_mask.reshape(-1, 1, 1).expand(
            -1, past_cur_chunk_num, H)
        # 소스를 유효 위치에 순서대로 채움 (grad 안전)
        agents_past_cur_chunk = agents_past_cur_chunk.masked_scatter(
            mask_agent, on_agents_past_cur_chunk)
        # (B, A*past_cur_chunk_num, H)로 복원
        agents_past_cur_chunk = agents_past_cur_chunk.reshape(
            B, agents_num * past_cur_chunk_num, H)

        # ---------- EGO (future) ----------
        # 타깃 버퍼 (B, future_chunk_num, H)
        ego_fut_chunk = torch.zeros((B, future_chunk_num, H),
                                    dtype=dtype,
                                    device=device)
        # 마스크 (B, future_chunk_num, H): 유효 배치의 모든 future chunk 채움
        # (B) -> (B, 1, 1) -> (B, future_chunk_num, H)
        mask_ego = ego_future_on_mask.reshape(-1, 1,
                                              1).expand(-1, future_chunk_num, H)
        # 소스를 유효 위치에 순서대로 채움 (grad 안전)
        ego_fut_chunk = ego_fut_chunk.masked_scatter(mask_ego, on_ego_fut_chunk)

        # (B, agents_num * past_cur_chunk_num , H), (B, future_chunk_num, H)
        return agents_past_cur_chunk, ego_fut_chunk

    def _add_relation_with_agents_current(
            self,
            agents_past_current: torch.Tensor,  # (B, A, T, 8) or (B, T, 8)
            agents_current_xyyaw: torch.Tensor,  # (B, A, 4) or (B, 4)
    ) -> torch.Tensor:
        """기준 자세(마지막 프레임) 대비 [Δx, Δy, cosΔyaw, sinΔyaw] 4채널을 추가해 8→12 채널로 확장.

        허용 입력 조합:
            1) agents_past_current: (B, A, T, 8), agents_current_xyyaw: (B, A, 4)
            2) agents_past_current: (B, T, 8),    agents_current_xyyaw: (B, 4)

        채널 정의:
            Δx = x_t - x_0
            Δy = y_t - y_0
            cosΔyaw = cos_t * cos_0 + sin_t * sin_0
            sinΔyaw = sin_t * cos_0 - cos_t * sin_0

        반환:
            입력과 동일한 leading shape, 마지막 채널이 12인 텐서
            (B, A, T, 12) 또는 (B, T, 12)
        """
        assert agents_past_current.shape[-1] == 8, \
            f"expected last dim=8, got {agents_past_current.shape[-1]}"

        dtype = agents_past_current.dtype
        device = agents_past_current.device

        if agents_past_current.dim() == 4:
            # (B, A, T, 8)  +  (B, A, 4)
            B, A, T, _ = agents_past_current.shape
            if not (agents_current_xyyaw.dim() == 3 and
                    agents_current_xyyaw.shape[0] == B and
                    agents_current_xyyaw.shape[1] == A and
                    agents_current_xyyaw.shape[2] == 4):
                raise ValueError(
                    f"Expected agents_current_xyyaw of shape (B, A, 4); got {tuple(agents_current_xyyaw.shape)}"
                )

            agent_past_current_xyyaw = agents_past_current[..., :
                                                           4]  # (B, A, T, 4)
            agents_current_xyyaw = agents_current_xyyaw.to(
                dtype=dtype, device=device)  # (B, A, 4)
            agents_current_xyyaw = agents_current_xyyaw.unsqueeze(
                2)  # (B, A, 1, 4) → T로 브로드캐스트

            # Δx, Δy
            delta_xy = agent_past_current_xyyaw[..., :2] - agents_current_xyyaw[
                ..., :2]  # (B, A, T, 2)

            # cosΔyaw, sinΔyaw
            cos_t, sin_t = agent_past_current_xyyaw[
                ..., 2], agent_past_current_xyyaw[..., 3]  # (B, A, T)
            cos_0, sin_0 = agents_current_xyyaw[..., 2], agents_current_xyyaw[
                ..., 3]  # (B, A, 1)
            cos_d = cos_t * cos_0 + sin_t * sin_0  # (B, A, T)
            sin_d = sin_t * cos_0 - cos_t * sin_0  # (B, A, T)
            delta_dir = torch.stack((cos_d, sin_d), dim=-1)  # (B, A, T, 2)

            delta_feat = torch.cat((delta_xy, delta_dir),
                                   dim=-1)  # (B, A, T, 4)
            return_ = torch.cat((agents_past_current, delta_feat),
                                dim=-1)  # (B, A, T, 12)
            assert return_.shape[-1] == 12, \
                f"Expected last dim=12, got {return_.shape[-1]}"
            return return_

        elif agents_past_current.dim() == 3:
            # (B, T, 8)  +  (B, 4)
            B, T, _ = agents_past_current.shape
            if not (agents_current_xyyaw.dim() == 2 and
                    agents_current_xyyaw.shape[0] == B and
                    agents_current_xyyaw.shape[1] == 4):
                raise ValueError(
                    f"Expected agents_current_xyyaw of shape (B, 4); got {tuple(agents_current_xyyaw.shape)}"
                )

            agent_past_current_xyyaw = agents_past_current[..., :4]  # (B, T, 4)
            agents_current_xyyaw = agents_current_xyyaw.to(
                dtype=dtype, device=device)  # (B, 4)
            agents_current_xyyaw = agents_current_xyyaw.unsqueeze(
                1)  # (B, 1, 4) → T로 브로드캐스트

            # Δx, Δy
            delta_xy = agent_past_current_xyyaw[..., :2] - agents_current_xyyaw[
                ..., :2]  # (B, T, 2)

            # cosΔyaw, sinΔyaw
            cos_t, sin_t = agent_past_current_xyyaw[
                ..., 2], agent_past_current_xyyaw[..., 3]
            cos_0, sin_0 = agents_current_xyyaw[...,
                                                2], agents_current_xyyaw[..., 3]
            cos_d = cos_t * cos_0 + sin_t * sin_0  # (B, T)
            sin_d = sin_t * cos_0 - cos_t * sin_0  # (B, T)
            delta_dir = torch.stack((cos_d, sin_d), dim=-1)  # (B, T, 2)

            delta_feat = torch.cat((delta_xy, delta_dir), dim=-1)  # (B, T, 4)
            return_ = torch.cat((agents_past_current, delta_feat),
                                dim=-1)  # (B, T, 12)
            assert return_.shape[-1] == 12, \
                f"Expected last dim=12, got {return_.shape[-1]}"
            return return_

        else:
            raise ValueError(
                "agents_past_current must be (B, A, T, 8) or (B, T, 8).")

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

        agents_past_cur_xyyaw = agents_past_current[:, :, :, :4].clone(
        )  # (B, agents_num, time_len, 4)

        ### add relation with agents_current
        agents_current_xyyaw = agents_past_current[:, :, -1, :4].clone(
        )  # shape (B, agents_num, 4)
        # agents_past_current: (B, agents_num, time_len, 8) -> (B, agents_num, time_len, 8 + 4)
        agents_past_current = self._add_relation_with_agents_current(
            agents_past_current, agents_current_xyyaw)
        ### add timestep
        # agents_past_current: (B, agents_num, time_len, 12 "+ 2K + 1")
        (agents_past_current
        ) = self._add_timestep_to_agents_past_cur(agents_past_current)

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
        agents_past_cur_on_p_mask = agents_past_cur_on_p_mask.to(
            agents_past_current.dtype)

        # on_agents_past_cur: (agents_past_cur_on_num, time_len, 12 + 2K + 1 "+ 1" )
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

        # ego_future: (B, future_len, 8) -> (B, future_len, 8 + 4)
        ego_current_xyyaw = agents_current_xyyaw[:, 0, :].clone()  # (B, 4)
        ego_future = self._add_relation_with_agents_current(
            ego_future, ego_current_xyyaw)
        ### add timestep
        # ego_future: (B, future_len, 12 "+ 2K + 1")
        ego_future = self._add_timestep_to_ego_fut(ego_future)
        ############
        """
        ego_future_off_p_mask: (B, future_len)
        ego_future_off_mask: (B)
        --------
        ego_future_on_p_mask: (B, future_len, 1)
        ego_future_on_mask: (B)
        """
        ego_future_off_p_mask, ego_future_off_mask = self._get_ego_future_mask(
            ego_future)
        ego_future_on_p_mask = (~ego_future_off_p_mask).float().unsqueeze(
            -1)  # (B, future_len, 1)
        ego_future_on_mask = ~ego_future_off_mask  # (B)
        ###########
        # on_ego_future:
        # (ego_future_on_num, future_len, 12 + 2K + 1 "+ 1")
        on_ego_future = self._filter_on_ego_future(ego_future,
                                                   ego_future_on_p_mask,
                                                   ego_future_on_mask)
        # on_all_time_num = agents_past_cur_on_num * time_len + ego_future_on_num * future_len
        agents_past_cur_on_num = on_agents_past_cur.shape[0]
        on_agents_past_cur_points_num = agents_past_cur_on_num * time_len
        ego_future_on_num = on_ego_future.shape[0]

        # on_all (on_all_time_num, 14 + 2k)
        # on_all_time_num =
        # agents_past_cur_on_num * time_len + on_ego_future_on_num * future_len
        on_all = self._concat_past_cur_and_future(on_agents_past_cur,
                                                  on_ego_future)

        # on_all: (on_all_time_num, channels_mlp_dim)
        on_all = self.channel_pre_project(on_all)
        # on_agents_past_cur: (agents_past_cur_on_num, time_len, channels_mlp_dim)
        on_agents_past_cur = on_all[:on_agents_past_cur_points_num, :].reshape(
            agents_past_cur_on_num, time_len, -1)
        # on_ego_future: (ego_future_on_num, future_len, channels_mlp_dim)
        on_ego_future = on_all[on_agents_past_cur_points_num:, :].reshape(
            ego_future_on_num, future_len, -1)
        """
        token_pre_project = hard split + gated attentional pooling
        """
        # ---------- 8) 균등 분할 경계 ----------

        past_chunk_start_idx, past_chunk_end_idx = self._compute_equal_chunks(
            seq_len=time_len, chunk_num=self.past_cur_chunk_num,
            device=device)  # (past_cur_chunk_num,), (past_cur_chunk_num,)
        #################

        # agents_past_cur_xyyaw: (B, agents_num, time_len, 4)
        # (B, agents_num * past_cur_chunk_num, 8)
        agents_past_cur_chunks_feature = self._get_agents_past_cur_chunks_feature(
            past_chunk_start_idx, past_chunk_end_idx,
            agents_past_cur_off_p_mask, agents_past_cur_xyyaw)

        #################
        fut_chunk_start_idx, fut_chunk_end_idx = self._compute_equal_chunks(
            seq_len=future_len, chunk_num=self.future_chunk_num,
            device=device)  # (future_chunk_num,), (future_chunk_num,)
        #################
        # ego_future_off_p_mask: (B, future_len)

        # ego_future_xyyaw: (B, future_len, 4)
        ego_future_xyyaw = ego_future[:, :, :4].clone()  # (B, future_len, 4)

        # ego_future_chunks_feature: (B, future_chunk_num, 4 + 4)
        ego_future_chunks_feature = self._get_ego_future_chunks_feature(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw)
        #################
        all_chunk_pos_feature = torch.cat(
            [agents_past_cur_chunks_feature, ego_future_chunks_feature], dim=1
        )  # (B, agents_num * past_cur_chunk_num + future_chunk_num, 4 + 4)
        #################
        """
        # agents_past_cur_off_p_mask: (B, agents_num, time_len)
        # agents_past_cur_on_mask: (B * agents_num)

        # on_agents_past_cur_off_p_mask: (agents_past_cur_on_num, time_len)
        """
        agents_past_cur_off_p_mask = agents_past_cur_off_p_mask.reshape(
            B * agents_num, time_len)  # (B * agents_num, time_len)
        on_agents_past_cur_off_p_mask = agents_past_cur_off_p_mask[
            agents_past_cur_on_mask]  # (agents_past_cur_on_num, time_len)

        # on_agents_past_cur_chunk:
        #   (agents_past_cur_on_num, past_cur_chunk_num, tokens_mlp_dim, channels_mlp_dim)
        # on_agents_past_cur_off_chunk_mask:
        #   (agents_past_cur_on_num, past_cur_chunk_num)

        (on_agents_past_cur_chunk, on_agents_past_cur_off_chunk_mask
        ) = self.traj_to_chunk_token(
            on_agents_past_cur,
            # (agents_past_cur_on_num, time_len, channels_mlp_dim)
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
        on_all_on_chunk = on_all_on_chunk.float().mean(dim=1).to(
            on_all_on_chunk.dtype)
        # agents_ego_fut_type_emb: (on_all_on_chunk_num, channels_mlp_dim)
        agents_ego_fut_type_emb = self._get_type_embedding(
            agents_type, ego_fut_type, agents_past_cur_on_mask,
            ego_future_on_mask, on_agents_past_cur_on_chunk_mask,
            on_ego_fut_on_chunk_mask, self.past_cur_chunk_num,
            self.future_chunk_num)

        # on_all_on_chunk: (on_all_on_chunk_num, channels_mlp_dim)
        on_all_on_chunk = on_all_on_chunk + (self.type_scale *
                                             agents_ego_fut_type_emb)
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

        ego_fut_global = self._get_ego_fut_global(
            ego_fut_chunk,  # (B, future_chunk_num, hidden_dim)
            ego_future_on_mask,  # # (B,) True=유효
            on_ego_fut_off_chunk_mask
        )  # (ego_future_on_num, future_chunk_num) True=무효

        # all_chunk: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
        # all_off_chunk_mask: (B, agents_num * past_cur_chunk_num + future_chunk_num)
        # all_chunk_pos_feature : (B, agents_num * past_cur_chunk_num + future_chunk_num, 4  + 4)
        # ego_fut_global: (B, hidden_dim)
        return all_chunk, all_off_chunk_mask, all_chunk_pos_feature, ego_fut_global

    @staticmethod
    def _masked_mean(ego_fut_chunk: torch.Tensor,
                     ego_fut_off_chunk_mask_full: torch.Tensor,
                     dim: int) -> torch.Tensor:
        """
        ego_fut_chunk:    (B, future_chunk_num, H) #
        ego_fut_off_chunk_mask_full: (B, future_chunk_num)  # True=무효 # (B, future_chunk_num)
        return: (B, H) # (B, hidden_dim)
        """
        # 고정 전제
        assert dim == 1, "dim은 1이어야 합니다."
        assert ego_fut_chunk.dim() == 3 and ego_fut_off_chunk_mask_full.dim(
        ) == 2, "ego_fut_chunk:(B,future_chunk_num,H), ego_fut_off_chunk_mask_full:(B,future_chunk_num) 필요"
        assert ego_fut_off_chunk_mask_full.dtype == torch.bool, "mask는 bool이어야 합니다."

        B, future_chunk_num, H = ego_fut_chunk.shape
        assert ego_fut_off_chunk_mask_full.shape == (B, future_chunk_num)

        ego_fut_on_chunk_mask_full = ~ego_fut_off_chunk_mask_full  # (B, future_chunk_num)
        on_chunk_num_per_batch = ego_fut_on_chunk_mask_full.sum(
            dim=dim, keepdim=True)  # (B, 1)
        on_chunk_num_per_batch = on_chunk_num_per_batch.clamp(
            min=1)  # (B, 1)  # 0 분모 방지

        ego_fut_on_chunk_mask_full = ego_fut_on_chunk_mask_full.unsqueeze(
            -1)  # (B, future_chunk_num, 1)

        ego_fut_on_chunk_sum = (ego_fut_chunk.float() *
                                ego_fut_on_chunk_mask_full.float()).sum(dim=dim)
        ego_fut_on_chunk_mean = (ego_fut_on_chunk_sum /
                                 on_chunk_num_per_batch.to(torch.float32)).to(
                                     ego_fut_chunk.dtype)

        assert ego_fut_on_chunk_mean.shape == (B, H)

        return ego_fut_on_chunk_mean

    def _get_ego_fut_global(
        self,
        ego_fut_chunk: torch.Tensor,  # (B, future_chunk_num, hidden_dim)
        ego_future_on_mask: torch.Tensor,  # (B) True=유효
        on_ego_fut_off_chunk_mask: torch.Tensor,
        # (ego_future_on_num, future_chunk_num) True=무효
    ) -> torch.Tensor:  # (B, hidden_dim)
        # --- (A) 배치 크기로 확장된 미래 chunk 마스크: (B, future_chunk_num) True=무효 ---
        B, future_chunk_num, hidden_dim = ego_fut_chunk.shape
        ego_fut_off_chunk_mask_full = torch.ones((B, future_chunk_num),
                                                 dtype=torch.bool,
                                                 device=ego_fut_chunk.device)
        ego_fut_off_chunk_mask_full[
            ego_future_on_mask] = on_ego_fut_off_chunk_mask  # 주입

        # --- (B) 프리노름 + 패딩을 0으로 고정(gradient 차단) ---
        ego_fut_chunk = self.ego_fut_in_norm(ego_fut_chunk)
        ego_fut_chunk = ego_fut_chunk.masked_fill(
            ego_fut_off_chunk_mask_full.unsqueeze(-1),
            0.0)  # (B, future_chunk_num, hidden_dim)

        # --- (C) PMA‑lite: 학습 시드들이 chunk들을 요약 ---
        # ego_fut_seeds: (1, ego_fut_num_seeds, hidden_dim) -> (B, ego_fut_num_seeds, hidden_dim)
        seeds = self.ego_fut_seeds.to(ego_fut_chunk.dtype).expand(
            B, self.ego_fut_num_seeds,
            hidden_dim)  # (B, ego_fut_num_seeds, hidden_dim)

        # 점수: (B, ego_fut_num_seeds, future_chunk_num) = (B, ego_fut_num_seeds, D) @ (B, future_chunk_num, D)^T / sqrt(D)
        logits = torch.einsum("bsd,bmd->bsm", seeds, ego_fut_chunk) / math.sqrt(
            max(1.0, float(hidden_dim)))
        # pad(True) → -inf
        if ego_fut_off_chunk_mask_full.any().item():
            logits = logits.masked_fill(
                ego_fut_off_chunk_mask_full.unsqueeze(1), float("-inf"))

        ego_fut_all_chunk_off = ego_fut_off_chunk_mask_full.all(dim=1)  # (B,)
        if ego_fut_all_chunk_off.any().item():
            logits = logits.clone()
            logits[ego_fut_all_chunk_off] = 0.0  # softmax NaN 방지
        # (B, ego_fut_num_seeds, future_chunk_num)
        attn = F.softmax(logits, dim=-1)
        if self.ego_fut_attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.ego_fut_attn_drop_p)

        # 시드별 요약: (B, ego_fut_num_seeds, future_chunk_num) @ (B, future_chunk_num, D) -> (B, ego_fut_num_seeds, D)
        y_seed = torch.einsum("bsm,bmd->bsd", attn, ego_fut_chunk)

        # 시드별 소형 FFN 잔차(게이트 0에서 시작)
        y_seed = y_seed + self.ego_fut_seed_ffn_alpha.to(
            y_seed.dtype) * self.ego_fut_seed_ffn(y_seed)

        # 여러 시드를 하나로: softmax 게이트 → (B, D)
        w = F.softmax(self.ego_fut_seed_gate(y_seed).squeeze(-1),
                      dim=1).unsqueeze(-1)  # (B,ego_fut_num_seeds,1)
        y_pool = (w * y_seed).sum(dim=1)  # (B, hidden_dim)

        # --- (D) DeepSets(mean/max) 잔차(게이트 0에서 시작) ---
        ego_fut_global_mean = self._masked_mean(ego_fut_chunk,
                                                ego_fut_off_chunk_mask_full,
                                                dim=1)
        ego_fut_global_max = self._masked_max_ego_chunks(
            ego_fut_chunk, ego_fut_off_chunk_mask_full, dim=1)

        ego_fut_global = y_pool \
                         + self.ego_fut_ds_mean_alpha.to(
            y_pool.dtype) * ego_fut_global_mean \
                         + self.ego_fut_ds_max_alpha.to(
            y_pool.dtype) * ego_fut_global_max  # (B, hidden_dim)

        # --- (E) all‑off 배치는 0 벡터 고정 ---
        if ego_fut_all_chunk_off.any().item():
            ego_fut_global = torch.where(ego_fut_all_chunk_off.unsqueeze(-1),
                                         torch.zeros_like(ego_fut_global),
                                         ego_fut_global)

        # --- (F) 정규화/드롭아웃 ---
        ego_fut_global = self.ego_fut_out_drop(
            self.ego_fut_out_norm(ego_fut_global))
        return ego_fut_global


class StaticFusionEncoder(nn.Module):

    def __init__(self,
                 dim,
                 drop_path_rate=0.3,
                 hidden_dim=192,
                 device="cuda",
                 num_fourier_frequencies=4,
                 time_gap=0.1,
                 time_min=-2.0,
                 time_max=8.0):
        super().__init__()
        self.time_gap = time_gap
        self.time_min = time_min
        self.time_max = time_max
        self._hidden_dim = hidden_dim
        self.projection = Mlp(in_features=dim,
                              hidden_features=hidden_dim,
                              out_features=hidden_dim,
                              act_layer=nn.GELU,
                              drop=drop_path_rate)

    def forward(self, static_info):
        """
        static_info: B, static_objects_num, D_10 (x, y, cos, sin, w, l, type(4))

        returns:
            static_encoding: (B, static_objects_num, hidden_dim)
            mask_p: (B, static_objects_num)
            static_feature: (B, static_objects_num, 4 + 4)
        """
        B, static_objects_num, _ = static_info.shape
        static_xyyaw = static_info[:, :, :4].clone(
        )  # (B, static_objects_num, 4)
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = self._get_static_feature(static_xyyaw)
        # [FIX] 오토캐스트 환경이면 버퍼 dtype을 현재 GPU autocast dtype으로
        out_dtype = (torch.get_autocast_gpu_dtype()
                     if torch.is_autocast_enabled() and static_info.is_cuda else
                     static_info.dtype)
        static_encoding = torch.zeros(
            (B * static_objects_num, self._hidden_dim),
            device=static_info.device,
            dtype=out_dtype)

        mask_p = torch.sum(torch.ne(static_info[..., :10], 0),
                           dim=-1).to(static_info.device) == 0

        valid_indices = ~mask_p.reshape(-1)

        if valid_indices.sum().item() > 0:
            static_info = static_info.reshape(B * static_objects_num, -1)
            static_info = static_info[valid_indices]
            static_info = self.projection(static_info)
            static_info = static_info.to(
                dtype=static_encoding.dtype)  # [FIX] 좌변 dtype 일치화
            static_encoding[valid_indices] = static_info
        else:
            # projection 파라미터를 0-스케일로 터치
            touch = static_encoding.new_zeros(())
            for p in self.projection.parameters():
                touch = touch + p.view(-1)[:1].sum()
            static_encoding = static_encoding + touch * 0.0
        static_encoding = static_encoding.reshape(
            B, static_objects_num, -1)  # (B, static_objects_num, hidden_dim)
        mask_p = mask_p.reshape(B,
                                static_objects_num)  # (B, static_objects_num)
        return static_encoding, mask_p, static_feature

    def _get_static_feature(
        self,
        static_xyyaw: torch.Tensor  # (B, static_objects_num, 4)
    ) -> torch.Tensor:
        B, static_objects_num, _ = static_xyyaw.shape
        # static_type: (B, static_objects_num, 4) # 4:
        static_type = torch.zeros(
            (B, static_objects_num, 4),
            device=static_xyyaw.device,
            dtype=static_xyyaw.dtype,
        )
        static_type[:, :, -2] = 1.0  # type
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = torch.cat([static_xyyaw, static_type], dim=-1)
        assert static_feature.shape == (B, static_objects_num, 8), \
            f"Expected static_feature shape (B, static_objects_num, 8), got {static_feature.shape}"
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
        nn.init.normal_(self.speed_limit_emb.weight, std=0.02)

        self.unknown_speed_emb = nn.Embedding(1, channels_mlp_dim)
        self.traffic_emb = nn.Linear(4, channels_mlp_dim)
        nn.init.normal_(self.traffic_emb.weight, std=0.02)

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

    def _get_lane_feature(
        self, lane_xyyaw: torch.Tensor
        # (B, lane_num, 4)
    ) -> torch.Tensor:  # (B, lane_num, 4 + 4)
        B, lane_num, _ = lane_xyyaw.shape
        # lane_type: (B, lane_num, 4)
        lane_type = torch.zeros(
            (B, lane_num, 4),
            device=lane_xyyaw.device,
            dtype=lane_xyyaw.dtype,
        )
        lane_type[:, :, -1] = 1.0  # type
        # static_feature: (B, lane_num, 4 + 4)
        lane_feature = torch.cat([lane_xyyaw, lane_type], dim=-1)
        return lane_feature

    def forward(self, lane_info, speed_limit, has_speed_limit):
        '''
        lane_info: B, lane_num, lane_len, D (x, y, x'-x, y'-y, x_left-x, y_left-y, x_right-x, y_right-y, traffic(4))
        speed_limit: B, lane_num, 1
        has_speed_limit: B, lane_num, 1

        returns:
            lane_embedding: (B, lane_num, hidden_dim)
            mask_p: (B, lane_num)
            lane_feature: (B, lane_num, 8)
        '''

        traffic = lane_info[:, :, 0, 8:]  # (B, lane_num, 4)
        lane_info = lane_info[..., :8]  # (B, lane_num, lane_len, 8)

        lane_pos = lane_info[:, :, int(self._lane_len /
                                       2), :4].clone()  # (B, lane_num, 4)
        # lane_feature: (B, lane_num, 4 +  4)
        lane_feature = self._get_lane_feature(lane_pos)

        B, lane_num, lane_len, _ = lane_info.shape
        mask_v = torch.sum(torch.ne(lane_info[..., :8], 0),
                           dim=-1).to(lane_info.device) == 0
        mask_p = torch.sum(~mask_v, dim=-1) == 0
        lane_info = lane_info.reshape(B * lane_num, lane_len, -1)

        valid_indices = ~mask_p.reshape(-1)

        num_valid = int(valid_indices.sum().item())

        if num_valid == 0:
            # 최종 임베딩 차원(H)
            H = self.emb_project.fc2.out_features  # timm.Mlp의 최종 out_features

            # ⚠️ autocast 켜진 경우, 다른 임베딩과 dtype을 맞춰줌 (fp16/bf16)
            out_dtype = (torch.get_autocast_gpu_dtype()
                         if torch.is_autocast_enabled() and lane_info.is_cuda
                         else lane_info.dtype)

            lane_embedding = torch.zeros((B, lane_num, H),
                                         device=lane_info.device,
                                         dtype=out_dtype)
            print(
                "[Warning] All lane inputs are padding. Returning zero embeddings."
            )
            # 파라미터 0-스케일 터치(DDP unused param 방지)
            touch = lane_embedding.new_zeros(())
            for mod in [
                    self.channel_pre_project, self.token_pre_project,
                    self.emb_project, self.speed_limit_emb,
                    self.unknown_speed_emb, self.traffic_emb, *self.blocks
            ]:
                for p in mod.parameters():
                    touch = touch + p.view(-1)[:1].sum()
            return lane_embedding + touch * 0.0, mask_p.reshape(
                B, lane_num), lane_feature

        lane_info = lane_info[valid_indices]

        lane_info = self.channel_pre_project(lane_info)
        lane_info = lane_info.permute(0, 2, 1)
        lane_info = self.token_pre_project(lane_info)
        lane_info = lane_info.permute(0, 2, 1)
        for block in self.blocks:
            lane_info = block(lane_info)

        lane_info = lane_info.float().mean(dim=1).to(lane_info.dtype)

        # Reshape speed_limit and traffic to match flattened dimensions
        speed_limit = speed_limit.reshape(B * lane_num, 1)
        has_speed_limit = has_speed_limit.to(torch.bool).reshape(
            B * lane_num, 1)
        traffic = traffic.reshape(B * lane_num, -1)

        # Apply embedding directly to valid speed limit data
        has_speed_limit = has_speed_limit[valid_indices].squeeze(-1)
        speed_limit = speed_limit[valid_indices].squeeze(-1)

        # [FIX] 버퍼 dtype을 lane_info(dtype)로 강제
        speed_limit_embedding = torch.zeros(
            (speed_limit.shape[0], self._channel),
            device=lane_info.device,
            dtype=lane_info.dtype,
        )
        if has_speed_limit.sum().item() > 0:
            speed_limit_with_limit = self.speed_limit_emb(
                speed_limit[has_speed_limit].unsqueeze(-1)).to(lane_info.dtype)
            speed_limit_embedding[has_speed_limit] = speed_limit_with_limit

        if (~has_speed_limit).sum().item() > 0:
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                (~has_speed_limit).sum().item(), -1).to(lane_info.dtype)
            speed_limit_embedding[~has_speed_limit] = speed_limit_no_limit

        # Process traffic lights directly for valid positions
        traffic = traffic[valid_indices]
        traffic_light_embedding = self.traffic_emb(
            traffic)  # Traffic light embedding for valid data

        lane_info = lane_info + speed_limit_embedding + traffic_light_embedding
        lane_info = self.emb_project(self.norm(lane_info))

        out_dtype = lane_info.dtype
        lane_embedding = torch.zeros((B * lane_num, lane_info.shape[-1]),
                                     device=lane_info.device,
                                     dtype=out_dtype)
        lane_embedding[valid_indices] = lane_info  # Fill in valid parts

        return lane_embedding.reshape(B, lane_num,
                                      -1), mask_p.reshape(B, -1), lane_feature


class FusionEncoder(nn.Module):

    def __init__(
            self,
            hidden_dim=192,
            num_heads=8,
            drop_path_rate=0.2,
            depth=3,
            attn_drop_p: float = 0.025,  # 권장 0.0~0.1
            ffn_drop_p: float = 0.05,  # 권장 0.0~0.1
            device='cuda'):
        super().__init__()

        # 1) CLS/scene 토큰과 그 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)
        # 0→drop_path_rate 선형 스케줄
        dpr_list = torch.linspace(0.0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                hidden_dim,
                num_heads,
                attn_drop_p=attn_drop_p,
                ffn_drop_p=ffn_drop_p,
                drop_path_p=dpr_list[i],
            ) for i in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
            self, encoding_input: torch.Tensor,
            encoding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """장면 융합 전용 포워드(배치별 전부 패딩 샘플은 건너뜀).

        모든 토큰이 패딩(True)인 배치는 연산을 생략하고 0을 반환한다.
        유효 토큰이 하나라도 있는 배치만 CLS 토큰을 붙여 블록을 통과시킨 뒤,
        최종 출력에서 CLS를 제거하고 원래 위치에 복원한다.


        # encoding_input:
        # encoding_mask:
        Args:
            encoding_input (torch.Tensor): 입력 토큰 시퀀스.
                모양: [B, token_num, H] (B, token_num, hidden_dim)
                - B: 배치 크기
                - token_num: 토큰 길이
                - H: 히든 차원
            encoding_mask (torch.Tensor): 키 패딩 마스크(True=패딩으로 무시).
                모양: [B, token_num]  (B, token_num)

        Returns:
            torch.Tensor: CLS 제외 최종 시퀀스 임베딩.
                모양: [B, token_num, H]
        """
        B, token_num, H = encoding_input.shape  # B, token_num, H 스칼라

        # 1) 전체 패딩 배치 식별 및 결과 버퍼 준비
        is_invalid_batch = encoding_mask.all(dim=1)  # [B]
        is_valid_batch = ~is_invalid_batch  # [B]
        out_tokens = encoding_input.new_zeros(B, token_num,
                                              H)  # [B, token_num, H]

        # 2) 유효 배치만 선택
        if is_valid_batch.any().item():
            # [on_B, token_num, H]
            on_batch_tokens = encoding_input[is_valid_batch]
            # [on_B, token_num]
            on_batch_token_mask = encoding_mask[is_valid_batch]

            on_B: int = on_batch_tokens.size(0)

            # 3) CLS 부착 및 CLS 위치 임베딩 추가
            cls_tokens = self.cls_token.expand(on_B, 1, H).to(
                on_batch_tokens.dtype)  # [on_B, 1, H]
            cls_with_tokens = torch.cat([cls_tokens, on_batch_tokens],
                                        dim=1)  # [on_B, token_num+1, H]
            cls_pos = self.cls_pos.to(cls_with_tokens.dtype)
            cls_with_tokens[:, 0:
                            1, :] = cls_with_tokens[:, 0:
                                                    1, :] + cls_pos  # [on_B, 1, H] += pos

            # 4) 마스크에 CLS(False) 추가
            cls_false = torch.zeros(
                on_B, 1, dtype=torch.bool,
                device=on_batch_token_mask.device)  # [on_B,1]
            cls_with_token_mask = torch.cat([cls_false, on_batch_token_mask],
                                            dim=1)  # [on_B, token_num+1]

            # 5) 블록 통과
            # cls_with_tokens  # [on_B, token_num+1, H]
            for block in self.blocks:
                cls_with_tokens = block(
                    cls_with_tokens,  # [on_B, token_num+1, H]
                    cls_with_token_mask)  # [on_B, token_num+1]
            cls_with_tokens = self.norm(
                cls_with_tokens)  # [on_B, token_num+1, H]
            # >>> add this to keep padded tokens truly zero <<<
            cls_with_tokens = cls_with_tokens.masked_fill(
                cls_with_token_mask.unsqueeze(-1), 0.0)

            # 6) CLS 제거 후 원래 배치 위치에 복원
            fused_wo_cls = cls_with_tokens[:, 1:, :]  # [on_B, token_num, H]
            out_tokens[is_valid_batch] = fused_wo_cls.to(
                out_tokens.dtype)  # [B, token_num, H]
        else:
            # 모든 배치가 패딩이면, 블록 파라미터를 0-스케일로 터치해 DDP unused param 방지
            touch = (self.cls_token[..., :1].sum() +
                     self.cls_pos[..., :1].sum())
            for blk in self.blocks:
                for p in blk.parameters():
                    touch = touch + p.view(-1)[:1].sum()
            out_tokens = out_tokens + touch * 0.0
            # ★ 추가: final LayerNorm 파라미터도 터치
            for p in self.norm.parameters():
                touch = touch + p.view(-1)[:1].sum()
        # 전부 패딩 배치는 out_tokens의 0 유지
        # out_tokens [B, token_num, H]
        # encoding_mask: [B, token_num]  # True=패딩
        return out_tokens, encoding_mask


"""
class FusionEncoder(nn.Module):


    def forward(self, encoding_input: torch.Tensor,
                encoding_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        장면 융합 포워드(항상 전체 배치를 블록에 통과).

        - CLS 토큰을 모든 배치에 부착한 뒤, 블록들을 항상 호출합니다.
        - 마스크(True=pad)는 CLS 앞에 False를 추가해 전달합니다.
        - 블록/정규화 이후, 마스크 위치는 0으로 유지합니다.
        - 최종적으로 CLS를 제거하고 (B, token_num, H)로 복원합니다.

        Args:
            encoding_input: (B, token_num, H)
            encoding_mask:  (B, token_num)  True=pad

        Returns:
            fused_wo_cls: (B, token_num, H)
            encoding_mask: (B, token_num)  # 입력 마스크 그대로 반환

        B, token_num, H = encoding_input.shape

        # 1) CLS 부착 + CLS 위치 임베딩
        x = torch.cat([self.cls_token.expand(B, 1, H).to(encoding_input.dtype),
                       encoding_input], dim=1)  # (B, 1+token_num, H)
        x[:, 0:1, :] = x[:, 0:1, :] + self.cls_pos.to(x.dtype)

        # 2) 마스크에 CLS(False) 추가
        cls_false = torch.zeros(B, 1, dtype=torch.bool, device=encoding_mask.device)
        mask_with_cls = torch.cat([cls_false, encoding_mask], dim=1)  # (B, 1+token_num)

        # 3) 블록 통과 (항상 호출)
        for block in self.blocks:
            x = block(x, mask_with_cls)  # SelfAttentionBlock는 (B, 1+L, H)/(B, 1+L) 기대

        # 4) 최종 정규화 + pad 위치 0 유지
        x = self.norm(x)
        x = x.masked_fill(mask_with_cls.unsqueeze(-1), 0.0)

        # 5) CLS 제거
        fused_wo_cls = x[:, 1:, :]  # (B, token_num, H)

        return fused_wo_cls.to(encoding_input.dtype), encoding_mask

"""
