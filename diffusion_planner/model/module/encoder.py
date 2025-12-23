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

from typing import Tuple, Dict, Optional, List, Iterable
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
      near_route_lanes:       (B, Pnn, route_num, D)
      near_route_lanes_mask:  (B, Pnn, route_num)   # True=pad(무효)

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
            near_route_lanes: torch.Tensor,  # (B, Pnn, route_num, D)
            near_route_lanes_mask: torch.Tensor,  # (B, Pnn, route_num) True=pad
    ) -> torch.Tensor:  # (B, Pnn, D)
        assert near_route_lanes.dim() == 4 and near_route_lanes_mask.dim() == 3
        B, Pnn, route_num, D = near_route_lanes.shape
        assert D == self.hidden_dim, f"hidden_dim mismatch: got {D}, expected {self.hidden_dim}"
        assert near_route_lanes_mask.shape == (B, Pnn, route_num)

        # ---- 준비: dtype/device & 모양 변환 ----
        near_route_lanes = self.in_norm(near_route_lanes)  # (B,Pnn,route_num,D)
        near_route_lanes_mask = near_route_lanes_mask.to(
            torch.bool)  # (B,Pnn,route_num)
        near_route_lanes = near_route_lanes.masked_fill(
            near_route_lanes_mask.unsqueeze(-1), 0.0)  # 패딩은 0 고정

        B_Pnn = B * Pnn
        route_lanes_3 = near_route_lanes.reshape(B_Pnn, route_num,
                                                 D)  # (B_Pnn,route_num,D)
        route_lanes_mask_2 = near_route_lanes_mask.reshape(
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
        self.road_safety_encoder = RoadSafetyFusionEncoder(
            hidden_dim=config.hidden_dim,
            num_seeds=getattr(config, "road_safety_num_seeds", 8),
            attn_drop_p=0.0,
            out_drop_p=0.0,
        )
        self.lane_encoder = LaneFusionEncoder(
            config.lane_len,
            drop_path_rate=config.encoder_drop_path_rate,
            hidden_dim=config.hidden_dim,
            depth=config.encoder_depth,
            num_fourier_frequencies=self.num_fourier_frequencies,
            time_gap=self.time_gap,
            time_min=self.time_min,
            time_max=self.time_max)
        lane_summary_num: int = int(getattr(config, "lane_summary_num", 0))
        if lane_summary_num > 0:
            self.lane_summary_pooler = LaneSummaryTokenPooler(
                hidden_dim=config.hidden_dim,
                num_seeds=lane_summary_num,
                attn_drop_p=0.0,
                out_drop_p=0.0,
            )
        else:
            self.lane_summary_pooler = None
        self.npc_route_encoder = NearAgentsRouteLaneEncoder(
            hidden_dim=config.hidden_dim, attn_drop_p=0.0)

        self.fusion = FusionEncoder(
            hidden_dim=config.hidden_dim,
            num_heads=config.num_heads,
            drop_path_rate=config.encoder_drop_path_rate,
            depth=config.encoder_depth,
            device=config.device)

        # position embedding encode
        # [x, y, cos, sin] + type_onehot(5) = 9
        # type_onehot: (ego, neighbor, static, lane, road_safety)
        self.pos_emb = nn.Linear(9, config.hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def iter_encoder_local_parameters(self) -> Iterator[nn.Parameter]:
        """로컬 인코더(Group A)에 속한 파라미터들을 순서대로 돌려줍니다.

        로컬 인코더는 개별 차량, 정적 물체, 차선, 도로 안전 요소, 위치 임베딩 등을 읽는 부분입니다.
        """
        local_modules = [
            self.agents_encoder,
            self.static_encoder,
            self.lane_encoder,
            self.road_safety_encoder,
        ]
        for module in local_modules:
            for param in module.parameters():
                yield param

        for param in self.pos_emb.parameters():
            # param: (hidden_dim, 9) 또는 (hidden_dim,)
            yield param

        # pos_scale: shape () 스칼라 파라미터
        yield self.pos_scale

    def iter_encoder_global_parameters(self) -> Iterator[nn.Parameter]:
        """글로벌 인코더(Group B)에 속한 파라미터들을 순서대로 돌려줍니다.

        글로벌 인코더는 여러 객체를 합쳐 장면 전체를 요약하는 부분입니다.
        """
        global_modules = [
            self.fusion,
            self.npc_route_encoder,
            self.lane_summary_pooler,
        ]
        for module in global_modules:
            if module is None:
                continue
            for param in module.parameters():
                yield param

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
    def _road_safety_inputs_are_all_none(
        stop_sign_points: Optional[torch.Tensor],
        stop_sign_is_valid: Optional[torch.Tensor],
        crosswalk_points: Optional[torch.Tensor],
        crosswalk_is_valid: Optional[torch.Tensor],
        speed_bump_points: Optional[torch.Tensor],
        speed_bump_is_valid: Optional[torch.Tensor],
        driveway_points: Optional[torch.Tensor],
        driveway_is_valid: Optional[torch.Tensor],
        road_edge: Optional[torch.Tensor],
        road_edge_is_valid: Optional[torch.Tensor],
        road_edge_type: Optional[torch.Tensor],
    ) -> bool:
        """road-safety 입력이 전부 None인지 확인합니다.

        이 함수의 목적은 아주 단순합니다.
        road-safety 관련 입력이 “하나도 안 들어온 경우”를 빠르게 감지해서,
        아래의 빈 토큰 생성 로직으로 안전하게 넘어가기 위함입니다.

        Args:
            stop_sign_points: (B, Ns, S, 2) 또는 None
            stop_sign_is_valid: (B, Ns) 또는 None
            crosswalk_points: (B, Nc, S, 2) 또는 None
            crosswalk_is_valid: (B, Nc) 또는 None
            speed_bump_points: (B, Nb, S, 2) 또는 None
            speed_bump_is_valid: (B, Nb) 또는 None
            driveway_points: (B, Nd, S, 2) 또는 None
            driveway_is_valid: (B, Nd) 또는 None
            road_edge: (B, Ne, S, 2) 또는 None
            road_edge_is_valid: (B, Ne) 또는 None
            road_edge_type: (B, Ne, 3) 또는 None

        Returns:
            bool:
                - 전부 None이면 True
                - 하나라도 텐서가 있으면 False
        """
        tensors = [
            stop_sign_points,
            stop_sign_is_valid,
            crosswalk_points,
            crosswalk_is_valid,
            speed_bump_points,
            speed_bump_is_valid,
            driveway_points,
            driveway_is_valid,
            road_edge,
            road_edge_is_valid,
            road_edge_type,
        ]
        return all(t is None for t in tensors)

    def _build_empty_road_safety_tokens(
        self,
        batch_size: int,
        ref_encoding: torch.Tensor,
        ref_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """road-safety 입력이 아예 없을 때 사용할 '빈 토큰'을 만듭니다.

        해야 하는 일은 2가지입니다.

        1) road-safety 토큰을 “있다고 가정”하되,
           실제 값은 전부 0이고, 마스크는 전부 True(=패딩, 무효)로 만들어
           이후 과정(Fusion 등)이 끊기지 않고 그대로 진행되게 합니다.

        2) 여러 GPU로 학습할 때(여러 장치로 나눠 학습하는 경우),
           road_safety_encoder 쪽 파라미터가 “이번 입력에서 전혀 안 쓰였다”고 판단되면
           학습이 중단될 수 있습니다.
           그래서 값에는 영향을 주지 않도록(0을 곱해서) 아주 약하게 연결해 둡니다.

        Args:
            batch_size (int):
                배치 크기 B.
            ref_encoding (torch.Tensor):
                road-safety 임베딩 텐서의 dtype/device를 맞추기 위한 기준 텐서.
                예: encoding_agents_chunk (B, N_agents_tok, H)
            ref_pos (torch.Tensor):
                road-safety 위치(pos) 텐서의 dtype/device를 맞추기 위한 기준 텐서.
                예: agents_chunk_pos (B, N_agents_tok, 9)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - encoding_road_safety: (B, M, H)  (값 0)
                - road_safety_mask:     (B, M)     (전부 True=패딩)
                - road_safety_pos:      (B, M, 9)  (값 0, type_onehot의 road_safety 위치만 1로 설정 가능)

                여기서
                - B = batch_size
                - M = self.road_safety_encoder.num_seeds
                - H = self.hidden_dim
        """
        B: int = int(batch_size)
        M: int = int(getattr(self.road_safety_encoder, "num_seeds"))
        H: int = int(self.hidden_dim)

        # ----- (A) road-safety 임베딩/마스크 -----
        # encoding_road_safety: (B, M, H)
        encoding_road_safety: torch.Tensor = torch.zeros(
            (B, M, H),
            device=ref_encoding.device,
            dtype=ref_encoding.dtype,
        )
        # road_safety_mask: (B, M)  True=pad
        road_safety_mask: torch.Tensor = torch.ones(
            (B, M),
            device=ref_encoding.device,
            dtype=torch.bool,
        )

        # ----- (B) road-safety pos -----
        # road_safety_pos: (B, M, 9) = [x,y,cos,sin] + type_onehot(5)
        # 여기서는 "빈 토큰"이므로 좌표는 0, 타입은 road_safety 인덱스만 1로 둡니다.
        road_safety_pos: torch.Tensor = torch.zeros(
            (B, M, 9),
            device=ref_pos.device,
            dtype=ref_pos.dtype,
        )
        if M > 0:
            # type_onehot(5)의 마지막(road_safety) 위치 = index 8
            road_safety_pos[:, :, 8] = 1.0

        # ----- (C) 파라미터를 0계수로 아주 약하게 연결(값은 그대로 0) -----
        # touch_f32: shape ()
        touch_f32: torch.Tensor = torch.zeros((),
                                              device=ref_encoding.device,
                                              dtype=torch.float32)
        for p in self.road_safety_encoder.parameters():
            touch_f32 = touch_f32 + p.view(-1)[:1].sum().float()

        # encoding dtype에 맞춰 캐스팅 후 0을 곱해 더함 (값 변화 없음)
        touch: torch.Tensor = touch_f32.to(dtype=ref_encoding.dtype)
        encoding_road_safety = encoding_road_safety + touch * 0.0

        return encoding_road_safety, road_safety_mask, road_safety_pos

    @staticmethod
    def _mask_all_routes_like(
        near_route_lanes: torch.Tensor,  # (B, Pnn, route_num, H)
        near_route_lanes_mask: torch.Tensor,  # (B, Pnn, route_num)
        drop_mask_b: torch.Tensor,  # (B,) bool, False인 샘플을 "경로 없음"으로 강제
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """배치 마스크에 따라 해당 샘플의 모든 route 토큰을 패딩 처리합니다.

        Args:
            near_route_lanes: (B, Pnn, route_num, H)
            near_route_lanes_mask: (B, Pnn, route_num)  # True=pad
            drop_mask_b: (B,) bool         # True=유지(keep), False=드롭(경로 없음)

        Returns:
            (near_route_lanes, near_route_lanes_mask)  # same shapes
        """
        if drop_mask_b.dtype != torch.bool:
            drop_mask_b = drop_mask_b.to(torch.bool)

        if (~drop_mask_b).any().item():
            # (B,) -> (B,1,1) for broadcast
            b_drop = (~drop_mask_b).view(-1, 1, 1)  # True = 드롭 대상 배치
            # 값/포지션: float 복사 마스크로 곱셈(새 텐서 반환)
            keep_f_lanes = (~b_drop).unsqueeze(-1).to(
                near_route_lanes.dtype)  # (B,1,1,1)
            near_route_lanes = near_route_lanes * keep_f_lanes

            # 마스크: out‑of‑place 결합(원본 텐서 저장소를 수정하지 않음)
            near_route_lanes_mask = near_route_lanes_mask | b_drop.expand_as(
                near_route_lanes_mask)
        return near_route_lanes, near_route_lanes_mask

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
            self,
            planner_future_11_dim: torch.Tensor,  # (B, future_len, 11)
            known_mask: torch.Tensor) -> torch.Tensor:
        """미래 궤적을 [처음 M_i 스텝 유지 + 나머지 0패딩]으로 변환합니다.

        주의:
        - 에이전트 인코더의 assert(길이 고정)를 만족시키기 위해 **길이는 그대로 N 유지**합니다.
        - 마스크 판단은 에이전서 내부에서 첫 8채널이 0인지로 이루어지므로,
          M_i 이후 프레임은 **앞 8채널을 0**으로 채웁니다.
        - 마지막 3채널(type)은 항상 ego를 의미하도록 **[1,0,0]**(또는 입력의 ego one-hot)을 유지합니다.

        Args:
            planner_future_11_dim (torch.Tensor):
                형태: [B, future_len, 11]
                채널: [x, y, cos, sin, vx, vy, w, l, type(3)]
            known_mask (torch.Tensor):
                형태: [B, future_len], bool
                True=조건 제공(유지), False=미제공(패딩)

        Returns:
            torch.Tensor:
                형태: [B, future_len, 11]
                처음 M_i는 원본 유지, 나머지는 앞 8채널 0, type 3채널은 ego one-hot 유지.
        """
        B, future_len, D = planner_future_11_dim.shape  # (B, future_len, 11)
        assert D == 11, "ego_future_full의 마지막 차원은 11이어야 합니다."

        # (B, 3) ego 타입 벡터를 첫 프레임에서 추출(이미 one-hot이라 가정)
        ego_type = planner_future_11_dim[:, 0, 8:11].clone()  # (B, 3)

        # 기본 패딩 텐서: 앞 8채널=0, type 3채널=ego one-hot 반복
        padded_default = planner_future_11_dim.new_zeros(
            B, future_len, D)  # (B, future_len, 11)
        padded_default[:, :,
                       8:11] = ego_type.unsqueeze(1).expand(-1, future_len, -1)

        # keep: (B, future_len, 1)  True=원본 유지
        keep = known_mask.unsqueeze(-1)

        # 최종: 알려진 구간은 원본, 나머지는 기본 패딩
        truncated_ego_future = torch.where(
            keep, planner_future_11_dim, padded_default)  # (B, future_len, 11)
        return truncated_ego_future

    def _prepare_encoder_inputs(
        self,
        inputs: Dict[str, torch.Tensor],
    ) -> Tuple[
            Optional[torch.
                     Tensor],  # ego_agent_past: (B, 1, time_len, 11) or None
            Optional[
                torch.
                Tensor],  # planner_future_11_dim: (B, future_len, 11) or None
            Optional[
                torch.
                Tensor],  # neighbor_agents_past: (B, A, time_len, 11) or None
            Optional[
                torch.
                Tensor],  # non_near_agents_past: (B, non_near_A, time_len, 11) or None
            Optional[torch.Tensor],  # static_objects: (B, P, D_static) or None
            Optional[torch.Tensor],  # lanes: (B, L, lane_len, D_lane) or None
            Optional[torch.Tensor],  # lanes_speed_limit: (B, L, 1) or None
            Optional[torch.Tensor],  # lanes_has_speed_limit: (B, L, 1) or None
            Optional[torch.
                     Tensor],  # agent_route_lane_order: (B, Pnn, L) or None
            Optional[torch.Tensor],  # lane_type: (B, L, 4) or None
            Optional[torch.Tensor],  # left_line_type: (B, L, 13) or None
            Optional[torch.Tensor],  # right_line_type: (B, L, 13) or None
            Optional[
                torch.
                Tensor],  # stop_sign_points: (B, Ns, safety_len, 2) or None
            Optional[torch.Tensor],  # stop_sign_is_valid: (B, Ns) bool or None
            Optional[
                torch.
                Tensor],  # crosswalk_points: (B, Nc, safety_len, 2) or None
            Optional[torch.Tensor],  # crosswalk_is_valid: (B, Nc) bool or None
            Optional[
                torch.
                Tensor],  # speed_bump_points: (B, Nb, safety_len, 2) or None
            Optional[torch.Tensor],  # speed_bump_is_valid: (B, Nb) bool or None
            Optional[torch.
                     Tensor],  # driveway_points: (B, Nd, safety_len, 2) or None
            Optional[torch.Tensor],  # driveway_is_valid: (B, Nd) bool or None
            Optional[torch.Tensor],  # road_edge: (B, E, safety_len, 2) or None
            Optional[torch.Tensor],  # road_edge_is_valid: (B, E) bool or None
            Optional[torch.Tensor],  # road_edge_type: (B, E, 3) or None
            int,  # B
            int,  # future_len
    ]:
        """입력 dict에서 인코더가 쓸 텐서들을 꺼내고 모양을 맞춥니다.

        요구사항 반영
        ------------
        - **모든 값**을 `inputs.get("key", None)` 방식으로 가져옵니다.
          즉, 키가 없으면 해당 출력은 `None` 이 됩니다.

        추가 동작
        --------
        1) ego 과거 궤적이 존재하면 (B, T, 11) → (B, 1, T, 11)로 바꿉니다.
        2) 속도 입력을 쓰지 않는 설정이면, 존재하는 텐서에 한해 vx, vy 채널([4:6])을 0으로 만듭니다.
        3) 평가 모드에서 route encoding을 강제로 무시하도록 켜면,
           존재하는 `agent_route_lane_order`에 한해 전부 -1로 채웁니다.

        Args:
            inputs (Dict[str, torch.Tensor]):
                키가 없을 수 있으므로, 모든 키는 optional로 취급합니다.
                예:
                    - "ego_agent_past":          (B, time_len, 11)
                    - "planner_future_11_dim":   (B, future_len, 11)
                    - "neighbor_agents_past":    (B, A, time_len, 11)
                    - "static_objects":          (B, P, D_static)
                    - "lanes":                   (B, L, lane_len, D_lane)
                    - "lanes_speed_limit":       (B, L, 1)
                    - "lanes_has_speed_limit":   (B, L, 1)
                    - "agent_route_lane_order":  (B, Pnn, L)
                    - "lane_type":               (B, L, 4)
                    - "left_line_type":          (B, L, 13)
                    - "right_line_type":         (B, L, 13)
                    - "stop_sign_points":        (B, Ns, safety_len, 2)
                    - "crosswalk_points":        (B, Nc, safety_len, 2)
                    - "speed_bump_points":       (B, Nb, safety_len, 2)
                    - "driveway_points":         (B, Nd, safety_len, 2)
                    - "road_edge":               (B, E, safety_len, 2)
                    - "road_edge_type":          (B, E, 3)

        Returns:
            Tuple[...]:
                각 키에 해당하는 텐서(없으면 None) + (B, future_len)

                - B:
                    우선순위로 배치 크기를 추정합니다.
                    1) neighbor_agents_past.shape[0]
                    2) ego_agent_past(before unsqueeze).shape[0]
                    3) planner_future_11_dim.shape[0]
                    4) 그 외 전부 None이면 0
                - future_len:
                    planner_future_11_dim이 있으면 shape[1], 없으면 0
        """
        # --- 1) 전부 get(...) 로 가져오기 ---
        ego_agent_past: Optional[torch.Tensor] = inputs.get(
            "ego_agent_past", None)  # (B, T, 11) or None
        planner_future_11_dim: Optional[torch.Tensor] = inputs.get(
            "planner_future_11_dim", None)  # (B, Tf, 11) or None
        neighbor_agents_past: Optional[torch.Tensor] = inputs.get(
            "neighbor_agents_past", None)  # (B, A, T, 11) or None
        non_near_agents_past: Optional[torch.Tensor] = inputs.get(
            "non_near_agents_past", None)  # (B, non_near_A, T, 11) or None

        static_objects: Optional[torch.Tensor] = inputs.get(
            "static_objects", None)  # (B, P, D_static) or None
        lanes: Optional[torch.Tensor] = inputs.get(
            "lanes", None)  # (B, L, lane_len, D_lane) or None
        lanes_speed_limit: Optional[torch.Tensor] = inputs.get(
            "lanes_speed_limit", None)  # (B, L, 1) or None
        lanes_has_speed_limit: Optional[torch.Tensor] = inputs.get(
            "lanes_has_speed_limit", None)  # (B, L, 1) or None
        agent_route_lane_order: Optional[torch.Tensor] = inputs.get(
            "agent_route_lane_order", None)  # (B, Pnn, L) or None

        lane_type: Optional[torch.Tensor] = inputs.get(
            "lane_type", None)  # (B, L, 4) or None
        left_line_type: Optional[torch.Tensor] = inputs.get(
            "left_line_type", None)  # (B, L, 13) or None
        right_line_type: Optional[torch.Tensor] = inputs.get(
            "right_line_type", None)  # (B, L, 13) or None

        stop_sign_points: Optional[torch.Tensor] = inputs.get(
            "stop_sign_points", None)  # (B, Ns, safety_len, 2) or None
        stop_sign_is_valid: Optional[torch.Tensor] = inputs.get(
            "stop_sign_is_valid", None)  # (B, Ns) bool or None
        crosswalk_points: Optional[torch.Tensor] = inputs.get(
            "crosswalk_points", None)  # (B, Nc, safety_len, 2) or None
        crosswalk_is_valid: Optional[torch.Tensor] = inputs.get(
            "crosswalk_is_valid", None)  # (B, Nc) bool or None
        speed_bump_points: Optional[torch.Tensor] = inputs.get(
            "speed_bump_points", None)  # (B, Nb, safety_len, 2) or None
        speed_bump_is_valid: Optional[torch.Tensor] = inputs.get(
            "speed_bump_is_valid", None)  # (B, Nb) bool or None
        driveway_points: Optional[torch.Tensor] = inputs.get(
            "driveway_points", None)  # (B, Nd, safety_len, 2) or None
        driveway_is_valid: Optional[torch.Tensor] = inputs.get(
            "driveway_is_valid", None)  # (B, Nd) bool or None

        road_edge: Optional[torch.Tensor] = inputs.get(
            "road_edge", None)  # (B, E, safety_len, 2) or None
        road_edge_is_valid: Optional[torch.Tensor] = inputs.get(
            "road_edge_is_valid", None)  # (B, E) bool or None
        road_edge_type: Optional[torch.Tensor] = inputs.get(
            "road_edge_type", None)  # (B, E, 3) or None

        # --- 2) ego/neighbor 속도 채널 제거 + ego 차원 맞추기 ---
        if ego_agent_past is not None:
            if not self.config.use_vel_input:
                ego_agent_past[:, :, 4:6] = 0.0  # vx, vy
            ego_agent_past = ego_agent_past.unsqueeze(1)  # (B, 1, T, 11)

        if neighbor_agents_past is not None:
            if not self.config.use_vel_input:
                neighbor_agents_past[:, :, :, 4:6] = 0.0  # vx, vy
        if non_near_agents_past is not None:
            if not self.config.use_vel_input:
                non_near_agents_past[:, :, :, 4:6] = 0.0  # vx, vy

        # --- 3) B / future_len 계산(없으면 0) ---
        future_len: int = int(planner_future_11_dim.shape[1]
                             ) if planner_future_11_dim is not None else 0

        B: int = 0
        if neighbor_agents_past is not None:
            B = int(neighbor_agents_past.shape[0])
        elif ego_agent_past is not None:
            B = int(ego_agent_past.shape[0])
        elif planner_future_11_dim is not None:
            B = int(planner_future_11_dim.shape[0])

        # --- 4) (옵션) 평가 모드에서 route encoding 강제 무시 ---
        self.neglect_route_encoding = False
        if (not self.training) and bool(self.neglect_route_encoding):
            if agent_route_lane_order is not None:
                agent_route_lane_order = torch.full_like(agent_route_lane_order,
                                                         fill_value=-1)

        return (
            ego_agent_past,
            planner_future_11_dim,
            neighbor_agents_past,
            non_near_agents_past,
            static_objects,
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            agent_route_lane_order,
            lane_type,
            left_line_type,
            right_line_type,
            stop_sign_points,
            stop_sign_is_valid,
            crosswalk_points,
            crosswalk_is_valid,
            speed_bump_points,
            speed_bump_is_valid,
            driveway_points,
            driveway_is_valid,
            road_edge,
            road_edge_is_valid,
            road_edge_type,
            B,
            future_len,
        )

    def _compute_ego_future_trajectory(
        self,
        planner_future_11_dim: torch.Tensor,  # (B, future_len, 11)
        B: int,
        future_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """훈련/추론 모드에 맞게 ego_future_trajectory 를 만든다.

        - 학습 중(self.training=True):
            · 각 배치별로 M_i ~ Uniform{0..future_len} 를 뽑고,
            · 처음 M_i 스텝만 유지, 나머지 스텝은 앞 8채널을 0으로 패딩한다.
        - 평가 중:
            · 그대로 planner_future_11_dim 을 사용한다.
            · 혹시 None 이면 (B, future_len, 11) all-zero 텐서를 생성한다.

        Args:
            planner_future_11_dim: (B, future_len, 11)
            B: 배치 크기.
            future_len: 미래 길이.
            device: 텐서를 올릴 디바이스.

        Returns:
            ego_future_trajectory: (B, future_len, 11)
        """
        if self.config.do_ego_predict:
            ego_future_trajectory = torch.zeros(
                (B, future_len, 11),
                device=device,
                dtype=planner_future_11_dim.dtype,
            )
            return ego_future_trajectory
        if self.training:
            # (1) M_i 샘플링: (B,)
            prefix_lengths: torch.Tensor = self._sample_uniform_prefix_lengths(
                batch_size=B,
                max_future_len=future_len,
                device=device,
            )  # (B,)

            # (2) M_i 로부터 알려진 구간 마스크 생성: (B, future_len)
            known_mask: torch.Tensor = self._build_known_mask_from_lengths(
                prefix_lengths=prefix_lengths,
                max_future_len=future_len,
            )  # (B, future_len)

            # (3) 알려진 구간만 유지하고 나머지는 앞 8채널 0으로 패딩
            ego_future_trajectory: torch.Tensor = \
                self._truncate_and_pad_ego_future_for_encoder(
                    planner_future_11_dim,  # (B, future_len, 11)
                    known_mask,             # (B, future_len)
                )  # (B, future_len, 11)
        else:
            ego_future_trajectory = planner_future_11_dim
            if ego_future_trajectory is None:
                ego_future_trajectory = torch.zeros(
                    (B, future_len, 11),
                    device=device,
                    dtype=planner_future_11_dim.dtype,
                )

        # 속도 입력을 쓰지 않으면 ego 미래의 vx, vy 도 0으로 맞춘다.
        if not self.config.use_vel_input:
            ego_future_trajectory[:, :, 4:6] = 0.0  # (B, future_len, 11)

        return ego_future_trajectory

    @staticmethod
    def _ensure_non_near_agents_past_tensor(
            ego_agent_past: torch.Tensor,  # (B, 1, T, 11)
            non_near_agents_past: Optional[torch.Tensor],
            # (B, A, T, 11) or None
    ) -> torch.Tensor:
        """non_near_agents_past를 항상 '텐서' 형태로 보장합니다.

        이 함수는 Encoder에서 AgentFusionEncoder로 넘기는 이웃 에이전트 입력이
        어떤 경우에도 안전하게 동작하도록 만들기 위한 안전장치입니다.

        동작 규칙:
            1) non_near_agents_past가 None이면,
               (B, 0, T, 11) 모양의 "빈 에이전트 텐서"를 만들어 반환합니다.
               이렇게 하면 AgentFusionEncoder 내부의 torch.cat이 항상 성공합니다.

            2) non_near_agents_past가 텐서이면,
               ego_agent_past와 배치/시간/특징 차원이 맞는지 확인한 뒤,
               device/dtype을 ego_agent_past에 맞춰 반환합니다.

        Args:
            ego_agent_past (torch.Tensor):
                shape: (B, 1, T, 11)
                ego의 과거~현재 궤적 입력입니다.

            non_near_agents_past (Optional[torch.Tensor]):
                shape: (B, A, T, 11) 또는 None
                DiT가 생성 대상으로 삼지 않는(제외된) 나머지 에이전트들의 과거~현재 입력입니다.
                A는 0일 수도 있습니다.

        Returns:
            torch.Tensor:
                shape: (B, A, T, 11)
                - non_near_agents_past가 None이면 A=0인 빈 텐서
                - 아니면 입력 텐서(단, device/dtype 정리됨)
        """
        if ego_agent_past.dim() != 4:
            raise ValueError(
                f"ego_agent_past must be (B, 1, T, 11). got {tuple(ego_agent_past.shape)}"
            )
        B: int = int(ego_agent_past.shape[0])
        T: int = int(ego_agent_past.shape[2])
        D: int = int(ego_agent_past.shape[3])
        if D != 11:
            raise ValueError(
                f"ego_agent_past last dim must be 11. got {D} (shape={tuple(ego_agent_past.shape)})"
            )

        # (1) None이면: (B, 0, T, 11) 생성
        if non_near_agents_past is None:
            return ego_agent_past.new_zeros((B, 0, T, D))

        # (2) 텐서면 shape 기본 검증
        if non_near_agents_past.dim() != 4:
            raise ValueError(
                f"non_near_agents_past must be (B, A, T, 11). got {tuple(non_near_agents_past.shape)}"
            )
        if int(non_near_agents_past.shape[0]) != B:
            raise ValueError(
                f"non_near_agents_past batch size mismatch. expected B={B}, got {int(non_near_agents_past.shape[0])}"
            )
        if int(non_near_agents_past.shape[2]) != T or int(
                non_near_agents_past.shape[3]) != D:
            raise ValueError(
                f"non_near_agents_past time/feat mismatch. expected (T={T}, D={D}), got (T={int(non_near_agents_past.shape[2])}, D={int(non_near_agents_past.shape[3])})"
            )

        # (3) device/dtype을 ego 기준으로 맞춤
        return non_near_agents_past.to(device=ego_agent_past.device,
                                       dtype=ego_agent_past.dtype)

    def _encode_agents_static_lanes(
        self,
        ego_agent_past: torch.Tensor,  # (B, 1, time_len, 11)
        non_near_agents_past: Optional[torch.Tensor], # (B, A, time_len, 11) or None
        ego_future_trajectory: torch.Tensor,  # (B, future_len, 11)
        static_objects: torch.Tensor,  # (B, P, D_static)
        lanes: torch.Tensor,  # (B, L, lane_len, D_lane)
        lanes_speed_limit: torch.Tensor,  # (B, L, 1)
        lanes_has_speed_limit: torch.Tensor,  # (B, L, 1)
        lane_type: Optional[torch.Tensor],  # (B, L, 4)
        left_line_type: Optional[torch.Tensor],  # (B, L, 13)
        right_line_type: Optional[torch.Tensor],  # (B, L, 13)
        stop_sign_points: Optional[
            torch.Tensor] = None,  # (B, Ns, safety_len, 2)
        stop_sign_is_valid: Optional[torch.Tensor] = None,  # (B, Ns)
        crosswalk_points: Optional[
            torch.Tensor] = None,  # (B, Nc, safety_len, 2)
        crosswalk_is_valid: Optional[torch.Tensor] = None,  # (B, Nc)
        speed_bump_points: Optional[
            torch.Tensor] = None,  # (B, Nb, safety_len, 2)
        speed_bump_is_valid: Optional[torch.Tensor] = None,  # (B, Nb)
        driveway_points: Optional[
            torch.Tensor] = None,  # (B, Nd, safety_len, 2)
        driveway_is_valid: Optional[torch.Tensor] = None,  # (B, Nd)
        road_edge: Optional[torch.Tensor] = None,  # (B, E, safety_len, 2)
        road_edge_is_valid: Optional[torch.Tensor] = None,  # (B, E)
        road_edge_type: Optional[torch.Tensor] = None,  # (B, E, 3)
    ) -> Tuple[
            torch.Tensor,  # encoding_agents_chunk: (B, N_agents_tok, H)
            torch.Tensor,  # agents_chunk_mask:     (B, N_agents_tok)
            torch.Tensor,  # agents_chunk_pos:      (B, N_agents_tok, 9)
            torch.Tensor,  # ego_fut_global:        (B, H)
            torch.Tensor,  # encoding_static:       (B, N_static, H)
            torch.Tensor,  # static_mask:           (B, N_static)
            torch.Tensor,  # static_pos:            (B, N_static, 9)
            torch.Tensor,  # encoding_lanes:        (B, N_lanes, H)
            torch.Tensor,  # lanes_mask:            (B, N_lanes)
            torch.Tensor,  # lane_pos:              (B, N_lanes, 9)
            torch.Tensor,  # encoding_road_safety:  (B, N_road_safety, H)
            torch.Tensor,  # road_safety_mask:      (B, N_road_safety)
            torch.Tensor,  # road_safety_pos:       (B, N_road_safety, 9)
    ]:
        """에이전트 / 정적 객체 / 차선 인코더를 한 번에 호출한다.

        Args:
            ego_agent_past: (B, 1, time_len, 11)
            neighbor_agents_past: (B, A, time_len, 11)
            ego_future_trajectory: (B, future_len, 11)
            static_objects: (B, P, D_static)
            lanes: (B, L, lane_len, D_lane)
            lanes_speed_limit: (B, L, 1)
            lanes_has_speed_limit: (B, L, 1)

        Returns:
            위 각 인코더에서 나오는 토큰/마스크/좌표 텐서들.
        """
        # ★ 핵심: None이면 (B,0,T,11)로 바꿔서 torch.cat이 항상 되게 만듦
        non_near_agents_past = self._ensure_non_near_agents_past_tensor(
            ego_agent_past=ego_agent_past,                  # (B,1,T,11)
            non_near_agents_past=non_near_agents_past,      # (B,A,T,11) or None
        )  # -> (B, A, T, 11)  (A는 0 가능)
        # --- agents encoder ---
        """
        encoding_agents_chunk: (B, agents_num * past_cur_chunk_num + future_chunk_num, hidden_dim)
        agents_chunk_mask:     (B, agents_num * past_cur_chunk_num + future_chunk_num)
        agents_chunk_pos :     (B, agents_num * past_cur_chunk_num + future_chunk_num, 9)
        ego_fut_global:        (B, hidden_dim)
        """
        (encoding_agents_chunk, agents_chunk_mask, agents_chunk_pos,
         ego_fut_global) = self.agents_encoder(
             ego_agent_past,  # (B,1,T,11)
             non_near_agents_past,  # (B,A,T,11)
             ego_future_trajectory,  # (B,future_len,11)
         )

        # PRAM 끄기: ego_fut_global은 구조상 존재하지만, 실제로는 0 벡터로 사용
        if not self.config.use_pram:
            ego_fut_global = self._zero_with_touch(ego_fut_global,
                                                   self._ego_fut_params())

        # --- static encoder ---
        """
        encoding_static: (B, static_objects_num, hidden_dim)
        static_mask:     (B, static_objects_num)
        static_pos:      (B, static_objects_num, 9)
        """
        encoding_static, static_mask, static_pos = self.static_encoder(
            static_objects)
        # --- road safety encoder (입력이 전부 None이면 "빈 토큰"으로 대체) ---
        """
        encoding_road_safety : (B, road_safety_num, hidden_dim)
        road_safety_mask     : (B, road_safety_num)
        road_safety_pos      : (B, road_safety_num, 9)
        """
        if self._road_safety_inputs_are_all_none(
                stop_sign_points,
                stop_sign_is_valid,
                crosswalk_points,
                crosswalk_is_valid,
                speed_bump_points,
                speed_bump_is_valid,
                driveway_points,
                driveway_is_valid,
                road_edge,
                road_edge_is_valid,
                road_edge_type,
        ):
            (encoding_road_safety, road_safety_mask,
             road_safety_pos) = self._build_empty_road_safety_tokens(
                 batch_size=int(encoding_agents_chunk.shape[0]),
                 ref_encoding=encoding_agents_chunk,  # (B, N_agents_tok, H)
                 ref_pos=agents_chunk_pos,  # (B, N_agents_tok, 9)
             )
        else:
            (encoding_road_safety, road_safety_mask,
             road_safety_pos) = self.road_safety_encoder(
                 stop_sign_points,
                 stop_sign_is_valid,
                 crosswalk_points,
                 crosswalk_is_valid,
                 speed_bump_points,
                 speed_bump_is_valid,
                 driveway_points,
                 driveway_is_valid,
                 road_edge,
                 road_edge_is_valid,
                 road_edge_type,
             )
        # --- lane encoder ---
        """
        encoding_lanes: (B, lane_num, hidden_dim)
        lanes_mask:     (B, lane_num)
        lane_pos:       (B, lane_num, 9)
        """
        encoding_lanes, lanes_mask, lane_pos = self.lane_encoder(
            lanes, lanes_speed_limit, lanes_has_speed_limit, lane_type,
            left_line_type, right_line_type)

        return (encoding_agents_chunk, agents_chunk_mask, agents_chunk_pos,
                ego_fut_global, encoding_static, static_mask, static_pos,
                encoding_lanes, lanes_mask, lane_pos, encoding_road_safety,
                road_safety_mask, road_safety_pos)

    def _add_pos_embedding_to_tokens(
            self,
            token_embeddings: torch.Tensor,  # (B, N, H)
            token_pos: torch.Tensor,  # (B, N, 9)
            token_mask: torch.Tensor,  # (B, N) True=pad
    ) -> torch.Tensor:
        """토큰 임베딩에 위치 정보를 더해줍니다.

        이 함수는 다음을 보장합니다.
        - 패딩 토큰(mask=True)은 위치 임베딩을 더하지 않습니다.
        - 유효 토큰(mask=False)만 골라 pos_emb(Linear(9→H))를 적용합니다.
        - pos_scale(스칼라)을 곱한 뒤 token_embeddings에 더합니다.

        Args:
            token_embeddings:
                shape: (B, N, H)
            token_pos:
                shape: (B, N, 9)
                [x,y,cos,sin] + type_onehot(5)
            token_mask:
                shape: (B, N)  True=패딩(무효)

        Returns:
            torch.Tensor:
                shape: (B, N, H)
                위치 임베딩이 더해진 토큰 텐서
        """
        if token_mask.dtype != torch.bool:
            token_mask = token_mask.to(torch.bool)

        if token_embeddings.dim() != 3 or token_pos.dim(
        ) != 3 or token_mask.dim() != 2:
            raise ValueError(
                f"Expected token_embeddings (B,N,H), token_pos (B,N,9), token_mask (B,N). "
                f"got {tuple(token_embeddings.shape)}, {tuple(token_pos.shape)}, {tuple(token_mask.shape)}"
            )

        B, N, H = token_embeddings.shape
        if token_pos.shape != (B, N, 9):
            raise ValueError(
                f"token_pos must be (B, N, 9). got {tuple(token_pos.shape)}")
        if token_mask.shape != (B, N):
            raise ValueError(
                f"token_mask must be (B, N). got {tuple(token_mask.shape)}")

        # (B*N,)
        mask_flat: torch.Tensor = token_mask.reshape(-1)
        # (B*N, 9)
        pos_flat: torch.Tensor = token_pos.reshape(B * N, 9)

        # 유효 토큰만 pos_emb 적용
        if (~mask_flat).any().item():
            pos_valid: torch.Tensor = self.pos_emb(
                pos_flat[~mask_flat])  # (N_valid, H)
            pos_valid = pos_valid.to(dtype=token_embeddings.dtype,
                                     device=token_embeddings.device)

            scale: torch.Tensor = self.pos_scale.to(
                dtype=token_embeddings.dtype, device=token_embeddings.device)
            pos_valid = scale * pos_valid  # (N_valid, H)

            pos_result_flat: torch.Tensor = torch.zeros(
                (B * N, H),
                device=token_embeddings.device,
                dtype=token_embeddings.dtype,
            )
            pos_result_flat[~mask_flat] = pos_valid  # (B*N, H)
            pos_result = pos_result_flat.view(B, N, H)  # (B,N,H)
        else:
            # 유효 토큰이 하나도 없으면 pos_result는 0
            pos_result = torch.zeros((B, N, H),
                                     device=token_embeddings.device,
                                     dtype=token_embeddings.dtype)

        return token_embeddings + pos_result

    def _build_fusion_inputs(
        self,
        encoding_agents_chunk: torch.Tensor,  # (B, N_agents_tok, H)
        agents_chunk_mask: torch.Tensor,  # (B, N_agents_tok)
        agents_chunk_pos: torch.Tensor,  # (B, N_agents_tok, 9)
        encoding_static: torch.Tensor,  # (B, N_static, H)
        static_mask: torch.Tensor,  # (B, N_static)
        static_pos: torch.Tensor,  # (B, N_static, 9)
        encoding_lanes: torch.Tensor,  # (B, N_lanes, H)
        lanes_mask: torch.Tensor,  # (B, N_lanes)
        lane_pos: torch.Tensor,  # (B, N_lanes, 9)
        encoding_road_safety: torch.Tensor,  # (B, N_road_safety, H)
        road_safety_mask: torch.Tensor,  # (B, N_road_safety)
        road_safety_pos: torch.Tensor,  # (B, N_road_safety, 9)
        B: int,
    ) -> Tuple[
            torch.Tensor,  # encoding_input_with_pos: (B, token_num, H)
            torch.Tensor,  # encoding_mask_2d:        (B, token_num)
            torch.
            Tensor,  # encoding_lanes_with_pos: (B, N_lanes, H)  (route-lane용: 전체 lane)
    ]:
        """agents/static/lanes/road_safety 토큰을 한 줄로 이어 붙이고, 위치 임베딩을 더합니다.

        변경점(핵심)
        ----------
        - FusionEncoder에는 lane 토큰 전체(N_lanes개)를 넣지 않고,
          lane_summary_num개(M개)의 요약 토큰을 만들어 넣을 수 있습니다.
        - route-lane 인코더(NearAgentsRouteLaneEncoder)는 기존처럼
          "전체 lane 토큰(원본 N_lanes개)"을 그대로 사용합니다.
          (agent_route_lane_order가 lane 인덱스를 그대로 쓰기 때문)

        Args:
            encoding_lanes / lanes_mask / lane_pos:
                - 전체 lane 토큰(원본) 정보입니다.
                - Fusion 입력에서는 필요하면 요약 토큰으로 바꿉니다.

        Returns:
            encoding_input_with_pos:
                (B, token_num, H)  FusionEncoder로 들어갈 토큰(요약 lane 포함 가능)
            encoding_mask_2d:
                (B, token_num)     FusionEncoder용 마스크(True=pad)
            encoding_lanes_with_pos:
                (B, N_lanes, H)    route-lane용 "전체 lane 토큰 + pos"
        """
        H: int = int(self.hidden_dim)

        # ------------------------------------------------------------------
        # (1) route-lane용 "전체 lane 토큰 + pos" 준비
        #     - lane_summary를 쓰는 경우에만 별도로 계산이 필요합니다.
        #     - lane_summary를 안 쓰면, 아래에서 Fusion 입력을 만들 때 lane slice로 얻습니다.
        # ------------------------------------------------------------------
        use_lane_summary: bool = (self.lane_summary_pooler is not None)

        if use_lane_summary:
            encoding_lanes_with_pos: torch.Tensor = self._add_pos_embedding_to_tokens(
                token_embeddings=encoding_lanes,  # (B,N_lanes,H)
                token_pos=lane_pos,  # (B,N_lanes,9)
                token_mask=lanes_mask,  # (B,N_lanes)
            )  # (B, N_lanes, H)
        else:
            encoding_lanes_with_pos = None  # 아래에서 slice로 채움

        # ------------------------------------------------------------------
        # (2) Fusion에 넣을 lane 토큰 결정
        #     - use_lane_summary=True: (B,M,H) 요약 토큰
        #     - use_lane_summary=False: (B,N_lanes,H) 원본 lane 토큰
        # ------------------------------------------------------------------
        if use_lane_summary:
            (
                encoding_lanes_for_fusion,  # (B, M, H)
                lanes_mask_for_fusion,  # (B, M)
                lane_pos_for_fusion,  # (B, M, 9)
            ) = self.lane_summary_pooler(
                lane_embeddings=encoding_lanes,
                lane_mask=lanes_mask,
                lane_pos=lane_pos,
            )
        else:
            encoding_lanes_for_fusion = encoding_lanes
            lanes_mask_for_fusion = lanes_mask
            lane_pos_for_fusion = lane_pos

        # ------------------------------------------------------------------
        # (3) 토큰/마스크/포지션 concat (Fusion용)
        # ------------------------------------------------------------------
        encoding_input: torch.Tensor = torch.cat(
            [
                encoding_agents_chunk, encoding_static,
                encoding_lanes_for_fusion, encoding_road_safety
            ],
            dim=1,
        )  # (B, token_num, H)

        encoding_mask_2d: torch.Tensor = torch.cat(
            [
                agents_chunk_mask, static_mask, lanes_mask_for_fusion,
                road_safety_mask
            ],
            dim=1,
        )  # (B, token_num)

        encoding_pos_2d: torch.Tensor = torch.cat(
            [
                agents_chunk_pos, static_pos, lane_pos_for_fusion,
                road_safety_pos
            ],
            dim=1,
        )  # (B, token_num, 9)

        # ------------------------------------------------------------------
        # (4) pos_emb 더하기
        # ------------------------------------------------------------------
        encoding_input_with_pos: torch.Tensor = self._add_pos_embedding_to_tokens(
            token_embeddings=encoding_input,  # (B, token_num, H)
            token_pos=encoding_pos_2d,  # (B, token_num, 9)
            token_mask=encoding_mask_2d,  # (B, token_num)
        )

        # ------------------------------------------------------------------
        # (5) lane_summary를 쓰지 않는 경우: 기존처럼 lane 구간을 slice해서 route-lane용으로 사용
        # ------------------------------------------------------------------
        if not use_lane_summary:
            n_agents = int(encoding_agents_chunk.size(1))
            n_static = int(encoding_static.size(1))
            n_lanes = int(encoding_lanes.size(1))

            lane_start = n_agents + n_static
            lane_end = lane_start + n_lanes

            encoding_lanes_with_pos = encoding_input_with_pos[:, lane_start:
                                                              lane_end, :]  # (B, N_lanes, H)

        assert encoding_lanes_with_pos is not None
        return encoding_input_with_pos, encoding_mask_2d, encoding_lanes_with_pos

    def _run_fusion_and_route_encoder(
            self,
            encoding_input_with_pos: torch.Tensor,  # (B, token_num, H)
            encoding_mask_2d: torch.Tensor,  # (B, token_num)
            encoding_lanes_with_pos: torch.Tensor,  # (B, L, H)
            lanes_mask: torch.Tensor,  # (B, L)
            agent_route_lane_order: torch.Tensor,  # (B, Pnn, L)
            ego_fut_global: torch.Tensor,  # (B, H)
    ) -> Dict[str, torch.Tensor]:
        """FusionEncoder + route-lane 인코더까지 실행해 출력 dict를 만든다.

        Args:
            encoding_input_with_pos: (B, token_num, H)  위치 임베딩까지 포함된 입력 토큰.
            encoding_mask_2d:        (B, token_num)     True=pad.
            encoding_lanes_with_pos: (B, L, H)          lane 토큰 부분.
            lanes_mask:              (B, L)             True=pad.
            agent_route_lane_order:  (B, Pnn, L)        각 에이전트별 route lane 순서(-1=없음).
            ego_fut_global:          (B, H)             ego 미래 요약 벡터.

        Returns:
            encoder_outputs:
                - "encoding":                  (B, token_num, H)
                - "encoding_mask":             (B, token_num)
                - "ego_fut_global":            (B, H)
                - "near_agents_route_lane_emb":(B, Pnn, H)
                - "route_known_mask":          (B, Pnn)  True=유효 route 보유 에이전트
        """
        encoder_outputs: Dict[str, torch.Tensor] = {}

        # 1) FusionEncoder 통과
        encoding_tokens, fused_mask = self.fusion(
            encoding_input_with_pos,  # (B, token_num, H)
            encoding_mask_2d,  # (B, token_num)
        )  # encoding_tokens: (B, token_num, H), fused_mask: (B, token_num)

        encoder_outputs["encoding"] = encoding_tokens
        encoder_outputs["encoding_mask"] = fused_mask
        encoder_outputs["ego_fut_global"] = ego_fut_global

        # 2) route-lane 인코딩 (lane 토큰만 사용)
        (near_agents_route_lane_emb,
         route_known_mask) = self._get_near_agents_route_lane_emb(
             encoding_lanes_with_pos,  # (B, L, H)
             lanes_mask,  # (B, L)
             agent_route_lane_order,  # (B, Pnn, L)
         )

        # PRAM 꺼져 있을 때는 route-lane embedding 도 0으로 처리
        if not self.config.use_pram:
            near_agents_route_lane_emb = self._zero_with_touch(
                near_agents_route_lane_emb, self.npc_route_encoder.parameters())
            route_known_mask = torch.zeros_like(route_known_mask).bool()

        encoder_outputs[
            "near_agents_route_lane_emb"] = near_agents_route_lane_emb  # (B, Pnn, H)
        encoder_outputs["route_known_mask"] = route_known_mask  # (B, Pnn)

        return encoder_outputs

    # --------------------------------------------------------------------- #
    # 리팩토링된 Encoder.forward
    # --------------------------------------------------------------------- #

    def forward(self, inputs: Dict[str,
                                   torch.Tensor]) -> Dict[str, torch.Tensor]:
        """인코더 전방 패스(무작위 길이 M만큼 ego 미래를 조건으로 사용하는 버전).

        입력 딕셔너리
            ego_agent_past : (B, time_len, 11) #
            ego_future_gt_3_dim : (B, future_len, 3)
            neighbor_agents_past : (B, agent_num, time_len, 11) #
            lanes : (B, lane_num, lane_len, 12) #
            lanes_speed_limit : (B, lane_num, 1) #
            lanes_has_speed_limit : (B, lane_num, 1) #
            route_lanes : (B, route_num, route_len, 12)
            route_lanes_speed_limit : (B, route_num, 1)
            route_lanes_has_speed_limit : (B, route_num, 1)
            static_objects : (B, static_num, 10) #
            near_future_gt_3_dim: (B, Pnn, future_len, 3)
            planner_future_11_dim: (B, future_len, 11) #
            agent_route_lane_order: (B, agent_num, lane_num)

            near_future_valid: (B, Pnn, future_len) 미래 유효 마스크.
            near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4) 현재+미래 x_T.
            batch_diffusion_time: (B,) diffusion 시간.
            cond_last_pos_norm: (B, Pnn, 4) cond 용 마지막 위치.

        처리 흐름:
            1) 입력 텐서 정리 및 속도 채널(vx, vy) 0 세팅(옵션).
            2) 학습 모드에서는 배치별로 ego 미래를 앞쪽 M 스텝만 남기고 나머지는 0으로 패딩.
            3) 에이전트/정적/차선 인코더를 통과시켜 토큰/마스크/위치 좌표를 만든다.
            4) 세 토큰을 한 줄로 이어 붙이고, 위치 임베딩(pos_emb)을 더해 FusionEncoder에 넣는다.
            5) lane 토큰으로부터 route-lane 요약 벡터를 뽑고, 최종 encoder 출력 dict를 만든다.

        Returns:
            Dict[str, torch.Tensor]:
                - "encoding":                  (B, token_num, hidden_dim)
                - "encoding_mask":             (B, token_num)
                - "ego_fut_global":            (B, hidden_dim)
                - "near_agents_route_lane_emb":(B, Pnn, hidden_dim)
                - "route_known_mask":          (B, Pnn)
        """
        device_type: str = inputs["ego_agent_past"].device.type

        with profile_block(
                "encoder.forward",
                enabled=self.config.profile_feasible,
                device_type=device_type,
        ):
            # ---- (1) 입력 텐서들 준비 ----
            (
                ego_agent_past,  # (B, 1, time_len, 11)
                planner_future_11_dim,  # (B, future_len, 11)
                neighbor_agents_past,  # (B, A, time_len, 11)
                non_near_agents_past,  # (B, non_near_A, time_len, 11)
                static_objects,  # (B, P, D_static)
                lanes,  # (B, L, lane_len, D_lane)
                lanes_speed_limit,  # (B, L, 1)
                lanes_has_speed_limit,  # (B, L, 1)
                agent_route_lane_order,  # (B, Pnn, L)
                lane_type,  # (B, L, 4) or None
                left_line_type,  # (B, L, 13) or None
                right_line_type,  # (B, L, 13) or None
                stop_sign_points,  # (B, stop_sign_num, safety_len, 2) or None
                stop_sign_is_valid,  # (B, stop_sign_num) or None
                crosswalk_points,  #  (B, crosswalk_num, safety_len, 2) or None
                crosswalk_is_valid,  # (B, crosswalk_num) or None
                speed_bump_points,  #  (B, speed_bump_num, safety_len, 2) or None
                speed_bump_is_valid,  # (B, speed_bump_num) or None
                driveway_points,  # (B, driveway_num, safety_len, 2) or None
                driveway_is_valid,  # (B, driveway_num) or None
                road_edge,  # (B, road_edge_num, safety_len, 2) or None
                road_edge_is_valid,  # (B, road_edge_num) or None
                road_edge_type,  # (B, road_edge_num, 3) or None
                B,
                future_len,
            ) = self._prepare_encoder_inputs(inputs)

            # ---- (2) ego_future_trajectory 생성 (훈련/추론 모드에 따라) ----
            ego_future_trajectory: torch.Tensor = self._compute_ego_future_trajectory(
                planner_future_11_dim=planner_future_11_dim,
                B=B,
                future_len=future_len,
                device=planner_future_11_dim.device,
            )  # (B, future_len, 11)

            # ---- (3) agents/static/lanes 인코딩 ----
            (
                encoding_agents_chunk,  # (B, N_agents_tok, H)
                agents_chunk_mask,  # (B, N_agents_tok)
                agents_chunk_pos,  # (B, N_agents_tok, 9)
                ego_fut_global,  # (B, H)
                encoding_static,  # (B, N_static, H)
                static_mask,  # (B, N_static)
                static_pos,  # (B, N_static, 9)
                encoding_lanes,  # (B, N_lanes, H)
                lanes_mask,  # (B, N_lanes)
                lane_pos,  # (B, N_lanes, 9),
                encoding_road_safety,  # (B, N_road_safety, H)
                road_safety_mask,  # (B, N_road_safety)
                road_safety_pos,  # (B, N_road_safety, 9)
            ) = self._encode_agents_static_lanes(
                ego_agent_past=ego_agent_past,
                non_near_agents_past=
                non_near_agents_past,  # (B, non_near_A, time_len, 11)
                ego_future_trajectory=ego_future_trajectory,
                static_objects=static_objects,
                lanes=lanes,
                lanes_speed_limit=lanes_speed_limit,
                lanes_has_speed_limit=lanes_has_speed_limit,
                lane_type=lane_type,
                left_line_type=left_line_type,
                right_line_type=right_line_type,
                stop_sign_points=stop_sign_points,
                stop_sign_is_valid=stop_sign_is_valid,
                crosswalk_points=crosswalk_points,
                crosswalk_is_valid=crosswalk_is_valid,
                speed_bump_points=speed_bump_points,
                speed_bump_is_valid=speed_bump_is_valid,
                driveway_points=driveway_points,
                driveway_is_valid=driveway_is_valid,
                road_edge=road_edge,
                road_edge_is_valid=road_edge_is_valid,
                road_edge_type=road_edge_type,
            )

            # ---- (4) Fusion 입력 토큰 + 위치 임베딩 구성 ----
            (
                encoding_input_with_pos,  # (B, token_num, H)
                encoding_mask_2d,  # (B, token_num)
                encoding_lanes_with_pos,  # (B, N_lanes, H)
            ) = self._build_fusion_inputs(
                encoding_agents_chunk=encoding_agents_chunk,
                agents_chunk_mask=agents_chunk_mask,
                agents_chunk_pos=agents_chunk_pos,
                encoding_static=encoding_static,
                static_mask=static_mask,
                static_pos=static_pos,
                encoding_lanes=encoding_lanes,
                lanes_mask=lanes_mask,
                lane_pos=lane_pos,
                encoding_road_safety=encoding_road_safety,
                road_safety_mask=road_safety_mask,
                road_safety_pos=road_safety_pos,
                B=B,
            )

            # ---- (5) FusionEncoder + route-lane 인코더 실행 ----
            encoder_outputs: Dict[
                str, torch.Tensor] = self._run_fusion_and_route_encoder(
                    encoding_input_with_pos=encoding_input_with_pos,
                    encoding_mask_2d=encoding_mask_2d,
                    encoding_lanes_with_pos=encoding_lanes_with_pos,
                    lanes_mask=lanes_mask,
                    agent_route_lane_order=agent_route_lane_order,
                    ego_fut_global=ego_fut_global,
                )

            return encoder_outputs

    def _get_near_agents_route_lane_emb(
        self,
        encoding_lanes: torch.Tensor,  # (B, lane_num, hidden_dim)
        lanes_mask: torch.Tensor,  # (B, lane_num)  True=pad
        agent_route_lane_order: torch.Tensor,
        # (B, Pnn, lane_num)  -1=not in route
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # agent_route_lane_order: (B, Pnn, lane_num)
        # -1: route가 아님

        route_num: int = int(self.config.route_num)

        (near_route_lanes, near_route_lanes_mask
        ) = self.build_route_lane_tensors_from_order(
            encoding_lanes=encoding_lanes,  # (B, lane_num, H)
            lanes_mask=lanes_mask,  # (B, lane_num)
            agent_route_lane_order=agent_route_lane_order,  # (B, Pnn, lane_num)
            route_num=route_num,  # ★ 핵심: route_num개만 뽑기
        )
        """
            near_route_lanes:       (B, Pnn, route_num, H)
            near_route_lanes_mask:  (B, Pnn, route_num)  # True=pad(무효)

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
            (near_route_lanes,
             near_route_lanes_mask) = self._mask_all_routes_like(
                 near_route_lanes, near_route_lanes_mask, route_keep_mask)

        # (B, Pnn, hidden_dim)
        near_agents_route_lane_emb = self.npc_route_encoder(
            near_route_lanes, near_route_lanes_mask)

        # (B, Pnn) True=해당 에이전트가 유효 route를 가짐
        route_known_mask = (~near_route_lanes_mask).any(
            dim=-1)  # (B, Pnn) True=known
        return near_agents_route_lane_emb, route_known_mask

    @staticmethod
    def build_route_lane_tensors_from_order(
            encoding_lanes: torch.Tensor,  # (B, lane_num, hidden_dim)
            lanes_mask: torch.Tensor,  # (B, lane_num)  True=pad
            agent_route_lane_order: torch.Tensor,
            # (B, Pnn, lane_num)  -1=not in route, 0..=rank
            route_num: int,  # 뽑을 route lane 개수 (고정 출력 길이)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """에이전트별 route 순서(agent_route_lane_order)에서 앞쪽 route lane만 route_num개 뽑습니다.

        목적
        ----
        기존 구현은 lane_num 전체를 정렬/가져온 뒤에 뒤쪽을 마스크로 버렸습니다.
        이 함수는 그 낭비를 없애고, **처음부터 route_num개만 선택**해서
        출력 텐서 크기를 (B, Pnn, route_num, H)로 고정합니다.

        동작 방식(쉽게 설명)
        -------------------
        1) 각 lane마다 "route 상의 순서 값"이 있습니다.
           - 값이 작을수록 더 앞쪽(더 중요한) lane 입니다.
           - route에 없는 lane은 -1 입니다.
           - lane 자체가 pad(True)인 lane도 무효입니다.

        2) route에 없거나(-1) pad인 곳은 아주 큰 값(BIG)으로 바꿔서,
           "앞쪽을 고르는 과정"에서 자동으로 뒤로 밀리게 만듭니다.

        3) 그 다음, 전체 정렬을 하지 않고
           **가장 작은 값(=가장 앞쪽) route_num개만** 뽑습니다.

        4) 뽑힌 lane 인덱스에 대해서만 gather를 수행해
           near_route_lanes를 (B, Pnn, route_num, H)로 만듭니다.

        5) 유효하지 않은 위치(=BIG로 뽑힌 것, 또는 pad lane)는 mask=True로 만들고,
           해당 위치의 값은 0으로 고정합니다.

        Args:
            encoding_lanes (torch.Tensor):
                차선 임베딩.
                shape: (B, lane_num, hidden_dim)

            lanes_mask (torch.Tensor):
                차선 패딩 마스크(True=패딩).
                shape: (B, lane_num)

            agent_route_lane_order (torch.Tensor):
                에이전트별 lane 순위 정보.
                shape: (B, Pnn, lane_num)
                - 값 >= 0 : route 위에 있는 lane, 숫자가 작을수록 더 앞쪽 lane
                - 값  < 0 : route 에 없음

            route_num (int):
                각 에이전트에 대해 뽑을 route lane 개수(고정 길이 출력).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                near_route_lanes:
                    에이전트별 "앞쪽 route lane route_num개" 임베딩.
                    shape: (B, Pnn, route_num, hidden_dim)

                near_route_lanes_mask:
                    위와 같은 위치의 패딩 마스크(True=무효).
                    shape: (B, Pnn, route_num)
        """
        B, lane_num, hidden_dim = encoding_lanes.shape
        Pnn: int = int(agent_route_lane_order.shape[1])
        route_num_int: int = int(route_num)

        # ---- (0) route_num이 0 이하인 경우: 빈 출력 ----
        if route_num_int <= 0:
            near_route_lanes = encoding_lanes.new_zeros(B, Pnn, 0, hidden_dim)
            near_route_lanes_mask = torch.ones((B, Pnn, 0),
                                               dtype=torch.bool,
                                               device=encoding_lanes.device)
            return near_route_lanes, near_route_lanes_mask

        # ---- (1) lane이 아예 없는 극단 상황 방어 ----
        if lane_num == 0:
            near_route_lanes = encoding_lanes.new_zeros(B, Pnn, route_num_int,
                                                        hidden_dim)
            near_route_lanes_mask = torch.ones((B, Pnn, route_num_int),
                                               dtype=torch.bool,
                                               device=encoding_lanes.device)
            return near_route_lanes, near_route_lanes_mask

        # ---- (2) dtype 정리 ----
        # route_lane_order: (B, Pnn, lane_num) [long]
        route_lane_order: torch.Tensor = agent_route_lane_order.to(torch.long)

        # lanes_mask: (B, lane_num) [bool]
        lanes_mask_bool: torch.Tensor = lanes_mask.to(torch.bool)

        # lanes_mask_exp: (B, 1, lane_num) -> (B, Pnn, lane_num)
        lanes_mask_exp: torch.Tensor = lanes_mask_bool.unsqueeze(1).expand(
            B, Pnn, lane_num)

        # valid_route_mask: (B, Pnn, lane_num)
        # route에 포함(+순서>=0) AND lane 자체도 pad가 아님
        valid_route_mask: torch.Tensor = (route_lane_order
                                          >= 0) & (~lanes_mask_exp)

        # BIG: 무효한 lane을 뒤로 밀기 위한 아주 큰 값
        BIG_INT: int = 2**30
        big_value: torch.Tensor = route_lane_order.new_full(
            (), BIG_INT)  # scalar on same device/dtype

        # order_for_sort: (B, Pnn, lane_num)
        # 유효: 실제 순서(0,1,2,...) / 무효: BIG
        order_for_sort: torch.Tensor = torch.where(valid_route_mask,
                                                   route_lane_order, big_value)

        # ---- (3) 전체 정렬 대신 "앞쪽 route_num개"만 선택 ----
        k: int = min(route_num_int, lane_num)

        # selected_vals: (B, Pnn, k)
        # selected_idx : (B, Pnn, k)  lane 인덱스
        selected_vals, selected_idx = torch.topk(
            order_for_sort,
            k=k,
            dim=-1,
            largest=False,  # 작은 값이 앞쪽
            sorted=True,  # 앞쪽부터 정렬된 상태로 반환
        )

        # route_lane_valid_mask_k: (B, Pnn, k)  True=유효(=BIG가 아님)
        route_lane_valid_mask_k: torch.Tensor = selected_vals != big_value

        # ---- (4) encoding_lanes에서 선택된 k개만 gather ----
        # encoding_lanes_expand: (B, 1, lane_num, H) -> (B, Pnn, lane_num, H)
        encoding_lanes_expand: torch.Tensor = encoding_lanes.unsqueeze(
            1).expand(B, Pnn, lane_num, hidden_dim)

        # gather_idx_H: (B, Pnn, k, H)
        gather_idx_H: torch.Tensor = selected_idx.unsqueeze(-1).expand(
            B, Pnn, k, hidden_dim)

        # near_route_lanes_k: (B, Pnn, k, H)
        near_route_lanes_k: torch.Tensor = torch.gather(encoding_lanes_expand,
                                                        2, gather_idx_H)

        # ---- (5) lane 마스크도 같은 인덱스로 gather ----
        # lanes_mask_expand: (B, 1, lane_num) -> (B, Pnn, lane_num)
        lanes_mask_expand: torch.Tensor = lanes_mask_bool.unsqueeze(1).expand(
            B, Pnn, lane_num)

        # gathered_lane_mask_k: (B, Pnn, k)
        gathered_lane_mask_k: torch.Tensor = torch.gather(
            lanes_mask_expand, 2, selected_idx)

        # 최종 마스크(선택된 k개에 대해):
        # - 원래 lane이 pad였거나
        # - route에 포함되지 않았던 lane(BIG로 들어온 것)은 True(무효)
        near_route_lanes_mask_k: torch.Tensor = gathered_lane_mask_k | (
            ~route_lane_valid_mask_k)

        # ---- (6) route_num이 lane_num보다 큰 경우를 대비한 패딩(고정 길이 유지) ----
        if k < route_num_int:
            pad_len: int = route_num_int - k

            # pad_lanes: (B, Pnn, pad_len, H) = 0
            pad_lanes: torch.Tensor = near_route_lanes_k.new_zeros(
                B, Pnn, pad_len, hidden_dim)

            # pad_mask: (B, Pnn, pad_len) = True(전부 무효)
            pad_mask: torch.Tensor = torch.ones((B, Pnn, pad_len),
                                                dtype=torch.bool,
                                                device=encoding_lanes.device)

            # near_route_lanes: (B, Pnn, route_num, H)
            near_route_lanes: torch.Tensor = torch.cat(
                [near_route_lanes_k, pad_lanes], dim=2)

            # near_route_lanes_mask: (B, Pnn, route_num)
            near_route_lanes_mask: torch.Tensor = torch.cat(
                [near_route_lanes_mask_k, pad_mask], dim=2)
        else:
            # near_route_lanes: (B, Pnn, route_num, H) where route_num == k
            near_route_lanes = near_route_lanes_k
            near_route_lanes_mask = near_route_lanes_mask_k

        # ---- (7) 무효 위치 값은 0으로 고정 ----
        near_route_lanes = near_route_lanes.masked_fill(
            near_route_lanes_mask.unsqueeze(-1), 0.0)

        return near_route_lanes, near_route_lanes_mask


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
            (B, agents_num, past_cur_chunk_num, 5),
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
    ) -> torch.Tensor:  # (B, agents_num * past_cur_chunk_num, 9)
        B, agents_num, time_len, _ = agents_past_cur_xyyaw.shape
        past_cur_chunk_num = past_chunk_start_idx.shape[0]
        ##########
        # agents_past_cur_chunks_xyyaw: (B, agents_num, past_cur_chunk_num, 4)
        agents_past_cur_chunks_xyyaw = self._get_agents_past_cur_chunks_xyyaw(
            past_chunk_start_idx, past_chunk_end_idx,
            agents_past_cur_off_p_mask, agents_past_cur_xyyaw)

        # (B, agents_num, past_cur_chunk_num, 5)
        agents_past_cur_chunks_type = self._get_agents_past_cur_chunks_type(
            agents_past_cur_chunks_xyyaw)
        # (B, agents_num, past_cur_chunk_num, 4 + 5)
        agents_past_cur_chunks_feature = torch.cat(
            [agents_past_cur_chunks_xyyaw, agents_past_cur_chunks_type], dim=-1)
        # (B, agents_num * past_cur_chunk_num, 4 + 5)
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
    ) -> torch.Tensor:  # (B, future_chunk_num, 5)
        B, future_chunk_num = ego_future_chunks_feature.shape[:2]
        ego_fut_chunks_type = torch.zeros(
            (B, future_chunk_num, 5),
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
    ) -> torch.Tensor:  # (B, future_chunk_num, 4 +  5)
        B, future_len, _ = ego_future_xyyaw.shape
        # (B, future_chunk_num, 4)
        ego_fut_chunks_xyyaw = self._get_ego_fut_chunks_pose(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw)
        # (B, future_chunk_num, 5)
        ego_fut_chunks_type = self._get_ego_fut_chunks_type(
            ego_fut_chunks_xyyaw)
        # (B, future_chunk_num, 4 + 5)
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

    def forward(self, ego_agent_past: torch.Tensor,
                neighbor_agents_past: torch.Tensor, ego_future):
        '''
        ego_agent_past: (B, 1, time_len, 11)
        neighbor_agents_past: (B, max_agent_num, time_len, 11)
        ego_future: (B, future_len, 11)

        (x, y, cos, sin, vx, vy, w, l, type(3)
        '''
        assert self.future_len == ego_future.shape[1], \
            f"ego_future.shape[1] should be {self.future_len}, but got {ego_future.shape[1]}"
        # (B, agents_num=1+max_agent_num, time_len, D)
        agents_past_current = torch.cat([ego_agent_past, neighbor_agents_past],
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
        channels_mlp_dim = on_all.shape[-1]
        on_agents_past_cur = on_all[:on_agents_past_cur_points_num, :].reshape(
            agents_past_cur_on_num, time_len, channels_mlp_dim)
        # on_ego_future: (ego_future_on_num, future_len, channels_mlp_dim)
        on_ego_future = on_all[on_agents_past_cur_points_num:, :].reshape(
            ego_future_on_num, future_len, channels_mlp_dim)
        """
        token_pre_project = hard split + gated attentional pooling
        """
        # ---------- 8) 균등 분할 경계 ----------

        past_chunk_start_idx, past_chunk_end_idx = self._compute_equal_chunks(
            seq_len=time_len, chunk_num=self.past_cur_chunk_num,
            device=device)  # (past_cur_chunk_num,), (past_cur_chunk_num,)
        #################

        # agents_past_cur_xyyaw: (B, agents_num, time_len, 4)
        # (B, agents_num * past_cur_chunk_num, 9)
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

        # ego_future_chunks_feature: (B, future_chunk_num, 4 + 5)
        ego_future_chunks_feature = self._get_ego_future_chunks_feature(
            fut_chunk_start_idx, fut_chunk_end_idx, ego_future_off_p_mask,
            ego_future_xyyaw)
        #################
        all_chunk_pos_feature = torch.cat(
            [agents_past_cur_chunks_feature, ego_future_chunks_feature], dim=1
        )  # (B, agents_num * past_cur_chunk_num + future_chunk_num, 4 + 5)
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
        # all_chunk_pos_feature : (B, agents_num * past_cur_chunk_num + future_chunk_num, 4  + 5)
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

    def forward(self, static_objects):
        """
        static_objects: B, static_objects_num, D_10 (x, y, cos, sin, w, l, type(4))

        returns:
            static_encoding: (B, static_objects_num, hidden_dim)
            mask_p: (B, static_objects_num)
            static_feature: (B, static_objects_num, 4 + 5)
        """
        B, static_objects_num, _ = static_objects.shape
        static_xyyaw = static_objects[:, :, :4].clone(
        )  # (B, static_objects_num, 4)
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = self._get_static_feature(static_xyyaw)
        # [FIX] 오토캐스트 환경이면 버퍼 dtype을 현재 GPU autocast dtype으로
        out_dtype = (torch.get_autocast_gpu_dtype()
                     if torch.is_autocast_enabled() and static_objects.is_cuda
                     else static_objects.dtype)
        static_encoding = torch.zeros(
            (B * static_objects_num, self._hidden_dim),
            device=static_objects.device,
            dtype=out_dtype)

        mask_p = torch.sum(torch.ne(static_objects[..., :10], 0),
                           dim=-1).to(static_objects.device) == 0

        valid_indices = ~mask_p.reshape(-1)

        if valid_indices.sum().item() > 0:
            static_objects = static_objects.reshape(B * static_objects_num, -1)
            static_objects = static_objects[valid_indices]
            static_objects = self.projection(static_objects)
            static_objects = static_objects.to(
                dtype=static_encoding.dtype)  # [FIX] 좌변 dtype 일치화
            static_encoding[valid_indices] = static_objects
        else:
            # projection 파라미터를 0-스케일로 터치
            touch = static_encoding.new_zeros(())
            for p in self.projection.parameters():
                touch = touch + p.view(-1)[:1].sum()
            static_encoding = static_encoding + touch * 0.0
        hidden_dim = static_encoding.shape[-1]
        static_encoding = static_encoding.reshape(
            B, static_objects_num,
            hidden_dim)  # (B, static_objects_num, hidden_dim)
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
            (B, static_objects_num, 5),
            device=static_xyyaw.device,
            dtype=static_xyyaw.dtype,
        )
        static_type[:, :, 2] = 1.0  # type
        # static_feature: (B, static_objects_num, 4 + 4)
        static_feature = torch.cat([static_xyyaw, static_type], dim=-1)
        assert static_feature.shape == (B, static_objects_num, 9), \
            f"Expected static_feature shape (B, static_objects_num, 9), got {static_feature.shape}"
        return static_feature


class RoadSafetyFusionEncoder(nn.Module):
    """Road-safety(정지표지/횡단보도/과속방지/진출입로/도로경계) 요소를
    작은 개수의 토큰으로 압축해 반환하는 인코더입니다.

    설계 목표
    ----------
    - 점(포인트)들을 그대로 토큰화하지 않고,
      1) 요소(폴리곤/폴리라인) 단위로 먼저 요약(1요소=1벡터),
      2) 전체 요소 집합을 다시 num_seeds개 요약 토큰으로 압축합니다.
    - 카테고리별 토큰 상한을 따로 두지 않습니다.
    - 가까운 것만 K개 선택 같은 휴리스틱을 사용하지 않습니다.
    - is_valid 입력을 그대로 신뢰해 마스크를 만들며,
      "점이 전부 0이면 무조건 invalid" 같은 추가 휴리스틱을 넣지 않습니다.

    반환
    ----------
    - encoding_road_safety: (B, num_seeds, hidden_dim)
    - road_safety_mask:     (B, num_seeds)  True=pad(무효)
    - road_safety_pos:      (B, num_seeds, 9) = [x,y,cos,sin] + type_onehot(5)
      여기서 type_onehot은 마지막 인덱스(4)를 road_safety로 둡니다.
    """

    def __init__(
        self,
        hidden_dim: int = 192,
        num_seeds: int = 8,
        ffn_ratio: float = 2.0,
        attn_drop_p: float = 0.0,
        out_drop_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim: int = int(hidden_dim)
        self.num_seeds: int = int(num_seeds)
        self.attn_drop_p: float = float(attn_drop_p)

        # (cx, cy, dir_x, dir_y, extent_x, extent_y, length, closed_flag)
        self._numeric_in_dim: int = 8

        self.numeric_mlp = Mlp(
            in_features=self._numeric_in_dim,
            hidden_features=max(32, int(self.hidden_dim * float(ffn_ratio))),
            out_features=self.hidden_dim,
            act_layer=nn.GELU,
            drop=0.0,
        )

        # 카테고리(one-hot 5) 임베딩 + 스칼라 게이트(작게 시작)
        self.category_emb = nn.Linear(5, self.hidden_dim, bias=True)
        self.category_scale = nn.Parameter(torch.tensor(0.01))

        # road_edge_type(3) 임베딩 + 스칼라 게이트(작게 시작)
        self.road_edge_type_emb = nn.Linear(3, self.hidden_dim, bias=True)
        self.road_edge_type_scale = nn.Parameter(torch.tensor(0.01))

        # 요소 임베딩 정규화
        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # 요약 토큰(학습되는 쿼리)
        self.seeds = nn.Parameter(
            torch.zeros(1, self.num_seeds, self.hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)

        # 요약 토큰 쪽 잔차 FFN(게이트 0 시작)
        ffn_hidden = max(32, int(self.hidden_dim * float(ffn_ratio)))
        self.seed_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, self.hidden_dim, bias=True),
        )
        self.seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # DeepSets(mean/max) 잔차 게이트(0 시작)
        self.ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        # 출력 정규화/드롭
        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_drop = nn.Dropout(
            out_drop_p) if out_drop_p > 0 else nn.Identity()

    # ------------------------- 유틸 함수들 ------------------------- #

    @staticmethod
    def _infer_batch_size(tensors: List[Optional[torch.Tensor]]) -> int:
        """입력 텐서들 중 하나에서 배치 크기(B)를 추정합니다.

        Args:
            tensors: None일 수도 있는 텐서들의 리스트.

        Returns:
            int: 배치 크기 B.
        """
        for t in tensors:
            if t is not None:
                return int(t.shape[0])
        raise ValueError("RoadSafetyFusionEncoder: 배치 크기(B)를 추정할 수 없습니다.")

    @staticmethod
    def _get_output_dtype(ref: torch.Tensor) -> torch.dtype:
        """AMP 환경이면 autocast dtype을, 아니면 ref dtype을 사용합니다."""
        if torch.is_autocast_enabled() and ref.is_cuda:
            try:
                return torch.get_autocast_gpu_dtype()
            except Exception:
                pass
        return ref.dtype

    def _touch_self_parameters(self, ref: torch.Tensor) -> torch.Tensor:
        """ref 텐서에 self 파라미터들을 0계수로 연결해 unused param 문제를 줄입니다.

        Args:
            ref: 기준 텐서. (shape 무관)

        Returns:
            ref와 같은 shape의 텐서(값은 같고, 그래프만 살짝 연결됨)
        """
        touch = ref.new_zeros(())
        for p in self.parameters():
            touch = touch + p.view(-1)[:1].sum().to(ref.dtype)
        return ref + touch * 0.0

    @staticmethod
    def _build_element_mask(
            points: torch.Tensor,  # (B, N, S, 2)
            is_valid: Optional[torch.Tensor],  # (B, N) or None
    ) -> torch.Tensor:
        """요소 단위 패딩 마스크(True=무효)를 만듭니다.

        주의:
            - is_valid가 주어지면 그것만 신뢰합니다.
            - is_valid가 None이면 "추가 휴리스틱"을 쓰지 않고, 전부 유효로 간주합니다.

        Args:
            points: (B, N, S, 2)
            is_valid: (B, N) 또는 None

        Returns:
            (B, N) bool 텐서. True=pad(무효)
        """
        B, N = int(points.shape[0]), int(points.shape[1])
        if is_valid is None:
            return torch.zeros((B, N), device=points.device, dtype=torch.bool)
        return (~is_valid.to(torch.bool)).reshape(B, N)

    @staticmethod
    def _principal_axis_dir(points_xy_f32: torch.Tensor,
                            eps: float = 1e-6) -> torch.Tensor:
        """점 구름이 '가장 길게 늘어난 방향'을 (cos, sin)으로 반환합니다.

        Args:
            points_xy_f32: (M, S, 2) float32
            eps: 수치 안정용 작은 값

        Returns:
            (M, 2) float32, [cos, sin]
        """
        mean = points_xy_f32.mean(dim=1)  # (M, 2)
        centered = points_xy_f32 - mean.unsqueeze(1)  # (M, S, 2)
        x = centered[..., 0]  # (M, S)
        y = centered[..., 1]  # (M, S)

        cov_xx = (x * x).mean(dim=1)  # (M,)
        cov_yy = (y * y).mean(dim=1)  # (M,)
        cov_xy = (x * y).mean(dim=1)  # (M,)

        theta = 0.5 * torch.atan2(2.0 * cov_xy, cov_xx - cov_yy + eps)  # (M,)
        return torch.stack(
            [torch.cos(theta), torch.sin(theta)], dim=-1)  # (M, 2)

    @staticmethod
    def _compute_geometry_features(
        points_valid: torch.Tensor,  # (M, S, 2)
        closed: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """요소(폴리곤/폴리라인)의 점 집합에서 간단한 기하 요약을 만듭니다.

        Args:
            points_valid: (M, S, 2)  유효 요소만 모은 점들
            closed: True면 폴리곤(닫힘), False면 폴리라인(열림)

        Returns:
            numeric_feat: (M, 8) float32
                [cx, cy, dir_x, dir_y, extent_x, extent_y, length, closed_flag]
            pos4: (M, 4) float32
                [cx, cy, dir_x, dir_y]  (Fusion에서 pos_emb용)
        """
        pts = points_valid.to(torch.float32)  # (M, S, 2)

        center = pts.mean(dim=1)  # (M, 2)
        dir_vec = RoadSafetyFusionEncoder._principal_axis_dir(pts)  # (M, 2)

        min_xy = pts.amin(dim=1)  # (M, 2)
        max_xy = pts.amax(dim=1)  # (M, 2)
        extent = max_xy - min_xy  # (M, 2)

        # 길이: 인접 점 거리 합
        diffs = pts[:, 1:, :] - pts[:, :-1, :]  # (M, S-1, 2)
        seglen = torch.sqrt((diffs * diffs).sum(dim=-1) + 1e-6).sum(
            dim=1, keepdim=True)  # (M, 1)

        # 폴리곤이면 마지막-처음도 더해 "둘레" 쪽으로
        if closed and pts.shape[1] >= 2:
            last = pts[:, -1, :]  # (M, 2)
            first = pts[:, 0, :]  # (M, 2)
            close_len = torch.sqrt(((last - first)**2).sum(dim=-1) +
                                   1e-6).unsqueeze(-1)  # (M, 1)
            seglen = seglen + close_len  # (M, 1)

        closed_flag = torch.full(
            (pts.shape[0], 1),
            1.0 if closed else 0.0,
            device=pts.device,
            dtype=torch.float32,
        )  # (M, 1)

        numeric_feat = torch.cat([center, dir_vec, extent, seglen, closed_flag],
                                 dim=-1)  # (M, 8)
        pos4 = torch.cat([center, dir_vec], dim=-1)  # (M, 4)
        return numeric_feat, pos4

    def _encode_category(
        self,
        points: torch.Tensor,  # (B, N, S, 2)
        is_valid: Optional[torch.Tensor],  # (B, N) or None
        category_index: int,
        closed: bool,
        road_edge_type: Optional[
            torch.Tensor] = None,  # (B, N, 3) road_edge만 사용
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """한 종류의 요소들을 (B, N, hidden_dim) 임베딩으로 바꿉니다.

        Args:
            points: (B, N, S, 2)
            is_valid: (B, N) 또는 None
            category_index: 0..4 (stop/cross/speed/driveway/edge)
            closed: 폴리곤이면 True, 폴리라인이면 False
            road_edge_type: (B, N, 3) 또는 None

        Returns:
            emb:  (B, N, H)
            mask: (B, N) True=pad(무효)
            pos4: (B, N, 4) [x,y,cos,sin]
        """
        B, N, S, _ = points.shape
        device = points.device
        out_dtype = self._get_output_dtype(points)

        element_mask = self._build_element_mask(points,
                                                is_valid)  # (B, N) True=pad
        valid_flat = (~element_mask).reshape(B * N)  # (B*N,)
        num_valid: int = int(valid_flat.sum().item())

        emb_flat = torch.zeros((B * N, self.hidden_dim),
                               device=device,
                               dtype=out_dtype)  # (B*N, H)
        pos4_flat = torch.zeros((B * N, 4), device=device,
                                dtype=out_dtype)  # (B*N, 4)

        if num_valid > 0:
            points_flat = points.reshape(B * N, S, 2)  # (B*N, S, 2)
            points_valid = points_flat[valid_flat]  # (M, S, 2)

            numeric_feat_f32, pos4_f32 = self._compute_geometry_features(
                points_valid, closed=closed)  # (M,8),(M,4)
            numeric_feat = numeric_feat_f32.to(device=device,
                                               dtype=out_dtype)  # (M,8)

            emb_valid = self.numeric_mlp(numeric_feat)  # (M, H)

            # 카테고리 임베딩(게이트로 작게 시작)
            onehot = torch.zeros((1, 5), device=device,
                                 dtype=emb_valid.dtype)  # (1,5)
            onehot[0, category_index] = 1.0
            cat_emb = self.category_emb(onehot).expand(num_valid, -1)  # (M, H)
            emb_valid = emb_valid + self.category_scale.to(
                emb_valid.dtype) * cat_emb

            # road_edge_type 임베딩(road_edge에서만)
            if road_edge_type is not None:
                if road_edge_type.shape[:2] != (
                        B, N) or road_edge_type.shape[-1] != 3:
                    raise ValueError(
                        f"road_edge_type must be (B, N, 3). got {tuple(road_edge_type.shape)}"
                    )
                edge_type_valid = road_edge_type.reshape(
                    B * N, 3)[valid_flat].to(device=device,
                                             dtype=emb_valid.dtype)  # (M,3)
                emb_valid = emb_valid + self.road_edge_type_scale.to(
                    emb_valid.dtype) * self.road_edge_type_emb(edge_type_valid)

            emb_valid = self.in_norm(emb_valid)  # (M, H)

            emb_flat[valid_flat] = emb_valid.to(out_dtype)
            pos4_flat[valid_flat] = pos4_f32.to(device=device, dtype=out_dtype)

        emb = emb_flat.reshape(B, N, self.hidden_dim)  # (B, N, H)
        pos4 = pos4_flat.reshape(B, N, 4)  # (B, N, 4)

        # pad 위치는 0 고정
        emb = emb.masked_fill(element_mask.unsqueeze(-1), 0.0)
        pos4 = pos4.masked_fill(element_mask.unsqueeze(-1), 0.0)

        return emb, element_mask, pos4

    @staticmethod
    def _concat_parts(
        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """카테고리별 결과를 dim=1(요소 축)으로 합칩니다."""
        embs = [p[0] for p in parts]
        masks = [p[1] for p in parts]
        pos4s = [p[2] for p in parts]
        return torch.cat(embs, dim=1), torch.cat(masks, dim=1), torch.cat(pos4s,
                                                                          dim=1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x:(B,N,H), mask:(B,N) True=pad -> (B,H)"""
        valid = (~mask).to(torch.float32)  # (B,N)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x:(B,N,H), mask:(B,N) True=pad -> (B,H), 전부 pad인 배치는 0"""
        very_neg = torch.finfo(
            x.dtype).min if x.dtype in (torch.float16,
                                        torch.bfloat16) else -1e30
        x2 = x.masked_fill(mask.unsqueeze(-1), very_neg)
        mx = x2.amax(dim=1)  # (B,H)
        all_off = mask.all(dim=1)  # (B,)
        return torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)

    @staticmethod
    def _normalize_dir(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """v:(...,2)를 길이 1로 정규화합니다."""
        n = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps)
        return v / n

    def _seed_pool(
        self,
        elements_emb: torch.Tensor,  # (B, N_total, H)
        elements_pos4: torch.Tensor,  # (B, N_total, 4)
        elements_mask: torch.Tensor,  # (B, N_total) True=pad
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """요소 집합을 num_seeds개의 '요약 토큰'으로 압축합니다.

        Returns:
            seed_emb: (B, M, H)
            seed_mask:(B, M) True=pad
            seed_pos4:(B, M, 4)
        """
        B, N_total, H = elements_emb.shape
        M = self.num_seeds
        device = elements_emb.device
        dtype = elements_emb.dtype

        if N_total == 0:
            seed_emb = torch.zeros((B, M, H), device=device, dtype=dtype)
            seed_pos4 = torch.zeros((B, M, 4), device=device, dtype=dtype)
            seed_mask = torch.ones((B, M), device=device, dtype=torch.bool)
            seed_emb = self._touch_self_parameters(seed_emb)
            return seed_emb, seed_mask, seed_pos4

        seeds = self.seeds.to(dtype).expand(B, M, H)  # (B,M,H)
        logits = torch.einsum("bmh,bnh->bmn", seeds, elements_emb) / math.sqrt(
            max(1.0, float(H)))  # (B,M,N)

        if elements_mask.any().item():
            logits = logits.masked_fill(elements_mask.unsqueeze(1),
                                        float("-inf"))

        all_off = elements_mask.all(dim=1)  # (B,)
        if all_off.any().item():
            logits = logits.clone()
            logits[all_off] = 0.0  # softmax NaN 방지

        attn = F.softmax(logits, dim=-1)  # (B,M,N)
        if self.attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop_p)

        seed_emb = torch.einsum("bmn,bnh->bmh", attn, elements_emb)  # (B,M,H)
        seed_emb = seed_emb + self.seed_ffn_alpha.to(dtype) * self.seed_ffn(
            seed_emb)

        # 집합 통계(mean/max)를 요약 토큰에 잔차로 추가 (게이트 0 시작)
        ds_mean = self._masked_mean(elements_emb, elements_mask)  # (B,H)
        ds_max = self._masked_max(elements_emb, elements_mask)  # (B,H)
        seed_emb = seed_emb + self.ds_mean_alpha.to(dtype) * ds_mean.unsqueeze(
            1) + self.ds_max_alpha.to(dtype) * ds_max.unsqueeze(1)

        seed_emb = self.out_drop(self.out_norm(seed_emb))  # (B,M,H)

        # pos4도 같은 attn으로 가중합 (방향은 정규화)
        pos_xy = torch.einsum("bmn,bnd->bmd", attn,
                              elements_pos4[..., :2])  # (B,M,2)
        pos_dir = torch.einsum("bmn,bnd->bmd", attn,
                               elements_pos4[..., 2:])  # (B,M,2)
        pos_dir = self._normalize_dir(pos_dir)
        seed_pos4 = torch.cat([pos_xy, pos_dir], dim=-1)  # (B,M,4)

        # batch마다 유효 요소 수에 따라 앞쪽 K개 seed만 활성화
        num_valid = (~elements_mask).sum(dim=1).to(torch.long)  # (B,)
        active_k = torch.clamp(num_valid, max=M)  # (B,)
        rank = torch.arange(M, device=device).unsqueeze(0)  # (1,M)
        seed_active = rank < active_k.unsqueeze(1)  # (B,M)
        seed_mask = ~seed_active  # True=pad

        seed_emb = seed_emb.masked_fill(seed_mask.unsqueeze(-1), 0.0)
        seed_pos4 = seed_pos4.masked_fill(seed_mask.unsqueeze(-1), 0.0)
        return seed_emb, seed_mask, seed_pos4

    @staticmethod
    def _build_pos9_from_pos4(seed_pos4: torch.Tensor) -> torch.Tensor:
        """pos4([x,y,cos,sin])에 type_onehot(5)을 붙여 (B,M,9)을 만듭니다."""
        B, M, _ = seed_pos4.shape
        token_type = torch.zeros((B, M, 5),
                                 device=seed_pos4.device,
                                 dtype=seed_pos4.dtype)
        token_type[:, :, 4] = 1.0  # road_safety
        return torch.cat([seed_pos4, token_type], dim=-1)  # (B,M,9)

    # ------------------------- forward ------------------------- #

    def forward(
            self,
            stop_sign_points: Optional[torch.Tensor],  # (B, Ns, S, 2)
            stop_sign_is_valid: Optional[torch.Tensor],  # (B, Ns)
            crosswalk_points: Optional[torch.Tensor],  # (B, Nc, S, 2)
            crosswalk_is_valid: Optional[torch.Tensor],  # (B, Nc)
            speed_bump_points: Optional[torch.Tensor],  # (B, Nb, S, 2)
            speed_bump_is_valid: Optional[torch.Tensor],  # (B, Nb)
            driveway_points: Optional[torch.Tensor],  # (B, Nd, S, 2)
            driveway_is_valid: Optional[torch.Tensor],  # (B, Nd)
            road_edge: Optional[torch.Tensor],  # (B, Ne, S, 2)
            road_edge_is_valid: Optional[torch.Tensor],  # (B, Ne)
            road_edge_type: Optional[torch.Tensor],  # (B, Ne, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Road-safety 입력을 요약 토큰으로 변환합니다.

        Returns:
            encoding_road_safety: (B, num_seeds, hidden_dim)
            road_safety_mask:     (B, num_seeds)  True=pad
            road_safety_pos:      (B, num_seeds, 9)
        """
        B = self._infer_batch_size([
            stop_sign_points, stop_sign_is_valid, crosswalk_points,
            crosswalk_is_valid, speed_bump_points, speed_bump_is_valid,
            driveway_points, driveway_is_valid, road_edge, road_edge_is_valid,
            road_edge_type
        ])

        parts: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        # 카테고리 인덱스: 0 stop, 1 crosswalk, 2 speed_bump, 3 driveway, 4 road_edge
        if stop_sign_points is not None:
            parts.append(
                self._encode_category(stop_sign_points,
                                      stop_sign_is_valid,
                                      0,
                                      closed=True))
        if crosswalk_points is not None:
            parts.append(
                self._encode_category(crosswalk_points,
                                      crosswalk_is_valid,
                                      1,
                                      closed=True))
        if speed_bump_points is not None:
            parts.append(
                self._encode_category(speed_bump_points,
                                      speed_bump_is_valid,
                                      2,
                                      closed=True))
        if driveway_points is not None:
            parts.append(
                self._encode_category(driveway_points,
                                      driveway_is_valid,
                                      3,
                                      closed=True))
        if road_edge is not None:
            parts.append(
                self._encode_category(road_edge,
                                      road_edge_is_valid,
                                      4,
                                      closed=False,
                                      road_edge_type=road_edge_type))

        if len(parts) == 0:
            raise ValueError(
                "RoadSafetyFusionEncoder: 모든 입력이 None이라 처리할 수 없습니다.")

        elements_emb, elements_mask, elements_pos4 = self._concat_parts(
            parts)  # (B,N,H),(B,N),(B,N,4)
        total_valid = int((~elements_mask).sum().item())

        # 유효 요소가 하나도 없으면: 전부 pad 토큰 반환 + unused param 방지 터치
        if total_valid == 0:
            ref = elements_emb
            seed_emb = torch.zeros((B, self.num_seeds, self.hidden_dim),
                                   device=ref.device,
                                   dtype=ref.dtype)
            seed_mask = torch.ones((B, self.num_seeds),
                                   device=ref.device,
                                   dtype=torch.bool)
            seed_pos4 = torch.zeros((B, self.num_seeds, 4),
                                    device=ref.device,
                                    dtype=ref.dtype)
            seed_emb = self._touch_self_parameters(seed_emb)
            road_safety_pos = self._build_pos9_from_pos4(seed_pos4)
            return seed_emb, seed_mask, road_safety_pos

        seed_emb, seed_mask, seed_pos4 = self._seed_pool(
            elements_emb, elements_pos4, elements_mask)
        road_safety_pos = self._build_pos9_from_pos4(seed_pos4)
        return seed_emb, seed_mask, road_safety_pos


class LaneSummaryTokenPooler(nn.Module):
    """많은 lane 토큰을 작은 개수의 '요약 토큰'으로 압축하는 모듈입니다.

    목적
    ----
    FusionEncoder는 토큰 수가 많아질수록 계산량이 매우 크게 늘어납니다.
    lane 토큰은 개수가 큰 편이라 Fusion 전체 비용을 크게 만드는 원인이 될 수 있습니다.

    이 모듈은 lane 토큰(L개)을 직접 Fusion에 넣는 대신,
    M개(num_seeds)의 요약 토큰으로 압축해서 Fusion에 넣을 수 있게 해줍니다.

    입력 / 출력 모양
    --------------
    입력:
        lane_embeddings: (B, L, H)
        lane_mask:       (B, L)     True=패딩(무효)
        lane_pos:        (B, L, 9)  [x,y,cos,sin] + type_onehot(5)

    출력:
        summary_embeddings: (B, M, H)
        summary_mask:       (B, M)     True=패딩(무효)
        summary_pos:        (B, M, 9)

    동작(쉽게 설명)
    --------------
    - 학습되는 M개의 "요약 기준 벡터"가 있고,
      각 요약 기준 벡터가 L개의 lane 토큰을 얼마나 참고할지 가중치를 만든 뒤,
      가중치로 lane 토큰들을 섞어서 요약 토큰을 만듭니다.
    - pos(위치/방향)도 같은 가중치로 함께 섞어 만들어 줍니다.
    - lane이 너무 적으면(유효 lane 수 < M) 뒤쪽 요약 토큰은 패딩 처리합니다.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_seeds: int,
        ffn_ratio: float = 2.0,
        attn_drop_p: float = 0.0,
        out_drop_p: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim: int = int(hidden_dim)
        self.num_seeds: int = int(num_seeds)
        self.attn_drop_p: float = float(attn_drop_p)

        self.in_norm = nn.LayerNorm(self.hidden_dim)

        # (1, M, H) 학습되는 요약 기준 벡터
        self.seeds = nn.Parameter(
            torch.zeros(1, self.num_seeds, self.hidden_dim))
        nn.init.trunc_normal_(self.seeds, std=0.02)

        # 요약 토큰에 붙는 작은 FFN (게이트 0에서 시작)
        ffn_hidden = max(32, int(self.hidden_dim * float(ffn_ratio)))
        self.seed_ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, ffn_hidden, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_hidden, self.hidden_dim, bias=True),
        )
        self.seed_ffn_alpha = nn.Parameter(torch.tensor(0.0))

        # 집합 통계(mean/max) 잔차 게이트(0에서 시작)
        self.ds_mean_alpha = nn.Parameter(torch.tensor(0.0))
        self.ds_max_alpha = nn.Parameter(torch.tensor(0.0))

        self.out_norm = nn.LayerNorm(self.hidden_dim)
        self.out_drop = nn.Dropout(
            out_drop_p) if out_drop_p > 0 else nn.Identity()

    @staticmethod
    def _normalize_dir(v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """방향 벡터를 길이 1로 정규화합니다.

        Args:
            v: (..., 2)
            eps: 0으로 나누기 방지용 작은 값

        Returns:
            (..., 2)  길이 1로 정규화된 방향 벡터
        """
        n = torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps)
        return v / n

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """패딩을 제외한 평균을 계산합니다.

        Args:
            x: (B, L, H)
            mask: (B, L) True=패딩(무효)

        Returns:
            (B, H)
        """
        valid = (~mask).to(torch.float32)  # (B,L)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom  # (B,H)

    @staticmethod
    def _masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """패딩을 제외한 최댓값을 계산합니다. 전부 패딩이면 0을 반환합니다.

        Args:
            x: (B, L, H)
            mask: (B, L) True=패딩(무효)

        Returns:
            (B, H)
        """
        very_neg = (torch.finfo(x.dtype).min
                    if x.dtype in (torch.float16, torch.bfloat16) else -1e30)
        x2 = x.masked_fill(mask.unsqueeze(-1), very_neg)  # (B,L,H)
        mx = x2.amax(dim=1)  # (B,H)
        all_off = mask.all(dim=1)  # (B,)
        return torch.where(all_off.unsqueeze(-1), torch.zeros_like(mx), mx)

    def _touch_self_parameters(self, ref: torch.Tensor) -> torch.Tensor:
        """값은 그대로 두고, 파라미터들을 0계수로 살짝 연결합니다(학습 안정용).

        Args:
            ref: 기준 텐서 (shape 무관)

        Returns:
            ref와 같은 shape 텐서(값은 같고 그래프만 연결됨)
        """
        touch = ref.new_zeros(())
        for p in self.parameters():
            touch = touch + p.view(-1)[:1].sum().to(ref.dtype)
        return ref + touch * 0.0

    def forward(
            self,
            lane_embeddings: torch.Tensor,  # (B, L, H)
            lane_mask: torch.Tensor,  # (B, L) True=pad
            lane_pos: torch.Tensor,  # (B, L, 9)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """lane 토큰을 M개 요약 토큰으로 압축합니다.

        Args:
            lane_embeddings:
                shape: (B, L, H)
            lane_mask:
                shape: (B, L)  True=패딩(무효)
            lane_pos:
                shape: (B, L, 9)
                [x,y,cos,sin] + type_onehot(5)

        Returns:
            summary_embeddings:
                shape: (B, M, H)
            summary_mask:
                shape: (B, M)  True=패딩(무효)
            summary_pos:
                shape: (B, M, 9)
        """
        if lane_mask.dtype != torch.bool:
            lane_mask = lane_mask.to(torch.bool)

        B, L, H = lane_embeddings.shape
        if H != self.hidden_dim:
            raise ValueError(
                f"hidden_dim mismatch. got {H}, expected {self.hidden_dim}")
        if lane_pos.shape != (B, L, 9):
            raise ValueError(
                f"lane_pos must be (B, L, 9). got {tuple(lane_pos.shape)}")

        M: int = int(self.num_seeds)
        device = lane_embeddings.device
        dtype = lane_embeddings.dtype

        # (A) M==0이면 빈 출력
        if M <= 0:
            empty_emb = lane_embeddings.new_zeros(B, 0, H)  # (B,0,H)
            empty_mask = torch.ones((B, 0), device=device,
                                    dtype=torch.bool)  # (B,0)
            empty_pos = lane_embeddings.new_zeros(B, 0, 9)  # (B,0,9)
            return empty_emb, empty_mask, empty_pos

        # (B) L==0이면 요약 불가 → 전부 패딩
        if L == 0:
            summary_embeddings = lane_embeddings.new_zeros(B, M, H)  # (B,M,H)
            summary_mask = torch.ones((B, M), device=device,
                                      dtype=torch.bool)  # (B,M)
            summary_pos = lane_embeddings.new_zeros(B, M, 9)  # (B,M,9)
            summary_embeddings = self._touch_self_parameters(summary_embeddings)
            return summary_embeddings, summary_mask, summary_pos

        # (C) 입력 정리: 패딩은 0으로 고정하고, 정규화 후 다시 0으로 고정
        lane_embeddings = lane_embeddings.masked_fill(lane_mask.unsqueeze(-1),
                                                      0.0)  # (B,L,H)
        lane_embeddings = self.in_norm(lane_embeddings)  # (B,L,H)
        lane_embeddings = lane_embeddings.masked_fill(lane_mask.unsqueeze(-1),
                                                      0.0)  # (B,L,H)

        # (D) 요약 가중치 계산: (B,M,L)
        seeds = self.seeds.to(dtype).expand(B, M, H)  # (B,M,H)
        logits = torch.einsum("bmh,blh->bml", seeds.float(),
                              lane_embeddings.float())
        logits = logits / math.sqrt(max(1.0, float(H)))  # (B,M,L)

        if lane_mask.any().item():
            logits = logits.masked_fill(lane_mask.unsqueeze(1), float("-inf"))

        # 전부 패딩인 배치 softmax NaN 방지
        all_off = lane_mask.all(dim=1)  # (B,)
        if all_off.any().item():
            logits = logits.clone()
            logits[all_off] = 0.0

        attn = F.softmax(logits, dim=-1)  # (B,M,L) float32
        if self.attn_drop_p > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_drop_p)

        # (E) 요약 토큰 생성: (B,M,H)
        summary_embeddings = torch.einsum("bml,blh->bmh", attn,
                                          lane_embeddings.float()).to(
                                              dtype)  # (B,M,H)

        summary_embeddings = summary_embeddings + self.seed_ffn_alpha.to(
            dtype) * self.seed_ffn(summary_embeddings)

        # 집합 통계 잔차(게이트 0에서 시작)
        ds_mean = self._masked_mean(lane_embeddings,
                                    lane_mask).to(dtype)  # (B,H)
        ds_max = self._masked_max(lane_embeddings, lane_mask).to(dtype)  # (B,H)
        summary_embeddings = summary_embeddings \
            + self.ds_mean_alpha.to(dtype) * ds_mean.unsqueeze(1) \
            + self.ds_max_alpha.to(dtype) * ds_max.unsqueeze(1)  # (B,M,H)

        summary_embeddings = self.out_drop(
            self.out_norm(summary_embeddings))  # (B,M,H)

        # (F) pos(대표 위치/방향)도 같은 가중치로 계산
        lane_pos4 = lane_pos[..., :4].to(torch.float32)  # (B,L,4)
        pos_xy = torch.einsum("bml,bld->bmd", attn,
                              lane_pos4[..., :2])  # (B,M,2)
        pos_dir = torch.einsum("bml,bld->bmd", attn, lane_pos4[...,
                                                               2:4])  # (B,M,2)
        pos_dir = self._normalize_dir(pos_dir)  # (B,M,2)
        pos4 = torch.cat([pos_xy, pos_dir], dim=-1).to(dtype)  # (B,M,4)

        # type_onehot(5)에서 lane 인덱스(3)만 1로
        token_type = torch.zeros((B, M, 5), device=device,
                                 dtype=dtype)  # (B,M,5)
        token_type[:, :, 3] = 1.0
        summary_pos = torch.cat([pos4, token_type], dim=-1)  # (B,M,9)

        # (G) 유효 lane 수가 적으면 뒤쪽 seed는 패딩 처리
        num_valid = (~lane_mask).sum(dim=1).to(torch.long)  # (B,)
        active_k = torch.clamp(num_valid, max=M)  # (B,)
        rank = torch.arange(M, device=device).unsqueeze(0)  # (1,M)
        seed_active = rank < active_k.unsqueeze(1)  # (B,M)
        summary_mask = ~seed_active  # (B,M) True=pad

        summary_embeddings = summary_embeddings.masked_fill(
            summary_mask.unsqueeze(-1), 0.0)
        summary_pos = summary_pos.masked_fill(summary_mask.unsqueeze(-1), 0.0)

        return summary_embeddings, summary_mask, summary_pos


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
        self.lane_type_emb = nn.Linear(4, channels_mlp_dim)
        nn.init.normal_(self.lane_type_emb.weight, std=0.02)
        self.left_line_type_emb = nn.Linear(13, channels_mlp_dim)
        nn.init.normal_(self.left_line_type_emb.weight, std=0.02)
        self.right_line_type_emb = nn.Linear(13, channels_mlp_dim)
        nn.init.normal_(self.right_line_type_emb.weight, std=0.02)
        # lane_type / line_type 임베딩 스칼라 게이트 (초기 0)
        # - 초반에는 영향이 거의 없고(0에 가까움),
        # - 학습이 "쓸만하다"고 판단하면 자동으로 커지도록 유도
        self.lane_type_alpha = nn.Parameter(torch.tensor(0.0))
        self.left_line_type_alpha = nn.Parameter(torch.tensor(0.0))
        self.right_line_type_alpha = nn.Parameter(torch.tensor(0.0))

        self.channel_pre_project = Mlp(in_features=10,
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

    @staticmethod
    def _apply_scalar_gate_to_embedding(
            embedding: torch.Tensor,  # (N, C)
            gate: torch.Tensor,  # ()
    ) -> torch.Tensor:  # (N, C)
        """임베딩에 '하나의 숫자(스칼라) 게이트'를 곱해 반영 정도를 조절합니다.

        이 함수는 lane_type / left_line_type / right_line_type처럼
        데이터에 따라 비어 있거나(정보가 없음), 노이즈가 있을 수 있는 입력의 영향을
        **학습이 알아서 켜고/끄도록** 만드는 용도입니다.

        동작:
            - gate는 학습되는 파라미터이며 시작값이 0입니다.
            - 시작값이 0이면 embedding을 더해도 영향이 거의 없어 초반 학습이 안정적입니다.
            - 학습이 진행되면서 필요하다고 판단되면 gate 값이 커져 해당 embedding의 영향이 커집니다.
            - 필요 없으면 gate가 0 근처에 머물러 영향이 계속 작게 유지됩니다.

        Args:
            embedding (torch.Tensor):
                모양: (N, C)
                - N: 유효 lane 개수(num_valid)
                - C: channel 차원(self._channel)
            gate (torch.Tensor):
                모양: () (스칼라)
                - embedding에 곱해질 하나의 숫자입니다.

        Returns:
            torch.Tensor:
                모양: (N, C)
                - gate가 반영된 embedding 입니다.
        """
        gate_value: torch.Tensor = gate.to(dtype=embedding.dtype,
                                           device=embedding.device)  # ()
        return embedding * gate_value  # (N, C)

    @staticmethod
    def _compute_boundary_valid_flags(
            lanes_xy_and_offsets: torch.Tensor,  # (B, lane_num, lane_len, 8)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """좌/우 경계 정보가 있는지 여부를 점 단위로 계산합니다.

        Args:
            lanes_xy_and_offsets (torch.Tensor):
                모양: (B, lane_num, lane_len, 8)
                마지막 채널 의미(예시):
                    0: x
                    1: y
                    2: 진행방향(또는 변화량) x
                    3: 진행방향(또는 변화량) y
                    4: 왼쪽 경계 변화량 x
                    5: 왼쪽 경계 변화량 y
                    6: 오른쪽 경계 변화량 x
                    7: 오른쪽 경계 변화량 y

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - left_is_valid:  (B, lane_num, lane_len) bool
                - right_is_valid: (B, lane_num, lane_len) bool

        Note:
            - 값이 0이면 "해당 경계 정보가 없다"라고 가정하고 True/False를 만듭니다.
        """
        if lanes_xy_and_offsets.dim(
        ) != 4 or lanes_xy_and_offsets.shape[-1] < 8:
            raise ValueError(
                f"lanes_xy_and_offsets must be (B, lane_num, lane_len, 8+). got {tuple(lanes_xy_and_offsets.shape)}"
            )

        # left: (B, lane_num, lane_len)
        left_is_valid: torch.Tensor = (lanes_xy_and_offsets[..., 4:6]
                                       != 0).any(dim=-1)
        # right: (B, lane_num, lane_len)
        right_is_valid: torch.Tensor = (lanes_xy_and_offsets[..., 6:8]
                                        != 0).any(dim=-1)

        return left_is_valid, right_is_valid

    def _embed_optional_lane_attribute(
        self,
        lane_attribute: Optional[torch.Tensor],  # (B, lane_num, F) or None
        embedding_layer: nn.Linear,  # F -> channel
        valid_indices: torch.Tensor,  # (B * lane_num,) bool
        batch_size: int,
        lane_num: int,
        out_dtype: torch.dtype,
        out_device: torch.device,
    ) -> torch.Tensor:
        """lane 속성(one-hot 또는 float 벡터)을 임베딩으로 바꿔서 valid lane에만 맞춰 반환합니다.

        Args:
            lane_attribute (Optional[torch.Tensor]):
                - None 이거나, 모양이 (B, lane_num, F) 인 텐서입니다.
                - None이면 "추가 정보 없음"으로 보고 0 임베딩을 반환합니다.
            embedding_layer (nn.Linear):
                - 입력 F를 channel 차원으로 바꾸는 선형 레이어입니다.
            valid_indices (torch.Tensor):
                - 모양: (B * lane_num,)
                - True인 위치가 "유효 lane"입니다.
            batch_size (int):
                - B
            lane_num (int):
                - lane_num
            out_dtype (torch.dtype):
                - 반환 임베딩의 dtype (보통 lanes.dtype)
            out_device (torch.device):
                - 반환 임베딩의 device (보통 lanes.device)

        Returns:
            torch.Tensor:
                - 모양: (num_valid, channel)
                - num_valid = valid_indices.sum()

        Note:
            - 입력이 None이거나, 유효 lane이 0개인 경우에도 학습이 안전하게 돌아가도록,
              값은 0이지만 레이어 파라미터를 아주 약하게 연결해 둡니다.
        """
        if valid_indices.dtype != torch.bool:
            valid_indices = valid_indices.to(torch.bool)

        num_valid: int = int(valid_indices.sum().item())
        # base: (num_valid, channel)
        base_embedding: torch.Tensor = torch.zeros(
            (num_valid, self._channel),
            device=out_device,
            dtype=out_dtype,
        )

        # 유효 lane이 없거나 입력이 None이면: 0 임베딩 + (레이어 파라미터 살짝 연결)
        if (lane_attribute is None) or (num_valid == 0):
            touch: torch.Tensor = base_embedding.new_zeros(())
            for p in embedding_layer.parameters():
                touch = touch + p.view(-1)[:1].sum().to(out_dtype)
            return base_embedding + touch * 0.0

        # ---- 입력 shape 점검 ----
        if lane_attribute.dim() != 3:
            raise ValueError(
                f"lane_attribute must be (B, lane_num, F). got {tuple(lane_attribute.shape)}"
            )
        if int(lane_attribute.shape[0]) != int(batch_size) or int(
                lane_attribute.shape[1]) != int(lane_num):
            raise ValueError(
                f"lane_attribute shape mismatch. expected (B={batch_size}, lane_num={lane_num}, F), "
                f"got {tuple(lane_attribute.shape)}")

        # ---- flatten 후 valid lane만 선택 ----
        lane_attribute = lane_attribute.to(device=out_device,
                                           dtype=out_dtype)  # (B, lane_num, F)
        lane_attribute_flat: torch.Tensor = lane_attribute.reshape(
            batch_size * lane_num, -1)  # (B*lane_num, F)

        if lane_attribute_flat.shape[0] != valid_indices.shape[0]:
            raise ValueError(
                f"valid_indices length mismatch. valid_indices={valid_indices.shape[0]}, "
                f"lane_attribute_flat={lane_attribute_flat.shape[0]}")

        lane_attribute_valid: torch.Tensor = lane_attribute_flat[
            valid_indices]  # (num_valid, F)
        lane_attribute_embedding: torch.Tensor = embedding_layer(
            lane_attribute_valid)  # (num_valid, channel)
        return lane_attribute_embedding.to(dtype=out_dtype)

    def _get_lane_feature(
        self, lane_xyyaw: torch.Tensor
        # (B, lane_num, 4)
    ) -> torch.Tensor:  # (B, lane_num, 4 + 5)
        B, lane_num, _ = lane_xyyaw.shape
        # lane_type: (B, lane_num, 4)
        lane_type = torch.zeros(
            (B, lane_num, 5),
            device=lane_xyyaw.device,
            dtype=lane_xyyaw.dtype,
        )
        lane_type[:, :, 3] = 1.0  # type
        # static_feature: (B, lane_num, 4 + 5)
        lane_feature = torch.cat([lane_xyyaw, lane_type], dim=-1)
        return lane_feature

    def forward(
        self,
        lanes: torch.Tensor,  # (B, lane_num, lane_len, D_lane)
        lanes_speed_limit: torch.Tensor,  # (B, lane_num, 1)
        lanes_has_speed_limit: torch.Tensor,  # (B, lane_num, 1)
        lane_type: Optional[torch.Tensor],  # (B, lane_num, 4) or None
        left_line_type: Optional[torch.Tensor],  # (B, lane_num, 13) or None
        right_line_type: Optional[torch.Tensor],  # (B, lane_num, 13) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """차선 정보를 lane 단위 임베딩으로 바꿔 반환합니다.

        Args:
            lanes (torch.Tensor):
                모양: (B, lane_num, lane_len, D_lane)
                예시 채널 구성:
                    - 앞 8개 채널: 차선 중심/경계 관련 값
                    - 뒤 4개 채널: 신호등(traffic) 관련 값
            lanes_speed_limit (torch.Tensor):
                모양: (B, lane_num, 1)
            lanes_has_speed_limit (torch.Tensor):
                모양: (B, lane_num, 1)  (0/1 또는 bool)
            lane_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 4) 또는 None
            left_line_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 13) 또는 None
            right_line_type (Optional[torch.Tensor]):
                모양: (B, lane_num, 13) 또는 None

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - lane_embedding: (B, lane_num, hidden_dim)
                - mask_p:         (B, lane_num)  True=해당 lane이 비어있음
                - lane_feature:   (B, lane_num, 8)  위치 임베딩용 [x,y,dir(2), type(4)]
        """
        # traffic: (B, lane_num, 4)
        traffic: torch.Tensor = lanes[:, :, 0, 8:]

        # lanes_8: (B, lane_num, lane_len, 8)
        lanes_8: torch.Tensor = lanes[..., :8]

        # ---- (1) 좌/우 경계 유효 플래그 추가 ----
        left_is_valid, right_is_valid = self._compute_boundary_valid_flags(
            lanes_8)  # (B, lane_num, lane_len), (B, lane_num, lane_len)

        # lanes_10: (B, lane_num, lane_len, 10)
        lanes_10: torch.Tensor = torch.cat(
            [
                lanes_8,
                left_is_valid.unsqueeze(-1).to(lanes_8.dtype),
                right_is_valid.unsqueeze(-1).to(lanes_8.dtype),
            ],
            dim=-1,
        )

        # ---- (2) lane 위치 특징(lane_feature) 생성 ----
        # lane_pos: (B, lane_num, 4)
        lane_pos: torch.Tensor = lanes_10[:, :,
                                          int(self._lane_len / 2), :4].clone()
        lane_feature: torch.Tensor = self._get_lane_feature(
            lane_pos)  # (B, lane_num, 8)

        # ---- (3) lane 유효 마스크 ----
        B, lane_num, lane_len, _ = lanes_10.shape

        # mask_v: (B, lane_num, lane_len)  True=해당 점이 비어있음
        mask_v: torch.Tensor = torch.sum(torch.ne(lanes_10[..., :8], 0),
                                         dim=-1).to(lanes_10.device) == 0
        # mask_p: (B, lane_num) True=해당 lane이 전부 비어있음
        mask_p: torch.Tensor = torch.sum(~mask_v, dim=-1) == 0

        # flatten
        lanes_flat: torch.Tensor = lanes_10.reshape(
            B * lane_num, lane_len, -1)  # (B*lane_num, lane_len, 10)
        valid_indices: torch.Tensor = ~mask_p.reshape(-1)  # (B*lane_num,) bool
        num_valid: int = int(valid_indices.sum().item())

        # ---- (4) 유효 lane이 하나도 없으면 바로 종료(0 반환) ----
        if num_valid == 0:
            H: int = int(self.emb_project.fc2.out_features)

            out_dtype = (torch.get_autocast_gpu_dtype()
                         if torch.is_autocast_enabled() and lanes_flat.is_cuda
                         else lanes_flat.dtype)
            lane_embedding = torch.zeros((B, lane_num, H),
                                         device=lanes_flat.device,
                                         dtype=out_dtype)

            # 여러 장치 학습에서도 안전하게 돌아가도록, 이번 경로에서 쓰이지 않은 레이어도 "살짝 연결"
            touch = lane_embedding.new_zeros(())
            touch_modules = [
                self.channel_pre_project,
                self.token_pre_project,
                self.emb_project,
                self.speed_limit_emb,
                self.unknown_speed_emb,
                self.traffic_emb,
                self.lane_type_emb,
                self.left_line_type_emb,
                self.right_line_type_emb,
                *self.blocks,
            ]
            for mod in touch_modules:
                for p in mod.parameters():
                    touch = touch + p.view(-1)[:1].sum().to(out_dtype)
            touch = touch + self.lane_type_alpha.to(out_dtype)
            touch = touch + self.left_line_type_alpha.to(out_dtype)
            touch = touch + self.right_line_type_alpha.to(out_dtype)
            return lane_embedding + touch * 0.0, mask_p.reshape(
                B, lane_num), lane_feature

        # ---- (5) 유효 lane만 인코딩 ----
        lanes_valid: torch.Tensor = lanes_flat[
            valid_indices]  # (num_valid, lane_len, 10)

        # (num_valid, lane_len, 10) -> (num_valid, lane_len, channel)
        lanes_valid = self.channel_pre_project(lanes_valid)
        lanes_valid = lanes_valid.permute(0, 2,
                                          1)  # (num_valid, channel, lane_len)
        lanes_valid = self.token_pre_project(
            lanes_valid)  # (num_valid, channel, tokens_mlp_dim)
        lanes_valid = lanes_valid.permute(
            0, 2, 1)  # (num_valid, tokens_mlp_dim, channel)

        for block in self.blocks:
            lanes_valid = block(lanes_valid)

        # (num_valid, tokens_mlp_dim, channel) -> (num_valid, channel)
        lanes_valid = lanes_valid.float().mean(dim=1).to(lanes_valid.dtype)

        # ---- (6) speed limit / traffic 임베딩 추가 ----
        lanes_speed_limit_flat: torch.Tensor = lanes_speed_limit.reshape(
            B * lane_num, 1)  # (B*lane_num, 1)
        lanes_has_speed_limit_flat: torch.Tensor = lanes_has_speed_limit.to(
            torch.bool).reshape(B * lane_num, 1)  # (B*lane_num, 1)
        traffic_flat: torch.Tensor = traffic.reshape(B * lane_num,
                                                     -1)  # (B*lane_num, 4)

        lanes_has_speed_limit_valid: torch.Tensor = lanes_has_speed_limit_flat[
            valid_indices].squeeze(-1)  # (num_valid,)
        lanes_speed_limit_valid: torch.Tensor = lanes_speed_limit_flat[
            valid_indices].squeeze(-1)  # (num_valid,)
        traffic_valid: torch.Tensor = traffic_flat[
            valid_indices]  # (num_valid, 4)

        # speed_limit_embedding: (num_valid, channel)
        speed_limit_embedding: torch.Tensor = torch.zeros(
            (num_valid, self._channel),
            device=lanes_valid.device,
            dtype=lanes_valid.dtype,
        )
        if lanes_has_speed_limit_valid.any().item():
            speed_limit_with_limit = self.speed_limit_emb(
                lanes_speed_limit_valid[lanes_has_speed_limit_valid].unsqueeze(
                    -1)).to(lanes_valid.dtype)
            speed_limit_embedding[
                lanes_has_speed_limit_valid] = speed_limit_with_limit

        if (~lanes_has_speed_limit_valid).any().item():
            speed_limit_no_limit = self.unknown_speed_emb.weight.expand(
                int((~lanes_has_speed_limit_valid).sum().item()),
                -1).to(lanes_valid.dtype)
            speed_limit_embedding[
                ~lanes_has_speed_limit_valid] = speed_limit_no_limit

        traffic_light_embedding: torch.Tensor = self.traffic_emb(
            traffic_valid).to(lanes_valid.dtype)  # (num_valid, channel)

        lanes_valid = lanes_valid + speed_limit_embedding + traffic_light_embedding  # (num_valid, channel)

        # ---- (7) (추가) lane_type / left_line_type / right_line_type 임베딩 (None 안전) ----
        lane_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=lane_type,
            embedding_layer=self.lane_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        left_line_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=left_line_type,
            embedding_layer=self.left_line_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        right_line_type_embedding: torch.Tensor = self._embed_optional_lane_attribute(
            lane_attribute=right_line_type,
            embedding_layer=self.right_line_type_emb,
            valid_indices=valid_indices,
            batch_size=B,
            lane_num=lane_num,
            out_dtype=lanes_valid.dtype,
            out_device=lanes_valid.device,
        )  # (num_valid, channel)

        # 변경 (게이트 적용)
        lanes_valid = (
            lanes_valid + self._apply_scalar_gate_to_embedding(
                lane_type_embedding, self.lane_type_alpha) +
            self._apply_scalar_gate_to_embedding(left_line_type_embedding,
                                                 self.left_line_type_alpha) +
            self._apply_scalar_gate_to_embedding(right_line_type_embedding,
                                                 self.right_line_type_alpha))
        # ---- (8) 최종 hidden_dim으로 투영 ----
        lanes_valid = self.emb_project(
            self.norm(lanes_valid))  # (num_valid, hidden_dim)

        # ---- (9) 원래 (B, lane_num) 자리로 복원 ----
        lane_embedding_flat: torch.Tensor = torch.zeros(
            (B * lane_num, lanes_valid.shape[-1]),
            device=lanes_valid.device,
            dtype=lanes_valid.dtype,
        )
        lane_embedding_flat[valid_indices] = lanes_valid

        lane_embedding: torch.Tensor = lane_embedding_flat.reshape(
            B, lane_num, -1)  # (B, lane_num, hidden_dim)
        return lane_embedding, mask_p.reshape(B, lane_num), lane_feature


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
