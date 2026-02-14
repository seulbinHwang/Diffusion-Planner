import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union, TypedDict, Optional, Callable
from scipy.signal import savgol_filter  # type: ignore
import math
import numpy as np
import torch.nn.functional as F
# feasible.py 내부, FeasibleProjector 클래스에 추가/대체 ----------------------
import torch
from typing import Tuple

Number = Union[float, int]
ArrayLike = Union[np.ndarray, List[Number], Tuple[Number, ...]]
import torch
from torch import Tensor

torch.set_printoptions(sci_mode=False, precision=6)

from typing import Tuple
# feasible.py 최상단 import 근처
from collections import OrderedDict
from typing import Dict, Tuple, Optional, Any

# <추가하자>
from typing import NamedTuple
import time
from contextlib import contextmanager
from typing import Dict, Iterator

import torch

# name -> 호출 횟수 / 평균 계산에 포함된 횟수 / 평균 누적(ms)
_PROFILE_TOTAL_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_CALL_COUNT: Dict[str, int] = {}
_PROFILE_AVG_TOTAL_MS: Dict[str, float] = {}

# 각 name별로 처음 N번 호출은 avg 계산에서 제외
_PROFILE_WARMUP_CALLS: int = 30


@contextmanager
def profile_block(
    name: str,
    enabled: bool = True,
    device_type: str = "cuda",
) -> Iterator[None]:
    """코드 블록 실행 시간을 ms 단위로 출력하고, 평균(avg)은 워밍업 이후만 누적합니다.

    동작:
        - name(문자열)별로 호출 횟수를 셉니다.
        - 각 name에서 처음 _PROFILE_WARMUP_CALLS번 호출은 avg 누적에서 제외합니다.
        - 그 다음 호출부터 avg_ms = (워밍업 제외 누적 시간) / (워밍업 제외 호출 횟수) 로 출력합니다.

    Args:
        name (str): 출력/누적의 키로 쓸 블록 이름.
        enabled (bool): False이면 계측 없이 그대로 실행합니다.
        device_type (str): "cuda"면 GPU 작업까지 포함해 재기 위해 앞/뒤로 synchronize 합니다.

    Yields:
        None
    """
    if not enabled:
        yield
        return

    is_cuda: bool = isinstance(device_type, str) and device_type.startswith("cuda")
    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    t0: float = time.perf_counter()
    yield

    if is_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed_ms: float = (time.perf_counter() - t0) * 1000.0

    # (1) 전체 호출 카운트(워밍업 포함)
    total_prev: int = _PROFILE_TOTAL_CALL_COUNT.get(name, 0)
    total_now: int = total_prev + 1
    _PROFILE_TOTAL_CALL_COUNT[name] = total_now

    # (2) avg 누적(워밍업 제외)
    avg_cnt_prev: int = _PROFILE_AVG_CALL_COUNT.get(name, 0)
    avg_sum_prev: float = _PROFILE_AVG_TOTAL_MS.get(name, 0.0)

    if total_now > int(_PROFILE_WARMUP_CALLS):
        avg_cnt_now: int = avg_cnt_prev + 1
        avg_sum_now: float = avg_sum_prev + float(elapsed_ms)
        _PROFILE_AVG_CALL_COUNT[name] = avg_cnt_now
        _PROFILE_AVG_TOTAL_MS[name] = avg_sum_now
    else:
        avg_cnt_now = avg_cnt_prev
        avg_sum_now = avg_sum_prev

    avg_ms: float = (avg_sum_now / float(avg_cnt_now)) if avg_cnt_now > 0 else 0.0
    warmup_left: int = max(0, int(_PROFILE_WARMUP_CALLS) - total_now)

    print(
        f"[PROFILE] {name}: {elapsed_ms:.3f} ms | avg {avg_ms:.3f} ms | n={avg_cnt_now} | warmup_left={warmup_left}"
    )



# <추가하자>
class PointLenInputs(NamedTuple):
    unnorm_points_xyyaw: torch.Tensor  # (B,Pnn,point_len,4)
    points_valid: torch.Tensor  # (B,Pnn,point_len)  bool
    past_cur_valid: Optional[torch.Tensor]  # (B,Pnn,past_len+1) bool or None
    cur_future_valid: torch.Tensor  # (B,Pnn,1+future_len) bool
    past_len: int
    point_len: int


# =========================
# [NEW] 하이퍼/제약 파라미터 컨테이너
# =========================
@dataclass
class _ConstraintHParams:
    """제약/적분 하이퍼파라미터(상수 모음)."""
    dt: float
    eps: float
    # 필요: STE용 밴드폭 η (권장 초기값)
    eta_slip: float = 0.07  # S0
    eta_speed: float = 0.05  # S1
    eta_inc: float = 0.10  # S2
    eta_yaw: float = 0.05  # S3
    eta_fric: float = 0.05  # S4


def _shape4(
    B: int, Pnn: int, T: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(B,Pnn,T,4) 분해 저장용 4개 버퍼를 (x,y,cos,sin) 순서로 생성.

    returns:
            각 (B,Pnn,T) shape 텐서
    """
    zeros = lambda: torch.zeros((B, Pnn, T), device=device, dtype=dtype)
    return zeros(), zeros(), zeros(), zeros()  # x, y, cos, sin


class ActorClass(Enum):
    """행위자(객체) 종류."""
    PEDESTRIAN = "Pedestrian"
    BICYCLE = "Bicycle"
    CAR = "Car"


@dataclass(frozen=True)
class DynamicLimits:
    """행위자 클래스별 동역학 한계치.

    모든 내부 단위는 SI(미터, 초, 라디안) 기준.

    Attributes:
        v_max_mps: 최대 속도 [m/s]
        v_max_kmph: 최대 속도 [km/h] (편의상 함께 보관)
        a_max_mps2: 최대 종가속도(가감속 절대치) [m/s^2]
        alpha_max_radps2: 최대 요각가속도 [rad/s^2]
        a_lat_max_mps2: 최대 횡가속도 [m/s^2]
        R_min_m: 최소 선회 반경 [m]
        omega_max_abs_radps: 최대 요각속도 절대치 [rad/s]
        v_b_y_max: (비홀로노믹) 바디 y방향 속도 허용 최대치 [m/s]
    """
    v_max_mps: float
    v_max_kmph: float
    a_max_mps2: float
    alpha_max_radps2: float
    a_lat_max_mps2: float
    R_min_m: float
    omega_max_abs_radps: float
    v_b_y_max: float = 0.1  # 기본값(기존 동작과 동일하게 유지)
    beta_max_rad: float = 0.0  # ✅ 추가: 0이면 비활성(기본 동작 유지)

    def as_dict(self) -> Dict[str, float]:
        """딕셔너리 형태로 반환."""
        return asdict(self)


class FeasibleProjector(nn.Module):

    def __init__(self,
                 config,
                 hidden_dim: int,
                 use_feasible_dl: bool = True,
                 use_feasible_filter: bool = True):
        """Control Correction Network 본체 모듈 정의.

        Notes:
            - 트렁크 은닉 차원 H는 입력으로만 알 수 있으므로,
              trunk 압축기(Compressor)는 첫 forward에서 지연 초기화합니다.
        """
        super().__init__()
        self.config = config
        self.use_feasible_dl = use_feasible_dl
        self.use_feasible_filter = use_feasible_filter
        # --- [NEW] Savitzky–Golay 커널 캐시(LRU) ---
        # --- [NEW] Savitzky–Golay 커널 캐시(LRU) ---
        # key: (W, polyorder, deriv_order, dt_key, dtype, device, reg_eps)
        # val: torch.Tensor (1, 1, W)
        self._sg_conv_kernel_cache: OrderedDict[Tuple[int, int, int, float,
                                                      torch.dtype, torch.device,
                                                      float],
                                                torch.Tensor] = OrderedDict()
        self._sg_conv_kernel_cache_max_size: int = 32

        # valid_count 계산용 ones 커널 캐시
        # key: (W, device, dtype) -> (1,1,W)
        self._sg_ones_kernel_cache: Dict[Tuple[int, torch.device, torch.dtype],
                                         torch.Tensor] = {}

        self.use_batch_integration = self.config.use_batch_integration  # 필요
        self.constraints_h_params = _ConstraintHParams(
            dt=0.1,
            eps=1e-6,
            # 필요: η 초기값(S0~S4)
            eta_slip=0.07,
            eta_speed=0.05,
            eta_inc=0.10,
            eta_yaw=0.05,
            eta_fric=0.05,
        )

        # ---- 동역학 한계(기존) ----
        self.constraints: Dict[ActorClass, DynamicLimits] = {
            ActorClass.PEDESTRIAN:
                DynamicLimits(
                    v_max_mps=5.0,
                    v_max_kmph=5.0 * 3.6,
                    a_max_mps2=4.7,
                    alpha_max_radps2=14.0,
                    a_lat_max_mps2=3.2,
                    R_min_m=0.00001,
                    omega_max_abs_radps=3.3,
                    v_b_y_max=1.3,
                    beta_max_rad=10.0,  # ✅ TODO: (라디안) 통계값 넣기
                ),
            ActorClass.BICYCLE:
                DynamicLimits(
                    v_max_mps=22.,
                    v_max_kmph=22. * 3.6,
                    a_max_mps2=5.5,
                    alpha_max_radps2=6.0,
                    a_lat_max_mps2=4.4,
                    R_min_m=0.5,
                    omega_max_abs_radps=2.0,
                    v_b_y_max=1.3,
                    beta_max_rad=0.7,
                ),
            ActorClass.CAR:
                DynamicLimits(
                    v_max_mps=35.,
                    v_max_kmph=35. * 3.6,
                    a_max_mps2=8.0,
                    alpha_max_radps2=1.75,
                    a_lat_max_mps2=4.2,
                    R_min_m=4.50,
                    omega_max_abs_radps=0.9,
                    v_b_y_max=1.0,
                    beta_max_rad=0.27,
                ),
        }

        # ------------------------------
        # 아키텍처 하이퍼파라미터(고정 폭)
        # ------------------------------
        self._Dx: int = 24  # state encoder 출력 채널 (prev/fut 각각)
        self._Du: int = 16  # control adapter 출력 채널
        self._Dc: int = 32  # trunk compressor 출력 채널
        self._Din: int = self._Dx * 2 + self._Du + self._Dc  # 16+16+32+64=192
        self._C: int = self._Din  # 메인 채널 폭(192)
        self._eps: float = 1e-6

        # [추가 필요] L_integration 경로 차단용 플래그 (state, u_base detach)
        self.detach_u_for_ctrl_losses: bool = True  # 필요

        # ------------------------------
        # Encoders (시간축 보존, 채널만 변환)
        # ------------------------------
        if self.use_feasible_dl:
            # 상태 인코더(현재 노드 X_prev: (x,y,cos,sin))
            self.state_prev_encoder = nn.Sequential(
                nn.LayerNorm(4),  # (B,Pnn,segment_len,4)
                nn.Linear(4, self._Dx),  #: #4 → 16
                nn.GELU(),
                nn.Linear(self._Dx, self._Dx),  #: 16 → 16 (= self._Dx)
            )
            self.state_fut_encoder = nn.Sequential(
                nn.LayerNorm(4),  # (B,Pnn,segment_len,4)
                nn.Linear(4, self._Dx),  #: 4 → 16
                nn.GELU(),
                nn.Linear(self._Dx, self._Dx),  # : 16 → 16 (= self._Dx)
            )
            self.control_adapter = nn.Sequential(
                nn.LayerNorm(3),  # (B,Pnn,segment_len,3)
                nn.Linear(3, self._Du),  # : 3 → 8 (= self._Du)
                nn.GELU(),
            )

            # (B,Pnn,H) -> (B,Pnn,_Dc=8) 로 trunk 압축
            self.trunk_compressor = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 3 * self._Dc),  #: hidden_dim → 64 (중간 폭 축소)
                nn.GELU(),
                nn.Linear(3 * self._Dc, self._Dc),  #: 64 → 8 (= self._Dc)
            )

            # ------------------------------
            # Stem (채널 정렬)
            # ------------------------------
            self.stem_norm = nn.LayerNorm(self._Din)
            self.stem_fc = nn.Linear(self._Din, self._C)
            self.stem_act = nn.GELU()

            # ------------------------------
            # TCN 4블록: depthwise(7) + dilation {1,2,4,8} + 1x1
            # ------------------------------
            self._kernel_size: int = 9
            self._dilations: List[int] = [1, 8]  #: 4블록→2블록
            self.tcn_depth = len(self._dilations)  # : 현재는 2
            self.tcn_pre_lns = nn.ModuleList(  #: 블록 수만큼 LayerNorm
                [nn.LayerNorm(self._C) for _ in range(self.tcn_depth)])

            # depthwise conv (C 채널, groups=C)
            def _same_pad(k: int, d: int) -> int:
                return d * (k // 2)

            self.tcn_dw = nn.ModuleList([
                nn.Conv1d(in_channels=self._C,
                          out_channels=self._C,
                          kernel_size=self._kernel_size,
                          padding=_same_pad(self._kernel_size, d),
                          dilation=d,
                          groups=self._C,
                          bias=False) for d in self._dilations
            ])
            # key: (block_idx, device, dtype)
            self._tcn_den_ones_kernel_cache: Dict[Tuple[int, torch.device,
                                                        torch.dtype],
                                                  torch.Tensor] = {}
            # [NEW] den(=커널이 보는 위치 중 유효한 개수) 계산을 conv 없이 하기 위한 캐시
            # key: (kernel_size, dilation, padding, T, device) -> positions_kT (k, T) long
            self._tcn_den_positions_cache: Dict[Tuple[int, int, int, int,
                                                      torch.device],
                                                torch.Tensor] = {}
            # 블록마다 kernel_size가 달라질 수 있는 형태를 대비해 "블록별 base 텐서"로 저장
            for block_idx in range(self.tcn_depth):
                self.register_buffer(
                    f"_tcn_den_ones_kernel_base_{block_idx}",
                    torch.ones((1, 1, self._kernel_size), dtype=torch.float32),
                )

            # ------------------------------
            # [A3] 마스크 검증(디버그용) on/off
            #   - 기본값: False (학습/추론 경로에서 비용 제거)
            #   - True로 켜고 싶으면 config.feasible_debug_check_mask = True
            # ------------------------------
            self.feasible_debug_check_mask: bool = bool(
                getattr(self.config, "feasible_debug_check_mask", False))

            # pointwise 1x1 (시간축 보존, 채널 결합)
            # - 기존: nn.Linear(C->C) + (B*Pnn,C,T) ↔ (B*Pnn,T,C) 변환
            # - 변경: nn.Conv1d(C->C, kernel_size=1)로 (B*Pnn, C, T)에서 그대로 처리
            self.tcn_linear = nn.ModuleList([
                nn.Conv1d(
                    in_channels=self._C,
                    out_channels=self._C,
                    kernel_size=1,
                    bias=True,
                ) for _ in range(self.tcn_depth)
            ])

            # ------------------------------
            # Head & Gate
            # ------------------------------
            # 잔차 초안 ΔU_raw
            # - 기존: (B,Pnn,T,C) 토큰별 Linear
            # - 변경: (B*Pnn,C,T)에서 1x1 Conv로 처리
            head_bottleneck_dim: int = max(1, self._C // 4)

            self.head = nn.Sequential(
                nn.Conv1d(self._C, head_bottleneck_dim, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Conv1d(head_bottleneck_dim, 3, kernel_size=1, bias=True),
            )

            # 마지막 Conv 0-init → 초기엔 U_ref ≈ U_base
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

            gate_hidden_dim = 64
            self.gate_mlp = nn.Sequential(
                nn.LayerNorm(self._C),
                nn.Linear(self._C, gate_hidden_dim),
                nn.GELU(),
                nn.Linear(gate_hidden_dim, 3),
            )
            # Gate를 시간별이 아니라 이웃별로 계산할지 여부 (기본: True)
            self.feasible_gate_agentwise: bool = False  #bool(getattr(self.config, "feasible_gate_agentwise", True))

            # gate 초기 스케일 s0 설정(보수적으로)
            s0 = 0.05
            b_init = math.log(math.exp(float(s0)) - 1.0)  # softplus^{-1}(s0)
            with torch.no_grad():
                self.gate_mlp[-1].bias.fill_(b_init)

    @staticmethod
    def _to_bool_mask(mask: torch.Tensor) -> torch.Tensor:
        """0/1 또는 True/False 형태의 유효 표시를 True/False로 통일합니다.

        Args:
            mask (torch.Tensor): (...,) 모양의 텐서. 값이 0/1 이거나 bool일 수 있습니다.

        Returns:
            torch.Tensor: (...,) bool 텐서.
        """
        if mask.dtype == torch.bool:
            return mask
        return mask > 0.5

    @staticmethod
    def _infer_tokenwise_module_output_dim(
        module: nn.Module,
        input_last_dim: int,
    ) -> int:
        """한 위치(한 시점)의 마지막 축 길이가, 주어진 작업을 거친 뒤 어떻게 바뀌는지 추정합니다.

        - 여기서 “한 위치”는 (N, C)에서 N 중 한 줄을 의미합니다.
        - 이 함수는 '유효 위치가 0개'인 경우에도 출력 모양을 만들기 위해 사용합니다.

        Args:
            module (nn.Module): (N, C_in) -> (N, C_out) 형태로 동작하는 작업.
            input_last_dim (int): 입력의 마지막 축 길이(C_in).

        Returns:
            int: 출력의 마지막 축 길이(C_out).

        Raises:
            ValueError: 지원하지 않는 작업이 들어온 경우.
        """
        if isinstance(module, nn.Linear):
            return int(module.out_features)
        if isinstance(module, nn.LayerNorm):
            return int(input_last_dim)
        if isinstance(module,
                      (nn.GELU, nn.ReLU, nn.SiLU, nn.Identity, nn.Dropout)):
            return int(input_last_dim)

        if isinstance(module, nn.Sequential):
            dim = int(input_last_dim)
            for sub in module:
                dim = FeasibleProjector._infer_tokenwise_module_output_dim(
                    sub, dim)
            return dim

        raise ValueError(
            f"[FeasibleProjector] 지원하지 않는 작업 타입입니다: {type(module)}. "
            "필요하면 _infer_tokenwise_module_output_dim에 규칙을 추가해 주세요.")

    def _get_sg_ones_kernel_1x1w(
        self,
        window_length: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """valid_count 계산에 쓰는 ones 커널(1,1,W)을 캐시해서 반환합니다.

        Args:
            window_length: 창 길이 W (홀수 권장)
            device: 커널을 둘 디바이스
            dtype: 커널 dtype

        Returns:
            ones_kernel: (1, 1, W)
        """
        W = int(window_length)
        key = (W, device, dtype)
        cached = self._sg_ones_kernel_cache.get(key, None)
        if cached is not None:
            return cached

        ones_kernel = torch.ones((1, 1, W), device=device, dtype=dtype)
        self._sg_ones_kernel_cache[key] = ones_kernel
        return ones_kernel

    def _get_sg_center_derivative_kernel_1x1w(
        self,
        window_length: int,
        polyorder: int,
        derivative_order: int,
        dt: float,
        *,
        device: torch.device,
        dtype: torch.dtype,
        regularization_epsilon: float,
    ) -> torch.Tensor:
        """'창이 전부 유효'한 경우에 쓰는 SG 중앙 미분 커널을 (1,1,W)로 만들어 캐시합니다.

        이 커널을 conv1d로 적용하면, 각 시점 t에서
            derivative[t] = sum_{i=0..W-1} w[i] * x[t + i - half]
        형태로 SG 중앙 미분 값을 얻습니다.

        Args:
            window_length: 창 길이 W
            polyorder: 다항식 차수 P
            derivative_order: 1 또는 2
            dt: 샘플 간 시간 간격(초)
            device: 결과 커널을 둘 디바이스
            dtype: 결과 커널 dtype
            regularization_epsilon: (A^T A)에 더해 줄 작은 값(수치 안정)

        Returns:
            kernel_1x1w: (1, 1, W)
        """
        W = int(window_length)
        P = int(polyorder)
        d = int(derivative_order)
        if d not in (1, 2):
            raise ValueError(f"derivative_order must be 1 or 2. got={d}")

        # dt는 float 키로 쓸 때 흔들리지 않도록 약간 반올림
        dt_key = float(round(float(dt), 12))
        reg_key = float(round(float(regularization_epsilon), 18))

        key = (W, P, d, dt_key, dtype, device, reg_key)
        cached = self._sg_conv_kernel_cache.get(key, None)
        if cached is not None:
            # LRU 갱신
            self._sg_conv_kernel_cache.move_to_end(key)
            return cached

        # ---- 커널 계산(한 번만) ----
        # 계산은 CPU float64로 해서 안정적으로 만든 뒤, device/dtype으로 옮깁니다.
        with torch.no_grad():
            cpu = torch.device("cpu")
            compute_dtype = torch.float64

            design_matrix, _, _ = self._sg_build_design_matrix_and_gram(
                window_length=W,
                polyorder=P,
                dt=float(dt),
                device=cpu,
                dtype=compute_dtype,
            )  # (W, P+1)

            # A: (W, P+1)
            A = design_matrix
            P1 = int(P + 1)

            # ATA: (P+1, P+1)
            ATA = A.transpose(0, 1).matmul(A)

            # 정규화 항 추가: (P+1, P+1)
            I = torch.eye(P1, device=cpu, dtype=compute_dtype)
            ATA_reg = ATA + float(regularization_epsilon) * I

            # e_d: (P+1,)
            e = torch.zeros((P1,), device=cpu, dtype=compute_dtype)
            e[d] = 1.0

            # g: (P+1,) = (ATA_reg)^{-1} e_d
            g = torch.linalg.solve(ATA_reg, e)

            # weights: (W,) = d! * A @ g
            scale = float(math.factorial(d))
            w = scale * (A.matmul(g))  # (W,)

            kernel = w.view(1, 1, W).to(device=device, dtype=dtype)

        # LRU 저장
        self._sg_conv_kernel_cache[key] = kernel
        self._sg_conv_kernel_cache.move_to_end(key)
        if len(self._sg_conv_kernel_cache) > int(
                self._sg_conv_kernel_cache_max_size):
            self._sg_conv_kernel_cache.popitem(last=False)

        return kernel

    def _apply_same_kernel_conv1d_per_channel(
            self,
            seq_bTC: torch.Tensor,  # (N, T, C)
            kernel_1x1w: torch.Tensor,  # (1, 1, W)
    ) -> torch.Tensor:
        """같은 (1,1,W) 커널을 각 채널에 독립적으로 적용해 (N,T,C)로 돌려줍니다.

        Args:
            seq_bTC: (N, T, C)
            kernel_1x1w: (1, 1, W)

        Returns:
            out_bTC: (N, T, C)
        """
        if seq_bTC.numel() == 0:
            return seq_bTC

        if seq_bTC.dim() != 3:
            raise ValueError(
                "_apply_same_kernel_conv1d_per_channel: seq_bTC는 (N,T,C) 3D여야 합니다. "
                f"got shape={tuple(seq_bTC.shape)}")
        if kernel_1x1w.dim() != 3 or int(kernel_1x1w.shape[0]) != 1 or int(
                kernel_1x1w.shape[1]) != 1:
            raise ValueError(
                "_apply_same_kernel_conv1d_per_channel: kernel은 (1,1,W)여야 합니다. "
                f"got shape={tuple(kernel_1x1w.shape)}")

        N, T, C = seq_bTC.shape
        W = int(kernel_1x1w.shape[-1])
        half = W // 2

        # (N, C, T)
        x_bCT = seq_bTC.permute(0, 2, 1).contiguous()

        # (C, 1, W)로 반복해서 groups=C로 채널별 독립 conv
        weight = kernel_1x1w.to(dtype=seq_bTC.dtype,
                                device=seq_bTC.device).repeat(int(C), 1, 1)

        y_bCT = F.conv1d(
            x_bCT,
            weight,
            bias=None,
            stride=1,
            padding=int(half),
            dilation=1,
            groups=int(C),
        )  # (N, C, T)

        out_bTC = y_bCT.permute(0, 2, 1).contiguous()  # (N, T, C)
        return out_bTC

    def _apply_tokenwise_module_packed(
            self,
            x: torch.Tensor,  # (..., C_in)
            token_valid_mask: torch.Tensor,  # (...) bool or 0/1
            module: nn.Module,
            *,
            valid_token_flat_idx: Optional[torch.Tensor] = None,  # (N_valid,)
    ) -> torch.Tensor:
        """유효한 위치(True)만 뽑아서 작업을 적용하고, 결과를 원래 모양으로 되돌립니다.

        - x의 마지막 축(C_in)은 값의 채널(특성) 축이라고 가정합니다.
        - token_valid_mask가 False인 위치는 결과를 항상 0으로 둡니다.
        - 이렇게 하면 False 구간에 대해 큰 계산을 아예 하지 않습니다.

        Args:
            x (torch.Tensor):
                shape: (..., C_in)
            token_valid_mask (torch.Tensor):
                shape: (...)
                x에서 마지막 축을 뺀 모양과 같아야 합니다.
            module (nn.Module):
                (N, C_in) -> (N, C_out) 형태로 동작하는 작업(예: LayerNorm, Linear, MLP 등).
            valid_token_flat_idx (Optional[torch.Tensor]):
                shape: (N_valid,)
                이미 구해 둔 “유효 위치”의 1차원 인덱스가 있으면 넘깁니다(중복 계산 방지).

        Returns:
            torch.Tensor:
                shape: (..., C_out)
                유효 위치만 계산된 값, 무효 위치는 0.

        Raises:
            ValueError: 입력 모양이 맞지 않는 경우.
        """
        if token_valid_mask.shape != x.shape[:-1]:
            raise ValueError(
                "[FeasibleProjector] token_valid_mask.shape 와 x.shape[:-1]가 다릅니다. "
                f"token_valid_mask.shape={tuple(token_valid_mask.shape)}, x.shape={tuple(x.shape)}"
            )

        # mask_bool: (...) bool
        mask_bool = self._to_bool_mask(token_valid_mask)

        # x_flat: (N_token, C_in), mask_flat: (N_token,)
        x_flat = x.reshape(-1, int(x.shape[-1]))
        mask_flat = mask_bool.reshape(-1)

        if valid_token_flat_idx is None:
            valid_token_flat_idx = mask_flat.nonzero(as_tuple=False).squeeze(-1)

        # 유효 위치가 하나도 없으면, 모양만 맞춰서 0 반환
        if int(valid_token_flat_idx.numel()) == 0:
            out_dim = self._infer_tokenwise_module_output_dim(
                module=module,
                input_last_dim=int(x.shape[-1]),
            )
            return x.new_zeros((*x.shape[:-1], int(out_dim)))

        # x_valid: (N_valid, C_in)
        x_valid = x_flat.index_select(0, valid_token_flat_idx)

        # y_valid: (N_valid, C_out)
        y_valid = module(x_valid)
        if y_valid.dim() != 2:
            raise ValueError(
                "[FeasibleProjector] module 출력은 (N_valid, C_out) 2D 형태여야 합니다. "
                f"got shape={tuple(y_valid.shape)}")

        out_dim = int(y_valid.shape[-1])

        # y_flat: (N_token, C_out)  (무효 위치는 0으로 유지)
        y_flat = y_valid.new_zeros((int(x_flat.shape[0]), out_dim))
        y_flat = y_flat.index_copy(0, valid_token_flat_idx, y_valid)

        # (..., C_out)
        y = y_flat.view(*x.shape[:-1], out_dim)
        return y

    def _infer_past_future_lengths_for_downsample(
            self,
            diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, time_len)
    ) -> Tuple[int, int, int]:
        """다운샘플링에 사용할 과거 길이와 미래 길이를 계산합니다.

        Args:
            diffusion_trajectory: 현재+미래 궤적. shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid: 과거~현재~미래 유효 마스크. shape: (B, Pnn, time_len)

        Returns:
            past_len: 과거 노드 개수.
            future_len: 미래 노드 개수(T).
            time_len: 전체 타임라인 길이.
        """
        B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
        mask_B, mask_Pnn, time_len = target_past_cur_future_valid.shape

        if (mask_B, mask_Pnn) != (B, Pnn):
            raise ValueError(
                "[FeasibleProjector] diffusion_trajectory 와 target_past_cur_future_valid 의 "
                f"(B,Pnn)이 다릅니다: (B,Pnn)=({B},{Pnn}), (mask_B,mask_Pnn)=({mask_B},{mask_Pnn})"
            )

        future_len: int = int(one_plus_T - 1)
        past_len: int = int(time_len - (1 + future_len))
        if past_len < 0:
            raise ValueError(
                f"[FeasibleProjector] time_len={time_len} 이(가) 1+future_len={1 + future_len} 보다 작습니다."
            )

        if time_len != past_len + 1 + future_len:
            raise ValueError(
                "[FeasibleProjector] time_len 과 (past_len + 1 + future_len) 가 일치하지 않습니다."
            )

        return past_len, future_len, time_len

    def _decide_use_past_for_downsample(
            self,
            past_len: int,
            target_past: Optional[
                torch.Tensor],  # shape: (B, Pnn, past_len, 11) or None
            unnorm_near_past_xyyaw: Optional[
                torch.Tensor],  # shape: (B, Pnn, past_len, 4) or None
    ) -> bool:
        """과거 구간을 다운샘플에 포함할지 여부를 결정합니다.

        - config.use_past_for_feasible 가 False 이면 과거를 사용하지 않습니다.
        - past_len 이 0 이면 자동으로 과거를 사용하지 않습니다.
        - 과거를 쓰겠다고 했는데 텐서가 None 이면 에러를 냅니다.
        """
        use_past: bool = bool(
            getattr(self.config, "use_past_for_feasible", True))

        if use_past and past_len <= 0:
            # 과거를 쓰기로 했지만 실제 과거가 없으면 자동으로 끕니다.
            use_past = False

        if use_past:
            if target_past is None or unnorm_near_past_xyyaw is None:
                raise ValueError(
                    "[FeasibleProjector] use_past_for_feasible=True 인데 "
                    "target_past / unnorm_near_past_xyyaw 가 None 입니다.")

        return use_past

    @staticmethod
    def _min_sg_window_length(polyorder: int) -> int:
        """Savitzky–Golay에서 쓸 창 길이(window_length)의 최소값(홀수)을 계산합니다.

        - 일반적으로 window_length는 polyorder보다 커야 합니다.
        - 안정적으로 쓰려면 보통 polyorder+2 이상이 안전합니다.
        - SG는 보통 홀수 길이를 쓰므로, 여기서 홀수로 맞춥니다.

        Args:
            polyorder (int): SG 다항식 차수

        Returns:
            int: 최소 창 길이(홀수, 최소 3)
        """
        p = int(max(polyorder, 0))
        w = max(3, p + 2)  # polyorder+2 이상
        if w % 2 == 0:
            w += 1
        return w

    def _build_stride_indices_with_past(
        self,
        past_len: int,
        future_len: int,
        stride_step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int, int]:
        """과거~현재~미래 전체 타임라인에서 사용할 다운샘플 index 를 생성합니다.

        현재 시점을 기준으로 앞뒤로 stride_step 간격으로 고르고,
        항상 현재 시점과 마지막 미래 시점을 포함합니다.

        Returns:
            sample_idx_full: 전체 타임라인 기준 index. shape: (K,)
            past_len_ds: 다운샘플 후 과거 노드 개수.
            future_len_ds: 다운샘플 후 미래 노드 개수.
        """
        idx_cur_full: int = past_len
        idx_last_full: int = past_len + future_len

        if stride_step == 1:
            sample_idx_full = torch.arange(
                0,
                idx_last_full + 1,
                device=device,
                dtype=torch.long,
            )  # shape: (time_len,)
        else:
            # 현재 index 기준으로 과거 / 미래를 대칭으로 선택
            back_rev = torch.arange(
                idx_cur_full,
                -1,
                -stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (K_back,)
            back = torch.flip(back_rev, dims=(0,))  # 과거 방향 오름차순

            forward = torch.arange(
                idx_cur_full + stride_step,
                idx_last_full + 1,
                stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (K_fwd,)

            sample_idx_full = torch.cat([back, forward], dim=0)  # shape: (K,)

        # 현재 index 가 정확히 한 번만 포함되는지 검사
        current_pos_tensor = (sample_idx_full == idx_cur_full).nonzero(
            as_tuple=False).view(-1)
        if current_pos_tensor.numel() != 1:
            raise ValueError("[FeasibleProjector] 다운샘플 index 계산 중 현재 시점이 "
                             "정확히 한 번 포함되지 않았습니다.")

        current_pos: int = int(current_pos_tensor.item())
        past_len_ds: int = current_pos
        future_len_ds: int = int(sample_idx_full.numel() - 1 - past_len_ds)

        return sample_idx_full, past_len_ds, future_len_ds

    def _downsample_with_past(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        target_past: torch.Tensor,  # shape: (B, Pnn, past_len, 11)
        target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, past_len+1+T)
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        unnorm_near_past_xyyaw: torch.Tensor,  # shape: (B, Pnn, past_len, 4)
        sample_idx_full: torch.Tensor,  # shape: (K,)
        past_len_ds: int,
        future_len_ds: int,
    ) -> Tuple[
            torch.
            Tensor,  # unnorm_diffusion_trajectory_stride # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # unnorm_near_past_xyyaw_stride # shape: (B, Pnn, past_len_ds, 4) or None
            torch.
            Tensor,  # near_past_cur_future_valid_stride # shape: (B, Pnn, past_len_ds+1+T_ds)
            torch.
            Tensor,  # diffusion_trajectory_stride_norm # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # near_past_xyyaw_stride_norm # shape: (B, Pnn, past_len_ds, 4) or None
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """과거~현재~미래 전체 타임라인 기준으로 stride 다운샘플을 적용합니다."""
        # 역정규화 포인트 결합: [과거, 현재+미래]
        unnorm_points_all = torch.cat(
            [unnorm_near_past_xyyaw, unnorm_diffusion_trajectory],
            dim=2,
        )  # shape: (B, Pnn, past_len+1+T, 4)

        points_valid_all = target_past_cur_future_valid.to(
            torch.bool)  # shape: (B, Pnn, past_len+1+T)

        # stride 적용
        unnorm_points_stride = unnorm_points_all[:, :,
                                                 sample_idx_full, :]  # shape: (B, Pnn, past_len_ds+1+T_ds, 4)
        valid_stride = points_valid_all[:, :,
                                        sample_idx_full]  # shape: (B, Pnn, past_len_ds+1+T_ds)

        if past_len_ds > 0:
            unnorm_near_past_xyyaw_stride: Optional[
                torch.Tensor] = unnorm_points_stride[:, :, :past_len_ds, :]
        else:
            unnorm_near_past_xyyaw_stride = None

        unnorm_diffusion_trajectory_stride = unnorm_points_stride[:, :,
                                                                  past_len_ds:, :]  # shape: (B, Pnn, 1+T_ds, 4)

        # 정규화 포인트도 동일 index 로 다운샘플
        near_past_xyyaw = target_past[..., :4]  # shape: (B, Pnn, past_len, 4)
        points_norm_all = torch.cat(
            [near_past_xyyaw, diffusion_trajectory],
            dim=2,
        )  # shape: (B, Pnn, past_len+1+T, 4)

        points_norm_stride = points_norm_all[:, :,
                                             sample_idx_full, :]  # shape: (B, Pnn, past_len_ds+1+T_ds, 4)

        if past_len_ds > 0:
            near_past_xyyaw_stride_norm: Optional[
                torch.Tensor] = points_norm_stride[:, :, :past_len_ds, :]
        else:
            near_past_xyyaw_stride_norm = None

        diffusion_trajectory_stride_norm = points_norm_stride[:, :,
                                                              past_len_ds:, :]  # shape: (B, Pnn, 1+T_ds, 4)

        near_past_cur_future_valid_stride = valid_stride  # shape: (B, Pnn, past_len_ds+1+T_ds)

        return (
            unnorm_diffusion_trajectory_stride,
            unnorm_near_past_xyyaw_stride,
            near_past_cur_future_valid_stride,
            diffusion_trajectory_stride_norm,
            near_past_xyyaw_stride_norm,
            past_len_ds,
            future_len_ds,
        )

    def _build_stride_indices_without_past(
        self,
        future_len: int,
        stride_step: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        """현재~미래 구간(1+T)만 사용하는 경우의 다운샘플 index 를 생성합니다.

        Returns:
            sample_idx_local: 현재~미래 구간에서 사용할 index. shape: (1+T_ds,)
            future_len_ds: 다운샘플 후 미래 노드 개수(T_ds).
        """
        total_points: int = 1 + future_len  # 현재 포함

        if stride_step == 1:
            sample_idx_local = torch.arange(
                0,
                total_points,
                device=device,
                dtype=torch.long,
            )  # shape: (1+T,)
        else:
            sample_idx_local = torch.arange(
                0,
                total_points,
                stride_step,
                device=device,
                dtype=torch.long,
            )  # shape: (1+T_ds,)

            # 마지막 index(T)는 반드시 포함
            if int(sample_idx_local[-1].item()) != future_len:
                raise ValueError(
                    "[FeasibleProjector] (use_past_for_feasible=False) 환경에서 "
                    f"stride_step={stride_step} 가 future_len={future_len} 을 정확히 나누지 못했습니다. "
                    "config.feasible_stride_dt 를 조정해 주세요.")

        future_len_ds: int = int(sample_idx_local.numel() - 1)
        return sample_idx_local, future_len_ds

    def _downsample_without_past(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        target_past_cur_future_valid: torch.Tensor,  # shape: (B, Pnn, time_len)
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        sample_idx_local: torch.Tensor,  # shape: (1+T_ds,)
        future_len_ds: int,
        future_len: int,
    ) -> Tuple[
            torch.Tensor,  # unnorm_diffusion_trajectory_stride
            Optional[torch.Tensor],  # unnorm_near_past_xyyaw_stride
            torch.Tensor,  # near_past_cur_future_valid_stride
            torch.Tensor,  # diffusion_trajectory_stride_norm
            Optional[torch.Tensor],  # near_past_xyyaw_stride_norm
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """과거를 사용하지 않고 현재~미래 구간만 stride 다운샘플합니다."""
        B, Pnn, one_plus_T, _ = diffusion_trajectory.shape
        if one_plus_T != 1 + future_len:
            raise ValueError(
                f"[FeasibleProjector] diffusion_trajectory.shape[2]={one_plus_T} "
                f"!= 1 + future_len={1 + future_len}")

        # 현재~미래 유효 마스크만 추출
        cur_future_valid = target_past_cur_future_valid[:, :, -(
            1 + future_len):].to(torch.bool)  # shape: (B, Pnn, 1+future_len)

        # 역정규화 궤적 다운샘플
        unnorm_diffusion_trajectory_stride = unnorm_diffusion_trajectory[:, :,
                                                                         sample_idx_local, :]  # shape: (B, Pnn, 1+T_ds, 4)
        unnorm_near_past_xyyaw_stride: Optional[torch.Tensor] = None

        # 정규화 궤적 다운샘플
        diffusion_trajectory_stride_norm = diffusion_trajectory[:, :,
                                                                sample_idx_local, :]  # shape: (B, Pnn, 1+T_ds, 4)
        near_past_xyyaw_stride_norm: Optional[torch.Tensor] = None

        # 유효 마스크도 동일 index 로 다운샘플
        near_past_cur_future_valid_stride = cur_future_valid[:, :,
                                                             sample_idx_local]  # shape: (B, Pnn, 1+T_ds)

        past_len_ds: int = 0

        return (
            unnorm_diffusion_trajectory_stride,
            unnorm_near_past_xyyaw_stride,
            near_past_cur_future_valid_stride,
            diffusion_trajectory_stride_norm,
            near_past_xyyaw_stride_norm,
            past_len_ds,
            future_len_ds,
        )

    def build_downsampled_feasible_inputs(
        self,
        diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4) 정규화 현재+미래
        target_past: Optional[
            torch.Tensor],  # shape: (B, Pnn, past_len, 11) 또는 None
        target_past_cur_future_valid: torch.
        Tensor,  # shape: (B, Pnn, time_len=1+past_len+T) bool
        unnorm_diffusion_trajectory: torch.Tensor,  # shape: (B, Pnn, 1+T, 4)
        unnorm_near_past_xyyaw: Optional[
            torch.Tensor],  # shape: (B, Pnn, past_len, 4) 또는 None
        stride_step: int,
    ) -> Tuple[
            torch.
            Tensor,  # unnorm_diffusion_trajectory_stride # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # unnorm_near_past_xyyaw_stride # shape: (B, Pnn, past_len_ds, 4) or None
            torch.
            Tensor,  # near_past_cur_future_valid_stride # shape: (B, Pnn, past_len_ds+1+T_ds)
            torch.
            Tensor,  # diffusion_trajectory_stride_norm # shape: (B, Pnn, 1+T_ds, 4)
            Optional[
                torch.
                Tensor],  # near_past_xyyaw_stride_norm # shape: (B, Pnn, past_len_ds, 4) or None
            int,  # past_len_ds
            int,  # future_len_ds
    ]:
        """FeasibleProjector용 다운샘플링 궤적/마스크를 구성하는 메인 함수.
        흐름:
            1) 과거 길이(past_len), 미래 길이(future_len)를 계산합니다.
            2) config + past_len 으로 과거 사용 여부(use_past)를 정합니다.
            3) use_past 에 따라 stride index 를 만들고,
            4) 정규화/역정규화 궤적 + 유효 마스크에 같은 index 를 적용합니다.
        """
        device: torch.device = diffusion_trajectory.device

        # 1) 과거/미래 길이 계산
        past_len, future_len, _ = self._infer_past_future_lengths_for_downsample(
            diffusion_trajectory=diffusion_trajectory,  # shape: (B, Pnn, 1+T, 4)
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # shape: (B, Pnn, 1+past_len+T)
        )

        # 2) config + past_len 으로 과거 사용 여부 결정
        use_past: bool = self._decide_use_past_for_downsample(
            past_len=past_len,
            target_past=target_past,
            unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
        )

        if use_past:
            assert target_past is not None
            assert unnorm_near_past_xyyaw is not None

            # 3-a) 과거+현재+미래 전체 타임라인용 stride index 계산
            """
            sample_idx_full: shape (K,)
            past_len_ds: int
            future_len_ds: int
            """
            (sample_idx_full, past_len_ds,
             future_len_ds) = self._build_stride_indices_with_past(
                 past_len=past_len,
                 future_len=future_len,
                 stride_step=stride_step,
                 device=device,
             )

            # 4-a) 전체 타임라인 기준으로 다운샘플 적용
            return self._downsample_with_past(
                diffusion_trajectory=diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                target_past=target_past,  # (B, Pnn, past_len, 11)
                target_past_cur_future_valid=
                target_past_cur_future_valid,  # (B, Pnn, time_len)
                unnorm_diffusion_trajectory=
                unnorm_diffusion_trajectory,  # (B, Pnn, 1+T, 4)
                unnorm_near_past_xyyaw=
                unnorm_near_past_xyyaw,  # (B, Pnn, past_len, 4)
                sample_idx_full=sample_idx_full,  # (K,)
                past_len_ds=past_len_ds,  # int
                future_len_ds=future_len_ds,  # int
            )

        # 3-b) 과거 미사용: 현재~미래(1+T) 구간만 사용하는 stride index 계산
        (sample_idx_local,
         future_len_ds) = self._build_stride_indices_without_past(
             future_len=future_len,
             stride_step=stride_step,
             device=device,
         )

        # 4-b) 현재~미래 구간만 다운샘플 적용
        return self._downsample_without_past(
            diffusion_trajectory=diffusion_trajectory,
            target_past_cur_future_valid=target_past_cur_future_valid,
            unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
            sample_idx_local=sample_idx_local,
            future_len_ds=future_len_ds,
            future_len=future_len,
        )

    def upsample_future_controls_from_stride(
        self,
        unnorm_seg_body_control_stride: torch.
        Tensor,  # (B, Pnn, segment_len_ds, 3)
        past_len_ds: int,
        future_len_ds: int,
        future_len_full: int,
        stride_step: int,
    ) -> torch.Tensor:
        """서브샘플링된 세그먼트 제어를 원래 future_len 길이로 선형 업샘플링합니다.

        Args:
            unnorm_seg_body_control_stride:
                (B, Pnn, segment_len_ds, 3)
                과거~현재~미래 또는 현재~미래 전체에 대한 서브샘플링 세그먼트 제어.
            past_len_ds:
                다운샘플 후 과거 세그먼트 개수.
                (use_past_for_feasible=False 인 경우 0)
            future_len_ds:
                다운샘플 후 미래 세그먼트 개수.
            future_len_full:
                원래 미래 세그먼트 개수(T). (예: 80)
            stride_step:
                시간 index 기준 다운샘플링 간격.

        Returns:
            (B, Pnn, future_len_full, 3)
                현재~미래 구간(base dt 해상도)의 제어 시퀀스.
        """
        B, Pnn, segment_len_ds, _ = unnorm_seg_body_control_stride.shape

        # 미래 부분만 추출
        start_future_idx: int = int(past_len_ds)
        if segment_len_ds < start_future_idx + future_len_ds:
            raise ValueError(
                "[FeasibleProjector] segment_len_ds 와 (past_len_ds, future_len_ds) 조합이 일치하지 않습니다."
            )
        # (B, Pnn, future_len_ds, 3)
        coarse_future_control = unnorm_seg_body_control_stride[:, :,
                                                               start_future_idx:
                                                               start_future_idx +
                                                               future_len_ds, :]

        # stride_step == 1 이면 다운샘플링이 없으므로 그대로 사용
        if stride_step == 1:
            if future_len_ds != future_len_full:
                raise ValueError(
                    "[FeasibleProjector] stride_step==1 인데 "
                    f"future_len_ds={future_len_ds} 와 future_len_full={future_len_full} 이 다릅니다."
                )
            return coarse_future_control  # (B, Pnn, future_len_full, 3)

        # stride_step > 1: 선형 업샘플링 필요
        B_flat: int = B * Pnn
        _, _, n_coarse, _ = coarse_future_control.shape
        if n_coarse != future_len_ds:
            raise ValueError(
                "[FeasibleProjector] coarse_future_control 길이와 future_len_ds 가 일치하지 않습니다."
            )

        if n_coarse <= 1:
            # coarse 값이 1개뿐이면 모든 future 시점에 동일한 제어를 복사
            return coarse_future_control.expand(B, Pnn, future_len_full, 3)

        # (B*Pnn, 3, n_coarse) 로 변환 후 1D linear interpolate
        controls_flat = coarse_future_control.reshape(
            B_flat, n_coarse, 3).permute(0, 2, 1)  # (B*Pnn, 3, n_coarse)

        # align_corners=True:
        #   index 0 ↔ 첫 미래 세그먼트, index n_coarse-1 ↔ 마지막 미래 세그먼트
        upsampled_flat = F.interpolate(
            controls_flat,  # (B*Pnn, 3, n_coarse)
            size=future_len_full,
            mode="linear",
            align_corners=True,
        )  # (B*Pnn, 3, future_len_full)

        upsampled = upsampled_flat.permute(0, 2, 1).reshape(
            B, Pnn, future_len_full, 3)  # (B, Pnn, future_len_full, 3)
        return upsampled

    def get_feasible_stride_params(
        self,
        future_len: int,
    ) -> Tuple[int, float, int, int]:
        """FeasibleProjector용 다운샘플링 간격과 SG 윈도 길이를 계산합니다.

        추가된 안전장치:
            - stride_step이 커져도 SG 창 길이가 1/3 같은 값으로 내려가서
              SG가 사실상 꺼지거나 품질이 흔들리지 않게,
              max_window_len_xy/yaw에 최소 하한을 강제합니다.
            - 최소 하한은 polyorder 기반(polyorder+2, 홀수)으로 잡습니다.
              (기본 polyorder=2 → 최소 5)

        Returns:
            stride_step: 정수 스트라이드
            dt_for_savgol: SG에 넘길 샘플 간 시간 간격 [초]
            max_window_len_xy: x,y에 사용할 최대 창 길이(샘플 수)
            max_window_len_yaw: yaw에 사용할 최대 창 길이(샘플 수)
        """
        base_dt: float = float(self.constraints_h_params.dt)

        desired_dt: float = float(
            getattr(self.config, "feasible_stride_dt", base_dt))
        if desired_dt < base_dt:
            desired_dt = base_dt

        stride_step: int = max(1, int(round(desired_dt / base_dt)))
        dt_for_savgol: float = base_dt * float(stride_step)

        default_window_xy: int = int(
            getattr(self.config, "feasible_sg_max_window_len_xy", 11))
        default_window_yaw: int = int(
            getattr(self.config, "feasible_sg_max_window_len_yaw", 7))

        window_time_xy: float = base_dt * float(default_window_xy)
        window_time_yaw: float = base_dt * float(default_window_yaw)

        max_window_len_xy: int = max(1,
                                     int(round(window_time_xy / dt_for_savgol)))
        max_window_len_yaw: int = max(
            1, int(round(window_time_yaw / dt_for_savgol)))

        # ---------- [추가] SG 최소 창 길이 보장 ----------
        # 현재 코드 경로에서 polyorder는 2로 쓰고 있으므로 기본값 2로 둡니다.
        # 필요하면 config로 바꿀 수 있게 해 둡니다.
        poly_xy: int = int(getattr(self.config, "feasible_sg_polyorder_xy", 2))
        poly_yaw: int = int(getattr(self.config, "feasible_sg_polyorder_yaw",
                                    2))

        min_xy: int = self._min_sg_window_length(poly_xy)  # 예: poly=2 -> 5
        min_yaw: int = self._min_sg_window_length(poly_yaw)  # 예: poly=2 -> 5

        max_window_len_xy = max(int(max_window_len_xy), int(min_xy))
        max_window_len_yaw = max(int(max_window_len_yaw), int(min_yaw))
        # -----------------------------------------------

        # 홀수 길이 강제
        if max_window_len_xy % 2 == 0:
            max_window_len_xy += 1
        if max_window_len_yaw % 2 == 0:
            max_window_len_yaw += 1

        # 너무 큰 창은 제한(기존 로직 유지)
        max_allow_window: int = int(future_len + 1)
        max_window_len_xy = min(max_window_len_xy, max_allow_window)
        max_window_len_yaw = min(max_window_len_yaw, max_allow_window)

        # cap 이후 짝수가 될 수 있으니 다시 홀수로 맞춤(가능하면 -1로)
        if max_window_len_xy % 2 == 0:
            max_window_len_xy = max(1, max_window_len_xy - 1)
        if max_window_len_yaw % 2 == 0:
            max_window_len_yaw = max(1, max_window_len_yaw - 1)

        # stride_step이 future_len을 정확히 나누어야 한다(기존 유지)
        if future_len % stride_step != 0:
            raise ValueError(
                "[FeasibleProjector] future_len="
                f"{future_len} 이(가) stride_step={stride_step} 로 나누어 떨어지지 않습니다. "
                "config.feasible_stride_dt 를 조정해서 future_len % stride_step == 0 이 되도록 해 주세요."
            )

        return stride_step, dt_for_savgol, max_window_len_xy, max_window_len_yaw

    # ----------------------------
    # [NEW] 시간축 전체 배치로 S0/S1/S3 제약 적용 (S2는 미사용)
    # ----------------------------
    def _apply_constraints_batch(
        self,
        vx_b_raw: torch.Tensor,  # (B,Pnn,T)
        vy_b_raw: torch.Tensor,  # (B,Pnn,T)
        omega_raw: torch.Tensor,  # (B,Pnn,T)
        key_to_limit_bp: Dict[str, torch.Tensor],
        hp: _ConstraintHParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """시간축 전체 배치로 S0/S1/S2/S3 제약을 적용합니다.

        호출 순서:
            S0(β) -> S1(speed) -> S2(accel/alpha) -> S3(omega)

        Returns:
            vx_after, vy_after, omega_after: 모두 (B,Pnn,T)
        """
        if not self.use_feasible_filter:
            return vx_b_raw, vy_b_raw, omega_raw

        # (S0) 사이드슬립 각(β) 상한 → v_y^b만 줄이기
        vx_after, vy_after = self._apply_S0_sideslip_angle_limit_ste(
            vx_b=vx_b_raw,
            vy_b=vy_b_raw,
            beta_max_rad=key_to_limit_bp["beta_max_rad"],
            eta=hp.eta_slip,
            eps=hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
        )

        # (S1) 속도 크기 제한
        vx_after, vy_after = self._apply_S1_speed_limit_ste(
            vx_b=vx_after,
            vy_b=vy_after,
            v_max=key_to_limit_bp["v_max"],
            eta=hp.eta_speed,
            eps=hp.eps,
        )

        # (S2) 가속도/각가속도 증분 제한 (시간축 전체 배치, for-loop 없음)
        vx_after, vy_after, omega_after = self._apply_S2_accel_alpha_limits_ste_batch(
            vx_b=vx_after,  # (B,Pnn,T)
            vy_b=vy_after,  # (B,Pnn,T)
            omega=omega_raw,  # (B,Pnn,T)
            a_max=key_to_limit_bp["a_max"],  # (B,Pnn)
            alpha_max=key_to_limit_bp["alpha_max"],  # (B,Pnn)
            dt=float(hp.dt),
            eta=float(hp.eta_inc),
            eps=float(hp.eps),
        )

        # (S3) 속도-연동 omega 한계로 clip
        omega_after = self._apply_S3_omega_clip_ste(
            vx_b=vx_after,
            vy_b=vy_after,
            omega=omega_after,
            a_lat_max=key_to_limit_bp["a_lat_max"],
            R_min=key_to_limit_bp["R_min"],
            omega_abs_max=key_to_limit_bp["omega_abs_max"],
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
            eta=hp.eta_yaw,
            eps=hp.eps,
        )

        return vx_after, vy_after, omega_after

    # ----------------------------
    # [NEW] 시간축 전체 배치 중점 적분 (cumsum 기반)
    # ----------------------------
    def _integrate_midpoint_batch(
        self,
        unnorm_near_current_state: torch.
        Tensor,  # (B, Pnn, 4)  [x0, y0, cos0, sin0]
        vx_b_seq: torch.Tensor,  # (B, Pnn, T)  바디 기준 x속도 시퀀스
        vy_b_seq: torch.Tensor,  # (B, Pnn, T)  바디 기준 y속도 시퀀스
        omega_seq: torch.Tensor,  # (B, Pnn, T)  요각속도 시퀀스
        hp: _ConstraintHParams,
    ) -> Dict[str, torch.Tensor]:
        """시간축 전체에 대해 '중점 적분'을 한 번에 수행한다.

        - 각 구간 k 에 대해,
          yaw_k 에서 시작해서 yaw_k + w_k*dt 까지 회전한다고 보고
          중간 각(yaw_mid)을 이용해 세계 좌표 속도를 계산한다.
        - 이 세계 좌표 속도를 dt만큼 계속 더해 가며 위치를 만든다.
        - 모든 계산을 시간축에 대해 cumsum 으로 처리하므로,
          python for 루프 없이 한 번에 연산한다.
        """
        B, Pnn, T = vx_b_seq.shape  # T = future_len
        device = vx_b_seq.device
        dtype = vx_b_seq.dtype

        dt: float = hp.dt
        eps: float = hp.eps

        # 초기 위치/자세 (노드 0)
        x0 = unnorm_near_current_state[..., 0]  # (B,Pnn)
        y0 = unnorm_near_current_state[..., 1]  # (B,Pnn)
        cos0 = unnorm_near_current_state[..., 2]  # (B,Pnn)
        sin0 = unnorm_near_current_state[..., 3]  # (B,Pnn)

        # yaw0 (라디안) 복원
        yaw0 = torch.atan2(sin0, cos0)  # (B,Pnn)

        # 각속도 적분: Δθ_k = w_k * dt
        dtheta_seq = omega_seq * dt  # (B,Pnn,T)

        # 누적합: sum_{j<=k} Δθ_j
        dtheta_prefix = torch.cumsum(dtheta_seq, dim=2)  # (B,Pnn,T)

        # 각 구간 시작 각도 yaw_k = yaw0 + sum_{j<k} Δθ_j  (exclusive cumsum)
        zero_pad = torch.zeros_like(dtheta_seq[..., :1])  # (B,Pnn,1)
        dtheta_exclusive = torch.cat(
            [zero_pad, dtheta_prefix[..., :-1]],
            dim=2,
        )  # (B,Pnn,T)
        yaw_start = yaw0.unsqueeze(-1) + dtheta_exclusive  # (B,Pnn,T)

        # 중점/종단 각도
        yaw_mid = yaw_start + 0.5 * dtheta_seq  # (B,Pnn,T)
        yaw_next = yaw_start + dtheta_seq  # (B,Pnn,T)

        cos_mid = torch.cos(yaw_mid)  # (B,Pnn,T)
        sin_mid = torch.sin(yaw_mid)  # (B,Pnn,T)

        # 세계 기준 중점 속도
        vwx_mid = cos_mid * vx_b_seq - sin_mid * vy_b_seq  # (B,Pnn,T)
        vwy_mid = sin_mid * vx_b_seq + cos_mid * vy_b_seq  # (B,Pnn,T)

        # dt 만큼 이동량
        dx_seq = vwx_mid * dt  # (B,Pnn,T)
        dy_seq = vwy_mid * dt  # (B,Pnn,T)

        # 누적합으로 위치 만들기
        x_cumsum = torch.cumsum(dx_seq, dim=2)  # (B,Pnn,T)
        y_cumsum = torch.cumsum(dy_seq, dim=2)  # (B,Pnn,T)

        x_next = x0.unsqueeze(-1) + x_cumsum  # (B,Pnn,T)
        y_next = y0.unsqueeze(-1) + y_cumsum  # (B,Pnn,T)

        # 최종 yaw(k+1) → cos/sin
        cos_next = torch.cos(yaw_next)  # (B,Pnn,T)
        sin_next = torch.sin(yaw_next)  # (B,Pnn,T)

        # 수치 오차 보정용 정규화
        norm_cs = torch.sqrt(cos_next * cos_next + sin_next * sin_next +
                             eps)  # (B,Pnn,T)
        cos_next = cos_next / norm_cs
        sin_next = sin_next / norm_cs

        key_to_all_states: Dict[str, torch.Tensor] = {
            "x_next": x_next,  # (B,Pnn,T)
            "y_next": y_next,  # (B,Pnn,T)
            "cos_next": cos_next,  # (B,Pnn,T)
            "sin_next": sin_next,  # (B,Pnn,T)
            "vx_after": vx_b_seq,  # (B,Pnn,T)
            "vy_after": vy_b_seq,  # (B,Pnn,T)
            "omega_after": omega_seq,  # (B,Pnn,T)
        }
        return key_to_all_states

    def _sg_build_design_matrix_and_gram(
        self,
        window_length: int,
        polyorder: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Savitzky–Golay 미분에 필요한 시간 관련 텐서들을 만든다.

        이 함수가 하는 일은 크게 두 가지다.

        1) 시간 특징 텐서(time_feature) 만들기
           - 한 윈도 안에 들어가는 점의 개수를 window_length = W 라고 하자. (예: W=11)
           - 윈도 안의 각 점에 대해, "중심에서 얼마나 떨어져 있는지" 시간을 만든다.
             예를 들어 W=5, dt=0.1 이면
                 인덱스 k = 0,1,2,3,4 에 대해
                 중심은 k=2 이고,
                 시간 값 τ_k 는 다음과 같이 잡는다.
                     k=0 → τ = -2*dt
                     k=1 → τ = -1*dt
                     k=2 → τ =  0
                     k=3 → τ = +1*dt
                     k=4 → τ = +2*dt
           - 그리고 각 위치마다 다음과 같은 값들을 한 줄로 모은다.
                 [1, τ, τ², τ³, ..., τ^P]
             여기서 P = polyorder 이다. (예: polyorder=2 이면 [1, τ, τ²])
           - 이렇게 해서 얻는 텐서가 time_feature 이다.
             shape: (W, P+1)
               - 첫 번째 축: 윈도 안에서의 위치(시간 순서)
               - 두 번째 축: [1, τ, τ², ...] 항들

        2) 시간 특징 쌍 텐서(time_feature_pair) 미리 계산하기
           - Savitzky–Golay 미분을 할 때는
             time_feature 를 가지고 여러 번 곱셈과 덧셈을 반복해서 쓰게 된다.
           - 매번 같은 곱셈을 반복하지 않기 위해,
             time_feature[row] 의 각 항들끼리 곱해 둔 값을 미리 계산해서 모아 둔다.
             이 값은 나중에 “작은 (P+1)×(P+1) 행렬”들을 만들 때 바로 재사용된다.
           - 이렇게 모아 둔 텐서가 time_feature_pair 이다.
             shape: (W, P+1, P+1)
               - 첫 번째 축: 윈도 안에서의 위치
               - 두 번째/세 번째 축: time_feature 의 각 열 조합에 대한 곱

        Args:
            window_length (int):
                한 번에 보는 윈도 길이 W (항상 홀수여야 한다. 예: 5, 7, 11 ...)
            polyorder (int):
                τ, τ², τ³ ... 를 몇 제곱까지 사용할지 (다항식 차수)
            dt (float):
                샘플 간 시간 간격 (초 단위)
            device (torch.device):
                결과 텐서를 놓을 디바이스 (cpu, cuda 등)
            dtype (torch.dtype):
                결과 텐서의 데이터 타입 (torch.float32, torch.float16 등)

        Returns:
            time_feature (torch.Tensor):
                shape: (W, P+1)
                윈도 안 각 위치에 대해 [1, τ, τ², ...] 값을 쌓아 놓은 텐서.

            time_feature_pair (torch.Tensor):
                shape: (W, P+1, P+1)
                time_feature 에서 나오는 값들끼리의 곱을 미리 모아 둔 텐서.
                나중에 (N,T) 전체에 대해 작은 (P+1,P+1) 행렬을 빠르게 만들 때 쓴다.
        """
        W: int = int(window_length)
        P_plus_one: int = int(polyorder) + 1

        # τ_k: (W,) = [-half*dt, ..., 0, ..., +half*dt]
        half_window: int = W // 2
        tau_k: Tensor = (torch.arange(W, device=device, dtype=dtype) -
                         float(half_window)) * float(dt)  # (W,)

        # design_matrix: (W, P+1)  = [1, τ, τ², ...]
        basis_list: List[Tensor] = [torch.ones_like(tau_k)]  # (W,)
        for degree in range(1, P_plus_one):
            basis_list.append(tau_k**degree)  # (W,)
        design_matrix: Tensor = torch.stack(basis_list, dim=-1)  # (W, P+1)

        # gram_per_k: (W, P+1, P+1)  = Φ_k^T Φ_k (k별 outer product)
        design_row: Tensor = design_matrix.unsqueeze(-1)  # (W, P+1, 1)
        design_col: Tensor = design_matrix.unsqueeze(-2)  # (W, 1, P+1)
        gram_per_k: Tensor = design_row * design_col  # (W, P+1, P+1)

        # power_per_k: (W, P+1)  = Φ_k (b 계산에 직접 사용)
        power_per_k: Tensor = design_matrix  # (W, P+1)

        return design_matrix, gram_per_k, power_per_k

    # <추가하자>
    def _prepare_points_and_masks(
            self,
            unnorm_diffusion_trajectory: torch.Tensor,  # (B,Pnn,1+future_len,4)
            unnorm_near_past_xyyaw: Optional[
                torch.Tensor],  # (B,Pnn,past_len,4) or None
            target_past_cur_future_valid: torch.
        Tensor,  # (B,Pnn,past_len+1+future_len) bool
    ) -> PointLenInputs:
        """(1) 과거/현재/미래 포인트 결합
           (2) past_cur / cur_future 마스크 분할
           (3) 단조성 검증


        return PointLenInputs(
            unnorm_points_xyyaw,  # (B,Pnn,point_len,4)
            points_valid, # (B,Pnn,point_len)  bool
            past_cur_valid, # (B,Pnn,past_len+1) bool or None
            cur_future_valid, # (B,Pnn,1+future_len) bool
            past_len, # int
            point_len
                if unnorm_near_past_xyyaw is None:
                    = 1 + future_len
                else:
                    = past_len + 1 + future_len
        """
        B, Pnn, T1_fut, C = unnorm_diffusion_trajectory.shape
        assert C == 4, "xyyaw 마지막 채널은 4여야 합니다."
        Bv, Pnnv, total_time_len = target_past_cur_future_valid.shape
        assert (Bv, Pnnv) == (B, Pnn), "valid 마스크 (B,Pnn) 불일치"
        if total_time_len < T1_fut:  # T1_fut = 1 + future_len
            raise ValueError("valid 마스크 길이가 (1+future_len)보다 짧습니다.")
        if unnorm_near_past_xyyaw is None:
            past_len = 0
            # point_len = 1 + future_len
            unnorm_points_xyyaw = unnorm_diffusion_trajectory  # (B,Pnn,1+future_len,4)
            # target_past_cur_future_valid: (B,Pnn,past_len+1+future_len)
            past_cur_valid = None  # 과거~현재 마스크 없음
            cur_future_valid = target_past_cur_future_valid[:, :, -T1_fut:].to(
                torch.bool)  # (B,Pnn,1+future_len)
            points_valid = cur_future_valid  # (B,Pnn,1+future_len)
            # 현재~미래: True*False* 검증
            self._assert_cur_future_valid_mask(
                cur_future_valid,
                context="savgol_filter_for_control_cur_future")
        else:
            assert unnorm_near_past_xyyaw.shape[:2] == (B, Pnn)
            assert unnorm_near_past_xyyaw.shape[-1] == 4
            past_len = int(unnorm_near_past_xyyaw.shape[2])
            time_len = past_len + 1
            # 포인트 결합
            # point_len = past_len + 1 + future_len
            unnorm_points_xyyaw = torch.cat(
                [unnorm_near_past_xyyaw, unnorm_diffusion_trajectory],
                dim=2)  # (B,Pnn,past_len+1+future_len,4)
            # 마스크 분리
            all_valid = target_past_cur_future_valid.to(
                torch.bool)  # (B,Pnn,past_len+1+future_len)
            past_cur_valid = all_valid[..., :time_len]  # (B,Pnn,past_len+1)
            cur_future_valid = all_valid[..., past_len:]  # (B,Pnn,1+future_len)
            points_valid = all_valid  # (B,Pnn,point_len)
            # 검증: 과거~현재(0*1*), 현재~미래(1*0*)
            self._assert_past_cur_valid_mask(
                past_cur_valid, context="savgol_filter_for_control_past_cur")
            # ERROR
            self._assert_cur_future_valid_mask(
                cur_future_valid,
                context="savgol_filter_for_control_cur_future")

        point_len = int(unnorm_points_xyyaw.shape[2])
        """
        point_len
            if unnorm_near_past_xyyaw is None:
                = 1 + future_len
            else:
                = past_len + 1 + future_len
        """

        return PointLenInputs(
            unnorm_points_xyyaw=unnorm_points_xyyaw,  # (B,Pnn,point_len,4)
            points_valid=points_valid,  # (B,Pnn,point_len)  bool
            past_cur_valid=past_cur_valid,  # (B,Pnn,past_len+1) bool or None
            cur_future_valid=cur_future_valid,  # (B,Pnn,1+future_len) bool
            past_len=past_len,  # int
            point_len=point_len,  # int
        )

    # =========================================================
    # [추가 필요] STE 유틸 (공통)
    # =========================================================
    @staticmethod
    def _smoothstep_quintic01(t: torch.Tensor) -> torch.Tensor:
        """[0,1]→[0,1], C² smoothstep: 10t^3 - 15t^4 + 6t^5."""
        return t**3 * (10 - 15 * t + 6 * t * t)

    @staticmethod
    def _ste_band_weight(r: torch.Tensor, eta: float) -> torch.Tensor:
        """w(r): inside=1, band=1-φ, outside=0."""
        eta = float(max(0.0, eta))
        if eta == 0.0:
            return (r <= 1.0).to(r.dtype)
        r_lo, r_hi = 1.0 - eta, 1.0 + eta
        t = ((r - r_lo) / (2.0 * eta)).clamp(0.0, 1.0)
        phi = FeasibleProjector._smoothstep_quintic01(t)
        w_mid = 1.0 - phi
        w = torch.where(r <= r_lo, torch.ones_like(r),
                        torch.where(r >= r_hi, torch.zeros_like(r), w_mid))
        return w

    @staticmethod
    def _ste_scalar_clip(x: torch.Tensor, limit: torch.Tensor, eta: float,
                         eps: float) -> torch.Tensor:
        """forward=hard clamp, backward=band-weighted identity."""
        limit = limit.to(dtype=x.dtype, device=x.device)
        y_hard = x.clamp(-limit, limit)
        r = x.abs() / (limit + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * x + (1.0 - w) * x.detach()
        y = y_sur + (y_hard - y_sur).detach()
        return y

    @staticmethod
    def _ste_increment_vec(dv: torch.Tensor, limit: torch.Tensor, eta: float,
                           eps: float) -> torch.Tensor:
        """증분(벡터노름) STE-clip.

        dv: (B, Pnn, 2)
        limit: (B, Pnn)
        """
        limit = limit.to(dtype=dv.dtype,
                         device=dv.device).unsqueeze(-1)  # (B,Pnn,1)
        norm = torch.linalg.norm(dv, dim=-1,
                                 keepdim=True).clamp_min(eps)  # (B,Pnn,1)
        s_hard = torch.clamp(limit / norm, max=1.0)  # (B,Pnn,1)
        dv_h = s_hard * dv  # (B,Pnn,2)
        r = (norm.squeeze(-1)) / (limit.squeeze(-1) + eps)
        w = FeasibleProjector._ste_band_weight(r, eta).unsqueeze(-1)
        dv_sur = w * dv + (1 - w) * dv.detach()
        return dv_sur + (dv_h - dv_sur).detach()

    @staticmethod
    def _ste_yawrate_clip(omega_raw: torch.Tensor, allow: torch.Tensor,
                          eta: float, eps: float) -> torch.Tensor:
        """S3용: forward hard(±allow), backward는 r=|raw|/allow(detached)로 밴드 가중.

        omega_raw: (B, Pnn)
        allow: (B, Pnn)
        """
        allow = allow.to(dtype=omega_raw.dtype,
                         device=omega_raw.device)  # (B,Pnn)
        y_hard = omega_raw.clamp(-allow, allow)
        r = omega_raw.abs() / (allow.detach() + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * omega_raw + (1 - w) * omega_raw.detach()
        y = y_sur + (y_hard - y_sur).detach()

        return y

    @staticmethod
    def _ste_friction_scale(ax: torch.Tensor, ay: torch.Tensor,
                            ax_max: torch.Tensor, ay_max: torch.Tensor,
                            eta: float, eps: float) -> torch.Tensor:
        """S4용 타원 스케일: forward s_hard=min(1,1/r), backward s≈w(r)."""
        ax_max = ax_max.to(dtype=ax.dtype, device=ax.device)
        ay_max = ay_max.to(dtype=ay.dtype, device=ay.device)
        r = torch.sqrt((ax / (ax_max + eps))**2 + (ay / (ay_max + eps))**2 +
                       eps)
        s_hard = torch.clamp(1.0 / r, max=1.0)
        w = FeasibleProjector._ste_band_weight(r, eta)
        s_sur = w  # inside=1, band∈(0,1), outside=0
        s = s_sur + (s_hard - s_sur).detach()
        return s

    @staticmethod
    def _weighted_avg_two(
        a: torch.Tensor,
        b: torch.Tensor,
        a_valid: torch.Tensor,
        b_valid: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """두 텐서를 동일 shape에서 가중 평균합니다.

        Args:
            a (torch.Tensor): 좌측 값 (... 혹은 (..., D))
            b (torch.Tensor): 우측 값 (... 혹은 (..., D))
            a_valid (torch.Tensor): 좌측 가중치 (... 또는 (...,1))
            b_valid (torch.Tensor): 우측 가중치 (... 또는 (...,1))
            eps (float): 0으로 나눔 방지용 epsilon

        Returns:
            torch.Tensor: 가중 평균 텐서 (a,b와 동일 shape)
        """
        denom = (a_valid + b_valid).clamp_min(eps)
        return (a_valid * a + b_valid * b) / denom

    @staticmethod
    def _world_to_body(
        vx_w: torch.Tensor,
        vy_w: torch.Tensor,
        cos_yaw: torch.Tensor,
        sin_yaw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """세계 기준 속도를 몸체 기준 속도로 회전합니다.

        수식:
            v^b = R(-ψ) v^w,  R(-ψ) = [[cosψ, sinψ], [-sinψ, cosψ]]

        Args:
            vx_w (torch.Tensor): 세계 x축 속도 [..., T]
            vy (torch.Tensor):   세계 y축 속도 [..., T]
            cos_yaw (torch.Tensor): [..., T] cos(ψ)
            sin_yaw (torch.Tensor): [..., T] sin(ψ)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (v_x^b, v_y^b), shape 동일
        """
        vxb = cos_yaw * vx_w + sin_yaw * vy_w
        vyb = -sin_yaw * vx_w + cos_yaw * vy_w
        return vxb, vyb

    # <추가하자>
    def _prepare_midpoint_inputs(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        unnorm_near_past_xyyaw: Optional[
            torch.Tensor],  # (B, Pnn, past_len, 4) or None
        unnorm_points_world_control: torch.Tensor,  # (B, Pnn, point_len, 3)
        target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len)+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """중점 제어 계산을 위해
        - 좌표 시퀀스(x, y, cos, sin)
        - 월드 프레임 제어(v_x^w, v_y^w, w)
        - 유효 마스크
        를 하나의 공통 타임라인 기준(point_len)으로 정리합니다.

        Returns:
            unnorm_points_xyyaw: (B, Pnn, point_len, 4)
            unnorm_points_world_control_aligned: (B, Pnn, point_len, 3)
            points_valid: (B, Pnn, point_len)  bool
            point_len: int
        """
        # savgol_filter_for_control과 동일한 규칙으로 포인트/마스크 정리
        point_len_inputs: PointLenInputs = self._prepare_points_and_masks(
            unnorm_diffusion_trajectory=
            unnorm_diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
            unnorm_near_past_xyyaw=
            unnorm_near_past_xyyaw,  # (B, Pnn, past_len, 4) or None
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, Pnn, time_len(=1+past_len)+future_len)
        )
        unnorm_points_xyyaw = point_len_inputs.unnorm_points_xyyaw  # (B, Pnn, point_len, 4)
        points_valid = point_len_inputs.points_valid  # (B, Pnn, point_len)
        B, Pnn, point_len, _ = unnorm_points_xyyaw.shape

        B_c, P_c, point_len_control, _ = unnorm_points_world_control.shape
        assert point_len_control == point_len, \
            f"포인트 개수(point_len={point_len})와 제어 시퀀스 길이(point_len_control={point_len_control})가 다릅니다."

        return unnorm_points_xyyaw, unnorm_points_world_control, points_valid, point_len

    def compute_midpoint_controls(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,
        # (B, Pnn, 1+future_len, 4)
        unnorm_near_past_xyyaw: Optional[torch.Tensor],
        # (B, Pnn, past_len, 4) or None
        unnorm_points_world_control: torch.Tensor,
        # (B, Pnn, point_len, 3)
        target_past_cur_future_valid: torch.Tensor,
        # (B, Pnn, time_len(=1+past_len) + future_len) bool
        *,
        point_len_inputs: Optional[PointLenInputs] = None,
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)
        """구간 [t_k, t_{k+1})마다 중점(midpoint) 제어 [v_x^b, v_y^b, w]를 계산한다.

        변경점:
            - point_len_inputs가 들어오면, 내부에서 다시 cat/마스크 재구성을 하지 않고
              그대로 재사용합니다. (SG 단계에서 이미 만든 결과 재사용)
        """
        # 1) 포인트/제어/마스크 정리
        if point_len_inputs is None:
            (
                unnorm_points_xyyaw,  # (B, Pnn, point_len, 4)
                unnorm_points_world_control_aligned,  # (B, Pnn, point_len, 3)
                points_valid,  # (B, Pnn, point_len) bool
                point_len,  # int
            ) = self._prepare_midpoint_inputs(
                unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
                unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
                unnorm_points_world_control=unnorm_points_world_control,
                target_past_cur_future_valid=target_past_cur_future_valid,
            )
        else:
            unnorm_points_xyyaw = point_len_inputs.unnorm_points_xyyaw  # (B,Pnn,point_len,4)
            points_valid = point_len_inputs.points_valid  # (B,Pnn,point_len) bool
            point_len = int(point_len_inputs.point_len)

            unnorm_points_world_control_aligned = unnorm_points_world_control  # (B,Pnn,point_len,3)

            # 최소 shape 검증(비용 거의 없음)
            if unnorm_points_world_control_aligned.dim() != 4:
                raise ValueError(
                    "unnorm_points_world_control은 (B,Pnn,point_len,3) 4D 텐서여야 합니다."
                )
            if int(unnorm_points_world_control_aligned.shape[-1]) != 3:
                raise ValueError(
                    "unnorm_points_world_control 마지막 채널은 3이어야 합니다.")
            if int(unnorm_points_world_control_aligned.shape[2]) != int(
                    point_len):
                raise ValueError(
                    "point_len_inputs.point_len 과 unnorm_points_world_control의 point_len이 다릅니다. "
                    f"point_len_inputs.point_len={int(point_len)}, "
                    f"unnorm_points_world_control.shape[2]={int(unnorm_points_world_control_aligned.shape[2])}"
                )
            if unnorm_points_xyyaw.shape[:3] != points_valid.shape:
                raise ValueError(
                    "point_len_inputs 내부 shape이 일치하지 않습니다. "
                    f"unnorm_points_xyyaw.shape[:3]={tuple(unnorm_points_xyyaw.shape[:3])}, "
                    f"points_valid.shape={tuple(points_valid.shape)}")

        if point_len < 2:
            raise ValueError("포인트 개수가 2개 미만이면 중점 제어를 계산할 수 없습니다.")

        eps = 1e-6

        # 2) 노드 / 제어 성분 분해
        (
            cos_all,  # (B, Pnn, point_len)
            sin_all,  # (B, Pnn, point_len)
            v_x_all,  # (B, Pnn, point_len)
            v_y_all,  # (B, Pnn, point_len)
            omega_all  # (B, Pnn, point_len)
        ) = self._split_midpoint_nodes_and_controls(
            unnorm_points_xyyaw=unnorm_points_xyyaw,  # (B, Pnn, point_len, 4)
            unnorm_points_world_control=unnorm_points_world_control_aligned,
            # (B, Pnn, point_len, 3)
        )

        # 3) 세그먼트 유효 마스크(시작/끝/구간) 계산
        start_valid, end_valid, seg_valid = self._build_midpoint_segment_valid_masks(
            points_valid=points_valid,  # (B, Pnn, point_len) bool
            value_dtype=v_x_all.dtype,
        )  # (B,Pnn,segment_len) 각각

        # 4) 세계 프레임 중점 속도/각속도 계산  (A: 3개를 한 번에 계산)
        v_x_mid_w, v_y_mid_w, omega_mid = self._compute_midpoint_world_values(
            v_x_all=v_x_all,
            v_y_all=v_y_all,
            omega_all=omega_all,
            start_valid=start_valid,
            end_valid=end_valid,
            eps=eps,
        )  # (B,Pnn,segment_len) 각각

        # 5) 중간 yaw(cos/sin) 계산
        cos_mid, sin_mid = self._compute_midpoint_yaw_from_cos_sin(
            cos_all=cos_all,
            sin_all=sin_all,
            start_valid=start_valid,
            end_valid=end_valid,
            eps=eps,
        )  # (B,Pnn,segment_len) 각각

        # 6) 세계 → 바디 프레임 회전 + 무효 구간 마스킹
        unnorm_seg_body_control = self._rotate_midpoint_world_to_body_and_apply_mask(
            v_x_mid_world=v_x_mid_w,
            v_y_mid_world=v_y_mid_w,
            omega_mid=omega_mid,
            cos_mid=cos_mid,
            sin_mid=sin_mid,
            seg_valid=seg_valid,
        )  # (B, Pnn, segment_len, 3)

        return unnorm_seg_body_control

    # <추가하자>
    def _build_midpoint_segment_valid_masks(
        self,
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        value_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """중점 제어 계산용 세그먼트 유효 마스크를 생성합니다.

        Args:
            points_valid: (B, Pnn, point_len) bool
                각 노드(x,y,cos,sin)가 유효한지 여부.
            value_dtype: torch.dtype
                v_x, v_y 등과 맞추기 위한 dtype (float32/float16 등)

        Returns:
            start_valid: (B, Pnn, segment_len)  # 왼쪽 끝 노드 유효 여부 (float)
            end_valid:   (B, Pnn, segment_len)  # 오른쪽 끝 노드 유효 여부 (float)
            seg_valid:   (B, Pnn, segment_len)  # 두 끝 모두 유효한 세그먼트(0/1 float)
        """
        # points_valid: (B, Pnn, point_len),  segment_len = point_len - 1
        valid_bool = points_valid.to(torch.bool)
        start_valid_bool = valid_bool[..., :-1]  # (B, Pnn, segment_len)
        end_valid_bool = valid_bool[..., 1:]  # (B, Pnn, segment_len)
        seg_valid_bool = start_valid_bool & end_valid_bool  # (B, Pnn, segment_len)

        start_valid = start_valid_bool.to(value_dtype)
        end_valid = end_valid_bool.to(value_dtype)
        seg_valid = seg_valid_bool.to(value_dtype)

        return start_valid, end_valid, seg_valid

    # <추가하자>
    def _rotate_midpoint_world_to_body_and_apply_mask(
            self,
            v_x_mid_world: torch.Tensor,  # (B, Pnn, segment_len)
            v_y_mid_world: torch.Tensor,  # (B, Pnn, segment_len)
            omega_mid: torch.Tensor,  # (B, Pnn, segment_len)
            cos_mid: torch.Tensor,  # (B, Pnn, segment_len)
            sin_mid: torch.Tensor,  # (B, Pnn, segment_len)
            seg_valid: torch.Tensor,  # (B, Pnn, segment_len)  float {0,1}
    ) -> torch.Tensor:
        """세계 속도 중점 + 중간 yaw → 바디 프레임 중점 제어로 변환하고, 무효 구간은 0으로 마스킹한다.

        Returns:
            unnorm_seg_body_control: (B, Pnn, segment_len, 3)
        """
        vxb_mid, vyb_mid = self._world_to_body(
            v_x_mid_world,  # (B, Pnn, segment_len)
            v_y_mid_world,  # (B, Pnn, segment_len)
            cos_mid,  # (B, Pnn, segment_len)
            sin_mid,  # (B, Pnn, segment_len)
        )

        vxb_mid = vxb_mid * seg_valid
        vyb_mid = vyb_mid * seg_valid
        omega_mid = omega_mid * seg_valid

        unnorm_seg_body_control = torch.stack(
            [vxb_mid, vyb_mid, omega_mid],
            dim=-1,
        )  # (B, Pnn, segment_len, 3)

        return unnorm_seg_body_control

    def _compute_midpoint_yaw_from_cos_sin(
        self,
        cos_all: torch.Tensor,  # (B, Pnn, point_len)
        sin_all: torch.Tensor,  # (B, Pnn, point_len)
        start_valid: torch.Tensor,  # (B, Pnn, segment_len)  float {0,1}
        end_valid: torch.Tensor,  # (B, Pnn, segment_len)  float {0,1}
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 끝점의 (cos, sin)로부터 '중간 방향'의 (cos, sin)을 계산합니다.

        핵심 아이디어:
            - 기존 방식(끝 벡터를 평균낸 뒤 정규화)은,
              두 방향이 서로 반대(또는 거의 반대)일 때 평균 벡터의 크기가 0에 가까워져
              중간 방향이 쉽게 튀는 문제가 생길 수 있습니다.
            - 여기서는 (cos, sin)을 각도로 바꾼 뒤(atan2),
              두 각도의 차이를 [-pi, pi] 범위로 접어서(가장 짧은 회전),
              그 중간 각도를 사용합니다. 그래서 평균 벡터 크기가 0으로 가는 문제가 없습니다.

        마스크 처리:
            - start_valid/end_valid는 각 세그먼트의 양 끝 노드 유효 여부(0/1)입니다.
            - 양 끝이 모두 유효하면: 두 각도로 midpoint를 계산합니다.
            - 한쪽만 유효하면: 유효한 쪽 각도를 그대로 사용합니다.
            - 둘 다 무효면: 0 라디안 방향(cos=1, sin=0)을 사용합니다.

        Args:
            cos_all: (B, Pnn, point_len)
            sin_all: (B, Pnn, point_len)
            start_valid: (B, Pnn, segment_len) float {0,1}
            end_valid: (B, Pnn, segment_len) float {0,1}
            eps: 작은 값(0으로 나눔 방지 등)

        Returns:
            cos_mid: (B, Pnn, segment_len)
            sin_mid: (B, Pnn, segment_len)
        """
        # (B, Pnn, segment_len)
        cos_start = cos_all[..., :-1]
        sin_start = sin_all[..., :-1]
        cos_end = cos_all[..., 1:]
        sin_end = sin_all[..., 1:]

        # float(0/1) -> bool
        start_ok = start_valid > 0.5
        end_ok = end_valid > 0.5

        # 값이 NaN/Inf면 무효로 취급(출력 NaN 전파 방지)
        start_ok = start_ok & torch.isfinite(cos_start) & torch.isfinite(
            sin_start)
        end_ok = end_ok & torch.isfinite(cos_end) & torch.isfinite(sin_end)

        # 무효인 곳은 안전한 방향(0rad: cos=1, sin=0)으로 대체
        cos_start_safe = torch.where(start_ok, cos_start,
                                     torch.ones_like(cos_start))
        sin_start_safe = torch.where(start_ok, sin_start,
                                     torch.zeros_like(sin_start))
        cos_end_safe = torch.where(end_ok, cos_end, torch.ones_like(cos_end))
        sin_end_safe = torch.where(end_ok, sin_end, torch.zeros_like(sin_end))

        # 각도(라디안): (B, Pnn, segment_len)
        yaw_start = torch.atan2(sin_start_safe, cos_start_safe)
        yaw_end = torch.atan2(sin_end_safe, cos_end_safe)

        # 각도 차이를 [-pi, pi] 범위로 접기(= wrap_to_pi 역할)
        delta_raw = yaw_end - yaw_start
        delta = torch.atan2(torch.sin(delta_raw), torch.cos(delta_raw))

        # midpoint 각도
        yaw_mid = yaw_start + 0.5 * delta

        # 한쪽만 유효한 경우는 유효한 쪽을 그대로 사용(기존 가중 평균과 같은 의도)
        yaw_mid = torch.where(start_ok & (~end_ok), yaw_start, yaw_mid)
        yaw_mid = torch.where((~start_ok) & end_ok, yaw_end, yaw_mid)
        yaw_mid = torch.where((~start_ok) & (~end_ok),
                              torch.zeros_like(yaw_mid), yaw_mid)

        # 다시 (cos, sin)
        cos_mid = torch.cos(yaw_mid)
        sin_mid = torch.sin(yaw_mid)

        # (선택) 혹시 모를 수치 오차 정리(거의 1이지만 안전용)
        norm = torch.sqrt(cos_mid * cos_mid + sin_mid * sin_mid + eps)
        cos_mid = cos_mid / norm
        sin_mid = sin_mid / norm

        return cos_mid, sin_mid

    # <추가하자>
    def _compute_midpoint_yaw_from_cos_sin_prev(
        self,
        cos_all: torch.Tensor,  # (B, Pnn, point_len)
        sin_all: torch.Tensor,  # (B, Pnn, point_len)
        start_valid: torch.Tensor,  # (B, Pnn, segment_len)
        end_valid: torch.Tensor,  # (B, Pnn, segment_len)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """두 끝점의 cos/sin과 유효 마스크로 중간 yaw의 cos/sin을 계산한다.

        Returns:
            cos_mid: (B, Pnn, segment_len)
            sin_mid: (B, Pnn, segment_len)
        """
        cos_start = cos_all[..., :-1]  # (B, Pnn, segment_len)
        cos_end = cos_all[..., 1:]  # (B, Pnn, segment_len)
        sin_start = sin_all[..., :-1]  # (B, Pnn, segment_len)
        sin_end = sin_all[..., 1:]  # (B, Pnn, segment_len)

        # cos_mid: (B, Pnn, segment_len)
        cos_mid = self._weighted_avg_two(
            cos_start,  # (B, Pnn, segment_len)
            cos_end,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        # sin_mid: (B, Pnn, segment_len)
        sin_mid = self._weighted_avg_two(
            sin_start,  # (B, Pnn, segment_len)
            sin_end,  # (B, Pnn, segment_len)
            start_valid,  # (B, Pnn, segment_len)
            end_valid,  # (B, Pnn, segment_len)
            eps=eps)  # (B, Pnn, segment_len)

        norm = (cos_mid * cos_mid + sin_mid * sin_mid).clamp_min(eps).sqrt()
        cos_mid = cos_mid / norm
        sin_mid = sin_mid / norm

        return cos_mid, sin_mid

    # <추가하자>
    def _compute_midpoint_world_values(
        self,
        v_x_all: torch.Tensor,  # (B, Pnn, point_len)
        v_y_all: torch.Tensor,  # (B, Pnn, point_len)
        omega_all: torch.Tensor,  # (B, Pnn, point_len)
        start_valid: torch.Tensor,  # (B, Pnn, segment_len)
        end_valid: torch.Tensor,  # (B, Pnn, segment_len)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """세계 프레임에서 중점 v_x^w, v_y^w, w 를 계산한다.

        변경점:
            - v_x / v_y / omega를 따로 3번 평균내지 않고,
              (.., 3)로 한 번에 묶어서 1번에 평균냅니다.
            - 수식은 기존과 동일합니다.

        Returns:
            v_x_mid_w: (B, Pnn, segment_len)
            v_y_mid_w: (B, Pnn, segment_len)
            omega_mid: (B, Pnn, segment_len)
        """
        # (B, Pnn, point_len, 3)
        u_all = torch.stack([v_x_all, v_y_all, omega_all], dim=-1)

        # (B, Pnn, segment_len, 3)
        u_start = u_all[..., :-1, :]
        u_end = u_all[..., 1:, :]

        # (B, Pnn, segment_len, 1)  <- 마지막 채널(3)에 브로드캐스트 되도록 1차원 추가
        w_start = start_valid.unsqueeze(-1)
        w_end = end_valid.unsqueeze(-1)

        # (B, Pnn, segment_len, 3)
        u_mid = self._weighted_avg_two(
            u_start,
            u_end,
            w_start,
            w_end,
            eps=eps,
        )

        v_x_mid_w = u_mid[..., 0]
        v_y_mid_w = u_mid[..., 1]
        omega_mid = u_mid[..., 2]
        return v_x_mid_w, v_y_mid_w, omega_mid

    def _get_tcn_dilated_positions_kT(
        self,
        *,
        kernel_size: int,
        dilation: int,
        padding: int,
        sequence_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        """(den 계산용) 각 시점 t에서 '커널이 참조하는 입력 위치' 표를 만든다.

        Args:
            kernel_size (int): 한 번에 보는 길이 k.
            dilation (int): 간격 d. (예: 2면 2칸씩 건너뜀)
            padding (int): 양쪽에 0을 붙이는 길이 p.
            sequence_length (int): 시간 길이 T.
            device (torch.device): 텐서를 만들 디바이스.

        Returns:
            torch.Tensor:
                positions_kT: (k, T) long
                    positions_kT[i, t] = t + (i * dilation) - padding

        Notes:
            - 이 값은 mask가 0/1일 때, 기존 den = conv1d(mask, ones, dilation, padding)과
              **동일한 위치들을** 참조한다.
            - dtype과 무관한 long 텐서라 캐시에 저장해 재사용한다.
        """
        k = int(kernel_size)
        d = int(dilation)
        p = int(padding)
        T = int(sequence_length)

        key = (k, d, p, T, device)
        cached = self._tcn_den_positions_cache.get(key, None)
        if cached is not None:
            return cached

        # offsets: (k,) = [0*d - p, 1*d - p, ..., (k-1)*d - p]
        offsets = torch.arange(k, device=device,
                               dtype=torch.long) * d - p  # (k,)

        # t_index: (T,) = [0, 1, ..., T-1]
        t_index = torch.arange(T, device=device, dtype=torch.long)  # (T,)

        # positions_kT: (k, T)
        positions_kT = offsets.unsqueeze(1) + t_index.unsqueeze(0)

        self._tcn_den_positions_cache[key] = positions_kT
        return positions_kT

    def _compute_den_from_contiguous_mask_b1t(
        self,
        mask_b1t: torch.Tensor,  # (N, 1, T) float(0/1) 또는 bool
        *,
        kernel_size: int,
        dilation: int,
        padding: int,
    ) -> torch.Tensor:
        """마스크가 '한 덩어리(0*1*0*)'라는 가정 하에 den을 conv 없이 계산한다.

        Args:
            mask_b1t (torch.Tensor):
                shape: (N, 1, T)
                값은 0/1 (또는 bool) 이어야 한다.
            kernel_size (int): k
            dilation (int): d
            padding (int): p

        Returns:
            torch.Tensor:
                den_b1t: (N, 1, T), dtype=mask_b1t.dtype
                각 시점 t에서, 커널이 참조하는 위치들 중 mask==1인 개수.

        Raises:
            ValueError:
                (디버그 모드에서) mask에 구멍(1→0→1)이 있으면 에러.
                이 경우는 '한 덩어리' 가정이 깨져서 결과가 달라질 수 있다.
        """
        if mask_b1t.numel() == 0:
            return mask_b1t

        if mask_b1t.dim() != 3 or int(mask_b1t.shape[1]) != 1:
            raise ValueError(
                "_compute_den_from_contiguous_mask_b1t: mask_b1t는 (N,1,T)여야 합니다. "
                f"got shape={tuple(mask_b1t.shape)}")

        N = int(mask_b1t.shape[0])
        T = int(mask_b1t.shape[2])

        # mask_bool: (N, T)
        mask_bool = (mask_b1t.squeeze(1) > 0.5
                    ) if mask_b1t.dtype != torch.bool else mask_b1t.squeeze(1)

        # has_any: (N,)  -> 마스크가 1을 하나라도 갖는지
        has_any = mask_bool.any(dim=-1)  # (N,)

        # start/end: (N,) long
        # - start: 첫 1의 위치
        # - end:   마지막 1의 위치
        mask_int = mask_bool.to(torch.int8)  # (N, T)

        start = torch.argmax(mask_int, dim=-1)  # (N,)  all-zero면 0이 나옴
        end_from_right = torch.argmax(torch.flip(mask_int, dims=[-1]),
                                      dim=-1)  # (N,)
        end = (T - 1) - end_from_right  # (N,) all-zero면 T-1이 나옴

        # all-zero 행 처리: start=0, end=-1 로 만들어서 span이 비게 함
        start = torch.where(has_any, start, torch.zeros_like(start))
        end = torch.where(has_any, end, torch.full_like(end, -1))

        # (선택/권장) 디버그 모드에서만 “한 덩어리(구멍 없음)” 검증
        if bool(getattr(self, "feasible_debug_check_mask", False)):
            # mask_sum: (N,)  실제 1의 개수
            mask_sum = mask_int.sum(dim=-1).to(torch.long)  # (N,)
            # span_len: (N,)  start~end 길이 (all-zero면 0)
            span_len = (end.to(torch.long) - start.to(torch.long) +
                        1).clamp_min(0)

            bad = has_any & (mask_sum != span_len)
            if bool(bad.any()):
                bad_idx = bad.nonzero(as_tuple=False).squeeze(-1)
                max_show = min(int(bad_idx.numel()), 8)
                sample = bad_idx[:max_show].tolist()
                raise ValueError(
                    "[_compute_den_from_contiguous_mask_b1t] mask가 '한 덩어리(0*1*0*)' 형태가 아닙니다. "
                    "즉, 1→0→1 같은 구멍이 있습니다. "
                    f"예시 row idx={sample} (N={N}, T={T}).")

        # positions_kT: (k, T) long
        positions_kT = self._get_tcn_dilated_positions_kT(
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=int(padding),
            sequence_length=int(T),
            device=mask_b1t.device,
        )

        # start/end broadcast:
        # start_nt11: (N,1,1), end_nt11: (N,1,1)
        start_nt11 = start.to(dtype=torch.long,
                              device=mask_b1t.device).view(N, 1, 1)
        end_nt11 = end.to(dtype=torch.long,
                          device=mask_b1t.device).view(N, 1, 1)

        # pos_1kT: (1, k, T)
        pos_1kT = positions_kT.view(1, int(kernel_size), T)

        # in_range: (N, k, T) bool
        in_range = (pos_1kT >= start_nt11) & (pos_1kT <= end_nt11)

        # den_nt: (N, T) -> (N,1,T)
        den_nt = in_range.sum(dim=1).to(dtype=mask_b1t.dtype)  # (N,T)
        den_b1t = den_nt.unsqueeze(1)  # (N,1,T)

        return den_b1t

    def _get_tcn_den_ones_kernel(
        self,
        block_idx: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """분모(den) 계산용 ones 커널을 (device, dtype)에 맞춰 반환합니다.

        depthwise conv에서 마스크 합(=커널 안 유효 샘플 수)을 구할 때
        den = conv1d(mask, ones_kernel)을 사용합니다.

        ones_kernel을 매번 새로 만들지 않기 위해:
        1) 블록별 base ones 텐서를 buffer로 저장해 두고,
        2) (device, dtype)별 변환 결과를 캐시에 저장해 재사용합니다.

        Args:
            block_idx (int):
                TCN 블록 인덱스 (0 ~ self.tcn_depth-1).
            device (torch.device):
                결과 텐서를 둘 디바이스.
            dtype (torch.dtype):
                결과 텐서의 데이터 타입. 보통 conv 입력과 동일하게 맞춥니다.

        Returns:
            torch.Tensor:
                shape: (1, 1, k)
                분모 계산에 사용할 ones 커널 텐서.
        """
        key = (int(block_idx), device, dtype)
        cached = self._tcn_den_ones_kernel_cache.get(key, None)
        if cached is not None:
            return cached

        base = getattr(self, f"_tcn_den_ones_kernel_base_{int(block_idx)}")
        kernel = base.to(device=device, dtype=dtype)
        self._tcn_den_ones_kernel_cache[key] = kernel
        return kernel

    def _depthwise_conv_masked_bct(
        self,
        x_bct: torch.Tensor,  # (B*Pnn, C, T)
        mask_b1t: torch.Tensor,  # (B*Pnn, 1, T)  float(0/1) 또는 bool
        block_idx: int,
    ) -> torch.Tensor:
        """마스크를 고려한 depthwise conv을 (B*Pnn, C, T) 형태에서 수행합니다.

        변경점(요청 반영):
            - den은 '유효 샘플 개수'이므로 값이 0,1,2,... 형태입니다.
            - den==0인 경우를 포함해 분모가 너무 작아지는 것을 막기 위해
              den을 최소 1.0으로 클램프합니다.
        """
        if x_bct.numel() == 0:
            return x_bct

        conv = self.tcn_dw[int(block_idx)]

        # dtype/device 정렬
        mask = mask_b1t.to(dtype=x_bct.dtype, device=x_bct.device)

        # -----------------
        # 분자: conv(x * mask)
        # -----------------
        y_num = conv(x_bct * mask)  # (B*Pnn, C, T)

        # -----------------
        # 분모: den (conv 없이 계산)
        # -----------------
        k = int(conv.kernel_size[0])
        pad = int(conv.padding[0])
        dil = int(conv.dilation[0])

        den = self._compute_den_from_contiguous_mask_b1t(
            mask_b1t=mask,  # (B*Pnn, 1, T)
            kernel_size=k,
            dilation=dil,
            padding=pad,
        )  # (B*Pnn, 1, T)

        # -----------------
        # 정규화
        #  - den은 유효 샘플 개수(0,1,2,...)라서 최소 1로 클램프하는 게 안전합니다.
        # -----------------
        den_safe = den.clamp_min(1.0)  # (B*Pnn, 1, T)
        y = y_num / den_safe  # (B*Pnn, C, T)
        return y

    def _depthwise_conv_masked(
        self,
        seq_bptc: torch.Tensor,  # (B,Pnn,T,C)
        seg_mask_bpt: torch.Tensor,  # (B,Pnn,T), float(0/1)
        block_idx: int,
    ) -> torch.Tensor:
        """정규화 depthwise conv (기존 (B,Pnn,T,C) 입력용 래퍼).

        내부에서는 (B*Pnn, C, T)로 바꾼 뒤,
        _depthwise_conv_masked_bct를 호출해 계산합니다.

        Args:
            seq_bptc: (B,Pnn,T,C)
            seg_mask_bpt: (B,Pnn,T) 0/1
            block_idx: 블록 인덱스

        Returns:
            (B,Pnn,T,C)
        """
        B, Pnn, T, C = seq_bptc.shape

        x = seq_bptc.permute(0, 1, 3, 2).reshape(B * Pnn, C, T)  # (B*Pnn,C,T)
        m = seg_mask_bpt.reshape(B * Pnn, 1, T)  # (B*Pnn,1,T)

        y = self._depthwise_conv_masked_bct(
            x_bct=x,
            mask_b1t=m,
            block_idx=block_idx,
        )  # (B*Pnn,C,T)

        out = y.view(B, Pnn, C, T).permute(0, 1, 3,
                                           2).contiguous()  # (B,Pnn,T,C)
        return out

    # =========================================================
    # [A] 마스크/입력 전처리
    # =========================================================
    def _build_segment_mask(
        self,
        near_cur_future_valid: torch.Tensor,  # (B,Pnn,1+T) bool
        value_dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """노드 유효 마스크로부터 세그먼트(구간) 유효 마스크를 만듭니다.

        - 노드가 (t_k, t_{k+1}) 둘 다 유효하면 그 구간을 유효로 봅니다.
        - seg_mask는 0/1 값을 갖는 실수 텐서입니다.

        Args:
            near_cur_future_valid:
                (B,Pnn,1+T) bool
            value_dtype:
                seg_mask 출력 dtype. None이면 float32 사용.

        Returns:
            seg_mask:
                (B,Pnn,T) float(0/1)
            seg_mask_1:
                (B,Pnn,T,1) float(0/1)
        """
        valid = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        seg_valid = (valid[..., :-1] & valid[..., 1:])  # (B,Pnn,T) bool

        if value_dtype is None:
            value_dtype = torch.float32

        seg_mask = seg_valid.to(dtype=value_dtype)  # (B,Pnn,T)
        seg_mask_1 = seg_mask.unsqueeze(-1)  # (B,Pnn,T,1)
        return seg_mask, seg_mask_1

    def _split_prev_fut(
            self,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+T, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """현재/미래 노드로 분리.

        Args:
            diffusion_trajectory (torch.Tensor): (B, Pnn, 1+T, 4) = [x, y, cos, sin]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - x_prev: (B, Pnn, T, 4)  ← t_k
                - x_fut : (B, Pnn, T, 4)  ← t_{k+1}
        """
        x_prev = diffusion_trajectory[..., :-1, :]  # (B,Pnn,T,4)
        x_fut = diffusion_trajectory[..., 1:, :]  # (B,Pnn,T,4)
        return x_prev, x_fut

    def _normalize_cos_sin(
        self,
        xycs: torch.Tensor,  # (B, Pnn, T, 4)
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """(cos, sin)을 단위원으로 재투영(수치 안전).

        Args:
            xycs (torch.Tensor): (B, Pnn, T, 4) = [x, y, cos, sin]
            eps (float): 0 나눗셈 방지용 epsilon

        Returns:
            torch.Tensor: (B, Pnn, T, 4)  재투영된 텐서
        """
        cs = xycs[..., 2:4]
        norm = torch.linalg.norm(cs, dim=-1, keepdim=True).clamp_min(eps)
        out = xycs.clone()
        out[..., 2:4] = cs / norm
        return out

    def _features_from_inputs(
            self,
            x_prev: torch.Tensor,  # (B, Pnn, segment_len, 4)
            x_fut: torch.Tensor,  # (B, Pnn, segment_len, 4)
            u_base: torch.Tensor,  # (B, Pnn, segment_len, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
            points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len) bool
    ) -> torch.Tensor:
        """입력을 통일 피처 Z_in으로 변환합니다(전체 한번에 계산 + 마스크로 0 고정).

        핵심 변경:
            - 예전에는 유효 구간만 뽑아서(state encoder / control adapter) 계산했습니다.
            - 이제는 (B,Pnn,T,·) 전체를 한 번에 계산한 뒤,
              유효하지 않은 구간은 seg_mask_1로 곱해 **출력을 0으로 고정**합니다.

        왜 결과가 같나:
            - 여기서 하는 작업들은 각 시점(토큰)별로 독립적으로 계산되므로,
              무효 구간을 같이 계산하더라도 유효 구간의 값은 변하지 않습니다.
            - 무효 구간은 곧바로 0으로 눌러서 출력/학습 신호가 사라집니다.

        Args:
            x_prev: (B, Pnn, T, 4)  구간 시작 노드의 [x,y,cos,sin]
            x_fut:  (B, Pnn, T, 4)  구간 끝 노드의 [x,y,cos,sin]
            u_base: (B, Pnn, T, 3)  베이스 제어 [v_x^b, v_y^b, w]
            dit_final_hidden_tokens: (B, Pnn, H)  트렁크 은닉
            points_valid: (B, Pnn, 1+T) bool  노드 유효 마스크(현재 포함)

        Returns:
            Z_in: (B, Pnn, T, self._Din)
                여기서 self._Din = 2*self._Dx + self._Du + self._Dc
        """
        # seg_mask_1: (B, Pnn, T, 1)  float(0/1), dtype는 입력 dtype으로 맞춤
        _, seg_mask_1 = self._build_segment_mask(
            near_cur_future_valid=points_valid,  # (B,Pnn,1+T)
            value_dtype=x_prev.dtype,
        )

        # u_base는 필요 시 detach만 적용 (기존 동작 유지)
        u_base_for_net = u_base.detach(
        ) if self.detach_u_for_ctrl_losses else u_base

        # --- token-wise 모듈들: 전체 텐서에 한 번에 적용 ---
        # feat_prev: (B, Pnn, T, self._Dx)
        feat_prev = self.state_prev_encoder(x_prev)
        # feat_fut: (B, Pnn, T, self._Dx)
        feat_fut = self.state_fut_encoder(x_fut)
        # feat_u: (B, Pnn, T, self._Du)
        feat_u = self.control_adapter(u_base_for_net)

        # --- 무효 구간은 0으로 고정(출력/역전파 신호 차단) ---
        mask_feat = seg_mask_1.to(dtype=feat_prev.dtype,
                                  device=feat_prev.device)  # (B,Pnn,T,1)
        feat_prev = feat_prev * mask_feat
        feat_fut = feat_fut * mask_feat
        feat_u = feat_u * mask_feat

        # trunk: (B, Pnn, self._Dc)
        trunk = self.trunk_compressor(dit_final_hidden_tokens)
        # trunk_rep: (B, Pnn, T, self._Dc)
        trunk_rep = trunk.unsqueeze(2).expand(-1, -1, int(x_prev.size(2)), -1)

        # Z_in: (B, Pnn, T, self._Din)
        Z_in = torch.cat([feat_prev, feat_fut, feat_u, trunk_rep], dim=-1)
        return Z_in

    def _split_midpoint_nodes_and_controls(
        self,
        unnorm_points_xyyaw: torch.Tensor,  # (B, Pnn, point_len, 4)
        unnorm_points_world_control: torch.Tensor,  # (B, Pnn, point_len, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """중점 제어 계산 전, 포인트 단위 성분을 분해한다.

        Returns:
            cos_all:   (B, Pnn, point_len)
            sin_all:   (B, Pnn, point_len)
            v_x_all:   (B, Pnn, point_len)
            v_y_all:   (B, Pnn, point_len)
            omega_all: (B, Pnn, point_len)
        """
        cos_all = unnorm_points_xyyaw[..., 2]  # (B, Pnn, point_len)
        sin_all = unnorm_points_xyyaw[..., 3]  # (B, Pnn, point_len)

        v_x_all = unnorm_points_world_control[..., 0]  # (B, Pnn, point_len)
        v_y_all = unnorm_points_world_control[..., 1]  # (B, Pnn, point_len)
        omega_all = unnorm_points_world_control[..., 2]  # (B, Pnn, point_len)

        return cos_all, sin_all, v_x_all, v_y_all, omega_all

    def _prepare_tcn_input(
        self,
        Z_in: torch.Tensor,  # (B, Pnn, T, Din)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)  float(0/1)
    ) -> torch.Tensor:
        """Stem 실행 후 무효 구간을 0으로 고정합니다(전체 한번에 계산 + 마스크).

        예전:
            - 유효 토큰만 뽑아서 stem_norm/stem_fc/stem_act 실행 후 다시 채움.

        이제:
            - (B,Pnn,T,·) 전체에 stem을 한 번에 적용
            - 마지막에 seg_mask_1로 곱해 무효 구간 출력은 0으로 고정

        Args:
            Z_in: (B, Pnn, T, Din)
            seg_mask_1: (B, Pnn, T, 1) 0/1

        Returns:
            Z_s: (B, Pnn, T, self._C)
        """
        # stem: token-wise 연산이므로 전체 텐서에 바로 적용
        Z = self.stem_norm(Z_in)  # (B,Pnn,T,Din)
        Z = self.stem_fc(Z)  # (B,Pnn,T,C)
        Z = self.stem_act(Z)  # (B,Pnn,T,C)

        # 무효 구간은 0 고정
        mask = seg_mask_1.to(dtype=Z.dtype, device=Z.device)  # (B,Pnn,T,1)
        Z_s = Z * mask
        return Z_s

    def _run_tcn(
        self,
        Z_s: torch.Tensor,  # (B, Pnn, T, C)
        seg_mask: torch.Tensor,  # (B, Pnn, T) 0/1
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)  (호환용, 내부에서는 seg_mask 사용)
    ) -> torch.Tensor:
        """TCN 블록 실행(토큰별 packed 제거, 마스크 기반 0 고정 유지).

        변경점:
            - pre LayerNorm을 유효 토큰만 뽑아서 계산하던 방식을 제거.
            - 전체를 한 번에 LayerNorm하고,
              depthwise conv는 기존처럼 mask를 사용해 "무효 구간이 섞이지 않게" 처리.
            - 블록 끝에서 (Z + Y) * mask 로 무효 구간을 0으로 고정(기존 유지).

        Args:
            Z_s: (B, Pnn, T, C)
            seg_mask: (B, Pnn, T) 0/1
            seg_mask_1: (B, Pnn, T, 1)  (호환용)

        Returns:
            out: (B, Pnn, T, C)
        """
        B, Pnn, T, C = Z_s.shape
        B_Pnn = int(B * Pnn)

        # seg_mask_bool: (B,Pnn,T)
        seg_mask_bool = self._to_bool_mask(seg_mask)

        # conv/residual에 쓸 float 마스크: (B*Pnn, 1, T)
        seg_mask_float = seg_mask_bool.to(dtype=Z_s.dtype,
                                          device=Z_s.device)  # (B,Pnn,T)
        seg_mask_flat = seg_mask_float.reshape(B_Pnn, int(T))  # (B*Pnn,T)
        mask_b1t = seg_mask_flat.unsqueeze(1)  # (B*Pnn,1,T)

        # Z: (B*Pnn, C, T)
        Z = Z_s.reshape(B_Pnn, int(T), int(C)).transpose(1, 2)  # (B*Pnn,C,T)

        for block_idx in range(self.tcn_depth):
            # (B*Pnn, T, C)
            Z_tc = Z.transpose(1, 2)

            # pre LayerNorm: 전체 한번에 계산
            Z_ln_tc = self.tcn_pre_lns[int(block_idx)](Z_tc)  # (B*Pnn,T,C)

            # depthwise conv: 기존대로 mask를 고려한 conv (무효가 섞이지 않음)
            Z_ln = Z_ln_tc.transpose(1, 2)  # (B*Pnn,C,T)
            Y = self._depthwise_conv_masked_bct(
                x_bct=Z_ln,
                mask_b1t=mask_b1t,
                block_idx=int(block_idx),
            )  # (B*Pnn,C,T)

            Y = F.gelu(Y)
            Y = self.tcn_linear[int(block_idx)](Y)  # (B*Pnn,C,T)

            # Residual + 무효 구간 0 고정(기존 유지)
            Z = (Z + Y) * mask_b1t

        out = Z.transpose(1, 2).reshape(B, Pnn, int(T), int(C))
        return out

    def _masked_time_mean(
        self,
        seq_bptc: torch.Tensor,  # (B, Pnn, T, C)
        seg_mask_1: torch.Tensor,  # (B, Pnn, T, 1)  float(0/1) 또는 bool
        *,
        eps: float,
    ) -> torch.Tensor:
        """시간축 평균을 '유효 구간'만 대상으로 계산합니다.

        Args:
            seq_bptc (torch.Tensor):
                shape: (B, Pnn, T, C)
                시간축(T)을 가진 값입니다.
            seg_mask_1 (torch.Tensor):
                shape: (B, Pnn, T, 1)
                유효 구간은 1, 무효 구간은 0인 마스크입니다.
            eps (float):
                0으로 나눔을 피하기 위한 작은 값입니다.

        Returns:
            torch.Tensor:
                shape: (B, Pnn, C)
                유효 구간만 평균낸 값입니다.
                유효 구간이 0개인 경우는 0을 반환합니다.
        """
        if seq_bptc.dim() != 4:
            raise ValueError(
                "_masked_time_mean: seq_bptc는 (B,Pnn,T,C) 4D 텐서여야 합니다. "
                f"got shape={tuple(seq_bptc.shape)}")
        if seg_mask_1.dim() != 4:
            raise ValueError(
                "_masked_time_mean: seg_mask_1은 (B,Pnn,T,1) 4D 텐서여야 합니다. "
                f"got shape={tuple(seg_mask_1.shape)}")
        if seq_bptc.shape[:3] != seg_mask_1.shape[:3] or int(
                seg_mask_1.shape[-1]) != 1:
            raise ValueError(
                "_masked_time_mean: seq_bptc와 seg_mask_1의 (B,Pnn,T) 또는 마지막 축이 맞지 않습니다. "
                f"seq_bptc.shape={tuple(seq_bptc.shape)}, seg_mask_1.shape={tuple(seg_mask_1.shape)}"
            )

        mask = seg_mask_1.to(dtype=seq_bptc.dtype,
                             device=seq_bptc.device)  # (B,Pnn,T,1)
        seq_sum = (seq_bptc * mask).sum(dim=2)  # (B,Pnn,C)
        count = mask.sum(dim=2).clamp_min(float(eps))  # (B,Pnn,1)
        return seq_sum / count  # (B,Pnn,C)

    def _predict_delta_u(
            self,
            Z_tcn: torch.Tensor,  # (B, Pnn, T, C)
            seg_mask_1: torch.Tensor,  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """잔차 제어 ΔU 산출: Head + softplus 게이트 스케일.

        변경점:
            - Gate를 시간마다(T) 계산하지 않고,
              (B,Pnn,3) 한 번만 계산한 뒤 시간축으로 복사합니다.
            - Gate 입력은 '유효 구간'만 대상으로 Z_tcn을 시간축 평균낸 값입니다.
            - config.feasible_gate_agentwise=False 로 두면 예전 방식(시간별 Gate)로 동작합니다.

        Args:
            Z_tcn (torch.Tensor):
                shape: (B, Pnn, T, C)
            seg_mask_1 (torch.Tensor):
                shape: (B, Pnn, T, 1)

        Returns:
            torch.Tensor:
                delta_u shape: (B, Pnn, T, 3)
        """
        B, Pnn, T, C = Z_tcn.shape
        B_Pnn = int(B * Pnn)

        # seg_mask: (B,Pnn,T,1) float(0/1)
        seg_mask = seg_mask_1.to(dtype=Z_tcn.dtype, device=Z_tcn.device)

        # -----------------------
        # Head: (B*Pnn, C, T) -> (B*Pnn, 3, T)
        # -----------------------
        z_bct = Z_tcn.reshape(B_Pnn, int(T), int(C)).transpose(1,
                                                               2)  # (B*Pnn,C,T)
        delta_u_b3t = self.head(z_bct)  # (B*Pnn,3,T)

        # delta_u_raw: (B,Pnn,T,3)
        delta_u_raw = delta_u_b3t.transpose(1, 2).reshape(B, Pnn, int(T), 3)
        delta_u_raw = delta_u_raw * seg_mask  # 무효 구간은 0 고정

        # -----------------------
        # Gate
        # -----------------------
        if bool(getattr(self, "feasible_gate_agentwise", False)):
            # (B,Pnn,C): 유효 구간만으로 시간축 평균
            gate_in = self._masked_time_mean(
                seq_bptc=Z_tcn.detach(),  # (B,Pnn,T,C)
                seg_mask_1=seg_mask,  # (B,Pnn,T,1)
                eps=float(self._eps),
            )

            # (B,Pnn,3): 이웃별로 1번만 계산
            gate_logits_agent = self.gate_mlp(gate_in)

            # (B,Pnn,T,3): 시간축으로 복사
            gate_logits = gate_logits_agent.unsqueeze(2).expand(
                -1, -1, int(T), -1)
        else:
            # 기존 방식: 시간별로 Gate 계산 (B,Pnn,T,3)
            gate_logits = self.gate_mlp(Z_tcn.detach())

        # 무효 구간은 0으로 맞춰 두기(출력은 어차피 0이지만 모양/의미를 맞춤)
        gate_logits = gate_logits * seg_mask  # (B,Pnn,T,3)

        gate_scale = F.softplus(gate_logits)  # (B,Pnn,T,3)
        delta_u = gate_scale * torch.tanh(delta_u_raw)  # (B,Pnn,T,3)
        return delta_u

    def _build_per_agent_limits(
        self,
        near_class_one_hot: torch.Tensor,  # (B,Pnn,3)
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """클래스별 스칼라 제약치를 (B,Pnn) 텐서로 확장."""
        car: DynamicLimits = self.constraints[ActorClass.CAR]
        ped: DynamicLimits = self.constraints[ActorClass.PEDESTRIAN]
        bic: DynamicLimits = self.constraints[ActorClass.BICYCLE]

        def cvec(getattr_name: str) -> torch.Tensor:
            vals = torch.tensor(
                [
                    getattr(car, getattr_name),
                    getattr(ped, getattr_name),
                    getattr(bic, getattr_name)
                ],
                device=device,
                dtype=dtype,
            )  # (3,)
            return (near_class_one_hot * vals).sum(dim=-1)  # (B,Pnn)

        v_max_bp = cvec("v_max_mps")
        a_max_bp = cvec("a_max_mps2")
        alpha_max_bp = cvec("alpha_max_radps2")
        a_lat_max_bp = cvec("a_lat_max_mps2")
        R_min_bp = cvec("R_min_m")
        omega_abs_bp = cvec("omega_max_abs_radps")
        v_b_y_max_bp = cvec("v_b_y_max")
        beta_max_rad_bp = cvec("beta_max_rad")  # ✅ 추가

        a_x_max_bp = a_max_bp
        a_y_max_bp = a_lat_max_bp

        is_ped = near_class_one_hot[..., 1] > 0.5
        is_nonholonomic = ~is_ped

        return dict(
            v_max=v_max_bp,
            a_max=a_max_bp,
            alpha_max=alpha_max_bp,
            a_lat_max=a_lat_max_bp,
            R_min=R_min_bp,
            omega_abs_max=omega_abs_bp,
            a_x_max=a_x_max_bp,
            a_y_max=a_y_max_bp,
            v_b_y_max=v_b_y_max_bp,
            beta_max_rad=beta_max_rad_bp,  # ✅ 추가
            is_nonholonomic=is_nonholonomic,
        )

    # ----------------------------
    # [MOD] (S3) 옆가속/최소R/절대|w| (중점 속도 사용)
    # ----------------------------

    # ----------------------------
    # [NEW] 입력 분해 및 버퍼 초기화
    # ----------------------------
    def _split_controls(
        self,
        unnorm_cur_future_seg_body_control: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """제어 텐서를 성분별로 분리.

        Args:
            unnorm_cur_future_seg_body_control (torch.Tensor): (B, Pnn, T, 3) [vxb, vyb, w]

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (vxb_raw, vyb_raw, omega_raw), 모두 (B,Pnn,T)
        """
        vx_b_raw = unnorm_cur_future_seg_body_control[..., 0]
        vy_b_raw = unnorm_cur_future_seg_body_control[..., 1]
        omega_raw = unnorm_cur_future_seg_body_control[..., 2]
        return vx_b_raw, vy_b_raw, omega_raw

    def _init_integration_buffers(
        self,
        B: int,
        Pnn: int,
        T: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """적분 산출/중간 결과 버퍼 생성."""
        x_next, y_next, cos_next, sin_next = _shape4(B, Pnn, T, device, dtype)
        vx_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        vy_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        omega_after = torch.zeros((B, Pnn, T), device=device, dtype=dtype)
        return {
            "x_next": x_next,
            "y_next": y_next,
            "cos_next": cos_next,
            "sin_next": sin_next,
            "vx_after": vx_after,
            "vy_after": vy_after,
            "omega_after": omega_after,
        }

    # === 아래 메서드들을 FeasibleProjector 클래스 "내부"에 추가하세요. ===

    # ----------------------------
    # [UTIL] 각도/벡터 보조 함수들
    # ----------------------------

    @staticmethod
    def _compute_mid_heading_from_cos_sin(
            cos_yaw_k: torch.Tensor,  # (...,)
            sin_yaw_k: torch.Tensor,  # (...,)
            half_delta_theta: torch.Tensor,  # (...,)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """중점(head)의 cos/sin 계산: θ_mid = θ_k + 0.5 * Δθ.

        Args:
            cos_yaw_k (torch.Tensor): (...,) cos(θ_k)
            sin_yaw_k (torch.Tensor): (...,) sin(θ_k)
            half_delta_theta (torch.Tensor): (...,) 0.5 * Δθ

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos_mid, sin_mid)  (둘 다 (...,))
        """
        c = torch.cos(half_delta_theta)
        s = torch.sin(half_delta_theta)
        cos_mid = cos_yaw_k * c - sin_yaw_k * s
        sin_mid = sin_yaw_k * c + cos_yaw_k * s
        return cos_mid, sin_mid

    @staticmethod
    def _advance_heading_cos_sin(
        cos_yaw_k: torch.Tensor,  # (...,)
        sin_yaw_k: torch.Tensor,  # (...,)
        delta_theta: torch.Tensor,  # (...,)
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """θ_{k+1} = θ_k + Δθ 를 (cos, sin)로 직접 업데이트하고 정규화.

        Args:
            cos_yaw_k (torch.Tensor): (...,) cos(θ_k)
            sin_yaw_k (torch.Tensor): (...,) sin(θ_k)
            delta_theta (torch.Tensor): (...,) Δθ
            eps (float): 수치 안정 epsilon

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (cos_{k+1}, sin_{k+1})
        """
        c = torch.cos(delta_theta)
        s = torch.sin(delta_theta)
        cos_next = cos_yaw_k * c - sin_yaw_k * s
        sin_next = sin_yaw_k * c + cos_yaw_k * s
        norm = torch.sqrt(cos_next * cos_next + sin_next * sin_next + eps)
        return cos_next / norm, sin_next / norm

    def _apply_S0_sideslip_angle_limit_ste(
            self,
            vx_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
            vy_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
            beta_max_rad: torch.Tensor,  # (B,Pnn)
            eta: float,
            eps: float,
            is_nonholonomic: torch.Tensor,  # (B,Pnn) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(S0-β) 사이드슬립 각(β) 상한을 만족하도록 v_y^b만 줄입니다.

        여기서 β는 “차량의 heading(바디 x축)과 속도 벡터 방향의 차이”로 보고,
        v_x^b의 부호와 무관하게 |v_x^b|를 사용합니다.

        목표:
            |β| = atan2(v_y^b, |v_x^b|) <= beta_max_rad

        위 조건을 v_y^b에 대한 형태로 바꾸면(β_max < 90도 가정):
            |v_y^b| <= (|v_x^b| + eps) * tan(beta_max_rad)

        구현 규칙:
            - v_y^b만 클리핑해서(크기만 줄여서) 상한을 만족시킵니다.
            - v_x^b는 그대로 둡니다.
            - beta_max_rad <= 0 인 경우는 “비활성”로 보고 적용하지 않습니다.
            - 보행자처럼 nonholonomic이 아닌 대상에는 적용하지 않습니다.

        Args:
            vx_b: (B,Pnn) 또는 (B,Pnn,T) 바디 x방향 속도.
            vy_b: (B,Pnn) 또는 (B,Pnn,T) 바디 y방향 속도.
            beta_max_rad: (B,Pnn) 라디안 단위 상한. 0 이하면 비활성.
            eta: STE 밴드 폭(기존 제약들과 동일한 역할).
            eps: 작은 값(0 나눗셈/수치 안정용).
            is_nonholonomic: (B,Pnn) bool. True인 대상에만 적용.

        Returns:
            (vx_b_out, vy_b_out):
                vx_b_out: 입력과 동일(변경 없음)
                vy_b_out: β 상한을 만족하도록 클리핑된 v_y^b
        """
        # beta_max_rad: (B,Pnn) -> dtype/device 정렬
        beta = beta_max_rad.to(dtype=vy_b.dtype, device=vy_b.device)

        # 시간축(T)이 있으면 (B,Pnn,1)로 늘려서 브로드캐스트
        if beta.dim() == vy_b.dim() - 1:
            beta = beta.unsqueeze(-1)  # (B,Pnn,1)

        # 활성 여부: beta_max_rad > 0 인 경우만 적용
        enabled = (beta_max_rad > 0.0)
        if enabled.dim() == vy_b.dim() - 1:
            enabled = enabled.unsqueeze(-1).expand_as(
                vy_b)  # (B,Pnn,T) or (B,Pnn)

        # nonholonomic 마스크도 시간축이 있으면 확장
        nonh = is_nonholonomic
        if nonh.dim() == vy_b.dim() - 1:
            nonh = nonh.unsqueeze(-1).expand_as(vy_b)  # (B,Pnn,T) or (B,Pnn)

        active = nonh & enabled  # (B,Pnn) or (B,Pnn,T)

        # beta 값은 음수면 의미가 없으니 0으로 올림
        beta = beta.clamp_min(0.0)

        # tan(beta_max)
        tan_beta = torch.tan(beta)

        # v_y 상한: (|v_x| + eps) * tan(beta_max)
        vx_abs = vx_b.abs()
        vy_limit = (vx_abs + float(eps)) * tan_beta  # (B,Pnn) or (B,Pnn,T)

        # 혹시 비정상 값이 생기면 안전하게 처리
        vy_limit = torch.where(
            torch.isfinite(vy_limit),
            vy_limit,
            torch.full_like(vy_limit, float("inf")),
        )

        # v_y만 STE-클립
        vy_new = self._ste_scalar_clip(vy_b, vy_limit, eta, eps)

        # 적용 대상(active)만 교체
        vy_out = torch.where(active, vy_new, vy_b)
        return vx_b, vy_out

    # =========================================================
    # [S0~S4] 제약 적용: **STE 버전** (forward=hard, backward=surrogate)
    # =========================================================
    def _apply_S0_nonholonomic_ste(
            self,
            vx_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
            vy_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
            v_b_y_max: torch.Tensor,  # (B,Pnn)
            eta: float,
            eps: float,
            is_nonholonomic: torch.Tensor,  # (B,Pnn) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S0: (차/자전거) 바디 y방향 속도를 v_b_y_max로 제한한다."""
        v_b_y_max = v_b_y_max.to(dtype=vy_b.dtype,
                                 device=vy_b.device)  # (B,Pnn)

        # 시간축(T)이 있으면 (B,Pnn,1)로 늘려서 브로드캐스트
        if v_b_y_max.dim() == vy_b.dim() - 1:
            v_b_y_max = v_b_y_max.unsqueeze(-1)  # (B,Pnn,1)

        vy_new = self._ste_scalar_clip(vy_b, v_b_y_max, eta, eps)

        # is_nonholonomic도 시간축이 있으면 맞춰서 늘림
        if is_nonholonomic.dim() == vy_b.dim() - 1:
            is_nonholonomic = is_nonholonomic.unsqueeze(-1).expand_as(vy_b)

        vy_out = torch.where(is_nonholonomic, vy_new, vy_b)
        return vx_b, vy_out

    def _apply_S1_speed_limit_ste(
            self, vx_b: torch.Tensor, vy_b: torch.Tensor, v_max: torch.Tensor,
            eta: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """라디얼(벡터) STE-clip."""
        v_max = v_max.to(dtype=vx_b.dtype, device=vx_b.device)
        # 추가하자: 시간축(T)이 있는 경우 v_max 를 (B,Pnn,1) 로 확장해 T에 브로드캐스트
        if v_max.dim() == vx_b.dim() - 1:
            v_max = v_max.unsqueeze(-1)  # (B,Pnn,1)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        s_hard = torch.clamp(v_max / speed.clamp_min(eps), max=1.0)  # (B,Pnn)
        vx_h, vy_h = s_hard * vx_b, s_hard * vy_b  # (B,Pnn)
        r = speed / (v_max + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        vx_sur, vy_sur = w * vx_b + (1 - w) * vx_b.detach(), w * vy_b + (
            1 - w) * vy_b.detach()
        vx_out = vx_sur + (vx_h - vx_sur).detach()
        vy_out = vy_sur + (vy_h - vy_sur).detach()
        return vx_out, vy_out

    def _apply_S2_accel_alpha_limits_ste(
            self, vx_b_prev: torch.Tensor, vy_b_prev: torch.Tensor,
            omega_prev: torch.Tensor, vx_b: torch.Tensor, vy_b: torch.Tensor,
            omega: torch.Tensor, a_max: torch.Tensor, alpha_max: torch.Tensor,
            dt: float, eta: float,
            eps: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 선형 속도 증분(벡터)
        dv = torch.stack([vx_b - vx_b_prev, vy_b - vy_b_prev],
                         dim=-1)  # (B,Pnn,2)
        dv_limit = (a_max * dt).to(dtype=dv.dtype, device=dv.device)  # (B,Pnn)
        # dv_ste: (B,Pnn,2)
        dv_ste = self._ste_increment_vec(dv, dv_limit, eta, eps)
        vx_b_new = vx_b_prev + dv_ste[..., 0]  # (B,Pnn)
        vy_b_new = vy_b_prev + dv_ste[..., 1]  # (B,Pnn)
        # 각속도 증분(스칼라)
        dw = omega - omega_prev  # (B,Pnn)
        dw_limit = (alpha_max * dt).to(dtype=dw.dtype,
                                       device=dw.device)  # (B,Pnn)
        dw_ste = self._ste_scalar_clip(dw, dw_limit, eta, eps)  # (B,Pnn)
        w_new = omega_prev + dw_ste
        return vx_b_new, vy_b_new, w_new

    def _apply_S2_accel_alpha_limits_ste_batch(
        self,
        vx_b: torch.Tensor,  # (B, Pnn, T)
        vy_b: torch.Tensor,  # (B, Pnn, T)
        omega: torch.Tensor,  # (B, Pnn, T)
        a_max: torch.Tensor,  # (B, Pnn)
        alpha_max: torch.Tensor,  # (B, Pnn)
        dt: float,
        eta: float,
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(S2) 가속도/각가속도 증분 제약을 시간축 전체 배치 연산으로 적용합니다.

        목표(이산 형태):
            k>=1 에 대해
              || [vx_k - vx_{k-1}, vy_k - vy_{k-1}] || <= a_max * dt
              |  omega_k - omega_{k-1} | <= alpha_max * dt

        구현 방식(시간축 for-loop 없음):
            1) dv, dw를 diff로 한 번에 계산 (T-1 길이)
            2) dv는 2D 노름 기준, dw는 스칼라 기준으로 STE-hard clip
            3) cumsum으로 다시 vx/vy/omega 시퀀스를 복원
               - 첫 시점(k=0)은 그대로 유지합니다(기존 sequential에서 k=0에 S2를 안 거는 것과 동일한 의도).

        Args:
            vx_b: (B, Pnn, T)
            vy_b: (B, Pnn, T)
            omega: (B, Pnn, T)
            a_max: (B, Pnn)  종/횡을 묶은 “속도 변화량” 상한에 쓰는 값 (m/s^2)
            alpha_max: (B, Pnn)  요각속도 변화 상한에 쓰는 값 (rad/s^2)
            dt: float  샘플 간 시간 간격(초)
            eta: float  STE 밴드 폭
            eps: float  수치 안정용

        Returns:
            (vx_out, vy_out, omega_out):
                모두 (B, Pnn, T)
        """
        if vx_b.dim() != 3 or vy_b.dim() != 3 or omega.dim() != 3:
            raise ValueError(
                "[_apply_S2_accel_alpha_limits_ste_batch] 입력은 (B,Pnn,T) 3D 텐서여야 합니다."
            )
        if vx_b.shape != vy_b.shape or vx_b.shape != omega.shape:
            raise ValueError(
                "[_apply_S2_accel_alpha_limits_ste_batch] vx/vy/omega shape이 서로 다릅니다. "
                f"vx={tuple(vx_b.shape)}, vy={tuple(vy_b.shape)}, omega={tuple(omega.shape)}"
            )

        B, Pnn, T = vx_b.shape
        if T <= 1:
            return vx_b, vy_b, omega

        # ------------------------
        # 1) 선형 속도 증분 dv: (B,Pnn,T-1,2)
        # ------------------------
        dvx = vx_b[..., 1:] - vx_b[..., :-1]  # (B,Pnn,T-1)
        dvy = vy_b[..., 1:] - vy_b[..., :-1]  # (B,Pnn,T-1)
        dv = torch.stack([dvx, dvy], dim=-1)  # (B,Pnn,T-1,2)

        dv_limit = (a_max.to(dtype=vx_b.dtype, device=vx_b.device) * float(dt)
                   )  # (B,Pnn)
        dv_limit_bpt = dv_limit.unsqueeze(-1).expand(B, Pnn,
                                                     T - 1)  # (B,Pnn,T-1)

        dv_ste = self._ste_increment_vec_nd(
            dv=dv,  # (B,Pnn,T-1,2)
            limit=dv_limit_bpt,  # (B,Pnn,T-1)
            eta=eta,
            eps=eps,
        )  # (B,Pnn,T-1,2)

        # cumsum으로 복원
        dv_prefix = torch.cumsum(dv_ste, dim=2)  # (B,Pnn,T-1,2)

        vx0 = vx_b[..., :1]  # (B,Pnn,1)
        vy0 = vy_b[..., :1]  # (B,Pnn,1)

        vx_tail = vx0 + dv_prefix[..., 0]  # (B,Pnn,T-1)
        vy_tail = vy0 + dv_prefix[..., 1]  # (B,Pnn,T-1)

        vx_out = torch.cat([vx0, vx_tail], dim=2)  # (B,Pnn,T)
        vy_out = torch.cat([vy0, vy_tail], dim=2)  # (B,Pnn,T)

        # ------------------------
        # 2) 요각속도 증분 dw: (B,Pnn,T-1)
        # ------------------------
        dw = omega[..., 1:] - omega[..., :-1]  # (B,Pnn,T-1)

        dw_limit = (alpha_max.to(dtype=omega.dtype, device=omega.device) *
                    float(dt))  # (B,Pnn)
        dw_limit_bpt = dw_limit.unsqueeze(-1).expand(B, Pnn,
                                                     T - 1)  # (B,Pnn,T-1)

        dw_ste = self._ste_scalar_clip(
            x=dw,  # (B,Pnn,T-1)
            limit=dw_limit_bpt,  # (B,Pnn,T-1)
            eta=eta,
            eps=eps,
        )  # (B,Pnn,T-1)

        dw_prefix = torch.cumsum(dw_ste, dim=2)  # (B,Pnn,T-1)

        w0 = omega[..., :1]  # (B,Pnn,1)
        w_tail = w0 + dw_prefix  # (B,Pnn,T-1)
        omega_out = torch.cat([w0, w_tail], dim=2)  # (B,Pnn,T)

        return vx_out, vy_out, omega_out

    # [추가 요망] (S3: 속도-연동 각속도 한계 — w clip, no slip angle)
    def _apply_S3_omega_clip_ste(
        self,
        vx_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
        vy_b: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
        omega: torch.Tensor,  # (B,Pnn) or (B,Pnn,T)
        a_lat_max: torch.Tensor,  # (B,Pnn)
        R_min: torch.Tensor,  # (B,Pnn)
        omega_abs_max: torch.Tensor,  # (B,Pnn)
        is_nonholonomic: torch.Tensor,  # (B,Pnn) bool
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """(S3) 속도-연동 omega 한계로 직접 clip.

        변경점(요청 반영):
            - R_min 기반 항(= r = v/|omega| 제약에서 나온 항)에서
              v 대신 |v_x^b| 를 사용합니다.
            - 즉, |omega| <= |v_x^b| / R_min
        """
        # 기존 speed는 a_lat_max 항에 계속 사용(그대로 유지)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b +
                           eps)  # (B,Pnn) or (B,Pnn,T)

        # ✅ r 제약에 쓰는 v는 |v_x^b|
        v_x_abs = vx_b.abs()  # (B,Pnn) or (B,Pnn,T)

        if a_lat_max.dim() == speed.dim() - 1:
            a_lat_max = a_lat_max.unsqueeze(-1)  # (B,Pnn,1)
            R_min = R_min.unsqueeze(-1)  # (B,Pnn,1)
            omega_abs_max = omega_abs_max.unsqueeze(-1)  # (B,Pnn,1)

        allow_lat = a_lat_max / (speed + eps)
        # ✅ 변경: speed 대신 |v_x^b|
        allow_R = v_x_abs / (R_min + eps)
        allow_abs = omega_abs_max

        allow_nonh = torch.minimum(torch.minimum(allow_lat, allow_R), allow_abs)
        allow_holo = allow_abs  #torch.minimum(allow_lat, allow_abs)

        if is_nonholonomic.dim() == allow_nonh.dim() - 1:
            is_nonholonomic = is_nonholonomic.unsqueeze(-1).expand_as(
                allow_nonh)

        allow = torch.where(is_nonholonomic, allow_nonh, allow_holo)
        omega_new = self._ste_yawrate_clip(omega, allow, eta=eta, eps=eps)
        return omega_new

    # [추가 요망] (S4: β 미사용, Δv와 w를 동일 스케일 s로 동시 축소)
    def _apply_S4_friction_circle_ste(
        self,
        vx_b_prev: torch.Tensor,
        vy_b_prev: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        a_x_max: torch.Tensor,
        a_y_max: torch.Tensor,  # (B,Pnn)
        dt: float,
        eta: float,
        eps: float,
        use_ax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(S4) (a_x/a_x,max)^2+(a_y/a_y,max)^2≤1 를 만족하도록
        Δv와 w를 동일 스케일 s∈(0,1]로 동시 축소.
           a_x ≈ Δv/dt,  a_y ≈ v_mid * w_k
        """
        # [추가하세요!] 바디 프레임 기준 중간 속도/증분 (부호 보존)
        vxb_mid = 0.5 * (vx_b_prev + vx_b_k)  # (B,Pnn)
        vyb_mid = 0.5 * (vy_b_prev + vy_b_k)  # (B,Pnn)
        v_mid = torch.sqrt(vxb_mid * vxb_mid + vyb_mid * vyb_mid +
                           eps)  # (B,Pnn)

        dvx = vx_b_k - vx_b_prev  # (B,Pnn)
        dvy = vy_b_k - vy_b_prev  # (B,Pnn)
        if not use_ax:
            dvx = torch.zeros_like(dvx)  # k=0 등에서는 종가속 항 비활성
        ax_est = dvx / dt  # (B,Pnn)  ← 부호 보존 종가속도
        ay_est = v_mid * omega_k  # (B,Pnn)  ← 곡률 유도 횡가속도
        # ay_est = ay_est + (dvy / dt)
        s = self._ste_friction_scale(ax_est,
                                     ay_est,
                                     a_x_max,
                                     a_y_max,
                                     eta=eta,
                                     eps=eps)  # (B,Pnn)
        # [추가하세요!] Δv 벡터 자체와 ω를 같은 비율로 축소 → 방향/부호 그대로 유지
        dv = torch.stack([dvx, dvy], dim=-1)  # (B,Pnn,2)
        dv_sc = s.unsqueeze(-1) * dv  # (B,Pnn,2)
        vx_b_new = vx_b_prev + dv_sc[..., 0]  # (B,Pnn)
        vy_b_new = vy_b_prev + dv_sc[..., 1]  # (B,Pnn)
        omega_new = s * omega_k  # (B,Pnn)
        return vx_b_new, vy_b_new, omega_new

    @staticmethod
    def _ste_increment_vec_nd(
        dv: torch.Tensor,  # (..., 2)
        limit: torch.Tensor,  # (...,)
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """증분(2D 벡터) 노름 제한을 STE 방식으로 적용합니다.

        forward(값 계산):
            - dv의 크기(2D 노름)가 limit보다 크면, limit에 맞게 같은 비율로 줄입니다.
            - dv의 방향(부호/방향)은 유지합니다.

        backward(학습 기울기):
            - 기존 코드의 밴드(eta) 규칙을 그대로 사용해
              limit 근처에서만 기울기가 자연스럽게 흐르도록 만듭니다.

        Args:
            dv (torch.Tensor):
                shape: (..., 2)
                연속한 두 시점의 속도 변화량입니다. 마지막 축 2는 [dvx, dvy]입니다.
            limit (torch.Tensor):
                shape: dv.shape[:-1]
                각 위치별 허용 변화량 크기(예: a_max*dt)입니다.
            eta (float):
                STE 밴드 폭(기존과 동일한 의미).
            eps (float):
                0 나눗셈 방지용 작은 값.

        Returns:
            torch.Tensor:
                shape: (..., 2)
                노름 제한이 적용된 변화량.
        """
        if limit.shape != dv.shape[:-1]:
            raise ValueError(
                "[_ste_increment_vec_nd] limit.shape 와 dv.shape[:-1]가 다릅니다. "
                f"limit.shape={tuple(limit.shape)}, dv.shape={tuple(dv.shape)}")

        limit_f = limit.to(dtype=dv.dtype, device=dv.device)  # (...,)
        limit_f_ex = limit_f.unsqueeze(-1)  # (..., 1)

        norm = torch.linalg.norm(dv, dim=-1,
                                 keepdim=True).clamp_min(eps)  # (..., 1)
        s_hard = torch.clamp(limit_f_ex / norm, max=1.0)  # (..., 1)
        dv_hard = s_hard * dv  # (..., 2)

        r = (norm.squeeze(-1)) / (limit_f + eps)  # (...,)
        w = FeasibleProjector._ste_band_weight(r, eta).unsqueeze(-1)  # (..., 1)

        dv_sur = w * dv + (1.0 - w) * dv.detach()  # (..., 2)
        dv_out = dv_sur + (dv_hard - dv_sur).detach()  # (..., 2)
        return dv_out

    # ----------------------------
    # [NEW] 한 스텝: 제약 S0~S4 적용(모두 STE 버전 호출)
    # ----------------------------
    def _apply_constraints_step(
        self,
        vx_b_prev: torch.Tensor,
        vy_b_prev: torch.Tensor,
        omega_prev: torch.Tensor,
        vx_b_k: torch.Tensor,
        vy_b_k: torch.Tensor,
        omega_k: torch.Tensor,
        hp: _ConstraintHParams,
        key_to_limit_bp: Dict[str, torch.Tensor],
        apply_S2: bool = True,
        apply_S4_ax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ✅ 추가: 사이드슬립 각(β) 상한 → v_y^b만 줄이기
        vx_b_k, vy_b_k = self._apply_S0_sideslip_angle_limit_ste(
            vx_b=vx_b_k,
            vy_b=vy_b_k,
            beta_max_rad=key_to_limit_bp["beta_max_rad"],
            eta=hp.eta_slip,
            eps=hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
        )
        # (S1)
        vx_b_k, vy_b_k = self._apply_S1_speed_limit_ste(
            vx_b_k, vy_b_k, key_to_limit_bp["v_max"], hp.eta_speed, hp.eps)

        # (S2)
        if apply_S2:
            vx_b_k, vy_b_k, omega_k = self._apply_S2_accel_alpha_limits_ste(
                vx_b_prev, vy_b_prev, omega_prev, vx_b_k, vy_b_k, omega_k,
                key_to_limit_bp["a_max"], key_to_limit_bp["alpha_max"], hp.dt,
                hp.eta_inc, hp.eps)

        # (S3)
        omega_k = self._apply_S3_omega_clip_ste(
            vx_b=vx_b_k,
            vy_b=vy_b_k,
            omega=omega_k,
            a_lat_max=key_to_limit_bp["a_lat_max"],
            R_min=key_to_limit_bp["R_min"],
            omega_abs_max=key_to_limit_bp["omega_abs_max"],
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
            eta=hp.eta_yaw,
            eps=hp.eps)

        return vx_b_k, vy_b_k, omega_k

    # ----------------------------
    # [NEW] 한 스텝: 중점 적분 (변경 없음)
    # ----------------------------
    def _integrate_midpoint_step(
        self,
        x_k: torch.Tensor,  # (B,Pnn)
        y_k: torch.Tensor,  # (B,Pnn)
        cos_yaw_k: torch.Tensor,  # (B,Pnn)
        sin_yaw_k: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,  # (B,Pnn)
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        hp: _ConstraintHParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_dtheta = 0.5 * omega_k * hp.dt
        # cos_mid, sin_mid: (B,Pnn)
        cos_mid, sin_mid = self._compute_mid_heading_from_cos_sin(
            cos_yaw_k, sin_yaw_k, half_dtheta)
        vwx_mid = cos_mid * vx_b_k - sin_mid * vy_b_k
        vwy_mid = sin_mid * vx_b_k + cos_mid * vy_b_k
        x_k1 = x_k + vwx_mid * hp.dt  # (B,Pnn)
        y_k1 = y_k + vwy_mid * hp.dt  # (B,Pnn)
        dtheta = omega_k * hp.dt  # (B,Pnn)
        cos_yaw_k1, sin_yaw_k1 = self._advance_heading_cos_sin(cos_yaw_k,
                                                               sin_yaw_k,
                                                               dtheta,
                                                               eps=hp.eps)
        return x_k1, y_k1, cos_yaw_k1, sin_yaw_k1

    # ----------------------------
    # [NEW] 결과 조립(+마스크/차이)
    # ----------------------------
    def _assemble_outputs(
            self,
            key_to_all_states: Dict[str, torch.Tensor],
            vx_b_raw: torch.Tensor,  # (B,Pnn,T)
            vy_b_raw: torch.Tensor,  # (B,Pnn,T)
            omega_raw: torch.Tensor,  # (B,Pnn,T)
            near_cur_future_valid: torch.Tensor,  # (B,Pnn,1+T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """시점별 유효성(노드/구간 마스크)을 반영해 출력 텐서를 구성합니다.

        Args:
            key_to_all_states: 적분 및 제약 후 상태/제어 모음.
                - x_next / y_next / cos_next / sin_next: (B,Pnn,T)  ← 노드(t=1..T)
                - vx_after / vy_after / omega_after:     (B,Pnn,T)  ← 구간 평균 제어
            vx_b_raw, vy_b_raw, omega_raw: (B,Pnn,T)  필터 입력 제어(비정규화)
            near_cur_future_valid: (B,Pnn,1+T)  노드 유효성. t=0(현재) 포함.

        Returns:
            Tuple:
                - unnorm_integrated_trajectory: (B,Pnn,future_len,4)  # 노드 기반 → mask_node[...,1:]로 마스킹
                - unnorm_control_constraint_diff: (B,Pnn,future_len,3)  # 구간 기반 → mask_interval로 마스킹
        """
        # --- 상태 조립(노드 기반) ---
        x_next, y_next = key_to_all_states["x_next"], key_to_all_states[
            "y_next"]
        cos_next, sin_next = key_to_all_states["cos_next"], key_to_all_states[
            "sin_next"]
        unnorm_integrated_trajectory = torch.stack(
            [x_next, y_next, cos_next, sin_next], dim=-1)  # (B,Pnn,T,4)

        # --- 제어 차이(구간 기반, detach로 필터 역전파 차단) ---
        unnorm_control_constraint_diff = torch.stack(
            [
                key_to_all_states["vx_after"].detach() - vx_b_raw,
                key_to_all_states["vy_after"].detach() - vy_b_raw,
                key_to_all_states["omega_after"].detach() - omega_raw,
            ],
            dim=-1,  # (B,Pnn,T,3)
        )

        # --- 시점별 마스크 구성 ---
        mask_node = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        mask_state = mask_node[..., 1:]  # (B,Pnn,T)    # 노드 t=1..T
        mask_interval = mask_node[..., :-1] & mask_node[
            ..., 1:]  # (B,Pnn,T)    # 구간 AND

        # --- 마스킹 적용 ---
        unnorm_integrated_trajectory = unnorm_integrated_trajectory.masked_fill(
            ~mask_state.unsqueeze(-1), 0.0)
        unnorm_control_constraint_diff = unnorm_control_constraint_diff.masked_fill(
            ~mask_interval.unsqueeze(-1), 0.0)
        return unnorm_integrated_trajectory, unnorm_control_constraint_diff

    # ----------------------------
    # [UTIL] 속도/증분 soft & hard 클립
    # ----------------------------

    # ==========================================
    # 4) S2 (증분 제한) soft clip 수정
    # ==========================================

    def _filter_and_integrate_sequential(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
            unnorm_cur_future_seg_body_control: torch.Tensor,
            # (B, Pnn, future_len, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Pnn, future_len, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype

        if future_len == 0:
            raise ValueError("future_len=0: 적분할 미래 세그먼트가 없습니다.")

        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot, device=device, dtype=dtype)

        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(
            unnorm_cur_future_seg_body_control)  # (B,Pnn,T) 각각

        # 초기 상태 (B,Pnn)
        x_k = unnorm_near_current_state[..., 0]
        y_k = unnorm_near_current_state[..., 1]
        cos_yaw_k = unnorm_near_current_state[..., 2]
        sin_yaw_k = unnorm_near_current_state[..., 3]

        vx_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)
        vy_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)
        omega_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)

        # ✅ in-place 버퍼 저장 대신, 리스트에 쌓아서 마지막에 stack
        x_list: List[torch.Tensor] = []
        y_list: List[torch.Tensor] = []
        cos_list: List[torch.Tensor] = []
        sin_list: List[torch.Tensor] = []
        vx_list: List[torch.Tensor] = []
        vy_list: List[torch.Tensor] = []
        omega_list: List[torch.Tensor] = []

        for k in range(future_len):
            apply_S2_k = (k > 0)
            apply_S4_ax_k = (k > 0)

            vx_b_k = vx_b_raw[..., k]  # (B,Pnn)
            vy_b_k = vy_b_raw[..., k]  # (B,Pnn)
            yaw_rate_k = omega_raw[..., k]  # (B,Pnn)

            if self.use_feasible_filter:
                vx_b_k, vy_b_k, yaw_rate_k = self._apply_constraints_step(
                    vx_b_prev=vx_b_prev,
                    vy_b_prev=vy_b_prev,
                    omega_prev=omega_prev,
                    vx_b_k=vx_b_k,
                    vy_b_k=vy_b_k,
                    omega_k=yaw_rate_k,
                    hp=self.constraints_h_params,
                    key_to_limit_bp=key_to_limit_bp,
                    apply_S2=apply_S2_k,
                    apply_S4_ax=apply_S4_ax_k,
                )

            x_k1, y_k1, cos_k1, sin_k1 = self._integrate_midpoint_step(
                x_k=x_k,
                y_k=y_k,
                cos_yaw_k=cos_yaw_k,
                sin_yaw_k=sin_yaw_k,
                vx_b_k=vx_b_k,
                vy_b_k=vy_b_k,
                omega_k=yaw_rate_k,
                hp=self.constraints_h_params,
            )

            x_list.append(x_k1)
            y_list.append(y_k1)
            cos_list.append(cos_k1)
            sin_list.append(sin_k1)
            vx_list.append(vx_b_k)
            vy_list.append(vy_b_k)
            omega_list.append(yaw_rate_k)

            x_k, y_k, cos_yaw_k, sin_yaw_k = x_k1, y_k1, cos_k1, sin_k1
            vx_b_prev, vy_b_prev, omega_prev = vx_b_k, vy_b_k, yaw_rate_k

        key_to_all_states: Dict[str, torch.Tensor] = {
            "x_next": torch.stack(x_list, dim=2),  # (B,Pnn,T)
            "y_next": torch.stack(y_list, dim=2),  # (B,Pnn,T)
            "cos_next": torch.stack(cos_list, dim=2),  # (B,Pnn,T)
            "sin_next": torch.stack(sin_list, dim=2),  # (B,Pnn,T)
            "vx_after": torch.stack(vx_list, dim=2),  # (B,Pnn,T)
            "vy_after": torch.stack(vy_list, dim=2),  # (B,Pnn,T)
            "omega_after": torch.stack(omega_list, dim=2),  # (B,Pnn,T)
        }

        return self._assemble_outputs(
            key_to_all_states=key_to_all_states,
            vx_b_raw=vx_b_raw,
            vy_b_raw=vy_b_raw,
            omega_raw=omega_raw,
            near_cur_future_valid=near_cur_future_valid,
        )

    def _compute_active_indices_from_near_cur_future_valid(
            self,
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """near_cur_future_valid로부터 '계산이 필요한 (batch, neighbor) 슬롯'만 고릅니다.

        Args:
            near_cur_future_valid (torch.Tensor):
                shape: (B, Pnn, 1+T)
                현재(0)~미래(T) 노드 유효 마스크. True*False* 형태(단조 감소)라고 가정합니다.

        Returns:
            active_indices (torch.Tensor):
                shape: (N_active,) dtype=torch.long
                (B*Pnn)으로 펼쳤을 때, 유효 세그먼트가 1개 이상 있는 row 인덱스 목록.
            active_mask_flat (torch.Tensor):
                shape: (B*Pnn,) dtype=torch.bool
                각 row가 active인지 여부.
        """
        valid = near_cur_future_valid.to(torch.bool)  # (B, Pnn, 1+T)
        seg_valid = valid[..., :-1] & valid[..., 1:]  # (B, Pnn, T)
        seg_valid_any = seg_valid.any(dim=-1)  # (B, Pnn)
        active_mask_flat = seg_valid_any.reshape(-1)  # (B*Pnn,)
        active_indices = active_mask_flat.nonzero(as_tuple=False).squeeze(
            -1)  # (N_active,)
        return active_indices, active_mask_flat

    def _gather_active_subset_for_filter_and_integrate(
        self,
        unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
        near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)
        unnorm_cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 3)
        near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
        active_indices: torch.Tensor,  # (N_active,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """active 슬롯만 모아서 (N_active, 1, ...) 형태로 압축합니다.

        Args:
            unnorm_near_current_state: (B, Pnn, 4)
            near_cur_future_valid: (B, Pnn, 1+T)
            unnorm_cur_future_seg_body_control: (B, Pnn, T, 3)
            near_class_one_hot: (B, Pnn, 3)
            active_indices: (N_active,) flatten row 인덱스

        Returns:
            unnorm_near_current_state_active: (N_active, 1, 4)
            near_cur_future_valid_active: (N_active, 1, 1+T)
            unnorm_cur_future_seg_body_control_active: (N_active, 1, T, 3)
            near_class_one_hot_active: (N_active, 1, 3)
        """
        B, Pnn, T, _ = unnorm_cur_future_seg_body_control.shape
        B_Pnn = int(B * Pnn)

        state_flat = unnorm_near_current_state.reshape(B_Pnn, 4)  # (B*Pnn, 4)
        valid_flat = near_cur_future_valid.reshape(B_Pnn, 1 + T)  # (B*Pnn, 1+T)
        ctrl_flat = unnorm_cur_future_seg_body_control.reshape(
            B_Pnn, T, 3)  # (B*Pnn, T, 3)
        class_flat = near_class_one_hot.reshape(B_Pnn, 3)  # (B*Pnn, 3)

        # index_select로 active row만 추출
        idx = active_indices.to(device=state_flat.device,
                                dtype=torch.long)  # (N_active,)
        state_active = state_flat.index_select(0, idx).unsqueeze(
            1)  # (N_active, 1, 4)
        valid_active = valid_flat.index_select(0, idx).unsqueeze(
            1)  # (N_active, 1, 1+T)
        ctrl_active = ctrl_flat.index_select(0, idx).unsqueeze(
            1)  # (N_active, 1, T, 3)
        class_active = class_flat.index_select(0, idx).unsqueeze(
            1)  # (N_active, 1, 3)

        return state_active, valid_active, ctrl_active, class_active

    def _scatter_active_subset_for_filter_and_integrate(
        self,
        unnorm_integrated_trajectory_active: torch.Tensor,
        # (N_active, 1, T, 4)
        unnorm_control_constraint_diff_active: torch.Tensor,
        # (N_active, 1, T, 3)
        active_indices: torch.Tensor,  # (N_active,)
        *,
        B: int,
        Pnn: int,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """active 결과를 (B,Pnn,...) 원래 위치로 되돌리고, inactive는 0으로 둡니다.

        핵심:
            - in-place index_copy_로 '값만 복사'하지 않고,
            - out-of-place index_copy로 "새 텐서"를 만들어
              active 결과 → 최종 출력으로 학습 신호가 이어지게 합니다.
        """
        B_Pnn = int(B * Pnn)

        out_traj_flat = torch.zeros((B_Pnn, T, 4), device=device,
                                    dtype=dtype)  # (B*Pnn, T, 4)
        out_diff_flat = torch.zeros((B_Pnn, T, 3), device=device,
                                    dtype=dtype)  # (B*Pnn, T, 3)

        if int(active_indices.numel()) > 0:
            idx = active_indices.to(device=device,
                                    dtype=torch.long)  # (N_active,)
            traj_src = unnorm_integrated_trajectory_active.squeeze(1).to(
                device=device, dtype=dtype)  # (N_active, T, 4)
            diff_src = unnorm_control_constraint_diff_active.squeeze(1).to(
                device=device, dtype=dtype)  # (N_active, T, 3)

            # ✅ out-of-place
            out_traj_flat = out_traj_flat.index_copy(0, idx, traj_src)
            out_diff_flat = out_diff_flat.index_copy(0, idx, diff_src)

        out_traj = out_traj_flat.view(B, Pnn, T, 4)  # (B, Pnn, T, 4)
        out_diff = out_diff_flat.view(B, Pnn, T, 3)  # (B, Pnn, T, 3)
        return out_traj, out_diff

    def filter_and_integrate(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T) bool
            unnorm_cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter + Integrate 최상위 래퍼(패딩 슬롯 계산 스킵 포함).

        변경점:
            - (B,Pnn) 슬롯 중에서, 유효 세그먼트가 1개도 없는 슬롯은
              기존에도 최종 출력이 전부 0이므로 계산을 아예 하지 않습니다.
            - 유효 세그먼트가 있는 슬롯만 모아서 기존 로직을 그대로 실행한 뒤,
              결과를 원래 위치로 되돌립니다.

        Returns:
            unnorm_integrated_trajectory: (B, Pnn, T, 4)
            unnorm_control_constraint_diff: (B, Pnn, T, 3)
        """
        self._assert_cur_future_valid_mask(
            near_cur_future_valid,
            context="filter_and_integrate",
        )

        B, Pnn, T, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype

        # 1) active 슬롯 인덱스 계산
        active_indices, active_mask_flat = self._compute_active_indices_from_near_cur_future_valid(
            near_cur_future_valid=near_cur_future_valid,  # (B,Pnn,1+T)
        )

        # 2) active가 하나도 없으면, 기존 코드 결과와 동일하게 전부 0 반환
        if int(active_indices.numel()) == 0:
            unnorm_integrated_trajectory = unnorm_cur_future_seg_body_control.new_zeros(
                (B, Pnn, T, 4))
            unnorm_control_constraint_diff = unnorm_cur_future_seg_body_control.new_zeros(
                (B, Pnn, T, 3))
            return unnorm_integrated_trajectory, unnorm_control_constraint_diff

        # 3) active subset만 gather (N_active, 1, ...)
        (
            unnorm_near_current_state_active,  # (N_active, 1, 4)
            near_cur_future_valid_active,  # (N_active, 1, 1+T)
            unnorm_cur_future_seg_body_control_active,  # (N_active, 1, T, 3)
            near_class_one_hot_active,  # (N_active, 1, 3)
        ) = self._gather_active_subset_for_filter_and_integrate(
            unnorm_near_current_state=unnorm_near_current_state,
            near_cur_future_valid=near_cur_future_valid,
            unnorm_cur_future_seg_body_control=
            unnorm_cur_future_seg_body_control,
            near_class_one_hot=near_class_one_hot,
            active_indices=active_indices,
        )

        # 4) 기존 로직을 active subset에 그대로 적용
        if not self.use_batch_integration:
            traj_active, diff_active = self._filter_and_integrate_sequential(
                unnorm_near_current_state=unnorm_near_current_state_active,
                near_cur_future_valid=near_cur_future_valid_active,
                unnorm_cur_future_seg_body_control=
                unnorm_cur_future_seg_body_control_active,
                near_class_one_hot=near_class_one_hot_active,
            )
        else:
            traj_active, diff_active = self._filter_and_integrate_batch(
                unnorm_near_current_state=unnorm_near_current_state_active,
                near_cur_future_valid=near_cur_future_valid_active,
                unnorm_cur_future_seg_body_control=
                unnorm_cur_future_seg_body_control_active,
                near_class_one_hot=near_class_one_hot_active,
            )

        # 5) scatter: (B,Pnn,...)로 되돌리고 inactive는 0 유지
        unnorm_integrated_trajectory, unnorm_control_constraint_diff = self._scatter_active_subset_for_filter_and_integrate(
            unnorm_integrated_trajectory_active=
            traj_active,  # (N_active, 1, T, 4)
            unnorm_control_constraint_diff_active=
            diff_active,  # (N_active, 1, T, 3)
            active_indices=active_indices,  # (N_active,)
            B=B,
            Pnn=Pnn,
            T=T,
            device=device,
            dtype=dtype,
        )
        return unnorm_integrated_trajectory, unnorm_control_constraint_diff

    # ================================================================
    # [REFACTOR] Savitzky–Golay 유틸들 (모두 torch-only, 미분 가능)
    # ================================================================
    def _assert_cur_future_valid_mask(
        self,
        valid_bpt: torch.Tensor,
        context: str = "savgol_filter_for_control",
    ) -> None:
        """유효 마스크가 행마다 True*False* (단조 감소)인지 검증합니다.

        주의:
            - 이 검증은 디버그용입니다.
            - config.feasible_debug_check_mask=True 일 때만 실행합니다.

        Args:
            valid_bpt: (B, Pnn, T1) bool
            context: 에러 메시지에 표시할 호출 위치 문자열.
        """
        if not bool(getattr(self, "feasible_debug_check_mask", False)):
            return

        assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 여야 합니다."
        B, Pnn, T1 = valid_bpt.shape
        v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
        has_01 = (d > 0).any(dim=1)  # 0→1 전이 여부
        if has_01.any():
            bad_idx = torch.nonzero(has_01, as_tuple=False).flatten()
            max_show = min(int(bad_idx.numel()), 8)
            bad_idx_sample = bad_idx[:max_show].tolist()
            b_list = [(i // Pnn) for i in bad_idx_sample]
            p_list = [(i % Pnn) for i in bad_idx_sample]
            raise ValueError(
                f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*). \n"
                f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.numel())},  \n"
                f"example (b,p)={list(zip(b_list, p_list))}.  \n"
                f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
            )

    # 지우개: 예전 `_compute_world_linear_velocity_via_sg` 구현은
    #        x, y를 각각 단일 채널 SG에 넣어 두 번 호출하던 코드입니다.
    #        해당 본문 전체를 지우고 아래 새 구현으로 교체하세요.

    def _filter_and_integrate_batch(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
            unnorm_cur_future_seg_body_control: torch.
        Tensor,  # (B, Pnn, future_len, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """시간축 전체를 한 번에 처리하는 배치 버전.

        - S2(가속/각가속 증분 제한)는 사용하지 않는다.
        - S0/S1/S3만 vx,vy,omega 시퀀스에 배치로 적용한다.
        - yaw 및 위치는 cumsum 기반 중점 적분으로 계산한다.
        """
        B, Pnn, future_len, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype

        if future_len == 0:
            raise ValueError("future_len=0: 적분할 미래 세그먼트가 없습니다.")

        # per-agent 제한값 (v_max, a_lat_max, R_min, ...)
        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot,
            device=device,
            dtype=dtype,
        )

        # (B,Pnn,future_len)
        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(
            unnorm_cur_future_seg_body_control)

        # 시간축 전체에 S0/S1/S3 배치 적용 (S2는 미사용)
        """ 3개 모두 (B,Pnn,future_len) 반환"""
        vx_b_after, vy_b_after, omega_after = self._apply_constraints_batch(
            vx_b_raw=vx_b_raw,  # (B,Pnn,T)
            vy_b_raw=vy_b_raw,  # (B,Pnn,T)
            omega_raw=omega_raw,  # (B,Pnn,T)
            key_to_limit_bp=key_to_limit_bp,
            hp=self.constraints_h_params,
        )

        # 중점 적분을 시간축 전체에 대해 배치로 수행
        key_to_all_states = self._integrate_midpoint_batch(
            unnorm_near_current_state=unnorm_near_current_state,  # (B,Pnn,4)
            vx_b_seq=vx_b_after,  # (B,Pnn,T)
            vy_b_seq=vy_b_after,  # (B,Pnn,T)
            omega_seq=omega_after,  # (B,Pnn,T)
            hp=self.constraints_h_params,
        )

        return self._assemble_outputs(
            key_to_all_states=key_to_all_states,
            vx_b_raw=vx_b_raw,
            vy_b_raw=vy_b_raw,
            omega_raw=omega_raw,
            near_cur_future_valid=near_cur_future_valid,
        )

    def _compute_world_linear_velocity_via_sg(
        self,
        x: torch.Tensor,  # (B, Pnn, point_len)
        y: torch.Tensor,  # (B, Pnn, point_len)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_len_xy: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """x,y 위치 값이 시간에 따라 어떻게 움직이는지 부드러운 속도로 바꿔 주는 함수.

        - 단순 차분이 아니라, 주변 여러 점을 한꺼번에 보고 "부드러운 기울기"를 추정한다.
        - x와 y 두 축을 한 번에 처리해서, 계산량을 줄이도록 설계했다.

        Args:
            x: (B, Pnn, point_len)
                각 객체의 시간별 x 좌표.
            y: (B, Pnn, point_len)
                각 객체의 시간별 y 좌표.
            points_valid: (B, Pnn, point_len) bool
                해당 시점에 위치가 실제로 존재하는지(True/False) 표시.
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                Savitzky–Golay 다항식 차수.
            max_window_len_xy: int
                x,y에 사용할 최대 창 길이.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - v_x^w: (B, Pnn, point_len)  x축 방향 세계 속도
                - v_y^w: (B, Pnn, point_len)  y축 방향 세계 속도
        """
        # : x,y를 한 번에 채널로 묶어서 SG 한 번만 호출
        positions_stack = torch.stack(
            [x, y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)

        velocities_stack = self._sg_derivative_multi_for_points(
            sequences_points=positions_stack,  # (B,Pnn,point_len,2)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_xy,
        )  # (B, Pnn, point_len, 2)

        v_x_world = velocities_stack[..., 0]  # (B, Pnn, point_len)
        v_y_world = velocities_stack[..., 1]  # (B, Pnn, point_len)

        return v_x_world, v_y_world

    # ================================================================
    # 본 기능: 위치→세계속도, yaw→요레이트, 그리고 body 회전
    # ================================================================
    def savgol_filter_for_control(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,
        # (B,Pnn,1+future_len,4) = [x, y, cos, sin]
        unnorm_near_past_xyyaw: Optional[torch.Tensor],
        # (B,Pnn,past_len,4) or None
        target_past_cur_future_valid: torch.Tensor,
        # (B,Pnn,past_len+1+future_len) bool
        *,
        dt: float = 0.1,
        polyorder: int = 2,
        max_window_len_xy: int = 11,
        max_window_len_yaw: int = 7,
        return_point_len_inputs: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, PointLenInputs]]:
        """Savitzky–Golay(마스크 인지)로 **세계속도/각속도**를 추정하여 반환.

        Returns:
            - return_point_len_inputs=False:
                unnorm_points_world_control:
                    past가 없으면: (B,Pnn,1+future_len,3)  = [v_x^w, v_y^w, w]
                    past가 있으면: (B,Pnn,past_len+1+future_len,3)

            - return_point_len_inputs=True:
                (unnorm_points_world_control, point_len_inputs)
                point_len_inputs.unnorm_points_xyyaw: (B,Pnn,point_len,4)
                point_len_inputs.points_valid:        (B,Pnn,point_len) bool
        """
        # 1) 포인트/마스크 준비 및 검증
        point_len_inputs: PointLenInputs = self._prepare_points_and_masks(
            unnorm_diffusion_trajectory=unnorm_diffusion_trajectory,
            unnorm_near_past_xyyaw=unnorm_near_past_xyyaw,
            target_past_cur_future_valid=target_past_cur_future_valid,
        )
        unnorm_points_xyyaw = point_len_inputs.unnorm_points_xyyaw  # (B,Pnn,point_len,4)
        points_valid = point_len_inputs.points_valid  # (B,Pnn,point_len) bool

        # 2) 분해
        x = unnorm_points_xyyaw[..., 0]  # (B,Pnn,point_len)
        y = unnorm_points_xyyaw[..., 1]  # (B,Pnn,point_len)
        cos_y = unnorm_points_xyyaw[..., 2]  # (B,Pnn,point_len)
        sin_y = unnorm_points_xyyaw[..., 3]  # (B,Pnn,point_len)

        # 3) SG-미분 (x/y → v_x^w, v_y^w) & (cos/sin → w)
        v_x, v_y = self._compute_world_linear_velocity_via_sg(
            x=x,
            y=y,
            points_valid=points_valid,
            dt=dt,
            polyorder=polyorder,
            max_window_len_xy=max_window_len_xy,
        )
        yaw_rate = self._compute_yaw_rate_via_sg(
            cos_y=cos_y,
            sin_y=sin_y,
            points_valid=points_valid,
            dt=dt,
            polyorder=polyorder,
            max_window_len_yaw=max_window_len_yaw,
        )

        # 4) 마스킹·스택 후 반환
        unnorm_points_world_control = self._mask_and_stack_world_controls(
            v_x=v_x,
            v_y=v_y,
            yaw_rate=yaw_rate,
            points_valid=points_valid,
        )  # (B,Pnn,point_len,3)

        if return_point_len_inputs:
            return unnorm_points_world_control, point_len_inputs

        return unnorm_points_world_control

    def _compute_yaw_rate_via_sg(
        self,
        cos_y: torch.Tensor,  # (B, Pnn, point_len)
        sin_y: torch.Tensor,  # (B, Pnn, point_len)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_len_yaw: int,
    ) -> torch.Tensor:
        """cos(ψ), sin(ψ) 시퀀스로부터 각속도 ψ̇를 부드럽게 추정하는 함수.

        - 먼저 cos, sin을 함께 SG에 넣어 두 값의 시간 변화량을 구한다.
        - 그 뒤, 회전 운동의 관계식을 이용해 ψ̇를 계산한다.
        - cos, sin 을 다시 단위원으로 정리해 수치적인 오류가 쌓이지 않도록 한다.

        Args:
            cos_y: (B, Pnn, point_len)
                각 시점의 cos(ψ) 값.
            sin_y: (B, Pnn, point_len)
                각 시점의 sin(ψ) 값.
            points_valid: (B, Pnn, point_len) bool
                해당 시점에 자세 정보가 실제로 존재하는지(True/False).
            dt: float
                샘플 간 시간 간격.
            polyorder: int
                Savitzky–Golay 다항식 차수.
            max_window_len_yaw: int
                yaw에 사용할 최대 창 길이.

        Returns:
            torch.Tensor: (B, Pnn, point_len)
                각 시점의 부드럽게 추정된 yaw_rate(ψ̇).
        """
        batch_size, num_neighbors, point_len = cos_y.shape  # (B,Pnn,T)

        # : cos,sin을 하나의 2채널 시퀀스로 묶어서 SG 한 번만 호출
        yaw_unit_stack = torch.stack(
            [cos_y, sin_y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)

        yaw_unit_derivative = self._sg_derivative_multi_for_points(
            sequences_points=yaw_unit_stack,  # (B,Pnn,point_len,2)
            points_valid=points_valid,  # (B,Pnn,point_len)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_len_yaw,
        )  # (B, Pnn, point_len, 2)

        dcos = yaw_unit_derivative[..., 0]  # (B, Pnn, point_len)
        dsin = yaw_unit_derivative[..., 1]  # (B, Pnn, point_len)

        # : cos,sin을 다시 단위원으로 투영해서 수치 오차를 줄인다.
        cos_sin_stack = torch.stack(
            [cos_y, sin_y],
            dim=-1,
        )  # (B, Pnn, point_len, 2)
        norm = torch.linalg.norm(
            cos_sin_stack,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)  # (B, Pnn, point_len, 1)

        cos_unit = (cos_sin_stack[..., 0:1] / norm).squeeze(
            -1)  # (B,Pnn,point_len)
        sin_unit = (cos_sin_stack[..., 1:2] / norm).squeeze(
            -1)  # (B,Pnn,point_len)

        # : ψ̇ = (cos * d(sin) - sin * d(cos)) / (cos² + sin²)
        denom = (cos_unit * cos_unit + sin_unit * sin_unit).clamp_min(
            1e-6)  # (B,Pnn,T)
        yaw_rate = (cos_unit * dsin - sin_unit * dcos) / denom  # (B,Pnn,T)

        # 무효 시점은 0으로 정리
        yaw_rate = torch.where(
            points_valid,
            yaw_rate,
            torch.zeros_like(yaw_rate),
        )  # (B,Pnn,T)

        return yaw_rate

    # <추가하자>
    def _infer_forward_lengths_and_validate_base(
        self,
        target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len) + future_len)
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[int, int, int, int, int, torch.Tensor, torch.Tensor]:
        """forward 용 기본 길이/shape 및 최소 검증을 수행한다.

        Returns:
            B: int
            Pnn: int
            past_len: int
            future_len: int
            segment_len: int
            valid_all: (B, Pnn, 1+past_len+future_len) bool
            cur_future_valid: (B, Pnn, 1+future_len) bool
        """
        B, Pnn, total_time_len = target_past_cur_future_valid.shape
        B2, Pnn2, one_plus_future_len, _ = diffusion_trajectory.shape

        future_len = one_plus_future_len - 1  # 현재 이후 미래 segment 개수
        if future_len <= 0:
            raise ValueError("[forward] diffusion_trajectory 길이가 2 미만입니다. "
                             "최소 (현재, 미래1) 두 노드가 필요합니다.")
        if (B2, Pnn2) != (B, Pnn):
            raise ValueError(
                "[_infer_forward_lengths_and_validate_base] "
                "near_past_cur_future_valid과 diffusion_trajectory의 "
                f"batch/agent 축이 다릅니다: "
                f"(B,Pnn)=({B},{Pnn}), (B2,Pnn2)=({B2},{Pnn2})")
        past_len = total_time_len - (1 + future_len)

        if past_len < 0:
            raise ValueError(
                f"[forward] 마스크 길이가 너무 짧습니다: total_time_len={total_time_len}, future_len={future_len}"
            )
        # past_len 은 '과거 노드 개수' (현재 포함 전까지)
        # <추가하자> 전체 길이 일관성 체크
        if total_time_len != past_len + 1 + future_len:
            raise ValueError(
                f"[forward] target_past_cur_future_valid.shape[2]={total_time_len} "
                f"!= past_len+1+future_len={past_len + 1 + future_len}")

        if (B2, Pnn2) != (B, Pnn):
            raise ValueError(
                f"배치/에이전트 축 불일치: (B,Pnn)=({B},{Pnn}), (B2,Pnn2)=({B2},{Pnn2})")

        # seg_body_control 이 들고 있는 segment 개수 = segment_len
        _, _, segment_len, _ = seg_body_control.shape

        # 노드 기준 전체 유효 마스크 (bool)
        valid_all = target_past_cur_future_valid.to(
            torch.bool)  # (B, Pnn, past_len+1+future_len)

        # 현재~미래 구간만 따로 떼서 1*0* 패턴 검증용(cur_future_valid)
        cur_future_valid = valid_all[..., past_len:]  # (B, Pnn, 1+future_len)
        self._assert_cur_future_valid_mask(cur_future_valid,
                                           context="forward_cur_future_valid")

        return B, Pnn, past_len, future_len, segment_len, valid_all, cur_future_valid

    # <추가하자>
    def _validate_forward_past_mask_if_needed(
            self,
            past_len: int,
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            valid_all: torch.Tensor,  # (B, Pnn, past_len+1+future_len) bool
    ) -> None:
        """과거 구간을 실제로 사용할 때만 0*1* 패턴 검증."""
        if past_len > 0 and near_past_xyyaw is not None:
            past_cur_valid = valid_all[..., :past_len +
                                       1]  # (B, Pnn, 1+past_len)
            self._assert_past_cur_valid_mask(past_cur_valid,
                                             context="forward_past_cur_valid")

    # <추가하자>
    def _build_forward_points_without_past(
            self,
            future_len: int,
            segment_len: int,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            cur_future_valid: torch.Tensor,  # (B, Pnn, 1+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거 포인트를 사용하지 않는 경우의 points/valid 구성."""
        expected_segment_len = future_len
        if segment_len != expected_segment_len:
            raise ValueError("[forward] near_past_xyyaw is None 인 경우, "
                             f"seg_body_control 의 segment_len={segment_len} 이 "
                             f"future_len={future_len} 과 다릅니다.")

        points_trajectory = diffusion_trajectory  # (B, Pnn, 1+future_len, 4)
        points_valid = cur_future_valid  # (B, Pnn, 1+future_len)
        return points_trajectory, points_valid

    # <추가하자>
    def _build_forward_points_with_past(
        self,
        past_len: int,
        future_len: int,
        segment_len: int,
        near_past_xyyaw: torch.Tensor,  # (B, Pnn, past_len, 4)
        diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        valid_all: torch.Tensor,  # (B, Pnn, 1+past_len+future_len) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """과거~현재~미래 포인트를 모두 사용하는 경우의 points/valid 구성."""
        if near_past_xyyaw.shape[2] != past_len:
            raise ValueError(
                "[forward] near_past_xyyaw 의 past_len 이 유효 마스크에서 계산한 "
                f"past_len={past_len} 과 다릅니다. "
                f"(near_past_xyyaw.shape[2]={near_past_xyyaw.shape[2]})")

        expected_segment_len = past_len + future_len
        if segment_len != expected_segment_len:
            raise ValueError(
                "[forward] near_past_xyyaw 가 있을 때 seg_body_control 의 "
                f"segment_len={segment_len} 이 "
                f"past_len+future_len={expected_segment_len} 과 다릅니다.")

        points_trajectory = torch.cat(
            [near_past_xyyaw, diffusion_trajectory],
            dim=2)  # (B, Pnn, past_len + 1 + future_len, 4)

        if points_trajectory.shape[2] != (1 + expected_segment_len):
            raise ValueError(
                "[forward] points_trajectory 길이(과거+현재+미래)가 "
                f"1+segment_len={1 + expected_segment_len} 과 일치하지 않습니다. "
                f"(실제 길이={points_trajectory.shape[2]})")

        # 네트워크 입장에서는 '사용 가능한 모든 segment' 를 보고 싶으므로
        # 전체 타임라인 마스크를 그대로 넘긴다. (0*1*0* 패턴 허용)
        points_valid = valid_all  # (B, Pnn, past_len+1+future_len)
        return points_trajectory, points_valid

    # <추가하자>
    def _prepare_forward_points_and_mask(
            self,
            target_past_cur_future_valid: torch.
        Tensor,  # (B, Pnn, time_len(=1+past_len) + future_len)
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward 에서 사용할 포인트 궤적과 노드 유효 마스크를 준비한다.

        Args:
            target_past_cur_future_valid:
                (B, Pnn, time_len(=1+past_len) + future_len) bool
                = [과거 0..past_len-1, 현재, 미래 1..future_len] 노드 유효 마스크.
            diffusion_trajectory:
                (B, Pnn, 1+future_len, 4) = [x, y, cos, sin]
                현재~미래 구간 포인트.
            near_past_xyyaw:
                None 이면 과거 포인트를 사용하지 않는 모드.
                Tensor 이면 (B, Pnn, past_len, 4) 로 과거 포인트를 포함.
            seg_body_control:
                (B, Pnn, segment_len, 3) = [v_x^b, v_y^b, w]_seg.
                segment_len 은
                  - near_past_xyyaw is None  → future_len
                  - near_past_xyyaw not None → past_len + future_len

        Returns:
            points_trajectory:
                (B, Pnn, 1+segment_len, 4)
                - 과거 없음: diffusion_trajectory (현재~미래)
                - 과거 있음: [near_past_xyyaw, diffusion_trajectory] concat (과거~현재~미래)
            points_valid:
                (B, Pnn, 1+segment_len) bool
                - 과거 없음: 현재~미래 부분(cur_future_valid)
                - 과거 있음: 과거~현재~미래 전체(target_past_cur_future_valid)
        """
        (
            B,  # int
            Pnn,  # int
            past_len,  # int
            future_len,  # int
            segment_len,  # int
            valid_all,  # (B, Pnn, 1+past_len+future_len)
            cur_future_valid,  # (B, Pnn, 1+future_len)
        ) = self._infer_forward_lengths_and_validate_base(
            target_past_cur_future_valid=
            target_past_cur_future_valid,  # (B, Pnn, time_len(=1+past_len) + future_len)
            diffusion_trajectory=
            diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
        )

        # 과거 마스크(0*1*)는 실제로 과거 포인트를 사용할 때만 검증
        self._validate_forward_past_mask_if_needed(
            past_len=past_len,  # int
            near_past_xyyaw=near_past_xyyaw,  # (B, Pnn, past_len, 4) or None
            valid_all=valid_all,  # (B, Pnn, 1+past_len+future_len) bool
        )

        # 분기: 과거 포인트 사용 여부
        if near_past_xyyaw is None:
            """
            points_trajectory: (B, Pnn, 1+future_len, 4)
            points_valid: (B, Pnn, 1+future_len)
            """
            (points_trajectory, points_valid
            ) = self._build_forward_points_without_past(
                future_len=future_len,
                segment_len=segment_len,
                diffusion_trajectory=
                diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
                cur_future_valid=cur_future_valid,  # (B, Pnn, 1+future_len) bool
            )
        else:
            """
            points_trajectory : (B, Pnn, past_len + 1 + future_len, 4)
            points_valid : (B, Pnn, past_len+ 1 +future_len)
            """
            (points_trajectory,
             points_valid) = self._build_forward_points_with_past(
                 past_len=past_len,
                 future_len=future_len,
                 segment_len=segment_len,
                 near_past_xyyaw=near_past_xyyaw,  # (B, Pnn, past_len, 4)
                 diffusion_trajectory=
                 diffusion_trajectory,  # (B, Pnn, 1+future_len, 4)
                 valid_all=valid_all,  # (B, Pnn, 1+past_len+future_len)
             )

        # 최종 shape 일관성 체크
        if points_valid.shape[2] != points_trajectory.shape[2]:
            raise ValueError(
                "[forward] points_trajectory 와 points_valid 의 time 축 길이가 다릅니다. "
                f"points_trajectory.shape[2]={points_trajectory.shape[2]}, "
                f"points_valid.shape[2]={points_valid.shape[2]}.")

        return points_trajectory, points_valid

    # <추가하자>
    def _mask_and_stack_world_controls(
            self,
            v_x: torch.Tensor,  # (B,Pnn,point_len)
            v_y: torch.Tensor,  # (B,Pnn,point_len)
            yaw_rate: torch.Tensor,  # (B,Pnn,point_len)
            points_valid: torch.Tensor,  # (B,Pnn,point_len)  bool
    ) -> torch.Tensor:
        """무효 시점 0 마스킹 후 [v_x^w, v_y^w, w] 스택."""
        v_x = torch.where(points_valid, v_x, torch.zeros_like(v_x))
        v_y = torch.where(points_valid, v_y, torch.zeros_like(v_y))
        yaw_rate = torch.where(points_valid, yaw_rate,
                               torch.zeros_like(yaw_rate))
        return torch.stack([v_x, v_y, yaw_rate], dim=-1)  # (B,Pnn,point_len,3)

    # [!추가하자!]
    def _build_forward_segment_mask_and_active_flat(
        self,
        points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len) bool
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
    ) -> Tuple[
            torch.Tensor,  # seg_mask: (B, Pnn, segment_len)
            torch.Tensor,  # seg_mask_1: (B, Pnn, segment_len, 1)
            torch.Tensor,  # seg_mask_flat: (B*Pnn, segment_len)
            torch.Tensor,  # seg_mask_1_flat: (B*Pnn, segment_len, 1)
            torch.Tensor,  # active_flat: (B*Pnn,)
            int,  # B
            int,  # Pnn
            int,  # segment_len
            int,  # B_Pnn
    ]:
        """forward용 세그먼트 마스크와 '유효 이웃 플래그'를 생성합니다.

        - 세그먼트 마스크는 양 끝 노드가 모두 유효(True)인 구간만 1.0입니다.
        - active_flat은 (batch, neighbor)마다 '한 segment라도 유효한지'를 나타내는 플래그입니다.

        Args:
            points_valid: (B, Pnn, 1+segment_len) bool
                노드 단위 유효 마스크.
            seg_body_control: (B, Pnn, segment_len, 3)
                세그먼트 단위 제어 시퀀스 [v_x^b, v_y^b, w].

        Returns:
            seg_mask: (B, Pnn, segment_len) float32
            seg_mask_1: (B, Pnn, segment_len, 1) float32
            seg_mask_flat: (B*Pnn, segment_len) float32
            seg_mask_1_flat: (B*Pnn, segment_len, 1) float32
            active_flat: (B*Pnn,) bool
                각 (batch, neighbor) 슬롯별로 '한 개 이상 유효 segment가 있으면' True.
            B: 배치 크기
            Pnn: neighbor 슬롯 수
            segment_len: 세그먼트 길이
            B_Pnn: B * Pnn (flatten된 row 수)
        """
        B, Pnn, segment_len, _ = seg_body_control.shape
        seg_mask, seg_mask_1 = self._build_segment_mask(points_valid)
        B_Pnn = B * Pnn
        seg_mask_flat = seg_mask.view(B_Pnn, segment_len)
        seg_mask_1_flat = seg_mask_1.view(B_Pnn, segment_len, 1)
        active_flat = seg_mask_flat.any(dim=-1)  # (B*Pnn,)
        return (
            seg_mask,
            seg_mask_1,
            seg_mask_flat,
            seg_mask_1_flat,
            active_flat,
            B,
            Pnn,
            segment_len,
            B_Pnn,
        )

    # [!추가하자!]
    def _gather_forward_active_subset(
        self,
        points_trajectory: torch.Tensor,  # (B, Pnn, 1+segment_len, 4)
        points_valid: torch.Tensor,  # (B, Pnn, 1+segment_len)
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
        seg_mask_flat: torch.Tensor,  # (B*Pnn, segment_len)
        seg_mask_1_flat: torch.Tensor,  # (B*Pnn, segment_len, 1)
        dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
        active_flat: torch.Tensor,  # (B*Pnn,)
        B_Pnn: int,
        segment_len: int,
    ) -> Tuple[
            torch.Tensor,  # active_indices: (N_active,)
            torch.
            Tensor,  # points_trajectory_active: (N_active, 1, 1+segment_len, 4)
            torch.Tensor,  # points_valid_active: (N_active, 1, 1+segment_len)
            torch.
            Tensor,  # seg_body_control_active: (N_active, 1, segment_len, 3)
            torch.Tensor,  # seg_mask_active: (N_active, 1, segment_len)
            torch.Tensor,  # seg_mask_1_active: (N_active, 1, segment_len, 1)
            torch.Tensor,  # dit_tokens_active: (N_active, 1, H)
    ]:
        """유효 segment가 있는 이웃들만 골라 (N_active, 1, ·) 형태로 모읍니다.
        """
        # (B*Pnn,) -> (N_active,)
        active_indices = active_flat.nonzero(as_tuple=False).squeeze(
            -1)  # (N_active,)

        points_trajectory_flat = points_trajectory.view(
            B_Pnn, 1 + segment_len, 4)  # (B*Pnn, 1+segment_len, 4)
        points_valid_flat = points_valid.view(
            B_Pnn, 1 + segment_len)  # (B*Pnn, 1+segment_len)
        seg_body_control_flat = seg_body_control.view(
            B_Pnn, segment_len, 3)  # (B*Pnn, segment_len, 3)
        dit_tokens_flat = dit_final_hidden_tokens.view(B_Pnn, -1)  # (B*Pnn, H)

        points_trajectory_active = points_trajectory_flat[
            active_indices].unsqueeze(1)  # (N_active, 1, 1+segment_len, 4)
        points_valid_active = points_valid_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, 1+segment_len)
        seg_body_control_active = seg_body_control_flat[
            active_indices].unsqueeze(1)  # (N_active, 1, segment_len, 3)
        seg_mask_active = seg_mask_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, segment_len)
        seg_mask_1_active = seg_mask_1_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, segment_len, 1)
        dit_tokens_active = dit_tokens_flat[active_indices].unsqueeze(
            1)  # (N_active, 1, H)

        return (
            active_indices,
            points_trajectory_active,
            points_valid_active,
            seg_body_control_active,
            seg_mask_active,
            seg_mask_1_active,
            dit_tokens_active,
        )

    # [!추가하자!]
    def _run_feasible_tcn_for_active_subset(
            self,
            points_trajectory_active: torch.
        Tensor,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active: torch.Tensor,  # (N_active, 1, 1+segment_len)
            seg_body_control_active: torch.
        Tensor,  # (N_active, 1, segment_len, 3)
            seg_mask_active: torch.Tensor,  # (N_active, 1, segment_len)
            seg_mask_1_active: torch.Tensor,  # (N_active, 1, segment_len, 1)
            dit_tokens_active: torch.Tensor,  # (N_active, 1, H)
    ) -> torch.Tensor:  # (N_active, segment_len, 3)
        """유효 이웃(active subset)에 대해서만 TCN 기반 ΔU를 계산합니다.

        Args:
            points_trajectory_active: (N_active, 1, 1+segment_len, 4)
                active 이웃들의 포인트 궤적.
            points_valid_active: (N_active, 1, 1+segment_len) bool
                포인트 유효 마스크.
            seg_body_control_active: (N_active, 1, segment_len, 3)
                세그먼트 제어 [v_x^b, v_y^b, w].
            seg_mask_active: (N_active, 1, segment_len)
                세그먼트 마스크(0/1).
            seg_mask_1_active: (N_active, 1, segment_len, 1)
                세그먼트 마스크(채널용).
            dit_tokens_active: (N_active, 1, H)
                디퓨전 트렁크 은닉.

        Returns:
            delta_u_active: (N_active, segment_len, 3)
                각 active 이웃에 대한 ΔU.
        """
        x_prev_active, x_fut_active = self._split_prev_fut(
            points_trajectory_active)  # (N_active, 1, segment_len, 4) each
        x_prev_active = self._normalize_cos_sin(x_prev_active)
        x_fut_active = self._normalize_cos_sin(x_fut_active)

        Z_in_active = self._features_from_inputs(
            x_prev=x_prev_active,  # (N_active, 1, segment_len, 4)
            x_fut=x_fut_active,  # (N_active, 1, segment_len, 4)
            u_base=seg_body_control_active,  # (N_active, 1, segment_len, 3)
            dit_final_hidden_tokens=dit_tokens_active,  # (N_active, 1, H)
            points_valid=points_valid_active,  # (N_active, 1, 1+segment_len)
        )  # (N_active, 1, segment_len, 192)

        Z_s_active = self._prepare_tcn_input(
            Z_in_active, seg_mask_1_active)  # (N_active, 1, segment_len, 192)
        Z_tcn_active = self._run_tcn(
            Z_s_active, seg_mask_active,
            seg_mask_1_active)  # (N_active, 1, segment_len, 192)

        delta_u_active = self._predict_delta_u(
            Z_tcn_active, seg_mask_1_active)  # (N_active, 1, segment_len, 3)
        delta_u_active = delta_u_active.squeeze(1)  # (N_active, segment_len, 3)
        return delta_u_active

    def _scatter_forward_delta_u(
        self,
        seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
        delta_u_active: torch.Tensor,  # (N_active, segment_len, 3)
        active_indices: torch.Tensor,  # (N_active,)
        B: int,
        Pnn: int,
        segment_len: int,
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)
        """active 이웃들에서 계산한 ΔU를 전체 (B,Pnn,segment_len,3) 텐서로 되돌립니다.

        핵심:
            - in-place로 '값만 복사'하지 않고,
            - out-of-place index_copy로 "새 텐서"를 만들어
              delta_u_active → u_ref로 학습 신호가 이어지게 합니다.
        """
        B_Pnn = int(B * Pnn)

        seg_body_control_flat = seg_body_control.view(
            B_Pnn, segment_len, 3)  # (B*Pnn, segment_len, 3)

        # inactive는 0, active만 채운 delta_u_flat을 "새로" 만든다.
        delta_u_flat = seg_body_control_flat.new_zeros(
            (B_Pnn, segment_len, 3))  # (B*Pnn, segment_len, 3)

        if int(active_indices.numel()) > 0:
            idx = active_indices.to(device=seg_body_control.device,
                                    dtype=torch.long)  # (N_active,)
            src = delta_u_active.to(
                device=seg_body_control.device,
                dtype=seg_body_control_flat.dtype)  # (N_active, segment_len, 3)
            delta_u_flat = delta_u_flat.index_copy(0, idx,
                                                   src)  # ✅ out-of-place

        delta_u = delta_u_flat.view(B, Pnn, segment_len,
                                    3)  # (B, Pnn, segment_len, 3)
        u_ref = seg_body_control + delta_u  # (B, Pnn, segment_len, 3)
        return u_ref

    def forward(
            self,
            target_past_cur_future_valid: torch.Tensor,
            # (B, Pnn, time_len(=1+past_len) + future_len) bool
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
            near_past_xyyaw: Optional[
                torch.Tensor],  # (B, Pnn, past_len, 4) or None
            seg_body_control: torch.Tensor,  # (B, Pnn, segment_len, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
    ) -> torch.Tensor:  # (B, Pnn, segment_len, 3)
        """Control Correction Network 전체 forward 경로.

        - 과거/현재/미래 포인트와 유효 마스크를 정리한 뒤,
        - 세그먼트 마스크를 만들고,
        - 유효 segment가 하나라도 있는 이웃들(active subset)에 대해서만
          TCN 기반 ΔU를 계산한 후,
        - 전체 (B,Pnn,segment_len,3) 제어 텐서로 다시 scatter 합니다.

        Args:
            target_past_cur_future_valid:
                (B, Pnn, time_len(=1+past_len) + future_len) bool
                과거~현재~미래 노드 유효 마스크.
            diffusion_trajectory:
                (B, Pnn, 1+future_len, 4) = [x, y, cos, sin]
                현재~미래 포인트 궤적.
            near_past_xyyaw:
                None 이면 현재~미래만 사용 (segment_len = future_len).
                Tensor 이면 (B, Pnn, past_len, 4) 로 과거~현재~미래 전체 사용
                (segment_len = past_len + future_len).
            seg_body_control:
                (B, Pnn, segment_len, 3)  = [v_x^b, v_y^b, w]_seg.
            dit_final_hidden_tokens:
                (B, Pnn, H)  디퓨전 트렁크 최종 은닉.

        Returns:
            u_ref: (B, Pnn, segment_len, 3)
                입력 seg_body_control 에 대한 보정 제어.
        """
        if not self.use_feasible_dl:
            return seg_body_control

        # 1) forward에서 사용할 포인트 궤적 + 노드 유효 마스크 준비
        #    points_trajectory: (B, Pnn, 1+segment_len, 4)
        #    points_valid:      (B, Pnn, 1+segment_len) bool
        points_trajectory, points_valid = self._prepare_forward_points_and_mask(
            target_past_cur_future_valid=target_past_cur_future_valid,
            diffusion_trajectory=diffusion_trajectory,
            near_past_xyyaw=near_past_xyyaw,
            seg_body_control=seg_body_control,
        )

        # 2) 세그먼트 마스크 및 active 플래그 생성
        (
            seg_mask,  # (B, Pnn, segment_len)
            seg_mask_1,  # (B, Pnn, segment_len, 1)
            seg_mask_flat,  # (B*Pnn, segment_len)
            seg_mask_1_flat,  # (B*Pnn, segment_len, 1)
            active_flat,  # (B*Pnn,)
            B,
            Pnn,
            segment_len,
            B_Pnn,
        ) = self._build_forward_segment_mask_and_active_flat(
            points_valid=points_valid,  # (B, Pnn, 1+segment_len) bool
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
        )

        # 2-1) 유효 segment가 하나도 없는 경우: 그대로 반환
        if not active_flat.any():
            return seg_body_control

        # 3) active 이웃 subset만 모으기
        (
            active_indices,  # (N_active,)
            points_trajectory_active,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active,  # (N_active, 1, 1+segment_len)
            seg_body_control_active,  # (N_active, 1, segment_len, 3)
            seg_mask_active,  # (N_active, 1, segment_len)
            seg_mask_1_active,  # (N_active, 1, segment_len, 1)
            dit_tokens_active,  # (N_active, 1, H)
        ) = self._gather_forward_active_subset(
            points_trajectory=points_trajectory,  # (B, Pnn, 1+segment_len, 4)
            points_valid=points_valid,  # (B, Pnn, 1+segment_len) bool
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
            seg_mask_flat=seg_mask_flat,  # (B*Pnn, segment_len)
            seg_mask_1_flat=seg_mask_1_flat,  # (B*Pnn, segment_len, 1)
            dit_final_hidden_tokens=dit_final_hidden_tokens,  # (B, Pnn, H)
            active_flat=active_flat,  # (B*Pnn,)
            B_Pnn=B_Pnn,
            segment_len=segment_len,
        )

        # 4) active subset에 대해서만 TCN + Head 수행해 ΔU 계산
        # (N_active, segment_len, 3)
        delta_u_active = self._run_feasible_tcn_for_active_subset(
            points_trajectory_active=
            points_trajectory_active,  # (N_active, 1, 1+segment_len, 4)
            points_valid_active=
            points_valid_active,  # (N_active, 1, 1+segment_len)
            seg_body_control_active=
            seg_body_control_active,  # (N_active, 1, segment_len, 3)
            seg_mask_active=seg_mask_active,  # (N_active, 1, segment_len)
            seg_mask_1_active=seg_mask_1_active,  # (N_active, 1, segment_len, 1)
            dit_tokens_active=dit_tokens_active,  # (N_active, 1, H)
        )

        # 5) ΔU를 전체 (B,Pnn,segment_len,3) 텐서로 scatter 후 U_ref 반환
        u_ref = self._scatter_forward_delta_u(
            seg_body_control=seg_body_control,  # (B, Pnn, segment_len, 3)
            delta_u_active=delta_u_active,  # (N_active, segment_len, 3)
            active_indices=active_indices,  # (N_active,)
            B=B,
            Pnn=Pnn,
            segment_len=segment_len,
        )
        return u_ref

    # ================================================================
    # [추가] 멀티 채널용 Savitzky–Golay 미분 유틸
    # ================================================================

    def _savgol_finite_difference_multi(
        self,
        sequence_multi_channel: torch.Tensor,  # (N, T, C)
        valid_mask_bT: torch.Tensor,  # (N, T) bool
        dt: float,
    ) -> torch.Tensor:
        """여러 값이 한 줄에 모여 있을 때, mask 를 고려해서
        가장 단순한 방식으로 시간 변화량을 구하는 함수.

        - 중심 시점이 invalid 이면 항상 0.
        - 중심이 valid 인 경우에만 이웃을 보고,
          - 양옆 모두 valid → 중앙 차분
          - 한쪽만 valid → 해당 방향 전/후방 차분
          - 양옆 모두 invalid → 0
        """
        num_rows, sequence_length, num_channels = sequence_multi_channel.shape  # (N, T, C)
        derivative_fd = torch.zeros_like(sequence_multi_channel)  # (N, T, C)

        if sequence_length <= 1:
            # 길이 0~1이면 변화량을 정의하기 애매하므로 그냥 0 유지
            return derivative_fd

        valid = valid_mask_bT.to(torch.bool)  # (N, T)

        # --- 길이 2인 특수 케이스 ---
        if sequence_length == 2:
            center_pair_valid = (valid[:, 0] & valid[:, 1]).unsqueeze(
                -1)  # (N,1)
            diff = (sequence_multi_channel[:, 1, :] -
                    sequence_multi_channel[:, 0, :]) / dt  # (N, C)
            diff = diff * center_pair_valid  # invalid 포함되면 0
            derivative_fd[:, 0, :] = diff
            derivative_fd[:, 1, :] = diff
            return derivative_fd

        # --- 길이 >= 3인 일반 케이스 ---

        # 1) 맨 앞(t=0): 앞은 없고, 오른쪽(t=1)만 본다.
        center0 = valid[:, 0]
        right0 = valid[:, 1]
        edge0_valid = (center0 & right0).unsqueeze(-1)  # (N,1)
        diff0 = (sequence_multi_channel[:, 1, :] -
                 sequence_multi_channel[:, 0, :]) / dt  # (N,C)
        derivative_fd[:, 0, :] = diff0 * edge0_valid

        # 2) 맨 뒤(t=T-1): 오른쪽은 없고, 왼쪽(t=T-2)만 본다.
        centerL = valid[:, sequence_length - 1]
        leftL = valid[:, sequence_length - 2]
        edgeL_valid = (centerL & leftL).unsqueeze(-1)  # (N,1)
        diffL = (
            sequence_multi_channel[:, sequence_length - 1, :] -
            sequence_multi_channel[:, sequence_length - 2, :]) / dt  # (N,C)
        derivative_fd[:, sequence_length - 1, :] = diffL * edgeL_valid

        # 3) 내부 구간(t=1..T-2): 중앙/전방/후방 차분 조합
        center = valid[:, 1:-1]  # (N, T-2)
        left = valid[:, :-2]  # (N, T-2)
        right = valid[:, 2:]  # (N, T-2)

        both = center & left & right
        only_left = center & left & (~right)
        only_right = center & (~left) & right
        # center 가 False 인 곳은 위 셋 다 False → derivative 0 유지

        both_f = both.unsqueeze(-1).to(
            sequence_multi_channel.dtype)  # (N,T-2,1)
        only_left_f = only_left.unsqueeze(-1).to(sequence_multi_channel.dtype)
        only_right_f = only_right.unsqueeze(-1).to(sequence_multi_channel.dtype)

        # 중앙 차분: (t+1 - t-1) / (2dt)
        central_diff = (sequence_multi_channel[:, 2:, :] -
                        sequence_multi_channel[:, :-2, :]) / (2.0 * dt
                                                             )  # (N,T-2,C)
        # 후방 차분: (t - t-1) / dt
        backward_diff = (sequence_multi_channel[:, 1:-1, :] -
                         sequence_multi_channel[:, :-2, :]) / dt  # (N,T-2,C)
        # 전방 차분: (t+1 - t) / dt
        forward_diff = (sequence_multi_channel[:, 2:, :] -
                        sequence_multi_channel[:, 1:-1, :]) / dt  # (N,T-2,C)

        derivative_fd[:, 1:-1, :] = (both_f * central_diff +
                                     only_left_f * backward_diff +
                                     only_right_f * forward_diff)

        return derivative_fd  # (N, T, C)

    def _savgol_finite_difference_second_multi(
        self,
        sequence_multi_channel: torch.Tensor,  # (N, T, C)
        valid_mask_bT: torch.Tensor,  # (N, T) bool
        dt: float,
    ) -> torch.Tensor:
        """여러 채널 시퀀스의 '두 번의 변화(2차 변화율)'을 아주 단순한 규칙으로 계산합니다.

        - 중심 시점이 유효(True)이고, 주변에 필요한 점들이 모두 유효(True)일 때만 계산합니다.
        - 그 외에는 0으로 둡니다.
        - 이 함수는 창 기반 계산이 어려운 구간(시퀀스가 너무 짧거나, 유효 점이 부족한 곳)의
          기본값(대체값)으로만 쓰입니다.

        Args:
            sequence_multi_channel: (N, T, C)
                시간축 T에 따라 변하는 값들(채널 C개).
            valid_mask_bT: (N, T) bool
                각 시점 값이 실제로 있는지(True/False).
            dt: float
                샘플 간 시간 간격(초).

        Returns:
            second_derivative_fd: (N, T, C)
                2차 변화율 결과. 계산 불가한 곳은 0.
        """
        num_rows, sequence_length, num_channels = sequence_multi_channel.shape  # (N,T,C)
        second_derivative_fd = torch.zeros_like(
            sequence_multi_channel)  # (N,T,C)

        if sequence_length < 3:
            return second_derivative_fd

        valid = valid_mask_bT.to(torch.bool)  # (N,T)
        dt2 = float(dt) * float(dt)

        # 내부 구간(t=1..T-2): (x_{t+1} - 2x_t + x_{t-1}) / dt^2
        center = valid[:, 1:-1]  # (N,T-2)
        left = valid[:, :-2]  # (N,T-2)
        right = valid[:, 2:]  # (N,T-2)
        ok = center & left & right  # (N,T-2)

        sec = (sequence_multi_channel[:, 2:, :] -
               2.0 * sequence_multi_channel[:, 1:-1, :] +
               sequence_multi_channel[:, :-2, :]) / dt2  # (N,T-2,C)

        second_derivative_fd[:, 1:-1, :] = sec * ok.unsqueeze(-1).to(
            sequence_multi_channel.dtype)

        # 맨 앞(t=0): (x2 - 2x1 + x0) / dt^2
        ok0 = valid[:, 0] & valid[:, 1] & valid[:, 2]  # (N,)
        sec0 = (sequence_multi_channel[:, 2, :] -
                2.0 * sequence_multi_channel[:, 1, :] +
                sequence_multi_channel[:, 0, :]) / dt2  # (N,C)
        second_derivative_fd[:, 0, :] = sec0 * ok0.unsqueeze(-1).to(
            sequence_multi_channel.dtype)

        # 맨 뒤(t=T-1): (x_{T-1} - 2x_{T-2} + x_{T-3}) / dt^2
        okL = valid[:, -1] & valid[:, -2] & valid[:, -3]  # (N,)
        secL = (sequence_multi_channel[:, -1, :] -
                2.0 * sequence_multi_channel[:, -2, :] +
                sequence_multi_channel[:, -3, :]) / dt2  # (N,C)
        second_derivative_fd[:, -1, :] = secL * okL.unsqueeze(-1).to(
            sequence_multi_channel.dtype)

        return second_derivative_fd  # (N,T,C)

    def _savgol_select_window_length(
        self,
        sequence_length: int,
        max_window_length: int,
    ) -> int:
        """실제로 사용할 Savitzky–Golay 창 길이를 결정하는 함수.

        - 전체 길이보다 긴 창은 쓸 수 없으니 자동으로 줄인다.
        - SG 특성상 홀수 길이가 필요하므로 짝수이면 1 줄여서 홀수로 바꾼다.

        Args:
            sequence_length: int
                시퀀스 길이 T.
            max_window_length: int
                설정해 둔 최대 창 길이.

        Returns:
            int: 실제로 사용할 창 길이(0 이상).
                 0이면 SG를 적용하지 않고 유한 차분만 사용한다는 뜻으로 쓸 수 있다.
        """
        window_length = int(min(max_window_length, sequence_length))
        if window_length <= 0:
            return 0
        if window_length % 2 == 0:
            window_length -= 1
        if window_length <= 0:
            window_length = 1
        return window_length

    def _savgol_build_mask_and_count_multi(
        self,
        valid_mask_bT: torch.Tensor,  # (N, T) bool
        window_length: int,
        value_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """유효 마스크로부터, 각 시점 주변 창에 실제로 데이터가 몇 개 있는지 계산하는 함수.

        - 중심 시점마다 길이 W짜리 창을 슬라이딩하면서,
          그 안에서 유효(True)한 샘플 수를 센다.

        Args:
            valid_mask_bT: (N, T) bool
                각 시점 값이 실제로 존재하는지(True/False)를 나타내는 마스크.
            window_length: int
                SG 창 길이 W.
            value_dtype: torch.dtype
                float32/float16 등, 계산에 사용할 실수 타입.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - mask_window: (N, T, W)  창 안에서의 유효 마스크(0/1 float)
                - valid_count: (N, T)     창 안 유효 샘플 개수
        """
        num_rows, sequence_length = valid_mask_bT.shape  # (N, T)
        half_window = window_length // 2

        valid_float = valid_mask_bT.to(dtype=value_dtype)  # (N, T)
        valid_padded = F.pad(
            valid_float,
            (half_window, half_window),
            mode="constant",
            value=0.0,
        )  # (N, T + 2*half_window)

        mask_window = valid_padded.unfold(
            dimension=-1,
            size=window_length,
            step=1,
        )  # (N, T, W)

        valid_count = mask_window.sum(dim=-1)  # (N, T)

        return mask_window, valid_count  # (N,T,W), (N,T)

    def _savgol_build_A_all_from_mask_multi(
            self,
            mask_window: torch.Tensor,  # (N, T, W)
            gram_per_k: torch.Tensor,  # (W, P+1, P+1)
    ) -> torch.Tensor:
        """창별 시간 정보(gram_per_k)와 마스크를 이용해
        각 시점에서 쓸 작은 행렬 A를 한 번에 만드는 함수.

        Args:
            mask_window: (N, T, W)
                각 시점 t 기준으로 주변 W개 위치가 유효한지(0/1) 나타내는 값.
            gram_per_k: (W, P+1, P+1)
                시간 기저 Φ_k에 대한 Φ_k^T Φ_k 값을 위치별로 쌓아둔 텐서.

        Returns:
            torch.Tensor: (N, T, P+1, P+1)
                각 (row, t)에서 사용할 작은 행렬 A.
        """
        num_rows, sequence_length, window_length = mask_window.shape  # (N, T, W)
        _, dim_p1, _ = gram_per_k.shape  # (W, P+1, P+1)

        mask_expanded = mask_window.unsqueeze(-1).unsqueeze(
            -1)  # (N, T, W, 1, 1)
        gram_expanded = gram_per_k.view(1, 1, window_length, dim_p1,
                                        dim_p1)  # (1, 1, W, P+1, P+1)

        A_all = (mask_expanded * gram_expanded).sum(dim=2)  # (N, T, P+1, P+1)
        return A_all

    def _savgol_build_b_all_multi(
        self,
        sequence_multi_channel: torch.Tensor,  # (N, T, C)
        mask_window: torch.Tensor,  # (N, T, W)
        power_per_k: torch.Tensor,  # (W, P+1)
        window_length: int,
    ) -> torch.Tensor:
        """여러 채널 값과 시간 기저를 이용해, 각 위치에서 쓸 작은 벡터 b를 한 번에 만드는 함수.

        Args:
            sequence_multi_channel: (N, T, C)
                시간에 따라 변하는 값들이 C개 채널로 모여 있는 텐서.
            mask_window: (N, T, W)
                각 시점 주변 창에 어떤 위치가 유효한지(0/1) 나타내는 값.
            power_per_k: (W, P+1)
                시간 기저 Φ_k = [1, τ, τ^2, ...] 를 위치별로 모아둔 텐서.
            window_length: int
                창 길이 W.

        Returns:
            torch.Tensor: (N, T, C, P+1)
                각 (row, t, channel)에 대해 작은 벡터 b.
        """
        num_rows, sequence_length, num_channels = sequence_multi_channel.shape  # (N,T,C)
        half_window = window_length // 2
        P_plus_one = power_per_k.shape[-1]

        # (1) 시퀀스를 창 단위로 펼치기
        sequence_flat = sequence_multi_channel.permute(0, 2, 1).contiguous()
        sequence_flat = sequence_flat.reshape(num_rows * num_channels,
                                              sequence_length)  # (N*C, T)

        sequence_padded = F.pad(sequence_flat, (half_window, half_window),
                                mode="constant",
                                value=0.0)
        sequence_window_flat = sequence_padded.unfold(dimension=-1,
                                                      size=window_length,
                                                      step=1)  # (N*C, T, W)

        sequence_window = sequence_window_flat.view(
            num_rows, num_channels, sequence_length,
            window_length).permute(0, 2, 3, 1)  # (N, T, W, C)

        # (2) 마스크를 곱해 유효 위치만 남기기
        weighted_values = sequence_window * mask_window.unsqueeze(
            -1)  # (N, T, W, C)
        weighted_values_ex = weighted_values.unsqueeze(-1)  # (N, T, W, C, 1)

        # (3) 시간 기저 Φ_k 를 채널마다 곱해서 b를 만들기
        power_expanded = power_per_k.view(1, 1, window_length, 1,
                                          P_plus_one)  # (1, 1, W, 1, P+1)

        b_all = (weighted_values_ex * power_expanded).sum(
            dim=2)  # (N, T, C, P+1)
        return b_all

    def _savgol_solve_multi(
        self,
        A_all: torch.Tensor,  # (N, T, P+1, P+1)
        b_all: torch.Tensor,  # (N, T, C, P+1)
        valid_count: torch.Tensor,  # (N, T)
        valid_center_mask: torch.Tensor,  # (N, T) bool
        fd_derivative: torch.Tensor,  # (N, T, C)
        polyorder: int,
        derivative_order: int,
        regularization_epsilon: float,
    ) -> torch.Tensor:
        """창 기반 작은 선형 문제를 풀어, 원하는 변화율(1차 또는 2차)을 계산합니다.

        - 계산이 안정적인 위치(유효 점이 충분하고 중심이 유효한 곳)만 선형 문제를 풀어 값을 얻습니다.
        - 나머지 위치는 미리 만든 기본값(fd_derivative)을 그대로 둡니다.
        - 중심 시점이 무효(False)인 위치는 최종적으로 0으로 정리합니다.

        Args:
            A_all: (N, T, P+1, P+1)
            b_all: (N, T, C, P+1)
            valid_count: (N, T)
            valid_center_mask: (N, T) bool
            fd_derivative: (N, T, C)
            polyorder: 다항식 차수(예: 2)
            derivative_order: 1 또는 2
            regularization_epsilon: 수치 안정용 작은 값

        Returns:
            derivative_out: (N, T, C)
        """
        num_rows, sequence_length, dim_p1, _ = A_all.shape  # (N, T, P+1, P+1)
        _, _, num_channels, _ = b_all.shape  # (N, T, C, P+1)

        if derivative_order not in (1, 2):
            raise ValueError(
                f"derivative_order must be 1 or 2. got={derivative_order}")

        # polyorder가 derivative_order보다 작으면, 창 기반 계산이 의미 없으니 기본값만 사용
        if int(polyorder) < int(derivative_order):
            out = torch.where(
                valid_center_mask.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )
            return out

        min_samples = int(polyorder) + 2

        derivative_out = fd_derivative.clone()  # (N, T, C)

        good_mask = (valid_count >= min_samples) & valid_center_mask  # (N, T)

        # 기존 코드와 동일: 맨 앞/뒤는 창 기반 계산을 쓰지 않음
        edge_margin: int = 2
        if sequence_length > 2 * edge_margin:
            good_mask[:, :edge_margin] = False
            good_mask[:, -edge_margin:] = False

        if not good_mask.any():
            derivative_out = torch.where(
                valid_center_mask.unsqueeze(-1),
                derivative_out,
                torch.zeros_like(derivative_out),
            )
            return derivative_out

        good_flat_idx = good_mask.view(-1).nonzero(as_tuple=False).squeeze(
            -1)  # (M,)

        A_flat = A_all.view(-1, dim_p1, dim_p1)  # (N*T, P+1, P+1)
        A_good = A_flat[good_flat_idx]  # (M, P+1, P+1)

        b_flat = b_all.view(-1, num_channels, dim_p1)  # (N*T, C, P+1)
        b_good = b_flat[good_flat_idx].permute(0, 2, 1)  # (M, P+1, C)

        identity_matrix = torch.eye(
            dim_p1,
            device=A_good.device,
            dtype=A_good.dtype,
        ).unsqueeze(0)  # (1, P+1, P+1)

        A_good_reg = A_good + float(
            regularization_epsilon) * identity_matrix  # (M, P+1, P+1)

        coefficients_good = torch.linalg.solve(A_good_reg,
                                               b_good)  # (M, P+1, C)

        # τ=0에서의 변화율:
        # 1차: coeff[1]
        # 2차: 2 * coeff[2]
        coef_idx = int(derivative_order)
        if derivative_order == 1:
            scale = 1.0
        else:
            scale = 2.0

        derivative_good = coefficients_good[:, coef_idx, :] * scale  # (M, C)

        derivative_out_flat = derivative_out.view(-1, num_channels)  # (N*T, C)
        derivative_out_flat[good_flat_idx] = derivative_good
        derivative_out = derivative_out_flat.view(num_rows, sequence_length,
                                                  num_channels)  # (N, T, C)

        derivative_out = torch.where(
            valid_center_mask.unsqueeze(-1),
            derivative_out,
            torch.zeros_like(derivative_out),
        )
        return derivative_out

    def _savgol_derivative_masked_multi_torch(
        self,
        seq_bTC: torch.Tensor,  # (N, T, C)
        valid_bT: torch.Tensor,  # (N, T) bool
        dt: float,
        polyorder: int,
        max_window_length: int,
        derivative_order: int = 1,
    ) -> torch.Tensor:
        """마스크를 고려해서 변화율(1차 또는 2차)을 계산합니다.

        구현 정책(요청 반영):
            1) 기본값은 fd(유한 차분)로 만든다.
            2) valid_count는 unfold 대신 conv1d로 계산한다.
            3) 창이 "전부 유효(valid_count == W)"인 구간만
               SG 고정 커널 conv1d 결과로 덮어쓴다.
            4) 나머지(경계/짧은 유효구간)는 fd를 유지한다.

        Args:
            seq_bTC: (N, T, C)
            valid_bT: (N, T) bool
            dt: 샘플 간 시간 간격(초)
            polyorder: SG 다항식 차수
            max_window_length: 사용할 최대 창 길이
            derivative_order: 1 또는 2

        Returns:
            derivative_out: (N, T, C)
        """
        if seq_bTC.numel() == 0:
            return seq_bTC

        if derivative_order not in (1, 2):
            raise ValueError(
                f"derivative_order must be 1 or 2. got={derivative_order}")

        N, T, C = seq_bTC.shape
        device = seq_bTC.device
        dtype = seq_bTC.dtype

        valid = valid_bT.to(torch.bool)  # (N, T)

        # 0) 기본값(fd)
        if derivative_order == 1:
            fd_derivative = self._savgol_finite_difference_multi(
                sequence_multi_channel=seq_bTC,  # (N,T,C)
                valid_mask_bT=valid,  # (N,T)
                dt=float(dt),
            )
        else:
            fd_derivative = self._savgol_finite_difference_second_multi(
                sequence_multi_channel=seq_bTC,  # (N,T,C)
                valid_mask_bT=valid,  # (N,T)
                dt=float(dt),
            )

        # polyorder가 derivative_order보다 작으면 SG 기반 계산은 의미가 없으니 fd만 사용
        if int(polyorder) < int(derivative_order):
            return torch.where(
                valid.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )

        # 1) 창 길이 선택
        window_length = self._savgol_select_window_length(
            sequence_length=int(T),
            max_window_length=int(max_window_length),
        )
        if window_length <= 0:
            return torch.where(
                valid.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )

        # SG 기본 조건: window_length > polyorder
        # 이 조건이 깨지면 커널을 만들 수 없으니 fd 유지
        if int(window_length) <= int(polyorder):
            return torch.where(
                valid.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )

        W = int(window_length)
        half = W // 2

        # 2) valid_count를 conv1d로 계산
        #    valid_count: (N, T)
        valid_float = valid.to(dtype=dtype, device=device)  # (N,T)
        ones_kernel = self._get_sg_ones_kernel_1x1w(
            window_length=W,
            device=device,
            dtype=dtype,
        )  # (1,1,W)

        valid_count = F.conv1d(
            valid_float.unsqueeze(1),  # (N,1,T)
            ones_kernel,  # (1,1,W)
            padding=int(half),
        ).squeeze(1)  # (N,T)

        # 3) "창이 전부 유효"인 곳만 SG conv로 덮어쓰기
        # float 비교 안정성 때문에 == 대신 >= W-0.5 사용
        full_window_mask = (valid_count >= (float(W) - 0.5)) & valid  # (N,T)

        # 기존 코드와 동일한 edge_margin 정책 유지(앞/뒤 몇 칸은 창 기반 계산을 쓰지 않음)
        edge_margin: int = 2
        if int(T) > 2 * edge_margin:
            full_window_mask[:, :edge_margin] = False
            full_window_mask[:, -edge_margin:] = False

        # full window가 한 군데도 없으면 그냥 fd 반환
        if not bool(full_window_mask.any()):
            return torch.where(
                valid.unsqueeze(-1),
                fd_derivative,
                torch.zeros_like(fd_derivative),
            )

        # 4) SG 중앙 미분 커널(conv) 계산
        reg_eps: float = 1e-6  # 기존 solve 경로의 안정화 항과 맞춤(동일하게 두는 게 안전)
        kernel_1x1w = self._get_sg_center_derivative_kernel_1x1w(
            window_length=W,
            polyorder=int(polyorder),
            derivative_order=int(derivative_order),
            dt=float(dt),
            device=device,
            dtype=dtype,
            regularization_epsilon=float(reg_eps),
        )  # (1,1,W)

        sg_derivative = self._apply_same_kernel_conv1d_per_channel(
            seq_bTC=seq_bTC,  # (N,T,C)
            kernel_1x1w=kernel_1x1w,  # (1,1,W)
        )  # (N,T,C)

        # 5) full window인 곳만 SG로 덮어쓰기, 나머지는 fd 유지
        derivative_out = torch.where(
            full_window_mask.unsqueeze(-1),
            sg_derivative,
            fd_derivative,
        )

        # 6) 중심이 invalid면 0
        derivative_out = torch.where(
            valid.unsqueeze(-1),
            derivative_out,
            torch.zeros_like(derivative_out),
        )
        return derivative_out

    @classmethod
    def loss_weights_by_progress(cls, progress: float,
                                 args: Any) -> Tuple[float, float, float]:
        """손실 가중치 스케줄러.

        Args:
            progress (float): 전체 학습 진행도 p∈[0,1]. 전역 스텝 기반 권장.

        Returns:
            Tuple[float, float, float]: (w_direct, w_integration, w_constraint)
        """
        p = float(max(0.0, min(1.0, progress)))
        # piecewise-linear for integration weight
        if p < args.p_sat:
            w_int = args.w_int_min + (args.w_int_max -
                                      args.w_int_min) * (p / args.p_sat)
        else:
            w_int = args.w_int_max
        return args.w_dir, w_int, args.w_const

    # ================================================================
    # [추가] (B,Pnn,point_len, C) 형태를 멀티 채널 SG에 넘겨주는 헬퍼
    # ================================================================
    def _sg_derivative_multi_for_points(
        self,
        sequences_points: torch.Tensor,  # (B, Pnn, point_len, C)
        points_valid: torch.Tensor,  # (B, Pnn, point_len) bool
        *,
        dt: float,
        polyorder: int,
        max_window_length: int,
        derivative_order: int = 1,
    ) -> torch.Tensor:
        """(B, Pnn, point_len, C) 모양의 데이터에서 원하는 변화율(1차/2차)을 계산합니다.

        Args:
            sequences_points: (B, Pnn, point_len, C)
            points_valid: (B, Pnn, point_len) bool
            dt: float
            polyorder: int
            max_window_length: int
            derivative_order: 1 또는 2

        Returns:
            derivatives_points: (B, Pnn, point_len, C)
        """
        batch_size, num_neighbors, point_len, num_channels = sequences_points.shape  # (B,Pnn,T,C)

        sequence_flat = sequences_points.reshape(
            batch_size * num_neighbors,
            point_len,
            num_channels,
        )  # (B*Pnn, T, C)
        valid_flat = points_valid.reshape(
            batch_size * num_neighbors,
            point_len,
        )  # (B*Pnn, T)

        derivative_flat = self._savgol_derivative_masked_multi_torch(
            seq_bTC=sequence_flat,  # (B*Pnn, T, C)
            valid_bT=valid_flat,  # (B*Pnn, T)
            dt=dt,
            polyorder=polyorder,
            max_window_length=max_window_length,
            derivative_order=derivative_order,
        )  # (B*Pnn, T, C)

        derivatives_points = derivative_flat.reshape(
            batch_size,
            num_neighbors,
            point_len,
            num_channels,
        )  # (B,Pnn,T,C)
        return derivatives_points

    def _assert_past_cur_valid_mask(
        self,
        valid_bpt: torch.Tensor,
        context: str = "savgol_filter_for_control_past_cur",
    ) -> None:
        """과거~현재(valid_bpt)의 유효 마스크가 행마다 0*1* (단조 증가)인지 검증합니다.

        주의:
            - 이 검증은 디버그용입니다.
            - config.feasible_debug_check_mask=True 일 때만 실행합니다.

        Args:
            valid_bpt: (B, Pnn, T1) bool
            context: 에러 메시지용 위치 정보
        """
        if not bool(getattr(self, "feasible_debug_check_mask", False)):
            return

        assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 이어야 합니다."
        B, Pnn, T1 = valid_bpt.shape
        v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)

        has_10 = (d < 0).any(dim=1)  # 1→0 전이(단조 증가 위반)
        if has_10.any():
            bad_idx = torch.nonzero(has_10, as_tuple=False).flatten()
            max_show = min(int(bad_idx.numel()), 8)
            sample = bad_idx[:max_show].tolist()
            b_list = [(i // Pnn) for i in sample]
            p_list = [(i % Pnn) for i in sample]
            raise ValueError(
                f"[{context}] past_cur 유효 마스크는 0*1* 형태여야 합니다(단조 증가). "
                f"1→0 전이가 감지되었습니다. 오류 row 수={int(bad_idx.numel())}, "
                f"예시 (b,p)={list(zip(b_list, p_list))}.")
