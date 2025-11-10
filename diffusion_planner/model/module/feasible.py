import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union, TypedDict, Optional
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
from typing import Tuple
# feasible.py 최상단 import 근처
from collections import OrderedDict
from typing import Dict, Tuple, Optional


def _yaw_rate_from_cos_sin_via_sg(
    cos_yaw: Tensor,  # (B, Pnn, T1)
    sin_yaw: Tensor,  # (B, Pnn, T1)
    valid_mask: Tensor,  # (B, Pnn, T1) bool
    *,
    dt: float,
    polyorder: int,
    max_window_len: int,
    eps: float = 1e-6,
    sg_derivative_fn=None,
) -> Tensor:
    """SG로 cos/sin을 각각 미분해 각속도 ψ̇를 계산합니다(unwrap 불필요).

    Args:
        cos_yaw: (B, Pnn, T1) cosψ
        sin_yaw: (B, Pnn, T1) sinψ
        valid_mask: (B, Pnn, T1) True=유효
        dt: 샘플 간격(초). 예: 0.1
        polyorder: SG 다항 차수(예: 2)
        max_window_len: SG 윈도 최대 길이(홀수 권장)
        eps: 수치 안정 epsilon
        sg_derivative_fn: (seq_bT, valid_bT, dt, polyorder, max_window_length) -> d/dt(seq_bT)
                          형태의 함수 주입. (당신 코드의 `_savgol_derivative_masked_torch` 전달)

    Returns:
        yaw_rate: (B, Pnn, T1) 각속도(rad/s)
    """
    assert sg_derivative_fn is not None, "sg_derivative_fn을 주입하세요."

    B, Pnn, T1 = cos_yaw.shape

    # 1) (선택) 단위원 재투영으로 안정화
    cs = torch.stack([cos_yaw, sin_yaw], dim=-1)  # (B,Pnn,T1,2)
    norm = torch.linalg.norm(cs, dim=-1,
                             keepdim=True).clamp_min(eps)  # (B,Pnn,T1,1)
    cos_u = (cs[..., 0:1] / norm).squeeze(-1)  # (B,Pnn,T1)
    sin_u = (cs[..., 1:2] / norm).squeeze(-1)  # (B,Pnn,T1)

    # 2) SG 미분 (마스크 인지, 토치 전용)
    dcos = sg_derivative_fn(cos_u.reshape(-1, T1), valid_mask.reshape(-1, T1),
                            dt, polyorder,
                            max_window_len).reshape(B, Pnn, T1)  # (B,Pnn,T1)
    dsin = sg_derivative_fn(sin_u.reshape(-1, T1), valid_mask.reshape(-1, T1),
                            dt, polyorder,
                            max_window_len).reshape(B, Pnn, T1)  # (B,Pnn,T1)

    # 3) ψ̇ = (cos*dsin - sin*dcos) / (cos²+sin²)
    denom = (cos_u * cos_u + sin_u * sin_u).clamp_min(eps)  # (B,Pnn,T1)
    yaw_rate = (cos_u * dsin - sin_u * dcos) / denom  # (B,Pnn,T1)

    # 4) 무효 시점은 0
    if valid_mask is not None:
        yaw_rate = torch.where(valid_mask, yaw_rate, torch.zeros_like(yaw_rate))
    return yaw_rate


# =========================
# [NEW] 하이퍼/제약 파라미터 컨테이너
# =========================
@dataclass
class _ConstraintHParams:
    """제약/적분 하이퍼파라미터(상수 모음)."""
    dt: float
    eps: float
    # 추가 필요: STE용 밴드폭 η (권장 초기값)
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
    """
    v_max_mps: float
    v_max_kmph: float
    a_max_mps2: float
    alpha_max_radps2: float
    a_lat_max_mps2: float
    R_min_m: float
    omega_max_abs_radps: float

    def as_dict(self) -> Dict[str, float]:
        """딕셔너리 형태로 반환."""
        return asdict(self)

    # ---- 파생 값/검사 편의 메서드 ----
    def max_curvature_inv_m(self) -> float:
        """최대 곡률 [1/m] (= 1 / R_min)."""
        return 1.0 / self.R_min_m

    def max_yaw_rate_at_speed(self, speed_mps: float) -> float:
        """주어진 속도에서 허용 가능한 최대 요각속도 [rad/s].

        규정된 절대 한계(omega_max_abs_radps)와
        최소 선회반경 기반 한계(v/R_min) 중 작은 값을 적용.

        Args:
            speed_mps: 속도 [m/s]

        Returns:
            float: 허용 가능한 최대 |w| [rad/s]
        """
        kinematic_limit = speed_mps / self.R_min_m
        return min(kinematic_limit, self.omega_max_abs_radps)

    def feasible_lateral_acc(self, speed_mps: float, radius_m: float) -> float:
        """주어진 속도/곡률반경에서 발생하는 횡가속도 [m/s^2] (명목값).

        a_lat = v^2 / R

        Args:
            speed_mps: 속도 [m/s]
            radius_m: 선회 반경 [m]

        Returns:
            float: 계산된 횡가속도 [m/s^2]
        """
        return (speed_mps**2) / max(radius_m, 1e-9)

    def is_lateral_feasible(self, speed_mps: float, radius_m: float) -> bool:
        """주어진 속도/반경이 횡가속도 한계를 만족하는지 여부."""
        return self.feasible_lateral_acc(speed_mps,
                                         radius_m) <= self.a_lat_max_mps2 + 1e-9

    def clip_speed(self, speed_mps: float) -> float:
        """최대 속도 한계로 속도를 클리핑."""
        return float(np.clip(speed_mps, 0.0, self.v_max_mps))

    def clip_accel(self, accel_mps2: float) -> float:
        """최대 종가속(가감속) 한계로 가속도를 클리핑."""
        return float(np.clip(accel_mps2, -self.a_max_mps2, self.a_max_mps2))

    def clip_yaw_rate(self, yaw_rate_radps: float, speed_mps: float) -> float:
        """주어진 속도에서 요각속도를 허용 범위로 클리핑."""
        limit = self.max_yaw_rate_at_speed(speed_mps)
        return float(np.clip(yaw_rate_radps, -limit, limit))


class FeasibleProjector(nn.Module):

    def __init__(self, hidden_dim: int):
        """Control Correction Network 본체 모듈 정의.

        Notes:
            - 트렁크 은닉 차원 H는 입력으로만 알 수 있으므로,
              trunk 압축기(Compressor)는 첫 forward에서 지연 초기화합니다.
        """
        super().__init__()
        # --- [NEW] Savitzky–Golay 커널 캐시(LRU) ---
        # key: (W, polyorder, deriv_order, dt, dtype, device)
        # val: torch.Tensor of shape (1, 1, W)
        self._sg_kernel_cache: "OrderedDict[Tuple[int, int, int, float, torch.dtype, torch.device], torch.Tensor]" = OrderedDict(
        )
        self._sg_kernel_cache_cap: int = 256  # 필요시 조절(메모리-속도 트레이드오프)
        # [추가 요망] SG 위치별(one‑sided/중앙) 가중치 캐시(LRU)
        # key: (W, polyorder, deriv_order, m, dt, dtype, device)  → val: (W,) weights
        self._sg_pos_cache: "OrderedDict[Tuple[int, int, int, int, float, torch.dtype, torch.device], torch.Tensor]" = OrderedDict(
        )
        self._sg_pos_cache_cap: int = 2048  # 필요시 조절

        self.constraints_h_params = _ConstraintHParams(
            dt=0.1,
            eps=1e-6,
            # 추가 필요: η 초기값(S0~S4)
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
                    v_max_mps=8.0,
                    v_max_kmph=28.8,
                    a_max_mps2=6.0,
                    alpha_max_radps2=8.0,
                    a_lat_max_mps2=5.0,
                    R_min_m=0.00001,
                    omega_max_abs_radps=3.6,
                ),
            ActorClass.BICYCLE:
                DynamicLimits(
                    v_max_mps=20.0,
                    v_max_kmph=72.0,
                    a_max_mps2=6.0,
                    alpha_max_radps2=3.0,
                    a_lat_max_mps2=6.5,
                    R_min_m=1.50,
                    omega_max_abs_radps=1.5,
                ),
            ActorClass.CAR:
                DynamicLimits(
                    v_max_mps=55.6,
                    v_max_kmph=200.0,
                    a_max_mps2=8.0,
                    alpha_max_radps2=2.5,
                    a_lat_max_mps2=8.0,
                    R_min_m=4.50,
                    omega_max_abs_radps=0.9,
                ),
        }

        # ------------------------------
        # 아키텍처 하이퍼파라미터(고정 폭)
        # ------------------------------
        self._Dx: int = 48  # state encoder 출력 채널 (prev/fut 각각)
        self._Du: int = 32  # control adapter 출력 채널
        self._Dc: int = 64  # trunk compressor 출력 채널
        self._Din: int = self._Dx * 2 + self._Du + self._Dc  # 48+48+32+64=192
        self._C: int = self._Din  # 메인 채널 폭(192)
        self._eps: float = 1e-6

        # [추가 필요] L_integration 경로 차단용 플래그 (state, u_base detach)
        self.detach_state_and_u_for_ctrl_losses: bool = True  # 추가 필요

        # ------------------------------
        # Encoders (시간축 보존, 채널만 변환)
        # ------------------------------
        # 상태 인코더(현재 노드 X_prev: (x,y,cos,sin))
        self.state_prev_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 48),
            nn.GELU(),
            nn.Linear(48, self._Dx),
        )
        # 상태 인코더(오른쪽 노드 X_fut)
        self.state_fut_encoder = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, 48),
            nn.GELU(),
            nn.Linear(48, self._Dx),
        )
        # 베이스 제어 어댑터(U_base: (vxb,vyb,w) — 정규화 값)
        self.control_adapter = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self._Du),
            nn.GELU(),
        )

        # 트렁크(디퓨전 은닉) 압축기는 H를 알아야 하므로 지연 초기화
        # (B,Pnn,H)->(B,Pnn,Dc)
        self.trunk_compressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 192),
            nn.GELU(),
            nn.Linear(192, self._Dc),
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
        self._kernel_size: int = 7
        self._dilations: List[int] = [1, 2, 4, 8]
        self.tcn_depth = len(self._dilations)
        self.tcn_pre_lns = nn.ModuleList(
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
        # pointwise 1x1 (시간축 보존, 채널 결합)
        self.tcn_linear = nn.ModuleList(
            [nn.Linear(self._C, self._C) for _ in range(self.tcn_depth)])

        # ------------------------------
        # Head & Gate
        # ------------------------------
        # 잔차 초안 ΔU_raw
        self.head = nn.Sequential(
            nn.Linear(self._C, self._C),
            nn.GELU(),
            nn.Linear(self._C, 3),
        )
        # 마지막 Linear 0-init → 초기엔 U_ref ≈ U_base
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        # 소프트 게이트 s = softplus(MLP_g(Z_s))
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(self._C),
            nn.Linear(self._C, 128),
            nn.GELU(),
            nn.Linear(128, 3),
        )
        # gate 초기 스케일 s0 설정(보수적으로)
        s0 = 0.05
        b_init = math.log(math.exp(float(s0)) - 1.0)  # softplus^{-1}(s0)
        with torch.no_grad():
            self.gate_mlp[-1].bias.fill_(b_init)

    def _get_savgol_pos_weights_cached(
        self,
        *,
        window_length: int,
        polyorder: int,
        deriv_order: int,
        dt: float,
        m: int,  # 창 내부 평가 위치(0..W-1)
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """창 길이 W에서 위치 m(0..W-1)의 k차 미분 SG 가중치 벡터를 반환.

        Returns:
            torch.Tensor: (W,)  — 창 샘플과 내적하면 해당 위치의 미분 근사값.
        """
        master_key = (int(window_length), int(polyorder), int(deriv_order),
                      int(m), float(dt))
        if master_key in self._sg_pos_cache:
            w_cpu_fp32 = self._sg_pos_cache.pop(master_key)
            self._sg_pos_cache[master_key] = w_cpu_fp32
        else:
            with torch.no_grad():
                W = int(window_length)  # 추가 요망!
                work_dtype = torch.float64
                x = torch.arange(
                    0, W, device=torch.device("cpu"),
                    dtype=work_dtype) - float(m)
                V = torch.stack([x**i for i in range(polyorder + 1)], dim=1)
                pinv = torch.linalg.pinv(V)
                e = torch.zeros(polyorder + 1,
                                device=torch.device("cpu"),
                                dtype=work_dtype)
                e[deriv_order] = math.factorial(deriv_order)
                w = (e @ pinv) / (dt**deriv_order)  # float64
                w_cpu_fp32 = w.to(torch.float32).contiguous()  # CPU/FP32로 보관
                if len(self._sg_pos_cache) >= self._sg_pos_cache_cap:
                    self._sg_pos_cache.popitem(last=False)
                self._sg_pos_cache[master_key] = w_cpu_fp32
        with torch.no_grad():
            return w_cpu_fp32.to(device=device, dtype=dtype)

    def _get_savgol_diff_kernel_cached(
        self,
        window_length: int,
        polyorder: int,
        deriv_order: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """사비츠키–골레이 1D 미분 커널을 LRU 캐시로 제공.

        Args:
            window_length (int): 커널 길이 W(홀수).
            polyorder (int): 다항 차수.
            deriv_order (int): 미분 차수(보통 1).
            dt (float): 샘플 간격(초). 커널 값에 직접 반영됨.
            device (torch.device): 커널을 올릴 디바이스.
            dtype (torch.dtype): 커널 dtype.

        Returns:
            torch.Tensor: (1, 1, W) 커널. requires_grad=False.
        """
        master_key = (int(window_length), int(polyorder), int(deriv_order),
                      float(dt))
        # 1) 조회(저장은 CPU/FP32)
        if master_key in self._sg_kernel_cache:
            kernel_cpu_fp32 = self._sg_kernel_cache.pop(master_key)
            self._sg_kernel_cache[master_key] = kernel_cpu_fp32
        else:
            # 2) 생성(항상 CPU/FP32)
            with torch.no_grad():

                kernel_cpu_fp32 = self._build_savgol_diff_kernel(
                    window_length=window_length,
                    polyorder=polyorder,
                    deriv_order=deriv_order,
                    dt=dt,
                    device=torch.device("cpu"),
                    dtype=torch.float32)
                if len(self._sg_kernel_cache) >= self._sg_kernel_cache_cap:
                    self._sg_kernel_cache.popitem(last=False)
                self._sg_kernel_cache[master_key] = kernel_cpu_fp32
        # 3) 요청 형식/장치로 변환해 반환
        with torch.no_grad():
            return kernel_cpu_fp32.to(device=device, dtype=dtype)

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
        y = y_sur + (y_hard - y_sur).detach()  # 추가 필요
        return y

    @staticmethod
    def _ste_radial_clip(vx: torch.Tensor, vy: torch.Tensor,
                         v_max: torch.Tensor, eta: float,
                         eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """라디얼(벡터) STE-clip."""
        v_max = v_max.to(dtype=vx.dtype, device=vx.device)
        speed = torch.sqrt(vx * vx + vy * vy + eps)
        s_hard = torch.clamp(v_max / speed.clamp_min(eps), max=1.0)
        vx_h, vy_h = s_hard * vx, s_hard * vy
        r = speed / (v_max + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        vx_sur, vy_sur = w * vx + (1 - w) * vx.detach(), w * vy + (
            1 - w) * vy.detach()
        vx_out = vx_sur + (vx_h - vx_sur).detach()  # 추가 필요
        vy_out = vy_sur + (vy_h - vy_sur).detach()  # 추가 필요
        return vx_out, vy_out

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
        r = omega_raw.abs() / (allow.detach() + eps)  # 추가 필요: allow detach 반영
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * omega_raw + (1 - w) * omega_raw.detach()
        y = y_sur + (y_hard - y_sur).detach()  # 추가 필요

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
        s = s_sur + (s_hard - s_sur).detach()  # 추가 필요
        return s

    # ------------------------------------------------------------------

    @classmethod
    def loss_weights_by_progress(cls,
                                 progress: float) -> Tuple[float, float, float]:
        """손실 가중치 스케줄러.

        Args:
            progress (float): 전체 학습 진행도 p∈[0,1]. 전역 스텝 기반 권장.

        Returns:
            Tuple[float, float, float]: (w_direct, w_integration, w_constraint)
        """
        p = float(max(0.0, min(1.0, progress)))
        # Constants
        p_sat = 0.60
        w_dir = 1.00
        w_int_min, w_int_max = 0.05, 2.00
        w_const = 0.02
        # piecewise-linear for integration weight
        if p <= p_sat:
            w_int = w_int_min + (w_int_max - w_int_min) * (p / p_sat)
        else:
            w_int = w_int_max
        return w_dir, w_int, w_const

    # ------------------------------------------------------------------
    # [ADD] 유틸리티 메서드: 두 값의 가중 평균 / 세계→바디 회전
    # ------------------------------------------------------------------
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

    def compute_midpoint_controls(
        self,
        unnorm_diffusion_trajectory: torch.Tensor,
        # (B, Pnn, 1+T, 4) [x,y,cos,sin] (세계/ego 프레임)
        unnorm_cur_future_control: torch.Tensor,
        # (B, Pnn, 1+T, 3) [v_x^w,v_y^w, w]
        near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)    (bool/0-1)
    ) -> torch.Tensor:  # (B, Pnn, T, 3)   [v_x^b, v_y^b, w]_mid
        """구간 [t_k, t_{k+1})의 **중점(midpoint) 제어**를 계산한다.

        논리/수식
        --------
        - 입력 속도 [v_x, v_y]는 **세계(ego) 프레임** 기준이라고 가정.
        - 서로 다른 시점의 body 프레임이 회전하므로,
            중간 속도는 **세계 프레임에서 두 끝점 속도를 가중 평균**하고,
            그 후 **중간 yaw(두 끝점의 cos/sin 가중 평균 → 정규화)** 으로 **몸체 프레임으로 회전**한다.
        - 요각속도 w는 세계 프레임 스칼라이므로 **끝점 가중 평균**으로 충분.
        - 유효성 마스크는 구간 양 끝점에 대해 가중치로 활용(둘 다 무효면 0).

        Args:
            unnorm_diffusion_trajectory: (B,Pnn,1+T,4) = [x, y, cos, sin] (세계/ego 기준)
            unnorm_cur_future_control:   (B,Pnn,1+T,3) = [v_x^w, v_y^w, w] (세계 기준)
            near_cur_future_valid:       (B,Pnn,1+T)   = True/False

        Returns:
            torch.Tensor: (B,Pnn,T,3) = [v_x^b_mid, v_y^b_mid, w_mid]
        """

        B, Pnn, T1, _ = unnorm_diffusion_trajectory.shape
        if T1 < 2:
            # 중점이 성립하려면 최소 2 스텝 필요
            raise ValueError("타임스텝이 2 미만이면 중점 제어를 계산할 수 없습니다.")
        eps = 1e-6

        # ---- 입력 분해
        cos_all = unnorm_diffusion_trajectory[..., 2]  # (B,Pnn,T1)
        sin_all = unnorm_diffusion_trajectory[..., 3]  # (B,Pnn,T1)
        v_w = unnorm_cur_future_control[..., :2]  # (B,Pnn,T1,2) = [vx^w,vy^w]
        omega = unnorm_cur_future_control[..., 2]  # (B,Pnn,T1)
        valid = (near_cur_future_valid
                 > 0).to(v_w.dtype)  # (B,Pnn,T1) as float {0,1}

        # 좌/우 끝점 쪼개기
        v_x_start_w, v_y_start_w = v_w[..., :-1, 0], v_w[..., :-1,
                                                         1]  # (B,Pnn,T)
        v_x_end_w, v_y_end_w = v_w[..., 1:, 0], v_w[..., 1:, 1]  # (B,Pnn,T)
        start_valid, end_valid = valid[..., :-1], valid[..., 1:]  # (B,Pnn,T)
        omega_start, omega_end = omega[..., :-1], omega[..., 1:]  # (B,Pnn,T)

        # ---- 세계 프레임에서 속도 중점(가중 평균)
        # (B,Pnn,T)
        v_x_mid_w = self._weighted_avg_two(
            v_x_start_w,  # (B,Pnn,T)
            v_x_end_w,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            eps=eps)
        v_y_mid_w = self._weighted_avg_two(
            v_y_start_w,  # (B,Pnn,T)
            v_y_end_w,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            eps=eps)  # (B,Pnn,T)

        # ---- 중간 yaw (cos/sin 가중평균 → 정규화)
        cos_start, sin_start = cos_all[..., :-1], sin_all[..., :-1]  # (B,Pnn,T)
        cos_end, sin_end = cos_all[..., 1:], sin_all[..., 1:]  # (B,Pnn,T)
        # (B,Pnn,T)
        cos_mid = self._weighted_avg_two(
            cos_start,  # (B,Pnn,T)
            cos_end,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            eps=eps)
        sin_mid = self._weighted_avg_two(sin_start, sin_end, start_valid,
                                         end_valid)
        norm = (cos_mid * cos_mid + sin_mid * sin_mid).clamp_min(eps).sqrt()
        cos_mid = cos_mid / norm
        sin_mid = sin_mid / norm

        # ---- 세계→바디 회전
        vxb_mid, vyb_mid = self._world_to_body(v_x_mid_w, v_y_mid_w, cos_mid,
                                               sin_mid)  # (B,Pnn,T)

        # ---- w 중점(가중 평균, 스칼라)
        omega_mid = self._weighted_avg_two(
            omega_start,  # (B,Pnn,T)
            omega_end,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            eps=eps)  # (B,Pnn,T)

        seg_is_valid = ((start_valid > 0) & (end_valid > 0)).to(vxb_mid.dtype)
        vxb_mid = vxb_mid * seg_is_valid
        vyb_mid = vyb_mid * seg_is_valid
        omega_mid = omega_mid * seg_is_valid

        # (B,Pnn,T,3)로 합치기
        return torch.stack([vxb_mid, vyb_mid, omega_mid], dim=-1)

    def forward(
            self,
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)  bool
            diffusion_trajectory: torch.
        Tensor,  # (B, Pnn, 1+T, 4) [x,y,cos,sin]
            cur_future_seg_body_control: torch.
        Tensor,  # (B, Pnn, T, 3)   U_base (정규화)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
    ) -> torch.Tensor:  # (B, Pnn, T, 3)   U_ref (정규화)
        """Control Correction Network 전체 경로를 짧게 연결.
        파이프라인:
            mask → prev/fut 분리(+unit-circle) → feature concat → Stem → TCN →
            Head+Gate → U_base와 잔차 결합
        """
        self._assert_prefix_valid_mask(near_cur_future_valid, context="forward")

        # 1) 마스크
        """
            - seg_mask: (B, Pnn, T) float(0/1)
            - seg_mask_1: (B, Pnn, T, 1)  → 브로드캐스트 편의를 위한 채널 차원 추가
        """
        seg_mask, seg_mask_1 = self._build_segment_mask(near_cur_future_valid)

        # 2) 입력 분해 + 단위원 재투영
        # (B, Pnn, T, 4)  ← t_k # (B, Pnn, T, 4)  ← t_{k+1}
        x_prev, x_fut = self._split_prev_fut(
            diffusion_trajectory)  # diffusion_trajectory: (B, Pnn, 1+T, 4)
        x_prev = self._normalize_cos_sin(x_prev)
        x_fut = self._normalize_cos_sin(x_fut)

        # 3) 피처 인코딩
        # Z_in: (B, Pnn, T, 192)
        Z_in = self._features_from_inputs(
            x_prev=x_prev, # (B, Pnn, T, 4)
            x_fut=x_fut, # (B, Pnn, T, 4)
            u_base=cur_future_seg_body_control, # (B, Pnn, T, 3)
            dit_final_hidden_tokens=dit_final_hidden_tokens, # (B, Pnn, H)
            near_cur_future_valid=near_cur_future_valid, # (B, Pnn, 1+T)
        )

        # 4) Stem → 5) TCN
        # Z_s: (B, Pnn, T, 192)
        Z_s = self._prepare_tcn_input(Z_in,
                               seg_mask_1)  # (B, Pnn, T, 192) # (B, Pnn, T, 1)
        # Z_tcn: (B, Pnn, T, 192)
        Z_tcn = self._run_tcn(
            Z_s, seg_mask,
            seg_mask_1)  # (B, Pnn, T, 192) # (B, Pnn, T) # (B, Pnn, T, 1)

        # 6) ΔU 예측 → 7) 결합
        # delta_u: (B, Pnn, T, 3)
        delta_u = self._predict_delta_u(Z_tcn, seg_mask_1)
        # u_ref: (B, Pnn, T, 3)
        u_ref = cur_future_seg_body_control + delta_u
        return u_ref

    # ------------------------------
    # 내부: mask‑aware depthwise conv (정규화 합성곱)
    # ------------------------------
    def _depthwise_conv_masked(
        self,
        seq_bptc: torch.Tensor,  # (B,Pnn,T,C)
        seg_mask_bpt: torch.Tensor,  # (B,Pnn,T), float(0/1)
        block_idx: int,
    ) -> torch.Tensor:
        """정규화 depthwise conv.

        Args:
            seq_bptc: (B,Pnn,T,C) 입력(마스크가 적용된 값 권장)
            seg_mask_bpt: (B,Pnn,T) 0/1 마스크
            block_idx: 블록 인덱스(0..3)

        Returns:
            (B,Pnn,T,C) 동일 길이 출력
        """
        B, Pnn, T, C = seq_bptc.shape
        conv = self.tcn_dw[block_idx]
        # (B*Pnn, C, T)
        x = seq_bptc.permute(0, 1, 3, 2).reshape(B * Pnn, C, T)
        m = seg_mask_bpt.reshape(B * Pnn, 1, T)  # (B*Pnn,1,T)

        # 분자: (x * m) 에 depthwise conv
        y_num = conv(x * m)  # (B*Pnn, C, T)

        # 분모: mask에 ones-kernel
        k = conv.kernel_size[0]
        pad = conv.padding[0]
        dil = conv.dilation[0]
        ones = torch.ones((1, 1, k), device=x.device, dtype=x.dtype)
        den = F.conv1d(m, ones, padding=pad, dilation=dil)  # (B*Pnn,1,T)
        y = y_num / den.clamp_min(self._eps)  # 브로드캐스트로 채널 공용 분모

        # (B,Pnn,T,C)
        y = y.view(B, Pnn, C, T).permute(0, 1, 3, 2).contiguous()
        return y

    # =========================================================
    # [A] 마스크/입력 전처리
    # =========================================================
    def _build_segment_mask(self, near_cur_future_valid):
        valid = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        seg_valid = (valid[..., :-1] & valid[..., 1:])  # (B,Pnn,T)  # ← AND
        seg_mask = seg_valid.to(torch.float32)
        seg_mask_1 = seg_mask.unsqueeze(-1)
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

    # =========================================================
    # [B] 피처 인코딩 (prev/fut/u_base/trunk)
    # =========================================================
    def _features_from_inputs(
            self,
            x_prev: torch.Tensor,  # (B, Pnn, T, 4)
            x_fut: torch.Tensor,  # (B, Pnn, T, 4)
            u_base: torch.Tensor,  # (B, Pnn, T, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)
    ) -> torch.Tensor:
        """입력을 통일 피처 Z_in으로 변환.

        파이프라인:
            prev/fut 인코딩(48) + control 어댑터(32) + trunk 압축(64) → concat(192)

        Args:
            x_prev (torch.Tensor): (B, Pnn, T, 4)
            x_fut (torch.Tensor): (B, Pnn, T, 4)
            u_base (torch.Tensor): (B, Pnn, T, 3)
            dit_final_hidden_tokens (torch.Tensor): (B, Pnn, H)

        Returns:
            torch.Tensor: (B, Pnn, T, 192)  = Z_in
        """
        # 함수 본문 초입에 추가
        valid = near_cur_future_valid.to(x_prev.dtype)  # (B,Pnn,1+T)  # 추가 요망!
        start_valid = valid[..., :-1].unsqueeze(-1)  # (B,Pnn,T,1)  # 추가 요망!
        end_valid = valid[..., 1:].unsqueeze(-1)  # (B,Pnn,T,1)  # 추가 요망!

        # prev/fut 노드 유효성으로 입력 자체 0화 (LN/Linear 이전 차단)
        x_prev = x_prev * start_valid  # 추가 요망!
        x_fut = x_fut * end_valid  # 추가 요망!

        # 세그먼트 유효성(AND)로 u_base 0화
        seg_mask, seg_mask_1 = self._build_segment_mask(
            near_cur_future_valid)  # 추가 요망!
        seg_mask_1 = seg_mask_1.to(u_base.dtype)  # 추가 요망!

        if self.detach_state_and_u_for_ctrl_losses:  # 추가 필요
            x_prev = x_prev.detach()
            x_fut = x_fut.detach()
            u_base_in = (u_base.detach()) * seg_mask_1
        else:
            u_base_in = u_base * seg_mask_1

        feat_prev = self.state_prev_encoder(x_prev)  # (B,Pnn,T,48)
        feat_fut = self.state_fut_encoder(x_fut)  # (B,Pnn,T,48)
        feat_u = self.control_adapter(u_base_in)  # (B,Pnn,T,32)
        # dit_final_hidden_tokens: (B,Pnn,H)
        # trunk: (B,Pnn,64)
        trunk = self.trunk_compressor(
            dit_final_hidden_tokens.detach())
        trunk_rep = trunk.unsqueeze(2).expand(-1, -1, x_prev.size(2),
                                              -1)  # (B,Pnn,T,64)

        Z_in = torch.cat([feat_prev, feat_fut, feat_u, trunk_rep],
                         dim=-1)  # (B,Pnn,T,192)
        return Z_in

    # =========================================================
    # [C] Stem / TCN 백본
    # =========================================================
    def _prepare_tcn_input(
        self,
        Z_in: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """Stem: LayerNorm → 1×1 Linear → GELU (시간 길이 보존)

        Args:
            Z_in (torch.Tensor): (B, Pnn, T, 192)
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 192)  (무효 시점 0-클램프 적용)
        """
        Z_s = self.stem_act(self.stem_fc(self.stem_norm(Z_in)))  # (B,Pnn,T,192)
        Z_s = Z_s * seg_mask_1
        return Z_s

    def _run_tcn(
        self,
        Z_s: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask: torch.Tensor,  # (B, Pnn, T)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """TCN 4블록 실행: [Pre-LN → depthwise(mask‑aware) → GELU → 1×1 → Residual] ×4

        Args:
            Z_s (torch.Tensor): (B, Pnn, T, 192)  Stem 출력
            seg_mask (torch.Tensor): (B, Pnn, T)  0/1
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 192)
        """
        Z = Z_s
        for block_idx in range(self.tcn_depth):
            # Z_ln: (B,Pnn,T,192)
            Z_ln = self.tcn_pre_lns[block_idx](Z) * seg_mask_1  # Pre-LN + mask
            # Y: (B,Pnn,T,192)
            Y = self._depthwise_conv_masked(Z_ln, seg_mask,
                                            block_idx)  # mask-aware
            Y = F.gelu(Y)
            Y = self.tcn_linear[block_idx](Y)  # pointwise 1×1
            Y = Y * seg_mask_1  # [추가] 블록 내부 중간단계에서도 0 고정
            Z = (Z + Y) * seg_mask_1  # Residual + mask
        return Z  # (B,Pnn,T,192)

    # =========================================================
    # [D] Head & Gate / 결합
    # =========================================================
    def _predict_delta_u(
        self,
        Z_tcn: torch.Tensor,  # (B, Pnn, T, 192)
        seg_mask_1: torch.Tensor  # (B, Pnn, T, 1)
    ) -> torch.Tensor:
        """잔차 제어 ΔU 산출: Head(초안) + softplus 게이트 스케일.

        ΔU = softplus(MLP_g(Z_s)) ⊙ tanh(Head(Z_tcn))

        Args:
            Z_tcn (torch.Tensor): (B, Pnn, T, 192)  TCN 출력
            Z_s (torch.Tensor): (B, Pnn, T, 192)  Stem 출력(게이트 입력)
            seg_mask_1 (torch.Tensor): (B, Pnn, T, 1)

        Returns:
            torch.Tensor: (B, Pnn, T, 3)  (무효 시점은 0)
        """
        delta_u_raw = self.head(Z_tcn)  # (B,Pnn,T,3)
        gate_scale = F.softplus(self.gate_mlp(Z_tcn.detach()))  # (B,Pnn,T,3)
        delta_u = gate_scale * torch.tanh(delta_u_raw)  # (B,Pnn,T,3)
        delta_u = delta_u * seg_mask_1  # 무효 시점 보정 0
        return delta_u

    def _fuse_controls(
            self,
            u_base: torch.Tensor,  # (B, Pnn, T, 3)
            delta_u: torch.Tensor,  # (B, Pnn, T, 3)
    ) -> torch.Tensor:
        """최종 결합: U_ref = U_base + ΔU

        Args:
            u_base (torch.Tensor): (B, Pnn, T, 3)  베이스 제어(정규화)
            delta_u (torch.Tensor): (B, Pnn, T, 3)  잔차 보정

        Returns:
            torch.Tensor: (B, Pnn, T, 3)  보정된 제어
        """
        return u_base + delta_u
        # ----------------------------
        # [UTIL+] 부드러운 max (LSE)
        # ----------------------------

    @staticmethod
    def _smooth_max(a: torch.Tensor,
                    b: torch.Tensor,
                    alpha: float = 10.0) -> torch.Tensor:
        """log-sum-exp로 근사한 smooth max.

        m(a,b) ≈ (1/α) log( e^{αa} + e^{αb} )
        """
        m = torch.maximum(a, b)
        return m + (torch.exp(alpha *
                              (a - m)) + torch.exp(alpha *
                                                   (b - m))).log() / alpha

    # ----------------------------
    # [NEW] per-agent 제약 텐서 빌드
    # ----------------------------
    def _build_per_agent_limits(
        self,
        near_class_one_hot: torch.Tensor,
        # (B,Pnn,3) 0:vehicle(CAR),1:ped,2:bicycle
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """클래스별 스칼라 제약치를 (B,Pnn) 텐서로 확장.

        Args:
            near_class_one_hot: (B, Pnn, 3)
             one‑hot (0: vehicle, 1: pedestrian, 2: bicycle)

        Returns:
            Dict[str, Tensor]:
            v_max/a_max/alpha_max/a_lat_max/R_min/omega_abs_max/a_x_max/a_y_max, is_nonholonomic
                * 모두 (B,Pnn) 모양
        """
        # 순서 주의: [vehicle(CAR), pedestrian, bicycle]
        car: DynamicLimits = self.constraints[ActorClass.CAR]
        ped: DynamicLimits = self.constraints[ActorClass.PEDESTRIAN]
        bic: DynamicLimits = self.constraints[ActorClass.BICYCLE]

        # 클래스별 상수 → 길이 3 텐서
        def cvec(getattr_name: str) -> torch.Tensor:
            vals = torch.tensor([
                getattr(car, getattr_name),
                getattr(ped, getattr_name),
                getattr(bic, getattr_name)
            ],
                                device=device,
                                dtype=dtype)  # (3,)
            # (B,Pnn,3) @ (3,) -> (B,Pnn)
            return (near_class_one_hot * vals).sum(dim=-1)

        v_max_bp = cvec("v_max_mps")
        a_max_bp = cvec("a_max_mps2")
        alpha_max_bp = cvec("alpha_max_radps2")
        a_lat_max_bp = cvec("a_lat_max_mps2")
        R_min_bp = cvec("R_min_m")
        omega_abs_bp = cvec("omega_max_abs_radps")

        # 마찰원 반경: 종/횡 가속 한계
        a_x_max_bp = a_max_bp
        a_y_max_bp = a_lat_max_bp

        # 보행자는 S0(비홀로노믹) 비활성, 자전거/차는 활성
        # near_class_one_hot[..., 1] 이 pedestrian
        is_ped = near_class_one_hot[..., 1] > 0.5  # (B,Pnn) bool
        is_nonholonomic = ~is_ped

        return dict(
            v_max=v_max_bp,  # (B,Pnn)
            a_max=a_max_bp,  # (B,Pnn)
            alpha_max=alpha_max_bp,  # (B,Pnn)
            a_lat_max=a_lat_max_bp,  # (B,Pnn)
            R_min=R_min_bp,  # (B,Pnn)
            omega_abs_max=omega_abs_bp,  # (B,Pnn)
            a_x_max=a_x_max_bp,  # (B,Pnn)
            a_y_max=a_y_max_bp,  # (B,Pnn)
            is_nonholonomic=is_nonholonomic,  # (B,Pnn) bool
        )

    # ----------------------------
    # [NEW] 제약/적분 하이퍼 고정값
    # ----------------------------

    # ----------------------------
    # [MOD] (S0) 비홀로노믹: 마스크 지원
    # ----------------------------
    def _apply_S0_nonholonomic(
            self,
            vx_b: torch.Tensor,  # (B,Pnn)
            vy_b: torch.Tensor,  # (B,Pnn)
            slip_epsilon: float,
            soft: bool,
            is_nonholonomic: torch.Tensor,  # (B,Pnn) bool, True면 S0 적용
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """(S0) 비홀로노믹: 횡슬립 억제 v_y^b ≈ 0 (클래스별 on/off)."""
        if soft:
            vy_new = slip_epsilon * torch.tanh(vy_b / max(slip_epsilon, 1e-12))
        else:
            vy_new = torch.clamp(vy_b, -slip_epsilon, slip_epsilon)
        vy_b = torch.where(is_nonholonomic, vy_new, vy_b)
        return vx_b, vy_b

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
    def _wrap_to_pi(angle_rad: torch.Tensor) -> torch.Tensor:
        """[-pi, pi]로 래핑.

        Args:
            angle_rad (torch.Tensor): 라디안 값 [...].

        Returns:
            torch.Tensor: 같은 shape, [-pi, pi] 래핑.
        """
        return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))

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

    # =========================================================
    # [S0~S4] 제약 적용: **STE 버전** (forward=hard, backward=surrogate)
    # =========================================================
    def _apply_S0_nonholonomic_ste(
        self,
        vx_b: torch.Tensor,  # (B,Pnn)
        vy_b: torch.Tensor,  # (B,Pnn)
        slip_epsilon: float,
        eta: float,  # 0.07
        eps: float,
        is_nonholonomic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S0: v_y hard-clip(+STE). 보행자 제외."""
        limit = torch.full_like(vy_b, float(slip_epsilon))  # (B,Pnn)
        vy_new = self._ste_scalar_clip(vy_b, limit, eta, eps)  # (B,Pnn)
        vy_out = torch.where(is_nonholonomic, vy_new, vy_b)
        return vx_b, vy_out

    def _apply_S1_speed_limit_ste(
            self, vx_b: torch.Tensor, vy_b: torch.Tensor, v_max: torch.Tensor,
            eta: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """라디얼(벡터) STE-clip."""
        v_max = v_max.to(dtype=vx_b.dtype, device=vx_b.device)
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        s_hard = torch.clamp(v_max / speed.clamp_min(eps), max=1.0)  # (B,Pnn)
        vx_h, vy_h = s_hard * vx_b, s_hard * vy_b  # (B,Pnn)
        r = speed / (v_max + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        vx_sur, vy_sur = w * vx_b + (1 - w) * vx_b.detach(), w * vy_b + (
            1 - w) * vy_b.detach()
        vx_out = vx_sur + (vx_h - vx_sur).detach()  # 추가 필요
        vy_out = vy_sur + (vy_h - vy_sur).detach()  # 추가 필요
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

    # [추가 요망] (S3: 속도-연동 각속도 한계 — w clip, no slip angle)
    def _apply_S3_omega_clip_ste(
        self,
        vx_b: torch.Tensor,  # (B,Pnn)
        vy_b: torch.Tensor,  # (B,Pnn)
        omega: torch.Tensor,  # (B,Pnn)
        a_lat_max: torch.Tensor,  # (B,Pnn)
        R_min: torch.Tensor,  # (B,Pnn)
        omega_abs_max: torch.Tensor,  # (B,Pnn)
        is_nonholonomic: torch.Tensor,  # (B,Pnn)  True: 차/자전거, False: 보행자
        eta: float,
        eps: float,
    ) -> torch.Tensor:
        """(S3) β/Δβ 없이, 속도-연동 w 한계로 직접 clip.

        w_allow (차/자전거) = min( a_lat_max/|v|, |v|/R_min, w_abs_max )
        w_allow (보행자)     = min( a_lat_max/|v|, w_abs_max )   # 제자리 회전 허용
        """
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        allow_lat = a_lat_max / (speed + eps)  # (B,Pnn)
        allow_R = speed / (R_min + eps)  # (B,Pnn)
        allow_abs = omega_abs_max

        # 비홀로노믹이면 R_min 항 포함, 보행자는 제외
        allow_nonh = torch.minimum(torch.minimum(allow_lat, allow_R), allow_abs)
        allow_holo = torch.minimum(allow_lat, allow_abs)
        allow = torch.where(is_nonholonomic, allow_nonh, allow_holo)  # (B,Pnn)

        # forward: hard clamp, backward: band-weighted surrogate (allow는 detach)
        # omega_new: (B,Pnn)
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

    # ----------------------------
    # [NEW] 한 스텝: 제약 S0~S4 적용(모두 STE 버전 호출)
    # ----------------------------
    def _apply_constraints_step(
        self,
        vx_b_prev: torch.Tensor,  # (B,Pnn)
        vy_b_prev: torch.Tensor,  # (B,Pnn)
        omega_prev: torch.Tensor,  # (B,Pnn)
        vx_b_k: torch.Tensor,  # (B,Pnn)
        vy_b_k: torch.Tensor,  # (B,Pnn)
        omega_k: torch.Tensor,  # (B,Pnn)
        hp: _ConstraintHParams,
        key_to_limit_bp: Dict[str, torch.Tensor],
        slip_epsilon: float = 0.20,
        apply_S2: bool = True,
        apply_S4_ax: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ key_to_limit_bp
        Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn)

        v_max/a_max/alpha_max/a_lat_max/R_min
        /omega_abs_max/a_x_max/a_y_max, is_nonholonomic

        self.constraints_h_params = _ConstraintHParams(
            dt=0.1,
            eps=1e-6,
            # 추가 필요: η 초기값(S0~S4)
            eta_slip=0.07,
            eta_speed=0.05,
            eta_inc=0.10,
            eta_yaw=0.05,
            eta_fric=0.05,
        )
        """
        # (S0)
        vx_b_k, vy_b_k = self._apply_S0_nonholonomic_ste(
            vx_b_k,  # (B,Pnn)
            vy_b_k,  # (B,Pnn)
            slip_epsilon,
            hp.eta_slip,  # 0.07
            hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"])  #
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
        # (S4)
        # vx_b_k, vy_b_k, omega_k = self._apply_S4_friction_circle_ste(
        #     vx_b_prev,
        #     vy_b_prev,
        #     vx_b_k,
        #     vy_b_k,
        #     omega_k,
        #     key_to_limit_bp["a_x_max"],
        #     key_to_limit_bp["a_y_max"],
        #     hp.dt,
        #     hp.eta_fric,
        #     hp.eps,
        #     use_ax=apply_S4_ax)
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
                - unnorm_integrated_trajectory: (B,Pnn,T,4)  # 노드 기반 → mask_node[...,1:]로 마스킹
                - unnorm_control_constraint_diff: (B,Pnn,T,3)  # 구간 기반 → mask_interval로 마스킹
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

    @staticmethod
    def _radial_hard_clip(
        vx: torch.Tensor,  # (B,Pnn) 또는 (...,)
        vy: torch.Tensor,  # (B,Pnn) 또는 (...,)
        v_max: Union[float, torch.Tensor],  # (B,Pnn) 또는 스칼라
        eps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """속도 벡터의 L2 노름을 v_max로 **하드** 제한(방사형 스케일)."""
        speed = torch.sqrt(vx * vx + vy * vy).clamp_min(eps)
        if not torch.is_tensor(v_max):
            v_max_t = torch.tensor(v_max, device=vx.device, dtype=vx.dtype)
        else:
            v_max_t = v_max.to(device=vx.device, dtype=vx.dtype)
        scale = torch.clamp(v_max_t / speed.clamp_min(eps), max=1.0)
        return vx * scale, vy * scale

    # ==========================================
    # 4) S2 (증분 제한) soft clip 수정
    # ==========================================

    @staticmethod
    def _increment_hard_clip(
        delta: torch.Tensor,  # (...,) 또는 (...,D)
        limit: Union[float, torch.Tensor],  # (...,)
        eps: float,
    ) -> torch.Tensor:
        """증분(스칼라/벡터 노름) 하드 클립.

        - 스칼라장: 원소별 clamp.
        - 벡터: 노름 기반 방사형 스케일.
        """
        if delta.ndim == 0 or (delta.ndim >= 1 and
                               delta.shape[-1] not in (2, 3)):
            if not torch.is_tensor(limit):
                limit_t = torch.tensor(limit,
                                       device=delta.device,
                                       dtype=delta.dtype)
            else:
                limit_t = limit.to(device=delta.device, dtype=delta.dtype)
            return torch.clamp(delta, -limit_t, limit_t)

        vec_last_dim = delta.ndim - 1
        norm = torch.linalg.norm(delta, dim=vec_last_dim,
                                 keepdim=True).clamp_min(eps)
        if not torch.is_tensor(limit):
            limit_t = torch.tensor(limit,
                                   device=delta.device,
                                   dtype=delta.dtype)
        else:
            limit_t = limit.to(device=delta.device, dtype=delta.dtype)
        if limit_t.ndim == delta.ndim - 1:
            limit_t = limit_t.unsqueeze(-1)
        scale = torch.clamp(limit_t / norm, max=1.0)
        return delta * scale

    # ============================
    # [REFACTORED] 본체: Filter + Integrate
    # ============================
    def filter_and_integrate(
            self,
            unnorm_near_current_state: torch.Tensor,  # (B, Pnn, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T) bool
            unnorm_cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 3)
            near_class_one_hot: torch.Tensor,  # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._assert_prefix_valid_mask(near_cur_future_valid,
                                       context="filter_and_integrate")
        B, Pnn, T, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype = unnorm_cur_future_seg_body_control.dtype
        if T == 0:
            raise ValueError("T=0: 적분할 미래 세그먼트가 없습니다.")
            # 맨 앞에 추가
        if self.detach_state_and_u_for_ctrl_losses:  # 추가 필요
            unnorm_near_current_state = unnorm_near_current_state.detach(
            )  # 추가 필요
        """ key_to_limit_bp
        Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn) 
        
        v_max/a_max/alpha_max/a_lat_max/R_min
        /omega_abs_max/a_x_max/a_y_max, is_nonholonomic
        """
        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot, device=device, dtype=dtype)
        # (B,Pnn,T)
        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(
            unnorm_cur_future_seg_body_control)
        """ key_to_all_states
        x_next / y_next / cos_next / sin_next: (B,Pnn,T)
        vx_after / vy_after / omega_after: (B,Pnn,T)
        """
        key_to_all_states: Dict[str,
                                torch.Tensor] = self._init_integration_buffers(
                                    B, Pnn, T, dtype, device)

        x_k = unnorm_near_current_state[..., 0]  # (B,Pnn)
        y_k = unnorm_near_current_state[..., 1]  # (B,Pnn)
        cos_yaw_k = unnorm_near_current_state[..., 2]  # (B,Pnn)
        sin_yaw_k = unnorm_near_current_state[..., 3]  # (B,Pnn)
        vx_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)  # (B,Pnn)
        vy_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)  # (B,Pnn)
        omega_prev = torch.zeros((B, Pnn), device=device,
                                 dtype=dtype)  # (B,Pnn)

        for k in range(T):
            apply_S2_k = (k > 0)
            apply_S4_ax_k = (k > 0)

            # (B,Pnn)
            vx_k, vy_k, yaw_rate_k = vx_b_raw[...,
                                              k], vy_b_raw[...,
                                                           k], omega_raw[..., k]
            """ key_to_limit_bp
            Dict[str, torch.Tensor]: Tensor 은 전부 (B,Pnn) 

            v_max/a_max/alpha_max/a_lat_max/R_min
            /omega_abs_max/a_x_max/a_y_max, is_nonholonomic
            """
            # [STE] S0~S4
            vx_k, vy_k, yaw_rate_k = self._apply_constraints_step(
                vx_b_prev,  # (B,Pnn)
                vy_b_prev,  # (B,Pnn)
                omega_prev,  # (B,Pnn)
                vx_k,  # (B,Pnn)
                vy_k,  # (B,Pnn)
                yaw_rate_k,  # (B,Pnn)
                hp=self.constraints_h_params,  # _ConstraintHParams
                key_to_limit_bp=key_to_limit_bp,
                slip_epsilon=0.20,
                apply_S2=apply_S2_k,
                apply_S4_ax=apply_S4_ax_k,
            )

            # 중점 적분
            x_k1, y_k1, cos_k1, sin_k1 = self._integrate_midpoint_step(
                x_k, y_k, cos_yaw_k, sin_yaw_k, vx_k, vy_k, yaw_rate_k,
                self.constraints_h_params)

            key_to_all_states["x_next"][..., k] = x_k1
            key_to_all_states["y_next"][..., k] = y_k1
            key_to_all_states["cos_next"][..., k] = cos_k1
            key_to_all_states["sin_next"][..., k] = sin_k1
            key_to_all_states["vx_after"][..., k] = vx_k
            key_to_all_states["vy_after"][..., k] = vy_k
            key_to_all_states["omega_after"][..., k] = yaw_rate_k

            x_k, y_k, cos_yaw_k, sin_yaw_k = x_k1, y_k1, cos_k1, sin_k1
            vx_b_prev, vy_b_prev, omega_prev = vx_k, vy_k, yaw_rate_k

        return self._assemble_outputs(key_to_all_states, vx_b_raw, vy_b_raw,
                                      omega_raw, near_cur_future_valid)

    # ================================================================
    # [REFACTOR] Savitzky–Golay 유틸들 (모두 torch-only, 미분 가능)
    # ================================================================
    @staticmethod
    def _ensure_odd(value: int) -> int:
        """홀수 보정."""
        return value if (value % 2 == 1) else (value - 1)

    @staticmethod
    def _choose_window_length(
        effective_length: int,
        polyorder: int,
        max_window_length: int,
    ) -> int:
        """유효 길이에 맞춰 SG 윈도 길이 결정(홀수, polyorder보다 큼).
        - 너무 짧으면(≤polyorder 또는 <3) 0을 반환해 유한차분으로 폴백.
        """
        # 1) 길이가 너무 짧으면 SG 사용 안 함
        if effective_length <= polyorder or effective_length < 3:
            return 0

        # 2) 상한 내에서 가능한 가장 큰 홀수 창
        lim = min(max_window_length, effective_length)
        win = lim if (lim % 2 == 1) else (lim - 1)

        # 3) 여전히 polyorder보다 작거나 같으면 사용 불가 → 폴백
        if win <= polyorder:
            return 0

        return win

    @staticmethod
    def _build_savgol_diff_kernel(
        window_length: int,
        polyorder: int,
        deriv_order: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """SG 1D 미분 커널 (conv1d용) 생성. (중심에서의 도함수 근사)

        Returns:
            (1, 1, window_length) 모양의 커널.
        """
        with torch.no_grad():
            # h=1인 정수 격자에서의 설계 → 실제 시간 미분은 dt**deriv 로 스케일
            half = window_length // 2
            work_dtype = torch.float64
            x = torch.arange(-half, half + 1, device=device, dtype=work_dtype)
            V = torch.stack([x**i for i in range(polyorder + 1)],
                            dim=1).to(work_dtype)
            pinv = torch.linalg.pinv(V)  # float64
            e = torch.zeros(polyorder + 1, device=device, dtype=work_dtype)
            e[deriv_order] = math.factorial(deriv_order)
            coeff = (e @ pinv) / (dt**deriv_order)  # float64
            kernel = coeff.to(dtype).view(1, 1, window_length)  # 최종 dtype으로 캐스팅
            return kernel

    @staticmethod
    def _conv1d_with_pad(
        seq_bt: torch.Tensor,  # (N, 1, T)
        kernel: torch.Tensor,  # (1, 1, W)
        pad: int,
        mode: str = "reflect",
    ) -> torch.Tensor:
        """경계 패딩 후 1D 컨볼루션.

        Args:
            seq_bt: (N, 1, T) 입력 시퀀스
            kernel: (1, 1, W) SG 미분 커널
            pad: 좌/우 패딩 크기
            mode: "reflect" 권장(경계 편향 완화). T<2 등 불가 상황은 자동 폴백(replicate).

        Returns:
            (N, 1, T) 동일 길이 출력
        """
        if pad > 0:
            T = seq_bt.shape[-1]
            pad_mode = mode
            # reflect는 T>=2 필요. 불가하면 replicate로 폴백
            if mode == "reflect" and T < 2:
                pad_mode = "replicate"
            seq_bt = F.pad(seq_bt, (pad, pad), mode=pad_mode)
        return F.conv1d(seq_bt, kernel, stride=1)

    @staticmethod
    def _finite_difference_derivative(
        seq_bT: torch.Tensor,  # (N, T)
        dt: float,
    ) -> torch.Tensor:
        """윈도가 너무 짧은 경우를 위한 유한차분(중심차분/전방/후방 혼용). 미분 가능."""
        x = seq_bT
        N, T = x.shape
        dx = torch.zeros_like(x)
        if T >= 3:
            dx[:, 1:-1] = (x[:, 2:] - x[:, :-2]) / (2.0 * dt)
            dx[:, 0] = (x[:, 1] - x[:, 0]) / dt
            dx[:, -1] = (x[:, -1] - x[:, -2]) / dt
        elif T == 2:
            dx[:, 0] = (x[:, 1] - x[:, 0]) / dt
            dx[:, 1] = (x[:, 1] - x[:, 0]) / dt
        else:
            dx.zero_()
        return dx  # (N, T)

    @staticmethod
    def _unwrap_phase_torch(
            phase_bT: torch.Tensor,  # (B_Pnn, T1), 라디안
    ) -> torch.Tensor:
        """numpy.unwrap 유사 동작(토치 전용). 마지막 차원 기준."""
        # d = ((Δφ + π) mod 2π) - π
        pi = math.pi
        d = torch.diff(phase_bT, dim=-1)
        d_mod = (d + pi) % (2 * pi) - pi
        # 경계 케이스(+/-π) 보정은 생략(학습에서는 거의 영향 X)
        first = phase_bT[..., :1]
        unwrapped = torch.cat([first, first + torch.cumsum(d_mod, dim=-1)],
                              dim=-1)
        return unwrapped  # (B_Pnn, T1)

    @staticmethod
    def _split_full_vs_partial_rows(
            valid_bT: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """전 구간 유효 row / 일부만 유효 row 인덱스 분리.
            valid_bT: (B_Pnn,T1)  True=유효

        Returns:
            idx_full (K,), idx_partial (M,)
        """
        full_mask = valid_bT.all(dim=1)  # (B_Pnn,)
        idx_full = torch.nonzero(full_mask, as_tuple=False).flatten()
        idx_partial = torch.nonzero(~full_mask, as_tuple=False).flatten()
        return idx_full, idx_partial

    # [추가 요망] FeasibleProjector 내부에 추가
    def _build_poswise_weight_banks_cached(
        self,
        *,
        T: int,
        polyorder: int,
        max_window_length: int,
        dt: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """왼/가운데/오른쪽 구간에 쓸 위치별 SG 미분 가중치 뱅크 생성(+캐시된 항목 재사용).

        Returns:
            w_left_bank:  (half, W)    # m=0..half-1
            w_center:     (W,)         # m=half
            w_right_bank: (half, W)    # m=half..W-1
            half:         int
        """
        W = self._choose_window_length(T, polyorder, max_window_length)
        if W < 3:
            return (torch.empty(0, device=device, dtype=dtype),
                    torch.empty(0, device=device, dtype=dtype),
                    torch.empty(0, device=device, dtype=dtype), 0)
        half = W // 2

        # 중앙(항상 동일)
        w_center = self._get_savgol_pos_weights_cached(window_length=W,
                                                       polyorder=polyorder,
                                                       deriv_order=1,
                                                       dt=dt,
                                                       m=half,
                                                       device=device,
                                                       dtype=dtype)  # (W,)

        # 왼쪽/오른쪽 뱅크
        w_left_list, w_right_list = [], []
        for m in range(0, half):  # 왼쪽
            w_left_list.append(
                self._get_savgol_pos_weights_cached(window_length=W,
                                                    polyorder=polyorder,
                                                    deriv_order=1,
                                                    dt=dt,
                                                    m=m,
                                                    device=device,
                                                    dtype=dtype))
        for m in range(half + 1, W):  # 기존: range(half, W)
            w_right_list.append(
                self._get_savgol_pos_weights_cached(window_length=W,
                                                    polyorder=polyorder,
                                                    deriv_order=1,
                                                    dt=dt,
                                                    m=m,
                                                    device=device,
                                                    dtype=dtype))

        w_left_bank = torch.stack(
            w_left_list, dim=0) if w_left_list else torch.empty(
                0, W, device=device, dtype=dtype)  # (half, W)
        w_right_bank = torch.stack(
            w_right_list, dim=0) if w_right_list else torch.empty(
                0, W, device=device, dtype=dtype)  # (half, W)
        return w_left_bank, w_center, w_right_bank, half

    # [수정 요망] 기존 _sg_derivative_full_rows 를 벡터화로 교체
    def _sg_derivative_full_rows(
        self,
        seq_bT: torch.Tensor,  # (N_full, T)
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """전 타임스텝 유효 row를 위치별(one‑sided/중앙) SG로 '완전 벡터화'해 미분."""
        if seq_bT.numel() == 0:
            return seq_bT
        device, dtype = seq_bT.device, seq_bT.dtype
        N, T = seq_bT.shape

        w_left_bank, w_center, w_right_bank, half = self._build_poswise_weight_banks_cached(
            T=T,
            polyorder=polyorder,
            max_window_length=max_window_length,
            dt=dt,
            device=device,
            dtype=dtype)
        if half == 0:  # W<3 → 유한차분
            return self._finite_difference_derivative(seq_bT, dt)

        W = w_center.numel()
        dx = torch.empty_like(seq_bT)

        # 1) 왼쪽: 첫 윈도(공통) × 위치별 가중치(half개) → (N, half)
        X_left = seq_bT[:, :W]  # (N, W)
        left = X_left @ w_left_bank.transpose(0, 1)  # (N, half)

        # 2) 가운데: unfold 슬라이딩 창 × 중앙 가중치 → (N, T-2*half)
        X_mid = seq_bT.unfold(dimension=-1, size=W, step=1)  # (N, T-W+1, W)
        mid = (X_mid * w_center.view(1, 1, W)).sum(-1)  # (N, T-W+1)

        # 3) 오른쪽: 마지막 윈도(공통) × 위치별 가중치(half개) → (N, half)
        X_right = seq_bT[:, -W:]  # (N, W)
        right = X_right @ w_right_bank.transpose(0, 1)  # (N, half)

        # 4) 조립
        dx[:, :half] = left
        dx[:, half:T - half] = mid
        dx[:, T - half:T] = right
        return dx

    # [수정 요망] 기존 _sg_derivative_partial_rows 개선
    def _sg_derivative_partial_rows(
        self,
        seq_bT: torch.Tensor,  # (N_partial, T)
        valid_bT: torch.Tensor,  # (N_partial, T) True=유효
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """일부만 유효한 row들을 '유효길이 L'별로 묶어 배치 벡터화."""
        if seq_bT.numel() == 0:
            return seq_bT
        device, dtype = seq_bT.device, seq_bT.dtype
        dx = torch.zeros_like(seq_bT)

        # 내부 구멍 금지(0→1 전이 금지)
        v = valid_bT.to(torch.int8)
        d = v[:, 1:] - v[:, :-1]
        if (d > 0).any():
            raise ValueError("[_sg_derivative_partial_rows] 내부 구멍(0→1 전이) 발견.")

        eff_len = valid_bT.sum(dim=1)  # (N_partial,)
        unique_L = torch.unique(eff_len)
        for L in unique_L.tolist():
            if L <= 0:
                continue
            sel = (eff_len == L)  # (N_partial,)
            rows = seq_bT[sel][:, :L]  # (M, L)
            drows = self._sg_derivative_full_rows(  # (M, L)
                rows,
                dt=dt,
                polyorder=polyorder,
                max_window_length=max_window_length)
            dx[sel, :L] = drows
            # 나머지(무효)는 0 유지
        return dx

    def _savgol_derivative_masked_torch(
        self,
        seq_bT: torch.Tensor,  # (B_Pnn,T1)
        valid_bT: torch.Tensor,  # (B_Pnn,T1) True=유효
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:  # (B_Pnn,T1)
        """[리팩터링] 유효 구간에서만 SG로 1차 미분. 무효는 0."""
        dx = torch.zeros_like(seq_bT)

        # idx_full (K,), idx_partial (M,)
        idx_full, idx_partial = self._split_full_vs_partial_rows(valid_bT)
        # 1) full valid rows (배치 처리)
        if idx_full.numel() > 0:
            dx_full = self._sg_derivative_full_rows(
                seq_bT[idx_full],  # (N_full,T1)
                dt,
                polyorder,
                max_window_length)
            dx[idx_full] = dx_full
        # 2) partial valid rows (슬로우패스)
        if idx_partial.numel() > 0:
            dx_part = self._sg_derivative_partial_rows(seq_bT[idx_partial],
                                                       valid_bT[idx_partial],
                                                       dt, polyorder,
                                                       max_window_length)
            dx[idx_partial] = dx_part
        # 무효 구간은 기본 0 유지
        return dx  # (B_Pnn,T1)

    # feasible.py 내 FeasibleProjector 클래스 안에 추가
    def _assert_prefix_valid_mask(
            self,
            valid_bpt: torch.Tensor,
            context: str = "savgol_filter_for_control") -> None:
        """유효 마스크가 행마다 True*False* (단조 감소)인지 검증.

        Args:
            valid_bpt: (B, Pnn, T1) bool, 시간 축 마지막.
            context: 에러 메시지에 표시할 호출 위치 문자열.

        Raises:
            ValueError: 0→1 전이가 하나라도 발견되면(내부 구멍 또는 선행 무효 후 유효)
        """
        assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 여야 합니다."
        B, Pnn, T1 = valid_bpt.shape
        v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
        d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
        has_01 = (d > 0).any(dim=1)  # 0→1 전이 여부
        if has_01.any():
            bad_idx = torch.nonzero(has_01, as_tuple=False).flatten()
            # 가독성을 위해 일부만 표시
            max_show = min(int(bad_idx.numel()), 8)
            bad_idx_sample = bad_idx[:max_show].tolist()
            # (b,p) 인덱스 매핑
            b_list = [(i // Pnn) for i in bad_idx_sample]
            p_list = [(i % Pnn) for i in bad_idx_sample]
            raise ValueError(
                f"[{context}] near_cur_future_valid가 행 단위 단조 감소(True*False*) 가정에 위배됩니다. "
                f"0→1 전이가 감지되었습니다. 오류 row 수={int(bad_idx.numel())}, "
                f"예시 (b,p)={list(zip(b_list, p_list))}. "
                f"내부 구멍(1→0→1)이나 선행 무효 후 유효(0→1)는 허용되지 않습니다.")

    # ================================================================
    # 본 기능: 위치→세계속도, yaw→요레이트, 그리고 body 회전
    # ================================================================
    def savgol_filter_for_control(
            self,
            unnorm_diffusion_trajectory: torch.Tensor,
            # (B, Pnn, 1+T, 4) = [x, y, cos, sin]
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)     True/1=유효
            *,
            dt: float = 0.1,
            polyorder: int = 2,
            max_window_len_xy: int = 11,  # ≈ 1.1s @10Hz
            max_window_len_yaw: int = 7,  # ≈ 0.7s @10Hz
    ) -> torch.Tensor:  # (B, Pnn, 1+T, 3) = [vxw, vyw, w]
        """ Savitzky–Golay(마스크 인지)로 **세계좌표계** 속도/각속도 추정.

        Args:
            unnorm_diffusion_trajectory: (B, Pnn, 1+T, 4) = [x, y, cos, sin]
            near_cur_future_valid: (B, Pnn, 1+T)  True/1 = 유효
            dt: 샘플 간격(초). nuPlan/데이터셋 특성상 0.1 권장.
            polyorder: SG 다항 차수(2 권장).
            max_window_len_xy: x,y에 사용할 윈도 길이 상한(홀수로 강제됨).
            max_window_len_yaw: yaw에 사용할 윈도 길이 상한(홀수로 강제됨).

        Returns:
            (B, Pnn, 1+T, 3) = [v_x^w, v_y^w, w]  (단위: m/s, m/s, rad/s)
        """
        B, Pnn, T1, _ = unnorm_diffusion_trajectory.shape

        # 1) 입력 분리
        x = unnorm_diffusion_trajectory[..., 0]  # (B,Pnn,T1)
        y = unnorm_diffusion_trajectory[..., 1]  # (B,Pnn,T1)
        cos_y = unnorm_diffusion_trajectory[..., 2]  # (B,Pnn,T1)
        sin_y = unnorm_diffusion_trajectory[..., 3]  # (B,Pnn,T1)
        valid = (near_cur_future_valid > 0).to(torch.bool)  # (B,Pnn,T1)

        # NEW: 입력 가정 강제 (내부 구멍 금지)
        self._assert_prefix_valid_mask(valid,
                                       context="savgol_filter_for_control")

        # 3) SG-미분: x, y, yaw 각각 (마스크 인지)
        v_x = self._savgol_derivative_masked_torch(
            x.reshape(-1, T1),  # (B_Pnn,T1)
            valid.reshape(-1, T1),  # (B_Pnn,T1)
            dt,
            polyorder,
            max_window_len_xy)  # v_x^w
        v_x = v_x.reshape(B, Pnn, T1)  # (B_Pnn,T1) -> (B,Pnn,T1)
        v_y = self._savgol_derivative_masked_torch(
            y.reshape(-1, T1),  # (B_Pnn,T1)
            valid.reshape(-1, T1),  # (B_Pnn,T1)
            dt,
            polyorder,
            max_window_len_xy)  # v_y^w
        v_y = v_y.reshape(B, Pnn, T1)

        # --- 변경: cos/sin 직접 미분 → ψ̇ ---
        yaw_rate = _yaw_rate_from_cos_sin_via_sg(
            cos_yaw=cos_y,  # (B,Pnn,T1)
            sin_yaw=sin_y,  # (B,Pnn,T1)
            valid_mask=valid,  # (B,Pnn,T1) bool
            dt=dt,
            polyorder=polyorder,
            max_window_len=max_window_len_yaw,
            eps=1e-6,
            sg_derivative_fn=self._savgol_derivative_masked_torch,
            # 당신이 이미 가진 함수
        )  # (B,Pnn,T1)

        # 5) 안전 마스킹(무효 시점은 0)
        v_x = torch.where(valid, v_x, torch.zeros_like(v_x))
        v_y = torch.where(valid, v_y, torch.zeros_like(v_y))
        yaw_rate = torch.where(valid, yaw_rate, torch.zeros_like(yaw_rate))
        unnorm_cur_future_control = torch.stack([v_x, v_y, yaw_rate],
                                                dim=-1)  # (B,Pnn,T1,3)
        return unnorm_cur_future_control


# ---------------------------------------------------------------------------
