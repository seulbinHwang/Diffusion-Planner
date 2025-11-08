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


# =========================
# [NEW] 하이퍼/제약 파라미터 컨테이너
# =========================
@dataclass
class _ConstraintHParams:
    """제약/적분 하이퍼파라미터(상수 모음)."""
    dt: float
    eps: float
    # 추가 필요: STE용 밴드폭 η (권장 초기값)
    eta_slip: float = 0.07   # S0
    eta_speed: float = 0.05  # S1
    eta_inc: float = 0.10    # S2
    eta_yaw: float = 0.05    # S3
    eta_fric: float = 0.05   # S4

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
            float: 허용 가능한 최대 |ω| [rad/s]
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
        # 베이스 제어 어댑터(U_base: (vxb,vyb,ω) — 정규화 값)
        self.control_adapter = nn.Sequential(
            nn.LayerNorm(3),
            nn.Linear(3, self._Du),
            nn.GELU(),
        )

        # 트렁크(디퓨전 은닉) 압축기는 H를 알아야 하므로 지연 초기화
        # (B,Pnn,H)->(B,Pnn,Dc)
        self.trunk_compressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, self._Dc),
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
            [nn.Linear(self._C, self._C) for _ in range(4)])

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

    # =========================================================
    # [추가 필요] STE 유틸 (공통)
    # =========================================================
    @staticmethod
    def _smoothstep_quintic01(t: torch.Tensor) -> torch.Tensor:
        """[0,1]→[0,1], C² smoothstep: 10t^3 - 15t^4 + 6t^5."""
        return t ** 3 * (10 - 15 * t + 6 * t * t)

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
    def _ste_scalar_clip(x: torch.Tensor,
                         limit: torch.Tensor,
                         eta: float,
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
                         v_max: torch.Tensor,
                         eta: float, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """라디얼(벡터) STE-clip."""
        v_max = v_max.to(dtype=vx.dtype, device=vx.device)
        speed = torch.sqrt(vx*vx + vy*vy + eps)
        s_hard = torch.clamp(v_max / speed.clamp_min(eps), max=1.0)
        vx_h, vy_h = s_hard * vx, s_hard * vy
        r = speed / (v_max + eps)
        w = FeasibleProjector._ste_band_weight(r, eta)
        vx_sur, vy_sur = w * vx + (1-w) * vx.detach(), w * vy + (1-w) * vy.detach()
        vx_out = vx_sur + (vx_h - vx_sur).detach()  # 추가 필요
        vy_out = vy_sur + (vy_h - vy_sur).detach()  # 추가 필요
        return vx_out, vy_out

    @staticmethod
    def _ste_increment_vec(dv: torch.Tensor,
                           limit: torch.Tensor,
                           eta: float, eps: float) -> torch.Tensor:
        """증분(벡터노름) STE-clip."""
        limit = limit.to(dtype=dv.dtype, device=dv.device).unsqueeze(-1)
        norm = torch.linalg.norm(dv, dim=-1, keepdim=True).clamp_min(eps)
        s_hard = torch.clamp(limit / norm, max=1.0)
        dv_h = s_hard * dv
        r = (norm.squeeze(-1)) / (limit.squeeze(-1) + eps)
        w = FeasibleProjector._ste_band_weight(r, eta).unsqueeze(-1)
        dv_sur = w * dv + (1-w) * dv.detach()
        return dv_h + (dv_sur - dv_h).detach()

    @staticmethod
    def _ste_increment_scalar(dx: torch.Tensor,
                              limit: torch.Tensor,
                              eta: float, eps: float) -> torch.Tensor:
        """증분(스칼라) STE-clip."""
        return FeasibleProjector._ste_scalar_clip(dx, limit, eta, eps)

    @staticmethod
    def _ste_yawrate_clip(omega_raw: torch.Tensor,
                          allow: torch.Tensor,
                          eta: float, eps: float) -> torch.Tensor:
        """S3용: forward hard(±allow), backward는 r=|raw|/allow(detached)로 밴드 가중."""
        allow = allow.to(dtype=omega_raw.dtype, device=omega_raw.device)
        y_hard = omega_raw.clamp(-allow, allow)
        r = omega_raw.abs() / (allow.detach() + eps)  # 추가 필요: allow detach 반영
        w = FeasibleProjector._ste_band_weight(r, eta)
        y_sur = w * omega_raw + (1-w) * omega_raw.detach()
        y = y_sur + (y_hard - y_sur).detach()  # 추가 필요

        return y

    @staticmethod
    def _ste_friction_scale(ax: torch.Tensor, ay: torch.Tensor,
                            ax_max: torch.Tensor, ay_max: torch.Tensor,
                            eta: float, eps: float) -> torch.Tensor:
        """S4용 타원 스케일: forward s_hard=min(1,1/r), backward s≈w(r)."""
        ax_max = ax_max.to(dtype=ax.dtype, device=ax.device)
        ay_max = ay_max.to(dtype=ay.dtype, device=ay.device)
        r = torch.sqrt((ax/(ax_max+eps))**2 + (ay/(ay_max+eps))**2 + eps)
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
        # (B, Pnn, 1+T, 3) [v_x^w,v_y^w, ω]
        near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T)    (bool/0-1)
    ) -> torch.Tensor:  # (B, Pnn, T, 3)   [v_x^b, v_y^b, ω]_mid
        """구간 [t_k, t_{k+1})의 **중점(midpoint) 제어**를 계산한다.

        논리/수식
        --------
        - 입력 속도 [v_x, v_y]는 **세계(ego) 프레임** 기준이라고 가정.
        - 서로 다른 시점의 body 프레임이 회전하므로,
            중간 속도는 **세계 프레임에서 두 끝점 속도를 가중 평균**하고,
            그 후 **중간 yaw(두 끝점의 cos/sin 가중 평균 → 정규화)** 으로 **몸체 프레임으로 회전**한다.
        - 요각속도 ω는 세계 프레임 스칼라이므로 **끝점 가중 평균**으로 충분.
        - 유효성 마스크는 구간 양 끝점에 대해 가중치로 활용(둘 다 무효면 0).

        Args:
            unnorm_diffusion_trajectory: (B,Pnn,1+T,4) = [x, y, cos, sin] (세계/ego 기준)
            unnorm_cur_future_control:   (B,Pnn,1+T,3) = [v_x^w, v_y^w, ω] (세계 기준)
            near_cur_future_valid:       (B,Pnn,1+T)   = True/False

        Returns:
            torch.Tensor: (B,Pnn,T,3) = [v_x^b_mid, v_y^b_mid, ω_mid]
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
        sin_mid = self._weighted_avg_two(
            sin_end,  # (B,Pnn,T)
            sin_start,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            eps=eps)  # (B,Pnn,T)
        norm = (cos_mid * cos_mid + sin_mid * sin_mid).clamp_min(eps).sqrt()
        cos_mid = cos_mid / norm
        sin_mid = sin_mid / norm

        # ---- 세계→바디 회전
        vxb_mid, vyb_mid = self._world_to_body(v_x_mid_w, v_y_mid_w, cos_mid,
                                               sin_mid)  # (B,Pnn,T)

        # ---- ω 중점(가중 평균, 스칼라)
        omega_mid = self._weighted_avg_two(
            omega_start,  # (B,Pnn,T)
            omega_end,  # (B,Pnn,T)
            start_valid,  # (B,Pnn,T)
            end_valid,  # (B,Pnn,T)
            eps=eps)  # (B,Pnn,T)

        # ---- 구간 유효 마스킹: 양 끝 모두 무효면 0
        seg_is_valid = ((start_valid > 0) | (end_valid > 0)).to(
            vxb_mid.dtype)  # (B,Pnn,T)
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
            x_prev=x_prev,
            x_fut=x_fut,
            u_base=cur_future_seg_body_control,
            dit_final_hidden_tokens=dit_final_hidden_tokens,
        )

        # 4) Stem → 5) TCN
        # Z_s: (B, Pnn, T, 192)
        Z_s = self._apply_stem(Z_in,
                               seg_mask_1)  # (B, Pnn, T, 192) # (B, Pnn, T, 1)
        # Z_tcn: (B, Pnn, T, 192)
        Z_tcn = self._run_tcn(
            Z_s, seg_mask,
            seg_mask_1)  # (B, Pnn, T, 192) # (B, Pnn, T) # (B, Pnn, T, 1)

        # 6) ΔU 예측 → 7) 결합
        # delta_u: (B, Pnn, T, 3)
        delta_u = self._predict_delta_u(Z_tcn, Z_s, seg_mask_1)
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
    def _build_segment_mask(
            self,
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T) bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """구간 마스크(세그먼트 유효성) 생성.

        규칙:
            - 구간 k는 양 끝 점 프레임 k, k+1 중 하나라도 유효하면 유효.
            - seg_mask: (B, Pnn, T) float(0/1)
            - seg_mask_1: (B, Pnn, T, 1)  → 브로드캐스트 편의를 위한 채널 차원 추가

        Args:
            near_cur_future_valid (torch.Tensor): (B, Pnn, 1+T) bool

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - seg_mask (torch.Tensor): (B, Pnn, T) float(0/1)
                - seg_mask_1 (torch.Tensor): (B, Pnn, T, 1) float(0/1)
        """
        valid = near_cur_future_valid.to(torch.bool)  # (B,Pnn,1+T)
        seg_valid = (valid[..., :-1] | valid[..., 1:])  # (B,Pnn,T)
        seg_mask = seg_valid.to(torch.float32)
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

    # =========================================================
    # [B] 피처 인코딩 (prev/fut/u_base/trunk)
    # =========================================================
    def _features_from_inputs(
            self,
            x_prev: torch.Tensor,  # (B, Pnn, T, 4)
            x_fut: torch.Tensor,  # (B, Pnn, T, 4)
            u_base: torch.Tensor,  # (B, Pnn, T, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
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
        if self.detach_state_and_u_for_ctrl_losses:  # 추가 필요
            x_prev = x_prev.detach()  # 추가 필요
            x_fut  = x_fut.detach()   # 추가 필요
            u_base_in = u_base.detach()  # 추가 필요
        else:
            u_base_in = u_base

        feat_prev = self.state_prev_encoder(x_prev)  # (B,Pnn,T,48)
        feat_fut = self.state_fut_encoder(x_fut)  # (B,Pnn,T,48)
        feat_u = self.control_adapter(u_base_in) # (B,Pnn,T,32)

        trunk = self.trunk_compressor(
            dit_final_hidden_tokens.detach())  # (B,Pnn,64)
        trunk_rep = trunk.unsqueeze(2).expand(-1, -1, x_prev.size(2),
                                              -1)  # (B,Pnn,T,64)

        Z_in = torch.cat([feat_prev, feat_fut, feat_u, trunk_rep],
                         dim=-1)  # (B,Pnn,T,192)
        return Z_in

    # =========================================================
    # [C] Stem / TCN 백본
    # =========================================================
    def _apply_stem(
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
            Z = (Z + Y) * seg_mask_1  # Residual + mask
        return Z  # (B,Pnn,T,192)

    # =========================================================
    # [D] Head & Gate / 결합
    # =========================================================
    def _predict_delta_u(
        self,
        Z_tcn: torch.Tensor,  # (B, Pnn, T, 192)
        Z_s: torch.Tensor,  # (B, Pnn, T, 192)  (게이트 입력)
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
        gate_scale = F.softplus(self.gate_mlp(Z_s))  # (B,Pnn,T,3)
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
            near_class_one_hot: (B, Pnn, 3)  one‑hot (0: vehicle, 1: pedestrian, 2: bicycle)

        Returns:
            Dict[str, Tensor]: v_max/a_max/alpha_max/a_lat_max/R_min/omega_abs_max/a_x_max/a_y_max, is_nonholonomic
                * 모두 (B,Pnn) 모양
        """
        # 순서 주의: [vehicle(CAR), pedestrian, bicycle]
        car = self.constraints[ActorClass.CAR]
        ped = self.constraints[ActorClass.PEDESTRIAN]
        bic = self.constraints[ActorClass.BICYCLE]

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

        return dict(v_max=v_max_bp,
                    a_max=a_max_bp,
                    alpha_max=alpha_max_bp,
                    a_lat_max=a_lat_max_bp,
                    R_min=R_min_bp,
                    omega_abs_max=omega_abs_bp,
                    a_x_max=a_x_max_bp,
                    a_y_max=a_y_max_bp,
                    is_nonholonomic=is_nonholonomic)

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
    # [MOD] (S3) 옆가속/최소R/절대|ω| (중점 속도 사용)
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
            unnorm_cur_future_seg_body_control (torch.Tensor): (B, Pnn, T, 3) [vxb, vyb, ω]

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
        vx_b: torch.Tensor, vy_b: torch.Tensor,
        slip_epsilon: float, eta: float, eps: float,
        is_nonholonomic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S0: v_y hard-clip(+STE). 보행자 제외."""
        limit = torch.full_like(vy_b, float(slip_epsilon))
        vy_new = self._ste_scalar_clip(vy_b, limit, eta, eps)
        vy_out = torch.where(is_nonholonomic, vy_new, vy_b)
        return vx_b, vy_out

    def _apply_S1_speed_limit_ste(
        self, vx_b: torch.Tensor, vy_b: torch.Tensor,
        v_max: torch.Tensor, eta: float, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._ste_radial_clip(vx_b, vy_b, v_max, eta, eps)

    def _apply_S2_accel_alpha_limits_ste(
        self,
        vx_b_prev: torch.Tensor, vy_b_prev: torch.Tensor, omega_prev: torch.Tensor,
        vx_b: torch.Tensor, vy_b: torch.Tensor, omega: torch.Tensor,
        a_max: torch.Tensor, alpha_max: torch.Tensor, dt: float,
        eta: float, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 선형 속도 증분(벡터)
        dv = torch.stack([vx_b - vx_b_prev, vy_b - vy_b_prev], dim=-1)
        dv_limit = (a_max * dt).to(dtype=dv.dtype, device=dv.device)
        dv_ste = self._ste_increment_vec(dv, dv_limit, eta, eps)
        vx_b_new = vx_b_prev + dv_ste[..., 0]
        vy_b_new = vy_b_prev + dv_ste[..., 1]
        # 각속도 증분(스칼라)
        dω = omega - omega_prev
        dω_limit = (alpha_max * dt).to(dtype=dω.dtype, device=dω.device)
        dω_ste = self._ste_increment_scalar(dω, dω_limit, eta, eps)
        ω_new = omega_prev + dω_ste
        return vx_b_new, vy_b_new, ω_new

    # [추가 요망] (S3: 속도-연동 각속도 한계 — ω clip, no slip angle)
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
        """(S3) β/Δβ 없이, 속도-연동 ω 한계로 직접 clip.

        ω_allow (차/자전거) = min( a_lat_max/|v|, |v|/R_min, ω_abs_max )
        ω_allow (보행자)     = min( a_lat_max/|v|, ω_abs_max )   # 제자리 회전 허용
        """
        speed = torch.sqrt(vx_b * vx_b + vy_b * vy_b + eps)  # (B,Pnn)
        allow_lat = a_lat_max / (speed + eps)
        allow_R = speed / (R_min + eps)
        allow_abs = omega_abs_max

        # 비홀로노믹이면 R_min 항 포함, 보행자는 제외
        allow_nonh = torch.minimum(torch.minimum(allow_lat, allow_R), allow_abs)
        allow_holo = torch.minimum(allow_lat, allow_abs)
        allow = torch.where(is_nonholonomic, allow_nonh, allow_holo)

        # forward: hard clamp, backward: band-weighted surrogate (allow는 detach)
        omega_new = self._ste_yawrate_clip(omega, allow, eta=eta, eps=eps)
        return omega_new

    # [추가 요망] (S4: β 미사용, Δv와 ω를 동일 스케일 s로 동시 축소)
    def _apply_S4_friction_circle_ste(
            self,
            vx_b_prev: torch.Tensor, vy_b_prev: torch.Tensor,  # (B,Pnn)
            vx_b_k: torch.Tensor, vy_b_k: torch.Tensor,  # (B,Pnn)
            omega_k: torch.Tensor,  # (B,Pnn)
            a_x_max: torch.Tensor, a_y_max: torch.Tensor,  # (B,Pnn)
            dt: float, eta: float, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(S4) (a_x/a_x,max)^2+(a_y/a_y,max)^2≤1 를 만족하도록
        Δv와 ω를 동일 스케일 s∈(0,1]로 동시 축소.
           a_x ≈ Δv/dt,  a_y ≈ v_mid * ω_k
        """
        # 스칼라 속도 크기
        speed_prev = torch.sqrt(
            vx_b_prev * vx_b_prev + vy_b_prev * vy_b_prev + eps)
        speed_k = torch.sqrt(vx_b_k * vx_b_k + vy_b_k * vy_b_k + eps)

        # v_mid, Δv, a_x/a_y 추정
        v_mid = 0.5 * (speed_prev + speed_k)
        delta_speed = speed_k - speed_prev
        ax_est = delta_speed / dt
        ay_est = v_mid * omega_k

        # 타원 규격화 노름 기반 스케일 (forward: hard, backward: band-weight)
        s = self._ste_friction_scale(ax_est, ay_est, a_x_max, a_y_max, eta=eta,
                                     eps=eps)

        # Δv, ω를 동시에 축소
        delta_speed_scaled = s * delta_speed
        omega_new = s * omega_k

        # 목표 속도 크기 및 방향 유지
        target_speed = (speed_prev + delta_speed_scaled).clamp_min(0.0)
        speed_thr = 1e-3
        dir_x = torch.where(
            speed_k > speed_thr, vx_b_k / speed_k,
            torch.where(speed_prev > speed_thr, vx_b_prev / speed_prev,
                        torch.ones_like(vx_b_k))
        )
        dir_y = torch.where(
            speed_k > speed_thr, vy_b_k / speed_k,
            torch.where(speed_prev > speed_thr, vy_b_prev / speed_prev,
                        torch.zeros_like(vy_b_k))
        )
        vx_b_new = dir_x * target_speed
        vy_b_new = dir_y * target_speed
        return vx_b_new, vy_b_new, omega_new

    # ----------------------------
    # [NEW] 한 스텝: 제약 S0~S4 적용(모두 STE 버전 호출)
    # ----------------------------
    def _apply_constraints_step(
        self,
        vx_b_prev: torch.Tensor, vy_b_prev: torch.Tensor, omega_prev: torch.Tensor,
        vx_b_k: torch.Tensor,   vy_b_k: torch.Tensor,   omega_k: torch.Tensor,
        hp: _ConstraintHParams,
        key_to_limit_bp: Dict[str, torch.Tensor],
        slip_epsilon: float = 0.20,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # (S0)
        vx_b_k, vy_b_k = self._apply_S0_nonholonomic_ste(
            vx_b_k, vy_b_k, slip_epsilon, hp.eta_slip, hp.eps,
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"]
        )
        # (S1)
        vx_b_k, vy_b_k = self._apply_S1_speed_limit_ste(
            vx_b_k, vy_b_k, key_to_limit_bp["v_max"], hp.eta_speed, hp.eps
        )
        # (S2)
        vx_b_k, vy_b_k, omega_k = self._apply_S2_accel_alpha_limits_ste(
            vx_b_prev, vy_b_prev, omega_prev,
            vx_b_k, vy_b_k, omega_k,
            key_to_limit_bp["a_max"], key_to_limit_bp["alpha_max"],
            hp.dt, hp.eta_inc, hp.eps
        )
        # (S3)
        omega_k = self._apply_S3_omega_clip_ste(
            vx_b=vx_b_k, vy_b=vy_b_k, omega=omega_k,
            a_lat_max=key_to_limit_bp["a_lat_max"],
            R_min=key_to_limit_bp["R_min"],
            omega_abs_max=key_to_limit_bp["omega_abs_max"],
            is_nonholonomic=key_to_limit_bp["is_nonholonomic"],
            eta=hp.eta_yaw, eps=hp.eps
        )
        # (S4)
        vx_b_k, vy_b_k, omega_k = self._apply_S4_friction_circle_ste(
            vx_b_prev, vy_b_prev, vx_b_k, vy_b_k, omega_k,
            key_to_limit_bp["a_x_max"], key_to_limit_bp["a_y_max"],
            hp.dt, hp.eta_fric, hp.eps
        )
        return vx_b_k, vy_b_k, omega_k

    # ----------------------------
    # [NEW] 한 스텝: 중점 적분 (변경 없음)
    # ----------------------------
    def _integrate_midpoint_step(
        self,
        x_k: torch.Tensor, y_k: torch.Tensor,
        cos_yaw_k: torch.Tensor, sin_yaw_k: torch.Tensor,
        vx_b_k: torch.Tensor, vy_b_k: torch.Tensor, omega_k: torch.Tensor,
        hp: _ConstraintHParams,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        half_dtheta = 0.5 * omega_k * hp.dt
        cos_mid, sin_mid = self._compute_mid_heading_from_cos_sin(cos_yaw_k, sin_yaw_k, half_dtheta)
        vwx_mid = cos_mid * vx_b_k - sin_mid * vy_b_k
        vwy_mid = sin_mid * vx_b_k + cos_mid * vy_b_k
        x_k1 = x_k + vwx_mid * hp.dt
        y_k1 = y_k + vwy_mid * hp.dt
        dtheta = omega_k * hp.dt
        cos_yaw_k1, sin_yaw_k1 = self._advance_heading_cos_sin(cos_yaw_k, sin_yaw_k, dtheta, eps=hp.eps)
        return x_k1, y_k1, cos_yaw_k1, sin_yaw_k1

    # ----------------------------
    # [NEW] 결과 조립(+마스크/차이)
    # ----------------------------
    def _assemble_outputs(
        self,
        key_to_all_states: Dict[str, torch.Tensor],
        vx_b_raw: torch.Tensor, vy_b_raw: torch.Tensor, omega_raw: torch.Tensor,
        near_current_valid: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_next_all, y_next_all   = key_to_all_states["x_next"],   key_to_all_states["y_next"]
        cos_next_all, sin_next_all = key_to_all_states["cos_next"], key_to_all_states["sin_next"]
        unnorm_integrated_trajectory = torch.stack(
            [x_next_all, y_next_all, cos_next_all, sin_next_all], dim=-1
        )
        unnorm_control_constraint_diff = torch.stack(
            [
                # [추가 요망] after(=Filter_soft(u))는 detach해서 grad가 필터로 역류하지 않도록
                key_to_all_states["vx_after"].detach() - vx_b_raw,
                key_to_all_states["vy_after"].detach() - vy_b_raw,
                key_to_all_states["omega_after"].detach() - omega_raw,
            ],
            dim=-1
        )
        valid_mask = near_current_valid.to(unnorm_integrated_trajectory.dtype).unsqueeze(-1).unsqueeze(-1)
        unnorm_integrated_trajectory = unnorm_integrated_trajectory * valid_mask
        unnorm_control_constraint_diff = unnorm_control_constraint_diff * valid_mask
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
        speed = torch.sqrt(vx * vx + vy * vy + eps)
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
        if delta.ndim == 0 or (
                delta.ndim >= 1 and delta.shape[-1] not in (2, 3)):
            if not torch.is_tensor(limit):
                limit_t = torch.tensor(limit, device=delta.device,
                                       dtype=delta.dtype)
            else:
                limit_t = limit.to(device=delta.device, dtype=delta.dtype)
            return torch.clamp(delta, -limit_t, limit_t)

        vec_last_dim = delta.ndim - 1
        norm = torch.linalg.norm(delta, dim=vec_last_dim,
                                 keepdim=True).clamp_min(eps)
        if not torch.is_tensor(limit):
            limit_t = torch.tensor(limit, device=delta.device,
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
        near_current_valid: torch.Tensor,         # (B, Pnn) bool
        unnorm_cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 3)
        near_class_one_hot: torch.Tensor,         # (B, Pnn, 3)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, Pnn, T, _ = unnorm_cur_future_seg_body_control.shape
        device = unnorm_cur_future_seg_body_control.device
        dtype  = unnorm_cur_future_seg_body_control.dtype
        if T == 0:
            raise ValueError("T=0: 적분할 미래 세그먼트가 없습니다.")
            # 맨 앞에 추가
        if self.detach_state_and_u_for_ctrl_losses:  # 추가 필요
            unnorm_near_current_state = unnorm_near_current_state.detach()  # 추가 필요
        key_to_limit_bp: Dict[str, torch.Tensor] = self._build_per_agent_limits(
            near_class_one_hot, device=device, dtype=dtype
        )
        vx_b_raw, vy_b_raw, omega_raw = self._split_controls(unnorm_cur_future_seg_body_control)
        key_to_all_states: Dict[str, torch.Tensor] = self._init_integration_buffers(B, Pnn, T, dtype, device)

        x_k = unnorm_near_current_state[..., 0]
        y_k = unnorm_near_current_state[..., 1]
        cos_yaw_k = unnorm_near_current_state[..., 2]
        sin_yaw_k = unnorm_near_current_state[..., 3]
        vx_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)
        vy_b_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)
        omega_prev = torch.zeros((B, Pnn), device=device, dtype=dtype)

        for k in range(T):
            vx_k, vy_k, yaw_rate_k = vx_b_raw[..., k], vy_b_raw[..., k], omega_raw[..., k]

            # [STE] S0~S4
            vx_k, vy_k, yaw_rate_k = self._apply_constraints_step(
                vx_b_prev, vy_b_prev, omega_prev,
                vx_k, vy_k, yaw_rate_k,
                hp=self.constraints_h_params,
                key_to_limit_bp=key_to_limit_bp,
                slip_epsilon=0.20
            )

            # 중점 적분
            x_k1, y_k1, cos_k1, sin_k1 = self._integrate_midpoint_step(
                x_k, y_k, cos_yaw_k, sin_yaw_k, vx_k, vy_k, yaw_rate_k, self.constraints_h_params
            )

            key_to_all_states["x_next"][..., k] = x_k1
            key_to_all_states["y_next"][..., k] = y_k1
            key_to_all_states["cos_next"][..., k] = cos_k1
            key_to_all_states["sin_next"][..., k] = sin_k1
            key_to_all_states["vx_after"][...,  k] = vx_k
            key_to_all_states["vy_after"][...,  k] = vy_k
            key_to_all_states["omega_after"][..., k] = yaw_rate_k

            x_k, y_k, cos_yaw_k, sin_yaw_k = x_k1, y_k1, cos_k1, sin_k1
            vx_b_prev, vy_b_prev, omega_prev = vx_k, vy_k, yaw_rate_k

        return self._assemble_outputs(key_to_all_states, vx_b_raw, vy_b_raw, omega_raw, near_current_valid)


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

        Args:
            effective_length: 현재 구간 길이 M (전체 길이 혹은 [i0, i1]).
            polyorder: 다항 차수(보통 2).
            max_window_length: 윈도 길이 상한.
        Returns:
            int: 사용 가능한 홀수 윈도 길이. (3 미만이면 SG 대신 유한차분 권장)
        """
        if effective_length <= 0:
            return 0
        win = min(max_window_length, effective_length)
        win = FeasibleProjector._ensure_odd(win)
        while win > effective_length or win <= polyorder:
            win -= 2
        return max(0, win)

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
        # h=1인 정수 격자에서의 설계 → 실제 시간 미분은 dt**deriv 로 스케일
        half = window_length // 2
        x = torch.arange(-half, half + 1, device=device, dtype=dtype)  # (W,)
        # Vandermonde: [1, x, x^2, ...]
        V = torch.stack([x**i for i in range(polyorder + 1)],
                        dim=1)  # (W, Pnn+1)
        # Moore–Penrose 역행렬
        pinv = torch.linalg.pinv(V)  # (Pnn+1, W)
        e = torch.zeros(polyorder + 1, device=device, dtype=dtype)
        e[deriv_order] = math.factorial(deriv_order)  # d^k/dx^k at 0
        # 계수: e^T * pinv  → (W,)
        coeff = (e @ pinv) / (dt**deriv_order)
        kernel = coeff.view(1, 1, window_length)  # (1,1,W)
        return kernel

    @staticmethod
    def _conv1d_with_replicate_pad(
        seq_bt: torch.Tensor,  # (N, 1, T)
        kernel: torch.Tensor,  # (1, 1, W)
        pad: int,
    ) -> torch.Tensor:
        """경계는 replicate 로 패딩, stride=1 conv."""
        if pad > 0:
            seq_bt = F.pad(seq_bt, (pad, pad), mode="replicate")
        out = F.conv1d(seq_bt, kernel, stride=1)
        return out  # (N, 1, T)

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

    def _sg_derivative_full_rows(
        self,
        seq_bT: torch.Tensor,  # (N_full,T1)
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """전 타임스텝 유효한 배치를 한 번에 SG-미분(conv1d) 처리."""
        if seq_bT.numel() == 0:
            return seq_bT
        device, dtype = seq_bT.device, seq_bT.dtype
        N, T1 = seq_bT.shape
        W = self._choose_window_length(T1, polyorder, max_window_length)
        if W < 3:
            return self._finite_difference_derivative(seq_bT, dt)  # (N, T1)
        kernel = self._build_savgol_diff_kernel(W, polyorder, 1, dt, device,
                                                dtype)  # (1,1,W)
        pad = W // 2
        out = self._conv1d_with_replicate_pad(seq_bT.unsqueeze(1), kernel,
                                              pad).squeeze(1)
        return out  # (N, T1)

    def _sg_derivative_partial_rows(
        self,
        seq_bT: torch.Tensor,  # (N_partial, T)
        valid_bT: torch.Tensor,  # (N_partial, T)  True=유효
        dt: float,
        polyorder: int,
        max_window_length: int,
    ) -> torch.Tensor:
        """일부 구간만 유효한 row들을 슬로우패스로 [i0,i1] 구간만 SG-미분."""
        if seq_bT.numel() == 0:
            return seq_bT
        device, dtype = seq_bT.device, seq_bT.dtype
        Np, T = seq_bT.shape
        dx = torch.zeros_like(seq_bT)
        for i in range(Np):
            v = valid_bT[i]  # (T,)
            if not bool(v.any()):
                continue
            idx = torch.nonzero(v, as_tuple=False).flatten()
            i0, i1 = int(idx[0]), int(idx[-1])
            seg = seq_bT[i, i0:i1 + 1].unsqueeze(0)  # (1, L)
            L = seg.shape[-1]
            W = self._choose_window_length(L, polyorder, max_window_length)
            if W < 3:
                dseg = self._finite_difference_derivative(seg, dt)  # (1, L)
            else:
                kernel = self._build_savgol_diff_kernel(W, polyorder, 1, dt,
                                                        device, dtype)
                pad = W // 2
                dseg = self._conv1d_with_replicate_pad(seg.unsqueeze(1), kernel,
                                                       pad).squeeze(1)  # (1,L)
            dx[i, i0:i1 + 1] = dseg[0]
        return dx  # (N_partial, T)

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
    ) -> torch.Tensor:  # (B, Pnn, 1+T, 3) = [vxb, vyb, ω]
        """Savitzky–Golay(마스크 인지)로 속도/각속도 추정 후 body로 회전.

        Args:
            unnorm_diffusion_trajectory: (B, Pnn, 1+T, 4) = [x, y, cos, sin]
            near_cur_future_valid: (B, Pnn, 1+T)  True/1 = 유효
            dt: 샘플 간격(초). nuPlan/데이터셋 특성상 0.1 권장.
            polyorder: SG 다항 차수(2 권장).
            max_window_len_xy: x,y에 사용할 윈도 길이 상한(홀수로 강제됨).
            max_window_len_yaw: yaw에 사용할 윈도 길이 상한(홀수로 강제됨).

        Returns:
            (B, Pnn, 1+T, 3) = [v_x^w, v_y^w, ω]  (단위: m/s, m/s, rad/s)
        """
        B, Pnn, T1, _ = unnorm_diffusion_trajectory.shape

        # 1) 입력 분리
        x = unnorm_diffusion_trajectory[..., 0]  # (B,Pnn,T1)
        y = unnorm_diffusion_trajectory[..., 1]  # (B,Pnn,T1)
        cos_y = unnorm_diffusion_trajectory[..., 2]  # (B,Pnn,T1)
        sin_y = unnorm_diffusion_trajectory[..., 3]  # (B,Pnn,T1)
        valid = (near_cur_future_valid > 0).to(torch.bool)  # (B,Pnn,T1)

        # 2) yaw(라디안) 추출 + unwrap (회전에는 cos/sin 그대로 사용)
        yaw = torch.atan2(sin_y, cos_y)  # (B,Pnn,T1)
        yaw_unwrapped = self._unwrap_phase_torch(yaw.reshape(-1,
                                                             T1))  # (B_Pnn, T1)
        yaw_unwrapped = yaw_unwrapped.reshape(B, Pnn, T1)  # (B,Pnn,T1)

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
        yaw_rate = self._savgol_derivative_masked_torch(
            yaw_unwrapped.reshape(-1, T1),  # (B_Pnn,T1)
            valid.reshape(-1, T1),  # (B_Pnn,T1)
            dt,
            polyorder,
            max_window_len_yaw)  # ω
        yaw_rate = yaw_rate.reshape(B, Pnn, T1)


        # 5) 안전 마스킹(무효 시점은 0)
        if valid is not None:
            v_x = torch.where(valid, v_x, torch.zeros_like(v_x))
            v_y = torch.where(valid, v_y, torch.zeros_like(v_y))
            yaw_rate = torch.where(valid, yaw_rate, torch.zeros_like(yaw_rate))
        unnorm_cur_future_control = torch.stack([v_x, v_y, yaw_rate],
                                                dim=-1)  # (B,Pnn,T1,3)
        return unnorm_cur_future_control


# ---------------------------------------------------------------------------
