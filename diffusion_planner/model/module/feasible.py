import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Iterable, List, Tuple, Union, TypedDict, Optional

import numpy as np

Number = Union[float, int]
ArrayLike = Union[np.ndarray, List[Number], Tuple[Number, ...]]


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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.constraints: Dict[ActorClass, DynamicLimits] = {
            ActorClass.PEDESTRIAN:
                DynamicLimits(
                    v_max_mps=8.0,
                    v_max_kmph=28.8,
                    a_max_mps2=6.0,
                    alpha_max_radps2=8.0,
                    a_lat_max_mps2=5.0,
                    R_min_m=0.00001,
                    omega_max_abs_radps=4.0,
                ),
            ActorClass.BICYCLE:
                DynamicLimits(
                    v_max_mps=20.0,
                    v_max_kmph=72.0,
                    a_max_mps2=6.0,
                    alpha_max_radps2=3.0,
                    a_lat_max_mps2=6.5,
                    R_min_m=1.50,
                    omega_max_abs_radps=2.0,
                ),
            ActorClass.CAR:
                DynamicLimits(
                    v_max_mps=55.6,
                    v_max_kmph=200.0,
                    a_max_mps2=8.0,
                    alpha_max_radps2=2.5,
                    a_lat_max_mps2=8.0,
                    R_min_m=4.50,
                    omega_max_abs_radps=1.2,
                ),
        }

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

    def savgol_filter_for_body_control(
            self,
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+T, 4)
            near_cur_future_valid: torch.Tensor,  # (B, Pnn, 1+T),
    ) -> torch.Tensor:  # (B, Pnn, 1+T, 3)
        raise NotImplementedError

    def compute_midpoint_controls(
        self,
        unnorm_cur_future_body_control: torch.Tensor,
        # (B, Pnn, 1+T, 3)
        near_cur_future_valid: torch.Tensor
        # (B, Pnn, 1+T)
    ) -> torch.Tensor:  # (B, Pnn, T, 3)
        raise NotImplementedError

    def forward(
            self,
            near_cur_future_valid: torch.Tensor,
            # (B, Pnn, 1+T)
            diffusion_trajectory: torch.Tensor,  # (B, Pnn, 1+T, 4)
            cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 3)
            dit_final_hidden_tokens: torch.Tensor,  # (B, Pnn, H)
    ) -> torch.Tensor:  # (B, Pnn, T, 3)
        raise NotImplementedError

    def filter_and_integrate(
        self,
        near_current_state: torch.Tensor,  # (B, Pnn, 4)
        near_current_valid: torch.Tensor,  # (B, Pnn)
        cur_future_seg_body_control: torch.Tensor,  # (B, Pnn, T, 4)
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # (B, Pnn, T, 4), (B, Pnn, T, 3)
        raise NotImplementedError
