# nuplan_extent/planning/simulation/observation/smoother.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np
import numpy.typing as npt

AgentKind = Literal["vehicle", "bicycle", "pedestrian"]


@dataclass
class SlipParams:
    """슬립각(몸체 슬립) 제약/분배 파라미터 묶음.

    Attributes:
        beta_body_max_deg (float): 허용 슬립각 절대치 상한 [deg].
        w_p (float): 위치(방향) 보정 가중.
        w_theta (float): 헤딩 보정 가중.
    """
    beta_body_max_deg: float
    w_p: float
    w_theta: float


@dataclass
class RateParams:
    """각속도/각가속 제약 및 저속 보정 파라미터 묶음.

    Attributes:
        v_dir (float): 저속-헤딩 스위치 임계속도 [m/s].
        v_floor (float): 저속 게이팅용 바닥 속도 [m/s].
        phi_dot_cap (float): 기본 각속도 상한(저속 안정화용) [rad/s].
        R_min (Optional[float]): 최소 회전반경 [m]. 보행자는 None 권장.
        aN_max (float): 허용 횡가속 상한 [m/s^2].
        alpha_max (float): 각가속 상한 [rad/s^2].
        straight_eps_deg (float): 직선 간주 임계 회전량 [deg].
    """
    v_dir: float
    v_floor: float
    phi_dot_cap: float
    R_min: Optional[float]
    aN_max: float
    alpha_max: float
    straight_eps_deg: float


@dataclass
class SmootherConfig:
    """에이전트 타입별 파라미터 번들.

    Attributes:
        veh (SlipParams, RateParams): 차량용 파라미터.
        bic (SlipParams, RateParams): 자전거용 파라미터.
        ped (SlipParams, RateParams): 보행자용 파라미터.
        dt (float): 샘플 간격 [s] (예: 0.1).
    """
    veh_slip: SlipParams
    veh_rate: RateParams
    bic_slip: SlipParams
    bic_rate: RateParams
    ped_slip: SlipParams
    ped_rate: RateParams
    dt: float


# ---------- 1. 전/후진 부호 σ_k 계산 -------------------------------------------------

# ---------- 4. 파이프라인 편의 함수 ---------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict
import numpy as np
import numpy.typing as npt

AgentKind = Literal["vehicle", "bicycle", "pedestrian"]

from typing import Tuple
import numpy as np
import numpy.typing as npt

# =============================================================================
# Helpers (module-level; compute_forward_backward_signs 와 동등한 위치)
# =============================================================================


def _compute_frame_padding_mask(
    raw: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    eps: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """프레임별 패딩 여부를 계산합니다. (0,0,0,0)이면 True.

    Args:
        raw (np.ndarray): (Pnn, 81, 4)
        eps (float): 절대값 합이 eps 이하이면 패딩으로 간주. 기본 0.0.

    Returns:
        np.ndarray: (Pnn, 81) bool. True → 패딩 프레임.
    """
    # 절대합 기준 판별: 정확히 0(또는 eps 이하)이면 패딩
    frame_abs_sum = np.abs(raw).sum(axis=2)  # (Pnn, 81)
    is_padding = frame_abs_sum <= eps
    return is_padding


def _extract_xy_theta(
    raw: npt.NDArray[np.float32],  # (Pnn, 81, 4)
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32],
           npt.NDArray[np.float32]]:
    """(x, y, θ) 성분을 분리합니다.

    Args:
        raw (np.ndarray): (Pnn, 81, 4) [x, y, cos, sin]

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            x:  (Pnn, 81)
            y:  (Pnn, 81)
            th: (Pnn, 81)   # θ = atan2(sin, cos)
    """
    x = raw[..., 0].astype(np.float32)
    y = raw[..., 1].astype(np.float32)
    cos_th = raw[..., 2].astype(np.float32)
    sin_th = raw[..., 3].astype(np.float32)
    th = np.arctan2(sin_th, cos_th).astype(np.float32)
    return x, y, th


def _compute_segment_motion_angles(
        x: npt.NDArray[np.float32],  # (Pnn, 81)
        y: npt.NDArray[np.float32],  # (Pnn, 81)
) -> npt.NDArray[np.float32]:
    """세그먼트 이동방향각 φ̂_i 를 계산합니다.

    정의:
        φ̂_i = atan2( y[i+1] - y[i],  x[i+1] - x[i] )

    Args:
        x (np.ndarray): (Pnn, 81)
        y (np.ndarray): (Pnn, 81)

    Returns:
        np.ndarray: (Pnn, 80)  # i = 0..79
    """
    dx = x[:, 1:] - x[:, :-1]  # (Pnn, 80)
    dy = y[:, 1:] - y[:, :-1]  # (Pnn, 80)
    with np.errstate(invalid="ignore"):
        phi_hat = np.arctan2(dy, dx).astype(np.float32)  # (Pnn, 80)
    return phi_hat


def _wrap_to_pi(angle: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """각도를 [-π, π] 구간으로 wrap합니다.

    구현:
        wrap(α) = atan2(sin α, cos α)

    Args:
        angle (np.ndarray): 임의 shape 라디안 각도.

    Returns:
        np.ndarray: 입력과 동일 shape, [-π, π]에 wrap된 각도.
    """
    return np.arctan2(np.sin(angle), np.cos(angle)).astype(np.float32)


def _compute_alignment_cosine(
        phi_hat: npt.NDArray[np.float32],  # (Pnn, 80)
        theta: npt.NDArray[np.float32],  # (Pnn, 81)
) -> npt.NDArray[np.float32]:
    """정렬도 d_i = cos(wrap(φ̂_i - θ_i))를 계산합니다.

    Args:
        phi_hat (np.ndarray): (Pnn, 80), 세그먼트 이동방향각
        theta (np.ndarray):   (Pnn, 81), 프레임 헤딩

    Returns:
        np.ndarray: (Pnn, 80), 정렬도 d_i
    """
    theta_i = theta[:, :-1]  # (Pnn, 80) 세그먼트 시작 프레임의 헤딩
    delta = _wrap_to_pi(phi_hat - theta_i)  # (Pnn, 80)
    d = np.cos(delta).astype(np.float32)  # (Pnn, 80)
    return d


def _compute_segment_valid_mask(
    frame_is_padding: npt.NDArray[np.bool_],  # (Pnn, 81)
    x: npt.NDArray[np.float32],  # (Pnn, 81)
    y: npt.NDArray[np.float32],  # (Pnn, 81)
    eps_move: float = 0.0,
) -> npt.NDArray[np.bool_]:
    """세그먼트 유효성 마스크를 계산합니다.

    규칙:
        - 세그먼트 i는 프레임 i, i+1 양쪽이 패딩이 아니어야 함.
        - 이동량 sqrt(dx^2+dy^2)가 eps_move 초과여야 함(0이면 방향각 정의 불능).

    Args:
        frame_is_padding (np.ndarray): (Pnn, 81) 프레임별 패딩 여부.
        x (np.ndarray): (Pnn, 81)
        y (np.ndarray): (Pnn, 81)
        eps_move (float): 이동량 임계값(기본 0.0 → 정확히 0만 무효 처리).

    Returns:
        np.ndarray: (Pnn, 80) bool. True → 유효 세그먼트.
    """
    valid_left = ~frame_is_padding[:, :-1]  # (Pnn, 80)
    valid_right = ~frame_is_padding[:, 1:]  # (Pnn, 80)
    dx = x[:, 1:] - x[:, :-1]
    dy = y[:, 1:] - y[:, :-1]
    move_norm = np.hypot(dx, dy)  # (Pnn, 80)
    move_ok = move_norm > eps_move
    return valid_left & valid_right & move_ok  # (Pnn, 80)


def _prepare_sliplimit_basics(
    near_agents_current: npt.NDArray[np.float32],  # (Pnn, 4)
    near_future_raw: npt.NDArray[np.float32],  # (Pnn, 80, 4)
    dt: float,
) -> Tuple[
        npt.NDArray[np.float32],  # cur_future_raw (Pnn,81,4)
        npt.NDArray[np.bool_],  # seg_valid (Pnn,80)
        npt.NDArray[np.float32],  # dist_raw (Pnn,80)
        npt.NDArray[np.float32],  # speed_raw (Pnn,80)
]:
    """슬립 단계 전용 기초량을 한 번만 계산해 반환."""
    cur_future_raw = _concat_current_and_future(near_agents_current,
                                                near_future_raw)  # (Pnn,81,4)
    x_raw, y_raw, _ = _extract_xy_theta(cur_future_raw)  # (Pnn,81)
    frame_is_padding = _compute_frame_padding_mask(cur_future_raw)  # (Pnn,81)
    seg_valid = _compute_segment_valid_mask(frame_is_padding,
                                            x_raw,
                                            y_raw,
                                            eps_move=0.0)  # (Pnn,80)
    dx_raw, dy_raw = _compute_dx_dy(x_raw, y_raw)  # (Pnn,80)
    dist_raw, speed_raw = _compute_segment_speeds(dx_raw, dy_raw,
                                                  float(dt))  # (Pnn,80)
    return cur_future_raw.astype(np.float32), seg_valid, dist_raw, speed_raw


def _segments_to_framewise_sigma(
        sigma_seg_int: npt.NDArray[np.int8],  # (Pnn, 80) in {-1,0,+1}
) -> npt.NDArray[np.int8]:
    """세그먼트 단위 σ를 프레임 길이(81)로 확장합니다.

    규칙:
        - 프레임 k=0..79 → 세그먼트 σ[k]를 그대로 사용.
        - 프레임 k=80(마지막)은 “알 수 없음(0)”으로 둠.

    Args:
        sigma_seg_int (np.ndarray): (Pnn, 80) in {-1,0,+1}

    Returns:
        np.ndarray: (Pnn, 81) in {-1,0,+1}
    """
    Pnn = sigma_seg_int.shape[0]
    out = np.zeros((Pnn, 81), dtype=np.int8)
    out[:, :80] = sigma_seg_int
    out[:, 80] = 0  # 마지막 프레임은 명시적으로 모름
    return out


# =============================================================================
# Public API
# =============================================================================


def compute_forward_backward_signs(
        near_cur_future_raw: npt.NDArray[
            np.float32],  # (Pnn, 81, 4) [x, y, cos, sin]
) -> npt.NDArray[np.int8]:
    """현재 1프레임 + 미래 80프레임에 대해 전/후진 부호 σ_k를 계산합니다.

    개요:
        - 각 세그먼트 i(프레임 i → i+1)에 대해 이동방향각 φ̂_i를 구하고,
        - 시작 프레임의 헤딩 θ_i와의 정렬도 d_i = cos(wrap(φ̂_i - θ_i))의 부호(sign)를
          이용해 전진(+1) / 후진(−1) / 모름(0)을 결정합니다.
        - 반환 shape은 (Pnn, 81)이며, 마지막 프레임의 σ는 0(모름)으로 둡니다.
        - 패딩 프레임(0,0,0,0)이 세그먼트 양단 중 하나라도 포함되거나,
          이동량이 0인 세그먼트는 모름(0)으로 둡니다.

    Args:
        near_cur_future_raw (np.ndarray): shape (Pnn, 81, 4)
            - [:, :, 0:2] = (x, y)
            - [:, :, 2]   = cos(yaw)
            - [:, :, 3]   = sin(yaw)
            - 패딩 슬롯은 전 프레임이 (0,0,0,0)로 채워짐.
              - 어떤 에이전트가 전부 패딩이면 결과 전체가 0.
              - 중간부터 패딩이면 그 이후 구간의 σ는 0.

    Returns:
        np.ndarray: `near_current_future_dir` (Pnn, 81), dtype=int8, 값 ∈ {-1, 0, +1}.
            - +1: 전진
            -  0: 모름(마지막 프레임, 패딩 포함/정지 세그먼트 등)
            - −1: 후진

    Raises:
        ValueError: 입력 shape가 (Pnn, 81, 4)가 아닐 때.
    """
    # 0) 입력 검증 및 기초 분해
    x, y, theta = _extract_xy_theta(near_cur_future_raw)  # (Pnn,81) ×3
    frame_is_padding = _compute_frame_padding_mask(
        near_cur_future_raw)  # (Pnn,81)

    # 1) 세그먼트 이동방향각 φ̂_i, 정렬도 d_i
    phi_hat = _compute_segment_motion_angles(
        x, y)  # (Pnn,80) # 가장 끝 점은 미포함 # 현재 + 미래 79 프레임
    d = _compute_alignment_cosine(phi_hat, theta)  # (Pnn,80) # 현재 + 미래 79 프레임

    # 2) 세그먼트 유효성 마스크(패딩/정지 구간 제외)
    seg_valid = _compute_segment_valid_mask(frame_is_padding,
                                            x,
                                            y,
                                            eps_move=0.0)  # (Pnn,80)

    # 3) σ_i 결정: 유효 구간만 sign(d_i), 그 외 0
    sigma_seg_int = np.zeros_like(
        d, dtype=np.int8)  # (Pnn,80) in {-1,0,+1} # 현재 + 미래 79 프레임
    # sign(d) → {+1(전진), -1(후진)}; 단, d==0이면 0으로 남음(모름)
    clean = np.nan_to_num(d,
                          nan=0.0,
                          posinf=np.finfo(d.dtype).max,
                          neginf=np.finfo(d.dtype).min)
    sigma_values = np.sign(clean).astype(np.int8, copy=False)
    sigma_seg_int[seg_valid] = sigma_values[seg_valid]

    # 4) (Pnn, 81) 프레임 길이로 확장 (마지막 프레임=0)
    near_current_future_dir = _segments_to_framewise_sigma(
        sigma_seg_int)  # (Pnn,81)

    # 5) 완전 패딩 행(전 프레임 패딩) 보호: 전부 0 유지
    full_padding_rows = np.all(frame_is_padding, axis=1)  # (Pnn,)
    if np.any(full_padding_rows):
        near_current_future_dir[full_padding_rows, :] = 0

    return near_current_future_dir


# ---------- 2. 슬립각 제한 (앵커 보정 + 순차 보정) ------------------------------------

# ---------- 3. 각속도 제약 + 스무딩 파이프라인 ---------------------------------------


def build_sigma_aware_direction_angles(
    near_current_future_a2: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    dt: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """σ-aware 유효 방향각 φ_eff 및 속력 v_k를 계산합니다.

    Returns:
        phi_eff: (Pnn, 80)   # 세그먼트 방향각(구간 k: p_k→p_{k+1})
        speeds:  (Pnn, 80)   # 구간 속력 v_k
    """
    pass


def choose_direction_angle_with_low_speed_fallback(
    phi_eff: npt.NDArray[np.float32],  # (Pnn, 80)
    headings: npt.NDArray[np.float32],  # (Pnn, 81)  # 각 프레임 heading (라디안)
    speeds: npt.NDArray[np.float32],  # (Pnn, 80)
    agent_kind_mask: Dict[AgentKind, npt.NDArray[np.bool_]],  # 각 타입별 (Pnn,) 마스크
    rate_params_by_kind: Dict[AgentKind, RateParams],
) -> npt.NDArray[np.float32]:
    """저속 구간에서는 헤딩을, 그 외에는 σ-aware 속도방향을 택해 φ_use를 만듭니다.

    Returns:
        phi_use: (Pnn, 80)
    """
    pass


def compute_effective_yaw_rate(
    phi_use: npt.NDArray[np.float32],  # (Pnn, 80)
    dt: float,
) -> npt.NDArray[np.float32]:
    """연속(unwrapped) 차분으로 유효 각속도 ω_eff(k)을 계산합니다.

    Returns:
        omega_eff: (Pnn, 79)  # 관례에 따라 k=0..78 (구간 차분)
    """
    pass


def clip_yaw_rate_once(
    omega_eff: npt.NDArray[np.float32],  # (Pnn, 79)
    speeds: npt.NDArray[np.float32],  # (Pnn, 80)
    agent_kind_mask: Dict[AgentKind, npt.NDArray[np.bool_]],
    rate_params_by_kind: Dict[AgentKind, RateParams],
) -> npt.NDArray[np.float32]:
    """프레임 단일 상한으로 각속도 1차 클립을 수행합니다.

    Returns:
        omega_clipped: (Pnn, 79)
    """
    pass


def tv_smooth_yaw_rate(
        omega_clipped: npt.NDArray[np.float32],  # (Pnn, 79)
        agent_kind_mask: Dict[AgentKind, npt.NDArray[np.bool_]],
        rate_params_by_kind: Dict[AgentKind, RateParams],
        speeds: npt.NDArray[np.float32],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """각가속 상한 기반의 TV 스무딩(Forward/Backward 2패스)을 적용합니다.

    Returns:
        omega_smooth: (Pnn, 79)
    """
    pass


def integrate_positions_from_yawrate(
    near_current_future_a2: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    phi_eff0: npt.NDArray[np.float32],  # (Pnn,) 시작 세그먼트 방향각
    omega_smooth: npt.NDArray[np.float32],  # (Pnn, 79)
    speeds: npt.NDArray[np.float32],  # (Pnn, 80)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    dt: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """원호 적분(sinc/cosc)으로 위치 갱신, σ-aware 적분으로 최종 φ_{k+1} 복원.

    Returns:
        near_future_positions: (Pnn, 80, 2)  # x,y
        phi_world_end:        (Pnn, 80)      # 각 세그먼트 끝 방향 φ_{k+1}
    """
    pass


def restore_heading_with_body_slip(
        phi_world_end: npt.NDArray[np.float32],  # (Pnn, 80)
        near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
        near_future_body_slip: npt.NDArray[np.float32],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """슬립각 보존 규칙으로 헤딩 θ_{k+1}를 복원합니다.

    Returns:
        headings_future: (Pnn, 80)  # 라디안
    """
    pass


def pack_future_xy_cos_sin(
        near_future_positions: npt.NDArray[np.float32],  # (Pnn, 80, 2)
        headings_future: npt.NDArray[np.float32],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """(x,y,cosθ,sinθ) 포맷으로 미래 80 프레임을 포장합니다.

    Returns:
        near_future_a3: (Pnn, 80, 4)
    """
    pass


# ... (위쪽: SlipParams / RateParams / SmootherConfig / compute_forward_backward_signs 및 헬퍼들 그대로) ...

# =============================================================================
# Slip-limit: module-level helpers
# =============================================================================


def _concat_current_and_future(
        near_agents_current: npt.NDArray[np.float32],  # (Pnn, 4)
        near_future_raw: npt.NDArray[np.float32],  # (Pnn, 80, 4)
) -> npt.NDArray[np.float32]:
    """현재 프레임과 미래 80 프레임을 하나로 이어 붙입니다.

    Args:
        near_agents_current: (Pnn, 4) = [x, y, cos, sin]
        near_future_raw: (Pnn, 80, 4)

    Returns:
        np.ndarray: (Pnn, 81, 4) = [current | future(80)]
    """
    return np.concatenate([near_agents_current[:, None, :], near_future_raw],
                          axis=1).astype(np.float32)


def _sigma_to_theta_eff(
        theta: npt.NDArray[np.float32],  # (...,) 라디안
        sigma: npt.NDArray[np.int8],  # (...,) in {-1,0,+1}
) -> npt.NDArray[np.float32]:
    """σ-aware 유효 헤딩 θ_eff = θ + (σ == -1 ? π : 0)

    Note:
        σ == 0 인 경우는 forward(π 더하지 않음)로 처리합니다.
    """
    return (theta + (np.pi * (sigma == -1))).astype(np.float32)


def _alpha_star(
    speed: npt.NDArray[np.float32],  # (Pnn, 80)
    dt: float,
    w_p: float,
    w_theta: float,
) -> npt.NDArray[np.float32]:
    """분배율 α* = [w_p (v dt)^2] / [w_p (v dt)^2 + w_theta]

    Args:
        speed: (Pnn, 80)
        dt: float
        w_p: float
        w_theta: float

    Returns:
        alpha: (Pnn, 80), 0..1
    """
    vdt2 = (speed * float(dt))**2  # (Pnn, 80)
    num = w_p * vdt2
    den = num + w_theta
    # 안전 처리: den==0 이면 0으로
    alpha = np.divide(num,
                      den,
                      out=np.zeros_like(num, dtype=np.float32),
                      where=(den > 0))
    return alpha.astype(np.float32)


def _clip_to_bounds(
    x: npt.NDArray[np.float32],  # arbitrary shape
    lo: float,
    hi: float,
) -> npt.NDArray[np.float32]:
    """값을 [lo, hi] 범위로 클리핑합니다."""
    return np.minimum(np.maximum(x, lo), hi).astype(np.float32)


def _compute_pedestrian_body_slip(
        cur_future_raw: npt.NDArray[np.float32],  # (Pnn, 81, 4)
        near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
) -> npt.NDArray[np.float32]:
    """보행자 몸체 슬립각 β_s (frame s 기준)를 원래 경로에서 계산합니다.

    β_s = wrap( φ_s − θ_s,eff ),  s = 0..79
    - φ_s: frame s→s+1 속도방향
    - θ_s,eff = θ_s + π·I[σ_s=-1]
    - frame 80은 frame 79 복사
    """
    x, y, theta = _extract_xy_theta(cur_future_raw)  # (Pnn,81)
    phi_hat = _compute_segment_motion_angles(x, y)  # (Pnn,80)
    frame_is_padding = _compute_frame_padding_mask(cur_future_raw)
    seg_valid = _compute_segment_valid_mask(frame_is_padding,
                                            x,
                                            y,
                                            eps_move=0.0)  # (Pnn,80)

    theta_eff_start = _sigma_to_theta_eff(
        theta[:, :-1], near_current_future_dir[:, :-1])  # (Pnn,80)
    beta = _wrap_to_pi(phi_hat - theta_eff_start).astype(np.float32)  # (Pnn,80)

    ped_slip = np.zeros((cur_future_raw.shape[0], 81), dtype=np.float32)
    ped_slip[:, :80] = np.where(seg_valid, beta, 0.0).astype(np.float32)
    ped_slip[:, 80] = ped_slip[:, 79]
    return ped_slip


def _compute_body_slip_from_states(
        cur_future_states: npt.NDArray[np.float32],  # (Pnn, 81, 4)  보정 후 상태
        near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
        seg_valid: npt.NDArray[np.bool_],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """보정된 점/헤딩으로부터 frame s 기준 슬립각 β_s를 계산해 반환.

    β_s = wrap( φ_s − θ_s,eff ), s=0..79. frame 80은 79 복사.
    """
    x, y, theta = _extract_xy_theta(cur_future_states)  # (Pnn,81)
    phi = _compute_segment_motion_angles(x, y)  # (Pnn,80)
    theta_eff_start = _sigma_to_theta_eff(
        theta[:, :-1], near_current_future_dir[:, :-1])  # (Pnn,80)
    beta = _wrap_to_pi(phi - theta_eff_start).astype(np.float32)  # (Pnn,80)

    out = np.zeros((cur_future_states.shape[0], 81), dtype=np.float32)
    out[:, :80] = np.where(seg_valid, beta, 0.0).astype(np.float32)
    out[:, 80] = 0.
    return out


# =============================================================================
# Anchor (current→p0) and Sequential (p0→...→p79) slip limiting
# =============================================================================


def anchor_adjustment_current_to_k0(
    cur_future_copy: npt.NDArray[np.float32],  # (Pnn, 81, 4) in/out
    cur_future_raw: npt.NDArray[np.float32],  # (Pnn, 81, 4) read-only
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    veh_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    cfg: SmootherConfig,
) -> npt.NDArray[np.float32]:
    """시작 구간(현재→p0) 앵커 보정: 차량/자전거만 p0 위치를 보정합니다.

    규칙:
        - 현재 헤딩 θ_cur 고정(불변).
        - σ_0 == -1 이면 θ_eff_cur = θ_cur + π, 그 외 θ_eff_cur = θ_cur.
        - α_{-1} = 0 이므로 방향만 수정(거리 보존).

    Args:
        cur_future_copy: (Pnn, 81, 4) 수정 대상 배열.
        cur_future_raw:   (Pnn, 81, 4) 원본 궤적(거리/속도/φ̂ 계산용).
        near_current_future_dir: (Pnn, 81) in {-1,0,+1}
        veh_valid_mask, bic_valid_mask: 처리 대상 마스크.
        cfg: 스무더 파라미터(β_max 등).

    Returns:
        cur_future_copy: (Pnn, 81, 4)  # p0 위치 보정이 반영됨.
    """
    # 유효 세그먼트(현재→p0)가 있는 에이전트만 처리
    x, y, theta = _extract_xy_theta(cur_future_raw)  # (Pnn,81)
    frame_is_padding = _compute_frame_padding_mask(cur_future_raw)  # (Pnn,81)
    seg_valid = _compute_segment_valid_mask(
        frame_is_padding, x, y, eps_move=0.0)  # (Pnn,80) # 현재 + 미래 79 프레임
    current_valid = seg_valid[:, 0]  # (Pnn,)

    target_rows = (veh_valid_mask | bic_valid_mask) & current_valid  # (Pnn,)
    if not np.any(target_rows):
        return cur_future_copy

    # 준비값
    p_cur = cur_future_copy[target_rows, 0, :2]  # (M,2)
    p0_raw = cur_future_raw[target_rows, 1, :2]  # (M,2)
    d0 = np.linalg.norm(p0_raw - p_cur, axis=1).astype(np.float32)  # (M,)

    theta_cur = theta[target_rows, 0]  # (M,)
    sigma0 = near_current_future_dir[target_rows, 0].astype(np.int8)  # (M,)
    theta_eff_cur = _sigma_to_theta_eff(theta_cur, sigma0)  # (M,)

    phi_hat_m1 = np.arctan2(p0_raw[:, 1] - p_cur[:, 1], p0_raw[:, 0] -
                            p_cur[:, 0]).astype(np.float32)  # (M,)

    # 타입별 β_max(rad) 설정
    # - True: veh, False&True: bic  (둘 다 False면 여기로 안 들어옴)
    rows_idx = np.where(target_rows)[0]
    veh_rows_local = veh_valid_mask[rows_idx]  # (M,)
    bic_rows_local = bic_valid_mask[rows_idx]  # (M,)
    beta_max = np.zeros_like(theta_eff_cur, dtype=np.float32)  # (M,)
    if np.any(veh_rows_local):
        beta_max[veh_rows_local] = _deg2rad(cfg.veh_slip.beta_body_max_deg)
    if np.any(bic_rows_local):
        beta_max[bic_rows_local] = _deg2rad(cfg.bic_slip.beta_body_max_deg)

    beta_hat = _wrap_to_pi(phi_hat_m1 - theta_eff_cur)  # (M,)
    beta_clipped = _clip_to_bounds(beta_hat, -beta_max, +beta_max)  # (M,)
    delta_beta = (beta_hat - beta_clipped).astype(np.float32)  # (M,)

    # α_{-1} = 0 → φ' = φ̂ - Δβ
    phi_prime = (phi_hat_m1 - delta_beta).astype(np.float32)  # (M,)
    # 거리 보존으로 p0 재배치
    p0_new = np.empty_like(p0_raw, dtype=np.float32)  # (M,2)
    p0_new[:, 0] = p_cur[:, 0] + d0 * np.cos(phi_prime)
    p0_new[:, 1] = p_cur[:, 1] + d0 * np.sin(phi_prime)

    # 위치만 갱신(헤딩(cos/sin)은 그대로 유지)
    cur_future_copy[target_rows, 1, 0:2] = p0_new
    return cur_future_copy  #  (Pnn, 81, 4)  # p0 위치 보정이 반영됨.


def sequential_slip_limit_closed_loop(
        cur_future_copy: npt.NDArray[np.float32],  # (Pnn, 81, 4) in/out
        cur_future_raw: npt.NDArray[
            np.float32],  # (Pnn, 81, 4) read-only (fallback 용)
        near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
        veh_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
        bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
        cfg: SmootherConfig,
        *,
        seg_valid_precomputed: Optional[npt.NDArray[
            np.bool_]] = None,  # (Pnn,80)
        dist_raw_precomputed: Optional[npt.NDArray[
            np.float32]] = None,  # (Pnn,80)
        speed_raw_precomputed: Optional[npt.NDArray[
            np.float32]] = None,  # (Pnn,80)
) -> npt.NDArray[np.float32]:
    """k=1..79에 대해 '시작 프레임 s의 유효 헤딩'과 '속도벡터 방향' 차이를 제한하는 폐루프 보정.

    규칙(세그먼트 [s→s+1]):
      - φ_s = atan2(y_{s+1}-y_s, x_{s+1}-x_s)  (현재 보정 상태 기준)
      - θ_s,eff = θ_s + π·I[σ_s=-1]
      - β̂_s = wrap(φ_s − θ_s,eff), β_clip = clip(β̂_s)
      - Δβ_s = β̂_s − β_clip
      - θ_s ← wrap(θ_s + α*_s Δβ_s)
      - φ'_s ← φ_s − (1−α*_s) Δβ_s
      - p_{s+1} ← p_s + d_s [cos φ'_s, sin φ'_s]   (d_s: raw 세그먼트 길이 유지)

    반환:
      - cur_future_copy: 보정 반영
    """
    dt = float(cfg.dt)
    Pnn = cur_future_copy.shape[0]

    # 미리 계산된 값이 있으면 사용, 없으면 fallback (호환성)
    seg_valid = seg_valid_precomputed # (Pnn,80)
    dist_raw = dist_raw_precomputed # (Pnn,80)
    speed_raw = speed_raw_precomputed # (Pnn,80)

    proc_rows = (veh_valid_mask | bic_valid_mask)  # (Pnn,)

    # 타입별 파라미터(에이전트별 스칼라 브로드캐스트)
    beta_max_agent = np.zeros((Pnn,), dtype=np.float32)
    w_p_agent = np.zeros((Pnn,), dtype=np.float32)
    w_th_agent = np.zeros((Pnn,), dtype=np.float32)
    if np.any(veh_valid_mask):
        beta_max_agent[veh_valid_mask] = _deg2rad(
            cfg.veh_slip.beta_body_max_deg)
        w_p_agent[veh_valid_mask] = float(cfg.veh_slip.w_p)
        w_th_agent[veh_valid_mask] = float(cfg.veh_slip.w_theta)
    if np.any(bic_valid_mask):
        beta_max_agent[bic_valid_mask] = _deg2rad(
            cfg.bic_slip.beta_body_max_deg)
        w_p_agent[bic_valid_mask] = float(cfg.bic_slip.w_p)
        w_th_agent[bic_valid_mask] = float(cfg.bic_slip.w_theta)


    # 시간 순방향 폐루프
    # seg_valid: (Pnn,80) ,  현재 + 미래 0 ~ 78 프레임
    for s in range(1, 80): # 미래 0 번째 ~ 78 번째 프레임
        valid_rows = proc_rows & seg_valid[:, s]
        if not np.any(valid_rows):
            continue

        # 현재 보정 상태에서의 속도방향 φ_s
        ps = cur_future_copy[valid_rows, s, 0:2]  # (M,2)
        psp = cur_future_copy[valid_rows, s + 1, 0:2]  # (M,2)
        phi_s = np.arctan2(psp[:, 1] - ps[:, 1],
                           psp[:, 0] - ps[:, 0]).astype(np.float32)

        # 시작 프레임 s의 헤딩 → σ-aware
        theta_s = np.arctan2(cur_future_copy[valid_rows, s, 3],
                             cur_future_copy[valid_rows, s,
                                             2]).astype(np.float32)
        sigma_s = near_current_future_dir[valid_rows, s].astype(np.int8)
        theta_eff_s = _sigma_to_theta_eff(theta_s, sigma_s)  # (M,)

        # β̂_s → clip → Δβ_s
        beta_hat = _wrap_to_pi(phi_s - theta_eff_s)  # (M,)
        beta_max = beta_max_agent[valid_rows]  # (M,)
        beta_clip = np.clip(beta_hat, -beta_max, +beta_max).astype(np.float32)
        delta_beta = (beta_hat - beta_clip).astype(np.float32)

        # α*_s (속도 기반 분배)
        v_s = speed_raw[valid_rows, s].astype(np.float32)
        vdt2 = (v_s * dt)**2
        num = (w_p_agent[valid_rows] * vdt2).astype(np.float32)
        den = (num + w_th_agent[valid_rows]).astype(np.float32)
        alpha_s = np.divide(num,
                            den,
                            out=np.zeros_like(num, dtype=np.float32),
                            where=(den > 0))

        # 헤딩 보정: θ_s ← wrap(θ_s + α*_s Δβ_s)
        theta_s_new = _wrap_to_pi(theta_s + alpha_s * delta_beta)
        cur_future_copy[valid_rows, s,
                        2] = np.cos(theta_s_new).astype(np.float32)
        cur_future_copy[valid_rows, s,
                        3] = np.sin(theta_s_new).astype(np.float32)

        # 이동방향 보정: φ'_s
        phi_prime = (phi_s - (1.0 - alpha_s) * delta_beta).astype(np.float32)

        # 위치 업데이트: p_{s+1} (원시 거리 보존)
        d_s = dist_raw[valid_rows, s].astype(np.float32)
        cur_future_copy[valid_rows, s + 1,
                        0] = ps[:,
                                0] + d_s * np.cos(phi_prime).astype(np.float32)
        cur_future_copy[valid_rows, s + 1,
                        1] = ps[:,
                                1] + d_s * np.sin(phi_prime).astype(np.float32)

    return cur_future_copy


# =============================================================================
# Public API: slip_limit_stage
# =============================================================================


def slip_limit_stage(
    near_agents_current: npt.NDArray[np.float32],  # (Pnn, 4)
    near_future_raw: npt.NDArray[np.float32],  # (Pnn, 80, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    veh_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    ped_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    cfg: SmootherConfig,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """[2단계] 슬립각 제한: 앵커 보정 → (수정된) 폐루프 보정 → 슬립각 β_s 산출.

    변경점:
      - Raw 기초량을 1회 계산해 재사용(중복 제거).
      - sequential 보정에서 '시작 헤딩' 기준으로 슬립 제한(요청 반영).
      - near_future_body_slip의 frame s에는 s의 슬립각(β_s)을 담아 반환.
        (frame 80은 frame 79 복사)
    """
    dt = float(cfg.dt)

    # 0) Raw 기초량 1회 계산
    """
    cur_future_raw : (Pnn,81,4)
    seg_valid : (Pnn,80)
    dist_raw : (Pnn,80)
    speed_raw : (Pnn,80)
    """
    cur_future_raw, seg_valid, dist_raw, speed_raw = _prepare_sliplimit_basics(
        near_agents_current=near_agents_current,
        near_future_raw=near_future_raw,
        dt=dt,
    )
    # cur_future_copy: (Pnn,81,4)
    cur_future_copy = cur_future_raw.copy()  # in/out

    # 1) 앵커 보정 (s=0, 차량/자전거, 거리 보존 · θ_cur 고정)
    cur_future_copy = anchor_adjustment_current_to_k0(
        cur_future_copy=cur_future_copy,
        cur_future_raw=cur_future_raw,
        near_current_future_dir=near_current_future_dir,
        veh_valid_mask=veh_valid_mask,
        bic_valid_mask=bic_valid_mask,
        cfg=cfg,
    )

    # 2) 수정된 폐루프 보정 (s=1..79, 시작 헤딩 기준)
    # cur_future_copy: (Pnn,81,4)
    # vehbic_slip_frames: (Pnn,81)
        # 첫 점 기록 안되어 있음 (자전거/자동차)
    cur_future_copy = sequential_slip_limit_closed_loop(
        cur_future_copy=cur_future_copy,
        cur_future_raw=cur_future_raw,
        near_current_future_dir=near_current_future_dir,
        veh_valid_mask=veh_valid_mask,
        bic_valid_mask=bic_valid_mask,
        cfg=cfg,
        seg_valid_precomputed=seg_valid,
        dist_raw_precomputed=dist_raw,
        speed_raw_precomputed=speed_raw,
    )

    # 3) (최종) 슬립각 β_s 계산을 '보정된 상태'에서 일괄 수행  → frame s ↦ β_s
    # (Pnn,81)
    near_future_body_slip = _compute_body_slip_from_states(
        cur_future_states=cur_future_copy,
        near_current_future_dir=near_current_future_dir,
        seg_valid=seg_valid,
    )  # (Pnn,81) with frame s = β_s, frame 80 = 0.

    # 반환
    near_current_future_a2 = cur_future_copy
    return near_current_future_a2, near_future_body_slip


# =========================
# Stage-3 helpers (module level)
# =========================

from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt

# ---- 공용 유틸 (이 파일의 다른 단계에서도 재사용 가능) -----------------------------


def _deg2rad(deg: float | npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """도(deg)를 라디안(rad)으로 변환.

    Args:
        deg: 스칼라 또는 배열 [deg]

    Returns:
        np.ndarray: 동일 shape, dtype=float32 [rad]
    """
    return (np.asarray(deg, dtype=np.float32) * np.pi / 180.0).astype(
        np.float32)


def _compute_dx_dy(
        x: npt.NDArray[np.float32],  # (Pnn, 81)
        y: npt.NDArray[np.float32],  # (Pnn, 81)
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """프레임 차분으로 세그먼트 이동량 (dx, dy)를 계산합니다.

    Args:
        x (np.ndarray): (Pnn, 81)
        y (np.ndarray): (Pnn, 81)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            dx: (Pnn, 80)
            dy: (Pnn, 80)
    """
    dx = (x[:, 1:] - x[:, :-1]).astype(np.float32)
    dy = (y[:, 1:] - y[:, :-1]).astype(np.float32)
    return dx, dy


def _compute_segment_speeds(
    dx: npt.NDArray[np.float32],  # (Pnn, 80)
    dy: npt.NDArray[np.float32],  # (Pnn, 80)
    dt: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """세그먼트 거리/속력(d, v)을 계산합니다.

    Args:
        dx (np.ndarray): (Pnn, 80)
        dy (np.ndarray): (Pnn, 80)
        dt (float): 샘플 간격 [s]

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            dist:  (Pnn, 80)  # 세그먼트 길이
            speed: (Pnn, 80)  # 세그먼트 속력
    """
    dist = np.hypot(dx, dy).astype(np.float32)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    speed = (dist / dt).astype(np.float32)
    return dist, speed


def _sigma_to_pi_offset(
        sigma_frame: npt.NDArray[np.int8],  # (..,) in {-1,0,+1}
) -> npt.NDArray[np.float32]:
    """σ 프레임 배열을 π 오프셋(후진=π, 그 외=0)으로 변환.

    Args:
        sigma_frame (np.ndarray): (...,) in {-1, 0, +1}

    Returns:
        np.ndarray: (...,) float32, 값은 {0 or π}
    """
    return np.where(sigma_frame == -1, np.pi, 0.0).astype(np.float32)


def _sinc(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """sinc(x) = sin(x)/x (x→0, 1). numpy의 np.sinc는 sin(πu)/(πu)이므로 변환.

    Args:
        x (np.ndarray): 임의 shape 라디안

    Returns:
        np.ndarray: 동일 shape, dtype=float32
    """
    return np.sinc((x.astype(np.float32)) / np.pi).astype(np.float32)


def _cosc(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """cosc(x) = (1 - cos x) / x; 수치안정형: 0.5*x*(sinc(x/2))^2.

    Args:
        x (np.ndarray): 임의 shape 라디안

    Returns:
        np.ndarray: 동일 shape, dtype=float32
    """
    s = _sinc(0.5 * x.astype(np.float32))
    return 0.5 * x.astype(np.float32) * (s * s).astype(np.float32)


def _sigma_adjust_phi(
        phi_world_seg: npt.NDArray[np.float32],  # (Pnn, 80)
        sigma_seg: npt.NDArray[np.int8],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """σ-aware 속도벡터 방향각 φ_eff = φ - (σ==-1?π:0)을 계산합니다.

    Args:
        phi_world_seg (np.ndarray): (Pnn, 80) 세계 좌표계 속도방향각
        sigma_seg (np.ndarray):     (Pnn, 80) 세그먼트 σ (frame k의 σ)

    Returns:
        np.ndarray: (Pnn, 80) φ_eff in [-π, π]
    """
    pi_off = _sigma_to_pi_offset(sigma_seg).astype(np.float32)  # (Pnn,80)
    return _wrap_to_pi(phi_world_seg - pi_off)


def _build_agent_param_arrays(
    veh_mask: npt.NDArray[np.bool_],  # (Pnn,)
    bic_mask: npt.NDArray[np.bool_],  # (Pnn,)
    ped_mask: npt.NDArray[np.bool_],  # (Pnn,)
    cfg: SmootherConfig,
) -> Dict[str, npt.NDArray[np.float32]]:
    """에이전트 타입별 RateParams를 (Pnn,) 배열로 브로드캐스트 준비.

    Returns:
        dict[str, np.ndarray]: 각 키는 ['v_dir','v_floor','phi_dot_cap','R_min','aN_max','alpha_max'].
            - 모두 shape (Pnn,), dtype=float32
            - R_min은 보행자에 대해 np.inf 로 설정(미적용).
    """
    Pnn = veh_mask.shape[0]
    out = {
        k: np.zeros((Pnn,), dtype=np.float32) for k in
        ['v_dir', 'v_floor', 'phi_dot_cap', 'R_min', 'aN_max', 'alpha_max']
    }

    # 차량
    if np.any(veh_mask):
        out['v_dir'][veh_mask] = float(cfg.veh_rate.v_dir)
        out['v_floor'][veh_mask] = float(cfg.veh_rate.v_floor)
        out['phi_dot_cap'][veh_mask] = float(cfg.veh_rate.phi_dot_cap)
        out['R_min'][veh_mask] = float(
            cfg.veh_rate.R_min if cfg.veh_rate.R_min is not None else np.inf)
        out['aN_max'][veh_mask] = float(cfg.veh_rate.aN_max)
        out['alpha_max'][veh_mask] = float(cfg.veh_rate.alpha_max)

    # 자전거
    if np.any(bic_mask):
        out['v_dir'][bic_mask] = float(cfg.bic_rate.v_dir)
        out['v_floor'][bic_mask] = float(cfg.bic_rate.v_floor)
        out['phi_dot_cap'][bic_mask] = float(cfg.bic_rate.phi_dot_cap)
        out['R_min'][bic_mask] = float(
            cfg.bic_rate.R_min if cfg.bic_rate.R_min is not None else np.inf)
        out['aN_max'][bic_mask] = float(cfg.bic_rate.aN_max)
        out['alpha_max'][bic_mask] = float(cfg.bic_rate.alpha_max)

    # 보행자 (R_min 미적용 → +∞)
    if np.any(ped_mask):
        out['v_dir'][ped_mask] = float(cfg.ped_rate.v_dir)
        out['v_floor'][ped_mask] = float(cfg.ped_rate.v_floor)
        out['phi_dot_cap'][ped_mask] = float(cfg.ped_rate.phi_dot_cap)
        out['R_min'][ped_mask] = np.inf
        out['aN_max'][ped_mask] = float(cfg.ped_rate.aN_max)
        out['alpha_max'][ped_mask] = float(cfg.ped_rate.alpha_max)

    return out


def _low_speed_switch_phi(
        phi_eff_seg: npt.NDArray[
            np.float32],  # (Pnn, 80) # "현재 + 미래 0, ..., 78"
        theta_frames: npt.NDArray[
            np.float32],  # (Pnn, 81) # "현재 + 미래 0, ..., 79"
        speeds: npt.NDArray[np.float32],  # (Pnn, 80) # "현재 + 미래 0, ..., 78"
        v_dir: npt.NDArray[np.float32],  # (Pnn,)
) -> npt.NDArray[np.float32]:
    """저속 구간은 헤딩을, 그 외에는 σ-aware 속도방향을 채택.

    규칙:
        phi_use[k] = theta[k]  if v[k] < v_dir
                    = phi_eff[k] otherwise

    Args:
        phi_eff_seg (np.ndarray): (Pnn, 80)
        theta_frames (np.ndarray): (Pnn, 81), 프레임 헤딩 [rad]
        speeds (np.ndarray): (Pnn, 80)
        v_dir (np.ndarray): (Pnn,) 에이전트별 임계속도

    Returns:
        np.ndarray: (Pnn, 80) phi_use_seg
    """
    use_heading = (speeds < v_dir[:,
                                  None])  # (Pnn,80) bool "현재 + 미래 0, ..., 78"
    phi_use = np.where(use_heading, theta_frames[:, :-1],
                       phi_eff_seg).astype(np.float32)  # "현재 + 미래 0, ..., 78"
    return _wrap_to_pi(phi_use)  # "현재 + 미래 0, ..., 78"


def _unwrap_diff_along_time(
    angles_frame: npt.NDArray[np.float32],  # (Pnn, 81) 프레임 기반 각도열(연속화 대상)
    dt: float,
) -> npt.NDArray[np.float32]:
    """프레임 기반 각도열을 언랩 후 차분해 ω[k] (세그먼트 80개)를 산출.

    Args:
        angles_frame (np.ndarray): (Pnn, 81)  # k=0..80
        dt (float): 샘플 간격 [s]

    Returns:
        np.ndarray: (Pnn, 80)  # k=0..79, ω[k] = (unwrap(φ_{k+1})-unwrap(φ_k))/dt
    """
    unwrapped = np.unwrap(angles_frame.astype(np.float64),
                          axis=1)  # 안정성 위해 float64로 언랩
    dphi = (unwrapped[:, 1:] - unwrapped[:, :-1]).astype(np.float32)  # (Pnn,80)
    if dt <= 0:
        raise ValueError("dt must be positive.")
    return (dphi / dt).astype(np.float32)


def _compute_omega_max(
        speeds: npt.NDArray[np.float32],  # (Pnn, 80)
        params: Dict[str, npt.NDArray[
            np.float32]],  # ('phi_dot_cap','R_min','aN_max','v_floor')
) -> npt.NDArray[np.float32]:
    """세그먼트별 허용 각속도 상한 ω_max(k)을 계산합니다.

    공식:
        v_star = max(v_k, v_floor)
        ω_max  = min( φ_dot_cap,  v_k/R_min,  aN_max / v_star )
        - 보행자: R_min = +∞ 로 설정되어 v/R_min 항이 무시됨.

    Returns:
        np.ndarray: (Pnn, 80)
    """
    v = speeds
    v_star = np.maximum(v,
                        params['v_floor'][:,
                                          None]).astype(np.float32)  # (Pnn,80)
    comp0 = params['phi_dot_cap'][:,
                                  None].astype(np.float32)  # (Pnn,1)→(Pnn,80)
    # R_min==+∞ → comp1=0? 가 아니라 v/R_min→0. 하지만 min에서 0이 우세해버리므로,
    # 보행자는 R_min=+∞ 이면 comp1=+∞ 로 넣어 "무시"되도록 한다.
    R = params['R_min'][:, None]
    comp1 = np.divide(v,
                      R,
                      out=np.full_like(v, np.inf, dtype=np.float32),
                      where=np.isfinite(R))
    comp2 = (params['aN_max'][:, None] / v_star).astype(np.float32)

    omega_max = np.minimum(comp0, np.minimum(comp1, comp2)).astype(np.float32)
    # 수치적으로 음수/NaN 방지
    omega_max = np.where(np.isfinite(omega_max), omega_max,
                         comp0).astype(np.float32)
    omega_max = np.maximum(omega_max, 0.0).astype(np.float32)
    return omega_max


def _clip_once(
        omega: npt.NDArray[np.float32],  # (Pnn, 80)
        omega_max: npt.NDArray[np.float32],  # (Pnn, 80)
        valid_mask: npt.NDArray[np.bool_],  # (Pnn, 80) 세그먼트 유효
) -> npt.NDArray[np.float32]:
    """한 번의 포인트별 상한 클립.

    Args:
        omega (np.ndarray): (Pnn, 80)
        omega_max (np.ndarray): (Pnn, 80)
        valid_mask (np.ndarray): (Pnn, 80)

    Returns:
        np.ndarray: (Pnn, 80)
    """
    mag = np.minimum(np.abs(omega), omega_max).astype(np.float32)
    clipped = np.sign(omega).astype(np.float32) * mag
    # 무효 세그먼트는 0 유지
    return np.where(valid_mask, clipped, 0.0).astype(np.float32)


def _tv_smooth_forward_backward(
    omega: npt.NDArray[np.float32],  # (Pnn, 80)
    omega_max: npt.NDArray[np.float32],  # (Pnn, 80)
    alpha_max_dt: npt.NDArray[np.float32],  # (Pnn,)  = alpha_max * dt
    valid_mask: npt.NDArray[np.bool_],  # (Pnn, 80)
    n_pass: int = 1,
) -> npt.NDArray[np.float32]:
    """각가속 제한 기반 TV 스무딩(Forward/Backward 왕복).

    Args:
        omega (np.ndarray): (Pnn, 80) 1차 클립된 요율
        omega_max (np.ndarray): (Pnn, 80) 스텝별 상한
        alpha_max_dt (np.ndarray): (Pnn,) 에이전트별 각가속 상한 * dt
        valid_mask (np.ndarray): (Pnn, 80) 세그먼트 유효
        n_pass (int): 왕복 반복 횟수(기본 1)

    Returns:
        np.ndarray: (Pnn, 80) 스무딩된 각속도
    """
    Pnn = omega.shape[0]
    out = omega.copy().astype(np.float32)

    for _ in range(max(1, int(n_pass))):
        # Forward
        fwd = np.zeros_like(out, dtype=np.float32)
        # k=0: 단순 상한/마스크
        lower = (-omega_max[:, 0]).astype(np.float32)
        upper = (+omega_max[:, 0]).astype(np.float32)
        fwd[:, 0] = np.clip(out[:, 0], lower, upper)
        fwd[:, 0] = np.where(valid_mask[:, 0], fwd[:, 0], 0.0)

        for k in range(1, 80):
            lo = fwd[:, k - 1] - alpha_max_dt
            hi = fwd[:, k - 1] + alpha_max_dt
            # 1차: 각가속 제한
            z = np.clip(out[:, k], lo, hi)
            # 2차: 스텝별 상한
            z = np.clip(z, -omega_max[:, k], +omega_max[:, k])
            # 유효 마스크
            fwd[:, k] = np.where(valid_mask[:, k], z, 0.0).astype(np.float32)

        # Backward
        bwd = np.zeros_like(out, dtype=np.float32)
        lower = (-omega_max[:, -1]).astype(np.float32)
        upper = (+omega_max[:, -1]).astype(np.float32)
        bwd[:, -1] = np.clip(fwd[:, -1], lower, upper)
        bwd[:, -1] = np.where(valid_mask[:, -1], bwd[:, -1], 0.0)

        for k in range(78, -1, -1):
            lo = bwd[:, k + 1] - alpha_max_dt
            hi = bwd[:, k + 1] + alpha_max_dt
            z = np.clip(fwd[:, k], lo, hi)
            z = np.clip(z, -omega_max[:, k], +omega_max[:, k])
            bwd[:, k] = np.where(valid_mask[:, k], z, 0.0).astype(np.float32)

        out = 0.5 * (fwd + bwd)
        out = np.where(valid_mask, out, 0.0).astype(np.float32)

    return out


def _integrate_arc_positions_and_phi(
    p0_xy: npt.NDArray[np.float32],  # (Pnn, 2) frame-0 위치
    sigma_frames: npt.NDArray[np.int8],  # (Pnn, 81) σ 프레임열
    phi_eff0: npt.NDArray[np.float32],  # (Pnn,)   시작 σ-aware 방향각
    dphi_seq: npt.NDArray[np.float32],  # (Pnn, 80) Δφ[k] = ω_smooth[k] dt
    speed: npt.NDArray[np.float32],  # (Pnn, 80) v[k]
    dt: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """σ-aware 축에서 φ를 적분하고, 원호적분으로 위치를 갱신합니다.

    절차:
        1) φ_eff_frame[0] = φ_eff0
           φ_eff_frame[k+1] = wrap( φ_eff_frame[k] + dφ[k] )
        2) 세계 방향각(시작/끝):
           φ_world_start[k] = wrap( φ_eff_frame[k]   + π*(σ_frame[k]  == -1) )
           φ_world_end  [k] = wrap( φ_eff_frame[k+1] + π*(σ_frame[k+1]== -1) )
        3) 원호 적분(각 스텝 이동거리 Δs = v[k] dt):
           (sinc/cosc)로 p_{k+1} 계산, 누적합

    Args:
        p0_xy (np.ndarray): (Pnn, 2) 시작 위치
        sigma_frames (np.ndarray): (Pnn, 81)
        phi_eff0 (np.ndarray): (Pnn,)
        dphi_seq (np.ndarray): (Pnn, 80)
        speed (np.ndarray): (Pnn, 80)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            positions_1_80: (Pnn, 80, 2)  # frame 1..80 위치
            phi_world_end:  (Pnn, 80)     # 각 세그먼트 끝(world) 방향각 φ_{k+1}
    """
    Pnn = p0_xy.shape[0]
    # φ_eff_frame: (Pnn, 81)
    phi_eff_frame = np.zeros((Pnn, 81), dtype=np.float32)
    phi_eff_frame[:, 0] = phi_eff0.astype(np.float32)
    # 누적 적분
    cum = np.cumsum(dphi_seq.astype(np.float32), axis=1)  # (Pnn,80)
    phi_eff_frame[:, 1:] = _wrap_to_pi(phi_eff_frame[:, [0]] + cum)

    # 세계 시작/끝 방향각
    pi_off_start = _sigma_to_pi_offset(sigma_frames[:, :-1])  # (Pnn,80)
    pi_off_end = _sigma_to_pi_offset(sigma_frames[:, 1:])  # (Pnn,80)
    phi_world_start = _wrap_to_pi(phi_eff_frame[:, :-1] + pi_off_start)
    phi_world_end = _wrap_to_pi(phi_eff_frame[:, 1:] + pi_off_end)

    # 원호 적분
    ds = (speed.astype(np.float32)) * np.float32(
        dt)  # (Pnn,80)  (cfg.dt가 상단 스코프에 있다면 전달 필요)
    # 이 헬퍼는 순수함수로 두기 위해 dt를 외부 전역에 의존하지 않도록, 상위에서 Δs를 전달해도 됩니다.
    # 여기서는 간편화를 위해 ds 계산을 유지합니다.

    sinc = _sinc(dphi_seq)
    cosc = _cosc(dphi_seq)
    c = np.cos(phi_world_start).astype(np.float32)
    s = np.sin(phi_world_start).astype(np.float32)

    step_dx = (ds * (c * sinc - s * cosc)).astype(np.float32)  # (Pnn,80)
    step_dy = (ds * (s * sinc + c * cosc)).astype(np.float32)  # (Pnn,80)
    cum_dx = np.cumsum(step_dx, axis=1).astype(np.float32)  # (Pnn,80)
    cum_dy = np.cumsum(step_dy, axis=1).astype(np.float32)  # (Pnn,80)

    x1_80 = (p0_xy[:, [0]] + cum_dx).astype(np.float32)  # (Pnn,80)
    y1_80 = (p0_xy[:, [1]] + cum_dy).astype(np.float32)  # (Pnn,80)

    positions = np.stack([x1_80, y1_80],
                         axis=-1).astype(np.float32)  # (Pnn,80,2)
    return positions, phi_world_end.astype(np.float32)


def _restore_heading_from_phi_and_slip(
        phi_world_end: npt.NDArray[np.float32],  # (Pnn, 80)
        sigma_frames: npt.NDArray[np.int8],  # (Pnn, 81)
        body_slip_frames: npt.NDArray[np.float32],  # (Pnn, 81) or (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """슬립각을 보존하며 θ_{k+1}를 복원합니다.

    공식:
        θ_eff_{k+1} = wrap( φ_{k+1} - β^b_k )
        θ_{k+1}     = wrap( θ_eff_{k+1} - (σ_{k+1}==-1 ? π : 0) )

    Args:
        phi_world_end (np.ndarray): (Pnn, 80)
        sigma_frames (np.ndarray): (Pnn, 81)
        body_slip_frames (np.ndarray): (Pnn, 81) 또는 (Pnn, 80)

    Returns:
        np.ndarray: (Pnn, 80) θ_{k+1}
    """
    # β^b_k 준비: (Pnn,80)
    if body_slip_frames.shape[1] == 81:
        beta_b = body_slip_frames[:, :80].astype(np.float32)
    else:
        beta_b = body_slip_frames.astype(np.float32)

    theta_eff_end = _wrap_to_pi(phi_world_end - beta_b)  # (Pnn,80)
    pi_off_end = _sigma_to_pi_offset(sigma_frames[:, 1:])  # (Pnn,80)
    theta_end = _wrap_to_pi(theta_eff_end - pi_off_end)  # (Pnn,80)
    return theta_end


def _pack_xy_cos_sin(
        pos_1_80: npt.NDArray[np.float32],  # (Pnn, 80, 2)
        headings_1_80: npt.NDArray[np.float32],  # (Pnn, 80)
) -> npt.NDArray[np.float32]:
    """(x,y,cosθ,sinθ) 포맷으로 포장.

    Args:
        pos_1_80 (np.ndarray): (Pnn, 80, 2)
        headings_1_80 (np.ndarray): (Pnn, 80)

    Returns:
        np.ndarray: (Pnn, 80, 4)
    """
    cosh = np.cos(headings_1_80).astype(np.float32)
    sinh = np.sin(headings_1_80).astype(np.float32)
    return np.concatenate([pos_1_80, cosh[..., None], sinh[..., None]],
                          axis=2).astype(np.float32)


# =========================
# Stage-3 main
# =========================


def yawrate_smooth_stage(
    near_current_future_a2: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    near_future_body_slip: npt.NDArray[np.float32],  # (Pnn, 81) or (Pnn, 80)
    veh_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    ped_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    cfg: SmootherConfig,
) -> npt.NDArray[np.float32]:
    """[3단계] 각속도 제약 + 스무딩: σ-aware 방향각, 저속 보정, TV 스무딩, 적분, 헤딩 복원.

    파이프라인:
        0) σ-aware 유효 방향각(세그먼트)과 속력 계산
        1) 저속 스위치: v<v_dir → 헤딩 사용, 그 외 φ_eff 사용 → φ_use_seg
        2) φ_use 프레임열 구성(길이 81), 언랩 차분 → ω_eff(80)
        3) 스텝 상한 ω_max 계산 후 1차 클립
        4) TV 스무딩(각가속 제한 Forward/Backward, 1회 왕복)
        5) Δφ 적분 + sinc/cosc 원호 적분으로 위치 갱신, φ_world_end 복원
        6) 슬립각 보존으로 헤딩 복원
        7) (x,y,cosθ,sinθ) 포장 및 유효 마스크로 쓰기

    주의:
        - 패딩 프레임(0,0,0,0)이 포함된 세그먼트는 무효로 간주하고 쓰기를 하지 않습니다.
        - 입력이 사람/자전거/자동차 혼재일 수 있으므로, 에이전트별 파라미터를 배열화합니다.

    Returns:
        np.ndarray: near_future_a3 (Pnn, 80, 4)
    """
    dt = float(cfg.dt)

    # 0) 기초량: 좌표/헤딩/세그먼트/σ
    x, y, theta_frames = _extract_xy_theta(
        near_current_future_a2)  # (Pnn,81)×3 # 현재 + 미래 0, ..., 79
    frame_is_padding = _compute_frame_padding_mask(
        near_current_future_a2)  # (Pnn,81) # 현재 + 미래 0, ..., 79
    seg_valid = _compute_segment_valid_mask(
        frame_is_padding, x, y,
        eps_move=0.0)  # (Pnn,80)  (양 끝 프레임이 non-padding) # 현재 + 미래 0, ..., 78

    dx, dy = _compute_dx_dy(x, y)  # (Pnn,80) # "현재 + 미래 0, ..., 78"
    _, speeds = _compute_segment_speeds(dx, dy,
                                        dt)  # (Pnn,80)  # "현재 + 미래 0, ..., 78"
    phi_world_seg = np.arctan2(dy, dx).astype(
        np.float32)  # (Pnn,80)  # "현재 + 미래 0, ..., 78"
    sigma_seg = near_current_future_dir[:, :-1].astype(
        np.int8)  # (Pnn,80), frame k의 σ 사용  # "현재 + 미래 0, ..., 78"
    sigma_frames = near_current_future_dir.astype(
        np.int8)  # (Pnn,81)  # "현재 + 미래 0, ..., 79"

    # σ-aware 방향각(세그먼트)
    # (Pnn,80)  # "현재 + 미래 0, ..., 78"
    phi_eff_seg = _sigma_adjust_phi(
        phi_world_seg, sigma_seg)

    # 1) 저속 스위치용 파라미터 준비(에이전트별)
    # Dict[str, npt.NDArray[np.float32]] #
    # 'v_dir','v_floor','phi_dot_cap','R_min','aN_max','alpha_max'
    # 모두 shape (Pnn,)
    params = _build_agent_param_arrays(veh_valid_mask, bic_valid_mask,
                                       ped_valid_mask, cfg)
    # "현재 + 미래 0, ..., 78"
    phi_use_seg = _low_speed_switch_phi(phi_eff_seg, theta_frames, speeds,
                                        params['v_dir'])  # (Pnn,80)

    # 2) φ_use 프레임열(81) 구성 → 언랩 차분으로 ω_eff(80)
    # "현재 + 미래 0, ..., 79"
    phi_use_frame = np.concatenate([phi_use_seg, phi_use_seg[
        :,
        -1:,
    ]], axis=1).astype(np.float32)  # (Pnn,81)
    # omega_eff : "현재 + 미래 0, ..., 78"
    # TODO:
    omega_eff = _unwrap_diff_along_time(phi_use_frame, dt)  # (Pnn,80)

    # 유효하지 않은 세그먼트는 ω=0으로
    omega_eff = np.where(seg_valid, omega_eff, 0.0).astype(np.float32)

    # 3) 스텝별 허용 상한 ω_max (v_floor 게이팅/반경/옆가속/기본캡 포함)
    # "현재 + 미래 0, ..., 78"
    omega_max = _compute_omega_max(speeds, params)  # (Pnn,80)
    # "현재 + 미래 0, ..., 78"
    omega_clip1 = _clip_once(omega_eff, omega_max, seg_valid)  # (Pnn,80)

    # 4) TV 스무딩 (각가속 제한) — 왕복 1회 (필요 시 cfg에 반복 횟수 추가 가능)
    alpha_max_dt = (params['alpha_max'] * dt).astype(np.float32)  # (Pnn,)
    # "현재 + 미래 0, ..., 78"
    omega_smooth = _tv_smooth_forward_backward(
        omega=omega_clip1,  # "현재 + 미래 0, ..., 78"
        omega_max=omega_max,  # "현재 + 미래 0, ..., 78"
        alpha_max_dt=alpha_max_dt,
        valid_mask=seg_valid,  # "현재 + 미래 0, ..., 78"
        n_pass=1,
    )  # (Pnn,80) # "현재 + 미래 0, ..., 78"

    # 5) φ 적분 → 위치 원호 적분
    # "현재 + 미래 0, ..., 78"
    dphi_seq = (omega_smooth * dt).astype(np.float32)  # (Pnn,80)
    # phi_eff0: 현재
    phi_eff0 = phi_eff_seg[:, 0].astype(np.float32)  # (Pnn,)
    p0_xy = near_current_future_a2[:, 0, :2].astype(np.float32)  # (Pnn,2)

    # 원호 적분 / φ_world_end 복원
    positions_1_80, phi_world_end = _integrate_arc_positions_and_phi(
        p0_xy=p0_xy,  # (Pnn,2) # 현재 위치
        sigma_frames=sigma_frames,  # (Pnn,81)  # "현재 + 미래 0, ..., 79"
        phi_eff0=phi_eff0,  # (Pnn,) # 현재
        dphi_seq=dphi_seq,  # (Pnn,80) # "현재 + 미래 0, ..., 78"
        speed=speeds,  # (Pnn,80)  # "현재 + 미래 0, ..., 78"
        dt=dt,
    )  # (Pnn,80,2), (Pnn,80)

    # 6) 헤딩 복원(슬립각 보존)
    headings_1_80 = _restore_heading_from_phi_and_slip(
        phi_world_end=phi_world_end,
        sigma_frames=sigma_frames,
        body_slip_frames=near_future_body_slip,
    )  # (Pnn,80)

    # 7) (x,y,cosθ,sinθ) 포장 + 유효 세그먼트에만 쓰기
    future_a3_full = _pack_xy_cos_sin(positions_1_80,
                                      headings_1_80)  # (Pnn,80,4)
    # 기본값: 입력 a2의 미래(1..80)를 복사해 두고
    near_future_a3 = near_current_future_a2[:, 1:, :].copy().astype(
        np.float32)  # (Pnn,80,4)
    # 쓰기 마스크: (Pnn,80,1)
    write_mask = seg_valid[..., None]  # (Pnn,80,1)
    near_future_a3 = np.where(write_mask, future_a3_full,
                              near_future_a3).astype(np.float32)

    return near_future_a3
