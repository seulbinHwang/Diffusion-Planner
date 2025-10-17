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
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
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
    theta: npt.NDArray[np.float32],    # (Pnn, 81)
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
    d = np.cos(delta).astype(np.float32)    # (Pnn, 80)
    return d


def _compute_segment_valid_mask(
    frame_is_padding: npt.NDArray[np.bool_],  # (Pnn, 81)
    x: npt.NDArray[np.float32],               # (Pnn, 81)
    y: npt.NDArray[np.float32],               # (Pnn, 81)
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


def _segments_to_framewise_sigma(
    sigma_seg_int: npt.NDArray[np.int8],     # (Pnn, 80) in {-1,0,+1}
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
    near_cur_future_raw: npt.NDArray[np.float32],  # (Pnn, 81, 4) [x, y, cos, sin]
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
    x, y, theta = _extract_xy_theta(near_cur_future_raw)            # (Pnn,81) ×3
    frame_is_padding = _compute_frame_padding_mask(near_cur_future_raw)  # (Pnn,81)

    # 1) 세그먼트 이동방향각 φ̂_i, 정렬도 d_i
    phi_hat = _compute_segment_motion_angles(x, y)                  # (Pnn,80) # 가장 끝 점은 미포함
    d = _compute_alignment_cosine(phi_hat, theta)                   # (Pnn,80)

    # 2) 세그먼트 유효성 마스크(패딩/정지 구간 제외)
    seg_valid = _compute_segment_valid_mask(frame_is_padding, x, y, eps_move=0.0)  # (Pnn,80)

    # 3) σ_i 결정: 유효 구간만 sign(d_i), 그 외 0
    sigma_seg_int = np.zeros_like(d, dtype=np.int8)  # (Pnn,80) in {-1,0,+1}
    # sign(d) → {+1(전진), -1(후진)}; 단, d==0이면 0으로 남음(모름)
    sigma_values = np.sign(d, dtype=np.int8)  # numpy>=1.20: dtype 인자 사용
    sigma_seg_int[seg_valid] = sigma_values[seg_valid]

    # 4) (Pnn, 81) 프레임 길이로 확장 (마지막 프레임=0)
    near_current_future_dir = _segments_to_framewise_sigma(sigma_seg_int)  # (Pnn,81)

    # 5) 완전 패딩 행(전 프레임 패딩) 보호: 전부 0 유지
    full_padding_rows = np.all(frame_is_padding, axis=1)  # (Pnn,)
    if np.any(full_padding_rows):
        near_current_future_dir[full_padding_rows, :] = 0

    return near_current_future_dir


# ---------- 2. 슬립각 제한 (앵커 보정 + 순차 보정) ------------------------------------


def anchor_adjustment_current_to_k0(
    near_agents_current: npt.NDArray[np.float32],  # (Pnn, 4)
    near_future_raw: npt.NDArray[np.float32],  # (Pnn, 80, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    veh_bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    params_by_kind: Dict[AgentKind, SlipParams],
    dt: float,
) -> npt.NDArray[np.float32]:
    """시작 구간(현재→k=0) 앵커 보정: p0 위치만 조정해 초기 슬립 위반을 완화합니다.

    Returns:
        near_cur_future_stage: (Pnn, 81, 4)
            - [current | corrected p0..p79]로 구성된 배열
            - 보행자 구간은 입력을 그대로 통과(보정 없음)
    """
    pass


def sequential_slip_limit_closed_loop(
    near_cur_future_stage: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    veh_bic_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    params_by_kind: Dict[AgentKind, SlipParams],
    dt: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """k=0..79 순차(폐루프) 슬립각 제한을 적용합니다.

    Returns:
        near_current_future_a2: (Pnn, 81, 4)
            - 사람: 입력 유지, 차량/자전거: 보정 후 current..p79
        near_future_body_slip: (Pnn, 80)
            - 세그먼트별(0..79) 몸체 슬립각 [rad]
            - 사람: 원래 궤적에서 계산한 슬립각
    """
    pass


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


# ---------- 4. 파이프라인 편의 함수 ---------------------------------------------------


def slip_limit_stage(
    near_agents_current: npt.NDArray[np.float32],  # (Pnn, 4)
    near_future_raw: npt.NDArray[np.float32],  # (Pnn, 80, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
        veh_valid_mask: np.ndarray,  # (Pnn,)
        bic_valid_mask: np.ndarray,  # (Pnn,)
    cfg: SmootherConfig,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """[2단계] 슬립각 제한: 앵커 보정 → 순차 보정.

    Returns:
        near_current_future_a2: (Pnn, 81, 4)
        near_future_body_slip: (Pnn, 80)
    """
    pass


def yawrate_smooth_stage(
    near_current_future_a2: npt.NDArray[np.float32],  # (Pnn, 81, 4)
    near_current_future_dir: npt.NDArray[np.int8],  # (Pnn, 81)
    near_future_body_slip: npt.NDArray[np.float32],  # (Pnn, 80)
        veh_valid_mask: np.ndarray,  # (Pnn,)
        bic_valid_mask: np.ndarray,  # (Pnn,)
    ped_valid_mask: npt.NDArray[np.bool_],  # (Pnn,)
    cfg: SmootherConfig,
) -> npt.NDArray[np.float32]:
    """[3단계] 각속도 제약 + 스무딩: σ-aware 방향각, 저속 보정, TV 스무딩, 적분, 헤딩 복원.

    Returns:
        near_future_a3: (Pnn, 80, 4)
    """
    pass
