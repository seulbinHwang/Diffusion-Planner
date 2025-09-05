from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import to_rgba

Array = np.ndarray
WorldModelFeature = Dict[str, Array]
TokenTrajDict = Dict[str, Array]  # value: (future_len, 4) with [x, y, cos, sin]

# =============================================================================
# 옵션/스타일
# =============================================================================


@dataclass
class DrawingOptions:
    """렌더링 옵션 모음.

    Attributes
    ----------
    draw_ego_past : bool
        ego_agent_past(21, 11) 시퀀스 렌더링 여부.
    draw_neighbor_past : bool
        neighbor_agents_past(agent_num, 21, 11) 시퀀스 렌더링 여부.
    draw_ego_pred : bool
        ego_agent_next_11_dim(interpol_num, 11) 예측 궤적 렌더링 여부.
    draw_ego_future_gt : bool
        ego_future_gt_11_dim(80, 11) GT 미래 궤적 렌더링 여부.
    draw_lane_boundaries : bool
        차선 좌/우 경계(LANE) 렌더링 여부(실선 #2d3ea7).
    draw_lane_centerline : bool
        차선 센터라인(BASELINE_PATHS) 점선 렌더링 여부(신호색상 반영).
    draw_token_future_arrows : bool
        token_to_future_traj_wrt_ego의 각 시점 화살표(길이 고정 2 m) 렌더링 여부.
    draw_velocity_arrows_past_all : bool
        과거 시퀀스의 모든 타임스텝에서 속도 화살표를 그림(True) / 현재만 그림(False).
    draw_velocity_arrows_pred_all : bool
        예측 시퀀스 모든 스텝에서 속도 화살표 그림 여부.
    draw_velocity_arrows_future_all : bool
        GT 미래 시퀀스 모든 스텝에서 속도 화살표 그림 여부.

    arrow_length_m : float
        속도/방향 화살표의 고정 길이 [m]. 기본 2.0.
    heading_line_scale : float
        헤딩선 길이 비율(사각형 길이 * heading_line_scale).

    background_color : str
        축/그림 배경색.
    show_axis : bool
        좌표축 눈금/테두리 표시 여부.
    fig_size : Tuple[float, float]
        matplotlib figure 크기 (inch).
    dpi : int
        저장 해상도(dpi).
    margin_m : float
        자동 범위 계산 시 여백 [m].
    equal_aspect : bool
        축 비율 1:1 유지 여부.

    invalid_eps : float
        invalid 판정 시 0으로 간주할 허용오차(기본 0.0).
        - agent: 앞 8차원이 모두 |value| ≤ invalid_eps 이면 invalid
        - lanes: 앞 8차원이 모두 |value| ≤ invalid_eps 이면 invalid
        - token: 앞 4차원이 모두 |value| ≤ invalid_eps 이면 invalid
    """

    draw_ego_past: bool = True
    draw_neighbor_past: bool = True
    draw_ego_pred: bool = True
    draw_ego_future_gt: bool = True
    draw_lane_boundaries: bool = True
    draw_lane_centerline: bool = True
    draw_token_future_arrows: bool = True

    draw_velocity_arrows_past_all: bool = True
    draw_velocity_arrows_pred_all: bool = True
    draw_velocity_arrows_future_all: bool = True

    arrow_length_m: float = 2.0
    heading_line_scale: float = 0.5

    background_color: str = "#121212"
    show_axis: bool = False
    fig_size: Tuple[float, float] = (9.0, 9.0)
    dpi: int = 200
    margin_m: float = 5.0
    equal_aspect: bool = True

    invalid_eps: float = 0.0


# 스타일 사전
EGO_PAST_STYLE = {
    "fill_color": "#FFFFFF",
    "line_color": "#808080",
    "line_width": 2.0,
    "fill_alpha_current": 1.0,
}
EGO_PRED_STYLE = {
    "line_color": "#FFFFFF",
    "line_width": 2.0,
    "velocity_line_color": "#00C8C8",
    "velocity_line_alpha": 0.8,
    "velocity_line_width": 2.0,
}
EGO_FUTURE_GT_STYLE = {
    "line_color": "#FFFFFF",
    "line_width": 1.0,
    "velocity_line_color": "#00C8C8",
    "velocity_line_alpha": 0.8,
    "velocity_line_width": 2.0,
}
NEIGHBOR_STYLE = {
    "vehicles": {
        "fill_color": "#84E573",
        "fill_alpha": 0.5,
        "line_color": "#84E573",
        "line_width": 1.0,
        "velocity_line_color": "#84E573",
        "velocity_line_width": 1.0,
    },
    "pedestrians": {
        "fill_color": "#4D83E1",
        "fill_alpha": 0.5,
        "line_color": "#4D83E1",
        "line_width": 1.0,
        "velocity_line_color": "#4D83E1",
        "velocity_line_width": 1.0,
    },
    "bicycles": {
        "fill_color": "#FF4D4D",
        "fill_alpha": 0.5,
        "line_color": "#FF4D4D",
        "line_width": 1.0,
        "velocity_line_color": "#FF4D4D",
        "velocity_line_width": 1.0,
    },
}
LANE_BOUNDARY_COLOR = "#2d3ea7"
SIGNAL_COLORS = {
    0: "#00C853",  # green
    1: "#FFD600",  # yellow
    2: "#D50000",  # red
    3: "#B0BEC5",  # unknown
}
TOKEN_FUTURE_STYLE = {
    "line_color": "#FFFFFF",
    "line_width": 1.0,
}

# =============================================================================
# 유틸리티(도형/화살표/클래스/유효성/범위)
# =============================================================================


def oriented_box_corners(x: float, y: float, cos_yaw: float, sin_yaw: float,
                         length: float, width: float) -> Array:
    """중심 (x,y), 크기 (length,width), 방향 (cos_yaw,sin_yaw)인 사각형 꼭짓점(4,2)을 계산."""
    half_L = length * 0.5
    half_W = width * 0.5
    local_corners = np.array(
        [[+half_L, +half_W], [+half_L, -half_W], [-half_L, -half_W],
         [-half_L, +half_W]],
        dtype=np.float32,
    )
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return (local_corners @ R.T) + np.array([x, y], dtype=np.float32)


def add_polygon(
    ax: plt.Axes,
    corners_xy: Array,
    edge_color: str,
    line_width: float,
    fill_color: Optional[str] = None,
    fill_alpha: Optional[float] = None,
    zorder: int = 10,
) -> None:
    """폴리곤(사각형)을 Axes에 추가."""
    if fill_color is None or fill_alpha is None:
        face = "none"
    else:
        face = to_rgba(fill_color, fill_alpha)
    ax.add_patch(
        Polygon(
            corners_xy,
            closed=True,
            facecolor=face,
            edgecolor=edge_color,
            linewidth=line_width,
            zorder=zorder,
            joinstyle="miter",
        ))


def add_heading_line(ax: plt.Axes,
                     x: float,
                     y: float,
                     cos_yaw: float,
                     sin_yaw: float,
                     nominal_length: float,
                     color: str,
                     line_width: float,
                     zorder: int = 12) -> None:
    """사각형 중심에서 헤딩 방향으로 선을 그림."""
    hx = x + nominal_length * cos_yaw
    hy = y + nominal_length * sin_yaw
    ax.plot([x, hx], [y, hy],
            linestyle="-",
            linewidth=line_width,
            color=color,
            zorder=zorder)


def add_velocity_arrow(ax: plt.Axes,
                       x: float,
                       y: float,
                       vx: float,
                       vy: float,
                       length_m: float,
                       line_color: str,
                       line_width: float,
                       line_alpha: Optional[float] = None,
                       zorder: int = 15) -> None:
    """속도/방향 벡터로 '길이 고정' 화살표를 그림."""
    mag = float(np.hypot(vx, vy))
    if mag < 1e-6:
        return
    dx = length_m * (vx / mag)
    dy = length_m * (vy / mag)
    ax.add_patch(
        FancyArrowPatch(
            (x, y),
            (x + dx, y + dy),
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=line_width,
            color=line_color,
            alpha=line_alpha,
            zorder=zorder,
            shrinkA=0.0,
            shrinkB=0.0,
        ))


def infer_agent_class(one_hot: Array) -> str:
    """원-핫(3,) → 'vehicles' | 'pedestrians' | 'bicycles' 반환."""
    idx = int(np.argmax(one_hot))
    return ["vehicles", "pedestrians", "bicycles"][idx]


def is_valid_agent_row(row11: Array, eps: float) -> bool:
    """에이전트 1타임스텝(11,)이 **유효**하면 True.
    - 규칙: 앞 8차원([x,y,cos,sin,vx,vy,length,width]) 중 하나라도 |value| > eps
    """
    return bool(np.any(np.abs(row11[..., :8]) > eps))


def is_valid_lane_point(point12: Array, eps: float) -> bool:
    """차선 포인트(12,)가 **유효**하면 True.
    - 규칙: 앞 8차원 중 하나라도 |value| > eps
    """
    return bool(np.any(np.abs(point12[..., :8]) > eps))


def is_valid_token_row(row4: Array, eps: float) -> bool:
    """토큰 포즈 1타임스텝(4,)이 **유효**하면 True.
    - 규칙: 4차원([x,y,cos,sin]) 중 하나라도 |value| > eps
    """
    return bool(np.any(np.abs(row4[..., :4]) > eps))


def collect_valid_xy_for_bounds(
    world_model_feature: WorldModelFeature,
    options: DrawingOptions,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict] = None,
) -> Tuple[List[float], List[float]]:
    """모든 요소에서 valid (x,y)만 모아 bounds 계산에 사용."""
    eps = options.invalid_eps
    xs: List[float] = []
    ys: List[float] = []

    # lanes: 각 포인트가 valid일 때 center/left/right 좌표 반영
    lanes = world_model_feature.get("lanes")
    if lanes is not None and lanes.size > 0:
        valid_mask = np.any(np.abs(lanes[:, :, :8]) > eps,
                            axis=2)  # (lane_num, lane_len)
        centers = lanes[:, :, 0:2]
        lefts = centers + lanes[:, :, 4:6]
        rights = centers + lanes[:, :, 6:8]
        if np.any(valid_mask):
            c_valid = centers[valid_mask]
            l_valid = lefts[valid_mask]
            r_valid = rights[valid_mask]
            xs.extend(c_valid[:, 0].tolist())
            ys.extend(c_valid[:, 1].tolist())
            xs.extend(l_valid[:, 0].tolist())
            ys.extend(l_valid[:, 1].tolist())
            xs.extend(r_valid[:, 0].tolist())
            ys.extend(r_valid[:, 1].tolist())

    # ego past, ego pred, ego future
    for key in ("ego_agent_past", "ego_agent_next_11_dim",
                "ego_future_gt_11_dim", "ego_agent_future_11_dim"):
        A = world_model_feature.get(key)
        if A is None or A.size == 0:
            continue
        valid_mask = np.any(np.abs(A[:, :8]) > eps, axis=1)  # (N,)
        if np.any(valid_mask):
            xy = A[valid_mask, 0:2]
            xs.extend(xy[:, 0].tolist())
            ys.extend(xy[:, 1].tolist())

    # neighbor past
    neigh = world_model_feature.get("neighbor_agents_past")
    if neigh is not None and neigh.size > 0:
        valid_mask = np.any(np.abs(neigh[:, :, :8]) > eps,
                            axis=2)  # (agent_num, T)
        if np.any(valid_mask):
            xy = neigh[:, :, 0:2][valid_mask]
            xs.extend(xy[:, 0].tolist())
            ys.extend(xy[:, 1].tolist())

    # tokens
    if token_to_future_traj_wrt_ego:
        for _, arr in token_to_future_traj_wrt_ego.items():
            if arr is None or arr.size == 0:
                continue
            if arr.shape[-1] != 4:
                continue
            valid_mask = np.any(np.abs(arr[:, :4]) > eps,
                                axis=1)  # (future_len,)
            if np.any(valid_mask):
                xy = arr[valid_mask, 0:2]
                xs.extend(xy[:, 0].tolist())
                ys.extend(xy[:, 1].tolist())

    return xs, ys


# =============================================================================
# 레이어별 드로잉 함수 (invalid 스킵 포함)
# =============================================================================


def draw_lane_boundaries(ax: plt.Axes, lanes: Array,
                         options: DrawingOptions) -> None:
    """좌/우 차선 경계를 실선으로 그림(양 끝점 모두 valid일 때만 선분을 그림)."""
    if lanes is None or lanes.size == 0:
        return

    eps = options.invalid_eps
    for lane_i in lanes:  # (lane_len, 12)
        center = lane_i[:, 0:2]
        left_vec = lane_i[:, 4:6]
        right_vec = lane_i[:, 6:8]
        valid = np.any(np.abs(lane_i[:, :8]) > eps, axis=1)  # (lane_len,)

        if center.shape[0] < 2:
            continue

        # 구간별로 체크하여 valid한 이웃 점 사이만 그림
        for j in range(center.shape[0] - 1):
            if not (valid[j] and valid[j + 1]):
                continue
            l0 = center[j] + left_vec[j]
            l1 = center[j + 1] + left_vec[j + 1]
            r0 = center[j] + right_vec[j]
            r1 = center[j + 1] + right_vec[j + 1]
            ax.plot([l0[0], l1[0]], [l0[1], l1[1]],
                    color=LANE_BOUNDARY_COLOR,
                    linewidth=1.0,
                    zorder=1)
            ax.plot([r0[0], r1[0]], [r0[1], r1[1]],
                    color=LANE_BOUNDARY_COLOR,
                    linewidth=1.0,
                    zorder=1)


def draw_lane_centerlines(ax: plt.Axes, lanes: Array,
                          options: DrawingOptions) -> None:
    """센터라인을 점선으로 그림(양 끝점 모두 valid일 때만 선분을 그림, 색은 신호 반영)."""
    if lanes is None or lanes.size == 0:
        return

    eps = options.invalid_eps
    for lane_i in lanes:  # (lane_len, 12)
        center = lane_i[:, 0:2]
        signals = lane_i[:, 8:12]
        valid = np.any(np.abs(lane_i[:, :8]) > eps, axis=1)  # (lane_len,)

        if center.shape[0] < 2:
            continue

        state_idx = np.argmax(signals, axis=1)  # (lane_len,)
        for j in range(center.shape[0] - 1):
            if not (valid[j] and valid[j + 1]):
                continue
            c0, c1 = center[j], center[j + 1]
            color = SIGNAL_COLORS.get(int(state_idx[j]), "#B0BEC5")
            ax.plot([c0[0], c1[0]], [c0[1], c1[1]],
                    color=color,
                    linewidth=1.2,
                    linestyle=(0, (4, 4)),
                    zorder=2)


def draw_neighbor_past(ax: plt.Axes, neighbor_agents_past: Array,
                       options: DrawingOptions) -> None:
    """이웃 에이전트 과거 시퀀스를 클래스별 색상으로 그림. invalid 스텝은 스킵."""
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    eps = options.invalid_eps
    agent_num, time_len, feat_dim = neighbor_agents_past.shape
    if feat_dim != 11:
        raise ValueError("neighbor_agents_past의 마지막 차원은 11이어야 합니다.")
    current_t = time_len - 1

    for a in range(agent_num):
        track = neighbor_agents_past[a]  # (T, 11)
        for t in range(time_len):
            row = track[t]
            if not is_valid_agent_row(row, eps):
                continue

            x, y = float(row[0]), float(row[1])
            c, s = float(row[2]), float(row[3])
            vx, vy = float(row[4]), float(row[5])
            L, W = float(row[6]), float(row[7])
            cls = infer_agent_class(row[8:11])
            st = NEIGHBOR_STYLE[cls]

            fill_color = st["fill_color"] if t == current_t else None
            fill_alpha = st["fill_alpha"] if t == current_t else None

            corners = oriented_box_corners(x, y, c, s, L, W)
            add_polygon(ax,
                        corners,
                        edge_color=st["line_color"],
                        line_width=st["line_width"],
                        fill_color=fill_color,
                        fill_alpha=fill_alpha,
                        zorder=5 if t == current_t else 4)
            add_heading_line(ax,
                             x,
                             y,
                             c,
                             s,
                             nominal_length=L * options.heading_line_scale,
                             color=st["line_color"],
                             line_width=st["line_width"],
                             zorder=6 if t == current_t else 4)
            if options.draw_velocity_arrows_past_all or t == current_t:
                add_velocity_arrow(ax,
                                   x,
                                   y,
                                   vx,
                                   vy,
                                   length_m=options.arrow_length_m,
                                   line_color=st["velocity_line_color"],
                                   line_width=st["velocity_line_width"],
                                   zorder=7 if t == current_t else 4)


def draw_ego_past(ax: plt.Axes, ego_agent_past: Array,
                  options: DrawingOptions) -> None:
    """이고 차량 과거 시퀀스를 그림(현재 프레임만 채움). invalid 스텝은 스킵."""
    if ego_agent_past is None or ego_agent_past.size == 0:
        return
    eps = options.invalid_eps
    time_len, feat_dim = ego_agent_past.shape
    if feat_dim != 11:
        raise ValueError("ego_agent_past의 마지막 차원은 11이어야 합니다.")
    current_t = time_len - 1

    for t in range(time_len):
        row = ego_agent_past[t]
        if not is_valid_agent_row(row, eps):
            continue

        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])
        vx, vy = float(row[4]), float(row[5])
        L, W = float(row[6]), float(row[7])

        fill_color = EGO_PAST_STYLE["fill_color"] if t == current_t else None
        fill_alpha = EGO_PAST_STYLE[
            "fill_alpha_current"] if t == current_t else None

        corners = oriented_box_corners(x, y, c, s, L, W)
        add_polygon(ax,
                    corners,
                    edge_color=EGO_PAST_STYLE["line_color"],
                    line_width=EGO_PAST_STYLE["line_width"],
                    fill_color=fill_color,
                    fill_alpha=fill_alpha,
                    zorder=20 if t == current_t else 9)
        add_heading_line(ax,
                         x,
                         y,
                         c,
                         s,
                         nominal_length=L * options.heading_line_scale,
                         color=EGO_PAST_STYLE["line_color"],
                         line_width=EGO_PAST_STYLE["line_width"],
                         zorder=21 if t == current_t else 9)
        if options.draw_velocity_arrows_past_all or t == current_t:
            add_velocity_arrow(ax,
                               x,
                               y,
                               vx,
                               vy,
                               length_m=options.arrow_length_m,
                               line_color=EGO_PAST_STYLE["line_color"],
                               line_width=EGO_PAST_STYLE["line_width"],
                               zorder=22 if t == current_t else 9)


def draw_ego_predicted(ax: plt.Axes, ego_agent_next_11_dim: Array,
                       options: DrawingOptions) -> None:
    """이고 차량 **예측** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if ego_agent_next_11_dim is None or ego_agent_next_11_dim.size == 0:
        return
    eps = options.invalid_eps
    interpol_num, feat_dim = ego_agent_next_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_agent_next_11_dim의 마지막 차원은 11이어야 합니다.")

    for t in range(interpol_num):
        row = ego_agent_next_11_dim[t]
        if not is_valid_agent_row(row, eps):
            continue

        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])
        vx, vy = float(row[4]), float(row[5])
        L, W = float(row[6]), float(row[7])

        corners = oriented_box_corners(x, y, c, s, L, W)
        add_polygon(ax,
                    corners,
                    edge_color=EGO_PRED_STYLE["line_color"],
                    line_width=EGO_PRED_STYLE["line_width"],
                    fill_color=None,
                    fill_alpha=None,
                    zorder=25)
        add_heading_line(ax,
                         x,
                         y,
                         c,
                         s,
                         nominal_length=L * options.heading_line_scale,
                         color=EGO_PRED_STYLE["line_color"],
                         line_width=EGO_PRED_STYLE["line_width"],
                         zorder=26)
        if options.draw_velocity_arrows_pred_all:
            add_velocity_arrow(ax,
                               x,
                               y,
                               vx,
                               vy,
                               length_m=options.arrow_length_m,
                               line_color=EGO_PRED_STYLE["velocity_line_color"],
                               line_width=EGO_PRED_STYLE["velocity_line_width"],
                               line_alpha=EGO_PRED_STYLE["velocity_line_alpha"],
                               zorder=27)


def draw_ego_future_gt(ax: plt.Axes, ego_future_gt_11_dim: Array,
                       options: DrawingOptions) -> None:
    """이고 차량 **GT 미래** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if ego_future_gt_11_dim is None or ego_future_gt_11_dim.size == 0:
        return
    eps = options.invalid_eps
    future_len, feat_dim = ego_future_gt_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_future_gt_11_dim의 마지막 차원은 11이어야 합니다.")

    for t in range(future_len):
        row = ego_future_gt_11_dim[t]
        if not is_valid_agent_row(row, eps):
            continue

        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])
        vx, vy = float(row[4]), float(row[5])
        L, W = float(row[6]), float(row[7])

        corners = oriented_box_corners(x, y, c, s, L, W)
        add_polygon(ax,
                    corners,
                    edge_color=EGO_FUTURE_GT_STYLE["line_color"],
                    line_width=EGO_FUTURE_GT_STYLE["line_width"],
                    fill_color=None,
                    fill_alpha=None,
                    zorder=24)
        add_heading_line(ax,
                         x,
                         y,
                         c,
                         s,
                         nominal_length=L * options.heading_line_scale,
                         color=EGO_FUTURE_GT_STYLE["line_color"],
                         line_width=EGO_FUTURE_GT_STYLE["line_width"],
                         zorder=24)
        if options.draw_velocity_arrows_future_all:
            add_velocity_arrow(
                ax,
                x,
                y,
                vx,
                vy,
                length_m=options.arrow_length_m,
                line_color=EGO_FUTURE_GT_STYLE["velocity_line_color"],
                line_width=EGO_FUTURE_GT_STYLE["velocity_line_width"],
                line_alpha=EGO_FUTURE_GT_STYLE["velocity_line_alpha"],
                zorder=24)


def draw_token_future_arrows(
    ax: plt.Axes,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict],
    options: DrawingOptions,
) -> None:
    """토큰 기준 미래 포즈를 '개별 화살표'로 그림.

    Parameters
    ----------
    token_to_future_traj_wrt_ego : Dict[str, np.ndarray] | None
        각 value: shape (future_len=80, 4) = [x, y, cos(yaw), sin(yaw)]
        - invalid 규칙: 4값 모두 0(±eps) → 스킵
        - valid 시: (x,y)에서 (cos,sin) 방향으로 길이 options.arrow_length_m(기본 2 m) 화살표
    options : DrawingOptions
    """
    if not token_to_future_traj_wrt_ego:
        return

    eps = options.invalid_eps
    for _, arr in token_to_future_traj_wrt_ego.items():
        if arr is None or arr.size == 0:
            continue
        if arr.ndim != 2 or arr.shape[1] != 4:
            raise ValueError(
                "token_to_future_traj_wrt_ego의 각 value는 (future_len, 4)이어야 합니다."
            )

        future_len = arr.shape[0]
        for t in range(future_len):
            row = arr[t]  # (4,) = [x, y, cos, sin]
            if not is_valid_token_row(row, eps):
                continue

            x, y = float(row[0]), float(row[1])
            c, s = float(row[2]), float(row[3])

            # 방향 벡터 (c, s)를 정규화하여 고정 길이 화살표
            add_velocity_arrow(
                ax,
                x,
                y,
                c,
                s,
                length_m=options.arrow_length_m,
                line_color=TOKEN_FUTURE_STYLE["line_color"],
                line_width=TOKEN_FUTURE_STYLE["line_width"],
                zorder=23  # 에이전트 윤곽(24~27) 바로 아래/사이에 위치하도록
            )


# =============================================================================
# Figure/Axis & 범위/저장
# =============================================================================


def create_figure_and_axes(
        options: DrawingOptions) -> Tuple[plt.Figure, plt.Axes]:
    """Figure/Axes 생성 및 배경색 설정."""
    fig = plt.figure(figsize=options.fig_size, dpi=options.dpi)
    ax = fig.add_subplot(111)
    ax.set_facecolor(options.background_color)
    fig.patch.set_facecolor(options.background_color)
    return fig, ax


def apply_axes_style(ax: plt.Axes, options: DrawingOptions) -> None:
    """좌표축 스타일 적용."""
    if options.equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    if not options.show_axis:
        ax.axis("off")


def compute_auto_bounds(
    world_model_feature: WorldModelFeature,
    options: DrawingOptions,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict] = None,
) -> Tuple[float, float, float, float]:
    """valid (x,y)만 모아 자동으로 축 범위를 산출."""
    xs, ys = collect_valid_xy_for_bounds(world_model_feature, options,
                                         token_to_future_traj_wrt_ego)
    if not xs or not ys:
        return -10.0, 10.0, -10.0, 10.0
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin, xmax, ymin, ymax


def set_axes_limits_with_margin(ax: plt.Axes, bounds: Tuple[float, float, float,
                                                            float],
                                margin_m: float) -> None:
    """산출된 (xmin,xmax,ymin,ymax)에 여백을 주어 축 영역을 설정."""
    xmin, xmax, ymin, ymax = bounds
    xspan = max(xmax - xmin, 1e-3)
    yspan = max(ymax - ymin, 1e-3)
    ax.set_xlim(xmin - margin_m, xmin + xspan + margin_m)
    ax.set_ylim(ymin - margin_m, ymin + yspan + margin_m)


def save_figure_to_png(fig: plt.Figure, save_path: str) -> None:
    """Figure를 PNG로 저장하고 Figure를 닫음."""
    plt.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# =============================================================================
# 메인(오케스트레이터)
# =============================================================================


def draw_world_model_to_png(
    world_model_feature: WorldModelFeature,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict],
    save_path: str,
    options: Optional[DrawingOptions] = None,
) -> None:
    """주어진 world_model_feature(+토큰 미래 포즈)를 그림으로 렌더링하고 PNG로 저장.

    Parameters
    ----------
    world_model_feature : Dict[str, np.ndarray]
        - 'ego_agent_past' : (time_len=21, 11)
        - 'neighbor_agents_past' : (agent_num, 21, 11)
        - 'ego_agent_next_11_dim' : (interpol_num, 11)
        - 'ego_future_gt_11_dim' : (future_len=80, 11)
        - 'lanes' : (lane_num, lane_len, 12)
          · 0-1: centerline (x,y)
          · 2-3: centerline diff (dx,dy)
          · 4-5: left boundary vector (dx,dy)
          · 6-7: right boundary vector (dx,dy)
          · 8-11: signal one-hot [green, yellow, red, unknown]
    token_to_future_traj_wrt_ego : Dict[str, np.ndarray], optional
        각 value: (future_len=80, 4) = [x, y, cos, sin].
        - invalid: 4값 모두 0(±eps) → 스킵
        - valid: (x,y)에서 (cos,sin) 방향으로 길이 2 m(=options.arrow_length_m) 화살표
    save_path : str
        저장할 PNG 경로.
    options : Optional[DrawingOptions]
        렌더링 옵션. None이면 기본 옵션 사용.

    Notes
    -----
    - invalid 스텝/포인트는 그리지 않음
      · agent: [x,y,cos,sin,vx,vy,length,width] 8개가 모두 0(±eps) → 스킵
      · lanes: 앞 8차원이 모두 0(±eps) → 스킵, 선분은 양 끝점 valid일 때만 그림
      · token: [x,y,cos,sin] 4개가 모두 0(±eps) → 스킵
    - 좌표계는 "현재 이고 뒷축 좌표계(＋x=ego heading)" 가정.
    """
    draw_option = options or DrawingOptions()

    # 1) Figure/Axes
    fig, ax = create_figure_and_axes(draw_option)

    # 2) 바닥 레이어(차선)
    lanes = world_model_feature.get("lanes")
    if draw_option.draw_lane_boundaries:
        draw_lane_boundaries(ax, lanes, draw_option)
    if draw_option.draw_lane_centerline:
        draw_lane_centerlines(ax, lanes, draw_option)

    # 3) 토큰 미래 화살표(개별) - 차선 위에, 에이전트 윤곽과 겹치지 않게 중간 zorder
    if draw_option.draw_token_future_arrows:
        draw_token_future_arrows(ax, token_to_future_traj_wrt_ego, draw_option)

    # 4) 에이전트(과거/예측/GT)
    if draw_option.draw_neighbor_past:
        draw_neighbor_past(ax, world_model_feature.get("neighbor_agents_past"),
                           draw_option)
    if draw_option.draw_ego_past:
        draw_ego_past(ax, world_model_feature.get("ego_agent_past"),
                      draw_option)
    if draw_option.draw_ego_pred:
        draw_ego_predicted(ax, world_model_feature.get("ego_agent_next_11_dim"),
                           draw_option)
    if draw_option.draw_ego_future_gt:
        data_ = world_model_feature.get("ego_future_gt_11_dim", None)
        if data_ is None:
            data_ = world_model_feature.get("ego_agent_future_11_dim", None)
        draw_ego_future_gt(ax, data_, draw_option)

    # 5) 축 범위/스타일
    bounds = compute_auto_bounds(world_model_feature, draw_option,
                                 token_to_future_traj_wrt_ego)
    set_axes_limits_with_margin(ax, bounds, draw_option.margin_m)
    apply_axes_style(ax, draw_option)

    # 6) 저장
    save_figure_to_png(fig, save_path)


if __name__ == "__main__":
    # world_model_feature 예시(실데이터로 교체)
    world_model_feature = {
        "ego_agent_past": np.zeros((21, 11), dtype=np.float32),
        "neighbor_agents_past": np.zeros((5, 21, 11), dtype=np.float32),
        "ego_agent_next_11_dim": np.zeros((30, 11), dtype=np.float32),
        "ego_future_gt_11_dim": np.zeros((80, 11), dtype=np.float32),
        "lanes": np.zeros((70, 50, 12), dtype=np.float32),
    }

    # 토큰 미래 포즈 예시: key는 무시되고 value만 사용됨
    token_to_future_traj_wrt_ego = {
        "token_a": np.array([[1.0, 2.0, 0.0, 1.0]] * 80, dtype=np.float32),
        # 위 방향 화살표 80개
        "token_b": np.array([[5.0, -3.0, 1.0, 0.0]] * 80, dtype=np.float32),
        # +x 방향 화살표 80개
    }

    options = DrawingOptions(
        draw_lane_boundaries=True,
        draw_lane_centerline=True,
        draw_token_future_arrows=True,  # 토큰 화살표 on
        draw_ego_past=True,
        draw_neighbor_past=True,
        draw_ego_pred=True,
        draw_ego_future_gt=True,
        background_color="#121212",
        show_axis=False,
        invalid_eps=0.0,
    )

    draw_world_model_to_png(
        world_model_feature,
        save_path="scene_with_tokens.png",
        options=options,
        token_to_future_traj_wrt_ego=token_to_future_traj_wrt_ego,
    )
