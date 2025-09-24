from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import to_rgba

Array = np.ndarray
WorldModelFeature = Dict[str, Array]
TokenTrajDict = Dict[str, Array]  # value: (future_len, 4) with [x, y, cos, sin]


class DrawInfos:

    def __init__(self):
        """
        딥러닝 input으로 쓰인 값들
        이 안에, "neighbor_track_token" 가 있고, 이게 neighbor_token_dist_order: List[Optional[str]]
        """
        self.model_input_key_to_unnorm_value: Dict[str, np.ndarray] = {}
        """
        딥러닝 output 값 그대로
        """
        self.diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] = {}  # (T, 4)
        """
        history Agent 만든걸 -> (History_len, 11) numpy로 변환한 것들
        npc 미래 궤적 보정 input으로 쓰이는걸 그려보기 위해 저장
        """
        self.diff_token_to_np_history_wrt_ego: Dict[str, np.ndarray] = {
        }  # (History_len, 11)
        """
        interpolation으로, 생성된 미래 궤적에 속도를 추가한 것
        """
        self.diff_token_to_interp_np_traj_wrt_ego: Dict[str, np.ndarray] = {
        }  # (Future_len, 11)
        """
        interpolation 궤적 생성 후, next_iteration 시점 waypoint를 array로 변환한 것
        """
        self.diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] = {}  # (11,)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_input_key_to_unnorm_value": self.model_input_key_to_unnorm_value,
            "diff_token_to_np_gen_traj_wrt_ego": self.diff_token_to_np_gen_traj_wrt_ego,
            "diff_token_to_np_history_wrt_ego": self.diff_token_to_np_history_wrt_ego,
            "diff_token_to_interp_np_traj_wrt_ego": self.diff_token_to_interp_np_traj_wrt_ego,
            "diff_token_to_next_wp_wrt_ego": self.diff_token_to_next_wp_wrt_ego,
        }

def is_valid_future_row_xyyaw(row3: Array, eps: float) -> bool:
    """미래 포인트(3,)=[x,y,yaw]가 **유효**하면 True.
    - 규칙: |x|>eps 또는 |y|>eps 이면 유효로 간주( yaw=0 이어도 상관 없음 )
    """
    if row3.shape[-1] != 3:
        raise ValueError("neighbor_agents_future의 마지막 차원은 3이어야 합니다.")
    return bool((abs(float(row3[0])) > eps) or (abs(float(row3[1])) > eps))


def draw_neighbor_future_points(ax: plt.Axes, neighbor_agents_future: Array,
                                options: DrawingOptions) -> None:
    """neighbor_agents_future (agent_num, future_len, 3=[x,y,yaw])를
    흰색 'x' 마커로 그리고, 각 에이전트의 첫 점 근처에 인덱스(0..agent_num-1)를 흰색으로 표기.

    규칙:
      - invalid: |x|<=eps and |y|<=eps → 스킵
      - 마커: 흰색 'x', 선 없음
      - 라벨: 첫 점이 유효할 때만 표시
    """
    if neighbor_agents_future is None or neighbor_agents_future.size == 0:
        return
    if neighbor_agents_future.ndim != 3 or neighbor_agents_future.shape[-1] != 3:
        raise ValueError(
            "neighbor_agents_future는 (agent_num, future_len, 3) 이어야 합니다.")

    eps = options.invalid_eps
    agent_num, future_len, _ = neighbor_agents_future.shape
    ms = options.neighbor_future_marker_size
    text_d = options.agent_index_offset_m
    text_color =  options.future_agent_index_color

    for a in range(agent_num):
        traj = neighbor_agents_future[a]  # (future_len, 3)
        # 모든 유효 포인트를 x마커로 그리기
        for t in range(future_len):
            row = traj[t]
            if not is_valid_future_row_xyyaw(row, eps):
                continue
            x, y = float(row[0]), float(row[1])
            ax.plot(x,
                    y,
                    marker='x',
                    markersize=ms,
                    linestyle='None',
                    color="#FFFFFF",
                    zorder=26)

        # 첫 점 라벨(유효할 때만)
        first = traj[0]
        if is_valid_future_row_xyyaw(first, eps):
            fx, fy = float(first[0]), float(first[1])
            ax.text(fx + text_d,
                    fy + text_d,
                    str(a),
                    color=text_color,
                    fontsize=options.agent_index_fontsize + 5,
                    ha='left',
                    va='bottom',
                    zorder=30)


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

    draw_ego_past: bool = True # check
    draw_neighbor_past: bool = True # check
    draw_ego_pred: bool = False # check
    draw_ego_future_gt: bool = False # check
    draw_lane_boundaries: bool = True # check
    draw_lane_centerline: bool = False # check
    draw_token_future_arrows: bool = True
    draw_neighbor_agents_future: bool = False
    draw_token_refined: bool = False

    draw_velocity_arrows_past_all: bool = False # check
    draw_velocity_arrows_pred_all: bool = False # check
    draw_velocity_arrows_future_all: bool = False # check

    arrow_length_m: float = 5.0
    heading_line_scale: float = 0.5

    background_color: str = "#121212"
    show_axis: bool = False
    fig_size: Tuple[float, float] = (18.0, 18.0)
    dpi: int = 400
    margin_m: float = 5.0
    equal_aspect: bool = True

    invalid_eps: float = 0.0

    agent_index_fontsize: int = 5  # 에이전트 번호 텍스트 폰트 크기
    agent_index_offset_m: float = 0.5  # 번호 텍스트를 포인트 옆으로 얼마나 띄울지(미터)
    past_agent_index_color: str = "#00C8C8"  # 번호 텍스트 색 # 청록색
    future_agent_index_color: str = "#808080" # 번호 텍스트 색 # 회색
    route_agent_index_color: str = "#00C8C8"  # 번호 텍스트 색 # 청록색
    neighbor_future_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기


    # ── [추가] 토큰 미래 궤적 드로잉 모드 및 스타일 ─────────────────────
    token_future_draw_mode: str = "point"  # 'arrow' 또는 'point'
    token_future_point_marker: str = "o"
    token_future_point_marker_size: float = 0.8
    token_future_arrow_length_m: float = 1.0

    # 🔽 [추가] 토큰 시작 인덱스 텍스트 표기용 옵션
    token_index_color: str = "#00C8C8"     # 청록색
    token_index_offset_m: float = 0.3      # y축으로 살짝 아래(미터 단위)
    token_index_fontsize: int = 5          # 기본은 에이전트 번호와 동일 크기
    # 🔽 새 옵션
    max_agents_to_draw: Optional[int] = None

# 스타일 사전
EGO_PAST_STYLE = {
    "fill_color": "#FFFFFF",  # 흰색
    "line_color": "#808080",  # 회색
    "line_width": 0.4,
    "fill_alpha_current": 1.0,
}
EGO_PRED_STYLE = {
    "line_color": "#00C8C8",           # 청록색
    "line_width": 0.4,
    "velocity_line_color": "#00C8C8",  # 청록색(시안)
    "velocity_line_alpha": 0.8,
    "velocity_line_width": 0.4,
}
EGO_FUTURE_GT_STYLE = {
    "line_color": "#FFFFFF",  # 흰색
    "line_width": 0.2,
    "velocity_line_color": "#00C8C8",  # 청록색(시안)
    "velocity_line_alpha": 0.8,
    "velocity_line_width": 0.4,
}
REFINED_FUTURE_STYLE = {
    "line_color": "#FF4D4D",           #  빨간색(밝은 빨강)
    "line_width": 0.2,
    "velocity_line_color": "#FF4D4D",  #  빨간색(밝은 빨강)
    "velocity_line_alpha": 0.8,
    "velocity_line_width": 0.4,
}
NEIGHBOR_STYLE = {
    "vehicles": {
        "fill_color": "#84E573",           # 연두색(라임 그린)
        "fill_alpha": 1.0,
        "line_color": "#84E573",           # 연두색(라임 그린)
        "line_width": 0.2,
        "velocity_line_color": "#84E573",  # 연두색(라임 그린)
        "velocity_line_width": 0.2,
    },
    "pedestrians": {
        "fill_color": "#4D83E1",           # 파란색(밝은 파랑)
        "fill_alpha": 1.0,
        "line_color": "#4D83E1",           # 파란색(밝은 파랑)
        "line_width": 0.2,
        "velocity_line_color": "#4D83E1",  # 파란색(밝은 파랑)
        "velocity_line_width": 0.2,
    },
    "bicycles": {
        "fill_color": "#FF4D4D",           # 빨간색(밝은 빨강)
        "fill_alpha": 1.0,
        "line_color": "#FF4D4D",           # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": "#FF4D4D",  # 빨간색(밝은 빨강)
        "velocity_line_width": 0.2,
    },
}
LANE_BOUNDARY_COLOR = "#2d3ea7"  # 남색(인디고 계열)
SIGNAL_COLORS = {
    0: "#00C853",  # 녹색(신호등 초록)
    1: "#FFD600",  # 노란색(신호등 노랑)
    2: "#D50000",  # 진한 빨간색(신호등 빨강)
    3: "#B0BEC5",  # 회색(청회색)
}
TOKEN_FUTURE_STYLE = {
    "line_color": "#FFFFFF",  # 흰색
    "line_width": 0.2,
    "index_color": "#FFFFFF",     # 흰색
}
TOKEN_REFINED_STYLE = {
    "line_color": "#FF4D4D",  # 흰색
    "line_width": 0.2,
}

NEW_WAYPOINT_STYLE = {
    "line_color": "#FFA500",  # 주황색
    "line_width": 0.2,
}


# =============================================================================
# 유틸리티(도형/화살표/클래스/유효성/범위)
# =============================================================================
from typing import Dict, Optional, Tuple

def _clip_agents_first_k_from_wmf(
    world_model_feature: Dict[str, Array],
    max_agents_to_draw: Optional[int],
) -> Tuple[Optional[Array], Optional[Array], Optional[Array], Optional[int]]:
    """world_model_feature에서 에이전트 축(0축)을 공유하는 3개 배열을 동일한 K로 슬라이스.
    대상 키:
      - 'neighbor_agents_past'   : (A, T, 11)
      - 'neighbor_agents_future' : (A, Tf, 3)
      - 'agent_route_lane_order' : (A, L)

    Args:
        world_model_feature: 입력 피쳐 dict.
        max_agents_to_draw: None이면 원본 그대로. 정수면 앞쪽 K개로 슬라이스.

    Returns:
        (neighbor_past_K, neighbor_future_K, route_order_K, K or None)
        - 각 요소는 해당 키가 없으면 None.
        - K는 실제로 적용된 개수(None이면 제한 없음).
    """
    if max_agents_to_draw is None:
        # 제한 없음: 원본 그대로 반환
        return (
            world_model_feature.get("neighbor_agents_past", None),
            world_model_feature.get("neighbor_agents_future", None),
            world_model_feature.get("agent_route_lane_order", None),
            None,
        )

    # 각 배열의 길이(A) 수집
    lengths = []
    for key in ("neighbor_agents_past", "neighbor_agents_future", "agent_route_lane_order"):
        arr = world_model_feature.get(key, None)
        if arr is not None and hasattr(arr, "shape") and len(arr.shape) >= 1:
            lengths.append(arr.shape[0])

    if not lengths:
        # 슬라이스할 것이 없음
        return (
            world_model_feature.get("neighbor_agents_past", None),
            world_model_feature.get("neighbor_agents_future", None),
            world_model_feature.get("agent_route_lane_order", None),
            None,
        )

    # 모든 배열에 공통으로 적용될 K 결정
    K = max(0, min(int(max_agents_to_draw), min(lengths)))

    def _clip(arr: Optional[Array]) -> Optional[Array]:
        if arr is None:
            return None
        if arr.shape[0] <= K:
            return arr
        return arr[:K, ...]

    neighbor_past_K   = _clip(world_model_feature.get("neighbor_agents_past", None))
    neighbor_future_K = _clip(world_model_feature.get("neighbor_agents_future", None))
    route_order_K     = _clip(world_model_feature.get("agent_route_lane_order", None))

    return neighbor_past_K, neighbor_future_K, route_order_K, K


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


# [Add]
def collect_valid_xy_for_bounds(
    world_model_feature: WorldModelFeature,
    options: DrawingOptions,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict] = None,
    token_to_refined_traj_wrt_ego: Optional[Dict[str, np.ndarray]] = None,
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

    # neighbor future points: (agent_num, future_len, 3)  -> (x,y)만 사용
    neigh_fut = world_model_feature.get("neighbor_agents_future")
    if neigh_fut is not None and neigh_fut.size > 0:
        if neigh_fut.ndim != 3 or neigh_fut.shape[-1] != 3:
            raise ValueError(
                "neighbor_agents_future는 (agent_num, future_len, 3) 이어야 합니다.")
        # 유효 마스크: |x|>eps or |y|>eps
        valid_mask = (np.abs(neigh_fut[..., 0]) > eps) | (np.abs(
            neigh_fut[..., 1]) > eps)  # (A, T)
        if np.any(valid_mask):
            xy = neigh_fut[..., :2][valid_mask]  # (K, 2)
            xs.extend(xy[:, 0].tolist())
            ys.extend(xy[:, 1].tolist())

    # [Add] refined 토큰(각 row 11,)에서도 valid (x, y) 수집
    if token_to_refined_traj_wrt_ego:
        for _, arr in token_to_refined_traj_wrt_ego.items():
            if arr is None or arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 11:
                continue  # 형식 불일치 시 스킵
            # agent와 동일 규칙: 앞 8차원 중 하나라도 |value| > eps → valid
            valid_mask = np.any(np.abs(arr[:, :8]) > eps, axis=1)  # (N,)
            if np.any(valid_mask):
                xy = arr[valid_mask, 0:2]  # (K, 2)
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


def draw_lane_centerlines(
    ax: plt.Axes,
    lanes: Array,
    options: DrawingOptions,
    agent_route_lane_order: Optional[Array] = None,
) -> None:
    """센터라인을 점선으로 그리거나, agent_route_lane_order가 주어지면 에이전트-차선 매핑을 텍스트로 표기한다.

    동작 모드
    ----------
    1) agent_route_lane_order is None:
        - 기존 로직 유지: 차선 센터라인을 **점선**으로 그림.
        - 양 끝점이 모두 유효한 구간만 선분을 그림.
        - 색은 lane의 signal(0~3: green/yellow/red/unknown)에 따라 사용.

    2) agent_route_lane_order is not None:
        - 센터라인 **선을 그리지 않음**.
        - 각 차선 j의 모든 유효 포인트 위치에 대해,
          agent_route_lane_order[:, j] != -1 인 모든 에이전트 i에 대해
          텍스트 **"i:rank"** 를 그 위치에 표기.
          (rank = agent_route_lane_order[i, j])
        - 동일 위치에 여러 텍스트가 겹치지 않도록, 세로로 약간(offset) 띄워서 적층.

    Args
    ----
    ax : plt.Axes
        Matplotlib 축.
    lanes : np.ndarray
        shape = (lane_num, lane_len, 12)
        · 0-1: centerline (x,y)
        · 2-3: centerline diff (dx,dy)
        · 4-5: left boundary vector (dx,dy)
        · 6-7: right boundary vector (dx,dy)
        · 8-11: signal one-hot [green, yellow, red, unknown]
    options : DrawingOptions
        그리기 옵션(색/두께/간격 등).
    agent_route_lane_order : Optional[np.ndarray]
        shape = (agent_num, lane_num), 각 [i, j] = 해당 에이전트 i에게서
        차선 j의 '가까운 순서 랭크(0,1,2,...)'; 경로에 없으면 -1.
    """
    if lanes is None or lanes.size == 0:
        return

    eps = options.invalid_eps
    lane_num = lanes.shape[0]

    # ────────────── (B) 텍스트 표기 모드 ──────────────
    if agent_route_lane_order is not None:
        if (agent_route_lane_order.ndim != 2 or
                agent_route_lane_order.shape[1] != lane_num):
            raise ValueError(
                "agent_route_lane_order의 shape는 (agent_num, lane_num) 이어야 하며 "
                f"lane_num({lane_num})과 두 번째 축이 같아야 합니다. "
                f"got {agent_route_lane_order.shape}"
            )

        base_fs = max(1, options.agent_index_fontsize - 1)  # 조금 작게
        vstep = 0.15 * 2.  # 같은 위치에 여러 개 쌓을 때 세로 간격(미터)

        # 각 차선 j 순회
        for j in range(lane_num):
            lane_j = lanes[j]                 # (lane_len, 12)
            center = lane_j[:, 0:2]           # (lane_len, 2)
            valid = np.any(np.abs(lane_j[:, :8]) > eps, axis=1)  # (lane_len,)

            # 이 차선을 자신의 경로에 포함하는 모든 agent i와 그 rank
            ranks_j: Array = agent_route_lane_order[:, j]        # (agent_num,)
            agent_idxs: Array = np.nonzero(ranks_j >= 0)[0]      # (K,)
            if agent_idxs.size == 0:
                continue

            # 유효 포인트마다 텍스트 찍기
            # (여러 agent가 있으면 위로 살짝씩 띄워서 겹침 완화)
            for p_idx in range(center.shape[0]):
                if not valid[p_idx]:
                    continue
                x, y = float(center[p_idx, 0]), float(center[p_idx, 1])

                for k, i in enumerate(agent_idxs):
                    rank_ij = int(ranks_j[int(i)])
                    label = f"{int(i)}--{rank_ij}"  # "에이전트인덱스:해당차선랭크"
                    label = f"{int(i)}"  # "에이전트인덱스"
                    ax.text(
                        x,
                        y + vstep * k,          # 위로 살짝씩 쌓기
                        label,
                        color=options.route_agent_index_color,
                        fontsize=base_fs,
                        ha="center",
                        va="bottom",
                        zorder=3,
                    )
        return  # 텍스트 모드에서는 선을 그리지 않음

    # ────────────── (A) 기존 점선 센터라인 모드 ──────────────
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
            ax.plot(
                [c0[0], c1[0]], [c0[1], c1[1]],
                color=color,
                linewidth=1.2,
                linestyle=(0, (4, 4)),
                zorder=2
            )


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
            W, L = float(row[6]), float(row[7])
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
            if options.draw_velocity_arrows_past_all:
                if t % 2 == 0:
                    offset = 2
                else:
                    offset = 1
                # [추가] 속도 크기 텍스트(km/h) - 모든 과거 지점
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                ax.text(
                    x,  # 점 위쪽에 표기
                    y + options.agent_index_offset_m * offset,
                    f"{speed_kmh:.1f}",
                    color=options.past_agent_index_color,
                    fontsize=2, #options.agent_index_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=30,
                    clip_on=True,  # tight 저장 시 bbox 폭주 방지
                )

def annotate_neighbor_indices_for_past(ax: plt.Axes,
neighbor_track_token: List[Optional[str]],
                                       neighbor_agents_past: Array,
                                       options: DrawingOptions) -> None:
    """neighbor_agents_past (agent_num, T=21, 11)의 '현재 상태'(마지막 스텝) 근처에
    에이전트 인덱스(0..agent_num-1)를 흰색 텍스트로 표기.
    - invalid 스텝은 스킵
    """
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[-1] != 11:
        raise ValueError(
            "neighbor_agents_past의 shape은 (agent_num, 21, 11) 이어야 합니다.")

    eps = options.invalid_eps
    agent_num, time_len, feat_dim = neighbor_agents_past.shape
    current_t = time_len - 1
    text_d = options.agent_index_offset_m
    text_color = options.past_agent_index_color

    for a in range(agent_num):
        track_token = neighbor_track_token[a]
        row = neighbor_agents_past[a, current_t]  # (11,)
        if not is_valid_agent_row(row, eps):
            assert track_token is None
            continue
        x, y = float(row[0]), float(row[1])
        # ax.text(x + text_d,
        #         y + text_d,
        #         str(track_token)[:5],
        #         color=text_color,
        #         fontsize=options.agent_index_fontsize,
        #         ha='left',
        #         va='bottom',
        #         zorder=30)


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
        W, L = float(row[6]), float(row[7])

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
        W, L = float(row[6]), float(row[7])

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
        W, L = float(row[6]), float(row[7])

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

# [Add]
def draw_token_refined_trajectories(
    ax: plt.Axes,
    token_to_refined_traj_wrt_ego: Optional[Dict[str, np.ndarray]],
    options: DrawingOptions,
    token_to_new_waypoint_array: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """토큰별 refined 궤적(연속 다스텝)과 **신규 waypoint(단일 11차원)**를 함께 그린다.

    - refined: REFINED_FUTURE_STYLE(빨강)로 연속 박스 + 첫 유효 포인트에 idx(빨강, 아래쪽 오프셋)
    - new waypoint: NEW_WAYPOINT_STYLE(주황) 테두리 박스 + idx(주황, 오른쪽 오프셋)

    Args:
        ax: Matplotlib 축.
        token_to_refined_traj_wrt_ego: Dict[str, np.ndarray] | None
            각 value: shape = (1 + future_len, 11)
            row(11,) = [x, y, cos, sin, vx, vy, length, width, onehot(3,)]
        options: 렌더링 옵션.
        token_to_new_waypoint_array: Dict[str, np.ndarray] | None
            각 value: shape = (11,) (단일 스텝)
    """
    if not token_to_refined_traj_wrt_ego and not token_to_new_waypoint_array:
        return

    eps = options.invalid_eps

    # 표시할 토큰 순서: refined의 key 순서 우선, new_waypoint에만 있는 토큰은 뒤에 추가
    ordered_tokens: List[str] = []
    if token_to_refined_traj_wrt_ego:
        ordered_tokens.extend(list(token_to_refined_traj_wrt_ego.keys()))
    if token_to_new_waypoint_array:
        for k in token_to_new_waypoint_array.keys():
            if k not in ordered_tokens:
                ordered_tokens.append(k)

    for idx, token in enumerate(ordered_tokens):
        # ── (A) refined 연속 궤적(빨강) ───────────────────────────────
        if token_to_refined_traj_wrt_ego and (token in token_to_refined_traj_wrt_ego):
            arr = token_to_refined_traj_wrt_ego[token]
            if arr is not None and arr.size > 0:
                if arr.ndim != 2 or arr.shape[1] != 11:
                    raise ValueError(
                        "token_to_refined_traj_wrt_ego의 각 value는 (1 + future_len, 11) 이어야 합니다."
                    )
                seq_len = arr.shape[0]
                label_drawn = False
                for t in range(seq_len):
                    row = arr[t]  # (11,)
                    if not is_valid_agent_row(row, eps):
                        continue
                    x, y = float(row[0]), float(row[1])
                    c, s = float(row[2]), float(row[3])
                    vx, vy = float(row[4]), float(row[5])
                    W, L = float(row[6]), float(row[7])

                    corners = oriented_box_corners(x, y, c, s, L, W)
                    add_polygon(
                        ax,
                        corners,
                        edge_color=REFINED_FUTURE_STYLE["line_color"],
                        line_width=REFINED_FUTURE_STYLE["line_width"],
                        fill_color=None,
                        fill_alpha=None,
                        zorder=24,
                    )
                    add_heading_line(
                        ax,
                        x,
                        y,
                        c,
                        s,
                        nominal_length=L * options.heading_line_scale,
                        color=REFINED_FUTURE_STYLE["line_color"],
                        line_width=REFINED_FUTURE_STYLE["line_width"],
                        zorder=24,
                    )
                    if options.draw_velocity_arrows_future_all:
                        # add_velocity_arrow(
                        #     ax,
                        #     x,
                        #     y,
                        #     vx,
                        #     vy,
                        #     length_m=options.arrow_length_m,
                        #     line_color=REFINED_FUTURE_STYLE["velocity_line_color"],
                        #     line_width=REFINED_FUTURE_STYLE["velocity_line_width"],
                        #     line_alpha=REFINED_FUTURE_STYLE["velocity_line_alpha"],
                        #     zorder=24,
                        # )
                        # [추가] 속도 크기 텍스트(km/h) - refined 모든 지점 (색: #FF4D4D)
                        speed_kmh = float(np.hypot(vx, vy)) * 3.6
                        if t % 2 == 0:
                            offset = 5
                        else:
                            offset = 3
                        ax.text(
                            x,
                            y + options.token_index_offset_m * offset,
                            f"{speed_kmh:.1f}",
                            color=REFINED_FUTURE_STYLE["line_color"],  # "#FF4D4D"
                            fontsize=2, #options.token_index_fontsize,
                            ha="center",
                            va="bottom",
                            zorder=25,
                            clip_on=True,
                        )
                    # 첫 유효 포인트에 빨간색 idx(아래쪽 오프셋)
                    # if not label_drawn:
                    #     ax.text(
                    #         x,
                    #         y - options.token_index_offset_m,
                    #         str(token)[:5],
                    #         color=REFINED_FUTURE_STYLE["line_color"],
                    #         fontsize=options.token_index_fontsize,
                    #         ha="center",
                    #         va="top",
                    #         zorder=25,
                    #     )
                    #     label_drawn = True

        # ── (B) 신규 waypoint(주황) ─────────────────────────────────
        if token_to_new_waypoint_array and (token in token_to_new_waypoint_array):
            wp = token_to_new_waypoint_array[token] # (11,)
            if wp is None:
                continue
            wp = np.asarray(wp)
            if wp.ndim == 2:
                wp = wp.squeeze()
            if wp.ndim != 1 or wp.shape[0] != 11:
                raise ValueError("token_to_new_waypoint_array의 각 value는 shape (11,) 이어야 합니다.")

            if not is_valid_agent_row(wp, eps):
                continue

            x, y = float(wp[0]), float(wp[1])
            c, s = float(wp[2]), float(wp[3])
            vx, vy = float(wp[4]), float(wp[5])
            W, L = float(wp[6]), float(wp[7])

            corners_wp = oriented_box_corners(x, y, c, s, L, W)
            # 테두리만 주황색으로
            add_polygon(
                ax,
                corners_wp,
                edge_color=NEW_WAYPOINT_STYLE["line_color"],
                line_width=NEW_WAYPOINT_STYLE["line_width"],
                fill_color=None,
                fill_alpha=None,
                zorder=28,  # refined(24)보다 위
            )
            add_heading_line(
                ax,
                x,
                y,
                c,
                s,
                nominal_length=L * options.heading_line_scale,
                color=NEW_WAYPOINT_STYLE["line_color"],
                line_width=NEW_WAYPOINT_STYLE["line_width"],
                zorder=28,
            )
            if options.draw_velocity_arrows_future_all:
                add_velocity_arrow(
                    ax,
                    x,
                    y,
                    vx,
                    vy,
                    length_m=options.arrow_length_m,
                    line_color=NEW_WAYPOINT_STYLE["line_color"],
                    line_width=NEW_WAYPOINT_STYLE["line_width"],
                    line_alpha=1.0,
                    zorder=28,
                )
                # [추가] 속도 크기 텍스트(km/h) - next waypoint 1점 (색: 주황)
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                ax.text(
                    x,
                    y + options.token_index_offset_m,
                    f"{speed_kmh:.1f}",
                    color=NEW_WAYPOINT_STYLE["line_color"],
                    fontsize=options.token_index_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=29,
                    clip_on=True,
                )
            # 번호 라벨: 주황색, "오른쪽"으로 살짝 이동
            # ax.text(
            #     x + options.token_index_offset_m,
            #     y,
            #     str(token)[:5],
            #     color=NEW_WAYPOINT_STYLE["line_color"],
            #     fontsize=options.token_index_fontsize,
            #     ha="left",
            #     va="center",
            #     zorder=29,
            # )

from typing import Optional, Literal
def draw_token_future_arrows(
    ax: plt.Axes,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict],
    options: DrawingOptions,
    draw_mode: Optional[Literal["arrow", "point"]] = None,
) -> None:
    """토큰 기준 미래 포즈를 '화살표(방향 포함)' 또는 '점(방향 미사용)'으로 그림.

    Args:
        ax: Matplotlib 축.
        token_to_future_traj_wrt_ego: Dict[str, np.ndarray] | None
            각 value: shape (future_len, 4) = [x, y, cos(yaw), sin(yaw)]
            - invalid 규칙: 4값 모두 0(±eps) → 스킵
        options: DrawingOptions
            - token_future_draw_mode: 'arrow' | 'point'
            - token_future_point_marker, token_future_point_marker_size
            - token_future_arrow_length_m
        draw_mode: Optional['arrow' | 'point']
            - 우선순위: draw_mode 인자(있으면) > options.token_future_draw_mode(없으면 'arrow')

    동작:
        - 'arrow' 모드:
            (x,y)에서 (cos,sin) 방향으로 고정 길이(options.token_future_arrow_length_m) 화살표
        - 'point' 모드:
            (x,y) 위치에 포인트만 표시(방향 미사용)

    Note:
        - t==0 (각 토큰의 첫 포인트)에는 토큰 문자열을 살짝 아래(y-오프셋)에 표시.
    """
    if not token_to_future_traj_wrt_ego:
        return

    # 모드 결정 (인자 > 옵션 > 기본값 'arrow')
    mode = (draw_mode or getattr(options, "token_future_draw_mode", "arrow")).lower()
    if mode not in {"arrow", "point"}:
        raise ValueError(f"Unsupported draw_mode: {mode}. Use 'arrow' or 'point'.")


    eps = options.invalid_eps
    point_marker = getattr(options, "token_future_point_marker", "o")
    point_ms = float(getattr(options, "token_future_point_marker_size", 0.8))
    arrow_len = float(getattr(options, "token_future_arrow_length_m", 1.0))

    # [ADD] 삽입순서 그대로 인덱스 부여를 위해 enumerate(dict.items()) 사용
    for idx, (token, arr) in enumerate(token_to_future_traj_wrt_ego.items()):  # [ADD]
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

            if mode == "arrow":
                # 방향 벡터 (c, s)를 정규화하여 고정 길이(옵션) 화살표
                add_velocity_arrow(
                    ax,
                    x,
                    y,
                    c,
                    s,
                    length_m=arrow_len,
                    line_color=TOKEN_FUTURE_STYLE["line_color"],
                    line_width=TOKEN_FUTURE_STYLE["line_width"],
                    zorder=23,
                )
            else:
                # 점만 표시(방향 정보 사용하지 않음)
                ax.plot(
                    x,
                    y,
                    marker=point_marker,
                    markersize=point_ms,
                    linestyle="None",
                    color=TOKEN_FUTURE_STYLE["line_color"],
                    zorder=23,
                )

            # 시작 포인트(t==0)에 토큰 식별 라벨(흰색)을 화살표/점 바로 아래에 표기
            # if t == 0:
            #     ax.text(
            #         x,
            #         y - options.agent_index_offset_m,
            #         str(token)[:5],
            #         color=TOKEN_FUTURE_STYLE["index_color"],
            #         fontsize=options.agent_index_fontsize,
            #         ha="center",
            #         va="top",
            #         zorder=24,
            #     )
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
    token_to_refined_traj_wrt_ego: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[float, float, float, float]:
    """valid (x,y)만 모아 자동으로 축 범위를 산출."""
    # [Add]
    xs, ys = collect_valid_xy_for_bounds(
        world_model_feature,
        options,
        token_to_future_traj_wrt_ego,
        token_to_refined_traj_wrt_ego,
    )
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


def draw_token_histories(
    ax: plt.Axes,
    current_token_to_np_history: Optional[Dict[str, np.ndarray]],
    options: DrawingOptions,
) -> None:
    """토큰별 과거 히스토리(각 row=11)를 주황색 박스/텍스트/속도화살표로 렌더링.

    Args
    ----
    ax : plt.Axes
        Matplotlib 축.
    current_token_to_np_history : Dict[str, np.ndarray] | None
        각 value: shape = (history_len, 11)
        row(11,) = [x, y, cos, sin, vx, vy, length, width, onehot(3,)]
    options : DrawingOptions
        렌더링 옵션. 화살표 길이/폰트/오프셋/두께 등.

    동작 규칙
    --------
    - invalid 스텝은 스킵: 앞 8차원 중 하나라도 |value| > eps(=options.invalid_eps)일 때만 유효.
    - 박스/헤딩/속도화살표 색상과 두께는 NEW_WAYPOINT_STYLE 사용(주황색).
    - 텍스트는 토큰 문자열을 **마지막 유효 프레임** 위치의 오른쪽에 주황색으로 표기.
    - 속도 화살표는 options.draw_velocity_arrows_past_all에 따라
      · True  → 모든 히스토리 스텝
      · False → 마지막(현재) 스텝만
    """
    if not current_token_to_np_history:
        return

    eps = options.invalid_eps
    line_color = NEW_WAYPOINT_STYLE["line_color"]
    line_width = NEW_WAYPOINT_STYLE["line_width"]

    for token, hist in current_token_to_np_history.items():
        if hist is None:
            continue
        hist = np.asarray(hist)
        if hist.ndim != 2 or hist.shape[1] != 11:
            # 형식 불일치 시 스킵
            continue

        history_len = hist.shape[0]
        current_t = history_len - 1
        last_valid_xy = None  # 텍스트 표기를 위한 마지막 유효 점

        for t in range(history_len):
            row = hist[t]  # (11,)
            if not is_valid_agent_row(row, eps):
                continue

            x, y   = float(row[0]), float(row[1])
            c, s   = float(row[2]), float(row[3])
            vx, vy = float(row[4]), float(row[5])
            W, L   = float(row[6]), float(row[7])

            # 사각형(테두리만 주황) + 헤딩선
            corners = oriented_box_corners(x, y, c, s, L, W)
            add_polygon(
                ax,
                corners,
                edge_color=line_color,
                line_width=line_width,
                fill_color=None,
                fill_alpha=None,
                zorder=19 if t == current_t else 18,
            )
            add_heading_line(
                ax,
                x,
                y,
                c,
                s,
                nominal_length=L * options.heading_line_scale,
                color=line_color,
                line_width=line_width,
                zorder=19 if t == current_t else 18,
            )

            # 속도 화살표: 모든 스텝
            if options.draw_velocity_arrows_past_all:
                add_velocity_arrow(
                    ax,
                    x, y,
                    vx, vy,
                    length_m=options.arrow_length_m,
                    line_color=line_color,
                    line_width=line_width,
                    line_alpha=1.0,
                    zorder=20 if t == current_t else 18,
                )
                # [추가] 속도 크기 텍스트(km/h) - 모든 히스토리 지점
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                if t % 2 == 0:
                    offset = 4
                else:
                    offset = 3
                ax.text(
                    x,
                    y - options.token_index_offset_m * offset,
                    f"{speed_kmh:.1f}",
                    color=line_color,
                    fontsize=2, #options.token_index_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=21,
                    clip_on=True,
                )
            # 마지막 유효 포인트 업데이트
            if (last_valid_xy is None) or (t >= current_t):
                last_valid_xy = (x, y)

        # 토큰 문자열 라벨(주황색): 마지막 유효 포인트 기준, 오른쪽으로 오프셋
        # if last_valid_xy is not None:
        #     lx, ly = last_valid_xy
        #     ax.text(
        #         lx,
        #         ly,
        #         str(token)[:5],
        #         color=line_color,
        #         fontsize=max(options.token_index_fontsize, options.agent_index_fontsize),
        #         ha="left",
        #         va="center",
        #         zorder=21,
        #     )
"""
    world_model_feature: WorldModelFeature,
    token_to_future_traj_wrt_ego: Optional[TokenTrajDict],
    token_to_refined_traj_wrt_ego: Optional[Dict[str, np.ndarray]],  # (1 + future_len=80, 11)
token_to_new_waypoint_array: Optional[Dict[str, np.ndarray]],  # (11,)
neighbor_track_token: Optional[List[Optional[str]]], # (agent_num,)
current_token_to_np_history: Dict[str, np.ndarray], # (history_len, 11)
"""
# [Add]
def draw_world_model_to_png(
    data_: Dict[str, Any],
    save_path: str,
    options: Optional[DrawingOptions] = None,
) -> None:
    """주어진 world_model_feature과 output을 그림으로 렌더링하고 PNG로 저장.

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
    # ── (0) 에이전트 앞쪽 K개로 통일 슬라이스 ─────────────────────────
    neigh_past_K, neigh_future_K, route_order_K, K = _clip_agents_first_k_from_wmf(
        world_model_feature, draw_option.max_agents_to_draw
    )

    # bounds 계산을 위해 dict 복사 후 슬라이스 반영
    wmf_for_bounds = dict(world_model_feature)
    if neigh_past_K is not None: wmf_for_bounds[
        "neighbor_agents_past"] = neigh_past_K
    if neigh_future_K is not None: wmf_for_bounds[
        "neighbor_agents_future"] = neigh_future_K
    if route_order_K is not None: wmf_for_bounds[
        "agent_route_lane_order"] = route_order_K


    # 1) Figure/Axes
    fig, ax = create_figure_and_axes(draw_option)

    # 2) 바닥 레이어(차선)
    lanes = world_model_feature.get("lanes")
    if draw_option.draw_lane_boundaries:
        draw_lane_boundaries(ax, lanes, draw_option)
    # agent_route_lane_order가 있으면 텍스트 표기 모드로 전환
    agent_route_lane_order = world_model_feature.get("agent_route_lane_order",
                                                     None)
    if draw_option.draw_lane_centerline:
        draw_lane_centerlines(
            ax,
            lanes,
            draw_option,
            agent_route_lane_order=None, #route_order_K,
        )

    # 이웃 에이전트 미래 포인트(x마커)
    if draw_option.draw_neighbor_agents_future and (neigh_future_K is not None):
        draw_neighbor_future_points(ax, neigh_future_K, draw_option)
    # 3) 토큰 미래 화살표(개별) - 차선 위에, 에이전트 윤곽과 겹치지 않게 중간 zorder
    if draw_option.draw_token_future_arrows:
        draw_token_future_arrows(ax,  token_to_future_traj_wrt_ego, draw_option)
    # [Add] refined 토큰 궤적(폴리곤/헤딩/속도)
    # [Add]
    if draw_option.draw_token_refined and (
            token_to_refined_traj_wrt_ego or token_to_new_waypoint_array):
        draw_token_refined_trajectories(
            ax,
            token_to_refined_traj_wrt_ego,
            draw_option,
            token_to_new_waypoint_array=token_to_new_waypoint_array,
        )
    # ── (4) 에이전트(과거/미래/예측/GT) ─────────────────────────────
    if draw_option.draw_neighbor_past and (neigh_past_K is not None):
        draw_neighbor_past(ax, neigh_past_K, draw_option)
        if neighbor_track_token is not None:
            annotate_neighbor_indices_for_past(ax, neighbor_track_token, neigh_past_K, draw_option)

    # if current_token_to_np_history:
    #     draw_token_histories(ax, current_token_to_np_history, draw_option)

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

    # ── (5) 축 범위/스타일 ───────────────────────────────────────────
    # [Add]
    bounds = compute_auto_bounds(
        wmf_for_bounds,
        draw_option,
        token_to_future_traj_wrt_ego,
        token_to_refined_traj_wrt_ego,
    )
    set_axes_limits_with_margin(ax, bounds, draw_option.margin_m)
    apply_axes_style(ax, draw_option)

    # 6) 저장
    save_figure_to_png(fig, save_path)


if __name__ == "__main__":
    # world_model_feature 예시(실데이터로 교체)
    world_model_feature = {
        "ego_agent_past":
            np.zeros((21, 11), dtype=np.float32),
        "neighbor_agents_past":
            np.zeros((5, 21, 11), dtype=np.float32),
        "ego_agent_next_11_dim":
            np.zeros((30, 11), dtype=np.float32),
        "ego_future_gt_11_dim":
            np.zeros((80, 11), dtype=np.float32),
        "lanes":
            np.zeros((70, 50, 12), dtype=np.float32),
        "neighbor_agents_future":
            np.array([
                np.column_stack([
                    np.linspace(0, 10, 20),
                    np.linspace(0, 0, 20),
                    np.zeros(20)
                ]),
                np.column_stack([
                    np.linspace(1, 8, 20),
                    np.linspace(2, 2, 20),
                    np.zeros(20)
                ]),
            ]),
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
