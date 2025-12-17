from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle

from matplotlib.colors import to_rgba

Array = np.ndarray
WorldModelFeature = Dict[str, Array]
TokenTrajDict = Dict[str, Array]  # value: (future_len, 4) with [x, y, cos, sin]

SILVER = "#C0C0C0"  # 은색 (interest)
GOLD = "#FFD700"  # 금색 (predict)
BLACK = "#000000"
PURPLE = "#E6E6FA"
RED = "#D50000"
YELLOW = "#FFFF00"  # 노란색(주황색 아님)
ORANGE = "#FFA500"
GREEN = "#008000"
GRAY = "#808080"
LIME = "#84E573"
LIGHTBLUE = "#4D83E1"
WHITE = "#FFFFFF"  # 미래 궤적 raw 예측 값
LIGHT_PINK = "#F7CCCC"  # slip 각 초과 제거한 궤적 값
PALE_RED = "#EE9999"  # 경로 제약 보정한 궤적 값
SOFT_RED = "#E66666"  #
BRIGHT_RED = "#DD3333"  #
PALE_CYAN = "#BFFFFF"  # 미래 궤적 refined 예측 값 # EGO Planner 궤적 값
LIGHT_CYAN = "#80FFFF"  # 미래 궤적 refined state 값 # EGO Planner 궤적 next state 값
BRIGHT_CYAN = "#40FFFF"  # neighbor_future_gt_3_dim
CYAN = "#00FFFF"  # 미래 궤적 GT 예측 값 (11 dim) # ego_future_gt_11_dim
DARK_BROWN = "#7B3F00"  # 고동색(짙은 갈색)


@dataclass
class DrawingOptions:
    """렌더링 옵션 모음.

    Attributes
    ----------
    EGO_draw_ego_past : bool
        ego_agent_past(21, 11) 시퀀스 렌더링 여부.
    draw_neighbor_past : bool
        neighbor_agents_past(max_agent_num, 21, 11) 시퀀스 렌더링 여부.
    EGO_draw_ego_agent_next_11_dim : bool
        ego_agent_next_11_dim(interpol_num, 11) 예측 궤적 렌더링 여부.
    EGO_draw_planner_future_11_dim : bool
        planner_future_11_dim(80, 11) GT 미래 궤적 렌더링 여부.
    draw_lane_boundaries : bool
        차선 좌/우 경계(LANE) 렌더링 여부(실선 #2d3ea7).
    draw_lane_centerline : bool
        차선 센터라인(BASELINE_PATHS) 점선 렌더링 여부(신호색상 반영).
    DIFF_draw_diff_future_gen_traj : bool
        token_to_future_traj_wrt_ego의 각 시점 화살표(길이 고정 2 m) 렌더링 여부.

    COMMON_vel_arrow_len_m : float
        속도/방향 화살표의 고정 길이 [m]. 기본 2.0.
    COMMON_heading_line_scale : float
        헤딩선 길이 비율(사각형 길이 * COMMON_heading_line_scale).

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
    background_color: str = BLACK
    show_axis: bool = False
    fig_size: Tuple[float, float] = (13.0, 13.0)
    dpi: int = 600
    margin_m: float = 5.0
    equal_aspect: bool = True
    invalid_eps: float = 0.0

    COMMON_vel_arrow_len_m: float = 5.0
    COMMON_heading_line_scale: float = 0.5

    ######## LANES ########
    LANE_draw_lane_boundaries: bool = True  # check
    LANE_boundary_width: float = 1.
    LANE_draw_lane_centerline: bool = False  # check
    LANE_draw_npc_agent_route: bool = False
    LANE_draw_vel_limit: bool = True
    LANE_npc_agent_route_draw_mode: str = "lane"  # "centerline" / "lane"
    LANE_route_agent_index_color: str = CYAN  # 번호 텍스트 색 # 청록색
    LANE_lane_boundary_color = PURPLE  # 남색(인디고 계열)

    LANE_speed_color: str = PURPLE  # 번호 텍스트 색 # 청록색
    LANE_speed_fontsize: int = 5  # 에이전트 번호 텍스트 폰트 크기

    LANE_signal_colors = {
        0: GREEN,  # 녹색(신호등 초록)
        1: YELLOW,  # 노란색(신호등 노랑)
        2: RED,  # 진한 빨간색(신호등 빨강)
        3: GRAY,  # 회색(청회색)
    }
    LANE_AGENT_index_fontsize: int = 10  # 에이전트 번호 텍스트 폰트 크기
    # ===== lane_type 기반 경계 색 =====
    LANE_surface_street_boundary_color: str = CYAN
    LANE_bike_lane_boundary_color: str = GREEN
    LANE_undefined_lane_boundary_color: str = SILVER

    # ===== 차선 선(road_line) 표현용 =====
    LANE_line_white_color: str = WHITE
    LANE_line_orange_color: str = ORANGE
    LANE_broken_linestyle: Any = (0, (4, 4))  # 점선 패턴
    LANE_solid_linestyle: str = "-"
    LANE_double_line_sep_m: float = 0.20  # 이중선 간격(미터)

    # ===== road_edge =====
    ROAD_draw_road_edge: bool = True
    ROAD_road_edge_color: str = RED
    ROAD_road_edge_marker_size: float = 8.0
    ROAD_road_edge_line_width: float = 0.8
    ROAD_road_edge_zorder: int = 3

    # ===== driveway =====
    DRIVEWAY_draw_driveway: bool = True
    DRIVEWAY_line_color: str = WHITE
    DRIVEWAY_line_width: float = 0.8
    DRIVEWAY_text: str = "driveway"
    DRIVEWAY_text_color: str = WHITE
    DRIVEWAY_text_fontsize: int = 5
    DRIVEWAY_polygon_zorder: int = 4
    ######### [EGO] ##############
    ########### [EGO] PAST ##################
    EGO_draw_ego_past: bool = True  # check
    EGO_draw_ego_only_current: bool = False  # check
    EGO_past_style = {
        "fill_color": WHITE,  # 흰색
        "line_color": GRAY,  # 회색
        "line_width": 0.4,
        "fill_alpha_current": 0.8,
    }
    EGO_draw_ego_past_vel: bool = False  # check
    # NEW: ego 주변 반경 원 옵션
    EGO_draw_radius_circle: bool = True  # ego 주변 원을 그릴지 여부
    EGO_radius_circle_m: float = 150.0  # 원 반지름 [m]
    EGO_radius_circle_edge_color: str = RED  # 원 테두리 색
    EGO_radius_circle_line_width: float = 0.8  # 원 테두리 두께
    ##############################
    ########### [EGO] FUTURE PLANNER NEXT STATE ##################
    EGO_future_traj_draw_mode: str = "point"  # 'rectangle' / 'arrow'/ 'point' / 'line'

    EGO_draw_ego_agent_next_11_dim: bool = False
    EGO_next_11_dim_style = {
        "line_color": LIGHT_CYAN,
        "line_width": 0.4,
        "velocity_line_color": LIGHT_CYAN,
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    EGO_draw_ego_agent_next_11_vel: bool = False
    ###################################################
    ########### [EGO] FUTURE PLANNER ##################
    EGO_draw_planner_future_11_dim: bool = True  # check
    EGO_planner_future_11_style = {
        "line_color": PALE_CYAN,
        "line_width": 1.4,
        "velocity_line_color": PALE_CYAN,
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    EGO_draw_planner_velocity: bool = True  # check
    ########## [EGO] FUTURE EGO GT 11 ##########
    EGO_draw_ego_future_gt_11_dim: bool = True
    EGO_future_gt_11_style = {
        "line_color": CYAN,
        "line_width": 0.2,
        "velocity_line_color": CYAN,
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    EGO_draw_future_11_velocity: bool = False  # check
    ######################################

    ######### [NEIGHBOR] #########
    ########### [NEIGHBOR] PAST ##################
    NEI_draw_neighbor_past: bool = True  # check
    NEI_draw_neighbor_only_current: bool = False  # check
    NEI_draw_velocity_arrow: bool = False  # check
    NEI_draw_velocity_text: bool = False  # check
    NEI_vel_text_y_offset: float = 0.5
    NEI_vel_text_color = CYAN
    NEI_vel_text_fontsize = 5

    NEI_neighbor_style = {
        "vehicles": {
            "fill_color": LIME,  # 연두색(라임 그린)
            "fill_alpha": 0.5,
            "line_color": LIME,  # 연두색(라임 그린)
            "line_width": 0.2,
            "velocity_line_color": LIME,  # 연두색(라임 그린)
            "velocity_line_width": 0.2,
        },
        "pedestrians": {
            "fill_color": LIGHTBLUE,
            "fill_alpha": 0.5,
            "line_color": LIGHTBLUE,
            "line_width": 0.2,
            "velocity_line_color": LIGHTBLUE,
            "velocity_line_width": 0.2,
        },
        "bicycles": {
            "fill_color": ORANGE,
            "fill_alpha": 0.5,
            "line_color": ORANGE,
            "line_width": 0.2,
            "velocity_line_color": ORANGE,
            "velocity_line_width": 0.2,
        },
    }
    NEI_draw_past_token: bool = True
    NEI_past_token_place_offset_m: float = 0.5
    NEI_past_token_color: str = CYAN  # TODO
    NEI_past_token_fontsize: int = 3
    ########### [NEIGHBOR] PAST OUTPUT ##################
    NEI_draw_neighbor_past_output = False
    NEI_neighbor_past_output_style = {
        "line_color": ORANGE,  # 주황색 # TODO
        "line_width": 0.2,
    }
    NEI_draw_neighbor_past_output_vel = False
    NEI_neighbor_past_output_vel_offset_m = 0.3
    NEI_neighbor_past_output_vel_fontsize = 2
    NEI_neighbor_past_output_token_fontsize = 2
    ########################
    ######### [NEIGHBOR] FUTURE GT #############
    DIFF_draw_diff_future_gt_3_dim: bool = False
    # NEW: neighbor_future_gt_3_dim (numpy 버전) on/off
    DIFF_draw_diff_np_future_gt_3_dim: bool = True
    DIFF_future_gt_3_dim_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기
    DIFF_future_gt_3_dim_COLOR: str = BRIGHT_CYAN  # 미래 포인트 'x' 마커 크기

    # NEW: GT 3차원 궤적 그리기 모드 ("point" / "line" / "arrow")
    DIFF_future_gt_3_dim_draw_mode: str = "line"
    DIFF_future_gt_3_dim_line_width: float = 0.8
    DIFF_future_gt_3_dim_arrow_len_m: float = 1.0

    DIFF_draw_diff_future_gt_3_dim_token: bool = False
    DIFF_future_gt_3_dim_text_offset_m: float = 0.  # 번호 텍스트를 포인트 옆으로 얼마나 띄울지(미터)
    DIFF_future_gt_3_dim_token_color: str = BRIGHT_CYAN
    DIFF_future_gt_3_dim_token_fontsize: int = 4  # 에이전트 번호 텍스트 폰트 크기
    ############################################
    DIFF_future_all_gt_3_dim_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기
    DIFF_future_all_gt_3_dim_COLOR: str = BRIGHT_CYAN  # 미래 포인트 'x' 마커 크기
    DIFF_draw_diff_future_all_gt_3_dim_token: bool = False
    DIFF_future_all_gt_3_dim_text_offset_m: float = 0.
    DIFF_future_all_gt_3_dim_token_color: str = BRIGHT_CYAN
    DIFF_future_all_gt_3_dim_token_fontsize: int = 4  # 에이전트 번호 텍스트 폰트 크기
    ######## [NEIGHBOR] FUTURE OUTPUT ##########
    DIFF_draw_diff_future_gen_traj: bool = True
    DIFF_draw_diff_future_gen_traj_token: bool = False
    DIFF_future_traj_draw_mode: str = "rectangle"  # 'rectangle' / 'arrow'/ 'point' / 'line'
    DIFF_future_gen_traj_point_marker: str = "o"
    DIFF_future_gen_traj_point_marker_size: float = 0.8
    DIFF_future_gen_traj_arrow_len_m: float = 1.0
    DIFF_future_gen_style = {
        "line_color": WHITE,  # 빨간색(밝은 빨강)
        "token_color": WHITE,  # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": WHITE,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_diff_future_gen_traj_vel: bool = False

    DIFF_draw_diff_future_int_traj_11: bool = True
    DIFF_future_int_style = {
        "line_color": RED,  # 빨간색(밝은 빨강)
        "token_color": RED,  # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": RED,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_diff_future_int_traj_11_vel: bool = False

    DIFF_draw_diff_future_int_traj_to_be: bool = False
    DIFF_future_slip_style = {
        "line_color": ORANGE,  # 빨간색(밝은 빨강)
        "token_color": ORANGE,  # 빨간색(밝은 빨강)
        "line_width": 1.4,
        "velocity_line_color": ORANGE,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_diff_future_int_traj_to_be_vel: bool = False

    DIFF_future_gen_trak_token_text_y_offset_m: float = 0.5

    ########################
    DIFF_draw_diff_future_gen_refined_traj: bool = True
    DIFF_future_gen_refined_style = {
        "line_color": RED,  # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": RED,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_future_gen_refined_velocity: bool = True
    DIFF_future_gen_refined_velocity_offset_m: float = 0.3
    DIFF_future_gen_refined_velocity_font_size = 5
    DIFF_future_gen_refined_token_offset_m: float = 0.3  # y축으로 살짝 아래(미터 단위)
    DIFF_new_waypoint_vel_text_y_offset_m = 1.
    DIFF_new_waypoint_vel_token_x_offset_m = 1.

    DIFF_new_waypoint_style = {
        "line_color": ORANGE,  # 주황색
        "line_width": 0.2,
    }
    ###################
    ######## ROAD SAFETY (stop sign / speed bump / crosswalk) ########
    SAFETY_draw_stop_sign_points: bool = True
    SAFETY_draw_speed_bump_points: bool = True
    SAFETY_draw_crosswalk_points: bool = True

    SAFETY_stop_sign_line_color: str = RED
    SAFETY_stop_sign_line_width: float = 0.8
    SAFETY_stop_sign_text: str = "stop_sign"
    SAFETY_stop_sign_text_color: str = RED
    SAFETY_stop_sign_text_fontsize: int = 5

    SAFETY_speed_bump_line_color: str = ORANGE
    SAFETY_speed_bump_line_width: float = 0.8
    SAFETY_speed_bump_text: str = "speed_bump"
    SAFETY_speed_bump_text_color: str = ORANGE
    SAFETY_speed_bump_text_fontsize: int = 5

    SAFETY_crosswalk_line_color: str = WHITE
    SAFETY_crosswalk_line_width: float = 0.8
    SAFETY_crosswalk_text: str = "crosswalk"
    SAFETY_crosswalk_text_color: str = WHITE
    SAFETY_crosswalk_text_fontsize: int = 5

    SAFETY_polygon_zorder: int = 4
    SAFETY_text_zorder: int = 5
    # NEW: neighbor_role에 따른 현재 프레임 테두리 색
    NEI_role_interest_edge_color: str = SILVER
    NEI_role_predict_edge_color: str = GOLD


# [ADD]
def _collect_valid_xy_from_input_data(
    input_data: WorldModelFeature,
    options: DrawingOptions,
) -> Tuple[List[float], List[float]]:
    """input_data에서 valid (x,y) 좌표만 수집해 반환.

    수집 대상
    - lanes: center/left/right (유효 포인트만)
    - ego: ego_agent_past, ego_agent_next_11_dim, planner_future_11_dim
    - neighbor: neighbor_agents_past
    - neighbor future GT: neighbor_future_gt_3_dim (x,y만 사용)
    """
    eps = options.invalid_eps
    xs_local: List[float] = []
    ys_local: List[float] = []

    # lanes: 각 포인트가 valid일 때 center/left/right 좌표 반영
    lanes = input_data.get("lanes")
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
            xs_local.extend(c_valid[:, 0].tolist())
            ys_local.extend(c_valid[:, 1].tolist())
            xs_local.extend(l_valid[:, 0].tolist())
            ys_local.extend(l_valid[:, 1].tolist())
            xs_local.extend(r_valid[:, 0].tolist())
            ys_local.extend(r_valid[:, 1].tolist())

    # ego past, ego pred, ego future
    for key in ("ego_agent_past", "ego_agent_next_11_dim",
                "planner_future_11_dim", "planner_future_11_dim"):
        A = input_data.get(key)
        if A is None or A.size == 0:
            continue
        valid_mask = np.any(np.abs(A[:, :8]) > eps, axis=1)  # (N,)
        if np.any(valid_mask):
            xy = A[valid_mask, 0:2]
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # neighbor past
    neigh = input_data.get("neighbor_agents_past")
    if neigh is not None and neigh.size > 0:
        valid_mask = np.any(np.abs(neigh[:, :, :8]) > eps,
                            axis=2)  # (max_agent_num, T)
        if np.any(valid_mask):
            xy = neigh[:, :, 0:2][valid_mask]
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # neighbor future points: (max_agent_num, future_len, 3) -> (x,y)만 사용
    neigh_fut = input_data.get("neighbor_future_gt_3_dim")
    if neigh_fut is not None and neigh_fut.size > 0:
        if neigh_fut.ndim != 3 or neigh_fut.shape[-1] != 3:
            raise ValueError(
                "neighbor_agents_future는 (max_agent_num, future_len, 3) 이어야 합니다."
            )
        valid_mask = (np.abs(neigh_fut[..., 0]) > eps) | (np.abs(
            neigh_fut[..., 1]) > eps)  # (A, T)
        if np.any(valid_mask):
            xy = neigh_fut[..., :2][valid_mask]  # (K, 2)
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())
    # road safety polygons:
    # - stop_sign_points / speed_bump_points / crosswalk_points
    # - shape: (N, P, 2)
    for road_safety_key in ("stop_sign_points", "speed_bump_points",
                            "crosswalk_points"):
        road_safety_points = input_data.get(road_safety_key)
        if road_safety_points is None:
            continue
        road_safety_points = np.asarray(road_safety_points)  # (N, P, 2)
        if road_safety_points.size == 0:
            continue
        if road_safety_points.ndim != 3 or road_safety_points.shape[-1] != 2:
            raise ValueError(
                f"{road_safety_key} shape는 (N, P, 2) 이어야 합니다. got {road_safety_points.shape}"
            )

        # 유효 점만 bounds에 반영 (한 점이라도 |.| > eps 이면 유효)
        valid_mask = np.any(np.abs(road_safety_points) > eps, axis=2)  # (N, P)
        if np.any(valid_mask):
            valid_xy = road_safety_points[valid_mask]  # (K, 2)
            xs_local.extend(valid_xy[:, 0].tolist())
            ys_local.extend(valid_xy[:, 1].tolist())
    # road_edge / driveway: shape (N, P, 2)
    for key in ("road_edge", "driveway"):
        pts = input_data.get(key)
        if pts is None:
            continue
        pts = np.asarray(pts)  # (N,P,2)
        if pts.size == 0:
            continue
        if pts.ndim != 3 or pts.shape[-1] != 2:
            raise ValueError(f"{key} shape는 (N, P, 2) 이어야 합니다. got {pts.shape}")

        valid_mask = np.any(np.abs(pts) > eps, axis=2)  # (N,P)
        if np.any(valid_mask):
            valid_xy = pts[valid_mask]  # (K,2)
            xs_local.extend(valid_xy[:, 0].tolist())
            ys_local.extend(valid_xy[:, 1].tolist())

    return xs_local, ys_local


# [ADD]
def _collect_valid_xy_from_output_data(
    output_data: Dict[str, Any],
    options: DrawingOptions,
) -> Tuple[List[float], List[float]]:
    """output_data의 모든 key를 검사하여 valid (x,y) 좌표만 수집해 반환.

    예상 키와 타입
    - diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, (T,11)] 또는 (11)
    - diff_token_to_np_history_wrt_ego: Dict[str, (H,11)] 또는 (11,)
    - diff_token_to_interp_np_traj_wrt_ego: Dict[str, (N,11)] 또는 (11,)
    - diff_token_to_next_wp_wrt_ego: Dict[str, (11,)] 또는 (1,11)
    """
    eps = options.invalid_eps
    xs_local: List[float] = []
    ys_local: List[float] = []

    if not output_data:
        return xs_local, ys_local

    # (1) diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, (T,11)] 또는
    token_to_traj = output_data.get("diff_token_to_np_gen_traj_11_wrt_ego")
    if isinstance(token_to_traj, dict) and len(token_to_traj) > 0:
        for _, arr in token_to_traj.items():
            if arr is None:
                continue
            arr = np.asarray(arr).copy()
            arr = arr[:, 0:4]  # (T,4)
            if arr.ndim == 2 and arr.shape[1] == 4:
                valid_mask = np.any(np.abs(arr[:, :4]) > eps, axis=1)
                if np.any(valid_mask):
                    xy = arr[valid_mask, 0:2]
                    xs_local.extend(xy[:, 0].tolist())
                    ys_local.extend(xy[:, 1].tolist())
            elif arr.ndim == 1 and arr.shape[0] == 4:
                if is_valid_token_row(arr, eps):
                    xs_local.append(float(arr[0]))
                    ys_local.append(float(arr[1]))

    # (2) diff_token_to_np_history_wrt_ego: Dict[str, (H,11)] 또는 (11,)
    hist_dict = output_data.get("diff_token_to_np_history_wrt_ego")
    if isinstance(hist_dict, dict) and len(hist_dict) > 0:
        for _, arr in hist_dict.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 11:
                valid_mask = np.any(np.abs(arr[:, :8]) > eps, axis=1)
                if np.any(valid_mask):
                    xy = arr[valid_mask, 0:2]
                    xs_local.extend(xy[:, 0].tolist())
                    ys_local.extend(xy[:, 1].tolist())
            elif arr.ndim == 1 and arr.shape[0] == 11:
                if is_valid_agent_row(arr, eps):
                    xs_local.append(float(arr[0]))
                    ys_local.append(float(arr[1]))

    # (3) diff_token_to_interp_np_traj_wrt_ego: Dict[str, (N,11)] 또는 (11,)
    interp_dict = output_data.get("diff_token_to_interp_np_traj_wrt_ego")
    if isinstance(interp_dict, dict) and len(interp_dict) > 0:
        for _, arr in interp_dict.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2 and arr.shape[1] == 11:
                valid_mask = np.any(np.abs(arr[:, :8]) > eps, axis=1)
                if np.any(valid_mask):
                    xy = arr[valid_mask, 0:2]
                    xs_local.extend(xy[:, 0].tolist())
                    ys_local.extend(xy[:, 1].tolist())
            elif arr.ndim == 1 and arr.shape[0] == 11:
                if is_valid_agent_row(arr, eps):
                    xs_local.append(float(arr[0]))
                    ys_local.append(float(arr[1]))

    # (4) diff_token_to_next_wp_wrt_ego: Dict[str, (11,)] 또는 (1,11)
    next_wp_dict = output_data.get("diff_token_to_next_wp_wrt_ego")
    if isinstance(next_wp_dict, dict) and len(next_wp_dict) > 0:
        for _, arr in next_wp_dict.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr.squeeze()
            if arr.ndim == 1 and arr.shape[0] == 11:
                if is_valid_agent_row(arr, eps):
                    xs_local.append(float(arr[0]))
                    ys_local.append(float(arr[1]))

    return xs_local, ys_local


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
        self.diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, np.ndarray] = {
        }  # (T, 11)
        """
        딥러닝 적분 출력값
        """
        self.diff_token_to_np_int_traj_wrt_ego: Dict[str,
                                                     np.ndarray] = {}  # (T, 4)
        """
        딥러닝 적분 출력삽을 11차원으로 
        """
        self.diff_token_to_np_int_traj_11_wrt_ego: Dict[str, np.ndarray] = {
        }  # (T, 11)
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
        }  # (1 + Future_len, 11)
        """
        interpolation 궤적 생성 후, next_iteration 시점 waypoint를 array로 변환한 것
        """
        self.diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] = {}  # (11,)

    def to_dict(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_data = self.model_input_key_to_unnorm_value
        output_data = {
            "diff_token_to_np_gen_traj_11_wrt_ego":
                self.diff_token_to_np_gen_traj_11_wrt_ego,
            "diff_token_to_np_int_traj_wrt_ego":
                self.diff_token_to_np_int_traj_wrt_ego,
            "diff_token_to_np_int_traj_11_wrt_ego":
                self.diff_token_to_np_int_traj_11_wrt_ego,
            "diff_token_to_np_history_wrt_ego":
                self.diff_token_to_np_history_wrt_ego,
            "diff_token_to_interp_np_traj_wrt_ego":
                self.diff_token_to_interp_np_traj_wrt_ego,
            "diff_token_to_next_wp_wrt_ego":
                self.diff_token_to_next_wp_wrt_ego,
        }
        return input_data, output_data


def is_valid_future_row_xyyaw(row3: Array, eps: float) -> bool:
    """미래 포인트(3,)=[x,y,yaw]가 **유효**하면 True.
    - 규칙: |x|>eps 또는 |y|>eps 이면 유효로 간주( yaw=0 이어도 상관 없음 )
    """
    if row3.shape[-1] != 3:
        raise ValueError("neighbor_agents_future의 마지막 차원은 3이어야 합니다.")
    return bool((abs(float(row3[0])) > eps) or (abs(float(row3[1])) > eps))


def draw_neighbor_future_gt_3_dim(
    ax: plt.Axes,
    diff_token_to_future_gt_3_dim: Dict[str, Array],
    options: DrawingOptions,
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """이웃 차량의 GT 미래 궤적(토큰별 [x, y, yaw] 시퀀스)을 그린다.

    모드별 동작
    -----------
    - "point":
        각 (x, y)를 독립된 점('x' 마커)으로 그림. (현재 동작과 동일)
    - "line":
        유효한 점들끼리 앞뒤 순서대로 선으로 이어 그림.
    - "arrow":
        각 (x, y)에서 yaw 방향으로 고정 길이 화살표를 그림.
        (길이: options.DIFF_future_gt_3_dim_arrow_len_m)

    Args:
        ax:
            Matplotlib 축 객체.
        diff_token_to_future_gt_3_dim:
            key = track_token(str),
            value = shape (T, 3) 의 배열. 각 행은 [x, y, yaw].
        options:
            DrawingOptions. invalid_eps, 색상/선두께/모드 설정을 사용한다.
        draw_token_list:
            특정 토큰만 그릴 때 사용하는 필터 리스트. None이면 전체 토큰을 그림.
    """
    eps: float = options.invalid_eps
    draw_mode: str = options.DIFF_future_gt_3_dim_draw_mode

    if draw_mode not in ("point", "line", "arrow"):
        raise ValueError(
            f"지원하지 않는 DIFF_future_gt_3_dim_draw_mode 값입니다: {draw_mode!r}. "
            f"'point', 'line', 'arrow' 중 하나여야 합니다.")

    for track_token, future_gt_3_dim in diff_token_to_future_gt_3_dim.items():
        # future_gt_3_dim: (T, 3)
        if draw_token_list is not None and track_token not in draw_token_list:
            continue

        if future_gt_3_dim.ndim != 2 or future_gt_3_dim.shape[1] != 3:
            raise ValueError(
                f"future_gt_3_dim 의 shape 은 (T, 3) 이어야 합니다. got {future_gt_3_dim.shape}"
            )

        future_len: int = int(future_gt_3_dim.shape[0])

        # 선 모드일 때 인접 포인트 연결을 위한 이전 점 저장용
        prev_xy: Optional[Tuple[float, float]] = None

        for t in range(future_len):
            row: Array = future_gt_3_dim[t]  # shape: (3,) = [x, y, yaw]
            if not is_valid_future_row_xyyaw(row, eps):
                continue

            x: float = float(row[0])
            y: float = float(row[1])
            yaw: float = float(row[2])

            if draw_mode == "point":
                # 점 모드: 기존 동작 유지
                ax.plot(
                    x,
                    y,
                    marker="x",
                    markersize=options.DIFF_future_gt_3_dim_marker_size,
                    linestyle="None",
                    color=options.DIFF_future_gt_3_dim_COLOR,
                    zorder=26,
                )

            elif draw_mode == "line":
                # 선 모드: 인접 유효 점끼리 선으로 연결
                if prev_xy is not None:
                    px, py = prev_xy
                    ax.plot(
                        [px, x],
                        [py, y],
                        linestyle="-",
                        linewidth=options.DIFF_future_gt_3_dim_line_width,
                        color=options.DIFF_future_gt_3_dim_COLOR,
                        zorder=26,
                    )
                prev_xy = (x, y)

            elif draw_mode == "arrow":
                # 화살표 모드: yaw 로부터 단위 방향 벡터 계산 후 고정 길이 화살표
                cos_yaw: float = float(np.cos(yaw))
                sin_yaw: float = float(np.sin(yaw))

                add_velocity_arrow(
                    ax=ax,
                    x=x,
                    y=y,
                    vx=cos_yaw,
                    vy=sin_yaw,
                    length_m=options.DIFF_future_gt_3_dim_arrow_len_m,
                    line_color=options.DIFF_future_gt_3_dim_COLOR,
                    line_width=options.DIFF_future_gt_3_dim_line_width,
                    line_alpha=None,
                    zorder=26,
                    t=t,
                )

        # 첫 점 라벨(유효할 때만)
        if options.DIFF_draw_diff_future_gt_3_dim_token and future_len > 0:
            first: Array = future_gt_3_dim[0]  # shape: (3,)
            if is_valid_future_row_xyyaw(first, eps):
                fx: float = float(first[0])
                fy: float = float(first[1])
                ax.text(
                    fx + options.DIFF_future_gt_3_dim_text_offset_m,
                    fy + options.DIFF_future_gt_3_dim_text_offset_m,
                    str(track_token)[:5],
                    color=options.DIFF_future_gt_3_dim_token_color,
                    fontsize=options.DIFF_future_gt_3_dim_token_fontsize,
                    ha="left",
                    va="bottom",
                    zorder=30,
                )


def _build_diff_token_to_future_gt_3_dim_from_neighbor_np(
        neighbor_future_gt_3_dim: Array,  # shape: (chosen_agent_num, T, 3)
        neighbor_track_token: List[str],  # 길이: chosen_agent_num
) -> Dict[str, Array]:
    """neighbor_future_gt_3_dim 배열을 토큰 딕셔너리 형태로 바꾼다.

    이 함수는
    - 각 에이전트 인덱스 i에 대해
      neighbor_track_token[i] 를 key 로,
      neighbor_future_gt_3_dim[i] (shape: (T, 3)) 를 value 로 쓰는
      `diff_token_to_future_gt_3_dim` 딕셔너리를 만들어준다.

    단, 길이나 모양이 맞지 않으면 바로 예외를 발생시켜
    잘못된 입력을 조기에 잡는다.

    Args:
        neighbor_future_gt_3_dim (np.ndarray):
            shape = (chosen_agent_num, T, 3).
            각 행 = 한 에이전트의 [x, y, yaw] 시퀀스.
        neighbor_track_token (List[str]):
            길이 = chosen_agent_num.
            각 에이전트에 대응되는 track_token 문자열 리스트.

    Returns:
        Dict[str, np.ndarray]:
            key   = track_token (문자열),
            value = 해당 에이전트의 future_gt_3_dim 배열 (shape: (T, 3)).
    """
    arr: Array = np.asarray(neighbor_future_gt_3_dim)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(
            f"`neighbor_future_gt_3_dim`은 (chosen_agent_num, T, 3) 이어야 합니다. "
            f"got {arr.shape}")

    chosen_agent_num: int = int(arr.shape[0])
    if len(neighbor_track_token) != chosen_agent_num:
        raise ValueError(
            f"`neighbor_track_token` 길이({len(neighbor_track_token)})와 "
            f"`neighbor_future_gt_3_dim`의 첫 축({chosen_agent_num}) 이 다릅니다.")

    diff_token_to_future_gt_3_dim: Dict[str, Array] = {}
    for agent_idx in range(chosen_agent_num):
        token: str = neighbor_track_token[agent_idx]
        # value: shape = (T, 3)
        future_gt_3_dim_agent: Array = arr[agent_idx]
        diff_token_to_future_gt_3_dim[token] = future_gt_3_dim_agent

    return diff_token_to_future_gt_3_dim


# =============================================================================
# 옵션/스타일
# =============================================================================

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


def _is_valid_road_safety_polygon_points(
    polygon_points: Array,  # (P, 2)
    eps: float,
) -> bool:
    """도로 안전 다각형 점들이 유효한지 확인한다.

    Args:
        polygon_points (np.ndarray): 모양이 (P, 2) 인 점 배열.
        eps (float): 0으로 볼 허용 오차.

    Returns:
        bool: 한 점이라도 |값|이 eps 보다 크면 True.
    """
    polygon_points = np.asarray(polygon_points)  # (P, 2)
    if polygon_points.ndim != 2 or polygon_points.shape[1] != 2:
        raise ValueError(
            f"polygon_points shape는 (P, 2) 이어야 합니다. got {polygon_points.shape}")
    return bool(np.any(np.abs(polygon_points) > eps))


def _compute_polygon_center_xy(
        polygon_points: Array,  # (P, 2)
) -> Tuple[float, float]:
    """다각형의 가운데 위치를 간단히 계산한다.

    이 함수는 다각형의 점들을 평균내서 가운데 좌표로 사용한다.
    (정확한 무게중심이 아니라, 라벨을 찍기 위한 '대략 중앙' 용도다.)

    Args:
        polygon_points (np.ndarray): 모양이 (P, 2) 인 점 배열.

    Returns:
        Tuple[float, float]: (center_x, center_y)
    """
    polygon_points = np.asarray(polygon_points)  # (P, 2)
    if polygon_points.ndim != 2 or polygon_points.shape[1] != 2:
        raise ValueError(
            f"polygon_points shape는 (P, 2) 이어야 합니다. got {polygon_points.shape}")
    center_xy: Array = polygon_points.mean(axis=0)  # (2,)
    return float(center_xy[0]), float(center_xy[1])


def draw_road_safety_polygons_with_text(
    ax: plt.Axes,
    road_safety_points: Array,  # (N, P, 2)
    label_text: str,
    line_color: str,
    line_width: float,
    text_color: str,
    text_fontsize: int,
    options: DrawingOptions,
    zorder: int,
) -> None:
    """도로 안전 다각형을 테두리만 그리고, 가운데에 글씨를 적는다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        road_safety_points (np.ndarray): 모양이 (N, P, 2) 인 점 배열.
            - N: 다각형 개수
            - P: 한 다각형을 이루는 점 개수
        label_text (str): 가운데에 적을 글씨.
        line_color (str): 테두리 색.
        line_width (float): 테두리 두께.
        text_color (str): 글씨 색.
        text_fontsize (int): 글씨 크기.
        options (DrawingOptions): invalid_eps 등을 쓰기 위한 옵션.
        zorder (int): 그리기 순서.
    """
    road_safety_points = np.asarray(road_safety_points)  # (N, P, 2)
    if road_safety_points.size == 0:
        return
    if road_safety_points.ndim != 3 or road_safety_points.shape[-1] != 2:
        raise ValueError(
            f"road_safety_points shape는 (N, P, 2) 이어야 합니다. got {road_safety_points.shape}"
        )

    eps: float = float(options.invalid_eps)
    num_polygons: int = int(road_safety_points.shape[0])
    for poly_idx in range(num_polygons):
        polygon_points: Array = road_safety_points[poly_idx]  # (P, 2)
        if not _is_valid_road_safety_polygon_points(polygon_points, eps):
            continue

        # 다각형 테두리(속은 비움)
        add_polygon(
            ax=ax,
            corners_xy=polygon_points,  # (P, 2)
            edge_color=line_color,
            line_width=line_width,
            fill_color=None,
            fill_alpha=None,
            zorder=zorder,
        )

        # 가운데 텍스트
        center_x, center_y = _compute_polygon_center_xy(polygon_points)
        ax.text(
            center_x,
            center_y,
            label_text,
            color=text_color,
            fontsize=text_fontsize,
            ha="center",
            va="center",
            zorder=options.SAFETY_text_zorder,
            clip_on=True,
        )


def draw_stop_sign_points(
    ax: plt.Axes,
    stop_sign_points: Optional[Array],  # (N, P, 2)
    options: DrawingOptions,
) -> None:
    """stop_sign_points 다각형을 빨간 테두리로 그린다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        stop_sign_points (Optional[np.ndarray]): 모양이 (N, P, 2) 인 점 배열.
        options (DrawingOptions): 색/두께/글씨 크기 옵션.
    """
    if not options.SAFETY_draw_stop_sign_points:
        return
    if stop_sign_points is None:
        return
    stop_sign_points = np.asarray(stop_sign_points)  # (N, P, 2)
    if stop_sign_points.size == 0:
        return

    draw_road_safety_polygons_with_text(
        ax=ax,
        road_safety_points=stop_sign_points,
        label_text=options.SAFETY_stop_sign_text,
        line_color=options.SAFETY_stop_sign_line_color,
        line_width=options.SAFETY_stop_sign_line_width,
        text_color=options.SAFETY_stop_sign_text_color,
        text_fontsize=options.SAFETY_stop_sign_text_fontsize,
        options=options,
        zorder=options.SAFETY_polygon_zorder,
    )


def draw_speed_bump_points(
    ax: plt.Axes,
    speed_bump_points: Optional[Array],  # (N, P, 2)
    options: DrawingOptions,
) -> None:
    """speed_bump_points 다각형을 주황 테두리로 그린다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        speed_bump_points (Optional[np.ndarray]): 모양이 (N, P, 2) 인 점 배열.
        options (DrawingOptions): 색/두께/글씨 크기 옵션.
    """
    if not options.SAFETY_draw_speed_bump_points:
        return
    if speed_bump_points is None:
        return
    speed_bump_points = np.asarray(speed_bump_points)  # (N, P, 2)
    if speed_bump_points.size == 0:
        return

    draw_road_safety_polygons_with_text(
        ax=ax,
        road_safety_points=speed_bump_points,
        label_text=options.SAFETY_speed_bump_text,
        line_color=options.SAFETY_speed_bump_line_color,
        line_width=options.SAFETY_speed_bump_line_width,
        text_color=options.SAFETY_speed_bump_text_color,
        text_fontsize=options.SAFETY_speed_bump_text_fontsize,
        options=options,
        zorder=options.SAFETY_polygon_zorder,
    )


def draw_crosswalk_points(
    ax: plt.Axes,
    crosswalk_points: Optional[Array],  # (N, P, 2)
    options: DrawingOptions,
) -> None:
    """crosswalk_points 다각형을 흰색 테두리로 그린다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        crosswalk_points (Optional[np.ndarray]): 모양이 (N, P, 2) 인 점 배열.
        options (DrawingOptions): 색/두께/글씨 크기 옵션.
    """
    if not options.SAFETY_draw_crosswalk_points:
        return
    if crosswalk_points is None:
        return
    crosswalk_points = np.asarray(crosswalk_points)  # (N, P, 2)
    if crosswalk_points.size == 0:
        return

    draw_road_safety_polygons_with_text(
        ax=ax,
        road_safety_points=crosswalk_points,
        label_text=options.SAFETY_crosswalk_text,
        line_color=options.SAFETY_crosswalk_line_color,
        line_width=options.SAFETY_crosswalk_line_width,
        text_color=options.SAFETY_crosswalk_text_color,
        text_fontsize=options.SAFETY_crosswalk_text_fontsize,
        options=options,
        zorder=options.SAFETY_polygon_zorder,
    )


def draw_road_safety(
    ax: plt.Axes,
    input_data: WorldModelFeature,
    options: DrawingOptions,
) -> None:
    """stop/speed_bump/crosswalk 다각형을 input_data에서 찾아 그린다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        input_data (Dict[str, np.ndarray]): draw_world_model_to_png 로 들어오는 입력 dict.
        options (DrawingOptions): 색/두께/글씨 크기 옵션.
    """
    stop_sign_points = input_data.get("stop_sign_points", None)  # (N, P, 2)
    speed_bump_points = input_data.get("speed_bump_points", None)  # (N, P, 2)
    crosswalk_points = input_data.get("crosswalk_points", None)  # (N, P, 2)

    draw_stop_sign_points(ax, stop_sign_points, options)
    draw_speed_bump_points(ax, speed_bump_points, options)
    draw_crosswalk_points(ax, crosswalk_points, options)


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
                       t: int,
                       line_alpha: Optional[float] = None,
                       zorder: int = 15,
                       text_y_offset: float = 0.5,
                       vel_text_fontsize: int = 5) -> None:
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
            mutation_scale=4.0,
            linewidth=line_width,
            color=line_color,
            alpha=line_alpha,
            zorder=zorder,
            shrinkA=0.0,
            shrinkB=0.0,
        ))
    if t % 20 == 0:
        offset = 5
    else:
        offset = 3
    if t % 10 != 0:
        return
    # [추가] 속도 크기 텍스트(km/h) - 모든 과거 지점
    speed_kmh = float(np.hypot(vx, vy)) * 3.6
    ax.text(
        x,  # 점 위쪽에 표기
        y + text_y_offset * offset,
        f"{speed_kmh:.1f}",
        color=line_color,
        fontsize=vel_text_fontsize,
        ha="center",
        va="bottom",
        zorder=30,
        clip_on=True,  # tight 저장 시 bbox 폭주 방지
    )


def infer_agent_class(one_hot: Array) -> str:
    """원-핫(3,) → 'vehicles' | 'pedestrians' | 'bicycles' 반환."""
    idx = int(np.argmax(one_hot))
    return ["vehicles", "pedestrians", "bicycles"][idx]


def is_valid_agent_row(row11: Array, eps: float) -> bool:
    """에이전트 1타임스텝(11,)이 **유효**하면 True.
    - 규칙: 앞 8차원([x,y,cos,sin,vx,vy,length,width]) 중 하나라도 |value| > eps
    """
    return bool(np.any(np.abs(row11[..., :8]) > eps))


def _get_neighbor_edge_color_for_current_t(
    agent_idx: int,
    neighbor_role: Optional[Array],  # shape: (A,2) bool, [interest, predict]
    default_edge_color: str,
    options: DrawingOptions,
) -> str:
    """현재 시점에서 neighbor_role에 따라 테두리 색을 결정한다.

    우선순위:
      - predict(True)  -> GOLD
      - interest(True) -> SILVER
      - 그 외          -> default_edge_color
    """
    if neighbor_role is None:
        return default_edge_color

    role_arr = np.asarray(neighbor_role)
    if role_arr.ndim != 2 or role_arr.shape[1] < 2:
        return default_edge_color

    if agent_idx < 0 or agent_idx >= role_arr.shape[0]:
        return default_edge_color

    is_interest = bool(role_arr[agent_idx, 0])
    is_predict = bool(role_arr[agent_idx, 1])

    # 둘 다 True일 수 있으면 predict를 우선(금색이 더 눈에 띄고, 일반적으로 "예측 대상" 강조)
    if is_predict:
        return options.NEI_role_predict_edge_color
    if is_interest:
        return options.NEI_role_interest_edge_color
    return default_edge_color


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
    input_data: WorldModelFeature,
    output_data: Dict[str, Any],
    options: DrawingOptions,
) -> Tuple[List[float], List[float]]:
    """모든 요소에서 valid (x,y)만 모아 bounds 계산에 사용.

    - lanes(center/left/right), ego/neighbor past, neighbor future GT
    - output_data의 모든 key도 검사하여 (x,y) 수집
    """
    xs_input, ys_input = _collect_valid_xy_from_input_data(input_data, options)
    xs_output, ys_output = _collect_valid_xy_from_output_data(
        output_data, options)

    xs: List[float] = []
    ys: List[float] = []
    xs.extend(xs_input)
    ys.extend(ys_input)
    xs.extend(xs_output)
    ys.extend(ys_output)

    return xs, ys


# =============================================================================
# 레이어별 드로잉 함수 (invalid 스킵 포함)
# =============================================================================
from dataclasses import dataclass


def _lane_type_one_hot_to_color(
    lane_type_row4: Optional[Array],  # shape: (4,)
    options: DrawingOptions,
) -> str:
    """lane_type(4,) 값으로 차선 경계 색을 고른다.

    Args:
        lane_type_row4: shape (4,). [FREEWAY, SURFACE_STREET, BIKE_LANE, UNDEFINED] one-hot.
        options: DrawingOptions.

    Returns:
        color: 선택된 색 문자열.
    """
    if lane_type_row4 is None:
        return options.LANE_lane_boundary_color

    lane_type_row4 = np.asarray(lane_type_row4).reshape(-1)  # shape: (4,)
    if lane_type_row4.shape[0] != 4:
        return options.LANE_lane_boundary_color

    if float(np.sum(np.abs(lane_type_row4))) == 0.0:
        idx = 3  # UNDEFINED
    else:
        idx = int(np.argmax(lane_type_row4))

    if idx == 0:  # FREEWAY
        return options.LANE_lane_boundary_color
    if idx == 1:  # SURFACE_STREET
        return options.LANE_surface_street_boundary_color
    if idx == 2:  # BIKE_LANE
        return options.LANE_bike_lane_boundary_color
    return options.LANE_undefined_lane_boundary_color


def _line_type_one_hot_to_index_10(
        line_type_row10: Array,  # shape: (10,)
) -> int:
    """left/right_line_type(10,) one-hot에서 가장 큰 인덱스를 고른다.

    Args:
        line_type_row10: shape (10,).

    Returns:
        idx: 0..9
    """
    line_type_row10 = np.asarray(line_type_row10).reshape(-1)  # shape: (10,)
    if line_type_row10.shape[0] != 10:
        raise ValueError(
            f"line_type shape는 (10,) 이어야 합니다. got {line_type_row10.shape}")

    if float(np.sum(np.abs(line_type_row10))) == 0.0:
        return 8  # UNKNOWN
    return int(np.argmax(line_type_row10))


@dataclass(frozen=True)
class _LaneBoundaryDrawPlan:
    """차선 경계 1쪽을 어떻게 그릴지 정리한 값들."""
    draw: bool
    color: str
    is_double: bool
    inner_linestyle: Any
    outer_linestyle: Any


def _make_lane_boundary_draw_plan(
    line_type_row10: Optional[Array],  # shape: (10,) or None
    default_color: str,
    options: DrawingOptions,
) -> _LaneBoundaryDrawPlan:
    """left/right_line_type에 따라 점선/실선/이중선/색을 결정한다.

    규칙:
      - None: 원래처럼(default) 그림
      - UNKNOWN(8): 원래처럼(default) 그림
      - INVALID(9): 안 그림
      - 나머지 8종: 타입에 맞게 색(흰/노란) + 점선/실선 + 이중선 반영

    Args:
        line_type_row10: shape (10,) 또는 None.
        default_color: UNKNOWN/None일 때 쓸 기본 색.
        options: DrawingOptions.

    Returns:
        plan: 그리기 계획(_LaneBoundaryDrawPlan).
    """
    # 기본(원래처럼)
    default_plan = _LaneBoundaryDrawPlan(
        draw=True,
        color=default_color,
        is_double=False,
        inner_linestyle=options.LANE_solid_linestyle,
        outer_linestyle=options.LANE_solid_linestyle,
    )

    if line_type_row10 is None:
        return default_plan

    idx = _line_type_one_hot_to_index_10(line_type_row10)

    # INVALID
    if idx == 9:
        return _LaneBoundaryDrawPlan(
            draw=False,
            color=default_color,
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    # UNKNOWN
    if idx == 8:
        return default_plan

    # 타입별 매핑
    # 0: BROKEN_SINGLE_WHITE
    # 1: SOLID_SINGLE_WHITE
    # 2: SOLID_DOUBLE_WHITE
    # 3: BROKEN_SINGLE_YELLOW
    # 4: BROKEN_DOUBLE_YELLOW
    # 5: SOLID_SINGLE_YELLOW
    # 6: SOLID_DOUBLE_YELLOW
    # 7: PASSING_DOUBLE_YELLOW
    if idx in (0, 1, 2):
        line_color = options.LANE_line_white_color
    else:
        line_color = options.LANE_line_orange_color

    if idx in (0, 3):
        # broken single
        return _LaneBoundaryDrawPlan(
            draw=True,
            color=line_color,
            is_double=False,
            inner_linestyle=options.LANE_broken_linestyle,
            outer_linestyle=options.LANE_broken_linestyle,
        )

    if idx in (1, 5):
        # solid single
        return _LaneBoundaryDrawPlan(
            draw=True,
            color=line_color,
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    if idx in (2, 6):
        # solid double
        return _LaneBoundaryDrawPlan(
            draw=True,
            color=line_color,
            is_double=True,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    if idx == 4:
        # broken double yellow
        return _LaneBoundaryDrawPlan(
            draw=True,
            color=line_color,
            is_double=True,
            inner_linestyle=options.LANE_broken_linestyle,
            outer_linestyle=options.LANE_broken_linestyle,
        )

    # PASSING_DOUBLE_YELLOW
    # "차선 쪽(안쪽)은 점선, 바깥쪽은 실선"으로 표현
    return _LaneBoundaryDrawPlan(
        draw=True,
        color=line_color,
        is_double=True,
        inner_linestyle=options.LANE_broken_linestyle,
        outer_linestyle=options.LANE_solid_linestyle,
    )


def _compute_inward_normal_for_segment(
        boundary_p0: Array,  # shape: (2,)
        boundary_p1: Array,  # shape: (2,)
        center_p0: Array,  # shape: (2,)
        center_p1: Array,  # shape: (2,)
) -> Array:
    """선분의 법선 방향 중 '차선 중심 쪽'을 향하는 단위 벡터를 만든다.

    Args:
        boundary_p0: 경계 선분 시작점, shape (2,)
        boundary_p1: 경계 선분 끝점, shape (2,)
        center_p0: 중심 선분 시작점, shape (2,)
        center_p1: 중심 선분 끝점, shape (2,)

    Returns:
        inward_normal: shape (2,) 단위 벡터
    """
    seg = (boundary_p1 - boundary_p0).astype(np.float32)  # shape: (2,)
    seg_len = float(np.hypot(seg[0], seg[1]))
    if seg_len < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    # 한쪽 법선
    normal = np.array([-seg[1], seg[0]],
                      dtype=np.float32) / seg_len  # shape: (2,)

    boundary_mid = 0.5 * (boundary_p0 + boundary_p1)  # shape: (2,)
    center_mid = 0.5 * (center_p0 + center_p1)  # shape: (2,)
    to_center = (center_mid - boundary_mid).astype(np.float32)  # shape: (2,)

    if float(np.dot(normal, to_center)) < 0.0:
        normal = -normal
    return normal


def _draw_single_line_segment(
    ax: plt.Axes,
    p0: Array,  # shape: (2,)
    p1: Array,  # shape: (2,)
    color: str,
    line_width: float,
    linestyle: Any,
    zorder: int,
) -> None:
    """선분 1개를 실선/점선으로 그린다."""
    ax.plot(
        [float(p0[0]), float(p1[0])],
        [float(p0[1]), float(p1[1])],
        color=color,
        linewidth=line_width,
        linestyle=linestyle,
        zorder=zorder,
    )


def _draw_double_line_segment(
    ax: plt.Axes,
    p0: Array,  # shape: (2,)
    p1: Array,  # shape: (2,)
    inward_normal: Array,  # shape: (2,)
    sep_m: float,
    color: str,
    line_width: float,
    inner_linestyle: Any,
    outer_linestyle: Any,
    zorder: int,
) -> None:
    """이중선을 '안쪽(차선 쪽)' / '바깥쪽' 두 줄로 그린다."""
    half = 0.5 * float(sep_m)
    shift = inward_normal.astype(np.float32) * half  # shape: (2,)

    p0_in = p0 + shift
    p1_in = p1 + shift
    p0_out = p0 - shift
    p1_out = p1 - shift

    _draw_single_line_segment(ax, p0_in, p1_in, color, line_width,
                              inner_linestyle, zorder)
    _draw_single_line_segment(ax, p0_out, p1_out, color, line_width,
                              outer_linestyle, zorder)


def draw_road_edge_points(
    ax: plt.Axes,
    road_edge: Optional[Array],  # shape: (E, safety_len=10, 2)
    road_edge_type: Optional[Array],  # shape: (E, 3)
    options: DrawingOptions,
) -> None:
    """road_edge를 빨간색 점들로 그린다.

    - road_edge: (E,10,2) 점들
    - road_edge_type: (E,3) one-hot
      * UNKNOWN -> 'o'
      * BOUNDARY -> '*'
      * MEDIAN -> 'x'

    Args:
        ax: Matplotlib 축.
        road_edge: shape (E,10,2)
        road_edge_type: shape (E,3)
        options: 색/크기 옵션.
    """
    if not options.ROAD_draw_road_edge:
        return
    if road_edge is None:
        return

    road_edge = np.asarray(road_edge)  # (E,10,2)
    if road_edge.size == 0:
        return
    if road_edge.ndim != 3 or road_edge.shape[-1] != 2:
        raise ValueError(
            f"road_edge shape는 (E,10,2) 이어야 합니다. got {road_edge.shape}")

    E = int(road_edge.shape[0])

    if road_edge_type is not None:
        road_edge_type = np.asarray(road_edge_type)  # (E,3)
        if road_edge_type.ndim != 2 or road_edge_type.shape != (E, 3):
            raise ValueError(
                f"road_edge_type shape는 (E,3) 이어야 합니다. got {road_edge_type.shape}, E={E}"
            )

    eps = float(options.invalid_eps)

    for i in range(E):
        pts = road_edge[i]  # (10,2)
        valid_mask = np.any(np.abs(pts) > eps, axis=1)  # (10,)
        if not np.any(valid_mask):
            continue

        if road_edge_type is None:
            type_idx = 0
        else:
            row3 = road_edge_type[i].reshape(-1)  # (3,)
            type_idx = int(np.argmax(row3)) if float(np.sum(
                np.abs(row3))) > 0.0 else 0

        marker = "o"
        if type_idx == 1:
            marker = "*"
        elif type_idx == 2:
            marker = "x"

        xy = pts[valid_mask]  # (K,2)
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            marker=marker,
            s=float(options.ROAD_road_edge_marker_size),
            linewidths=float(options.ROAD_road_edge_line_width),
            edgecolors=options.ROAD_road_edge_color,
            facecolors="none",
            zorder=int(options.ROAD_road_edge_zorder),
        )


def draw_driveway_points(
    ax: plt.Axes,
    driveway: Optional[Array],  # shape: (D, safety_len=10, 2)
    options: DrawingOptions,
) -> None:
    """driveway를 흰색 테두리 다각형으로 그리고, 가운데에 'driveway' 글씨를 적는다.

    Args:
        ax: Matplotlib 축.
        driveway: shape (D,10,2)
        options: 색/두께/글씨 옵션.
    """
    if not options.DRIVEWAY_draw_driveway:
        return
    if driveway is None:
        return

    driveway = np.asarray(driveway)  # (D,10,2)
    if driveway.size == 0:
        return
    if driveway.ndim != 3 or driveway.shape[-1] != 2:
        raise ValueError(
            f"driveway shape는 (D,10,2) 이어야 합니다. got {driveway.shape}")

    # 기존 안전 폴리곤 함수 재사용(테두리 + 중앙 텍스트)
    draw_road_safety_polygons_with_text(
        ax=ax,
        road_safety_points=driveway,
        label_text=options.DRIVEWAY_text,
        line_color=options.DRIVEWAY_line_color,
        line_width=options.DRIVEWAY_line_width,
        text_color=options.DRIVEWAY_text_color,
        text_fontsize=options.DRIVEWAY_text_fontsize,
        options=options,
        zorder=options.DRIVEWAY_polygon_zorder,
    )


def draw_lane_boundaries(
        ax: plt.Axes,
        lanes: Array,  # (lane_num, lane_len, 12)
        agent_route_lane_order: Optional[Array],  # (max_agent_num, lane_num)
        options: DrawingOptions,
        draw_token_int_list: Optional[List[int]] = None,
        lane_type: Optional[Array] = None,  # (lane_num, 4)
        left_line_type: Optional[Array] = None,  # (lane_num, 10)
        right_line_type: Optional[Array] = None,  # (lane_num, 10)
) -> None:
    """차선 좌/우 경계를 그린다.

    입력 shape
      - lanes: (L, T, 12)
      - lane_type: (L, 4)  (선택)
      - left_line_type/right_line_type: (L, 10) (선택)

    동작 요약
      - lane_type이 있으면 경계 기본 색을 lane 종류에 맞게 바꿈
      - left/right_line_type이 있으면 흰/노란 + 점선/실선 + 이중선까지 반영
      - UNKNOWN이면 원래처럼 그리고, INVALID면 그리지 않음
    """
    if lanes is None or lanes.size == 0:
        return

    # route highlight 필터링(기존 로직 유지)
    if (draw_token_int_list is not None and
            agent_route_lane_order is not None and
            options.LANE_draw_npc_agent_route and
            options.LANE_npc_agent_route_draw_mode == "lane"):
        filtered_agent_route_lane_order = []
        for agent_idx in range(agent_route_lane_order.shape[0]):
            if agent_idx in draw_token_int_list:
                filtered_agent_route_lane_order.append(
                    agent_route_lane_order[agent_idx])
        agent_route_lane_order = np.array(filtered_agent_route_lane_order)
        if agent_route_lane_order.shape[0] == 0:
            agent_route_lane_order = None
    else:
        agent_route_lane_order = None

    L = int(lanes.shape[0])
    eps = float(options.invalid_eps)

    # shape 체크(있을 때만)
    if lane_type is not None:
        lane_type = np.asarray(lane_type)
        if lane_type.ndim != 2 or lane_type.shape != (L, 4):
            raise ValueError(
                f"lane_type shape는 (lane_num,4) 이어야 합니다. got {lane_type.shape}, L={L}"
            )

    if left_line_type is not None:
        left_line_type = np.asarray(left_line_type)
        if left_line_type.ndim != 2 or left_line_type.shape != (L, 10):
            raise ValueError(
                f"left_line_type shape는 (lane_num,10) 이어야 합니다. got {left_line_type.shape}, L={L}"
            )

    if right_line_type is not None:
        right_line_type = np.asarray(right_line_type)
        if right_line_type.ndim != 2 or right_line_type.shape != (L, 10):
            raise ValueError(
                f"right_line_type shape는 (lane_num,10) 이어야 합니다. got {right_line_type.shape}, L={L}"
            )

    for idx, lane_i in enumerate(lanes):  # lane_i: (lane_len, 12)
        center = lane_i[:, 0:2]  # (T,2)
        left_vec = lane_i[:, 4:6]  # (T,2)
        right_vec = lane_i[:, 6:8]  # (T,2)
        valid = np.any(np.abs(lane_i[:, :8]) > eps, axis=1)  # (T,)

        if center.shape[0] < 2:
            continue

        # lane_type 기반 기본 색
        lane_type_row4 = lane_type[idx] if lane_type is not None else None
        base_color = _lane_type_one_hot_to_color(lane_type_row4, options)

        # left/right line 계획
        left_plan = _make_lane_boundary_draw_plan(
            left_line_type[idx] if left_line_type is not None else None,
            default_color=base_color,
            options=options,
        )
        right_plan = _make_lane_boundary_draw_plan(
            right_line_type[idx] if right_line_type is not None else None,
            default_color=base_color,
            options=options,
        )

        for j in range(center.shape[0] - 1):
            if not (valid[j] and valid[j + 1]):
                continue

            c0 = center[j]
            c1 = center[j + 1]
            l0 = c0 + left_vec[j]
            l1 = c1 + left_vec[j + 1]
            r0 = c0 + right_vec[j]
            r1 = c1 + right_vec[j + 1]

            # left
            if left_plan.draw:
                if left_plan.is_double:
                    inward_n = _compute_inward_normal_for_segment(
                        l0, l1, c0, c1)  # (2,)
                    _draw_double_line_segment(
                        ax=ax,
                        p0=l0,
                        p1=l1,
                        inward_normal=inward_n,
                        sep_m=float(options.LANE_double_line_sep_m),
                        color=left_plan.color,
                        line_width=float(options.LANE_boundary_width),
                        inner_linestyle=left_plan.inner_linestyle,
                        outer_linestyle=left_plan.outer_linestyle,
                        zorder=1,
                    )
                else:
                    _draw_single_line_segment(
                        ax=ax,
                        p0=l0,
                        p1=l1,
                        color=left_plan.color,
                        line_width=float(options.LANE_boundary_width),
                        linestyle=left_plan.inner_linestyle,
                        zorder=1,
                    )

            # right
            if right_plan.draw:
                if right_plan.is_double:
                    inward_n = _compute_inward_normal_for_segment(
                        r0, r1, c0, c1)  # (2,)
                    _draw_double_line_segment(
                        ax=ax,
                        p0=r0,
                        p1=r1,
                        inward_normal=inward_n,
                        sep_m=float(options.LANE_double_line_sep_m),
                        color=right_plan.color,
                        line_width=float(options.LANE_boundary_width),
                        inner_linestyle=right_plan.inner_linestyle,
                        outer_linestyle=right_plan.outer_linestyle,
                        zorder=1,
                    )
                else:
                    _draw_single_line_segment(
                        ax=ax,
                        p0=r0,
                        p1=r1,
                        color=right_plan.color,
                        line_width=float(options.LANE_boundary_width),
                        linestyle=right_plan.inner_linestyle,
                        zorder=1,
                    )

        # ===== 기존 route 강조(필요 시) =====
        if agent_route_lane_order is not None:
            try:
                agent_route_a_lane_order = agent_route_lane_order[:,
                                                                  idx]  # (filtered_agent_num,)
            except Exception:
                raise ValueError(
                    f"agent_route_lane_order shape {agent_route_lane_order.shape} incompatible with lane idx {idx}"
                )
            has_route_mask = (agent_route_a_lane_order != -1)
            if np.any(has_route_mask):
                color = CYAN
                for j in range(center.shape[0] - 1):
                    if not (valid[j] and valid[j + 1]):
                        continue
                    c0 = center[j]
                    c1 = center[j + 1]
                    l0 = c0 + left_vec[j]
                    l1 = c1 + left_vec[j + 1]
                    r0 = c0 + right_vec[j]
                    r1 = c1 + right_vec[j + 1]

                    # line_type이 INVALID면 route도 그리지 않음
                    if left_plan.draw:
                        ax.plot([l0[0], l1[0]], [l0[1], l1[1]],
                                color=color,
                                linewidth=options.LANE_boundary_width,
                                zorder=2)
                    if right_plan.draw:
                        ax.plot([r0[0], r1[0]], [r0[1], r1[1]],
                                color=color,
                                linewidth=options.LANE_boundary_width,
                                zorder=2)


def draw_lane_centerlines(
    ax: plt.Axes,
    lanes: Array,  # (lane_num, lane_len, 12)
    lanes_speed_limit: Array,  #  (lane_num, 1)
    lanes_has_speed_limit: Array,  # (lane_num, 1)
    options: DrawingOptions,
    agent_route_lane_order: Optional[Array] = None,  # (max_agent_num, lane_num)
    draw_token_int_list: Optional[List[int]] = None,
) -> None:
    """센터라인을 점선으로 그리거나, agent_route_lane_order가 주어지면 에이전트-차선 매핑을 텍스트로 표기한다.

    동작 모드
    ----------
    1) agent_route_lane_order is None:
        - 기존 로직 유지: 차선 센터라인을 **점선**으로 그림.
        - 양 끝점이 모두 유효한 구간만 선분을 그림.
        - 색은 lane의 signal(0~3: green/yellow/red/unknown)에 따라 사용.
        - (추가) options.LANE_draw_vel_limit=True 이고 lanes_has_speed_limit이 True인 차선에는
          해당 lanes_speed_limit 값을 km/h로 변환하여 센터라인 위에 숫자를 표시.

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
        shape = (max_agent_num, lane_num), 각 [i, j] = 해당 에이전트 i에게서
        차선 j의 '가까운 순서 랭크(0,1,2,...)'; 경로에 없으면 -1.
    """
    if lanes is None or lanes.size == 0:
        return
    if not (options.LANE_draw_npc_agent_route == True and
            options.LANE_npc_agent_route_draw_mode == "centerline"):
        agent_route_lane_order = None
    eps = options.invalid_eps
    lane_num = lanes.shape[0]

    # 추가: 속도 제한 텍스트를 그릴지 여부 플래그
    use_speed_limit_label = (agent_route_lane_order is None and
                             options.LANE_draw_vel_limit and
                             (lanes_speed_limit is not None) and
                             (lanes_has_speed_limit is not None))

    # ────────────── (B) 텍스트 표기 모드 ──────────────
    if agent_route_lane_order is not None:
        if draw_token_int_list is None:
            draw_all = True
        else:
            draw_all = False
        vstep = 0.3  # 같은 위치에 여러 개 쌓을 때 세로 간격(미터)

        # 각 차선 lane_idx 순회
        for lane_idx in range(lane_num):
            lane_j = lanes[lane_idx]  # (lane_len, 12)
            lane_j_center = lane_j[:, 0:2]  # (lane_len, 2)
            # (lane_len,)
            lane_point_valid_mask = np.any(np.abs(lane_j[:, :8]) > eps, axis=1)

            # 이 차선을 자신의 경로에 포함하는 모든 agent i와 그 rank
            # agent_route_lane_order: (max_agent_num, lane_num),
            ranks_j: Array = agent_route_lane_order[:,
                                                    lane_idx]  # (max_agent_num,)
            valid_agent_idxs: Array = np.nonzero(ranks_j >= 0)[0]  # (K,)
            if valid_agent_idxs.size == 0:
                continue

            # 유효 포인트마다 텍스트 찍기
            # (여러 agent가 있으면 위로 살짝씩 띄워서 겹침 완화)
            for point_idx in range(lane_j_center.shape[0]):
                if point_idx % 4 != 0:
                    continue
                if not lane_point_valid_mask[point_idx]:
                    continue
                point_x, point_y = float(lane_j_center[point_idx, 0]), float(
                    lane_j_center[point_idx, 1])
                for count, agent_idx in enumerate(valid_agent_idxs):
                    if (not draw_all) and (agent_idx
                                           not in draw_token_int_list):
                        continue
                    rank_ij = int(ranks_j[int(agent_idx)])
                    # label = f"{int(agent_idx)}--{rank_ij}"  # "에이전트인덱스:해당차선랭크"
                    label = f"{rank_ij}"  # "에이전트인덱스:해당차선랭크"
                    # label = f"{int(agent_idx)}"  # "에이전트인덱스"
                    ax.text(
                        point_x,
                        point_y + vstep * count,  # 위로 살짝씩 쌓기
                        label,
                        color=options.LANE_route_agent_index_color,
                        fontsize=options.LANE_AGENT_index_fontsize,
                        ha="center",
                        va="bottom",
                        zorder=3,
                    )
        return  # 텍스트 모드에서는 선을 그리지 않음

    # ────────────── (A) 기존 점선 센터라인 모드 ──────────────
    for lane_idx, lane_i in enumerate(lanes):  # 추가: enumerate로 lane_idx 사용
        center = lane_i[:, 0:2]
        signals = lane_i[:, 8:12]
        valid = np.any(np.abs(lane_i[:, :8]) > eps, axis=1)  # (lane_len,)

        if center.shape[0] < 2:
            continue

        # 추가: 이 lane에 속도 제한 정보가 있는지 확인
        has_speed_limit = False  # 추가
        speed_kmh = None  # 추가
        if use_speed_limit_label:  # 추가
            flag_val = lanes_has_speed_limit[lane_idx]  # 추가
            if np.ndim(flag_val) > 0:  # 추가
                flag_val = flag_val[0]  # 추가
            if bool(flag_val):  # 추가
                has_speed_limit = True  # 추가
                speed_val = lanes_speed_limit[lane_idx]  # 추가
                if np.ndim(speed_val) > 0:  # 추가
                    speed_val = speed_val[0]  # 추가
                # m/s 로 들어왔다고 가정하고 km/h 로 변환  # 추가
                speed_kmh = float(speed_val) * 3.6  # 추가

        state_idx = np.argmax(signals, axis=1)  # (lane_len,)
        for j in range(center.shape[0] - 1):
            if not (valid[j] and valid[j + 1]):
                continue
            c0, c1 = center[j], center[j + 1]
            color = options.LANE_signal_colors.get(int(state_idx[j]), GRAY)
            ax.plot([c0[0], c1[0]], [c0[1], c1[1]],
                    color=color,
                    linewidth=1.2,
                    linestyle=(0, (4, 4)),
                    zorder=2)

        # 추가: 속도 제한이 있는 lane이면 센터라인 중간쯤에 숫자(km/h)를 표시
        if use_speed_limit_label and has_speed_limit and (speed_kmh
                                                          is not None):  # 추가
            valid_indices = np.nonzero(valid)[0]  # 추가
            if valid_indices.size > 0:  # 추가
                mid_idx = int(valid_indices[len(valid_indices) // 2])  # 추가
                px, py = float(center[mid_idx, 0]), float(center[mid_idx,
                                                                 1])  # 추가
                if has_speed_limit and (speed_kmh is not None):  # 추가
                    label = f"{speed_kmh:.1f}"  # 예: "50.0" km/h  # 추가
                else:  # 추가
                    label = "none"  # 속도 제한이 없는 도로  # 추가
                ax.text(
                    px,
                    py,
                    label,  # 소수 첫째 자리까지 km/h로 표시  # 추가
                    color=options.LANE_speed_color,  # 추가
                    fontsize=options.LANE_speed_fontsize,  # 추가
                    ha="center",
                    va="center",
                    zorder=3,
                )  # 추가


def draw_neighbor_past(
    ax: plt.Axes,
    neighbor_agents_past: Array,
    options: DrawingOptions,
    neighbor_role: Optional[Array] = None,  # NEW: (A,2) bool
    draw_token_int_list: Optional[List[int]] = None,
) -> None:
    """이웃 에이전트 과거 시퀀스를 클래스별 색상으로 그림. invalid 스텝은 스킵."""
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    eps = options.invalid_eps
    max_agent_num, time_len, feat_dim = neighbor_agents_past.shape
    if feat_dim != 11:
        raise ValueError("neighbor_agents_past의 마지막 차원은 11이어야 합니다.")
    current_t = time_len - 1

    for agent_idx in range(max_agent_num):
        if (draw_token_int_list
                is not None) and (agent_idx not in draw_token_int_list
                                 ) or options.NEI_draw_neighbor_only_current:
            draw_all_time_for_target = False
        else:
            draw_all_time_for_target = True
        track = neighbor_agents_past[agent_idx]  # (T, 11)
        for t in range(time_len):
            if not draw_all_time_for_target and t != current_t:
                continue
            row = track[t]
            if not is_valid_agent_row(row, eps):
                continue

            x, y = float(row[0]), float(row[1])
            c, s = float(row[2]), float(row[3])
            vx, vy = float(row[4]), float(row[5])
            W, L = float(row[6]), float(row[7])
            cls = infer_agent_class(row[8:11])
            neighbor_cls_style = options.NEI_neighbor_style[cls]

            fill_color = neighbor_cls_style[
                "fill_color"] if t == current_t else None
            fill_alpha = neighbor_cls_style[
                "fill_alpha"] if t == current_t else None
            # NEW: 현재 시점만 neighbor_role에 따라 테두리 색 변경
            edge_color = neighbor_cls_style["line_color"]
            line_width = neighbor_cls_style["line_width"]
            if t == current_t:
                line_width = 0.6
                edge_color = _get_neighbor_edge_color_for_current_t(
                    agent_idx=agent_idx,
                    neighbor_role=neighbor_role,
                    default_edge_color=neighbor_cls_style["line_color"],
                    options=options,
                )
            corners = oriented_box_corners(x, y, c, s, L, W)
            add_polygon(
                ax,
                corners,
                edge_color=edge_color,  # CHANGED
                line_width=line_width,
                fill_color=fill_color,
                fill_alpha=fill_alpha,
                zorder=5 if t == current_t else 4,
            )
            add_heading_line(
                ax,
                x,
                y,
                c,
                s,
                nominal_length=L * options.COMMON_heading_line_scale,
                color=edge_color,  # CHANGED
                line_width=neighbor_cls_style["line_width"],
                zorder=6 if t == current_t else 4,
            )
            if options.NEI_draw_velocity_arrow and t == current_t:
                add_velocity_arrow(
                    ax,
                    x,
                    y,
                    vx,
                    vy,
                    length_m=options.COMMON_vel_arrow_len_m,
                    line_color=neighbor_cls_style["velocity_line_color"],
                    line_width=neighbor_cls_style["velocity_line_width"],
                    zorder=7 if t == current_t else 4,
                    t=t)
            if options.NEI_draw_velocity_text:
                if t % 20 == 0:
                    offset = 5
                else:
                    offset = 3
                if t % 10 != 0:
                    continue
                # [추가] 속도 크기 텍스트(km/h) - 모든 과거 지점
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                ax.text(
                    x,  # 점 위쪽에 표기
                    y + options.NEI_vel_text_y_offset * offset,
                    f"{speed_kmh:.1f}",
                    color=options.NEI_vel_text_color,
                    fontsize=options.NEI_vel_text_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=30,
                    clip_on=True,  # tight 저장 시 bbox 폭주 방지
                )


def annotate_neighbor_indices_for_past(ax: plt.Axes, neighbor_track_token: List[
    Optional[str]], neighbor_agents_past: Array,
                                       options: DrawingOptions) -> None:
    """neighbor_agents_past (max_agent_num, T=21, 11)의 '현재 상태'(마지막 스텝) 근처에
    에이전트 인덱스(0..max_agent_num-1)를 흰색 텍스트로 표기.
    - invalid 스텝은 스킵
    """
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[-1] != 11:
        raise ValueError(
            "neighbor_agents_past의 shape은 (max_agent_num, 21, 11) 이어야 합니다.")

    eps = options.invalid_eps
    max_agent_num, time_len, feat_dim = neighbor_agents_past.shape
    current_t = time_len - 1
    for agent_idx in range(max_agent_num):
        track_token = neighbor_track_token[agent_idx]
        row = neighbor_agents_past[agent_idx, current_t]  # (11,)
        if not is_valid_agent_row(row, eps):
            assert track_token is None
            continue
        x, y = float(row[0]), float(row[1])
        ax.text(
            x + options.NEI_past_token_place_offset_m,
            y + options.NEI_past_token_place_offset_m,
            str(track_token),  #[:5],
            color=options.NEI_past_token_color,
            fontsize=options.NEI_past_token_fontsize,
            ha='left',
            va='bottom',
            zorder=30)


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
        if options.EGO_draw_ego_only_current and t != current_t:
            continue
        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])
        vx, vy = float(row[4]), float(row[5])
        W, L = float(row[6]), float(row[7])

        fill_color = options.EGO_past_style[
            "fill_color"] if t == current_t else None
        fill_alpha = options.EGO_past_style[
            "fill_alpha_current"] if t == current_t else None

        corners = oriented_box_corners(x, y, c, s, L, W)
        add_polygon(ax,
                    corners,
                    edge_color=options.EGO_past_style["line_color"],
                    line_width=options.EGO_past_style["line_width"],
                    fill_color=fill_color,
                    fill_alpha=fill_alpha,
                    zorder=20 if t == current_t else 9)
        add_heading_line(ax,
                         x,
                         y,
                         c,
                         s,
                         nominal_length=L * options.COMMON_heading_line_scale,
                         color=options.EGO_past_style["line_color"],
                         line_width=options.EGO_past_style["line_width"],
                         zorder=21 if t == current_t else 9)
        if options.EGO_draw_ego_past_vel and t == current_t:
            add_velocity_arrow(ax,
                               x,
                               y,
                               vx,
                               vy,
                               length_m=options.COMMON_vel_arrow_len_m,
                               line_color=options.EGO_past_style["line_color"],
                               line_width=options.EGO_past_style["line_width"],
                               zorder=22 if t == current_t else 9,
                               t=t)


def draw_ego_agent_next_11_dim(ax: plt.Axes, ego_agent_next_11_dim: Array,
                               options: DrawingOptions) -> None:
    """이고 차량 **예측** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if ego_agent_next_11_dim is None or ego_agent_next_11_dim.size == 0:
        return
    interpol_num, feat_dim = ego_agent_next_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_agent_next_11_dim의 마지막 차원은 11이어야 합니다.")
    if options.EGO_future_traj_draw_mode == "rectangle":
        draw_token_trajectory_rects_unfilled(
            ax,
            ego_agent_next_11_dim,
            options.EGO_next_11_dim_style,
            options,
            zorder=24,
            draw_velocity=options.EGO_draw_ego_agent_next_11_vel)
    elif options.EGO_future_traj_draw_mode in ["arrow", "point", "line"]:
        draw_token_trajectory_non_rects(
            ax,
            ego_agent_next_11_dim,
            options.EGO_next_11_dim_style,
            options,
            mode=options.EGO_future_traj_draw_mode,
            zorder=24,
            draw_velocity=options.EGO_draw_ego_agent_next_11_vel,
        )


def draw_planner_future_11_dim(ax: plt.Axes, planner_future_11_dim: Array,
                               options: DrawingOptions) -> None:
    """이고 차량 **GT 미래** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if planner_future_11_dim is None or planner_future_11_dim.size == 0:
        return
    future_len, feat_dim = planner_future_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_future_gt_11_dim의 마지막 차원은 11이어야 합니다.")
    if options.EGO_future_traj_draw_mode == "rectangle":
        draw_token_trajectory_rects_unfilled(
            ax,
            planner_future_11_dim,
            options.EGO_planner_future_11_style,
            options,
            zorder=24,
            draw_velocity=options.EGO_draw_planner_velocity)
    elif options.EGO_future_traj_draw_mode in ["arrow", "point", "line"]:
        draw_token_trajectory_non_rects(
            ax,
            planner_future_11_dim,
            options.EGO_planner_future_11_style,
            options,
            mode=options.EGO_future_traj_draw_mode,
            zorder=24,
            draw_velocity=options.EGO_draw_planner_velocity)


def draw_ego_future_gt_11_dim(ax: plt.Axes, ego_future_gt_11_dim: Array,
                              options: DrawingOptions) -> None:
    """이고 차량 **GT 미래** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if ego_future_gt_11_dim is None or ego_future_gt_11_dim.size == 0:
        return
    eps = options.invalid_eps
    future_len, feat_dim = ego_future_gt_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_agent_next_11_dim의 마지막 차원은 11이어야 합니다.")
    if options.EGO_future_traj_draw_mode == "rectangle":
        draw_token_trajectory_rects_unfilled(
            ax,
            ego_future_gt_11_dim,
            options.EGO_future_gt_11_style,
            options,
            zorder=24,
            draw_velocity=options.EGO_draw_future_11_velocity)
    elif options.EGO_future_traj_draw_mode in ["arrow", "point", "line"]:
        draw_token_trajectory_non_rects(
            ax,
            ego_future_gt_11_dim,
            options.EGO_future_gt_11_style,
            options,
            mode=options.EGO_future_traj_draw_mode,
            zorder=24,
            draw_velocity=options.EGO_draw_future_11_velocity,
        )


def draw_ego_radius_circle(
    ax: plt.Axes,
    draw_option: DrawingOptions,
) -> None:
    """ego 를 원점(0,0)으로 보고, 그 주변에 빨간색 테두리 원을 그린다.

    이 그림은 "ego 기준 좌표"라고 가정하고,
    ego 위치를 (0, 0) 라고 보고 그 자리 중심으로 원을 그린다.

    Args:
        ax: Matplotlib 축 객체.
        draw_option: 원을 그릴지 여부와 반경/색/두께 정보가 들어 있는 옵션.
    """
    # 옵션이 꺼져 있으면 아무 것도 그리지 않음
    if not draw_option.EGO_draw_radius_circle:
        return

    radius_m: float = float(draw_option.EGO_radius_circle_m)

    # 중심: (0, 0), 반지름: radius_m
    circle: Circle = Circle(
        (0.0, 0.0),  # 중심 좌표 (ego 기준 좌표계)
        radius_m,  # 반지름 [m]
        fill=False,  # 안은 비우고
        edgecolor=draw_option.EGO_radius_circle_edge_color,
        linewidth=draw_option.EGO_radius_circle_line_width,
        linestyle="--",
        zorder=3,  # 차선(1~2) 위, 차량(5~) 아래 정도
    )
    ax.add_patch(circle)


# [Add]
def draw_diff_future_gen_refined_traj(
        ax: plt.Axes,
        diff_token_to_interp_np_traj_wrt_ego: Optional[Dict[str, np.ndarray]],
        options: DrawingOptions,
        diff_token_to_next_wp_wrt_ego: Optional[Dict[str, np.ndarray]],
        draw_token_list: Optional[List[str]] = None) -> None:
    """토큰별 refined 궤적(연속 다스텝)과 **신규 waypoint(단일 11차원)**를 함께 그린다.

    - refined: DIFF_future_gen_refined_style(빨강)로 연속 박스 + 첫 유효 포인트에 idx(빨강, 아래쪽 오프셋)
    - new waypoint: options.DIFF_new_waypoint_style(주황) 테두리 박스 + idx(주황, 오른쪽 오프셋)

    Args:
        ax: Matplotlib 축.
        diff_token_to_interp_np_traj_wrt_ego: Dict[str, np.ndarray] | None
            각 value: shape = (1 + future_len, 11)
            row(11,) = [x, y, cos, sin, vx, vy, length, width, onehot(3,)]
        options: 렌더링 옵션.
        diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] | None
            각 value: shape = (11,) (단일 스텝)
    """
    if not diff_token_to_interp_np_traj_wrt_ego and not diff_token_to_next_wp_wrt_ego:
        return

    eps = options.invalid_eps

    # 표시할 토큰 순서: refined의 key 순서 우선, new_waypoint에만 있는 토큰은 뒤에 추가
    ordered_tokens: List[str] = []
    if diff_token_to_interp_np_traj_wrt_ego:
        ordered_tokens.extend(list(diff_token_to_interp_np_traj_wrt_ego.keys()))

    for idx, token in enumerate(ordered_tokens):
        if draw_token_list is not None and token not in draw_token_list:
            continue
        # ── (A) refined 연속 궤적(빨강) ───────────────────────────────
        if diff_token_to_interp_np_traj_wrt_ego and (
                token in diff_token_to_interp_np_traj_wrt_ego):
            interp_np_traj = diff_token_to_interp_np_traj_wrt_ego[
                token]  # (1+future_len, 11)
            if interp_np_traj is not None and interp_np_traj.size > 0:
                seq_len = interp_np_traj.shape[0]
                label_drawn = False
                for t in range(seq_len):
                    row = interp_np_traj[t]  # (11,)
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
                        edge_color=options.
                        DIFF_future_gen_refined_style["line_color"],
                        line_width=options.
                        DIFF_future_gen_refined_style["line_width"],
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
                        nominal_length=L * options.COMMON_heading_line_scale,
                        color=options.
                        DIFF_future_gen_refined_style["line_color"],
                        line_width=options.
                        DIFF_future_gen_refined_style["line_width"],
                        zorder=24,
                    )
                    if options.DIFF_draw_future_gen_refined_velocity:
                        speed_kmh = float(np.hypot(vx, vy)) * 3.6
                        if t % 20 == 0:
                            offset = 5
                        else:
                            offset = 3
                        if t % 10 != 0:
                            continue
                        ax.text(
                            x,
                            y +
                            options.DIFF_future_gen_refined_velocity_offset_m *
                            offset,
                            f"{speed_kmh:.1f}",
                            color=options.
                            DIFF_future_gen_refined_style["line_color"],
                            fontsize=options.
                            DIFF_future_gen_refined_velocity_font_size,
                            ha="center",
                            va="bottom",
                            zorder=25,
                            clip_on=True,
                        )
                    # 첫 유효 포인트에 빨간색 idx(아래쪽 오프셋)
                    # if not label_drawn:
                    #     ax.text(
                    #         x,
                    #         y - options.DIFF_future_gen_refined_token_offset_m,
                    #         str(token)[:5],
                    #         color=options.DIFF_future_gen_refined_style["line_color"],
                    #         fontsize=options.DIFF_future_gen_refined_velocity_font_size,
                    #         ha="center",
                    #         va="top",
                    #         zorder=25,
                    #     )
                    #     label_drawn = True

        # ── (B) 신규 waypoint(주황) ─────────────────────────────────
        if diff_token_to_next_wp_wrt_ego and (token
                                              in diff_token_to_next_wp_wrt_ego):
            wp = diff_token_to_next_wp_wrt_ego[token]  # (11,)
            if wp is None:
                continue
            wp = np.asarray(wp)
            if wp.ndim == 2:
                wp = wp.squeeze()
            if wp.ndim != 1 or wp.shape[0] != 11:
                raise ValueError(
                    "token_to_new_waypoint_array의 각 value는 shape (11,) 이어야 합니다."
                )

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
                edge_color=options.DIFF_new_waypoint_style["line_color"],
                line_width=options.DIFF_new_waypoint_style["line_width"],
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
                nominal_length=L * options.COMMON_heading_line_scale,
                color=options.DIFF_new_waypoint_style["line_color"],
                line_width=options.DIFF_new_waypoint_style["line_width"],
                zorder=28,
            )
            if options.DIFF_draw_future_gen_refined_velocity:
                add_velocity_arrow(
                    ax,
                    x,
                    y,
                    vx,
                    vy,
                    length_m=options.COMMON_vel_arrow_len_m,
                    line_color=options.DIFF_new_waypoint_style["line_color"],
                    line_width=options.DIFF_new_waypoint_style["line_width"],
                    line_alpha=1.0,
                    zorder=28,
                    t=0,
                )
                # [추가] 속도 크기 텍스트(km/h) - next waypoint 1점 (색: 주황)
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                ax.text(
                    x,
                    y + options.DIFF_new_waypoint_vel_text_y_offset_m,
                    f"{speed_kmh:.1f}",
                    color=options.DIFF_new_waypoint_style["line_color"],
                    fontsize=options.DIFF_future_gen_refined_velocity_font_size,
                    ha="center",
                    va="bottom",
                    zorder=29,
                    clip_on=True,
                )

            # 번호 라벨: 주황색, "오른쪽"으로 살짝 이동
            # ax.text(
            #     x + options.DIFF_new_waypoint_vel_token_x_offset_m,
            #     y,
            #     str(token)[:5],
            #     color=options.DIFF_new_waypoint_style["line_color"],
            #     fontsize=options.DIFF_future_gen_refined_velocity_font_size,
            #     ha="left",
            #     va="center",
            #     zorder=29,
            # )


from typing import Optional, Literal

from typing import Optional, Dict, List, Tuple, Any


def draw_token_trajectory_rects_unfilled(
    ax: plt.Axes,
    traj_11: Array,
    style: Dict[str, Any],
    options: DrawingOptions,
    zorder: int,
    draw_velocity: Optional[bool] = None,
) -> Optional[Tuple[float, float]]:
    """(T, 11) 궤적을 '속이 비어 있는 사각형 + 헤딩선'으로 렌더링.

    Args:
        ax: Matplotlib 축.
        traj_11: (T, 11) 배열. row = [x, y, cos, sin, vx, vy, width, length, onehot(3,)]
                 ※ width=row[6], length=row[7]이며, 사각형 그릴 때 (length, width) 순서에 유의.
        style: {"line_color": str, "line_width": float} 키 사용.
        options: DrawingOptions (invalid_eps, COMMON_heading_line_scale 등 사용).
        zorder: matplotlib z-order.

    Returns:
        Optional[(x0, y0)]: 첫 번째 **유효** 프레임의 (x, y). 없으면 None.
    """
    if traj_11 is None or traj_11.size == 0:
        return None
    arr = np.asarray(traj_11)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 11:
        # 요청 스펙: (T,11)만 지원
        return None

    eps = options.invalid_eps
    first_valid_xy: Optional[Tuple[float, float]] = None

    for t in range(arr.shape[0]):
        row = arr[t]  # (11,)
        if not is_valid_agent_row(row, eps):
            continue

        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])
        vx, vy = float(row[4]), float(row[5])
        W, L = float(row[6]), float(row[7])  # 주의: 저장은 (W, L) 순서, draw는 (L, W)

        corners = oriented_box_corners(x, y, c, s, L, W)
        add_polygon(
            ax,
            corners,
            edge_color=style["line_color"],
            line_width=style["line_width"],
            fill_color=None,  # 속 비우기
            fill_alpha=None,
            zorder=zorder,
        )
        add_heading_line(
            ax,
            x,
            y,
            c,
            s,
            nominal_length=L * options.COMMON_heading_line_scale,
            color=style["line_color"],
            line_width=style["line_width"],
            zorder=zorder,
        )
        if draw_velocity is True:
            # draw_token_trajectory_rects_unfilled
            add_velocity_arrow(
                ax,
                x,
                y,
                vx,
                vy,
                length_m=options.COMMON_vel_arrow_len_m,
                line_color=style["velocity_line_color"],
                line_width=style["velocity_line_width"],
                line_alpha=style["velocity_line_alpha"],
                zorder=zorder,
                t=t,
            )

        if first_valid_xy is None:
            first_valid_xy = (x, y)

    return first_valid_xy


def draw_token_trajectory_non_rects(
        ax: plt.Axes,
        traj_11: Array,
        style: Dict[str, Any],
        options: DrawingOptions,
        mode: str,
        zorder: int,
        draw_velocity: bool = False,
        text_y_offset: float = 0.5,
        vel_text_fontsize: int = 2) -> Optional[Tuple[float, float]]:
    if traj_11 is None or traj_11.size == 0:
        return None
    if traj_11.ndim != 2 or traj_11.shape[1] != 11:
        raise ValueError(
            "token_to_future_traj_wrt_ego의 각 value는 (future_len, 4)이어야 합니다.")
    eps = options.invalid_eps
    first_valid_xy: Optional[Tuple[float, float]] = None
    traj = traj_11[:, :4]  # (future_len, 4)
    vx = traj_11[:, 4]  # (future_len)
    vy = traj_11[:, 5]  # shape: (future_len,)
    future_len = traj.shape[0]
    for t in range(future_len):
        row = traj[t]  # (4,) = [x, y, cos, sin]
        if not is_valid_token_row(row, eps):
            continue
        x, y = float(row[0]), float(row[1])
        c, s = float(row[2]), float(row[3])

        if draw_velocity:
            if t % 20 == 0:
                offset = 5
            else:
                offset = 3
            if t % 10 == 0:
                # [추가] 속도 크기 텍스트(km/h) - 모든 과거 지점
                speed_kmh = float(np.hypot(vx[t], vy[t])) * 3.6
                ax.text(
                    x,  # 점 위쪽에 표기
                    y + text_y_offset * offset,
                    f"{speed_kmh:.1f}",
                    color=style["line_color"],
                    fontsize=vel_text_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=30,
                    clip_on=True,  # tight 저장 시 bbox 폭주 방지
                )

        if mode == "arrow":
            # 방향 벡터 (c, s)를 정규화하여 고정 길이(옵션) 화살표
            add_velocity_arrow(
                ax,
                x,
                y,
                c,
                s,
                length_m=options.DIFF_future_gen_traj_arrow_len_m,
                line_color=style["line_color"],
                line_width=style["line_width"],
                zorder=zorder,
                t=t,
            )
        elif mode == "point":
            # 점만 표시(방향 정보 사용하지 않음)
            ax.plot(
                x,
                y,
                marker=options.DIFF_future_gen_traj_point_marker,
                markersize=options.DIFF_future_gen_traj_point_marker_size,
                linestyle="None",
                color=style["line_color"],
                zorder=zorder,
            )
        elif mode == "line":
            if t < future_len - 1:
                next_row = traj[t + 1]
                if not is_valid_token_row(next_row, eps):
                    continue
                next_x, next_y = float(next_row[0]), float(next_row[1])
                ax.plot(
                    [x, next_x],
                    [y, next_y],
                    linestyle="-",
                    color=style["line_color"],
                    linewidth=style["line_width"],
                    zorder=zorder,
                )

        #  시작 포인트(t==0)에 토큰 식별 라벨(흰색)을 화살표/점 바로 아래에 표기
        if first_valid_xy is None:
            first_valid_xy = (x, y)

    return first_valid_xy


def draw_traj_dict_as_unfilled_rects(
    ax: plt.Axes,
    token_to_traj_11: Optional[Dict[str, Array]],
    style: Dict[str, Any],
    options: DrawingOptions,
    zorder: int,
    draw_token_list: Optional[List[str]] = None,
    annotate_token: bool = False,
    annotate_fontsize: Optional[int] = None,
    annotate_offset: float = 0.0,
    draw_velocity: bool = False,
) -> None:
    """Dict[str, (T,11)]를 '속 빈 사각형'으로 렌더링하고 필요 시 토큰 라벨을 추가.

    Args:
        ax: Matplotlib 축.
        token_to_traj_11: Dict[token, (T,11)].
        style: {"line_color": str, "line_width": float}.
        options: DrawingOptions.
        zorder: z-order.
        draw_token_list: 필터링할 토큰 목록. None이면 전체.
        annotate_token: True면 각 토큰의 **첫 유효 프레임** 위치에 토큰 문자열 라벨을 표시.
        annotate_fontsize: 라벨 폰트 크기. None이면 options.DIFF_future_gt_3_dim_token_fontsize 사용.
    """
    if not token_to_traj_11:
        return

    if annotate_fontsize is None:
        annotate_fontsize = options.DIFF_future_gt_3_dim_token_fontsize

    for token, traj in token_to_traj_11.items():
        if draw_token_list is not None and token not in draw_token_list:
            continue

        first_xy = draw_token_trajectory_rects_unfilled(
            ax=ax,
            traj_11=traj,
            style=style,
            options=options,
            zorder=zorder,
            draw_velocity=draw_velocity,
        )
        if annotate_token and (first_xy is not None):
            fx, fy = first_xy
            ax.text(
                fx,
                fy - annotate_offset,
                str(token)[:5],
                color=style["line_color"],
                fontsize=annotate_fontsize,
                ha="center",
                va="top",
                zorder=zorder + 1,
                clip_on=True,
            )


#
def draw_traj_dict_as_non_square(
    ax: plt.Axes,
    token_to_traj_11: Optional[TokenTrajDict],
    style: Dict[str, Any],
    options: DrawingOptions,
    draw_token_list: Optional[List[str]] = None,
    annotate_token: bool = False,
    annotate_fontsize: Optional[int] = None,
    annotate_offset: float = 0.0,
    draw_velocity: bool = False,
) -> None:
    """토큰 기준 미래 포즈를 '화살표(방향 포함)' 또는 '점(방향 미사용)'으로 그림.

    Args:
        ax: Matplotlib 축.
        token_to_traj_11: Dict[str, np.ndarray] | None
            각 value: shape (future_len, 4) = [x, y, cos(yaw), sin(yaw)]
            - invalid 규칙: 4값 모두 0(±eps) → 스킵
        options: DrawingOptions
            - DIFF_future_traj_draw_mode: 'arrow' | 'point'
            - DIFF_future_gen_traj_point_marker, DIFF_future_gen_traj_point_marker_size
            - DIFF_future_gen_traj_arrow_len_m
        draw_mode: Optional['arrow' | 'point']
            - 우선순위: draw_mode 인자(있으면) > options.DIFF_future_traj_draw_mode(없으면 'arrow')

    동작:
        - 'arrow' 모드:
            (x,y)에서 (cos,sin) 방향으로 고정 길이(options.DIFF_future_gen_traj_arrow_len_m) 화살표
        - 'point' 모드:
            (x,y) 위치에 포인트만 표시(방향 미사용)

    Note:
        - t==0 (각 토큰의 첫 포인트)에는 토큰 문자열을 살짝 아래(y-오프셋)에 표시.
    """
    if not token_to_traj_11:
        return

    mode = options.DIFF_future_traj_draw_mode
    if mode not in {"arrow", "point", "line"}:
        raise ValueError(
            f"Unsupported draw_mode: {mode}. Use 'arrow' or 'point' or 'line'.")

    for idx, (token, traj_11) in enumerate(token_to_traj_11.items()):  # [ADD]
        if draw_token_list is not None and token not in draw_token_list:
            continue
        first_xy = draw_token_trajectory_non_rects(
            ax,
            traj_11,
            style,
            options,
            mode,
            zorder=20,
            draw_velocity=draw_velocity,
        )
        if annotate_token and (first_xy is not None):
            fx, fy = first_xy
            ax.text(
                fx,
                fy - annotate_offset,
                str(token),
                color=style["token_color"],
                fontsize=annotate_fontsize,
                ha="center",
                va="top",
                zorder=24,
            )


def draw_diff_future_traj_w_square(
    ax: plt.Axes,
    diff_token_to_np_gen_traj_11_wrt_ego: Optional[Dict[str, Array]],
    diff_token_to_np_int_traj_wrt_ego: Optional[Dict[str, Array]],
    diff_token_to_np_int_traj_11_wrt_ego: Optional[Dict[str, Array]],
    options: DrawingOptions,
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """세 종류의 (T,11) 미래 궤적을 '속이 비어 있는 사각형'으로 렌더링.

    각 row(11,) = [x, y, cos, sin, vx, vy, width, length, onehot(3,)]

    렌더링 규칙
    ----------
    1) invalid 스텝 스킵: 앞 8차원 중 하나라도 |value|>eps 일 때만 유효(`is_valid_agent_row` 사용)
    2) 모두 **속 비움(fill 없음)** + 헤딩선 표시
    3) 겹침 순서(z-order): gen=22 → slip=23 → smooth=24 (smooth가 맨 위)
    4) 토큰 라벨: `gen`의 첫 유효 프레임 기준 1회 표기
       - 토큰 라벨 on/off: `options.DIFF_draw_diff_future_gen_traj_token`
       - 위치 오프셋: `options.DIFF_future_gen_trak_token_text_y_offset_m`
       - 색/두께: 각 스타일의 line_color/line_width 사용

    Args:
        ax: Matplotlib 축.
        diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, (T,11)] | None
        diff_token_to_np_int_traj_wrt_ego: Dict[str, (T,4)] | None
        diff_token_to_np_int_traj_11_wrt_ego: Dict[str, (T,11)] | None
        options: DrawingOptions
        draw_token_list: 특정 토큰만 그리고 싶을 때 지정. None이면 전체.
    """
    # 1) 원본 gen (흰색) — 라벨은 여기서만

    if options.DIFF_future_traj_draw_mode == "rectangle":
        if options.DIFF_draw_diff_future_gen_traj:
            draw_traj_dict_as_unfilled_rects(
                ax=ax,
                token_to_traj_11=diff_token_to_np_gen_traj_11_wrt_ego,
                style=options.DIFF_future_gen_style,
                options=options,
                zorder=22,
                draw_token_list=draw_token_list,
                annotate_token=options.DIFF_draw_diff_future_gen_traj_token,
                annotate_fontsize=options.DIFF_future_gt_3_dim_token_fontsize,
                annotate_offset=options.
                DIFF_future_gen_trak_token_text_y_offset_m,
                draw_velocity=options.DIFF_draw_diff_future_gen_traj_vel,
            )

        if options.DIFF_draw_diff_future_int_traj_11:
            draw_traj_dict_as_unfilled_rects(
                ax=ax,
                token_to_traj_11=diff_token_to_np_int_traj_11_wrt_ego,
                style=options.DIFF_future_int_style,
                options=options,
                zorder=24,
                draw_token_list=draw_token_list,
                annotate_token=False,
                draw_velocity=options.DIFF_draw_diff_future_int_traj_11_vel,
            )

        if options.DIFF_draw_diff_future_int_traj_to_be:
            draw_traj_dict_as_unfilled_rects(
                ax=ax,
                token_to_traj_11=diff_token_to_np_int_traj_wrt_ego,
                style=options.DIFF_future_slip_style,
                options=options,
                zorder=23,
                draw_token_list=draw_token_list,
                annotate_token=False,
                draw_velocity=options.DIFF_draw_diff_future_int_traj_to_be_vel,
            )

    elif options.DIFF_future_traj_draw_mode in ["arrow", "point", "line"]:
        if options.DIFF_draw_diff_future_gen_traj:
            draw_traj_dict_as_non_square(
                ax=ax,
                token_to_traj_11=diff_token_to_np_gen_traj_11_wrt_ego,
                style=options.DIFF_future_gen_style,
                options=options,
                draw_token_list=draw_token_list,
                annotate_token=options.DIFF_draw_diff_future_gen_traj_token,
                annotate_fontsize=options.DIFF_future_gt_3_dim_token_fontsize,
                annotate_offset=options.
                DIFF_future_gen_trak_token_text_y_offset_m,
            )
        if options.DIFF_draw_diff_future_int_traj_11:
            draw_traj_dict_as_non_square(
                ax=ax,
                token_to_traj_11=diff_token_to_np_int_traj_11_wrt_ego,
                style=options.DIFF_future_int_style,
                options=options,
                draw_token_list=draw_token_list)
        if options.DIFF_draw_diff_future_int_traj_to_be:
            draw_traj_dict_as_non_square(
                ax=ax,
                token_to_traj_11=diff_token_to_np_int_traj_wrt_ego,
                style=options.DIFF_future_slip_style,
                options=options,
                draw_token_list=draw_token_list)


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
    input_data: WorldModelFeature,
    output_data: Optional[Dict[str, Any]],
    options: DrawingOptions,
) -> Tuple[float, float, float, float]:
    """valid (x,y)만 모아 자동으로 축 범위를 산출."""
    # [Add]
    xs, ys = collect_valid_xy_for_bounds(
        input_data,
        output_data,
        options,
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


def draw_neighbor_past_output(
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
    - 박스/헤딩/속도화살표 색상과 두께는 DIFF_new_waypoint_style 사용(주황색).
    - 텍스트는 토큰 문자열을 **마지막 유효 프레임** 위치의 오른쪽에 주황색으로 표기.
    - 속도 화살표는 options.draw_velocity_arrows_past_all에 따라
      · True  → 모든 히스토리 스텝
      · False → 마지막(현재) 스텝만
    """
    if not current_token_to_np_history:
        return

    eps = options.invalid_eps
    line_color = options.NEI_neighbor_past_output_style["line_color"]
    line_width = options.NEI_neighbor_past_output_style["line_width"]

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

            x, y = float(row[0]), float(row[1])
            c, s = float(row[2]), float(row[3])
            vx, vy = float(row[4]), float(row[5])
            W, L = float(row[6]), float(row[7])

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
                nominal_length=L * options.COMMON_heading_line_scale,
                color=line_color,
                line_width=line_width,
                zorder=19 if t == current_t else 18,
            )

            # 속도 화살표: 모든 스텝
            if options.NEI_draw_neighbor_past_output_vel:
                add_velocity_arrow(
                    ax,
                    x,
                    y,
                    vx,
                    vy,
                    length_m=options.COMMON_vel_arrow_len_m,
                    line_color=line_color,
                    line_width=line_width,
                    line_alpha=1.0,
                    zorder=20 if t == current_t else 18,
                    t=t,
                )
                # [추가] 속도 크기 텍스트(km/h) - 모든 히스토리 지점
                speed_kmh = float(np.hypot(vx, vy)) * 3.6
                if t % 2 == 0:
                    offset = 4
                else:
                    offset = 3
                ax.text(
                    x,
                    y - options.NEI_neighbor_past_output_vel_offset_m * offset,
                    f"{speed_kmh:.1f}",
                    color=line_color,
                    fontsize=options.NEI_neighbor_past_output_vel_fontsize,
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
        #         fontsize=options.NEI_neighbor_past_output_token_fontsize,
        #         ha="left",
        #         va="center",
        #         zorder=21,
        #     )


def get_agent_idx_from_tokens(
        draw_token_list: Optional[List[str]],
        neighbor_track_token: Optional[List[Optional[str]]]
) -> Optional[List[int]]:
    """token_candidates에 포함된 토큰을 가진 이웃 차량의 인덱스를 반환."""
    if neighbor_track_token is None or draw_token_list is None:
        return None
    draw_token_int_list = []
    if neighbor_track_token is None:
        return draw_token_int_list
    for idx, token in enumerate(neighbor_track_token):
        if token is not None and token in draw_token_list:
            draw_token_int_list.append(idx)
    return draw_token_int_list


def draw_ego(ax: plt.Axes, input_data: WorldModelFeature,
             draw_option: DrawingOptions):
    if draw_option.EGO_draw_ego_past:
        draw_ego_past(ax, input_data.get("ego_agent_past"), draw_option)
    if draw_option.EGO_draw_ego_agent_next_11_dim:
        draw_ego_agent_next_11_dim(ax, input_data.get("ego_agent_next_11_dim"),
                                   draw_option)
    ### [EGO FUTURE PLANNER] ###
    if draw_option.EGO_draw_planner_future_11_dim:
        data_ = input_data.get("planner_future_11_dim", None)
        draw_planner_future_11_dim(ax, data_, draw_option)
    ### [EGO FUTURE GT 11] ###
    if draw_option.EGO_draw_ego_future_gt_11_dim:
        data_ = input_data.get("ego_future_gt_11_dim", None)
        draw_ego_future_gt_11_dim(ax, data_, draw_option)


def draw_neighbor_past_all(ax: plt.Axes,
                           input_data: WorldModelFeature,
                           output_data: Optional[Dict[str, Any]],
                           draw_option: DrawingOptions,
                           draw_token_list: Optional[List[str]] = None):
    ### [NEIGHBOR PAST] ###
    neighbor_agents_past = input_data.get("neighbor_agents_past", None)
    if draw_option.NEI_draw_neighbor_past and (neighbor_agents_past
                                               is not None):
        draw_token_int_list: Optional[List[int]] = get_agent_idx_from_tokens(
            draw_token_list, input_data.get("neighbor_track_token", None))
        neighbor_role = input_data.get("neighbor_role", None)  # NEW
        draw_neighbor_past(
            ax,
            neighbor_agents_past,
            draw_option,
            neighbor_role=neighbor_role,  # NEW
            draw_token_int_list=draw_token_int_list,
        )
        # List[Optional[str]]
        neighbor_track_token = input_data.get("neighbor_track_token", None)
        if draw_option.NEI_draw_past_token and neighbor_track_token is not None:
            annotate_neighbor_indices_for_past(ax, neighbor_track_token,
                                               neighbor_agents_past,
                                               draw_option)
    #########################################
    ### [NEIGHBOR PAST OUTPUT] ###
    # (history_len, 11)
    diff_token_to_np_history_wrt_ego = output_data.get(
        "diff_token_to_np_history_wrt_ego", None)
    if draw_option.NEI_draw_neighbor_past_output and diff_token_to_np_history_wrt_ego is not None:
        draw_neighbor_past_output(ax, diff_token_to_np_history_wrt_ego,
                                  draw_option)


def draw_neighbor_future_all(ax: plt.Axes,
                             input_data: WorldModelFeature,
                             output_data: Optional[Dict[str, Any]],
                             draw_option: DrawingOptions,
                             draw_token_list: Optional[List[str]] = None):
    ### [NEIGHBOR FUTURE GT] ###
    diff_token_to_future_gt_3_dim = input_data.get(
        "diff_token_to_future_gt_3_dim", None)

    # NEW: numpy 버전 neighbor_future_gt_3_dim 사용
    neighbor_future_gt_3_dim: Optional[Array] = input_data.get(
        "neighbor_future_gt_3_dim", None)
    neighbor_track_token: Optional[List[str]] = input_data.get(
        "neighbor_track_token", None)

    if draw_option.DIFF_draw_diff_np_future_gt_3_dim:
        # 설정 충돌 방지: 둘 다 True면 에러
        if draw_option.DIFF_draw_diff_future_gt_3_dim:
            raise ValueError(
                "DIFF_draw_diff_np_future_gt_3_dim=True 인 경우 "
                "DIFF_draw_diff_future_gt_3_dim 는 반드시 False 여야 합니다.")
        if neighbor_future_gt_3_dim is None:
            raise ValueError("DIFF_draw_diff_np_future_gt_3_dim=True 인데 "
                             "`neighbor_future_gt_3_dim` 이 input_data 에 없습니다.")
        if neighbor_track_token is None:
            raise ValueError("DIFF_draw_diff_np_future_gt_3_dim=True 인데 "
                             "`neighbor_track_token` 이 input_data 에 없습니다.")

        # numpy (max_agent_num, T, 3) + track_token 리스트 → dict[str, (T,3)]
        diff_token_to_future_gt_3_dim_from_np: Dict[str, Array] = (
            _build_diff_token_to_future_gt_3_dim_from_neighbor_np(
                neighbor_future_gt_3_dim=neighbor_future_gt_3_dim,
                neighbor_track_token=neighbor_track_token,
            ))

        draw_neighbor_future_gt_3_dim(
            ax,
            diff_token_to_future_gt_3_dim_from_np,
            draw_option,
            draw_token_list,
        )

    # 기존 dict 버전 사용하는 경우
    elif draw_option.DIFF_draw_diff_future_gt_3_dim and (
            diff_token_to_future_gt_3_dim is not None):
        draw_neighbor_future_gt_3_dim(
            ax,
            diff_token_to_future_gt_3_dim,
            draw_option,
            draw_token_list,
        )

    ### [NEIGHBOR FUTURE OUTPUT] ###
    diff_token_to_np_gen_traj_11_wrt_ego = output_data.get(
        "diff_token_to_np_gen_traj_11_wrt_ego", None)
    diff_token_to_np_int_traj_wrt_ego = output_data.get(
        "diff_token_to_np_int_traj_wrt_ego", None)
    diff_token_to_np_int_traj_11_wrt_ego = output_data.get(
        "diff_token_to_np_int_traj_11_wrt_ego", None)
    draw_diff_future_traj_w_square(ax, diff_token_to_np_gen_traj_11_wrt_ego,
                                   diff_token_to_np_int_traj_wrt_ego,
                                   diff_token_to_np_int_traj_11_wrt_ego,
                                   draw_option, draw_token_list)


def draw_neighbor(ax: plt.Axes,
                  input_data: WorldModelFeature,
                  output_data: Optional[Dict[str, Any]],
                  draw_option: DrawingOptions,
                  draw_token_list: Optional[List[str]] = None):
    draw_neighbor_past_all(ax, input_data, output_data, draw_option,
                           draw_token_list)
    draw_neighbor_future_all(ax, input_data, output_data, draw_option,
                             draw_token_list)


def draw_lane(
    ax: plt.Axes,
    input_data: WorldModelFeature,
    draw_option: DrawingOptions,
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """차선(경계/센터라인) 레이어를 그린다.

    Args:
        ax: Matplotlib Axes.
        input_data: world model 입력 데이터(dict).
        draw_option: 렌더링 옵션.
        draw_token_list: 특정 에이전트 토큰만 텍스트 표기하고 싶을 때 사용. None이면 전체.
    """
    lanes = input_data.get("lanes")  # shape: (lane_num, lane_len, 12)
    lanes_speed_limit = input_data.get(
        "lanes_speed_limit")  # shape: (lane_num, 1)
    lanes_has_speed_limit = input_data.get(
        "lanes_has_speed_limit")  # shape: (lane_num, 1)

    # ✅ 추가: 캐시에 들어있는 타입 정보들을 draw_lane_boundaries로 넘겨야 실제로 반영됩니다.
    lane_type = input_data.get("lane_type",
                               None)  # shape: (lane_num, 4) or None
    left_line_type = input_data.get("left_line_type",
                                    None)  # shape: (lane_num, 10) or None
    right_line_type = input_data.get("right_lane_type",
                                     None)  # shape: (lane_num, 10) or None

    # shape: (max_agent_num, lane_num) or None
    agent_route_lane_order: Optional[Array] = input_data.get(
        "agent_route_lane_order", None)

    draw_token_int_list: Optional[List[int]] = get_agent_idx_from_tokens(
        draw_token_list, input_data.get("neighbor_track_token", None))

    if draw_option.LANE_draw_lane_boundaries:
        draw_lane_boundaries(
            ax=ax,
            lanes=lanes,
            agent_route_lane_order=agent_route_lane_order,
            options=draw_option,
            draw_token_int_list=draw_token_int_list,
            lane_type=lane_type,
            left_line_type=left_line_type,
            right_line_type=right_line_type,
        )

    if draw_option.LANE_draw_lane_centerline:
        draw_lane_centerlines(
            ax,
            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,
            draw_option,
            agent_route_lane_order=agent_route_lane_order,
            draw_token_int_list=draw_token_int_list,
        )


# [Add]
def draw_world_model_to_png(
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]],
    save_path: str,
    options: Optional[DrawingOptions] = None,
) -> None:
    draw_token_list: List[str] = [
        "58a9e2ba05555824"
    ]  # ["1be4dfd6d2f852a9", "f476b2c85dd7508c", "88dbeb62be085df7"]
    draw_token_list: List[str] = [
        "f476b2c85dd7508c"
    ]  # ["1be4dfd6d2f852a9", "f476b2c85dd7508c", "88dbeb62be085df7"] # d6ff7e795dd051ac
    # draw_token_list = ["d6ff7e795dd051ac"] # aee2dbe7e9245b23
    # draw_token_list = ["8f85cc67cb005921"]
    draw_token_list = [
        "725916938b635a6a",
        "6c229171ce82522f",
        "301f25e8eec059c8",
        "2045120275235b70",
        "4a5c4367c1a05eec",
        "f476b2c85dd7508c",
        "79628c71e7235e74",
        "88dbeb62be085df7",
        "323efed07e795031",
        "5f0cde9b72c74a8a",
        "2e3b8f4c16485428"
        "9a43610571815b5c",
        "3aef014a6b26521e",
        "7d1c247dedfe562d",
        "994e1d789d9f55f6",
        "cacdd4aaa39a5463",
        "4d5baf8fab3551f0",
        "15927b42cd1a52d5",
        "a70597f8077d5853",
    ]
    draw_token_list = None

    draw_option = options or DrawingOptions()

    # 1) Figure/Axes
    fig, ax = create_figure_and_axes(draw_option)

    #########################################
    draw_lane(ax, input_data, draw_option, draw_token_list)

    # [ADD] road_edge / driveway
    draw_road_edge_points(
        ax=ax,
        road_edge=input_data.get("road_edge", None),
        road_edge_type=input_data.get("road_edge_type", None),
        options=draw_option,
    )
    draw_driveway_points(
        ax=ax,
        driveway=input_data.get("driveway", None),
        options=draw_option,
    )

    draw_road_safety(ax, input_data, draw_option)
    draw_ego(ax, input_data, draw_option)
    # NEW: ego 주변 반경 원
    draw_ego_radius_circle(ax, draw_option)
    draw_neighbor(ax, input_data, output_data, draw_option, draw_token_list)
    #########################################

    # # ── (5) 축 범위/스타일 ───────────────────────────────────────────
    # # [Add]
    bounds = compute_auto_bounds(input_data, output_data, draw_option)
    set_axes_limits_with_margin(ax, bounds, draw_option.margin_m)
    apply_axes_style(ax, draw_option)

    # 6) 저장
    save_figure_to_png(fig, save_path)
