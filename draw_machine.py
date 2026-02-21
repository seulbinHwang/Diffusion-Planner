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
    dpi: int = 300
    margin_m: float = 5.0
    equal_aspect: bool = True
    invalid_eps: float = 0.0

    COMMON_vel_arrow_len_m: float = 5.0
    COMMON_heading_line_scale: float = 0.5
    LANE_boundary_vector_invalid_eps: float = 0.0
    ######## LANES ########

    LANE_draw_lane_boundaries: bool = True  # check
    LANE_boundary_width: float = 1.
    LANE_draw_lane_centerline: bool = True  # check
    LANE_draw_npc_agent_route: bool = False
    LANE_draw_vel_limit: bool = True
    LANE_npc_agent_route_draw_mode: str = "lane"  # "centerline" / "lane"
    LANE_route_agent_index_color: str = CYAN  # 번호 텍스트 색 # 청록색
    LANE_lane_boundary_color = PURPLE  # 남색(인디고 계열)

    # ===== lane_type 기반 센터라인 색(디버그 시각화용) =====
    LANE_freeway_centerline_color: str = PURPLE
    LANE_surface_street_centerline_color: str = CYAN
    LANE_bike_lane_centerline_color: str = GREEN
    LANE_undefined_centerline_color: str = SILVER

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

    # ===== lane_type 색을 '바탕선'으로 먼저 그릴지 =====
    LANE_draw_lane_type_underlay: bool = True
    LANE_lane_type_underlay_width: float = 2.4
    LANE_lane_type_underlay_zorder: int = 0
    LANE_lane_marking_zorder: int = 1

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
    EGO_future_traj_draw_mode: str = "arrow"  # 'rectangle' / 'arrow'/ 'point' / 'line'

    EGO_draw_ego_agent_next_11_dim: bool = True
    EGO_draw_diffusion: bool = True
    EGO_draw_diffusion_mode = "integrate"  # "direct" / "integrate" / "interp"
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
    EGO_draw_planner_velocity: bool = False  # check
    ########## [EGO] FUTURE EGO GT 11 ##########
    EGO_draw_ego_future_gt_11_dim: bool = True
    EGO_future_gt_11_style = {
        "line_color": CYAN,
        "line_width": 0.6,
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
    NEI_draw_past_token: bool = False
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
    DIFF_draw_diff_future_gt_3_dim: bool = True
    # NEW: neighbor_future_gt_3_dim (numpy 버전) on/off
    DIFF_draw_future_gt_3_dim_wo_token: bool = False
    DIFF_future_gt_3_dim_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기
    DIFF_future_gt_3_dim_COLOR: str = DARK_BROWN  # 미래 포인트 'x' 마커 크기

    # NEW: GT 3차원 궤적 그리기 모드 ("point" / "line" / "arrow")
    DIFF_future_gt_3_dim_draw_mode: str = "line"
    DIFF_future_gt_3_dim_line_width: float = 0.4
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
    DIFF_future_traj_draw_mode: str = "line"  # 'rectangle' / 'arrow'/ 'point' / 'line'
    DIFF_future_gen_traj_point_marker: str = "o"
    DIFF_future_gen_traj_point_marker_size: float = 0.8
    DIFF_future_gen_traj_arrow_len_m: float = 1.0
    DIFF_future_gen_style = {
        "line_color": WHITE,  # 빨간색(밝은 빨강)
        "token_color": WHITE,  # 빨간색(밝은 빨강)
        "line_width": 0.5,
        "velocity_line_color": WHITE,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_diff_future_gen_traj_vel: bool = False

    DIFF_draw_diff_future_int_traj_11: bool = True
    DIFF_future_int_style = {
        "line_color": LIGHTBLUE,  # 빨간색(밝은 빨강)
        "token_color": LIGHTBLUE,  # 빨간색(밝은 빨강)
        "line_width": 0.8,
        "velocity_line_color": LIGHTBLUE,  # 빨간색(밝은 빨강)
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

    DIFF_draw_diff_interp_np_traj_11: bool = False
    DIFF_future_interp_style = {
        "line_color": GREEN,  # 빨간색(밝은 빨강)
        "token_color": GREEN,  # 빨간색(밝은 빨강)
        "line_width": 0.7,
        "velocity_line_color": GREEN,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_diff_interp_np_traj_11_vel: bool = False

    ########################
    DIFF_future_gen_refined_style = {
        "line_color": RED,  # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": RED,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
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
    - center_lanes: (lane_num, lane_len, 8) (유효 포인트만)
    - road_line: (line_num, lane_len, 2) (유효 포인트만)
    - ego: ego_agent_past, ego_agent_next_11_dim, planner_future_11_dim
    - neighbor: neighbor_agents_past
    - neighbor future GT: neighbor_future_gt_3_dim (x,y만 사용)
    - road safety polygons: stop_sign_points/speed_bump_points/crosswalk_points
    - road_edge / driveway
    """
    eps = options.invalid_eps
    xs_local: List[float] = []
    ys_local: List[float] = []

    # lanes: 각 포인트가 valid일 때 center/left/right 좌표 반영
    lanes = input_data.get("lanes")
    if lanes is not None and np.asarray(lanes).size > 0:
        lanes_arr: Array = np.asarray(lanes)  # (lane_num, lane_len, 12)
        valid_mask = np.any(np.abs(lanes_arr[:, :, :8]) > eps,
                            axis=2)  # (lane_num, lane_len)
        centers = lanes_arr[:, :, 0:2]
        lefts = centers + lanes_arr[:, :, 4:6]
        rights = centers + lanes_arr[:, :, 6:8]
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
    for key in ("ego_agent_past", "ego_agent_next_11_dim", "ego_future_gt_11_dim"):
        A = input_data.get(key)
        if A is None or np.asarray(A).size == 0:
            continue
        A_arr: Array = np.asarray(A)  # (N,11) 가정
        valid_mask = np.any(np.abs(A_arr[:, :8]) > eps, axis=1)  # (N,)
        if np.any(valid_mask):
            xy = A_arr[valid_mask, 0:2]
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # neighbor past
    neigh = input_data.get("neighbor_agents_past")
    if neigh is not None and np.asarray(neigh).size > 0:
        neigh_arr: Array = np.asarray(neigh)  # (max_agent_num, T, 11)
        valid_mask = np.any(np.abs(neigh_arr[:, :, :8]) > eps, axis=2)  # (A, T)
        if np.any(valid_mask):
            xy = neigh_arr[:, :, 0:2][valid_mask]
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # neighbor future points: (max_agent_num, future_len, 3) -> (x,y)만 사용
    neigh_fut = input_data.get("neighbor_future_gt_3_dim")
    if neigh_fut is not None and np.asarray(neigh_fut).size > 0:
        neigh_fut_arr: Array = np.asarray(neigh_fut)  # (A, T, 3)
        if neigh_fut_arr.ndim != 3 or neigh_fut_arr.shape[-1] != 3:
            raise ValueError(
                "neighbor_agents_future는 (max_agent_num, future_len, 3) 이어야 합니다."
            )
        valid_mask = (np.abs(neigh_fut_arr[..., 0]) > eps) | (np.abs(
            neigh_fut_arr[..., 1]) > eps)  # (A, T)
        if np.any(valid_mask):
            xy = neigh_fut_arr[..., :2][valid_mask]  # (K, 2)
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # road safety polygons: shape (N, P, 2)
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
        딥러닝 output 값 그대로 # (1+T, 11)
        """
        self.ego_np_traj_11_wrt_ego: Optional[np.ndarray] = None
        """
        딥러닝 적분 출력값 # (1+T, 11)
        """
        self.ego_np_int_traj_11_wrt_ego: Optional[np.ndarray] = None
        """
        딥러닝 output 값 그대로
        """
        self.diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, np.ndarray] = {
        }  # (1+T, 11)
        """
        딥러닝 적분 출력값
        """
        self.diff_token_to_np_int_traj_wrt_ego: Dict[str, np.ndarray] = {
        }  # (1+T, 4)
        """
        딥러닝 적분 출력을 11차원으로 
        """
        self.diff_token_to_np_int_traj_11_wrt_ego: Dict[str, np.ndarray] = {
        }  # (1+T, 11)
        """
        history Agent 만든걸 -> (History_len, 11) numpy로 변환한 것들
        npc 미래 궤적 보정 input으로 쓰이는걸 그려보기 위해 저장
        """
        self.diff_token_to_np_history_wrt_ego: Dict[str, np.ndarray] = {
        }  # (History_len, 11)
        """
            interpolation으로, 생성된 미래 궤적에 속도를 추가한 것
            
        EGO_draw_planner_future_11_dim = True일 떄,
            EGO_draw_diffusion = True이면, ego_interp_np_traj_wrt_ego 로 그림그리고, (+assert not None)
            EGO_draw_diffusion = False이면, planner_future_11_dim 로 그림그리자.
        """
        self.ego_interp_np_traj_wrt_ego: Optional[np.ndarray] = None
        # (1 + Future_len, 11)
        """
            interpolation으로, 생성된 미래 궤적에 속도를 추가한 것
        """
        self.diff_token_to_interp_np_traj_wrt_ego: Dict[str, np.ndarray] = {
        }  # (1 + Future_len, 11)
        """
        interpolation 궤적 생성 후, next_iteration 시점 waypoint를 array로 변환한 것
        
        EGO_draw_ego_agent_next_11_dim = True일 떄, 
            EGO_draw_diffusion = True이면, ego_next_wp_wrt_ego 로 그림그리고, (+assert not None)
            EGO_draw_diffusion = False이면, ego_agent_next_11_dim 로 그림그리자.
        """
        self.ego_next_wp_wrt_ego: Optional[np.ndarray] = None  # (11,)
        """
        interpolation 궤적 생성 후, next_iteration 시점 waypoint를 array로 변환한 것
        """
        self.diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] = {}  # (11,)

    def to_dict(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_data = self.model_input_key_to_unnorm_value
        output_data = {
            "ego_np_traj_11_wrt_ego":
                self.ego_np_traj_11_wrt_ego,
            "ego_np_int_traj_11_wrt_ego":
                self.ego_np_int_traj_11_wrt_ego,
            "diff_token_to_np_gen_traj_11_wrt_ego":
                self.diff_token_to_np_gen_traj_11_wrt_ego,  # (1+T, 11)
            "diff_token_to_np_int_traj_wrt_ego":
                self.diff_token_to_np_int_traj_wrt_ego,  # (1+T, 4)
            "diff_token_to_np_int_traj_11_wrt_ego":
                self.diff_token_to_np_int_traj_11_wrt_ego,  # (1+T, 11)
            "diff_token_to_np_history_wrt_ego":
                self.diff_token_to_np_history_wrt_ego,
            "ego_interp_np_traj_wrt_ego":
                self.ego_interp_np_traj_wrt_ego,
            "diff_token_to_interp_np_traj_wrt_ego":
                self.diff_token_to_interp_np_traj_wrt_ego,
            "ego_next_wp_wrt_ego":
                self.ego_next_wp_wrt_ego,
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


def _lane_type_one_hot_to_index_4(
        lane_type_row4: Optional[Array],  # shape: (4,)
) -> int:
    """차선 종류를 나타내는 4칸짜리 값을 0~3 정수로 바꾼다.

    이 함수는 차선 종류가 들어있는 (4,) 배열을 받아서,
    가장 값이 큰 위치를 "차선 종류"로 봅니다.

    값이 전부 0인 경우도 있을 수 있는데,
    그때는 "미정(UNDEFINED)"으로 처리합니다.

    Args:
        lane_type_row4 (Optional[np.ndarray]):
            shape = (4,).
            값의 의미/순서:
            - 0: FREEWAY
            - 1: SURFACE_STREET
            - 2: BIKE_LANE
            - 3: UNDEFINED
            None이면 UNDEFINED로 처리합니다.

    Returns:
        int:
            0~3 중 하나.
            - 0: FREEWAY
            - 1: SURFACE_STREET
            - 2: BIKE_LANE
            - 3: UNDEFINED
    """
    if lane_type_row4 is None:
        return 3  # UNDEFINED

    lane_type_arr: Array = np.asarray(lane_type_row4).reshape(-1)  # shape: (4,)
    if lane_type_arr.shape[0] != 4:
        return 3  # 형식이 이상하면 UNDEFINED로 안전 처리

    if float(np.sum(np.abs(lane_type_arr))) == 0.0:
        return 3  # 전부 0이면 UNDEFINED

    return int(np.argmax(lane_type_arr))


def _lane_type_index_to_centerline_color(
    lane_type_idx: int,
    options: DrawingOptions,
) -> str:
    """차선 종류 인덱스(0~3)를 센터라인에 쓸 색으로 바꾼다.

    Args:
        lane_type_idx (int):
            0~3 값.
            - 0: FREEWAY
            - 1: SURFACE_STREET
            - 2: BIKE_LANE
            - 3: UNDEFINED
        options (DrawingOptions):
            색 설정을 가져오기 위한 옵션 객체.

    Returns:
        str:
            matplotlib에서 쓸 색 문자열.
    """
    if lane_type_idx == 0:
        return options.LANE_freeway_centerline_color
    if lane_type_idx == 1:
        return options.LANE_surface_street_centerline_color
    if lane_type_idx == 2:
        return options.LANE_bike_lane_centerline_color
    return options.LANE_undefined_centerline_color


def _lane_type_row4_to_centerline_color(
    lane_type_row4: Optional[Array],  # shape: (4,)
    options: DrawingOptions,
) -> str:
    """(4,) 차선 종류 값을 센터라인 색으로 바로 바꾼다.

    Args:
        lane_type_row4 (Optional[np.ndarray]):
            shape = (4,). 차선 종류 정보.
        options (DrawingOptions):
            색 설정을 가져오기 위한 옵션 객체.

    Returns:
        str:
            센터라인에 쓸 색 문자열.
    """
    lane_type_idx: int = _lane_type_one_hot_to_index_4(lane_type_row4)
    return _lane_type_index_to_centerline_color(lane_type_idx, options)


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


from typing import Literal


@dataclass(frozen=True)
class _LaneBoundaryDrawPlan:
    """차선 경계(왼쪽 또는 오른쪽)를 어떤 방식으로 그릴지 정리한 설정값.

    Attributes:
        draw (bool):
            True면 그리기, False면 그리지 않는다.
        draw_mode (Literal["line", "marker"]):
            - "line": 선(실선/점선/이중선)으로 그림
            - "marker": 점(마커)으로 찍어서 그림
        color (str):
            선/점의 색.
        marker (Optional[str]):
            draw_mode="marker"일 때 사용할 점 모양.
            예) "o", "*", "x"
        is_double (bool):
            draw_mode="line"일 때만 의미가 있다.
            True면 이중선(두 줄)로 그린다.
        inner_linestyle (Any):
            draw_mode="line"이고 이중선인 경우, 차선 안쪽 줄 스타일.
        outer_linestyle (Any):
            draw_mode="line"이고 이중선인 경우, 차선 바깥쪽 줄 스타일.
    """
    draw: bool
    draw_mode: Literal["line", "marker"]
    color: str
    marker: Optional[str]
    is_double: bool
    inner_linestyle: Any
    outer_linestyle: Any


def _normalize_line_type_row_to_road_line10_and_road_edge3(
        line_type_row: Array,  # shape: (10,) or (13,)
) -> Tuple[Array, Array]:
    """left/right_line_type 한 줄을 (road line 10칸, road edge 3칸)으로 정리한다.

    이 프로젝트에서는 left/right_line_type이 보통 길이 13으로 들어오며,
    의미를 아래처럼 나눠서 해석한다고 가정합니다.

    - 앞 10칸: 차선 선(흰색/노란색, 실선/점선, 단선/이중선 등) 종류
    - 뒤  3칸: 도로 가장자리(road edge) 종류
        · [UNKNOWN, BOUNDARY, MEDIAN] 순서라고 가정

    예전 데이터처럼 길이 10으로 들어오는 경우도 있을 수 있어서,
    그때는 "road edge는 없다"고 보고 뒤 3칸을 전부 0으로 채워 반환합니다.

    Args:
        line_type_row (np.ndarray):
            shape = (13,) 또는 (10,).
            한 차선의 한쪽 경계 타입 정보를 담고 있다.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - road_line_row10: shape = (10,)
            - road_edge_row3: shape = (3,)
    """
    arr: Array = np.asarray(line_type_row).reshape(-1)  # shape: (N,)

    if arr.shape[0] == 10:
        road_line_row10: Array = arr.astype(np.float32)  # shape: (10,)
        road_edge_row3: Array = np.zeros((3,), dtype=np.float32)  # shape: (3,)
        return road_line_row10, road_edge_row3

    if arr.shape[0] == 13:
        road_line_row10 = arr[:10].astype(np.float32)  # shape: (10,)
        road_edge_row3 = arr[10:].astype(np.float32)  # shape: (3,)
        return road_line_row10, road_edge_row3

    raise ValueError(
        f"line_type_row는 shape (10,) 또는 (13,) 이어야 합니다. got {arr.shape}")


def _road_edge_row3_to_marker(
        road_edge_row3: Array,  # shape: (3,)
) -> str:
    """road edge 타입(3칸)을 보고 점 모양(marker)을 고른다.

    규칙:
    - BOUNDARY  -> "*" (별표 모양 점)
    - MEDIAN    -> "x" (엑스 모양 점)
    - UNKNOWN   -> "o" (동그라미 점)

    Args:
        road_edge_row3 (np.ndarray):
            shape = (3,)
            [UNKNOWN, BOUNDARY, MEDIAN] 순서라고 가정한다.

    Returns:
        str:
            matplotlib marker 문자열("o", "*", "x" 중 하나)
    """
    arr: Array = np.asarray(road_edge_row3).reshape(-1)  # shape: (3,)
    if arr.shape[0] != 3:
        raise ValueError(f"road_edge_row3 shape는 (3,) 이어야 합니다. got {arr.shape}")

    if float(np.sum(np.abs(arr))) == 0.0:
        idx: int = 0  # 전부 0이면 UNKNOWN으로 안전 처리
    else:
        idx = int(np.argmax(arr))

    if idx == 1:  # BOUNDARY
        return "*"
    if idx == 2:  # MEDIAN
        return "x"
    return "o"  # UNKNOWN


def _make_lane_boundary_draw_plan(
    line_type_row: Optional[Array],  # shape: (13,) or (10,) or None
    default_color: str,
    options: DrawingOptions,
) -> _LaneBoundaryDrawPlan:
    """left/right_line_type에 따라 차선 경계를 '선' 또는 '점'으로 그릴지 결정한다.

    이번 변경의 핵심 규칙
    --------------------
    1) left/right_line_type이 (13,)인 경우:
       - 뒤 3칸(road edge)이 0이 아니면 → road edge로 보고 "점"으로 찍는다.
         · BOUNDARY -> '*'
         · MEDIAN   -> 'x'
         · UNKNOWN  -> 'o'
       - 이 때 색은 road edge 전용 색(options.ROAD_road_edge_color)을 사용한다.

    2) 앞 10칸(기존 road line)만 의미가 있는 경우:
       - INVALID(기존 규칙의 '무효')이면 그리지 않는다.
       - UNKNOWN(기존 규칙의 '알 수 없음')이면 이제 "선"이 아니라 'o' 점으로 찍는다.
       - 나머지는 기존처럼 흰/노란 + 실선/점선 + 단선/이중선 규칙으로 선을 그린다.

    Args:
        line_type_row (Optional[np.ndarray]):
            shape = (13,) 또는 (10,) 또는 None.
        default_color (str):
            타입이 없거나(또는 UNKNOWN을 road line로 해석할 때) 기본으로 쓸 색.
        options (DrawingOptions):
            색/두께/패턴 등의 설정값.

    Returns:
        _LaneBoundaryDrawPlan:
            실제 그리기 방식(선/점), 색, 스타일 정보를 담은 값.
    """
    # 기본(기존처럼): 단선 실선 + default_color
    default_plan = _LaneBoundaryDrawPlan(
        draw=True,
        draw_mode="line",
        color=default_color,
        marker=None,
        is_double=False,
        inner_linestyle=options.LANE_solid_linestyle,
        outer_linestyle=options.LANE_solid_linestyle,
    )

    if line_type_row is None:
        return default_plan

    road_line_row10, road_edge_row3 = _normalize_line_type_row_to_road_line10_and_road_edge3(
        line_type_row=line_type_row)

    # (A) road edge가 존재하면 → 점으로 표시(별표/엑스/동그라미)
    if float(np.sum(np.abs(road_edge_row3))) > 0.0:
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="marker",
            color=options.ROAD_road_edge_color,
            marker=_road_edge_row3_to_marker(road_edge_row3),
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    # (B) 기존 road line(10칸) 규칙으로 처리
    idx10: int = _line_type_one_hot_to_index_10(road_line_row10)

    # INVALID
    if idx10 == 9:
        return _LaneBoundaryDrawPlan(
            draw=False,
            draw_mode="line",
            color=default_color,
            marker=None,
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    # UNKNOWN: 이번 요청대로 'o' 점으로 표시
    if idx10 == 8:
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="marker",
            color=default_color,
            marker="o",
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    # 나머지(0~7): 기존 매핑 유지
    # 0: BROKEN_SINGLE_WHITE
    # 1: SOLID_SINGLE_WHITE
    # 2: SOLID_DOUBLE_WHITE
    # 3: BROKEN_SINGLE_YELLOW
    # 4: BROKEN_DOUBLE_YELLOW
    # 5: SOLID_SINGLE_YELLOW
    # 6: SOLID_DOUBLE_YELLOW
    # 7: PASSING_DOUBLE_YELLOW
    if idx10 in (0, 1, 2):
        line_color: str = options.LANE_line_white_color
    else:
        line_color = options.LANE_line_orange_color

    if idx10 in (0, 3):
        # broken single
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="line",
            color=line_color,
            marker=None,
            is_double=False,
            inner_linestyle=options.LANE_broken_linestyle,
            outer_linestyle=options.LANE_broken_linestyle,
        )

    if idx10 in (1, 5):
        # solid single
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="line",
            color=line_color,
            marker=None,
            is_double=False,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    if idx10 in (2, 6):
        # solid double
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="line",
            color=line_color,
            marker=None,
            is_double=True,
            inner_linestyle=options.LANE_solid_linestyle,
            outer_linestyle=options.LANE_solid_linestyle,
        )

    if idx10 == 4:
        # broken double yellow
        return _LaneBoundaryDrawPlan(
            draw=True,
            draw_mode="line",
            color=line_color,
            marker=None,
            is_double=True,
            inner_linestyle=options.LANE_broken_linestyle,
            outer_linestyle=options.LANE_broken_linestyle,
        )

    # PASSING_DOUBLE_YELLOW:
    # "차선 쪽(안쪽)은 점선, 바깥쪽은 실선"으로 표현
    return _LaneBoundaryDrawPlan(
        draw=True,
        draw_mode="line",
        color=line_color,
        marker=None,
        is_double=True,
        inner_linestyle=options.LANE_broken_linestyle,
        outer_linestyle=options.LANE_solid_linestyle,
    )


def _draw_lane_boundary_points(
    ax: plt.Axes,
    boundary_xy: Array,  # shape: (T, 2)
    point_drawable: Array,  # shape: (T,)
    marker: str,
    edge_color: str,
    marker_size: float,
    edge_line_width: float,
    zorder: int,
) -> None:
    """차선 경계를 '점(마커)'으로 찍어서 그린다.

    선이 아니라 점으로 표현해야 하는 타입(예: road edge, unknown)일 때 사용한다.

    Args:
        ax (plt.Axes):
            Matplotlib 축 객체.
        boundary_xy (np.ndarray):
            shape = (T, 2). 각 시점의 경계 좌표들.
        point_drawable (np.ndarray):
            shape = (T,) bool.
            True인 위치만 점으로 찍는다.
        marker (str):
            점 모양("o", "*", "x" 등).
        edge_color (str):
            점 테두리 색.
        marker_size (float):
            scatter의 s 값(점 크기).
        edge_line_width (float):
            점 테두리 두께.
        zorder (int):
            그리기 순서.
    """
    bxy: Array = np.asarray(boundary_xy)  # shape: (T, 2)
    ok: Array = np.asarray(point_drawable).astype(bool).reshape(
        -1)  # shape: (T,)

    if bxy.ndim != 2 or bxy.shape[1] != 2:
        raise ValueError(f"boundary_xy shape는 (T,2) 이어야 합니다. got {bxy.shape}")
    if ok.ndim != 1 or ok.shape[0] != bxy.shape[0]:
        raise ValueError(
            f"point_drawable shape는 (T,) 이어야 합니다. got {ok.shape}, T={bxy.shape[0]}"
        )

    if not np.any(ok):
        return

    xy: Array = bxy[ok]  # shape: (K, 2)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        marker=marker,
        s=float(marker_size),
        linewidths=float(edge_line_width),
        edgecolors=edge_color,
        facecolors="none",
        zorder=int(zorder),
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


def is_valid_xy_point(
    point2: Array,  # shape: (2,)
    eps: float,
) -> bool:
    """(x, y) 점이 유효한 값인지 확인한다.

    이 프로젝트에서는 "무효 점"을 (0, 0)으로 채워 넣는 규칙을 쓰고 있다.
    그래서 (x, y)가 둘 다 0(또는 eps 이내)이면 무효로 본다.

    Args:
        point2 (np.ndarray): shape = (2,). [x, y]
        eps (float): 0으로 볼 허용 오차.

    Returns:
        bool: 유효하면 True, 무효면 False.
    """
    p: Array = np.asarray(point2).reshape(-1)  # (2,)
    if p.shape[0] != 2:
        raise ValueError(f"point2 shape는 (2,) 이어야 합니다. got {p.shape}")
    x: float = float(p[0])
    y: float = float(p[1])
    return bool((abs(x) > eps) or (abs(y) > eps))


def _compute_unit_normal_for_segment(
        p0: Array,  # shape: (2,)
        p1: Array,  # shape: (2,)
) -> Array:
    """선분(p0->p1)에 수직인 '단위 방향'을 만든다.

    이 함수는 road_line의 이중선을 그릴 때 사용한다.
    - 선분 방향과 직각인 방향(좌/우 중 한쪽)을 하나 고른다.
    - 길이가 1이 되도록 크기를 맞춘다.
    - 선분이 너무 짧으면 (0,0)을 돌려준다.

    Args:
        p0 (np.ndarray): shape = (2,)
        p1 (np.ndarray): shape = (2,)

    Returns:
        np.ndarray: shape = (2,) 인 단위 벡터(수직 방향)
    """
    a: Array = np.asarray(p0).astype(np.float32).reshape(2,)  # (2,)
    b: Array = np.asarray(p1).astype(np.float32).reshape(2,)  # (2,)
    seg: Array = (b - a).astype(np.float32)  # (2,)
    seg_len: float = float(np.hypot(seg[0], seg[1]))
    if seg_len < 1e-6:
        return np.array([0.0, 0.0], dtype=np.float32)

    # 한쪽 수직 방향(좌/우 중 한쪽) 선택
    normal: Array = np.array([-seg[1], seg[0]],
                             dtype=np.float32) / seg_len  # (2,)
    return normal


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


def _compute_boundary_vector_point_valid_mask(
    boundary_vec: Array,  # shape: (T, 2)
    eps: float,
    skip_zero_vector: bool,
) -> Array:
    """경계 벡터(dx, dy)가 실제로 있는 지점을 고르는 True/False 배열을 만든다.

    lanes 데이터에서 left_vec/right_vec은 (dx, dy) 형태로 들어온다.
    이때 (dx, dy)가 (0, 0)이면 "그 시점에는 경계가 없다"는 의미로 사용한다.

    따라서 경계선을 그릴 때
    - (dx, dy)가 (0, 0)인 점을 포함해서 선을 이어 그리면,
      실제로는 경계가 없는 구간인데도 선이 생기거나,
      센터라인과 겹치는 이상한 선이 생길 수 있다.

    이 함수는 boundary_vec의 각 행을 보고,
    - |dx| > eps 또는 |dy| > eps 이면 "경계가 있음(True)"
    - 둘 다 eps 이하면 "경계가 없음(False)"
    으로 판단한다.

    skip_zero_vector=False이면(기존 동작을 유지하고 싶을 때),
    모든 점을 True로 반환해서 경계점 필터링을 하지 않는다.

    Args:
        boundary_vec (np.ndarray): shape (T, 2). 각 행은 [dx, dy].
        eps (float): 0이라고 볼 허용 오차.
        skip_zero_vector (bool): True면 (0,0)인 점은 False로 만든다.

    Returns:
        np.ndarray: shape (T,) bool. True인 위치만 경계 점으로 사용 가능하다.
    """
    boundary_vec_arr: Array = np.asarray(boundary_vec)  # (T, 2)
    if boundary_vec_arr.ndim != 2 or boundary_vec_arr.shape[1] != 2:
        raise ValueError(
            f"boundary_vec shape는 (T, 2) 이어야 합니다. got {boundary_vec_arr.shape}")

    T: int = int(boundary_vec_arr.shape[0])
    if not skip_zero_vector:
        return np.ones((T,), dtype=bool)

    return np.any(np.abs(boundary_vec_arr) > float(eps), axis=1)  # (T,)


def _compute_consecutive_segment_drawable_mask(
        lane_point_valid_mask: Array,  # shape: (T,)
        boundary_point_valid_mask: Array,  # shape: (T,)
) -> Array:
    """연속된 두 점(i, i+1)을 선으로 이을 수 있는지에 대한 True/False 배열을 만든다.

    경계선을 선분으로 그릴 때는 "점이 2개"가 필요하다.
    그래서 i번째 점과 i+1번째 점이 모두 조건을 만족할 때만,
    (i -> i+1) 선분을 그리도록 True/False 배열을 만든다.

    선분을 그릴 조건:
    - lane_point_valid_mask[i] 와 lane_point_valid_mask[i+1] 가 둘 다 True
    - boundary_point_valid_mask[i] 와 boundary_point_valid_mask[i+1] 가 둘 다 True

    Args:
        lane_point_valid_mask (np.ndarray): shape (T,) bool.
            lanes 포인트 자체가 유효한지(기존 valid 규칙).
        boundary_point_valid_mask (np.ndarray): shape (T,) bool.
            left_vec/right_vec가 (0,0)이 아닌지(경계가 실제로 있는지).

    Returns:
        np.ndarray: shape (T-1,) bool.
            i번째 값이 True면 (i -> i+1) 선분을 그릴 수 있다.
    """
    lane_ok: Array = np.asarray(lane_point_valid_mask).astype(bool).reshape(
        -1)  # (T,)
    bound_ok: Array = np.asarray(boundary_point_valid_mask).astype(
        bool).reshape(-1)  # (T,)

    if lane_ok.shape[0] != bound_ok.shape[0]:
        raise ValueError(
            f"lane_point_valid_mask와 boundary_point_valid_mask의 길이가 같아야 합니다. "
            f"got {lane_ok.shape[0]} vs {bound_ok.shape[0]}")

    T: int = int(lane_ok.shape[0])
    if T < 2:
        return np.zeros((0,), dtype=bool)

    return (lane_ok[:-1] & lane_ok[1:] & bound_ok[:-1] & bound_ok[1:])  # (T-1,)


def _draw_lane_boundary_segments_with_plan(
    ax: plt.Axes,
    center_xy: Array,  # shape: (T, 2)
    boundary_xy: Array,  # shape: (T, 2)
    segment_drawable: Array,  # shape: (T-1,)
    plan: _LaneBoundaryDrawPlan,
    options: DrawingOptions,
    zorder: int,
) -> None:
    """경계선을 (단선/이중선/점선/실선) 계획(plan)에 맞춰 선분 단위로 그린다.

    이 함수는 이미 계산된 좌표를 받아서,
    segment_drawable이 True인 구간만 선분으로 그린다.

    - 단선(plan.is_double=False): 한 줄로 그림
    - 이중선(plan.is_double=True): 두 줄로 그림
      · 두 줄 사이 간격은 options.LANE_double_line_sep_m
      · "차선 중심 방향"을 기준으로 안쪽/바깥쪽을 나눠서 그린다

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        center_xy (np.ndarray): shape (T, 2). 센터라인 좌표.
        boundary_xy (np.ndarray): shape (T, 2). 경계 좌표(= center + left_vec/right_vec).
        segment_drawable (np.ndarray): shape (T-1,) bool.
            True인 i에 대해서만 (i -> i+1) 선분을 그린다.
        plan (_LaneBoundaryDrawPlan): 색/이중선/실선·점선 계획.
        options (DrawingOptions): 두께/간격 옵션.
        zorder (int): 그리기 순서.
    """
    if not plan.draw:
        return

    cxy: Array = np.asarray(center_xy)  # (T, 2)
    bxy: Array = np.asarray(boundary_xy)  # (T, 2)
    seg_ok: Array = np.asarray(segment_drawable).astype(bool).reshape(
        -1)  # (T-1,)

    if cxy.ndim != 2 or cxy.shape[1] != 2:
        raise ValueError(f"center_xy shape는 (T,2) 이어야 합니다. got {cxy.shape}")
    if bxy.ndim != 2 or bxy.shape[1] != 2:
        raise ValueError(f"boundary_xy shape는 (T,2) 이어야 합니다. got {bxy.shape}")
    if seg_ok.ndim != 1 or seg_ok.shape[0] != max(int(cxy.shape[0]) - 1, 0):
        raise ValueError(
            f"segment_drawable shape는 (T-1,) 이어야 합니다. got {seg_ok.shape}, T={cxy.shape[0]}"
        )

    seg_indices: Array = np.nonzero(seg_ok)[0]  # (K,)
    for j in seg_indices.tolist():
        c0: Array = cxy[j].astype(np.float32)  # (2,)
        c1: Array = cxy[j + 1].astype(np.float32)  # (2,)
        p0: Array = bxy[j].astype(np.float32)  # (2,)
        p1: Array = bxy[j + 1].astype(np.float32)  # (2,)

        if plan.is_double:
            inward_n: Array = _compute_inward_normal_for_segment(
                boundary_p0=p0,
                boundary_p1=p1,
                center_p0=c0,
                center_p1=c1,
            )  # (2,)
            _draw_double_line_segment(
                ax=ax,
                p0=p0,
                p1=p1,
                inward_normal=inward_n,
                sep_m=float(options.LANE_double_line_sep_m),
                color=plan.color,
                line_width=float(options.LANE_boundary_width),
                inner_linestyle=plan.inner_linestyle,
                outer_linestyle=plan.outer_linestyle,
                zorder=int(zorder),
            )
        else:
            _draw_single_line_segment(
                ax=ax,
                p0=p0,
                p1=p1,
                color=plan.color,
                line_width=float(options.LANE_boundary_width),
                linestyle=plan.inner_linestyle,
                zorder=int(zorder),
            )


def _draw_lane_boundary_highlight_segments(
    ax: plt.Axes,
    boundary_xy: Array,  # shape: (T, 2)
    segment_drawable: Array,  # shape: (T-1,)
    color: str,
    line_width: float,
    zorder: int,
) -> None:
    """경계선을 강조색으로 선분 단위로 그린다(이중선/점선 규칙 없이 단순 선).

    route 강조(CYAN)처럼 "그냥 눈에 띄게 덧그리기" 용도다.
    segment_drawable이 True인 구간만 이어서 그린다.

    Args:
        ax (plt.Axes): Matplotlib 축 객체.
        boundary_xy (np.ndarray): shape (T, 2). 경계 좌표.
        segment_drawable (np.ndarray): shape (T-1,) bool. True인 구간만 그림.
        color (str): 선 색.
        line_width (float): 선 두께.
        zorder (int): 그리기 순서.
    """
    bxy: Array = np.asarray(boundary_xy)  # (T, 2)
    seg_ok: Array = np.asarray(segment_drawable).astype(bool).reshape(
        -1)  # (T-1,)

    if bxy.ndim != 2 or bxy.shape[1] != 2:
        raise ValueError(f"boundary_xy shape는 (T,2) 이어야 합니다. got {bxy.shape}")
    if seg_ok.ndim != 1 or seg_ok.shape[0] != max(int(bxy.shape[0]) - 1, 0):
        raise ValueError(
            f"segment_drawable shape는 (T-1,) 이어야 합니다. got {seg_ok.shape}, T={bxy.shape[0]}"
        )

    seg_indices: Array = np.nonzero(seg_ok)[0]  # (K,)
    for j in seg_indices.tolist():
        p0: Array = bxy[j]  # (2,)
        p1: Array = bxy[j + 1]  # (2,)
        ax.plot(
            [float(p0[0]), float(p1[0])],
            [float(p0[1]), float(p1[1])],
            color=color,
            linewidth=float(line_width),
            zorder=int(zorder),
        )


def draw_lane_boundaries(
        ax: plt.Axes,
        lanes: Array,  # (lane_num, lane_len, 12)
        agent_route_lane_order: Optional[Array],  # (max_agent_num, lane_num)
        options: DrawingOptions,
        draw_token_int_list: Optional[List[int]] = None,
        lane_type: Optional[Array] = None,  # (lane_num, 4)
        left_line_type: Optional[
            Array] = None,  # (lane_num, 13) or (lane_num, 10)
        right_line_type: Optional[
            Array] = None,  # (lane_num, 13) or (lane_num, 10)
) -> None:
    """차선 좌/우 경계를 그린다.

    추가 규칙(이번 수정의 핵심)
    --------------------------
    - left_vec/right_vec의 (dx,dy)가 (0,0)이면 "해당 시점에는 경계가 없음"으로 본다.
    - 경계선을 선분으로 그릴 때는,
      연속된 두 점 모두 "차선 포인트 유효 + 경계 벡터 유효"일 때만 그린다.
      그래서 경계가 끊기는 구간은 선도 끊어진다.

    - left/right_line_type이 (L,13)으로 확장되면서,
      특정 타입은 선이 아니라 점으로 찍어서 표시한다.
      · road edge boundary -> '*'
      · road edge median   -> 'x'
      · unknown            -> 'o'
    """
    if lanes is None or np.asarray(lanes).size == 0:
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

    lanes_arr: Array = np.asarray(lanes)  # (L, T, 12)
    L: int = int(lanes_arr.shape[0])
    eps: float = float(options.invalid_eps)

    # shape 체크(있을 때만)
    if lane_type is not None:
        lane_type = np.asarray(lane_type)
        if lane_type.ndim != 2 or lane_type.shape != (L, 4):
            raise ValueError(
                f"lane_type shape는 (lane_num,4) 이어야 합니다. got {lane_type.shape}, L={L}"
            )

    if left_line_type is not None:
        left_line_type = np.asarray(left_line_type)
        if left_line_type.ndim != 2 or left_line_type.shape[
                0] != L or left_line_type.shape[1] not in (10, 13):
            raise ValueError(
                f"left_line_type shape는 (lane_num,13) 또는 (lane_num,10) 이어야 합니다. "
                f"got {left_line_type.shape}, L={L}")

    if right_line_type is not None:
        right_line_type = np.asarray(right_line_type)
        if right_line_type.ndim != 2 or right_line_type.shape[
                0] != L or right_line_type.shape[1] not in (10, 13):
            raise ValueError(
                f"right_line_type shape는 (lane_num,13) 또는 (lane_num,10) 이어야 합니다. "
                f"got {right_line_type.shape}, L={L}")

    boundary_vec_eps: float = float(options.LANE_boundary_vector_invalid_eps)
    skip_zero_boundary_vec: bool = True

    for idx in range(L):
        lane_i: Array = lanes_arr[idx]  # (T, 12)

        center: Array = lane_i[:, 0:2]  # (T, 2)
        left_vec: Array = lane_i[:, 4:6]  # (T, 2)  [dx, dy]
        right_vec: Array = lane_i[:, 6:8]  # (T, 2)  [dx, dy]

        lane_point_valid: Array = np.any(np.abs(lane_i[:, :8]) > eps,
                                         axis=1)  # (T,)

        T: int = int(center.shape[0])
        if T < 2:
            continue

        # (A) "경계가 실제로 존재하는 점" True/False 배열 만들기
        left_boundary_point_valid: Array = _compute_boundary_vector_point_valid_mask(
            boundary_vec=left_vec,  # (T, 2)
            eps=boundary_vec_eps,
            skip_zero_vector=skip_zero_boundary_vec,
        )  # (T,)

        right_boundary_point_valid: Array = _compute_boundary_vector_point_valid_mask(
            boundary_vec=right_vec,  # (T, 2)
            eps=boundary_vec_eps,
            skip_zero_vector=skip_zero_boundary_vec,
        )  # (T,)

        # (B) 실제로 그릴 "선분" True/False 배열 만들기
        left_segment_drawable: Array = _compute_consecutive_segment_drawable_mask(
            lane_point_valid_mask=lane_point_valid,  # (T,)
            boundary_point_valid_mask=left_boundary_point_valid  # (T,)
        )  # (T-1,)

        right_segment_drawable: Array = _compute_consecutive_segment_drawable_mask(
            lane_point_valid_mask=lane_point_valid,  # (T,)
            boundary_point_valid_mask=right_boundary_point_valid  # (T,)
        )  # (T-1,)

        # (C) 실제 경계 좌표 계산
        left_xy: Array = center + left_vec  # (T, 2)
        right_xy: Array = center + right_vec  # (T, 2)

        # lane_type 기반 기본 색(현 코드 유지)
        base_color: str = options.LANE_lane_boundary_color

        # left/right line 계획
        left_plan: _LaneBoundaryDrawPlan = _make_lane_boundary_draw_plan(
            left_line_type[idx] if left_line_type is not None else None,
            default_color=base_color,
            options=options,
        )
        right_plan: _LaneBoundaryDrawPlan = _make_lane_boundary_draw_plan(
            right_line_type[idx] if right_line_type is not None else None,
            default_color=base_color,
            options=options,
        )

        # (D) 경계 그리기: 선(line) 또는 점(marker)
        if left_plan.draw and left_plan.draw_mode == "marker":
            point_drawable: Array = (lane_point_valid &
                                     left_boundary_point_valid)  # (T,)
            _draw_lane_boundary_points(
                ax=ax,
                boundary_xy=left_xy,  # (T,2)
                point_drawable=point_drawable,  # (T,)
                marker=str(left_plan.marker or "o"),
                edge_color=left_plan.color,
                marker_size=float(options.ROAD_road_edge_marker_size),
                edge_line_width=float(options.ROAD_road_edge_line_width),
                zorder=1,
            )
        else:
            _draw_lane_boundary_segments_with_plan(
                ax=ax,
                center_xy=center,  # (T, 2)
                boundary_xy=left_xy,  # (T, 2)
                segment_drawable=left_segment_drawable,  # (T-1,)
                plan=left_plan,
                options=options,
                zorder=1,
            )

        if right_plan.draw and right_plan.draw_mode == "marker":
            point_drawable: Array = (lane_point_valid &
                                     right_boundary_point_valid)  # (T,)
            _draw_lane_boundary_points(
                ax=ax,
                boundary_xy=right_xy,  # (T,2)
                point_drawable=point_drawable,  # (T,)
                marker=str(right_plan.marker or "o"),
                edge_color=right_plan.color,
                marker_size=float(options.ROAD_road_edge_marker_size),
                edge_line_width=float(options.ROAD_road_edge_line_width),
                zorder=1,
            )
        else:
            _draw_lane_boundary_segments_with_plan(
                ax=ax,
                center_xy=center,  # (T, 2)
                boundary_xy=right_xy,  # (T, 2)
                segment_drawable=right_segment_drawable,  # (T-1,)
                plan=right_plan,
                options=options,
                zorder=1,
            )

        # ===== 기존 route 강조(필요 시) =====
        # 점(marker) 모드에서는 route 강조선을 그려도 의미가 애매해서,
        # 기존처럼 "선(line)일 때만" 강조를 유지한다.
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
                highlight_color: str = CYAN

                if left_plan.draw and left_plan.draw_mode == "line":
                    _draw_lane_boundary_highlight_segments(
                        ax=ax,
                        boundary_xy=left_xy,  # (T, 2)
                        segment_drawable=left_segment_drawable,  # (T-1,)
                        color=highlight_color,
                        line_width=float(options.LANE_boundary_width),
                        zorder=2,
                    )

                if right_plan.draw and right_plan.draw_mode == "line":
                    _draw_lane_boundary_highlight_segments(
                        ax=ax,
                        boundary_xy=right_xy,  # (T, 2)
                        segment_drawable=right_segment_drawable,  # (T-1,)
                        color=highlight_color,
                        line_width=float(options.LANE_boundary_width),
                        zorder=2,
                    )


def _signal_rows4_to_state_indices(
        signals_rowwise: Array,  # shape: (T, 4)
) -> Array:
    """신호 상태(4칸)를 0~3 정수로 바꾼 배열을 만든다.

    lanes의 signal 값은 한 시점마다 길이 4짜리 배열로 들어온다고 가정한다.
    (예: [green, yellow, red, unknown] 순서)

    이 함수는 각 시점의 4칸 중 "값이 가장 큰 위치"를 상태로 고른다.
    다만 어떤 시점은 4칸이 전부 0일 수 있는데,
    그 경우는 "알 수 없음(unknown)"으로 보고 3을 넣는다.

    Args:
        signals_rowwise (np.ndarray):
            shape = (T, 4)
            T는 차선 포인트 개수.

    Returns:
        np.ndarray:
            shape = (T,)
            각 원소는 0~3.
    """
    arr: Array = np.asarray(signals_rowwise)  # (T, 4)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError(
            f"signals_rowwise shape는 (T,4) 이어야 합니다. got {arr.shape}")

    row_sum: Array = np.sum(np.abs(arr), axis=1)  # (T,)
    idx: Array = np.argmax(arr, axis=1).astype(np.int64)  # (T,)
    idx = np.where(row_sum > 0.0, idx, 3).astype(np.int64)  # (T,)
    return idx


def _choose_lane_centerline_color(
    signal_state_idx: int,
    lane_type_color: Optional[str],
    options: DrawingOptions,
) -> str:
    """센터라인 색을 고른다(신호등이 lane_type보다 우선).

    우선순위:
    1) 신호등이 green/yellow/red(0/1/2)이면 그 색을 사용한다.
    2) 신호등이 unknown(3)이면,
       lane_type 색이 있으면 lane_type 색을 사용한다.
    3) 둘 다 없으면 unknown 색(회색)을 사용한다.

    Args:
        signal_state_idx (int):
            0~3 값. (0: green, 1: yellow, 2: red, 3: unknown)
        lane_type_color (Optional[str]):
            lane_type이 있을 때 미리 계산해 둔 센터라인 색.
        options (DrawingOptions):
            신호등 색 테이블(options.LANE_signal_colors)을 사용한다.

    Returns:
        str:
            matplotlib에서 쓸 색 문자열.
    """
    if int(signal_state_idx) in (0, 1, 2):
        return options.LANE_signal_colors.get(int(signal_state_idx), GRAY)

    if lane_type_color is not None:
        return lane_type_color

    return options.LANE_signal_colors.get(3, GRAY)


def draw_lane_centerlines(
        ax: plt.Axes,
        lanes: Array,  # (lane_num, lane_len, 12)
        lanes_speed_limit: Array,  #  (lane_num, 1)
        lanes_has_speed_limit: Array,  # (lane_num, 1)
        options: DrawingOptions,
        agent_route_lane_order: Optional[
            Array] = None,  # (max_agent_num, lane_num)
        draw_token_int_list: Optional[List[int]] = None,
        lane_type: Optional[Array] = None,  # (lane_num, 4) or None
) -> None:
    """차선 센터라인을 그린다.

    모드
    ----
    1) 텍스트 모드(agent_route_lane_order가 사용되는 경우)
        - 센터라인 선은 그리지 않고,
        - 차선 위에 에이전트-차선 매핑 정보를 글씨로 표시한다.
        (기존 동작 유지)

    2) 센터라인 선 모드(일반 모드)
        - 이번 수정으로 우선순위가 다음처럼 바뀐다.
          (신호등 색이 lane_type보다 우선)

          1) 신호등이 green/yellow/red이면 → 신호등 색
          2) 신호등이 unknown이면
             lane_type이 있으면 → lane_type 색
             lane_type이 없으면 → unknown(회색)

    Args:
        ax (plt.Axes):
            Matplotlib 축 객체.
        lanes (np.ndarray):
            shape = (lane_num, lane_len, 12)
            · 0-1: centerline (x,y)
            · 8-11: signal 값(4칸)
        lanes_speed_limit (np.ndarray):
            shape = (lane_num, 1)
        lanes_has_speed_limit (np.ndarray):
            shape = (lane_num, 1)
        options (DrawingOptions):
            그리기 옵션.
        agent_route_lane_order (Optional[np.ndarray]):
            텍스트 모드에서만 사용.
        draw_token_int_list (Optional[List[int]]):
            텍스트 모드에서 특정 에이전트만 표시할 때 사용.
        lane_type (Optional[np.ndarray]):
            shape = (lane_num, 4).
            신호등이 unknown일 때만 센터라인 색의 후보로 사용된다.
    """
    if lanes is None or lanes.size == 0:
        return

    # route 텍스트 모드 조건(기존 로직 유지)
    if not (options.LANE_draw_npc_agent_route is True and
            options.LANE_npc_agent_route_draw_mode == "centerline"):
        agent_route_lane_order = None

    eps: float = float(options.invalid_eps)
    lane_num: int = int(lanes.shape[0])

    # lane_type shape 체크(있을 때만)
    lane_type_arr: Optional[Array] = None
    if lane_type is not None:
        lane_type_arr = np.asarray(lane_type)
        if lane_type_arr.ndim != 2 or lane_type_arr.shape != (lane_num, 4):
            raise ValueError(
                f"lane_type shape는 (lane_num,4) 이어야 합니다. got {lane_type_arr.shape}, lane_num={lane_num}"
            )

    # 속도 제한 텍스트 표시 여부(기존 유지)
    use_speed_limit_label: bool = (agent_route_lane_order is None and
                                   options.LANE_draw_vel_limit and
                                   (lanes_speed_limit is not None) and
                                   (lanes_has_speed_limit is not None))

    # ────────────── (B) 텍스트 표기 모드 ──────────────
    if agent_route_lane_order is not None:
        draw_all: bool = (draw_token_int_list is None)
        vstep: float = 0.3

        for lane_idx in range(lane_num):
            lane_j: Array = lanes[lane_idx]  # (lane_len, 12)
            lane_j_center: Array = lane_j[:, 0:2]  # (lane_len, 2)

            lane_point_valid_mask: Array = np.any(np.abs(lane_j[:, :8]) > eps,
                                                  axis=1)  # (lane_len,)

            ranks_j: Array = agent_route_lane_order[:,
                                                    lane_idx]  # (max_agent_num,)
            valid_agent_idxs: Array = np.nonzero(ranks_j >= 0)[0]
            if valid_agent_idxs.size == 0:
                continue

            for point_idx in range(lane_j_center.shape[0]):
                if point_idx % 4 != 0:
                    continue
                if not bool(lane_point_valid_mask[point_idx]):
                    continue

                point_x: float = float(lane_j_center[point_idx, 0])
                point_y: float = float(lane_j_center[point_idx, 1])

                for count, agent_idx in enumerate(valid_agent_idxs.tolist()):
                    if (not draw_all) and (agent_idx
                                           not in draw_token_int_list):
                        continue
                    rank_ij: int = int(ranks_j[int(agent_idx)])
                    label: str = f"{rank_ij}"

                    ax.text(
                        point_x,
                        point_y + vstep * float(count),
                        label,
                        color=options.LANE_route_agent_index_color,
                        fontsize=options.LANE_AGENT_index_fontsize,
                        ha="center",
                        va="bottom",
                        zorder=3,
                    )
        return

    # ────────────── (A) 센터라인 선 모드 ──────────────
    for lane_idx, lane_i in enumerate(lanes):
        center: Array = lane_i[:, 0:2]  # (lane_len, 2)
        signals: Array = lane_i[:, 8:12]  # (lane_len, 4)
        valid_mask: Array = np.any(np.abs(lane_i[:, :8]) > eps, axis=1)

        if center.shape[0] < 2:
            continue

        # lane_type 색(있으면 미리 계산)
        lane_type_color: Optional[str] = None
        if lane_type_arr is not None:
            lane_type_row4: Array = lane_type_arr[lane_idx]  # (4,)
            lane_type_color = _lane_type_row4_to_centerline_color(
                lane_type_row4=lane_type_row4,
                options=options,
            )

        # 신호 상태 index(항상 계산) — 신호가 unknown인지 판단하려고 필요
        signal_state_idx: Array = _signal_rows4_to_state_indices(
            signals)  # (lane_len,)

        # 속도 제한 텍스트 준비(기존 유지)
        has_speed_limit: bool = False
        speed_kmh: Optional[float] = None
        if use_speed_limit_label:
            flag_val = lanes_has_speed_limit[lane_idx]
            if np.ndim(flag_val) > 0:
                flag_val = flag_val[0]
            if bool(flag_val):
                has_speed_limit = True
                speed_val = lanes_speed_limit[lane_idx]
                if np.ndim(speed_val) > 0:
                    speed_val = speed_val[0]
                speed_kmh = float(speed_val) * 3.6

        # 선분 그리기: "신호 색 우선"
        for j in range(center.shape[0] - 1):
            if not (bool(valid_mask[j]) and bool(valid_mask[j + 1])):
                continue

            c0: Array = center[j]
            c1: Array = center[j + 1]

            color: str = _choose_lane_centerline_color(
                signal_state_idx=int(signal_state_idx[j]),
                lane_type_color=lane_type_color,
                options=options,
            )

            ax.plot(
                [float(c0[0]), float(c1[0])],
                [float(c0[1]), float(c1[1])],
                color=color,
                linewidth=1.2,
                linestyle=(0, (4, 4)),
                zorder=2,
            )

        # 속도 제한 텍스트(기존 유지)
        if use_speed_limit_label and has_speed_limit and (speed_kmh
                                                          is not None):
            valid_indices: Array = np.nonzero(valid_mask)[0]
            if valid_indices.size > 0:
                mid_idx: int = int(valid_indices[len(valid_indices) // 2])
                px: float = float(center[mid_idx, 0])
                py: float = float(center[mid_idx, 1])
                label: str = f"{speed_kmh:.1f}"

                ax.text(
                    px,
                    py,
                    label,
                    color=options.LANE_speed_color,
                    fontsize=options.LANE_speed_fontsize,
                    ha="center",
                    va="center",
                    zorder=3,
                )


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
    diff_token_to_interp_np_traj_wrt_ego: Optional[Dict[
        str, Array]],  # DIFF_draw_diff_interp_np_traj_11
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
        if options.DIFF_draw_diff_interp_np_traj_11:
            draw_traj_dict_as_unfilled_rects(
                ax=ax,
                token_to_traj_11=diff_token_to_interp_np_traj_wrt_ego,
                style=options.DIFF_future_interp_style,
                options=options,
                zorder=25,
                draw_token_list=draw_token_list,
                annotate_token=False,
                draw_velocity=options.DIFF_draw_diff_interp_np_traj_11_vel,
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
        if options.DIFF_draw_diff_interp_np_traj_11:
            draw_traj_dict_as_non_square(
                ax=ax,
                token_to_traj_11=diff_token_to_interp_np_traj_wrt_ego,
                style=options.DIFF_future_interp_style,
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
             output_data: WorldModelFeature, draw_option: DrawingOptions):
    if draw_option.EGO_draw_ego_past:
        draw_ego_past(ax, input_data.get("ego_agent_past"), draw_option)
    if draw_option.EGO_draw_ego_agent_next_11_dim:
        if draw_option.EGO_draw_diffusion:
            ego_next_state = output_data.get("ego_next_wp_wrt_ego")
            if ego_next_state is not None:
                ego_next_state = ego_next_state[None, ...]  # (1, 11)
        else:
            ego_next_state = input_data.get("ego_agent_next_11_dim")
        if ego_next_state is not None:
            draw_ego_agent_next_11_dim(ax, ego_next_state, draw_option)
    ### [EGO FUTURE PLANNER] ###
    if draw_option.EGO_draw_planner_future_11_dim:
        if draw_option.EGO_draw_diffusion:
            if draw_option.EGO_draw_diffusion_mode == "direct":
                ego_future_11_dim = output_data.get("ego_np_traj_11_wrt_ego")
            elif draw_option.EGO_draw_diffusion_mode == "integrate":
                ego_future_11_dim = output_data.get(
                    "ego_np_int_traj_11_wrt_ego")

            elif draw_option.EGO_draw_diffusion_mode == "interp":
                ego_future_11_dim = output_data.get(
                    "ego_interp_np_traj_wrt_ego")

            else:
                raise ValueError(
                    f"Unsupported EGO_draw_diffusion_mode: {draw_option.EGO_draw_diffusion_mode}"
                )
        else:
            ego_future_11_dim = input_data.get("ego_future_gt_11_dim", None)
        if ego_future_11_dim is not None:
            draw_planner_future_11_dim(ax, ego_future_11_dim, draw_option)
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

    if draw_option.DIFF_draw_future_gt_3_dim_wo_token:
        # 설정 충돌 방지: 둘 다 True면 에러
        if draw_option.DIFF_draw_diff_future_gt_3_dim:
            raise ValueError(
                "DIFF_draw_future_gt_3_dim_wo_token=True 인 경우 "
                "DIFF_draw_diff_future_gt_3_dim 는 반드시 False 여야 합니다.")
        if neighbor_future_gt_3_dim is None:
            raise ValueError("DIFF_draw_future_gt_3_dim_wo_token=True 인데 "
                             "`neighbor_future_gt_3_dim` 이 input_data 에 없습니다.")
        if neighbor_track_token is None:
            raise ValueError("DIFF_draw_future_gt_3_dim_wo_token=True 인데 "
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
        "diff_token_to_np_gen_traj_11_wrt_ego", None)  # (1+T, 11)
    diff_token_to_np_int_traj_wrt_ego = output_data.get(
        "diff_token_to_np_int_traj_wrt_ego", None)  # (1 + future_len, 4)
    diff_token_to_np_int_traj_11_wrt_ego = output_data.get(
        "diff_token_to_np_int_traj_11_wrt_ego", None)  # (1 + future_len, 11)
    diff_token_to_interp_np_traj_wrt_ego = output_data.get(
        "diff_token_to_interp_np_traj_wrt_ego", None)  # (1 + future_len, 11)
    draw_diff_future_traj_w_square(ax, diff_token_to_np_gen_traj_11_wrt_ego,
                                   diff_token_to_np_int_traj_wrt_ego,
                                   diff_token_to_np_int_traj_11_wrt_ego,
                                   diff_token_to_interp_np_traj_wrt_ego,
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
    """차선(경계/센터라인/추가 center_lanes/road_line) 레이어를 그린다.

    Args:
        ax: Matplotlib Axes.
        input_data: world model 입력 데이터(dict).
        draw_option: 렌더링 옵션.
        draw_token_list: 특정 에이전트 토큰만 텍스트 표기하고 싶을 때 사용. None이면 전체.
    """
    lanes = input_data.get("lanes")  # (lane_num, lane_len, 12) or None
    lanes_speed_limit = input_data.get(
        "lanes_speed_limit")  # (lane_num, 1) or None
    lanes_has_speed_limit = input_data.get(
        "lanes_has_speed_limit")  # (lane_num, 1) or None

    lane_type = input_data.get("lane_type", None)  # (lane_num, 4) or None
    left_line_type = input_data.get("left_line_type",
                                    None)  # (lane_num, 10) or None
    right_line_type = input_data.get("right_line_type",
                                     None)  # (lane_num, 10) or None

    agent_route_lane_order: Optional[Array] = input_data.get(
        "agent_route_lane_order", None)  # (A, lane_num) or None

    draw_token_int_list: Optional[List[int]] = get_agent_idx_from_tokens(
        draw_token_list, input_data.get("neighbor_track_token", None))

    # (A) 기존 lanes 기반: 경계 + 센터라인
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
            lane_type=lane_type,
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
    draw_ego(ax, input_data, output_data, draw_option)
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
def _list_sorted_png_frame_paths(save_dir: str) -> List[str]:
    """폴더 안의 PNG 프레임들을 '숫자 파일명' 기준으로 정렬해 반환합니다.

    이 함수가 하는 일
    ---------------
    - save_dir 안에서 확장자가 .png 인 파일을 찾습니다.
    - 파일명(stem)이 '0', '1', '2' 처럼 숫자로만 된 것만 프레임으로 인정합니다.
      (예: "0001.png" 도 숫자이므로 포함됩니다.)
    - 숫자 값 기준으로 오름차순 정렬합니다.
      (문자열 정렬이 아니라, 2 < 10 < 11 같은 "숫자 정렬"입니다.)

    Args:
        save_dir (str):
            프레임 PNG들이 들어있는 폴더 경로. shape: ()

    Returns:
        List[str]:
            정렬된 PNG 파일 경로 리스트.
            각 원소는 파일 경로 문자열이며, 길이는 프레임 개수입니다. shape: (N,)

    Raises:
        FileNotFoundError:
            폴더가 없을 때 발생합니다.
        ValueError:
            숫자 파일명의 PNG가 하나도 없을 때 발생합니다.
    """
    import os
    from pathlib import Path

    dir_path = Path(save_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"save_dir 폴더가 없습니다: {save_dir}")

    png_paths = []
    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".png":
            continue
        if not p.stem.isdigit():
            continue
        png_paths.append(p)

    if len(png_paths) == 0:
        raise ValueError(
            f"'{save_dir}' 안에서 숫자 파일명(예: 0.png, 1.png, 2.png ...)의 PNG를 찾지 못했습니다."
        )

    png_paths.sort(key=lambda x: int(x.stem))
    return [str(p) for p in png_paths]


def _load_png_as_rgb_uint8(
    png_path: str,
    target_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    """PNG 한 장을 (H, W, 3) RGB uint8 배열로 읽고, 필요하면 크기를 맞춥니다.

    왜 이 함수가 필요한가?
    ----------------------
    - PNG는 (H, W) 흑백이거나, (H, W, 4) RGBA처럼 알파 채널이 있을 수 있습니다.
    - 영상으로 만들려면 보통 (H, W, 3) RGB 형태가 가장 다루기 쉽습니다.
    - 프레임마다 크기가 다르면 영상 writer에서 오류가 나기 쉬워서,
      첫 프레임 크기로 통일합니다.

    Args:
        png_path (str):
            읽을 PNG 파일 경로. shape: ()
        target_hw (Optional[Tuple[int, int]]):
            (target_h, target_w).
            - None이면 원본 크기 그대로 사용합니다.
            - 값이 있으면 해당 크기로 리사이즈합니다. shape: (2,)

    Returns:
        np.ndarray:
            shape = (H, W, 3), dtype = uint8
            RGB 이미지 배열입니다.
    """
    from PIL import Image

    with Image.open(png_path) as img:
        img_rgb = img.convert("RGB")  # (W, H) 기반 내부 포맷이지만 결과는 RGB 3채널
        if target_hw is not None:
            target_h, target_w = int(target_hw[0]), int(target_hw[1])
            # PIL resize는 (W, H) 순서로 받습니다.
            if (img_rgb.size[0] != target_w) or (img_rgb.size[1] != target_h):
                img_rgb = img_rgb.resize((target_w, target_h), resample=Image.BILINEAR)

        frame_rgb: np.ndarray = np.asarray(img_rgb, dtype=np.uint8)  # (H, W, 3)
    return frame_rgb


def _get_default_video_fps() -> int:
    """영상 FPS를 기본값으로 결정합니다.

    이 함수가 하는 일
    ---------------
    - 기본 FPS는 10으로 둡니다.
    - 환경변수 DRAW_MACHINE_VIDEO_FPS 가 있으면 그 값을 우선 사용합니다.
      (예: export DRAW_MACHINE_VIDEO_FPS=20)

    Returns:
        int:
            FPS 값(최소 1). shape: ()
    """
    import os

    raw = str(os.environ.get("DRAW_MACHINE_VIDEO_FPS", "")).strip()
    if raw == "":
        return 10
    try:
        fps = int(raw)
        return int(max(1, fps))
    except ValueError:
        return 10


from typing import Tuple
import numpy as np


def _pad_rgb_uint8_frame_to_even_hw(frame_rgb: np.ndarray) -> np.ndarray:
    """RGB 프레임의 가로/세로 픽셀 수를 '짝수'로 맞춰서 반환합니다.

    이 함수가 필요한 이유
    --------------------
    mp4(H.264)로 저장할 때는 영상 프레임의 가로/세로 픽셀 수가
    둘 다 2로 나눠 떨어져야 하는 경우가 많습니다.
    (지금처럼 세로가 2275 같은 홀수면 저장이 실패합니다.)

    그래서 이 함수는
    - 세로(H)가 홀수면: 아래쪽에 검은색 1픽셀 줄을 추가
    - 가로(W)가 홀수면: 오른쪽에 검은색 1픽셀 줄을 추가
    해서 (H, W)를 짝수로 만들어 줍니다.

    Args:
        frame_rgb (np.ndarray):
            shape = (H, W, 3), dtype = uint8
            RGB 이미지 프레임입니다.

    Returns:
        np.ndarray:
            shape = (H_even, W_even, 3), dtype = uint8
            가로/세로가 짝수로 맞춰진 RGB 프레임입니다.

    Raises:
        ValueError:
            입력 프레임이 (H, W, 3) 형태가 아닐 때 발생합니다.
    """
    frame_rgb_arr = np.asarray(frame_rgb)

    if frame_rgb_arr.ndim != 3 or frame_rgb_arr.shape[2] != 3:
        raise ValueError(
            f"frame_rgb는 shape (H, W, 3) 이어야 합니다. got {frame_rgb_arr.shape}"
        )

    if frame_rgb_arr.dtype != np.uint8:
        frame_rgb_arr = frame_rgb_arr.astype(np.uint8)

    height: int = int(frame_rgb_arr.shape[0])
    width: int = int(frame_rgb_arr.shape[1])

    pad_h: int = int(height % 2)  # 0 또는 1
    pad_w: int = int(width % 2)   # 0 또는 1

    if pad_h == 0 and pad_w == 0:
        return frame_rgb_arr

    padded_frame: np.ndarray = np.pad(
        frame_rgb_arr,
        pad_width=((0, pad_h), (0, pad_w), (0, 0)),
        mode="constant",
        constant_values=0,  # 검은색
    )
    return padded_frame

from typing import List
import numpy as np


def _write_mp4_from_png_frames(
    frame_paths: List[str],
    output_mp4_path: str,
    fps: int,
) -> None:
    """PNG 프레임 시퀀스로 MP4 영상을 저장합니다(가로/세로를 짝수로 맞춤).

    핵심 동작
    --------
    - 첫 프레임 크기 (H, W)를 기준으로 모든 프레임을 같은 크기로 맞춥니다.
    - 그 다음, mp4 인코더가 실패하지 않도록 프레임의 (H, W)를 짝수로 만듭니다.
      (홀수면 아래/오른쪽에 검은색 1픽셀을 추가)

    Args:
        frame_paths (List[str]):
            PNG 경로 리스트. 시간 순서대로 정렬되어 있어야 합니다. shape: (N,)
        output_mp4_path (str):
            저장할 mp4 파일 경로. shape: ()
        fps (int):
            초당 프레임 수. shape: ()

    Raises:
        RuntimeError:
            mp4 저장에 필요한 라이브러리/인코더가 없어서 실패할 때 발생합니다.
    """
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "mp4 저장을 위해 imageio가 필요합니다. "
            "설치: pip install imageio imageio-ffmpeg"
        ) from e

    if len(frame_paths) == 0:
        raise ValueError("frame_paths가 비어 있습니다. PNG 프레임이 필요합니다.")

    # (1) 첫 프레임으로 기준 크기(H, W) 결정
    first_frame_rgb: np.ndarray = _load_png_as_rgb_uint8(
        png_path=frame_paths[0],
        target_hw=None,
    )  # shape: (H, W, 3)

    base_h: int = int(first_frame_rgb.shape[0])
    base_w: int = int(first_frame_rgb.shape[1])
    target_hw = (base_h, base_w)

    # (2) writer 생성
    # macro_block_size=1은 "16의 배수로 강제 리사이즈" 같은 걸 피하는 데 도움
    try:
        writer = imageio.get_writer(
            output_mp4_path,
            fps=int(fps),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
    except Exception:
        writer = imageio.get_writer(output_mp4_path, fps=int(fps))

    # (3) 프레임을 읽어서 -> (H,W,3) 맞추고 -> (H,W)를 짝수로 만든 뒤 append
    with writer:
        for png_path in frame_paths:
            frame_rgb: np.ndarray = _load_png_as_rgb_uint8(
                png_path=png_path,
                target_hw=target_hw,
            )  # shape: (base_h, base_w, 3)

            frame_rgb_even: np.ndarray = _pad_rgb_uint8_frame_to_even_hw(
                frame_rgb
            )  # shape: (H_even, W_even, 3)

            writer.append_data(frame_rgb_even)



def _write_gif_from_png_frames(
    frame_paths: List[str],
    output_gif_path: str,
    fps: int,
) -> None:
    """PNG 프레임 시퀀스로 GIF 영상을 저장합니다.

    구현 방식
    --------
    - mp4와 동일하게 프레임을 한 장씩 읽어서 writer에 바로 추가합니다.
    - GIF는 보통 프레임 시간(초/프레임)을 duration으로 설정합니다.
      duration = 1 / fps

    Args:
        frame_paths (List[str]):
            PNG 경로 리스트. 시간 순서대로 정렬되어 있어야 합니다. shape: (N,)
        output_gif_path (str):
            저장할 gif 파일 경로. shape: ()
        fps (int):
            초당 프레임 수. shape: ()

    Raises:
        RuntimeError:
            gif 저장에 필요한 라이브러리/플러그인이 없어서 실패할 때 발생합니다.
    """
    try:
        import imageio.v2 as imageio
    except Exception as e:
        raise RuntimeError(
            "gif 저장을 위해 imageio가 필요합니다. 설치: pip install imageio"
        ) from e

    first_frame = _load_png_as_rgb_uint8(frame_paths[0], target_hw=None)  # (H, W, 3)
    target_h, target_w = int(first_frame.shape[0]), int(first_frame.shape[1])

    duration_sec = 1.0 / float(max(1, int(fps)))

    # mode="I"는 프레임을 이어붙이는 방식(일반적인 애니메이션 GIF)
    with imageio.get_writer(output_gif_path, mode="I", duration=duration_sec) as writer:
        for p in frame_paths:
            frame = _load_png_as_rgb_uint8(p, target_hw=(target_h, target_w))  # (H, W, 3)
            writer.append_data(frame)


def make_video_from_all_png(save_dir: str, draw_scenario_id: str,
                            new_save_dir: str,
                            run_count: Optional[int] = None,
                            ) -> None:
    """폴더 내 PNG들을 시간 순서대로 이어서 mp4/gif 영상을 저장합니다.

    전제(입력 폴더 구조)
    -------------------
    - save_dir 은 폴더이고, 그 안에 아래처럼 저장되어 있다고 가정합니다.

      0.png, 1.png, 2.png, ...
      10.png, 11.png, ...

    - 파일명은 "숫자"여야 하며, 이 숫자가 시간 순서를 의미합니다.

    저장 결과
    --------
    - mp4:  new_save_dir / f"{draw_scenario_id}.mp4"
    - gif:  new_save_dir / f"{draw_scenario_id}.gif"

    구현 특징
    --------
    - 숫자 기준 정렬(문자열 정렬이 아님)
    - 프레임 크기가 서로 다르면, 첫 프레임 크기 (H, W)에 맞춰 리사이즈
    - 프레임을 한 장씩 읽어서 바로 writer에 넣어 메모리 사용을 줄임
      · 각 프레임 배열 shape = (H, W, 3), dtype=uint8

    Args:
        save_dir (str):
            프레임 PNG들이 들어있는 폴더 경로. shape: ()
        draw_scenario_id (str):
            저장 파일명에 사용할 시나리오 ID. shape: ()

    Returns:
        None
    """
    import os
    from pathlib import Path

    frame_paths: List[str] = _list_sorted_png_frame_paths(save_dir)
    fps: int = _get_default_video_fps()

    out_dir = Path(new_save_dir)
    if run_count is None:
        out_mp4_path = str(out_dir / f"{draw_scenario_id}.mp4")
        out_gif_path = str(out_dir / f"{draw_scenario_id}.gif")
    else:
        out_mp4_path = str(out_dir / f"{draw_scenario_id}_{run_count}.mp4")
        out_gif_path = str(out_dir / f"{draw_scenario_id}_{run_count}.gif")

    # 같은 이름 파일이 있으면 덮어쓰는 게 자연스러워서, 미리 지우는 방식을 사용합니다.
    # (writer가 덮어쓰기를 지원하지 않는 환경도 있을 수 있어서 안전 처리)
    for out_path in (out_mp4_path, out_gif_path):
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception:
            pass

    _write_mp4_from_png_frames(
        frame_paths=frame_paths,
        output_mp4_path=out_mp4_path,
        fps=int(fps),
    )
    _write_gif_from_png_frames(
        frame_paths=frame_paths,
        output_gif_path=out_gif_path,
        fps=int(fps),
    )

if __name__ == "__main__":
    save_dir = "/home/user/PycharmProjects/Diffusion-Planner/training_log/nuplan_womd/2025-12-25-05:37:26/debug_vis_1bc8784d42f1b632"
    draw_scenario_id = "1bc8784d42f1b632"
    new_save_dir = "/home/user/PycharmProjects/Diffusion-Planner/training_log/nuplan_womd/2025-12-25-05:37:26"
    make_video_from_all_png(save_dir, draw_scenario_id, new_save_dir)