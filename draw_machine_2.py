from __future__ import annotations
import render_fast_collections
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.colors import to_rgba

Array = np.ndarray
WorldModelFeature = Dict[str, Array]
TokenTrajDict = Dict[str, Array]  # value: (future_len, 4) with [x, y, cos, sin]

BLACK = "#000000"
PURPLE = "#E6E6FA"
RED = "#D50000"
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
                            axis=2)  # (agent_num, T)
        if np.any(valid_mask):
            xy = neigh[:, :, 0:2][valid_mask]
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    # neighbor future points: (agent_num, future_len, 3) -> (x,y)만 사용
    neigh_fut = input_data.get("neighbor_future_gt_3_dim")
    if neigh_fut is not None and neigh_fut.size > 0:
        if neigh_fut.ndim != 3 or neigh_fut.shape[-1] != 3:
            raise ValueError(
                "neighbor_agents_future는 (agent_num, future_len, 3) 이어야 합니다.")
        valid_mask = (np.abs(neigh_fut[..., 0]) > eps) | (np.abs(
            neigh_fut[..., 1]) > eps)  # (A, T)
        if np.any(valid_mask):
            xy = neigh_fut[..., :2][valid_mask]  # (K, 2)
            xs_local.extend(xy[:, 0].tolist())
            ys_local.extend(xy[:, 1].tolist())

    return xs_local, ys_local


# [ADD]
def _collect_valid_xy_from_output_data(
    output_data: Dict[str, Any],
    options: DrawingOptions,
) -> Tuple[List[float], List[float]]:
    """output_data의 모든 key를 검사하여 valid (x,y) 좌표만 수집해 반환.

    예상 키와 타입
    - diff_token_to_np_gen_traj_wrt_ego: Dict[str, (T,4)] 또는 (4,)
    - diff_token_to_np_history_wrt_ego: Dict[str, (H,11)] 또는 (11,)
    - diff_token_to_interp_np_traj_wrt_ego: Dict[str, (N,11)] 또는 (11,)
    - diff_token_to_next_wp_wrt_ego: Dict[str, (11,)] 또는 (1,11)
    """
    eps = options.invalid_eps
    xs_local: List[float] = []
    ys_local: List[float] = []

    if not output_data:
        return xs_local, ys_local

    # (1) diff_token_to_np_gen_traj_wrt_ego: Dict[str, (T,4)] 또는 (4,)
    token_to_traj = output_data.get("diff_token_to_np_gen_traj_wrt_ego")
    if isinstance(token_to_traj, dict) and len(token_to_traj) > 0:
        for _, arr in token_to_traj.items():
            if arr is None:
                continue
            arr = np.asarray(arr)
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
        self.diff_token_to_np_gen_traj_wrt_ego: Dict[str,
                                                     np.ndarray] = {}  # (T, 4)
        """
        딥러닝 output 값에서, 슬립 초과한거 제거한거
        """
        self.diff_token_to_np_slip_traj_wrt_ego: Dict[str,
                                                      np.ndarray] = {}  # (T, 4)
        """
        딥러닝 output 값에서, 슬립 초과 제거 + 경로 제약 보정한거
        """
        self.diff_token_to_np_smooth_traj_wrt_ego: Dict[str, np.ndarray] = {
        }  # (T, 4)
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
            "diff_token_to_np_gen_traj_wrt_ego":
                self.diff_token_to_np_gen_traj_wrt_ego,
            "diff_token_to_np_slip_traj_wrt_ego":
                self.diff_token_to_np_slip_traj_wrt_ego,
            "diff_token_to_np_smooth_traj_wrt_ego":
                self.diff_token_to_np_smooth_traj_wrt_ego,
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
        options,
        draw_token_list: Optional[List[str]] = None) -> None:
    """
    # Dict[str, np.ndarray] # len : valid_agent_num

    neighbor_future_gt_3_dim (agent_num, future_len, 3=[x,y,yaw])를
    흰색 'x' 마커로 그리고, 각 에이전트의 첫 점 근처에 인덱스(0..agent_num-1)를 흰색으로 표기.

    규칙:
      - invalid: |x|<=eps and |y|<=eps → 스킵
      - 마커: 흰색 'x', 선 없음
      - 라벨: 첫 점이 유효할 때만 표시
    """

    eps = options.invalid_eps
    for track_token, future_gt_3_dim in diff_token_to_future_gt_3_dim.items():
        if draw_token_list is not None and track_token not in draw_token_list:
            continue
        future_len = future_gt_3_dim.shape[0]
        # 모든 유효 포인트를 x마커로 그리기
        for t in range(future_len):
            row = future_gt_3_dim[t]
            if not is_valid_future_row_xyyaw(row, eps):
                continue
            x, y = float(row[0]), float(row[1])
            ax.plot(x,
                    y,
                    marker='x',
                    markersize=options.DIFF_future_gt_3_dim_marker_size,
                    linestyle='None',
                    color=options.DIFF_future_gt_3_dim_COLOR,
                    zorder=26)
        # 첫 점 라벨(유효할 때만)
        if options.DIFF_draw_diff_future_gt_3_dim_token:
            first = future_gt_3_dim[0]
            if is_valid_future_row_xyyaw(first, eps):
                fx, fy = float(first[0]), float(first[1])
                ax.text(fx + options.DIFF_future_gt_3_dim_text_offset_m,
                        fy + options.DIFF_future_gt_3_dim_text_offset_m,
                        str(track_token)[:5],
                        color=options.DIFF_future_gt_3_dim_token_color,
                        fontsize=options.DIFF_future_gt_3_dim_token_fontsize,
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
    EGO_draw_ego_past : bool
        ego_agent_past(21, 11) 시퀀스 렌더링 여부.
    draw_neighbor_past : bool
        neighbor_agents_past(agent_num, 21, 11) 시퀀스 렌더링 여부.
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
    fig_size: Tuple[float, float] = (18.0, 18.0)
    dpi: int = 400
    margin_m: float = 5.0
    equal_aspect: bool = True
    invalid_eps: float = 0.0

    COMMON_vel_arrow_len_m: float = 5.0
    COMMON_heading_line_scale: float = 0.1

    ######## LANES ########
    LANE_draw_lane_boundaries: bool = True  # check
    LANE_boundary_width: float = 0.2
    LANE_draw_lane_centerline: bool = True  # check
    LANE_draw_agent_route_lane_order: bool = False  # check
    LANE_route_agent_index_color: str = CYAN  # 번호 텍스트 색 # 청록색
    LANE_lane_boundary_color = PURPLE  # 남색(인디고 계열)
    LANE_signal_colors = {
        0: GREEN,  # 녹색(신호등 초록)
        1: ORANGE,  # 노란색(신호등 노랑)
        2: RED,  # 진한 빨간색(신호등 빨강)
        3: GRAY,  # 회색(청회색)
    }
    LANE_AGENT_index_fontsize: int = 2  # 에이전트 번호 텍스트 폰트 크기

    ######### [EGO] ##############
    ########### [EGO] PAST ##################
    EGO_draw_ego_past: bool = True  # check
    EGO_past_style = {
        "fill_color": WHITE,  # 흰색
        "line_color": GRAY,  # 회색
        "line_width": 0.4,
        "fill_alpha_current": 0.8,
    }
    EGO_draw_ego_past_vel: bool = False  # check

    ##############################
    ########### [EGO] FUTURE PLANNER NEXT STATE ##################
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
    EGO_draw_planner_future_11_dim: bool = False  # check
    EGO_planner_future_11_style = {
        "line_color": PALE_CYAN,
        "line_width": 0.2,
        "velocity_line_color": PALE_CYAN,
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    EGO_draw_planner_velocity: bool = False  # check
    ########## [EGO] FUTURE EGO GT 11 ##########
    EGO_draw_ego_future_gt_11_dim: bool = False
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
    NEI_draw_velocity_arrow: bool = False  # check
    NEI_draw_velocity_text: bool = False  # check
    NEI_vel_text_y_offset: float = 0.5
    NEI_vel_text_color = CYAN
    NEI_vel_text_fontsize = 2

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
    NEI_past_token_fontsize: int = 5
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
    DIFF_future_gt_3_dim_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기
    DIFF_future_gt_3_dim_COLOR: str = BRIGHT_CYAN  # 미래 포인트 'x' 마커 크기
    DIFF_draw_diff_future_gt_3_dim_token: bool = False
    DIFF_future_gt_3_dim_text_offset_m: float = 0.  # 번호 텍스트를 포인트 옆으로 얼마나 띄울지(미터)
    DIFF_future_gt_3_dim_token_color: str = BRIGHT_CYAN
    DIFF_future_gt_3_dim_token_fontsize: int = 4  # 에이전트 번호 텍스트 폰트 크기
    ############################################
    DIFF_draw_diff_future_all_gt_3_dim: bool = True
    DIFF_future_all_gt_3_dim_marker_size: float = 0.4  # 미래 포인트 'x' 마커 크기
    DIFF_future_all_gt_3_dim_COLOR: str = BRIGHT_CYAN  # 미래 포인트 'x' 마커 크기
    DIFF_draw_diff_future_all_gt_3_dim_token: bool = False
    DIFF_future_all_gt_3_dim_text_offset_m: float = 0.
    DIFF_future_all_gt_3_dim_token_color: str = BRIGHT_CYAN
    DIFF_future_all_gt_3_dim_token_fontsize: int = 4  # 에이전트 번호 텍스트 폰트 크기
    ######## [NEIGHBOR] FUTURE OUTPUT ##########
    DIFF_draw_diff_future_gen_traj: bool = True
    DIFF_draw_diff_future_gen_traj_token: bool = False
    DIFF_future_gen_traj_mode: str = "arrow"  # 'arrow' 또는 'point'
    DIFF_future_gen_traj_point_marker: str = "o"
    DIFF_future_gen_traj_point_marker_size: float = 0.8
    DIFF_future_gen_traj_arrow_len_m: float = 1.0
    DIFF_future_gen_traj_style = {
        "line_color": WHITE,  # 흰색
        "line_width": 0.3,
        "index_color": WHITE,  # 흰색
    }
    DIFF_draw_diff_future_slip_traj: bool = True
    DIFF_future_slip_traj_style = {
        "line_color": LIGHT_PINK,  # 흰색
        "line_width": 0.2,
        "index_color": LIGHT_PINK,  # 흰색
    }
    DIFF_draw_diff_future_smooth_traj: bool = True
    DIFF_future_smooth_traj_style = {
        "line_color": PALE_RED,  # 흰색
        "line_width": 0.1,
        "index_color": PALE_RED,  # 흰색
    }
    DIFF_future_gen_trak_token_text_y_offset_m: float = 0.5

    ########################
    DIFF_draw_diff_future_gen_refined_traj: bool = False
    DIFF_future_gen_refined_style = {
        "line_color": PALE_CYAN,  # 빨간색(밝은 빨강)
        "line_width": 0.2,
        "velocity_line_color": PALE_CYAN,  # 빨간색(밝은 빨강)
        "velocity_line_alpha": 0.8,
        "velocity_line_width": 0.4,
    }
    DIFF_draw_future_gen_refined_velocity: bool = False
    DIFF_future_gen_refined_velocity_offset_m: float = 0.3
    DIFF_future_gen_refined_velocity_font_size = 2
    DIFF_future_gen_refined_token_offset_m: float = 0.3  # y축으로 살짝 아래(미터 단위)
    DIFF_new_waypoint_vel_text_y_offset_m = 1.
    DIFF_new_waypoint_vel_token_x_offset_m = 1.

    DIFF_new_waypoint_style = {
        "line_color": ORANGE,  # 주황색
        "line_width": 0.2,
    }
    ###################


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
            mutation_scale=3.0,
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


def draw_lane_boundaries(ax: plt.Axes, lanes: Array,
                         options: DrawingOptions) -> None:
    """좌/우 차선 경계를 실선으로 그림(양 끝점 모두 valid일 때만 선분을 그림).

    (lane_num, lane_len, 12)
    """
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
                    color=options.LANE_lane_boundary_color,
                    linewidth=options.LANE_boundary_width,
                    zorder=1)
            ax.plot([r0[0], r1[0]], [r0[1], r1[1]],
                    color=options.LANE_lane_boundary_color,
                    linewidth=options.LANE_boundary_width,
                    zorder=1)


def draw_lane_centerlines(
    ax: plt.Axes,
    lanes: Array,
    options: DrawingOptions,
    agent_route_lane_order: Optional[Array] = None,
    draw_token_int_list: Optional[List[int]] = None,
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
    if options.LANE_draw_agent_route_lane_order == False:
        agent_route_lane_order = None
    eps = options.invalid_eps
    lane_num = lanes.shape[0]

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
            # agent_route_lane_order: (agent_num, lane_num),
            ranks_j: Array = agent_route_lane_order[:, lane_idx]  # (agent_num,)
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
            color = options.LANE_signal_colors.get(int(state_idx[j]), GRAY)
            ax.plot([c0[0], c1[0]], [c0[1], c1[1]],
                    color=color,
                    linewidth=1.2,
                    linestyle=(0, (4, 4)),
                    zorder=2)


def draw_neighbor_past(ax: plt.Axes,
                       neighbor_agents_past: Array,
                       options: DrawingOptions,
                       draw_token_int_list: Optional[List[int]] = None) -> None:
    """이웃 에이전트 과거 시퀀스를 클래스별 색상으로 그림. invalid 스텝은 스킵."""
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    eps = options.invalid_eps
    agent_num, time_len, feat_dim = neighbor_agents_past.shape
    if feat_dim != 11:
        raise ValueError("neighbor_agents_past의 마지막 차원은 11이어야 합니다.")
    current_t = time_len - 1

    for agent_idx in range(agent_num):
        if (draw_token_int_list is not None) and (agent_idx
                                                  not in draw_token_int_list):
            draw_all_time = False
        else:
            draw_all_time = True
        track = neighbor_agents_past[agent_idx]  # (T, 11)
        for t in range(time_len):
            if not draw_all_time and t != current_t:
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

            corners = oriented_box_corners(x, y, c, s, L, W)
            add_polygon(ax,
                        corners,
                        edge_color=neighbor_cls_style["line_color"],
                        line_width=neighbor_cls_style["line_width"],
                        fill_color=fill_color,
                        fill_alpha=fill_alpha,
                        zorder=5 if t == current_t else 4)
            add_heading_line(ax,
                             x,
                             y,
                             c,
                             s,
                             nominal_length=L *
                             options.COMMON_heading_line_scale,
                             color=neighbor_cls_style["line_color"],
                             line_width=neighbor_cls_style["line_width"],
                             zorder=6 if t == current_t else 4)
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
                    zorder=7 if t == current_t else 4)
            if options.NEI_draw_velocity_text:
                if t % 2 == 0:
                    offset = 2
                else:
                    offset = 1
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
    for agent_idx in range(agent_num):
        track_token = neighbor_track_token[agent_idx]
        row = neighbor_agents_past[agent_idx, current_t]  # (11,)
        if not is_valid_agent_row(row, eps):
            assert track_token is None
            continue
        x, y = float(row[0]), float(row[1])
        ax.text(x + options.NEI_past_token_place_offset_m,
                y + options.NEI_past_token_place_offset_m,
                str(track_token)[:5],
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
                               zorder=22 if t == current_t else 9)


def draw_ego_agent_next_11_dim(ax: plt.Axes, ego_agent_next_11_dim: Array,
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
                    edge_color=options.EGO_next_11_dim_style["line_color"],
                    line_width=options.EGO_next_11_dim_style["line_width"],
                    fill_color=None,
                    fill_alpha=None,
                    zorder=25)
        add_heading_line(ax,
                         x,
                         y,
                         c,
                         s,
                         nominal_length=L * options.COMMON_heading_line_scale,
                         color=options.EGO_next_11_dim_style["line_color"],
                         line_width=options.EGO_next_11_dim_style["line_width"],
                         zorder=26)
        if options.EGO_draw_ego_agent_next_11_vel:
            add_velocity_arrow(
                ax,
                x,
                y,
                vx,
                vy,
                length_m=options.COMMON_vel_arrow_len_m,
                line_color=options.EGO_next_11_dim_style["velocity_line_color"],
                line_width=options.EGO_next_11_dim_style["velocity_line_width"],
                line_alpha=options.EGO_next_11_dim_style["velocity_line_alpha"],
                zorder=27)


def draw_planner_future_11_dim(ax: plt.Axes, planner_future_11_dim: Array,
                               options: DrawingOptions) -> None:
    """이고 차량 **GT 미래** 시퀀스를 그림(미래 위치는 채우지 않음). invalid 스텝은 스킵."""
    if planner_future_11_dim is None or planner_future_11_dim.size == 0:
        return
    eps = options.invalid_eps
    future_len, feat_dim = planner_future_11_dim.shape
    if feat_dim != 11:
        raise ValueError("ego_future_gt_11_dim의 마지막 차원은 11이어야 합니다.")

    for t in range(future_len):
        row = planner_future_11_dim[t]
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
            edge_color=options.EGO_planner_future_11_style["line_color"],
            line_width=options.EGO_planner_future_11_style["line_width"],
            fill_color=None,
            fill_alpha=None,
            zorder=24)
        add_heading_line(
            ax,
            x,
            y,
            c,
            s,
            nominal_length=L * options.COMMON_heading_line_scale,
            color=options.EGO_planner_future_11_style["line_color"],
            line_width=options.EGO_planner_future_11_style["line_width"],
            zorder=24)
        if options.EGO_draw_planner_velocity:
            add_velocity_arrow(
                ax,
                x,
                y,
                vx,
                vy,
                length_m=options.COMMON_vel_arrow_len_m,
                line_color=options.
                EGO_planner_future_11_style["velocity_line_color"],
                line_width=options.
                EGO_planner_future_11_style["velocity_line_width"],
                line_alpha=options.
                EGO_planner_future_11_style["velocity_line_alpha"],
                zorder=24)


def draw_ego_future_gt_11_dim(ax: plt.Axes, ego_future_gt_11_dim: Array,
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
                    edge_color=options.EGO_future_gt_11_style["line_color"],
                    line_width=options.EGO_future_gt_11_style["line_width"],
                    fill_color=None,
                    fill_alpha=None,
                    zorder=24)
        add_heading_line(
            ax,
            x,
            y,
            c,
            s,
            nominal_length=L * options.COMMON_heading_line_scale,
            color=options.EGO_future_gt_11_style["line_color"],
            line_width=options.EGO_future_gt_11_style["line_width"],
            zorder=24)
        if options.EGO_draw_future_11_velocity:
            add_velocity_arrow(ax,
                               x,
                               y,
                               vx,
                               vy,
                               length_m=options.COMMON_vel_arrow_len_m,
                               line_color=options.
                               EGO_future_gt_11_style["velocity_line_color"],
                               line_width=options.
                               EGO_future_gt_11_style["velocity_line_width"],
                               line_alpha=options.
                               EGO_future_gt_11_style["velocity_line_alpha"],
                               zorder=24)


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
                        # add_velocity_arrow(
                        #     ax,
                        #     x,
                        #     y,
                        #     vx,
                        #     vy,
                        #     length_m=options.COMMON_vel_arrow_len_m,
                        #     line_color=options.DIFF_future_gen_refined_style["velocity_line_color"],
                        #     line_width=options.DIFF_future_gen_refined_style["velocity_line_width"],
                        #     line_alpha=options.DIFF_future_gen_refined_style["velocity_line_alpha"],
                        #     zorder=24,
                        # )
                        speed_kmh = float(np.hypot(vx, vy)) * 3.6
                        if t % 2 == 0:
                            offset = 5
                        else:
                            offset = 3
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


def draw_diff_future_gen_traj(
        ax: plt.Axes,
        diff_token_to_np_gen_traj_wrt_ego: Optional[TokenTrajDict],
        diff_token_to_np_slip_traj_wrt_ego: Optional[TokenTrajDict],
        diff_token_to_np_smooth_traj_wrt_ego: Optional[TokenTrajDict],
        options: DrawingOptions,
        draw_token_list: Optional[List[str]] = None) -> None:
    """토큰 기준 미래 포즈를 '화살표(방향 포함)' 또는 '점(방향 미사용)'으로 그림.

    Args:
        ax: Matplotlib 축.
        diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] | None
            각 value: shape (future_len, 4) = [x, y, cos(yaw), sin(yaw)]
            - invalid 규칙: 4값 모두 0(±eps) → 스킵
        options: DrawingOptions
            - DIFF_future_gen_traj_mode: 'arrow' | 'point'
            - DIFF_future_gen_traj_point_marker, DIFF_future_gen_traj_point_marker_size
            - DIFF_future_gen_traj_arrow_len_m
        draw_mode: Optional['arrow' | 'point']
            - 우선순위: draw_mode 인자(있으면) > options.DIFF_future_gen_traj_mode(없으면 'arrow')

    동작:
        - 'arrow' 모드:
            (x,y)에서 (cos,sin) 방향으로 고정 길이(options.DIFF_future_gen_traj_arrow_len_m) 화살표
        - 'point' 모드:
            (x,y) 위치에 포인트만 표시(방향 미사용)

    Note:
        - t==0 (각 토큰의 첫 포인트)에는 토큰 문자열을 살짝 아래(y-오프셋)에 표시.
    """
    if not diff_token_to_np_gen_traj_wrt_ego:
        return

    mode = options.DIFF_future_gen_traj_mode
    if mode not in {"arrow", "point"}:
        raise ValueError(
            f"Unsupported draw_mode: {mode}. Use 'arrow' or 'point'.")

    eps = options.invalid_eps
    # [ADD] 삽입순서 그대로 인덱스 부여를 위해 enumerate(dict.items()) 사용
    for idx, (token, np_gen_traj_wrt_ego) in enumerate(
            diff_token_to_np_gen_traj_wrt_ego.items()):  # [ADD]
        if draw_token_list is not None and token not in draw_token_list:
            continue
        if np_gen_traj_wrt_ego is None or np_gen_traj_wrt_ego.size == 0:
            continue
        if np_gen_traj_wrt_ego.ndim != 2 or np_gen_traj_wrt_ego.shape[1] != 4:
            raise ValueError(
                "token_to_future_traj_wrt_ego의 각 value는 (future_len, 4)이어야 합니다."
            )

        future_len = np_gen_traj_wrt_ego.shape[0]
        for t in range(future_len):
            row = np_gen_traj_wrt_ego[t]  # (4,) = [x, y, cos, sin]
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
                    length_m=options.DIFF_future_gen_traj_arrow_len_m,
                    line_color=options.DIFF_future_gen_traj_style["line_color"],
                    line_width=options.DIFF_future_gen_traj_style["line_width"],
                    zorder=23,
                )
            else:
                # 점만 표시(방향 정보 사용하지 않음)
                ax.plot(
                    x,
                    y,
                    marker=options.DIFF_future_gen_traj_point_marker,
                    markersize=options.DIFF_future_gen_traj_point_marker_size,
                    linestyle="None",
                    color=options.DIFF_future_gen_traj_style["line_color"],
                    zorder=23,
                )

            #  시작 포인트(t==0)에 토큰 식별 라벨(흰색)을 화살표/점 바로 아래에 표기
            if t == 0 and options.DIFF_draw_diff_future_gen_traj_token:
                ax.text(
                    x,
                    y - options.DIFF_future_gen_trak_token_text_y_offset_m,
                    str(token),
                    color=options.DIFF_future_gen_traj_style["index_color"],
                    fontsize=options.LANE_AGENT_index_fontsize,
                    ha="center",
                    va="top",
                    zorder=24,
                )

    if options.DIFF_draw_diff_future_slip_traj and diff_token_to_np_slip_traj_wrt_ego:
        for idx, (token, np_slip_traj_wrt_ego) in enumerate(
                diff_token_to_np_slip_traj_wrt_ego.items()):  # [ADD]
            if draw_token_list is not None and token not in draw_token_list:
                continue
            if np_slip_traj_wrt_ego is None or np_slip_traj_wrt_ego.size == 0:
                continue
            if np_slip_traj_wrt_ego.ndim != 2 or np_slip_traj_wrt_ego.shape[
                    1] != 4:
                raise ValueError(
                    "token_to_future_traj_wrt_ego의 각 value는 (future_len, 4)이어야 합니다."
                )

            future_len = np_slip_traj_wrt_ego.shape[0]
            for t in range(future_len):
                row = np_slip_traj_wrt_ego[t]  # (4,) = [x, y, cos, sin]
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
                        length_m=options.DIFF_future_gen_traj_arrow_len_m,
                        line_color=options.
                        DIFF_future_slip_traj_style["line_color"],
                        line_width=options.
                        DIFF_future_slip_traj_style["line_width"],
                        zorder=23,
                    )
                else:
                    # 점만 표시(방향 정보 사용하지 않음)
                    ax.plot(
                        x,
                        y,
                        marker=options.DIFF_future_gen_traj_point_marker,
                        markersize=options.
                        DIFF_future_gen_traj_point_marker_size,
                        linestyle="None",
                        color=options.DIFF_future_slip_traj_style["line_color"],
                        zorder=23,
                    )

    if options.DIFF_draw_diff_future_smooth_traj and diff_token_to_np_smooth_traj_wrt_ego:
        for idx, (token, np_smooth_traj_wrt_ego) in enumerate(
                diff_token_to_np_slip_traj_wrt_ego.items()):  # [ADD]
            if draw_token_list is not None and token not in draw_token_list:
                continue
            if np_smooth_traj_wrt_ego is None or np_smooth_traj_wrt_ego.size == 0:
                continue
            if np_smooth_traj_wrt_ego.ndim != 2 or np_smooth_traj_wrt_ego.shape[
                    1] != 4:
                raise ValueError(
                    "token_to_future_traj_wrt_ego의 각 value는 (future_len, 4)이어야 합니다."
                )

            future_len = np_smooth_traj_wrt_ego.shape[0]
            for t in range(future_len):
                row = np_smooth_traj_wrt_ego[t]  # (4,) = [x, y, cos, sin]
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
                        length_m=options.DIFF_future_gen_traj_arrow_len_m,
                        line_color=options.
                        DIFF_future_smooth_traj_style["line_color"],
                        line_width=options.
                        DIFF_future_smooth_traj_style["line_width"],
                        zorder=23,
                    )
                else:
                    # 점만 표시(방향 정보 사용하지 않음)
                    ax.plot(
                        x,
                        y,
                        marker=options.DIFF_future_gen_traj_point_marker,
                        markersize=options.
                        DIFF_future_gen_traj_point_marker_size,
                        linestyle="None",
                        color=options.
                        DIFF_future_smooth_traj_style["line_color"],
                        zorder=23,
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
    plt.savefig(save_path, dpi=fig.get_dpi(),
                facecolor=fig.get_facecolor())  # bbox_inches="tight",
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
        draw_neighbor_past(ax, neighbor_agents_past, draw_option,
                           draw_token_int_list)
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


def draw_diff_future_all_gt_3_dim(
        ax: plt.Axes,
        diff_token_to_future_all_gt_3_dim: Dict[str, Array],
        options: DrawingOptions,
        draw_token_list: Optional[List[str]] = None) -> None:
    """near_future_all_gt_3_dim (Pnn, future_all_len, 3=[x,y,yaw])를
    흰색 'x' 마커로 그리고, 각 에이전트의 첫 점 근처에 인덱스(0..Pnn-1)를 흰색으로 표기.

    규칙:
      - invalid: |x|<=eps and |y|<=eps → 스킵
      - 마커: 흰색 'x', 선 없음
      - 라벨: 첫 점이 유효할 때만 표시
    """

    eps = options.invalid_eps
    for track_token, future_all_gt_3_dim in diff_token_to_future_all_gt_3_dim.items(
    ):
        if draw_token_list is not None:
            if track_token not in draw_token_list:
                continue
        future_all_len = future_all_gt_3_dim.shape[0]  # (future_all_len, 3)
        # 모든 유효 포인트를 x마커로 그리기
        for t in range(future_all_len):
            row = future_all_gt_3_dim[t]
            if not is_valid_future_row_xyyaw(row, eps):
                continue
            x, y = float(row[0]), float(row[1])
            ax.plot(x,
                    y,
                    marker='x',
                    markersize=options.DIFF_future_all_gt_3_dim_marker_size,
                    linestyle='None',
                    color=options.DIFF_future_all_gt_3_dim_COLOR,
                    zorder=26)
        # 첫 점 라벨(유효할 때만)
        if options.DIFF_draw_diff_future_all_gt_3_dim_token:
            first = future_all_gt_3_dim[0]
            if is_valid_future_row_xyyaw(first, eps):
                fx, fy = float(first[0]), float(first[1])
                ax.text(
                    fx + options.DIFF_future_all_gt_3_dim_text_offset_m,
                    fy + options.DIFF_future_all_gt_3_dim_text_offset_m,
                    str(track_token)[:5],
                    color=options.DIFF_future_all_gt_3_dim_token_color,
                    fontsize=options.DIFF_future_all_gt_3_dim_token_fontsize,
                    ha='left',
                    va='bottom',
                    zorder=30)


def draw_neighbor_future_all(ax: plt.Axes,
                             input_data: WorldModelFeature,
                             output_data: Optional[Dict[str, Any]],
                             draw_option: DrawingOptions,
                             draw_token_list: Optional[List[str]] = None):
    ### [NEIGHBOR FUTURE GT] ###
    diff_token_to_future_gt_3_dim = input_data.get(
        "diff_token_to_future_gt_3_dim", None)
    if draw_option.DIFF_draw_diff_future_gt_3_dim and (
            diff_token_to_future_gt_3_dim is not None):
        draw_neighbor_future_gt_3_dim(ax, diff_token_to_future_gt_3_dim,
                                      draw_option, draw_token_list)
    ### [NEIGHBOR FUTURE OUTPUT] ###
    diff_token_to_np_gen_traj_wrt_ego = output_data.get(
        "diff_token_to_np_gen_traj_wrt_ego", None)
    diff_token_to_np_slip_traj_wrt_ego = output_data.get(
        "diff_token_to_np_slip_traj_wrt_ego", None)
    diff_token_to_np_smooth_traj_wrt_ego = output_data.get(
        "diff_token_to_np_smooth_traj_wrt_ego", None)
    if draw_option.DIFF_draw_diff_future_gen_traj:
        draw_diff_future_gen_traj(ax, diff_token_to_np_gen_traj_wrt_ego,
                                  diff_token_to_np_slip_traj_wrt_ego,
                                  diff_token_to_np_smooth_traj_wrt_ego,
                                  draw_option, draw_token_list)

    diff_token_to_interp_np_traj_wrt_ego = output_data.get(
        "diff_token_to_interp_np_traj_wrt_ego", None)
    diff_token_to_next_wp_wrt_ego = output_data.get(
        "diff_token_to_next_wp_wrt_ego", None)
    if draw_option.DIFF_draw_diff_future_gen_refined_traj:
        draw_diff_future_gen_refined_traj(ax,
                                          diff_token_to_interp_np_traj_wrt_ego,
                                          draw_option,
                                          diff_token_to_next_wp_wrt_ego,
                                          draw_token_list)
    diff_token_to_future_all_gt_3_dim = input_data.get(
        "diff_token_to_future_all_gt_3_dim", None)  # (future_all_len, 3)
    if draw_option.DIFF_draw_diff_future_all_gt_3_dim and (
            diff_token_to_future_all_gt_3_dim is not None):
        draw_diff_future_all_gt_3_dim(ax, diff_token_to_future_all_gt_3_dim,
                                      draw_option, draw_token_list)
    ########################################


def draw_neighbor(ax: plt.Axes,
                  input_data: WorldModelFeature,
                  output_data: Optional[Dict[str, Any]],
                  draw_option: DrawingOptions,
                  draw_token_list: Optional[List[str]] = None):
    draw_neighbor_past_all(ax, input_data, output_data, draw_option,
                           draw_token_list)
    draw_neighbor_future_all(ax, input_data, output_data, draw_option,
                             draw_token_list)


def draw_lane(ax: plt.Axes,
              input_data: WorldModelFeature,
              draw_option: DrawingOptions,
              draw_token_list: Optional[List[str]] = None):
    lanes = input_data.get("lanes")

    render_fast_collections.apply_rasterization(ax, rasterization_zorder=10)
    if draw_option.LANE_draw_lane_boundaries:
        render_fast_collections.draw_lane_boundaries_fast(
            ax,
            lanes.astype(np.float32),
            boundary_color=draw_option.LANE_lane_boundary_color,
            linewidth=draw_option.LANE_boundary_width,
            invalid_eps=draw_option.invalid_eps,
        )
        # draw_lane_boundaries(ax, lanes, draw_option)
    # agent_route_lane_order가 있으면 텍스트 표기 모드로 전환
    if draw_option.LANE_draw_lane_centerline:
        """ 디버깅 용으로 작성해놓음
        - token_candidates 에 route를 확인하고 싶은 agent의 token을 넣어주면 됨
        - draw_token_list 를 None으로 설정하면 -> 모든 차량에 대해서 text를 그리게 됨
        """
        draw_token_int_list: Optional[List[int]] = get_agent_idx_from_tokens(
            draw_token_list, input_data.get("neighbor_track_token", None))
        render_fast_collections.draw_lane_centerlines_fast(
            ax,
            lanes.astype(np.float32),
            signal_colors=draw_option.LANE_signal_colors,
            draw_agent_route_lane_order=bool(
                draw_option.LANE_draw_agent_route_lane_order),
            agent_route_lane_order=input_data.get("agent_route_lane_order",
                                                  None),
            draw_token_int_list=draw_token_int_list,
            label_stride=4,  # 원본의 point_idx % 4 규칙 유지
            route_label_color=draw_option.LANE_route_agent_index_color,
            route_label_fontsize=draw_option.LANE_AGENT_index_fontsize,
            route_label_vstep_m=0.3,
            dashed_linewidth=1.2,
            dashed_pattern=(0, (4, 4)),
            invalid_eps=draw_option.invalid_eps,
        )


def lock_axes_bounds_before_drawing(
    ax: plt.Axes,
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]],
    options: "DrawingOptions",
) -> Tuple[float, float, float, float]:
    """그리기 전에 축 범위를 계산·고정하고, autoscale을 꺼서 이후 드로잉 동안 축이 변하지 않게 한다.

    동작:
        1) compute_auto_bounds(...)로 (xmin, xmax, ymin, ymax) 계산
        2) set_axes_limits_with_margin(...)으로 여백 포함해 축 고정
        3) ax.set_autoscale_on(False)로 autoscale 비활성화
        4) apply_axes_style(...)로 축 스타일(비율/축표시) 적용
           - 당신 코드에서는 adjustable='box'라 limits를 바꾸지 않음

    Returns:
        (xmin, xmax, ymin, ymax): 계산된 원시 범위(여백 전). 디버깅/로그용.
    """
    # ① 우선 범위 계산(숫자 배열에서만 계산하므로 빠름)
    xmin, xmax, ymin, ymax = compute_auto_bounds(input_data, output_data,
                                                 options)

    # ② 여백 포함해서 축 고정
    set_axes_limits_with_margin(ax, (xmin, xmax, ymin, ymax), options.margin_m)

    # ③ autoscale 비활성화(그 이후 add_patch/plot 등 호출 시 축 갱신 안 함)
    ax.set_autoscale_on(False)  # == ax.autoscale(False)

    # ④ 축 스타일 적용(비율/equal, 축 숨김 등)
    apply_axes_style(ax, options)

    return xmin, xmax, ymin, ymax


def set_axes_limits_with_small_auto_margin(
    ax,
    bounds: tuple[float, float, float, float],
    frac: float = 0.02,  # 스팬의 2%
    min_m: float = 0.5,
    max_m: float = 3.0,
) -> None:
    xmin, xmax, ymin, ymax = bounds
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)
    span = max(xspan, yspan)
    m = max(min(span * frac, max_m), min_m)
    ax.set_xlim(xmin - m, xmax + m)
    ax.set_ylim(ymin - m, ymax + m)
    ax.set_autoscale_on(False)  # 🔒 이후 추가되는 아티스트가 축을 바꾸지 못하게


def resize_figure_to_data_aspect(
        fig,
        bounds: tuple[float, float, float, float],
        target_long_side_px: int = 1200,  # 긴 변 픽셀 목표(용량 통제 핵심)
) -> None:
    xmin, xmax, ymin, ymax = bounds
    xspan = max(xmax - xmin, 1e-6)
    yspan = max(ymax - ymin, 1e-6)
    aspect = xspan / yspan

    if aspect >= 1.0:
        w_px = target_long_side_px
        h_px = max(1, int(round(w_px / aspect)))
    else:
        h_px = target_long_side_px
        w_px = max(1, int(round(h_px * aspect)))

    dpi = fig.get_dpi()
    fig.set_size_inches(w_px / dpi, h_px / dpi)


def make_axes_fill_figure(fig, ax) -> None:
    # Figure 바깥 여백 제거 + Axes를 Figure 전체로 확장
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])


# [Add]
def draw_world_model_to_png(
    input_data: Dict[str, Any],
    output_data: Optional[Dict[str, Any]],
    save_path: str,
    options: Optional[DrawingOptions] = None,
) -> None:
    # draw_token_list: List[str] = ["63070d02949f5bd8"] # ["1be4dfd6d2f852a9", "f476b2c85dd7508c", "88dbeb62be085df7"]
    draw_token_list = None
    draw_option = options or DrawingOptions()

    # 1) Figure/Axes
    fig, ax = create_figure_and_axes(draw_option)
    # 2) 🔑 그리기 전에: 범위 계산 → 축 고정 → 비율 지정 → Figure/Axes 배치
    bounds = compute_auto_bounds(input_data, output_data, draw_option)
    set_axes_limits_with_small_auto_margin(ax,
                                           bounds,
                                           frac=0.02,
                                           min_m=0.5,
                                           max_m=3.0)
    ax.set_aspect('equal', adjustable='box')  # 데이터 비율 유지(축 한계는 그대로)
    ax.set_autoscale_on(False)  # 이후 추가되는 아티스트가 축을 건드리지 못함
    resize_figure_to_data_aspect(fig, bounds, target_long_side_px=1200)
    make_axes_fill_figure(fig, ax)
    apply_axes_style(ax, draw_option)  # 축 숨김 등(축 범위엔 영향 없음)
    #########################################
    draw_lane(ax, input_data, draw_option, draw_token_list)
    draw_ego(ax, input_data, draw_option)
    draw_neighbor(ax, input_data, output_data, draw_option, draw_token_list)
    #########################################

    # # ── (5) 축 범위/스타일 ───────────────────────────────────────────
    # # [Add]
    # bounds = compute_auto_bounds(input_data, output_data, draw_option)
    # set_axes_limits_with_margin(ax, bounds, draw_option.margin_m)
    # apply_axes_style(ax, draw_option)

    # 6) 저장
    save_figure_to_png(fig, save_path)
