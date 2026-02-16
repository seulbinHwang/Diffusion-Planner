import numpy as np
from tqdm import tqdm
import matplotlib
from matplotlib.collections import PolyCollection
import time
from typing import Dict, Tuple, Union, List, Optional, Any

matplotlib.use('Agg')  # GUI 백엔드 사용 안함 (메모리 절약)
import matplotlib.pyplot as plt
import contextlib

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
import copy
from typing import Deque
from nuplan.planning.simulation.observation.observation_type import Observation
import os
import torch
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from typing import Dict, Tuple, Union, List, Optional
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.ego_state import EgoState
import draw_machine_fast
# matplotlib 설정 추가
plt.rcParams['figure.max_open_warning'] = 0  # 경고 메시지 비활성화
matplotlib.rcParams['figure.max_open_warning'] = 0
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines, LaneSegmentTrafficLightData)
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    build_ego_past_feature,
    build_neighbor_past_feature,
    build_static_feature,
    sampled_tracked_objects_to_array_list,
    sampled_ego_objects_to_array_list,
    sampled_static_objects_to_array_list,
    agent_future_all_process,
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_data_dict_to_device_tensors, get_npc_route_roadblock_ids, get_neighbor_track_tokens
# [ADDED] 통계 저장용
import json
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # 타입 판정용
from diffusion_planner.data_process.road_safety_process import extract_stop_sign_points, extract_crosswalk_points
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]
from typing import Any, Tuple, Optional
import numpy as np
import torch


_FEASIBLE_SG_PROJECTOR: Optional[Any] = None


def _get_feasible_sg_projector() -> Any:
    """FeasibleProjector의 SG(yaw_rate) 계산만 재사용하기 위한 싱글턴을 반환합니다.

    Returns:
        Any:
            FeasibleProjector 인스턴스(내부적으로 torch 기반 SG 계산 기능을 사용).
    """
    global _FEASIBLE_SG_PROJECTOR
    if _FEASIBLE_SG_PROJECTOR is not None:
        return _FEASIBLE_SG_PROJECTOR

    # 무거운 import는 "필요할 때만" 하도록 lazy import
    from diffusion_planner.model.module.feasible import FeasibleProjector  # pylint: disable=import-error

    class _DummyCfg:
        """FeasibleProjector 초기화에 필요한 최소 설정."""
        use_batch_integration: bool = False

    # SG 유틸만 쓸 거라서 use_feasible_dl=False로 두면 네트워크 모듈을 만들지 않습니다.
    _FEASIBLE_SG_PROJECTOR = FeasibleProjector(
        config=_DummyCfg(),
        hidden_dim=1,
        use_feasible_dl=False,
        use_feasible_filter=False,
    )
    _FEASIBLE_SG_PROJECTOR.eval()
    return _FEASIBLE_SG_PROJECTOR

def compute_past_future_yaw_rate_from_cs_yaw_via_feasible_sg(
    past_future_cs_yaw: np.ndarray,  # (T,2) 또는 (A,T,2)
    past_future_valid: np.ndarray,   # (T,) 또는 (A,T) bool
    *,
    dt: float = 0.1,
    polyorder: int = 2,
    max_window_len_yaw: int = 7,
) -> np.ndarray:
    """(cos(yaw), sin(yaw)) 시퀀스를 그대로 사용해 yaw_rate를 계산합니다.

    목적:
      - work()에서 이미 뽑아둔 past+future의 (cos,sin) 값을 그대로 써서
        FeasibleProjector 내부 SG 미분 로직으로 yaw_rate를 구합니다.
      - (cos,sin)을 다시 만들거나, (x,y,cos,sin) 전체를 다시 구성하는 중복을 줄입니다.

    Args:
        past_future_cs_yaw (np.ndarray):
            - ego: (T, 2)
            - neighbor: (A, T, 2)
            - 마지막 2는 [cos(yaw), sin(yaw)]
        past_future_valid (np.ndarray):
            - ego: (T,) bool
            - neighbor: (A, T) bool
            - True면 유효 노드, False면 무효 노드
        dt (float): 샘플 간 시간 간격. shape: ()
        polyorder (int): SG 다항 차수. shape: ()
        max_window_len_yaw (int): yaw 미분 창 길이. shape: ()

    Returns:
        np.ndarray:
            - ego: (T,)
            - neighbor: (A, T)
            - 무효 노드는 0.0
    """
    if past_future_cs_yaw.ndim == 2:
        cs = past_future_cs_yaw[None, ...]  # (1, T, 2)
        valid = past_future_valid[None, ...]  # (1, T)
        squeeze_agent = True
    elif past_future_cs_yaw.ndim == 3:
        cs = past_future_cs_yaw  # (A, T, 2)
        valid = past_future_valid  # (A, T)
        squeeze_agent = False
    else:
        raise ValueError(
            f"past_future_cs_yaw must be 2D or 3D. got {past_future_cs_yaw.shape}"
        )

    if cs.shape[-1] != 2:
        raise ValueError(f"last dim must be 2(cos,sin). got {cs.shape}")
    if valid.shape != cs.shape[:2]:
        raise ValueError(
            f"past_future_valid shape mismatch: valid={valid.shape}, cs={cs.shape}"
        )

    A = int(cs.shape[0])
    T = int(cs.shape[1])
    if A == 0:
        out = np.zeros((0, T), dtype=np.float32)
        return out[0] if squeeze_agent else out

    cs_f32 = np.ascontiguousarray(cs.astype(np.float32, copy=False))         # (A,T,2)
    valid_b = np.ascontiguousarray(valid.astype(bool, copy=False))           # (A,T)

    # torch: (B=1, Pnn=A, T)
    cs_t = torch.from_numpy(cs_f32).unsqueeze(0)                             # (1,A,T,2)
    valid_t = torch.from_numpy(valid_b).unsqueeze(0)                         # (1,A,T) bool

    cos_y = cs_t[..., 0]                                                     # (1,A,T)
    sin_y = cs_t[..., 1]                                                     # (1,A,T)

    projector = _get_feasible_sg_projector()

    with torch.no_grad():
        yaw_rate_t = projector._compute_yaw_rate_via_sg(
            cos_y=cos_y,
            sin_y=sin_y,
            points_valid=valid_t,
            dt=float(dt),
            polyorder=int(polyorder),
            max_window_len_yaw=int(max_window_len_yaw),
        )  # (1,A,T)

    yaw_rate = yaw_rate_t.squeeze(0).cpu().numpy().astype(np.float32, copy=False)  # (A,T)
    return yaw_rate[0] if squeeze_agent else yaw_rate




def _to_scalar_dt(value: Union[float, np.ndarray], ref: NDArray[np.generic]) -> np.floating:
    """dt를 ref와 같은 dtype의 '스칼라'로 정리합니다."""
    dt_arr = np.asarray(value, dtype=ref.dtype)
    if dt_arr.size != 1:
        raise ValueError(f"dt는 스칼라여야 합니다. got shape={dt_arr.shape}, size={dt_arr.size}")
    return dt_arr.reshape(()).item()


def _normalize_cos_sin(
    cos_seq: ArrayF,
    sin_seq: ArrayF,
    eps: float,
) -> Tuple[ArrayF, ArrayF]:
    """(cos, sin) 쌍을 길이 1이 되도록 정규화합니다."""
    r = np.sqrt(cos_seq * cos_seq + sin_seq * sin_seq + eps).astype(cos_seq.dtype, copy=False)
    return (cos_seq / r).astype(cos_seq.dtype, copy=False), (sin_seq / r).astype(sin_seq.dtype, copy=False)


def _wrap_to_pi(delta: ArrayF) -> ArrayF:
    """각도 차이를 (-pi, pi] 범위로 접습니다."""
    return np.arctan2(np.sin(delta), np.cos(delta)).astype(delta.dtype, copy=False)


def differentiate_numpy_pose3_to_control3(
    cur_future_pose_gt_3_dim: ArrayF,  # (P, 1+T, 3) = (x, y, heading)
    dt: Union[float, np.ndarray],
    *,
    eps: float = 1e-8,
    normalize_yaw: bool = True,
    wrap_heading: bool = True,
) -> ArrayF:
    """(x,y,heading) 궤적에서 구간별 제어(vx_b, vy_b, omega)를 차분으로 복원합니다.

    입력:
        cur_future_pose_gt_3_dim: (P -> , 1+T, 3)
            - 마지막 3은 (x, y, heading[rad])
            - 시간축은 k=0..T (총 1+T개 상태)

    출력:
        cur_future_control_gt_3_dim: (P, T, 3)
            - 마지막 3은 (v_x^b, v_y^b, omega)
            - 시간축은 구간 k=0..T-1 (총 T개 구간)
    """
    pose = np.asarray(cur_future_pose_gt_3_dim)
    if pose.ndim != 3 or int(pose.shape[-1]) != 3:
        raise ValueError(
            "cur_future_pose_gt_3_dim은 (P, 1+T, 3) 3D 배열이어야 합니다. "
            f"got shape={pose.shape}"
        )

    # float dtype 강제(삼각함수/나눗셈 안정)
    pose = pose.astype(np.float32 if pose.dtype.kind != "f" else pose.dtype, copy=False)

    P, time_len, _ = pose.shape  # last dim=3
    T = int(time_len - 1)
    if T <= 0:
        raise ValueError(f"time_len(=1+T)은 최소 2여야 합니다. got time_len={time_len}")

    dt_s = _to_scalar_dt(dt, ref=pose)
    if (not np.isfinite(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt_s}")

    # 분해: (P, 1+T)
    x = pose[..., 0]
    y = pose[..., 1]
    heading = pose[..., 2]

    # 구간별 slice: (P, T)
    x0, x1 = x[..., :-1], x[..., 1:]
    y0, y1 = y[..., :-1], y[..., 1:]
    th0, th1 = heading[..., :-1], heading[..., 1:]

    # 1) Δθ, omega
    delta_theta = (th1 - th0).astype(pose.dtype, copy=False)  # (P, T)
    if wrap_heading:
        delta_theta = _wrap_to_pi(delta_theta)  # (P, T)
    omega = (delta_theta / dt_s).astype(pose.dtype, copy=False)  # (P, T)

    # 2) midpoint heading -> (cos, sin)
    th_mid = (th0 + 0.5 * delta_theta).astype(pose.dtype, copy=False)  # (P, T)
    cos_mid = np.cos(th_mid).astype(pose.dtype, copy=False)            # (P, T)
    sin_mid = np.sin(th_mid).astype(pose.dtype, copy=False)            # (P, T)

    if normalize_yaw:
        cos_mid, sin_mid = _normalize_cos_sin(cos_mid, sin_mid, eps=float(eps))

    # 3) world velocity
    vwx = ((x1 - x0) / dt_s).astype(pose.dtype, copy=False)  # (P, T)
    vwy = ((y1 - y0) / dt_s).astype(pose.dtype, copy=False)  # (P, T)

    # 4) world -> body (inverse rotation by midpoint heading)
    vx_b = (cos_mid * vwx + sin_mid * vwy).astype(pose.dtype, copy=False)     # (P, T)
    vy_b = (-sin_mid * vwx + cos_mid * vwy).astype(pose.dtype, copy=False)   # (P, T)

    # 출력: (P, T, 3)
    cur_future_control_gt_3_dim = np.stack([vx_b, vy_b, omega], axis=-1).astype(pose.dtype, copy=False)
    return cur_future_control_gt_3_dim


def integrate_numpy_control3_to_pose3_midpoint(
    cur_future_control_gt_3_dim: ArrayF,  # (P, T, 3) = (v_x^b, v_y^b, omega)
    near_current_pose_3_dim: ArrayF,  # (P, 3) = (x0, y0, heading0)
    dt: Union[float, np.ndarray],
    *,
    eps: float = 1e-8,
    normalize_yaw: bool = True,
    wrap_heading: bool = True,
) -> ArrayF:
    """(v_x^b, v_y^b, omega) 시퀀스를 midpoint 적분하여 (x,y,heading) 궤적으로 복원합니다.

    입력:
        cur_future_control_gt_3_dim: (P, T, 3)
            - 마지막 3은 (v_x^b, v_y^b, omega)
            - 시간축은 구간 k=0..T-1 (총 T개 구간)
        near_current_pose_3_dim: (P, 3)
            - (x0, y0, heading0) 현재 노드 상태

    출력:
        cur_future_pose_gt_3_dim: (P, 1+T, 3)
            - 마지막 3은 (x, y, heading[rad])
            - 시간축은 노드 k=0..T (총 1+T개 상태)
    """
    control = np.asarray(cur_future_control_gt_3_dim)
    pose0 = np.asarray(near_current_pose_3_dim)
    # float dtype 강제(삼각함수/나눗셈 안정)
    control = control.astype(np.float32 if control.dtype.kind != "f" else control.dtype, copy=False)
    pose0 = pose0.astype(control.dtype, copy=False)

    P, T, _ = control.shape
    if T <= 0:
        raise ValueError(f"T는 최소 1이어야 합니다. got T={T}")
    if pose0.shape[0] != P:
        raise ValueError(
            f"P 차원이 일치해야 합니다. control P={P}, pose0 P={pose0.shape[0]}"
        )

    dt_s = _to_scalar_dt(dt, ref=control)
    if (not np.isfinite(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt_s}")

    vx_b = control[..., 0]
    vy_b = control[..., 1]
    omega = control[..., 2]

    x0 = pose0[..., 0]
    y0 = pose0[..., 1]
    yaw0 = pose0[..., 2]

    # 각속도 적분: Δθ_k = w_k * dt
    dtheta_seq = (omega * dt_s).astype(control.dtype, copy=False)  # (P,T)
    dtheta_prefix = np.cumsum(dtheta_seq, axis=1)  # (P,T)
    zero_pad = np.zeros_like(dtheta_seq[..., :1])  # (P,1)
    dtheta_exclusive = np.concatenate([zero_pad, dtheta_prefix[..., :-1]], axis=1)  # (P,T)
    yaw_start = (yaw0[..., None] + dtheta_exclusive).astype(control.dtype, copy=False)  # (P,T)

    # 중점/종단 각도
    yaw_mid = (yaw_start + 0.5 * dtheta_seq).astype(control.dtype, copy=False)  # (P,T)
    yaw_next = (yaw_start + dtheta_seq).astype(control.dtype, copy=False)  # (P,T)
    if wrap_heading:
        yaw_next = _wrap_to_pi(yaw_next)

    cos_mid = np.cos(yaw_mid).astype(control.dtype, copy=False)
    sin_mid = np.sin(yaw_mid).astype(control.dtype, copy=False)
    if normalize_yaw:
        cos_mid, sin_mid = _normalize_cos_sin(cos_mid, sin_mid, eps=float(eps))

    # 세계 기준 중점 속도
    vwx_mid = (cos_mid * vx_b - sin_mid * vy_b).astype(control.dtype, copy=False)
    vwy_mid = (sin_mid * vx_b + cos_mid * vy_b).astype(control.dtype, copy=False)

    dx_seq = (vwx_mid * dt_s).astype(control.dtype, copy=False)
    dy_seq = (vwy_mid * dt_s).astype(control.dtype, copy=False)

    x_next = (x0[..., None] + np.cumsum(dx_seq, axis=1)).astype(control.dtype, copy=False)
    y_next = (y0[..., None] + np.cumsum(dy_seq, axis=1)).astype(control.dtype, copy=False)

    # 출력 조립: (P, 1+T, 3)
    cur_future_pose_gt_3_dim = np.zeros((P, T + 1, 3), dtype=control.dtype)
    cur_future_pose_gt_3_dim[:, 0, 0] = x0
    cur_future_pose_gt_3_dim[:, 0, 1] = y0
    cur_future_pose_gt_3_dim[:, 0, 2] = yaw0
    cur_future_pose_gt_3_dim[:, 1:, 0] = x_next
    cur_future_pose_gt_3_dim[:, 1:, 1] = y_next
    cur_future_pose_gt_3_dim[:, 1:, 2] = yaw_next
    return cur_future_pose_gt_3_dim




class DataProcessor(object):

    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)
        self.config = config
        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon
        self.set_coord_as_center = config.set_coord_as_center
        self.caching_max_agent_num = config.caching_max_agent_num
        self.max_agent_num = config.max_agent_num
        self._use_filter_radius, self._filter_radius = self._read_filter_radius_settings_from_config(
            config)
        self.caching_max_static_num = config.caching_max_static_num
        self.max_static_num = config.max_static_num
        # [변경] 타입별 상한 신설: 보행자/자전거
        self.max_pedestrians = None  #getattr(config, "max_pedestrians", 7)  #128)
        self.max_bicycles = None  #getattr(config, "max_bicycles", 3)  #64)
        self.all_car_token_to_rr_ids: Optional[Dict[str,
                                                    Optional[List[str]]]] = None
        self.init_cur_fut_agents_world_8_list: Optional[List[np.ndarray]] = None
        self._map_elements = [
            'LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'
        ]  # name of map features to be extracted.
        self._caching_max_map_elements = {
            'LANE': config.caching_max_lane_num,
            'LEFT_BOUNDARY': config.caching_max_lane_num,
            'RIGHT_BOUNDARY': config.caching_max_lane_num,
            'ROUTE_LANES': config.caching_max_lane_num
        }  # maximum number of elements to extract per feature layer.
        self._max_map_elements = {
            'LANE': config.max_lane_num,
            'LEFT_BOUNDARY': config.max_lane_num,
            'RIGHT_BOUNDARY': config.max_lane_num,
            'ROUTE_LANES': config.max_lane_num
        }  # maximum number of elements to extract per feature layer.
        self._map_points_num = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.lane_len
        }  # maximum number of points per feature to extract per feature layer.
        # =========================
        # 저장 성능 튜닝 옵션
        # =========================
        # 기본값 0: fsync를 아예 하지 않음(가장 빠름)
        # 필요하면 config에 아래 값을 추가해서 "N개마다 1번"만 강제 반영 가능
        self._save_fsync_every_n: int = int(
            getattr(config, "save_fsync_every_n", 0) or 0)
        self._save_dir_fsync_every_n: int = int(
            getattr(config, "save_dir_fsync_every_n", 0) or 0)

        # 압축 유지(기존 동작 그대로). 원하면 False로 바꿔 더 빠르게 할 수 있음(파일은 커짐)
        self._save_use_compression: bool = bool(
            getattr(config, "save_use_compression", True))

        # 프로세스(워커) 내부에서 저장 횟수 카운트
        self._save_counter: int = 0
        # 디버그 플롯 파일명 증가용 카운터
        self._debug_plot_counter: int = 0

    @staticmethod
    def _build_origin_world_pose(
            ego_cur_pose_np: np.
        ndarray,  # shape: (3,) = [x_world, y_world, yaw_world]
    ) -> np.ndarray:
        """현재 샘플의 “ego 기준 좌표계 원점”이 세계좌표계에서 어디인지 (x, y, cos, sin)으로 만든다.

        우리가 저장하는 대부분의 값은
        “현재 ego 위치를 (0,0) 원점으로 둔 좌표계(ego 기준 좌표계)”에서 표현됩니다.

        그런데 나중에 이 값을 다시 세계좌표계로 복원하려면,
        “그 원점이 세계좌표계에서는 어디였는지”가 반드시 필요합니다.

        이 함수는 그 정보를 아래 형태로 만들어 줍니다.

        - origin_world_pose = [x_world, y_world, cos(yaw_world), sin(yaw_world)]
          shape: (4,)

        여기서 (x_world, y_world, yaw_world)는 입력 `ego_cur_pose_np`에서 가져옵니다.
        `ego_cur_pose_np`는 이미 코드에서 ego 기준 좌표계 변환의 기준점(원점)으로 쓰는 값이므로,
        이 값을 그대로 저장하면 “데이터를 만들 때 사용한 기준점”과 완전히 일치합니다.

        Args:
            ego_cur_pose_np (np.ndarray):
                shape: (3,)
                - [x_world, y_world, yaw_world]
                - 세계좌표계에서의 현재 ego 위치/방향(라디안)

        Returns:
            np.ndarray:
                shape: (4,)
                - [x_world, y_world, cos(yaw_world), sin(yaw_world)]
                - dtype: float32
        """
        if not isinstance(ego_cur_pose_np, np.ndarray):
            raise TypeError(
                f"`ego_cur_pose_np`는 np.ndarray 여야 합니다. got {type(ego_cur_pose_np)}"
            )
        if ego_cur_pose_np.shape != (3,):
            raise ValueError(
                f"`ego_cur_pose_np` shape는 (3,) 이어야 합니다. got {ego_cur_pose_np.shape}"
            )

        x_world: float = float(ego_cur_pose_np[0])
        y_world: float = float(ego_cur_pose_np[1])
        yaw_world: float = float(ego_cur_pose_np[2])

        # origin_world_pose: shape (4,) = [x, y, cos(yaw), sin(yaw)]
        origin_world_pose: np.ndarray = np.array(
            [x_world, y_world,
             np.cos(yaw_world),
             np.sin(yaw_world)],
            dtype=np.float32,
        )
        return origin_world_pose

    @staticmethod
    def _slice_neighbor_cur_fut_horizon_11dim(
        neighbor_cur_fut_all_gt_11_dim: np.ndarray,  # shape: (N, T_all, 11)
        iteration: int,
        future_len: int,
    ) -> np.ndarray:
        """neighbor의 (현재~미래) 전체 시퀀스에서, 특정 iteration 기준으로 (현재+미래) 구간만 고정 길이로 뽑는다.

        이 함수가 필요한 이유
        ---------------------
        observation_adapter()에서는 scenario 전체 길이만큼의 (현재~미래) 데이터를 미리 만들어두고,
        매 step마다 iteration 위치에서 앞으로 future_len 만큼을 잘라서 씁니다.

        그런데 “유효점과 유효점 사이에 무효점이 있으면 안 된다” 규칙을 적용할 때,
        **현재 step에서 필요한 101개(=past_len + future_len)**만 보고 처리해야 합니다.
        (멀리 뒤의 미래 프레임이 섞이면, 그 정보 때문에 앞 구간이 잘못 채워질 수 있습니다.)

        그래서 이 함수는:
        - neighbor_cur_fut_all_gt_11_dim (N, T_all, 11) 에서
        - [iteration ... iteration + future_len] (총 1+future_len 프레임)
          을 뽑아서 (N, 1+future_len, 11) 로 반환합니다.
        - 범위를 벗어나는 프레임은 0으로 패딩합니다.

        Args:
            neighbor_cur_fut_all_gt_11_dim (np.ndarray):
                shape: (N, T_all, 11)
                - N: agent 수
                - T_all: scenario 전체 타임 길이(현재 포함)
                - 11: [x, y, cos, sin, vx, vy, width, length, onehot(3)]
            iteration (int):
                현재 step 인덱스(0 기반).
                이 값이 “현재 프레임” 위치라고 가정합니다.
            future_len (int):
                “현재 이후”로 몇 프레임을 쓸지 (현재 제외).
                예: 80 이면 출력 길이는 81(=현재1 + 미래80)

        Returns:
            np.ndarray:
                shape: (N, 1 + future_len, 11)
                - index 0: 현재 프레임
                - index 1..future_len: 미래 프레임
                - 부족한 구간은 0으로 채워짐
        """
        if neighbor_cur_fut_all_gt_11_dim.ndim != 3 or neighbor_cur_fut_all_gt_11_dim.shape[
                -1] != 11:
            raise ValueError(
                f"`neighbor_cur_fut_all_gt_11_dim` shape는 (N, T_all, 11)이어야 합니다. "
                f"got {neighbor_cur_fut_all_gt_11_dim.shape}")
        if iteration < 0:
            raise ValueError(f"`iteration`은 0 이상이어야 합니다. got {iteration}")
        if future_len < 0:
            raise ValueError(f"`future_len`은 0 이상이어야 합니다. got {future_len}")

        N: int = int(neighbor_cur_fut_all_gt_11_dim.shape[0])
        T_all: int = int(neighbor_cur_fut_all_gt_11_dim.shape[1])
        out_len: int = int(1 + future_len)

        # out: (N, 1+future_len, 11)
        out: np.ndarray = np.zeros((N, out_len, 11),
                                   dtype=neighbor_cur_fut_all_gt_11_dim.dtype)

        start: int = int(iteration)
        end: int = int(iteration + out_len)

        if start >= T_all:
            return out

        copy_end: int = int(min(T_all, end))
        copy_len: int = int(copy_end - start)  # 복사 가능한 길이

        out[:, :copy_len, :] = neighbor_cur_fut_all_gt_11_dim[:,
                                                              start:copy_end, :]
        return out

    def _build_neighbor_future_gt_from_past_and_cur_fut_11dim(
        self,
        neighbor_agents_past: np.ndarray,  # shape: (N, Tp, 11)
        neighbor_cur_fut_gt_11_dim: np.
        ndarray,  # shape: (N, 1+Tf, 11)  (0번이 현재)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """neighbor의 past와 (현재+미래)를 합쳐서, 규칙을 만족하도록 정리한 뒤 최종 출력들을 만든다.

        반드시 만족해야 하는 규칙(한 agent 기준)
        --------------------------------------
        1) 현재 점(past의 마지막)이 무효면:
           - 101개 전체(과거~현재~미래)가 전부 무효(=전부 0)여야 합니다.

        2) 현재 점이 유효면:
           - 101개 전체에서 “유효점과 유효점 사이에 무효점”이 있으면 안 됩니다.
           - 즉, 유효한 프레임들이 중간에 끊기면,
             그 끊긴 구간을 x/y/cos/sin/vx/vy의 “직선 중간값”으로 채워서
             유효 구간이 한 덩어리로 이어지게 만듭니다.
           - 유효 구간 밖(prefix/suffix)은 0으로 둡니다.

        Args:
            neighbor_agents_past (np.ndarray):
                shape: (N, Tp, 11)
                - Tp = time_len (예: 21)
                - 마지막 index (Tp-1)이 “현재 프레임”
            neighbor_cur_fut_gt_11_dim (np.ndarray):
                shape: (N, 1+Tf, 11)
                - index 0이 “현재 프레임”
                - index 1..Tf 가 “미래”
                - Tf = future_len (예: 80)

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - fixed_neighbor_agents_past: shape (N, Tp, 11)
                - neighbor_future_gt_11_dim:  shape (N, Tf, 11)  (현재 제외)
                - neighbor_future_gt_3_dim:   shape (N, Tf, 3)   [x, y, yaw]
        """
        fixed_neighbor_agents_past, neighbor_future_gt_11_dim = self._merge_and_interpolate_neighbor_11dim(
            neighbor_agents_past=neighbor_agents_past,
            neighbor_cur_fut_gt_11_dim=neighbor_cur_fut_gt_11_dim,
        )

        # 11 -> 3 (yaw는 cos/sin으로 계산)
        neighbor_future_gt_3_dim: np.ndarray = self._traj11_to_traj3_yaw(
            neighbor_future_gt_11_dim)

        return fixed_neighbor_agents_past, neighbor_future_gt_11_dim, neighbor_future_gt_3_dim

    def _enforce_no_invalid_between_valid_in_ego_past(
        self,
        ego_agent_past: np.ndarray,  # shape: (Tp, 11)
        *,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """ego_agent_past(과거~현재)에서 '없는 과거 프레임(0 패딩)'이 끼어도 규칙이 깨지지 않게 정리한다.

        핵심 요구사항(이번 수정)
        -----------------------
        - ego_agent_past는 과거→현재 순서다.
        - past가 부족한 경우 "없는 과거"는 앞쪽(prefix)에만 0으로 채워질 수 있다.
        - 따라서 유효 구간은 [first_valid ... current]로 한 덩어리여야 하며,
          그 밖(prefix)은 전부 0이어야 한다.

        유효/무효 판정 기준(중요)
        -----------------------
        - '없는 프레임'은 size/type 값이 섞여 들어올 수도 있으므로,
          유효 판정은 **앞 6개 값(x,y,cos,sin,vx,vy)**만 본다.
          (즉, width/length/one-hot 때문에 무효가 유효로 오인되는 걸 막는다)

        규칙
        ----
        - 현재 프레임(=past의 마지막)이 유효일 때:
          · past 안에서 유효점과 유효점 사이에 무효점(0)이 끼면 안 된다.
          · 유효 프레임들의 first~last 구간을 연속 유효 구간으로 만들고,
            그 안의 빈 프레임은 x/y/cos/sin/vx/vy를 선형 보간으로 채운다.
          · 유효 구간 밖(prefix/suffix)은 0으로 둔다.
        - 현재 프레임이 무효면:
          · 안전하게 past 전체를 0으로 만든다.

        Args:
            ego_agent_past (np.ndarray):
                shape: (Tp, 11)
            eps (float):
                0과 아주 가까운 값을 “0”처럼 볼 때 쓰는 기준

        Returns:
            np.ndarray:
                shape: (Tp, 11)
                규칙을 만족하도록 정리된 ego_agent_past (새 배열)
        """
        if ego_agent_past.ndim != 2 or ego_agent_past.shape[-1] != 11:
            raise ValueError(
                f"`ego_agent_past` shape는 (Tp, 11)이어야 합니다. got {ego_agent_past.shape}"
            )

        Tp: int = int(ego_agent_past.shape[0])
        if Tp == 0:
            return ego_agent_past

        traj: np.ndarray = ego_agent_past.astype(np.float32,
                                                 copy=True)  # (Tp, 11)
        current_index: int = Tp - 1

        # ✅ 유효 판정은 '앞 6개(x,y,cos,sin,vx,vy)'만 사용
        # valid_mask_1d: (Tp,)
        valid_mask_1d: np.ndarray = (np.abs(traj[:, :6]) > eps).any(axis=1)

        # 현재가 무효면 past 전체를 0으로
        if not bool(valid_mask_1d[current_index]):
            return np.zeros_like(traj)

        valid_idx: np.ndarray = np.nonzero(valid_mask_1d)[0]
        if valid_idx.size == 0:
            return np.zeros_like(traj)

        region_mask: np.ndarray = np.zeros((Tp,), dtype=bool)

        if valid_idx.size == 1:
            region_mask[int(valid_idx[0])] = True
        else:
            first_valid: int = int(valid_idx[0])
            last_valid: int = int(valid_idx[-1])
            region_mask[first_valid:last_valid + 1] = True

            # 중간 구멍이 있으면 x/y/cos/sin/vx/vy를 채움 (0~5)
            if last_valid - first_valid + 1 > valid_idx.size:
                xs: np.ndarray = valid_idx.astype(np.float64)  # (K,)
                seg_idx: np.ndarray = np.arange(first_valid,
                                                last_valid + 1,
                                                dtype=np.float64)

                for dim_idx in range(6):  # 0~5
                    ys: np.ndarray = traj[valid_idx, dim_idx].astype(np.float64,
                                                                     copy=False)
                    interp_vals: np.ndarray = np.interp(seg_idx, xs, ys)
                    traj[first_valid:last_valid + 1,
                         dim_idx] = interp_vals.astype(np.float32, copy=False)

        # 타입/크기는 “현재 프레임 값”을 대표로 씀
        # 수정 (해결)
        type_vec: np.ndarray = traj[current_index, 8:11].astype(np.float32,
                                                                copy=True)
        rep_size: np.ndarray = traj[current_index, 6:8].astype(np.float32,
                                                               copy=True)

        # one-hot은 유효 구간에만
        traj[:, 8:11] = 0.0
        traj[region_mask, 8:11] = type_vec

        # 유효 구간 밖은 완전 0
        traj[~region_mask, :] = 0.0

        # width/length 채우기 + cos/sin 정리
        traj_b: np.ndarray = traj[None, :, :]  # (1, Tp, 11)
        region_b: np.ndarray = region_mask[None, :]  # (1, Tp)
        rep_size_b: np.ndarray = rep_size[None, :]  # (1, 2)

        traj_b = self._fill_width_length_with_representative_size(
            traj_11=traj_b,
            valid_mask=region_b,
            rep_size=rep_size_b,
        )
        traj_b = self._normalize_cos_sin_in_traj_11(
            traj_11=traj_b,
            valid_mask=region_b,
        )
        return traj_b[0]  # (Tp, 11)

    def _get_map_query_radius_m(self) -> float:
        """지도/도로시설(정지표지, 횡단보도 등)을 조회할 때 사용할 반경(m)을 반환합니다.

        이 반경은 '필터링을 켜고/끄는 옵션(use_filter_radius)'과 성격이 다릅니다.

        - use_filter_radius 는 '에이전트/정적 객체를 거리로 잘라낼지'를 제어합니다.
        - 하지만 지도/도로시설은 반경이 없으면 가져오는 데이터가 너무 커져서
          속도/메모리 문제가 생길 수 있습니다.

        그래서 지도/도로시설 조회 반경은 use_filter_radius 와 무관하게
        항상 filter_radius 값을 그대로 사용하도록 분리합니다.

        Returns:
            float:
                meter 단위 지도/도로시설 조회 반경.
        """
        return float(self._filter_radius)

    @staticmethod
    def _read_filter_radius_settings_from_config(
            config: object) -> Tuple[bool, float]:
        """config에서 필터링 설정(use_filter_radius, filter_radius)을 읽습니다.

        이 클래스는 오직 config에 아래 두 값이 "명시적으로 존재"할 때만 동작하도록 강제합니다.
          - use_filter_radius (bool)
          - filter_radius (float, meter)

        둘 중 하나라도 없으면, 잘못된 설정으로 조용히 다른 기본값을 쓰는 일을 막기 위해
        즉시 에러를 내고 중단합니다.

        Args:
            config: args_util.get_args()가 반환한 인자 객체(보통 argparse.Namespace)

        Returns:
            Tuple[bool, float]:
                - use_filter_radius: bool
                - filter_radius_m: float (meter)

        Raises:
            RuntimeError: use_filter_radius 또는 filter_radius가 config에 없을 때
        """
        missing: List[str] = []
        if not hasattr(config, "use_filter_radius"):
            missing.append("use_filter_radius")
        if not hasattr(config, "filter_radius"):
            missing.append("filter_radius")

        if len(missing) > 0:
            raise RuntimeError(
                "필수 설정이 config에 없습니다. "
                "args_util.py에 `--use_filter_radius`와 `--filter_radius`를 반드시 선언해야 합니다. "
                f"missing={missing}")

        use_filter_radius = bool(getattr(config, "use_filter_radius"))
        filter_radius_m = float(getattr(config, "filter_radius"))
        return use_filter_radius, filter_radius_m

    def _get_effective_filter_radius_m(self) -> Optional[float]:
        """현재 설정(use_filter_radius)에 따라 실제로 사용할 반경을 반환합니다.

        - use_filter_radius=True: filter_radius(m)를 반환합니다.
        - use_filter_radius=False: None을 반환해서, 호출 측에서 필터링을 건너뛰게 합니다.

        Returns:
            Optional[float]:
                - float: meter 단위 반경
                - None: 필터링을 하지 않음
        """
        if bool(self._use_filter_radius):
            return float(self._filter_radius)
        return None

    @staticmethod
    def _adjust_ego_future_outputs_to_center_frame(
        ego_state: EgoState,
        ego_future_gt_3_dim: np.ndarray,  # shape: (T, 3)
        ego_future_gt_11_dim: np.ndarray,  # shape: (T, 11)
        *,
        set_coord_as_center: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ego 미래 궤적 출력이 rear axle 기준일 때, center 기준으로 x,y만 보정한다.

        주의(이번 수정의 핵심)
        ---------------------
        - 무효 프레임(앞 8차원이 전부 0인 프레임)은 (0,0,0,0,...) 상태를 유지해야 한다.
        - 따라서 set_coord_as_center=True 여도 무효 프레임에는 x,y 이동(offset)을 적용하지 않는다.
          (유효 프레임에만 적용)

        Args:
            ego_state (EgoState):
                현재 ego 상태.
            ego_future_gt_3_dim (np.ndarray):
                shape: (T, 3)
                [x, y, yaw] 형태의 ego 미래 궤적(ego 로컬 좌표계).
            ego_future_gt_11_dim (np.ndarray):
                shape: (T, 11)
                [x, y, cos, sin, vx, vy, width, length, onehot(3)] 형태의 ego 미래 궤적.
            set_coord_as_center (bool):
                True면 center 기준으로 보정, False면 입력 그대로 반환.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                (ego_future_gt_3_dim, ego_future_gt_11_dim)
                - 둘 다 입력 배열을 **in-place로** 수정한 뒤 그대로 반환합니다.
        """
        if not set_coord_as_center:
            return ego_future_gt_3_dim, ego_future_gt_11_dim

        # 미래 길이가 0이면 할 게 없음
        if ego_future_gt_11_dim.size == 0:
            return ego_future_gt_3_dim, ego_future_gt_11_dim

        # (1) 월드 좌표에서 rear_axle -> center 변위
        dx_world: float = float(ego_state.center.x - ego_state.rear_axle.x)
        dy_world: float = float(ego_state.center.y - ego_state.rear_axle.y)

        # (2) 현재 heading 기준 로컬 좌표로 회전 (world -> ego heading frame)
        heading: float = float(ego_state.rear_axle.heading)
        c: float = float(np.cos(heading))
        s: float = float(np.sin(heading))

        # local = R^T * world,  R = [[c, -s],[s, c]]
        offset_x_local: float = dx_world * c + dy_world * s
        offset_y_local: float = -dx_world * s + dy_world * c

        # ✅ (핵심) 유효/무효 마스크
        # - 무효 프레임 정의: [x, y, cos, sin, vx, vy, width, length] 8개가 전부 0이면 무효
        eps: float = 1e-8
        # valid_mask: shape (T,)
        valid_mask: np.ndarray = (np.abs(ego_future_gt_11_dim[:, :6])
                                  > eps).any(axis=1)

        # 유효 프레임에만 원점 이동 적용
        ego_future_gt_3_dim[valid_mask, 0] -= offset_x_local
        ego_future_gt_3_dim[valid_mask, 1] -= offset_y_local
        ego_future_gt_11_dim[valid_mask, 0] -= offset_x_local
        ego_future_gt_11_dim[valid_mask, 1] -= offset_y_local

        return ego_future_gt_3_dim, ego_future_gt_11_dim

    @staticmethod
    def _normalize_cos_sin_in_traj_11(
        traj_11: np.ndarray,  # shape: (N, T, 11)
        valid_mask: np.ndarray,  # shape: (N, T), True면 유효 프레임
        *,
        eps: float = 1e-6,
        cos_index: int = 2,
        sin_index: int = 3,
    ) -> np.ndarray:
        """(cos, sin) 채널을 "진짜 cos/sin"처럼 보이도록 길이를 1로 맞춥니다.

        왜 필요한가
        ----------
        중간 프레임을 채우는 과정에서 cos/sin을 숫자 그대로 선형으로 섞으면,
        (cos, sin)이 단위원 위(길이 1)에 있지 않을 수 있습니다.
        그러면 모델 입장에서 "방향 정보"가 애매해질 수 있습니다.

        이 함수는 다음 규칙으로 정리합니다.
        1) valid_mask=True 인 프레임만 처리합니다.
        2) (cos^2 + sin^2)의 제곱근이 eps보다 크면,
           cos와 sin을 그 길이로 나눠서 길이를 1로 맞춥니다.
        3) 길이가 너무 작으면(=방향 정보가 사실상 없는 경우),
           해당 프레임의 cos/sin을 0으로 둡니다.
        4) valid_mask=False 인 프레임은 cos/sin을 0으로 둡니다.

        Args:
            traj_11 (np.ndarray):
                shape (N, T, 11)
            valid_mask (np.ndarray):
                shape (N, T)
            eps (float):
                0 나누기 방지용 작은 값
            cos_index (int):
                cos 채널 인덱스(기본 2)
            sin_index (int):
                sin 채널 인덱스(기본 3)

        Returns:
            np.ndarray:
                shape (N, T, 11)
                traj_11을 직접 수정한 뒤 그대로 반환합니다.
        """
        if traj_11.ndim != 3 or traj_11.shape[-1] != 11:
            raise ValueError(
                f"`traj_11` shape는 (N, T, 11)이어야 합니다. got {traj_11.shape}")
        if valid_mask.shape != traj_11.shape[:2]:
            raise ValueError(
                f"`valid_mask` shape는 (N, T)이어야 합니다. got {valid_mask.shape}, expected {traj_11.shape[:2]}"
            )

        cos_v = traj_11[:, :, cos_index]  # (N, T)
        sin_v = traj_11[:, :, sin_index]  # (N, T)

        # norm: (N, T)
        norm = np.sqrt(cos_v * cos_v + sin_v * sin_v)

        # 나눗셈 안전장치
        norm_safe = np.where(norm > eps, norm, 1.0)

        cos_unit = cos_v / norm_safe
        sin_unit = sin_v / norm_safe

        # valid_mask=True 이고 norm>eps 인 곳만 (cos_unit, sin_unit) 사용
        good = valid_mask & (norm > eps)

        traj_11[:, :, cos_index] = np.where(good, cos_unit,
                                            0.0).astype(traj_11.dtype,
                                                        copy=False)
        traj_11[:, :, sin_index] = np.where(good, sin_unit,
                                            0.0).astype(traj_11.dtype,
                                                        copy=False)
        return traj_11

    # [ADDED] 통계 유틸 함수들
    # =========================
    @staticmethod
    def _count_valid_neighbors_by_type(
            neighbor_agents_past: np.ndarray,  # (N, Tp, 11)
    ) -> Tuple[int, int, int]:
        """마지막 시점의 에이전트 상태로 유효/타입을 판정해 수를 셉니다.

        규칙:
          - 유효성: 마지막 시점의 앞 8차원(kinematics/size)이 모두 0이면 무효로 간주
            · 즉, valid = any(|state_last[:8]| > eps)
          - 타입: 마지막 3차원(one-hot) = [vehicle, pedestrian, bicycle]
            · 임계값 0.5 초과를 1로 해석(부동소수 오차 대비)

        Args:
            neighbor_agents_past (np.ndarray):
                - shape: (N, Tp, 11)
                - 마지막 차원 11 = [x, y, cos, sin, vx, vy, width, length, onehot_vehicle, onehot_ped, onehot_bike]

        Returns:
            Tuple[int, int, int]: (vehicle_count, pedestrian_count, bicycle_count)

        Raises:
            ValueError: 입력의 마지막 차원 크기가 11이 아닌 경우.
        """
        if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[
                -1] != 11:
            raise ValueError(
                f"`neighbor_agents_past` shape는 (N, Tp, 11)이어야 합니다. "
                f"got {neighbor_agents_past.shape}")

        # 마지막 시점만 사용
        last: np.ndarray = neighbor_agents_past[:, -1, :]  # (N, 11)

        # 유효성 마스크: 앞 8차원 중 하나라도 |.| > eps 이면 유효
        eps = 1e-8
        valid_mask: np.ndarray = (np.abs(last[:, :8]) > eps).any(axis=1)  # (N,)

        # 타입 one-hot (vehicle, pedestrian, bicycle)
        type_oh: np.ndarray = last[:, 8:11]  # (N, 3)
        veh_mask = type_oh[:, 0] > 0.5
        ped_mask = type_oh[:, 1] > 0.5
        bik_mask = type_oh[:, 2] > 0.5

        vehicle_count = int(np.sum(valid_mask & veh_mask))
        pedestrian_count = int(np.sum(valid_mask & ped_mask))
        bicycle_count = int(np.sum(valid_mask & bik_mask))

        return vehicle_count, pedestrian_count, bicycle_count

    @staticmethod
    def _compute_lane_speed_stats(
        vector_map_output: Dict[str,
                                np.ndarray],) -> Tuple[float, Optional[float]]:
        """차선 관련 통계를 계산한다.

        분모는 '유효 차선' 개수:
            - vector_map_output['lanes'] 의 각 차선 텐서 합(|.|) > 0

        통계:
            - 속도제한 차선 비율(%):
                100 * (#(유효 ∧ has_speed_limit True)) / (#유효)
            - 속도제한 차선들의 평균 제한속도(km/h):
                mean(lanes_speed_limit[유효 ∧ True]) * 3.6
                (없으면 None 반환)

        Args:
            vector_map_output: map_process(...) 가 반환한 dict

        Returns:
            Tuple[float, Optional[float]]: (ratio_percent, mean_speed_kmh or None)
        """
        lanes: np.ndarray = vector_map_output[
            'lanes']  # (lane_num, lane_len, 12)
        has_speed: np.ndarray = vector_map_output[
            'lanes_has_speed_limit']  # (lane_num, 1) bool
        speed_mps: np.ndarray = vector_map_output[
            'lanes_speed_limit']  # (lane_num, 1) float

        # 유효 차선 판정:
        #   - 앞 8채널(x, y, vec, left/right 등)이 전부 0이면 패딩으로 간주
        #   - 즉, lanes[..., :8]의 모든 값이 0인 lane 은 무시
        lanes_front8: np.ndarray = lanes[..., :8]  # (lane_num, lane_len, 8)
        lanes_valid_mask = (np.abs(lanes_front8).sum(axis=(1, 2))
                            > 0)  # (lane_num,)
        if lanes_valid_mask.sum() == 0:
            return 0.0, None

        has_speed_mask = (has_speed.reshape(-1).astype(bool)
                         ) & lanes_valid_mask  # (lane_num,)
        ratio_percent = float(100.0 * has_speed_mask.sum() /
                              lanes_valid_mask.sum())

        mean_speed_kmh: Optional[float] = None
        if has_speed_mask.any():
            mean_speed_kmh = float(
                speed_mps.reshape(-1)[has_speed_mask].mean() * 3.6)

        return ratio_percent, mean_speed_kmh

    def _save_sample_stats_json(
        self,
        map_name: str,
        token: str,
        stats: Dict[str, Union[int, float, None]],
    ) -> None:
        """샘플별 통계를 `<save_path>/<map>_<token>.stats.json` 으로 저장한다.

        원자적 저장을 위해 `.tmp`로 쓴 뒤 최종 파일명으로 교체한다.

        Args:
            map_name: 맵 이름
            token: 시나리오 토큰
            stats: 저장할 통계 딕셔너리
        """
        if not self._save_dir:
            return
        os.makedirs(self._save_dir, exist_ok=True)
        json_temp_folder = os.path.join(self._save_dir, "json_temp")
        os.makedirs(json_temp_folder, exist_ok=True)
        out_path = os.path.join(json_temp_folder,
                                f"{map_name}_{token}.stats.json")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp_path, out_path)

    def _get_car_token_to_rr_ids(
        self,
        all_car_token_to_rr_ids: Dict[str, Optional[List[str]]],
        neighbor_track_token: List[str]  # len = chosen_agent_num
    ) -> Dict[str, List[str]]:  # len = chosen_car_num
        car_token_to_rr_ids: Dict[str, Optional[List[str]]] = {}
        for token in neighbor_track_token:
            if token in all_car_token_to_rr_ids:
                car_token_to_rr_ids[token] = all_car_token_to_rr_ids[token]
        return car_token_to_rr_ids

    def _get_past_cur_ego_feature(
        self,
        scenario: Optional[NuPlanScenario] = None,
        history_buffer: Optional[SimulationHistoryBuffer] = None,
        *,
        set_coord_as_center: bool = False,
    ) -> Tuple[EgoState, Point2D, float, np.ndarray, np.ndarray,
               Optional[np.ndarray]]:
        """시나리오 또는 history buffer 에서 ego 궤적을 공통 포맷으로 추출한다.

        (중간 설명은 기존 docstring 유지하되, 아래 한 줄만 추가 개념으로 보면 됩니다)
        - set_coord_as_center=True 이면:
          ego 기준 좌표 변환의 기준점을 rear axle이 아니라 ego center로 잡습니다.
          즉, ego_point2d / ego_heading / ego_cur_pose_np 가 center 기준으로 설정됩니다.
        """
        if (scenario is None and history_buffer is None) or \
           (scenario is not None and history_buffer is not None):
            raise ValueError("scenario 또는 history_buffer 중 정확히 하나만 전달해야 합니다.")

        if scenario is not None:
            ego_state: EgoState = scenario.initial_ego_state
        else:
            ego_state = history_buffer.current_state[
                0]  # type: ignore[union-attr]

        # ✅ 기준점 선택: rear_axle(default) vs center
        if set_coord_as_center:
            ref = ego_state.center
        else:
            ref = ego_state.rear_axle

        ego_point2d = Point2D(ref.x, ref.y)
        ego_heading: float = float(ref.heading)
        ego_cur_pose_np = np.array([ref.x, ref.y, ref.heading],
                                   dtype=np.float64)  # shape: (3,)

        if scenario is not None:
            (past_cur_ego_world_10,
             past_cur_time_np) = get_ego_past_array_from_scenario(
                 scenario,
                 self.num_past_poses,
                 self.past_time_horizon,
             )
        else:
            ego_state_buffer: Deque[EgoState] = history_buffer.ego_state_buffer
            past_cur_ego_world_10 = sampled_ego_objects_to_array_list(
                ego_state_buffer)
            past_cur_time_np = None

        assert past_cur_ego_world_10.shape[0] == self.num_past_poses + 1, \
            f"Expected past_cur_ego_world_10 shape[0] == {self.num_past_poses + 1}, got {past_cur_ego_world_10.shape[0]}"

        return (
            ego_state,
            ego_point2d,
            ego_heading,
            ego_cur_pose_np,
            past_cur_ego_world_10,
            past_cur_time_np,
        )

    def _prepare_car_token_to_rr_ids(
        self,
        scenario: NuPlanScenario,
        use_route_lanes: bool = False,
        neighbor_track_token: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        if use_route_lanes and self.all_car_token_to_rr_ids is None:
            present_tracked_objects: TrackedObjects \
                = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses)
            ]
            past_cur_tracked_objects = past_tracked_objects + [
                present_tracked_objects
            ]
            self.all_car_token_to_rr_ids: Dict[
                str, List[str]] = get_npc_route_roadblock_ids(
                    scenario,
                    past_cur_tracked_objects,
                    neighbor_track_token=None)
        elif not use_route_lanes:
            self.all_car_token_to_rr_ids = {}
        # len = chosen_car_num
        car_token_to_rr_ids: Dict[str,
                                  List[str]] = self._get_car_token_to_rr_ids(
                                      self.all_car_token_to_rr_ids,
                                      neighbor_track_token)
        return car_token_to_rr_ids

    def _get_cur_fut_agents_world_8_list(
        self,
        scenario: NuPlanScenario,
        token_to_id: Dict[str, int],
        do_inference: bool,
    ):
        if do_inference:
            if self.init_cur_fut_agents_world_8_list is None:
                scenario_duration: float = scenario.duration_s.time_s + self.future_time_horizon
                num_samples = int(scenario_duration * 10.0)
                """
                self.init_cur_fut_agents_world_8_list: List[np.ndarray]
                    - 길이: 1 + num_samples
                    - 각 원소 shape: (frame_agents_num_t, 8)
                """
                (self.init_cur_fut_agents_world_8_list,
                 _) = self._get_future_tracked_objects_array_list(
                     scenario,
                     token_to_id=token_to_id,
                     iteration=0,
                     future_time_horizon=scenario_duration,
                     num_samples=num_samples)

            # 깊은 복사 후, 선택 에이전트들만 뽑아서 ego 기준으로 변환
            cur_fut_agents_world_8_list = copy.deepcopy(
                self.init_cur_fut_agents_world_8_list)
        else:
            (cur_fut_agents_world_8_list,
             _) = self._get_future_tracked_objects_array_list(
                 scenario, token_to_id=token_to_id, iteration=0)
        return cur_fut_agents_world_8_list

    # Use for inference
    def observation_adapter(
        self,
        iteration: int,
        history_buffer: SimulationHistoryBuffer,
        traffic_light_data: List[TrafficLightStatusData],
        map_api: NuPlanMap,
        device='cpu',
        scenario: Optional[NuPlanScenario] = None,
        use_route_lanes: bool = False,
        squeeze: bool = False,
    ) -> Dict[str, torch.Tensor]:

        (ego_state, ego_point2d, ego_heading, ego_cur_pose_np,
         past_cur_ego_world_10, _) = self._get_past_cur_ego_feature(
             history_buffer=history_buffer,
             set_coord_as_center=self.set_coord_as_center,
         )

        # ✅ 추가: ego 기준 좌표계 원점의 세계좌표 포즈 저장
        # origin_world_pose: shape (4,) = [x_world, y_world, cos(yaw), sin(yaw)]
        origin_world_pose: np.ndarray = self._build_origin_world_pose(
            ego_cur_pose_np)

        ego_agent_past = build_ego_past_feature(
            past_cur_ego_world_10=past_cur_ego_world_10,
            ego_cur_pose_np=ego_cur_pose_np,
        )
        # ✅ 추가: 과거가 부족해 0으로 채운 프레임(prefix)이 있으면 확실히 0으로 정리
        ego_agent_past = self._enforce_no_invalid_between_valid_in_ego_past(
            ego_agent_past)
        # Past observations including the current
        observation_buffer: Deque[
            Observation] = history_buffer.observation_buffer

        (
            past_cur_agents_world_8_list,
            past_cur_agents_types_list,
            present_static_feat_5,
            static_types_list,
            token_to_id,
            _,
            _,
        ) = self._get_past_cur_agents_feature(
            observation_buffer=observation_buffer)

        (neighbor_agents_past, agents_cur_frame_indices, neighbors_id,
         neighbor_track_token) = build_neighbor_past_feature(
             past_cur_agents_world_8_list=past_cur_agents_world_8_list,
             past_cur_agents_types_list=past_cur_agents_types_list,
             max_agent_num=self.max_agent_num,
             ego_cur_pose_np=ego_cur_pose_np,
             max_pedestrians=self.max_pedestrians,
             max_bicycles=self.max_bicycles,
             token_to_id=token_to_id,
             filter_radius=None,
         )

        ego_time_len = ego_agent_past.shape[0]
        neighbor_time_len = neighbor_agents_past.shape[1]
        assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
            f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"

        cur_fut_agents_world_8_list = self._get_cur_fut_agents_world_8_list(
            scenario, token_to_id, do_inference=True)

        # (N, 1 + Tf_all, 11)
        neighbor_cur_fut_all_gt_11_dim = agent_future_all_process(
            ego_cur_pose_np=ego_cur_pose_np,
            cur_fut_agents_world_8_list=cur_fut_agents_world_8_list,
            neighbor_token_id=neighbors_id,
            neighbor_agents_past=neighbor_agents_past,
        )

        # ✅ (요구조건 a) “현재 iteration 기준”으로 (현재+미래 80)만 뽑아서
        #    (past 21)과 합친 101개 기준으로 규칙을 강제
        neighbor_cur_fut_horizon_gt_11_dim = self._slice_neighbor_cur_fut_horizon_11dim(
            neighbor_cur_fut_all_gt_11_dim=neighbor_cur_fut_all_gt_11_dim,
            iteration=iteration,
            future_len=self.num_future_poses,
        )

        # ✅ 규칙 적용 + 최종 neighbor_future_gt_3_dim / neighbor_future_gt_11_dim 생성
        neighbor_agents_past, neighbor_future_gt_11_dim, neighbor_future_gt_3_dim = \
            self._build_neighbor_future_gt_from_past_and_cur_fut_11dim(
                neighbor_agents_past=neighbor_agents_past,
                neighbor_cur_fut_gt_11_dim=neighbor_cur_fut_horizon_gt_11_dim,
            )

        # (선택) 기존처럼 “전체 길이” future_all도 계속 내보내고 싶다면:
        # - 여기서는 추가적인 101 규칙 적용이 아니라, raw 변환만 제공합니다.
        # - shape: (N, Tf_all, 3)
        neighbor_future_all_gt_11_dim = neighbor_cur_fut_all_gt_11_dim[:,
                                                                       1:, :]  # 현재(0) 제외
        neighbor_future_all_gt_3_dim = self._traj11_to_traj3_yaw(
            neighbor_future_all_gt_11_dim)

        static_objects = build_static_feature(
            present_static_feat_5=present_static_feat_5,
            static_types_list=static_types_list,
            max_static_num=self.max_static_num,
            ego_cur_pose_np=ego_cur_pose_np,
            filter_radius=self._get_effective_filter_radius_m(),
        )
        key_to_array = {
            "origin_world_pose": origin_world_pose,  # (4,)
            "ego_agent_past": ego_agent_past,  # (time_len, 11)
            "neighbor_agents_past": neighbor_agents_past,
            # (chosen_agent_num, time_len, 11)

            # ✅ (요구조건 a) 최종 출력(규칙 적용된) future GT
            "neighbor_future_gt_3_dim": neighbor_future_gt_3_dim,
            # (chosen_agent_num, future_len, 3)
            "neighbor_future_gt_11_dim": neighbor_future_gt_11_dim,
            # (chosen_agent_num, future_len, 11)

            # 기존 키 유지(필요시):
            "neighbor_future_all_gt_3_dim": neighbor_future_all_gt_3_dim,
            # (chosen_agent_num, future_all_len, 3)
            "static_objects": static_objects,  # (chosen_static_num, 10)
        }

        key_to_road_safety = self._get_road_safety_features(
            scenario=scenario,
            ego_cur_pose_np=ego_cur_pose_np,
        )
        key_to_array.update(key_to_road_safety)

        (
            route_roadblock_ids,
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        ) = self._prepare_map(
            scenario=scenario,
            ego_state=ego_state,
            ego_point2d=ego_point2d,
            ego_heading=ego_heading,
            map_api=map_api,
            traffic_light_data=traffic_light_data,
        )

        car_token_to_rr_ids: Dict[
            str, List[str]] = self._prepare_car_token_to_rr_ids(
                scenario=scenario,
                use_route_lanes=use_route_lanes,
                neighbor_track_token=neighbor_track_token,
            )

        neighbor_agents_current = neighbor_agents_past[:, -1, :]
        map_key_to_array = map_process(
            route_roadblock_ids, car_token_to_rr_ids, neighbor_track_token,
            neighbor_agents_current, ego_cur_pose_np, elements_to_obj_polylines,
            elements_to_traffic_light, speed_limit_dict,
            lanes_roadblock_id_list, self._map_elements, self._max_map_elements,
            self._map_points_num)
        key_to_array.update(map_key_to_array)

        key_to_array = convert_data_dict_to_device_tensors(
            key_to_array, device, squeeze)

        key_to_array["neighbor_track_token"] = neighbor_track_token

        return key_to_array

    @staticmethod
    def zero_out_random_time_prefix(
            neighbor_agents_past: np.ndarray) -> np.ndarray:
        """주어진 neighbor_agents_past 텐서에서
        (max_agent_num, time_len, feature_dim) 형태를 가정하고,
        0 ~ time_len-1 사이에서 랜덤 target을 뽑아
        neighbor_agents_past[:, :target, :8] 구간을 0으로 만드는 함수.

        Args:
            neighbor_agents_past (np.ndarray):
                입력 텐서. shape = (num_agents, time_len, 11)

        Returns:
            np.ndarray:
                특정 시간 구간을 0으로 채운 텐서. shape 동일.
        """
        num_agents, time_len, feature_dim = neighbor_agents_past.shape

        # 0부터 time_len-1 사이 랜덤 target 선택
        target: int = np.random.randint(0, time_len // 2)
        print("target:", target)

        # 복사본을 만들어 수정 (원본을 바꾸고 싶으면 copy 제거)
        modified_past: np.ndarray = neighbor_agents_past.copy()

        # 첫 8개 feature만 0으로 세팅
        modified_past[:, :target, :8] = 0.0

        return modified_past

    def _merge_and_interpolate_ego_11dim(
        self,
        ego_agent_past: np.ndarray,  # shape: (Tp, 11)
        ego_future_gt_11_dim: np.ndarray,  # shape: (Tf, 11)
        *,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ego 과거~현재 + 미래를 합친 뒤, 유효~유효 사이에 무효(0)가 끼지 않게 만든다."""
        if ego_agent_past.ndim != 2 or ego_agent_past.shape[-1] != 11:
            raise ValueError(
                f"`ego_agent_past` shape는 (Tp, 11)이어야 합니다. got {ego_agent_past.shape}"
            )
        if ego_future_gt_11_dim.ndim != 2 or ego_future_gt_11_dim.shape[
                -1] != 11:
            raise ValueError(
                f"`ego_future_gt_11_dim` shape는 (Tf, 11)이어야 합니다. got {ego_future_gt_11_dim.shape}"
            )

        Tp: int = int(ego_agent_past.shape[0])
        Tf: int = int(ego_future_gt_11_dim.shape[0])
        if Tp == 0 or Tf == 0:
            return ego_agent_past, ego_future_gt_11_dim

        full: np.ndarray = np.concatenate(
            [ego_agent_past, ego_future_gt_11_dim], axis=0).astype(np.float32,
                                                                   copy=True)
        T_full: int = int(full.shape[0])
        current_index: int = Tp - 1

        # ✅ 유효 판정은 '앞 6개(x,y,cos,sin,vx,vy)'만 사용
        valid_mask_1d: np.ndarray = (np.abs(full[:, :6])
                                     > eps).any(axis=1)  # (T_full,)

        if not bool(valid_mask_1d[current_index]):
            return ego_agent_past, ego_future_gt_11_dim

        valid_idx: np.ndarray = np.nonzero(valid_mask_1d)[0]
        if valid_idx.size == 0:
            return ego_agent_past, ego_future_gt_11_dim

        region_mask: np.ndarray = np.zeros((T_full,), dtype=bool)

        if valid_idx.size == 1:
            region_mask[int(valid_idx[0])] = True
        else:
            first_valid: int = int(valid_idx[0])
            last_valid: int = int(valid_idx[-1])
            region_mask[first_valid:last_valid + 1] = True

            if last_valid - first_valid + 1 > valid_idx.size:
                xs: np.ndarray = valid_idx.astype(np.float64)
                seg_idx: np.ndarray = np.arange(first_valid,
                                                last_valid + 1,
                                                dtype=np.float64)
                for dim_idx in range(6):
                    ys: np.ndarray = full[valid_idx, dim_idx].astype(np.float64,
                                                                     copy=False)
                    interp_vals: np.ndarray = np.interp(seg_idx, xs, ys)
                    full[first_valid:last_valid + 1,
                         dim_idx] = interp_vals.astype(np.float32, copy=False)

        type_vec: np.ndarray = ego_agent_past[-1,
                                              8:11].astype(np.float32,
                                                           copy=False)  # (3,)
        rep_size: np.ndarray = ego_agent_past[-1,
                                              6:8].astype(np.float32,
                                                          copy=False)  # (2,)

        full[:, 8:11] = 0.0
        full[region_mask, 8:11] = type_vec
        full[~region_mask, :] = 0.0

        full_b: np.ndarray = full[None, :, :]  # (1, T_full, 11)
        region_b: np.ndarray = region_mask[None, :]  # (1, T_full)
        rep_size_b: np.ndarray = rep_size[None, :]  # (1, 2)

        full_b = self._fill_width_length_with_representative_size(
            traj_11=full_b,
            valid_mask=region_b,
            rep_size=rep_size_b,
        )
        full_b = self._normalize_cos_sin_in_traj_11(
            traj_11=full_b,
            valid_mask=region_b,
        )
        full = full_b[0]  # (T_full, 11)

        new_ego_agent_past: np.ndarray = full[:Tp, :]
        new_ego_future_11: np.ndarray = full[Tp:, :]
        return new_ego_agent_past, new_ego_future_11

    @staticmethod
    def _traj11_to_traj3_yaw(traj_11: np.ndarray) -> np.ndarray:
        """11차원 궤적을 [x, y, yaw] 3차원으로 바꾼다.

        Args:
            traj_11 (np.ndarray):
                - (T, 11) 또는 (N, T, 11)
                - 11차원 = [x, y, cos, sin, vx, vy, width, length, onehot(3)]

        Returns:
            np.ndarray:
                - (T, 3) 또는 (N, T, 3)
                - 3차원 = [x, y, yaw]
        """
        if traj_11.ndim == 2:
            yaw = np.arctan2(traj_11[:, 3], traj_11[:, 2])
            return np.stack([traj_11[:, 0], traj_11[:, 1], yaw], axis=-1)
        if traj_11.ndim == 3:
            yaw = np.arctan2(traj_11[:, :, 3], traj_11[:, :, 2])
            return np.stack([traj_11[:, :, 0], traj_11[:, :, 1], yaw], axis=-1)
        raise ValueError(
            f"`traj_11`은 (T,11) 또는 (N,T,11) 이어야 합니다. got {traj_11.shape}")

    @staticmethod
    def _get_ego_cur_future_gt_3_dim(
        ego_current_4_dim: np.ndarray,  # shape: (4,)
        ego_future_gt_3_dim: np.ndarray,  # shape: (Tf, 3)
        *,
        eps: float = 1e-8,
    ) -> np.ndarray: # (1+Tf, 3)
        """ego의 현재(x,y,cos,sin)와 미래(x,y,yaw)를 이어 붙여 (1+Tf,3)으로 만든다.

        규칙:
        - 현재가 무효(전부 0)이면, 현재+미래 전체를 0으로 만든다.
        """
        future_len: int = int(ego_future_gt_3_dim.shape[0])
        out: np.ndarray = np.zeros((1 + future_len, 3), dtype=np.float32)

        is_valid: bool = bool((np.abs(ego_current_4_dim) > eps).any())
        if not is_valid:
            return out

        yaw: float = float(np.arctan2(ego_current_4_dim[3],
                                      ego_current_4_dim[2]))
        out[0, 0] = float(ego_current_4_dim[0])
        out[0, 1] = float(ego_current_4_dim[1])
        out[0, 2] = yaw

        if future_len > 0:
            out[1:, :] = ego_future_gt_3_dim.astype(np.float32, copy=False)
        return out

    @staticmethod
    def _get_neighbor_cur_future_gt_3_dim(
        neighbor_agents_current_4_dim: np.ndarray,  # shape: (N, 4)
        neighbor_future_gt_3_dim: np.ndarray,  # shape: (N, Tf, 3)
        *,
        eps: float = 1e-8,
    ) -> np.ndarray: # (N, 1+Tf, 3)
        """neighbor의 현재(x,y,cos,sin)와 미래(x,y,yaw)를 이어 붙여 (N,1+Tf,3)을 만든다.

        규칙:
        - 현재가 무효(전부 0)이면, 현재+미래 전체를 0으로 만든다.
        """

        num_agents: int = int(neighbor_agents_current_4_dim.shape[0])
        future_len: int = int(neighbor_future_gt_3_dim.shape[1])
        out: np.ndarray = np.zeros((num_agents, 1 + future_len, 3),
                                   dtype=np.float32)
        if num_agents == 0:
            return out

        valid_mask: np.ndarray = (np.abs(neighbor_agents_current_4_dim)
                                  > eps).any(axis=1)  # (N,)
        if not np.any(valid_mask):
            return out

        yaw = np.arctan2(neighbor_agents_current_4_dim[:, 3],
                         neighbor_agents_current_4_dim[:, 2]).astype(
                             np.float32,
                             copy=False,
                         )
        out[:, 0, 0] = neighbor_agents_current_4_dim[:, 0].astype(np.float32,
                                                                  copy=False)
        out[:, 0, 1] = neighbor_agents_current_4_dim[:, 1].astype(np.float32,
                                                                  copy=False)
        out[:, 0, 2] = yaw
        out[:, 1:, :] = neighbor_future_gt_3_dim.astype(np.float32, copy=False)

        if not np.all(valid_mask):
            out[~valid_mask, :, :] = 0.0
        return out

    @staticmethod
    def _get_near_future_segment_valid(
        ego_cur_future_gt_11_dim: np.ndarray,  # shape: (1+Tf, 11)
        neighbor_cur_future_gt_11_dim: np.ndarray,  # shape: (N, 1+Tf, 11)
        *,
        eps: float = 1e-8,
    ) -> Tuple[np.ndarray, np.ndarray]: # (1+N, Tf)
        """ego+neighbor의 현재~미래 유효 마스크를 만든다.

        유효 기준: 11차원 중 앞 8개(x,y,cos,sin,vx,vy,width,length) 값이 전부 0이면 무효.
        """

        ego_valid: np.ndarray = (np.abs(ego_cur_future_gt_11_dim[:, :8])
                                 > eps).any(axis=1)  # (1+Tf,)
        neighbor_valid: np.ndarray = (
            np.abs(neighbor_cur_future_gt_11_dim[:, :, :8]) > eps).any(
                axis=2)  # (N, 1+Tf)
        near_cur_future_valid: np.ndarray = np.concatenate([ego_valid[None, :], neighbor_valid], axis=0) # (1+N, 1+Tf)
        mask_interval = near_cur_future_valid[..., :-1] & near_cur_future_valid[
            ..., 1:]  # (1+N, Tf)
        return near_cur_future_valid, mask_interval

    def _save_integration_debug_plot(
        self,
        all_cur_future_gt_3_dim: np.ndarray,  # shape: (P, 1+T, 3)
        cur_future_pose_integrated_3_dim: np.ndarray,  # shape: (P, 1+T, 3)
        near_cur_future_valid: np.ndarray,  # shape: (P, 1+T)
        cur_agent_size_2_dim: np.ndarray,  # shape: (P, 2) = (width, length)
    ) -> None:
        """GT vs integrated pose를 배치 플롯으로 PNG 저장한다."""
        if not getattr(self.config, "save_integration_traj", False):
            return
        if not self._save_dir:
            return
        if cur_agent_size_2_dim.ndim != 2 or cur_agent_size_2_dim.shape[-1] != 2:
            raise ValueError(
                "cur_agent_size_2_dim shape must be (P, 2). "
                f"got {cur_agent_size_2_dim.shape}"
            )
        if cur_agent_size_2_dim.shape[0] != all_cur_future_gt_3_dim.shape[0]:
            raise ValueError(
                "cur_agent_size_2_dim P must match trajectory P. "
                f"got {cur_agent_size_2_dim.shape[0]} vs {all_cur_future_gt_3_dim.shape[0]}"
            )
        debug_dir = os.path.join(self._save_dir, "debug_integration")
        os.makedirs(debug_dir, exist_ok=True)
        plot_idx = int(self._debug_plot_counter)
        self._debug_plot_counter = plot_idx + 1
        save_path = os.path.join(
            debug_dir, f"integrate_compare_{plot_idx:06d}.png")

        mask = near_cur_future_valid.astype(bool)
        mask_xy = mask[..., None]

        gt_xy = all_cur_future_gt_3_dim[..., :2].astype(np.float32, copy=False)
        int_xy = cur_future_pose_integrated_3_dim[..., :2].astype(np.float32, copy=False)

        gt_xy = np.where(mask_xy, gt_xy, np.nan)
        int_xy = np.where(mask_xy, int_xy, np.nan)

        x_gt = gt_xy[..., 0]
        y_gt = gt_xy[..., 1]
        x_int = int_xy[..., 0]
        y_int = int_xy[..., 1]

        fig, ax = plt.subplots(figsize=(6, 6))
        # 범례용 더미 라인
        dummy_gt, = ax.plot([], [], color="tab:blue", linestyle="-", label="gt")
        dummy_int, = ax.plot([], [], color="tab:orange", linestyle="--", label="integrated")

        # 배치 플롯 (행=agent, 열=time)
        ax.plot(x_gt.T, y_gt.T, color="tab:blue", alpha=0.6, linewidth=1.0)
        ax.plot(x_int.T, y_int.T, color="tab:orange", alpha=0.6, linewidth=1.0, linestyle="--")

        # heading 방향이 표시된 사각형(속 채우지 않음)
        heading_gt = all_cur_future_gt_3_dim[..., 2].astype(np.float32, copy=False)
        heading_int = cur_future_pose_integrated_3_dim[..., 2].astype(np.float32, copy=False)

        width = cur_agent_size_2_dim[:, 0].astype(np.float32, copy=False)
        length = cur_agent_size_2_dim[:, 1].astype(np.float32, copy=False)
        valid_size = np.isfinite(width) & np.isfinite(length) & (width > 0.0) & (length > 0.0)

        width = width[:, None]
        length = length[:, None]
        half_w = 0.5 * width
        half_l = 0.5 * length

        # GT 사각형
        cos_gt = np.cos(heading_gt)
        sin_gt = np.sin(heading_gt)
        dx_gt = cos_gt * half_l
        dy_gt = sin_gt * half_l
        wx_gt = -sin_gt * half_w
        wy_gt = cos_gt * half_w

        p1x = x_gt + dx_gt + wx_gt
        p1y = y_gt + dy_gt + wy_gt
        p2x = x_gt + dx_gt - wx_gt
        p2y = y_gt + dy_gt - wy_gt
        p3x = x_gt - dx_gt - wx_gt
        p3y = y_gt - dy_gt - wy_gt
        p4x = x_gt - dx_gt + wx_gt
        p4y = y_gt - dy_gt + wy_gt

        poly_gt = np.stack(
            [
                np.stack([p1x, p1y], axis=-1),
                np.stack([p2x, p2y], axis=-1),
                np.stack([p3x, p3y], axis=-1),
                np.stack([p4x, p4y], axis=-1),
            ],
            axis=2,
        )  # (P, T, 4, 2)

        mask_box = mask & valid_size[:, None]
        poly_gt_flat = poly_gt.reshape(-1, 4, 2)
        poly_gt_flat = poly_gt_flat[mask_box.reshape(-1)]
        if poly_gt_flat.size > 0:
            gt_boxes = PolyCollection(
                poly_gt_flat,
                facecolors="none",
                edgecolors="tab:blue",
                linewidths=0.6,
                alpha=0.4,
            )
            ax.add_collection(gt_boxes)

        # Integrated 사각형
        cos_int = np.cos(heading_int)
        sin_int = np.sin(heading_int)
        dx_int = cos_int * half_l
        dy_int = sin_int * half_l
        wx_int = -sin_int * half_w
        wy_int = cos_int * half_w

        q1x = x_int + dx_int + wx_int
        q1y = y_int + dy_int + wy_int
        q2x = x_int + dx_int - wx_int
        q2y = y_int + dy_int - wy_int
        q3x = x_int - dx_int - wx_int
        q3y = y_int - dy_int - wy_int
        q4x = x_int - dx_int + wx_int
        q4y = y_int - dy_int + wy_int

        poly_int = np.stack(
            [
                np.stack([q1x, q1y], axis=-1),
                np.stack([q2x, q2y], axis=-1),
                np.stack([q3x, q3y], axis=-1),
                np.stack([q4x, q4y], axis=-1),
            ],
            axis=2,
        )  # (P, T, 4, 2)

        poly_int_flat = poly_int.reshape(-1, 4, 2)
        poly_int_flat = poly_int_flat[mask_box.reshape(-1)]
        if poly_int_flat.size > 0:
            int_boxes = PolyCollection(
                poly_int_flat,
                facecolors="none",
                edgecolors="tab:orange",
                linewidths=0.6,
                alpha=0.4,
                linestyles="--",
            )
            ax.add_collection(int_boxes)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(handles=[dummy_gt, dummy_int], loc="best")
        fig.tight_layout()
        fig.savefig(
            save_path,
            dpi=900,
            bbox_inches="tight",
            pad_inches=0.02,
        )
        plt.close(fig)

    def _merge_and_interpolate_neighbor_11dim(
        self,
        neighbor_agents_past: np.ndarray,  # (max_agent_num, Tp, 11)
        neighbor_cur_fut_gt_11_dim: np.ndarray,
        # (max_agent_num, Tf, 11)  # 0번이 현재
    ) -> Tuple[np.ndarray, np.ndarray]:
        """neighbor 과거/현재와 현재/미래를 이어 붙인 뒤, "중간에 비었다가 다시 살아나는" 문제를 막는다.

        이 함수가 보장하는 규칙(한 agent 기준)
        -----------------------------------
        (a) 현재 프레임(=past의 마지막)이 무효라면:
            - 과거~미래 전체 프레임을 전부 0으로 만든다.
            - 즉, "현재는 없는데 미래에 갑자기 나타나는" 케이스를 없앤다.

        (b) 현재 프레임이 유효라면:
            - 전체 시퀀스를 과거→미래로 봤을 때,
              유효 프레임과 유효 프레임 사이에 무효(0) 프레임이 끼지 않게 만든다.
            - 방법:
              1) 전체 시퀀스에서 유효 프레임들의 첫 index(first_valid)와 마지막 index(last_valid)를 찾는다.
              2) first_valid~last_valid 구간은 "연속 유효 구간"으로 만들고,
                 그 안에 비어 있던 프레임은 x/y/cos/sin/vx/vy를 직선 중간값으로 채운다.
              3) 구간 밖(prefix/suffix)은 그대로 0으로 둔다.

        Args:
            neighbor_agents_past (np.ndarray):
                shape: (N, Tp, 11)
            neighbor_cur_fut_gt_11_dim (np.ndarray):
                shape: (N, Tf, 11)
                index 0이 "현재"라고 가정한다.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - new_neighbor_agents_past: shape (N, Tp, 11)
                - new_neighbor_future_11:  shape (N, Tf-1, 11)  # 현재 제외
        """
        # 기본 shape 검사
        if neighbor_agents_past.ndim != 3 or neighbor_cur_fut_gt_11_dim.ndim != 3:
            raise ValueError(
                f"`neighbor_agents_past` / `neighbor_cur_fut_gt_11_dim`는 "
                f"(max_agent_num, time_len, 11) 형태여야 합니다. "
                f"got {neighbor_agents_past.shape}, {neighbor_cur_fut_gt_11_dim.shape}"
            )
        if neighbor_agents_past.shape[-1] != 11 or \
                neighbor_cur_fut_gt_11_dim.shape[-1] != 11:
            raise ValueError(
                f"두 입력의 마지막 차원은 11이어야 합니다. "
                f"got {neighbor_agents_past.shape[-1]}, {neighbor_cur_fut_gt_11_dim.shape[-1]}"
            )
        if neighbor_agents_past.shape[0] != neighbor_cur_fut_gt_11_dim.shape[0]:
            raise ValueError(
                "neighbor_agents_past 와 neighbor_cur_fut_gt_11_dim 의 agent 축 크기가 다릅니다."
            )

        max_agent_num: int = int(neighbor_agents_past.shape[0])
        past_len: int = int(neighbor_agents_past.shape[1])  # Tp
        fut_len_with_current: int = int(
            neighbor_cur_fut_gt_11_dim.shape[1])  # Tf (0번 포함)

        # future 프레임이 아예 없으면 그대로 반환
        if fut_len_with_current == 0:
            return neighbor_agents_past, neighbor_cur_fut_gt_11_dim

        # current(0번) 프레임은 항상 한 번 제거
        # neighbor_future_wo_current: (N, Tf-1, 11)
        neighbor_future_wo_current: np.ndarray = neighbor_cur_fut_gt_11_dim[:,
                                                                            1:, :]

        # 에이전트가 0명이면 보간 없이 바로 반환
        if max_agent_num == 0:
            return neighbor_agents_past, neighbor_future_wo_current

        # full_traj_11: (N, Tp + (Tf-1), 11)
        full_traj_11: np.ndarray = np.concatenate(
            [neighbor_agents_past, neighbor_future_wo_current], axis=1)
        T_full: int = int(full_traj_11.shape[1])
        current_index: int = past_len - 1  # full_traj에서 현재는 past의 마지막

        # (1) size 대표값 계산 (기본: 과거~현재만 사용)
        # stable_size: (N, 2) = [width_rep, length_rep]
        stable_size: np.ndarray = self._estimate_stable_neighbor_sizes(
            full_traj_11=full_traj_11,
            past_len=past_len,
            use_future=False,
        )

        # (2) "유효 프레임" 마스크 계산 (앞 8개 값 중 하나라도 0이 아니면 유효)
        full_off_p_mask, _ = self._get_agents_past_cur_mask_np(full_traj_11)
        full_valid_mask: np.ndarray = ~full_off_p_mask  # (N, T_full)

        # (3) rule (a): 현재 프레임이 무효면 전체 0
        current_valid_mask: np.ndarray = full_valid_mask[:,
                                                         current_index]  # (N,)
        invalid_agents: np.ndarray = ~current_valid_mask

        # 결과 버퍼
        full_traj_interp: np.ndarray = full_traj_11.astype(np.float32,
                                                           copy=True)

        # 각 agent별 "연속 유효 구간" 마스크
        # region_mask_all: (N, T_full)
        region_mask_all: np.ndarray = np.zeros((max_agent_num, T_full),
                                               dtype=bool)

        # 현재가 무효인 agent는 전부 0으로 만들고 끝
        if np.any(invalid_agents):
            full_traj_interp[invalid_agents, :, :] = 0.0
            # region_mask_all은 그대로 False

        # (4) rule (b): 현재가 유효인 agent는 유효~유효 사이 구멍을 채움
        for agent_idx in range(max_agent_num):
            if invalid_agents[agent_idx]:
                continue

            agent_valid_idx: np.ndarray = \
            np.nonzero(full_valid_mask[agent_idx])[0]
            if agent_valid_idx.size == 0:
                # (현재는 유효인데 valid_idx가 0인 경우는 거의 없지만, 안전하게 0 처리)
                full_traj_interp[agent_idx, :, :] = 0.0
                continue

            first_valid: int = int(agent_valid_idx[0])
            last_valid: int = int(agent_valid_idx[-1])

            if agent_valid_idx.size == 1:
                # 유효 프레임이 1개면 "유효-무효-유효" 자체가 성립하지 않으므로 그대로 둠
                region_mask_all[agent_idx, first_valid] = True
            else:
                # first~last를 "연속 유효 구간"으로 선언
                region_mask_all[agent_idx, first_valid:last_valid + 1] = True

                # 중간에 빈 프레임이 있을 때만 x/y/cos/sin/vx/vy를 직선 중간값으로 채움
                if last_valid - first_valid + 1 > agent_valid_idx.size:
                    xs: np.ndarray = agent_valid_idx.astype(np.float64)  # (K,)
                    seg_idx: np.ndarray = np.arange(first_valid,
                                                    last_valid + 1,
                                                    dtype=np.float64)

                    for dim_idx in range(6):  # 0~5: [x, y, cos, sin, vx, vy]
                        ys: np.ndarray = full_traj_11[agent_idx,
                                                      agent_valid_idx,
                                                      dim_idx].astype(
                                                          np.float64,
                                                          copy=False)
                        interp_vals: np.ndarray = np.interp(seg_idx, xs, ys)
                        full_traj_interp[agent_idx, first_valid:last_valid + 1,
                                         dim_idx] = interp_vals.astype(
                                             np.float32, copy=False)

            # 타입(one-hot)은 agent당 하나로 고정해서 "연속 유효 구간"에만 채움
            type_candidates: np.ndarray = full_traj_11[agent_idx, :,
                                                       8:11]  # (T_full, 3)
            type_valid_mask: np.ndarray = (np.abs(type_candidates).sum(axis=1)
                                           > 0)
            if np.any(type_valid_mask):
                type_vec: np.ndarray = type_candidates[type_valid_mask][
                    0].astype(np.float32, copy=False)  # (3,)
            else:
                type_vec = np.zeros((3,), dtype=np.float32)

            full_traj_interp[agent_idx, :, 8:11] = 0.0
            full_traj_interp[agent_idx, region_mask_all[agent_idx],
                             8:11] = type_vec

            # 연속 유효 구간 밖은 완전히 0으로
            full_traj_interp[agent_idx, ~region_mask_all[agent_idx], :] = 0.0

        # (5) width/length를 대표값으로 통일해서 "연속 유효 구간"에만 채움
        full_traj_interp = self._fill_width_length_with_representative_size(
            traj_11=full_traj_interp,
            valid_mask=region_mask_all,  # (N, T_full)
            rep_size=stable_size,  # (N, 2)
        )

        # (6) cos/sin 길이를 1로 정리 (연속 유효 구간에만 적용)
        full_traj_interp = self._normalize_cos_sin_in_traj_11(
            traj_11=full_traj_interp,
            valid_mask=region_mask_all,
        )

        # 과거/현재와 미래로 다시 분리
        new_neighbor_agents_past: np.ndarray = full_traj_interp[:, :past_len, :]
        new_neighbor_future_11: np.ndarray = full_traj_interp[:,
                                                              past_len:, :]  # 현재 제외된 미래

        return new_neighbor_agents_past, new_neighbor_future_11

    @staticmethod
    def _get_agents_past_cur_mask_np(
            neighbor_agents_past: np.ndarray,  # (agents_num, time_len, 11)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """에이전트 과거/현재 시퀀스에서 유효성 마스크를 **NumPy 입출력**으로 계산한다.

        정의
        ----
        - 프레임 유효(on/off) 판정(포인트 단위):
          마지막 차원 앞 8개([x, y, cos, sin, vx, vy, width, length]) 값 중
          하나라도 0이 아니면 **유효(True)**, 모두 0이면 **무효(False)**.
        - 에이전트 유효(on/off) 판정(에이전트 단위):
          해당 에이전트의 모든 프레임이 무효이면 **무효(True)**.

        Args:
            neighbor_agents_past (np.ndarray):
                에이전트 과거/현재 시퀀스. shape = (agents_num, time_len, 11)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - agents_past_cur_off_p_mask (np.ndarray): shape = (agents_num, time_len), dtype=bool
                  각 프레임이 **무효(True)** 인지 여부(포인트 단위 마스크).
                - agents_past_cur_off_mask (np.ndarray): shape = (agents_num,), dtype=bool
                  에이전트 전체가 **무효(True)** 인지 여부(에이전트 단위 마스크).

        Raises:
            ValueError: 입력이 (N, T, 11) 형태가 아니거나 마지막 차원(<8)일 때.
        """
        if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[-1] < 8:
            raise ValueError(
                f"`neighbor_agents_past`는 (agents_num, time_len, 11) 형태여야 하며 "
                f"마지막 차원은 최소 8이어야 합니다. got {neighbor_agents_past.shape}")

        # (agents_num, time_len, 8)  — 0이 아니면 True
        agents_past_current_is_not_zero = (neighbor_agents_past[..., :8] != 0)

        # (agents_num, time_len) — 8개 값 중 하나라도 0이 아니면 유효
        agents_past_current_not_zero_num = agents_past_current_is_not_zero.sum(
            axis=-1)
        agents_past_cur_off_p_mask = (agents_past_current_not_zero_num == 0
                                     )  # 무효(True)

        # (agents_num) — 에이전트 단위: 유효 프레임 수가 0이면 무효(True)
        agents_past_cur_on_p_mask = ~agents_past_cur_off_p_mask
        agents_past_cur_off_mask = (agents_past_cur_on_p_mask.sum(axis=-1) == 0)

        return agents_past_cur_off_p_mask.astype(
            bool), agents_past_cur_off_mask.astype(bool)

    def _get_past_cur_agents_feature(
        self,
        scenario: Optional[NuPlanScenario] = None,
        observation_buffer: Optional[Deque[Observation]] = None,
    ) -> Tuple[
            List[np.ndarray],  # past_cur_agents_world_8_list
            List[List[TrackedObjectType]],  # past_cur_agents_types_list
            np.ndarray,  # present_static_feat_5
            List[TrackedObjectType],  # static_types_list
            Dict[str, int],  # token_to_id
            Optional[TrackedObjects],  # present_tracked_objects
            Optional[List[TrackedObjects]],  # past_cur_tracked_objects
    ]:
        """과거+현재 에이전트 / 정적 객체 정보를 공통 포맷으로 추출하는 함수.

        두 가지 입력 경로를 지원합니다.

        1) 오프라인 전처리 (scenario 기반)
            - nuPlanScenario 에서
              · 과거+현재 프레임의 동적 객체(차량/보행자/자전거) 배열
              · 현재 프레임의 정적 객체(표지판, 배리어 등) 배열
              을 추출합니다.

        2) 온라인 시뮬레이션 (observation_buffer 기반)
            - 시뮬레이터의 observation_buffer(연속 관측치)에서
              같은 형태의 정보를 뽑아냅니다.

        두 입력을 동시에 쓰거나, 둘 다 안 주면 오류를 발생시킵니다.

        Args:
            scenario (Optional[NuPlanScenario]):
                - 오프라인 전처리용 nuPlan 시나리오.
                - 과거/현재의 TrackedObjects 를 직접 얻을 때 사용.
            observation_buffer (Optional[Deque[Observation]]):
                - 시뮬레이션 중 관측 버퍼(과거 → 현재 순서).
                - 각 원소는 보통 `DetectionsTracks` 타입이며,
                  그 안에 `.tracked_objects` 가 들어 있습니다.

        Returns:
                - past_cur_agents_world_8_list:
                    · 길이: num_frames
                    · 각 원소: (frame_agents_num, 8) float 배열
                    · 각 행 = 한 에이전트, 열 = ID/속도/방향/크기/위치 등
                - past_cur_agents_types_list:
                    · 길이: num_frames
                    · 각 프레임에서 에이전트 타입(차량/보행자/자전거) 리스트
                - present_static_feat_5:
                    · 모양: (cur_static_num, 5)
                    · [x, y, heading, width, length] (현재 프레임의 정적 객체)
                - static_types_list:
                    · 길이: cur_static_num
                    · 각 정적 객체의 타입 리스트
                - token_to_id: Dict[str, int]:
                    · 현재 프레임에 등장하는 에이전트 토큰 → 정수 ID
                    - (딕셔너리)
                - present_tracked_objects:
                    · scenario 경로일 때만 유효(현재 프레임의 TrackedObjects)
                    · observation_buffer 경로에서는 None
                - past_cur_tracked_objects:
                    · scenario 경로일 때만 유효(과거+현재 TrackedObjects 리스트)
                    · observation_buffer 경로에서는 None
        """
        # 입력 유효성 검사
        if (scenario is None and observation_buffer is None) or \
           (scenario is not None and observation_buffer is not None):
            raise ValueError(
                "scenario 또는 observation_buffer 중 정확히 하나만 전달해야 합니다.")
        # --------------------------------------------------
        # 1) scenario 기반 (오프라인 전처리 / work()에서 사용)
        # --------------------------------------------------
        if scenario is not None:
            # 현재 프레임의 동적 객체
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            # 과거 프레임의 동적 객체들
            past_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses,
                )
            ]

            # 과거 + 현재를 시간순으로 이어붙인 리스트
            past_cur_tracked_objects = past_tracked_objects + [
                present_tracked_objects
            ]

            agents_source_seq = past_cur_tracked_objects  # List[TrackedObjects]
            static_source = present_tracked_objects  # TrackedObjects
        else:
            # --------------------------------------------------
            # 2) observation_buffer 기반 (온라인 inference / observation_adapter)
            # --------------------------------------------------
            assert observation_buffer is not None  # 타입 체커용

            present_tracked_objects = None
            past_cur_tracked_objects = None

            agents_source_seq = observation_buffer  # Deque[Observation]
            static_source = observation_buffer[-1]  # 가장 최근 프레임, Observation

        # 공통 로직: 에이전트 시퀀스 → 프레임별 에이전트 배열/타입
        past_cur_agents_world_8_list, past_cur_agents_types_list, token_to_id = \
            sampled_tracked_objects_to_array_list(agents_source_seq)

        # 공통 로직: 현재 프레임의 정적 객체 배열/타입
        present_static_feat_5, static_types_list = \
            sampled_static_objects_to_array_list(static_source)

        # 하나의 return 지점
        return (
            past_cur_agents_world_8_list,  # List[np.ndarray], #  (frame_agents_num, 8)
            past_cur_agents_types_list,  # List[List[TrackedObjectType]],
            present_static_feat_5,  # np.ndarray, (len(static_obj), 5)
            static_types_list,  # List[TrackedObjectType],
            token_to_id,  # Dict[str, int],
            present_tracked_objects,  # Optional[TrackedObjects],
            past_cur_tracked_objects,  # Optional[List[TrackedObjects]]
        )

    def _prepare_map(
        self,
        scenario: NuPlanScenario,
        ego_state: EgoState,
        ego_point2d: Point2D,
        ego_heading: float,
        map_api: NuPlanMap,
        traffic_light_data: Optional[List[TrafficLightStatusData]] = None,
    ) -> Tuple[List[str], Dict[str, MapObjectPolylines], Dict[
            str, LaneSegmentTrafficLightData], Dict[str, np.ndarray],
               List[str]]:
        """지도 관련 입력(route/차선/신호/속도제한)을 한 번에 준비하는 공통 유틸.

        공통 흐름:
          1) 시나리오의 route_roadblock_ids 를 가져와 끊어진 구간을 보정한다.
          2) ego 주변의 차선/경계/신호/속도제한 정보를 get_neighbor_vector_set_map 으로 뽑는다.
             - 온라인(inference) 경로: 외부에서 넘어온 traffic_light_data 사용
             - 오프라인(work) 경로: traffic_light_data 가 None 이므로 iteration=0 기준으로 자체 조회

        Args:
            scenario: nuPlan 시나리오 객체.
            ego_state: 현재 ego 상태 (rear_axle 기준).
            ego_point2d: ego 위치 (x, y).
            ego_heading: ego 진행 방향(rad).
            map_api: NuPlanMap 인스턴스.
            traffic_light_data:
                - observation_adapter 경로: 현재 시점의 신호등 리스트를 그대로 전달
                - work 경로: None → 시나리오 0번 iteration 에서 조회
        """
        # 1) route roadblock 보정
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, list(route_roadblock_ids))
        else:
            route_roadblock_ids = []

        # 2) 신호등 데이터 준비
        if traffic_light_data is None:
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

        # 3) ego 주변 차선/경계/신호/속도제한 추출
        """
    1. elements_to_obj_polylines: Dict[str, MapObjectPolylines],
       - 키: 맵 요소 이름 문자열 "LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "CROSSWALK", ...
       - 값: 해당 요소를 이루는 점들의 모음(MapObjectPolylines)
    - 내부 구조: [num_elements, num_points_i, 2]
    2. elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
       - 키: 맵 요소 이름 문자열(현재 "LANE"만 사용)
       - 값: 해당 요소에 대응되는 신호등 상태 정보 (LaneSegmentTrafficLightData)
            - 내부 구조: (num_lanes, 4) one-hot
    3. speed_limit_dict: Dict[str, np.ndarray],
       - "lane_has_speed_limit": (num_lanes,), bool
       - "lane_speed_limit": (num_lanes,), float32
    4. lanes_roadblock_id_list: List[str],
       - 각 차선이 속한 도로 묶음(roadblock) ID 리스트 (길이 = num_lanes)
        """
        (
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        ) = get_neighbor_vector_set_map(
            map_api,
            self._map_elements,
            ego_point2d,
            ego_heading,
            self._get_map_query_radius_m(),
            traffic_light_data,
        )

        return (
            route_roadblock_ids,
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        )

    def _get_road_safety_features(
        self,
        scenario: NuPlanScenario,
        ego_cur_pose_np: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        key_to_road_safety = {}
        stop_sign_points = extract_stop_sign_points(
            scenario,
            ego_cur_pose_np,
            self.config.safety_len,
            self._get_map_query_radius_m(),
        )
        crosswalk_points = extract_crosswalk_points(
            scenario,
            ego_cur_pose_np,
            self.config.safety_len,
            self._get_map_query_radius_m(),
        )
        key_to_road_safety["stop_sign_points"] = stop_sign_points
        key_to_road_safety["crosswalk_points"] = crosswalk_points
        return key_to_road_safety

    @staticmethod
    def _concat_past_future_vxy_from_traj11(
        past_cur_traj_11: np.ndarray,  # (time_len,11) or (A,time_len,11)
        future_traj_11: np.ndarray,    # (future_len,11) or (A,future_len,11)
    ) -> np.ndarray:
        """past~current~future의 (v_x, v_y) 시퀀스를 만든다.

        Args:
            past_cur_traj_11 (np.ndarray):
                - ego: (time_len, 11)
                - neighbor: (A, time_len, 11)
                - time_len = past_len+1 (현재 포함)
            future_traj_11 (np.ndarray):
                - ego: (future_len, 11)
                - neighbor: (A, future_len, 11)
                - future_len = 현재 제외 미래 길이
        Returns:
            np.ndarray:
                - ego: (time_len+future_len, 2)
                - neighbor: (A, time_len+future_len, 2)
                - 마지막 2는 [v_x, v_y]
        """
        past = np.asarray(past_cur_traj_11)
        fut = np.asarray(future_traj_11)

        if past.ndim != fut.ndim:
            raise ValueError(f"past/future ndim mismatch: past={past.shape}, future={fut.shape}")
        if past.shape[-1] != 11 or fut.shape[-1] != 11:
            raise ValueError(f"last dim must be 11: past={past.shape}, future={fut.shape}")

        if past.ndim == 2:
            axis_time = 0
        elif past.ndim == 3:
            if past.shape[0] != fut.shape[0]:
                raise ValueError(f"A(agent) dim mismatch: past={past.shape}, future={fut.shape}")
            axis_time = 1
        else:
            raise ValueError(f"traj_11 must be 2D or 3D. got past={past.shape}")

        past_vxy = past[..., 4:6]
        fut_vxy = fut[..., 4:6]
        vxy = np.concatenate([past_vxy, fut_vxy], axis=axis_time).astype(np.float32, copy=False)
        return vxy

    @staticmethod
    def _build_past_future_yaw_inputs_from_traj11(
        past_cur_traj_11: np.ndarray,  # (time_len,11) or (A,time_len,11)
        future_traj_11: np.ndarray,    # (future_len,11) or (A,future_len,11)
        *,
        eps_valid: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """yaw_rate 계산에 필요한 (cs_yaw, valid_mask)를 만든다.

        - valid_mask: 앞 8개 값([x,y,cos,sin,vx,vy,width,length])이 전부 0이면 무효(False)

        Args:
            past_cur_traj_11 (np.ndarray):
                - ego: (time_len, 11)
                - neighbor: (A, time_len, 11)
            future_traj_11 (np.ndarray):
                - ego: (future_len, 11)
                - neighbor: (A, future_len, 11)
            eps_valid (float):
                0 판정 기준(아주 작은 값).

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - past_future_cs_yaw:
                    · ego: (time_len+future_len, 2)
                    · neighbor: (A, time_len+future_len, 2)
                    · 마지막 2는 [cos(yaw), sin(yaw)]
                - past_future_valid:
                    · ego: (time_len+future_len,) bool
                    · neighbor: (A, time_len+future_len) bool
        """
        past = np.asarray(past_cur_traj_11)
        fut = np.asarray(future_traj_11)

        if past.ndim != fut.ndim:
            raise ValueError(f"past/future ndim mismatch: past={past.shape}, future={fut.shape}")
        if past.shape[-1] != 11 or fut.shape[-1] != 11:
            raise ValueError(f"last dim must be 11: past={past.shape}, future={fut.shape}")

        if past.ndim == 2:
            axis_time = 0
        elif past.ndim == 3:
            if past.shape[0] != fut.shape[0]:
                raise ValueError(f"A(agent) dim mismatch: past={past.shape}, future={fut.shape}")
            axis_time = 1
        else:
            raise ValueError(f"traj_11 must be 2D or 3D. got past={past.shape}")

        # valid: (time_len,) or (A,time_len)
        past_valid = (np.abs(past[..., :8]) > float(eps_valid)).any(axis=-1)
        fut_valid = (np.abs(fut[..., :8]) > float(eps_valid)).any(axis=-1)
        past_future_valid = np.concatenate([past_valid, fut_valid], axis=axis_time).astype(bool, copy=False)

        # cs_yaw: (time_len,2) or (A,time_len,2)
        past_cs = past[..., 2:4]
        fut_cs = fut[..., 2:4]
        past_future_cs_yaw = np.concatenate([past_cs, fut_cs], axis=axis_time).astype(np.float32, copy=False)

        return past_future_cs_yaw, past_future_valid

    def _build_past_future_control_vxy_yawrate_from_traj11(
        self,
        past_cur_traj_11: np.ndarray,  # (time_len,11) or (A,time_len,11)
        future_traj_11: np.ndarray,    # (future_len,11) or (A,future_len,11)
        *,
        dt: float,
        polyorder: int,
        max_window_len_yaw: int,
        eps_valid: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """past~current~future 기반 control(3) = [v_x, v_y, yaw_rate] 를 만든다.

        흐름:
          1) vxy 시퀀스 만들기
          2) cs_yaw + valid 만들기
          3) cs_yaw + valid로 SG(yaw_rate) 계산
          4) vxy + yaw_rate를 concat해서 control(3) 만들기

        Args:
            past_cur_traj_11: (time_len,11) 또는 (A,time_len,11)
            future_traj_11:   (future_len,11) 또는 (A,future_len,11)
            dt: 샘플 간 시간 간격
            polyorder: SG 다항 차수
            max_window_len_yaw: yaw_rate 미분 창 길이
            eps_valid: 유효 판정 기준

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - past_future_control:
                    · ego: (time_len+future_len, 3)
                    · neighbor: (A, time_len+future_len, 3)
                    · 마지막 3은 [v_x, v_y, yaw_rate]
                - past_future_yaw_rate:
                    · ego: (time_len+future_len,)
                    · neighbor: (A, time_len+future_len)
        """
        past_future_vxy = self._concat_past_future_vxy_from_traj11(
            past_cur_traj_11=past_cur_traj_11,
            future_traj_11=future_traj_11,
        )

        past_future_cs_yaw, past_future_valid = self._build_past_future_yaw_inputs_from_traj11(
            past_cur_traj_11=past_cur_traj_11,
            future_traj_11=future_traj_11,
            eps_valid=float(eps_valid),
        )

        past_future_yaw_rate = compute_past_future_yaw_rate_from_cs_yaw_via_feasible_sg(
            past_future_cs_yaw=past_future_cs_yaw,
            past_future_valid=past_future_valid,
            dt=float(dt),
            polyorder=int(polyorder),
            max_window_len_yaw=int(max_window_len_yaw),
        ).astype(np.float32, copy=False)

        # vxy + yaw_rate -> control(3)
        if past_future_vxy.ndim == 2:
            # ego: (T,2) + (T,1) -> (T,3)
            if past_future_yaw_rate.ndim != 1 or past_future_yaw_rate.shape[0] != past_future_vxy.shape[0]:
                raise ValueError(
                    f"ego yaw_rate shape mismatch: vxy={past_future_vxy.shape}, yaw_rate={past_future_yaw_rate.shape}"
                )
            past_future_control = np.concatenate(
                [past_future_vxy, past_future_yaw_rate[:, None]],
                axis=1,
            ).astype(np.float32, copy=False)
        else:
            # neighbor: (A,T,2) + (A,T,1) -> (A,T,3)
            if past_future_yaw_rate.ndim != 2 or past_future_yaw_rate.shape[:2] != past_future_vxy.shape[:2]:
                raise ValueError(
                    f"neighbor yaw_rate shape mismatch: vxy={past_future_vxy.shape}, yaw_rate={past_future_yaw_rate.shape}"
                )
            past_future_control = np.concatenate(
                [past_future_vxy, past_future_yaw_rate[:, :, None]],
                axis=2,
            ).astype(np.float32, copy=False)

        return past_future_control, past_future_yaw_rate
    # Use for data preprocess
    def work(self, scenarios: List[NuPlanScenario]) -> None:
        for scenario in scenarios:
            map_name = scenario._map_name
            scenario_token = scenario.token
            map_api = scenario.map_api

            (ego_state, ego_point2d, ego_heading, ego_cur_pose_np,
             past_cur_ego_world_10,
             past_cur_time_np) = self._get_past_cur_ego_feature(
                 scenario=scenario,
                 set_coord_as_center=self.set_coord_as_center,
             )
            # ✅ 추가: ego 기준 좌표계 원점의 세계좌표 포즈 저장
            # origin_world_pose: shape (4,) = [x_world, y_world, cos(yaw), sin(yaw)]
            origin_world_pose: np.ndarray = self._build_origin_world_pose(
                ego_cur_pose_np)

            ego_agent_past = build_ego_past_feature(
                past_cur_ego_world_10=past_cur_ego_world_10,
                ego_cur_pose_np=ego_cur_pose_np,
            )
            # ✅ 추가: 과거가 부족해 0으로 채운 프레임(prefix)이 있으면 확실히 0으로 정리
            ego_agent_past = self._enforce_no_invalid_between_valid_in_ego_past(
                ego_agent_past)
            # ─────────────────────────────────────────────
            # ✅ (요구조건 b) ego: past+future(101) 기반으로 규칙 적용
            #    그리고 ego_future_gt_3_dim은 “규칙 적용된 11dim”에서 다시 생성
            # ─────────────────────────────────────────────

            (_, ego_future_gt_11_dim_raw) = get_ego_future_array_from_scenario(
                scenario=scenario,
                current_ego_state=ego_state,
                num_future_poses=self.num_future_poses,
                future_time_horizon=self.future_time_horizon,
                # ✅ 과거를 만들 때 쓴 ego_cur_pose_np를 그대로 넘겨서,
                #    미래도 "완전히 같은 기준"으로 만들기
                ego_cur_pose_np=ego_cur_pose_np,
                # (참고) ego_cur_pose_np가 주어지면 이 값은 사실상 의미가 없지만,
                #        호출 의도를 명확히 하려고 함께 전달
                set_coord_as_center=self.set_coord_as_center,
            )
            ego_agent_past, ego_future_gt_11_dim = self._merge_and_interpolate_ego_11dim(
                ego_agent_past=ego_agent_past,
                ego_future_gt_11_dim=ego_future_gt_11_dim_raw,
            )

            # ✅ 3차원은 반드시 “정리된 11차원”에서 다시 만들기
            ego_future_gt_3_dim = self._traj11_to_traj3_yaw(
                ego_future_gt_11_dim)

            # ✅ 여기서는 추가 center 보정 호출을 하지 않습니다.
            # 이유:
            # - 미래도 이미 ego_cur_pose_np(=과거와 동일 기준)로 만들어졌기 때문입니다.
            # - set_coord_as_center=True일 때 _adjust... 를 또 호출하면 "두 번 이동"이 될 수 있습니다.

            (
                past_cur_agents_world_8_list,
                past_cur_agents_types_list,
                present_static_feat_5,
                static_types_list,
                token_to_id,
                present_tracked_objects,
                past_cur_tracked_objects,
            ) = self._get_past_cur_agents_feature(scenario=scenario)

            (neighbor_agents_past, agents_cur_frame_indices, neighbors_id,
             neighbor_track_token) = build_neighbor_past_feature(
                 past_cur_agents_world_8_list=past_cur_agents_world_8_list,
                 past_cur_agents_types_list=past_cur_agents_types_list,
                 max_agent_num=self.caching_max_agent_num,
                 ego_cur_pose_np=ego_cur_pose_np,
                 max_pedestrians=self.max_pedestrians,
                 max_bicycles=self.max_bicycles,
                 token_to_id=token_to_id,
                 filter_radius=self._get_effective_filter_radius_m(),
             )

            ego_time_len = ego_agent_past.shape[0]
            neighbor_time_len = neighbor_agents_past.shape[1]
            assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
                f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"

            cur_fut_agents_world_8_list = self._get_cur_fut_agents_world_8_list(
                scenario, token_to_id, do_inference=False)

            neighbor_cur_fut_gt_11_dim = agent_future_all_process(
                ego_cur_pose_np=ego_cur_pose_np,
                cur_fut_agents_world_8_list=cur_fut_agents_world_8_list,
                neighbor_token_id=neighbors_id,
                neighbor_agents_past=neighbor_agents_past,
            )

            # ─────────────────────────────────────────────
            # ✅ (요구조건 a) neighbor: past+future(101) 기반으로 규칙 적용 후
            #    neighbor_future_gt_3_dim / neighbor_future_gt_11_dim 최종 생성
            # ─────────────────────────────────────────────
            neighbor_agents_past, neighbor_future_gt_11_dim, neighbor_future_gt_3_dim = \
                self._build_neighbor_future_gt_from_past_and_cur_fut_11dim(
                    neighbor_agents_past=neighbor_agents_past,
                    neighbor_cur_fut_gt_11_dim=neighbor_cur_fut_gt_11_dim,
                )

            static_objects = build_static_feature(
                present_static_feat_5=present_static_feat_5,
                static_types_list=static_types_list,
                max_static_num=self.caching_max_static_num,
                ego_cur_pose_np=ego_cur_pose_np,
                filter_radius=self._get_effective_filter_radius_m(),
            )
            # ego_cur_future_gt_3_dim: (1+Tf, 3)
            ego_cur_future_gt_3_dim = self._get_ego_cur_future_gt_3_dim(
                ego_current_4_dim = ego_agent_past[-1, :4],  # (4,) # x, y, cos(yaw), sin(yaw)
                ego_future_gt_3_dim = ego_future_gt_3_dim,  # (future_len, 3) # x, y, yaw
            )
            # neighbor_cur_future_gt_3_dim: (N, 1+Tf, 3)
            neighbor_cur_future_gt_3_dim = self._get_neighbor_cur_future_gt_3_dim(
                neighbor_agents_current_4_dim = neighbor_agents_past[:, -1, :4],  # (N, 4) # x, y, cos(yaw), sin(yaw)
                neighbor_future_gt_3_dim = neighbor_future_gt_3_dim,  # (N, future_len, 3) # x, y, yaw
            )
            # ego는 (1+Tf, 3) 2D이므로 batch 차원을 추가해 (1, 1+Tf, 3)으로 맞춘다.
            all_cur_future_gt_3_dim = np.concatenate(
                [ego_cur_future_gt_3_dim[None, ...], neighbor_cur_future_gt_3_dim], axis=0
            ) # (1+N, 1+Tf, 3)
            # near_future_segment_valid: (1+N, Tf)
            near_cur_future_valid, near_future_segment_valid = self._get_near_future_segment_valid(
                ego_cur_future_gt_11_dim=np.concatenate([ego_agent_past[-1:, :], ego_future_gt_11_dim], axis=0),  # (1+future_len, 11)
                neighbor_cur_future_gt_11_dim=np.concatenate(
                    [neighbor_agents_past[:, -1:, :], neighbor_future_gt_11_dim],
                    axis=1,
                ),  # (N, 1+future_len, 11)
            )
            # cur_future_control_gt_3_dim: (1+N, Tf, 3)
            cur_future_control_gt_3_dim = differentiate_numpy_pose3_to_control3(all_cur_future_gt_3_dim, dt=0.1)
            cur_future_control_gt_3_dim[~near_future_segment_valid] = 0.0
            # cur_future_control_gt_3_dim 을 적분하여 다시 pose로 만들자. (무효점은 0.)
            # cur_future_pose_integrated_3_dim = integrate_numpy_control3_to_pose3_midpoint(
            #     cur_future_control_gt_3_dim,  # (1+N, Tf, 3)
            #     all_cur_future_gt_3_dim[:, 0, :],  # (1+N, 3) 현재 pose
            #     dt=0.1,
            # )  # (1+N, 1+Tf, 3)
            # cur_future_pose_integrated_3_dim[~near_cur_future_valid] = 0.0
            # # 현재 프레임의 width/length (ego + neighbors)
            # cur_agent_size_2_dim = np.concatenate(
            #     [ego_agent_past[-1, 6:8][None, :], neighbor_agents_past[:, -1, 6:8]],
            #     axis=0,
            # )  # (1+N, 2)
            # # all_cur_future_gt_3_dim 와 cur_future_pose_integrated_3_dim 을 그림으로 그리기 (png로)
            # self._save_integration_debug_plot(
            #     all_cur_future_gt_3_dim=all_cur_future_gt_3_dim,
            #     cur_future_pose_integrated_3_dim=cur_future_pose_integrated_3_dim,
            #     near_cur_future_valid=near_cur_future_valid,
            #     cur_agent_size_2_dim=cur_agent_size_2_dim,
            # )
            # =========================================================
            # ✅ [ADDED] past+future 기반 control(3) = [v_x, v_y, yaw_rate]
            # =========================================================
            ego_past_future_control, ego_past_future_gt_yaw_rate = self._build_past_future_control_vxy_yawrate_from_traj11(
                past_cur_traj_11=ego_agent_past,         # (time_len, 11)
                future_traj_11=ego_future_gt_11_dim,     # (future_len, 11)
                dt=0.1,
                polyorder=2,
                max_window_len_yaw=7,
                eps_valid=1e-8,
            )

            neighbor_past_future_control, neighbor_past_future_gt_yaw_rate = self._build_past_future_control_vxy_yawrate_from_traj11(
                past_cur_traj_11=neighbor_agents_past,      # (A, time_len, 11)
                future_traj_11=neighbor_future_gt_11_dim,   # (A, future_len, 11)
                dt=0.1,
                polyorder=2,
                max_window_len_yaw=7,
                eps_valid=1e-8,
            )


            # cur_future_control_gt_3_dim 에서,
            key_to_array = {
                "origin_world_pose": origin_world_pose,  # (4,)
                "ego_agent_past": ego_agent_past,  # (time_len, 11)
                "ego_future_gt_3_dim": ego_future_gt_3_dim,  # (future_len, 3)
                "ego_future_gt_11_dim": ego_future_gt_11_dim,
                # (future_len, 11)
                "neighbor_agents_past": neighbor_agents_past,
                # (chosen_agent_num, time_len, 11)

                # ✅ (요구조건 a) 최종 출력
                "neighbor_future_gt_3_dim": neighbor_future_gt_3_dim,
                # (chosen_agent_num, future_len, 3)
                "neighbor_future_gt_11_dim": neighbor_future_gt_11_dim,
                # (chosen_agent_num, future_len, 11)
                "static_objects": static_objects,  # (chosen_static_num, 10)
            }

            key_to_road_safety = self._get_road_safety_features(
                scenario=scenario,
                ego_cur_pose_np=ego_cur_pose_np,
            )
            key_to_array.update(key_to_road_safety)

            (
                route_roadblock_ids,
                elements_to_obj_polylines,
                elements_to_traffic_light,
                speed_limit_dict,
                lanes_roadblock_id_list,
            ) = self._prepare_map(
                scenario=scenario,
                ego_state=ego_state,
                ego_point2d=ego_point2d,
                ego_heading=ego_heading,
                map_api=map_api,
            )

            car_token_to_rr_ids: Dict[str,
                                      List[str]] = get_npc_route_roadblock_ids(
                                          scenario, past_cur_tracked_objects,
                                          neighbor_track_token)

            neighbor_agents_current = neighbor_agents_past[:, -1, :]

            map_key_to_array = map_process(
                route_roadblock_ids, car_token_to_rr_ids, neighbor_track_token,
                neighbor_agents_current, ego_cur_pose_np,
                elements_to_obj_polylines, elements_to_traffic_light,
                speed_limit_dict, lanes_roadblock_id_list, self._map_elements,
                self._caching_max_map_elements, self._map_points_num)
            key_to_array.update(map_key_to_array)

            chore_data = {
                "map_name": map_name,
                "token": scenario_token,
            }
            key_to_array.update(chore_data)

            if self.config.make_statistics_when_caching:
                veh_cnt, ped_cnt, bic_cnt = self._count_valid_neighbors_by_type(
                    neighbor_agents_past=neighbor_agents_past)
                ratio_percent, mean_speed_kmh = self._compute_lane_speed_stats(
                    map_key_to_array)
                stats_payload = {
                    "vehicle_count":
                        int(veh_cnt),
                    "pedestrian_count":
                        int(ped_cnt),
                    "bicycle_count":
                        int(bic_cnt),
                    "lane_speed_limit_ratio_percent":
                        float(ratio_percent),
                    "mean_speed_limit_kmh": (None if mean_speed_kmh is None else
                                             float(mean_speed_kmh)),
                }
                self._save_sample_stats_json(map_name, scenario_token,
                                             stats_payload)

            final_file_name = f"{key_to_array['map_name']}_{key_to_array['token']}"
            ego_agent_past = key_to_array["ego_agent_past"]  # (time_len, 11)
            ego_future_gt_11_dim = key_to_array[
                "ego_future_gt_11_dim"]  # (future_len, 11)
            self.save_to_disk(self._save_dir, final_file_name, key_to_array)

            key_to_array["neighbor_track_token"] = neighbor_track_token

            if self.config.save_image:
                save_dir = os.path.join(self._save_dir, "debug_vis")
                save_path = os.path.join(save_dir, f"{final_file_name}.png")
                os.makedirs(save_dir, exist_ok=True)
                key_to_array["token_to_future_traj_wrt_ego"] = None
                draw_machine_fast.draw_world_model_to_png(key_to_array,
                                                          output_data={},
                                                          save_path=save_path)

    @staticmethod
    def _estimate_stable_neighbor_sizes(
        full_traj_11: np.ndarray,  # shape: (N, T_full, 11)
        past_len: int,
        *,
        use_future: bool = False,
        eps: float = 1e-3,
        width_max: float = 20.0,
        length_max: float = 60.0,
    ) -> np.ndarray:  # shape: (N, 2)
        """이웃 에이전트별 width/length 대표값(하나)을 만든다.

        배경
        ----
        nuPlan의 박스 width/length는 프레임마다 조금씩 흔들릴 수 있다.
        그런데 실제 물체의 크기는 시간에 따라 바뀌지 않는 값이므로,
        여러 프레임을 보고 "대표 크기" 하나를 만든 뒤 시간축 전체에 쓰는 편이 안정적이다.

        이 함수가 하는 일
        -----------------
        - 각 에이전트(i)에 대해, 여러 프레임에서 관측된 width/length를 모은다.
        - 그 중에서 "쓸 만한 값"만 남긴 뒤,
          정렬했을 때 가운데 값(중간값)을 대표값으로 선택한다.
          (한두 번 튀는 값이 있어도 평균보다 덜 흔들리기 때문)

        "쓸 만한 값" 조건
        ----------------
        1) 해당 프레임이 패딩이 아님:
           - [x, y, cos, sin, vx, vy] 중 하나라도 0이 아니면 패딩이 아니라고 본다.
        2) width > eps, length > eps
        3) 너무 큰 값은 버린다:
           - width <= width_max, length <= length_max

        시간 구간 선택
        ------------
        - use_future=False:
            과거~현재(past_len 프레임)만 보고 대표값을 만든다.
            (실제로 미래가 없는 환경과 맞추려면 이게 더 안전하다.)
        - use_future=True:
            과거~현재~미래 전체(full_traj_11 전체 프레임)를 보고 대표값을 만든다.
            (완전 오프라인에서 더 많이 평균내고 싶을 때 선택)

        값이 하나도 없을 때(예외 처리)
        ----------------------------
        - 위 조건을 통과한 width/length가 하나도 없다면,
          과거~현재의 마지막 프레임(현재 프레임)의 width/length를 fallback으로 쓴다.
          그것마저 0이면 결과도 0으로 남는다.

        Args:
            full_traj_11 (np.ndarray):
                shape = (N, T_full, 11)
                [x, y, cos, sin, vx, vy, width, length, onehot(3)]
            past_len (int):
                shape 관점에서 과거~현재 길이.
                full_traj_11[:, :past_len, :] 구간이 과거~현재라고 본다.
            use_future (bool):
                True면 미래까지 포함해서 대표 크기를 만든다.
            eps (float):
                0에 매우 가까운 값들을 "없는 값"으로 보기 위한 기준.
            width_max (float):
                말도 안 되게 큰 width를 버리기 위한 상한.
            length_max (float):
                말도 안 되게 큰 length를 버리기 위한 상한.

        Returns:
            np.ndarray:
                shape = (N, 2)
                각 에이전트의 [width_rep, length_rep] (float32).
        """
        if full_traj_11.ndim != 3 or full_traj_11.shape[-1] != 11:
            raise ValueError(
                f"`full_traj_11` shape는 (N, T, 11)이어야 합니다. got {full_traj_11.shape}"
            )
        if past_len <= 0 or past_len > full_traj_11.shape[1]:
            raise ValueError(
                f"`past_len`은 1 이상이며 T_full 이하이어야 합니다. got past_len={past_len}, T_full={full_traj_11.shape[1]}"
            )

        N: int = int(full_traj_11.shape[0])
        T_full: int = int(full_traj_11.shape[1])

        # 대표값 계산에 사용할 구간 길이
        T_src: int = T_full if use_future else int(past_len)

        # src_traj: (N, T_src, 11)
        src_traj: np.ndarray = full_traj_11[:, :T_src, :]

        # 패딩이 아닌 프레임 마스크: (N, T_src)
        #  - [x, y, cos, sin, vx, vy] 중 하나라도 0이 아니면 True
        dynamic_valid: np.ndarray = (np.abs(src_traj[:, :, :6])
                                     > eps).any(axis=-1)

        # size: (N, T_src, 2) = [width, length]
        size: np.ndarray = src_traj[:, :, 6:8]

        # size 값이 "쓸 만한지" 마스크: (N, T_src)
        size_valid: np.ndarray = ((size[:, :, 0] > eps) &
                                  (size[:, :, 1] > eps) &
                                  (size[:, :, 0] <= float(width_max)) &
                                  (size[:, :, 1] <= float(length_max)) &
                                  dynamic_valid)

        # fallback: 현재 프레임(과거~현재의 마지막) size
        # fallback_size: (N, 2)
        fallback_size: np.ndarray = full_traj_11[:, past_len - 1,
                                                 6:8].astype(np.float32,
                                                             copy=False)

        # out: (N, 2)
        out: np.ndarray = fallback_size.copy()

        # 에이전트별로 대표값(중간값) 계산
        for i in range(N):
            # valid_vals: (K, 2)  K는 유효 샘플 개수(가변)
            valid_vals: np.ndarray = size[i][size_valid[i]]
            if valid_vals.shape[0] == 0:
                continue
            # 중간값(정렬했을 때 가운데 값): (2,)
            out[i] = np.median(valid_vals, axis=0).astype(np.float32,
                                                          copy=False)

        return out.astype(np.float32, copy=False)

    @staticmethod
    def _fill_width_length_with_representative_size(
            traj_11: np.ndarray,  # shape: (N, T, 11)
            valid_mask: np.ndarray,  # shape: (N, T)
            rep_size: np.ndarray,  # shape: (N, 2)
    ) -> np.ndarray:
        """width/length 채널(6:8)을 대표값으로 통일해서 넣는다.

        이 함수가 하는 일
        -----------------
        - traj_11의 width/length 채널을 먼저 대표값으로 채운다.
        - 그 다음 valid_mask가 False인 프레임은 width/length를 0으로 만든다.
          (즉, "유효한 프레임에서만" size가 존재하도록 맞춘다.)

        Args:
            traj_11 (np.ndarray):
                shape = (N, T, 11)
                [x, y, cos, sin, vx, vy, width, length, onehot(3)]
            valid_mask (np.ndarray):
                shape = (N, T)
                True면 유효 프레임, False면 패딩/무효 프레임이라고 본다.
            rep_size (np.ndarray):
                shape = (N, 2)
                각 에이전트의 [width_rep, length_rep].

        Returns:
            np.ndarray:
                shape = (N, T, 11)
                width/length가 대표값으로 통일된 traj_11 (입력을 직접 수정하고 그대로 반환).
        """
        if traj_11.ndim != 3 or traj_11.shape[-1] != 11:
            raise ValueError(
                f"`traj_11` shape는 (N, T, 11)이어야 합니다. got {traj_11.shape}")
        if valid_mask.shape != traj_11.shape[:2]:
            raise ValueError(
                f"`valid_mask` shape는 (N, T)이어야 합니다. got {valid_mask.shape}, expected {traj_11.shape[:2]}"
            )
        if rep_size.shape != (traj_11.shape[0], 2):
            raise ValueError(
                f"`rep_size` shape는 (N, 2)이어야 합니다. got {rep_size.shape}, expected {(traj_11.shape[0], 2)}"
            )

        # 대표 size를 모든 프레임에 채우고, 유효 마스크로 무효 프레임은 0 처리
        # traj_11[:, :, 6:8]: (N, T, 2)
        traj_11[:, :, 6:8] = rep_size[:, None, :].astype(traj_11.dtype,
                                                         copy=False)
        traj_11[:, :, 6:8] *= valid_mask[:, :, None].astype(traj_11.dtype,
                                                            copy=False)
        return traj_11

    def _get_future_tracked_objects_array_list(
        self,
        scenario: NuPlanScenario,
        token_to_id: Dict[str, int],
        iteration: int = 0,
        future_time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """현재 시점부터 일정 시간 동안의 모든 에이전트 상태를
        프레임별 배열 리스트로 뽑아낸다.

        하는 일 요약
        -------------
        1) 주어진 iteration 에서
           - 현재 프레임의 TrackedObjects
           - 그 이후 future_time_horizon 동안, num_samples 개의 미래 TrackedObjects
           를 가져온다.

        2) `sampled_tracked_objects_to_array_list` 를 통해,
           각 프레임을 (frame_agents_num, 8) 형태의 배열로 바꾼다.
           - 각 행: [track_id, vx, vy, heading, width, length, x, y]
           - 프레임마다 에이전트 수(frame_agents_num)는 달라질 수 있다.
           - 리스트 순서는 [현재, t+1, t+2, ...] 시간 순서.

        3) 동시에, track_token(문자열)을 일관된 정수 ID 로 바꿔주는
           token_to_id 매핑 사전도 함께 만든다.

        Args:
            iteration (int, optional):
                기준이 되는 현재 step 인덱스(0 기반).
            future_time_horizon (Optional[float], optional):
                현재 이후로 몇 초까지 볼 것인지. None 이면 self.future_time_horizon 사용.
            num_samples (Optional[int], optional):
                몇 개의 미래 프레임을 뽑을지. None 이면 self.num_future_poses 사용.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]:
                - cur_fut_agents_world_8_list: List[np.ndarray]
                    · 길이: 1 + num_samples
                    · 각 원소 shape: (frame_agents_num_t, 8)
                      [track_id, vx, vy, heading, width, length, x, y]
                    · 리스트 순서: [현재, t+1, t+2, ...]
                - token_to_id: Dict[str, int]
                    · 전체 프레임에서 등장한 track_token → 정수 ID 매핑 사전.
        """
        present_tracked_objects: TrackedObjects = scenario.get_tracked_objects_at_iteration(
            iteration).tracked_objects

        if future_time_horizon is None:
            future_time_horizon = self.future_time_horizon
        if num_samples is None:
            num_samples = self.num_future_poses

        # 미래 프레임들의 TrackedObjects 리스트
        future_tracked_objects: List[TrackedObjects] = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=future_time_horizon,
                num_samples=num_samples,
            )
        ]

        # [현재] + [미래들] 을 하나의 시퀀스로 합친다.
        sampled_future_observations: List[TrackedObjects] = [
            present_tracked_objects
        ] + future_tracked_objects

        # cur_fut_agents_world_8_list: List[np.ndarray]
        #   - 각 원소: (frame_agents_num, 8)
        # token_to_id: Dict[str, int]
        (cur_fut_agents_world_8_list, _,
         token_to_id) = sampled_tracked_objects_to_array_list(
             sampled_future_observations, token_to_id)

        return cur_fut_agents_world_8_list, token_to_id

    @staticmethod
    def _fsync_directory(dir_path: str) -> None:
        """파일 이름 교체(os.replace)가 디스크에 기록되도록 디렉토리를 fsync 한다.

        주의:
            - 이 동작은 꽤 느릴 수 있어 "매번" 하지 않고 필요할 때만 호출하는 용도입니다.
            - 운영체제/파일시스템에 따라 동작이 다를 수 있으므로 실패해도 무시합니다.

        Args:
            dir_path (str): 저장 폴더 경로
        """
        try:
            dir_fd = os.open(dir_path, os.O_RDONLY)
        except Exception:
            return

        try:
            os.fsync(dir_fd)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                os.close(dir_fd)

    def save_to_disk(self, dir: str, final_file_name: str,
                     data: Dict[str, Any]) -> None:
        """npz 파일을 '임시 파일(.tmp) -> 최종 파일 교체' 방식으로 저장합니다.

        이번 변경의 핵심
        --------------
        - 기존: 매 파일마다 `os.fsync()`로 "디스크가 진짜 쓸 때까지" 기다림 → 매우 느려질 수 있음
        - 변경: 기본값으로는 `os.fsync()`를 하지 않음 → OS가 쓰기를 모아서 처리 가능 → 속도 개선

        필요하면(안전성 조금 더 원할 때)
        -----------------------------
        - config.save_fsync_every_n > 0 이면, N개 저장마다 1번만 파일 fsync 수행
        - config.save_dir_fsync_every_n > 0 이면, N개 저장마다 1번만 디렉토리 fsync 수행
          (파일 이름 교체까지 디스크에 남기고 싶을 때)

        Args:
            dir (str): 저장 폴더 경로
            final_file_name (str): 확장자 제외 파일명
            data (Dict[str, Any]):
                np.savez 또는 np.savez_compressed에 넘길 값들.
                (예: np.ndarray, 숫자, 문자열 등)

        Raises:
            BaseException:
                저장 중 오류가 나면 예외를 그대로 올리고,
                남아있는 .tmp 파일은 정리합니다.
        """
        final_path = f"{dir}/{final_file_name}.npz"
        tmp_path = final_path + ".tmp"

        os.makedirs(dir, exist_ok=True)

        # 이번 저장이 몇 번째 저장인지(성공한 저장만 카운트)
        next_count: int = int(self._save_counter + 1)

        # "N개마다 1번"만 강제 반영
        need_file_fsync: bool = (self._save_fsync_every_n > 0 and
                                 (next_count % self._save_fsync_every_n == 0))
        need_dir_fsync: bool = (self._save_dir_fsync_every_n > 0 and
                                (next_count % self._save_dir_fsync_every_n
                                 == 0))

        try:
            # 1) 임시 파일에 먼저 저장
            with open(tmp_path, "wb") as f:
                if self._save_use_compression:
                    np.savez_compressed(f, **data)
                else:
                    np.savez(f, **data)

                # ✅ 기본은 fsync 안 함(속도 목적)
                # 필요할 때만(예: N개마다 1번) 파일 fsync
                if need_file_fsync:
                    f.flush()
                    os.fsync(f.fileno())

            # 2) 저장이 끝난 임시 파일을 최종 파일명으로 교체(원자적 교체)
            os.replace(tmp_path, final_path)

            # 필요할 때만 디렉토리 fsync(파일 이름 교체까지 디스크에 남기고 싶을 때)
            if need_dir_fsync:
                self._fsync_directory(dir)

            # 성공한 저장만 카운트 반영
            self._save_counter = next_count

        except BaseException:  # Ctrl+C 포함
            with contextlib.suppress(Exception):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise
