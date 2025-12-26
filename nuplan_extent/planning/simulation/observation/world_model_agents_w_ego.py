from typing import cast, List, Dict, Optional, Deque, Tuple, Union
import numpy as np
import numpy.typing as npt
import draw_machine as draw_machine
from nuplan.common.actor_state.dynamic_car_state import get_velocity_shifted
from nuplan_extent.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states, transform_ego_predictions_to_states
from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from collections import deque
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

np.set_printoptions(precision=3, suppress=True)
from nuplan_extent.planning.simulation.observation.smoother import (
    compute_forward_backward_signs,
    slip_limit_stage,
    yawrate_smooth_stage,
    SmootherConfig,
    SlipParams,
    RateParams,
)
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.observation.abstract_ml_agents import AbstractMLAgents
from nuplan.common.actor_state.scene_object import SceneObjectMetadata
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sort_dict
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan_extent.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan_extent.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.common.actor_state.oriented_box import OrientedBox

from scipy.spatial.distance import cdist
# /Users/user/PycharmProjects/nuplan-devkit/nuplan/common/actor_state/tracked_objects.py
from nuplan.common.utils.interpolatable_state import InterpolatableState
from decimal import Decimal, ROUND_HALF_UP
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative, ego_local_traj3_to_global
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.common.geometry.convert import numpy_array_to_absolute_velocity
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.common.actor_state.waypoint import Waypoint
# [Add]
from collections import OrderedDict
from typing import Literal


def agent_to_feature_vector(
    agent: Agent,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Agent(또는 AgentState) 하나를 (11,) 크기의 특징 벡터로 변환합니다.

    벡터 구성 순서:
    - [0] center x
    - [1] center y
    - [2] center yaw
    - [3] center v_x
    - [4] center v_y
    - [5] 패딩 0.0
    - [6] 패딩 0.0
    - [7] 1.0
    - [8] 0.0
    - [9] 0.0

    위와 같이 마지막 세 항목이 항상 [1, 0, 0]이 되도록 끝에 고정 배치합니다.

    Args:
        agent: 특징을 추출할 에이전트. `diffusion_planner.data_process.ego_process.Agent`
            또는 호환되는 `nuplan.common.actor_state.agent_state.AgentState`.
        dtype: 반환 numpy 배열의 dtype.

    Returns:
        np.ndarray: shape (output_dim,)의 실수 벡터.
        - 기본 6개 상태: (x, y, cos(yaw), sin(yaw), v_x, v_y)
        - 패딩: (output_dim - 9)개 만큼 0.0
        - 마지막 3개: (1.0, 0.0, 0.0)

    Raises:
        ValueError: output_dim < 9 인 경우 (마지막 3칸 고정 포함 기본 구성을 담을 수 없음).
        AttributeError: agent에 center/velocity 정보가 없거나, 포즈에서 yaw를 찾을 수 없는 경우.

    Note:
        - nuPlan의 `StateSE2`는 보통 `x`, `y`, `heading` 속성을 갖습니다.
        - `StateVector2D`는 보통 `x`, `y`로 속도를 제공합니다(여기서는 v_x, v_y로 사용).
    """

    # 중심 위치/자세
    if not hasattr(agent, "center"):
        raise AttributeError("agent.center 속성이 없습니다.")
    center_pose = agent.center
    x_pos: float = float(center_pose.x)
    y_pos: float = float(center_pose.y)
    yaw_rad: float = float(center_pose.heading)

    # 속도
    if not hasattr(agent, "velocity"):
        raise AttributeError("agent.velocity 속성이 없습니다.")
    v_x: float = float(agent.velocity.x)
    v_y: float = float(agent.velocity.y)

    # 기본 6개 성분
    feature_list = [
        x_pos, y_pos, yaw_rad, v_x, v_y, agent.box.width, agent.box.length, 1.0,
        0.0, 0.0
    ]

    feature_vector = np.asarray(feature_list, dtype=dtype)

    return feature_vector


# [Add]
def sort_refined_trajs_by_first_xy_l2(
    diff_token_to_interp_np_traj_wrt_ego: Dict[str, np.ndarray],
    *,
    invalid_eps: float = 0.0,
) -> Dict[str, np.ndarray]:
    """refined 궤적 사전을 '첫 시점 (x,y) L2 거리' 기준으로 오름차순 정렬해 반환합니다.

    각 value는 shape (T, 11)이며, row = [x, y, cos, sin, vx, vy, length, width, onehot(3,)] 입니다.
    첫 시점의 좌표는 `arr[0, :2]`로 가정합니다.

    정렬 규칙
    - 기본: d_i = sqrt(x_0^2 + y_0^2) (작을수록 앞)
    - invalid_eps > 0 일 때, |x_0| ≤ eps AND |y_0| ≤ eps 이면 d_i = +inf 로 간주해 맨 뒤로 보냅니다.
      (예: 0으로 채워진 더미/미사용 시퀀스를 뒤로 밀고 싶을 때 유용)

    Args:
        diff_token_to_interp_np_traj_wrt_ego: Dict[str, np.ndarray]
            - 각 value: (T, 11), T ≥ 1
        invalid_eps: float, optional
            - 0.0(기본): 무효 처리 없음(정말로 원점이면 거리 0으로 취급)
            - >0.0: |x|,|y|가 eps 이내면 무효(+inf)로 취급해 뒤로 정렬

    Returns:
        Dict[str, np.ndarray]: OrderedDict로 반환(삽입 순서가 '가까운 → 먼' 순서).
                               타입힌트는 Dict지만 실제 객체는 OrderedDict입니다.

    Raises:
        ValueError: 배열 shape가 (T, 11)이 아닌 항목이 있을 때.
    """
    if not diff_token_to_interp_np_traj_wrt_ego:
        return diff_token_to_interp_np_traj_wrt_ego

    sortable_triplets: List[Tuple[str, float, np.ndarray]] = []
    for token, arr in diff_token_to_interp_np_traj_wrt_ego.items():
        if not isinstance(arr,
                          np.ndarray) or arr.ndim != 2 or arr.shape[1] != 11:
            raise ValueError(
                f"`{token}`의 refined traj shape가 (T, 11)이 아닙니다: got {getattr(arr, 'shape', None)}"
            )
        if arr.shape[0] < 1:
            # T == 0인 경우는 비정상 입력으로 간주하고 무한대로 밀어냄
            dist = float("inf")
        else:
            x0, y0 = float(arr[0, 0]), float(arr[0, 1])
            if invalid_eps > 0.0 and (abs(x0) <= invalid_eps and
                                      abs(y0) <= invalid_eps):
                dist = float("inf")
            else:
                # 첫 시점 L2 거리
                dist = float(np.hypot(x0, y0))

        sortable_triplets.append((token, dist, arr))

    # 정렬: 거리 오름차순, 거리 동률이면 token 사전순으로 안정 타이브레이크
    sortable_triplets.sort(key=lambda t: (t[1], t[0]))

    ordered: "OrderedDict[str, np.ndarray]" = OrderedDict(
        (token, arr) for token, _dist, arr in sortable_triplets)
    # Dict[str, np.ndarray]로 반환(삽입 순서 유지)
    return ordered


def waypoint_to_numpy10(waypoint: Waypoint,
                        *,
                        velocity_fill: float = 0.0) -> npt.NDArray[np.float64]:
    """Waypoint를 10차원 numpy 벡터로 변환합니다.

    벡터 구성 (shape=(10,)):
    - [0] center x (m)
    - [1] center y (m)
    - [2] center yaw (radian)
    - [3] center v_x (m/s)
    - [4] center v_y (m/s)
    - [5] width (m)
    - [6] length (m)
    - [7] 1.0
    - [8] 0.0
    - [9] 0.0

    Args:
        waypoint (Waypoint): 변환할 Waypoint 객체.
        velocity_fill (float, optional): waypoint.velocity가 None일 때 v_x, v_y에 채울 값.
            기본값은 0.0입니다. 결측을 명시하고 싶다면 np.nan을 전달하세요.

    Returns:
        numpy.typing.NDArray[np.float64]: shape가 (10,)인 벡터.
    """
    center = waypoint.oriented_box.center
    width = waypoint.oriented_box.width
    length = waypoint.oriented_box.length

    if waypoint.velocity is None:
        v_x = float(velocity_fill)
        v_y = float(velocity_fill)
    else:
        v_x = float(waypoint.velocity.x)
        v_y = float(waypoint.velocity.y)

    vector = np.array(
        [
            float(center.x),  # x
            float(center.y),  # y
            float(center.heading),  # yaw [rad]
            v_x,  # v_x [m/s]
            v_y,  # v_y [m/s]
            float(width),  # width [m]
            float(length),  # length [m]
            1.0,  # constant
            0.0,  # constant
            0.0,  # constant
        ],
        dtype=np.float64,
    )
    # 안전 확인: (10,) 보장
    assert vector.shape == (10,), f"Expected shape (10,), got {vector.shape}"
    return vector


def ego_state_to_numpy10(ego_state: EgoState,
                         *,
                         velocity_fill: float = 0.0) -> np.ndarray:
    center = ego_state.center
    width = ego_state.car_footprint.width
    length = ego_state.car_footprint.length

    if ego_state.dynamic_car_state.center_velocity_2d is None:
        v_x = float(velocity_fill)
        v_y = float(velocity_fill)
    else:
        v_x = float(ego_state.dynamic_car_state.center_velocity_2d.x)
        v_y = float(ego_state.dynamic_car_state.center_velocity_2d.y)

    vector = np.array(
        [
            float(center.x),  # x
            float(center.y),  # y
            float(center.heading),  # yaw [rad]
            v_x,  # v_x [m/s]
            v_y,  # v_y [m/s]
            float(width),  # width [m]
            float(length),  # length [m]
            1.0,  # constant
            0.0,  # constant
            0.0,  # constant
        ],
        dtype=np.float64,
    )
    # 안전 확인: (10,) 보장
    assert vector.shape == (10,), f"Expected shape (10,), got {vector.shape}"
    return vector


def observations_to_agents_buffer(
        observations_buffer: Deque[Observation],
        near_track_token_dist_order: List[str]) -> Deque[List[Agent]]:
    """Observation 버퍼에서 **차량(vehicles)** 만 추출해 Agent 리스트 버퍼로 변환한다.

    Args:
        observations_buffer (Deque[Observation]): 시간 순서로 정렬된 관측 버퍼.

    Returns:
        Deque[List[Agent]]: 각 관측에서 추출한 차량 Agent 리스트를 저장한 버퍼.

    Raises:
        TypeError: 관측이 ``DetectionsTracks`` 타입이 아닐 경우.
    """
    agents_buffer: Deque[List[Agent]] = deque(maxlen=observations_buffer.maxlen)
    for observation in observations_buffer:
        if isinstance(observation, DetectionsTracks):
            vehicles = [
                agent for agent in observation.tracked_objects.get_agents()
                if (agent.tracked_object_type in {
                    TrackedObjectType.VEHICLE,
                    TrackedObjectType.PEDESTRIAN,
                    TrackedObjectType.BICYCLE,
                }) and (agent.track_token in near_track_token_dist_order)
            ]
            agents_buffer.append(vehicles)
        else:
            raise TypeError("observations_buffer는 DetectionsTracks만 포함해야 합니다. "
                            f"받은 타입: {type(observation)}")
    return agents_buffer


def get_token_to_history(
        observation_buffer: Deque[Observation], iteration: SimulationIteration,
        near_track_token_dist_order: List[str]) -> Dict[str, Deque[Agent]]:
    """현재 시점에 존재하는 Agent들의 과거 기록을 생성한다.

    버퍼의 마지막 원소(현재 시점)에 존재하는 Agent들만을 대상으로 하며, 시간 역순(현재→과거)으로
    각 Agent의 상태를 수집한다. 특정 과거시점에 해당 Agent가 존재하지 않으면, 더 이상 과거 상태는
    수집하지 않는다.

    Args:
        agents_buffer (Deque[List[Agent]]): 시간 순서대로 정렬된 Agent 버퍼. 길이 :math:`T`의 버퍼이며,
            각 시점마다 임의 길이의 ``Agent`` 리스트를 포함한다.

    Returns:
        Dict[str, Deque[EgoState]]: 현재 시점에 존재하는 Agent 수 :math:`N` 만큼의 리스트를 반환한다. 각 내부
            리스트는 길이 :math:`T`이며, 시간 역순(현재→과거)으로 해당 Agent의 히스토리를 담고 있다.
    """
    # Deque[EgoState]
    # agents_buffer: 과거 -> 현재
    agents_buffer: Deque[List[Agent]] = observations_to_agents_buffer(
        observation_buffer, near_track_token_dist_order
    )  # self.observation_buffer: Deque[Observation]

    max_len = agents_buffer.maxlen
    if not agents_buffer:
        return {}
    current_vehicles: List[Agent] = [agent for agent in agents_buffer[-1]]
    current_token_list: List[str] = [
        agent.track_token for agent in current_vehicles
    ]
    current_token_to_agent_history: Dict[str, Deque] = {
        agent.track_token: deque(maxlen=max_len) for agent in current_vehicles
    }
    # 현재 시점 상태 추가
    for a_vehicle in current_vehicles:
        assert a_vehicle.metadata.timestamp_us == iteration.time_point.time_us, \
            f"현재 시점 관측의 timestamp_us ({a_vehicle.metadata.timestamp_us})가 " \
            f"iteration.time_point.time_us ({iteration.time_point.time_us})와 다른데, 그 차이는 " \
            f" {a_vehicle.metadata.timestamp_us - iteration.time_point.time_us} 입니다."
        current_token_to_agent_history[a_vehicle.track_token].append(a_vehicle)

    # max_len
    # list(agents_buffer) : List[List[Agent]] 과거 -> 현재
    # list(agents_buffer)[:-1] : List[List[Agent]] 과거 -> 현재-1
    # reversed_vehicle_buffer: List[List[Agent]] 현재-1 -> 과거
    reversed_vehicle_buffer = reversed(list(agents_buffer)[:-1])
    for past_vehicles in reversed_vehicle_buffer:
        # 가장 최근 -> 가장 오래된 순서로 과거 시점 상태 추가
        past_agent_lookup: Dict[str, Agent] = {
            past_agent.track_token: past_agent for past_agent in past_vehicles
        }
        for current_token in current_token_list:
            agent_history: Deque[Agent] = current_token_to_agent_history[
                current_token]
            if agent_history[-1] is None:
                continue  # 이미 이 토큰의 히스토리 수집 종료
            history_agent = past_agent_lookup.get(current_token, None)
            # 현재 -> 과거 순서로 추가
            current_token_to_agent_history[current_token].append(history_agent)
    # histories를 시간 순서(과거→현재)로 뒤집기
    for token, history in current_token_to_agent_history.items():
        # 이제 history는  과거 -> 현재가 되었따.
        history.reverse()
        # 만약 history 의 첫 원소가 None이면, 제거한다.
        if history[0] is None:
            history.popleft()
        # converted = deque((agent for agent in history), maxlen=history.maxlen)
        current_token_to_agent_history[token] = history

    return current_token_to_agent_history


def rotation_matrix(theta: float) -> np.ndarray:
    """주어진 θ로부터 2×2 회전 행렬 반환."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def ego_to_global(traj_ego: np.ndarray, ego_pos: np.ndarray,
                  ego_yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """
    traj_ego: (T,4) array of [x_ego, y_ego, cos_ego_yaw, sin_ego_yaw]
    ego_pos: (2,) global 위치
    ego_yaw: 스칼라 ego yaw
    returns:
      coords_global: (T,2) global x,y
      yaw_global:   (T,) global yaw
    """
    coords_ego = traj_ego[:, :2]  # (T,2)
    yaw_ego_frame = np.arctan2(traj_ego[:, 3], traj_ego[:, 2])  # (T,)
    R_e2g = rotation_matrix(ego_yaw)  # ego→global 회전
    coords_global = coords_ego.dot(R_e2g.T) + ego_pos  # (T,2)
    yaw_global = yaw_ego_frame + ego_yaw  # (T,)
    return coords_global, yaw_global


def global_to_local(coords_global: np.ndarray, yaw_global: np.ndarray,
                    veh_pos: np.ndarray, veh_yaw: float) -> np.ndarray:
    """
    coords_global: (T,2), yaw_global: (T,)
    veh_pos: (2,), veh_yaw: 스칼라
    returns:
      future_traj_wrt_npc_rear: (T,4) array of [x_local, y_local, cos_local_yaw, sin_local_yaw]
    """
    R_g2v = rotation_matrix(-veh_yaw)  # global→veh 회전
    delta = coords_global - veh_pos  # (T,2)
    coords_local = delta.dot(R_g2v.T)  # (T,2)
    yaw_local = yaw_global - veh_yaw  # (T,)

    cos_l = np.cos(yaw_local)[:, None]  # (T,1)
    sin_l = np.sin(yaw_local)[:, None]  # (T,1)
    return np.concatenate([coords_local, cos_l, sin_l], axis=1)


def transform_trajectory(future_traj_wrt_ego: np.ndarray,
                         ego_anchor_xy: np.ndarray, ego_yaw: float,
                         agent_global_xyyaw: np.ndarray) -> np.ndarray:
    """ego 기준 로컬 궤적을, 각 agent의 로컬 좌표계 궤적으로 변환합니다.

    중요한 점
    ----------
    여기서 ego_anchor_xy는
    - 예전 모델이면 ego rear axle 위치
    - 새 모델이면 ego center 위치
    를 넣어야 합니다.

    Args:
        future_traj_wrt_ego (np.ndarray):
            shape (T, 4) = [x_ego, y_ego, cos(yaw_rel), sin(yaw_rel)]
            ego 로컬 좌표계 기준 예측 궤적입니다.
        ego_anchor_xy (np.ndarray):
            shape (2,) = [x_global, y_global]
            ego 로컬 좌표계의 원점이 되는 글로벌 위치입니다.
        ego_yaw (float):
            ego 로컬 좌표계의 기준 yaw (global yaw) 입니다.
        agent_global_xyyaw (np.ndarray):
            shape (3,) = [x_global, y_global, yaw_global]
            대상 agent의 현재 글로벌 포즈입니다.

    Returns:
        np.ndarray:
            shape (T, 4) = [x_local, y_local, cos(local_yaw), sin(local_yaw)]
            agent 로컬 좌표계 기준 궤적입니다.
    """
    coords_g, yaw_g = ego_to_global(future_traj_wrt_ego, ego_anchor_xy, ego_yaw)
    veh_center_xy = agent_global_xyyaw[:2]
    veh_yaw = float(agent_global_xyyaw[2])
    return global_to_local(coords_g, yaw_g, veh_center_xy, veh_yaw)


def convert_center_to_rear_axle(traj_center: np.ndarray,
                                rear_wheelbase: float) -> np.ndarray:
    """
    차량 중심 기준 궤적을 뒷축 중심 기준 궤적으로 변환

    Args:
        traj_center: (T, 4) array of [x_center, y_center, cos_yaw, sin_yaw]
        rear_wheelbase:  (중심에서 뒷축까지의 거리)

    Returns:
        traj_rear_axle: (T, 4) array of [x_rear_axle, y_rear_axle, cos_yaw, sin_yaw]
    """
    # 각 시점에서 차량의 방향 벡터 (뒤쪽 방향)
    cos_yaw = traj_center[:, 2]  # (T,)
    sin_yaw = traj_center[:, 3]  # (T,)

    # 뒷축 방향으로의 오프셋 벡터 계산 (차량 좌표계에서 뒤쪽은 -x 방향)
    offset_x = -rear_wheelbase * cos_yaw  # (T,)
    offset_y = -rear_wheelbase * sin_yaw  # (T,)

    # 뒷축 중심 좌표 계산
    x_rear_axle = traj_center[:, 0] + offset_x  # (T,)
    y_rear_axle = traj_center[:, 1] + offset_y  # (T,)

    # 결과 조합 (yaw는 그대로 유지)
    traj_rear_axle = np.column_stack(
        [x_rear_axle, y_rear_axle, cos_yaw, sin_yaw])

    return traj_rear_axle


class WorldModelAgentsWEgo(AbstractMLAgents):
    """
    Simulate agents based on World model.
    """

    # nuplan/planning/script/config/simulation/planner/log_future_planner.yaml

    def __init__(
        self,
        model: TorchModuleWrapper,  #
        scenario: AbstractScenario,  #
        open_loop_detections_types: List[str],  #
        radius: float,  #
        target_velocity: float  # 안씀 TODO: 도로 속도가 없으면 입히는 로직을 추가해야하나?
    ) -> None:
        """
        Initializes the WorldModelAgents class.
        :param model: Model to use for inference.
            - 여기서는 nuplan_extent/planning/training/modeling/models/world_model.py
                - 의 WorldModel(TorchModuleWrapper) 객체를 기대한다.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.config = model.config
        self.predicted_neighbor_num = self.config.predicted_neighbor_num
        # ["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"]
        self._planner_step_gap_s = self._step_interval_us / 1e6  # [s]
        self.use_route_lanes = self.config.use_route_lanes
        self.use_ego_plan = self.config.use_ego_plan
        set_coord_as_center: bool = bool(
            getattr(self.config, "set_coord_as_center", False))
        self._ego_ref_point: Literal["rear_axle", "center"] = (
            "center" if set_coord_as_center else "rear_axle")
        self.updated_ego_state = None
        self.ego_trajectory = None

    def _get_ego_reference_se2(self, ego_state: EgoState) -> StateSE2:
        """현재 설정된 ego 기준점(rear axle 또는 center)의 포즈를 반환합니다.

        왜 필요한가
        ----------
        모델이 사용하는 ego 로컬 좌표계의 원점이
        - rear axle 인지
        - center 인지
        에 따라, ego 로컬→글로벌 변환에서 "평행이동(translation)"이 달라집니다.

        이 함수는 그 기준점을 한 곳에서 결정해 주기 위해 존재합니다.

        Args:
            ego_state (EgoState):
                현재 ego 상태입니다.

        Returns:
            StateSE2:
                ego 기준점의 SE2 포즈입니다.
                - self._ego_ref_point == "rear_axle" 이면 ego_state.rear_axle
                - self._ego_ref_point == "center"    이면 ego_state.center
        """
        if self._ego_ref_point == "center":
            return ego_state.center
        return ego_state.rear_axle

    def _get_ego_reference_global_xyyaw(
        self,
        ego_state: EgoState,
        *,
        dtype: np.dtype = np.float64,
    ) -> npt.NDArray[np.floating]:
        """ego 기준점의 글로벌 [x, y, yaw]를 numpy로 반환합니다.

        Args:
            ego_state (EgoState):
                현재 ego 상태입니다.
            dtype (np.dtype):
                반환 배열 dtype 입니다.

        Returns:
            np.ndarray:
                shape (3,) 의 배열입니다.
                [x_global, y_global, yaw_global]
        """
        ref: StateSE2 = self._get_ego_reference_se2(ego_state)
        out: npt.NDArray[np.floating] = np.array([ref.x, ref.y, ref.heading],
                                                 dtype=dtype)
        assert out.shape == (3,), f"Expected shape (3,), got {out.shape}"
        return out

    def _get_ego_reference_global_xy_and_yaw(
        self,
        ego_state: EgoState,
        *,
        dtype: np.dtype = np.float64,
    ) -> Tuple[npt.NDArray[np.floating], float]:
        """ego 기준점의 글로벌 위치(x,y)와 yaw를 분리해 반환합니다.

        Args:
            ego_state (EgoState):
                현재 ego 상태입니다.
            dtype (np.dtype):
                x,y 배열 dtype 입니다.

        Returns:
            Tuple[np.ndarray, float]:
                - xy: shape (2,) = [x_global, y_global]
                - yaw: float (rad)
        """
        ref: StateSE2 = self._get_ego_reference_se2(ego_state)
        xy: npt.NDArray[np.floating] = np.array([ref.x, ref.y],
                                                dtype=dtype)  # shape (2,)
        yaw: float = float(ref.heading)
        assert xy.shape == (2,), f"Expected shape (2,), got {xy.shape}"
        return xy, yaw

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        self.current_observation = None

        unique_agents: Dict[str, TrackedObject] = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type in {
                TrackedObjectType.VEHICLE,
                TrackedObjectType.PEDESTRIAN,
                TrackedObjectType.BICYCLE,
            }
        }
        dynamic_agents: Dict[str, TrackedObject] = sort_dict(unique_agents)
        self._agents: Dict[str, TrackedObject] = {
            **dynamic_agents,
        }

    # world_model_agents.py  ─ 클래스 내부 헬퍼(스켈레톤): 파라미터 프리셋
    def _build_smoother_config(self) -> SmootherConfig:
        """후처리 스무더 파라미터 번들을 구성합니다. (스켈레톤: 값은 예시/초기값으로 두고 나중에 조정)

        Returns:
            SmootherConfig: 에이전트 타입별 파라미터 모음.
        """
        veh_slip = SlipParams(beta_body_max_deg=10.0, w_p=0.1, w_theta=1.0)
        bic_slip = SlipParams(beta_body_max_deg=15.0, w_p=0.2, w_theta=1.0)
        ped_slip = SlipParams(beta_body_max_deg=18.0, w_p=1.0,
                              w_theta=0.5)  # 사용은 안하지만 형태 통일

        veh_rate = RateParams(v_dir=0.4,
                              v_floor=0.4,
                              phi_dot_cap=1.0,
                              R_min=6.0,
                              aN_max=4.0,
                              alpha_max=1.0,
                              straight_eps_deg=1.0)
        bic_rate = RateParams(v_dir=0.35,
                              v_floor=0.3,
                              phi_dot_cap=1.2,
                              R_min=4.0,
                              aN_max=2.5,
                              alpha_max=1.6,
                              straight_eps_deg=1.0)
        ped_rate = RateParams(v_dir=0.45,
                              v_floor=0.3,
                              phi_dot_cap=2.5,
                              R_min=None,
                              aN_max=3.0,
                              alpha_max=1.8,
                              straight_eps_deg=1.0)
        dt = float(self._planner_step_gap_s)  # 보통 0.1s

        return SmootherConfig(
            veh_slip=veh_slip,
            veh_rate=veh_rate,
            bic_slip=bic_slip,
            bic_rate=bic_rate,
            ped_slip=ped_slip,
            ped_rate=ped_rate,
            dt=dt,
        )

    def _compute_sorted_distances(
        self, ego_state: EgoState, token_to_agent: Dict[str, Agent]
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """ego와 각 agent 사이의 거리를 계산해 정렬된 결과를 반환한다.

        주의
        ----
        ego의 기준점이 rear axle인지 center인지에 따라,
        "ego와의 거리"가 아주 미세하게 달라질 수 있습니다.
        학습/추론 일관성을 위해 self._ego_ref_point를 그대로 사용합니다.

        Args:
            ego_state (EgoState): 기준이 되는 ego 상태.
            token_to_agent (Dict[str, Agent]): 거리를 계산할 agent 사전. 길이 = N.

        Returns:
            tuple[list[str], npt.NDArray[np.float64]]:
                - track token이 거리 오름차순으로 정렬된 리스트.
                - 정렬된 거리 배열로 shape (N,)이다.
        """
        if len(token_to_agent) == 0:
            return [], np.empty((0,), dtype=np.float64)

        tokens: list[str] = list(
            token_to_agent.keys())  # len(tokens) == len(distances)

        agents_xy: npt.NDArray[np.float32] = np.array(
            [token_to_agent[token].center.point.array for token in tokens],
            dtype=np.float32,
        )  # shape (N, 2)

        ego_ref: StateSE2 = self._get_ego_reference_se2(ego_state)
        ego_xy: npt.NDArray[np.float32] = np.expand_dims(
            ego_ref.point.array, axis=0).astype(np.float32)  # shape (1, 2)

        distances: npt.NDArray[np.float64] = cdist(
            ego_xy, agents_xy).flatten()  # shape (N,)
        sorted_indices: npt.NDArray[np.int64] = np.argsort(
            distances)  # shape (N,)

        sorted_tokens: list[str] = [tokens[i] for i in sorted_indices]
        sorted_distances: npt.NDArray[np.float64] = distances[sorted_indices]
        return sorted_tokens, sorted_distances

    def _get_interpol_time_points(
            self, iteration: SimulationIteration) -> List[TimePoint]:
        sim_step_gap_s: float = self.sim_step_gap_time_point.time_s
        """
        sim_step_gap_s : 0.15 (시뮬레이션 시간 간격 (초))
        self._planner_step_gap_s : 0.1 이면 (future trajectory의 점 사이 시간 간격)
            q = 1.5 -> interpol_num = 2
        [현실]
            sim_step_gap_s: 0.09992 self._planner_step_gap_s: 0.1 interpol_num: 1
        """
        q = Decimal(str(sim_step_gap_s)) / Decimal(str(
            self._planner_step_gap_s))
        interpol_num = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        interpol_num = max(interpol_num, 1)
        """
        if interpol_num = 2,
            interpol_indices = [1, 2]
            interpol_points_times = [0.1, 0.2]
            next_state_interpol_time_points = [TimePoint(current_time + 0.1s), TimePoint(current_time + 0.2s)]
        [현실] 
            interpol_indices: [1] 
            interpol_points_times: [0.1]
            next_state_interpol_time_points = [TimePoint(current_time + 0.1s)]
        """
        interpol_indices = np.linspace(0,
                                       interpol_num,
                                       num=interpol_num + 1,
                                       dtype=int)[1:]  # (interpol_num, )
        interpol_points_times = interpol_indices * self._planner_step_gap_s  # (interpol_num, )
        next_state_interpol_time_points: List[TimePoint] = []
        for interpol_time in interpol_points_times:
            time_point = TimePoint(time_us=int(iteration.time_point.time_us +
                                               interpol_time * 1e6))
            next_state_interpol_time_points.append(time_point)
        return next_state_interpol_time_points

    def _get_interp_next_ego_states(
        self, current_ego_state: EgoState, next_ego_state: EgoState,
        next_state_interpol_time_points: List[TimePoint]
    ) -> List[InterpolatableState]:
        """
        next_state_interpol_time_points = [TimePoint(current_time + 0.1s), ...]
        """
        states: List[EgoState] = [current_ego_state, next_ego_state]
        next_ego_trajectory = InterpolatedTrajectory(trajectory=states)
        interp_next_ego_states: List[
            InterpolatableState] = next_ego_trajectory.get_state_at_times(
                next_state_interpol_time_points)
        return interp_next_ego_states

    def _interpolated_state_to_local_np(
            self, interp_next_ego_state: List[InterpolatableState],
            current_ego_state: EgoState) -> npt.NDArray[np.float32]:
        """ego 미래 상태들을 diffusion planner 입력 배열로 변환한다.

        Returns:
            npt.NDArray[np.float32]:
                shape (interpol_num, 11)
                [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
                 width, length, 1, 0, 0]
        """
        interpol_num = len(interp_next_ego_state)
        interpol_abs_next_ego: npt.NDArray[np.float64] = np.zeros(
            (interpol_num, 10), dtype=np.float64)  # shape (T, 10)
        interpol_abs_next_ego[:, 7] = 1  # is vehicle

        for i, state in enumerate(interp_next_ego_state):
            interpol_abs_next_ego[i, 0] = state.center.x
            interpol_abs_next_ego[i, 1] = state.center.y
            interpol_abs_next_ego[i, 2] = state.center.heading

            # EgoState 속도는 ego 좌표계 기준이므로 world로 회전 변환
            v_local = state.dynamic_car_state.center_velocity_2d
            he = float(state.center.heading)
            c, s = np.cos(he), np.sin(he)
            vx_w = c * float(v_local.x) - s * float(v_local.y)
            vy_w = s * float(v_local.x) + c * float(v_local.y)
            interpol_abs_next_ego[i, 3] = vx_w
            interpol_abs_next_ego[i, 4] = vy_w

            interpol_abs_next_ego[i, 5] = state.car_footprint.width
            interpol_abs_next_ego[i, 6] = state.car_footprint.length

        # ✅ [변경] anchor를 rear axle 고정이 아니라 "현재 설정된 ego 기준점"으로 사용
        anchor: npt.NDArray[np.floating] = self._get_ego_reference_global_xyyaw(
            current_ego_state, dtype=np.float32)  # shape (3,)

        interp_next_ego_11_dim = convert_absolute_quantities_to_relative(
            interpol_abs_next_ego, anchor, 'ego')
        return interp_next_ego_11_dim.astype(np.float32)

    def _from_ego_fut_traj_to_np(
            self, ego_future_trajectory: InterpolatedTrajectory,
            current_ego_state: EgoState) -> Optional[npt.NDArray[np.float32]]:
        """미래 ego 궤적을 diffusion planner 입력 배열로 변환한다."""
        planner_future_11_dim = None
        if ego_future_trajectory is not None:
            future_states: List[EgoState] = list(
                ego_future_trajectory.get_sampled_trajectory())
            future_states = future_states[1:]  # remove current
            valid_future_len = len(future_states)
            assert valid_future_len <= self.config.future_len, \
                f"미래 상태 개수({valid_future_len})가 config.future_len({self.config.future_len})보다 큽니다."

            global_ego_fut_traj_10: npt.NDArray[np.float64] = np.zeros(
                (self.config.future_len, 10),
                dtype=np.float64)  # shape (future_len, 10)

            for i, state in enumerate(future_states):
                global_ego_fut_traj_10[i, 0] = state.center.x
                global_ego_fut_traj_10[i, 1] = state.center.y
                global_ego_fut_traj_10[i, 2] = state.center.heading

                v_local = state.dynamic_car_state.center_velocity_2d
                he = float(state.center.heading)
                c, s = np.cos(he), np.sin(he)
                vx_w = c * float(v_local.x) - s * float(v_local.y)
                vy_w = s * float(v_local.x) + c * float(v_local.y)
                global_ego_fut_traj_10[i, 3] = vx_w
                global_ego_fut_traj_10[i, 4] = vy_w

                global_ego_fut_traj_10[i, 5] = state.car_footprint.width
                global_ego_fut_traj_10[i, 6] = state.car_footprint.length
                global_ego_fut_traj_10[i, 7] = 1  # is vehicle

            # ✅ [변경] anchor를 rear axle 고정이 아니라 "현재 설정된 ego 기준점"으로 사용
            anchor: npt.NDArray[
                np.floating] = self._get_ego_reference_global_xyyaw(
                    current_ego_state, dtype=np.float32)  # shape (3,)

            planner_future_11_dim = convert_absolute_quantities_to_relative(
                global_ego_fut_traj_10, anchor, 'ego')  # shape (future_len, 11)

            planner_future_11_dim[valid_future_len:, :] = 0.0
            planner_future_11_dim = planner_future_11_dim.astype(np.float32)
            if not self.use_ego_plan:
                # x, y, cos, sin, vx, vy, width, length, is_vehicle, 0, 0
                planner_future_11_dim[:, :7] = 0.0

        return planner_future_11_dim

    def set_vis_features(self, is_vis_features: bool, vis_features_path: str):
        """
        Set params for saving features, for visualization, only when simulation feature video callback is on.
        :param is_vis_features: whether to save features
        :param vis_features_path: path to save features
        """
        self._is_vis_features = is_vis_features
        self._vis_features_path = vis_features_path

    def from_next_ego_state_to_traj_np(
        self,
        iteration: SimulationIteration,
        next_ego_state: Optional[EgoState] = None,
    ) -> Optional[np.ndarray]:
        interp_next_ego_11_dim = None
        if next_ego_state is not None:
            next_state_interpol_time_points: List[
                TimePoint] = self._get_interpol_time_points(iteration)
            # interp_next_ego_state: List[EgoState]
            interp_next_ego_state: List[
                InterpolatableState] = self._get_interp_next_ego_states(
                    self._ego_anchor_state, next_ego_state,
                    next_state_interpol_time_points)
            # (interpol_num, 11)
            interp_next_ego_11_dim = self._interpolated_state_to_local_np(
                interp_next_ego_state, self._ego_anchor_state)
        return interp_next_ego_11_dim

    def _get_model_input(
        self, iteration: SimulationIteration, history: SimulationHistoryBuffer,
        interp_next_ego_11_dim: Optional[npt.NDArray[np.float32]],
        planner_future_11_dim: Optional[npt.NDArray[np.float32]]
    ) -> Tuple[Dict[str, AbstractModelFeature], List[str], Dict[
            str, np.ndarray], np.ndarray, np.ndarray]:
        initialization = HorizonPlannerInitialization(
            # 시나리오가 끝나고도 계속 진행했을 때 최종적으로 도달해야 하는 포즈 (존재하지 않을 수도 있음)
            mission_goal=self._scenario.get_mission_goal(),
            # (x, y, yaw) 의 StateSE2
            map_api=self._scenario.map_api,
            scenario=self._scenario,
            # 전문 운전자(ground truth)의 실제 마지막 상태 (항상 존재)
            expert_goal_state=self._scenario.get_expert_goal_state(),
            use_route_lanes=self.use_route_lanes
            # (x, y, yaw) 의 StateSE2
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            iteration.index)
        # target_agents_mask: np.ndarray, (max_agent_num,) bool
        # diffusion_agents_tokens: List[str] # len: valid diffusion agent num
        diffusion_agents_tokens = None
        current_input = PlannerInput(iteration, history, traffic_light_data,
                                     diffusion_agents_tokens,
                                     interp_next_ego_11_dim,
                                     planner_future_11_dim)
        # WorldModelFeatureBuilder.get_features_from_simulation
        # from nuplan_extent/planning/training/preprocessing/feature_builders/world_model_feature_builder.py
        """
to_feature_tensor: numpy → torch.Tensor (shape 그대로)
to_device: 텐서를 GPU/CPU 디바이스로 이동 (shape 그대로)
collate([feature]): 배치 차원 B=1 추가 → (…, …) → (1, …, …)
        """
        """

        """
        model_input_key_to_value: Dict[
            str, AbstractModelFeature] = self._model_loader.build_features(
                current_input, initialization)
        model_input_key_to_unnorm_value: Dict[
            str, np.ndarray] = self._model_loader.feature_builders[
                0].unnormalized_features
        # near_track_token: len = "Pnn 이하의 길이"
        near_track_token_dist_order: List[
            str] = model_input_key_to_unnorm_value[
                "near_track_token"]  # ("Pnn 이하의 길이", )
        diff_token_to_future_all_gt_3_dim: Dict[
            str, np.ndarray] = model_input_key_to_unnorm_value[
                "diff_token_to_future_all_gt_3_dim"]  # Dict[str, np.ndarray] # len : valid_agent_num
        neighbor_agents_past = model_input_key_to_unnorm_value[
            "neighbor_agents_past"]  # (N, T, 11)
        neighbor_agents_current = neighbor_agents_past[:, -1, :]  # (N, 11)
        ego_agent_past = model_input_key_to_unnorm_value[
            "ego_agent_past"]  # (T, 11)
        ego_agent_current = ego_agent_past[-1, :]  # (11, )
        self._draw_infos.model_input_key_to_unnorm_value = model_input_key_to_unnorm_value
        return (model_input_key_to_value, near_track_token_dist_order,
                diff_token_to_future_all_gt_3_dim, neighbor_agents_current,
                ego_agent_current)

    def _update_diffusion_agents_observation(
            self, iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            next_ego_state: Optional[EgoState],
            ego_future_trajectory: Optional[InterpolatedTrajectory]) -> None:
        # (interpol_num, 11)
        interp_next_ego_11_dim: Optional[
            np.ndarray] = self.from_next_ego_state_to_traj_np(
                iteration, next_ego_state)
        # (future_len, 11)
        planner_future_11_dim: Optional[
            np.ndarray] = self._from_ego_fut_traj_to_np(ego_future_trajectory,
                                                        self._ego_anchor_state)

        # model_input_key_to_value: Dict[str, AbstractModelFeature]
        # near_track_token_dist_order: List[str] # len = "Pnn 이하의 길이"
        (model_input_key_to_value, near_track_token_dist_order,
         diff_token_to_future_all_gt_3_dim, neighbor_agents_current,
         ego_agent_current) = self._get_model_input(iteration, history,
                                                    interp_next_ego_11_dim,
                                                    planner_future_11_dim)

        # Infer model
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)
        self.infer_model(model_input_key_to_value, iteration, next_iteration,
                         near_track_token_dist_order,
                         diff_token_to_future_all_gt_3_dim,
                         neighbor_agents_current, ego_agent_current)

    def update_observation(
            self,
            iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            next_ego_state: Optional[EgoState] = None,
            ego_future_trajectory: Optional[InterpolatedTrajectory] = None
    ) -> None:
        self._draw_infos = draw_machine.DrawInfos()
        self.ego_state_buffer: Deque[EgoState] = history.ego_state_buffer
        self.observation_buffer: Deque[Observation] = history.observation_buffer
        # EgoState, Observation
        self._ego_anchor_state, self.current_observation = history.current_state
        self.sim_step_gap_time_point: TimePoint = next_iteration.time_point - iteration.time_point

        self._update_diffusion_agents_observation(iteration, next_iteration,
                                                  history, next_ego_state,
                                                  ego_future_trajectory)
        if self._is_vis_features:
            input_data, output_data = self._draw_infos.to_dict()
            draw_machine.draw_world_model_to_png(input_data, output_data,
                                                 self._vis_features_path)

    @staticmethod
    def extract_rear_wheelbases(agents: Dict[str, Agent],
                                track_tokens: List[str]) -> List[float]:
        """주어진 토큰 순서대로 차량의 뒷축까지 거리를 추출한다.

        Args:
            agents (Dict[str, Agent]): shape (M,) track token을 키로 하는 agent 사전.
            track_tokens (List[str]): shape (N,) 뒷축 거리를 추출할 track token 목록.

        Returns:
            List[float]: shape (N,) 각 차량의 중심에서 뒷축까지 거리 [m].

        Raises:
            KeyError: 주어진 track token에 해당하는 agent가 없을 때 발생.
        """
        rear_wheelbases: List[float] = []
        for token in track_tokens:
            agent = agents[token]
            box = agent.box
            assert isinstance(box, CarFootprint)
            rear_wheelbases.append(float(box.rear_axle_to_center_dist))
        return rear_wheelbases

    def get_rear_wheelbases(
            self, current_token_to_agent: Dict[str, Agent],
            diffusion_agents_track_tokens: List[str]) -> Dict[str, float]:
        """각 차량의 중심에서 뒷축까지 거리를 계산한다.
        Args:
            current_token_to_agent (List[Agent]): shape (M,) 현재 프레임에서 관측된 agent 목록.
            diffusion_agents_track_tokens (List[str]): shape (N,) wheelbase를 추출할 track token.
        Returns:
            List[float]: shape (N,) 각 차량의 뒷축까지 거리 [m].

        """

        token_to_rear_wheelbase: Dict[str, float] = {}
        for token in diffusion_agents_track_tokens:
            agent = current_token_to_agent[token]
            box = agent.box
            assert isinstance(box, CarFootprint)
            token_to_rear_wheelbase[token] = float(box.rear_axle_to_center_dist)
        return token_to_rear_wheelbase

    def outputs_to_ego_trajectory(
        self,
        ego_future_traj_wrt_ego: np.ndarray,  # (T, 4)
        ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:
        heading = np.arctan2(ego_future_traj_wrt_ego[:, 3],
                             ego_future_traj_wrt_ego[:, 2])[..., None]
        ego_future_traj_wrt_ego = np.concatenate(
            [ego_future_traj_wrt_ego[..., :2], heading], axis=-1)  # (T, 3)

        states = transform_ego_predictions_to_states(
            ego_future_traj_wrt_ego,  # (T, 3)
            ego_state_history,
            self._future_horizon,
            self._planner_step_gap_s,
            set_coord_as_center=self.config.set_coord_as_center,
        )
        return states

    def outputs_to_trajectory(
            self,
            future_traj_wrt_npc_center: np.ndarray,  # (T, 4)
            ego_state_history: Deque[Agent],
            sim_step_gap_s: float) -> List[InterpolatableState]:
        heading = np.arctan2(future_traj_wrt_npc_center[:, 3],
                             future_traj_wrt_npc_center[:, 2])[..., None]
        future_traj_wrt_npc_center = np.concatenate(
            [future_traj_wrt_npc_center[..., :2], heading], axis=-1)  # (T, 3)

        waypoints = transform_predictions_to_states(
            future_traj_wrt_npc_center,  # (T, 3)
            ego_state_history,
            self._future_horizon,
            self._planner_step_gap_s,
            sim_step_gap_s)
        return waypoints

    def get_rear_axle_pose(
            self, current_ego_state: EgoState) -> Tuple[np.ndarray, float]:
        """차량 뒷축의 위치와 방향을 계산한다.

        Returns:
            Tuple[np.ndarray, float]: shape (2,), 차량 뒷축 [x, y]와 방향 yaw_rad.

        """
        rear_axle = current_ego_state.car_footprint.rear_axle
        position: np.ndarray = np.array([
            rear_axle.x,
            rear_axle.y,
        ],
                                        dtype=float)  # shape: (2,)
        return position, rear_axle.heading,

    def _get_diff_token_to_np_history_to_draw(
            self,
            diffusion_token_to_agent_history: Dict[str, Deque[Agent]],
            cur_ego_global_xyyaw: np.ndarray,  # shape: (3,)
    ):
        diff_token_to_np_history_wrt_ego: Dict[str, np.ndarray] = {}
        for token, history in diffusion_token_to_agent_history.items():
            history_array = [
                agent_to_feature_vector(agent) for agent in history
            ]
            history_array = np.stack(history_array,
                                     axis=0)  # (len(history), 10)
            # convert absolute to relative

            history_array = convert_absolute_quantities_to_relative(
                history_array, cur_ego_global_xyyaw)
            diff_token_to_np_history_wrt_ego[
                token] = history_array  # (len(history), 11)
        self._draw_infos.diff_token_to_np_history_wrt_ego = diff_token_to_np_history_wrt_ego

    def _get_diff_token_to_cur_xyyaw(
            self,
            near_track_token_dist_order: List[str],  # # len == "Pnn 이하의 길이",
    ) -> Dict[str, np.ndarray]:
        current_agents: List[
            Agent] = self.current_observation.tracked_objects.get_agents()
        # current_agents: 쓰임
        # np.ndarray: (3,)
        diff_token_to_global_xyyaw: Dict[str, np.ndarray] = {
            agent.track_token:
                np.array([agent.center.x, agent.center.y, agent.center.heading])
            for agent in current_agents
            if agent.track_token is not None and
            agent.track_token in near_track_token_dist_order
        }  # shape (3,)
        return diff_token_to_global_xyyaw

    def get_diff_token_to_interpol_traj(
            self,
            diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray],  # (T, 4)
            diff_token_to_global_xyyaw: Dict[str, np.ndarray],  # (token, (3,))
            diffusion_token_to_agent_history: Dict[
                str, Deque[Agent]],  # (token, Deque[Agent])
            cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
    ) -> Dict[str, AbstractTrajectory]:
        sim_step_gap_s: float = self.sim_step_gap_time_point.time_s

        # ✅ [변경] ego 기준점(원점)을 rear axle 고정이 아니라, 설정값에 따름
        ego_anchor_xy, ego_yaw = self._get_ego_reference_global_xy_and_yaw(
            self._ego_anchor_state,
            dtype=np.float64)  # ego_anchor_xy: (2,), ego_yaw: float

        ########## TO DRAW ##########
        diff_token_to_interp_np_traj_wrt_ego: Dict[str,
                                                   np.ndarray] = {}  # (1+T, 11)
        #############################
        diff_token_to_interpol_traj: Dict[str, AbstractTrajectory] = {}
        # (T, 4) # 길이: Pnn 중, 실제로 궤적 생성한 대상들만.
        for token, future_traj_wrt_ego in diff_token_to_np_gen_traj_wrt_ego.items(
        ):
            # future_traj_wrt_ego: (T, 4)
            # future_traj_wrt_ego 값이 전부 0. 이면 무시
            if np.allclose(future_traj_wrt_ego, 0.0):
                raise ValueError(
                    f"future_traj_wrt_ego for token {token} is all zeros.")
            agent_global_xyyaw = diff_token_to_global_xyyaw[token]
            # ✅ 여기서 ego_anchor_xy가 center인지 rear axle인지가 "글로벌 변환의 평행이동"을 결정
            # future_traj_wrt_npc_center: (T, 4)
            future_traj_wrt_npc_center: np.ndarray = transform_trajectory(
                future_traj_wrt_ego,
                ego_anchor_xy,
                ego_yaw,
                agent_global_xyyaw,
            )  # (T, 4)
            self_history: Deque[Agent] = diffusion_token_to_agent_history[token]
            future_trajectory = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(
                    future_traj_wrt_npc_center, self_history, sim_step_gap_s))
            diff_token_to_interpol_traj[token] = future_trajectory
            ########## TO DRAW ##########
            diff_token_to_interp_np_traj_wrt_ego[
                token] = self._get_rel_future_arrays_to_draw(
                    future_trajectory, cur_ego_global_xyyaw)
            #############################
        self._draw_infos.diff_token_to_interp_np_traj_wrt_ego = diff_token_to_interp_np_traj_wrt_ego
        return diff_token_to_interpol_traj

    def _get_rel_future_arrays_to_draw(
            self,
            future_trajectory: InterpolatedTrajectory,
            cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
    ):
        future_waypoints: List[
            Waypoint] = future_trajectory.get_sampled_trajectory()
        global_future_arrays = [
            waypoint_to_numpy10(wp) for wp in future_waypoints
        ]  # List[(10,)]
        global_future_arrays = np.stack(global_future_arrays,
                                        axis=0)  # (1+T, 10)
        # rel_future_arrays: (1+T, 11)
        rel_future_arrays = convert_absolute_quantities_to_relative(
            global_future_arrays,
            cur_ego_global_xyyaw)  # cur_ego_global_xyyaw: (3,)
        return rel_future_arrays

    def from_np_to_waypoint_list(
        self,
        a_near_future_all_gt_3_dim: Optional[np.ndarray]  # (future_all_len, 3)
    ) -> List[Waypoint]:
        a_near_future_all_waypoints: List[Waypoint] = []
        if a_near_future_all_gt_3_dim is not None:
            for t in range(a_near_future_all_gt_3_dim.shape[0]):
                state = a_near_future_all_gt_3_dim[t, :]  # (3,)
                waypoint = Waypoint(time_point=TimePoint(time_us=0),
                                    oriented_box=OrientedBox(
                                        center=StateSE2(x=float(state[0]),
                                                        y=float(state[1]),
                                                        heading=float(
                                                            state[2])),
                                        length=0,
                                        width=0,
                                        height=0,
                                    ))
                a_near_future_all_waypoints.append(waypoint)
        return a_near_future_all_waypoints

    def _update_diffusion_agents(
        self,
        diff_token_to_interpol_traj: Dict[str, AbstractTrajectory],
        next_iteration: SimulationIteration,
        cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
        diff_token_to_future_all_gt_3_dim: Dict[
            str, np.ndarray]  # len : valid_agent_num
    ) -> Optional[EgoState]:
        if self.ego_trajectory is None:
            updated_ego_state = None
        else:
            # EgoState의 속도는 자차 좌표계 기준 벡터
            # updated_ego_state: EgoState
            updated_ego_state = self.ego_trajectory.get_state_at_time(
                next_iteration.time_point)
            self._draw_infos.ego_next_wp_wrt_ego = self._get_new_local_ego_state_array_to_draw(
                updated_ego_state, cur_ego_global_xyyaw)  # (11)

        # [NEW] 1) GT(ego frame) → Global frame 변환
        gt_global: Dict[str, np.ndarray] = {}
        for token, local_traj_xyh in diff_token_to_future_all_gt_3_dim.items():
            gt_global[token] = ego_local_traj3_to_global(
                local_traj_xyh=local_traj_xyh,  # (T,3) in ego
                cur_ego_global_xyyaw=cur_ego_global_xyyaw,  # (3,) global
            )  # (T,3) in global
        diff_token_to_future_all_gt_3_dim = gt_global

        diff_token_to_updated_agent: Dict[str, Agent] = {}
        diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] = {}  # (1, 11)
        for diff_token, interpol_traj in diff_token_to_interpol_traj.items():

            agent_ = self._agents[diff_token]
            new_timestamp_us = next_iteration.time_point.time_us
            new_metadata = SceneObjectMetadata(new_timestamp_us,
                                               agent_.metadata.token,
                                               agent_.metadata.track_id,
                                               agent_.metadata.track_token,
                                               agent_.metadata.category_name)

            # EgoState의 속도는 자차 좌표계 기준 벡터
            updated_waypoint: Waypoint = interpol_traj.get_state_at_time(
                next_iteration.time_point)
            updated_agent = Agent(
                tracked_object_type=agent_.tracked_object_type,
                oriented_box=updated_waypoint.oriented_box,
                velocity=updated_waypoint.velocity,
                metadata=new_metadata,
            )
            a_near_future_all_gt_3_dim = diff_token_to_future_all_gt_3_dim[
                diff_token]  # (future_all_len, 3)
            updated_agent.predictions = [
                # GT 궤적
                PredictedTrajectory(probability=0.5,
                                    waypoints=self.from_np_to_waypoint_list(
                                        a_near_future_all_gt_3_dim)),
                # 모델 생성 궤적
                PredictedTrajectory(
                    probability=0.5,
                    waypoints=interpol_traj.get_sampled_trajectory())
            ]
            diff_token_to_updated_agent[diff_token] = updated_agent
            ########## TO DRAW ##########
            diff_token_to_next_wp_wrt_ego[
                diff_token] = self._get_new_local_waypoint_array_to_draw(
                    updated_waypoint, cur_ego_global_xyyaw)  # (11)
            #############################
        self._agents = diff_token_to_updated_agent
        self._draw_infos.diff_token_to_next_wp_wrt_ego = diff_token_to_next_wp_wrt_ego
        return updated_ego_state

    def _get_new_local_ego_state_array_to_draw(
            self,
            updated_ego_state: EgoState,
            cur_ego_global_xyyaw: np.ndarray  # shape (3,)
    ):
        new_ego_state_array = ego_state_to_numpy10(updated_ego_state).reshape(
            1, -1)  # (1, 10)
        # new_local_ego_state_array: (1, 11)
        new_local_ego_state_array = convert_absolute_quantities_to_relative(
            new_ego_state_array,
            cur_ego_global_xyyaw)  # cur_ego_global_xyyaw: (3,)
        # (1, 11) -> (11,)
        new_local_ego_state_array = new_local_ego_state_array.reshape(-1)
        return new_local_ego_state_array

    def _get_new_local_waypoint_array_to_draw(
            self,
            updated_waypoint: Waypoint,
            cur_ego_global_xyyaw: np.ndarray  # shape (3,)
    ):
        new_waypoint_array = waypoint_to_numpy10(updated_waypoint).reshape(
            1, -1)  # (1, 10)
        # new_local_waypoint_array: (1, 11)
        new_local_waypoint_array = convert_absolute_quantities_to_relative(
            new_waypoint_array,
            cur_ego_global_xyyaw)  # cur_ego_global_xyyaw: (3,)
        # (1, 11) -> (11,)
        new_local_waypoint_array = new_local_waypoint_array.reshape(-1)
        return new_local_waypoint_array

    def _filter_trajectory(
        self,
        future_np_trajs_wrt_ego: np.ndarray,
        neighbor_agents_past: np.ndarray,  # (Pnn,time_len, 11)
        veh_valid_mask: np.ndarray,  # (Pnn,)
        bic_valid_mask: np.ndarray,  # (Pnn,)
        ped_valid_mask: np.ndarray  # (Pnn,)
    ) -> npt.NDArray[np.float32]:
        # 0) 원시 입력 준비
        near_cur_future_raw: np.ndarray = future_np_trajs_wrt_ego.astype(
            np.float32)  # (Pnn, 81, 4)
        near_agents_current: np.ndarray = near_cur_future_raw[:,
                                                              0, :]  # (Pnn, 4)
        near_future_raw: np.ndarray = near_cur_future_raw[:,
                                                          1:, :]  # (Pnn, 80, 4)

        # 1) σ_k 계산 (현재+미래 81프레임)
        near_current_future_dir = compute_forward_backward_signs(
            near_cur_future_raw=near_cur_future_raw)  # (Pnn, 81) bool

        # 2) 슬립각 제한 단계 (앵커 보정→순차 보정)
        cfg = self._build_smoother_config()  # 아래 헬퍼를 스켈레톤으로 추가
        near_current_future_a2, near_future_body_slip = slip_limit_stage(
            near_agents_current=near_agents_current,
            near_future_raw=near_future_raw,
            near_current_future_dir=near_current_future_dir,
            veh_valid_mask=veh_valid_mask,
            bic_valid_mask=bic_valid_mask,
            ped_valid_mask=ped_valid_mask,
            cfg=cfg,
        )  # (Pnn,81,4), (Pnn,80)
        near_future_a3 = near_current_future_a2[:, 1:, :]  # (Pnn, 80, 4)
        # 3) 각속도 제약 + 스무딩 단계
        # near_future_a3 = yawrate_smooth_stage(
        #     near_current_future_a2=near_current_future_a2,
        #     near_current_future_dir=near_current_future_dir,
        #     near_future_body_slip=near_future_body_slip, # (Pnn,81)
        #     veh_valid_mask=veh_valid_mask,
        #     bic_valid_mask=bic_valid_mask,
        #     ped_valid_mask=ped_valid_mask,
        #     cfg=cfg,
        # )  # (Pnn, 80, 4)
        return near_current_future_a2, near_future_a3

    def _get_token_to_np_traj_wrt_ego(
            self,
            model_inputs: AbstractModelFeature,
            near_track_token_dist_order: List[str],  # len == "Pnn 이하의 길이"
            neighbor_agents_current: np.ndarray,  # (Pnn, 11)
            ego_agent_current: np.ndarray,  # (11,)
    ) -> Tuple[Dict[str, np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            diff_token_to_np_gen_traj_wrt_ego
                Dict[str, np.ndarray]: (token, (T, 4)) # ego 좌표계 기준 차량 중심의 값
                # len == valid diffusion agent num

        """
        """
        # nuplan_extent/planning/training/modeling/models/world_model.py
        # WorldModel.forward
        # future_np_trajs_wrt_ego: ((1+)Pnn, 1+T, 4)
        # ego와 거리 순으로 모든 agent가 들어있다는 가정!!! (생성 안했으면, 빈 값을 준다.)
        """
        (future_np_trajs_wrt_ego,
         future_np_int_trajs_wrt_ego) = self._model_loader.infer(model_inputs)
        future_np_trajs_wrt_ego: np.ndarray = future_np_trajs_wrt_ego.detach(
        ).cpu().numpy()  # ((1+)Pnn, 1+T, 4)
        future_np_int_trajs_wrt_ego: np.ndarray = future_np_int_trajs_wrt_ego.detach(
        ).cpu().numpy()  # ((1+)Pnn, 1+T, 4)
        if self.config.do_ego_predict:
            ego_np_traj_wrt_ego = future_np_trajs_wrt_ego[0, 1:, :]  # (T, 4)
            ego_np_int_traj_wrt_ego = future_np_int_trajs_wrt_ego[
                0, 1:, :]  # (T, 4)
            # ego_agent_current: (11)
            ego_np_traj_11_wrt_ego = np.tile(
                ego_agent_current,
                (future_np_trajs_wrt_ego.shape[1], 1))  # (1+T, 11)
            ego_np_traj_11_wrt_ego[1:, :4] = ego_np_traj_wrt_ego  # (1+T, 11)

            ego_np_int_traj_11_wrt_ego = np.tile(
                ego_agent_current,
                (future_np_int_trajs_wrt_ego.shape[1], 1))  # (1+T, 11)
            ego_np_int_traj_11_wrt_ego[1: , :4] = ego_np_int_traj_wrt_ego  # (1+T, 11)

            # 첫 번째 궤적은 ego 궤적이므로 제외
            future_np_trajs_wrt_ego = future_np_trajs_wrt_ego[
                1:, :, :]  # (Pnn, 1+T, 4)
            future_np_int_trajs_wrt_ego = future_np_int_trajs_wrt_ego[
                1:, :, :]  # (Pnn, 1+T, 4)
            self._draw_infos.ego_np_traj_11_wrt_ego = ego_np_traj_11_wrt_ego  # (1+T, 11) TODO: 속도는 잘못된 값이 들어가 있음.
            self._draw_infos.ego_np_int_traj_11_wrt_ego = ego_np_int_traj_11_wrt_ego  # (1+T, 11) TODO: 속도는 잘못된 값이 들어가 있음.

        else:
            ego_np_traj_wrt_ego = None
            ego_np_int_traj_wrt_ego = None

        gen_npc_slot_len = future_np_trajs_wrt_ego.shape[0]
        # neighbor_agents_current: (Pnn, 11)
        # (T, 4) # 길이: Pnn 중, 실제로 궤적 생성한 대상들만.
        diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] = {}
        diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, np.ndarray] = {}

        diff_token_to_np_int_traj_wrt_ego: Dict[str, np.ndarray] = {}
        diff_token_to_np_int_traj_11_wrt_ego: Dict[str, np.ndarray] = {}

        for idx, token in enumerate(
                near_track_token_dist_order):  #  # len == "Pnn 이하의 길이"
            neighbor_agent_current = neighbor_agents_current[idx]  # (11)
            # neighbor_agent_current: (11) -> (T, 11)
            np_gen_traj_11_wrt_ego = np.tile(
                neighbor_agent_current,
                (future_np_trajs_wrt_ego.shape[1], 1))  # (1+T, 11)
            np_gen_int_traj_11_wrt_ego = np.tile(
                neighbor_agent_current,
                (future_np_int_trajs_wrt_ego.shape[1], 1))  # (1+T, 11)
            if idx >= gen_npc_slot_len:
                break
            future_np_traj_wrt_ego = future_np_trajs_wrt_ego[
                idx, :, :]  # (1+T, 4)
            np_gen_traj_11_wrt_ego[:, :4] = future_np_traj_wrt_ego  # (1+T, 11)
            future_np_int_traj_wrt_ego = future_np_int_trajs_wrt_ego[
                idx, :, :]  # (1+T, 4)
            np_gen_int_traj_11_wrt_ego[:, :
                                       4] = future_np_int_traj_wrt_ego  # (1+T, 11)

            np_traj_sum = future_np_traj_wrt_ego.sum()  # (1+T, 4) 의 합
            np_int_traj_sum = future_np_int_traj_wrt_ego.sum()  # (1+T, 4) 의 합
            if np.allclose(np_traj_sum, 0.0):
                raise ValueError(f"{idx} 번째 대상의 생성 궤적이 모두 0입니다.")
            if np.allclose(np_int_traj_sum, 0.0):
                raise ValueError(f"{idx} 번째 대상의 통합 궤적이 모두 0입니다.")

            diff_token_to_np_gen_traj_wrt_ego[token] = future_np_traj_wrt_ego[
                1:, :]  # (T, 4)
            diff_token_to_np_gen_traj_11_wrt_ego[
                token] = np_gen_traj_11_wrt_ego  # (1+T, 11) # TODO: 속도는 잘못된 값이 들어가 있음.

            diff_token_to_np_int_traj_wrt_ego[
                token] = future_np_int_traj_wrt_ego[1:, :]  # (T, 4)
            diff_token_to_np_int_traj_11_wrt_ego[
                token] = np_gen_int_traj_11_wrt_ego  # (1+T, 11) # TODO: 속도는 잘못된 값이 들어가 있음.

        ### 디버깅용 ###
        self._draw_infos.diff_token_to_np_gen_traj_11_wrt_ego = diff_token_to_np_gen_traj_11_wrt_ego  # (1+T, 11)
        self._draw_infos.diff_token_to_np_int_traj_11_wrt_ego = diff_token_to_np_int_traj_11_wrt_ego  # (1+T, 11) TODO: 속도는 잘못된 값이 들어가 있음.

        if self.config.use_integration_trajectory:
            return diff_token_to_np_int_traj_wrt_ego, ego_np_int_traj_wrt_ego
        return diff_token_to_np_gen_traj_wrt_ego, ego_np_traj_wrt_ego

    def infer_model(
            self,
            model_input_key_to_value: Dict[str, AbstractModelFeature],
            iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            near_track_token_dist_order: List[str],  # len == "Pnn 이하의 길이",
            diff_token_to_future_all_gt_3_dim: Dict[
                str, np.ndarray],  # len : valid_agent_num,
            neighbor_agents_current: np.ndarray,  # (agents_num, 11)
            ego_agent_current: np.ndarray,  # (11)
    ) -> None:
        self.updated_ego_state = None
        self.ego_trajectory = None
        model_inputs: AbstractModelFeature = model_input_key_to_value[
            "world_model_feature"]
        # diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] # (T, 4) # "Pnn 이하의 길이"
        # diffusion_tokens_dist_order: List[str] # valid diffusion agent 토큰 리스트 (거리 오름차순) # "Pnn 이하의 길이"
        # ego_np_gen_traj_wrt_ego: np.ndarray # (T, 4) or None
        (diff_token_to_np_gen_traj_wrt_ego,
         ego_np_gen_traj_wrt_ego) = self._get_token_to_np_traj_wrt_ego(
             model_inputs, near_track_token_dist_order, neighbor_agents_current,
             ego_agent_current)
        cur_ego_global_xyyaw: npt.NDArray[
            np.floating] = self._get_ego_reference_global_xyyaw(
                self._ego_anchor_state, dtype=np.float64)  # shape (3,)
        # Dict[str, np.ndarray] # (token, (3,)) # 현재 시점의 위치/방향
        diff_token_to_global_xyyaw = self._get_diff_token_to_cur_xyyaw(
            near_track_token_dist_order)

        # Dict[str, Deque[Agent]]
        diffusion_token_to_agent_history = get_token_to_history(
            self.observation_buffer, iteration, near_track_token_dist_order)
        # (token, (len(history), 11))
        ###### 디버깅용 ######
        self._get_diff_token_to_np_history_to_draw(
            diffusion_token_to_agent_history, cur_ego_global_xyyaw)
        ####################
        if ego_np_gen_traj_wrt_ego is None:
            self.ego_trajectory = None
        else:
            self.ego_trajectory: AbstractTrajectory = InterpolatedTrajectory(
                trajectory=self.outputs_to_ego_trajectory(
                    ego_np_gen_traj_wrt_ego, self.ego_state_buffer))
            # (1 + Future_len, 11)
            self._draw_infos.ego_interp_np_traj_wrt_ego = self._get_ego_rel_future_arrays_to_draw(
                self.ego_trajectory, cur_ego_global_xyyaw)

        # diff_token_to_interpol_traj: Dict[str, AbstractTrajectory]
        diff_token_to_interpol_traj = self.get_diff_token_to_interpol_traj(
            diff_token_to_np_gen_traj_wrt_ego, diff_token_to_global_xyyaw,
            diffusion_token_to_agent_history, cur_ego_global_xyyaw)
        self.updated_ego_state = self._update_diffusion_agents(
            diff_token_to_interpol_traj, next_iteration, cur_ego_global_xyyaw,
            diff_token_to_future_all_gt_3_dim)

    def _get_ego_rel_future_arrays_to_draw(
            self,
            future_trajectory: InterpolatedTrajectory,
            cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
    ):
        future_egostates: List[
            EgoState] = future_trajectory.get_sampled_trajectory()
        global_future_arrays = [
            ego_state_to_numpy10(ego_state) for ego_state in future_egostates
        ]  # List[(10,)]
        global_future_arrays = np.stack(global_future_arrays,
                                        axis=0)  # (1+T, 10)
        # rel_future_arrays: (1+T, 11)
        rel_future_arrays = convert_absolute_quantities_to_relative(
            global_future_arrays,
            cur_ego_global_xyyaw)  # cur_ego_global_xyyaw: (3,)
        return rel_future_arrays

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        pass

    def _update_observation_with_predictions(
            self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass
