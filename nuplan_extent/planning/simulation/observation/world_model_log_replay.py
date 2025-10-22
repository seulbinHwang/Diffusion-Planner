from typing import cast, List, Dict, Optional, Deque, Tuple, Union
import numpy as np
import numpy.typing as npt
import draw_machine as draw_machine
from nuplan.common.actor_state.dynamic_car_state import get_velocity_shifted
from nuplan_extent.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
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
from nuplan.common.actor_state.tracked_objects import TrackedObjects
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
from scipy.optimize import linear_sum_assignment
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


def observations_to_agents_buffer(
        observations_buffer: Deque[Observation],
        diffusion_agents_track_tokens: List[str]) -> Deque[List[Agent]]:
    """Observation 버퍼에서 **차량(vehicles)** 만 추출해 Agent 리스트 버퍼로 변환한다.

    Args:
        observations_buffer (Deque[Observation]): 시간 순서로 정렬된 관측 버퍼.

    Returns:
        Deque[List[Agent]]: 각 관측에서 추출한 차량 Agent 리스트를 저장한 버퍼.

    Raises:
        TypeError: 관측이 ``DetectionsTracks`` 타입이 아닐 경우.
    """
    vehicles_buffer: Deque[List[Agent]] = deque(
        maxlen=observations_buffer.maxlen)
    for observation in observations_buffer:
        if isinstance(observation, DetectionsTracks):
            # Agent 중에, 오직 차량(VEHICLE) 타입만 추출한다.
            vehicles = [
                agent for agent in observation.tracked_objects.get_agents()
                if (agent.tracked_object_type in {
                    TrackedObjectType.VEHICLE,
                    TrackedObjectType.PEDESTRIAN,
                    TrackedObjectType.BICYCLE,
                }) and (agent.track_token in diffusion_agents_track_tokens)
            ]
            vehicles_buffer.append(vehicles)
        else:
            raise TypeError("observations_buffer는 DetectionsTracks만 포함해야 합니다. "
                            f"받은 타입: {type(observation)}")
    return vehicles_buffer


def get_token_to_history(
        observation_buffer: Deque[Observation], iteration: SimulationIteration,
        diffusion_agents_track_tokens: List[str]) -> Dict[str, Deque[Agent]]:
    """현재 시점에 존재하는 Agent들의 과거 기록을 생성한다.

    버퍼의 마지막 원소(현재 시점)에 존재하는 Agent들만을 대상으로 하며, 시간 역순(현재→과거)으로
    각 Agent의 상태를 수집한다. 특정 과거시점에 해당 Agent가 존재하지 않으면, 더 이상 과거 상태는
    수집하지 않는다.

    Args:
        vehicles_buffer (Deque[List[Agent]]): 시간 순서대로 정렬된 Agent 버퍼. 길이 :math:`T`의 버퍼이며,
            각 시점마다 임의 길이의 ``Agent`` 리스트를 포함한다.

    Returns:
        Dict[str, Deque[EgoState]]: 현재 시점에 존재하는 Agent 수 :math:`N` 만큼의 리스트를 반환한다. 각 내부
            리스트는 길이 :math:`T`이며, 시간 역순(현재→과거)으로 해당 Agent의 히스토리를 담고 있다.
    """
    # Deque[EgoState]
    # vehicles_buffer: 과거 -> 현재
    vehicles_buffer: Deque[List[Agent]] = observations_to_agents_buffer(
        observation_buffer, diffusion_agents_track_tokens
    )  # self.observation_buffer: Deque[Observation]

    max_len = vehicles_buffer.maxlen
    if not vehicles_buffer:
        return {}
    current_vehicles: List[Agent] = [agent for agent in vehicles_buffer[-1]]
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
    # list(vehicles_buffer) : List[List[Agent]] 과거 -> 현재
    # list(vehicles_buffer)[:-1] : List[List[Agent]] 과거 -> 현재-1
    # reversed_vehicle_buffer: List[List[Agent]] 현재-1 -> 과거
    reversed_vehicle_buffer = reversed(list(vehicles_buffer)[:-1])
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
                         ego_rear_axle_xy: np.ndarray, ego_yaw: float,
                         agent_xyyaw: np.ndarray) -> np.ndarray:
    """
    ego계 기준 future_traj_wrt_ego → 각 vehicle 로컬계 기준 (T,4) trajectory.
    """
    coords_g, yaw_g = ego_to_global(future_traj_wrt_ego, ego_rear_axle_xy,
                                    ego_yaw)
    veh_center_xy = agent_xyyaw[:2]
    veh_yaw = float(agent_xyyaw[2])
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


# nuplan_extent/planning/simulation/observation/world_model_agents.py
class WorldModelLogReplay(AbstractMLAgents):
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
        # TODO: 먼저, 시나리오에서 도로 속도가 없는 경우에, 딥러닝 모델을 학습 시켜서 대응 했음.
        Initializes the WorldModelLogReplay class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.config = model.config
        self.predicted_neighbor_num = self.config.predicted_neighbor_num
        self._radius = radius
        self._planner_step_gap_s = self._step_interval_us / 1e6  # [s]
        self.use_route_lanes = True
        self.use_ego_plan = True

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
                              phi_dot_cap=5.0,
                              R_min=3.0,
                              aN_max=12.0,
                              alpha_max=4.0,
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

    def _update_observation_core(self) -> None:
        """Inherited, see superclass."""
        self._get_current_agents()
        diffusion_agents_tokens = set(self._diffusion_agents.keys())
        for token, tracked_object in self._agents.items():
            if token in diffusion_agents_tokens:
                diffusion_agent = self._diffusion_agents[token]
                tracked_object.predictions = diffusion_agent.predictions

    def _get_current_agents(self):
        all_things: DetectionsTracks = self._scenario.get_tracked_objects_at_iteration(
            self.current_iteration)
        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in all_things.tracked_objects
        }
        self._agents = sort_dict(unique_agents)

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        self.current_iteration = 0
        self.current_observation = None
        self._diffusion_agents = {}
        self._get_current_agents()

    def _get_diffusion_agents(self, ego_state: EgoState) -> None:
        """
        ego 기준, '타입별 상한(cap) 적용 → 정사각형(한 변 2*radius) 필터' 순서로 diffusion 대상 에이전트를 선별한다.

        변경된 절차:
            1) ego와의 유클리드 거리 오름차순으로 정렬
            2) 거리 정렬 순서를 유지한 채, 보행자/자전거 상한(cap)을 우선 적용하고
               남는 슬롯은 차량으로 채워 최대 predicted_neighbor_num개 선택
               - cap: self.config.max_pedestrians / self.config.max_bicycles
                 (없으면 predicted_neighbor_num을 사용하여 사실상 상한 없음)
               - 최종 후보 집합 내부 순서는 전역 거리 오름차순 유지
            3) 위 후보 집합을 ego 기준 정사각형(|x_e| ≤ radius AND |y_e| ≤ radius)으로 최종 필터링
               - 이후 부족 슬롯은 재보충하지 않음(요청 사양)
            4) self._diffusion_agents를 선택된 토큰으로 슬라이스(거리 오름차순 유지)

        시간 복잡도:
            - 거리 정렬 O(N log N), 나머지는 NumPy 벡터 연산(O(N))
        """
        # 시나리오에서 후보 에이전트 수집 (차량/보행자/자전거만)
        unique_agents: Dict[str, TrackedObject] = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.get_tracked_objects_at_iteration(
                self.current_iteration).tracked_objects
            if (tracked_object.track_token is not None) and
            (tracked_object.tracked_object_type in {
                TrackedObjectType.VEHICLE,
                TrackedObjectType.PEDESTRIAN,
                TrackedObjectType.BICYCLE,
            })
        }

        # 1) 거리 기준 오름차순 정렬
        sorted_tokens, sorted_distances = self._compute_sorted_distances(
            ego_state, unique_agents)
        if len(sorted_tokens) == 0:
            self._diffusion_agents = {}
            return

        # 2) 타입별(cap) 적용 → 최대 K개 후보 구성 (전역 거리 오름차순 유지)
        K: int = int(self.predicted_neighbor_num)
        max_pedestrians_cfg: int = int(
            getattr(self.config, "max_pedestrians", 7))
        max_bicycles_cfg: int = int(getattr(self.config, "max_bicycles", 3))
        ped_cap: int = max(0, min(max_pedestrians_cfg, K))
        bike_cap_hint: int = max(0, min(max_bicycles_cfg, K))  # 남은 슬롯과 함께 다시 제한

        # 타입별로 거리순 분할
        pedestrian_tokens: List[str] = []
        bicycle_tokens: List[str] = []
        vehicle_tokens: List[str] = []
        for token in sorted_tokens:
            obj_type = unique_agents[token].tracked_object_type
            if obj_type == TrackedObjectType.PEDESTRIAN:
                pedestrian_tokens.append(token)
            elif obj_type == TrackedObjectType.BICYCLE:
                bicycle_tokens.append(token)
            elif obj_type == TrackedObjectType.VEHICLE:
                vehicle_tokens.append(token)
            else:
                # 현재 대상 외 타입은 무시
                pass

        # cap 적용: 보행자 → 자전거 → 차량(상한 없음)
        selected_pedestrian_tokens: List[str] = pedestrian_tokens[:ped_cap]
        remaining_slots: int = K - len(selected_pedestrian_tokens)
        selected_bicycle_tokens: List[str] = []
        if remaining_slots > 0:
            bike_take = min(bike_cap_hint, remaining_slots)
            selected_bicycle_tokens = bicycle_tokens[:bike_take]
            remaining_slots -= len(selected_bicycle_tokens)

        selected_vehicle_tokens: List[str] = []
        if remaining_slots > 0:
            selected_vehicle_tokens = vehicle_tokens[:remaining_slots]

        # 전역 거리 오름차순 유지: 기존 정렬 리스트를 따라 재조합
        selected_set = set(selected_pedestrian_tokens +
                           selected_bicycle_tokens + selected_vehicle_tokens)
        pre_square_selected_tokens: List[str] = [
            t for t in sorted_tokens if t in selected_set
        ]
        if len(pre_square_selected_tokens) == 0:
            self._diffusion_agents = {}
            return

        # 3) 정사각형(|x_e| ≤ radius AND |y_e| ≤ radius) 최종 필터
        #    - ego 뒷축을 원점, ego yaw 정렬 좌표계에서 판정
        ego_xy: np.ndarray = np.asarray(ego_state.rear_axle.point.array,
                                        dtype=np.float32)  # (2,)
        ego_yaw: float = float(ego_state.rear_axle.heading)
        R_g2e: np.ndarray = rotation_matrix(-ego_yaw)  # 세계→ego 회전 (2,2)

        # 선택된 후보들의 월드 좌표(거리 오름차순 순서)
        agents_xy_pre: np.ndarray = np.array([
            unique_agents[t].center.point.array
            for t in pre_square_selected_tokens
        ],
                                             dtype=np.float32)  # (M,2)

        delta_xy_pre: np.ndarray = agents_xy_pre - ego_xy[None, :]  # (M,2)
        local_xy_pre: np.ndarray = delta_xy_pre.dot(R_g2e.T)  # (M,2)
        abs_local_xy_pre: np.ndarray = np.abs(local_xy_pre)  # (M,2)

        half_side_length: float = float(self._radius)
        square_mask_pre: np.ndarray = (
            (abs_local_xy_pre[:, 0] <= half_side_length) &
            (abs_local_xy_pre[:, 1] <= half_side_length))  # (M,)

        if not np.any(square_mask_pre):
            self._diffusion_agents = {}
            return

        final_selected_tokens_within_square: List[str] = [
            t for t, keep in zip(pre_square_selected_tokens, square_mask_pre)
            if bool(keep)
        ]

        # 4) 사전 슬라이스(거리 오름차순 유지). 정사각형 필터 이후 부족 슬롯은 재보충하지 않음.
        self._diffusion_agents = {
            t: unique_agents[t] for t in final_selected_tokens_within_square
        }

    def _get_interpol_time_points(
            self, iteration: SimulationIteration) -> List[TimePoint]:
        step_s_time: float = self.sim_step_gap_time_point.time_s
        """
        step_s_time : 0.15 (시뮬레이션 시간 간격 (초))
        self._planner_step_gap_s : 0.1 이면 (future trajectory의 점 사이 시간 간격)
            q = 1.5 -> interpol_num = 2
        [현실]
            step_s_time: 0.09992 self._planner_step_gap_s: 0.1 interpol_num: 1
        """
        q = Decimal(str(step_s_time)) / Decimal(str(self._planner_step_gap_s))
        interpol_num = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        interpol_num = max(interpol_num, 1)
        """
        if interpol_num = 2,
            interpol_indices = [1, 2]
            interpol_points_times = [0.1, 0.2]
            interpol_time_points = [TimePoint(current_time + 0.1s), TimePoint(current_time + 0.2s)]
        [현실] 
            interpol_indices: [1] 
            interpol_points_times: [0.1]
            interpol_time_points = [TimePoint(current_time + 0.1s)]
        """
        interpol_indices = np.linspace(0,
                                       interpol_num,
                                       num=interpol_num + 1,
                                       dtype=int)[1:]  # (interpol_num, )
        interpol_points_times = interpol_indices * self._planner_step_gap_s  # (interpol_num, )
        interpol_time_points: List[TimePoint] = []
        for interpol_time in interpol_points_times:
            time_point = TimePoint(time_us=int(iteration.time_point.time_us +
                                               interpol_time * 1e6))
            interpol_time_points.append(time_point)
        return interpol_time_points

    def _get_next_ego_plans(
            self, current_ego_state: EgoState, next_ego_state: EgoState,
            interpol_time_points: List[TimePoint]) -> List[InterpolatableState]:
        """
        interpol_time_points = [TimePoint(current_time + 0.1s)]
        """
        states: List[InterpolatableState] = [current_ego_state, next_ego_state]
        next_ego_trajectory = InterpolatedTrajectory(trajectory=states)
        next_ego_plans: List[
            InterpolatableState] = next_ego_trajectory.get_state_at_times(
                interpol_time_points)
        return next_ego_plans

    def _preprocess_next_ego_plans(
            self, next_ego_plans: List[EgoState],
            current_ego_state: EgoState) -> npt.NDArray[np.float32]:
        """ego 미래 상태들을 diffusion planner 입력 배열로 변환한다.

        Args:
            next_ego_plans (List[InterpolatableState]): 변환할 ego 상태 리스트.
                - 내 경우에는 실제로 돌려보니 길이가 1이었음.
            current_ego_state (EgoState): 기준이 되는 현재 ego 상태.

        Returns:
            npt.NDArray[np.float64]: (interpol_num, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """

        interpol_num = len(next_ego_plans)
        absolute: npt.NDArray[np.float64] = np.zeros(
            (interpol_num, 10), dtype=np.float64)  # shape (T, 10)
        absolute[:, 7] = 1  # is vehicle

        for i, state in enumerate(next_ego_plans):
            absolute[i, 0] = state.center.x  # 절대 위치
            absolute[i, 1] = state.center.y  # 절대 위치
            absolute[i, 2] = state.center.heading  # 절대 헤딩
            # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
            v_local = state.dynamic_car_state.center_velocity_2d
            # v_global = numpy_array_to_absolute_velocity(
            #     state.center,
            #     np.array([[v_local.x, v_local.y]], dtype=np.float32))[0]
            he = float(state.center.heading)
            c, s = np.cos(he), np.sin(he)
            vx_w = c * float(v_local.x) - s * float(v_local.y)
            vy_w = s * float(v_local.x) + c * float(v_local.y)
            absolute[i, 3] = vx_w  # 자차량 좌표계 속도 -> 글로벌 좌표계 속도
            absolute[i, 4] = vy_w  # 자차량 좌표계 속도 -> 글로벌 좌표계 속도
            absolute[i, 5] = state.car_footprint.width
            absolute[i, 6] = state.car_footprint.length

        anchor = np.array([
            current_ego_state.rear_axle.x,
            current_ego_state.rear_axle.y,
            current_ego_state.rear_axle.heading,
        ],
                          dtype=np.float32)  # shape (3,)
        # absolute: (interpol_num, 10)
        # relative: (interpol_num, 11)
        relative: np.ndarray = convert_absolute_quantities_to_relative(
            absolute, anchor, 'ego')  # shape (interpol_num, 11)
        assert (interpol_num, 11) == relative.shape
        return relative.astype(np.float32)

    def _preprocess_ego_future_traj(
            self, ego_future_trajectory: InterpolatedTrajectory,
            current_ego_state: EgoState) -> npt.NDArray[np.float32]:
        """미래 ego 궤적을 diffusion planner 입력 배열로 변환한다.

        Args:
            ego_future_trajectory (InterpolatedTrajectory): 변환할 미래 ego 궤적.
            current_ego_state (EgoState): 기준이 되는 현재 ego 상태.

        Returns:
            npt.NDArray[np.float64]: (config.future_len, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """

        future_states: List[EgoState] = list(
            ego_future_trajectory.get_sampled_trajectory())
        future_len_plus_1 = len(future_states)
        absolute: npt.NDArray[np.float64] = np.zeros(
            (self.config.future_len, 10), dtype=np.float64)  # shape (T, 7)

        for i, state in enumerate(future_states):
            absolute[i, 0] = state.center.x
            absolute[i, 1] = state.center.y
            absolute[i, 2] = state.center.heading
            # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
            v_local = state.dynamic_car_state.center_velocity_2d
            # v_global = numpy_array_to_absolute_velocity(
            #     state.center,
            #     np.array([[v_local.x, v_local.y]], dtype=np.float32))[0]
            he = float(state.center.heading)
            c, s = np.cos(he), np.sin(he)
            vx_w = c * float(v_local.x) - s * float(v_local.y)
            vy_w = s * float(v_local.x) + c * float(v_local.y)
            absolute[i, 3] = vx_w
            absolute[i, 4] = vy_w
            absolute[i, 5] = state.car_footprint.width
            absolute[i, 6] = state.car_footprint.length
            absolute[i, 7] = 1  # is vehicle

        anchor = np.array([
            current_ego_state.rear_axle.x,
            current_ego_state.rear_axle.y,
            current_ego_state.rear_axle.heading,
        ],
                          dtype=np.float32)  # shape (3,)
        relative: np.ndarray = convert_absolute_quantities_to_relative(
            absolute, anchor, 'ego')  # shape (T, 11)
        relative[future_len_plus_1:, :] = 0.0
        return relative.astype(np.float32)

    def set_vis_features(self, is_vis_features: bool, vis_features_path: str):
        """
        Set params for saving features, for visualization, only when simulation feature video callback is on.
        :param is_vis_features: whether to save features
        :param vis_features_path: path to save features
        """
        self._is_vis_features = is_vis_features
        self._vis_features_path = vis_features_path

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

        Args:
            interp_next_ego_state (List[InterpolatableState]): 변환할 ego 상태 리스트.
                - 내 경우에는 실제로 돌려보니 길이가 1이었음.
            current_ego_state (EgoState): 기준이 되는 현재 ego 상태.

        Returns:
            npt.NDArray[np.float64]: (interpol_num, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """

        interpol_num = len(interp_next_ego_state)
        interpol_abs_next_ego: npt.NDArray[np.float64] = np.zeros(
            (interpol_num, 10), dtype=np.float64)  # shape (T, 10)
        interpol_abs_next_ego[:, 7] = 1  # is vehicle

        for i, state in enumerate(interp_next_ego_state):
            interpol_abs_next_ego[i, 0] = state.center.x  # 절대 위치
            interpol_abs_next_ego[i, 1] = state.center.y  # 절대 위치
            interpol_abs_next_ego[i, 2] = state.center.heading  # 절대 헤딩
            # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
            v_local = state.dynamic_car_state.center_velocity_2d
            he = float(state.center.heading)
            c, s = np.cos(he), np.sin(he)
            vx_w = c * float(v_local.x) - s * float(v_local.y)
            vy_w = s * float(v_local.x) + c * float(v_local.y)
            interpol_abs_next_ego[i, 3] = vx_w  # 자차량 좌표계 -> 글로벌 좌표계 속도
            interpol_abs_next_ego[i, 4] = vy_w  # 자차량 좌표계 -> 글로벌 좌표계 속도
            interpol_abs_next_ego[i, 5] = state.car_footprint.width
            interpol_abs_next_ego[i, 6] = state.car_footprint.length

        anchor = np.array([
            current_ego_state.rear_axle.x,
            current_ego_state.rear_axle.y,
            current_ego_state.rear_axle.heading,
        ],
                          dtype=np.float32)  # shape (3,)
        # interpol_abs_next_ego: (interpol_num, 10)
        # interp_next_ego_11_dim: (interpol_num, 11)
        interp_next_ego_11_dim = convert_absolute_quantities_to_relative(
            interpol_abs_next_ego, anchor, 'ego')
        return interp_next_ego_11_dim.astype(np.float32)

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

    def _from_ego_fut_traj_to_np(
            self, ego_future_trajectory: InterpolatedTrajectory,
            current_ego_state: EgoState) -> Optional[npt.NDArray[np.float32]]:
        """미래 ego 궤적을 diffusion planner 입력 배열로 변환한다.

        Args:
            ego_future_trajectory (InterpolatedTrajectory): 변환할 미래 ego 궤적.
            current_ego_state (EgoState): 기준이 되는 현재 ego 상태.

        Returns:
            npt.NDArray[np.float64]: (config.future_len, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """
        planner_future_11_dim = None
        if ego_future_trajectory is not None:
            # (future_len, 11)
            future_states: List[EgoState] = list(
                ego_future_trajectory.get_sampled_trajectory(
                ))  # len: 1 + valid_future_len
            # remove first current state
            future_states = future_states[1:]  # len = valid_future_len
            valid_future_len = len(future_states)
            assert valid_future_len <= self.config.future_len, \
                f"미래 상태 개수({valid_future_len})가 config.future_len({self.config.future_len})보다 큽니다."
            global_ego_fut_traj_10: npt.NDArray[np.float64] = np.zeros(
                (self.config.future_len, 10),
                dtype=np.float64)  # shape (future_len, 7)

            for i, state in enumerate(future_states):  # valid_future_len
                global_ego_fut_traj_10[i, 0] = state.center.x
                global_ego_fut_traj_10[i, 1] = state.center.y
                global_ego_fut_traj_10[i, 2] = state.center.heading
                # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
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

            anchor = np.array([
                current_ego_state.rear_axle.x,
                current_ego_state.rear_axle.y,
                current_ego_state.rear_axle.heading,
            ],
                              dtype=np.float32)  # shape (3,)
            planner_future_11_dim = convert_absolute_quantities_to_relative(
                global_ego_fut_traj_10, anchor, 'ego')  # shape (future_len, 11)
            planner_future_11_dim[valid_future_len:, :] = 0.
            planner_future_11_dim = planner_future_11_dim.astype(np.float32)
        return planner_future_11_dim

    def _get_model_input(
        self, iteration: SimulationIteration, history: SimulationHistoryBuffer,
        interp_next_ego_11_dim: Optional[npt.NDArray[np.float32]],
        planner_future_11_dim: Optional[npt.NDArray[np.float32]]
    ) -> Tuple[Dict[str, AbstractModelFeature], List[Optional[str]], Dict[
            str, np.ndarray], np.ndarray]:
        # Construct input features
        # route_roadblock_ids: Optional[Dict[str, List[str]]
        if self.use_route_lanes:
            route_roadblock_ids = self._scenario.get_route_roadblock_ids()
        else:
            route_roadblock_ids = None
        initialization = HorizonPlannerInitialization(
            # 시나리오가 끝나고도 계속 진행했을 때 최종적으로 도달해야 하는 포즈 (존재하지 않을 수도 있음)
            mission_goal=self._scenario.get_mission_goal(),
            # (x, y, yaw) 의 StateSE2
            route_roadblock_ids=route_roadblock_ids,
            map_api=self._scenario.map_api,
            scenario=self._scenario,
            # 전문 운전자(ground truth)의 실제 마지막 상태 (항상 존재)
            expert_goal_state=self._scenario.get_expert_goal_state(),
            # (x, y, yaw) 의 StateSE2
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            iteration.index)
        # target_agents_mask: np.ndarray, (agent_num,) bool
        # diffusion_agents_tokens: List[str] # len: valid diffusion agent num
        diffusion_agents_tokens = None  #list(self._diffusion_agents.keys())
        current_input = PlannerInput(iteration, history, traffic_light_data,
                                     diffusion_agents_tokens,
                                     interp_next_ego_11_dim,
                                     planner_future_11_dim)
        # WorldModelFeatureBuilder.get_features_from_simulation
        # from nuplan_extent/planning/training/preprocessing/feature_builders/world_model_feature_builder.py
        model_input_key_to_value: Dict[
            str, AbstractModelFeature] = self._model_loader.build_features(
                current_input, initialization)
        model_input_key_to_unnorm_value: Dict[
            str, np.ndarray] = self._model_loader.feature_builders[
                0].unnormalized_features
        # neighbor_token_dist_order: len = agent_num
        neighbor_token_dist_order: List[
            Optional[str]] = model_input_key_to_unnorm_value[
                "neighbor_track_token"]  # (agent_num, )
        diff_token_to_future_gt_3_dim: Dict[
            str, np.ndarray] = model_input_key_to_unnorm_value[
                "diff_token_to_future_gt_3_dim"]  # Dict[str, np.ndarray] # len : valid_agent_num
        neighbor_agents_past = model_input_key_to_unnorm_value[
            "neighbor_agents_past"]
        self._draw_infos.model_input_key_to_unnorm_value = model_input_key_to_unnorm_value
        return (model_input_key_to_value, neighbor_token_dist_order,
                diff_token_to_future_gt_3_dim, neighbor_agents_past)

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
        if self.use_ego_plan:
            planner_future_11_dim: Optional[
                np.ndarray] = self._from_ego_fut_traj_to_np(ego_future_trajectory,
                                                            self._ego_anchor_state)
        else:
            planner_future_11_dim = None
        # model_input_key_to_value: Dict[str, AbstractModelFeature]
        # neighbor_token_dist_order: List[Optional[str]] # len = agent_num
        # diff_token_to_future_gt_3_dim: Dict[str, np.ndarray] # len : valid_agent_num
        (model_input_key_to_value, neighbor_token_dist_order,
         diff_token_to_future_gt_3_dim,
         neighbor_agents_past) = self._get_model_input(iteration, history,
                                                       interp_next_ego_11_dim,
                                                       planner_future_11_dim)

        # Infer model
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)
        self.infer_model(model_input_key_to_value, iteration, next_iteration,
                         neighbor_token_dist_order,
                         diff_token_to_future_gt_3_dim, neighbor_agents_past)

    def update_observation(
            self,
            iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            next_ego_state: Optional[EgoState] = None,
            ego_future_trajectory: Optional[InterpolatedTrajectory] = None
    ) -> None:
        """
        - 자동차
            - ego 기준, radius 안에 들어오면 -> diffusion 생성 대상
            - ego 기준, radius 밖에 있으면 -> 삭제. 관리 안함
        - log-replay
            -
        """
        self._draw_infos = draw_machine.DrawInfos()

        self.observation_buffer: Deque[Observation] = history.observation_buffer
        # EgoState, Observation
        self._ego_anchor_state, self.current_observation = history.current_state
        # self._get_diffusion_agents(self._ego_anchor_state)

        self.sim_step_gap_time_point: TimePoint = next_iteration.time_point - iteration.time_point

        self.current_iteration: int = next_iteration.index

        # world_model_feature: Dict[str, np.ndarray]
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)

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

    def outputs_to_trajectory(self, future_traj_wrt_npc_center: np.ndarray,
                              ego_state_history: Deque[Agent],
                              step_s_time: float) -> List[InterpolatableState]:
        heading = np.arctan2(future_traj_wrt_npc_center[:, 3],
                             future_traj_wrt_npc_center[:, 2])[..., None]
        future_traj_wrt_npc_center = np.concatenate(
            [future_traj_wrt_npc_center[..., :2], heading], axis=-1)

        waypoints = transform_predictions_to_states(future_traj_wrt_npc_center,
                                                    ego_state_history,
                                                    self._future_horizon,
                                                    self._planner_step_gap_s,
                                                    step_s_time)
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

    def get_npc_center_poses(self,
                             current_agent: Agent) -> Tuple[np.ndarray, float]:
        """각 차량의 중앙 위치와 방향을 계산한다.

        Returns:
            Tuple[np.ndarray, float]: shape (2,), 차량 중앙 [x, y]와 방향 yaw_rad.

        """# .center.x
        position: np.ndarray = np.array([
            current_agent.center.x,
            current_agent.center.y,
        ],
                                        dtype=float)  # shape: (2,)

        return position, current_agent.center.heading

    def _compute_sorted_distances(
        self, ego_state: EgoState, token_to_agent: Dict[str, Agent]
    ) -> tuple[list[str], npt.NDArray[np.float64]]:
        """ego와 각 agent 사이의 거리를 계산해 정렬된 결과를 반환한다.

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
            dtype=np.float32)  # shape (N, 2)
        ego_xy: npt.NDArray[np.float32] = np.expand_dims(
            ego_state.rear_axle.point.array,
            axis=0).astype(np.float32)  # shape (1, 2)
        distances: npt.NDArray[np.float64] = cdist(
            ego_xy, agents_xy).flatten()  # shape (N,)
        sorted_indices: npt.NDArray[np.int64] = np.argsort(
            distances)  # shape (N,)
        sorted_tokens: list[str] = [tokens[i] for i in sorted_indices]

        sorted_distances: npt.NDArray[np.float64] = distances[sorted_indices]
        return sorted_tokens, sorted_distances

    def _filter_trajectory(
        self,
        future_np_trajs_wrt_ego: np.ndarray,
        neighbor_agents_past: np.ndarray,  # (Pnn,time_len, 11)
        veh_valid_mask: np.ndarray,  # (Pnn,)
        bic_valid_mask: np.ndarray,  # (Pnn,)
        ped_valid_mask: np.ndarray,  # (Pnn,)
        draw_idx
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
        near_future_a3 = near_current_future_a2[: , 1:, :]  # (Pnn, 80, 4)

        # # 3) 각속도 제약 + 스무딩 단계
        # near_future_a3 = yawrate_smooth_stage(
        #     near_current_future_a2=near_current_future_a2,
        #     near_current_future_dir=near_current_future_dir,
        #     near_future_body_slip=near_future_body_slip,
        #     veh_valid_mask=veh_valid_mask,
        #     bic_valid_mask=bic_valid_mask,
        #     ped_valid_mask=ped_valid_mask,
        #     cfg=cfg,
        #     draw_idx=draw_idx
        # )  # (Pnn, 80, 4)
        return near_current_future_a2, near_future_a3

    def _get_token_to_np_traj_wrt_ego(
        self,
        model_inputs: AbstractModelFeature,
        neighbor_token_dist_order: List[Optional[str]],  # len == agent_num
        neighbor_agents_past: np.ndarray
        # (agent_num, time_len, 11)
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Returns:
            diff_token_to_np_gen_traj_wrt_ego
                Dict[str, np.ndarray]: (token, (T, 4)) # ego 좌표계 기준 차량 중심의 값
                # len == valid diffusion agent num
            diffusion_tokens_dist_order
                List[str]: # self._diffusion_agents 의 토큰 리스트 (거리 오름차순)
                # len == valid diffusion agent num
        """
        """
        # nuplan_extent/planning/training/modeling/models/world_model.py
        # WorldModel.forward
        # future_np_trajs_wrt_ego: (Pnn, 1+T, 4) # diffusion_agents 에서 생성하라는거만 생성햇음.
        # ego와 거리 순으로 모든 agent가 들어있다는 가정!!! (생성 안했으면, 빈 값을 준다.)
        """
        future_np_trajs_wrt_ego: np.ndarray = self._model_loader.infer(
            model_inputs).detach().cpu().numpy()
        gen_slot_len = future_np_trajs_wrt_ego.shape[0]
        # neighbor_agents_past: (Pnn, time_len, 11)
        # (T, 4) # 길이: Pnn 중, 실제로 궤적 생성한 대상들만.
        diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] = {}
        diff_token_to_np_gen_traj_11_wrt_ego: Dict[str, np.ndarray] = {}
        diff_token_to_np_hist_wrt_ego: Dict[str, np.ndarray] = {}
        veh_valid_mask = []
        bic_valid_mask = []
        ped_valid_mask = []

        self._diffusion_agents = {}
        draw_idx = None
        for idx, token in enumerate(neighbor_token_dist_order):
            if token == "f476b2c85dd7508c":
                draw_idx = idx
            agent_current = neighbor_agents_past[idx, 0]  # (11)
            # agent_current: (11) -> (T, 11)
            np_gen_traj_wrt_ego = np.tile(
                agent_current,
                (future_np_trajs_wrt_ego.shape[1] - 1, 1))  # (T, 11)
            if idx >= gen_slot_len:
                break
            np_traj_wrt_ego = future_np_trajs_wrt_ego[idx, 1:, :]  # (T, 4)
            np_gen_traj_wrt_ego[:, :4] = np_traj_wrt_ego  # (T, 11)
            np_traj_sum = np_traj_wrt_ego.sum()  # (T, 4) 의 합
            if np.allclose(np_traj_sum, 0.0) or token is None:
                veh_valid_mask.append(False)
                bic_valid_mask.append(False)
                ped_valid_mask.append(False)
                continue
            diff_token_to_np_gen_traj_wrt_ego[token] = np_traj_wrt_ego
            diff_token_to_np_gen_traj_11_wrt_ego[
                token] = np_gen_traj_wrt_ego  # (T, 11)
            agent_past = neighbor_agents_past[idx]  # (time_len, 11)
            agent_class = agent_past[-1,
                                     8:]  # (3,) one-hot # vehicle, ped, bicycle
            veh_valid_mask.append(bool(agent_class[0]))
            ped_valid_mask.append(bool(agent_class[1]))
            bic_valid_mask.append(bool(agent_class[2]))
            diff_token_to_np_hist_wrt_ego[token] = agent_past
        ### 디버깅용 ###
        self._draw_infos.diff_token_to_np_gen_traj_11_wrt_ego = diff_token_to_np_gen_traj_11_wrt_ego

        veh_valid_mask = np.array(veh_valid_mask, dtype=bool)  # (Pnn,)
        bic_valid_mask = np.array(bic_valid_mask, dtype=bool)  # (Pnn,)
        ped_valid_mask = np.array(ped_valid_mask, dtype=bool)  # (Pnn,)
        near_current_future_a2, near_future_a3 = self._filter_trajectory(
            future_np_trajs_wrt_ego, neighbor_agents_past, veh_valid_mask,
            bic_valid_mask, ped_valid_mask, draw_idx)
        diff_token_to_np_slip_traj_11_wrt_ego: Dict[str, np.ndarray] = {}
        diff_token_to_np_smooth_traj_11_wrt_ego: Dict[str, np.ndarray] = {}
        for idx, token in enumerate(neighbor_token_dist_order):
            agent_current = neighbor_agents_past[idx, 0]  # (11)
            # agent_current: (11) -> (T, 11)
            np_slip_traj_wrt_ego = np.tile(
                agent_current, (near_future_a3.shape[1], 1))  # (T, 11)
            np_smooth_traj_wrt_ego = np.tile(
                agent_current, (near_future_a3.shape[1], 1))  # (T, 11)
            if idx >= gen_slot_len:
                break
            np_slip_traj_wrt_ego[:, :4] = near_current_future_a2[
                idx, 1:, :]  # (T, 4)
            np_smooth_traj_wrt_ego[:, :4] = near_future_a3[idx, :, :]  # (T, 4)
            np_traj_sum = np_traj_wrt_ego.sum()  # (T, 4) 의 합
            if np.allclose(np_traj_sum, 0.0) or token is None:
                continue
            diff_token_to_np_slip_traj_11_wrt_ego[token] = np_slip_traj_wrt_ego
            diff_token_to_np_smooth_traj_11_wrt_ego[
                token] = np_smooth_traj_wrt_ego
            self._diffusion_agents[token] = self._agents[token]
        self._draw_infos.diff_token_to_np_slip_traj_11_wrt_ego = diff_token_to_np_slip_traj_11_wrt_ego
        self._draw_infos.diff_token_to_np_smooth_traj_11_wrt_ego = diff_token_to_np_smooth_traj_11_wrt_ego

        diffusion_tokens_dist_order, _ = self._compute_sorted_distances(
            self._ego_anchor_state, self._diffusion_agents)
        return diff_token_to_np_slip_traj_11_wrt_ego, diffusion_tokens_dist_order

    def _get_diff_token_to_cur_xyyaw(
            self,
            diffusion_tokens_dist_order: List[
                str],  # valid diffusion agent 토큰 리스트 (거리 오름차순)
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
            agent.track_token in diffusion_tokens_dist_order
        }  # shape (3,)
        return diff_token_to_global_xyyaw

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

    def _get_diff_token_to_interpol_traj_wrt_ego(
            self,
            diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray],  # (T, 4)
            diff_token_to_global_xyyaw: Dict[str, np.ndarray],  # (token, (3,))
            diffusion_token_to_agent_history: Dict[
                str, Deque[Agent]],  # (token, Deque[Agent])
            cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
    ) -> Dict[str, AbstractTrajectory]:
        sim_step_gap_s: float = self.sim_step_gap_time_point.time_s
        ego_rear_axle_xy, ego_yaw = self.get_rear_axle_pose(
            self._ego_anchor_state)
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
                raise ValueError("future_traj_wrt_ego 값이 전부 0. 입니다.")
            agent_xyyaw = diff_token_to_global_xyyaw[token]
            future_traj_wrt_npc_center = transform_trajectory(
                future_traj_wrt_ego, ego_rear_axle_xy, ego_yaw,
                agent_xyyaw)  # (T, 4)
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

    def infer_model(
        self,
        model_input_key_to_value: Dict[str, AbstractModelFeature],
        iteration: SimulationIteration,
        next_iteration: SimulationIteration,
        neighbor_token_dist_order: List[Optional[str]],  # len == agent_num,
        diff_token_to_future_gt_3_dim: Dict[
            str, np.ndarray],  # len : valid_agent_num # (future_len, 3)
        neighbor_agents_past: np.ndarray  # (agents_num, time_len, 11)
    ) -> None:
        model_inputs: AbstractModelFeature = model_input_key_to_value[
            "world_model_feature"]
        # diff_token_to_np_gen_traj_wrt_ego: Dict[str, np.ndarray] # (T, 4)
        # diffusion_tokens_dist_order: List[str] # valid diffusion agent 토큰 리스트 (거리 오름차순)
        (diff_token_to_np_gen_traj_wrt_ego,
         diffusion_tokens_dist_order) = self._get_token_to_np_traj_wrt_ego(
             model_inputs, neighbor_token_dist_order, neighbor_agents_past)

        cur_ego_global_xyyaw = np.array([
            self._ego_anchor_state.rear_axle.x,
            self._ego_anchor_state.rear_axle.y,
            self._ego_anchor_state.rear_axle.heading
        ],
                                        dtype=np.float64)  # shape (3,)
        # Dict[str, np.ndarray] # (token, (3,)) # 현재 시점의 위치/방향
        diff_token_to_global_xyyaw = self._get_diff_token_to_cur_xyyaw(
            diffusion_tokens_dist_order)

        # Dict[str, Deque[Agent]]
        diffusion_token_to_agent_history = get_token_to_history(
            self.observation_buffer, iteration, diffusion_tokens_dist_order)
        # (token, (len(history), 11))
        ###### 디버깅용 ######
        self._get_diff_token_to_np_history_to_draw(
            diffusion_token_to_agent_history, cur_ego_global_xyyaw)
        ####################
        # diff_token_to_interpol_traj: Dict[str, AbstractTrajectory]
        diff_token_to_interpol_traj = self._get_diff_token_to_interpol_traj_wrt_ego(
            diff_token_to_np_gen_traj_wrt_ego, diff_token_to_global_xyyaw,
            diffusion_token_to_agent_history, cur_ego_global_xyyaw)
        self._update_diffusion_agents(diff_token_to_interpol_traj,
                                      next_iteration, cur_ego_global_xyyaw,
                                      diff_token_to_future_gt_3_dim)
        self._update_observation_core()

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        pass

    def from_np_to_waypoint_list(
        self,
        a_near_future_gt_3_dim: Optional[np.ndarray]  # (future_len, 3)
    ) -> List[Waypoint]:
        a_near_future_waypoints: List[Waypoint] = []
        if a_near_future_gt_3_dim is not None:
            for t in range(a_near_future_gt_3_dim.shape[0]):
                state = a_near_future_gt_3_dim[t, :]  # (3,)
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
                a_near_future_waypoints.append(waypoint)
        return a_near_future_waypoints

    def _update_diffusion_agents(
        self,
        diff_token_to_interpol_traj: Dict[str, AbstractTrajectory],
        next_iteration: SimulationIteration,
        cur_ego_global_xyyaw: np.ndarray,  # shape (3,)
        diff_token_to_future_gt_3_dim: Dict[str,
                                            np.ndarray]  # len : valid_agent_num
    ) -> None:
        # [NEW] 1) GT(ego frame) → Global frame 변환
        gt_global: Dict[str, np.ndarray] = {}
        for token, local_traj_xyh in diff_token_to_future_gt_3_dim.items():
            gt_global[token] = ego_local_traj3_to_global(
                local_traj_xyh=local_traj_xyh,  # (T,3) in ego
                cur_ego_global_xyyaw=cur_ego_global_xyyaw,  # (3,) global
            )  # (T,3) in global
        diff_token_to_future_gt_3_dim = gt_global
        diff_token_to_updated_agent: Dict[str, Agent] = {}
        diff_token_to_next_wp_wrt_ego: Dict[str, np.ndarray] = {}  # (1, 11)
        for diff_token, interpol_traj in diff_token_to_interpol_traj.items():

            agent_ = self._diffusion_agents[diff_token]
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
            a_near_future_gt_3_dim = diff_token_to_future_gt_3_dim[
                diff_token]  # (future_len, 3)
            updated_agent.predictions = [
                # GT 궤적
                PredictedTrajectory(probability=0.5,
                                    waypoints=self.from_np_to_waypoint_list(
                                        a_near_future_gt_3_dim)),
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

        self._diffusion_agents = diff_token_to_updated_agent
        self._draw_infos.diff_token_to_next_wp_wrt_ego = diff_token_to_next_wp_wrt_ego

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

    def _update_observation_with_predictions(
            self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass
