from typing import cast, List, Dict, Optional, Deque, Tuple, Union
import numpy as np
import numpy.typing as npt
import draw_machine
from nuplan.common.actor_state.dynamic_car_state import get_velocity_shifted
from nuplan_extent.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from collections import deque
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

np.set_printoptions(precision=3, suppress=True)

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

from scipy.spatial.distance import cdist
# /Users/user/PycharmProjects/nuplan-devkit/nuplan/common/actor_state/tracked_objects.py
from nuplan.common.utils.interpolatable_state import InterpolatableState
from decimal import Decimal, ROUND_HALF_UP
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
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
    - [4] center v_x
    - [5] center v_y
    - [6] 패딩 0.0
    - [7] 패딩 0.0
    - [8] 1.0
    - [9] 0.0
    - [10] 0.0

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

    # 방향 → cos/sin
    cos_yaw: float = float(np.cos(yaw_rad))
    sin_yaw: float = float(np.sin(yaw_rad))

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
    token_to_refined_traj_wrt_ego: Dict[str, np.ndarray],
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
        token_to_refined_traj_wrt_ego: Dict[str, np.ndarray]
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
    if not token_to_refined_traj_wrt_ego:
        return token_to_refined_traj_wrt_ego

    sortable_triplets: List[Tuple[str, float, np.ndarray]] = []
    for token, arr in token_to_refined_traj_wrt_ego.items():
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
                if (agent.track_token in diffusion_agents_track_tokens)
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
    TODO: 이 방식이 최선인가?

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
    # only for car.
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
                         veh_center_xy: np.ndarray,
                         veh_yaw: float) -> np.ndarray:
    """
    ego계 기준 future_traj_wrt_ego → 각 vehicle 로컬계 기준 (T,4) trajectory.
    """
    coords_g, yaw_g = ego_to_global(future_traj_wrt_ego, ego_rear_axle_xy,
                                    ego_yaw)
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
        self.current_iteration = 0
        self._step_interval = self._step_interval_us / 1e6  # [s]

    def get_observation(self) -> DetectionsTracks:
        """Inherited, see superclass."""
        all_things: DetectionsTracks = self._scenario.get_tracked_objects_at_iteration(
            self.current_iteration)
        tracked_objects: TrackedObjects = all_things.tracked_objects
        new_tracked_objects_list: List[TrackedObject] = []
        diffusion_agents_tokens = list(self._diffusion_agents.keys())
        for tracked_object in tracked_objects:
            # tracked_object: Agent # 바꿔치기 하면됨.
            if tracked_object.track_token in diffusion_agents_tokens:
                diffusion_agent = self._diffusion_agents[
                    tracked_object.track_token]
                tracked_object.predictions = diffusion_agent.predictions
            new_tracked_objects_list.append(tracked_object)
        all_things.tracked_objects = TrackedObjects(new_tracked_objects_list)

        return all_things

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        self.current_iteration = 0
        self.current_observation = None
        self._diffusion_agents = {}

    def _get_open_loop_track_objects(
            self, iteration: int) -> Dict[str, TrackedObject]:
        """
        Get open-loop tracked objects from scenario.
        :param iteration: The simulation iteration.
        :return: A list of TrackedObjects.
        """
        detections = self._scenario.get_tracked_objects_at_iteration(iteration)
        tracked_objects = detections.tracked_objects.get_tracked_objects_of_types(
            self._open_loop_detections_types)
        return {
            tracked_object.track_token: tracked_object
            for tracked_object in tracked_objects
            if tracked_object.track_token is not None
        }

    def _get_interpol_time_points(
            self, iteration: SimulationIteration) -> List[TimePoint]:
        step_s_time: float = self.step_time_point.time_s
        """
        step_s_time : 0.15 (시뮬레이션 시간 간격 (초))
        self._step_interval : 0.1 이면 (future trajectory의 점 사이 시간 간격)
            q = 1.5 -> interpol_num = 2
        [현실]
            step_s_time: 0.09992 self._step_interval: 0.1 interpol_num: 1
        """
        q = Decimal(str(step_s_time)) / Decimal(str(self._step_interval))
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
        interpol_points_times = interpol_indices * self._step_interval  # (interpol_num, )
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

    def _update_diffusion_agents_observation(
        self, iteration: SimulationIteration,
        next_iteration: SimulationIteration, history: SimulationHistoryBuffer,
        next_ego_state: Optional[EgoState],
        ego_future_trajectory: Optional[InterpolatedTrajectory]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:

        ego_agent_next_11_dim = None
        if next_ego_state is not None:
            interpol_time_points: List[
                TimePoint] = self._get_interpol_time_points(iteration)

            next_ego_plans: List[EgoState] = self._get_next_ego_plans(
                self._ego_anchor_state, next_ego_state, interpol_time_points)
            # (interpol_num, 11)
            ego_agent_next_11_dim = self._preprocess_next_ego_plans(
                next_ego_plans, self._ego_anchor_state)

        ego_agent_future_11_dim = None
        if ego_future_trajectory is not None:
            # (config.future_len, 11)
            ego_agent_future_11_dim = self._preprocess_ego_future_traj(
                ego_future_trajectory, self._ego_anchor_state)

        # Construct input features
        initialization = HorizonPlannerInitialization(
            # 시나리오가 끝나고도 계속 진행했을 때 최종적으로 도달해야 하는 포즈 (존재하지 않을 수도 있음)
            mission_goal=self._scenario.get_mission_goal(),
            # (x, y, yaw) 의 StateSE2
            route_roadblock_ids=self._scenario.get_route_roadblock_ids(),
            map_api=self._scenario.map_api,
            scenario=self._scenario,
            # 전문 운전자(ground truth)의 실제 마지막 상태 (항상 존재)
            expert_goal_state=self._scenario.get_expert_goal_state(),
            # (x, y, yaw) 의 StateSE2
        )
        traffic_light_data = self._scenario.get_traffic_light_status_at_iteration(
            next_iteration.index)
        # diffusion_agents_track_tokens: list[str]

        current_input = PlannerInput(
            next_iteration,
            history,
            traffic_light_data,
            # self.diffusion_agents_track_tokens,
            ego_agent_next_11_dim,
            ego_agent_future_11_dim)
        features: Dict[
            str, AbstractModelFeature] = self._model_loader.build_features(
                current_input, initialization)
        world_model_feature: Dict[
            str, np.ndarray] = self._model_loader.feature_builders[
                0].unnormalized_features
        neighbor_track_token: List[
            Optional[str]] = world_model_feature["neighbor_track_token"]
        # Infer model
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)
        token_to_future_traj_wrt_ego, token_to_refined_traj_wrt_ego, token_to_new_waypoint_array = self.infer_model(
            features, iteration, next_iteration, neighbor_track_token)

        return world_model_feature, token_to_future_traj_wrt_ego, token_to_refined_traj_wrt_ego, token_to_new_waypoint_array, neighbor_track_token

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
        self.observation_buffer: Deque[Observation] = history.observation_buffer
        # EgoState, Observation
        self._ego_anchor_state, self.current_observation = history.current_state
        self.current_iteration: int = next_iteration.index
        self.step_time_point: TimePoint = next_iteration.time_point - iteration.time_point
        # world_model_feature: Dict[str, np.ndarray]
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)

        (world_model_feature, token_to_future_traj_wrt_ego,
         token_to_refined_traj_wrt_ego, token_to_new_waypoint_array,
         neighbor_track_token) = self._update_diffusion_agents_observation(
             iteration, next_iteration, history, next_ego_state,
             ego_future_trajectory)
        if self._is_vis_features:
            pass
            # draw_machine.draw_world_model_to_png(world_model_feature,
            #                                      token_to_future_traj_wrt_ego,
            #                                      token_to_refined_traj_wrt_ego,
            #                                      token_to_new_waypoint_array,
            #                                      neighbor_track_token,
            #                                      self.current_token_to_np_history,
            #                                      self._vis_features_path)

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
                                                    self._step_interval,
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

    # [Add]
    # [Remove]
    def _get_token_to_traj_wrt_ego(
            self,
            agents: Dict[str, TrackedObject],
            near_future_tarjs_wrt_ego: np.ndarray,
            # (Pnn, 1+T, 4)
            ego_rear_axle_xy: np.ndarray,
            ego_yaw: float) -> Dict[str, np.ndarray]:  # (T, 4) # length == Pnn

        near_current_wrt_ego = near_future_tarjs_wrt_ego[:, 0, :]  # (Pnn, 4)
        # (Pnn, 2) # (Pnn, )
        global_xy, global_yaw = ego_to_global(near_current_wrt_ego,
                                              ego_rear_axle_xy, ego_yaw)
        global_xyyaw = np.concatenate([global_xy, global_yaw[:, None]],
                                      axis=1)  # (Pnn, 3)
        token_to_xy_yaw: Dict[str, np.ndarray] = {}
        for token, a_agent in agents.items():
            state_se2 = a_agent.center
            state_xy_yaw = np.array(
                [state_se2.x, state_se2.y, state_se2.heading])  # (3,)
            token_to_xy_yaw[token] = state_xy_yaw
        token_to_traj_wrt_ego: Dict[str, np.ndarray] = {}  # (T, 4)

        for a_global_xyyaw in global_xyyaw:
            """
            TODO: token_to_xy_yaw 중에서, a_global_xyyaw와  완벽히 일치하는 agent를 찾아야 함. (반드시 존재)
            """

    # [Add]
    from typing import Literal

    def infer_model(
            self, features: Dict[str, AbstractModelFeature],
            iteration: SimulationIteration, next_iteration: SimulationIteration,
            neighbor_track_token: List[Optional[str]]) -> Dict[str, np.ndarray]:
        ego_rear_axle_xy, ego_yaw = self.get_rear_axle_pose(
            self._ego_anchor_state)
        feature: AbstractModelFeature = features["world_model_feature"]
        # near_future_tarjs_wrt_ego: (Pnn, 1+T, 4)
        near_future_tarjs_wrt_ego: np.ndarray = self._model_loader.infer(
            feature).detach().cpu().numpy()
        Pnn, T_1, dim = near_future_tarjs_wrt_ego.shape
        assert T_1 == self.config.future_len + 1
        near_track_token = neighbor_track_token[:self.
                                                predicted_neighbor_num]  # len == Pnn
        # token_to_traj_wrt_ego: Dict[str, np.ndarray] # (T, 4)
        token_to_traj_wrt_ego: Dict[str, np.ndarray] = {}
        for idx, token in enumerate(near_track_token):
            if token is not None:
                a_near_future_tarjs_wrt_ego = near_future_tarjs_wrt_ego[
                    idx, 1:, :]  # (T, 4)
                token_to_traj_wrt_ego[token] = a_near_future_tarjs_wrt_ego
        # near_track_token: List[str] # len == Pnn
        near_track_token = list(token_to_traj_wrt_ego.keys())

        # 진짜 존재하는 대상만
        token_to_refined_traj_wrt_ego: Dict[str, np.ndarray] = {}  # (T, 11)

        current_token_to_agent_history: Dict[
            str, Deque[Agent]] = get_token_to_history(self.observation_buffer,
                                                      iteration,
                                                      near_track_token)

        anchor_ego_state = np.array([
            self._ego_anchor_state.rear_axle.x,
            self._ego_anchor_state.rear_axle.y,
            self._ego_anchor_state.rear_axle.heading
        ],
                                    dtype=np.float64)  # shape (3,)
        #########################
        self.current_token_to_np_history: Dict[str, np.ndarray] = {
        }  #(token, (len(history), 11))
        for token, history in current_token_to_agent_history.items():
            history_array = [
                agent_to_feature_vector(agent) for agent in history
            ]
            history_array = np.stack(history_array,
                                     axis=0)  # (len(history), 10)
            # convert absolute to relative

            history_array = convert_absolute_quantities_to_relative(
                history_array, anchor_ego_state)
            self.current_token_to_np_history[
                token] = history_array  # (len(history), 11)
        #########################
        current_agents: List[
            Agent] = self.current_observation.tracked_objects.get_agents()
        # current_agents: 쓰임
        current_token_to_agent: Dict[str, Agent] = {
            agent.track_token: agent
            for agent in current_agents
            if agent.track_token is not None
        }  # shape (M,)

        token_to_interpol_traj: Dict[str, AbstractTrajectory] = {}

        anchor_ego_state = np.array([
            self._ego_anchor_state.rear_axle.x,
            self._ego_anchor_state.rear_axle.y,
            self._ego_anchor_state.rear_axle.heading
        ],
                                    dtype=np.float64)  # shape (3,)
        step_s_time: float = self.step_time_point.time_s
        for token, future_traj_wrt_ego in token_to_traj_wrt_ego.items():
            # future_traj_wrt_ego: (T, 4)
            # future_traj_wrt_ego 값이 전부 0. 이면 무시
            if np.allclose(future_traj_wrt_ego, 0.0):
                raise ValueError("future_traj_wrt_ego 값이 전부 0. 입니다.")
            self_history: Deque[Agent] = current_token_to_agent_history[
                token]
            agent_ = self_history[-1]
            agent_center_xy, agent_yaw = self.get_npc_center_poses(
                agent_)
            future_traj_wrt_npc_center = transform_trajectory(
                future_traj_wrt_ego, ego_rear_axle_xy, ego_yaw, agent_center_xy,
                agent_yaw)  # (T, 4)

            future_trajectory = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(
                    future_traj_wrt_npc_center, self_history, step_s_time))
            ################ 그림 그리기 용 ############
            future_waypoints: List[
                Waypoint] = future_trajectory.get_sampled_trajectory()
            global_future_arrays = [
                waypoint_to_numpy10(wp) for wp in future_waypoints
            ]  # List[(10,)]
            global_future_arrays = np.stack(global_future_arrays,
                                            axis=0)  # (T, 10)
            ################
            # local_future_arrays: (T, 11)
            local_future_arrays = convert_absolute_quantities_to_relative(
                global_future_arrays,
                anchor_ego_state)  # anchor_ego_state: (3,)
            token_to_refined_traj_wrt_ego[token] = local_future_arrays
            ################
            token_to_interpol_traj[token] = future_trajectory

        token_to_new_agent: Dict[str, Agent] = {}
        # 진짜 존재하는 대상만
        token_to_new_waypoint_array: Dict[str, np.ndarray] = {}  # (1, 11)
        for agent_token, interpol_traj in token_to_interpol_traj.items():
            self_history: Deque[Agent] = current_token_to_agent_history[
                agent_token]
            agent_ = self_history[-1]
            new_timestamp_us = next_iteration.time_point.time_us
            new_metadata = SceneObjectMetadata(new_timestamp_us,
                                               agent_.metadata.token,
                                               agent_.metadata.track_id,
                                               agent_.metadata.track_token,
                                               agent_.metadata.category_name)

            # EgoState의 속도는 자차 좌표계 기준 벡터
            new_waypoint: Waypoint = interpol_traj.get_state_at_time(
                next_iteration.time_point)
            ################ 그림 그리기 용 ############
            new_waypoint_array = waypoint_to_numpy10(new_waypoint).reshape(
                1, -1)  # (1, 10)
            new_local_waypoint_array = convert_absolute_quantities_to_relative(
                new_waypoint_array, anchor_ego_state)  # anchor_ego_state: (3,)
            token_to_new_waypoint_array[
                agent_token] = new_local_waypoint_array  # (1, 11)

            ################
            new_agent = Agent(
                tracked_object_type=agent_.tracked_object_type,
                oriented_box=new_waypoint.oriented_box,
                velocity=new_waypoint.velocity,
                metadata=new_metadata,
            )
            new_agent.predictions = [
                PredictedTrajectory(
                    probability=1.,
                    waypoints=interpol_traj.get_sampled_trajectory())
            ]
            token_to_new_agent[agent_token] = new_agent

        self._diffusion_agents = token_to_new_agent

        # [Add] 첫 시점 (x,y) L2 거리가 작은 순서로 정렬
        token_to_refined_traj_wrt_ego = sort_refined_trajs_by_first_xy_l2(
            token_to_refined_traj_wrt_ego,
            invalid_eps=0.0,  # 0.0: (0,0)도 진짜 근거리로 취급 / 필요시 1e-6 ~ 1e-3로 조정
        )
        token_to_new_waypoint_array = sort_refined_trajs_by_first_xy_l2(
            token_to_new_waypoint_array,
            invalid_eps=0.0,  # 0.0: (0,0)도 진짜 근거리로 취급 / 필요시 1e-6 ~ 1e-3로 조정
        )
        # token_to_new_waypoint_array 의 각 values: (1, 11) -> (11,) 로 바꿔줌
        for token in token_to_new_waypoint_array.keys():
            token_to_new_waypoint_array[token] = token_to_new_waypoint_array[
                token].reshape(-1)
        # token_to_future_traj_wrt_ego: ego 좌표계 기준 차량 중심의 값 Dict (T, 4)
        return token_to_traj_wrt_ego, token_to_refined_traj_wrt_ego, token_to_new_waypoint_array

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        pass

    def _update_observation_with_predictions(
            self, agent_predictions: TargetsType) -> None:
        """
        Update smart agent using the predictions from the ML model
        :param agent_predictions: The prediction output from the ML_model
        """
        pass
