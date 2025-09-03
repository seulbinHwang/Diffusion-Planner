from typing import cast, List, Dict, Optional, Deque, Tuple
import numpy as np
import numpy.typing as npt
import torch
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.car_footprint import CarFootprint
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from collections import deque
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.simulation.observation.abstract_ml_agents import AbstractMLAgents
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import sort_dict
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES, TrackedObjectType
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan_extent.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from scipy.spatial.distance import cdist
# /Users/user/PycharmProjects/nuplan-devkit/nuplan/common/actor_state/tracked_objects.py
from nuplan.common.utils.interpolatable_state import InterpolatableState
from decimal import Decimal, ROUND_HALF_UP
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.common.actor_state.dynamic_car_state import DynamicCarState
from nuplan.planning.simulation.observation.observation_type import Observation


def observations_to_agents_buffer(
        observations_buffer: Deque[Observation]) -> Deque[List[Agent]]:
    """Observation 버퍼를 Agent 리스트 버퍼로 변환한다.

    Args:
        observations_buffer (Deque[Observation]): 시간 순서로 정렬된 관측 버퍼.

    Returns:
        Deque[List[Agent]]: 각 관측에서 추출한 Agent 리스트를 저장한 버퍼.

    Raises:
        TypeError: 관측이 ``DetectionsTracks`` 타입이 아닐 경우.
    """
    agents_buffer: Deque[List[Agent]] = deque(maxlen=observations_buffer.maxlen)
    for observation in observations_buffer:
        if isinstance(observation, DetectionsTracks):
            agents_buffer.append(observation.tracked_objects.get_agents())
        else:
            # [FIX] 더 친절한 에러 메시지 (혹은 스킵을 원하면 continue)
            raise TypeError(f"observations_buffer는 DetectionsTracks만 포함해야 합니다. "
                            f"받은 타입: {type(observation)}")
    return agents_buffer


def build_current_ego_state_histories(
        agents_buffer: Deque[List[Agent]]) -> Dict[str, Deque[EgoState]]:
    """현재 시점에 존재하는 Agent들의 과거 기록을 생성한다.

    버퍼의 마지막 원소(현재 시점)에 존재하는 Agent들만을 대상으로 하며, 시간 역순(현재→과거)으로
    각 Agent의 상태를 수집한다. 특정 과거시점에 해당 Agent가 존재하지 않으면, 더 이상 과거 상태는
    수집하지 않는다.
    TODO: 이 방식이 최선인가?

    Args:
        agents_buffer (Deque[List[Agent]]): 시간 순서대로 정렬된 Agent 버퍼. 길이 :math:`T`의 버퍼이며,
            각 시점마다 임의 길이의 ``Agent`` 리스트를 포함한다.

    Returns:
        Dict[str, Deque[EgoState]]: 현재 시점에 존재하는 Agent 수 :math:`N` 만큼의 리스트를 반환한다. 각 내부
            리스트는 길이 :math:`T`이며, 시간 역순(현재→과거)으로 해당 Agent의 히스토리를 담고 있다.
    """
    max_len = agents_buffer.maxlen
    if not agents_buffer:
        return {}
    current_agents: List[Agent] = [
        agent for agent in agents_buffer[-1] if agent.track_token is not None
    ]
    current_token_to_idx = {
        agent.track_token: idx for idx, agent in enumerate(current_agents)
    }
    token_to_history: Dict[str, Deque] = {
        agent.track_token: deque(maxlen=max_len) for agent in current_agents
    }
    # 현재 시점 상태 추가
    for agent in current_agents:
        token_to_history[agent.track_token].append(agent)
    # max_len
    for past_agents in reversed(list(agents_buffer)[:-1]):
        # 가장 최근 -> 가장 오래된 순서로 과거 시점 상태 추가
        past_agent_lookup: Dict[str, Agent] = {
            agent.track_token: agent for agent in past_agents
        }
        for current_token in current_token_to_idx.keys():
            agent_history: Deque[Agent] = token_to_history[current_token]
            if agent_history[-1] is None:
                continue  # 이미 이 토큰의 히스토리 수집 종료
            history_agent = past_agent_lookup.get(current_token, None)
            token_to_history[current_token].append(history_agent)
    # histories를 시간 순서(과거→현재)로 뒤집기
    for token, history in token_to_history.items():
        history.reverse()
        # 만약 history 의 첫 원소가 None이면, 제거한다.
        if history[0] is None:
            history.popleft()
        converted = deque((agent_to_ego_state(agent) for agent in history),
                          maxlen=history.maxlen)
        token_to_history[token] = converted
    return token_to_history


def agent_to_ego_state(
    agent: Agent,
    tire_steering_angle: float = 0.0,
    is_in_auto_mode: bool = False,
) -> EgoState:
    """Agent 인스턴스를 EgoState로 변환합니다.

    Args:
        agent (Agent): CarFootprint 정보를 포함한 에이전트.
        tire_steering_angle (float): 타이어 조향각 [rad]. 기본값은 0.0.
        is_in_auto_mode (bool): 자율 주행 모드 여부. 기본값은 False.

    Returns:
        EgoState: 생성된 EgoState 인스턴스.

    Raises:
        TypeError: ``agent.box``가 ``CarFootprint``가 아닌 경우.
    """

    if not isinstance(agent.box, CarFootprint):
        raise TypeError("agent.box는 CarFootprint 타입이어야 합니다.")

    car_footprint: CarFootprint = agent.box
    vehicle_params: VehicleParameters = car_footprint.vehicle_parameters

    rear_axle_velocity: StateVector2D = agent.velocity or StateVector2D(
        0.0, 0.0)
    rear_axle_acceleration: StateVector2D = StateVector2D(0.0, 0.0)
    angular_velocity = (agent.angular_velocity
                        if agent.angular_velocity is not None else 0.0)
    dynamic_car_state = DynamicCarState.build_from_rear_axle(
        rear_axle_to_center_dist=vehicle_params.rear_axle_to_center,
        rear_axle_velocity_2d=rear_axle_velocity,
        rear_axle_acceleration_2d=rear_axle_acceleration,
        angular_velocity=angular_velocity,
    )

    return EgoState(
        car_footprint=car_footprint,
        dynamic_car_state=dynamic_car_state,
        tire_steering_angle=tire_steering_angle,
        is_in_auto_mode=is_in_auto_mode,
        time_point=TimePoint(agent.metadata.timestamp_us),
    )


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


def transform_trajectory(npc_traj_wrt_ego_rear: np.ndarray,
                         ego_rear_axle_xy: np.ndarray, ego_yaw: float,
                         veh_rear_axle_xy: np.ndarray,
                         veh_yaw: float) -> np.ndarray:
    """
    ego계 기준 npc_traj_wrt_ego_rear → 각 vehicle 로컬계 기준 (T,4) trajectory.
    """
    coords_g, yaw_g = ego_to_global(npc_traj_wrt_ego_rear, ego_rear_axle_xy,
                                    ego_yaw)
    return global_to_local(coords_g, yaw_g, veh_rear_axle_xy, veh_yaw)


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


class WorldModelAgents(AbstractMLAgents):
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
        Initializes the WorldModelAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.config = model.config
        self.predicted_neighbor_num = self.config.predicted_neighbor_num
        self.current_iteration = 0
        self._open_loop_detections_types: List[TrackedObjectType] = []
        self._initialize_open_loop_detection_types(open_loop_detections_types)
        self._radius = radius
        self._target_velocity = target_velocity
        self._step_interval = self._step_interval_us / 1e6  # [s]

    def _initialize_open_loop_detection_types(
            self, open_loop_detections: List[str]) -> None:
        """
        Initializes open-loop detections with the enum types from TrackedObjectType
        :param open_loop_detections: A list of open-loop detections types as strings
        :return: A list of open-loop detections types as strings as the corresponding TrackedObjectType
        """
        for _type in open_loop_detections:
            try:
                self._open_loop_detections_types.append(
                    TrackedObjectType[_type])
            except KeyError:
                raise ValueError(
                    f"The given detection type {_type} does not exist or is not supported!"
                )

    def _initialize_agents(self) -> None:
        """
        Initializes the agents based on the first step of the scenario
        """
        self.current_iteration = 0
        self.current_observation = None

        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        }
        self._diffusion_agents = sort_dict(unique_agents)
        self._filter_agents_out_of_range(self._ego_anchor_state)
        self._log_replay_agents = sort_dict(
            self._get_open_loop_track_objects(self.current_iteration))
        self._agents: Dict[str, TrackedObject] = {
            **self._diffusion_agents,
            **self._log_replay_agents
        }

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

    def _compute_sorted_distances(
            self, ego_state: EgoState,
            agents: Dict[str,
                         Agent]) -> tuple[list[str], npt.NDArray[np.float64]]:
        """ego와 각 agent 사이의 거리를 계산해 정렬된 결과를 반환한다.

        Args:
            ego_state (EgoState): 기준이 되는 ego 상태.
            agents (Dict[str, Agent]): 거리를 계산할 agent 사전. 길이 = N.

        Returns:
            tuple[list[str], npt.NDArray[np.float64]]:
                - track token이 거리 오름차순으로 정렬된 리스트.
                - 정렬된 거리 배열로 shape (N,)이다.
        """
        if len(agents) == 0:
            return [], np.empty((0,), dtype=np.float64)
        tokens: list[str] = list(agents.keys())  # len(tokens) == len(distances)
        agent_xy: npt.NDArray[np.float32] = np.array(
            [agents[token].center.point.array for token in tokens],
            dtype=np.float32)  # shape (N, 2)
        ego_xy: npt.NDArray[np.float32] = np.expand_dims(
            ego_state.center.point.array,
            axis=0).astype(np.float32)  # shape (1, 2)
        distances: npt.NDArray[np.float64] = cdist(
            ego_xy, agent_xy).flatten()  # shape (N,)
        sorted_indices: npt.NDArray[np.int64] = np.argsort(distances)
        sorted_tokens: list[str] = [tokens[i] for i in sorted_indices]
        sorted_distances: npt.NDArray[np.float64] = distances[sorted_indices]
        return sorted_tokens, sorted_distances

    def _filter_agents_out_of_range(self, ego_state: EgoState) -> None:
        """ego 기준 반경 내, 가장 가까운 self.predicted_neighbor_num 대의 agent들만
        선택하고 나머지는 버린다.

        Args:
            ego_state (EgoState): 기준이 되는 ego 상태.
        """
        # sorted_tokens: list[str] # len == N
        # sorted_distances: np.ndarray (N,)
        sorted_tokens, sorted_distances = self._compute_sorted_distances(
            ego_state, self._diffusion_agents)
        within_radius_tokens: List[str] = [
            token for token, dist in zip(sorted_tokens, sorted_distances)
            if dist <= self._radius
        ]
        selected_tokens: List[
            str] = within_radius_tokens[:self.predicted_neighbor_num]
        self._diffusion_agents = {
            token: self._diffusion_agents[token] for token in selected_tokens
        }

    def _get_interpol_time_points(
            self, iteration: SimulationIteration) -> List[TimePoint]:
        step_s_time: float = self.step_time_point.time_s
        """
        step_s_time : 0.15 (시뮬레이션 시간 간격 (초))
        self._step_interval : 0.1 이면 (future trajectory의 점 사이 시간 간격)
            q = 1.5 -> interpol_num = 2
        """
        q = Decimal(str(step_s_time)) / Decimal(str(self._step_interval))
        interpol_num = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        interpol_num = max(interpol_num, 1)
        """
        if interpol_num = 2,
            interpol_indices = [1, 2]
            interpol_points_times = [0.1, 0.2]
            interpol_time_points = [TimePoint(current_time + 0.1s), TimePoint(current_time + 0.2s)]
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
            absolute[i, 0] = state.center.x
            absolute[i, 1] = state.center.y
            absolute[i, 2] = state.center.heading
            absolute[i, 3] = state.dynamic_car_state.center_velocity_2d.x
            absolute[i, 4] = state.dynamic_car_state.center_velocity_2d.y
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
            npt.NDArray[np.float64]: (T, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """

        future_states: List[EgoState] = list(
            ego_future_trajectory.get_sampled_trajectory())
        future_len__plus_1 = len(future_states)
        absolute: npt.NDArray[np.float64] = np.zeros(
            (future_len__plus_1, 10), dtype=np.float64)  # shape (T, 7)
        absolute[:, 7] = 1  # is vehicle

        for i, state in enumerate(future_states):
            absolute[i, 0] = state.center.x
            absolute[i, 1] = state.center.y
            absolute[i, 2] = state.center.heading
            absolute[i, 3] = state.dynamic_car_state.center_velocity_2d.x
            absolute[i, 4] = state.dynamic_car_state.center_velocity_2d.y
            absolute[i, 5] = state.car_footprint.width
            absolute[i, 6] = state.car_footprint.length

        anchor = np.array([
            current_ego_state.rear_axle.x,
            current_ego_state.rear_axle.y,
            current_ego_state.rear_axle.heading,
        ],
                          dtype=np.float32)  # shape (3,)

        relative: np.ndarray = convert_absolute_quantities_to_relative(
            absolute, anchor, 'ego')  # shape (T, 11)
        assert (future_len__plus_1, 11) == relative.shape
        return relative.astype(np.float32)

    def _update_diffusion_agents_observation(
            self, iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            next_ego_state: Optional[EgoState],
            ego_future_trajectory: Optional[InterpolatedTrajectory]) -> None:

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
        # Infer model
        self.infer_model(features, next_iteration)

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
        self._ego_anchor_state, self.current_observation = history.current_state
        self.current_iteration = next_iteration.index
        self.step_time_point = next_iteration.time_point - iteration.time_point
        self._update_diffusion_agents_observation(iteration, next_iteration,
                                                  history, next_ego_state,
                                                  ego_future_trajectory)
        self._filter_agents_out_of_range(self._ego_anchor_state)
        self._log_replay_agents = sort_dict(
            self._get_open_loop_track_objects(self.current_iteration))
        self._agents = {**self._diffusion_agents, **self._log_replay_agents}

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

    def outputs_to_trajectory(
            self, future_traj_wrt_npc_rear: np.ndarray,
            ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:
        heading = np.arctan2(future_traj_wrt_npc_rear[:, 3],
                             future_traj_wrt_npc_rear[:, 2])[..., None]
        future_traj_wrt_npc_rear = np.concatenate(
            [future_traj_wrt_npc_rear[..., :2], heading], axis=-1)

        states = transform_predictions_to_states(future_traj_wrt_npc_rear,
                                                 ego_state_history,
                                                 self._future_horizon,
                                                 self._step_interval)
        return states

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

    def get_rear_axle_poses(self,
                            current_agent: Agent) -> Tuple[np.ndarray, float]:
        """각 차량의 뒷축 위치와 방향을 계산한다.

        Returns:
            Tuple[np.ndarray, float]: shape (2,), 차량 뒷축 [x, y]와 방향 yaw_rad.

        """
        box = current_agent.box
        assert isinstance(box, CarFootprint)
        rear_axle = box.rear_axle
        position: np.ndarray = np.array([
            rear_axle.x,
            rear_axle.y,
        ],
                                        dtype=float)  # shape: (2,)

        return position, rear_axle.heading

    def infer_model(self, features: Dict[str, AbstractModelFeature],
                    next_iteration: SimulationIteration) -> None:
        """


        """
        feature: AbstractModelFeature = features["world_model_feature"]
        # near_future_tarjs_wrt_ego: (Pnn, T, 4)
        near_future_tarjs_wrt_ego: np.ndarray = self._model_loader.infer(
            feature).detach().cpu().numpy()
        # list[str]
        self.diffusion_agents_track_tokens, _ = self._compute_sorted_distances(
            self._ego_anchor_state, self._diffusion_agents)

        token_to_future_traj_wrt_ego: Dict[str, np.ndarray] = {}
        near_number = near_future_tarjs_wrt_ego.shape[0]
        assert near_number == self.predicted_neighbor_num
        assert near_number <= len(self.diffusion_agents_track_tokens)
        for idx, token in enumerate(self.diffusion_agents_track_tokens):
            token_to_future_traj_wrt_ego[token] = near_future_tarjs_wrt_ego[idx]

        # Deque[EgoState]
        agents_buffer: Deque[List[Agent]] = observations_to_agents_buffer(
            self.observation_buffer)
        token_to_history: Dict[
            str,
            Deque[EgoState]] = build_current_ego_state_histories(agents_buffer)
        current_agents: List[
            Agent] = self.current_observation.tracked_objects.get_agents()
        # current_agents: 쓰임
        current_token_to_agent: Dict[str, Agent] = {
            agent.track_token: agent
            for agent in current_agents
            if agent.track_token is not None
        }  # shape (M,)

        token_to_rear_wheelbase: Dict[str, float] = self.get_rear_wheelbases(
            current_token_to_agent, self.diffusion_agents_track_tokens)
        ego_rear_axle_xy, ego_yaw = self.get_rear_axle_pose(
            self._ego_anchor_state)
        token_to_interpol_traj: Dict[str, AbstractTrajectory] = {}
        # (Pnn)
        for token, future_traj_wrt_ego in token_to_future_traj_wrt_ego.items():
            # future_traj_wrt_ego: (T, 4)
            # future_traj_wrt_ego 값이 전부 0. 이면 무시
            if np.allclose(future_traj_wrt_ego, 0.0):
                continue
            rear_wheelbase = token_to_rear_wheelbase[token]
            future_traj_wrt_ego = convert_center_to_rear_axle(
                future_traj_wrt_ego, rear_wheelbase)  # (T, 4)
            # ego 뒷축 좌표계 → vehicle 뒷축 좌표계 로 일괄 변환
            agent_rear_axle_xy, agent_yaw = self.get_rear_axle_poses(
                current_token_to_agent[token])
            future_traj_wrt_npc_rear = transform_trajectory(
                future_traj_wrt_ego, ego_rear_axle_xy, ego_yaw,
                agent_rear_axle_xy, agent_yaw)  # (T, 4)
            self_history: Deque[EgoState] = token_to_history[token]
            future_trajectory = InterpolatedTrajectory(
                trajectory=self.outputs_to_trajectory(future_traj_wrt_npc_rear,
                                                      self_history))
            token_to_interpol_traj[token] = future_trajectory

        new_agents = {}
        for agent_token, interpol_traj in token_to_interpol_traj.items():
            agent_ = self._diffusion_agents[agent_token]
            # TODO: next_iteration.time_point 가 맞나? self.step_time 이 맞나?
            new_state: EgoState = interpol_traj.get_state_at_time(
                next_iteration.time_point)
            new_agent = Agent(
                tracked_object_type=agent_.tracked_object_type,
                oriented_box=new_state.car_footprint,
                velocity=new_state.dynamic_car_state.center_velocity_2d,
                metadata=agent_.metadata,
            )
            new_agent.predictions = [
                PredictedTrajectory(
                    probability=1.,
                    waypoints=interpol_traj.get_sampled_trajectory())
            ]
            new_agents[agent_token] = new_agent

        self._diffusion_agents = new_agents
