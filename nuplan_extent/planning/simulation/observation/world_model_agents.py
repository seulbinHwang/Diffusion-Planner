from typing import cast, List, Dict, Optional
import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.convert import numpy_array_to_absolute_pose, numpy_array_to_absolute_velocity
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
from nuplan.planning.simulation.observation.observation_type import Observation


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

    def __init__(self, model: TorchModuleWrapper, scenario: AbstractScenario,
                 open_loop_detections_types: List[str], radius: float,
                 target_velocity: float) -> None:
        """
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
        self.plan_dt = 0.1  # [s] # TODO: remove hardcoding

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
        self.current_ego_state = None
        self.current_observation = None

        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        }
        self._diffusion_agents = sort_dict(unique_agents)
        self._log_replay_agents = sort_dict(
            self._get_open_loop_track_objects(self.current_iteration))
        self._agents = {**self._diffusion_agents, **self._log_replay_agents}

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

    def _get_next_relative_ego_pose(self, history: SimulationHistoryBuffer,
                                    ego_state: EgoState) -> StateSE2:
        # using current frame ego state instead of history.ego_states[-1]
        self._ego_anchor_state = history.ego_states[-1]
        next_global_pose = ego_state.rear_axle

        ego_to_global = self._ego_anchor_state.rear_axle.as_matrix()
        global_to_ego = np.linalg.inv(ego_to_global)
        next_relative_matrix = global_to_ego @ next_global_pose.as_matrix()
        next_relative_pose = StateSE2.from_matrix(next_relative_matrix)
        return next_relative_pose

    def _compute_sorted_distances(
            self, ego_state: EgoState,
            agents: Dict[str,
                         Agent]) -> tuple[list[str], npt.NDArray[np.float64]]:
        """ego와 각 agent 사이의 거리를 계산해 정렬된 결과를 반환한다.

        Args:
            ego_state (EgoState): 기준이 되는 ego 상태.
            agents (Dict[str, Agent]): 거리를 계산할 agent 사전.

        Returns:
            tuple[list[str], npt.NDArray[np.float64]]:
                - track token이 거리 오름차순으로 정렬된 리스트.
                - 정렬된 거리 배열로 shape (N,)이다.
        """
        if len(agents) == 0:
            return [], np.empty((0,), dtype=np.float64)

        agent_xy: npt.NDArray[np.float32] = np.array(
            [agent.center.point.array for agent in agents.values()],
            dtype=np.float32)  # shape (N, 2)
        ego_xy: npt.NDArray[np.float32] = np.expand_dims(
            ego_state.center.point.array,
            axis=0).astype(np.float32)  # shape (1, 2)
        distances: npt.NDArray[np.float64] = cdist(
            ego_xy, agent_xy).flatten()  # shape (N,)
        tokens: list[str] = list(agents.keys())
        sorted_indices: npt.NDArray[np.int64] = np.argsort(distances)
        sorted_tokens: list[str] = [tokens[i] for i in sorted_indices]
        sorted_distances: npt.NDArray[np.float64] = distances[sorted_indices]
        return sorted_tokens, sorted_distances

    def _filter_agents_out_of_range(self, ego_state: EgoState) -> None:
        """ego 기준 반경 내 가장 가까운 agent들을 선택한다.

        Args:
            ego_state (EgoState): 기준이 되는 ego 상태.
        """
        sorted_tokens, sorted_distances = self._compute_sorted_distances(
            ego_state, self._diffusion_agents)
        within_radius_tokens = [
            token for token, dist in zip(sorted_tokens, sorted_distances)
            if dist <= self._radius
        ]
        selected_tokens = within_radius_tokens[:self.predicted_neighbor_num]
        self._diffusion_agents = {
            token: self._diffusion_agents[token] for token in selected_tokens
        }

    def _get_interpol_time_points(
            self, iteration: SimulationIteration) -> List[TimePoint]:
        self.step_s_time: float = self.step_time.time_s
        """
        self.step_s_time : 0.15
        self.plan_dt : 0.1 이면
            q = 1.5 -> interpol_num = 2
        """
        q = Decimal(str(self.step_s_time)) / Decimal(str(self.plan_dt))
        interpol_num = int(q.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        interpol_num = max(interpol_num, 1)
        """
        if interpol_num = 2,
            interpol_indices = [1, 2]
            interpol_points_times = [0.1, 0.2]
            interpol_time_points = [TimePoint(ego_time + 0.1s), TimePoint(ego_time + 0.2s)]
        """
        interpol_indices = np.linspace(0,
                                       interpol_num,
                                       num=interpol_num + 1,
                                       dtype=int)[1:]  # (interpol_num, )
        interpol_points_times = interpol_indices * self.plan_dt  # (interpol_num, )
        interpol_time_points = []
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

    def _ego_plans_to_diffusion_array(
            self, next_ego_plans: List[EgoState],
            current_ego_state: EgoState) -> npt.NDArray[np.float64]:
        """ego 미래 상태들을 diffusion planner 입력 배열로 변환한다.

        Args:
            next_ego_plans (List[InterpolatableState]): 변환할 ego 상태 리스트.
            current_ego_state (EgoState): 기준이 되는 현재 ego 상태.

        Returns:
            npt.NDArray[np.float64]: (T, 11) 모양의 배열. 열 구성은
            [x_local, y_local, cos(yaw_local), sin(yaw_local), vx, vy,
            width, length, 1, 0, 0] 이다.
        """

        num_plans = len(next_ego_plans)
        absolute: npt.NDArray[np.float64] = np.zeros(
            (num_plans, 10), dtype=np.float64)  # shape (T, 10)
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
        # absolute: (T, 10)
        # relative: (T, 11)
        relative: np.ndarray = convert_absolute_quantities_to_relative(
            absolute, anchor, 'ego')  # shape (T, 11)
        return relative

    def _ego_future_to_diffusion_array(
            self, ego_future_trajectory: InterpolatedTrajectory,
            current_ego_state: EgoState) -> npt.NDArray[np.float64]:
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
        num_states = len(future_states)
        absolute: npt.NDArray[np.float64] = np.zeros(
            (num_states, 10), dtype=np.float64)  # shape (T, 7)
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
        return relative

    def _update_diffusion_agents_observation(
            self, iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            next_ego_state: Optional[EgoState],
            ego_future_trajectory: Optional[InterpolatedTrajectory]) -> None:
        self.current_iteration = next_iteration.index
        self.step_time = next_iteration.time_point - iteration.time_point
        # current_ego_state: EgoState
        # current_observation: Observation
        self.current_ego_state, self.current_observation = history.current_state

        ego_agent_next_11_dim = None
        if next_ego_state is not None:
            interpol_time_points = self._get_interpol_time_points(iteration)

            next_ego_plans = self._get_next_ego_plans(self.current_ego_state,
                                                      next_ego_state,
                                                      interpol_time_points)
            # (interpol_num, 11)
            ego_agent_next_11_dim = self._ego_plans_to_diffusion_array(
                next_ego_plans, self.current_ego_state)

        ego_agent_future_11_dim = None
        if ego_future_trajectory is not None:
            ego_agent_future_11_dim = self._ego_future_to_diffusion_array(
                ego_future_trajectory, self.current_ego_state)

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
        diffusion_agents_track_tokens, _ = self._compute_sorted_distances(
            self.current_ego_state, self._diffusion_agents)
        current_input = PlannerInput(next_iteration, history,
                                     traffic_light_data,
                                     diffusion_agents_track_tokens,
                                     ego_agent_next_11_dim.astype(np.float32),
                                     ego_agent_future_11_dim.astype(np.float32))
        features: Dict[
            str, AbstractModelFeature] = self._model_loader.build_features(
                current_input, initialization)

        # Infer model

        self._infer_model(features, diffusion_agents_track_tokens)

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
        ego_state = history.current_state[0]
        self._filter_agents_out_of_range(ego_state)
        self._update_diffusion_agents_observation(iteration, next_iteration,
                                                  history, next_ego_state,
                                                  ego_future_trajectory)
        self._log_replay_agents = sort_dict(
            self._get_open_loop_track_objects(self.current_iteration))
        self._agents = {**self._diffusion_agents, **self._log_replay_agents}

    def get_rear_wheelbases(
            self, agents: List[Agent],
            diffusion_agents_track_tokens: List[str]) -> List[float]:
        # TODO: current_agents 중에서, track_token에 해당하는 것만 추출해야함.
        """각 차량의 중심에서 뒷축까지 거리를 계산한다.

        Returns:
            List[float]: shape (N,) 각 차량의 뒷축까지 거리 [m].

        """
        rear_wheelbases: List[float] = []
        for agent in agents:
            box = agent.box
            rear_wheelbases.append(float(box.rear_axle_to_center_dist))
        return rear_wheelbases

    def _infer_model(self, features: FeaturesType,
                     diffusion_agents_track_tokens: List[str]) -> None:
        # npc_future_trajectories: (Pnn, T, 4)
        """
        TODO: self._model_loader.infer 가 track_token 추출해야함.

        """
        npc_future_trajectories: torch.Tensor = self._model_loader.infer(
            features)
        """
        TODO: npc_future_trajectories 는 x, y, cos(yaw), sin(yaw) 로 되어있음.
        그런데, 각 차량의 중심에 대한 x, y, yaw 값임. (ego 좌표계 기준)
        나는 npc_future_trajectories를, 각 챠량의 rear_axle 좌표계 기준으로 바꾸고 싶음.
        
        self.current_observation 을 사용해서.
        """
        current_agents: List[
            Agent] = self.current_observation.tracked_objects.get_agents()
        rear_wheelbases = self.get_rear_wheelbases(
            current_agents, diffusion_agents_track_tokens)
        # npc_future_trajectories from torch.Tensor to numpy
        npc_future_trajectories = npc_future_trajectories.detach().numpy(
        )  # (Pnn, T, 4)
        for npc_idx in range(npc_future_trajectories.shape[0]):
            traj = npc_future_trajectories[npc_idx].cpu().numpy()  # (T, 4)
            rear_wheelbase = rear_wheelbases[npc_idx]
            traj = convert_center_to_rear_axle(traj, rear_wheelbase)  # (T, 4)
            # ego 뒷축 좌표계 → vehicle 뒷축 좌표계 로 일괄 변환
            future_traj_wrt_npc_rear = transform_trajectory(
                traj, ego_rear_axle_xy, ego_yaw,
                np.array(veh.rear_axle_xy, dtype=np.float32), veh.heading_theta)

        for agent_token, agent_prediction in predictions.items():
            agent_meta = self._diffusion_agents[agent_token]
            new_state: EgoState = agent_prediction.get_state_at_time(
                self.step_time)
            new_agent = Agent(
                tracked_object_type=agent_meta.tracked_object_type,
                oriented_box=new_state.car_footprint,
                velocity=new_state.dynamic_car_state.center_velocity_2d,
                metadata=agent_meta.metadata,
            )
            new_agent.predictions = [
                PredictedTrajectory(
                    probability=1.,
                    waypoints=agent_prediction.get_sampled_trajectory())
            ]

            self._diffusion_agents[agent_token] = new_agent
