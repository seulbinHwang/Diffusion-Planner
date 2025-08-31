from typing import cast, List, Dict, Optional
import numpy as np
import numpy.typing as npt

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
from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.common.actor_state.tracked_objects import TrackedObjects
# /Users/user/PycharmProjects/nuplan-devkit/nuplan/common/actor_state/tracked_objects.py


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
        self.current_iteration = 0
        self._open_loop_detections_types: List[TrackedObjectType] = []
        self._initialize_open_loop_detection_types(open_loop_detections_types)
        self._radius = radius
        self._target_velocity = target_velocity

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

        unique_agents = {
            tracked_object.track_token: tracked_object
            for tracked_object in
            self._scenario.initial_tracked_objects.tracked_objects
            if tracked_object.tracked_object_type == TrackedObjectType.VEHICLE
        }
        self._diffusion_agents = sort_dict(unique_agents)
        self._log_replay_agents = sort_dict(self._get_open_loop_track_objects(
            self.current_iteration))
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

    def update_observation(
            self,
            iteration: SimulationIteration,
            next_iteration: SimulationIteration,
            history: SimulationHistoryBuffer,
            ego_state: Optional[EgoState] = None,
            ego_future_trajectory: Optional[InterpolatedTrajectory] = None
    ) -> None:
        """
        - 자동차
            - ego 기준, radius 안에 들어오면 -> diffusion 생성 대상
            - ego 기준, radius 밖에 있으면 -> 삭제. 관리 안함
        -
        """
        self.current_iteration = next_iteration.index
        if ego_state is None:
            next_relative_pose = None
        else:
            next_relative_pose = self._get_next_relative_ego_pose(
                history, ego_state)
        self.step_time = next_iteration.time_point - iteration.time_point

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
        current_input = PlannerInput(next_iteration, history,
                                     traffic_light_data)
        features: Dict[
            str, AbstractModelFeature] = self._model_loader.build_features(
                current_input, initialization)

        # Infer model
        next_ego_state = np.array([
            next_relative_pose.x, next_relative_pose.y,
            next_relative_pose.heading
        ])
        features["next_ego_state"] = next_ego_state[None, None, :]
        predictions = self._infer_model(features)

    def _infer_model(self,
                     features: FeaturesType) -> Dict[str, AbstractModelFeature]:
        """Inherited, see superclass."""
        predictions: Dict[str, AbstractModelFeature] = self._model_loader.infer(
            features)
        # TODO: predictions 를 "track_token" : InterpolatedTrajectory(AbstractModelFeature) 형태로?
        """
        
        """
