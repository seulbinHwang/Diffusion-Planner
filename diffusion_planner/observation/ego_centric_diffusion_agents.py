from typing import List, cast

import numpy as np
import numpy.typing as npt

from nuplan.common.actor_state.agent import Agent, PredictedTrajectory
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.waypoint import Waypoint
from nuplan.common.geometry.convert import numpy_array_to_absolute_pose, numpy_array_to_absolute_velocity
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.observation.abstract_ml_agents import AbstractMLAgents
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.features.agents_trajectories import AgentsTrajectories


def _convert_prediction_to_predicted_trajectory(
        agent: TrackedObject, poses: List[StateSE2],
        xy_velocities: List[StateVector2D],
        step_interval_us: float) -> PredictedTrajectory:
    """
    Convert each agent predictions into a PredictedTrajectory.
    :param agent: The agent the predictions are for.
    :param poses: A list of poses that makes up the predictions
    :param xy_velocities: A list of velocities in world frame corresponding to each pose.
    :return: The predictions parsed into PredictedTrajectory.
    """
    waypoints = [Waypoint(TimePoint(0), agent.box, agent.velocity)]
    waypoints += [
        Waypoint(
            # step + 1 because the first Waypoint is the current state.
            TimePoint(int((step + 1) * step_interval_us)),
            OrientedBox.from_new_pose(agent.box, pose),
            velocity,
        ) for step, (pose, velocity) in enumerate(zip(poses, xy_velocities))
    ]
    return PredictedTrajectory(1.0, waypoints)


class EgoCentricDiffusionAgents(AbstractMLAgents):
    """ Observations
    ModelLoader
        - initialize() <-> Diffusion_Planner
            - torch/model 초기화
        - build_features() <-> DataProcessor.observation_adapter()
        - infer()
    ModelLoader.TorchModuleWrapper (=self._model_loader._model)

    """

    def __init__(self, model: TorchModuleWrapper,
                 scenario: AbstractScenario) -> None:
        """
        Initializes the EgoCentricMLAgents class.
        :param model: Model to use for inference.
        :param scenario: scenario
        """
        super().__init__(model, scenario)
        self.prediction_type = 'agents_trajectory'

    @property
    def _ego_velocity_anchor_state(self) -> StateSE2:
        """
        Returns the ego's velocity state vector as an anchor state for transformation.
        :return: A StateSE2 representing ego's velocity state as an anchor state
        """
        ego_velocity = self._ego_anchor_state.dynamic_car_state.rear_axle_velocity_2d
        return StateSE2(ego_velocity.x, ego_velocity.y,
                        self._ego_anchor_state.rear_axle.heading)

    def _infer_model(self, features: FeaturesType) -> TargetsType:
        """
        FeaturesType = Dict[str, ModelInputFeature]
        {"processed_data": ModelInputFeature}
        """
        """Inherited, see superclass."""
        # Propagate model
        predictions = self._model_loader.infer(features)

        # Extract trajectory prediction
        if self.prediction_type not in predictions:
            raise ValueError(
                f"Prediction does not have the output '{self.prediction_type}'")
        # predictions[self.prediction_type]: AbstractModelFeature
        agents_prediction_tensor = cast(AgentsTrajectories,
                                        predictions[self.prediction_type]).data

        # Retrieve first (and only) batch as a numpy array
        agents_prediction = agents_prediction_tensor[0].cpu().detach().numpy()
        neighbor_selected_agents_token_str = predictions[
            "neighbor_selected_agents_token_str"]  # List[List[str]]
        return {
            self.prediction_type:
                AgentsTrajectories(
                    [cast(npt.NDArray[np.float32],
                          agents_prediction)]).get_agents_only_trajectories(),
            "neighbor_selected_agents_token_str":
                neighbor_selected_agents_token_str
        }

    def _update_observation_with_predictions(self,
                                             predictions: TargetsType) -> None:
        """Inherited, see superclass."""
        """
        각각 에이전트의 미래 위치와 속도를 절대 좌표로 변환합니다.
        생성된 미래 궤적을 바탕으로, 특정 시간(self.step_time)에 해당하는 에이전트의 
        상태(new_state)를 궤적에서 추출합니다.
        이 새로운 상태 정보를 이용해 새로운 에이전트 객체를 생성합니다
        """
        assert self._agents, "The agents have not been initialized. Please make sure they are initialized!"
        # predictions[self.prediction_type] : AgentsTrajectories
        agent_predictions = cast(AgentsTrajectories,
                                 predictions[self.prediction_type])
        neighbor_selected_agents_token_str = predictions[
            "neighbor_selected_agents_token_str"]  # List[List[str]]

        agent_predictions.reshape_to_agents()
        agent_poses = agent_predictions.poses[
            0]  # Fetch the first batch for the pose data # (num_agents, num_frames, 2)
        agent_velocities = agent_predictions.xy_velocity[
            0]  # (num_agents, num_frames, 2)
        a_neighbor_selected_agents_token_str = neighbor_selected_agents_token_str[
            0]  # List[str]
        # Remove agents that are not in a_neighbor_selected_agents_token_str.
        self._agents = {
            k: v
            for k, v in self._agents.items()
            if k in a_neighbor_selected_agents_token_str
        }
        for agent_token, poses_horizon, xy_velocity_horizon in zip(
                a_neighbor_selected_agents_token_str, agent_poses,
                agent_velocities):
            """
            agent_token : str
            agent : SceneObject(TrackedObject) 
            poses_horizon: (num_frames, 2)
            xy_velocity_horizon: (num_frames, 2)
            """
            """
            poses: List[StateSE2]
            xy_velocities: List[StateVector2D]
            """
            agent = self._agents.get(agent_token, None)
            if agent is None:
                continue
            # Convert to global coordinates
            poses = numpy_array_to_absolute_pose(
                self._ego_anchor_state.rear_axle, poses_horizon)
            xy_velocities = numpy_array_to_absolute_velocity(
                self._ego_velocity_anchor_state, xy_velocity_horizon)
            future_trajectory: PredictedTrajectory = _convert_prediction_to_predicted_trajectory(
                agent, poses, xy_velocities, self._step_interval_us)

            # Propagate agent according to simulation time
            new_state = future_trajectory.trajectory.get_state_at_time(
                self.step_time)

            new_agent = Agent(
                tracked_object_type=agent.tracked_object_type,
                oriented_box=new_state.oriented_box,
                velocity=new_state.velocity,
                metadata=agent.metadata,
            )
            new_agent.predictions = [future_trajectory]

            self._agents[agent_token] = new_agent
