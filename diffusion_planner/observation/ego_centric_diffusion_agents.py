from typing import List, cast

import numpy as np
import numpy.typing as npt
import math
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
from nuplan.planning.simulation.controller.tracker.tracker_utils import get_velocity_curvature_profiles_with_derivatives_from_poses
from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.planning.simulation.planner.ml_planner.model_loader import ModelLoader

np.set_printoptions(suppress=True)
np.set_printoptions(suppress=True, precision=1)


def wrap_to_pi(angle):
    """
    angle을 [-π, π) 범위에 맞게 wrap-around.
    """
    return (angle + math.pi) % (2 * math.pi) - math.pi


def interpolate_angle(heading_agent, heading_lookahead, interpolation_ratio):
    """
    heading_agent에서 heading_lookahead 방향으로 interpolation_ratio만큼 보간하여
    angle wrap-around를 고려한 보간값을 반환한다.
    """
    # 1. 두 각의 차이를 [-π, π) 범위로 조정
    angle_diff = wrap_to_pi(heading_lookahead - heading_agent)

    # 2. 선형 보간
    interpolated = heading_agent + interpolation_ratio * angle_diff

    # 3. 보간된 각도 역시 [-π, π) 범위로 wrap
    return wrap_to_pi(interpolated)

def _convert_prediction_to_predicted_trajectory(
        agent: TrackedObject, poses: List[StateSE2],
        xy_velocities: List[StateVector2D],
        step_interval_us: float,
        include_current_point: bool = False) -> PredictedTrajectory:
    """
    Convert each agent predictions into a PredictedTrajectory.
    :param agent: The agent the predictions are for.
    :param poses: A list of poses that makes up the predictions
    :param xy_velocities: A list of velocities in world frame corresponding to each pose.
    :return: The predictions parsed into PredictedTrajectory.
    """
    # waypoints = [Waypoint(TimePoint(0), agent.box, agent.velocity)]
    if include_current_point:
        waypoints = [Waypoint(TimePoint(0), agent.box, agent.velocity)]
        waypoints += [
            Waypoint(
                # step + 1 because the first Waypoint is the current state.
                TimePoint(int((step + 1) * step_interval_us)),
                OrientedBox.from_new_pose(agent.box, pose),
                velocity,
            ) for step, (pose, velocity) in enumerate(zip(poses, xy_velocities))
        ]
    else:
        waypoints = []
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
        self.idx = 0

    @property
    def _ego_velocity_anchor_state(self) -> StateSE2:
        """
        Returns the ego's velocity state vector as an anchor state for transformation.
        :return: A StateSE2 representing ego's velocity state as an anchor state
        """
        ego_velocity = self._ego_anchor_state.dynamic_car_state.rear_axle_velocity_2d
        return StateSE2(0., 0.,
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
        agents_prediction = agents_prediction_tensor[0].cpu().detach().numpy(
        )  # (num_frames, num_agents, 6)
        neighbor_selected_agents_token_str = predictions[
            "neighbor_selected_agents_token_str"]  # List[List[str]]
        # (num_frames, num_agents, 6)
        agents_trajectories = AgentsTrajectories(
            [cast(npt.NDArray[np.float32], agents_prediction)])
        # agent_predictions = agents_trajectories.get_agents_only_trajectories()
        return_ = {
            self.prediction_type:
                agents_trajectories,  # Exclude current state
            "neighbor_selected_agents_token_str":
                neighbor_selected_agents_token_str,
            "past_agents_trajectory":
                predictions["past_agents_trajectory"], # [P, V_past, 6]
        }
        return return_


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
        past_agents_trajectory = predictions["past_agents_trajectory"] # [P, V_past, 6]
        agent_predictions.reshape_to_agents()  # (agents, num_frames, 6)
        a = agent_predictions.data[0]  # (agents, num_frames, 6)
        agent_poses = agent_predictions.poses[
            0]  # Fetch the first batch for the pose data # (num_agents, num_frames, 3)

        agent_velocities = agent_predictions.xy_velocity[
            0]  # (num_agents, num_frames, 2)
        a_neighbor_selected_agents_token_str = neighbor_selected_agents_token_str[
            0]  # List[str]
        # Remove agents that are not in a_neighbor_selected_agents_token_str.
        agents = {
            k: v
            for k, v in self._agents.items()
            if k in a_neighbor_selected_agents_token_str
        }
        self._agents = agents
        assert self._step_interval_us == 0.1 * 1e6, "The step interval is not 0.1 seconds!"
        step_dt_s = float(
            self._step_interval_us) * 1e-6  # microseconds -> seconds
        idx = 0
        for agent_token, poses_horizon, xy_velocity_horizon in zip(
                a_neighbor_selected_agents_token_str, agent_poses,
                agent_velocities):
            """
            agent_token : str
            agent : SceneObject(TrackedObject) 
            poses_horizon: (num_frames, 3)
            xy_velocity_horizon: (num_frames, 2)
            """
            """
            poses: List[StateSE2]
            xy_velocities: List[StateVector2D]
            """
            agent = self._agents.get(agent_token, None)
            if agent is None:
                continue
            # velocity_profile, _, _, _ = get_velocity_curvature_profiles_with_derivatives_from_poses(
            #     discretization_time=step_dt_s,
            #     poses=poses_horizon,  # (num_frames, 3)
            #     jerk_penalty=1e-4,
            #     curvature_rate_penalty=1e-2,
            # )
            # # velocity_profile (num_frames-1, ) -> (num_frames, )
            # velocity_profile = np.concatenate(
            #     (velocity_profile, velocity_profile[-1:]))
            past_agent_trajectory = past_agents_trajectory[idx]  # [V_past, 6]
            past_xy_trajectory = past_agent_trajectory[:, :2]  # [V_past, 2]
            past_cos_theta = past_agent_trajectory[:, 2]  # [V_past, ]
            past_sin_theta = past_agent_trajectory[:, 3]  # [V_past, ]
            past_heading = np.arctan2(past_sin_theta, past_cos_theta)  # [V_past, ]
            past_poses_horizon = np.concatenate(
                (past_xy_trajectory, past_heading[:, None]), axis=1)  # [V_past, 3]
            past_future_poses_horizon = np.concatenate(
                (past_poses_horizon, poses_horizon), axis=0)  # [V_past + V, 3]
            past_vxy_horizon = past_agent_trajectory[:, 4:6]  # [V_past, 2]
            past_future_vxy_horizon = np.concatenate(
                (past_vxy_horizon, xy_velocity_horizon), axis=0)  # [V_past + V, 2]
            # Convert to global coordinates
            past_future_poses: List[StateSE2] = numpy_array_to_absolute_pose(
                self._ego_anchor_state.rear_axle, past_future_poses_horizon)
            past_future_poses_np = []
            for pose in past_future_poses:
                a = np.array(pose.serialize()) # [x, y, heading] (3, )
                past_future_poses_np.append(a)
            past_future_poses_np = np.array(past_future_poses_np) # [V_past + V, 3]
            past_future_poses_np_xy = past_future_poses_np[...,:2] # [V_past + V, 2]
            past_future_poses_np_norm = np.sum(past_future_poses_np_xy, axis=1) # [V_past + V, ]

            past_future_xy_velocities: List[StateVector2D] = numpy_array_to_absolute_velocity(
                self._ego_velocity_anchor_state, past_future_vxy_horizon)
            past_future_trajectory_vxy_norm = []
            for velocity in past_future_xy_velocities:
                a = velocity.array # [vx, vy] (2, )
                past_future_trajectory_vxy_norm.append(a)
            past_future_trajectory_vxy_norm = np.array(past_future_trajectory_vxy_norm) # [V_past + V, 2]
            past_future_trajectory_vxy_norm = np.linalg.norm(past_future_trajectory_vxy_norm, axis=1) # [V_past + V, ]
            save_path = f"[{self.idx}]{idx}.png"
            # self._save_debug_graph(
            #     past_future_poses_np_norm, # [V]
            #     past_future_poses_np[...,2], # [V]
            #     past_future_trajectory_vxy_norm, # [V]
            #     save_path
            # )
            idx += 1
            # Convert to global coordinates
            poses: List[StateSE2] = numpy_array_to_absolute_pose(
                self._ego_anchor_state.rear_axle, poses_horizon)
            xy_velocities: List[StateVector2D] = numpy_array_to_absolute_velocity(
                self._ego_velocity_anchor_state, xy_velocity_horizon)


            # 이후 xy_velocities_global 에 미래 시점까지 글로벌 속도가 들어있음
            # --------------------------------------------------
            future_trajectory: PredictedTrajectory = _convert_prediction_to_predicted_trajectory(
                agent, poses, xy_velocities, self._step_interval_us)
            # future_trajectory_w_current: 미래 위치 예측 에 현재 상태를 추가하여 예측된 궤적 생성
            future_trajectory_w_current: PredictedTrajectory = _convert_prediction_to_predicted_trajectory(
                agent, poses, xy_velocities, self._step_interval_us, include_current_point=True)
            # Propagate agent according to simulation time
            """
            future_trajectory: PredictedTrajectory
            future_trajectory.trajectory: InterpolatedTrajectory(AbstractTrajectory)
            
            """
            future_point_horizon = 12
            look_ahead_time_point = TimePoint(
                int(future_point_horizon * self._step_interval_us))
            print("------------------------------------")
            print("look_ahead_time_point: ", look_ahead_time_point)
            # lookahead_state: Waypoint
            lookahead_state = future_trajectory.trajectory.get_state_at_time(
                look_ahead_time_point)
            lookahead_state_center_x: float = lookahead_state.x
            lookahead_state_center_y: float = lookahead_state.y
            lookahead_state_heading: float = lookahead_state.heading
            lookahead_state_position = np.array([
                lookahead_state_center_x, lookahead_state_center_y,
                lookahead_state_heading
            ])  # (3, )
            lookahead_state_velocity: StateVector2D = lookahead_state.velocity.array  # (2, )
            print("lookahead_state_position: ", lookahead_state_position)
            print("lookahead_state_velocity: ", lookahead_state_velocity*3.6)
            ## current agent position and velocity
            agent_center_x: float = agent.box.center.x
            agent_center_y: float = agent.box.center.y
            agent_heading: float = agent.box.center.heading
            agent_position = np.array(
                [agent_center_x, agent_center_y, agent_heading])  # (3, )
            agent_velocity = agent.velocity.array  # (2, )
            print("agent_position: ", agent_position)
            print("agent_velocity: ", agent_velocity*3.6)
            # interpolate between current and future state.
            print("self.step_time.time_us: ", self.step_time.time_us)
            print("int(future_point_horizon * self._step_interval_us): ",
                    int(future_point_horizon * self._step_interval_us))
            interpolation_ratio = self.step_time.time_us / int(
                future_point_horizon * self._step_interval_us)
            print("interpolation_ratio: ", interpolation_ratio)
            ###########
            # 기존에 lookahead_state_position은 [x, y, heading] shape (3,)
            # lookahead_state_velocity는 [vx, vy] shape (2,)

            # 1) x, y는 그대로 선형 보간
            delta_x = lookahead_state_position[0] - agent_position[0]
            delta_y = lookahead_state_position[1] - agent_position[1]
            interp_x = agent_position[0] + interpolation_ratio * delta_x
            interp_y = agent_position[1] + interpolation_ratio * delta_y

            # # 2) heading 보간 (angle wrap-around 고려)
            heading_agent = agent_position[2] # ~ [-pi, +pi) # float
            heading_lookahead = lookahead_state_position[2] # ~ [-pi, +pi)  # float
            #
            # # 2.1) heading 차이(lookahead - agent)를 [-pi, +pi) 범위로 wrap
            # heading_diff = heading_lookahead - heading_agent
            # heading_diff = math.atan2(math.sin(heading_diff),
            #                           math.cos(heading_diff))
            #
            # # 2.2) interpolation_ratio만큼 선형 보간
            # interp_heading = heading_agent + interpolation_ratio * heading_diff
            #
            # # 2.3) 최종 heading도 다시 [-pi, +pi) 범위로 wrap
            # interp_heading = math.atan2(math.sin(interp_heading),
            #                             math.cos(interp_heading))

            #######################
            interp_heading = interpolate_angle(heading_agent, heading_lookahead, interpolation_ratio)
            ########################

            interpolated_position = np.array(
                [interp_x, interp_y, interp_heading])
            print("interpolated_position: ", interpolated_position)
            # velocity는 그대로 선형 보간
            delta_v = lookahead_state_velocity - agent_velocity  # shape (2,)
            interpolated_velocity = agent_velocity + interpolation_ratio * delta_v  # shape (2,)
            print("interpolated_velocity: ", interpolated_velocity*3.6)
            pose = StateSE2(
                float(interpolated_position[0]),
                float(interpolated_position[1]),
                float(interpolated_position[2])
            )

            velocity = StateVector2D(
                float(interpolated_velocity[0]),
                float(interpolated_velocity[1])
            )

            ###########

            # new_state = future_trajectory.trajectory.get_state_at_time(
            #     self.step_time)
            next_waypoint = Waypoint(
                # step + 1 because the first Waypoint is the current state.
                TimePoint(int(self.step_time.time_us)),
                OrientedBox.from_new_pose(agent.box, pose),  # pose: StateSE2
                velocity,  # StateVector2D
            )
            new_agent = Agent(
                tracked_object_type=agent.tracked_object_type,
                oriented_box=next_waypoint.oriented_box,
                velocity=next_waypoint.velocity,
                metadata=agent.metadata,
            )
            new_agent.predictions = [future_trajectory_w_current]

            self._agents[agent_token] = new_agent
        self.idx += 1

        # if self.idx >= 2:
        #     raise NotImplementedError

    def _save_debug_graph(
            self,
            positions: np.ndarray,
            headings_rad: np.ndarray,
            velocities_mps: np.ndarray,
            output_path: str
    ) -> None:
        """
        positions : 임의 위치(가령 m 단위), shape (K,)
        headings_rad : 라디안 단위 heading, shape (N,)
        velocities_mps : m/s 단위 velocity, shape (M,)
        output_path : 결과 그래프를 저장할 파일 경로

        - 1행 3열 subplot 구조로 그림:
          (1) Heading subplot: 라디안 -> deg 변환
          (2) Velocity subplot: m/s -> km/h 변환
          (3) Position subplot: positions 그대로

        모든 subplot에서 x축은 index (0 ~ len-1)
        """
        import matplotlib.pyplot as plt
        import numpy as np




        # (1) Position: 그대로 사용
        x_position = np.arange(len(positions))  # 0..K-1

        # (2) Heading: rad -> deg
        headings_deg = np.degrees(headings_rad)
        x_heading = np.arange(len(headings_deg))  # 0..N-1

        # (3) Velocity: m/s -> km/h
        velocities_kmh = velocities_mps * 3.6
        x_velocity = np.arange(len(velocities_kmh))  # 0..M-1

        # 서브플롯 구성: 1행 3열
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), tight_layout=True)



        # --- Subplot (0): Position ---
        axes[0].plot(x_position, positions, marker='o', linestyle='-',
                     color='green')
        axes[0].set_xlabel("Index")
        axes[0].set_ylabel("Position (m)")  # 필요에 따라 단위 라벨 수정
        axes[0].set_title("Position Profile")

        # --- Subplot (1): Heading (deg) ---
        axes[1].plot(x_heading, headings_deg, marker='o', linestyle='-')
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Heading (deg)")
        axes[1].set_title("Heading Profile")
        # --- Subplot (3): Velocity (km/h) ---
        axes[2].plot(x_velocity, velocities_kmh, marker='o', linestyle='-',
                     color='orange')
        axes[2].set_xlabel("Index")
        axes[2].set_ylabel("Velocity (km/h)")
        axes[2].set_title("Velocity Profile")

        # 저장
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
