import numpy as np
import numpy.typing as npt
from typing import List

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative

"""
  File "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner/model/module/encoder.py", line 1701, in forward
    ego_fut_global = self._get_ego_fut_global(
  File "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner/model/module/encoder.py", line 1780, in _get_ego_fut_global
    weights = (weights * row_valid).to(ego_fut_chunk.dtype)
RuntimeError: The size of tensor a (4) must match the size of tensor b (256) at non-singleton dimension 1
"""

def get_ego_past_array_from_scenario(scenario: NuPlanScenario, num_past_poses,
                                     past_time_horizon):

    current_ego_state = scenario.initial_ego_state

    past_ego_states = scenario.get_ego_past_trajectory(
        iteration=0, num_samples=num_past_poses, time_horizon=past_time_horizon)
    # list(past_ego_states): List[EgoState]
    sampled_past_ego_states: List[EgoState] = list(past_ego_states) + [current_ego_state]
    # past_ego_states_array: np (21, 7)
    past_ego_states_array = sampled_past_ego_states_to_array(
        sampled_past_ego_states)

    past_time_stamps = list(
        scenario.get_past_timestamps(
            iteration=0,
            num_samples=num_past_poses,
            time_horizon=past_time_horizon)) + [scenario.start_time]

    def sampled_past_timestamps_to_array(
            past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
        flat = [t.time_us for t in past_time_stamps]
        return np.array(flat, dtype=np.int64)

    past_time_stamps_array = sampled_past_timestamps_to_array(past_time_stamps)

    return past_ego_states_array, past_time_stamps_array


def sampled_past_ego_states_to_array(
        past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:

    output = np.zeros((len(past_ego_states), 7), dtype=np.float64)
    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].center.x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].center.y
        output[
            i,
            EgoInternalIndex.heading()] = past_ego_states[i].center.heading
        output[i, EgoInternalIndex.vx(
        )] = past_ego_states[i].dynamic_car_state.center_velocity_2d.x
        output[i, EgoInternalIndex.vy(
        )] = past_ego_states[i].dynamic_car_state.center_velocity_2d.y
        # output[i, EgoInternalIndex.ax(
        # )] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
        # output[i, EgoInternalIndex.ay(
        # )] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y

        output[i, EgoInternalIndex.ax(
        )] = past_ego_states[i].car_footprint.width
        output[i, EgoInternalIndex.ay(
        )] = past_ego_states[i].car_footprint.length

    return output


def sampled_future_ego_states_to_array(
        future_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:
    """Convert future ego states to a numpy array.

    Args:
        future_ego_states: List of future ego states.

    Returns:
        Array of shape (T, 7) with elements
        [x, y, heading, vx, vy, width, length].
    """

    output = np.zeros((len(future_ego_states), 7), dtype=np.float64)
    for i in range(len(future_ego_states)):
        output[i, EgoInternalIndex.x()] = future_ego_states[i].center.x
        output[i, EgoInternalIndex.y()] = future_ego_states[i].center.y
        output[i, EgoInternalIndex.heading()] = future_ego_states[i].center.heading
        output[i, EgoInternalIndex.vx(
        )] = future_ego_states[i].dynamic_car_state.center_velocity_2d.x
        output[i, EgoInternalIndex.vy(
        )] = future_ego_states[i].dynamic_car_state.center_velocity_2d.y
        output[i, EgoInternalIndex.ax(
        )] = future_ego_states[i].car_footprint.width
        output[i, EgoInternalIndex.ay(
        )] = future_ego_states[i].car_footprint.length

    return output


def get_ego_future_array_from_scenario(scenario: NuPlanScenario,
                                       current_ego_state: EgoState,
                                       num_future_poses: int,
                                       future_time_horizon: float) -> npt.NDArray[np.float32]:
    """Return ego future states in ego-centric coordinates with rich features.

    The returned array has shape (T, 11) where each element consists of
    [x, y, cos(yaw), sin(yaw), vx, vy, width, length, one_hot(3)].
    """

    future_trajectory_absolute_states = scenario.get_ego_future_trajectory(
        iteration=0,
        num_samples=num_future_poses,
        time_horizon=future_time_horizon)

    future_states_array = sampled_future_ego_states_to_array(
        list(future_trajectory_absolute_states))

    anchor_ego_state = np.array([
        current_ego_state.rear_axle.x,
        current_ego_state.rear_axle.y,
        current_ego_state.rear_axle.heading,
    ],
                                dtype=np.float64)

    future_states_array = convert_absolute_quantities_to_relative(
        future_states_array, anchor_ego_state, 'ego')

    future = np.zeros(
        (future_states_array.shape[0], future_states_array.shape[1] + 1 + 3),
        dtype=np.float32,
    )
    future[:, :2] = future_states_array[:, :2]
    future[:, 2] = np.cos(future_states_array[:, 2])
    future[:, 3] = np.sin(future_states_array[:, 2])
    future[:, 4:8] = future_states_array[:, 3:]
    future[:, 8] = 1.0  # ego vehicle is always a car

    # Get all future poses of the ego relative to the ego coordinate system
    future_trajectory_relative_poses = convert_absolute_to_relative_poses(
        current_ego_state.rear_axle,
        [state.rear_axle for state in future_trajectory_absolute_states])

    return future_trajectory_relative_poses, future


def calculate_additional_ego_states(ego_agent_past, time_stamp):
    # ego_agent_past: (N, 7) where N is the number of past states.
    # 7: x, y, heading, vx, vy, width, length
    # transform haeding to cos h, sin h and calculate the steering_angle and yaw_rate for current state

    current_state = ego_agent_past[-1]
    prev_state = ego_agent_past[-2]

    dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

    cur_velocity = current_state[3]
    angle_diff = current_state[2] - prev_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    if abs(cur_velocity) < 0.2:
        steering_angle = 0.0
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
    else:
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity))
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)
    # ego_agent_past: (T, 7)
    # past: (T, 8) # +3 for one-hot encoding of the agent type (car, pedestrian, cyclist) and ego is always car.
    past = np.zeros((ego_agent_past.shape[0], ego_agent_past.shape[1] + 1 + 3), dtype=np.float32)
    # past: x, y, cos(heading), sin(heading), vx, vy, width, length, agent_type
    past[:, :2] = ego_agent_past[:, :2]
    past[:, 2] = np.cos(ego_agent_past[:, 2])
    past[:, 3] = np.sin(ego_agent_past[:, 2])
    past[:, 4:8] = ego_agent_past[:, 3:]
    # add one-hot encoding for agent type.
    past[:, 8] = 1.0  # ego is always car


    current = np.zeros((ego_agent_past.shape[1] + 3), dtype=np.float32)
    current[:2] = current_state[:2]
    current[2] = np.cos(current_state[2])
    current[3] = np.sin(current_state[2])
    current[4:8] = current_state[3:7]
    current[8] = steering_angle
    current[9] = yaw_rate

    return past, current
