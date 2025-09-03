"""
Module: Agent Data Preprocessing Functions
Description: This module contains functions for agents related data processing.

Categories:
    1. Get list of agent array from raw data
    2. Get agents array for model input
"""
import numpy as np
from typing import Dict, Deque, List, Tuple, Optional
from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObject
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import EgoState
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
from nuplan.common.geometry.convert import numpy_array_to_absolute_velocity


# =====================
# 1. Get list of agent array from raw data
# =====================
def _extract_agent_array(
        tracked_objects: TrackedObjects, token_to_id, object_types
) -> Tuple[np.ndarray, Dict[str, int], List[TrackedObjectType]]:
    """
    Extracts the relevant data from the agents present in a past detection into a array.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a array as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a array.
        TrackedObjects
    :token_to_id: A dictionary used to assign track tokens to integer IDs.
        {}
    :object_type: TrackedObjectType to filter agents by.
        object_types = [
            TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE
        ]
    :return: The generated array and the updated token_to_id dict.
    """
    agents: List[TrackedObject] = tracked_objects.get_tracked_objects_of_types(
        object_types)
    agent_types = []
    frame_agents_num = len(agents)
    # (frame_agents_num, 8)
    frame_agents_feature = np.zeros(
        (frame_agents_num, AgentInternalIndex.dim()), dtype=np.float64)
    max_id_number = len(token_to_id)

    for idx, agent in enumerate(agents):
        if agent.track_token not in token_to_id:
            token_to_id[agent.track_token] = max_id_number
            max_id_number += 1
        int_id = token_to_id[agent.track_token]

        frame_agents_feature[idx,
                             AgentInternalIndex.track_token()] = float(int_id)
        frame_agents_feature[idx, AgentInternalIndex.vx()] = agent.velocity.x
        frame_agents_feature[idx, AgentInternalIndex.vy()] = agent.velocity.y
        frame_agents_feature[
            idx, AgentInternalIndex.heading()] = agent.center.heading
        frame_agents_feature[idx, AgentInternalIndex.width()] = agent.box.width
        frame_agents_feature[idx,
                             AgentInternalIndex.length()] = agent.box.length
        frame_agents_feature[idx, AgentInternalIndex.x()] = agent.center.x
        frame_agents_feature[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return frame_agents_feature, token_to_id, agent_types


def sampled_tracked_objects_to_array_list(
        past_tracked_objects: Deque[Observation])\
        -> Tuple[List[np.ndarray], List[List[TrackedObjectType]]]:
    """
    Arrayifies the agents features from the provided past detections.
    For N past detections, output is a list of length N, with each array as described in `_extract_agent_array()`.
    :param past_tracked_objects: The tracked objects to arrayify.
    :return: The arrayified objects.
    """
    object_types = [
        TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN,
        TrackedObjectType.BICYCLE
    ]
    all_frame_agents_feature = []
    all_frame_agents_types = []
    token_to_id = {}

    for past_idx in range(len(past_tracked_objects)):
        if type(past_tracked_objects[past_idx]) == DetectionsTracks:
            track_object: TrackedObjects = past_tracked_objects[
                past_idx].tracked_objects
        else:
            track_object = past_tracked_objects[past_idx]
        # Tuple[np.ndarray ((frame_agents_num, 8)), Dict[str, int], List[TrackedObjectType]]
        frame_agents_feature, token_to_id, agent_types = _extract_agent_array(
            track_object, token_to_id, object_types)
        all_frame_agents_feature.append(frame_agents_feature)
        all_frame_agents_types.append(agent_types)

    return all_frame_agents_feature, all_frame_agents_types


def _extract_ego_array(track_ego: EgoState) -> np.ndarray:
    # (10)
    frame_ego_feature = np.zeros((10,), dtype=np.float64)

    frame_ego_feature[0] = track_ego.center.x
    frame_ego_feature[1] = track_ego.center.y
    frame_ego_feature[2] = track_ego.center.heading
    # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
    v_local = track_ego.dynamic_car_state.center_velocity_2d
    v_global = numpy_array_to_absolute_velocity(
        track_ego.center, np.array([[v_local.x, v_local.y]],
                                   dtype=np.float32))[0]
    frame_ego_feature[3] = v_global.x
    frame_ego_feature[4] = v_global.y
    frame_ego_feature[5] = track_ego.car_footprint.width
    frame_ego_feature[6] = track_ego.car_footprint.length
    frame_ego_feature[7:10] = [1, 0, 0]  # Mark as VEHICLE

    return frame_ego_feature


def sampled_ego_objects_to_array_list(
        ego_state_buffer: Deque[EgoState]) -> np.ndarray:
    all_frame_ego_feature = []

    for past_idx in range(len(ego_state_buffer)):
        # 가장 과거 -> 가장 최근 순서
        track_ego: EgoState = ego_state_buffer[past_idx]
        frame_agents_feature = _extract_ego_array(track_ego)
        all_frame_ego_feature.append(frame_agents_feature)
    all_frame_ego_feature = np.stack(all_frame_ego_feature)  # (num_frames, 10)

    return all_frame_ego_feature


def sampled_static_objects_to_array_list(present_tracked_objects: Observation):

    static_object_types = [
        TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER,
        TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT
    ]

    if type(present_tracked_objects) == DetectionsTracks:
        present_tracked_objects: TrackedObjects = present_tracked_objects.tracked_objects

    static_obj: List[
        TrackedObject] = present_tracked_objects.get_tracked_objects_of_types(
            static_object_types)
    static_objects_types = []
    present_static_feature = np.zeros((len(static_obj), 5), dtype=np.float64)

    for idx, agent in enumerate(static_obj):
        present_static_feature[idx, 0] = agent.center.x
        present_static_feature[idx, 1] = agent.center.y
        present_static_feature[idx, 2] = agent.center.heading
        present_static_feature[idx, 3] = agent.box.width
        present_static_feature[idx, 4] = agent.box.length
        static_objects_types.append(agent.tracked_object_type)

    return present_static_feature, static_objects_types


# =====================
# 2. Get agents array for model input
# =====================
def _filter_agents_array(
    all_frame_agents_feature: List[
        np.ndarray],  # (frame_agents_num, 8) # frame_agents_num 길이가 가변적
    reverse: bool = False
) -> List[np.ndarray]:
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param all_frame_agents_feature: The past agents in the scene. A list of [num_frames] arrays, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_frame_agents_feature = all_frame_agents_feature[
        -1] if reverse else all_frame_agents_feature[0]
    # target_frame_agents_id: (target_frame_agents_num,)
    target_frame_agents_id = target_frame_agents_feature[:,
                                                         AgentInternalIndex.
                                                         track_token()]
    for past_idx in range(len(all_frame_agents_feature)):
        frame_exist_agents = []
        # (frame_agents_num, 8)
        frame_agents_feature: np.ndarray = all_frame_agents_feature[past_idx]
        for agent_idx in range(frame_agents_feature.shape[0]):
            if target_frame_agents_feature.shape[0] > 0:
                agent_id = float(
                    frame_agents_feature[agent_idx,
                                         int(AgentInternalIndex.track_token())])
                is_in_target_frame = bool(
                    (agent_id == target_frame_agents_id).max())
                if is_in_target_frame:
                    frame_exist_agents.append(
                        frame_agents_feature[agent_idx, :].squeeze())

        if len(frame_exist_agents) > 0:
            all_frame_agents_feature[past_idx] = np.stack(
                frame_exist_agents)  # (frame_save_agents_num, 8)
        else:
            all_frame_agents_feature[past_idx] = np.empty(
                (0, frame_agents_feature.shape[1]), dtype=np.float32)  # (0, 8)

    return all_frame_agents_feature


def _pad_agent_states(
    all_frame_cur_exists_agents: List[np.ndarray], reverse: bool
) -> List[
        np.ndarray]:  # Return 값의 np.ndarray shape: (current_agents_num, 8) 로 동일
    # (saved_agents_num, 8) # saved_agents_num 길이가 가변적
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param all_frame_cur_exists_agents: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """

    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        all_frame_cur_exists_agents = all_frame_cur_exists_agents[::-1]

    key_frame_agents = all_frame_cur_exists_agents[0]  # (current_agents_num, 8)

    key_frame_id_to_row: Dict[int, int] = {}
    for cur_agent_idx, agent_id in enumerate(key_frame_agents[:, track_id_idx]):
        key_frame_id_to_row[int(agent_id)] = cur_agent_idx

    # current_state: (current_agents_num, 8)
    """
    t = -1에 있었는데, t= -2에 없어진 agent의 경우, t=-2에 t=-1의 state로 padding
    """
    current_state = np.zeros(
        (key_frame_agents.shape[0], key_frame_agents.shape[1]),
        dtype=np.float64)
    for time_idx in range(len(all_frame_cur_exists_agents)):
        frame_cur_exists_agents = all_frame_cur_exists_agents[
            time_idx]  # (agents_num, 8)

        # Update current frame
        for agent_row_idx in range(frame_cur_exists_agents.shape[0]):
            # frame_cur_exists_agents: (agents_num, 8)
            agent_id = int(frame_cur_exists_agents[agent_row_idx, track_id_idx])
            key_frame_row: int = key_frame_id_to_row[agent_id]
            current_state[key_frame_row, :] = frame_cur_exists_agents[
                agent_row_idx, :]

        # Save current state
        all_frame_cur_exists_agents[time_idx] = current_state.copy()

    if reverse:
        all_frame_cur_exists_agents = all_frame_cur_exists_agents[::-1]

    return all_frame_cur_exists_agents


def _pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = np.zeros(
        (len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]),
        dtype=np.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[
                    frame[:, track_id_idx] == row_idx]

    return pad_agent_trajectories


def agent_past_process(
        all_frame_ego_feature: np.ndarray,  # (num_frames, 10)
        all_frame_agents_feature: List[
            np.ndarray],  # (frame_agents_num, 8) # frame_agents_num 길이가 가변적
        all_frame_agents_types: List[List[TrackedObjectType]],
        agent_num: int,
        present_static_feature: np.ndarray,  # (cur_static_num, 5)
        static_objects_types: List[TrackedObjectType],
        num_static: int,
        max_ped_bike: int,
        anchor_ego_state: np.ndarray,  #(3,)
) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    # ego_agent_past: (num_frames, 11)
    # neighbor_agents_past: (agent_num, num_frames, 11)
    # sorted_cur_neighbor_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
    # static_objects: (num_static, 10)
    """
    This function process the data from the raw agent data.
    :param all_frame_ego_feature: The input array data of the ego past.
    :param all_frame_agents_feature: The input array data of agents in the past.
    :param all_frame_agents_types: The type of agents in the past.
    :param agent_num: Clip the number of agents.
    :param present_static_feature: The input array data of static objects in the past.
    :param static_objects_types: The type of static objects in the past.
    :param num_static: Clip the number of static objects.
    :param max_ped_bike: Clip the total number of ped and bike.
    :param anchor_ego_state: Ego current state
    :return: ego_agent_past, agents, sorted_cur_neighbor_indices, present_static_feature
    """
    agents_states_dim = 8  # x, y, cos h, sin h, vx, vy, length, width
    if all_frame_ego_feature is not None:
        # all_frame_ego_feature: (num_frames, 10)
        # ego_agent_past: (num_frames, 11)
        ego_agent_past = convert_absolute_quantities_to_relative(
            all_frame_ego_feature, anchor_ego_state)
        assert ego_agent_past.shape[1] == 11
        ego_agent_past = ego_agent_past.astype(np.float32)
    else:
        ego_agent_past = None
    # (saved_agents_num, 8) # saved_agents_num 길이가 가변적
    all_frame_cur_exists_agents: List[np.ndarray] = _filter_agents_array(
        all_frame_agents_feature, reverse=True)
    current_agent_types: List[TrackedObjectType] = all_frame_agents_types[-1]

    if all_frame_cur_exists_agents[-1].shape[0] == 0:
        # Return zero array when there are no agents in the scene
        all_frame_np_agents_local = np.zeros(
            (len(all_frame_cur_exists_agents), 0, agents_states_dim))
    else:
        all_frame_cur_exists_agents_local: List[np.ndarray] = []
        # all_frame_cur_exists_agents 와 frame_cur_exists_agents_local의
        # np.ndarray shape: (current_agents_num, 8) 로 동일
        all_frame_cur_exists_agents: List[np.ndarray] = _pad_agent_states(
            all_frame_cur_exists_agents, reverse=True)

        for frame_cur_exists_agents in all_frame_cur_exists_agents:
            # frame_cur_exists_agents: (current_agents_num, 8)
            # a: (current_agents_num, 8)
            a = convert_absolute_quantities_to_relative(
                frame_cur_exists_agents,
                anchor_ego_state,  #(3,)
                'agent')
            all_frame_cur_exists_agents_local.append(a)

        # Calculate yaw rate
        # agents_states_dim: id, vx, vy, heading, width, length, x, y
        # all_frame_np_agents_local: (num_frames, current_agents_num, 8)
        # all_frame_np_agents_local 8: x, y, cos h, sin h, vx, vy, length, width
        all_frame_np_agents_local = np.zeros(
            (len(all_frame_cur_exists_agents_local),
             all_frame_cur_exists_agents_local[0].shape[0], agents_states_dim))

        for past_idx in range(len(all_frame_cur_exists_agents_local)):
            frame_cur_exists_agents_local = all_frame_cur_exists_agents_local[
                past_idx]  # (current_agents_num, 8)
            all_frame_np_agents_local[
                past_idx, :,
                0] = frame_cur_exists_agents_local[:, AgentInternalIndex.x(
                )].squeeze()
            all_frame_np_agents_local[
                past_idx, :,
                1] = frame_cur_exists_agents_local[:, AgentInternalIndex.y(
                )].squeeze()
            all_frame_np_agents_local[past_idx, :, 2] = np.cos(
                all_frame_cur_exists_agents_local[past_idx]
                [:, AgentInternalIndex.heading()].squeeze())
            all_frame_np_agents_local[past_idx, :, 3] = np.sin(
                all_frame_cur_exists_agents_local[past_idx]
                [:, AgentInternalIndex.heading()].squeeze())
            all_frame_np_agents_local[
                past_idx, :,
                4] = frame_cur_exists_agents_local[:,
                                                   AgentInternalIndex.vx(
                                                   )].squeeze()
            all_frame_np_agents_local[
                past_idx, :,
                5] = frame_cur_exists_agents_local[:,
                                                   AgentInternalIndex.vy(
                                                   )].squeeze()
            all_frame_np_agents_local[
                past_idx, :,
                6] = frame_cur_exists_agents_local[:,
                                                   AgentInternalIndex.width(
                                                   )].squeeze()
            all_frame_np_agents_local[
                past_idx, :,
                7] = frame_cur_exists_agents_local[:,
                                                   AgentInternalIndex.length(
                                                   )].squeeze()
    #  present_static_feature: (cur_static_num, 5)
    # present_static_feature_6: (cur_static_num, 6)
    present_static_feature_6 = np.zeros((present_static_feature.shape[0], 6))
    if present_static_feature.shape[0] != 0:
        # present_static_feature_local: (cur_static_num, 5)
        present_static_feature_local = convert_absolute_quantities_to_relative(
            present_static_feature, anchor_ego_state, 'static')

        present_static_feature_6[:, 0] = present_static_feature_local[:, 0]
        present_static_feature_6[:, 1] = present_static_feature_local[:, 1]
        present_static_feature_6[:, 2] = np.cos(present_static_feature_local[:,
                                                                             2])
        present_static_feature_6[:, 3] = np.sin(present_static_feature_local[:,
                                                                             2])
        present_static_feature_6[:, 4] = present_static_feature_local[:, 3]
        present_static_feature_6[:, 5] = present_static_feature_local[:, 4]
    '''
    Post-process the agents array to select a fixed number of agents closest to the ego vehicle.
    neighbor_agents_past: <np.ndarray: agent_num, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The agent_num is padded or trimmed to fit the predefined number of agents across.
    '''
    # Initialize the result array
    # all_frame_np_agents_local: (num_frames, current_agents_num, 8)
    # neighbor_agents_past: (agent_num, num_frames, 11)
    neighbor_agents_past = np.zeros(
        (agent_num, all_frame_np_agents_local.shape[0],
         all_frame_np_agents_local.shape[-1] + 3),
        dtype=np.float32)
    # dist_from_cur_agent_to_ego: (current_agents_num,)
    dist_from_cur_agent_to_ego = np.linalg.norm(
        all_frame_np_agents_local[-1, :, :2], axis=-1)

    # Sort indices by distance
    # sorted_cur_agent_indices: (current_agents_num,) # order method: ascending
    sorted_cur_agent_indices = np.argsort(dist_from_cur_agent_to_ego)

    # Collect the indices of pedestrians and bicycles
    ped_bike_sorted_indices = [
        sorted_idx for sorted_idx in sorted_cur_agent_indices
        if current_agent_types[sorted_idx] in (TrackedObjectType.PEDESTRIAN,
                                               TrackedObjectType.BICYCLE)
    ]
    vehicle_sorted_indices = [
        sorted_idx for sorted_idx in sorted_cur_agent_indices
        if current_agent_types[sorted_idx] == TrackedObjectType.VEHICLE
    ]

    # If the total number of available agents is less than or equal to agent_num, no need to filter further

    if len(ped_bike_sorted_indices) + len(vehicle_sorted_indices) <= agent_num:
        """
        sorted_cur_neighbor_indices 는 agent_num 보다 작을수도 있다.
        """
        sorted_cur_neighbor_indices = sorted_cur_agent_indices[:agent_num]
    else:
        # Limit the number of pedestrians and bicycles to max_ped_bike, while retaining the remaining ones for later use
        selected_ped_bike_indices = ped_bike_sorted_indices[:max_ped_bike]
        unselected_ped_bike_indices = ped_bike_sorted_indices[max_ped_bike:]

        # Combine the limited pedestrians/bicycles and all available vehicles
        sorted_cur_neighbor_indices = selected_ped_bike_indices + vehicle_sorted_indices

        # If the combined selection is still less than agent_num, fill the remaining slots with additional pedestrians and bicycles
        remaining_slots = agent_num - len(sorted_cur_neighbor_indices)
        if remaining_slots > 0:
            sorted_cur_neighbor_indices += unselected_ped_bike_indices[:
                                                                       remaining_slots]

        # Sort and limit the selected indices to agent_num

        sorted_cur_neighbor_indices = sorted(
            sorted_cur_neighbor_indices,
            key=lambda idx: dist_from_cur_agent_to_ego[idx])[:agent_num]
        """
        sorted_cur_neighbor_indices 의 길이는 무조건 agent_num 이다.
        """
    # Populate the final agents array with the selected agents' features
    for sort_idx, cur_neighbor_idx in enumerate(sorted_cur_neighbor_indices):
        # neighbor_agents_past: (agent_num, num_frames, 11)
        # all_frame_np_agents_local: (num_frames, current_agents_num, 8)
        # current_agent_types: List[TrackedObjectType]
        eight_ = all_frame_np_agents_local.shape[-1]
        neighbor_agents_past[
            sort_idx, :, :
            eight_] = all_frame_np_agents_local[:, cur_neighbor_idx, :eight_]
        if current_agent_types[cur_neighbor_idx] == TrackedObjectType.VEHICLE:
            neighbor_agents_past[sort_idx, :, eight_:] = [1, 0,
                                                          0]  # Mark as VEHICLE
        elif current_agent_types[
                cur_neighbor_idx] == TrackedObjectType.PEDESTRIAN:
            neighbor_agents_past[sort_idx, :,
                                 eight_:] = [0, 1, 0]  # Mark as PEDESTRIAN
        else:  # TrackedObjectType.BICYCLE
            neighbor_agents_past[sort_idx, :, eight_:] = [0, 0,
                                                          1]  # Mark as BICYCLE
    # present_static_feature_6: (cur_static_num, 6)
    static_objects = np.zeros(
        (num_static, present_static_feature_6.shape[-1] + 4), dtype=np.float32)
    static_distance_to_ego = np.linalg.norm(present_static_feature_6[:, :2],
                                            axis=-1)
    static_indices = list(np.argsort(static_distance_to_ego))[:num_static]
    six_ = present_static_feature_6.shape[-1]
    for i, j in enumerate(static_indices):
        static_objects[i, :six_] = present_static_feature_6[j, :six_]
        if static_objects_types[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, six_:] = [1, 0, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.BARRIER:
            static_objects[i, six_:] = [0, 1, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, six_:] = [0, 0, 1, 0]
        else:
            static_objects[i, six_:] = [0, 0, 0, 1]

    return ego_agent_past, neighbor_agents_past, sorted_cur_neighbor_indices, static_objects


def agent_future_process(anchor_ego_state, future_tracked_objects, num_agents,
                         agent_index):

    agent_future = _filter_agents_array(future_tracked_objects)
    local_coords_agent_states = []
    for agent_state in agent_future:
        local_coords_agent_states.append(
            convert_absolute_quantities_to_relative(agent_state,
                                                    anchor_ego_state, 'agent'))
    padded_agent_states = _pad_agent_states_with_zeros(
        local_coords_agent_states)

    # fill agent features into the array
    agent_futures = np.zeros(shape=(num_agents,
                                    padded_agent_states.shape[0] - 1, 3),
                             dtype=np.float32)
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[1:, j, [
            AgentInternalIndex.x(),
            AgentInternalIndex.y(),
            AgentInternalIndex.heading()
        ]]

    return agent_futures
