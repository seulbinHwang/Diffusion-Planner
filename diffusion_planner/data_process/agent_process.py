"""
Module: Agent Data Preprocessing Functions
Description: This module contains functions for agents related data processing.

Categories:
    1. Get list of agent array from raw data
    2. Get agents array for model input
"""
import numpy as np
from typing import Dict, List, Optional, Tuple

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks

from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative


# =====================
# 1. Get list of agent array from raw data
# =====================
def _extract_agent_array(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a array.
    Only objects of specified type will be transformed. Others will be ignored.
    The output is a array as described in AgentInternalIndex
    :param tracked_objects: The tracked objects to turn into a array.
    :track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :object_type: TrackedObjectType to filter agents by.
    :return: The generated array and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = np.zeros((len(agents), AgentInternalIndex.dim()), dtype=np.float64)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_array_list(past_tracked_objects):
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
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        if type(past_tracked_objects[i]) == DetectionsTracks:
            track_object = past_tracked_objects[i].tracked_objects
        else:
            track_object = past_tracked_objects[i]
        # track_token_ids: A dictionary used to assign track tokens (str) to integer IDs.
        arrayified, track_token_ids, agent_types = _extract_agent_array(
            track_object, track_token_ids, object_types)
        output.append(arrayified)
        output_types.append(agent_types)
    # 코드 추가
    neighbor_agents_current_ids = output[-1][:,
                                             AgentInternalIndex.track_token(
                                             )]  # (N,)
    ids_track_token = {
        track_ids: track_token
        for track_token, track_ids in track_token_ids.items()
    }
    neighbor_agents_current_track_tokens = []  # List[str]
    for current_id in neighbor_agents_current_ids:
        neighbor_agents_current_track_tokens.append(
            ids_track_token[current_id])
    # 코드 추가 끝
    return output, output_types, neighbor_agents_current_track_tokens


def sampled_static_objects_to_array_list(present_tracked_objects):

    static_object_types = [
        TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER,
        TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT
    ]

    if type(present_tracked_objects) == DetectionsTracks:
        present_tracked_objects = present_tracked_objects.tracked_objects

    static_obj = present_tracked_objects.get_tracked_objects_of_types(
        static_object_types)
    agent_types = []
    output = np.zeros((len(static_obj), 5), dtype=np.float64)

    for idx, agent in enumerate(static_obj):
        output[idx, 0] = agent.center.x
        output[idx, 1] = agent.center.y
        output[idx, 2] = agent.center.heading
        output[idx, 3] = agent.box.width
        output[idx, 4] = agent.box.length
        agent_types.append(agent.tracked_object_type)

    return output, agent_types


# =====================
# 2. Get agents array for model input
# =====================
def _filter_agents_array(agents, reverse: bool = False):
    """
    Filter detections to keep only agents which appear in the first frame (or last frame if reverse=True)
    :param agents: The past agents in the scene. A list of [num_frames] arrays, each complying with the AgentInternalIndex schema
    :param reverse: if True, the last element in the list will be used as the filter
    :return: filtered agents in the same format as the input `agents` parameter
    """
    target_array = agents[-1] if reverse else agents[0]
    for i in range(len(agents)):

        rows = []
        for j in range(agents[i].shape[0]):
            if target_array.shape[0] > 0:
                agent_id: float = float(
                    agents[i][j, int(AgentInternalIndex.track_token())])
                is_in_target_frame: bool = bool(
                    (agent_id == target_array[:,
                                              AgentInternalIndex.track_token()]
                    ).max())
                if is_in_target_frame:
                    rows.append(agents[i][j, :].squeeze())

        if len(rows) > 0:
            agents[i] = np.stack(rows)
        else:
            agents[i] = np.empty((0, agents[i].shape[1]), dtype=np.float32)

    return agents


def _pad_agent_states(agent_trajectories, reverse: bool):
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

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """

    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]

    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val)] = idx

    current_state = np.zeros((key_frame.shape[0], key_frame.shape[1]),
                             dtype=np.float64)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]

        # Update current frame
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx])]
            current_state[mapped_row, :] = frame[row_idx, :]

        # Save current state
        agent_trajectories[idx] = current_state.copy()

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    return agent_trajectories


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


def agent_past_process(past_ego_states, past_tracked_objects,
                       tracked_objects_types,
                       neighbor_agents_current_track_tokens, num_agents,
                       static_objects, static_objects_types, num_static,
                       max_ped_bike, anchor_ego_state):
    """
    This function process the data from the raw agent data.
    :param past_ego_states: The input array data of the ego past.
    :param past_tracked_objects: The input array data of agents in the past.
    :param tracked_objects_types: The type of agents in the past.
    :param num_agents: Clip the number of agents.
    :param static_objects: The input array data of static objects in the past.
    :param static_objects_types: The type of static objects in the past.
    :param num_static: Clip the number of static objects.
    :param max_ped_bike: Clip the total number of ped and bike.
    :param anchor_ego_state: Ego current state
    :return: ego, agents, selected_indices, static_objects
    """
    agents_states_dim = 8  # x, y, cos h, sin h, vx, vy, length, width
    ego_history = past_ego_states
    agents = past_tracked_objects

    if past_ego_states is not None:
        ego = convert_absolute_quantities_to_relative(ego_history,
                                                      anchor_ego_state)
    else:
        ego = None

    agent_history = _filter_agents_array(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    if agent_history[-1].shape[0] == 0:
        # Return zero array when there are no agents in the scene
        agents_array = np.zeros((len(agent_history), 0, agents_states_dim))
    else:
        local_coords_agent_states = []
        padded_agent_states = _pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(
                convert_absolute_quantities_to_relative(agent_state,
                                                        anchor_ego_state,
                                                        'agent'))

        # Calculate yaw rate
        agents_array = np.zeros(
            (len(local_coords_agent_states),
             local_coords_agent_states[0].shape[0], agents_states_dim))

        for i in range(len(local_coords_agent_states)):
            agents_array[i, :,
                         0] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.x(
                                                           )].squeeze()
            agents_array[i, :,
                         1] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.y(
                                                           )].squeeze()
            agents_array[i, :,
                         2] = np.cos(local_coords_agent_states[i]
                                     [:,
                                      AgentInternalIndex.heading()].squeeze())
            agents_array[i, :,
                         3] = np.sin(local_coords_agent_states[i]
                                     [:,
                                      AgentInternalIndex.heading()].squeeze())
            agents_array[i, :,
                         4] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.
                                                           vx()].squeeze()
            agents_array[i, :,
                         5] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.
                                                           vy()].squeeze()
            agents_array[i, :,
                         6] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.
                                                           width()].squeeze()
            agents_array[i, :,
                         7] = local_coords_agent_states[i][:,
                                                           AgentInternalIndex.
                                                           length()].squeeze()

    static_objects_array = np.zeros((static_objects.shape[0], 6))
    if static_objects.shape[0] != 0:
        local_coords_static_objects_states = convert_absolute_quantities_to_relative(
            static_objects, anchor_ego_state, 'static')

        static_objects_array[:, 0] = local_coords_static_objects_states[:, 0]
        static_objects_array[:, 1] = local_coords_static_objects_states[:, 1]
        static_objects_array[:,
                             2] = np.cos(local_coords_static_objects_states[:,
                                                                            2])
        static_objects_array[:,
                             3] = np.sin(local_coords_static_objects_states[:,
                                                                            2])
        static_objects_array[:, 4] = local_coords_static_objects_states[:, 3]
        static_objects_array[:, 5] = local_coords_static_objects_states[:, 4]
    '''
    Post-process the agents array to select a fixed number of agents closest to the ego vehicle.
    agents: <np.ndarray: num_agents, num_frames, 11>]].
        Agent type is one-hot encoded: [1, 0, 0] vehicle, [0, 1, 0] pedestrain, [0, 0, 1] bicycle 
            and added to the feature of the agent
        The num_agents is padded or trimmed to fit the predefined number of agents across.
    '''
    # Initialize the result array
    agents = np.zeros(
        (num_agents, agents_array.shape[0], agents_array.shape[-1] + 3),
        dtype=np.float32)

    distance_to_ego = np.linalg.norm(agents_array[-1, :, :2], axis=-1)

    # Sort indices by distance
    sorted_indices = np.argsort(distance_to_ego)

    # Collect the indices of pedestrians and bicycles
    ped_bike_indices = [
        i for i in sorted_indices
        if agent_types[i] in (TrackedObjectType.PEDESTRIAN,
                              TrackedObjectType.BICYCLE)
    ]
    vehicle_indices = [
        i for i in sorted_indices if agent_types[i] == TrackedObjectType.VEHICLE
    ]

    # If the total number of available agents is less than or equal to num_agents, no need to filter further
    if len(ped_bike_indices) + len(vehicle_indices) <= num_agents:
        selected_indices = sorted_indices[:num_agents]
    else:
        # Limit the number of pedestrians and bicycles to max_ped_bike, while retaining the remaining ones for later use
        selected_ped_bike_indices = ped_bike_indices[:max_ped_bike]
        remaining_ped_bike_indices = ped_bike_indices[max_ped_bike:]

        # Combine the limited pedestrians/bicycles and all available vehicles
        selected_indices = selected_ped_bike_indices + vehicle_indices

        # If the combined selection is still less than num_agents, fill the remaining slots with additional pedestrians and bicycles
        remaining_slots = num_agents - len(selected_indices)
        if remaining_slots > 0:
            selected_indices += remaining_ped_bike_indices[:remaining_slots]

        # Sort and limit the selected indices to num_agents
        selected_indices = sorted(
            selected_indices, key=lambda idx: distance_to_ego[idx])[:num_agents]
    # === 추가된 부분: 각 selected agent의 문자열 track_token 목록 생성 ===
    agents_track_token: List[Optional[str]] = []
    if len(selected_indices) > 0:
        # 원본 마지막 프레임의 DetectionsTracks 가져오기
        for idx in selected_indices:
            agents_track_token.append(
                neighbor_agents_current_track_tokens[idx])
    # 슬롯이 부족하면 None으로 패딩
    while len(agents_track_token) < num_agents:
        agents_track_token.append(None)
    # === 추가된 부분 끝 ===

    # Populate the final agents array with the selected agents' features
    for i, j in enumerate(selected_indices):
        agents[i, :, :agents_array.
               shape[-1]] = agents_array[:, j, :agents_array.shape[-1]]
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents[i, :, agents_array.shape[-1]:] = [1, 0, 0]  # Mark as VEHICLE
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents[i, :, agents_array.shape[-1]:] = [0, 1,
                                                     0]  # Mark as PEDESTRIAN
        else:  # TrackedObjectType.BICYCLE
            agents[i, :, agents_array.shape[-1]:] = [0, 0, 1]  # Mark as BICYCLE

    static_objects = np.zeros((num_static, static_objects_array.shape[-1] + 4),
                              dtype=np.float32)
    static_distance_to_ego = np.linalg.norm(static_objects_array[:, :2],
                                            axis=-1)
    static_indices = list(np.argsort(static_distance_to_ego))[:num_static]

    for i, j in enumerate(static_indices):
        static_objects[
            i, :static_objects_array.shape[-1]] = static_objects_array[
                j, :static_objects_array.shape[-1]]
        if static_objects_types[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, static_objects_array.shape[-1]:] = [1, 0, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.BARRIER:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 1, 0, 0]
        elif static_objects_types[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 1, 0]
        else:
            static_objects[i, static_objects_array.shape[-1]:] = [0, 0, 0, 1]

    if ego is not None:
        ego = ego.astype(np.float32)

    return ego, agents, selected_indices, static_objects, agents_track_token


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
