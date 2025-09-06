"""
Module: Coordination Transformation Functions and Numpy-Tensor Transformation
Description: This module contains functions for transforming the coordination to ego-centric coordination and Numpy-Tensor transformation.

Categories:
    1. Ego, agent, static coordination transformation
    2. Map coordination transformation
    3. Numpy-Tensor transformation
"""
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
import numpy as np
import torch
from typing import Deque, Dict, List, Optional, Set, Type
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from shapely.geometry import Point
from collections import defaultdict
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer


def get_npc_route_roadblock_ids(
        scenario: NuPlanScenario) -> Dict[str, Optional[List[str]]]:

    def select_nearest_connectors_by_mean_distance(
        connector_candidates: List[RoadBlockGraphEdgeMapObject],
        sampled_trajectory_points: List["Point2D"],
        *,
        tolerance: float = 1e-6,
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """
        궤적과 RoadBlock-Connector 후보군 사이의 평균 수선 거리를 계산해
        가장 가까운(=평균 거리가 최소) Connector 들을 **모두** 반환한다.

        Args:
            connector_candidates : 거리 비교 대상이 되는 RoadBlock-Connector 객체 리스트
            sampled_trajectory_points : ROADBLOCK_CONNECTOR 구간에서 수집한 궤적 포인트들
            tolerance : float
                부동소수점 오차 보정을 위한 허용 오차.
                ``abs(dist - min_dist) ≤ tolerance`` 이면 동률로 처리
            verbose : bool
                True 이면 각 후보의 거리와 선택 결과를 stdout 으로 출력

        Returns:
            List[RoadBlockGraphEdgeMapObject] :
                최소 평균 거리를 가진 Connector 객체(들).
                (복수일 수 있음)
        """
        # 1) 각 후보 ↔ 궤적 사이 평균 수선거리 계산
        mean_distance_by_connector: Dict[RoadBlockGraphEdgeMapObject,
                                         float] = {}
        for connector in connector_candidates:
            mean_dist: float = _mean_perpendicular_distance(
                connector, sampled_trajectory_points)
            mean_distance_by_connector[connector] = mean_dist

        # 2) 최솟값과 동률(±tolerance)인 후보 추출
        minimum_distance: float = min(mean_distance_by_connector.values())
        nearest_connectors: List[RoadBlockGraphEdgeMapObject] = [
            conn for conn, dist in mean_distance_by_connector.items()
            if abs(dist - minimum_distance) <= tolerance
        ]

        return nearest_connectors

    def _decide_roadblock_ids_at_connector(
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'],
        sampled_points_inside_connector: List['Point2D'],
        roadblock_sequence: List[str],
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'],
        current_roadblocks: Set['RoadBlockGraphEdgeMapObject'],
    ) -> None:
        graph_linkable_connectors = []
        graph_linkable_connectors_candidates = []
        incoming_and_outcoming_condition = len(
            previous_roadblocks_set) > 0 and len(current_roadblocks) > 0
        incoming_or_outgoing_condition = len(
            previous_roadblocks_set) > 0 or len(current_roadblocks) > 0
        if incoming_and_outcoming_condition:
            for conn in connector_candidate_objects:
                incoming_ids = {rb.id for rb in conn.incoming_edges}
                previous_ids = {rb.id for rb in previous_roadblocks_set}
                incoming_condition = bool(previous_ids & incoming_ids)

                outgoing_ids = {rb.id for rb in conn.outgoing_edges}
                current_ids = {rb.id for rb in current_roadblocks}
                outgoing_condition = bool(current_ids & outgoing_ids)

                if incoming_condition and outgoing_condition:
                    graph_linkable_connectors.append(conn)
            if graph_linkable_connectors:
                roadblock_sequence.extend(
                    [conn.id for conn in graph_linkable_connectors])

                return
        if (incoming_and_outcoming_condition or incoming_or_outgoing_condition):
            for conn in connector_candidate_objects:
                incoming_ids = {rb.id for rb in conn.incoming_edges}
                previous_ids = {rb.id for rb in previous_roadblocks_set}
                incoming_condition = bool(previous_ids & incoming_ids)

                outgoing_ids = {rb.id for rb in conn.outgoing_edges}
                current_ids = {rb.id for rb in current_roadblocks}
                outgoing_condition = bool(current_ids & outgoing_ids)

                if incoming_condition or outgoing_condition:
                    graph_linkable_connectors_candidates.append(conn)
            if graph_linkable_connectors_candidates:
                # 평균 거리 기반 최적 RBC 선택
                closest_connectors = select_nearest_connectors_by_mean_distance(
                    graph_linkable_connectors_candidates,
                    sampled_points_inside_connector,
                    tolerance=1e-6,
                )
                roadblock_sequence.extend(
                    conn.id for conn in closest_connectors)
                return
        # 평균 거리 기반 최적 RBC 선택
        closest_connectors = select_nearest_connectors_by_mean_distance(
            connector_candidate_objects,
            sampled_points_inside_connector,
            tolerance=1e-6,
        )
        roadblock_sequence.extend(conn.id for conn in closest_connectors)

    def _mean_perpendicular_distance(
            roadblock_connector, trajectory_points: List['Point2D']) -> float:
        """궤적 점들과 RBC 폴리곤 간 평균 거리를 계산."""
        polygon = roadblock_connector.polygon  # NuPlan 에서는 scaled‑width polygon 제공
        return float(
            np.mean([
                Point(pt.x, pt.y).distance(polygon) for pt in trajectory_points
            ]))

    # ─────────── 1단계: 차량별 프레임 수집 ────────────
    token_to_trajectory: Dict[str, List['SceneObject']] = defaultdict(list)
    total_horizon_s = (
        scenario.get_time_point(scenario.get_number_of_iterations() - 1).time_s
        - scenario.get_time_point(0).time_s)
    for det_batch in scenario.get_future_tracked_objects(0, total_horizon_s):
        for det in det_batch.tracked_objects:
            if det.tracked_object_type == TrackedObjectType.VEHICLE:
                token_to_trajectory[det.track_token].append(det)

    token_to_route_roadblock_ids: Dict[str, Optional[List[str]]] = {}
    # TODO: token_to_position 는 디버깅용 이므로, 디버깅이 끝나면 지우는 것이 좋습니다.
    token_to_position: Dict[str, Optional[List[np.ndarray]]] = {}
    # ─────────── 2단계: 에이전트별 경로 생성 ────────────
    for agent_token, agent_list in token_to_trajectory.items():
        token_to_position[agent_token] = []
        if not agent_list:
            token_to_route_roadblock_ids[agent_token] = None
            continue

        roadblock_sequence: List[str] = []
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'] = set()
        inside_connector_flag = False
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'] = set()
        sampled_points_inside_connector: List['Point2D'] = []

        for time_idx, agent_ in enumerate(agent_list):  # 시간 순
            npc_point = agent_.center.point
            token_to_position[agent_token].append(
                np.array([npc_point.x, npc_point.y]))
            current_roadblocks = set(
                scenario.map_api.get_all_map_objects(
                    npc_point, SemanticMapLayer.ROADBLOCK))
            current_connectors = set(
                scenario.map_api.get_all_map_objects(
                    npc_point, SemanticMapLayer.ROADBLOCK_CONNECTOR))
            if current_roadblocks and current_connectors:
                raise ValueError(
                    "Both RoadBlock and RoadBlock-Connector found at the same point. "
                )
            # ── (A) RBC 영역 ─────────────────────────────
            if current_connectors:
                if not inside_connector_flag:  # 새 구간 시작
                    connector_candidate_objects.clear()
                    sampled_points_inside_connector.clear()
                    inside_connector_flag = True
                connector_candidate_objects.update(current_connectors)
                sampled_points_inside_connector.append(npc_point)
                if time_idx == len(
                        agent_list) - 1:  # 마지막 프레임 # 8b5f797c287856f0
                    inside_connector_flag = _decide_roadblock_ids_at_connector(
                        connector_candidate_objects,
                        sampled_points_inside_connector, roadblock_sequence,
                        previous_roadblocks_set, current_roadblocks)
                    inside_connector_flag = False
                    connector_candidate_objects.clear()
                    sampled_points_inside_connector.clear()
                continue

            # ── (B) RoadBlock 영역 ───────────────────────
            if current_roadblocks:
                # 방금 전까지 RBC였다면 후보 결정 필요
                if inside_connector_flag:
                    inside_connector_flag = _decide_roadblock_ids_at_connector(
                        connector_candidate_objects,
                        sampled_points_inside_connector, roadblock_sequence,
                        previous_roadblocks_set, current_roadblocks)
                    inside_connector_flag = False
                    connector_candidate_objects.clear()
                    sampled_points_inside_connector.clear()
                # 현재 RoadBlock id 추가 (중복 방지)
                for roadblock in current_roadblocks:
                    if not roadblock_sequence or roadblock_sequence[
                            -1] != roadblock.id:
                        roadblock_sequence.append(roadblock.id)
                previous_roadblocks_set = current_roadblocks

        token_to_route_roadblock_ids[
            agent_token] = roadblock_sequence if roadblock_sequence else None
        # token_to_position: Dict[str, Optional[List[np.ndarray]]] -> Dict[str, Optional[np.ndarray]]
        if token_to_position[agent_token]:
            token_to_position[agent_token] = np.array(
                token_to_position[agent_token])
    return token_to_route_roadblock_ids, token_to_position


# =====================
# 1. Ego, agent, static coordination transformation
# =====================
def _local_to_local_transforms(global_states1, global_states2):
    """
    Converts the global_states1' local coordinates to global_states2's local coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(global_states2)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states1)

    transforms = np.matmul(local_xform_inv, transforms)
    return transforms


def _state_se2_array_to_transform_matrix(input_data):

    x: float = float(input_data[0])
    y: float = float(input_data[1])
    h: float = float(input_data[2])

    cosine = np.cos(h)
    sine = np.sin(h)

    return np.array([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]])


def _state_se2_array_to_transform_matrix_batch(input_data):

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = np.column_stack((
        input_data[:, 0],
        input_data[:, 1],
        np.cos(input_data[:, 2]),
        np.sin(input_data[:, 2]),
        np.ones_like(input_data[:, 0]),
    ))

    # See below for reshaping example
    reshaping_array = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 array, and then reshapes to a Nx3x3 array
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_array).reshape(-1, 3, 3)


def _transform_matrix_to_state_se2_array_batch(input_data):
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 array of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted array.
    """

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = np.arctan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result


def _global_state_se2_array_to_local(global_states, local_state):
    """
    Transforms the StateSE2 in array from to the frame of reference in local_frame.

    :param global_states: A array of Nx3, where the columns are [x, y, heading].
    :param local_state: A array of [x, y, h] of the frame to which to transform.
    :return: The transformed coordinates.
    """

    local_xform = _state_se2_array_to_transform_matrix(local_state)
    local_xform_inv = np.linalg.inv(local_xform)

    transforms = _state_se2_array_to_transform_matrix_batch(global_states)

    transforms = np.matmul(local_xform_inv, transforms)

    output = _transform_matrix_to_state_se2_array_batch(transforms)

    return output


def _global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * np.cos(
        anchor_heading) + velocity[:, 1] * np.sin(anchor_heading)
    velocity_y = velocity[:, 1] * np.cos(
        anchor_heading) - velocity[:, 0] * np.sin(anchor_heading)

    return np.stack([velocity_x, velocity_y], axis=-1)


def convert_absolute_quantities_to_relative(
        agent_state,  # (N, _)
        ego_state,  # (3,)
        agent_type='ego'):
    """
    Converts the agent or ego history to ego-centric coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = np.array(
        [
            float(ego_state[EgoInternalIndex.x()]),
            float(ego_state[EgoInternalIndex.y()]),
            float(ego_state[EgoInternalIndex.heading()]),
        ],
        dtype=np.float64,
    )

    if agent_type == 'ego':
        time_num, state_dim = agent_state.shape
        new_agent_state = np.zeros((agent_state.shape[0], state_dim + 1),
                                   dtype=np.float64)
        new_agent_state[:, 6:] = agent_state[:, 5:]
        agent_global_poses = agent_state[:, [
            EgoInternalIndex.x(),
            EgoInternalIndex.y(),
            EgoInternalIndex.heading()
        ]]  # (N, 3)
        # agent_global_poses, ego_pose: 절대 좌표계 기준 값
        transforms = _local_to_local_transforms(agent_global_poses, ego_pose)
        transformed_poses = _transform_matrix_to_state_se2_array_batch(
            transforms)  # transformed_poses: ego 좌표계 기준 값
        new_agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0]
        new_agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1]
        new_agent_state[:, 2] = np.cos(transformed_poses[:, 2])
        new_agent_state[:, 3] = np.sin(transformed_poses[:, 2])

        # local vel,acc to local
        # agent_local_vel: 자차량 좌표계 기준 속도 벡터
        agent_local_vel = agent_state[:, [
            EgoInternalIndex.vx(), EgoInternalIndex.vy()
        ]]
        agent_local_vel = np.expand_dims(np.concatenate(
            (agent_local_vel, np.zeros(
                (agent_local_vel.shape[0], 1))), axis=-1),
                                         axis=-1)
        transformed_vel = np.matmul(transforms,
                                    agent_local_vel).squeeze(axis=-1)
        new_agent_state[:, 4] = transformed_vel[:, 0]
        new_agent_state[:, 5] = transformed_vel[:, 1]
        agent_state = new_agent_state
    elif agent_type == 'agent':
        agent_global_poses = agent_state[:, [
            AgentInternalIndex.x(),
            AgentInternalIndex.y(),
            AgentInternalIndex.heading()
        ]]
        agent_global_velocities = agent_state[:, [
            AgentInternalIndex.vx(
            ), AgentInternalIndex.vy()
        ]]
        transformed_poses = _global_state_se2_array_to_local(
            agent_global_poses, ego_pose)
        transformed_velocities = _global_velocity_to_local(
            agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0]
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1]
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2]
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0]
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1]
    elif agent_type == 'static':
        agent_global_poses = agent_state[:, [0, 1, 2]]
        transformed_poses = _global_state_se2_array_to_local(
            agent_global_poses, ego_pose)
        agent_state[:, 0] = transformed_poses[:, 0]
        agent_state[:, 1] = transformed_poses[:, 1]
        agent_state[:, 2] = transformed_poses[:, 2]

    return agent_state


# =====================
# 2. Map coordination transformation
# =====================
def coordinates_to_local_frame(coords, anchor_state, precision=None):
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <np.array: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate array. If None, then it will be inferred from the input precisions.
    :return: <np.array: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError(
                "Mixed datatypes provided to coordinates_to_local_frame without precision specifier."
            )
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = _state_se2_array_to_transform_matrix(anchor_state)
    transform = np.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = np.pad(coords,
                    pad_width=((0, 0), (0, 1)),
                    mode='constant',
                    constant_values=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = np.matmul(transform, coords.T)

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.T
    result = result[:, :2]

    return result


def vector_set_coordinates_to_local_frame(
    coords,
    avails,
    anchor_state,
    output_precision=np.float32,
):
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <np.array: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <np.array: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output array.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """

    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.shape
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords,
                                        anchor_state,
                                        precision=np.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.astype(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords


# =====================
# 3. Numpy-Tensor transformation
# =====================
def convert_to_model_inputs(data, device, squeeze):
    tensor_data = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and v.dtype == np.bool_:
            a = torch.tensor(v, dtype=torch.bool).to(device)
            if not squeeze:
                a = a.unsqueeze(0)
            tensor_data[k] = a
        else:
            b = torch.tensor(v, dtype=torch.float32).to(device)
            if not squeeze:
                b = b.unsqueeze(0)
            tensor_data[k] = b

    return tensor_data
