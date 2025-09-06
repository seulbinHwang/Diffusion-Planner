"""
Module: Coordination Transformation Functions and Numpy-Tensor Transformation
Description: This module contains functions for transforming the coordination to ego-centric coordination and Numpy-Tensor transformation.

Categories:
    1. Ego, agent, static coordination transformation
    2. Map coordination transformation
    3. Numpy-Tensor transformation
"""
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from types import SimpleNamespace
import torch
from typing import Deque, Dict, List, Optional, Set, Type, Tuple
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from shapely.geometry import Point
from collections import defaultdict
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.actor_state.state_representation import Point2D
import shapely.geometry as geom
from shapely import affinity
import math
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from typing import List, Optional, Union
import numpy as np


def get_neighbor_track_tokens(
    present_tracked_objects: Union[object, List[object]],
    neighbor_indices: Union[np.ndarray, List[int]],
    agents_num: int,
) -> List[Optional[str]]:
    """현재 프레임의 트래킹 객체와 `neighbor_indices`를 이용해,
    `neighbor_agents_past`에 최종 선정된 에이전트(agents_num)의 track token을 반환합니다.

    이 함수는 `agent_past_process(...)`가 반환한 `neighbor_indices`의 **순서가
    `neighbor_agents_past`의 0축(에이전트 축) 순서와 동일**하다는 가정하에 동작합니다.
    각 인덱스는 현재 프레임의 트래킹 객체 리스트(또는 컨테이너)에서의 위치를 가리킵니다.

    Args:
        present_tracked_objects (Union[object, List[object]]):
            - 현재 프레임의 트래킹 객체 모음.
            - NuPlan의 `TrackedObjects` 컨테이너(속성 `.tracked_objects` 보유) 또는
              `List[TrackedObject]` 형태 모두 지원합니다.
            - 각 객체는 `track_token`(str) 속성을 갖는다고 가정합니다.
        neighbor_indices (Union[np.ndarray, List[int]]):
            - shape: (K,), K ≤ agents_num
            - `agent_past_process`가 선택한 이웃 에이전트들의 **현재 프레임 인덱스** 배열.
              길이가 `agents_num`보다 짧을 수 있으며, 이 경우 나머지는 `None`으로 채웁니다.
        agents_num (int):
            - `neighbor_agents_past`의 에이전트 축 크기(고정 개수).

    Returns:
        List[Optional[str]]:
            - `neighbor_track_token`, 길이 = `agents_num`
            - 각 원소는 선택된 에이전트의 `track_token`(str) 또는 매칭 불가 시 `None`.

    Examples:
        >>> # present_tracked_objects: TrackedObjects 컨테이너 혹은 List[TrackedObject]
        >>> # neighbor_indices: np.array([5, 2, 0])  # 세 명만 실제 선택됨
        >>> # agents_num = 5  # 패딩 포함 고정 크기
        >>> tokens = get_neighbor_track_tokens(present_tracked_objects, neighbor_indices, agents_num)
        >>> len(tokens)
        5
        >>> tokens[:3]   # 앞의 3개는 실제 선택된 에이전트 토큰
        ['abc123', 'def456', 'ghi789']
        >>> tokens[3:]   # 남는 두 개는 패딩: None
        [None, None]
    """
    # 1) 현재 프레임의 객체 리스트 확보 (컨테이너/리스트 모두 지원)
    if hasattr(present_tracked_objects, "tracked_objects"):
        objects_list = list(
            present_tracked_objects.tracked_objects)  # NuPlan 컨테이너
    elif isinstance(present_tracked_objects, (list, tuple)):
        objects_list = list(present_tracked_objects)  # 이미 리스트/튜플
    else:
        # 마지막 방어선: 이터러블이면 리스트로 변환, 아니면 빈 리스트
        try:
            objects_list = list(present_tracked_objects)
        except TypeError:
            objects_list = []

    # 2) 현재 프레임 토큰 테이블 구성
    present_tokens: List[Optional[str]] = []
    for obj in objects_list:
        token = getattr(obj, "track_token", None)
        if token is None:
            # 혹시 구현체에 따라 이름이 다를 수 있으므로 보조 키도 점검
            token = getattr(obj, "token", None)
        present_tokens.append(token)

    # 3) neighbor_indices 정규화 (길이 K ≤ agents_num)
    if neighbor_indices is None:
        idx_array = np.empty((0,), dtype=int)
    else:
        idx_array = np.asarray(neighbor_indices).reshape(-1)
        # 실수형으로 들어온 경우가 있어도 안전하게 정수 변환
        idx_array = idx_array.astype(int, copy=False)

    # 4) 에이전트 개수(agents_num)에 맞춰 토큰 리스트 생성 (부족분은 None 패딩)
    neighbor_track_token: List[Optional[str]] = []
    total_present = len(present_tokens)

    for i in range(agents_num):
        token_i: Optional[str] = None
        if i < idx_array.shape[0]:
            idx = int(idx_array[i])
            if 0 <= idx < total_present:
                token_i = present_tokens[idx]
        neighbor_track_token.append(token_i)

    return neighbor_track_token


def get_directional_proximal_map_objects(
    map_api: AbstractMap,
    point: Point2D,
    radius: float,
    heading: float,
    layers: List[SemanticMapLayer],
) -> Dict[SemanticMapLayer, List[MapObject]]:
    """주어진 heading 방향에 맞춘 정사각형 영역 내의 객체를 조회한다.

    Args:
        point (Point2D): [m] 중심점의 좌표.
        radius (float): [m] 정사각형 한 변의 절반 길이.
        heading (float): [rad] 정사각형을 회전시킬 heading. 0은 세계 좌표계 x축과 정렬.
        layers (List[SemanticMapLayer]): 조회할 레이어 목록.

    Returns:
        Dict[SemanticMapLayer, List[MapObject]]: 레이어별로 발견된 MapObject 목록.
    """

    x_min, x_max = point.x - radius, point.x + radius
    y_min, y_max = point.y - radius, point.y + radius
    patch = geom.box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch,
                            math.degrees(heading),
                            origin=(point.x, point.y))

    supported_layers = map_api.get_available_map_objects()
    unsupported_layers = [
        layer for layer in layers if layer not in supported_layers
    ]

    assert (
        len(unsupported_layers) == 0
    ), f"Object representation for layer(s): {unsupported_layers} is unavailable"

    object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

    for layer in layers:
        object_map[layer] = map_api._get_proximity_map_object(patch, layer)

    return object_map


def get_npc_route_roadblock_ids(
    scenario: NuPlanScenario,
    neighbor_track_token: List[Optional[str]],
    radius: float = 100.,
    vehicle_num: int = 256,
) -> Dict[str, Optional[List[str]]]:

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

    def _filter_vehicle_tokens_in_square(ego_state: EgoState,
                                         detections: DetectionsTracks,
                                         radius: float) -> Set[str]:
        """정사각형 영역 내에 위치한 차량 토큰 추출.

        Args:
            ego_state (EgoState): 정사각형 중심이 되는 ego 상태.
            detections (DetectionsTracks): 초기 시점의 트래킹 결과.
            radius (float): 정사각형 한 변의 절반 길이 [m].

        Returns:
            Set[str]: 영역 내부 차량의 트랙 토큰 집합.
        """
        center_x = ego_state.rear_axle.x
        center_y = ego_state.rear_axle.y
        min_x, max_x = center_x - radius, center_x + radius
        min_y, max_y = center_y - radius, center_y + radius
        tokens: Set[str] = set()
        for obj in detections.tracked_objects:
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            px, py = obj.center.x, obj.center.y
            if min_x <= px <= max_x and min_y <= py <= max_y:
                tokens.add(obj.track_token)
        return tokens

    def _select_nearest_vehicle_tokens(ego_state: EgoState,
                                       detections: DetectionsTracks,
                                       candidates: Set[str],
                                       vehicle_num: int) -> Set[str]:
        """ego와의 거리 순서대로 최대 ``vehicle_num``개 차량 토큰 선택.

        Args:
            ego_state (EgoState): 거리 계산 기준이 되는 ego 상태.
            detections (DetectionsTracks): 초기 시점의 트래킹 결과.
            candidates (Set[str]): 거리 비교 대상 토큰 집합.
            vehicle_num (int): 선택할 최대 차량 대수.

        Returns:
            Set[str]: 거리 기준 상위 ``vehicle_num``개의 차량 토큰.
        """
        center_x = ego_state.rear_axle.x
        center_y = ego_state.rear_axle.y
        distances: List[Tuple[float, str]] = []
        for obj in detections.tracked_objects:
            if obj.track_token not in candidates or obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            dx = obj.center.x - center_x
            dy = obj.center.y - center_y
            dist = float(np.hypot(dx, dy))
            distances.append((dist, obj.track_token))
        distances.sort(key=lambda x: x[0])
        return {token for _, token in distances[:vehicle_num]}

    # ─────────── 1단계: 차량별 프레임 수집 ────────────
    token_to_trajectory: Dict[str, List['SceneObject']] = defaultdict(list)
    ##########
    initial_detections = scenario.get_tracked_objects_at_iteration(0)
    square_tokens = _filter_vehicle_tokens_in_square(scenario.initial_ego_state,
                                                     initial_detections, radius)
    valid_tokens = _select_nearest_vehicle_tokens(scenario.initial_ego_state,
                                                  initial_detections,
                                                  square_tokens, vehicle_num)
    ##########
    total_horizon_s = (
        scenario.get_time_point(scenario.get_number_of_iterations() - 1).time_s
        - scenario.get_time_point(0).time_s)
    for det_batch in scenario.get_future_tracked_objects(0, total_horizon_s):
        for det in det_batch.tracked_objects:
            if det.tracked_object_type == TrackedObjectType.VEHICLE and (
                    det.track_token in valid_tokens):
                token_to_trajectory[det.track_token].append(det)

    token_to_route_roadblock_ids: Dict[str, Optional[List[str]]] = {}
    # ─────────── 2단계: 에이전트별 경로 생성 ────────────
    for agent_token, agent_list in token_to_trajectory.items():
        if not agent_list:
            token_to_route_roadblock_ids[agent_token] = None
            continue
        ###########
        roadblock_sequence: List[str] = []
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'] = set()
        inside_connector_flag = False
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'] = set()
        sampled_points_inside_connector: List['Point2D'] = []

        for time_idx, agent_ in enumerate(agent_list):  # 시간 순
            npc_point = agent_.center.point
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

        if roadblock_sequence:
            start = agent_list[0]
            npc_state = SimpleNamespace(rear_axle=StateSE2(
                start.center.x, start.center.y, start.center.heading))
            corrected_ids = route_roadblock_correction(
                npc_state,
                scenario.map_api,
                roadblock_sequence,
                remove_route_loops_flag=False)
            token_to_route_roadblock_ids[agent_token] = corrected_ids
        else:
            token_to_route_roadblock_ids[agent_token] = None
    return token_to_route_roadblock_ids


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
