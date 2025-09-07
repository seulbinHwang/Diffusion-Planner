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
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from typing import List, Optional, Union
import numpy as np

from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import math
import shapely.geometry as geom
from shapely import affinity
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map import SemanticMapLayer
from nuplan.common.actor_state.state_representation import Point2D
from typing import Set
import math
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import \
    DetectionsTracks
from nuplan.common.actor_state.tracked_objects_types import \
    TrackedObjectType

def _map_object_to_geometry(obj: MapObject) -> Optional[geom.base.BaseGeometry]:
    """MapObject로부터 Shapely 기하를 얻는다.

    우선순위:
        1) obj.polygon 이 존재하면 그대로 사용
        2) obj.baseline_path.discrete_path 가 있으면 LineString으로 구성
        3) 둘 다 없으면 None

    Args:
        obj (MapObject): NuPlan 맵 객체.

    Returns:
        Optional[geom.base.BaseGeometry]: Shapely Polygon/LineString/Point 등. 없으면 None.
    """
    # 1) 다각형이 있는 타입(예: Lane, RoadBlock, Connector 등)
    polygon = getattr(obj, "polygon", None)
    if polygon is not None:
        return polygon

    # 2) 차선류 등: baseline_path → LineString
    baseline_path = getattr(obj, "baseline_path", None)
    if baseline_path is not None and hasattr(baseline_path, "discrete_path"):
        pts = [(n.x, n.y) for n in baseline_path.discrete_path]
        if len(pts) >= 2:
            return geom.LineString(pts)
        elif len(pts) == 1:
            return geom.Point(pts[0])

    # 3) 기타(공개 속성으로는 기하 획득 불가)
    return None


def get_directional_proximal_map_objects(
    map_api: AbstractMap,
    point: Point2D,
    heading: float,
    radius: float,
    layers: List[SemanticMapLayer],
) -> Dict[SemanticMapLayer, List[MapObject]]:
    """heading 방향으로 회전된 정사각형(변=2*radius) 내부와 교차하는 객체를 조회한다.

    구현 방식(공개 API 기반):
        1) 먼저 R' = radius * sqrt(2) 로 키운 축정렬 AABB로 coarse 후보를
           `get_proximal_map_objects`로 가져온다.
        2) Shapely로 heading만큼 회전시킨 정사각형 패치를 만들고,
           각 MapObject의 기하(Polygon 또는 LineString)와 교차하는 것만 남긴다.

    Args:
        map_api (AbstractMap): 맵 API(NuPlanMap 등).
        point (Point2D): [m] 패치 중심 좌표.
        radius (float): [m] 회전 정사각형의 반변 길이(= 한 변의 절반).
        heading (float): [rad] 패치 회전 각. 0이면 세계 좌표 x축 정렬.
        layers (List[SemanticMapLayer]): 조회할 레이어 목록.

    Returns:
        Dict[SemanticMapLayer, List[MapObject]]: 레이어별 교차 객체 목록.
    """
    supported_layers = map_api.get_available_map_objects()
    unsupported = [ly for ly in layers if ly not in supported_layers]
    assert len(unsupported) == 0, (
        f"Object representation for layer(s): {unsupported} is unavailable")

    # (1) AABB(축정렬) 기반 coarse 후보 조회: 회전 정사각형을 항상 포함하도록 R' = R * sqrt(2)
    coarse_radius = float(radius) * math.sqrt(2.0)
    coarse_candidates = map_api.get_proximal_map_objects(
        point, coarse_radius, layers)

    # (2) 회전된 정사각형 패치 생성
    x_min, x_max = point.x - radius, point.x + radius
    y_min, y_max = point.y - radius, point.y + radius
    rotated_patch = geom.box(x_min, y_min, x_max, y_max)
    rotated_patch = affinity.rotate(rotated_patch,
                                    math.degrees(heading),
                                    origin=(point.x, point.y))

    # (3) 실제 교차 여부로 필터링
    filtered: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)
    for layer in layers:
        objs = coarse_candidates.get(layer, [])
        for obj in objs:
            shape = _map_object_to_geometry(obj)
            if shape is None:
                # 공개 속성으로 기하를 얻을 수 없는 타입은 보수적으로 스킵(축정렬 결과만으로는 방향성 보장 불가)
                continue
            if shape.intersects(rotated_patch):
                filtered[layer].append(obj)

    return filtered


def build_agent_route_lane_order(
    npc_route_indices: List[List[int]],
    lane_num: Optional[int] = None,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    """NPC별로 선택된 route 차선 인덱스(npc_route_indices)를 기준으로,
    각 에이전트에 대해 '가까운 차선 순서'를 정수 랭크로 기록한 행렬을 생성합니다.

    규칙:
        - 한 에이전트의 npc_route_indices[i] 가 [5, 2, 7] 이라면,
          해당 에이전트 행에서 lane 5→0, lane 2→1, lane 7→2 로 표기합니다.
        - 그 에이전트의 route가 아닌 차선은 -1로 표기합니다.
        - npc_route_indices[i] 안에 중복 인덱스가 있으면 최초 등장에만 랭크를 부여합니다.

    Args:
        npc_route_indices (List[List[int]]):
            - 길이: agent_num
            - 각 원소는 해당 에이전트가 선택한 '가까운 순' 차선 인덱스 리스트
              (예: [lane_idx_0, lane_idx_1, ...]).
        lane_num (Optional[int], optional):
            - 전체 차선 개수. 지정하지 않으면 npc_route_indices 전체에서
              등장한 최대 인덱스의 +1로 유추합니다.
              (모든 리스트가 비어 있으면 0으로 처리)
        dtype (np.dtype, optional):
            - 반환 행렬의 dtype. 기본값 np.int32.

    Returns:
        np.ndarray:
            - `agent_route_lane_order`, shape = (agent_num, lane_num), dtype = `dtype`
            - 각 [i, j] 원소는:
                · j번 차선이 에이전트 i의 npc_route에서 가까운 순서로 몇 번째인지(0,1,2,...)를 나타냄
                · 해당 에이전트의 route가 아니면 -1

    Raises:
        ValueError: npc_route_indices에 음수 인덱스가 있거나,
                    lane_num 유추 후 범위를 벗어나는 인덱스가 발견된 경우.

    Examples:
        >>> npc_route_indices = [[5, 2, 7], [1, 0]]
        >>> arr = build_agent_route_lane_order(npc_route_indices, lane_num=8)
        >>> # arr[0]: lane 5->0, lane 2->1, lane 7->2, 나머지 -1
        >>> # arr[1]: lane 1->0, lane 0->1, 나머지 -1
    """
    agent_num: int = len(npc_route_indices)

    # lane_num 미지정 시, 등장한 최대 인덱스 기반으로 유추
    if lane_num is None:
        max_idx = -1
        for idx_list in npc_route_indices:
            if any(i < 0 for i in idx_list):
                raise ValueError("npc_route_indices에 음수 인덱스가 포함되어 있습니다.")
            if idx_list:
                max_idx = max(max_idx, max(idx_list))
        lane_num = max_idx + 1 if max_idx >= 0 else 0

    # 기본값 -1로 초기화
    agent_route_lane_order = np.full((agent_num, lane_num), -1, dtype=dtype)

    # 에이전트별로 가까운 순서 랭크 부여
    for agent_i, lane_list in enumerate(npc_route_indices):
        seen = set()
        rank = 0
        for lane_idx in lane_list:
            if lane_idx < 0:
                raise ValueError(f"음수 인덱스가 발견되었습니다: {lane_idx}")
            if lane_idx >= lane_num:
                raise ValueError(
                    f"lane_idx {lane_idx} 가 lane_num={lane_num} 범위를 벗어났습니다.")
            if lane_idx in seen:
                continue
            agent_route_lane_order[agent_i, lane_idx] = rank
            seen.add(lane_idx)
            rank += 1

    return agent_route_lane_order


def _select_token_and_ordered_npc_route_indices(
    token_to_lane_on_routes: Dict[str, List[bool]],  # 길이 == agent_num (키 개수)
    neighbor_track_token: List[Optional[str]],  # 길이 == agent_num, ego와 가까운 순서
    neighbor_agents_current: np.ndarray,  # shape: (agent_num, 11), ego와 가까운 순서
    vector_map_lanes: np.ndarray,  # shape: (lane_num, P, D), 좌표는 [:, :, :2]
    max_route_lanes: int,
) -> np.ndarray:
    """토큰별 '경로 위 차선' 인덱스를 가까운 차선부터 최대 `max_route_lanes`개 선택하고,
    에이전트(이웃) 순서(이미 ego와 가까운 순)대로 `npc_route_indices`를 생성합니다.

    설계 가정:
    - `token_to_lane_on_routes`, `neighbor_track_token`, `neighbor_agents_current`의 길이는 모두 `agent_num`으로 동일합니다.
    - `neighbor_track_token`와 `neighbor_agents_current`는 **ego와 가까운 순서**로 이미 정렬되어 있습니다.
    - `max_route_lanes < lane_num`이 보장됩니다.

    선택 기준(각 에이전트 i에 대해):
    - 모든 차선과 에이전트 현재 위치 간 최소거리(폴리라인 점들 중 최소)를 계산해 차선을 가까운→먼 순으로 정렬
    - 그 순서대로 `token_to_lane_on_routes[token][lane_idx]`가 True인 차선만 최대 `max_route_lanes`개 선택

    Args:
        token_to_lane_on_routes (Dict[str, List[bool]]):
            - 키: 에이전트 토큰(str) — 총 키 개수 == agent_num
            - 값: 길이 `lane_num`의 불리언 리스트 (해당 차선이 NPC 경로 위인지)
        neighbor_track_token (List[Optional[str]]): 길이 `agent_num`. 에이전트 토큰(없을 수 있어 None).
        neighbor_agents_current (np.ndarray): shape=(agent_num, 11). 거리 계산에 x=[:,0], y=[:,1] 사용.
        vector_map_lanes (np.ndarray): shape=(lane_num, P, D). 거리 계산에 좌표 성분 [:, :, :2] 사용.
        max_route_lanes (int): NPC 당 최대 선택할 차선 개수. (항상 lane_num보다 작음)

    Returns:
        np.ndarray:
            - `agent_route_lane_order`, shape = (agent_num, lane_num), dtype = `dtype`
            - 각 [i, j] 원소는:
                · j번 차선이 에이전트 i의 npc_route에서 가까운 순서로 몇 번째인지(0,1,2,...)를 나타냄
                · 해당 에이전트의 route가 아니면 -1


    Raises:
        ValueError: 입력 shape/길이가 맞지 않거나, 불리언 마스크 길이 ≠ lane_num 인 경우.
    """
    # --- 기본 검증 ---
    if neighbor_agents_current.ndim != 2 or neighbor_agents_current.shape[1] < 2:
        raise ValueError(
            f"`neighbor_agents_current` shape가 올바르지 않습니다: {neighbor_agents_current.shape}"
        )
    agent_num = neighbor_agents_current.shape[0]
    lane_num = int(vector_map_lanes.shape[0])

    if len(neighbor_track_token) != agent_num:
        raise ValueError(
            "`neighbor_track_token` 길이와 `neighbor_agents_current`의 첫 축 크기가 다릅니다."
        )
    if vector_map_lanes.ndim != 3 or vector_map_lanes.shape[2] < 2:
        raise ValueError(
            f"`vector_map_lanes` shape가 올바르지 않습니다: {vector_map_lanes.shape}")
    if not (0 < max_route_lanes < lane_num):
        raise ValueError(
            f"`max_route_lanes`는 0 < max_route_lanes < lane_num 을 만족해야 합니다. "
            f"(max_route_lanes={max_route_lanes}, lane_num={lane_num})")

    # --- 준비: 차선 좌표 ---
    lanes_xy = vector_map_lanes[:, :, :2]  # (lane_num, P, 2)

    token_to_route_indices: Dict[str, List[int]] = {}
    npc_route_indices: List[List[int]] = []

    # 중복/None 토큰을 위해 고유 키를 만드는 헬퍼
    def _unique_key(base: Optional[str], idx: int) -> str:
        key = base if base is not None else "__none__"
        # 동일 키가 이미 존재하면 인덱스 suffix로 고유화
        return key if key not in token_to_route_indices else f"{key}#{idx}"

    # --- 에이전트(이미 ego 근접 순) 순회 ---
    for agent_idx in range(agent_num):
        token = neighbor_track_token[agent_idx]
        if token is None:
            continue

        # 마스크 획득 (없으면 전부 False로 처리)
        ###############################
        lane_on_routes = token_to_lane_on_routes[
            token]  # (Dict[str, List[bool]]):
        # lane_on_routes: List[bool], 길이 == lane_num

        agent_xy = neighbor_agents_current[agent_idx, :2]  # (2,)

        # 각 차선 폴리라인과의 최소거리 (lane_num,)
        diff = lanes_xy - agent_xy[None, None, :]  # (lane_num, P, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (lane_num, P)
        min_dists = np.min(dists, axis=1)  # (lane_num,)
        lane_dist_order = np.argsort(min_dists)  # (lane_num,)

        # 가까운 순서대로 True인 차선만 최대 max_route_lanes개 선택
        selected: List[int] = []
        lane_on_routes_arr = np.asarray(lane_on_routes, dtype=bool)
        for lane_idx in lane_dist_order:
            if lane_on_routes_arr[lane_idx]:
                selected.append(int(lane_idx))
                if len(selected) >= max_route_lanes:
                    break

        token_to_route_indices[token] = selected
        npc_route_indices.append(selected)

    # 반환 길이 검증(요구조건: 둘 다 agent_num 크기 보장)
    assert len(token_to_route_indices) == agent_num, \
        f"token_to_route_indices 키 개수({len(token_to_route_indices)}) != agent_num({agent_num})"
    assert len(npc_route_indices) == agent_num, \
        f"npc_route_indices 길이({len(npc_route_indices)}) != agent_num({agent_num})"
    agent_route_lane_order = build_agent_route_lane_order(
        npc_route_indices, lane_num=lane_num)  # 검증용 호출
    return agent_route_lane_order


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


def get_npc_route_roadblock_ids(
    scenario: NuPlanScenario,
    neighbor_track_token: List[Optional[str]],
    radius: float = 100.,
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



    def _filter_vehicle_tokens_in_square(
        ego_state: EgoState,
        detections: DetectionsTracks,
        radius: float,
    ) -> Set[str]:
        """ego heading에 정렬된 정사각형(변=2*radius) 내부에 있는 차량 토큰을 반환한다.

        정사각형 축 정의:
            - 좌표계 원점: ego rear_axle (x, y)
            - 축 방향: ego heading에 정렬된 로컬 x/y 축(ego 좌표계)
            - 포함 판정: 로컬 좌표 (x_local, y_local) 가 모두 [-radius, +radius] 범위에 있으면 포함

        구현 메모:
            - 월드 → ego 로컬 변환은 ego heading에 대해 -heading 회전과 평행이동을 적용한 것과 동일.
            - 수치적 안정성을 위해 비교는 경계 포함(<=)으로 처리.

        Args:
            ego_state: ego의 상태(특히 rear_axle.x, rear_axle.y, rear_axle.heading 사용)
            detections: 현재 프레임의 트래킹 결과(DetectionsTracks)
            radius: 정사각형 한 변의 절반 길이 [m]

        Returns:
            Set[str]: 정사각형 내부에 있는 VEHICLE 타입 객체들의 track_token 집합
        """
        cx = float(ego_state.rear_axle.x)
        cy = float(ego_state.rear_axle.y)
        h = float(ego_state.rear_axle.heading)
        cos_h = math.cos(h)
        sin_h = math.sin(h)

        tokens: Set[str] = set()

        for obj in detections.tracked_objects:
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue

            # 월드 좌표에서 ego 원점으로 평행이동
            dx = float(obj.center.x) - cx
            dy = float(obj.center.y) - cy

            # 월드 → ego 로컬 회전(각도 -h 적용과 동일)
            # x_local =  cos(h)*dx + sin(h)*dy
            # y_local = -sin(h)*dx + cos(h)*dy
            x_local = dx * cos_h + dy * sin_h
            y_local = -dx * sin_h + dy * cos_h

            if (-radius <= x_local <= radius) and (-radius <= y_local <=
                                                   radius):
                token = getattr(obj, "track_token", None)
                if token is not None:
                    tokens.add(token)

        return tokens

    # ─────────── 1단계: 차량별 프레임 수집 ────────────
    token_to_trajectory: Dict[str, List['SceneObject']] = defaultdict(list)
    ##########
    initial_detections = scenario.get_tracked_objects_at_iteration(0)
    # 1) 반경 내 토큰
    square_tokens = _filter_vehicle_tokens_in_square(scenario.initial_ego_state,
                                                     initial_detections, radius)
    # 2) neighbor_track_token과의 교집합으로 후보 제한(None 제거)
    neighbor_token_set = {t for t in neighbor_track_token if t is not None}
    candidate_tokens = square_tokens & neighbor_token_set
    # 3) ego와의 거리 순으로 상위 vehicle_num개만 선택

    ##########
    # a = scenario.get_time_point(scenario.get_number_of_iterations() - 1).time_s
    # b = scenario.get_time_point(0).time_s
    # total_horizon_s = a - b
    # print("a:", a, "b:", b, "total_horizon_s:", total_horizon_s)
    total_horizon_s = 8.
    for det_batch in scenario.get_future_tracked_objects(0, total_horizon_s):
        for det in det_batch.tracked_objects:
            if det.tracked_object_type == TrackedObjectType.VEHICLE and (
                    det.track_token in candidate_tokens):
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
