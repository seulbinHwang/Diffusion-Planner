import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple, Union, List
from typing import Dict, List, Tuple, Optional

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.occupancy_map.strtree_occupancy_map import STRTreeOccupancyMapFactory
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject


def normalize_angle(angle: np.ndarray):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class BreadthFirstSearchRoadBlock:
    """
    A class that performs iterative breadth first search. The class operates on the roadblock graph.
    """

    def __init__(self,
                 start_roadblock_id: str,
                 map_api: Optional[AbstractMap],
                 forward_search: bool = True):
        """
        Constructor of BreadthFirstSearchRoadBlock class
        :param start_roadblock_id: roadblock id where graph starts
        :param map_api: map class in nuPlan
        :param forward_search: whether to search in driving direction, defaults to True
        """
        self._map_api: Optional[AbstractMap] = map_api
        self._queue = deque([self.id_to_roadblock(start_roadblock_id), None])
        self._parent: Dict[str, Optional[RoadBlockGraphEdgeMapObject]] = dict()
        self._forward_search = forward_search

        #  lazy loaded
        self._target_roadblock_ids: List[str] = None

    def search(
            self, target_roadblock_id: Union[str, List[str]],
            max_depth: int) -> Tuple[List[RoadBlockGraphEdgeMapObject], bool]:
        """
        Apply BFS to find route to target roadblock.
        :param target_roadblock_id: id of target roadblock
        :param max_depth: maximum search depth
        :return: tuple of route and whether a path was found
        """

        if isinstance(target_roadblock_id, str):
            target_roadblock_id = [target_roadblock_id]
        self._target_roadblock_ids = target_roadblock_id

        start_edge = self._queue[0]

        # Initial search states
        path_found: bool = False
        end_edge: RoadBlockGraphEdgeMapObject = start_edge
        end_depth: int = 1
        depth: int = 1

        self._parent[start_edge.id + f"_{depth}"] = None

        while self._queue:
            current_edge = self._queue.popleft()

            # Early exit condition
            if self._check_end_condition(depth, max_depth):
                break

            # Depth tracking
            if current_edge is None:
                depth += 1
                self._queue.append(None)
                if self._queue[0] is None:
                    break
                continue

            # Goal condition
            if self._check_goal_condition(current_edge, depth, max_depth):
                end_edge = current_edge
                end_depth = depth
                path_found = True
                break

            neighbors = (current_edge.outgoing_edges if self._forward_search
                         else current_edge.incoming_edges)

            # Populate queue
            for next_edge in neighbors:
                # if next_edge.id in self._candidate_lane_edge_ids_old:
                self._queue.append(next_edge)
                self._parent[next_edge.id + f"_{depth + 1}"] = current_edge
                end_edge = next_edge
                end_depth = depth + 1

        return self._construct_path(end_edge, end_depth), path_found

    def id_to_roadblock(self, id: str) -> RoadBlockGraphEdgeMapObject:
        """
        Retrieves roadblock from map-api based on id
        :param id: id of roadblock
        :return: roadblock class
        """
        block = self._map_api._get_roadblock(id)
        block = block or self._map_api._get_roadblock_connector(id)
        return block

    @staticmethod
    def _check_end_condition(depth: int, max_depth: int) -> bool:
        """
        Check if the search should end regardless if the goal condition is met.
        :param depth: The current depth to check.
        :param target_depth: The target depth to check against.
        :return: whether depth exceeds the target depth.
        """
        return depth > max_depth

    def _check_goal_condition(
        self,
        current_edge: RoadBlockGraphEdgeMapObject,
        depth: int,
        max_depth: int,
    ) -> bool:
        """
        Check if the current edge is at the target roadblock at the given depth.
        :param current_edge: edge to check.
        :param depth: current depth to check.
        :param max_depth: maximum depth the edge should be at.
        :return: True if the lane edge is contain the in the target roadblock. False, otherwise.
        """
        return current_edge.id in self._target_roadblock_ids and depth <= max_depth

    def _construct_path(self, end_edge: RoadBlockGraphEdgeMapObject,
                        depth: int) -> List[RoadBlockGraphEdgeMapObject]:
        """
        Constructs a path when goal was found.
        :param end_edge: The end edge to start back propagating back to the start edge.
        :param depth: The depth of the target edge.
        :return: The constructed path as a list of RoadBlockGraphEdgeMapObject
        """
        path = [end_edge]
        path_id = [end_edge.id]

        while self._parent[end_edge.id + f"_{depth}"] is not None:
            path.append(self._parent[end_edge.id + f"_{depth}"])
            path_id.append(path[-1].id)
            end_edge = self._parent[end_edge.id + f"_{depth}"]
            depth -= 1

        if self._forward_search:
            path.reverse()
            path_id.reverse()

        return (path, path_id)


def get_current_roadblock_candidates(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    heading_error_thresh: float = np.pi / 4,
    displacement_error_thresh: float = 3,
) -> Tuple[RoadBlockGraphEdgeMapObject, List[RoadBlockGraphEdgeMapObject]]:
    """
    Determines a set of roadblock candidate where ego is located
    :param ego_state: class containing ego state
    :param map_api: map object
    :param route_roadblocks_dict: dictionary of on-route roadblocks
    :param heading_error_thresh: maximum heading error, defaults to np.pi/4
    :param displacement_error_thresh: maximum displacement, defaults to 3
    :return: tuple of most promising roadblock and other candidates
    """
    ego_pose: StateSE2 = ego_state.rear_axle
    roadblock_candidates = []

    layers = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    roadblock_dict = map_api.get_proximal_map_objects(point=ego_pose.point,
                                                      radius=1.0,
                                                      layers=layers)
    roadblock_candidates = (
        roadblock_dict[SemanticMapLayer.ROADBLOCK] +
        roadblock_dict[SemanticMapLayer.ROADBLOCK_CONNECTOR])

    if not roadblock_candidates:
        for layer in layers:
            roadblock_id_, distance = map_api.get_distance_to_nearest_map_object(
                point=ego_pose.point, layer=layer)
            roadblock = map_api.get_map_object(roadblock_id_, layer)

            if roadblock:
                roadblock_candidates.append(roadblock)

    on_route_candidates, on_route_candidate_displacement_errors = [], []
    candidates, candidate_displacement_errors = [], []

    roadblock_displacement_errors = []
    roadblock_heading_errors = []

    for idx, roadblock in enumerate(roadblock_candidates):
        lane_displacement_error, lane_heading_error = np.inf, np.inf

        for lane in roadblock.interior_edges:
            lane_discrete_path: List[
                StateSE2] = lane.baseline_path.discrete_path
            lane_discrete_points = np.array(
                [state.point.array for state in lane_discrete_path],
                dtype=np.float64)
            lane_state_distances = ((lane_discrete_points -
                                     ego_pose.point.array[None, ...])**2.0).sum(
                                         axis=-1)**0.5
            argmin = np.argmin(lane_state_distances)

            heading_error = np.abs(
                normalize_angle(lane_discrete_path[argmin].heading -
                                ego_pose.heading))
            displacement_error = lane_state_distances[argmin]

            if displacement_error < lane_displacement_error:
                lane_heading_error, lane_displacement_error = (
                    heading_error,
                    displacement_error,
                )

            if (heading_error < heading_error_thresh and
                    displacement_error < displacement_error_thresh):
                if roadblock.id in route_roadblocks_dict.keys():
                    on_route_candidates.append(roadblock)
                    on_route_candidate_displacement_errors.append(
                        displacement_error)
                else:
                    candidates.append(roadblock)
                    candidate_displacement_errors.append(displacement_error)

        roadblock_displacement_errors.append(lane_displacement_error)
        roadblock_heading_errors.append(lane_heading_error)

    if on_route_candidates:  # prefer on-route roadblocks
        return (
            on_route_candidates[np.argmin(
                on_route_candidate_displacement_errors)],
            on_route_candidates,
        )
    elif candidates:  # fallback to most promising candidate
        return candidates[np.argmin(candidate_displacement_errors)], candidates

    # otherwise, just find any close roadblock
    return (
        roadblock_candidates[np.argmin(roadblock_displacement_errors)],
        roadblock_candidates,
    )


from typing import Dict, List, Tuple, Optional


def _build_route_roadblocks_dict(
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
) -> Dict[str, RoadBlockGraphEdgeMapObject]:
    """주어진 ID 리스트를 이용해, 실제 도로 조각 객체 딕셔너리를 만든다.

    Args:
        map_api: nuPlan 맵 API.
        route_roadblock_ids: 경로를 구성하는 roadblock ID 리스트.

    Returns:
        Dict[str, RoadBlockGraphEdgeMapObject]:
            - key: roadblock id
            - value: 해당 id에 해당하는 RoadBlock 또는 RoadBlockConnector 객체
    """
    route_roadblock_dict: Dict[str, RoadBlockGraphEdgeMapObject] = {}
    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(
            id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        route_roadblock_dict[id_] = block
    return route_roadblock_dict


def _fix_route_start_offroute(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
    route_roadblocks_dict: Dict[str, RoadBlockGraphEdgeMapObject],
    search_depth_backward: int,
    search_depth_forward: int,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """경로의 '시작 지점'이 실제 ego 위치와 어긋나 있을 때 앞부분을 보정한다.

    아이디어:
        1) 먼저 ego 가 지금 어느 도로 조각 위에 있는지 추정한다.
           (headings, 거리 등을 이용해 후보들을 찾음)
        2) ego 가 올라탄 도로 조각이 기존 경로에 없다면:
            - (1안) 경로의 가장 첫 도로에서 ego 쪽으로 거슬러 올라가는 연결을 찾는다.
                   → 찾으면 그 연결을 경로 앞에 붙인다.
            - (2안) 실패하면, ego 쪽에서 기존 경로의 앞부분 쪽으로 이어지는 짧은 경로를 찾는다.
                   → 찾은 지점 이전의 경로는 버리고, ego→경로 연결을 앞에 붙인다.

    이렇게 해서:
        - 차량이 실제로 있는 위치에서부터 경로가 자연스럽게 이어지도록
          경로 앞부분을 한 번 정리해 준다.
    """
    # ego 가 올라탄 도로 조각 및 후보들
    starting_block, starting_block_candidates = get_current_roadblock_candidates(
        ego_state, map_api, route_roadblocks_dict)
    starting_block_ids = [rb.id for rb in starting_block_candidates]

    # 이미 경로 안에 있으면 아무 것도 하지 않음
    if starting_block.id in route_roadblock_ids:
        return route_roadblocks, route_roadblock_ids

    # --- 1단계: 경로 첫 도로에서 ego 쪽으로 거슬러 올라가 보기 (뒤에서부터 잇기) ---
    graph_search = BreadthFirstSearchRoadBlock(route_roadblock_ids[0],
                                               map_api,
                                               forward_search=False)
    (path,
     path_id), path_found = graph_search.search(starting_block_ids,
                                                max_depth=search_depth_backward)

    if path_found and path:
        # path = [경로 첫 도로 ... ego 쪽 도로] 이므로,
        # 이미 경로에 포함된 첫 도로(path[-1])는 제외하고 나머지를 앞에 붙인다.
        route_roadblocks[:0] = path[:-1]
        route_roadblock_ids[:0] = path_id[:-1]
        return route_roadblocks, route_roadblock_ids

    # --- 2단계: ego 쪽에서 기존 경로 앞부분 쪽으로 이어지는 길 찾기 (앞에서부터 잇기) ---
    graph_search = BreadthFirstSearchRoadBlock(starting_block.id,
                                               map_api,
                                               forward_search=True)
    (path,
     path_id), path_found = graph_search.search(route_roadblock_ids[:3],
                                                max_depth=search_depth_forward)

    if path_found and path and path_id:
        # path 의 마지막 id 가 기존 경로 어디에 닿았는지 찾기
        end_roadblock_idx = int(
            np.argmax(np.array(route_roadblock_ids) == path_id[-1]))

        # 닿기 전까지의 기존 경로는 버리고, 그 뒤부터 유지
        route_roadblocks = route_roadblocks[end_roadblock_idx + 1:]
        route_roadblock_ids = route_roadblock_ids[end_roadblock_idx + 1:]

        # ego→경로 연결 path 를 맨 앞에 붙인다
        route_roadblocks[:0] = path
        route_roadblock_ids[:0] = path_id

    return route_roadblocks, route_roadblock_ids


def _fix_disconnected_route_segments(
    map_api: AbstractMap,
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
    search_depth_forward: int,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """경로 중간에 '뜬 구간(도로가 직접 맞닿지 않는 구간)'이 있으면,
    그 사이를 메워 줄 중간 도로 조각들을 찾아 끼워 넣는다.

    예를 들어,
        [A, B, C] 라는 경로가 있을 때
        실제 지도에서는 A → X → Y → B → C 이렇게 이어져야 한다면,
        여기서 X, Y 를 자동으로 찾아 A 와 B 사이에 삽입하는 식이다.

    알고리즘:
        1) 인접한 두 도로 조각 (i, i+1)에 대해
           - i+1 이 i 를 '이전 도로'로 갖고 있는지(incoming 연결) 확인한다.
           - 있으면 이미 지도상으로 붙어 있으므로 통과.
        2) 없다면, i 에서 i+1 로 이어지는 짧은 경로를 한 번 더 찾아본다.
           - 찾은 경로 길이가 3 이상이면, 양 끝(i, i+1)을 제외한 중간 조각들만
             "끼워 넣을 후보"로 기록한다.
        3) 모든 쌍에 대해 후보를 모은 뒤,
           기록해 둔 위치에 중간 조각들을 실제로 삽입한다.
    """
    roadblocks_to_append: Dict[int, Tuple[List[RoadBlockGraphEdgeMapObject],
                                          List[str]]] = {}

    for i in range(len(route_roadblocks) - 1):
        # 다음 도로 조각이 현재 도로를 incoming edge 로 갖는지 확인
        next_incoming_block_ids = [
            _roadblock.id
            for _roadblock in route_roadblocks[i + 1].incoming_edges
        ]
        is_incoming = route_roadblock_ids[i] in next_incoming_block_ids

        if is_incoming:
            continue

        # 직접 붙어 있지 않다면, 사이를 메울 수 있는 짧은 경로 탐색
        graph_search = BreadthFirstSearchRoadBlock(route_roadblock_ids[i],
                                                   map_api,
                                                   forward_search=True)
        (path, path_id), path_found = graph_search.search(
            route_roadblock_ids[i + 1], max_depth=search_depth_forward)

        # path: [i, ..., i+1] 이므로, 중간 조각은 path[1:-1]
        if path_found and path and len(path) >= 3:
            mid_path = path[1:-1]
            mid_ids = path_id[1:-1]
            roadblocks_to_append[i] = (mid_path, mid_ids)

    # 실제 삽입 (앞에서부터 삽입하면 인덱스가 밀리므로 offset 사용)
    offset = 1
    for i, (mid_path, mid_ids) in roadblocks_to_append.items():
        route_roadblocks[i + offset:i + offset] = mid_path
        route_roadblock_ids[i + offset:i + offset] = mid_ids
        offset += len(mid_path)

    return route_roadblocks, route_roadblock_ids


def _maybe_remove_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
    remove_route_loops_flag: bool,
) -> Tuple[List[RoadBlockGraphEdgeMapObject], List[str]]:
    """옵션에 따라 경로 끝부분의 '빙글빙글 도는 루프'를 잘라낸다.

    - remove_route_loops_flag 가 False 이면 아무 작업도 하지 않는다.
    - True 이면 remove_route_loops(...)를 호출해
      교차면적이 큰 루프 구간 이후를 잘라낸다.
    """
    if not remove_route_loops_flag:
        return route_roadblocks, route_roadblock_ids

    # remove_route_loops 는 (route_roadblocks, route_roadblock_ids) 를 반환
    route_roadblocks, route_roadblock_ids = remove_route_loops(
        route_roadblocks, route_roadblock_ids)
    return route_roadblocks, route_roadblock_ids


def route_roadblock_correction(
    ego_state: EgoState,
    map_api: AbstractMap,
    route_roadblock_ids: List[str],
    search_depth_backward: int = 15,
    search_depth_forward: int = 30,
    remove_route_loops_flag: bool = True,
) -> List[str]:
    """주어진 roadblock ID 시퀀스를
    실제 차량이 달릴 수 있는 '자연스러운 경로'에 가깝게 다듬는다.

    이 함수가 하는 일은, 한마디로 말하면
    **"지도 위에 흩어져 있는 도로 조각 ID 목록을,
    끊기지 않고 앞뒤가 자연스럽게 이어지는 경로로 정리"**하는 것이다.

    구체적으로는 세 가지 문제를 순서대로 손본다.

    1) 시작점이 어긋난 경우(차량이 경로 밖에서 시작하는 경우) 앞부분 보정
       -------------------------------------------------------------------
       - 먼저 ego(또는 NPC)가 지금 어느 도로 조각 위에 서 있는지 추정한다.
       - 그 도로 조각이 원래 경로에 없다면,
         1. 경로의 가장 앞 도로에서 ego 쪽으로 거꾸로 따라 올라가며
            이어지는 도로들을 찾는다.
            · 찾으면 그 도로들을 경로 앞에 붙여서 "ego → 경로"가 이어지도록 한다.
         2. 실패하면 이번엔 ego 쪽에서 기존 경로의 앞부분 쪽으로
            짧은 연결 경로를 찾아본다.
            · 연결 지점 이전의 오래된 경로는 버리고,
              ego → 연결 지점까지의 경로를 앞에 붙인다.

       이렇게 해서, “현재 위치에서부터 경로가 시작되는 느낌”이 되도록
       출발 부분을 한 번 정리한다.

    2) 중간에 끊긴 구간 메우기
       ------------------------
       - 경로 안에서 연달아 나오는 두 도로 조각 (A, B)을 살펴보며,
         실제 지도에서도 A 바로 다음에 B 로 진입 가능한지 확인한다.
       - 만약 둘 사이에 직접적인 이어짐이 없다면,
         지도에서 A 에서 출발해 B 에 도달하는 짧은 도로열을 다시 찾아보고,
         그 중간 조각들만 A와 B 사이에 끼워 넣는다.
       - 예를 들면, 경로가 [A, B, C] 인데
         실제론 A → X → Y → B → C 여야 한다면,
         X, Y 를 자동으로 찾아 A 와 B 사이에 넣는 식이다.

       이 과정을 거치면, 경로 중간에 순간이동하는 느낌 없이
       한 조각에서 다음 조각으로 자연스럽게 이어진다.

    3) 끝부분에서 빙글빙글 도는 루프 잘라내기(선택)
       --------------------------------------------
       - 어떤 시나리오에서는 경로가 뒤쪽에서 자기 자신과 크게 겹치며
         "빙글빙글 도는 구간" 이 생길 수 있다.
       - `remove_route_loops_flag=True` 이면,
         이런 루프 구간을 찾아 그 지점 이후를 잘라낸다.
       - False 로 두면 루프를 그대로 둔다
         (예: NPC의 전체 이동 궤적을 보고 싶을 때).

    Args:
        ego_state:
            - 현재 차량(ego 또는 NPC)의 상태.
            - 이 위치를 기준으로 "경로가 어디서 시작해야 자연스러운지" 를 계산한다.
        map_api:
            - nuPlan 지도 API 핸들.
        route_roadblock_ids: List[str]
            - 보정 전 경로를 이루는 roadblock id 리스트.
        search_depth_backward:
            - ego 기준 앞부분을 보정할 때,
              경로의 첫 roadblock 쪽으로 최대 몇 단계까지 거슬러 올라갈지(대략적인 깊이 제한).
        search_depth_forward:
            - 중간 연결/앞부분 보정 시,
              앞으로 얼마나 짧은 경로까지 탐색할지(깊이 제한).
        remove_route_loops_flag:
            - True 이면 끝부분 루프를 잘라낸다.
            - False 이면 그대로 둔다.

    Returns:
        List[str]:
            - 보정이 끝난 후의 roadblock id 시퀀스.
            - 항상 입력과 같은 형식의 리스트이며,
              길이는 짧아지거나 길어질 수 있다.
    """
    # 1) id → roadblock 객체 매핑 준비
    route_roadblocks_dict: Dict[
        str, RoadBlockGraphEdgeMapObject] = _build_route_roadblocks_dict(
            map_api=map_api,
            route_roadblock_ids=route_roadblock_ids,
        )

    # dict 는 삽입 순서를 보존하므로, 원래 경로 순서 그대로 리스트화
    route_roadblocks: List[RoadBlockGraphEdgeMapObject] = list(
        route_roadblocks_dict.values())
    route_roadblock_ids_ordered: List[str] = list(route_roadblocks_dict.keys())

    # 2) 시작점 보정 (ego 가 경로 바깥에서 시작할 가능성 처리)
    route_roadblocks, route_roadblock_ids_ordered = _fix_route_start_offroute(
        ego_state=ego_state,
        map_api=map_api,
        route_roadblocks=route_roadblocks,
        route_roadblock_ids=route_roadblock_ids_ordered,
        route_roadblocks_dict=route_roadblocks_dict,
        search_depth_backward=search_depth_backward,
        search_depth_forward=search_depth_forward,
    )

    # 3) 경로 중간 끊긴 구간 메우기
    route_roadblocks, route_roadblock_ids_ordered = \
        _fix_disconnected_route_segments(
            map_api=map_api,
            route_roadblocks=route_roadblocks,
            route_roadblock_ids=route_roadblock_ids_ordered,
            search_depth_forward=search_depth_forward,
        )

    # 4) 필요 시 루프 제거
    route_roadblocks, route_roadblock_ids_ordered = _maybe_remove_loops(
        route_roadblocks=route_roadblocks,
        route_roadblock_ids=route_roadblock_ids_ordered,
        remove_route_loops_flag=remove_route_loops_flag,
    )

    # 최종 결과: id 리스트만 반환
    return route_roadblock_ids_ordered


def remove_route_loops(
    route_roadblocks: List[RoadBlockGraphEdgeMapObject],
    route_roadblock_ids: List[str],
) -> Tuple[List[str], List[RoadBlockGraphEdgeMapObject]]:
    """
    Remove ending of route, if the roadblock are intersecting the route (forming a loop).
    :param route_roadblocks: input route roadblocks
    :param route_roadblock_ids: input route roadblocks ids
    :return: tuple of ids and roadblocks of route without loops
    """

    roadblock_occupancy_map = None
    loop_idx = None

    for idx, roadblock in enumerate(route_roadblocks):
        # loops only occur at intersection, thus searching for roadblock-connectors.
        if str(roadblock.__class__.__name__) == "NuPlanRoadBlockConnector":
            if not roadblock_occupancy_map:
                roadblock_occupancy_map = STRTreeOccupancyMapFactory.get_from_geometry(
                    [roadblock.polygon], [roadblock.id])
                continue

            strtree, index_by_id = roadblock_occupancy_map._build_strtree()
            indices = strtree.query(roadblock.polygon)
            if len(indices) > 0:
                for geom in strtree.geometries.take(indices):
                    area = geom.intersection(roadblock.polygon).area
                    if area > 1:
                        loop_idx = idx
                        break
                if loop_idx:
                    break

            roadblock_occupancy_map.insert(roadblock.id, roadblock.polygon)

    if loop_idx:
        route_roadblocks = route_roadblocks[:loop_idx]
        route_roadblock_ids = route_roadblock_ids[:loop_idx]

    return route_roadblocks, route_roadblock_ids
