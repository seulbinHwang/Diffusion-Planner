"""
Module: Coordination Transformation Functions and Numpy-Tensor Transformation
Description: This module contains functions for transforming the coordination to ego-centric coordination and Numpy-Tensor transformation.

Categories:
    1. Ego, agent, static coordination transformation
    2. Map coordination transformation
    3. Numpy-Tensor transformation
"""
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from types import SimpleNamespace
from nuplan.database.nuplan_db.nuplan_scenario_queries import \
    get_end_sensor_time_from_db
from nuplan.database.nuplan_db.nuplan_db_utils import get_lidarpc_sensor_data
import torch
from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObject
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex, AgentInternalIndex
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from shapely.geometry import Point
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from typing import List, Optional, Union, Sequence
import numpy as np
from nuplan.common.actor_state.tracked_objects import TrackedObject
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
# utils.py (적절한 위치에 추가)
from typing import List, Optional, Sequence
import warnings
from types import SimpleNamespace
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.maps.nuplan_map.utils import get_roadblock_ids_from_trajectory


def get_npc_route_roadblock_ids(
        scenario: NuPlanScenario,
        sampled_past_observations: List[TrackedObjects],
        neighbor_track_token: List[Optional[str]],  # 길이 = agent_num
) -> Dict[str, List[str]]:
    """
    get_future_tracked_objects를 이용해 한 번에 궤적을 수집하고,
    get_roadblock_ids_from_trajectory로 연결성 기반 ID 시퀀스를 추출합니다.
    """

    # iteration=0 시점부터 시나리오 끝까지 future 트랙 객체를 한줄로 가져옴
    # 전체 horizon은 시나리오 총 길이(초)로 지정
    # horizon = max(30.0,  _scenario_total_horizon_s(scenario))
    horizon = 20.
    num_samples = int(horizon / 0.1)
    # 1) 에이전트별 StateSE2 리스트 수집
    neighbor_track_token_set = set(
        [str(t) for t in neighbor_track_token if t is not None])
    if not neighbor_track_token_set:
        return {}
    trajectories: Dict[str, List[SimpleNamespace]] = defaultdict(list)
    first_pose: Dict[str, SimpleNamespace] = {}
    future_observations: List[TrackedObjects] = []
    for dets in scenario.get_future_tracked_objects(0, horizon, num_samples):
        future_observations.append(dets.tracked_objects)
    past_future_observations: List[TrackedObjects] = []
    past_future_observations.extend(sampled_past_observations)
    past_future_observations.extend(future_observations)
    for tracked_objects in past_future_observations:
        for obj in tracked_objects:
            # obj: TrackedObjects
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            token = str(obj.track_token)
            if token not in neighbor_track_token_set:
                continue
            # heading은 실제로 사용하지 않지만, 넣어도 무방(여기서는 0.0 또는 obj.center.heading 가능)
            rear_axle_state = StateSE2(obj.center.x, obj.center.y,
                                       obj.center.heading)
            pseudo_ego = SimpleNamespace(rear_axle=rear_axle_state)
            if first_pose.get(token, None) is None:
                first_pose[token] = pseudo_ego

            trajectories[token].append(pseudo_ego)

    # 2) 연결성 기반 roadblock ID 추출
    result: Dict[str, List[str]] = {}
    for token, pseudo_ego_states in trajectories.items():
        if not pseudo_ego_states:
            result[token] = []
            continue
        # 덕 타이핑: pseudo_ego_states[*].rear_axle.point 만 참조됨
        rb_ids: List[str] = get_roadblock_ids_from_trajectory(
            scenario.map_api, pseudo_ego_states)
        if len(rb_ids) == 0:
            result[token] = []
            continue
        corrected_ids = route_roadblock_correction(
            first_pose[token],
            scenario.map_api,
            rb_ids,
        )

        result[token] = corrected_ids
    return result


def _prefer_rr_on_conflict(
    rb_ids: Sequence[str],
    rbc_ids: Sequence[str],
    route_rr_ids: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> List[str]:
    """RB vs RBC 동시 검출 시 우선순위 규칙으로 선택한다.

    규칙:
      1) route_rr_ids 와의 교집합이 존재하면 그쪽을 우선
         - RBC ∩ route  → 우선 반환
         - RB  ∩ route  → 차선책
      2) 그렇지 않다면 RBC 우선 (교차로/연결부 가정)
      3) 그래도 비어 있으면 RB
      4) 그래도 없으면 []

    Args:
        rb_ids: RoadBlock id 리스트
        rbc_ids: RoadBlock-Connector id 리스트
        route_rr_ids: 시나리오의 글로벌 경로 roadblock ids
        verbose: 경고 메시지 출력 여부

    Returns:
        우선순위 규칙으로 정한 id 리스트(비어 있을 수 있음)
    """
    rb_ids = list(rb_ids) if rb_ids else []
    rbc_ids = list(rbc_ids) if rbc_ids else []

    if route_rr_ids:
        route_set = set(route_rr_ids)
        rbc_on_route = [rid for rid in rbc_ids if rid in route_set]
        if rbc_on_route:
            if verbose:
                warnings.warn(f"[RR-Resolve] RBC∩Route 선택: {rbc_on_route}")
            return rbc_on_route

        rb_on_route = [rid for rid in rb_ids if rid in route_set]
        if rb_on_route:
            if verbose:
                warnings.warn(f"[RR-Resolve] RB∩Route 선택: {rb_on_route}")
            return rb_on_route

    if rbc_ids:
        if verbose:
            warnings.warn(f"[RR-Resolve] Route 교집합 없음 → RBC 우선: {rbc_ids}")
        return rbc_ids

    if rb_ids:
        if verbose:
            warnings.warn(f"[RR-Resolve] RBC 없음 → RB 사용: {rb_ids}")
        return rb_ids

    if verbose:
        warnings.warn(f"[RR-Resolve] 비어 있음 → []")
    return []


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
    npc_route_indices: List[
        List[int]],  #  # 길이: agent_num, 원소: List[int] (길이 가변적)
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
    car_token_to_lane_on_routes: Dict[
        str, List[bool]],  # 길이 <= agent_num (키 개수). 각 값: 길이 lane_num
    neighbor_track_token: List[Optional[str]],  # 길이 = agent_num
    neighbor_agents_current: np.
    ndarray,  # shape: (agent_num, 11) → x=[:,0], y=[:,1]
    vector_map_lanes: np.ndarray,  # shape: (lane_num, P, D) → 좌표는 [:, :, :2]
    route_num: int,
) -> np.ndarray:

    from typing import List

    # ────────────── 보조 함수: 입력 검증 ──────────────
    def _validate_inputs() -> Tuple[int, int]:
        if neighbor_agents_current.ndim != 2 or neighbor_agents_current.shape[
                1] < 2:
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
                f"`vector_map_lanes` shape가 올바르지 않습니다: {vector_map_lanes.shape}"
            )
        if not (0 < route_num <= lane_num):
            raise ValueError(
                f"`route_num`는 0 < route_num < lane_num 을 만족해야 합니다. "
                f"(route_num={route_num}, lane_num={lane_num})")
        return agent_num, lane_num

    # ────────────── 보조 함수: 각 에이전트 ↔ lane 최소거리 정렬 ──────────────
    def _lane_min_dist_order(lanes_xy: np.ndarray,
                             agent_xy: np.ndarray) -> np.ndarray:
        """lane을 에이전트와의 최소거리 기준으로 정렬한 인덱스 반환.

        Args:
            lanes_xy (np.ndarray): shape=(lane_num, P, 2) lane 폴리라인 좌표.
            agent_xy (np.ndarray): shape=(2,) 에이전트 현재 위치.

        Returns:
            np.ndarray: shape=(lane_num,) 최소거리 오름차순 lane 인덱스 배열.
        """
        diff = lanes_xy - agent_xy[None, None, :]  # (lane_num, P, 2)
        dists = np.linalg.norm(diff, axis=-1)  # (lane_num, P)
        min_dists = np.min(dists, axis=1)  # (lane_num,)
        return np.argsort(min_dists)  # (lane_num,)

    # ────────────── 보조 함수: 가까운 순으로 True lane만 최대 K 선택 ──────────────
    def _select_lanes_by_order(
        lane_dist_order: np.ndarray,
        lane_on_routes_mask: np.ndarray,
        max_pick: int,
    ) -> List[int]:
        """정렬 순서대로 True인 lane만 최대 `max_pick`개 선택한다.

        Args:
            lane_dist_order (np.ndarray): shape=(lane_num,) 정렬된 lane 인덱스.
            lane_on_routes_mask (np.ndarray): shape<=(lane_num,) bool 마스크.
            max_pick (int): 최대 선택 개수.

        Returns:
            List[int]: 선택된 lane 인덱스 리스트(길이 ≤ max_pick).
        """
        selected: List[int] = []
        # print(f"len(lane_dist_order): {len(lane_dist_order)}, len(lane_on_routes_mask): {len(lane_on_routes_mask)}")
        for lane_idx in lane_dist_order:
            if lane_idx >= len(lane_on_routes_mask):
                continue
            if lane_on_routes_mask[lane_idx]:
                selected.append(int(lane_idx))
                if len(selected) >= max_pick:
                    break
        return selected

    """토큰별 '경로 위 차선'을 가까운 순으로 최대 `route_num`개 뽑아 랭크 행렬을 만든다.

    절차(에이전트 i에 대해):
      1) lane 폴리라인과 에이전트 현재 위치 간 최소거리로 **lane**을 가까운→먼 순으로 정렬
      2) 그 순서대로 `car_token_to_lane_on_routes[token][lane_idx]`가 True인 lane만
         최대 `route_num`개 선택하여 `npc_route_indices[i] = [lane_idx0, lane_idx1, ...]`
      3) 모든 에이전트 i에 대해 위 작업 후 `build_agent_route_lane_order`로
         (agent_num, lane_num) 랭크 행렬을 생성(해당 에이전트의 route가 아니면 -1)

    Args:
        car_token_to_lane_on_routes (Dict[str, List[bool]]):
            길이 = agent_num 보다 작거나 같음 . 키=토큰, 값=길이 lane_num의 불리언 마스크.
        neighbor_track_token (List[Optional[str]]):
            길이 = agent_num. 에이전트 토큰(없으면 None).
        neighbor_agents_current (np.ndarray):
            shape = (agent_num, 11). 현재 프레임의 이웃 에이전트 상태(ego 기준).
            - x = [:, 0], y = [:, 1] 만 사용.
        vector_map_lanes (np.ndarray):
            shape = (lane_num, P, D). lane 폴리라인. 좌표는 [:, :, :2] 사용.
        route_num (int):
            각 에이전트에 대해 선택할 최대 차선 개수(0 < route_num < lane_num).

    Returns:
        np.ndarray:
            `agent_route_lane_order`, shape = (agent_num, lane_num), dtype = np.int32.
            - [i, j] = 에이전트 i에서 lane j의 "가까운 True 차선 순위(0,1,2,...)".
            - 해당 에이전트의 route가 아니면 -1.

    Raises:
        ValueError: 입력 길이/shape가 올바르지 않거나 `route_num` 제약 위반 시.

    Notes:
        - `neighbor_track_token`과 `neighbor_agents_current`는 **ego와 가까운 순**이라 가정.
        - 거리 계산은 lane 폴리라인의 모든 점과의 최소거리(유클리드) 기준.
    """
    # ────────────── 메인 ──────────────
    agent_num, lane_num = _validate_inputs()
    lanes_xy = vector_map_lanes[:, :, :2]  # (lane_num, P, 2)

    npc_route_indices: List[List[int]] = [
    ]  # 길이: agent_num, 원소: List[int] (길이 가변적)

    for agent_idx in range(agent_num):
        selected: List[int] = []

        token = neighbor_track_token[agent_idx]  # Optional[str]
        if token is None:
            lane_on_routes = [False] * lane_num
        else:
            # car_token_to_lane_on_routes (Dict[str, List[bool]])
            lane_on_routes = car_token_to_lane_on_routes.get(
                token, [False] * lane_num)

        # (1) 경로가 전혀 없으면 건너뜀
        lane_on_routes_arr = np.asarray(lane_on_routes,
                                        dtype=bool)  # (lane_num,)
        if lane_on_routes_arr.sum() == 0:
            npc_route_indices.append(selected)
            continue
        valid_len = len(lane_on_routes)
        if valid_len == 0:
            npc_route_indices.append([])
            continue
        # (2) 가까운 lane 정렬
        agent_xy = neighbor_agents_current[agent_idx, :2]  # (2,)
        lane_dist_order = _lane_min_dist_order(
            lanes_xy[:valid_len],
            agent_xy)  # shape=(lane_num,) 최소거리 오름차순 lane 인덱스 배열.

        # (3) 정렬 순으로 True lane만 최대 route_num개 선택
        selected = _select_lanes_by_order(lane_dist_order, lane_on_routes_arr,
                                          route_num)

        npc_route_indices.append(selected)

    # 반환 길이 검증
    assert len(npc_route_indices) == agent_num, \
        f"npc_route_indices 길이({len(npc_route_indices)}) != agent_num({agent_num})"

    agent_route_lane_order = build_agent_route_lane_order(npc_route_indices,
                                                          lane_num=lane_num)
    return agent_route_lane_order


def get_neighbor_track_tokens(
    present_tracked_objects: TrackedObjects,
    neighbor_indices: Union[Sequence[int], np.ndarray],
    agents_num: int,
    object_types: Optional[Sequence[TrackedObjectType]] = None,
) -> List[Optional[str]]:
    """현재 프레임에서 선별된 이웃의 `track_token`을, 에이전트 슬롯 순서대로 반환합니다.

    개요
    - `agent_past_process(...)`가 반환한 `neighbor_indices`는
      "현재 프레임(=리스트의 마지막 프레임)의 에이전트 배열"의 **행 인덱스**입니다.
    - 같은 현재 프레임의 `TrackedObjects`에서 `VEHICLE/PEDESTRIAN/BICYCLE`만
      `_extract_agent_array`와 동일한 순서로 나열해두면,
      `neighbor_indices[k]` → 해당 행의 `TrackedObject.track_token`으로 1:1 매핑할 수 있습니다.
    - 반환 리스트 길이는 항상 `agents_num`이며, 유효한 이웃보다 슬롯이 많으면 나머지는 `None`으로 채웁니다.

    Args:
        present_tracked_objects:
            - 현재 프레임의 관측. `TrackedObjects` 또는 `DetectionsTracks`(또는 `tracked_objects` 속성 보유형).
        neighbor_indices:
            - 모양: **(K, )**, `int` 인덱스. `agent_past_process`의 `sorted_cur_neighbor_indices`.
            - 이 순서가 곧 `neighbor_agents_past`의 행 순서(거리 오름차순 등)입니다.
        agents_num:
            - 최종 슬롯 개수. 반환 리스트 길이가 됩니다.
        object_types:
            - 필터링할 타입. 기본은 `[VEHICLE, PEDESTRIAN, BICYCLE]`.
              `_extract_agent_array`와 동일해야 인덱스 정합이 보장됩니다.

    Returns:
        List[Optional[str]]:
            - 길이: `agents_num`
            - 각 원소는 해당 슬롯의 `track_token`(문자열). 비어 있으면 `None`.

    Notes:
        - 인덱스 범위를 벗어나거나 타입 불일치로 매핑이 안 되면 `None`을 넣습니다.
        - `neighbor_indices` 길이가 `agents_num`보다 길 경우, 앞 `agents_num`개만 사용합니다.
    """
    if agents_num < 0:
        raise ValueError(f"`agents_num`은 음수가 될 수 없습니다. got {agents_num}")

    tracked_objects = present_tracked_objects

    # `_extract_agent_array`와 동일한 타입 필터 순서 유지
    if object_types is None:
        object_types = (
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        )

    # 현재 프레임에서 관심 타입만 '그 순서 그대로' 나열
    current_agents: List[
        TrackedObject] = tracked_objects.get_tracked_objects_of_types(
            object_types)  # type: ignore[assignment]
    tokens_in_present_order: List[str] = [
        str(agent.track_token) for agent in current_agents
    ]  # (M,)

    # 반환 버퍼 준비
    neighbor_track_token: List[Optional[str]] = [None] * int(agents_num)

    # neighbor_indices 정규화(int list)
    if neighbor_indices is None:
        return neighbor_track_token
    # numpy, list, tuple 등 모두 int 리스트로 캐스팅
    idx_list: List[int] = list(
        map(int,
            np.asarray(neighbor_indices).reshape(-1).tolist()))

    # 앞에서부터 agents_num개만 매핑
    max_fill = min(len(idx_list), agents_num)
    for slot_idx in range(max_fill):
        src_idx = idx_list[slot_idx]
        if 0 <= src_idx < len(tokens_in_present_order):
            neighbor_track_token[slot_idx] = tokens_in_present_order[src_idx]
        else:
            # 범위를 벗어나면 안전하게 None 유지
            neighbor_track_token[slot_idx] = None

    return neighbor_track_token


# 시나리오 전체 horizon(초) 계산: 시작~끝 타임스탬프 차이
def _scenario_total_horizon_s(scn: AbstractScenario) -> float:
    """
    시나리오 시작 시각(초)부터 **로그 파일의 끝 시각(초)** 까지의 horizon을 계산한다.
    - 시나리오 토큰이 1개뿐이라 duration이 0이어도, DB의 end time을 사용해 올바르게 계산한다.

    Args:
        scn: nuPlan Scenario 객체

    Returns:
        float: horizon [s]
    """
    # 1) 시작 시각(초): 공개 API 사용
    start_s = float(scn.get_time_point(0).time_s)

    # 3) fallback: DB의 실제 끝 시각(마이크로초)으로 계산
    # 내부 모듈: nuPlan devkit 표준

    # NuPlanScenario는 _log_file을 보유 (public은 아니지만 일반적으로 접근 가능)
    log_file_path: str = getattr(scn, "_log_file")
    end_us: int = get_end_sensor_time_from_db(log_file_path,
                                              get_lidarpc_sensor_data())
    end_s = float(end_us) * 1e-6
    return max(0.0, end_s - start_s)


def get_npc_route_roadblock_ids2(
    scenario: NuPlanScenario,
    neighbor_track_token: List[Optional[str]],
) -> Dict[str, Optional[List[str]]]:

    from collections import defaultdict
    from typing import Dict, List, Optional, Set

    def select_nearest_connectors_by_mean_distance(
        connector_candidates: List[RoadBlockGraphEdgeMapObject],
        sampled_trajectory_points: List["Point2D"],
        *,
        tolerance: float = 1e-6,
    ) -> List[RoadBlockGraphEdgeMapObject]:
        """평균 수선거리로 가장 가까운 Connector 후보(들)를 선택한다.

        Args:
            connector_candidates (List[RoadBlockGraphEdgeMapObject]):
                후보 Connector 객체 리스트. 길이 K(가변).
            sampled_trajectory_points (List[Point2D]):
                Connector 영역에서 샘플링한 궤적 점들. 길이 T(가변).
            tolerance (float):
                부동소수 오차 허용치. 최솟값과의 차이가 `≤ tolerance`면 동률로 간주.

        Returns:
            List[RoadBlockGraphEdgeMapObject]:
                평균 수선거리가 최솟값인 Connector 객체(들).
        """
        mean_distance_by_connector: Dict[RoadBlockGraphEdgeMapObject,
                                         float] = {}
        for connector in connector_candidates:
            mean_dist: float = _mean_perpendicular_distance(
                connector, sampled_trajectory_points)
            mean_distance_by_connector[connector] = mean_dist

        minimum_distance: float = min(mean_distance_by_connector.values())
        return [
            conn for conn, dist in mean_distance_by_connector.items()
            if abs(dist - minimum_distance) <= tolerance
        ]

    def _mean_perpendicular_distance(
        roadblock_connector: RoadBlockGraphEdgeMapObject,
        trajectory_points: List["Point2D"],
    ) -> float:
        """궤적 점들과 Connector 폴리곤 간 평균 수선거리를 계산한다.

        Args:
            roadblock_connector (RoadBlockGraphEdgeMapObject):
                거리 계산 대상 Connector.
            trajectory_points (List[Point2D]):
                궤적 포인트 리스트. 길이 T.

        Returns:
            float: 평균 수선거리 값.
        """
        polygon = roadblock_connector.polygon
        return float(
            np.mean([
                Point(pt.x, pt.y).distance(polygon) for pt in trajectory_points
            ]))

    def _decide_roadblock_ids_at_connector(
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'],
        sampled_points_inside_connector: List['Point2D'],
        roadblock_sequence: List[str],
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'],
        current_roadblocks: Set['RoadBlockGraphEdgeMapObject'],
    ) -> None:
        """Connector 구간 종료 시, 후보 중 연결성/거리 기준으로 선택하여 시퀀스에 확정한다.

        우선 연결성(이전/다음 RoadBlock 연결) 조건을 만족하는 후보를 우선 선택하고,
        그렇지 않으면 연결성 중 어느 하나라도 만족하는 후보들 중에서 평균 수선거리
        최소 후보(들)를 선택한다. 해당 경우가 없으면 모든 후보 중 평균 수선거리
        최소 후보(들)를 선택한다.

        Args:
            connector_candidate_objects (Set[RoadBlockGraphEdgeMapObject]):
                구간 동안 누적된 Connector 후보 집합.
            sampled_points_inside_connector (List[Point2D]):
                해당 Connector 구간에서 수집한 궤적 점들. 길이 T.
            roadblock_sequence (List[str]):
                결과를 축적할 RoadBlock/Connector id 시퀀스(가변).
            previous_roadblocks_set (Set[RoadBlockGraphEdgeMapObject]):
                직전 프레임의 RoadBlock 집합.
            current_roadblocks (Set[RoadBlockGraphEdgeMapObject]):
                현재 프레임의 RoadBlock 집합(다음 구간 연결 확인용).
        """
        graph_linkable_connectors: List[RoadBlockGraphEdgeMapObject] = []
        graph_linkable_connectors_candidates: List[
            RoadBlockGraphEdgeMapObject] = []

        incoming_and_outcoming_condition = len(
            previous_roadblocks_set) > 0 and len(current_roadblocks) > 0
        incoming_or_outgoing_condition = len(
            previous_roadblocks_set) > 0 or len(current_roadblocks) > 0

        # (1) 이전/다음 RoadBlock 모두와 연결되는 Connector 우선
        if incoming_and_outcoming_condition:
            for conn in connector_candidate_objects:
                incoming_ids = {rb.id for rb in conn.incoming_edges}
                previous_ids = {rb.id for rb in previous_roadblocks_set}
                outgoing_ids = {rb.id for rb in conn.outgoing_edges}
                current_ids = {rb.id for rb in current_roadblocks}

                if bool(previous_ids & incoming_ids) and bool(current_ids &
                                                              outgoing_ids):
                    graph_linkable_connectors.append(conn)

            if graph_linkable_connectors:
                roadblock_sequence.extend(
                    [conn.id for conn in graph_linkable_connectors])
                return

        # (2) 이전 또는 다음 RoadBlock 중 하나와 연결되면 후보로 인정 후 거리 최소
        if (incoming_and_outcoming_condition or incoming_or_outgoing_condition):
            for conn in connector_candidate_objects:
                incoming_ids = {rb.id for rb in conn.incoming_edges}
                previous_ids = {rb.id for rb in previous_roadblocks_set}
                outgoing_ids = {rb.id for rb in conn.outgoing_edges}
                current_ids = {rb.id for rb in current_roadblocks}

                if bool(previous_ids & incoming_ids) or bool(current_ids &
                                                             outgoing_ids):
                    graph_linkable_connectors_candidates.append(conn)

            if graph_linkable_connectors_candidates:
                closest_connectors = select_nearest_connectors_by_mean_distance(
                    graph_linkable_connectors_candidates,
                    sampled_points_inside_connector,
                    tolerance=1e-6,
                )
                roadblock_sequence.extend(
                    conn.id for conn in closest_connectors)
                return

        # (3) 연결성 조건 없으면 전체 후보 중 거리 최소
        closest_connectors = select_nearest_connectors_by_mean_distance(
            connector_candidate_objects,
            sampled_points_inside_connector,
            tolerance=1e-6,
        )
        roadblock_sequence.extend(conn.id for conn in closest_connectors)

    # ───────────────────────── 보조 함수들(가독성) ─────────────────────────

    def _collect_candidate_tokens(
            neighbor_track_token: List[Optional[str]]) -> Set[str]:
        """None을 제외한 후보 토큰 집합을 만든다.

        Args:
            neighbor_track_token (List[Optional[str]]):
                길이 = agent_num. 토큰 또는 None.

        Returns:
            Set[str]: 후보 토큰 집합.
        """
        return {t for t in neighbor_track_token if t is not None}

    def _collect_future_vehicle_trajectories(
        scenario: NuPlanScenario,
        candidate_tokens: Set[str],
        total_horizon_s: float = 20.,
    ) -> Dict[str, List[TrackedObject]]:
        """미래 기간 동안의 차량 궤적을 토큰별로 수집한다.

        Args:
            scenario (NuPlanScenario): 시나리오.
            candidate_tokens (Set[str]): 후보 토큰 집합.
            total_horizon_s (float): 수집할 미래 수평선(초).

        Returns:
            Dict[str, List[SceneObject]]:
                키=토큰, 값=해당 차량의 시간 순 궤적 리스트(길이 가변).
        """
        car_token_to_object_list: Dict[str,
                                       List[TrackedObject]] = defaultdict(list)
        num_samples = int(total_horizon_s * 10)  # 0.1 s 간격
        for det_batch in scenario.get_future_tracked_objects(
                0, total_horizon_s, num_samples):
            for det in det_batch.tracked_objects:
                if det.tracked_object_type == TrackedObjectType.VEHICLE and (
                        det.track_token in candidate_tokens):
                    car_token_to_object_list[det.track_token].append(det)
        return car_token_to_object_list

    def _finalize_connector_segment_if_open(
        inside_connector_flag: bool,
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'],
        sampled_points_inside_connector: List['Point2D'],
        roadblock_sequence: List[str],
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'],
        current_roadblocks: Set['RoadBlockGraphEdgeMapObject'],
    ) -> bool:
        """열린 Connector 구간이 있으면 후보 결정 후 버퍼를 리셋한다.

        Returns:
            bool: 정리 후 inside_connector_flag(False).
        """
        if inside_connector_flag:
            _decide_roadblock_ids_at_connector(
                connector_candidate_objects,
                sampled_points_inside_connector,
                roadblock_sequence,
                previous_roadblocks_set,
                current_roadblocks,
            )
            inside_connector_flag = False
            connector_candidate_objects.clear()
            sampled_points_inside_connector.clear()
        return inside_connector_flag

    """NPC 차량들의 경로(RoadBlock/RoadBlock-Connector 시퀀스)를 토큰별로 구성한다.

    이 함수는 `neighbor_track_token`에 제시된 에이전트 토큰들(차량만)에 대해,
    시나리오의 **미래 궤적(약 20 s)**을 주행 순서로 훑어보며 RoadBlock·RoadBlock‑Connector
    교차 여부를 추출한다. Connector 구간에서는 평균 수선거리 기반으로 **가장 가까운**
    Connector(동률 허용)를 선택하며, 구간 전환 시점에만 최종 확정한다.
    구축된 시퀀스는 `route_roadblock_correction`로 보정한 뒤 토큰별로 반환한다.

    Args:
        scenario (NuPlanScenario):
            NuPlan 시나리오 객체.
        neighbor_track_token (List[Optional[str]]):
            길이 = `agent_num`. 각 슬롯에 NPC의 `track_token`(없으면 `None`).

    Returns:
        car_token_to_rr_ids
        Dict[str, Optional[List[str]]]:
            키 = 토큰(str).
            값 = 해당 NPC의 **보정된** RoadBlock id 시퀀스(List[str]) 또는 `None`
            (미추출 시).
            길이 : agent_num 중, 자동차 토큰 개수.

    Raises:
        ValueError: 동일 프레임에서 RoadBlock과 RoadBlock‑Connector가 동시에
            관측될 경우(데이터 불일치).

    Notes:
        - 궤적 수집은 `TrackedObjectType.VEHICLE` 에 한정.
        - Connector 구간이 여러 후보를 만들면 평균 수선거리가 **최소**인 후보들을 모두 선택.
        - 내부 보조 함수들로 단계별 처리를 분리(가독성 향상).
    """

    # ───────────────────────── 메인 로직(동작 동일) ─────────────────────────

    candidate_tokens = _collect_candidate_tokens(neighbor_track_token)

    car_token_to_object_list: Dict[
        str, List[TrackedObject]] = _collect_future_vehicle_trajectories(
            scenario,
            candidate_tokens,  # 기존 필터는 유지 (neighbor 토큰 기반)
            total_horizon_s=20.,
        )
    car_token_to_rr_ids: Dict[str, Optional[List[str]]] = {}

    for car_token, car_list in car_token_to_object_list.items():
        if not car_list:
            car_token_to_rr_ids[car_token] = None
            continue

        roadblock_sequence: List[str] = []
        previous_roadblocks_set: Set['RoadBlockGraphEdgeMapObject'] = set()
        inside_connector_flag = False
        connector_candidate_objects: Set['RoadBlockGraphEdgeMapObject'] = set()
        sampled_points_inside_connector: List['Point2D'] = []

        for time_idx, car_ in enumerate(car_list):
            npc_point = car_.center.point
            current_roadblocks = set(
                scenario.map_api.get_all_map_objects(
                    npc_point, SemanticMapLayer.ROADBLOCK))
            current_connectors = set(
                scenario.map_api.get_all_map_objects(
                    npc_point, SemanticMapLayer.ROADBLOCK_CONNECTOR))

            # 동시 검출 예외
            # 둘 다 잡힌 경우: tie-breaker 로 하나만 사용하도록 current_*를 덮어쓴다.
            if current_roadblocks and current_connectors:
                # 시나리오의 글로벌 경로 (빈 리스트 형태 [''] 는 None 으로 처리)
                try:
                    route_rr_ids = scenario.get_route_roadblock_ids()
                    if isinstance(route_rr_ids, list) and route_rr_ids == ['']:
                        route_rr_ids = None
                except Exception:
                    route_rr_ids = None

                rb_ids_list = [rb.id for rb in current_roadblocks]
                rbc_ids_list = [rc.id for rc in current_connectors]

                chosen_ids = _prefer_rr_on_conflict(
                    rb_ids=rb_ids_list,
                    rbc_ids=rbc_ids_list,
                    route_rr_ids=route_rr_ids,
                    verbose=False,
                )

                if chosen_ids:
                    # 선택 결과와 교집합 되는 쪽을 남기고, 반대편은 비운다.
                    chosen_rbc = {
                        rc for rc in current_connectors if rc.id in chosen_ids
                    }
                    chosen_rb = {
                        rb for rb in current_roadblocks if rb.id in chosen_ids
                    }

                    if chosen_rbc and not chosen_rb:
                        current_connectors = chosen_rbc
                        current_roadblocks = set()
                    elif chosen_rb and not chosen_rbc:
                        current_roadblocks = chosen_rb
                        current_connectors = set()
                    else:
                        # 혹시 양쪽과도 교집합이 없거나 둘 다 생기는 예외 상황이면 RBC 우선
                        if chosen_rbc:
                            current_connectors = chosen_rbc
                            current_roadblocks = set()
                        elif chosen_rb:
                            current_roadblocks = chosen_rb
                            current_connectors = set()
                        else:
                            current_roadblocks = set()  # RBC 우선 fallback
                else:
                    # tie-breaker가 비었으면 RBC 우선
                    current_roadblocks = set()

            # ── (A) Connector 영역 ──
            if current_connectors:
                if not inside_connector_flag:
                    connector_candidate_objects.clear()
                    sampled_points_inside_connector.clear()
                    inside_connector_flag = True
                connector_candidate_objects.update(current_connectors)
                sampled_points_inside_connector.append(npc_point)

                # 마지막 프레임이면 곧바로 결정
                if time_idx == len(car_list) - 1:
                    inside_connector_flag = _finalize_connector_segment_if_open(
                        inside_connector_flag,
                        connector_candidate_objects,
                        sampled_points_inside_connector,
                        roadblock_sequence,
                        previous_roadblocks_set,
                        current_roadblocks,
                    )
                continue

            # ── (B) RoadBlock 영역 ──
            if current_roadblocks:
                # 직전이 Connector 구간이면 우선 결정
                inside_connector_flag = _finalize_connector_segment_if_open(
                    inside_connector_flag,
                    connector_candidate_objects,
                    sampled_points_inside_connector,
                    roadblock_sequence,
                    previous_roadblocks_set,
                    current_roadblocks,
                )
                # RoadBlock id 중복 없이 추가
                for roadblock in current_roadblocks:
                    if not roadblock_sequence or roadblock_sequence[
                            -1] != roadblock.id:
                        roadblock_sequence.append(roadblock.id)
                previous_roadblocks_set = current_roadblocks

        # 결과 보정
        if roadblock_sequence:
            start = car_list[0]
            npc_state = SimpleNamespace(rear_axle=StateSE2(
                start.center.x, start.center.y, start.center.heading))
            corrected_ids = route_roadblock_correction(
                npc_state,
                scenario.map_api,
                roadblock_sequence,
                remove_route_loops_flag=False,
            )
            car_token_to_rr_ids[car_token] = roadblock_sequence
        else:
            car_token_to_rr_ids[car_token] = None

    return car_token_to_rr_ids


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
        # agent_local_vel = agent_state[:, [
        #     EgoInternalIndex.vx(), EgoInternalIndex.vy()
        # ]]
        # agent_local_vel = np.expand_dims(np.concatenate(
        #     (agent_local_vel, np.zeros(
        #         (agent_local_vel.shape[0], 1))), axis=-1),
        #                                  axis=-1)
        # transformed_vel = np.matmul(transforms,
        #                             agent_local_vel).squeeze(axis=-1)
        # --- velocity (world -> anchor ego frame) ---
        agent_global_velocities = agent_state[:, [
            EgoInternalIndex.vx(), EgoInternalIndex.vy()
        ]]
        transformed_velocities = _global_velocity_to_local(
            agent_global_velocities, ego_pose[-1])

        new_agent_state[:, 4] = transformed_velocities[:, 0]
        new_agent_state[:, 5] = transformed_velocities[:, 1]
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
from typing import Any, Dict, Mapping, Union
import numpy as np
import torch


def convert_to_model_inputs(
        data: Mapping[str, Any],
        device: Union[torch.device, str],
        squeeze: bool
) -> Dict[str, torch.Tensor]:
    """
    딕셔너리 값을 torch.Tensor로 변환합니다.

    규칙
    - bool 계열은 torch.bool 유지
    - 그 외 수치형은 torch.float32로 캐스팅
    - squeeze=False 이면 배치 차원(앞쪽) 1을 추가

    Args:
        data: {키: 값} 형태. 값은 torch.Tensor / np.ndarray / 파이썬 스칼라/리스트 모두 허용
        device: 배치 후 올릴 디바이스 (예: "cuda:0", torch.device("cpu"))
        squeeze: False면 앞쪽에 차원 하나 unsqueeze(0)

    Returns:
        키별 torch.Tensor 딕셔너리
    """
    out: Dict[str, torch.Tensor] = {}

    for k, v in data.items():
        # 1) 이미 Tensor인 경우: 재생성 하지 말고 .to(...) 만
        if isinstance(v, torch.Tensor):
            target_dtype = torch.bool if v.dtype == torch.bool else torch.float32
            t = v.to(device=device, dtype=target_dtype, non_blocking=True)

        # 2) Numpy 배열인 경우: 복사 최소화를 위해 from_numpy/as_tensor 사용
        elif isinstance(v, np.ndarray):
            if v.dtype == np.bool_:
                # bool은 dtype 보존 -> 이후 device로만 이동
                t = torch.from_numpy(v).to(device=device, non_blocking=True)
                if t.dtype != torch.bool:
                    t = t.to(dtype=torch.bool)
            else:
                # 수치형은 float32로
                t = torch.from_numpy(v).to(device=device, dtype=torch.float32,
                                           non_blocking=True)

        # 3) 파이썬 bool 스칼라
        elif isinstance(v, (bool, np.bool_)):
            t = torch.tensor(v, dtype=torch.bool, device=device)

        # 4) 나머지(리스트/스칼라 등): as_tensor로 한 번에
        else:
            t = torch.as_tensor(v, dtype=torch.float32, device=device)

        if not squeeze:
            t = t.unsqueeze(0)

        out[k] = t

    return out
