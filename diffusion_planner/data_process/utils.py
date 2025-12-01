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

from typing import Dict
import numpy as np
import numpy.typing as npt


def ego_local_traj3_to_global(
    local_traj_xyh: npt.NDArray[
        np.floating],  # shape: (T, 3) = [x_e, y_e, yaw_e]
    cur_ego_global_xyyaw: npt.NDArray[
        np.floating],  # shape: (3,) = [x_g, y_g, yaw_g]
    *,
    invalid_eps: float = 0.0,
) -> npt.NDArray[np.float64]:
    """ego 좌표계 (x, y, heading) 시퀀스를 세계 절대 좌표계로 변환하되,
    (0., 0., 0.)인 무효 행은 제거하고 유효 행만 반환합니다.

    Args:
        local_traj_xyh (np.ndarray): shape (T, 3). 각 행은 [x_e, y_e, yaw_e].
        cur_ego_global_xyyaw (np.ndarray): shape (3,). [x_g, y_g, yaw_g].
        invalid_eps (float, optional): 무효 판정 허용 오차.
            - 0.0: 정확히 (0., 0., 0.)만 무효
            - >0.0: |x_e|, |y_e|, |yaw_e| 모두 eps 이하이면 무효

    Returns:
        np.ndarray: shape (T_valid, 3). 각 행은 [x_g, y_g, yaw_g].
                    유효 행이 하나도 없으면 (0, 3) 배열을 반환.
    """
    if local_traj_xyh.ndim != 2 or local_traj_xyh.shape[1] != 3:
        raise ValueError(
            f"`local_traj_xyh` shape must be (T,3), got {local_traj_xyh.shape}")
    if cur_ego_global_xyyaw.shape != (3,):
        raise ValueError(
            f"`cur_ego_global_xyyaw` shape must be (3,), got {cur_ego_global_xyyaw.shape}"
        )

    # ── 1) 무효 행 필터링: (x, y, yaw) 모두 0(또는 eps 이내)이면 제거 ─────────────────
    if invalid_eps <= 0.0:
        invalid_mask = (local_traj_xyh[:, 0] == 0.0) & (
            local_traj_xyh[:, 1] == 0.0) & (local_traj_xyh[:, 2] == 0.0)
    else:
        invalid_mask = (
            np.isclose(local_traj_xyh[:, 0], 0.0, atol=invalid_eps) &
            np.isclose(local_traj_xyh[:, 1], 0.0, atol=invalid_eps) &
            np.isclose(local_traj_xyh[:, 2], 0.0, atol=invalid_eps))
    valid_mask = ~invalid_mask
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64)

    local_valid = local_traj_xyh[valid_mask]  # (T_valid, 3)

    # ── 2) ego→global 변환 ────────────────────────────────────────────────────────
    x_e = local_valid[:, 0]
    y_e = local_valid[:, 1]
    yaw_e = local_valid[:, 2]

    x_g0, y_g0, yaw_g0 = map(float,
                             cur_ego_global_xyyaw.tolist())  # 글로벌 기준(ego 현재 포즈)
    c, s = np.cos(yaw_g0), np.sin(yaw_g0)
    x_g = x_e * c - y_e * s + x_g0
    y_g = x_e * s + y_e * c + y_g0
    yaw_g = yaw_e + yaw_g0
    # 필요하면 yaw_g = (yaw_g + np.pi) % (2 * np.pi) - np.pi  # [-pi, pi] 정규화

    return np.stack([x_g, y_g, yaw_g], axis=1).astype(np.float64)


def get_npc_route_roadblock_ids(
    scenario: NuPlanScenario,
    past_cur_tracked_objects: List[TrackedObjects],
    neighbor_track_token: Optional[List[str]],  # 길이 = chosen_agent_num
    horizon=20.,
) -> Dict[str, List[str]]:
    """
    get_future_tracked_objects 를 이용해 한 번에 궤적을 수집하고,
    get_roadblock_ids_from_trajectory 로 연결성 기반 ID 시퀀스를 추출합니다.
    """

    # iteration=0 시점부터 시나리오 끝까지 future 트랙 객체를 한줄로 가져옴
    # 전체 horizon은 시나리오 총 길이(초)로 지정
    # horizon = max(30.0,  _scenario_total_horizon_s(scenario))
    num_samples = int(horizon / 0.1)
    # 1) 에이전트별 StateSE2 리스트 수집
    if neighbor_track_token is None:
        allow_all_token = True
        neighbor_track_token_set = {}
    else:
        allow_all_token = False
        neighbor_track_token_set = set([str(t) for t in neighbor_track_token])
        if not neighbor_track_token_set:
            return {}

    future_observations: List[TrackedObjects] = []
    for dets in scenario.get_future_tracked_objects(0, horizon, num_samples):
        future_observations.append(dets.tracked_objects)
    past_future_observations: List[TrackedObjects] = []
    past_future_observations.extend(past_cur_tracked_objects)
    past_future_observations.extend(future_observations)

    token_to_state_list: Dict[str, List[SimpleNamespace]] = defaultdict(list)
    first_pose: Dict[str, SimpleNamespace] = {}
    for tracked_objects in past_future_observations:
        # tracked_objects: TrackedObjects
        for obj in tracked_objects:
            # obj: TrackedObject
            if obj.tracked_object_type != TrackedObjectType.VEHICLE:
                continue
            token = str(obj.track_token)
            if (not allow_all_token) and (token
                                          not in neighbor_track_token_set):
                continue
            # heading은 실제로 사용하지 않지만, 넣어도 무방(여기서는 0.0 또는 obj.center.heading 가능)
            rear_axle_state = StateSE2(obj.center.x, obj.center.y,
                                       obj.center.heading)
            pseudo_ego = SimpleNamespace(rear_axle=rear_axle_state)
            if first_pose.get(token, None) is None:
                first_pose[token] = pseudo_ego

            token_to_state_list[token].append(pseudo_ego)

    # 2) 연결성 기반 roadblock ID 추출
    car_token_to_rb_ids_list: Dict[str, List[str]] = {}
    for token, states_list in token_to_state_list.items():
        if not states_list:
            raise RuntimeError(
                f"Internal error: token_to_state_list[{token}] is empty.")
        # 덕 타이핑: states_list[*].rear_axle.point 만 참조됨
        rb_ids_list: List[str] = get_roadblock_ids_from_trajectory(
            scenario.map_api, states_list)
        if len(rb_ids_list) == 0:
            car_token_to_rb_ids_list[token] = []
            continue
        corrected_ids = route_roadblock_correction(
            first_pose[token],
            scenario.map_api,
            rb_ids_list,
        )

        car_token_to_rb_ids_list[token] = corrected_ids
    return car_token_to_rb_ids_list


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
    """맵 객체(MapObject)에서 도형(점/선/면) 정보를 꺼내는 작은 도우미 함수.

    이 함수는 나중에
    `get_directional_proximal_map_objects` 같은 곳에서
    “이 물체가 내가 만든 영역과 겹치는지”를 확인하기 위해,
    맵 객체를 Shapely에서 이해할 수 있는 도형으로 바꿔주는 역할을 한다.

    동작 규칙
    --------
    1) 먼저 `obj.polygon` 이 있는지 확인한다.
       - 예: 도로 묶음(roadblock), 교차로, 횡단보도 등은 보통 폴리곤(면)으로 제공된다.
       - 있으면 그대로 돌려준다.
         · 반환 도형 예시: Polygon

    2) 폴리곤이 없다면, 차선처럼 “중심선 polyline” 형태를 갖고 있는지 본다.
       - `obj.baseline_path.discrete_path` 가 있는 경우:
         · 이 안에는 (x, y, heading) 형태의 점들이 순서대로 들어있다고 보면 된다.
           - 길이: N
           - 좌표 배열로 바꾸면 개념적으로 (N, 2) 모양
         · 이 점들로 Shapely LineString 을 만들어 반환한다.
           - N >= 2 인 경우에만 선(LineString)으로 만들고,
           - N == 1 이면 점(Point)으로 반환한다.

    3) 위 두 가지 경우 모두 아니면, 공개된 속성만으로는 모양을 알 수 없으므로
       `None` 을 돌려준다.

    Args:
        obj (MapObject):
            - NuPlan 맵에서 가져온 아무 종류의 맵 객체.
            - 예: 차선, 차선연결, 도로묶음, 교차로 등.

    Returns:
        Optional[geom.base.BaseGeometry]:
            - Polygon / LineString / Point 같은 Shapely 도형 객체.
            - 도형 정보를 만들 수 없을 때는 `None`.
    """
    # 1) 다각형이 있는 타입(예: Lane, RoadBlock, Connector 등)
    polygon = getattr(obj, "polygon", None)
    if polygon is not None:
        return polygon

    # 2) 차선류 등: baseline_path → LineString
    baseline_path = getattr(obj, "baseline_path", None)
    if baseline_path is not None and hasattr(baseline_path, "discrete_path"):
        # pts: 길이 = N, 각 원소 = (x, y)  → 개념적 shape: (N, 2)
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
    """ego 진행 방향을 기준으로 회전된 정사각형 안에 걸치는 맵 객체를 조회한다.

    이 함수는 NuPlanMap 의 :meth:`get_proximal_map_objects` 와 비슷하지만,
    패치 모양이 다르다.

    - 기존: point 를 중심으로 한 **가로·세로 방향 정사각형**
        · [x - radius, x + radius] × [y - radius, y + radius]
    - 이 함수: point 를 중심으로 한 **ego heading 방향에 맞춰 회전된 정사각형**
        · 한 변 길이: 2 * radius
        · 한 변이 ego heading 과 평행, 다른 변은 그에 수직

    포함 기준
    ----------
    각 레이어의 모든 객체에 대해, 객체의 도형(geometry)이
    회전된 정사각형과 `intersects` 인지 검사한다.

    Shapely 의 `intersects` 는
    “도형이 서로 **한 점이라도 겹치면** True” 이므로, 아래 경우 모두 포함된다.

    * 정사각형 안에 완전히 들어온 경우
    * 정사각형 모서리에 살짝 걸치는 경우
    * 거의 밖에 있지만, 일부 꼭짓점이나 변이 정사각형에 닿는 경우

    내부 동작 흐름
    --------------
    1) 지원 레이어 확인
        - `map_api.get_available_map_objects()` 로 실제 지원 레이어 목록을 가져온다.
        - 요청한 `layers` 중 지원되지 않는 레이어가 있으면 assert 로 바로 실패시킨다.

    2) 회전 전 정사각형 패치 생성 (축에 정렬된 네모)
        - x 방향 범위: [point.x - radius, point.x + radius]
        - y 방향 범위: [point.y - radius, point.y + radius]
        - 이 범위로 shapely 의 `geom.box(...)` 를 사용해 네모(Polygon)를 만든다.
          · `patch`: Polygon, 모양 = 축에 정렬된 정사각형
          · 개념적 shape: 직사각형 이지만, 여기서는 항상 정사각형
            - 한 변 길이 = 2 * radius

    3) ego heading 기준으로 정사각형 회전
        - heading(라디안) → degree 로 변환: `angle_deg = heading * 180 / π`
        - `affinity.rotate(patch, angle_deg, origin=(point.x, point.y))` 호출
            · origin 을 ego 위치로 지정해서,
              정사각형이 ego 위치를 중심으로 회전하게 만든다.
        - 결과:
            · `rotated_patch`: Polygon
            · 한 변은 ego 진행 방향과 평행,
              나머지 한 변은 그에 정확히 수직

    4) 레이어별로 geometry ∩ rotated_patch 검사
        - 여기서부터는 NuPlanMap 구체 구현에 의존하므로
          `map_api` 가 `NuPlanMap` 인지 확인 후 캐스팅한다.
        - 각 레이어에 대해:
            a) `layer_df = map_api._get_vector_map_layer(layer)` 로
               해당 레이어의 벡터 데이터를 가져온다.
               · `layer_df["geometry"]`: 각 행의 도형(Polygon 등), shape ≈ (num_objects,)
            b) `mask = layer_df["geometry"].intersects(rotated_patch)` 로
               각 도형이 회전된 정사각형과 겹치는지 계산한다.
               · `mask`: pandas Series(bool), shape: (num_objects,)
                   - True  → 정사각형과 최소 한 점이라도 겹침
                   - False → 전혀 안 겹침
            c) `map_object_ids = layer_df.loc[mask]["fid"]` 로
               겹치는 행들의 id 를 뽑는다.
            d) `map_api.get_map_object(fid, layer)` 를 사용해
               실제 `MapObject` 인스턴스를 얻는다.
               이렇게 얻은 객체들을 리스트로 모아 `object_map[layer]` 에 저장한다.

    자료 구조 / shape 정리
    ----------------------
    입력
      - map_api (AbstractMap):
          · 실제로는 NuPlanMap 인스턴스여야 한다.
          · 그렇지 않으면 TypeError 를 일으킨다.

      - point (Point2D):
          · ego 위치 (x, y), 단위 m

      - heading (float):
          · ego 진행 방향(라디안)

      - radius (float):
          · 회전된 정사각형 한 변의 절반 길이 [m]
          · 정사각형 전체 크기 = (2 * radius) × (2 * radius)

      - layers (List[SemanticMapLayer]):
          · 예: [SemanticMapLayer.LANE, SemanticMapLayer.ROADBLOCK]

    중간 변수
      - patch: geom.Polygon
          · 축에 정렬된 정사각형
          · 좌표 범위: x ∈ [x-radius, x+radius], y ∈ [y-radius, y+radius]

      - rotated_patch: geom.Polygon
          · ego heading 에 따라 회전된 정사각형
          · patch 와 꼭짓점 좌표는 같지만, 회전된 상태

      - layer_df: VectorLayer (실제로는 GeoDataFrame)
          · 각 레이어의 벡터 데이터
          · `layer_df["geometry"]`: 각 객체의 도형, 길이 ≈ num_objects

      - mask: pandas.Series[bool]
          · shape: (num_objects,)
          · True 인 인덱스는 rotated_patch 와 교차하는 객체

    출력
      - object_map: Dict[SemanticMapLayer, List[MapObject]]
          · key: 입력으로 넘긴 각 레이어
          · value: 해당 레이어에서 “회전된 정사각형과 조금이라도 겹치는” MapObject 리스트
          · 각 리스트 길이 = 해당 레이어에서 조건을 만족하는 객체 수

    Args:
        map_api (AbstractMap):
            NuPlanMap 인스턴스여야 한다(내부 벡터 레이어 접근 필요).
        point (Point2D):
            정사각형 중심이 될 ego 위치 (x, y).
        heading (float):
            ego 진행 방향 (라디안).
        radius (float):
            정사각형 한 변의 절반 길이 [m].
        layers (List[SemanticMapLayer]):
            조회할 레이어 목록.

    Returns:
        Dict[SemanticMapLayer, List[MapObject]]:
            레이어별로, 회전된 정사각형과 조금이라도 겹치는 MapObject 들을 모은 딕셔너리.

    Raises:
        TypeError:
            - map_api 가 NuPlanMap 타입이 아닐 때.
        AssertionError:
            - 요청한 레이어 중 현재 맵에서 지원하지 않는 레이어가 있을 때.
    """
    if not isinstance(map_api, NuPlanMap):
        raise TypeError(
            f"`get_directional_proximal_map_objects` 는 NuPlanMap 전용입니다. "
            f"받은 타입: {type(map_api)!r}")

    # 1) 지원 레이어 확인
    supported_layers: List[
        SemanticMapLayer] = map_api.get_available_map_objects()
    unsupported_layers: List[SemanticMapLayer] = [
        layer for layer in layers if layer not in supported_layers
    ]
    assert len(unsupported_layers) == 0, (
        f"Object representation for layer(s): {unsupported_layers} is unavailable"
    )

    # 2) 회전 전 축정렬 정사각형 생성
    x_min, x_max = point.x - radius, point.x + radius
    y_min, y_max = point.y - radius, point.y + radius
    patch: geom.Polygon = geom.box(x_min, y_min, x_max, y_max)

    # 3) ego heading 기준으로 정사각형 회전 (deg 단위 필요)
    angle_deg: float = float(heading) * 180.0 / np.pi
    rotated_patch: geom.Polygon = affinity.rotate(
        patch,
        angle_deg,
        origin=(point.x, point.y),
    )

    object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

    # 4) 각 레이어에서 rotated_patch 와 intersects 인 객체만 선택
    for layer in layers:
        # VectorLayer 는 GeoDataFrame 과 비슷한 구조라고 보면 된다.
        layer_df = map_api._get_vector_map_layer(layer)

        # geometry 컬럼과 회전된 정사각형의 "겹침 여부"를 벡터화해서 계산
        # mask: (num_objects,) bool
        mask = layer_df["geometry"].intersects(rotated_patch)

        # mask 가 True 인 행들의 fid 를 가져온다.
        map_object_ids = layer_df.loc[mask]["fid"]

        # fid 로 실제 MapObject 인스턴스를 생성
        object_map[layer] = [
            map_api.get_map_object(map_object_id, layer)
            for map_object_id in map_object_ids
        ]

    return object_map


def get_circular_proximal_map_objects(
    map_api: AbstractMap,
    point: Point2D,
    radius: float,
    layers: List[SemanticMapLayer],
) -> Dict[SemanticMapLayer, List[MapObject]]:
    """원(동그라미) 반경 안에 *조금이라도 걸치는* 맵 객체들을 레이어별로 모아준다.

    이 함수는 NuPlanMap 의 :meth:`get_proximal_map_objects` 와 비슷하지만,
    **축에 정렬된 네모** 대신 **원 모양 영역**을 기준으로 객체를 찾는다.

    - 기존: point 를 중심으로 한 네모 영역
        · x ∈ [point.x - radius, point.x + radius]
        · y ∈ [point.y - radius, point.y + radius]
    - 이 함수: point 를 중심으로 한 **원(반지름 radius)**

    포함 기준
    ----------
    각 레이어의 모든 객체에 대해, 그 객체의 도형(geometry)이
    이 원과 shapely 의 `intersects` 여부를 체크한다.

    - `intersects(...)` 가 True 인 경우:
        · 원 안에 완전히 들어온 경우
        · 원 경계에 살짝 걸친 경우
        · 객체 대부분은 밖에 있지만 일부 모서리/변만 원에 닿는 경우
      모두 **포함**된다.

    내부 동작 순서
    --------------
    1) 입력 맵 타입 확인
        - `map_api` 가 실제로 NuPlanMap 인스턴스인지 확인한다.
          (내부 벡터 레이어에 접근해야 하므로 필수)

    2) 지원 레이어 검증
        - `map_api.get_available_map_objects()` 로 현재 맵에서 지원하는 레이어 목록을 얻는다.
        - 요청한 `layers` 중 지원하지 않는 레이어가 있으면 `assert` 로 바로 실패시킨다.

    3) 원(원판) 도형 생성
        - 중심점: `center = geom.Point(point.x, point.y)`
        - 원 도형: `patch = center.buffer(radius)`
          · `patch` 는 Shapely Polygon 이고, 원을 다각형으로 근사한 결과.
          · 개념적으로는 “반지름이 radius 인 원”이라고 보면 된다.

    4) 레이어별 geometry ∩ 원 검사
        - 각 레이어에 대해:
            a) `layer_df = map_api._get_vector_map_layer(layer)`
               · NuPlan 내부의 벡터 레이어(GeoDataFrame 유사 구조) 조회
               · `layer_df["geometry"]` 컬럼에는 각 행의 도형(Polygon 등)이 들어 있음
                 - shape: (num_objects,)
            b) `mask = layer_df["geometry"].intersects(patch)`
               · `mask`: 길이 (num_objects,) 의 bool Series
               · True  → 해당 geometry 가 원과 한 점이라도 겹친다
               · False → 전혀 겹치지 않는다
            c) `map_object_ids = layer_df.loc[mask]["fid"]`
               · 원과 겹치는 객체들의 id(fId)만 추출
            d) 각 id 에 대해 `map_api.get_map_object(fid, layer)` 를 호출하여
               실제 `MapObject` 인스턴스를 만들고 리스트에 담는다.
        - 이렇게 만들어진 리스트를 `object_map[layer]` 에 저장한다.

    자료 구조 / shape 정리
    ----------------------
    입력
      - map_api (AbstractMap):
          · 실제 타입: NuPlanMap (아니면 TypeError 발생)
      - point (Point2D):
          · ego 위치 (x, y), 단위 m
      - radius (float):
          · 원의 반지름 [m]
      - layers (List[SemanticMapLayer]):
          · 예: [SemanticMapLayer.LANE, SemanticMapLayer.ROADBLOCK]

    중간 변수
      - center: shapely.geometry.Point
          · (point.x, point.y)
      - patch: shapely.geometry.Polygon
          · `center.buffer(radius)` 로 만든 원 모양 영역
      - layer_df: VectorLayer (GeoDataFrame 비슷)
          · `layer_df["geometry"]`: 길이 = num_objects
      - mask: pandas.Series(bool)
          · shape: (num_objects,)
          · True 인 인덱스만 원과 겹치는 객체

    출력
      - object_map: Dict[SemanticMapLayer, List[MapObject]]
          · key: 입력으로 받은 각 레이어
          · value: 해당 레이어에서 원과 조금이라도 겹치는 맵 객체 리스트

    Args:
        map_api (AbstractMap):
            NuPlanMap 인스턴스여야 한다. (내부 벡터 레이어 접근 필요)
        point (Point2D):
            원의 중심이 될 포인트 (x, y).
        radius (float):
            원의 반경 [m].
        layers (List[SemanticMapLayer]):
            조회할 레이어 목록.

    Returns:
        Dict[SemanticMapLayer, List[MapObject]]:
            레이어별로, 중심 원과 한 점이라도 겹치는 MapObject 들을 모은 딕셔너리.

    Raises:
        TypeError:
            - map_api 가 NuPlanMap 타입이 아닐 때.
        AssertionError:
            - 요청한 레이어 중 지원되지 않는 레이어가 있을 때.
    """
    # 1) 요청 레이어가 실제로 지원되는지 확인
    supported_layers: List[
        SemanticMapLayer] = map_api.get_available_map_objects()
    unsupported_layers: List[SemanticMapLayer] = [
        layer for layer in layers if layer not in supported_layers
    ]
    assert len(unsupported_layers) == 0, (
        f"Object representation for layer(s): {unsupported_layers} is unavailable"
    )

    # 2) 원(패치) 생성: 중심은 point, 반경은 radius
    center: geom.Point = geom.Point(point.x, point.y)
    patch: geom.Polygon = center.buffer(radius)

    object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

    # 3) 각 레이어에서, 원과 intersects 인 geometry 만 선택
    for layer in layers:
        layer_df = map_api._get_vector_map_layer(layer)

        # geometry 가 원(patch)와 한 점이라도 겹치는 행만 선택
        # mask: (num_objects,) bool
        mask = layer_df["geometry"].intersects(patch)
        map_object_ids = layer_df.loc[mask]["fid"]

        object_map[layer] = [
            map_api.get_map_object(map_object_id, layer)
            for map_object_id in map_object_ids
        ]

    return object_map


def build_agent_route_lane_order(
    npc_route_on_chosen_lane_idx_list: List[
        List[int]],  # 길이: chosen_agent_num, 각 원소 길이 가변
    chosen_lane_num: int,
    dtype: np.dtype = np.int32,
) -> np.ndarray:  # shape: (chosen_agent_num, chosen_lane_num)
    """에이전트별로 '경로 위에서 가까운 순으로 선택된 차선 인덱스'를
    2차원 행렬 형태로 정리한다.

    이 함수는 이미 **거리 기준으로 정렬**된 차선 인덱스 리스트를 받아,
    각 에이전트에 대해 다음 규칙으로 랭크를 부여한다.

    규칙(에이전트 i 기준)
    --------------------
    - `npc_route_on_chosen_lane_idx_list[i]` 에는
      이 에이전트가 사용하는 차선 인덱스가
      [가까운 차선, 그 다음, ...] 순서로 들어있다.
        예: [5, 2, 7]  이면
            · lane 5 → rank 0
            · lane 2 → rank 1
            · lane 7 → rank 2
    - 같은 lane 인덱스가 여러 번 등장하면,
      **가장 처음 등장한 위치만** 랭크를 부여한다.
      · 예: [2, 5, 2, 7]  → lane 2 는 한 번만 취급
    - 이 에이전트가 사용하지 않는 차선(lane)은 -1 로 남겨둔다.

    구현 아이디어
    -------------
    1) 최종 결과 배열을 -1 로 초기화한다.
       - `agent_route_lane_order`: shape = (chosen_agent_num, chosen_lane_num)

    2) 각 에이전트에 대해:
       - Python 리스트를 넘파이 배열로 바꾼다. shape = (L,)
       - 음수 인덱스 / 범위 밖 인덱스가 있으면 즉시 에러를 낸다.
       - `np.unique(..., return_index=True)` 를 이용해
         **처음 등장 위치 기준**으로 중복을 제거한다.
         (unique 값 자체는 정렬되지만, 첫 등장 위치 인덱스를 다시 정렬해
          원래 순서를 복원한다.)
       - 유일한 lane 인덱스 배열에 대해
         [0, 1, 2, ...] 랭크를 만들어 한 번에 대입한다.

    Args:
        npc_route_on_chosen_lane_idx_list:
            - 길이: chosen_agent_num
            - 각 원소: 한 에이전트가 경로로 사용하는 차선 인덱스 리스트
              (이미 "가까운 순서"로 정렬되어 있다고 가정).
        chosen_lane_num:
            - 전체 고려하는 차선 개수.
            - 모든 인덱스는 0 이상, chosen_lane_num 미만이어야 한다.
        dtype:
            - 반환 배열의 dtype (기본값: np.int32).

    Returns:
        np.ndarray:
            - `agent_route_lane_order`
            - shape: (chosen_agent_num, chosen_lane_num)
            - 각 [i, j] 값:
                · j번째 차선이 에이전트 i의 route 상에서 가까운 순서일 때
                  0, 1, 2, ... 랭크 값
                · 그 에이전트의 route에 없는 차선이면 -1
    """
    chosen_agent_num: int = len(npc_route_on_chosen_lane_idx_list)

    # 결과 행렬 초기화: (chosen_agent_num, chosen_lane_num)
    agent_route_lane_order: np.ndarray = np.full(
        (chosen_agent_num, chosen_lane_num),
        -1,
        dtype=dtype,
    )

    # 에이전트가 없거나 차선이 없으면 바로 반환
    if chosen_agent_num == 0 or chosen_lane_num == 0:
        return agent_route_lane_order

    for agent_i, route_on_chosen_lane_idx in enumerate(
            npc_route_on_chosen_lane_idx_list):
        # 이 에이전트는 route 상에 차선이 없는 경우
        if not route_on_chosen_lane_idx:
            continue

        # lane_idx_list: (L,)
        lane_idx_list: np.ndarray = np.asarray(route_on_chosen_lane_idx,
                                               dtype=int)

        # 범위 체크 (벡터화)
        if np.any(lane_idx_list < 0):
            bad_idx: int = int(lane_idx_list[lane_idx_list < 0][0])
            raise ValueError(f"음수 인덱스가 발견되었습니다: {bad_idx}")
        if np.any(lane_idx_list >= chosen_lane_num):
            bad_idx = int(lane_idx_list[lane_idx_list >= chosen_lane_num][0])
            raise ValueError(
                f"lane_idx {bad_idx} 가 chosen_lane_num={chosen_lane_num} 범위를 벗어났습니다."
            )

        # 중복 제거 (첫 등장 순서 유지)
        # unique_vals: (M,), first_indices: (M,)
        unique_vals, first_indices = np.unique(lane_idx_list, return_index=True)
        # 원래 등장 순서대로 정렬
        order: np.ndarray = np.argsort(first_indices)  # (M,)
        unique_lane_idx: np.ndarray = unique_vals[order]  # (M,)

        # 랭크 벡터: [0, 1, 2, ...] shape: (M,)
        ranks: np.ndarray = np.arange(unique_lane_idx.shape[0], dtype=dtype)

        # 한 번에 대입
        agent_route_lane_order[agent_i, unique_lane_idx] = ranks

    return agent_route_lane_order


def _lane_min_dist_order(
        lanes_xy: np.ndarray,  # shape: (chosen_lane_num, lane_len, 2)
        neighbor_current_xy: np.ndarray,  # shape: (2,)
) -> np.ndarray:  # shape: (chosen_lane_num,)
    """에이전트 위치에서 각 차선까지의 최소 거리를 계산해, 가까운 순으로 정렬된 인덱스를 만든다.

    Args:
        lanes_xy:
            차선 폴리라인 좌표.
            shape = (chosen_lane_num, lane_len, 2).
        neighbor_current_xy:
            에이전트 현재 위치 [x, y].
            shape = (2,).

    Returns:
        np.ndarray:
            lane 인덱스가 "에이전트와 가까운 순"으로 정렬된 배열.
            shape = (chosen_lane_num,).
    """
    # diff: (chosen_lane_num, lane_len, 2)
    diff: np.ndarray = lanes_xy - neighbor_current_xy[None, None, :]
    # dists: (chosen_lane_num, lane_len)
    dists: np.ndarray = np.linalg.norm(diff, axis=-1)
    # min_dists: (chosen_lane_num,)
    min_dists: np.ndarray = np.min(dists, axis=1)
    # 가까운 순 정렬 인덱스 반환
    return np.argsort(min_dists)


def _select_lanes_by_order(
        chosen_lane_dist_order: np.ndarray,  # shape: (chosen_lane_num,)
        chosen_lanes_route_mask_arr: np.ndarray,  # shape: (chosen_lane_num,)
) -> List[int]:
    """거리 순 정렬 결과와 True/False 마스크를 이용해,
    “경로 위에 있는 차선” 인덱스만 골라낸다.

    제한 없이(True인 차선은 모두 사용) 선택한다.

    Args:
        chosen_lane_dist_order:
            에이전트와의 최소 거리 기준으로 정렬된 lane 인덱스.
            shape = (chosen_lane_num,).
        chosen_lanes_route_mask_arr:
            해당 lane 이 그 차량의 경로 위에 있는지 여부.
            shape = (chosen_lane_num,).

    Returns:
        List[int]:
            경로 위에 있는 lane 인덱스 리스트.
            가까운 순으로 정렬되어 있다.
    """
    route_on_chosen_lane_idx: List[int] = []
    for lane_idx in chosen_lane_dist_order:
        if chosen_lanes_route_mask_arr[lane_idx]:
            route_on_chosen_lane_idx.append(int(lane_idx))

    return route_on_chosen_lane_idx


def _select_token_and_ordered_npc_route_indices(
    car_token_to_chosen_lanes_route_mask: Dict[
        str, List[bool]],  # 길이: chosen_car_num / 값: 길이 = chosen_lane_num
    neighbor_track_token: List[str],  # 길이: chosen_agent_num
    neighbor_agents_current: np.ndarray,  # shape: (chosen_agent_num, 11)
    vector_map_lanes: np.ndarray,  # shape: (chosen_lane_num, lane_len, D)
) -> np.ndarray:  # shape: (chosen_agent_num, chosen_lane_num)
    """토큰별 "경로 위 차선" 정보를 이용해 에이전트×차선 랭크 행렬을 만든다.

    한 에이전트에 대해 하는 일
    --------------------------
    1) 자신의 차량 토큰으로 `chosen_lanes_route_mask` 를 가져온다.
       - 길이 = chosen_lane_num, bool 리스트
       - True 인 lane 만 이 에이전트의 경로 위에 있는 차선이다.

    2) `vector_map_lanes` 에서 좌표 부분만 꺼내고(lanes_xy),
       현재 에이전트 위치(neighbor_current_xy)와의 최소 거리 기준으로
       lane 인덱스를 가까운 순서로 정렬한다.
       - `_lane_min_dist_order` 사용
       - shape = (chosen_lane_num,)

    3) 정렬된 인덱스에서 `chosen_lanes_route_mask` 가 True 인 것만 골라
       "이 에이전트 입장에서 가까운 route 차선" 리스트를 만든다.
       - `_select_lanes_by_order` 사용
       - 예: [2, 5, 7]

    4) 모든 에이전트에 대해 위 과정을 반복해
       `npc_route_on_chosen_lane_idx_list` (에이전트별 lane 인덱스 리스트들)를 만들고,
       마지막에 `build_agent_route_lane_order(...)` 를 호출해
       (chosen_agent_num, chosen_lane_num) 랭크 행렬로 정리한다.

    최종 결과
    --------
    - 반환값 `agent_route_lane_order[i, j]`:
        · 에이전트 i 에 대해 j번째 lane 이
          route 상에서 얼마나 "앞 순서"에 있는지(0,1,2,...)를 나타낸다.
        · 해당 에이전트가 사용하지 않는 lane 은 -1 이다.

    Returns:
        np.ndarray:
            - `agent_route_lane_order`
            - shape: (chosen_agent_num, chosen_lane_num)
            - dtype: np.int32 (기본)
    """
    chosen_agent_num: int = int(neighbor_agents_current.shape[0])
    chosen_lane_num: int = int(vector_map_lanes.shape[0])

    # lanes_xy: (chosen_lane_num, lane_len, 2)
    lanes_xy: np.ndarray = vector_map_lanes[:, :, :2]

    # npc_route_on_chosen_lane_idx_list:
    #   길이 = chosen_agent_num, 각 원소: 선택된 lane 인덱스 리스트
    npc_route_on_chosen_lane_idx_list: List[List[int]] = []

    for agent_idx in range(chosen_agent_num):
        token: str = neighbor_track_token[agent_idx]

        # 각 차량 토큰에 대한 "route 위 차선" 마스크 (길이 = chosen_lane_num)
        chosen_lanes_route_mask: List[bool] = \
            car_token_to_chosen_lanes_route_mask[token]

        # chosen_lanes_route_mask_arr: (chosen_lane_num,)
        chosen_lanes_route_mask_arr: np.ndarray = np.asarray(
            chosen_lanes_route_mask,
            dtype=bool,
        )

        # 이 에이전트 경로 위에 있는 차선이 하나도 없으면 빈 리스트
        if chosen_lanes_route_mask_arr.sum() == 0:
            npc_route_on_chosen_lane_idx_list.append([])
            continue

        # neighbor_current_xy: (2,)
        neighbor_current_xy: np.ndarray = neighbor_agents_current[agent_idx, :2]

        # chosen_lane_dist_order: (chosen_lane_num,)
        chosen_lane_dist_order: np.ndarray = _lane_min_dist_order(
            lanes_xy=lanes_xy,
            neighbor_current_xy=neighbor_current_xy,
        )

        # 경로 위(True)인 lane 전부 선택 (가까운 순으로)
        route_on_chosen_lane_idx: List[int] = _select_lanes_by_order(
            chosen_lane_dist_order=chosen_lane_dist_order,
            chosen_lanes_route_mask_arr=chosen_lanes_route_mask_arr,
        )
        npc_route_on_chosen_lane_idx_list.append(route_on_chosen_lane_idx)

    # 에이전트 수와 길이 정합성 체크
    assert len(npc_route_on_chosen_lane_idx_list) == chosen_agent_num, (
        f"npc_route_on_chosen_lane_idx_list 길이({len(npc_route_on_chosen_lane_idx_list)}) "
        f"!= chosen_agent_num({chosen_agent_num})")

    # (chosen_agent_num, chosen_lane_num)
    agent_route_lane_order: np.ndarray = build_agent_route_lane_order(
        npc_route_on_chosen_lane_idx_list=npc_route_on_chosen_lane_idx_list,
        chosen_lane_num=chosen_lane_num,
    )
    return agent_route_lane_order.astype(np.int64)


def get_neighbor_track_tokens(
    present_tracked_objects: TrackedObjects,
    agents_cur_frame_indices: Union[Sequence[int], np.ndarray],
    agents_num: int,
    object_types: Optional[Sequence[TrackedObjectType]] = None,
) -> List[Optional[str]]:
    """현재 프레임에서 선택된 이웃 에이전트들의 `track_token` 리스트를 만든다.

    개요
    ----
    이 함수는 다음 두 정보를 합쳐서,
    **“이웃 에이전트 슬롯 순서에 맞는 track_token 리스트”**를 만들어 줍니다.

    1) `present_tracked_objects`
        - 현재 프레임에서 감지된 모든 객체 묶음입니다.
        - 여기서 차량/보행자/자전거 등 관심 있는 타입만 추려,
          내부적으로 “에이전트 배열”을 만들었다고 가정합니다.
        - 이때의 순서는 `_extract_agent_array` 에서 사용한 것과 동일합니다
          (즉, 같은 타입 필터 순서로 정렬됨).

    2) `agents_cur_frame_indices`
        - `agent_past_process` 가 선택한 이웃 에이전트의
          “현재 프레임 기준 행 인덱스” 목록입니다.
        - 길이 K(≤ agents_num) 인 정수 시퀀스이며,
          이 순서가 곧 이웃 에이전트 슬롯 순서가 됩니다.

    이 함수는,
    - 현재 프레임에서 관심 타입 에이전트들을 순서대로 나열한 뒤
    - `agents_cur_frame_indices[k]` 를 이용해 해당 행의 `track_token` 을 꺼내
      `neighbor_track_token[k]` 에 채워 넣습니다.
    - 슬롯 개수 `agents_num` 만큼의 리스트를 항상 반환하며,
      인덱스 범위를 벗어나거나 매핑할 수 없는 경우에는 `None` 으로 채웁니다.

    Args:
        present_tracked_objects (TrackedObjects):
            현재 프레임의 감지 결과.
            여러 타입의 객체를 포함할 수 있으며,
            내부에서 `get_tracked_objects_of_types(object_types)` 로
            관심 타입만 추려 사용합니다.
        agents_cur_frame_indices (Union[Sequence[int], np.ndarray]):
            - shape: (K,)
            - 이웃 에이전트들이 현재 프레임 에이전트 배열에서 차지하는 행 인덱스들.
            - `agent_past_process` 의 `agents_cur_frame_indices` 를 그대로 넘겨 사용합니다.
        agents_num (int):
            - 출력할 이웃 슬롯의 개수입니다.
            - 반환되는 리스트 길이가 됩니다.
        object_types (Optional[Sequence[TrackedObjectType]]):
            - 필터링할 객체 타입 목록입니다.
            - 기본값은 `(VEHICLE, PEDESTRIAN, BICYCLE)` 이며,
              `_extract_agent_array` 에서 사용한 타입 순서와 동일해야
              인덱스 매핑이 올바르게 유지됩니다.

    Returns:
        List[Optional[str]]:
            - 길이: `agents_num`
            - 각 원소는 해당 이웃 슬롯에 대응하는 `track_token` (문자열) 이거나,
              매핑할 수 없을 때는 `None` 입니다.
            - `agents_cur_frame_indices` 가 `None` 이면
              길이 `agents_num` 의 `[None, None, ...]` 리스트를 반환합니다.

    Raises:
        ValueError:
            - `agents_num` 이 음수인 경우.

    Notes:
        - `agents_cur_frame_indices` 의 길이가 `agents_num` 보다 길면,
          앞에서부터 `agents_num` 개까지만 사용합니다.
        - 현재 프레임에서 관심 타입으로 필터링한 에이전트 개수를 `M` 이라 할 때,
          인덱스가 `0 <= idx < M` 범위를 벗어나면 해당 슬롯은 `None` 으로 남습니다.
    """
    if agents_num < 0:
        raise ValueError(f"`agents_num`은 음수가 될 수 없습니다. got {agents_num}")

    # `_extract_agent_array`와 동일한 타입 필터 순서 유지
    if object_types is None:
        object_types = (
            TrackedObjectType.VEHICLE,
            TrackedObjectType.PEDESTRIAN,
            TrackedObjectType.BICYCLE,
        )

    # 현재 프레임에서 관심 타입만 '그 순서 그대로' 나열
    # current_agents: List[TrackedObject], 길이 = M
    current_agents: List[
        TrackedObject] = present_tracked_objects.get_tracked_objects_of_types(
            object_types)  # type: ignore[assignment]
    # tokens_in_present_order: (M,) — 관심 타입 에이전트들의 track_token 문자열
    tokens_in_present_order: List[str] = [
        str(agent.track_token) for agent in current_agents
    ]

    # 반환 버퍼 준비: 길이 = agents_num
    neighbor_track_token: List[Optional[str]] = [None] * int(agents_num)

    # agents_cur_frame_indices 정규화(int list)
    if agents_cur_frame_indices is None:
        return neighbor_track_token
    # numpy, list, tuple 등 모두 int 리스트로 캐스팅
    # idx_list: List[int], 길이 = K
    idx_list: List[int] = list(
        map(int,
            np.asarray(agents_cur_frame_indices).reshape(-1).tolist()))

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
def _local_to_local_transforms(
        global_states1: np.ndarray,  # (N, 3) = [x1, y1, heading1] ...
        global_states2: np.ndarray,  # (3,)   = [x_ref, y_ref, heading_ref]
) -> np.ndarray:
    """한 좌표계 기준의 포즈 집합을, 다른 좌표계 기준으로 한 번에 변환하는 함수.

    이 함수는 다음과 같이 동작합니다.

    - `global_states2`:
      · 새로운 기준 좌표계(로컬 프레임)의 포즈 [x, y, heading] 입니다.
    - `global_states1`:
      · 예전 기준(세계 좌표계라고 가정)에서 표현된 포즈들의 집합입니다.
      · 각 행이 하나의 포즈 [x, y, heading] 입니다.

    절차:
        1. `global_states2`로부터 3x3 변환행렬(포즈 → 동차변환)을 만든다.
        2. 이를 역행렬로 뒤집어, "세계 → 새로운 로컬 프레임" 변환행렬을 얻는다.
        3. `global_states1`의 각 포즈에 대해서도 3x3 변환행렬을 만든다.
        4. (2)의 행렬을 (3)에 왼쪽에서 곱해, 모두 새로운 로컬 좌표계 기준으로 변환한다.

    결과적으로,
    - 입력으로 주어진 여러 포즈의 변환행렬 묶음이
      "새 기준 좌표계에서 본 포즈"로 바뀐 형태로 반환됩니다.

    Args:
        global_states1 (np.ndarray):
            - shape: (N, 3)
            - 각 행: [x, y, heading] (예: 세계 좌표계 기준 포즈들).
        global_states2 (np.ndarray):
            - shape: (3,)
            - 기준이 될 포즈 [x_ref, y_ref, heading_ref].

    Returns:
        np.ndarray:
            - shape: (N, 3, 3)
            - 각 원소는 `global_states1`의 각 포즈를
              `global_states2` 기준 로컬 프레임으로 본 3x3 변환행렬입니다.
    """
    # local_xform: (3, 3) — 기준 포즈(global_states2)에 대한 동차 변환행렬
    local_xform = _state_se2_array_to_transform_matrix(global_states2)
    # local_xform_inv: (3, 3) — 기준 포즈의 역변환(세계→로컬)
    local_xform_inv = np.linalg.inv(local_xform)

    # transforms: (N, 3, 3) — global_states1 의 각 포즈에 대한 변환행렬
    transforms = _state_se2_array_to_transform_matrix_batch(global_states1)

    # (N, 3, 3) — 새 로컬 프레임 기준으로 재표현된 변환행렬들
    transforms = np.matmul(local_xform_inv, transforms)
    return transforms


def _state_se2_array_to_transform_matrix(
        input_data: np.ndarray,  # (3,) = [x, y, heading]
) -> np.ndarray:  # (3, 3)
    """단일 SE(2) 상태 [x, y, heading] 을 3x3 동차 변환행렬로 바꾸는 함수.

    행렬 구조:
        [[ cos(h), -sin(h), x ],
         [ sin(h),  cos(h), y ],
         [   0   ,    0   , 1 ]]

    이 행렬을 점 [x', y', 1]^T 에 곱하면, 회전+평행이동이 한 번에 적용됩니다.

    Args:
        input_data (np.ndarray):
            - shape: (3,)
            - [x, y, heading] (라디안).

    Returns:
        np.ndarray:
            - shape: (3, 3)
            - 주어진 포즈를 표현하는 2D SE(2) 동차 변환행렬.
    """
    x: float = float(input_data[0])
    y: float = float(input_data[1])
    h: float = float(input_data[2])

    cosine = np.cos(h)
    sine = np.sin(h)

    # (3, 3)
    return np.array([[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]])


def _state_se2_array_to_transform_matrix_batch(
        input_data: np.ndarray,  # (N, 3) = [[x1, y1, h1], ..., [xN, yN, hN]]
) -> np.ndarray:  # (N, 3, 3)
    """여러 개의 [x, y, heading] 포즈를 한 번에 3x3 변환행렬 묶음으로 바꾸는 함수.

    이 함수는 각 행이 [x, y, heading] 인 2D 포즈 배열을 입력받아,
    각 포즈마다 SE(2) 동차 변환행렬을 만들어 (N, 3, 3) 형태로 반환합니다.

    내부 아이디어:
        1. 각 포즈를 [x, y, cos(h), sin(h), 1] 형태로 확장한다.
        2. 미리 준비된 `reshaping_array` (5x9) 를 곱해,
           [c, -s, x, s, c, y, 0, 0, 1] 형태의 행(길이 9)을 만든다.
        3. 이를 (3, 3) 으로 reshape 하면, 개별 변환행렬이 완성된다.
        4. 이런 행을 N개 쌓아 (N, 3, 3) 배열을 얻는다.

    Args:
        input_data (np.ndarray):
            - shape: (N, 3)
            - 각 행: [x, y, heading] (라디안).

    Returns:
        np.ndarray:
            - shape: (N, 3, 3)
            - 각 원소는 해당 행 포즈에 대한 동차 변환행렬입니다.
    """
    # input_data: (N, 3) = [x, y, heading]
    # processed_input: (N, 5) = [x, y, cos(h), sin(h), 1]
    processed_input = np.column_stack((
        input_data[:, 0],
        input_data[:, 1],
        np.cos(input_data[:, 2]),
        np.sin(input_data[:, 2]),
        np.ones_like(input_data[:, 0]),
    ))

    # reshaping_array: (5, 9)
    reshaping_array = np.array([
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, -1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1],
    ])
    # processed_input @ reshaping_array: (N, 9)
    # → 각 행이 [c, -s, x, s, c, y, 0, 0, 1] 꼴이 되고,
    #   이를 (N, 3, 3) 으로 reshape 하면 변환행렬 세트가 됨.
    return (processed_input @ reshaping_array).reshape(-1, 3, 3)


def _transform_matrix_to_state_se2_array_batch(
        input_data: np.ndarray,  # (N, 3, 3)
) -> np.ndarray:  # (N, 3)
    """여러 개의 3x3 변환 행렬을 [x, y, heading] 형태의 포즈 배열로 되돌리는 함수.

    이 함수는 SE(2) 동차 변환행렬 묶음(회전+이동 정보를 가진 3x3 행렬들)을 입력으로 받아,
    각 행렬에 대해 다음 정보를 추출합니다.

    - x: 3번째 열의 x 성분 (translation x)
    - y: 3번째 열의 y 성분 (translation y)
    - heading: 회전 행렬의 첫 번째 열로부터 atan2를 사용해 추출한 각도

    즉, 다음과 같은 과정을 거칩니다.

    1. 각 3x3 행렬의 첫 번째 열을 모아서
       [cos(heading), sin(heading), _] 꼴의 벡터들을 만든다.
    2. 이로부터 `atan2(sin, cos)` 계산으로 heading(라디안)을 구한다.
    3. 원래 변환행렬의 3번째 열(translation [x, y, 1])에 대해,
       마지막 요소를 heading 값으로 덮어써 [x, y, heading] 형태로 만든다.

    Args:
        input_data (np.ndarray):
            - shape: (N, 3, 3)
            - 각 [i, :, :] 는 하나의 SE(2) 동차 변환행렬입니다.

    Returns:
        np.ndarray:
            - shape: (N, 3)
            - 각 행은 [x, y, heading] 형태의 포즈를 나타냅니다.
    """
    # first_columns: (N, 3) — 각 변환행렬의 첫 번째 열 [cos, sin, 0]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    # angles: (N,) — atan2(sin, cos) 로부터 구한 heading
    angles = np.arctan2(first_columns[:, 1], first_columns[:, 0])

    # result: (N, 3) — 원래는 3번째 열 [x, y, 1] 이었음
    result = input_data[:, :, 2]
    # 마지막 성분을 heading 으로 덮어써 [x, y, heading] 으로 만듦
    result[:, 2] = angles

    return result


def _global_state_se2_array_to_local(
    global_states: np.ndarray,  # (N, 3) = [x_world, y_world, heading_world]
    local_state: np.ndarray,  # (3,)   = [x_ref, y_ref, heading_ref]
) -> np.ndarray:  # (N, 3) = [x_local, y_local, heading_local]
    """여러 점의 [x, y, heading]을 기준 포즈(local_state) 기준 로컬 좌표계로 변환한다.

    개념적으로 이 함수는
    - `global_states` : 세계(월드) 좌표계 기준의 포즈들 집합
    - `local_state`   : “새 기준 좌표계”가 될 포즈(예: ego 차량의 현재 포즈)
    를 받아서, 각 포즈를 `local_state` 기준으로 보았을 때의
    로컬 좌표 [x_local, y_local, heading_local] 로 바꿔 줍니다.

    처리 순서:
        1. `local_state`로부터 3x3 변환 행렬(세계 → 로컬 프레임)을 만든다.
        2. `global_states`의 각 [x, y, heading]을 3x3 동차 변환행렬로 바꾼다.
        3. (1)의 역행렬을 (2)에 곱해, 모든 포즈를 로컬 좌표계 기준으로 재표현한다.
        4. 변환된 3x3 행렬 묶음을 다시 [x, y, heading] 형식의 배열로 되돌린다.

    Args:
        global_states (np.ndarray):
            - shape: (N, 3)
            - 각 행: [x_world, y_world, heading_world]
              (세계 좌표계 기준 포즈들).
        local_state (np.ndarray):
            - shape: (3,)
            - [x_ref, y_ref, heading_ref]
              로컬 좌표계의 기준이 되는 포즈(예: ego 포즈).

    Returns:
        np.ndarray:
            - shape: (N, 3)
            - 각 행: [x_local, y_local, heading_local]
              · `local_state`를 원점/기준으로 하는 좌표계 기준 포즈입니다.
    """
    # local_xform: (3, 3) — 기준 포즈(local_state)에 대한 동차 변환행렬 (world→기준)
    local_xform = _state_se2_array_to_transform_matrix(local_state)
    # local_xform_inv: (3, 3) — 기준 포즈의 역변환 (기준→world) 의 역 → (world→local)
    local_xform_inv = np.linalg.inv(local_xform)

    # transforms: (N, 3, 3) — 각 global_state 를 world 기준 변환행렬로 표현
    transforms = _state_se2_array_to_transform_matrix_batch(global_states)

    # transforms: (N, 3, 3) — world 기준 포즈들을 local_state 기준 로컬 프레임으로 변환
    transforms = np.matmul(local_xform_inv, transforms)

    # output: (N, 3) — [x_local, y_local, heading_local]
    output = _transform_matrix_to_state_se2_array_batch(transforms)

    return output


def _global_velocity_to_local(
    velocity: np.ndarray,  # (N, 2) = [vx_world, vy_world]
    anchor_heading: float,  # 스칼라 heading(rad) 또는 브로드캐스트 가능한 값
) -> np.ndarray:  # (N, 2) = [vx_local, vy_local]
    """월드 좌표계 기준 속도 벡터를 ego(또는 기준 heading) 좌표계 기준 속도로 회전 변환한다.

    이 함수는 2D 속도 벡터 [vx, vy] (세계 좌표계 기준)를,
    기준 차량(ego)의 heading(방향각)을 기준으로 회전시켜
    ego 좌표계 기준 속도 [vx_local, vy_local] 로 바꾸어 줍니다.

    변환 방식:
        - 기준 heading = θ 라 할 때,
          · vx_local = vx * cos(θ) + vy * sin(θ)
          · vy_local = vy * cos(θ) - vx * sin(θ)

    직관적으로,
    - 세계 기준으로 측정된 속도를,
    - ego 차량이 바라보는 방향을 x축으로 하는 좌표계로 "돌려서" 표현한다고 보면 됩니다.

    Args:
        velocity (np.ndarray):
            - shape: (N, 2)
            - 각 행: [vx_world, vy_world] (세계 좌표계 기준 속도).
        anchor_heading (float):
            - 기준이 되는 heading 값(rad).
            - 보통 ego 차량의 heading 을 넣어 사용합니다.
            - 스칼라이지만, 넘파이 브로드캐스팅 덕분에 벡터화 연산이 가능합니다.

    Returns:
        np.ndarray:
            - shape: (N, 2)
            - 각 행: [vx_local, vy_local]
            - 기준 heading 좌표계(ego 기준)로 회전된 속도 벡터입니다.
    """
    # velocity_x: (N,) — ego 기준 x 방향 속도
    velocity_x = velocity[:, 0] * np.cos(
        anchor_heading) + velocity[:, 1] * np.sin(anchor_heading)
    # velocity_y: (N,) — ego 기준 y 방향 속도
    velocity_y = velocity[:, 1] * np.cos(
        anchor_heading) - velocity[:, 0] * np.sin(anchor_heading)

    # (N, 2) 로 스택
    return np.stack([velocity_x, velocity_y], axis=-1)


def _build_ego_pose_from_state(
        ego_cur_pose_np: np.ndarray,  # (3,)
) -> np.ndarray:  # (3,)
    """EgoState 배열에서 ego 기준 좌표 변환에 사용할 [x, y, heading] 벡터를 만든다.

    이 함수는 ego_cur_pose_np 배열에서
    - x 좌표
    - y 좌표
    - heading(방향, rad 단위)
    세 값을 뽑아서, 부동소수 형태의 1차원 벡터로 만들어준다.

    Args:
        ego_cur_pose_np (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] 를 담고 있는 배열.

    Returns:
        np.ndarray:
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] 를 float64 타입으로 담은 벡터.
    """
    # ego_pose: (3,) = [x_ego, y_ego, heading_ego]
    ego_pose = np.array(
        [
            float(ego_cur_pose_np[EgoInternalIndex.x()]),
            float(ego_cur_pose_np[EgoInternalIndex.y()]),
            float(ego_cur_pose_np[EgoInternalIndex.heading()]),
        ],
        dtype=np.float64,
    )
    return ego_pose


def _convert_ego_history_to_relative(
        agent_state: np.ndarray,  # (time_num, state_dim_ego=10)
        ego_pose: np.ndarray,  # (3,)
) -> np.ndarray:  # (time_num, state_dim_ego+1=11)
    """ego(자차) 궤적을 월드 좌표계에서 ego 기준 상대 좌표계로 변환한다.

    이고의 과거~현재 상태 시퀀스를 받아서,
    - 위치/방향: ego 기준 좌표계로 변환
    - heading: cos, sin 두 값으로 나누어 저장
    - 속도: 월드 기준 속도를 ego 기준 속도로 회전 변환
    - 차량 크기/타입(one-hot) 등 뒤쪽 값은 그대로 복사

    최종적으로 원래보다 차원이 1 늘어난 (N, state_dim+1) 형태의 배열을 만든다.

    Args:
        agent_state (np.ndarray):
            - shape: (time_num, state_dim_ego=10)
            - ego seq 궤적 (월드 좌표계).
        ego_pose (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] (월드 좌표계 기준 현재 ego 상태).

    Returns:
        np.ndarray:
            - shape: (time_num, state_dim_ego+1=11)
            - ego 기준 상대 좌표계로 변환된 ego 궤적.
    """
    # agent_state: (time_num, state_dim_ego=10)
    time_num, state_dim = agent_state.shape

    # new_agent_state: (time_num, state_dim_ego+1=11)
    new_agent_state = np.zeros((agent_state.shape[0], state_dim + 1),
                               dtype=np.float64)

    # 크기/타입 등 뒤쪽 항목 복사
    new_agent_state[:, 6:] = agent_state[:, 5:]

    # agent_global_poses: (time_num, 3) = [x, y, heading]
    agent_global_poses = agent_state[:, [
        EgoInternalIndex.x(),
        EgoInternalIndex.y(),
        EgoInternalIndex.heading()
    ]]  # (N, 3)

    # transforms: (time_num, 3, 3)  — 월드→ego 변환 행렬
    # agent_global_poses: (time_num, 3), 절대 좌표계 기준 값
    # ego_pose # (3,) : 절대 좌표계 기준 값
    transforms = _local_to_local_transforms(agent_global_poses, ego_pose)

    # transformed_poses: (time_num, 3) — ego 좌표계 기준 [x, y, heading]
    transformed_poses = _transform_matrix_to_state_se2_array_batch(
        transforms)  # transformed_poses: ego 좌표계 기준 값

    # 위치/방향(→cos,sin) 갱신
    new_agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0]
    new_agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1]
    new_agent_state[:, 2] = np.cos(transformed_poses[:, 2])
    new_agent_state[:, 3] = np.sin(transformed_poses[:, 2])

    # --- velocity (world -> anchor ego frame) ---
    # agent_global_velocities: (time_num, 2) = [vx_world, vy_world]
    agent_global_velocities = agent_state[:, [
        EgoInternalIndex.vx(), EgoInternalIndex.vy()
    ]]

    # transformed_velocities: (time_num, 2) = [vx_ego, vy_ego]
    transformed_velocities = _global_velocity_to_local(agent_global_velocities,
                                                       ego_pose[-1])

    new_agent_state[:, 4] = transformed_velocities[:, 0]
    new_agent_state[:, 5] = transformed_velocities[:, 1]

    return new_agent_state


def _convert_agent_states_to_relative(
        agent_state: np.ndarray,  # (N, state_dim_agent)
        ego_pose: np.ndarray,  # (3,)
) -> np.ndarray:  # (N, state_dim_agent)
    """주변 에이전트(차량/보행자/자전거)의 상태를 ego 기준 상대 좌표계로 변환한다.

    월드 좌표계 기준으로 기록된 주변 에이전트들의 상태에서
    - 위치 (x, y)
    - 방향 (heading)
    - 속도 (vx, vy)
    만 이고 기준 좌표계로 바꿔준다.

    나머지 값들(차량 크기, 기타 특성)은 그대로 유지하며,
    입력 배열을 in-place 로 수정한 뒤 반환한다.

    Args:
        agent_state (np.ndarray):
            - shape: (N, state_dim_agent)
                [track_id, vx, vy, heading, width, length, x, y]
            - 주변 에이전트 상태 배열.
              스키마는 AgentInternalIndex 를 따른다.
        ego_pose (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] (월드 좌표계 기준 현재 ego 상태).

    Returns:
        np.ndarray:
            - shape: (N, state_dim_agent)
                - [track_id, vx, vy, heading, width, length, x, y]
            - 위치/방향/속도가 ego 기준으로 바뀐 에이전트 상태 배열.
    """
    # agent_global_poses: (N, 3) = [x, y, heading]
    agent_global_poses = agent_state[:, [
        AgentInternalIndex.x(),
        AgentInternalIndex.y(),
        AgentInternalIndex.heading()
    ]]

    # agent_global_velocities: (N, 2) = [vx_world, vy_world]
    agent_global_velocities = agent_state[:, [
        AgentInternalIndex.vx(
        ), AgentInternalIndex.vy()
    ]]

    # transformed_poses: (N, 3) = [x_ego, y_ego, heading_ego]
    transformed_poses = _global_state_se2_array_to_local(
        agent_global_poses, ego_pose)

    # transformed_velocities: (N, 2) = [vx_ego, vy_ego]
    transformed_velocities = _global_velocity_to_local(agent_global_velocities,
                                                       ego_pose[-1])

    # 위치/방향/속도 갱신 (in-place)
    agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0]
    agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1]
    agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2]
    agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0]
    agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1]

    return agent_state


def _convert_static_states_to_relative(
        agent_state: np.ndarray,  # (N, state_dim_static)
        ego_pose: np.ndarray,  # (3,)
) -> np.ndarray:  # (N, state_dim_static)
    """정적 객체(표지판, 배리어 등)의 위치/방향을 ego 기준 상대 좌표계로 변환한다.

    정적 객체의 상태 배열에서 앞의 세 값
    - x 좌표
    - y 좌표
    - heading(방향)
    만 ego 기준 좌표계로 변환하고, 나머지 값(크기 등)은 그대로 둔다.

    입력 배열을 in-place 로 수정한 뒤 반환한다.

    Args:
        agent_state (np.ndarray):
            - shape: (N, state_dim_static)
            - 정적 객체 상태 배열. 앞 3차원이 [x, y, heading].
        ego_pose (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] (월드 좌표계 기준 현재 ego 상태).

    Returns:
        np.ndarray:
            - shape: (N, state_dim_static)
            - 위치/방향이 ego 기준으로 바뀐 정적 객체 상태 배열.
    """
    # agent_global_poses: (N, 3) = [x, y, heading]
    agent_global_poses = agent_state[:, [0, 1, 2]]

    # transformed_poses: (N, 3) = [x_ego, y_ego, heading_ego]
    transformed_poses = _global_state_se2_array_to_local(
        agent_global_poses, ego_pose)

    # 위치/방향 갱신 (in-place)
    agent_state[:, 0] = transformed_poses[:, 0]
    agent_state[:, 1] = transformed_poses[:, 1]
    agent_state[:, 2] = transformed_poses[:, 2]

    return agent_state


def convert_absolute_quantities_to_relative(
    agent_state: np.ndarray,  # (N, state_dim)
    ego_cur_pose_np: np.ndarray,  # (3,)
    agent_type: str = 'ego',
) -> np.ndarray:
    """월드 좌표계 기준 상태들을 **ego(자차) 기준 상대 좌표계**로 변환하는 함수.

    이 함수는 세 가지 경우를 처리합니다.

    1) agent_type == 'ego'
        - 입력: 이고(자차)의 과거/현재 궤적 (월드 좌표계) # (num_frames, 10)
            - x, y, heading, vx, vy, width, length, (car, pedestrian, cyclist)
        - 출력: 이고 기준으로 다시 표현된 궤적 # (num_frames, 11)
          (위치/방향/속도는 ego 기준, 차체 크기와 타입(one-hot)은 그대로 유지)
        - 결과 shape: (N, original_dim + 1)
          · heading → cos, sin 두 차원으로 나뉘면서 1차원 증가

    2) agent_type == 'agent'
        - 입력: 주변 에이전트(차량/보행자/자전거 등)의 상태 (월드 좌표계)
            - [track_id, vx, vy, heading, width, length, x, y]
        - 출력: 이고 기준 상대 좌표계로 변환된 에이전트 상태
          · 위치/방향/속도만 ego 기준으로 바뀌고, 나머지는 그대로 유지
        - in-place 방식으로 `agent_state`를 수정 후 반환

    3) agent_type == 'static'
        - 입력: 정적 객체(표지판, 배리어 등)의 상태 (월드 좌표계)
          · [x, y, heading] + 크기 등
        - 출력: 이고 기준 상대 좌표계로 변환된 정적 객체 상태
          · 위치/방향만 ego 기준으로 바뀜

    Args:
        agent_state (np.ndarray):
            - shape: (N, state_dim)
            - 변환 대상 상태 배열.
              · ego 모드: 이고 궤적
              · agent 모드: 주변 동적 객체(에이전트)
              · static 모드: 정적 객체
        ego_cur_pose_np (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, heading_ego] (월드 좌표계 기준 이고 현재 상태)
        agent_type (str, optional):
            - 'ego'   : 이고 궤적 변환 모드
            - 'agent' : 동적 에이전트 변환 모드
            - 'static': 정적 객체 변환 모드

    Returns:
        np.ndarray:
            - 변환된 상태 배열.
            - 'ego' 모드: shape (N, state_dim + 1)
                - x, y, cos, sin, vx, vy, width, length, (car, pedestrian, cyclist)
            - 'agent' 모드: shape (N, state_dim) (in-place 수정)
                - [track_id, vx, vy, heading, width, length, x, y]
            - 'static' 모드: shape (N, state_dim) (in-place 수정)
    """
    # ego_pose: (3,) = [x_ego, y_ego, heading_ego]
    ego_pose = _build_ego_pose_from_state(ego_cur_pose_np)

    if agent_type == 'ego':
        # (time_num, state_dim_ego+1=11)
        agent_state = _convert_ego_history_to_relative(agent_state, ego_pose)

    elif agent_type == 'agent':
        # (N, state_dim_agent)
        agent_state = _convert_agent_states_to_relative(agent_state, ego_pose)

    elif agent_type == 'static':
        # (N, state_dim_static)
        agent_state = _convert_static_states_to_relative(agent_state, ego_pose)

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


def convert_data_dict_to_device_tensors(
    data: Mapping[str, Any],
    device: Union[torch.device, str],
    squeeze: bool,
) -> Dict[str, torch.Tensor]:
    """
    파이썬/NumPy 기반 딕셔너리를 모델 입력용 torch.Tensor 딕셔너리로 변환한다.

    기능 요약
    --------
    - 값이 이미 torch.Tensor 인 경우: 다시 만들지 않고 .to(device, dtype) 만 호출
    - NumPy 배열: from_numpy 로 래핑 후 float32 또는 bool 로 캐스팅
    - 파이썬 스칼라/리스트: torch.as_tensor(...) 로 생성
    - squeeze=False 인 경우: 앞쪽에 배치 차원 1개를 추가(unsqueeze(0))
    - 키가 "agent_route_lane_order" 인 텐서는 항상 torch.int64 로 맞춤

    Args:
        data:
            키-값 딕셔너리.
            값은 torch.Tensor / np.ndarray / 파이썬 수치/리스트 등을 허용한다.
        device:
            텐서를 올릴 디바이스. (예: "cuda:0", torch.device("cpu"))
        squeeze:
            False 이면, 모든 텐서 앞에 배치 차원 1을 추가한다.

    Returns:
        Dict[str, torch.Tensor]:
            각 키에 대응하는 torch.Tensor 로 구성된 딕셔너리.
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
                t = torch.from_numpy(v).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

        # 3) 파이썬 bool 스칼라
        elif isinstance(v, (bool, np.bool_)):
            t = torch.tensor(v, dtype=torch.bool, device=device)

        # 4) 나머지(리스트/스칼라 등): as_tensor로 한 번에
        else:
            t = torch.as_tensor(v, dtype=torch.float32, device=device)

        if not squeeze:
            t = t.unsqueeze(0)

        if k == "agent_route_lane_order":
            t = t.to(torch.int64)

        out[k] = t

    return out
