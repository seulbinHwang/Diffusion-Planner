"""
Module: Map Data Preprocessing Functions
Description: This module contains functions for Map related data processing.

Categories:
    1. Get lanes, speed limit, traffic light and lane's roadblock ids
    2. Get maps array for model input
"""

from typing import List, Dict, Tuple, Set, Optional
import numpy as np
from shapely import LineString

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.nuplan_map.utils import get_distance_between_map_object_and_point
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines, VectorFeatureLayer, LaneSegmentLaneIDs,
    VectorFeatureLayerMapping, LaneSegmentTrafficLightData,
    get_traffic_light_encoding, get_map_object_polygons)

from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame, _select_token_and_ordered_npc_route_indices, get_directional_proximal_map_objects, get_circular_proximal_map_objects


# =====================
# 1. Get lanes, speed limit, traffic light and lane's roadblock ids
# =====================
def _collect_lane_map_objects(
    map_api: AbstractMap,
    ego_point_2d: Point2D,
    ego_heading: float,
    radius: float,
) -> List[MapObject]:
    """ego 주변의 차선·차선연결 객체들을 “앞쪽 위주 + 거리순”으로 모은다.

    이 함수는 다음과 같은 상황에서 쓸 수 있다.

    - "지금 내 차 주변, 일정 거리 안에 어떤 차선들이 있는지 알고 싶다."
    - "특히 진행 방향 앞쪽에 있는 차선들만 보고 싶다."

    동작 순서
    --------
    1) 조회 대상 레이어를 정한다.
        - 차선: `SemanticMapLayer.LANE`
        - 차선연결(교차로 내 짧은 연결 도로 등): `SemanticMapLayer.LANE_CONNECTOR`

    2) `get_directional_proximal_map_objects` 를 이용해,
       ego 앞쪽 방향으로 회전된 정사각형(한 변 길이 = 2 * radius) 안에
       걸쳐 있는 차선/차선연결 객체들을 가져온다.
        - 반환 값 예시:
          · `layers[SemanticMapLayer.LANE]`           → Lane 객체 리스트
          · `layers[SemanticMapLayer.LANE_CONNECTOR]` → LaneConnector 객체 리스트

    3) 두 레이어에서 가져온 객체들을 하나의 리스트로 합친 뒤,
       ego 위치와의 **유클리드 거리(직선 거리)** 를 기준으로
       가까운 순서대로 정렬한다.

    Returns:
        List[MapObject]:
            ego 주변에서 찾은 차선·차선연결 객체를
            ego와의 거리 기준으로 가까운 순서대로 정렬한 리스트.
    """
    layer_names: List[SemanticMapLayer] = [
        SemanticMapLayer.LANE,
        SemanticMapLayer.LANE_CONNECTOR,
    ]

    # layers: Dict[SemanticMapLayer, List[MapObject]]
    layers = get_circular_proximal_map_objects(
        map_api=map_api,
        point=ego_point_2d,
        # heading=ego_heading,
        radius=radius,
        layers=layer_names,
    )

    # map_objects 길이 = num_lanes
    map_objects: List[MapObject] = []
    for layer_name in layer_names:
        map_objects += layers[layer_name]

    # ego 위치와의 거리 기준으로 가까운 순으로 정렬
    map_objects.sort(key=lambda map_obj: float(
        get_distance_between_map_object_and_point(ego_point_2d, map_obj)))

    return map_objects


def _extract_lane_polyline_and_metadata(
    map_objects: List[MapObject],) -> Tuple[
        List[List[Point2D]],  # lanes_mid
        List[List[Point2D]],  # lanes_left
        List[List[Point2D]],  # lanes_right
        List[str],  # lane_ids
        List[float],  # lane_speed_limit
        List[bool],  # lane_has_speed_limit
        List[str],  # lane_roadblock_ids
    ]:
    """차선/차선연결 객체 리스트에서 선 정보와 부가 정보를 뽑아낸다.

    동작 내용:
      - 각 객체에 대해
        · 중심을 따라 그려진 선(중심선)의 점들을 모은다.
        · 왼쪽 경계선 점들을 모은다.
        · 오른쪽 경계선 점들을 모은다.
        · 차선에 붙은 문자열 꼬리표(id)를 모은다.
        · 속도 제한 값, 속도 제한이 있는지 여부를 모은다.
        · 이 차선이 속한 도로 묶음 id 를 모은다.

    자료 구조 / 모양:
      - lanes_mid / lanes_left / lanes_right:
        · 타입: List[List[Point2D]]
        · 개념적 모양: [num_lanes, num_points_i]
          (각 차선마다 점 개수가 다를 수 있음)
      - lane_ids:
        · 길이: num_lanes
      - lane_speed_limit:
        · 길이: num_lanes
        · 각 값: float, 단위 m/s, 제한이 없으면 0.0
      - lane_has_speed_limit:
        · 길이: num_lanes
        · 각 값: bool, True면 실제 제한 속도 있음
      - lane_roadblock_ids:
        · 길이: num_lanes
        · 각 값: 문자열 id

    Args:
        map_objects (List[MapObject]):
            ego 주변에서 수집된 차선/차선연결 객체 리스트.
            길이 = num_lanes.

    Returns:
        Tuple[...]:
            lanes_mid, lanes_left, lanes_right, lane_ids,
            lane_speed_limit, lane_has_speed_limit, lane_roadblock_ids
    """
    # lanes_*: List[List[Point2D]]  — [num_lanes, num_points_i]
    lanes_mid: List[List[Point2D]] = []
    lanes_left: List[List[Point2D]] = []
    lanes_right: List[List[Point2D]] = []

    # lane_ids: 길이 = num_lanes
    lane_ids: List[str] = []

    # lane_speed_limit: 길이 = num_lanes, 각 값 float (m/s)
    lane_speed_limit: List[float] = []
    # lane_has_speed_limit: 길이 = num_lanes, bool
    lane_has_speed_limit: List[bool] = []
    # lane_roadblock_ids: 길이 = num_lanes, str
    lane_roadblock_ids: List[str] = []

    for map_obj in map_objects:
        # 중심을 따라 그려진 선(중심선)
        baseline_path_polyline: List[Point2D] = [
            Point2D(node.x, node.y)
            for node in map_obj.baseline_path.discrete_path
        ]
        lanes_mid.append(baseline_path_polyline)

        # 왼쪽 경계선
        lanes_left.append([
            Point2D(node.x, node.y)
            for node in map_obj.left_boundary.discrete_path
        ])

        # 오른쪽 경계선
        lanes_right.append([
            Point2D(node.x, node.y)
            for node in map_obj.right_boundary.discrete_path
        ])

        # 차선 id (문자열 꼬리표)
        lane_ids.append(map_obj.id)

        # 속도 제한 값 / 존재 여부
        if map_obj.speed_limit_mps is None:
            lane_speed_limit.append(0.0)
            lane_has_speed_limit.append(False)
        else:
            lane_speed_limit.append(float(map_obj.speed_limit_mps))
            lane_has_speed_limit.append(True)

        # 이 차선이 속한 도로 묶음 id
        lane_roadblock_ids.append(map_obj.get_roadblock_id())

    return (
        lanes_mid,
        lanes_left,
        lanes_right,
        lane_ids,
        lane_speed_limit,
        lane_has_speed_limit,
        lane_roadblock_ids,
    )


def _get_lane_polylines(
    map_api: AbstractMap,
    ego_point_2d: Point2D,
    ego_heading: float,
    radius: float,
) -> Tuple[
        MapObjectPolylines,  # lanes_mid
        MapObjectPolylines,  # lanes_left
        MapObjectPolylines,  # lanes_right
        LaneSegmentLaneIDs,  # lane_ids
        List[float],  # lane_speed_limit
        List[bool],  # lane_has_speed_limit
        List[str],  # lane_roadblock_ids
]:
    """ego 주변의 차선 중심선·경계선·속도제한·도로묶음 ID를 한 번에 추출한다.

    이 함수는 “ego 주변에 어떤 차선들이 있고, 그 차선의 모양과 속도 제한 정보는
    무엇인지”를 한 번에 얻기 위한 편의 함수이다.

    동작 순서
    --------
    1) `_collect_lane_map_objects` 호출
        - ego 위치와 진행 방향, 반경을 기준으로,
          앞쪽 위주의 차선 / 차선연결 객체들을 가져온다.
        - 거리 기준으로 가까운 순서로 정렬된 `map_objects` 리스트를 얻는다.

    2) `_extract_lane_polyline_and_metadata` 호출
        - 각 `map_obj` 에 대해 다음 정보를 뽑아 리스트로 모은다.
            · 중심선: 차선 중앙을 따라 이어진 점들 (Point2D 리스트)
            · 왼쪽 경계선: 차선 좌측 테두리 점들
            · 오른쪽 경계선: 차선 우측 테두리 점들
            · 차선 ID: 문자열 꼬리표
            · 속도 제한 값: m/s, 없으면 0.0
            · 속도 제한 존재 여부: bool
            · 도로 묶음 ID(roadblock id)

    3) nuPlan에서 사용하는 래퍼 타입으로 감싸기
        - `MapObjectPolylines(lanes_mid_list)`  → 중심선 모음
        - `MapObjectPolylines(lanes_left_list)` → 왼쪽 경계선 모음
        - `MapObjectPolylines(lanes_right_list)`→ 오른쪽 경계선 모음
        - `LaneSegmentLaneIDs(lane_ids_list)`   → 차선 ID 모음

    자료 구조 / 모양(개념)
    ----------------------
    - 중심선/경계선 (반환 값 1~3번째):
        MapObjectPolylines 내부:
            · polylines: List[List[Point2D]]
            · 각 내부 리스트: 한 차선, 길이는 해당 차선의 점 개수
            · (x, y) 좌표 형태이므로, 넘파이로 바꾸면
              (num_lanes, num_points_i, 2) 모양이 된다.
    Returns:
            1. lanes_mid:
                - 타입: MapObjectPolylines
                - 의미: 각 차선의 중심선 좌표 모음
            2. lanes_left:
                - 타입: MapObjectPolylines
                - 의미: 각 차선의 왼쪽 경계선 좌표 모음
            3. lanes_right:
                - 타입: MapObjectPolylines
                - 의미: 각 차선의 오른쪽 경계선 좌표 모음
            4. lane_ids:
                - 타입: LaneSegmentLaneIDs
                - 의미: 각 차선의 ID 문자열 리스트
            5. lane_speed_limit:
                - 타입: List[float]
                - 의미: 각 차선의 속도 제한 값(m/s), 없으면 0.0
            6. lane_has_speed_limit:
                - 타입: List[bool]
                - 의미: 해당 차선이 실제 속도 제한을 가지고 있는지 여부
            7. lane_roadblock_ids:
                - 타입: List[str]
                - 의미: 각 차선이 속한 도로 묶음(roadblock) ID
    """
    # 1) ego 주변 차선 / 차선연결 객체 수집 및 정렬
    # map_objects: 거리순임
    map_objects: List[MapObject] = _collect_lane_map_objects(
        map_api=map_api,
        ego_point_2d=ego_point_2d,
        ego_heading=ego_heading,
        radius=radius,
    )

    # 2) 각 객체에서 선/경계/속도/도로 묶음 정보 추출
    (
        lanes_mid_list,  # List[List[Point2D]]:  [num_lanes, num_points_i]
        lanes_left_list,  # List[List[Point2D]]:  [num_lanes, num_points_i]
        lanes_right_list,  # List[List[Point2D]]:  [num_lanes, num_points_i]
        lane_ids_list,
        lane_speed_limit,
        lane_has_speed_limit,
        lane_roadblock_ids,
    ) = _extract_lane_polyline_and_metadata(map_objects)

    # 3) nuplan 유틸과 호환되는 래퍼 타입으로 감싸서 반환
    lanes_mid = MapObjectPolylines(lanes_mid_list)
    lanes_left = MapObjectPolylines(lanes_left_list)
    lanes_right = MapObjectPolylines(lanes_right_list)
    lane_ids = LaneSegmentLaneIDs(lane_ids_list)

    return (
        lanes_mid,
        lanes_left,
        lanes_right,
        lane_ids,
        lane_speed_limit,
        lane_has_speed_limit,
        lane_roadblock_ids,
    )


def _parse_vector_feature_layers(
    map_elements: List[str],) -> List[VectorFeatureLayer]:
    """문자열로 들어온 맵 요소 이름들을 내부 Enum 형태로 바꿔준다.

    예:
        - "LANE"          -> VectorFeatureLayer.LANE
        - "LEFT_BOUNDARY" -> VectorFeatureLayer.LEFT_BOUNDARY
        - ...

    Args:
        map_elements (List[str]):
            사용할 맵 요소 이름 리스트.

    Returns:
        List[VectorFeatureLayer]:
            변환된 Enum 리스트.

    Raises:
        ValueError: 지원하지 않는 이름이 들어온 경우.
    """
    feature_layers: List[VectorFeatureLayer] = []
    for feature_name in map_elements:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"
            )
    return feature_layers


def _fill_lane_related_features(
    map_api: AbstractMap,
    ego_point_2d: Point2D,
    ego_heading: float,
    radius: float,
    feature_layers: List[VectorFeatureLayer],
    traffic_light_status_data: List[TrafficLightStatusData],
    elements_to_obj_polylines: Dict[str, MapObjectPolylines],
    elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
    speed_limit_dict: Dict[str, np.ndarray],
) -> List[str]:
    """차선과 직접 관련된 정보들을 채운다.

    이 함수는 다음과 같은 일을 한 번에 처리한다.

      1) `_get_lane_polylines`를 호출하여,
         - 중심선 / 왼쪽 경계선 / 오른쪽 경계선
         - 차선 id
         - 속도 제한 값 / 여부
         - 차선이 속한 도로 묶음 id
         를 가져온다.

      2) 반환된 정보를 아래와 같이 나누어 저장한다.
         - elements_to_obj_polylines["LANE"]           : 중심선
         - elements_to_obj_polylines["LEFT_BOUNDARY"]  : 왼쪽 경계선 (옵션)
         - elements_to_obj_polylines["RIGHT_BOUNDARY"] : 오른쪽 경계선 (옵션)
         - speed_limit_dict["lane_has_speed_limit"]: (num_lanes,) bool 배열
         - speed_limit_dict["lane_speed_limit"]    : (num_lanes,) float32 배열
         - elements_to_traffic_light["LANE"]         : 차선별 신호등 상태(one-hot)

      3) 마지막으로 각 차선이 어떤 도로 묶음에 속하는지 나타내는
         문자열 리스트(roadblock id 리스트)를 그대로 반환한다.

    자료 구조 / 모양:
      - elements_to_obj_polylines["LANE"]:
          MapObjectPolylines
          · 내부적으로 [num_lanes, num_points_i, 2] 구조의 점 리스트
      - speed_limit_dict["lane_has_speed_limit"]:
          np.ndarray, shape = (num_lanes,), dtype = bool
      - speed_limit_dict["lane_speed_limit"]:
          np.ndarray, shape = (num_lanes,), dtype = float32
      - elements_to_traffic_light["LANE"]:
          LaneSegmentTrafficLightData
          · 내부 one-hot 모양: (num_lanes, 4)
            [초록, 노랑, 빨강, 알 수 없음]

    Args:
        map_api (AbstractMap):
            전체 지도를 다루는 객체.
        ego_point_2d (Point2D):
            ego 위치 (x, y), 단위 m.
        ego_heading (float):
            ego 진행 방향(라디안).
        radius (float):
            ego 주변에서 차선을 찾을 거리 범위. 단위 m.
        feature_layers (List[VectorFeatureLayer]):
            사용할 맵 요소 Enum 리스트.
        traffic_light_status_data (List[TrafficLightStatusData]):
            현재 시점에 사용할 신호등 상태 리스트.
        elements_to_obj_polylines (Dict[str, MapObjectPolylines]):
            맵 요소별 선 정보를 담는 딕셔너리(출력용, in-place 수정).
        elements_to_traffic_light (Dict[str, LaneSegmentTrafficLightData]):
            맵 요소별 신호등 정보를 담는 딕셔너리(출력용, in-place 수정).
        speed_limit_dict (Dict[str, np.ndarray]):
            차선별 속도제한 정보를 담는 딕셔너리(출력용, in-place 수정).

    Returns:
        List[str]:
            lanes_roadblock_id_list:
                각 차선이 속한 도로 묶음(roadblock)의 id 리스트.
                길이 = num_lanes.

 Note:
        elements_to_obj_polylines / elements_to_traffic_light / speed_limit_dict 는
        인자로 받은 딕셔너리를 함수 내부에서 그대로 수정(in-place)하므로,
        별도로 return 하지 않아도 호출 측에서 변경된 내용을 바로 사용할 수 있습니다.
        즉, 이 함수가 직접 반환하는 값은 새로 만들어지는 리스트인 lanes_roadblock_id_list 하나뿐이고,
        나머지 부가 정보는 모두 참조를 통해 바깥 딕셔너리에 채워집니다.
    """

    (
        lanes_mid,  # MapObjectPolylines
        lanes_left,  # MapObjectPolylines
        lanes_right,  # MapObjectPolylines
        lane_ids,  # LaneSegmentLaneIDs
        lane_speed_limit,  # List[float]
        lane_has_speed_limit,  # List[bool]
        lanes_roadblock_id_list,  # List[str]
    ) = _get_lane_polylines(
        map_api=map_api,
        ego_point_2d=ego_point_2d,
        ego_heading=ego_heading,
        radius=radius,
    )

    # lane 중심선
    elements_to_obj_polylines[VectorFeatureLayer.LANE.name] = lanes_mid

    # 속도 제한 정보: (num_lanes,)
    speed_limit_dict["lane_has_speed_limit"] = np.array(
        lane_has_speed_limit, dtype=np.bool_)  # shape: (num_lanes,)
    speed_limit_dict["lane_speed_limit"] = np.array(
        lane_speed_limit, dtype=np.float32)  # shape: (num_lanes,)

    # 차선별 신호등 상태(one-hot)
    # LaneSegmentTrafficLightData
    elements_to_traffic_light[VectorFeatureLayer.LANE.name] = \
        get_traffic_light_encoding(lane_ids, traffic_light_status_data)

    # 왼쪽/오른쪽 경계선은 옵션
    if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
        elements_to_obj_polylines[
            VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines)
    if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
        elements_to_obj_polylines[
            VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(
                lanes_right.polylines)

    return lanes_roadblock_id_list


def _fill_polygon_feature_layers(
    map_api: AbstractMap,
    ego_point_2d: Point2D,
    radius: float,
    feature_layers: List[VectorFeatureLayer],
    elements_to_obj_polylines: Dict[str, MapObjectPolylines],
) -> None:
    """정지선, 횡단보도 등 “면 형태”의 물체들을 elements_to_obj_polylines 딕셔너리에 채운다.

    이 함수는 차선 이외의 객체들(정지선, 횡단보도 등)을
    ego 주변에서 찾아서, 각 객체를 이루는 점 목록으로 저장한다.

    자료 구조 / 모양:
      - elements_to_obj_polylines[feature_layer.name]:
          MapObjectPolylines
          · 내부 리스트 모양: [num_objects, num_points_i, 2]

    Args:
        map_api (AbstractMap):
            전체 지도를 다루는 객체.
        ego_point_2d (Point2D):
            ego 위치 (x, y), 단위 m.
        radius (float):
            ego 주변에서 물체를 찾을 거리 범위. 단위 m.
        feature_layers (List[VectorFeatureLayer]):
            사용할 맵 요소 Enum 리스트.
        elements_to_obj_polylines (Dict[str, MapObjectPolylines]):
            맵 요소별 선/면 정보를 담는 딕셔너리(출력용, in-place 수정).
    """
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers(
        ):
            polygons = get_map_object_polygons(
                map_api=map_api,
                point=ego_point_2d,
                radius=radius,
                layer_name=VectorFeatureLayerMapping.semantic_map_layer(
                    feature_layer),
            )
            elements_to_obj_polylines[feature_layer.name] = polygons


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_elements: List[str],
    ego_point_2d: Point2D,
    ego_heading: float,
    radius: float,
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[
        Dict[str, MapObjectPolylines],  # elements_to_obj_polylines
        Dict[str, LaneSegmentTrafficLightData],  # elements_to_traffic_light
        Dict[str, np.ndarray],  # speed_limit_dict
        List[str],  # lanes_roadblock_id_list
]:
    """ego 주변의 도로 모양, 신호등, 속도제한 정보를 한 번에 뽑아낸다.

    한 줄로 요약하면,
    **“ego 주변 일정 거리 안에 있는, 도로 선과 신호 정보들을
    사용하기 편한 자료 구조로 모아서 돌려주는 함수”**

    동작 순서:
      1) `map_elements` 문자열 리스트를 내부 Enum 형태로 바꾼다.
         - 예: ["LANE", "LEFT_BOUNDARY"] → [VectorFeatureLayer.LANE, ...]
      2) 차선 정보가 필요한 경우:
         - `_get_lane_polylines` 로
           · 중심선 / 왼쪽 경계 / 오른쪽 경계
           · 차선 ID
           · 속도제한 값 / 여부
           · 각 차선이 속한 도로 묶음 ID
           를 가져온다.
         - 이 정보를
           · elements_to_obj_polylines["LANE"], elements_to_obj_polylines["LEFT_BOUNDARY"], elements_to_obj_polylines["RIGHT_BOUNDARY"]
           · speed_limit_dict["lane_has_speed_limit"], speed_limit_dict["lane_speed_limit"]
           · elements_to_traffic_light["LANE"]
           에 나누어 담는다.
      3) 정지선/횡단보도 등 polygon 형태의 물체가 필요하면,
         `_fill_polygon_feature_layers` 를 통해 elements_to_obj_polylines 에 추가한다.

    Args:
        map_api (AbstractMap):
            전체 지도를 다루는 객체.
        map_elements (List[str]):
            추출하고 싶은 맵 요소 이름 리스트.
            예: ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']
        ego_point_2d (Point2D):
            ego 위치 (x, y), 단위 m.
        ego_heading (float):
            ego 진행 방향(라디안). 차선 검색 방향에 사용.
        radius (float):
            ego 주변에서 맵 요소를 찾을 거리 범위. 단위 m.
        traffic_light_status_data (List[TrafficLightStatusData]):
            현재 시점의 신호등 상태 리스트.

    Returns:
    1. elements_to_obj_polylines: Dict[str, MapObjectPolylines],
       - 키: 맵 요소 이름 문자열 "LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "CROSSWALK", ...
       - 값: 해당 요소를 이루는 점들의 모음(MapObjectPolylines)
    - 내부 구조: [num_elements, num_points_i, 2]
    2. elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
       - 키: 맵 요소 이름 문자열(현재 "LANE"만 사용)
       - 값: 해당 요소에 대응되는 신호등 상태 정보 (LaneSegmentTrafficLightData)
            - 내부 구조: (num_lanes, 4) one-hot
    3. speed_limit_dict: Dict[str, np.ndarray],
       - "lane_has_speed_limit": (num_lanes,), bool
       - "lane_speed_limit": (num_lanes,), float32
    4. lanes_roadblock_id_list: List[str],
       - 각 차선이 속한 도로 묶음(roadblock) ID 리스트 (길이 = num_lanes)
    """
    # elements_to_obj_polylines: 각 맵 요소 이름 → 선/면 정보
    elements_to_obj_polylines: Dict[str, MapObjectPolylines] = {}
    # elements_to_traffic_light: 각 맵 요소 이름 → 신호등 정보
    elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData] = {}
    # speed_limit_dict: 차선 관련 속도 제한 정보
    speed_limit_dict: Dict[str, np.ndarray] = {}

    # 1) 문자열 feature 이름을 Enum 으로 변환
    feature_layers: List[VectorFeatureLayer] = _parse_vector_feature_layers(
        map_elements=map_elements)

    # 2) 차선 관련 정보 채우기
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_roadblock_id_list: List[str] = _fill_lane_related_features(
            map_api=map_api,
            ego_point_2d=ego_point_2d,
            ego_heading=ego_heading,
            radius=radius,
            feature_layers=feature_layers,
            traffic_light_status_data=traffic_light_status_data,
            elements_to_obj_polylines=elements_to_obj_polylines,
            elements_to_traffic_light=elements_to_traffic_light,
            speed_limit_dict=speed_limit_dict,
        )
    # (원본 코드와 동일하게, LANE 이 feature_layers 에 없으면 lanes_roadblock_id_list 변수가
    #  정의되지 않은 상태로 종료 시 에러가 나게 되어 있음. 실제 사용처에서는
    #  항상 LANE 을 포함하므로 동작은 동일하다.)

    # 3) 정지선/횡단보도 등 polygon 계열 요소 채우기
    _fill_polygon_feature_layers(
        map_api=map_api,
        ego_point_2d=ego_point_2d,
        radius=radius,
        feature_layers=feature_layers,
        elements_to_obj_polylines=elements_to_obj_polylines,
    )

    return elements_to_obj_polylines, elements_to_traffic_light, speed_limit_dict, lanes_roadblock_id_list


# =====================
# 2. Get maps array for model input
# =====================
def _interpolate_points(line, num_point):
    line = LineString(line)
    new_line = np.concatenate([
        line.interpolate(d).coords._coords
        for d in np.linspace(0, line.length, num_point)
    ])

    return new_line


def _select_lanes_by_ego_distance_to_keep(
    ego_pose: np.ndarray,  # (3,) = [ego_x, ego_y, ego_heading]
    feature_coords: List[
        np.ndarray],  # 길이 = chosen_lane_num, 각 원소: (num_points_i, 2)
    map_max_elements: int,
) -> Tuple[List[int], int]:
    """ego 와의 거리 기준으로 lane 인덱스를 정렬하고, 유지할 인덱스만 골라낸다.

    이 함수는 다음 두 가지를 한다.
    1) 각 차선 중심선의 모든 점들 중에서 ego 와의 **최소 거리**를 구한다.
    2) 거리가 가까운 순으로 정렬한 뒤,
       - lane 개수가 map_max_elements 보다 크면 → 앞에서 map_max_elements 개만 선택
       - lane 개수가 map_max_elements 이하면 → 전부 선택

    Args:
        ego_pose (np.ndarray):
            ego 현재 상태 [x, y, heading]. shape = (3,).
        feature_coords (List[np.ndarray]):
            각 차선 중심선 좌표 리스트.
            - 길이: chosen_lane_num
            - 각 원소 shape: (num_points_i, 2)
        map_max_elements (int):
            유지할 수 있는 lane 의 최대 개수.

    Returns:
        Tuple[List[int], int]:
            - selected_lane_indices:
                거리 기준으로 정렬된 후, 실제로 사용할 lane 인덱스들.
                길이 = chosen_lane_num.
            - chosen_lane_num:
                실제로 사용할 lane 개수.
                = min(chosen_lane_num, map_max_elements)
    """
    chosen_lane_num: int = len(feature_coords)

    if chosen_lane_num == 0 or map_max_elements <= 0:
        return [], 0

    # 각 lane 과 ego 사이의 최소 거리 계산
    distance_map: Dict[int, float] = {}
    for lane_index, lane_coords in enumerate(feature_coords):
        # lane_coords: (num_points_i, 2)
        dist: float = float(
            np.linalg.norm(lane_coords - ego_pose[None, :2], axis=-1).min())
        distance_map[lane_index] = dist

    # 거리 기준 오름차순 정렬
    sorted_items: List[Tuple[int, float]] = sorted(distance_map.items(),
                                                   key=lambda item: item[1])

    chosen_lane_num: int = min(chosen_lane_num, map_max_elements)
    selected_lane_indices: List[int] = [
        lane_index for lane_index, _ in sorted_items[:chosen_lane_num]
    ]

    return selected_lane_indices, chosen_lane_num


def _initialize_lane_geometry_arrays(
    chosen_lane_num: int,
    map_points_num: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """lane 중심선/왼쪽/오른쪽 경계선 좌표를 담을 배열을 만든다.

    모든 배열은 아직 0으로만 채워져 있으며,
    이후에 실제 lane 데이터가 채워진다.

    Args:
        chosen_lane_num (int):
            실제로 사용할 lane 개수.
        map_points_num (int):
            각 lane 이 가지게 될 고정 포인트 수.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - chosen_center_xy:
                lane 중심선 좌표.
                shape: (chosen_lane_num, map_points_num, 2), dtype: float64
            - left_array:
                lane 왼쪽 경계선 좌표.
                shape: (chosen_lane_num, map_points_num, 2), dtype: float64
            - right_array:
                lane 오른쪽 경계선 좌표.
                shape: (chosen_lane_num, map_points_num, 2), dtype: float64
    """
    chosen_center_xy: np.ndarray = np.zeros(
        (chosen_lane_num, map_points_num, 2), dtype=np.float64)
    left_array: np.ndarray = np.zeros((chosen_lane_num, map_points_num, 2),
                                      dtype=np.float64)
    right_array: np.ndarray = np.zeros((chosen_lane_num, map_points_num, 2),
                                       dtype=np.float64)

    return chosen_center_xy, left_array, right_array


def _initialize_lane_attribute_arrays(
    chosen_lane_num: int,
    map_points_num: int,
    traffic_light_encoding_dim: int,
    has_traffic_light: bool,
) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Optional[np.ndarray],
]:
    """lane 의 부가 정보(available, 속도제한, 신호)를 담을 배열을 만든다.

    Args:
        chosen_lane_num (int):
            실제로 사용할 lane 개수.
        map_points_num (int):
            각 lane 이 가지게 될 고정 포인트 수.
        traffic_light_encoding_dim (int):
            신호등 one-hot 벡터 길이 (예: 4).
        has_traffic_light (bool):
            신호등 정보(feature_tl_data)가 실제로 존재하는지 여부.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
            - lane_xy_valid_mask:
                실제 포인트인지, 비어 있는 자리인지 표시.
                shape: (chosen_lane_num, map_points_num), dtype: bool
            - lane_has_speed_limit_array:
                lane 별로 속도제한이 있는지 여부.
                shape: (chosen_lane_num, 1), dtype: bool
            - lane_speed_limit_array:
                lane 별 속도제한 값(m/s).
                shape: (chosen_lane_num, 1), dtype: float32
            - lanes_point_tl_array:
                포인트별 신호 상태(one-hot) 배열 또는 None.
                shape: (chosen_lane_num, map_points_num, traffic_light_encoding_dim),
                dtype: float32
    """
    lane_xy_valid_mask: np.ndarray = np.zeros((chosen_lane_num, map_points_num),
                                              dtype=np.bool_)
    lane_has_speed_limit_array: np.ndarray = np.zeros((chosen_lane_num, 1),
                                                      dtype=np.bool_)
    lane_speed_limit_array: np.ndarray = np.zeros((chosen_lane_num, 1),
                                                  dtype=np.float32)

    lanes_point_tl_array: Optional[np.ndarray]
    if has_traffic_light:
        lanes_point_tl_array = np.zeros(
            (chosen_lane_num, map_points_num, traffic_light_encoding_dim),
            dtype=np.float32,
        )
    else:
        lanes_point_tl_array = None

    return (
        lane_xy_valid_mask,
        lane_has_speed_limit_array,
        lane_speed_limit_array,
        lanes_point_tl_array,
    )


def _fill_lane_arrays_for_selected_lanes(
        selected_lane_indices: List[int],  # 길이 = chosen_lane_num
        feature_coords: List[
            np.ndarray],  # 길이 = chosen_lane_num, 각 원소: (num_points_i, 2)
        left_boundary: List[
            np.ndarray],  # 길이 = chosen_lane_num, 각 원소: (num_points_i, 2)
        right_boundary: List[
            np.ndarray],  # 길이 = chosen_lane_num, 각 원소: (num_points_i, 2)
        lanes_roadblock_id_list: List[str],  # 길이 = chosen_lane_num
        lane_has_speed_limit: np.ndarray,  # (chosen_lane_num,)
        lane_speed_limit: np.ndarray,  # (chosen_lane_num,)
        feature_tl_data: Optional[List[
            np.
            ndarray]],  # 길이 = chosen_lane_num, 각 원소: (num_points_i, dim) 또는 None
        map_points_num: int,
        chosen_center_xy: np.ndarray,  # (chosen_lane_num, map_points_num, 2)
        left_array: np.ndarray,  # (chosen_lane_num, map_points_num, 2)
        right_array: np.ndarray,  # (chosen_lane_num, map_points_num, 2)
        lane_xy_valid_mask: np.ndarray,  # (chosen_lane_num, map_points_num)
        lane_has_speed_limit_array: np.ndarray,  # (chosen_lane_num, 1)
        lane_speed_limit_array: np.ndarray,  # (chosen_lane_num, 1)
        lanes_point_tl_array: Optional[
            np.ndarray],  # (chosen_lane_num, map_points_num, dim) 또는 None
) -> List[str]:
    """선택된 lane 인덱스들에 대해, 실제 좌표/속도제한/신호 데이터를 배열에 채워 넣는다.

    이 함수는 이미 크기가 정해져 있는 출력 배열들에 대해
    한 줄(한 개의 lane)씩 아래 내용을 채워 넣는다.

    - 중심선 좌표 (chosen_center_xy)
    - 왼쪽/오른쪽 경계선 좌표 (left_array / right_array)
    - 해당 위치가 실제 포인트인지 여부 (lane_xy_valid_mask)
    - 속도제한 유무/값 (lane_has_speed_limit_array / lane_speed_limit_array)
    - roadblock id (chosen_lanes_rb_id_list 리스트에 문자열로 append)
    - 신호등 정보(feature_tl_data가 있을 때만 tl_data_array에 복사)

    Args:
        selected_lane_indices (List[int]):
            ego 와 가까운 순으로 정렬된 뒤, 실제로 사용할 lane 인덱스 리스트.
        feature_coords (List[np.ndarray]):
            각 lane 중심선 좌표 리스트. 각 원소 shape: (num_points_i, 2).
        left_boundary / right_boundary (List[np.ndarray]):
            각 lane 의 왼쪽/오른쪽 경계선 좌표 리스트.
            각 원소 shape: (num_points_i, 2).
        lanes_roadblock_id_list (List[str]):
            각 lane 이 속한 roadblock id 리스트.
        lane_has_speed_limit, lane_speed_limit (np.ndarray):
            lane 전체에 대한 속도제한 유무/값. shape: (chosen_lane_num,).
        feature_tl_data (Optional[List[np.ndarray]]):
            lane 별 신호 상태 리스트. 각 원소 shape: (num_points_i, dim) 또는 None.
        map_points_num (int):
            lane 당 고정 포인트 수.
        chosen_center_xy, left_array, right_array, lane_xy_valid_mask,
        lane_has_speed_limit_array, lane_speed_limit_array, lanes_point_tl_array:
            이미 크기가 만들어진 출력 배열들.

    Returns:
        List[str]:
            chosen_lanes_rb_id_list: 선택된 lane 들의 roadblock id 리스트.
                         길이 = # len: chosen_lane_num
    """
    chosen_lanes_rb_id_list: List[str] = []

    for out_idx, src_idx in enumerate(selected_lane_indices):
        element_coords: np.ndarray = feature_coords[
            src_idx]  # (num_points_i, 2)
        left_coords: np.ndarray = left_boundary[src_idx]  # (num_points_i, 2)
        right_coords: np.ndarray = right_boundary[src_idx]  # (num_points_i, 2)

        # 포인트 수를 map_points_num 으로 맞추기 (보간/자르기)
        element_coords = _interpolate_points(element_coords, map_points_num)
        left_coords = _interpolate_points(left_coords, map_points_num)
        right_coords = _interpolate_points(right_coords, map_points_num)

        # 좌표/마스크 채우기
        chosen_center_xy[out_idx] = element_coords  # (map_points_num, 2)
        left_array[out_idx] = left_coords  # (map_points_num, 2)
        right_array[out_idx] = right_coords  # (map_points_num, 2)
        lane_xy_valid_mask[out_idx] = True  # (map_points_num,)

        # 속도제한 / roadblock id 채우기
        lane_has_speed_limit_array[out_idx,
                                   0] = bool(lane_has_speed_limit[src_idx])
        lane_speed_limit_array[out_idx, 0] = float(lane_speed_limit[src_idx])
        chosen_lanes_rb_id_list.append(lanes_roadblock_id_list[src_idx])

        # 신호등 데이터도 있으면 그대로 복사
        if lanes_point_tl_array is not None and feature_tl_data is not None:
            # feature_tl_data[src_idx] 의 shape 은
            # (map_points_num, traffic_light_encoding_dim) 이라고 가정
            lanes_point_tl_array[out_idx] = feature_tl_data[src_idx]

    return chosen_lanes_rb_id_list  # len: chosen_lane_num


def _prune_route_by_connectivity(route_roadblock_ids: List[str],
                                 roadblock_ids: Set[str]) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """
    pruned_route_roadblock_ids: List[str] = []
    route_start = False  # wait for route to come into query radius before declaring broken connection

    for roadblock_id in route_roadblock_ids:

        if roadblock_id in roadblock_ids:
            pruned_route_roadblock_ids.append(roadblock_id)
            route_start = True

        elif route_start:  # connection broken
            break

    return pruned_route_roadblock_ids


def _lane_polyline_process(
    polylines: np.ndarray,  # shape: (chosen_lane_num, lane_len, 2)
    left_boundary: np.ndarray,  # shape: (chosen_lane_num, lane_len, 2)
    right_boundary: np.ndarray,  # shape: (chosen_lane_num, lane_len, 2)
    lane_xy_valid_mask: np.ndarray,  # shape: (chosen_lane_num, lane_len)
    traffic_light: np.ndarray,  # shape: (chosen_lane_num, lane_len, 4)
) -> np.ndarray:  # shape: (chosen_lane_num, lane_len, 12)
    """차선 폴리라인과 좌우 경계선, 신호 정보를 한 번에 벡터로 합친다.

    각 차선의 한 점에 대해 다음 값을 이어 붙여 길이 12 벡터를 만든다.

    - 현재 점 위치 (x, y)
    - 이전 점과의 차이 벡터 (간단한 진행 방향 정보)
    - 왼쪽 경계선까지의 상대 위치 (left - center)
    - 오른쪽 경계선까지의 상대 위치 (right - center)
    - 신호등 one-hot (길이 4)

    Args:
        polylines: 차선 중심선 좌표.
            shape = (chosen_lane_num, lane_len, 2).
        left_boundary: 왼쪽 경계선 좌표.
            shape = (chosen_lane_num, lane_len, 2).
        right_boundary: 오른쪽 경계선 좌표.
            shape = (chosen_lane_num, lane_len, 2).
        lane_xy_valid_mask: 유효 포인트 마스크.
            shape = (chosen_lane_num, lane_len).
            lane_xy_valid_mask[i, 0] 이 False 이면 i 번째 차선 전체를 패딩으로 간주한다.
        traffic_light: 신호등 상태 one-hot.
            shape = (chosen_lane_num, lane_len, 4).

    Returns:
        np.ndarray: 차선 벡터 표현.
            shape = (chosen_lane_num, lane_len, 12), dtype = float32.
    """
    dim: int = 12  # 각 포인트의 특성 길이
    chosen_lane_num: int = polylines.shape[0]
    lane_len: int = polylines.shape[1]
    # new_polylines: (chosen_lane_num, lane_len, 12)
    new_polylines: np.ndarray = np.zeros(
        shape=(chosen_lane_num, lane_len, dim),
        dtype=np.float32,
    )
    for lane_i in range(chosen_lane_num):
        # 이 lane 이 전부 패딩이면 건너뜀
        if not bool(lane_xy_valid_mask[lane_i][0]):
            continue

        # polyline: (lane_len, 2)
        polyline: np.ndarray = polylines[lane_i]

        # 이웃 점 차이 벡터: (lane_len, 2)
        polyline_vector: np.ndarray = polyline[
            1:] - polyline[:-1]  # (lane_len-1, 2)
        polyline_vector = np.insert(
            polyline_vector,
            lane_len - 1,
            0.0,
            axis=0,
        )  # (lane_len, 2), 마지막 한 점에 0 벡터 추가

        # 좌우 경계선 방향이 중심선 시작점과 가깝도록 정렬
        # left_boundary[lane_i]: (lane_len, 2)
        if np.linalg.norm(left_boundary[lane_i, -1] -
                          polyline[0]) < np.linalg.norm(left_boundary[lane_i,
                                                                      0] -
                                                        polyline[0]):
            left_boundary[lane_i] = np.flip(left_boundary[lane_i], axis=0)

        # right_boundary[lane_i]: (lane_len, 2)
        if np.linalg.norm(right_boundary[lane_i, -1] -
                          polyline[0]) < np.linalg.norm(right_boundary[lane_i,
                                                                       0] -
                                                        polyline[0]):
            right_boundary[lane_i] = np.flip(right_boundary[lane_i], axis=0)

        # polyline_to_left/right: (lane_len, 2)
        polyline_to_left: np.ndarray = left_boundary[lane_i] - polyline
        polyline_to_right: np.ndarray = right_boundary[lane_i] - polyline

        # traffic_light[lane_i]: (lane_len, 4)
        # concat 결과: (lane_len, 2+2+2+2+4=12)
        new_polylines[lane_i] = np.concatenate(
            [
                polyline,
                polyline_vector,
                polyline_to_left,
                polyline_to_right,
                traffic_light[lane_i],
            ],
            axis=-1,
        )

    return new_polylines


def _build_mask_for_token(
    npc_route_ids_list: List[str],
    chosen_lanes_rb_id_list: List[str],  # len: chosen_lane_num
    chosen_lanes_rb_id_set: Set[str],
) -> List[bool]:
    """단일 토큰(차량)에 대해, 어떤 차선이 그 차량 경로 위에 있는지 True/False 리스트로 만든다.

    동작 요약
    ----------
    1) npc_route_ids_list (해당 차량의 roadblock 시퀀스) 중에서,
       실제 차선 목록(chosen_lanes_rb_id_list)에 존재하는 것만 추린다.
    2) 그 roadblock 들이 서로 끊기지 않는 연속 구간이 되도록
       `_prune_route_by_connectivity` 로 한 번 더 정리한다.
    3) 최종적으로, chosen_lanes_rb_id_list 를 순회하면서
       각 roadblock id 가 “정리된 경로 집합” 안에 있는지 검사한다.
       → [True, False, ...] 형태의 리스트 반환.

    Args:
        npc_route_ids_list (Optional[List[str]]):
            - 해당 차량의 roadblock id 시퀀스.
            - None 이면 경로 정보가 없다고 보고 전부 False 로 처리.
        chosen_lanes_rb_id_list (List[str]):
            - 현재 샘플에서 사용 중인 lane 들의 roadblock id 리스트.
            - 길이 = chosen_lane_num.
        chosen_lanes_rb_id_set (Set[str]):
            - 위 리스트를 집합으로 만든 것 (빠른 포함 검사용).

    Returns:
        List[bool]:
            - 길이 = chosen_lane_num
            - j번째 값이 True 이면, chosen_lanes_rb_id_list[j] 가
              이 차량의 “연속성이 보장된” 경로에 포함된 roadblock 이라는 뜻.
    """
    # chosen_lanes_rb_id_list 안에 실제 존재하는 후보만 필터링
    npc_route_ids_in_chosen_set: Set[str] = {
        npc_a_route_id for npc_a_route_id in npc_route_ids_list
        if npc_a_route_id in chosen_lanes_rb_id_set
    }

    # 연속 구간 보정
    ongoing_npc_rrb_ids_in_chosen: List[str] = _prune_route_by_connectivity(
        npc_route_ids_list, npc_route_ids_in_chosen_set)
    ongoing_npc_rrb_ids_in_chosen_set: Set[str] = set(
        ongoing_npc_rrb_ids_in_chosen)

    # 각 lane 의 roadblock id 가 보정된 경로 집합 안에 있는지 검사
    chosen_lanes_npc_route_mask = [
        rb_id in ongoing_npc_rrb_ids_in_chosen_set
        for rb_id in chosen_lanes_rb_id_list
    ]
    return chosen_lanes_npc_route_mask


def _compute_car_token_to_chosen_lanes_route_mask(
        car_token_to_rr_ids: Dict[str, List[str]],  # 길이: chosen_car_num
        chosen_lanes_rb_id_list: List[str],  # 길이: chosen_lane_num
) -> Dict[str, List[bool]]:
    """토큰별 NPC 경로가 현재 추출된 차선(chosen_lanes_rb_id_list)에 포함되는지 불리언 마스크로 반환한다.

    각 차량 토큰에 대해,
    - 그 차량의 경로(RoadBlock ID 시퀀스)와
    - 현재 샘플에서 사용 중인 차선들의 roadblock id 리스트(chosen_lanes_rb_id_list)를 비교하여,

    “각 차선이 이 차량의 경로 위에 있는지”를 True/False 리스트로 만들어 준다.
    연결성 보정을 위해 `_prune_route_by_connectivity` 를 사용해
    경로가 끊기지 않는 연속 구간만 유지한다.

    Args:
        car_token_to_rr_ids (Dict[str, List[str]]):
            - 키: 차량 토큰 문자열
            - 값: 해당 차량의 roadblock id 시퀀스(List[str]).
            - 길이 = chosen_car_num.
        chosen_lanes_rb_id_list (List[str]):
            - 길이 = chosen_lane_num.
            - 현재 샘플에서 사용 중인 lane 들의 roadblock id (ego 기준 가까운 순 정렬).

    Returns:
        Dict[str, List[bool]]:
            - `car_token_to_chosen_lanes_route_mask`.
            - 각 키(차량 토큰)에 대해 값은 길이 chosen_lane_num 의 True/False 리스트.
              · j번째 값이 True 이면, chosen_lanes_rb_id_list[j] 가
                해당 차량의 보정된 경로에 포함된 roadblock 임을 의미.
    """
    chosen_lanes_rb_id_set: Set[str] = set(chosen_lanes_rb_id_list)
    car_token_to_chosen_lanes_route_mask: Dict[str, List[bool]] = {}

    for chosen_car_token, npc_route_ids_list in car_token_to_rr_ids.items():
        # npc_route_ids_list: List[str]
        # chosen_lanes_npc_route_mask
        car_token_to_chosen_lanes_route_mask[
            chosen_car_token] = _build_mask_for_token(
                npc_route_ids_list,
                chosen_lanes_rb_id_list,
                chosen_lanes_rb_id_set,
            )

    return car_token_to_chosen_lanes_route_mask


"""
        route_roadblock_ids: List[str],
        car_token_to_rr_ids: Dict[str, List[str]],  # 길이: chosen_car_num
        neighbor_track_token: List[str],  # 길이: chosen_agent_num
        neighbor_agents_current,  # # (chosen_agent_num, 11)
        ego_cur_pose_np: np.ndarray, # (3)
        elements_to_obj_polylines: Dict[str, MapObjectPolylines],
        elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
        speed_limit_dict:Dict[str, np.ndarray],
        lanes_roadblock_id_list: List[str],
"""


def _build_list_array_data_from_polylines(
    elements_to_obj_polylines: Dict[str, MapObjectPolylines],
    elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
) -> Dict[str, List[np.ndarray]]:
    """맵 폴리라인/신호등 정보를 순수 넘파이 리스트 형태로 풀어서 담는다.

    이 함수는 두 개의 입력 딕셔너리에서 실제 좌표/신호 데이터를 꺼내어
    다음과 같은 구조로 바꾼다.

    - 입력 1: elements_to_obj_polylines
        * 키: "LANE", "LEFT_BOUNDARY" 같은 문자열
        * 값: MapObjectPolylines 객체
            - 내부에 여러 개의 선(차선, 경계선 등)이 들어 있고,
              각 선은 (점 개수, 2) 형태의 (x, y) 좌표 배열로 구성된다.

    - 입력 2: elements_to_traffic_light
        * 키: "LANE" 등
        * 값: LaneSegmentTrafficLightData 객체
            - 내부에 각 차선의 각 점마다 신호등 상태를 나타내는
              (점 개수, 4) 형태의 배열이 들어있다.
              (예: [초록, 노랑, 빨강, 알 수 없음]을 0/1로 표시)

    - 출력: map_elements_to_list_array
        * 키:
            - "coords.<feature_name>"  예: "coords.LANE"
            - "traffic_light_data.<feature_name>"  예: "traffic_light_data.LANE"
        * 값:
            - List[np.ndarray]
            - 리스트 길이: 해당 feature 의 요소 개수(num_elements)
            - 각 원소: 한 요소의 좌표/신호 배열
               · "coords.LANE" -> [ (P_0, 2) 배열, (P_1, 2) 배열, ... ]
                    - len(list) = num_lanes
                · "traffic_light_data.LANE" -> [ (P_0, 4) 배열, ... ]
                    - len(list) = num_lanes

    이 단계에서는 좌표계를 바꾸지 않고, 단순히
    “클래스 감싼 상태 → 순수 넘파이 배열 리스트”로 풀어주는 역할만 한다.

    """
    map_elements_to_list_array: Dict[str, List[np.ndarray]] = {}

    for feature_name, map_object_polylines in elements_to_obj_polylines.items():
        # 좌표를 순수 넘파이 배열 리스트로 변환
        list_feature_coords: List[np.ndarray] = []
        # map_object_polylines.to_vector(): List[List[List[float]]]
        for element_coords in map_object_polylines.to_vector():
            # element_coords: List[List[float]] ->  (num_points, 2)
            list_feature_coords.append(
                np.array(element_coords, dtype=np.float64))
        map_elements_to_list_array[
            f"coords.{feature_name}"] = list_feature_coords

        # 신호등 정보도 있으면 같이 풀어준다.
        if feature_name in elements_to_traffic_light:
            list_feature_tl_data: List[np.ndarray] = []
            for element_tl_data in elements_to_traffic_light[
                    feature_name].to_vector():
                # element_tl_data: (num_points, 4)
                list_feature_tl_data.append(
                    np.array(element_tl_data, dtype=np.float64))
            map_elements_to_list_array[
                f"traffic_light_data.{feature_name}"] = list_feature_tl_data

    return map_elements_to_list_array


def _build_lane_core_arrays(
    ego_cur_pose_np: np.ndarray,  # (3,) = [ego_x, ego_y, ego_heading]
    feature_coords: List[np.ndarray],  # 길이 = num_lanes, 각 원소: (num_points_i, 2)
    feature_tl_data: Optional[List[
        np.ndarray]],  # 길이 = num_lanes, 각 원소: (num_points_i, 4) 또는 None
    speed_limit_dict: Dict[str, np.ndarray],
    lanes_roadblock_id_list: List[str],  # 길이 = num_lanes
    map_elements_to_list_array: Dict[str, List[np.ndarray]],
    map_max_elements: Dict[str, int],
    map_points_num: Dict[str, int],
    traffic_light_encoding_dim: int,
) -> Tuple[
        np.
        ndarray,  # chosen_center_xy             (chosen_lane_num, map_points_num["LANE"], 2)
        np.
        ndarray,  # lane_xy_valid_mask             (chosen_lane_num, map_points_num["LANE"])
        Optional[
            np.
            ndarray],  # tl_data            (chosen_lane_num, map_points_num["LANE"], 4) 또는 None
        np.
        ndarray,  # left_coords_local  (chosen_lane_num, map_points_num["LANE"], 2)
        np.
        ndarray,  # right_coords_local (chosen_lane_num, map_points_num["LANE"], 2)
        np.ndarray,  # lane_speed_limit_array       (chosen_lane_num, 1)
        np.ndarray,  # lane_has_speed_limit_array   (chosen_lane_num, 1)
        List[str],  # chosen_lanes_rb_id_list                  길이 ≈ chosen_lane_num
]:
    """LANE 폴리라인/경계/속도제한/신호 정보를 “고정 포인트 + ego 기준 좌표계”로 만든다.

    이 함수는 다음 순서로 동작한다.

    1) 원시 데이터 준비
       - LEFT/RIGHT_BOUNDARY 좌표 리스트를 꺼낸다.
       - lane 속도제한 1차원 배열을 꺼낸다.
       - feature_tl_data 가 있다면, LANE 개수와 길이가 맞는지 검사한다.

    2) ego 와의 거리 기준으로 사용할 lane 선택
       - `_select_lanes_by_ego_distance_to_keep` 을 호출해,
         ego 에서 가까운 lane 인덱스를 거리 순으로 정렬하고,
         최대 `map_max_elements["LANE"]` 개까지만 남긴다.
       - 결과:
         · selected_lane_indices: 실제 사용할 lane 인덱스 리스트
         · chosen_lane_num: 실제 lane 개수 (= min(chosen_lane_num, map_max_elements["LANE"]))

    3) 출력 배열 생성
       - `_initialize_lane_geometry_arrays` 로 중심선/왼쪽/오른쪽 경계선 배열 생성
         · chosen_center_xy / left_array / right_array
           shape: (chosen_lane_num, map_points_num["LANE"], 2)
       - `_initialize_lane_attribute_arrays` 로
         avail/속도제한/신호 배열 생성
         · lane_xy_valid_mask
           shape: (chosen_lane_num, map_points_num["LANE"])
         · lane_has_speed_limit_array / lane_speed_limit_array
           shape: (chosen_lane_num, 1)
         · lanes_point_tl_array (옵션)
           shape: (chosen_lane_num, map_points_num["LANE"], 4)

    4) 선택된 lane 들에 실제 데이터 채우기
       - `_fill_lane_arrays_for_selected_lanes` 를 호출해
         위 배열들을 채운다.
       - chosen_lanes_rb_id_list (선택된 lane 의 roadblock id 리스트)를 함께 얻는다.

    5) 왼쪽/오른쪽 경계선을 ego 기준 좌표계로 변환
       - `vector_set_coordinates_to_local_frame` 를 사용한다.
       - left_coords_local / right_coords_local 반환.

    최종적으로,
    - chosen_center_xy / lane_xy_valid_mask / tl_data:
        고정 포인트 개수를 가진 LANE 중심선/avail/신호 배열.
    - left_coords_local / right_coords_local:
        ego 기준 좌표계로 변환된 LANE 양쪽 경계선.
    - lane_speed_limit_array / lane_has_speed_limit_array:
        선택된 lane 들의 속도 제한 값/유무.
    - chosen_lanes_rb_id_list:
        선택된 lane 들이 속한 roadblock id 리스트.
    """
    # LEFT/RIGHT_BOUNDARY 원시 좌표 리스트
    left_boundary_list: List[np.ndarray] = map_elements_to_list_array[
        "coords.LEFT_BOUNDARY"]  # 각 원소: (num_points_i, 2)
    right_boundary_list: List[np.ndarray] = map_elements_to_list_array[
        "coords.RIGHT_BOUNDARY"]  # 각 원소: (num_points_i, 2)

    # 신호 데이터가 있다면, lane 개수와 길이가 맞는지 검사
    if feature_tl_data is not None and len(feature_coords) != len(
            feature_tl_data):
        raise ValueError(
            f"Size between feature coords and traffic light data inconsistent: "
            f"{len(feature_coords)}, {len(feature_tl_data)}")

    # 속도제한 1차원 배열 (chosen_lane_num,)
    lane_has_speed_limit: np.ndarray = speed_limit_dict["lane_has_speed_limit"]
    lane_speed_limit: np.ndarray = speed_limit_dict["lane_speed_limit"]

    # 1) ego 와의 거리 기준으로 사용할 lane 인덱스 선택
    selected_lane_indices, chosen_lane_num = _select_lanes_by_ego_distance_to_keep(
        ego_pose=ego_cur_pose_np,  # (3,)
        feature_coords=
        feature_coords,  # List[np.ndarray]  길이 = num_lanes, 각 원소: (num_points_i, 2)
        map_max_elements=map_max_elements["LANE"],
    )

    # 2) 출력 배열 생성 (기하 정보)
    chosen_center_xy, left_array, right_array = _initialize_lane_geometry_arrays(
        chosen_lane_num=chosen_lane_num,
        map_points_num=map_points_num["LANE"],
    )

    # 3) 출력 배열 생성 (부가 정보)
    (
        lane_xy_valid_mask,
        lane_has_speed_limit_array,
        lane_speed_limit_array,
        lanes_point_tl_array,
    ) = _initialize_lane_attribute_arrays(
        chosen_lane_num=chosen_lane_num,
        map_points_num=map_points_num["LANE"],
        traffic_light_encoding_dim=traffic_light_encoding_dim,
        has_traffic_light=(feature_tl_data is not None),
    )

    # 4) 선택된 lane 들에 실제 데이터 채우기
    chosen_lanes_rb_id_list: List[str] = _fill_lane_arrays_for_selected_lanes(
        selected_lane_indices=selected_lane_indices,
        feature_coords=feature_coords,
        left_boundary=left_boundary_list,
        right_boundary=right_boundary_list,
        lanes_roadblock_id_list=lanes_roadblock_id_list,
        lane_has_speed_limit=lane_has_speed_limit,
        lane_speed_limit=lane_speed_limit,
        feature_tl_data=feature_tl_data,
        map_points_num=map_points_num["LANE"],
        chosen_center_xy=chosen_center_xy,
        left_array=left_array,
        right_array=right_array,
        lane_xy_valid_mask=lane_xy_valid_mask,
        lane_has_speed_limit_array=lane_has_speed_limit_array,
        lane_speed_limit_array=lane_speed_limit_array,
        lanes_point_tl_array=lanes_point_tl_array,
    )

    # 5) LEFT/RIGHT_BOUNDARY 를 ego 기준 좌표계로 변환
    left_coords_local: np.ndarray = vector_set_coordinates_to_local_frame(
        coords=left_array,  # (chosen_lane_num, map_points_num["LANE"], 2)
        avails=lane_xy_valid_mask,  # (chosen_lane_num, map_points_num["LANE"])
        anchor_state=ego_cur_pose_np,
    )
    right_coords_local: np.ndarray = vector_set_coordinates_to_local_frame(
        coords=right_array,  # (chosen_lane_num, map_points_num["LANE"], 2)
        avails=lane_xy_valid_mask,  # (chosen_lane_num, map_points_num["LANE"])
        anchor_state=ego_cur_pose_np,
    )

    # 중심선/avail/신호는 그대로 반환

    return (
        chosen_center_xy,
        lane_xy_valid_mask,
        lanes_point_tl_array,
        left_coords_local,
        right_coords_local,
        lane_speed_limit_array,
        lane_has_speed_limit_array,
        chosen_lanes_rb_id_list,
    )


def _build_lane_route_and_npc_masks(
    chosen_lanes_rb_id_list: List[str],  # len ≈ chosen_lane_num
    route_roadblock_ids: List[str],
    car_token_to_rr_ids: Dict[str, List[str]],  # 길이: chosen_car_num
) -> Tuple[
        List[bool],  # chosen_lanes_route_mask  len = chosen_lane_num
        Dict[str, List[bool]],  # car_token_to_chosen_lanes_route_mask
]:
    """ego 경로(route)와 각 차량별 route를 기준으로,
    차선이 경로 위에 있는지 여부를 두 가지 관점에서 계산한다.

    동작 요약
    --------
    1) ego route 기준 마스크(chosen_lanes_route_mask)
       - `route_roadblock_ids` 와 `chosen_lanes_rb_id_list` 를 비교해,
         둘 다에 등장하는 roadblock id 만 추린다.
       - `_prune_route_by_connectivity` 로 이 id 들이 끊김 없이 이어지는
         연속 구간이 되도록 정리한다.
       - 각 차선의 roadblock id 가 이 “연속된 route id 집합” 안에 있으면 True.

    2) 차량별 route 기준 마스크(car_token_to_chosen_lanes_route_mask)
       - `car_token_to_rr_ids[token]` (차량별 roadblock 시퀀스)를
         `_compute_car_token_to_chosen_lanes_route_mask` 에 넘겨,
         각 token 에 대해 "각 차선이 그 차량의 route 위에 있는지"를
         True/False 리스트로 만든다.

    Args:
        route_roadblock_ids:
            시나리오 전체 route 를 구성하는 roadblock id 시퀀스.
        chosen_lanes_rb_id_list:
            현재 선택된 차선들의 roadblock id 리스트. 길이 = chosen_lane_num.
        car_token_to_rr_ids:
            차량 토큰 → 해당 차량의 roadblock id 시퀀스(List[str]).

    Returns:
            - chosen_lanes_route_mask: List[bool]
                각 차선이 전체 route 위에 있는지 여부 리스트.
                길이 = chosen_lane_num.
            - car_token_to_chosen_lanes_route_mask:  Dict[str, List[bool]]
                차량 토큰별로, 각 차선이 그 차량 경로 위에 있는지 여부 리스트.
                key의 개수: chosen_car_num,
                값 리스트의 길이 = chosen_lane_num.
    """
    # 1) route 와 LANE 의 roadblock id 를 비교해, 실제 route 위에 있는 LANE 만 찾는다.
    chosen_lanes_route_mask: List[bool] = []

    # route 에도 등장하면서, 현재 선택된 차선들에도 등장하는 roadblock 만 추림
    rrb_ids_in_chosen_lane: List[str] = [
        a_route_rb_id for a_route_rb_id in route_roadblock_ids
        if a_route_rb_id in chosen_lanes_rb_id_list
    ]
    # 연속되지 않는 부분은 잘라낸다.
    ongoing_rrb_ids_in_chosen_lane: List[str] = _prune_route_by_connectivity(
        route_roadblock_ids,
        set(rrb_ids_in_chosen_lane),
    )

    for a_chosen_lane_rb_id in chosen_lanes_rb_id_list:
        chosen_lanes_route_mask.append(
            a_chosen_lane_rb_id in ongoing_rrb_ids_in_chosen_lane)

    # 2) 각 차량 경로 기준으로, 어떤 LANE 이 그 차의 경로 위에 있는지 표시한다.
    car_token_to_chosen_lanes_route_mask: Dict[str, List[bool]] = \
        _compute_car_token_to_chosen_lanes_route_mask(
            car_token_to_rr_ids=car_token_to_rr_ids,
            chosen_lanes_rb_id_list=chosen_lanes_rb_id_list,
        )

    return chosen_lanes_route_mask, car_token_to_chosen_lanes_route_mask


def _prepare_array_output_and_lane_info(
    map_elements_to_list_array: Dict[str, List[np.ndarray]],
    map_elements: List[
        str],  # 예: ['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES']
    ego_cur_pose_np: np.ndarray,  # shape: (3,)
    speed_limit_dict: Dict[str, np.ndarray],
    lanes_roadblock_id_list: List[str],  # len = num_lanes
    route_roadblock_ids: List[str],
    car_token_to_rr_ids: Dict[str, List[str]],  # len = chosen_car_num
    map_max_elements: Dict[str, int],
    map_points_num: Dict[str, int],
) -> Tuple[
        Dict[str, np.ndarray],  # map_elements_to_processed_array
        List[bool],  # chosen_lanes_route_mask
        np.ndarray,  # lane_speed_limit_array        shape: (chosen_lane_num, 1)
        np.ndarray,  # lane_has_speed_limit_array    shape: (chosen_lane_num, 1)
        Dict[str, List[bool]],  # car_token_to_chosen_lanes_route_mask
]:
    """맵 폴리라인/신호 리스트 ->고정 크기 배열(dict)과 보조 정보들

    동작 요약
    --------
    1) LANE 처리
       - `_build_lane_core_arrays` 로
            · ego 기준으로 가까운 차선만 고르고
            · 포인트 수를 `map_points_num["LANE"]` 로 맞추고
            · 좌표/유효마스크/신호/속도제한/roadblock id 를 모두 배열로 만든다.
       - `_build_lane_route_and_npc_masks` 로
            · 각 route 차선이 전체 route 위에 있는지 (chosen_lanes_route_mask)
            · 각 차량 토큰별로, 각 route 차선이 그 차량 경로 위에 있는지
           (car_token_to_chosen_lanes_route_mask) 를 계산한다.
       - 왼쪽/오른쪽 경계선 좌표는 이미 ego 기준 좌표계이므로
         `"vector_set_map.coords.LEFT_BOUNDARY"`, `"RIGHT_BOUNDARY"` 키에 저장한다.

    2) 그 외 feature(LANE, ROUTE_LANES 등) 처리
       - LANE 에 대해서는
         `chosen_center_xy`(차선 중심선)와 `lane_xy_valid_mask` 를 ego 기준 좌표계로 바꿔
         `"vector_set_map.coords.LANE"` / `"vector_set_map.availabilities.LANE"` 로 저장한다.
       - ROUTE_LANES 등 다른 feature 이름도 같은 방식으로
         좌표/마스크/신호를 `"vector_set_map.*.<feature_name>"` 형식의 키에 채운다.
       - LANE 에서 계산한 `chosen_center_xy`, `lane_xy_valid_mask`, `tl_data` 를 재사용하므로
         실제 연산은 LANE 한 번만 수행되고 나머지는 복사/좌표계 변환만 이뤄진다.

    Args:
        map_elements_to_list_array:
            - `"coords.LANE"`, `"coords.LEFT_BOUNDARY"` 등 문자열 키로
              각 feature 에 대한 좌표 리스트를 담은 dict.
            - 예: "coords.LANE" → [ (P0, 2), (P1, 2), ... ] 길이 = num_lanes.
        map_elements:
            - 처리할 맵 feature 이름 리스트.
            - 예: ["LANE", "LEFT_BOUNDARY", "RIGHT_BOUNDARY", "ROUTE_LANES"].
        ego_cur_pose_np:
            - 현재 ego 상태 [x, y, heading], shape = (3,).
        speed_limit_dict:
            - 차선 속도제한 정보 dict.
            - "lane_has_speed_limit": (num_lanes,)
            - "lane_speed_limit": (num_lanes,).
        lanes_roadblock_id_list:
            - 각 차선이 속한 roadblock id 리스트, 길이 = num_lanes.
        route_roadblock_ids:
            - ego route 를 구성하는 roadblock id 시퀀스.
        car_token_to_rr_ids:
            - 차량 토큰 → 그 차량의 route roadblock id 시퀀스.
        map_max_elements:
            - feature 별 최대 요소 개수. 예: {"LANE": lane_num, ...}.
        map_points_num:
            - feature 별 고정 포인트 수. 예: {"LANE": lane_len, ...}.

    Returns:
        Tuple[
            map_elements_to_processed_array,
            chosen_lanes_route_mask,
            lane_speed_limit_array,
            lane_has_speed_limit_array,
            car_token_to_chosen_lanes_route_mask,
        ]
        - map_elements_to_processed_array:
            `"vector_set_map.coords.<feature_name>"`,
            `"vector_set_map.availabilities.<feature_name>"`,
            `"vector_set_map.traffic_light_data.<feature_name>"` 등을 포함하는 dict.
        - chosen_lanes_route_mask:
            각 차선이 전체 route 위에 있는지 여부 리스트 (길이 = chosen_lane_num).
        - lane_speed_limit_array:
            선택된 차선의 속도제한 값(m/s), shape = (chosen_lane_num, 1).
        - lane_has_speed_limit_array:
            선택된 차선이 실제 속도제한을 가지는지 여부, shape = (chosen_lane_num, 1).
        - car_token_to_chosen_lanes_route_mask:
            차량 토큰별로, 각 차선이 그 차량 경로 위에 있는지 여부 리스트.
    """
    map_elements_to_processed_array: Dict[str, np.ndarray] = {}
    chosen_lanes_route_mask: List[bool] = []

    lane_has_speed_limit_array: Optional[
        np.ndarray] = None  # shape: (chosen_lane_num, 1)
    lane_speed_limit_array: Optional[
        np.ndarray] = None  # shape: (chosen_lane_num, 1)
    car_token_to_chosen_lanes_route_mask: Dict[str, List[bool]] = {}

    traffic_light_encoding_dim: int = LaneSegmentTrafficLightData.encoding_dim()

    chosen_center_xy: Optional[
        np.ndarray] = None  # shape: (chosen_lane_num, lane_len, 2)
    lane_xy_valid_mask: Optional[
        np.ndarray] = None  # shape: (chosen_lane_num, lane_len)
    tl_data: Optional[
        np.ndarray] = None  # shape: (chosen_lane_num, lane_len, 4) 또는 None

    for feature_name in map_elements:
        coords_key: str = f"coords.{feature_name}"
        if coords_key not in map_elements_to_list_array:
            continue

        # feature_coords: List[(num_points_i, 2)]
        feature_coords: List[
            np.ndarray] = map_elements_to_list_array[coords_key]
        tl_key: str = f"traffic_light_data.{feature_name}"
        feature_tl_data: Optional[List[np.ndarray]] = (
            map_elements_to_list_array[tl_key]
            if tl_key in map_elements_to_list_array else None)

        if feature_name == "LANE":
            (
                chosen_center_xy,  # (chosen_lane_num, map_points_num["LANE"], 2)
                lane_xy_valid_mask,  # (chosen_lane_num, map_points_num["LANE"])
                tl_data,  # (chosen_lane_num, map_points_num["LANE"], 4) 또는 None
                left_coords_local,  # (chosen_lane_num, map_points_num["LANE"], 2)
                right_coords_local,  # (chosen_lane_num, map_points_num["LANE"], 2)
                lane_speed_limit_array,  # (chosen_lane_num, 1)
                lane_has_speed_limit_array,  # (chosen_lane_num, 1)
                chosen_lanes_rb_id_list,  # List[str], len = chosen_lane_num
            ) = _build_lane_core_arrays(
                ego_cur_pose_np=ego_cur_pose_np,
                feature_coords=feature_coords,
                feature_tl_data=feature_tl_data,
                speed_limit_dict=speed_limit_dict,
                lanes_roadblock_id_list=lanes_roadblock_id_list,
                map_elements_to_list_array=map_elements_to_list_array,
                map_max_elements=map_max_elements,
                map_points_num=map_points_num,
                traffic_light_encoding_dim=traffic_light_encoding_dim,
            )

            # route/NPC 기준 포함 여부 마스크
            """
            - chosen_lanes_route_mask: List[bool]
                각 차선이 전체 route 위에 있는지 여부 리스트.
                길이 = chosen_lane_num.
            - car_token_to_chosen_lanes_route_mask:  Dict[str, List[bool]]
                차량 토큰별로, 각 차선이 그 차량 경로 위에 있는지 여부 리스트.
                key의 개수: chosen_car_num,
                값 리스트의 길이 = chosen_lane_num.
            """
            (
                chosen_lanes_route_mask,
                car_token_to_chosen_lanes_route_mask,
            ) = _build_lane_route_and_npc_masks(
                chosen_lanes_rb_id_list=chosen_lanes_rb_id_list,
                route_roadblock_ids=route_roadblock_ids,
                car_token_to_rr_ids=car_token_to_rr_ids,
            )

            map_elements_to_processed_array[
                "vector_set_map.coords.LEFT_BOUNDARY"] = left_coords_local
            map_elements_to_processed_array[
                "vector_set_map.coords.RIGHT_BOUNDARY"] = right_coords_local

        elif feature_name in ("LEFT_BOUNDARY", "RIGHT_BOUNDARY"):
            # LANE 처리에서 이미 채웠으므로 여기서는 건너뜀
            continue

        # LANE / ROUTE_LANES 등에 대해 공통으로 수행되는 좌표/마스크/신호 채우기
        coords_local: np.ndarray = vector_set_coordinates_to_local_frame(
            coords=chosen_center_xy,  # type: ignore[arg-type]
            avails=lane_xy_valid_mask,  # type: ignore[arg-type]
            anchor_state=ego_cur_pose_np,
        )

        map_elements_to_processed_array[
            f"vector_set_map.coords.{feature_name}"] = coords_local
        map_elements_to_processed_array[
            f"vector_set_map.availabilities.{feature_name}"] = lane_xy_valid_mask  # type: ignore[arg-type]

        if tl_data is not None:
            map_elements_to_processed_array[
                f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data

    # LANE 이 반드시 포함된다는 기존 가정 유지
    assert lane_has_speed_limit_array is not None
    assert lane_speed_limit_array is not None

    return (
        map_elements_to_processed_array,
        chosen_lanes_route_mask,
        lane_speed_limit_array,
        lane_has_speed_limit_array,
        car_token_to_chosen_lanes_route_mask,
    )


def _build_lane_vector_and_agent_route_order(
    map_elements_to_processed_array: Dict[str, np.ndarray],
    neighbor_track_token: List[str],  # 길이: chosen_agent_num
    neighbor_agents_current: np.ndarray,  # shape: (chosen_agent_num, 11)
    car_token_to_chosen_lanes_route_mask: Dict[str, List[bool]],
) -> Tuple[np.ndarray, np.ndarray]:
    """LANE 차선 벡터(feature 12차원)와 에이전트별 차선 순서 행렬을 만든다.

    하는 일
    -------
    1) `map_elements_to_processed_array` 에서 LANE 관련 배열들을 꺼낸다.
       - 중심선 좌표, 좌/우 경계선, 신호 one-hot, 유효 마스크.
    2) `_lane_polyline_process` 를 호출해
       각 포인트마다 (위치, 이전점과의 차이, 좌/우 경계까지의 상대 위치, 신호 상태)를
       이어붙인 길이 12 벡터를 만든다.
       → 결과: `vector_map_lanes` shape = (lane_num, lane_len, 12)
    3) `_select_token_and_ordered_npc_route_indices` 를 사용해
       - 각 에이전트별로, “경로 위에 있는 차선들”을
         ego 기준 거리 순으로 정렬하고 랭크를 부여한다.
       → 결과: `agent_route_lane_order` shape = (agent_num, lane_num)

    Args:
        map_elements_to_processed_array:
            - `"vector_set_map.coords.LANE"`: (lane_num, lane_len, 2)
            - `"vector_set_map.coords.LEFT_BOUNDARY"`: (lane_num, lane_len, 2)
            - `"vector_set_map.coords.RIGHT_BOUNDARY"`: (lane_num, lane_len, 2)
            - `"vector_set_map.traffic_light_data.LANE"`: (lane_num, lane_len, 4)
            - `"vector_set_map.availabilities.LANE"`: (lane_num, lane_len)
        neighbor_track_token:
            - 에이전트 슬롯별 track_token, 길이 = agent_num.
        neighbor_agents_current:
            - 현재 프레임 이웃 에이전트 상태.
            - shape = (agent_num, 11).
        - car_token_to_chosen_lanes_route_mask:  Dict[str, List[bool]]
            차량 토큰별로, 각 차선이 그 차량 경로 위에 있는지 여부 리스트.
            key의 개수: chosen_car_num,
            값 리스트의 길이 = chosen_lane_num.


    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - vector_map_lanes:
                차선 벡터 표현. shape = (lane_num, lane_len, 12).
            - agent_route_lane_order:
                에이전트별 차선 순서 랭크 행렬. shape = (agent_num, lane_num), dtype=int64.
    """
    # polylines: (lane_num, lane_len, 2)
    polylines: np.ndarray = map_elements_to_processed_array[
        "vector_set_map.coords.LANE"]
    # left/right_boundary: (lane_num, lane_len, 2)
    left_boundary: np.ndarray = map_elements_to_processed_array[
        "vector_set_map.coords.LEFT_BOUNDARY"]
    right_boundary: np.ndarray = map_elements_to_processed_array[
        "vector_set_map.coords.RIGHT_BOUNDARY"]
    # traffic_light_state: (lane_num, lane_len, 4)
    traffic_light_state: np.ndarray = map_elements_to_processed_array[
        "vector_set_map.traffic_light_data.LANE"]
    # lane_xy_valid_mask: (lane_num, lane_len)
    lane_xy_valid_mask: np.ndarray = map_elements_to_processed_array[
        "vector_set_map.availabilities.LANE"]

    # (1) 차선 벡터 특징(12차원) 만들기
    vector_map_lanes: np.ndarray = _lane_polyline_process(
        polylines=polylines,
        left_boundary=left_boundary,
        right_boundary=right_boundary,
        lane_xy_valid_mask=lane_xy_valid_mask,
        traffic_light=traffic_light_state,
    )  # shape: (lane_num, lane_len, 12)

    # (2) 에이전트별 route 차선 순서 행렬 만들기
    agent_route_lane_order: np.ndarray = _select_token_and_ordered_npc_route_indices(
        car_token_to_chosen_lanes_route_mask,
        neighbor_track_token,
        neighbor_agents_current,
        vector_map_lanes,
    )  # shape: (agent_num, lane_num)

    # 타입 통일: int64
    if agent_route_lane_order.dtype != np.int64:
        agent_route_lane_order = agent_route_lane_order.astype(np.int64)

    return vector_map_lanes, agent_route_lane_order


def _build_route_lane_vectors(
        vector_map_lanes: np.ndarray,  # shape: (lane_num, lane_len, 12)
        chosen_lanes_route_mask: List[bool],  # 길이 = lane_num
        lane_speed_limit_array: np.ndarray,  # shape: (lane_num, 1)
        lane_has_speed_limit_array: np.ndarray,  # shape: (lane_num, 1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """전체 차선 벡터에서 “route 위에 있는 차선들”만 골라 route_lanes 배열을 만든다.

    이 함수는 이미 만들어진 차선 벡터들(`vector_map_lanes`)과
    “이 차선이 route 위에 있는지”를 나타내는 불리언 리스트를 받아서,

    - route 위에 있는 차선들만 순서대로 골라
      `route_lanes` 배열을 만들고
    - 그 차선들의 속도제한 값/존재 여부 배열도 함께 만든다.

    Args:
        vector_map_lanes:
            모든 차선의 벡터 표현.
            shape = (lane_num, lane_len, 12).
        chosen_lanes_route_mask:
            각 차선이 route 위에 있는지 여부.
            길이 = lane_num.
        lane_speed_limit_array:
            모든 차선의 속도제한 값(m/s).
            shape = (lane_num, 1).
        lane_has_speed_limit_array:
            모든 차선의 속도제한 존재 여부.
            shape = (lane_num, 1).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - vector_map_route_lanes:
                route 위에 있는 차선들만 모은 배열.
                shape = (route_lane_num, lane_len, 12).
            - route_lanes_speed_limit:
                route 위에 있는 차선들의 속도제한 값.
                shape = (route_lane_num, 1).
            - route_lanes_has_speed_limit:
                route 위에 있는 차선들의 속도제한 존재 여부.
                shape = (route_lane_num, 1).
    """
    if vector_map_lanes.size == 0:
        # 차선 자체가 없는 경우: 모두 0 크기 배열 반환
        vector_map_route_lanes = np.zeros((0, 0, 12), dtype=np.float32)
        route_lanes_speed_limit = np.zeros((0, 1), dtype=np.float32)
        route_lanes_has_speed_limit = np.zeros((0, 1), dtype=np.bool_)
        return vector_map_route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit

    lane_num: int = int(vector_map_lanes.shape[0])
    lane_len: int = int(vector_map_lanes.shape[1])
    lane_feat_dim: int = int(vector_map_lanes.shape[2])

    if len(chosen_lanes_route_mask) != lane_num:
        raise ValueError(
            f"chosen_lanes_route_mask 길이({len(chosen_lanes_route_mask)})와 "
            f"lane_num({lane_num})이 다릅니다.")

    # route 위에 있는 lane 인덱스만 골라냄
    route_lane_indices: List[int] = [
        idx for idx, is_on_route in enumerate(chosen_lanes_route_mask)
        if is_on_route
    ]
    route_lane_num: int = len(route_lane_indices)

    # route 전용 배열 생성
    vector_map_route_lanes: np.ndarray = np.zeros(
        (route_lane_num, lane_len, lane_feat_dim), dtype=np.float32)
    route_lanes_speed_limit: np.ndarray = np.zeros((route_lane_num, 1),
                                                   dtype=np.float32)
    route_lanes_has_speed_limit: np.ndarray = np.zeros((route_lane_num, 1),
                                                       dtype=np.bool_)

    # 실제 route 상에 있는 lane 들만 복사
    for loc, lane_idx in enumerate(route_lane_indices):
        vector_map_route_lanes[loc] = vector_map_lanes[lane_idx]
        route_lanes_speed_limit[loc] = lane_speed_limit_array[lane_idx]
        route_lanes_has_speed_limit[loc] = lane_has_speed_limit_array[lane_idx]

    return vector_map_route_lanes, route_lanes_speed_limit, route_lanes_has_speed_limit


def map_process(
    route_roadblock_ids: List[str],
    car_token_to_rr_ids: Dict[str, List[str]],  # 길이: chosen_car_num
    neighbor_track_token: List[str],  # 길이: chosen_agent_num
    neighbor_agents_current: np.ndarray,  # shape: (agent_num, 11)
    ego_cur_pose_np: np.ndarray,  # shape: (3,)
    elements_to_obj_polylines: Dict[str, MapObjectPolylines],
    elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData],
    speed_limit_dict: Dict[str, np.ndarray],
    lanes_roadblock_id_list: List[str],
    map_elements: List[str],
    map_max_elements: Dict[str, int],
    map_points_num: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """지도 관련 원시 정보(route/차선/신호/속도제한)를 모델 입력용 넘파이 배열로 가공한다.

    크게 세 단계로 나뉜다.

    Args:
        route_roadblock_ids:
            - 전체 route 를 이루는 roadblock id 시퀀스.
        car_token_to_rr_ids:
            - 차량 토큰 → 해당 차량의 roadblock id 시퀀스.
        neighbor_track_token:
            - 이웃 에이전트 slot 순서에 맞는 track_token 리스트.
        neighbor_agents_current:
            - 현재 프레임 이웃 에이전트 상태. shape = (agent_num, 11).
        ego_cur_pose_np:
            - ego 현재 상태 [x, y, heading]. shape = (3,).
        elements_to_obj_polylines:
            - feature 이름 → MapObjectPolylines.
        elements_to_traffic_light:
            - feature 이름 → LaneSegmentTrafficLightData.
        speed_limit_dict:
            - "lane_has_speed_limit": (num_lanes,)
            - "lane_speed_limit": (num_lanes,).
        lanes_roadblock_id_list:
            - 각 차선이 속한 roadblock id 리스트. 길이 = num_lanes.
        map_elements:
            - 사용할 feature 이름 리스트.
        map_max_elements:
            - feature 별 최대 요소 개수.
        map_points_num:
            - feature 별 고정 포인트 수.

    Returns:
        Dict[str, np.ndarray]:
            - "lanes": (lane_num, lane_len, 12)
            - "lanes_speed_limit": (lane_num, 1)
            - "lanes_has_speed_limit": (lane_num, 1)
            - "route_lanes": (route_lane_num, lane_len, 12)
            - "route_lanes_speed_limit": (route_lane_num, 1)
            - "route_lanes_has_speed_limit": (route_lane_num, 1)
            - "agent_route_lane_order": (agent_num, lane_num), dtype=int64.
    """
    """ _build_list_array_data_from_polylines
    1) 폴리라인/신호 raw 데이터 → Dict[str, List[np.ndarray]]
         MapObjectPolylines / LaneSegmentTrafficLightData 를
         순수 넘파이 배열 리스트 구조로 바꾼다.
         예) "coords.LANE" → [ (P0, 2), (P1, 2), ... ].
    """
    map_elements_to_list_array: Dict[
        str, List[np.ndarray]] = _build_list_array_data_from_polylines(
            elements_to_obj_polylines=elements_to_obj_polylines,
            elements_to_traffic_light=elements_to_traffic_light,
        )
    """ `_prepare_array_output_and_lane_info`:
    2) 리스트 → 고정 크기 배열 + 보조 정보
         · LANE 에 대해서는 "ego 기준으로 가까운 차선만 고르고" "포인트 수를 맞춘" 뒤, 
                좌표/마스크/신호/속도제한/roadblock id 를 모두 배열로 만든다.
         · "각 차선이 route 위에 있는지" 여부, "각 차량 경로와의 포함 여부"도 함께 계산
         · "그 외 feature 들도 ego 기준 좌표계로 변환"해
           `"vector_set_map.*.<feature_name>"` 표준 키 아래에 넣는다.
    """
    (
        map_elements_to_processed_array,
        chosen_lanes_route_mask,
        lane_speed_limit_array,
        lane_has_speed_limit_array,
        car_token_to_chosen_lanes_route_mask,
    ) = _prepare_array_output_and_lane_info(
        map_elements_to_list_array=map_elements_to_list_array,
        map_elements=map_elements,
        ego_cur_pose_np=ego_cur_pose_np,
        speed_limit_dict=speed_limit_dict,
        lanes_roadblock_id_list=lanes_roadblock_id_list,
        route_roadblock_ids=route_roadblock_ids,
        car_token_to_rr_ids=car_token_to_rr_ids,
        map_max_elements=map_max_elements,
        map_points_num=map_points_num,
    )
    """ _build_lane_vector_and_agent_route_order
         차선 폴리라인 + 경계선 + 신호를 묶어 (lane_num, lane_len, 12) 벡터로 만들고,
         에이전트별 route lane 순서 랭크 행렬 (agent_num, lane_num)을 만든다.
    """
    (vector_map_lanes,
     agent_route_lane_order) = _build_lane_vector_and_agent_route_order(
         map_elements_to_processed_array=map_elements_to_processed_array,
         neighbor_track_token=neighbor_track_token,
         neighbor_agents_current=neighbor_agents_current,
         car_token_to_chosen_lanes_route_mask=
         car_token_to_chosen_lanes_route_mask,
     )
    """ _build_route_lane_vectors
         route 위에 있는 차선만 골라 별도의 route_lanes 배열과
         그에 대응하는 속도제한 정보를 만든다.
    """
    (
        vector_map_route_lanes,
        route_lanes_speed_limit,
        route_lanes_has_speed_limit,
    ) = _build_route_lane_vectors(
        vector_map_lanes=vector_map_lanes,
        chosen_lanes_route_mask=chosen_lanes_route_mask,
        lane_speed_limit_array=lane_speed_limit_array,
        lane_has_speed_limit_array=lane_has_speed_limit_array,
    )

    vector_map_output: Dict[str, np.ndarray] = {
        "lanes": vector_map_lanes,  # (lane_num, lane_len, 12)
        "lanes_speed_limit": lane_speed_limit_array,  # (lane_num, 1)
        "lanes_has_speed_limit": lane_has_speed_limit_array,  # (lane_num, 1)
        "route_lanes": vector_map_route_lanes,  # (route_lane_num, lane_len, 12)
        "route_lanes_speed_limit":
            route_lanes_speed_limit,  # (route_lane_num, 1)
        "route_lanes_has_speed_limit":
            route_lanes_has_speed_limit,  # (route_lane_num, 1)
        "agent_route_lane_order":
            agent_route_lane_order,  # (agent_num, lane_num)
    }
    return vector_map_output
