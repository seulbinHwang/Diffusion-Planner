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


def _convert_lane_to_fixed_size(ego_pose, feature_coords, speed_limit_dict,
                                lanes_roadblock_id_list, left_boundary,
                                right_boundary, feature_tl_data, max_elements,
                                max_points, traffic_light_encoding_dim):

    if feature_tl_data is not None and len(feature_coords) != len(
            feature_tl_data):
        raise ValueError(
            f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}"
        )

    lane_has_speed_limit = speed_limit_dict['lane_has_speed_limit']
    lane_speed_limit = speed_limit_dict['lane_speed_limit']

    # trim or zero-pad elements to maintain fixed size
    coords_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)
    left_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)
    right_array = np.zeros((max_elements, max_points, 2), dtype=np.float64)

    lane_has_speed_limit_array = np.zeros((max_elements, 1), dtype=np.bool_)
    lane_speed_limit_array = np.zeros((max_elements, 1), dtype=np.float32)
    lane_routes = []

    avails_array = np.zeros((max_elements, max_points), dtype=np.bool_)
    tl_data_array = (np.zeros(
        (max_elements, max_points, traffic_light_encoding_dim),
        dtype=np.float32) if feature_tl_data is not None else None)

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = np.linalg.norm(e - ego_pose[None, :2], axis=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
        left_coords = left_boundary[element_idx[0]]
        right_coords = right_boundary[element_idx[0]]

        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = _interpolate_points(element_coords, max_points)
        left_coords = _interpolate_points(left_coords, max_points)
        right_coords = _interpolate_points(right_coords, max_points)

        coords_array[idx] = element_coords
        left_array[idx] = left_coords
        right_array[idx] = right_coords
        avails_array[idx] = True  # specify real vs zero-padded data

        lane_has_speed_limit_array[idx] = lane_has_speed_limit[element_idx[0]]
        lane_speed_limit_array[idx] = lane_speed_limit[element_idx[0]]
        lane_routes.append(lanes_roadblock_id_list[element_idx[0]])

        if tl_data_array is not None and feature_tl_data is not None:
            tl_data_array[idx] = feature_tl_data[element_idx[0]]

    return coords_array, left_array, right_array, tl_data_array, avails_array, lane_has_speed_limit_array, lane_speed_limit_array, lane_routes


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


def _lane_polyline_process(polylines, left_boundary, right_boundary, avails,
                           traffic_light):
    dim = 12
    new_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1],
                                    dim),
                             dtype=np.float32)

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_vector = polyline[1:] - polyline[:-1]
            polyline_vector = np.insert(polyline_vector,
                                        polyline_vector.shape[0],
                                        0,
                                        axis=0)

            if np.linalg.norm(left_boundary[i, -1] -
                              polyline[0]) < np.linalg.norm(left_boundary[i,
                                                                          0] -
                                                            polyline[0]):
                left_boundary[i] = np.flip(left_boundary[i], axis=0)

            if np.linalg.norm(right_boundary[i, -1] -
                              polyline[0]) < np.linalg.norm(right_boundary[i,
                                                                           0] -
                                                            polyline[0]):
                right_boundary[i] = np.flip(right_boundary[i], axis=0)

            polyline_to_left = left_boundary[i] - polyline
            polyline_to_right = right_boundary[i] - polyline

            new_polylines[i] = np.concatenate([
                polyline, polyline_vector, polyline_to_left, polyline_to_right,
                traffic_light[i]
            ],
                                              axis=-1)

    return new_polylines


def _compute_lane_on_npc_routes(
        car_token_to_rr_ids: Dict[
            str, Optional[List[str]]],  # 길이 ≤ agent_num (차량만 포함 가능)
        lane_routes: List[str],  # 길이 <= lane_num 이하
) -> Dict[str, List[bool]]:
    """토큰별 NPC 경로가 현재 추출된 차선(lane_routes)에 포함되는지 불리언 마스크로 반환한다.

    각 토큰의 경로(RoadBlock ID 시퀀스)를 현재 샘플에서 추출된 lane의 roadblock id 리스트
    `lane_routes`에 대하여 멤버십으로 투영한다. 연결성 보정을 위해
    `_prune_route_by_connectivity`를 사용하여 연속 구간만 유지한다.

    Args:
        car_token_to_rr_ids (Dict[str, Optional[List[str]]]):
            키 = 토큰(str), 값 = 보정된 RoadBlock ID 시퀀스(List[str]) 또는 None.
            길이 ≤ agent_num (차량만 선별).
        lane_routes (List[str]):
            길이 <= lane_num. 현재 샘플에서 추출된 lane들의 roadblock id(거리 순 정렬).

    Returns:
        Dict[str, List[bool]]:
            `token_to_lane_on_routes`. 키=토큰, 값=길이  <= lane_num의 불리언 리스트.
            각 j에 대해 lane_routes[j]가 해당 토큰의 보정 경로에 포함되면 True.

    Notes:
        - 시간 복잡도 절감을 위해 `lane_routes`는 집합으로 변환 후 멤버십 체크.
        - 연결성 보정은 `_prune_route_by_connectivity(route_ids, ids_in_lane_set)` 호출.
    """
    from typing import Dict, List, Optional, Set

    def _build_mask_for_token(
        npc_route_ids: Optional[List[str]],
        lane_routes: List[str],
        lane_routes_set: Set[str],
    ) -> List[bool]:
        """단일 토큰에 대한 lane 포함 마스크를 생성한다.

        Args:
            npc_route_ids (Optional[List[str]]): 보정된 NPC 경로 ID 시퀀스(가변 길이) 또는 None.
            lane_routes (List[str]): 길이 <= lane_num. 현재 샘플 lane의 roadblock ID.
            lane_routes_set (Set[str]): `lane_routes`의 집합 표현.

        Returns:
            List[bool]: 길이 = len(lane_routes). 포함 여부 불리언 마스크.
        """
        valid_lane_num: int = len(lane_routes)
        if npc_route_ids is None:
            return [False] * valid_lane_num

        # lane_routes 안에 실제 존재하는 후보만 필터링
        candidate_ids_in_lane: Set[str] = {
            rid for rid in npc_route_ids if rid in lane_routes_set
        }
        # 연속 구간 보정
        pruned_route_ids_list: List[str] = _prune_route_by_connectivity(
            npc_route_ids, candidate_ids_in_lane)
        pruned_route_ids_set: Set[str] = set(pruned_route_ids_list)
        a = [route in pruned_route_ids_set for route in lane_routes]
        return a

    lane_routes_set: Set[str] = set(lane_routes)
    car_token_to_lane_on_routes: Dict[str, List[bool]] = {}
    for token, npc_route_ids in car_token_to_rr_ids.items():
        # npc_route_ids: Optional[List[str]]
        car_token_to_lane_on_routes[token] = _build_mask_for_token(
            npc_route_ids, lane_routes, lane_routes_set)
    return car_token_to_lane_on_routes


def map_process(
        route_roadblock_ids,
        car_token_to_rr_ids: Dict[str, Optional[List[str]]],
        neighbor_track_token: List[Optional[str]],  # 길이: agent_num
        neighbor_agents_current,  # # (agent_num, 11)
        anchor_ego_state,
        elements_to_obj_polylines,
        elements_to_traffic_light,
        speed_limit_dict,
        lanes_roadblock_id_list,
        map_elements,
        max_elements,
        max_points):
    """
    This function process the data from the raw vector set map data.
    :param route_roadblock_ids: route road block ids.
    :param anchor_ego_state: ego current state.
    :param elements_to_obj_polylines: dictionary mapping feature name to polyline vector sets.
    :param elements_to_traffic_light: traffic light status of lanes.
    :param speed_limit_dict: speed limit of lanes.
    :param lanes_roadblock_id_list: road block ids of lanes.
    :param map_elements: Name of map features to extract.
    :param max_elements: clip the number of map elements.
    :param max_points: clip the number of point for each element.
    :return: dict of the map elements.
    """
    list_array_data = {}

    for feature_name, feature_coords in elements_to_obj_polylines.items():
        list_feature_coords = []

        # Pack coords into array list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(
                np.array(element_coords, dtype=np.float64))
        list_array_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into array list if it exists
        if feature_name in elements_to_traffic_light:
            list_feature_tl_data = []

            for element_tl_data in elements_to_traffic_light[
                    feature_name].to_vector():
                list_feature_tl_data.append(
                    np.array(element_tl_data, dtype=np.float64))
            list_array_data[
                f"traffic_light_data.{feature_name}"] = list_feature_tl_data
    """
    Vector set map data structure, including:
    coords: Dict[str, List[<np.ndarray: num_elements, num_points, 2>]].
            The (x, y) coordinates of each point in a map element across map elements per sample.
    traffic_light_data: Dict[str, List[<np.ndarray: num_elements, num_points, 4>]].
            One-hot encoding of traffic light status for each point in a map element across map elements per sample.
            Encoding: green [1, 0, 0, 0] yellow [0, 1, 0, 0], red [0, 0, 1, 0], unknown [0, 0, 0, 1]
    """

    array_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_elements:
        if f"coords.{feature_name}" in list_array_data:
            feature_coords = list_array_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_array_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_array_data else
                None)

            if feature_name == 'LANE':
                coords, left_coords, right_coords, tl_data, avails, lane_has_speed_limit_array, lane_speed_limit_array, lane_routes = _convert_lane_to_fixed_size(
                    anchor_ego_state,
                    feature_coords,
                    speed_limit_dict,
                    lanes_roadblock_id_list,
                    list_array_data[f"coords.LEFT_BOUNDARY"],
                    list_array_data[f"coords.RIGHT_BOUNDARY"],
                    feature_tl_data,
                    max_elements[feature_name],
                    max_points[feature_name],
                    traffic_light_encoding_dim if feature_name in [
                        VectorFeatureLayer.LANE.name,
                    ] else None,
                )
                left_coords = vector_set_coordinates_to_local_frame(
                    left_coords, avails, anchor_ego_state)
                right_coords = vector_set_coordinates_to_local_frame(
                    right_coords, avails, anchor_ego_state)
                array_output[
                    f"vector_set_map.coords.LEFT_BOUNDARY"] = left_coords
                array_output[
                    f"vector_set_map.coords.RIGHT_BOUNDARY"] = right_coords
                '''
                Get roadblock polygon
                '''
                lane_on_route = []
                pruned_lane_roadblock_ids = [
                    route for route in route_roadblock_ids
                    if route in lane_routes
                ]
                pruned_route_roadblock_ids = _prune_route_by_connectivity(
                    route_roadblock_ids, pruned_lane_roadblock_ids)

                # car_token_to_lane_on_routes: 길이 agent_num 보다 작거나 같음 (차량만 포함 가능)
                car_token_to_lane_on_routes: Dict[
                    str, List[bool]] = _compute_lane_on_npc_routes(
                        car_token_to_rr_ids, lane_routes)

                for route in lane_routes:
                    lane_on_route.append(route in pruned_route_roadblock_ids)

            elif feature_name == 'LEFT_BOUNDARY' or feature_name == 'RIGHT_BOUNDARY':
                continue

            coords = vector_set_coordinates_to_local_frame(
                coords, avails, anchor_ego_state)

            array_output[f"vector_set_map.coords.{feature_name}"] = coords
            array_output[
                f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                array_output[
                    f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data
    """
    Post-precoss the map elements to different map types. Each map type is a array with the following shape.
    """

    for feature_name in map_elements:
        if feature_name == "LANE":
            polylines = array_output[f'vector_set_map.coords.{feature_name}']
            left_boundary = array_output[f"vector_set_map.coords.LEFT_BOUNDARY"]
            right_boundary = array_output[
                f"vector_set_map.coords.RIGHT_BOUNDARY"]
            traffic_light_state = array_output[
                f'vector_set_map.traffic_light_data.{feature_name}']
            avails = array_output[
                f'vector_set_map.availabilities.{feature_name}']
            vector_map_lanes = _lane_polyline_process(polylines, left_boundary,
                                                      right_boundary, avails,
                                                      traffic_light_state)
            """
            agent_route_lane_order: shape = (agent_num, lane_num), dtype = `dtype`
            - 각 [i, j] 원소는:
                · j번 차선이 에이전트 i의 npc_route에서 가까운 순서로 몇 번째인지(0,1,2,...)를 나타냄
                · 해당 에이전트의 route가 아니면 -1
            """

            agent_route_lane_order = _select_token_and_ordered_npc_route_indices(
                car_token_to_lane_on_routes, neighbor_track_token,
                neighbor_agents_current, vector_map_lanes,
                max_elements["ROUTE_LANES"])
            if isinstance(agent_route_lane_order, np.ndarray):
                if agent_route_lane_order.dtype != np.int64:
                    agent_route_lane_order = agent_route_lane_order.astype(
                        np.int64)
            else:
                agent_route_lane_order = np.asarray(agent_route_lane_order,
                                                    dtype=np.int64)

        elif feature_name == "ROUTE_LANES":
            loc = 0
            # TODO: add has speed limit
            vector_map_route_lanes = np.zeros(
                (max_elements["ROUTE_LANES"], vector_map_lanes.shape[-2],
                 vector_map_lanes.shape[-1]),
                dtype=np.float32)
            route_lanes_speed_limit = np.zeros((max_elements["ROUTE_LANES"], 1),
                                               dtype=np.float32)
            route_lanes_has_speed_limit = np.zeros(
                (max_elements["ROUTE_LANES"], 1), dtype=np.bool_)
            for i in range(len(lane_on_route)):
                if lane_on_route[i] == True:
                    vector_map_route_lanes[loc] = vector_map_lanes[i]
                    route_lanes_speed_limit[loc] = lane_speed_limit_array[i]
                    route_lanes_has_speed_limit[
                        loc] = lane_has_speed_limit_array[i]
                    loc += 1
                if loc == max_elements["ROUTE_LANES"]:
                    break
        else:
            pass

    vector_map_output = {
        'lanes': vector_map_lanes,  # (lane_num, lane_len, 12)
        'lanes_speed_limit': lane_speed_limit_array,  # (lane_num, 1)
        'lanes_has_speed_limit': lane_has_speed_limit_array,  # (lane_num, 1)
        'route_lanes': vector_map_route_lanes,  # (route_num, lane_len, 12)
        'route_lanes_speed_limit': route_lanes_speed_limit,  # (route_num, 1)
        'route_lanes_has_speed_limit':
            route_lanes_has_speed_limit,  # (route_num, 1),
        "agent_route_lane_order":
            agent_route_lane_order  # (agent_num, lane_num) # -1 if not on route # <- np.int64 보장
    }

    return vector_map_output
