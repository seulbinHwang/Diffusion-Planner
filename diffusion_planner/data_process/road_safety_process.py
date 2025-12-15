"""
도로 안전 요소를 추출하는 유틸리티.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, StopLineType
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario


def _to_polygon(shape: Polygon | MultiPolygon) -> Polygon:
    """
    다각형 형태로 변환한다.

    Args:
        shape (Polygon | MultiPolygon): 지도에서 받아온 도형 정보.

    Returns:
        Polygon: 가장 큰 면적을 가지는 단일 다각형.
    """
    if isinstance(shape, MultiPolygon):
        polygons: List[Polygon] = [poly for poly in shape.geoms]
        polygons.sort(key=lambda poly: poly.area, reverse=True)
        return polygons[0]
    return shape


def _sample_polygon_boundary(
    polygon: Polygon,
    num_samples: int,
) -> np.ndarray:
    """
    다각형 외곽을 따라 등간격으로 점을 만든다.

    Args:
        polygon (Polygon): 대상 다각형.
        num_samples (int): 생성할 점 개수.

    Returns:
        np.ndarray: 모양이 (num_samples, 2) 인 점 좌표.
    """
    exterior = polygon.exterior
    boundary = exterior if exterior.length > 0 else polygon.boundary
    distances = np.linspace(0.0, boundary.length, num_samples, endpoint=False)
    samples = np.array(
        [list(boundary.interpolate(dist).coords)[0] for dist in distances],
        dtype=float)
    return samples


def _to_ego_coordinates(points: np.ndarray,
                        ego_cur_pose_np: Sequence[float]) -> np.ndarray:
    """
    전역 좌표를 자차 기준 좌표계로 변환한다.

    Args:
        points (np.ndarray): 모양이 (P, 2) 인 전역 좌표 배열.
        ego_cur_pose_np (Sequence[float]): (x, y, yaw_rad) 정보를 담은 시퀀스.

    Returns:
        np.ndarray: 모양이 (P, 2) 인 자차 기준 좌표 배열.
    """
    ego_x, ego_y, ego_yaw = ego_cur_pose_np
    translated = points - np.array([[ego_x, ego_y]])
    cos_yaw = np.cos(-ego_yaw)
    sin_yaw = np.sin(-ego_yaw)
    rotation = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
    ego_frame = translated @ rotation.T
    return ego_frame


def _is_within_radius(points: np.ndarray, ego_cur_pose_np: Sequence[float],
                      radius: Optional[float]) -> bool:
    """
    주어진 점 중 반경 제한을 만족하는 것이 있는지 확인한다.

    Args:
        points (np.ndarray): 모양이 (P, 2) 인 전역 좌표 배열.
        ego_cur_pose_np (Sequence[float]): (x, y, yaw_rad) 정보를 담은 시퀀스.
        radius (Optional[float]): 허용 반경. None이면 제한 없음.

    Returns:
        bool: 반경을 만족하는 점이 있으면 True.
    """
    if radius is None:
        return True
    ego_x, ego_y, _ = ego_cur_pose_np
    deltas = points - np.array([[ego_x, ego_y]])
    distances = np.linalg.norm(deltas, axis=1)
    return bool(np.any(distances <= radius))


def _load_polygons_from_layer(map_api, layer_name: str) -> List[Polygon]:
    """
    지도에서 특정 벡터 레이어의 도형을 불러온다.

    Args:
        map_api: NuPlanMap 인스턴스.
        layer_name (str): 지도 벡터 레이어 이름.

    Returns:
        List[Polygon]: 레이어에 포함된 다각형 목록.
    """
    layer = map_api._maps_db.load_vector_layer(
        map_api._map_name, layer_name)  # type: ignore[attr-defined]
    shapes: List[Polygon] = []
    for geom in layer.geometry:
        polygon = _to_polygon(geom)
        shapes.append(polygon)
    return shapes


def _extract_polygons_from_objects(objects: Iterable) -> List[Polygon]:
    """
    맵 객체 목록에서 다각형만 추출한다.

    Args:
        objects (Iterable): polygon 속성을 가진 맵 객체 모음.

    Returns:
        List[Polygon]: 추출한 다각형 목록.
    """
    polygons: List[Polygon] = []
    for obj in objects:
        if hasattr(obj, "polygon"):
            polygons.append(_to_polygon(obj.polygon))
    return polygons


def extract_crosswalk_points(
        scenario: NuPlanScenario,
        ego_cur_pose_np: np.ndarray,  # (3,),
        safety_len: int,
        radius: Optional[float] = None) -> np.ndarray:
    """
    자차 주변 혹은 전체 크로스워크를 등간격 점으로 반환한다.

    Args:
        scenario (NuPlanScenario): 대상 시나리오.
        ego_cur_pose_np (Tuple[float, float, float]): (x, y, yaw_rad) 자차 포즈.
        radius (Optional[float]): 관심 반경. None이면 전부 반환.

    Returns:
        np.ndarray: 모양이 (N, 20, 2) 인 자차 기준 크로스워크 점들.
    """
    map_api = scenario.map_api
    polygons: List[Polygon]
    if radius is not None:
        proximal = map_api.get_proximal_map_objects(
            Point2D(ego_cur_pose_np[0], ego_cur_pose_np[1]), radius,
            [SemanticMapLayer.CROSSWALK])
        polygons = _extract_polygons_from_objects(
            proximal.get(SemanticMapLayer.CROSSWALK, []))
    else:
        crosswalk_layer = map_api._get_vector_map_layer(
            SemanticMapLayer.CROSSWALK)  # type: ignore[attr-defined]
        polygons = [_to_polygon(geom) for geom in crosswalk_layer.geometry]

    sampled_list: List[np.ndarray] = []
    for polygon in polygons:
        sampled = _sample_polygon_boundary(polygon, safety_len)
        if _is_within_radius(sampled, ego_cur_pose_np, radius):
            ego_points = _to_ego_coordinates(sampled, ego_cur_pose_np)
            sampled_list.append(ego_points)

    if not sampled_list:
        return np.zeros((0, safety_len, 2))
    return np.stack(sampled_list, axis=0)


def extract_stop_sign_points(
        scenario: NuPlanScenario,
        ego_cur_pose_np: np.ndarray,  # (3,)
        safety_len: int,
        radius: Optional[float] = None) -> np.ndarray:
    """
    자차 주변 혹은 전체 정지표지 정지선을 등간격 점으로 반환한다.

    Args:
        scenario (NuPlanScenario): 대상 시나리오.
        ego_cur_pose_np ([float, float, float]): (x, y, yaw_rad) 자차 포즈.
        radius (Optional[float]): 관심 반경. None이면 전부 반환.

    Returns:
        np.ndarray: 모양이 (N, 20, 2) 인 자차 기준 정지선 점들.
    """
    map_api = scenario.map_api
    polygons: List[Polygon] = []

    if radius is not None:
        proximal = map_api.get_proximal_map_objects(
            Point2D(ego_cur_pose_np[0], ego_cur_pose_np[1]), radius,
            [SemanticMapLayer.STOP_LINE])
        stop_lines = proximal.get(SemanticMapLayer.STOP_LINE, [])
        filtered = [
            obj for obj in stop_lines
            if getattr(obj, "stop_line_type", None) == StopLineType.STOP_SIGN
        ]
        polygons = _extract_polygons_from_objects(filtered)
    else:
        stop_layer = map_api._get_vector_map_layer(
            SemanticMapLayer.STOP_LINE)  # type: ignore[attr-defined]
        for geom, stop_type in zip(stop_layer.geometry,
                                   stop_layer["stop_polygon_type_fid"]):
            if stop_type == StopLineType.STOP_SIGN.value:
                polygons.append(_to_polygon(geom))

    sampled_list: List[np.ndarray] = []
    for polygon in polygons:
        sampled = _sample_polygon_boundary(polygon, safety_len)
        if _is_within_radius(sampled, ego_cur_pose_np, radius):
            ego_points = _to_ego_coordinates(sampled, ego_cur_pose_np)
            sampled_list.append(ego_points)

    if not sampled_list:
        return np.zeros((0, safety_len, 2))
    return np.stack(sampled_list, axis=0)
