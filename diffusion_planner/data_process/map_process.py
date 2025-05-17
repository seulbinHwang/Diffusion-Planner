"""
Module: Map Data Preprocessing Functions
Description: This module contains functions for Map related data processing.

Categories:
    1. Get lanes, speed limit, traffic light and lane's roadblock ids
    2. Get maps array for model input
"""

from typing import List, Dict, Tuple, Set
import numpy as np
from shapely import LineString

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.nuplan_map.utils import get_distance_between_map_object_and_point
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, SemanticMapLayer
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines, VectorFeatureLayer, LaneSegmentLaneIDs,
    VectorFeatureLayerMapping, LaneSegmentTrafficLightData,
    get_traffic_light_encoding, get_map_object_polygons)

from diffusion_planner.data_process.utils import vector_set_coordinates_to_local_frame


# =====================
# 1. Get lanes, speed limit, traffic light and lane's roadblock ids
# =====================
def _get_lane_polylines(
    map_api: AbstractMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines,
           LaneSegmentLaneIDs]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
        lane_speed_limit: lane's speed limit.
        lane_has_speed_limit: whether lane has speed limit.
        lane_roadblock_ids: lane's roadblock ids.
    """
    lanes_mid: List[List[Point2D]] = [
    ]  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_left: List[List[Point2D]] = [
    ]  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_right: List[List[Point2D]] = [
    ]  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lane_ids: List[str] = []  # shape: [num_lanes]
    lane_speed_limit = []
    lane_has_speed_limit = []
    lane_roadblock_ids = []
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    map_objects = []

    for layer_name in layer_names:
        map_objects += layers[layer_name]
    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: float(
        get_distance_between_map_object_and_point(point, map_obj)))

    for map_obj in map_objects:
        # center lane
        baseline_path_polyline = [
            Point2D(node.x, node.y)
            for node in map_obj.baseline_path.discrete_path
        ]
        lanes_mid.append(baseline_path_polyline)

        # boundaries
        lanes_left.append([
            Point2D(node.x, node.y)
            for node in map_obj.left_boundary.discrete_path
        ])
        lanes_right.append([
            Point2D(node.x, node.y)
            for node in map_obj.right_boundary.discrete_path
        ])

        # lane ids
        lane_ids.append(map_obj.id)

        # speed limit
        if map_obj.speed_limit_mps is None:
            lane_speed_limit.append(0.0)
            lane_has_speed_limit.append(False)
        else:
            lane_speed_limit.append(map_obj.speed_limit_mps)
            lane_has_speed_limit.append(True)

        lane_roadblock_ids.append(map_obj.get_roadblock_id())

    return (MapObjectPolylines(lanes_mid), MapObjectPolylines(lanes_left),
            MapObjectPolylines(lanes_right), LaneSegmentLaneIDs(lane_ids),
            lane_speed_limit, lane_has_speed_limit, lane_roadblock_ids)


def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str,
                                               LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
        speed_limit: Lane's speed limit
        lane_route: route lane
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    speed_limit = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"
            )

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids, lane_speed_limit, lane_has_speed_limit, lane_route = _get_lane_polylines(
            map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid
        speed_limit['lane_has_speed_limit'] = np.array(lane_has_speed_limit,
                                                       dtype=np.bool_)
        speed_limit['lane_speed_limit'] = np.array(lane_speed_limit,
                                                   dtype=np.float32)

        # lane traffic light data
        traffic_light_data[
            VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
                lane_ids, traffic_light_status_data)

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(
                lanes_right.polylines)

    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers(
        ):
            polygons = get_map_object_polygons(
                map_api, point, radius,
                VectorFeatureLayerMapping.semantic_map_layer(feature_layer))
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data, speed_limit, lane_route


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


def _convert_lane_to_fixed_size(ego_pose, feature_coords, speed_limit,
                                lane_route, left_boundary, right_boundary,
                                feature_tl_data, max_elements, max_points,
                                traffic_light_encoding_dim):

    if feature_tl_data is not None and len(feature_coords) != len(
            feature_tl_data):
        raise ValueError(
            f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}"
        )

    lane_has_speed_limit = speed_limit['lane_has_speed_limit']
    lane_speed_limit = speed_limit['lane_speed_limit']

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
        lane_routes.append(lane_route[element_idx[0]])

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

- roadblock_ids = pruned_lane_roadblock_ids: List[str]
  - lane_routes 중, route_roadblock_ids 인 친구들
  - ego 거리 순으로 선별되어 있음
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


def map_process(route_roadblock_ids, anchor_ego_state, coords,
                traffic_light_data, speed_limit, lane_route, map_features,
                max_elements, max_points):
    """
    This function process the data from the raw vector set map data.
    :param route_roadblock_ids: route road block ids.
    :param anchor_ego_state: ego current state.
    :param coords: dictionary mapping feature name to polyline vector sets.
    :param traffic_light_data: traffic light status of lanes.
    :param speed_limit: speed limit of lanes.
    :param lane_route: road block ids of lanes.
    :param map_features: Name of map features to extract.
    :param max_elements: clip the number of map elements.
    :param max_points: clip the number of point for each element.
    :return: dict of the map elements.
    """
    list_array_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into array list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(
                np.array(element_coords, dtype=np.float64))
        list_array_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into array list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
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

    for feature_name in map_features:
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
                    speed_limit,
                    lane_route,
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

    for feature_name in map_features:
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

    vector_map_output = {'lanes': vector_map_lanes,
                         'lanes_speed_limit': lane_speed_limit_array,
                         'lanes_has_speed_limit': lane_has_speed_limit_array, \
                         'route_lanes': vector_map_route_lanes,
                         'route_lanes_speed_limit': route_lanes_speed_limit,
                         'route_lanes_has_speed_limit': route_lanes_has_speed_limit}

    return vector_map_output
