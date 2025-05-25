from typing import Optional, Dict, List
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.actor_state.state_representation import Point2D

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list, agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon


class DataProcessor(object):

    def __init__(self, config):
        self._agent_color_map: Dict[str, tuple] = {}
        self._color_cmap = plt.cm.get_cmap('tab20')
        self._next_color_idx = 0

        self._save_dir = getattr(config, "save_path", None)

        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100  #100  # [m] query radius scope relative to the current pose.

        self._map_features = [
            'LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'
        ]  # name of map features to be extracted.
        self._max_elements = {
            'LANE': config.lane_num,
            'LEFT_BOUNDARY': config.lane_num,
            'RIGHT_BOUNDARY': config.lane_num,
            'ROUTE_LANES': config.route_num
        }  # maximum number of elements to extract per feature layer.
        self._max_points = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.route_len
        }  # maximum number of points per feature to extract per feature layer.
        self.predicted_neighbor_num = config.predicted_neighbor_num
        self.step_index = 0

    # Use for inference
    def observation_adapter(self,
                            history_buffer,
                            traffic_light_data,
                            map_api,
                            route_roadblock_ids,
                            npc_route_roadblock_ids,
                            tokens_to_position, # TODO: 디버깅 용으로, 추후 삭제
                            device='cpu'):
        '''
        ego
        '''
        ego_agent_past = None  # inference no need ego_agent_past
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.rear_axle.heading
        ],
                                    dtype=np.float64)
        # tokens_to_position: Dict[str, Optional[np.ndarray]] w.r.t. world coordinates
        # TODO: change from world coordinates to ego coordinates
        ego_x, ego_y, ego_yaw = anchor_ego_state
        c, s = np.cos(ego_yaw), np.sin(ego_yaw)
        # Rotation matrix to go from world → ego frame: R(-yaw)
        # TODO: 디버깅 용으로, 추후 삭제
        R = np.array([[ c,  s],
                      [-s,  c]])
        tokens_to_position_ego: Dict[str, Optional[np.ndarray]] = {}
        for token, pos in tokens_to_position.items():
            if pos is None:
                tokens_to_position_ego[token] = None
            else:
                # pos may be shape (2,) or (N,2)
                shifted = pos - np.array([ego_x, ego_y])
                # apply rotation: world → ego
                if shifted.ndim == 1:
                    tokens_to_position_ego[token] = R.dot(shifted)
                else:
                    tokens_to_position_ego[token] = shifted.dot(R.T)
        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        (neighbor_agents_past, neighbor_agents_types,
         neighbor_agents_current_track_tokens
        ) = sampled_tracked_objects_to_array_list(observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(
            observation_buffer[-1])
        (_, neighbor_agents_past, _, static_objects,
         agents_track_token) = agent_past_process(
             ego_agent_past, neighbor_agents_past, neighbor_agents_types,
             neighbor_agents_current_track_tokens, self.num_agents,
             static_objects, static_objects_types, self.num_static,
             self.max_ped_bike, anchor_ego_state)
        neighbor_agents_current = neighbor_agents_past[
            :,
            -1,
        ]  # (32, 11)
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids)
        near_token_to_route_roadblock_ids: Dict[str, Optional[List[str]]] = {}
        near_token_to_raw_route_roadblock_ids: Dict[str,
                                                    Optional[List[str]]] = {}
        near_agents_current = []
        near_agent_count = 0
        near_agent_tokens = []
        for agent_idx, token in enumerate(
                agents_track_token):  # List[Optional[str]]
            if token is None:
                break
            if near_agent_count >= self.predicted_neighbor_num:
                break
            # 1) 토큰에 해당하는 객체 찾기
            last_obs = observation_buffer[-1]  # DetectionsTracks
            obj = next(
                (o for o in last_obs.tracked_objects if o.track_token == token),
                None)
            if abs(obj.center.x -
                   ego_x) > self._radius / 2 or abs(obj.center.y -
                                                    ego_y) > self._radius / 2:
                continue  # 사각형 밖이면 결과에 포함하지 않음

            # 3) 기존 npc_route_roadblock_ids[token] 에 보정 함수 적용
            # npc_route_roadblock_ids: Dict[str, Optional[List[str]]]
            # a_npc_route_roadblock_ids: Optional[List[str]]
            a_npc_route_roadblock_ids = npc_route_roadblock_ids.get(token)
            if a_npc_route_roadblock_ids is None:
                near_token_to_raw_route_roadblock_ids[token] = None
                near_token_to_route_roadblock_ids[token] = None
            else:
                near_token_to_raw_route_roadblock_ids[
                    token] = a_npc_route_roadblock_ids
                point = StateSE2(obj.center.x, obj.center.y, obj.center.heading)
                npc_state = SimpleNamespace(rear_axle=point)
                near_token_to_route_roadblock_ids[
                    token] = route_roadblock_correction(
                        npc_state, map_api, a_npc_route_roadblock_ids)
                # neighbor_agents_current: (32, 11)
            near_agents_current.append(neighbor_agents_current[agent_idx])
            near_agent_count += 1
            near_agent_tokens.append(token)
        # near_agents_current : (near_agent_count, 11)
        near_agents_current = np.array(near_agents_current)

        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius,
            traffic_light_data)
        vector_map, lane_on_raw_npc_routes = map_process(
            route_roadblock_ids, near_token_to_route_roadblock_ids,
            near_token_to_raw_route_roadblock_ids, near_agents_current,
            anchor_ego_state, coords, traffic_light_data, speed_limit,
            lane_route, self._map_features, self._max_elements,
            self._max_points)

        data = {
            "neighbor_agents_past":
                neighbor_agents_past[:, -21:],
            "ego_current_state":
                np.array(
                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32
                ),  # ego centric x, y, cos, sin, vx, vy, ax, ay, steering angle, yaw rate, we only use x, y, cos, sin during inference
            "static_objects":
                static_objects
        }
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)

        # 시각화 코드 호출
        for idx, token in enumerate(near_agent_tokens):
            type_one_hot_vector = near_agents_current[idx, 8]
            if type_one_hot_vector == 1:
                self._draw_step(
                    lanes=vector_map['lanes'],  # (70, 20, 12)
                    neighbor_agents=neighbor_agents_current,  # (32, 11)
                    near_agent=near_agents_current[idx],  # (11)
                    near_agent_token=token,
                    npc_route=vector_map['npc_route_lanes']
                    [idx],  # (25, 20, 12)
                    ego_state=ego_state,
                    lane_on_raw_npc_routes=lane_on_raw_npc_routes[
                        idx],  # List[bool]
                    positions = tokens_to_position_ego[token],
                    save_dir="visualization",
                    step_index=self.step_index)
        self.step_index += 1
        return data

    def _draw_step(self, lanes: np.ndarray, neighbor_agents: np.ndarray,
                   near_agent: np.ndarray, near_agent_token: str,
                   npc_route: np.ndarray, ego_state: StateSE2,
                   lane_on_raw_npc_routes: List[bool],
                   positions: Optional[np.ndarray], # (n, 2) or None
        save_dir: Optional[str],
                   step_index: int) -> None:
        """
        한 스텝 분량의 맵과 에이전트를 시각화하여 JPG 파일로 저장합니다.

        Args:
            lanes: 전체 차선 벡터 배열, shape=(N, P, D). (70, 20, 12)
            neighbor_agents: 주변 32대 차량 상태 배열, shape=(M, 11). # (32, 11)
            near_agent: 핵심 npc 차량 상태 배열, shape=(11). # (11)
            npc_route: (25 , 20, 12) # len(npc_routes) == len(near_agents)
            ego_state: ego의 현재 상태(StateSE2 객체).
            save_dir: 그림 저장 디렉토리.
            token: 시나리오 토큰(파일명 식별자).
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        # 1. 전체 차선 (흰선)
        for idx, lane in enumerate(lanes):
            # lane: (20, 12)
            valid = ~np.all(lane[:, :2] == 0, axis=1)  # valid: (20,)
            pts = lane[valid]
            if pts.shape[0] < 2:
                continue
            center = pts[:, :2]
            left = center + pts[:, 4:6]
            right = center + pts[:, 6:8]
            if lane_on_raw_npc_routes[idx] == True:
                color_ = 'red'
                linewidth = 1
                ax.plot(center[:, 0],
                        center[:, 1],
                        '--',
                        color=color_,
                        linewidth=linewidth)
            else:
                color_ = 'white'
                linewidth = 1

            ax.plot(left[:, 0],
                    left[:, 1],
                    '-',
                    color=color_,
                    linewidth=linewidth)
            ax.plot(right[:, 0],
                    right[:, 1],
                    '-',
                    color=color_,
                    linewidth=linewidth)
        # 5-6. Ego 차량 및 경로 (cyan)
        # 5. Ego 차량 (cyan, ego 프레임이므로 회전 0)
        ego_w, ego_l = 2.0, 4.5
        # rear_axle(0,0) 기준으로, 좌하단 모서리만 이동
        ego_rect = Rectangle(
            (-ego_l / 2, -ego_w / 2),  # left-bottom corner
            ego_l,
            ego_w,  # width=length, height=width
            edgecolor='cyan',
            facecolor='none',
            linewidth=2)
        ax.add_patch(ego_rect)

        # 400m, 200m 사각형
        # 6. 400m/200m 세계좌표축 정렬 사각형
        yaw = ego_state.rear_axle.heading  # ego 프레임→세계축 회전각
        for size in (int(self._radius * 2), int(self._radius * 2)):
            half = size / 2
            # ego 프레임에서의 사각형 중심이 (0,0)인 네 꼭짓점
            corners = np.array([[-half, -half], [-half, half], [half, half],
                                [half, -half]])  # shape=(4,2)

            # 세계축 정렬을 위해 -yaw 만큼 회전
            c, s = np.cos(-yaw), np.sin(-yaw)
            rot = np.array([[c, -s], [s, c]])
            world_corners = corners.dot(rot.T)

            square = Polygon(world_corners,
                             closed=True,
                             edgecolor='magenta',
                             facecolor='none',
                             linewidth=1)
            ax.add_patch(square)

        # 2. 주변 차량 (흰색 사각형)
        for state in neighbor_agents:  # (32, 11)
            x, y = state[0], state[1]
            cos_yaw, sin_yaw = state[2], state[3]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            width, length = state[6], state[7]
            # 중심->좌하단 오프셋
            dx, dy = -length / 2, -width / 2
            ox = dx * np.cos(yaw) - dy * np.sin(yaw)
            oy = dx * np.sin(yaw) + dy * np.cos(yaw)
            rect = Rectangle((x + ox, y + oy),
                             length,
                             width,
                             angle=np.degrees(yaw),
                             color='white')
            ax.add_patch(rect)

        # (3-4) 핵심 npc 차량 및 경로: 토큰별 색 고정 사용
        # 토큰별로 컬러 매핑 (처음 등장시 할당)
        if near_agent_token not in self._agent_color_map:
            color = self._color_cmap(self._next_color_idx % 20)
            self._agent_color_map[near_agent_token] = color
            self._next_color_idx += 1
        else:
            color = self._agent_color_map[near_agent_token]

        # 차량 박스
        x, y = near_agent[0], near_agent[1]
        cos_yaw, sin_yaw = near_agent[2], near_agent[3]
        yaw = np.arctan2(sin_yaw, cos_yaw)
        width, length = near_agent[6], near_agent[7]
        dx, dy = -length / 2, -width / 2
        ox = dx * np.cos(yaw) - dy * np.sin(yaw)
        oy = dx * np.sin(yaw) + dy * np.cos(yaw)
        npc_rect = Rectangle((x + ox, y + oy),
                             length,
                             width,
                             angle=np.degrees(yaw),
                             color=color)
        ax.add_patch(npc_rect)

        # NPC 경로
        valid_elems = ~np.all(npc_route == 0, axis=(1, 2))
        for lane in npc_route[valid_elems]:
            valid = ~np.all(lane[:, :2] == 0, axis=1)
            pts = lane[valid]
            if pts.shape[0] < 2:
                continue
            center = pts[:, :2]
            left = center + pts[:, 4:6]
            right = center + pts[:, 6:8]
            # ax.plot(center[:,0], center[:,1], '--', color=color, linewidth=1)
            ax.plot(left[:, 0], left[:, 1], '-', color=color, linewidth=1)
            ax.plot(right[:, 0], right[:, 1], '-', color=color, linewidth=1)

        # 7. positions 경로 (파란 실선)
        if positions is not None and positions.ndim == 2 and positions.shape[1] == 2:
            # positions: shape=(T,2)
            ax.plot(positions[:, 0], positions[:, 1],
                    '-', color='blue', linewidth=1, label='positions')

        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.title(f"Visualization Step: {step_index}", color='white')
        save_dir = os.path.join(save_dir, near_agent_token)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"vis_{step_index}.jpg"),
                        dpi=150,
                        facecolor='black')
        plt.close(fig)

    def _draw_step_all(self, lanes: np.ndarray, neighbor_agents: np.ndarray,
                       near_agents: np.ndarray, near_agent_tokens: List[str],
                       npc_routes: List[np.ndarray], ego_route: np.ndarray,
                       ego_state: StateSE2, save_dir: Optional[str],
                       token: int) -> None:
        """
        한 스텝 분량의 맵과 에이전트를 시각화하여 JPG 파일로 저장합니다.

        Args:
            lanes: 전체 차선 벡터 배열, shape=(N, P, D). (70, 20, 12)
            neighbor_agents: 주변 32대 차량 상태 배열, shape=(M, 11). # (32, 11)
            near_agents: 핵심 npc 차량 상태 배열, shape=(K, 11). # (near_agent_count, 11)
            npc_routes: 각 핵심 npc 차량 경로 차선 배열 리스트. # List[(25 , 20, 12)] # len(npc_routes) == len(near_agents)
            ego_route: ego 차량 경로 차선 벡터, shape=(R, P, D).  # (25, 20, 12)
            ego_state: ego의 현재 상태(StateSE2 객체).
            save_dir: 그림 저장 디렉토리.
            token: 시나리오 토큰(파일명 식별자).
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor('black')
        # 1. 전체 차선 (흰선)
        for lane in lanes:
            # lane: (20, 12)
            valid = ~np.all(lane[:, :2] == 0, axis=1)  # valid: (20,)
            pts = lane[valid]
            if pts.shape[0] < 2:
                continue
            center = pts[:, :2]
            left = center + pts[:, 4:6]
            right = center + pts[:, 6:8]
            ax.plot(center[:, 0],
                    center[:, 1],
                    '--',
                    color='white',
                    linewidth=1)
            ax.plot(left[:, 0], left[:, 1], '-', color='white', linewidth=1)
            ax.plot(right[:, 0], right[:, 1], '-', color='white', linewidth=1)
        # 5-6. Ego 차량 및 경로 (cyan)
        # 5. Ego 차량 (cyan, ego 프레임이므로 회전 0)
        ego_w, ego_l = 2.0, 4.5
        # rear_axle(0,0) 기준으로, 좌하단 모서리만 이동
        ego_rect = Rectangle(
            (-ego_l / 2, -ego_w / 2),  # left-bottom corner
            ego_l,
            ego_w,  # width=length, height=width
            edgecolor='cyan',
            facecolor='none',
            linewidth=2)
        ax.add_patch(ego_rect)
        for lane in ego_route:  # lane.shape == (20,12)
            # (x,y) 둘 다 0인 포인트만 골라서 제외
            valid = ~np.all(lane[:, :2] == 0, axis=1)  # valid.shape == (20,)
            pts = lane[valid]  # pts.shape == (num_valid, 12)
            if pts.shape[0] < 2:
                continue
            center = pts[:, :2]
            left = center + pts[:, 4:6]
            right = center + pts[:, 6:8]
            ax.plot(center[:, 0], center[:, 1], '--', color='cyan', linewidth=1)
            ax.plot(left[:, 0], left[:, 1], '-', color='cyan', linewidth=1)
            ax.plot(right[:, 0], right[:, 1], '-', color='cyan', linewidth=1)

        # 400m, 200m 사각형
        # 6. 400m/200m 세계좌표축 정렬 사각형
        yaw = ego_state.rear_axle.heading  # ego 프레임→세계축 회전각
        for size in (int(self._radius * 2), int(self._radius)):
            half = size / 2
            # ego 프레임에서의 사각형 중심이 (0,0)인 네 꼭짓점
            corners = np.array([[-half, -half], [-half, half], [half, half],
                                [half, -half]])  # shape=(4,2)

            # 세계축 정렬을 위해 -yaw 만큼 회전
            c, s = np.cos(-yaw), np.sin(-yaw)
            rot = np.array([[c, -s], [s, c]])
            world_corners = corners.dot(rot.T)

            square = Polygon(world_corners,
                             closed=True,
                             edgecolor='magenta',
                             facecolor='none',
                             linewidth=2)
            ax.add_patch(square)

        # 2. 주변 차량 (흰색 사각형)
        for state in neighbor_agents:  # (32, 11)
            x, y = state[0], state[1]
            cos_yaw, sin_yaw = state[2], state[3]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            width, length = state[6], state[7]
            # 중심->좌하단 오프셋
            dx, dy = -length / 2, -width / 2
            ox = dx * np.cos(yaw) - dy * np.sin(yaw)
            oy = dx * np.sin(yaw) + dy * np.cos(yaw)
            rect = Rectangle((x + ox, y + oy),
                             length,
                             width,
                             angle=np.degrees(yaw),
                             color='white')
            ax.add_patch(rect)

        # (3-4) 핵심 npc 차량 및 경로: 토큰별 색 고정 사용
        for idx, (state, tk) in enumerate(zip(near_agents, near_agent_tokens)):
            # 토큰별로 컬러 매핑 (처음 등장시 할당)
            if tk not in self._agent_color_map:
                color = self._color_cmap(self._next_color_idx % 20)
                self._agent_color_map[tk] = color
                self._next_color_idx += 1
            else:
                color = self._agent_color_map[tk]

            # 차량 박스
            x, y = state[0], state[1]
            cos_yaw, sin_yaw = state[2], state[3]
            yaw = np.arctan2(sin_yaw, cos_yaw)
            width, length = state[6], state[7]
            dx, dy = -length / 2, -width / 2
            ox = dx * np.cos(yaw) - dy * np.sin(yaw)
            oy = dx * np.sin(yaw) + dy * np.cos(yaw)
            npc_rect = Rectangle((x + ox, y + oy),
                                 length,
                                 width,
                                 angle=np.degrees(yaw),
                                 color=color)
            ax.add_patch(npc_rect)

            # NPC 경로
            route = npc_routes[idx]
            valid_elems = ~np.all(route == 0, axis=(1, 2))
            for lane in route[valid_elems]:
                valid = ~np.all(lane[:, :2] == 0, axis=1)
                pts = lane[valid]
                if pts.shape[0] < 2:
                    continue
                center = pts[:, :2]
                left = center + pts[:, 4:6]
                right = center + pts[:, 6:8]
                ax.plot(center[:, 0],
                        center[:, 1],
                        '--',
                        color=color,
                        linewidth=1)
                ax.plot(left[:, 0], left[:, 1], '-', color=color, linewidth=1)
                ax.plot(right[:, 0], right[:, 1], '-', color=color, linewidth=1)

        # for size in (400, 200):
        #     half = size / 2
        #     square = Rectangle((ex - half, ey - half),
        #                        size,
        #                        size,
        #                        edgecolor='magenta',
        #                        facecolor='none',
        #                        linewidth=2)
        #     ax.add_patch(square)

        ax.set_aspect('equal', 'box')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        plt.title(f"Visualization Step: {token}", color='white')
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, f"vis_{token}.jpg"),
                        dpi=150,
                        facecolor='black')
        plt.close(fig)

    # Use for data preprocess
    def work(self, scenarios):

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api
            '''
            ego & agents past
            '''
            ego_state = scenario.initial_ego_state
            ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
            anchor_ego_state = np.array([
                ego_state.rear_axle.x, ego_state.rear_axle.y,
                ego_state.rear_axle.heading
            ],
                                        dtype=np.float64)
            ego_agent_past, time_stamps_past = get_ego_past_array_from_scenario(
                scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses)
            ]
            sampled_past_observations = past_tracked_objects + [
                present_tracked_objects
            ]
            neighbor_agents_past, neighbor_agents_types, _ = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)

            static_objects, static_objects_types = sampled_static_objects_to_array_list(
                present_tracked_objects)

            ego_agent_past, neighbor_agents_past, neighbor_indices, static_objects, _ = \
                agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids)

            coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
                map_api, self._map_features, ego_coords, self._radius,
                traffic_light_data)

            vector_map = map_process(route_roadblock_ids, anchor_ego_state,
                                     coords, traffic_light_data, speed_limit,
                                     lane_route, self._map_features,
                                     self._max_elements, self._max_points)
            '''
            ego & agents future
            '''
            ego_agent_future = get_ego_future_array_from_scenario(
                scenario, ego_state, self.num_future_poses,
                self.future_time_horizon)

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0,
                    time_horizon=self.future_time_horizon,
                    num_samples=self.num_future_poses)
            ]

            sampled_future_observations = [present_tracked_objects
                                          ] + future_tracked_objects
            (future_tracked_objects_array_list, _,
             _) = sampled_tracked_objects_to_array_list(
                 sampled_future_observations)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state, future_tracked_objects_array_list,
                self.num_agents, neighbor_indices)
            '''
            ego current
            '''
            ego_current_state = calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)

            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
                "static_objects": static_objects
            }
            data.update(vector_map)

            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)
