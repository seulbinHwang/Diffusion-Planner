import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # GUI 백엔드 사용 안함 (메모리 절약)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import wandb
import os
from typing import Dict, Tuple, Union  # NEW: type annotation 추가
from nuplan.common.actor_state.state_representation import Point2D

# matplotlib 설정 추가
plt.rcParams['figure.max_open_warning'] = 0  # 경고 메시지 비활성화
matplotlib.rcParams['figure.max_open_warning'] = 0

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_static_objects_to_array_list, agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs


class DataProcessor(object):

    def __init__(self, config):

        self._save_dir = getattr(config, "save_path", None)
        self.config = config
        self.past_time_horizon = 2  # [seconds]
        self.num_past_poses = 10 * self.past_time_horizon
        self.future_time_horizon = 8  # [seconds]
        self.num_future_poses = 10 * self.future_time_horizon

        self.num_agents = config.agent_num
        self.num_static = config.static_objects_num
        self.max_ped_bike = 10  # Limit the number of pedestrians and bicycles in the agent.
        self._radius = 100  # [m] query radius scope relative to the current pose.

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
        # wandb 사용 여부를 한 곳에서만 판단
        if getattr(config, "use_wandb", True) and wandb.run is None:
            # 컨트롤러 run id를 같은 group 으로 묶어 두면 대시보드가 깔끔
            wandb.init(
                project=getattr(config, "wandb_project", "Diffusion-Planner"),
                entity=getattr(config, "wandb_entity", None),
                group=getattr(config, "wandb_group", None),  # ← 컨트롤러 ID
                job_type="preprocess-worker",
                name=f"{config.name}-worker-{os.getpid()}",
                mode=getattr(config, "wandb_mode", "online"),
                reinit=True,  # fork 안전
                settings=wandb.Settings(start_method="fork"),
            )
        self._wandb_enabled = wandb.run is not None

    # Use for inference
    def observation_adapter(self,
                            history_buffer,
                            traffic_light_data,
                            map_api,
                            route_roadblock_ids,
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
        '''
        neighbor
        '''
        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        neighbor_agents_past, neighbor_agents_types = sampled_tracked_objects_to_array_list(
            observation_buffer)
        static_objects, static_objects_types = sampled_static_objects_to_array_list(
            observation_buffer[-1])
        _, neighbor_agents_past, _, static_objects = \
            agent_past_process(ego_agent_past, neighbor_agents_past, neighbor_agents_types, self.num_agents, static_objects, static_objects_types, self.num_static, self.max_ped_bike, anchor_ego_state)
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        route_roadblock_ids = route_roadblock_correction(
            ego_state, map_api, route_roadblock_ids)
        coords, traffic_light_data, speed_limit, lane_route = get_neighbor_vector_set_map(
            map_api, self._map_features, ego_coords, self._radius,
            traffic_light_data)
        vector_map = map_process(route_roadblock_ids, anchor_ego_state, coords,
                                 traffic_light_data, speed_limit, lane_route,
                                 self._map_features, self._max_elements,
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

        return data

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
                                        dtype=np.float64) # shape (3,)
            # ego_agent_past: np (21, 7) # x, y, theta, vx, vy, width, length
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
            neighbor_agents_past, neighbor_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)

            (static_objects, static_objects_types
            ) = sampled_static_objects_to_array_list(present_tracked_objects)

            (ego_agent_past, neighbor_agents_past,
             neighbor_indices, static_objects) = agent_past_process(
                 ego_agent_past, neighbor_agents_past, neighbor_agents_types,
                 self.num_agents, static_objects, static_objects_types,
                 self.num_static, self.max_ped_bike, anchor_ego_state)
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids)

            (coords, traffic_light_data, speed_limit,
             lane_route) = get_neighbor_vector_set_map(map_api,
                                                       self._map_features,
                                                       ego_coords, self._radius,
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
                self.future_time_horizon)  # (T, 7)
            # x, y, cos(yaw), sin(yaw), vx, vy, width, length, agent_type
            ego_agent_future_processed = np.zeros(
                (ego_agent_future.shape[0], 11), dtype=np.float32)
            ego_agent_future_processed[:, :2] = ego_agent_future[:, :2]
            ego_agent_future_processed[:, 2] = np.cos(ego_agent_future[:, 2])
            ego_agent_future_processed[:, 3] = np.sin(ego_agent_future[:, 2])
            ego_agent_future_processed[:, 4:8] = ego_agent_future[:, 3:]
            ego_agent_future_processed[:, 8] = 1.0  # ego is always vehicle
            ego_agent_future = ego_agent_future_processed

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
            (future_tracked_objects_array_list,
             _) = sampled_tracked_objects_to_array_list(
                 sampled_future_observations)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state, future_tracked_objects_array_list,
                self.num_agents, neighbor_indices)
            '''
            ego current
            
            
            '''
            # ego_current_state = calculate_additional_ego_states(
            #     ego_agent_past, time_stamps_past)
            # ego_agent_past: (T, 7) -> (T=21, 11)
            ego_agent_past, ego_current_state = calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)
            T, D = ego_agent_past.shape
            assert T == self.num_past_poses + 1, "Ego agent past states should have T+1 time steps"
            assert D == 11, "Ego agent past states should have 8 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length)"

            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_agent_past": ego_agent_past,
                "ego_current_state": ego_current_state,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
                "static_objects": static_objects
            }
            data.update(vector_map)

            # 디버깅용 그림 그리기
            if  self._wandb_enabled or self.config.save_image:
                print("Visualizing scenario:", map_name, token)
                self._visualize_scenario(ego_agent_past, neighbor_agents_past,
                                         vector_map['lanes'], map_name, token)

            self.save_to_disk(self._save_dir, data)

    def _visualize_scenario(
        self,
        ego_agent_past: np.ndarray,  # (T=21, 11)
        neighbor_agents_past: np.ndarray,  # (P=32, T=21, 11)
        lanes: np.ndarray,  # (N=70, V=25, 12)
        map_name: str,
        token: str,
    ) -> None:
        """시나리오 데이터를 시각화하여 파일로 저장합니다.

        Args:
            ego_agent_past: (T=21, 11) ego vehicle 과거 상태
                - x, y, cos(yaw), sin(yaw), v_x, v_y, width, length
            neighbor_agents_past: (P=32, T=21, 11) neighbor agents 과거 상태
                - x, y, cos(yaw), sin(yaw), v_x, v_y, width, length + 3 one hot
            lanes: (N=70, V=25, 12) lane polylines
                - x, y, dx, dy, left_dx, left_dy, right_dx, right_dy, traffic(4)
            map_name: Map name for saving
            token: Scenario token for saving
        """
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.set_facecolor('black')
        fig.patch.set_facecolor('black')
        ax.set_aspect('equal')

        # 1. 차선 그리기 (양 끝선만)
        for i in range(lanes.shape[0]):
            lane_data = lanes[i, :, :]  # (V=25, 12) - 전체 차선 데이터

            # 유효한 포인트만 필터링 (12차원 전체가 0이 아닌 경우)
            valid_mask = np.any(lane_data != 0, axis=1)  # (V=25,)
            if not np.any(valid_mask):
                continue

            # 유효한 데이터만 추출
            valid_lane_data = lane_data[valid_mask]  # (N_valid, 12)
            lane_points = valid_lane_data[:, :2]  # (N_valid, 2) - x, y 좌표
            left_vectors = valid_lane_data[:, 4:6]  # (N_valid, 2) - 왼쪽 경계까지의 벡터
            right_vectors = valid_lane_data[:, 6:8]  # (N_valid, 2) - 오른쪽 경계까지의 벡터

            # 왼쪽 경계선 계산 및 그리기
            left_boundary = lane_points + left_vectors
            if len(left_boundary) > 1:
                ax.plot(left_boundary[:, 0],
                        left_boundary[:, 1],
                        'w-',
                        linewidth=1,
                        alpha=0.7)

            # 오른쪽 경계선 계산 및 그리기
            right_boundary = lane_points + right_vectors
            if len(right_boundary) > 1:
                ax.plot(right_boundary[:, 0],
                        right_boundary[:, 1],
                        'w-',
                        linewidth=1,
                        alpha=0.7)

        # 2. Ego vehicle 그리기
        for t in range(ego_agent_past.shape[0]):  # T=21
            # 전체 8차원 벡터가 모두 0인 경우만 유효하지 않은 데이터로 처리
            if np.all(ego_agent_past[t] == 0):
                continue
            x, y = ego_agent_past[t, 0], ego_agent_past[t, 1]
            cos_yaw, sin_yaw = ego_agent_past[t, 2], ego_agent_past[t, 3]
            width, length = ego_agent_past[t, 6], ego_agent_past[t, 7]

            yaw = math.atan2(sin_yaw, cos_yaw)
            # -2. + t * (2. / 20)  # 시간 정보 계산
            time_info = -self.past_time_horizon + t * (self.past_time_horizon /
                                                       self.num_past_poses)
            # 사각형 차량 모양 그리기
            rect = self._draw_vehicle_rectangle(
                ax,
                x,
                y,
                yaw,
                width,
                length,
                'red',
                alpha=0.8 if t == ego_agent_past.shape[0] - 1 else 0.5,
                show_heading=True,  # 명시적으로 추가
                time_info=time_info  # 시간 정보 추가
            )

        # 3. Neighbor vehicles 그리기
        for p in range(neighbor_agents_past.shape[0]):  # P=32
            agent_trajectory = neighbor_agents_past[p, :, :]  # (T=21, 11)

            # 각 시점별로 유효성 확인 (11차원 전체가 0이 아닌 경우)
            valid_mask = np.any(agent_trajectory != 0, axis=1)  # (T=21,)
            if not np.any(valid_mask):
                continue  # 모든 시점이 invalid한 경우만 skip

            # 초반 10개는 보라색, 나머지는 파란색으로 구분
            if p < 10:
                past_color = 'mediumpurple'
                current_color = 'purple'
            else:
                past_color = 'lightblue'
                current_color = 'blue'

            # 과거 위치들을 점으로 그리기 (valid한 시점만)
            for t in range(agent_trajectory.shape[0] - 1):  # 현재 위치 제외
                if not valid_mask[t]:  # invalid한 시점은 skip
                    continue
                x, y = agent_trajectory[t, 0], agent_trajectory[t, 1]
                ax.plot(x,
                        y,
                        'o',
                        color=past_color,
                        markersize=2,
                        alpha=0.6)

            # 현재 위치 (마지막 시간)를 사각형으로 그리기
            if valid_mask[-1]:  # 마지막 시점이 valid한 경우만
                current_state = agent_trajectory[-1, :]
                x, y = current_state[0], current_state[1]
                cos_yaw, sin_yaw = current_state[2], current_state[3]
                width, length = current_state[6], current_state[7]

                yaw = math.atan2(sin_yaw, cos_yaw)

                # 사각형 차량 모양 그리기
                rect = self._draw_vehicle_rectangle(ax,
                                                    x,
                                                    y,
                                                    yaw,
                                                    width,
                                                    length,
                                                    current_color,
                                                    alpha=0.8)

        # 4. 그래프 설정
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='gray')
        # 5. 범례 추가
        legend_elements = [
            plt.Line2D([0], [0],
                       marker='s',
                       color='w',
                       markerfacecolor='red',
                       markersize=10,
                       label='Ego Vehicle',
                       linestyle='None'),
            plt.Line2D([0], [0],
                       marker='s',
                       color='w',
                       markerfacecolor='purple',
                       markersize=10,
                       label='Priority Neighbor (Top 10)',
                       linestyle='None'),
            plt.Line2D([0], [0],
                       marker='s',
                       color='w',
                       markerfacecolor='blue',
                       markersize=10,
                       label='Other Neighbor (Current)',
                       linestyle='None'),
            plt.Line2D([0], [0],
                       marker='o',
                       color='w',
                       markerfacecolor='mediumpurple',
                       markersize=6,
                       label='Priority Neighbor (Past)',
                       linestyle='None'),
            plt.Line2D([0], [0],
                       marker='o',
                       color='w',
                       markerfacecolor='lightblue',
                       markersize=6,
                       label='Other Neighbor (Past)',
                       linestyle='None'),
            plt.Line2D([0], [0],
                       color='white',
                       linewidth=2,
                       label='Lane Boundaries')
        ]
        ax.legend(handles=legend_elements,
                  loc='upper right',
                  labelcolor='white',
                  facecolor='black',
                  edgecolor='white')

        # 6. 제목 설정
        ax.set_title(f'Scenario Visualization - {map_name}_{token}',
                     color='white',
                     fontsize=14,
                     pad=20)
        fig.tight_layout(pad=0.5)

        # 7. wandb에 이미지 업로드
        if self._wandb_enabled:
            # (1) Figure → wandb.Image 직접 전달
            wandb_image = wandb.Image(fig, caption=f"{map_name}_{token}")

            # (2) 메트릭/태그 딕셔너리 구성
            log_data = {
                "scenario_visualization": wandb_image,
            }

            # (3) step 명시적으로 지정
            wandb.log(log_data)
            print(f"Logged visualization to wandb for {map_name}_{token}")
        elif self.config.save_image:
            # save to disk if wandb is not enabled
            save_path = os.path.join(self._save_dir, f"{map_name}_{token}.png")
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1,
                        facecolor='black', edgecolor='none')
            print(f"Saved visualization to {save_path}")
        plt.close(fig)  # 메모리 절약을 위해 닫기

    def _draw_vehicle_rectangle(
            self,
            ax: plt.Axes,
            x: float,
            y: float,
            yaw: float,
            width: float,
            length: float,
            color: str,
            alpha: float = 0.8,
            show_heading: bool = True,
            time_info: float = None,  # 시간 정보 (초 단위)
    ) -> patches.Polygon:
        """차량을 사각형으로 그리는 헬퍼 함수입니다.

        Args:
            ax: Matplotlib axes object
            x: Vehicle center x coordinate
            y: Vehicle center y coordinate
            yaw: Vehicle heading angle in radians
            width: Vehicle width in meters
            length: Vehicle length in meters
            color: Vehicle color string
            alpha: Transparency level (0-1)
            show_heading: Whether to show heading arrow inside rectangle
            time_info: Time information in seconds (e.g., -2.0, -1.9, ..., 0.0)

        Returns:
            Polygon patch object representing the vehicle
        """
        # 사각형의 중심에서 각 모서리까지의 상대 좌표 (4, 2)
        corners = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                            [length / 2, width / 2], [-length / 2, width / 2]])

        # 회전 행렬 적용 (2, 2)
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])

        # 회전된 코너 좌표 (4, 2)
        rotated_corners = corners @ rotation_matrix.T

        # 절대 좌표로 변환 (4, 2)
        absolute_corners = rotated_corners + np.array([x, y])

        # 다각형으로 그리기
        polygon = patches.Polygon(absolute_corners,
                                  closed=True,
                                  facecolor=color,
                                  edgecolor='white',
                                  alpha=alpha,
                                  linewidth=1)
        ax.add_patch(polygon)

        # heading 방향 화살표 추가 (사각형 내부)
        if show_heading:
            # 화살표 길이를 차량 크기에 맞게 조정 (사각형 내부에 맞도록)
            arrow_length = min(width, length) * 0.6
            dx = arrow_length * np.cos(yaw)
            dy = arrow_length * np.sin(yaw)

            # 화살표 색상 결정 (차량 색상과 구분되도록)
            arrow_color = 'yellow' if color == 'red' else 'cyan'

            # 화살표 크기 조정
            head_width = min(width, length) * 0.15
            head_length = min(width, length) * 0.1

            ax.arrow(x,
                     y,
                     dx,
                     dy,
                     head_width=head_width,
                     head_length=head_length,
                     fc=arrow_color,
                     ec=arrow_color,
                     linewidth=1.5,
                     alpha=0.9)

        # 시간 정보 텍스트 추가
        if time_info is not None:
            # 텍스트 색상 결정 (배경과 구분되도록)
            text_color = 'white' if color in ['red', 'blue'] else 'black'

            # 시간 정보 포맷팅 (소수점 1자리까지)
            time_text = f"{time_info:.1f}s"

            # 텍스트 크기 조정 (차량 크기에 비례)
            fontsize = min(width, length) * 2.5
            fontsize = max(6, min(fontsize, 10))  # 최소 6, 최대 10

            # 텍스트를 차량 중앙 상단에 배치
            text_offset_y = max(width, length) * 0.3
            ax.text(x,
                    y + text_offset_y,
                    time_text,
                    ha='center',
                    va='center',
                    fontsize=fontsize,
                    color=text_color,
                    weight='bold',
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor='black',
                              alpha=0.7,
                              edgecolor='none'))

        return polygon

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)
