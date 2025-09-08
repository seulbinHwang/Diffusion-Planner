import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # GUI 백엔드 사용 안함 (메모리 절약)
import matplotlib.pyplot as plt
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
import math
import wandb
import os
import torch
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from typing import Dict, Tuple, Union, List, Optional
from nuplan.common.actor_state.state_representation import Point2D
import draw_machine
# matplotlib 설정 추가
plt.rcParams['figure.max_open_warning'] = 0  # 경고 메시지 비활성화
matplotlib.rcParams['figure.max_open_warning'] = 0

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_ego_objects_to_array_list, sampled_static_objects_to_array_list,
    agent_future_process)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_to_model_inputs, get_npc_route_roadblock_ids, get_neighbor_track_tokens
# [ADDED] 통계 저장용
import json
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType  # 타입 판정용


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

    # [ADDED] 통계 유틸 함수들
    # =========================
    @staticmethod
    def _count_valid_neighbors_by_type(
            neighbor_agents_past: np.ndarray,  # (N, Tp, 11)
    ) -> Tuple[int, int, int]:
        """마지막 시점의 에이전트 상태로 유효/타입을 판정해 수를 셉니다.

        규칙:
          - 유효성: 마지막 시점의 앞 8차원(kinematics/size)이 모두 0이면 무효로 간주
            · 즉, valid = any(|state_last[:8]| > eps)
          - 타입: 마지막 3차원(one-hot) = [vehicle, pedestrian, bicycle]
            · 임계값 0.5 초과를 1로 해석(부동소수 오차 대비)

        Args:
            neighbor_agents_past (np.ndarray):
                - shape: (N, Tp, 11)
                - 마지막 차원 11 = [x, y, cos, sin, vx, vy, width, length, onehot_vehicle, onehot_ped, onehot_bike]

        Returns:
            Tuple[int, int, int]: (vehicle_count, pedestrian_count, bicycle_count)

        Raises:
            ValueError: 입력의 마지막 차원 크기가 11이 아닌 경우.
        """
        if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[
                -1] != 11:
            raise ValueError(
                f"`neighbor_agents_past` shape는 (N, Tp, 11)이어야 합니다. "
                f"got {neighbor_agents_past.shape}")

        # 마지막 시점만 사용
        last: np.ndarray = neighbor_agents_past[:, -1, :]  # (N, 11)

        # 유효성 마스크: 앞 8차원 중 하나라도 |.| > eps 이면 유효
        eps = 1e-8
        valid_mask: np.ndarray = (np.abs(last[:, :8]) > eps).any(axis=1)  # (N,)

        # 타입 one-hot (vehicle, pedestrian, bicycle)
        type_oh: np.ndarray = last[:, 8:11]  # (N, 3)
        veh_mask = type_oh[:, 0] > 0.5
        ped_mask = type_oh[:, 1] > 0.5
        bik_mask = type_oh[:, 2] > 0.5

        vehicle_count = int(np.sum(valid_mask & veh_mask))
        pedestrian_count = int(np.sum(valid_mask & ped_mask))
        bicycle_count = int(np.sum(valid_mask & bik_mask))

        return vehicle_count, pedestrian_count, bicycle_count

    @staticmethod
    def _compute_lane_speed_stats(
        vector_map_output: Dict[str,
                                np.ndarray],) -> Tuple[float, Optional[float]]:
        """차선 관련 통계를 계산한다.

        분모는 '유효 차선' 개수:
            - vector_map_output['lanes'] 의 각 차선 텐서 합(|.|) > 0

        통계:
            - 속도제한 차선 비율(%):
                100 * (#(유효 ∧ has_speed_limit True)) / (#유효)
            - 속도제한 차선들의 평균 제한속도(km/h):
                mean(lanes_speed_limit[유효 ∧ True]) * 3.6
                (없으면 None 반환)

        Args:
            vector_map_output: map_process(...) 가 반환한 dict

        Returns:
            Tuple[float, Optional[float]]: (ratio_percent, mean_speed_kmh or None)
        """
        lanes: np.ndarray = vector_map_output[
            'lanes']  # (lane_num, lane_len, 12)
        has_speed: np.ndarray = vector_map_output[
            'lanes_has_speed_limit']  # (lane_num, 1) bool
        speed_mps: np.ndarray = vector_map_output[
            'lanes_speed_limit']  # (lane_num, 1) float

        # 유효 차선 판정: 모든 성분이 0인 행은 패딩으로 간주
        lanes_valid_mask = (np.abs(lanes).sum(axis=(1, 2)) > 0)  # (lane_num,)
        if lanes_valid_mask.sum() == 0:
            return 0.0, None

        has_speed_mask = (has_speed.reshape(-1).astype(bool)
                         ) & lanes_valid_mask  # (lane_num,)
        ratio_percent = float(100.0 * has_speed_mask.sum() /
                              lanes_valid_mask.sum())

        mean_speed_kmh: Optional[float] = None
        if has_speed_mask.any():
            mean_speed_kmh = float(
                speed_mps.reshape(-1)[has_speed_mask].mean() * 3.6)

        return ratio_percent, mean_speed_kmh

    def _save_sample_stats_json(
        self,
        map_name: str,
        token: str,
        stats: Dict[str, Union[int, float, None]],
    ) -> None:
        """샘플별 통계를 `<save_path>/<map>_<token>.stats.json` 으로 저장한다.

        원자적 저장을 위해 `.tmp`로 쓴 뒤 최종 파일명으로 교체한다.

        Args:
            map_name: 맵 이름
            token: 시나리오 토큰
            stats: 저장할 통계 딕셔너리
        """
        if not self._save_dir:
            return
        os.makedirs(self._save_dir, exist_ok=True)
        json_temp_folder = os.path.join(self._save_dir, "json_temp")
        os.makedirs(json_temp_folder, exist_ok=True)
        out_path = os.path.join(json_temp_folder,
                                f"{map_name}_{token}.stats.json")
        tmp_path = out_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp_path, out_path)

    def _filter_agents_within_radius(
        self,
        neighbor_agents_past: Optional[np.ndarray],
        neighbor_agents_future: Optional[np.ndarray] = None,
        neighbor_indices: Optional[Union[np.ndarray, List[int]]] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """ego 중심 정사각형 영역(가로·세로 2*radius)으로 에이전트를 클리핑하고,
        대응하는 `neighbor_indices`도 함께 마스킹합니다.

        판정 규칙:
            - 마지막 시점의 상대좌표 (x, y)에 대해  |x| <= r  AND  |y| <= r  이면 영역 내부(True)

        Args:
            neighbor_agents_past (np.ndarray):
                - shape: (agent_num, Tp, 11)
                - 상대 좌표계 과거 에이전트 시퀀스.
            neighbor_agents_future (Optional[np.ndarray], optional):
                - shape: (agent_num, Tf, 3)
                - 상대 좌표계 미래 에이전트 시퀀스. 기본값 None.
            neighbor_indices (Optional[Union[np.ndarray, List[int]]], optional):
                - shape: (N,) (N은 agent_num 이하)
                - 현재 프레임의 트래킹 객체 리스트에서 각 에이전트가 가리키는 인덱스.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
                - filtered_neighbor_agents_past:   (agent_num, Tp, 11)
                  영역 밖 에이전트는 0으로 채움(개수 고정).
                - filtered_neighbor_agents_future: (agent_num, Tf, 3) 또는 None
                  입력이 None이 아니면 동일 규칙으로 0 마스킹.
                - filtered_neighbor_indices:       (N`,) # N`는 N 이하
                  입력 `neighbor_indices`가 주어진 경우에만 반환하며,
                  똑같이 거리 판정에 따라, 벗어나는 에이전트는 데이터에서 지운다.

        Notes:
            - 본 함수는 개수를 유지하는 **마스킹** 방식입니다(압축 X).
        """
        # 마지막 시점 상대 좌표 (agent_num, 2)  ← ego 기준이므로 ego는 정중앙(0,0)
        cur_xy = neighbor_agents_past[:, -1, :2]  # (agent_num, 2)

        # 정사각형 내부 판정: |x| <= r AND |y| <= r  → (agent_num,)
        mask_x = np.abs(cur_xy[:, 0]) <= self._radius  # (agent_num,)
        mask_y = np.abs(cur_xy[:, 1]) <= self._radius  # (agent_num,)
        mask = mask_x & mask_y  # (agent_num,), True=정사각형 내부(유효)

        # 브로드캐스팅을 위한 차원 확장: (agent_num, 1, 1)
        mask_expanded = mask[:, None, None]

        # 정사각형 바깥 에이전트는 전체 시퀀스를 0으로 만듦(개수는 고정)
        # filtered_neighbor_agents_past: (agent_num, Tp, 11)
        filtered_neighbor_agents_past = neighbor_agents_past * mask_expanded

        filtered_neighbor_agents_future = None
        if neighbor_agents_future is not None:
            # filtered_neighbor_agents_future: (agent_num, Tf, 3)
            filtered_neighbor_agents_future = neighbor_agents_future * mask_expanded
        # Indices 마스킹 (옵션)
        filtered_neighbor_indices: Optional[np.ndarray] = None
        if neighbor_indices is not None:
            neighbor_indices = np.array(
                neighbor_indices)  # (N,) # N은 agent_num 이하
            N_len_mask = mask[:len(neighbor_indices)]  # (N,)
            filtered_neighbor_indices = neighbor_indices[
                N_len_mask]  # (N`,) # N`는 N 이하

        return filtered_neighbor_agents_past, filtered_neighbor_agents_future, filtered_neighbor_indices

    # Use for inference
    def observation_adapter(self,
                            history_buffer,
                            traffic_light_data,
                            map_api,
                            route_roadblock_ids,
                            device='cpu',
                            scenario: Optional[NuPlanScenario] = None,
                            squeeze=False) -> Dict[str, torch.Tensor]:
        '''
        ego
        '''
        ego_state = history_buffer.current_state[0]
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ego_heading: float = ego_state.rear_axle.heading  #
        anchor_ego_state = np.array([
            ego_state.rear_axle.x, ego_state.rear_axle.y,
            ego_state.rear_axle.heading
        ],
                                    dtype=np.float64)
        '''_scenario_total_horizon_s
        neighbor
        '''
        ego_state_buffer = history_buffer.ego_state_buffer
        # all_frame_ego_feature: np.ndarray: (num_frames, 10)
        # 가장 과거 -> 가장 최근 순서
        all_frame_ego_feature = sampled_ego_objects_to_array_list(
            ego_state_buffer)

        observation_buffer = history_buffer.observation_buffer  # Past observations including the current
        # all_frame_agents_feature: List[np.ndarray], (frame_agents_num, 8) # frame_agents_num 길이가 가변적
        # all_frame_agents_types:  List[List[TrackedObjectType]]
        (all_frame_agents_feature, all_frame_agents_types
        ) = sampled_tracked_objects_to_array_list(observation_buffer)

        # present_static_feature: np.ndarray, (len(static_obj), 5)
        # static_objects_types: List[TrackedObjectType]
        (present_static_feature,
         static_objects_types) = sampled_static_objects_to_array_list(
             observation_buffer[-1])
        # neighbor_agents_past: (agent_num, num_frames, 11)
        # static_objects: (num_static, 10)
        """
    # ego_agent_past: (num_frames, 11)
    # neighbor_agents_past: (agent_num, num_frames, 11)
    # sorted_cur_neighbor_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
    # static_objects: (num_static, 10)
        """
        (ego_agent_past, neighbor_agents_past, neighbor_indices,
         static_objects, final_veh_num) = agent_past_process(
             all_frame_ego_feature, all_frame_agents_feature,
             all_frame_agents_types, self.num_agents, present_static_feature,
             static_objects_types, self.num_static, self.max_ped_bike,
             anchor_ego_state)
        neighbor_agents_past, _, neighbor_indices = \
            self._filter_agents_within_radius(neighbor_agents_past,
                                              None, neighbor_indices)
        # 현재 프레임의 트래킹 컨테이너로부터, 선별된 neighbor들의 track token 추출
        neighbor_track_token: List[Optional[str]] = get_neighbor_track_tokens(
            present_tracked_objects=history_buffer.observation_buffer[-1].
            tracked_objects,
            neighbor_indices=neighbor_indices,
            agents_num=self.num_agents,
        )
        '''
        Map
        '''
        # Simply fixing disconnected routes without pre-searching for reference lines
        if route_roadblock_ids and route_roadblock_ids != ['']:
            route_roadblock_ids: List[str] = route_roadblock_correction(
                ego_state, map_api, list(route_roadblock_ids))
        else:
            route_roadblock_ids = []
        # route_roadblock_ids: List[str] = route_roadblock_correction(
        #     ego_state, map_api, route_roadblock_ids)
        (coords, traffic_light_data, speed_limit,
         lane_route) = get_neighbor_vector_set_map(map_api, self._map_features,
                                                   ego_coords, ego_heading,
                                                   self._radius,
                                                   traffic_light_data)
        # # 길아: agent_num 보다 작을 수 있음(자동차만 선별했기 때문)
        present_tracked_objects: TrackedObjects = scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects: List[TrackedObjects] = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_past_tracked_objects(
                iteration=0,
                time_horizon=self.past_time_horizon,
                num_samples=self.num_past_poses)
        ]
        sampled_past_observations = past_tracked_objects + [
            present_tracked_objects
        ]
        car_token_to_rr_ids: Dict[
            str, Optional[List[str]]] = get_npc_route_roadblock_ids(
                scenario, sampled_past_observations, neighbor_track_token)
        # (agent_num, 11)
        neighbor_agents_current = neighbor_agents_past[:, -1, :]
        vector_map = map_process(route_roadblock_ids, car_token_to_rr_ids,
                                 neighbor_track_token, neighbor_agents_current,
                                 anchor_ego_state, coords, traffic_light_data,
                                 speed_limit, lane_route, self._map_features,
                                 self._max_elements, self._max_points)

        data = {
            "ego_agent_past": ego_agent_past[-21:],  # (time_len, 11)
            "neighbor_agents_past":
                neighbor_agents_past[:, -21:],  # (agent_num, time_len, 11)
            "static_objects": static_objects
        }
        # data: Dict[str, np.ndarray]
        data.update(vector_map)
        # data: Dict[str, torch.Tensor]
        data = convert_to_model_inputs(data, device, squeeze)
        if "agent_route_lane_order" in vector_map and isinstance(
                vector_map["agent_route_lane_order"], np.ndarray):
            vector_map["agent_route_lane_order"] = vector_map[
                "agent_route_lane_order"].astype(np.int64)

        return data

    @staticmethod
    def _get_agents_past_cur_mask_np(
            neighbor_agents_past: np.ndarray,  # (agents_num, time_len, 11)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """에이전트 과거/현재 시퀀스에서 유효성 마스크를 **NumPy 입출력**으로 계산한다.

        정의
        ----
        - 프레임 유효(on/off) 판정(포인트 단위):
          마지막 차원 앞 8개([x, y, cos, sin, vx, vy, width, length]) 값 중
          하나라도 0이 아니면 **유효(True)**, 모두 0이면 **무효(False)**.
        - 에이전트 유효(on/off) 판정(에이전트 단위):
          해당 에이전트의 모든 프레임이 무효이면 **무효(True)**.

        Args:
            neighbor_agents_past (np.ndarray):
                에이전트 과거/현재 시퀀스. shape = (agents_num, time_len, 11)

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - agents_past_cur_off_p_mask (np.ndarray): shape = (agents_num, time_len), dtype=bool
                  각 프레임이 **무효(True)** 인지 여부(포인트 단위 마스크).
                - agents_past_cur_off_mask (np.ndarray): shape = (agents_num,), dtype=bool
                  에이전트 전체가 **무효(True)** 인지 여부(에이전트 단위 마스크).

        Raises:
            ValueError: 입력이 (N, T, 11) 형태가 아니거나 마지막 차원(<8)일 때.
        """
        if neighbor_agents_past.ndim != 3 or neighbor_agents_past.shape[-1] < 8:
            raise ValueError(
                f"`neighbor_agents_past`는 (agents_num, time_len, 11) 형태여야 하며 "
                f"마지막 차원은 최소 8이어야 합니다. got {neighbor_agents_past.shape}")

        # (agents_num, time_len, 8)  — 0이 아니면 True
        agents_past_current_is_not_zero = (neighbor_agents_past[..., :8] != 0)

        # (agents_num, time_len) — 8개 값 중 하나라도 0이 아니면 유효
        agents_past_current_not_zero_num = agents_past_current_is_not_zero.sum(
            axis=-1)
        agents_past_cur_off_p_mask = (agents_past_current_not_zero_num == 0
                                     )  # 무효(True)

        # (agents_num) — 에이전트 단위: 유효 프레임 수가 0이면 무효(True)
        agents_past_cur_on_p_mask = ~agents_past_cur_off_p_mask
        agents_past_cur_off_mask = (agents_past_cur_on_p_mask.sum(axis=-1) == 0)

        return agents_past_cur_off_p_mask.astype(
            bool), agents_past_cur_off_mask.astype(bool)

        return agents_past_cur_off_p_mask, agents_past_cur_off_mask

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
            ego_heading = ego_state.rear_axle.heading
            anchor_ego_state = np.array([
                ego_state.rear_axle.x, ego_state.rear_axle.y,
                ego_state.rear_axle.heading
            ],
                                        dtype=np.float64)  # shape (3,)
            # all_frame_ego_feature: np (21, 10) # x, y, theta, vx, vy, width, length
            all_frame_ego_feature, time_stamps_past = get_ego_past_array_from_scenario(
                scenario, self.num_past_poses, self.past_time_horizon)

            present_tracked_objects: TrackedObjects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses)
            ]
            sampled_past_observations = past_tracked_objects + [
                present_tracked_objects
            ]
            # all_frame_agents_feature: List[np.ndarray], (frame_agents_num, 8) # frame_agents_num 길이가 가변적
            # all_frame_agents_types:  List[List[TrackedObjectType]]
            all_frame_agents_feature, all_frame_agents_types = \
                sampled_tracked_objects_to_array_list(sampled_past_observations)
            # present_static_feature: np.ndarray, (len(static_obj), 5)
            # static_objects_types: List[TrackedObjectType]
            (present_static_feature, static_objects_types
            ) = sampled_static_objects_to_array_list(present_tracked_objects)

            # : ego_agent_past: (num_frames, 11)
            # neighbor_agents_past: (agent_num, num_frames, 11)
            # neighbor_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
            (ego_agent_past, neighbor_agents_past, neighbor_indices,
             static_objects, final_veh_num) = agent_past_process(
                 all_frame_ego_feature, all_frame_agents_feature,
                 all_frame_agents_types, self.num_agents,
                 present_static_feature, static_objects_types, self.num_static,
                 self.max_ped_bike, anchor_ego_state)

            neighbor_agents_past, _, neighbor_indices = \
                self._filter_agents_within_radius(neighbor_agents_past,
                                                 None, neighbor_indices)
            neighbor_track_token: List[
                Optional[str]] = get_neighbor_track_tokens(
                    present_tracked_objects=present_tracked_objects,
                    neighbor_indices=neighbor_indices,
                    agents_num=self.num_agents,
                )
            # (agent_num, 11)
            neighbor_agents_current = neighbor_agents_past[:, -1, :]
            '''
            Map
            '''
            route_roadblock_ids = scenario.get_route_roadblock_ids()
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))
            if route_roadblock_ids != ['']:
                route_roadblock_ids = route_roadblock_correction(
                    ego_state, map_api, route_roadblock_ids)
            # # 길아: agent_num 보다 작을 수 있음(자동차만 선별했기 때문)
            car_token_to_rr_ids: Dict[
                str, Optional[List[str]]] = get_npc_route_roadblock_ids(
                    scenario, sampled_past_observations, neighbor_track_token)

            (coords, traffic_light_data, speed_limit,
             lane_route) = get_neighbor_vector_set_map(map_api,
                                                       self._map_features,
                                                       ego_coords, ego_heading,
                                                       self._radius,
                                                       traffic_light_data)
            vector_map = map_process(route_roadblock_ids, car_token_to_rr_ids,
                                     neighbor_track_token,
                                     neighbor_agents_current, anchor_ego_state,
                                     coords, traffic_light_data, speed_limit,
                                     lane_route, self._map_features,
                                     self._max_elements, self._max_points)

            # [ADDED] ────────── 샘플별 통계 계산 & 저장 ──────────
            try:
                veh_cnt, ped_cnt, bic_cnt = self._count_valid_neighbors_by_type(
                    neighbor_agents_past=neighbor_agents_past)
                ratio_percent, mean_speed_kmh = self._compute_lane_speed_stats(
                    vector_map)
                stats_payload = {
                    "vehicle_count":
                        int(veh_cnt),
                    "pedestrian_count":
                        int(ped_cnt),
                    "bicycle_count":
                        int(bic_cnt),
                    "lane_speed_limit_ratio_percent":
                        float(ratio_percent),
                    # None 은 JSON 으로 null 저장
                    "mean_speed_limit_kmh": (None if mean_speed_kmh is None else
                                             float(mean_speed_kmh)),
                }
                self._save_sample_stats_json(map_name, token, stats_payload)
            except Exception as _e:
                # 통계 수집이 실패해도 전처리 전체는 계속 진행
                print(
                    f"[Warn] stats collection failed for {map_name}_{token}: {_e}"
                )
            '''
            ego & agents future
            ego_agent_future : rear axle x,y, ~~~
            ego_agent_future_11_dim : center x,y, ~~~
            '''
            (ego_agent_future,
             ego_agent_future_11_dim) = get_ego_future_array_from_scenario(
                 scenario, ego_state, self.num_future_poses,
                 self.future_time_horizon)
            Tf, Df = ego_agent_future_11_dim.shape
            assert Tf == self.num_future_poses, (
                "Ego agent future states should have T time steps")
            assert Df == 11, (
                "Ego agent future states should have 11 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, agent type)"
            )

            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects
            future_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_future_tracked_objects(
                    iteration=0,
                    time_horizon=self.future_time_horizon,
                    num_samples=self.num_future_poses)
            ]

            sampled_future_observations: List[TrackedObjects] = [
                present_tracked_objects
            ] + future_tracked_objects
            (future_tracked_objects_array_list,
             _) = sampled_tracked_objects_to_array_list(
                 sampled_future_observations)
            # neighbor_agents_future: (num_agents, future_len, 3)
            neighbor_agents_future = agent_future_process(
                anchor_ego_state, future_tracked_objects_array_list,
                self.num_agents, neighbor_indices)
            _, neighbor_agents_future, _ = \
                self._filter_agents_within_radius(neighbor_agents_past,
                                                 neighbor_agents_future)
            '''
            ego current
            
            
            '''
            # ego_agent_past: (T, 11)
            ego_current_state = calculate_additional_ego_states(
                ego_agent_past, time_stamps_past)
            T, D = ego_agent_past.shape
            assert T == self.num_past_poses + 1, "Ego agent past states should have T+1 time steps"
            assert D == 11, "Ego agent past states should have 8 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length)"
            # gather data
            data = {
                "map_name": map_name,
                "token": token,
                "ego_agent_past": ego_agent_past,  # (time_len, 11) # DONE
                "ego_current_state": ego_current_state,  # (10,) # TODO
                # TODO: ego_agent_future 의 shape이 (0,) 인 경우가 있음. (왜 그런지는 모르겠음)
                "ego_agent_future":
                    ego_agent_future,  # rear_axle x,y # (future_len, 3) # DONE
                "ego_agent_future_11_dim":
                    ego_agent_future_11_dim,  # center x,y # (future_len, 11) # DONE
                "neighbor_agents_past":
                    neighbor_agents_past,  # (num_agents, time_len, 11) # DONE
                "neighbor_agents_future":
                    neighbor_agents_future,  # (num_agents, future_len, 3) # DONE
                "static_objects": static_objects  # (num_static, 5) # TODO
            }
            data.update(vector_map)

            # 디버깅용 그림 그리기
            save_dir = os.path.join(self._save_dir, "debug_vis")
            save_path = os.path.join(save_dir, f"{map_name}_{token}.png")
            os.makedirs(save_dir, exist_ok=True)
            if self._wandb_enabled or self.config.save_image:
                print("Visualizing scenario:", map_name, token)
                draw_machine.draw_world_model_to_png(
                    data,
                    token_to_future_traj_wrt_ego=None,
                    save_path=save_path)
            self.save_to_disk(self._save_dir, data)

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)
