import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')  # GUI 백엔드 사용 안함 (메모리 절약)
import matplotlib.pyplot as plt
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.maps.nuplan_map.nuplan_map import NuPlanMap
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
import copy
from typing import Deque
from nuplan.planning.simulation.observation.observation_type import Observation
import os
import torch
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from typing import Dict, Tuple, Union, List, Optional
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.actor_state.ego_state import EgoState
import draw_machine
# matplotlib 설정 추가
plt.rcParams['figure.max_open_warning'] = 0  # 경고 메시지 비활성화
matplotlib.rcParams['figure.max_open_warning'] = 0

from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    agent_past_process, sampled_tracked_objects_to_array_list,
    sampled_ego_objects_to_array_list, sampled_static_objects_to_array_list,
    agent_future_process, agent_future_all_process)
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
        # [변경] 타입별 상한 신설: 보행자/자전거
        self.max_pedestrians = getattr(config, "max_pedestrians", 7)  #128)
        self.max_bicycles = getattr(config, "max_bicycles", 3)  #64)
        # 안전 검사: 타입별 상한 합이 전체 슬롯보다 크지 않도록
        assert self.max_pedestrians >= 0 and self.max_bicycles >= 0
        assert (self.max_pedestrians + self.max_bicycles) <= self.num_agents, \
            f"Type caps exceed agent_num: {self.max_pedestrians}+{self.max_bicycles} > {self.num_agents}"

        self._radius = 100  # [m] query radius scope relative to the current pose.
        self.all_car_token_to_rr_ids: Optional[Dict[str,
                                                    Optional[List[str]]]] = None
        self.init_future_tracked_objects_array_list: Optional[List[
            np.ndarray]] = None
        self._init_token_to_id: Optional[Dict[str, int]] = None
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
        # if getattr(config, "use_wandb", True) and wandb.run is None:
        #     # 컨트롤러 run id를 같은 group 으로 묶어 두면 대시보드가 깔끔
        #     wandb.init(
        #         project=getattr(config, "wandb_project", "Diffusion-Planner"),
        #         entity=getattr(config, "wandb_entity", None),
        #         group=getattr(config, "wandb_group", None),  # ← 컨트롤러 ID
        #         job_type="preprocess-worker",
        #         name=f"-worker-{os.getpid()}",
        #         mode=getattr(config, "wandb_mode", "online"),
        #         reinit=True,  # fork 안전
        #         settings=wandb.Settings(start_method="fork"),
        #     )
        # self._wandb_enabled = wandb.run is not None

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

    def _get_car_token_to_rr_ids(
        self, all_car_token_to_rr_ids: Dict[str, Optional[List[str]]],
        neighbor_track_token: List[Optional[str]]
    ) -> Dict[str, Optional[List[str]]]:
        car_token_to_rr_ids: Dict[str, Optional[List[str]]] = {}
        for token in neighbor_track_token:
            if token is None:
                continue
            if token in all_car_token_to_rr_ids:
                car_token_to_rr_ids[token] = all_car_token_to_rr_ids[token]
            else:
                car_token_to_rr_ids[token] = None
        return car_token_to_rr_ids

    def _get_past_cur_ego_feature(
        self,
        scenario: Optional[NuPlanScenario] = None,
        history_buffer: Optional[SimulationHistoryBuffer] = None,
    ) -> Tuple[EgoState, Point2D, float, np.ndarray, np.ndarray,
               Optional[np.ndarray]]:
        """시나리오 또는 history buffer 에서 ego 궤적을 공통 포맷으로 추출한다.

        Args:
            scenario (Optional[NuPlanScenario]):
                - 오프라인 전처리용 시나리오. (캐싱용)
                - history_buffer 가 None 일 때만 사용.
            history_buffer (Optional[SimulationHistoryBuffer]):
                - 온라인 / inference 용 history buffer.
                - scenario 가 None 일 때만 사용.

        Returns:
            Tuple[
                EgoState,
                Point2D,
                float,
                np.ndarray,          # ego_cur_pose_np, shape (3,)
                np.ndarray,          # past_cur_ego_world_10, shape (T, 10)
                Optional[np.ndarray] # past_cur_time_np, shape (T,) 또는 None
            ]
        """
        # 둘 다 None 이거나 둘 다 존재하면 에러
        if (scenario is None and history_buffer is None) or \
           (scenario is not None and history_buffer is not None):
            raise ValueError("scenario 또는 history_buffer 중 정확히 하나만 전달해야 합니다.")

        # 공통: 현재 ego 상태
        if scenario is not None:
            ego_state: EgoState = scenario.initial_ego_state
        else:
            ego_state = history_buffer.current_state[
                0]  # type: ignore[union-attr]

        ego_point2d = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        ego_heading: float = ego_state.rear_axle.heading
        ego_cur_pose_np = np.array(
            [
                ego_state.rear_axle.x, ego_state.rear_axle.y,
                ego_state.rear_axle.heading
            ],
            dtype=np.float64,
        )  # shape: (3,)

        # 분기: 시나리오 기반 (오프라인) vs history_buffer 기반 (온라인)
        if scenario is not None:
            # past_cur_ego_world_10: np.ndarray, shape (T, 10)
            # past_cur_time_np:      np.ndarray, shape (T,)
            (past_cur_ego_world_10,
             past_cur_time_np) = get_ego_past_array_from_scenario(
                 scenario,
                 self.num_past_poses,
                 self.past_time_horizon,
             )
        else:
            ego_state_buffer: Deque[EgoState] = history_buffer.ego_state_buffer
            # past_cur_ego_world_10: np.ndarray, shape (T, 10)
            # x, y, heading, vx, vy, width, length, (car, pedestrian, cyclist)
            past_cur_ego_world_10 = sampled_ego_objects_to_array_list(
                ego_state_buffer)
            past_cur_time_np = None
        assert past_cur_ego_world_10.shape[0] == self.num_past_poses + 1, \
            f"Expected past_cur_ego_world_10 shape[0] == {self.num_past_poses + 1}, got {past_cur_ego_world_10.shape[0]}"
        return (
            ego_state,  # EgoState
            ego_point2d,  # Point2D
            ego_heading,  # float
            ego_cur_pose_np,  # np.ndarray, shape (3,)
            past_cur_ego_world_10,  # np.ndarray, shape (T, 10)
            past_cur_time_np,  # Optional[np.ndarray] # shape (T,)
        )

    # Use for inference
    def observation_adapter(self,
                            iteration: int,
                            history_buffer: SimulationHistoryBuffer,
                            traffic_light_data: List[TrafficLightStatusData],
                            map_api: NuPlanMap,
                            route_roadblock_ids: Optional[Dict[str, List[str]]],
                            device='cpu',
                            scenario: Optional[NuPlanScenario] = None,
                            use_route_lanes: bool = False,
                            squeeze=False) -> Dict[str, torch.Tensor]:
        '''
        ego
        '''
        (ego_state, ego_point2d, ego_heading, ego_cur_pose_np,
         past_cur_ego_world_10,
         _) = self._get_past_cur_ego_feature(history_buffer=history_buffer)
        #   # Past observations including the current
        observation_buffer: Deque[
            Observation] = history_buffer.observation_buffer
        """
        - past_cur_agents_world_8_list: List[np.ndarray]
            · 길이: num_frames
            · 각 원소: (frame_agents_num, 8) 열 = ID/속도/방향/크기/위치 
        - past_cur_agents_types_list: List[List[TrackedObjectType]]
            · 길이: num_frames
            · 각 프레임에서 에이전트 타입(차량/보행자/자전거) 리스트
        - present_static_feat_5: np.ndarray
            · 모양: (cur_static_num, 5)
            · [x, y, heading, width, length] (현재 프레임의 정적 객체)
        - static_types_list: List[TrackedObjectType]
            · 길이: cur_static_num
            · 각 정적 객체의 타입 리스트
        """
        (
            past_cur_agents_world_8_list,
            past_cur_agents_types_list,
            present_static_feat_5,
            static_types_list,
            _,
            _,
        ) = self._get_past_cur_agents_feature(
            observation_buffer=observation_buffer)

        # neighbor_agents_past: (agent_num, num_frames, 11)
        # static_objects: (num_static, 10)
        """
    # ego_agent_past: (num_frames, 11)
    # neighbor_agents_past: (agent_num, num_frames, 11)
    # sorted_cur_neighbor_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
    # static_objects: (num_static, 10)
        """
        (ego_agent_past, neighbor_agents_past, agents_cur_frame_indices,
         static_objects, neighbors_id) = agent_past_process(
             past_cur_ego_world_10, past_cur_agents_world_8_list,
             past_cur_agents_types_list, self.num_agents, present_static_feat_5,
             static_types_list, self.num_static, self.max_pedestrians,
             self.max_bicycles, ego_cur_pose_np)
        ego_time_len = ego_agent_past.shape[0]
        neighbor_time_len = neighbor_agents_past.shape[0]
        assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
            f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"
        """
        
        neighbors_id: np.ndarray, (agent_num,) # -1 for padding
        token_to_id: Dict[str, int]
        """
        # id_to_token = {v: k for k, v in token_to_id.items()}
        # neighbor_agents_track_token: List[Optional[str]] = []
        # for track_id in neighbors_id:
        #     if track_id == -1:
        #         neighbor_agents_track_token.append(None)
        #     else:
        #         neighbor_agents_track_token.append(id_to_token[track_id])
        ###################3
        # 현재 프레임의 트래킹 컨테이너로부터, 선별된 neighbor들의 track token 추출
        current_observation: Observation = observation_buffer[-1]
        neighbor_track_token: List[Optional[str]] = get_neighbor_track_tokens(
            present_tracked_objects=current_observation.tracked_objects,
            agents_cur_frame_indices=agents_cur_frame_indices,
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
        # traffic_light_data: Dict[str, LaneSegmentTrafficLightData]
        (coords, traffic_light_data,
         speed_limit, lane_route) = get_neighbor_vector_set_map(
             map_api, self._map_features, ego_point2d, ego_heading,
             self._radius, traffic_light_data)  # List[TrafficLightStatusData]
        # # 길아: agent_num 보다 작을 수 있음(자동차만 선별했기 때문)

        if use_route_lanes and self.all_car_token_to_rr_ids is None:
            present_tracked_objects: TrackedObjects = scenario.initial_tracked_objects.tracked_objects
            past_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses)
            ]
            past_cur_tracked_objects = past_tracked_objects + [
                present_tracked_objects
            ]
            self.all_car_token_to_rr_ids: Dict[
                str, Optional[List[str]]] = get_npc_route_roadblock_ids(
                    scenario,
                    past_cur_tracked_objects,
                    neighbor_track_token=None)
        elif not use_route_lanes:
            self.all_car_token_to_rr_ids = {}
        car_token_to_rr_ids: Dict[
            str, Optional[List[str]]] = self._get_car_token_to_rr_ids(
                self.all_car_token_to_rr_ids, neighbor_track_token)
        # (agent_num, 11)
        neighbor_agents_current = neighbor_agents_past[:, -1, :]
        vector_map = map_process(
            route_roadblock_ids,
            car_token_to_rr_ids,
            neighbor_track_token,
            neighbor_agents_current,
            ego_cur_pose_np,
            coords,
            traffic_light_data,  # traffic_light_data: Dict[str, LaneSegmentTrafficLightData]
            speed_limit,
            lane_route,
            self._map_features,
            self._max_elements,
            self._max_points)
        # (num_agents, future_len, 3)
        # FOR OPEN-LOOP SIMULATION.
        # neighbor_future_gt_3_dim = self._get_neighbor_future_gt_3_dim(
        #     scenario, ego_cur_pose_np, neighbor_agents_past, agents_cur_frame_indices,
        #     iteration)  # (num_agents, future_len, 3)
        # FOR CLOSED-LOOP SIMULATION.

        if self.init_future_tracked_objects_array_list is None:
            scenario_duration: float = scenario.duration_s.time_s + self.future_time_horizon
            num_samples = int(scenario_duration * 10.)
            # future_tracked_objects_array_list: List[ np.ndarray ((frame_agents_num, 8)) ]
            # 길이: 1 + num_future_poses
            # frame_agents_num: 각 프레임마다 다름
            self.init_future_tracked_objects_array_list, self._init_token_to_id = self._get_future_tracked_objects_array_list(
                scenario,
                iteration=0,
                future_time_horizon=scenario_duration,
                num_samples=num_samples)
        # neighbor_agents_track_token: List[Optional[str]], (agent_num,)
        # neighbor_token_id: List[Optional[int]], (agent_num,)
        neighbor_token_id = []
        for track_token in neighbor_track_token:
            if track_token is None:
                neighbor_token_id.append(None)
            else:
                neighbor_token_id.append(self._init_token_to_id[track_token])
        # (agents_num, 1 + Tf = future_all_len, 3)
        init_future_tracked_objects_array_list = copy.deepcopy(
            self.init_future_tracked_objects_array_list)
        neighbor_future_all_gt_3_dim = agent_future_all_process(
            ego_cur_pose_np, init_future_tracked_objects_array_list,
            neighbor_token_id)
        neighbor_future_gt_3_dim = neighbor_future_all_gt_3_dim[:, iteration:
                                                                iteration +
                                                                self.
                                                                num_future_poses, :]
        # (agents_num, future_len, 3)
        # neighbor_agents_past = self.zero_out_random_time_prefix(neighbor_agents_past)

        data = {
            "ego_agent_past": ego_agent_past,  # (time_len, 11)
            "neighbor_agents_past":
                neighbor_agents_past,  # (agent_num, time_len, 11)
            "neighbor_future_gt_3_dim":
                neighbor_future_gt_3_dim,  # (num_agents, future_len, 3)
            "neighbor_future_all_gt_3_dim":
                neighbor_future_all_gt_3_dim,  # (num_agents, future_all_len, 3)
            "static_objects": static_objects
        }
        if "agent_route_lane_order" in vector_map:
            aro = vector_map["agent_route_lane_order"]
            if isinstance(aro, np.ndarray) and aro.dtype != np.int64:
                vector_map["agent_route_lane_order"] = aro.astype(np.int64)
        # data: Dict[str, np.ndarray]
        data.update(vector_map)
        # data: Dict[str, torch.Tensor]
        data = convert_to_model_inputs(data, device, squeeze)
        # agent
        data[
            "neighbor_track_token"] = neighbor_track_token  # List[Optional[str]], (agent_num,)
        # 변환 후에도 안전하게 보정
        if "agent_route_lane_order" in data:
            data["agent_route_lane_order"] = data["agent_route_lane_order"].to(
                torch.int64)
        return data

    @staticmethod
    def zero_out_random_time_prefix(
            neighbor_agents_past: np.ndarray) -> np.ndarray:
        """주어진 neighbor_agents_past 텐서에서
        (agent_num, time_len, feature_dim) 형태를 가정하고,
        0 ~ time_len-1 사이에서 랜덤 target을 뽑아
        neighbor_agents_past[:, :target, :8] 구간을 0으로 만드는 함수.

        Args:
            neighbor_agents_past (np.ndarray):
                입력 텐서. shape = (num_agents, time_len, 11)

        Returns:
            np.ndarray:
                특정 시간 구간을 0으로 채운 텐서. shape 동일.
        """
        num_agents, time_len, feature_dim = neighbor_agents_past.shape

        # 0부터 time_len-1 사이 랜덤 target 선택
        target: int = np.random.randint(0, time_len // 2)
        print("target:", target)

        # 복사본을 만들어 수정 (원본을 바꾸고 싶으면 copy 제거)
        modified_past: np.ndarray = neighbor_agents_past.copy()

        # 첫 8개 feature만 0으로 세팅
        modified_past[:, :target, :5] = 0.0

        return modified_past

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

    def _get_past_cur_agents_feature(
        self,
        scenario: Optional[NuPlanScenario] = None,
        observation_buffer: Optional[Deque[Observation]] = None,
    ) -> Tuple[
            List[np.ndarray],  # past_cur_agents_world_8_list
            List[List[TrackedObjectType]],  # past_cur_agents_types_list
            np.ndarray,  # present_static_feat_5
            List[TrackedObjectType],  # static_types_list
            Optional[TrackedObjects],  # present_tracked_objects
            Optional[List[TrackedObjects]],  # past_cur_tracked_objects
    ]:
        """과거+현재 에이전트 / 정적 객체 정보를 공통 포맷으로 추출하는 함수.

        두 가지 입력 경로를 지원합니다.

        1) 오프라인 전처리 (scenario 기반)
            - nuPlanScenario 에서
              · 과거+현재 프레임의 동적 객체(차량/보행자/자전거) 배열
              · 현재 프레임의 정적 객체(표지판, 배리어 등) 배열
              을 추출합니다.

        2) 온라인 시뮬레이션 (observation_buffer 기반)
            - 시뮬레이터의 observation_buffer(연속 관측치)에서
              같은 형태의 정보를 뽑아냅니다.

        두 입력을 동시에 쓰거나, 둘 다 안 주면 오류를 발생시킵니다.

        Args:
            scenario (Optional[NuPlanScenario]):
                - 오프라인 전처리용 nuPlan 시나리오.
                - 과거/현재의 TrackedObjects 를 직접 얻을 때 사용.
            observation_buffer (Optional[Deque[Observation]]):
                - 시뮬레이션 중 관측 버퍼(과거 → 현재 순서).
                - 각 원소는 보통 `DetectionsTracks` 타입이며,
                  그 안에 `.tracked_objects` 가 들어 있습니다.

        Returns:
                - past_cur_agents_world_8_list:
                    · 길이: num_frames
                    · 각 원소: (frame_agents_num, 8) float 배열
                    · 각 행 = 한 에이전트, 열 = ID/속도/방향/크기/위치 등
                - past_cur_agents_types_list:
                    · 길이: num_frames
                    · 각 프레임에서 에이전트 타입(차량/보행자/자전거) 리스트
                - present_static_feat_5:
                    · 모양: (cur_static_num, 5)
                    · [x, y, heading, width, length] (현재 프레임의 정적 객체)
                - static_types_list:
                    · 길이: cur_static_num
                    · 각 정적 객체의 타입 리스트
                - present_tracked_objects:
                    · scenario 경로일 때만 유효(현재 프레임의 TrackedObjects)
                    · observation_buffer 경로에서는 None
                - past_cur_tracked_objects:
                    · scenario 경로일 때만 유효(과거+현재 TrackedObjects 리스트)
                    · observation_buffer 경로에서는 None
        """
        # 입력 유효성 검사
        if (scenario is None and observation_buffer is None) or \
           (scenario is not None and observation_buffer is not None):
            raise ValueError(
                "scenario 또는 observation_buffer 중 정확히 하나만 전달해야 합니다.")
        # --------------------------------------------------
        # 1) scenario 기반 (오프라인 전처리 / work()에서 사용)
        # --------------------------------------------------
        if scenario is not None:
            # 현재 프레임의 동적 객체
            present_tracked_objects = scenario.initial_tracked_objects.tracked_objects

            # 과거 프레임의 동적 객체들
            past_tracked_objects: List[TrackedObjects] = [
                tracked_objects.tracked_objects
                for tracked_objects in scenario.get_past_tracked_objects(
                    iteration=0,
                    time_horizon=self.past_time_horizon,
                    num_samples=self.num_past_poses,
                )
            ]

            # 과거 + 현재를 시간순으로 이어붙인 리스트
            past_cur_tracked_objects = past_tracked_objects + [
                present_tracked_objects
            ]

            agents_source_seq = past_cur_tracked_objects  # List[TrackedObjects]
            static_source = present_tracked_objects  # TrackedObjects
        else:
            # --------------------------------------------------
            # 2) observation_buffer 기반 (온라인 inference / observation_adapter)
            # --------------------------------------------------
            assert observation_buffer is not None  # 타입 체커용

            present_tracked_objects = None
            past_cur_tracked_objects = None

            agents_source_seq = observation_buffer  # Deque[Observation]
            static_source = observation_buffer[-1]  # 가장 최근 프레임, Observation

        # 공통 로직: 에이전트 시퀀스 → 프레임별 에이전트 배열/타입
        past_cur_agents_world_8_list, past_cur_agents_types_list, _ = \
            sampled_tracked_objects_to_array_list(agents_source_seq)

        # 공통 로직: 현재 프레임의 정적 객체 배열/타입
        present_static_feat_5, static_types_list = \
            sampled_static_objects_to_array_list(static_source)

        # 하나의 return 지점
        return (
            past_cur_agents_world_8_list,  # List[np.ndarray], #  (frame_agents_num, 8)
            past_cur_agents_types_list,  # List[List[TrackedObjectType]],
            present_static_feat_5,  # np.ndarray, (len(static_obj), 5)
            static_types_list,  # List[TrackedObjectType],
            present_tracked_objects,  # Optional[TrackedObjects],
            past_cur_tracked_objects,  # Optional[List[TrackedObjects]]
        )

    # Use for data preprocess
    def work(self, scenarios: List[NuPlanScenario]) -> None:

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            token = scenario.token
            map_api = scenario.map_api
            '''
            ego & agents past
            '''
            (ego_state, ego_point2d, ego_heading, ego_cur_pose_np,
             past_cur_ego_world_10,
             past_cur_time_np) = self._get_past_cur_ego_feature(
                 scenario=scenario)
            """
            - past_cur_agents_world_8_list: List[np.ndarray]
                · 길이: num_frames
                · 각 원소: (frame_agents_num, 8) 열 = ID/속도/방향/크기/위치 
            - past_cur_agents_types_list: List[List[TrackedObjectType]]
                · 길이: num_frames
                · 각 프레임에서 에이전트 타입(차량/보행자/자전거) 리스트
            - present_static_feat_5: np.ndarray
                · 모양: (cur_static_num, 5)
                · [x, y, heading, width, length] (현재 프레임의 정적 객체)
            - static_types_list: List[TrackedObjectType]
                · 길이: cur_static_num
                · 각 정적 객체의 타입 리스트
            - present_tracked_objects: TrackedObjects
                · scenario 경로일 때만 유효(현재 프레임의 TrackedObjects)
            - past_cur_tracked_objects: List[TrackedObjects]
                · scenario 경로일 때만 유효(과거+현재 TrackedObjects 리스트)
            """
            (
                past_cur_agents_world_8_list,
                past_cur_agents_types_list,
                present_static_feat_5,
                static_types_list,
                present_tracked_objects,
                past_cur_tracked_objects,
            ) = self._get_past_cur_agents_feature(scenario=scenario)

            # : ego_agent_past: (num_frames, 11)
            # neighbor_agents_past: (agent_num, num_frames, 11)
            # agents_cur_frame_indices: np.ndarray (_,) # 길이는 agent_num 혹은 그 이하
            (ego_agent_past, neighbor_agents_past, agents_cur_frame_indices,
             static_objects, neighbors_id) = agent_past_process(
                 past_cur_ego_world_10, past_cur_agents_world_8_list,
                 past_cur_agents_types_list, self.num_agents,
                 present_static_feat_5, static_types_list, self.num_static,
                 self.max_pedestrians, self.max_bicycles, ego_cur_pose_np)
            ego_time_len = ego_agent_past.shape[0]
            neighbor_time_len = neighbor_agents_past.shape[0]
            assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
                f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"
            neighbor_track_token: List[
                Optional[str]] = get_neighbor_track_tokens(
                    present_tracked_objects=present_tracked_objects,
                    agents_cur_frame_indices=agents_cur_frame_indices,
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
                    scenario, past_cur_tracked_objects, neighbor_track_token)

            (coords, traffic_light_data, speed_limit,
             lane_route) = get_neighbor_vector_set_map(map_api,
                                                       self._map_features,
                                                       ego_point2d, ego_heading,
                                                       self._radius,
                                                       traffic_light_data)
            vector_map = map_process(route_roadblock_ids, car_token_to_rr_ids,
                                     neighbor_track_token,
                                     neighbor_agents_current, ego_cur_pose_np,
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
            ego_future_gt_3_dim : rear axle x,y, ~~~
            planner_future_11_dim : center x,y, ~~~
            '''
            (ego_future_gt_3_dim,
             ego_future_gt_11_dim) = get_ego_future_array_from_scenario(
                 scenario, ego_state, self.num_future_poses,
                 self.future_time_horizon)
            Tf, Df = ego_future_gt_11_dim.shape
            assert Tf == self.num_future_poses, (
                "Ego agent future states should have T time steps")
            assert Df == 11, (
                "Ego agent future states should have 11 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, agent type)"
            )
            neighbor_future_gt_3_dim = self._get_neighbor_future_gt_3_dim(
                scenario, ego_cur_pose_np, neighbor_agents_past,
                agents_cur_frame_indices)  # (num_agents, future_len, 3)
            '''
            ego current
            
            
            '''
            # ego_agent_past: (T, 11)
            # TODO:ego_current_state 14 차원으로 나옴
            ego_current_state = calculate_additional_ego_states(
                ego_agent_past, past_cur_time_np)
            T, D = ego_agent_past.shape
            assert T == self.num_past_poses + 1, "Ego agent past states should have T+1 time steps"
            assert D == 11, "Ego agent past states should have 8 dimensions (x, y, cos(yaw), sin(yaw), v_x, v_y, width, length)"
            # gather data
            input_data = {
                "map_name": map_name,
                "token": token,
                "ego_current_state": ego_current_state,  # (10,) # TODO
                ############ SAME AS INFERENCE ############
                "ego_agent_past": ego_agent_past,  # (time_len, 11) # DONE
                "neighbor_agents_past":
                    neighbor_agents_past,  # (num_agents, time_len, 11) # DONE
                "static_objects": static_objects,  # (num_static, 5) # TODO
                ############################################
                ############ LEARNING ONLY #################
                # TODO: ego_future_gt_3_dim 의 shape이 (0,) 인 경우가 있음. (왜 그런지는 모르겠음)
                "ego_future_gt_3_dim":
                    ego_future_gt_3_dim,  # rear_axle x,y # (future_len, 3) # DONE
                "ego_future_gt_11_dim":
                    ego_future_gt_11_dim,  # center x,y # (future_len, 11) # DONE
                "neighbor_future_gt_3_dim":
                    neighbor_future_gt_3_dim,  # (num_agents, future_len, 3) # DONE
            }
            ############################################
            # [ADD] 저장 전 안전 보정 (훈련용 npz)
            aro = vector_map.get("agent_route_lane_order", None)
            if isinstance(aro, np.ndarray) and aro.dtype != np.int64:
                vector_map["agent_route_lane_order"] = aro.astype(np.int64)
            input_data.update(vector_map)

            # 디버깅용 그림 그리기
            save_dir = os.path.join(self._save_dir, "debug_vis")
            save_path = os.path.join(save_dir, f"{map_name}_{token}.png")
            os.makedirs(save_dir, exist_ok=True)
            # if self._wandb_enabled or self.config.save_image:

            self.save_to_disk(self._save_dir, input_data)
            if self.config.save_image:
                print("Visualizing scenario:", map_name, token)
                input_data["token_to_future_traj_wrt_ego"] = None,
                draw_machine.draw_world_model_to_png(input_data,
                                                     output_data=None,
                                                     save_path=save_path)

    def _get_future_tracked_objects_array_list(
        self,
        scenario: NuPlanScenario,
        iteration: int = 0,
        future_time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        present_tracked_objects: TrackedObjects = scenario.get_tracked_objects_at_iteration(
            iteration).tracked_objects
        if future_time_horizon is None:
            future_time_horizon = self.future_time_horizon
        if num_samples is None:
            num_samples = self.num_future_poses

        future_tracked_objects: List[TrackedObjects] = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=future_time_horizon,
                num_samples=num_samples)
        ]

        sampled_future_observations: List[TrackedObjects] = [
            present_tracked_objects
        ] + future_tracked_objects

        # future_tracked_objects_array_list: List[ np.ndarray ((frame_agents_num, 8)) ]
        # 길이: 1 + num_future_poses
        # frame_agents_num: 각 프레임마다 다름
        (future_tracked_objects_array_list, _, token_to_id
        ) = sampled_tracked_objects_to_array_list(sampled_future_observations)
        return future_tracked_objects_array_list, token_to_id

    def _get_neighbor_future_gt_3_dim(
        self,
        scenario: NuPlanScenario,
        ego_cur_pose_np: np.ndarray,  # (3,)
        neighbor_agents_past: np.ndarray,  # (num_agents, Tp, 11)
        agents_cur_frame_indices: Union[np.ndarray, List[int]],
        iteration: int = 0,
    ) -> np.ndarray:  # (num_agents, Tf, 3)
        # future_tracked_objects_array_list: List[ np.ndarray ((frame_agents_num, 8)) ]
        # 길이: 1 + num_future_poses
        # frame_agents_num: 각 프레임마다 다름
        future_tracked_objects_array_list, _ = self._get_future_tracked_objects_array_list(
            scenario, iteration)
        # neighbor_future_gt_3_dim: (num_agents, future_len, 3)
        neighbor_future_gt_3_dim = agent_future_process(
            ego_cur_pose_np, future_tracked_objects_array_list, self.num_agents,
            agents_cur_frame_indices)
        # _, neighbor_future_gt_3_dim, _ = \
        #     self._filter_agents_within_radius(neighbor_agents_past,
        #                                       neighbor_future_gt_3_dim)
        return neighbor_future_gt_3_dim

    def save_to_disk(self, dir, data):
        os.makedirs(dir, exist_ok=True)
        final_path = f"{dir}/{data['map_name']}_{data['token']}.npz"
        tmp_path = final_path + ".tmp"

        try:
            # 1) 임시 파일에 먼저 완전히 기록
            with open(tmp_path, "wb") as f:
                np.savez(f, **data)
                f.flush()
                os.fsync(f.fileno())  # 디스크 동기화(리눅스에서 유효)

            # 2) 원자적 치환(부분 파일이 최종 경로에 나타나지 않음)
            os.replace(tmp_path, final_path)

        except Exception:
            # 실패 시 임시파일만 제거(최종 파일은 손대지 않음)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            raise
