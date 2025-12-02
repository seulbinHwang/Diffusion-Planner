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
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    MapObjectPolylines, LaneSegmentTrafficLightData)
from diffusion_planner.data_process.roadblock_utils import route_roadblock_correction
from diffusion_planner.data_process.agent_process import (
    build_ego_past_feature,
    build_neighbor_past_feature,
    build_static_feature,
    sampled_tracked_objects_to_array_list,
    sampled_ego_objects_to_array_list,
    sampled_static_objects_to_array_list,
    agent_future_all_process,
)
from diffusion_planner.data_process.map_process import get_neighbor_vector_set_map, map_process
from diffusion_planner.data_process.ego_process import get_ego_past_array_from_scenario, get_ego_future_array_from_scenario, calculate_additional_ego_states
from diffusion_planner.data_process.utils import convert_data_dict_to_device_tensors, get_npc_route_roadblock_ids, get_neighbor_track_tokens
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

        self.max_agent_num = config.max_agent_num
        self.max_static_num = config.max_static_num
        # [변경] 타입별 상한 신설: 보행자/자전거
        self.max_pedestrians = None  #getattr(config, "max_pedestrians", 7)  #128)
        self.max_bicycles = None  #getattr(config, "max_bicycles", 3)  #64)
        self._filter_radius = 150  # [m] query radius scope relative to the current pose.
        self.all_car_token_to_rr_ids: Optional[Dict[str,
                                                    Optional[List[str]]]] = None
        self.init_cur_fut_agents_world_8_list: Optional[List[np.ndarray]] = None
        self._map_elements = [
            'LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'
        ]  # name of map features to be extracted.
        self._map_max_elements = {
            'LANE': config.lane_num,
            'LEFT_BOUNDARY': config.lane_num,
            'RIGHT_BOUNDARY': config.lane_num,
            'ROUTE_LANES': config.route_num
        }  # maximum number of elements to extract per feature layer.
        self._map_points_num = {
            'LANE': config.lane_len,
            'LEFT_BOUNDARY': config.lane_len,
            'RIGHT_BOUNDARY': config.lane_len,
            'ROUTE_LANES': config.route_len
        }  # maximum number of points per feature to extract per feature layer.

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
        print("self._save_dir:", self._save_dir)
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
        self,
        all_car_token_to_rr_ids: Dict[str, Optional[List[str]]],
        neighbor_track_token: List[str]  # len = chosen_agent_num
    ) -> Dict[str, List[str]]:  # len = chosen_car_num
        car_token_to_rr_ids: Dict[str, Optional[List[str]]] = {}
        for token in neighbor_track_token:
            if token in all_car_token_to_rr_ids:
                car_token_to_rr_ids[token] = all_car_token_to_rr_ids[token]
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

    def _prepare_car_token_to_rr_ids(
        self,
        scenario: NuPlanScenario,
        use_route_lanes: bool = False,
        neighbor_track_token: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        if use_route_lanes and self.all_car_token_to_rr_ids is None:
            present_tracked_objects: TrackedObjects \
                = scenario.initial_tracked_objects.tracked_objects
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
                str, List[str]] = get_npc_route_roadblock_ids(
                    scenario,
                    past_cur_tracked_objects,
                    neighbor_track_token=None)
        elif not use_route_lanes:
            self.all_car_token_to_rr_ids = {}
        # len = chosen_car_num
        car_token_to_rr_ids: Dict[str,
                                  List[str]] = self._get_car_token_to_rr_ids(
                                      self.all_car_token_to_rr_ids,
                                      neighbor_track_token)
        return car_token_to_rr_ids

    def _get_cur_fut_agents_world_8_list(
        self,
        scenario: NuPlanScenario,
        token_to_id: Dict[str, int],
        do_inference: bool,
    ):
        if do_inference:
            if self.init_cur_fut_agents_world_8_list is None:
                scenario_duration: float = scenario.duration_s.time_s + self.future_time_horizon
                num_samples = int(scenario_duration * 10.0)
                """
                self.init_cur_fut_agents_world_8_list: List[np.ndarray]
                    - 길이: 1 + num_samples
                    - 각 원소 shape: (frame_agents_num_t, 8)
                """
                (self.init_cur_fut_agents_world_8_list,
                 _) = self._get_future_tracked_objects_array_list(
                     scenario,
                     token_to_id=token_to_id,
                     iteration=0,
                     future_time_horizon=scenario_duration,
                     num_samples=num_samples)

            # 깊은 복사 후, 선택 에이전트들만 뽑아서 ego 기준으로 변환
            cur_fut_agents_world_8_list = copy.deepcopy(
                self.init_cur_fut_agents_world_8_list)
        else:
            (cur_fut_agents_world_8_list,
             _) = self._get_future_tracked_objects_array_list(
                 scenario, token_to_id=token_to_id, iteration=0)
        return cur_fut_agents_world_8_list

    # Use for inference
    def observation_adapter(self,
                            iteration: int,
                            history_buffer: SimulationHistoryBuffer,
                            traffic_light_data: List[TrafficLightStatusData],
                            map_api: NuPlanMap,
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
        # 1) ego 과거 궤적 (T, 11)
        ego_agent_past = build_ego_past_feature(
            past_cur_ego_world_10=past_cur_ego_world_10,
            ego_cur_pose_np=ego_cur_pose_np,
        )

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
            token_to_id,
            _,
            _,
        ) = self._get_past_cur_agents_feature(
            observation_buffer=observation_buffer)
        """
        neighbor_agents_past: (chosen_agent_num, num_frames, 11)
        agents_cur_frame_indices: shape (chosen_agent_num) (현재 프레임 기준 인덱스)
        neighbors_id: np.ndarray (chosen_agent_num,) (에이전트 ID)
        neighbor_track_token: List[str] (chosen_agent_num,)
        """
        # 2) neighbor 과거 궤적 (K, T, 11) + 인덱스/ID
        (neighbor_agents_past, agents_cur_frame_indices, neighbors_id,
         neighbor_track_token) = build_neighbor_past_feature(
             past_cur_agents_world_8_list=past_cur_agents_world_8_list,
             past_cur_agents_types_list=past_cur_agents_types_list,
             max_agent_num=self.max_agent_num,
             ego_cur_pose_np=ego_cur_pose_np,
             max_pedestrians=self.max_pedestrians,
             max_bicycles=self.max_bicycles,
             token_to_id=token_to_id,
             filter_radius=self._filter_radius,
         )

        ego_time_len = ego_agent_past.shape[0]
        neighbor_time_len = neighbor_agents_past.shape[
            1]  # (K, T, 11) 이므로 axis=1
        assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
            f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"
        """
        - neighbor_future_gt_3_dim:
            · shape: (chosen_agent_num, Tf, 3)
        - neighbor_future_all_gt_3_dim:
            · shape: (chosen_agent_num, 1 + Tf_all, 3)
        """
        cur_fut_agents_world_8_list = self._get_cur_fut_agents_world_8_list(
            scenario, token_to_id, do_inference=True)

        # neighbor_future_all_gt_3_dim: (chosen_agent_num, 1 + Tf_all, 3)
        neighbor_future_all_gt_3_dim = agent_future_all_process(
            ego_cur_pose_np, cur_fut_agents_world_8_list, neighbors_id)

        # neighbor_future_gt_3_dim: (chosen_agent_num, 1+Tf, 3)
        neighbor_future_gt_3_dim = neighbor_future_all_gt_3_dim[:, iteration:
                                                                iteration +
                                                                self.
                                                                num_future_poses, :]

        # 3) static 객체 (K_static, 10)
        static_objects = build_static_feature(
            present_static_feat_5=present_static_feat_5,
            static_types_list=static_types_list,
            max_static_num=self.max_static_num,
            ego_cur_pose_np=ego_cur_pose_np,
            filter_radius=self._filter_radius,
        )
        key_to_array = {
            "ego_agent_past": ego_agent_past,  # (time_len, 11)
            "neighbor_agents_past":
                neighbor_agents_past,  # (chosen_agent_num, time_len, 11)
            "neighbor_future_gt_3_dim":
                neighbor_future_gt_3_dim,  # (chosen_agent_num, 1+ future_len, 3)
            "neighbor_future_all_gt_3_dim":
                neighbor_future_all_gt_3_dim,  # (chosen_agent_num, 1+ future_all_len, 3)
            "static_objects": static_objects,  # (chosen_static_num, 10)
        }
        ###################
        (
            route_roadblock_ids,
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        ) = self._prepare_map(
            scenario=scenario,
            ego_state=ego_state,
            ego_point2d=ego_point2d,
            ego_heading=ego_heading,
            map_api=map_api,
            traffic_light_data=traffic_light_data,
        )
        # len = chosen_car_num
        car_token_to_rr_ids: Dict[
            str, List[str]] = self._prepare_car_token_to_rr_ids(
                scenario=scenario,
                use_route_lanes=use_route_lanes,
                neighbor_track_token=neighbor_track_token,
            )

        # (agent_num, 11)
        neighbor_agents_current = neighbor_agents_past[:, -1, :]
        map_key_to_array = map_process(
            route_roadblock_ids,
            car_token_to_rr_ids,  # len = chosen_car_num
            neighbor_track_token,
            neighbor_agents_current,
            ego_cur_pose_np,
            elements_to_obj_polylines,
            elements_to_traffic_light,  # elements_to_traffic_light: Dict[str, LaneSegmentTrafficLightData]
            speed_limit_dict,
            lanes_roadblock_id_list,
            self._map_elements,
            self._map_max_elements,
            self._map_points_num)
        # key_to_array: Dict[str, np.ndarray]
        key_to_array.update(map_key_to_array)
        # key_to_array: Dict[str, torch.Tensor]
        key_to_array = convert_data_dict_to_device_tensors(
            key_to_array, device, squeeze)
        # agent
        key_to_array[
            "neighbor_track_token"] = neighbor_track_token  # List[str], (chosen_agent_num,)
        return key_to_array

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
            Dict[str, int],  # token_to_id
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
                - token_to_id: Dict[str, int]:
                    · 현재 프레임에 등장하는 에이전트 토큰 → 정수 ID
                    - (딕셔너리)
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
        past_cur_agents_world_8_list, past_cur_agents_types_list, token_to_id = \
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
            token_to_id,  # Dict[str, int],
            present_tracked_objects,  # Optional[TrackedObjects],
            past_cur_tracked_objects,  # Optional[List[TrackedObjects]]
        )

    def _prepare_map(
        self,
        scenario: NuPlanScenario,
        ego_state: EgoState,
        ego_point2d: Point2D,
        ego_heading: float,
        map_api: NuPlanMap,
        traffic_light_data: Optional[List[TrafficLightStatusData]] = None,
    ) -> Tuple[List[str], Dict[str, MapObjectPolylines], Dict[
            str, LaneSegmentTrafficLightData], Dict[str, np.ndarray],
               List[str]]:
        """지도 관련 입력(route/차선/신호/속도제한)을 한 번에 준비하는 공통 유틸.

        공통 흐름:
          1) 시나리오의 route_roadblock_ids 를 가져와 끊어진 구간을 보정한다.
          2) ego 주변의 차선/경계/신호/속도제한 정보를 get_neighbor_vector_set_map 으로 뽑는다.
             - 온라인(inference) 경로: 외부에서 넘어온 traffic_light_data 사용
             - 오프라인(work) 경로: traffic_light_data 가 None 이므로 iteration=0 기준으로 자체 조회

        Args:
            scenario: nuPlan 시나리오 객체.
            ego_state: 현재 ego 상태 (rear_axle 기준).
            ego_point2d: ego 위치 (x, y).
            ego_heading: ego 진행 방향(rad).
            map_api: NuPlanMap 인스턴스.
            traffic_light_data:
                - observation_adapter 경로: 현재 시점의 신호등 리스트를 그대로 전달
                - work 경로: None → 시나리오 0번 iteration 에서 조회
        """
        # 1) route roadblock 보정
        route_roadblock_ids = scenario.get_route_roadblock_ids()
        if route_roadblock_ids != ['']:
            route_roadblock_ids = route_roadblock_correction(
                ego_state, map_api, list(route_roadblock_ids))
        else:
            route_roadblock_ids = []

        # 2) 신호등 데이터 준비
        if traffic_light_data is None:
            traffic_light_data = list(
                scenario.get_traffic_light_status_at_iteration(0))

        # 3) ego 주변 차선/경계/신호/속도제한 추출
        """
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
        (
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        ) = get_neighbor_vector_set_map(
            map_api,
            self._map_elements,
            ego_point2d,
            ego_heading,
            self._filter_radius,
            traffic_light_data,
        )

        return (
            route_roadblock_ids,
            elements_to_obj_polylines,
            elements_to_traffic_light,
            speed_limit_dict,
            lanes_roadblock_id_list,
        )

    # Use for data preprocess
    def work(self, scenarios: List[NuPlanScenario]) -> None:

        for scenario in tqdm(scenarios):
            map_name = scenario._map_name
            scenario_token = scenario.token
            map_api = scenario.map_api
            '''
            ego & agents past
            '''
            (ego_state, ego_point2d, ego_heading, ego_cur_pose_np,
             past_cur_ego_world_10,
             past_cur_time_np) = self._get_past_cur_ego_feature(
                 scenario=scenario)
            # 1) ego 과거 궤적
            ego_agent_past = build_ego_past_feature(
                past_cur_ego_world_10=past_cur_ego_world_10,
                ego_cur_pose_np=ego_cur_pose_np,
            )
            '''
            ego & agents future
            ego_future_gt_3_dim : rear axle x,y, ~~~ (future_len, 3)
            planner_future_11_dim : center x,y, ~~~ (future_len, 11)
            '''
            (ego_future_gt_3_dim,
             ego_future_gt_11_dim) = get_ego_future_array_from_scenario(
                 scenario, ego_state, self.num_future_poses,
                 self.future_time_horizon)
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
                token_to_id,
                present_tracked_objects,
                past_cur_tracked_objects,
            ) = self._get_past_cur_agents_feature(scenario=scenario)
            """
            neighbor_agents_past: (chosen_agent_num, num_frames, 11)
            agents_cur_frame_indices: shape (chosen_agent_num) (현재 프레임 기준 인덱스)
            neighbors_id: np.ndarray (chosen_agent_num,) (에이전트 ID)
            neighbor_track_token: List[str] (chosen_agent_num,)
            """
            # 2) neighbor 과거 궤적
            (neighbor_agents_past, agents_cur_frame_indices, neighbors_id,
             neighbor_track_token) = build_neighbor_past_feature(
                 past_cur_agents_world_8_list=past_cur_agents_world_8_list,
                 past_cur_agents_types_list=past_cur_agents_types_list,
                 max_agent_num=self.max_agent_num,
                 ego_cur_pose_np=ego_cur_pose_np,
                 max_pedestrians=self.max_pedestrians,
                 max_bicycles=self.max_bicycles,
                 token_to_id=token_to_id,
                 filter_radius=self._filter_radius,
             )

            ego_time_len = ego_agent_past.shape[0]
            neighbor_time_len = neighbor_agents_past.shape[1]
            assert ego_time_len == neighbor_time_len == self.num_past_poses + 1, \
                f"Expected time length {self.num_past_poses + 1}, got ego {ego_time_len}, neighbor {neighbor_time_len}"

            # cur_fut_agents_world_8_list: List[ np.ndarray ((frame_agents_num, 8)) ]
            # 길이: 1 + num_future_poses
            # frame_agents_num: 각 프레임마다 다름
            cur_fut_agents_world_8_list = self._get_cur_fut_agents_world_8_list(
                scenario, token_to_id, do_inference=False)

            # neighbor_future_gt_3_dim: (num_agents, 1+future_len, 3)
            neighbor_future_gt_3_dim = agent_future_all_process(
                ego_cur_pose_np, cur_fut_agents_world_8_list, neighbors_id)

            # 3) static 객체
            static_objects = build_static_feature(
                present_static_feat_5=present_static_feat_5,
                static_types_list=static_types_list,
                max_static_num=self.max_static_num,
                ego_cur_pose_np=ego_cur_pose_np,
                filter_radius=self._filter_radius,
            )

            key_to_array = {
                "ego_agent_past":
                    ego_agent_past,  # (chosen_agent_num, time_len, 11)
                "ego_future_gt_3_dim": ego_future_gt_3_dim,  # (future_len, 3)
                "ego_future_gt_11_dim":
                    ego_future_gt_11_dim,  # (future_len, 11)
                "neighbor_agents_past":
                    neighbor_agents_past,  # (chosen_agent_num, time_len, 11)
                "neighbor_future_gt_3_dim":
                    neighbor_future_gt_3_dim,  # (chosen_agent_num, 1+future_len, 3)
                "static_objects": static_objects,  # (chosen_static_num, 10)
            }
            '''
            Map
            '''
            (
                route_roadblock_ids,
                elements_to_obj_polylines,
                elements_to_traffic_light,
                speed_limit_dict,
                lanes_roadblock_id_list,
            ) = self._prepare_map(
                scenario=scenario,
                ego_state=ego_state,
                ego_point2d=ego_point2d,
                ego_heading=ego_heading,
                map_api=map_api,
                # traffic_light_data=None  → iteration 0 기준으로 내부에서 가져옴
            )
            # 길이 : agent_num 보다 작을 수 있음(자동차만 선별했기 때문)
            car_token_to_rr_ids: Dict[str,
                                      List[str]] = get_npc_route_roadblock_ids(
                                          scenario, past_cur_tracked_objects,
                                          neighbor_track_token)
            # (agent_num, 11)
            neighbor_agents_current = neighbor_agents_past[:, -1, :]
            map_key_to_array = map_process(
                route_roadblock_ids, car_token_to_rr_ids, neighbor_track_token,
                neighbor_agents_current, ego_cur_pose_np,
                elements_to_obj_polylines, elements_to_traffic_light,
                speed_limit_dict, lanes_roadblock_id_list, self._map_elements,
                self._map_max_elements, self._map_points_num)
            key_to_array.update(map_key_to_array)
            # gather data
            chore_data = {
                "map_name": map_name,
                "token": scenario_token,
            }
            key_to_array.update(chore_data)
            # [ADDED] ────────── 샘플별 통계 계산 & 저장 ──────────
            if self.config.make_statistics_when_caching:
                veh_cnt, ped_cnt, bic_cnt = self._count_valid_neighbors_by_type(
                    neighbor_agents_past=neighbor_agents_past)
                ratio_percent, mean_speed_kmh = self._compute_lane_speed_stats(
                    map_key_to_array)
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
                self._save_sample_stats_json(map_name, scenario_token,
                                             stats_payload)

            ############################################
            final_file_name = f"{key_to_array['map_name']}_{key_to_array['token']}"

            self.save_to_disk(self._save_dir, final_file_name, key_to_array)
            key_to_array[
                "neighbor_track_token"] = neighbor_track_token  # List[str], (chosen_agent_num,)
            if self.config.save_image:
                # 디버깅용 그림 그리기
                save_dir = os.path.join(self._save_dir, "debug_vis")
                save_path = os.path.join(save_dir, f"{final_file_name}.png")
                os.makedirs(save_dir, exist_ok=True)
                print("Visualizing scenario:", save_path)
                key_to_array["token_to_future_traj_wrt_ego"] = None
                draw_machine.draw_world_model_to_png(key_to_array,
                                                     output_data={},
                                                     save_path=save_path)

    def _get_future_tracked_objects_array_list(
        self,
        scenario: NuPlanScenario,
        token_to_id: Dict[str, int],
        iteration: int = 0,
        future_time_horizon: Optional[float] = None,
        num_samples: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], Dict[str, int]]:
        """현재 시점부터 일정 시간 동안의 모든 에이전트 상태를
        프레임별 배열 리스트로 뽑아낸다.

        하는 일 요약
        -------------
        1) 주어진 iteration 에서
           - 현재 프레임의 TrackedObjects
           - 그 이후 future_time_horizon 동안, num_samples 개의 미래 TrackedObjects
           를 가져온다.

        2) `sampled_tracked_objects_to_array_list` 를 통해,
           각 프레임을 (frame_agents_num, 8) 형태의 배열로 바꾼다.
           - 각 행: [track_id, vx, vy, heading, width, length, x, y]
           - 프레임마다 에이전트 수(frame_agents_num)는 달라질 수 있다.
           - 리스트 순서는 [현재, t+1, t+2, ...] 시간 순서.

        3) 동시에, track_token(문자열)을 일관된 정수 ID 로 바꿔주는
           token_to_id 매핑 사전도 함께 만든다.

        Args:
            iteration (int, optional):
                기준이 되는 현재 step 인덱스(0 기반).
            future_time_horizon (Optional[float], optional):
                현재 이후로 몇 초까지 볼 것인지. None 이면 self.future_time_horizon 사용.
            num_samples (Optional[int], optional):
                몇 개의 미래 프레임을 뽑을지. None 이면 self.num_future_poses 사용.

        Returns:
            Tuple[List[np.ndarray], Dict[str, int]]:
                - cur_fut_agents_world_8_list: List[np.ndarray]
                    · 길이: 1 + num_samples
                    · 각 원소 shape: (frame_agents_num_t, 8)
                      [track_id, vx, vy, heading, width, length, x, y]
                    · 리스트 순서: [현재, t+1, t+2, ...]
                - token_to_id: Dict[str, int]
                    · 전체 프레임에서 등장한 track_token → 정수 ID 매핑 사전.
        """
        present_tracked_objects: TrackedObjects = scenario.get_tracked_objects_at_iteration(
            iteration).tracked_objects

        if future_time_horizon is None:
            future_time_horizon = self.future_time_horizon
        if num_samples is None:
            num_samples = self.num_future_poses

        # 미래 프레임들의 TrackedObjects 리스트
        future_tracked_objects: List[TrackedObjects] = [
            tracked_objects.tracked_objects
            for tracked_objects in scenario.get_future_tracked_objects(
                iteration=iteration,
                time_horizon=future_time_horizon,
                num_samples=num_samples,
            )
        ]

        # [현재] + [미래들] 을 하나의 시퀀스로 합친다.
        sampled_future_observations: List[TrackedObjects] = [
            present_tracked_objects
        ] + future_tracked_objects

        # cur_fut_agents_world_8_list: List[np.ndarray]
        #   - 각 원소: (frame_agents_num, 8)
        # token_to_id: Dict[str, int]
        (cur_fut_agents_world_8_list, _,
         token_to_id) = sampled_tracked_objects_to_array_list(
             sampled_future_observations, token_to_id)

        return cur_fut_agents_world_8_list, token_to_id

    def save_to_disk(self, dir: str, final_file_name: str,
                     data: Dict[str, np.ndarray]) -> None:
        """샘플 데이터를 안전하게 디스크에 저장한다(.npz, 원자적 저장 방식).

        이 함수는 한 시나리오에서 만들어진 모든 넘파이 배열과 메타 정보를
        하나의 `.npz` 파일로 저장한다. 저장 과정에서 **부분만 써진 깨진 파일**이
        남지 않도록, 항상 임시 파일(`.tmp`)에 먼저 쓴 뒤 최종 파일명으로 교체한다.

        파일 이름 규칙
        -------------
        - 최종 경로:
            `<dir>/<map_name>_<token>.npz`
        - 예:
            >>> dir = "/tmp/nuplan_cache"
            >>> data["map_name"] = "us_ma"
            >>> data["token"] = "abcd1234"
            → "/tmp/nuplan_cache/us_ma_abcd1234.npz"

        저장 방식(알고리즘)
        ------------------
        1) 저장 폴더가 없다면 `os.makedirs(dir, exist_ok=True)` 로 만든다.
        2) 최종 파일 경로를 `<dir>/<map_name>_<token>.npz` 로 만든다.
        4) 임시 파일에 `np.savez` 로 모든 데이터를 쓴 뒤:
           - `f.flush()` 로 버퍼를 비우고
           - `os.fsync(f.fileno())` 로 디스크에 강제로 기록한다.
        5) 모든 것이 성공하면 `os.replace(tmp_path, final_file_name)` 로
           임시 파일을 최종 파일 이름으로 한 번에 교체한다.
           → 이 순간만 파일이 바뀌므로, 중간 상태의 깨진 파일이 보이지 않는다.
        6) 도중에 예외가 나면:
           - 최종 파일은 건드리지 않고
           - 남아 있을 수 있는 임시 파일만 지운 뒤 예외를 다시 올린다.

        Args:
            dir (str):
                - npz 파일을 저장할 디렉터리 경로.
                - 존재하지 않으면 내부에서 자동으로 생성한다.
            data (Dict[str, np.ndarray]):
                - 저장할 키-값 딕셔너리.

        """
        final_path_npz = f"{final_file_name}.npz"
        final_path = f"{dir}/{final_path_npz}"

        os.makedirs(dir, exist_ok=True)
        tmp_path = final_path + ".tmp"

        try:
            print("!!!!!!!!!!![TRY] NPZ saved to:", final_path)

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
