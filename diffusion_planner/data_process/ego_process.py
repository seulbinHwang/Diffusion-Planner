import numpy as np
import numpy.typing as npt
from typing import List, Tuple, Generator

from nuplan.common.actor_state.state_representation import TimePoint
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
from nuplan.common.geometry.convert import numpy_array_to_absolute_velocity
from typing import List, Tuple
import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import EgoInternalIndex
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
from nuplan.common.geometry.convert import numpy_array_to_absolute_velocity


def get_ego_past_array_from_scenario(
        scenario: NuPlanScenario, num_past_poses: int,
        past_time_horizon: float) -> Tuple[np.ndarray, np.ndarray]:

    current_ego_state: EgoState = scenario.initial_ego_state

    past_ego_states: Generator[EgoState, None,
                               None] = scenario.get_ego_past_trajectory(
                                   iteration=0,
                                   num_samples=num_past_poses,
                                   time_horizon=past_time_horizon)
    # list(past_ego_states): List[EgoState]
    sampled_past_ego_states: List[EgoState] = list(past_ego_states) + [
        current_ego_state
    ]
    # past_cur_ego_array: np (21, 10)
    #  x, y, theta, vx, vy, width, length, (car, pedestrian, cyclist)
    past_cur_ego_array = sampled_past_ego_states_to_array(
        sampled_past_ego_states)

    past_time_stamps: List[TimePoint] = list(
        scenario.get_past_timestamps(
            iteration=0,
            num_samples=num_past_poses,
            time_horizon=past_time_horizon)) + [scenario.start_time]

    def sampled_past_timestamps_to_array(
            past_time_stamps: List[TimePoint]) -> npt.NDArray[np.float32]:
        flat: List[int] = [t.time_us for t in past_time_stamps]
        return np.array(flat, dtype=np.int64)  # shape: (21)

    # past_time_stamps_array: np (21,)
    past_time_stamps_array = sampled_past_timestamps_to_array(past_time_stamps)

    return past_cur_ego_array, past_time_stamps_array


def sampled_past_ego_states_to_array(
        past_ego_states: List[EgoState]) -> npt.NDArray[np.float32]:  # (21, 10)
    # 원래 있던 함수임
    past_cur_num = len(past_ego_states)
    past_cur_ego_array = np.zeros((past_cur_num, 10), dtype=np.float64)
    for time_i in range(0, past_cur_num, 1):
        past_cur_ego_array[
            time_i, EgoInternalIndex.x()] = past_ego_states[time_i].center.x
        past_cur_ego_array[
            time_i, EgoInternalIndex.y()] = past_ego_states[time_i].center.y
        heading_ = past_ego_states[time_i].center.heading
        past_cur_ego_array[time_i, EgoInternalIndex.heading()] = heading_
        # --- 자차좌표계 → 세계좌표계 속도 변환: 회전만 적용 ---
        v_local = past_ego_states[
            time_i].dynamic_car_state.center_velocity_2d  # body-frame velocity
        he = float(heading_)
        c, s = np.cos(he), np.sin(he)
        vx_w = c * float(v_local.x) - s * float(v_local.y)
        vy_w = s * float(v_local.x) + c * float(v_local.y)
        past_cur_ego_array[time_i, EgoInternalIndex.vx()] = vx_w
        past_cur_ego_array[time_i, EgoInternalIndex.vy()] = vy_w
        # past_cur_ego_array[time_i, EgoInternalIndex.ax(
        # )] = past_ego_states[time_i].dynamic_car_state.rear_axle_acceleration_2d.x
        # past_cur_ego_array[time_i, EgoInternalIndex.ay(
        # )] = past_ego_states[time_i].dynamic_car_state.rear_axle_acceleration_2d.y

        past_cur_ego_array[
            time_i,
            EgoInternalIndex.ax()] = past_ego_states[time_i].car_footprint.width
        past_cur_ego_array[time_i, EgoInternalIndex.ay(
        )] = past_ego_states[time_i].car_footprint.length
        past_cur_ego_array[time_i, 7:10] = [
            1, 0, 0
        ]  # one-hot encoding for agent type (car, pedestrian, cyclist)

    return past_cur_ego_array


def sampled_future_ego_states_to_array(
        future_ego_states: List[EgoState]) -> npt.NDArray[np.float64]:
    """미래 ego 상태 리스트를 “월드 좌표계 기준 10차원 배열”로 바꾼다.

    이 함수는 시나리오에서 가져온 여러 개의 미래 ego 상태(EgoState)를
    한 줄짜리 숫자 배열로 정리해준다. 나중에 다른 함수에서
    ego 기준 좌표계로 바꾸기 전에, “월드 기준 원본 값”을 담는 역할이다.

    각 시점마다 다음과 같은 값들을 담는다.

    - 위치 : x, y                   (월드 좌표)
    - 방향 : heading                (라디안, 월드 기준)
    - 속도 : vx, vy                 (월드 좌표 기준 속도, ego 바디속도를 회전해서 구함)
    - 차체 크기 : width, length
    - 타입 one-hot : [1, 0, 0]      (항상 차량이라고 가정: car=1, ped=0, bike=0)

    Args:
        future_ego_states (List[EgoState]):
            - 길이: future_len
            - 각 원소는 한 시점의 ego 상태(EgoState).

    Returns:
        np.ndarray:
            - fut_ego_world_10
            - shape: (future_len, 10)
            - 열 순서:
                [x, y, heading, vx, vy, width, length, car, pedestrian, cyclist]
            - dtype: float64
    """
    future_len: int = len(future_ego_states)
    # fut_ego_world_10: (future_len, 10)
    fut_ego_world_10 = np.zeros((future_len, 10), dtype=np.float64)

    for time_i in range(future_len):
        # 위치 (x, y)
        fut_ego_world_10[
            time_i, EgoInternalIndex.x()] = future_ego_states[time_i].center.x
        fut_ego_world_10[
            time_i, EgoInternalIndex.y()] = future_ego_states[time_i].center.y

        # 방향 heading (월드 좌표 기준)
        fut_ego_world_10[time_i, EgoInternalIndex.heading(
        )] = future_ego_states[time_i].center.heading

        # --- 자차좌표계 → 세계좌표계 속도 변환: 회전만 적용 ---
        # v_local: Ego body-frame 속도 (vx_body, vy_body)
        v_local = future_ego_states[
            time_i].dynamic_car_state.center_velocity_2d  # body-frame velocity
        he = float(future_ego_states[time_i].center.heading)
        c, s = np.cos(he), np.sin(he)
        vx_w = c * float(v_local.x) - s * float(v_local.y)
        vy_w = s * float(v_local.x) + c * float(v_local.y)
        fut_ego_world_10[time_i, EgoInternalIndex.vx()] = vx_w
        fut_ego_world_10[time_i, EgoInternalIndex.vy()] = vy_w

        # 차체 크기 (width, length)
        fut_ego_world_10[time_i, EgoInternalIndex.ax(
        )] = future_ego_states[time_i].car_footprint.width
        fut_ego_world_10[time_i, EgoInternalIndex.ay(
        )] = future_ego_states[time_i].car_footprint.length

        # 타입 one-hot (car, pedestrian, cyclist) = (1, 0, 0)
        fut_ego_world_10[time_i, 7:10] = [1, 0, 0]

    return fut_ego_world_10


def get_ego_future_array_from_scenario(
    scenario: NuPlanScenario,
    current_ego_state: EgoState,
    num_future_poses: int,
    future_time_horizon: float,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """시나리오에서 ego의 미래 궤적을 가져와,
    ego 기준 좌표계로 변환한 결과를 (T,3) / (T,11) 두 가지 형태로 돌려준다.

    전체 흐름
    ----------
    1) nuPlan 시나리오에서, 현재 ego 상태 기준
       `num_future_poses`, `future_time_horizon` 조건에 맞게
       미래 ego 상태들을 가져온다.
       - future_ego_states: List[EgoState], 길이 T

    2) `sampled_future_ego_states_to_array` 로
       각 시점을 10차원 월드 좌표 배열로 바꾼다.
       - fut_ego_world_10: shape (T, 10)
         · [x, y, heading, vx, vy, width, length, one-hot(3)]

    3) 현재 ego 포즈(current_ego_state.rear_axle)를
       [x_ego, y_ego, yaw_ego] 형태의 벡터로 만든다.
       - ego_cur_pose_np: shape (3,)

    4) `convert_absolute_quantities_to_relative(..., 'ego')` 를 호출해
       월드 좌표 기반의 10차원 배열을 ego 기준 좌표계로 바꾸면서
       heading 을 cos, sin 두 값으로 풀어 1차원을 늘린다.
       - fut_ego_local_11: shape (T, 11), dtype float32
         · [x, y, cos(yaw), sin(yaw), vx, vy, width, length, one-hot(3)]

    5) x, y 값으로부터 heading 을 다시 뽑아 (단순 arctan2 사용)
       (T, 3) = [x, y, heading] 형태의 간단한 궤적도 만들어서 함께 반환한다.
       - fut_ego_local_xyh: shape (T, 3)

    Args:
        scenario (NuPlanScenario):
            nuPlan 시나리오 객체.
        current_ego_state (EgoState):
            현재 ego 상태. (보통 initial_ego_state 또는 시뮬레이터의 현재 상태)
        num_future_poses (int):
            몇 개의 미래 시점을 샘플링할지 (T 값).
        future_time_horizon (float):
            현재부터 몇 초 뒤까지를 커버할지 [초].

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - fut_ego_local_xyh:
                · shape: (T, 3)
                · 각 행: [x_ego, y_ego, heading_ego] (ego 기준 좌표계)
                · dtype: float32
            - fut_ego_local_11:
                · shape: (T, 11)
                · 각 행:
                    [x, y, cos(yaw), sin(yaw), vx, vy,
                     width, length, onehot_car, onehot_ped, onehot_bike]
                · dtype: float32
    """
    # future_ego_states: List[EgoState], 길이 T
    future_ego_states = scenario.get_ego_future_trajectory(
        iteration=0,
        num_samples=num_future_poses,
        time_horizon=future_time_horizon)

    # fut_ego_world_10: (T, 10)
    fut_ego_world_10 = sampled_future_ego_states_to_array(
        list(future_ego_states))

    # ego_cur_pose_np: (3,) = [x_ego, y_ego, yaw_ego] (월드 좌표계)
    ego_cur_pose_np = np.array(
        [
            current_ego_state.rear_axle.x,
            current_ego_state.rear_axle.y,
            current_ego_state.rear_axle.heading,
        ],
        dtype=np.float64,
    )

    # fut_ego_local_11: (T, 11)  ← 'ego' 모드로 상대 좌표 변환 후 float32
    fut_ego_local_11 = convert_absolute_quantities_to_relative(
        fut_ego_world_10, ego_cur_pose_np, 'ego').astype(np.float32)

    # fut_ego_local_xy: (T, 2)  ← x,y 만 분리
    fut_ego_local_xy = fut_ego_local_11[:, :2]
    fut_ego_local_cos_yaw = fut_ego_local_11[:, 2]
    fut_ego_local_sin_yaw = fut_ego_local_11[:, 3]

    # fut_ego_local_heading: (T,)  ← x,y 에서 heading 추출 (현재 구현 그대로 유지)
    fut_ego_local_heading = np.arctan2(fut_ego_local_sin_yaw, fut_ego_local_cos_yaw)

    # fut_ego_local_xyh: (T, 3) = [x, y, heading]
    fut_ego_local_xyh = np.concatenate(
        [fut_ego_local_xy, fut_ego_local_heading[:, None]], axis=-1)

    return fut_ego_local_xyh, fut_ego_local_11


def calculate_additional_ego_states(ego_agent_past, time_stamp):
    # ego_agent_past: (N, 7) where N is the number of past states.
    # 7: x, y, heading, vx, vy, width, length
    # transform haeding to cos h, sin h and calculate the steering_angle and yaw_rate for current state

    current_state = ego_agent_past[-1]
    prev_state = ego_agent_past[-2]

    dt = (time_stamp[-1] - time_stamp[-2]) * 1e-6

    cur_velocity = current_state[3]
    angle_diff = current_state[2] - prev_state[2]
    angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
    yaw_rate = angle_diff / dt

    if abs(cur_velocity) < 0.2:
        steering_angle = 0.0
        yaw_rate = 0.0  # if the car is almost stopped, the yaw rate is unreliable
    else:
        steering_angle = np.arctan(
            yaw_rate * get_pacifica_parameters().wheel_base / abs(cur_velocity))
        steering_angle = np.clip(steering_angle, -2 / 3 * np.pi, 2 / 3 * np.pi)
        yaw_rate = np.clip(yaw_rate, -0.95, 0.95)
    # ego_agent_past: (T, 7)
    # past: (T, 8) # +3 for one-hot encoding of the agent type (car, pedestrian, cyclist) and ego is always car.

    current = np.zeros((ego_agent_past.shape[1] + 3), dtype=np.float32)
    current[:2] = current_state[:2]
    current[2] = np.cos(current_state[2])
    current[3] = np.sin(current_state[2])
    current[4:8] = current_state[3:7]
    current[8] = steering_angle
    current[9] = yaw_rate

    return current
