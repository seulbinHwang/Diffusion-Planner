"""
Module: Agent Data Preprocessing Functions
Description: This module contains functions for agents related data processing.

Categories:
    1. Get list of agent array from raw data
    2. Get agents array for model input
"""
import numpy as np
from typing import Dict, Deque, List, Tuple, Optional, Union
from nuplan.common.actor_state.tracked_objects import TrackedObjects, TrackedObject
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation
from nuplan.common.actor_state.ego_state import EgoState
from diffusion_planner.data_process.utils import convert_absolute_quantities_to_relative
from nuplan.common.geometry.convert import numpy_array_to_absolute_velocity


# =====================
# 1. Get list of agent array from raw data
# =====================
def _extract_agent_array(
    tracked_objects: TrackedObjects,
    token_to_id: Dict[str, int],
) -> Tuple[np.ndarray, Dict[str, int], List[TrackedObjectType]]:
    """단일 시점의 감지 결과에서 에이전트 정보를 배열로 뽑아내는 함수.

    이 함수는 한 프레임 안에 존재하는 여러 객체들 중,
    지정된 종류(차량, 보행자, 자전거 등)에 해당하는 것만 골라서
    **숫자 배열**로 정리해 줍니다. 각 에이전트는 다음과 같은 값들을 가집니다.

    - 고유 ID(정수, track_token 을 숫자로 매핑한 값)
    - 속도(vx, vy)
    - 진행 방향(heading, radian)
    - 너비/길이(width, length)
    - 위치(x, y)

    또한 문자열 기반의 track_token 을 재사용 가능한 정수 ID로 바꾸기 위해
    `token_to_id` 딕셔너리를 갱신합니다.

    Args:
        tracked_objects (TrackedObjects):
            - 현재 프레임에서 감지된 모든 객체들의 모음입니다.
        token_to_id (Dict[str, int]):
            - track_token(문자열)을 정수 ID로 매핑하는 사전입니다.
            - 새로 등장한 track_token 은 여기서 새로운 ID를 부여받습니다.

    Returns:
        Tuple[np.ndarray, Dict[str, int], List[TrackedObjectType]]:
            - a_frame_agents_feat_8 (np.ndarray):
                - 모양: (frame_agents_num, AgentInternalIndex.dim() = 8 )
                - int id , vx, vy, heading, width, length, x, y
                - dtype: float64
                - 각 행은 한 에이전트에 해당하며,
                  `AgentInternalIndex` 정의 순서대로 값이 들어 있습니다.
            - token_to_id (Dict[str, int]):
                - 갱신된 track_token → 정수 ID 매핑 사전입니다.
            - agent_types (List[TrackedObjectType]):
                - 각 행(에이전트)에 해당하는 객체 타입 목록입니다.

    """
    object_types: List[TrackedObjectType] = [
        TrackedObjectType.VEHICLE,
        TrackedObjectType.PEDESTRIAN,
        TrackedObjectType.BICYCLE,
    ]
    # 선택한 타입(object_types)에 해당하는 에이전트만 모음
    agents: List[TrackedObject] = tracked_objects.get_tracked_objects_of_types(
        object_types)
    agent_types: List[TrackedObjectType] = []

    frame_agents_num: int = len(agents)
    # a_frame_agents_feat_8: (frame_agents_num, dim=8)
    a_frame_agents_feat_8: np.ndarray = np.zeros(
        (frame_agents_num, AgentInternalIndex.dim()), dtype=np.float64)

    max_id_number: int = len(token_to_id)

    # 프레임 내 index 순회
    for idx, agent in enumerate(agents):
        # 문자열 track_token → 정수 ID로 매핑
        if agent.track_token not in token_to_id:
            token_to_id[agent.track_token] = max_id_number
            max_id_number += 1
        int_id: int = token_to_id[agent.track_token]

        # 각 에이전트의 특성 값을 배열에 채움
        a_frame_agents_feat_8[idx,
                              AgentInternalIndex.track_token()] = float(int_id)
        a_frame_agents_feat_8[idx, AgentInternalIndex.vx()] = agent.velocity.x
        a_frame_agents_feat_8[idx, AgentInternalIndex.vy()] = agent.velocity.y
        a_frame_agents_feat_8[
            idx, AgentInternalIndex.heading()] = agent.center.heading
        a_frame_agents_feat_8[idx, AgentInternalIndex.width()] = agent.box.width
        a_frame_agents_feat_8[idx,
                              AgentInternalIndex.length()] = agent.box.length
        a_frame_agents_feat_8[idx, AgentInternalIndex.x()] = agent.center.x
        a_frame_agents_feat_8[idx, AgentInternalIndex.y()] = agent.center.y

        agent_types.append(agent.tracked_object_type)

    return a_frame_agents_feat_8, token_to_id, agent_types


def sampled_tracked_objects_to_array_list(
    tracked_objects_list: List[Union[TrackedObjects, DetectionsTracks]],
) -> Tuple[List[np.ndarray], List[List[TrackedObjectType]], Dict[str, int]]:
    """여러 시점의 감지 결과를 프레임별 에이전트 배열 리스트로 바꾸는 함수.

    이 함수는 연속된 여러 시점(프레임)의 감지 결과를 입력으로 받아,
    각 시점마다 에이전트들을 숫자 배열로 바꾸어 리스트로 모아 줍니다.

    간단히 말해,
    - "시간에 따라 에이전트들이 어떻게 움직였는지"를
      프레임별 2차원 배열 목록으로 만드는 역할을 합니다.
    - 각 배열의 행은 에이전트 하나,
      열은 ID, 속도, 크기, 위치 등의 값입니다.

    시간 흐름:
        - 입력 리스트의 0번 인덱스: 가장 과거 시점
        - 마지막 인덱스: 가장 최근 시점

    Args:
        tracked_objects_list (List[Union[TrackedObjects, DetectionsTracks]]):
            - 시간 순서대로 정렬된 감지 결과 리스트입니다.
            - 각 원소는
              - `TrackedObjects` 이거나,
              - `DetectionsTracks` (이 안에 `.tracked_objects`가 들어 있음)
              둘 중 하나일 수 있습니다.

    Returns:
        Tuple[List[np.ndarray], List[List[TrackedObjectType]], Dict[str, int]]:
            - past_cur_agents_world_8_list (List[np.ndarray]):
                - 길이: num_frames
                - 각 원소 모양: (frame_agents_num, AgentInternalIndex.dim() = 8)
                - int id , vx, vy, heading, width, length, x, y
                - 각 시점의 에이전트 배열입니다. 행이 에이전트, 열이 특성입니다.
            - past_cur_agents_types_list (List[List[TrackedObjectType]]):
                - 길이: num_frames
                - 각 시점에서, 에이전트별 타입(차량/보행자/자전거 등)을 담은 리스트입니다.
            - token_to_id (Dict[str, int]):
                - 전체 시퀀스 동안 등장한 track_token을
                  정수 ID로 매핑하는 사전입니다.

    """

    # past_cur_agents_world_8_list:
    #   - 길이: num_frames
    #   - 각 원소: np.ndarray, shape (frame_agents_num, 8)
    past_cur_agents_world_8_list: List[np.ndarray] = []

    # past_cur_agents_types_list:
    #   - 길이: num_frames
    #   - 각 원소: List[TrackedObjectType], 길이 = frame_agents_num
    past_cur_agents_types_list: List[List[TrackedObjectType]] = []

    # track_token(문자열) → int ID
    token_to_id: Dict[str, int] = {}

    for timestep_idx in range(len(tracked_objects_list)):
        # 현재 시점의 원시 감지 결과
        if type(tracked_objects_list[timestep_idx]) == DetectionsTracks:
            tracked_objects: TrackedObjects = tracked_objects_list[
                timestep_idx].tracked_objects
        else:
            tracked_objects = tracked_objects_list[
                timestep_idx]  # type: ignore[assignment]

        # a_frame_agents_feat_8: (frame_agents_num, 8)
        #   int id , vx, vy, heading, width, length, x, y
        # token_to_id: Dict[str, int]
        # agent_types: List[TrackedObjectType], 길이 = frame_agents_num
        a_frame_agents_feat_8, token_to_id, agent_types = _extract_agent_array(
            tracked_objects, token_to_id)

        past_cur_agents_world_8_list.append(a_frame_agents_feat_8)
        past_cur_agents_types_list.append(agent_types)
    """
    past_cur_agents_world_8_list: List[np.ndarray]
        - 각 원소: (frame_agents_num, 8)  # frame_agents_num 은 프레임마다 다름
    past_cur_agents_types_list: List[List[TrackedObjectType]]
        - 각 원소 길이: frame_agents_num
    token_to_id: Dict[str, int]
        - track_token 문자열 → int ID 매핑
    """
    return past_cur_agents_world_8_list, past_cur_agents_types_list, token_to_id


def _extract_ego_array(track_ego: EgoState) -> np.ndarray:
    # (10)
    frame_ego_feature = np.zeros((10,), dtype=np.float64)
    # x, y, heading, vx, vy, width, length, (car, pedestrian, cyclist)

    frame_ego_feature[0] = track_ego.center.x
    frame_ego_feature[1] = track_ego.center.y
    frame_ego_feature[2] = track_ego.center.heading
    # EgoState의 속도는 자차량 좌표계 기준 벡터이므로, 세계 좌표계로 변환이 필요하다.
    v_local = track_ego.dynamic_car_state.center_velocity_2d
    he = float(track_ego.center.heading)
    c, s = np.cos(he), np.sin(he)
    vx_w = c * float(v_local.x) - s * float(v_local.y)
    vy_w = s * float(v_local.x) + c * float(v_local.y)
    frame_ego_feature[3] = vx_w
    frame_ego_feature[4] = vy_w
    frame_ego_feature[5] = track_ego.car_footprint.width
    frame_ego_feature[6] = track_ego.car_footprint.length
    frame_ego_feature[7:10] = [1, 0, 0]  # Mark as VEHICLE

    return frame_ego_feature


def sampled_ego_objects_to_array_list(
        ego_state_buffer: Deque[EgoState]) -> np.ndarray:  # (num_frames, 10)
    # x, y, heading, vx, vy, width, length, (car, pedestrian, cyclist)
    all_frame_ego_feature = []

    for past_idx in range(len(ego_state_buffer)):
        # 가장 과거 -> 가장 최근 순서
        track_ego: EgoState = ego_state_buffer[past_idx]
        frame_agents_feature = _extract_ego_array(track_ego)
        all_frame_ego_feature.append(frame_agents_feature)
    all_frame_ego_feature = np.stack(all_frame_ego_feature)  # (num_frames, 10)

    return all_frame_ego_feature


def sampled_static_objects_to_array_list(
        present_tracked_objects: TrackedObjects):

    static_object_types = [
        TrackedObjectType.CZONE_SIGN, TrackedObjectType.BARRIER,
        TrackedObjectType.TRAFFIC_CONE, TrackedObjectType.GENERIC_OBJECT
    ]

    if type(present_tracked_objects) == DetectionsTracks:
        (present_tracked_objects
        ): TrackedObjects = present_tracked_objects.tracked_objects

    static_obj: List[
        TrackedObject] = present_tracked_objects.get_tracked_objects_of_types(
            static_object_types)
    static_types_list = []
    present_static_feat_5 = np.zeros((len(static_obj), 5), dtype=np.float64)

    for idx, agent in enumerate(static_obj):
        present_static_feat_5[idx, 0] = agent.center.x
        present_static_feat_5[idx, 1] = agent.center.y
        present_static_feat_5[idx, 2] = agent.center.heading
        present_static_feat_5[idx, 3] = agent.box.width
        present_static_feat_5[idx, 4] = agent.box.length
        static_types_list.append(agent.tracked_object_type)

    return present_static_feat_5, static_types_list


# =====================
# 2. Get agents array for model input
# =====================
def _filter_agents_array(
    all_frame_agents_world_8: List[
        np.ndarray],  # len = num_frames, 각 원소 shape: (frame_agents_num, 8)
    reverse: bool = False,
) -> List[
        np.ndarray]:  # len = num_frames, 각 원소 shape: (frame_save_agents_num, 8)
    """프레임 전체에서 **같은 에이전트들만** 남기도록 걸러내는 함수.

    한 시퀀스 안에 여러 시점(프레임)의 에이전트 목록이 있을 때,
    첫 프레임(또는 `reverse=True`이면 마지막 프레임)에 등장한 에이전트만
    나머지 모든 프레임에서도 유지하고, 그렇지 않은 에이전트는 제거합니다.

    쉽게 말해서,
    - “기준 시점에 존재하던 에이전트들만 끝까지 추적하겠다”
    라는 필터링을 수행하는 함수입니다.

    각 프레임의 배열은
    `AgentInternalIndex` 순서( track_id, vx, vy, heading, width, length, x, y )
    를 따릅니다.

    **행 순서에 대한 설명**

    - 기준 프레임(첫 프레임 또는 마지막 프레임)에서:
      · 기준 프레임 배열의 행을 위에서부터 순차적으로 보면서,
        해당 에이전트가 “기준 프레임에 존재하는지”를 검사하고,
        통과하는 행만 그대로 순서대로 쌓습니다.
      · 따라서 기준 프레임의 출력 배열 `(frame_save_agents_num, 8)` 에서
        행의 순서는 **입력 기준 프레임에서의 원래 순서를 그대로 유지**합니다.
    - 다른 프레임들에서도:
      · 각 프레임의 입력 배열 `(frame_agents_num, 8)` 을 위에서부터 순회하면서,
        그 행의 track_id 가 기준 프레임 id 집합에 속하면 그 행을 그대로 추가합니다.
      · 이때도 “해당 프레임에서의 등장 순서”를 유지한 채 필터링만 할 뿐,
        별도의 정렬이나 재배열은 하지 않습니다.

    즉, 출력 리스트 각 원소의 행 순서는
    - **그 프레임의 입력 배열에서의 상대적 순서를 그대로 유지**하면서,
      기준 프레임에 없는 에이전트 행만 제거된 형태입니다.

    Args:
        all_frame_agents_world_8 (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (frame_agents_num, 8)
            - 각 행은 한 에이전트에 해당하고, 열은 ID/속도/방향/크기/위치 정보를 담습니다.
        reverse (bool, optional):
            - False:
                · 첫 번째 프레임(가장 과거 시점)을 기준 프레임으로 사용합니다.
            - True:
                · 마지막 프레임(가장 최근 시점)을 기준 프레임으로 사용합니다.

    Returns:
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (frame_save_agents_num, 8)
            - 기준 프레임에 등장했던 에이전트만 남기고, 각 프레임마다 배열을 다시 구성한 결과입니다.
              에이전트 수(frame_save_agents_num)는 프레임마다 같을 수도, 다를 수도 있습니다.
            - 각 프레임의 행 순서는, 해당 프레임 입력 배열에서의 순서를 유지한 채
              필터링만 적용된 것입니다(추가적인 정렬 없음).
    """
    # target_frame_agents_feature: (target_frame_agents_num, 8)
    target_frame_agents_feature = all_frame_agents_world_8[
        -1] if reverse else all_frame_agents_world_8[0]
    # target_frame_agents_id: (target_frame_agents_num,)
    target_frame_agents_id = target_frame_agents_feature[:,
                                                         AgentInternalIndex.
                                                         track_token()]
    for time_idx in range(len(all_frame_agents_world_8)):
        frame_exist_agents = []  # len: frame_save_agents_num # 길이 가변적
        # frame_agents_feature: (frame_agents_num, 8)
        frame_agents_feature: np.ndarray = all_frame_agents_world_8[
            time_idx]  # (_, 8)
        for agent_idx in range(frame_agents_feature.shape[0]):
            if target_frame_agents_feature.shape[0] > 0:
                agent_id = float(
                    frame_agents_feature[agent_idx,
                                         int(AgentInternalIndex.track_token())])
                is_in_target_frame = bool(
                    (agent_id == target_frame_agents_id).max())
                if is_in_target_frame:
                    frame_exist_agents.append(
                        frame_agents_feature[agent_idx, :].squeeze())

        if len(frame_exist_agents) > 0:
            # (frame_save_agents_num, 8)
            all_frame_agents_world_8[time_idx] = np.stack(frame_exist_agents)
        else:
            # 기준 프레임에 존재하지 않는 에이전트만 있었던 경우 → 빈 배열 유지
            all_frame_agents_world_8[time_idx] = np.empty(
                (0, frame_agents_feature.shape[1]), dtype=np.float32)  # (0, 8)

    return all_frame_agents_world_8


def _filter_agents_array_w_token(
    all_frame_agents_feature: List[
        np.ndarray],  # (frame_agents_num, 8) # frame_agents_num 길이가 가변적
    neighbor_token_id: List[Optional[int]],  # (agent_num, )
) -> List[
        np.
        ndarray]:  # len: num_frames, np.ndarray shape: (frame_save_agents_num, 8) # frame_save_agents_num: 길이 가변적
    neighbor_token_id_wo_none = [
        token for token in neighbor_token_id if token is not None
    ]  #
    neighbor_token_id_wo_none_array = np.array(neighbor_token_id_wo_none,
                                               dtype=np.float32)  #
    for time_idx in range(len(all_frame_agents_feature)):
        frame_exist_agents = []  # len: frame_save_agents_num # 길이 가변적
        # (frame_agents_num, 8)
        frame_agents_feature: np.ndarray = all_frame_agents_feature[
            time_idx]  # (_, 8)
        for agent_idx in range(frame_agents_feature.shape[0]):
            frame_a_agent_feature = frame_agents_feature[agent_idx, :]  # (8,)
            if neighbor_token_id_wo_none_array.shape[0] > 0:
                agent_id = float(frame_a_agent_feature[int(
                    AgentInternalIndex.track_token())])
                is_in_target_frame = bool(
                    (agent_id == neighbor_token_id_wo_none_array).max())
                if is_in_target_frame:
                    frame_exist_agents.append(frame_a_agent_feature.squeeze())

        if len(frame_exist_agents) > 0:
            all_frame_agents_feature[time_idx] = np.stack(
                frame_exist_agents)  # (frame_save_agents_num, 8)
        else:
            all_frame_agents_feature[time_idx] = np.empty(
                (0, frame_agents_feature.shape[1]), dtype=np.float32)  # (0, 8)

    return all_frame_agents_feature


def _pad_agent_states(
    all_frame_cur_exists_agents: List[
        np.ndarray],  # len = num_frames, 각 원소 shape: (current_agents_num, 8)
    reverse: bool,
) -> List[np.ndarray]:  # len = num_frames, 각 원소 shape: (current_agents_num, 8)
    """프레임마다 빠지는 에이전트를 **이전 상태로 채우며** 궤적을 이어주는 함수.

    여러 시점에 걸쳐 등장하는 에이전트들의 상태가 있을 때,
    기준 프레임에 존재하는 에이전트만을 대상으로,
    각 시점에서 값이 사라진 에이전트는 **가장 가까운 과거(또는 미래) 상태**로 채워 넣습니다.

    예시(시간이 과거→현재 방향일 때):
        - t1: a1, a2, a3
        - t2: a1, a3 만 존재

        → t2에서 a2가 사라졌으므로,
          t2의 a2 상태는 t1 시점의 a2 상태로 복사하여 채웁니다.

    `reverse=True` 를 주면 시간 방향을 뒤집어서,
    “가장 최근 상태를 기준으로 과거 쪽을 채우는” 방식으로 동작합니다.

    **행(에이전트) 순서에 대한 설명**

    - 기준 프레임 선택:
      · `reverse=False` 인 경우: 리스트의 첫 번째 프레임이 기준(t=0).
      · `reverse=True` 인 경우: 리스트를 뒤집은 뒤 첫 번째 프레임이 기준(가장 최근 시점).

    - 기준 프레임에서:
      · 기준 프레임 배열의 각 행(에이전트)을 위에서부터 읽으며,
        그 순서대로 `track_id → 기준 행 인덱스` 매핑(`key_frame_id_to_row`)을 만든다.
      · 따라서 **기준 프레임의 행 순서가 이후 모든 프레임에서 “정답 순서”가 된다.**

    - 다른 프레임들에서:
      · 각 프레임의 입력 배열 `(current_agents_num_t, 8)` 을 위에서부터 순회하면서,
        각 행의 `track_id` 를 기준 프레임의 `key_frame_id_to_row` 에서 찾아,
        해당 에이전트가 기준 프레임에서 차지하는 행 위치에 복사한다.
      · 즉, 현재 프레임에서의 등장 순서와 상관없이,
        **기준 프레임에서의 행 순서에 맞춰 재배열**된다.
      · 기준 프레임에 없는 `track_id` 는 매핑이 없으므로 사용되지 않는다.

    결국, 반환 리스트의 각 원소 `(current_agents_num, 8)` 에서:
        - `current_agents_num` 과 행 순서는 **기준 프레임의 에이전트 순서와 완전히 동일**하고,
        - 모든 시점에서 같은 순서·같은 개수를 유지하도록 패딩/복사가 이루어진다.

    Args:
        all_frame_cur_exists_agents (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num_t, 8)
              (단, 이 함수가 끝난 후에는 모든 프레임에서 같은 current_agents_num 으로 맞춰집니다.)
            - 각 행은 한 에이전트에 해당하고, 열은
              [track_id, vx, vy, heading, width, length, x, y] 입니다.
        reverse (bool):
            - False:
                · 리스트의 첫 번째 프레임을 기준(t=0)으로 보고, 이후 시점으로 진행하며 패딩합니다.
            - True:
                · 리스트를 뒤집어서 마지막 프레임을 기준으로 보고, 과거 방향으로 패딩한 뒤 다시 뒤집습니다.

    Returns:
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
            - 모든 프레임에서 같은 에이전트 순서와 개수를 유지하며,
              중간에 사라진 구간은 이전(또는 이후) 상태로 채운 결과입니다.
            - 모든 프레임의 행 순서와 개수는 **기준 프레임의 행 순서/개수와 동일**합니다.
    """
    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        all_frame_cur_exists_agents = all_frame_cur_exists_agents[::-1]

    # key_frame_agents: (current_agents_num, 8)  — 기준 프레임
    key_frame_agents = all_frame_cur_exists_agents[0]

    key_frame_id_to_row: Dict[int, int] = {}
    for cur_agent_idx, agent_id in enumerate(key_frame_agents[:, track_id_idx]):
        key_frame_id_to_row[int(agent_id)] = cur_agent_idx

    # current_state: (current_agents_num, 8)
    """
    t = -1에 있었는데, t= -2에 없어진 agent의 경우, t=-2에 t=-1의 state로 padding
    """
    current_state = np.zeros(
        (key_frame_agents.shape[0], key_frame_agents.shape[1]),
        dtype=np.float64)
    for time_idx in range(len(all_frame_cur_exists_agents)):
        # frame_cur_exists_agents: (agents_num_t, 8)
        frame_cur_exists_agents = all_frame_cur_exists_agents[time_idx]

        # Update current frame
        for agent_row_idx in range(frame_cur_exists_agents.shape[0]):
            # frame_cur_exists_agents: (agents_num_t, 8)
            agent_id = int(frame_cur_exists_agents[agent_row_idx, track_id_idx])
            key_frame_row: int = key_frame_id_to_row[agent_id]
            current_state[key_frame_row, :] = frame_cur_exists_agents[
                agent_row_idx, :]

        # Save current state (복사본 저장)
        all_frame_cur_exists_agents[time_idx] = current_state.copy()

    if reverse:
        all_frame_cur_exists_agents = all_frame_cur_exists_agents[::-1]

    return all_frame_cur_exists_agents


def _pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()

    pad_agent_trajectories = np.zeros(
        (len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]),
        dtype=np.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                pad_agent_trajectories[idx, row_idx] = frame[
                    frame[:, track_id_idx] == row_idx]

    return pad_agent_trajectories


def _pad_agent_states_with_zeros_w_token(
        agent_trajectories: List[np.ndarray],
        neighbor_token_id: List[Optional[int]],  # (agent_num, )
):
    """
    주어진 이웃 토큰(neighbor_token_id)의 (x,y,heading)을
    모든 시점에 대해 (agents_num, 1+Tf, 3) 텐서로 채웁니다.
    - 토큰이 해당 시점 프레임에 없으면 0으로 유지합니다.
    - 기존 구현의 동작과 출력 형태를 그대로 보존합니다.
    - 이웃 에이전트에 대한 내부 for-loop을 없애고 벡터화합니다.
    """
    track_id_idx = AgentInternalIndex.track_token()
    x_idx = AgentInternalIndex.x()
    y_idx = AgentInternalIndex.y()
    heading_idx = AgentInternalIndex.heading()

    agent_num = len(neighbor_token_id)
    future_all_len = len(agent_trajectories)

    # 출력 버퍼: (agents_num, 1+Tf, 3)
    pad_agent_trajectories = np.zeros((agent_num, future_all_len, 3),
                                      dtype=np.float32)

    # 이웃 토큰을 벡터로 준비 (None → NaN 로 마스킹)
    # neighbor_token_id_array: shape (agents_num,)
    neighbor_token_id_array = np.array([
        np.nan if token is None else float(token) for token in neighbor_token_id
    ],
                                       dtype=np.float32)
    valid_neighbor_mask = ~np.isnan(neighbor_token_id_array)
    if not np.any(valid_neighbor_mask):
        return pad_agent_trajectories  # 전부 None이면 바로 반환

    valid_neighbor_indices = np.nonzero(valid_neighbor_mask)[0]  # (K,)
    valid_neighbor_tokens = neighbor_token_id_array[valid_neighbor_mask]  # (K,)

    # 각 시점 프레임에 대해 한 번씩만 처리 (프레임 길이가 가변이므로 이 루프는 필요)
    for timestep_idx, frame in enumerate(agent_trajectories):
        # frame: (frame_agents_num, 8) 혹은 (0, 8)
        if frame.size == 0:
            continue

        # 현재 프레임의 에이전트 track_id들
        frame_ids = frame[:, track_id_idx].astype(np.float32,
                                                  copy=False)  # (F,)

        # 교집합 계산: frame에 실제로 존재하는 이웃 토큰만 추림
        #  - inter: 공통 토큰 값(정렬됨)
        #  - idx_frame: frame_ids 내 위치
        #  - idx_neighbors: valid_neighbor_tokens 내 위치
        inter, idx_frame, idx_neighbors = np.intersect1d(frame_ids,
                                                         valid_neighbor_tokens,
                                                         assume_unique=False,
                                                         return_indices=True)
        if inter.size == 0:
            continue

        # 해당 행의 (x, y, heading) 한 번에 수집
        xyh = frame[idx_frame][:, [x_idx, y_idx, heading_idx]].astype(
            np.float32, copy=False)  # (M, 3)

        # 원래 neighbor_token_id 위치로 되돌려 채움
        neighbor_positions = valid_neighbor_indices[idx_neighbors]  # (M,)
        pad_agent_trajectories[neighbor_positions, timestep_idx, :] = xyh

    return pad_agent_trajectories


def build_ego_past_feature(
        past_cur_ego_world_10: np.ndarray,  # (num_frames, 10)
        ego_cur_pose_np: np.ndarray,  # (3,)
) -> np.ndarray:  # (num_frames, 11)
    """이고 차량의 과거+현재 궤적을 이고 기준 상대 좌표계로 변환한다.

    - 입력은 월드 좌표계 기준 이고 궤적(여러 시점의 상태 값)이다.
    - 기준이 되는 현재 이고 상태(`ego_cur_pose_np`)를 중심으로
      모든 시점의 좌표/속도 등을 상대 좌표계로 바꾼다.
    - 모델에서 바로 쓸 수 있도록 float32 형으로 정리한다.

    Args:
        past_cur_ego_world_10 (np.ndarray):
            - x, y, heading, vx, vy, width, length, (car, pedestrian, cyclist)
            - shape: (num_frames, 10)
            - 월드 좌표계 이고 궤적.
        ego_cur_pose_np (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, yaw_ego], 기준 이고 상태.

    Returns:
        [np.ndarray]:
            - shape: (num_frames, 11)
            - 이고 기준 상대 좌표계 궤적.
    """
    # past_cur_ego_world_10: (num_frames, 10)
    # ego_agent_past: (num_frames, 11)
    ego_agent_past = convert_absolute_quantities_to_relative(
        past_cur_ego_world_10, ego_cur_pose_np)
    assert ego_agent_past.shape[1] == 11
    ego_agent_past = ego_agent_past.astype(np.float32)
    return ego_agent_past


def _convert_all_frames_agents_to_ego_local(
        all_frame_cur_exists_agents: List[
            np.ndarray],  # len = num_frames, 각 원소: (current_agents_num, 8)
        ego_cur_pose_np: np.ndarray,  # (3,)
) -> List[np.ndarray]:
    """패딩된 에이전트 상태들을 모두 ego 기준 상대 좌표계로 변환한다.

    처리 단계:
        1) `_pad_agent_states` 로 모든 프레임에서 같은 에이전트 집합/순서를 맞춘다.
        2) 각 프레임의 (world) 상태를 `convert_absolute_quantities_to_relative(..., 'agent')`
           로 ego 기준 상대 좌표계로 변환한다.

    Args:
        all_frame_cur_exists_agents (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
            - [track_id, vx, vy, heading, width, length, x, y]
        ego_cur_pose_np (np.ndarray):
            - shape: (3,), [x_ego, y_ego, yaw_ego]

    Returns:
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
            - 모든 프레임이 ego 기준 상대 좌표계로 변환된 에이전트 상태.
    """
    # 1) 프레임별로 에이전트 집합/순서를 기준 프레임에 맞춰 패딩
    # all_frame_cur_exists_agents: len = num_frames,
    #   각 원소 shape: (current_agents_num, 8)
    all_frame_cur_exists_agents = _pad_agent_states(all_frame_cur_exists_agents,
                                                    reverse=True)

    # 에이전트가 하나도 없는 경우 그대로 반환
    if len(all_frame_cur_exists_agents
          ) == 0 or all_frame_cur_exists_agents[0].shape[0] == 0:
        return all_frame_cur_exists_agents

    num_frames: int = len(all_frame_cur_exists_agents)
    current_agents_num: int = all_frame_cur_exists_agents[0].shape[0]

    # 2) (프레임, 에이전트) 축을 하나로 합쳐서 배치 변환
    # stacked_agents: (num_frames, current_agents_num, 8)
    stacked_agents: np.ndarray = np.stack(all_frame_cur_exists_agents, axis=0)
    # flat_agents: (num_frames * current_agents_num, 8)
    flat_agents: np.ndarray = stacked_agents.reshape(-1,
                                                     stacked_agents.shape[-1])

    # 한 번에 ego 기준으로 좌표계 변환
    # flat_agents_local: (num_frames * current_agents_num, 8)
    flat_agents_local: np.ndarray = convert_absolute_quantities_to_relative(
        flat_agents,
        ego_cur_pose_np,  # (3,)
        'agent',
    )

    # 다시 (num_frames, current_agents_num, 8)로 복원
    all_frame_cur_exists_agents_local_3d: np.ndarray = flat_agents_local.reshape(
        num_frames,
        current_agents_num,
        flat_agents_local.shape[-1],
    )

    # 호출부와 타입 호환을 위해 프레임 단위 리스트로 분리
    all_frame_cur_exists_agents_local: List[np.ndarray] = [
        all_frame_cur_exists_agents_local_3d[i] for i in range(num_frames)
    ]

    return all_frame_cur_exists_agents_local


def _pack_ego_local_agents(
    all_frame_cur_exists_agents_local: List[
        np.ndarray],  # len = num_frames, 각: (current_agents_num, 8)
    agents_states_dim: int,
) -> np.ndarray:
    """ego 기준 에이전트 상태 리스트를 3D 텐서로 모아 쌓는다.

    각 프레임별 에이전트 상태(ego 기준)를 받아서,
    - [x, y, cos(heading), sin(heading), vx, vy, width, length, id]
      형식의 9차원 특성으로 재구성한다.

    Args:
        all_frame_cur_exists_agents_local (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
                - [track_id, vx, vy, heading, width, length, x, y]
        agents_states_dim (int):
            - x, y, cos h, sin h, vx, vy, length, width 의 차원 수(=8).

    Returns:
        np.ndarray:
            - shape: (num_frames, current_agents_num, agents_states_dim + 1)
            - 마지막 채널(+1)은 track_id 를 저장한다.
                - [x, y, cos(heading), sin(heading), vx, vy, width, length, id]
    """
    # all_frame_np_agents_local: (num_frames, current_agents_num, agents_states_dim + 1)
    #   마지막 채널(+1)은 track_id 저장용
    num_frames = len(all_frame_cur_exists_agents_local)
    current_agents_num = all_frame_cur_exists_agents_local[0].shape[0]
    all_frame_np_agents_local = np.zeros((
        num_frames,
        current_agents_num,
        agents_states_dim + 1,
    ))

    for past_idx in range(len(all_frame_cur_exists_agents_local)):
        # frame_cur_exists_agents_local: (current_agents_num, 8)
        #   [track_id, vx, vy, heading, width, length, x, y]
        frame_cur_exists_agents_local = all_frame_cur_exists_agents_local[
            past_idx]
        # x
        all_frame_np_agents_local[
            past_idx, :, 0] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            x()].squeeze()
        # y
        all_frame_np_agents_local[
            past_idx, :, 1] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            y()].squeeze()
        # cos(heading)
        all_frame_np_agents_local[past_idx, :, 2] = np.cos(
            all_frame_cur_exists_agents_local[past_idx]
            [:, AgentInternalIndex.heading()].squeeze())
        # sin(heading)
        all_frame_np_agents_local[past_idx, :, 3] = np.sin(
            all_frame_cur_exists_agents_local[past_idx]
            [:, AgentInternalIndex.heading()].squeeze())
        # vx
        all_frame_np_agents_local[
            past_idx, :, 4] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            vx()].squeeze()
        # vy
        all_frame_np_agents_local[
            past_idx, :, 5] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            vy()].squeeze()
        # width
        all_frame_np_agents_local[
            past_idx, :, 6] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            width()].squeeze()
        # length
        all_frame_np_agents_local[
            past_idx, :, 7] = frame_cur_exists_agents_local[:,
                                                            AgentInternalIndex.
                                                            length()].squeeze()
        # id (track_token)
        all_frame_np_agents_local[
            past_idx, :,
            8] = frame_cur_exists_agents_local[:,
                                               AgentInternalIndex.track_token(
                                               )].squeeze()

    return all_frame_np_agents_local


def build_agents_past_ego_frame_array(
    past_cur_agents_world_8_list: List[
        np.ndarray],  # len = num_frames, 각 원소: (frame_agents_num, 8)
    ego_cur_pose_np: np.ndarray,  # (3,)
    agents_states_dim: int,
) -> np.ndarray:
    """과거+현재 에이전트 상태를 이고 기준 상대 좌표계 3차원 텐서로 만드는 함수.

    처리 단계:
        1) `_filter_agents_array` 로 현재 프레임에 존재하는 에이전트만 남긴다.
        2) `_pad_agent_states` 로 프레임마다 사라지는 에이전트를 이전 상태로 채운다.
        3) 각 프레임을 이고 기준 상대 좌표계로 변환한다.
        4) [x, y, cos(yaw), sin(yaw), vx, vy, width, length, id] 를 모아
           (num_frames, current_agents_num, 9) 텐서를 만든다.

    Args:
        past_cur_agents_world_8_list (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (frame_agents_num, 8)
            - [track_id, vx, vy, heading, width, length, x, y]
        ego_cur_pose_np (np.ndarray):
            - shape: (3,), [x_ego, y_ego, yaw_ego]
        agents_states_dim (int):
            - x, y, cos h, sin h, vx, vy, length, width 의 차원 수(=8).

    Returns:
        np.ndarray
            - all_frame_np_agents_local (np.ndarray):
                · shape: (num_frames, current_agents_num, agents_states_dim + 1 = 9)
                -  [x, y, cos(heading), sin(heading), vx, vy, width, length, id]
    """
    # 1) 현재 프레임 기준 에이전트 필터링 및 타입 수집
    # all_frame_cur_exists_agents: len = num_frames,
    #   각 원소 shape: (frame_save_agents_num, 8)
    all_frame_cur_exists_agents: List[np.ndarray] = _filter_agents_array(
        past_cur_agents_world_8_list, reverse=True)

    if all_frame_cur_exists_agents[-1].shape[0] == 0:
        # Return zero array when there are no agents in the scene
        # all_frame_np_agents_local: (num_frames, 0, agents_states_dim)
        all_frame_np_agents_local = np.zeros(
            (len(all_frame_cur_exists_agents), 0, agents_states_dim))
    else:
        # 2) (world→ego) 프레임별 에이전트 상태 변환
        """ all_frame_cur_exists_agents_local
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8) # current_agents_num 고정
                = [track_id, vx, vy, heading, width, length, x, y]
        """
        all_frame_cur_exists_agents_local = _convert_all_frames_agents_to_ego_local(
            all_frame_cur_exists_agents=all_frame_cur_exists_agents,
            ego_cur_pose_np=ego_cur_pose_np,
        )

        # 3) (프레임 × 에이전트 × 특성) 3D 텐서로 패킹
        # (num_frames, current_agents_num, agents_states_dim + 1 = 9)
        #  [x, y, cos(heading), sin(heading), vx, vy, width, length, id]
        all_frame_np_agents_local = _pack_ego_local_agents(
            all_frame_cur_exists_agents_local=all_frame_cur_exists_agents_local,
            agents_states_dim=agents_states_dim,
        )

    return all_frame_np_agents_local


def _build_present_static_feature_6(
        present_static_feat_5: np.ndarray,  # (cur_static_num, 5)
        ego_cur_pose_np: np.ndarray,  # (3,)
) -> np.ndarray:
    """현재 프레임의 정적 객체 정보를 -> 이고 기준 6차원 표현으로 확장한다.

    - 입력: [x, y, heading, width, length] (월드 좌표계)
    - 출력: [x, y, cos(heading), sin(heading), width, length] (ego-relative)

    Args:
        present_static_feat_5 (np.ndarray):
            - shape: (cur_static_num, 5)
        ego_cur_pose_np (np.ndarray):
            - shape: (3,), [x_ego, y_ego, yaw_ego]

    Returns:
        np.ndarray:
            - shape: (cur_static_num, 6)
            - [x, y, cos(heading), sin(heading), width, length]
    """
    # present_static_feature_6: (cur_static_num, 6)
    present_static_feature_6 = np.zeros((present_static_feat_5.shape[0], 6))
    if present_static_feat_5.shape[0] != 0:
        # present_static_feature_local: (cur_static_num, 5)
        present_static_feature_local = convert_absolute_quantities_to_relative(
            present_static_feat_5, ego_cur_pose_np, 'static')

        present_static_feature_6[:, 0] = present_static_feature_local[:, 0]
        present_static_feature_6[:, 1] = present_static_feature_local[:, 1]
        present_static_feature_6[:, 2] = np.cos(present_static_feature_local[:,
                                                                             2])
        present_static_feature_6[:, 3] = np.sin(present_static_feature_local[:,
                                                                             2])
        present_static_feature_6[:, 4] = present_static_feature_local[:, 3]
        present_static_feature_6[:, 5] = present_static_feature_local[:, 4]

    return present_static_feature_6


from typing import Optional, List, Tuple  # 이미 있으면 중복 import는 제거해도 OK


def _compute_valid_sorted_indices(
    all_frame_np_agents_local: np.
    ndarray,  # (num_frames, current_agents_num, 9)
    filter_radius: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """filter_radius 를 적용한 뒤, ego와의 거리 기준으로 에이전트 인덱스를 정렬한다.

    Args:
        all_frame_np_agents_local:
            - shape: (num_frames, current_agents_num, 9)
        filter_radius:
            - None 이 아니면, ego 기준 거리 <= filter_radius 인 에이전트만 사용.

    Returns:
        sorted_cur_agent_indices:
            - shape: (M,)  # M = 유효 에이전트 수
            - ego 로부터 가까운 순서대로 정렬된 현재 프레임 인덱스.
        dist_from_cur_agent_to_ego:
            - shape: (current_agents_num,)
            - 각 에이전트의 ego 기준 2D 거리.
    """
    # 현재 프레임(마지막 프레임 기준)에서 ego까지의 거리: (current_agents_num,)
    dist_from_cur_agent_to_ego = np.linalg.norm(
        all_frame_np_agents_local[-1, :, :2], axis=-1)

    current_agents_num: int = all_frame_np_agents_local.shape[1]

    # filter_radius 내의 에이전트만 후보로 사용
    if filter_radius is not None:
        valid_mask = dist_from_cur_agent_to_ego <= float(filter_radius)
        valid_indices = np.nonzero(valid_mask)[0]  # (M,)
    else:
        valid_indices = np.arange(current_agents_num,
                                  dtype=int)  # (current_agents_num,)

    if valid_indices.size == 0:
        # 유효한 에이전트가 하나도 없는 경우
        return np.zeros((0,), dtype=int), dist_from_cur_agent_to_ego

    # 유효한 에이전트들만 뽑아서 거리 기준 정렬
    dist_valid = dist_from_cur_agent_to_ego[valid_indices]  # (M,)
    order_local = np.argsort(dist_valid)  # (M,)
    sorted_cur_agent_indices = valid_indices[order_local]  # (M,)

    return sorted_cur_agent_indices, dist_from_cur_agent_to_ego


def _select_indices_with_type_cap(
        sorted_cur_agent_indices: np.ndarray,  # (M,)
        current_agent_types_list: List[TrackedObjectType],
        max_agent_num: int,
        max_pedestrians: int,
        max_bicycles: int,
        dist_from_cur_agent_to_ego: np.ndarray,  # (current_agents_num,)
) -> np.ndarray:
    """보행자/자전거 상한을 적용하여 에이전트를 선택한다.

    절차:
        1) sorted_cur_agent_indices 를 타입별로 세 그룹으로 나눈다.
           - 보행자 / 자전거 / 차량
        2) 보행자 → 자전거 → 차량 순으로 슬롯을 채우되,
           각각 max_pedestrians / max_bicycles / 나머지 로 상한을 둔다.
        3) 최종 선택 집합을 ego-거리 기준으로 다시 정렬하고,
           max_agent_num 개까지 사용한다.

    Returns:
        agents_cur_frame_indices:
            - shape: (K,), K ≤ max_agent_num
    """
    if max_agent_num <= 0:
        return np.zeros((0,), dtype=int)

    # 타입별로 거리 오름차순 리스트 분리
    ped_sorted_indices = [
        idx for idx in sorted_cur_agent_indices
        if current_agent_types_list[idx] == TrackedObjectType.PEDESTRIAN
    ]
    bike_sorted_indices = [
        idx for idx in sorted_cur_agent_indices
        if current_agent_types_list[idx] == TrackedObjectType.BICYCLE
    ]
    vehicle_sorted_indices = [
        idx for idx in sorted_cur_agent_indices
        if current_agent_types_list[idx] == TrackedObjectType.VEHICLE
    ]

    # 보행자/자전거 상한 적용 (max_agent_num을 넘지 않도록)
    ped_cap = min(max_pedestrians, max_agent_num)
    sel_peds = ped_sorted_indices[:ped_cap]

    remain = max_agent_num - len(sel_peds)
    bike_cap = min(max_bicycles, remain)
    sel_bikes = bike_sorted_indices[:bike_cap]

    remain -= len(sel_bikes)
    sel_vehs = vehicle_sorted_indices[:max(0, remain)]

    # 1차 선택 결과
    selected_indices = sel_peds + sel_bikes + sel_vehs

    if not selected_indices:
        return np.zeros((0,), dtype=int)

    # 최종 후보를 ego-거리 기준으로 다시 정렬 후 max_agent_num 까지만 사용
    agents_cur_frame_indices = np.array(
        sorted(
            selected_indices,
            key=lambda idx: dist_from_cur_agent_to_ego[idx],
        )[:max_agent_num],
        dtype=int,
    )
    return agents_cur_frame_indices


def _build_neighbor_vectors(
        all_frame_np_agents_local: np.
    ndarray,  # (num_frames, current_agents_num, 9)
        current_agent_types_list: List[TrackedObjectType],
        agents_states_dim: int,
        agents_cur_frame_indices: np.ndarray,  # (chosen_agent_num,)
) -> Tuple[np.ndarray, np.ndarray]:
    """선택된 에이전트 인덱스들에 대해, 한 번에 이웃 궤적 벡터와 track_id 벡터를 만든다.

    입력 텐서 구조
    --------------
    - all_frame_np_agents_local: (T, N, 9)
        · T: num_frames (과거 + 현재 시점 수)
        · N: 현재 프레임에서 살아남은 에이전트 수
        · 채널: x, y, cos(heading), sin(heading), vx, vy, width, length, id

    - current_agent_types_list: 길이 N 의 리스트.
        · 각 인덱스 i 에 대해 TrackedObjectType.VEHICLE / PEDESTRIAN / BICYCLE 중 하나.

    - agents_cur_frame_indices: (K,)
        · 현재 프레임(마지막 시점 T-1)의 에이전트 배열에서
          “이웃으로 선택된 행 인덱스” 들.
        · 이 순서가 곧 출력 텐서의 에이전트 축 순서가 된다.

    알고리즘 개요
    -------------
    1) 동적 상태 8차원 복사 (완전 벡터화)
        - 후보 텐서에서 선택된 인덱스만 골라 (T, K, 8) 로 뽑은 뒤,
          축을 바꿔 (K, T, 8) 로 만든다.
        - 이걸 neighbor_agents_past[:, :, :8] 에 그대로 넣는다.

    2) track_id 추출 (벡터화)
        - 현재 프레임 T-1 에서 선택된 인덱스의 id 채널(8)을
          한 번에 뽑아서 neighbors_id (K,) 로 만든다.

    3) 타입 one-hot 생성 (부분 벡터화 + 브로드캐스트)
        - current_agent_types_list 를 numpy 배열로 만든 후,
          agents_cur_frame_indices 로 인덱싱해 길이 K 의 neighbors_types 배열을 만든다.
        - neighbors_types == VEHICLE / PEDESTRIAN / BICYCLE 로
          boolean mask 세 개를 만든 다음,
          type_one_hot[mask, col] = 1.0 방식으로 한 번에 채운다.
        - 마지막으로 (K, 3) → (K, 1, 3) 로 늘리고
          시간축(T) 방향으로 브로드캐스트 해서 neighbor_agents_past[:, :, 8:] 에 넣는다.


    Args:
        all_frame_np_agents_local:
            이웃 후보 에이전트들의 ego 기준 과거+현재 시퀀스.
            - shape: (num_frames, current_agents_num, 9)
        current_agent_types_list:
            - 길이: current_agents_num
            - 각 에이전트 인덱스에 대응하는 TrackedObjectType 값.
        agents_states_dim:
            - 동적 상태 차원 수 (=8).
        agents_cur_frame_indices:
            - shape: (chosen_agent_num,)
            - 현재 프레임에서 선택된 에이전트 행 인덱스들.

    Returns:
        neighbor_agents_past:
            - shape: (chosen_agent_num, num_frames, 11)
            - [x, y, cos, sin, vx, vy, width, length,
               track_id, onehot_vehicle, onehot_ped, onehot_bike]
        neighbors_id:
            - shape: (chosen_agent_num,)
            - 각 이웃 에이전트의 track_id (현재 프레임 기준).
    """
    num_frames: int = all_frame_np_agents_local.shape[0]
    agents_cur_frame_indices = np.asarray(agents_cur_frame_indices, dtype=int)
    chosen_agent_num: int = int(agents_cur_frame_indices.shape[0])

    if chosen_agent_num == 0:
        neighbor_agents_past = np.zeros((0, num_frames, agents_states_dim + 3),
                                        dtype=np.float32)
        neighbors_id = np.zeros((0,), dtype=np.float32)
        return neighbor_agents_past, neighbors_id

    eight_ = agents_states_dim

    # 1) 동적 상태 8차원 배치 추출 및 축 재배치
    # (T, N, 9) → (T, chosen_agent_num, 8) → (chosen_agent_num, T, 8)
    dynamic_states: np.ndarray = all_frame_np_agents_local[:,
                                                           agents_cur_frame_indices, :
                                                           eight_].transpose(
                                                               1, 0, 2)

    # 출력 텐서 초기화
    neighbor_agents_past = np.zeros(
        (chosen_agent_num, num_frames, agents_states_dim + 3),
        dtype=np.float32,
    )
    neighbor_agents_past[:, :, :eight_] = dynamic_states

    # 2) track_id 벡터 배치 추출 (현재 프레임 = 마지막 프레임 기준)
    # shape: (chosen_agent_num,)
    neighbors_id = all_frame_np_agents_local[-1, agents_cur_frame_indices,
                                             eight_].astype(np.float32)

    # 3) 타입 one-hot (chosen_agent_num, 3) 생성 → (chosen_agent_num, T, 3) 브로드캐스트
    type_one_hot = np.zeros((chosen_agent_num, 3), dtype=np.float32)

    # current_agent_types_list 를 배열로 바꾸고, 선택된 인덱스만 추출
    # neighbors_types: shape (chosen_agent_num,), dtype=object (TrackedObjectType 인스턴스들)
    neighbors_types = np.asarray(current_agent_types_list,
                                 dtype=object)[agents_cur_frame_indices]

    veh_mask = neighbors_types == TrackedObjectType.VEHICLE
    ped_mask = neighbors_types == TrackedObjectType.PEDESTRIAN
    bike_mask = neighbors_types == TrackedObjectType.BICYCLE

    # 각 타입별 column에 1 할당 (행 단위로 일괄 처리)
    type_one_hot[veh_mask, 0] = 1.0
    type_one_hot[ped_mask, 1] = 1.0
    type_one_hot[bike_mask, 2] = 1.0

    # (chosen_agent_num, 3) → (chosen_agent_num, 1, 3) 로 늘려서 시간축(T)에 브로드캐스트
    # → (chosen_agent_num, T, 3)
    neighbor_agents_past[:, :, eight_:] = type_one_hot[:, None, :]

    return neighbor_agents_past, neighbors_id


def _select_neighbor_agents_and_build_past(
    all_frame_np_agents_local: np.
    ndarray,  # (num_frames, current_agents_num, 9)
    current_agent_types_list: List[
        TrackedObjectType],  # 길이 = current_agents_num
    agents_states_dim: int,
    max_agent_num: int,
    max_pedestrians: Optional[int],
    max_bicycles: Optional[int],
    filter_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이고 기준 에이전트 텐서에서, 가까운 에이전트들을 선택하여
    이웃 궤적 텐서를 구성한다.

    특징:
        - `max_agent_num` 은 상한(최대 개수)만 의미한다. 실제 선택 수 chosen_agent_num ≤ max_agent_num.
        - `max_pedestrians` 또는 `max_bicycles` 가 None 이면
          타입 상한을 모두 끄고, 거리 기준으로만 가까운 순서대로 선택한다.
        - `filter_radius` 가 주어지면, ego 로부터 그 거리 안에 있는 에이전트만
          후보로 사용한다(그 밖은 완전히 무시).

    Returns:
        neighbor_agents_past: (chosen_agent_num, num_frames, 11)
        agents_cur_frame_indices: shape chosen_agent_num (현재 프레임 기준 인덱스)
        neighbors_id:         (chosen_agent_num,)
    """
    num_frames: int = all_frame_np_agents_local.shape[0]

    # 0) radius 필터 + 거리 기준 정렬된 인덱스 계산
    """
        sorted_cur_agent_indices: shape: (M,)  # M = 유효 에이전트 수
            - ego 로부터 가까운 순서대로 정렬된 현재 프레임 인덱스.
        dist_from_cur_agent_to_ego: shape: (current_agents_num,)
            - 각 에이전트의 ego 기준 2D 거리.
    """
    sorted_cur_agent_indices, dist_from_cur_agent_to_ego = \
        _compute_valid_sorted_indices(
            all_frame_np_agents_local=all_frame_np_agents_local, # (num_frames, current_agents_num, 9)
            filter_radius=filter_radius,
        )

    # 유효한 후보가 없거나, max_agent_num 이 0 이하인 경우
    if sorted_cur_agent_indices.size == 0 or max_agent_num <= 0:
        neighbor_agents_past = np.zeros((0, num_frames, agents_states_dim + 3),
                                        dtype=np.float32)
        neighbors_id = np.zeros((0,), dtype=np.float32)
        return neighbor_agents_past, neighbors_id, []

    # 1) 타입 상한을 끄는 경우: 거리 기준으로만 선택
    if (max_pedestrians is None) or (max_bicycles is None):
        # agents_cur_frame_indices: shape: (chosen_agent_num,), chosen_agent_num ≤ max_agent_num
        agents_cur_frame_indices = sorted_cur_agent_indices[:max_agent_num]
    else:
        # 2) 타입 상한을 적용하는 경우
        # agents_cur_frame_indices: shape: (chosen_agent_num,), chosen_agent_num ≤ max_agent_num
        agents_cur_frame_indices = _select_indices_with_type_cap(
            sorted_cur_agent_indices=sorted_cur_agent_indices,
            current_agent_types_list=current_agent_types_list,
            max_agent_num=max_agent_num,
            max_pedestrians=max_pedestrians,
            max_bicycles=max_bicycles,
            dist_from_cur_agent_to_ego=dist_from_cur_agent_to_ego,
        )

    # 실제 선택된 에이전트 수 chosen_agent_num
    chosen_agent_num = int(agents_cur_frame_indices.shape[0])
    if chosen_agent_num == 0:
        neighbor_agents_past = np.zeros((0, num_frames, agents_states_dim + 3),
                                        dtype=np.float32)
        neighbors_id = np.zeros((0,), dtype=np.float32)
        return neighbor_agents_past, np.array([], dtype=int), neighbors_id

    # 3) 선택된 인덱스로부터 최종 텐서 구성
    """
        neighbor_agents_past: shape: (chosen_agent_num, num_frames, 11)
            - [x, y, cos, sin, vx, vy, width, length,
               track_id, onehot_vehicle, onehot_ped, onehot_bike]
        neighbors_id: shape: (chosen_agent_num,)
            - 각 이웃 에이전트의 track_id (현재 프레임 기준).
    """
    neighbor_agents_past, neighbors_id = _build_neighbor_vectors(
        all_frame_np_agents_local=
        all_frame_np_agents_local,  # (num_frames, current_agents_num, 9)
        current_agent_types_list=current_agent_types_list,
        agents_states_dim=agents_states_dim,
        agents_cur_frame_indices=agents_cur_frame_indices,  # (chosen_agent_num,)
    )

    # numpy → List[int]
    agents_cur_frame_indices_list: List[int] = list(
        map(int,
            np.asarray(agents_cur_frame_indices).tolist()))
    agents_cur_frame_indices = np.asarray(agents_cur_frame_indices_list,
                                          dtype=int)  # (K,)
    return neighbor_agents_past, agents_cur_frame_indices, neighbors_id


from typing import Optional  # 이미 있을 가능성 높음


def _build_static_objects(
    present_static_feature_6: np.ndarray,  # (cur_static_num, 6)
    static_types_list: List[TrackedObjectType],  # 길이 = cur_static_num
    max_static_num: int,
    filter_radius: Optional[float] = None,
) -> np.ndarray:
    """정적 객체 정보를 거리 기준으로 정렬하고, 타입 one-hot 을 붙여 배열로 만든다.

    - 이고와의 거리를 기준으로 가까운 순서로 최대 `max_static_num` 개 선택.
    - `filter_radius` 가 주어지면, ego 기준 거리 <= filter_radius 인 것만 후보.
    - 실제 개수가 max_static_num 이하이면 zero-padding 없이 그 개수만 반환.

    Returns:
        static_objects: (K, 10), K = min(유효 정적 객체 수, max_static_num)
    """
    cur_static_num = int(present_static_feature_6.shape[0])

    if cur_static_num == 0 or max_static_num <= 0:
        return np.zeros((0, present_static_feature_6.shape[-1] + 4),
                        dtype=np.float32)

    # ego 기준 거리
    static_distance_to_ego = np.linalg.norm(
        present_static_feature_6[:, :2],
        axis=-1,
    )  # (cur_static_num,)

    # filter_radius 적용
    if filter_radius is not None:
        valid_mask = static_distance_to_ego <= float(filter_radius)
        valid_indices = np.nonzero(valid_mask)[0]
    else:
        valid_indices = np.arange(cur_static_num, dtype=int)

    if valid_indices.size == 0:
        return np.zeros((0, present_static_feature_6.shape[-1] + 4),
                        dtype=np.float32)

    # 유효 객체들 안에서 거리 기준 오름차순
    dist_valid = static_distance_to_ego[valid_indices]
    order_local = np.argsort(dist_valid)
    sorted_indices = valid_indices[order_local]

    # 실제 사용할 개수 K
    K = min(len(sorted_indices), max_static_num)

    static_objects = np.zeros(
        (K, present_static_feature_6.shape[-1] + 4),
        dtype=np.float32,
    )
    six_ = present_static_feature_6.shape[-1]

    for i, j in enumerate(sorted_indices[:K]):
        static_objects[i, :six_] = present_static_feature_6[j, :six_]
        if static_types_list[j] == TrackedObjectType.CZONE_SIGN:
            static_objects[i, six_:] = [1, 0, 0, 0]
        elif static_types_list[j] == TrackedObjectType.BARRIER:
            static_objects[i, six_:] = [0, 1, 0, 0]
        elif static_types_list[j] == TrackedObjectType.TRAFFIC_CONE:
            static_objects[i, six_:] = [0, 0, 1, 0]
        else:
            static_objects[i, six_:] = [0, 0, 0, 1]

    return static_objects


def build_static_feature(
    present_static_feat_5: np.ndarray,  # (cur_static_num, 5)
    static_types_list: List[TrackedObjectType],  # 길이 = cur_static_num
    max_static_num: int,
    ego_cur_pose_np: np.ndarray,  # (3,)
    filter_radius: Optional[float] = None,
) -> np.ndarray:
    """현재 프레임의 정적 객체들을 ego 기준 좌표계로 변환하고,
    가까운 순으로 최대 max_static_num 개까지만 선택한다.

    처리 순서
    ----------
    1) `present_static_feat_5` (월드 좌표계)를
       `_build_present_static_feature_6` 를 이용해 ego 기준으로 변환:
       - 입력: [x, y, heading, width, length]
       - 출력: [x, y, cos(heading), sin(heading), width, length]
       → shape: (cur_static_num, 6)

    2) `_build_static_objects` 를 호출해,
       - ego와의 2D 거리,
       - `filter_radius`,
       - `max_static_num`
       을 기준으로 정적 객체를 선택하고 타입 one-hot 을 붙인다.
       - 최종 출력: (K, 10)
         · [x, y, cos, sin, width, length,
            onehot_CZONE, onehot_BARRIER, onehot_CONE, onehot_GENERIC]
         · K = min(유효 정적 객체 수, max_static_num)

    Args:
        present_static_feat_5:
            - shape: (cur_static_num, 5)
            - [x, y, heading, width, length] (월드 좌표계)
        static_types_list:
            - 길이: cur_static_num
            - 각 정적 객체의 타입 (TrackedObjectType).
        max_static_num:
            - 선택할 정적 객체 수의 상한값.
            - 실제 K는 K ≤ max_static_num.
        ego_cur_pose_np:
            - shape: (3,), [x_ego, y_ego, yaw_ego]
        filter_radius:
            - None 이 아니면 ego 기준 거리 <= filter_radius 인 정적 객체만 후보.
            - None 이면 거리 제한 없음.

    Returns:
        np.ndarray:
            - static_objects
            - shape: (K, 10)
            - [x, y, cos, sin, width, length,
               onehot_CZONE, onehot_BARRIER, onehot_CONE, onehot_GENERIC]
    """
    # (cur_static_num, 6)
    present_static_feature_6 = _build_present_static_feature_6(
        present_static_feat_5=present_static_feat_5,
        ego_cur_pose_np=ego_cur_pose_np,
    )

    static_objects = _build_static_objects(
        present_static_feature_6=present_static_feature_6,
        static_types_list=static_types_list,
        max_static_num=max_static_num,
        filter_radius=filter_radius,
    )
    return static_objects


def build_neighbor_past_feature(
    past_cur_agents_world_8_list: List[np.ndarray],
    past_cur_agents_types_list: List[List[TrackedObjectType]],
    max_agent_num: int,
    ego_cur_pose_np: np.ndarray,  # (3,)
    max_pedestrians: Optional[int],
    max_bicycles: Optional[int],
    token_to_id: Dict[str, int],
    filter_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """이웃 에이전트의 과거+현재 궤적을 ego 기준으로 변환하고,
    가까운 에이전트들만 골라 (chosen_agent_num, T, 11) 텐서로 만든다.

    개념 요약
    --------
    1) `past_cur_agents_world_8_list` 를 이용해, 각 프레임에서
       [track_id, vx, vy, heading, width, length, x, y] 상태를 ego 기준으로 변환한다.
       → `build_agents_past_ego_frame_array` 로 (T, N, 9) 텐서 생성
         · T: num_frames, N: 현재 프레임에서 ego 기준 존재 여부 필터링 후 남는 에이전트 수
         · 채널: [x, y, cos(h), sin(h), vx, vy, width, length, id]

    2) 현재 프레임(T-1) 기준으로,
       - ego와의 거리,
       - 타입(차량/보행자/자전거),
       - 옵션으로 `filter_radius`,
       - 옵션으로 `max_pedestrians`, `max_bicycles`
       를 이용해 어떤 에이전트를 K개까지 선택할지 결정한다.
       → `_select_neighbor_agents_and_build_past` 호출

       · `max_pedestrians` 또는 `max_bicycles` 가 None 이면:
         → 타입은 무시하고, 단순히 거리 가까운 순으로 `max_agent_num` 개까지 선택.
       · 둘 다 정수면:
         → 보행자/자전거 상한 적용 후, 차량으로 나머지 채움.

       · `filter_radius` 가 주어지면:
         → ego 기준 거리 <= filter_radius 인 에이전트만 후보.

    3) 선택된 에이전트들에 대해서만
       (chosen_agent_num, T, 11) 텐서를 만든다.
       · 앞 8차원: [x, y, cos, sin, vx, vy, width, length]
       · 뒤 3차원: 타입 one-hot (vehicle, pedestrian, bicycle)
       · chosen_agent_num ≤ max_agent_num (실제 장면에 따라 K는 매번 달라질 수 있음)

    Args:
        past_cur_agents_world_8_list:
            - 길이: num_frames
            - 각 원소 shape: (frame_agents_num, 8)
            - [track_id, vx, vy, heading, width, length, x, y] (월드 좌표계)
        past_cur_agents_types_list:
            - 길이: num_frames
            - 각 프레임에서 에이전트 타입 리스트 (TrackedObjectType).
            - 현재 프레임(마지막 원소)의 타입 정보를 사용.
        max_agent_num:
            - 선택할 이웃 에이전트 수의 상한값.
            - 실제 선택 수 chosen_agent_num 는 chosen_agent_num ≤ max_agent_num.
        ego_cur_pose_np:
            - shape: (3,), [x_ego, y_ego, yaw_ego]
            - ego 현재 포즈.
        max_pedestrians:
            - None 이면 타입 상한 미사용 (거리 순으로만 선택).
            - 정수면 보행자 최대 개수 상한.
        max_bicycles:
            - None 이면 타입 상한 미사용 (거리 순으로만 선택).
            - 정수면 자전거 최대 개수 상한.
        filter_radius:
            - None 이 아니면 ego 기준 거리 <= filter_radius 인 에이전트만 후보.
            - None 이면 거리 제한 없음.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - neighbor_agents_past:
                · shape: (chosen_agent_num, num_frames, 11)
                · chosen_agent_num = 선택된 이웃 수 (chosen_agent_num ≤ max_agent_num)
                · [x, y, cos, sin, vx, vy, width, length, id,
                   onehot_vehicle, onehot_ped, onehot_bike]
            - agents_cur_frame_indices:
                · shape: (chosen_agent_num,)
                · 현재 프레임(마지막 프레임)의 에이전트 배열에서의 행 인덱스.
                · get_neighbor_track_tokens 에서 track_token 매핑에 사용.
            - neighbor_track_token:
                · shape: (chosen_agent_num,)
                · 선택된 이웃 에이전트들의 track_token 리스트.
    """
    agents_states_dim = 8  # x, y, cos h, sin h, vx, vy, width, length

    # 현재 프레임의 타입 정보 (길이 = current_agents_num)
    current_agent_types_list = past_cur_agents_types_list[-1]

    # (num_frames, current_agents_num, 9)
    all_frame_np_agents_local = build_agents_past_ego_frame_array(
        past_cur_agents_world_8_list=past_cur_agents_world_8_list,
        ego_cur_pose_np=ego_cur_pose_np,
        agents_states_dim=agents_states_dim,
    )

    # 필터/타입 상한/거리 기준으로 이웃 선택 + (chosen_agent_num, T, 11) 텐서 구성
    """
    neighbor_agents_past: (chosen_agent_num, num_frames, 11)
    agents_cur_frame_indices: shape (chosen_agent_num) (현재 프레임 기준 인덱스)
    neighbors_id:         (chosen_agent_num,)
    """
    neighbor_agents_past, agents_cur_frame_indices, neighbors_id = \
        _select_neighbor_agents_and_build_past(
            all_frame_np_agents_local=all_frame_np_agents_local,
            current_agent_types_list=current_agent_types_list,
            agents_states_dim=agents_states_dim,
            max_agent_num=max_agent_num,
            max_pedestrians=max_pedestrians,
            max_bicycles=max_bicycles,
            filter_radius=filter_radius,
        )
    id_to_token = {v: k for k, v in token_to_id.items()}
    neighbor_track_token: List[str] = []
    for track_id in neighbors_id:
        if track_id == -1:
            raise ValueError("Neighbor agent has invalid track_id -1.")
        else:
            neighbor_track_token.append(id_to_token[track_id])
    return neighbor_agents_past, agents_cur_frame_indices, neighbor_track_token


def agent_future_process(
    anchor_ego_state: np.ndarray,  # (3,)
    future_tracked_objects: List[
        np.ndarray],  # 길이 = 1(현재)+Tf, 각 원소: (frame_agents_num, 8)
    num_agents: int,  # 출력 이웃 슬롯 수
    neighbor_indices: Union[np.ndarray,
                            List[int]],  # 길이 K(≤ num_agents), 현재 프레임의 행 인덱스들
) -> np.ndarray:  # **(num_agents, Tf, 3)**
    """현재 시점에서 선택된 에이전트 집합(행 인덱스)을 유지한 채 미래 시퀀스를 (x, y, heading)로 생성합니다.
    개요:
        - 입력으로 들어온 `future_tracked_objects`(현재+미래 프레임들의 원시 감지 배열)를
          현재 프레임에 존재하는 에이전트만 남기도록 정렬/필터링합니다.
        - 각 프레임을 ego 기준 상대좌표계로 변환한 뒤, 에이전트 행 정렬을 고정하고
          결측 에이전트는 0으로 패딩합니다.
        - 과거 처리 단계에서 결정된 `neighbor_indices`(= 현재 프레임의 선택/순서)를 그대로 사용해
          같은 에이전트만, 같은 순서로 서브셋팅하여 최종 (num_agents, Tf, 3) 배열을 만듭니다.

    Args:
        anchor_ego_state (np.ndarray):
            - 모양: **(3,)**
            - 의미: [x_ego, y_ego, yaw_ego] (월드 좌표계, 현재 시점).
            - 용도: 절대 좌표/속도를 ego 상대좌표계로 변환할 때 기준으로 사용.
        future_tracked_objects (List[np.ndarray]):
            - 길이: **1 + Tf** (인덱스 0이 현재, 1..Tf 가 미래 프레임)
            - 각 원소 모양: **(frame_agents_num, 8)**
            - 스키마: `AgentInternalIndex` 순서( track_id, vx, vy, heading, width, length, x, y ).
        num_agents (int):
            - 출력할 이웃 슬롯(행)의 고정 개수. K < num_agents 인 경우 남는 행은 0으로 채워짐.
        neighbor_indices (Union[np.ndarray, List[int]]):
            - 모양: **(K, )**, 값: 정수 인덱스, **K ≤ num_agents**.
            - 의미: 과거 처리(`agent_past_process`)에서 결정된 "현재 프레임의 에이전트 행 인덱스 집합".
              이 순서가 최종 출력의 행 순서가 됩니다.

    Returns:
        np.ndarray:
            - 모양: **(num_agents, Tf, 3)**
            - 채널: [x, y, heading] (모두 **ego 상대좌표계**)
            - dtype: `np.float32`
            - 특이사항: 선택된 에이전트가 미래 프레임에서 사라진 구간은 0으로 패딩됩니다.
              또한 `neighbor_indices` 길이가 `num_agents`보다 작으면 남은 행은 전부 0입니다.

    Notes:
        - 내부 단계
            1) `_filter_agents_array(future_tracked_objects)`
               → 현재 프레임(리스트의 첫 원소)에 존재하는 에이전트만 남기도록 프레임별 배열을 일치.
            2) `convert_absolute_quantities_to_relative(..., anchor_ego_state, 'agent')`
               → 각 프레임을 ego 상대좌표계로 변환.
            3) `_pad_agent_states_with_zeros(...)`
               → 현재 프레임의 행 순서를 기준으로 모든 프레임의 행을 고정, 결측은 0 패딩.
            4) `neighbor_indices`로 서브셋팅
               → 과거에서 고른 동일 에이전트만, 동일한 행 순서로 (x, y, heading) 추출.
        - 안전성 전제: `neighbor_indices`는 현재 프레임의 유효 행 범위 내 정수 인덱스여야 합니다
          (보통 `agent_past_process`의 반환값을 그대로 넘기므로 보장됩니다).

    """
    # len: num_frames, np.ndarray shape: (frame_save_agents_num, 8) # frame_save_agents_num: 길이 가변적
    agent_future = _filter_agents_array(future_tracked_objects)

    local_coords_agent_states = []
    for agent_state in agent_future:
        # agent_state: (frame_agents_num, 8)  → ego 상대좌표계로 변환
        local_coords_agent_states.append(
            convert_absolute_quantities_to_relative(agent_state,
                                                    anchor_ego_state, 'agent'))

    # padded_agent_states: (1 + Tf, current_agents_num, 8)
    #  - 현재 프레임의 행 순서 고정, 결측은 0으로 패딩
    padded_agent_states = _pad_agent_states_with_zeros(
        local_coords_agent_states)
    future_len = padded_agent_states.shape[0] - 1  # Tf
    padded_agent_states_future = padded_agent_states[
        1:, :, :]  # (Tf, current_agents_num, 8)
    # 최종 결과 버퍼: (num_agents, Tf, 3)  ← 현재(인덱스 0) 제외한 미래 구간만 사용
    agent_futures = np.zeros(shape=(num_agents, future_len, 3),
                             dtype=np.float32)

    # agent_index의 순서가 곧 출력 행 순서가 된다.
    for i, key_frame_idx in enumerate(neighbor_indices):
        # padded_agent_states[1:, key_frame_idx, [x, y, heading]] → (Tf, 3)
        agent_futures[i] = padded_agent_states_future[:, key_frame_idx, [
            AgentInternalIndex.x(),
            AgentInternalIndex.y(),
            AgentInternalIndex.heading()
        ]]

    return agent_futures


def agent_future_all_process(
        anchor_ego_state: np.ndarray,  # (3,)
        future_tracked_objects: List[
            np.ndarray],  # 길이 = 1(현재)+Tf, 각 원소: (frame_agents_num, 8)
        neighbor_token_id: List[Optional[int]],  # (agent_num, )
) -> np.ndarray:  # (agents_num, 1 + Tf, 3)
    # len: num_frames, np.ndarray shape: (frame_save_agents_num, 8) # frame_save_agents_num: 길이 가변적
    agent_future = _filter_agents_array_w_token(future_tracked_objects,
                                                neighbor_token_id)

    local_coords_agent_states = []
    for agent_state in agent_future:
        # agent_state: (frame_agents_num, 8)  → ego 상대좌표계로 변환
        local_coords_agent_states.append(
            convert_absolute_quantities_to_relative(agent_state,
                                                    anchor_ego_state, 'agent'))

    # padded_agent_states: (agents_num, 1 + Tf, 3)
    padded_agent_states = _pad_agent_states_with_zeros_w_token(
        local_coords_agent_states, neighbor_token_id)
    return padded_agent_states
