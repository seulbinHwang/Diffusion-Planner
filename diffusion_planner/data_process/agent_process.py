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
    token_to_id: Optional[Dict[str, int]] = None,
) -> Tuple[List[np.ndarray], List[List[TrackedObjectType]], Dict[str, int]]:
    """여러 시점의 감지 결과를 프레임별 에이전트 배열 리스트로 바꾸는 함수.

    이 함수는 연속된 여러 시점(프레임)의 감지 결과를 입력으로 받아,
    각 시점마다 에이전트들을 숫자 배열로 바꾸어 리스트로 모아 줍니다.

    간단히 말해,
    - "시간에 따라 에이전트들이 어떻게 움직였는지"를
      프레임별 2차원 배열 목록으로 만드는 역할을 합니다.
    시간 흐름:
        - 입력 리스트의 0번 인덱스: 가장 과거 시점
        - 마지막 인덱스: 가장 최근 시점

    Args:
        tracked_objects_list (List[Union[TrackedObjects, DetectionsTracks]]):
            - 시간 순서대로 정렬된 감지 결과 리스트입니다.

    Returns:
        Tuple[List[np.ndarray], List[List[TrackedObjectType]], Dict[str, int]]:
            - agents_seq_world_8_list (List[np.ndarray]):
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

    # agents_seq_world_8_list:
    #   - 길이: num_frames
    #   - 각 원소: np.ndarray, shape (frame_agents_num, 8)
    agents_seq_world_8_list: List[np.ndarray] = []

    # past_cur_agents_types_list:
    #   - 길이: num_frames
    #   - 각 원소: List[TrackedObjectType], 길이 = frame_agents_num
    past_cur_agents_types_list: List[List[TrackedObjectType]] = []

    # track_token(문자열) → int ID
    if token_to_id is None:
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

        agents_seq_world_8_list.append(a_frame_agents_feat_8)
        past_cur_agents_types_list.append(agent_types)
    """
    agents_seq_world_8_list: List[np.ndarray]
        - 각 원소: (frame_agents_num, 8)  # frame_agents_num 은 프레임마다 다름
    past_cur_agents_types_list: List[List[TrackedObjectType]]
        - 각 원소 길이: frame_agents_num
    token_to_id: Dict[str, int]
        - track_token 문자열 → int ID 매핑
    """
    return agents_seq_world_8_list, past_cur_agents_types_list, token_to_id


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


def _filter_agents_array_w_id(
        cur_fut_agents_world_8_list: List[
            np.ndarray],  # len = num_frames_all, 각 원소: (frame_agents_num_t, 8)
        neighbor_token_id: np.ndarray,  # shape: (chosen_agent_num,)
) -> List[np.ndarray]:
    """각 프레임의 에이전트 배열에서, 원하는 track_id(neighbor_token_id)에
    해당하는 에이전트만 남긴다.

    하는 일
    --------
    - 입력으로 "현재 + 여러 미래 프레임"에 대한 에이전트 상태가 들어온다.
      · cur_fut_agents_world_8_list : List[np.ndarray]
        - 길이: num_frames_all
        shape: (frame_agents_num_t, 8)
        [track_id, vx, vy, heading, width, length, x, y]

    - neighbor_token_id 에는 "우리가 계속 추적하고 싶은 에이전트 ID" 들이 들어 있다.
      예: [10, 25, 31] 같은 1차원 배열.

    - 각 프레임에 대해:
        1) 해당 프레임의 track_id 열을 보고,
           neighbor_token_id 중 어떤 것들이 있는지 찾는다.
        2) 그 ID 를 가진 행만 남겨서 새 배열(frame_save_agents_num_t, 8)을 만든다.
        3) 한 프레임에 하나도 없다면 (0, 8) 빈 배열을 넣는다.

    결과적으로,
    - 원래 프레임 수는 그대로 유지하고
    - 각 프레임마다 "선택된 에이전트들만 남은 배열" 리스트를 돌려준다.

    Args:
        cur_fut_agents_world_8_list:
            - 길이: num_frames_all
            - 각 원소 shape: (frame_agents_num_t, 8)
              [track_id, vx, vy, heading, width, length, x, y] (월드 좌표).
        neighbor_token_id:
            - shape: (chosen_agent_num,)
            - 선택된 에이전트들의 track_id 배열.

    Returns:
        List[np.ndarray]:
            - cur_fut_chosen_agents_world_8_list
            - 길이: num_frames_all
            - 각 원소 shape: (frame_save_agents_num_t, 8)
              · frame_save_agents_num_t 는 프레임마다 달라질 수 있다.
              · 선택된 에이전트가 하나도 없으면 (0, 8) 배열.
    """
    cur_fut_chosen_agents_world_8_list: List[np.ndarray] = []

    # 선택된 토큰이 하나도 없으면, 모든 프레임에 대해 (0, 8) 빈 배열 반환
    if neighbor_token_id.size == 0:
        for frame_agents_world_8 in cur_fut_agents_world_8_list:
            cur_fut_chosen_agents_world_8_list.append(
                np.empty((0, frame_agents_world_8.shape[1]),
                         dtype=frame_agents_world_8.dtype))
        return cur_fut_chosen_agents_world_8_list

    # 비교를 편하게 하기 위해 float32 로 맞춘다.
    neighbor_token_id_f32 = neighbor_token_id.astype(np.float32, copy=False)

    for frame_agents_world_8 in cur_fut_agents_world_8_list:
        # frame_agents_world_8: (frame_agents_num_t, 8) 또는 (0, 8)
        if frame_agents_world_8.size == 0:
            cur_fut_chosen_agents_world_8_list.append(
                np.empty((0, frame_agents_world_8.shape[1]),
                         dtype=frame_agents_world_8.dtype))
            continue

        # 현재 프레임의 track_id 열: (frame_agents_num_t,)
        frame_ids = frame_agents_world_8[:, AgentInternalIndex.track_token()]

        # 이 프레임에서 neighbor_token_id 에 속하는 행만 True
        mask = np.isin(
            frame_ids.astype(np.float32, copy=False),
            neighbor_token_id_f32,  # (chosen_agent_num,)
        )  # shape: (frame_agents_num_t,)

        if np.any(mask):
            # 선택된 행만 남긴 배열: (frame_save_agents_num_t, 8)
            cur_fut_chosen_agents_world_8_list.append(
                frame_agents_world_8[mask])
        else:
            cur_fut_chosen_agents_world_8_list.append(
                np.empty((0, frame_agents_world_8.shape[1]),
                         dtype=frame_agents_world_8.dtype))

    return cur_fut_chosen_agents_world_8_list


def _pad_agent_states(
    all_frame_cur_exists_agents: List[
        np.ndarray],  # len = num_frames, 각 원소 shape: (current_agents_num_t, 8)
    reverse: bool,
) -> List[np.ndarray]:  # len = num_frames, 각 원소 shape: (current_agents_num, 8)
    """프레임마다 빠지는 에이전트를 0으로 채우면서, 에이전트 순서를 기준 프레임에 맞춘다.

    간단한 규칙:
        - 기준 프레임(현재 프레임 기준)에서 살아 있는 에이전트 집합을 고정한다.
        - 각 시점마다:
            · 해당 시점에 관측된 에이전트는 그 값으로 채우고
            · 관측되지 않은 에이전트는 [0, 0, ..., 0] 으로 둔다.
        - 모든 프레임의 행 개수/순서는 기준 프레임과 같다.

    Args:
        all_frame_cur_exists_agents (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num_t, 8)
              [track_id, vx, vy, heading, width, length, x, y]
        reverse (bool):
            - True 이면 리스트를 뒤집어서 마지막 프레임을 기준으로 사용한다.

    Returns:
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
              기준 프레임 에이전트 순서로 정렬되고, 없는 시점은 0으로 채워진 배열.
    """
    track_id_idx = AgentInternalIndex.track_token()

    # 시간 방향 뒤집기 (현재 프레임을 기준 프레임으로 쓰기 위해)
    if reverse:
        all_frame_cur_exists_agents = all_frame_cur_exists_agents[::-1]

    if len(all_frame_cur_exists_agents) == 0:
        return all_frame_cur_exists_agents

    # 기준 프레임: (current_agents_num, 8)
    key_frame_agents: np.ndarray = all_frame_cur_exists_agents[0]
    current_agents_num: int = int(key_frame_agents.shape[0])
    feature_dim: int = int(key_frame_agents.shape[1])

    if current_agents_num == 0:
        # 에이전트가 아예 없으면 그대로 반환
        return all_frame_cur_exists_agents

    # 기준 프레임에서 track_id → 행 인덱스 매핑
    key_frame_id_to_row: Dict[int, int] = {}
    for cur_agent_idx, agent_id in enumerate(key_frame_agents[:, track_id_idx]):
        key_frame_id_to_row[int(agent_id)] = cur_agent_idx

    # 기준 프레임의 id 벡터 (shape: (current_agents_num,))
    key_frame_ids: np.ndarray = key_frame_agents[:, track_id_idx].copy()

    new_all_frame_cur_exists_agents: List[np.ndarray] = []

    for time_idx in range(len(all_frame_cur_exists_agents)):
        # frame_cur_exists_agents: (current_agents_num_t, 8)
        frame_cur_exists_agents: np.ndarray = all_frame_cur_exists_agents[
            time_idx]

        # frame_state: (current_agents_num, 8), 처음엔 전부 0
        frame_state: np.ndarray = np.zeros(
            (current_agents_num, feature_dim),
            dtype=np.float64,
        )

        # track_id 칸은 기준 프레임 id 로 채워 둔다.
        frame_state[:, track_id_idx] = key_frame_ids

        # 현재 프레임에서 실제로 관측된 에이전트만 해당 행 위치에 복사
        for agent_row_idx in range(frame_cur_exists_agents.shape[0]):
            agent_id = int(frame_cur_exists_agents[agent_row_idx, track_id_idx])
            if agent_id not in key_frame_id_to_row:
                # 기준 프레임에 없는 id 는 이웃 후보가 아니므로 무시
                continue
            key_frame_row: int = key_frame_id_to_row[agent_id]
            frame_state[key_frame_row, :] = frame_cur_exists_agents[
                agent_row_idx, :]

        new_all_frame_cur_exists_agents.append(frame_state)

    # 시간 방향을 다시 원래대로 복원
    if reverse:
        new_all_frame_cur_exists_agents = new_all_frame_cur_exists_agents[::-1]

    return new_all_frame_cur_exists_agents


def _pad_agent_states_with_zeros_w_id(
        cur_fut_chosen_agents_local_8_list:
    List[
        np.
        ndarray],  # 길이: num_frames_all, 각 원소 shape: (frame_save_agents_num_t, 8)
        neighbor_token_id: np.ndarray,  # shape: (chosen_agent_num,)
        neighbor_types_one_hot: Optional[
            np.ndarray] = None,  # shape: (chosen_agent_num, 3) 또는 None
) -> np.ndarray:
    """선택된 이웃 토큰에 대해, 시간 전체 구간의 궤적을 고정 크기 텐서로 채운다.

    각 시점에서 관측된 에이전트의 상태는 해당 슬롯에 그대로 넣고,
    관측되지 않은 시점은 0으로 둔다. 타입 one-hot은 시간축 전체에 복사한다.

    Args:
        cur_fut_chosen_agents_local_8_list (List[np.ndarray]):
            - 길이: num_frames_all
            - 각 원소 shape: (frame_save_agents_num_t, 8)
              [track_id, vx, vy, heading, width, length, x, y] (ego 기준)
        neighbor_token_id (np.ndarray):
            - shape: (chosen_agent_num,)
            - 이웃 에이전트의 track_id 배열.
        neighbor_types_one_hot (Optional[np.ndarray]):
            - shape: (chosen_agent_num, 3)
            - [onehot_vehicle, onehot_ped, onehot_bike]
            - None 이면 타입 one-hot은 모두 0으로 둔다.

    Returns:
         np.ndarray:
            - cur_fut_chosen_agents_full_11:
                · shape: (chosen_agent_num, num_frames_all, 11)
                · [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]
    """
    track_id_idx = AgentInternalIndex.track_token()
    vx_idx = AgentInternalIndex.vx()
    vy_idx = AgentInternalIndex.vy()
    heading_idx = AgentInternalIndex.heading()
    width_idx = AgentInternalIndex.width()
    length_idx = AgentInternalIndex.length()
    x_idx = AgentInternalIndex.x()
    y_idx = AgentInternalIndex.y()

    # neighbor_token_id: (chosen_agent_num,)
    chosen_agent_num: int = int(neighbor_token_id.shape[0])
    # cur_fut_chosen_agents_local_8_list 길이: num_frames_all
    cur_future_all_len: int = len(cur_fut_chosen_agents_local_8_list)

    # 출력 버퍼 (모두 0으로 초기화)

    # cur_fut_chosen_agents_full_11: (chosen_agent_num, num_frames_all, 11)
    cur_fut_chosen_agents_full_11: np.ndarray = np.zeros(
        (chosen_agent_num, cur_future_all_len, 11),
        dtype=np.float32,
    )

    if chosen_agent_num == 0:
        return cur_fut_chosen_agents_full_11

    # 타입 one-hot: (chosen_agent_num, 3) → (chosen_agent_num, 1, 3) → 시간축 브로드캐스트
    if neighbor_types_one_hot is not None:
        # neighbor_types_one_hot: (chosen_agent_num, 3)
        neighbor_types_one_hot = neighbor_types_one_hot.astype(np.float32,
                                                               copy=False)
        # cur_fut_chosen_agents_full_11[:, :, 8:11]: (chosen_agent_num, num_frames_all, 3)
        cur_fut_chosen_agents_full_11[:, :,
                                      8:] = neighbor_types_one_hot[:, None, :]

    # 각 시점 프레임 처리
    for timestep_idx, frame_chosen_agents_local_8 in enumerate(
            cur_fut_chosen_agents_local_8_list):
        # frame_chosen_agents_local_8: (frame_chosen_agents_num_t, 8) 또는 (0, 8)
        if frame_chosen_agents_local_8.shape[0] == 0:
            continue

        # 현재 프레임의 track_id 들: (frame_chosen_agents_num_t,)
        frame_ids: np.ndarray = frame_chosen_agents_local_8[:,
                                                            track_id_idx].astype(
                                                                np.float32,
                                                                copy=False)

        # frame_ids 와 neighbor_token_id 의 공통 원소 및 위치
        # inter: 공통 track_id 값들 (길이 M)
        # idx_frame: frame_ids 에서의 인덱스 (shape: (M,))
        # idx_neighbors: neighbor_token_id 에서의 인덱스 (shape: (M,))
        inter, idx_frame, idx_neighbors = np.intersect1d(
            frame_ids,
            neighbor_token_id.astype(np.float32, copy=False),
            assume_unique=False,
            return_indices=True,
        )
        if inter.size == 0:
            continue

        # 선택된 행만 모으기
        # frame_selected: (M, 8)
        frame_selected: np.ndarray = frame_chosen_agents_local_8[idx_frame]

        # 위치/각도: (M,)
        x_local: np.ndarray = frame_selected[:, x_idx].astype(np.float32,
                                                              copy=False)
        y_local: np.ndarray = frame_selected[:, y_idx].astype(np.float32,
                                                              copy=False)
        heading_local: np.ndarray = frame_selected[:, heading_idx].astype(
            np.float32, copy=False)

        # 속도/크기: (M,)
        vx_local: np.ndarray = frame_selected[:, vx_idx].astype(np.float32,
                                                                copy=False)
        vy_local: np.ndarray = frame_selected[:, vy_idx].astype(np.float32,
                                                                copy=False)
        width_local: np.ndarray = frame_selected[:,
                                                 width_idx].astype(np.float32,
                                                                   copy=False)
        length_local: np.ndarray = frame_selected[:,
                                                  length_idx].astype(np.float32,
                                                                     copy=False)

        # ---- (2) full 11차원 채우기: [x, y, cos, sin, vx, vy, w, h, one_hot(3)] ----
        # cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx, :8]: (M, 8)
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx, 0] = x_local
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx, 1] = y_local
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx,
                                      2] = np.cos(heading_local)
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx,
                                      3] = np.sin(heading_local)
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx, 4] = vx_local
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx, 5] = vy_local
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx,
                                      6] = width_local
        cur_fut_chosen_agents_full_11[idx_neighbors, timestep_idx,
                                      7] = length_local
        # 8:11 (one_hot)은 위에서 시간축 전체에 이미 채워져 있음

    return cur_fut_chosen_agents_full_11


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
            np.ndarray],  # len = num_frames, 각 원소: (current_agents_num_t, 8)
        ego_cur_pose_np: np.ndarray,  # (3,)
) -> List[np.ndarray]:
    """각 프레임의 에이전트 상태를 먼저 ego 기준으로 바꾸고,
    이후에 프레임별 크기를 맞춰준다.

    처리 단계:
        1) 각 시점의 (world) 에이전트 상태를 개별적으로 ego 기준으로 변환.
           - 입력: (current_agents_num_t, 8)
           - 출력: (current_agents_num_t, 8)  (좌표계만 변경)
        2) `_pad_agent_states` 를 이용해
           - 기준 프레임 에이전트 집합/순서에 맞춰
           - 없는 시점은 0으로 채운다.

    Args:
        all_frame_cur_exists_agents (List[np.ndarray]):
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num_t, 8)
              [track_id, vx, vy, heading, width, length, x, y]
              (월드 좌표계 기준)
        ego_cur_pose_np (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, yaw_ego]

    Returns:
        List[np.ndarray]:
            - 길이: num_frames
            - 각 원소 shape: (current_agents_num, 8)
              ego 기준 좌표계이며, 기준 프레임 에이전트 순서를 따른다.
              관측이 없는 시점은 0으로 채워진다.
    """
    # 에이전트가 없으면 그대로 반환
    if len(all_frame_cur_exists_agents) == 0:
        return all_frame_cur_exists_agents

    all_frame_cur_exists_agents_local_world: List[np.ndarray] = []

    for frame_agents_world_8 in all_frame_cur_exists_agents:
        # frame_agents_world_8: (current_agents_num_t, 8)
        if frame_agents_world_8.shape[0] == 0:
            # 관측이 아예 없는 프레임은 0 배열 유지
            all_frame_cur_exists_agents_local_world.append(
                np.zeros_like(frame_agents_world_8, dtype=np.float64))
            continue

        # 복사본을 만들어 좌표계를 변환 (shape: (current_agents_num_t, 8))
        frame_agents_world_8_copy = frame_agents_world_8.astype(np.float64,
                                                                copy=True)
        frame_agents_local_8 = convert_absolute_quantities_to_relative(
            frame_agents_world_8_copy,
            ego_cur_pose_np,
            'agent',
        )
        all_frame_cur_exists_agents_local_world.append(frame_agents_local_8)

    # 기준 프레임에 맞춰 에이전트 개수/순서를 고정하고, 없는 시점은 0 유지
    all_frame_cur_exists_agents_local: List[np.ndarray] = _pad_agent_states(
        all_frame_cur_exists_agents=all_frame_cur_exists_agents_local_world,
        reverse=True,
    )

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
    all_frame_np_agents_local = np.zeros(
        (num_frames, current_agents_num, agents_states_dim + 1),
        dtype=np.float64,
    )

    for past_idx in range(len(all_frame_cur_exists_agents_local)):
        # frame_cur_exists_agents_local: (current_agents_num, 8)
        #   [track_id, vx, vy, heading, width, length, x, y]
        frame_cur_exists_agents_local = all_frame_cur_exists_agents_local[past_idx]

        # --- 패딩 row(관측 없는 프레임) 마스크 계산 -------------------
        # frame_non_id: (current_agents_num, 8)
        frame_non_id = frame_cur_exists_agents_local.copy()
        # track_id 는 그대로 두고 싶기 때문에, 마스크 계산할 때만 0으로 만들어 둔다.
        frame_non_id[:, AgentInternalIndex.track_token()] = 0.0
        # empty_mask: (current_agents_num,)
        #  -> track_id 를 제외한 나머지 값들이 모두 0인 row 를 "빈 상태/패딩"으로 본다.
        empty_mask = (np.abs(frame_non_id).sum(axis=-1) == 0.0)

        # heading: (current_agents_num,)
        # ※ squeeze() 쓰지 말고, 항상 1차원 배열로 유지
        heading = frame_cur_exists_agents_local[:, AgentInternalIndex.heading()]
        heading = heading.astype(np.float64, copy=False).reshape(-1)  # (current_agents_num,)

        # cos_heading, sin_heading: (current_agents_num,)
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)

        # 관측이 없는 row 는 cos/sin 도 0으로 강제해서
        # [x, y, cos, sin, vx, vy, w, l] 8채널이 전부 0이 되도록 맞춘다.
        cos_heading[empty_mask] = 0.0
        sin_heading[empty_mask] = 0.0

        # x
        all_frame_np_agents_local[
            past_idx, :, 0
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.x()].squeeze()
        # y
        all_frame_np_agents_local[
            past_idx, :, 1
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.y()].squeeze()
        # cos(heading)
        all_frame_np_agents_local[past_idx, :, 2] = cos_heading
        # sin(heading)
        all_frame_np_agents_local[past_idx, :, 3] = sin_heading
        # vx
        all_frame_np_agents_local[
            past_idx, :, 4
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.vx()].squeeze()
        # vy
        all_frame_np_agents_local[
            past_idx, :, 5
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.vy()].squeeze()
        # width
        all_frame_np_agents_local[
            past_idx, :, 6
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.width()].squeeze()
        # length
        all_frame_np_agents_local[
            past_idx, :, 7
        ] = frame_cur_exists_agents_local[:, AgentInternalIndex.length()].squeeze()
        # id (track_token)
        all_frame_np_agents_local[
            past_idx, :, 8
        ] = frame_cur_exists_agents_local[
            :, AgentInternalIndex.track_token()
        ].squeeze()

        # (선택 사항) 완전히 all-zero 패딩을 원하면, 아래 한 줄을 켜서
        # 빈 row 의 track_id 도 0으로 만들 수 있다.
        # all_frame_np_agents_local[past_idx, empty_mask, 8] = 0.0

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
        valid_mask = dist_from_cur_agent_to_ego <= float(
            filter_radius)  # (current_agents_num,)
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
        caching_max_agent_num: int,
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
           caching_max_agent_num 개까지 사용한다.

    Returns:
        agents_cur_frame_indices:
            - shape: (K,), K ≤ caching_max_agent_num
    """
    if caching_max_agent_num <= 0:
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
    ped_cap = min(max_pedestrians, caching_max_agent_num)
    sel_peds = ped_sorted_indices[:ped_cap]

    remain = caching_max_agent_num - len(sel_peds)
    bike_cap = min(max_bicycles, remain)
    sel_bikes = bike_sorted_indices[:bike_cap]

    remain -= len(sel_bikes)
    sel_vehs = vehicle_sorted_indices[:max(0, remain)]

    # 1차 선택 결과
    selected_indices = sel_peds + sel_bikes + sel_vehs

    if not selected_indices:
        return np.zeros((0,), dtype=int)

    # 최종 후보를 ego-거리 기준으로 다시 정렬 후 caching_max_agent_num 까지만 사용
    agents_cur_frame_indices = np.array(
        sorted(
            selected_indices,
            key=lambda idx: dist_from_cur_agent_to_ego[idx],
        )[:caching_max_agent_num],
        dtype=int,
    )
    return agents_cur_frame_indices


def _build_neighbor_vectors(
        all_frame_np_agents_local: np.
    ndarray,  # (num_frames, current_agents_num, 9)
        current_agent_types_list:
    List[
        TrackedObjectType],  # List[TrackedObjectType],  # 길이 = current_agents_num
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
    caching_max_agent_num: int,
    max_pedestrians: Optional[int],
    max_bicycles: Optional[int],
    filter_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """이고 기준 에이전트 텐서에서, 가까운 에이전트들을 선택하여
    이웃 궤적 텐서를 구성한다.

    특징:
        - `caching_max_agent_num` 은 상한(최대 개수)만 의미한다. 실제 선택 수 chosen_agent_num ≤ caching_max_agent_num.
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

    # 유효한 후보가 없거나, caching_max_agent_num 이 0 이하인 경우
    if sorted_cur_agent_indices.size == 0 or caching_max_agent_num <= 0:
        neighbor_agents_past = np.zeros(
            (0, num_frames, agents_states_dim + 3),
            dtype=np.float32,
        )
        agents_cur_frame_indices = np.zeros((0,), dtype=int)
        neighbors_id = np.zeros((0,), dtype=np.float32)
        return neighbor_agents_past, agents_cur_frame_indices, neighbors_id

    # 1) 타입 상한을 끄는 경우: 거리 기준으로만 선택
    if (max_pedestrians is None) or (max_bicycles is None):
        # agents_cur_frame_indices: shape: (chosen_agent_num,), chosen_agent_num ≤ caching_max_agent_num
        agents_cur_frame_indices = sorted_cur_agent_indices[:
                                                            caching_max_agent_num]
    else:
        # 2) 타입 상한을 적용하는 경우
        # agents_cur_frame_indices: shape: (chosen_agent_num,), chosen_agent_num ≤ caching_max_agent_num
        agents_cur_frame_indices = _select_indices_with_type_cap(
            sorted_cur_agent_indices=sorted_cur_agent_indices,
            current_agent_types_list=
            current_agent_types_list,  # List[TrackedObjectType],  # 길이 = current_agents_num
            caching_max_agent_num=caching_max_agent_num,
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
        current_agent_types_list=
        current_agent_types_list,  # List[TrackedObjectType],  # 길이 = current_agents_num
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
    caching_max_agent_num: int,
    ego_cur_pose_np: np.ndarray,  # (3,)
    max_pedestrians: Optional[int],
    max_bicycles: Optional[int],
    token_to_id: Dict[str, int],
    filter_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
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
         → 타입은 무시하고, 단순히 거리 가까운 순으로 `caching_max_agent_num` 개까지 선택.
       · 둘 다 정수면:
         → 보행자/자전거 상한 적용 후, 차량으로 나머지 채움.

       · `filter_radius` 가 주어지면:
         → ego 기준 거리 <= filter_radius 인 에이전트만 후보.

    3) 선택된 에이전트들에 대해서만
       (chosen_agent_num, T, 11) 텐서를 만든다.
       · 앞 8차원: [x, y, cos, sin, vx, vy, width, length]
       · 뒤 3차원: 타입 one-hot (vehicle, pedestrian, bicycle)
       · chosen_agent_num ≤ caching_max_agent_num (실제 장면에 따라 K는 매번 달라질 수 있음)

    Args:
        past_cur_agents_world_8_list:
            - 길이: num_frames
            - 각 원소 shape: (frame_agents_num, 8)
            - [track_id, vx, vy, heading, width, length, x, y] (월드 좌표계)
        past_cur_agents_types_list:
            - 길이: num_frames
            - 각 프레임에서 에이전트 타입 리스트 (TrackedObjectType).
            - 현재 프레임(마지막 원소)의 타입 정보를 사용.
        caching_max_agent_num:
            - 선택할 이웃 에이전트 수의 상한값.
            - 실제 선택 수 chosen_agent_num 는 chosen_agent_num ≤ caching_max_agent_num.
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
                · chosen_agent_num = 선택된 이웃 수 (chosen_agent_num ≤ caching_max_agent_num)
                · [x, y, cos, sin, vx, vy, width, length, id,
                   onehot_vehicle, onehot_ped, onehot_bike]
            - agents_cur_frame_indices:
                · shape: (chosen_agent_num,)
                · 현재 프레임(마지막 프레임)의 에이전트 배열에서의 행 인덱스.
                · get_neighbor_track_tokens 에서 track_token 매핑에 사용.
            - neighbors_id:
                · shape: (chosen_agent_num,)
                · 각 이웃 에이전트의 track_id (현재 프레임 기준).
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
            caching_max_agent_num=caching_max_agent_num,
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
    return neighbor_agents_past, agents_cur_frame_indices, neighbors_id, neighbor_track_token


def agent_future_all_process(
        ego_cur_pose_np: np.ndarray,  # shape: (3,) = [x_ego, y_ego, yaw_ego]
        cur_fut_agents_world_8_list: List[
            np.ndarray],  # 길이: 1+Tf_all, 각 원소 shape: (frame_agents_num_t, 8)
        neighbor_token_id: np.ndarray,  # shape: (chosen_agent_num,)
        neighbor_agents_past: Optional[
            np.ndarray] = None,  # (chosen_agent_num, Tp, 11) 또는 None
) -> np.ndarray:
    """선택된 이웃 에이전트에 대해, 현재+미래 전체 구간의 궤적을 3차원/11차원 둘 다 만든다.

    흐름:
        1) track_id 로 원하는 에이전트만 골라서 프레임별 리스트로 만든다.
        2) 각 프레임을 ego 기준 좌표계로 변환한다.
        3) 시간-에이전트 고정 크기 텐서로 채우되, 관측 없는 시점은 0으로 둔다.
        4) 타입 one-hot 은 neighbor_agents_past 에서 가져와 시간축 전체에 복사한다.

    Args:
        ego_cur_pose_np (np.ndarray):
            - shape: (3,)
            - [x_ego, y_ego, yaw_ego]
        cur_fut_agents_world_8_list (List[np.ndarray]):
            - 길이: num_frames_all = 1 + Tf_all
            - 각 원소 shape: (frame_agents_num_t, 8)
              [track_id, vx, vy, heading, width, length, x, y] (월드 좌표계)
        neighbor_token_id (np.ndarray):
            - shape: (chosen_agent_num,)
            - 선택된 이웃 에이전트의 track_id 배열.
        neighbor_agents_past (Optional[np.ndarray]):
            - shape: (chosen_agent_num, Tp, 11)
              과거 이웃 궤적 텐서. 마지막 3차원에 타입 one-hot 이 들어있다.
              None 이면 미래 쪽 타입 one-hot 은 0으로 둔다.

    Returns:
        np.ndarray
            - cur_fut_chosen_agents_full_11:
                · shape: (chosen_agent_num, num_frames_all, 11)
                · [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]
    """
    # 1) track_id 기준으로 원하는 에이전트만 남기기
    # cur_fut_chosen_agents_world_8_list:
    #   길이: num_frames_all, 각 원소 shape: (frame_save_agents_num_t, 8)
    cur_fut_chosen_agents_world_8_list: List[
        np.ndarray] = _filter_agents_array_w_id(
            cur_fut_agents_world_8_list,
            neighbor_token_id,
        )

    # 2) 각 프레임을 ego 기준 좌표계로 변환
    cur_fut_chosen_agents_local_8_list: List[np.ndarray] = []
    for frame_chosen_agents_world_8 in cur_fut_chosen_agents_world_8_list:
        # frame_chosen_agents_world_8: (frame_save_agents_num_t, 8)
        # → [track_id, vx, vy, heading, width, length, x, y] (ego 기준) 으로 변환
        frame_local_8: np.ndarray = convert_absolute_quantities_to_relative(
            frame_chosen_agents_world_8,
            ego_cur_pose_np,
            'agent',
        )
        cur_fut_chosen_agents_local_8_list.append(frame_local_8)

    # 3) 타입 one-hot 준비 (neighbor_agents_past 가 있을 때만)
    neighbor_types_one_hot: Optional[np.ndarray] = None
    if neighbor_agents_past is not None and neighbor_agents_past.size > 0:
        # neighbor_agents_past: (chosen_agent_num, Tp, 11)
        # 타입 one-hot 은 마지막 차원 8:11
        # neighbor_types_one_hot: (chosen_agent_num, 3)
        neighbor_types_one_hot = neighbor_agents_past[:, -1,
                                                      8:11].astype(np.float32,
                                                                   copy=False)

    # 4) 시간축 패딩 + xyh / full_11 텐서 만들기
    """
    cur_fut_chosen_agents_full_11: (chosen_agent_num, num_frames_all, 11)
    """
    cur_fut_chosen_agents_full_11 = _pad_agent_states_with_zeros_w_id(
        cur_fut_chosen_agents_local_8_list=cur_fut_chosen_agents_local_8_list,
        neighbor_token_id=neighbor_token_id,
        neighbor_types_one_hot=neighbor_types_one_hot,
    )

    return cur_fut_chosen_agents_full_11
