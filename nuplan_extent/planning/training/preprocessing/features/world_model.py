from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from typing import Any, Dict, List, Optional, Sequence, Set

import torch
from torch.utils.data.dataloader import default_collate

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)

# 필수 필드 정의 기준:
# - dataclass 정의에서 "default가 없는 필드(항상 있어야 하는 입력)"만 필수로 둡니다.
# - static_objects는 Optional[FeatureDataType] = None 으로 바뀌었으므로 필수에서 제외합니다.
_REQUIRED_FIELD_NAMES: Set[str] = {
    "ego_agent_past",
    "neighbor_agents_past",
    "lanes",
    "lanes_speed_limit",
    "lanes_has_speed_limit",
}


def _validate_required_fields_not_none(
    field_dict: Dict[str, Any],
    required_field_names: Sequence[str],
) -> None:
    """필수 필드가 None이 아닌지 확인합니다.

    여기서 말하는 "필수 필드"는,
    학습/추론 파이프라인이 정상 동작하려면 항상 값이 있어야 하는 입력들입니다.

    이 검사를 두는 이유는,
    나중 단계에서 shape 접근 같은 곳에서 더 복잡하게 터지는 대신
    "필수 입력이 없다"는 사실을 여기서 바로 알 수 있게 하기 위함입니다.

    Args:
        field_dict: 필드명 -> 값 매핑 딕셔너리.
        required_field_names: 반드시 None이 아니어야 하는 필드명 목록.

    Raises:
        ValueError: 필수 필드 중 하나라도 None이면 발생.
    """
    missing: List[str] = []
    for name in required_field_names:
        if field_dict.get(name, None) is None:
            missing.append(name)

    if len(missing) > 0:
        raise ValueError(f"필수 필드가 None입니다. 필수 필드는 항상 값이 있어야 합니다: {missing}")


def _to_tensor_or_none(
    value: Any,
    *,
    device: Optional[torch.device],
    make_contiguous: bool,
) -> Optional[torch.Tensor]:
    """값을 torch.Tensor로 바꾸되, None이면 None을 유지합니다.

    이 함수는 다음을 한 번에 처리합니다.
    - value가 None이면: 그대로 None 반환
    - value가 numpy 배열/리스트 등이라면: torch.Tensor로 변환
    - value가 이미 torch.Tensor라면: 그대로 사용
    - device가 주어지면: 해당 위치(CPU/GPU)로 이동
    - make_contiguous=True면: 텐서가 메모리에 연속된 형태가 되도록 정리

    Args:
        value: 입력 값. (예: np.ndarray, torch.Tensor, 또는 None)
        device: 옮길 위치. None이면 이동하지 않음.
        make_contiguous: True면 .contiguous()를 적용.

    Returns:
        Optional[torch.Tensor]:
            - None이면 None
            - 아니면 torch.Tensor

    Note:
        shape은 입력이 가진 shape을 그대로 유지합니다.
        예) lanes: (B, lane_num, lane_len, 12) -> 그대로 유지
    """
    if value is None:
        return None

    tensor = value if isinstance(value, torch.Tensor) else to_tensor(value)

    if device is not None:
        tensor = tensor.to(device=device)

    if make_contiguous:
        tensor = tensor.contiguous()

    return tensor


def _collate_field_values(
    values: List[Any],
    *,
    field_name: str,
    allow_none: bool,
) -> Any:
    """하나의 필드에 대해 배치로 묶습니다.

    규칙:
    - allow_none=False(필수 필드):
        값 중 하나라도 None이면 에러
    - allow_none=True(선택 필드):
        * 전부 None이면 -> 결과도 None
        * 일부만 None이면 -> 에러(배치 내 일관성이 깨짐)
        * 전부 값이 있으면 -> default_collate로 묶기

    Args:
        values: batch에서 해당 필드만 뽑은 리스트.
            예) ego_agent_past라면 길이 B인 리스트, 각 원소 shape (time_len, 11)
        field_name: 에러 메시지에 넣을 필드 이름.
        allow_none: None을 허용할지 여부.

    Returns:
        collated_value:
            - allow_none=True이고 전부 None이면 None
            - 그 외에는 default_collate 결과
              예) (B, time_len, 11) 또는 (B, lane_num, lane_len, 12)

    Raises:
        ValueError: 배치 내 None 섞임/필수 필드 None 등 일관성 문제.
    """
    has_any_none = any(v is None for v in values)

    if not allow_none:
        if has_any_none:
            raise ValueError(f"필수 필드 '{field_name}'에 None이 포함되어 있습니다. "
                             f"필수 필드는 배치에서 항상 값이 있어야 합니다.")
        return default_collate(values)

    if has_any_none:
        if not all(v is None for v in values):
            raise ValueError(
                f"필드 '{field_name}'는 배치에서 전부 None이거나 전부 값이 있어야 합니다. "
                f"현재는 일부만 None입니다.")
        return None

    return default_collate(values)


def _slice_batch_value_or_none(value: Any, batch_index: int) -> Any:
    """배치 텐서/배열에서 batch_index만 꺼내되, None이면 None을 유지합니다.

    Args:
        value: 배치 형태의 값 또는 None.
            예) lanes: (B, lane_num, lane_len, 12)
        batch_index: 꺼낼 인덱스 i (0 <= i < B)

    Returns:
        Any:
            - value가 None이면 None
            - 아니면 value[i] (배치 차원만 제거)
              예) (lane_num, lane_len, 12)

    Note:
        이 함수는 "배치 차원(B)"만 제거합니다.
    """
    if value is None:
        return None
    return value[batch_index]


@dataclass
class WorldModelFeature(AbstractModelFeature):
    ########### SAME AS LEARNING INPUT ###########
    ego_agent_past: FeatureDataType  # (time_len, 11)
    neighbor_agents_past: FeatureDataType  # (max_agent_num, time_len, 11)
    near_agents_past: FeatureDataType  # (Pnn, time_len, 11)4
    non_near_agents_past: FeatureDataType  # (max_agent_num - Pnn, time_len, 11)
    ########### SAME AS LEARNING INPUT ###########
    lanes: FeatureDataType  # (lane_num, lane_len, 12)
    lanes_speed_limit: FeatureDataType  # (lane_num, 1)
    lanes_has_speed_limit: FeatureDataType  # (lane_num, 1)
    route_lanes: Optional[FeatureDataType] = None  # (route_num, lane_len, 12)
    route_lanes_speed_limit: Optional[FeatureDataType] = None  # (route_num, 1)
    route_lanes_has_speed_limit: Optional[
        FeatureDataType] = None  # (route_num, 1)
    agent_route_lane_order: Optional[
        FeatureDataType] = None  # (Pnn, lane_num) # -1 if not on route

    static_objects: Optional[FeatureDataType] = None  # (static_objects_num, 10)

    ################################################
    stop_sign_points: Optional[
        FeatureDataType] = None  # (stop_sign_num, safety_len, 2)
    crosswalk_points: Optional[
        FeatureDataType] = None  # (crosswalk_num, safety_len, 2)
    speed_bump_points: Optional[
        FeatureDataType] = None  # (speed_bump_num, safety_len, 2)
    driveway_points: Optional[
        FeatureDataType] = None  # (driveway_num, safety_len, 2)
    #####################################
    #############
    lane_type: Optional[FeatureDataType] = None  # (lane_num, 4)
    left_line_type: Optional[FeatureDataType] = None  # (lane_num, 13)
    right_line_type: Optional[FeatureDataType] = None  # (lane_num, 13)
    road_edge: Optional[
        FeatureDataType] = None  # (chosen_edge_num, safety_len, 2)
    road_edge_type: Optional[FeatureDataType] = None  # (chosen_edge_num, 3)
    ################################################
    ################ inference only ################
    ego_agent_next_11_dim: Optional[
        FeatureDataType] = None  # (interpol_num, 11)
    planner_future_11_dim: Optional[FeatureDataType] = None  # (future_len, 11)

    #########################
    ######## validity ########
    # diffusion_planner/utils/validity.py 의 build_validity_key_dict()가 만드는
    # 모든 `~~~_is_valid` 키를 그대로 필드로 반영합니다.
    ego_agent_past_is_valid: Optional[
        FeatureDataType] = None  # (time_len,) bool
    ego_future_gt_is_valid: Optional[
        FeatureDataType] = None  # (future_len,) bool

    neighbor_agents_past_is_valid: Optional[
        FeatureDataType] = None  # (max_agent_num, time_len) bool
    neighbor_agents_is_valid: Optional[
        FeatureDataType] = None  # (max_agent_num,) bool
    neighbor_future_gt_is_valid: Optional[
        FeatureDataType] = None  # (max_agent_num,) bool

    stop_sign_is_valid: Optional[
        FeatureDataType] = None  # (stop_sign_num,) bool
    crosswalk_is_valid: Optional[
        FeatureDataType] = None  # (crosswalk_num,) bool
    speed_bump_is_valid: Optional[
        FeatureDataType] = None  # (speed_bump_num,) bool
    driveway_is_valid: Optional[FeatureDataType] = None  # (driveway_num,) bool

    lanes_len_is_valid: Optional[
        FeatureDataType] = None  # (lane_num, lane_len) bool
    lanes_is_valid: Optional[FeatureDataType] = None  # (lane_num,) bool

    static_objects_is_valid: Optional[
        FeatureDataType] = None  # (static_objects_num,) bool

    route_lanes_len_is_valid: Optional[
        FeatureDataType] = None  # (route_num, lane_len) bool
    route_lanes_is_valid: Optional[FeatureDataType] = None  # (route_num,) bool

    agent_route_lane_order_is_valid: Optional[
        FeatureDataType] = None  # (Pnn,) bool
    road_edge_is_valid: Optional[
        FeatureDataType] = None  # (chosen_edge_num,) bool

    #########################

    def to_feature_tensor(self) -> WorldModelFeature:
        """모든 필드를 torch.Tensor로 변환해 반환합니다.

        동작 규칙:
        - 값이 None인 필드는 그대로 None 유지
        - None이 아닌 값은 torch.Tensor로 변환
        - 변환된 텐서는 메모리 형태를 정리(contiguous)해서 반환

        Returns:
            WorldModelFeature:
                모든 값이 torch.Tensor(또는 None)인 새 객체.

        Note:
            대표 shape(배치 전 기준):
            - ego_agent_past: (time_len, 11)
            - neighbor_agents_past: (max_agent_num, time_len, 11)
            - lanes: (lane_num, lane_len, 12)
            - static_objects: (static_objects_num, 10) or None
            - driveway_points: (driveway_num, safety_len, 2) or None
            - ego_agent_past_is_valid: (time_len,) bool or None
            - lanes_is_valid: (lane_num,) bool or None
        """
        tensor_dict: Dict[str, Any] = {}
        for field in dataclass_fields(type(self)):
            name = field.name
            value = getattr(self, name)
            tensor_dict[name] = _to_tensor_or_none(value,
                                                   device=None,
                                                   make_contiguous=True)

        _validate_required_fields_not_none(tensor_dict, _REQUIRED_FIELD_NAMES)
        return type(self)(**tensor_dict)

    def to_device(self, device: torch.device) -> WorldModelFeature:
        """모든 텐서를 지정한 위치(CPU/GPU)로 옮겨 반환합니다.

        동작 규칙:
        - 값이 None인 필드는 그대로 None 유지
        - None이 아닌 값은 torch.Tensor로 변환 후 device로 이동

        Args:
            device: 옮길 위치. 예) torch.device("cuda"), torch.device("cpu")

        Returns:
            WorldModelFeature:
                모든 텐서가 device로 이동된 새 객체.

        Note:
            대표 shape(배치 후 기준 예시):
            - lanes: (B, lane_num, lane_len, 12)
            - static_objects: (B, static_objects_num, 10) or None
            - road_edge: (B, chosen_edge_num, safety_len, 2) or None
            - lanes_len_is_valid: (B, lane_num, lane_len) bool or None
        """
        tensor_dict: Dict[str, Any] = {}
        for field in dataclass_fields(type(self)):
            name = field.name
            value = getattr(self, name)
            tensor_dict[name] = _to_tensor_or_none(value,
                                                   device=device,
                                                   make_contiguous=False)

        _validate_required_fields_not_none(tensor_dict, _REQUIRED_FIELD_NAMES)
        return type(self)(**tensor_dict)

    @classmethod
    def collate(cls, batch: List[WorldModelFeature]) -> WorldModelFeature:
        """여러 샘플(WorldModelFeature)을 하나의 배치로 묶습니다.

        동작 규칙:
        - 필수 필드(ego_agent_past 등)는 배치 내에 None이 있으면 에러
        - 선택 필드(static_objects 포함)는
          "전부 None" 또는 "전부 값 있음"만 허용합니다.

        Args:
            batch: 길이 B의 샘플 리스트.

        Returns:
            WorldModelFeature:
                배치 차원(B)이 앞에 붙은 형태의 WorldModelFeature.

        Raises:
            ValueError:
                - 필수 필드에 None이 섞인 경우
                - 선택 필드가 일부만 None인 경우
        """
        collated_dict: Dict[str, Any] = {}

        for field in dataclass_fields(cls):
            name = field.name
            values = [getattr(item, name) for item in batch]
            allow_none = name not in _REQUIRED_FIELD_NAMES
            collated_dict[name] = _collate_field_values(values,
                                                        field_name=name,
                                                        allow_none=allow_none)

        _validate_required_fields_not_none(collated_dict, _REQUIRED_FIELD_NAMES)
        return cls(**collated_dict)

    def unpack(self) -> List[WorldModelFeature]:
        """배치 형태의 WorldModelFeature를 샘플 리스트로 다시 풀어냅니다.

        Returns:
            List[WorldModelFeature]:
                길이 B의 리스트. 각 원소는 "한 샘플"에 해당.

        Raises:
            ValueError: 필수 필드가 None인 상태에서 unpack을 시도하면 발생.
        """
        batch_size = int(to_tensor(
            self.ego_agent_past).shape[0])  # (B, time_len, 11)

        features: List[WorldModelFeature] = []
        for i in range(batch_size):
            item_dict: Dict[str, Any] = {}
            for field in dataclass_fields(type(self)):
                name = field.name
                value = getattr(self, name)
                item_dict[name] = _slice_batch_value_or_none(value, i)

            _validate_required_fields_not_none(item_dict, _REQUIRED_FIELD_NAMES)
            features.append(type(self)(**item_dict))

        return features

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> WorldModelFeature:
        return WorldModelFeature(**data)


    def to_tensor_dict(
        self,
        *,
        make_contiguous: bool = False,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """WorldModelFeature를 Dict[str, Optional[torch.Tensor]]로 변환합니다.

        각 필드에 대해:
        - 값이 None이면 결과 딕셔너리에서도 None 유지
        - None이 아니면 `to_tensor`로 torch.Tensor로 변환 (이미 텐서여도 그대로 처리)
        - 선택적으로 device로 이동 및 contiguous 메모리 보장

        Args:
            make_contiguous: True면 각 텐서에 .contiguous() 적용.
            device: 지정 시 각 텐서를 해당 device로 이동.

        Returns:
            Dict[str, Optional[torch.Tensor]]: 필드명 → 텐서(or None) 매핑 딕셔너리.

        Note:
            대표적인 텐서 shape 예시(배치 차원 포함):
            - "ego_agent_past": (B, time_len, 11)
            - "neighbor_agents_past": (B, max_agent_num, time_len, 11)
            - "static_objects": (B, static_objects_num, 10)
            - "lanes": (B, lane_num, lane_len, 12)
            - "lanes_speed_limit": (B, lane_num, 1)
            - "lanes_has_speed_limit": (B, lane_num, 1)
            - "route_lanes": (B, route_num, lane_len, 12) or None
            - "route_lanes_speed_limit": (B, route_num, 1) or None
            - "route_lanes_has_speed_limit": (B, route_num, 1) or None
            - "agent_route_lane_order": (B, max_agent_num, lane_num) or None
            - "target_agents_mask": (B, max_agent_num) or None
            - "near_route_lanes": (B, Pnn, lane_len, 12) or None
            - "near_route_lanes_speed_limit": (B, Pnn, 1) or None
            - "near_route_lanes_has_speed_limit": (B, Pnn, 1) or None
            - "ego_agent_next_11_dim": (B, interpol_num, 11) or None
            - "planner_future_11_dim": (B, future_len, 11) or None
        """

        # 지역 import로 의존성 최소화 (클래스 외부 수정 없이 동작)

        def _to_tensor_or_none(value: Any) -> Optional[torch.Tensor]:
            if value is None:
                return None
            tensor = value if isinstance(value,
                                         torch.Tensor) else to_tensor(value)
            if device is not None:
                tensor = tensor.to(device=device)
            if make_contiguous:
                tensor = tensor.contiguous()
            return tensor

        tensor_dict: Dict[str, Optional[torch.Tensor]] = {}
        # type(self)를 사용해 상속/확장에도 안전하게 모든 데이터클래스 필드 순회
        for field in dataclass_fields(type(self)):
            field_name = field.name
            field_value = getattr(self, field_name)
            tensor_dict[field_name] = _to_tensor_or_none(field_value)

        return tensor_dict