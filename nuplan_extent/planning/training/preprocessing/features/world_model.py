from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from torch.utils.data.dataloader import default_collate

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)
from dataclasses import fields as dataclass_fields

import torch


# nuplan/planning/script/builders/simulation_builder.py
@dataclass
class WorldModelFeature(AbstractModelFeature):
    ########### SAME AS LEARNING INPUT ###########
    ego_agent_past: FeatureDataType  # (time_len, 11)
    neighbor_agents_past: FeatureDataType  # (agent_num, time_len, 11)
    static_objects: FeatureDataType  # (static_objects_num, 10)
    ################################################
    ########### SAME AS LEARNING INPUT ###########
    lanes: FeatureDataType  # (lane_num, lane_len, 12)
    lanes_speed_limit: FeatureDataType  # (lane_num, 1)
    lanes_has_speed_limit: FeatureDataType  # (lane_num, 1)
    route_lanes: Optional[FeatureDataType]  # (route_num, lane_len, 12)
    route_lanes_speed_limit: Optional[FeatureDataType]  # (route_num, 1)
    route_lanes_has_speed_limit: Optional[FeatureDataType]  # (route_num, 1)
    agent_route_lane_order: Optional[
        FeatureDataType]  # (agent_num, lane_num) # -1 if not on route
    ################################################
    ################ inference only ################
    target_agents_mask: Optional[FeatureDataType]  # (agent_num,) bool
    ego_agent_next_11_dim: Optional[
        FeatureDataType] = None  # (interpol_num, 11)
    planner_future_11_dim: Optional[FeatureDataType] = None  # (future_len, 11)
    ################################################

    def to_feature_tensor(self) -> WorldModelFeature:
        """Convert numpy arrays to torch tensors."""
        return WorldModelFeature(
            ego_agent_past=to_tensor(self.ego_agent_past).contiguous(),
            neighbor_agents_past=to_tensor(
                self.neighbor_agents_past).contiguous(),
            static_objects=to_tensor(self.static_objects).contiguous(),
            lanes=to_tensor(self.lanes).contiguous(),
            lanes_speed_limit=to_tensor(self.lanes_speed_limit).contiguous(),
            lanes_has_speed_limit=to_tensor(
                self.lanes_has_speed_limit).contiguous(),
            route_lanes=None if self.route_lanes is None else to_tensor(
                self.route_lanes).contiguous(),
            route_lanes_speed_limit=None if self.route_lanes_speed_limit is None
            else to_tensor(self.route_lanes_speed_limit).contiguous(),
            route_lanes_has_speed_limit=None
            if self.route_lanes_has_speed_limit is None else to_tensor(
                self.route_lanes_has_speed_limit).contiguous(),
            agent_route_lane_order=None if self.agent_route_lane_order is None
            else to_tensor(self.agent_route_lane_order).contiguous(),
            target_agents_mask=None if self.target_agents_mask is None else
            to_tensor(self.target_agents_mask).contiguous(),
            ego_agent_next_11_dim=None if self.ego_agent_next_11_dim is None
            else to_tensor(self.ego_agent_next_11_dim).contiguous(),
            planner_future_11_dim=None if self.planner_future_11_dim is None else
            to_tensor(self.planner_future_11_dim).contiguous(),
        )

    def to_device(self, device: torch.device) -> WorldModelFeature:
        """Move feature tensors to a specific device."""
        return WorldModelFeature(
            ego_agent_past=to_tensor(self.ego_agent_past).to(device=device),
            neighbor_agents_past=to_tensor(
                self.neighbor_agents_past).to(device=device),
            static_objects=to_tensor(self.static_objects).to(device=device),
            lanes=to_tensor(self.lanes).to(device=device),
            lanes_speed_limit=to_tensor(
                self.lanes_speed_limit).to(device=device),
            lanes_has_speed_limit=to_tensor(
                self.lanes_has_speed_limit).to(device=device),
            route_lanes=None if self.route_lanes is None else to_tensor(
                self.route_lanes).to(device=device),
            route_lanes_speed_limit=None if self.route_lanes_speed_limit is None
            else to_tensor(self.route_lanes_speed_limit).to(device=device),
            route_lanes_has_speed_limit=None if self.route_lanes_has_speed_limit
            is None else to_tensor(self.route_lanes_has_speed_limit).to(
                device=device),
            agent_route_lane_order=None if self.agent_route_lane_order is None
            else to_tensor(self.agent_route_lane_order).to(device=device),
            target_agents_mask=None if self.target_agents_mask is None else
            to_tensor(self.target_agents_mask).to(device=device),
            ego_agent_next_11_dim=None if self.ego_agent_next_11_dim is None
            else to_tensor(self.ego_agent_next_11_dim).to(device=device),
            planner_future_11_dim=None if self.planner_future_11_dim is None else
            to_tensor(self.planner_future_11_dim).to(device=device),
        )

    @classmethod
    def collate(cls, batch: List[WorldModelFeature]) -> WorldModelFeature:
        """Batch features together ensuring consistency of optional attributes."""

        def _collate_optional(name: str):
            values = [getattr(item, name) for item in batch]
            if any(v is None for v in values):
                if not all(v is None for v in values):
                    raise ValueError(
                        f"Attribute '{name}' must be either all None or all not None in the batch."
                    )
                return None
            return default_collate(values)

        return WorldModelFeature(
            ego_agent_past=default_collate([b.ego_agent_past for b in batch]),
            neighbor_agents_past=default_collate(
                [b.neighbor_agents_past for b in batch]),
            static_objects=default_collate([b.static_objects for b in batch]),
            lanes=default_collate([b.lanes for b in batch]),
            lanes_speed_limit=default_collate(
                [b.lanes_speed_limit for b in batch]),
            lanes_has_speed_limit=default_collate(
                [b.lanes_has_speed_limit for b in batch]),
            route_lanes=_collate_optional("route_lanes"),
            route_lanes_speed_limit=_collate_optional(
                "route_lanes_speed_limit"),
            route_lanes_has_speed_limit=_collate_optional(
                "route_lanes_has_speed_limit"),
            agent_route_lane_order=_collate_optional("agent_route_lane_order"),
            target_agents_mask=_collate_optional("target_agents_mask"),
            ego_agent_next_11_dim=_collate_optional("ego_agent_next_11_dim"),
            planner_future_11_dim=_collate_optional("planner_future_11_dim"),
        )

    @classmethod
    def deserialize(
        cls, data: Dict[str, Any]
    ) -> WorldModelFeature:  # pragma: no cover - simple wrapper
        return WorldModelFeature(**data)

    def unpack(
            self
    ) -> List[WorldModelFeature]:  # pragma: no cover - simple wrapper
        batch_size = to_tensor(self.ego_agent_past).shape[0]
        features: List[WorldModelFeature] = []
        for i in range(batch_size):
            features.append(
                WorldModelFeature(
                    ego_agent_past=self.ego_agent_past[i],  # DONE
                    neighbor_agents_past=self.neighbor_agents_past[i],  # DONE
                    static_objects=self.static_objects[i],
                    lanes=self.lanes[i],  # DONE
                    lanes_speed_limit=self.lanes_speed_limit[i],
                    lanes_has_speed_limit=self.lanes_has_speed_limit[i],
                    route_lanes=None
                    if self.route_lanes is None else self.route_lanes[i],
                    route_lanes_speed_limit=None if self.route_lanes_speed_limit
                    is None else self.route_lanes_speed_limit[i],
                    route_lanes_has_speed_limit=None
                    if self.route_lanes_has_speed_limit is None else
                    self.route_lanes_has_speed_limit[i],
                    agent_route_lane_order=None if self.agent_route_lane_order
                    is None else self.agent_route_lane_order[i],
                    target_agents_mask=None if self.target_agents_mask is None
                    else self.target_agents_mask[i],
                    ego_agent_next_11_dim=None if self.ego_agent_next_11_dim
                    is None else self.ego_agent_next_11_dim[i],
                    planner_future_11_dim=None if self.planner_future_11_dim
                    is None else self.planner_future_11_dim[i],
                ))
        return features

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
            - "neighbor_agents_past": (B, agent_num, time_len, 11)
            - "static_objects": (B, static_objects_num, 10)
            - "lanes": (B, lane_num, lane_len, 12)
            - "lanes_speed_limit": (B, lane_num, 1)
            - "lanes_has_speed_limit": (B, lane_num, 1)
            - "route_lanes": (B, route_num, lane_len, 12) or None
            - "route_lanes_speed_limit": (B, route_num, 1) or None
            - "route_lanes_has_speed_limit": (B, route_num, 1) or None
            - "agent_route_lane_order": (B, agent_num, lane_num) or None
            - "target_agents_mask": (B, agent_num) or None
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
