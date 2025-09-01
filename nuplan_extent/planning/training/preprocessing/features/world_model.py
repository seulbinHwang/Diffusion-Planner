from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data.dataloader import default_collate

from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    AbstractModelFeature,
    FeatureDataType,
    to_tensor,
)


@dataclass
class WorldModelFeature(AbstractModelFeature):
    ego_agent_past: FeatureDataType  # (time_len, 11)
    neighbor_agents_past: FeatureDataType  # (agent_num, time_len, 11)
    static_objects: FeatureDataType  # (static_objects_num, 10)
    ###########
    lanes: FeatureDataType  # (lane_num, lane_len, 12)
    lanes_speed_limit: FeatureDataType  # (lane_num, 1)
    lanes_has_speed_limit: FeatureDataType  # (lane_num, 1)
    route_lanes: Optional[FeatureDataType]  # (route_num, lane_len, 12)
    route_lanes_speed_limit: Optional[FeatureDataType]  # (route_num, 1)
    route_lanes_has_speed_limit: Optional[FeatureDataType]  # (route_num, 1)
    near_route_lanes: Optional[FeatureDataType]  # (Pnn, lane_len, 12)
    near_route_lanes_speed_limit: Optional[FeatureDataType]  # (Pnn, 1)
    near_route_lanes_has_speed_limit: Optional[FeatureDataType]  # (Pnn, 1)
    ###########
    next_ego_state: Optional[FeatureDataType] = None  # (11,)
    ego_agent_future_11_dim: Optional[
        FeatureDataType] = None  # (future_time_len, 11)

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
            near_route_lanes=None if self.near_route_lanes is None else
            to_tensor(self.near_route_lanes).contiguous(),
            near_route_lanes_speed_limit=None
            if self.near_route_lanes_speed_limit is None else to_tensor(
                self.near_route_lanes_speed_limit).contiguous(),
            near_route_lanes_has_speed_limit=None
            if self.near_route_lanes_has_speed_limit is None else to_tensor(
                self.near_route_lanes_has_speed_limit).contiguous(),
            next_ego_state=None if self.next_ego_state is None else to_tensor(
                self.next_ego_state).contiguous(),
            ego_agent_future_11_dim=None if self.ego_agent_future_11_dim is None
            else to_tensor(self.ego_agent_future_11_dim).contiguous(),
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
            near_route_lanes=None if self.near_route_lanes is None else
            to_tensor(self.near_route_lanes).to(device=device),
            near_route_lanes_speed_limit=None
            if self.near_route_lanes_speed_limit is None else to_tensor(
                self.near_route_lanes_speed_limit).to(device=device),
            near_route_lanes_has_speed_limit=None
            if self.near_route_lanes_has_speed_limit is None else to_tensor(
                self.near_route_lanes_has_speed_limit).to(device=device),
            next_ego_state=None if self.next_ego_state is None else to_tensor(
                self.next_ego_state).to(device=device),
            ego_agent_future_11_dim=None if self.ego_agent_future_11_dim is None
            else to_tensor(self.ego_agent_future_11_dim).to(device=device),
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
            near_route_lanes=_collate_optional("near_route_lanes"),
            near_route_lanes_speed_limit=_collate_optional(
                "near_route_lanes_speed_limit"),
            near_route_lanes_has_speed_limit=_collate_optional(
                "near_route_lanes_has_speed_limit"),
            next_ego_state=_collate_optional("next_ego_state"),
            ego_agent_future_11_dim=_collate_optional(
                "ego_agent_future_11_dim"),
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
                    ego_agent_past=self.ego_agent_past[i],
                    neighbor_agents_past=self.neighbor_agents_past[i],
                    static_objects=self.static_objects[i],
                    lanes=self.lanes[i],
                    lanes_speed_limit=self.lanes_speed_limit[i],
                    lanes_has_speed_limit=self.lanes_has_speed_limit[i],
                    route_lanes=None
                    if self.route_lanes is None else self.route_lanes[i],
                    route_lanes_speed_limit=None if self.route_lanes_speed_limit
                    is None else self.route_lanes_speed_limit[i],
                    route_lanes_has_speed_limit=None
                    if self.route_lanes_has_speed_limit is None else
                    self.route_lanes_has_speed_limit[i],
                    near_route_lanes=None if self.near_route_lanes is None else
                    self.near_route_lanes[i],
                    near_route_lanes_speed_limit=None
                    if self.near_route_lanes_speed_limit is None else
                    self.near_route_lanes_speed_limit[i],
                    near_route_lanes_has_speed_limit=None
                    if self.near_route_lanes_has_speed_limit is None else
                    self.near_route_lanes_has_speed_limit[i],
                    next_ego_state=None
                    if self.next_ego_state is None else self.next_ego_state[i],
                    ego_agent_future_11_dim=None if self.ego_agent_future_11_dim
                    is None else self.ego_agent_future_11_dim[i],
                ))
        return features
