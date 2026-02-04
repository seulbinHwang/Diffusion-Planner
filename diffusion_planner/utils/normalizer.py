from copy import copy, deepcopy
import torch

from diffusion_planner.utils.train_utils import openjson
from pathlib import Path
try:
    from hydra.utils import to_absolute_path
except Exception:

    def to_absolute_path(path: str) -> str:
        return str(Path(path).expanduser().resolve())


import torch
from typing import Any


class StateNormalizer:

    def __init__(self, mean: object, std: object) -> None:
        self.mean = self._to_1d4(mean, name="mean")  # (4,)
        self.std = self._to_1d4(std, name="std")  # (4,)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        mean = data["neighbor"]["mean"]
        std = data["neighbor"]["std"]
        return cls(mean, std)

    @classmethod
    def from_json2(cls, args_dict):
        path_str = args_dict.get("normalization_file_path",
                                 "normalization.json")
        data = openjson(to_absolute_path(path_str))
        mean = data["neighbor"]["mean"]
        std = data["neighbor"]["std"]
        return cls(mean, std)

    @staticmethod
    def _to_1d4(values: object, name: str) -> torch.Tensor:
        values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if values_t.numel() != 4:
            raise ValueError(
                f"{name}는 총 4개 값이어야 합니다. "
                f"(받은 원소 개수={values_t.numel()}, 받은 shape={tuple(torch.as_tensor(values).shape)})"
            )
        return values_t

    @staticmethod
    def _reshape_stats_for_data(stats_1d4: torch.Tensor,
                                data: torch.Tensor) -> torch.Tensor:
        """stats(4,)를 data(...,4)에 맞게 reshape + dtype/device를 data와 맞춥니다.

        Args:
            stats_1d4: (4,)
            data: (..., 4)

        Returns:
            (..., 4)에 broadcast 가능한 shape의 stats 텐서.
            dtype/device는 data와 동일.
        """
        leading_ones = [1] * (data.ndim - 1)
        return stats_1d4.to(device=data.device,
                            dtype=data.dtype).view(*leading_ones, 4)

    @staticmethod
    def _broadcast_valid_mask(valid_mask: Any,
                              data: torch.Tensor) -> torch.Tensor:
        """valid_mask를 data와 같은 shape로 브로드캐스트 가능한 형태로 정리합니다.

        Args:
            valid_mask: 보통 data.shape[:-1] 모양의 bool 마스크.
            data: (..., 4) 텐서.

        Returns:
            torch.Tensor: data와 같은 shape로 broadcast 가능한 bool 마스크.
                - 일반적으로 (...., 1)로 확장됩니다.

        Raises:
            ValueError: valid_mask 모양이 data와 맞지 않는 경우.
        """
        mask = torch.as_tensor(valid_mask, device=data.device).to(torch.bool)

        if mask.shape == data.shape:
            return mask

        # 보통 valid_mask는 data의 마지막 채널(4)을 뺀 모양임: (...,)
        if mask.shape == data.shape[:-1]:
            return mask.unsqueeze(-1)

        raise ValueError(
            "valid_mask 모양이 data와 맞지 않습니다. "
            f"valid_mask.shape={tuple(mask.shape)}, data.shape={tuple(data.shape)}"
        )

    @staticmethod
    def _mask_out_of_place(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """in-place 없이 마스크 False 위치를 0으로 만듭니다.

        Args:
            x: data와 동일 shape 텐서.
            mask: x와 동일 shape (또는 broadcast 가능한) bool 마스크.

        Returns:
            torch.Tensor: mask가 False인 위치가 0인 새 텐서.
        """
        return torch.where(mask, x, torch.zeros_like(x))

    def __call__(self, data: torch.Tensor, valid_mask) -> torch.Tensor:
        """data를 정규화합니다(마스크는 out-of-place로 적용).

        핵심 변경:
            - mean/std를 data와 같은 dtype으로 맞춘 뒤 계산합니다.
              (입력이 bf16면 bf16로 계산/출력)

        Args:
            data: (..., 4)
            valid_mask: 보통 data.shape[:-1] 모양의 bool 마스크

        Returns:
            (..., 4) 정규화 결과. invalid 위치는 0.
        """
        if data.shape[-1] != 4:
            raise ValueError(
                f"data의 마지막 차원은 4여야 합니다. (받은 shape={tuple(data.shape)})")

        # autocast는 끄고, dtype은 '입력 data dtype'을 그대로 존중합니다.
        with torch.amp.autocast(data.device.type, enabled=False):
            mean = self._reshape_stats_for_data(self.mean, data)
            std = self._reshape_stats_for_data(self.std, data)
            norm_data = (data - mean) / std
            mask = self._broadcast_valid_mask(valid_mask, data)
            return self._mask_out_of_place(norm_data, mask)

    def inverse(self, data: torch.Tensor, valid_mask) -> torch.Tensor:
        """정규화된 data를 역변환합니다(마스크는 out-of-place로 적용).

        핵심 변경:
            - mean/std를 data와 같은 dtype으로 맞춘 뒤 계산합니다.
              (입력이 bf16면 bf16로 계산/출력)

        Args:
            data: (..., 4)
            valid_mask: 보통 data.shape[:-1] 모양의 bool 마스크

        Returns:
            (..., 4) 역변환 결과. invalid 위치는 0.
        """
        if data.shape[-1] != 4:
            raise ValueError(
                f"data의 마지막 차원은 4여야 합니다. (받은 shape={tuple(data.shape)})")

        with torch.amp.autocast(data.device.type, enabled=False):
            mean = self._reshape_stats_for_data(self.mean, data)
            std = self._reshape_stats_for_data(self.std, data)
            inv_data = data * std + mean
            mask = self._broadcast_valid_mask(valid_mask, data)
            return self._mask_out_of_place(inv_data, mask)

    def to_dict(self) -> dict:
        """현재 mean/std를 저장용 dict로 바꿉니다.

        Returns:
            dict: {"mean": ..., "std": ...}
                mean/std는 (1, 1, 4) 모양의 중첩 리스트로 내보냅니다.
        """
        mean_1x1x4 = self.mean.view(1, 1, 4).detach().cpu().numpy().tolist()
        std_1x1x4 = self.std.view(1, 1, 4).detach().cpu().numpy().tolist()
        return {"mean": mean_1x1x4, "std": std_1x1x4}


from copy import copy
from typing import Dict, Any, Optional
import torch


class ObservationNormalizer:

    def __init__(self, normalization_dict: Dict[str, Dict[str, torch.Tensor]]):
        self._normalization_dict = {k: v for k, v in normalization_dict.items()}

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32)
                }
        return cls(ndt)

    @classmethod
    def from_json2(cls, args_dict):
        path_str = args_dict.get("normalization_file_path",
                                 "normalization.json")
        data = openjson(to_absolute_path(path_str))

        ndt = {}
        for k, v in data.items():
            if k in ["ego", "neighbor"]:
                continue
            ndt[k] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float32),
                "std": torch.tensor(v["std"], dtype=torch.float32),
            }
        return cls(ndt)

    @staticmethod
    def _infer_device_type_from_dict(data: Dict[str, Any]) -> str:
        """dict 안의 텐서 중 하나를 찾아 device type을 고릅니다."""
        for v in data.values():
            if torch.is_tensor(v):
                return v.device.type
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _apply_valid_mask_out_of_place(x: torch.Tensor,
                                       valid: torch.Tensor) -> torch.Tensor:
        """x에서 valid가 False인 위치를 0으로 만든 새 텐서를 반환합니다.

        허용하는 valid 모양
        - valid.shape == x.shape
        - valid.shape == x.shape[:-1] (마지막 채널만 없는 형태)
        - valid가 x의 "앞쪽 축"까지만 있는 형태
          예) x=(B,N,S,2) 이고 valid=(B,N) 인 경우 → valid를 (B,N,1,1)로 늘려서 적용

        Args:
            x (torch.Tensor): 임의 shape 텐서.
            valid (torch.Tensor): 유효 마스크 텐서.

        Returns:
            torch.Tensor: x와 같은 shape 텐서. invalid 위치는 0.

        Raises:
            ValueError: valid 마스크 shape이 x와 맞지 않는 경우.
        """
        v = valid.to(dtype=torch.bool, device=x.device)

        if v.ndim > x.ndim:
            raise ValueError(
                "valid 마스크 차원 수가 데이터보다 큽니다. "
                f"valid.ndim={int(v.ndim)}, x.ndim={int(x.ndim)}, "
                f"valid.shape={tuple(v.shape)}, x.shape={tuple(x.shape)}")

        # 1) 완전 동일 모양
        if v.shape == x.shape:
            mask = v
            return torch.where(mask, x, torch.zeros_like(x))

        # 2) 마지막 채널만 없는 모양: (.. ) -> (..,1)
        if v.shape == x.shape[:-1]:
            mask = v.unsqueeze(-1)
            return torch.where(mask, x, torch.zeros_like(x))

        # 3) "앞쪽 축"까지만 있는 모양: 뒤쪽 축을 1로 늘려서 맞춤
        prefix_ok = True
        for i in range(int(v.ndim)):
            v_dim = int(v.shape[i])
            x_dim = int(x.shape[i])
            if v_dim != 1 and v_dim != x_dim:
                prefix_ok = False
                break

        if prefix_ok:
            if v.ndim < x.ndim:
                expand_shape = tuple(v.shape) + (1,) * int(x.ndim - v.ndim)
                mask = v.reshape(expand_shape)
            else:
                mask = v

            return torch.where(mask, x, torch.zeros_like(x))

        raise ValueError(
            "valid 마스크 shape이 데이터와 맞지 않습니다. "
            "허용: valid.shape == x.shape, valid.shape == x.shape[:-1], "
            "또는 valid가 x의 앞쪽 축과 맞고(중간에 1은 허용) 뒤쪽 축은 자동 확장 가능한 경우입니다. "
            f"valid.shape={tuple(v.shape)}, x.shape={tuple(x.shape)}")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """정규화(mean/std 적용) + invalid 마스킹을 수행합니다.

        핵심 변경:
            - mean/std를 각 입력 텐서(x)와 같은 dtype으로 맞춘 뒤 계산합니다.
              (입력이 bf16면 bf16로 계산/출력)
        """
        device_type = self._infer_device_type_from_dict(data)
        with torch.amp.autocast(device_type, enabled=False):
            norm_data = copy(data)

            # 1) 정의된 키만 정규화
            for k, v in self._normalization_dict.items():
                if (k not in data) or (v is None) or (data[k] is None):
                    continue
                x = data[k]
                if not torch.is_tensor(x):
                    continue
                if not torch.is_floating_point(x):
                    continue

                mean = v["mean"].to(device=x.device, dtype=x.dtype)
                std = v["std"].to(device=x.device, dtype=x.dtype)
                norm_data[k] = (x - mean) / std

            # 2) 마스킹은 딱 1번만(out-of-place)
            self._mask_invalid_data(norm_data)

            # 3) 패스스루 키
            if "agent_route_lane_order" in data:
                norm_data["agent_route_lane_order"] = data[
                    "agent_route_lane_order"].to(torch.long)

            return norm_data

    def inverse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """역정규화(std/mean 되돌림) + invalid 마스킹을 수행합니다.

        핵심 변경:
            - mean/std를 각 입력 텐서(x)와 같은 dtype으로 맞춘 뒤 계산합니다.
              (입력이 bf16면 bf16로 계산/출력)
        """
        device_type = self._infer_device_type_from_dict(data)
        with torch.amp.autocast(device_type, enabled=False):
            norm_data = copy(data)

            for k, v in self._normalization_dict.items():
                if (k not in data) or (v is None) or (data[k] is None):
                    continue
                x = data[k]
                if not torch.is_tensor(x):
                    continue
                if not torch.is_floating_point(x):
                    continue

                mean = v["mean"].to(device=x.device, dtype=x.dtype)
                std = v["std"].to(device=x.device, dtype=x.dtype)
                norm_data[k] = x * std + mean

            self._mask_invalid_data(norm_data)

            if "agent_route_lane_order" in data:
                norm_data["agent_route_lane_order"] = data[
                    "agent_route_lane_order"].to(torch.long)

            return norm_data

    def _mask_invalid_data(self, norm_data: Dict[str, Any]) -> None:
        """'*_is_valid' 마스크가 False인 위치를 0으로 만든 텐서를 다시 dict에 넣습니다.

        주의:
            - in-place로 원본 텐서를 직접 바꾸지 않습니다.
            - dict 안에 같은 텐서 참조가 있어도 안전합니다.
        """

        def _mask(data_key: str, valid_key: str) -> None:
            valid = norm_data.get(valid_key, None)
            x = norm_data.get(data_key, None)
            if valid is None or x is None:
                return
            if not torch.is_tensor(valid) or not torch.is_tensor(x):
                return
            norm_data[data_key] = self._apply_valid_mask_out_of_place(x, valid)

        _mask("ego_agent_past", "ego_agent_past_is_valid")
        _mask("planner_future_11_dim", "ego_future_gt_is_valid")
        _mask("neighbor_agents_past", "neighbor_agents_past_is_valid")

        _mask("stop_sign_points", "stop_sign_is_valid")
        _mask("crosswalk_points", "crosswalk_is_valid")

        _mask("lanes", "lanes_len_is_valid")
        _mask("lanes_speed_limit", "lanes_is_valid")

        _mask("static_objects", "static_objects_is_valid")

        _mask("route_lanes", "route_lanes_len_is_valid")
        _mask("route_lanes_speed_limit", "route_lanes_is_valid")

        _mask("speed_bump_points", "speed_bump_is_valid")
        _mask("driveway_points", "driveway_is_valid")
        _mask("road_edge", "road_edge_is_valid")

        _mask("near_agents_past", "near_agents_past_is_valid")
        _mask("non_near_agents_past", "non_near_agents_past_is_valid")

    def to_dict(self):
        return {
            k: {
                kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()
            } for k, v in self._normalization_dict.items()
        }