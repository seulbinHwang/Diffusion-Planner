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
from typing import Any, Optional

def _pick_normalization_file_path(args_dict: dict[str, Any]) -> str:
    """args_dict를 보고 사용할 normalization json 경로를 결정합니다.

    Args:
        args_dict (dict[str, Any]): 설정 dict.
            - use_body_vel: body 좌표 속도 사용 여부(True/False 또는 "true"/"false" 등 문자열 가능)
            - normalization_file_path: 사용자가 직접 지정한 json 경로(선택)

    Returns:
        str: 선택된 normalization json 경로 문자열.

    Notes:
        - use_body_vel == True  -> "normalization_use_body_vel_new.json"
        - use_body_vel == False -> "normalization_new.json"
        - normalization_file_path 가 위 두 기본 파일명 중 하나를 가리키면(use_body_vel 규칙을 적용).
        - normalization_file_path가 "다른 파일"이면 그 값을 그대로 사용합니다.
    """
    raw_flag = args_dict.get("use_body_vel", False)

    # 문자열/숫자도 안전하게 bool로 해석
    if isinstance(raw_flag, bool):
        use_body_vel = raw_flag
    elif isinstance(raw_flag, (int, float)):
        use_body_vel = (raw_flag != 0)
    elif isinstance(raw_flag, str):
        s = raw_flag.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            use_body_vel = True
        elif s in ("0", "false", "f", "no", "n", "off"):
            use_body_vel = False
        else:
            use_body_vel = False
    else:
        use_body_vel = bool(raw_flag)

    desired = "normalization_use_body_vel_new.json" if use_body_vel else "normalization_new.json"

    given = args_dict.get("normalization_file_path", None)
    if given is None:
        return desired

    given_str = str(given).strip()
    if given_str == "":
        return desired

    # normalization_file_path가 "기본 두 파일" 중 하나면 use_body_vel 규칙을 우선
    try:
        given_name = Path(given_str).name
    except Exception:
        given_name = given_str

    if given_name in ("normalization_new.json", "normalization_use_body_vel_new.json"):
        return desired

    # 그 외엔 사용자가 지정한 경로를 존중
    return given_str

class StateNormalizer:
    """입력 텐서의 마지막 차원에 따라 정규화/역정규화를 수행하는 클래스입니다.

    - 마지막 차원 4: normalization.json의 "neighbor" 통계(mean/std) 사용
    - 마지막 차원 3: normalization.json의 "seg_body_control" 통계(mean/std) 사용
    - valid_mask가 False인 위치는 결과를 0으로 만듭니다(out-of-place).
    """

    def __init__(
        self,
        mean: object,
        std: object,
        seg_mean: Optional[object] = None,
        seg_std: Optional[object] = None,
    ) -> None:
        """정규화 통계를 저장합니다.

        Args:
            mean: "neighbor" 평균. 길이 4여야 합니다.
            std: "neighbor" 표준편차. 길이 4여야 합니다.
            seg_mean: "seg_body_control" 평균. 길이 3이어야 합니다(선택).
            seg_std: "seg_body_control" 표준편차. 길이 3이어야 합니다(선택).

        Notes:
            - seg_mean/seg_std를 주지 않으면, 마지막 차원이 3인 입력을 처리할 때 에러를 냅니다.
        """
        self.mean_4 = self._to_1d(values=mean, numel=4, name="neighbor.mean")  # (4,)
        self.std_4 = self._to_1d(values=std, numel=4, name="neighbor.std")    # (4,)

        self.mean_3 = None if seg_mean is None else self._to_1d(values=seg_mean, numel=3, name="seg_body_control.mean")  # (3,)
        self.std_3 = None if seg_std is None else self._to_1d(values=seg_std, numel=3, name="seg_body_control.std")      # (3,)

    @classmethod
    def from_json(cls, args) -> "StateNormalizer":
        data = openjson(args.normalization_file_path)
        mean4 = data["neighbor"]["mean"]
        std4 = data["neighbor"]["std"]
        mean3 = data["seg_body_control"]["mean"]
        std3 = data["seg_body_control"]["std"]
        return cls(mean=mean4, std=std4, seg_mean=mean3, seg_std=std3)

    @classmethod
    def from_json2(cls, args_dict) -> "StateNormalizer":
        path_str = _pick_normalization_file_path(args_dict)
        print("path_str: ", path_str, "  ", type(path_str))
        data = openjson(to_absolute_path(path_str))
        mean4 = data["neighbor"]["mean"]
        std4 = data["neighbor"]["std"]
        mean3 = data["seg_body_control"]["mean"]
        std3 = data["seg_body_control"]["std"]
        print("mean3: ", mean3, "std3: ", std3)
        return cls(mean=mean4, std=std4, seg_mean=mean3, seg_std=std3)

    @staticmethod
    def _to_1d(values: object, numel: int, name: str) -> torch.Tensor:
        """입력을 1차원 텐서로 만들고, 원소 개수가 기대값과 같은지 검사합니다.

        Args:
            values: 리스트/튜플/텐서 등 숫자 값.
            numel: 기대하는 원소 개수.
            name: 에러 메시지에 표시할 이름.

        Returns:
            torch.Tensor: shape (numel,)의 float32 텐서.

        Raises:
            ValueError: 원소 개수가 numel이 아니면 발생합니다.
        """
        values_t = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if int(values_t.numel()) != int(numel):
            raise ValueError(
                f"{name}는 총 {numel}개 값이어야 합니다. "
                f"(받은 원소 개수={values_t.numel()}, 받은 shape={tuple(torch.as_tensor(values).shape)})"
            )
        return values_t

    @staticmethod
    def _reshape_stats_for_data(stats_1d: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """stats(C,)를 data(...,C)에 맞게 reshape + dtype/device를 data와 맞춥니다."""
        c = int(data.shape[-1])
        if int(stats_1d.numel()) != c:
            raise ValueError(
                "stats의 길이와 data 마지막 차원이 맞지 않습니다. "
                f"stats.numel()={int(stats_1d.numel())}, data.shape={tuple(data.shape)}"
            )
        leading_ones = [1] * (data.ndim - 1)
        return stats_1d.to(device=data.device, dtype=data.dtype).view(*leading_ones, c)

    @staticmethod
    def _broadcast_valid_mask(valid_mask: Any, data: torch.Tensor) -> torch.Tensor:
        mask = torch.as_tensor(valid_mask, device=data.device).to(torch.bool)

        if mask.shape == data.shape:
            return mask

        if mask.shape == data.shape[:-1]:
            return mask.unsqueeze(-1)

        raise ValueError(
            "valid_mask 모양이 data와 맞지 않습니다. "
            f"valid_mask.shape={tuple(mask.shape)}, data.shape={tuple(data.shape)}"
        )

    @staticmethod
    def _mask_out_of_place(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, x, torch.zeros_like(x))

    def _select_stats(self, last_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        """data의 마지막 차원 크기에 맞는 mean/std를 고릅니다."""
        if last_dim == 4:
            return self.mean_4, self.std_4
        if last_dim == 3:
            if self.mean_3 is None or self.std_3 is None:
                raise ValueError("마지막 차원이 3인 입력을 처리하려면 seg_mean/seg_std가 필요합니다.")
            return self.mean_3, self.std_3
        raise ValueError(f"지원하지 않는 마지막 차원입니다. last_dim={last_dim} (허용: 3 또는 4)")

    def __call__(self, data: torch.Tensor, valid_mask) -> torch.Tensor:
        last_dim = int(data.shape[-1])
        mean_1d, std_1d = self._select_stats(last_dim)

        with torch.amp.autocast(data.device.type, enabled=False):
            mean = self._reshape_stats_for_data(mean_1d, data)
            std = self._reshape_stats_for_data(std_1d, data)
            norm_data = (data - mean) / std
            mask = self._broadcast_valid_mask(valid_mask, data)
            return self._mask_out_of_place(norm_data, mask)

    def inverse(self, data: torch.Tensor, valid_mask) -> torch.Tensor:
        last_dim = int(data.shape[-1])
        mean_1d, std_1d = self._select_stats(last_dim)

        with torch.amp.autocast(data.device.type, enabled=False):
            mean = self._reshape_stats_for_data(mean_1d, data)
            std = self._reshape_stats_for_data(std_1d, data)
            inv_data = data * std + mean
            mask = self._broadcast_valid_mask(valid_mask, data)
            return self._mask_out_of_place(inv_data, mask)

    def to_dict(self) -> dict:
        """기존 동작 호환: neighbor(4차원) mean/std만 저장 포맷으로 내보냅니다."""
        mean_1x1x4 = self.mean_4.view(1, 1, 4).detach().cpu().numpy().tolist()
        std_1x1x4 = self.std_4.view(1, 1, 4).detach().cpu().numpy().tolist()
        return {"mean": mean_1x1x4, "std": std_1x1x4}

    def to_dict_extended(self) -> dict:
        """확장 저장: neighbor(4) + seg_body_control(3) 둘 다 내보냅니다."""
        out = {
            "neighbor": {
                "mean": self.mean_4.detach().cpu().numpy().tolist(),
                "std": self.std_4.detach().cpu().numpy().tolist(),
            }
        }
        if self.mean_3 is not None and self.std_3 is not None:
            out["seg_body_control"] = {
                "mean": self.mean_3.detach().cpu().numpy().tolist(),
                "std": self.std_3.detach().cpu().numpy().tolist(),
            }
        return out


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
            if k not in ["ego", "neighbor", "future_seg_control_gt_3_dim"]:
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32)
                }
        return cls(ndt)

    @classmethod
    def from_json2(cls, args_dict):
        path_str = _pick_normalization_file_path(args_dict)
        print("path_str: ", path_str, "  ", type(path_str))
        data = openjson(to_absolute_path(path_str))

        ndt = {}
        for k, v in data.items():
            if k in ["ego", "neighbor", "future_seg_control_gt_3_dim"]:
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
            aro = data.get("agent_route_lane_order", None)
            if torch.is_tensor(aro):
                norm_data["agent_route_lane_order"] = aro.to(torch.long)

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

            aro = data.get("agent_route_lane_order", None)
            if torch.is_tensor(aro):
                norm_data["agent_route_lane_order"] = aro.to(torch.long)

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
        _mask("past_seg_control_gt_3_dim", "past_seg_control_is_valid")

    def to_dict(self):
        return {
            k: {
                kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()
            } for k, v in self._normalization_dict.items()
        }