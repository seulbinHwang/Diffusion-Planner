from copy import copy, deepcopy
import torch

from diffusion_planner.utils.train_utils import openjson
from pathlib import Path
try:
    from hydra.utils import to_absolute_path
except Exception:

    def to_absolute_path(path: str) -> str:
        return str(Path(path).expanduser().resolve())


class StateNormalizer:

    def __init__(self, mean, std):
        # mean, std 는 결국 (1, 1, 4) 로 맞춰서 저장
        mean_t = torch.as_tensor(mean).float()
        std_t = torch.as_tensor(std).float()

        # 1D [4] 나 2D [1,4] 등이 들어와도 무조건 (1,1,4)로 reshape
        if mean_t.ndim == 1:  # (4,)
            mean_t = mean_t.view(1, 1, -1)
            std_t = std_t.view(1, 1, -1)
        elif mean_t.ndim == 2:  # (1,4) 같은 경우
            mean_t = mean_t.view(1, mean_t.size(0), mean_t.size(1))
            std_t = std_t.view(1, std_t.size(0), std_t.size(1))
        # (1,1,4) 로 이미 들어온 경우는 그대로 사용

        self.mean = mean_t  # (1,1,4)
        self.std = std_t  # (1,1,4)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)

        # ✅ neighbor 한 개에 대한 통계만 사용 (shape: (4,))
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

    def __call__(self, data):
        # data: (256, 10, 80, 4)
        # mean, std: (10, 1, 4)
        with torch.amp.autocast(data.device.type, enabled=False):
            # mask: (256, 10, 80)
            mask = torch.sum(torch.ne(data, 0), dim=-1) == 0
            norm_data = (data - self.mean.to(data.device)) / self.std.to(
                data.device)
            norm_data[mask] = 0
            return norm_data

    def inverse(self, data):
        with torch.amp.autocast(data.device.type, enabled=False):
            mask = torch.sum(torch.ne(data, 0), dim=-1) == 0
            inv_data = data * self.std.to(data.device) + self.mean.to(
                data.device)
            inv_data[mask] = 0
            return inv_data

    def to_dict(self):
        return {
            "mean": self.mean.detach().cpu().numpy().tolist(),
            "std": self.std.detach().cpu().numpy().tolist()
        }


class ObservationNormalizer:
    # [ADD] 정규화에서 절대 건드리지 말아야 할 키(항상 원본 그대로 유지)
    PASSTHROUGH_KEYS = {"agent_route_lane_order"}

    def __init__(self, normalization_dict):
        # [ADD] 혹시 dict 안에 들어있더라도 패스스루 키는 제거
        self._normalization_dict = {
            k: v
            for k, v in normalization_dict.items()
            if k not in self.PASSTHROUGH_KEYS
        }

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
            if k in cls.PASSTHROUGH_KEYS:
                continue
            ndt[k] = {
                "mean": torch.tensor(v["mean"], dtype=torch.float32),
                "std": torch.tensor(v["std"], dtype=torch.float32),
            }
        return cls(ndt)

    def __call__(self, data):
        device_type = list(data.values())[0].device.type
        with torch.amp.autocast(device_type, enabled=False):
            norm_data = copy(data)
            for k, v in self._normalization_dict.items():
                if (k not in data) or (
                        data[k] is None):  # Check if key `k` exists in `data`
                    continue
                if k in [
                        "ego_agent_past", "planner_future_11_dim",
                        "ego_future_gt_11_dim", "neighbor_agents_past"
                ]:
                    mask = torch.sum(torch.ne(data[k][..., :8], 0), dim=-1) == 0
                else:
                    mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
                norm_data[k] = (data[k] - v["mean"].to(
                    data[k].device)) / v["std"].to(data[k].device)
                norm_data[k][mask] = 0
            # 2) 패스스루 키는 원본 그대로(타입까지 보존/강제)
            if "agent_route_lane_order" in data:
                norm_data["agent_route_lane_order"] = data[
                    "agent_route_lane_order"].to(torch.long)

            return norm_data

    def inverse(self, data: dict) -> dict:
        device_type = list(data.values())[0].device.type

        with torch.amp.autocast(device_type, enabled=False):
            norm_data = copy(data)

            # 역정규화도 정의된 키만 수행
            for k, v in self._normalization_dict.items():
                if (k not in data) or (v is None):
                    continue
                if k in [
                        "ego_agent_past", "planner_future_11_dim",
                        "ego_future_gt_11_dim", "neighbor_agents_past"
                ]:
                    mask = torch.sum(torch.ne(data[k][..., :8], 0), dim=-1) == 0
                else:
                    mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
                norm_data[k] = data[k] * v["std"].to(
                    data[k].device) + v["mean"].to(data[k].device)
                norm_data[k][mask] = 0

            # 패스스루 키는 원본 그대로 (정수 유지)
            if "agent_route_lane_order" in data:
                norm_data["agent_route_lane_order"] = data[
                    "agent_route_lane_order"].to(torch.long)

            return norm_data

    def to_dict(self):
        return {
            k: {
                kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()
            } for k, v in self._normalization_dict.items()
        }
