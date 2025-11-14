from copy import copy, deepcopy
import torch

from diffusion_planner.utils.train_utils import openjson
from pathlib import Path  # 추가
try:  # 추가
    from hydra.utils import to_absolute_path  # 추가
except Exception:  # 추가

    def to_absolute_path(path: str) -> str:  # 추가
        return str(Path(path).expanduser().resolve())  # 추가


class StateNormalizer:

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)  # (10, 1, 4)
        # to float32
        self.mean = self.mean.float()
        self.std = torch.as_tensor(std)  # (10, 1, 4)
        self.std = self.std.float()

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)

        mean = [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std = [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    @classmethod
    def from_json2(cls, args_dict):
        # args_dict["normalization_file_path"]: "normalization.json"
        path_str = args_dict.get("normalization_file_path",
                                 "normalization.json")  # 추가
        data = openjson(to_absolute_path(path_str))  # 추가
        mean = [[data["neighbor"]["mean"]]
               ] * args_dict["predicted_neighbor_num"]
        std = [[data["neighbor"]["std"]]] * args_dict["predicted_neighbor_num"]
        return cls(mean, std)

    def __call__(self, data):
        # data: (256, 10, 80, 4)
        # mean, std: (10, 1, 4)
        with torch.cuda.amp.autocast(enabled=False):
            return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
        with torch.cuda.amp.autocast(enabled=False):
            return data * self.std.to(data.device) + self.mean.to(data.device)

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
                                 "normalization.json")  # 추가
        data = openjson(to_absolute_path(path_str))  # 추가

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
        with torch.cuda.amp.autocast(enabled=False):
            norm_data = copy(data)
            for k, v in self._normalization_dict.items():
                if k not in data:  # Check if key `k` exists in `data`
                    continue
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
        with torch.cuda.amp.autocast(enabled=False):
            norm_data = copy(data)

            # 역정규화도 정의된 키만 수행
            for k, v in self._normalization_dict.items():
                if k not in data:
                    continue
                mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
                norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(
                    data[k].device)
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
