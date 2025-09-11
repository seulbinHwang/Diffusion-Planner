from copy import copy, deepcopy
import torch

from diffusion_planner.utils.train_utils import openjson


class StateNormalizer:

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean) # (10, 1, 4)
        self.std = torch.as_tensor(std) # (10, 1, 4)

    @classmethod
    def from_json(cls, args):
        data = openjson(args.normalization_file_path)
        a  = [data["neighbor"]["mean"]]
        print(
            f"[StateNormalizer] a: shape: {data['neighbor']['mean']}")

        mean = [[data["neighbor"]["mean"]]] * args.predicted_neighbor_num
        std =  [[data["neighbor"]["std"]]] * args.predicted_neighbor_num
        return cls(mean, std)

    @classmethod
    def from_json2(cls, args_dict):
        # args_dict["normalization_file_path"]: "normalization.json"
        # TODO: 내가 원하는 normalization_file_path는 내가 코드를 실행한(래포지토리의 가장 상위) 위치에 있는 normalization.json
        data = openjson(args_dict["normalization_file_path"])
        print("[StateNormalizer2] data:", data)
        mean = [[data["neighbor"]["mean"]]] * args_dict["predicted_neighbor_num"]
        std =  [[data["neighbor"]["std"]]] * args_dict["predicted_neighbor_num"]
        return cls(mean, std)

    def __call__(self, data):
        # data: (256, 10, 80, 4)
        # mean, std: (10, 1, 4)
        return (data - self.mean.to(data.device)) / self.std.to(data.device)

    def inverse(self, data):
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
            k: v for k, v in normalization_dict.items()
            if k not in self.PASSTHROUGH_KEYS
        }

    @classmethod
    def from_json(cls, args):
        if isinstance(args, str):
            path = args
        else:
            path = args.normalization_file_path

        data = openjson(path)
        print("[ObservationNormalizer2] data:", data)

        ndt = {}
        for k, v in data.items():
            if k not in ["ego", "neighbor"]:
                print(f"[ObservationNormalizer] key: {k} is and value: {v['mean'].shape}")
                ndt[k] = {
                    "mean": torch.tensor(v["mean"], dtype=torch.float32),
                    "std": torch.tensor(v["std"], dtype=torch.float32)
                }
        return cls(ndt)

    @classmethod
    def from_json2(cls, args_dict):
        path = args_dict.normalization_file_path
        data = openjson(path)
        print("[ObservationNormalizer2] data:", data)

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
            norm_data["agent_route_lane_order"] = data["agent_route_lane_order"].to(torch.long)

        return norm_data

    def inverse(self, data: dict) -> dict:
        norm_data = copy(data)

        # 역정규화도 정의된 키만 수행
        for k, v in self._normalization_dict.items():
            if k not in data:
                continue
            mask = torch.sum(torch.ne(data[k], 0), dim=-1) == 0
            norm_data[k] = data[k] * v["std"].to(data[k].device) + v["mean"].to(data[k].device)
            norm_data[k][mask] = 0

        # 패스스루 키는 원본 그대로 (정수 유지)
        if "agent_route_lane_order" in data:
            norm_data["agent_route_lane_order"] = data["agent_route_lane_order"].to(torch.long)

        return norm_data

    def to_dict(self):
        return {
            k: {
                kk: vv.detach().cpu().numpy().tolist() for kk, vv in v.items()
            } for k, v in self._normalization_dict.items()
        }
