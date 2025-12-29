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


# src/smart/metrics/wosac_metrics.py
class StateNormalizer:

    def __init__(self, mean: object, std: object) -> None:
        """상태 벡터(마지막 차원 4개 값)를 정규화/역정규화하는 클래스입니다.

        - mean/std는 내부적으로 항상 shape (4,) 로 보관합니다.
        - 실제 계산할 때는 입력 data의 차원 수(ndim)에 맞춰 mean/std를 (1, ..., 1, 4)로 바꿔서,
          data가 (N, 4)이든 (B, T, 4)이든 (B, A, T, 4)이든 같은 방식으로 처리합니다.

        Args:
            mean: 평균 값. 총 원소 개수가 4개여야 합니다. (예: (4,), (1,4), (1,1,4) 등)
            std: 표준편차 값. 총 원소 개수가 4개여야 합니다. (예: (4,), (1,4), (1,1,4) 등)

        Raises:
            ValueError: mean/std의 총 원소 개수가 4가 아니면 발생합니다.
        """
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
        """입력 값을 float32 torch 텐서 shape (4,)로 정리합니다.

        Args:
            values: 평균/표준편차 값. 어떤 모양이든 총 원소 개수가 4개면 허용합니다.
            name: 에러 메시지에 사용할 이름(예: "mean", "std").

        Returns:
            torch.Tensor: float32 텐서, shape (4,).

        Raises:
            ValueError: 총 원소 개수가 4가 아니면 발생합니다.
        """
        values_t = torch.as_tensor(values,
                                   dtype=torch.float32).reshape(-1)  # (N,)
        if values_t.numel() != 4:
            raise ValueError(
                f"{name}는 총 4개 값이어야 합니다. "
                f"(받은 원소 개수={values_t.numel()}, 받은 shape={tuple(torch.as_tensor(values).shape)})"
            )
        return values_t  # (4,)

    @staticmethod
    def _reshape_stats_for_data(stats_1d4: torch.Tensor,
                                data: torch.Tensor) -> torch.Tensor:
        """stats(4,)를 data(...,4)에 맞게 (1, ..., 1, 4) 모양으로 바꿉니다.

        Args:
            stats_1d4: 통계 값 텐서, shape (4,).
            data: 입력 데이터 텐서, shape (..., 4).

        Returns:
            torch.Tensor: stats를 data의 ndim에 맞춘 텐서, shape (1, ..., 1, 4).
                예)
                - data shape (N, 4)      -> (1, 4)
                - data shape (B, T, 4)   -> (1, 1, 4)
                - data shape (B,A,T,4)   -> (1, 1, 1, 4)
        """
        # data: (..., 4)
        leading_ones = [1] * (data.ndim - 1)
        return stats_1d4.to(device=data.device).view(*leading_ones, 4)

    @staticmethod
    def _zero_vector_mask(data: torch.Tensor) -> torch.Tensor:
        """마지막 4개 값이 모두 0인 위치를 True로 표시하는 마스크를 만듭니다.

        Args:
            data: 입력 데이터 텐서, shape (..., 4).

        Returns:
            torch.Tensor: bool 마스크, shape (...).
                data[..., :]가 전부 0이면 True, 아니면 False.
        """
        # mask: (...), 마지막 dim(4)을 기준으로 "전부 0인지" 확인
        return torch.sum(torch.ne(data, 0), dim=-1) == 0

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """data를 정규화합니다.

        - 입력 data의 마지막 차원이 4인지만 확인합니다. (나머지 차원 수/크기는 무엇이든 OK)
        - data[..., :]가 전부 0인 위치는 정규화 후에도 그대로 0으로 유지합니다.

        Args:
            data: 입력 데이터 텐서, shape (..., 4).

        Returns:
            torch.Tensor: 정규화된 텐서, shape (..., 4).
        """
        if data.shape[-1] != 4:
            raise ValueError(
                f"data의 마지막 차원은 4여야 합니다. (받은 shape={tuple(data.shape)})")

        with torch.amp.autocast(data.device.type, enabled=False):
            # mask: (...), 마지막 4개 값이 전부 0인 위치
            mask = self._zero_vector_mask(data)  # (...,)

            mean = self._reshape_stats_for_data(self.mean, data)  # (1,...,1,4)
            std = self._reshape_stats_for_data(self.std, data)  # (1,...,1,4)

            norm_data = (data - mean) / std  # (..., 4)
            norm_data[mask] = 0
            return norm_data

    def inverse(self, data: torch.Tensor) -> torch.Tensor:
        """정규화된 data를 원래 값으로 되돌립니다.

        - 입력 data의 마지막 차원이 4인지만 확인합니다.
        - data[..., :]가 전부 0인 위치는 역변환 후에도 그대로 0으로 유지합니다.

        Args:
            data: 입력 데이터 텐서, shape (..., 4).

        Returns:
            torch.Tensor: 역변환된 텐서, shape (..., 4).
        """
        if data.shape[-1] != 4:
            raise ValueError(
                f"data의 마지막 차원은 4여야 합니다. (받은 shape={tuple(data.shape)})")

        with torch.amp.autocast(data.device.type, enabled=False):
            mask = self._zero_vector_mask(data)  # (...,)

            mean = self._reshape_stats_for_data(self.mean, data)  # (1,...,1,4)
            std = self._reshape_stats_for_data(self.std, data)  # (1,...,1,4)

            inv_data = data * std + mean  # (..., 4)
            inv_data[mask] = 0
            return inv_data

    def to_dict(self) -> dict:
        """현재 mean/std를 저장용 dict로 바꿉니다.

        Returns:
            dict: {"mean": ..., "std": ...}
                mean/std는 (1, 1, 4) 모양의 중첩 리스트로 내보냅니다.
        """
        mean_1x1x4 = self.mean.view(1, 1, 4).detach().cpu().numpy().tolist()
        std_1x1x4 = self.std.view(1, 1, 4).detach().cpu().numpy().tolist()
        return {"mean": mean_1x1x4, "std": std_1x1x4}


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
