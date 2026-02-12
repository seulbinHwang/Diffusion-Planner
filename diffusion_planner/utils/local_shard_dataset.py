from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


_LOCAL_SHARD_FORMAT = "dp_local_shards_v1"


def is_local_shard_manifest_path(path: Any) -> bool:
    """입력 경로가 로컬 shard manifest.json 형태인지 '대략' 판별합니다.

    Args:
        path: 파일 경로. shape: ()

    Returns:
        bool:
            True면 로컬 shard manifest일 가능성이 큼.
    """
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isfile(path):
        return False
    if not path.lower().endswith(".json"):
        return False

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return isinstance(obj, dict) and obj.get("format") == _LOCAL_SHARD_FORMAT
    except Exception:
        return False


def _load_manifest(manifest_path: str) -> Dict[str, Any]:
    """manifest.json을 읽고 dict로 돌려줍니다."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"manifest must be dict. got type={type(obj)}")
    if obj.get("format") != _LOCAL_SHARD_FORMAT:
        raise ValueError(
            f"unsupported manifest format: {obj.get('format')}. expected={_LOCAL_SHARD_FORMAT}"
        )
    return obj


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "t", "yes", "y"):
            return True
        if v in ("0", "false", "f", "no", "n"):
            return False
    return bool(default)


@dataclass(frozen=True)
class ShardPaths:
    """shard 파일 경로 묶음."""
    shard_id: int
    bin_path: str
    idx_path: str


class LocalShardDataset(Dataset):
    """로컬 shard(.bin/.idx)에서 샘플을 읽어오는 Dataset.

    핵심 목표
    --------
    - 학습 중에는 원본 npz를 열지 않습니다.
    - (offset, length) 표(.idx)를 이용해 .bin에서 blob만 읽고,
      그 blob을 복원해서 sample dict를 만듭니다.
    - sample dict는 기존 DiffusionPlannerData와 "키 구조"가 같도록,
      expected_keys 기준으로 없는 키는 None으로 채웁니다.

    주의
    ----
    - 이 Dataset은 'train shard'를 가정합니다.
      (validation/test shard까지 만들고 싶으면 동일 방식으로 확장 가능합니다.)
    """

    def __init__(
        self,
        shard_root: str,
        manifest_path: str,
        *,
        expected_predicted_neighbor_num: int,
        expected_use_agent_route_lane_order: bool,
        expected_eval_method: str = "train",
    ) -> None:
        """
        Args:
            shard_root (str):
                shard 루트. 예: "/workspace/local_shards_v1". shape: ()
            manifest_path (str):
                manifest.json 경로. 예: "/workspace/local_shards_v1/meta/manifest.json". shape: ()
            expected_predicted_neighbor_num (int):
                현재 학습 args.predicted_neighbor_num 값. shape: ()
            expected_use_agent_route_lane_order (bool):
                현재 학습 args.use_agent_route_lane_order 값. shape: ()
            expected_eval_method (str):
                보통 "train". shape: ()
        """
        if not isinstance(shard_root, str) or not shard_root:
            raise ValueError("shard_root must be a non-empty str.")
        if not isinstance(manifest_path, str) or not manifest_path:
            raise ValueError("manifest_path must be a non-empty str.")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")

        self._shard_root: str = shard_root
        self._manifest_path: str = manifest_path

        manifest = _load_manifest(manifest_path)

        self._shard_size: int = _safe_int(manifest.get("shard_size"), 0)
        self._shard_count: int = _safe_int(manifest.get("shard_count"), 0)
        self._num_samples: int = _safe_int(manifest.get("num_samples"), 0)

        if self._shard_size <= 0 or self._shard_count <= 0 or self._num_samples <= 0:
            raise ValueError(
                f"invalid manifest numbers: shard_size={self._shard_size}, "
                f"shard_count={self._shard_count}, num_samples={self._num_samples}"
            )
        if self._num_samples != (self._shard_size * self._shard_count):
            raise ValueError(
                f"num_samples mismatch: num_samples={self._num_samples}, "
                f"shard_size*shard_count={self._shard_size * self._shard_count}"
            )

        manifest_predicted = _safe_int(manifest.get("predicted_neighbor_num"), -1)
        manifest_use_aro = _safe_bool(manifest.get("use_agent_route_lane_order"), False)
        manifest_eval_method = str(manifest.get("eval_method", "")).lower()

        if int(manifest_predicted) != int(expected_predicted_neighbor_num):
            raise ValueError(
                "predicted_neighbor_num mismatch.\n"
                f"  manifest={manifest_predicted}\n"
                f"  current={expected_predicted_neighbor_num}\n"
                "→ shard를 현재 설정으로 다시 생성해야 합니다."
            )
        if bool(manifest_use_aro) != bool(expected_use_agent_route_lane_order):
            raise ValueError(
                "use_agent_route_lane_order mismatch.\n"
                f"  manifest={manifest_use_aro}\n"
                f"  current={expected_use_agent_route_lane_order}\n"
                "→ shard를 현재 설정으로 다시 생성해야 합니다."
            )
        if manifest_eval_method != str(expected_eval_method).lower():
            raise ValueError(
                "eval_method mismatch.\n"
                f"  manifest={manifest_eval_method}\n"
                f"  current={expected_eval_method}\n"
                "→ shard manifest를 확인하거나 shard를 다시 생성해야 합니다."
            )

        expected_keys = manifest.get("expected_keys", None)
        if not isinstance(expected_keys, list) or not all(isinstance(k, str) for k in expected_keys):
            raise TypeError("manifest.expected_keys must be List[str].")
        self._expected_keys: List[str] = list(expected_keys)

        shards = manifest.get("shards", None)
        if not isinstance(shards, list) or len(shards) != self._shard_count:
            raise ValueError(
                f"manifest.shards length mismatch. got={len(shards) if isinstance(shards, list) else None}, "
                f"expected={self._shard_count}"
            )

        self._shard_paths: List[ShardPaths] = []
        for item in shards:
            if not isinstance(item, dict):
                raise TypeError("manifest.shards[*] must be dict.")
            sid = _safe_int(item.get("shard_id"), -1)
            bin_rel = item.get("bin", None)
            idx_rel = item.get("idx", None)
            if sid < 0 or not isinstance(bin_rel, str) or not isinstance(idx_rel, str):
                raise ValueError(f"invalid shard entry: {item}")

            bin_path = bin_rel if os.path.isabs(bin_rel) else os.path.join(self._shard_root, bin_rel)
            idx_path = idx_rel if os.path.isabs(idx_rel) else os.path.join(self._shard_root, idx_rel)

            self._shard_paths.append(ShardPaths(shard_id=sid, bin_path=bin_path, idx_path=idx_path))

        # --- worker 프로세스 단위 캐시 ---
        self._cached_shard_id: Optional[int] = None
        self._cached_bin_fp: Optional[io.BufferedReader] = None
        self._cached_idx_table: Optional[np.ndarray] = None  # shape: (shard_size, 2), dtype=uint64

    @property
    def shard_size(self) -> int:
        return int(self._shard_size)

    @property
    def shard_count(self) -> int:
        return int(self._shard_count)

    @property
    def expected_keys(self) -> List[str]:
        return list(self._expected_keys)

    def __len__(self) -> int:
        return int(self._num_samples)

    def _close_cached_files(self) -> None:
        if self._cached_bin_fp is not None:
            try:
                self._cached_bin_fp.close()
            except Exception:
                pass
        self._cached_bin_fp = None
        self._cached_idx_table = None
        self._cached_shard_id = None

    def _load_idx_table(self, idx_path: str) -> np.ndarray:
        """idx 파일을 메모리로 읽어 (shard_size,2) 테이블로 만든다."""
        with open(idx_path, "rb") as f:
            raw = f.read()
        arr = np.frombuffer(raw, dtype=np.uint64)
        if arr.size != int(self._shard_size) * 2:
            raise ValueError(
                f"idx table size mismatch: got={arr.size}, expected={int(self._shard_size) * 2}, path={idx_path}"
            )
        table = arr.reshape(int(self._shard_size), 2)  # (shard_size, 2)
        return table

    def _ensure_shard_open(self, shard_id: int) -> None:
        """요청된 shard가 캐시에 없으면, 그 shard의 bin/idx를 열어 캐시에 올린다."""
        sid = int(shard_id)
        if self._cached_shard_id == sid and self._cached_bin_fp is not None and self._cached_idx_table is not None:
            return

        self._close_cached_files()

        if sid < 0 or sid >= int(self._shard_count):
            raise IndexError(f"shard_id out of range: {sid}")

        paths = self._shard_paths[sid]
        if not os.path.isfile(paths.bin_path):
            raise FileNotFoundError(f"bin not found: {paths.bin_path}")
        if not os.path.isfile(paths.idx_path):
            raise FileNotFoundError(f"idx not found: {paths.idx_path}")

        self._cached_bin_fp = open(paths.bin_path, "rb")
        self._cached_idx_table = self._load_idx_table(paths.idx_path)
        self._cached_shard_id = sid

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """샘플 1개를 읽어 sample dict로 반환합니다."""
        i = int(idx)
        if i < 0 or i >= int(self._num_samples):
            raise IndexError(f"index out of range: {i}")

        shard_id = i // int(self._shard_size)
        local_id = i % int(self._shard_size)

        self._ensure_shard_open(shard_id)

        assert self._cached_bin_fp is not None
        assert self._cached_idx_table is not None

        offset_u64 = self._cached_idx_table[int(local_id), 0]
        length_u64 = self._cached_idx_table[int(local_id), 1]
        offset = int(offset_u64)
        length = int(length_u64)

        self._cached_bin_fp.seek(offset)
        blob = self._cached_bin_fp.read(length)
        if len(blob) != length:
            raise IOError(
                f"failed to read full blob: want={length}, got={len(blob)}, shard_id={shard_id}, local_id={local_id}"
            )

        # blob -> npz -> dict
        sample: Dict[str, Any] = {}
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            files_set = set(data.files)

            for k in self._expected_keys:
                if k not in files_set:
                    sample[k] = None
                    continue

                v = data[k]
                # 문자열 scalar(np.array("..."))는 python str로 바꿔서 기존 형태에 더 가깝게 맞춤
                if isinstance(v, np.ndarray) and v.shape == () and v.dtype.kind in ("U", "S"):
                    try:
                        sample[k] = str(v.item())
                    except Exception:
                        sample[k] = str(v)
                else:
                    sample[k] = v

        return sample


class ShardDistributedSampler(Sampler[int]):
    """train 전용: shard 단위 셔플 + rank 분배 + worker 분배를 수행하는 Sampler.

    변경 목표
    --------
    - shard_count가 현재 world_size로 나누어 떨어지지 않아도 학습 가능해야 합니다.
    - 각 rank가 처리하는 샘플 수(=step 수)가 같아야 분산 학습이 멈추지 않습니다.

    동작 규칙
    --------
    1) 에폭마다 shard id(0..shard_count-1) 순서를 동일하게 섞습니다(모든 rank 동일).
    2) shard_count가 world_size로 나누어 떨어지지 않으면,
       "앞에서부터 world_size로 나누어 떨어지는 개수"까지만 이번 에폭에서 사용합니다.
       (남는 shard는 다음 에폭에서는 섞인 결과에 따라 포함될 수 있습니다.)
    3) 사용하기로 한 shard들을 rank별로 똑같이 나눕니다.
    4) rank 안에서는 worker별로 shard를 나눠 주고,
       DataLoader의 round-robin과 맞물리게 "배치 단위"로 인덱스를 내보냅니다.
    """

    def __init__(
        self,
        dataset: LocalShardDataset,
        *,
        num_replicas: int,
        rank: int,
        seed: int,
        num_workers: int,
        per_rank_batch_size: int,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(max(1, num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)

        self.num_workers = int(max(1, num_workers))
        self.per_rank_batch_size = int(max(1, per_rank_batch_size))

        self.shard_size = int(dataset.shard_size)
        self.shard_count = int(dataset.shard_count)

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas-1]. got rank={self.rank}, num_replicas={self.num_replicas}"
            )

        # ✅ 변경: shard_count가 world_size로 나눠떨어질 필요 없음
        usable_shard_count = (self.shard_count // self.num_replicas) * self.num_replicas
        if usable_shard_count <= 0:
            raise ValueError(
                "not enough shards for current world_size.\n"
                f"  shard_count={self.shard_count}\n"
                f"  world_size={self.num_replicas}\n"
                "→ shard_count >= world_size 인지 확인하세요."
            )

        self.usable_shard_count = int(usable_shard_count)  # shape: ()
        self.dropped_shards_per_epoch = int(self.shard_count - self.usable_shard_count)  # shape: ()

        # rank당 shard 개수는 항상 동일(DDP 정합)
        self.shards_per_rank = self.usable_shard_count // self.num_replicas
        self.num_samples = self.shards_per_rank * self.shard_size

        # (기존 로직 유지) shard 안에서 배치 단위로 끊어 내보내려면 나눠떨어지는게 가장 깔끔
        if (self.shard_size % self.per_rank_batch_size) != 0:
            raise ValueError(
                f"shard_size must be divisible by per_rank_batch_size.\n"
                f"  shard_size={self.shard_size}\n"
                f"  per_rank_batch_size={self.per_rank_batch_size}\n"
                "→ 보통 args.batch_size를 (per_rank_batch_size * world_size) 형태로 맞추면 해결됩니다."
            )

        self.batches_per_shard = self.shard_size // self.per_rank_batch_size
        self.epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return int(self.num_samples)

    def _shuffle_shard_ids_for_epoch(self) -> List[int]:
        shard_ids = list(range(self.shard_count))
        rng = np.random.default_rng(int(self.seed) + int(self.epoch))
        rng.shuffle(shard_ids)
        return shard_ids

    def __iter__(self) -> Iterator[int]:
        # 1) shard 순서 셔플 (모든 rank에서 동일)
        shard_ids = self._shuffle_shard_ids_for_epoch()

        # ✅ 2) 이번 에폭에서 사용할 shard만 선택 (world_size로 나눠떨어지는 개수)
        shard_ids = shard_ids[: int(self.usable_shard_count)]

        # 3) rank별로 동일한 크기로 분할
        start = int(self.rank) * int(self.shards_per_rank)
        end = start + int(self.shards_per_rank)
        rank_shards = shard_ids[start:end]  # length=shards_per_rank

        # 4) worker별 round-robin 분배 (겹치지 않게)
        worker_shards: List[List[int]] = [
            rank_shards[w::self.num_workers] for w in range(self.num_workers)
        ]
        max_wave = max((len(ws) for ws in worker_shards), default=0)

        # 5) "배치 단위"로 worker0→worker1→... 순서로 내보냄
        for wave_idx in range(max_wave):
            for k in range(self.batches_per_shard):
                for w in range(self.num_workers):
                    if wave_idx >= len(worker_shards[w]):
                        continue
                    shard_id = int(worker_shards[w][wave_idx])

                    base = shard_id * self.shard_size + k * self.per_rank_batch_size
                    for i in range(self.per_rank_batch_size):
                        yield int(base + i)
