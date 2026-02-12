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
    """입력 경로가 로컬 shard manifest.json 형태인지 '대략' 판별합니다."""
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


class _ReadonlyMemoryViewFile:
    """memoryview를 '파일처럼' 읽게 해주는 가벼운 래퍼입니다.

    목적
    ----
    - bytes 덩어리(chunk) 안의 일부 구간(memoryview 슬라이스)을
      np.load가 읽을 수 있도록 "read/seek/tell" 인터페이스를 제공합니다.
    - 중요한 점:
      - 이 객체는 **원본 chunk를 복사해서 들고 있지 않습니다.**
      - 즉, 샘플마다 blob 전체를 bytes로 새로 만드는(tobytes) 큰 복사를 피합니다.

    주의
    ----
    - np.load(zip 기반)는 내부적으로 read()를 호출하므로
      read()가 bytes를 만들어 반환하는 "작은 복사"는 남습니다.
      하지만 우리가 원래 하던 "샘플 전체 blob을 한 번 더 통째로 복사"는 제거됩니다.

    Args:
        view (memoryview):
            npz blob 바이트를 가리키는 뷰. shape: (N_bytes,)
    """

    def __init__(self, view: memoryview) -> None:
        self._view: memoryview = view  # shape: (N_bytes,)
        self._pos: int = 0

    def tell(self) -> int:
        """현재 읽기 위치를 반환합니다."""
        return int(self._pos)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """읽기 위치를 이동합니다."""
        n = int(self._view.nbytes)
        off = int(offset)

        if whence == io.SEEK_SET:
            new_pos = off
        elif whence == io.SEEK_CUR:
            new_pos = int(self._pos) + off
        elif whence == io.SEEK_END:
            new_pos = n + off
        else:
            raise ValueError(f"invalid whence: {whence}")

        new_pos = max(0, min(n, int(new_pos)))
        self._pos = int(new_pos)
        return int(self._pos)

    def read(self, size: int = -1) -> bytes:
        """현재 위치부터 size 만큼 읽어 bytes로 반환합니다."""
        n = int(self._view.nbytes)
        if size is None or int(size) < 0:
            size_i = n - int(self._pos)
        else:
            size_i = int(size)

        if size_i <= 0 or int(self._pos) >= n:
            return b""

        end = min(n, int(self._pos) + int(size_i))
        out = self._view[int(self._pos):end].tobytes()
        self._pos = int(end)
        return out

    def close(self) -> None:
        """zipfile이 close()를 부를 수 있어서 준비합니다(실제 자원 해제는 없음)."""
        return


class LocalShardDataset(Dataset):
    """로컬 shard(.bin/.idx)에서 샘플을 읽어오는 Dataset.

    핵심 변경(성능)
    -------------
    - __getitems__(indices)에서:
      - 큰 chunk 1번 read
      - 샘플별로 bytes(tobytes) 복사 생성 ❌ 제거
      - 대신 memoryview 슬라이스(복사 없는 뷰)로 np.load 수행 ✅

    기대 효과(팩트 기반)
    -------------------
    - 배치 1개(예: 256개)당 "샘플 blob 전체 복사" 단계가 사라집니다.
    - 데이터가 클수록(샤드가 커질수록) CPU/메모리 대역폭 낭비가 줄 가능성이 큽니다.
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
        return arr.reshape(int(self._shard_size), 2)  # shape: (shard_size, 2)

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

        # buffering을 크게 두면 큰 read를 할 때 유리한 경우가 많습니다.
        self._cached_bin_fp = open(paths.bin_path, "rb", buffering=1024 * 1024)
        self._cached_idx_table = self._load_idx_table(paths.idx_path)
        self._cached_shard_id = sid

    def _decode_blob_view_to_sample(self, blob_view: memoryview) -> Dict[str, Any]:
        """npz blob(memoryview)을 sample dict로 복원합니다.

        핵심
        ----
        - blob_view는 chunk 내부를 가리키는 "복사 없는 뷰"입니다.
        - np.load가 요구하는 read/seek/tell을 제공하기 위해 _ReadonlyMemoryViewFile을 씁니다.
        - 여기서 샘플 blob 전체를 bytes로 새로 만들지 않습니다.

        Args:
            blob_view: npz blob 바이트 뷰. shape: (N_bytes,)

        Returns:
            sample: Dict[str, Any]
        """
        fobj = _ReadonlyMemoryViewFile(blob_view)

        sample: Dict[str, Any] = {}
        with np.load(fobj, allow_pickle=False) as data:
            files_set = set(data.files)
            for k in self._expected_keys:
                if k not in files_set:
                    sample[k] = None
                    continue

                v = data[k]
                if isinstance(v, np.ndarray) and v.shape == () and v.dtype.kind in ("U", "S"):
                    try:
                        sample[k] = str(v.item())
                    except Exception:
                        sample[k] = str(v)
                else:
                    sample[k] = v
        return sample

    def _read_one_blob(self, *, offset: int, length: int) -> bytes:
        """bin 파일에서 (offset,length)만큼 읽어 blob(bytes)을 돌려줍니다."""
        assert self._cached_bin_fp is not None
        self._cached_bin_fp.seek(int(offset))
        blob = self._cached_bin_fp.read(int(length))
        if len(blob) != int(length):
            raise IOError(f"failed to read full blob: want={length}, got={len(blob)}")
        return blob

    def _read_one_chunk_for_many_blobs(
        self,
        offsets: np.ndarray,
        lengths: np.ndarray,
    ) -> Tuple[memoryview, int, np.ndarray, np.ndarray]:
        """여러 blob이 포함된 구간을 한 번에 읽고, blob 뷰를 만들 준비를 합니다.

        Args:
            offsets: uint64 배열. shape: (N,)
            lengths: uint64 배열. shape: (N,)

        Returns:
            Tuple[memoryview, int, np.ndarray, np.ndarray]:
                - chunk_view: 읽어온 큰 덩어리 뷰. shape: (chunk_bytes,)
                - start: 이 chunk가 원본 bin에서 시작한 offset. shape: ()
                - rel_offsets: chunk_view 기준 상대 offset(int64). shape: (N,)
                - lens: 각 blob 길이(int64). shape: (N,)
        """
        assert self._cached_bin_fp is not None

        offs = offsets.astype(np.int64, copy=False)  # shape: (N,)
        lens = lengths.astype(np.int64, copy=False)  # shape: (N,)

        start = int(np.min(offs))
        end = int(np.max(offs + lens))
        total_len = int(max(0, end - start))

        self._cached_bin_fp.seek(start)
        chunk = self._cached_bin_fp.read(total_len)
        if len(chunk) != total_len:
            raise IOError(f"failed to read chunk: want={total_len}, got={len(chunk)}")

        rel_offsets = (offs - start).astype(np.int64, copy=False)  # shape: (N,)
        lens2 = lens.astype(np.int64, copy=False)  # shape: (N,)

        return memoryview(chunk), int(start), rel_offsets, lens2

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """샘플 1개를 읽어 sample dict로 반환합니다."""
        i = int(idx)
        if i < 0 or i >= int(self._num_samples):
            raise IndexError(f"index out of range: {i}")

        shard_id = i // int(self._shard_size)
        local_id = i % int(self._shard_size)

        self._ensure_shard_open(shard_id)

        assert self._cached_idx_table is not None
        offset_u64 = self._cached_idx_table[int(local_id), 0]
        length_u64 = self._cached_idx_table[int(local_id), 1]

        # 단일 샘플은 기존처럼 bytes로 읽습니다(배치 최적화는 __getitems__에서만).
        blob = self._read_one_blob(offset=int(offset_u64), length=int(length_u64))
        return self._decode_blob_view_to_sample(memoryview(blob))

    def __getitems__(self, indices: Sequence[int]) -> List[Dict[str, Any]]:
        """여러 인덱스를 한 번에 읽습니다(배치 최적화).

        변경점(핵심)
        ----------
        - 기존: chunk read 후 mv[...] .tobytes()로 샘플 bytes를 새로 만들었다(큰 복사).
        - 변경: chunk read 후 mv[...] 슬라이스(memoryview)를 그대로 np.load에 전달한다(큰 복사 제거).

        Args:
            indices: 인덱스 목록. length=N

        Returns:
            List[Dict[str, Any]]: length=N
        """
        if not indices:
            return []

        n = int(len(indices))
        out: List[Optional[Dict[str, Any]]] = [None] * n

        groups: Dict[int, List[Tuple[int, int]]] = {}
        shard_size = int(self._shard_size)

        for pos, raw_i in enumerate(indices):
            i = int(raw_i)
            if i < 0 or i >= int(self._num_samples):
                raise IndexError(f"index out of range: {i}")

            sid = i // shard_size
            lid = i % shard_size
            groups.setdefault(int(sid), []).append((int(pos), int(lid)))

        for sid, pos_lids in groups.items():
            self._ensure_shard_open(int(sid))
            assert self._cached_idx_table is not None

            pos_list = [p for p, _ in pos_lids]      # shape: (N_g,)
            lid_list = [lid for _, lid in pos_lids]  # shape: (N_g,)

            lids = np.asarray(lid_list, dtype=np.int64)                # shape: (N_g,)
            offsets_u64 = self._cached_idx_table[lids, 0]              # shape: (N_g,)
            lengths_u64 = self._cached_idx_table[lids, 1]              # shape: (N_g,)

            chunk_view, _, rel_offsets, lens = self._read_one_chunk_for_many_blobs(
                offsets=offsets_u64,
                lengths=lengths_u64,
            )

            for p, rel, ln in zip(pos_list, rel_offsets.tolist(), lens.tolist()):
                r = int(rel)
                l = int(ln)
                blob_view = chunk_view[r:r + l]  # shape: (l,)
                out[p] = self._decode_blob_view_to_sample(blob_view)

        final: List[Dict[str, Any]] = []
        for item in out:
            if item is None:
                raise RuntimeError("internal error: some samples were not filled.")
            final.append(item)
        return final


class ShardDistributedSampler(Sampler[int]):
    """train 전용: shard 단위 셔플 + rank 분배 + worker 분배를 수행하는 Sampler."""

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

        if (self.shard_count % self.num_replicas) != 0:
            raise ValueError(
                f"shard_count must be divisible by world_size. shard_count={self.shard_count}, world_size={self.num_replicas}"
            )

        if (self.shard_size % self.per_rank_batch_size) != 0:
            raise ValueError(
                f"shard_size must be divisible by per_rank_batch_size. "
                f"shard_size={self.shard_size}, per_rank_batch_size={self.per_rank_batch_size}"
            )

        self.shards_per_rank = self.shard_count // self.num_replicas
        self.num_samples = self.shards_per_rank * self.shard_size

        self.batches_per_shard = self.shard_size // self.per_rank_batch_size

        self.epoch: int = 0

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas-1]. got rank={self.rank}, num_replicas={self.num_replicas}"
            )

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
        shard_ids = self._shuffle_shard_ids_for_epoch()

        start = int(self.rank) * int(self.shards_per_rank)
        end = start + int(self.shards_per_rank)
        rank_shards = shard_ids[start:end]

        worker_shards: List[List[int]] = [
            rank_shards[w::self.num_workers] for w in range(self.num_workers)
        ]
        max_wave = max((len(ws) for ws in worker_shards), default=0)

        for wave_idx in range(max_wave):
            for k in range(self.batches_per_shard):
                for w in range(self.num_workers):
                    if wave_idx >= len(worker_shards[w]):
                        continue
                    shard_id = int(worker_shards[w][wave_idx])

                    base = shard_id * self.shard_size + k * self.per_rank_batch_size
                    for i in range(self.per_rank_batch_size):
                        yield int(base + i)
