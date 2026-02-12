from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from diffusion_planner.utils.dataset import DiffusionPlannerData


_LOCAL_SHARD_FORMAT = "dp_local_shards_v1"


# -----------------------------
# 병렬 빌드용 전역(워커 프로세스 안에서만 사용)
# -----------------------------
_WORKER_DS: Optional[DiffusionPlannerData] = None
_WORKER_OUT_ROOT: Optional[str] = None
_WORKER_SHARD_SIZE: int = 0
_WORKER_BIN_BUFFER_BYTES: int = 0


def _bool_arg(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _available_cpu_count() -> int:
    """현재 프로세스가 '실제로 쓸 수 있는' CPU 개수를 추정합니다.

    이유
    ----
    컨테이너/쿠버네티스에서는 host 전체 CPU 개수(os.cpu_count)와,
    실제로 할당된 CPU 개수가 다를 수 있습니다.
    가능한 경우 sched_getaffinity를 사용해 현재 프로세스에 허용된 CPU 개수를 우선합니다.

    Returns:
        int:
            사용 가능한 CPU 개수. shape: ()
    """
    try:
        return int(len(os.sched_getaffinity(0)))
    except Exception:
        c = os.cpu_count() or 1
        return int(max(1, c))


def _default_build_workers() -> int:
    """shard 빌드 병렬 worker 기본값을 정합니다.

    설계 의도
    --------
    - 이 작업은 "원본 파일 읽기 + 변환 + 저장"이 섞여 있어,
      너무 많은 프로세스를 쓰면 저장소가 먼저 포화될 수 있습니다.
    - 그래서 기본값은 보수적으로 8개(또는 그 이하)로 둡니다.
      (필요하면 CLI에서 더 키우면 됩니다.)

    Returns:
        int:
            기본 worker 수. shape: ()
    """
    cpu = _available_cpu_count()
    return int(max(1, min(8, cpu)))


def _format_duration_hm(seconds: float) -> str:
    """초 단위 시간을 'XhYYm' 형태로 바꿉니다.

    Args:
        seconds (float): 초 단위 시간. shape: ()

    Returns:
        str:
            예: 0h03m, 2h15m 같은 문자열. shape: ()
    """
    sec = float(max(0.0, seconds))
    hours = int(sec // 3600.0)
    minutes = int((sec % 3600.0) // 60.0)
    return f"{hours}h{minutes:02d}m"


def _safe_rate(units_done: int, elapsed_sec: float) -> float:
    """처리 속도(단위/초)를 안전하게 계산합니다.

    Args:
        units_done (int): 지금까지 처리한 단위 개수. shape: ()
        elapsed_sec (float): 지금까지 경과한 시간(초). shape: ()

    Returns:
        float:
            단위/초. shape: ()
    """
    done = int(max(0, units_done))
    t = float(max(1e-9, elapsed_sec))
    return float(done) / t


def _estimate_total_and_remaining_sec(
    total_units: int,
    units_done: int,
    elapsed_sec: float,
) -> Tuple[Optional[float], Optional[float]]:
    """현재까지의 평균 속도로 전체/남은 시간을 추정합니다.

    Args:
        total_units (int): 전체 단위 개수. shape: ()
        units_done (int): 처리 완료 단위 개수. shape: ()
        elapsed_sec (float): 경과 시간(초). shape: ()

    Returns:
        Tuple[Optional[float], Optional[float]]:
            - est_total_sec: 예상 총 소요 시간(초) 또는 None. shape: ()
            - est_remaining_sec: 예상 남은 시간(초) 또는 None. shape: ()
    """
    total = int(max(0, total_units))
    done = int(max(0, units_done))
    if total <= 0 or done <= 0:
        return None, None

    rate = _safe_rate(done, elapsed_sec)  # units/sec
    if rate <= 0.0:
        return None, None

    est_total = float(total) / rate
    est_remaining = max(0.0, est_total - float(elapsed_sec))
    return float(est_total), float(est_remaining)


class _ProgressPrinter:
    """shard 생성 진행 상황을 '매 N분마다' 출력하는 도구입니다.

    출력 내용:
        - 예상 총 소요 시간(시/분)
        - 지금까지 경과 시간(시/분)
        - 진행률(%)
        - 샘플/샤드 기준 진행량
    """

    def __init__(
        self,
        *,
        total_samples: int,
        total_shards: int,
        interval_sec: float,
    ) -> None:
        """초기화.

        Args:
            total_samples (int): 전체 샘플 개수. shape: ()
            total_shards (int): 전체 shard 개수. shape: ()
            interval_sec (float): 출력 간격(초). shape: ()
        """
        self._total_samples: int = int(max(0, total_samples))
        self._total_shards: int = int(max(0, total_shards))
        self._interval_sec: float = float(max(1.0, interval_sec))

        self._start_wall_sec: float = float(time.perf_counter())
        self._last_print_wall_sec: float = self._start_wall_sec

    def maybe_print(
        self,
        *,
        done_samples: int,
        done_shards: int,
        force: bool = False,
    ) -> None:
        """필요한 시점(매 N분 또는 강제)마다 진행 상황을 출력합니다.

        Args:
            done_samples (int): 처리 완료 샘플 개수. shape: ()
            done_shards (int): 완료된 shard 개수. shape: ()
            force (bool): True면 간격과 무관하게 즉시 출력. shape: ()

        Returns:
            None
        """
        now = float(time.perf_counter())
        if (not force) and ((now - self._last_print_wall_sec) < self._interval_sec):
            return

        elapsed = now - self._start_wall_sec
        ds = int(max(0, done_samples))
        dsh = int(max(0, done_shards))

        total = max(1, self._total_samples)
        percent = (float(ds) / float(total)) * 100.0

        est_total_sec, est_remaining_sec = _estimate_total_and_remaining_sec(
            total_units=self._total_samples,
            units_done=ds,
            elapsed_sec=elapsed,
        )

        elapsed_hm = _format_duration_hm(elapsed)
        if est_total_sec is None or est_remaining_sec is None:
            total_hm = "?"
            remain_hm = "?"
        else:
            total_hm = _format_duration_hm(est_total_sec)
            remain_hm = _format_duration_hm(est_remaining_sec)

        rate_sps = _safe_rate(ds, elapsed)

        print(
            "[local_shards][progress] "
            f"{percent:.2f}% | elapsed={elapsed_hm} | remaining≈{remain_hm} | total≈{total_hm} | "
            f"samples={ds}/{self._total_samples} | shards={dsh}/{self._total_shards} | "
            f"avg_speed≈{rate_sps:.1f} samples/s",
            flush=True,
        )

        self._last_print_wall_sec = now


def _to_save_array(value: Any, key: str) -> Optional[np.ndarray]:
    """np.savez에 넣을 수 있는 형태로 변환합니다.

    규칙
    ----
    - None이면 저장하지 않습니다.
    - np.ndarray는 그대로 사용.
    - str / 숫자 scalar는 np.asarray로 0-d 배열로 바꿉니다.
    - object dtype 배열은 allow_pickle=False에서 위험/불편하므로 금지합니다.

    Returns:
        Optional[np.ndarray]:
            저장할 배열. shape: 다양
    """
    if value is None:
        return None

    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise ValueError(f"[shard] object dtype is not allowed. key='{key}'")
        return value

    arr = np.asarray(value)
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "O":
        raise ValueError(f"[shard] object dtype is not allowed. key='{key}'")
    return arr


def _serialize_sample_to_npz_buffer(sample: Dict[str, Any], buffer: io.BytesIO) -> memoryview:
    """sample dict를 '압축 없는 npz' 형태로 buffer에 쓰고, 그 메모리 뷰를 돌려줍니다.

    속도 최적화 포인트
    -----------------
    - 매 샘플마다 bytes 객체를 새로 만들지 않고(buffer를 재사용),
      buffer.getbuffer()로 얻은 memoryview를 바로 파일에 씁니다.
    - 이렇게 하면 "버퍼->bytes 복사" 1번을 줄일 수 있습니다.

    Args:
        sample (Dict[str, Any]):
            샘플 dict. value는 np.ndarray 또는 scalar 등. shape: 다양
        buffer (io.BytesIO):
            재사용할 메모리 버퍼. shape: ()

    Returns:
        memoryview:
            npz 바이트를 가리키는 memoryview. shape: (N,)
            (주의) caller는 write 후 반드시 view.release()를 호출해야 합니다.
    """
    buffer.seek(0)
    buffer.truncate(0)

    save_dict: Dict[str, np.ndarray] = {}
    for k, v in sample.items():
        arr = _to_save_array(v, key=str(k))
        if arr is None:
            continue
        save_dict[str(k)] = arr

    np.savez(buffer, **save_dict)  # ✅ 변경
    return buffer.getbuffer()


def _write_idx_table_fast(idx_path: str, idx_table: np.ndarray) -> None:
    """(offset,length) 테이블을 idx 파일로 빠르게 저장합니다.

    포맷
    ----
    - (shard_size, 2) shape의 uint64 테이블을
      row-major(C-order)로 그대로 저장합니다.
    - 각 행: [offset, length]  # 둘 다 uint64

    Args:
        idx_path (str): 저장할 idx 파일 경로. shape: ()
        idx_table (np.ndarray):
            shape: (shard_size, 2), dtype=uint64
    """
    if not isinstance(idx_table, np.ndarray):
        raise TypeError("idx_table must be a numpy array.")
    if idx_table.ndim != 2 or idx_table.shape[1] != 2:
        raise ValueError(f"idx_table shape must be (N,2). got {idx_table.shape}")

    raw = idx_table.astype(np.dtype("<u8"), copy=False).tobytes(order="C")
    with open(idx_path, "wb") as f:
        f.write(raw)


def _compute_keep_count(
    total_count: int,
    *,
    shard_size: int,
    world_size: int,
) -> int:
    """shard_size에 딱 맞게 떨어지도록 keep_count를 계산합니다.

    변경 의도
    --------
    - shard를 "GPU 개수와 무관"하게 만들기 위해
      keep_count를 shard_size 배수로만 맞춥니다.

    Args:
        total_count: 원본 파일 개수. shape: ()
        shard_size: 2048. shape: ()
        world_size: (호환용) 더 이상 keep_count 계산에 사용하지 않음. shape: ()

    Returns:
        keep_count: shape: ()
    """
    total = int(max(0, total_count))
    unit = int(max(1, shard_size))
    keep = (total // unit) * unit
    return int(keep)


def _build_one_shard_files(
    *,
    ds: DiffusionPlannerData,
    out_root: str,
    shard_id: int,
    shard_size: int,
    bin_buffer_bytes: int,
) -> Set[str]:
    """shard 1개(.bin/.idx)를 생성합니다.

    병렬 처리를 위해 shard 단위로 작업을 쪼갭니다.
    shard 파일은 shard_id마다 경로가 고유하므로, 여러 프로세스가 동시에 실행해도 충돌이 없습니다.

    Args:
        ds (DiffusionPlannerData):
            원본 데이터셋 로더. ds[i] -> sample dict. shape: ()
        out_root (str):
            shard 루트. 예: /workspace/local_shards_v1. shape: ()
        shard_id (int):
            생성할 shard 번호. shape: ()
        shard_size (int):
            shard 당 샘플 개수. 예: 2048. shape: ()
        bin_buffer_bytes (int):
            bin 파일 write 버퍼 크기(바이트). shape: ()

    Returns:
        Set[str]:
            이 shard에서 관측한 sample dict 키 집합. shape: (K,)
    """
    sid = int(shard_id)
    shard_size_i = int(max(1, shard_size))
    out_root_abs = os.path.abspath(out_root)

    bin_rel = os.path.join("data", f"shard_{sid:06d}.bin")
    idx_rel = os.path.join("data", f"shard_{sid:06d}.idx")
    bin_path = os.path.join(out_root_abs, bin_rel)
    idx_path = os.path.join(out_root_abs, idx_rel)

    keys_set: Set[str] = set()

    idx_table = np.zeros((shard_size_i, 2), dtype=np.uint64)  # shape: (shard_size, 2)
    offset = 0

    buff = io.BytesIO()

    with open(bin_path, "wb", buffering=int(max(0, bin_buffer_bytes))) as bin_f:
        base = sid * shard_size_i
        for j in range(shard_size_i):
            global_idx = base + j
            sample = ds[global_idx]

            keys_set.update(sample.keys())

            view = _serialize_sample_to_npz_buffer(sample, buff)
            try:
                length = int(view.nbytes)
                bin_f.write(view)
            finally:
                view.release()

            idx_table[j, 0] = np.uint64(offset)
            idx_table[j, 1] = np.uint64(length)
            offset += length

    _write_idx_table_fast(idx_path, idx_table)
    return keys_set


def _init_worker(
    src_data_dir: str,
    src_list_json: str,
    predicted_neighbor_num: int,
    use_agent_route_lane_order: bool,
    raw_list: List[str],
    shard_size: int,
    out_root: str,
    bin_buffer_bytes: int,
) -> None:
    """multiprocessing worker 초기화 함수입니다.

    각 worker 프로세스는:
      - DiffusionPlannerData를 1번 생성하고
      - data_list를 동일한 shuffled raw_list로 교체한 뒤
      - 이후 여러 shard를 연속으로 처리합니다.

    Args:
        src_data_dir (str): 원본 npz 폴더. shape: ()
        src_list_json (str): 원본 파일 리스트 json. shape: ()
        predicted_neighbor_num (int): dataset 옵션. shape: ()
        use_agent_route_lane_order (bool): dataset 옵션. shape: ()
        raw_list (List[str]): 전역 shuffle된 파일 리스트. length: (keep_count,)
        shard_size (int): shard 당 샘플 수. shape: ()
        out_root (str): shard 출력 루트. shape: ()
        bin_buffer_bytes (int): bin write buffer 크기. shape: ()
    """
    global _WORKER_DS, _WORKER_OUT_ROOT, _WORKER_SHARD_SIZE, _WORKER_BIN_BUFFER_BYTES

    ds = DiffusionPlannerData(
        src_data_dir,
        src_list_json,
        int(predicted_neighbor_num),
        eval_method="train",
        use_data_percent=100.0,
        use_agent_route_lane_order=bool(use_agent_route_lane_order),
    )
    ds.data_list = raw_list

    _WORKER_DS = ds
    _WORKER_OUT_ROOT = os.path.abspath(out_root)
    _WORKER_SHARD_SIZE = int(shard_size)
    _WORKER_BIN_BUFFER_BYTES = int(max(0, bin_buffer_bytes))


def _build_one_shard_in_worker(shard_id: int) -> Tuple[int, Set[str]]:
    """worker 프로세스에서 shard 1개를 생성하고, 키 집합을 반환합니다.

    Args:
        shard_id (int): shard 번호. shape: ()

    Returns:
        Tuple[int, Set[str]]:
            - shard_id: shape: ()
            - keys_set: shape: (K,)
    """
    global _WORKER_DS, _WORKER_OUT_ROOT, _WORKER_SHARD_SIZE, _WORKER_BIN_BUFFER_BYTES

    if _WORKER_DS is None or _WORKER_OUT_ROOT is None:
        raise RuntimeError("worker is not initialized. (_WORKER_DS/_WORKER_OUT_ROOT is None)")

    keys_set = _build_one_shard_files(
        ds=_WORKER_DS,
        out_root=_WORKER_OUT_ROOT,
        shard_id=int(shard_id),
        shard_size=int(_WORKER_SHARD_SIZE),
        bin_buffer_bytes=int(_WORKER_BIN_BUFFER_BYTES),
    )
    return int(shard_id), keys_set


def build_local_shards(
    *,
    src_data_dir: str,
    src_list_json: str,
    out_root: str,
    shard_size: int,
    world_size: int,
    seed: int,
    predicted_neighbor_num: int,
    use_agent_route_lane_order: bool,
    force_rebuild: bool,
    progress_interval_min: float,
    progress_check_every_samples: int,
    build_workers: int,
    bin_buffer_mb: int,
) -> None:
    """로컬 shard를 생성합니다(오프라인 1회).

    기존 메커니즘(느렸던 이유)
    -------------------------
    - 단일 프로세스가 모든 샘플을 순서대로 처리했습니다.
      즉, "원본 파일 읽기 → 변환 → 저장"을 1개씩만 진행했습니다.
    - 이 과정은 저장소 대기(파일 읽기/쓰기)와 CPU 작업(변환/직렬화)이 섞여 있어,
      1개 프로세스만 쓰면 CPU가 놀거나(대기) 저장소가 놀 수 있습니다.

    변경 메커니즘(빠르게 만든 핵심)
    ------------------------------
    - shard 단위(예: 2048개 묶음)로 일을 쪼개고,
      여러 프로세스가 서로 다른 shard 파일을 동시에 만들도록 했습니다.
      shard_000001.bin/idx 와 shard_000002.bin/idx 는 서로 다른 파일이라 충돌이 없습니다.
    - 추가로, idx 저장은 파이썬 루프 대신 numpy 배열을 한 번에 써서 오버헤드를 줄였습니다.
    - npz 직렬화는 BytesIO를 재사용하고, bytes 복사를 1번 줄였습니다.

    Args:
        src_data_dir:
            원본 npz 폴더. 예: /mnt/nuplan/dataset/processed. shape: ()
        src_list_json:
            원본 파일명 리스트 json. shape: ()
        out_root:
            출력 루트. 예: /workspace/local_shards_v1. shape: ()
        shard_size:
            2048. shape: ()
        world_size:
            (호환용) shard 생성에는 직접 영향이 없고 manifest에만 기록. shape: ()
        seed:
            전역 1회 섞기 시드. shape: ()
        predicted_neighbor_num:
            dataset 옵션. shape: ()
        use_agent_route_lane_order:
            dataset 옵션. shape: ()
        force_rebuild:
            True면 기존 결과가 있어도 다시 생성.
        progress_interval_min:
            진행 출력 주기(분). shape: ()
        progress_check_every_samples:
            (단일 프로세스 모드에서만 의미) 몇 샘플마다 출력 체크를 할지. shape: ()
        build_workers:
            shard 빌드 프로세스 수.
            - 0 이하: 자동값 사용
            - 1: 단일 프로세스
            - 2 이상: 병렬 빌드
        bin_buffer_mb:
            bin 파일 write buffer 크기(MB). shape: ()
    """
    out_root = os.path.abspath(out_root)
    data_dir = os.path.join(out_root, "data")
    meta_dir = os.path.join(out_root, "meta")
    manifest_path = os.path.join(meta_dir, "manifest.json")
    sources_txt_path = os.path.join(meta_dir, "source_files_shuffled.txt")

    _ensure_dir(out_root)
    _ensure_dir(data_dir)
    _ensure_dir(meta_dir)

    if os.path.isfile(manifest_path) and (not force_rebuild):
        print(f"[local_shards] manifest exists -> skip build: {manifest_path}", flush=True)
        return

    ds = DiffusionPlannerData(
        src_data_dir,
        src_list_json,
        predicted_neighbor_num,
        eval_method="train",
        use_data_percent=100.0,
        use_agent_route_lane_order=use_agent_route_lane_order,
    )

    total_count = int(len(ds))
    keep_count = _compute_keep_count(
        total_count,
        shard_size=int(shard_size),
        world_size=int(world_size),
    )
    if keep_count <= 0:
        raise ValueError(f"keep_count computed as {keep_count}. total_count={total_count}")

    raw_list: List[str] = list(ds.data_list[:keep_count])
    dropped = int(total_count - keep_count)

    rng = np.random.default_rng(int(seed))
    rng.shuffle(raw_list)

    with open(sources_txt_path, "w", encoding="utf-8") as f:
        for name in raw_list:
            f.write(str(name) + "\n")

    ds.data_list = raw_list

    shard_size_i = int(shard_size)
    shard_count = keep_count // shard_size_i
    if shard_count * shard_size_i != keep_count:
        raise ValueError("internal mismatch: shard_count*shard_size != keep_count")

    bw = int(build_workers)
    if bw <= 0:
        bw = _default_build_workers()
    bw = int(max(1, bw))

    bin_buffer_bytes = int(max(0, int(bin_buffer_mb)) * 1024 * 1024)

    print(
        f"[local_shards] total={total_count}, keep={keep_count}, dropped={dropped}, "
        f"shard_size={shard_size_i}, shard_count={shard_count}",
        flush=True,
    )
    print(
        f"[local_shards] build_workers={bw}, bin_buffer_mb={int(bin_buffer_mb)}",
        flush=True,
    )
    print(
        f"[local_shards] progress_interval_min={float(progress_interval_min):.2f} min, "
        f"progress_check_every_samples={int(progress_check_every_samples)}",
        flush=True,
    )

    interval_sec = float(max(0.1, progress_interval_min)) * 60.0
    progress = _ProgressPrinter(
        total_samples=int(keep_count),
        total_shards=int(shard_count),
        interval_sec=interval_sec,
    )
    progress.maybe_print(done_samples=0, done_shards=0, force=True)

    expected_keys_set: Set[str] = set()
    shards_meta: List[Dict[str, Any]] = []

    done_samples = 0
    done_shards = 0

    for sid in range(shard_count):
        shards_meta.append(
            {
                "shard_id": int(sid),
                "bin": os.path.join("data", f"shard_{sid:06d}.bin"),
                "idx": os.path.join("data", f"shard_{sid:06d}.idx"),
            }
        )

    if bw >= 2 and shard_count >= 2:
        try:
            ctx = mp.get_context("fork")
        except Exception:
            ctx = mp.get_context()

        shard_ids = list(range(shard_count))
        chunksize = int(max(1, shard_count // (bw * 4)))

        with ctx.Pool(
            processes=bw,
            initializer=_init_worker,
            initargs=(
                str(src_data_dir),
                str(src_list_json),
                int(predicted_neighbor_num),
                bool(use_agent_route_lane_order),
                raw_list,
                int(shard_size_i),
                str(out_root),
                int(bin_buffer_bytes),
            ),
        ) as pool:
            for _, keys in pool.imap_unordered(_build_one_shard_in_worker, shard_ids, chunksize=chunksize):
                expected_keys_set.update(keys)

                done_shards += 1
                done_samples += shard_size_i

                progress.maybe_print(
                    done_samples=int(done_samples),
                    done_shards=int(done_shards),
                    force=False,
                )
    else:
        check_every = int(max(1, progress_check_every_samples))
        for sid in range(shard_count):
            base = sid * shard_size_i

            idx_table = np.zeros((shard_size_i, 2), dtype=np.uint64)
            offset = 0

            bin_rel = os.path.join("data", f"shard_{sid:06d}.bin")
            idx_rel = os.path.join("data", f"shard_{sid:06d}.idx")
            bin_path = os.path.join(out_root, bin_rel)
            idx_path = os.path.join(out_root, idx_rel)

            buff = io.BytesIO()

            with open(bin_path, "wb", buffering=int(max(0, bin_buffer_bytes))) as bin_f:
                for j in range(shard_size_i):
                    global_idx = base + j
                    sample = ds[global_idx]

                    expected_keys_set.update(sample.keys())

                    view = _serialize_sample_to_npz_buffer(sample, buff)
                    try:
                        length = int(view.nbytes)
                        bin_f.write(view)
                    finally:
                        view.release()

                    idx_table[j, 0] = np.uint64(offset)
                    idx_table[j, 1] = np.uint64(length)
                    offset += length

                    done_samples += 1
                    if (done_samples % check_every) == 0:
                        progress.maybe_print(
                            done_samples=int(done_samples),
                            done_shards=int(done_shards),
                            force=False,
                        )

            _write_idx_table_fast(idx_path, idx_table)

            done_shards += 1
            progress.maybe_print(
                done_samples=int(done_samples),
                done_shards=int(done_shards),
                force=False,
            )

    expected_keys = sorted(list(expected_keys_set))

    manifest: Dict[str, Any] = {
        "format": _LOCAL_SHARD_FORMAT,
        "eval_method": "train",
        "created_at_unix": int(time.time()),
        "out_root": out_root,
        "source_data_dir": str(src_data_dir),
        "source_list_json": str(src_list_json),
        "source_total_count": int(total_count),
        "num_samples": int(keep_count),
        "dropped_samples": int(dropped),
        "shard_size": int(shard_size_i),
        "shard_count": int(shard_count),
        "world_size": int(world_size),
        "world_size_hint_only": True,
        "seed_global_shuffle_once": int(seed),
        "predicted_neighbor_num": int(predicted_neighbor_num),
        "use_agent_route_lane_order": bool(use_agent_route_lane_order),
        "expected_keys": expected_keys,
        "sources_txt": os.path.relpath(sources_txt_path, out_root),
        "build_workers": int(bw),
        "bin_buffer_mb": int(bin_buffer_mb),
        "shards": shards_meta,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    progress.maybe_print(
        done_samples=int(done_samples),
        done_shards=int(done_shards),
        force=True,
    )

    print(f"[local_shards] DONE. manifest={manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser("build_local_shards")
    parser.add_argument("--src_data_dir", type=str, required=True)
    parser.add_argument("--src_list_json", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)

    parser.add_argument("--shard_size", type=int, default=2048)
    parser.add_argument("--world_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3407)

    parser.add_argument("--predicted_neighbor_num", type=int, default=448)
    parser.add_argument("--use_agent_route_lane_order", type=_bool_arg, default=False)

    parser.add_argument("--force_rebuild", type=_bool_arg, default=False)

    parser.add_argument(
        "--progress_interval_min",
        type=float,
        default=5.0,
        help="몇 분마다 진행 상황(경과/예상 남은 시간/%)을 출력할지",
    )
    parser.add_argument(
        "--progress_check_every_samples",
        type=int,
        default=256,
        help="(단일 프로세스) 몇 샘플마다 '출력할 시간인지'만 확인할지(오버헤드 줄이기용)",
    )

    parser.add_argument(
        "--build_workers",
        type=int,
        default=0,
        help="shard 생성에 쓸 프로세스 수. 0 이하이면 자동.",
    )
    parser.add_argument(
        "--bin_buffer_mb",
        type=int,
        default=8,
        help="bin 파일 write buffer 크기(MB).",
    )

    args = parser.parse_args()

    build_local_shards(
        src_data_dir=args.src_data_dir,
        src_list_json=args.src_list_json,
        out_root=args.out_root,
        shard_size=int(args.shard_size),
        world_size=int(args.world_size),
        seed=int(args.seed),
        predicted_neighbor_num=int(args.predicted_neighbor_num),
        use_agent_route_lane_order=bool(args.use_agent_route_lane_order),
        force_rebuild=bool(args.force_rebuild),
        progress_interval_min=float(args.progress_interval_min),
        progress_check_every_samples=int(args.progress_check_every_samples),
        build_workers=int(args.build_workers),
        bin_buffer_mb=int(args.bin_buffer_mb),
    )


if __name__ == "__main__":
    main()
