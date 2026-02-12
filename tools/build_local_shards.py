from __future__ import annotations

import argparse
import io
import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from diffusion_planner.utils.dataset import DiffusionPlannerData


_LOCAL_SHARD_FORMAT = "dp_local_shards_v1"


def _bool_arg(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if not isinstance(v, str):
        return False
    s = v.strip().lower()
    return s in ("1", "true", "t", "yes", "y")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _is_overlay_like(path: str) -> Optional[bool]:
    """df 없이도 '대략' 판단하려고 시도합니다(정확 판별은 어려움)."""
    try:
        st = os.statvfs(path)
        _ = st.f_frsize * st.f_bavail
        return None
    except Exception:
        return None


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

    출력 내용(요구사항 반영):
        - 예상 총 소요 시간(시/분)
        - 지금까지 경과 시간(시/분)
        - 진행률(%)
        - (추가) 샘플/샤드 기준 진행량

    Note:
        - ETA는 "지금까지 평균 속도" 기반이라 초반에는 흔들릴 수 있습니다.
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

        # 샘플/초는 참고용(요구사항에는 없지만 튜닝에 도움)
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

    # 문자열 / 숫자 스칼라
    arr = np.asarray(value)
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "O":
        raise ValueError(f"[shard] object dtype is not allowed. key='{key}'")
    return arr


def _serialize_sample_to_blob(sample: Dict[str, Any]) -> bytes:
    """sample dict를 '압축 없는 npz 바이트'로 직렬화합니다."""
    save_dict: Dict[str, np.ndarray] = {}
    for k, v in sample.items():
        arr = _to_save_array(v, key=str(k))
        if arr is None:
            continue
        save_dict[str(k)] = arr

    buff = io.BytesIO()
    # ✅ 압축 없음: np.savez
    np.savez(buff, **save_dict)
    return buff.getvalue()


def _write_idx_table(idx_path: str, entries: List[Tuple[int, int]]) -> None:
    """(offset,length) 목록을 idx 파일로 저장합니다.

    저장 포맷
    --------
    - 각 엔트리: uint64 offset + uint64 length (16바이트)
    - 총 엔트리 수 = shard_size
    """
    with open(idx_path, "wb") as f:
        for offset, length in entries:
            f.write(struct.pack("<QQ", int(offset), int(length)))

def _compute_keep_count(
    total_count: int,
    *,
    shard_size: int,
    world_size: int,
) -> int:
    """shard_size에 딱 맞게 떨어지도록 keep_count를 계산합니다.

    변경 의도
    --------
    - 기존에는 (shard_size * world_size) 배수로 맞춰 잘랐습니다.
      → 그래서 shard를 만들 때 쓰던 GPU 개수와 학습 때 GPU 개수가 달라지면,
        shard_count가 나눠떨어지지 않아 학습이 막힐 수 있었습니다.
    - 이제는 shard를 "GPU 개수와 무관"하게 만들기 위해
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
) -> None:
    """로컬 shard를 생성합니다(오프라인 1회).

    처리 순서(설계도 그대로)
    ----------------------
    1) 원본 json 리스트 로드
    2) keep_count를 계산해 drop_last와 동치로 자름
    3) 전역 1회 shuffle
    4) 2048개씩 shard를 만들며 .bin/.idx 생성
    5) manifest.json 저장

    추가 기능
    --------
    - 매 N분마다 진행 상황(경과/예상 남은 시간/예상 총 시간/진행률)을 출력합니다.

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
            6. shape: ()
        seed:
            전역 1회 섞기 시드. shape: ()
        predicted_neighbor_num:
            448. shape: ()
        use_agent_route_lane_order:
            dataset 옵션. shape: ()
        force_rebuild:
            True면 기존 결과가 있어도 다시 생성.
        progress_interval_min:
            진행 출력 주기(분). shape: ()
        progress_check_every_samples:
            몇 샘플마다 "출력할 시간인지"만 확인할지. shape: ()
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

    # 1) 기존 DiffusionPlannerData 로직을 그대로 재사용해서 sample dict를 만든다.
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

    # drop_last와 동치: 뒤쪽 remainder는 쓰지 않음
    raw_list: List[str] = list(ds.data_list[:keep_count])  # length=keep_count
    dropped = int(total_count - keep_count)

    # 2) 전역 1회 섞기
    rng = np.random.default_rng(int(seed))
    rng.shuffle(raw_list)

    # 디버깅용: shard 구성 원본 파일 리스트(섞인 순서)를 저장
    with open(sources_txt_path, "w", encoding="utf-8") as f:
        for name in raw_list:
            f.write(str(name) + "\n")

    # 3) dataset에 그대로 주입해서 __getitem__이 "같은 로직"으로 동작하도록 함
    ds.data_list = raw_list

    shard_size_i = int(shard_size)
    shard_count = keep_count // shard_size_i
    if shard_count * shard_size_i != keep_count:
        raise ValueError("internal mismatch: shard_count*shard_size != keep_count")

    print(
        f"[local_shards] total={total_count}, keep={keep_count}, dropped={dropped}, "
        f"shard_size={shard_size_i}, shard_count={shard_count}",
        flush=True,
    )
    print(
        f"[local_shards] progress_interval_min={float(progress_interval_min):.2f} min, "
        f"progress_check_every_samples={int(progress_check_every_samples)}",
        flush=True,
    )

    # 진행 출력기 준비
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
    check_every = int(max(1, progress_check_every_samples))

    for sid in range(shard_count):
        bin_rel = os.path.join("data", f"shard_{sid:06d}.bin")
        idx_rel = os.path.join("data", f"shard_{sid:06d}.idx")
        bin_path = os.path.join(out_root, bin_rel)
        idx_path = os.path.join(out_root, idx_rel)

        entries: List[Tuple[int, int]] = []
        offset = 0

        with open(bin_path, "wb") as bin_f:
            base = sid * shard_size_i
            for j in range(shard_size_i):
                global_idx = base + j
                sample = ds[global_idx]  # 기존 dataset.py 로직 그대로

                expected_keys_set.update(sample.keys())

                blob = _serialize_sample_to_blob(sample)
                length = len(blob)

                bin_f.write(blob)
                entries.append((offset, length))
                offset += length

                done_samples += 1
                if (done_samples % check_every) == 0:
                    progress.maybe_print(
                        done_samples=int(done_samples),
                        done_shards=int(done_shards),
                        force=False,
                    )

        _write_idx_table(idx_path, entries)

        shards_meta.append(
            {
                "shard_id": int(sid),
                "bin": bin_rel,
                "idx": idx_rel,
            }
        )

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
        "shards": shards_meta,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

    # 마지막 한 번 강제 출력(100% 근처)
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

    # ✅ 추가: 진행 출력 옵션
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
        help="몇 샘플마다 '출력할 시간인지'만 확인할지(오버헤드 줄이기용)",
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
    )


if __name__ == "__main__":
    main()
