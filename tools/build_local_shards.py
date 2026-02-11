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
    """world_size와 shard_size에 딱 맞게 떨어지도록 keep_count를 계산합니다.

    목표
    ----
    - shard_count = keep_count / shard_size 가 정수
    - shard_count 가 world_size 로 나누어 떨어짐 (rank당 shard 수 동일)

    Args:
        total_count: 원본 파일 개수. shape: ()
        shard_size: 2048. shape: ()
        world_size: 6. shape: ()

    Returns:
        keep_count: shape: ()
    """
    total = int(max(0, total_count))
    unit = int(shard_size) * int(max(1, world_size))
    if unit <= 0:
        return 0
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
) -> None:
    """로컬 shard를 생성합니다(오프라인 1회).

    처리 순서(설계도 그대로)
    ----------------------
    1) 원본 json 리스트 로드
    2) keep_count를 계산해 drop_last와 동치로 자름
    3) 전역 1회 shuffle
    4) 2048개씩 shard를 만들며 .bin/.idx 생성
    5) manifest.json 저장

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

    if (shard_count % int(world_size)) != 0:
        raise ValueError(
            f"shard_count must be divisible by world_size. shard_count={shard_count}, world_size={world_size}"
        )

    print(
        f"[local_shards] total={total_count}, keep={keep_count}, dropped={dropped}, "
        f"shard_size={shard_size_i}, shard_count={shard_count}",
        flush=True,
    )

    expected_keys_set: Set[str] = set()

    shards_meta: List[Dict[str, Any]] = []

    t0 = time.perf_counter()
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

        _write_idx_table(idx_path, entries)

        shards_meta.append(
            {
                "shard_id": int(sid),
                "bin": bin_rel,
                "idx": idx_rel,
            }
        )

        if (sid + 1) % 5 == 0 or (sid + 1) == shard_count:
            elapsed = time.perf_counter() - t0
            print(f"[local_shards] built shard {sid+1}/{shard_count} (elapsed={elapsed:.1f}s)", flush=True)

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
        "seed_global_shuffle_once": int(seed),
        "predicted_neighbor_num": int(predicted_neighbor_num),
        "use_agent_route_lane_order": bool(use_agent_route_lane_order),
        "expected_keys": expected_keys,
        "sources_txt": os.path.relpath(sources_txt_path, out_root),
        "shards": shards_meta,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)

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
    )


if __name__ == "__main__":
    main()
