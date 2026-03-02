#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import struct
import time
import zipfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from typing import BinaryIO, Iterator, List, Tuple


# -----------------------------
# Config
# -----------------------------
ROOT_DIR_DEFAULT = "/workspace/local_shards_v_world"

BIN_WIDTH = 25
LANE_MAX = 300
SAFETY_MAX = 200

LANES_MEMBER = "lanes.npy"
ROAD_EDGE_MEMBER = "road_edge.npy"
SAFETY_MEMBERS = [
    "stop_sign_points.npy",
    "crosswalk_points.npy",
    "speed_bump_points.npy",
    "driveway_points.npy",
]


# -----------------------------
# File listing (depth=1)
# -----------------------------
def iter_npz_paths(root_dir: str) -> Iterator[str]:
    """Yield .npz file paths under root_dir (depth=1 only)."""
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".npz"):
                yield entry.path


def count_npz_files(root_dir: str) -> int:
    """Count .npz files under root_dir (depth=1 only)."""
    return sum(1 for _ in iter_npz_paths(root_dir))


def split_into_chunks(paths_iter: Iterator[str], chunk_size: int) -> Iterator[List[str]]:
    """Group paths into lists of length chunk_size."""
    chunk: List[str] = []
    for p in paths_iter:
        chunk.append(p)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# -----------------------------
# Fast "first dimension" read from NPY inside NPZ
# -----------------------------
def _parse_first_dim_from_npy_meta(meta_bytes: bytes) -> int:
    """Parse the first dimension from the NPY header 'shape' field.

    Returns:
        First dimension (non-negative int). If shape is (), returns 1.
        If parsing fails, returns 0.
    """
    idx = meta_bytes.find(b"'shape':")
    if idx == -1:
        idx = meta_bytes.find(b'"shape":')
    if idx == -1:
        return 0

    lpar = meta_bytes.find(b"(", idx)
    if lpar == -1:
        return 0

    i = lpar + 1
    n = len(meta_bytes)

    # Skip spaces/tabs
    while i < n and meta_bytes[i] in (32, 9):  # space, tab
        i += 1

    # shape == ()
    if i < n and meta_bytes[i] == 41:  # ')'
        return 1

    # Read first integer in shape tuple
    sign = 1
    if i < n and meta_bytes[i] == 45:  # '-'
        sign = -1
        i += 1

    num = 0
    found_digit = False
    while i < n:
        c = meta_bytes[i]
        if 48 <= c <= 57:  # '0'..'9'
            num = num * 10 + (c - 48)
            found_digit = True
            i += 1
        else:
            break

    if not found_digit:
        return 0

    val = sign * num
    return val if val >= 0 else 0


def read_npy_first_dim(fp: BinaryIO) -> int:
    """Read only the first dimension (shape[0]) from an NPY stream (no data load)."""
    magic = fp.read(6)
    if magic != b"\x93NUMPY":
        return 0

    ver = fp.read(2)
    if len(ver) != 2:
        return 0

    major = ver[0]

    if major == 1:
        hlen_bytes = fp.read(2)
        if len(hlen_bytes) != 2:
            return 0
        header_len = struct.unpack("<H", hlen_bytes)[0]
    else:
        hlen_bytes = fp.read(4)
        if len(hlen_bytes) != 4:
            return 0
        header_len = struct.unpack("<I", hlen_bytes)[0]

    meta = fp.read(header_len)
    if len(meta) != header_len:
        return 0

    return _parse_first_dim_from_npy_meta(meta)


def first_dim_from_npz(zf: zipfile.ZipFile, member_name: str) -> int:
    """Read shape[0] for a given member_name (e.g., 'lanes.npy') inside an NPZ."""
    try:
        with zf.open(member_name, "r") as f:
            return read_npy_first_dim(f)
    except KeyError:
        return 0


# -----------------------------
# Histogram helpers
# -----------------------------
def bin_index(count: int, bin_width: int, max_in_last_bin: int) -> int:
    """Map a non-negative count to a histogram bin index.

    Bins:
      - [0, 25), [25, 50), ...
      - The last regular bin includes the max value: [max-bin_width, max]
      - Values > max go into an overflow bin.
    """
    if count < 0:
        count = 0

    regular_bins = max_in_last_bin // bin_width
    overflow_index = regular_bins
    last_regular = regular_bins - 1

    if count > max_in_last_bin:
        return overflow_index
    if count == max_in_last_bin:
        return last_regular

    idx = count // bin_width
    if idx > last_regular:
        idx = last_regular
    return idx


def merge_hist(dst: List[int], src: List[int]) -> None:
    """Accumulate histogram counts in-place."""
    for i in range(len(dst)):
        dst[i] += src[i]


def format_histogram_lines(hist: List[int], bin_width: int, max_in_last_bin: int, total: int) -> str:
    """Format histogram as lines: 'bin label : count (percent%)'."""
    regular_bins = max_in_last_bin // bin_width
    last_regular = regular_bins - 1

    lines: List[str] = []
    for i, cnt in enumerate(hist):
        pct = (cnt / total * 100.0) if total > 0 else 0.0

        if i < last_regular:
            start = i * bin_width
            end = start + bin_width
            label = f">= {start} and < {end}"
        elif i == last_regular:
            start = i * bin_width
            end = max_in_last_bin
            label = f">= {start} and <= {end}"
        else:
            label = f">= {max_in_last_bin + 1}"

        lines.append(f"{label:>18} : {cnt:>10}  ({pct:6.2f}%)")

    return "\n".join(lines)


# -----------------------------
# Worker: process a chunk of NPZ files
# -----------------------------
def process_npz_chunk(paths: List[str]) -> Tuple[List[int], List[int], List[int], int, int]:
    """Process a list of NPZ files and return partial histograms.

    For each file, we read only shape[0] for:
      - lanes.npy
      - road_edge.npy
      - safety_sum = stop_sign_points + crosswalk_points + speed_bump_points + driveway_points
    """
    lanes_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)
    safety_hist = [0] * (SAFETY_MAX // BIN_WIDTH + 1)
    road_edge_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)

    processed = 0
    errors = 0

    for p in paths:
        try:
            with zipfile.ZipFile(p, "r") as zf:
                lanes_cnt = first_dim_from_npz(zf, LANES_MEMBER)
                road_edge_cnt = first_dim_from_npz(zf, ROAD_EDGE_MEMBER)

                safety_sum = 0
                for mem in SAFETY_MEMBERS:
                    safety_sum += first_dim_from_npz(zf, mem)

            lanes_hist[bin_index(lanes_cnt, BIN_WIDTH, LANE_MAX)] += 1
            road_edge_hist[bin_index(road_edge_cnt, BIN_WIDTH, LANE_MAX)] += 1
            safety_hist[bin_index(safety_sum, BIN_WIDTH, SAFETY_MAX)] += 1

            processed += 1
        except Exception:
            errors += 1

    return lanes_hist, safety_hist, road_edge_hist, processed, errors


# -----------------------------
# Main runner
# -----------------------------
def run(
    root_dir: str,
    workers: int,
    chunk_size: int,
    report_interval_sec: int,
    max_in_flight: int,
) -> None:
    """Run histogram stats over all NPZ files under root_dir (depth=1)."""
    print(f"[1/2] Counting .npz files in: {root_dir}", flush=True)
    total = count_npz_files(root_dir)
    print(f"[2/2] Total .npz files: {total}", flush=True)

    lanes_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)
    safety_hist = [0] * (SAFETY_MAX // BIN_WIDTH + 1)
    road_edge_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)

    processed = 0
    errors = 0

    start_ts = time.monotonic()
    last_report_ts = start_ts

    chunk_iter = split_into_chunks(iter_npz_paths(root_dir), chunk_size)

    in_flight = set()

    def _submit_next(ex: ProcessPoolExecutor) -> bool:
        """Submit the next chunk to the executor. Return False if no more chunks."""
        nonlocal in_flight
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return False
        in_flight.add(ex.submit(process_npz_chunk, chunk))
        return True

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # Fill initial pipeline (bounded to avoid large memory usage)
        for _ in range(max_in_flight):
            if not _submit_next(ex):
                break

        while in_flight:
            done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for fut in done:
                lh, sh, eh, p_cnt, e_cnt = fut.result()
                merge_hist(lanes_hist, lh)
                merge_hist(safety_hist, sh)
                merge_hist(road_edge_hist, eh)
                processed += p_cnt
                errors += e_cnt

                # Keep pipeline full
                _submit_next(ex)

            now = time.monotonic()
            if now - last_report_ts >= report_interval_sec:
                elapsed = now - start_ts
                speed = processed / elapsed if elapsed > 0 else 0.0
                remaining = total - processed
                eta_sec = (remaining / speed) if speed > 0 else None

                elapsed_min = elapsed / 60.0
                pct = (processed / total * 100.0) if total > 0 else 0.0

                if eta_sec is None:
                    print(
                        f"[Progress] {processed}/{total} ({pct:.2f}%), "
                        f"elapsed {elapsed_min:.1f} min, ETA computing, errors {errors}",
                        flush=True,
                    )
                else:
                    eta_min = eta_sec / 60.0
                    print(
                        f"[Progress] {processed}/{total} ({pct:.2f}%), "
                        f"elapsed {elapsed_min:.1f} min, ETA {eta_min:.1f} min, errors {errors}",
                        flush=True,
                    )

                last_report_ts = now

    total_elapsed = time.monotonic() - start_ts
    print("\n==== Final Results ====", flush=True)
    print(
        f"Success: {processed} / Total: {total} / Errors: {errors} / "
        f"Elapsed: {total_elapsed / 60.0:.1f} min",
        flush=True,
    )

    print("\n[1] lanes count histogram", flush=True)
    print(format_histogram_lines(lanes_hist, BIN_WIDTH, LANE_MAX, processed), flush=True)

    print("\n[2] road-safety total (4 keys) count histogram", flush=True)
    print(format_histogram_lines(safety_hist, BIN_WIDTH, SAFETY_MAX, processed), flush=True)

    print("\n[3] road_edge count histogram", flush=True)
    print(format_histogram_lines(road_edge_hist, BIN_WIDTH, LANE_MAX, processed), flush=True)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default=ROOT_DIR_DEFAULT)
    p.add_argument("--workers", type=int, default=(os.cpu_count() or 1))
    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--report_interval_sec", type=int, default=180)  # 3 minutes
    p.add_argument("--max_in_flight", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_in_flight = args.max_in_flight
    if max_in_flight <= 0:
        # A good default is ~3x workers (keeps pipeline full without holding too many paths in memory).
        max_in_flight = max(2, args.workers * 3)

    run(
        root_dir=args.root_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        report_interval_sec=args.report_interval_sec,
        max_in_flight=max_in_flight,
    )