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
# 설정(요구사항 기준)
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
# 파일 목록 수집
# -----------------------------
def iter_npz_paths(root_dir: str) -> Iterator[str]:
    """지정한 폴더(한 단계)에서 .npz 파일 경로를 차례대로 내보냅니다.

    Args:
        root_dir: .npz 파일들이 들어있는 폴더 경로입니다. 하위 폴더는 보지 않습니다(depth=1).

    Yields:
        .npz 파일의 전체 경로 문자열입니다.
    """
    with os.scandir(root_dir) as it:
        for entry in it:
            if entry.is_file() and entry.name.endswith(".npz"):
                yield entry.path


def count_npz_files(root_dir: str) -> int:
    """폴더(한 단계) 안의 .npz 파일 개수를 셉니다.

    진행률(%)과 남은 시간(추정)을 정확히 계산하려면 전체 파일 수가 필요해서,
    실제 처리 전에 한 번 빠르게 세어둡니다.

    Args:
        root_dir: .npz 파일들이 들어있는 폴더 경로입니다(하위 폴더는 보지 않음).

    Returns:
        .npz 파일 개수입니다.
    """
    return sum(1 for _ in iter_npz_paths(root_dir))


def split_into_chunks(paths_iter: Iterator[str], chunk_size: int) -> Iterator[List[str]]:
    """경로들을 chunk_size 단위 리스트로 묶어서 내보냅니다.

    아주 많은 파일을 처리할 때, 파일 하나당 작업을 나누면 작업 배분/전달 비용이 커질 수 있어서
    여러 파일을 묶어서 한 번에 처리하도록 합니다.

    Args:
        paths_iter: .npz 파일 경로를 내보내는 반복자입니다.
        chunk_size: 한 작업(chunk)에 담을 파일 개수입니다.

    Yields:
        .npz 파일 경로 리스트입니다.
    """
    chunk: List[str] = []
    for p in paths_iter:
        chunk.append(p)
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


# -----------------------------
# NPZ 내부의 NPY "개수(첫 축 길이)"만 빠르게 읽기
# -----------------------------
def _parse_first_dim_from_npy_meta(meta_bytes: bytes) -> int:
    """NPY 파일의 '크기 정보' 바이트에서 첫 번째 길이(개수)만 뽑아냅니다.

    이 함수는 실제 좌표/값들을 읽지 않습니다.
    예를 들어 아래처럼 저장되어 있다면,
      - lanes: (lane_num, ...)  -> lane_num
      - road_edge: (num, ...)   -> num
      - stop_sign_points: (num, ...) -> num
    여기서 우리가 원하는 값은 항상 shape의 첫 번째 값입니다.

    Args:
        meta_bytes: NPY 파일 앞부분에 들어있는 "크기 정보" 바이트입니다.

    Returns:
        shape의 첫 번째 값(개수)입니다.
        shape가 비어있는 경우(예: ())는 1로 취급합니다.
        파싱 실패 시 0을 반환합니다.
    """
    # 보통 numpy가 만드는 문자열은 "'shape': ( ... )" 형태를 포함합니다.
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

    # 공백/탭 건너뛰기
    while i < n and meta_bytes[i] in (32, 9):  # space, tab
        i += 1

    # shape == () 인 경우
    if i < n and meta_bytes[i] == 41:  # ')'
        return 1

    # 첫 숫자만 읽기(첫 번째 축)
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
    """NPZ 안의 .npy 파일에서 '첫 번째 길이(개수)'만 빠르게 읽습니다.

    핵심은 "데이터 전체를 읽지 않고", 파일 맨 앞쪽의 "크기 정보"만 읽는 것입니다.

    Args:
        fp: zipfile.ZipFile.open(...)으로 얻은 파일 객체(바이너리 읽기)입니다.

    Returns:
        배열 shape의 첫 번째 값(개수)입니다.
        예: lanes가 (lane_num, ...)이면 lane_num을 반환합니다.
        읽기/파싱에 실패하면 0을 반환합니다.
    """
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
        # 2.x / 3.x는 4바이트 길이를 사용합니다.
        hlen_bytes = fp.read(4)
        if len(hlen_bytes) != 4:
            return 0
        header_len = struct.unpack("<I", hlen_bytes)[0]

    meta = fp.read(header_len)
    if len(meta) != header_len:
        return 0

    return _parse_first_dim_from_npy_meta(meta)


def first_dim_from_npz(zf: zipfile.ZipFile, member_name: str) -> int:
    """NPZ(zip) 안에서 특정 멤버(.npy)의 첫 번째 길이(개수)를 읽습니다.

    Args:
        zf: 열린 zipfile.ZipFile 객체입니다.
        member_name: 예) "lanes.npy", "road_edge.npy" 같은 멤버 이름입니다.

    Returns:
        해당 배열의 첫 번째 길이(개수)입니다.
        멤버가 없으면 0을 반환합니다.
    """
    try:
        with zf.open(member_name, "r") as f:
            return read_npy_first_dim(f)
    except KeyError:
        return 0


# -----------------------------
# 히스토그램 누적
# -----------------------------
def bin_index(count: int, bin_width: int, max_in_last_bin: int) -> int:
    """개수를 히스토그램 구간 인덱스로 바꿉니다.

    구간 규칙:
      - 0 이상 25 미만, 25 이상 50 미만, ...
      - 마지막 구간은 "최대값 이하"까지 포함합니다.
        예: 275 이상 300 이하
      - 최대값을 넘으면 마지막에 "초과" 구간으로 모읍니다.
        예: 301 이상

    Args:
        count: 개수(예: lanes 개수)입니다.
        bin_width: 구간 폭(여기서는 25)입니다.
        max_in_last_bin: 마지막 구간의 최대값(예: 300 또는 200)입니다.

    Returns:
        히스토그램 리스트에서의 인덱스입니다.
    """
    if count < 0:
        count = 0

    regular_bins = max_in_last_bin // bin_width  # 마지막(초과) 구간 제외한 구간 수가 아니라, "초과 구간 인덱스"
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
    """히스토그램 카운트를 더합니다.

    Args:
        dst: 누적 대상 리스트입니다(길이는 고정).
        src: 더할 리스트입니다(길이는 dst와 동일).
    """
    for i in range(len(dst)):
        dst[i] += src[i]


def format_histogram_lines(hist: List[int], bin_width: int, max_in_last_bin: int, total: int) -> str:
    """히스토그램을 '구간별 개수 + 비율(%)' 텍스트로 만듭니다.

    Args:
        hist: 구간별 개수 리스트입니다.
        bin_width: 구간 폭(25)입니다.
        max_in_last_bin: 마지막 구간의 최대값(예: 300 또는 200)입니다.
        total: 전체 파일 수(비율 계산용)입니다.

    Returns:
        출력용 문자열입니다.
    """
    regular_bins = max_in_last_bin // bin_width
    last_regular = regular_bins - 1

    lines: List[str] = []
    for i, cnt in enumerate(hist):
        pct = (cnt / total * 100.0) if total > 0 else 0.0

        if i < last_regular:
            start = i * bin_width
            end = start + bin_width
            label = f"{start} 이상 {end} 미만"
        elif i == last_regular:
            start = i * bin_width
            end = max_in_last_bin
            label = f"{start} 이상 {end} 이하"
        else:
            label = f"{max_in_last_bin + 1} 이상"

        lines.append(f"{label:>14} : {cnt:>10}  ({pct:6.2f}%)")

    return "\n".join(lines)


# -----------------------------
# 병렬 처리(가장 중요한 부분)
# -----------------------------
def process_npz_chunk(paths: List[str]) -> Tuple[List[int], List[int], List[int], int, int]:
    """여러 개의 .npz 파일을 묶어서 통계를 계산합니다(워커 프로세스에서 실행).

    각 파일마다 아래 "개수"만 읽습니다:
      - lanes: (lane_num, ...) -> lane_num
      - road_edge: (num, ...) -> num
      - stop_sign_points / crosswalk_points / speed_bump_points / driveway_points:
        각각 (num, ...) -> num, 이 4개를 더한 값을 "road safety 합"으로 사용

    Args:
        paths: .npz 파일 경로 리스트입니다.

    Returns:
        (lanes_hist, safety_hist, road_edge_hist, processed_cnt, error_cnt)
        - *_hist: 구간별 개수 리스트
        - processed_cnt: 성공 처리한 파일 수
        - error_cnt: 열기/파싱 실패한 파일 수
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


def run(
    root_dir: str,
    workers: int,
    chunk_size: int,
    report_interval_sec: int,
    max_in_flight: int,
) -> None:
    """전체 실행 함수입니다.

    Args:
        root_dir: .npz 파일이 있는 폴더 경로(depth=1).
        workers: 동시에 사용할 CPU 작업 수입니다.
        chunk_size: 한 번에 묶어서 처리할 파일 개수입니다.
        report_interval_sec: 진행 상황 출력 간격(초)입니다. 기본 180초(3분).
        max_in_flight: 동시에 돌리고 있는 작업(미완료 chunk)의 최대 개수입니다.
    """
    print(f"[1/2] 전체 .npz 파일 수 세는 중: {root_dir}", flush=True)
    total = count_npz_files(root_dir)
    print(f"[2/2] 전체 .npz 파일 수: {total}", flush=True)

    lanes_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)
    safety_hist = [0] * (SAFETY_MAX // BIN_WIDTH + 1)
    road_edge_hist = [0] * (LANE_MAX // BIN_WIDTH + 1)

    processed = 0
    errors = 0

    start_ts = time.monotonic()
    last_report_ts = start_ts

    chunk_iter = split_into_chunks(iter_npz_paths(root_dir), chunk_size)

    def _submit_next(ex: ProcessPoolExecutor) -> bool:
        """다음 chunk를 하나 제출합니다. 더 이상 없으면 False."""
        nonlocal in_flight
        try:
            chunk = next(chunk_iter)
        except StopIteration:
            return False
        in_flight.add(ex.submit(process_npz_chunk, chunk))
        return True

    in_flight = set()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        # 처음에 일정 개수만큼만 작업을 채워 넣습니다(메모리 폭증 방지).
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

                # 다음 chunk를 하나 더 제출해서 "동시에 도는 작업 수"를 유지합니다.
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
                        f"[진행] {processed}/{total} ({pct:.2f}%), 경과 {elapsed_min:.1f}분, 남은 시간 계산중, 에러 {errors}",
                        flush=True,
                    )
                else:
                    eta_min = eta_sec / 60.0
                    print(
                        f"[진행] {processed}/{total} ({pct:.2f}%), 경과 {elapsed_min:.1f}분, 남은 {eta_min:.1f}분(추정), 에러 {errors}",
                        flush=True,
                    )

                last_report_ts = now

    total_elapsed = time.monotonic() - start_ts
    print("\n==== 최종 결과 ====", flush=True)
    print(
        f"처리 성공: {processed} / 전체: {total} / 에러: {errors} / 총 경과: {total_elapsed / 60.0:.1f}분",
        flush=True,
    )

    print("\n[1] lanes 개수 히스토그램", flush=True)
    print(format_histogram_lines(lanes_hist, BIN_WIDTH, LANE_MAX, processed), flush=True)

    print("\n[2] road safety 합(4개 key) 개수 히스토그램", flush=True)
    print(format_histogram_lines(safety_hist, BIN_WIDTH, SAFETY_MAX, processed), flush=True)

    print("\n[3] road_edge 개수 히스토그램", flush=True)
    print(format_histogram_lines(road_edge_hist, BIN_WIDTH, LANE_MAX, processed), flush=True)


def parse_args() -> argparse.Namespace:
    """명령줄 인자를 파싱합니다."""
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", type=str, default=ROOT_DIR_DEFAULT)
    p.add_argument("--workers", type=int, default=(os.cpu_count() or 1))
    p.add_argument("--chunk_size", type=int, default=500)
    p.add_argument("--report_interval_sec", type=int, default=180)  # 3분
    p.add_argument("--max_in_flight", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    max_in_flight = args.max_in_flight
    if max_in_flight <= 0:
        # 너무 많이 쌓아두면 경로 문자열이 메모리에 많이 남습니다.
        # 보통 workers의 2~4배 정도면 충분합니다.
        max_in_flight = max(2, args.workers * 3)

    run(
        root_dir=args.root_dir,
        workers=args.workers,
        chunk_size=args.chunk_size,
        report_interval_sec=args.report_interval_sec,
        max_in_flight=max_in_flight,
    )