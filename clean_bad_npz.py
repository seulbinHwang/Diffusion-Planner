#!/usr/bin/env python3
"""
clean_bad_npz.py

Scans .npz files in a dataset directory to find unloadable (bad) files,
shows progress with tqdm, uses multiprocessing for speed,
then optionally deletes the bad files and updates the JSON list.
"""

import os
import json
import argparse
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# 워커 프로세스에서 사용할 전역 데이터 디렉토리
DATA_DIR = None

def init_worker(data_dir):
    """워커 프로세스 초기화 시 DATA_DIR 설정"""
    global DATA_DIR
    DATA_DIR = data_dir

def is_bad_npz(rel_path):
    """
    DATA_DIR/rel_path 경로의 .npz 파일을 검사
    문제가 있으면 rel_path, 정상 파일이면 None 반환
    """
    full_path = os.path.join(DATA_DIR, rel_path)
    if not os.path.isfile(full_path):
        return rel_path
    try:
        with open(full_path, 'rb') as f:
            np.load(f, allow_pickle=True)
        return None
    except Exception:
        return rel_path

def main():
    parser = argparse.ArgumentParser(
        description="Scan and clean bad .npz files with a progress bar."
    )
    parser.add_argument('--data_dir',
                        required=True,
                        help="디렉토리 경로 (.npz 파일들이 있는 곳)")
    parser.add_argument('--data_list',
                        required=True,
                        help="상대 경로로 된 .npz 파일 리스트(JSON)")
    args = parser.parse_args()

    # JSON에서 상대 경로 리스트 읽기
    with open(args.data_list, 'r') as f:
        rel_paths = json.load(f)

    total = len(rel_paths)
    num_workers = max(1, cpu_count() - 1)
    print(f"Scanning {total} files using {num_workers} workers…")

    # 멀티프로세싱 풀 생성
    with Pool(initializer=init_worker,
              initargs=(args.data_dir,),
              processes=num_workers) as pool:
        # 진행률 표시와 함께 검사
        bad = [
            rel for rel in
            tqdm(pool.imap_unordered(is_bad_npz, rel_paths), total=total)
            if rel
        ]

    # 결과 요약
    print(f"\nTotal files: {total}")
    print(f"Bad files:   {len(bad)}  ({len(bad)/total*100:.2f}%)\n")

    if not bad:
        print("No bad files detected. Exiting.")
        return

    print("List of bad files:")
    for rel in bad:
        print("  -", rel)

    # 삭제 여부 확인
    ans = input("\nDelete these files? [y/N]: ").strip().lower()
    if ans == 'y':
        for rel in bad:
            path = os.path.join(args.data_dir, rel)
            try:
                os.remove(path)
                print(f"Deleted: {rel}")
            except Exception as e:
                print(f"Failed to delete {rel}: {e}")
        print("Deletion complete.\n")
    else:
        print("No files deleted.\n")

    # JSON 업데이트 여부 확인
    ans = input("Remove entries from JSON list? [y/N]: ").strip().lower()
    if ans == 'y':
        backup = args.data_list + ".bak"
        os.rename(args.data_list, backup)
        updated = [rel for rel in rel_paths if rel not in bad]
        with open(args.data_list, 'w') as f:
            json.dump(updated, f, indent=2)
        print(f"Updated JSON saved (backup: {backup}).")
    else:
        print("JSON list unchanged.")

if __name__ == "__main__":
    main()

"""

chmod +x clean_bad_npz.py
./clean_bad_npz.py \
  --data_dir /mnt/nuplan/dataset/processed \
  --data_list /mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json

"""