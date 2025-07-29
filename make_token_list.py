#!/usr/bin/env python
"""
nuPlan 데이터베이스(.db)에서 scenario.token만 수집해 JSON 저장.
실행 예:
  python make_token_list.py \
      --data_path /mnt/nuplan/dataset/nuplan-v1.1/splits/trainval \
      --output   scenario_tokens_trainval.json
"""
from __future__ import annotations
import argparse, json, os, sqlite3
from glob import glob
from tqdm import tqdm
from pathlib import Path

def extract_tokens(db_path: str) -> list[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.execute("SELECT token FROM scenario;")
    tokens = [row[0] for row in cur.fetchall()]
    con.close()
    return tokens

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="nuPlan trainval split 폴더 (log_xxx.db 들이 있는 상위)")
    ap.add_argument("--output", default="scenario_tokens.json")
    args = ap.parse_args()

    db_files = glob(os.path.join(args.data_path, "**/*.db"), recursive=True)
    print(f"DB 파일 {len(db_files)}개 스캔 시작...")

    all_tokens: set[str] = set()
    for db in tqdm(db_files, unit="db"):
        try:
            all_tokens.update(extract_tokens(db))
        except sqlite3.DatabaseError as e:
            print(f"[경고] 손상 DB 건너뜀: {Path(db).name} ({e})")

    with open(args.output, "w") as f:
        json.dump(sorted(all_tokens), f, indent=2)

    print(f"총 {len(all_tokens):,}개 토큰 저장 → {args.output}")
