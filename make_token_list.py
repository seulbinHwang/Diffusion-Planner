#!/usr/bin/env python
""" scenario_tokens json 생성 v2 – log DB는 자동 건너뜀 """

from __future__ import annotations
import argparse, json, os, sqlite3
from glob import glob
from tqdm import tqdm
from pathlib import Path

def extract_tokens(db_path: str) -> list[str]:
    """scenario 테이블이 있을 때만 token 추출"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='scenario';"
    )
    has_table = cur.fetchone() is not None
    if not has_table:
        con.close()
        return []                       # log DB → 토큰 없음

    tokens = [row[0] for row in con.execute("SELECT token FROM scenario;")]
    con.close()
    return tokens

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output", default="scenario_tokens_trainval.json")
    args = ap.parse_args()

    db_files = glob(os.path.join(args.data_path, "**/*.db"), recursive=True)
    print(f"총 {len(db_files)}개 DB 파일 스캔…")

    all_tokens: set[str] = set()
    skipped:   list[str] = []

    for db in tqdm(db_files, unit="db"):
        try:
            toks = extract_tokens(db)
            if toks:
                all_tokens.update(toks)
            else:
                skipped.append(Path(db).name)       # log DB 목록
        except sqlite3.DatabaseError as e:
            print(f"[경고] 손상 DB 건너뜀: {Path(db).name} ({e})")

    with open(args.output, "w") as f:
        json.dump(sorted(all_tokens), f, indent=2)
    print(f"✔ 토큰 {len(all_tokens):,}개 저장 → {args.output}")

    # 필요하면 스킵한 log DB 파일 이름도 기록
    if skipped:
        with open("log_db_skipped.txt", "w") as f:
            f.write("\n".join(skipped))
        print(f"ℹ log DB {len(skipped)}개는 scenario 테이블이 없어 건너뜀 → log_db_skipped.txt")

"""
python make_token_list.py \
  --data_path /mnt/nuplan/dataset/nuplan-v1.1/splits/trainval \
  --output    scenario_tokens_trainval.json
"""