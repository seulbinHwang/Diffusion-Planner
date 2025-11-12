# tools/make_args_json_from_cli.py
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import args_util  # get_args()를 사용

EXCLUDE_KEYS = {"state_normalizer", "observation_normalizer"}

def ns_to_dict(ns: argparse.Namespace,
               drop_none: bool = True,
               exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Namespace를 dict로 변환하면서 직렬화 불가/불필요 키를 제외."""
    d = vars(ns)
    if drop_none:
        d = {k: v for k, v in d.items() if v is not None}
    if exclude:
        excl = set(exclude)
        d = {k: v for k, v in d.items() if k not in excl}
    return d

def dump_json(obj: Dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def parse_cli_for_generator() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="get_args() 값만으로 args_base_custom.json 생성"
    )
    p.add_argument("--out-json", required=True, help="생성할 JSON 경로")
    # get_args용 인자들은 '--' 뒤에 그대로 붙인다.
    p.add_argument("rest", nargs=argparse.REMAINDER,
                   help="-- 로 구분한 뒤의 get_args용 CLI들")
    return p.parse_args()

def main() -> None:
    gen_args = parse_cli_for_generator()

    # '--' 분리
    raw_cli = gen_args.rest
    if raw_cli and raw_cli[0] == "--":
        raw_cli = raw_cli[1:]

    # 1) get_args 호출
    # args_util.get_args(argv=None) 시그니처를 수정하기 어렵다면, 아래 B안을 쓰면 됨.
    # --- A안: get_args가 argv 파라미터를 받도록 수정한 경우 ---
    # ns = args_util.get_args(raw_cli)

    # --- B안(수정 불필요): sys.argv 트릭 ---
    prev_argv = sys.argv[:]
    try:
        sys.argv = [prev_argv[0]] + (raw_cli or [])
        ns = args_util.get_args()
    finally:
        sys.argv = prev_argv

    # 2) 직렬화 가능한 dict로 변환 (+ normalizer 키 제외)
    out = ns_to_dict(ns, drop_none=True, exclude=EXCLUDE_KEYS)

    # 3) 저장
    dump_json(out, gen_args.out_json)
    print(f"[OK] wrote: {gen_args.out_json}")

if __name__ == "__main__":
    main()
