import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Set, Tuple


def _is_location_type(filename: str) -> bool:
    """파일명이 '위치 정보 포함 타입'인지 판별합니다.

    규칙:
      - 확장자를 제거한 stem 문자열에 '-' 문자가 있으면 위치 정보 포함 타입으로 간주합니다.
      - '-'가 없으면 토큰만 있는 타입으로 간주합니다.

    Args:
      filename: JSON에 들어있는 파일명(또는 경로일 수도 있음)

    Returns:
      위치 정보 포함 타입이면 True, 아니면 False
    """
    name = os.path.basename(filename.strip())

    # 확장자 제거(속도/간단함)
    if name.endswith(".npz"):
        stem = name[:-4]
    else:
        stem = name.rsplit(".", 1)[0]

    return "-" in stem


def _load_unique_filenames(json_path: Path) -> List[str]:
    """JSON 배열에서 파일명을 읽어와서, basename 기준으로 중복을 제거한 리스트를 반환합니다.

    Args:
      json_path: 입력 JSON 경로 (예: diffusion_planner_fine_tuning.json)

    Returns:
      중복 제거된 파일명 리스트 (basename 기준)

    Raises:
      FileNotFoundError: 파일이 없을 때
      ValueError: JSON이 리스트가 아니거나, 문자열이 아닌 항목이 있을 때
    """
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list.")

    seen: Set[str] = set()
    out: List[str] = []

    for x in data:
        if not isinstance(x, str):
            raise ValueError("JSON must contain only strings (filenames).")

        name = os.path.basename(x.strip())
        if not name:
            continue

        if name not in seen:
            seen.add(name)
            out.append(name)

    if not out:
        raise ValueError("JSON list is empty (no filenames).")

    return out


def _sample_n(items: List[str], n: int, seed: int) -> List[str]:
    """리스트에서 중복 없이 n개를 랜덤 샘플링합니다.

    Args:
      items: 후보 리스트
      n: 남길 개수
      seed: 랜덤 시드(재현용)

    Returns:
      샘플링된 리스트(길이 n)

    Raises:
      ValueError: items 개수가 n보다 작을 때
    """
    if n < 0:
        raise ValueError("n must be >= 0")

    if len(items) < n:
        raise ValueError(f"Not enough items to sample: need {n}, but got {len(items)}")

    rng = random.Random(seed)
    return rng.sample(items, n)


def build_2n_finetuning_json(
    input_json_path: Path,
    output_json_path: Path,
    n: int,
    seed: int = 0,
) -> None:
    """입력 JSON에서 '토큰만' N개 + '위치정보' N개를 랜덤으로 남긴 JSON을 생성합니다.

    Args:
      input_json_path: 입력 JSON 경로
      output_json_path: 출력 JSON 경로
      n: 각 그룹에서 남길 개수(N)
      seed: 랜덤 시드(재현 가능하게 하고 싶을 때 사용)

    Raises:
      ValueError: 그룹별 데이터가 N개보다 적으면 에러
    """
    names = _load_unique_filenames(input_json_path)

    token_only: List[str] = []
    location_type: List[str] = []

    for name in names:
        if _is_location_type(name):
            location_type.append(name)
        else:
            token_only.append(name)

    token_keep = _sample_n(token_only, n, seed=seed)
    loc_keep = _sample_n(location_type, n, seed=seed + 1)  # 두 그룹이 같은 seed로 섞이는 걸 피하려고 +1

    kept = token_keep + loc_keep

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print("[DONE]")
    print(f"  input_json          = {input_json_path}")
    print(f"  output_json         = {output_json_path}")
    print(f"  token_only_total    = {len(token_only)}")
    print(f"  location_type_total = {len(location_type)}")
    print(f"  token_only_kept     = {len(token_keep)}")
    print(f"  location_type_kept  = {len(loc_keep)}")
    print(f"  output_total        = {len(kept)} (expected {2*n})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json",
        help="Input JSON path",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Keep N items from token-only group and N items from location-type group (total 2N).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. If omitted, auto-generate with 2N in filename.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)

    # 사용자가 output을 안 주면, 요청하신 '...fine_tuning{2N}.json' 형태로 생성
    if args.output is None:
        two_n = 2 * args.n
        output_path = Path(f"/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_fine_tuning{two_n}.json")
    else:
        output_path = Path(args.output)

    build_2n_finetuning_json(
        input_json_path=input_path,
        output_json_path=output_path,
        n=args.n,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()


# python3 make_2n_finetuning_json.py --n 5000 --seed 0