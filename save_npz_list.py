from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List


def list_npz_filenames_depth1(input_dir: Path) -> List[str]:
    """지정한 폴더 바로 아래(depth=1)의 .npz 파일 이름만 모아서 반환합니다.

    Args:
        input_dir (Path): .npz 파일들이 들어있는 폴더 경로.

    Returns:
        List[str]: 폴더 바로 아래에 있는 .npz 파일 이름 리스트(정렬됨).
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"입력 경로가 폴더가 아닙니다: {input_dir}")

    filenames: List[str] = []
    for p in input_dir.glob("*.npz"):  # depth=1: 하위 폴더로는 내려가지 않음
        if p.is_file():
            filenames.append(p.name)

    filenames.sort()
    return filenames


def save_filenames_as_json(filenames: List[str], output_json_path: Path) -> None:
    """파일 이름 리스트를 JSON 배열 형태로 저장합니다.

    Args:
        filenames (List[str]): 저장할 파일 이름 리스트.
        output_json_path (Path): 저장할 JSON 파일 경로.

    Returns:
        None
    """
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as f:
        json.dump(filenames, f, ensure_ascii=False, indent=4)


def main() -> None:
    """폴더의 .npz 파일명을 모아 JSON으로 저장합니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        default="~/womd_v1_3/processed_womd_0124/training",
        help="depth=1로 .npz 파일을 찾을 폴더",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="~/PycharmProjects/Diffusion-Planner/diffusion_planner_training_womd.json",
        help="파일명 리스트를 저장할 json 경로",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir).expanduser()
    output_json = Path(args.output_json).expanduser()

    filenames = list_npz_filenames_depth1(input_dir)
    save_filenames_as_json(filenames, output_json)

    print(f"입력 폴더: {input_dir}")
    print(f"찾은 .npz 개수: {len(filenames)}")
    print(f"저장 완료: {output_json}")


if __name__ == "__main__":
    main()