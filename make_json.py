import json
from pathlib import Path
from typing import List


def list_npz_filenames(folder_path: str) -> List[str]:
    """폴더 안의 .npz 파일 이름만 모아서 정렬해 반환합니다.

    Args:
        folder_path: .npz 파일들이 있는 폴더 경로

    Returns:
        filenames: (N,) 형태의 리스트.
            예: ["aaa.npz", "bbb.npz", ...]
    """
    p = Path(folder_path)
    if not p.exists() or not p.is_dir():
        raise FileNotFoundError(f"폴더가 없습니다: {folder_path}")

    filenames = [f.name for f in p.glob("*.npz") if f.is_file()]
    filenames.sort()
    return filenames


def save_list_as_json(items: List[str], output_json_path: str) -> None:
    """문자열 리스트를 JSON 파일로 저장합니다.

    Args:
        items: 저장할 문자열 리스트
        output_json_path: 저장할 json 파일 경로

    Returns:
        None
    """
    out_path = Path(output_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    """지정 폴더의 .npz 파일 이름을 모아 JSON으로 저장합니다."""
    src_dir = "/media/user/D/dataset/processed_womd_0124/training"
    dst_json = "/media/user/E/projects/Diffusion-Planner/diffusion_planner/processed_womd_0124_training.json"

    filenames = list_npz_filenames(src_dir)
    if len(filenames) == 0:
        raise RuntimeError(f".npz 파일이 없습니다: {src_dir}")

    save_list_as_json(filenames, dst_json)
    print(f"Saved {len(filenames)} filenames to: {dst_json}")


if __name__ == "__main__":
    main()
