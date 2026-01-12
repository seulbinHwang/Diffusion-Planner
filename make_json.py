import json
from pathlib import Path

SRC_DIR = Path("/mnt/nuplan/dataset/processed_rollout")  # 여기서 npz 파일명 수집 (하위 폴더 X)
OUT_JSON = Path("/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_fine_tuning_train.json")

def main():
    if not SRC_DIR.exists():
        raise FileNotFoundError(f"Source directory not found: {SRC_DIR}")

    # 하위 폴더 제외: glob 사용
    npz_names = sorted(
        p.name
        for p in SRC_DIR.glob("*.npz")
        if p.is_file()
    )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(npz_names, f, ensure_ascii=False, indent=2)

    print(f"Found {len(npz_names)} .npz files in: {SRC_DIR}")
    print(f"Saved JSON to: {OUT_JSON}")

if __name__ == "__main__":
    main()
