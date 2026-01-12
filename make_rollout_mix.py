import json
import os
from pathlib import Path
from typing import List, Tuple


JSON_PATH = Path("/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_fine_tuning.json")
SRC_DIR = Path("/mnt/nuplan/dataset/processed")          # 여기서 찾기 (하위 폴더 X)
DST_DIR = Path("/mnt/nuplan/dataset/processed_rollout")  # 여기로 "이동"(복사 X)


def load_filenames(json_path: Path) -> List[str]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("JSON must be a list of strings (filenames or paths).")

    # 혹시 JSON에 경로가 들어있어도 파일명만 쓰도록 basename 처리
    names = [Path(x).name for x in data if x.strip()]
    if not names:
        raise ValueError("JSON list is empty (no filenames).")

    # 중복 체크
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Duplicate filenames in JSON: {dup}")

    return names


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)

    names = load_filenames(JSON_PATH)

    # 1) 사전 검증: 전부 존재하는지(하위 폴더 탐색 X), 목적지에 겹치는지
    missing = []
    src_paths = []
    dst_paths = []
    conflicts = []

    for name in names:
        src = SRC_DIR / name
        dst = DST_DIR / name

        # 하위 폴더는 안 찾음: 딱 이 경로만 체크
        if not src.is_file():
            missing.append(str(src))
        if dst.exists():
            conflicts.append(str(dst))

        src_paths.append(src)
        dst_paths.append(dst)

    if missing:
        raise FileNotFoundError(
            "Some files were not found in the top-level of /mnt/nuplan/dataset/processed:\n"
            + "\n".join(missing)
        )

    if conflicts:
        raise FileExistsError(
            "Destination already has conflicting files (would risk overwrite):\n"
            + "\n".join(conflicts)
        )

    # 2) 이동 수행 + 실패 시 롤백
    moved: List[Tuple[Path, Path]] = []  # (src_original, dst_current)
    try:
        for src, dst in zip(src_paths, dst_paths):
            # 같은 파일시스템이면 rename(이동), 아니면 shutil.move가 필요하지만
            # 여기서는 overwrite 방지를 위해 os.rename 사용(사전 conflicts 체크 완료)
            os.rename(src, dst)
            moved.append((src, dst))

        print(f"✅ Moved {len(moved)} files to: {DST_DIR}")

    except Exception as e:
        print(f"❌ Error during move: {e}\nTrying rollback...")

        rollback_errors = []
        for src, dst in reversed(moved):
            try:
                # 되돌릴 때도 overwrite 방지: 원래 자리에 뭔가 생겼으면 에러
                if src.exists():
                    rollback_errors.append(
                        f"Rollback blocked: source already exists: {src} (cannot restore {dst})"
                    )
                    continue
                if not dst.exists():
                    rollback_errors.append(f"Rollback missing: moved file not found at {dst}")
                    continue
                os.rename(dst, src)
            except Exception as re:
                rollback_errors.append(f"Rollback failed for {dst} -> {src}: {re}")

        if rollback_errors:
            raise RuntimeError(
                "Move failed and rollback had problems:\n"
                + "\n".join(rollback_errors)
            ) from e

        raise RuntimeError("Move failed, but rollback succeeded (restored original state).") from e


if __name__ == "__main__":
    main()
