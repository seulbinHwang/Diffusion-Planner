import os
import time


SRC_DIR = "/mnt/nuplan/dataset/processed_rollout"
DST_DIR = "/mnt/nuplan/dataset/processed"


def move_npz_without_step(src_dir: str, dst_dir: str) -> None:
    """최상위 폴더에서 'step'이 없는 .npz 파일을 dst_dir로 이동합니다.

    Args:
        src_dir (str): 원본 폴더 경로. 예: "/mnt/nuplan/dataset/processed_rollout"
        dst_dir (str): 대상 폴더 경로. 예: "/mnt/nuplan/dataset/processed"

    Returns:
        None
    """
    os.makedirs(dst_dir, exist_ok=True)

    moved = 0
    scanned = 0
    t0 = time.time()

    # 최상위만 스캔 (하위 폴더는 entry.is_file에서 제외됨)
    with os.scandir(src_dir) as it:
        for entry in it:
            scanned += 1

            if not entry.is_file(follow_symlinks=False):
                continue

            name_lower = entry.name.lower()
            if not name_lower.endswith(".npz"):
                continue

            if "step" in name_lower:
                continue

            src_path = entry.path  # (str) full path
            dst_path = os.path.join(dst_dir, entry.name)

            # 같은 파일시스템이면 매우 빠르게 이동(rename)
            os.rename(src_path, dst_path)
            moved += 1

            # 너무 자주 출력하면 느려져서 간격을 둠
            if moved % 2000 == 0:
                elapsed = time.time() - t0
                print(f"[progress] moved={moved}, scanned={scanned}, elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"[done] moved={moved}, scanned={scanned}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    move_npz_without_step(SRC_DIR, DST_DIR)
