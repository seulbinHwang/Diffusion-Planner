import json
import os
import stat
import heapq
from pathlib import Path
from typing import List, Set, Tuple


JSON_PATH = Path("/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_fine_tuning.json")
SRC_DIR = Path("/mnt/nuplan/dataset/processed")          # 여기서 찾기 (하위 폴더 X)
DST_DIR = Path("/mnt/nuplan/dataset/processed_rollout")  # 여기로 이동(복사 X)

# names가 이 값 이상이면, per-file stat 대신 디렉토리 1회 스캔(scandir)로 검증하는 게 보통 더 빠릅니다.
SCAN_THRESHOLD = 1000


def _sample(items: Set[str], k: int = 20) -> str:
    """큰 에러 목록을 전부 출력하면 느리니, 상위 k개만 정렬해서 보여줍니다."""
    if not items:
        return ""
    head = heapq.nsmallest(k, items)
    more = len(items) - len(head)
    msg = "\n".join(head)
    if more > 0:
        msg += f"\n... (+{more} more)"
    return msg


def load_filenames_fast(json_path: Path) -> List[str]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON must be a list of strings (filenames).")

    names: List[str] = []
    seen: Set[str] = set()
    dup: Set[str] = set()

    # O(N) 중복 체크 + basename 강제
    for x in data:
        if not isinstance(x, str):
            raise ValueError("JSON must contain only strings.")
        x = x.strip()
        if not x:
            continue

        name = os.path.basename(x)  # Path(x).name 보다 가벼움
        if name in seen:
            dup.add(name)
        else:
            seen.add(name)
            names.append(name)

    if not names:
        raise ValueError("JSON list is empty (no filenames).")

    if dup:
        raise ValueError(
            f"Duplicate filenames in JSON: {len(dup)} items\n{_sample(dup)}"
        )

    return names


def precheck_by_scandir(names: List[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """대량일 때 빠른 검증: SRC/DST 디렉토리를 한 번씩 쭉 스캔."""
    wanted = set(names)

    missing = set(wanted)  # 스캔하면서 발견하면 제거
    bad_type: Set[str] = set()  # 디렉토리 등(파일로 이동 불가)

    with os.scandir(SRC_DIR) as it:
        for ent in it:
            n = ent.name
            if n in missing:
                # 디렉토리는 이동 대상에서 제외(에러)
                try:
                    if ent.is_dir(follow_symlinks=False):
                        bad_type.add(n)
                    # 파일/심볼릭링크 등 디렉토리 아닌 것은 OK
                    missing.remove(n)
                except OSError:
                    # stat 실패도 안전하게 bad 취급
                    bad_type.add(n)
                    missing.discard(n)

    conflicts: Set[str] = set()
    with os.scandir(DST_DIR) as it:
        for ent in it:
            if ent.name in wanted:
                conflicts.add(ent.name)

    return missing, conflicts, bad_type


def precheck_by_stat(names: List[str], src_fd: int, dst_fd: int) -> Tuple[Set[str], Set[str], Set[str]]:
    """소량일 때 빠른 검증: 필요한 파일만 stat (dir_fd 사용으로 경로해석 비용 감소)."""
    missing: Set[str] = set()
    conflicts: Set[str] = set()
    bad_type: Set[str] = set()

    for name in names:
        # SRC 존재/타입 체크
        try:
            st = os.stat(name, dir_fd=src_fd, follow_symlinks=False)
        except FileNotFoundError:
            missing.add(name)
        else:
            if stat.S_ISDIR(st.st_mode):
                bad_type.add(name)

        # DST 충돌 체크
        try:
            os.stat(name, dir_fd=dst_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            conflicts.add(name)

    return missing, conflicts, bad_type


def main() -> None:
    # 목적지 폴더는 만들어둠
    DST_DIR.mkdir(parents=True, exist_ok=True)

    # 같은 파일시스템에서 rename이 가장 빠름 (다르면 EXDEV 나서 실패)
    if os.stat(SRC_DIR).st_dev != os.stat(DST_DIR).st_dev:
        raise RuntimeError(
            "SRC_DIR and DST_DIR are on different filesystems (st_dev differs). "
            "Fast move via os.rename cannot be used. (would require copy+delete)"
        )

    names = load_filenames_fast(JSON_PATH)

    # 디렉토리 FD 열어두면, rename/stat 시 경로 join/해석 비용이 줄어듭니다.
    src_fd = os.open(SRC_DIR, os.O_RDONLY)
    dst_fd = os.open(DST_DIR, os.O_RDONLY)
    try:
        # 1) 사전 검증(빠르게)
        if len(names) >= SCAN_THRESHOLD:
            missing, conflicts, bad_type = precheck_by_scandir(names)
        else:
            missing, conflicts, bad_type = precheck_by_stat(names, src_fd, dst_fd)

        if bad_type:
            raise IsADirectoryError(
                f"Some entries in SRC_DIR are directories (not movable as files): {len(bad_type)} items\n"
                f"{_sample(bad_type)}"
            )

        if missing:
            raise FileNotFoundError(
                f"Some files were not found in the TOP level of {SRC_DIR}: {len(missing)} items\n"
                f"{_sample(missing)}"
            )

        if conflicts:
            raise FileExistsError(
                f"Destination already has conflicting names in {DST_DIR}: {len(conflicts)} items\n"
                f"{_sample(conflicts)}"
            )

        # 2) 이동(rename) + 실패 시 롤백
        moved: List[str] = []
        try:
            # renameat 사용: (SRC_DIR/name) -> (DST_DIR/name)
            for name in names:
                os.rename(name, name, src_dir_fd=src_fd, dst_dir_fd=dst_fd)
                moved.append(name)

            print(f"✅ Moved {len(moved)} files to: {DST_DIR}")

        except Exception as e:
            rollback_errors: List[str] = []

            # 이미 옮긴 것 되돌리기 (DST -> SRC)
            for name in reversed(moved):
                try:
                    # rollback 시에도 overwrite 방지: SRC에 같은 이름이 생겼으면 막음
                    try:
                        os.stat(name, dir_fd=src_fd, follow_symlinks=False)
                        rollback_errors.append(
                            f"Rollback blocked: {SRC_DIR / name} already exists (cannot restore {DST_DIR / name})"
                        )
                        continue
                    except FileNotFoundError:
                        pass

                    os.rename(name, name, src_dir_fd=dst_fd, dst_dir_fd=src_fd)

                except Exception as re:
                    rollback_errors.append(f"Rollback failed for {name}: {re}")

            if rollback_errors:
                raise RuntimeError(
                    "Move failed and rollback had problems:\n" + "\n".join(rollback_errors)
                ) from e

            raise RuntimeError("Move failed, but rollback succeeded (restored original state).") from e

    finally:
        os.close(src_fd)
        os.close(dst_fd)


if __name__ == "__main__":
    main()