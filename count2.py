import json
from collections import defaultdict
from pathlib import Path


def make_fine_tuning_list(
    input_json: str = "diffusion_planner_training.json",
    output_json: str = "diffusion_planner_fine_tuning.json",
    start_remove: bool = True,  # True면 "첫 번째부터 제거" (제거/유지/제거/유지...), False면 "첫 번째부터 유지"
) -> None:
    input_path = Path(input_json)
    output_path = Path(output_json)

    with input_path.open("r", encoding="utf-8") as f:
        filenames = json.load(f)

    # 위치 prefix(예: us-nv-las-vegas-strip)별로 번갈아 제거/유지 토글
    remove_next_by_prefix = defaultdict(lambda: start_remove)

    kept = []
    stats = {
        "token_only_kept": 0,
        "location_total": 0,
        "location_kept": 0,
        "location_removed": 0,
    }

    for fn in filenames:
        # 혹시 문자열이 아니면 그대로 보존(원하면 여기서 continue로 버려도 됨)
        if not isinstance(fn, str):
            kept.append(fn)
            continue

        # 확장자 제거 (속도 고려해 Path 대신 문자열 처리)
        stem = fn[:-4] if fn.endswith(".npz") else fn.rsplit(".", 1)[0]

        # "-" 포함 여부로 위치정보 포함 타입 판별
        if "-" not in stem:
            kept.append(fn)
            stats["token_only_kept"] += 1
            continue

        stats["location_total"] += 1

        # 위치 prefix는 "_" 앞부분으로 가정: "us-nv-..._TOKEN"
        prefix = stem.split("_", 1)[0]

        if remove_next_by_prefix[prefix]:
            # 이번 건 제거
            stats["location_removed"] += 1
        else:
            # 이번 건 유지
            kept.append(fn)
            stats["location_kept"] += 1

        # 다음에는 반대로
        remove_next_by_prefix[prefix] = not remove_next_by_prefix[prefix]

    # 출력 파일 작성 (원본 input_json은 수정하지 않음)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False, indent=2)

    print(f"[DONE] wrote: {output_path}")
    print(f"  token_only_kept    = {stats['token_only_kept']}")
    print(f"  location_total     = {stats['location_total']}")
    print(f"  location_kept      = {stats['location_kept']}")
    print(f"  location_removed   = {stats['location_removed']}")
    print(f"  output_total_count = {len(kept)}")


if __name__ == "__main__":
    make_fine_tuning_list(
        input_json="diffusion_planner_training.json",
        output_json="diffusion_planner_fine_tuning.json",
        start_remove=True,  # "제거부터 시작" (원하면 False로 바꾸면 첫 번째는 유지)
    )
