import json
from pathlib import Path

def count_npz_name_types(json_path: str | Path) -> dict[str, int]:
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        filenames = json.load(f)

    token_only = 0
    with_location = 0
    others = 0

    for name in filenames:
        # 안전장치: 문자열이 아니면 스킵/카운트
        if not isinstance(name, str):
            others += 1
            continue

        # 확장자 제거
        stem = Path(name).stem  # e.g. "us-nv-las-vegas-strip_000b1f81f08750af"

        # "-" 포함 여부로 타입 분류
        if "-" in stem:
            with_location += 1
        else:
            token_only += 1

    return {
        "token_only": token_only,
        "with_location": with_location,
        "others": others,
        "total": len(filenames),
    }

if __name__ == "__main__":
    result = count_npz_name_types("diffusion_planner_training.json")
    print(result)
