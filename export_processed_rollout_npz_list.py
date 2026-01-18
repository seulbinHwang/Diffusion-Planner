import os
from pathlib import Path

# (선택) 더 빠른 JSON serializer가 있으면 사용
try:
    import orjson  # type: ignore

    def _json_quote(s: str) -> str:
        # orjson.dumps는 bytes를 반환하므로 decode 필요
        return orjson.dumps(s).decode("utf-8")

except Exception:
    import json

    def _json_quote(s: str) -> str:
        return json.dumps(s, ensure_ascii=False)


def write_npz_filenames_to_json(
    input_dir: str,
    output_json_path: str,
    *,
    sort_names: bool = False,
) -> None:
    """폴더(하위폴더 제외)에서 .npz 파일명만 JSON 배열로 저장합니다.

    Args:
        input_dir (str):
            .npz 파일들이 있는 폴더 경로.
            - 하위 폴더 안 파일은 고려하지 않습니다.
        output_json_path (str):
            저장할 JSON 파일 경로.
        sort_names (bool):
            True면 파일명을 정렬해서 저장합니다.
            False면 디렉터리 순회 순서대로 저장합니다(더 빠름).

    Returns:
        None
    """
    in_dir = Path(input_dir)
    out_path = Path(output_json_path)

    if not in_dir.is_dir():
        raise FileNotFoundError(f"input_dir not found or not a directory: {in_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sort_names:
        # 정렬이 필요하면 메모리에 올려서 정렬 (파일 수가 매우 많으면 비용 증가)
        names = []
        with os.scandir(in_dir) as it:
            for entry in it:
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".npz"):
                    names.append(entry.name)
        names.sort()

        with open(out_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
            f.write("[\n")
            for i, name in enumerate(names):
                if i > 0:
                    f.write(",\n")
                f.write(f"  {_json_quote(name)}")
            f.write("\n]\n")
        return

    # 빠른 모드: 스트리밍으로 바로 write (메모리 사용 최소)
    with open(out_path, "w", encoding="utf-8", buffering=1024 * 1024) as f:
        f.write("[\n")
        first = True
        with os.scandir(in_dir) as it:
            for entry in it:
                if not entry.is_file(follow_symlinks=False):
                    continue
                name = entry.name
                if not name.endswith(".npz"):
                    continue
                if first:
                    first = False
                    f.write(f"  {_json_quote(name)}")
                else:
                    f.write(f",\n  {_json_quote(name)}")
        f.write("\n]\n")


if __name__ == "__main__":
    INPUT_DIR = "/mnt/nuplan/dataset/processed_rollout"
    OUTPUT_JSON = "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_amortized_fine_tuning.json"

    write_npz_filenames_to_json(
        input_dir=INPUT_DIR,
        output_json_path=OUTPUT_JSON,
        sort_names=False,  # True로 하면 정렬된 결과(대신 느려질 수 있음)
    )
    print(f"Saved: {OUTPUT_JSON}")
