#!/usr/bin/env bash
set -Eeuo pipefail

###################################
# User Configuration Section
###################################
NUBES_WOMD_FOLDER_NAME="processed_womd_final_150_0102"
NUBES_WOMD_SPLIT_NAME="validation"
NUBES_WOMD_TF_SPLIT_NAME="validation_tfrecords_splitted"

VALIDATION_SET_PATH="/mnt/nuplan/dataset/processed_validation"   # 디렉터리 자체는 유지, 내용만 비움
VALIDATION_TF_SET_PATH="/mnt/nuplan/dataset/validation_tfrecords_splitted"   # 디렉터리 자체는 유지, 내용만 비움
VALIDATION_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_validation.json"

###################################
# 0) 학습 리스트 JSON 삭제 (있으면)
###################################
if [ -f "$VALIDATION_SET_LIST_PATH" ]; then
  echo "[REMOVE] AT PVC, We remove $VALIDATION_SET_LIST_PATH"
  rm -f "$VALIDATION_SET_LIST_PATH"
else
  echo "[REMOVE] AT PVC, We cannot remove $VALIDATION_SET_LIST_PATH cause it does not exist."
fi

###################################
# 1) processed 디렉터리 내용물 전부 삭제(숨김 포함), 디렉터리는 유지
###################################
if [ -d "$VALIDATION_SET_PATH" ]; then
  echo "[REMOVE] AT PVC, We remove all contents under $VALIDATION_SET_PATH [start]"
  # 안전 가드: 절대 루트나 빈 문자열은 청소 금지
  if [[ "$VALIDATION_SET_PATH" == "/" || "$VALIDATION_SET_PATH" == "" ]]; then
    echo "Refusing to clean unsafe directory: '$VALIDATION_SET_PATH'"; exit 1
  fi
  # 자식만 삭제(숨김 포함), 디렉터리 자체는 유지
  find "$VALIDATION_SET_PATH" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "[REMOVE] AT PVC, We remove all contents under $VALIDATION_SET_PATH [end]"
else
  echo "[Creating] AT PVC, we create folder $VALIDATION_SET_PATH"
  mkdir -p "$VALIDATION_SET_PATH"
fi

if [ -d "$VALIDATION_TF_SET_PATH" ]; then
  echo "[REMOVE] AT PVC, We remove all contents under $VALIDATION_TF_SET_PATH [start]"
  # 안전 가드: 절대 루트나 빈 문자열은 청소 금지
  if [[ "$VALIDATION_TF_SET_PATH" == "/" || "$VALIDATION_TF_SET_PATH" == "" ]]; then
    echo "Refusing to clean unsafe directory: '$VALIDATION_TF_SET_PATH'"; exit 1
  fi
  # 자식만 삭제(숨김 포함), 디렉터리 자체는 유지
  find "$VALIDATION_TF_SET_PATH" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
  echo "[REMOVE] AT PVC, We remove all contents under $VALIDATION_TF_SET_PATH [end]"
else
  echo "[Creating] AT PVC, we create folder $VALIDATION_TF_SET_PATH"
  mkdir -p "$VALIDATION_TF_SET_PATH"
fi


NUBES_JOBS=$(( ( $(nproc) * 4 ) / 5 )); if [ "$NUBES_JOBS" -lt 1 ]; then NUBES_JOBS=1; fi
echo "[NUBES_JOBS] total_cores=$(nproc), use_cores=${NUBES_JOBS} (4/5)"

###################################
# 2) NUBES -> PVC 다운로드
###################################

echo "[DOWNLOAD] WOMD dataset / NUBES validation -> PVC [start]"
nubescli dir-download \
    labs-mlops/ad/research/pnc/hsb/dataset/${NUBES_WOMD_FOLDER_NAME}/${NUBES_WOMD_SPLIT_NAME} \
    "$VALIDATION_SET_PATH" \
    -j "$NUBES_JOBS" \
    -s \
    --no-progress
echo "[DOWNLOAD] WOMD dataset / NUBES validation -> PVC [end]"

echo "[DOWNLOAD] WOMD dataset / NUBES validation TF -> PVC [start]"
nubescli dir-download \
    labs-mlops/ad/research/pnc/hsb/dataset/${NUBES_WOMD_FOLDER_NAME}/${NUBES_WOMD_TF_SPLIT_NAME} \
    "$VALIDATION_TF_SET_PATH" \
    -j "$NUBES_JOBS" \
    -s \
    --no-progress
echo "[DOWNLOAD] WOMD dataset / NUBES validation TF  -> PVC [end]"

###################################
# 3) (추가) VALIDATION_SET_PATH 바로 아래의 .npz 파일 목록을 JSON으로 저장
#    - 하위 폴더는 보지 않음
#    - .npz 확장자 포함한 파일명만 저장
###################################
echo "[LIST] Writing npz file list to $VALIDATION_SET_LIST_PATH [start]"
mkdir -p "$(dirname "$VALIDATION_SET_LIST_PATH")"

python3 - << 'PY'
import os, json, sys

validation_set_path = os.environ.get("VALIDATION_SET_PATH", "/mnt/nuplan/dataset/processed_validation")
list_path = os.environ.get("VALIDATION_SET_LIST_PATH", "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_validation.json")

if not os.path.isdir(validation_set_path):
    print(f"[ERROR] VALIDATION_SET_PATH does not exist: {validation_set_path}", file=sys.stderr)
    sys.exit(1)

files = []
for name in os.listdir(validation_set_path):
    if not name.endswith(".npz"):
        continue
    full = os.path.join(validation_set_path, name)
    if os.path.isfile(full):
        files.append(name)

files.sort()

os.makedirs(os.path.dirname(list_path), exist_ok=True)
with open(list_path, "w", encoding="utf-8") as f:
    json.dump(files, f, indent=2, ensure_ascii=False)

print(f"[LIST] Saved {len(files)} entries -> {list_path}")
PY
echo "[LIST] Writing npz file list to $VALIDATION_SET_LIST_PATH [end]"
