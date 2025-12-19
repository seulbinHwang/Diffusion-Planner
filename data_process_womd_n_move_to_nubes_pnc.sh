#!/bin/bash
set -e

# ---------------- CPU 분리 설정 ----------------
CPUSET="0-55"
NUM_CPUS=56

export DP_MAX_CPUS=${NUM_CPUS}
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ----------------------------------------------
# ✅ 스크립트(현재 쉘) 자체를 CPUSET에 고정 → 이후 실행되는 하위 작업들도 그대로 따라감
taskset -cp "${CPUSET}" $$ >/dev/null

echo "[WOMD] Using CPUSET=${CPUSET}, NUM_CPUS=${NUM_CPUS}"

echo "Step 1: Running data processing..."

WOMD_DATA_ROOT="/media/user/E/dataset/womd_v1_3"
TRAIN_SET_NAME="processed_womd_final_150"
SAVE_PATH="/media/user/D/dataset/${TRAIN_SET_NAME}"

CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES= PYTORCH_ENABLE_MPS_FALLBACK=0 \
taskset -c "${CPUSET}" \
python data_process_womd.py \
  --womd_data_path "$WOMD_DATA_ROOT" \
  --num_workers ${NUM_CPUS} \
  --overwrite_womd_cache false \
  --save_path "$SAVE_PATH" \
  --save_image false

echo "Data processing finished."
echo "---------------------------------"
echo "Step 2: Uploading processed data..."

taskset -c "${CPUSET}" \
nubescli dir-upload "labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_SET_NAME}" \
  "$SAVE_PATH" \
  -e -j ${NUM_CPUS}

echo "Upload complete."
echo "Pipeline finished successfully."
