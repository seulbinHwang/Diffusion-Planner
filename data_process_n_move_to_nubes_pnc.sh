#!/bin/bash
set -e

# ---------------- CPU 분리 설정 ----------------
CPUSET="0-111"
NUM_CPUS=112

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
# ----------------------------------------------
# ✅ 스크립트(현재 쉘) 자체를 CPUSET에 고정 → 이후 실행되는 하위 작업들도 그대로 따라감
taskset -cp "${CPUSET}" $$ >/dev/null

echo "[NUPLAN] Using CPUSET=${CPUSET}, NUM_CPUS=${NUM_CPUS}"

echo "Step 1: Running data processing..."

NUPLAN_DATA_PATH="/media/user/E/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="/media/user/E/dataset/maps"

TRAIN_SET_NAME="processed_nuplan_0124"
SAVE_PATH="/media/user/D/dataset/${TRAIN_SET_NAME}"
TRAIN_JSON_PATH="${TRAIN_SET_NAME}_json"

CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES= PYTORCH_ENABLE_MPS_FALLBACK=0 \
taskset -c "${CPUSET}" \
python data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$SAVE_PATH" \
  --total_scenarios 1000000 \
  --reset_save_path False \
  --save_image false \
  --num_workers ${NUM_CPUS}

echo "Data processing finished."
echo "---------------------------------"
echo "Step 2: Cleaning bad NPZ files..."

DATA_LIST_PATH="./diffusion_planner_training.json"

taskset -c "${CPUSET}" \
python ./clean_bad_npz.py \
  --data_dir "$SAVE_PATH" \
  --data_list "$DATA_LIST_PATH"

echo "Cleaning finished."
echo "---------------------------------"
echo "Step 3: Uploading processed data..."

taskset -c "${CPUSET}" \
nubescli dir-upload "labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_SET_NAME}" \
  "$SAVE_PATH" \
  -e -j ${NUM_CPUS}

taskset -c "${CPUSET}" \
nubescli upload \
  "labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_JSON_PATH}/diffusion_planner_training.json" \
  "/media/user/E/projects/Diffusion-Planner/diffusion_planner_training.json"

echo "Upload complete."
echo "Pipeline finished successfully."
