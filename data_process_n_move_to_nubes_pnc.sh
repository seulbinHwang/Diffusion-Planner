#!/bin/bash
# This script runs the data processing and then cleans the generated dataset.

# [추가] 96코어 고정 및 내부 스레드 1로 제한
export DP_MAX_CPUS=32
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_NUM_INTEROP_THREADS=1

# Exit immediately if a command exits with a non-zero status.
set -e


echo "Step 1: Running data processing..."
# 기존의 export 라인은 지우고, 아래 한 줄로 대체

# Configuration from data_process_pnc.sh
# You can modify these paths if needed.
NUPLAN_DATA_PATH="/media/user/E/dataset/nuplan-v1.1/splits/trainval"
NUPLAN_MAP_PATH="/media/user/E/dataset/maps"
# 공통 경로 변수 (한 곳만 바꾸면 전체에 반영됨)
TRAIN_SET_NAME="processed_many"
TRAIN_SET_PATH="/media/user/D/dataset/${TRAIN_SET_NAME}"
TRAIN_JSON_PATH="${TRAIN_SET_NAME}_json"

# Run the data processing script
# This is the command from data_process_pnc.sh
CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES= PYTORCH_ENABLE_MPS_FALLBACK=0 \
taskset -c 0-31 \
python data_process.py \
  --data_path "$NUPLAN_DATA_PATH" \
  --map_path "$NUPLAN_MAP_PATH" \
  --save_path "$TRAIN_SET_PATH" \
  --total_scenarios 1000000 \
  --reset_save_path False

echo "Data processing finished."
echo "---------------------------------"
echo "Step 2: Cleaning bad NPZ files..."


# The data_process.py script generates 'diffusion_planner_training.json' in the current directory.
DATA_LIST_PATH="./diffusion_planner_training.json"

# Make the cleaning script executable
chmod +x clean_bad_npz.py

# Run the cleaning script.
# The --data_dir corresponds to TRAIN_SET_PATH, and --data_list is the generated JSON file.
# 파이썬으로 호출하는 게 가장 안전 (chmod 불필요)
python ./clean_bad_npz.py \
  --data_dir "$TRAIN_SET_PATH" \
  --data_list "$DATA_LIST_PATH"

echo "Cleaning finished."
echo "---------------------------------"
echo "Step 3: Uploading processed data..."

nubescli dir-upload "labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_SET_NAME}" \
                    "$TRAIN_SET_PATH" \
                    -e -j 64

nubescli upload labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_JSON_PATH}/diffusion_planner_training.json \
                    /media/user/E/projects/Diffusion-Planner/diffusion_planner_training.json
echo "Upload complete."
echo "Pipeline finished successfully."
