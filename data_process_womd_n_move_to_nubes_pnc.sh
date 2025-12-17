#!/bin/bash
# This script runs the data processing and then cleans the generated dataset.

# [추가] 96코어 고정 및 내부 스레드 1로 제한
export DP_MAX_CPUS=96
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Exit immediately if a command exits with a non-zero status.
set -e


echo "Step 1: Running data processing..."
# 기존의 export 라인은 지우고, 아래 한 줄로 대체

# Configuration from data_process_pnc.sh
# You can modify these paths if needed.
WOMD_DATA_ROOT="/media/user/E/dataset/womd_v1_3"
# 공통 경로 변수 (한 곳만 바꾸면 전체에 반영됨)
TRAIN_SET_NAME="processed_womd_final"
TRAIN_SET_PATH="${WOMD_DATA_ROOT}/scenario/${TRAIN_SET_NAME}"

# Run the data processing script
# This is the command from data_process_pnc.sh
CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES= PYTORCH_ENABLE_MPS_FALLBACK=0 \
taskset -c 0-95 \
python data_process_womd.py --womd_data_path "$WOMD_DATA_ROOT" --save_folder "$TRAIN_SET_NAME" --num_workers 96 --overwrite_womd_cache false

echo "Data processing finished."
echo "---------------------------------"
echo "Step 2: Uploading processed data..."

nubescli dir-upload "labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_SET_NAME}" \
                    "$TRAIN_SET_PATH" \
                    -e -j 64

echo "Upload complete."
echo "Pipeline finished successfully."
