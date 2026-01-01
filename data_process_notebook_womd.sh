###################################
# User Configuration Section
###################################
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export WOMD_DATA_ROOT="/home/user/womd_v1_3"
SAVE_PATH="${WOMD_DATA_ROOT}/processed_womd_final"
###################################

python data_process_womd.py --womd_data_path "$WOMD_DATA_ROOT" \
  --save_path "$SAVE_PATH" \
  --save_image false \
  --num_workers 1 \
  --womd_splits "testing"
