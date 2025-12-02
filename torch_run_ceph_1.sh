#!/usr/bin/env bash
set -Eeuo pipefail
export CUDA_LAUNCH_BLOCKING=1

export PYTHONUNBUFFERED=1

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
TRAIN_SET_NAME="processed_route_with_small"
TRAIN_JSON_PATH="${TRAIN_SET_NAME}_json"
###################################

echo "[Preflight] Cleaning previous artifacts..."
echo "[Preflight] Done."

RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/mnt/nuplan/logs/$RUN_ID
mkdir -p "$LOG_DIR"

#export CUDA_VISIBLE_DEVICES=0,1


# 디버그 env 설정 (기존 그대로)
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISABLE_ADDR2LINE=1
export CUDA_DEVICE_MAX_CONNECTIONS=32
printf "[ENV] %-28s %s\n" "OMP_NUM_THREADS:"            "${OMP_NUM_THREADS-<unset>}"
printf "[ENV] %-28s %s\n" "CUDA_DEVICE_MAX_CONNECTIONS:" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"
DEBUG_LOG=1

if (( DEBUG_LOG )); then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=3
else
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=1
fi

export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"

# ⬇ 여기서부터 wandb artifact 기반 resume ⬇
"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run \
  --nnodes 1 \
  --nproc-per-node 2 \
  --standalone \
  --log_dir "$LOG_DIR" \
  --redirects 3 \
  --tee "$TEE" \
  train_predictor.py \
  --port 23001 \
  --train_set "$TRAIN_SET_PATH"/ \
  --train_set_list "$TRAIN_SET_LIST_PATH" \
  --name "feasible_full_time_use_vel_gpu_2_exp_A" \
  --batch_size 1024 \
  --profile_feasible false \
  --use_vel_input true \
  --resume_model_from_wandb latest \
  "$@"
