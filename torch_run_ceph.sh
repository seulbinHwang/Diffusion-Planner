#!/usr/bin/env bash
set -Eeuo pipefail

if [[ -w /proc/self/oom_score_adj ]]; then
  ( echo 500 > /proc/self/oom_score_adj ) 2>/dev/null || true
fi
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

export WANDB_DEBUG=0
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_DEVICE_MAX_CONNECTIONS=32
export PYTHONWARNINGS="ignore:nvfuser is no longer supported in torch script:UserWarning"

RUN_ID=$(date +%Y%m%d-%H%M%S)

DEBUG_LOG=0

TORCHRUN_LOG_ARGS=()
PY_ARGS=()

if (( DEBUG_LOG )); then
  LOG_DIR="/mnt/nuplan/logs/$RUN_ID"
  mkdir -p "$LOG_DIR"

  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT

  export TORCH_SHOW_CPP_STACKTRACES=1
  export PYTHONFAULTHANDLER=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_DISABLE_ADDR2LINE=1

  export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"

  export PYTHONUNBUFFERED=1
  PY_ARGS=(-u -X faulthandler)

  TORCHRUN_LOG_ARGS=(--log_dir "$LOG_DIR" --redirects 3 --tee 3)
else
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT

  unset TORCH_SHOW_CPP_STACKTRACES || true
  unset PYTHONFAULTHANDLER || true
  unset TORCH_NCCL_ASYNC_ERROR_HANDLING || true
  unset TORCH_DISABLE_ADDR2LINE || true

  export TORCHELASTIC_ERROR_FILE="/tmp/torchelastic_error_${RUN_ID}.json"

  unset PYTHONUNBUFFERED || true
  PY_ARGS=()

  TORCHRUN_LOG_ARGS=()
fi

printf "[ENV] %-28s %s\n" "DEBUG_LOG:" "$DEBUG_LOG"
printf "[ENV] %-28s %s\n" "TORCHELASTIC_ERROR_FILE:" "${TORCHELASTIC_ERROR_FILE-<unset>}"
printf "[ENV] %-28s %s\n" "CUDA_DEVICE_MAX_CONNECTIONS:" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"

RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"

# ============================================================
# ✅ (ADD) 로컬 shard 모드
# ============================================================
USE_LOCAL_SHARDS=1
LOCAL_SHARD_ROOT="/workspace/local_shards_v1"
LOCAL_SHARD_MANIFEST="$LOCAL_SHARD_ROOT/meta/manifest.json"

SRC_TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"
SRC_TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"

if (( USE_LOCAL_SHARDS )); then
  mkdir -p "$LOCAL_SHARD_ROOT"
  echo "[LOCAL_SHARDS] df -T $LOCAL_SHARD_ROOT"
  df -T "$LOCAL_SHARD_ROOT" || true

  if [[ ! -f "$LOCAL_SHARD_MANIFEST" ]]; then
    echo "[LOCAL_SHARDS] manifest not found -> build shards"
    "$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m tools.build_local_shards \
      --src_data_dir "$SRC_TRAIN_SET_PATH" \
      --src_list_json "$SRC_TRAIN_SET_LIST_PATH" \
      --out_root "$LOCAL_SHARD_ROOT" \
      --shard_size 2048 \
      --world_size 6 \
      --seed 3407 \
      --predicted_neighbor_num 448 \
      --use_agent_route_lane_order false \
      --force_rebuild true \
      --progress_interval_min 5 \
      --build_workers 24

  else
    echo "[LOCAL_SHARDS] manifest exists -> skip build: $LOCAL_SHARD_MANIFEST"
  fi

  TRAIN_SET_PATH="$LOCAL_SHARD_ROOT"
  TRAIN_SET_LIST_PATH="$LOCAL_SHARD_MANIFEST"
else
  TRAIN_SET_PATH="$SRC_TRAIN_SET_PATH"
  TRAIN_SET_LIST_PATH="$SRC_TRAIN_SET_LIST_PATH"
fi

echo "[TRAIN] train_set=$TRAIN_SET_PATH"
echo "[TRAIN] train_set_list=$TRAIN_SET_LIST_PATH"

"$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 6 --standalone \
  "${TORCHRUN_LOG_ARGS[@]}" \
  train_predictor.py \
    --train_set "$TRAIN_SET_PATH"/ \
    --train_set_list "$TRAIN_SET_LIST_PATH" \
    --name "final_46_savez" \
    --batch_size 1536 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-6 \
    --profile_feasible False \
    --use_feasible False \
    --use_feasible_dl False \
    --use_feasible_filter False \
    --feasible_stride_dt 0.1 \
    --num_workers 4 \
    --prefetch_factor 6
