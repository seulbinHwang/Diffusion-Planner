#!/usr/bin/env bash
set -Eeuo pipefail

# ---------------- CPU split config ----------------
CPUSET="0-127"
NUM_CPUS=128
export DP_CPUSET="0-127"

# (Optional) Prevent libraries from spawning too many threads
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

# Pin this script (current shell) to CPUSET so child processes inherit it
if ! command -v taskset >/dev/null 2>&1; then
  echo "[ERROR] Could not find the 'taskset' command. (Please install util-linux)" >&2
  exit 1
fi
taskset -cp "${CPUSET}" $$ >/dev/null

echo "[CPU] Using CPUSET=${CPUSET}, NUM_CPUS=${NUM_CPUS}"
# ----------------------------------------------

# (:-0 means “use 0 if missing”)
RUN_COUNT="${1:-0}"
echo "[INFO] run_count=${RUN_COUNT}"
###################################
# User Configuration Section
###################################
USER_PATH="/media/user"
#WOMD_PATH="${USER_PATH}/womd_v1_3"
WOMD_PATH="${USER_PATH}/D/dataset"
# `~/womd_v1_3/processed_womd_final/validation`

# Automatically use the python from the current environment entered via `conda run`
resolve_python_in_current_env() {
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    echo "${CONDA_PREFIX}/bin/python"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  return 1
}

RUN_PYTHON_PATH="$(resolve_python_in_current_env || true)"
if [[ -z "${RUN_PYTHON_PATH}" ]]; then
  echo "[ERROR] Could not find python in the current environment. (Check CONDA_PREFIX / PATH)" >&2
  exit 1
fi
echo "[INFO] RUN_PYTHON_PATH=${RUN_PYTHON_PATH}"

EVAL_SET_PATH="${WOMD_PATH}/processed_womd_0124/training"
EVAL_SET_LIST_PATH="${USER_PATH}/E/projects/Diffusion-Planner/processed_womd_0124_training.json"
###################################
# If the validation list json is missing, create it from *.npz in EVAL_SET_PATH
###################################
ensure_eval_set_list_json() {
  local validation_dir="$1"
  local json_path="$2"
  local python_bin="$3"

  if [[ -f "$json_path" ]]; then
    echo "[INFO] Found existing eval_set_list json: $json_path"
    return 0
  fi

  if [[ ! -d "$validation_dir" ]]; then
    echo "[ERROR] EVAL_SET_PATH does not exist or is not a directory: $validation_dir" >&2
    return 1
  fi

  mkdir -p "$(dirname "$json_path")"

  # Pass via env for safe handling (paths with spaces are safe)
  export _DP_VALIDATION_DIR="$validation_dir"
  export _DP_VALIDATION_JSON="$json_path"

  # (Optional) Explicit taskset here too (script is already pinned; this is redundant but safe)
  taskset -c "${CPUSET}" "$python_bin" - <<'PY'
import glob
import json
import os

validation_dir = os.environ["_DP_VALIDATION_DIR"]
json_path = os.environ["_DP_VALIDATION_JSON"]

# Collect only *.npz directly under validation_dir (non-recursive)
npz_paths = glob.glob(os.path.join(validation_dir, "*.npz"))

# Keep only file names (including extension)
npz_names = [os.path.basename(p) for p in npz_paths]
npz_names.sort()

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(npz_names, f, indent=2, ensure_ascii=False)

print(f"[INFO] Created eval_set_list json: {json_path} (num_files={len(npz_names)})")
PY

  unset _DP_VALIDATION_DIR
  unset _DP_VALIDATION_JSON
}

ensure_eval_set_list_json "$EVAL_SET_PATH" "$EVAL_SET_LIST_PATH" "$RUN_PYTHON_PATH"


# ----------------------------
# Progress print interval (sec) (common)
# - If 0 or negative, disable progress output for heartbeat / WOSACMetrics / WOSACSubmission
# ----------------------------
export DP_PROGRESS_SEC="${DP_PROGRESS_SEC:-60}"
printf "[ENV] %-28s %s\n" "DP_PROGRESS_SEC:" "${DP_PROGRESS_SEC-<unset>}"


###################################
export PYTHONWARNINGS="ignore::FutureWarning"
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_DEBUG=1
export PYTHONUNBUFFERED=1

# Auto-detect CUDA_HOME
if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  elif [[ -d /usr/local/cuda ]]; then
    CUDA_HOME="/usr/local/cuda"
  else
    CUDA_HOME="$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -n 1 || true)"
  fi
fi

if [[ -n "${CUDA_HOME:-}" && -d "$CUDA_HOME" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  echo "[INFO] CUDA_HOME=$CUDA_HOME"
else
  echo "[WARN] Could not auto-detect CUDA_HOME. Please set CUDA_HOME manually."
fi

RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR="${WOMD_PATH}/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# Debug: python/CPP stacks, NCCL early-fail
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISABLE_ADDR2LINE=1

# Increase the maximum number of concurrent CUDA “connections/queues” maintained by the runtime on one GPU
export CUDA_DEVICE_MAX_CONNECTIONS=32

# ----------------------------
# WOSAC (TensorFlow) CPU tuning
# - If already set in the environment, keep it
# - Otherwise set defaults to reduce CPU thread/process contention
# ----------------------------
export DP_WOSAC_TF_THREADS="${DP_WOSAC_TF_THREADS:-3}"
export DP_WOSAC_CPU_FRACTION="${DP_WOSAC_CPU_FRACTION:-0.6}"

printf "[ENV] %-28s %s\n" "DP_WOSAC_TF_THREADS:"   "${DP_WOSAC_TF_THREADS-<unset>}"
printf "[ENV] %-28s %s\n" "DP_WOSAC_CPU_FRACTION:" "${DP_WOSAC_CPU_FRACTION-<unset>}"
printf "[ENV] %-28s %s\n" "CUDA_DEVICE_MAX_CONNECTIONS:" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"

DEBUG_LOG=1   # 1: verbose debug, 0: normal

if (( DEBUG_LOG )); then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=3
else
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=1
fi

# Make torchelastic save child (rank0) errors as JSON (important)
export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"


# Main run (explicit taskset again; redundant but ensures CPU affinity)
taskset -c "${CPUSET}" \
"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 1 --standalone \
  --log_dir "$LOG_DIR" --redirects 3 --tee "$TEE" \
  eval_predictor.py \
  --port 23001 \
  --eval_set "$EVAL_SET_PATH" \
  --eval_set_list "$EVAL_SET_LIST_PATH" \
  --resume_wandb_model_name latest \
  --resume_model_only True \
  --load_name "wosac_test_full" \
  --name "wosac_test_full" \
  --eval_method "validation" \
  --batch_size 256 \
  --use_deepspeed True \
  --wosac_sub_is_active False \
  --wosac_metric_is_active True \
  --save_image False \
  --save_video False \
  --finish_when_no_updated_pt False \
  --validate_scenario_rollouts False \
  --run_count "$RUN_COUNT" \
  --total_save_image_trial_num 1 \
  --rollout_time_chunk_size 1 \
  --rollout_number 32 \
  --use_data_percent 100 \
  --use_amortized_diffusion True \
  --do_data_statistics True

#  --resume_local_path_model_path "/mnt/nuplan/projects/Diffusion-Planner/training_log/new-adaLN-weighted-loss-h-two/2025-09-21-13:25:45" \
