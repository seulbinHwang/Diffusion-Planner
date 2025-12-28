#!/usr/bin/env bash
set -Eeuo pipefail
###################################
# User Configuration Section
###################################
USER_PATH="/home/user"
WOMD_PATH="${USER_PATH}/womd_v1_3"
RUN_PYTHON_PATH="${USER_PATH}/miniforge3/envs/diffusion_planner/bin/python"
# `~/womd_v1_3/processed_womd_final/validation`
VALIDATION_SET_PATH="${WOMD_PATH}/processed_womd_final/validation" 
VALIDATION_SET_LIST_PATH="${USER_PATH}/PycharmProjects/Diffusion-Planner/diffusion_planner_validation.json"
###################################
# If validation list json is missing, create it from *.npz in VALIDATION_SET_PATH
###################################
ensure_validation_set_list_json() {
  local validation_dir="$1"
  local json_path="$2"
  local python_bin="$3"

  if [[ -f "$json_path" ]]; then
    echo "[INFO] Found existing validation_set_list json: $json_path"
    return 0
  fi

  if [[ ! -d "$validation_dir" ]]; then
    echo "[ERROR] VALIDATION_SET_PATH does not exist or is not a directory: $validation_dir" >&2
    return 1
  fi

  mkdir -p "$(dirname "$json_path")"

  # 안전한 전달을 위해 env로 넘김 (경로에 공백이 있어도 안전)
  export _DP_VALIDATION_DIR="$validation_dir"
  export _DP_VALIDATION_JSON="$json_path"

  "$python_bin" - <<'PY'
import glob
import json
import os

validation_dir = os.environ["_DP_VALIDATION_DIR"]
json_path = os.environ["_DP_VALIDATION_JSON"]

# validation_dir 바로 아래의 *.npz만 수집 (재귀 아님)
npz_paths = glob.glob(os.path.join(validation_dir, "*.npz"))

# "파일명(확장자 포함)"만 추출
npz_names = [os.path.basename(p) for p in npz_paths]
npz_names.sort()

with open(json_path, "w", encoding="utf-8") as f:
    json.dump(npz_names, f, indent=2, ensure_ascii=False)

print(f"[INFO] Created validation_set_list json: {json_path} (num_files={len(npz_names)})")
PY

  unset _DP_VALIDATION_DIR
  unset _DP_VALIDATION_JSON
}

ensure_validation_set_list_json "$VALIDATION_SET_PATH" "$VALIDATION_SET_LIST_PATH" "$RUN_PYTHON_PATH"

###################################
export WANDB_DEBUG=1   # ← 여기 추가
export PYTHONUNBUFFERED=1
export CUDA_HOME="/usr/local/cuda-12.4"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR="${WOMD_PATH}/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

# 디버그: 파이썬/CPP 스택, NCCL 조기실패
export TORCH_SHOW_CPP_STACKTRACES=1
export PYTHONFAULTHANDLER=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISABLE_ADDR2LINE=1
# CUDA 런타임이 한 GPU에서 동시에 유지하는 “연결/큐(=스케줄링 슬롯)”의 상한을 32로 늘려라
export CUDA_DEVICE_MAX_CONNECTIONS=32
# CPU에서 돌아가는 연산(전처리, 일부 텐서 연산, BLAS 등)의 스레드 수를 컨트롤해서, GPU 학습 중 CPU 과도한 스레드 난립 방지
printf "[ENV] %-28s %s\n" "OMP_NUM_THREADS:"            "${OMP_NUM_THREADS-<unset>}"
printf "[ENV] %-28s %s\n" "CUDA_DEVICE_MAX_CONNECTIONS:" "${CUDA_DEVICE_MAX_CONNECTIONS-<unset>}"
DEBUG_LOG=1   # 1: 상세 디버그, 0: 일반 학습

if (( DEBUG_LOG )); then
  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=3
else
  export NCCL_DEBUG=WARN
  export NCCL_DEBUG_SUBSYS=INIT
  TEE=1
fi

# torchelastic가 자식(rank0) 오류를 JSON으로 저장하게 함 (아주 중요!)
export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"



"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone --log_dir "$LOG_DIR" --redirects 3 --tee "$TEE" \
 eval_predictor.py \
 --port 23001 \
  --validation_set "$VALIDATION_SET_PATH" \
  --validation_set_list "$VALIDATION_SET_LIST_PATH" \
  --resume_wandb_model_name latest \
  --resume_model_only True \
  --load_name "nuplan_womd" \
  --name "nuplan_womd" \
  --batch_size 8 \
  --use_deepspeed True
#  --resume_local_path_model_path "/mnt/nuplan/projects/Diffusion-Planner/training_log/new-adaLN-weighted-loss-h-two/2025-09-21-13:25:45" \
