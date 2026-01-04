#!/usr/bin/env bash
set -Eeuo pipefail

# (:-0은 “없으면 0”이라는 뜻)
RUN_COUNT="${1:-0}"
echo "[INFO] run_count=${RUN_COUNT}"
###################################
# User Configuration Section
###################################
USER_PATH="/media/user"
#WOMD_PATH="${USER_PATH}/womd_v1_3"
WOMD_PATH="${USER_PATH}/D/dataset"
# `~/womd_v1_3/processed_womd_final/validation`

# ✅ conda run으로 들어온 환경의 python을 자동으로 사용
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
  echo "[ERROR] 현재 환경에서 python을 찾지 못했습니다. (CONDA_PREFIX/ PATH 확인 필요)" >&2
  exit 1
fi
echo "[INFO] RUN_PYTHON_PATH=${RUN_PYTHON_PATH}"

EVAL_SET_PATH="${WOMD_PATH}/processed_womd_final_150_0102/validation"
EVAL_SET_LIST_PATH="${USER_PATH}/E/projects/Diffusion-Planner/diffusion_planner_validation.json"
###################################
# If validation list json is missing, create it from *.npz in EVAL_SET_PATH
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

print(f"[INFO] Created eval_set_list json: {json_path} (num_files={len(npz_names)})")
PY

  unset _DP_VALIDATION_DIR
  unset _DP_VALIDATION_JSON
}

ensure_eval_set_list_json "$EVAL_SET_PATH" "$EVAL_SET_LIST_PATH" "$RUN_PYTHON_PATH"


# ----------------------------
# 진행 상황 출력 주기(초) (공통)
# - 0 또는 음수면 heartbeat / WOSACMetrics / WOSACSubmission 진행 출력 모두 끔
# ----------------------------
export DP_PROGRESS_SEC="${DP_PROGRESS_SEC:-60}"
printf "[ENV] %-28s %s\n" "DP_PROGRESS_SEC:" "${DP_PROGRESS_SEC-<unset>}"


###################################
export PYTHONWARNINGS="ignore::FutureWarning:timm"
export TF_CPP_MIN_LOG_LEVEL=2
export WANDB_DEBUG=1   # ← 여기 추가
export PYTHONUNBUFFERED=1
# CUDA 경로 자동 설정
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
  echo "[WARN] CUDA_HOME를 자동으로 찾지 못했습니다. CUDA_HOME를 수동으로 지정해 주세요."
fi

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
# ----------------------------
# WOSAC(TensorFlow) CPU 튜닝
# - env에 이미 값이 있으면 그 값을 그대로 사용
# - 없으면 기본값을 넣어서 CPU 스레드/프로세스 경쟁을 줄임
# ----------------------------
export DP_WOSAC_TF_THREADS="${DP_WOSAC_TF_THREADS:-2}"
export DP_WOSAC_CPU_FRACTION="${DP_WOSAC_CPU_FRACTION:-0.75}"

printf "[ENV] %-28s %s\n" "DP_WOSAC_TF_THREADS:"   "${DP_WOSAC_TF_THREADS-<unset>}"
printf "[ENV] %-28s %s\n" "DP_WOSAC_CPU_FRACTION:" "${DP_WOSAC_CPU_FRACTION-<unset>}"
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
  --eval_set "$EVAL_SET_PATH" \
  --eval_set_list "$EVAL_SET_LIST_PATH" \
  --resume_wandb_model_name latest \
  --resume_model_only True \
  --load_name "nuplan_womd" \
  --name "nuplan_womd" \
  --eval_method "validation" \
  --batch_size 256 \
  --use_deepspeed True \
  --wosac_sub_is_active False \
  --wosac_metric_is_active True \
  --save_image False \
  --save_video False \
  --finish_when_no_updated_pt True \
  --validate_scenario_rollouts False \
  --run_count "$RUN_COUNT" \
  --total_save_image_trial_num 1 \
  --rollout_time_chunk_size 5

#  --resume_local_path_model_path "/mnt/nuplan/projects/Diffusion-Planner/training_log/new-adaLN-weighted-loss-h-two/2025-09-21-13:25:45" \
