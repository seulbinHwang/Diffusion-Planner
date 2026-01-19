#!/usr/bin/env bash
set -Eeuo pipefail

# (:-0은 “없으면 0”이라는 뜻)
RUN_COUNT="${1:-0}"
echo "[INFO] run_count=${RUN_COUNT}"
###################################
# User Configuration Section
###################################
USER_PATH="/mnt/nuplan"
#WOMD_PATH="${USER_PATH}/womd_v1_3"
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

TRAIN_SET_PATH="${USER_PATH}/dataset/processed"
TRAIN_SET_LIST_PATH="${USER_PATH}/projects/Diffusion-Planner/diffusion_planner_fine_tuning.json"
###################################
# If validation list json is missing, create it from *.npz in TRAIN_SET_PATH
###################################

# ----------------------------
# 진행 상황 출력 주기(초) (공통)
# - 0 또는 음수면 heartbeat / WOSACMetrics / WOSACSubmission 진행 출력 모두 끔
# ----------------------------
export DP_PROGRESS_SEC="${DP_PROGRESS_SEC:-60}"
printf "[ENV] %-28s %s\n" "DP_PROGRESS_SEC:" "${DP_PROGRESS_SEC-<unset>}"


###################################
export PYTHONWARNINGS="ignore::FutureWarning"
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
LOG_DIR="${USER_PATH}/logs/$RUN_ID"
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
export DP_WOSAC_TF_THREADS="${DP_WOSAC_TF_THREADS:-3}"
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



"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run --nnodes 1 --nproc-per-node 6 --standalone --log_dir "$LOG_DIR" --redirects 3 --tee "$TEE" \
 finetune_data_maker.py \
 --port 23001 \
  --eval_set "$TRAIN_SET_PATH" \
  --eval_set_list "$TRAIN_SET_LIST_PATH" \
  --resume_wandb_model_name latest \
  --resume_model_only True \
  --load_name "amortized_fine_tuning_lr_2e5" \
  --name "amortized_fine_tuning_lr_2e5" \
  --eval_method "train" \
  --batch_size 1536 \
  --use_deepspeed True \
  --wosac_sub_is_active False \
  --wosac_metric_is_active False \
  --save_image False \
  --save_video False \
--save_cache_path "/mnt/nuplan/dataset/processed_rollout" \
  --save_inference_data True \
  --finish_when_no_updated_pt False \
  --run_count "$RUN_COUNT" \
  --fine_tune_gen_k 64 \
  --rollout_time_chunk_size 1 \
  --fine_tune_temperature 0.8 \
  --use_recovery True \
  --select_jointly True \
  --recovery_threshold_m 1. \
  --scenario_finish_step 20 \
  --time_step_for_compare 60 \
  --use_data_percent 2 \
  --rollout_number 3 \
  --use_amortized_diffusion True
