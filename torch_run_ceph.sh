#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================
# (ADD) OOM 상황에서 "학습 런처"가 먼저 죽기 쉽게(best-effort)
# - 권한/환경에 따라 실패할 수 있으니 실패해도 무시합니다.
# ============================================================
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

# 사용자가 이 값만 0/1로 바꿔서 모드 선택
DEBUG_LOG=0   # 1: 상세 디버그, 0: 빠른 학습(파일 로그 최소)

# -------------------------
# 로그/디버그 옵션 분기
# -------------------------
TORCHRUN_LOG_ARGS=()
PY_ARGS=()

if (( DEBUG_LOG )); then
  # ✅ 디버그 모드: 파일 로그 허용 (기존 의도 유지)
  LOG_DIR="/mnt/nuplan/logs/$RUN_ID"
  mkdir -p "$LOG_DIR"

  export NCCL_DEBUG=INFO
  export NCCL_DEBUG_SUBSYS=INIT

  # 파이썬 디버그(에러 때 도움)
  export TORCH_SHOW_CPP_STACKTRACES=1
  export PYTHONFAULTHANDLER=1
  export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
  export TORCH_DISABLE_ADDR2LINE=1

  # torchelastic 에러 파일(디버그 모드에서는 Ceph에 저장)
  export TORCHELASTIC_ERROR_FILE="$LOG_DIR/torchelastic_error.json"

  # 출력이 많아도 “바로바로” 보이게(디버그 편의)
  export PYTHONUNBUFFERED=1
  PY_ARGS=(-u -X faulthandler)

  # torchrun이 rank별 stdout/stderr를 파일로 저장 + 콘솔에도 출력(디버그 편의)
  TORCHRUN_LOG_ARGS=(--log_dir "$LOG_DIR" --redirects 3 --tee 3)

else
  # ✅ 빠른 학습 모드: "학습 중 파일 로그"는 끔
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

# -------------------------
# User Configuration
# -------------------------
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"

"$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 6 --standalone \
  "${TORCHRUN_LOG_ARGS[@]}" \
  train_predictor.py \
    --train_set "$TRAIN_SET_PATH"/ \
    --train_set_list "$TRAIN_SET_LIST_PATH" \
    --name "final_48_synchronize" \
    --batch_size 1536 \
    --learning_rate 1e-3 \
    --min_learning_rate 1e-6 \
    --profile_feasible False \
    --use_feasible False \
    --use_feasible_dl False \
    --use_feasible_filter False \
    --feasible_stride_dt 0.1
