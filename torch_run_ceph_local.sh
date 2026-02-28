#!/usr/bin/env bash
set -Eeuo pipefail

# 스크립트가 어디서 실행되든, 이 파일이 있는 폴더로 이동(상대 경로 안전)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# (ADD) OOM 상황에서 "학습 런처"가 먼저 죽기 쉽게(best-effort)
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

# -------------------------
# User Configuration
# -------------------------
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"

# (원본) CephRBD 경로
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"

# (로컬) 학습에 사용할 경로
LOCAL_TRAIN_SET_PATH="/workspace/local_shards_v_world"
LOCAL_TRAIN_SET_LIST_PATH="/workspace/local_shards_v_world/diffusion_planner_training.json"

# 복사 강제 옵션: 1이면 항상 다시 복사
FORCE_REBUILD=0

## 로컬 데이터 준비(복사 1회 + 로컬 리스트 생성)
"$RUN_PYTHON_PATH" tools/prepare_local_train_set.py \
  --src_root "$TRAIN_SET_PATH" \
  --src_list "$TRAIN_SET_LIST_PATH" \
  --dst_root "$LOCAL_TRAIN_SET_PATH" \
  --dst_list "$LOCAL_TRAIN_SET_LIST_PATH" \
  --force_rebuild "$FORCE_REBUILD" \
  --num_workers 24
 python /mnt/nuplan/projects/Diffusion-Planner/add_control_to_npz.py --skip_sample_keys
"$RUN_PYTHON_PATH" add_control_to_npz.py --skip_sample_keys --workers 24
#
#export DP_ENABLE_CPU_MONITOR=0
#
#### ---- 학습은 로컬 데이터로 ----
#"$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m torch.distributed.run \
#  --nnodes 1 --nproc-per-node 6 --standalone \
#  "${TORCHRUN_LOG_ARGS[@]}" \
#  train_predictor.py \
#    --train_set "$LOCAL_TRAIN_SET_PATH"/ \
#    --train_set_list "$LOCAL_TRAIN_SET_LIST_PATH" \
#    --name "w_integ_loss_thres_1_p_sat_0.25_no_lane_summary_yaw_weight_2_no_amor" \
#    --batch_size 1536 \
#    --learning_rate 1e-3 \
#    --min_learning_rate 1e-6 \
#    --pose_based False \
#    --profile_feasible False \
#    --use_feasible True \
#    --use_feasible_dl False \
#    --use_feasible_filter False \
#    --feasible_grad_to_dit True \
#    --feasible_stride_dt 0.1 \
#    --num_workers 3 \
#    --max_grad_norm 1.0 \
#    --prefetch_factor 8 \
#    --feasible_learn_noise_thresh 1.0 \
#    --p_sat 0.25 \
#    --w_dir 1.0 \
#    --w_int_min 0.05 \
#    --w_int_max 2.0 \
#    --w_const 0.0 \
#    --lane_summary_num 0 \
#    --use_amortized_diffusion False
