#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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
  export TORCHELASTIC_ERROR_FILE="/tmp/torchelastic_error_${RUN_ID}.json"
fi

RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"

LOCAL_TRAIN_SET_PATH="/workspace/local_shards_v1"
LOCAL_TRAIN_SET_LIST_PATH="/workspace/local_shards_v1/diffusion_planner_training.json"

# ====== 여기만 너 상황에 맞게 바꾸면 됨 ======
PRETRAIN_NAME="w_integ_loss_thres_1_p_sat_0.25"   # torch_run_ceph_local.sh에서 썼던 --name
FT_NAME="ft_freeze_encoder_wosac_${RUN_ID}"       # 새 fine-tune 실험 이름
RESUME_ALIAS="latest"                             # 현재 코드가 latest만 허용
# ==============================================

export DP_ENABLE_CPU_MONITOR=0

"$RUN_PYTHON_PATH" "${PY_ARGS[@]}" -m torch.distributed.run \
  --nnodes 1 --nproc-per-node 6 --standalone \
  "${TORCHRUN_LOG_ARGS[@]}" \
  train_predictor.py \
    --train_set "$LOCAL_TRAIN_SET_PATH"/ \
    --train_set_list "$LOCAL_TRAIN_SET_LIST_PATH" \
    --name "w_integ_loss_thres_1_p_sat_0.25_ft_test" \
    --load_name "w_integ_loss_thres_1_p_sat_0.25" \
    --resume_wandb_model_name latest \
    --resume_model_only True \
    --freeze_encoder_local False \
    --freeze_encoder_global False \
    --batch_size 1536 \
    --warm_up_epoch 0 \
    --train_epochs 275 \
    --learning_rate 4e-4 \
    --min_learning_rate 2e-5 \
    --pose_based False \
    --profile_feasible False \
    --use_feasible True \
    --use_feasible_dl False \
    --use_feasible_filter False \
    --feasible_grad_to_dit True \
    --feasible_stride_dt 0.1 \
    --num_workers 3 \
    --max_grad_norm 1.0 \
    --prefetch_factor 8 \
    --feasible_learn_noise_thresh 1.0 \
    --auto_tune_aux_weights false \
    --use_amortized_diffusion true \
    --p_sat 0.0 \
    --w_dir 1.0 \
