#!/usr/bin/env bash
set -Eeuo pipefail
export WANDB_DEBUG=1   # ← 여기 추가
export PYTHONUNBUFFERED=1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOG_DIR=/mnt/nuplan/logs/$RUN_ID
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


###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"   # 디렉터리 자체는 유지, 내용만 비움
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
###################################


"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run \
--nnodes 1 --nproc-per-node 6 --standalone --log_dir "$LOG_DIR" \
--redirects 3 --tee "$TEE" \
 train_predictor.py \
  --train_set "$TRAIN_SET_PATH"/ \
  --train_set_list "$TRAIN_SET_LIST_PATH" \
  --name "wosac_test_full" \
  --batch_size 1344 \
  --learning_rate 3e-4
