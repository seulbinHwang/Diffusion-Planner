#!/usr/bin/env bash
set -Eeuo pipefail

export PYTHONUNBUFFERED=1

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"   # 디렉터리 자체는 유지, 내용만 비움
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
TRAIN_SET_NAME="processed_route_with_small"
TRAIN_JSON_PATH="${TRAIN_SET_NAME}_json"
###################################
#
# ---------------- Preflight clean-up ----------------
echo "[Preflight] Cleaning previous artifacts..."
#
## 1) 학습 리스트 JSON 삭제
#if [ -f "$TRAIN_SET_LIST_PATH" ]; then
#  echo " - Removing $TRAIN_SET_LIST_PATH"
#  rm -f "$TRAIN_SET_LIST_PATH"
#else
#  echo " - No existing training list to remove."
#fi
#
## 2) processed 디렉터리 내용물 전부 삭제(숨김 포함), 디렉터리는 유지
#if [ -d "$TRAIN_SET_PATH" ]; then
#  echo " - Removing all contents under $TRAIN_SET_PATH"
#  # 안전 가드: 절대 루트나 빈 문자열은 청소 금지
#  if [[ "$TRAIN_SET_PATH" == "/" || "$TRAIN_SET_PATH" == "" ]]; then
#    echo "Refusing to clean unsafe directory: '$TRAIN_SET_PATH'"; exit 1
#  fi
#  # 자식만 삭제(숨김 포함), 디렉터리 자체는 유지
#  find "$TRAIN_SET_PATH" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
#else
#  echo " - Creating $TRAIN_SET_PATH"
#  mkdir -p "$TRAIN_SET_PATH"
#fi

echo "[Preflight] Done."
# ----------------------------------------------------

#echo "Start downloading diffusion_planner_training.json"
#nubescli download \
#    labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_JSON_PATH}/diffusion_planner_training.json \
#    "$TRAIN_SET_LIST_PATH" \
#    --no-progress
#echo "Finish downloading diffusion_planner_training.json"

#echo "Start downloading processed dataset"
#nubescli dir-download \
#    labs-mlops/ad/research/pnc/hsb/dataset/${TRAIN_SET_NAME} \
#    "$TRAIN_SET_PATH" \
#    -j "$(nproc)" \
#    -s \
#    --no-progress
#echo "Finish downloading processed dataset"

#export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

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
DEBUG_LOG=0   # 1: 상세 디버그, 0: 일반 학습

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


if false; then
	1.	파이썬을 4개 띄워서 동시에 train_predictor.py 를 실행해.
(각 프로세스는 다른 GPU를 맡게 됨)
	2.	각 프로세스에 환경변수가 자동으로 주입돼:
	•	WORLD_SIZE=4 (전체 프로세스 수)
	•	RANK=0..3 (팀에서 내 전역 등번호)
	•	LOCAL_RANK=0..3 (이 노드에서 내가 쓸 GPU 번호)
	3.	네 코드의 ddp_setup_universal()가 이 값들을 읽어서:
	•	내가 쓸 GPU를 LOCAL_RANK로 지정하고
	•	프로세스 그룹(통신 방)을 열고
	•	“전체 인원 몇 명, 난 몇 번”인지 정보(global_rank, rank, world_size)를 리턴해.
	4.	이후엔 DistributedDataParallel(DDP) 로 감싼 모델이
	•	각 GPU가 서로 다른 데이터 조각을 처리하고
	•	역전파 때 그라디언트를 평균해서
	•	똑같은 업데이트를 동시에 적용해.
fi

"$RUN_PYTHON_PATH" -u -X faulthandler -m torch.distributed.run --nnodes 1 --nproc-per-node 2 --port 23001 --standalone --log_dir "$LOG_DIR" --redirects 3 --tee "$TEE" \
 train_predictor.py \
  --train_set "$TRAIN_SET_PATH"/ \
  --train_set_list "$TRAIN_SET_LIST_PATH" \
  --name "new-lr_schedule-weighted-loss-h-two_1536" \
  --batch_size 1536 \
  "$@"
#  --resume_local_path_model_path "/mnt/nuplan/projects/Diffusion-Planner/training_log/new-adaLN-weighted-loss-h-two/2025-09-21-13:25:45" \
