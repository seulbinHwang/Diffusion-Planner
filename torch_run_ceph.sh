#!/usr/bin/env bash
set -euo pipefail

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python"
TRAIN_SET_PATH="/mnt/nuplan/dataset/processed"   # 디렉터리 자체는 유지, 내용만 비움
TRAIN_SET_LIST_PATH="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
###################################
#
# ---------------- Preflight clean-up ----------------
#echo "[Preflight] Cleaning previous artifacts..."
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
#
#echo "[Preflight] Done."
## ----------------------------------------------------
#
#echo "Start downloading diffusion_planner_training.json"
#nubescli download \
#    labs-mlops/ad/research/pnc/hsb/dataset/processed_ego_past_future_json/diffusion_planner_training.json \
#    "$TRAIN_SET_LIST_PATH" \
#    --no-progress
#echo "Finish downloading diffusion_planner_training.json"
#
#echo "Start downloading processed dataset"
#nubescli dir-download \
#    labs-mlops/ad/research/pnc/hsb/dataset/processed_ego_past_future \
#    "$TRAIN_SET_PATH" \
#    -j "$(nproc)" \
#    -w \
#    --no-progress
#echo "Finish downloading processed dataset"

#export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

"$RUN_PYTHON_PATH" -m torch.distributed.run --nnodes 1 --nproc-per-node 4 --standalone train_predictor.py \
  --train_set "$TRAIN_SET_PATH"/ \
  --train_set_list "$TRAIN_SET_LIST_PATH"
