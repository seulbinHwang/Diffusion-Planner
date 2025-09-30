# === 진단 모드: 터미널에 원인 직접 보기 ===
set -Eeuo pipefail

# 1) 파이썬 예외/크래시를 상세히 (Signal로 죽어도 트레이스 찍힘)
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=default

# 2) Ray + C++ 프로세스 로그를 터미널로 직출 & 상세 레벨
export RAY_LOG_TO_STDERR=1          # Ray가 자체 로그를 드라이버 STDERR로도 보냄
export GLOG_logtostderr=1           # gcs_server, raylet, plasma_store 같은 C++ 로그를 STDERR로
export RAY_BACKEND_LOG_LEVEL=debug  # Ray 파이썬 로그 레벨
export RAY_CPP_LOG_LEVEL=debug      # Ray C++ 로그 레벨(지원되면)
export RAY_DEDUP_LOGS=0             # 중복 로그 억제 끄기(원인 파악엔 생략 없이 보는게 유리)


#export HYDRA_FULL_ERROR=1 # Hydra 풀스택
#export OC_CAUSE=1 # OmegaConf 에러 원인 체인
#
#export CUDA_LAUNCH_BLOCKING=1 # (이미 쓰는 중) CUDA 디버깅 편의
###################################
# User Configuration Section
###################################
# Set environment variables

#unset RAY_ADDRESS
#ray stop -f >/dev/null 2>&1 || true
HOME_DIR="/home/user" #"/home/user"
PROJECTS_FOLDER="PycharmProjects"
DATASET_DIR=${HOME_DIR}/nuplan
PROJECTS_DIR="${HOME_DIR}/${PROJECTS_FOLDER}"

export NUPLAN_DEVKIT_ROOT="${PROJECTS_DIR}/nuplan-devkit"  #"REPLACE_WITH_NUPLAN_DEVIKIT_DIR"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
# data_root: ${oc.env:NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval
export NUPLAN_DATA_ROOT="${DATASET_DIR}/dataset" #"REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="${DATASET_DIR}/dataset/maps" #"REPLACE_WITH_MAPS_DIR" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="${DATASET_DIR}" #"REPLACE_WITH_EXP_DIR" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

ARGS_FILE="${PROJECTS_DIR}/Diffusion-Planner/checkpoints/args_base.json"
CKPT_FILE="${PROJECTS_DIR}/Diffusion-Planner/checkpoints/npc_model.pth"
# nuplan/planning/script/config/simulation/main_callback/time_callback.yaml
# Dataset split to use
# Options:
#   - "test14-random" # 14개 시나리오 # 각 유형별로 무작위로 20개의 시나리오를 선택해 평가
#   - "test14-hard" # 14개 시나리오 # 각 유형에서 100회 시뮬레이션 후 성능이 가장 낮은 20개의 시나리오(“롱테일”)를 선택해 스트레스 테스트를 진행
#   - "val14" # 14개 시나리오 # 전체 검증 세트를 대상으로 평가
SPLIT="val14"  # e.g., "val14"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="log_replay_diffusion_log_replay_agents" # e.g., "closed_loop_reactive_agents"
###################################
# nuplan/planning/script/experiments/simulation/closed_loop_reactive_agents.yaml

BRANCH_NAME=CHALLENGE


if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
SPLIT="val14_mini"  # e.g., "val14"
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE") # FILENAME: npc_model.pth
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}" # FILENAME_WITHOUT_EXTENSION: npc_model

# $ARGS_FILE /home/user/PycharmProjects/Diffusion-Planner/checkpoints/args.json
# $SCENARIO_BUILDER nuplan_challenge
# $SPLIT test14-random

# nuplan_extent.planning.script.config.simulation
# nuplan_extent/planning/script/config/simulation/callback/simulation_feature_video_callback.yaml

# nuplan_extent.planning.script.experiments.simulation/  log_replay_reactive_diffusion_agents.yaml
# 달라진점: simulation ("log_replay_reactive_diffusion_agents") / observation

stdbuf -oL -eL python nuplan_extent/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    observation.model_config.config.args_file=$ARGS_FILE \
    +callback=simulation_feature_video_callback \
    observation.model_config.ckpt_path=$CKPT_FILE \
    observation.model_config.feature_builders.0.config.args_file=$ARGS_FILE \
    observation.checkpoint_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=sequential \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=1. \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://nuplan_extent.planning.script.experiments, pkg://nuplan_extent.planning.script.config.simulation, pkg://nuplan_extent.planning.script.config.common, pkg://nuplan.planning.script.config.simulation.observation, pkg://diffusion_planner.config.scenario_filter, pkg://nuplan.planning.script.config.common , pkg://diffusion_planner.config, pkg://nuplan.planning.script.experiments ]"

#    worker.threads_per_node=128 \
