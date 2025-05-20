export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1

###################################
# User Configuration Section
###################################
# Set environment variables
export NUPLAN_DEVKIT_ROOT="/home/user/PycharmProjects/nuplan-devkit"  #"REPLACE_WITH_NUPLAN_DEVIKIT_DIR"  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
export NUPLAN_DATA_ROOT="/home/user/PycharmProjects/Diffusion-Planner/data" #"REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")
export NUPLAN_MAPS_ROOT="/home/user/PycharmProjects/Diffusion-Planner/data/nuplan-v1.1/maps" #"REPLACE_WITH_MAPS_DIR" # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
export NUPLAN_EXP_ROOT="/home/user/PycharmProjects/Diffusion-Planner/data/nuplan-v1.1/exp" #"REPLACE_WITH_EXP_DIR" # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")

# Dataset split to use
# Options:
#   - "test14-random" # 14개 시나리오 # 각 유형별로 무작위로 20개의 시나리오를 선택해 평가
#   - "test14-hard" # 14개 시나리오 # 각 유형에서 100회 시뮬레이션 후 성능이 가장 낮은 20개의 시나리오(“롱테일”)를 선택해 스트레스 테스트를 진행
#   - "val14" # 14개 시나리오 # 전체 검증 세트를 대상으로 평가
SPLIT="test14-random"  # e.g., "val14"

# Challenge type
# Options:
#   - "closed_loop_nonreactive_agents"
#   - "closed_loop_reactive_agents"
CHALLENGE="closed_loop_reactive_agents" # e.g., "closed_loop_reactive_agents"
###################################


BRANCH_NAME=diffusion_planner_release
ARGS_FILE=/home/user/PycharmProjects/Diffusion-Planner/checkpoints/args.json
CKPT_FILE=/home/user/PycharmProjects/Diffusion-Planner/checkpoints/model.pth

if [ "$SPLIT" == "val14" ]; then
    SCENARIO_BUILDER="nuplan"
else
    SCENARIO_BUILDER="nuplan_challenge"
fi
echo "Processing $CKPT_FILE..."
FILENAME=$(basename "$CKPT_FILE")
FILENAME_WITHOUT_EXTENSION="${FILENAME%.*}"
# diffusion_planner/planner/planner.py 의 DiffusionPlanner
PLANNER=diffusion_planner
# print PLANNER
echo "PLANNER: $PLANNER"
# $ARGS_FILE /home/user/PycharmProjects/Diffusion-Planner/checkpoints/args.json
# $SCENARIO_BUILDER nuplan_challenge
# $SPLIT test14-random
python $NUPLAN_DEVKIT_ROOT/nuplan/planning/script/run_simulation.py \
    +simulation=$CHALLENGE \
    planner=$PLANNER \
    planner.diffusion_planner.config.args_file=$ARGS_FILE \
    planner.diffusion_planner.ckpt_path=$CKPT_FILE \
    scenario_builder=$SCENARIO_BUILDER \
    scenario_filter=$SPLIT \
    experiment_uid=$PLANNER/$SPLIT/$BRANCH_NAME/${FILENAME_WITHOUT_EXTENSION}_$(date "+%Y-%m-%d-%H-%M-%S") \
    verbose=true \
    worker=ray_distributed \
    worker.threads_per_node=128 \
    distributed_mode='SINGLE_NODE' \
    number_of_gpus_allocated_per_simulation=0.15 \
    enable_simulation_progress_bar=true \
    hydra.searchpath="[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments  ]"

#  nuplan.planning.script.config.simulation