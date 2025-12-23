###################################
# User Configuration Section
###################################
export NUPLAN_DATA_ROOT="/home/user/nuplan/dataset" #"REPLACE_WITH_DATA_DIR"  # nuplan dataset absolute path (e.g. "/data")


NUPLAN_DATA_PATH="${NUPLAN_DATA_ROOT}/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="${NUPLAN_DATA_ROOT}/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="${NUPLAN_DATA_ROOT}/processed_compressed" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 12 \
--save_image true

