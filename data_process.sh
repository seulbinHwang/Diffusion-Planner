###################################
# User Configuration Section
###################################
NUPLAN_DATA_PATH="/mnt/nuplan/dataset/nuplan-v1.1/splits/trainval" # nuplan training data path (e.g., "/data/nuplan-v1.1/trainval")
NUPLAN_MAP_PATH="/mnt/nuplan/dataset/maps" # nuplan map path (e.g., "/data/nuplan-v1.1/maps")

TRAIN_SET_PATH="/mnt/nuplan/dataset/processed/" # preprocess training data
###################################

python data_process.py \
--data_path $NUPLAN_DATA_PATH \
--map_path $NUPLAN_MAP_PATH \
--save_path $TRAIN_SET_PATH \
--total_scenarios 1000000 \

