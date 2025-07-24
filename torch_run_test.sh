#export CUDA_VISIBLE_DEVICES=0,1,2,3 #,4,5,6,7

###################################
# User Configuration Section
###################################
RUN_PYTHON_PATH="/home/hsb/miniforge/envs/diffusion_planner/bin/python" #"/mnt/nuplan/miniforge/envs/diffusion_planner/bin/python" # python path

# Set training data path
TRAIN_SET_PATH="/home/hsb/nuplan/dataset/processed" #"/mnt/nuplan/dataset/processed/" # preprocess data using data_process.sh
TRAIN_SET_LIST_PATH="/home/hsb/PycharmProjects/Diffusion-Planner/diffusion_planner_training.json" # "/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
# "/home/hsb/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json"
###################################

$RUN_PYTHON_PATH -m torch.distributed.run --nnodes 1 --nproc-per-node 1 --standalone train_predictor.py \
--train_set  $TRAIN_SET_PATH \
--train_set_list  $TRAIN_SET_LIST_PATH \

