###################################
# User Configuration Section
###################################
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=2
export WOMD_DATA_ROOT="/home/user/womd_v1_3"

###################################

python data_process_womd.py --womd_data_path "$WOMD_DATA_ROOT" --save_folder "final_test" --save_image true
