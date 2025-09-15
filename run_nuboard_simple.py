# Useful imports
import os
from pathlib import Path
import tempfile
import hydra
import sys
# User Configuration Section
HOME_DIR="/home/user/nuplan"
RESULT_FOLDER = "/home/user/nuplan/exp/simulation/log_replay_reactive_diffusion_agents/val14/CHALLENGE/npc_model_2025-09-14-21-02-39" # simulation result absolute path (e.g., "/data/nuplan-v1.1/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2025-01-25-18-29-09")
env_variables = {
    "NUPLAN_DEVKIT_ROOT": "/home/user/PycharmProjects/nuplan-devkit",  # nuplan-devkit absolute path (e.g., "/home/user/nuplan-devkit")
    "NUPLAN_DATA_ROOT": f"{HOME_DIR}/dataset", # nuplan dataset absolute path (e.g. "/data")
    "NUPLAN_MAPS_ROOT": f"{HOME_DIR}/dataset/maps", # nuplan maps absolute path (e.g. "/data/nuplan-v1.1/maps")
    "NUPLAN_EXP_ROOT": f"{HOME_DIR}", # nuplan experiment absolute path (e.g. "/data/nuplan-v1.1/exp")
    "NUPLAN_SIMULATION_ALLOW_ANY_BUILDER":"1"
}

for k, v in env_variables.items():
    os.environ[k] = v

# Location of path with all nuBoard configs
CONFIG_PATH = '../nuplan-devkit/nuplan/planning/script/config/nuboard' # relative path to nuplan-devkit
######################
CONFIG_NAME = 'default_nuboard'

# Initialize configuration management system
hydra.core.global_hydra.GlobalHydra.instance().clear()  # reinitialize hydra if already initialized
hydra.initialize(config_path=CONFIG_PATH)

ml_planner_simulation_folder = RESULT_FOLDER
ml_planner_simulation_folder = [dp for dp, _, fn in os.walk(ml_planner_simulation_folder) if True in ['.nuboard' in x for x in fn]]
print("ml_planner_simulation_folder:", ml_planner_simulation_folder)
"""ml_planner_simulation_folder
['/home/user/nuplan/exp/simulation/log_replay_reactive_diffusion_agents/val14/CHALLENGE/npc_model_2025-09-14-21-02-39']
"""
# Compose the configuration
cfg = hydra.compose(config_name=CONFIG_NAME, overrides=[
    'scenario_builder=nuplan',  # set the database (same as simulation) used to fetch data for visualization
    f'simulation_path={ml_planner_simulation_folder}',  # nuboard file path(s), if left empty the user can open the file inside nuBoard
    'hydra.searchpath=[pkg://diffusion_planner.config.scenario_filter, pkg://diffusion_planner.config, pkg://nuplan.planning.script.config.common, pkg://nuplan.planning.script.experiments]',
    'port_number=6599'
])
#############################
from nuplan_extent.planning.script.run_nuboard import main as main_nuboard

# Run nuBoard
main_nuboard(cfg)