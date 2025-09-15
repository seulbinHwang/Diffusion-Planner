import logging
import os
from pathlib import Path
from shutil import rmtree
from typing import List, Optional, Union

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
# nuplan/planning/script/run_simulation.py
from nuplan.common.utils.s3_utils import is_s3_path
from nuplan_extent.planning.script.builders.simulation_builder import build_simulations
from nuplan.planning.script.builders.simulation_callback_builder import (
    build_callbacks_worker,
    build_simulation_callbacks,
)
from nuplan.planning.script.utils import run_runners, set_default_path, set_up_common_builder
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
import ray

ray.init(num_gpus=1)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# If set, use the env. variable to overwrite the default dataset and experiment paths
set_default_path()

# If set, use the env. variable to overwrite the Hydra config
print("NUPLAN_HYDRA_CONFIG_PATH:", os.getenv('NUPLAN_HYDRA_CONFIG_PATH'))
# ✅ nuplan-devkit의 simulation 디렉토리를 기본값으로 사용
# nuplan-devkit 루트 (환경변수 이용, fallback은 현재 디렉토리)
DEVKIT_ROOT = Path(os.environ.get("NUPLAN_DEVKIT_ROOT", ".")).resolve()

# simulation config 디렉토리
DEVKIT_SIM_CONFIG_DIR = DEVKIT_ROOT / "nuplan" / "planning" / "script" / "config" / "simulation"

# Hydra config path/name
CONFIG_PATH = os.environ.get("NUPLAN_HYDRA_CONFIG_PATH",
                             str(DEVKIT_SIM_CONFIG_DIR))
CONFIG_NAME = "default_simulation"

print("CONFIG_PATH:", CONFIG_PATH)

# CONFIG_PATH = os.getenv('NUPLAN_HYDRA_CONFIG_PATH', 'config/simulation')
# print("CONFIG_PATH:", CONFIG_PATH)
#
# if os.environ.get('NUPLAN_HYDRA_CONFIG_PATH') is not None:
#     CONFIG_PATH = os.path.join('../../../../', CONFIG_PATH)
#
# if os.path.basename(CONFIG_PATH) != 'simulation':
#     CONFIG_PATH = os.path.join(CONFIG_PATH, 'simulation')
#
# CONFIG_NAME = 'default_simulation'


def run_simulation(
    cfg: DictConfig,
    planners: Optional[Union[AbstractPlanner, List[AbstractPlanner]]] = None
) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Helper function for main to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    :param planners: Pre-built planner(s) to run in simulation. Can either be a single planner or list of planners.
    """
    # Fix random seed
    pl.seed_everything(cfg.seed, workers=True)

    profiler_name = 'building_simulation'
    common_builder = set_up_common_builder(cfg=cfg, profiler_name=profiler_name)

    # Build simulation callbacks
    callbacks_worker_pool = build_callbacks_worker(cfg)
    callbacks = build_simulation_callbacks(cfg=cfg,
                                           output_dir=common_builder.output_dir,
                                           worker=callbacks_worker_pool)

    # Remove planner from config to make sure run_simulation does not receive multiple planner specifications.
    if planners and 'planner' in cfg.keys():
        logger.info(
            'Using pre-instantiated planner. Ignoring planner in config')
        OmegaConf.set_struct(cfg, False)
        cfg.pop('planner')
        OmegaConf.set_struct(cfg, True)

    # Construct simulations
    if isinstance(planners, AbstractPlanner):
        planners = [planners]

    runners = build_simulations(
        cfg=cfg,
        worker=common_builder.worker,
        callbacks=callbacks,
        callbacks_worker=callbacks_worker_pool,
        pre_built_planners=planners,
    )

    if common_builder.profiler:
        # Stop simulation construction profiling
        common_builder.profiler.save_profiler(profiler_name)

    logger.info('Running simulation...')
    run_runners(runners=runners,
                common_builder=common_builder,
                cfg=cfg,
                profiler_name='running_simulation')
    logger.info('Finished running simulation!')


def clean_up_s3_artifacts() -> None:
    """
    Cleanup lingering s3 artifacts that are written locally.
    This happens because some minor write-to-s3 functionality isn't yet implemented.
    """
    # Lingering artifacts get written locally to a 's3:' directory. Hydra changes
    # the working directory to a subdirectory of this, so we serach the working
    # path for it.
    working_path = os.getcwd()
    s3_dirname = "s3:"
    s3_ind = working_path.find(s3_dirname)
    if s3_ind != -1:
        local_s3_path = working_path[:working_path.find(s3_dirname) +
                                     len(s3_dirname)]
        rmtree(local_s3_path)


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME)
def main(cfg: DictConfig) -> None:
    """
    Execute all available challenges simultaneously on the same scenario. Calls run_simulation to allow planner to
    be specified via config or directly passed as argument.
    :param cfg: Configuration that is used to run the experiment.
        Already contains the changes merged from the experiment's config to default config.
    """
    assert cfg.simulation_log_main_path is None, 'Simulation_log_main_path must not be set when running simulation.'

    # Execute simulation with preconfigured planner(s).
    run_simulation(cfg=cfg)

    if is_s3_path(Path(cfg.output_dir)):
        clean_up_s3_artifacts()


if __name__ == '__main__':
    main()
