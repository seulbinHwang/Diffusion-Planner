from typing import cast

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.script.builders.model_builder import build_torch_module_wrapper
from nuplan.planning.script.builders.utils.utils_type import is_TorchModuleWrapper_config
from nuplan.planning.simulation.observation.abstract_observation import AbstractObservation
from nuplan.planning.training.modeling.lightning_module_wrapper import LightningModuleWrapper


def build_observations(observation_cfg: DictConfig,
                       scenario: AbstractScenario) -> AbstractObservation:
    """
    Instantiate observations
    :param observation_cfg: config of a planner
    :param scenario: scenario
    :return AbstractObservation
    """
    if is_TorchModuleWrapper_config(observation_cfg):
        # Build model and feature builders needed to run an ML model in simulation
        torch_module_wrapper = build_torch_module_wrapper(
            observation_cfg.model_config)
        # 삭제
        # model = LightningModuleWrapper.load_from_checkpoint(
        #     observation_cfg.checkpoint_path, model=torch_module_wrapper
        # ).model

        # : PL 체크포인트 우선 시도, 실패(KeyError) 시 순수 state_dict로 폴백
        try:
            lmw = LightningModuleWrapper.load_from_checkpoint(
                observation_cfg.checkpoint_path,
                model=torch_module_wrapper,
                strict=False)
            model = lmw.model
        except KeyError as e:
            if "pytorch-lightning_version" not in str(e):
                raise
            # 순수 state_dict(.pth 등) 로딩 폴백
            ckpt = torch.load(observation_cfg.checkpoint_path,
                              map_location="cpu")
            state = ckpt.get("state_dict", ckpt)

            # 접두사 정리: 'model.' / 'module.' 제거
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("model."):
                    nk = nk[len("model."):]
                if nk.startswith("module."):
                    nk = nk[len("module."):]
                new_state[nk] = v

            missing, unexpected = torch_module_wrapper.load_state_dict(
                new_state, strict=False)
            model = torch_module_wrapper

        # Remove config elements that are redundant to MLPlanner
        config = observation_cfg.copy()
        OmegaConf.set_struct(config, False)
        config.pop('model_config')
        config.pop('checkpoint_path')
        OmegaConf.set_struct(config, True)

        observation: AbstractObservation = instantiate(config,
                                                       model=model,
                                                       scenario=scenario)
    else:
        observation = cast(AbstractObservation,
                           instantiate(observation_cfg, scenario=scenario))

    return observation
