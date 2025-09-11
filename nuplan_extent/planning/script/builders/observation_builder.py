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

        # 추가: PL 체크포인트 우선 시도, 실패(KeyError) 시 순수 state_dict로 폴백
        try:  # 추가
            lmw = LightningModuleWrapper.load_from_checkpoint(  # 추가
                observation_cfg.checkpoint_path,
                model=torch_module_wrapper,
                strict=False)  # 추가
            model = lmw.model  # 추가
        except KeyError as e:  # 추가
            if "pytorch-lightning_version" not in str(e):  # 추가
                raise  # 추가
            # 순수 state_dict(.pth 등) 로딩 폴백  # 추가
            ckpt = torch.load(observation_cfg.checkpoint_path,
                              map_location="cpu")  # 추가
            state = ckpt.get("state_dict", ckpt)  # 추가

            # 접두사 정리: 'model.' / 'module.' 제거  # 추가
            new_state = {}  # 추가
            for k, v in state.items():  # 추가
                nk = k  # 추가
                if nk.startswith("model."):  # 추가
                    nk = nk[len("model."):]  # 추가
                if nk.startswith("module."):  # 추가
                    nk = nk[len("module."):]  # 추가
                new_state[nk] = v  # 추가

            missing, unexpected = torch_module_wrapper.load_state_dict(
                new_state, strict=False)  # 추가
            model = torch_module_wrapper  # 추가

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
