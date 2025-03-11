import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput)

from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.utils.config import Config


def identity(ego_state, predictions):
    return predictions


class DiffusionPlanner(AbstractPlanner):

    def __init__(
        self,
        config: Config,
        ckpt_path: str,
        past_trajectory_sampling: TrajectorySampling,
        future_trajectory_sampling: TrajectorySampling,
        enable_ema: bool = True,
        device: str = "cpu",
    ):
        """
{
    'config':
        {
            '_target_': 'diffusion_planner.utils.config.Config',
            '_convert_': 'all',
            'args_file': './checkpoints/args.json'
        },
    'ckpt_path': './checkpoints/model.pth',
    'past_trajectory_sampling':
        {
            '_target_': 'nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling',
            '_convert_': 'all',
            'num_poses': 20,
            'time_horizon': 2
        },
    'future_trajectory_sampling':
        {
            '_target_': 'nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling',
            '_convert_': 'all',
            'num_poses': 80,
            'time_horizon': 8
        },
    'device': 'cuda'
}
        """

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"

        self._future_horizon = future_trajectory_sampling.time_horizon  # [s]
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses  # [s]

        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device

        self._planner = Diffusion_Planner(config)

        self.data_processor = DataProcessor(config)

        self.observation_normalizer = config.observation_normalizer
        self.npc_trajectories = None

    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"

    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        initialization = self._simulation.initialize()
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path)

            if self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']
            # use for ddp
            model_state_dict = {
                k[len("module."):]: v
                for k, v in state_dict.items()
                if k.startswith("module.")
            }
            self._planner.load_state_dict(model_state_dict)
        else:
            print("load random model")

        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(
            self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        """ model_inputs
neighbor_agents_past: torch.Size([1, 32, 21, 11])
ego_current_state: torch.Size([1, 4])
static_objects: torch.Size([1, 5, 10])
lanes: torch.Size([1, 70, 20, 12])
lanes_speed_limit: torch.Size([1, 70, 1])
lanes_has_speed_limit: torch.Size([1, 70, 1])
route_lanes: torch.Size([1, 25, 20, 12])
route_lanes_speed_limit: torch.Size([1, 25, 1])
route_lanes_has_speed_limit: torch.Size([1, 25, 1])
        """
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(
            history, traffic_light_data, self._map_api,
            self._route_roadblock_ids, self._device)

        return model_inputs

    def outputs_to_trajectory(
            self, outputs: Dict[str, torch.Tensor],
            ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:
        # a = outputs['prediction'] # [B, P, V_future, 4] = (1, 11, 80, 4)z
        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(
            np.float64)  # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[...,
                                                                   None]  # T, 1
        predictions = np.concatenate([predictions[..., :2], heading],
                                     axis=-1)  # T, 3

        states = transform_predictions_to_states(predictions, ego_state_history,
                                                 self._future_horizon,
                                                 self._step_interval)

        return states

    def compute_planner_trajectory(
            self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)
        """ inputs
neighbor_agents_past: torch.Size([1, 32, 21, 11])
ego_current_state: torch.Size([1, 4])
static_objects: torch.Size([1, 5, 10])
lanes: torch.Size([1, 70, 20, 12])
lanes_speed_limit: torch.Size([1, 70, 1])
lanes_has_speed_limit: torch.Size([1, 70, 1])
route_lanes: torch.Size([1, 25, 20, 12])
route_lanes_speed_limit: torch.Size([1, 25, 1])
route_lanes_has_speed_limit: torch.Size([1, 25, 1])
        """
        inputs = self.observation_normalizer(inputs)
        """
outputs: Dict
    {
        ...
        [training-only] "score": Predicted future states, [B, P, 1 + V_future, 4]
        [inference-only] "prediction": Predicted future states, [B, P, V_future, 4] = (1, 11, 80, 4)
        ...
    }
        """
        _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(
                outputs, current_input.history.ego_states))


        ######### 추가한 부분 ########
        npc_predictions = outputs['prediction'][
            0, 1:].detach().cpu().numpy().astype(
                np.float64)  # [P, V_future, 4] = (10, 80, 4)
        npc_headings = np.arctan2(npc_predictions[:, :, 3],
                                  npc_predictions[:, :,
                                                  2])[...,
                                                      None]  # [P, V_future, 1]
        npc_predictions = np.concatenate(
            [npc_predictions[..., :2], npc_headings],
            axis=-1)  # [P, V_future, 3]4
        self.npc_trajectories = {}
        for i in range(npc_predictions.shape[0]):
            self.npc_trajectories[i] = InterpolatedTrajectory(
                transform_predictions_to_states(
                    npc_predictions[i], current_input.history.ego_states,
                    self._future_horizon, self._step_interval))
        return trajectory
