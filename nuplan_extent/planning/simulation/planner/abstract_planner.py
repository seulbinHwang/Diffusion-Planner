from typing import List, Optional, Dict

from dataclasses import dataclass
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData
from nuplan.planning.training.preprocessing.features.abstract_model_feature import (
    FeatureDataType,)


@dataclass(frozen=True)
class HorizonPlannerInitialization:
    """
    This class represents required data to initialize a planner.
    """

    # The state which was achieved by expert driver in a scenario
    route_roadblock_ids: List[str]  # Roadblock ids comprising goal route
    # The mission goal which commonly is not achievable in a single scenario
    mission_goal: StateSE2
    map_api: AbstractMap  # The API towards maps.
    expert_goal_state: Optional[StateSE2] = None
    npc_route_roadblock_ids: Optional[Dict[str, List[str]]] = None
    scenario: Optional[AbstractScenario] = None


@dataclass(frozen=True)
class PlannerInput:
    """
    Input to a planner for which a trajectory should be computed.
    """

    iteration: SimulationIteration  # Iteration and time in a simulation progress
    history: SimulationHistoryBuffer  # Rolling buffer containing past observations and states.
    traffic_light_data: Optional[
        List[TrafficLightStatusData]]  # The traffic light status data
    # diffusion_agents_track_tokens: List[
    #     str]  # The track tokens of diffusion agents to be considered
    ego_agent_next_11_dim: Optional[
        FeatureDataType] = None  # (interpol_num, 11)
    ego_agent_future_11_dim: Optional[
        FeatureDataType] = None  # (future_len, 11)
