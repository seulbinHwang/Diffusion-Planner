from __future__ import annotations
from diffusion_planner.utils.validity import add_validity_keys_inplace

from dataclasses import fields as dataclass_fields
from typing import Any, Dict, Type, Optional, List
import numpy as np
from nuplan_extent.planning.training.preprocessing.utils.near_agents import add_near_agents_info_inplace
from nuplan_extent.planning.training.preprocessing.utils.near_agents import get_near_track_token
import torch

from nuplan_extent.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan_extent.planning.simulation.planner.abstract_planner import HorizonPlannerInitialization
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import (
    AbstractFeatureBuilder,
    AbstractModelFeature,
)
from nuplan_extent.planning.training.preprocessing.features.world_model import WorldModelFeature
from diffusion_planner.utils.config import Config
from diffusion_planner.data_process.data_processor import DataProcessor
from diffusion_planner.data_process.utils import convert_data_dict_to_device_tensors
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer


class WorldModelFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self, config: Config) -> None:
        """
        Initializes the RasterFeatureBuilder class.
        """
        self._config = config
        self.data_processor = DataProcessor(config)
        self.observation_normalizer = config.observation_normalizer
        self.unnormalized_features: Optional[Dict[str, np.ndarray]] = None

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "world_model_feature"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return WorldModelFeature  # type: ignore

    def _get_token_to_current_xy(
        self,
        target_track_token: List[Optional[str]]  # len: agents_num
    ) -> Dict[str, np.ndarray]:
        neighbor_agents_past = self.unnormalized_features[
            "neighbor_agents_past"]  # (max_agent_num, time_len, 11)
        neighbor_current_xy = neighbor_agents_past[:,
                                                   -1, :2]  # (max_agent_num, 2)
        token_to_current_xy = {}
        for idx, token in enumerate(target_track_token):
            if token is None:
                continue
            token_to_current_xy[token] = neighbor_current_xy[idx]  # (2,)
        return token_to_current_xy

    def _post_process_unnormalized_features(
            self,
            near_track_token: List[str],  # len: "Pnn 이하의 값"
    ) -> None:

        diff_token_to_future_gt_3_dim: Dict[str, np.ndarray] = {
        }  # (future_len, 3) # "Pnn 이하의 길이"
        neighbor_future_gt_3_dim = self.unnormalized_features.get(
            "neighbor_future_gt_3_dim", None)  # (max_agent_num, future_len, 3)
        if neighbor_future_gt_3_dim is not None:
            for idx, token in enumerate(near_track_token):
                diff_token_to_future_gt_3_dim[token] = neighbor_future_gt_3_dim[
                    idx]  # (future_len, 3)

        neighbor_future_all_gt_3_dim = self.unnormalized_features.get(
            "neighbor_future_all_gt_3_dim",
            None)  # (max_agent_num, future_all_len, 3) # "Pnn 이하의 길이"
        diff_token_to_future_all_gt_3_dim: Dict[str, np.ndarray] = {}
        if neighbor_future_all_gt_3_dim is not None:
            for idx, token in enumerate(near_track_token):
                future_all_gt_3_dim = neighbor_future_all_gt_3_dim[
                    idx]  # (future_all_len, 3)
                diff_token_to_future_all_gt_3_dim[token] = future_all_gt_3_dim

        neighbor_agents_past = self.unnormalized_features[
            "neighbor_agents_past"]  # (max_agent_num, time_len, 11)

        self.unnormalized_features[
            "diff_token_to_future_gt_3_dim"] = diff_token_to_future_gt_3_dim
        self.unnormalized_features[
            "diff_token_to_future_all_gt_3_dim"] = diff_token_to_future_all_gt_3_dim
        self.unnormalized_features[
            "neighbor_agents_past"] = neighbor_agents_past

    def _get_model_input_value_for_world_model_field(
        self,
        model_inputs: Dict[str, torch.Tensor],
        field_name: str,
    ) -> Any:
        """WorldModelFeature 필드에 해당하는 값을 model_inputs에서 찾아 반환합니다.

        대부분의 경우는 `field_name == model_inputs의 key`라서 그대로 꺼내면 됩니다.
        다만 일부 데이터는 같은 의미인데도 key 이름이 바뀌어 들어오는 경우가 있어,
        그런 필드는 "후보 key 목록"을 순서대로 확인해 첫 번째로 찾은 값을 사용합니다.

        예시:
        - driveway_points 필드는 상황에 따라 model_inputs에
          "driveway_points"로 들어오기도 하고, "driveway"로 들어오기도 합니다.
          이 경우 둘 중 먼저 존재하는 key의 텐서를 사용합니다.

        Args:
            model_inputs: 전처리 및 normalize 이후의 입력 딕셔너리.
                value는 torch.Tensor이며 shape는 key마다 다릅니다.
                예)
                - ego_agent_past: (time_len, 11)
                - lanes: (lane_num, lane_len, 12)
                - ego_agent_past_is_valid: (time_len,) bool
            field_name: WorldModelFeature dataclass의 필드명.

        Returns:
            Any:
                - model_inputs에서 값을 찾으면 해당 값(torch.Tensor 등)
                - 못 찾으면 None
        """
        key_candidates_by_field: Dict[str, List[str]] = {
            # validity.py 에서도 driveway는 ["driveway_points", "driveway"] 순으로 확인합니다.
            "driveway_points": ["driveway_points", "driveway"],
        }

        candidates = key_candidates_by_field.get(field_name, [field_name])
        for key in candidates:
            if key in model_inputs:
                return model_inputs[key]
        return None

    def _build_world_model_feature_kwargs(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """WorldModelFeature 생성에 필요한 모든 필드 값을 dict로 구성합니다.

        이 함수의 목표는 다음과 같습니다.
        1) WorldModelFeature에 새 필드가 추가되더라도,
           get_features_from_simulation 쪽에서 "필드 누락"이 생기지 않게 합니다.
        2) builder가 만들 수 없는 값은 None으로 채워서,
           WorldModelFeature 인스턴스가 항상 "모든 필드"를 가지게 합니다.

        동작 규칙:
        - WorldModelFeature dataclass에 정의된 모든 필드를 순회합니다.
        - 각 필드에 대해:
          * model_inputs에서 같은 이름의 key를 찾고,
            필요한 경우 후보 key(예: driveway_points vs driveway)도 확인합니다.
          * 끝까지 못 찾으면 None을 넣습니다.

        Args:
            model_inputs: normalize 이후의 입력 딕셔너리. value: torch.Tensor.

        Returns:
            Dict[str, Any]:
                WorldModelFeature(**kwargs)에 바로 넣을 수 있는 매핑.
                값은 torch.Tensor / np.ndarray / None 중 하나입니다.
        """
        overrides: Dict[str, Any] = {}

        feature_kwargs: Dict[str, Any] = {}
        for field in dataclass_fields(WorldModelFeature):
            field_name = field.name
            if field_name in overrides:
                feature_kwargs[field_name] = overrides[field_name]
                continue
            feature_kwargs[
                field_name] = self._get_model_input_value_for_world_model_field(
                    model_inputs=model_inputs,
                    field_name=field_name,
                )
        return feature_kwargs

    def get_features_from_simulation(
        self,
        current_input: PlannerInput,
        initialization: HorizonPlannerInitialization,
    ) -> WorldModelFeature:
        history_buffer: SimulationHistoryBuffer = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)

        model_inputs: Dict[
            str, torch.Tensor] = self.data_processor.observation_adapter(
                current_input.iteration.index,
                history_buffer,
                traffic_light_data,
                initialization.map_api,
                scenario=initialization.scenario,
                use_route_lanes=initialization.use_route_lanes,
                squeeze=True,
            )

        # (interpol_num, 11)
        model_inputs[
            "ego_agent_next_11_dim"] = current_input.ego_agent_next_11_dim
        # (future_len, 11)
        model_inputs[
            "planner_future_11_dim"] = current_input.planner_future_11_dim

        # List[Optional[str]], (max_agent_num,)
        neighbor_track_token = model_inputs["neighbor_track_token"]
        model_inputs.pop("neighbor_track_token")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_inputs = convert_data_dict_to_device_tensors(model_inputs,
                                                           device,
                                                           squeeze=True)

        add_validity_keys_inplace(model_inputs, missing_policy="none")

        add_near_agents_info_inplace(
            model_inputs,
            predicted_neighbor_num=self._config.predicted_neighbor_num,
        )
        # unnormalized_features 저장 (torch -> numpy)
        self.unnormalized_features = {}
        for key, value in model_inputs.items():
            #
            if value is None:
                # ego_future_gt_is_valid / speed_bump_is_valid / driveway_is_valid / road_edge_is_valid
                continue
            self.unnormalized_features[key] = value.detach().cpu().numpy()

        self.unnormalized_features[
            "neighbor_track_token"] = neighbor_track_token
        # "Pnn 이하의 값"
        near_track_token: List[str] = get_near_track_token(
            neighbor_track_token, self._config.predicted_neighbor_num)
        self.unnormalized_features["near_track_token"] = near_track_token

        # normalize
        model_inputs = self.observation_normalizer(model_inputs)

        # target agent mask 계산 # (max_agent_num,) bool
        self._post_process_unnormalized_features(near_track_token)

        # WorldModelFeature의 "모든 필드"를 채워서 생성
        world_model_feature_kwargs = self._build_world_model_feature_kwargs(
            model_inputs=model_inputs,)
        world_model_feature = WorldModelFeature(**world_model_feature_kwargs)
        return world_model_feature

        for idx in range(max_agent_num):
            if neighbor_track_token[idx] in diffusion_agents_tokens:
                target_agents_mask[idx] = True
        return target_agents_mask

    def get_features_from_scenario(
            self, scenario: "AbstractScenario") -> WorldModelFeature:
        """시뮬레이션 전용 빌더입니다. 학습/오프라인 전처리 경로에서 호출되면 에러를 냅니다."""
        raise NotImplementedError(
            "[WorldModelFeatureBuilder] get_features_from_scenario is not implemented. "
            "This builder is intended for simulation-only. "
            "If you need scenario-based preprocessing, implement this method to "
            "extract features directly from the scenario.")
