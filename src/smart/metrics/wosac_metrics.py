# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
from typing import Optional
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2


def _assert_scenario_id_matches_rollouts(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
) -> None:
    """원본 시나리오 ID와, 예측 결과에 적힌 시나리오 ID가 같은지 확인합니다.

    왜 필요한가?
    ----------
    - 원본 데이터(정답)와 예측 결과가 서로 다른 시나리오로 섞이면,
      점수가 정상적으로 계산되지 않거나, 엉뚱한 점수가 나올 수 있습니다.

    Args:
        scenario (scenario_pb2.Scenario):
            원본 시나리오(정답이 들어있는 데이터).
            - scenario.scenario_id: str
        scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
            예측 결과 묶음.
            - scenario_rollouts.scenario_id: str

    Returns:
        None

    Raises:
        ValueError:
            두 ID가 다르면 발생합니다.
    """
    scenario_id_from_file: str = str(getattr(scenario, "scenario_id", ""))
    scenario_id_from_rollouts: str = str(
        getattr(scenario_rollouts, "scenario_id", ""))

    if scenario_id_from_file and scenario_id_from_rollouts:
        if scenario_id_from_file != scenario_id_from_rollouts:
            raise ValueError("시나리오 ID가 서로 다릅니다. "
                             f"(파일에서 읽은 ID='{scenario_id_from_file}', "
                             f"예측 결과 ID='{scenario_id_from_rollouts}')")


def validate_wosac_rollouts_or_raise(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    *,
    check_scenario_id: bool = True,
) -> None:
    """예측 결과가 '필수 대상만' 그리고 '필수 대상 전부' 들어있는지 공식 규칙으로 검사합니다.

    이 함수가 확인하는 것(공식 API가 검사)
    ------------------------------
    1) "반드시 포함해야 하는 대상"이 하나도 빠지지 않았는지
    2) "포함하면 안 되는 대상"이 섞이지 않았는지
    3) 각 대상의 예측 길이가 규칙(보통 80칸)과 맞는지
    4) 예측 결과 묶음 개수가 규칙(보통 32개)과 맞는지

    Args:
        scenario (scenario_pb2.Scenario):
            원본 시나리오(정답이 들어있는 데이터).
        scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
            예측 결과 묶음.
        check_scenario_id (bool):
            True이면, 시나리오 ID도 같이 비교합니다.

    Returns:
        None

    Raises:
        ValueError:
            규칙을 하나라도 어기면 발생합니다.
            (어떤 대상이 빠졌는지/섞였는지 등이 메시지에 들어갑니다.)
    """
    if check_scenario_id:
        _assert_scenario_id_matches_rollouts(
            scenario=scenario,
            scenario_rollouts=scenario_rollouts,
        )

    try:
        submission_specs.validate_scenario_rollouts(
            scenario_rollouts=scenario_rollouts,
            original_scenario=scenario,
            challenge_type=submission_specs.ChallengeType.SIM_AGENTS,
        )
    except ValueError as e:
        sid = str(getattr(scenario, "scenario_id", ""))
        raise ValueError(f"[WOSAC 규칙 위반] scenario_id='{sid}' | {e}") from e


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self, prefix: str, ego_only: bool = False) -> None:
        super().__init__()
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.wosac_config = self.load_metrics_config()

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "traffic_light_violation_likelihood",
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
            "simulated_traffic_light_violation_rate",
        ]
        # 이건 “나중에 여러 GPU 프로세스의 값을 합쳐서 한 덩어리로 만들 수 있게” 설계된 표시야.
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter",
                       default=tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state(
            "z_only_average_displacement_error",
            default=tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "z_only_min_average_displacement_error",
            default=tensor(0.0),
            dist_reduce_fx="sum",
        )
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_z_only_displacement_metrics(
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
        challenge_type: submission_specs.ChallengeType = submission_specs.
        ChallengeType.SIM_AGENTS,
    ) -> Dict[str, float]:
        """z 값만으로 '얼마나 틀렸는지'를 계산합니다.

        왜 필요한가?
        ------------
        현재 로그에서
        - val_closed/ADE 는 (x, y)만 보고 계산합니다.
        - WOSAC의 average_displacement_error / min_ade 는 (x, y, z)를 같이 씁니다.

        그래서 z가 잘못 들어가면 WOSAC 쪽 값이 크게 나빠질 수 있는데,
        그 원인이 정말 z 때문인지 빠르게 확인하려면
        'z만' 따로 떼어서 오차를 출력하는 게 가장 확실합니다.

        계산 방법(정확히)
        ----------------
        1) 시나리오(정답)에서 평가 대상 물체들의 미래 z와 valid(유효/무효)를 읽습니다.
           - 정답 z: shape (M, T)
           - 정답 valid: shape (M, T)
           - M: 평가 대상 물체 개수
           - T: 미래 길이(보통 80)

        2) 예측(rollouts)에서 같은 물체들의 미래 z를 읽습니다.
           - 예측 z: rollout마다 (M, T)
           - rollout 개수 R(보통 32)

        3) 각 rollout r, 물체 m에 대해
           valid인 시간만 골라 |z_pred - z_gt| 의 평균을 냅니다.
           - 물체별 z 평균오차: shape (R, M)

        4) 아래 두 값을 만듭니다.
           - average_displacement_error_z_only:
               모든 rollout과 모든 물체에 대해 평균낸 값 (스칼라)
           - min_average_displacement_error_z_only:
               rollout마다 물체 평균을 낸 뒤, 그 중 가장 작은 값 (스칼라)

        Args:
            scenario (scenario_pb2.Scenario):
                정답이 들어있는 시나리오.
            scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
                예측이 들어있는 rollouts.
            challenge_type (submission_specs.ChallengeType):
                기본은 SIM_AGENTS.

        Returns:
            Dict[str, float]:
                - "average_displacement_error_z_only": float
                - "min_average_displacement_error_z_only": float
        """
        config = submission_specs.get_submission_config(challenge_type)
        start_idx = int(config.current_time_index) + 1
        end_idx = start_idx + int(config.n_simulation_steps)
        num_steps = int(config.n_simulation_steps)

        # 평가 대상 물체 id들 (AV + tracks_to_predict)
        eval_object_ids = submission_specs.get_evaluation_sim_agent_ids(
            scenario, challenge_type)

        # 빠른 조회를 위한 track 사전
        track_by_id: Dict[int, scenario_pb2.Track] = {
            int(track.id): track for track in scenario.tracks
        }

        # 정답 z/valid 준비: object_id -> (z[T], valid[T])
        gt_z_by_id: Dict[int, np.ndarray] = {}
        gt_valid_by_id: Dict[int, np.ndarray] = {}

        for obj_id in eval_object_ids:
            obj_id_int = int(obj_id)
            track = track_by_id.get(obj_id_int, None)
            if track is None:
                continue

            states = track.states[start_idx:end_idx]
            if len(states) != num_steps:
                # 길이가 기대와 다르면 안전하게 제외
                continue

            gt_z = np.asarray([s.center_z for s in states],
                              dtype=np.float32)  # (T,)
            gt_valid = np.asarray([s.valid for s in states],
                                  dtype=np.bool_)  # (T,)

            gt_z_by_id[obj_id_int] = gt_z
            gt_valid_by_id[obj_id_int] = gt_valid

        if len(gt_z_by_id) == 0 or len(scenario_rollouts.joint_scenes) == 0:
            return {
                "average_displacement_error_z_only": 0.0,
                "min_average_displacement_error_z_only": 0.0,
            }

        total_sum = 0.0
        total_count = 0
        rollout_means: List[float] = []

        # rollout(R) 반복
        for joint_scene in scenario_rollouts.joint_scenes:
            pred_z_by_id: Dict[int, np.ndarray] = {}

            for traj in joint_scene.simulated_trajectories:
                obj_id_int = int(traj.object_id)
                if obj_id_int not in gt_z_by_id:
                    continue

                pred_z = np.asarray(traj.center_z, dtype=np.float32)  # (T,)
                if pred_z.shape[0] != num_steps:
                    continue
                pred_z_by_id[obj_id_int] = pred_z

            # 이번 rollout에서의 물체별 z 평균오차들
            per_object_errors: List[float] = []

            for obj_id_int, gt_z in gt_z_by_id.items():
                pred_z = pred_z_by_id.get(obj_id_int, None)
                if pred_z is None:
                    continue

                gt_valid = gt_valid_by_id[obj_id_int]  # (T,)
                valid_cnt = int(gt_valid.sum())
                if valid_cnt == 0:
                    continue

                abs_err = np.abs(pred_z - gt_z)  # (T,)
                valid_f = gt_valid.astype(np.float32)  # (T,)
                z_mean_err = float((abs_err * valid_f).sum() / float(valid_cnt))

                per_object_errors.append(z_mean_err)
                total_sum += z_mean_err
                total_count += 1

            if len(per_object_errors) > 0:
                rollout_means.append(float(np.mean(per_object_errors)))

        avg_z_only = float(total_sum /
                           float(total_count)) if total_count > 0 else 0.0
        min_z_only = float(
            min(rollout_means)) if len(rollout_means) > 0 else 0.0

        return {
            "average_displacement_error_z_only": avg_z_only,
            "min_average_displacement_error_z_only": min_z_only,
        }

    @staticmethod
    def _compute_scenario_metrics(
        config,
        scenario_file,
        scenario_rollout,
        ego_only,
    ) -> Tuple[sim_agents_metrics_pb2.SimAgentMetrics, Dict[str, float]]:
        scenario = scenario_pb2.Scenario()

        debug = True
        first_id = None
        second_id = None

        dataset = tf.data.TFRecordDataset([scenario_file], compression_type="")
        for idx, data in enumerate(dataset):
            tmp = scenario_pb2.Scenario()
            tmp.ParseFromString(bytes(data.numpy()))

            if idx == 0:
                scenario.CopyFrom(tmp)
                first_id = tmp.scenario_id
                if not debug:
                    break
            elif idx == 1:
                second_id = tmp.scenario_id
                break

        if debug:
            rollout_id = getattr(scenario_rollout, "scenario_id", "")
            print(f"[WOSAC_DEBUG] file={scenario_file}")
            print(
                f"[WOSAC_DEBUG] first_id={first_id} second_id={second_id} rollout_id={rollout_id}"
            )

            if second_id is not None:
                print(
                    "[WOSAC_DEBUG][WARNING] 이 파일 안에 시나리오가 2개 이상 들어있습니다(샤드일 가능성)."
                )

            if rollout_id and first_id and (rollout_id != first_id):
                print(
                    "[WOSAC_DEBUG][ERROR] rollout_id != 파일의 첫 scenario_id 입니다. (정답-예측 매칭이 깨졌을 가능성 큼)"
                )

        for data in tf.data.TFRecordDataset([scenario_file],
                                            compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break

        if ego_only:
            for i in range(len(scenario.tracks)):
                if i != scenario.sdc_track_index:
                    for t in range(91):
                        scenario.tracks[i].states[t].valid = False
            while len(scenario.tracks_to_predict) > 1:
                scenario.tracks_to_predict.pop()
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

        validate_wosac_rollouts_or_raise(
            scenario=scenario,
            scenario_rollouts=scenario_rollout,
            # 이 함수 인자 이름이 scenario_rollout(단수)라서 이렇게 넣는 게 맞아요
        )
        # 기존 WOSAC metric (그대로)
        scenario_metrics = wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout)

        # z만 따로 계산한 metric (추가)
        z_only_metrics = WOSACMetrics._compute_z_only_displacement_metrics(
            scenario=scenario,
            scenario_rollouts=scenario_rollout,
            challenge_type=submission_specs.ChallengeType.SIM_AGENTS,
        )

        return scenario_metrics, z_only_metrics

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:

        if os.environ.get("CUDA_VISIBLE_DEVICES", "") in ["", "0"]:
            if not self.is_mp_init:
                self.is_mp_init = True
                mp.set_start_method("forkserver", force=True)
            with mp.Pool(processes=len(scenario_rollouts)) as pool:
                pool_scenario_metrics = pool.starmap(
                    self._compute_scenario_metrics,
                    zip(
                        itertools.repeat(self.wosac_config),
                        scenario_files,
                        scenario_rollouts,
                        itertools.repeat(self.ego_only),
                    ),
                )
                pool.close()
                pool.join()
        else:
            pool_scenario_metrics = []
            for _scenario, _scenario_rollout in zip(scenario_files,
                                                    scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(self.wosac_config, _scenario,
                                                   _scenario_rollout,
                                                   self.ego_only))

        for scenario_metrics, z_only in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += scenario_metrics.average_displacement_error
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
            self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
            self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
            self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
            self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
            self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
            self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
            self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
            self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

            self.z_only_average_displacement_error += tensor(
                float(z_only.get("average_displacement_error_z_only", 0.0)))
            self.z_only_min_average_displacement_error += tensor(
                float(z_only.get("min_average_displacement_error_z_only", 0.0)))

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(scenario_id="",
                                                              **metrics_dict)
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics)

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric":
                final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics":
                final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics":
                final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics":
                final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade":
                final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter":
                self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        out_dict[
            f"{self.prefix}/wosac_likelihood/average_displacement_error_z_only"] = (
                self.z_only_average_displacement_error / self.scenario_counter)
        out_dict[f"{self.prefix}/wosac/min_ade_z_only"] = (
            self.z_only_min_average_displacement_error / self.scenario_counter)
        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        """WOSAC 2025 Sim Agents 평가 설정 파일을 읽습니다.

        - 2024 설정 파일(challenge_2024_config.textproto)이 아니라,
          2025 설정 파일(challenge_2025_sim_agents_config.textproto)을 사용합니다.
        - 이렇게 해야 2025 점수 계산 규칙(예: 신호등 위반 항목 포함)이 제대로 반영됩니다.
        """
        config_path = (Path(wosac_metrics.__file__).parent /
                       "challenge_2025_sim_agents_config.textproto")
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
