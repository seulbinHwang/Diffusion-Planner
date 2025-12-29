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
from typing import Dict, List

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
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
        ]
        # 이건 “나중에 여러 GPU 프로세스의 값을 합쳐서 한 덩어리로 만들 수 있게” 설계된 표시야.
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter", default=tensor(0.0), dist_reduce_fx="sum")
        tf.config.set_visible_devices([], "GPU")

    @staticmethod
    def _compute_scenario_metrics(
        config, scenario_file, scenario_rollout, ego_only
    ) -> sim_agents_metrics_pb2.SimAgentMetrics:
        scenario = scenario_pb2.Scenario()

        debug = True
        first_id = None
        second_id = None

        dataset = tf.data.TFRecordDataset([scenario_file], compression_type="")
        for idx, data in enumerate(dataset):
            tmp = scenario_pb2.Scenario()
            tmp.ParseFromString(bytes(data.numpy()))

            if idx == 0:
                scenario.CopyFrom(tmp)  # ✅ 실제 계산에는 "첫 레코드"를 그대로 사용
                first_id = tmp.scenario_id
                if not debug:
                    break  # ✅ 원래처럼 빠르게 끝
            elif idx == 1:
                second_id = tmp.scenario_id  # ✅ 두 번째 레코드가 있으면 "샤드 파일" 가능성 매우 큼
                break

        if debug:
            rollout_id = getattr(scenario_rollout, "scenario_id", "")
            print(f"[WOSAC_DEBUG] file={scenario_file}")
            print(
                f"[WOSAC_DEBUG] first_id={first_id} second_id={second_id} rollout_id={rollout_id}")

            if second_id is not None:
                print(
                    "[WOSAC_DEBUG][WARNING] 이 파일 안에 시나리오가 2개 이상 들어있습니다(샤드일 가능성).")

            if rollout_id and first_id and (rollout_id != first_id):
                print(
                    "[WOSAC_DEBUG][ERROR] rollout_id != 파일의 첫 scenario_id 입니다. (정답-예측 매칭이 깨졌을 가능성 큼)")

        for data in tf.data.TFRecordDataset([scenario_file], compression_type=""):
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

        return wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout
        )

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
            for _scenario, _scenario_rollout in zip(scenario_files, scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.wosac_config, _scenario, _scenario_rollout, self.ego_only
                    )
                )

        for scenario_metrics in pool_scenario_metrics:
            self.scenario_counter += 1
            self.metametric += scenario_metrics.metametric
            self.average_displacement_error += (
                scenario_metrics.average_displacement_error
            )
            self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
            self.linear_acceleration_likelihood += (
                scenario_metrics.linear_acceleration_likelihood
            )
            self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
            self.angular_acceleration_likelihood += (
                scenario_metrics.angular_acceleration_likelihood
            )
            self.distance_to_nearest_object_likelihood += (
                scenario_metrics.distance_to_nearest_object_likelihood
            )
            self.collision_indication_likelihood += (
                scenario_metrics.collision_indication_likelihood
            )
            self.time_to_collision_likelihood += (
                scenario_metrics.time_to_collision_likelihood
            )
            self.distance_to_road_edge_likelihood += (
                scenario_metrics.distance_to_road_edge_likelihood
            )
            self.offroad_indication_likelihood += (
                scenario_metrics.offroad_indication_likelihood
            )
            self.min_average_displacement_error += (
                scenario_metrics.min_average_displacement_error
            )
            self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
            self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(
            scenario_id="", **metrics_dict
        )
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics
        )

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric": final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics": final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics": final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics": final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade": final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter": self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        config_path = (
            Path(wosac_metrics.__file__).parent / "challenge_2024_config.textproto"
        )
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
