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
import shutil

import tarfile
from pathlib import Path
from typing import Dict, List

import hydra
from omegaconf import ListConfig
from torch import Tensor
from torchmetrics.metric import Metric
from waymo_open_dataset.protos import sim_agents_submission_pb2
import os
from src.utils import RankedLogger
from src.utils.wosac_utils import get_scenario_id_int_tensor

log = RankedLogger(__name__, rank_zero_only=False)

# smart/metrics/wosac_submission.py
class WOSACSubmission(Metric):
    def __init__(
        self, is_active: bool,
        save_path: str,
            eval_method: str,
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.method_name = "DRAFT"
            self.authors = ["Seulbin Hwang"]
            self.affiliation = "NaverLabs"
            self.description = ("We generate multimodal future trajectories "
                                "with diffusion and enforce physical plausibility "
                                "by projecting them into feasible unicycle controls "
                                "with an infeasible-control penalty.")
            self.method_link = "not available yet"
            self.account_name = "h.sb@naverlabs.com"
            self.buffer_scenario_rollouts = []
            self.i_file = 0
            save_path = os.path.join(save_path, eval_method)
            # remove existing save_path directory
            if os.path.exists(save_path):
                shutil.rmtree(save_path)

            
            self.submission_dir = Path(os.path.join(save_path, f"wosac_submission"))

            # Make directory if it doesn't exist
            self.submission_dir.mkdir(parents=True, exist_ok=True)

            self.submission_scenario_id = []

            self.data_keys = [
                "scenario_id",
                "agent_id",
                "agent_batch",
                "pred_traj",
                "pred_z",
                "pred_head",
            ]
            for k in self.data_keys:
                self.add_state(k, default=[], dist_reduce_fx="cat")

    def update(
        self,
        scenario_id: List[str], # length: B (시나리오 수)
        agent_id: Tensor, # [n_ag]
        agent_batch: Tensor, # [n_ag], 각 에이전트가 어느 시나리오에 속하는지 나타내는 인덱스
        pred_traj: Tensor, # [n_ag, n_rollout, n_step, 2]
        pred_z: Tensor, # [n_ag, n_rollout, n_step]
        pred_head: Tensor, # [n_ag, n_rollout, n_step]
        global_rank: int,
    ) -> None:
        """
        *“이번 배치에서 만든 예측 결과를,
            나중에 WOSAC 제출 파일로 바꾸기 좋게 차곡차곡 모아두는 함수”*
        나중에 여러 배치/여러 GPU에서 나온 결과를
            한꺼번에 모아서 WOSAC 제출 형식으로 만들기 위해서
        여러 GPU 결과가 섞이지 않게 “묶음 번호(agent_batch)”를 안전하게 바꿈
            나중에 _unbatch()로 시나리오별로 다시 나눌 때
            다른 GPU의 데이터가 한 시나리오로 섞이는 사고를 막기 위해서
        """
        _device = pred_traj.device
        self.agent_id.append(agent_id)
        self.scenario_id.append(get_scenario_id_int_tensor(scenario_id, _device))
        self.pred_traj.append(pred_traj)
        self.pred_z.append(pred_z)
        self.pred_head.append(pred_head)

        batch_size = len(scenario_id)
        self.agent_batch.append(agent_batch + batch_size * global_rank)

    def compute(self) -> Dict[str, Tensor]:
        return {k: getattr(self, k) for k in self.data_keys}

    def aggregate_rollouts(
        self, scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
    ) -> None:
        """ GPU 중 한대에서만 실행됩니다.
        같은 scenario_id가 또 들어오면 한 번만 담도록 막습니다.
        버퍼에 차곡차곡 쌓기
            바로 파일로 쓰지 않고, 메모리(리스트)에 잠깐 모아둡니다.
        300개 넘으면 묶어서 파일로 저장(_save_shard)
            너무 많이 쌓여서 메모리가 커지지 않게 “300개 단위로 잘라서”
            submission.binproto-xxxxx 파일로 저장하고,
            저장 후 버퍼를 비웁니다.
        """
        for rollout in scenario_rollouts:
            if rollout.scenario_id not in self.submission_scenario_id:
                self.submission_scenario_id.append(rollout.scenario_id)
                self.buffer_scenario_rollouts.append(rollout)
                if len(self.buffer_scenario_rollouts) > 300:
                    self._save_shard()

    def save_sub_file(self) -> None:
        """
        남아있는 버퍼를 먼저 파일 1개로 저장

        지금까지 모아둔 제출 데이터를 최종 제출 파일(.tar.gz)로 정리해서 저장

        """
        self._save_shard()
        self.i_file = 0
        tar_file_name = self.submission_dir.as_posix() + ".tar.gz"

        log.info(f"Saving wosac submission files to {tar_file_name}")

        shard_files = sorted([p.as_posix() for p in self.submission_dir.glob("*")])
        with tarfile.open(tar_file_name, "w:gz") as tar:
            for output_filename in shard_files:
                tar.add(
                    output_filename,
                    arcname=output_filename + f"-of-{len(shard_files):05d}",
                )
        log.info(f"DONE: Saved wosac submission files to {tar_file_name}")

    def _save_shard(self) -> None:
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.buffer_scenario_rollouts,
            submission_type=sim_agents_submission_pb2.SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
            account_name=self.account_name,
            unique_method_name=self.method_name,
            authors=self.authors,
            affiliation=self.affiliation,
            description=self.description,
            method_link=self.method_link,
            uses_lidar_data=False,
            uses_camera_data=False,
            uses_public_model_pretraining=False,
            num_model_parameters="7M",
            acknowledge_complies_with_closed_loop_requirement=True,
        )
        output_filename = self.submission_dir / f"submission.binproto-{self.i_file:05d}"
        log.info(f"Saving wosac submission files to {output_filename}")
        with open(output_filename, "wb") as f:
            f.write(shard_submission.SerializeToString())
        self.i_file += 1
        self.buffer_scenario_rollouts = []
