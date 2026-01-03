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
import time
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
        self,
        is_active: bool,
        save_path: str,
        eval_method: str,
            global_rank: int
    ) -> None:
        super().__init__()
        self.is_active = is_active
        if self.is_active:
            self.method_name = "DRAFT"
            self.authors = ["Seulbin Hwang"]
            self.affiliation = "NaverLabs"
            self.description = (
                "We generate multimodal future trajectories "
                "with diffusion and enforce physical plausibility "
                "by projecting them into feasible unicycle controls "
                "with an infeasible-control penalty.")
            self.method_link = "not available yet"
            self.account_name = "h.sb@naverlabs.com"
            self.buffer_scenario_rollouts = []
            # ----------------------------
            # 진행 상황(Progress) 출력 설정 (공통: DP_PROGRESS_SEC)
            # - DP_PROGRESS_SEC <= 0 이면 출력 안 함
            # ----------------------------
            self._progress_interval_sec: float = self._read_progress_interval_sec(
                default_sec=60.0)
            self._progress_start_time_sec: float = float(time.perf_counter())
            self._progress_last_print_time_sec: float = float(
                self._progress_start_time_sec)

            self._progress_total_received: int = 0
            self._progress_total_duplicates: int = 0
            self._saved_shard_count: int = 0

            # 중복 scenario 방지용(빠른 조회)
            self._submission_scenario_id_set = set()

            self.i_file = 0
            save_path = os.path.join(save_path, eval_method)
            # remove existing save_path directory
            if global_rank == 0 and os.path.exists(save_path):
                shutil.rmtree(save_path)

            self.submission_dir = Path(
                os.path.join(save_path, f"wosac_submission"))

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
        scenario_id: List[str],  # length: B (시나리오 수)
        agent_id: Tensor,  # [n_ag]
        agent_batch: Tensor,  # [n_ag], 각 에이전트가 어느 시나리오에 속하는지 나타내는 인덱스
        pred_traj: Tensor,  # [n_ag, n_rollout, n_step, 2]
        pred_z: Tensor,  # [n_ag, n_rollout, n_step]
        pred_head: Tensor,  # [n_ag, n_rollout, n_step]
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
        self.scenario_id.append(get_scenario_id_int_tensor(
            scenario_id, _device))
        self.pred_traj.append(pred_traj)
        self.pred_z.append(pred_z)
        self.pred_head.append(pred_head)

        batch_size = len(scenario_id)
        self.agent_batch.append(agent_batch + batch_size * global_rank)

    def compute(self) -> Dict[str, Tensor]:
        return {k: getattr(self, k) for k in self.data_keys}

    @staticmethod
    def _read_progress_interval_sec(default_sec: float = 60.0) -> float:
        """DP_PROGRESS_SEC 환경변수에서 진행 출력 주기(초)를 읽습니다.

        - 값이 없거나 숫자로 변환이 안 되면 default_sec 사용
        - 0 이하이면 진행 출력 기능을 끕니다.

        Args:
            default_sec (float): 기본값(초). shape: ()

        Returns:
            float: 출력 주기(초). shape: ()
        """
        raw = os.environ.get("DP_PROGRESS_SEC", "")
        if str(raw).strip() == "":
            return float(default_sec)
        try:
            return float(str(raw).strip())
        except ValueError:
            return float(default_sec)

    @staticmethod
    def _is_rank_zero() -> bool:
        """현재 프로세스가 rank 0인지 확인합니다.

        torchrun 환경에서는 RANK가 설정됩니다.
        없으면 단일 프로세스로 보고 rank 0으로 취급합니다.
        """
        return str(os.environ.get("RANK", "0")).strip() == "0"

    @staticmethod
    def _format_duration_hms(duration_sec: float) -> str:
        """초 단위 시간을 'Hh Mm Ss' 문자열로 바꿉니다."""
        total_sec = int(max(0.0, float(duration_sec)))
        hours = total_sec // 3600
        minutes = (total_sec % 3600) // 60
        seconds = total_sec % 60
        return f"{hours}h {minutes:02d}m {seconds:02d}s"

    def _maybe_print_progress(self, force: bool = False) -> None:
        """조건이 맞으면 WOSACSubmission 진행 상황을 출력합니다."""
        if not self._is_rank_zero():
            return
        interval_sec = float(getattr(self, "_progress_interval_sec", 0.0))
        if interval_sec <= 0.0:
            return

        now = float(time.perf_counter())
        last = float(getattr(self, "_progress_last_print_time_sec", now))
        if (not force) and ((now - last) < interval_sec):
            return

        start = float(getattr(self, "_progress_start_time_sec", now))
        elapsed = now - start

        try:
            unique_cnt = int(
                len(getattr(self, "_submission_scenario_id_set", set())))
        except Exception:
            unique_cnt = int(len(getattr(self, "submission_scenario_id", [])))

        buffer_cnt = int(len(getattr(self, "buffer_scenario_rollouts", [])))
        shard_cnt = int(getattr(self, "_saved_shard_count", 0))
        recv_cnt = int(getattr(self, "_progress_total_received", 0))
        dup_cnt = int(getattr(self, "_progress_total_duplicates", 0))

        elapsed_str = self._format_duration_hms(elapsed)
        print(
            f"[WOSACSubmission] progress: unique={unique_cnt}, buffer={buffer_cnt}, "
            f"shards_saved={shard_cnt}, received={recv_cnt}, dup_skipped={dup_cnt} "
            f"(elapsed {elapsed_str})",
            flush=True,
        )
        self._progress_last_print_time_sec = float(now)

    def aggregate_rollouts(
        self,
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
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
            self._progress_total_received += 1

            sid = rollout.scenario_id
            if sid in self._submission_scenario_id_set:
                self._progress_total_duplicates += 1
                continue

            self._submission_scenario_id_set.add(sid)
            self.submission_scenario_id.append(sid)
            self.buffer_scenario_rollouts.append(rollout)

            if len(self.buffer_scenario_rollouts) > 300:
                self._save_shard()
                self._maybe_print_progress(force=True)

        self._maybe_print_progress(force=False)

    def save_sub_file(self) -> None:
        """
        남아있는 버퍼를 먼저 파일 1개로 저장

        지금까지 모아둔 제출 데이터를 최종 제출 파일(.tar.gz)로 정리해서 저장
        """
        self._save_shard()
        self.i_file = 0
        tar_file_name = self.submission_dir.as_posix() + ".tar.gz"

        log.info(f"Saving wosac submission files to {tar_file_name}")

        shard_files = sorted(
            [p.as_posix() for p in self.submission_dir.glob("*")])
        total = int(len(shard_files))

        interval_sec = float(getattr(self, "_progress_interval_sec", 0.0))
        tar_start_t = float(time.perf_counter())
        tar_last_t = float(tar_start_t)

        with tarfile.open(tar_file_name, "w:gz") as tar:
            for idx, output_filename in enumerate(shard_files, start=1):
                tar.add(
                    output_filename,
                    arcname=output_filename + f"-of-{len(shard_files):05d}",
                )

                if interval_sec > 0.0:
                    now = float(time.perf_counter())
                    if (now - tar_last_t) >= interval_sec or idx >= total:
                        elapsed = now - tar_start_t
                        elapsed_str = self._format_duration_hms(elapsed)
                        if self._is_rank_zero():
                            print(
                                f"[WOSACSubmission] tar progress: {idx}/{total} (elapsed {elapsed_str})",
                                flush=True,
                            )
                        tar_last_t = now

        log.info(f"DONE: Saved wosac submission files to {tar_file_name}")
        self._maybe_print_progress(force=True)

    def _save_shard(self) -> None:
        shard_submission = sim_agents_submission_pb2.SimAgentsChallengeSubmission(
            scenario_rollouts=self.buffer_scenario_rollouts,
            submission_type=sim_agents_submission_pb2.
            SimAgentsChallengeSubmission.SIM_AGENTS_SUBMISSION,
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
        self._saved_shard_count += 1
        self.buffer_scenario_rollouts = []

        self._maybe_print_progress(force=True)
