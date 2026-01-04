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

import torch
from omegaconf import ListConfig
from torch import Tensor
from torchmetrics.metric import Metric
from waymo_open_dataset.protos import sim_agents_submission_pb2
import os
from src.utils import RankedLogger
from src.utils.wosac_utils import get_scenario_id_int_tensor
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


def _is_distributed_ready() -> bool:
    """여러 프로세스가 서로 값을 주고받을 준비가 되었는지 확인합니다.

    이 함수가 하는 일
    ----------------
    - 여러 GPU를 쓰는 실행에서는, 프로세스(rank)들이 "작은 숫자"를 서로 공유해야 할 때가 있습니다.
      (여기서는 이번 step에서 각 rank가 처리한 시나리오 개수)
    - 준비가 안 된 상태에서 통신을 시도하면 즉시 에러가 나므로,
      통신 가능 여부를 먼저 확인합니다.

    Returns:
        bool: True이면 값 공유(통신)를 사용할 수 있습니다. shape: ()
    """
    if not torch.distributed.is_available():
        return False
    return bool(torch.distributed.is_initialized())


def _compute_global_scenario_offset_for_this_rank(
    local_scenario_count: int,
    device: torch.device,
) -> int:
    """이번 step에서 "내 rank 시나리오가 전체에서 몇 번째부터 시작하는지(offset)"를 계산합니다.

    문제 상황(왜 필요한가?)
    ----------------------
    - rank마다 이번 step의 batch_size(=시나리오 개수)가 같다고 보장되지 않습니다.
    - 그런데 기존처럼 `batch_size * rank`로 오프셋을 만들면,
      마지막 근처에서 rank별 batch_size가 달라지는 순간 "시나리오 번호가 겹치거나 비는" 문제가 생깁니다.

    해결 방식(핵심)
    --------------
    - 이번 step에서 각 rank의 local_scenario_count를 전부 모읍니다.
      예: sizes = [B0, B1, B2, B3]  (길이 = world_size)
    - 내 rank의 offset은 앞 rank들의 합입니다.
      offset = B0 + B1 + ... + B(rank-1)

    Args:
        local_scenario_count (int): 이 rank가 이번 step에서 처리한 시나리오 개수. shape: ()
        device (torch.device): 통신에 사용할 숫자 텐서를 만들 장치. shape: ()
            - NCCL(일반적인 multi-GPU)에서는 보통 CUDA 텐서가 안전합니다.

    Returns:
        int: 내 rank 시나리오가 "전체(모든 rank 합친)"에서 시작하는 위치. shape: ()
    """
    safe_local = int(max(0, int(local_scenario_count)))

    if not _is_distributed_ready():
        return 0

    world_size = int(torch.distributed.get_world_size())
    rank = int(torch.distributed.get_rank())

    # local_count_tensor: shape (1,)
    local_count_tensor = torch.tensor([safe_local], device=device, dtype=torch.long)

    # gathered: List[Tensor], length = world_size, 각 텐서는 shape (1,)
    gathered: List[torch.Tensor] = [
        torch.zeros_like(local_count_tensor) for _ in range(world_size)
    ]
    torch.distributed.all_gather(gathered, local_count_tensor)

    # sizes: length = world_size
    sizes = [int(t.item()) for t in gathered]

    offset = int(sum(sizes[:rank]))
    return offset


def _apply_global_offset_to_agent_batch(
    agent_batch: Tensor,
    local_scenario_count: int,
) -> Tensor:
    """agent_batch에 전역 offset을 더해, 여러 rank 결과를 합쳐도 번호가 겹치지 않게 만듭니다.

    Args:
        agent_batch (Tensor): 각 agent가 "로컬 배치에서 몇 번째 시나리오에 속하는지" 나타내는 값.
            shape: (n_agent,)
            - 값 범위는 보통 [0, local_scenario_count-1] 입니다.
        local_scenario_count (int): 이번 step에서 이 rank의 시나리오 개수(B). shape: ()

    Returns:
        Tensor: 전역 기준으로 바뀐 agent_batch.
            shape: (n_agent,)
            - 값 범위는 보통 [offset, offset + local_scenario_count - 1] 쪽으로 이동합니다.
    """
    if not isinstance(agent_batch, torch.Tensor):
        raise TypeError("agent_batch는 torch.Tensor여야 합니다.")
    if agent_batch.dim() != 1:
        raise ValueError(
            f"agent_batch는 (n_agent,) 이어야 합니다. 현재 shape={tuple(agent_batch.shape)}"
        )

    offset = _compute_global_scenario_offset_for_this_rank(
        local_scenario_count=int(local_scenario_count),
        device=agent_batch.device,
    )
    return agent_batch + int(offset)


log = RankedLogger(__name__, rank_zero_only=False)


# smart/metrics/wosac_submission.py
class WOSACSubmission(Metric):

    def __init__(
            self,
            is_active: bool,
            save_path: str,
            eval_method: str,
            global_rank: int,
            resume: bool = False,
            existing_scenario_id_set: Optional[Set[str]] = None,
            start_shard_index: Optional[int] = None,
            existing_shard_count: Optional[int] = None,
    ) -> None:
        """WOSAC 제출 파일(binproto)을 저장/누적하는 객체를 만듭니다.

        Args:
            is_active (bool): True면 저장 기능을 켭니다. shape: ()
            save_path (str): 실험 루트 폴더. shape: ()
            eval_method (str): "validation" 또는 "test". shape: ()
            global_rank (int): 현재 프로세스 rank. shape: ()
            resume (bool): True면 기존 결과를 지우지 않고 이어서 저장합니다. shape: ()
            existing_scenario_id_set (Optional[Set[str]]):
                재개 모드에서 이미 저장된 scenario_id 집합.
                - None이면 빈 집합으로 시작합니다.
                size: (N_done,) 또는 None
            start_shard_index (Optional[int]):
                재개 모드에서 다음에 쓸 binproto 번호.
                - None이면 0부터 시작합니다.
                shape: ()
            existing_shard_count (Optional[int]):
                재개 모드에서 이미 존재하는 shard 개수(진행 출력용).
                shape: ()
        """
        super().__init__()
        self.is_active = bool(is_active)
        if not self.is_active:
            return

        self.method_name = "DRAFT"
        self.authors = ["Seulbin Hwang"]
        self.affiliation = "NaverLabs"
        self.description = (
            "We generate multimodal future trajectories "
            "with diffusion and enforce physical plausibility "
            "by projecting them into feasible unicycle controls "
            "with an infeasible-control penalty."
        )
        self.method_link = "not available yet"
        self.account_name = "h.sb@naverlabs.com"

        self.buffer_scenario_rollouts: List[
            sim_agents_submission_pb2.ScenarioRollouts] = []

        self._progress_interval_sec = self._read_progress_interval_sec(
            default_sec=60.0)
        self._progress_start_time_sec = float(time.perf_counter())
        self._progress_last_print_time_sec = float(
            self._progress_start_time_sec)

        self._progress_total_received = 0
        self._progress_total_duplicates = 0

        # ✅ 중복 방지 set 복구(재개 모드에서만 의미가 큼)
        self._submission_scenario_id_set: Set[str] = set(
            existing_scenario_id_set or set())

        # ✅ 이미 저장된 shard 개수(진행 출력용)
        self._saved_shard_count = int(existing_shard_count or 0)

        # ✅ 저장 경로 준비
        save_root = os.path.join(str(save_path), str(eval_method))

        # ✅ 기존 결과 폴더 삭제는 "재개 모드가 아닐 때만"
        if int(global_rank) == 0 and os.path.exists(save_root) and (
        not bool(resume)):
            shutil.rmtree(save_root)

        self.submission_dir = Path(
            os.path.join(save_root, "wosac_submission"))
        self.submission_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 다음 파일 번호(00146부터 이어쓰기 등)
        if bool(resume):
            self.i_file = int(start_shard_index or 0)
        else:
            self.i_file = 0

        self.submission_scenario_id: List[str] = []

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

        agent_batch_global = _apply_global_offset_to_agent_batch(
            agent_batch=agent_batch,  # shape: (n_agent,)
            local_scenario_count=int(batch_size),  # shape: ()
        )
        self.agent_batch.append(agent_batch_global)

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
        # ✅ 버퍼가 비어 있으면 파일을 만들지 않습니다.
        if int(len(self.buffer_scenario_rollouts)) == 0:
            return
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
