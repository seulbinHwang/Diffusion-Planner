from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Any, Dict, Optional, List, Callable
import wandb
import torch.nn as nn
from diffusion_planner.loss import _sanitize_norm_inputs
from tools.predictor_utils import (
    build_dataset_and_sampler,
    build_data_loader,
    create_diffusion_planner_and_ema,
    build_deepspeed_inference_config,
    init_distributed,
    set_save_path,
    prepare_wandb_resume,
    set_distributed_flag_from_env,
    maybe_resume_from_checkpoint,
    setup_logger_and_purge,
)
import threading
import numpy as np
import draw_machine
import args_util
import shutil
import sys, faulthandler, traceback
from diffusion_planner.utils.train_utils import set_seed
from timm.utils import ModelEma
from diffusion_planner.utils import ddp
import time
from tqdm import tqdm
from diffusion_planner.utils.target_feature import \
    build_target_future_tensors_and_masks_for_inference

from diffusion_planner.train_epoch import _prepare_batch_for_device
from diffusion_planner.utils.normalizer import StateNormalizer, ObservationNormalizer
from waymo_open_dataset.protos import sim_agents_submission_pb2
from functools import lru_cache
from typing import Any, List, Tuple, Optional
import torch
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils.sim_agents import submission_specs

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype
from src.utils.wosac_utils import get_scenario_id_int_tensor, \
    get_scenario_rollouts
from src.smart.metrics import WOSACMetrics, minADE, WOSACSubmission
from torch import optim
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger

import os
from typing import Optional
import argparse
import torch

_VALIDATION_HEARTBEAT: Optional["_ValidationHeartbeat"] = None


def _read_float_env_safe(env_key: str, default: float) -> float:
    """환경변수에서 실수 값을 안전하게 읽습니다.

    - 값이 없거나 숫자로 바꿀 수 없으면 default를 사용합니다.

    Args:
        env_key (str): 읽을 환경변수 이름. shape: ()
        default (float): 기본값. shape: ()

    Returns:
        float: 읽어온 값(실패 시 default). shape: ()
    """
    raw = os.environ.get(env_key, "")
    if str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


def _get_validation_heartbeat_interval_sec(default_sec: float = 60.0) -> float:
    """검증 중 상태 문장을 몇 초마다 찍을지 결정합니다.

    - DP_VALIDATION_HEARTBEAT_SEC 값을 사용합니다.
    - 0 이하이면 상태 출력 기능을 끕니다.

    Args:
        default_sec (float): 기본 주기(초). shape: ()

    Returns:
        float: 출력 주기(초). 0 이하이면 비활성. shape: ()
    """
    v = float(_read_float_env_safe("DP_VALIDATION_HEARTBEAT_SEC", default_sec))
    return float(v)


def _set_validation_batch_progress_in_args(
    args: argparse.Namespace,
    batch_idx: int,
    total_batch_steps: int,
) -> None:
    """현재 배치 번호를 args에 기록해, 다른 함수에서도 쉽게 표시할 수 있게 합니다.

    Args:
        args (argparse.Namespace): 설정 객체. shape: ()
        batch_idx (int): 현재 배치 번호(1부터). shape: ()
        total_batch_steps (int): 전체 배치 수. shape: ()

    Returns:
        None
    """
    try:
        setattr(args, "_dp_val_batch_idx", int(batch_idx))
        setattr(args, "_dp_val_total_batch_steps", int(total_batch_steps))
    except Exception:
        return


def _get_validation_batch_progress_tag(args: Any) -> str:
    """args에 저장된 배치 정보를 사람이 보기 쉬운 문자열로 만듭니다.

    Returns:
        str: 예) "batch 3/173" 또는 정보가 없으면 "batch ?/?". shape: ()
    """
    b = getattr(args, "_dp_val_batch_idx", None)
    t = getattr(args, "_dp_val_total_batch_steps", None)
    try:
        b_i = int(b)
        t_i = int(t)
        if b_i > 0 and t_i > 0:
            return f"batch {b_i}/{t_i}"
    except Exception:
        pass
    return "batch ?/?"


def _format_duration_hms(duration_sec: float) -> str:
    """초 단위 시간을 'Hh Mm Ss' 문자열로 바꿉니다."""
    total_sec = int(max(0.0, float(duration_sec)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours}h {minutes:02d}m {seconds:02d}s"


class _ValidationHeartbeat:
    """검증이 오래 걸릴 때도 “아직 실행 중”임을 주기적으로 보여줍니다.

    동작
    ----
    - start() 이후 interval_sec마다 한 번,
      마지막으로 update()로 설정된 stage(현재 단계) 문장을 출력합니다.
    - stop()을 호출하면 출력이 멈춥니다.
    """

    def __init__(self, interval_sec: float) -> None:
        self._interval_sec: float = float(interval_sec)
        self._start_time_sec: float = float(time.perf_counter())
        self._stage: str = "initializing"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        """상태 출력 루프를 시작합니다."""
        self._thread.start()

    def update(self, stage: str) -> None:
        """현재 단계 문장을 바꿉니다.

        Args:
            stage (str): 사람이 읽을 단계 문장. shape: ()
        """
        with self._lock:
            self._stage = str(stage)

    def stop(self) -> None:
        """상태 출력 루프를 멈춥니다."""
        self._stop_event.set()

    def _run(self) -> None:
        while True:
            if self._stop_event.wait(self._interval_sec):
                return
            with self._lock:
                stage = self._stage
            elapsed = time.perf_counter() - self._start_time_sec
            elapsed_str = _format_duration_hms(elapsed)
            print(f"[HEARTBEAT] {elapsed_str} | {stage}", flush=True)


def _start_validation_heartbeat_if_needed(args: argparse.Namespace) -> None:
    """환경변수 설정이 켜져 있으면 heartbeat를 시작합니다.

    - rank 0(로그를 찍는 프로세스)에서만 시작합니다.

    Args:
        args (argparse.Namespace): 설정 객체. shape: ()

    Returns:
        None
    """
    global _VALIDATION_HEARTBEAT

    interval_sec = float(_get_validation_heartbeat_interval_sec())
    if interval_sec <= 0.0:
        return

    if not _is_main_process_for_logging(args):
        return

    if _VALIDATION_HEARTBEAT is not None:
        return

    hb = _ValidationHeartbeat(interval_sec=interval_sec)
    _VALIDATION_HEARTBEAT = hb
    hb.start()


def _update_validation_heartbeat_stage(args: argparse.Namespace,
                                       stage: str) -> None:
    """heartbeat가 켜져 있으면 현재 단계 문장을 업데이트합니다.

    Args:
        args (argparse.Namespace): 설정 객체. shape: ()
        stage (str): 단계 문장. shape: ()

    Returns:
        None
    """
    global _VALIDATION_HEARTBEAT
    if _VALIDATION_HEARTBEAT is None:
        return
    if not _is_main_process_for_logging(args):
        return
    _VALIDATION_HEARTBEAT.update(stage)


def _stop_validation_heartbeat_if_needed(args: argparse.Namespace) -> None:
    """heartbeat가 켜져 있으면 종료합니다.

    Args:
        args (argparse.Namespace): 설정 객체. shape: ()

    Returns:
        None
    """
    global _VALIDATION_HEARTBEAT
    if _VALIDATION_HEARTBEAT is None:
        return
    if not _is_main_process_for_logging(args):
        return

    try:
        _VALIDATION_HEARTBEAT.stop()
    finally:
        _VALIDATION_HEARTBEAT = None


def _get_world_size_from_env() -> int:
    """환경변수에서 world_size 값을 읽습니다.

    왜 필요한가?
    ------------
    torchrun을 --nproc-per-node 1로 실행하면,
    args.ddp가 True여도 실제로는 프로세스가 1개라서
    기다림(동기화)이 대부분 의미가 없습니다.

    그래서 WORLD_SIZE 값을 읽어서
    "정말로 여러 프로세스가 동시에 돌고 있는지"를 판단합니다.

    Args:
        없음

    Returns:
        int:
            world_size 값. shape: ()
            - 환경변수에 없거나 값이 이상하면 1로 취급합니다.
    """
    raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(str(raw).strip())
    except ValueError:
        world_size = 1
    return int(max(1, world_size))


def _should_cuda_synchronize_for_validation(args: argparse.Namespace) -> bool:
    """Validation에서 torch.cuda.synchronize()가 정말 필요한지 판단합니다.

    목표
    ----
    - 프로세스가 1개(world_size=1)면, 대부분은 굳이 기다릴 필요가 없습니다.
      (게다가 이후에 .cpu() 같은 동작이 있으면 거기서 어차피 기다리게 됩니다)
    - 프로세스가 2개 이상(world_size>1)일 때만,
      결과/로그 타이밍이 뒤섞이지 않게 필요한 지점에서만 기다리도록 합니다.

    Args:
        args (argparse.Namespace):
            - args.ddp (bool): 분산 모드 플래그
            - args.device (str): "cuda", "cuda:0", "cpu" 등. shape: ()

    Returns:
        bool:
            - True: world_size>1 이고, CUDA를 쓰는 경우 (동기화 수행)
            - False: 그 외 (동기화 생략)
            shape: ()
    """
    if not bool(getattr(args, "ddp", False)):
        return False

    if int(_get_world_size_from_env()) <= 1:
        return False

    device_str = str(getattr(args, "device", ""))
    if not device_str.startswith("cuda"):
        return False

    if not torch.cuda.is_available():
        return False

    return True


def _get_rank_from_env() -> int:
    """환경변수에서 rank 값을 읽습니다.

    왜 필요한가?
    ------------
    process group이 아직 초기화되지 않은 상태에서는
    torch.distributed.get_rank()를 부르면 에러가 날 수 있습니다.
    그래서 안전하게 환경변수 RANK를 사용합니다.

    Args:
        없음

    Returns:
        int:
            rank 값. shape: ()
            - 환경변수에 없으면 0으로 취급합니다.
    """
    rank_str = os.environ.get("RANK", "0")
    try:
        return int(rank_str)
    except ValueError:
        return 0


def _is_torch_process_group_initialized() -> bool:
    """torch.distributed process group이 준비되어 있는지 확인합니다.

    Args:
        없음

    Returns:
        bool:
            - True: torch.distributed 사용 가능 + init_process_group 완료
            - False: 아직 초기화되지 않음
    """
    if not torch.distributed.is_available():
        return False
    return bool(torch.distributed.is_initialized())


import json
from typing import Any, Optional

try:
    import fcntl  # 리눅스/유닉스에서 파일 잠금에 사용
except ImportError:
    fcntl = None  # type: ignore

_VIS_BUDGET_REACHED_LOCAL: bool = False


def _get_total_save_image_trial_num(args: Any) -> int:
    """전체 시각화 저장 횟수 제한 값을 안전하게 읽습니다.

    Args:
        args (Any):
            args.total_save_image_trial_num(int)를 기대합니다.

    Returns:
        int:
            - -1: 제한 없음
            - 0 이상: 허용되는 총 저장 "횟수"
    """
    raw = getattr(args, "total_save_image_trial_num", -1)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return -1


def _get_run_count(args: Any) -> int:
    """이번 실행(run_count) 값을 안전하게 읽습니다.

    Args:
        args (Any):
            args.run_count(int)가 있으면 사용합니다.

    Returns:
        int:
            run_count 값. 없거나 이상하면 0.
    """
    raw = getattr(args, "run_count", 0)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _safe_remove_file_if_exists(file_path: str) -> bool:
    """파일이 있으면 지우고, 없으면 조용히 넘어갑니다.

    이 함수가 필요한 이유
    -------------------
    - 이전 실행에서 남겨둔 "카운터 파일"이 있으면,
      다음 실행에서 저장 횟수가 이미 다 찬 것으로 인식될 수 있습니다.
    - 그래서 프로그램 시작 시점에 파일을 지워서,
      이번 실행(run_count)의 저장 횟수를 "0부터" 다시 시작하게 합니다.

    Args:
        file_path (str):
            지우고 싶은 파일 경로. shape: ()

    Returns:
        bool:
            - True: 파일이 실제로 존재해서 삭제한 경우
            - False: 파일이 없었거나(이미 삭제됨), 파일이 아니거나, 삭제에 실패한 경우
    """
    if not isinstance(file_path, str) or file_path.strip() == "":
        return False

    try:
        # 파일이 아닌 경우(예: 디렉터리)는 안전하게 건드리지 않습니다.
        if not os.path.isfile(file_path):
            return False

        os.remove(file_path)
        return True

    except FileNotFoundError:
        # 다른 프로세스가 먼저 지웠을 수도 있으니 정상 케이스로 봅니다.
        return False
    except Exception:
        return False


def _purge_visualization_budget_counter_file_at_program_start(
    args: argparse.Namespace,
    global_rank: int,
    world_size: int,
) -> None:
    """프로그램 시작 시, 이번 실행(run_count)의 시각화 저장 카운터 파일을 초기화합니다.

    동작 방식
    --------
    1) args.save_path / args.eval_method / args.run_count를 이용해
       이번 실행(run_count)에 해당하는 카운터 파일 경로를 계산합니다.
       - 파일 예: "{save_path}/.dp_vis_budget_validation_run1.json"
    2) DDP(멀티 프로세스) 환경에서는 global_rank==0(대표 프로세스)만 삭제합니다.
       - 여러 프로세스가 동시에 지우거나/만드는 타이밍이 겹치면,
         오히려 실행 중간에 카운터가 리셋되는 위험이 있습니다.
    3) 삭제 후, DDP 환경이면 barrier로 모든 프로세스가 "삭제 완료" 이후에만 진행하도록 맞춥니다.
    4) 로컬 캐시 플래그(_VIS_BUDGET_REACHED_LOCAL)도 False로 되돌려,
       이번 실행에서 다시 정상적으로 저장 시도를 할 수 있게 합니다.

    Args:
        args (argparse.Namespace):
            아래 값들을 사용합니다. (모두 shape: ())
            - args.save_path (str): 카운터 파일이 있는 폴더
            - args.eval_method (str): 파일 이름 구분용
            - args.run_count (int): 파일 이름 구분용
        global_rank (int):
            현재 프로세스 rank. shape: ()
        world_size (int):
            전체 프로세스 수. shape: ()

    Returns:
        None
    """
    global _VIS_BUDGET_REACHED_LOCAL
    _VIS_BUDGET_REACHED_LOCAL = False

    counter_path = _get_visualization_budget_counter_path(args)
    if counter_path is None:
        return

    # ✅ 대표 프로세스만 삭제 (중간에 리셋되는 레이스 방지)
    if int(global_rank) == 0:
        deleted = _safe_remove_file_if_exists(counter_path)
        if deleted:
            print(f"[VIS_BUDGET] 기존 카운터 파일 삭제: {counter_path}")
        else:
            # 파일이 없거나 삭제 불필요인 경우도 흔하니, 너무 시끄럽지 않게 출력합니다.
            print(f"[VIS_BUDGET] 카운터 파일 없음(또는 삭제 불필요): {counter_path}")

    # ✅ DDP면 여기서 동기화해서, 모든 rank가 "삭제 이후"에만 진행하도록 보장
    if int(world_size) > 1 and _is_torch_process_group_initialized():
        torch.distributed.barrier()


def _get_visualization_budget_counter_path(args: Any) -> Optional[str]:
    """이번 실행(run_count)에서 공유할 '카운터 파일' 경로를 만듭니다.

    이 파일은 여러 프로세스가 같은 위치를 봐야 하므로,
    모든 프로세스가 공유하는 폴더(여기서는 args.save_path) 아래에 둡니다.

    Args:
        args (Any):
            - args.save_path (str): 저장 폴더
            - args.eval_method (str): validation/test 등(파일 이름 구분용)
            - args.run_count (int): 이번 실행 번호(파일 이름 구분용)

    Returns:
        Optional[str]:
            카운터 파일 경로.
            save_path가 없으면 None.
    """
    save_path = getattr(args, "save_path", None)
    if not isinstance(save_path, str) or not save_path:
        return None

    eval_method = str(getattr(args, "eval_method", "eval"))
    run_count = _get_run_count(args)

    # run_count마다 파일을 분리 → "이번 실행" 단위로 제한이 적용됨
    file_name = f".dp_vis_budget_{eval_method}_run{run_count}.json"
    return os.path.join(save_path, file_name)


def _read_count_from_json_text(text: str) -> int:
    """카운터 파일 내용에서 count 값을 읽습니다.

    Args:
        text (str):
            파일에 들어있는 문자열.

    Returns:
        int:
            읽은 count. 실패하면 0.
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0
    try:
        obj = json.loads(text)
        value = obj.get("count", 0)
        return int(value)
    except Exception:
        return 0


def _try_reserve_visualization_slot(counter_path: str, limit: int) -> bool:
    """공유 카운터 파일을 이용해 '저장 1회'를 예약합니다.

    동작 방식
    ----------
    - counter_path 파일을 열고(없으면 생성),
    - 잠깐 잠근 뒤(여러 프로세스가 동시에 접근해도 숫자가 꼬이지 않게),
    - 현재 count를 읽고:
        * count < limit 이면 count를 1 올리고 True
        * count >= limit 이면 False

    Args:
        counter_path (str):
            카운터 파일 경로.
        limit (int):
            허용되는 총 저장 횟수(0 이상).

    Returns:
        bool:
            - True: 이번에 저장을 진행해도 됨(예약 성공)
            - False: 이미 limit을 다 써서 저장 금지
    """
    if int(limit) < 0:
        return True
    if int(limit) == 0:
        return False

    os.makedirs(os.path.dirname(counter_path), exist_ok=True)
    with open(counter_path, "a+", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)

        f.seek(0)
        raw = f.read()
        count = _read_count_from_json_text(raw)

        if int(count) >= int(limit):
            return False

        new_count = int(count) + 1
        f.seek(0)
        f.truncate()
        f.write(json.dumps({"count": new_count}, ensure_ascii=False))
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

        return True


def _reserve_visualization_budget_if_needed(args: Any) -> bool:
    """이번에 시각화를 저장해도 되는지 판단하고, 가능하면 1회분을 예약합니다.

    Args:
        args (Any):
            - args.total_save_image_trial_num (int): 전체 제한(-1이면 무제한)
            - args.save_path (str): 카운터 파일을 둘 폴더
            - args.eval_method (str), args.run_count (int): 카운터 파일 이름 구분용

    Returns:
        bool:
            - True: 저장 허용(예약 성공)
            - False: 저장 금지(제한 초과 또는 설정 문제)
    """
    global _VIS_BUDGET_REACHED_LOCAL

    limit = _get_total_save_image_trial_num(args)
    print("limit:", limit)
    if int(limit) < 0:
        return True

    if _VIS_BUDGET_REACHED_LOCAL:
        return False

    counter_path = _get_visualization_budget_counter_path(args)
    if counter_path is None:
        # save_path가 없으면 안전하게 저장을 막습니다(파일 폭증 방지).
        _VIS_BUDGET_REACHED_LOCAL = True
        return False

    try:
        ok = _try_reserve_visualization_slot(counter_path=counter_path,
                                             limit=int(limit))
        print("ok:", ok)
    except Exception:
        ok = False

    if not ok:
        _VIS_BUDGET_REACHED_LOCAL = True
    return ok


def _maybe_distributed_barrier(
    args: argparse.Namespace,
    *,
    context: str = "",
) -> None:
    """가능한 경우에만 torch.distributed.barrier()를 호출합니다.

    변경점(핵심)
    ----------
    - world_size가 1이면(프로세스 1개) barrier는 의미가 없으므로 바로 건너뜁니다.
    - world_size가 2 이상일 때만 process group 초기화 여부를 확인하고 barrier를 호출합니다.

    Args:
        args (argparse.Namespace):
            args.ddp 값을 참고합니다. shape: ()
        context (str):
            로그를 찍을 때 어디서 호출했는지 표시용 문자열. shape: ()

    Returns:
        None
    """
    if int(_get_world_size_from_env()) <= 1:
        return

    use_ddp = bool(getattr(args, "ddp", False))
    if not use_ddp:
        return

    if not _is_torch_process_group_initialized():
        rank = _get_rank_from_env()
        if rank == 0:
            warned_flag_name = "_dp_warned_skip_barrier"
            if not getattr(_maybe_distributed_barrier, warned_flag_name, False):
                setattr(_maybe_distributed_barrier, warned_flag_name, True)
                ctx = f" ({context})" if context else ""
                print(f"[DDP] process group이 초기화되지 않아 barrier를 건너뜁니다{ctx}.")
        return

    torch.distributed.barrier()


def model_validation(
    args: argparse.Namespace,
    global_rank: int,
    rank: int,
    world_size: int,
    use_deepspeed: bool,
) -> None:
    """전체 학습 파이프라인을 실행하는 상위 함수."""
    torch.cuda.empty_cache()

    _start_validation_heartbeat_if_needed(args)
    _update_validation_heartbeat_stage(args, "preparing validation")

    try:
        # 2) seed 고정
        set_seed(args.seed + global_rank)

        _update_validation_heartbeat_stage(args, "building data loader")

        # 3) augmentation, Dataset, Sampler
        batch_size = args.batch_size
        eval_set, validation_sampler = build_dataset_and_sampler(
            args,
            args.eval_set,
            args.eval_set_list,
            args.eval_method,
            world_size,
            global_rank,
        )
        validation_loader = build_data_loader(
            args,
            eval_set,
            validation_sampler,
            batch_size,
            world_size,
        )
        _maybe_distributed_barrier(args, context="after build_data_loader")

        _update_validation_heartbeat_stage(args, "building model")

        diffusion_planner, model_ema, base_model = create_diffusion_planner_and_ema(
            args=args,
            rank=rank,
            use_deepspeed=use_deepspeed,
        )
        if use_deepspeed:
            import deepspeed
            ds_config = build_deepspeed_inference_config(args)
            ds_engine = deepspeed.init_inference(model=diffusion_planner,
                                                 config=ds_config)
            diffusion_planner = ds_engine.module  # 이후 diffusion_planner로 그대로 추론

        _update_validation_heartbeat_stage(args, "loading checkpoint")

        # 6) 체크포인트 재개
        (diffusion_planner, optimizer, scheduler, model_ema, init_epoch,
         wandb_id, train_epochs,
         allow_val_change) = maybe_resume_from_checkpoint(
             args=args,
             diffusion_planner=diffusion_planner,
             optimizer=None,
             scheduler=None,
             model_ema=model_ema,
             global_rank=global_rank,
             use_deepspeed=use_deepspeed,
         )
        args._global_update_step = 0

        min_ade = minADE(is_active=args.min_ade_is_active).to(
            torch.device(args.device))
        wosac_metrics = WOSACMetrics("val_closed", args.wosac_metric_is_active)
        wosac_submission = WOSACSubmission(
            is_active=args.wosac_sub_is_active,
            save_path=args.save_path,
            eval_method=args.eval_method,
            global_rank=global_rank,
        )
        wandb_logger = setup_logger_and_purge(
            args=args,
            global_rank=global_rank,
            wandb_id=None,
            allow_val_change=allow_val_change,
        )

        _update_validation_heartbeat_stage(args, "running validation loop")

        run_validation_loop(
            args=args,
            diffusion_planner=diffusion_planner,
            model_ema=model_ema,
            validation_loader=validation_loader,
            wosac_metrics=wosac_metrics,
            wosac_submission=wosac_submission,
            min_ade=min_ade,
            wandb_logger=wandb_logger,
            global_rank=global_rank,
        )

        _update_validation_heartbeat_stage(args, "finalizing")
        _finalize_eval_cleanup(args, global_rank, wandb_logger)

    finally:
        _stop_validation_heartbeat_if_needed(args)


def _finalize_eval_cleanup(
    args: argparse.Namespace,
    global_rank: int,
    wandb_logger: Logger,
) -> None:
    # 1) 분산 학습일 때만 barrier 호출
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if global_rank == 0:
        wandb_logger.finish()
    # 2) W&B / TensorBoard 종료
    if args.use_wandb and wandb.run is not None:
        wandb.finish()

    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    tb_dir = os.path.join(args.save_path, "tb")
    try:
        shutil.rmtree(tb_dir)
        print(f"[CLEANUP] 디렉터리 삭제: {tb_dir}")
    except FileNotFoundError:
        print(f"[CLEANUP] 디렉터리 없음 (이미 삭제됨): {tb_dir}")
    except Exception as e:
        print(f"[CLEANUP] 디렉터리 삭제 오류: {tb_dir}, {e}")


def run_validation_loop(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    validation_loader: DataLoader,
    wosac_metrics: WOSACMetrics,
    wosac_submission: WOSACSubmission,
    min_ade: minADE,
    wandb_logger: Logger,
    global_rank: int,
) -> None:
    """전체 epoch 루프를 돌면서 학습, 속도 측정, 로깅, 체크포인트 저장을 수행한다."""
    elapsed_training_time_hour: float = 0.0
    # 전체 업데이트 스텝 수 설정 및 global step 초기화 보장
    batch_num_in_one_val_epoch: int = max(1, len(validation_loader))
    # ✅ (중요) epoch 시작 전에 sampler epoch를 먼저 세팅
    # - resume(init_epoch>0) 시에도 첫 epoch부터 올바른 shuffle이 나오도록 함
    # Dict[str, torch.Tensor]
    (epoch_wosac_metrics, epoch_elapsed_time_sec) = validate_one_epoch(
        epoch=0,
        total_epochs=1,
        validation_loader=validation_loader,
        diffusion_planner=diffusion_planner,
        args=args,
        model_ema=model_ema,
        batch_num_in_one_val_epoch=batch_num_in_one_val_epoch,
        wosac_metrics=wosac_metrics,
        wosac_submission=wosac_submission,
        min_ade=min_ade,
    )
    if global_rank == 0:
        wandb_logger.log_metrics(epoch_wosac_metrics, step=args.run_count)
        # print "epoch_elapsed_time_sec"
        print(f"[Validation] completed in "
              f"{epoch_elapsed_time_sec:.2f} sec.")
        """epoch_wosac_metrics 를 출력합니다."""
        if _is_main_process_for_logging(args):
            print(f"=== Validation Metrics ===")
            for k, v in epoch_wosac_metrics.items():
                # v 가 tensor인 경우 item()으로 스칼라 값 추출, float인 경우 그대로 사용
                value = v.item() if isinstance(v, torch.Tensor) else float(v)
                print(f"{k}: {value:.6f}")


def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    batch_num_in_one_val_epoch: int,
    wosac_metrics: WOSACMetrics,
    wosac_submission: WOSACSubmission,
    min_ade: minADE,
) -> Tuple[Dict[str, torch.Tensor], float]:
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{total_epochs}")

    epoch_t0 = time.perf_counter()
    epoch_wosac_metrics: Dict[str, torch.Tensor] = validation_epoch(
        validation_loader,
        diffusion_planner,
        args,
        model_ema,
        batch_num_in_one_val_epoch,
        wosac_metrics,
        wosac_submission,
        min_ade,
    )

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ✅ world_size>1일 때만 동기화
    if _should_cuda_synchronize_for_validation(args):
        torch.cuda.synchronize()

    _maybe_distributed_barrier(args, context="validate_one_epoch end")

    epoch_elapsed_time_sec = time.perf_counter() - epoch_t0
    return epoch_wosac_metrics, epoch_elapsed_time_sec


def _is_main_process_for_logging(args: argparse.Namespace) -> bool:
    """로그(프린트)를 한 프로세스에서만 찍을지 판단합니다.

    왜 필요한가?
    ------------
    - DDP(여러 프로세스가 동시에 도는 방식)에서는 같은 메시지를 프로세스 수만큼
      반복 출력해서 로그가 지저분해질 수 있습니다.
    - 그래서 DDP일 때는 보통 "rank 0"만 출력하도록 제한합니다.
    - DDP가 아니면(싱글 프로세스) 그냥 출력합니다.

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        args (argparse.Namespace):
            args.ddp 값을 사용합니다.

    Returns:
        bool:
            - True: 이 프로세스가 로그를 출력해도 되는 경우
            - False: 로그를 출력하지 않는 것이 좋은 경우
    """
    use_ddp = bool(getattr(args, "ddp", False))
    if not use_ddp:
        return True

    if not ddp.is_dist_avail_and_initialized():
        # 분산 초기화가 아직 안 된 상태면, 안전하게 출력 허용(중복 가능성 낮음)
        return True

    return int(ddp.get_rank()) == 0


def _should_show_validation_progress(args: argparse.Namespace) -> bool:
    """Validation 진행률/ETA 로그를 출력할지 결정합니다.

    목표
    ----
    - args.verbose가 True일 때만 진행률을 보여줍니다.
    - DDP(여러 프로세스) 환경이면 rank 0에서만 출력합니다.
      (rank 0이 아닌 프로세스까지 출력하면 로그가 여러 번 반복되어 보기 어렵습니다)
    - DDP가 아니면(싱글 프로세스) verbose만 True면 출력합니다.

    Args:
        args (argparse.Namespace):
            - args.verbose (bool): 자세히 출력할지 여부
            - args.ddp (bool): 분산 모드 여부(있으면 사용)

    Returns:
        bool:
            - True  : 진행률/ETA 출력 허용
            - False : 출력하지 않음
    """
    verbose = bool(getattr(args, "verbose", False))
    if not verbose:
        return False

    return _is_main_process_for_logging(args)


def _get_progress_print_intervals(
    args: argparse.Namespace,
    total_steps: int,
) -> Tuple[float, int]:
    """진행 상황을 얼마나 자주 출력할지(시간 기준/스텝 기준) 정합니다.

    출력 주기 설정 방식
    -------------------
    - 시간 기준: 일정 시간이 지나면 한 번 출력합니다.
    - 스텝 기준: 일정 스텝 수(배치 수)가 지나면 한 번 출력합니다.
    - 둘 중 하나라도 만족하면 출력하도록 해서,
      너무 오래 조용히 있거나(시간 기준만 사용),
      너무 자주 찍히는(스텝 기준만 사용) 문제를 줄입니다.

    기본값(Args에 값이 없을 때)
    ---------------------------
    - progress_print_interval_sec: 300초(5분)
    - progress_print_interval_steps: 전체의 약 5%마다(= total_steps // 20)

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        args (argparse.Namespace):
            아래 속성이 있으면 사용합니다.
            - args.progress_print_interval_sec (float 또는 int)
            - args.progress_print_interval_steps (int)
        total_steps (int):
            전체 스텝 수(여기서는 전체 batch 개수).

    Returns:
        Tuple[float, int]:
            (print_interval_sec, print_interval_steps)
            - print_interval_sec: 최소 1.0 이상
            - print_interval_steps: 최소 1 이상
    """
    default_interval_sec = 100.0
    default_interval_steps = max(1, int(total_steps // 20))

    raw_interval_sec = getattr(args, "progress_print_interval_sec",
                               default_interval_sec)
    try:
        interval_sec = float(raw_interval_sec)
    except (TypeError, ValueError):
        interval_sec = float(default_interval_sec)
    interval_sec = max(1.0, interval_sec)

    raw_interval_steps = getattr(args, "progress_print_interval_steps",
                                 default_interval_steps)
    try:
        interval_steps = int(raw_interval_steps)
    except (TypeError, ValueError):
        interval_steps = int(default_interval_steps)
    interval_steps = max(1, interval_steps)

    return interval_sec, interval_steps


def _seconds_to_hours_minutes(duration_sec: float) -> Tuple[int, int]:
    """초(second) 단위 시간을 (시간, 분)으로 바꿉니다.

    이 함수가 하는 일
    ------------------
    - 남은 시간(또는 경과 시간)을 사람이 보기 쉬운 "몇시간 몇분" 형태로 바꾸기 위해,
      초 단위 값을 '시간'과 '분'으로 나눠줍니다.

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        duration_sec (float):
            초 단위 시간.
            - 0보다 작게 들어오면 0으로 취급합니다.

    Returns:
        Tuple[int, int]:
            (hours, minutes)
            - hours: 0 이상의 정수
            - minutes: 0~59 사이의 정수
    """
    safe_sec = max(0.0, float(duration_sec))
    hours = int(safe_sec // 3600.0)
    minutes = int((safe_sec - float(hours) * 3600.0) // 60.0)
    minutes = max(0, min(59, minutes))
    return hours, minutes


def _build_progress_eta_message(
    prefix: str,
    done_steps: int,
    total_steps: int,
    elapsed_sec: float,
) -> str:
    """진행률과 남은 예상 시간을 한 줄 메시지로 만듭니다.

    남은 예상 시간 계산 방법(단순하고 안정적으로)
    -------------------------------------------
    - 지금까지 걸린 시간(elapsed_sec)을 done_steps로 나눠서
      "스텝 1개당 평균 시간"을 구합니다.
    - 남은 스텝 수(total_steps - done_steps)에 평균 시간을 곱해서
      "앞으로 남은 시간"을 추정합니다.

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        prefix (str):
            로그 앞에 붙일 이름(예: "Validation")
        done_steps (int):
            지금까지 처리한 스텝 수(여기서는 처리한 batch 수)
        total_steps (int):
            전체 스텝 수(전체 batch 수)
        elapsed_sec (float):
            시작부터 지금까지의 경과 시간(초)

    Returns:
        str:
            예시:
            "[Validation] 진행률 12.50% (50/400) | 남은 예상 시간: 1시간 23분 | 경과: 0시간 17분"
    """
    safe_total = max(1, int(total_steps))
    safe_done = max(0, int(done_steps))
    safe_done = min(safe_done, safe_total)

    progress_percent = 100.0 * float(safe_done) / float(safe_total)

    # 평균 시간을 이용한 남은 시간 추정
    safe_elapsed = max(0.0, float(elapsed_sec))
    avg_time_per_step = safe_elapsed / float(max(1, safe_done))
    remaining_steps = max(0, safe_total - safe_done)
    remaining_sec = avg_time_per_step * float(remaining_steps)

    eta_h, eta_m = _seconds_to_hours_minutes(remaining_sec)
    elapsed_h, elapsed_m = _seconds_to_hours_minutes(safe_elapsed)

    return (f"[{prefix}] 진행률 {progress_percent:.2f}% "
            f"({safe_done}/{safe_total}) | "
            f"남은 예상 시간: {eta_h}시간 {eta_m}분 | "
            f"경과: {elapsed_h}시간 {elapsed_m}분")


def _init_progress_eta_state(
    args: argparse.Namespace,
    total_steps: int,
    prefix: str,
) -> Dict[str, Any]:
    """진행률/남은 시간 출력에 필요한 상태값을 초기화합니다.

    상태값(state)에 들어가는 내용
    -----------------------------
    - start_time_sec: 시작 시각(초 단위 숫자)
    - last_print_time_sec: 마지막으로 출력한 시각
    - last_print_step: 마지막으로 출력했을 때의 done_steps
    - print_interval_sec: 시간 기준 출력 주기
    - print_interval_steps: 스텝 기준 출력 주기
    - total_steps: 전체 스텝 수
    - prefix: 출력 메시지 접두어

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        args (argparse.Namespace):
            출력 주기 설정값을 가져오는 데 사용합니다.
        total_steps (int):
            전체 스텝 수(전체 batch 수).
        prefix (str):
            출력 메시지 접두어.

    Returns:
        Dict[str, Any]:
            출력 상태를 담은 dict.
    """
    now_sec = time.perf_counter()
    interval_sec, interval_steps = _get_progress_print_intervals(
        args=args,
        total_steps=int(total_steps),
    )

    return {
        "prefix": str(prefix),
        "total_steps": int(max(1, total_steps)),
        "start_time_sec": float(now_sec),
        "last_print_time_sec": float(now_sec),
        "last_print_step": int(0),
        "print_interval_sec": float(interval_sec),
        "print_interval_steps": int(interval_steps),
    }


def _maybe_print_progress_eta(
    state: Dict[str, Any],
    done_steps: int,
    writer: Callable[[str], None],
) -> None:
    """조건을 만족할 때만 진행률/남은 예상 시간을 출력합니다.

    출력 조건(하나라도 만족하면 출력)
    --------------------------------
    1) 첫 출력: done_steps == 1
    2) 마지막 출력: done_steps == total_steps
    3) 스텝 기준: (done_steps - last_print_step) >= print_interval_steps
    4) 시간 기준: (now - last_print_time_sec) >= print_interval_sec

    주의:
        - 이 함수는 텐서/배열을 다루지 않습니다.

    Args:
        state (Dict[str, Any]):
            _init_progress_eta_state()에서 만든 상태 dict.
        done_steps (int):
            지금까지 처리한 스텝 수(배치 수).
        writer (Callable[[str], None]):
            문자열을 실제로 출력하는 함수.
            예: tqdm 객체의 data_epoch.write 또는 그냥 print

    Returns:
        None
    """
    total_steps = int(state.get("total_steps", 1))
    total_steps = max(1, total_steps)

    safe_done = int(done_steps)
    safe_done = max(0, min(safe_done, total_steps))

    now_sec = time.perf_counter()
    last_print_time_sec = float(state.get("last_print_time_sec", now_sec))
    last_print_step = int(state.get("last_print_step", 0))

    interval_sec = float(state.get("print_interval_sec", 300.0))
    interval_steps = int(
        state.get("print_interval_steps", max(1, total_steps // 20)))
    interval_sec = max(1.0, interval_sec)
    interval_steps = max(1, interval_steps)

    should_print = False
    if safe_done == 1 or safe_done == total_steps:
        should_print = True
    if (safe_done - last_print_step) >= interval_steps:
        should_print = True
    if (now_sec - last_print_time_sec) >= interval_sec:
        should_print = True

    if not should_print:
        return

    elapsed_sec = now_sec - float(state.get("start_time_sec", now_sec))
    prefix = str(state.get("prefix", "Progress"))

    message = _build_progress_eta_message(
        prefix=prefix,
        done_steps=safe_done,
        total_steps=total_steps,
        elapsed_sec=float(elapsed_sec),
    )
    writer(message)

    state["last_print_time_sec"] = float(now_sec)
    state["last_print_step"] = int(safe_done)


def validation_epoch(
    data_loader: DataLoader,
    model: nn.Module,
    args: argparse.Namespace,
    ema: Optional[ModelEma],
    batch_num_in_one_val_epoch: int,
    wosac_metrics: WOSACMetrics,
    wosac_submission: WOSACSubmission,
    min_ade: minADE,
) -> Dict[str, torch.Tensor]:
    model.eval()
    # if args.ddp:
    #     torch.cuda.synchronize()

    total_batch_steps = int(max(1, batch_num_in_one_val_epoch))

    should_show_progress: bool = _should_show_validation_progress(args)

    progress_state: Optional[Dict[str, Any]] = None
    if should_show_progress:
        progress_state = _init_progress_eta_state(
            args=args,
            total_steps=total_batch_steps,
            prefix="Validation",
        )

    with tqdm(
            data_loader,
            desc="Validation",
            unit="batch",
            total=total_batch_steps,
            dynamic_ncols=True,
            disable=(not should_show_progress),
    ) as data_epoch:
        for batch_idx, batch in enumerate(data_epoch, start=1):
            _set_validation_batch_progress_in_args(args, batch_idx,
                                                   total_batch_steps)
            _update_validation_heartbeat_stage(
                args, f"{_get_validation_batch_progress_tag(args)} | 배치 읽는 중")

            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )

            norm_inputs: Dict[str, torch.Tensor] = args.observation_normalizer(
                inputs)

            validate_func(
                args=args,
                model=model,
                ema=ema,
                norm_inputs=norm_inputs,
                outputs=outputs,
                state_normalizer=args.state_normalizer,
                observation_normalizer=args.observation_normalizer,
                wosac_metrics=wosac_metrics,
                wosac_submission=wosac_submission,
                min_ade=min_ade,
            )

            # if args.ddp:
            #     torch.cuda.synchronize()

            # ✅ rank=0 + verbose=True 일 때만, 간단 진행률/ETA 출력
            if progress_state is not None:
                _maybe_print_progress_eta(
                    state=progress_state,
                    done_steps=batch_idx,
                    writer=data_epoch.write,
                )
    ddp_rank: int = int(ddp.get_rank()) if bool(getattr(args, "ddp",
                                                        False)) else 0
    if ddp_rank == 0:
        if wosac_submission.is_active:
            wosac_submission.save_sub_file()
    if wosac_metrics.is_active:
        epoch_wosac_metrics: Dict[str, torch.Tensor] = wosac_metrics.compute()
        if min_ade.is_active:
            epoch_wosac_metrics[
                "val_closed/ADE_debug/wosac_average_displacement_error_xy"] = (
                    min_ade.compute_wosac_like_average_displacement_error())
            epoch_wosac_metrics[
                "val_closed/ADE_debug/wosac_min_average_displacement_error_xy"] = (
                    min_ade.compute_wosac_like_min_average_displacement_error())
            epoch_wosac_metrics["val_closed/min_ADE(custom)"] = min_ade.compute(
            )
            min_ade.reset()
        wosac_metrics.reset()
    else:
        epoch_wosac_metrics = {}

    return epoch_wosac_metrics


def _is_deepspeed_engine(model: nn.Module) -> bool:
    """모델이 DeepSpeed 엔진 래퍼인지 간단히 판별합니다.

    왜 필요한가?
    ------------
    - 어떤 경우엔 model이 "그냥 nn.Module"이고,
    - 어떤 경우엔 DeepSpeed가 감싼 엔진 객체일 수 있습니다.
    - 엔진 객체는 보통 step/backward/module 같은 속성을 가집니다.

    Args:
        model (nn.Module): 검사할 모델(또는 엔진)

    Returns:
        bool:
            - True  : DeepSpeed 엔진으로 보이는 경우
            - False : 일반 nn.Module로 보이는 경우
    """
    has_module = hasattr(model, "module")
    has_train_step_api = hasattr(model, "step") or hasattr(model, "backward")
    return bool(has_module and has_train_step_api)


def _forward_model_for_validation(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """검증에서 모델 forward를 1번 수행하고, decoder_output만 반환합니다.

    Args:
        args (Any):
            args.use_deepspeed, args.device 등을 사용합니다.
        model (nn.Module):
            실제 forward를 수행할 모델.
            - EMA 모델(ema.ema)일 수도 있고, 원본 모델일 수도 있습니다.
        norm_inputs (Dict[str, torch.Tensor]):
            모델 입력 dict(정규화된 관측값).
            주요 텐서 예:
              - ego_agent_past: (B, time_len, 11)
              - near_agents_past: (B, Pnn, time_len, 11)
              - target_future_valid: (B, (1+)Pnn, future_len)

    Returns:
        Dict[str, Any]:
            decoder_output dict.
            예:
              - integrated_trajectory: (B, (1+)Pnn, 1+T, 4)
    """
    use_deepspeed_requested = bool(getattr(args, "use_deepspeed", False))
    use_deepspeed_now = use_deepspeed_requested and _is_deepspeed_engine(model)

    if use_deepspeed_now:
        _, decoder_output = model(norm_inputs)
        return decoder_output

    device_type = torch.device(getattr(args, "device", "cuda")).type
    use_amp = (device_type == "cuda")

    if use_amp:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, decoder_output = model(norm_inputs)
        return decoder_output

    # CPU fallback
    _, decoder_output = model(norm_inputs)
    return decoder_output


def _get_cuda_rng_devices_for_fork_rng(
        device: torch.device) -> Optional[List[int]]:
    """torch.random.fork_rng에서 사용할 GPU 장치 번호 리스트를 만듭니다.

    왜 필요한가?
    ------------
    torch.random.fork_rng는 "현재 랜덤 상태를 잠깐 저장했다가, 블록이 끝나면 원래대로 복구"하는 기능입니다.
    CUDA를 쓸 때는 CPU용 랜덤 상태뿐 아니라 GPU용 랜덤 상태도 같이 관리해야 하므로,
    fork_rng에 GPU 장치 번호를 넘겨주는 게 안전합니다.

    Args:
        device (torch.device):
            현재 추론이 돌아가는 장치.
            예: torch.device("cuda"), torch.device("cuda:0"), torch.device("cpu")

    Returns:
        Optional[List[int]]:
            - CUDA 사용 시: [device_index] (예: [0])
            - CPU 사용 시: None
    """
    if device.type != "cuda":
        return None
    if not torch.cuda.is_available():
        return None

    device_index = device.index
    if device_index is None:
        device_index = int(torch.cuda.current_device())
    return [int(device_index)]


def _repeat_tensor_first_dim(tensor: torch.Tensor,
                             repeat_factor: int) -> torch.Tensor:
    """텐서의 첫 번째 차원(B)을 repeat_factor만큼 반복해 (B*R, ...)로 늘립니다.

    예시
    ----
    - 입력: tensor shape (B, T, C)
    - 출력: tensor shape (B*R, T, C)

    Args:
        tensor (torch.Tensor):
            입력 텐서.
            shape: (B, ...) 또는 (1, ...) 또는 스칼라(=dim 0)일 수 있습니다.
        repeat_factor (int):
            반복 횟수 R. 1 이상.

    Returns:
        torch.Tensor:
            반복된 텐서.
            - tensor.dim() == 0 이면 그대로 반환합니다.
            - tensor.shape[0] == B 또는 1이면 첫 번째 차원을 늘려 반환합니다.
    """
    if int(repeat_factor) <= 1:
        return tensor
    if tensor.dim() == 0:
        return tensor

    # (B, ...)이면 (B*R, ...)로 반복
    return tensor.repeat_interleave(int(repeat_factor), dim=0)


def _repeat_list_first_dim(values: List[Any], repeat_factor: int) -> List[Any]:
    """리스트를 batch 길이 기준으로 repeat_factor만큼 반복해 길이를 늘립니다.

    예시
    ----
    - 입력: 길이 B 리스트
    - 출력: 길이 B*R 리스트
      (각 원소를 R번씩 연속으로 반복)

    Args:
        values (List[Any]):
            길이 B인 리스트.
        repeat_factor (int):
            반복 횟수 R.

    Returns:
        List[Any]:
            길이 B*R인 리스트.
    """
    if int(repeat_factor) <= 1:
        return list(values)

    out: List[Any] = []
    for v in values:
        out.extend([v] * int(repeat_factor))
    return out


def _expand_value_for_rollout_batch(
    value: Any,
    batch_size: int,
    rollout_repeat: int,
) -> Any:
    """입력 값(value)을 'rollout을 batch로 펼친' 형태에 맞게 확장합니다.

    확장 규칙(안전하게)
    -------------------
    1) torch.Tensor:
       - shape[0] == batch_size 인 경우: (B, ...) -> (B*R, ...)
       - shape[0] == 1 인 경우도: (1, ...) -> (1*R, ...) 형태가 되지만,
         실제로는 batch와 맞추기 위해 (B*R, ...)로 늘리는 게 더 자연스러운 경우가 많습니다.
         다만 여기서는 "현재 코드 안정성"을 위해 아래처럼 처리합니다:
           - shape[0] == batch_size: repeat_interleave로 확장
           - 그 외: 그대로 둠
       (즉, 확실히 batch축이라고 판단되는 경우만 확장합니다.)

    2) list:
       - len(value) == batch_size 인 경우: 길이 B -> 길이 B*R

    3) dict / tuple:
       - 내부 원소를 재귀적으로 같은 규칙으로 처리합니다.

    4) 그 외:
       - 그대로 둡니다.

    Args:
        value (Any):
            norm_inputs의 한 값.
        batch_size (int):
            원래 배치 크기 B. shape: ()
        rollout_repeat (int):
            이번에 동시에 처리할 rollout 개수 R. shape: ()

    Returns:
        Any:
            rollout batch 형태에 맞게 확장된 값.
    """
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and int(value.shape[0]) == int(batch_size):
            # (B, ...) -> (B*R, ...)
            return _repeat_tensor_first_dim(value, rollout_repeat)
        return value

    if isinstance(value, list):
        if len(value) == int(batch_size):
            return _repeat_list_first_dim(value, rollout_repeat)
        return list(value)

    if isinstance(value, dict):
        return {
            k: _expand_value_for_rollout_batch(v, batch_size, rollout_repeat)
            for k, v in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _expand_value_for_rollout_batch(v, batch_size, rollout_repeat)
            for v in value)

    return value


def _expand_norm_inputs_for_rollout_batch(
    norm_inputs: Dict[str, Any],
    batch_size: int,
    rollout_repeat: int,
) -> Dict[str, Any]:
    """norm_inputs를 rollout을 batch 차원으로 펼친 형태로 확장합니다.

    목표 shape
    ---------
    - 원래: B
    - 확장: B*R

    예시(주요 텐서)
    -------------
    - ego_agent_past:         (B, T_past, 11)      -> (B*R, T_past, 11)
    - near_agents_past:       (B, Pnn, T_past, 11) -> (B*R, Pnn, T_past, 11)
    - target_future_valid:    (B, 1+Pnn, T_fut)    -> (B*R, 1+Pnn, T_fut)
    - origin_world_pose:      (B, 4)               -> (B*R, 4)
    - scenario_id(list[str]): 길이 B               -> 길이 B*R

    Args:
        norm_inputs (Dict[str, Any]):
            정규화된 입력 dict.
        batch_size (int):
            원래 배치 크기 B.
        rollout_repeat (int):
            이번에 동시에 처리할 rollout 개수 R.

    Returns:
        Dict[str, Any]:
            batch 차원이 (B*R)로 확장된 입력 dict.
            (이 dict는 이후 과정에서 값이 바뀌어도 원본 norm_inputs에 영향이 없도록
             Tensor는 새로 만들어지는 방식으로 확장됩니다.)
    """
    out: Dict[str, Any] = {}
    for k, v in norm_inputs.items():
        out[k] = _expand_value_for_rollout_batch(
            value=v,
            batch_size=int(batch_size),
            rollout_repeat=int(rollout_repeat),
        )
    return out


def _shrink_rollout_batch_to_draw_idx(
    unnorm_inputs_copy: Dict[str, Any],
    draw_batch_idx: int,
) -> Dict[str, Any]:
    for k, v in unnorm_inputs_copy.items():
        if isinstance(v, torch.Tensor):
            unnorm_inputs_copy[k] = v[draw_batch_idx]
        elif isinstance(v, list):
            unnorm_inputs_copy[k] = [v[draw_batch_idx]]
    #
    return unnorm_inputs_copy


def _torch_to_numpy(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu().numpy()
        else:
            out[k] = v
    return out


def _prepare_data_for_draw(
    args,
    norm_inputs: Dict[str, Any],
    draw_batch_idx: int,
) -> Tuple[
        str,
        torch.Tensor,
        str,
]:
    """draw용으로 scenario_id와 near_target_id를 준비합니다.

    Args:
        norm_inputs (Dict[str, Any]):
            - scenario_id: 길이 B 리스트
            - target_id: 길이 (1+Pnn) 리스트
    """
    draw_scenario_id = norm_inputs["scenario_id"][draw_batch_idx]  # str
    draw_target_id = norm_inputs["target_id"][draw_batch_idx]  # ((1+)Pnn)
    draw_near_target_id = draw_target_id[1:]  # (Pnn,)

    save_dir = os.path.join(args.save_path, f"debug_vis_{draw_scenario_id}")
    os.makedirs(save_dir, exist_ok=True)

    return draw_scenario_id, draw_near_target_id, save_dir


from typing import Any, Dict, Optional, Tuple

import torch


def _predict_rollouts_batched_one_chunk(
    args: Any,
    model: torch.nn.Module,
    norm_inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    state_normalizer: Any,
    observation_normalizer: "ObservationNormalizer",
    rollout_repeat: int,
    rollout_start_idx: int,
    base_seed: int,
    ddp_rank: int,
    batch_size: int,
    one_or_pnn: int,
    save_image: bool,
    save_video: bool,
    draw_batch_idx: int = 0,
) -> torch.Tensor:
    """rollout을 batch로 펼쳐서(=B*R) 한 번에 autoregressive rollout을 생성합니다.

    최적화(핵심)
    -----------
    1) step마다 ObservationNormalizer.inverse(...)를 반복하지 않습니다.
       - chunk 시작 시점에 unnorm 입력 dict를 한 번 만들고 계속 유지합니다.
    2) lanes/route_lanes/static_objects 유효 마스크를 한 번 만들고 재사용합니다.
       - _transform_origin 내부의 torch.any(...) 스캔을 줄입니다.
    3) (추가) time-chunk(N스텝) 방식:
       - args.rollout_time_chunk_size=N (없으면 1)
       - 모델은 chunk마다 1번만 실행하고,
         integrated_trajectory의 1..gap 포즈를 world로 변환해 슬라이스로 저장합니다.
       - past 업데이트는 gap개를 한 번에 shift+concat 합니다.
       - 좌표 기준 변환은 gap번째(가장 미래) ego 포즈로 1번만 합니다.
       - draw는 chunk마다 1장만 저장합니다.

    Returns:
        torch.Tensor:
            shape: (B, (1+)Pnn, rollout_repeat, future_len, 4)
    """
    # (B, future_len, 4)
    ego_future_gt_4_dim = outputs["ego_future_gt_4_dim"]
    norm_ego_future_gt_4_dim = state_normalizer(ego_future_gt_4_dim)
    norm_inputs["ego_future_gt_4_dim"] = norm_ego_future_gt_4_dim

    # (B, Pnn, future_len, 4)
    near_future_gt_4_dim = outputs["near_future_gt_4_dim"]
    norm_near_future_gt_4_dim = state_normalizer(near_future_gt_4_dim)
    norm_inputs["near_future_gt_4_dim"] = norm_near_future_gt_4_dim

    # -----------------------------------------
    # ✅ 저장 요청값은 보관만 하고,
    #    카운트 증가는 "첫 그림 그리기 직전"에만 수행
    # -----------------------------------------
    save_image_requested: bool = bool(save_image)
    save_video_requested: bool = bool(save_video)

    save_image = False
    save_video = False
    vis_slot_checked: bool = False

    draw_scenario_id: str = ""
    save_dir: str = ""
    draw_near_target_id: Optional[torch.Tensor] = None  # (Pnn,) 또는 None

    future_len: int = int(getattr(args, "future_len"))
    rollout_repeat = int(rollout_repeat)
    merged_batch: int = int(batch_size * rollout_repeat)

    # (B, ...) -> (B*R, ...)
    norm_inputs_copy_init: Dict[str,
                                Any] = _expand_norm_inputs_for_rollout_batch(
                                    norm_inputs=norm_inputs,
                                    batch_size=int(batch_size),
                                    rollout_repeat=int(rollout_repeat),
                                )

    # ✅ unnorm 입력을 chunk 시작에 1번만 만들어 유지
    unnorm_inputs_copy: Dict[str, Any] = _initialize_unnorm_inputs_for_rollout(
        norm_inputs_copy=norm_inputs_copy_init,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
    )

    # ✅ lanes/route_lanes/static_objects 유효 마스크 캐시(1회)
    cached_valid_masks: Dict[
        str, torch.Tensor] = _build_cached_valid_masks_for_static_map_features(
            unnorm_inputs_copy=unnorm_inputs_copy,)

    # origin_world_pose는 world 기준으로 쓰는 값이므로 "unnorm"을 기대합니다.
    unnorm_origin_pose_world = unnorm_inputs_copy.get("origin_world_pose", None)
    if not isinstance(unnorm_origin_pose_world, torch.Tensor):
        raise KeyError("norm_inputs에 'origin_world_pose'(torch.Tensor)가 필요합니다.")
    if unnorm_origin_pose_world.dim() != 2 or int(
            unnorm_origin_pose_world.shape[-1]) != 4:
        raise ValueError("origin_world_pose는 (B*R, 4) 형태여야 합니다. "
                         f"현재 shape={tuple(unnorm_origin_pose_world.shape)}")

    # 출력 미리 할당: (B*R, (1+)Pnn, future_len, 4)
    target_joint_scene_world = torch.empty(
        (merged_batch, one_or_pnn, future_len, 4),
        device=unnorm_origin_pose_world.device,
        dtype=unnorm_origin_pose_world.dtype,
    )

    # ✅ time-chunk 크기 N (없으면 1)
    requested_time_chunk_size = int(getattr(args, "rollout_time_chunk_size", 1))
    requested_time_chunk_size = int(max(1, requested_time_chunk_size))

    ego_agent_past_unnorm = unnorm_inputs_copy.get("ego_agent_past", None)
    if not isinstance(ego_agent_past_unnorm, torch.Tensor):
        raise KeyError(
            "unnorm_inputs_copy에 'ego_agent_past'(torch.Tensor)가 필요합니다.")
    past_len = int(ego_agent_past_unnorm.shape[1])
    past_len = int(max(1, past_len))

    # (A) N > past_len 방지: time_chunk_size는 past_len 이하로 제한
    time_chunk_size = int(min(requested_time_chunk_size, past_len))

    with torch.inference_mode():
        step_start = 0
        while step_start < future_len:
            remaining = int(future_len - step_start)
            gap = int(min(time_chunk_size,
                          remaining))  # (B) 마지막 chunk는 gap < N 가능

            # ✅ chunk 시작 시점에만 unnorm -> norm 1번
            norm_inputs_copy: Dict[
                str, Any] = _build_norm_inputs_from_unnorm_inputs(
                    unnorm_inputs_copy=unnorm_inputs_copy,
                    state_normalizer=state_normalizer,
                    observation_normalizer=observation_normalizer,
                )

            # inference_noise: (B*R, (1+)Pnn, future_len, 4)
            inference_noise = _build_inference_noise_for_rollout_chunk(
                reference_tensor=norm_inputs_copy[
                    "ego_agent_past"],  # device/dtype 기준
                batch_size=int(batch_size),
                one_or_pnn=int(one_or_pnn),
                future_len=int(future_len),
                rollout_start_idx=int(rollout_start_idx),
                rollout_repeat=int(rollout_repeat),
                base_seed=int(base_seed),
                ddp_rank=int(ddp_rank),
                step_idx=int(step_start),  # ✅ chunk 시작 step을 seed에 반영
                noise_std=0.5,
            )
            norm_inputs_copy["inference_noise"] = inference_noise

            decoder_output = _forward_model_for_validation(
                args=args,
                model=model,
                norm_inputs=norm_inputs_copy,
            )

            normed_trajectories = decoder_output.get("integrated_trajectory",
                                                     None)
            if normed_trajectories is None:
                raise KeyError(
                    "decoder_output에서 'integrated_trajectory'를 찾을 수 없습니다. "
                    "현재 validate_func는 integrated_trajectory를 사용하도록 구현되어 있습니다.")

            # integrated_trajectory가 실제로 제공하는 예측 길이 확인(안전)
            max_pred_step = int(
                normed_trajectories.shape[2]) - 1  # (1..max_pred_step 가능)
            if gap > max_pred_step:
                gap = int(max_pred_step)
            if gap <= 0:
                raise RuntimeError(
                    "integrated_trajectory의 시간 길이가 너무 짧습니다. "
                    f"integrated_trajectory.shape={tuple(normed_trajectories.shape)}"
                )

            # ---------------------------------------------------------
            # ✅ 첫 forward 성공 이후에만 시각화 슬롯 예약
            # ---------------------------------------------------------
            if (not vis_slot_checked) and (save_image_requested or
                                           save_video_requested):
                allowed = _reserve_visualization_budget_if_needed(args)
                vis_slot_checked = True

                if allowed:
                    save_video = bool(save_video_requested)
                    save_image = bool(save_image_requested or
                                      save_video_requested)

                    if save_image:
                        (draw_scenario_id, draw_near_target_id,
                         save_dir) = _prepare_data_for_draw(
                             args, norm_inputs, draw_batch_idx)
                else:
                    save_image = False
                    save_video = False

            # ✅ draw는 chunk마다 1번만 (step_start 기준으로 파일명 저장)
            if save_image:
                if draw_near_target_id is None:
                    raise RuntimeError(
                        "save_image=True 인데 draw_near_target_id가 준비되지 않았습니다.")

                norm_inputs_for_draw = norm_inputs_copy
                if "inference_noise" in norm_inputs_for_draw:
                    norm_inputs_for_draw = dict(norm_inputs_copy)  # 얕은 복사
                    norm_inputs_for_draw.pop("inference_noise", None)

                _draw_one_batch_one_rollout(
                    save_dir=save_dir,
                    norm_inputs_copy=norm_inputs_for_draw,
                    normed_trajectories=normed_trajectories,
                    state_normalizer=state_normalizer,
                    observation_normalizer=observation_normalizer,
                    draw_batch_idx=draw_batch_idx,
                    step_idx=int(step_start),
                    draw_near_target_id=draw_near_target_id,
                )

            # ---------------------------------------------------------
            # ✅ chunk의 1..gap 포즈를 한 번에 꺼내서 처리
            # ---------------------------------------------------------
            # ego:  (B*R, gap, 4)
            normed_ego_pose_chunk = normed_trajectories[:, 0, 1:gap + 1, :]
            # near: (B*R, Pnn, gap, 4)
            normed_near_pose_chunk = normed_trajectories[:, 1:, 1:gap + 1, :]

            # target: (B*R, (1+)Pnn, gap, 4)
            normed_target_pose_chunk = torch.cat(
                [normed_ego_pose_chunk[:, None, :, :], normed_near_pose_chunk],
                dim=1,
            )

            # unnorm: (B*R, (1+)Pnn, gap, 4)
            unnorm_target_pose_chunk = state_normalizer.inverse(
                normed_target_pose_chunk)

            # ---------------------------------------------------------
            # ✅ world 변환 + output 슬라이스 저장 (0.1초 해상도 유지)
            # ---------------------------------------------------------
            # (B*R*(1+)Pnn, gap, 4)
            unnorm_target_pose_chunk_flat = unnorm_target_pose_chunk.reshape(
                -1, gap, 4)

            # (B*R*(1+)Pnn, gap, 4) world
            unnorm_target_pose_chunk_world_flat = _covert_from_ego_to_world(
                target_poses=unnorm_target_pose_chunk_flat,
                origin_world_pose=unnorm_origin_pose_world,  # (B*R,4)
            )

            # (B*R, (1+)Pnn, gap, 4)
            unnorm_target_pose_chunk_world = unnorm_target_pose_chunk_world_flat.reshape(
                merged_batch, one_or_pnn, gap, 4)

            # (B*R, (1+)Pnn, future_len, 4) 중 [step_start:step_start+gap]만 채우기
            target_joint_scene_world[:, :, step_start:step_start +
                                     gap, :] = unnorm_target_pose_chunk_world

            # 다음 chunk를 위한 origin 갱신(ego 기준, world 좌표) - 마지막 스텝만 사용
            unnorm_origin_pose_world = unnorm_target_pose_chunk_world[:, 0,
                                                                      -1, :]  # (B*R, 4)

            # ---------------------------------------------------------
            # ✅ 입력 업데이트: gap개를 한 번에 반영 + 마지막 포즈로 1번만 좌표 변환
            # ---------------------------------------------------------
            unnorm_ego_pose_chunk = unnorm_target_pose_chunk[:,
                                                             0, :, :]  # (B*R, gap, 4)
            unnorm_near_pose_chunk = unnorm_target_pose_chunk[:,
                                                              1:, :, :]  # (B*R, Pnn, gap, 4)

            unnorm_inputs_copy = _update_merged_inputs_unnorm_inplace_for_time_chunk(
                unnorm_inputs_copy=unnorm_inputs_copy,
                unnorm_ego_pose_chunk=unnorm_ego_pose_chunk,
                unnorm_near_pose_chunk=unnorm_near_pose_chunk,
                cached_valid_masks=cached_valid_masks,
            )

            step_start += gap

    # (B*R, (1+)Pnn, future_len, 4) -> (B, R, (1+)Pnn, future_len, 4)
    target_joint_scene_world = target_joint_scene_world.reshape(
        batch_size,
        rollout_repeat,
        one_or_pnn,
        future_len,
        4,
    )
    # (B, (1+)Pnn, R, future_len, 4)
    target_joint_scene_world = target_joint_scene_world.permute(0, 2, 1, 3,
                                                                4).contiguous()

    if save_video:
        if save_dir == "" or draw_scenario_id == "":
            raise RuntimeError(
                "save_video=True 인데 save_dir / draw_scenario_id가 준비되지 않았습니다.")
        draw_machine.make_video_from_all_png(
            save_dir,
            draw_scenario_id,
            new_save_dir=args.save_path,
            run_count=args.run_count,
        )

    return target_joint_scene_world


def _get_ego_future_11(
    ego_current: np.ndarray,
    unnorm_trajectory_np: np.ndarray,
) -> np.ndarray:
    """
    unnorm_trajectory_np: ((1+)Pnn, 1+T, 4)
    """
    one_fut = unnorm_trajectory_np.shape[1]
    ego_future_11 = np.tile(ego_current[None, :], (one_fut, 1))  # (1+T, 11)
    ego_np_int_traj_4_wrt_ego = unnorm_trajectory_np[0]
    ego_future_11[:, 0:4] = ego_np_int_traj_4_wrt_ego
    return ego_future_11


def _get_near_future_11(
    near_agents_current: np.ndarray,
    unnorm_trajectory: torch.Tensor,
) -> np.ndarray:
    """
    near_future_11: (Pnn, 1+T, 11)

    처음에 near_future_11 를 만들 때, near_agents_current 를 시간 축으로 복제해서 만들자.
    """
    one_fut = unnorm_trajectory.shape[1]
    near_future_11 = np.tile(near_agents_current[:, None, :],
                             (1, one_fut, 1))  # (Pnn, 1+T, 11)
    # np_int_traj_4_wrt_ego: (Pnn, 1+T, 4)
    np_int_traj_4_wrt_ego = unnorm_trajectory[1:].cpu().numpy()
    near_future_11[:, :, 0:4] = np_int_traj_4_wrt_ego  # (Pnn, 1+T, 11)
    return near_future_11


def _draw_one_batch_one_rollout(
        save_dir: str,
        norm_inputs_copy: Dict[str, Any],
        normed_trajectories: torch.Tensor,
        state_normalizer: Any,
        observation_normalizer: ObservationNormalizer,
        draw_batch_idx: int,
        step_idx: int,
        draw_near_target_id: torch.Tensor,  # (Pnn,)
) -> None:
    """
    TODO: diff_token_to_future_gt_3_dim : Dict[str, np.ndarray] # (1+T, 3)


        ego_future_gt_4_dim: torch.Tensor # (B, future_len, 4)
        near_future_gt_4_dim: torch.Tensor # (B, Pnn, future_len, 4)
    """
    # 역정규화: ((1+)Pnn, 1+T, 4)
    unnorm_trajectory = state_normalizer.inverse(
        normed_trajectories[draw_batch_idx])
    unnorm_trajectory_np = unnorm_trajectory.cpu().numpy()
    norm_ego_future_gt_4_dim = norm_inputs_copy[
        "ego_future_gt_4_dim"]  # (B*R, future_len, 4)
    norm_near_future_gt_4_dim = norm_inputs_copy[
        "near_future_gt_4_dim"]  # (B*R, Pnn, future_len, 4)
    ego_future_gt_4_dim = state_normalizer.inverse(
        norm_ego_future_gt_4_dim).cpu().numpy()  # (B*R, future_len, 4)
    ego_future_gt_4_dim = ego_future_gt_4_dim[draw_batch_idx]  # (future_len, 4)
    # near_future_gt_4_dim: (B
    near_future_gt_4_dim = state_normalizer.inverse(
        norm_near_future_gt_4_dim).cpu().numpy(
        )  # (B*R, Pnn, future_len, 4) # 4 = x, y, cos(yaw), sin(yaw)
    # near_future_gt_3_dim : x, y, yaw
    near_future_gt_3_dim = np.concatenate([
        near_future_gt_4_dim[:, :, :, 0:2],
        np.arctan2(near_future_gt_4_dim[:, :, :, 3:4],
                   near_future_gt_4_dim[:, :, :, 2:3])
    ],
                                          axis=-1)
    near_future_gt_3_dim = near_future_gt_3_dim[
        draw_batch_idx]  # (Pnn, future_len, 3)

    unnorm_inputs_copy = observation_normalizer.inverse(norm_inputs_copy)
    unnorm_inputs_copy = _shrink_rollout_batch_to_draw_idx(
        unnorm_inputs_copy, draw_batch_idx)
    unnorm_inputs_np = _torch_to_numpy(unnorm_inputs_copy)

    output_data = {}
    """
    "ego_next_wp_wrt_ego" 
         (11,)
    "ego_np_int_traj_11_wrt_ego"
        (1+T, 11)

    "diff_token_to_np_int_traj_11_wrt_ego"
        (1 + future_len, 11)


    """
    ego_current = unnorm_inputs_np["ego_agent_past"][-1]  # (11)
    # ego_future_11 : (1+T, 11)
    ego_future_11 = _get_ego_future_11(ego_current, unnorm_trajectory_np)
    output_data["ego_np_int_traj_11_wrt_ego"] = ego_future_11
    output_data["ego_next_wp_wrt_ego"] = ego_future_11[1]
    ###############################
    ego_gt_future_11 = ego_future_11[1:, :].copy()
    ego_gt_future_11[:, 0:4] = ego_future_gt_4_dim  # (future_len, 4)
    unnorm_inputs_np["ego_future_gt_11_dim"] = ego_gt_future_11  # (T, 11)
    diff_token_to_future_gt_3_dim = {}
    for target_id, gt_future_3 in zip(draw_near_target_id,
                                      near_future_gt_3_dim):
        diff_token_to_future_gt_3_dim[
            f"{target_id}"] = gt_future_3  # (future_len, 3)
    unnorm_inputs_np[
        "diff_token_to_future_gt_3_dim"] = diff_token_to_future_gt_3_dim
    ###############################
    near_agents_current = unnorm_inputs_np[
        "near_agents_past"][:, -1, :]  # (Pnn, 11)
    # near_future_11: (Pnn, 1+T, 11)
    near_future_11 = _get_near_future_11(near_agents_current, unnorm_trajectory)
    diff_token_to_np_int_traj_11_wrt_ego = {}
    for target_id, np_int_traj_11 in zip(draw_near_target_id, near_future_11):
        # np_int_traj_11: (1+T, 11)
        diff_token_to_np_int_traj_11_wrt_ego[f"{target_id}"] = np_int_traj_11
    output_data[
        "diff_token_to_np_int_traj_11_wrt_ego"] = diff_token_to_np_int_traj_11_wrt_ego
    draw_machine.draw_world_model_to_png(unnorm_inputs_np,
                                         output_data=output_data,
                                         save_path=os.path.join(
                                             save_dir, f"{step_idx}.png"))


def _predict_rollouts_batched(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    state_normalizer: Any,
    observation_normalizer: ObservationNormalizer,
    rollout_number: int,
    rollout_chunk_size: int,
    base_seed: int,
    ddp_rank: int,
) -> torch.Tensor:
    """전체 rollout_number개 rollout을 '배치로 묶어서' 빠르게 예측합니다.

    변경점(중요)
    -----------
    - chunk마다 seed를 바꾸는 방식 대신,
      rollout_idx(전역) + step_idx 기준으로 noise를 외부에서 만들어 넣습니다.
    - 그래서 OOM fallback으로 chunk_size가 바뀌어도 결과가 유지됩니다.

    Returns:
        torch.Tensor:
            shape: (B, (1+)Pnn, rollout_number, future_len, 4)
    """
    rollout_number = int(rollout_number)
    rollout_chunk_size = int(max(1, rollout_chunk_size))
    rollout_chunk_size = int(min(rollout_chunk_size, rollout_number))

    ego_agent_past = norm_inputs.get("ego_agent_past", None)
    target_future_valid = norm_inputs.get("target_future_valid", None)

    batch_size: int = int(target_future_valid.shape[0])
    one_or_pnn: int = int(target_future_valid.shape[1])
    future_len: int = int(getattr(args, "future_len"))

    out = torch.empty(
        (batch_size, one_or_pnn, rollout_number, future_len, 4),
        device=ego_agent_past.device,
        dtype=ego_agent_past.dtype,
    )

    start = 0
    trial = 0
    while start < rollout_number:
        cur = int(min(rollout_chunk_size, rollout_number - start))

        chunk_pred = _predict_rollouts_batched_one_chunk(
            args=args,
            model=model,
            norm_inputs=norm_inputs,
            outputs=outputs,
            state_normalizer=state_normalizer,
            observation_normalizer=observation_normalizer,
            rollout_repeat=int(cur),
            rollout_start_idx=int(start),
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            batch_size=batch_size,
            one_or_pnn=one_or_pnn,
            save_image=args.save_image and trial == 0,
            save_video=args.save_video and trial == 0,
        )  # (B, (1+)Pnn, cur, future_len, 4)

        out[:, :, start:start + cur, :, :] = chunk_pred
        start += cur
        trial += 1

    return out


from typing import Optional
import torch


def _make_rollout_step_seed(
    base_seed: int,
    ddp_rank: int,
    rollout_idx: int,
    step_idx: int,
) -> int:
    """rollout 인덱스 + step 인덱스까지 반영한 seed를 만듭니다.

    Args:
        base_seed (int):
            기본 seed 값. shape: ()
        ddp_rank (int):
            프로세스 rank. shape: ()
        rollout_idx (int):
            전역 rollout 인덱스(0부터). shape: ()
        step_idx (int):
            autoregressive step 인덱스(0부터). shape: ()

    Returns:
        int:
            이 (rollout_idx, step_idx)에만 대응되는 seed 값. shape: ()
    """
    base = _make_rollout_seed(
        base_seed=int(base_seed),
        ddp_rank=int(ddp_rank),
        rollout_idx=int(rollout_idx),
    )
    return int(base) + int(step_idx)


def _build_inference_noise_for_rollout_chunk(
    reference_tensor: torch.Tensor,
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_start_idx: int,
    rollout_repeat: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    noise_std: float = 0.5,
) -> torch.Tensor:
    """현재 chunk의 (B*R) 배치에 넣을 inference noise를 만듭니다.

    이 함수의 목표는 “chunk_size가 바뀌어도” 같은 (rollout_idx, step_idx)에서는
    항상 같은 noise가 나오도록 만드는 것입니다.

    생성 규칙
    --------
    - 전역 rollout 인덱스 g = rollout_start_idx + r (r=0..R-1)
    - seed = base_seed + rank + g (기존 규칙 유지) + step_idx
    - 각 rollout(g)마다 (B, one_or_pnn, future_len, 4) noise를 만들고,
      이를 (B, R, one_or_pnn, future_len, 4)에 채운 뒤,
      최종적으로 (B*R, one_or_pnn, future_len, 4)로 펼쳐 반환합니다.

    Args:
        reference_tensor (torch.Tensor):
            device/dtype 기준용 텐서.
            shape: 아무거나 가능 (값은 사용 안 함)
        batch_size (int):
            원래 배치 크기 B. shape: ()
        one_or_pnn (int):
            (1+Pnn) 크기. shape: ()
        future_len (int):
            모델이 한 번에 예측하는 미래 길이. shape: ()
        rollout_start_idx (int):
            이번 chunk가 담당하는 전역 rollout 시작 인덱스. shape: ()
        rollout_repeat (int):
            이번 chunk의 rollout 개수 R. shape: ()
        base_seed (int):
            기본 seed 값. shape: ()
        ddp_rank (int):
            프로세스 rank. shape: ()
        step_idx (int):
            autoregressive step 인덱스. shape: ()
        noise_std (float):
            noise 표준편차. 기본 0.5. shape: ()

    Returns:
        torch.Tensor:
            inference_noise 텐서.
            shape: (B*R, one_or_pnn, future_len, 4)
    """
    if int(rollout_repeat) <= 0:
        raise ValueError(
            f"rollout_repeat는 1 이상이어야 합니다. rollout_repeat={rollout_repeat}")
    if int(batch_size) <= 0:
        raise ValueError(f"batch_size는 1 이상이어야 합니다. batch_size={batch_size}")
    if int(one_or_pnn) <= 0:
        raise ValueError(f"one_or_pnn는 1 이상이어야 합니다. one_or_pnn={one_or_pnn}")
    if int(future_len) <= 0:
        raise ValueError(f"future_len은 1 이상이어야 합니다. future_len={future_len}")

    device = reference_tensor.device
    dtype = reference_tensor.dtype

    # noise_stack: (B, R, one_or_pnn, future_len, 4)
    noise_stack = torch.empty(
        (int(batch_size), int(rollout_repeat), int(one_or_pnn), int(future_len),
         4),
        device=device,
        dtype=dtype,
    )

    gen = torch.Generator(device=device)

    for r in range(int(rollout_repeat)):
        global_rollout_idx = int(rollout_start_idx) + int(r)
        seed = _make_rollout_step_seed(
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            rollout_idx=int(global_rollout_idx),
            step_idx=int(step_idx),
        )
        gen.manual_seed(int(seed))

        # noise_r: (B, one_or_pnn, future_len, 4)
        noise_r = torch.randn(
            (int(batch_size), int(one_or_pnn), int(future_len), 4),
            device=device,
            dtype=dtype,
            generator=gen,
        ) * float(noise_std)

        noise_stack[:, r, :, :, :] = noise_r

    # (B, R, ...) -> (B*R, ...)
    return noise_stack.reshape(
        int(batch_size) * int(rollout_repeat),
        int(one_or_pnn),
        int(future_len),
        4,
    )


def _make_rollout_seed(
    base_seed: int,
    ddp_rank: int,
    rollout_idx: int,
) -> int:
    """rollout_idx마다 다른 샘플링이 나오도록 seed를 만듭니다.

    목표
    -----
    - rollout_idx가 다르면 seed도 달라져서, 모델 내부에서 뽑는 랜덤 노이즈가 달라지게 합니다.
    - ddp_rank도 섞어서, 멀티프로세스에서 seed 충돌 가능성을 줄입니다.

    Args:
        base_seed (int):
            args.seed 같은 기본 seed 값.
        ddp_rank (int):
            DDP global rank. DDP를 안 쓰면 0을 넣으면 됩니다.
        rollout_idx (int):
            rollout 인덱스 (0 ~ ROLLOUT_NUMBER-1)

    Returns:
        int:
            torch.manual_seed에 넣을 seed 값.
    """
    # 숫자들은 "겹치지 않게 섞는 용도"이며, 너무 큰 의미는 없습니다.
    return int(base_seed) + int(ddp_rank) * 100_000 + int(rollout_idx) * 1_000


def _get_cached_rollout_chunk_size_for_oom_fallback(
    args: Any,
    rollout_number: int,
    requested_chunk_size: int,
) -> int:
    """이번 배치에서 처음 시도할 rollout_chunk_size를 정합니다.

    목적
    ----
    rollout을 한 번에 많이 묶으면 빠르지만, GPU 메모리가 부족하면 실패할 수 있습니다.
    한 번 실패하면 같은 배치를 다시 시도하느라 시간이 크게 늘어납니다.

    그래서 이 함수는:
    - 예전에 "성공했던 chunk 크기"가 있으면 그 값을 기억했다가,
      다음 배치에서는 그 값보다 크게 시작하지 않게 합니다.
    - 이렇게 하면, 실패(메모리 부족)로 인한 재시도가 매 배치마다 반복되는 일을 줄일 수 있습니다.

    Args:
        args (Any):
            args 안에 아래 값이 있을 수 있습니다.
            - args._dp_cached_rollout_chunk_size (선택): 이전 배치에서 성공했던 chunk 크기. shape: ()
        rollout_number (int):
            전체 rollout 개수 R. shape: ()
        requested_chunk_size (int):
            원래 설정된 chunk 크기. shape: ()

    Returns:
        int:
            이번 배치에서 "처음" 시도할 chunk 크기. shape: ()
            - 1 이상, rollout_number 이하
            - cached 값이 있으면, requested와 cached 중 작은 값으로 시작합니다.
    """
    r = int(max(1, int(rollout_number)))
    requested = int(max(1, min(int(requested_chunk_size), r)))

    cached_raw = getattr(args, "_dp_cached_rollout_chunk_size", None)
    if cached_raw is None:
        return requested

    try:
        cached = int(cached_raw)
    except (TypeError, ValueError):
        return requested

    cached = int(max(1, min(cached, r)))
    return int(min(requested, cached))


def _set_cached_rollout_chunk_size_for_oom_fallback(
    args: Any,
    chunk_size: int,
) -> None:
    """이번 배치에서 성공한 rollout_chunk_size를 args에 저장합니다.

    목적
    ----
    한 번 메모리 부족(OOM)로 실패하면, 같은 배치를 다시 계산해야 해서 시간이 많이 듭니다.
    그래서 "이번에 성공한 chunk 크기"를 저장해 두고,
    다음 배치부터는 그 값으로 바로 시작하도록 합니다.

    Args:
        args (Any):
            값을 저장할 args 객체. shape: ()
        chunk_size (int):
            이번에 실제로 성공한 chunk 크기. shape: ()

    Returns:
        None
    """
    try:
        setattr(args, "_dp_cached_rollout_chunk_size", int(chunk_size))
    except Exception:
        # args가 특이한 객체여서 setattr이 실패해도,
        # 캐시 저장은 성능 최적화용이므로 실행을 멈추지 않습니다.
        return


def _predict_rollouts_batched_with_oom_fallback(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    state_normalizer: StateNormalizer,
    observation_normalizer: ObservationNormalizer,
    rollout_number: int,
    requested_rollout_chunk_size: int,
    base_seed: int,
    ddp_rank: int,
) -> torch.Tensor:
    """rollout을 batch 차원으로 펼쳐 빠르게 예측하되, GPU 메모리가 부족하면 chunk 크기를 자동으로 줄입니다.

    추가 최적화(중요)
    --------------
    - 한 번이라도 성공했던 chunk 크기를 args에 저장해 두고,
      다음 배치부터는 그 값(또는 그보다 작은 값)으로 바로 시작합니다.
    - 이렇게 하면 "매 배치마다 32->16->8->..." 같은 실패 재시도가 반복되는 시간을 줄일 수 있습니다.

    Returns:
        torch.Tensor:
            shape: (B, (1+)Pnn, rollout_number, future_len, 4)
    """
    rollout_number_i = int(rollout_number)
    rollout_number_i = int(max(1, rollout_number_i))

    # ✅ 1) 이번 배치에서 "처음 시도할" chunk_size 결정 (캐시 반영)
    start_chunk_size = _get_cached_rollout_chunk_size_for_oom_fallback(
        args=args,
        rollout_number=int(rollout_number_i),
        requested_chunk_size=int(requested_rollout_chunk_size),
    )
    start_chunk_size = int(max(1, min(start_chunk_size, rollout_number_i)))

    # ✅ 2) 시도 후보 만들기: "start_chunk_size"부터 시작해서 더 작은 값만 시도
    #     (큰 값(예: 32)을 다시 시도해서 또 실패하는 일을 줄이기 위함)
    candidates: List[int] = [int(start_chunk_size)]
    for c in (32, 16, 8, 4, 2, 1):
        if c <= rollout_number_i and c < start_chunk_size and c not in candidates:
            candidates.append(int(c))

    last_oom: Optional[RuntimeError] = None

    for chunk_size in candidates:
        try:
            out = _predict_rollouts_batched(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                outputs=outputs,
                state_normalizer=state_normalizer,
                observation_normalizer=observation_normalizer,
                rollout_number=int(rollout_number_i),
                rollout_chunk_size=int(chunk_size),
                base_seed=int(base_seed),
                ddp_rank=int(ddp_rank),
            )

            # ✅ 성공한 chunk_size 저장 → 다음 배치부터는 여기서 바로 시작
            _set_cached_rollout_chunk_size_for_oom_fallback(
                args=args,
                chunk_size=int(chunk_size),
            )
            return out

        except RuntimeError as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg) or ("cuda oom" in msg)

            if not is_oom:
                raise

            last_oom = e

            if getattr(args, "device",
                       "cuda").startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if _is_main_process_for_logging(args):
                print(
                    f"[RolloutBatch] CUDA OOM 발생. rollout_chunk_size={chunk_size} 실패 -> "
                    f"더 작은 chunk로 재시도합니다.")
            continue

    assert last_oom is not None
    raise last_oom


def _build_valid_agent_mask_for_wosac(
    agent_id: torch.Tensor,
    target_future_valid: torch.Tensor,
) -> torch.Tensor:
    """WOSAC 계산에 넣을 "유효 agent"만 고르는 True/False 표시 텐서를 만듭니다.

    왜 필요한가?
    -------------
    입력 데이터는 (ego + near agents)를 고정 크기(예: 1+Pnn=128)로 맞추기 위해
    실제로 존재하지 않는 agent 칸을 0으로 채우는 경우가 많습니다.
    이때 agent_id=0 같은 값이 그대로 WOSAC로 들어가면,
    Waymo metric 내부에서 "시나리오에 없는 object_id"로 판단되어 즉시 에러가 납니다.

    동작 규칙(안전한 쪽 기준)
    ------------------------
    - agent_id가 0인 경우:
      - 그 agent가 실제로 존재하는지(target_future_valid에 True가 하나라도 있는지) 확인합니다.
      - 존재하지 않으면(전부 False면) 패딩으로 보고 제거합니다.
    - agent_id가 0이 아닌 경우:
      - 기본적으로 유지합니다.

    Args:
        agent_id (torch.Tensor):
            agent의 object_id.
            shape: (N,)
            - N = B * (1+Pnn)
        target_future_valid (torch.Tensor):
            agent별 미래 구간 유효 여부.
            shape: (B, 1+Pnn, T)
            - T = future_len

    Returns:
        torch.Tensor:
            WOSAC에 넣어도 되는 agent만 True인 표시 텐서.
            shape: (N,)
    """
    if agent_id.dim() != 1:
        raise ValueError(
            f"agent_id는 (N,) 이어야 합니다. 현재 shape={tuple(agent_id.shape)}")
    if target_future_valid.dim() != 3:
        raise ValueError("target_future_valid는 (B, 1+Pnn, T) 이어야 합니다. "
                         f"현재 shape={tuple(target_future_valid.shape)}")

    # (B, 1+Pnn, T) -> (B, 1+Pnn) -> (N,)
    agent_has_any_future: torch.Tensor = torch.any(
        target_future_valid.to(dtype=torch.bool),
        dim=-1).reshape(-1)  # shape: (N,)

    if int(agent_has_any_future.shape[0]) != int(agent_id.shape[0]):
        raise ValueError(
            "agent_id의 길이(N)와 target_future_valid에서 펼친 길이가 다릅니다. "
            f"N(agent_id)={int(agent_id.shape[0])}, "
            f"N(from target_future_valid)={int(agent_has_any_future.shape[0])}")

    # 패딩으로 자주 쓰이는 케이스: agent_id == 0 이면서 미래 valid가 전부 False
    is_padded_zero: torch.Tensor = (agent_id == 0) & (~agent_has_any_future
                                                     )  # (N,)
    valid_mask: torch.Tensor = ~is_padded_zero  # (N,)

    return agent_has_any_future


def _filter_rollout_tensors_by_agent_mask(
    agent_id: torch.Tensor,
    agent_batch: torch.Tensor,
    pred_traj: torch.Tensor,
    pred_z: torch.Tensor,
    pred_head: torch.Tensor,
    agent_valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor]:
    """agent_valid_mask가 True인 agent만 남기도록 rollout 관련 텐서들을 같이 걸러냅니다.

    Args:
        agent_id (torch.Tensor):
            shape: (N,)
        agent_batch (torch.Tensor):
            각 agent가 어느 배치(0~B-1)에 속하는지.
            shape: (N,)
        pred_traj (torch.Tensor):
            예측 궤적(x,y).
            shape: (N, R, T, 2)
        pred_z (torch.Tensor):
            예측 높이(z).
            shape: (N, R, T)
        pred_head (torch.Tensor):
            예측 heading(rad).
            shape: (N, R, T)
        agent_valid_mask (torch.Tensor):
            남길 agent는 True.
            shape: (N,)

    Returns:
        Tuple[torch.Tensor, ...]:
            (filtered_agent_id, filtered_agent_batch, filtered_pred_traj,
             filtered_pred_z, filtered_pred_head)

            - filtered_agent_id:   (N2,)
            - filtered_agent_batch:(N2,)
            - filtered_pred_traj:  (N2, R, T, 2)
            - filtered_pred_z:     (N2, R, T)
            - filtered_pred_head:  (N2, R, T)
    """
    if agent_valid_mask.dim() != 1:
        raise ValueError("agent_valid_mask는 (N,) 이어야 합니다. "
                         f"현재 shape={tuple(agent_valid_mask.shape)}")
    if int(agent_id.shape[0]) != int(agent_valid_mask.shape[0]):
        raise ValueError("agent_id와 agent_valid_mask의 길이가 다릅니다. "
                         f"N(agent_id)={int(agent_id.shape[0])}, "
                         f"N(mask)={int(agent_valid_mask.shape[0])}")

    # 첫 차원(N)을 기준으로 동일하게 필터링
    filtered_agent_id = agent_id[agent_valid_mask]
    filtered_agent_batch = agent_batch[agent_valid_mask]
    filtered_pred_traj = pred_traj[agent_valid_mask]
    filtered_pred_z = pred_z[agent_valid_mask]
    filtered_pred_head = pred_head[agent_valid_mask]

    return (
        filtered_agent_id,
        filtered_agent_batch,
        filtered_pred_traj,
        filtered_pred_z,
        filtered_pred_head,
    )


def _prepare_inference_model_for_validation(
    model: nn.Module,
    ema: Optional[ModelEma],
) -> nn.Module:
    """Validation에서 실제 예측에 사용할 모델을 고르고 eval 모드로 둡니다.

    하는 일
    ------
    1) ema가 있으면 ema 안에 들어있는 모델(평균낸 파라미터 모델)을 우선 사용합니다.
    2) ema가 없거나 이상하면 원래 model을 사용합니다.
    3) 반환하는 모델을 eval() 상태로 둡니다.

    Args:
        model (nn.Module):
            원본 모델.
        ema (Optional[ModelEma]):
            평균낸 파라미터 모델을 담고 있을 수 있는 객체.
            - None이면 평균 모델을 쓰지 않습니다.

    Returns:
        nn.Module:
            Validation에서 forward에 사용할 모델.
    """
    inference_model = model
    if ema is not None:
        ema_model = getattr(ema, "ema", None)
        if isinstance(ema_model, nn.Module):
            inference_model = ema_model
    inference_model.eval()
    return inference_model


def _sanitize_norm_inputs_for_validation(
    norm_inputs: Dict[str, Any],) -> Dict[str, Any]:
    """입력 dict에서 텐서 값만 NaN/Inf 등을 안전한 값으로 정리합니다.

    왜 분리했나?
    ------------
    norm_inputs에는 torch.Tensor 외에도 list(str) 같은 값이 섞일 수 있습니다.
    이 함수는 텐서만 골라 정리한 뒤, 다시 원래 dict 구조로 합쳐줍니다.

    Args:
        norm_inputs (Dict[str, Any]):
            모델 입력 dict.
            - 텐서 예: (B, T, C) 등
            - 리스트 예: scenario_id: List[str] (길이 B)

    Returns:
        Dict[str, Any]:
            텐서 값들이 정리된 새 dict.
            - 텐서가 아닌 값은 그대로 유지합니다.
    """
    tensor_only: Dict[str, torch.Tensor] = {}
    non_tensor_only: Dict[str, Any] = {}

    for k, v in norm_inputs.items():
        if isinstance(v, torch.Tensor):
            tensor_only[k] = v
        else:
            non_tensor_only[k] = v

    sanitized_tensors = _sanitize_norm_inputs(tensor_only)

    merged: Dict[str, Any] = dict(non_tensor_only)
    merged.update(sanitized_tensors)
    return merged


def _get_batch_size_from_ego_agent_past(norm_inputs: Dict[str, Any]) -> int:
    """ego_agent_past에서 배치 크기(B)를 얻습니다.

    Args:
        norm_inputs (Dict[str, Any]):
            "ego_agent_past"를 포함해야 합니다.
            - ego_agent_past: torch.Tensor, shape (B, T_past, 11)

    Returns:
        int:
            배치 크기 B.
    """
    ego_agent_past = norm_inputs.get("ego_agent_past", None)
    if not isinstance(ego_agent_past, torch.Tensor):
        raise KeyError("norm_inputs에 'ego_agent_past' (torch.Tensor)가 필요합니다.")
    if ego_agent_past.dim() != 3:
        raise ValueError("ego_agent_past는 (B, T_past, 11) 형태여야 합니다. "
                         f"현재 shape={tuple(ego_agent_past.shape)}")
    return int(ego_agent_past.shape[0])


def _get_rollout_settings_for_validation(
        args: Any) -> Tuple[int, int, int, int]:
    """Validation에서 rollout(여러 샘플) 생성에 필요한 설정값을 모아줍니다.

    Args:
        args (Any):
            아래 속성이 있으면 사용합니다.
            - args.rollout_number (int): 전체 rollout 개수 R
            - args.rollout_chunk_size (int): 한 번에 묶어서 처리할 rollout 개수
            - args.seed (int): 기본 seed
            - args.ddp (bool): 분산 여부

    Returns:
        Tuple[int, int, int, int]:
            (rollout_number, requested_rollout_chunk_size, base_seed, ddp_rank)

            - rollout_number: R
            - requested_rollout_chunk_size: 한 번에 묶을 크기(1~R)
            - base_seed: 기본 seed
            - ddp_rank: 분산 rank (싱글이면 0)
    """
    rollout_number: int = int(getattr(args, "rollout_number", 32))
    requested_rollout_chunk_size: int = int(
        getattr(args, "rollout_chunk_size", rollout_number))
    requested_rollout_chunk_size = max(
        1, min(requested_rollout_chunk_size, rollout_number))

    base_seed: int = int(getattr(args, "seed", 0))
    ddp_rank: int = int(ddp.get_rank()) if bool(getattr(args, "ddp",
                                                        False)) else 0

    return rollout_number, requested_rollout_chunk_size, base_seed, ddp_rank


def _build_pred_traj_and_pred_head_from_world_rollouts(
    target_scenario_rollouts_world: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """world rollout 결과를 metric 입력 형태(pred_traj, pred_head)로 바꿉니다.

    입력 shape
    ---------
    - target_scenario_rollouts_world: (B, (1+)Pnn, R, future_len, 4)
        (B, (1+)Pnn, rollout_number, future_len, 4)

    출력 shape
    ---------
    - pred_traj: (N, R, future_len, 2)
      - N = B * (1+)Pnn
      - 마지막 2: (x, y)
    - pred_head: (N, R, future_len)
      - 값: 방향(각도) 라디안

    Args:
        target_scenario_rollouts_world (torch.Tensor):
            shape: (B, (1+)Pnn, R, future_len, 4)
    """
    # (B, (1+)Pnn, R, future_len, 2) -> (N, R, future_len, 2)
    pred_traj = target_scenario_rollouts_world[:, :, :, :, :2]
    pred_traj = pred_traj.reshape(
        pred_traj.shape[0] * pred_traj.shape[1],
        pred_traj.shape[2],
        pred_traj.shape[3],
        pred_traj.shape[4],
    )

    # heading: cos/sin -> atan2
    pred_head_cos = target_scenario_rollouts_world[:, :, :, :, 2]
    pred_head_sin = target_scenario_rollouts_world[:, :, :, :, 3]
    pred_head = torch.atan2(pred_head_sin, pred_head_cos)

    # (B, (1+)Pnn, R, future_len) -> (N, R, future_len)
    pred_head = pred_head.reshape(
        pred_head.shape[0] * pred_head.shape[1],
        pred_head.shape[2],
        pred_head.shape[3],
    )
    return pred_traj, pred_head


def _get_string_list_from_norm_inputs(
    norm_inputs: Dict[str, Any],
    key: str,
    expected_length: int,
) -> List[str]:
    """norm_inputs에서 List[str] 값을 꺼내고 길이를 확인합니다.

    Args:
        norm_inputs (Dict[str, Any]):
            입력 dict.
        key (str):
            꺼낼 키 이름. 예: "scenario_id", "tfrecord_path"
        expected_length (int):
            리스트 길이로 기대하는 값(보통 batch_size=B).

    Returns:
        List[str]:
            길이 expected_length 인 문자열 리스트.

    Raises:
        AssertionError:
            타입이 list가 아니거나 길이가 다르면 에러.
    """
    value = norm_inputs.get(key, None)
    assert isinstance(value, list), f"{key}는 List[str] 타입이어야 합니다."
    assert int(len(value)) == int(expected_length), (
        f"{key} 길이가 batch_size와 맞지 않습니다. "
        f"len({key})={len(value)}, expected={expected_length}")
    # (내용이 str인지까지 강하게 검사하면 비용이 늘 수 있어, 여기서는 리스트 타입/길이만 보장합니다)
    return value


def _get_target_id_flat_from_norm_inputs(
        norm_inputs: Dict[str, Any]) -> torch.Tensor:
    """target_id를 (N,) 형태로 펼쳐서 반환합니다.

    기대 입력/출력 shape
    -------------------
    - 입력: target_id: (B, (1+)Pnn)  (A = (1+)Pnn)
    - 출력: target_id_flat: (N,)  (N = B*(1+)Pnn)

    Args:
        norm_inputs (Dict[str, Any]):
            "target_id"를 포함해야 합니다.

    Returns:
        torch.Tensor:
            target_id_flat, shape (N,)
    """
    target_id = norm_inputs.get("target_id", None)
    if not isinstance(target_id, torch.Tensor):
        raise KeyError("norm_inputs에 'target_id' (torch.Tensor)가 필요합니다.")
    if target_id.dim() != 2:
        raise ValueError("target_id는 (B, (1+)Pnn) 형태여야 합니다. "
                         f"현재 shape={tuple(target_id.shape)}")
    return target_id.reshape(-1)


def _build_agent_batch_tensor(
    batch_size: int,
    one_Pnn: int,
    device: torch.device,
) -> torch.Tensor:
    """각 agent가 어떤 batch(0~B-1)에 속하는지 나타내는 텐서를 만듭니다.

    출력 shape
    ---------
    - agent_batch: (N,)
      - N = B * one_Pnn
      - 예: one_Pnn=128이면,
        [0,0,0,...(128개), 1,1,1,...(128개), ..., B-1,...] 형태

    Args:
        batch_size (int):
            B
        one_Pnn (int):
            A = (1+)Pnn
        device (torch.device):
            생성될 텐서 device.

    Returns:
        torch.Tensor:
            agent_batch, shape (N,), dtype torch.long
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size는 1 이상이어야 합니다. batch_size={batch_size}")
    if one_Pnn <= 0:
        raise ValueError(f"agents_per_batch는 1 이상이어야 합니다. one_Pnn={one_Pnn}")

    batch_index = torch.arange(batch_size, device=device,
                               dtype=torch.long)  # (B,)
    agent_batch = batch_index.repeat_interleave(int(one_Pnn), dim=0)  # (N,)
    return agent_batch


def _expand_target_z_to_rollout_grid(
    norm_inputs: Dict[str, Any],
    n_rollout: int,
    n_step: int,
    device: torch.device,
) -> torch.Tensor:
    """target_z를 (N, R, future_len) 형태로 늘려서 반환합니다.

    입력/출력 shape
    --------------
    - 입력: target_z: (B, (1+)Pnn)
    - 출력: target_z_grid: (N, R, future_len)
      - N = B*(1+)Pnn
      - future_len = n_step

    Args:
        norm_inputs (Dict[str, Any]):
            "target_z"를 포함해야 합니다.
        n_rollout (int):
            R
        n_step (int):
            future_len
        device (torch.device):
            출력 텐서 device.

    Returns:
        torch.Tensor:
            target_z_grid, shape (N, R, future_len)
    """
    target_z = norm_inputs.get("target_z", None)
    if not isinstance(target_z, torch.Tensor):
        raise KeyError("norm_inputs에 'target_z' (torch.Tensor)가 필요합니다.")
    if target_z.dim() != 2:
        raise ValueError("target_z는 (B, (1+)Pnn) 형태여야 합니다. "
                         f"현재 shape={tuple(target_z.shape)}")

    target_z_flat = target_z.reshape(-1).to(device=device)  # (N,)
    target_z_grid = target_z_flat[:,
                                  None, None].expand(-1, int(n_rollout),
                                                     int(n_step)).contiguous()
    return target_z_grid


def _get_sim_agents_challenge_type_from_args(
    args: Any,) -> submission_specs.ChallengeType:
    """args에서 Sim Agents 평가 규칙 종류를 안전하게 고릅니다.

    이 함수가 필요한 이유
    -------------------
    Waymo 1.6.7 환경에서는 challenge type이 "protobuf enum"이 아니라
    submission_specs.ChallengeType(파이썬 enum) 입니다.

    Args:
        args (Any):
            아래 중 하나가 있으면 사용합니다.
            - args.sim_agents_challenge_type: str 또는 ChallengeType
            - args.challenge_type: str 또는 ChallengeType

    Returns:
        submission_specs.ChallengeType:
            - SIM_AGENTS 또는 SCENARIO_GEN
            shape: ()
    """
    raw = getattr(args, "sim_agents_challenge_type", None)
    if raw is None:
        raw = getattr(args, "challenge_type", None)

    if raw is None:
        return submission_specs.ChallengeType.SIM_AGENTS

    if isinstance(raw, submission_specs.ChallengeType):
        return raw

    raw_str = str(raw).strip().lower()

    if raw_str in ("sim_agents", "simagents", "sim_agent", "sim"):
        return submission_specs.ChallengeType.SIM_AGENTS
    if raw_str in ("scenario_gen", "scenariogen", "scenario", "gen"):
        return submission_specs.ChallengeType.SCENARIO_GEN

    raise ValueError("challenge_type을 해석할 수 없습니다. "
                     f"raw='{raw}'. 예: 'sim_agents' 또는 'scenario_gen'")


def _read_scenario_proto_from_tfrecord(
    tfrecord_path: str,
    scenario_id: str,
) -> scenario_pb2.Scenario:
    """tfrecord_path에서 scenario_id에 해당하는 Scenario proto를 읽어옵니다.

    이 함수가 하는 일
    ----------------
    - TFRecord 파일은 "Scenario proto가 여러 개 들어있는 파일"입니다.
    - 그 안을 앞에서부터 읽으면서 scenario_id가 같은 레코드를 찾습니다.
    - 찾으면 scenario_pb2.Scenario()로 파싱해서 반환합니다.

    성능 메모
    ---------
    - 한 파일에 시나리오가 여러 개 들어있으면, 찾을 때까지 순차 탐색이 필요합니다.
    - 그래서 아래의 캐시 함수(_get_evaluation_sim_agent_ids_cached)를 통해
      같은 (tfrecord_path, scenario_id) 조합이 반복될 때는 재탐색을 피할 수 있게 합니다.

    Args:
        tfrecord_path (str):
            시나리오가 들어있는 TFRecord 파일 경로. shape: ()
        scenario_id (str):
            찾고 싶은 시나리오 id 문자열. shape: ()

    Returns:
        scenario_pb2.Scenario:
            찾은 시나리오 proto.

    Raises:
        FileNotFoundError:
            tfrecord_path가 없을 때.
        ValueError:
            파일 안에서 scenario_id를 찾지 못했을 때.
        ImportError:
            tensorflow가 설치되어 있지 않을 때.
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("Scenario proto를 TFRecord에서 읽으려면 tensorflow가 필요합니다. "
                          "환경에 tensorflow가 설치되어 있는지 확인해 주세요.") from e

    tfrecord_path = str(tfrecord_path)
    scenario_id = str(scenario_id)

    compression_type = "GZIP" if tfrecord_path.endswith(".gz") else ""
    dataset = tf.data.TFRecordDataset(tfrecord_path,
                                      compression_type=compression_type)

    for raw_record in dataset:
        record_bytes = raw_record.numpy()  # bytes
        scenario = scenario_pb2.Scenario()
        scenario.ParseFromString(record_bytes)

        if str(scenario.scenario_id) == scenario_id:
            return scenario

    raise ValueError(
        "TFRecord에서 scenario_id를 찾지 못했습니다. "
        f"tfrecord_path='{tfrecord_path}', scenario_id='{scenario_id}'")


@lru_cache(maxsize=4096)
def _get_evaluation_sim_agent_ids_cached(
    tfrecord_path: str,
    scenario_id: str,
    challenge_type: submission_specs.ChallengeType,
) -> Tuple[int, ...]:
    """(tfrecord_path, scenario_id) -> 평가 대상 object_id 목록을 캐시해서 반환합니다.

    Args:
        tfrecord_path (str):
            TFRecord 파일 경로. shape: ()
        scenario_id (str):
            시나리오 id. shape: ()
        challenge_type (submission_specs.ChallengeType):
            평가 규칙 종류. shape: ()

    Returns:
        Tuple[int, ...]:
            평가에 포함할 object_id들의 튜플.
            길이 K는 시나리오마다 다를 수 있습니다.
    """
    scenario = _read_scenario_proto_from_tfrecord(
        tfrecord_path=str(tfrecord_path),
        scenario_id=str(scenario_id),
    )

    eval_ids = submission_specs.get_evaluation_sim_agent_ids(
        scenario=scenario,
        challenge_type=challenge_type,
    )

    return tuple(int(x) for x in list(eval_ids))


def _build_padded_eval_object_ids_tensor_for_batch(
    tfrecord_path: List[str],
    scenario_id: List[str],
    challenge_type: submission_specs.ChallengeType,
    device: torch.device,
    pad_value: int = 0,
) -> torch.Tensor:
    """배치(B개 시나리오)의 eval_object_ids를 (B, K) 텐서로 패딩해서 만듭니다.

    Shape
    -----
    - 입력:
      - tfrecord_path: List[str], 길이 B
      - scenario_id:   List[str], 길이 B
    - 출력:
      - eval_object_ids: torch.Tensor, shape (B, K), dtype torch.long

    Args:
        tfrecord_path (List[str]):
            배치의 TFRecord 경로 리스트. len = B
        scenario_id (List[str]):
            배치의 시나리오 id 리스트. len = B
        challenge_type (submission_specs.ChallengeType):
            평가 규칙 종류. shape: ()
        device (torch.device):
            반환 텐서 device.
        pad_value (int):
            패딩 값(기본 0).

    Returns:
        torch.Tensor:
            eval_object_ids 텐서. shape (B, K)
    """
    if int(len(tfrecord_path)) != int(len(scenario_id)):
        raise ValueError(
            "tfrecord_path와 scenario_id 길이가 다릅니다. "
            f"len(tfrecord_path)={len(tfrecord_path)}, len(scenario_id)={len(scenario_id)}"
        )

    batch_size = int(len(scenario_id))
    eval_id_lists: List[List[int]] = []
    max_k = 0

    for p, sid in zip(tfrecord_path, scenario_id):
        ids = list(
            _get_evaluation_sim_agent_ids_cached(
                tfrecord_path=str(p),
                scenario_id=str(sid),
                challenge_type=challenge_type,
            ))
        eval_id_lists.append(ids)
        if len(ids) > max_k:
            max_k = int(len(ids))

    k = int(max(1, max_k))

    out = torch.full(
        (batch_size, k),
        fill_value=int(pad_value),
        device=device,
        dtype=torch.long,
    )

    for b_idx, ids in enumerate(eval_id_lists):
        if len(ids) == 0:
            continue
        ids_tensor = torch.tensor(ids, device=device, dtype=torch.long)  # (Kb,)
        out[b_idx, :ids_tensor.numel()] = ids_tensor

    return out


def _update_min_ade_for_validation_batch(
    outputs: Dict[str, torch.Tensor],
    norm_inputs: Dict[str, Any],
    pred_traj: torch.Tensor,
    agent_batch: torch.Tensor,  # (N,)
    target_id: torch.Tensor,  # (N,)
    eval_object_ids: Optional[torch.Tensor],  # (B, K) or (K,) or None
    min_ade: minADE,
) -> None:
    """minADE metric을 업데이트합니다(평가 대상 object id를 함께 전달).

    하는 일
    ------
    1) GT(정답) 미래 궤적을 (ego 기준)에서 (world 기준)으로 바꿉니다.
    2) GT가 0으로 채워진(없는 데이터) 구간은 valid=False로 둡니다.
    3) pred_traj(예측)과 GT를 minADE에 누적 업데이트합니다.
    4) eval_object_ids가 있으면, minADE 내부에서 "평가 대상 agent만" 평가에 포함합니다.

    Shape 요약
    ---------
    - outputs["ego_future_gt_4_dim"]: (B, T, 4)
    - outputs["near_future_gt_4_dim"]: (B, Pnn, T, 4)
    - GT 합치면: (B, (1+)Pnn, T, 4)
    - 펼치면: (N, T, 4), N=B*(1+Pnn)
    - world로 바꾸고 xy만 쓰면: (N, T, 2)
    - pred_traj: (N, R, T, 2)
    - target_id: (N,)
    - eval_object_ids: (B, K) 또는 (K,)

    Args:
        outputs (Dict[str, torch.Tensor]):
            GT 텐서들이 들어있는 dict.
        norm_inputs (Dict[str, Any]):
            "origin_world_pose"가 필요합니다.
            - origin_world_pose: (B, 4)
        pred_traj (torch.Tensor):
            예측 xy 궤적.
            shape: (N, R, T, 2)
        agent_batch (torch.Tensor):
            각 agent가 어느 시나리오(batch index)에 속하는지.
            shape: (N,)
        target_id (torch.Tensor):
            agent object id.
            shape: (N,)
        eval_object_ids (Optional[torch.Tensor]):
            시나리오별 평가 대상 object id 목록.
            shape: (B, K) 또는 (K,) (없으면 None)
        min_ade (minADE):
            update(...)로 누적합니다.

    Returns:
        None
    """
    unnorm_ego_future_gt_4_dim = outputs["ego_future_gt_4_dim"]  # (B, T, 4)
    unnorm_near_future_gt_4_dim = outputs[
        "near_future_gt_4_dim"]  # (B, Pnn, T, 4)

    # (B, (1+)Pnn, T, 4)
    unnorm_target_future_gt_4_dim = torch.cat(
        [
            unnorm_ego_future_gt_4_dim[:, None, :, :],
            unnorm_near_future_gt_4_dim,
        ],
        dim=1,
    )

    # (N, T, 4)
    unnorm_target_future_gt_4_dim_flat = unnorm_target_future_gt_4_dim.reshape(
        -1,
        unnorm_target_future_gt_4_dim.shape[2],
        unnorm_target_future_gt_4_dim.shape[3],
    )

    unnorm_origin_pose_world = norm_inputs["origin_world_pose"]  # (B, 4)

    # (N, T, 4) (world)
    unnorm_target_future_gt_4_dim_world = _covert_from_ego_to_world(
        target_poses=unnorm_target_future_gt_4_dim_flat,  # (N, T, 4)
        origin_world_pose=unnorm_origin_pose_world,  # (B, 4)
    )

    # (N, T, 2)
    unnorm_target_future_gt_xy_world = unnorm_target_future_gt_4_dim_world[:, :, :
                                                                           2]

    # (N, T)  마지막 4차원이 전부 0이면 그 스텝은 무효
    target_future_valid_flat = torch.any(
        unnorm_target_future_gt_4_dim_flat != 0,
        dim=-1,
    ).to(dtype=torch.bool)
    if min_ade.is_active:
        min_ade.update(
            pred=pred_traj,
            target=unnorm_target_future_gt_xy_world,
            target_valid=target_future_valid_flat,
            agent_batch=agent_batch,
            agent_id=target_id,  # (N,)
            eval_object_ids=eval_object_ids,  # (B, K) or (K,) or None
        )


def validate_func(
    args: Any,
    model: nn.Module,
    ema: Optional[ModelEma],
    norm_inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    state_normalizer: StateNormalizer,
    observation_normalizer: ObservationNormalizer,
    wosac_metrics: WOSACMetrics,
    wosac_submission: WOSACSubmission,
    min_ade: minADE,
) -> None:
    """validation에서 예측 rollouts를 만들고 metric을 업데이트합니다."""
    tag = _get_validation_batch_progress_tag(args)

    _update_validation_heartbeat_stage(args, f"{tag} | selecting model")

    inference_model: nn.Module = _prepare_inference_model_for_validation(
        model=model, ema=ema)

    _update_validation_heartbeat_stage(args, f"{tag} | preparing inputs")

    norm_inputs = _sanitize_norm_inputs_for_validation(norm_inputs)
    future_len: int = int(getattr(args, "future_len"))
    target_future_valid = build_target_future_tensors_and_masks_for_inference(
        args,
        norm_inputs,
        future_len,
    )
    norm_inputs["target_future_valid"] = target_future_valid

    batch_size = _get_batch_size_from_ego_agent_past(norm_inputs)
    rollout_number, requested_rollout_chunk_size, base_seed, ddp_rank = _get_rollout_settings_for_validation(
        args)

    _update_validation_heartbeat_stage(
        args, f"{tag} | predicting rollouts (rollout={rollout_number})")

    target_scenario_rollouts_world = _predict_rollouts_batched_with_oom_fallback(
        args=args,
        model=inference_model,
        norm_inputs=norm_inputs,
        outputs=outputs,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
        rollout_number=int(rollout_number),
        requested_rollout_chunk_size=int(requested_rollout_chunk_size),
        base_seed=int(base_seed),
        ddp_rank=int(ddp_rank),
    )

    _update_validation_heartbeat_stage(args, f"{tag} | postprocessing rollouts")

    pred_traj, pred_head = _build_pred_traj_and_pred_head_from_world_rollouts(
        target_scenario_rollouts_world=target_scenario_rollouts_world)

    scenario_id: List[str] = _get_string_list_from_norm_inputs(
        norm_inputs=norm_inputs,
        key="scenario_id",
        expected_length=batch_size,
    )
    if args.eval_method == "validation":
        tfrecord_path: List[str] = _get_string_list_from_norm_inputs(
            norm_inputs=norm_inputs,
            key="tfrecord_path",
            expected_length=batch_size,
        )

    target_id = _get_target_id_flat_from_norm_inputs(norm_inputs)
    one_Pnn = int(target_future_valid.shape[1])

    agent_batch = _build_agent_batch_tensor(
        batch_size=batch_size,
        one_Pnn=one_Pnn,
        device=pred_traj.device,
    )

    pred_z = _expand_target_z_to_rollout_grid(
        norm_inputs=norm_inputs,
        n_rollout=int(pred_traj.shape[1]),
        n_step=int(pred_traj.shape[2]),
        device=pred_traj.device,
    )

    device = pred_traj.device

    # wosac_agent_valid_mask: (N,)
    wosac_agent_valid_mask = _build_valid_agent_mask_for_wosac(
        agent_id=target_id,
        target_future_valid=target_future_valid,
    )

    (
        wosac_agent_id,
        wosac_agent_batch,
        wosac_pred_traj,
        wosac_pred_z,
        wosac_pred_head,
    ) = _filter_rollout_tensors_by_agent_mask(
        agent_id=target_id,
        agent_batch=agent_batch,
        pred_traj=pred_traj,
        pred_z=pred_z,
        pred_head=pred_head,
        agent_valid_mask=wosac_agent_valid_mask,
    )

    need_scenario_rollouts: bool = bool(
        getattr(wosac_submission, "is_active", False) or
        getattr(wosac_metrics, "is_active", False))
    scenario_rollouts: Optional[List[
        sim_agents_submission_pb2.ScenarioRollouts]] = None
    if need_scenario_rollouts:
        _update_validation_heartbeat_stage(args,
                                           f"{tag} | packaging WOSAC inputs")
        scenario_rollouts = get_scenario_rollouts(
            scenario_id=get_scenario_id_int_tensor(scenario_id, device),
            agent_id=wosac_agent_id, # (N2,)
            agent_batch=wosac_agent_batch, # (N2, )
            pred_traj=wosac_pred_traj, #
            pred_z=wosac_pred_z,
            pred_head=wosac_pred_head,
        )

    if wosac_submission.is_active:
        _update_validation_heartbeat_stage(
            args, f"{tag} | collecting data for submission")

        wosac_submission.update(
            scenario_id=scenario_id,
            agent_id=wosac_agent_id,
            agent_batch=wosac_agent_batch,
            pred_traj=wosac_pred_traj,
            pred_z=wosac_pred_z,
            pred_head=wosac_pred_head,
            global_rank=int(ddp_rank),
        )

        world_size_now: int = int(
            ddp.get_world_size()) if ddp.is_dist_avail_and_initialized() else 1

        if world_size_now <= 1:
            if int(ddp_rank) == 0 and scenario_rollouts is not None:
                wosac_submission.aggregate_rollouts(scenario_rollouts)
        else:
            # ✅ 모든 rank가 같이 호출해야 함 (여기서 분산 통신이 일어남)
            _gpu_dict_sync = wosac_submission.compute()

            # ✅ 실제로 rollouts를 만들고 파일에 쌓는 건 rank0만
            if int(ddp_rank) == 0:
                for k in _gpu_dict_sync.keys():
                    if type(_gpu_dict_sync[k]) is list:
                        _gpu_dict_sync[k] = _gpu_dict_sync[k][0]
                scenario_rollouts_all = get_scenario_rollouts(**_gpu_dict_sync)
                wosac_submission.aggregate_rollouts(scenario_rollouts_all)

        wosac_submission.reset()

    if wosac_metrics.is_active:
        if scenario_rollouts is None:
            raise RuntimeError(
                "wosac_metrics가 active인데 scenario_rollouts가 생성되지 않았습니다.")

        _update_validation_heartbeat_stage(
            args, f"{tag} | computing WOSAC metrics (scenarios={batch_size})")
        wosac_metrics.update(tfrecord_path, scenario_rollouts, should_validate=args.validate_scenario_rollouts)

    if min_ade.is_active:
        _update_validation_heartbeat_stage(args, f"{tag} | computing minADE")

        eval_object_ids: Optional[torch.Tensor] = None
        if bool(min_ade.only_eval_targets_to_predict):
            challenge_type = _get_sim_agents_challenge_type_from_args(args)
            eval_object_ids = _build_padded_eval_object_ids_tensor_for_batch(
                tfrecord_path=tfrecord_path,
                scenario_id=scenario_id,
                challenge_type=challenge_type,
                device=pred_traj.device,
                pad_value=0,
            )

        _update_min_ade_for_validation_batch(
            outputs=outputs,
            norm_inputs=norm_inputs,
            pred_traj=pred_traj,
            agent_batch=agent_batch,
            target_id=target_id,
            eval_object_ids=eval_object_ids,
            min_ade=min_ade,
        )

    _update_validation_heartbeat_stage(args, f"{tag} | finalizing batch")


# _transform_origin에서 실제로 좌표를 바꾸는 키들(먼저 들어가는 키가 Tensor가 되도록 순서 고정)
_ORIGIN_TRANSFORM_TENSOR_KEYS_IN_ORDER: List[str] = [
    "ego_agent_past",
    "near_agents_past",
    "non_near_agents_past",
    "neighbor_agents_past",
    "ego_future_gt_4_dim",
    "near_future_gt_4_dim",
    "stop_sign_points",
    "crosswalk_points",
    "speed_bump_points",
    "driveway_points",
    "road_edge",
    "lanes",
    "route_lanes",
    "static_objects",
]

# points를 변환할 때 같이 필요한 "유효/무효" 마스크 키들
_ORIGIN_TRANSFORM_MASK_KEYS_IN_ORDER: List[str] = [
    "stop_sign_is_valid",
    "crosswalk_is_valid",
    "speed_bump_is_valid",
    "driveway_is_valid",
    "road_edge_is_valid",
]

# StateNormalizer로 따로 처리해야 하는 (x,y,cos,sin) 4차원 포즈 키들
_STATE_NORMALIZER_4DIM_KEYS: List[str] = [
    "ego_future_gt_4_dim",
    "near_future_gt_4_dim",
]


def _build_origin_transform_working_sets(
    norm_inputs_copy: Dict[str, Any],) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """원점/방향 기준 바꾸기에 필요한 값들만 골라서 묶습니다.

    이 함수가 하는 일
    ----------------
    rollout 한 스텝마다 `_transform_origin`이 여러 키의 (x, y, 방향)을 바꿉니다.
    그런데 전체 dict를 매번 다루면 불필요한 연산이 많아질 수 있어서,
    실제로 변환에 쓰이는 키만 따로 모읍니다.

    Args:
        norm_inputs_copy (Dict[str, Any]):
            모델 입력 dict(값 크기 맞춘 상태).
            예시 shape:
              - ego_agent_past: (B, T_past, 11)
              - near_agents_past: (B, Pnn, T_past, 11)
              - lanes: (B, lane_num, lane_len, 12)
              - stop_sign_points: (B, N, L, 2)
              - stop_sign_is_valid: (B, N, L) 또는 (B, N)

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            (working_tensors, working_masks)

            - working_tensors:
                `_transform_origin`이 직접 좌표를 바꿀 가능성이 큰 값들.
                (대부분 torch.Tensor)
            - working_masks:
                points 변환에 필요한 "유효/무효" 표시 값들.
                (torch.Tensor일 수도 있고, 0/1 값일 수도 있습니다)
    """
    working_tensors: Dict[str, Any] = {}
    for key in _ORIGIN_TRANSFORM_TENSOR_KEYS_IN_ORDER:
        if key in norm_inputs_copy:
            working_tensors[key] = norm_inputs_copy[key]

    working_masks: Dict[str, Any] = {}
    for key in _ORIGIN_TRANSFORM_MASK_KEYS_IN_ORDER:
        if key in norm_inputs_copy:
            working_masks[key] = norm_inputs_copy[key]

    return working_tensors, working_masks


def _transform_origin_after_restore_then_rescale_inplace(
    norm_inputs_copy: Dict[str, Any],
    normed_ego_next_pose: torch.Tensor,
    state_normalizer: StateNormalizer,
    observation_normalizer: ObservationNormalizer,
) -> Dict[str, Any]:
    """`_transform_origin`을 "원래 값" 기준으로 적용한 뒤, 다시 입력용 값 크기로 맞춥니다.

    왜 이게 필요한가?
    ----------------
    `_transform_origin`은 (x, y)를 옮기고 회전시키는 계산을 합니다.
    이 계산은 **원래 단위(예: 미터 기준의 x, y)** 에서 해야 의미가 자연스럽습니다.

    그런데 입력이 이미 "평균/표준편차로 크기를 바꾼 값" 상태라면,
    같은 이동/회전을 해도 실제 의미가 어긋날 수 있습니다.
    그래서 아래 순서로 처리합니다.

    처리 순서
    --------
    1) `_transform_origin`이 만질 값들만 골라서,
       원래 값으로 되돌립니다. (ObservationNormalizer.inverse + 일부는 StateNormalizer.inverse)
    2) ego 다음 포즈도 원래 값으로 되돌립니다.
       - normed_ego_next_pose: (B, 4) -> unnorm_ego_next_pose: (B, 4)
    3) 원래 값 상태에서 `_transform_origin`을 실행합니다.
    4) 변환된 결과를 다시 모델 입력용 값 크기로 맞춥니다.
       (ObservationNormalizer + 일부는 StateNormalizer)

    Args:
        norm_inputs_copy (Dict[str, Any]):
            모델 입력 dict(값 크기 맞춘 상태).
            주요 shape 예:
              - ego_agent_past: (B, T_past, 11)
              - near_agents_past: (B, Pnn, T_past, 11)
              - lanes: (B, lane_num, lane_len, 12)
        normed_ego_next_pose (torch.Tensor):
            ego 다음 포즈(값 크기 맞춘 상태).
            shape: (B, 4)  # (x, y, cos, sin)
        state_normalizer (StateNormalizer):
            (x, y, cos, sin) 4개 값에 대해 "원래 값 ↔ 값 크기 맞춘 값" 변환에 사용합니다.
        observation_normalizer (ObservationNormalizer):
            입력 dict 여러 키에 대해 "원래 값 ↔ 값 크기 맞춘 값" 변환에 사용합니다.

    Returns:
        Dict[str, Any]:
            norm_inputs_copy(같은 dict 객체)에 결과를 덮어쓴 뒤 반환합니다.
    """
    working_tensors, working_masks = _build_origin_transform_working_sets(
        norm_inputs_copy=norm_inputs_copy)

    # 변환할 게 없으면 그대로 반환
    if len(working_tensors) == 0:
        return norm_inputs_copy

    # 1) dict 값들을 "원래 값"으로 되돌리기
    #    unnorm_working_tensors: Dict[str, Any]
    unnorm_working_tensors: Dict[str, Any] = observation_normalizer.inverse(
        working_tensors)

    # 1-1) 일부 키는 StateNormalizer로 되돌려야 정확함 (4차원 포즈)
    for key in _STATE_NORMALIZER_4DIM_KEYS:
        if key in working_tensors and isinstance(working_tensors[key],
                                                 torch.Tensor):
            # working_tensors[key]: (..., 4)
            unnorm_working_tensors[key] = state_normalizer.inverse(
                working_tensors[key])

    # 마스크는 값 크기 변환을 거치지 않고 그대로 사용 (0/1 또는 bool이면 충분)
    unnorm_working_tensors.update(working_masks)

    # 2) ego 다음 포즈도 원래 값으로 되돌리기
    #    unnorm_ego_next_pose: (B, 4)
    unnorm_ego_next_pose: torch.Tensor = state_normalizer.inverse(
        normed_ego_next_pose)

    # 3) 원래 값 기준으로 좌표 기준 변환 수행(내부에서 in-place로 바뀜)
    _transform_origin(
        unnorm_working_tensors,  # Dict[str, Any]
        unnorm_ego_next_pose,  # (B, 4)
    )

    # 4) 다시 모델 입력용 값 크기로 맞추기
    #    norm_working_tensors: Dict[str, Any]
    norm_working_tensors: Dict[str, Any] = observation_normalizer(
        unnorm_working_tensors)

    for key in _STATE_NORMALIZER_4DIM_KEYS:
        if key in unnorm_working_tensors and isinstance(
                unnorm_working_tensors[key], torch.Tensor):
            # unnorm_working_tensors[key]: (..., 4)
            norm_working_tensors[key] = state_normalizer(
                unnorm_working_tensors[key])

    # 마스크는 원본 그대로 유지(값 크기 변환 결과로 바뀌지 않게)
    norm_working_tensors.update(working_masks)

    # 5) 원본 norm_inputs_copy에 덮어쓰기
    for k, v in norm_working_tensors.items():
        norm_inputs_copy[k] = v

    return norm_inputs_copy


def _update_merged_inputs(
    norm_inputs_copy: Dict[str, Any],
    normed_ego_next_pose: torch.Tensor,  # (B*R, 4)
    normed_near_next_pose: torch.Tensor,  # (B*R, Pnn, 4)
    state_normalizer: StateNormalizer,
    observation_normalizer: ObservationNormalizer,
) -> Dict[str, Any]:
    """rollout 한 스텝 진행을 위해 past를 갱신하고, 새 기준(ego next)으로 좌표를 다시 맞춥니다.

    Args:
        norm_inputs_copy (Dict[str, Any]):
            모델 입력 dict(값 크기 맞춘 상태).
            예:
              - ego_agent_past: (B*R, T_past, 11)
              - near_agents_past: (B*R, Pnn, T_past, 11)
        normed_ego_next_pose (torch.Tensor):
            모델이 예측한 ego 다음 포즈(값 크기 맞춘 상태), shape: (B*R, 4)
        normed_near_next_pose (torch.Tensor):
            모델이 예측한 near 다음 포즈(값 크기 맞춘 상태), shape: (B*R, Pnn, 4)
        state_normalizer (StateNormalizer):
            4차원 포즈(x,y,cos,sin) 값 크기 변환 도구.
        observation_normalizer (ObservationNormalizer):
            입력 dict 전체 값 크기 변환 도구.

    Returns:
        Dict[str, Any]:
            갱신된 norm_inputs_copy
    """
    # ego_agent_past: (B*R, T_past, 11)
    ego_agent_past = norm_inputs_copy["ego_agent_past"]
    ego_next_11_dim = ego_agent_past[:, -1, :].clone()  # (B*R, 11)
    ego_next_11_dim[:, :4] = normed_ego_next_pose  # (B*R, 11)
    norm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, 1:, :], ego_next_11_dim[:, None, :]],
        dim=1)  # (B*R, T_past, 11)

    # near_agents_past: (B*R, Pnn, T_past, 11)
    near_agents_past = norm_inputs_copy["near_agents_past"]
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B*R, Pnn, 11)
    near_next_11_dim[:, :, :4] = normed_near_next_pose  # (B*R, Pnn, 11)
    norm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2)  # (B*R, Pnn, T_past, 11)

    non_near_agents_past = norm_inputs_copy[
        "non_near_agents_past"]  # (B*R, Nnn, T_past, 11)
    assert non_near_agents_past.shape[1] == 0, "현재 non-near agent는 처리하지 않습니다."

    neighbor_agents_past = norm_inputs_copy[
        "neighbor_agents_past"]  # (B*R, agent_num, T_past, 11)
    neighbor_agents_num = neighbor_agents_past.shape[1]
    assert neighbor_agents_num == near_agents_past.shape[1], \
        "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."
    norm_inputs_copy["neighbor_agents_past"] = torch.cat(
        [neighbor_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2)  # (B*R, agent_num, T_past, 11)

    # ✅ 핵심 수정: (원래 값으로 되돌림 -> _transform_origin -> 다시 값 크기 맞춤)
    norm_inputs_copy = _transform_origin_after_restore_then_rescale_inplace(
        norm_inputs_copy=norm_inputs_copy,
        normed_ego_next_pose=normed_ego_next_pose,  # (B*R, 4)
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
    )
    return norm_inputs_copy


from typing import Any, Dict, List

import torch


def _is_rollout_mutable_norm_input_key(key: str) -> bool:
    """rollout 중에 값이 바뀔 가능성이 큰 key인지 판단합니다.

    목적
    ----
    rollout을 만들 때 `norm_inputs`를 복사해야 하는데,
    모든 텐서를 clone() 하면 큰 지도 텐서(lanes/route_lanes/road_edge 등)까지
    매 rollout마다 GPU 메모리 복사가 발생해서 시간이 크게 늘어날 수 있습니다.

    그래서 이 함수는 "rollout 동안 실제로 업데이트될 가능성이 큰 key"만 골라냅니다.

    기준(안전한 쪽으로 잡은 규칙)
    --------------------------
    - agent의 과거 상태(past) 계열:
      - ego_agent_past:            shape (B, T_past, 11)
      - near_agents_past:          shape (B, Pnn, T_past, 11)
      - non_near_agents_past:      shape (B, Nnn, T_past, 11)
      - neighbor_agents_past:      shape (B, (1+)Pnn,  T_past, 11)
    - origin 관련:
      - origin_world_pose 등 "origin"이 들어간 텐서: shape (B, 4) 또는 (B, ...)

    Args:
        key (str): norm_inputs의 key 문자열. shape: ()

    Returns:
        bool:
            - True: rollout 중에 수정될 가능성이 큰 key
            - False: 보통은 고정(공유해도 안전한) key
    """
    key_lower = str(key).lower()

    # 1) 명시적으로 가장 자주 업데이트되는 키들
    if key_lower in (
            "ego_agent_past",
            "near_agents_past",
            "non_near_agents_past",
            "neighbor_agents_past",
    ):
        return True

    # 2) 이름에 origin이 포함되면, rollout에서 기준 좌표가 갱신될 수 있으므로 clone 후보로 봅니다.
    if "origin" in key_lower:
        return True

    # 3) “*_agents_past” 류를 일반화해서 커버(혹시 키 이름이 조금 다른 경우 대비)
    #    예: some_module_near_agents_past 같은 형태도 잡히게
    if key_lower.endswith("_agent_past") or key_lower.endswith("_agents_past"):
        return True

    return False


def _collect_rollout_clone_keys(norm_inputs: Dict[str, Any]) -> List[str]:
    """rollout 복사에서 clone이 필요한 key 목록을 수집합니다.

    Args:
        norm_inputs (Dict[str, Any]):
            모델 입력 dict.
            각 value는 torch.Tensor / list[str] / 숫자 / None 등이 될 수 있습니다.

    Returns:
        List[str]:
            clone 대상 key 목록.
            (리스트 길이는 보통 몇 개 수준으로 유지되는 것을 기대합니다.)
    """
    keys_to_clone: List[str] = []
    for k, v in norm_inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if _is_rollout_mutable_norm_input_key(k):
            keys_to_clone.append(k)
    return keys_to_clone


def _assert_no_shared_tensor_storage_for_keys(
    original: Dict[str, Any],
    copied: Dict[str, Any],
    keys_to_check: List[str],
) -> None:
    """선택된 key들에 대해서만, 원본과 복사본이 같은 저장공간을 공유하는지 검사합니다.

    왜 필요한가?
    ------------
    전체 key를 대상으로 "저장공간 공유 금지"를 강제하면,
    이번 최적화(정적 텐서 공유)가 의도적으로 깨집니다.

    그래서 "rollout 중에 바뀔 수 있는 키"만 골라서
    그 키들만은 반드시 storage가 분리되어 있는지 검사합니다.

    Args:
        original (Dict[str, Any]): 원본 norm_inputs.
        copied (Dict[str, Any]): rollout용 복사본.
        keys_to_check (List[str]): 검사할 key 리스트.

    Raises:
        AssertionError:
            - keys_to_check 중 어떤 key가 원본과 복사본이 같은 storage를 공유하면 에러.
    """
    for key in keys_to_check:
        original_value = original.get(key, None)
        copied_value = copied.get(key, None)

        if not isinstance(original_value, torch.Tensor):
            continue
        if not isinstance(copied_value, torch.Tensor):
            raise AssertionError(
                f"[copy check] key='{key}' 원본은 Tensor인데 복사본이 Tensor가 아닙니다.")

        original_ptr = _get_tensor_storage_ptr(original_value)
        copied_ptr = _get_tensor_storage_ptr(copied_value)
        if int(original_ptr) == int(copied_ptr):
            raise AssertionError(
                f"[copy check] key='{key}' 텐서가 같은 저장공간을 공유합니다. "
                f"(원본/복사 storage ptr 동일)")


def _clone_norm_inputs_for_rollout(
    norm_inputs: Dict[str, Any],
    sanity_check_tensor_storage: bool = False,
) -> Dict[str, Any]:
    """rollout에서 사용할 norm_inputs를 "필요한 것만 clone" 하여 복사합니다.

    기존 방식의 문제
    --------------
    rollout마다 dict 안의 모든 torch.Tensor를 clone() 하면,
    lanes/route_lanes/road_edge처럼 큰 지도 텐서까지 매번 복사됩니다.
    rollout이 32개면 복사 비용이 32배가 되어, GPU 메모리 복사/할당이 병목이 되기 쉽습니다.

    변경된 복사 규칙
    --------------
    - dict 자체는 얕게 복사합니다. (out = dict(norm_inputs))
    - rollout에서 변경될 가능성이 큰 key(ego/near/neighbor past, origin 관련)만 clone() 합니다.
      - 예: ego_agent_past (B, T_past, 11), near_agents_past (B, Pnn, T_past, 11)
    - 그 외 큰 정적 텐서(지도/정적 피처)는 원본을 공유합니다(읽기 전용으로 사용한다는 전제).

    Args:
        norm_inputs (Dict[str, Any]): 원본 입력 dict.
        sanity_check_tensor_storage (bool):
            True이면, 선택된 clone 대상 키들에 한해서
            원본과 복사본이 저장공간을 공유하지 않는지 검사합니다.

    Returns:
        Dict[str, Any]:
            rollout용 입력 dict.
            - clone 대상 텐서는 새 메모리
            - 그 외 텐서는 원본과 같은 객체 참조(공유)
    """
    # 1) dict는 얕게 복사 (키/구조만 새로 만들고 value는 참조 공유)
    rollout_inputs: Dict[str, Any] = dict(norm_inputs)

    # 2) “rollout 중 변할 가능성이 큰 텐서 key”만 골라 clone
    keys_to_clone = _collect_rollout_clone_keys(norm_inputs)
    for k in keys_to_clone:
        rollout_inputs[k] = _clone_nested_value(norm_inputs[k])

    # 3) 디버그 체크(선택 키만)
    if sanity_check_tensor_storage:
        _assert_no_shared_tensor_storage_for_keys(
            original=norm_inputs,
            copied=rollout_inputs,
            keys_to_check=keys_to_clone,
        )

    return rollout_inputs


def _clone_nested_value(value: Any) -> Any:
    """중첩 구조 안의 값을 복사합니다.

    Args:
        value: dict/list/tuple/torch.Tensor 등 어떤 값이든 들어올 수 있습니다.

    Returns:
        copied_value: 복사된 값.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()

    if isinstance(value, dict):
        return {k: _clone_nested_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_clone_nested_value(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_clone_nested_value(v) for v in value)

    # 그 외 타입은 보통 불변이거나, rollout 중에 수정하지 않는다고 가정합니다.
    return value


def _assert_no_shared_tensor_storage(
    original: Dict[str, Any],
    copied: Dict[str, Any],
) -> None:
    """최상위 dict에서 텐서 저장공간 공유 여부를 확인합니다.

    Args:
        original: 원본 dict.
        copied: 복사본 dict.

    Raises:
        AssertionError: 같은 텐서 저장공간을 공유하는 경우.
    """
    for key, original_value in original.items():
        copied_value = copied.get(key, None)
        if not isinstance(original_value, torch.Tensor):
            continue
        if not isinstance(copied_value, torch.Tensor):
            raise AssertionError(
                f"[copy check] key='{key}'는 원본이 Tensor인데 복사본이 Tensor가 아닙니다.")

        original_ptr = _get_tensor_storage_ptr(original_value)
        copied_ptr = _get_tensor_storage_ptr(copied_value)
        if original_ptr == copied_ptr:
            raise AssertionError(
                f"[copy check] key='{key}' 텐서가 같은 저장공간을 공유합니다. "
                f"(원본/복사 storage ptr 동일)")


def _get_tensor_storage_ptr(tensor: torch.Tensor) -> int:
    """텐서의 저장공간 포인터(정수)를 얻습니다.

    Args:
        tensor: torch.Tensor

    Returns:
        storage_ptr: 저장공간 시작 주소(정수).
    """
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().data_ptr())
    return int(tensor.storage().data_ptr())


def _normalize_cos_sin_for_rotation(
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin)을 회전에 쓰기 좋게 길이 1로 맞춥니다.

    - 길이가 충분히 크면: (cos, sin)을 길이 1로 나눠 정리합니다.
    - 길이가 거의 0이면: 회전을 못하므로 (1, 0)으로 둡니다(회전 없음).

    Args:
        cos_values: shape = (...)
        sin_values: shape = (...)
        eps: 0 나눗셈 방지 값.

    Returns:
        cos_normalized: shape = (...)
        sin_normalized: shape = (...)
    """
    raw_norm = torch.sqrt(cos_values**2 + sin_values**2)
    safe_norm = torch.clamp(raw_norm, min=eps)

    cos_normalized = cos_values / safe_norm
    sin_normalized = sin_values / safe_norm

    too_small = raw_norm < eps
    cos_normalized = torch.where(too_small, torch.ones_like(cos_normalized),
                                 cos_normalized)
    sin_normalized = torch.where(too_small, torch.zeros_like(sin_normalized),
                                 sin_normalized)
    return cos_normalized, sin_normalized


def _build_valid_mask_for_pose_4_dim(pose_4_dim: torch.Tensor) -> torch.Tensor:
    """(x, y, cos, sin) 4차원 포즈 텐서에서 "유효한 타임스텝"만 True인 마스크를 만듭니다.

    유효/무효 규칙
    ------------
    - 마지막 차원 4개 값이 모두 0이면 무효(False)
    - 하나라도 0이 아니면 유효(True)

    Args:
        pose_4_dim (torch.Tensor):
            포즈 텐서.
            shape: (..., 4)
            마지막 4는 (x, y, cos, sin)

    Returns:
        torch.Tensor:
            유효 마스크.
            shape: (...,)
            dtype: torch.bool
    """
    if pose_4_dim.dim() < 2 or int(pose_4_dim.shape[-1]) != 4:
        raise ValueError("pose_4_dim은 (..., 4) 이어야 합니다. "
                         f"현재 shape={tuple(pose_4_dim.shape)}")

    # valid_mask: (...,)  True=유효
    valid_mask = torch.any(pose_4_dim != 0.0, dim=-1).to(dtype=torch.bool)
    return valid_mask


def _transform_pose_4_dim_inplace(
    pose_4_dim: torch.Tensor,
    delta_xy: torch.Tensor,
    cos_delta: torch.Tensor,
    sin_delta: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    """(x, y, cos, sin) 포즈 텐서를 "새 원점/방향" 기준으로 바꿉니다.

    이 함수가 하는 일
    --------------
    - pose_4_dim의 위치(x, y)를
      1) delta_xy 만큼 이동(빼기)
      2) cos_delta/sin_delta 만큼 회전
      해서 새 기준 좌표로 바꿉니다.
    - pose_4_dim의 방향(cos, sin)도 같은 기준으로 회전해 줍니다.
    - valid_mask가 False인 칸(무효 타임스텝)은 값을 그대로 둡니다.

    Args:
        pose_4_dim (torch.Tensor):
            포즈 텐서.
            shape: (B, T, 4) 또는 (B, P, T, 4) 등 (..., 4)
            마지막 4는 (x, y, cos, sin)
        delta_xy (torch.Tensor):
            새 원점으로 삼을 ego 위치.
            shape: (B, 2)
        cos_delta (torch.Tensor):
            새 기준 회전에 쓰는 cos 값.
            shape: (B,)
        sin_delta (torch.Tensor):
            새 기준 회전에 쓰는 sin 값.
            shape: (B,)
        valid_mask (torch.Tensor):
            변환을 적용할 칸(True) / 그대로 둘 칸(False).
            shape: pose_4_dim.shape[:-1]
            dtype: torch.bool

    Returns:
        None
    """
    if pose_4_dim.dim() < 2 or int(pose_4_dim.shape[-1]) != 4:
        raise ValueError("pose_4_dim은 (..., 4) 이어야 합니다. "
                         f"현재 shape={tuple(pose_4_dim.shape)}")

    expected_mask_shape = tuple(int(x) for x in pose_4_dim.shape[:-1])
    if tuple(int(x) for x in valid_mask.shape) != expected_mask_shape:
        raise ValueError(
            "valid_mask shape가 pose_4_dim과 맞지 않습니다. "
            f"expected={expected_mask_shape}, got={tuple(valid_mask.shape)}")

    valid_mask_bool = valid_mask.to(dtype=torch.bool)

    # 1) (x, y) 변환
    pos_xy = pose_4_dim[..., 0:2]  # shape: (..., 2)
    pos_xy_new = _transform_points_to_new_origin(
        points_xy=pos_xy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )  # shape: (..., 2)
    pose_4_dim[..., 0:2] = torch.where(
        valid_mask_bool[..., None],
        pos_xy_new,
        pos_xy,
    )

    # 2) (cos, sin) 방향 변환
    cos_h = pose_4_dim[..., 2]  # shape: (...)
    sin_h = pose_4_dim[..., 3]  # shape: (...)
    cos_new, sin_new = _rotate_heading_cos_sin_to_new_origin(
        cos_heading=cos_h,
        sin_heading=sin_h,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )  # shape: (...), (...)

    pose_4_dim[..., 2] = torch.where(valid_mask_bool, cos_new, cos_h)
    pose_4_dim[..., 3] = torch.where(valid_mask_bool, sin_new, sin_h)


from typing import Any, Dict, Optional

import torch


def _initialize_unnorm_inputs_for_rollout(
    norm_inputs_copy: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
) -> Dict[str, Any]:
    """rollout에서 계속 유지할 '원래 값(unnorm)' 입력 dict를 1번만 만들어 둡니다.

    목표
    ----
    기존 코드는 매 step마다:
      1) norm -> unnorm (inverse)
      2) 좌표 기준 변환
      3) unnorm -> norm
    을 반복합니다.

    여기서는 이 중 (1)을 매 step마다 하지 않도록,
    rollout 시작 시점에 unnorm dict를 한 번 만들어서 계속 들고 갑니다.

    Args:
        norm_inputs_copy (Dict[str, Any]):
            정규화된 입력 dict.
            예시(rollout batch로 펼친 뒤):
              - ego_agent_past: (B*R, T_past, 11)
              - near_agents_past: (B*R, Pnn, T_past, 11)
              - lanes: (B*R, lane_num, lane_len, 12)
              - static_objects: (B*R, static_num, 10)
        state_normalizer (StateNormalizer):
            (x, y, cos, sin) 4개 값에 대한 정규화/역정규화 도구.
        observation_normalizer (ObservationNormalizer):
            입력 dict 여러 키에 대한 정규화/역정규화 도구.

    Returns:
        Dict[str, Any]:
            unnorm 입력 dict.
            - 좌표 변환은 이 dict에 대해 수행합니다.
            - 값이 Tensor가 아닌 항목(list 등)은 그대로 들어있을 수 있습니다.
    """
    # 1) 기본은 ObservationNormalizer 기준으로 unnorm 변환
    #    - 키가 normalizer dict에 없으면 그대로 유지됩니다.
    unnorm_inputs_copy: Dict[str, Any] = observation_normalizer.inverse(
        norm_inputs_copy)

    # 2) 일부 4차원 포즈 키는 StateNormalizer가 기준이므로 추가로 덮어씁니다.
    #    - shape 예:
    #      - ego_future_gt_4_dim: (B*R, future_len, 4)
    #      - near_future_gt_4_dim: (B*R, Pnn, future_len, 4)
    for key in _STATE_NORMALIZER_4DIM_KEYS:
        value = norm_inputs_copy.get(key, None)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            unnorm_inputs_copy[key] = state_normalizer.inverse(value)

    return unnorm_inputs_copy


def _build_norm_inputs_from_unnorm_inputs(
    unnorm_inputs_copy: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
) -> Dict[str, Any]:
    """현재 unnorm 입력을 모델 입력용 norm dict로 만듭니다.

    목표
    ----
    - 좌표 변환은 unnorm에서 수행합니다.
    - 모델 입력은 norm 값이 필요하므로, step마다 unnorm -> norm을 1번만 합니다.
      (기존처럼 norm->unnorm->norm 왕복을 하지 않습니다)

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            원래 단위(unnorm) 입력 dict.
            주요 shape 예:
              - ego_agent_past: (B*R, T_past, 11)
              - near_agents_past: (B*R, Pnn, T_past, 11)
              - lanes: (B*R, lane_num, lane_len, 12)
        state_normalizer (StateNormalizer):
            (x, y, cos, sin) 4개 값에 대한 정규화 도구.
        observation_normalizer (ObservationNormalizer):
            입력 dict 여러 키에 대한 정규화 도구.

    Returns:
        Dict[str, Any]:
            정규화된 입력 dict(모델 forward에 넣을 dict).
    """
    norm_inputs_step: Dict[str,
                           Any] = observation_normalizer(unnorm_inputs_copy)

    # StateNormalizer 기준 키는 ObservationNormalizer 결과를 덮어쓰는 방식으로 맞춥니다.
    for key in _STATE_NORMALIZER_4DIM_KEYS:
        value = unnorm_inputs_copy.get(key, None)
        if isinstance(value, torch.Tensor) and value.numel() > 0:
            norm_inputs_step[key] = state_normalizer(value)

    return norm_inputs_step


def _build_cached_valid_masks_for_static_map_features(
    unnorm_inputs_copy: Dict[str, Any],) -> Dict[str, torch.Tensor]:
    """lanes/route_lanes/static_objects의 유효 마스크를 1번만 만들고 재사용합니다.

    왜 필요한가?
    ------------
    _transform_origin 내부에서는 매 step마다 torch.any(...!=0)로
    lanes/route_lanes/static_objects의 유효 칸을 다시 찾습니다.
    이건 큰 텐서를 통째로 훑는 일이어서 step 수가 많으면 누적 비용이 큽니다.

    여기서는 chunk 시작 시점에 한 번만 마스크를 만든 뒤,
    이후 step에서는 그 마스크를 그대로 사용합니다.

    전제(안전한 조건)
    ---------------
    - "없는 칸"은 원래부터 0 패딩이고,
    - 좌표 변환 시에도 그 칸은 변환하지 않아서(마스크로 막아서) 계속 0으로 남습니다.
    - 따라서 "어떤 칸이 데이터가 있는 칸인지" 패턴은 rollout 동안 바뀌지 않습니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            unnorm 입력 dict.
            shape 예:
              - lanes: (B*R, lane_num, lane_len, 12)
              - route_lanes: (B*R, route_lane_num, lane_len, 12)
              - static_objects: (B*R, static_num, 10)

    Returns:
        Dict[str, torch.Tensor]:
            cached_valid_masks dict.
            - "lanes":        (B*R, lane_num, lane_len) bool
            - "route_lanes":  (B*R, route_lane_num, lane_len) bool
            - "static_objects": (B*R, static_num) bool
    """
    cached: Dict[str, torch.Tensor] = {}

    lanes = unnorm_inputs_copy.get("lanes", None)
    if isinstance(lanes, torch.Tensor) and lanes.numel() > 0:
        # lanes: (B*R, lane_num, lane_len, 12)
        cached["lanes"] = torch.any(lanes[..., :8] != 0.0,
                                    dim=-1).to(dtype=torch.bool)

    route_lanes = unnorm_inputs_copy.get("route_lanes", None)
    if isinstance(route_lanes, torch.Tensor) and route_lanes.numel() > 0:
        # route_lanes: (B*R, route_lane_num, lane_len, 12)
        cached["route_lanes"] = torch.any(route_lanes[..., :8] != 0.0,
                                          dim=-1).to(dtype=torch.bool)

    static_objects = unnorm_inputs_copy.get("static_objects", None)
    if isinstance(static_objects, torch.Tensor) and static_objects.numel() > 0:
        # static_objects: (B*R, static_num, 10)
        cached["static_objects"] = torch.any(static_objects[..., :4] != 0.0,
                                             dim=-1).to(dtype=torch.bool)

    return cached


def _pick_cached_mask_if_shape_matches(
    cached_valid_masks: Optional[Dict[str, torch.Tensor]],
    key: str,
    expected_shape: torch.Size,
) -> Optional[torch.Tensor]:
    """cached_valid_masks에서 마스크를 꺼내되, shape가 맞을 때만 사용합니다.

    Args:
        cached_valid_masks (Optional[Dict[str, torch.Tensor]]):
            미리 만든 유효 마스크 dict 또는 None.
        key (str):
            꺼낼 키 이름. 예: "lanes"
        expected_shape (torch.Size):
            현재 텐서의 shape[:-1] 기대 모양.

    Returns:
        Optional[torch.Tensor]:
            - shape가 맞는 마스크가 있으면 bool 텐서 반환
            - 없거나 shape가 다르면 None
    """
    if cached_valid_masks is None:
        return None

    mask = cached_valid_masks.get(key, None)
    if not isinstance(mask, torch.Tensor):
        return None

    if tuple(int(x) for x in mask.shape) != tuple(
            int(x) for x in expected_shape):
        return None

    return mask.to(dtype=torch.bool)


from typing import Any, Dict, Optional
import torch


def _update_merged_inputs_unnorm_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_ego_pose_chunk: torch.Tensor,  # (B*R, gap, 4)
    unnorm_near_pose_chunk: torch.Tensor,  # (B*R, Pnn, gap, 4)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """여러 스텝(gap개)을 한 번에 past에 반영하고, 마지막 스텝을 새 기준으로 좌표를 맞춥니다.

    이 함수가 하는 일
    ---------------
    기존 방식은 매 스텝마다:
      - past 1개 갱신
      - 좌표 기준 변환 1회
    를 반복했습니다.

    여기서는 gap개 예측 포즈가 이미 준비되어 있다고 가정하고,
      1) past에서 앞쪽 gap개를 버리고
      2) 뒤에 gap개를 한 번에 붙이고
      3) 마지막(gap번째) ego 포즈를 기준으로 좌표 기준 변환을 1번만 수행합니다.

    중요 포인트
    ----------
    - past의 11차원 상태 중 (x, y, cos, sin) 4개만 예측값으로 채우고,
      나머지 값(속도 등)은 "기존 past의 마지막 상태"를 복사해서 유지합니다.
    - gap은 보통 T_past 이하로 들어오도록(외부에서) 조절하는 것을 권장합니다.
      그래야 길이를 유지하면서 "앞에서 버리고 뒤에 붙이기"가 자연스럽습니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            현재 시점 기준 입력 dict (원래 단위).
            주요 텐서 shape:
              - ego_agent_past:        (B*R, T_past, 11)
              - near_agents_past:      (B*R, Pnn, T_past, 11)
              - neighbor_agents_past:  (B*R, Pnn, T_past, 11)
              - non_near_agents_past:  (B*R, 0,   T_past, 11)  (현재는 0만 허용)
        unnorm_ego_pose_chunk (torch.Tensor):
            예측된 ego 포즈 시퀀스(원래 단위, 현재 기준 좌표).
            shape: (B*R, gap, 4)  # (x, y, cos, sin)
        unnorm_near_pose_chunk (torch.Tensor):
            예측된 near agent 포즈 시퀀스(원래 단위, 현재 기준 좌표).
            shape: (B*R, Pnn, gap, 4)
        cached_valid_masks (Optional[Dict[str, torch.Tensor]]):
            큰 지도 텐서(lanes/route_lanes/static_objects) 유효 마스크 캐시.

    Returns:
        Dict[str, Any]:
            같은 dict(unnorm_inputs_copy)에 결과를 덮어쓴 뒤 반환합니다.
    """
    if not isinstance(unnorm_ego_pose_chunk, torch.Tensor):
        raise TypeError("unnorm_ego_pose_chunk는 torch.Tensor여야 합니다.")
    if unnorm_ego_pose_chunk.dim() != 3 or int(
            unnorm_ego_pose_chunk.shape[-1]) != 4:
        raise ValueError("unnorm_ego_pose_chunk는 (B*R, gap, 4) 형태여야 합니다. "
                         f"현재 shape={tuple(unnorm_ego_pose_chunk.shape)}")

    if not isinstance(unnorm_near_pose_chunk, torch.Tensor):
        raise TypeError("unnorm_near_pose_chunk는 torch.Tensor여야 합니다.")
    if unnorm_near_pose_chunk.dim() != 4 or int(
            unnorm_near_pose_chunk.shape[-1]) != 4:
        raise ValueError("unnorm_near_pose_chunk는 (B*R, Pnn, gap, 4) 형태여야 합니다. "
                         f"현재 shape={tuple(unnorm_near_pose_chunk.shape)}")

    gap = int(unnorm_ego_pose_chunk.shape[1])
    if gap <= 0:
        return unnorm_inputs_copy

    # -------------------------
    # 1) ego past 업데이트
    # -------------------------
    ego_agent_past = unnorm_inputs_copy.get("ego_agent_past", None)
    if not isinstance(ego_agent_past, torch.Tensor):
        raise KeyError(
            "unnorm_inputs_copy에 'ego_agent_past'(torch.Tensor)가 필요합니다.")
    if ego_agent_past.dim() != 3 or int(ego_agent_past.shape[-1]) != 11:
        raise ValueError("ego_agent_past는 (B*R, T_past, 11) 형태여야 합니다. "
                         f"현재 shape={tuple(ego_agent_past.shape)}")

    past_len = int(ego_agent_past.shape[1])
    if gap > past_len:
        raise ValueError(
            "gap이 past_len보다 큽니다. "
            "외부에서 rollout_time_chunk_size를 past_len 이하로 제한하는 것을 권장합니다. "
            f"gap={gap}, past_len={past_len}")

    # ego_last: (B*R, 11)
    ego_last = ego_agent_past[:, -1, :].clone()
    # ego_chunk_11: (B*R, gap, 11)
    ego_chunk_11 = ego_last[:, None, :].expand(-1, gap, -1).clone()
    ego_chunk_11[:, :, 0:4] = unnorm_ego_pose_chunk  # (x,y,cos,sin)

    # (B*R, T_past, 11)
    unnorm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, gap:, :], ego_chunk_11],
        dim=1,
    )

    # -------------------------
    # 2) near past 업데이트
    # -------------------------
    near_agents_past = unnorm_inputs_copy.get("near_agents_past", None)
    if not isinstance(near_agents_past, torch.Tensor):
        raise KeyError(
            "unnorm_inputs_copy에 'near_agents_past'(torch.Tensor)가 필요합니다.")
    if near_agents_past.dim() != 4 or int(near_agents_past.shape[-1]) != 11:
        raise ValueError("near_agents_past는 (B*R, Pnn, T_past, 11) 형태여야 합니다. "
                         f"현재 shape={tuple(near_agents_past.shape)}")
    if int(near_agents_past.shape[2]) != past_len:
        raise ValueError(
            "near_agents_past의 T_past가 ego_agent_past와 다릅니다. "
            f"ego T_past={past_len}, near T_past={int(near_agents_past.shape[2])}"
        )

    pnn = int(near_agents_past.shape[1])
    if int(unnorm_near_pose_chunk.shape[1]) != pnn:
        raise ValueError(
            "unnorm_near_pose_chunk의 Pnn이 near_agents_past와 다릅니다. "
            f"Pnn(past)={pnn}, Pnn(chunk)={int(unnorm_near_pose_chunk.shape[1])}"
        )
    if int(unnorm_near_pose_chunk.shape[2]) != gap:
        raise ValueError(
            "unnorm_near_pose_chunk의 gap이 unnorm_ego_pose_chunk와 다릅니다. "
            f"gap(ego)={gap}, gap(near)={int(unnorm_near_pose_chunk.shape[2])}")

    # near_last: (B*R, Pnn, 11)
    near_last = near_agents_past[:, :, -1, :].clone()
    # near_chunk_11: (B*R, Pnn, gap, 11)
    near_chunk_11 = near_last[:, :, None, :].expand(-1, -1, gap, -1).clone()
    near_chunk_11[:, :, :, 0:4] = unnorm_near_pose_chunk

    unnorm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, gap:, :], near_chunk_11],
        dim=2,
    )

    # -------------------------
    # 3) non-near는 현재 미지원(기존 규칙 유지)
    # -------------------------
    non_near_agents_past = unnorm_inputs_copy.get("non_near_agents_past", None)
    if isinstance(non_near_agents_past, torch.Tensor):
        assert int(
            non_near_agents_past.shape[1]) == 0, "현재 non-near agent는 처리하지 않습니다."

    # -------------------------
    # 4) neighbor past 업데이트 (near와 동일하게 취급)
    # -------------------------
    neighbor_agents_past = unnorm_inputs_copy.get("neighbor_agents_past", None)
    if not isinstance(neighbor_agents_past, torch.Tensor):
        raise KeyError(
            "unnorm_inputs_copy에 'neighbor_agents_past'(torch.Tensor)가 필요합니다.")
    if neighbor_agents_past.dim() != 4 or int(
            neighbor_agents_past.shape[-1]) != 11:
        raise ValueError(
            "neighbor_agents_past는 (B*R, agent_num, T_past, 11) 형태여야 합니다. "
            f"현재 shape={tuple(neighbor_agents_past.shape)}")
    if int(neighbor_agents_past.shape[2]) != past_len:
        raise ValueError(
            "neighbor_agents_past의 T_past가 ego_agent_past와 다릅니다. "
            f"ego T_past={past_len}, neighbor T_past={int(neighbor_agents_past.shape[2])}"
        )

    neighbor_agents_num = int(neighbor_agents_past.shape[1])
    assert neighbor_agents_num == pnn, "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."

    neighbor_last = neighbor_agents_past[:, :, -1, :].clone()  # (B*R, Pnn, 11)
    neighbor_chunk_11 = neighbor_last[:, :, None, :].expand(-1, -1, gap,
                                                            -1).clone()
    neighbor_chunk_11[:, :, :, 0:4] = unnorm_near_pose_chunk

    unnorm_inputs_copy["neighbor_agents_past"] = torch.cat(
        [neighbor_agents_past[:, :, gap:, :], neighbor_chunk_11],
        dim=2,
    )

    # -------------------------
    # 5) 좌표 기준 변환은 "마지막(gap번째) ego 포즈"로 1번만
    # -------------------------
    unnorm_ego_pose_at_last = unnorm_ego_pose_chunk[:, -1, :]  # (B*R, 4)
    _transform_origin(
        unnorm_inputs_copy,
        unnorm_ego_pose_at_last,
        cached_valid_masks=cached_valid_masks,
    )

    return unnorm_inputs_copy


def _update_merged_inputs_unnorm_inplace(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_ego_next_pose: torch.Tensor,  # (B*R, 4)
    unnorm_near_next_pose: torch.Tensor,  # (B*R, Pnn, 4)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    """unnorm 상태에서 past를 갱신하고, 새 ego 기준으로 좌표를 다시 맞춥니다.

    기존과 같은 의미를 유지하면서 속도를 올리는 핵심 변화
    -------------------------------------------
    - 기존: norm past 갱신 -> (step마다) norm->unnorm(inverse) -> 좌표변환 -> unnorm->norm
    - 변경: unnorm past 갱신 -> 좌표변환
           (모델 입력 norm은 step 시작 시점에 1번만 만들어 사용)

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            unnorm 입력 dict.
            주요 shape 예:
              - ego_agent_past: (B*R, T_past, 11)
              - near_agents_past: (B*R, Pnn, T_past, 11)
              - neighbor_agents_past: (B*R, (1+)Pnn, T_past, 11)
        unnorm_ego_next_pose (torch.Tensor):
            예측된 ego 다음 포즈(원래 단위).
            shape: (B*R, 4)  # (x, y, cos, sin)
        unnorm_near_next_pose (torch.Tensor):
            예측된 near 다음 포즈(원래 단위).
            shape: (B*R, Pnn, 4)
        cached_valid_masks (Optional[Dict[str, torch.Tensor]]):
            lanes/route_lanes/static_objects의 유효 마스크 캐시.
            shape는 _build_cached_valid_masks_for_static_map_features 참고.

    Returns:
        Dict[str, Any]:
            갱신된 unnorm_inputs_copy (같은 dict에 덮어씀).
    """
    # ego_agent_past: (B*R, T_past, 11)
    ego_agent_past = unnorm_inputs_copy["ego_agent_past"]
    ego_next_11_dim = ego_agent_past[:, -1, :].clone()  # (B*R, 11)
    ego_next_11_dim[:, :4] = unnorm_ego_next_pose  # (B*R, 11)
    unnorm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, 1:, :], ego_next_11_dim[:, None, :]],
        dim=1,
    )  # (B*R, T_past, 11)

    # near_agents_past: (B*R, Pnn, T_past, 11)
    near_agents_past = unnorm_inputs_copy["near_agents_past"]
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B*R, Pnn, 11)
    near_next_11_dim[:, :, :4] = unnorm_near_next_pose  # (B*R, Pnn, 11)
    unnorm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2,
    )  # (B*R, Pnn, T_past, 11)

    non_near_agents_past = unnorm_inputs_copy[
        "non_near_agents_past"]  # (B*R, Nnn, T_past, 11)
    assert int(
        non_near_agents_past.shape[1]) == 0, "현재 non-near agent는 처리하지 않습니다."

    neighbor_agents_past = unnorm_inputs_copy[
        "neighbor_agents_past"]  # (B*R, agent_num, T_past, 11)
    neighbor_agents_num = int(neighbor_agents_past.shape[1])
    assert neighbor_agents_num == int(
        near_agents_past.shape[1]
    ), "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."

    unnorm_inputs_copy["neighbor_agents_past"] = torch.cat(
        [neighbor_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2,
    )  # (B*R, agent_num, T_past, 11)

    # ✅ 좌표 기준 변환: unnorm 상태에서 바로 수행 (inverse/normalize 왕복 제거)
    _transform_origin(
        unnorm_inputs_copy,
        unnorm_ego_next_pose,  # (B*R, 4)
        cached_valid_masks=cached_valid_masks,
    )

    return unnorm_inputs_copy


def _transform_origin(
    norm_inputs_copy: Dict[str, torch.Tensor],
    normed_ego_next_pose: torch.Tensor,  # (B*R, 4)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """normed_ego_next_pose 기준으로 입력 전체의 좌표 기준을 바꿉니다.

    핵심 아이디어는 간단합니다.

    - ego의 다음 위치/방향을 "새 기준"으로 삼습니다.
    - 그러면 ego의 다음 위치는 (0, 0)이 되고,
      ego의 다음 방향은 정면(각도 0)이 되도록 입력 전체를 같이 바꿔줘야 합니다.

    변환은 모든 좌표에 대해 아래 2단계로 이뤄집니다.
        1) 이동: 모든 (x, y)에서 ego의 다음 위치 (dx, dy)를 뺍니다.
        2) 회전: ego의 다음 방향만큼 반대로 회전합니다.

    최적화(중요)
    ----------
    lanes/route_lanes/static_objects는 크기가 커서,
    매 step마다 torch.any(...!=0)로 유효 마스크를 다시 계산하면 시간이 많이 듭니다.
    cached_valid_masks가 있으면, 그 마스크를 재사용합니다.

    Args:
        norm_inputs_copy (Dict[str, torch.Tensor]):
            입력 dict.
        normed_ego_next_pose (torch.Tensor):
            (B, 4) ego 다음 프레임 (x, y, cos, sin)
        cached_valid_masks (Optional[Dict[str, torch.Tensor]]):
            lanes/route_lanes/static_objects 유효 마스크 캐시.
            - None이면 기존 방식(매번 계산)을 사용합니다.

    Returns:
        Dict[str, torch.Tensor]:
            좌표 기준이 변환된 입력 dict.
    """
    (delta_xy, cos_delta, sin_delta,
     yaw_delta) = _extract_delta_pose_params(normed_ego_next_pose)

    # 1) ego_agent_past: (B, time_len, 11)
    if "ego_agent_past" in norm_inputs_copy:
        ego_agent_past = norm_inputs_copy["ego_agent_past"]
        if isinstance(ego_agent_past,
                      torch.Tensor) and ego_agent_past.numel() > 0:
            valid_mask = torch.any(ego_agent_past[..., :8] != 0.0,
                                   dim=-1)  # (B, time_len)
            _transform_state_11_dim_inplace(
                state_11=ego_agent_past,
                delta_xy=delta_xy,
                cos_delta=cos_delta,
                sin_delta=sin_delta,
                valid_mask=valid_mask,
            )

    # 3) agent past 11 dim: near/non_near/neighbor
    for key in [
            "near_agents_past", "non_near_agents_past", "neighbor_agents_past"
    ]:
        if key not in norm_inputs_copy:
            continue
        agents_past = norm_inputs_copy[key]
        if not isinstance(agents_past,
                          torch.Tensor) or agents_past.numel() == 0:
            continue

        valid_mask = torch.any(agents_past[..., :8] != 0.0,
                               dim=-1)  # (B, (1+)Pnn, time_len)
        _transform_state_11_dim_inplace(
            state_11=agents_past,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    # 4) future_gt_4_dim: ego + near
    future_gt_keys = ["ego_future_gt_4_dim", "near_future_gt_4_dim"]
    for future_key in future_gt_keys:
        if future_key not in norm_inputs_copy:
            continue

        future_pose_4 = norm_inputs_copy[future_key]
        if (not isinstance(future_pose_4,
                           torch.Tensor)) or future_pose_4.numel() == 0:
            continue

        valid_mask = _build_valid_mask_for_pose_4_dim(future_pose_4)  # (...,)

        _transform_pose_4_dim_inplace(
            pose_4_dim=future_pose_4,
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=valid_mask,  # future_pose_4.shape[:-1]
        )

    # 5) points + explicit valid mask
    point_and_mask_keys = [
        ("stop_sign_points", "stop_sign_is_valid"),
        ("crosswalk_points", "crosswalk_is_valid"),
        ("speed_bump_points", "speed_bump_is_valid"),
        ("driveway_points", "driveway_is_valid"),
        ("road_edge", "road_edge_is_valid"),
    ]
    for point_key, mask_key in point_and_mask_keys:
        if point_key not in norm_inputs_copy or mask_key not in norm_inputs_copy:
            continue
        points_xy = norm_inputs_copy[point_key]  # (B, N, L, 2)
        points_valid = norm_inputs_copy[mask_key]  # (B, N, L) 또는 (B, N)
        if (not isinstance(points_xy, torch.Tensor)) or points_xy.numel() == 0:
            continue
        if not isinstance(points_valid, torch.Tensor):
            continue

        if points_valid.dim() == 2 and points_xy.dim() == 4:
            points_valid = points_valid.unsqueeze(-1).expand_as(points_xy[...,
                                                                          0])

        _transform_points_2d_inplace(
            points_xy=points_xy,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=points_valid.to(dtype=torch.bool),
        )

    # 6) lanes / route_lanes: (B, lane_num, lane_len, 12)
    for lane_key in ["lanes", "route_lanes"]:
        if lane_key not in norm_inputs_copy:
            continue
        lane_12 = norm_inputs_copy[lane_key]
        if (not isinstance(lane_12, torch.Tensor)) or lane_12.numel() == 0:
            continue

        cached_mask = _pick_cached_mask_if_shape_matches(
            cached_valid_masks=cached_valid_masks,
            key=lane_key,
            expected_shape=lane_12.shape[:-1],  # (B, lane_num, lane_len)
        )
        if cached_mask is not None:
            valid_mask = cached_mask
        else:
            valid_mask = torch.any(lane_12[..., :8] != 0.0,
                                   dim=-1).to(dtype=torch.bool)

        _transform_lane_12_dim_inplace(
            lane_12=lane_12,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    # 7) static_objects: (B, static_num, 10)
    if "static_objects" in norm_inputs_copy:
        static_objects = norm_inputs_copy["static_objects"]
        if isinstance(static_objects,
                      torch.Tensor) and static_objects.numel() > 0:
            cached_mask = _pick_cached_mask_if_shape_matches(
                cached_valid_masks=cached_valid_masks,
                key="static_objects",
                expected_shape=static_objects.shape[:-1],  # (B, static_num)
            )
            if cached_mask is not None:
                valid_mask = cached_mask
            else:
                valid_mask = torch.any(static_objects[..., :4] != 0.0,
                                       dim=-1).to(dtype=torch.bool)

            _transform_static_object_10_dim_inplace(
                static_10=static_objects,
                delta_xy=delta_xy,
                cos_delta=cos_delta,
                sin_delta=sin_delta,
                valid_mask=valid_mask,
            )

    return norm_inputs_copy


def _extract_delta_pose_params(
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ego 다음 포즈에서 이동/회전에 필요한 값들을 뽑습니다.

    Args:
        normed_ego_next_pose: (B, 4) = (x, y, cos(각도), sin(각도))

    Returns:
        delta_xy: (B, 2)
        cos_delta: (B,)
        sin_delta: (B,)
        yaw_delta: (B,)
    """
    if normed_ego_next_pose.dim() != 2 or normed_ego_next_pose.shape[-1] != 4:
        raise ValueError("normed_ego_next_pose는 (B, 4) 여야 합니다. "
                         f"현재 shape={tuple(normed_ego_next_pose.shape)}")

    delta_xy = normed_ego_next_pose[:, 0:2]  # (B, 2)
    cos_raw = normed_ego_next_pose[:, 2]  # (B,)
    sin_raw = normed_ego_next_pose[:, 3]  # (B,)

    cos_delta, sin_delta = _normalize_cos_sin_for_rotation(cos_raw, sin_raw)
    yaw_delta = torch.atan2(sin_delta, cos_delta)
    return delta_xy, cos_delta, sin_delta, yaw_delta


def _transform_points_to_new_origin(
        points_xy: torch.Tensor,  # (B, ..., 2)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """(x, y) 점들을 새 원점 기준으로 변환합니다(이동+회전)."""
    if points_xy.shape[-1] != 2:
        raise ValueError(
            f"points_xy의 마지막 차원은 2여야 합니다. shape={tuple(points_xy.shape)}")

    delta_xy_b = delta_xy
    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(points_xy.dim() - 2):
        delta_xy_b = delta_xy_b.unsqueeze(1)
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    dx = points_xy[..., 0] - delta_xy_b[..., 0]
    dy = points_xy[..., 1] - delta_xy_b[..., 1]

    x_new = cos_b * dx + sin_b * dy
    y_new = -sin_b * dx + cos_b * dy
    return torch.stack([x_new, y_new], dim=-1)


def _rotate_vectors_to_new_origin(
        vectors_xy: torch.Tensor,  # (B, ..., 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """(vx, vy) 같은 벡터를 새 기준으로 회전만 적용합니다."""
    if vectors_xy.shape[-1] != 2:
        raise ValueError(
            f"vectors_xy의 마지막 차원은 2여야 합니다. shape={tuple(vectors_xy.shape)}")

    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(vectors_xy.dim() - 2):
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    vx = vectors_xy[..., 0]
    vy = vectors_xy[..., 1]
    vx_new = cos_b * vx + sin_b * vy
    vy_new = -sin_b * vx + cos_b * vy
    return torch.stack([vx_new, vy_new], dim=-1)


def _rotate_heading_cos_sin_to_new_origin(
        cos_heading: torch.Tensor,  # (B, ...)
        sin_heading: torch.Tensor,  # (B, ...)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) 방향을 새 기준으로 바꿉니다(각도 빼기)."""
    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(cos_heading.dim() - 1):
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    cos_new = cos_heading * cos_b + sin_heading * sin_b
    sin_new = sin_heading * cos_b - cos_heading * sin_b
    return cos_new, sin_new


def _wrap_angle_to_pi(angle_rad: torch.Tensor) -> torch.Tensor:
    """각도를 [-pi, pi] 범위로 정리합니다."""
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _transform_state_11_dim_inplace(
        state_11: torch.Tensor,  # (B, ..., 11)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """11차원 상태의 (x,y,cos,sin,vx,vy)를 새 기준으로 변환합니다."""
    if state_11.shape[-1] != 11:
        raise ValueError(
            f"state_11 마지막 차원은 11이어야 합니다. shape={tuple(state_11.shape)}")

    pos_xy = state_11[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    state_11[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    cos_h = state_11[..., 2]
    sin_h = state_11[..., 3]
    cos_new, sin_new = _rotate_heading_cos_sin_to_new_origin(
        cos_h, sin_h, cos_delta, sin_delta)
    state_11[..., 2] = torch.where(valid_mask, cos_new, cos_h)
    state_11[..., 3] = torch.where(valid_mask, sin_new, sin_h)

    vel_xy = state_11[..., 4:6]
    vel_xy_new = _rotate_vectors_to_new_origin(vel_xy, cos_delta, sin_delta)
    state_11[..., 4:6] = torch.where(valid_mask[..., None], vel_xy_new, vel_xy)


def _transform_future_3_dim_inplace(
        future_3: torch.Tensor,  # (B, ..., 3)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        yaw_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """3차원 미래 정답(x,y,각도)을 새 기준으로 변환합니다."""
    if future_3.shape[-1] != 3:
        raise ValueError(
            f"future_3 마지막 차원은 3이어야 합니다. shape={tuple(future_3.shape)}")

    pos_xy = future_3[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    future_3[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    heading = future_3[..., 2]  # (B, ...)
    yaw_b = yaw_delta
    for _ in range(heading.dim() - 1):
        yaw_b = yaw_b.unsqueeze(1)

    heading_new = _wrap_angle_to_pi(heading - yaw_b)
    future_3[..., 2] = torch.where(valid_mask, heading_new, heading)


def _transform_points_2d_inplace(
        points_xy: torch.Tensor,  # (B, ..., 2)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """2D 점(x,y)을 valid_mask가 True인 곳만 새 기준으로 변환합니다."""
    if points_xy.shape[-1] != 2:
        raise ValueError(
            f"points_xy 마지막 차원은 2이어야 합니다. shape={tuple(points_xy.shape)}")

    points_new = _transform_points_to_new_origin(points_xy, delta_xy, cos_delta,
                                                 sin_delta)
    points_xy[...] = torch.where(valid_mask[..., None], points_new, points_xy)


def _transform_lane_12_dim_inplace(
        lane_12: torch.Tensor,  # (B, ..., 12)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """12차원 차선 텐서의 (x,y,dx,dy,dx_left,dy_left,dx_right,dy_right)를 새 기준으로 변환합니다."""
    if lane_12.shape[-1] != 12:
        raise ValueError(
            f"lane_12 마지막 차원은 12이어야 합니다. shape={tuple(lane_12.shape)}")

    pos_xy = lane_12[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    lane_12[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    for start_idx in [2, 4, 6]:
        vec_xy = lane_12[..., start_idx:start_idx + 2]
        vec_new = _rotate_vectors_to_new_origin(vec_xy, cos_delta, sin_delta)
        lane_12[...,
                start_idx:start_idx + 2] = torch.where(valid_mask[..., None],
                                                       vec_new, vec_xy)


def _transform_static_object_10_dim_inplace(
        static_10: torch.Tensor,  # (B, ..., 10)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """10차원 고정 물체의 (x,y,cos,sin)을 새 기준으로 변환합니다."""
    if static_10.shape[-1] != 10:
        raise ValueError(
            f"static_10 마지막 차원은 10이어야 합니다. shape={tuple(static_10.shape)}")

    pos_xy = static_10[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    static_10[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    cos_h = static_10[..., 2]
    sin_h = static_10[..., 3]
    cos_new, sin_new = _rotate_heading_cos_sin_to_new_origin(
        cos_h, sin_h, cos_delta, sin_delta)
    static_10[..., 2] = torch.where(valid_mask, cos_new, cos_h)
    static_10[..., 3] = torch.where(valid_mask, sin_new, sin_h)


def _expand_origin_world_pose_for_agents(
    origin_world_pose: torch.Tensor,  # (B, 4) 또는 (N, 4) 또는 (1, 4)
    n_agent: int,
) -> torch.Tensor:
    """agent 개수에 맞게 origin_world_pose를 늘려서 (N, 4)로 만듭니다.

    이 함수는 "배치 단위로 주어진 world 기준 원점/방향"을,
    "agent 단위(1+Pnn까지 펼친 개수)"로 맞춰주는 역할을 합니다.

    예를 들어,
    - origin_world_pose가 (B, 4)이고,
    - ego_pose가 (N, ..., 4)이며 N = B * (1+Pnn) 라면,
    배치의 origin_world_pose를 각 배치의 agent 수만큼 반복해서 (N, 4)로 확장합니다.

    Args:
        origin_world_pose: (B, 4) 또는 (N, 4) 또는 (1, 4).
            마지막 차원 4는 (x, y, cos, sin) 입니다.
        n_agent: 펼친 agent 개수. (예: B * (1+Pnn))

    Returns:
        origin_world_pose_per_agent: (N, 4)
            각 agent에 대응되는 world 기준 원점/방향.
    """
    if origin_world_pose.dim() != 2 or origin_world_pose.shape[-1] != 4:
        raise ValueError(
            "origin_world_pose는 (B, 4) 또는 (N, 4) 또는 (1, 4) 여야 합니다. "
            f"현재 shape={tuple(origin_world_pose.shape)}")

    origin_count = int(origin_world_pose.shape[0])
    if origin_count == n_agent:
        return origin_world_pose

    if origin_count == 1:
        return origin_world_pose.repeat(n_agent, 1)

    if n_agent % origin_count != 0:
        raise ValueError(
            "n_agent가 origin_world_pose의 첫 번째 차원(B)로 나누어 떨어져야 합니다. "
            f"n_agent={n_agent}, origin_count(B)={origin_count}")

    repeat_factor = n_agent // origin_count
    return origin_world_pose.repeat_interleave(repeat_factor, dim=0)


def _covert_from_ego_to_world(
        target_poses: torch.Tensor,  # (N, 4) 또는 (N, T, 4)
        origin_world_pose: torch.Tensor,  # (B, 4) 또는 (N, 4) 또는 (1, 4)
) -> torch.Tensor:
    """ego 기준 포즈를 world 기준 포즈로 바꿉니다.

    이 함수는 (x, y, cos, sin) 형태의 포즈를 변환합니다.

    - target_poses는 "ego를 원점(0,0)으로 보는 좌표"에서의 값입니다.
    - origin_world_pose는 "ego 원점이 world에서 어디에 있고, 어느 방향을 보고 있는지"를 나타냅니다.

    변환 방법은 다음 순서로 진행합니다.

    1) 위치(x, y) 변환
       - ego 기준 (x, y)를 origin_world_pose의 방향만큼 돌린 뒤,
         origin_world_pose의 (x, y)를 더해서 world 위치로 만듭니다.

    2) 방향(cos, sin) 변환
       - ego 기준 방향과 origin_world_pose의 방향을 합쳐서 world 방향으로 만듭니다.
       - (cos, sin)이 길이 1이 아니게 흔들릴 수 있으니, 변환 전에 길이를 1로 정리합니다.

    3) 무효 프레임 처리
       - 마지막 차원 4개 값이 모두 0인 경우는 "패딩(없는 데이터)"로 보고,
         변환 결과도 0을 유지합니다.
         (이 처리를 안 하면, (0,0,0,0)이 origin 위치/방향으로 바뀌어버리는 문제가 생깁니다.)

    Args:
        target_poses: (N, 4) 또는 (N, T, 4)
            - N: agent 개수(예: B*(1+Pnn))
            - T: 시간 길이(future_len 등)
            - 마지막 4: (x, y, cos, sin)  (ego 기준)
        origin_world_pose: (B, 4) 또는 (N, 4) 또는 (1, 4)
            - B: 배치 크기
            - 마지막 4: (x, y, cos, sin)  (world 기준)

    Returns:
        world_pose: ego_pose와 같은 shape
            - (N, 4) 또는 (N, T, 4)
            - 마지막 4: (x, y, cos, sin)  (world 기준)
    """
    if target_poses.dim() not in (2, 3) or target_poses.shape[-1] != 4:
        raise ValueError("ego_pose는 (N, 4) 또는 (N, T, 4) 여야 합니다. "
                         f"현재 shape={tuple(target_poses.shape)}")

    n_agent = int(target_poses.shape[0])
    origin_world_pose_per_agent = _expand_origin_world_pose_for_agents(
        origin_world_pose=origin_world_pose,
        n_agent=n_agent,
    )  # (N(=B*(1+Pnn), 4)

    # origin pose (world)
    origin_xy = origin_world_pose_per_agent[:, 0:2]  # (N, 2)
    origin_cos_raw = origin_world_pose_per_agent[:, 2]  # (N,)
    origin_sin_raw = origin_world_pose_per_agent[:, 3]  # (N,)
    origin_cos, origin_sin = _normalize_cos_sin_for_rotation(
        origin_cos_raw, origin_sin_raw)  # (N,), (N,)

    # ego pose (ego frame)
    ego_xy = target_poses[..., 0:2]  # (N, 2) 또는 (N, T, 2)
    ego_cos_raw = target_poses[..., 2]  # (N,) 또는 (N, T)
    ego_sin_raw = target_poses[..., 3]  # (N,) 또는 (N, T)
    ego_cos, ego_sin = _normalize_cos_sin_for_rotation(ego_cos_raw, ego_sin_raw)

    # 브로드캐스팅을 위해 (N, 2)/(N,) -> (N, 1, 2)/(N, 1) 로 확장 (T가 있는 경우)
    if target_poses.dim() == 2:
        origin_xy_b = origin_xy  # (N, 2)
        origin_cos_b = origin_cos  # (N,)
        origin_sin_b = origin_sin  # (N,)
    else:
        origin_xy_b = origin_xy[:, None, :]  # (N, 1, 2)
        origin_cos_b = origin_cos[:, None]  # (N, 1)
        origin_sin_b = origin_sin[:, None]  # (N, 1)

    # 1) 위치 변환: world_xy = origin_xy + R(origin_yaw) * ego_xy
    ego_x = ego_xy[..., 0]
    ego_y = ego_xy[..., 1]
    world_x = origin_xy_b[..., 0] + origin_cos_b * ego_x - origin_sin_b * ego_y
    world_y = origin_xy_b[..., 1] + origin_sin_b * ego_x + origin_cos_b * ego_y

    # 2) 방향 변환: yaw_world = yaw_origin + yaw_ego
    # cos(yaw_o + yaw_e) = cos_o*cos_e - sin_o*sin_e
    # sin(yaw_o + yaw_e) = sin_o*cos_e + cos_o*sin_e
    world_cos = origin_cos_b * ego_cos - origin_sin_b * ego_sin
    world_sin = origin_sin_b * ego_cos + origin_cos_b * ego_sin

    world_pose = torch.stack([world_x, world_y, world_cos, world_sin], dim=-1)

    # 3) 무효 프레임(전부 0)인 경우는 그대로 0을 유지
    valid_mask = torch.any(target_poses != 0.0, dim=-1)  # (N,) 또는 (N, T)
    world_pose = torch.where(valid_mask[..., None], world_pose,
                             torch.zeros_like(world_pose))
    return world_pose


def main() -> None:
    """train_predictor 진입점.

    처리 순서:
      1) 명령행 인자를 읽고(args_util.get_args), 필요 시 W&B 아티팩트에서
         체크포인트 파일(latest.pth 등)을 내려받아 args.save_path 를 채운다.
      2) 환경변수 WORLD_SIZE 를 기준으로 args.distributed 를 설정해,
         이후 ddp 설정이 올바르게 동작하도록 만든다.
      3) model_validation(args) 를 호출해 전체 학습 파이프라인을 수행하고,
         예외가 발생하면 rank 정보를 찍고 전체 스택을 출력한다.
    """
    # 1) 분산 초기화 및 rank 정보
    args = args_util.get_args()
    global_rank, rank, world_size, use_deepspeed = init_distributed(args)

    set_save_path(
        args=args,
        global_rank=global_rank,
    )
    should_finish = prepare_wandb_resume(args)
    set_distributed_flag_from_env(args)

    # ✅ (추가) 프로그램 시작 시, 이번 run_count 카운터 파일이 있으면 삭제해서 초기화
    _purge_visualization_budget_counter_file_at_program_start(
        args=args,
        global_rank=int(global_rank),
        world_size=int(world_size),
    )

    # Run
    try:
        model_validation(args, global_rank, rank, world_size, use_deepspeed)
    except BaseException:
        rank = int(os.environ.get("RANK", -1))
        print(
            f"\n[rank{rank}] Unhandled exception (printing full traceback):",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
