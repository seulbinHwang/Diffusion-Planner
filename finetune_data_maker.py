import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from torch.utils.data import DataLoader

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
import draw_machine_fast
import args_util
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

import torch

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype

from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger

import argparse
import json
from typing import Optional

_VALIDATION_HEARTBEAT: Optional["_ValidationHeartbeat"] = None

from typing import Any, Dict, Optional, Tuple, Set, List, Callable
import numpy as np
import os
import contextlib

from typing import Any, Dict, List

import torch
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

_INFERENCE_NPZ_EXCLUDED_EXACT_KEYS: Set[str] = {
    "diff_token_to_future_gt_3_dim",
    "non_near_agents_past",
    "near_agents_past",
    "ego_future_gt_11_dim",
    "near_future_gt_4_dim",
    "ego_future_gt_4_dim",
    "target_future_valid",
    "scenario_id",
}

try:
    import fcntl  # 리눅스/유닉스에서 파일 잠금에 사용
except ImportError:
    fcntl = None  # type: ignore


def _should_exclude_key_for_inference_npz_save(key: str) -> bool:
    """추론 npz 저장에서 제외할 key인지 판단합니다.

    제외 규칙
    --------
    1) key 이름이 "is_valid"로 끝나면 저장하지 않습니다.
       - 예: road_edge_is_valid, stop_sign_is_valid 등
    2) 특정 key는 이름이 정확히 일치하면 저장하지 않습니다.
       - diff_token_to_future_gt_3_dim
       - non_near_agents_past
       - near_agents_past

    Args:
        key (str): dict의 key 문자열. shape: ()

    Returns:
        bool:
            - True: 저장에서 제외
            - False: 저장에 포함
            shape: ()
    """
    k = str(key)
    if k.endswith("is_valid"):
        return True
    return k in _INFERENCE_NPZ_EXCLUDED_EXACT_KEYS


def _build_inference_npz_payload_for_save(
    sample_dict: Dict[str, Any],) -> Dict[str, Any]:
    """npz로 저장할 dict를 '규칙에 맞게' 골라서 만듭니다.

    규칙
    ----
    1) key가 "is_valid"로 끝나면 저장하지 않습니다.
    2) 특정 key는 저장하지 않습니다.
       - diff_token_to_future_gt_3_dim
       - non_near_agents_past
       - near_agents_past
    3) planner_future_11_dim 과 ego_future_gt_11_dim 이 완전히 같아야 하며,
       저장은 ego_future_gt_11_dim 만 남기고 planner_future_11_dim 은 제외합니다.

    Args:
        sample_dict (Dict[str, Any]): 샘플 1개 dict. shape: ()

    Returns:
        Dict[str, Any]:
            np.savez_compressed에 넣을 dict. shape: ()
    """

    out: Dict[str, Any] = {}
    for k, v in sample_dict.items():
        key = str(k)
        if _should_exclude_key_for_inference_npz_save(key):
            continue
        if not isinstance(v, np.ndarray):
            continue
        if key == "planner_future_11_dim":
            # (3) planner는 저장하지 않음(ego와 동일하다는 assert는 위에서 수행)
            key = "ego_future_gt_11_dim"

        out[key] = v

    return out


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

        _update_validation_heartbeat_stage(args, "running validation loop")
        """전체 epoch 루프를 돌면서 학습, 속도 측정, 로깅, 체크포인트 저장을 수행한다."""
        # 전체 업데이트 스텝 수 설정 및 global step 초기화 보장
        batch_num_in_one_val_epoch: int = max(1, len(validation_loader))
        # ✅ (중요) epoch 시작 전에 sampler epoch를 먼저 세팅
        # - resume(init_epoch>0) 시에도 첫 epoch부터 올바른 shuffle이 나오도록 함
        # Dict[str, torch.Tensor]
        epoch_elapsed_time_sec = validate_one_epoch(
            epoch=0,
            total_epochs=1,
            validation_loader=validation_loader,
            diffusion_planner=diffusion_planner,
            args=args,
            model_ema=model_ema,
            batch_num_in_one_val_epoch=batch_num_in_one_val_epoch,
        )
        print("epoch_elapsed_time_sec:", epoch_elapsed_time_sec)

        _update_validation_heartbeat_stage(args, "finalizing")
        _finalize_eval_cleanup(global_rank)

    finally:
        _stop_validation_heartbeat_if_needed(args)


def _finalize_eval_cleanup(global_rank: int,) -> None:
    # 1) 분산 실행이면 여기서 한 번 모여서, 모두가 validation을 끝낸 뒤 정리로 넘어가게 합니다.
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    # 3) rank0가 writer를 닫을 때까지 다른 rank가 너무 빨리 빠져나가지 않게 한 번 더 맞춥니다.
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    batch_num_in_one_val_epoch: int,
) -> float:
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{total_epochs}")

    epoch_t0 = time.perf_counter()
    validation_epoch(
        validation_loader,
        diffusion_planner,
        args,
        model_ema,
        batch_num_in_one_val_epoch,
    )

    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ✅ world_size>1일 때만 동기화
    if _should_cuda_synchronize_for_validation(args):
        torch.cuda.synchronize()

    _maybe_distributed_barrier(args, context="validate_one_epoch end")

    epoch_elapsed_time_sec = time.perf_counter() - epoch_t0
    return epoch_elapsed_time_sec


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
) -> None:
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
              - integrated_trajectory: (B, (1+)Pnn, 1+future_len, 4)
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


def _shrink_batch_to_draw_idx(
    unnorm_inputs_copy: Dict[str, Any],
    draw_batch_idx: int,
) -> Dict[str, Any]:
    a_unnorm_inputs_copy = {}
    for k, v in unnorm_inputs_copy.items():
        if isinstance(v, torch.Tensor):
            a_unnorm_inputs_copy[k] = v[draw_batch_idx]
        elif isinstance(v, list):
            a_unnorm_inputs_copy[k] = [v[draw_batch_idx]]
    #
    return a_unnorm_inputs_copy


def _torch_to_numpy(inputs: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.cpu().numpy()
        else:
            out[k] = v
    return out


from dataclasses import dataclass
from typing import Any, Dict, Optional
import torch


@dataclass
class _RolloutVisualizationState:
    """rollout 과정에서 그림/영상 저장 상태를 한 곳에 모아 관리합니다.

    Attributes:
        requested_save_image (bool): 사용자가 이미지 저장을 원했는지. shape: ()
        requested_save_video (bool): 사용자가 영상 저장을 원했는지. shape: ()
        enabled_save_image (bool): 예산 체크까지 통과해서 실제 이미지 저장을 할지. shape: ()
        enabled_save_video (bool): 예산 체크까지 통과해서 실제 영상 저장을 할지. shape: ()
        budget_checked (bool): 예산 체크를 이미 했는지. shape: ()
        draw_batch_idx (int):
            현재 배치(B)에서 어떤 샘플을 그릴지 인덱스.
            - rollout을 1개씩 순차 실행하면 rollout 차원은 없으므로,
              그냥 "배치 인덱스"로만 사용합니다.
            shape: ()
        draw_scenario_id (str): 저장 폴더/영상 이름에 쓰는 시나리오 id. shape: ()
        save_dir (str): PNG가 저장될 폴더 경로. shape: ()
        draw_near_target_id (Optional[torch.Tensor]):
            near 대상 id 목록. 보통 길이 Pnn.
            shape: (Pnn,) 또는 None
    """
    requested_save_image: bool
    requested_save_video: bool
    enabled_save_image: bool
    enabled_save_video: bool
    budget_checked: bool
    draw_batch_idx: int
    draw_scenario_id: str
    save_dir: str
    draw_near_target_id: Optional[torch.Tensor]  # shape: (Pnn,)


def _init_rollout_visualization_state(
    *,
    save_image: bool,
    save_video: bool,
    draw_batch_idx: int,
) -> _RolloutVisualizationState:
    """rollout 시각화(그림/영상) 상태를 초기값으로 만듭니다.

    Args:
        save_image (bool): 사용자 요청(이미지 저장). shape: ()
        save_video (bool): 사용자 요청(영상 저장). shape: ()
        draw_batch_idx (int): 배치(B)에서 그릴 샘플 인덱스. shape: ()

    Returns:
        _RolloutVisualizationState: 초기 상태 객체. shape: ()
    """
    return _RolloutVisualizationState(
        requested_save_image=bool(save_image),
        requested_save_video=bool(save_video),
        enabled_save_image=False,
        enabled_save_video=False,
        budget_checked=False,
        draw_batch_idx=int(draw_batch_idx),
        draw_scenario_id="",
        save_dir="",
        draw_near_target_id=None,
    )


def _maybe_prepare_rollout_visualization_once(
    *,
    args: Any,
    norm_inputs: Dict[str, Any],
    state: _RolloutVisualizationState,
) -> _RolloutVisualizationState:
    """예산 체크 + 저장 폴더/ID 준비를 '딱 1번'만 수행합니다.

    동작:
        1) 사용자가 이미지/영상 저장을 요청하지 않았으면 아무것도 하지 않습니다.
        2) 요청이 있어도 예산(횟수 제한) 체크에 실패하면 저장을 비활성화합니다.
        3) 성공하면:
           - 영상 요청이면 프레임 PNG도 필요하므로 이미지 저장도 켭니다.
           - scenario_id, near_target_id 목록, 저장 폴더(save_dir)를 준비합니다.

    Args:
        args (Any): 설정 객체. shape: ()
        norm_inputs (Dict[str, Any]):
            원본 모델 입력(dict). 여기서 scenario_id/target_id를 읽습니다. shape: ()
        batch_size (int): 원본 배치 크기 B. shape: ()
        state (_RolloutVisualizationState): 현재 상태. shape: ()

    Returns:
        _RolloutVisualizationState: 갱신된 상태. shape: ()
    """
    if bool(state.budget_checked):
        return state

    state.budget_checked = True

    if (not bool(state.requested_save_image)) and (not bool(
            state.requested_save_video)):
        return state

    # 영상이면 PNG 프레임이 필요하므로 이미지도 켭니다.
    state.enabled_save_video = bool(state.requested_save_video)
    state.enabled_save_image = bool(state.requested_save_image or
                                    state.requested_save_video)

    if not bool(state.enabled_save_image):
        return state

    draw_scenario_id, draw_near_target_id, save_dir = _prepare_data_for_draw(
        args=args,
        norm_inputs=norm_inputs,
        draw_batch_idx=int(state.draw_batch_idx),
    )
    state.draw_scenario_id = str(draw_scenario_id)
    state.save_dir = str(save_dir)
    state.draw_near_target_id = draw_near_target_id  # shape: (Pnn,)
    return state


def _maybe_draw_rollout_visualization_frame(
    *,
    state: _RolloutVisualizationState,
    unnorm_inputs_copy: Dict[str, Any],
    normed_selected_traj: torch.Tensor,
    state_normalizer: Any,
    step_idx: int,
) -> None:
    """조건이 맞으면 PNG 프레임 1장을 저장합니다.

    Args:
        state (_RolloutVisualizationState): 시각화 상태. shape: ()
        unnorm_inputs_copy (Dict[str, Any]): 현재 시점 모델 입력(정규화). shape: ()
        normed_selected_traj (torch.Tensor):
            선택된 경로(정규화).
            shape: (B*R, 1+Pnn, 1+T, 4)
        state_normalizer (Any): (x,y,cos,sin) 변환 도구. shape: ()
        step_idx (int): 저장 파일 이름에 넣을 step 값. shape: ()

    Returns:
        None
    """
    if not bool(state.enabled_save_image):
        return

    if state.draw_near_target_id is None:
        raise RuntimeError("이미지 저장이 켜져 있는데 draw_near_target_id가 준비되지 않았습니다.")
    if str(state.save_dir).strip() == "":
        raise RuntimeError("이미지 저장이 켜져 있는데 save_dir가 비어 있습니다.")

    # _prepare_data_for_one_batch_draw 내부에서:
    # - (B)에서 state.draw_batch_idx 샘플 1개만 뽑아 numpy로 만듭니다.
    (a_unnorm_inputs_np, a_unnorm_trajectory_np, a_unnorm_near_future_gt_3_dim,
     a_unnorm_ego_future_gt_4_dim
    ) = _prepare_data_for_one_batch_draw(
        unnorm_inputs_copy=unnorm_inputs_copy,
        normed_trajectories=normed_selected_traj,  # shape: (B*R, 1+Pnn, 1+T, 4)
        state_normalizer=state_normalizer,
        draw_batch_idx=int(state.draw_batch_idx),
    )

    _draw_one_batch_one_rollout(
        save_dir=str(state.save_dir),
        a_unnorm_inputs_np=a_unnorm_inputs_np,
        a_unnorm_trajectory_np=a_unnorm_trajectory_np,  # shape: ((1+)Pnn, 1+T, 4)
        a_unnorm_ego_future_gt_4_dim=a_unnorm_ego_future_gt_4_dim,  # shape: (future_len, 4)
        a_unnorm_near_future_gt_3_dim=
        a_unnorm_near_future_gt_3_dim,  # shape: (Pnn, future_len, 3)
        step_idx=int(step_idx),
        draw_near_target_id=state.draw_near_target_id,  # shape: (Pnn,)
    )


def _finalize_rollout_visualization_video_if_needed(
    *,
    args: Any,
    state: _RolloutVisualizationState,
) -> None:
    """PNG로부터 영상을 만들지 결정하고, 필요하면 생성합니다.

    Args:
        args (Any): 설정 객체. shape: ()
        state (_RolloutVisualizationState): 시각화 상태. shape: ()

    Returns:
        None
    """
    if not bool(state.enabled_save_video):
        return

    if str(state.save_dir).strip() == "" or str(
            state.draw_scenario_id).strip() == "":
        raise RuntimeError(
            "영상 저장이 켜져 있는데 save_dir / draw_scenario_id가 준비되지 않았습니다.")

    draw_machine_fast.make_video_from_all_png(
        str(state.save_dir),
        str(state.draw_scenario_id),
        new_save_dir=state.save_dir,
        run_count=args.run_count,
    )


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
    target_id = norm_inputs.get("target_id", None)
    if target_id is not None:
        draw_target_id = target_id[draw_batch_idx]  # ((1+)Pnn)
    else:
        # manually set draw_target_id as 0, 1, ..., Pnn
        one_Pnn = len(norm_inputs["near_agents_past"][draw_batch_idx])
        draw_target_id = torch.arange(0, one_Pnn + 1)  # ((1+)Pnn)

    draw_near_target_id = draw_target_id[1:]  # (Pnn,)

    save_dir = os.path.join(args.save_cache_path,
                            f"debug_vis_{draw_scenario_id}")
    os.makedirs(save_dir, exist_ok=True)

    return draw_scenario_id, draw_near_target_id, save_dir


def _build_inference_npz_file_name(
    scenario_id: str,
    step_count: int,
    sample_idx: int,
) -> str:
    """추론/rollout 중간 상태 npz 파일 이름을 만듭니다.

    왜 sample_idx를 넣나?
    -------------------
    같은 scenario_id에 대해 rollout을 R번 만들면,
    (B*R) 배치 안에서 scenario_id가 반복됩니다.
    step_count만 파일명에 넣으면 같은 이름이 여러 번 만들어져서 덮어쓰게 됩니다.

    그래서 파일명에 (B*R) 배치 인덱스(sample_idx)를 함께 넣어서
    "배치/rollout별로 파일이 각각 남도록" 합니다.

    Args:
        scenario_id (str): 시나리오 id 문자열. shape: ()
        step_count (int): rollout 루프 저장 step 번호. shape: ()
        sample_idx (int): (B*R) 배치에서의 샘플 인덱스. shape: ()

    Returns:
        str: 예) "{scenario_id}_b0007_step0003.npz" 형태. shape: ()
    """
    safe_scenario_id = str(scenario_id).replace("/",
                                                "_").replace("\\", "_").replace(
                                                    os.sep, "_")
    return f"{safe_scenario_id}_b{int(sample_idx):04d}_step{int(step_count):04d}.npz"


def _tensor_to_cpu_numpy_for_npz_save(tensor: torch.Tensor) -> np.ndarray:
    """저장(npz)용으로 torch.Tensor를 CPU numpy로 안전하게 바꿉니다.

    이 함수가 하는 일
    ---------------
    - GPU 텐서를 CPU로 옮기고, numpy 배열로 바꿉니다.
    - bfloat16은 numpy가 직접 처리하기 어려운 경우가 있어서,
      저장 단계에서는 float32로 바꿔 안전하게 저장할 수 있게 합니다.

    Args:
        tensor (torch.Tensor):
            저장할 텐서.
            - 보통 shape: (B, ...) 형태를 기대합니다.
            - B는 배치 크기(샘플 개수)입니다.

    Returns:
        np.ndarray:
            CPU numpy 배열.
            - shape: tensor와 동일 (예: (B, ...))
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor는 torch.Tensor여야 합니다. type={type(tensor)}")

    # numpy는 bfloat16을 바로 못 받는 경우가 많아서 저장 단계에서는 float32로 변환합니다.
    if tensor.dtype == torch.bfloat16:
        cpu_tensor = tensor.detach().to(device="cpu", dtype=torch.float32)
    else:
        cpu_tensor = tensor.detach().cpu()

    return cpu_tensor.numpy()


def _pose_4_dim_numpy_to_pose_3_dim_numpy(pose_4_dim: np.ndarray) -> np.ndarray:
    """(x, y, cos, sin) 4차원 포즈를 (x, y, yaw) 3차원 포즈로 바꿉니다.

    Args:
        pose_4_dim (np.ndarray):
            마지막 차원이 4인 배열.
            - shape 예:
              - ego:  (future_len, 4)
              - near: (Pnn, future_len, 4)

    Returns:
        np.ndarray:
            마지막 차원이 3인 배열.
            - shape 예:
              - ego:  (future_len, 3)
              - near: (Pnn, future_len, 3)

    Raises:
        ValueError:
            마지막 차원이 4가 아니면 에러.
    """
    if pose_4_dim.ndim < 1 or int(pose_4_dim.shape[-1]) != 4:
        raise ValueError("pose_4_dim의 마지막 차원은 4여야 합니다. "
                         f"현재 shape={tuple(pose_4_dim.shape)}")
    """ valid_mask
    마지막 4 차원이 전부 0. 이면 무효점입니다.
    
    valid_mask 
        - shape 예
            - ego:  (future_len,)
            - near: (Pnn, future_len)
    """
    valid_mask = np.any(pose_4_dim[..., 0:4] != 0, axis=-1)  #

    # pose_4_dim[..., 0:2]: (.., 2)
    xy = pose_4_dim[..., 0:2]
    # yaw: (..,)
    yaw = np.arctan2(pose_4_dim[..., 3], pose_4_dim[..., 2])
    # (.., 3)
    dim_3 = np.concatenate([xy, yaw[..., None]], axis=-1)
    # 무효점은 전부 0으로 만듭니다.
    dim_3 = dim_3 * valid_mask[..., None]
    return dim_3


def _gpu_tensor_to_cpu_np(
    unnorm_inputs_copy: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    """unnorm_inputs_copy를 "저장하기 좋은 CPU 자료"로 한 번만 변환해 캐시로 만듭니다.

    핵심 아이디어
    ------------
    - 여기서는 키별로 1번만 CPU numpy로 바꿔두고,
      이후에는 루프에서 인덱싱만 합니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            저장할 입력 dict.
            - torch.Tensor는 보통 shape: (B, ...) 형태를 기대합니다.
            - list는 보통 길이 B를 기대합니다.
        batch_size (int):
            배치 크기 B. shape: ()

    Returns:
        Dict[str, Any]:
            저장용 캐시 dict.
            - torch.Tensor -> np.ndarray (CPU)
            - list -> 그대로(list)
            - None -> 그대로(None)

    Raises:
        ValueError:
            텐서의 첫 차원 또는 리스트 길이가 batch_size와 맞지 않으면 에러.
        AssertionError:
            지원하지 않는 타입이 들어오면 에러.
    """
    cpu_cache: Dict[str, Any] = {}

    for key, value in unnorm_inputs_copy.items():
        if isinstance(value, torch.Tensor):
            if value.dim() == 0:
                # 스칼라 텐서는 그대로 1개 값으로 저장(샘플별 인덱싱 불가)
                cpu_cache[key] = _tensor_to_cpu_numpy_for_npz_save(
                    value)  # shape: ()
                continue

            if int(value.shape[0]) != int(batch_size):
                raise ValueError(
                    "저장 대상 텐서는 첫 차원이 batch_size여야 합니다. "
                    f"key='{key}', tensor_shape={tuple(value.shape)}, batch_size={batch_size}"
                )

            # (B, ...) -> numpy (B, ...)
            cpu_cache[key] = _tensor_to_cpu_numpy_for_npz_save(value)

        elif isinstance(value, list):
            if int(len(value)) != int(batch_size):
                raise ValueError(
                    "저장 대상 list는 길이가 batch_size여야 합니다. "
                    f"key='{key}', len={len(value)}, batch_size={batch_size}")
            cpu_cache[key] = value

        else:
            assert value is None, f"Unsupported type {type(value)} for key '{key}'"
            cpu_cache[key] = value

    return cpu_cache


def _slice_one_sample_from_cpu_batch_cache(
    unnorm_inputs_np: Dict[str, Any],
    sample_idx: int,
    batch_size: int,
) -> Dict[str, Any]:
    """CPU 캐시에서 sample_idx에 해당하는 샘플 1개만 뽑아 저장 dict를 만듭니다.

    Args:
        unnorm_inputs_np (Dict[str, Any]):
            _gpu_tensor_to_cpu_np()가 만든 캐시.
            - numpy 배열이면 보통 shape: (B, ...)
            - list면 길이 B
        sample_idx (int):
            뽑을 샘플 인덱스. shape: ()
        batch_size (int):
            배치 크기 B. shape: ()

    Returns:
        Dict[str, Any]:
            np.savez_compressed에 바로 넣을 수 있는 1개 샘플 dict.
            - torch.Tensor는 이미 numpy로 바뀐 상태이며,
              여기서는 인덱싱만 합니다.
    """
    out: Dict[str, Any] = {}

    for key, value in unnorm_inputs_np.items():
        if isinstance(value, np.ndarray):
            # 스칼라(shape=())면 그대로 저장, (B, ...)면 [sample_idx]로 한 샘플만
            if value.ndim >= 1 and int(value.shape[0]) == int(batch_size):
                out[key] = value[int(sample_idx)]
            else:
                out[key] = value
        elif isinstance(value, list):
            out[key] = value[int(sample_idx)]
        else:
            out[key] = value

    return out


def _remove_invalid_data(npz_payload_dict: Dict[str, Any]) -> Dict[str, Any]:
    dim_11_12_keys = {
        "neighbor_agents_past",  # (chosen_agent_num, time_len, 11)
        "lanes",  # (chosen_lane_num, lane_len, 12)
        "route_lanes",  # (chosen_route_lane_num, route_len, 12)
    }
    neighbor_agents_past = npz_payload_dict["neighbor_agents_past"]
    if neighbor_agents_past is not None:
        neighbor_agents_current = neighbor_agents_past[:,
                                                       -1, :]  # (chosen_agent_num, 11)
        neighbor_agents_past_is_invalid = np.all(
            neighbor_agents_current[...,
                                    0:8] == 0, axis=-1)  # (chosen_agent_num)
        neighbor_agents_past_is_valid = np.logical_not(
            neighbor_agents_past_is_invalid)  # (chosen_agent_num)
        npz_payload_dict["neighbor_agents_past"] = neighbor_agents_past[
            neighbor_agents_past_is_valid]  # (valid_chosen_agent_num, time_len, 11)
        npz_payload_dict["neighbor_future_gt_3_dim"] = npz_payload_dict[
            "neighbor_future_gt_3_dim"][
                neighbor_agents_past_is_valid]  # (valid_chosen_agent_num, future_len, 3)

    lanes = npz_payload_dict["lanes"]  # (chosen_lane_num, lane_len, 12)
    if lanes is not None:
        lanes_point_is_invalid = np.all(lanes[..., 0:8] == 0,
                                        axis=-1)  # (chosen_lane_num, lane_len)
        # lanes_point_is_invalid_sum: (chosen_lane_num)
        lanes_point_is_invalid_sum = np.sum(lanes_point_is_invalid, axis=-1)
        # lanes_point_is_invalid_sum 이 0 인 경우 -> 유효한 lane
        lanes_is_valid = lanes_point_is_invalid_sum == 0  # (chosen_lane_num)
        npz_payload_dict["lanes"] = lanes[
            lanes_is_valid]  # (valid_chosen_lane_num, lane_len, 12)
        npz_payload_dict["lanes_speed_limit"] = npz_payload_dict[
            "lanes_speed_limit"][lanes_is_valid]  # (valid_chosen_lane_num, 1)
        npz_payload_dict["lanes_has_speed_limit"] = npz_payload_dict[
            "lanes_has_speed_limit"][
                lanes_is_valid]  # (valid_chosen_lane_num, 1)
        # lane_type: (chosen_lane_num, 4) / "left_line_type" : (chosen_lane_num, 13) / "right_line_type" : (chosen_lane_num, 13)
        lane_type = npz_payload_dict.get("lane_type", None)
        if lane_type is not None:
            npz_payload_dict["lane_type"] = lane_type[
                lanes_is_valid]  # (valid_chosen_lane_num, 4)
            npz_payload_dict["left_line_type"] = npz_payload_dict[
                "left_line_type"][lanes_is_valid]  # (valid_chosen_lane_num, 13)
            npz_payload_dict["right_line_type"] = npz_payload_dict[
                "right_line_type"][
                    lanes_is_valid]  # (valid_chosen_lane_num, 13)

    agent_route_lane_order = npz_payload_dict.get("agent_route_lane_order",
                                                  None)
    if agent_route_lane_order is not None:
        # "agent_route_lane_order",#check#check  # (chosen_agent_num, chosen_lane_num) -> (valid_chosen_agent_num, valid_chosen_lane_num)
        npz_payload_dict["agent_route_lane_order"] = agent_route_lane_order[
            neighbor_agents_past_is_valid][:,
                                           lanes_is_valid]  # (valid_chosen_agent_num, valid_chosen_lane_num)

    route_lanes = npz_payload_dict.get(
        "route_lanes", None)  # (chosen_route_lane_num, route_len, 12)
    if route_lanes is not None:
        route_lanes_point_is_invalid = np.all(
            route_lanes[..., 0:8] == 0,
            axis=-1)  # (chosen_route_lane_num, route_len)
        # route_lanes_point_is_invalid_sum: (chosen_route_lane_num)
        route_lanes_point_is_invalid_sum = np.sum(route_lanes_point_is_invalid,
                                                  axis=-1)
        # route_lanes_point_is_invalid_sum 이 0 인 경우 -> 유효한 route_lane
        route_lanes_is_valid = route_lanes_point_is_invalid_sum == 0  # (chosen_route_lane_num)
        npz_payload_dict["route_lanes"] = route_lanes[
            route_lanes_is_valid]  # (valid_chosen_route_lane_num, route_len, 12)
        npz_payload_dict["route_lanes_speed_limit"] = npz_payload_dict[
            "route_lanes_speed_limit"][
                route_lanes_is_valid]  # (valid_chosen_route_lane_num, 1)
        npz_payload_dict["route_lanes_has_speed_limit"] = npz_payload_dict[
            "route_lanes_has_speed_limit"][
                route_lanes_is_valid]  # (valid_chosen_route_lane_num, 1)
    """
    # stop_sign_points : (stop_sign_num, safety_len, 2) / crosswalk_points: (crosswalk_num, safety_len, 2)
    speed_bump_points : (speed_bump_num, safety_len, 2) / driveway_points : (driveway_num, safety_len, 2)
    road_edge : (chosen_edge_num, safety_len, 2) 
    이 5개는 2차원 (x,y) 값이 전부 0. 이면 무효다. 그리고 전 safety_len 점이 모두 유효해야 유효한 객체다.
    """
    stop_sign_points = npz_payload_dict.get(
        "stop_sign_points", None)  # (stop_sign_num, safety_len, 2)
    if stop_sign_points is not None:
        stop_sign_points_is_invalid = np.all(
            stop_sign_points[...,
                             0:2] == 0, axis=-1)  # (stop_sign_num, safety_len)
        stop_sign_points_is_invalid_sum = np.sum(stop_sign_points_is_invalid,
                                                 axis=-1)
        stop_sign_points_is_valid = stop_sign_points_is_invalid_sum == 0  # (stop_sign_num)
        npz_payload_dict["stop_sign_points"] = stop_sign_points[
            stop_sign_points_is_valid]  # (valid_stop_sign_num, safety_len, 2)

    speed_bump_points = npz_payload_dict.get(
        "speed_bump_points", None)  # (speed_bump_num, safety_len, 2)
    if speed_bump_points is not None:
        speed_bump_points_is_invalid = np.all(
            speed_bump_points[..., 0:2] == 0,
            axis=-1)  # (speed_bump_num, safety_len)
        speed_bump_points_is_invalid_sum = np.sum(speed_bump_points_is_invalid,
                                                  axis=-1)
        speed_bump_points_is_valid = speed_bump_points_is_invalid_sum == 0  # (speed_bump_num)
        npz_payload_dict["speed_bump_points"] = speed_bump_points[
            speed_bump_points_is_valid]  # (valid_speed_bump_num, safety_len, 2)

    crosswalk_points = npz_payload_dict.get(
        "crosswalk_points", None)  # (crosswalk_num, safety_len, 2)
    if crosswalk_points is not None:
        crosswalk_points_is_valid = ~np.all(
            crosswalk_points[...,
                             0:2] == 0, axis=-1)  #(crosswalk_num, safety_len)
        crosswalk_points_is_invalid_sum = np.sum(~crosswalk_points_is_valid,
                                                 axis=-1)
        crosswalk_points_is_valid = crosswalk_points_is_invalid_sum == 0  # (crosswalk_num)
        npz_payload_dict["crosswalk_points"] = crosswalk_points[
            crosswalk_points_is_valid]  # (valid_crosswalk_num, safety_len, 2)

    driveway_points = npz_payload_dict.get(
        "driveway_points", None)  # (driveway_num, safety_len, 2)
    if driveway_points is not None:
        driveway_points_is_invalid = np.all(
            driveway_points[...,
                            0:2] == 0, axis=-1)  #(driveway_num, safety_len)
        driveway_points_is_invalid_sum = np.sum(driveway_points_is_invalid,
                                                axis=-1)
        driveway_points_is_valid = driveway_points_is_invalid_sum == 0  # (driveway_num)
        npz_payload_dict["driveway_points"] = driveway_points[
            driveway_points_is_valid]  # (valid_driveway_num, safety_len, 2)

    road_edge = npz_payload_dict.get("road_edge",
                                     None)  # (chosen_edge_num, safety_len, 2)
    if road_edge is not None:
        road_edge_is_invalid = np.all(road_edge[..., 0:2] == 0,
                                      axis=-1)  #(chosen_edge_num, safety_len)
        road_edge_is_invalid_sum = np.sum(road_edge_is_invalid, axis=-1)
        road_edge_is_valid = road_edge_is_invalid_sum == 0  # (chosen_edge_num)
        npz_payload_dict["road_edge"] = road_edge[
            road_edge_is_valid]  # (valid_chosen_edge_num, safety_len, 2)

        # road_edge_type : (chosen_edge_num, 3)  -> (valid_chosen_edge_num, 3)
        npz_payload_dict["road_edge_type"] = npz_payload_dict["road_edge_type"][
            road_edge_is_valid]  # (valid_chosen_edge_num , 3)

    # static_objects : (chosen_static_num, 10) 뒤 10개 속성 중, 앞 5개가 전부 0. 이면 무효
    static_objects = npz_payload_dict.get("static_objects",
                                          None)  # (chosen_static_num, 10)
    if static_objects is not None:
        static_objects_is_invalid = np.all(static_objects[..., 0:5] == 0,
                                           axis=-1)  #(chosen_static_num)
        static_objects_is_valid = np.logical_not(
            static_objects_is_invalid)  # (chosen_static_num)
        npz_payload_dict["static_objects"] = static_objects[
            static_objects_is_valid]  # (valid_chosen_static_num, 10)

    return npz_payload_dict


def _apply_unvalid_at_unnorm_selected_traj_raw(
        unnorm_selected_traj_raw: torch.Tensor,  # (B, 1+Pnn, 1+future_len, 4)
        gt_valid_mask: torch.Tensor,  # (B, 1+Pnn, future_len)
) -> torch.Tensor:  # (B, 1+Pnn, 1+future_len, 4)
    # 중요: unnorm_selected_traj 에서, GT가 없는 구간은 0. 으로 채워집니다.
    unnorm_selected_traj_raw_future_len = unnorm_selected_traj_raw[:, :,
                                                                   1:, :]  # (B, 1+Pnn, future_len, 4)
    # apply gt_valid_mask to unnorm_selected_traj_raw_future_len
    unnorm_selected_traj_raw_future_len = unnorm_selected_traj_raw_future_len * \
                                          gt_valid_mask[
                                              ..., None]  # (B, 1+Pnn, future_len, 4)
    unnorm_selected_traj = unnorm_selected_traj_raw.clone()
    unnorm_selected_traj[:, :, 1:, :] = unnorm_selected_traj_raw_future_len
    return unnorm_selected_traj


def _predict_one_rollout_sequential(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    batch_size: int,
    one_or_pnn: int,
    save_image: bool,
    save_video: bool,
    sample_idx_offset: int,
    draw_batch_idx: int = 0,
) -> None:
    """rollout을 1개만(1번만) 순차 실행합니다.

    Args:
        args (Any): 설정 객체. shape: ()
        model (nn.Module): 예측 모델. shape: ()
        norm_inputs (Dict[str, Any]):
            정규화된 입력 dict.
            주요 텐서 shape 예:
              - ego_agent_past: (B, T_past, 11)
              - near_agents_past: (B, Pnn, T_past, 11)
              - target_future_valid: (B, 1+Pnn, future_len)
        state_normalizer (StateNormalizer): (x,y,cos,sin) 변환 도구. shape: ()
        observation_normalizer (ObservationNormalizer): 입력 dict 변환 도구. shape: ()
        rollout_idx (int): 0부터 시작하는 rollout 번호. shape: ()
        base_seed (int): 기본 seed. shape: ()
        ddp_rank (int): 분산 rank. shape: ()
        batch_size (int): 배치 크기 B. shape: ()
        one_or_pnn (int): (1+Pnn). shape: ()
        save_image (bool): 이 rollout에서 이미지 저장 여부. shape: ()
        save_video (bool): 이 rollout에서 영상 저장 여부. shape: ()
        sample_idx_offset (int):
            npz 파일명 충돌 방지용 오프셋(보통 rollout_idx * B).
            shape: ()
        draw_batch_idx (int): 배치(B)에서 시각화할 샘플 인덱스. shape: ()

    Returns:
        None
    """
    # ✅ 시각화 상태
    vis_state: _RolloutVisualizationState = _init_rollout_visualization_state(
        save_image=bool(save_image),
        save_video=bool(save_video),
        draw_batch_idx=int(draw_batch_idx),
    )

    future_len: int = int(getattr(args, "future_len"))

    # norm_inputs는 공유 객체일 수 있으니, rollout 내부에서는 얕은 복사본을 사용합니다.
    norm_inputs_copy_init: Dict[str, Any] = dict(norm_inputs)
    # unnorm_inputs_copy: 값들이 "원래 단위"인 dict (B 기준)
    unnorm_inputs_copy: Dict[str, Any] = _initialize_unnorm_inputs_for_rollout(
        norm_inputs_copy=norm_inputs_copy_init,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
    )

    cached_valid_masks: Dict[
        str, torch.Tensor] = _build_cached_valid_masks_for_static_map_features(
            unnorm_inputs_copy=unnorm_inputs_copy)

    # agent_length_m/agent_width_m: (B, 1+Pnn)
    agent_length_m, agent_width_m = _extract_agent_box_size_m_from_past_states(
        ego_agent_past=unnorm_inputs_copy["ego_agent_past"],  # (B, T_past, 11)
        near_agents_past=unnorm_inputs_copy[
            "near_agents_past"],  # (B, Pnn, T_past, 11)
    )
    time_chunk_size = int(min(args.rollout_time_chunk_size, args.time_len))
    with torch.inference_mode():
        step_count = 0
        step_start = 0

        while step_start < future_len:
            remaining = int(future_len - step_start)
            gap = int(min(time_chunk_size, remaining))

            # norm_inputs_step: 현재 step에서 모델에 넣을 정규화 입력
            norm_inputs_step: Dict[
                str, Any] = _build_norm_inputs_from_unnorm_inputs(
                    unnorm_inputs_copy=unnorm_inputs_copy,
                    state_normalizer=state_normalizer,
                    observation_normalizer=observation_normalizer,
                )

            # GT 미래(원래 단위)
            unnorm_ego_future_gt_4_dim = unnorm_inputs_copy[
                "ego_future_gt_4_dim"]  # (B, future_len, 4)
            unnorm_near_future_gt_4_dim = unnorm_inputs_copy[
                "near_future_gt_4_dim"]  # (B, Pnn, future_len, 4)

            # 후보 K개 중 best 선택
            """
            - best_normed_traj: (B_all, 1+Pnn, 1+T, 4)
            - best_distance_m_per_agent: (B_all, 1+Pnn)
            """
            best_normed_traj, best_dist_m = _select_best_trajectory_by_sample_k(
                args=args,
                model=model,
                norm_inputs_step=norm_inputs_step,
                state_normalizer=state_normalizer,
                unnorm_gt_ego_future_4_dim=unnorm_ego_future_gt_4_dim,
                # (B, future_len, 4)
                unnorm_gt_near_future_4_dim=unnorm_near_future_gt_4_dim,
                # (B, Pnn, future_len, 4)
                agent_length_m=agent_length_m,  # (B, 1+Pnn)
                agent_width_m=agent_width_m,  # (B, 1+Pnn)
                batch_size=int(batch_size),
                one_or_pnn=int(one_or_pnn),
                future_len=int(future_len),
                rollout_idx=int(rollout_idx),
                base_seed=int(base_seed),
                ddp_rank=int(ddp_rank),
                step_idx=int(step_start),
            )
            # best_normed_traj: (B, 1+Pnn, 1+future_len, 4)
            unnorm_best_traj = state_normalizer.inverse(best_normed_traj)
            # unnorm_selected_traj_raw: (B, 1+Pnn, 1+future_len, 4)
            # gt_valid_mask: (B_all, 1+Pnn, future_len)
            unnorm_selected_traj_raw, gt_valid_mask = _apply_recovery_if_needed(
                args=args,
                unnorm_selected_traj=
                unnorm_best_traj,  # (B, 1+Pnn, 1+future_len, 4)
                expert_distance_m=best_dist_m,  # (B, 1+Pnn)
                unnorm_gt_ego_future_4_dim=
                unnorm_ego_future_gt_4_dim,  # (B, future_len, 4)
                unnorm_gt_near_future_4_dim=
                unnorm_near_future_gt_4_dim,  # (B, Pnn, future_len, 4)
                future_len=int(future_len),
            )
            unnorm_selected_traj = _apply_unvalid_at_unnorm_selected_traj_raw(
                unnorm_selected_traj_raw, gt_valid_mask)
            normed_selected_traj = state_normalizer(unnorm_selected_traj)

            # ✅ (그림/영상) 준비는 1번만
            vis_state = _maybe_prepare_rollout_visualization_once(
                args=args,
                norm_inputs=norm_inputs,
                state=vis_state,  # _RolloutVisualizationState
            )

            # ✅ (그림) 프레임 저장
            _maybe_draw_rollout_visualization_frame(
                state=vis_state,
                unnorm_inputs_copy=unnorm_inputs_copy,
                normed_selected_traj=normed_selected_traj,  # (B, 1+Pnn, 1+T, 4)
                state_normalizer=state_normalizer,
                step_idx=int(step_start),
            )
            # ✅ npz 저장 (execute 전)
            step_count_for_save = int(step_count) + 1
            if args.save_inference_data:
                (demo_ego_future_4_dim, demo_near_future_4_dim
                ) = _build_generated_demo_futures_from_selected_traj(
                    unnorm_selected_traj=
                    unnorm_selected_traj,  # (B, 1+Pnn, 1+future_len, 4) # 중요: unnorm_selected_traj 에서, GT가 없는 구간은 0. 으로 채워집니다.
                )

                unnorm_inputs_for_save = dict(unnorm_inputs_copy)
                unnorm_inputs_for_save[
                    "ego_future_gt_4_dim"] = demo_ego_future_4_dim  # (B, future_len, 4)
                unnorm_inputs_for_save[
                    "near_future_gt_4_dim"] = demo_near_future_4_dim  # (B, Pnn, future_len, 4)
                _save_inference_data(
                    args.save_cache_path,
                    unnorm_inputs_for_save,
                    step_count=int(step_count_for_save),
                    sample_idx_offset=int(
                        sample_idx_offset
                    ),  # npz 파일명 충돌 방지용 오프셋(보통 rollout_idx * B).
                )

                limit_steps = int(
                    getattr(args, "rollout_step_count_for_save", -1))
                if limit_steps > 0 and step_count_for_save >= limit_steps:
                    break

            # ✅ execute: 앞 gap 스텝 반영
            # unnorm_best_traj: (B, 1+Pnn, 1+future_len, 4)
            # unnorm_target_pose_chunk: (B, 1+Pnn, gap, 4)
            unnorm_target_pose_chunk = unnorm_best_traj[:, :, 1:gap + 1, :]
            unnorm_ego_pose_chunk = unnorm_target_pose_chunk[:,
                                                             0, :, :]  # (B, gap, 4)
            unnorm_near_pose_chunk = unnorm_target_pose_chunk[:,
                                                              1:, :, :]  # (B, Pnn, gap, 4)
            unnorm_inputs_copy = _update_merged_inputs_unnorm_inplace_for_time_chunk(
                unnorm_inputs_copy=unnorm_inputs_copy,
                unnorm_ego_pose_chunk=unnorm_ego_pose_chunk,  # (B, gap, 4)
                unnorm_near_pose_chunk=unnorm_near_pose_chunk,  # (B, Pnn, gap, 4)
                cached_valid_masks=cached_valid_masks,
            )

            step_start += int(gap)
            step_count += 1

    # ✅ (영상) PNG -> 영상 생성
    _finalize_rollout_visualization_video_if_needed(
        args=args,
        state=vis_state,
    )


from typing import Any, Dict
import torch
import torch.nn as nn


def _predict_rollouts_sequential(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
    rollout_number: int,
    base_seed: int,
    ddp_rank: int,
) -> None:
    """rollout_number 만큼 rollout을 1개씩(for문) 순차 실행합니다.

    목표:
        - rollout을 묶어서 처리하지 않습니다.
        - 코드 흐름을 단순하게 유지합니다.

    Args:
        args (Any): 설정 객체. shape: ()
        model (nn.Module): 예측 모델. shape: ()
        norm_inputs (Dict[str, Any]):
            정규화된 입력 dict.
            예: ego_agent_past (B, T_past, 11), target_future_valid (B, 1+Pnn, future_len)
        outputs (Dict[str, Any]):
            GT 미래 등이 들어있는 dict.
            예: ego_future_gt_4_dim (B, future_len, 4)
        state_normalizer (StateNormalizer): (x,y,cos,sin) 변환 도구. shape: ()
        observation_normalizer (ObservationNormalizer): 입력 dict 변환 도구. shape: ()
        rollout_number (int): rollout 개수 R. shape: ()
        base_seed (int): 기본 seed. shape: ()
        ddp_rank (int): 분산 rank. shape: ()

    Returns:
        None
    """
    rollout_number_i = int(max(1, int(rollout_number)))

    target_future_valid = norm_inputs.get("target_future_valid", None)
    if not isinstance(target_future_valid, torch.Tensor):
        raise KeyError(
            "norm_inputs에 'target_future_valid'(torch.Tensor)가 필요합니다.")

    batch_size = int(target_future_valid.shape[0])  # B
    one_or_pnn = int(target_future_valid.shape[1])  # (1+Pnn)

    # ✅ GT 미래를 norm_inputs에 1번만 넣어 둡니다.
    ego_future_gt_4_dim = outputs["ego_future_gt_4_dim"]  # (B, future_len, 4)
    near_future_gt_4_dim = outputs[
        "near_future_gt_4_dim"]  # (B, Pnn, future_len, 4)

    norm_inputs["ego_future_gt_4_dim"] = state_normalizer(ego_future_gt_4_dim)
    norm_inputs["near_future_gt_4_dim"] = state_normalizer(near_future_gt_4_dim)

    draw_batch_idx = int(getattr(args, "draw_batch_idx", 0))

    for r in range(rollout_number_i):
        # ✅ 이미지/영상은 파일명 충돌 위험이 있어서 "첫 rollout만" 켜는 게 안전합니다.
        enable_image = bool(getattr(args, "save_image", False)) and int(r) == 0
        enable_video = bool(getattr(args, "save_video", False)) and int(r) == 0

        # ✅ npz 파일명 충돌 방지용
        sample_idx_offset = int(r) * int(batch_size)

        _predict_one_rollout_sequential(
            args=args,
            model=model,
            norm_inputs=norm_inputs,
            state_normalizer=state_normalizer,
            observation_normalizer=observation_normalizer,
            rollout_idx=int(r),
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            batch_size=int(batch_size),
            one_or_pnn=int(one_or_pnn),
            save_image=bool(enable_image),
            save_video=bool(enable_video),
            sample_idx_offset=int(sample_idx_offset),
            draw_batch_idx=int(draw_batch_idx),
        )


def _save_inference_data(
    dir: str,
    unnorm_inputs_copy: Dict[str, Any],
    step_count: int,
    *,
    sample_idx_offset: int = 0,
) -> None:
    """rollout 중간 상태(unnorm_inputs_copy)를 npz로 저장합니다.

    핵심:
        - 순차 rollout에서는 (rollout마다) 배치 인덱스가 반복됩니다.
        - 그래서 파일명 충돌(덮어쓰기)을 막으려면 sample_idx_offset이 필요합니다.

    Args:
        dir (str): 저장 폴더 경로. shape: ()
        unnorm_inputs_copy (Dict[str, Any]):
            저장할 입력 dict.
            주요 텐서 shape 예:
              - ego_future_gt_4_dim:  (B, future_len, 4)
              - near_future_gt_4_dim: (B, Pnn, future_len, 4)
        step_count (int): rollout 루프 저장 step 번호. shape: ()
        sample_idx_offset (int):
            파일 이름의 b{sample_idx}에 더할 값.
            예: rollout_idx * B
            shape: ()

    Returns:
        None
    """
    os.makedirs(dir, exist_ok=True)

    ego_future_gt_4_dim = unnorm_inputs_copy[
        "ego_future_gt_4_dim"]  # (B, future_len, 4)
    # batch_size: (B)
    batch_size = int(ego_future_gt_4_dim.shape[0])



    unnorm_inputs_np = _gpu_tensor_to_cpu_np(
        unnorm_inputs_copy=unnorm_inputs_copy,
        batch_size=int(batch_size),
    )

    enable_fsync = bool(
        _read_float_env_safe("DP_INFERENCE_NPZ_FSYNC", 1.0) > 0.0)
    """
    True : 더 안전하지만 느려질 수 있음
    False : 갑작스런 전원/크래시 때 마지막 파일이 유실될 가능성이 조금 더 커집니다. (대신 빠를 수 있음)
    """

    for current_batch_idx in range(batch_size):
        a_inputs_copy_dict: Dict[str,
                                 Any] = _slice_one_sample_from_cpu_batch_cache(
                                     unnorm_inputs_np=unnorm_inputs_np,
                                     sample_idx=int(current_batch_idx),
                                     batch_size=int(batch_size),
                                 )

        # ego_future_gt_4_np: (future_len, 4) -> ego_future_gt_3_np: (future_len, 3)
        ego_future_gt_4_np = a_inputs_copy_dict["ego_future_gt_4_dim"]
        a_inputs_copy_dict[
            "ego_future_gt_3_dim"] = _pose_4_dim_numpy_to_pose_3_dim_numpy(
                ego_future_gt_4_np)
        ego_future_valid_mask = np.any(ego_future_gt_4_np[..., 0:4] != 0,
                                       axis=-1)  # (future_len, )
        ego_agent_past = a_inputs_copy_dict["ego_agent_past"]  # (past_len, 11)
        ego_agent_current = ego_agent_past[-1, :]  # (11,)
        # ego_agent_current 을 확장하여, (future_len, 11) 모양으로 만든다.
        planner_future_11_dim = np.tile(
            ego_agent_current[np.newaxis, :],
            (int(ego_future_gt_4_np.shape[0]), 1),
        )  # (future_len, 11)
        planner_future_11_dim[:, 0:4] = ego_future_gt_4_np  # (future_len, 11)
        planner_future_11_dim[~ego_future_valid_mask, :8] = 0.0
        a_inputs_copy_dict["planner_future_11_dim"] = planner_future_11_dim

        near_future_gt_4_np = a_inputs_copy_dict["near_future_gt_4_dim"]
        a_inputs_copy_dict[
            "neighbor_future_gt_3_dim"] = _pose_4_dim_numpy_to_pose_3_dim_numpy(
                near_future_gt_4_np)

        # ✅ 순차 rollout용: 파일명 충돌 방지
        global_sample_idx = int(sample_idx_offset) + int(current_batch_idx)

        final_file_name = _build_inference_npz_file_name(
            scenario_id=str(a_inputs_copy_dict["scenario_id"]),
            step_count=int(step_count),
            sample_idx=int(global_sample_idx),
        )
        final_path = os.path.join(dir, final_file_name)
        tmp_path = final_path + ".tmp"

        npz_payload_dict: Dict[str,
                               Any] = _build_inference_npz_payload_for_save(
                                   a_inputs_copy_dict)
        npz_payload_dict = _remove_invalid_data(npz_payload_dict)

        try:
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **npz_payload_dict)
                f.flush()
                if enable_fsync:
                    os.fsync(f.fileno())
            os.replace(tmp_path, final_path)

        except BaseException:
            with contextlib.suppress(Exception):
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            raise


def _prepare_data_for_one_batch_draw(
    unnorm_inputs_copy: Dict[str, Any],
    normed_trajectories: torch.Tensor,
    state_normalizer: Any,
    draw_batch_idx: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    # 역정규화: ((1+)Pnn, 1+T, 4)
    a_unnorm_trajectory = state_normalizer.inverse(
        normed_trajectories[draw_batch_idx])
    a_unnorm_trajectory_np = a_unnorm_trajectory.cpu().numpy()

    unnorm_ego_future_gt_4_dim = unnorm_inputs_copy["ego_future_gt_4_dim"].cpu(
    ).numpy()  # (B*R, future_len, 4)
    a_unnorm_ego_future_gt_4_dim = unnorm_ego_future_gt_4_dim[
        draw_batch_idx]  # (future_len, 4)

    unnorm_near_future_gt_4_dim = unnorm_inputs_copy[
        "near_future_gt_4_dim"].cpu().numpy()  # (B*R, Pnn, future_len, 4)
    # near_future_gt_4_dim: (B
    # near_future_gt_3_dim : x, y, yaw
    unnorm_near_future_gt_3_dim = np.concatenate([
        unnorm_near_future_gt_4_dim[:, :, :, 0:2],
        np.arctan2(unnorm_near_future_gt_4_dim[:, :, :, 3:4],
                   unnorm_near_future_gt_4_dim[:, :, :, 2:3])
    ],
                                                 axis=-1)
    a_unnorm_near_future_gt_3_dim = unnorm_near_future_gt_3_dim[
        draw_batch_idx]  # (Pnn, future_len, 3)
    a_unnorm_inputs_copy = _shrink_batch_to_draw_idx(unnorm_inputs_copy,
                                                     draw_batch_idx)
    a_unnorm_inputs_np = _torch_to_numpy(a_unnorm_inputs_copy)
    return a_unnorm_inputs_np, a_unnorm_trajectory_np, a_unnorm_near_future_gt_3_dim, a_unnorm_ego_future_gt_4_dim


from typing import Any, Dict, Tuple
import torch.nn as nn

from typing import Any, Tuple
import torch


def _extract_agent_box_size_m_from_past_states(
    *,
    ego_agent_past: torch.Tensor,
    near_agents_past: torch.Tensor,
    default_length_m: float = 0.,
    default_width_m: float = 0.,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ego + near 전체 에이전트의 (length, width)를 11차원 past 상태에서 읽어옵니다.

    배경/의도
    --------
    - 각 에이전트는 차량 크기(길이/너비)가 다를 수 있습니다.
    - past 상태의 11차원에는 length, width가 포함되어 있으므로(인덱스 6, 7),
    - 구현 편의상 "마지막 시간 스텝(현재 시점)"의 값을 사용합니다.

    유효/무효 규칙
    -------------
    - 11차원 중 앞 8개 값(0~7)이 전부 0이면 그 시점은 무효로 봅니다.
    - 무효인 경우 length/width도 0일 가능성이 크므로,
      안전하게 기본값(default_length_m/default_width_m)을 넣습니다.

    Args:
        ego_agent_past (torch.Tensor):
            ego 과거 상태.
            shape: (B_all, T_past, 11)
            여기서 B_all은 rollout batch까지 펼친 크기(B*R)입니다.
        near_agents_past (torch.Tensor):
            near 과거 상태.
            shape: (B_all, Pnn, T_past, 11)
        default_length_m (float):
            무효 에이전트에 넣을 기본 길이(m). shape: ()
        default_width_m (float):
            무효 에이전트에 넣을 기본 너비(m). shape: ()

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (agent_length_m, agent_width_m)
            - agent_length_m shape: (B_all, 1+Pnn)
            - agent_width_m  shape: (B_all, 1+Pnn)
            agent 순서는 [ego(0번)] + [near(1..Pnn)] 입니다.
    """
    # 현재 시점(마지막 time index)에서 size 읽기
    # ego_current:  (B_all, 11)
    ego_current = ego_agent_past[:, -1, :]
    # near_current: (B_all, Pnn, 11)
    near_current = near_agents_past[:, :, -1, :]

    # 유효 여부: 앞 8개가 전부 0이면 무효
    # ego_valid:  (B_all,)
    ego_valid = torch.any(ego_current[:, :8] != 0.0, dim=-1)
    # near_valid: (B_all, Pnn)
    near_valid = torch.any(near_current[..., :8] != 0.0, dim=-1)

    # length/width 인덱스: 6, 7
    # ego_length:  (B_all,)
    # ego_width:   (B_all,)
    ego_length = ego_current[:, 6]
    ego_width = ego_current[:, 7]

    # near_length: (B_all, Pnn)
    # near_width:  (B_all, Pnn)
    near_length = near_current[..., 6]
    near_width = near_current[..., 7]

    # (B_all, 1+Pnn)
    agent_length = torch.cat([ego_length[:, None], near_length], dim=1)
    agent_width = torch.cat([ego_width[:, None], near_width], dim=1)

    # (B_all, 1+Pnn)
    agent_valid = torch.cat([ego_valid[:, None], near_valid], dim=1)

    # 무효 에이전트는 기본값으로 대체
    default_len = torch.full_like(agent_length, float(default_length_m))
    default_wid = torch.full_like(agent_width, float(default_width_m))
    agent_length = torch.where(agent_valid, agent_length, default_len)
    agent_width = torch.where(agent_valid, agent_width, default_wid)

    # 거리 계산용으로 float32 + 최소값 클램프
    agent_length = agent_length.to(dtype=torch.float32)
    agent_width = agent_width.to(dtype=torch.float32)

    return agent_length, agent_width


def _build_box_corners_xy_from_pose_4_dim_with_size(
    pose_4_dim: torch.Tensor,
    length_m: torch.Tensor,
    width_m: torch.Tensor,
) -> torch.Tensor:
    """(x, y, cos, sin) 포즈와 (length, width)로 박스 4개 코너의 (x,y)를 만듭니다.

    입력/출력 shape
    --------------
    - pose_4_dim: (..., 4)
    - length_m:   pose_4_dim.shape[:-1] 와 동일한 shape
    - width_m:    pose_4_dim.shape[:-1] 와 동일한 shape
    - 반환 corners_xy: (..., 4, 2)

    코너 순서(예측/정답이 동일한 규칙이면 어느 순서든 상관 없음)
    ----------------------------------------------------------
      1) 앞-왼쪽
      2) 앞-오른쪽
      3) 뒤-오른쪽
      4) 뒤-왼쪽

    Args:
        pose_4_dim (torch.Tensor):
            포즈 텐서.
            shape: (..., 4)
            마지막 4는 (x, y, cos(yaw), sin(yaw))
        length_m (torch.Tensor):
            박스 길이(m).
            shape: pose_4_dim.shape[:-1]
        width_m (torch.Tensor):
            박스 너비(m).
            shape: pose_4_dim.shape[:-1]

    Returns:
        torch.Tensor:
            코너 위치 (x,y).
            shape: (..., 4, 2)
    """
    # (...,)
    x = pose_4_dim[..., 0]
    y = pose_4_dim[..., 1]
    cos_h = pose_4_dim[..., 2]
    sin_h = pose_4_dim[..., 3]

    # (cos, sin) 길이 정리(회전에 안전하게)
    cos_h, sin_h = _normalize_cos_sin_for_rotation(
        cos_h.to(dtype=torch.float32),
        sin_h.to(dtype=torch.float32),
    )
    cos_h = cos_h.to(dtype=pose_4_dim.dtype)
    sin_h = sin_h.to(dtype=pose_4_dim.dtype)

    # half_l/half_w: (...)
    half_l = 0.5 * length_m.to(dtype=pose_4_dim.dtype)
    half_w = 0.5 * width_m.to(dtype=pose_4_dim.dtype)

    # dx/dy: (..., 4)
    dx = torch.stack([half_l, half_l, -half_l, -half_l], dim=-1)
    dy = torch.stack([half_w, -half_w, -half_w, half_w], dim=-1)

    # x_corner/y_corner: (..., 4)
    x_corner = x[..., None] + cos_h[..., None] * dx - sin_h[..., None] * dy
    y_corner = y[..., None] + sin_h[..., None] * dx + cos_h[..., None] * dy

    # (..., 4, 2)
    return torch.stack([x_corner, y_corner], dim=-1)


def _compute_expert_guidance_distance_m_per_agent(
        *,
        state_normalizer: Any,
        normed_trajectory: torch.Tensor,  # (B_all, 1+Pnn, 1+future_len, 4)
        unnorm_gt_ego_future_4_dim: torch.Tensor,  # (B_all, future_len, 4)
        unnorm_gt_near_future_4_dim: torch.
    Tensor,  # (B_all, Pnn, future_len, 4)
        compare_steps: int,
        agent_length_m: torch.Tensor,  # (B_all, 1+Pnn)
        agent_width_m: torch.Tensor,  # (B_all, 1+Pnn)
) -> torch.Tensor:  # (B_all, 1+Pnn)
    """후보 경로(예측)와 정답 미래 경로를 비교해, 에이전트별 '코너 평균 거리'를 계산합니다.

    계산 방식
    --------
    - 미래 첫 N스텝(compare_steps)만 비교합니다.
    - 에이전트마다 차량 크기(length/width)가 다를 수 있으므로,
      해당 에이전트의 크기를 사용해 4개 코너 위치를 만든 뒤 비교합니다.
    - 정답 경로가 0으로 채워진(=무효) 시간 구간은 비교에서 제외합니다.
    - 추가: 예측 경로 자체가 전체 0 패딩인(=존재하지 않는) agent는 거리 결과를 0으로 고정합니다.

    Returns:
        torch.Tensor:
            에이전트별 거리 점수.
            shape: (B_all, 1+Pnn)
            dtype: float32

    normed_trajectory 은 원래 무효 agent에 대해서는 전부 0 출력을 내놓습니다.
    하지만, 유효 agent에 대해서는 future_len 전부 유효 출력을 내놓습니다.
    """
    pred_future_len = int(normed_trajectory.shape[2]) - 1  # (1+T) -> T
    gt_future_len = int(unnorm_gt_ego_future_4_dim.shape[1])
    assert pred_future_len == gt_future_len, "Error: mismatched future_len between prediction and ground truth."
    assert compare_steps <= pred_future_len, "compare_steps exceeds future length."
    # pred_future_normed: (B_all, 1+Pnn, H, 4)
    pred_future_normed = normed_trajectory[:, :, 1:compare_steps + 1, :]
    # pred_future_unnorm: (B_all, 1+Pnn, H, 4)
    pred_future_unnorm = state_normalizer.inverse(pred_future_normed)

    # gt_all: (B_all, 1+Pnn, H, 4)
    ego_gt = unnorm_gt_ego_future_4_dim[:, :compare_steps, :]
    near_gt = unnorm_gt_near_future_4_dim[:, :, :compare_steps, :]
    gt_all = torch.cat([ego_gt[:, None, :, :], near_gt], dim=1)

    # 정답이 0패딩된 구간은 제외: valid_mask (B_all, 1+Pnn, H)
    valid_mask = torch.any(gt_all != 0.0, dim=-1)

    # 거리 계산은 float32로 안정화
    pred_f = pred_future_unnorm.to(dtype=torch.float32)  # (B_all, 1+Pnn, H, 4)
    gt_f = gt_all.to(dtype=torch.float32)  # (B_all, 1+Pnn, H, 4)

    # size를 시간축으로 확장: (B_all, 1+Pnn, H)
    length_h = agent_length_m.to(dtype=torch.float32)[:, :, None].expand(
        -1, -1, compare_steps)
    width_h = agent_width_m.to(dtype=torch.float32)[:, :, None].expand(
        -1, -1, compare_steps)

    # corners: (B_all, 1+Pnn, H, 4, 2)
    pred_corners = _build_box_corners_xy_from_pose_4_dim_with_size(
        pose_4_dim=pred_f,  # (B_all, 1+Pnn, H, 4)
        length_m=length_h,  # (B_all, 1+Pnn, H)
        width_m=width_h,  # (B_all, 1+Pnn, H)
    )
    gt_corners = _build_box_corners_xy_from_pose_4_dim_with_size(
        pose_4_dim=gt_f,  # (B_all, 1+Pnn, H, 4)
        length_m=length_h,  # (B_all, 1+Pnn, H)
        width_m=width_h,  # (B_all, 1+Pnn, H)
    )

    # diff: (B_all, 1+Pnn, H, 4, 2)
    diff = pred_corners - gt_corners

    # corner_dist: (B_all, 1+Pnn, H, 4)
    corner_dist = torch.sqrt(
        torch.clamp(diff[..., 0]**2 + diff[..., 1]**2, min=0.0))

    # step_dist: (B_all, 1+Pnn, H)  코너 평균
    step_dist = corner_dist.mean(dim=-1)

    valid_f = valid_mask.to(dtype=torch.float32)  # (B_all, 1+Pnn, H)
    sum_dist = (step_dist * valid_f).sum(dim=-1)  # (B_all, 1+Pnn)
    denom = torch.clamp(valid_f.sum(dim=-1), min=1.0)  # (B_all, 1+Pnn)

    dist = sum_dist / denom  # (B_all, 1+Pnn)

    # ✅ 무효 agent는 거리 결과를 0으로 고정 (에러/선택 로직 흔들림 방지)
    # ✅ 무효 agent 감지: (1+T, 4)가 전부 0이면 "존재하지 않는 agent"로 봅니다.
    # agent_has_any_value: (B_all, 1+Pnn)
    agent_has_any_value = torch.any(
        torch.any(normed_trajectory != 0.0, dim=-1),  # (B_all, 1+Pnn, 1+T)
        dim=-1,  # -> (B_all, 1+Pnn)
    )
    # invalid_agent_mask: (B_all, 1+Pnn)  True=무효 agent
    invalid_agent_mask = ~agent_has_any_value
    dist = torch.where(invalid_agent_mask, torch.zeros_like(dist), dist)

    return dist


# _select_best_trajectory_by_sample_k에서 "후보를 한 번에 몇 개씩 묶어 처리할지"를 캐시하는 args 속성 이름
_DP_SAMPLE_K_CANDIDATE_BATCH_ATTR_NAME = "_dp_sample_k_candidate_batch_size"


def _is_gpu_oom_error(err: BaseException) -> bool:
    """GPU 메모리 부족(Out Of Memory) 때문에 난 에러인지 확인합니다.

    후보 경로를 여러 개 한 번에 계산하면 빨라지지만,
    한 번에 처리하는 개수가 너무 크면 GPU 메모리가 부족해질 수 있습니다.

    이 함수는 예외(err)가 "GPU 메모리 부족" 때문에 난 것인지 판별합니다.

    Args:
        err (BaseException): 발생한 예외 객체. shape: ()

    Returns:
        bool:
            - True: GPU 메모리 부족으로 보이는 경우
            - False: 그 외의 경우
            shape: ()
    """
    # torch 버전에 따라 전용 타입이 없을 수 있어서 문자열도 함께 확인합니다.
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
            err, torch.cuda.OutOfMemoryError):
        return True

    msg = str(err).lower()
    if "out of memory" in msg:
        return True
    if "cuda" in msg and "memory" in msg:
        return True
    return False


def _clear_gpu_cache_after_oom() -> None:
    """GPU 메모리 부족 후 다음 시도를 위해 캐시를 정리합니다.

    동작
    ----
    - 파이썬이 들고 있던 임시 객체를 가능한 한 빨리 정리합니다.
    - GPU 캐시도 비워서 다음 시도가 더 잘 되도록 돕습니다.

    Returns:
        None
    """
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _repeat_inputs_for_candidate_batch(
    *,
    norm_inputs_step: Dict[str, Any],
    repeat: int,
    batch_size: int,
) -> Dict[str, Any]:
    """후보 개수(repeat)만큼 입력 배치를 늘린 dict를 만듭니다.

    이 함수가 하는 일
    ---------------
    후보를 여러 개 한 번에 모델에 넣으려면,
    입력의 첫 번째 차원(B)을 repeat배로 늘린 (B*repeat, ...) 형태가 필요합니다.

    처리 규칙
    --------
    - torch.Tensor 이고 shape[0] == B 인 경우:
        (B, ...) -> (B*repeat, ...) 로 늘립니다.
    - list 이고 길이가 B 인 경우:
        길이 B -> 길이 B*repeat 로 늘립니다.
    - 그 외 값(None, 숫자, 길이가 B가 아닌 리스트 등):
        그대로 둡니다.

    Args:
        norm_inputs_step (Dict[str, Any]):
            원본 입력 dict. shape: ()
        repeat (int):
            후보를 몇 개 묶어서 한 번에 계산할지. shape: ()
        batch_size (int):
            원본 배치 크기 B. shape: ()

    Returns:
        Dict[str, Any]:
            배치가 늘어난 입력 dict. shape: ()
    """
    r = int(max(1, int(repeat)))
    b = int(max(1, int(batch_size)))

    out: Dict[str, Any] = {}
    for k, v in norm_inputs_step.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == b:
            # v: (B, ...)
            repeat_factors = (r,) + (1,) * (int(v.dim()) - 1)
            out[k] = v.repeat(*repeat_factors)  # (B*r, ...)
        elif isinstance(v, list) and int(len(v)) == b:
            out[k] = v * r  # 길이: B*r
        else:
            out[k] = v
    return out


def _build_inference_noise_batch_for_candidate_range(
    *,
    reference_tensor: torch.Tensor,
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    noise_std: float,
    seed_stride: int,
    cand_start_idx: int,
    cand_count: int,
) -> torch.Tensor:
    """cand_start_idx부터 cand_count개 후보의 노이즈를 한 번에 만들어 합칩니다.

    Args:
        reference_tensor (torch.Tensor):
            device/dtype 기준 텐서. 보통 ego_agent_past 사용.
            shape 예: (B, T_past, 11)
        batch_size (int): 배치 크기 B. shape: ()
        one_or_pnn (int): (1+Pnn). shape: ()
        future_len (int): 미래 길이. shape: ()
        rollout_idx (int): rollout 번호. shape: ()
        base_seed (int): 기본 seed. shape: ()
        ddp_rank (int): 분산 rank. shape: ()
        step_idx (int): rollout 안의 step 번호. shape: ()
        noise_std (float): 노이즈 크기. shape: ()
        seed_stride (int): 후보마다 seed를 띄우는 간격. shape: ()
        cand_start_idx (int): 시작 후보 인덱스. shape: ()
        cand_count (int): 후보 개수. shape: ()

    Returns:
        torch.Tensor:
            후보들을 이어붙인 노이즈 텐서.
            shape: (B*cand_count, 1+Pnn, future_len, 4)
    """
    c = int(max(1, int(cand_count)))
    b = int(max(1, int(batch_size)))

    noises = []
    for local_i in range(c):
        cand_idx = int(cand_start_idx) + int(local_i)
        cand_base_seed = int(base_seed) + int(cand_idx) * int(seed_stride)

        # noise_i: (B, 1+Pnn, future_len, 4)
        noise_i = _build_inference_noise_for_rollout_chunk(
            reference_tensor=reference_tensor,
            batch_size=b,
            one_or_pnn=int(one_or_pnn),
            future_len=int(future_len),
            rollout_idx=int(rollout_idx),
            base_seed=int(cand_base_seed),
            ddp_rank=int(ddp_rank),
            step_idx=int(step_idx),
            noise_std=float(noise_std),
        )
        noises.append(noise_i)

    # noise_stack: (cand_count, B, 1+Pnn, future_len, 4)
    noise_stack = torch.stack(noises, dim=0)

    # noise_flat: (cand_count*B, 1+Pnn, future_len, 4)
    noise_flat = noise_stack.reshape(
        int(c) * int(b),
        int(one_or_pnn),
        int(future_len),
        4,
    )
    return noise_flat


def _forward_and_score_candidate_batch(
    *,
    args: Any,
    model: nn.Module,
    norm_inputs_step: Dict[str, Any],
    state_normalizer: Any,
    unnorm_gt_ego_future_4_dim: torch.Tensor,
    unnorm_gt_near_future_4_dim: torch.Tensor,
    agent_length_m: torch.Tensor,
    agent_width_m: torch.Tensor,
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    cand_start_idx: int,
    cand_count: int,
    seed_stride: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """후보 cand_count개를 한 번에 모델에 넣고, 거리 점수까지 계산합니다.

    Args:
        args (Any): 설정 객체. shape: ()
        model (nn.Module): 예측 모델. shape: ()
        norm_inputs_step (Dict[str, Any]): 현재 step 입력. shape: ()
        state_normalizer (Any): (x,y,cos,sin) 변환 도구. shape: ()
        unnorm_gt_ego_future_4_dim (torch.Tensor): 정답 ego 미래. shape: (B, future_len, 4)
        unnorm_gt_near_future_4_dim (torch.Tensor): 정답 near 미래. shape: (B, Pnn, future_len, 4)
        agent_length_m (torch.Tensor): 에이전트 길이. shape: (B, 1+Pnn)
        agent_width_m (torch.Tensor): 에이전트 너비. shape: (B, 1+Pnn)
        batch_size (int): B. shape: ()
        one_or_pnn (int): 1+Pnn. shape: ()
        future_len (int): 미래 길이. shape: ()
        rollout_idx (int): rollout 번호. shape: ()
        base_seed (int): 기본 seed. shape: ()
        ddp_rank (int): rank. shape: ()
        step_idx (int): step 번호. shape: ()
        cand_start_idx (int): 후보 시작 인덱스. shape: ()
        cand_count (int): 후보 개수. shape: ()
        seed_stride (int): seed 간격. shape: ()

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (cand_traj, cand_dist)
            - cand_traj: shape (cand_count, B, 1+Pnn, 1+future_len, 4)
            - cand_dist: shape (cand_count, B, 1+Pnn)
    """
    c = int(max(1, int(cand_count)))
    b = int(max(1, int(batch_size)))

    # 1) 후보별 노이즈 만들기
    # inference_noise_flat: (B*cand_count, 1+Pnn, future_len, 4)
    inference_noise_flat = _build_inference_noise_batch_for_candidate_range(
        reference_tensor=norm_inputs_step["ego_agent_past"],
        batch_size=b,
        one_or_pnn=int(one_or_pnn),
        future_len=int(future_len),
        rollout_idx=int(rollout_idx),
        base_seed=int(base_seed),
        ddp_rank=int(ddp_rank),
        step_idx=int(step_idx),
        noise_std=float(getattr(args, "fine_tune_temperature", 0.0)),
        seed_stride=int(seed_stride),
        cand_start_idx=int(cand_start_idx),
        cand_count=int(c),
    )

    # 2) 입력 dict를 (B*cand_count, ...)로 늘리기
    cand_inputs = _repeat_inputs_for_candidate_batch(
        norm_inputs_step=norm_inputs_step,
        repeat=int(c),
        batch_size=int(b),
    )
    cand_inputs["inference_noise"] = inference_noise_flat  # (B*c, 1+Pnn, future_len, 4)

    # 3) 모델 forward (한 번)
    decoder_output = _forward_model_for_validation(
        args=args,
        model=model,
        norm_inputs=cand_inputs,
    )

    cand_traj_flat = decoder_output.get("integrated_trajectory", None)
    if not isinstance(cand_traj_flat, torch.Tensor):
        raise RuntimeError("decoder_output에 'integrated_trajectory'가 없습니다.")

    # cand_traj_flat: (B*cand_count, 1+Pnn, 1+future_len, 4)
    cand_traj = cand_traj_flat.reshape(
        int(c),
        int(b),
        int(one_or_pnn),
        1 + int(future_len),
        4,
    )

    # 4) 정답/크기 텐서도 (B*cand_count, ...)로 늘려서 거리 계산
    # gt_ego_rep:  (B*c, future_len, 4)
    gt_ego_rep = unnorm_gt_ego_future_4_dim.repeat(int(c), 1, 1)
    # gt_near_rep: (B*c, Pnn, future_len, 4)
    gt_near_rep = unnorm_gt_near_future_4_dim.repeat(int(c), 1, 1, 1)
    # len_rep/wid_rep: (B*c, 1+Pnn)
    len_rep = agent_length_m.repeat(int(c), 1)
    wid_rep = agent_width_m.repeat(int(c), 1)

    # cand_dist_flat: (B*c, 1+Pnn)
    cand_dist_flat = _compute_expert_guidance_distance_m_per_agent(
        state_normalizer=state_normalizer,
        normed_trajectory=cand_traj_flat,
        unnorm_gt_ego_future_4_dim=gt_ego_rep,
        unnorm_gt_near_future_4_dim=gt_near_rep,
        compare_steps=int(getattr(args, "time_step_for_compare", 1)),
        agent_length_m=len_rep,
        agent_width_m=wid_rep,
    )

    # cand_dist: (cand_count, B, 1+Pnn)
    cand_dist = cand_dist_flat.reshape(int(c), int(b), int(one_or_pnn))

    return cand_traj, cand_dist


def _select_best_from_candidate_batch_jointly(
    *,
    best_traj: Optional[torch.Tensor],
    best_dist: Optional[torch.Tensor],
    best_score: Optional[torch.Tensor],
    cand_traj: torch.Tensor,
    cand_dist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """샘플마다 후보 1개를 고르는 방식(select_jointly=True)으로 best를 갱신합니다.

    Args:
        best_traj (Optional[torch.Tensor]):
            지금까지의 best 경로.
            shape: (B, 1+Pnn, 1+future_len, 4) 또는 None
        best_dist (Optional[torch.Tensor]):
            지금까지의 best 거리(에이전트별).
            shape: (B, 1+Pnn) 또는 None
        best_score (Optional[torch.Tensor]):
            지금까지의 best 점수(샘플별 1개 값).
            shape: (B,) 또는 None
        cand_traj (torch.Tensor):
            이번 배치의 후보 경로들.
            shape: (Kc, B, 1+Pnn, 1+future_len, 4)
        cand_dist (torch.Tensor):
            이번 배치의 후보 거리들.
            shape: (Kc, B, 1+Pnn)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (best_traj, best_dist, best_score)
            - best_traj shape: (B, 1+Pnn, 1+future_len, 4)
            - best_dist shape: (B, 1+Pnn)
            - best_score shape: (B,)
    """
    # cand_score: (Kc, B)
    cand_score = cand_dist.to(dtype=torch.float32).mean(dim=2)

    # group_best_score: (B,)
    # group_best_idx:   (B,)
    group_best_score, group_best_idx = cand_score.min(dim=0)

    # cand_traj_perm: (B, Kc, 1+Pnn, 1+future_len, 4)
    cand_traj_perm = cand_traj.permute(1, 0, 2, 3, 4)
    b = int(cand_traj_perm.shape[0])
    a = int(cand_traj_perm.shape[2])
    t = int(cand_traj_perm.shape[3])

    # idx_traj: (B, 1, 1+Pnn, 1+future_len, 4)
    idx_traj = group_best_idx[:, None, None, None, None].expand(b, 1, a, t, 4)
    group_best_traj = torch.take_along_dim(cand_traj_perm, idx_traj,
                                           dim=1).squeeze(1)

    # cand_dist_perm: (B, Kc, 1+Pnn)
    cand_dist_perm = cand_dist.permute(1, 0, 2)
    idx_dist = group_best_idx[:, None, None].expand(b, 1, a)
    group_best_dist = torch.take_along_dim(cand_dist_perm, idx_dist,
                                           dim=1).squeeze(1)

    if best_traj is None or best_dist is None or best_score is None:
        return group_best_traj, group_best_dist, group_best_score

    better = group_best_score < best_score  # (B,)
    if torch.any(better):
        best_traj[better] = group_best_traj[better]
        best_dist[better] = group_best_dist[better]
    best_score = torch.where(better, group_best_score, best_score)
    return best_traj, best_dist, best_score


def _select_best_from_candidate_batch_per_agent(
    *,
    best_traj: Optional[torch.Tensor],
    best_dist: Optional[torch.Tensor],
    cand_traj: torch.Tensor,
    cand_dist: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """에이전트마다 후보를 따로 고르는 방식(select_jointly=False)으로 best를 갱신합니다.

    Args:
        best_traj (Optional[torch.Tensor]):
            지금까지의 best 경로.
            shape: (B, 1+Pnn, 1+future_len, 4) 또는 None
        best_dist (Optional[torch.Tensor]):
            지금까지의 best 거리(에이전트별).
            shape: (B, 1+Pnn) 또는 None
        cand_traj (torch.Tensor):
            이번 배치의 후보 경로들.
            shape: (Kc, B, 1+Pnn, 1+future_len, 4)
        cand_dist (torch.Tensor):
            이번 배치의 후보 거리들.
            shape: (Kc, B, 1+Pnn)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (best_traj, best_dist)
            - best_traj shape: (B, 1+Pnn, 1+future_len, 4)
            - best_dist shape: (B, 1+Pnn)
    """
    # group_best_dist: (B, 1+Pnn)
    # group_best_idx:  (B, 1+Pnn)
    group_best_dist, group_best_idx = cand_dist.min(dim=0)

    # cand_traj_perm: (B, 1+Pnn, Kc, 1+future_len, 4)
    cand_traj_perm = cand_traj.permute(1, 2, 0, 3, 4)
    b = int(cand_traj_perm.shape[0])
    a = int(cand_traj_perm.shape[1])
    t = int(cand_traj_perm.shape[3])

    # idx_traj: (B, 1+Pnn, 1, 1+future_len, 4)
    idx_traj = group_best_idx[..., None, None, None].expand(b, a, 1, t, 4)
    group_best_traj = torch.take_along_dim(cand_traj_perm, idx_traj,
                                           dim=2).squeeze(2)

    if best_traj is None or best_dist is None:
        return group_best_traj, group_best_dist

    better = group_best_dist < best_dist  # (B, 1+Pnn)
    if torch.any(better):
        best_traj[better] = group_best_traj[better]
    best_dist = torch.where(better, group_best_dist, best_dist)
    return best_traj, best_dist

def _select_best_trajectory_by_sample_k(
    *,
    args: Any,
    model: nn.Module,
    norm_inputs_step: Dict[str, Any],
    state_normalizer: Any,
    unnorm_gt_ego_future_4_dim: torch.Tensor,  # (B, future_len, 4)
    unnorm_gt_near_future_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    agent_length_m: torch.Tensor,  # (B, 1+Pnn)
    agent_width_m: torch.Tensor,  # (B, 1+Pnn)
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """같은 입력에서 후보 K개를 만들고, 규칙에 따라 최종 1개를 고릅니다.

    변경된 핵심
    ----------
    - 기존처럼 후보를 1개씩(for문) 처리하지 않고,
      후보를 "몇 개씩 묶어서" 한 번에 모델에 넣습니다.
    - 처음 실행에서는 args.fine_tune_gen_k부터 시작해서,
      GPU 메모리 부족이 나면 절반으로 줄이며 안전한 묶음 크기를 찾습니다.
    - 한 번 안전한 묶음 크기가 정해지면(args에 저장),
      이후 호출에서는 계속 그 크기로만 묶어서 처리합니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (best_normed_traj, best_distance_m_per_agent)
            - best_normed_traj: (B, 1+Pnn, 1+future_len, 4)
            - best_distance_m_per_agent: (B, 1+Pnn)
    """
    best_traj: Optional[torch.Tensor] = None
    best_dist: Optional[torch.Tensor] = None
    best_joint_score: Optional[torch.Tensor] = None  # (B,)

    seed_stride = 10_000_000
    select_jointly: bool = bool(args.select_jointly)

    k_total = int(max(1, int(getattr(args, "fine_tune_gen_k", 1))))

    # ✅ (중요) 처음엔 K부터 시작, 한 번 안전한 값이 잡히면 args에 저장된 값 사용
    cached_group = getattr(args, _DP_SAMPLE_K_CANDIDATE_BATCH_ATTR_NAME, None)
    if cached_group is None:
        cand_group_size = int(k_total)
    else:
        cand_group_size = int(max(1, int(cached_group)))
        cand_group_size = int(min(cand_group_size, k_total))

    cand_start = 0
    while cand_start < k_total:
        print("args._dp_sample_k_candidate_batch_size:", args._dp_sample_k_candidate_batch_size)
        group_count = int(min(cand_group_size, k_total - cand_start))

        try:
            # cand_traj_batch: (group_count, B, 1+Pnn, 1+future_len, 4)
            # cand_dist_batch: (group_count, B, 1+Pnn)
            cand_traj_batch, cand_dist_batch = _forward_and_score_candidate_batch(
                args=args,
                model=model,
                norm_inputs_step=norm_inputs_step,
                state_normalizer=state_normalizer,
                unnorm_gt_ego_future_4_dim=unnorm_gt_ego_future_4_dim,
                unnorm_gt_near_future_4_dim=unnorm_gt_near_future_4_dim,
                agent_length_m=agent_length_m,
                agent_width_m=agent_width_m,
                batch_size=int(batch_size),
                one_or_pnn=int(one_or_pnn),
                future_len=int(future_len),
                rollout_idx=int(rollout_idx),
                base_seed=int(base_seed),
                ddp_rank=int(ddp_rank),
                step_idx=int(step_idx),
                cand_start_idx=int(cand_start),
                cand_count=int(group_count),
                seed_stride=int(seed_stride),
            )

        except BaseException as e:
            # ✅ OOM이면 절반으로 줄이고 같은 cand_start에서 다시 시도
            if _is_gpu_oom_error(e):
                if cand_group_size <= 1:
                    raise
                cand_group_size = max(1, int(cand_group_size) // 2)
                _clear_gpu_cache_after_oom()

                # 더 작은 값은 안전하므로 저장(이후 호출도 이 값 사용)
                setattr(args, _DP_SAMPLE_K_CANDIDATE_BATCH_ATTR_NAME,
                        int(cand_group_size))
                continue
            raise

        # ✅ 첫 성공(또는 더 작은 값으로 성공) 시점에 고정값을 args에 저장
        if getattr(args, _DP_SAMPLE_K_CANDIDATE_BATCH_ATTR_NAME, None) is None:
            setattr(args, _DP_SAMPLE_K_CANDIDATE_BATCH_ATTR_NAME,
                    int(cand_group_size))

        if select_jointly:
            best_traj, best_dist, best_joint_score = _select_best_from_candidate_batch_jointly(
                best_traj=best_traj,
                best_dist=best_dist,
                best_score=best_joint_score,
                cand_traj=cand_traj_batch,
                cand_dist=cand_dist_batch,
            )
        else:
            best_traj, best_dist = _select_best_from_candidate_batch_per_agent(
                best_traj=best_traj,
                best_dist=best_dist,
                cand_traj=cand_traj_batch,
                cand_dist=cand_dist_batch,
            )

        cand_start += int(group_count)

        # (메모리 압박 완화) 다음 루프 전에 참조 해제
        del cand_traj_batch
        del cand_dist_batch

    assert isinstance(best_traj, torch.Tensor)
    assert isinstance(best_dist, torch.Tensor)
    return best_traj, best_dist


def _build_recovery_step_count_per_agent(
        *,
        args: Any,
        gt_valid_step_count_per_agent: torch.Tensor,  # (B_all, 1+Pnn)
) -> torch.Tensor:  # (B_all, 1+Pnn)
    """Recovery에서 '몇 스텝까지' GT 쪽으로 섞을지 에이전트별로 정합니다.

    """
    time_step_for_recover = int(args.time_step_for_recover)
    cfg = torch.full_like(gt_valid_step_count_per_agent,
                          fill_value=time_step_for_recover)  # (B_all, 1+Pnn)
    n_recovery = torch.clamp(cfg, min=1)  # (B_all, 1+Pnn)
    return n_recovery.to(dtype=torch.long)


def _build_recovery_lambda_per_agent(
    *,
    n_rec_per_agent: torch.Tensor,  # (B_all, 1+Pnn)
    future_len: int,
    device: torch.device,
) -> torch.Tensor:  # (B_all, 1+Pnn, future_len, 1)
    """시간이 지날수록 GT 비중이 커지도록 섞기 비율(lambda_)을 만듭니다.

    규칙
    ----
    - t=1..future_len에 대해 lambda_ = min(t / n_recovery, 1) 입니다.
    - 에이전트마다 n_rec 이 다를 수 있으므로, lam도 에이전트별로 다릅니다.

    Args:
        n_rec_per_agent (torch.Tensor):
            에이전트별 섞기 기준 스텝 수.
            shape: (B_all, 1+Pnn)
        future_len (int):
            전체 미래 길이. shape: ()
        device (torch.device):
            lam을 만들 장치. shape: ()

    Returns:
        torch.Tensor:
            섞기 비율.
            shape: (B_all, 1+Pnn, future_len, 1)
            dtype: torch.float32
    """
    steps = torch.arange(1,
                         int(future_len) + 1,
                         device=device,
                         dtype=torch.float32)  # (T,)
    lambda_ = steps[None, None, :] / n_rec_per_agent.to(
        dtype=torch.float32)[:, :, None]  # (B_all, 1+Pnn, T)
    lambda_ = lambda_.clamp(max=1.0)
    return lambda_[:, :, :, None]  # (B_all, 1+Pnn, T, 1)


def _mix_pred_and_gt_future_for_recovery(
    *,
    unnorm_pred_future: torch.Tensor,  # (B_all, 1+Pnn, future_len, 4)
    gt_future: torch.Tensor,  # (B_all, 1+Pnn, future_len, 4)
    gt_valid_mask: torch.Tensor,  # (B_all, 1+Pnn, future_len) bool
    lambda_: torch.Tensor,  # (B_all, 1+Pnn, future_len, 1) float32
) -> torch.Tensor:  # (B_all, 1+Pnn, future_len, 4)
    """예측 미래와 GT 미래를 섞되, GT가 없는 칸은 예측 값을 유지합니다.
    """
    pred_f = unnorm_pred_future.to(dtype=torch.float32)
    gt_f = gt_future.to(dtype=torch.float32)

    mixed_f = (1.0 - lambda_) * pred_f + lambda_ * gt_f  # (B_all, 1+Pnn, T, 4)

    # ✅ GT가 없는 칸은 섞지 않고 pred 유지
    mixed_f = torch.where(gt_valid_mask[..., None], mixed_f, pred_f)

    return mixed_f.to(dtype=unnorm_pred_future.dtype)


def _normalize_heading_cos_sin_in_pose_4_dim_inplace(
        *,
        pose_4_dim: torch.Tensor,  # (..., 4)
        valid_mask: torch.Tensor,  # pose_4_dim.shape[:-1]
) -> None:
    """(cos, sin)이 깨지지 않도록 유효한 칸만 길이를 1로 맞춥니다.

    Args:
        pose_4_dim (torch.Tensor):
            (x, y, cos, sin) 포즈 텐서.
            shape: (..., 4)
        valid_mask (torch.Tensor):
            정리할 칸(True) / 건드리지 않을 칸(False).
            shape: pose_4_dim.shape[:-1]

    Returns:
        None
    """
    if pose_4_dim.dim() < 2 or int(pose_4_dim.shape[-1]) != 4:
        raise ValueError("pose_4_dim은 (..., 4) 형태여야 합니다. "
                         f"현재 shape={tuple(pose_4_dim.shape)}")
    expected = tuple(int(x) for x in pose_4_dim.shape[:-1])
    if tuple(int(x) for x in valid_mask.shape) != expected:
        raise ValueError("valid_mask shape가 pose_4_dim과 맞지 않습니다. "
                         f"expected={expected}, got={tuple(valid_mask.shape)}")

    cos_v = pose_4_dim[..., 2]
    sin_v = pose_4_dim[..., 3]

    cos_n, sin_n = _normalize_cos_sin_for_rotation(
        cos_v.to(dtype=torch.float32),
        sin_v.to(dtype=torch.float32),
    )
    cos_n = cos_n.to(dtype=pose_4_dim.dtype)
    sin_n = sin_n.to(dtype=pose_4_dim.dtype)

    vm = valid_mask.to(dtype=torch.bool)
    pose_4_dim[..., 2] = torch.where(vm, cos_n, cos_v)
    pose_4_dim[..., 3] = torch.where(vm, sin_n, sin_v)


def _apply_recovery_if_needed(
    *,
    args: Any,
    unnorm_selected_traj: torch.Tensor,  # (B*R, 1+Pnn, 1+future_len, 4)
    expert_distance_m: torch.Tensor,  # (B_all, 1+Pnn)
    unnorm_gt_ego_future_4_dim: torch.Tensor,  # (B*R, future_len, 4)
    unnorm_gt_near_future_4_dim: torch.Tensor,  # (B*R, Pnn, future_len, 4)
    future_len: int,
) -> Tuple[
        torch.Tensor, torch.
        Tensor]:  # (B_all, 1+Pnn, 1+future_len, 4), (B_all, 1+Pnn, future_len) bool
    """선택된 경로가 GT에서 너무 멀면, 에이전트별로 GT 쪽으로 부드럽게 섞습니다.

    ----------
    - GT 유효 길이를 'ego만' 보지 않고, (샘플, 에이전트)별로 계산해 반영합니다.
    - GT가 0으로 비어있는 시간 구간은 섞지 않고 예측 값을 그대로 유지합니다.

    Returns:
        torch.Tensor:
            복구가 반영된 경로(원래 단위).
            shape: (B_all, 1+Pnn, 1+future_len, 4)
    """
    # gt_all: (B_all, 1+Pnn, future_len, 4)
    gt_all = torch.cat(
        [
            unnorm_gt_ego_future_4_dim[:, None, :, :],
            unnorm_gt_near_future_4_dim
        ],
        dim=1,
    )

    # gt_valid_mask: (B_all, 1+Pnn, future_len)
    gt_valid_mask = torch.any(gt_all != 0.0, dim=-1).to(dtype=torch.bool)

    # ✅ (중요) GT 유효 스텝 수를 에이전트별로 계산
    # gt_valid_count: (B_all, 1+Pnn)
    gt_valid_count = gt_valid_mask.sum(dim=-1).to(dtype=torch.long)

    if not args.use_recovery:
        return unnorm_selected_traj, gt_valid_mask

    threshold_m = 3.0  # 논문 값 고정
    trigger = expert_distance_m > float(threshold_m)  # (B_all, 1+Pnn)

    if not torch.any(trigger):
        return unnorm_selected_traj, gt_valid_mask

    pred_avail_len = int(unnorm_selected_traj.shape[2]) - 1
    if int(pred_avail_len) != int(future_len):
        raise ValueError(
            "unnorm_selected_traj의 미래 길이와 future_len이 다릅니다. "
            f"pred_avail_len={pred_avail_len}, future_len={future_len}")

    device = unnorm_selected_traj.device

    # unnorm_pred_future: (B_all, 1+Pnn, future_len, 4)
    unnorm_pred_future = unnorm_selected_traj[:, :, 1:, :]

    # GT가 아예 없는 에이전트는 recovery 대상에서 제외(논리적으로 더 안전)
    trigger = trigger & (gt_valid_count > 0)
    if not torch.any(trigger):
        return unnorm_selected_traj, gt_valid_mask

    # n_recovery: (B_all, 1+Pnn)
    n_recovery = _build_recovery_step_count_per_agent(
        args=args,
        gt_valid_step_count_per_agent=gt_valid_count,  # (B_all, 1+Pnn)
    )

    # lambda_: (B_all, 1+Pnn, future_len, 1)
    lambda_ = _build_recovery_lambda_per_agent(
        n_rec_per_agent=n_recovery,
        future_len=int(future_len),
        device=device,
    )

    # mixed_future: (B_all, 1+Pnn, future_len, 4)
    mixed_future = _mix_pred_and_gt_future_for_recovery(
        unnorm_pred_future=unnorm_pred_future,  # (B_all, 1+Pnn, future_len, 4)
        gt_future=gt_all,  # (B_all, 1+Pnn, future_len, 4)
        gt_valid_mask=gt_valid_mask,  # (B_all, 1+Pnn, future_len)
        lambda_=lambda_,  #  (B_all, 1+Pnn, future_len, 1)
    )

    # trigger 적용: (B_all, 1+Pnn, 1, 1)
    trigger_b = trigger[:, :, None, None]
    # out_future: (B_all, 1+Pnn, future_len, 4)
    out_future = torch.where(trigger_b, mixed_future, unnorm_pred_future)

    out = unnorm_selected_traj.clone()  # (B*R, 1+Pnn, 1+future_len, 4)
    out[:, :, 1:, :] = out_future

    # (cos, sin) 정리는 "0이 아닌 칸"에만
    future_pose = out[:, :, 1:, :]  # (B_all, 1+Pnn, T, 4)
    valid_pose = torch.any(future_pose != 0.0, dim=-1)  # (B_all, 1+Pnn, T)
    _normalize_heading_cos_sin_in_pose_4_dim_inplace(
        pose_4_dim=future_pose,
        valid_mask=valid_pose,
    )
    out[:, :, 1:, :] = future_pose

    return out, gt_valid_mask


def _build_generated_demo_futures_from_selected_traj(
    *,
    unnorm_selected_traj: torch.Tensor,  # (B*R, 1+Pnn, 1+future_len, 4)
) -> Tuple[torch.Tensor,
           torch.Tensor]:  # (B*R, future_len, 4) / (B*R, Pnn, future_len, 4)
    """선택(및 복구)된 경로로 저장용 미래 GT 텐서를 만듭니다.

    동작
    ----
    - 선택된 경로의 미래(1..future_len)를 저장용 GT로 사용합니다.
    - 단, (샘플, 에이전트)별로 GT가 실제로 존재하는 스텝 수만 남기고,
      그 뒤는 0으로 만듭니다.
      (rollout에서 step이 앞으로 갈수록 뒤쪽 GT가 0으로 사라지는 상황을 그대로 유지)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - demo_ego_future_4_dim:  (B_all, future_len, 4)
            - demo_near_future_4_dim: (B_all, Pnn, future_len, 4)
    """
    # selected_future_all: (B_all, 1+Pnn, future_len, 4)
    selected_future_all = unnorm_selected_traj[:, :, 1:, :]
    # demo_ego:  (B_all, future_len, 4)
    # demo_near: (B_all, Pnn, future_len, 4)  (Pnn이 0이면 (B_all,0,future_len,4))
    demo_ego = selected_future_all[:, 0, :, :]
    demo_near = selected_future_all[:, 1:, :, :]

    return demo_ego, demo_near


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
    unnorm_trajectory_np: np.ndarray,
) -> np.ndarray:
    """
    near_future_11: (Pnn, 1+T, 11)

    처음에 near_future_11 를 만들 때, near_agents_current 를 시간 축으로 복제해서 만들자.
    """
    one_fut = unnorm_trajectory_np.shape[1]
    near_future_11 = np.tile(near_agents_current[:, None, :],
                             (1, one_fut, 1))  # (Pnn, 1+T, 11)
    # np_int_traj_4_wrt_ego: (Pnn, 1+T, 4)
    np_int_traj_4_wrt_ego = unnorm_trajectory_np[1:]
    near_future_11[:, :, 0:4] = np_int_traj_4_wrt_ego  # (Pnn, 1+T, 11)
    return near_future_11

def _draw_one_batch_one_rollout(
        save_dir: str,
        a_unnorm_inputs_np: Dict[str, Any],
        a_unnorm_trajectory_np: np.ndarray,  # ((1+)Pnn, 1+T, 4)
        a_unnorm_ego_future_gt_4_dim: np.ndarray,  # (future_len, 4)
        a_unnorm_near_future_gt_3_dim: np.ndarray,  # (Pnn, future_len, 3)
        step_idx: int,
        draw_near_target_id: torch.Tensor,  # (Pnn,)
) -> None:
    """
        a_unnorm_near_future_gt_3_dim: torch.Tensor # (B, future_len, 4)
        near_future_gt_4_dim: torch.Tensor # (B, Pnn, future_len, 4)
    """

    output_data = {}
    """
    "ego_next_wp_wrt_ego" 
         (11,)
    "ego_np_int_traj_11_wrt_ego"
        (1+T, 11)

    "diff_token_to_np_int_traj_11_wrt_ego"
        (1 + future_len, 11)


    """
    ###############################
    ego_current = a_unnorm_inputs_np["ego_agent_past"][-1]  # (11)
    # ego_future_11 : (1+T, 11)
    ego_future_11 = _get_ego_future_11(ego_current, a_unnorm_trajectory_np)
    output_data["ego_np_int_traj_11_wrt_ego"] = ego_future_11
    output_data["ego_next_wp_wrt_ego"] = ego_future_11[1]
    ###############################
    ego_gt_future_11 = ego_future_11[1:, :].copy()
    ego_gt_future_11[:, 0:4] = a_unnorm_ego_future_gt_4_dim  # (future_len, 4)
    a_unnorm_inputs_np["ego_future_gt_11_dim"] = ego_gt_future_11  # (T, 11)
    ###############################
    near_agents_current = a_unnorm_inputs_np[
        "near_agents_past"][:, -1, :]  # (Pnn, 11)
    # near_future_11: (Pnn, 1+T, 11)
    near_future_11 = _get_near_future_11(near_agents_current,
                                         a_unnorm_trajectory_np)
    diff_token_to_np_int_traj_11_wrt_ego = {}
    for target_id, np_int_traj_11 in zip(draw_near_target_id, near_future_11):
        # np_int_traj_11: (1+T, 11)
        diff_token_to_np_int_traj_11_wrt_ego[f"{target_id}"] = np_int_traj_11
    output_data[
        "diff_token_to_np_int_traj_11_wrt_ego"] = diff_token_to_np_int_traj_11_wrt_ego
    ###############################
    diff_token_to_future_gt_3_dim = {}
    for target_id, gt_future_3 in zip(draw_near_target_id,
                                      a_unnorm_near_future_gt_3_dim):
        diff_token_to_future_gt_3_dim[
            f"{target_id}"] = gt_future_3  # (future_len, 3)
    a_unnorm_inputs_np[
        "diff_token_to_future_gt_3_dim"] = diff_token_to_future_gt_3_dim
    ###############################

    draw_machine_fast.draw_world_model_to_png(a_unnorm_inputs_np,
                                              output_data=output_data,
                                              save_path=os.path.join(
                                                  save_dir, f"{step_idx}.png"))


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
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    noise_std: float = 0.5,
) -> torch.Tensor:
    """현재 rollout 1개에서 사용할 무작위 텐서를 만듭니다.

    이 함수가 하는 일
    --------------
    - 모델이 같은 입력에서도 여러 후보 경로를 만들 수 있도록,
      (B, 1+Pnn, future_len, 4) 모양의 무작위 텐서를 만듭니다.
    - rollout을 1개씩 순차 실행하는 설정에서는 rollout이 "항상 1개"이므로,
      rollout_idx(몇 번째 rollout인지)만 알면 충분합니다.
    - step_idx(현재 rollout 안에서 몇 번째 시간인지)도 함께 섞어서,
      시간이 달라지면 무작위 텐서도 달라지게 합니다.

    같은 입력을 다시 만들 수 있는 조건
    ------------------------------
    - 아래 값들이 같으면 항상 같은 텐서가 나옵니다.
      (base_seed, ddp_rank, rollout_idx, step_idx)

    Args:
        reference_tensor (torch.Tensor):
            device/dtype를 맞추기 위한 기준 텐서입니다.
            shape: (B, ...)
        batch_size (int):
            배치 크기 B 입니다. shape: ()
        one_or_pnn (int):
            에이전트 수 (1+Pnn) 입니다. shape: ()
        future_len (int):
            미래 길이 입니다. shape: ()
        rollout_idx (int):
            지금 생성 중인 rollout 번호(0부터) 입니다. shape: ()
        base_seed (int):
            기본 seed 값입니다. 후보 K개를 만들 때는 후보마다 이 값이 달라집니다. shape: ()
        ddp_rank (int):
            분산 실행 시 프로세스 번호입니다. 싱글이면 0입니다. shape: ()
        step_idx (int):
            rollout 안에서의 시간 인덱스입니다(0부터). shape: ()
        noise_std (float):
            무작위 값의 크기를 조절합니다. shape: ()

    Returns:
        torch.Tensor:
            무작위 텐서.
            shape: (B, 1+Pnn, future_len, 4)
    """
    device = reference_tensor.device
    dtype = reference_tensor.dtype

    gen = torch.Generator(device=device)
    seed = _make_rollout_step_seed(
        base_seed=int(base_seed),
        ddp_rank=int(ddp_rank),
        rollout_idx=int(rollout_idx),
        step_idx=int(step_idx),
    )
    gen.manual_seed(int(seed))

    # noise: (B, 1+Pnn, future_len, 4)
    noise = torch.randn(
        (int(batch_size), int(one_or_pnn), int(future_len), 4),
        device=device,
        dtype=dtype,
        generator=gen,
    ) * float(noise_std)

    return noise


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


def _get_rollout_settings_for_validation(args: Any) -> Tuple[int, int, int]:
    """Validation에서 rollout(여러 샘플) 생성에 필요한 설정값을 모아줍니다.

    Args:
        args (Any):
            아래 속성이 있으면 사용합니다.
            - args.rollout_number (int): 전체 rollout 개수 R
            - args.seed (int): 기본 seed
            - args.ddp (bool): 분산 여부

    Returns:
        Tuple[int, int, int]:
            (rollout_number, base_seed, ddp_rank)
            - rollout_number: R
            - base_seed: 기본 seed
            - ddp_rank: 분산 rank (싱글이면 0)
    """
    rollout_number: int = int(getattr(args, "rollout_number", 3))
    rollout_number = int(max(1, rollout_number))

    base_seed: int = int(getattr(args, "seed", 0))
    ddp_rank: int = int(ddp.get_rank()) if bool(getattr(args, "ddp",
                                                        False)) else 0

    return rollout_number, base_seed, ddp_rank


def validate_func(
    args: Any,
    model: nn.Module,
    ema: Optional[ModelEma],
    norm_inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    state_normalizer: StateNormalizer,
    observation_normalizer: ObservationNormalizer,
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

    rollout_number, base_seed, ddp_rank = _get_rollout_settings_for_validation(
        args)

    _update_validation_heartbeat_stage(
        args,
        f"{tag} | predicting rollouts sequentially (rollout={rollout_number})")

    # ✅ rollout을 묶어서 처리하지 않고, 1개씩 순차 실행
    _predict_rollouts_sequential(
        args=args,
        model=inference_model,
        norm_inputs=norm_inputs,
        outputs=outputs,
        state_normalizer=state_normalizer,
        observation_normalizer=observation_normalizer,
        rollout_number=int(rollout_number),
        base_seed=int(base_seed),
        ddp_rank=int(ddp_rank),
    )


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
    gap = int(unnorm_ego_pose_chunk.shape[1])
    if gap <= 0:
        return unnorm_inputs_copy

    # -------------------------
    # 1) ego past 업데이트
    # -------------------------
    ego_agent_past = unnorm_inputs_copy.get("ego_agent_past", None)
    past_len = int(ego_agent_past.shape[1])
    if gap > past_len:
        raise ValueError(
            "gap이 past_len보다 큽니다. "
            "외부에서 rollout_time_chunk_size를 past_len 이하로 제한하는 것을 권장합니다. "
            f"gap={gap}, past_len={past_len}")
    # ego_current_11_dim: (B, 11)
    ego_current_11_dim = ego_agent_past[:, -1, :].clone()
    # ego_chunk_11: (B, gap, 11)
    ego_chunk_11 = ego_current_11_dim[:, None, :].expand(-1, gap, -1).clone()
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
    pnn = int(near_agents_past.shape[1])

    # near_last: (B, Pnn, 11)
    near_last = near_agents_past[:, :, -1, :].clone()
    # near_chunk_11: (B, Pnn, gap, 11)
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
        gap=gap,
        cached_valid_masks=cached_valid_masks,
    )

    return unnorm_inputs_copy


def _transform_origin(
    unnorm_inputs_copy: Dict[str, torch.Tensor],
    normed_ego_next_pose: torch.Tensor,  # (B*R, 4)
    gap: int,
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
        unnorm_inputs_copy (Dict[str, torch.Tensor]):
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
    if "ego_agent_past" in unnorm_inputs_copy:
        ego_agent_past = unnorm_inputs_copy["ego_agent_past"]
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
        if key not in unnorm_inputs_copy:
            continue
        agents_past = unnorm_inputs_copy[key]
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
        if future_key not in unnorm_inputs_copy:
            continue

        future_pose_4 = unnorm_inputs_copy[future_key]
        if (not isinstance(future_pose_4,
                           torch.Tensor)) or future_pose_4.numel() == 0:
            continue
        future_len = future_pose_4.shape[-2]

        # ✅ 겹치는 슬라이스 대입 방지: RHS에 clone()
        future_pose_4[..., :future_len -
                      gap, :] = future_pose_4[..., gap:, :].clone()
        future_pose_4[..., future_len - gap:, :] = 0.0

        valid_mask = _build_valid_mask_for_pose_4_dim(future_pose_4)
        _transform_pose_4_dim_inplace(
            pose_4_dim=future_pose_4,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    planner_future_11_dim = unnorm_inputs_copy.get("planner_future_11_dim",
                                                   None)
    if planner_future_11_dim is not None:
        future_len = planner_future_11_dim.shape[-2]

        # ✅ 여기 또한 겹치는 슬라이스 대입 방지: RHS에 clone()
        planner_future_11_dim[..., :future_len - gap, :] = \
        planner_future_11_dim[..., gap:, :].clone()
        planner_future_11_dim[..., future_len - gap:, :] = 0.0

        valid_mask = torch.any(planner_future_11_dim[..., :8] != 0.0, dim=-1)
        _transform_state_11_dim_inplace(
            state_11=planner_future_11_dim,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
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
        if point_key not in unnorm_inputs_copy or mask_key not in unnorm_inputs_copy:
            continue
        points_xy = unnorm_inputs_copy[point_key]  # (B, N, L, 2)
        points_valid = unnorm_inputs_copy[mask_key]  # (B, N, L) 또는 (B, N)
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
        if lane_key not in unnorm_inputs_copy:
            continue
        lane_12 = unnorm_inputs_copy[lane_key]
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
    if "static_objects" in unnorm_inputs_copy:
        static_objects = unnorm_inputs_copy["static_objects"]
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

    return unnorm_inputs_copy


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
