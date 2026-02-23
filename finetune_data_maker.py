import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import copy
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
from numpy.typing import NDArray
ArrayF = NDArray[np.floating]

_INFERENCE_NPZ_EXCLUDED_EXACT_KEYS: Set[str] = {
    "diff_token_to_future_gt_3_dim",
    "non_near_agents_past",
    "near_agents_past",
    "near_future_gt_4_dim",
    "ego_future_gt_4_dim",
    "target_future_valid",
    "scenario_id",
    "near_future_gt_3_dim",
}

try:
    import fcntl  # 리눅스/유닉스에서 파일 잠금에 사용
except ImportError:
    fcntl = None  # type: ignore

import copy
from typing import Any, Dict, Tuple

import numpy as np
import torch
# =========================
# finetune_data_maker.py 상단(import 아래) 어딘가에 추가
# =========================
from typing import Any
import os
import torch.nn as nn


def _force_disable_amortized_diffusion_for_finetune_data_maker(args: Any) -> None:
    """이 스크립트에서 use_amortized_diffusion을 항상 False로 강제합니다.

    이유:
        후보 K개를 한 번에 만들어(best 선택) 롤아웃을 진행하는 구조에서는,
        모델 내부에 남는 버퍼 때문에 다음 스텝에서 상태가 섞일 수 있습니다.
        그래서 이 스크립트에서는 amortized diffusion을 사용하지 않도록 고정합니다.

    Args:
        args (Any): argparse.Namespace 같은 설정 객체. shape: ()

    Returns:
        None
    """
    was_enabled = bool(getattr(args, "use_amortized_diffusion", False))
    setattr(args, "use_amortized_diffusion", False)

    # 로그는 rank 0만 찍기(DDP일 때 중복 방지)
    if was_enabled:
        rank = 0
        try:
            rank = int(os.environ.get("RANK", "0"))
        except Exception:
            rank = 0
        if rank == 0:
            print(
                "[finetune_data_maker] use_amortized_diffusion=True 설정이 감지되어 "
                "안전을 위해 False로 강제합니다."
            )


def _force_disable_amortized_diffusion_in_model(model: nn.Module) -> None:
    """모델 내부 설정에서도 use_amortized_diffusion을 False로 강제합니다.

    - args를 False로 고정해도, 체크포인트 로드/구성 경로에 따라
      모델 내부 config 값이 다시 True가 되는 상황을 방지합니다.
    - Decoder 내부 버퍼(_amortized_buffer)가 남아 있으면 None으로 비웁니다.

    Args:
        model (nn.Module): 추론에 사용할 모델. shape: ()

    Returns:
        None
    """
    # 1) 자주 쓰는 위치들에서 config를 찾아 use_amortized_diffusion을 끕니다.
    candidates = []

    cfg0 = getattr(model, "config", None)
    if cfg0 is not None:
        candidates.append(cfg0)

    dec = getattr(model, "decoder", None)
    if dec is not None:
        cfg1 = getattr(dec, "config", None)
        if cfg1 is not None:
            candidates.append(cfg1)

        # 2) Decoder 버퍼는 혹시 남아있으면 비움
        if hasattr(dec, "_amortized_buffer"):
            try:
                setattr(dec, "_amortized_buffer", None)
            except Exception:
                pass

    for cfg in candidates:
        if hasattr(cfg, "use_amortized_diffusion"):
            try:
                setattr(cfg, "use_amortized_diffusion", False)
            except Exception:
                pass

def _clone_nested_value_for_rollout(
    value: Any,
    memo: Dict[int, Any],
) -> Any:
    """rollout을 서로 독립적으로 만들기 위해 값을 안전하게 복사합니다.

    이 함수가 필요한 이유
    -------------------
    - rollout 안에서는 inputs/outputs 안의 텐서 값들이 '그 자리에서' 바뀝니다.
    - 같은 텐서 객체를 여러 rollout이 공유하면,
      1번째 rollout이 바꾼 값이 2번째 rollout 시작 상태가 되어 버립니다.
    - 그래서 rollout마다 "독립 사본"이 필요합니다.

    복사 규칙
    --------
    1) torch.Tensor:
       - value.detach().clone() 으로 새 텐서를 만듭니다.
       - shape: (B, ...) 처럼 어떤 모양이든 그대로 유지됩니다.
    2) np.ndarray:
       - value.copy() 로 새 배열을 만듭니다.
    3) dict / list / tuple / set:
       - 안쪽 원소도 같은 규칙으로 재귀적으로 복사합니다.
    4) 그 외(문자열, 숫자, None 등):
       - 보통 수정되지 않으므로 그대로 둡니다.
       - 만약 복사가 꼭 필요한 특수 객체면 deepcopy를 시도합니다.

    Args:
        value (Any): 복사할 값. shape: ()
        memo (Dict[int, Any]):
            이미 복사한 객체를 다시 복사하지 않기 위한 캐시.
            같은 객체가 여러 곳에서 공유되는 경우에도 일관되게 복사되도록 합니다. shape: ()

    Returns:
        Any: 복사된 값(또는 그대로의 값). shape: ()
    """
    obj_id = id(value)
    if obj_id in memo:
        return memo[obj_id]

    # 1) torch.Tensor
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        memo[obj_id] = cloned
        return cloned

    # 2) numpy array
    if isinstance(value, np.ndarray):
        cloned = value.copy()
        memo[obj_id] = cloned
        return cloned

    # 3) dict
    if isinstance(value, dict):
        out: Dict[Any, Any] = {}
        memo[obj_id] = out  # 재귀 중 자기참조를 막기 위해 먼저 등록
        for k, v in value.items():
            out[k] = _clone_nested_value_for_rollout(v, memo)
        return out

    # 4) list
    if isinstance(value, list):
        out_list: list = []
        memo[obj_id] = out_list
        for item in value:
            out_list.append(_clone_nested_value_for_rollout(item, memo))
        return out_list

    # 5) tuple
    if isinstance(value, tuple):
        out_tuple = tuple(
            _clone_nested_value_for_rollout(item, memo) for item in value)
        memo[obj_id] = out_tuple
        return out_tuple

    # 6) set
    if isinstance(value, set):
        out_set = set(
            _clone_nested_value_for_rollout(item, memo) for item in value)
        memo[obj_id] = out_set
        return out_set

    # 7) 나머지: deepcopy 시도(실패하면 그대로 사용)
    try:
        cloned = copy.deepcopy(value)
    except Exception:
        cloned = value

    memo[obj_id] = cloned
    return cloned


def _clone_inputs_outputs_for_independent_rollout(
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """각 rollout이 같은 시작점에서 출발하도록 inputs/outputs의 독립 사본을 만듭니다.

    핵심
    ----
    - inputs/outputs 안의 torch.Tensor들이 rollout 중에 바뀌므로,
      rollout마다 clone된 새 텐서를 써야 합니다.
    - memo를 inputs/outputs에 공통으로 써서,
      (혹시 같은 객체를 공유하던 값이 있다면) 복사 후에도 공유 관계가 유지되게 합니다.

    Args:
        inputs (Dict[str, Any]): 원본 입력 dict. shape: ()
        outputs (Dict[str, Any]): 원본 출력 dict. shape: ()

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            (inputs_copy, outputs_copy)
            - 둘 다 rollout 1회 실행에서만 쓰는 독립 사본입니다. shape: ()
    """
    memo: Dict[int, Any] = {}

    inputs_any = _clone_nested_value_for_rollout(inputs, memo)
    outputs_any = _clone_nested_value_for_rollout(outputs, memo)

    if not isinstance(inputs_any, dict):
        raise TypeError(f"inputs 복사 결과가 dict가 아닙니다. type={type(inputs_any)}")
    if not isinstance(outputs_any, dict):
        raise TypeError(f"outputs 복사 결과가 dict가 아닙니다. type={type(outputs_any)}")

    # typing 관점에서 Dict[str, Any]로 확정
    return dict(inputs_any), dict(outputs_any)


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
            "fine_tune_data_maker",
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
        # ✅ (추가) 혹시 중간에 값이 바뀌었어도 다시 한 번 강제
        _force_disable_amortized_diffusion_for_finetune_data_maker(args)
        _force_disable_amortized_diffusion_in_model(diffusion_planner)
        # EMA 모델도 실제 추론에 쓰이므로 같이 강제
        ema_model = getattr(model_ema, "ema",
                            None) if model_ema is not None else None
        if isinstance(ema_model, nn.Module):
            _force_disable_amortized_diffusion_in_model(ema_model)
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
            )

            validate_func(
                args=args,
                model=model,
                ema=ema,
                inputs=inputs,
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


def _to_numpy_for_draw(value: Any) -> Any:
    """draw 입력을 위해 torch.Tensor를 CPU numpy로 바꾸고, 내부까지 재귀 변환합니다.

    Args:
        value (Any):
            변환할 값.
            - torch.Tensor / list / dict / tuple 등이 들어올 수 있습니다.

    Returns:
        Any:
            torch.Tensor는 CPU numpy로 바뀌고,
            list/dict/tuple 안쪽도 같은 규칙으로 변환된 값.
    """
    if isinstance(value, torch.Tensor):
        # shape: (..,) 또는 ()
        return value.detach().to("cpu").numpy()

    if isinstance(value, np.ndarray):
        return value

    if isinstance(value, dict):
        return {k: _to_numpy_for_draw(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_to_numpy_for_draw(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_to_numpy_for_draw(v) for v in value)

    return value


def _torch_to_numpy(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """dict 안의 Tensor를 numpy로 바꾸되, list/dict 내부까지 재귀 변환합니다."""
    out: Dict[str, Any] = {}
    for k, v in inputs.items():
        out[k] = _to_numpy_for_draw(v)
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
    scenario_id: List[str],
    target_id: List[torch.Tensor],
    one_or_pnn: int,
    state: _RolloutVisualizationState,
) -> _RolloutVisualizationState:
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
        scenario_id=scenario_id,
        target_id=target_id,
        one_or_pnn=one_or_pnn,
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
    unnorm_outputs_copy: Dict[str, Any],
    unnorm_selected_gt_traj: torch.Tensor,
    step_idx: int,
) -> None:
    """조건이 맞으면 PNG 프레임 1장을 저장합니다.

    Args:
        state (_RolloutVisualizationState): 시각화 상태. shape: ()
        unnorm_inputs_copy (Dict[str, Any]): 현재 시점 모델 입력(정규화). shape: ()
        unnorm_selected_gt_traj (torch.Tensor):
            선택된 경로(정규화).
            shape: (B, 1+Pnn, 1+T, 4)
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
    (a_unnorm_inputs_np, a_unnorm_selected_gt_traj_np,
     a_unnorm_near_future_gt_3_dim, a_unnorm_ego_future_gt_4_dim
    ) = _prepare_data_for_one_batch_draw(
        unnorm_inputs_copy=unnorm_inputs_copy,
        unnorm_outputs_copy=unnorm_outputs_copy,
        unnorm_selected_gt_traj=unnorm_selected_gt_traj,  # (B, 1+Pnn, 1+T, 4)
        draw_batch_idx=int(state.draw_batch_idx),
    )

    _draw_one_batch_one_rollout(
        save_dir=str(state.save_dir),
        a_unnorm_inputs_np=a_unnorm_inputs_np,
        a_unnorm_selected_gt_traj_np=
        a_unnorm_selected_gt_traj_np,  # shape: ((1+)Pnn, 1+T, 4)
        a_unnorm_ego_future_gt_4_dim=
        a_unnorm_ego_future_gt_4_dim,  # shape: (future_len, 4)
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
    scenario_id: List[str],
    target_id: List[torch.Tensor],
    one_or_pnn: int,
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
    draw_scenario_id = scenario_id[draw_batch_idx]  # str
    if target_id is not None:
        draw_target_id = target_id[draw_batch_idx]  # ((1+)Pnn)
    else:
        # manually set draw_target_id as 0, 1, ..., Pnn
        draw_target_id = torch.arange(0, one_or_pnn)  # ((1+)Pnn)

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
    (B) 배치 안에서 scenario_id가 반복됩니다.
    step_count만 파일명에 넣으면 같은 이름이 여러 번 만들어져서 덮어쓰게 됩니다.

    그래서 파일명에 (B) 배치 인덱스(sample_idx)를 함께 넣어서
    "배치/rollout별로 파일이 각각 남도록" 합니다.

    Args:
        scenario_id (str): 시나리오 id 문자열. shape: ()
        step_count (int): rollout 루프 저장 step 번호. shape: ()
        sample_idx (int): (B) 배치에서의 샘플 인덱스. shape: ()

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
    unnorm_inputs: Dict[str, Any],
    batch_size: int,
) -> Dict[str, Any]:
    """unnorm_inputs_copy를 "저장하기 좋은 CPU 자료"로 한 번만 변환해 캐시로 만듭니다.

    핵심 아이디어
    ------------
    - 여기서는 키별로 1번만 CPU numpy로 바꿔두고,
      이후에는 루프에서 인덱싱만 합니다.

    Args:
        unnorm_inputs (Dict[str, Any]):
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

    for key, value in unnorm_inputs.items():
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
    unnorm_data_np: Dict[str, Any],
    sample_idx: int,
    batch_size: int,
) -> Dict[str, Any]:
    """CPU 캐시에서 sample_idx에 해당하는 샘플 1개만 뽑아 저장 dict를 만듭니다.

    Args:
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

    for key, value in unnorm_data_np.items():
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


def _format_id_for_draw_key(token_id: Any) -> str:
    """dict key로 쓰기 좋은 'id 문자열'을 만듭니다.

    목표
    ----
    - torch.Tensor 스칼라(예: tensor(17, device='cuda:0'))가 들어와도
      key는 항상 "17" 같은 안정적인 문자열이 되게 합니다.

    Args:
        token_id (Any):
            id 값.
            - 보통 torch.Tensor 스칼라 또는 int/np 스칼라가 들어옵니다.
            shape: () 또는 (1,)

    Returns:
        str:
            key 문자열. 예: "17"
            shape: ()
    """
    if isinstance(token_id, torch.Tensor):
        # shape: () 또는 (1,)
        if int(token_id.numel()) == 1:
            return str(int(token_id.detach().to("cpu").item()))
        # 혹시 여러 개면 "1_2_3" 형태로(안전장치)
        flat = token_id.detach().to("cpu").reshape(-1).tolist()
        return "_".join(str(int(x)) for x in flat)

    if isinstance(token_id, np.ndarray):
        if int(token_id.size) == 1:
            return str(int(np.asarray(token_id).item()))
        flat = np.asarray(token_id).reshape(-1).tolist()
        return "_".join(str(int(x)) for x in flat)

    if isinstance(token_id, (np.integer, int)):
        return str(int(token_id))

    return str(token_id)


def _remove_invalid_data(
    npz_payload_dict: Dict[str, Any],
    neighbor_agents_is_valid: Optional[np.ndarray],  # (chosen_agent_num,)
    lanes_is_valid: Optional[np.ndarray],  # (lane_num,)
    route_lanes_is_valid: Optional[np.ndarray],  # (lane_num,)
    stop_sign_is_valid: Optional[np.ndarray],  # (stop_sign_num,)
    speed_bump_is_valid: Optional[np.ndarray],
    crosswalk_is_valid: Optional[np.ndarray],
    driveway_is_valid: Optional[np.ndarray],
    road_edge_is_valid: Optional[np.ndarray],
    static_objects_is_valid: Optional[np.ndarray],
) -> Dict[str, Any]:
    neighbor_agents_past = npz_payload_dict.get("neighbor_agents_past", None)
    if neighbor_agents_past is not None:
        assert neighbor_agents_is_valid is not None, (
            "neighbor_agents_is_valid가 None인데 neighbor_agents_past가 존재합니다."
        )
        neighbor_agents_is_valid = neighbor_agents_is_valid.astype(bool)
        npz_payload_dict["neighbor_agents_past"] = npz_payload_dict[
            "neighbor_agents_past"
        ][neighbor_agents_is_valid]  # (valid_chosen_agent_num, time_len, 11)
        npz_payload_dict["neighbor_future_gt_3_dim"] = npz_payload_dict[
            "neighbor_future_gt_3_dim"
        ][neighbor_agents_is_valid]  # (valid_chosen_agent_num, future_len, 3)
        npz_payload_dict["neighbor_future_gt_11_dim"] = npz_payload_dict[
            "neighbor_future_gt_11_dim"
        ][neighbor_agents_is_valid]  # (valid_chosen_agent_num, future_len, 11)

    # ------------------------------------------------------------
    # ✅ (추가) future_seg_control_gt_3_dim: (1+Pnn, future_len, 3)
    # - ego(0번)는 항상 유지(True)
    # - neighbor(1..Pnn)는 neighbor_agents_is_valid 로 필터링
    # ✅ (추가) past_seg_control_gt_3_dim 도 동일한 full_mask로 같이 필터링
    # ------------------------------------------------------------
    future_seg_control_gt_3_dim = npz_payload_dict.get("future_seg_control_gt_3_dim", None)
    if future_seg_control_gt_3_dim is not None:
        if not isinstance(future_seg_control_gt_3_dim, np.ndarray):
            raise TypeError(
                "future_seg_control_gt_3_dim은 np.ndarray여야 합니다. "
                f"type={type(future_seg_control_gt_3_dim)}"
            )
        if future_seg_control_gt_3_dim.ndim != 3 or int(future_seg_control_gt_3_dim.shape[-1]) != 3:
            raise ValueError(
                "future_seg_control_gt_3_dim은 (1+Pnn, future_len, 3)이어야 합니다. "
                f"got shape={tuple(future_seg_control_gt_3_dim.shape)}"
            )

        agent_count = int(future_seg_control_gt_3_dim.shape[0])  # 1+Pnn
        pnn = int(max(0, agent_count - 1))

        if pnn == 0:
            # ego만 있는 경우: 그대로 둠
            npz_payload_dict["future_seg_control_gt_3_dim"] = future_seg_control_gt_3_dim
        else:
            if neighbor_agents_is_valid is None:
                raise ValueError(
                    "future_seg_control_gt_3_dim에 neighbor가 있는데 "
                    "neighbor_agents_is_valid가 None입니다."
                )

            neighbor_mask = np.asarray(neighbor_agents_is_valid).astype(bool)
            if neighbor_mask.ndim != 1:
                neighbor_mask = neighbor_mask.reshape(-1)

            if int(neighbor_mask.shape[0]) != int(pnn):
                raise ValueError(
                    "neighbor_agents_is_valid 길이와 future_seg_control_gt_3_dim의 Pnn이 다릅니다. "
                    f"len(mask)={int(neighbor_mask.shape[0])}, Pnn={int(pnn)}, "
                    f"control_shape={tuple(future_seg_control_gt_3_dim.shape)}"
                )

            # ego는 항상 True
            full_mask = np.concatenate([np.array([True], dtype=bool), neighbor_mask], axis=0)  # (1+Pnn,)

            # (1+valid_neighbor_num, future_len, 3)
            npz_payload_dict["future_seg_control_gt_3_dim"] = future_seg_control_gt_3_dim[full_mask]

            # ------------------------------------------------------------
            # ✅ (핵심) past_seg_control_gt_3_dim도 같은 full_mask로 필터링
            # 기대 shape: (1+Pnn, past_len, 3)
            # ------------------------------------------------------------
            past_seg_control_gt_3_dim = npz_payload_dict.get("past_seg_control_gt_3_dim", None)
            if past_seg_control_gt_3_dim is not None:
                if not isinstance(past_seg_control_gt_3_dim, np.ndarray):
                    raise TypeError(
                        "past_seg_control_gt_3_dim은 np.ndarray여야 합니다. "
                        f"type={type(past_seg_control_gt_3_dim)}"
                    )
                if past_seg_control_gt_3_dim.ndim != 3 or int(past_seg_control_gt_3_dim.shape[-1]) != 3:
                    raise ValueError(
                        "past_seg_control_gt_3_dim은 (1+Pnn, past_len, 3)이어야 합니다. "
                        f"got shape={tuple(past_seg_control_gt_3_dim.shape)}"
                    )
                if int(past_seg_control_gt_3_dim.shape[0]) != int(agent_count):
                    raise ValueError(
                        "past_seg_control_gt_3_dim의 agent 축(첫 차원)이 "
                        "future_seg_control_gt_3_dim과 일치해야 합니다. "
                        f"past_agent_count={int(past_seg_control_gt_3_dim.shape[0])}, "
                        f"future_agent_count={int(agent_count)}"
                    )

                # (1+valid_neighbor_num, past_len, 3)
                npz_payload_dict["past_seg_control_gt_3_dim"] = past_seg_control_gt_3_dim[full_mask]

    # --- 이하 기존 코드 그대로 ---
    lanes = npz_payload_dict["lanes"]  # (chosen_lane_num, lane_len, 12)
    if lanes is not None:
        assert lanes_is_valid is not None, (
            "lanes_is_valid가 None인데 lanes가 존재합니다."
        )
        lanes_is_valid = lanes_is_valid.astype(bool)
        npz_payload_dict["lanes"] = lanes[lanes_is_valid]
        npz_payload_dict["lanes_speed_limit"] = npz_payload_dict["lanes_speed_limit"][lanes_is_valid]
        npz_payload_dict["lanes_has_speed_limit"] = npz_payload_dict["lanes_has_speed_limit"][lanes_is_valid]

        lane_type = npz_payload_dict.get("lane_type", None)
        left_line_type = npz_payload_dict.get("left_line_type", None)
        right_line_type = npz_payload_dict.get("right_line_type", None)
        if lane_type is not None:
            npz_payload_dict["lane_type"] = lane_type[lanes_is_valid]
        if left_line_type is not None:
            npz_payload_dict["left_line_type"] = npz_payload_dict["left_line_type"][lanes_is_valid]
        if right_line_type is not None:
            npz_payload_dict["right_line_type"] = npz_payload_dict["right_line_type"][lanes_is_valid]

    agent_route_lane_order = npz_payload_dict.get("agent_route_lane_order", None)
    if agent_route_lane_order is not None and neighbor_agents_past is not None and lanes is not None:
        npz_payload_dict["agent_route_lane_order"] = agent_route_lane_order[
            neighbor_agents_is_valid
        ][:, lanes_is_valid]

    route_lanes = npz_payload_dict.get("route_lanes", None)
    if route_lanes is not None:
        assert route_lanes_is_valid is not None, (
            "route_lanes_is_valid가 None인데 route_lanes가 존재합니다."
        )
        route_lanes_is_valid = route_lanes_is_valid.astype(bool)
        npz_payload_dict["route_lanes"] = route_lanes[route_lanes_is_valid]
        npz_payload_dict["route_lanes_speed_limit"] = npz_payload_dict["route_lanes_speed_limit"][route_lanes_is_valid]
        npz_payload_dict["route_lanes_has_speed_limit"] = npz_payload_dict["route_lanes_has_speed_limit"][route_lanes_is_valid]

    stop_sign_points = npz_payload_dict.get("stop_sign_points", None)
    if stop_sign_points is not None:
        assert stop_sign_is_valid is not None, (
            "stop_sign_points_is_valid가 None인데 stop_sign_points가 존재합니다."
        )
        stop_sign_is_valid = stop_sign_is_valid.astype(bool)
        npz_payload_dict["stop_sign_points"] = stop_sign_points[stop_sign_is_valid]

    speed_bump_points = npz_payload_dict.get("speed_bump_points", None)
    if speed_bump_points is not None:
        assert speed_bump_is_valid is not None, (
            "speed_bump_points_is_valid가 None인데 speed_bump_points가 존재합니다."
        )
        speed_bump_is_valid = speed_bump_is_valid.astype(bool)
        npz_payload_dict["speed_bump_points"] = speed_bump_points[speed_bump_is_valid]

    crosswalk_points = npz_payload_dict.get("crosswalk_points", None)
    if crosswalk_points is not None:
        assert crosswalk_is_valid is not None, (
            "crosswalk_points_is_valid가 None인데 crosswalk_points가 존재합니다."
        )
        crosswalk_is_valid = crosswalk_is_valid.astype(bool)
        npz_payload_dict["crosswalk_points"] = crosswalk_points[crosswalk_is_valid]

    driveway_points = npz_payload_dict.get("driveway_points", None)
    if driveway_points is not None:
        assert driveway_is_valid is not None, (
            "driveway_points_is_valid가 None인데 driveway_points가 존재합니다."
        )
        driveway_is_valid = driveway_is_valid.astype(bool)
        npz_payload_dict["driveway_points"] = driveway_points[driveway_is_valid]

    road_edge = npz_payload_dict.get("road_edge", None)
    if road_edge is not None:
        assert road_edge_is_valid is not None, (
            "road_edge_is_valid가 None인데 road_edge가 존재합니다."
        )
        road_edge_is_valid = road_edge_is_valid.astype(bool)
        npz_payload_dict["road_edge"] = road_edge[road_edge_is_valid]
        npz_payload_dict["road_edge_type"] = npz_payload_dict["road_edge_type"][road_edge_is_valid]

    static_objects = npz_payload_dict.get("static_objects", None)
    if static_objects is not None:
        assert static_objects_is_valid is not None, (
            "static_objects_is_valid가 None인데 static_objects가 존재합니다."
        )
        static_objects_is_valid = static_objects_is_valid.astype(bool)
        npz_payload_dict["static_objects"] = static_objects[static_objects_is_valid]

    return npz_payload_dict



def _apply_unvalid_at_unnorm_selected_traj_raw(
        unnorm_selected_traj_raw: torch.Tensor,  # (B, 1+Pnn, 1+future_len, 4)
        target_cur_fut_gt_is_valid: torch.Tensor,  # (B, 1+Pnn, 1+future_len)
) -> torch.Tensor:  # (B, 1+Pnn, 1+future_len, 4)
    # 중요: unnorm_selected_traj 에서, GT가 없는 구간은 0. 으로 채워집니다.
    unnorm_selected_traj = unnorm_selected_traj_raw * target_cur_fut_gt_is_valid.unsqueeze(
        -1).to(
            dtype=unnorm_selected_traj_raw.dtype)  # (B, 1+Pnn, 1+future_len, 4)
    return unnorm_selected_traj


from typing import Any


def _build_target_cur_future_is_valid(
    *,
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_outputs_copy: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """rollout step에서 target 유효 마스크들을 만듭니다.

    하는 일
    ------
    2) 현재 시점에서 ego/near가 존재하는지(유효한지) 마스크를 만듭니다.
    3) (현재 1칸 + 미래 future_len칸) 전체 유효 마스크를 합쳐 만듭니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            필요한 키/shape
            - ego_agent_past_is_valid: (B, past_len)  bool 또는 0/1
            - near_agents_is_valid: (B, Pnn)  bool 또는 0/1

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - target_cur_fut_gt_is_valid: (B, 1+Pnn, 1+future_len) bool
    """
    ego_future_gt_is_valid = unnorm_outputs_copy["ego_future_gt_is_valid"]
    ego_future_gt_is_valid = ego_future_gt_is_valid.to(dtype=torch.bool)

    near_future_gt_is_valid = unnorm_outputs_copy["near_future_gt_is_valid"]
    near_future_gt_is_valid = near_future_gt_is_valid.to(dtype=torch.bool)

    target_future_gt_is_valid = torch.cat(
        [ego_future_gt_is_valid.unsqueeze(1), near_future_gt_is_valid],
        dim=1,
    )  # (B, 1+Pnn, future_len)

    ego_agent_past_is_valid = unnorm_inputs_copy["ego_agent_past_is_valid"]
    ego_agent_past_is_valid = ego_agent_past_is_valid.to(dtype=torch.bool)

    near_agents_is_valid = unnorm_inputs_copy["near_agents_is_valid"]
    near_agents_is_valid = near_agents_is_valid.to(dtype=torch.bool)

    # ego_agent_current_is_valid: (B,)
    ego_agent_current_is_valid = ego_agent_past_is_valid[:, -1]

    target_current_is_valid = torch.cat(
        [ego_agent_current_is_valid.unsqueeze(1), near_agents_is_valid],
        dim=1,
    )  # (B, 1+Pnn)

    target_cur_fut_gt_is_valid = torch.cat(
        [target_current_is_valid.unsqueeze(2), target_future_gt_is_valid],
        dim=2,
    )  # (B, 1+Pnn, 1+future_len)

    return target_current_is_valid, target_future_gt_is_valid, target_cur_fut_gt_is_valid


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


def _convert_target_chunk_to_world(
    unnorm_target_pose_chunk: torch.Tensor,
    gap: int,
    unnorm_origin_pose_world: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """원래 단위 포즈 chunk를 world 좌표로 바꿔 저장하고, 다음 origin_world_pose를 계산합니다.

    하는 일
    ------
    - unnorm_origin_pose_world가 None이면 아무 것도 하지 않고 None을 반환합니다.
    - None이 아니면:
      1) (B*R, (1+)Pnn, gap, 4) 포즈를 펼쳐서 world 좌표로 변환합니다.


    Returns:
        Optional[torch.Tensor]:
            갱신된 origin_world_pose.
            - 성공 시 shape: (B*R, (1+)Pnn, gap, 4)
            - origin이 None이면 None
    """
    merged_batch = int(unnorm_target_pose_chunk.shape[0])  # B*R
    one_or_pnn = int(unnorm_target_pose_chunk.shape[1])  # 1+Pnn

    if unnorm_origin_pose_world is None:
        return None

    if int(gap) <= 0:
        raise ValueError(f"gap은 1 이상이어야 합니다. gap={gap}")
    # (B*R*(1+)Pnn, gap, 4)
    unnorm_target_pose_chunk_flat = unnorm_target_pose_chunk.reshape(
        -1, int(gap), 4)

    # world 변환: (B*R*(1+)Pnn, gap, 4)
    unnorm_target_pose_chunk_world_flat = _covert_from_ego_to_world(
        target_poses=unnorm_target_pose_chunk_flat,
        origin_world_pose=unnorm_origin_pose_world,  # (B*R, 4)
    )

    # (B*R, (1+)Pnn, gap, 4)
    unnorm_target_pose_chunk_world = unnorm_target_pose_chunk_world_flat.reshape(
        int(merged_batch), int(one_or_pnn), int(gap), 4)

    return unnorm_target_pose_chunk_world


def _predict_one_rollout_sequential(
    args: Any,
    model: nn.Module,
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_outputs_copy: Dict[str, Any],
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

    Returns:
        None
    """
    # ✅ 시각화 상태
    vis_state: _RolloutVisualizationState = _init_rollout_visualization_state(
        save_image=bool(save_image),
        save_video=bool(save_video),
        draw_batch_idx=int(draw_batch_idx),
    )

    future_len = args.future_len

    # ✅ 추가: 이번 롤아웃에서 "최대 몇 칸까지만" 진행할지
    scenario_finish_step = args.scenario_finish_step
    scenario_id = unnorm_inputs_copy["scenario_id"]  # length B list of str
    target_id = unnorm_inputs_copy.get(
        "target_id", None)  # length (1+Pnn) list of torch.Tensor

    # ✅ GT 미래를 norm_inputs에 1번만 넣어 둡니다.
    cached_valid_masks: Dict[
        str, torch.Tensor] = _build_cached_valid_masks_for_static_map_features(
            unnorm_inputs_copy=unnorm_inputs_copy)

    # unnorm_agent_length_m/unnorm_agent_width_m: (B, 1+Pnn)
    (unnorm_agent_length_m,
     unnorm_agent_width_m) = _extract_agent_box_size_m_from_past_states(
         ego_agent_past=unnorm_inputs_copy["ego_agent_past"],  # (B, T_past, 11)
         ego_agent_past_is_valid=unnorm_inputs_copy[
             "ego_agent_past_is_valid"],  # (B, T_past)
         near_agents_past=unnorm_inputs_copy[
             "near_agents_past"],  # (B, Pnn, T_past, 11)
         near_agents_past_is_valid=unnorm_inputs_copy[
             "near_agents_past_is_valid"],  # (B, Pnn, T_past)
     )
    time_chunk_size = args.rollout_time_chunk_size
    with torch.inference_mode():
        step_count = 0
        step_start = 0
        while step_start < int(scenario_finish_step):
            remaining = int(scenario_finish_step - step_start)
            gap = int(min(time_chunk_size, remaining))

            # norm_inputs_step: 현재 step에서 모델에 넣을 정규화 입력
            (norm_inputs_step,
             norm_outputs_step) = _build_norm_inputs_from_unnorm_inputs(
                 unnorm_inputs_copy=unnorm_inputs_copy,
                 unnorm_outputs_copy=unnorm_outputs_copy,
                 state_normalizer=state_normalizer,
                 observation_normalizer=observation_normalizer,
             )
            # 후보 K개 중 best 선택
            """
            - best_normed_traj: (B, 1+Pnn, 1+T, 4)
            - best_distance_m_per_agent: (B, 1+Pnn)
            - best_control_seq
              - pose_based=False => (B, 1+Pnn, future_len, 3)
              - pose_based=True => None
            """
            (best_normed_traj,
             best_dist_m,
             best_control_seq) = _select_best_trajectory_by_sample_k(
                args=args,
                model=model,
                norm_inputs_step=norm_inputs_step,
                state_normalizer=state_normalizer,
                unnorm_outputs_copy=unnorm_outputs_copy,
                unnorm_agent_length_m=unnorm_agent_length_m,
                unnorm_agent_width_m=unnorm_agent_width_m,
                batch_size=int(batch_size),
                one_or_pnn=int(one_or_pnn),
                future_len=int(future_len),
                rollout_idx=int(rollout_idx),
                base_seed=int(base_seed),
                ddp_rank=int(ddp_rank),
                step_idx=int(step_start),
                gap=gap,
            )

            # (B, 1+Pnn, future_len), (B, 1+Pnn, 1+future_len)
            (target_current_is_valid, target_future_gt_is_valid,
             target_cur_fut_gt_is_valid) = _build_target_cur_future_is_valid(
                 unnorm_inputs_copy=unnorm_inputs_copy,
                 unnorm_outputs_copy=unnorm_outputs_copy,
            )
            # target_current_is_valid: (B, 1+Pnn) -> (B, 1+Pnn, 1) -> (B, 1+Pnn, 1+future_len)
            target_current_is_valid_expand = target_current_is_valid.unsqueeze(
                2).expand(-1, -1, 1 + int(future_len))

            unnorm_best_traj = state_normalizer.inverse(
                data=best_normed_traj,  # (B, 1+Pnn, 1+future_len, 4)
                valid_mask=
                target_current_is_valid_expand,  # (B, 1+Pnn, 1+future_len)
            )
            if best_control_seq is None:
                unnorm_target_control_chunk = None
            else:
                # best_control_seq : (B, 1+Pnn, future_len, 3)
                unnorm_best_control_seq = state_normalizer.inverse(
                    data=best_control_seq,
                    valid_mask=target_current_is_valid_expand[:, :, 1:], # (B, 1+Pnn, future_len)
                )
                # unnorm_target_control_chunk: (B, 1+Pnn, gap, 3)
                unnorm_target_control_chunk = unnorm_best_control_seq[:, :, :gap, :]

            # unnorm_selected_traj_raw: (B, 1+Pnn, 1+future_len, 4)
            unnorm_selected_traj_raw = _apply_recovery_if_needed(
                args=args,
                unnorm_best_traj=unnorm_best_traj,  # (B, 1+Pnn, 1+future_len, 4)
                expert_distance_m=best_dist_m,  # (B, 1+Pnn)
                unnorm_outputs_copy=unnorm_outputs_copy,
                target_future_gt_is_valid=
                target_future_gt_is_valid,  # (B, 1+Pnn, future_len)
                target_current_is_valid_expand=
                target_current_is_valid_expand,  # (B, 1+Pnn, 1+future_len)
                future_len=int(future_len),
            )
            # (B, 1+Pnn, 1+future_len, 4)
            unnorm_selected_gt_traj = _apply_unvalid_at_unnorm_selected_traj_raw(
                unnorm_selected_traj_raw, target_cur_fut_gt_is_valid)

            # ✅ (그림/영상) 준비는 1번만
            vis_state = _maybe_prepare_rollout_visualization_once(
                args=args,
                scenario_id=scenario_id,
                target_id=target_id,
                one_or_pnn=one_or_pnn,
                state=vis_state,  # _RolloutVisualizationState
            )

            # ✅ (그림) 프레임 저장
            _maybe_draw_rollout_visualization_frame(
                state=vis_state,
                unnorm_inputs_copy=unnorm_inputs_copy,
                unnorm_outputs_copy=unnorm_outputs_copy,
                unnorm_selected_gt_traj=
                unnorm_selected_gt_traj,  # (B, 1+Pnn, 1+T, 4)
                step_idx=int(step_start),
            )
            # ✅ npz 저장 (execute 전)
            step_count_for_save = int(step_count) + 1
            if args.save_inference_data:
                unnorm_inputs_for_save = dict(unnorm_inputs_copy)
                unnorm_outputs_for_save = dict(unnorm_outputs_copy)
                # ego_future_gt_4_dim: (B, future_len, 4)
                # near_future_gt_4_dim: (B, Pnn, future_len, 4)
                # future_seg_control_gt_3_dim: (B, (1+)Pnn, future_len, 3)
                (ego_future_gt_4_dim, near_future_gt_4_dim,
                 future_seg_control_gt_3_dim
                 ) = _build_generated_demo_futures_from_selected_traj(
                    unnorm_selected_gt_traj=unnorm_selected_gt_traj,
                    # (B, 1+Pnn, 1+future_len, 4)
                    target_cur_fut_gt_is_valid=target_cur_fut_gt_is_valid,
                    # (B, 1+Pnn, 1+future_len)
                    dt=float(0.1),
                )
                unnorm_outputs_for_save[
                    "ego_future_gt_4_dim"] = ego_future_gt_4_dim  # (B, future_len, 4)
                unnorm_outputs_for_save[
                    "near_future_gt_4_dim"] = near_future_gt_4_dim  # (B, Pnn, future_len, 4)

                # ✅ (추가) pose_based=False용 학습 데이터 저장
                # shape: (B, 1+Pnn, future_len, 3)
                unnorm_outputs_for_save[
                    "future_seg_control_gt_3_dim"] = future_seg_control_gt_3_dim
                _save_inference_data(
                    args.save_cache_path,
                    unnorm_inputs_for_save,
                    unnorm_outputs_for_save,
                    step_count=int(step_count_for_save),
                    sample_idx_offset=int(
                        sample_idx_offset
                    ),  # npz 파일명 충돌 방지용 오프셋(보통 rollout_idx * B).
                )

            # ✅ execute: 앞 gap 스텝 반영
            # unnorm_best_traj: (B, 1+Pnn, 1+future_len, 4)
            # unnorm_target_pose_chunk: (B, 1+Pnn, gap, 4)
            unnorm_target_pose_chunk = unnorm_best_traj[:, :, 1:gap + 1, :]

            unnorm_origin_pose_world = unnorm_inputs_copy[
                "origin_world_pose"]  # (B, 4) (x,y,cos,sin)
            # unnorm_origin_pose_world: (B, (1+)Pnn, gap, 4)
            unnorm_target_pose_chunk_world = _convert_target_chunk_to_world(
                unnorm_target_pose_chunk=unnorm_target_pose_chunk,
                gap=int(gap),
                unnorm_origin_pose_world=unnorm_origin_pose_world,
            )
            # (B, 4)
            unnorm_inputs_copy[
                "origin_world_pose"] = unnorm_target_pose_chunk_world[:, 0,
                                                                      -1, :]
            unnorm_ego_pose_chunk = unnorm_target_pose_chunk[:,
                                                             0, :, :]  # (B, gap, 4)
            unnorm_near_pose_chunk = unnorm_target_pose_chunk[:,
                                                              1:, :, :]  # (B, Pnn, gap, 4)
            (unnorm_inputs_copy, unnorm_outputs_copy
            ) = _update_merged_inputs_unnorm_inplace_for_time_chunk(
                unnorm_inputs_copy=unnorm_inputs_copy,
                unnorm_outputs_copy=unnorm_outputs_copy,
                unnorm_ego_pose_chunk=unnorm_ego_pose_chunk,  # (B, gap, 4)
                unnorm_near_pose_chunk=unnorm_near_pose_chunk,  # (B, Pnn, gap, 4)
                unnorm_target_control_chunk=unnorm_target_control_chunk,
                # Optional[(B, (1+)Pnn, gap, 3)]
                cached_valid_masks=cached_valid_masks,
            )

            step_start += int(gap)
            step_count += 1

    # ✅ (영상) PNG -> 영상 생성
    _finalize_rollout_visualization_video_if_needed(
        args=args,
        state=vis_state,
    )

def _augment_inputs_with_future_seg_control_for_npz(
    a_inputs_dict: Dict[str, Any],
    a_outputs_dict: Dict[str, Any],
) -> None:
    """future_seg_control_gt_3_dim을 npz 저장 대상(dict)에 포함시킵니다.

    - rollout에서 만든 future_seg_control_gt_3_dim은 보통 outputs 쪽에 들어옵니다.
    - 하지만 저장 payload는 a_inputs_dict 기준으로 만들어지므로,
      outputs에 있는 값을 inputs로 옮겨 담습니다.

    기대 shape
    - future_seg_control_gt_3_dim: (1+Pnn, future_len, 3)

    Args:
        a_inputs_dict (Dict[str, Any]): 샘플 1개 입력 dict. shape: ()
        a_outputs_dict (Dict[str, Any]): 샘플 1개 출력 dict. shape: ()

    Returns:
        None
    """
    ctrl = a_outputs_dict.get("future_seg_control_gt_3_dim", None)
    if ctrl is None:
        return

    if not isinstance(ctrl, np.ndarray):
        raise TypeError(
            "future_seg_control_gt_3_dim은 np.ndarray여야 합니다. "
            f"type={type(ctrl)}"
        )
    if ctrl.ndim != 3 or int(ctrl.shape[-1]) != 3:
        raise ValueError(
            "future_seg_control_gt_3_dim은 (1+Pnn, future_len, 3)이어야 합니다. "
            f"got shape={tuple(ctrl.shape)}"
        )

    a_inputs_dict["future_seg_control_gt_3_dim"] = ctrl


from typing import Any, Dict
import torch
import torch.nn as nn


def _predict_rollouts_sequential(
    args: Any,
    model: nn.Module,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
    rollout_number: int,
    base_seed: int,
    ddp_rank: int,
) -> None:
    """rollout_number 만큼 rollout을 1개씩(for문) 순차 실행합니다.

    변경된 핵심
    ----------
    - 매 rollout 시작마다 inputs/outputs를 '안쪽까지' 새로 복사해서,
      rollout끼리 상태가 이어지지 않게 합니다.
    - 즉, rollout_number를 늘리면
      "같은 시작점에서 여러 후보"가 만들어집니다.

    Args:
        args (Any): 설정 객체. shape: ()
        model (nn.Module): 예측 모델. shape: ()
        inputs (Dict[str, Any]): 입력 dict. (B, ...) 형태 텐서들을 포함. shape: ()
        outputs (Dict[str, Any]): 출력/정답 dict. (B, ...) 형태 텐서들을 포함. shape: ()
        state_normalizer (StateNormalizer): 포즈 변환 도구. shape: ()
        observation_normalizer (ObservationNormalizer): 입력 변환 도구. shape: ()
        rollout_number (int): rollout 개수 R. shape: ()
        base_seed (int): 기본 seed. shape: ()
        ddp_rank (int): 분산 rank. shape: ()

    Returns:
        None
    """
    rollout_number_i = int(max(1, int(rollout_number)))

    target_future_valid = inputs.get("target_future_valid", None)
    if not isinstance(target_future_valid, torch.Tensor):
        raise KeyError("inputs에 'target_future_valid'(torch.Tensor)가 필요합니다.")

    batch_size = int(target_future_valid.shape[0])  # B
    one_or_pnn = int(target_future_valid.shape[1])  # (1+Pnn)

    draw_batch_idx = int(getattr(args, "draw_batch_idx", 0))

    for r in range(rollout_number_i):
        # ✅ 이미지/영상은 파일명 충돌 위험이 있어서 "첫 rollout만" 켜는 게 안전합니다.
        enable_image = bool(getattr(args, "save_image", False)) and int(r) == 0
        enable_video = bool(getattr(args, "save_video", False)) and int(r) == 0

        # ✅ npz 파일명 충돌 방지용
        sample_idx_offset = int(r) * int(batch_size)

        # ✅ (핵심) rollout마다 inputs/outputs를 독립 사본으로 만들어 전달
        unnorm_inputs_copy, unnorm_outputs_copy = _clone_inputs_outputs_for_independent_rollout(
            inputs=inputs,
            outputs=outputs,
        )

        _predict_one_rollout_sequential(
            args=args,
            model=model,
            unnorm_inputs_copy=unnorm_inputs_copy,
            unnorm_outputs_copy=unnorm_outputs_copy,
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


from typing import Any, Dict, Tuple
import os
import contextlib
import numpy as np


def _to_bool_mask_np(mask: Any) -> np.ndarray:
    """유효 마스크를 안전하게 True/False 배열로 바꿉니다.

    Args:
        mask (Any): np.ndarray 또는 그에 준하는 값. shape: (T,) 또는 (Pnn,T) 등

    Returns:
        np.ndarray: bool 마스크. shape: mask와 동일
    """
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return m
    return (m != 0)


def _augment_inputs_with_ego_futures_for_npz(
    a_inputs_dict: Dict[str, Any],
    a_outputs_dict: Dict[str, Any],
) -> None:
    """(함수 1) ego 미래 파생 값들을 a_inputs_dict에 채웁니다.

    하는 일
    ------
    1) ego_future_gt_4_dim -> ego_future_gt_3_dim 생성

    Returns:
        None
    """
    ego_future_gt_is_valid_raw = a_outputs_dict.get("ego_future_gt_is_valid",
                                                   None)
    if ego_future_gt_is_valid_raw is None:
        ego_future_gt_is_valid_raw = a_outputs_dict.get(
            "ego_future_gt_is_valid", None)
    if ego_future_gt_is_valid_raw is None:
        raise KeyError("ego_future_gt_is_valid 키가 없습니다.")
    ego_future_gt_is_valid = _to_bool_mask_np(ego_future_gt_is_valid_raw)

    # ego_future_gt_4_np: (future_len, 4)
    ego_future_gt_4_np = a_outputs_dict["ego_future_gt_4_dim"]

    # ego_future_gt_3_dim: (future_len, 3) # a_inputs_dict["ego_future_gt_3_dim"]
    ego_future_gt_3_dim = _pose_4_dim_numpy_to_pose_3_dim_numpy(
        ego_future_gt_4_np)
    # ego_future_gt_3_dim 에서, 무효 칸은 0. 으로 처리
    ego_future_gt_3_dim[~ego_future_gt_is_valid, :] = 0.0
    a_inputs_dict[
        "ego_future_gt_3_dim"] = ego_future_gt_3_dim  # (future_len, 3)

    # ego_agent_past: (time_len, 11)
    ego_agent_past = a_inputs_dict["ego_agent_past"]
    # ego_agent_current: (11,)
    ego_agent_current = ego_agent_past[-1, :]

    # ego_future_gt_11_dim: (future_len, 11)
    ego_future_gt_11_dim = np.tile(
        ego_agent_current[np.newaxis, :],
        (int(ego_future_gt_4_np.shape[0]), 1),
    )
    ego_future_gt_11_dim[:, 0:4] = ego_future_gt_4_np  # (future_len, 11)

    # ego_future_gt_is_valid: (future_len,) bool
    ego_future_gt_11_dim[~ego_future_gt_is_valid, :8] = 0.0
    a_inputs_dict["ego_future_gt_11_dim"] = ego_future_gt_11_dim


def _augment_inputs_with_neighbor_futures_for_npz(
    a_inputs_dict: Dict[str, Any],
    a_outputs_dict: Dict[str, Any],
) -> None:
    """(함수 2) near/neighbor 미래 파생 값들을 a_inputs_dict에 채웁니다.

    하는 일
    ------
    1) near_future_gt_4_dim -> neighbor_future_gt_3_dim 생성
    2) neighbor 현재 상태(11차원)로 neighbor_future_gt_11_dim 생성
       - GT가 없는 칸은 앞 8개 값을 0으로 처리 (for문 없이 한 번에 처리)

    Args:
        a_inputs_dict: 샘플 1개 입력 dict. shape: ()
        a_outputs_dict: 샘플 1개 출력 dict. shape: ()

    Returns:
        None
    """
    # neighbor_agents_past: (Pnn, time_len, 11)
    neighbor_agents_past = a_inputs_dict.get("neighbor_agents_past", None)
    if neighbor_agents_past is None:
        return
    near_future_gt_is_valid_raw = a_inputs_dict.get("near_future_gt_is_valid",
                                                    None)
    if near_future_gt_is_valid_raw is None:
        near_future_gt_is_valid_raw = a_outputs_dict.get(
            "near_future_gt_is_valid", None)
    if near_future_gt_is_valid_raw is None:
        raise KeyError("near_future_gt_is_valid 키가 없습니다.")

    # near_future_gt_is_valid: (Pnn, future_len) bool
    near_future_gt_is_valid = _to_bool_mask_np(near_future_gt_is_valid_raw)

    # near_future_gt_4_np: (Pnn, future_len, 4)
    near_future_gt_4_np = a_outputs_dict["near_future_gt_4_dim"]

    # neighbor_future_gt_3_dim: (Pnn, future_len, 3)
    neighbor_future_gt_3_dim = _pose_4_dim_numpy_to_pose_3_dim_numpy(
        near_future_gt_4_np)
    # neighbor_future_gt_3_dim 에서, 무효 칸은 0. 으로 처리
    neighbor_future_gt_3_dim[~near_future_gt_is_valid, :] = 0.0
    # (Pnn, future_len, 3)
    a_inputs_dict["neighbor_future_gt_3_dim"] = neighbor_future_gt_3_dim

    # neighbor_agents_current: (Pnn, 11)
    neighbor_agents_current = neighbor_agents_past[:, -1, :]

    future_len = int(near_future_gt_4_np.shape[1])

    # neighbor_future_11_dim: (Pnn, future_len, 11)
    neighbor_future_11_dim = np.tile(
        neighbor_agents_current[:, np.newaxis, :],
        (1, future_len, 1),
    )

    # (Pnn, future_len, 11)에서 (x,y,cos,sin)만 GT로 교체
    neighbor_future_11_dim[:, :, 0:4] = near_future_gt_4_np
    # ✅ 무효 칸은 앞 8개를 0으로 (배치 처리)
    # mask_f: (Pnn, future_len, 1)
    mask_f = near_future_gt_is_valid.astype(neighbor_future_11_dim.dtype)[:, :,
                                                                          None]
    neighbor_future_11_dim[:, :, :8] *= mask_f  # (Pnn, future_len, 8)

    a_inputs_dict["neighbor_future_gt_11_dim"] = neighbor_future_11_dim


def _build_npz_paths_and_payload(
    save_dir: str,
    a_inputs_dict: Dict[str, Any],
    step_count: int,
    global_sample_idx: int,
) -> Tuple[str, str, Dict[str, Any]]:
    """(함수 3) 저장 경로와 npz payload를 만듭니다.

    Args:
        save_dir (str): 저장 폴더 경로. shape: ()
        a_inputs_dict (Dict[str, Any]): 샘플 1개 입력 dict. shape: ()
        step_count (int): step 번호. shape: ()
        global_sample_idx (int): rollout 포함한 전역 샘플 인덱스. shape: ()

    Returns:
        Tuple[str, str, Dict[str, Any]]:
            - final_path: 최종 저장 경로. shape: ()
            - tmp_path: 임시 저장 경로. shape: ()
            - npz_payload_dict: 저장할 dict. shape: ()
    """
    final_file_name = _build_inference_npz_file_name(
        scenario_id=str(a_inputs_dict["scenario_id"]),
        step_count=int(step_count),
        sample_idx=int(global_sample_idx),
    )
    final_path = os.path.join(str(save_dir), final_file_name)
    tmp_path = final_path + ".tmp"
    neighbor_agents_is_valid = a_inputs_dict.get("neighbor_agents_is_valid",
                                                 None)
    lanes_is_valid = a_inputs_dict.get("lanes_is_valid", None)  # (L_max)
    route_lanes_is_valid = a_inputs_dict.get("route_lanes_is_valid",
                                             None)  # (R_max)
    stop_sign_is_valid = a_inputs_dict.get("stop_sign_is_valid",
                                           None)  # (N_max)
    speed_bump_is_valid = a_inputs_dict.get("speed_bump_is_valid",
                                            None)  # (S_max)
    crosswalk_is_valid = a_inputs_dict.get("crosswalk_is_valid",
                                           None)  # (C_max)
    driveway_is_valid = a_inputs_dict.get("driveway_is_valid", None)  # (D_max)
    road_edge_is_valid = a_inputs_dict.get("road_edge_is_valid",
                                           None)  # (E_max)
    static_objects_is_valid = a_inputs_dict.get("static_objects_is_valid",
                                                None)  # (O_max)

    npz_payload_dict: Dict[str, Any] = _build_inference_npz_payload_for_save(
        a_inputs_dict)

    npz_payload_dict = _remove_invalid_data(
        npz_payload_dict, neighbor_agents_is_valid, lanes_is_valid,
        route_lanes_is_valid, stop_sign_is_valid, speed_bump_is_valid,
        crosswalk_is_valid, driveway_is_valid, road_edge_is_valid,
        static_objects_is_valid)
    return final_path, tmp_path, npz_payload_dict


def _save_inference_data(
    dir: str,
    unnorm_inputs: Dict[str, Any],
    unnorm_outputs: Dict[str, Any],
    step_count: int,
    *,
    sample_idx_offset: int = 0,
) -> None:
    """rollout 중간 상태를 npz로 저장합니다(내부를 역할별 함수로 분리한 버전)."""
    os.makedirs(dir, exist_ok=True)
    # (B, future_len, 4)
    ego_future_gt_4_dim = unnorm_outputs["ego_future_gt_4_dim"]
    batch_size = int(ego_future_gt_4_dim.shape[0])  # B

    unnorm_inputs_np = _gpu_tensor_to_cpu_np(
        unnorm_inputs=unnorm_inputs,
        batch_size=int(batch_size),
    )
    unnorm_outputs_np = _gpu_tensor_to_cpu_np(
        unnorm_inputs=unnorm_outputs,
        batch_size=int(batch_size),
    )

    enable_fsync = bool(
        _read_float_env_safe("DP_INFERENCE_NPZ_FSYNC", 1.0) > 0.0)
    enable_fsync = False  # 기존 코드 동작 유지

    for current_batch_idx in range(batch_size):
        a_inputs_dict = _slice_one_sample_from_cpu_batch_cache(
            unnorm_data_np=unnorm_inputs_np,
            sample_idx=int(current_batch_idx),
            batch_size=int(batch_size),
        )
        a_outputs_dict = _slice_one_sample_from_cpu_batch_cache(
            unnorm_data_np=unnorm_outputs_np,
            sample_idx=int(current_batch_idx),
            batch_size=int(batch_size),
        )

        # 함수 1
        _augment_inputs_with_ego_futures_for_npz(
            a_inputs_dict=a_inputs_dict,
            a_outputs_dict=a_outputs_dict,
        )

        # 함수 2
        _augment_inputs_with_neighbor_futures_for_npz(
            a_inputs_dict=a_inputs_dict,
            a_outputs_dict=a_outputs_dict,
        )

        # ✅ (추가) future_seg_control_gt_3_dim을 inputs 쪽에 포함
        _augment_inputs_with_future_seg_control_for_npz(
            a_inputs_dict=a_inputs_dict,
            a_outputs_dict=a_outputs_dict,
        )

        # 함수 3
        global_sample_idx = int(sample_idx_offset) + int(current_batch_idx)
        final_path, tmp_path, npz_payload_dict = _build_npz_paths_and_payload(
            save_dir=str(dir),
            a_inputs_dict=a_inputs_dict,
            step_count=int(step_count),
            global_sample_idx=int(global_sample_idx),
        )

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
    unnorm_outputs_copy: Dict[str, Any],
    unnorm_selected_gt_traj: torch.Tensor,  # (B, 1+Pnn, 1+T, 4)
    draw_batch_idx: int,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    # 역정규화: ((1+)Pnn, 1+T, 4)
    a_unnorm_selected_gt_traj = unnorm_selected_gt_traj[draw_batch_idx]
    a_unnorm_selected_gt_traj_np = a_unnorm_selected_gt_traj.cpu().numpy()

    unnorm_ego_future_gt_4_dim = unnorm_outputs_copy["ego_future_gt_4_dim"].cpu(
    ).numpy()  # (B, future_len, 4)
    a_unnorm_ego_future_gt_4_dim = unnorm_ego_future_gt_4_dim[
        draw_batch_idx]  # (future_len, 4)

    unnorm_near_future_gt_4_dim = unnorm_outputs_copy[
        "near_future_gt_4_dim"].cpu().numpy()  # (B, Pnn, future_len, 4)
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
    return a_unnorm_inputs_np, a_unnorm_selected_gt_traj_np, a_unnorm_near_future_gt_3_dim, a_unnorm_ego_future_gt_4_dim


from typing import Any, Dict, Tuple
import torch.nn as nn

from typing import Any, Tuple
import torch


def _extract_agent_box_size_m_from_past_states(
    *,
    ego_agent_past: torch.Tensor,  # (B, T_past, 11)
    ego_agent_past_is_valid: torch.Tensor,  # (B, T_past)
    near_agents_past: torch.Tensor,  # (B, Pnn, T_past, 11)
    near_agents_past_is_valid: torch.Tensor,  # (B, Pnn, T_past)
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
    - 무효인 경우 length/width도 0일 가능성이 크므로,
      안전하게 기본값(default_length_m/default_width_m)을 넣습니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (unnorm_agent_length_m, unnorm_agent_width_m)
            - unnorm_agent_length_m shape: (B_all, 1+Pnn)
            - unnorm_agent_width_m  shape: (B_all, 1+Pnn)
            agent 순서는 [ego(0번)] + [near(1..Pnn)] 입니다.
    """
    # 현재 시점(마지막 time index)에서 size 읽기
    # ego_current_11_dim:  (B_all, 11)
    ego_current_11_dim = ego_agent_past[:, -1, :]
    # near_current_11_dim: (B_all, Pnn, 11)
    near_current_11_dim = near_agents_past[:, :, -1, :]

    # 유효 여부: 앞 8개가 전부 0이면 무효
    # ego_valid:  (B_all,)
    ego_valid = ego_agent_past_is_valid[:, -1]
    # near_valid: (B_all, Pnn)
    near_valid = near_agents_past_is_valid[:, :, -1]

    # length/width 인덱스: 6, 7
    # ego_length:  (B_all,)
    # ego_width:   (B_all,)
    ego_length = ego_current_11_dim[:, 6]
    ego_width = ego_current_11_dim[:, 7]

    # near_length: (B_all, Pnn)
    # near_width:  (B_all, Pnn)
    near_length = near_current_11_dim[..., 6]
    near_width = near_current_11_dim[..., 7]

    # (B_all, 1+Pnn)
    agent_length = torch.cat([ego_length[:, None], near_length], dim=1)
    agent_width = torch.cat([ego_width[:, None], near_width], dim=1)

    # (B_all, 1+Pnn)
    agent_valid = torch.cat([ego_valid[:, None], near_valid], dim=1)
    agent_valid = agent_valid.to(dtype=torch.bool)

    # 무효 에이전트는 기본값으로 대체
    default_len = torch.full_like(agent_length, float(default_length_m))
    default_wid = torch.full_like(agent_width, float(default_width_m))
    agent_length = torch.where(agent_valid, agent_length, default_len)
    agent_width = torch.where(agent_valid, agent_width, default_wid)

    # 거리 계산용으로 float32
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


from typing import Any, Dict, Tuple
import torch


def _extract_target_future_gt_and_valid_for_compare(
    *,
    unnorm_outputs_step: Dict[str, torch.Tensor],
    compare_steps: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """정답 미래 포즈와 유효 마스크를 비교 구간 길이로 잘라 (ego+near)로 합칩니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, int]:
            - unnorm_gt_compare_4_dim: (B_all, 1+Pnn, H, 4)
            - target_compare_gt_is_valid: (B_all, 1+Pnn, H)  bool
    """
    ego_future_gt_4_dim = unnorm_outputs_step[
        "ego_future_gt_4_dim"]  # (B_all, T, 4)
    ego_future_gt_is_valid = unnorm_outputs_step[
        "ego_future_gt_is_valid"]  # (B_all, T)

    near_future_gt_4_dim = unnorm_outputs_step[
        "near_future_gt_4_dim"]  # (B_all, Pnn, T, 4)
    near_future_gt_is_valid = unnorm_outputs_step[
        "near_future_gt_is_valid"]  # (B_all, Pnn, T)

    future_len = int(ego_future_gt_4_dim.shape[1])

    # target_compare_gt_is_valid: (B_all, 1+Pnn, T) -> (B_all, 1+Pnn, H)
    target_future_gt_is_valid = torch.cat(
        [
            ego_future_gt_is_valid[:, None, :],
            near_future_gt_is_valid,
        ],
        dim=1,
    ).to(dtype=torch.bool)

    target_compare_gt_is_valid = target_future_gt_is_valid[:, :, :int(
        compare_steps)]  # (B_all, 1+Pnn, H)

    # unnorm_gt_compare_4_dim: (B_all, 1+Pnn, H, 4)
    ego_gt = ego_future_gt_4_dim[:, :int(compare_steps), :]  # (B_all, H, 4)
    near_gt = near_future_gt_4_dim[:, :, :int(
        compare_steps), :]  # (B_all, Pnn, H, 4)
    unnorm_gt_compare_4_dim = torch.cat([ego_gt[:, None, :, :], near_gt], dim=1)

    return unnorm_gt_compare_4_dim, target_compare_gt_is_valid


def _compute_corner_mean_distance_m_per_agent(
        *,
        unnorm_pred_future_4_dim: torch.Tensor,  # (B_all, 1+Pnn, H, 4)
        unnorm_gt_future_4_dim: torch.Tensor,  # (B_all, 1+Pnn, H, 4)
        target_compare_gt_is_valid: torch.Tensor,  # (B_all, 1+Pnn, H) bool
        unnorm_agent_length_m: torch.Tensor,  # (B_all, 1+Pnn)
        unnorm_agent_width_m: torch.Tensor,  # (B_all, 1+Pnn)
) -> torch.Tensor:
    """예측/정답 포즈를 박스 코너로 바꾼 뒤, 코너 평균 거리로 에이전트별 점수를 만듭니다.

    Args:
        unnorm_pred_future_4_dim (torch.Tensor):
            예측 포즈(원래 단위). shape: (B_all, 1+Pnn, H, 4)
        unnorm_gt_future_4_dim (torch.Tensor):
            정답 포즈(원래 단위). shape: (B_all, 1+Pnn, H, 4)
        target_compare_gt_is_valid (torch.Tensor):
            정답이 존재하는 칸만 True. shape: (B_all, 1+Pnn, H) bool
        unnorm_agent_length_m (torch.Tensor):
            에이전트 길이. shape: (B_all, 1+Pnn)
        unnorm_agent_width_m (torch.Tensor):
            에이전트 너비. shape: (B_all, 1+Pnn)

    Returns:
        torch.Tensor:
            dist_m: 에이전트별 평균 거리(미터). shape: (B_all, 1+Pnn)
            dtype: float32
    """
    # H: 비교 스텝 수
    compare_steps = int(unnorm_pred_future_4_dim.shape[2])

    # float32로 안정화
    pred_f = unnorm_pred_future_4_dim.to(
        dtype=torch.float32)  # (B_all, 1+Pnn, H, 4)
    gt_f = unnorm_gt_future_4_dim.to(
        dtype=torch.float32)  # (B_all, 1+Pnn, H, 4)

    # 길이/너비를 시간축으로 늘림: (B_all, 1+Pnn, H)
    length_h = unnorm_agent_length_m.to(dtype=torch.float32)[:, :, None].expand(
        -1, -1, compare_steps)
    width_h = unnorm_agent_width_m.to(dtype=torch.float32)[:, :, None].expand(
        -1, -1, compare_steps)

    # corners: (B_all, 1+Pnn, H, 4, 2)
    pred_corners = _build_box_corners_xy_from_pose_4_dim_with_size(
        pose_4_dim=pred_f,
        length_m=length_h,
        width_m=width_h,
    )
    gt_corners = _build_box_corners_xy_from_pose_4_dim_with_size(
        pose_4_dim=gt_f,
        length_m=length_h,
        width_m=width_h,
    )

    # diff: (B_all, 1+Pnn, H, 4, 2)
    diff = pred_corners - gt_corners

    # corner_dist: (B_all, 1+Pnn, H, 4)
    corner_dist = torch.sqrt(
        torch.clamp(diff[..., 0]**2 + diff[..., 1]**2, min=0.0))

    # step_dist: (B_all, 1+Pnn, H)  코너 평균
    step_dist = corner_dist.mean(dim=-1)

    valid_f = target_compare_gt_is_valid.to(
        dtype=torch.float32)  # (B_all, 1+Pnn, H)
    sum_dist = (step_dist * valid_f).sum(dim=-1)  # (B_all, 1+Pnn)
    denom = torch.clamp(valid_f.sum(dim=-1), min=1.0)  # (B_all, 1+Pnn)

    dist = (sum_dist / denom).to(dtype=torch.float32)  # (B_all, 1+Pnn)
    return dist


def _build_target_agent_current_is_valid_mask(
    norm_inputs_step: Dict[str, torch.Tensor],) -> torch.Tensor:
    """현재 시점에서 agent가 존재하는지(True/False) 마스크를 만듭니다.

    Args:
        norm_inputs_step (Dict[str, torch.Tensor]):
            입력 dict. shape: ()
            필요한 키와 shape:
              - ego_agent_past_is_valid: (B_all, T_past)
              - near_agents_past_is_valid: (B_all, Pnn, T_past)

    Returns:
        torch.Tensor:
            target_agent_current_is_valid: (B_all, 1+Pnn) bool
    """
    ego_past_is_valid = norm_inputs_step[
        "ego_agent_past_is_valid"]  # (B_all, T_past)
    near_past_is_valid = norm_inputs_step[
        "near_agents_past_is_valid"]  # (B_all, Pnn, T_past)
    ego_cur = ego_past_is_valid[:, -1].to(dtype=torch.bool)  # (B_all,)
    near_cur = near_past_is_valid[:, :, -1].to(dtype=torch.bool)  # (B_all, Pnn)

    # (B_all, 1+Pnn)
    return torch.cat([ego_cur[:, None], near_cur], dim=1)


def _zero_out_distance_for_invalid_agents(
    dist_m_per_agent: torch.Tensor,
    agent_current_is_valid: torch.Tensor,
) -> torch.Tensor:
    """존재하지 않는 agent의 거리 점수를 0으로 고정합니다.

    Args:
        dist_m_per_agent (torch.Tensor):
            거리 점수. shape: (B_all, 1+Pnn)
        agent_current_is_valid (torch.Tensor):
            현재 존재 여부. shape: (B_all, 1+Pnn) bool

    Returns:
        torch.Tensor:
            0 처리된 거리 점수. shape: (B_all, 1+Pnn)
    """
    invalid_mask = ~agent_current_is_valid.to(
        dtype=torch.bool)  # (B_all, 1+Pnn)
    return torch.where(invalid_mask, torch.zeros_like(dist_m_per_agent),
                       dist_m_per_agent)


def _compute_expert_guidance_distance_m_per_agent(
        *,
        state_normalizer: Any,
        normed_trajectory: torch.Tensor,  # (B_all, 1+Pnn, 1+future_len, 4)
        norm_br_inputs_step: Dict[str, torch.Tensor],
        unnorm_br_outputs_step: Dict[str, torch.Tensor],
        compare_steps: int,
        unnorm_agent_length_m: torch.Tensor,  # (B_all, 1+Pnn)
        unnorm_agent_width_m: torch.Tensor,  # (B_all, 1+Pnn)
) -> torch.Tensor:  # (B_all, 1+Pnn)
    """후보 경로(예측)와 정답 미래 경로를 비교해, 에이전트별 '코너 평균 거리'를 계산합니다.

    Returns:
        torch.Tensor:
            에이전트별 거리 점수.
            shape: (B_all, 1+Pnn)
            dtype: float32
    """
    # 1) 정답(미래) + 유효 마스크 준비 (비교 구간으로 자름)
    # unnorm_gt_compare_4_dim: (B_all, 1+Pnn, H, 4),
    # target_compare_gt_is_valid: (B_all, 1+Pnn, H)
    (unnorm_gt_compare_4_dim, target_compare_gt_is_valid
    ) = _extract_target_future_gt_and_valid_for_compare(
        unnorm_outputs_step=unnorm_br_outputs_step,
        compare_steps=int(compare_steps),
    )

    # 2) 예측(미래) 비교 구간 뽑기 + 길이 검증
    # # (B_all, 1+Pnn, H, 4)
    normed_pred_future_4_dim = normed_trajectory[:, :,
                                                 1:int(compare_steps) + 1, :]
    # 3) 예측을 원래 단위로 되돌리되, 정답이 없는 칸은 0으로 유지
    # unnorm_pred_future_4_dim :  (B_all, 1+Pnn, H, 4)
    unnorm_pred_future_4_dim = state_normalizer.inverse(
        data=normed_pred_future_4_dim,  # (B_all, 1+Pnn, H, 4)
        valid_mask=target_compare_gt_is_valid,  # (B_all, 1+Pnn, H)
    )

    # 4) 코너 평균 거리로 에이전트별 점수 계산(정답이 없는 칸은 제외)
    # dist: (B_all, 1+Pnn), float32
    dist = _compute_corner_mean_distance_m_per_agent(
        unnorm_pred_future_4_dim=unnorm_pred_future_4_dim,  # (B_all, 1+Pnn, H, 4)
        unnorm_gt_future_4_dim=unnorm_gt_compare_4_dim,  # (B_all, 1+Pnn, H, 4)
        target_compare_gt_is_valid=
        target_compare_gt_is_valid,  # (B_all, 1+Pnn, H)
        unnorm_agent_length_m=unnorm_agent_length_m,  # (B_all, 1+Pnn)
        unnorm_agent_width_m=unnorm_agent_width_m,  # (B_all, 1+Pnn)
    )

    # 5) "존재하지 않는 agent"는 0으로 고정
    # target_agent_current_is_valid: (B_all, 1+Pnn) bool
    target_agent_current_is_valid = _build_target_agent_current_is_valid_mask(
        norm_inputs_step=norm_br_inputs_step,)
    # dist: (B_all, 1+Pnn)
    dist = _zero_out_distance_for_invalid_agents(
        dist_m_per_agent=dist,  # (B_all, 1+Pnn)
        agent_current_is_valid=target_agent_current_is_valid,  # (B_all, 1+Pnn)
    )
    return dist


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
    norm_data_step: Dict[str, Any],
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
        norm_data_step (Dict[str, Any]):
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
    for k, v in norm_data_step.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(
                v.shape[0]) == b:
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
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    pose_based: bool,
    noise_std: float,
    seed_stride: int,
    cand_start_idx: int,
    cand_count: int,
) -> torch.Tensor:
    """cand_start_idx부터 cand_count개 후보의 노이즈를 한 번에 만들어 합칩니다.

    Returns:
        torch.Tensor:
            후보들을 이어붙인 노이즈 텐서.
            shape:
              - pose_based=True  -> (B*cand_count, 1+Pnn, future_len, 4)
              - pose_based=False -> (B*cand_count, 1+Pnn, future_len, 3)
    """
    last_dim = 4 if bool(pose_based) else 3

    c = int(max(1, int(cand_count)))
    b = int(max(1, int(batch_size)))

    noises = []
    for local_i in range(c):
        cand_idx = int(cand_start_idx) + int(local_i)
        cand_base_seed = int(base_seed) + int(cand_idx) * int(seed_stride)

        # noise_i: (B, 1+Pnn, future_len, 4 or 3)
        noise_i = _build_inference_noise_for_rollout_chunk(
            device=device,  # device/dtype 기준
            dtype=dtype,  # device/dtype 기준
            batch_size=b,
            one_or_pnn=int(one_or_pnn),
            future_len=int(future_len),
            rollout_idx=int(rollout_idx),
            base_seed=int(cand_base_seed),
            ddp_rank=int(ddp_rank),
            step_idx=int(step_idx),
            pose_based=pose_based,
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
        last_dim,
    )
    return noise_flat


def _forward_and_score_candidate_batch(
    *,
    args: Any,
    model: nn.Module,
    norm_inputs_step: Dict[str, Any],
    state_normalizer: Any,
    unnorm_outputs_copy: Dict[str, torch.Tensor],
    unnorm_agent_length_m: torch.Tensor,
    unnorm_agent_width_m: torch.Tensor,
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
    gap: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """후보 cand_count개를 한 번에 모델에 넣고, 거리 점수까지 계산합니다.

    변경점(핵심)
    ----------
    - amortized + step_idx>0 에서 Decoder가 내부에서 randn_like로 랜덤을 만들지 않도록,
      "amortized_random_noise"를 cand_idx 기반 seed로 만들어 inputs로 전달합니다.
    - 이로써 후보의 랜덤이 GPU RNG 상태가 아니라 cand_idx/seed로 고정됩니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (cand_traj, cand_dist)
            - cand_traj: (cand_count, B, 1+Pnn, 1+future_len, 4)
            - cand_dist: (cand_count, B, 1+Pnn)
            - target_future_control_seq: (cand_count, B, (1+)Pnn, future_len, 3) or None
    """
    rollout_repeat = int(max(1, int(cand_count)))
    b = int(max(1, int(batch_size)))

    use_amortized = bool(getattr(args, "use_amortized_diffusion", False))

    reference_tensor = norm_inputs_step["ego_agent_past"]
    device = reference_tensor.device
    dtype = reference_tensor.dtype

    # ------------------------------------------------------------
    # (1) 후보별 노이즈 준비
    #   - inference_noise: step_idx==0(또는 non-amortized)에서 쓰는 "초기 xT 노이즈"
    #   - amortized_random_noise: amortized + step_idx>0에서 쓰는 "이전 버퍼에 섞을 랜덤"
    #     (Decoder 내부 randn_like를 대체)
    # ------------------------------------------------------------
    inference_noise_flat: Optional[torch.Tensor] = None
    amortized_random_noise_flat: Optional[torch.Tensor] = None

    need_initial_inference_noise = (not use_amortized) or (use_amortized and
                                                           int(step_idx) == 0)
    need_amortized_random_noise = use_amortized
    pose_based_flag = bool(getattr(args, "pose_based", True))
    if need_initial_inference_noise:
        # inference_noise_flat: (B*rollout_repeat, 1+Pnn, future_len, 4 or 3)
        inference_noise_flat = _build_inference_noise_batch_for_candidate_range(
            device=device,
            dtype=dtype,
            batch_size=b,
            one_or_pnn=int(one_or_pnn),
            future_len=int(future_len),
            rollout_idx=int(rollout_idx),
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            step_idx=int(step_idx),
            pose_based=pose_based_flag,
            noise_std=float(getattr(args, "fine_tune_temperature", 0.5)),
            seed_stride=int(seed_stride),
            cand_start_idx=int(cand_start_idx),
            cand_count=int(rollout_repeat),
        )

    if need_amortized_random_noise:
        # ✅ Decoder의 기존 torch.randn_like(...)와 같은 분포를 맞추기 위해 noise_std=1.0 사용
        # amortized_random_noise_flat: (B*rollout_repeat, 1+Pnn, future_len, 4)
        amortized_random_noise_flat = _build_inference_noise_batch_for_candidate_range(
            device=device,
            dtype=dtype,
            batch_size=b,
            one_or_pnn=int(one_or_pnn),
            future_len=int(future_len),
            rollout_idx=int(rollout_idx),
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            step_idx=int(step_idx),
            pose_based=pose_based_flag,
            noise_std=1.0,
            seed_stride=int(seed_stride),
            cand_start_idx=int(cand_start_idx),
            cand_count=int(rollout_repeat),
        )

    # outputs도 후보 개수만큼 반복
    unnorm_br_outputs_step = _repeat_inputs_for_candidate_batch(
        norm_data_step=unnorm_outputs_copy,
        repeat=int(rollout_repeat),
        batch_size=int(b),
    )

    # rollout_time_chunk_size: (B,)
    norm_inputs_step["rollout_time_chunk_size"] = torch.tensor(
        [int(gap)] * int(b),
        dtype=torch.int64,
        device=norm_inputs_step["ego_agent_past"].device,
    )

    # (2) 입력 dict를 (B*rollout_repeat, ...)로 늘리기
    norm_br_inputs_step = _repeat_inputs_for_candidate_batch(
        norm_data_step=norm_inputs_step,
        repeat=int(rollout_repeat),
        batch_size=int(b),
    )

    # ✅ Decoder가 읽는 키들
    norm_br_inputs_step["inference_noise"] = inference_noise_flat
    norm_br_inputs_step["amortized_random_noise"] = amortized_random_noise_flat

    # (3) 모델 forward (한 번)
    decoder_output = _forward_model_for_validation(
        args=args,
        model=model,
        norm_inputs=norm_br_inputs_step,
    )

    cand_traj_flat = decoder_output[
        "integrated_trajectory"]  # (B*R, 1+Pnn, 1+T, 4)
    if args.pose_based:
        target_future_control_seq = None
    else:
        # (B*R, (1+)Pnn, T, 3)
        # ✅ 우선: Feasible가 최종으로 사용한 control
        target_future_control_seq = decoder_output.get(
            "control_sequence", None)

        # fallback: 구버전/예외 상황에서는 기존 score 사용
        if not isinstance(target_future_control_seq, torch.Tensor):
            raise KeyError("Decoder 출력에 'control_sequence' 키가 없거나 텐서가 아닙니다. "
                           "rollout_time_chunk_size > 1이면서 pose_based=False인 경우, "
                           "모델이 'control_sequence'를 출력하도록 해야 합니다.")


    cand_traj = cand_traj_flat.reshape(
        int(rollout_repeat),
        int(b),
        int(one_or_pnn),
        1 + int(future_len),
        4,
    )
    # target_future_control_seq: (B*R, (1+)Pnn, T, 3) or None -> (R, B, (1+)Pnn, T, 3)
    target_future_control_seq = target_future_control_seq.reshape(
        int(rollout_repeat),
        int(b),
        int(one_or_pnn),
        int(future_len),
        -1,
    ) if target_future_control_seq is not None else None

    # len_rep/wid_rep: (B*R, 1+Pnn)
    len_rep = unnorm_agent_length_m.repeat(int(rollout_repeat), 1)
    wid_rep = unnorm_agent_width_m.repeat(int(rollout_repeat), 1)

    # cand_dist_flat: (B*R, 1+Pnn)
    cand_dist_flat = _compute_expert_guidance_distance_m_per_agent(
        state_normalizer=state_normalizer,
        normed_trajectory=cand_traj_flat,
        norm_br_inputs_step=norm_br_inputs_step,
        unnorm_br_outputs_step=unnorm_br_outputs_step,
        compare_steps=int(getattr(args, "time_step_for_compare", 1)),
        unnorm_agent_length_m=len_rep,
        unnorm_agent_width_m=wid_rep,
    )

    cand_dist = cand_dist_flat.reshape(int(rollout_repeat), int(b),
                                       int(one_or_pnn))

    return cand_traj, cand_dist, target_future_control_seq


def _compute_candidate_score_mean_over_valid_agents(
        *,
        cand_dist: torch.Tensor,  # (Kc, B, 1+Pnn)
        agent_current_is_valid: torch.Tensor,  # (B, 1+Pnn) bool
) -> torch.Tensor:
    """후보 점수를 '현재 존재하는 agent만' 평균내서 계산합니다.

    왜 필요한가?
    -----------
    cand_dist에서 존재하지 않는 agent의 거리를 0으로 만들어 두면,
    단순 mean(dim=2)은 0이 많이 섞일수록 평균이 작아져서
    "유효 agent가 적은 후보"가 유리해질 수 있습니다.

    그래서 평균을 낼 때는, 현재 존재하는 agent(True)만 포함해서
    (합 / 유효 개수)로 점수를 만듭니다.

    Args:
        cand_dist (torch.Tensor):
            후보별 거리 점수.
            shape: (Kc, B, 1+Pnn)
        agent_current_is_valid (torch.Tensor):
            현재 시점에서 agent가 존재하는지 마스크.
            shape: (B, 1+Pnn)
            True: 존재(유효), False: 없음(무효)

    Returns:
        torch.Tensor:
            후보 점수(작을수록 좋음).
            shape: (Kc, B)
            dtype: float32
    """
    if cand_dist.dim() != 3:
        raise ValueError(
            f"cand_dist는 (Kc,B,1+Pnn) 여야 합니다. got shape={tuple(cand_dist.shape)}"
        )
    if agent_current_is_valid.dim() != 2:
        raise ValueError("agent_current_is_valid는 (B,1+Pnn) 여야 합니다. "
                         f"got shape={tuple(agent_current_is_valid.shape)}")
    if int(cand_dist.shape[1]) != int(agent_current_is_valid.shape[0]) or int(
            cand_dist.shape[2]) != int(agent_current_is_valid.shape[1]):
        raise ValueError(
            "cand_dist와 agent_current_is_valid의 (B, 1+Pnn) shape가 맞지 않습니다. "
            f"cand_dist={tuple(cand_dist.shape)}, agent_current_is_valid={tuple(agent_current_is_valid.shape)}"
        )

    # valid_f: (B, 1+Pnn) float32 (True->1, False->0)
    valid_f = agent_current_is_valid.to(dtype=torch.float32)

    # denom: (B,)  유효 agent 수 (0이면 1로 클램프해서 0나눗셈 방지)
    denom = torch.clamp(valid_f.sum(dim=1), min=1.0)

    # sum_dist: (Kc, B)
    sum_dist = (cand_dist.to(dtype=torch.float32) *
                valid_f.unsqueeze(0)).sum(dim=2)

    # score: (Kc, B)
    score = sum_dist / denom.unsqueeze(0)
    return score


from typing import Optional, Tuple
import torch


def _select_best_from_candidate_batch_jointly(
        *,
        best_traj: Optional[torch.Tensor],
        best_dist: Optional[torch.Tensor],
        best_score: Optional[torch.Tensor],
        best_control_seq: Optional[torch.Tensor],
        cand_traj: torch.Tensor,  # (Kc, B, 1+Pnn, 1+future_len, 4)
        cand_dist: torch.Tensor,  # (Kc, B, 1+Pnn)
        cand_control_seq: Optional[torch.Tensor],  # (Kc, B, 1+Pnn, future_len, C) or None
        agent_current_is_valid: torch.Tensor,  # (B, 1+Pnn) bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """샘플마다 후보 1개를 고르는 방식(select_jointly=True)으로 best를 갱신합니다.

    핵심
    ----
    - traj/dist를 고를 때 사용한 후보 인덱스(group_best_idx)를,
      control_seq에도 그대로 적용해서 "같은 후보"의 control을 뽑습니다.
    - 더 좋은 후보로 바뀐 샘플은 traj/dist/score 뿐 아니라 control_seq도 같이 바뀝니다.

    Args:
        best_traj: 기존 best traj 또는 None.
            - shape: (B, 1+Pnn, 1+future_len, 4) 또는 None
        best_dist: 기존 best dist 또는 None.
            - shape: (B, 1+Pnn) 또는 None
        best_score: 기존 best score 또는 None.
            - shape: (B,) 또는 None
        best_control_seq: 기존 best control 또는 None.
            - shape: (B, 1+Pnn, future_len, C) 또는 None
        cand_traj: 후보 traj 묶음.
            - shape: (Kc, B, 1+Pnn, 1+future_len, 4)
        cand_dist: 후보 dist 묶음.
            - shape: (Kc, B, 1+Pnn)
        cand_control_seq: 후보 control 묶음(있을 때만).
            - shape: (Kc, B, 1+Pnn, future_len, C) 또는 None
        agent_current_is_valid: 현재 시점 존재하는 agent만 True.
            - shape: (B, 1+Pnn)

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            (best_traj, best_dist, best_score, best_control_seq)
            - best_traj: (B, 1+Pnn, 1+future_len, 4)
            - best_dist: (B, 1+Pnn)
            - best_score: (B,)
            - best_control_seq: (B, 1+Pnn, future_len, C) 또는 None
    """
    # cand_score: (Kc, B)  # 유효 agent만 평균
    cand_score = _compute_candidate_score_mean_over_valid_agents(
        cand_dist=cand_dist,  # (Kc, B, 1+Pnn)
        agent_current_is_valid=agent_current_is_valid,  # (B, 1+Pnn)
    )

    # group_best_score: (B,)
    # group_best_idx:   (B,)
    group_best_score, group_best_idx = cand_score.min(dim=0)

    # -------------------------
    # (1) traj 선택
    # -------------------------
    # cand_traj_perm: (B, Kc, 1+Pnn, 1+future_len, 4)
    cand_traj_perm = cand_traj.permute(1, 0, 2, 3, 4)
    b = int(cand_traj_perm.shape[0])
    one_pnn = int(cand_traj_perm.shape[2])
    t_traj = int(cand_traj_perm.shape[3])

    # idx_traj: (B, 1, 1+Pnn, 1+future_len, 4)
    idx_traj = group_best_idx[:, None, None, None,
                              None].expand(b, 1, one_pnn, t_traj, 4)
    group_best_traj = torch.take_along_dim(
        cand_traj_perm, idx_traj, dim=1).squeeze(1)

    # -------------------------
    # (2) dist 선택
    # -------------------------
    # cand_dist_perm: (B, Kc, 1+Pnn)
    cand_dist_perm = cand_dist.permute(1, 0, 2)
    idx_dist = group_best_idx[:, None, None].expand(b, 1, one_pnn)
    group_best_dist = torch.take_along_dim(
        cand_dist_perm, idx_dist, dim=1).squeeze(1)

    # -------------------------
    # (3) control 선택(있을 때만)
    # -------------------------
    group_best_control_seq: Optional[torch.Tensor] = None
    if isinstance(cand_control_seq, torch.Tensor):
        # cand_ctl_perm: (B, Kc, 1+Pnn, future_len, C)
        cand_ctl_perm = cand_control_seq.permute(1, 0, 2, 3, 4)
        t_ctl = int(cand_ctl_perm.shape[3])
        c_ctl = int(cand_ctl_perm.shape[4])

        # idx_ctl: (B, 1, 1+Pnn, future_len, C)
        idx_ctl = group_best_idx[:, None, None, None,
                                 None].expand(b, 1, one_pnn, t_ctl, c_ctl)
        group_best_control_seq = torch.take_along_dim(
            cand_ctl_perm, idx_ctl, dim=1).squeeze(1)

    # -------------------------
    # (4) best 갱신
    # -------------------------
    if best_traj is None or best_dist is None or best_score is None:
        return group_best_traj, group_best_dist, group_best_score, group_best_control_seq

    better = group_best_score < best_score  # (B,)
    if torch.any(better):
        best_traj[better] = group_best_traj[better]
        best_dist[better] = group_best_dist[better]
        if best_control_seq is not None and group_best_control_seq is not None:
            best_control_seq[better] = group_best_control_seq[better]

    best_score = torch.where(better, group_best_score, best_score)
    return best_traj, best_dist, best_score, best_control_seq



from typing import Optional, Tuple
import torch


def _select_best_from_candidate_batch_per_agent(
    *,
    best_traj: Optional[torch.Tensor],
    best_dist: Optional[torch.Tensor],
    best_control_seq: Optional[torch.Tensor],
    cand_traj: torch.Tensor,  # (Kc, B, 1+Pnn, 1+future_len, 4)
    cand_dist: torch.Tensor,  # (Kc, B, 1+Pnn)
    cand_control_seq: Optional[torch.Tensor],  # (Kc, B, 1+Pnn, future_len, C) or None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """에이전트마다 후보를 따로 고르는 방식(select_jointly=False)으로 best를 갱신합니다.

    핵심
    ----
    - traj/dist를 고를 때 사용한 후보 인덱스(group_best_idx)를,
      control_seq에도 그대로 적용해서 "같은 후보"의 control을 뽑습니다.
    - 더 좋은 후보로 바뀐 (샘플,에이전트) 위치만 traj/dist/control을 같이 갱신합니다.

    Args:
        best_traj: 기존 best traj 또는 None.
            - shape: (B, 1+Pnn, 1+future_len, 4) 또는 None
        best_dist: 기존 best dist 또는 None.
            - shape: (B, 1+Pnn) 또는 None
        best_control_seq: 기존 best control 또는 None.
            - shape: (B, 1+Pnn, future_len, C) 또는 None
        cand_traj: 후보 traj 묶음.
            - shape: (Kc, B, 1+Pnn, 1+future_len, 4)
        cand_dist: 후보 dist 묶음.
            - shape: (Kc, B, 1+Pnn)
        cand_control_seq: 후보 control 묶음(있을 때만).
            - shape: (Kc, B, 1+Pnn, future_len, C) 또는 None

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            (best_traj, best_dist, best_control_seq)
            - best_traj: (B, 1+Pnn, 1+future_len, 4)
            - best_dist: (B, 1+Pnn)
            - best_control_seq: (B, 1+Pnn, future_len, C) 또는 None
    """
    # group_best_dist: (B, 1+Pnn)
    # group_best_idx:  (B, 1+Pnn)
    group_best_dist, group_best_idx = cand_dist.min(dim=0)

    # -------------------------
    # (1) traj 선택 (agent별 idx)
    # -------------------------
    # cand_traj_perm: (B, 1+Pnn, Kc, 1+future_len, 4)
    cand_traj_perm = cand_traj.permute(1, 2, 0, 3, 4)
    b = int(cand_traj_perm.shape[0])
    one_pnn = int(cand_traj_perm.shape[1])
    t_traj = int(cand_traj_perm.shape[3])

    # idx_traj: (B, 1+Pnn, 1, 1+future_len, 4)
    idx_traj = group_best_idx[..., None, None,
                              None].expand(b, one_pnn, 1, t_traj, 4)
    group_best_traj = torch.take_along_dim(
        cand_traj_perm, idx_traj, dim=2).squeeze(2)

    # -------------------------
    # (2) control 선택 (agent별 idx, 있을 때만)
    # -------------------------
    group_best_control_seq: Optional[torch.Tensor] = None
    if isinstance(cand_control_seq, torch.Tensor):
        # cand_ctl_perm: (B, 1+Pnn, Kc, future_len, C)
        cand_ctl_perm = cand_control_seq.permute(1, 2, 0, 3, 4)
        t_ctl = int(cand_ctl_perm.shape[3])
        c_ctl = int(cand_ctl_perm.shape[4])

        # idx_ctl: (B, 1+Pnn, 1, future_len, C)
        idx_ctl = group_best_idx[..., None, None,
                                 None].expand(b, one_pnn, 1, t_ctl, c_ctl)
        group_best_control_seq = torch.take_along_dim(
            cand_ctl_perm, idx_ctl, dim=2).squeeze(2)

    # -------------------------
    # (3) best 갱신
    # -------------------------
    if best_traj is None or best_dist is None:
        return group_best_traj, group_best_dist, group_best_control_seq

    better = group_best_dist < best_dist  # (B, 1+Pnn)
    if torch.any(better):
        best_traj[better] = group_best_traj[better]
        if best_control_seq is not None and group_best_control_seq is not None:
            best_control_seq[better] = group_best_control_seq[better]

    best_dist = torch.where(better, group_best_dist, best_dist)
    return best_traj, best_dist, best_control_seq



from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn


def _select_best_trajectory_by_sample_k(
    *,
    args: Any,
    model: nn.Module,
    norm_inputs_step: Dict[str, Any],
    state_normalizer: Any,
    unnorm_outputs_copy: Dict[str, torch.Tensor],
    unnorm_agent_length_m: torch.Tensor,  # (B, 1+Pnn)
    unnorm_agent_width_m: torch.Tensor,  # (B, 1+Pnn)
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    gap: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """같은 입력에서 후보 K개를 만들고, 규칙에 따라 최종 1개를 고릅니다.

    변경점(핵심)
    ----------
    - best traj를 고른 후보 인덱스(idx)를 그대로 써서,
      (pose_based=False인 경우) best control_seq도 같은 후보에서 함께 뽑습니다.
    - pose_based=True면 control_seq는 None을 유지합니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
            (best_normed_traj, best_distance_m_per_agent, best_control_seq)
            - best_normed_traj: (B, 1+Pnn, 1+future_len, 4)
            - best_distance_m_per_agent: (B, 1+Pnn)
            - best_control_seq:
                - pose_based=True  -> None
                - pose_based=False -> (B, 1+Pnn, future_len, C)  (보통 C=3)
    """
    select_jointly: bool = bool(getattr(args, "select_jointly", False))

    # K(후보 개수)
    fine_tune_gen_k_raw = getattr(args, "fine_tune_gen_k", 1)
    try:
        fine_tune_gen_k = int(fine_tune_gen_k_raw)
    except (TypeError, ValueError):
        fine_tune_gen_k = 1
    fine_tune_gen_k = int(max(1, fine_tune_gen_k))

    seed_stride = 10_000_000

    try:
        # cand_traj_batch: (K, B, 1+Pnn, 1+future_len, 4)
        # cand_dist_batch: (K, B, 1+Pnn)
        # target_future_control_seq: (K, B, 1+Pnn, future_len, C) or None
        (cand_traj_batch, cand_dist_batch,
         target_future_control_seq) = _forward_and_score_candidate_batch(
             args=args,
             model=model,
             norm_inputs_step=norm_inputs_step,
             state_normalizer=state_normalizer,
             unnorm_outputs_copy=unnorm_outputs_copy,
             unnorm_agent_length_m=unnorm_agent_length_m,
             unnorm_agent_width_m=unnorm_agent_width_m,
             batch_size=int(batch_size),
             one_or_pnn=int(one_or_pnn),
             future_len=int(future_len),
             rollout_idx=int(rollout_idx),
             base_seed=int(base_seed),
             ddp_rank=int(ddp_rank),
             step_idx=int(step_idx),
             cand_start_idx=0,
             cand_count=int(fine_tune_gen_k),
             seed_stride=int(seed_stride),
             gap=int(gap),
         )
    except BaseException as e:
        if _is_gpu_oom_error(e):
            _clear_gpu_cache_after_oom()
            raise RuntimeError(
                "GPU 메모리가 부족해서 fine_tune_gen_k 후보를 한 번에 처리하지 못했습니다. "
                f"fine_tune_gen_k={fine_tune_gen_k}. "
                "이 설정을 낮추거나, 후보별 버퍼를 안전하게 관리하는 방식이 필요합니다."
            ) from e
        raise

    best_control_seq: Optional[torch.Tensor] = None

    if select_jointly:
        # (B, 1+Pnn)
        target_agent_current_is_valid = _build_target_agent_current_is_valid_mask(
            norm_inputs_step=norm_inputs_step,
        )
        best_traj, best_dist, _best_score, best_control_seq = _select_best_from_candidate_batch_jointly(
            best_traj=None,
            best_dist=None,
            best_score=None,
            best_control_seq=None,
            cand_traj=cand_traj_batch,  # (K, B, 1+Pnn, 1+future_len, 4)
            cand_dist=cand_dist_batch,  # (K, B, 1+Pnn)
            cand_control_seq=target_future_control_seq,  # (K, B, 1+Pnn, future_len, C) or None
            agent_current_is_valid=target_agent_current_is_valid,  # (B, 1+Pnn)
        )
    else:
        best_traj, best_dist, best_control_seq = _select_best_from_candidate_batch_per_agent(
            best_traj=None,
            best_dist=None,
            best_control_seq=None,
            cand_traj=cand_traj_batch,  # (K, B, 1+Pnn, 1+future_len, 4)
            cand_dist=cand_dist_batch,  # (K, B, 1+Pnn)
            cand_control_seq=target_future_control_seq,  # (K, B, 1+Pnn, future_len, C) or None
        )

    # (메모리 압박 완화)
    del cand_traj_batch
    del cand_dist_batch
    del target_future_control_seq

    return best_traj, best_dist, best_control_seq



def _build_recovery_step_count_per_agent(
        *,
        args: Any,
        reference_tensor: torch.Tensor,  # (B_all, 1+Pnn)
) -> torch.Tensor:  # (B_all, 1+Pnn)
    """Recovery에서 '몇 스텝까지' GT 쪽으로 섞을지 에이전트별로 정합니다.

    """
    time_step_for_recover = int(args.time_step_for_recover)
    cfg = torch.full_like(reference_tensor,
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
    unnorm_best_fut_traj: torch.Tensor,  # (B_all, 1+Pnn, future_len, 4)
    gt_future: torch.Tensor,  # (B_all, 1+Pnn, future_len, 4)
    target_future_gt_is_valid: torch.Tensor,  # (B_all, 1+Pnn, future_len) bool
    lambda_: torch.Tensor,  # (B_all, 1+Pnn, future_len, 1) float32
) -> torch.Tensor:  # (B_all, 1+Pnn, future_len, 4)
    """예측 미래와 GT 미래를 섞되, GT가 없는 칸은 예측 값을 유지합니다.
    """
    pred_f = unnorm_best_fut_traj.to(dtype=torch.float32)
    gt_f = gt_future.to(dtype=torch.float32)

    mixed_f = (1.0 - lambda_) * pred_f + lambda_ * gt_f  # (B_all, 1+Pnn, T, 4)

    # ✅ GT가 없는 칸은 섞지 않고 pred 유지
    mixed_f = torch.where(target_future_gt_is_valid[..., None], mixed_f, pred_f)

    return mixed_f.to(dtype=unnorm_best_fut_traj.dtype)


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
    unnorm_best_traj: torch.Tensor,  # (B, 1+Pnn, 1+future_len, 4)
    expert_distance_m: torch.Tensor,  # (B_all, 1+Pnn)
    unnorm_outputs_copy: Dict[str, torch.Tensor],
    target_future_gt_is_valid: torch.Tensor,  # (B_all, 1+Pnn, future_len) bool
    target_current_is_valid_expand: torch.
    Tensor,  # (B_all, 1+Pnn, 1+future_len) bool
    future_len: int,
) -> torch.Tensor:  # (B_all, 1+Pnn, 1+future_len, 4), (B_all, 1+Pnn, future_len) bool
    """선택된 경로가 GT에서 너무 멀면, 에이전트별로 GT 쪽으로 부드럽게 섞습니다.

    ----------
    - GT 유효 길이를 'ego만' 보지 않고, (샘플, 에이전트)별로 계산해 반영합니다.
    - GT가 0으로 비어있는 시간 구간은 섞지 않고 예측 값을 그대로 유지합니다.

    Returns:
        torch.Tensor:
            복구가 반영된 경로(원래 단위).
            shape: (B_all, 1+Pnn, 1+future_len, 4)
    """

    # ✅ (중요) GT 유효 스텝 수를 에이전트별로 계산
    # target_gt_valid: (B_all, 1+Pnn)
    target_gt_valid = target_future_gt_is_valid.sum(dim=-1).to(dtype=torch.long)

    if not args.use_recovery:
        return unnorm_best_traj

    recovery_need_mask = expert_distance_m > float(
        args.recovery_threshold_m)  # (B_all, 1+Pnn)

    if not torch.any(recovery_need_mask):
        return unnorm_best_traj

    pred_avail_len = int(unnorm_best_traj.shape[2]) - 1
    if int(pred_avail_len) != int(future_len):
        raise ValueError(
            "unnorm_selected_traj의 미래 길이와 future_len이 다릅니다. "
            f"pred_avail_len={pred_avail_len}, future_len={future_len}")

    device = unnorm_best_traj.device

    # unnorm_best_fut_traj: (B_all, 1+Pnn, future_len, 4)
    unnorm_best_fut_traj = unnorm_best_traj[:, :, 1:, :]

    # GT가 아예 없는 에이전트는 recovery 대상에서 제외(논리적으로 더 안전)
    recovery_need_mask = recovery_need_mask & (target_gt_valid > 0)
    if not torch.any(recovery_need_mask):
        return unnorm_best_traj

    # n_recovery: (B_all, 1+Pnn)
    n_recovery = _build_recovery_step_count_per_agent(
        args=args,
        reference_tensor=target_gt_valid,  # (B_all, 1+Pnn)
    )

    # lambda_: (B_all, 1+Pnn, future_len, 1)
    lambda_ = _build_recovery_lambda_per_agent(
        n_rec_per_agent=n_recovery,
        future_len=int(future_len),
        device=device,
    )

    unnorm_ego_future_gt_4_dim = unnorm_outputs_copy["ego_future_gt_4_dim"]
    unnorm_near_future_gt_4_dim = unnorm_outputs_copy["near_future_gt_4_dim"]
    # unnorm_target_future_gt_4_dim: (B_all, 1+Pnn, future_len, 4)
    unnorm_target_future_gt_4_dim = torch.cat(
        [
            unnorm_ego_future_gt_4_dim[:, None, :, :],
            unnorm_near_future_gt_4_dim
        ],
        dim=1,
    )
    # mixed_future: (B_all, 1+Pnn, future_len, 4)
    mixed_future = _mix_pred_and_gt_future_for_recovery(
        unnorm_best_fut_traj=
        unnorm_best_fut_traj,  # (B_all, 1+Pnn, future_len, 4)
        gt_future=unnorm_target_future_gt_4_dim,  # (B_all, 1+Pnn, future_len, 4)
        target_future_gt_is_valid=
        target_future_gt_is_valid,  # (B_all, 1+Pnn, future_len)
        lambda_=lambda_,  #  (B_all, 1+Pnn, future_len, 1)
    )

    # recovery_need_mask 적용: (B_all, 1+Pnn, 1, 1)
    trigger_b = recovery_need_mask[:, :, None, None]
    # out_future: (B_all, 1+Pnn, future_len, 4)
    out_future = torch.where(trigger_b, mixed_future, unnorm_best_fut_traj)

    # (cos, sin) 정리는 "0이 아닌 칸"에만
    _normalize_heading_cos_sin_in_pose_4_dim_inplace(
        pose_4_dim=out_future,  # (B_all, 1+Pnn, future_len, 4)
        valid_mask=
        target_current_is_valid_expand[:, :, 1:],  # (B_all, 1+Pnn, future_len)
    )
    out = unnorm_best_traj.clone()  # (B, 1+Pnn, 1+future_len, 4)
    out[:, :, 1:, :] = out_future

    return out

def _traj11_to_traj3_heading(traj_11: torch.Tensor) -> torch.Tensor:
    """(x,y,cos,sin,...) 상태열에서 (x,y,heading)으로 바꿉니다.

    주의:
        - 입력은 마지막 차원에 최소 4개(x,y,cos,sin)가 있어야 합니다.
        - 이름은 traj11이지만, 실제로는 (..., >=4)도 허용합니다.

    Args:
        traj_11 (torch.Tensor): 상태열. shape: (..., C) where C>=4

    Returns:
        torch.Tensor: (x,y,heading) 상태열. shape: (..., 3)
    """
    if not isinstance(traj_11, torch.Tensor):
        raise TypeError(f"traj_11은 torch.Tensor여야 합니다. got {type(traj_11)}")
    if traj_11.dim() < 1:
        raise ValueError(f"traj_11은 최소 1차원이어야 합니다. got dim={traj_11.dim()}")
    if int(traj_11.shape[-1]) < 4:
        raise ValueError(
            "traj_11 마지막 차원은 최소 4(x,y,cos,sin)이어야 합니다. "
            f"got shape={tuple(traj_11.shape)}"
        )

    x = traj_11[..., 0]
    y = traj_11[..., 1]
    cos_h = traj_11[..., 2]
    sin_h = traj_11[..., 3]
    heading = torch.atan2(sin_h, cos_h)
    return torch.stack([x, y, heading], dim=-1)

from typing import Tuple
import torch


def _wrap_to_pi(delta: torch.Tensor) -> torch.Tensor:
    """각도 차이를 (-pi, pi] 범위로 접습니다.

    Args:
        delta (torch.Tensor): 각도 차이(라디안). shape: (...,)

    Returns:
        torch.Tensor: (-pi, pi] 범위로 접힌 각도 차이. shape: (...,)
    """
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _normalize_cos_sin(
    cos_seq: torch.Tensor,
    sin_seq: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) 쌍을 길이 1이 되도록 정리합니다.

    Args:
        cos_seq (torch.Tensor): cos 값들. shape: (...)
        sin_seq (torch.Tensor): sin 값들. shape: (...)
        eps (float): 0 나눗셈 방지 값. shape: ()

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            (cos_norm, sin_norm) - 입력과 동일한 shape
    """
    r = torch.sqrt(cos_seq * cos_seq + sin_seq * sin_seq + float(eps))
    cos_norm = cos_seq / r
    sin_norm = sin_seq / r
    return cos_norm, sin_norm
from typing import Union

def _to_scalar_dt(value: Union[float, np.ndarray], ref: NDArray[np.generic]) -> np.floating:
    """dt를 ref와 같은 dtype의 '스칼라'로 정리합니다."""
    dt_arr = np.asarray(value, dtype=ref.dtype)
    if dt_arr.size != 1:
        raise ValueError(f"dt는 스칼라여야 합니다. got shape={dt_arr.shape}, size={dt_arr.size}")
    return dt_arr.reshape(()).item()

def differentiate_numpy_pose3_to_control3(
    cur_future_pose_gt_3_dim: torch.Tensor,  # (B, P, 1+T, 3)
    dt: float,
    *,
    eps: float = 1e-8,
    normalize_yaw: bool = True,
    wrap_heading: bool = True,
) -> torch.Tensor:
    """(x,y,heading) 상태열에서 구간 제어(vx_b, vy_b, yaw_rate)를 만듭니다 (torch/batch 지원).

    입력:
        cur_future_pose_gt_3_dim:
            - shape: (B, P, 1+T, 3)
            - 마지막 3: (x, y, heading[rad])

    출력:
        future_seg_control_gt_3_dim:
            - shape: (B, P, T, 3)
            - 마지막 3: (v_x^b, v_y^b, yaw_rate)
    """
    if not isinstance(cur_future_pose_gt_3_dim, torch.Tensor):
        raise TypeError("cur_future_pose_gt_3_dim은 torch.Tensor여야 합니다.")
    if cur_future_pose_gt_3_dim.dim() != 4 or int(cur_future_pose_gt_3_dim.shape[-1]) != 3:
        raise ValueError(
            "cur_future_pose_gt_3_dim은 (B, P, 1+T, 3)이어야 합니다. "
            f"got shape={tuple(cur_future_pose_gt_3_dim.shape)}"
        )

    dt_f = float(dt)
    if (not torch.isfinite(torch.tensor(dt_f))) or dt_f <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt_f}")

    pose = cur_future_pose_gt_3_dim.to(dtype=torch.float32)

    # (B, P, 1+T)
    x = pose[..., 0]
    y = pose[..., 1]
    heading = pose[..., 2]

    # (B, P, T)
    x0, x1 = x[..., :-1], x[..., 1:]
    y0, y1 = y[..., :-1], y[..., 1:]
    th0, th1 = heading[..., :-1], heading[..., 1:]

    # yaw_rate
    delta_theta = th1 - th0  # (B, P, T)
    if wrap_heading:
        delta_theta = _wrap_to_pi(delta_theta)
    yaw_rate = delta_theta / dt_f  # (B, P, T)

    # 중간 방향
    th_mid = th0 + 0.5 * delta_theta
    cos_mid = torch.cos(th_mid)
    sin_mid = torch.sin(th_mid)
    if normalize_yaw:
        cos_mid, sin_mid = _normalize_cos_sin(cos_mid, sin_mid, eps=float(eps))

    # world 속도
    vwx = (x1 - x0) / dt_f
    vwy = (y1 - y0) / dt_f

    # world -> body (중간 방향 기준)
    vx_b = cos_mid * vwx + sin_mid * vwy
    vy_b = -sin_mid * vwx + cos_mid * vwy

    out = torch.stack([vx_b, vy_b, yaw_rate], dim=-1)  # (B, P, T, 3)
    return out.to(dtype=cur_future_pose_gt_3_dim.dtype)

from typing import Tuple
import torch


def _build_generated_demo_futures_from_selected_traj(
    *,
    unnorm_selected_gt_traj: torch.Tensor,  # (B, 1+Pnn, 1+future_len, C>=4)
    target_cur_fut_gt_is_valid: torch.Tensor,  # (B, 1+Pnn, 1+future_len) bool/0-1
    dt: float = 0.1,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """선택된 경로로 저장용 미래 포즈와 미래 구간 제어를 만듭니다.

    유효 기준(단일 기준)
    - frame_is_valid는 target_cur_fut_gt_is_valid를 그대로 사용합니다.
      shape: (B, 1+Pnn, 1+future_len)
    - 구간(seg) 유효는 양 끝 프레임이 둘 다 유효할 때만 True 입니다.
      seg_is_valid[t] = frame_is_valid[t] AND frame_is_valid[t+1]
      shape: (B, 1+Pnn, future_len)
    - seg_is_valid=False인 구간의 control은 0으로 고정합니다.

    Args:
        unnorm_selected_gt_traj (torch.Tensor):
            선택된 (현재+미래) 포즈 시퀀스.
            shape: (B, 1+Pnn, 1+future_len, C>=4)
            마지막 차원 앞 4개는 (x, y, cos, sin) 이라고 가정합니다.
        target_cur_fut_gt_is_valid (torch.Tensor):
            (현재 포함) GT 존재 여부 마스크.
            shape: (B, 1+Pnn, 1+future_len)
        dt (float):
            시간 간격(초). shape: ()
        eps (float):
            0 나눗셈 방지용 작은 값. shape: ()

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - ego_future_gt_4_dim:  (B, future_len, 4)
            - near_future_gt_4_dim: (B, Pnn, future_len, 4)
            - future_seg_control_gt_3_dim: (B, 1+Pnn, future_len, 3)
              마지막 3은 (v_x^b, v_y^b, yaw_rate)
    """
    if not isinstance(unnorm_selected_gt_traj, torch.Tensor):
        raise TypeError("unnorm_selected_gt_traj는 torch.Tensor여야 합니다.")
    if unnorm_selected_gt_traj.dim() != 4:
        raise ValueError(
            "unnorm_selected_gt_traj는 (B, 1+Pnn, 1+future_len, C) 이어야 합니다. "
            f"got shape={tuple(unnorm_selected_gt_traj.shape)}"
        )
    if int(unnorm_selected_gt_traj.shape[-1]) < 4:
        raise ValueError(
            "unnorm_selected_gt_traj 마지막 차원은 최소 4(x,y,cos,sin)이어야 합니다. "
            f"got shape={tuple(unnorm_selected_gt_traj.shape)}"
        )

    if not isinstance(target_cur_fut_gt_is_valid, torch.Tensor):
        raise TypeError("target_cur_fut_gt_is_valid는 torch.Tensor여야 합니다.")

    b = int(unnorm_selected_gt_traj.shape[0])
    one_pnn = int(unnorm_selected_gt_traj.shape[1])
    t_all = int(unnorm_selected_gt_traj.shape[2])  # 1+future_len
    future_len = int(t_all - 1)

    if future_len <= 0:
        raise ValueError(f"future_len은 1 이상이어야 합니다. got 1+future_len={t_all}")

    # target_cur_fut_gt_is_valid: (B, 1+Pnn, 1+future_len)
    frame_is_valid = target_cur_fut_gt_is_valid.to(dtype=torch.bool)
    if frame_is_valid.dim() != 3:
        raise ValueError(
            "target_cur_fut_gt_is_valid는 (B, 1+Pnn, 1+future_len) 이어야 합니다. "
            f"got shape={tuple(frame_is_valid.shape)}"
        )
    if (int(frame_is_valid.shape[0]) != b
            or int(frame_is_valid.shape[1]) != one_pnn
            or int(frame_is_valid.shape[2]) != t_all):
        raise ValueError(
            "target_cur_fut_gt_is_valid shape가 unnorm_selected_gt_traj와 맞지 않습니다. "
            f"traj={tuple(unnorm_selected_gt_traj.shape[:3])}, "
            f"valid={tuple(frame_is_valid.shape)}"
        )

    dt_f = float(dt)
    if (not torch.isfinite(torch.tensor(dt_f))) or dt_f <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt_f}")

    # ------------------------------------------------------------
    # (1) 저장용 미래 포즈(4차원)
    # ------------------------------------------------------------
    selected_future_pose4 = unnorm_selected_gt_traj[:, :, 1:, 0:4]  # (B, 1+Pnn, future_len, 4)
    demo_ego = selected_future_pose4[:, 0, :, :]    # (B, future_len, 4)
    demo_near = selected_future_pose4[:, 1:, :, :]  # (B, Pnn, future_len, 4)

    # ------------------------------------------------------------
    # (2) seg 유효: 양 끝 프레임이 둘 다 유효일 때만 True
    #     seg_is_valid: (B, 1+Pnn, future_len)
    # ------------------------------------------------------------
    seg_is_valid = frame_is_valid[..., :-1] & frame_is_valid[..., 1:]

    # ------------------------------------------------------------
    # (3) control 계산: pose(4) -> (x,y,heading) -> control(3)
    #     pose3: (B, 1+Pnn, 1+future_len, 3)
    #     control: (B, 1+Pnn, future_len, 3)
    # ------------------------------------------------------------
    pose4_for_heading = unnorm_selected_gt_traj[..., 0:4]  # (B, 1+Pnn, 1+future_len, 4)
    pose3 = _traj11_to_traj3_heading(pose4_for_heading)  # (B, 1+Pnn, 1+future_len, 3)

    future_seg_control_gt_3_dim = differentiate_numpy_pose3_to_control3(
        pose3,
        dt=dt_f,
        eps=float(eps),
        normalize_yaw=True,
        wrap_heading=True,
    )  # (B, 1+Pnn, future_len, 3)

    # ------------------------------------------------------------
    # (4) 무효 구간은 0으로 고정
    # ------------------------------------------------------------
    future_seg_control_gt_3_dim = future_seg_control_gt_3_dim * seg_is_valid.unsqueeze(-1).to(
        dtype=future_seg_control_gt_3_dim.dtype
    )

    return demo_ego, demo_near, future_seg_control_gt_3_dim



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
        a_unnorm_selected_gt_traj_np: np.ndarray,  # ((1+)Pnn, 1+T, 4)
        a_unnorm_ego_future_gt_4_dim: np.ndarray,  # (future_len, 4)
        a_unnorm_near_future_gt_3_dim: np.ndarray,  # (Pnn, future_len, 3)
        step_idx: int,
        draw_near_target_id: torch.Tensor,  # (Pnn,)
) -> None:
    output_data = {}

    ego_current = a_unnorm_inputs_np["ego_agent_past"][-1]  # (11,)
    ego_future_11 = _get_ego_future_11(
        ego_current, a_unnorm_selected_gt_traj_np)  # (1+T, 11)
    output_data["ego_np_int_traj_11_wrt_ego"] = ego_future_11
    output_data["ego_next_wp_wrt_ego"] = ego_future_11[1]

    ego_gt_future_11 = ego_future_11[1:, :].copy()  # (T, 11)
    ego_gt_future_11[:, 0:4] = a_unnorm_ego_future_gt_4_dim  # (future_len, 4)
    a_unnorm_inputs_np["ego_future_gt_11_dim"] = ego_gt_future_11

    near_agents_current = a_unnorm_inputs_np[
        "near_agents_past"][:, -1, :]  # (Pnn, 11)
    near_future_11 = _get_near_future_11(
        near_agents_current, a_unnorm_selected_gt_traj_np)  # (Pnn, 1+T, 11)

    # ✅ key를 항상 "17" 같은 형태로 만들기
    diff_token_to_np_int_traj_11_wrt_ego: Dict[str, np.ndarray] = {}
    for target_id, np_int_traj_11 in zip(draw_near_target_id, near_future_11):
        key = _format_id_for_draw_key(target_id)
        diff_token_to_np_int_traj_11_wrt_ego[key] = np_int_traj_11
    output_data[
        "diff_token_to_np_int_traj_11_wrt_ego"] = diff_token_to_np_int_traj_11_wrt_ego

    diff_token_to_future_gt_3_dim: Dict[str, np.ndarray] = {}
    for target_id, gt_future_3 in zip(draw_near_target_id,
                                      a_unnorm_near_future_gt_3_dim):
        key = _format_id_for_draw_key(target_id)
        diff_token_to_future_gt_3_dim[key] = gt_future_3  # (future_len, 3)
    a_unnorm_inputs_np[
        "diff_token_to_future_gt_3_dim"] = diff_token_to_future_gt_3_dim

    draw_machine_fast.draw_world_model_to_png(
        a_unnorm_inputs_np,
        output_data=output_data,
        save_path=os.path.join(save_dir, f"{step_idx}.png"),
    )


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
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    one_or_pnn: int,
    future_len: int,
    rollout_idx: int,
    base_seed: int,
    ddp_rank: int,
    step_idx: int,
    pose_based: bool,
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
    Returns:
        torch.Tensor:
            무작위 텐서.
            shape:
              - pose_based=True  -> (B, 1+Pnn, future_len, 4)
              - pose_based=False -> (B, 1+Pnn, future_len, 3)
    """
    last_dim = 4 if bool(pose_based) else 3
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
        (int(batch_size), int(one_or_pnn), int(future_len), last_dim),
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
    inputs: Dict[str, torch.Tensor],
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

    inputs = _sanitize_norm_inputs_for_validation(inputs)
    future_len: int = int(getattr(args, "future_len"))
    # target_agents_current_valid: (B, (1+)Pnn)
    # target_future_valid :  (B, (1+)Pnn, future_len)
    (target_agents_current_valid,
     target_future_valid) = build_target_future_tensors_and_masks_for_inference(
         args,
         inputs,
         future_len,
     )
    inputs["target_future_valid"] = target_future_valid

    rollout_number, base_seed, ddp_rank = _get_rollout_settings_for_validation(
        args)

    _update_validation_heartbeat_stage(
        args,
        f"{tag} | predicting rollouts sequentially (rollout={rollout_number})")

    # ✅ rollout을 묶어서 처리하지 않고, 1개씩 순차 실행
    _predict_rollouts_sequential(
        args=args,
        model=inference_model,
        inputs=inputs,
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


def _build_norm_inputs_from_unnorm_inputs(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_outputs_copy: Dict[str, Any],
    state_normalizer: "StateNormalizer",
    observation_normalizer: "ObservationNormalizer",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """현재 unnorm 입력을 모델 입력용 norm dict로 만듭니다.

    목표
    ----
    - 좌표 변환은 unnorm에서 수행합니다.
    - 모델 입력은 norm 값이 필요하므로, step마다 unnorm -> norm을 1번만 합니다.
      (기존처럼 norm->unnorm->norm 왕복을 하지 않습니다)


    Returns:
        Dict[str, Any]:
            정규화된 입력 dict(모델 forward에 넣을 dict).
    """
    norm_inputs_step: Dict[str,
                           Any] = observation_normalizer(unnorm_inputs_copy)
    norm_outputs_step = {k: v.clone() for k, v in unnorm_outputs_copy.items()}
    norm_outputs_step["ego_future_gt_4_dim"] = state_normalizer(
        data=unnorm_outputs_copy["ego_future_gt_4_dim"],
        valid_mask=unnorm_outputs_copy["ego_future_gt_is_valid"])
    # (B, Pnn, future_len, 4)
    norm_outputs_step["near_future_gt_4_dim"] = state_normalizer(
        data=unnorm_outputs_copy["near_future_gt_4_dim"],
        valid_mask=unnorm_outputs_copy["near_future_gt_is_valid"])
    # (B, (1+)Pnn, future_len, 3)
    norm_outputs_step["future_seg_control_gt_3_dim"] = state_normalizer(
        data=norm_outputs_step["future_seg_control_gt_3_dim"],
        valid_mask=norm_outputs_step["future_seg_control_is_valid"],
    )

    return norm_inputs_step, norm_outputs_step


def _build_cached_valid_masks_for_static_map_features(
    unnorm_inputs_copy: Dict[str, Any],) -> Dict[str, torch.Tensor]:

    cached: Dict[str, torch.Tensor] = {}
    lanes_len_is_valid = unnorm_inputs_copy.get("lanes_len_is_valid", None)
    if isinstance(lanes_len_is_valid,
                  torch.Tensor) and lanes_len_is_valid.numel() > 0:
        # lanes: (B, lane_num, lane_len)
        cached["lanes"] = lanes_len_is_valid.to(dtype=torch.bool)

    route_lanes_len_is_valid = unnorm_inputs_copy.get(
        "route_lanes_len_is_valid", None)
    if isinstance(route_lanes_len_is_valid,
                  torch.Tensor) and route_lanes_len_is_valid.numel() > 0:
        # route_lanes: (B, route_lane_num, lane_len, 12)
        cached["route_lanes"] = route_lanes_len_is_valid.to(dtype=torch.bool)

    static_objects_is_valid = unnorm_inputs_copy.get("static_objects_is_valid",
                                                     None)
    if isinstance(static_objects_is_valid,
                  torch.Tensor) and static_objects_is_valid.numel() > 0:
        # static_objects: (B, static_num, 10)
        cached["static_objects"] = static_objects_is_valid.to(dtype=torch.bool)

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


def _update_ego_past_and_valid_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_ego_pose_chunk: torch.Tensor,
) -> None:
    """1) ego past와 ego_agent_past_is_valid를 gap만큼 앞으로 당겨 업데이트합니다.

    Returns:
        None

    Notes:
        - ego_agent_past는 앞쪽 gap개를 버리고, 뒤에 gap개를 붙입니다.
        - 붙이는 gap개의 11차원 중 앞 4차원(x,y,cos,sin)만 새 예측으로 채우고,
          나머지는 "직전 프레임의 11차원"을 복사해서 유지합니다.
        - ego_agent_past_is_valid도 동일하게 앞쪽 gap을 버리고,
          뒤에 True(또는 1)로 채운 gap개를 붙입니다.
    """
    ego_agent_past = unnorm_inputs_copy["ego_agent_past"]  # (B, time_len, 11)

    gap = int(unnorm_ego_pose_chunk.shape[1])

    if gap <= 0:
        return

    # ego_current_11_dim: (B, 11)
    ego_current_11_dim = ego_agent_past[:, -1, :].clone()

    # unnorm_ego_chunk_11: (B, gap, 11)
    unnorm_ego_chunk_11 = ego_current_11_dim[:, None, :].expand(-1, gap,
                                                                -1).clone()
    unnorm_ego_chunk_11[:, :, 0:4] = unnorm_ego_pose_chunk  # (x,y,cos,sin)

    # ego_agent_past: (B, time_len, 11)
    unnorm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, gap:, :], unnorm_ego_chunk_11],
        dim=1,
    )

    ego_agent_past_is_valid = unnorm_inputs_copy[
        "ego_agent_past_is_valid"]  # (B, time_len)
    ego_agent_past_is_valid = ego_agent_past_is_valid.to(dtype=torch.bool)
    # ego_chunk_is_valid: (B, gap)
    ego_chunk_is_valid = torch.ones(
        (int(unnorm_ego_pose_chunk.shape[0]), gap),
        dtype=ego_agent_past_is_valid.dtype,
        device=ego_agent_past_is_valid.device,
    )
    # ego_agent_past_is_valid: (B, time_len)
    unnorm_inputs_copy["ego_agent_past_is_valid"] = torch.cat(
        [ego_agent_past_is_valid[:, gap:], ego_chunk_is_valid],
        dim=1,
    )


def _update_near_past_and_valid_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_near_pose_chunk: torch.Tensor,
) -> None:
    """2) near past와 near_agents_past_is_valid를 gap만큼 앞으로 당겨 업데이트합니다.

\    --------
    - 각 near agent의 "현재 시점(기존 past의 마지막)" valid를 그대로 이어 붙입니다.
      그래서 원래 존재하지 않던 agent(False)가 rollout이 진행되며 True로 바뀌는 일이 없습니다.
    - 추가로, 무효 agent는 새로 붙는 state 값도 0으로 유지해
      "값이 0이면 무효" 같은 후속 로직에서도 agent가 생기지 않게 합니다.

    Returns:
        None
    """
    near_agents_past = unnorm_inputs_copy[
        "near_agents_past"]  # (B, Pnn, time_len, 11)
    near_agents_past_is_valid = unnorm_inputs_copy[
        "near_agents_past_is_valid"]  # (B, Pnn, time_len)

    gap = int(unnorm_near_pose_chunk.shape[2])
    if gap <= 0:
        return

    # near_current: (B, Pnn, 11)
    near_current = near_agents_past[:, :, -1, :].clone()

    # near_chunk_11: (B, Pnn, gap, 11)
    near_chunk_11 = near_current[:, :, None, :].expand(-1, -1, gap, -1).clone()
    near_chunk_11[:, :, :, 0:4] = unnorm_near_pose_chunk  # (x,y,cos,sin)

    # ✅ (중요) 새로 붙이는 valid는 "현재 시점 valid"를 그대로 이어 붙입니다.
    # near_current_is_valid: (B, Pnn) bool
    near_current_is_valid = near_agents_past_is_valid[:, :,
                                                      -1].to(dtype=torch.bool)

    # near_chunk_is_valid: (B, Pnn, gap) bool
    near_chunk_is_valid = near_current_is_valid[:, :, None].expand(-1, -1, gap)

    # ✅ 무효 agent는 값도 0으로 유지(값 기반 무효 판정 로직까지 안전하게)
    # mask_f: (B, Pnn, gap, 1)
    mask_f = near_chunk_is_valid.to(dtype=near_chunk_11.dtype)[..., None]
    near_chunk_11 = near_chunk_11 * mask_f  # (B, Pnn, gap, 11)

    # near_agents_past: (B, Pnn, time_len, 11)
    unnorm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, gap:, :], near_chunk_11],
        dim=2,
    )

    # near_agents_past_is_valid: (B, Pnn, time_len)
    unnorm_inputs_copy["near_agents_past_is_valid"] = torch.cat(
        [near_agents_past_is_valid[:, :, gap:], near_chunk_is_valid],
        dim=2,
    )

    # (선택) 현재 시점 near 유효 여부를 여기서도 최신으로 맞춰 둠
    # near_agents_is_valid: (B, Pnn)
    unnorm_inputs_copy["near_agents_is_valid"] = near_current_is_valid.to(
        dtype=near_agents_past_is_valid.dtype).clone()


def _assert_non_near_agents_not_supported_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],) -> None:
    """3) non-near agent 입력이 들어온 경우, 현재 지원 조건을 검사합니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            입력 dict.
            - non_near_agents_past가 torch.Tensor로 존재할 수 있습니다.

    Returns:
        None

    Raises:
        AssertionError:
            non_near_agents_past가 존재하면서 agent 수 차원이 0이 아니면 에러.
    """
    non_near_agents_past = unnorm_inputs_copy.get("non_near_agents_past", None)
    if isinstance(non_near_agents_past, torch.Tensor):
        assert int(
            non_near_agents_past.shape[1]) == 0, "현재 non-near agent는 처리하지 않습니다."


def _update_neighbor_past_and_valid_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],) -> None:
    """4) neighbor past를 near와 동일하게 갱신하고, 관련 valid 플래그들을 맞춥니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            입력 dict(원래 단위, inplace 갱신).
            사용하는 키/shape:
              - neighbor_agents_past: (B, Pnn, time_len, 11) 또는 None
              - near_agents_past: (B, Pnn, time_len, 11)
              - near_agents_past_is_valid: (B, Pnn, time_len)
            생성/갱신하는 키:
              - neighbor_agents_past
              - neighbor_agents_past_is_valid
              - near_agents_is_valid: (B, Pnn)  # 마지막 time 스텝의 valid
              - neighbor_agents_is_valid: (B, Pnn)


    Returns:
        None
    """
    neighbor_agents_past = unnorm_inputs_copy.get("neighbor_agents_past", None)
    if neighbor_agents_past is None:
        unnorm_inputs_copy["neighbor_agents_past"] = None
        return

    near_agents_past = unnorm_inputs_copy["near_agents_past"]
    near_agents_past_is_valid = unnorm_inputs_copy["near_agents_past_is_valid"]

    # neighbor_agents_past: near를 그대로 복사
    unnorm_inputs_copy["neighbor_agents_past"] = near_agents_past.clone()

    # neighbor_agents_past_is_valid: near valid를 그대로 복사
    neighbor_agents_past_is_valid = near_agents_past_is_valid.clone()
    unnorm_inputs_copy[
        "neighbor_agents_past_is_valid"] = neighbor_agents_past_is_valid

    # neighbor_agents_is_valid / near_agents_is_valid: 마지막 시점 valid만 뽑아서 저장
    # neighbor_agents_is_valid: (B, Pnn)
    neighbor_agents_is_valid = neighbor_agents_past_is_valid[:, :, -1].clone()
    unnorm_inputs_copy["near_agents_is_valid"] = neighbor_agents_is_valid
    unnorm_inputs_copy["neighbor_agents_is_valid"] = neighbor_agents_is_valid


def _update_future_gt_and_valid_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_outputs_copy: Dict[str, Any],
    gap: int,
) -> None:
    """5) ego/near 미래 GT(및 planner)와 valid 플래그를 gap만큼 앞으로 당겨 업데이트합니다.

    Args:
        unnorm_outputs_copy (Dict[str, Any]):
            출력/정답 dict(원래 단위, inplace 갱신).
            필요한 키/shape:
              - ego_future_gt_4_dim: (B, future_len, 4)
              - near_future_gt_4_dim: (B, Pnn, future_len, 4)
              - ego_future_gt_is_valid: (B, future_len)
              - near_future_gt_is_valid: (B, Pnn, future_len)

        gap (int):
            이번에 한 번에 진행한 스텝 수. shape: ()

    Returns:
        None

    Notes:
        - 값 텐서는 time 축을 gap만큼 당기고, 뒤쪽 gap 구간은 0으로 채웁니다.
        - valid 플래그도 동일하게 당기고, 뒤쪽 gap 구간은 False(0)으로 채웁니다.
        - valid는 최종적으로 bool로 맞춥니다.
    """
    if int(gap) <= 0:
        return

    # ego_future_gt_4_dim shift
    ego_future_gt_4_dim = unnorm_outputs_copy[
        "ego_future_gt_4_dim"]  # (B, future_len, 4)
    future_len = int(ego_future_gt_4_dim.shape[1])
    ego_future_gt_4_dim[:, :future_len -
                        gap, :] = ego_future_gt_4_dim[:, gap:, :].clone()
    ego_future_gt_4_dim[:, future_len - gap:, :] = 0.0
    unnorm_outputs_copy["ego_future_gt_4_dim"] = ego_future_gt_4_dim

    # ego_future_gt_is_valid shift
    ego_future_gt_is_valid = unnorm_outputs_copy[
        "ego_future_gt_is_valid"]  # (B, future_len)
    ego_future_gt_is_valid[:, :future_len -
                           gap] = ego_future_gt_is_valid[:, gap:].clone()
    ego_future_gt_is_valid[:, future_len - gap:] = 0
    ego_future_gt_is_valid = ego_future_gt_is_valid.to(dtype=torch.bool)

    unnorm_outputs_copy["ego_future_gt_is_valid"] = ego_future_gt_is_valid

    # near future shift
    near_future_gt_4_dim = unnorm_outputs_copy[
        "near_future_gt_4_dim"]  # (B, Pnn, future_len, 4)
    near_future_gt_4_dim[:, :, :future_len -
                         gap, :] = near_future_gt_4_dim[:, :, gap:, :].clone()
    near_future_gt_4_dim[:, :, future_len - gap:, :] = 0.0
    unnorm_outputs_copy["near_future_gt_4_dim"] = near_future_gt_4_dim

    near_future_gt_is_valid = unnorm_outputs_copy[
        "near_future_gt_is_valid"]  # (B, Pnn, future_len)
    near_future_gt_is_valid[:, :, :future_len -
                            gap] = near_future_gt_is_valid[:, :, gap:].clone()
    near_future_gt_is_valid[:, :, future_len - gap:] = 0
    near_future_gt_is_valid = near_future_gt_is_valid.to(dtype=torch.bool)

    unnorm_outputs_copy["near_future_gt_is_valid"] = near_future_gt_is_valid

    # neighbor는 near와 같은 valid로 맞춤
    unnorm_inputs_copy["neighbor_future_gt_is_valid"] = near_future_gt_is_valid
    """
    future_seg_control_gt_3_dim: (B*R, (1+)Pnn, future_len, 3)
    future_seg_control_gt_3_dim 도 마찬가지로 shift 해주고, 뒤쪽 gap 구간은 0으로 채우고, bool로 맞춰야 합니다.
    """
    future_seg_control_gt_3_dim = unnorm_outputs_copy["future_seg_control_gt_3_dim"]
    future_seg_control_gt_3_dim[:, :, :future_len - gap, :] = future_seg_control_gt_3_dim[:, :, gap:, :].clone()
    future_seg_control_gt_3_dim[:, :, future_len - gap:, :] = 0.0
    unnorm_outputs_copy["future_seg_control_gt_3_dim"] = future_seg_control_gt_3_dim
    # future_seg_control_is_valid : (B*R, (1+)Pnn, future_len) 도 업데이트 해야함.
    """
    future_seg_control_is_valid 도 마찬가지로 shift 해주고, 뒤쪽 gap 구간은 0으로 채우고, bool로 맞춰야 합니다.
    """
    future_seg_control_is_valid = unnorm_outputs_copy[
        "future_seg_control_is_valid"]  # (B*R, (1+)Pnn, future_len)
    future_seg_control_is_valid[:, :, :future_len -
                                gap] = future_seg_control_is_valid[:, :, gap:].clone()
    future_seg_control_is_valid[:, :, future_len - gap:] = 0
    future_seg_control_is_valid = future_seg_control_is_valid.to(dtype=torch.bool)
    unnorm_outputs_copy["future_seg_control_is_valid"] = future_seg_control_is_valid


def _match_device_and_dtype(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """x를 ref와 같은 device/dtype으로 맞춥니다.

    Args:
        x (torch.Tensor): 맞출 텐서. shape: (임의, ...)
        ref (torch.Tensor): 기준 텐서. shape: (임의, ...)

    Returns:
        torch.Tensor:
            device/dtype이 ref와 동일한 텐서.
            shape: x와 동일
    """
    if x.device == ref.device and x.dtype == ref.dtype:
        return x
    return x.to(device=ref.device, dtype=ref.dtype)

def _update_target_seg_control_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_target_control_chunk: Optional[torch.Tensor],
) -> None:
    """이번 chunk의 예측 구간 제어를 past_seg_control_gt_3_dim에 반영합니다.

    추가 반영(요청사항)
    ----------------
    - 무효(패딩) agent가 control로 “살아나는” 걸 막기 위해,
      ego_chunk_is_valid + near_chunk_is_valid 로 control chunk를 0 마스킹한 뒤에만
      past_seg_control_gt_3_dim 에 누적합니다.
    - (추가 안전장치) past_ctrl과 dtype/device가 다르면, cat 전에 chunk 쪽을 past_ctrl에 맞춥니다.
      이미 같으면 아무 것도 하지 않습니다.

    Args:
        unnorm_inputs_copy (Dict[str, Any]):
            rollout 동안 유지하는 입력 dict.
            - past_seg_control_gt_3_dim: (B, 1+Pnn, past_len, 3)
            - ego_agent_past_is_valid: (B, time_len)
            - near_agents_past_is_valid: (B, Pnn, time_len)
        unnorm_target_control_chunk (Optional[torch.Tensor]):
            이번 chunk에서 예측한 control.
            - None이면 아무 것도 하지 않습니다.
            - shape: (B, 1+Pnn, gap, 3)

    Returns:
        None
    """
    if unnorm_target_control_chunk is None:
        return

    past_ctrl = unnorm_inputs_copy.get("past_seg_control_gt_3_dim", None)
    if not isinstance(past_ctrl, torch.Tensor):
        raise KeyError(
            "unnorm_target_control_chunk가 있는데 'past_seg_control_gt_3_dim' 키가 없습니다."
        )

    if past_ctrl.dim() != 4 or int(past_ctrl.shape[-1]) != 3:
        raise ValueError(
            "past_seg_control_gt_3_dim은 (B, 1+Pnn, past_len, 3) 이어야 합니다. "
            f"got shape={tuple(past_ctrl.shape)}"
        )

    if unnorm_target_control_chunk.dim() != 4 or int(unnorm_target_control_chunk.shape[-1]) != 3:
        raise ValueError(
            "unnorm_target_control_chunk는 (B, 1+Pnn, gap, 3) 이어야 합니다. "
            f"got shape={tuple(unnorm_target_control_chunk.shape)}"
        )

    if int(past_ctrl.shape[0]) != int(unnorm_target_control_chunk.shape[0]) or int(past_ctrl.shape[1]) != int(
        unnorm_target_control_chunk.shape[1]
    ):
        raise ValueError(
            "batch 축(B) 또는 agent 축(1+Pnn) 크기가 맞지 않습니다. "
            f"past={tuple(past_ctrl.shape[:2])}, chunk={tuple(unnorm_target_control_chunk.shape[:2])}"
        )

    past_len = int(past_ctrl.shape[2])
    gap = int(unnorm_target_control_chunk.shape[2])
    if past_len <= 0 or gap <= 0:
        return

    """
    """
    ego_agent_cur_is_valid = unnorm_inputs_copy["ego_agent_past_is_valid"][:, -1:] # (B*R, 1)
    near_agents_cur_is_valid = unnorm_inputs_copy["near_agents_past_is_valid"][:, :, -1] # (B, Pnn,)
    current_seg_control_is_valid = torch.cat([ego_agent_cur_is_valid, near_agents_cur_is_valid], dim=1) # (B*R, 1+Pnn)
    current_seg_control_is_valid = current_seg_control_is_valid.unsqueeze(-1) # (B*R, 1+Pnn, 1)
    # current_seg_chunk_control_is_valid :  (B*R, 1+Pnn, 1) -> expand -> (B*R, 1+Pnn, gap)
    current_seg_chunk_control_is_valid = current_seg_control_is_valid.expand(-1, -1, gap)
    # mask_f: (B*R, 1+Pnn, gap) -> (B*R, 1+Pnn, gap, 1)
    # 1) ego/near valid로 control chunk 마스킹
    mask_f = current_seg_chunk_control_is_valid.unsqueeze(-1)

    masked_control = unnorm_target_control_chunk * mask_f  # (B*R, 1+Pnn, gap, 3)

    # 2) ✅ dtype/device 불일치가 있으면 여기서만 맞춤 (이미 같으면 no-op)
    masked_control = _match_device_and_dtype(masked_control, past_ctrl)
    if gap <=0 or gap > past_len:
        raise ValueError(f"gap={gap}, past_len={past_len}")

    # 일반 케이스: 앞쪽 move개 버리고, 뒤에 move개 붙이기
    left = past_ctrl[:, :, gap:, :]               # (B*R, 1+Pnn, past_len-move, 3)
    unnorm_inputs_copy["past_seg_control_gt_3_dim"] = torch.cat([left, masked_control], dim=2)

    # past_seg_control_is_valid: (B*R, 1+Pnn, past_len) 이것도 업데이트 해야함
    past_seg_control_is_valid = unnorm_inputs_copy["past_seg_control_is_valid"] # (B*R, 1+Pnn, past_len)
    unnorm_inputs_copy["past_seg_control_is_valid"] = torch.cat(
        [past_seg_control_is_valid[:, :, gap:], current_seg_chunk_control_is_valid], dim=2
    )


def _update_merged_inputs_unnorm_inplace_for_time_chunk(
    unnorm_inputs_copy: Dict[str, Any],
    unnorm_outputs_copy: Dict[str, Any],
    unnorm_ego_pose_chunk: torch.Tensor,  # (B, gap, 4)
    unnorm_near_pose_chunk: torch.Tensor,  # (B, Pnn, gap, 4)
        unnorm_target_control_chunk: Optional[torch.Tensor],
        # (B, 1+Pnn, gap, 3)
        cached_valid_masks: Optional[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    1) 여러 스텝(gap개)을 한 번에 past/GT에 반영하고,
    2) 좌표 기준을 마지막 스텝으로 맞춥니다.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            (unnorm_inputs_copy, unnorm_outputs_copy)
    """
    gap = int(unnorm_ego_pose_chunk.shape[1])
    if gap <= 0:
        return unnorm_inputs_copy, unnorm_outputs_copy

    # 1) ego past/valid
    _update_ego_past_and_valid_inplace_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,
        unnorm_ego_pose_chunk=unnorm_ego_pose_chunk,
    )

    # 2) near past/valid (+ Pnn 획득)
    _update_near_past_and_valid_inplace_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,
        unnorm_near_pose_chunk=unnorm_near_pose_chunk,
    )
    # 3) non-near 지원 조건 확인
    _assert_non_near_agents_not_supported_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,)

    # 4) neighbor past/valid (near와 동일하게)
    _update_neighbor_past_and_valid_inplace_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,)
    # ✅ (추가) past_seg_control_gt_3_dim 업데이트  ( past_seg_control_is_valid 도 업데이트 해야함)
    _update_target_seg_control_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,
        unnorm_target_control_chunk=unnorm_target_control_chunk,  # (B, 1+Pnn, gap, 3) or None
    )
    # 5) 미래 GT/valid 갱신
    _update_future_gt_and_valid_inplace_for_time_chunk(
        unnorm_inputs_copy=unnorm_inputs_copy,
        unnorm_outputs_copy=unnorm_outputs_copy,
        gap=int(gap),
    )

    # 6) 좌표 기준 변환은 "마지막(gap번째) ego 포즈"로 1번만
    unnorm_ego_new_cur_pose = unnorm_ego_pose_chunk[:, -1, :]  # (B, 4)
    _transform_origin(
        unnorm_inputs_copy,
        unnorm_outputs_copy,
        unnorm_ego_new_cur_pose,
        cached_valid_masks=cached_valid_masks,
    )

    return unnorm_inputs_copy, unnorm_outputs_copy


def _transform_origin_step1_ego_past_inplace(
        unnorm_inputs_copy: Dict[str, Any],
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> None:
    """1) ego_agent_past를 새 기준으로 바꿉니다.

    Args:
        unnorm_inputs_copy: 입력 dict (inplace 변경).
            - ego_agent_past: (B, time_len, 11)
            - ego_agent_past_is_valid: (B, time_len)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)

    Returns:
        None
    """
    ego_agent_past = unnorm_inputs_copy["ego_agent_past"]
    ego_agent_past_is_valid = unnorm_inputs_copy["ego_agent_past_is_valid"]
    _transform_state_11_dim_inplace(
        state_11=ego_agent_past,  # (B, time_len, 11)
        delta_xy=delta_xy,  # (B, 2)
        cos_delta=cos_delta,  # (B,)
        sin_delta=sin_delta,  # (B,)
        valid_mask=ego_agent_past_is_valid,  # (B, time_len)
    )


def _transform_origin_step2_agents_past_inplace(
        unnorm_inputs_copy: Dict[str, Any],
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> None:
    """2) near/non_near/neighbor의 past를 새 기준으로 바꿉니다.

    Args:
        unnorm_inputs_copy: 입력 dict (inplace 변경).
            - near_agents_past: (B, Pnn, time_len, 11)
            - non_near_agents_past: (B, Nnn, time_len, 11) 또는 (B, 0, time_len, 11)
            - neighbor_agents_past: (B, Pnn, time_len, 11) 또는 None
            - 각 *_is_valid: (B, P?, time_len) (없을 수도 있음)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)

    Returns:
        None
    """
    for key, valid_key in (
        ("near_agents_past", "near_agents_past_is_valid"),
        ("non_near_agents_past", "non_near_agents_past_is_valid"),
        ("neighbor_agents_past", "neighbor_agents_past_is_valid"),
    ):
        agents_past = unnorm_inputs_copy.get(key, None)
        if not isinstance(agents_past,
                          torch.Tensor) or agents_past.numel() == 0:
            continue

        valid_mask = unnorm_inputs_copy[valid_key]

        _transform_state_11_dim_inplace(
            state_11=agents_past,  # (B, P?, time_len, 11)
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=valid_mask,  # (B, P?, time_len)
        )


def _transform_origin_step3_future_gt_inplace(
        unnorm_outputs_copy: Dict[str, Any],
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> None:
    """3) ego/near의 미래 GT(4차원 포즈)를 새 기준으로 바꿉니다.

    Args:
        unnorm_outputs_copy: 출력/정답 dict (inplace 변경).
            - ego_future_gt_4_dim: (B, future_len, 4)
            - near_future_gt_4_dim: (B, Pnn, future_len, 4)
            - 각 *_is_valid: (B, ...) (없을 수도 있음)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)

    Returns:
        None
    """
    for pose_key, valid_key in (
        ("ego_future_gt_4_dim", "ego_future_gt_is_valid"),
        ("near_future_gt_4_dim", "near_future_gt_is_valid"),
    ):
        pose_4 = unnorm_outputs_copy.get(pose_key, None)
        if not isinstance(pose_4, torch.Tensor) or pose_4.numel() == 0:
            continue

        valid_mask = unnorm_outputs_copy[valid_key]

        _transform_pose_4_dim_inplace(
            pose_4_dim=pose_4,  # (..., 4)
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=valid_mask,  # pose_4.shape[:-1]
        )


def _transform_origin_step5_points_inplace(
        unnorm_inputs_copy: Dict[str, Any],
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> None:
    """5) stop/crosswalk/speed_bump/driveway/road_edge 점들을 새 기준으로 바꿉니다.

    Args:
        unnorm_inputs_copy: 입력 dict (inplace 변경).
            - *_points: (B, N, L, 2) 등
            - *_is_valid: (B, N, L) 또는 (B, N)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)

    Returns:
        None
    """
    point_and_mask_keys = (
        ("stop_sign_points", "stop_sign_is_valid"),
        ("crosswalk_points", "crosswalk_is_valid"),
        ("speed_bump_points", "speed_bump_is_valid"),
        ("driveway_points", "driveway_is_valid"),
        ("road_edge", "road_edge_is_valid"),
    )

    for point_key, valid_key in point_and_mask_keys:
        points_xy = unnorm_inputs_copy.get(point_key, None)

        if not isinstance(points_xy, torch.Tensor) or points_xy.numel() == 0:
            continue
        points_valid = unnorm_inputs_copy[valid_key]

        # points_valid: (B, N) + points_xy: (B, N, L, 2) 인 케이스 대응
        if points_valid.dim() == 2 and points_xy.dim() == 4:
            points_valid = points_valid.unsqueeze(-1).expand_as(
                points_xy[..., 0])  # (B, N, L)

        _transform_points_2d_inplace(
            points_xy=points_xy,  # (B, ..., 2)
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=points_valid.to(dtype=torch.bool),  # (B, ...)
        )


def _transform_origin_step6_lanes_inplace(
    unnorm_inputs_copy: Dict[str, Any],
    delta_xy: torch.Tensor,  # (B, 2)
    cos_delta: torch.Tensor,  # (B,)
    sin_delta: torch.Tensor,  # (B,)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]],
) -> None:
    """6) lanes/route_lanes를 새 기준으로 바꿉니다(캐시 마스크 사용).

    Args:
        unnorm_inputs_copy: 입력 dict (inplace 변경).
            - lanes: (B, lane_num, lane_len, 12)
            - route_lanes: (B, route_lane_num, lane_len, 12)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)
        cached_valid_masks: 미리 만든 유효 마스크 dict.
            - "lanes": (B, lane_num, lane_len) bool
            - "route_lanes": (B, route_lane_num, lane_len) bool

    Returns:
        None
    """
    for lane_key in ("lanes", "route_lanes"):
        lane_12 = unnorm_inputs_copy.get(lane_key, None)
        if not isinstance(lane_12, torch.Tensor) or lane_12.numel() == 0:
            continue

        cached_mask = _pick_cached_mask_if_shape_matches(
            cached_valid_masks=cached_valid_masks,
            key=lane_key,
            expected_shape=lane_12.shape[:-1],  # (B, lane_num, lane_len)
        )
        if cached_mask is None:
            raise NotImplementedError(
                "현재 lanes/route_lanes의 유효 마스크는 캐시된 값이 반드시 필요합니다.")

        _transform_lane_12_dim_inplace(
            lane_12=lane_12,  # (B, ..., 12)
            delta_xy=delta_xy,  # (B, 2)
            cos_delta=cos_delta,  # (B,)
            sin_delta=sin_delta,  # (B,)
            valid_mask=cached_mask,  # (B, ...)
        )


def _transform_origin_step7_static_objects_inplace(
    unnorm_inputs_copy: Dict[str, Any],
    delta_xy: torch.Tensor,  # (B, 2)
    cos_delta: torch.Tensor,  # (B,)
    sin_delta: torch.Tensor,  # (B,)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]],
) -> None:
    """7) static_objects를 새 기준으로 바꿉니다(캐시 마스크 사용).

    Args:
        unnorm_inputs_copy: 입력 dict (inplace 변경).
            - static_objects: (B, static_num, 10)
        delta_xy: 새 기준의 위치 이동 값. shape (B, 2)
        cos_delta: 새 기준의 방향 cos 값. shape (B,)
        sin_delta: 새 기준의 방향 sin 값. shape (B,)
        cached_valid_masks: 미리 만든 유효 마스크 dict.
            - "static_objects": (B, static_num) bool

    Returns:
        None
    """
    static_objects = unnorm_inputs_copy.get("static_objects", None)
    if not isinstance(static_objects,
                      torch.Tensor) or static_objects.numel() == 0:
        return

    cached_mask = _pick_cached_mask_if_shape_matches(
        cached_valid_masks=cached_valid_masks,
        key="static_objects",
        expected_shape=static_objects.shape[:-1],  # (B, static_num)
    )
    if cached_mask is None:
        raise NotImplementedError(
            "현재 static_objects의 유효 마스크는 캐시된 값이 반드시 필요합니다.")

    _transform_static_object_10_dim_inplace(
        static_10=static_objects,  # (B, static_num, 10)
        delta_xy=delta_xy,  # (B, 2)
        cos_delta=cos_delta,  # (B,)
        sin_delta=sin_delta,  # (B,)
        valid_mask=cached_mask,  # (B, static_num)
    )


def _transform_origin(
    unnorm_inputs_copy: Dict[str, torch.Tensor],
    unnorm_outputs_copy: Dict[str, torch.Tensor],
    unnorm_ego_new_cur_pose: torch.Tensor,  # (B, 4)
    cached_valid_masks: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """unnorm_ego_new_cur_pose 기준으로 입력 전체의 좌표 기준을 바꿉니다."""
    (delta_xy, cos_delta, sin_delta,
     yaw_delta) = _extract_delta_pose_params(unnorm_ego_new_cur_pose)

    # 1) ego past
    _transform_origin_step1_ego_past_inplace(
        unnorm_inputs_copy=unnorm_inputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )

    # 2) agents past (near/non_near/neighbor)
    _transform_origin_step2_agents_past_inplace(
        unnorm_inputs_copy=unnorm_inputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )

    # 3) future GT (ego/near)
    _transform_origin_step3_future_gt_inplace(
        unnorm_outputs_copy=unnorm_outputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )

    # 5) points
    _transform_origin_step5_points_inplace(
        unnorm_inputs_copy=unnorm_inputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
    )

    # 6) lanes / route_lanes (캐시 마스크 필수)
    _transform_origin_step6_lanes_inplace(
        unnorm_inputs_copy=unnorm_inputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
        cached_valid_masks=cached_valid_masks,
    )

    # 7) static_objects (캐시 마스크 필수)
    _transform_origin_step7_static_objects_inplace(
        unnorm_inputs_copy=unnorm_inputs_copy,
        delta_xy=delta_xy,
        cos_delta=cos_delta,
        sin_delta=sin_delta,
        cached_valid_masks=cached_valid_masks,
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
    valid_mask = valid_mask.to(dtype=torch.bool)
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
    # ✅ (추가) 이 스크립트에서는 amortized diffusion을 항상 끔
    _force_disable_amortized_diffusion_for_finetune_data_maker(args)

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
