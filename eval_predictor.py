import argparse
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Any, Dict, Optional, List, Callable
import torch
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
)
import args_util
import os
import sys, faulthandler, traceback
from diffusion_planner.utils.train_utils import set_seed
from timm.utils import ModelEma
from diffusion_planner.utils import ddp
import time
from tqdm import tqdm
from diffusion_planner.utils.target_feature import \
    build_target_future_tensors_and_masks_for_inference

from diffusion_planner.train_epoch import _prepare_batch_for_device
from diffusion_planner.utils.normalizer import StateNormalizer
from waymo_open_dataset.protos import sim_agents_submission_pb2

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype
from src.utils.wosac_utils import get_scenario_id_int_tensor, \
    get_scenario_rollouts
from src.smart.metrics import WOSACMetrics, minADE
from torch import optim
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger


def model_validation(
    args: argparse.Namespace,
    global_rank: int,
    rank: int,
    world_size: int,
    use_deepspeed: bool,
) -> None:
    """전체 학습 파이프라인을 실행하는 상위 함수."""
    torch.cuda.empty_cache()

    # 2) seed 고정
    set_seed(args.seed + global_rank)

    # 3) augmentation, Dataset, Sampler
    batch_size = args.batch_size
    """
    validation_set: DiffusionPlannerData
    validation_sampler: DistributedSampler
    """
    validation_set, validation_sampler = build_dataset_and_sampler(
        args,
        args.validation_set,
        args.validation_set_list,
        "validation",
        world_size,
        global_rank,
    )
    validation_loader = build_data_loader(
        args,
        validation_set,
        validation_sampler,
        batch_size,
        world_size,
    )
    if args.ddp and not use_deepspeed:
        torch.distributed.barrier()

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

    # 6) 체크포인트 재개
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id,
     train_epochs, allow_val_change) = maybe_resume_from_checkpoint(
         args=args,
         diffusion_planner=diffusion_planner,
         optimizer=None,
         scheduler=None,
         model_ema=model_ema,
         global_rank=global_rank,
         use_deepspeed=use_deepspeed,
     )
    args._global_update_step = 0

    min_ade = minADE()
    wosac_metrics = WOSACMetrics("val_closed")

    run_validation_loop(
        args=args,
        diffusion_planner=diffusion_planner,
        model_ema=model_ema,
        validation_loader=validation_loader,
        wosac_metrics=wosac_metrics,
        min_ade=min_ade,
    )

    _finalize_training_cleanup()


def _finalize_training_cleanup():
    # 1) 분산 학습일 때만 barrier 호출
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def run_validation_loop(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    validation_loader: DataLoader,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
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
        min_ade=min_ade,
    )
    # print "epoch_elapsed_time_sec"
    print(f"[Validation] completed in "
          f"{epoch_elapsed_time_sec:.2f} sec.")


def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    batch_num_in_one_val_epoch: int,
    wosac_metrics: WOSACMetrics,
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
        min_ade,
    )
    """
CUDA를 쓰는 경우, PyTorch가 캐시로 잡아둔 GPU 메모리(다음 연산 재사용용)를 OS/GPU 쪽에 반납
메모리 피크를 낮춰 OOM 위험을 줄이려는 목적이지만, 너무 자주 하면 약간의 성능 손해가 날 수 있어요.
    """
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    """
torch.cuda.synchronize() : 현재 GPU에서 큐에 쌓인 비동기 연산이 다 끝날 때까지 대기
torch.distributed.barrier() :
    모든 rank(프로세스)가 이 지점에 도착할 때까지 서로 기다리게 해서, 어떤 rank만 앞서가거나(예: 저장/로그/다음 epoch) 뒤처지는 상황을 막습니다.
    """
    if args.ddp:
        torch.cuda.synchronize()
        torch.distributed.barrier()

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
    min_ade: minADE,
) -> Dict[str, torch.Tensor]:
    model.eval()
    if args.ddp:
        torch.cuda.synchronize()

    total_batch_steps = int(max(1, batch_num_in_one_val_epoch))

    progress_state: Optional[Dict[str, Any]] = None
    if _is_main_process_for_logging(args):
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
    ) as data_epoch:
        for batch_idx, batch in enumerate(data_epoch, start=1):
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
                min_ade=min_ade,
            )

            if args.ddp:
                torch.cuda.synchronize()

            if progress_state is not None:
                _maybe_print_progress_eta(
                    state=progress_state,
                    done_steps=batch_idx,
                    writer=data_epoch.write,
                )

    epoch_wosac_metrics: Dict[str, torch.Tensor] = wosac_metrics.compute()
    epoch_wosac_metrics["val_closed/ADE"] = min_ade.compute()
    wosac_metrics.reset()
    min_ade.reset()

    return epoch_wosac_metrics


def _select_inference_model_for_validation(
    model: nn.Module,
    ema: Optional[ModelEma],
) -> nn.Module:
    """검증 단계에서 실제로 예측을 만들 때 사용할 모델을 고릅니다.

    기본 아이디어
    ----------------
    - 학습 중에는 모델 파라미터가 계속 바뀌기 때문에, 평가할 때는
      "최근 파라미터" 그대로보다, "조금씩 평균낸 파라미터(EMA)"가 더 안정적인 경우가 많습니다.
    - 그래서 ema가 있으면 ema.ema 모델을 사용하고,
      ema가 없으면 원래 model을 그대로 사용합니다.

    Args:
        model (nn.Module):
            원래 학습/추론에 쓰는 모델.
        ema (Optional[ModelEma]):
            timm ModelEma 객체.
            - ema.ema: EMA 파라미터가 들어있는 nn.Module
            - None 이면 EMA를 쓰지 않습니다.

    Returns:
        nn.Module:
            실제 inference에 사용할 모델.
            - 보통 ema.ema 또는 model
    """
    if ema is None:
        return model

    ema_model = getattr(ema, "ema", None)
    if isinstance(ema_model, nn.Module):
        return ema_model

    return model


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


def _predict_rollouts_batched_one_chunk(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    state_normalizer: Any,
    rollout_repeat: int,
    rng_seed: int,
) -> torch.Tensor:
    """rollout을 batch로 펼쳐서(=B*R) 한 번에 autoregressive rollout을 생성합니다.

    이 함수가 하는 일
    ----------------
    - rollout_repeat개 만큼 입력을 복제해서 batch를 (B*R)로 키웁니다.
    - future_len 스텝 동안:
        1) 모델 forward 1번으로 (B*R)개 샘플의 다음 스텝 예측을 동시에 얻고,
        2) 그 다음 스텝을 입력 past에 반영(업데이트)하고,
        3) world 좌표로 바꿔서 결과를 저장합니다.

    반환 텐서 shape
    --------------
    - 반환: (B, (1+)Pnn, rollout_repeat, future_len, 4)
      마지막 4는 (x, y, cos, sin)

    Args:
        args (Any):
            - args.future_len (int)
            - args.device (str)
        model (nn.Module):
            eval 상태의 inference 모델.
        norm_inputs (Dict[str, Any]):
            원래 배치 입력.
            주요 텐서 예:
              - ego_agent_past: (B, T_past, 11)
              - near_agents_past: (B, Pnn, T_past, 11)
              - origin_world_pose: (B, 4)
              - target_future_valid: (B, 1+Pnn, future_len)
        state_normalizer (Any):
            state_normalizer.inverse(...)를 사용할 수 있어야 합니다.
            - 입력: (B*R, 1+Pnn, 4)
            - 출력: (B*R, 1+Pnn, 4)
        rollout_repeat (int):
            이번 chunk에서 동시에 만들 rollout 개수 R.
        rng_seed (int):
            이 chunk의 랜덤 seed.

    Returns:
        torch.Tensor:
            shape: (B, (1+)Pnn, rollout_repeat, future_len, 4)
    """
    device = torch.device(getattr(args, "device", "cuda"))

    ego_agent_past = norm_inputs.get("ego_agent_past", None)
    if not isinstance(ego_agent_past, torch.Tensor):
        raise ValueError("norm_inputs['ego_agent_past']가 torch.Tensor가 아닙니다.")
    batch_size: int = int(ego_agent_past.shape[0])

    target_future_valid = norm_inputs.get("target_future_valid", None)
    if not isinstance(target_future_valid, torch.Tensor):
        raise ValueError(
            "norm_inputs['target_future_valid']가 torch.Tensor가 아닙니다.")
    one_or_pnn: int = int(target_future_valid.shape[1])

    future_len: int = int(getattr(args, "future_len"))
    rollout_repeat = int(rollout_repeat)
    merged_batch: int = int(batch_size * rollout_repeat)

    # (B, ...) -> (B*R, ...)
    norm_inputs_copy: Dict[str, Any] = _expand_norm_inputs_for_rollout_batch(
        norm_inputs=norm_inputs,
        batch_size=batch_size,
        rollout_repeat=rollout_repeat,
    )

    origin_world_pose = norm_inputs_copy.get("origin_world_pose", None)
    if not isinstance(origin_world_pose, torch.Tensor):
        raise ValueError(
            "norm_inputs['origin_world_pose']가 torch.Tensor가 아닙니다.")
    # origin_world_pose: (B*R, 4)
    unnorm_origin_world_pose: torch.Tensor = origin_world_pose

    # 출력 미리 할당: (B*R, (1+)Pnn, future_len, 4)
    # dtype은 입력 기반으로 맞춥니다.
    out_dtype = ego_agent_past.dtype
    target_joint_scene_world = torch.empty(
        (merged_batch, one_or_pnn, future_len, 4),
        device=ego_agent_past.device,
        dtype=out_dtype,
    )

    rng_devices = _get_cuda_rng_devices_for_fork_rng(device)

    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        torch.manual_seed(int(rng_seed))
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rng_seed))

        with torch.inference_mode():
            for step_idx in range(future_len):
                decoder_output = _forward_model_for_validation(
                    args=args,
                    model=model,
                    norm_inputs=norm_inputs_copy,
                )

                # integrated_trajectory: (B*R, (1+)Pnn, 1+T, 4)
                normed_trajectories = decoder_output.get(
                    "integrated_trajectory", None)
                if normed_trajectories is None:
                    raise KeyError(
                        "decoder_output에서 'integrated_trajectory'를 찾을 수 없습니다. "
                        "현재 validate_func는 integrated_trajectory를 사용하도록 구현되어 있습니다."
                    )

                # ego: (B*R, 4)
                normed_ego_next_pose = normed_trajectories[:, 0, 1, :]
                # near: (B*R, Pnn, 4)
                normed_near_next_pose = normed_trajectories[:, 1:, 1, :]

                # 입력 업데이트(autoregressive)
                norm_inputs_copy = _update_merged_inputs(
                    norm_inputs_copy=norm_inputs_copy,
                    normed_ego_next_pose=normed_ego_next_pose,
                    normed_near_next_pose=normed_near_next_pose,
                )

                # (B*R, (1+)Pnn, 4)
                normed_target_next_pose = torch.cat(
                    [normed_ego_next_pose[:, None, :], normed_near_next_pose],
                    dim=1,
                )

                # 역정규화: (B*R, (1+)Pnn, 4)
                unnorm_target_next_pose = state_normalizer.inverse(
                    normed_target_next_pose)

                # world 변환을 위해 (B*R*(1+)Pnn, 4)로 펼침
                # flat: (N, 4), N = (B*R)*(1+Pnn)
                unnorm_target_next_pose_flat = unnorm_target_next_pose.reshape(
                    -1, 4)

                # world_flat: (N, 4)
                world_flat = _covert_from_ego_to_world(
                    target_poses=unnorm_target_next_pose_flat,
                    origin_world_pose=unnorm_origin_world_pose,  # (B*R, 4)
                )

                # world: (B*R, (1+)Pnn, 4)
                world_pose = world_flat.reshape(merged_batch, one_or_pnn, 4)

                # step 저장: target_joint_scene_world[:, :, step_idx, :] = (B*R, (1+)Pnn, 4)
                target_joint_scene_world[:, :, step_idx, :] = world_pose

                # 다음 스텝을 위한 origin 갱신(ego 기준)
                # unnorm_origin_world_pose: (B*R, 4)
                unnorm_origin_world_pose = world_pose[:, 0, :]

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
    return target_joint_scene_world


def _predict_rollouts_batched(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, Any],
    state_normalizer: Any,
    rollout_number: int,
    rollout_chunk_size: int,
    base_seed: int,
    ddp_rank: int,
) -> torch.Tensor:
    """전체 rollout_number개 rollout을 '배치로 묶어서' 빠르게 예측합니다.

    처리 방식
    --------
    - 가능한 한 rollout을 크게 묶을수록(rollout_chunk_size가 클수록) forward 호출이 줄어 빨라집니다.
    - 다만 GPU 메모리가 부족하면 한 번에 너무 크게 묶는 게 어려울 수 있어,
      chunk로 나눠서 처리할 수 있게 구성했습니다.

    반환 텐서 shape
    --------------
    - (B, (1+)Pnn, rollout_number, future_len, 4)

    Args:
        args (Any):
            args.future_len, args.device 등을 사용합니다.
        model (nn.Module):
            inference 모델.
        norm_inputs (Dict[str, Any]):
            원래 배치 입력.
        state_normalizer (Any):
            inverse(...)를 제공해야 합니다.
        rollout_number (int):
            전체 rollout 개수(예: 32).
        rollout_chunk_size (int):
            한 번에 묶어서 돌릴 rollout 개수.
            - 보통 32로 두면 가장 빠릅니다.
            - 메모리 문제가 있으면 16/8로 줄이면 됩니다.
        base_seed (int):
            기본 seed.
        ddp_rank (int):
            분산 rank. (싱글 프로세스면 0)

    Returns:
        torch.Tensor:
            shape: (B, (1+)Pnn, rollout_number, future_len, 4)
    """
    rollout_number = int(rollout_number)
    rollout_chunk_size = int(max(1, rollout_chunk_size))
    rollout_chunk_size = int(min(rollout_chunk_size, rollout_number))

    ego_agent_past = norm_inputs.get("ego_agent_past", None)
    if not isinstance(ego_agent_past, torch.Tensor):
        raise ValueError("norm_inputs['ego_agent_past']가 torch.Tensor가 아닙니다.")
    batch_size: int = int(ego_agent_past.shape[0])

    target_future_valid = norm_inputs.get("target_future_valid", None)
    if not isinstance(target_future_valid, torch.Tensor):
        raise ValueError(
            "norm_inputs['target_future_valid']가 torch.Tensor가 아닙니다.")
    one_or_pnn: int = int(target_future_valid.shape[1])

    future_len: int = int(getattr(args, "future_len"))

    # 출력 미리 할당: (B, (1+)Pnn, rollout_number, future_len, 4)
    out = torch.empty(
        (batch_size, one_or_pnn, rollout_number, future_len, 4),
        device=ego_agent_past.device,
        dtype=ego_agent_past.dtype,
    )

    start = 0
    while start < rollout_number:
        cur = int(min(rollout_chunk_size, rollout_number - start))

        # chunk마다 seed를 바꿔서 결과가 서로 다르게 나오도록 합니다.
        # (원래 코드의 rollout_seed 규칙을 일부 재사용)
        chunk_seed = _make_rollout_seed(
            base_seed=int(base_seed),
            ddp_rank=int(ddp_rank),
            rollout_idx=int(start),
        )

        chunk_pred = _predict_rollouts_batched_one_chunk(
            args=args,
            model=model,
            norm_inputs=norm_inputs,
            state_normalizer=state_normalizer,
            rollout_repeat=cur,
            rng_seed=int(chunk_seed),
        )  # (B, (1+)Pnn, cur, future_len, 4)

        out[:, :, start:start + cur, :, :] = chunk_pred
        start += cur

    return out


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


def _predict_single_rollout(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    state_normalizer: StateNormalizer,
    rollout_seed: int,
) -> torch.Tensor:
    """rollout 1개(=미래 future_len 프레임)를 예측해서 world 좌표계 궤적을 만듭니다.

    동작 방식(쉽게 설명)
    ---------------------
    - 같은 입력(norm_inputs)이라도, diffusion 샘플링은 내부에서 랜덤 노이즈를 쓰기 때문에
      seed를 바꾸면 결과가 달라질 수 있습니다.
    - 그래서 rollout마다 torch RNG를 잠깐 분리(fork)하고,
      rollout_seed로 고정한 다음 모델을 호출합니다.
    - 그리고 step별로:
        1) 모델이 만든 미래 궤적(정규화된 integrated_trajectory)에서 "다음 1스텝"을 꺼냅니다.
        2) 그 다음 1스텝을 입력 past에 반영해서 입력을 업데이트합니다.
        3) 예측된 다음 포즈를 world 좌표계로 변환해서 저장합니다.
    - 이렇게 하면 autoregressive하게 future_len 스텝을 쌓아갑니다.

    Args:
        args (Any):
            args.future_len, args.device, args.use_deepspeed 등을 사용합니다.
        model (nn.Module):
            실제 inference 모델(EMA 모델일 수도 있음).
        norm_inputs (Dict[str, torch.Tensor]):
            정규화된 입력 dict.
            주요 텐서 예:
              - ego_agent_past: (B, time_len, 11)
              - near_agents_past: (B, Pnn, time_len, 11)
              - origin_world_pose: (B, 4)
              - target_future_valid: (B, (1+)Pnn, future_len)  (True=유효)
        state_normalizer (StateNormalizer):
            정규화 ↔ 원복 변환에 사용합니다.
        rollout_seed (int):
            이번 rollout에서 사용할 seed.
            rollout_idx마다 다르게 넣어줘야 서로 다른 결과를 얻을 수 있습니다.

    Returns:
        torch.Tensor:
            world 좌표계 예측 결과.
            shape: (B, (1+)Pnn, future_len, 4)
            마지막 4는 (x, y, cos, sin)
    """
    device = torch.device(getattr(args, "device", "cuda"))
    rng_devices = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device(
        )
        rng_devices = [int(device_index)]

    with torch.random.fork_rng(devices=rng_devices, enabled=True):
        torch.manual_seed(int(rollout_seed))
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rollout_seed))

        # Dict[str, Any]
        norm_inputs_copy = _clone_norm_inputs_for_rollout(
            norm_inputs,
            sanity_check_tensor_storage=False,
        )

        # unnorm_origin_world_pose: (B, 4)
        unnorm_origin_world_pose = norm_inputs_copy["origin_world_pose"]

        target_joint_scene: List[torch.Tensor] = []

        with torch.inference_mode():
            for step_idx in range(int(args.future_len)):
                decoder_output = _forward_model_for_validation(
                    args=args,
                    model=model,
                    norm_inputs=norm_inputs_copy,
                )

                # integrated_trajectory: (B, (1+)Pnn, 1+T, 4)
                normed_trajectories = decoder_output.get(
                    "integrated_trajectory")
                if normed_trajectories is None:
                    raise KeyError(
                        "decoder_output에서 'integrated_trajectory'를 찾을 수 없습니다. "
                        "현재 validate_func는 integrated_trajectory를 사용하도록 구현되어 있습니다."
                    )

                # (B, 1+T, 4)
                normed_ego_trajectories = normed_trajectories[:, 0, :, :]
                # (B, Pnn, 1+T, 4)
                normed_near_trajectories = normed_trajectories[:, 1:, :, :]

                # 다음 1스텝만 사용
                normed_ego_next_pose = normed_ego_trajectories[:,
                                                               1, :]  # (B, 4)
                normed_near_next_pose = normed_near_trajectories[:, :,
                                                                 1, :]  # (B, Pnn, 4)

                # 입력 업데이트(autoregressive)
                norm_inputs_copy = _update_merged_inputs(
                    norm_inputs_copy=norm_inputs_copy,
                    normed_ego_next_pose=normed_ego_next_pose,
                    normed_near_next_pose=normed_near_next_pose,
                )

                # (B, (1+)Pnn, 4)
                normed_target_next_pose = torch.cat(
                    [normed_ego_next_pose[:, None, :], normed_near_next_pose],
                    dim=1,
                )

                # 역정규화: (B, (1+)Pnn, 4)
                unnorm_target_next_pose = state_normalizer.inverse(
                    normed_target_next_pose)

                # world 변환
                # (B*(1+)Pnn, 4)
                unnorm_target_next_pose_flat = unnorm_target_next_pose.reshape(
                    -1, unnorm_target_next_pose.shape[2])

                # (B*(1+)Pnn, 4)
                unnorm_target_next_pose_world_flat = _covert_from_ego_to_world(
                    target_poses=unnorm_target_next_pose_flat,
                    origin_world_pose=unnorm_origin_world_pose,
                )

                # (B, (1+)Pnn, 4)
                unnorm_target_next_pose_world = unnorm_target_next_pose_world_flat.reshape(
                    unnorm_target_next_pose.shape[0],
                    unnorm_target_next_pose.shape[1],
                    unnorm_target_next_pose.shape[2],
                )

                target_joint_scene.append(unnorm_target_next_pose_world)

                # 다음 스텝 world 기준 origin 갱신 (ego 기준)
                # unnorm_origin_world_pose: (B, 4)
                unnorm_origin_world_pose = unnorm_target_next_pose_world[:,
                                                                         0, :]

        # (B, (1+)Pnn, future_len, 4)
        return torch.stack(target_joint_scene, dim=2)


def validate_func(
    args: Any,
    model: nn.Module,
    ema: Optional[ModelEma],
    norm_inputs: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    state_normalizer: StateNormalizer,
    observation_normalizer: Any,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
) -> None:
    """validation에서 예측 rollouts를 만들고 metric을 업데이트합니다.

    요구사항 반영 내용
    ------------------
    1) EMA 모델로 inference:
        - ema가 있으면 ema.ema를 사용합니다.
    2) rollout_idx마다 다른 예측:
        - rollout_idx마다 다른 seed를 걸고,
          그 seed로 torch RNG를 잠깐 분리(fork)해서 모델을 호출합니다.
        - 이렇게 하면 모델 내부에서 뽑는 랜덤 노이즈가 rollout마다 달라집니다.

    Args:
        args: argparse 인자들.
        model: 원본 모델.
        ema: EMA 래퍼(ModelEma) 또는 None.
        norm_inputs: 정규화된 입력 dict.
        state_normalizer: 상태 역정규화용.
        observation_normalizer: (현재 함수 내부에서는 직접 사용하지 않지만, 인터페이스 유지용)
        wosac_metrics: WOSAC metric 업데이트용.
        min_ade: ADE metric 업데이트용.

    Returns:
        None
    """
    inference_model = _select_inference_model_for_validation(model=model,
                                                             ema=ema)
    inference_model.eval()

    # norm_inputs 의 각 텐서 NaN/Inf 체크
    norm_inputs = _sanitize_norm_inputs(norm_inputs)
    # target_future_valid : (B, (1+)Pnn, future_len)  True=유효
    target_future_valid = build_target_future_tensors_and_masks_for_inference(
        args,
        norm_inputs,
        args.future_len,
    )
    norm_inputs["target_future_valid"] = target_future_valid

    batch_size = norm_inputs["ego_agent_past"].shape[0]

    ROLLOUT_NUMBER = 32

    # rollout 결과 모으기: 각 원소는 (B, (1+)Pnn, future_len, 4)
    target_scenario_rollouts_list: List[torch.Tensor] = []

    ddp_rank = int(ddp.get_rank()) if getattr(args, "ddp", False) else 0
    base_seed = int(getattr(args, "seed", 0))

    for rollout_idx in range(ROLLOUT_NUMBER):
        rollout_seed = _make_rollout_seed(
            base_seed=base_seed,
            ddp_rank=ddp_rank,
            rollout_idx=rollout_idx,
        )

        # (B, (1+)Pnn, future_len, 4)
        target_joint_scene = _predict_single_rollout(
            args=args,
            model=inference_model,  # ✅ EMA 모델로 forward
            norm_inputs=norm_inputs,
            state_normalizer=state_normalizer,
            rollout_seed=rollout_seed,  # ✅ rollout_idx마다 다른 seed
        )

        target_scenario_rollouts_list.append(target_joint_scene)

    # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 4)
    target_scenario_rollouts_np = torch.stack(target_scenario_rollouts_list,
                                              dim=2)

    pred_traj = target_scenario_rollouts_np[:, :, :, :, :2]
    pred_traj = pred_traj.reshape(
        pred_traj.shape[0] * pred_traj.shape[1],
        pred_traj.shape[2],
        pred_traj.shape[3],
        pred_traj.shape[4],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step, 2]

    pred_head_cos_yaw = target_scenario_rollouts_np[:, :, :, :, 2]
    pred_head_sin_yaw = target_scenario_rollouts_np[:, :, :, :, 3]
    pred_head = torch.atan2(pred_head_sin_yaw, pred_head_cos_yaw)
    pred_head = pred_head.reshape(
        pred_head.shape[0] * pred_head.shape[1],
        pred_head.shape[2],
        pred_head.shape[3],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step]

    scenario_id: List[str] = norm_inputs["scenario_id"]
    assert isinstance(scenario_id, list), "scenario_id는 List[str] 타입이어야 합니다."
    assert batch_size == len(
        scenario_id), "batch_size와 scenario_id 길이가 맞지 않습니다."

    target_id = norm_inputs["target_id"]  # (B, (1+)Pnn)
    target_id = target_id.reshape(-1)  # [n_agent]

    one_or_Pnn = target_future_valid.shape[1]
    agent_batch: List[int] = []
    for batch_idx in range(batch_size):
        agent_batch.extend([batch_idx] * one_or_Pnn)
    agent_batch = torch.tensor(
        agent_batch,
        dtype=torch.long,
        device=pred_traj.device,
    )  # [n_agent]

    target_z = norm_inputs["target_z"]  # (B, (1+)Pnn)
    target_z = target_z.reshape(-1)  # [n_agent]
    # target_z: [n_agent] -> [n_agent, n_rollout, n_step] 으로 확장
    target_z = target_z[:, None, None].expand(
        -1,
        pred_traj.shape[1],
        pred_traj.shape[2],
    ).contiguous()

    device = pred_traj.device
    # scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
    scenario_rollouts = get_scenario_rollouts(
        scenario_id=get_scenario_id_int_tensor(scenario_id, device),
        agent_id=target_id,
        agent_batch=agent_batch,
        pred_traj=pred_traj,
        pred_z=target_z,
        pred_head=pred_head,
    )

    unnorm_ego_future_gt_4_dim = outputs[
        "ego_future_gt_4_dim"]  # (B, future_len, 4)
    unnorm_near_future_gt_4_dim = outputs[
        "near_future_gt_4_dim"]  # (B, Pnn, future_len, 4)
    unnorm_target_future_gt_4_dim = torch.cat(
        [
            unnorm_ego_future_gt_4_dim[:, None, :, :],
            unnorm_near_future_gt_4_dim,
        ],
        dim=1,
    )  # (B, (1+)Pnn, future_len, 4)

    # [n_agent, future_len, 4]
    unnorm_target_future_gt_4_dim_flat = unnorm_target_future_gt_4_dim.reshape(
        -1, unnorm_target_future_gt_4_dim.shape[2],
        unnorm_target_future_gt_4_dim.shape[3])

    unnorm_origin_world_pose = norm_inputs["origin_world_pose"]  # (B, 4)

    # [n_agent, future_len, 4]
    unnorm_target_future_gt_4_dim_world = _covert_from_ego_to_world(
        target_poses=unnorm_target_future_gt_4_dim_flat,
        origin_world_pose=unnorm_origin_world_pose,
    )

    # unnorm_target_future_gt_xy_world: [n_agent, future_len, 2]
    unnorm_target_future_gt_xy_world = unnorm_target_future_gt_4_dim_world[:, :, :
                                                                           2]

    target_future_valid_flat = torch.any(
        unnorm_target_future_gt_4_dim_flat[:, :, :4] != 0,
        dim=-1,
    )  # [n_agent, future_len]

    min_ade.update(
        pred=pred_traj,  # [n_agent, n_rollout, n_step, 2]
        target=unnorm_target_future_gt_xy_world,  # [n_agent, future_len, 2]
        target_valid=target_future_valid_flat,  # [n_agent, future_len]
    )

    tfrecord_path = norm_inputs["tfrecord_path"]
    assert isinstance(tfrecord_path,
                      list), "tfrecord_path는 List[str] 타입이어야 합니다."
    assert batch_size == len(
        tfrecord_path), "batch_size와 tfrecord_path 길이가 맞지 않습니다."
    wosac_metrics.update(tfrecord_path, scenario_rollouts)


def _update_merged_inputs(
        norm_inputs_copy: Dict[str, torch.Tensor],
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
        normed_near_next_pose: torch.Tensor,  # (B, Pnn, 4)
) -> Dict[str, torch.Tensor]:
    ego_agent_past = norm_inputs_copy["ego_agent_past"]  # (B, time_len, 11)
    ego_next_11_dim = ego_agent_past[:, -1, :].clone()  # (B, 11)
    ego_next_11_dim[:, :4] = normed_ego_next_pose  # (B, 11)
    norm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, 1:, :], ego_next_11_dim[:, None, :]],
        dim=1)  # (B, time_len, 11)

    near_agents_past = norm_inputs_copy[
        "near_agents_past"]  # (B, Pnn, T_past, 11)
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B, Pnn, 11)
    near_next_11_dim[:, :, :4] = normed_near_next_pose  # (B, Pnn, 11)
    norm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2)  # (B, Pnn, T_past, 11)

    non_near_agents_past = norm_inputs_copy[
        "non_near_agents_past"]  # (B, Nnn, T_past, 11)
    assert non_near_agents_past.shape[1] == 0, "현재 non-near agent는 처리하지 않습니다."

    neighbor_agents_past = norm_inputs_copy[
        "neighbor_agents_past"]  # (B, agent_num, T_past, 11)
    neighbor_agents_num = neighbor_agents_past.shape[1]
    assert neighbor_agents_num == near_agents_past.shape[
        1], "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."
    norm_inputs_copy["neighbor_agents_past"] = torch.cat(
        [neighbor_agents_past[:, :, 1:, :], near_next_11_dim[:, :, None, :]],
        dim=2)  # (B, agent_num, T_past, 11)

    norm_inputs_copy = _transform_origin(norm_inputs_copy, normed_ego_next_pose)
    return norm_inputs_copy


def _clone_norm_inputs_for_rollout(
    norm_inputs: Dict[str, Any],
    sanity_check_tensor_storage: bool = False,
) -> Dict[str, Any]:
    """rollout에서 입력 dict를 안전하게 복사합니다.

    목적은 2가지입니다.

    1) rollout 중에 `norm_inputs_copy`를 계속 수정하더라도,
       원본 `norm_inputs`가 같이 바뀌지 않게 만들기
    2) 값이 `torch.Tensor`인 경우에는 실제 메모리까지 새로 만들어서(copy),
       같은 저장공간을 공유하지 않게 만들기

    복사 규칙:
        - 값이 torch.Tensor 이면: `clone()`으로 새 텐서를 만듭니다.
        - 값이 dict/list/tuple 이면: 안쪽 원소까지 재귀적으로 같은 규칙을 적용합니다.
        - 그 외 타입(int/float/str 등)은: 보통 rollout 중에 수정하지 않으므로 그대로 둡니다.

    Args:
        norm_inputs: 모델 입력 dict.
        sanity_check_tensor_storage: True이면 최상위 key 기준으로
            원본 텐서와 복사 텐서가 같은 저장공간을 공유하지 않는지 확인합니다.

    Returns:
        rollout_inputs: 복사된 입력 dict.
    """
    rollout_inputs: Dict[str, Any] = {
        key: _clone_nested_value(value) for key, value in norm_inputs.items()
    }

    if sanity_check_tensor_storage:
        _assert_no_shared_tensor_storage(norm_inputs, rollout_inputs)

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


def _transform_origin(
        norm_inputs_copy: Dict[str, torch.Tensor],
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
) -> Dict[str, torch.Tensor]:
    """normed_ego_next_pose 기준으로 입력 전체의 좌표 기준을 바꿉니다.

    핵심 아이디어는 간단합니다.

    - ego의 다음 위치/방향을 "새 기준"으로 삼습니다.
    - 그러면 ego의 다음 위치는 (0, 0)이 되고,
      ego의 다음 방향은 정면(각도 0)이 되도록 입력 전체를 같이 바꿔줘야 합니다.

    변환은 모든 좌표에 대해 아래 2단계로 이뤄집니다.
        1) 이동: 모든 (x, y)에서 ego의 다음 위치 (dx, dy)를 뺍니다.
        2) 회전: ego의 다음 방향만큼 반대로 회전합니다.

    무효 데이터:
        각 텐서마다 정해진 규칙(0 패딩 또는 별도 valid mask)에 따라
        무효인 값은 변환하지 않고 원래 값을 유지합니다.

    Args:
        norm_inputs_copy: 모델 입력 dict.
        normed_ego_next_pose: (B, 4) ego 다음 프레임 (x, y, cos, sin)

    Returns:
        norm_inputs_copy: 좌표 기준이 변환된 입력 dict.
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
                               dim=-1)  # (B, A, time_len)
        _transform_state_11_dim_inplace(
            state_11=agents_past,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    # 5) points + explicit valid mask
    point_and_mask_keys = [
        ("stop_sign_points", "stop_sign_points_is_valid"),
        ("crosswalk_points", "crosswalk_points_is_valid"),
        ("speed_bump_points", "speed_bump_points_is_valid"),
        ("driveway_points", "driveway_points_is_valid"),
        ("road_edge", "road_edge_is_valid"),
    ]
    for point_key, mask_key in point_and_mask_keys:
        if point_key not in norm_inputs_copy or mask_key not in norm_inputs_copy:
            continue
        points_xy = norm_inputs_copy[point_key]  # (B, N, L, 2)
        points_valid = norm_inputs_copy[mask_key]  # (B, N, L)
        if (not isinstance(points_xy, torch.Tensor)) or points_xy.numel() == 0:
            continue
        if not isinstance(points_valid, torch.Tensor):
            continue
        # if points_valid shape (B, N) -> (B, N, L)
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

        valid_mask = torch.any(lane_12[..., :8] != 0.0,
                               dim=-1)  # (B, lane_num, lane_len)
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
            valid_mask = torch.any(static_objects[..., :4] != 0.0,
                                   dim=-1)  # (B, static_num)
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

    - ego_pose는 "ego를 원점(0,0)으로 보는 좌표"에서의 값입니다.
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
    )  # (N, 4)

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
    prepare_wandb_resume(args)
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
        traceback.print_exc()  # <-- 표준에러로 자세한 스택
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
