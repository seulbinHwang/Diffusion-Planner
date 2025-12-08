from torch.optim.lr_scheduler import (
    SequentialLR,
    LinearLR,
    CosineAnnealingLR,
    MultiplicativeLR,
    LambdaLR,
)
from torch.optim import Optimizer
import torch
import math
from typing import Callable
import functools


def stagewise_warmup_cosine_lr_factor(
    step: int,
    total_update_steps: int,
    stage1_total_steps: int,
    stage2_total_steps: int,
    stage3_total_steps: int,
    warmup_steps: int,
    stage1_max_lr: float,
    stage2_max_lr: float,
    stage3_max_lr: float,
) -> float:
    """Stage1/2/3 구간에 따라 현재 step에서 사용할 lr 비율을 계산한다.

    이 함수는 "실제 lr"이 아니라
      · base_lr = stage1_max_lr
    를 기준으로 한 **배율 값**을 돌려준다.

    전체 흐름:
      - Stage1:
        · 앞쪽 warmup_steps: 0 → stage1_max_lr 선형 증가
        · 나머지 Stage1: stage1_max_lr → stage2_max_lr cosine 감소
      - Stage2:
        · 전체: stage2_max_lr → stage3_max_lr cosine 감소
      - Stage3:
        · 전체: stage3_max_lr → (stage3_max_lr * 0.5) cosine 감소
        · Stage3 이후: (stage3_max_lr * 0.5) 고정

    Args:
        step:
            - shape: ()
            - 0부터 시작하는 전역 step 인덱스.
        total_update_steps:
            - shape: ()
            - 이론상 전체 업데이트 step 수 (Stage 합보다 작을 수도 있고 클 수도 있음).
        stage1_total_steps:
            - shape: ()
            - Stage1 에 할당된 step 수.
        stage2_total_steps:
            - shape: ()
            - Stage2 에 할당된 step 수.
        stage3_total_steps:
            - shape: ()
            - Stage3 에 할당된 step 수.
        warmup_steps:
            - shape: ()
            - Stage1 앞부분에서 0 → stage1_max_lr 로 올릴 step 수.
        stage1_max_lr:
            - shape: ()
            - Stage1 구간에서의 최대 lr.
        stage2_max_lr:
            - shape: ()
            - Stage2 구간에서의 시작 lr.
        stage3_max_lr:
            - shape: ()
            - Stage3 구간에서의 시작 lr.

    Returns:
        float:
            - shape: ()
            - base_lr(stage1_max_lr)에 곱해 사용할 lr 배율 (예: 0.0 ~ 1.0+).
    """
    total_update_steps = max(1, int(total_update_steps))
    s1_total = max(1, int(stage1_total_steps))
    s2_total = max(0, int(stage2_total_steps))
    s3_total = max(0, int(stage3_total_steps))

    # Stage1 안에서 warmup 구간과 cosine 구간 나누기
    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps >= s1_total:
        # 최소 1 step은 cosine 구간으로 남겨둔다.
        warmup_steps = max(0, s1_total - 1)

    s1_warmup = warmup_steps
    s1_decay_steps = max(1, s1_total - s1_warmup)

    s1_end = s1_total
    s2_start = s1_end
    s2_end = s2_start + s2_total
    s3_start = s2_end
    s3_end = s3_start + s3_total

    # Stage3 끝 기준으로 전체 step 수 한 번 더 정리
    total_ref_steps = max(total_update_steps, s3_end)

    base_lr = float(stage1_max_lr) if stage1_max_lr > 0.0 else 1.0
    eta3_min = float(stage3_max_lr) * 0.5

    # step 범위를 [0, total_ref_steps-1] 안으로 고정
    t = max(0, min(int(step), total_ref_steps - 1))

    # --- Stage1: warmup + cosine(stage1_max → stage2_max) ---
    if t < s1_warmup and s1_warmup > 0:
        # 0 → stage1_max_lr로 선형 증가
        factor = float(t + 1) / float(s1_warmup)
        lr_t = stage1_max_lr * factor
    elif t < s1_end:
        # warmup 이후 Stage1 끝까지: stage1_max_lr → stage2_max_lr
        k = t - s1_warmup
        T = max(1, s1_decay_steps - 1)
        # k=0 → stage1_max_lr, k=T → stage2_max_lr
        cos_inner = math.pi * float(k) / float(T)
        lr_t = stage2_max_lr + 0.5 * (stage1_max_lr - stage2_max_lr) * (
            1.0 + math.cos(cos_inner))
    # --- Stage2: cosine(stage2_max → stage3_max) ---
    elif t < s2_end and s2_total > 0:
        k = t - s2_start
        T = max(1, s2_total - 1)
        cos_inner = math.pi * float(k) / float(T)
        lr_t = stage3_max_lr + 0.5 * (stage2_max_lr - stage3_max_lr) * (
            1.0 + math.cos(cos_inner))
    # --- Stage3: cosine(stage3_max → stage3_max*0.5) ---
    elif t < s3_end and s3_total > 0:
        k = t - s3_start
        T = max(1, s3_total - 1)
        cos_inner = math.pi * float(k) / float(T)
        lr_t = eta3_min + 0.5 * (stage3_max_lr -
                                 eta3_min) * (1.0 + math.cos(cos_inner))
    else:
        # Stage3 끝 이후에는 최소 lr 유지
        lr_t = eta3_min

    return float(lr_t) / float(base_lr)


def build_stagewise_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_update_steps: int,
    stage1_total_steps: int,
    stage2_total_steps: int,
    stage3_total_steps: int,
    warmup_steps: int,
    stage1_max_lr: float,
    stage2_max_lr: float,
    stage3_max_lr: float,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Stage1/2/3 구간에 맞춰 warmup + cosine 형태로 lr을 조절하는 스케줄러를 만든다.

    전체 step 흐름:
      - [Stage1]
        · 처음 warmup_steps 동안: 0 → stage1_max_lr 로 선형 증가
        · 이후 Stage1 끝날 때까지: stage1_max_lr → stage2_max_lr 로 cosine 감소
      - [Stage2]
        · Stage2 전체: stage2_max_lr → stage3_max_lr 로 cosine 감소
      - [Stage3]
        · Stage3 전체: stage3_max_lr → (stage3_max_lr * 0.5) 로 cosine 감소
        · 이후 step 부터는 (stage3_max_lr * 0.5) 를 유지

    이 스케줄러는 "기준 lr" 을 Stage1 최대값으로 두고,
    각 step에서 (현재 lr / Stage1 최대 lr) 비율을 돌려준다.
    Optimizer의 base lr는 Stage1 최대 lr로 맞춰져 있어야 한다.

    Args:
        optimizer:
            - torch.optim.Optimizer 인스턴스.
            - 각 param_group["lr"] 텐서 shape: () (스칼라 float).
        total_update_steps:
            - shape: ()
            - 전체 update step 수 (대략적인 upper bound).
        stage1_total_steps / stage2_total_steps / stage3_total_steps:
            - shape: ()
            - 각각 Stage1, Stage2, Stage3 에서 사용할 step 수.
        warmup_steps:
            - shape: ()
            - Stage1 앞부분에서 0 → stage1_max_lr 로 올릴 step 수.
        stage1_max_lr / stage2_max_lr / stage3_max_lr:
            - shape: ()
            - 각 Stage 구간 시작 시 기준으로 삼을 최대 lr 값.

    Returns:
        torch.optim.lr_scheduler._LRScheduler:
            - LambdaLR 기반 스케줄러.
            - 내부에서 stagewise_warmup_cosine_lr_factor(...) 를 사용해
              step 마다 lr 배율(scale factor)을 계산한다.
    """
    # lr_lambda(step:int) -> float 형태의 콜백을 partial 로 만든다.
    lr_lambda: Callable[[int], float] = functools.partial(
        stagewise_warmup_cosine_lr_factor,
        total_update_steps=total_update_steps,
        stage1_total_steps=stage1_total_steps,
        stage2_total_steps=stage2_total_steps,
        stage3_total_steps=stage3_total_steps,
        warmup_steps=warmup_steps,
        stage1_max_lr=stage1_max_lr,
        stage2_max_lr=stage2_max_lr,
        stage3_max_lr=stage3_max_lr,
    )

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )
    return scheduler


def CosineAnnealingWarmUpRestarts(optimizer,
                                  epoch,
                                  warm_up_epoch,
                                  start_factor=0.1):
    assert epoch >= warm_up_epoch
    T_warmup = warm_up_epoch

    warmup_scheduler = LinearLR(optimizer,
                                start_factor=start_factor,
                                total_iters=warm_up_epoch - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, fixed_scheduler],
                             milestones=[T_warmup])

    return scheduler


def build_pytorch_warmup_cosine_scheduler(
    optimizer: Optimizer,
    total_update_steps: int,
    warmup_steps: int,
    eta_min: float = 0.0,
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    선형 워밍업 → 코사인 디케이 → (NEW) eta_min에서 HOLD
    """
    remaining = max(0, total_update_steps - warmup_steps)

    # 워밍업 스케줄러 (base_lr=η_max 기준)
    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1.0 /
            max(1, warmup_steps),  # 첫 step에서 η_max / warmup_steps
            end_factor=1.0,
            total_iters=warmup_steps,
        )
    else:
        warmup = None

    # 코사인 디케이 스케줄러
    if remaining > 0:
        cosine = CosineAnnealingLR(optimizer, T_max=remaining, eta_min=eta_min)
    else:
        cosine = None

    # (NEW) 꼬리 고정: 이후에는 LR을 그대로 유지 (곱하기 1.0)
    hold = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)

    # 조합
    if warmup is not None and cosine is not None:
        # warmup_steps 시점에 warmup→cosine, warmup_steps+remaining 시점에 cosine→hold
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine, hold],
            milestones=[warmup_steps, warmup_steps + remaining],
        )
    elif warmup is not None:
        # total_update_steps == warmup_steps 인 케이스: 워밍업 이후 고정
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, hold],
            milestones=[warmup_steps],
        )
    elif cosine is not None:
        # 워밍업 없음: 코사인 이후 고정
        scheduler = SequentialLR(
            optimizer,
            schedulers=[cosine, hold],
            milestones=[remaining],
        )
    else:
        # 아무 것도 없으면 그냥 고정
        scheduler = hold

    return scheduler
