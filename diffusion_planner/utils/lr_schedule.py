from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, MultiplicativeLR
from torch.optim import Optimizer
import torch


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
            start_factor=1.0 / max(1, warmup_steps),  # 첫 step에서 η_max / warmup_steps
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