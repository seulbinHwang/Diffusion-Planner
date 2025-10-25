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
    """④ 선형 워밍업 → 코사인 디케이 스케줄 (PyTorch 코어)

    Piecewise 스케줄:
      - 0..T_w: LinearLR(start_factor=0 → end_factor=1)
      - T_w..T: CosineAnnealingLR(T_max=T - T_w, eta_min=η_min)

    전제:
      - 옵티마이저의 현재 LR이 **η_max(B)** 여야 함.
        (warmup은 base_lr * factor로 동작하므로, base_lr = η_max로 설정)

    Args:
        optimizer (Optimizer): PyTorch 옵티마이저.
        total_update_steps (int): 총 스텝 수 T.
        warmup_steps (int): 워밍업 스텝 수 T_w(B).
        eta_min (float): 코사인 디케이 최소 학습률 η_min.

    Returns:
        _LRScheduler: SequentialLR 스케줄러.
    """
    remaining = max(0, total_update_steps - warmup_steps)

    if warmup_steps > 0:
        warmup = LinearLR(
            optimizer,
            start_factor=1 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        if remaining > 0:
            cosine = CosineAnnealingLR(optimizer,
                                       T_max=remaining,
                                       eta_min=eta_min)
            scheduler = SequentialLR(optimizer,
                                     schedulers=[warmup, cosine],
                                     milestones=[warmup_steps])
        else:
            scheduler = warmup
    else:
        # 워밍업이 0이면 바로 코사인
        scheduler = CosineAnnealingLR(optimizer,
                                      T_max=max(1, remaining),
                                      eta_min=eta_min)
    return scheduler
