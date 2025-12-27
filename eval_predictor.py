import argparse
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
from train_predictor import DiffusionPlannerCollate
from typing import Tuple, Any, Dict, Optional, List
import torch
import torch.nn as nn
from torch import optim
from timm.utils import ModelEma
from diffusion_planner.utils import ddp
import time
from tqdm import tqdm

def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    aug: Optional[object],
    batch_num_in_one_val_epoch: int,
) -> Tuple[Dict[str, float], float, float]:
    """하나의 epoch 동안 학습을 실행하고, 손실과 걸린 시간을 계산한다.

    Args:
        epoch: 현재 epoch 인덱스(0 기반).
        total_epochs: 전체 epoch 수.
        validation_loader: 학습용 DataLoader. 배치 텐서는 (B, ·) shape.
        diffusion_planner: 학습 중인 모델(nn.Module 또는 DDP/DeepSpeed 래퍼).
        optimizer: 옵티마이저.
        scheduler: 학습률 스케줄러.
        args: 학습 설정.
        model_ema: EMA 래퍼 또는 None.
        aug: augmentation 객체(StatePerturbation/NPCStatePerturbation 등) 또는 None.

    Returns:
        validation_loss: key별 epoch 평균 손실 딕셔너리. 각 값은 scalar float.
        validation_total_loss: 최종 합 손실 scalar float.
        epoch_elapsed_time_sec: 해당 epoch에 걸린 시간(초).
    """
    # train_epoch 안에서 각 step 배치 텐서는 대략
    #   ego_agent_past: (B, T_past, 11)
    #   neighbor_agents_past: (B, A, T_past, 11)
    #   lanes: (B, L, lane_len, 12)
    #   static_objects: (B, S, 10)
    #   neighbor_future_gt_3_dim: (B, A, T_future, 3)
    # 와 같은 shape로 사용된다.
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{total_epochs}")

    epoch_t0 = time.perf_counter()
    """ validation_loss: Dict[str, float]
    <diffusion_loss_func 가 출력해주는 loss_dict>
        - neighbor_prediction_loss : 이웃 예측 손실 텐서
        - integration_loss : 통합 손실 텐서
        - constraint_loss : 제약 손실 텐서

        - neighbor_prediction_loss_xy / integration_loss_xy
        - neighbor_prediction_loss_yaw / integration_loss_yaw
        - neighbor_prediction_loss_xy_early / integration_loss_xy_early
        - neighbor_prediction_loss_yaw_early / integration_loss_yaw_early
        - constraint_diff_vx_b / constraint_diff_vy_b / constraint_diff_yaw_rate

    <diffusion_loss_func 출력 후 _compute_loss_dict 가 추가하는 항목>
    - learn_progress / direct_loss_weight / int_loss_weight / const_loss_weight :
        진행도 및 가중치 기록용 텐서
    - loss_dict : 최종 합 손실 텐서.
    """

    validation_loss, validation_total_loss = validation_epoch(
        validation_loader,
        diffusion_planner,
        args,
        model_ema,
        batch_num_in_one_val_epoch,
        aug,
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
    return validation_loss, validation_total_loss, epoch_elapsed_time_sec


def validation_epoch(
        data_loader,
        model: nn.Module,
        args: argparse.Namespace,
        ema: Optional[object],
        batch_num_in_one_val_epoch: int,
        aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행한다.

    처리 순서:
      1) 모델을 train 모드로 두고, DDP 사용 시 CUDA 동기화.
      2) 전체 업데이트 스텝 수(batch_num_in_one_val_epoch)를 계산해 진행도(progress) 기준을 잡는다.
      3) 각 배치에 대해
         - device 로 이동 및 상한 클리핑(_prepare_batch_for_device)
         - augmentation, near future mask/ near future 4차원 궤적 생성
         - 관측 normalization 
         - diffusion 손실 / feasible 손실 합성
         - 역전파, optimizer/scheduler step, EMA 업데이트
         - 배치별 loss 를 epoch_loss_dict_list 리스트에 모은다.
      4) epoch 종료 후, 배치별 loss 를 평균(get_epoch_mean_loss).
      5) DDP 사용 시 rank 0 기준으로 평균 후 출력/반환.

    Args:
        data_loader:
            PyTorch DataLoader. 각 요소는 collate_fn 이 만든 배치 dict.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        optimizer:
            torch.optim.Optimizer 인스턴스.
        args:
            학습 설정/상태 Namespace.
        ema:
            EMA 래퍼(ModelEma 등) 또는 None.
        scheduler:
            학습률 스케줄러. 배치마다 step() 이 호출된다.
        aug:
            StatePerturbation 또는 NPCStatePerturbation, 또는 None.

    Returns:
        - epoch_mean_loss: 손실 항목별 평균 dict. ( Dict[str, float] )
        - epoch_mean_loss["loss"]: 최종 스칼라 손실 값. float
    """
    epoch_loss_dict_list: List[Dict[str, torch.Tensor]] = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            # 1) device 이동 + 상한 클리핑 + 정답 분리
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )
            # near_future_gt_3_dim: (B, Pnn, future_len, 3)
            near_future_gt_3_dim: torch.Tensor = outputs["near_future_gt_3_dim"]
            # 3) near future 4차원 궤적 + mask 생성
            # near_future_gt_4_dim: (B, Pnn, future_len, 4)
            # near_future_mask:    (B, Pnn, future_len)
            near_future_gt_4_dim, near_future_mask = \
                _build_near_future_4dim_and_mask(near_future_gt_3_dim)
            # 2) augmentation 적용
            # inputs, ego_future_gt_3_dim, near_future_gt_3_dim = \
            #     _apply_augmentation(
            #         inputs=inputs,
            #         ego_future_gt_3_dim=ego_future_gt_3_dim, # (B, future_len, 3)
            #         near_future_gt_3_dim=near_future_gt_3_dim, # (B, Pnn, future_len, 3)
            #         ego_future_gt_mask=ego_future_gt_mask, # (B, future_len)
            #         near_future_mask=near_future_mask, # (B, Pnn, future_len)
            #         aug=aug,
            #         args=args,
            #     )

            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)

            # 5) loss 계산 + 역전파 + optimizer/scheduler step
            """
            이번 배치(batch)를 학습하기 전에, 이전 배치에서 남아있는 기울기(gradient) 값을 깨끗이 지우는 작업

            (DeepSpeed를 쓰면 optimizer가 모델 내부에 묶여 동작하는 경우가 많아서, “모델에게” 초기화를 맡기는 방식이 맞습니다.)

            set_to_none=True는 기울기 값을 “0으로 채우기”보다 **아예 비워(None으로 만들기)**에 가까워서, 보통 메모리/속도 면에서 조금 더 유리할 수 있습니다.
            """
            if args.use_deepspeed and hasattr(
                    model, "zero_grad"):
                model.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            # base_model: DDP/DeepSpeed 래퍼 벗긴 실제 모델
            base_model = ddp.get_model(model, args.ddp)
            sde_marginal_prob = base_model.sde.marginal_prob

            # 5-1) diffusion 기본 loss 계산 (neighbor / integration / constraint 등)
            raw_loss_dict: Dict[str, torch.Tensor] = {}
            raw_loss_dict, _ = diffusion_loss_func(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                marginal_prob=sde_marginal_prob,
                ego_future_gt_4_dim=outputs["ego_future_gt_4_dim"],
                near_future_gt_4_dim=near_future_gt_4_dim,
                near_future_mask=near_future_mask,
                state_normalizer=args.state_normalizer,
                loss_dict=raw_loss_dict,
                model_type=args.diffusion_model_type,  # 보통 "x_start" 또는 "score"
                observation_normalizer=args.observation_normalizer,
            )

            # 5-2) feasible weight 로 최종 loss 합성
            loss_dict: Dict[str, torch.Tensor] = _compute_loss_dict(
                loss_dict=raw_loss_dict,
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                batch_num_in_one_val_epoch=batch_num_in_one_val_epoch,
            )

            total_loss: float = _backward_and_step(
                loss_dict=loss_dict,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )
            # 6) WD warmdown, EMA 업데이트
            _apply_weight_decay_warmdown(optimizer)
            _update_ema_if_needed(ema, model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss_dict_list.append(loss_dict)

            # 전역 스텝 누적 (진행도 계산에 사용)
            args._global_update_step += 1

    # --- 에폭 평균 손실 계산 ---
    """ epoch_mean_loss (각 GPU마다 계산해 둠)
    {"loss": 0.42, "neighbor_prediction_loss": 0.3, ...}
    """
    epoch_mean_loss: Dict[str,
    float] = get_epoch_mean_loss(epoch_loss_dict_list)

    if args.ddp:
        """ ddp.reduce_and_average_losses
            두 GPU가 서로 값을 더해서 합을 만든 뒤(world_size로 나눔)
            → 모든 GPU가 동일한 평균 값을 가지게 함
        통신하는 내용
            스칼라 몇 개밖에 안 되는 작은 숫자
        """
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss,
            torch.device(args.device),
        )
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]