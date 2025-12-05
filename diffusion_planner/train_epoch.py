from tqdm import tqdm
import torch
from torch import nn
from typing import Tuple
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.model.module.feasible import FeasibleProjector
# =====================================================================

from typing import Dict, Tuple, Optional, List
import argparse

import torch
from torch import nn
from typing import Tuple
...
from diffusion_planner.model.module.feasible import FeasibleProjector

# =====================================================================


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """배치 dict의 모든 텐서를 지정한 device 로 옮긴다.

    Args:
        batch: DataLoader 에서 온 배치 dict. 각 값은 torch.Tensor.
        device: "cuda:0", "cpu" 등.

    Returns:
        Dict[str, torch.Tensor]:
            동일 key 를 가지되, 텐서가 모두 device 상에 놓인 dict.
    """
    return {
        key: tensor.to(device, non_blocking=True)
        for key, tensor in batch.items()
    }


def _clip_input_axes_by_args(
    batch_on_device: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """모델 설정 상한에 맞춰 입력 텐서들의 agent / lane 축을 in-place 로 자른다.

    처리 규칙 (do_not_clip=False 일 때만 적용):
      - neighbor_agents_past      : agent 축 → args.max_agent_num
      - lanes / lanes_*           : lane  축 → args.max_use_lane_num
      - route_lanes / route_lanes_* : lane 축 → args.max_use_lane_num
      - agent_route_lane_order    : (agent, lane) 축 → (args.predicted_neighbor_num, args.max_use_lane_num)
    """
    if getattr(args, "do_not_clip", False):
        return

    # ---- neighbor_agents_past: (B, A, T, 11) → A ≤ max_agent_num ----
    max_agent_num = int(getattr(args, "max_agent_num", 0))
    if "neighbor_agents_past" in batch_on_device and max_agent_num > 0:
        neighbor_agents_past = batch_on_device["neighbor_agents_past"]
        if neighbor_agents_past.dim() == 4:
            # neighbor_agents_past: (B, A_c, T, 11)
            keep_agents = min(max_agent_num, neighbor_agents_past.shape[1])
            batch_on_device["neighbor_agents_past"] = \
                neighbor_agents_past[:, :keep_agents, :, :]

    # ---- lanes / lanes_* : lane 축 → max_use_lane_num ----
    max_use_lane_num = int(getattr(args, "max_use_lane_num", 0))
    if "lanes" in batch_on_device and max_use_lane_num > 0:
        lanes = batch_on_device["lanes"]
        if lanes.dim() == 4:
            # lanes: (B, L_c, lane_len, 12)
            keep_lanes = min(max_use_lane_num, lanes.shape[1])
            if keep_lanes < lanes.shape[1]:
                batch_on_device["lanes"] = lanes[:, :keep_lanes, :, :]
                if "lanes_speed_limit" in batch_on_device:
                    # (B, L_c, 1)
                    batch_on_device["lanes_speed_limit"] = \
                        batch_on_device["lanes_speed_limit"][:, :keep_lanes, :]
                if "lanes_has_speed_limit" in batch_on_device:
                    # (B, L_c, 1)
                    batch_on_device["lanes_has_speed_limit"] = \
                        batch_on_device["lanes_has_speed_limit"][:, :keep_lanes, :]

    # ---- route_lanes / route_lanes_* : lane 축 → max_use_lane_num ----
    if "route_lanes" in batch_on_device and max_use_lane_num > 0:
        route_lanes = batch_on_device["route_lanes"]
        if route_lanes.dim() == 4:
            # route_lanes: (B, R_c, route_len, 12)
            keep_route_lanes = min(max_use_lane_num, route_lanes.shape[1])
            if keep_route_lanes < route_lanes.shape[1]:
                batch_on_device["route_lanes"] = \
                    route_lanes[:, :keep_route_lanes, :, :]
                if "route_lanes_speed_limit" in batch_on_device:
                    # (B, R_c, 1)
                    batch_on_device["route_lanes_speed_limit"] = \
                        batch_on_device["route_lanes_speed_limit"][:, :keep_route_lanes, :]
                if "route_lanes_has_speed_limit" in batch_on_device:
                    # (B, R_c, 1)
                    batch_on_device["route_lanes_has_speed_limit"] = \
                        batch_on_device["route_lanes_has_speed_limit"][:, :keep_route_lanes, :]

    # ---- agent_route_lane_order: (B, A_c, L_c) → (A', L') ----
    predicted_neighbor_num = int(getattr(args, "predicted_neighbor_num", 0))
    if "agent_route_lane_order" in batch_on_device:
        arl = batch_on_device["agent_route_lane_order"]
        if arl.dim() == 3:
            # arl: (B, A_c, L_c)
            bsz, c_agent, c_lane = arl.shape

            keep_agents_for_route = c_agent
            if predicted_neighbor_num > 0:
                keep_agents_for_route = min(predicted_neighbor_num, c_agent)

            lane_dim_input = c_lane
            if "lanes" in batch_on_device:
                # lanes: (B, L', lane_len, 12)
                lane_dim_input = batch_on_device["lanes"].shape[1]
            keep_lanes_for_route = min(
                max_use_lane_num or lane_dim_input,
                lane_dim_input,
            )

            batch_on_device["agent_route_lane_order"] = \
                arl[:, :keep_agents_for_route, :keep_lanes_for_route]


def _clip_targets_by_predicted_neighbors(
    outputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """정답 텐서 중 neighbor 축(predicted_neighbor_num)만 자른다.

    현재는 near_future_gt_3_dim 만 처리한다.
      - near_future_gt_3_dim: (B, A_c, Tf, 3) → A_c ≤ predicted_neighbor_num

    do_not_clip=True 이면 아무 것도 자르지 않는다.
    """
    if getattr(args, "do_not_clip", False):
        return

    if "near_future_gt_3_dim" not in outputs:
        return

    predicted_neighbor_num = int(getattr(args, "predicted_neighbor_num", 0))
    near_future_gt_3_dim = outputs["near_future_gt_3_dim"]
    if near_future_gt_3_dim.dim() != 4 or predicted_neighbor_num <= 0:
        return

    # near_future_gt_3_dim: (B, A_c, Tf, 3) → (B, A', Tf, 3)
    keep_agents = min(predicted_neighbor_num, near_future_gt_3_dim.shape[1])
    outputs["near_future_gt_3_dim"] = \
        near_future_gt_3_dim[:, :keep_agents, :, :]


def _prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: str,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """배치 dict를 GPU/CPU로 옮기고, 모델 상한에 맞게 축을 잘라 입력/정답을 나눈다.

    전제: collate_fn(DiffusionPlannerCollate)에서 이미
      - neighbor_agents_past         : (B, data_max_agent_num, time_len, 11)
      - near_future_gt_3_dim         : (B, data_max_agent_num, future_len, 3)
      - static_objects               : (B, data_max_static_num, 10)
      - lanes / lanes_*              : (B, data_max_lane_num, lane_len, ·)
      - route_lanes / route_lanes_*  : (B, data_max_route_num, route_len, ·)
      - agent_route_lane_order       : (B, data_max_agent_num, data_max_lane_num)
    형태로 패딩이 끝난 상태라고 가정한다.

    여기서는 (do_not_clip=False 인 경우에만)
      - neighbor_agents_past / near_future_gt_3_dim: agent 축 → max_agent_num / predicted_neighbor_num
      - lanes / lanes_*                            : lane  축 → max_use_lane_num
      - route_lanes / route_lanes_*                : lane  축 → max_use_lane_num
      - agent_route_lane_order                     : (agent, lane) 축 → (predicted_neighbor_num, max_use_lane_num)
      - static_objects                              : 그대로 유지

    를 수행한다.
    """
    # 1) device 로 옮기기
    batch_on_device: Dict[str,
                          torch.Tensor] = _move_batch_to_device(batch, device)

    # 2) dtype 특수 처리: route lane order는 항상 long 형으로 맞춘다.
    if "agent_route_lane_order" in batch_on_device:
        # (B, A_c, L_c)
        batch_on_device["agent_route_lane_order"] = batch_on_device[
            "agent_route_lane_order"].long()

    # 3) 입력 텐서들의 agent / lane 축을 모델 상한에 맞게 자르기
    if args is not None:
        _clip_input_axes_by_args(batch_on_device, args)

    # 4) 정답 키만 outputs 로 분리
    target_keys = {"ego_future_gt_3_dim", "near_future_gt_3_dim"}
    outputs: Dict[str, torch.Tensor] = {}
    for key in list(batch_on_device.keys()):
        if key in target_keys:
            outputs[key] = batch_on_device.pop(key)

    # 5) 정답 중 neighbor 축(predicted_neighbor_num)만 안쪽에서 추가 클리핑
    if args is not None:
        _clip_targets_by_predicted_neighbors(outputs, args)

    inputs: Dict[str, torch.Tensor] = batch_on_device
    return inputs, outputs


# =====================================================================
# 아래부터 train_epoch 내부를 역할별 함수로 분리
# =====================================================================


def _init_global_step_and_total_updates(
    args: argparse.Namespace,
    batch_num_in_epoch: int,
) -> int:
    """전역 스텝 상태를 초기화하고, 전체 업데이트 스텝 수를 계산한다.

    Args:
        args:
            학습 설정/상태를 담고 있는 Namespace.
            - train_epochs: 전체 학습 epoch 수.
            - _global_update_step: 없으면 0으로 새로 만든다.
        batch_num_in_epoch:
            현재 epoch에서 배치 개수. len(data_loader).

    Returns:
        int:
            batch_num_in_all_epoch = train_epochs * batch_num_in_epoch (최소 1).
    """
    if not hasattr(args, "_global_update_step"):
        args._global_update_step = 0
    batch_num_in_all_epoch: int = max(
        1,
        int(args.train_epochs) * int(batch_num_in_epoch))
    return batch_num_in_all_epoch


def _apply_augmentation(
    inputs: Dict[str, torch.Tensor],
    ego_future_gt_3_dim: torch.Tensor,  # (B, Tf, 3)
    near_future_gt_3_dim: torch.Tensor,  # (B, A, Tf, 3)
    aug: Optional[StatePerturbation],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """ego / 이웃 궤적에 대해 augmentation 을 적용한다.

    Args:
        inputs:
            모델 입력용 배치 dict. 각 값은 (B, ...) 텐서.
        ego_future_gt_3_dim:
            이고 미래 궤적. shape: (B, future_len, 3).
        near_future_gt_3_dim:
            이웃 미래 궤적. shape: (B, agent_num, future_len, 3).
        aug:
            - StatePerturbation 인스턴스이면 ego/neighbor 둘 다에 적용.
            - NPCStatePerturbation 인스턴스이면 neighbor 에만 적용.
        args:
            augmentation 이 NPCStatePerturbation 인 경우에 필요한 설정.

    Returns:
        Tuple[inputs, ego_future_gt_3_dim, near_future_gt_3_dim]:
            augmentation 이 반영된 새 텐서들.
    """

    if isinstance(aug, StatePerturbation):
        inputs, ego_future_gt_3_dim, near_future_gt_3_dim = aug(
            inputs, ego_future_gt_3_dim, near_future_gt_3_dim)

    if isinstance(aug, NPCStatePerturbation):
        inputs, near_future_gt_3_dim = aug(inputs, near_future_gt_3_dim, args)
    # augmentation 과정에서 값이 바뀌었더라도,
    # 원래 패딩이었던 위치는 다시 전부 0으로 되돌린다.
    # augmentation 전에 패딩 위치 기억 (B, A, Tf, 1)
    pad_mask_near: torch.Tensor = (near_future_gt_3_dim == 0).all(dim=-1,
                                                                  keepdim=True)
    near_future_gt_3_dim = near_future_gt_3_dim.masked_fill(pad_mask_near, 0.0)

    return inputs, ego_future_gt_3_dim, near_future_gt_3_dim


def _build_near_future_4dim_and_mask(
    near_future_gt_3_dim: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """yaw를 cos/sin으로 확장하고, 비어 있는 구간 마스크를 만든다.

    처리 내용:
      1) (x, y, yaw) 궤적에서 yaw를 cos/sin 두 값으로 나눠
         (x, y, cos, sin) 4차원 궤적을 만든다.
      2) (x, y, yaw)가 모두 0인 프레임을 "비어 있는 프레임"으로 보고
         near_future_mask 를 True/False로 만든다.
      3) 비어 있는 위치의 4차원 값은 0으로 덮어쓴다.

    Args:
        near_future_gt_3_dim:
            - shape: (B, agent_num, future_len, 3)
            - [..., 0:2] = (x, y), [..., 2] = yaw.

    Returns:
        near_future_gt_4_dim:
            - shape: (B, agent_num, future_len, 4)
            - [x, y, cos(yaw), sin(yaw)].
        near_future_mask:
            - shape: (B, agent_num, future_len)
            - True 이면 해당 프레임은 완전히 비어 있는 프레임.
    """
    # near_future_gt_3_dim: (B, A, Tf, 3)
    near_future_mask: torch.Tensor = torch.sum(
        torch.ne(near_future_gt_3_dim[..., :3], 0),
        dim=-1,
    ) == 0  # (B, A, Tf)

    # yaw: (B, A, Tf)
    yaw: torch.Tensor = near_future_gt_3_dim[..., 2]

    # near_future_gt_4_dim: (B, A, Tf, 4)
    near_future_gt_4_dim: torch.Tensor = torch.cat(
        [
            near_future_gt_3_dim[..., :2],  # (B, A, Tf, 2)
            torch.stack(
                [
                    yaw.cos(),  # (B, A, Tf)
                    yaw.sin(),  # (B, A, Tf)
                ],
                dim=-1,  # -> (B, A, Tf, 2)
            ),
        ],
        dim=-1,
    )

    near_future_gt_4_dim[near_future_mask] = 0.0

    if not torch.isfinite(near_future_gt_4_dim).all():
        raise ValueError("Non-finite values detected in near_future_gt_4_dim")

    return near_future_gt_4_dim, near_future_mask


def _compute_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    batch_num_in_all_epoch: int,
) -> Dict[str, torch.Tensor]:
    """diffusion 손실과 feasible 가중합까지 포함한 loss_dict dict 를 계산한다.

    Args:
        args:
            - _global_update_step: 현재까지 진행된 전체 스텝 수.
            - train_epochs 등은 batch_num_in_all_epoch 계산에 이미 반영되어 있음.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        norm_inputs:
            관측값 정규화가 적용된 입력 dict. 각 텐서 shape: (B, ...).
        near_future_gt_4_dim:
            - shape: (B, agent_num, future_len, 4)
            - [x, y, cos(yaw), sin(yaw)].
        near_future_mask:
            - shape: (B, agent_num, future_len)
            - True 인 위치는 완전히 비어 있는 프레임.

    Returns:
        loss_dict:
            - 다양한 부분 손실과 최종 합(loss_dict["loss"])를 담은 dict.
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
            - feasible_progress / feasible_w_dir / feasible_w_int / feasible_w_const :
                진행도 및 가중치 기록용 텐서
            - loss_dict : 최종 합 손실 텐서.

    """

    # 진행도 0~1 계산
    progress: float = min(
        1.0,
        args._global_update_step / float(max(1, batch_num_in_all_epoch - 1)),
    )
    w_dir, w_int, w_const = FeasibleProjector.loss_weights_by_progress(progress)

    if not args.use_direct_loss:
        w_dir = 0.0
        w_int = 1.0

    # 진행도/가중치 기록(평균 로그용)
    loss_dict["feasible_progress"] = torch.tensor(
        float(progress), device=next(model.parameters()).device)
    loss_dict["feasible_w_dir"] = torch.tensor(float(w_dir),
                                               device=next(
                                                   model.parameters()).device)
    loss_dict["feasible_w_int"] = torch.tensor(float(w_int),
                                               device=next(
                                                   model.parameters()).device)
    loss_dict["feasible_w_const"] = torch.tensor(float(w_const),
                                                 device=next(
                                                     model.parameters()).device)

    # 개별 손실이 없을 수도 있으니 기본값 0 텐서로 처리
    device = norm_inputs["ego_agent_past"].device
    l_dir = loss_dict.get("neighbor_prediction_loss",
                          torch.tensor(0.0, device=device))
    l_int = loss_dict.get("integration_loss", torch.tensor(0.0, device=device))
    l_con = loss_dict.get("constraint_loss", torch.tensor(0.0, device=device))

    # 최종 손실 합성
    loss_dict["loss"] = w_dir * l_dir + w_int * l_int + w_const * l_con

    return loss_dict


def _backward_and_step(
    loss_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
) -> float:
    """역전파/그래디언트 클리핑/스케줄러/옵티마이저 업데이트를 한 번 수행한다.

    Args:
        loss_dict:
            - "loss" 키에 최종 scalar 손실 텐서가 들어 있는 dict.
              · loss_dict["loss"]의 shape: ()  스칼라 텐서.
        model:
            학습 중인 모델.  # shape: (모델 파라미터 개수에 따라 자유)
            - 일반 모드: nn.Module 또는 DDP 래퍼.
            - ZeRO-2 모드: deepspeed.DeepSpeedEngine.
        optimizer:
            torch.optim.Optimizer 또는 DeepSpeed가 감싼 Optimizer.
        scheduler:
            학습률 스케줄러. 일반 모드에서만 직접 step()을 호출한다.
        args:
            학습 설정/상태 Namespace.
            - args.use_deepspeed: True이면 DeepSpeed 엔진을 사용한다.

    Returns:
        float:
            loss_dict["loss"].item() 값 (logging 용).
    """
    # total_loss는 scalar float 값
    total_loss: float = float(loss_dict["loss"].item())
    loss_tensor: torch.Tensor = loss_dict["loss"]

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False)) \
        and hasattr(model, "backward") and hasattr(model, "step")

    if use_deepspeed:
        # DeepSpeed 엔진을 사용하는 경우:
        # 1) model.backward(loss)로 분산/혼합정밀 포함해 역전파 수행
        # 2) 필요하면 엔진 쪽 gradient norm 클리핑
        # 3) model.step()이 optimizer.step() + lr_scheduler.step()까지 처리
        model.backward(loss_tensor)
        if hasattr(model, "clip_grad_norm"):
            # DeepSpeedEngine.clip_grad_norm는 내부 파라미터 shard 기준으로 norm을 계산한다.
            model.clip_grad_norm(max_norm=10.0)
        model.step()
    else:
        # 일반 PyTorch / DDP 모드
        loss_tensor.backward()
        # 모든 파라미터의 gradient L2 norm을 10.0으로 클리핑
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        # 스케줄러 / 옵티마이저 스텝
        scheduler.step()
        optimizer.step()

    return total_loss


def _apply_weight_decay_warmdown(optimizer: torch.optim.Optimizer) -> None:
    """각 파라미터 그룹의 lr 비율에 맞춰 weight_decay 를 선형으로 조정한다.

    Args:
        optimizer:
            AdamW 옵티마이저. 각 param_group 에 "wd_max", "lr_max" 키가 설정되어 있어야 한다.
    """
    for pg in optimizer.param_groups:
        wd_max = pg.get("wd_max", None)
        lr_max = pg.get("lr_max", None)
        if wd_max is None or lr_max is None:
            continue
        if wd_max > 0.0 and lr_max > 0.0:
            pg["weight_decay"] = wd_max * (pg["lr"] / lr_max)


def _update_ema_if_needed(ema: Optional[object], model: nn.Module) -> None:
    """EMA 객체가 있으면 한 스텝 업데이트한다.

    Args:
        ema:
            timm.utils.ModelEma 와 같은 EMA 래퍼 또는 None.
        model:
            현재 학습 중인 모델. ema.update(model) 의 인자로 사용된다.
    """
    if ema is not None:
        # DDP / DeepSpeed 래퍼가 씌워져 있으면 .module을 사용해 실제 모듈 기준으로 EMA를 업데이트한다.
        src_model: nn.Module = getattr(model, "module", model)
        ema.update(src_model)


# =====================================================================


def train_epoch(
    data_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ema: Optional[object],
    scheduler,
    aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행한다.

    처리 순서:
      1) 모델을 train 모드로 두고, DDP 사용 시 CUDA 동기화.
      2) 전체 업데이트 스텝 수(batch_num_in_all_epoch)를 계산해 진행도(progress) 기준을 잡는다.
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

    # 전체 업데이트 스텝 수 설정 및 global step 초기화 보장
    batch_num_in_all_epoch: int = _init_global_step_and_total_updates(
        args,
        batch_num_in_epoch=len(data_loader),
    )

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            # 1) device 이동 + 상한 클리핑 + 정답 분리
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )
            # ego_future_gt_3_dim: (B, future_len, 3)
            ego_future_gt_3_dim: torch.Tensor = outputs["ego_future_gt_3_dim"]
            ego_future_len = ego_future_gt_3_dim.shape[1]
            assert ego_future_len == args.future_len, \
                f"ego future len mismatch: {ego_future_len} vs {args.future_len}"
            # near_future_gt_3_dim: (B, agent_num, future_len, 3)
            near_future_gt_3_dim: torch.Tensor = outputs["near_future_gt_3_dim"]

            # 2) augmentation 적용
            inputs, ego_future_gt_3_dim, near_future_gt_3_dim = \
                _apply_augmentation(
                    inputs=inputs,
                    ego_future_gt_3_dim=ego_future_gt_3_dim,
                    near_future_gt_3_dim=near_future_gt_3_dim,
                    aug=aug,
                    args=args,
                )

            # 3) near future 4차원 궤적 + mask 생성
            # near_future_gt_4_dim: (B, agent_num, future_len, 4)
            # near_future_mask:    (B, agent_num, future_len)
            near_future_gt_4_dim, near_future_mask = \
                _build_near_future_4dim_and_mask(near_future_gt_3_dim)

            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)

            # 5) loss 계산 + 역전파 + optimizer/scheduler step
            if getattr(args, "use_deepspeed", False) and hasattr(
                    model, "zero_grad"):
                # DeepSpeedEngine.zero_grad()를 사용해서 분산 그라디언트 버퍼를 비운다.
                model.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            # marginal_prob 함수 핸들 얻기
            sde_marginal_prob = ddp.get_model(model, args.ddp).sde.marginal_prob
            loss_dict: Dict[str, torch.Tensor] = _compute_loss_dict(
                loss_dict=loss_dict,
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                batch_num_in_all_epoch=batch_num_in_all_epoch,
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
    epoch_mean_loss: Dict[str,
                          float] = get_epoch_mean_loss(epoch_loss_dict_list)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss,
            torch.device(args.device),
        )
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
