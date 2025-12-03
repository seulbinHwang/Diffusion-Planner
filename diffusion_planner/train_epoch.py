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

from typing import Dict, Tuple, Optional
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

    route_lanes / static_objects 는 do_not_clip=True 이면 그대로 둔다.
    """
    # 전역 스위치: True 이면 어떤 클리핑도 수행하지 않는다.
    if getattr(args, "do_not_clip", False):
        return

    # ---- neighbor_agents_past: (B, A, T, 11) → A ≤ max_agent_num ----
    max_agent_num = int(getattr(args, "max_agent_num", 0))
    if "neighbor_agents_past" in batch_on_device and max_agent_num > 0:
        neighbor_agents_past = batch_on_device["neighbor_agents_past"]
        if neighbor_agents_past.dim() == 4:
            keep_agents = min(max_agent_num, neighbor_agents_past.shape[1])
            batch_on_device["neighbor_agents_past"] = \
                neighbor_agents_past[:, :keep_agents, :, :]

    # ---- lanes / lanes_* : lane 축 → max_use_lane_num ----
    max_use_lane_num = int(getattr(args, "max_use_lane_num", 0))
    if "lanes" in batch_on_device and max_use_lane_num > 0:
        lanes = batch_on_device["lanes"]
        if lanes.dim() == 4:
            keep_lanes = min(max_use_lane_num, lanes.shape[1])
            if keep_lanes < lanes.shape[1]:
                batch_on_device["lanes"] = lanes[:, :keep_lanes, :, :]
                if "lanes_speed_limit" in batch_on_device:
                    batch_on_device["lanes_speed_limit"] = \
                        batch_on_device["lanes_speed_limit"][:, :keep_lanes, :]
                if "lanes_has_speed_limit" in batch_on_device:
                    batch_on_device["lanes_has_speed_limit"] = \
                        batch_on_device["lanes_has_speed_limit"][:, :keep_lanes, :]

    # ---- route_lanes / route_lanes_* : lane 축 → max_use_lane_num ----
    if "route_lanes" in batch_on_device and max_use_lane_num > 0:
        route_lanes = batch_on_device["route_lanes"]
        if route_lanes.dim() == 4:
            keep_route_lanes = min(max_use_lane_num, route_lanes.shape[1])
            if keep_route_lanes < route_lanes.shape[1]:
                batch_on_device["route_lanes"] = \
                    route_lanes[:, :keep_route_lanes, :, :]
                if "route_lanes_speed_limit" in batch_on_device:
                    batch_on_device["route_lanes_speed_limit"] = \
                        batch_on_device["route_lanes_speed_limit"][:, :keep_route_lanes, :]
                if "route_lanes_has_speed_limit" in batch_on_device:
                    batch_on_device["route_lanes_has_speed_limit"] = \
                        batch_on_device["route_lanes_has_speed_limit"][:, :keep_route_lanes, :]

    # ---- agent_route_lane_order: (B, A_c, L_c) → (A', L') ----
    predicted_neighbor_num = int(getattr(args, "predicted_neighbor_num", 0))
    if "agent_route_lane_order" in batch_on_device:
        arl = batch_on_device["agent_route_lane_order"]
        if arl.dim() == 3:
            bsz, c_agent, c_lane = arl.shape
            # agent 축은 predicted_neighbor_num 까지만 사용
            keep_agents_for_route = c_agent
            if predicted_neighbor_num > 0:
                keep_agents_for_route = min(predicted_neighbor_num, c_agent)

            # lane 축은 위에서 자른 lanes 의 lane 수와 맞춘다.
            lane_dim_input = c_lane
            if "lanes" in batch_on_device:
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

    # (B, A_c, Tf, 3) → (B, A', Tf, 3)
    keep_agents = min(predicted_neighbor_num, near_future_gt_3_dim.shape[1])
    outputs["near_future_gt_3_dim"] = \
        near_future_gt_3_dim[:, :keep_agents, :, :]


def _prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: str,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """배치 dict를 GPU/CPU로 옮기고, 모델 상한에 맞게 축을 잘라 입력/정답을 나눈다.

    이 함수는 DataLoader에서 넘어온 배치를:

      1) 지정한 device 로 옮기고,
      2) 모델 설정 상한(max_agent_num, predicted_neighbor_num, max_use_lane_num)에 맞춰
         agent / lane 관련 축만 앞쪽에서 잘라 준 뒤,
      3) 정답 텐서(ego_future_gt_3_dim, near_future_gt_3_dim)를 따로 분리한다.

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


def train_epoch(data_loader,
                model,
                optimizer,
                args,
                ema,
                scheduler,
                aug: StatePerturbation = None):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    # ---- 전역 스텝/총 스텝 계산(진행도 p) ----
    # 동일 args 객체를 통해 에폭 간 누적 유지
    if not hasattr(args, "_global_update_step"):
        args._global_update_step = 0
    total_step_of_this_epoch = len(data_loader)
    total_update_steps = max(1, args.train_epochs * total_step_of_this_epoch)

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            inputs, outputs = _prepare_batch_for_device(batch, args.device,
                                                        args)
            # ego_future_gt_3_dim: (B, 1 + future_len, 3)
            ego_future_gt_3_dim = outputs["ego_future_gt_3_dim"]
            # near_future_gt_3_dim: (B, predicted_neighbor_num, future_len, 3)
            near_future_gt_3_dim = outputs["near_future_gt_3_dim"]
            # ================================================
            # Normalize to ego-centric
            if isinstance(aug, StatePerturbation):
                inputs, ego_future_gt_3_dim, near_future_gt_3_dim = aug(
                    inputs, ego_future_gt_3_dim, near_future_gt_3_dim)
            if isinstance(aug, NPCStatePerturbation):
                inputs, near_future_gt_3_dim = aug(inputs, near_future_gt_3_dim,
                                                   args)

            # near_future_mask: (B, predicted_neighbor_num, future_len) # if all zeros in (x, y, yaw), then near_future_mask is True.
            near_future_mask = torch.sum(torch.ne(near_future_gt_3_dim[..., :3],
                                                  0),
                                         dim=-1) == 0

            # (B, predicted_neighbor_num, future_len, 3) -> (B, predicted_neighbor_num, future_len, 4)
            near_future_gt_4_dim = torch.cat(
                [
                    near_future_gt_3_dim[..., :2],
                    torch.stack([
                        near_future_gt_3_dim[..., 2].cos(),
                        near_future_gt_3_dim[..., 2].sin()
                    ],
                                dim=-1),
                ],
                dim=-1,
            )
            near_future_gt_4_dim[near_future_mask] = 0.
            if not torch.isfinite(near_future_gt_4_dim).all():
                raise ValueError(
                    "Non-finite values detected in near_future_gt_4_dim")
            norm_inputs = args.observation_normalizer(inputs)

            # call the model
            optimizer.zero_grad(set_to_none=True)
            loss = {}
            """
            near_future_gt_4_dim.shape: [8, 10, 80, 4]
            near_future_mask.shape: [8, 10, 80]
            """
            loss, _ = diffusion_loss_func(
                args, model, norm_inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (near_future_gt_4_dim, near_future_mask), args.state_normalizer,
                loss, args.diffusion_model_type, args.observation_normalizer)

            # ----- (NEW) 진행도 기반 가중 합성 --------------------------------
            progress = min(
                1.0, args._global_update_step /
                float(max(1, total_update_steps - 1)))
            w_dir, w_int, w_const = FeasibleProjector.loss_weights_by_progress(
                progress)
            if not args.use_direct_loss:
                w_dir = 0.0
                w_int = 1.

            # <추가하자> 에폭 평균으로 보고하기 위해 loss dict에 저장(이 값들을 나중에 lr/*로 로그)
            loss["feasible_progress"] = float(progress)
            loss["feasible_w_dir"] = float(w_dir)
            loss["feasible_w_int"] = float(w_int)
            loss["feasible_w_const"] = float(w_const)

            # 개별 손실이 존재하지 않는 경우(예: score 모드) 대비 안전 get
            l_dir = loss.get(
                "neighbor_prediction_loss",
                torch.tensor(0.0, device=inputs["ego_agent_past"].device))
            l_int = loss.get(
                "integration_loss",
                torch.tensor(0.0, device=inputs["ego_agent_past"].device))
            l_con = loss.get(
                "constraint_loss",
                torch.tensor(0.0, device=inputs["ego_agent_past"].device))
            loss["loss"] = w_dir * l_dir + w_int * l_int + w_const * l_con

            total_loss = loss["loss"].item()  # scalar

            # loss backward
            loss["loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            scheduler.step()
            optimizer.step()
            # === WD warmdown: lr 비례로 그룹별 WD 갱신 ===
            for pg in optimizer.param_groups:
                wd_max = pg.get("wd_max", None)
                lr_max = pg.get("lr_max", None)
                if wd_max is None or lr_max is None:
                    continue  # 안전 장치
                if wd_max > 0.0 and lr_max > 0.0:
                    pg["weight_decay"] = wd_max * (pg["lr"] / lr_max)
            # ===========================================

            if ema is not None:
                """
                수식대로 EMA 가중치를 한 번 갱신
                """
                ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss.append(loss)

            # 전역 스텝 누적(에폭 간 유지)
            args._global_update_step += 1

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss, torch.device(args.device))
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
