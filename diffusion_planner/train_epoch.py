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

from typing import Dict, Tuple
import torch
from torch import nn
from typing import Tuple
...
from diffusion_planner.model.module.feasible import FeasibleProjector
# =====================================================================


def _prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """배치 dict를 GPU/CPU로 옮기고, 입력/정답을 구분해 준다.

    이 함수는 Dataset에서 넘어온 배치(dict)를 한 번에 원하는 장치로 옮기고,
    그중에서 정답 역할을 하는 텐서만 outputs로 분리한다.

    Args:
        batch:
            Dataset에서 온 배치.
            각 값의 대략적인 형태는 다음과 같다.
            - "ego_agent_past"            : (B, time_len, 11)
            - "ego_future_gt_3_dim"       : (B, 1 + future_len, 3)
            - "neighbor_agents_past"      : (B, max_agent_num, time_len, 11)
            - "lanes"                     : (B, lane_num, lane_len, 12)
            - "lanes_speed_limit"         : (B, lane_num, 1)
            - "lanes_has_speed_limit"     : (B, lane_num, 1)
            - "route_lanes"               : (B, route_lane_num, route_len, 12)
            - "route_lanes_speed_limit"   : (B, route_lane_num, 1)
            - "route_lanes_has_speed_limit": (B, route_lane_num, 1)
            - "static_objects"            : (B, max_static_num, 10)
            - "near_future_gt_3_dim"      : (B, Pnn, future_len, 3)
            - "planner_future_11_dim"     : (B, future_len, 11)
            - "agent_route_lane_order"    : (B, Pnn, lane_num)
        device:
            텐서를 옮길 대상 장치. 예: "cuda:0", "cpu" 등.

    Returns:
        inputs:
            모델 forward에 바로 넣을 입력 텐서 dict.
            (정답 텐서를 제외한 나머지 키/값, shape는 위와 동일하게 (B, ...) 구조)
        outputs:
            정답 텐서 dict.
            - "ego_future_gt_3_dim"  : (B, 1 + future_len, 3)
            - "near_future_gt_3_dim" : (B, Pnn, future_len, 3)
    """
    # 1) 배치 전체를 장치로 이동
    batch_on_device: Dict[str, torch.Tensor] = {
        key: tensor.to(device, non_blocking=True)
        for key, tensor in batch.items()
    }

    # 2) dtype 특수 처리: route lane order는 항상 long 형으로 맞춘다.
    if "agent_route_lane_order" in batch_on_device:
        # (B, Pnn, lane_num)
        batch_on_device["agent_route_lane_order"] = batch_on_device[
            "agent_route_lane_order"].long()

    # 3) 정답 키만 outputs로 분리
    target_keys = {"ego_future_gt_3_dim", "near_future_gt_3_dim"}
    outputs: Dict[str, torch.Tensor] = {}

    for key in list(batch_on_device.keys()):
        if key in target_keys:
            outputs[key] = batch_on_device.pop(key)

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
            inputs, outputs = _prepare_batch_for_device(batch, args.device)
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
