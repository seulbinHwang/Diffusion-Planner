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
from diffusion_planner.loss import AMP_DTYPE  # = torch.bfloat16

# =====================================================================


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
            # prepare data
            inputs = {
                "ego_agent_past":
                    batch[0].to(args.device, non_blocking=True),
                # "ego_current_state":
                #     batch[1].to(args.device),
                "neighbor_agents_past":
                    batch[3].to(args.device, non_blocking=True),
                "lanes":
                    batch[4].to(args.device, non_blocking=True),
                "lanes_speed_limit":
                    batch[5].to(args.device, non_blocking=True),
                "lanes_has_speed_limit":
                    batch[6].to(args.device, non_blocking=True),
                "route_lanes":
                    batch[7].to(args.device, non_blocking=True),
                "route_lanes_speed_limit":
                    batch[8].to(args.device, non_blocking=True),
                "route_lanes_has_speed_limit":
                    batch[9].to(args.device, non_blocking=True),
                "static_objects":
                    batch[10].to(args.device, non_blocking=True),
                "planner_future_11_dim":
                    batch[12].to(args.device, non_blocking=True),
                "agent_route_lane_order":
                    batch[13].to(args.device,
                                 dtype=torch.long,
                                 non_blocking=True),
            }

            ego_future_gt_3_dim = batch[2].to(args.device, non_blocking=True)
            near_future_gt_3_dim = batch[11].to(
                args.device,
                non_blocking=True)  # (B, predicted_neighbor_num, future_len, 3)
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
            state_device = near_future_gt_4_dim.device

            # ----- (NEW) 진행도 기반 가중 합성 --------------------------------
            progress = min(
                1.0, args._global_update_step /
                float(max(1, total_update_steps - 1)))
            w_dir, w_int, w_const = FeasibleProjector.loss_weights_by_progress(
                progress)
            # 개별 손실이 존재하지 않는 경우(예: score 모드) 대비 안전 get
            l_dir = loss.get("neighbor_prediction_loss",
                             torch.tensor(0.0, device=state_device))
            l_int = loss.get("integration_loss",
                             torch.tensor(0.0, device=state_device))
            l_con = loss.get("constraint_loss",
                             torch.tensor(0.0, device=state_device))
            total = w_dir * l_dir + w_int * l_int + w_const * l_con
            # if not torch.isfinite(total):
            #     # feasible만 스킵하고 DiT는 진행
            #     total = loss[
            #         "neighbor_prediction_loss"]  # 또는 integration/constraint 제외
            #     print("integration_loss:", l_int, "constraint_loss:", l_con)
            loss["loss"] = total

            total_loss = loss["loss"].item()  # scalar

            # loss backward
            loss["loss"].backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(
                ddp.get_model(model,
                              args.ddp).decoder.decoder.dit.feasible_projector.parameters(),
                max_norm=1.0  # 1~5 시도
            )
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
            torch.cuda.synchronize()
            if ema is not None:
                """
                수식대로 EMA 가중치를 한 번 갱신
                """
                ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss.append({
                k: (v.detach().item()
                    if torch.is_tensor(v) and v.numel() == 1 else
                    (v.detach().float().mean().item()
                     if torch.is_tensor(v) else v)) for k, v in loss.items()
            })

            # 전역 스텝 누적(에폭 간 유지)
            args._global_update_step += 1

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss, torch.device(args.device))
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
