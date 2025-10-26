from tqdm import tqdm
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation

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
                    batch[13].to(args.device, dtype=torch.long, non_blocking=True),
            }

            ego_future_gt_3_dim = batch[2].to(args.device, non_blocking=True)
            near_future_gt_3_dim = batch[11].to(
                args.device, non_blocking=True)  # (B, predicted_neighbor_num, future_len, 3)
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
                model, norm_inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (near_future_gt_4_dim, near_future_mask), args.state_normalizer,
                loss, args.diffusion_model_type)
            loss["loss"] = loss["neighbor_prediction_loss"]

            total_loss = loss["loss"].item()  # scalar

            # loss backward
            loss["loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 15)
            scheduler.step()
            optimizer.step()

            if ema is not None:
                ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss, torch.device(args.device))
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
