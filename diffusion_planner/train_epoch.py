from tqdm import tqdm
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation

def train_epoch(data_loader,
                model,
                optimizer,
                args,
                ema,
                aug: StatePerturbation = None):
    epoch_loss = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()
    print("0")
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        print("1")
        for batch in data_epoch:
            print("2")
            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt, = ego_agent_future

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            inputs = {
                'ego_current_state': batch[0].to(args.device),
                'neighbor_agents_past': batch[2].to(args.device),
                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),
                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),
                'static_objects': batch[10].to(args.device),
            }

            ego_future = batch[1].to(args.device)
            neighbors_future = batch[3].to(args.device)
            neighbors_future_all = batch[11].to(args.device)
            # Normalize to ego-centric
            if isinstance(aug, StatePerturbation):
                inputs, ego_future, neighbors_future = aug(
                    inputs, ego_future, neighbors_future)
            if isinstance(aug, NPCStatePerturbation):
                inputs, neighbors_future = aug(
                    inputs, neighbors_future_all, args)
            #
            #
            # # heading to cos sin
            # ego_future = torch.cat(
            #     [
            #         ego_future[..., :2],
            #         torch.stack(
            #             [ego_future[..., 2].cos(), ego_future[..., 2].sin()],
            #             dim=-1),
            #     ],
            #     dim=-1,
            # )
            #
            # mask = torch.sum(torch.ne(neighbors_future[..., :3], 0),
            #                  dim=-1) == 0
            # neighbors_future = torch.cat(
            #     [
            #         neighbors_future[..., :2],
            #         torch.stack([
            #             neighbors_future[..., 2].cos(),
            #             neighbors_future[..., 2].sin()
            #         ],
            #                     dim=-1),
            #     ],
            #     dim=-1,
            # )
            # neighbors_future[mask] = 0.
            # inputs = args.observation_normalizer(inputs)
            #
            # # call the mdoel
            # optimizer.zero_grad()
            # loss = {}
            #
            # loss, _ = diffusion_loss_func(
            #     model, inputs,
            #     ddp.get_model(model, args.ddp).sde.marginal_prob,
            #     (ego_future, neighbors_future, mask), args.state_normalizer,
            #     loss, args.diffusion_model_type)
            #
            # loss['loss'] = loss[
            #     'neighbor_prediction_loss'] + args.alpha_planning_loss * loss[
            #         'ego_planning_loss']
            #
            # total_loss = loss['loss'].item()
            #
            # # loss backward
            # loss['loss'].backward()
            #
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            # optimizer.step()
            #
            # ema.update(model)
            #
            # if args.ddp:
            #     torch.cuda.synchronize()
            #
            # data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))
            # epoch_loss.append(loss)
    raise NotImplementedError("The training loop is not fully implemented yet.")
    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss, torch.device(args.device))
    print("3")
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss['loss']
