from tqdm import tqdm
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype


# === [ADD] 간단한 Pnn 커리큘럼 함수 (에폭 기반) =========================
def _compute_pnn_curriculum(epoch_idx: int,
                            pnn_max: int,
                            pnn_min: int = 32,
                            step: int = 32) -> int:
    """
    예: epoch 별로 32 -> 64 -> 96 -> ... 식으로 증가.
    - epoch_idx: 0부터 시작한다고 가정
    - pnn_max: 최종 상한(보통 캐시/설정의 predicted_neighbor_num 최대값)
    - pnn_min: 시작값(기본 32)
    - step: 증가 간격(기본 32)
    """
    target = pnn_min + epoch_idx * step
    return int(min(pnn_max, max(pnn_min, target)))


# =====================================================================


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
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            '''
            data structure in batch: Tuple(Tensor) 
            
            ego_agent_past,
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
            neighbors_future_gt_all
            ego_future_gt_11_dim, = ego_agent_future_11_dim


            '''

            # prepare data
            inputs = {
                "ego_agent_past":
                    batch[0].to(args.device),
                "ego_current_state":
                    batch[1].to(args.device),
                "neighbor_agents_past":
                    batch[3].to(args.device),
                "lanes":
                    batch[5].to(args.device),
                "lanes_speed_limit":
                    batch[6].to(args.device),
                "lanes_has_speed_limit":
                    batch[7].to(args.device),
                "route_lanes":
                    batch[8].to(args.device),
                "route_lanes_speed_limit":
                    batch[9].to(args.device),
                "route_lanes_has_speed_limit":
                    batch[10].to(args.device),
                "static_objects":
                    batch[11].to(args.device),
                "ego_future_gt_11_dim":
                    batch[13].to(args.device),
                "agent_route_lane_order":
                    batch[14].to(args.device, dtype=torch.long),
            }

            ego_future = batch[2].to(args.device)
            # ["neighbor_agents_future"][:self._predicted_neighbor_num]
            # (B, num_agents, future_len, 3) -> (B, predicted_neighbor_num, future_len, 3)
            neighbors_future = batch[4].to(args.device)
            neighbors_future_all = batch[12].to(args.device)
            # Normalize to ego-centric
            if isinstance(aug, StatePerturbation):
                inputs, ego_future, neighbors_future = aug(
                    inputs, ego_future, neighbors_future)
            if isinstance(aug, NPCStatePerturbation):
                inputs, neighbors_future = aug(inputs, neighbors_future_all,
                                               args)
            # --- (증강 끝난 뒤) 여기서 커리큘럼 Pnn/컨텍스트 클립 적용 ---
            # 현재 에폭의 Pnn (없으면 기본 predicted_neighbor_num 사용)
            pnn_curr = int(
                getattr(args, "curr_predicted_neighbor_num",
                        args.predicted_neighbor_num))

            # 이 배치에서 실제 사용할 Pnn (해당 샘플에 존재하는 최대치로 clamp)
            # 컨텍스트 에이전트는 ego-거리순으로 Top-clip_num만 사용
            clip_num = min(args.agent_num, pnn_curr * 2)

            # 1) 예측 대상의 GT(미래)만 Pnn_eff로 제한  → 손실·디코더의 Q 크기 결정
            neighbors_future = neighbors_future[:, :
                                                pnn_curr]  # (B, pnn_curr, T, 3)

            # 2) route conditioning도 Pnn_eff로 제한  → 디코더가 Pnn을 이 길이에 맞춤
            inputs["agent_route_lane_order"] = inputs[
                "agent_route_lane_order"][:, :pnn_curr, ...] # (B, pnn_curr, lane_num)
            # (이 값은 Encoder에서 (B, pnn_curr, ...) 경로 임베딩으로 변환되어
            #  decoder.forward에서 Pnn 동적 길이의 기준으로 쓰임)

            # 3) 컨텍스트 에이전트는 Top-clip_num만 남기고 나머지는 0으로 마스킹
            #    (인코더의 varlen 경로가 무효 토큰은 완전히 건너뜀 → 연산량↓)
            if clip_num < inputs["neighbor_agents_past"].shape[1]:
                inputs["neighbor_agents_past"][:, clip_num:, :, :] = 0.0
            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0),
                             dim=-1) == 0
            # (B, predicted_neighbor_num, future_len, 3) -> (B, predicted_neighbor_num, future_len, 4)
            neighbors_future = torch.cat(
                [
                    neighbors_future[..., :2],
                    torch.stack([
                        neighbors_future[..., 2].cos(),
                        neighbors_future[..., 2].sin()
                    ],
                                dim=-1),
                ],
                dim=-1,
            )
            neighbors_future[mask] = 0.
            if not torch.isfinite(neighbors_future).all():
                raise ValueError(
                    "Non-finite values detected in neighbors_future")
            norm_inputs = args.observation_normalizer(inputs)

            # call the model
            optimizer.zero_grad(set_to_none=True)
            loss = {}
            """
            ego_future.shape: [8, 80, 4]
            neighbors_future.shape: [8, 10, 80, 4]
            mask.shape: [8, 10, 80]
            """
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                loss, _ = diffusion_loss_func(
                    model, norm_inputs,
                    ddp.get_model(model, args.ddp).sde.marginal_prob,
                    (neighbors_future, mask), args.state_normalizer, loss,
                    args.diffusion_model_type)
                loss["loss"] = loss["neighbor_prediction_loss"]

            total_loss = loss["loss"].item()  # scalar

            # loss backward
            loss["loss"].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
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
