from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def _compute_xy_yaw_losses(
        score: torch.Tensor, near_future_norm_gt: torch.Tensor,
        near_future_valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute separate XY and Yaw RMSE losses for ego and neighbors.

    Args:
        score (Tensor(B, Pnn, T, 4)):
            Model output trajectories, where last dim is [dx, dy, cos(yaw), sin(yaw)].
        near_future_norm_gt (Tensor[B, Pnn, T, 4]):
            Ground truth trajectories in same format as score.
        near_future_valid (BoolTensor[B, Pnn, T]):
            Mask for valid neighbor entries (excludes ego at index 0).
    Returns:
        Dict with:
        - 'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        - 'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
    """
    # score[..., :2]: Tensor[B, Pnn, T, 2] -> (x, y)
    pred_xy = score[..., :2]  # [B, Pnn, T, 2]
    gt_xy = near_future_norm_gt[..., :2]
    # Euclidean distance: sqrt((dx)^2 + (dy)^2)
    dist_ = torch.sqrt(((pred_xy - gt_xy).pow(2).sum(-1)) + 1e-6)  # [B, Pnn, T]
    # neighbors: exclude ego index 0
    valid_dist = dist_[near_future_valid]  # [num_valid]
    valid_dist_mean = valid_dist.mean() if valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    # Compute yaw angles from cos/sin
    pred_cos = score[..., 2]  # [B, P, T]
    pred_sin = score[..., 3]
    gt_cos = near_future_norm_gt[..., 2]
    gt_sin = near_future_norm_gt[..., 3]
    yaw_pred = torch.atan2(pred_sin, pred_cos)  # [B, P, T]
    yaw_gt = torch.atan2(gt_sin, gt_cos)
    # Angular error wrapped to [-pi, pi]
    yaw_err = (yaw_pred - yaw_gt +
               torch.pi) % (2 * torch.pi) - torch.pi  # [B, P, T]
    dist_yaw = torch.abs(yaw_err)  # abs error in radians
    masked_yaw = dist_yaw[near_future_valid]
    neigh_yaw = masked_yaw.mean() if masked_yaw.numel() > 0 else torch.tensor(
        0.0, device=dist_yaw.device)

    return {
        'neighbor_prediction_loss_xy': valid_dist_mean,  # scalar
        'neighbor_prediction_loss_yaw': neigh_yaw,  # scalar
    }


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],
    futures: Tuple[torch.Tensor, torch.Tensor],
    norm: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    eps: float = 1e-3,
):
    """
    ego_future.shape: [8, 80, 4] # [B, T, 4]
    near_future_gt.shape: [8, Pnn, 80, 4] # [B, Pnn, T, 4]
    near_future_mask.shape: [8, Pnn, 80] # [B, Pnn, T]
    """
    near_future_gt, near_future_mask = futures
    # ego_future: [B. T, 4]
    # near_future_gt: [B, Pnn, T]
    near_future_valid = ~near_future_mask

    B, Pnn, T, _ = near_future_gt.shape
    # ego_current: [B, 4]
    # neighbors_current: [B, Pnn, 4]
    near_current_xyyaw = inputs["neighbor_agents_past"][:, :Pnn, -1, :4]
    # near_current_mask: [B, Pnn]
    near_current_mask = torch.sum(torch.ne(near_current_xyyaw[..., :4], 0),
                                  dim=-1) == 0  # [B, Pnn]
    # near_future_mask: (B, Pnn, T)
    # near_cur_future_mask: [B, Pnn, 1+T]
    near_cur_future_mask = torch.concat(
        (near_current_mask.unsqueeze(-1), near_future_mask), dim=-1)
    # near_future_gt: [B, Pnn, T, 4]
    # near_current_xyyaw: [B, Pnn, 4]
    # batch_diffusion_time: [B,] diffusion time uniformly sampled in [eps, 1]
    # random_noise: [B, Pnn + 1, T, 4] noise sampled from standard normal
    batch_diffusion_time = torch.rand(
        B, device=near_future_gt.device) * (1 - eps) + eps  # [B,]
    random_noise = torch.randn_like(
        near_future_gt, device=near_future_gt.device)  # [B, Pnn, T, 4]

    # near_cur_future_norm_gt: [B, Pnn, 1+T, 4]
    near_cur_future_norm_gt = torch.cat(
        [near_current_xyyaw[:, :, None, :],
         norm(near_future_gt)], dim=2)  # [B, Pnn, 1 + T, 4]
    # near_cur_future_mask: [B, Pnn, 1+T]
    near_cur_future_norm_gt[near_cur_future_mask] = 0.0
    near_future_norm_gt = near_cur_future_norm_gt[:, :, 1:, :]  # [B, Pnn, T, 4]
    """
    <forward pass>
    
    
    # mean.shape: torch.Size([B, Pnn, T, 4])
    # std.shape: torch.Size([B, 1, 1, 1])
    """
    mean, std = marginal_prob(near_future_norm_gt, batch_diffusion_time)
    # std.shape after view: torch.Size([B, 1, 1, 1])

    std = std.view(-1, *([1] * (len(near_future_norm_gt.shape) - 1)))
    #  near_future_norm_xT.shape: torch.Size([B, Pnn, T, 4])
    near_future_norm_xT = mean + std * random_noise
    # near_cur_future_norm_xT.shape after concat: torch.Size([B, Pnn, 1+T, 4])
    near_cur_future_norm_xT = torch.cat(
        [near_cur_future_norm_gt[:, :, :1, :], near_future_norm_xT], dim=2)

    merged_inputs = {
        **inputs,
        "near_cur_future_norm_xT": near_cur_future_norm_xT,  # [B, Pnn, 1 + T, 4]
        "diffusion_time": batch_diffusion_time,  # [B,]
    }

    _, decoder_output = model(merged_inputs)
    # decoder_output["score"]: (B, Pnn, (1 + T) , 4)
    score = decoder_output["score"][:, :, 1:, :]  # (B, Pnn, T, 4)

    if model_type == "score":
        dpm_loss = torch.sum((score * std + random_noise)**2, dim=-1)
    elif model_type == "x_start":
        # near_future_norm_gt: [B, Pnn, T, 4]
        # dpm_loss: (B, Pnn, T)
        dpm_loss = torch.sum((score - near_future_norm_gt)**2, dim=-1)
    # near_future_valid: [B, Pnn, T]
    masked_prediction_loss = dpm_loss[near_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean(
        )  # float
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(
            0.0, device=masked_prediction_loss.device)

    # compute and merge xy/yaw losses via helper
    xy_yaw_losses = _compute_xy_yaw_losses(score, near_future_norm_gt,
                                           near_future_valid)
    loss.update(xy_yaw_losses)

    assert not torch.isnan(
        dpm_loss).sum(), f"loss cannot be nan, random_noise={random_noise}"
    """
    loss
        "neighbor_prediction_loss" (float): mean RMSE over neighbor coords.
        'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
    """
    return loss, decoder_output
