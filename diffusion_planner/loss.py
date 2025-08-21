from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def _compute_xy_yaw_losses(
        score: torch.Tensor, neighbor_future_gt: torch.Tensor,
        neighbors_future_valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute separate XY and Yaw RMSE losses for ego and neighbors.

    Args:
        score (Tensor(B, Pnn, T, 4)):
            Model output trajectories, where last dim is [dx, dy, cos(yaw), sin(yaw)].
        neighbor_future_gt (Tensor[B, Pnn, T, 4]):
            Ground truth trajectories in same format as score.
        neighbors_future_valid (BoolTensor[B, Pnn, T]):
            Mask for valid neighbor entries (excludes ego at index 0).
    Returns:
        Dict with:
        - 'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        - 'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
    """
    # score[..., :2]: Tensor[B, Pnn, T, 2] -> (x, y)
    pred_xy = score[..., :2]  # [B, Pnn, T, 2]
    gt_xy = neighbor_future_gt[..., :2]
    # Euclidean distance: sqrt((dx)^2 + (dy)^2)
    dist_ = torch.sqrt(((pred_xy - gt_xy).pow(2).sum(-1)) + 1e-6)  # [B, Pnn, T]
    # neighbors: exclude ego index 0
    valid_dist = dist_[neighbors_future_valid]  # [num_valid]
    valid_dist_mean = valid_dist.mean() if valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    # Compute yaw angles from cos/sin
    pred_cos = score[..., 2]  # [B, P, T]
    pred_sin = score[..., 3]
    gt_cos = neighbor_future_gt[..., 2]
    gt_sin = neighbor_future_gt[..., 3]
    yaw_pred = torch.atan2(pred_sin, pred_cos)  # [B, P, T]
    yaw_gt = torch.atan2(gt_sin, gt_cos)
    # Angular error wrapped to [-pi, pi]
    yaw_err = (yaw_pred - yaw_gt +
               torch.pi) % (2 * torch.pi) - torch.pi  # [B, P, T]
    dist_yaw = torch.abs(yaw_err)  # abs error in radians
    masked_yaw = dist_yaw[neighbors_future_valid]
    neigh_yaw = masked_yaw.mean() if masked_yaw.numel() > 0 else torch.tensor(
        0.0, device=dist_yaw.device)

    return {
        'neighbor_prediction_loss_xy': valid_dist_mean, # scalar
        'neighbor_prediction_loss_yaw': neigh_yaw, # scalar
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
    neighbors_future.shape: [8, 10, 80, 4] # [B, Pnn, T, 4]
    neighbor_future_mask.shape: [8, 10, 80] # [B, Pnn, T]
    """
    ego_future, neighbors_future, neighbor_future_mask = futures
    # ego_future: [B. T, 4]
    # neighbors_future: [B, Pnn, T]
    neighbors_future_valid = ~neighbor_future_mask

    B, Pnn, T, _ = neighbors_future.shape
    # ego_current: [B, 4]
    # neighbors_current: [B, Pnn, 4]
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs[
        "neighbor_agents_past"][:, :Pnn, -1, :4]
    # neighbor_current_mask: [B, Pnn]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0),
                                      dim=-1) == 0  # [B, Pnn]
    # neighbor_future_mask: (B, Pnn, T)
    # neighbor_mask: [B, Pnn, 1+T]
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)
    # neighbors_future: [B, Pnn, T, 4]
    # neighbors_current: [B, Pnn, 4]
    # t: [B,] diffusion time uniformly sampled in [eps, 1]
    # z: [B, Pnn + 1, T, 4] noise sampled from standard normal
    t = torch.rand(B, device=neighbors_future.device) * (1 - eps) + eps  # [B,]
    z = torch.randn_like(neighbors_future,
                         device=neighbors_future.device)  # [B, Pnn, T, 4]

    # neighbor_cur_future_gt: [B, Pnn, 1+T, 4]
    neighbor_cur_future_gt = torch.cat(
        [neighbors_current[:, :, None, :],
         norm(neighbors_future)], dim=2)  # [B, Pnn, 1 + T, 4]
    # neighbor_mask: [B, Pnn, 1+T]
    neighbor_cur_future_gt[neighbor_mask] = 0.0
    neighbor_future_gt = neighbor_cur_future_gt[:, :, 1:, :]  # [B, Pnn, T, 4]
    """
    <forward pass>
    
    
    # mean.shape: torch.Size([B, Pnn, T, 4])
    # std.shape: torch.Size([B, 1, 1, 1])
    """
    mean, std = marginal_prob(neighbor_future_gt, t)
    # std.shape after view: torch.Size([B, 1, 1, 1])

    std = std.view(-1, *([1] * (len(neighbor_future_gt.shape) - 1)))
    #  xT.shape: torch.Size([B, Pnn, T, 4])
    xT = mean + std * z
    # xT.shape after concat: torch.Size([B, Pnn, 1+T, 4])
    xT = torch.cat([neighbor_cur_future_gt[:, :, :1, :], xT], dim=2)

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,  # [B, Pnn, 1 + T, 4]
        "diffusion_time": t,  # [B,]
    }

    _, decoder_output = model(merged_inputs)
    ####### TODO: 아래부터 다시 검증
    # decoder_output["score"]: (B, Pnn, (1 + T) * 4)
    score = decoder_output["score"][:, :, 1:, :]  # (B, Pnn, T, 4)

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        # neighbor_future_gt: [B, Pnn, T, 4]
        # dpm_loss: (B, Pnn, T)
        dpm_loss = torch.sum((score - neighbor_future_gt)**2, dim=-1)
    # neighbors_future_valid: [B, Pnn, T]
    masked_prediction_loss = dpm_loss[neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean() # float
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(
            0.0, device=masked_prediction_loss.device)

    # compute and merge xy/yaw losses via helper
    xy_yaw_losses = _compute_xy_yaw_losses(score, neighbor_future_gt,
                                           neighbors_future_valid)
    loss.update(xy_yaw_losses)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"
    """
    loss
        "neighbor_prediction_loss" (float): mean RMSE over neighbor coords.
        'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
    """
    return loss, decoder_output
