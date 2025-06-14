from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer


def _compute_xy_yaw_losses(score: torch.Tensor, gt: torch.Tensor,
                           mask_valid: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Compute separate XY and Yaw RMSE losses for ego and neighbors.

    Args:
        score (Tensor[B, P, T, 4]):
            Model output trajectories, where last dim is [dx, dy, cos(yaw), sin(yaw)].
        gt (Tensor[B, P, T, 4]):
            Ground truth trajectories in same format as score.
        mask_valid (BoolTensor[B, P-1, T]):
            Mask for valid neighbor entries (excludes ego at index 0).
    Returns:
        Dict with:
        - 'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        - 'ego_planning_loss_xy' (float): mean Euclidean distance over ego coords.
        - 'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
        - 'ego_planning_loss_yaw' (float): mean abs angular error (rad) for ego.
    """
    # score[..., :2]: Tensor[B, P, T, 2] -> (x, y)
    pred_xy = score[..., :2] # [B, P, T, 2]
    gt_xy = gt[..., :2]
    # Euclidean distance: sqrt((dx)^2 + (dy)^2)
    dist_xy = torch.sqrt(((pred_xy - gt_xy).pow(2).sum(-1)) + 1e-6)  # [B, P, T]
    # neighbors: exclude ego index 0
    masked_xy = dist_xy[:, 1:, :][mask_valid]  # [num_valid]
    neigh_xy = masked_xy.mean() if masked_xy.numel() > 0 else torch.tensor(
        0.0, device=dist_xy.device)
    ego_xy = dist_xy[:, 0, :].mean()

    # Compute yaw angles from cos/sin
    pred_cos = score[..., 2]  # [B, P, T]
    pred_sin = score[..., 3]
    gt_cos = gt[..., 2]
    gt_sin = gt[..., 3]
    yaw_pred = torch.atan2(pred_sin, pred_cos)  # [B, P, T]
    yaw_gt = torch.atan2(gt_sin, gt_cos)
    # Angular error wrapped to [-pi, pi]
    yaw_err = (yaw_pred - yaw_gt +
               torch.pi) % (2 * torch.pi) - torch.pi  # [B, P, T]
    dist_yaw = torch.abs(yaw_err)  # abs error in radians
    masked_yaw = dist_yaw[:, 1:, :][mask_valid]
    neigh_yaw = masked_yaw.mean() if masked_yaw.numel() > 0 else torch.tensor(
        0.0, device=dist_yaw.device)
    ego_yaw = dist_yaw[:, 0, :].mean()

    return {
        'neighbor_prediction_loss_xy': neigh_xy,
        'ego_planning_loss_xy': ego_xy,
        'neighbor_prediction_loss_yaw': neigh_yaw,
        'ego_planning_loss_yaw': ego_yaw
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
    ego_future, neighbors_future, neighbor_future_mask = futures
    # ego_future: [B. T, 4]
    # neighbors_future: [B, Pn, T, 4]
    neighbors_future_valid = ~neighbor_future_mask  # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs[
        "neighbor_agents_past"][:, :Pn, -1, :4]
    # ego_current: [B, 4]
    # neighbors_current: [B, Pn, 4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0),
                                      dim=-1) == 0
    neighbor_mask = torch.concat(
        (neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]],
                          dim=1)  # [B, P = 1 + Pn, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current],
                               dim=1)  # [B, P, 4]

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps  # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device)  # [B, P, T, 4]

    all_gt = torch.cat([current_states[:, :, None, :],
                        norm(gt_future)], dim=2)  # [B, P, 1 + T, 4]
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape) - 1)))

    xT = mean + std * z  # xT: [B, P, T, 4]
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)  # [B, P, 1 + T, 4]

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,  # [B, P, 1 + T, 4]
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs)  # [B, P, 1 + T, 4]
    score = decoder_output["score"][:, :, 1:, :]  # [B, P, T, 4]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)

    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(
            0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    # compute and merge xy/yaw losses via helper
    gt = all_gt[..., 1:, :]
    xy_yaw_losses = _compute_xy_yaw_losses(score, gt, neighbors_future_valid)
    loss.update(xy_yaw_losses)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output
