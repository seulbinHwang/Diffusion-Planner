from typing import Any, Callable, Dict, List, Tuple
import logging

import torch
import torch.nn as nn

from diffusion_planner.utils.normalizer import StateNormalizer

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype


def _require_finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Ensure ``tensor`` has no NaN or Inf values.

    Args:
        name: Name of the tensor for logging.
        tensor: Tensor to validate.

    Returns:
        The original tensor if it contains only finite values.

    Raises:
        ValueError: If NaN or Inf values are detected in the tensor.
    """
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        msg = f"{name} contains NaN or Inf values"
        logging.error(msg)
        raise ValueError(msg)
    return tensor


# [add] ----------------------------------------------------------------------
def _build_half_life_weights(
    T: int,  # 80
    *,
    dt_s: float = 0.1,
    half_life_s: float = 2.0,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """half-life 기반 시간 가중치 텐서를 생성합니다.

    각 시간 스텝 j(0-index)에 대해 t_j = (j+1)*dt_s 로 두고,
    w_j = 0.5 ** (t_j / half_life_s) 로 정의합니다.
    예) half_life_s=2.0이면, 2초→1/2, 4초→1/4, 8초→1/16.

    Args:
        T (int): 미래 스텝 수(프레임 수).
        dt_s (float): 프레임 간 시간 간격(초). 기본 0.1초.
        half_life_s (float): half-life(초).
        device (torch.device): 출력 텐서가 위치할 디바이스.
        dtype (torch.dtype): 출력 텐서 dtype.

    Returns:
        torch.Tensor: (1, 1, T) 모양의 가중치 텐서.  # shape: [1, 1, T]

    Note:
        - 시간축은 [dt_s, 2*dt_s, ..., T*dt_s].
        - 브로드캐스트를 위해 (1,1,T)로 반환합니다.
    """
    # t: (T,) = [dt_s, 2*dt_s, ..., T*dt_s]
    t = torch.arange(1, T + 1, device=device, dtype=dtype) * float(dt_s)  # [T]
    # w: (T,) = 0.5 ** (t / half_life_s)
    w = torch.pow(0.5, t / float(half_life_s))  # [T]
    return w.view(1, 1, T)  # [1, 1, T]


# ----------------------------------------------------------------------------
from typing import Dict, Optional


def _compute_control_xy_yaw_diff(
    control_diff_denorm: torch.
    Tensor,  # [B, P, T, 3], (vx[m/s], vy[m/s], yaw_rate[rad/s] 또는 deg/s)
    valid_mask: torch.Tensor,  # [B, P, T], True=유효
    *,
    prefix: str = "constraint_diff",
    omega_in_radian: bool = True,  # True면 rad/s → deg/s 변환
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """제어 차이(vx, vy, yaw_rate)의 규모를 물리 단위로 요약 지표로 반환.

    Args:
        control_diff_denorm: [B, P, T, 3]
            (vx^b, vy^b, yaw_rate)의 '차이' (이미 denormalize된 값).
            vx, vy 단위는 m/s, yaw_rate는 rad/s(기본 가정).
        valid_mask: [B, P, T] True=유효 프레임.
        prefix: 반환 딕셔너리 키 접두사.
        w_t: [1, 1, T] 시간 가중치. None이면 균등 가중.
        omega_in_radian: True면 yaw_rate(rad/s)를 deg/s로 변환해서 리포트.
        eps: 분모 보호용 epsilon.

    Returns:
        Dict[str, torch.Tensor]:
            - f"{prefix}_vx_b_abs_mean_mps"
            - f"{prefix}_vy_b_abs_mean_mps"
            - f"{prefix}_yaw_rate_abs_mean_dps"
            - f"{prefix}_vx_b_rmse_mps"
            - f"{prefix}_vy_b_rmse_mps"
            - f"{prefix}_yaw_rate_rmse_dps"
    """
    if control_diff_denorm.dim() != 4 or control_diff_denorm.size(-1) != 3:
        raise ValueError(
            f"control_diff_denorm must be [B,P,T,3], got {tuple(control_diff_denorm.shape)}"
        )
    if valid_mask.shape != control_diff_denorm.shape[:3]:
        raise ValueError(
            f"valid_mask shape {tuple(valid_mask.shape)} must match [B,P,T] of control_diff_denorm {tuple(control_diff_denorm.shape[:3])}"
        )

    vx = control_diff_denorm[..., 0]  # [B,P,T]  m/s
    vy = control_diff_denorm[..., 1]  # [B,P,T]  m/s
    omega = control_diff_denorm[..., 2]  # [B,P,T]  rad/s (기본 가정)

    if omega_in_radian:
        omega_deg = torch.rad2deg(omega)  # deg/s
    else:
        omega_deg = omega  # 이미 deg/s 라고 가정

    valid_f = valid_mask.to(dtype=vx.dtype)  # 0/1

    # 균등 가중
    weight = valid_f
    denom = weight.sum().clamp_min(eps)  # 스칼라

    # ----- Mean Abs (L1) -----
    vx_abs_mean = (vx.abs() * weight).sum() / denom
    vy_abs_mean = (vy.abs() * weight).sum() / denom
    omg_abs_mean_deg = (omega_deg.abs() * weight).sum() / denom

    return {
        f"{prefix}_vx_b": vx_abs_mean,  # m/s
        f"{prefix}_vy_b": vy_abs_mean,  # m/s
        f"{prefix}_yaw_rate": omg_abs_mean_deg,  # deg/s
    }


def _compute_xy_yaw_losses(
        score_denorm: torch.Tensor,
        near_future_gt: torch.Tensor,
        near_future_valid: torch.Tensor,
        early_stage_num: int = 5,
        prefix: str = "neighbor_prediction_loss") -> Dict[str, torch.Tensor]:
    """
    Compute separate XY and Yaw RMSE losses for ego and neighbors.

    Args:
        score_denorm (Tensor(B, Pnn, T, 4)):
            Model output trajectories, where last dim is [dx, dy, cos(yaw), sin(yaw)].
        near_future_gt (Tensor[B, Pnn, T, 4]):
            Ground truth trajectories in same format as score_denorm.
        near_future_valid (BoolTensor[B, Pnn, T]):
            Mask for valid neighbor entries (excludes ego at index 0).
    Returns:
        Dict with:
        - 'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        - 'neighbor_prediction_loss_yaw' (float): mean abs angular error (deg) for neighbors.
    """
    # score_denorm[..., :2]: Tensor[B, Pnn, T, 2] -> (x, y)
    pred_xy = score_denorm[..., :2]  # [B, Pnn, T, 2]
    gt_xy = near_future_gt[..., :2]
    # Euclidean distance: sqrt((dx)^2 + (dy)^2)
    dist_ = torch.sqrt(((pred_xy - gt_xy).pow(2).sum(-1)) + 1e-6)  # [B, Pnn, T]

    valid_dist = dist_[near_future_valid]  # [num_valid]
    valid_dist_mean = valid_dist.mean() if valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    early_stage_dist_ = dist_[..., :
                              early_stage_num]  # [B, Pnn, early_stage_num]
    near_future_early_valid = near_future_valid[
        ..., :early_stage_num]  # [B, Pnn, early_stage_num]
    early_valid_dist = early_stage_dist_[
        near_future_early_valid]  # [num_early_valid]
    early_valid_dist_mean = early_valid_dist.mean() if early_valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    # Compute yaw angles from cos/sin
    pred_cos = score_denorm[..., 2]  # [B, P, T]
    pred_sin = score_denorm[..., 3]
    gt_cos = near_future_gt[..., 2]
    gt_sin = near_future_gt[..., 3]
    yaw_pred = torch.atan2(pred_sin, pred_cos)  # [B, P, T]
    yaw_gt = torch.atan2(gt_sin, gt_cos)
    # Angular error wrapped to [-pi, pi]
    yaw_err = (yaw_pred - yaw_gt +
               torch.pi) % (2 * torch.pi) - torch.pi  # [B, P, T]
    yaw_err_deg = torch.rad2deg(yaw_err)  # [-180, 180]

    dist_yaw = torch.abs(yaw_err_deg)  # abs error in radians # [B, P, T]
    masked_yaw = dist_yaw[near_future_valid]
    neigh_yaw = masked_yaw.mean() if masked_yaw.numel() > 0 else torch.tensor(
        0.0, device=dist_yaw.device)

    early_stage_dist_yaw = dist_yaw[
        ..., :early_stage_num]  # [B, Pnn, early_stage_num]
    early_valid_yaw = early_stage_dist_yaw[near_future_early_valid]
    early_valid_yaw_mean = early_valid_yaw.mean() if early_valid_yaw.numel(
    ) > 0 else torch.tensor(0.0, device=dist_yaw.device)

    return {
        f'{prefix}_xy': valid_dist_mean,  # scalar
        f'{prefix}_yaw': neigh_yaw,  # scalar # degree
        f'{prefix}_xy_early': early_valid_dist_mean,  # scalar
        f'{prefix}_yaw_early': early_valid_yaw_mean,  # scalar # degree
    }


# [ADD] ----------------------------------------------------------------------
def _masked_weighted_mse_from_diff(
    diff: torch.Tensor,  # (B, P, T, 3)
    valid_mask: torch.Tensor,  # (B, P, T)  (1=valid)
    w_t: torch.Tensor,  # (1, 1, T)
    eps: float = 1e-6,
) -> torch.Tensor:
    """마스크·시간가중 MSE(차원 합) 계산.

    Args:
        diff: (B, P, T, 3) 차이 텐서. 예) u - Filter_soft(u).detach()
        valid_mask: (B, P, T) 유효 마스크 (True/1=유효)
        w_t: (1, 1, T) half-life 기반 시간 가중치
        eps: 분모 보호용 epsilon

    Returns:
        스칼라 손실 (torch.Tensor, shape=[])
    """
    # (B, P, T, C) -> (B, P, T)
    squared = (diff**2).sum(dim=-1)
    valid_f = valid_mask.to(dtype=squared.dtype)

    weighted = squared * w_t  # (B, P, T)
    denom = (valid_f * w_t).sum().clamp_min(eps)  # 스칼라

    loss_val = (weighted * valid_f).sum() / denom
    return loss_val


# ----------------------------------------------------------------------------


def diffusion_loss_func(
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
    futures: Tuple[torch.Tensor, torch.Tensor],
    state_normalizer: StateNormalizer,
    loss: Dict[str, Any],
    model_type: str,
    observation_normalizer,
    eps: float = 1e-3,
):
    """

    near_future_gt_4_dim.shape: [8, Pnn, 80, 4] # [B, Pnn, T, 4]
        ["neighbor_future_gt_3_dim"][:self._predicted_neighbor_num]
    near_future_mask.shape: [8, Pnn, 80] # [B, Pnn, T]
    """
    near_future_gt_4_dim, near_future_mask = futures
    near_future_gt_4_dim = _require_finite("near_future_gt_4_dim",
                                           near_future_gt_4_dim)

    checked_norm_inputs: Dict[str, torch.Tensor] = {}
    for k, v in norm_inputs.items():
        if isinstance(v, torch.Tensor):
            checked_norm_inputs[k] = _require_finite(f"norm_inputs['{k}']", v)
        else:
            checked_norm_inputs[k] = v
    norm_inputs = checked_norm_inputs

    # ego_future: [B. T, 4]
    # near_future_gt_4_dim: [B, Pnn, T]
    near_future_valid = ~near_future_mask

    B, Pnn, T, _ = near_future_gt_4_dim.shape
    # ego_current: [B, 4]
    # neighbors_current: [B, Pnn, 4]
    near_current_xyyaw_norm = norm_inputs["neighbor_agents_past"][:, :Pnn,
                                                                  -1, :4]
    # near_current_mask: [B, Pnn]
    near_current_mask = torch.sum(torch.ne(near_current_xyyaw_norm[..., :4], 0),
                                  dim=-1) == 0  # [B, Pnn]
    # near_future_mask: (B, Pnn, T)
    # near_cur_future_mask: [B, Pnn, 1+T]
    near_cur_future_mask = torch.concat(
        (near_current_mask.unsqueeze(-1), near_future_mask), dim=-1)
    assert near_cur_future_mask.shape == (B, Pnn, 1 + T)
    # near_future_gt_4_dim: [B, Pnn, T, 4]
    # near_current_xyyaw_norm: [B, Pnn, 4]
    # batch_diffusion_time: [B,] diffusion time uniformly sampled in [eps, 1]
    """ batch_diffusion_time
    1 에 가까울수록 더 많은 noise가 추가됨.
    """
    batch_diffusion_time = torch.rand(
        B, device=near_future_gt_4_dim.device) * (1 - eps) + eps  # [B,]
    # --- '노이즈가 적은(t<=0.3)' 구간만 쓰는 마스크 -------------------------
    LOW_NOISE_FRAC = 0.30
    t_threshold = LOW_NOISE_FRAC  # 0.30
    low_t_mask = (batch_diffusion_time <= t_threshold)  # [B]  True=저노이즈
    low_t_mask_bt = low_t_mask.view(B, 1, 1)  # [B,1,1] → [B,P,T] 브로드캐스트
    # random_noise: [B, Pnn + 1, T, 4] noise sampled from standard normal
    random_noise = torch.randn_like(
        near_future_gt_4_dim,
        device=near_future_gt_4_dim.device)  # [B, Pnn, T, 4]

    # near_cur_future_norm_gt: [B, Pnn, 1+T, 4]
    normed_near_future_gt_4_dim = state_normalizer(
        near_future_gt_4_dim)  # (B, Pnn, T, 4)
    normed_near_future_gt_4_dim[near_future_mask] = 0.0
    cond_last_pos_norm = normed_near_future_gt_4_dim[:, :, -1, :]  # [B, Pnn, 4]
    normed_near_future_gt_4_dim = _require_finite(
        "state_normalizer(near_future_gt_4_dim)", normed_near_future_gt_4_dim)
    near_cur_future_norm_gt = torch.cat(
        [near_current_xyyaw_norm[:, :, None, :], normed_near_future_gt_4_dim],
        dim=2)  # [B, Pnn, 1 + T, 4]
    # near_cur_future_mask: [B, Pnn, 1+T]
    near_cur_future_norm_gt[near_cur_future_mask] = 0.0
    near_future_norm_gt = near_cur_future_norm_gt[:, :, 1:, :]  # [B, Pnn, T, 4]
    """
    <forward pass>
    
    
    # mean.shape: torch.Size([B, Pnn, T, 4])
    # std.shape: torch.Size([B, 1, 1, 1])
    """
    mean, std = marginal_prob(near_future_norm_gt, batch_diffusion_time)
    mean = _require_finite("marginal_prob mean", mean)
    std = _require_finite("marginal_prob std", std)
    # std.shape after view: torch.Size([B, 1, 1, 1])
    std = std.view(-1, *([1] * (len(near_future_norm_gt.shape) - 1)))
    #  near_future_norm_xT.shape: [B, Pnn, T, 4]
    near_future_norm_xT = mean + std * random_noise
    # near_cur_future_norm_xT.shape after concat: torch.Size([B, Pnn, 1+T, 4])
    near_cur_future_norm_xT = torch.cat(
        [near_cur_future_norm_gt[:, :, :1, :], near_future_norm_xT], dim=2)
    assert near_cur_future_norm_xT.shape == (B, Pnn, 1 + T, 4)

    merged_inputs = {
        **norm_inputs,
        "near_future_valid": near_future_valid,  # [B, Pnn, T]
        "near_cur_future_norm_xT":
            near_cur_future_norm_xT,  # [B, Pnn, 1 + T, 4]
        "diffusion_time": batch_diffusion_time,  # [B,]
        "cond_last_pos_norm": cond_last_pos_norm,  # [B, Pnn, 4]
    }
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        _, decoder_output = model(merged_inputs)

    # decoder_output["score"]: (B, Pnn, (1 + T) , 4)
    score = decoder_output["score"][:, :, 1:, :]  # (B, Pnn, T, 4)
    score = _require_finite("decoder_output['score']", score)
    assert score.shape == (B, Pnn, T, 4)

    if model_type == "score":
        dpm_loss = torch.sum((score * std + random_noise)**2, dim=-1)
    elif model_type == "x_start":
        # near_future_gt_4_dim: [B, Pnn, T, 4]
        # dpm_loss: (B, Pnn, T)
        dpm_loss = torch.sum((score - near_future_norm_gt)**2, dim=-1)
    # near_future_valid: [B, Pnn, T]
    valid = near_future_valid.float()

    # [add] ---- 시간 가중치(half-life) 적용 ------------------------------------
    # half-life과 dt(초)는 필요 시 조정 가능
    time_step_s: float = 0.1  # 0.1초 간격(데이터/시뮬 규격에 맞게 조정)
    half_life_s: float = 2.  # 2초에서 가중치 1/2
    w_t: torch.Tensor = _build_half_life_weights(
        T,  # 80
        dt_s=time_step_s,  # 0.1
        half_life_s=half_life_s,  # 2.0
        device=dpm_loss.device,
        dtype=dpm_loss.dtype,
    )  # [1, 1, T]
    # 유효 마스크와 함께 곱해서 "가중 평균"으로 정규화
    weighted_dpm = dpm_loss * w_t  # (B, Pnn, T)
    denom = (valid * w_t).sum().clamp(min=1e-6)  # 스칼라(가중치 포함 유효개수)
    valid_dpm_loss = weighted_dpm * valid  # (B, Pnn, T)
    loss_val = valid_dpm_loss.sum() / denom  # 스칼라(gradient O)
    loss["neighbor_prediction_loss"] = loss_val

    if model_type == "x_start":
        valid_low = near_future_valid & low_t_mask_bt  # (B, Pnn, T) bool
        valid_low_f = valid_low.float()
        ###### L_integration loss 추가 ######
        if "integrated_trajectory" in decoder_output:

            integrated_trajectory = decoder_output[
                "integrated_trajectory"][:, :, 1:, :]  # (B, Pnn, T, 4)
            # near_future_gt_4_dim: [B, Pnn, T, 4]
            # integration_loss: (B, Pnn, T)
            integration_loss = torch.sum(
                (integrated_trajectory - near_future_norm_gt)**2, dim=-1)
            weighted_integration = integration_loss * w_t  # (B, Pnn, T)
            denom_low = (valid_low_f * w_t).sum().clamp_min(1e-6)
            integration_loss_val = (weighted_integration *
                                    valid_low_f).sum() / denom_low

        else:
            integrated_trajectory = None
            # 안전 fallback: 해당 항 미제공 시 0 손실
            integration_loss_val = torch.zeros((),
                                               device=loss_val.device,
                                               dtype=loss_val.dtype)
        loss["integration_loss"] = integration_loss_val
        ###### L_constraint loss 추가 ######
        # ------ L_constraint (정식 구현) --------------------------------------
        # decoder_output["control_constraint_diff"]: (B, P, T, 3)
        #   = u - Filter_soft(u).detach()  (모델 내부에서 detach 적용되어야 함)
        # 학습 신호는 u(=보정기 경로)로만 흘러가도록 설계됨.
        if "control_constraint_diff" in decoder_output:
            control_constraint_diff = _require_finite(
                "decoder_output['control_constraint_diff']",
                decoder_output["control_constraint_diff"])  # (B, P, T, 3)
            constraint_loss_val = _masked_weighted_mse_from_diff(
                control_constraint_diff,  # (B,P,T,3)
                valid_low,  # <-- 기존 near_future_valid 대신
                w_t
            )
        else:
            # 안전 fallback: 해당 항 미제공 시 0 손실
            control_constraint_diff = None
            constraint_loss_val = torch.zeros(
                (),
                device=integration_loss_val.device,
                dtype=integration_loss_val.dtype)
        loss["constraint_loss"] = constraint_loss_val

    # denom = valid.sum().clamp(min=1) # denom: scalar
    # valid_dpm_loss = dpm_loss * valid # (B, Pnn, T)
    # loss_val = valid_dpm_loss.sum() / denom  # 항상 requires_grad=True

    # compute and merge xy/yaw losses via helper
    if model_type == "x_start":
        score_denorm = state_normalizer.inverse(score)  # [B,P,T,4]
        near_future_gt = state_normalizer.inverse(near_future_norm_gt)
        with torch.no_grad():
            xy_yaw_losses = _compute_xy_yaw_losses(score_denorm, near_future_gt,
                                                   near_future_valid)
            loss.update(xy_yaw_losses)

            if integrated_trajectory is not None:
                integrated_trajectory_denorm = state_normalizer.inverse(
                    integrated_trajectory)  # [B,P,T,4]
                integ_xy_yaw_losses = _compute_xy_yaw_losses(
                    integrated_trajectory_denorm,
                    near_future_gt,
                    near_future_valid,
                    prefix="integration_loss")
                loss.update(integ_xy_yaw_losses)
            if control_constraint_diff is not None:
                temp_dict = {
                    "cur_future_seg_body_control": control_constraint_diff
                }
                temp_dict = observation_normalizer.inverse(
                    temp_dict)  # [B,P,T,3]
                constraint_diff_denorm = temp_dict[
                    "cur_future_seg_body_control"]  # [B,P,T,3]
                constraint_xy_yaw_losses = _compute_control_xy_yaw_diff(
                    constraint_diff_denorm,
                    near_future_valid,
                    prefix="constraint_diff")
                loss.update(constraint_xy_yaw_losses)

    assert torch.isfinite(dpm_loss).all().item(
    ), f"loss cannot be nan, random_noise={random_noise}"
    """
    loss
        "neighbor_prediction_loss" (float): mean RMSE over neighbor coords.
        'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        'neighbor_prediction_loss_yaw' (float): mean abs angular error (rad) for neighbors.
    """
    return loss, decoder_output
