from typing import Any, Callable, Dict, List, Tuple, Optional
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
    future_len: int,  # 80
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
        future_len (int): 미래 스텝 수(프레임 수).
        dt_s (float): 프레임 간 시간 간격(초). 기본 0.1초.
        half_life_s (float): half-life(초).
        device (torch.device): 출력 텐서가 위치할 디바이스.
        dtype (torch.dtype): 출력 텐서 dtype.

    Returns:
        torch.Tensor: (1, 1, future_len) 모양의 가중치 텐서.  # shape: [1, 1, future_len]

    Note:
        - 시간축은 [dt_s, 2*dt_s, ..., future_len*dt_s].
        - 브로드캐스트를 위해 (1,1,future_len)로 반환합니다.
    """
    # future_len: (future_len,) = [dt_s, 2*dt_s, ..., future_len*dt_s]
    future_len = torch.arange(1, future_len + 1, device=device, dtype=dtype) * float(dt_s)  # [future_len]
    # w: (future_len,) = 0.5 ** (future_len / half_life_s)
    w = torch.pow(0.5, future_len / float(half_life_s))  # [future_len]
    return w.view(1, 1, future_len)  # [1, 1, future_len]


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
def _sanitize_norm_inputs(
    norm_inputs: Dict[str, torch.Tensor],) -> Dict[str, torch.Tensor]:
    """정규화된 입력 dict 안 텐서들이 NaN/Inf 인지 확인하고 새 dict 로 돌려준다.

    Args:
        norm_inputs: 모델 입력용 정규화 관측 dict.

    Returns:
        각 텐서가 유한값인지 검사한 뒤 담은 새 dict.
    """
    checked_norm_inputs: Dict[str, torch.Tensor] = {}
    for k, v in norm_inputs.items():
        if isinstance(v, torch.Tensor):
            checked_norm_inputs[k] = _require_finite(f"norm_inputs['{k}']", v)
        else:
            checked_norm_inputs[k] = v
    return checked_norm_inputs
    # 각 v: 보통 (B, ·) 모양 텐서들


def _build_future_masks_and_current_state(
    near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    near_future_mask: torch.Tensor,  # (B, Pnn, future_len)
    norm_inputs: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 마스크와 현재 상태, 현재+미래 마스크를 만든다.

    Args:
        near_future_gt_4_dim: (B, Pnn, future_len, 4) 미래 궤적 (정규화 전).
        near_future_mask: (B, Pnn, future_len) True면 값이 없는 프레임.
        norm_inputs: 정규화된 관측 dict. neighbor_agents_past 포함.

    Returns:
        near_future_valid: (B, Pnn, future_len) True면 유효 프레임.
        near_cur_future_mask: (B, Pnn, 1+future_len) 현재+미래 마스크.
        near_current_xyyaw_norm: (B, Pnn, 4) 현재 시점 상태.

    """
    # near_future_gt_4_dim: (B, Pnn, future_len, 4)
    B, Pnn, future_len, _ = near_future_gt_4_dim.shape

    # near_future_valid: (B, Pnn, future_len)
    near_future_valid: torch.Tensor = ~near_future_mask

    # neighbor_agents_past: (B, agent_num, time_len, 11)
    neighbor_agents_past: torch.Tensor = norm_inputs["neighbor_agents_past"]
    # near_current_xyyaw_norm: (B, Pnn, 4)  마지막 과거 프레임의 (x, y, cos, sin)
    near_current_xyyaw_norm: torch.Tensor = neighbor_agents_past[:, :Pnn,
                                                                 -1, :4]

    # near_current_mask: (B, Pnn)
    near_current_mask: torch.Tensor = torch.sum(
        torch.ne(near_current_xyyaw_norm[..., :4], 0),
        dim=-1,
    ) == 0

    # near_cur_future_mask: (B, Pnn, 1+future_len)
    near_cur_future_mask: torch.Tensor = torch.concat(
        (near_current_mask.unsqueeze(-1), near_future_mask),
        dim=-1,
    )
    assert near_cur_future_mask.shape == (B, Pnn, 1 + future_len)

    return near_future_valid, near_cur_future_mask, near_current_xyyaw_norm


def _sample_diffusion_time_and_noise(
    near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, T, 4)
    eps: float,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적 크기에 맞춰 diffusion time 과 노이즈를 샘플링한다.

    Args:
        near_future_gt_4_dim: (B, Pnn, T, 4) 미래 궤적.
        eps: 시간 샘플링 하한.
        args: args.feasible_learn_noise_thresh, args.use_direct_loss 사용.

    Returns:
        batch_diffusion_time: (B,) 각 배치별 diffusion 시간.
        low_t_mask: (B,) 저노이즈 배치 마스크.
        low_t_mask_bt: (B, 1, 1) 저노이즈 마스크(브로드캐스트용).
        random_noise: (B, Pnn, T, 4) 노이즈 샘플.
    """
    # near_future_gt_4_dim: (B, Pnn, T, 4)
    B: int = near_future_gt_4_dim.shape[0]

    # batch_diffusion_time: (B,)
    batch_diffusion_time: torch.Tensor = torch.rand(
        B,
        device=near_future_gt_4_dim.device,
    ) * (1 - eps) + eps

    t_threshold: float = float(args.feasible_learn_noise_thresh)
    if not getattr(args, "use_direct_loss", False):
        t_threshold = 1.0
    # low_t_mask: (B,)
    low_t_mask: torch.Tensor = batch_diffusion_time <= t_threshold
    # low_t_mask_bt: (B,1,1)
    low_t_mask_bt: torch.Tensor = low_t_mask.view(B, 1, 1)

    # random_noise: (B, Pnn, T, 4)
    random_noise: torch.Tensor = torch.randn_like(
        near_future_gt_4_dim,
        device=near_future_gt_4_dim.device,
    )
    return batch_diffusion_time, low_t_mask, low_t_mask_bt, random_noise


def _normalize_futures_and_build_xT(
    near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    near_future_mask: torch.Tensor,  # (B, Pnn, future_len)
    near_current_xyyaw_norm: torch.Tensor,  # (B, Pnn, 4)
    near_cur_future_mask: torch.Tensor,  # (B, Pnn, 1+future_len)
    batch_diffusion_time: torch.Tensor,  # (B,)
    random_noise: torch.Tensor,  # (B, Pnn, future_len, 4)
    state_normalizer: StateNormalizer,
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적을 정규화하고, x_T 샘플과 std 를 만든다.

    Args:
        near_future_gt_4_dim: (B, Pnn, future_len, 4) 미래 궤적(denorm).
        near_future_mask: (B, Pnn, future_len) 미래 마스크.
        near_current_xyyaw_norm: (B, Pnn, 4) 현재 상태(정규화).
        near_cur_future_mask: (B, Pnn, 1+future_len) 현재+미래 마스크.
        batch_diffusion_time: (B,) diffusion 시간.
        random_noise: (B, Pnn, future_len, 4) 노이즈.
        state_normalizer: 상태 정규화/역정규화 도우미.
        marginal_prob: SDE 의 marginal_prob 함수.

    Returns:
        near_future_norm_gt: (B, Pnn, future_len, 4) 정규화된 미래 궤적.
        near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4) 현재+미래 x_T 샘플.
        cond_last_pos_norm: (B, Pnn, 4) 미래 마지막 프레임(정규화).
        std: (B, 1, 1, 1) 노이즈 표준편차.
    """
    # near_future_gt_4_dim: (B, Pnn, future_len, 4)
    B, Pnn, future_len, _ = near_future_gt_4_dim.shape

    # normed_near_future_gt_4_dim: (B, Pnn, future_len, 4)
    normed_near_future_gt_4_dim: torch.Tensor = state_normalizer(
        near_future_gt_4_dim)
    normed_near_future_gt_4_dim[near_future_mask] = 0.0
    # cond_last_pos_norm: (B, Pnn, 4)
    cond_last_pos_norm: torch.Tensor = normed_near_future_gt_4_dim[:, :, -1, :]

    normed_near_future_gt_4_dim = _require_finite(
        "state_normalizer(near_future_gt_4_dim)", normed_near_future_gt_4_dim)

    # near_cur_future_norm_gt: (B, Pnn, 1+future_len, 4)
    near_cur_future_norm_gt: torch.Tensor = torch.cat(
        [near_current_xyyaw_norm[:, :, None, :], normed_near_future_gt_4_dim],
        dim=2,
    )
    near_cur_future_norm_gt[near_cur_future_mask] = 0.0
    # near_future_norm_gt: (B, Pnn, future_len, 4)
    near_future_norm_gt: torch.Tensor = near_cur_future_norm_gt[:, :, 1:, :]

    # mean, std_raw: 각각 (B, Pnn, future_len, 4) 또는 (B,) 등 marginal_prob 설계에 맞게 반환
    mean, std_raw = marginal_prob(near_future_norm_gt, batch_diffusion_time)
    mean = _require_finite("marginal_prob mean", mean)
    std_raw = _require_finite("marginal_prob std", std_raw)

    # std: (B, 1, 1, 1) 로 reshape
    std: torch.Tensor = std_raw.view(
        -1,
        *([1] * (len(near_future_norm_gt.shape) - 1)),
    )

    # near_future_noise_xT: (B, Pnn, future_len, 4)
    near_future_noise_xT: torch.Tensor = mean + std * random_noise

    # near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4)
    near_cur_future_norm_xT: torch.Tensor = torch.cat(
        [near_cur_future_norm_gt[:, :, :1, :], near_future_noise_xT],
        dim=2,
    )
    assert near_cur_future_norm_xT.shape == (B, Pnn, 1 + future_len, 4)

    return near_future_norm_gt, near_cur_future_norm_xT, cond_last_pos_norm, std


def _forward_model_with_autocast(
        model: nn.Module,
        norm_inputs: Dict[str, torch.Tensor],
        near_future_valid: torch.Tensor,  # (B, Pnn, future_len)
        near_cur_future_norm_xT: torch.Tensor,  # (B, Pnn, 1+future_len, 4)
        batch_diffusion_time: torch.Tensor,  # (B,)
        cond_last_pos_norm: torch.Tensor,  # (B, Pnn, 4)
) -> Dict[str, torch.Tensor]:
    """모델 입력 dict 를 만들고 AMP 로 forward 를 수행한다.

    Args:
        model: 학습 중인 모델.
        norm_inputs: 정규화된 관측 dict.
        near_future_valid: (B, Pnn, future_len) 미래 유효 마스크.
        near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4) 현재+미래 x_T.
        batch_diffusion_time: (B,) diffusion 시간.
        cond_last_pos_norm: (B, Pnn, 4) cond 용 마지막 위치.

    Returns:
        decoder_output: model 의 두 번째 반환값 dict.
    """
    merged_inputs: Dict[str, torch.Tensor] = {
        **norm_inputs,
        "near_future_valid": near_future_valid,  # (B, Pnn, future_len)
        "near_cur_future_norm_xT": near_cur_future_norm_xT,  # (B, Pnn, 1+future_len, 4)
        "diffusion_time": batch_diffusion_time,  # (B,)
        "cond_last_pos_norm": cond_last_pos_norm,  # (B, Pnn, 4)
    }
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        _, decoder_output = model(merged_inputs)
    return decoder_output  # decoder_output["score"], ["integrated_trajectory"], ...


def _extract_score_from_decoder(
    decoder_output: Dict[str, torch.Tensor],
    B: int,
    Pnn: int,
    future_len: int,
) -> torch.Tensor:
    """decoder_output 에서 미래 score 만 꺼내고 모양을 확인한다.

    Args:
        decoder_output: model(...) 의 두 번째 반환 dict.
        B: 배치 크기.
        Pnn: 이웃 개수.
        future_len: 미래 길이.

    Returns:
        score: (B, Pnn, future_len, 4) 예측된 미래 궤적.
    """
    # decoder_output["score"]: (B, Pnn, 1+future_len, 4)
    score: torch.Tensor = decoder_output["score"][:, :, 1:, :]
    score = _require_finite("decoder_output['score']", score)
    assert score.shape == (B, Pnn, future_len, 4)
    return score


def _compute_dpm_loss(
        args: Any,
        model_type: str,
        score: torch.Tensor,  # (B, Pnn, future_len, 4)
        std: torch.Tensor,  # (B, 1, 1, 1)
        random_noise: torch.Tensor,  # (B, Pnn, future_len, 4)
        near_future_norm_gt: torch.Tensor,  # (B, Pnn, future_len, 4)
) -> torch.Tensor:
    """기존 diffusion 손실(score/x_start)을 (B,P,future_len) 형태로 계산한다.

    Args:
        args: args.use_huber_loss 사용.
        model_type: "score" 또는 "x_start".
        score: (B, Pnn, future_len, 4) 모델 출력.
        std: (B, 1, 1, 1) 노이즈 표준편차.
        random_noise: (B, Pnn, future_len, 4) 노이즈.
        near_future_norm_gt: (B, Pnn, future_len, 4) GT (정규화).

    Returns:
        dpm_loss: (B, Pnn, future_len) 위치별 손실 값.
    """
    HUBER_DELTA: float = 1.0

    if getattr(args, "use_huber_loss", True):
        # err: (B, Pnn, future_len, 4)
        if model_type == "score":
            err: torch.Tensor = score * std + random_noise
        elif model_type == "x_start":
            err = score - near_future_norm_gt
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        abs_err: torch.Tensor = err.abs()  # (B, Pnn, future_len, 4)
        quad: torch.Tensor = 0.5 * err.pow(2)  # (B, Pnn, future_len, 4)
        lin: torch.Tensor = HUBER_DELTA * (abs_err - 0.5 * HUBER_DELTA
                                          )  # (B, Pnn, future_len, 4)
        huber: torch.Tensor = torch.where(abs_err <= HUBER_DELTA, quad,
                                          lin)  # (B, Pnn, future_len, 4)
        dpm_loss: torch.Tensor = huber.sum(dim=-1)  # (B, Pnn, future_len)
    else:
        if model_type == "score":
            dpm_loss = torch.sum((score * std + random_noise)**2,
                                 dim=-1)  # (B, Pnn, future_len)
        elif model_type == "x_start":
            dpm_loss = torch.sum((score - near_future_norm_gt)**2,
                                 dim=-1)  # (B, Pnn, future_len)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return dpm_loss


def _aggregate_weighted_loss(
    per_step_loss: torch.Tensor,  # (B, Pnn, T)
    valid_mask: torch.Tensor,  # (B, Pnn, T) bool 또는 float
    w_t: torch.Tensor,  # (1, 1, T)
    eps: float = 1e-6,
) -> torch.Tensor:
    """시간 가중치와 유효 마스크를 써서 (B,P,T) 손실을 스칼라로 합친다.

    Args:
        per_step_loss: (B, Pnn, T) 위치별 손실.
        valid_mask: (B, Pnn, T) 유효 프레임 마스크.
        w_t: (1, 1, T) 시간 가중치.
        eps: 분모 보호용 작은 값.

    Returns:
        scalar_loss: 스칼라 손실 텐서 (shape=[]).
    """
    # per_step_loss, valid_mask, w_t : 모두 (B,P,T) 브로드캐스트 호환
    valid_f: torch.Tensor = valid_mask.to(dtype=per_step_loss.dtype)
    weighted: torch.Tensor = per_step_loss * w_t  # (B, Pnn, T)
    denom: torch.Tensor = (valid_f * w_t).sum().clamp_min(eps)  # 스칼라
    scalar_loss: torch.Tensor = (weighted * valid_f).sum() / denom
    return scalar_loss


def _compute_integration_and_constraint_losses(
    args: Any,
    decoder_output: Dict[str, torch.Tensor],
    near_future_norm_gt: torch.Tensor,  # (B, Pnn, T, 4)
    near_future_valid: torch.Tensor,  # (B, Pnn, T)
    low_t_mask_bt: torch.Tensor,  # (B, 1, 1)
    w_t: torch.Tensor,  # (1, 1, T)
    base_loss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor]]:
    """통합 궤적/제어 편차 기반 보조 손실을 계산한다.

    Args:
        args: args.use_feasible_dl 사용.
        decoder_output: 모델 decoder 출력 dict.
        near_future_norm_gt: (B, Pnn, T, 4) 미래 GT (정규화).
        near_future_valid: (B, Pnn, T) 미래 유효 마스크.
        low_t_mask_bt: (B, 1, 1) 저노이즈 배치 마스크.
        w_t: (1, 1, T) 시간 가중치.
        base_loss: neighbor_prediction_loss 와 같은 dtype/device 기준.

    Returns:
        integration_loss_val: 통합 궤적 손실 스칼라.
        constraint_loss_val: 제어 제약 손실 스칼라.
        integrated_trajectory: (B, Pnn, T, 4) 또는 None.
        control_constraint_diff: (B, Pnn, T, 3) 또는 None.
    """
    # valid_low: (B, Pnn, T)
    valid_low: torch.Tensor = near_future_valid & low_t_mask_bt
    valid_low_f: torch.Tensor = valid_low.float()

    # --- L_integration ---
    if "integrated_trajectory" in decoder_output and getattr(
            args, "use_feasible_dl", False):
        # integrated_trajectory_full: (B, Pnn, 1+T, 4)
        integrated_trajectory_full: torch.Tensor = _require_finite(
            "decoder_output['integrated_trajectory']",
            decoder_output["integrated_trajectory"],
        )
        # integrated_trajectory: (B, Pnn, T, 4)
        integrated_trajectory: torch.Tensor = integrated_trajectory_full[:, :,
                                                                         1:, :]

        # integration_loss: (B, Pnn, T)
        integration_loss: torch.Tensor = torch.sum(
            (integrated_trajectory - near_future_norm_gt)**2,
            dim=-1,
        )
        # weighted_integration: (B, Pnn, T)
        weighted_integration: torch.Tensor = integration_loss * w_t
        denom_low: torch.Tensor = (valid_low_f * w_t).sum().clamp_min(1e-6)
        integration_loss_val: torch.Tensor = \
            (weighted_integration * valid_low_f).sum() / denom_low
    else:
        integrated_trajectory = None
        integration_loss_val = torch.zeros(
            (),
            device=base_loss.device,
            dtype=base_loss.dtype,
        )

    # --- L_constraint ---
    if "control_constraint_diff" in decoder_output and getattr(
            args, "use_feasible_dl", False):
        control_constraint_diff_full: torch.Tensor = _require_finite(
            "decoder_output['control_constraint_diff']",
            decoder_output["control_constraint_diff"],
        )  # (B, P, T, 3)

        constraint_loss_val: torch.Tensor = _masked_weighted_mse_from_diff(
            control_constraint_diff_full,  # (B, Pnn, T, 3)
            valid_low,  # (B, Pnn, T)
            w_t,  # (1, 1, T)
        )
        control_constraint_diff: Optional[
            torch.Tensor] = control_constraint_diff_full
    else:
        control_constraint_diff = None
        constraint_loss_val = torch.zeros(
            (),
            device=integration_loss_val.device,
            dtype=integration_loss_val.dtype,
        )

    return integration_loss_val, constraint_loss_val, integrated_trajectory, control_constraint_diff


def _add_xy_yaw_metric_losses(
    loss_dict: Dict[str, Any],
    state_normalizer: StateNormalizer,
    observation_normalizer: Any,
    score: torch.Tensor,  # (B, Pnn, T, 4)
    near_future_norm_gt: torch.Tensor,  # (B, Pnn, T, 4)
    near_future_valid: torch.Tensor,  # (B, Pnn, T)
    integrated_trajectory: Optional[torch.Tensor],
    control_constraint_diff: Optional[torch.Tensor],
) -> None:
    """xy / yaw 관련 부가 지표들을 loss_dict dict 에 추가한다.

    Args:
        loss_dict: 손실 및 통계 값을 담는 dict (in-place 업데이트).
        state_normalizer: 상태 역정규화용 도우미.
        observation_normalizer: 제어 역정규화용 도우미.
        score: (B, Pnn, T, 4) 모델 예측(정규화).
        near_future_norm_gt: (B, Pnn, T, 4) 미래 GT(정규화).
        near_future_valid: (B, Pnn, T) 미래 유효 마스크.
        integrated_trajectory: (B, Pnn, T, 4) 또는 None.
        control_constraint_diff: (B, Pnn, T, 3) 또는 None.
    """
    # score_denorm: (B, Pnn, T, 4)
    score_denorm: torch.Tensor = state_normalizer.inverse(score)
    # near_future_gt: (B, Pnn, T, 4)
    near_future_gt: torch.Tensor = state_normalizer.inverse(near_future_norm_gt)

    with torch.no_grad():
        # 기본 score 에 대한 xy/yaw 오차
        xy_yaw_losses = _compute_xy_yaw_losses(
            score_denorm,
            near_future_gt,
            near_future_valid,
        )
        loss_dict.update(xy_yaw_losses)

        # 통합 궤적에 대한 xy/yaw 오차
        if integrated_trajectory is not None:
            # integrated_trajectory_denorm: (B, Pnn, T, 4)
            integrated_trajectory_denorm: torch.Tensor = \
                state_normalizer.inverse(integrated_trajectory)
            integ_xy_yaw_losses = _compute_xy_yaw_losses(
                integrated_trajectory_denorm,
                near_future_gt,
                near_future_valid,
                prefix="integration_loss",
            )
            loss_dict.update(integ_xy_yaw_losses)

        # 제어 편차에 대한 통계 값
        if control_constraint_diff is not None:
            temp_dict = {"seg_body_control": control_constraint_diff}
            temp_dict = observation_normalizer.inverse(temp_dict)
            # constraint_diff_denorm: (B, Pnn, T, 3)
            constraint_diff_denorm: torch.Tensor = temp_dict["seg_body_control"]
            constraint_xy_yaw_losses = _compute_control_xy_yaw_diff(
                constraint_diff_denorm,
                near_future_valid,
                prefix="constraint_diff",
            )
            loss_dict.update(constraint_xy_yaw_losses)


def diffusion_loss_func(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
    near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    near_future_mask: torch.Tensor,  # (B, Pnn, future_len)
    state_normalizer: StateNormalizer,
    loss_dict: Dict[str, Any],
    model_type: str,
    observation_normalizer: Any,
    eps: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """diffusion 학습에 쓰이는 전체 손실을 계산한다.

    처리 흐름:
      1) 미래 궤적/마스크, 정규화 입력을 안전하게 준비.
      2) 현재 상태와 미래를 이어 붙인 시퀀스를 만들고, diffusion time/노이즈 샘플링.
      3) SDE marginal_prob 를 통해 x_T 샘플 생성 후, 모델 forward.
      4) 기본 diffusion 손실(dpm_loss) + half-life 시간 가중 평균.
      5) x_start 모드일 때 통합 궤적/제어 제약 손실, xy/yaw 통계까지 loss dict 에 추가.

    Args:
        args: 학습 설정/옵션이 들어 있는 객체.
        model: 학습 중인 모델(nn.Module 또는 DDP 래퍼).
        norm_inputs: 정규화된 관측 dict.
        marginal_prob: SDE marginal_prob 함수.
        near_future_gt_4_dim: (B, Pnn, future_len, 4) 미래 궤적 (denorm).
        near_future_mask: (B, Pnn, future_len) True면 값이 없는 프레임.
        state_normalizer: 상태 정규화/역정규화 도우미.
        loss_dict: 손실/통계를 쌓아갈 dict (in-place 업데이트).
        model_type: "score" 또는 "x_start".
        observation_normalizer: 제어 역정규화 도우미.
        eps: diffusion time 샘플링 하한.

    Returns:
        loss_dict: 다양한 손실 항목이 담긴 dict.
        decoder_output: 모델 decoder 의 출력 dict.
    """
    # near_future_gt_4_dim: (B, Pnn, future_len, 4)
    near_future_gt_4_dim = _require_finite("near_future_gt_4_dim",
                                           near_future_gt_4_dim)

    # norm_inputs 의 각 텐서 NaN/Inf 체크
    norm_inputs = _sanitize_norm_inputs(norm_inputs)

    # 기본 크기 정보
    B, Pnn, future_len, _ = near_future_gt_4_dim.shape

    # 미래/현재 마스크 및 현재 상태 준비
    # near_future_valid: (B, Pnn, future_len)
    # near_cur_future_mask: (B, Pnn, 1+future_len)
    # near_current_xyyaw_norm: (B, Pnn, 4)
    near_future_valid, near_cur_future_mask, near_current_xyyaw_norm = \
        _build_future_masks_and_current_state(
            near_future_gt_4_dim,
            near_future_mask,
            norm_inputs,
        )

    # diffusion time / low noise mask / random noise 샘플링
    # batch_diffusion_time: (B,)
    # low_t_mask: (B,)
    # low_t_mask_bt: (B,1,1)
    # random_noise: (B, Pnn, future_len, 4)
    (batch_diffusion_time, low_t_mask, low_t_mask_bt,
     random_noise) = _sample_diffusion_time_and_noise(
         near_future_gt_4_dim,
         eps,
         args,
     )

    # 미래 궤적 정규화 + x_T 샘플 생성
    # near_future_norm_gt: (B, Pnn, future_len, 4)
    # near_cur_future_norm_xT: (B, Pnn, 1+future_len, 4)
    # cond_last_pos_norm: (B, Pnn, 4)
    # std: (B, 1, 1, 1)
    (near_future_norm_gt, near_cur_future_norm_xT, cond_last_pos_norm,
     std) = _normalize_futures_and_build_xT(
         near_future_gt_4_dim,
         near_future_mask,
         near_current_xyyaw_norm,
         near_cur_future_mask,
         batch_diffusion_time,
         random_noise,
         state_normalizer,
         marginal_prob,
     )

    # 모델 forward + decoder_output 생성
    decoder_output: Dict[str, torch.Tensor] = _forward_model_with_autocast(
        model=model, #
        norm_inputs=norm_inputs, #
        near_future_valid=near_future_valid, # (B, Pnn, future_len)
        near_cur_future_norm_xT=near_cur_future_norm_xT, # (B, Pnn, 1+future_len, 4)
        batch_diffusion_time=batch_diffusion_time, # (B,)
        cond_last_pos_norm=cond_last_pos_norm, # (B, Pnn, 4)
    )

    # score: (B, Pnn, future_len, 4)
    score: torch.Tensor = _extract_score_from_decoder(
        decoder_output=decoder_output,
        B=B,
        Pnn=Pnn,
        future_len=future_len,
    )

    # dpm_loss: (B, Pnn, future_len)
    dpm_loss: torch.Tensor = _compute_dpm_loss(
        args=args,
        model_type=model_type,
        score=score, # (B, Pnn, future_len, 4)
        std=std, # (B, 1, 1, 1)
        random_noise=random_noise, # (B, Pnn, future_len, 4)
        near_future_norm_gt=near_future_norm_gt, # (B, Pnn, future_len, 4)
    )

    # 시간 가중치(w_t) 생성
    time_step_s: float = 0.1
    half_life_s: float = 2.0
    # w_t: (1, 1, future_len)
    w_t: torch.Tensor = _build_half_life_weights(
        future_len,
        dt_s=time_step_s,
        half_life_s=half_life_s,
        device=dpm_loss.device,
        dtype=dpm_loss.dtype,
    )

    # neighbor_prediction_loss (스칼라)
    loss_val: torch.Tensor = _aggregate_weighted_loss(
        per_step_loss=dpm_loss, # (B, Pnn, future_len)
        valid_mask=near_future_valid, # (B, Pnn, future_len)
        w_t=w_t, # (1, 1, future_len)
        eps=1e-6,
    )
    loss_dict["neighbor_prediction_loss"] = loss_val

    # x_start 모드에서만 Feasible 관련 손실/지표 계산
    if model_type == "x_start":
        # 통합 궤적/제어 제약 손실
        """
        integration_loss_val: 스칼라
        constraint_loss_val: 스칼라
        
        integrated_trajectory: (B, Pnn, T, 4) 또는 None
        control_constraint_diff: (B, Pnn, T, 3) 또는 None
        
        integration_loss_val: 
        """
        (integration_loss_val, constraint_loss_val, integrated_trajectory,
         control_constraint_diff) = _compute_integration_and_constraint_losses(
             args=args,
             decoder_output=decoder_output,
             near_future_norm_gt=near_future_norm_gt,
             near_future_valid=near_future_valid,
             low_t_mask_bt=low_t_mask_bt,
             w_t=w_t,
             base_loss=loss_val,
         )

        loss_dict["integration_loss"] = integration_loss_val
        loss_dict["constraint_loss"] = constraint_loss_val

        # xy/yaw 관련 추가 metric 들
        _add_xy_yaw_metric_losses(
            loss_dict=loss_dict,
            state_normalizer=state_normalizer,
            observation_normalizer=observation_normalizer,
            score=score,
            near_future_norm_gt=near_future_norm_gt,
            near_future_valid=near_future_valid,
            integrated_trajectory=integrated_trajectory,
            control_constraint_diff=control_constraint_diff,
        )

    # dpm_loss 전체가 유한값인지 마지막으로 검사
    assert torch.isfinite(dpm_loss).all().item(), \
        f"loss cannot be nan, random_noise={random_noise}"

    return loss_dict, decoder_output
