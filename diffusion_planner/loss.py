from typing import Any, Callable, Dict, List, Tuple, Optional
import logging

import torch
import torch.nn as nn
from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils.target_feature import build_target_future_tensors_and_masks
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
AMP_DTYPE = torch.bfloat16  # A100 권장 dtype


def _require_finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """tensor에 NaN/Inf가 있는지 확인합니다.

    주의
    ----
    bool/int 텐서는 NaN/Inf라는 개념이 없어서,
    이런 텐서에는 검사를 하지 않고 그대로 통과시킵니다.

    Args:
        name (str): 로그에 찍을 텐서 이름.
        tensor (torch.Tensor): 검사할 텐서.

    Returns:
        torch.Tensor:
            - float/complex 텐서면 NaN/Inf가 없는지 확인 후 그대로 반환합니다.
            - bool/int 텐서면 검사 없이 그대로 반환합니다.

    Raises:
        ValueError:
            float/complex 텐서에서 NaN 또는 Inf가 발견되면 발생합니다.
    """
    if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
        return tensor

    if not torch.isfinite(tensor).all():
        msg = f"{name} contains NaN or Inf values"
        logging.error(msg)
        raise ValueError(msg)
    return tensor


# [add] ----------------------------------------------------------------------
def _build_half_life_weights(
    future_len: int,
    *,
    dt_s: float = 0.1,
    half_life_s: float = 2.0,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """시간이 멀어질수록 가중치를 줄이는 (1,1,T) 텐서를 만든다.

    Args:
        future_len (int): 미래 프레임 수 T.
        dt_s (float): 프레임 간 시간 간격(초).
        half_life_s (float): 이 시간이 지나면 가중치가 절반이 되도록 만드는 기준 시간(초).
        device (torch.device): 출력 텐서 디바이스.
        dtype (torch.dtype): 출력 dtype. 기본은 float32.

    Returns:
        torch.Tensor:
            시간 가중치 텐서. shape: (1, 1, future_len)
    """
    # t: (T,) = [dt, 2*dt, ..., T*dt]
    t = torch.arange(1, future_len + 1, device=device,
                     dtype=torch.float32) * float(dt_s)
    # w: (T,) = 0.5 ** (t / half_life)
    w = torch.pow(torch.tensor(0.5, device=device, dtype=torch.float32),
                  t / float(half_life_s))
    return w.to(dtype=dtype).view(1, 1, -1)


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
        target_future_gt: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
        target_future_valid: torch.Tensor,
        early_stage_num: int = 5,
        prefix: str = "neighbor_prediction_loss") -> Dict[str, torch.Tensor]:
    """
    Compute separate XY and Yaw RMSE losses for ego and neighbors.

    Args:
        score_denorm (Tensor(B, Pnn, T, 4)):
            Model output trajectories, where last dim is [dx, dy, cos(yaw), sin(yaw)].
        target_future_gt # (B, (1+)Pnn, future_len, 4)
            Ground truth trajectories in same format as score_denorm.
        target_future_valid (BoolTensor[B, Pnn, T]):
            Mask for valid neighbor entries (excludes ego at index 0).
    Returns:
        Dict with:
        - 'neighbor_prediction_loss_xy' (float): mean Euclidean distance over neighbor coords.
        - 'neighbor_prediction_loss_yaw' (float): mean abs angular error (deg) for neighbors.
    """
    # score_denorm[..., :2]: Tensor[B, Pnn, T, 2] -> (x, y)
    pred_xy = score_denorm[..., :2]  # [B, Pnn, T, 2]
    gt_xy = target_future_gt[..., :2]  # # (B, (1+)Pnn, future_len, 2)
    # Euclidean distance: sqrt((dx)^2 + (dy)^2)
    dist_ = torch.sqrt(((pred_xy - gt_xy).pow(2).sum(-1)) + 1e-6)  # [B, Pnn, T]

    # target_future_valid :# (B, Pnn, T)
    valid_dist = dist_[target_future_valid]  # [num_valid]
    valid_dist_mean = valid_dist.mean() if valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    early_stage_dist_ = dist_[..., :
                              early_stage_num]  # [B, Pnn, early_stage_num]
    # target_future_valid :# (B, Pnn, T)
    near_future_early_valid = target_future_valid[
        ..., :early_stage_num]  # [B, Pnn, early_stage_num]
    early_valid_dist = early_stage_dist_[
        near_future_early_valid]  # [num_early_valid]
    early_valid_dist_mean = early_valid_dist.mean() if early_valid_dist.numel(
    ) > 0 else torch.tensor(0.0, device=dist_.device)

    # Compute yaw angles from cos/sin
    pred_cos = score_denorm[..., 2]  # [B, P, T]
    pred_sin = score_denorm[..., 3]
    gt_cos = target_future_gt[..., 2]  # (B, (1+)Pnn, future_len)
    gt_sin = target_future_gt[..., 3]  # (B, (1+)Pnn, future_len)
    yaw_pred = torch.atan2(pred_sin, pred_cos)  # [B, P, T]
    yaw_gt = torch.atan2(gt_sin, gt_cos)
    # Angular error wrapped to [-pi, pi]
    yaw_err = (yaw_pred - yaw_gt +
               torch.pi) % (2 * torch.pi) - torch.pi  # [B, P, T]
    yaw_err_deg = torch.rad2deg(yaw_err)  # [-180, 180]

    dist_yaw = torch.abs(yaw_err_deg)  # abs error in radians # [B, P, T]
    # target_future_valid :# (B, Pnn, T)
    masked_yaw = dist_yaw[target_future_valid]
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
    diff: torch.Tensor,  # (B, P, T, C)
    valid_mask: torch.Tensor,  # (B, P, T)
    w_t: torch.Tensor,  # (1, 1, T)
    eps: float = 1e-6,
) -> torch.Tensor:
    """차이(diff)로부터 마스크/시간가중 MSE를 스칼라로 만든다.

    Args:
        diff (torch.Tensor): shape (B, P, T, C)
        valid_mask (torch.Tensor): shape (B, P, T)
        w_t (torch.Tensor): shape (1, 1, T)
        eps (float): 분모 보호용 작은 값

    Returns:
        torch.Tensor: 스칼라 손실 텐서. shape=[]
    """
    diff_f = diff.float()  # float32
    squared = (diff_f**2).sum(dim=-1)  # (B,P,T)
    valid_f = (valid_mask > 0).float()  # (B,P,T)
    w_f = w_t.float()  # (1,1,T)

    weighted = squared * w_f
    denom = (valid_f * w_f).sum().clamp_min(eps)
    return (weighted * valid_f).sum() / denom


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

    # neighbor_agents_past: (B, Pnn, time_len, 11)
    near_agents_past: torch.Tensor = norm_inputs["near_agents_past"]
    # near_current_xyyaw_norm: (B, Pnn, 4)  마지막 과거 프레임의 (x, y, cos, sin)
    near_current_xyyaw_norm: torch.Tensor = near_agents_past[:, :, -1, :4]

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
    target_future_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    eps: float,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적 크기에 맞춰 diffusion time 과 노이즈를 샘플링한다.

    Args:
        target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4) 미래 궤적.
        eps: 시간 샘플링 하한.
        args: args.feasible_learn_noise_thresh, args.use_direct_loss 사용.

    Returns:
        batch_diffusion_time: # (B,) or (B, future_len) 각 배치별 diffusion 시간.
        low_t_mask: (B,) 저노이즈 배치 마스크.
        low_t_mask_bt: (B, 1, 1) 저노이즈 마스크(브로드캐스트용).
        random_noise: (B, (1+)Pnn, future_len, 4) 노이즈 샘플.
    """
    # target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
    B: int = target_future_gt_4_dim.shape[0]
    future_len: int = int(target_future_gt_4_dim.shape[2])  # ✅ 실제 텐서에서 가져오기

    # 매번 50% 50% 확률로 배치 전체를 amortized 모드 또는 일반 모드로 처리
    use_amortized_mode: bool = (torch.rand(1).item() < 0.5)

    if args.use_amortized_diffusion and use_amortized_mode:
        # Paper t_hat: t_hat_tau = max(0, (tau - T_history) / T_future).
        # Here we add noise ONLY to FUTURE frames, so tau corresponds to 1..T_future,
        # which yields t in (0, 1].
        tau = torch.arange(
            1, future_len + 1,
            device=target_future_gt_4_dim.device,
            dtype=torch.float32,
        )  # (T,)
        t_tau = tau / float(future_len)  # (T,) = [1/T, 2/T, ..., 1]

        # (B, T) without extra memory
        batch_diffusion_time: torch.Tensor = t_tau.unsqueeze(0).expand(B, -1)

        # (B,) low-noise mask (kept as before)
        low_t_mask = torch.ones(
            B, dtype=torch.bool, device=target_future_gt_4_dim.device
        )
        low_t_mask_bt = low_t_mask.view(B, 1, 1)

    else:
        # Paper uses U(0,1). eps is a numerical guard to avoid exact t=0.
        batch_diffusion_time: torch.Tensor = (
            torch.rand(
                B,
                device=target_future_gt_4_dim.device,
                dtype=torch.float32,
            ) * (1 - eps) + eps
        )

        t_threshold: float = float(args.feasible_learn_noise_thresh)
        if not getattr(args, "use_direct_loss", False):
            t_threshold = 1.0

        low_t_mask: torch.Tensor = batch_diffusion_time <= t_threshold
        low_t_mask_bt: torch.Tensor = low_t_mask.view(B, 1, 1)

    random_noise: torch.Tensor = torch.randn_like(
        target_future_gt_4_dim,
        device=target_future_gt_4_dim.device,
    )
    return batch_diffusion_time, low_t_mask, low_t_mask_bt, random_noise


def _normalize_futures_and_build_xT(
    target_future_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    target_current_xyyaw_norm: torch.Tensor,  # (B, (1+)Pnn, 4)
    target_cur_future_mask: torch.Tensor,  # (B, (1+)Pnn, 1+future_len)
    batch_diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
    random_noise: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    state_normalizer: StateNormalizer,
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적을 정규화하고, x_T 샘플과 std 를 만든다.

    Args:
        target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4) 미래 궤적(denorm).
        target_current_xyyaw_norm: (B, (1+)Pnn, 4) 현재 상태(정규화).
        target_cur_future_mask: (B, (1+)Pnn, 1+future_len) 현재+미래 마스크.
        batch_diffusion_time: (B,) or (B, future_len) 배치별 diffusion 시간.
        random_noise: (B, (1+)Pnn, future_len, 4) 노이즈.
        state_normalizer: 상태 정규화/역정규화 도우미.
        marginal_prob: SDE 의 marginal_prob 함수.

    Returns:
        target_future_norm_gt: (B, (1+)Pnn, future_len, 4) 정규화된 미래 궤적.
        target_cur_future_norm_xT: (B, (1+)Pnn, 1+future_len, 4) 현재+미래 x_T 샘플.
        cond_last_pos_norm: (B, (1+)Pnn, 4) 미래 마지막 프레임(정규화).
        std: (B, 1, 1, 1) 노이즈 표준편차.
    """
    # target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
    B, one_or_Pnn, future_len, _ = target_future_gt_4_dim.shape

    # normed_target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
    normed_target_future_gt_4_dim: torch.Tensor = state_normalizer(
        target_future_gt_4_dim)
    target_future_mask = target_cur_future_mask[:, :,
                                                1:]  # (B, (1+)Pnn, future_len)
    normed_target_future_gt_4_dim[target_future_mask] = 0.0
    # cond_last_pos_norm: (B, (1+)Pnn, 4)
    cond_last_pos_norm: torch.Tensor = normed_target_future_gt_4_dim[:, :,
                                                                     -1, :]

    normed_target_future_gt_4_dim = _require_finite(
        "state_normalizer(near_future_gt_4_dim)", normed_target_future_gt_4_dim)

    # target_cur_future_norm_gt: (B, (1+)Pnn, 1+future_len, 4)
    target_cur_future_norm_gt: torch.Tensor = torch.cat(
        [
            target_current_xyyaw_norm[:, :, None, :],
            normed_target_future_gt_4_dim
        ],
        dim=2,
    )
    target_cur_future_norm_gt[target_cur_future_mask] = 0.0
    # target_future_norm_gt: (B, (1+)Pnn, future_len, 4)
    target_future_norm_gt: torch.Tensor = target_cur_future_norm_gt[:, :, 1:, :]

    # marginal_prob: <diffusion_planner/model/diffusion_utils/sde.py> 의 VPSDE_linear 클래스의 메서드
    mean, std = marginal_prob(target_future_norm_gt, batch_diffusion_time)
    """
    mean : (B, (1+)Pnn, future_len, 4)
    
    std : (B, 1, 1, 1) or (B, 1, future_len, 1)
    """
    mean = _require_finite("marginal_prob mean", mean)
    std = _require_finite("marginal_prob std", std)
    assert std.ndim == 4, "std_raw must be (B, _, _, _)"
    # target_future_noise_xT: (B, (1+)Pnn, future_len, 4)
    target_future_noise_xT: torch.Tensor = mean + std * random_noise
    target_future_noise_xT[target_future_mask] = 0.0

    target_cur_norm_gt = target_cur_future_norm_gt[:, :, :
                                                   1, :]  # (B, (1+)Pnn, 1, 4)

    # target_cur_future_norm_gt: (B, (1+)Pnn, 1+future_len, 4)
    target_cur_future_norm_xT: torch.Tensor = torch.cat(
        [target_cur_norm_gt, target_future_noise_xT],
        dim=2,
    )
    target_cur_future_norm_xT[target_cur_future_mask] = 0.0
    assert target_cur_future_norm_xT.shape == (B, one_or_Pnn, 1 + future_len, 4)

    return target_future_norm_gt, target_cur_future_norm_xT, cond_last_pos_norm, std


def _forward_model_with_autocast(
    model: Diffusion_Planner,
    norm_inputs: Dict[str, torch.Tensor],
    target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
    target_cur_future_norm_xT: torch.Tensor,  # (B, (1+)Pnn, 1+future_len, 4)
    batch_diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
    low_t_mask: torch.Tensor,  # (B,)
    cond_last_pos_norm: torch.Tensor,  # (B, (1+)Pnn, 4)
    use_deepspeed: bool,
) -> Dict[str, torch.Tensor]:
    """모델 입력 dict 를 만들고 AMP 로 forward 를 수행한다.

    Args:
        model: 학습 중인 모델.
        norm_inputs: 정규화된 관측 dict.
            ego_agent_past : (B, time_len, 11) #
            ego_future_gt_3_dim : (B, future_len, 3)
            neighbor_agents_past : (B, agent_num, time_len, 11) #
            lanes : (B, lane_num, lane_len, 12) #
            lanes_speed_limit : (B, lane_num, 1) #
            lanes_has_speed_limit : (B, lane_num, 1) #
            route_lanes : (B, route_num, route_len, 12)
            route_lanes_speed_limit : (B, route_num, 1)
            route_lanes_has_speed_limit : (B, route_num, 1)
            static_objects : (B, static_num, 10) #
            near_future_gt_3_dim: (B, Pnn, future_len, 3)
            planner_future_11_dim: (B, future_len, 11) #
            agent_route_lane_order: (B, agent_num, lane_num)

        target_future_valid: (B, (1 +) Pnn, future_len) 미래 유효 마스크.
        target_cur_future_norm_xT: (B, (1+)Pnn, 1+future_len, 4) 현재+미래 x_T.
        batch_diffusion_time: (B,) or (B, T) diffusion 시간.
        cond_last_pos_norm: (B, (1+)Pnn, 4) cond 용 마지막 위치.

    Returns:
        decoder_output: model 의 두 번째 반환값 dict.


        "near_future_valid": near_future_valid,  # (B, Pnn, future_len)
        "near_cur_future_norm_xT":
            near_cur_future_norm_xT,  # (B, Pnn, 1+future_len, 4)
    """
    merged_inputs: Dict[str, torch.Tensor] = {
        **norm_inputs,
        "target_future_valid":
            target_future_valid,  # (B, (1 +) Pnn, future_len)
        "target_cur_future_norm_xT":
            target_cur_future_norm_xT,  # (B, (1+)Pnn, 1+future_len, 4)
        "diffusion_time": batch_diffusion_time,  # (B,) or (B, T)
        "low_t_mask": low_t_mask,  # (B,)
        "cond_last_pos_norm": cond_last_pos_norm,  # (B, (1+)Pnn, 4)
    }
    is_ds_engine = hasattr(model, "backward") and hasattr(
        model, "step") and hasattr(model, "module")

    if use_deepspeed:
        assert is_ds_engine, "use_deepspeed=True 인데 model 이 DS engine 아님"
        # target_past_cur_future_valid
        _, decoder_output = model(merged_inputs)  # DS가 torch_autocast로 처리
    else:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, decoder_output = model(merged_inputs)
    return decoder_output  # decoder_output["score"], ["integrated_trajectory"], ...


def _extract_score_from_decoder(
    decoder_output: Dict[str, torch.Tensor],
    B: int,
    one_or_Pnn: int,
    future_len: int,
) -> torch.Tensor:
    """decoder_output 에서 미래 score 만 꺼내고 모양을 확인한다.

    Args:
        decoder_output: model(...) 의 두 번째 반환 dict.
        B: 배치 크기.
        one_or_Pnn: 이웃 개수.
        future_len: 미래 길이.

    Returns:
        score: (B, one_or_Pnn, future_len, 4) 예측된 미래 궤적.
    """
    # decoder_output["score"]: (B, one_or_Pnn, 1+future_len, 4)
    score: torch.Tensor = decoder_output["score"][:, :, 1:, :]
    score = _require_finite("decoder_output['score']", score)
    assert score.shape == (B, one_or_Pnn, future_len, 4)
    return score


def _compute_dpm_loss(
        args: Any,
        model_type: str,
        score: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
        std: torch.Tensor,  # (B, 1, 1, 1)
        random_noise: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
        target_future_norm_gt: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
) -> torch.Tensor:
    """기존 diffusion 손실(score/x_start)을 (B,P,future_len) 형태로 계산한다.

    Args:
        args: args.use_huber_loss 사용.
        model_type: "score" 또는 "x_start".

    Returns:
        dpm_loss: (B, (1+)Pnn, future_len) 위치별 손실 값.
    """
    HUBER_DELTA: float = 1.0

    if getattr(args, "use_huber_loss", True):
        # err: (B, (1+)Pnn, future_len, 4)
        if model_type == "score":
            err: torch.Tensor = score * std + random_noise
        elif model_type == "x_start":
            err = score - target_future_norm_gt
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        abs_err: torch.Tensor = err.abs()  # (B, (1+)Pnn, future_len, 4)
        quad: torch.Tensor = 0.5 * err.pow(2)  # (B, (1+)Pnn, future_len, 4)
        lin: torch.Tensor = HUBER_DELTA * (abs_err - 0.5 * HUBER_DELTA
                                          )  # (B, (1+)Pnn, future_len, 4)
        huber: torch.Tensor = torch.where(abs_err <= HUBER_DELTA, quad,
                                          lin)  # (B, (1+)Pnn, future_len, 4)
        dpm_loss: torch.Tensor = huber.sum(dim=-1)  # (B, (1+)Pnn, future_len)
    else:
        if model_type == "score":
            dpm_loss = torch.sum((score * std + random_noise)**2,
                                 dim=-1)  # (B, (1+)Pnn, future_len)
        elif model_type == "x_start":
            dpm_loss = torch.sum((score - target_future_norm_gt)**2,
                                 dim=-1)  # (B, (1+)Pnn, future_len, 4)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    return dpm_loss


def _aggregate_weighted_loss(
    per_step_loss: torch.Tensor,  # (B, (1+)Pnn, T)
    target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len) bool 또는 0/1
    w_t: torch.Tensor,  # (1, 1, T)
    eps: float = 1e-6,
) -> torch.Tensor:
    """시간 가중치와 유효 마스크로 (B,P,T) 손실을 스칼라로 합친다.

    구현 안정성
    ----------
    AMP(bfloat16) 환경에서는 합(sum)이나 분모 계산이 거칠어질 수 있어서,
    이 함수 내부 계산은 float32로 올려서 진행합니다.
    Returns:
        torch.Tensor: 스칼라 손실 텐서. shape=[]
    """
    loss_f = per_step_loss.float()  # (B,P,T) float32
    valid_f = (target_future_valid
               > 0).float()  # (B, (1 +) Pnn, future_len) float32
    w_f = w_t.float()  # (1,1,T) float32

    weighted = loss_f * w_f  # (B,P,T)
    denom = (valid_f * w_f).sum().clamp_min(eps)  # scalar
    return (weighted * valid_f).sum() / denom


def _compute_integration_and_constraint_losses(
    args: Any,
    decoder_output: Dict[str, torch.Tensor],
    target_future_norm_gt: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
    low_t_mask_bt: torch.Tensor,  # (B, 1, 1)
    w_t: torch.Tensor,  # (1, 1, T)
    base_loss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor]]:
    """통합 궤적/제어 편차 기반 보조 손실을 계산한다.

    내부 계산은 float32로 수행해서 수치 불안정 가능성을 줄입니다.

    Args:
        target_future_norm_gt: (B, (1+)Pnn, future_len, 4)
        target_future_valid: ((B, (1 +) Pnn, future_len)
        low_t_mask_bt: (B, 1, 1)
        w_t: (1, 1, T)

    Returns:
        integration_loss_val: 스칼라, float32
        constraint_loss_val: 스칼라, float32
        integrated_trajectory: (B, (1+)Pnn, T, 4) 또는 None
        control_constraint_diff: (B, (1+)Pnn, T, 3) 또는 None
    """
    valid_low = target_future_valid & low_t_mask_bt  # (B, (1 +) Pnn, future_len) bool
    valid_low_f = valid_low.float()

    integrated_trajectory: Optional[torch.Tensor] = None
    control_constraint_diff: Optional[torch.Tensor] = None

    # --- L_integration ---
    if "integrated_trajectory" in decoder_output and getattr(
            args, "use_feasible_dl", False):
        integrated_full = _require_finite(
            "decoder_output['integrated_trajectory']",
            decoder_output["integrated_trajectory"],
        )  # (B, (1+)Pnn, 1+T, 4)
        integrated_trajectory = integrated_full[:, :,
                                                1:, :]  # (B, (1+)Pnn, T, 4)

        # per_step: (B,P,T) float32
        diff = (integrated_trajectory -
                target_future_norm_gt).float()  # (B, (1+)Pnn, future_len, 4)
        per_step = (diff**2).sum(dim=-1)

        w_f = w_t.float()
        denom = (valid_low_f * w_f).sum().clamp_min(1e-6)
        integration_loss_val = (per_step * w_f * valid_low_f).sum() / denom
    else:
        integration_loss_val = torch.zeros((),
                                           device=base_loss.device,
                                           dtype=torch.float32)

    # --- L_constraint ---
    if "control_constraint_diff" in decoder_output and getattr(
            args, "use_feasible_dl", False):
        control_constraint_diff = _require_finite(
            "decoder_output['control_constraint_diff']",
            decoder_output["control_constraint_diff"],
        )  # (B,P,T,3)

        constraint_loss_val = _masked_weighted_mse_from_diff(
            control_constraint_diff,
            valid_low,
            w_t,
        )
    else:
        constraint_loss_val = torch.zeros((),
                                          device=base_loss.device,
                                          dtype=torch.float32)

    return integration_loss_val, constraint_loss_val, integrated_trajectory, control_constraint_diff


def _add_xy_yaw_metric_losses(
    loss_dict: Dict[str, Any],
    state_normalizer: StateNormalizer,
    observation_normalizer: Any,
    score: torch.Tensor,  # (B, Pnn, T, 4)
    target_future_norm_gt: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    target_future_valid: torch.Tensor,  # (B, Pnn, T)
    integrated_trajectory: Optional[torch.Tensor],
    control_constraint_diff: Optional[torch.Tensor],
) -> None:
    """xy / yaw 관련 부가 지표들을 loss_dict dict 에 추가한다.

    Args:
        loss_dict: 손실 및 통계 값을 담는 dict (in-place 업데이트).
        state_normalizer: 상태 역정규화용 도우미.
        observation_normalizer: 제어 역정규화용 도우미.
        score: (B, Pnn, T, 4) 모델 예측(정규화).
        target_future_norm_gt: (B, (1+)Pnn, future_len, 4) 미래 GT(정규화).
        target_future_valid: (B, (1 +) Pnn, future_len) 미래 유효 마스크.
        integrated_trajectory: (B, Pnn, T, 4) 또는 None.
        control_constraint_diff: (B, Pnn, T, 3) 또는 None.
    """
    # score_denorm: (B, Pnn, T, 4)
    score_denorm: torch.Tensor = state_normalizer.inverse(score)
    # target_future_gt: # (B, (1+)Pnn, future_len, 4)
    target_future_gt: torch.Tensor = state_normalizer.inverse(
        target_future_norm_gt)

    with torch.no_grad():
        # 기본 score 에 대한 xy/yaw 오차
        xy_yaw_losses = _compute_xy_yaw_losses(
            score_denorm,
            target_future_gt,  # (B, (1+)Pnn, future_len, 4)
            target_future_valid,  # (B, Pnn, T)
        )
        loss_dict.update(xy_yaw_losses)

        # 통합 궤적에 대한 xy/yaw 오차
        if integrated_trajectory is not None:
            # integrated_trajectory_denorm: (B, Pnn, T, 4)
            integrated_trajectory_denorm: torch.Tensor = \
                state_normalizer.inverse(integrated_trajectory)
            integ_xy_yaw_losses = _compute_xy_yaw_losses(
                integrated_trajectory_denorm,
                target_future_gt,  # (B, (1+)Pnn, future_len, 4)
                target_future_valid,  # (B, Pnn, T)
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
                target_future_valid,  # (B, Pnn, T)
                prefix="constraint_diff",
            )
            loss_dict.update(constraint_xy_yaw_losses)


def _assert_cur_future_valid_mask(
        valid_bpt: torch.Tensor,
        context: str = "savgol_filter_for_control") -> None:
    """유효 마스크가 행마다 True*False* (단조 감소)인지 검증.

    Args:
        valid_bpt: (B, Pnn, T1) bool, 시간 축 마지막.
        context: 에러 메시지에 표시할 호출 위치 문자열.

    Raises:
        ValueError: 0→1 전이가 하나라도 발견되면(내부 구멍 또는 선행 무효 후 유효)
    """
    assert valid_bpt.dim() == 3, "valid_bpt는 (B,Pnn,T1) 여야 합니다."
    B, Pnn, T1 = valid_bpt.shape
    v = valid_bpt.reshape(-1, T1).to(torch.int8)  # (B*Pnn, T1)
    d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
    has_01 = (d > 0).any(dim=1)  # 0→1 전이 여부
    if has_01.any():
        bad_idx = torch.nonzero(has_01, as_tuple=False).flatten()
        # 가독성을 위해 일부만 표시
        max_show = min(int(bad_idx.numel()), 8)
        bad_idx_sample = bad_idx[:max_show].tolist()
        # (b,p) 인덱스 매핑
        b_list = [(i // Pnn) for i in bad_idx_sample]
        p_list = [(i % Pnn) for i in bad_idx_sample]
        raise ValueError(
            f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*). \n"
            f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.numel())},  \n"
            f"example (b,p)={list(zip(b_list, p_list))}.  \n"
            f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
        )


def diffusion_loss_func(
    args: Any,
    model: Diffusion_Planner,
    norm_inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
    ego_future_gt_4_dim: torch.Tensor,  # (B, future_len, 4)
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
        model: 학습 중인 모델(Diffusion_Planner 또는 DDP 래퍼).
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
    # 미래/현재 마스크 및 현재 상태 준비
    # near_future_valid: (B, Pnn, future_len)
    # near_cur_future_mask: (B, Pnn, 1+future_len)
    # near_current_xyyaw_norm: (B, Pnn, 4)
    near_future_valid, near_cur_future_mask, near_current_xyyaw_norm = \
        _build_future_masks_and_current_state(
            near_future_gt_4_dim, # (B, Pnn, future_len, 4)
            near_future_mask, # (B, Pnn, future_len)
            norm_inputs,
        )
    (
        target_future_gt_4_dim,  # (B, (1+)Pnn, future_len, 4)
        target_cur_future_mask,  # (B, (1+)Pnn, 1+future_len)
        target_current_xyyaw_norm,  # (B, (1+)Pnn, 4)
        target_future_valid,  # (B, (1+)Pnn, future_len)
    ) = build_target_future_tensors_and_masks(
        args=args,
        norm_inputs=norm_inputs,
        ego_future_gt_4_dim=ego_future_gt_4_dim,  # (B, future_len, 4)
        near_future_gt_4_dim=near_future_gt_4_dim,  # (B, Pnn, future_len, 4)
        near_cur_future_mask=near_cur_future_mask,  # (B, Pnn, 1 + future_len)
        near_current_xyyaw_norm=near_current_xyyaw_norm,  # (B, Pnn, 4)
        near_future_valid=near_future_valid,  # (B, Pnn, future_len)
    )

    B, one_or_Pnn, future_len, = target_future_valid.shape
    """
    # diffusion time / low noise mask / random noise 샘플링
    
    # batch_diffusion_time: # (B,) or (B, future_len)
    # low_t_mask: (B,)
    # low_t_mask_bt: (B,1,1)
    # random_noise: (B, (1+)Pnn, future_len, 4)
    """
    (batch_diffusion_time, low_t_mask, low_t_mask_bt,
     random_noise) = _sample_diffusion_time_and_noise(
         target_future_gt_4_dim,  # (B, (1+)Pnn, future_len, 4)
         eps,
         args,
     )

    # 미래 궤적 정규화 + x_T 샘플 생성
    """
        target_future_norm_gt: (B, (1+)Pnn, future_len, 4) 정규화된 미래 궤적.
        target_cur_future_norm_xT: (B, (1+)Pnn, 1+future_len, 4) 현재+미래 x_T 샘플.
        cond_last_pos_norm: (B, (1+)Pnn, 4) 미래 마지막 프레임(정규화).
        std: (B, 1, 1, 1) 노이즈 표준편차.
    """
    (target_future_norm_gt, target_cur_future_norm_xT, cond_last_pos_norm,
     std) = _normalize_futures_and_build_xT(
         target_future_gt_4_dim,
         target_current_xyyaw_norm,  # (B, (1+)Pnn, 4)
         target_cur_future_mask,
         batch_diffusion_time,  # (B,) or (B, future_len)
         random_noise,
         state_normalizer,
         marginal_prob,
     )

    # 모델 forward + decoder_output 생성
    decoder_output: Dict[str, torch.Tensor] = _forward_model_with_autocast(
        model=model,  #
        norm_inputs=norm_inputs,  #
        target_future_valid=target_future_valid,  # (B, (1 +) Pnn, future_len)
        target_cur_future_norm_xT=
        target_cur_future_norm_xT,  # (B, (1+)Pnn, 1+future_len, 4)
        batch_diffusion_time=batch_diffusion_time,  # (B,) or (B, future_len)
        low_t_mask=low_t_mask,  # (B,)
        cond_last_pos_norm=cond_last_pos_norm,  # (B, (1+)Pnn, 4)
        use_deepspeed=getattr(args, "use_deepspeed", False),
    )

    # score:  (B, one_or_Pnn, future_len, 4)
    score: torch.Tensor = _extract_score_from_decoder(
        decoder_output=decoder_output,
        B=B,
        one_or_Pnn=one_or_Pnn,
        future_len=future_len,
    )

    # dpm_loss: (B, (1+)Pnn, future_len)
    dpm_loss: torch.Tensor = _compute_dpm_loss(
        args=args,
        model_type=model_type,
        score=score,  # (B, (1+)Pnn, future_len, 4)
        std=std,  # (B, 1, 1, 1)
        random_noise=random_noise,  # (B, (1+)Pnn, future_len, 4)
        target_future_norm_gt=
        target_future_norm_gt,  # (B, (1+)Pnn, future_len, 4)
    )

    # 시간 가중치(w_t) 생성
    time_step_s: float = 0.1
    half_life_s: float = 2.0
    # w_t: (1, 1, future_len)
    w_t: torch.Tensor = _build_half_life_weights(
        future_len,
        dt_s=time_step_s,
        half_life_s=half_life_s,
        device=dpm_loss.device,  # (B, (1+)Pnn, future_len)
        dtype=torch.float32,
    )

    # neighbor_prediction_loss (스칼라)
    loss_val: torch.Tensor = _aggregate_weighted_loss(
        per_step_loss=dpm_loss,  # (B, (1+)Pnn, future_len)
        target_future_valid=target_future_valid,  # (B, (1 +) Pnn, future_len)
        w_t=w_t,  # (1, 1, future_len)
        eps=1e-6,
    )
    loss_dict["neighbor_prediction_loss"] = loss_val

    # x_start 모드에서만 Feasible 관련 손실/지표 계산
    if model_type == "x_start":
        # 통합 궤적/제어 제약 손실
        """
        integration_loss_val: 스칼라
        constraint_loss_val: 스칼라
        
        integrated_trajectory: (B, (1+)Pnn, T, 4) 또는 None
        control_constraint_diff: (B, (1+)Pnn, T, 3) 또는 None
        
=        """
        (integration_loss_val, constraint_loss_val, integrated_trajectory,
         control_constraint_diff) = _compute_integration_and_constraint_losses(
             args=args,
             decoder_output=decoder_output,
             target_future_norm_gt=
             target_future_norm_gt,  # (B, (1+)Pnn, future_len, 4)
             target_future_valid=
             target_future_valid,  # (B, (1 +) Pnn, future_len)
             low_t_mask_bt=low_t_mask_bt,  # (B,1,1)
             w_t=w_t,  # (1, 1, future_len)
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
            target_future_norm_gt=
            target_future_norm_gt,  # (B, (1+)Pnn, future_len, 4)
            target_future_valid=target_future_valid,  # (B, (1 +) Pnn, future_len)
            integrated_trajectory=
            integrated_trajectory,  # (B, (1+)Pnn, T, 4) 또는 None
            control_constraint_diff=
            control_constraint_diff,  # (B, (1+)Pnn, T, 3) 또는 None
        )

    # dpm_loss 전체가 유한값인지 마지막으로 검사
    assert torch.isfinite(dpm_loss).all().item(), \
        f"loss cannot be nan, random_noise={random_noise}"

    return loss_dict, decoder_output
