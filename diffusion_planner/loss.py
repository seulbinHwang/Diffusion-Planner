from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple, Optional

import logging

import torch
import torch.nn as nn
from diffusion_planner.utils.normalizer import StateNormalizer
from diffusion_planner.utils.target_feature import build_target_future_tensors_and_masks, build_target_future_tensors_and_masks_vel

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype

from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


def _unwrap_to_core_module(model: nn.Module) -> nn.Module:
    """DeepSpeed/DDP 래퍼가 씌워져 있어도 실제 nn.Module을 꺼냅니다.

    Args:
        model (nn.Module): 래핑될 수도 있는 모델. shape: ()

    Returns:
        nn.Module: 가능한 한 안쪽의 실제 nn.Module. shape: ()
    """
    cur: nn.Module = model
    for _ in range(8):
        inner = getattr(cur, "module", None)
        if isinstance(inner, nn.Module):
            cur = inner
        else:
            break
    return cur


def _collect_tensor_dtypes_in_dict(data: Dict[str, Any]) -> Dict[str, str]:
    """dict 안의 torch.Tensor 항목들의 dtype을 모아 요약합니다.

    Args:
        data (Dict[str, Any]): 입력 dict. value가 Tensor일 수도 있음.

    Returns:
        Dict[str, str]: key -> dtype 문자열 요약.
    """
    out: Dict[str, str] = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            out[str(k)] = str(v.dtype)
    return out


def _find_fused_mlp_modules(root: nn.Module) -> List[Tuple[str, nn.Module]]:
    """모델 내부에서 fused MLP 모듈(FusedMlpGelu)을 찾아 반환합니다.

    Args:
        root (nn.Module): 실제 nn.Module. shape: ()

    Returns:
        List[Tuple[str, nn.Module]]:
            - (모듈 이름, 모듈) 리스트
            - 모듈 이름은 root.named_modules() 기준의 경로 문자열
    """
    fused_list: List[Tuple[str, nn.Module]] = []
    for name, m in root.named_modules():
        # import 순환을 피하려고 클래스 이름 문자열로만 판정
        if m.__class__.__name__ == "FusedMlpGelu":
            fused_list.append((name, m))
    return fused_list


def forward_once_and_check_bf16(
    model: nn.Module,
    merged_inputs: Dict[str, Any],
    *,
    use_deepspeed: bool,
    amp_dtype: torch.dtype = torch.bfloat16,
    strict_bf16: bool = True,
    max_records: int = 16,
) -> Dict[str, torch.Tensor]:
    """forward 1회 실행 중 fused MLP가 bf16으로 잘 돌고 있는지 확인합니다.

    이 함수가 하는 일
    - fused MLP(FusedMlpGelu) 모듈들에 forward hook을 달아,
      실제로 fused MLP로 들어가는 입력 텐서의 dtype/shape/device를 기록합니다.
    - fused MLP(fc1/fc2) 파라미터 dtype도 같이 기록합니다.
    - forward는 사용자가 제시한 로직 그대로 실행합니다:
        - use_deepspeed=True  -> autocast 없이 model(merged_inputs)
        - use_deepspeed=False -> torch.autocast("cuda", dtype=amp_dtype)로 감싸서 실행
    - strict_bf16=True이면:
        - fused MLP 입력 dtype이 torch.bfloat16이 아니거나
        - fused MLP 파라미터 dtype이 torch.bfloat16이 아니면
      바로 RuntimeError를 발생시킵니다.

    Args:
        model (nn.Module): DeepSpeedEngine 또는 일반 nn.Module. shape: ()
        merged_inputs (Dict[str, Any]): 모델 입력 dict. 텐서 value들의 shape은 배치에 따라 다름.
        use_deepspeed (bool): DeepSpeed 경로 사용 여부.
        amp_dtype (torch.dtype): non-DS 경로에서 사용할 autocast dtype (기본 bf16).
        strict_bf16 (bool): True면 bf16 아니면 즉시 실패.
        max_records (int): hook 기록 최대 개수(너무 많이 쌓이는 것 방지).

    Returns:
        Dict[str, torch.Tensor]:
            decoder_output dict. (model(merged_inputs)의 두 번째 반환값)
    """
    core: nn.Module = _unwrap_to_core_module(model)
    fused_mlps: List[Tuple[str, nn.Module]] = _find_fused_mlp_modules(core)

    records: List[Dict[str, Any]] = []
    handles: List[Any] = []

    def _make_hook(mod_name: str) -> Any:
        """forward hook 생성"""

        def _hook(mod: nn.Module, inputs: Tuple[Any, ...], output: Any) -> None:
            # inputs[0]이 보통 x 입니다.
            if len(records) >= int(max_records):
                return
            x = inputs[0] if len(inputs) > 0 else None
            if not isinstance(x, torch.Tensor):
                return

            rec: Dict[str, Any] = {
                "module": mod_name,
                "class": mod.__class__.__name__,
                "x_dtype": x.dtype,
                "x_device": str(x.device),
                "x_shape": tuple(int(s) for s in x.shape),  # shape: (..)
            }

            # FusedMlpGelu는 fc1/fc2가 있다고 가정(너 코드 기준)
            fc1 = getattr(mod, "fc1", None)
            fc2 = getattr(mod, "fc2", None)
            if isinstance(fc1, nn.Linear) and isinstance(fc2, nn.Linear):
                rec["fc1_w_dtype"] = fc1.weight.dtype
                rec["fc2_w_dtype"] = fc2.weight.dtype
                rec["fc1_b_dtype"] = (fc1.bias.dtype
                                      if fc1.bias is not None else None)
                rec["fc2_b_dtype"] = (fc2.bias.dtype
                                      if fc2.bias is not None else None)

            records.append(rec)

        return _hook

    # hook 장착
    for name, m in fused_mlps:
        handles.append(m.register_forward_hook(_make_hook(name)))

    # (옵션) DeepSpeed 엔진 플래그 출력에 도움되는 값들
    ds_bf16_enabled: Optional[bool] = None
    if use_deepspeed and hasattr(model, "bfloat16_enabled"):
        try:
            ds_bf16_enabled = bool(model.bfloat16_enabled())
        except Exception:
            ds_bf16_enabled = None

    # 입력 dtype 요약(참고용)
    input_dtype_map = _collect_tensor_dtypes_in_dict(merged_inputs)

    # forward 실행 (사용자 제시 로직 그대로)
    if use_deepspeed:
        _, decoder_output = model(merged_inputs)
    else:
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            _, decoder_output = model(merged_inputs)

    # hook 제거
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass

    # ---- 결과 출력 ----
    print("\n==================== [BF16 CHECK] ====================")
    print(f"- use_deepspeed: {use_deepspeed}")
    if ds_bf16_enabled is not None:
        print(f"- deepspeed_engine.bfloat16_enabled(): {ds_bf16_enabled}")
    print(
        f"- torch.is_autocast_enabled() (after forward): {torch.is_autocast_enabled()}"
    )
    print("- merged_inputs tensor dtypes (요약):")
    for k, v in sorted(input_dtype_map.items()):
        print(f"  - {k}: {v}")

    print(f"- found FusedMlpGelu modules: {len(fused_mlps)}")
    print(f"- captured fused-mlp call records: {len(records)}")

    if len(records) > 0:
        # record 몇 개만 보기
        show_n = min(8, len(records))
        print(f"- fused-mlp records (상위 {show_n}개):")
        for i in range(show_n):
            r = records[i]
            print("  - "
                  f"[{i}] {r.get('module')} | "
                  f"x_dtype={r.get('x_dtype')} | "
                  f"x_shape={r.get('x_shape')} | "
                  f"fc1_w_dtype={r.get('fc1_w_dtype')} | "
                  f"fc2_w_dtype={r.get('fc2_w_dtype')}")
    print("======================================================\n")

    # ---- strict 검사 ----
    if strict_bf16:
        if len(records) == 0:
            raise RuntimeError("BF16 체크 실패: fused MLP 호출 기록이 0개입니다. "
                               "이 배치에서 유효 토큰이 0개였거나, 해당 경로가 실행되지 않았을 수 있습니다.")

        for r in records:
            x_dtype = r.get("x_dtype", None)
            fc1_w_dtype = r.get("fc1_w_dtype", None)
            fc2_w_dtype = r.get("fc2_w_dtype", None)

            if x_dtype != torch.bfloat16:
                raise RuntimeError(
                    "BF16 체크 실패: fused MLP 입력 dtype이 bf16이 아닙니다. "
                    f"module={r.get('module')}, x_dtype={x_dtype}")
            if fc1_w_dtype != torch.bfloat16 or fc2_w_dtype != torch.bfloat16:
                raise RuntimeError(
                    "BF16 체크 실패: fused MLP 파라미터 dtype이 bf16이 아닙니다. "
                    f"module={r.get('module')}, fc1_w_dtype={fc1_w_dtype}, fc2_w_dtype={fc2_w_dtype}"
                )

    return decoder_output


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

import math
from typing import Any


def _get_diffusion_time_eps(args: Any, fallback_eps: float) -> float:
    """diffusion 시간 샘플링에서 사용할 eps를 한 값으로 통일합니다.

    eps는 시간값 t가 0이나 1에 너무 가까워지는 것을 피하기 위한 작은 값입니다.
    학습(loss)과 추론(decoder)에서 같은 eps를 쓰도록,
    args.diffusion_time_eps가 있으면 그 값을 우선 사용합니다.

    Args:
        args (Any): 설정 객체. diffusion_time_eps 속성이 있을 수 있습니다. shape: ()
        fallback_eps (float): args에 값이 없거나 읽기 실패 시 사용할 기본값. shape: ()

    Returns:
        float: 사용할 eps 값. (0, 1) 범위로 안전하게 제한됩니다. shape: ()
    """
    raw = getattr(args, "diffusion_time_eps", None)
    if raw is None:
        eps_val = float(fallback_eps)
    else:
        try:
            eps_val = float(raw)
        except Exception:
            eps_val = float(fallback_eps)

    if not math.isfinite(eps_val):
        eps_val = float(fallback_eps)

    # t_max = 1 - eps 형태로 쓰이므로, (0, 1) 범위를 벗어나면 깨질 수 있어 안전 클램프
    eps_val = max(1e-8, min(eps_val, 1.0 - 1e-8))
    return float(eps_val)


def _to_bool_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """마스크 텐서를 bool로 통일합니다.

    이 함수는 마스크의 의미(True/False)를 바꾸지 않고, dtype만 안전하게 bool로 맞춥니다.
    - 이미 bool이면 그대로 반환합니다.
    - float이면 (mask > threshold)로 True/False를 만듭니다.
    - int면 (mask != 0)로 True/False를 만듭니다.

    Args:
        mask (torch.Tensor): 임의 shape 마스크 텐서. bool/int/float 모두 허용.
        threshold (float): float 마스크에서 True로 볼 기준값.

    Returns:
        torch.Tensor: 입력과 동일 shape의 bool 텐서.
    """
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return mask > float(threshold)
    return mask != 0


def _integrate_midpoint_controls_to_pose_denorm(
    unnorm_target_cur_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, 4)
    seg_body_control_denorm: torch.Tensor,  # (B,(1 +) Pnn,future_len,3)
    target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
    *,
    dt: float = 0.1,
    eps: float = 1e-6,
) -> torch.Tensor:
    """바디 프레임 제어(denorm)를 현재 포즈(denorm)에서 시작해 중점 적분으로 미래 포즈를 만든다.

    이 함수는 FeasibleProjector._integrate_midpoint_batch 와 동일한 방식으로 적분한다.
    지표 계산용이라 연산은 float32로 올려 안정적으로 처리한다.

    Args:
        unnorm_target_cur_gt_4_dim:
            현재 포즈. shape: (B, P, 4) = [x0, y0, cos0, sin0]
        seg_body_control_denorm:
            미래 세그먼트 제어. shape: (B, P, T, 3) = [v_x^b, v_y^b, yaw_rate]
            여기서 yaw_rate 단위는 rad/s 를 가정한다.
        target_future_valid:
            미래 노드 유효 마스크. shape: (B, P, T) (bool 또는 0/1)
        dt:
            시간 간격(초). 기본 0.1
        eps:
            수치 안정용 작은 값.

    Returns:
        score_pose_denorm:
            미래 포즈(노드) 시퀀스. shape: (B, P, T, 4) = [x, y, cos, sin]
            무효 스텝은 0으로 마스킹된다.
    """
    B, P, T, _ = seg_body_control_denorm.shape
    future_valid = _to_bool_mask(target_future_valid).to(
        device=seg_body_control_denorm.device)

    # 현재 유효 여부(현재 포즈가 invalid면 보통 cos/sin이 0으로 마스킹되어 있음)
    cos0 = unnorm_target_cur_gt_4_dim[..., 2]
    sin0 = unnorm_target_cur_gt_4_dim[..., 3]
    cur_valid = ((cos0 * cos0 + sin0 * sin0) > 0.25)  # (B,P) bool

    # 세그먼트 유효: (t_k, t_{k+1}) 양 끝 노드가 모두 유효해야 함
    # 여기서는 현재 노드 유효를 cur_valid로, 미래 노드 유효를 future_valid로 둔다.
    seg_valid = cur_valid.unsqueeze(-1) & future_valid  # (B,P,T) bool

    # float32로 올려서 안정적으로 적분(지표용)
    cur_f = unnorm_target_cur_gt_4_dim.float()  # (B,P,4)
    ctrl_f = seg_body_control_denorm.float()  # (B,P,T,3)
    seg_valid_f = seg_valid.to(dtype=torch.float32,
                               device=ctrl_f.device)  # (B,P,T)

    vx_b = ctrl_f[..., 0] * seg_valid_f  # (B,P,T)
    vy_b = ctrl_f[..., 1] * seg_valid_f  # (B,P,T)
    omega = ctrl_f[..., 2] * seg_valid_f  # (B,P,T)

    x0 = cur_f[..., 0]  # (B,P)
    y0 = cur_f[..., 1]  # (B,P)
    cos0 = cur_f[..., 2]  # (B,P)
    sin0 = cur_f[..., 3]  # (B,P)

    yaw0 = torch.atan2(sin0, cos0)  # (B,P)

    dtheta = omega * float(dt)  # (B,P,T)
    dtheta_prefix = torch.cumsum(dtheta, dim=2)  # (B,P,T)
    zero_pad = torch.zeros_like(dtheta[..., :1])  # (B,P,1)
    dtheta_exclusive = torch.cat([zero_pad, dtheta_prefix[..., :-1]],
                                 dim=2)  # (B,P,T)

    yaw_start = yaw0.unsqueeze(-1) + dtheta_exclusive  # (B,P,T)
    yaw_mid = yaw_start + 0.5 * dtheta  # (B,P,T)
    yaw_next = yaw_start + dtheta  # (B,P,T)

    cos_mid = torch.cos(yaw_mid)  # (B,P,T)
    sin_mid = torch.sin(yaw_mid)  # (B,P,T)

    vwx_mid = cos_mid * vx_b - sin_mid * vy_b  # (B,P,T)
    vwy_mid = sin_mid * vx_b + cos_mid * vy_b  # (B,P,T)

    dx = vwx_mid * float(dt)  # (B,P,T)
    dy = vwy_mid * float(dt)  # (B,P,T)

    x_next = x0.unsqueeze(-1) + torch.cumsum(dx, dim=2)  # (B,P,T)
    y_next = y0.unsqueeze(-1) + torch.cumsum(dy, dim=2)  # (B,P,T)

    cos_next = torch.cos(yaw_next)  # (B,P,T)
    sin_next = torch.sin(yaw_next)  # (B,P,T)
    norm_cs = torch.sqrt(cos_next * cos_next + sin_next * sin_next +
                         float(eps))  # (B,P,T)
    cos_next = cos_next / norm_cs
    sin_next = sin_next / norm_cs

    score_pose_denorm = torch.stack([x_next, y_next, cos_next, sin_next],
                                    dim=-1)  # (B,P,T,4)

    # 노드 유효 마스크: 현재가 무효면 미래도 전부 무효 처리
    node_valid = future_valid & cur_valid.unsqueeze(-1)  # (B,P,T)
    score_pose_denorm = score_pose_denorm.masked_fill(~node_valid.unsqueeze(-1),
                                                      0.0)

    return score_pose_denorm


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
    Tensor,  # [B, P, T, 3], (vx[m/s], vy[m/s], yaw_rate[rad/s])
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
        score_denorm: torch.Tensor,  # (B, (1+)Pnn, T, 4)
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
    target_future_valid = _to_bool_mask(target_future_valid).to(
        device=score_denorm.device)
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


def _compute_vxy_yaw_losses(
        score_denorm: torch.Tensor,  # (B, (1+)Pnn, T, 3)
        target_future_gt: torch.Tensor,  # (B, (1+)Pnn, future_len, 3)
        target_future_valid: torch.Tensor,
        early_stage_num: int = 5,
        prefix: str = "neighbor_prediction_loss") -> Dict[str, torch.Tensor]:
    target_future_valid = _to_bool_mask(target_future_valid).to(
        device=score_denorm.device)
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
    yaw_pred = score_denorm[..., 2]  # [B, P, T]
    yaw_gt = target_future_gt[..., 2]  # [B, P, T]
    yaw_err_deg = torch.rad2deg(yaw_pred - yaw_gt)
    dist_yaw = torch.abs(yaw_err_deg)  # abs error in degrees # [B, P, T]
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
        f'{prefix}_vxy': valid_dist_mean,  # scalar
        f'{prefix}_yaw_rate': neigh_yaw,  # scalar # degree
        f'{prefix}_vxy_early': early_valid_dist_mean,  # scalar
        f'{prefix}_yaw_rate_early': early_valid_yaw_mean,  # scalar # degree
    }


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


def _sample_diffusion_time_and_noise(
    # (B, (1+)Pnn, future_len or 1+future_len, 4 or 3)
    normed_target_seq_gt: torch.Tensor,
    eps: float,
    args: Any,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적 크기에 맞춰 diffusion time 과 노이즈를 샘플링한다."""
    device_ = normed_target_seq_gt.device
    B: int = normed_target_seq_gt.shape[0]
    one_Pnn = normed_target_seq_gt.shape[1]
    feature_dim = normed_target_seq_gt.shape[-1]
    future_len = args.future_len
    one_future_len = future_len + 1

    use_amortized_mode: bool = (torch.rand(1).item() < 0.5)

    if args.use_amortized_diffusion and use_amortized_mode:
        # len(tau) = future_len
        tau = torch.arange(
            1,
            one_future_len,
            device=device_,
            dtype=torch.float32,
        )  # (T,) # [ 1, 2, ..., T ]

        t_tau = tau / float(future_len)  # (T,) = [1/T, 2/T, ..., 1]

        # 정확히 t=1을 피해서 수치 리스크를 줄임 (일반 모드도 t=1은 안 씀)
        t_max: float = max(1.0 - float(eps), 0.0)
        t_tau = torch.clamp(t_tau, max=t_max)  # (T,)

        batch_diffusion_time: torch.Tensor = t_tau.unsqueeze(0).expand(
            B, -1)  # (B, T)
        low_t_mask = torch.ones(B, dtype=torch.bool, device=device_)
    else:
        batch_diffusion_time: torch.Tensor = (torch.rand(
            B,
            device=device_,
            dtype=torch.float32,
        ) * (1 - eps) + eps)  # (B,)

        t_threshold: float = float(args.feasible_learn_noise_thresh)
        # low_t_mask: (B,)
        low_t_mask: torch.Tensor = batch_diffusion_time <= t_threshold

    # random_noise : (B, (1+)Pnn, future_len, 4) or (B, (1+)Pnn, future_len, 3)
    # **평균이 0이고 표준편차가 1인 “정규분포(가우시안)”**에서 뽑은 무작위 숫자
    random_noise: torch.Tensor = torch.randn(
        (B, one_Pnn, future_len, feature_dim),
        device=device_,
        dtype=normed_target_seq_gt.dtype,
    )
    return batch_diffusion_time, low_t_mask, random_noise


def _normalize_futures_and_build_xT(
    normed_target_cur_seq_gt: torch.Tensor,  # (B, (1+)Pnn, 4) or None
    normed_target_future_seq_gt: torch.
    Tensor,  # (B, (1+)Pnn, future_len, 4 or 3)
    target_seq_is_valid: torch.
    Tensor,  # (B, (1+)Pnn, 1+future_len or future_len)
    batch_diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
    random_noise: torch.Tensor,  # (B, (1+)Pnn, future_len, 4 or 3)
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """미래 궤적을 정규화하고, x_T 샘플과 std 를 만든다.
    return
        target_seq_norm_xT
            (B, (1+)Pnn, 1+future_len, 4) 현재 GT + 미래 x_T
            (B, (1+)Pnn, future_len, 3) 미래 x_T
        std
            (B, 1, 1, 1)
    
    """
    B, one_or_Pnn, future_len, _ = normed_target_future_seq_gt.shape
    # # marginal_prob: <diffusion_planner/model/diffusion_utils/sde.py> 의 VPSDE_linear 클래스의 메서드
    mean, std = marginal_prob(normed_target_future_seq_gt, batch_diffusion_time)
    assert std.ndim == 4, "std_raw must be (B, _, _, _)"

    # ✅ (핵심) 전체 마스크를 bool로 통일
    target_seq_is_valid_bool = _to_bool_mask(
        target_seq_is_valid)  # (B, (1+)Pnn, 1+future_len or future_len)
    # target_future_is_valid: (B, (1+)Pnn, future_len)
    target_future_is_valid = target_seq_is_valid_bool[:, :, -future_len:]

    # target_future_noise_xT: (B, (1+)Pnn, future_len, 4 or 3)
    target_future_noise_xT: torch.Tensor = mean + std * random_noise

    invalid_future = (~target_future_is_valid).unsqueeze(-1)  # (B,(1+)Pnn,T,1)
    target_future_noise_xT = target_future_noise_xT.masked_fill(
        invalid_future, 0.0)

    # target_seq_norm_xT: (B, (1+)Pnn, 1+future_len, 4) or (B, (1+)Pnn, future_len, 3)
    if normed_target_cur_seq_gt is not None:
        target_seq_norm_xT: torch.Tensor = torch.cat(
            [normed_target_cur_seq_gt, target_future_noise_xT],
            dim=2,
        )
    else:
        target_seq_norm_xT = target_future_noise_xT

    invalid_cur_future = (~target_seq_is_valid_bool).unsqueeze(
        -1)  # (B, (1+)Pnn, 1+future_len or future_len, 1)
    target_seq_norm_xT = target_seq_norm_xT.masked_fill(invalid_cur_future, 0.0)

    return target_seq_norm_xT, std


def _forward_model_with_autocast(
        model: nn.Module,
        norm_inputs: Dict[str, torch.Tensor],
        target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
        target_seq_norm_xT: torch.
    Tensor,  # (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
        batch_diffusion_time: torch.Tensor,  # (B,) or (B, future_len)
        low_t_mask: torch.Tensor,  # (B,)
        use_deepspeed: bool,
        past_seg_control_gt_3_dim: Optional[
            torch.Tensor],  # (B, (1+)Pnn, past_len, 3) or None
) -> Dict[str, torch.Tensor]:
    """모델 입력 dict 를 만들고 AMP 로 forward 를 수행한다.

    Returns:
        decoder_output: model 의 두 번째 반환값 dict.

    """
    target_future_valid = _to_bool_mask(target_future_valid)

    merged_inputs: Dict[str, torch.Tensor] = {
        **norm_inputs,
        "target_future_valid":
            target_future_valid,  # (B, (1 +) Pnn, future_len)
        "target_seq_norm_xT":
            target_seq_norm_xT,  # (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
        "diffusion_time": batch_diffusion_time,  # (B,) or (B, T)
        "low_t_mask": low_t_mask,  # (B,)
        "past_seg_control_gt_3_dim":
            past_seg_control_gt_3_dim,  # (B, (1+)Pnn, past_len, 3) or None
    }
    # merged_inputs 만들어진 직후
    # decoder_output = forward_once_and_check_bf16(
    #     model=model,
    #     merged_inputs=merged_inputs,
    #     use_deepspeed=use_deepspeed,
    #     amp_dtype=torch.bfloat16,
    #     strict_bf16=True,  # bf16 아니면 바로 에러
    # )
    is_ds_engine = hasattr(model, "backward") and hasattr(
        model, "step") and hasattr(model, "module")

    if use_deepspeed:
        assert is_ds_engine, "use_deepspeed=True 인데 model 이 DS engine 아님"
        # target_past_cur_future_valid
        _, decoder_output = model(merged_inputs)  # DS가 torch_autocast로 처리
    else:
        with torch.autocast("cuda", dtype=AMP_DTYPE):
            _, decoder_output = model(merged_inputs)
    return decoder_output  # decoder_output["x0"], ["integrated_trajectory"], ...


def extract_from_decoder(
        future_len: int,
        decoder_output: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """decoder_output 에서 미래 diffusion_output 만 꺼내고 모양을 확인한다.
        # (B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
    Args:
        decoder_output: model(...) 의 두 번째 반환 dict.
    Returns:
        diffusion_output: (B, (1+)Pnn, future_len, 4) 또는 (B, (1+)Pnn, future_len, 3)
    """
    # decoder_output["x0"]: (B, one_or_Pnn, 1+future_len, 4)
    diffusion_output: torch.Tensor = decoder_output[
        "diffusion_output"]  # (B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
    assert diffusion_output.shape[2] == future_len, f"Expected diffusion_output shape to be (B, (1+)Pnn, {future_len}, C), but got {diffusion_output.shape}"
    # "diffusion_sequence"
    """
    diffusion_trajectory
        pose_based:
            (B, (1+)Pnn, (1+T),4)
        else:
            (B, (1+)Pnn, T,3)
    """
    diffusion_trajectory: torch.Tensor = decoder_output["diffusion_trajectory"][:, :, -future_len:] # (B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
    assert diffusion_trajectory.shape[2] == future_len, f"Expected diffusion_output shape to be (B, (1+)Pnn, {future_len}, C), but got {diffusion_output.shape}"

    diffusion_output = _require_finite("decoder_output['diffusion_output']", diffusion_output)
    return diffusion_output, diffusion_trajectory


def _compute_dpm_loss(
    args: Any,
    model_type: str,
    diffusion_output: torch.Tensor,            # (B, (1+)Pnn, T, C)
    std: torch.Tensor,                         # (B, 1, 1, 1) or (B, 1, T, 1)
    random_noise: torch.Tensor,                # (B, (1+)Pnn, T, C)
    normed_target_future_seq_gt: torch.Tensor, # (B, (1+)Pnn, T, C)
) -> torch.Tensor:
    """diffusion 학습 손실을 (B,P,T) 형태로 계산한다. (손실 계산은 항상 float32)

    목적
    ----
    - x_start 모드와 v 모드 모두에서 손실 계산 정밀도를 동일하게 만들기 위해,
      diff/err 및 제곱/합(Huber 포함) 계산을 float32로 통일합니다.

    Args:
        args: 학습 설정 객체. args.use_huber_loss를 사용합니다.
        model_type:
            - "x_start": 정답 x0와의 차이로 손실 계산
            - "v": 정답 v와의 차이로 손실 계산 (v = α*ε - σ*x0)
        diffusion_output: 모델 출력. shape: (B, P, T, C)
        std: σ(t). shape: (B, 1, 1, 1) 또는 (B, 1, T, 1)
        random_noise: ε. shape: (B, P, T, C)
        normed_target_future_seq_gt: x0 정답. shape: (B, P, T, C)

    Returns:
        dpm_loss: 타임스텝별 손실. shape: (B, P, T), dtype: float32
    """
    HUBER_DELTA: float = 1.0
    use_huber: bool = bool(getattr(args, "use_huber_loss", False))

    # ----- target 선택 -----
    if model_type == "x_start":
        target: torch.Tensor = normed_target_future_seq_gt  # (B,P,T,C)
    elif model_type == "v":
        target = _get_v_target_future_seq_gt(
            normed_target_future_seq_gt=normed_target_future_seq_gt,  # (B,P,T,C)
            random_noise=random_noise,                                # (B,P,T,C)
            std=std,                                                  # (B,1,1,1) or (B,1,T,1)
            target_future_valid=None,
        )  # (B,P,T,C) float32
    elif model_type == "score":
        raise NotImplementedError("score 방식은 현재 코드에서 지원하지 않음")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ----- (핵심) 손실 계산은 항상 float32로 통일 -----
    diff_f32: torch.Tensor = diffusion_output.float() - target.float()  # (B,P,T,C)

    if use_huber:
        abs_err: torch.Tensor = diff_f32.abs()                         # (B,P,T,C)
        quad: torch.Tensor = 0.5 * diff_f32.pow(2)                     # (B,P,T,C)
        lin: torch.Tensor = HUBER_DELTA * (abs_err - 0.5 * HUBER_DELTA) # (B,P,T,C)
        huber: torch.Tensor = torch.where(abs_err <= HUBER_DELTA, quad, lin)  # (B,P,T,C)
        dpm_loss: torch.Tensor = huber.sum(dim=-1)                     # (B,P,T) float32
    else:
        dpm_loss = diff_f32.pow(2).sum(dim=-1)                         # (B,P,T) float32

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
    decoder_output: Dict[str, torch.Tensor],
    norm_target_future_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
    low_t_mask_3_ndim: torch.Tensor,  # (B, 1, 1)
    w_t: torch.Tensor,  # (1, 1, T)
    base_loss: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor],
           Optional[torch.Tensor]]:
    """통합 궤적/제어 편차 기반 보조 손실을 계산한다.

    - model_type과 무관하게(예: "x_start"/"v") 보조 손실로 사용할 수 있습니다.

    내부 계산은 float32로 수행해서 수치 불안정 가능성을 줄입니다.

    Args:
        norm_target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
        target_future_valid: ((B, (1 +) Pnn, future_len)
        low_t_mask_3_ndim: (B, 1, 1)
        w_t: (1, 1, T)

    Returns:
        integration_loss_val: 스칼라, float32
        constraint_loss_val: 스칼라, float32
        integrated_trajectory: (B, (1+)Pnn, T, 4) 또는 None
        control_constraint_diff: (B, (1+)Pnn, T, 3) 또는 None
    """
    target_future_valid = _to_bool_mask(target_future_valid)
    low_t_mask_3_ndim = _to_bool_mask(low_t_mask_3_ndim)
    valid_low = target_future_valid & low_t_mask_3_ndim  # (B, (1 +) Pnn, future_len) bool
    valid_low_f = valid_low.float()

    integrated_trajectory: Optional[torch.Tensor] = None
    control_constraint_diff: Optional[torch.Tensor] = None

    # --- L_integration ---
    if ("integrated_trajectory" in decoder_output):
        integrated_full = _require_finite(
            "decoder_output['integrated_trajectory']",
            decoder_output["integrated_trajectory"],
        )  # (B, (1+)Pnn, 1+T, 4)
        integrated_trajectory = integrated_full[:, :,
                                                1:, :]  # (B, (1+)Pnn, T, 4)

        # per_step: (B,(1+)Pnn,T) float32
        diff = (
            integrated_trajectory -
            norm_target_future_gt_4_dim).float()  # (B, (1+)Pnn, future_len, 4)
        per_step = (diff**2).sum(dim=-1)

        w_f = w_t.float()
        denom = (valid_low_f * w_f).sum().clamp_min(1e-6)
        integration_loss_val = (per_step * w_f * valid_low_f).sum() / denom
    else:
        integration_loss_val = torch.zeros((),
                                           device=base_loss.device,
                                           dtype=torch.float32)

    # --- L_constraint ---
    if "control_constraint_diff" in decoder_output:
        control_constraint_diff = _require_finite(
            "decoder_output['control_constraint_diff']",
            decoder_output["control_constraint_diff"],
        )  # (B,P,T,3)
        constraint_loss_val = _masked_weighted_mse_from_diff(
            control_constraint_diff,  # (B, (1+)Pnn, future_len, 3)
            valid_low,  # (B, (1 +) Pnn, future_len)
            w_t,  # (1, 1, future_len)
        )
    else:
        constraint_loss_val = torch.zeros((),
                                          device=base_loss.device,
                                          dtype=torch.float32)

    return integration_loss_val, constraint_loss_val, integrated_trajectory, control_constraint_diff


def _add_xy_yaw_metric_losses(
        pose_based: bool,
        loss_dict: Dict[str, Any],
        state_normalizer: StateNormalizer,
        observation_normalizer: Any,
        diffusion_trajectory: torch.Tensor,  # (B,P,T,4) or (B,P,T,3)
        normed_target_future_seq_gt: torch.Tensor,  # (B,P,T,4 or 3)
        norm_target_future_gt_4_dim: torch.
    Tensor,  # (B, (1+)Pnn, future_len, 4)
        target_future_valid: torch.Tensor,  # (B, (1 +) Pnn, future_len)
        integrated_trajectory: Optional[torch.Tensor],  # (B,P,T,4) or None
        control_constraint_diff: Optional[torch.Tensor],  # (B,P,T,3) or None
        unnorm_target_cur_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, 4)
) -> None:
    """xy / yaw 관련 보기용 지표를 loss_dict에 추가합니다."""
    with torch.no_grad():
        target_future_valid_bool = _to_bool_mask(target_future_valid).to(
            device=diffusion_trajectory.device)

        if pose_based:
            # score_denorm: (B,P,T,4)
            # target_future_valid_bool :  (B, (1 +) Pnn, future_len)
            score_denorm: torch.Tensor = state_normalizer.inverse(
                diffusion_trajectory, target_future_valid_bool)

            # target_future_gt: (B,P,T,4)
            target_future_gt: torch.Tensor = state_normalizer.inverse(
                normed_target_future_seq_gt, target_future_valid_bool)

            xy_yaw_losses = _compute_xy_yaw_losses(
                score_denorm,  # (B,P,T,4)
                target_future_gt,  # (B,P,T,4)
                target_future_valid_bool,  # (B, (1 +) Pnn, future_len)
            )
            loss_dict.update(xy_yaw_losses)

        else:
            # --- (1) control 공간(vx,vy,yaw_rate) 지표 ---
            temp_dict = {"future_seg_control_gt_3_dim": diffusion_trajectory}
            denorm_temp_dict = observation_normalizer.inverse(temp_dict)
            score_denorm = denorm_temp_dict["future_seg_control_gt_3_dim"]  # (B,P,T,3)

            temp_dict = {"future_seg_control_gt_3_dim": normed_target_future_seq_gt}
            denorm_temp_dict = observation_normalizer.inverse(temp_dict)
            target_future_ctrl_gt = denorm_temp_dict[
                "future_seg_control_gt_3_dim"]  # (B,P,T,3)

            vxy_yaw_losses = _compute_vxy_yaw_losses(
                score_denorm,  # (B,P,T,3)
                target_future_ctrl_gt,  # (B,P,T,3)
                target_future_valid_bool,  # (B, (1 +) Pnn, future_len)
            )
            loss_dict.update(vxy_yaw_losses)

            # --- (2) control을 중점 적분해서 pose로 만든 뒤 pose 기준 xy/yaw 지표 ---

            # GT pose (denorm): # (B, (1+)Pnn, future_len, 4)
            target_future_pose_gt = state_normalizer.inverse(
                norm_target_future_gt_4_dim,  # (B, (1+)Pnn, future_len, 4)
                target_future_valid_bool  # (B, (1 +) Pnn, future_len)
            )

            # pred control -> pred pose (denorm): (B,P,T,4)
            score_pose_denorm = _integrate_midpoint_controls_to_pose_denorm(
                unnorm_target_cur_gt_4_dim=
                unnorm_target_cur_gt_4_dim,  # (B, (1+)Pnn, 4)
                seg_body_control_denorm=
                score_denorm,  # (B,(1 +) Pnn,future_len,3)
                target_future_valid=
                target_future_valid_bool,  # (B, (1 +) Pnn, future_len)
                dt=0.1,
                eps=1e-6,
            )

            xy_yaw_losses = _compute_xy_yaw_losses(
                score_pose_denorm,  # (B,P,T,4)
                target_future_pose_gt,  # (B,P,T,4)
                target_future_valid_bool,  # (B, (1 +) Pnn, future_len)
            )
            loss_dict.update(xy_yaw_losses)

        # --- (3) integrated_trajectory 기준 xy/yaw 오차 ---
        if integrated_trajectory is not None:
            # (B,P,T,4) 기준으로 정렬(혹시 P가 다르면 score/integrated 기준으로 slice)
            P_int = int(integrated_trajectory.shape[1])

            norm_gt_int = norm_target_future_gt_4_dim
            if int(norm_gt_int.shape[1]) != P_int:
                norm_gt_int = norm_gt_int[:, :P_int, :, :]

            target_future_gt_4_dim: torch.Tensor = state_normalizer.inverse(
                norm_gt_int, target_future_valid_bool)
            integrated_trajectory_denorm: torch.Tensor = state_normalizer.inverse(
                integrated_trajectory, target_future_valid_bool)

            integ_xy_yaw_losses = _compute_xy_yaw_losses(
                integrated_trajectory_denorm,
                target_future_gt_4_dim,
                target_future_valid_bool,
                prefix="integration_loss",
            )
            loss_dict.update(integ_xy_yaw_losses)

        # --- (4) control_constraint_diff 물리 단위 통계 ---
        if control_constraint_diff is not None:
            temp_dict = {"future_seg_control_gt_3_dim": control_constraint_diff}
            temp_dict = observation_normalizer.inverse(temp_dict)
            constraint_diff_denorm: torch.Tensor = temp_dict[
                "future_seg_control_gt_3_dim"]  # (B,P,T,3)

            constraint_xy_yaw_losses = _compute_control_xy_yaw_diff(
                constraint_diff_denorm,
                target_future_valid_bool,
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


def _split_normed_target_seq_gt(
    pose_based: bool,
    normed_target_seq_gt: torch.
    Tensor  # (B, (1+)Pnn,  (1+future_len, 4) or (future_len, 3))
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    # normed_target_cur_seq_gt: (B, (1+)Pnn, 4) OR None
    if pose_based:
        normed_target_cur_seq_gt = normed_target_seq_gt[:, :, :1, :]
        # normed_target_future_seq_gt: (B, (1+)Pnn, future_len, 4 or 3)
        normed_target_future_seq_gt: torch.Tensor = normed_target_seq_gt[:, :,
                                                                         1:, :]
    else:
        normed_target_cur_seq_gt = None
        normed_target_future_seq_gt = normed_target_seq_gt  # (B, (1+)Pnn, future_len, 3)

    return normed_target_cur_seq_gt, normed_target_future_seq_gt


def _get_near_cur_future_gt_is_valid(
    norm_inputs: Dict[str, torch.Tensor],
    norm_outputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    # norm_near_current_4_dim: (B, Pnn, 4)
    near_future_gt_is_valid = norm_outputs[
        "near_future_gt_is_valid"]  # (B, Pnn, future_len)
    near_agents_is_valid = norm_inputs["near_agents_is_valid"]  # (B, Pnn)

    near_future_gt_is_valid = (near_future_gt_is_valid > 0)  # bool
    near_agents_is_valid = (near_agents_is_valid > 0)  # bool
    near_cur_future_gt_is_valid = torch.cat(
        (near_agents_is_valid.unsqueeze(-1), near_future_gt_is_valid),
        dim=-1)  # (B, Pnn, 1 + future_len)
    return near_cur_future_gt_is_valid


def _get_xy_yaw_metric_interval_steps(args: Any) -> int:
    """xy/yaw 보기용 지표를 몇 step마다 계산할지 정합니다.

    목적:
        - 학습에는 쓰이지 않는 지표 계산을 매 step에서 하지 않도록 줄입니다.

    규칙:
        1) args.xy_yaw_metric_interval_steps 가 있으면 그 값을 사용합니다.
            - 0 이하: 지표 계산 안 함
            - 1: 매 step 계산(기존 동작)
            - N>=2: N step마다 계산
        2) 옵션이 없으면 기본값을 사용합니다.
            - DDP면 50 step마다
            - DDP 아니면 10 step마다

    Args:
        args (Any): 학습 설정 객체

    Returns:
        int: 지표 계산 step 간격
    """
    if hasattr(args, "xy_yaw_metric_interval_steps"):
        try:
            return int(getattr(args, "xy_yaw_metric_interval_steps"))
        except Exception:
            return 0

    use_ddp = bool(getattr(args, "ddp", False))
    return 50 if use_ddp else 10


def _should_compute_xy_yaw_metrics_this_step(args: Any) -> bool:
    """현재 step에서 xy/yaw 보기용 지표를 계산할지 결정합니다.

    Args:
        args (Any):
            - args._global_update_step: 현재까지 step 수(정수)
            - args.xy_yaw_metric_interval_steps: (선택) 계산 간격

    Returns:
        bool: True면 이번 step에서 지표 계산
    """
    interval = _get_xy_yaw_metric_interval_steps(args)
    if interval <= 0:
        return False

    step_idx = int(getattr(args, "_global_update_step", 0))
    if interval == 1:
        return True

    return (step_idx % interval) == 0

from typing import Optional
import torch

def _get_v_target_future_seq_gt(
    normed_target_future_seq_gt: torch.Tensor,  # (B, (1+)Pnn, T, C)
    random_noise: torch.Tensor,                 # (B, (1+)Pnn, T, C)  == epsilon
    std: torch.Tensor,                          # (B, 1, 1, 1) or (B, 1, T, 1) == sigma(t)
    target_future_valid: Optional[torch.Tensor] = None,  # (B, (1+)Pnn, T) or None
) -> torch.Tensor:
    """v 예측 학습에 쓰는 정답(v target)을 만든다.

    이 프로젝트(VPSDE_linear)에서 noisy 값은 아래처럼 만들어집니다.

        x_t = α(t) * x_0 + σ(t) * ε

    여기서
      - x_0  : 정답(노이즈 없는) 값
      - ε    : 평균 0, 표준편차 1인 무작위 텐서 (random_noise)
      - σ(t) : std 텐서
      - α(t) : VPSDE_linear 정의로부터 α(t) = sqrt(1 - σ(t)^2)

    v 예측의 정답은 아래입니다.

        v = α(t) * ε - σ(t) * x_0

    Args:
        normed_target_future_seq_gt: 정답 미래 시퀀스. shape: (B, P, T, C)
        random_noise: x_t를 만들 때 사용한 무작위 텐서(ε). shape: (B, P, T, C)
        std: σ(t). shape: (B, 1, 1, 1) 또는 (B, 1, T, 1)
        target_future_valid: 유효 마스크. shape: (B, P, T). None이면 마스킹하지 않음.

    Returns:
        v_target: v 정답 텐서. shape: (B, P, T, C), dtype: float32
    """
    # ----- shape 체크 -----
    if normed_target_future_seq_gt.shape != random_noise.shape:
        raise ValueError(
            "normed_target_future_seq_gt and random_noise must have same shape. "
            f"got x0={tuple(normed_target_future_seq_gt.shape)}, eps={tuple(random_noise.shape)}"
        )

    if std.dim() != 4:
        raise ValueError(f"std must be 4D (B,1,1,1) or (B,1,T,1). got {tuple(std.shape)}")

    B, P, T, C = normed_target_future_seq_gt.shape  # (B, P, T, C)

    if int(std.shape[0]) != int(B) or int(std.shape[1]) != 1 or int(std.shape[3]) != 1:
        raise ValueError(
            "std shape must be (B,1,1,1) or (B,1,T,1). "
            f"got std={tuple(std.shape)}, expected B={B}"
        )
    if int(std.shape[2]) not in (1, int(T)):
        raise ValueError(
            "std time dimension must be 1 or T. "
            f"got std.shape[2]={int(std.shape[2])}, T={int(T)}"
        )

    if target_future_valid is not None:
        if target_future_valid.shape != (B, P, T):
            raise ValueError(
                "target_future_valid must have shape (B,P,T). "
                f"got {tuple(target_future_valid.shape)}, expected {(B,P,T)}"
            )

    # ----- 계산 (float32로 올려서 안정적으로) -----
    x0_f = normed_target_future_seq_gt.float()  # (B, P, T, C)
    eps_f = random_noise.float()                # (B, P, T, C)
    sigma_f = std.float()                       # (B, 1, 1, 1) or (B, 1, T, 1)

    # VPSDE_linear에서 α(t) = sqrt(1 - σ(t)^2)
    alpha_sq = (1.0 - sigma_f * sigma_f).clamp_min(0.0)  # (B,1,1,1) or (B,1,T,1)
    alpha_f = torch.sqrt(alpha_sq)         # (B,1,1,1) or (B,1,T,1)

    # v = α*ε - σ*x0
    v_target = alpha_f * eps_f - sigma_f * x0_f          # (B, P, T, C)

    # (선택) invalid는 0으로 덮어서 디버그/로그가 더 깔끔해지게
    if target_future_valid is not None:
        valid_bpt = _to_bool_mask(target_future_valid).to(device=v_target.device)  # (B,P,T)
        v_target = v_target.masked_fill(~valid_bpt.unsqueeze(-1), 0.0)             # (B,P,T,C)

    v_target = _require_finite("v_target_future_seq_gt", v_target)
    return v_target

def diffusion_loss_func(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    norm_outputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
    state_normalizer: StateNormalizer,
    loss_dict: Dict[str, Any],
    model_type: str,
    observation_normalizer: Any,
    eps: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """diffusion 학습에 쓰이는 전체 손실을 계산합니다.

    이 함수는 아래 2종류의 손실을 계산해 loss_dict에 넣습니다.

    1) 주 손실: neighbor_prediction_loss
       - model_type에 따라 목표가 달라집니다.
         - "x_start": 미래 x0(노이즈 없는 값) 예측 오차
         - "v": v 예측 오차 (v = α*ε - σ*x0)
       - 계산은 (B,P,T,C) 오차를 만든 뒤 (B,P,T)로 줄이고,
         유효 마스크/시간 가중치로 스칼라 1개로 합칩니다.

    2) 보조 손실: integration_loss, constraint_loss
       - model_type과 무관하게 계산합니다. ("x_start"/"v" 모두)
       - decoder_output에 아래 키가 존재할 때만 유효합니다.
         - "integrated_trajectory"  : 적분된 미래 궤적 (있으면 integration_loss 계산)
         - "control_constraint_diff": 제약 위반량(또는 그 차이) (있으면 constraint_loss 계산)
       - 위 키가 없으면 해당 보조 손실은 0으로 둡니다.

    참고:
        - 이 함수는 보조 손실 값을 loss_dict에 "항상 기록"합니다.
        - 보조 손실을 실제로 총 손실에 얼마나 반영할지(가중치/사용 여부)는
          학습 루프(또는 상위 로직)에서 결정하는 구조를 전제로 합니다.

    Returns:
        Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
            - loss_dict: 손실/지표가 담긴 dict (in-place 업데이트됨)
            - decoder_output: 모델 decoder 출력 dict
    """
    # norm_inputs 의 각 텐서 NaN/Inf 체크
    # norm_inputs = _sanitize_norm_inputs(norm_inputs)

    # near_cur_future_gt_is_valid : (B, Pnn, 1 + future_len)
    near_cur_future_gt_is_valid = _get_near_cur_future_gt_is_valid(
        norm_inputs, norm_outputs)
    B, one_or_Pnn, one_future_len = near_cur_future_gt_is_valid.shape
    future_len = one_future_len - 1

    # ego_cur_gt_4_dim: (B, 4)
    norm_ego_cur_gt_4_dim = norm_inputs["ego_agent_past"][:, -1, :4]
    # (B, 1) + (B, future_len) = (B, 1 + future_len)
    ego_cur_future_gt_is_valid = torch.cat(
        [
            norm_inputs["ego_agent_past_is_valid"][:, -1:],
            norm_inputs["ego_future_gt_is_valid"]
        ],
        dim=1,
    )
    # (B, future_len, 4)
    ego_future_gt_4_dim = norm_outputs["ego_future_gt_4_dim"]
    # (B, Pnn, future_len, 4)
    near_future_gt_4_dim = norm_outputs["near_future_gt_4_dim"]
    if args.do_ego_predict:
        norm_target_future_gt_4_dim = torch.cat(
            [ego_future_gt_4_dim.unsqueeze(1), near_future_gt_4_dim],
            dim=1)  # (B, (1+)Pnn, future_len, 4)
    else:
        norm_target_future_gt_4_dim = near_future_gt_4_dim

    # norm_near_current_4_dim : (B, Pnn, 4)
    norm_near_current_4_dim = norm_inputs["near_agents_past"][:, :, -1, :4]

    if args.pose_based:
        (
            normed_target_seq_gt,  # (B, (1+)Pnn, 1+future_len, 4)
            target_seq_is_valid,  # (B, (1+)Pnn, 1+future_len)
        ) = build_target_future_tensors_and_masks(
            args=args,
            norm_ego_cur_gt_4_dim=norm_ego_cur_gt_4_dim,  # (B, 4)
            normed_ego_future_gt_4_dim=ego_future_gt_4_dim,  # (B, future_len, 4)
            ego_cur_future_gt_is_valid=ego_cur_future_gt_is_valid,
            # (B, 1 + future_len)
            norm_near_current_4_dim=norm_near_current_4_dim,  # (B, Pnn, 4)
            normed_near_future_gt_4_dim=norm_outputs[
                "near_future_gt_4_dim"],  # (B, Pnn, future_len, 4)
            near_cur_future_gt_is_valid=
            near_cur_future_gt_is_valid,  # (B, Pnn, 1 + future_len)
        )
        past_seg_control_gt_3_dim = None
    else:  # velocity_based
        """
        past_future_seg_control_gt_3_dim: (B, 1+Pnn, past_len + future_len, 3)
        
        normed_target_seq_gt : (B, (1+)Pnn,  (1+future_len, 4) or (future_len, 3))
        target_seq_is_valid : (B, (1+)Pnn, future_len)
        """
        # past_seg_control_gt_3_dim: (B, 1+Pnn, past_len, 3)
        past_seg_control_gt_3_dim = norm_inputs["past_seg_control_gt_3_dim"]
        future_seg_control_gt_3_dim = norm_inputs[
            "future_seg_control_gt_3_dim"] # (B, (1+)Pnn, future_len, 3)
        past_future_seg_control_gt_3_dim = torch.cat([
            past_seg_control_gt_3_dim,
            future_seg_control_gt_3_dim,
        ],
        dim=2
        )  # (B, 1+Pnn, past_len + future_len, 3)
        assert past_future_seg_control_gt_3_dim.shape[2] == (args.time_len - 1 +
                                                             future_len)
        if not args.do_ego_predict:
            past_seg_control_gt_3_dim = past_seg_control_gt_3_dim[:, 1:, :, :]
            future_seg_control_gt_3_dim = future_seg_control_gt_3_dim[:, 1:, :, :]

        (
            normed_target_seq_gt,  # (B, (1+)Pnn,  (1+future_len, 4) or (future_len, 3))
            target_seq_is_valid,  # (B, (1+)Pnn, future_len)
        ) = build_target_future_tensors_and_masks_vel(
            args=args,
            future_seg_control_gt_3_dim=
            future_seg_control_gt_3_dim,  # (B, 1+Pnn, future_len, 3)
            ego_cur_future_gt_is_valid=
            ego_cur_future_gt_is_valid,  # (B, 1 + future_len)
            near_cur_future_gt_is_valid=
            near_cur_future_gt_is_valid,  # (B, Pnn, 1 + future_len)
        )

    # ✅ (핵심) upstream dtype 변화(0/1 float 등) 대비
    target_seq_is_valid = _to_bool_mask(target_seq_is_valid)
    """
    # diffusion time / low noise mask / random noise 샘플링
    
    # batch_diffusion_time: # (B,) or (B, future_len)
    # low_t_mask: (B,)
    # random_noise: (B, (1+)Pnn, future_len, 4 or 3)
    """
    eps_used: float = _get_diffusion_time_eps(args=args, fallback_eps=float(eps))
    (batch_diffusion_time, low_t_mask, random_noise
    ) = _sample_diffusion_time_and_noise(
        normed_target_seq_gt,  # (B, (1+)Pnn,  (1+future_len, 4) or (future_len, 3))
        eps_used,
        args,
    )
    # low_t_mask_3_ndim: (B,1,1)
    low_t_mask_3_ndim: torch.Tensor = low_t_mask.view(B, 1, 1)
    """
    # normed_target_cur_seq_gt: (B, (1+)Pnn, 4) or None
    # normed_target_future_seq_gt: (B, (1+)Pnn, future_len, 4 or 3)
    """
    (normed_target_cur_seq_gt,
     normed_target_future_seq_gt) = _split_normed_target_seq_gt(
         args.pose_based, normed_target_seq_gt
     )  # (B, (1+)Pnn,  (1+future_len, 4) or (future_len, 3))
    """
        target_seq_norm_xT: 
            (B, (1+)Pnn, 1+future_len, 4) 현재 GT + 미래 x_T
            (B, (1+)Pnn, future_len, 3) 미래 x_T
        std: (B, 1, 1, 1) 노이즈 표준편차
    """
    (target_seq_norm_xT, std) = _normalize_futures_and_build_xT(
        normed_target_cur_seq_gt,  # (B, (1+)Pnn, 4) or None
        normed_target_future_seq_gt,  # (B, (1+)Pnn, future_len, 4 or 3)
        target_seq_is_valid,  # (B, (1+)Pnn, 1+future_len or future_len)
        batch_diffusion_time,  # (B,) or (B, future_len)
        random_noise,  # (B, (1+)Pnn, future_len, 4 or 3)
        marginal_prob,
    )
    target_future_valid = target_seq_is_valid[:, :,
                                              -future_len:]  # (B, (1+)Pnn, future_len)
    # 모델 forward + decoder_output 생성
    decoder_output: Dict[str, torch.Tensor] = _forward_model_with_autocast(
        model=model,  #
        norm_inputs=norm_inputs,  #
        target_future_valid=target_future_valid,  # (B, (1 +) Pnn, future_len)
        target_seq_norm_xT=
        target_seq_norm_xT,  # (B, (1+)Pnn, (1+future_len, 4) or (future_len, 3))
        batch_diffusion_time=batch_diffusion_time,  # (B,) or (B, future_len)
        low_t_mask=low_t_mask,  # (B,)
        use_deepspeed=args.use_deepspeed,
        past_seg_control_gt_3_dim=
        past_seg_control_gt_3_dim,  # (B, (1+)Pnn, past_len, 3) or None
    )

    # diffusion_output:  # (B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
    # diffusion_trajectory : (B, (1+)Pnn, T,4) or (B, (1+)Pnn, T,3)
    diffusion_output, diffusion_trajectory = extract_from_decoder(future_len=args.future_len, decoder_output=decoder_output)

    # dpm_loss: (B, (1+)Pnn, future_len)
    dpm_loss: torch.Tensor = _compute_dpm_loss(
        args=args,
        model_type=model_type,
        diffusion_output=diffusion_output,  # (B, (1+)Pnn, future_len, 4) or (B, (1+)Pnn, future_len, 3)
        std=std,  # (B, 1, 1, 1)
        random_noise=random_noise,  # (B, (1+)Pnn, future_len, 4 or 3)
        normed_target_future_seq_gt=
        normed_target_future_seq_gt,  # (B, (1+)Pnn, future_len, 4 or 3)
    )
    if args.use_timestep_weight_loss:
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
    else:
        # 균등 가중치
        # w_t: (1, 1, future_len)
        w_t = torch.ones((1, 1, future_len),
                         device=dpm_loss.device,
                         dtype=torch.float32)

    # neighbor_prediction_loss: loss_val = (스칼라)
    loss_val: torch.Tensor = _aggregate_weighted_loss(
        per_step_loss=dpm_loss,  # (B, (1+)Pnn, future_len)
        target_future_valid=target_future_valid,  # (B, (1 +) Pnn, future_len)
        w_t=w_t,  # (1, 1, future_len)
        eps=1e-6,
    )
    loss_dict["neighbor_prediction_loss"] = loss_val

    # 통합 궤적/제어 제약 손실
    """
    integration_loss_val: 스칼라
    constraint_loss_val: 스칼라
    
    integrated_trajectory: (B, (1+)Pnn, T, 4) 또는 None
    control_constraint_diff: (B, (1+)Pnn, T, 3) 또는 None
    
=        """
    # norm_target_future_gt_4_dim  # (B, (1+)Pnn, future_len, 4)
    (integration_loss_val, constraint_loss_val, integrated_trajectory,
     control_constraint_diff) = _compute_integration_and_constraint_losses(
         decoder_output=decoder_output,
         norm_target_future_gt_4_dim=
         norm_target_future_gt_4_dim,  # (B, (1+)Pnn, future_len, 4)
         target_future_valid=
         target_future_valid,  # (B, (1 +) Pnn, future_len)
         low_t_mask_3_ndim=low_t_mask_3_ndim,  # (B,1,1)
         w_t=w_t,  # (1, 1, future_len)
         base_loss=loss_val,  # scalar
     )

    loss_dict["integration_loss"] = integration_loss_val
    loss_dict["constraint_loss"] = constraint_loss_val

    # xy/yaw 관련 보기용 지표는 매 step이 아니라 "가끔"만 계산합니다.
    if _should_compute_xy_yaw_metrics_this_step(args):

        # 현재 유효 마스크 (ego / near)
        ego_cur_gt_is_valid = _to_bool_mask(
            ego_cur_future_gt_is_valid[:, 0])  # (B,)
        near_cur_gt_is_valid = _to_bool_mask(
            near_cur_future_gt_is_valid[:, :, 0])  # (B, Pnn)

        # do_ego_predict에 따라 target_cur(=현재 포즈) 구성
        if bool(getattr(args, "do_ego_predict", True)):
            # (B, 1+Pnn, 4)
            norm_target_cur_gt_4_dim = torch.cat(
                [
                    norm_ego_cur_gt_4_dim.unsqueeze(1),
                    norm_near_current_4_dim
                ],
                dim=1,
            )
            # (B, 1+Pnn)
            target_cur_gt_is_valid = torch.cat(
                [ego_cur_gt_is_valid.unsqueeze(1), near_cur_gt_is_valid],
                dim=1,
            )
        else:
            # ego 미예측이면 neighbor만 맞춰줌
            # (B, Pnn, 4)
            norm_target_cur_gt_4_dim = norm_near_current_4_dim
            # (B, Pnn)
            target_cur_gt_is_valid = near_cur_gt_is_valid

        # unnorm_target_cur_gt_4_dim: (B, (1+)Pnn, 4) 또는 (B, Pnn, 4)
        unnorm_target_cur_gt_4_dim = state_normalizer.inverse(
            norm_target_cur_gt_4_dim,
            target_cur_gt_is_valid,
        )
        _add_xy_yaw_metric_losses(
            pose_based=args.pose_based,
            loss_dict=loss_dict,
            state_normalizer=state_normalizer,
            observation_normalizer=observation_normalizer,
            diffusion_trajectory=diffusion_trajectory,  #  # (B,(1+)Pnn,T,4) or (B, (1+)Pnn, T, 3)
            normed_target_future_seq_gt=
            normed_target_future_seq_gt,  # # (B, (1+)Pnn, future_len, 4 or 3)
            norm_target_future_gt_4_dim=
            norm_target_future_gt_4_dim,  # (B, (1+)Pnn, future_len, 4)
            target_future_valid=
            target_future_valid,  # (B, (1 +) Pnn, future_len)
            integrated_trajectory=
            integrated_trajectory,  # (B, (1+)Pnn, T, 4) 또는 None
            control_constraint_diff=
            control_constraint_diff,  # (B, (1+)Pnn, T, 3) 또는 None
            unnorm_target_cur_gt_4_dim=
            unnorm_target_cur_gt_4_dim,  # (B, (1+)Pnn, 4)
        )
    # # dpm_loss 전체가 유한값인지 마지막으로 검사
    # assert torch.isfinite(dpm_loss).all().item(), \
    #     f"loss cannot be nan, random_noise={random_noise}"

    return loss_dict, decoder_output
