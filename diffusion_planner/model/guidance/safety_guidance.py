from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


def _to_bool_mask(mask: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """0/1 또는 bool 텐서를 bool로 통일합니다.

    Args:
        mask (torch.Tensor): shape: 임의
        threshold (float): float 입력일 때 True 기준

    Returns:
        torch.Tensor: shape 동일, dtype=bool
    """
    if mask.dtype == torch.bool:
        return mask
    if torch.is_floating_point(mask):
        return mask > float(threshold)
    return mask != 0


def _get_rollout_dt_sec(config: Any, model: Any) -> float:
    """세그먼트 적분에 쓸 dt(초)를 안전한 우선순위로 고릅니다.

    우선순위:
        1) model.feasible_projector.constraints_h_params.dt
        2) config.rollout_dt / config.dt / config.step_dt
        3) 0.1

    Args:
        config (Any): config 객체
        model (Any): DiT(또는 유사) 모델 객체

    Returns:
        float: dt(sec)
    """
    fp = getattr(model, "feasible_projector", None)
    if fp is not None and hasattr(fp, "constraints_h_params"):
        try:
            dt_fp = float(fp.constraints_h_params.dt)
            if dt_fp > 0.0:
                return dt_fp
        except Exception:
            pass

    for key in ("rollout_dt", "dt", "step_dt"):
        if hasattr(config, key):
            try:
                dt_cfg = float(getattr(config, key))
                if dt_cfg > 0.0:
                    return dt_cfg
            except Exception:
                pass

    return 0.1


def _build_valid_masks(
    target_past_cur_future_valid: torch.Tensor,  # (B,P,time_len+T)
    future_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """현재/미래 노드 valid와 미래 세그먼트 valid를 만듭니다.

    Args:
        target_past_cur_future_valid (torch.Tensor):
            shape: (B, P, time_len + T), bool 또는 0/1
        future_len (int): T

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - cur_valid: (B, P) bool
            - future_node_valid: (B, P, T) bool
            - future_seg_valid: (B, P, T) bool  (노드 (k) & (k+1))
    """
    valid_all = _to_bool_mask(target_past_cur_future_valid)
    cur_future_valid = valid_all[:, :, -(1 + int(future_len)):]  # (B,P,1+T)
    cur_valid = cur_future_valid[:, :, 0]  # (B,P)
    future_node_valid = cur_future_valid[:, :, 1:]  # (B,P,T)
    future_seg_valid = cur_future_valid[:, :, :-1] & cur_future_valid[:, :, 1:]  # (B,P,T)
    return cur_valid, future_node_valid, future_seg_valid


def _build_agent_radius_and_vehicle_mask(
    target_agents_past: torch.Tensor,  # (B,P,time_len,11)
    *,
    r_veh: float,
    r_ped: float,
    r_bike: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """에이전트 반경 r과 차량 마스크를 만듭니다.

    Args:
        target_agents_past (torch.Tensor):
            shape: (B, P, time_len, 11)
        r_veh (float): 차량 반경
        r_ped (float): 보행자 반경
        r_bike (float): 자전거 반경

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - r: (B, P) float
            - is_vehicle: (B, P) bool
    """
    cur_11 = target_agents_past[:, :, -1, :]  # (B,P,11)
    type_onehot = cur_11[:, :, 8:11].to(torch.float32)  # (B,P,3)

    r_vec = torch.tensor([float(r_veh), float(r_ped), float(r_bike)],
                         device=type_onehot.device,
                         dtype=type_onehot.dtype)  # (3,)
    r = (type_onehot * r_vec.view(1, 1, 3)).sum(dim=-1)  # (B,P)
    is_vehicle = type_onehot[:, :, 0] > 0.5  # (B,P)
    return r, is_vehicle


def _build_xy_from_x0_pose_based(
    x0_flat: torch.Tensor,             # (B,P,F)
    future_len: int,
    K: int,
    future_node_valid: torch.Tensor,   # (B,P,T)
    state_normalizer: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """pose_based=True에서 x0_flat로부터 미래 xy(K스텝)와 valid를 만듭니다.

    Args:
        x0_flat (torch.Tensor): (B,P,F)
        future_len (int): T
        K (int): K
        future_node_valid (torch.Tensor): (B,P,T) bool
        state_normalizer (Any): inverse 가능 객체

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - xy: (B,P,K,2)
            - xy_valid: (B,P,K) bool
    """
    B, P, F = x0_flat.shape
    if F % 4 != 0:
        raise ValueError(f"pose_based=True인데 x0_flat last dim이 4의 배수가 아닙니다. F={F}")

    T_any = int(F // 4)
    if T_any < int(future_len):
        raise ValueError(f"pose_based=True인데 x0_flat이 future_len보다 짧습니다. T_any={T_any}, T={future_len}")

    x_seq = x0_flat.reshape(B, P, T_any, 4)                  # (B,P,T_any,4)
    pose_fut_norm = x_seq[:, :, -int(future_len):, :]        # (B,P,T,4)

    # unnorm pose: (B,P,T,4)
    pose_fut = state_normalizer.inverse(
        data=pose_fut_norm,
        valid_mask=future_node_valid,
    )

    xy = pose_fut[:, :, :int(K), 0:2]                        # (B,P,K,2)
    xy_valid = _to_bool_mask(future_node_valid[:, :, :int(K)])  # (B,P,K)
    return xy, xy_valid


def _build_xy_from_x0_control_pose_free(
    x0_flat: torch.Tensor,               # (B,P,F)
    future_len: int,
    K: int,
    *,
    target_agents_past: torch.Tensor,    # (B,P,time_len,11)
    cur_valid: torch.Tensor,             # (B,P) bool
    future_seg_valid: torch.Tensor,      # (B,P,T) bool
    state_normalizer: Any,
    config: Any,
    model: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """pose_based=False에서 x0_flat(control)로부터 K스텝 미래 xy와 valid를 만듭니다.

    방법(그대로 구현):
        - 미래 control(마지막 T)을 unnorm으로 되돌림
        - 현재 pose를 unnorm으로 되돌림
        - K스텝을 dt로 적분해서 xy 생성

    Args:
        x0_flat (torch.Tensor): (B,P,F)
        future_len (int): T
        K (int): K
        target_agents_past (torch.Tensor): (B,P,time_len,11)
        cur_valid (torch.Tensor): (B,P) bool
        future_seg_valid (torch.Tensor): (B,P,T) bool
        state_normalizer (Any): inverse 가능 객체
        config (Any): use_body_vel 등
        model (Any): dt 추출용

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - xy: (B,P,K,2)
            - xy_valid: (B,P,K) bool
    """
    B, P, F = x0_flat.shape
    if F % 3 != 0:
        raise ValueError(f"pose_based=False인데 x0_flat last dim이 3의 배수가 아닙니다. F={F}")

    T_any = int(F // 3)
    if T_any < int(future_len):
        raise ValueError(f"pose_based=False인데 x0_flat이 future_len보다 짧습니다. T_any={T_any}, T={future_len}")

    x_seq = x0_flat.reshape(B, P, T_any, 3)                  # (B,P,T_any,3)
    ctrl_fut_norm = x_seq[:, :, -int(future_len):, :]        # (B,P,T,3)

    seg_valid = _to_bool_mask(future_seg_valid)              # (B,P,T)

    # unnorm control: (B,P,T,3)
    ctrl_fut = state_normalizer.inverse(
        data=ctrl_fut_norm,
        valid_mask=seg_valid,
    )

    # 현재 pose(정규화) -> unnorm: (B,P,4)
    cur_pose_norm = target_agents_past[:, :, -1, :4]         # (B,P,4)
    cur_pose = state_normalizer.inverse(
        data=cur_pose_norm,
        valid_mask=_to_bool_mask(cur_valid),
    )                                                        # (B,P,4)

    cur_xy = cur_pose[:, :, 0:2]                             # (B,P,2)

    # 전제: (cos, sin) = (채널2, 채널3)
    cur_yaw = torch.atan2(cur_pose[:, :, 3], cur_pose[:, :, 2])  # (B,P)

    dt = float(_get_rollout_dt_sec(config=config, model=model))
    use_body_vel = bool(getattr(config, "use_body_vel", True))

    xy = x0_flat.new_zeros((B, P, int(K), 2))                # (B,P,K,2)

    xy_k = cur_xy                                             # (B,P,2)
    yaw_k = cur_yaw                                           # (B,P)

    for k in range(int(K)):
        vx = ctrl_fut[:, :, k, 0]                            # (B,P)
        vy = ctrl_fut[:, :, k, 1]                            # (B,P)
        om = ctrl_fut[:, :, k, 2]                            # (B,P)

        if use_body_vel:
            cos_y = torch.cos(yaw_k)                         # (B,P)
            sin_y = torch.sin(yaw_k)                         # (B,P)
            vx_w = cos_y * vx - sin_y * vy
            vy_w = sin_y * vx + cos_y * vy
        else:
            vx_w = vx
            vy_w = vy

        dxy = torch.stack([vx_w, vy_w], dim=-1)              # (B,P,2)
        xy_k1 = xy_k + float(dt) * dxy                       # (B,P,2)
        xy[:, :, k, :] = xy_k1

        yaw_k = yaw_k + float(dt) * om
        xy_k = xy_k1

    xy_valid = _to_bool_mask(seg_valid[:, :, :int(K)]) & _to_bool_mask(cur_valid).unsqueeze(-1)  # (B,P,K)
    return xy, xy_valid


def _compute_agent_collision_energy(
    xy: torch.Tensor,          # (B,P,K,2)
    xy_valid: torch.Tensor,    # (B,P,K) bool
    r: torch.Tensor,           # (B,P)
    *,
    margin_agent: float,
    s_agent: float,
    w_time: torch.Tensor,      # (K,)
    eps: float,
) -> torch.Tensor:
    """에이전트-에이전트 충돌 에너지 E_agent를 계산합니다.

    구현은 (B,P,P,K,2) 같은 큰 텐서를 만들지 않기 위해,
    K 루프를 돌면서 (B,P,P) 단위로 누적합니다.

    Args:
        xy: (B,P,K,2)
        xy_valid: (B,P,K) bool
        r: (B,P)
        margin_agent: float
        s_agent: float
        w_time: (K,)
        eps: float

    Returns:
        torch.Tensor: (B,) float32
    """
    B, P, K, _ = xy.shape
    if P <= 1 or K <= 0:
        return xy.new_zeros((B,), dtype=torch.float32)

    device = xy.device
    tri_mask = torch.triu(torch.ones((int(P), int(P)), device=device, dtype=torch.bool), diagonal=1)  # (P,P)
    tri_mask_f = tri_mask.to(dtype=torch.float32).unsqueeze(0)  # (1,P,P)

    E = xy.new_zeros((B,), dtype=torch.float32)

    r_f32 = r.to(torch.float32)  # (B,P)
    for k in range(int(K)):
        pos = xy[:, :, k, :].to(torch.float32)               # (B,P,2)
        v_k = _to_bool_mask(xy_valid[:, :, k])               # (B,P)

        # (B,P,P)
        d = torch.cdist(pos, pos, p=2.0)
        d = torch.sqrt(d * d + float(eps))  # 안전(0 근처)

        safe = r_f32.unsqueeze(2) + r_f32.unsqueeze(1) + float(margin_agent)  # (B,P,P)
        pen = safe - d  # (B,P,P)

        soft = F.softplus(pen / float(s_agent)) * float(s_agent)  # (B,P,P)

        pair_valid = (v_k.unsqueeze(2) & v_k.unsqueeze(1)).to(torch.float32)  # (B,P,P)
        term = soft * pair_valid * tri_mask_f  # (B,P,P)

        E = E + term.sum(dim=(1, 2)) * float(w_time[k])

    return E


def _compute_road_edge_energy(
    xy: torch.Tensor,            # (B,P,K,2)
    xy_valid: torch.Tensor,      # (B,P,K) bool
    is_vehicle: torch.Tensor,    # (B,P) bool
    *,
    road_edge: Optional[torch.Tensor],          # (B,Ne,S,2) or None
    road_edge_is_valid: Optional[torch.Tensor], # (B,Ne) or None
    w_time: torch.Tensor,        # (K,)
    r_veh: float,
    margin_edge: float,
    s_edge: float,
    eps: float,
) -> torch.Tensor:
    """(차량만) road_edge에 가까워지는 정도를 벌점으로 주는 E_edge를 계산합니다.

    Args:
        xy: (B,P,K,2)
        xy_valid: (B,P,K) bool
        is_vehicle: (B,P) bool
        road_edge: (B,Ne,S,2) or None
        road_edge_is_valid: (B,Ne) or None
        w_time: (K,)
        r_veh: float
        margin_edge: float
        s_edge: float
        eps: float

    Returns:
        torch.Tensor: (B,) float32
    """
    B, P, K, _ = xy.shape
    if road_edge is None or int(road_edge.numel()) == 0:
        return xy.new_zeros((B,), dtype=torch.float32)

    if road_edge.dim() != 4 or int(road_edge.shape[-1]) != 2:
        return xy.new_zeros((B,), dtype=torch.float32)

    B2, Ne, S, _ = road_edge.shape
    if B2 != B or Ne <= 0 or S <= 1:
        return xy.new_zeros((B,), dtype=torch.float32)

    device = road_edge.device
    road_edge_f32 = road_edge.to(torch.float32)  # (B,Ne,S,2)

    p0 = road_edge_f32[:, :, :-1, :]  # (B,Ne,S-1,2)
    p1 = road_edge_f32[:, :, 1:, :]   # (B,Ne,S-1,2)

    M = int(Ne) * int(S - 1)
    p0f = p0.reshape(B, M, 2)         # (B,M,2)
    p1f = p1.reshape(B, M, 2)         # (B,M,2)

    v = p1f - p0f                      # (B,M,2)
    vv = (v * v).sum(dim=-1) + float(eps)  # (B,M)

    if road_edge_is_valid is None:
        seg_edge_valid = torch.ones((B, M), device=device, dtype=torch.bool)
    else:
        rev = _to_bool_mask(road_edge_is_valid).to(device=device)   # (B,Ne)
        seg_edge_valid = rev.unsqueeze(-1).expand(B, Ne, S - 1).reshape(B, M)  # (B,M)

    big = 1e6
    E = xy.new_zeros((B,), dtype=torch.float32)

    is_vehicle_b = _to_bool_mask(is_vehicle).to(device=xy.device)  # (B,P)
    for k in range(int(K)):
        pos = xy[:, :, k, :].to(torch.float32)          # (B,P,2)
        v_k = _to_bool_mask(xy_valid[:, :, k])          # (B,P)

        # (B,P,M,2)
        w = pos.unsqueeze(2) - p0f.unsqueeze(1)

        # (B,P,M)
        dot = (w * v.unsqueeze(1)).sum(dim=-1)
        t = (dot / vv.unsqueeze(1)).clamp(0.0, 1.0)

        # (B,P,M,2)
        closest = p0f.unsqueeze(1) + t.unsqueeze(-1) * v.unsqueeze(1)

        # (B,P,M)
        d = torch.sqrt(((pos.unsqueeze(2) - closest) ** 2).sum(dim=-1) + float(eps))
        d = torch.where(seg_edge_valid.unsqueeze(1), d, torch.full_like(d, float(big)))

        d_min = d.min(dim=2).values  # (B,P)

        pen = (float(r_veh) + float(margin_edge)) - d_min  # (B,P)
        soft = F.softplus(pen / float(s_edge)) * float(s_edge)  # (B,P)

        veh_valid = (is_vehicle_b & v_k).to(torch.float32)  # (B,P)
        E_k = (soft * veh_valid).sum(dim=1)                 # (B,)
        E = E + E_k * float(w_time[k])

    return E


def safety_guidance_fn(
    x_in: torch.Tensor,   # 여기서는 x0_flat을 받는 규약
    t_input: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """안전 점수(score)를 계산해 (B,)로 반환합니다.

    입력 규약(중요)
    - x_in은 x0_pred(flat)로 들어온다고 가정합니다.
      shape: (B, P, F)

    score 정의
    \[
    score(b) = - g(t_b)\cdot(\lambda_{agent}E_{agent}(b) + \lambda_{edge}E_{edge}(b))
    \]

    Args:
        x_in (torch.Tensor): x0_flat, shape (B,P,F)
        t_input (torch.Tensor): (B,)
        cond (Optional[torch.Tensor]): 사용 안 함
        **kwargs (Any):
            - model_condition:
                - target_agents_past: (B,P,time_len,11)
                - target_past_cur_future_valid: (B,P,time_len+T) bool
            - inputs:
                - road_edge: (B,Ne,S,2) or None
                - road_edge_is_valid: (B,Ne) or None
            - state_normalizer
            - config
            - model

    Returns:
        torch.Tensor: score, shape (B,), float32
    """
    if x_in.dim() != 3:
        raise ValueError(f"safety_guidance_fn: x_in must be (B,P,F). got {tuple(x_in.shape)}")

    model_condition = kwargs.get("model_condition", None)
    if not isinstance(model_condition, dict):
        raise ValueError("safety_guidance_fn: kwargs['model_condition']가 dict가 아닙니다.")

    target_agents_past = model_condition.get("target_agents_past", None)
    target_past_cur_future_valid = model_condition.get("target_past_cur_future_valid", None)

    if not isinstance(target_agents_past, torch.Tensor):
        raise ValueError("safety_guidance_fn: model_condition['target_agents_past']가 없습니다.")
    if not isinstance(target_past_cur_future_valid, torch.Tensor):
        raise ValueError("safety_guidance_fn: model_condition['target_past_cur_future_valid']가 없습니다.")

    config = kwargs.get("config", None)
    model = kwargs.get("model", None)
    state_normalizer = kwargs.get("state_normalizer", None)
    if state_normalizer is None:
        raise ValueError("safety_guidance_fn: kwargs['state_normalizer']가 없습니다.")
    if model is None:
        raise ValueError("safety_guidance_fn: kwargs['model']가 없습니다.")
    if config is None:
        raise ValueError("safety_guidance_fn: kwargs['config']가 없습니다.")

    inputs = kwargs.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    # -------------------------
    # 하이퍼 (기본값은 설계도 그대로)
    # -------------------------
    future_len = int(getattr(config, "future_len"))
    T = int(future_len)
    K = min(int(getattr(config, "safety_K", 10)), T)

    eps = float(getattr(config, "safety_eps", 1e-6))

    r_veh = float(getattr(config, "safety_r_veh", 1.2))
    r_ped = float(getattr(config, "safety_r_ped", 0.35))
    r_bike = float(getattr(config, "safety_r_bike", 0.7))

    margin_agent = float(getattr(config, "safety_margin_agent", 0.2))
    s_agent = float(getattr(config, "safety_s_agent", 0.2))
    tau_time = float(getattr(config, "safety_tau_time", 3.0))

    margin_edge = float(getattr(config, "safety_margin_edge", 0.3))
    s_edge = float(getattr(config, "safety_s_edge", 0.1))

    t_th = float(getattr(config, "safety_t_th", 0.6))
    p_pow = float(getattr(config, "safety_gate_p", 2.0))

    lambda_agent = float(getattr(config, "safety_lambda_agent", 1.0))
    lambda_edge = float(getattr(config, "safety_lambda_edge", 2.0))

    # 시간 가중치 w_time: (K,)
    w_time = torch.exp(
        -torch.arange(int(K), device=x_in.device, dtype=torch.float32) / float(max(tau_time, 1e-6))
    )  # (K,)

    # -------------------------
    # valid mask 준비
    # -------------------------
    cur_valid, future_node_valid, future_seg_valid = _build_valid_masks(
        target_past_cur_future_valid=target_past_cur_future_valid,
        future_len=T,
    )

    # -------------------------
    # xy, xy_valid 만들기
    # -------------------------
    if bool(getattr(config, "pose_based", True)):
        xy, xy_valid = _build_xy_from_x0_pose_based(
            x0_flat=x_in,
            future_len=T,
            K=K,
            future_node_valid=future_node_valid,
            state_normalizer=state_normalizer,
        )
    else:
        xy, xy_valid = _build_xy_from_x0_control_pose_free(
            x0_flat=x_in,
            future_len=T,
            K=K,
            target_agents_past=target_agents_past,
            cur_valid=cur_valid,
            future_seg_valid=future_seg_valid,
            state_normalizer=state_normalizer,
            config=config,
            model=model,
        )

    # -------------------------
    # r, is_vehicle
    # -------------------------
    r, is_vehicle = _build_agent_radius_and_vehicle_mask(
        target_agents_past=target_agents_past,
        r_veh=r_veh,
        r_ped=r_ped,
        r_bike=r_bike,
    )

    # -------------------------
    # E_agent
    # -------------------------
    E_agent = _compute_agent_collision_energy(
        xy=xy,                      # (B,P,K,2)
        xy_valid=xy_valid,          # (B,P,K)
        r=r,                        # (B,P)
        margin_agent=margin_agent,
        s_agent=s_agent,
        w_time=w_time,              # (K,)
        eps=eps,
    )  # (B,)

    # -------------------------
    # E_edge (차량만)
    # -------------------------
    road_edge = inputs.get("road_edge", None)
    road_edge_is_valid = inputs.get("road_edge_is_valid", None)

    E_edge = _compute_road_edge_energy(
        xy=xy,
        xy_valid=xy_valid,
        is_vehicle=is_vehicle,
        road_edge=road_edge,
        road_edge_is_valid=road_edge_is_valid,
        w_time=w_time,
        r_veh=r_veh,
        margin_edge=margin_edge,
        s_edge=s_edge,
        eps=eps,
    )  # (B,)

    # -------------------------
    # 시간 게이트 g(t)
    # -------------------------
    B = int(x_in.shape[0])
    t = t_input.view(B).to(device=x_in.device, dtype=torch.float32)  # (B,)

    t_th_safe = float(max(t_th, 1e-6))
    g_raw = (float(t_th) - t) / t_th_safe
    g = torch.clamp(g_raw, 0.0, 1.0) ** float(p_pow)  # (B,)

    # -------------------------
    # 최종 score
    # -------------------------
    E_safety = float(lambda_agent) * E_agent + float(lambda_edge) * E_edge  # (B,)
    score = -g * E_safety  # (B,)
    return score.to(torch.float32)