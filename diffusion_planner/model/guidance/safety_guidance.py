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

def _sanitize_positions(
    pos: torch.Tensor,        # (..., 2)
    pos_valid: torch.Tensor,  # (...) bool/0/1/float
) -> torch.Tensor:
    """거리 계산 전에 위치값을 안전하게 정리합니다.

    하는 일:
        1) pos_valid=False인 칸의 좌표는 (0,0)으로 강제합니다.
        2) NaN/Inf 같은 비정상 값도 (0,0)으로 강제합니다.
        3) 출력은 float32로 통일합니다. (거리 계산 안정성)

    Args:
        pos (torch.Tensor): shape (..., 2). 위치 좌표.
        pos_valid (torch.Tensor): shape (...). 유효 여부.

    Returns:
        torch.Tensor: shape (..., 2), dtype=float32. 정리된 위치 좌표.
    """
    v = _to_bool_mask(pos_valid).to(device=pos.device)              # (...), bool
    pos_f32 = pos.to(torch.float32)                                 # (...,2), f32
    pos_f32 = torch.where(v.unsqueeze(-1), pos_f32, torch.zeros_like(pos_f32))
    pos_f32 = torch.where(torch.isfinite(pos_f32), pos_f32, torch.zeros_like(pos_f32))
    return pos_f32


def _min_distance_point_to_segments(
    pos: torch.Tensor,            # (B, N, 2) float32
    seg_p0: torch.Tensor,         # (B, M, 2) float32
    seg_p1: torch.Tensor,         # (B, M, 2) float32
    seg_valid: torch.Tensor,      # (B, M) bool
    *,
    eps: float,
    big: float = 1e6,
    max_chunk_elems: int = 20_000_000,
) -> torch.Tensor:
    """여러 선분 중 가장 가까운 거리를 계산합니다.

    주의:
        - (B,N,M) 같은 큰 중간 텐서가 한 번에 너무 커질 수 있어서,
          내부에서 선분(M)을 몇 덩어리로 나눠 계산합니다.
          (결과는 완전히 동일합니다. 메모리만 아끼는 목적입니다.)

    Args:
        pos (torch.Tensor): shape (B, N, 2), dtype=float32. 점 좌표.
        seg_p0 (torch.Tensor): shape (B, M, 2), dtype=float32. 선분 시작점.
        seg_p1 (torch.Tensor): shape (B, M, 2), dtype=float32. 선분 끝점.
        seg_valid (torch.Tensor): shape (B, M), dtype=bool. False인 선분은 무시.
        eps (float): 0으로 나누기 방지 + sqrt 안정성용 작은 값.
        big (float): 무시할 선분에 줄 아주 큰 거리값.
        max_chunk_elems (int): 내부 중간 텐서 크기 제한(대략 B*N*chunk 기준).

    Returns:
        torch.Tensor: shape (B, N), dtype=float32. 각 점의 최소 거리.
    """
    B, N, _ = pos.shape
    M = int(seg_p0.shape[1])
    if M <= 0:
        return pos.new_full((B, N), float(big))                      # (B,N)

    # v: (B,M,2)
    v = seg_p1 - seg_p0                                              # (B,M,2)
    v_len2 = (v * v).sum(dim=-1)                                     # (B,M)
    vv_safe = v_len2 + float(eps)                                    # (B,M)

    pos_norm2 = (pos * pos).sum(dim=-1)                              # (B,N)
    d_min = pos.new_full((B, N), float(big))                         # (B,N)

    BN = max(1, int(B) * int(N))
    chunk_size = int(max(1, min(M, int(max_chunk_elems) // BN)))

    for start in range(0, M, chunk_size):
        end = min(M, start + chunk_size)
        C = int(end - start)

        a = seg_p0[:, start:end, :]                                  # (B,C,2)
        v_c = v[:, start:end, :]                                     # (B,C,2)
        v_len2_c = v_len2[:, start:end]                              # (B,C)
        vv_safe_c = vv_safe[:, start:end]                            # (B,C)
        seg_valid_c = seg_valid[:, start:end]                        # (B,C)

        a_norm2 = (a * a).sum(dim=-1)                                # (B,C)

        # (B,N,C) 형태를 만들되, 불필요한 (..,2) 브로드캐스트를 줄이기 위해 bmm를 사용
        p_a = torch.bmm(pos, a.transpose(1, 2).contiguous())          # (B,N,C)
        p_v = torch.bmm(pos, v_c.transpose(1, 2).contiguous())        # (B,N,C)
        a_v = (a * v_c).sum(dim=-1)                                   # (B,C)

        dot = p_v - a_v.unsqueeze(1)                                  # (B,N,C)
        t = (dot / vv_safe_c.unsqueeze(1)).clamp(0.0, 1.0)            # (B,N,C)

        dist2_pa = pos_norm2.unsqueeze(2) + a_norm2.unsqueeze(1) - 2.0 * p_a  # (B,N,C)
        dist2 = dist2_pa - 2.0 * t * dot + (t * t) * v_len2_c.unsqueeze(1)    # (B,N,C)
        dist2 = torch.clamp(dist2, min=0.0)                           # (B,N,C)

        d = torch.sqrt(dist2 + float(eps))                            # (B,N,C)
        d = torch.where(seg_valid_c.unsqueeze(1), d, torch.full_like(d, float(big)))

        d_chunk_min = d.min(dim=2).values                             # (B,N)
        d_min = torch.minimum(d_min, d_chunk_min)                     # (B,N)

    return d_min

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

    안전장치(핵심):
        - seg_valid=False인 구간의 (vx,vy,omega)는 적분 전에 0으로 강제합니다.
        - cur_valid=False인 에이전트의 현재 pose도 0으로 강제합니다.
        - NaN/Inf는 0으로 강제합니다.

    성능 개선(핵심):
        - K스텝 적분을 for문으로 돌지 않고, 누적합(cumsum)으로 한 번에 계산합니다.
          (결과는 기존 for문과 동일합니다.)

    Returns:
        xy: (B,P,K,2)
        xy_valid: (B,P,K) bool
    """
    B, P, F = x0_flat.shape
    if F % 3 != 0:
        raise ValueError(f"pose_based=False인데 x0_flat last dim이 3의 배수가 아닙니다. F={F}")

    T_any = int(F // 3)
    if T_any < int(future_len):
        raise ValueError(f"pose_based=False인데 x0_flat이 future_len보다 짧습니다. T_any={T_any}, T={future_len}")

    x_seq = x0_flat.reshape(B, P, T_any, 3)                  # (B,P,T_any,3)
    ctrl_fut_norm = x_seq[:, :, -int(future_len):, :]        # (B,P,T,3)

    seg_valid = _to_bool_mask(future_seg_valid).to(device=x0_flat.device)  # (B,P,T)

    # unnorm control: (B,P,T,3)
    ctrl_fut = state_normalizer.inverse(
        data=ctrl_fut_norm,
        valid_mask=seg_valid,
    )

    # ✅ 무효 구간은 0으로 강제 (NaN/Inf 전파 방지)
    ctrl_fut = torch.where(seg_valid.unsqueeze(-1), ctrl_fut, torch.zeros_like(ctrl_fut))
    ctrl_fut = torch.where(torch.isfinite(ctrl_fut), ctrl_fut, torch.zeros_like(ctrl_fut))

    # 현재 pose(정규화) -> unnorm: (B,P,4)
    cur_pose_norm = target_agents_past[:, :, -1, :4]         # (B,P,4)
    cur_valid_bool = _to_bool_mask(cur_valid).to(device=x0_flat.device)  # (B,P)

    cur_pose = state_normalizer.inverse(
        data=cur_pose_norm,
        valid_mask=cur_valid_bool,
    )                                                        # (B,P,4)

    # ✅ 현재가 무효면 pose를 0으로 강제 (NaN/Inf 전파 방지)
    cur_pose = torch.where(cur_valid_bool.unsqueeze(-1), cur_pose, torch.zeros_like(cur_pose))
    cur_pose = torch.where(torch.isfinite(cur_pose), cur_pose, torch.zeros_like(cur_pose))

    cur_xy = cur_pose[:, :, 0:2]                             # (B,P,2)
    cur_yaw = torch.atan2(cur_pose[:, :, 3], cur_pose[:, :, 2])  # (B,P)

    dt = float(_get_rollout_dt_sec(config=config, model=model))
    use_body_vel = bool(getattr(config, "use_body_vel", True))

    K_use = int(min(int(K), int(future_len)))
    if K_use <= 0:
        xy = x0_flat.new_zeros((B, P, 0, 2))                 # (B,P,0,2)
        xy_valid = seg_valid[:, :, :0] & cur_valid_bool.unsqueeze(-1)  # (B,P,0)
        return xy, xy_valid

    ctrl_use = ctrl_fut[:, :, :K_use, :]                     # (B,P,K,3)
    vx = ctrl_use[:, :, :, 0]                                # (B,P,K)
    vy = ctrl_use[:, :, :, 1]                                # (B,P,K)
    om = ctrl_use[:, :, :, 2]                                # (B,P,K)

    if use_body_vel:
        # yaw_before[k] = cur_yaw + dt * sum_{j<k} om[j]
        cumsum_om = torch.cumsum(om, dim=2)                  # (B,P,K)
        yaw_before = cur_yaw.unsqueeze(-1) + float(dt) * (cumsum_om - om)  # (B,P,K)

        cos_y = torch.cos(yaw_before)                        # (B,P,K)
        sin_y = torch.sin(yaw_before)                        # (B,P,K)
        vx_w = cos_y * vx - sin_y * vy                       # (B,P,K)
        vy_w = sin_y * vx + cos_y * vy                       # (B,P,K)
    else:
        vx_w = vx
        vy_w = vy

    dxy = torch.stack([vx_w, vy_w], dim=-1)                  # (B,P,K,2)

    # xy[k] = cur_xy + dt * sum_{j<=k} dxy[j]
    xy_f32 = cur_xy.to(torch.float32).unsqueeze(2) + float(dt) * torch.cumsum(
        dxy.to(torch.float32), dim=2
    )                                                        # (B,P,K,2)

    # 기존 코드와 동일하게, 최종 xy dtype은 x0_flat dtype을 따르게 맞춥니다.
    xy = xy_f32.to(dtype=x0_flat.dtype)                      # (B,P,K,2)

    xy_valid = seg_valid[:, :, :K_use] & cur_valid_bool.unsqueeze(-1)  # (B,P,K)
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

    안전장치(핵심):
        - 거리 계산 전에, 무효 위치는 (0,0)으로 강제합니다.
        - NaN/Inf도 (0,0)으로 강제합니다.

    성능 개선(핵심):
        - K스텝을 for문으로 돌며 cdist를 K번 호출하지 않고,
          (B*K)로 묶어서 cdist를 한 번 호출합니다.

    Returns:
        torch.Tensor: shape (B,), dtype=float32
    """
    B, P, K, _ = xy.shape
    if P <= 1 or K <= 0:
        return xy.new_zeros((B,), dtype=torch.float32)

    device = xy.device

    # (B,P,K,2) float32, 무효/비정상 값 0 처리
    pos = _sanitize_positions(pos=xy, pos_valid=xy_valid)            # (B,P,K,2)
    v = _to_bool_mask(xy_valid).to(device=device)                    # (B,P,K) bool

    # cdist 한 번에 처리하기 위해 (B*K,P,2)로 묶기
    pos_bkp = pos.permute(0, 2, 1, 3).reshape(B * K, P, 2)           # (B*K,P,2)

    d = torch.cdist(pos_bkp, pos_bkp, p=2.0)                         # (B*K,P,P)
    d = torch.sqrt(d * d + float(eps))                               # (B*K,P,P)
    d = d.view(B, K, P, P)                                           # (B,K,P,P)

    r_f32 = r.to(torch.float32)                                      # (B,P)
    safe = r_f32.unsqueeze(2) + r_f32.unsqueeze(1) + float(margin_agent)  # (B,P,P)
    safe = safe.unsqueeze(1)                                         # (B,1,P,P)

    pen = safe - d                                                   # (B,K,P,P)
    soft = F.softplus(pen / float(s_agent)) * float(s_agent)         # (B,K,P,P)

    v_kp = v.permute(0, 2, 1)                                        # (B,K,P)
    pair_valid = (v_kp.unsqueeze(3) & v_kp.unsqueeze(2)).to(torch.float32)  # (B,K,P,P)

    tri_mask = torch.triu(
        torch.ones((int(P), int(P)), device=device, dtype=torch.bool),
        diagonal=1
    )                                                                # (P,P)
    tri_mask_f = tri_mask.to(torch.float32).view(1, 1, P, P)          # (1,1,P,P)

    term = soft * pair_valid * tri_mask_f                            # (B,K,P,P)
    E_k = term.sum(dim=(2, 3))                                       # (B,K)

    w = w_time.to(device=device, dtype=torch.float32).view(1, K)     # (1,K)
    E = (E_k * w).sum(dim=1)                                         # (B,)
    return E


def _compute_road_edge_energy(
    xy: torch.Tensor,            # (B,P,K,2)  "비정규화된" 좌표
    xy_valid: torch.Tensor,      # (B,P,K) bool
    is_vehicle: torch.Tensor,    # (B,P) bool
    *,
    road_edge: Optional[torch.Tensor],          # (B,Ne,S,2)  "비정규화된" 좌표
    road_edge_is_valid: Optional[torch.Tensor], # (B,Ne) or (B,Ne,S) or (B,Ne,S-1) or None
    w_time: torch.Tensor,        # (K,)
    r_veh: float,
    margin_edge: float,
    s_edge: float,
    eps: float,
) -> torch.Tensor:
    """(차량만) road_edge에 가까워지는 정도를 벌점으로 주는 E_edge를 계산합니다.

    전제(중요)
    - xy와 road_edge는 반드시 "같은 단위(비정규화된 단위)"여야 합니다.
      즉, 둘 다 원래 좌표 단위로 맞춘 뒤에 거리 계산을 합니다.

    Args:
        xy (torch.Tensor): (B,P,K,2)
        road_edge (torch.Tensor): (B,Ne,S,2)

    Returns:
        torch.Tensor: (B,) float32
    """
    B, P, K, _ = xy.shape
    if road_edge is None or int(getattr(road_edge, "numel", lambda: 0)()) == 0:
        return xy.new_zeros((B,), dtype=torch.float32)

    if (not isinstance(road_edge, torch.Tensor)) or road_edge.dim() != 4 or int(road_edge.shape[-1]) != 2:
        return xy.new_zeros((B,), dtype=torch.float32)

    device = xy.device

    # road_edge: float32 + NaN/Inf -> 0
    road_edge_f32 = road_edge.to(device=device, dtype=torch.float32)  # (B,Ne,S,2)
    road_edge_f32 = torch.where(torch.isfinite(road_edge_f32), road_edge_f32, torch.zeros_like(road_edge_f32))

    B2, Ne, S, _ = road_edge_f32.shape
    if B2 != B or int(Ne) <= 0 or int(S) <= 1:
        return xy.new_zeros((B,), dtype=torch.float32)

    # ---------- seg valid 만들기 ----------
    def _build_seg_valid(
        valid: Optional[torch.Tensor],
        *,
        B: int,
        Ne: int,
        S: int,
        device: torch.device,
    ) -> torch.Tensor:
        """road_edge_is_valid에서 선분(valid) 마스크를 만듭니다.

        Returns:
            torch.Tensor: (B, Ne*(S-1)) bool
        """
        seg_len = int(S - 1)
        if seg_len <= 0:
            return torch.zeros((B, 0), device=device, dtype=torch.bool)

        if valid is None:
            seg = torch.ones((B, Ne, seg_len), device=device, dtype=torch.bool)  # (B,Ne,S-1)
            return seg.reshape(B, Ne * seg_len)

        v = _to_bool_mask(valid).to(device=device)  # bool
        if v.dim() == 2:
            if int(v.shape[0]) != B or int(v.shape[1]) != Ne:
                raise ValueError(
                    "road_edge_is_valid shape mismatch (2D). "
                    f"expected={(B, Ne)}, got={tuple(v.shape)}"
                )
            seg = v.unsqueeze(-1).expand(B, Ne, seg_len)  # (B,Ne,S-1)
            return seg.reshape(B, Ne * seg_len)

        if v.dim() == 3:
            if int(v.shape[0]) != B or int(v.shape[1]) != Ne:
                raise ValueError(
                    "road_edge_is_valid shape mismatch (3D). "
                    f"expected B={B}, Ne={Ne}, got={tuple(v.shape)}"
                )
            if int(v.shape[2]) == int(S):
                # 점 단위 valid -> 선분 valid
                seg = v[:, :, :-1] & v[:, :, 1:]  # (B,Ne,S-1)
                return seg.reshape(B, Ne * seg_len)
            if int(v.shape[2]) == int(S - 1):
                # 이미 선분 단위 valid
                return v.reshape(B, Ne * seg_len)
            raise ValueError(
                "road_edge_is_valid last dim must be S or (S-1). "
                f"S={S}, got last_dim={int(v.shape[2])}"
            )

        raise ValueError(
            "road_edge_is_valid must be 2D or 3D (or None). "
            f"got dim={int(v.dim())}"
        )

    # 선분 끝점들
    p0 = road_edge_f32[:, :, :-1, :]                                  # (B,Ne,S-1,2)
    p1 = road_edge_f32[:, :, 1:, :]                                   # (B,Ne,S-1,2)

    M = int(Ne) * int(S - 1)
    p0f = p0.reshape(B, M, 2)                                          # (B,M,2)
    p1f = p1.reshape(B, M, 2)                                          # (B,M,2)

    seg_edge_valid = _build_seg_valid(
        road_edge_is_valid,
        B=int(B),
        Ne=int(Ne),
        S=int(S),
        device=device,
    )                                                                  # (B,M)

    is_vehicle_b = _to_bool_mask(is_vehicle).to(device=device)         # (B,P)
    if not bool(is_vehicle_b.any()):
        return xy.new_zeros((B,), dtype=torch.float32)

    # (B,P,K,2) float32, 무효/비정상 값 0 처리
    pos = _sanitize_positions(pos=xy, pos_valid=xy_valid)              # (B,P,K,2)
    v_k = _to_bool_mask(xy_valid).to(device=device)                    # (B,P,K) bool

    # 점들을 (B,N,2)로 펼쳐서 한 번에 처리 (N=P*K)
    pos_flat = pos.reshape(B, int(P) * int(K), 2)                      # (B,N,2)

    d_min_flat = _min_distance_point_to_segments(
        pos=pos_flat,                                                  # (B,N,2)
        seg_p0=p0f,                                                    # (B,M,2)
        seg_p1=p1f,                                                    # (B,M,2)
        seg_valid=seg_edge_valid,                                      # (B,M)
        eps=eps,
        big=1e6,
    )                                                                  # (B,N)

    d_min = d_min_flat.view(B, P, K)                                   # (B,P,K)

    pen = (float(r_veh) + float(margin_edge)) - d_min                  # (B,P,K)
    soft = F.softplus(pen / float(s_edge)) * float(s_edge)             # (B,P,K)

    veh_valid = (is_vehicle_b.unsqueeze(-1) & v_k).to(torch.float32)   # (B,P,K)
    E_k = (soft * veh_valid).sum(dim=1)                                # (B,K)

    w = w_time.to(device=device, dtype=torch.float32).view(1, K)       # (1,K)
    E = (E_k * w).sum(dim=1)                                           # (B,)
    return E

def _inverse_road_edge_to_unnorm(
    *,
    road_edge_norm: torch.Tensor,                     # (B, Ne, S, 2)
    road_edge_is_valid: Optional[torch.Tensor],       # (B,Ne) or (B,Ne,S) or (B,Ne,S-1)
    observation_normalizer: Any,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """정규화된 road_edge를 원래 단위(비정규화)로 되돌립니다.

    Args:
        road_edge_norm (torch.Tensor):
            road_edge 점 좌표(정규화된 값).
            shape: (B, Ne, S, 2)
        road_edge_is_valid (Optional[torch.Tensor]):
            유효 마스크.
            - shape: (B, Ne)      : 폴리라인 단위 유효
            - shape: (B, Ne, S)   : 점 단위 유효
            - shape: (B, Ne, S-1) : 선분 단위 유효
        observation_normalizer (Any):
            observation_normalizer.inverse(dict)를 제공하는 객체.

    Returns:
        Tuple[torch.Tensor, Optional[torch.Tensor]]:
            - road_edge_unnorm: 비정규화된 road_edge.
              shape: (B, Ne, S, 2)
            - road_edge_is_valid_out: 입력과 같은 의미의 valid(단, device는 road_edge_unnorm에 맞춤)
              shape: 입력과 동일
    """
    if not hasattr(observation_normalizer, "inverse"):
        raise TypeError("observation_normalizer는 inverse(...) 메서드가 있어야 합니다.")

    pack: Dict[str, Any] = {"road_edge": road_edge_norm}
    if isinstance(road_edge_is_valid, torch.Tensor):
        pack["road_edge_is_valid"] = road_edge_is_valid

    unnorm = observation_normalizer.inverse(pack)
    road_edge_unnorm = unnorm.get("road_edge", None)
    if not isinstance(road_edge_unnorm, torch.Tensor):
        raise RuntimeError("observation_normalizer.inverse(...) 결과에 'road_edge'가 없습니다.")

    if isinstance(road_edge_is_valid, torch.Tensor):
        road_edge_is_valid_out = road_edge_is_valid.to(device=road_edge_unnorm.device)
    else:
        road_edge_is_valid_out = None

    return road_edge_unnorm, road_edge_is_valid_out

def safety_guidance_fn(
    x_in: torch.Tensor,   # 여기서는 x0_flat을 받는 규약
    t_input: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """안전 점수(score)를 계산해 (B,)로 반환합니다.

    반영된 안전장치:
        - g(t)==0이면 즉시 0 반환(연결은 유지해서 grad가 0으로 나오게).
        - road_edge / road_edge_is_valid를 x0와 같은 device로 이동.
    """
    x0_from_kwargs = kwargs.get("x0_pred", None)
    if isinstance(x0_from_kwargs, torch.Tensor):
        x0_use = x0_from_kwargs
    else:
        raise ValueError("safety_guidance_fn: x0_pred가 torch.Tensor가 아닙니다.")
    if x0_use.dim() != 3:
        raise ValueError(f"safety_guidance_fn: x0_use must be (B,P,F). got {tuple(x0_use.shape)}")

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

    B = int(x0_use.shape[0])
    device = x0_use.device

    # -------------------------
    # (A) 시간 게이트 g(t) 먼저 계산 + early-exit
    # -------------------------
    t_th = float(getattr(config, "safety_t_th", 0.6))
    p_pow = float(getattr(config, "safety_gate_p", 2.0))

    t = t_input.view(B).to(device=device, dtype=torch.float32)  # (B,)
    t_th_safe = float(max(t_th, 1e-6))
    g_raw = (float(t_th) - t) / t_th_safe
    g = torch.clamp(g_raw, 0.0, 1.0) ** float(p_pow)  # (B,)

    # ✅ 전 배치에서 g(t)=0이면, 비싼 계산을 건너뛰고 0 리턴
    #    (단, grad 연결은 유지: x0_use에 0을 곱하는 형태)
    if not bool((g > 0.0).any()):
        zero = x0_use.to(torch.float32).reshape(B, -1).sum(dim=1) * 0.0  # (B,)
        return zero

    # -------------------------
    # 나머지 하이퍼
    # -------------------------
    future_len = int(getattr(config, "future_len"))
    T = int(future_len)
    K = min(int(getattr(config, "safety_k", 10)), T)

    eps = float(getattr(config, "safety_eps", 1e-6))

    r_veh = float(getattr(config, "safety_r_veh", 1.2))
    r_ped = float(getattr(config, "safety_r_ped", 0.35))
    r_bike = float(getattr(config, "safety_r_bike", 0.7))

    margin_agent = float(getattr(config, "safety_margin_agent", 0.2))
    s_agent = float(getattr(config, "safety_s_agent", 0.2))
    tau_time = float(getattr(config, "safety_tau_time", 3.0))

    margin_edge = float(getattr(config, "safety_margin_edge", 0.3))
    s_edge = float(getattr(config, "safety_s_edge", 0.1))

    lambda_agent = float(getattr(config, "safety_lambda_agent", 1.0))
    lambda_edge = float(getattr(config, "safety_lambda_edge", 2.0))

    w_time = torch.exp(
        -torch.arange(int(K), device=device, dtype=torch.float32) / float(max(tau_time, 1e-6))
    )  # (K,)

    # -------------------------
    # inputs + road_edge device 정렬
    # -------------------------
    inputs = kwargs.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    observation_normalizer = kwargs.get("observation_normalizer", None)
    if observation_normalizer is None:
        raise ValueError(
            "safety_guidance_fn: kwargs['observation_normalizer']가 없습니다.")

    # -------------------------
    # inputs + road_edge: 정규화 -> 비정규화로 되돌려서 사용
    # -------------------------
    inputs = kwargs.get("inputs", {})
    if not isinstance(inputs, dict):
        inputs = {}

    road_edge_norm = inputs.get("road_edge", None)
    road_edge_is_valid = inputs.get("road_edge_is_valid", None)

    if isinstance(road_edge_norm, torch.Tensor):
        road_edge_norm = road_edge_norm.to(device=device)
        if isinstance(road_edge_is_valid, torch.Tensor):
            road_edge_is_valid = road_edge_is_valid.to(device=device)

        # ✅ 핵심: road_edge를 observation_normalizer 기준으로 비정규화
        road_edge_unnorm, road_edge_is_valid = _inverse_road_edge_to_unnorm(
            road_edge_norm=road_edge_norm,  # (B,Ne,S,2)
            road_edge_is_valid=road_edge_is_valid,  # (B,Ne) or ...
            observation_normalizer=observation_normalizer,
        )
    else:
        road_edge_unnorm = None
        road_edge_is_valid = None


    # -------------------------
    # valid mask
    # -------------------------
    cur_valid, future_node_valid, future_seg_valid = _build_valid_masks(
        target_past_cur_future_valid=target_past_cur_future_valid,
        future_len=T,
    )

    # -------------------------
    # xy, xy_valid
    # -------------------------
    if bool(getattr(config, "pose_based", True)):
        xy, xy_valid = _build_xy_from_x0_pose_based(
            x0_flat=x0_use,
            future_len=T,
            K=K,
            future_node_valid=future_node_valid,
            state_normalizer=state_normalizer,
        )
    else:
        xy, xy_valid = _build_xy_from_x0_control_pose_free(
            x0_flat=x0_use,
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
    # E_agent / E_edge
    # -------------------------
    E_agent = _compute_agent_collision_energy(
        xy=xy,
        xy_valid=xy_valid,
        r=r,
        margin_agent=margin_agent,
        s_agent=s_agent,
        w_time=w_time,
        eps=eps,
    )

    E_edge = _compute_road_edge_energy(
        xy=xy,
        xy_valid=xy_valid,
        is_vehicle=is_vehicle,
        road_edge=road_edge_unnorm,
        road_edge_is_valid=road_edge_is_valid,
        w_time=w_time,
        r_veh=r_veh,
        margin_edge=margin_edge,
        s_edge=s_edge,
        eps=eps,
    )

    # -------------------------
    # 최종 score
    # -------------------------
    E_safety = float(lambda_agent) * E_agent + float(lambda_edge) * E_edge  # (B,)
    score = -g * E_safety  # (B,)
    return score.to(torch.float32)