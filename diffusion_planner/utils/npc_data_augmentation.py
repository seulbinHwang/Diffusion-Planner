from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import os
from typing import Any, Dict, Optional, Tuple, List
Tensor = torch.Tensor
from typing import Dict, Tuple

_TRIU_PAIR_IDX_CACHE: Dict[Tuple[torch.device, int], Tuple[Tensor, Tensor]] = {}


def _get_triu_pair_indices(device: torch.device, A: int) -> Tuple[Tensor, Tensor]:
    """i<j인 (i,j) pair 인덱스를 캐시로 가져옵니다.

    Args:
        device (torch.device): 반환 텐서가 올라갈 디바이스.
        A (int): agent 수.

    Returns:
        Tuple[Tensor, Tensor]:
            - i_idx: shape (P,) long. i<j인 i 인덱스
            - j_idx: shape (P,) long. i<j인 j 인덱스
            여기서 P = A*(A-1)//2
    """
    key = (device, int(A))
    cached = _TRIU_PAIR_IDX_CACHE.get(key, None)
    if cached is not None:
        return cached

    ij = torch.triu_indices(int(A), int(A), offset=1, device=device)  # (2,P)
    i_idx = ij[0]
    j_idx = ij[1]
    _TRIU_PAIR_IDX_CACHE[key] = (i_idx, j_idx)
    return i_idx, j_idx


def _obb_intersects_others_vs_others_augmask_fast(
    oth_xy: Tensor,       # (B, A, 2)
    oth_cs: Tensor,       # (B, A, 2) = (cos, sin)
    oth_wl: Tensor,       # (B, A, 2) = (width, length)
    oth_valid: Tensor,    # (B, A) bool
    aug_mask: Tensor,     # (B, A) bool  (증강된 neighbor 표시)
    eps: float = 1e-6,
) -> Tensor:
    """증강된 agent가 포함된 pair만 뽑아서(희소하게) SAT 충돌을 검사합니다.

    핵심:
        - 기존 구현은 (B,A,A) 전체 SAT를 다 계산한 뒤 pair_mask로 걸러서 낭비가 큽니다.
        - 여기서는 i<j pair 인덱스를 만든 다음,
          (aug_mask[i] or aug_mask[j]) 인 pair만 "리스트로 뽑아서" 그 pair에 대해서만 SAT를 계산합니다.
        - 결과(어떤 pair라도 겹치면 True)는 기존과 동일합니다.

    Args:
        oth_xy (Tensor): shape (B, A, 2)
        oth_cs (Tensor): shape (B, A, 2)
        oth_wl (Tensor): shape (B, A, 2)
        oth_valid (Tensor): shape (B, A) bool(또는 0/1)
        aug_mask (Tensor): shape (B, A) bool(또는 0/1)
        eps (float): 수치 안정용.

    Returns:
        Tensor: shape (B,) bool.
            - True면 그 샘플에서 "증강된 neighbor가 포함된 어떤 pair"라도 겹칩니다.
    """
    if oth_xy.ndim != 3 or int(oth_xy.shape[-1]) != 2:
        raise ValueError(f"oth_xy must be (B,A,2). got shape={tuple(oth_xy.shape)}")
    if oth_cs.ndim != 3 or int(oth_cs.shape[-1]) != 2:
        raise ValueError(f"oth_cs must be (B,A,2). got shape={tuple(oth_cs.shape)}")
    if oth_wl.ndim != 3 or int(oth_wl.shape[-1]) != 2:
        raise ValueError(f"oth_wl must be (B,A,2). got shape={tuple(oth_wl.shape)}")
    if oth_valid.ndim != 2:
        raise ValueError(f"oth_valid must be (B,A). got shape={tuple(oth_valid.shape)}")
    if aug_mask.ndim != 2:
        raise ValueError(f"aug_mask must be (B,A). got shape={tuple(aug_mask.shape)}")

    B = int(oth_xy.shape[0])
    A = int(oth_xy.shape[1])
    if A <= 1:
        return torch.zeros((B,), device=oth_xy.device, dtype=torch.bool)

    if int(oth_cs.shape[0]) != B or int(oth_cs.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_cs")
    if int(oth_wl.shape[0]) != B or int(oth_wl.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_wl")
    if int(oth_valid.shape[0]) != B or int(oth_valid.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_valid")
    if int(aug_mask.shape[0]) != B or int(aug_mask.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: aug_mask")

    device = oth_xy.device
    vmask = _as_bool_mask(oth_valid)          # (B,A)
    amask = _as_bool_mask(aug_mask)          # (B,A)

    # (i,j) i<j pair 인덱스 (P,)
    i_idx, j_idx = _get_triu_pair_indices(device, A)

    # (B,P): "증강 포함 pair"만 True
    pair_has_aug = amask[:, i_idx] | amask[:, j_idx]
    # (B,P): 둘 다 valid인 pair만 True
    pair_valid = vmask[:, i_idx] & vmask[:, j_idx]
    pair_keep = pair_has_aug & pair_valid

    sel = torch.nonzero(pair_keep, as_tuple=False)  # (M,2) where [:,0]=b, [:,1]=p
    if int(sel.numel()) == 0:
        return torch.zeros((B,), device=device, dtype=torch.bool)

    b_idx = sel[:, 0]  # (M,)
    p_idx = sel[:, 1]  # (M,)

    ii = i_idx.index_select(0, p_idx)  # (M,)
    jj = j_idx.index_select(0, p_idx)  # (M,)

    # (M,2)로 pair의 두 박스만 뽑아서 SAT 계산 (완전 동일한 SAT 로직)
    a_xy = oth_xy[b_idx, ii]  # (M,2)
    b_xy = oth_xy[b_idx, jj]  # (M,2)

    a_cs = oth_cs[b_idx, ii]  # (M,2)
    b_cs = oth_cs[b_idx, jj]  # (M,2)

    a_wl = oth_wl[b_idx, ii]  # (M,2)
    b_wl = oth_wl[b_idx, jj]  # (M,2)

    a_cos = a_cs[:, 0]
    a_sin = a_cs[:, 1]
    b_cos = b_cs[:, 0]
    b_sin = b_cs[:, 1]

    aW = 0.5 * a_wl[:, 0]
    aL = 0.5 * a_wl[:, 1]
    bW = 0.5 * b_wl[:, 0]
    bL = 0.5 * b_wl[:, 1]

    tx = b_xy[:, 0] - a_xy[:, 0]
    ty = b_xy[:, 1] - a_xy[:, 1]

    # t in A frame
    tAx = tx * a_cos + ty * a_sin
    tAy = -tx * a_sin + ty * a_cos

    # Rotation matrix between A and B
    R00 = a_cos * b_cos + a_sin * b_sin
    R01 = a_cos * (-b_sin) + a_sin * b_cos
    R10 = (-a_sin) * b_cos + a_cos * b_sin
    R11 = (-a_sin) * (-b_sin) + a_cos * b_cos

    eps_f = float(eps)
    absR00 = torch.abs(R00) + eps_f
    absR01 = torch.abs(R01) + eps_f
    absR10 = torch.abs(R10) + eps_f
    absR11 = torch.abs(R11) + eps_f

    cond0 = torch.abs(tAx) <= (aL + bL * absR00 + bW * absR01)
    cond1 = torch.abs(tAy) <= (aW + bL * absR10 + bW * absR11)

    # t in B frame: tB = R^T * tA
    tBx = tAx * R00 + tAy * R10
    tBy = tAx * R01 + tAy * R11

    cond2 = torch.abs(tBx) <= (bL + aL * absR00 + aW * absR10)
    cond3 = torch.abs(tBy) <= (bW + aL * absR01 + aW * absR11)

    hit = cond0 & cond1 & cond2 & cond3  # (M,)

    # (B,)로 “어느 pair라도 hit면 True” 만들기
    out_i32 = torch.zeros((B,), device=device, dtype=torch.int32)
    out_i32.scatter_reduce_(0, b_idx, hit.to(torch.int32), reduce="amax", include_self=True)
    return out_i32.to(torch.bool)

def _as_bool_mask(mask: Tensor) -> Tensor:
    """마스크를 bool로 통일합니다.

    Args:
        mask (Tensor): shape: (...)

    Returns:
        Tensor: bool 마스크. shape: mask.shape
    """
    return mask if mask.dtype == torch.bool else (mask > 0)


def _wrap_to_pi(delta: Tensor) -> Tensor:
    """각도 차이를 [-pi, pi] 범위로 접습니다.

    Args:
        delta (Tensor): shape: (...)

    Returns:
        Tensor: shape: delta.shape
    """
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _normalize_cos_sin(
    cos_v: Tensor,
    sin_v: Tensor,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """(cos, sin)을 길이 1로 정규화합니다.

    Args:
        cos_v (Tensor): shape: (...)
        sin_v (Tensor): shape: (...)
        eps (float): 0 나눔 방지용 작은 값.

    Returns:
        Tuple[Tensor, Tensor]:
            - cos_n: shape (...)
            - sin_n: shape (...)
    """
    # sqrt + division 대신 rsqrt + multiply로 바꿔서 더 가볍게 처리
    inv_r = torch.rsqrt(cos_v * cos_v + sin_v * sin_v + float(eps))
    return cos_v * inv_r, sin_v * inv_r


def _rot_vec_by_yaw(vx: Tensor, vy: Tensor, cos_y: Tensor,
                    sin_y: Tensor) -> Tuple[Tensor, Tensor]:
    """R(yaw)로 (vx,vy)를 회전합니다.

    Args:
        vx (Tensor): shape: (...)
        vy (Tensor): shape: (...)
        cos_y (Tensor): shape: (...)  cos(yaw)
        sin_y (Tensor): shape: (...)  sin(yaw)

    Returns:
        Tuple[Tensor, Tensor]:
            - rx: shape (...)
            - ry: shape (...)
    """
    rx = cos_y * vx - sin_y * vy
    ry = sin_y * vx + cos_y * vy
    return rx, ry


def _rot_vec_by_minus_yaw(vx: Tensor, vy: Tensor, cos_y: Tensor,
                          sin_y: Tensor) -> Tuple[Tensor, Tensor]:
    """R(-yaw)로 (vx,vy)를 회전합니다.

    Args:
        vx (Tensor): shape: (...)
        vy (Tensor): shape: (...)
        cos_y (Tensor): shape: (...) cos(yaw)
        sin_y (Tensor): shape: (...) sin(yaw)

    Returns:
        Tuple[Tensor, Tensor]:
            - rx: shape (...)
            - ry: shape (...)
    """
    rx = cos_y * vx + sin_y * vy
    ry = -sin_y * vx + cos_y * vy
    return rx, ry

def _compute_acc_at_t_from_vxvy(vxvy: Tensor, valid: Tensor, dt: float, t: int) -> Tensor:
    """속도(vx,vy)로 특정 시점 t의 가속도(ax,ay)만 계산합니다.

    기존 _compute_acc_from_vxvy는 (..., T, 2) 전체 가속도 시퀀스를 만들지만,
    이 함수는 t 한 시점만 계산해서 불필요한 큰 텐서 생성/연산을 없앱니다.
    계산 규칙은 완전히 동일합니다.

    Args:
        vxvy (Tensor):
            shape: (..., T, 2)
        valid (Tensor):
            shape: (..., T) bool
        dt (float):
            시간 간격
        t (int):
            가속도를 뽑을 시간 인덱스 (0..T-1)

    Returns:
        Tensor:
            acc_t shape: (..., 2)
            - t<=0 이거나 valid[t] & valid[t-1]가 False면 0
            - 그 외는 (v[t] - v[t-1]) / dt
    """
    if int(vxvy.shape[-1]) != 2:
        raise ValueError(f"vxvy last dim must be 2. got shape={tuple(vxvy.shape)}")
    if vxvy.ndim != valid.ndim + 1:
        raise ValueError(
            f"vxvy ndims must be valid.ndim+1. vxvy={vxvy.ndim}, valid={valid.ndim}"
        )

    T = int(vxvy.shape[-2])
    if int(t) <= 0 or int(t) >= T:
        return torch.zeros((*vxvy.shape[:-2], 2), device=vxvy.device, dtype=vxvy.dtype)

    dv = vxvy[..., int(t), :] - vxvy[..., int(t) - 1, :]  # (..., 2)
    dv.mul_(1.0 / float(dt))

    seg_valid = valid[..., int(t)] & valid[..., int(t) - 1]  # (...,)
    dv = torch.where(seg_valid.unsqueeze(-1), dv, torch.zeros_like(dv))
    return dv

def _compute_acc_from_vxvy(vxvy: Tensor, valid: Tensor, dt: float) -> Tensor:
    """속도(vx,vy) 차분으로 가속도(ax,ay)를 만듭니다.

    규칙:
      - acc[t] = (v[t] - v[t-1]) / dt  (t>=1)
      - valid[t] & valid[t-1]일 때만 계산
      - 그 외는 0

    Args:
        vxvy (Tensor):
            shape: (..., T, 2)
        valid (Tensor):
            shape: (..., T) bool
        dt (float):
            시간 간격.

    Returns:
        Tensor:
            acc shape: (..., T, 2)
    """
    if int(vxvy.shape[-1]) != 2:
        raise ValueError(
            f"vxvy last dim must be 2. got shape={tuple(vxvy.shape)}")
    if vxvy.ndim != valid.ndim + 1:
        raise ValueError(
            f"vxvy ndims must be valid.ndim+1. vxvy={vxvy.ndim}, valid={valid.ndim}"
        )

    T = int(vxvy.shape[-2])
    acc = torch.zeros_like(vxvy)

    if T <= 1:
        return acc

    dv = vxvy[..., 1:, :] - vxvy[..., :-1, :]  # (..., T-1, 2)
    dv.mul_(1.0 / float(dt))

    # 일단 전체를 채우고, invalid 구간만 0으로 덮어씀 (결과는 동일)
    acc[..., 1:, :].copy_(dv)

    seg_valid = valid[..., 1:] & valid[..., :-1]  # (..., T-1)
    acc[..., 1:, :].masked_fill_(~seg_valid.unsqueeze(-1), 0.0)
    return acc

_UPPER_TRI_MASK_CACHE: Dict[Tuple[torch.device, int], Tensor] = {}

def _get_upper_tri_mask(device: torch.device, A: int) -> Tensor:
    """(A,A)에서 i<j만 True인 마스크를 캐시로 가져옵니다.

    Args:
        device (torch.device): 마스크가 올라갈 디바이스.
        A (int): agent 수.

    Returns:
        Tensor: shape (A, A) bool. i<j 위치만 True.
    """
    key = (device, int(A))
    m = _UPPER_TRI_MASK_CACHE.get(key, None)
    if m is None:
        m = torch.triu(
            torch.ones((int(A), int(A)), device=device, dtype=torch.bool),
            diagonal=1,
        )
        _UPPER_TRI_MASK_CACHE[key] = m
    return m

def _obb_intersects_others_vs_others(
    oth_xy: Tensor,  # (B, A, 2)
    oth_cs: Tensor,  # (B, A, 2) = (cos, sin)
    oth_wl: Tensor,  # (B, A, 2) = (width, length)
    oth_valid: Tensor,  # (B, A) bool
    pair_mask: Optional[Tensor] = None,  # (B, A, A) bool
    eps: float = 1e-6,
) -> Tensor:
    """다른 agent들끼리 현재 시점에서 겹치는지 검사합니다.

    Args:
        oth_xy (Tensor): shape (B, A, 2). agent 중심 위치 (x,y).
        oth_cs (Tensor): shape (B, A, 2). agent 방향 (cos,sin).
        oth_wl (Tensor): shape (B, A, 2). (width,length).
        oth_valid (Tensor): shape (B, A) bool. 유효 agent 마스크.
        pair_mask (Optional[Tensor]): shape (B, A, A) bool.
            - True인 (i,j) 쌍만 검사합니다.
            - None이면 모든 i<j 쌍을 검사합니다.
        eps (float): 수치 안정용 작은 값.

    Returns:
        Tensor: shape (B,) bool. True면 그 샘플에서 어떤 pair라도 겹칩니다.
    """
    if oth_xy.ndim != 3 or int(oth_xy.shape[-1]) != 2:
        raise ValueError(f"oth_xy must be (B,A,2). got shape={tuple(oth_xy.shape)}")
    B = int(oth_xy.shape[0])
    A = int(oth_xy.shape[1])
    if A <= 1:
        return torch.zeros((B,), device=oth_xy.device, dtype=torch.bool)

    if (not isinstance(oth_cs, torch.Tensor)) or oth_cs.ndim != 3 or int(oth_cs.shape[-1]) != 2:
        raise ValueError(f"oth_cs must be (B,A,2). got shape={tuple(getattr(oth_cs, 'shape', []))}")
    if (not isinstance(oth_wl, torch.Tensor)) or oth_wl.ndim != 3 or int(oth_wl.shape[-1]) != 2:
        raise ValueError(f"oth_wl must be (B,A,2). got shape={tuple(getattr(oth_wl, 'shape', []))}")
    if (not isinstance(oth_valid, torch.Tensor)) or oth_valid.ndim != 2:
        raise ValueError(f"oth_valid must be (B,A). got shape={tuple(getattr(oth_valid, 'shape', []))}")

    if int(oth_cs.shape[0]) != B or int(oth_cs.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_cs")
    if int(oth_wl.shape[0]) != B or int(oth_wl.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_wl")
    if int(oth_valid.shape[0]) != B or int(oth_valid.shape[1]) != A:
        raise ValueError("batch/agent size mismatch: oth_valid")

    x = oth_xy[..., 0]  # (B,A)
    y = oth_xy[..., 1]  # (B,A)

    cos_v = oth_cs[..., 0]  # (B,A)
    sin_v = oth_cs[..., 1]  # (B,A)

    # half extents
    half_W = 0.5 * oth_wl[..., 0]  # (B,A)
    half_L = 0.5 * oth_wl[..., 1]  # (B,A)

    # pairwise center diff: t = cJ - cI  -> (B,A,A)
    tx = x[:, None, :] - x[:, :, None]
    ty = y[:, None, :] - y[:, :, None]

    cos_i = cos_v[:, :, None]  # (B,A,1)
    sin_i = sin_v[:, :, None]  # (B,A,1)
    cos_j = cos_v[:, None, :]  # (B,1,A)
    sin_j = sin_v[:, None, :]  # (B,1,A)

    # t in I frame
    tIx = tx * cos_i + ty * sin_i
    tIy = -tx * sin_i + ty * cos_i

    # Rotation matrix between axes of I and J
    R00 = cos_i * cos_j + sin_i * sin_j
    R01 = cos_i * (-sin_j) + sin_i * cos_j
    R10 = (-sin_i) * cos_j + cos_i * sin_j
    R11 = (-sin_i) * (-sin_j) + cos_i * cos_j

    absR00 = torch.abs(R00) + float(eps)
    absR01 = torch.abs(R01) + float(eps)
    absR10 = torch.abs(R10) + float(eps)
    absR11 = torch.abs(R11) + float(eps)

    aW = half_W[:, :, None]  # (B,A,1)
    aL = half_L[:, :, None]  # (B,A,1)
    bW = half_W[:, None, :]  # (B,1,A)
    bL = half_L[:, None, :]  # (B,1,A)

    # axis uI / vI
    cond0 = torch.abs(tIx) <= (aL + bL * absR00 + bW * absR01)
    cond1 = torch.abs(tIy) <= (aW + bL * absR10 + bW * absR11)

    # t in J frame: tJ = R^T * tI
    tJx = tIx * R00 + tIy * R10
    tJy = tIx * R01 + tIy * R11

    # axis uJ / vJ
    cond2 = torch.abs(tJx) <= (bL + aL * absR00 + aW * absR10)
    cond3 = torch.abs(tJy) <= (bW + aL * absR01 + aW * absR11)

    # valid pair mask (i<j only)
    valid_ij = oth_valid[:, :, None] & oth_valid[:, None, :]  # (B,A,A)
    upper = _get_upper_tri_mask(oth_xy.device, A)  # (A,A)
    valid_ij = valid_ij & upper  # broadcast to (B,A,A)

    if pair_mask is not None:
        pm = _as_bool_mask(pair_mask)
        if pm.shape != (B, A, A):
            raise ValueError(f"pair_mask must be (B,A,A). got shape={tuple(pm.shape)}")
        valid_ij = valid_ij & pm

    intersect = valid_ij & cond0 & cond1 & cond2 & cond3  # (B,A,A)
    collide_sample = torch.any(intersect, dim=(1, 2))  # (B,)
    return collide_sample

def _obb_intersects_ego_vs_others(
    ego_xy: Tensor,  # (B, 2)
    ego_cs: Tensor,  # (B, 2) = (cos, sin)
    ego_wl: Tensor,  # (B, 2) = (width, length)
    oth_xy: Tensor,  # (B, A, 2)
    oth_cs: Tensor,  # (B, A, 2)
    oth_wl: Tensor,  # (B, A, 2)
    oth_valid: Tensor,  # (B, A) bool
    eps: float = 1e-6,
) -> Tensor:
    """ego(사각형)와 다른 agent(사각형)들이 겹치는지(SAT) 검사합니다.

    Args:
        ego_xy: (B,2) 중심 위치 (x,y)
        ego_cs: (B,2) (cos,sin)
        ego_wl: (B,2) (width,length)
        oth_xy: (B,A,2)
        oth_cs: (B,A,2)
        oth_wl: (B,A,2)
        oth_valid: (B,A) bool
        eps: 수치 안정용.

    Returns:
        Tensor:
            collide_sample: (B,) bool
            - True면 그 샘플에서 ego가 어떤 agent와라도 겹침.
    """
    B = int(ego_xy.shape[0])
    if int(oth_xy.shape[0]) != B:
        raise ValueError("batch size mismatch in collision check")

    A = int(oth_xy.shape[1])
    if A == 0:
        return torch.zeros((B,), device=ego_xy.device, dtype=torch.bool)

    ego_cos = ego_cs[:, 0]  # (B,)
    ego_sin = ego_cs[:, 1]  # (B,)
    oth_cos = oth_cs[..., 0]  # (B,A)
    oth_sin = oth_cs[..., 1]  # (B,A)

    # half extents
    aW = 0.5 * ego_wl[:, 0]  # (B,)
    aL = 0.5 * ego_wl[:, 1]  # (B,)
    bW = 0.5 * oth_wl[..., 0]  # (B,A)
    bL = 0.5 * oth_wl[..., 1]  # (B,A)

    # center diff: t = cB - cA
    tx = oth_xy[..., 0] - ego_xy[:, None, 0]  # (B,A)
    ty = oth_xy[..., 1] - ego_xy[:, None, 1]  # (B,A)

    # t in ego frame (uA, vA)
    tAx = tx * ego_cos[:, None] + ty * ego_sin[:, None]  # (B,A)
    tAy = -tx * ego_sin[:, None] + ty * ego_cos[:, None]  # (B,A)

    # Rotation matrix R where R_ij = dot(axisA_i, axisB_j)
    # axisA_0 = uA = (cosA, sinA)
    # axisA_1 = vA = (-sinA, cosA)
    # axisB_0 = uB = (cosB, sinB)
    # axisB_1 = vB = (-sinB, cosB)
    R00 = ego_cos[:, None] * oth_cos + ego_sin[:, None] * oth_sin
    R01 = ego_cos[:, None] * (-oth_sin) + ego_sin[:, None] * oth_cos
    R10 = (-ego_sin)[:, None] * oth_cos + ego_cos[:, None] * oth_sin
    R11 = (-ego_sin)[:, None] * (-oth_sin) + ego_cos[:, None] * oth_cos

    absR00 = torch.abs(R00) + float(eps)
    absR01 = torch.abs(R01) + float(eps)
    absR10 = torch.abs(R10) + float(eps)
    absR11 = torch.abs(R11) + float(eps)

    # (1) axis uA
    cond0 = torch.abs(tAx) <= (aL[:, None] + bL * absR00 + bW * absR01)
    # (2) axis vA
    cond1 = torch.abs(tAy) <= (aW[:, None] + bL * absR10 + bW * absR11)

    # t in B frame: tB = R^T * tA
    tBx = tAx * R00 + tAy * R10
    tBy = tAx * R01 + tAy * R11

    # (3) axis uB
    cond2 = torch.abs(tBx) <= (bL + aL[:, None] * absR00 + aW[:, None] * absR10)
    # (4) axis vB
    cond3 = torch.abs(tBy) <= (bW + aL[:, None] * absR01 + aW[:, None] * absR11)

    intersect = oth_valid & cond0 & cond1 & cond2 & cond3  # (B,A)
    collide_sample = torch.any(intersect, dim=1)  # (B,)
    return collide_sample


@dataclass(frozen=True)
class _TypeParams:
    """agent 타입별 상수 묶음."""
    # 증강 후보 속도 조건
    min_speed_for_aug: float
    # past/future 보간 길이(초)
    T_past: float
    T_fut: float
    # perturb range (self 좌표계 기준)
    dy: float
    dyaw: float
    dvx: float
    dvy: float
    dax: float
    day: float
    # lateral clamp(너무 옆으로 튀지 않게)
    vy_clamp: float
    ay_clamp: float


class NPCStatePerturbation:
    """ego/near(current) 증강 + 새 ego 좌표계 변환 + 보간 + 파생값 재생성을 수행합니다.

    이 클래스는 "배치 dict"를 in-place로 수정합니다.
    invalid(_is_valid=False)인 위치는 끝까지 0을 유지합니다.

    사용 위치(권장):
        train_epoch()에서 _prepare_batch_for_device(...) 직후,
        observation_normalizer(...) 호출 전.

    기대 입력 key / shape (대표):
        - ego_agent_past: (B, Tp, 11)
        - ego_future_gt_11_dim: (B, Tf, 11)
        - neighbor_agents_past: (B, A, Tp, 11)
        - neighbor_future_gt_11_dim: (B, A, Tf, 11)

        - ego_agent_past_is_valid: (B, Tp)
        - ego_future_gt_is_valid: (B, Tf)           # outputs에 있음
        - neighbor_agents_past_is_valid: (B, A, Tp)
        - neighbor_future_gt_is_valid: (B, A, Tf)

        - lanes: (B, L, 20, 12)
        - lanes_len_is_valid: (B, L, 20)
        - stop_sign_points/crosswalk_points/...: (B, N, 10, 2) + (B, N) is_valid
        - static_objects: (B, S, 10) + static_objects_is_valid: (B, S) (없거나 None이면 스킵)

    Note:
        - neighbor = near를 강제하기 위해, 처리 후 near_* key들을 neighbor_*로 맞춥니다.
        - past/future seg control은 torch로 다시 계산해서 갱신합니다(NumPy 변환 없음).
    """

    # agent 11-dim index 고정
    IDX_X = 0
    IDX_Y = 1
    IDX_COS = 2
    IDX_SIN = 3
    IDX_VX = 4
    IDX_VY = 5
    IDX_W = 6
    IDX_L = 7
    IDX_OH_CAR = 8
    IDX_OH_PED = 9
    IDX_OH_CYC = 10

    def __init__(self, *, dt: float, time_len: int, future_len: int) -> None:
        self._dt: float = float(dt)
        self._Tp: int = int(time_len)
        self._Tf: int = int(future_len)

        # 타입별 상수(너 표/규칙 그대로)
        self._car = _TypeParams(
            min_speed_for_aug=2.0,
            T_past=2.0,
            T_fut=2.0,
            dy=0.75,
            dyaw=0.35,
            dvx=1.00,
            dvy=0.50,
            dax=0.20,
            day=0.10,
            vy_clamp=0.50,
            ay_clamp=0.30,
        )
        self._cyc = _TypeParams(
            min_speed_for_aug=1.0,
            T_past=1.5,
            T_fut=1.5,
            dy=0.40,
            dyaw=0.40,
            dvx=0.50,
            dvy=0.20,
            dax=0.20,
            day=0.10,
            vy_clamp=0.30,
            ay_clamp=0.30,
        )
        self._ped = _TypeParams(
            min_speed_for_aug=0.3,
            T_past=1.0,
            T_fut=1.0,
            dy=0.30,
            dyaw=0.60,
            dvx=0.30,
            dvy=0.30,
            dax=0.30,
            day=0.30,
            vy_clamp=0.50,
            ay_clamp=0.50,
        )

        # (device,dtype,N)->basis cache
        self._basis_cache: Dict[Tuple[torch.device, torch.dtype, int],
                                Tuple[Tensor, ...]] = {}

    def _get_quintic_basis(self, device: torch.device, dtype: torch.dtype,
                           N: int) -> Tuple[Tensor, ...]:
        """quintic Hermite basis와 1차 미분 basis를 캐시해서 가져옵니다.

        Args:
            device: 텐서 device
            dtype: 텐서 dtype
            N: 구간 step 수 (예: 10, 15, 20)

        Returns:
            Tuple[Tensor,...]:
                h00,h10,h20,h01,h11,h21, dh00,dh10,dh20,dh01,dh11,dh21
                각 텐서 shape: (N+1,)
        """
        key = (device, dtype, int(N))
        if key in self._basis_cache:
            return self._basis_cache[key]

        N_i = int(N)
        if N_i <= 0:
            raise ValueError(f"N must be positive. got N={N_i}")

        # s = 0..1
        s = torch.linspace(0.0, 1.0, steps=N_i + 1, device=device, dtype=dtype)
        s2 = s * s
        s3 = s2 * s
        s4 = s3 * s
        s5 = s4 * s

        # position basis
        h00 = 1 - 10 * s3 + 15 * s4 - 6 * s5
        h10 = s - 6 * s3 + 8 * s4 - 3 * s5
        h20 = 0.5 * s2 - 1.5 * s3 + 1.5 * s4 - 0.5 * s5
        h01 = 10 * s3 - 15 * s4 + 6 * s5
        h11 = -4 * s3 + 7 * s4 - 3 * s5
        h21 = 0.5 * s3 - s4 + 0.5 * s5

        # first derivative w.r.t s
        dh00 = -30 * s2 + 60 * s3 - 30 * s4
        dh10 = 1 - 18 * s2 + 32 * s3 - 15 * s4
        dh20 = s - 4.5 * s2 + 6 * s3 - 2.5 * s4
        dh01 = 30 * s2 - 60 * s3 + 30 * s4
        dh11 = -12 * s2 + 28 * s3 - 15 * s4
        dh21 = 1.5 * s2 - 4 * s3 + 2.5 * s4

        out = (h00, h10, h20, h01, h11, h21, dh00, dh10, dh20, dh01, dh11, dh21)
        self._basis_cache[key] = out
        return out

    def _quintic_2d(
        self,
        *,
        p0: Tensor,
        v0: Tensor,
        a0: Tensor,
        p1: Tensor,
        v1: Tensor,
        a1: Tensor,
        T: float,
        N: int,
    ) -> Tuple[Tensor, Tensor]:
        """2차원 quintic 보간으로 (p(t), v(t)) 시퀀스를 만듭니다.

        기존 코드는 x축과 y축을 각각 _quintic_1d로 2번 계산합니다.
        이 함수는 같은 수식을 (x,y) 2차원을 한 번에 계산해서
        불필요한 중복 연산을 크게 줄입니다. 결과는 완전히 동일합니다.

        Args:
            p0, v0, a0:
                시작 조건. shape (M, 2)
            p1, v1, a1:
                끝 조건. shape (M, 2)
            T:
                전체 시간(초)
            N:
                step 수 (N+1 샘플 생성)

        Returns:
            Tuple[Tensor, Tensor]:
                - p_seq: shape (M, N+1, 2)
                - v_seq: shape (M, N+1, 2)
        """
        if p0.ndim != 2 or int(p0.shape[-1]) != 2:
            raise ValueError(f"p0 must be (M,2). got shape={tuple(p0.shape)}")
        if v0.ndim != 2 or int(v0.shape[-1]) != 2:
            raise ValueError(f"v0 must be (M,2). got shape={tuple(v0.shape)}")
        if a0.ndim != 2 or int(a0.shape[-1]) != 2:
            raise ValueError(f"a0 must be (M,2). got shape={tuple(a0.shape)}")
        if p1.shape != p0.shape or v1.shape != v0.shape or a1.shape != a0.shape:
            raise ValueError("p1/v1/a1 must have same shape as p0/v0/a0")

        device = p0.device
        dtype = p0.dtype
        h00, h10, h20, h01, h11, h21, dh00, dh10, dh20, dh01, dh11, dh21 = self._get_quintic_basis(
            device, dtype, int(N)
        )

        TT = float(T)

        # basis: (1, N+1, 1)
        h00v = h00[None, :, None]
        h10v = h10[None, :, None]
        h20v = h20[None, :, None]
        h01v = h01[None, :, None]
        h11v = h11[None, :, None]
        h21v = h21[None, :, None]

        dh00v = dh00[None, :, None]
        dh10v = dh10[None, :, None]
        dh20v = dh20[None, :, None]
        dh01v = dh01[None, :, None]
        dh11v = dh11[None, :, None]
        dh21v = dh21[None, :, None]

        # (M, 1, 2)
        p0v = p0[:, None, :]
        v0v = v0[:, None, :]
        a0v = a0[:, None, :]
        p1v = p1[:, None, :]
        v1v = v1[:, None, :]
        a1v = a1[:, None, :]

        p_seq = (
            h00v * p0v
            + h10v * (TT * v0v)
            + h20v * ((TT * TT) * a0v)
            + h01v * p1v
            + h11v * (TT * v1v)
            + h21v * ((TT * TT) * a1v)
        )

        dp_ds = (
            dh00v * p0v
            + dh10v * (TT * v0v)
            + dh20v * ((TT * TT) * a0v)
            + dh01v * p1v
            + dh11v * (TT * v1v)
            + dh21v * ((TT * TT) * a1v)
        )

        v_seq = dp_ds / TT
        return p_seq, v_seq

    def _quintic_1d(
        self,
        *,
        x0: Tensor,
        v0: Tensor,
        a0: Tensor,
        x1: Tensor,
        v1: Tensor,
        a1: Tensor,
        T: float,
        N: int,
    ) -> Tuple[Tensor, Tensor]:
        """1차원 quintic 보간으로 (x(t), v(t)) 시퀀스를 만듭니다.

        Args:
            x0,v0,a0: 시작 경계조건, shape: (M,)
            x1,v1,a1: 끝 경계조건, shape: (M,)
            T: 전체 시간(초)
            N: step 수 (N*dt == T 가 되도록 사용 권장)

        Returns:
            Tuple[Tensor,Tensor]:
                x_seq: shape (M, N+1)  (시작~끝 포함)
                v_seq: shape (M, N+1)
        """
        device = x0.device
        dtype = x0.dtype
        h00, h10, h20, h01, h11, h21, dh00, dh10, dh20, dh01, dh11, dh21 = self._get_quintic_basis(
            device, dtype, int(N))

        TT = float(T)
        # x(s)
        x_seq = (h00[None, :] * x0[:, None] + h10[None, :] *
                 (TT * v0)[:, None] + h20[None, :] * ((TT * TT) * a0)[:, None] +
                 h01[None, :] * x1[:, None] + h11[None, :] *
                 (TT * v1)[:, None] + h21[None, :] * ((TT * TT) * a1)[:, None])

        # v(t) = (1/T) * dx/ds
        dx_ds = (dh00[None, :] * x0[:, None] + dh10[None, :] *
                 (TT * v0)[:, None] + dh20[None, :] *
                 ((TT * TT) * a0)[:, None] + dh01[None, :] * x1[:, None] +
                 dh11[None, :] * (TT * v1)[:, None] + dh21[None, :] *
                 ((TT * TT) * a1)[:, None])
        v_seq = dx_ds / TT
        return x_seq, v_seq

    def _traj11_to_pose3(self, traj11: Tensor) -> Tensor:
        """(x,y,cos,sin,...) 11차원에서 (x,y,yaw) 3차원을 뽑습니다.

        Args:
            traj11 (Tensor): shape (..., T, 11)

        Returns:
            Tensor: pose3 shape (..., T, 3) where last=(x,y,yaw)
        """
        x = traj11[..., self.IDX_X]
        y = traj11[..., self.IDX_Y]
        yaw = torch.atan2(traj11[..., self.IDX_SIN], traj11[..., self.IDX_COS])
        return torch.stack([x, y, yaw], dim=-1)

    def _pose3_to_control3(
        self,
        pose3: Tensor,  # (..., T, 3)
        frame_valid: Tensor,  # (..., T) bool
        *,
        dt: float,
        use_body_vel: bool,
    ) -> Tuple[Tensor, Tensor]:
        """(x,y,yaw) 궤적을 (vx,vy,omega) 구간 궤적으로 바꿉니다(torch).

        Args:
            pose3:
                shape (..., T, 3) last=(x,y,yaw)
            frame_valid:
                shape (..., T) bool
            dt:
                시간 간격
            use_body_vel:
                True면 (몸체 기준 vx,vy,omega)
                False면 (좌표계 기준 vx,vy,omega)

        Returns:
            Tuple[Tensor,Tensor]:
                controls: shape (..., T-1, 3)
                seg_valid: shape (..., T-1) bool
        """
        T = int(pose3.shape[-2])
        if T <= 1:
            controls = torch.zeros((*pose3.shape[:-2], 0, 3),
                                   device=pose3.device,
                                   dtype=pose3.dtype)
            seg_valid = torch.zeros((*pose3.shape[:-2], 0),
                                    device=pose3.device,
                                    dtype=torch.bool)
            return controls, seg_valid

        x = pose3[..., 0]
        y = pose3[..., 1]
        yaw = pose3[..., 2]

        x0 = x[..., :-1]
        x1 = x[..., 1:]
        y0 = y[..., :-1]
        y1 = y[..., 1:]
        th0 = yaw[..., :-1]
        th1 = yaw[..., 1:]

        dth = _wrap_to_pi(th1 - th0)
        omega = dth / float(dt)

        vwx = (x1 - x0) / float(dt)
        vwy = (y1 - y0) / float(dt)

        if not bool(use_body_vel):
            controls = torch.stack([vwx, vwy, omega], dim=-1)
        else:
            th_mid = th0 + 0.5 * dth
            cos_mid = torch.cos(th_mid)
            sin_mid = torch.sin(th_mid)
            cos_mid, sin_mid = _normalize_cos_sin(cos_mid, sin_mid)

            vx_b = cos_mid * vwx + sin_mid * vwy
            vy_b = -sin_mid * vwx + cos_mid * vwy
            controls = torch.stack([vx_b, vy_b, omega], dim=-1)

        seg_valid = frame_valid[..., :-1] & frame_valid[..., 1:]
        controls = torch.where(seg_valid.unsqueeze(-1), controls,
                               torch.zeros_like(controls))
        return controls, seg_valid

    def _get_scenario_id_for_batch_index(self, inputs: Dict[str, Any],
                                         batch_index: int) -> str:
        """배치에서 scenario_id를 안전하게 꺼냅니다.

        Args:
            inputs (Dict[str, Any]):
                apply_inplace 입력 dict.
            batch_index (int):
                배치 인덱스. shape: ()

        Returns:
            str:
                scenario_id 문자열. 없으면 "b{batch_index}" 형태로 반환.
        """
        sid = inputs.get("scenario_id", None)
        if isinstance(sid, list) and 0 <= int(batch_index) < len(sid):
            return str(sid[int(batch_index)])
        return f"b{int(batch_index):03d}"

    def _debug_should_run(
        self,
        *,
        debug_vis_dir: Optional[str],
        debug_step: Optional[int],
        debug_max_scenes: int,
        debug_every_n_steps: int,
    ) -> bool:
        """디버그 PNG 저장을 이번 스텝에 할지 결정합니다.

        Args:
            debug_vis_dir (Optional[str]):
                저장 폴더. None/""면 저장 안 함.
            debug_step (Optional[int]):
                현재 전역 step. None이면 항상 허용.
            debug_max_scenes (int):
                한 번에 저장할 최대 scene 수. 0 이하면 저장 안 함.
            debug_every_n_steps (int):
                step % N == 0 일 때만 저장.

        Returns:
            bool:
                True면 이번 호출에서 저장 로직을 수행.
        """
        if (debug_vis_dir is None) or (str(debug_vis_dir).strip() == ""):
            return False
        if int(debug_max_scenes) <= 0:
            return False
        every = max(1, int(debug_every_n_steps))
        if debug_step is None:
            return True
        return (int(debug_step) % every) == 0

    def _detach_to_cpu_for_debug(self, t: Tensor) -> Tensor:
        """디버그용으로 텐서를 CPU로 안전하게 복사합니다.

        Args:
            t (Tensor): 임의 shape 텐서.

        Returns:
            Tensor:
                - float 텐서는 float32로 변환 후 CPU
                - 그 외 dtype은 그대로 CPU
        """
        tt = t.detach()
        if torch.is_floating_point(tt):
            tt = tt.to(dtype=torch.float32)
        return tt.cpu()

    def _snapshot_one_scene_for_debug(
        self,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Tensor],
        batch_index: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """배치에서 특정 샘플 1개를 "증강 전(before)" 용으로 스냅샷합니다.

        스냅샷 대상(그림에 필요한 것만):
          - ego/neighbor past+future 11-dim + is_valid
          - lanes + lanes_len_is_valid
          - road safety points + is_valid
          - driveway/road_edge + is_valid
          - static_objects + is_valid(있으면)

        Args:
            inputs (Dict[str, Any]):
                apply_inplace 입력 dict(아직 수정 전).
            outputs (Dict[str, Tensor]):
                apply_inplace 출력 dict(아직 수정 전).
            batch_index (int):
                배치 인덱스. shape: ()

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                (before_inputs, before_outputs)
                - 텐서들은 batch 차원을 제거한 "샘플 단위" shape로 CPU에 복사됨.
        """
        b = int(batch_index)

        # input에서 1개 샘플만 가져올 key들
        input_keys: List[str] = [
            "ego_agent_past",
            "ego_agent_past_is_valid",
            "ego_future_gt_11_dim",
            "neighbor_agents_past",
            "neighbor_agents_past_is_valid",
            "neighbor_future_gt_11_dim",
            "neighbor_future_gt_is_valid",
            "lanes",
            "lanes_len_is_valid",
            "stop_sign_points",
            "stop_sign_is_valid",
            "crosswalk_points",
            "crosswalk_is_valid",
            "speed_bump_points",
            "speed_bump_is_valid",
            "driveway_points",
            "driveway_is_valid",
            "road_edge",
            "road_edge_is_valid",
            "static_objects",
            "static_objects_is_valid",
        ]

        # output에서 1개 샘플만 가져올 key들
        output_keys: List[str] = [
            "ego_future_gt_is_valid",
        ]

        before_inputs: Dict[str, Any] = {}
        before_outputs: Dict[str, Any] = {}

        # scenario_id(문자열)도 같이 저장(있으면)
        sid = inputs.get("scenario_id", None)
        if isinstance(sid, list) and 0 <= b < len(sid):
            before_inputs["scenario_id"] = str(sid[b])
        else:
            before_inputs["scenario_id"] = f"b{b:03d}"

        for k in input_keys:
            v = inputs.get(k, None)
            if isinstance(v, torch.Tensor):
                before_inputs[k] = self._detach_to_cpu_for_debug(v[b])
            else:
                before_inputs[k] = None

        for k in output_keys:
            v = outputs.get(k, None)
            if isinstance(v, torch.Tensor):
                before_outputs[k] = self._detach_to_cpu_for_debug(v[b])
            else:
                before_outputs[k] = None

        return before_inputs, before_outputs

    def _build_debug_before_cache(
        self,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Tensor],
        candidate_indices: Tensor,  # (K,)
        max_cache: int,
    ) -> Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]]:
        """증강 전(before) 스냅샷 캐시를 만듭니다(후보 일부만).

        Args:
            inputs/outputs:
                apply_inplace 입력/출력(아직 수정 전).
            candidate_indices (Tensor):
                shape (K,) long. before 후보 배치 인덱스들.
            max_cache (int):
                최대 저장 개수.

        Returns:
            Dict[int, Tuple[before_inputs, before_outputs]]:
                key는 batch_index(int).
        """
        cache: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        if (not isinstance(candidate_indices, torch.Tensor)) or int(candidate_indices.numel()) == 0:
            return cache

        take = min(int(candidate_indices.numel()), max(1, int(max_cache)))
        cand_cpu = candidate_indices[:take].detach().to("cpu")
        for b in cand_cpu.tolist():
            bi = int(b)
            cache[bi] = self._snapshot_one_scene_for_debug(
                inputs=inputs,
                outputs=outputs,
                batch_index=bi,
            )
        return cache

    def _draw_before_after_png_extreme(
        self,
        *,
        before_inputs: Dict[str, Any],
        before_outputs: Dict[str, Any],
        after_inputs: Dict[str, Any],
        after_outputs: Dict[str, Tensor],
        batch_index: int,
        save_path: str,
        aug_ego: bool,
        aug_nbr_mask: Optional[Tensor],  # (A,) bool or None
        frame_params: Tuple[float, float, float, float],  # (xe,ye,ce,se)
        past_stride: int,
        future_stride: int,
        vel_arrow_len_m: float,
    ) -> None:
        """before/after 변화가 '극명'하게 보이도록 2x2 PNG를 저장합니다.

        패널 구성:
            - (0,0) BEFORE(aligned): before를 after frame으로 변환해서 표시
            - (0,1) AFTER
            - (1,0) OVERLAY: before/after를 겹쳐 그리고, aug agent에 Δ 화살표/수치 표시
            - (1,1) ZOOM: aug agent 주변 자동 확대(변화가 작은 경우에도 잘 보임)

        Args:
            before_inputs/before_outputs:
                배치 차원 없는(샘플 단위) before 스냅샷. CPU 텐서.
            after_inputs/after_outputs:
                현재 배치(dict). batch_index로 1개 샘플을 꺼냅니다.
            aug_ego:
                ego가 증강된 경우 True.
            aug_nbr_mask:
                shape (A,) bool. neighbor 증강 여부.
            frame_params:
                (xe, ye, ce, se) = after frame 변환 파라미터.
                - old -> new: x' = ce*(x-xe) + se*(y-ye), y' = -se*(x-xe) + ce*(y-ye)
            past_stride/future_stride:
                박스 표시 간격.
            vel_arrow_len_m:
                속도 방향 화살표 고정 길이[m].
        """
        os.environ.setdefault("MPLBACKEND", "Agg")
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon, FancyArrowPatch

        xe, ye, ce, se = (float(frame_params[0]), float(frame_params[1]),
                          float(frame_params[2]), float(frame_params[3]))

        def _to_np_f32(t: Tensor) -> np.ndarray:
            return t.detach().to(dtype=torch.float32).cpu().numpy()

        def _to_np_bool(t: Tensor) -> np.ndarray:
            arr = t.detach().cpu().numpy()
            if arr.dtype == np.bool_:
                return arr
            return (arr != 0)

        def _wrap_pi(a: np.ndarray) -> np.ndarray:
            return np.arctan2(np.sin(a), np.cos(a)).astype(np.float32, copy=False)

        def _yaw_from_cs(c: np.ndarray, s_: np.ndarray) -> np.ndarray:
            return np.arctan2(s_, c).astype(np.float32, copy=False)

        def _speed(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
            return np.hypot(vx, vy).astype(np.float32, copy=False)

        def _tf_xy(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            dx = x - xe
            dy = y - ye
            x2 = ce * dx + se * dy
            y2 = -se * dx + ce * dy
            return x2.astype(np.float32, copy=False), y2.astype(np.float32, copy=False)

        def _tf_vec(vx: np.ndarray, vy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            vx2 = ce * vx + se * vy
            vy2 = -se * vx + ce * vy
            return vx2.astype(np.float32, copy=False), vy2.astype(np.float32, copy=False)

        def _tf_traj11(traj11: np.ndarray) -> np.ndarray:
            out = traj11.copy()
            x2, y2 = _tf_xy(out[..., self.IDX_X], out[..., self.IDX_Y])
            out[..., self.IDX_X] = x2
            out[..., self.IDX_Y] = y2

            hc2, hs2 = _tf_vec(out[..., self.IDX_COS], out[..., self.IDX_SIN])
            out[..., self.IDX_COS] = hc2
            out[..., self.IDX_SIN] = hs2

            vx2, vy2 = _tf_vec(out[..., self.IDX_VX], out[..., self.IDX_VY])
            out[..., self.IDX_VX] = vx2
            out[..., self.IDX_VY] = vy2
            return out

        def _tf_lanes(lanes12: np.ndarray) -> np.ndarray:
            out = lanes12.copy()
            cx2, cy2 = _tf_xy(out[..., 0], out[..., 1])
            out[..., 0] = cx2
            out[..., 1] = cy2
            for s0 in (2, 4, 6):
                vx2, vy2 = _tf_vec(out[..., s0], out[..., s0 + 1])
                out[..., s0] = vx2
                out[..., s0 + 1] = vy2
            return out

        def _tf_points(points: np.ndarray) -> np.ndarray:
            # (...,2)
            out = points.copy()
            x2, y2 = _tf_xy(out[..., 0], out[..., 1])
            out[..., 0] = x2
            out[..., 1] = y2
            return out

        def _tf_static_objects(so10: np.ndarray) -> np.ndarray:
            out = so10.copy()
            x2, y2 = _tf_xy(out[..., 0], out[..., 1])
            out[..., 0] = x2
            out[..., 1] = y2
            hc2, hs2 = _tf_vec(out[..., 2], out[..., 3])
            out[..., 2] = hc2
            out[..., 3] = hs2
            return out

        def _oriented_box_corners(x: float, y: float, c: float, s_: float, length: float, width: float) -> np.ndarray:
            half_L = 0.5 * float(length)
            half_W = 0.5 * float(width)
            local = np.array(
                [[+half_L, +half_W], [+half_L, -half_W], [-half_L, -half_W], [-half_L, +half_W]],
                dtype=np.float32,
            )
            R = np.array([[c, -s_], [s_, c]], dtype=np.float32)
            return (local @ R.T) + np.array([x, y], dtype=np.float32)

        def _add_arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float, color: str, lw: float, alpha: float, z: int) -> None:
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    arrowstyle="-|>",
                    mutation_scale=8.0,
                    linewidth=float(lw),
                    color=color,
                    alpha=float(alpha),
                    zorder=int(z),
                    shrinkA=0.0,
                    shrinkB=0.0,
                )
            )

        def _add_vel_arrow_unit(ax: plt.Axes, x: float, y: float, vx: float, vy: float, color: str, lw: float, alpha: float, z: int) -> None:
            mag = float(np.hypot(vx, vy))
            if mag < 1e-6:
                return
            dx = float(vel_arrow_len_m) * (vx / mag)
            dy = float(vel_arrow_len_m) * (vy / mag)
            _add_arrow(ax, x, y, x + dx, y + dy, color=color, lw=lw, alpha=alpha, z=z)

        def _draw_map(ax: plt.Axes, lanes_np: Optional[np.ndarray], lanes_valid: Optional[np.ndarray]) -> None:
            ax.set_facecolor("#000000")
            if lanes_np is None or lanes_valid is None:
                return

            segments_center: List[np.ndarray] = []
            segments_left: List[np.ndarray] = []
            segments_right: List[np.ndarray] = []

            L = int(lanes_np.shape[0])
            eps_vec = 1e-6
            for li in range(L):
                vmask = lanes_valid[li].astype(bool)
                if not bool(np.any(vmask)):
                    continue
                lane = lanes_np[li]  # (20,12)
                center = lane[:, 0:2]
                left_vec = lane[:, 4:6]
                right_vec = lane[:, 6:8]

                seg_ok = vmask[:-1] & vmask[1:]
                if bool(np.any(seg_ok)):
                    p0 = center[:-1][seg_ok]
                    p1 = center[1:][seg_ok]
                    segments_center.append(np.stack([p0, p1], axis=1))

                left_ok = (np.linalg.norm(left_vec, axis=1) > eps_vec)
                right_ok = (np.linalg.norm(right_vec, axis=1) > eps_vec)

                segL = seg_ok & left_ok[:-1] & left_ok[1:]
                if bool(np.any(segL)):
                    left_xy = center + left_vec
                    p0 = left_xy[:-1][segL]
                    p1 = left_xy[1:][segL]
                    segments_left.append(np.stack([p0, p1], axis=1))

                segR = seg_ok & right_ok[:-1] & right_ok[1:]
                if bool(np.any(segR)):
                    right_xy = center + right_vec
                    p0 = right_xy[:-1][segR]
                    p1 = right_xy[1:][segR]
                    segments_right.append(np.stack([p0, p1], axis=1))

            if len(segments_center) > 0:
                seg = np.concatenate(segments_center, axis=0)
                ax.add_collection(LineCollection(seg, colors="#808080", linewidths=0.6, linestyles=(0, (4, 4)), zorder=1, alpha=0.25))
            if len(segments_left) > 0:
                seg = np.concatenate(segments_left, axis=0)
                ax.add_collection(LineCollection(seg, colors="#E6E6FA", linewidths=0.8, linestyles="-", zorder=2, alpha=0.35))
            if len(segments_right) > 0:
                seg = np.concatenate(segments_right, axis=0)
                ax.add_collection(LineCollection(seg, colors="#E6E6FA", linewidths=0.8, linestyles="-", zorder=2, alpha=0.35))

        def _draw_agents(
            ax: plt.Axes,
            *,
            ego_past: np.ndarray, ego_past_v: np.ndarray,
            ego_fut: np.ndarray, ego_fut_v: np.ndarray,
            nbr_past: np.ndarray, nbr_past_v: np.ndarray,
            nbr_fut: np.ndarray, nbr_fut_v: np.ndarray,
            aug_nbr_np: Optional[np.ndarray],
            mode: str,
        ) -> Tuple[List[float], List[float]]:
            """mode:
            - 'before' / 'after' : 일반 표시(aug는 약간 진하게)
            - 'overlay_before' / 'overlay_after' : 겹쳐그리기용(aug만 매우 진하게, 비-aug는 거의 안 보이게)
            """
            xs: List[float] = []
            ys: List[float] = []

            Tp = int(ego_past.shape[0])
            t_cur = Tp - 1

            def _draw_traj(
                traj: np.ndarray,
                valid: np.ndarray,
                *,
                color: str,
                lw: float,
                alpha: float,
                fill_cur: bool,
                fill_color: Optional[str],
                fill_alpha: float,
                stride: int,
                z: int,
            ) -> None:
                T = int(traj.shape[0])
                for t in range(0, T, int(stride)):
                    if not bool(valid[t]):
                        continue
                    row = traj[t]
                    x = float(row[self.IDX_X]); y = float(row[self.IDX_Y])
                    c = float(row[self.IDX_COS]); s_ = float(row[self.IDX_SIN])
                    r = float(np.hypot(c, s_))
                    if r < 1e-8:
                        c, s_ = 1.0, 0.0
                    else:
                        c, s_ = c / r, s_ / r

                    vx = float(row[self.IDX_VX]); vy = float(row[self.IDX_VY])
                    W = float(row[self.IDX_W]);  L = float(row[self.IDX_L])

                    corners = _oriented_box_corners(x, y, c, s_, L, W)
                    face = "none"
                    a = float(alpha)
                    if fill_cur and (t == t_cur) and (fill_color is not None):
                        face = fill_color
                        a = float(fill_alpha)

                    ax.add_patch(
                        Polygon(
                            corners,
                            closed=True,
                            facecolor=face,
                            edgecolor=color,
                            linewidth=float(lw),
                            alpha=float(a),
                            zorder=int(z + (2 if (t == t_cur) else 0)),
                        )
                    )

                    # heading line
                    hx = x + 0.5 * L * c
                    hy = y + 0.5 * L * s_
                    ax.plot([x, hx], [y, hy], color=color, linewidth=float(lw), alpha=float(alpha), zorder=int(z + 1))

                    # velocity arrow: 현재 프레임만
                    if t == t_cur:
                        _add_vel_arrow_unit(ax, x, y, vx, vy, color=color, lw=max(0.6, float(lw)), alpha=float(alpha), z=int(z + 2))

                    xs.append(x); ys.append(y)

            # ego
            if mode in ("before", "after"):
                ego_alpha = 1.0 if aug_ego else 0.7
                ego_lw = 1.3 if aug_ego else 0.9
                ego_color = "#FFD700" if aug_ego else "#BBBBBB"
            else:
                ego_alpha = 1.0
                ego_lw = 1.6
                ego_color = "#00FFFF" if mode == "overlay_before" else "#FFD700"

            _draw_traj(
                ego_past, ego_past_v,
                color=ego_color, lw=ego_lw, alpha=ego_alpha,
                fill_cur=True, fill_color="#FFFFFF", fill_alpha=0.55 if aug_ego else 0.35,
                stride=int(past_stride), z=30,
            )
            _draw_traj(
                ego_fut, ego_fut_v,
                color=ego_color, lw=max(0.5, ego_lw * 0.6), alpha=ego_alpha,
                fill_cur=False, fill_color=None, fill_alpha=0.0,
                stride=int(future_stride), z=20,
            )

            # neighbors
            A = int(nbr_past.shape[0])
            for i in range(A):
                if not bool(nbr_past_v[i, t_cur]):
                    continue

                hi = bool(aug_nbr_np[i]) if (aug_nbr_np is not None and i < int(aug_nbr_np.shape[0])) else False
                base_color = "#84E573"  # default car-like
                # 타입별 색
                if float(nbr_past[i, t_cur, self.IDX_OH_PED]) > 0.5:
                    base_color = "#4D83E1"
                elif float(nbr_past[i, t_cur, self.IDX_OH_CYC]) > 0.5:
                    base_color = "#FFA500"

                if mode in ("before", "after"):
                    col = "#FFD700" if hi else base_color
                    a = 1.0 if hi else 0.15
                    lw = 1.1 if hi else 0.35
                    draw_full = hi  # ✅ 일반 패널에서도 변화된 agent만 “시간축”을 그림
                else:
                    # overlay에서는 hi만 거의 전부 보이게
                    col = "#00FFFF" if (mode == "overlay_before") else "#FFD700"
                    a = 1.0 if hi else 0.03
                    lw = 1.2 if hi else 0.25
                    draw_full = hi

                if draw_full:
                    _draw_traj(
                        nbr_past[i], nbr_past_v[i],
                        color=col, lw=lw, alpha=a,
                        fill_cur=True, fill_color=base_color, fill_alpha=0.18 if hi else 0.05,
                        stride=int(past_stride), z=12,
                    )
                    _draw_traj(
                        nbr_fut[i], nbr_fut_v[i],
                        color=col, lw=max(0.4, lw * 0.7), alpha=a,
                        fill_cur=False, fill_color=None, fill_alpha=0.0,
                        stride=int(future_stride), z=10,
                    )
                else:
                    # 비-aug는 현재만 아주 희미하게
                    row = nbr_past[i, t_cur]
                    x = float(row[self.IDX_X]); y = float(row[self.IDX_Y])
                    c = float(row[self.IDX_COS]); s_ = float(row[self.IDX_SIN])
                    r = float(np.hypot(c, s_))
                    if r < 1e-8:
                        c, s_ = 1.0, 0.0
                    else:
                        c, s_ = c / r, s_ / r
                    W = float(row[self.IDX_W]); L = float(row[self.IDX_L])
                    corners = _oriented_box_corners(x, y, c, s_, L, W)
                    ax.add_patch(
                        Polygon(corners, closed=True, fill=False, edgecolor=base_color,
                                linewidth=0.25, alpha=float(a), zorder=5)
                    )
                    xs.append(x); ys.append(y)

            return xs, ys

        # ---------- before(샘플) / after(배치) numpy 준비 ----------
        def _get_scene_from_before() -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            # before는 CPU 텐서(샘플 단위) 그대로 들어있음
            out["ego_past"] = _to_np_f32(before_inputs["ego_agent_past"])
            out["ego_past_v"] = _to_np_bool(_as_bool_mask(before_inputs["ego_agent_past_is_valid"]))
            out["ego_fut"] = _to_np_f32(before_inputs["ego_future_gt_11_dim"])
            out["ego_fut_v"] = _to_np_bool(_as_bool_mask(before_outputs["ego_future_gt_is_valid"]))

            out["nbr_past"] = _to_np_f32(before_inputs["neighbor_agents_past"])
            out["nbr_past_v"] = _to_np_bool(_as_bool_mask(before_inputs["neighbor_agents_past_is_valid"]))
            out["nbr_fut"] = _to_np_f32(before_inputs["neighbor_future_gt_11_dim"])
            out["nbr_fut_v"] = _to_np_bool(_as_bool_mask(before_inputs["neighbor_future_gt_is_valid"]))

            lanes = before_inputs.get("lanes", None)
            lv = before_inputs.get("lanes_len_is_valid", None)
            out["lanes"] = _to_np_f32(lanes) if isinstance(lanes, torch.Tensor) else None
            out["lanes_v"] = _to_np_bool(_as_bool_mask(lv)) if isinstance(lv, torch.Tensor) else None

            return out

        def _get_scene_from_after() -> Dict[str, Any]:
            b = int(batch_index)
            out: Dict[str, Any] = {}
            out["ego_past"] = _to_np_f32(after_inputs["ego_agent_past"][b])
            out["ego_past_v"] = _to_np_bool(_as_bool_mask(after_inputs["ego_agent_past_is_valid"][b]))
            out["ego_fut"] = _to_np_f32(after_inputs["ego_future_gt_11_dim"][b])
            out["ego_fut_v"] = _to_np_bool(_as_bool_mask(after_outputs["ego_future_gt_is_valid"][b]))

            out["nbr_past"] = _to_np_f32(after_inputs["neighbor_agents_past"][b])
            out["nbr_past_v"] = _to_np_bool(_as_bool_mask(after_inputs["neighbor_agents_past_is_valid"][b]))
            out["nbr_fut"] = _to_np_f32(after_inputs["neighbor_future_gt_11_dim"][b])
            out["nbr_fut_v"] = _to_np_bool(_as_bool_mask(after_inputs["neighbor_future_gt_is_valid"][b]))

            lanes = after_inputs.get("lanes", None)
            lv = after_inputs.get("lanes_len_is_valid", None)
            if isinstance(lanes, torch.Tensor) and isinstance(lv, torch.Tensor):
                out["lanes"] = _to_np_f32(lanes[b])
                out["lanes_v"] = _to_np_bool(_as_bool_mask(lv[b]))
            else:
                out["lanes"] = None
                out["lanes_v"] = None
            return out

        before = _get_scene_from_before()
        after = _get_scene_from_after()

        # before를 after frame으로 정렬
        before_aligned = {
            "ego_past": _tf_traj11(before["ego_past"]),
            "ego_past_v": before["ego_past_v"],
            "ego_fut": _tf_traj11(before["ego_fut"]),
            "ego_fut_v": before["ego_fut_v"],
            "nbr_past": _tf_traj11(before["nbr_past"]),
            "nbr_past_v": before["nbr_past_v"],
            "nbr_fut": _tf_traj11(before["nbr_fut"]),
            "nbr_fut_v": before["nbr_fut_v"],
            "lanes": _tf_lanes(before["lanes"]) if before["lanes"] is not None else None,
            "lanes_v": before["lanes_v"],
        }

        aug_nbr_np = _to_np_bool(aug_nbr_mask) if isinstance(aug_nbr_mask, torch.Tensor) else None

        # ---------- figure ----------
        fig, axes = plt.subplots(2, 2, figsize=(26.0, 14.0), dpi=220)
        fig.patch.set_facecolor("#000000")

        ax00, ax01 = axes[0, 0], axes[0, 1]
        ax10, ax11 = axes[1, 0], axes[1, 1]

        # 지도는 after 기준으로 그리는게 가장 안정적(둘이 거의 같아야 정상)
        for ax in (ax00, ax01, ax10, ax11):
            _draw_map(ax, after["lanes"], after["lanes_v"])
            ax.axis("off")
            ax.set_aspect("equal", adjustable="box")

        ax00.set_title("BEFORE (aligned to AFTER frame)", color="#FFFFFF", fontsize=12)
        ax01.set_title("AFTER", color="#FFFFFF", fontsize=12)
        ax10.set_title("OVERLAY + Δ vectors (aug only)", color="#FFFFFF", fontsize=12)
        ax11.set_title("ZOOM (aug agents)", color="#FFFFFF", fontsize=12)

        # (0,0) before aligned
        xs0, ys0 = _draw_agents(
            ax00,
            ego_past=before_aligned["ego_past"], ego_past_v=before_aligned["ego_past_v"],
            ego_fut=before_aligned["ego_fut"], ego_fut_v=before_aligned["ego_fut_v"],
            nbr_past=before_aligned["nbr_past"], nbr_past_v=before_aligned["nbr_past_v"],
            nbr_fut=before_aligned["nbr_fut"], nbr_fut_v=before_aligned["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="before",
        )

        # (0,1) after
        xs1, ys1 = _draw_agents(
            ax01,
            ego_past=after["ego_past"], ego_past_v=after["ego_past_v"],
            ego_fut=after["ego_fut"], ego_fut_v=after["ego_fut_v"],
            nbr_past=after["nbr_past"], nbr_past_v=after["nbr_past_v"],
            nbr_fut=after["nbr_fut"], nbr_fut_v=after["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="after",
        )

        # (1,0) overlay (before=cyan, after=gold)
        _draw_agents(
            ax10,
            ego_past=before_aligned["ego_past"], ego_past_v=before_aligned["ego_past_v"],
            ego_fut=before_aligned["ego_fut"], ego_fut_v=before_aligned["ego_fut_v"],
            nbr_past=before_aligned["nbr_past"], nbr_past_v=before_aligned["nbr_past_v"],
            nbr_fut=before_aligned["nbr_fut"], nbr_fut_v=before_aligned["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="overlay_before",
        )
        _draw_agents(
            ax10,
            ego_past=after["ego_past"], ego_past_v=after["ego_past_v"],
            ego_fut=after["ego_fut"], ego_fut_v=after["ego_fut_v"],
            nbr_past=after["nbr_past"], nbr_past_v=after["nbr_past_v"],
            nbr_fut=after["nbr_fut"], nbr_fut_v=after["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="overlay_after",
        )

        # Δ 벡터 + 수치 표시(현재 프레임 기준)
        Tp = int(after["ego_past"].shape[0])
        t_cur = Tp - 1

        def _annotate_delta_for_one_agent(
            ax: plt.Axes,
            *,
            name: str,
            p_before: np.ndarray,  # (2,)
            yaw_before: float,
            spd_before: float,
            p_after: np.ndarray,   # (2,)
            yaw_after: float,
            spd_after: float,
            color: str,
        ) -> None:
            dx = float(p_after[0] - p_before[0])
            dy = float(p_after[1] - p_before[1])
            dp = float(np.hypot(dx, dy))
            dyaw = float(_wrap_pi(np.array([yaw_after - yaw_before], dtype=np.float32))[0])
            dyaw_deg = dyaw * 180.0 / float(np.pi)
            dspd = float(spd_after - spd_before)

            _add_arrow(ax, float(p_before[0]), float(p_before[1]),
                       float(p_after[0]), float(p_after[1]),
                       color=color, lw=1.8, alpha=0.95, z=80)

            ax.text(
                float(p_after[0]),
                float(p_after[1]),
                f"{name}\n|Δp|={dp:.2f}m\nΔyaw={dyaw_deg:.1f}deg\nΔv={dspd:.2f}m/s",
                color=color,
                fontsize=8,
                ha="left",
                va="bottom",
                zorder=90,
                clip_on=True,
            )

        # ego delta
        ego_b = before_aligned["ego_past"][t_cur]
        ego_a = after["ego_past"][t_cur]
        pB = ego_b[0:2]; pA = ego_a[0:2]
        yawB = float(_yaw_from_cs(np.array([ego_b[2]]), np.array([ego_b[3]]))[0])
        yawA = float(_yaw_from_cs(np.array([ego_a[2]]), np.array([ego_a[3]]))[0])
        spB = float(_speed(np.array([ego_b[4]]), np.array([ego_b[5]]))[0])
        spA = float(_speed(np.array([ego_a[4]]), np.array([ego_a[5]]))[0])

        if bool(aug_ego):
            _annotate_delta_for_one_agent(
                ax10,
                name="EGO",
                p_before=pB, yaw_before=yawB, spd_before=spB,
                p_after=pA, yaw_after=yawA, spd_after=spA,
                color="#FF3333",
            )

        # neighbor delta (aug만)
        if aug_nbr_np is not None:
            A = int(aug_nbr_np.shape[0])
            for i in range(A):
                if not bool(aug_nbr_np[i]):
                    continue
                nb_b = before_aligned["nbr_past"][i, t_cur]
                nb_a = after["nbr_past"][i, t_cur]
                pb = nb_b[0:2]; pa = nb_a[0:2]
                yb = float(_yaw_from_cs(np.array([nb_b[2]]), np.array([nb_b[3]]))[0])
                ya = float(_yaw_from_cs(np.array([nb_a[2]]), np.array([nb_a[3]]))[0])
                sb = float(_speed(np.array([nb_b[4]]), np.array([nb_b[5]]))[0])
                sa = float(_speed(np.array([nb_a[4]]), np.array([nb_a[5]]))[0])
                _annotate_delta_for_one_agent(
                    ax10,
                    name=f"N{i}",
                    p_before=pb, yaw_before=yb, spd_before=sb,
                    p_after=pa, yaw_after=ya, spd_after=sa,
                    color="#FF3333",
                )

        # (1,1) zoom: overlay를 그대로 그리고 축만 줄임
        _draw_agents(
            ax11,
            ego_past=before_aligned["ego_past"], ego_past_v=before_aligned["ego_past_v"],
            ego_fut=before_aligned["ego_fut"], ego_fut_v=before_aligned["ego_fut_v"],
            nbr_past=before_aligned["nbr_past"], nbr_past_v=before_aligned["nbr_past_v"],
            nbr_fut=before_aligned["nbr_fut"], nbr_fut_v=before_aligned["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="overlay_before",
        )
        _draw_agents(
            ax11,
            ego_past=after["ego_past"], ego_past_v=after["ego_past_v"],
            ego_fut=after["ego_fut"], ego_fut_v=after["ego_fut_v"],
            nbr_past=after["nbr_past"], nbr_past_v=after["nbr_past_v"],
            nbr_fut=after["nbr_fut"], nbr_fut_v=after["nbr_fut_v"],
            aug_nbr_np=aug_nbr_np,
            mode="overlay_after",
        )

        # ---------- 축 범위 ----------
        # global: 너무 커지지 않게 clamp
        xs = (xs0 + xs1)
        ys = (ys0 + ys1)
        if len(xs) == 0:
            global_half = 60.0
        else:
            global_half = max(30.0, min(120.0, float(max(np.max(np.abs(xs)), np.max(np.abs(ys))) + 8.0)))

        for ax in (ax00, ax01, ax10):
            ax.set_xlim(-global_half, global_half)
            ax.set_ylim(-global_half, global_half)

        # zoom: aug agent 주변 자동
        zoom_pts: List[Tuple[float, float]] = []
        if bool(aug_ego):
            zoom_pts.append((float(pA[0]), float(pA[1])))
            zoom_pts.append((float(pB[0]), float(pB[1])))
        if aug_nbr_np is not None:
            A = int(aug_nbr_np.shape[0])
            for i in range(A):
                if not bool(aug_nbr_np[i]):
                    continue
                pb = before_aligned["nbr_past"][i, t_cur, 0:2]
                pa = after["nbr_past"][i, t_cur, 0:2]
                zoom_pts.append((float(pb[0]), float(pb[1])))
                zoom_pts.append((float(pa[0]), float(pa[1])))

        if len(zoom_pts) == 0:
            zoom_half = 20.0
            zx, zy = 0.0, 0.0
        else:
            zx = float(np.mean([p[0] for p in zoom_pts]))
            zy = float(np.mean([p[1] for p in zoom_pts]))
            max_dx = float(max(abs(p[0] - zx) for p in zoom_pts))
            max_dy = float(max(abs(p[1] - zy) for p in zoom_pts))
            zoom_half = max(8.0, min(50.0, max(max_dx, max_dy) + 6.0))

        ax11.set_xlim(zx - zoom_half, zx + zoom_half)
        ax11.set_ylim(zy - zoom_half, zy + zoom_half)

        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)

    def _maybe_save_augmented_debug_png(
        self,
        *,
        inputs: Dict[str, Any],
        outputs: Dict[str, Tensor],
        idx_keep: Tensor,   # (Bk,)
        aug_ego: Tensor,    # (B,)
        aug_nbr: Tensor,    # (B,A)
        debug_vis_dir: Optional[str],
        debug_step: Optional[int],
        debug_max_scenes: int,
        debug_every_n_steps: int,
        past_stride: int,
        future_stride: int,
        vel_arrow_len_m: float,
        debug_before_cache: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]],
        debug_frame_params_cache: Dict[int, Tuple[float, float, float, float]],  # ✅ 추가
    ) -> None:
        if not self._debug_should_run(
            debug_vis_dir=debug_vis_dir,
            debug_step=debug_step,
            debug_max_scenes=int(debug_max_scenes),
            debug_every_n_steps=int(debug_every_n_steps),
        ):
            return

        if (debug_vis_dir is None) or (str(debug_vis_dir).strip() == ""):
            return

        try:
            os.makedirs(str(debug_vis_dir), exist_ok=True)
        except Exception:
            return

        if (not isinstance(idx_keep, torch.Tensor)) or int(idx_keep.numel()) == 0:
            return

        idx_keep_cpu = idx_keep.detach().to("cpu").tolist()

        picked: List[int] = []
        for b in idx_keep_cpu:
            bi = int(b)
            if (bi in debug_before_cache) and (bi in debug_frame_params_cache):
                picked.append(bi)
            if len(picked) >= int(debug_max_scenes):
                break

        if len(picked) == 0:
            return

        step_i = None if debug_step is None else int(debug_step)
        step_str = f"{step_i:09d}" if step_i is not None else "stepNA"

        for bi in picked:
            before_inp, before_out = debug_before_cache[bi]
            frame_params = debug_frame_params_cache[bi]  # (xe,ye,ce,se)

            scenario_id = str(before_inp.get("scenario_id", f"b{bi:03d}"))
            save_path = os.path.join(
                str(debug_vis_dir),
                f"{step_str}_{scenario_id}_extreme.png",
            )

            aug_ego_b = bool(aug_ego[bi].item()) if isinstance(aug_ego, torch.Tensor) else False
            aug_nbr_b = aug_nbr[bi].detach() if (isinstance(aug_nbr, torch.Tensor) and aug_nbr.ndim == 2) else None

            self._draw_before_after_png_extreme(
                before_inputs=before_inp,
                before_outputs=before_out,
                after_inputs=inputs,
                after_outputs=outputs,
                batch_index=int(bi),
                save_path=str(save_path),
                aug_ego=bool(aug_ego_b),
                aug_nbr_mask=aug_nbr_b,
                frame_params=frame_params,
                past_stride=max(1, int(past_stride)),
                future_stride=max(1, int(future_stride)),
                vel_arrow_len_m=float(vel_arrow_len_m),
            )

    def _draw_before_after_png(
        self,
        *,
        before_inputs: Dict[str, Any],
        before_outputs: Dict[str, Any],
        after_inputs: Dict[str, Any],
        after_outputs: Dict[str, Tensor],
        batch_index: int,
        save_path: str,
        aug_ego: bool,
        aug_nbr_mask: Optional[Tensor],  # (A,) bool or None
        past_stride: int,
        future_stride: int,
        vel_arrow_len_m: float,
    ) -> None:
        """좌(증강 전) / 우(증강 후) 비교 PNG를 저장합니다.

        강조 규칙:
          - 증강된 ego/neighbor는 진하게(금색, 두꺼운 선, alpha=1)
          - 나머지는 옅게(alpha 낮게)

        Args:
            before_inputs/before_outputs:
                batch 차원 없는 "샘플 단위" 스냅샷(모두 CPU).
            after_inputs/after_outputs:
                현재 배치(dict). 여기서 batch_index로 1개 샘플을 뽑아 그림.
        """
        os.environ.setdefault("MPLBACKEND", "Agg")
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Polygon, FancyArrowPatch

        def _to_np_f32(t: Tensor) -> np.ndarray:
            return t.detach().to(dtype=torch.float32).cpu().numpy()

        def _to_np_bool(t: Tensor) -> np.ndarray:
            arr = t.detach().cpu().numpy()
            if arr.dtype == np.bool_:
                return arr
            return (arr != 0)

        def _norm_cs(c: float, s: float) -> Tuple[float, float]:
            r = float(np.hypot(c, s))
            if r < 1e-8:
                return 1.0, 0.0
            return c / r, s / r

        def _oriented_box_corners(x: float, y: float, c: float, s: float, length: float, width: float) -> np.ndarray:
            half_L = 0.5 * float(length)
            half_W = 0.5 * float(width)
            local = np.array(
                [[+half_L, +half_W], [+half_L, -half_W], [-half_L, -half_W], [-half_L, +half_W]],
                dtype=np.float32,
            )
            R = np.array([[c, -s], [s, c]], dtype=np.float32)
            return (local @ R.T) + np.array([x, y], dtype=np.float32)

        def _add_vel_arrow_unit(ax: plt.Axes, x: float, y: float, vx: float, vy: float, color: str, lw: float, alpha: float, z: int) -> None:
            mag = float(np.hypot(vx, vy))
            if mag < 1e-6:
                return
            dx = float(vel_arrow_len_m) * (vx / mag)
            dy = float(vel_arrow_len_m) * (vy / mag)
            ax.add_patch(
                FancyArrowPatch(
                    (x, y),
                    (x + dx, y + dy),
                    arrowstyle="-|>",
                    mutation_scale=6.0,
                    linewidth=float(lw),
                    color=color,
                    alpha=float(alpha),
                    zorder=int(z),
                    shrinkA=0.0,
                    shrinkB=0.0,
                )
            )

        def _add_heading_line(ax: plt.Axes, x: float, y: float, c: float, s: float, length: float, color: str, lw: float, alpha: float, z: int) -> None:
            hx = x + 0.5 * float(length) * c
            hy = y + 0.5 * float(length) * s
            ax.plot([x, hx], [y, hy], color=color, linewidth=float(lw), alpha=float(alpha), zorder=int(z))

        def _cls_color_from_row(row11: np.ndarray) -> str:
            if float(row11[self.IDX_OH_CAR]) > 0.5:
                return "#84E573"  # car
            if float(row11[self.IDX_OH_PED]) > 0.5:
                return "#4D83E1"  # ped
            return "#FFA500"      # cyc

        def _draw_traj_rects(
            ax: plt.Axes,
            traj11: np.ndarray,       # (T,11)
            valid_t: np.ndarray,      # (T,)
            *,
            edge_color: str,
            base_alpha: float,
            line_width: float,
            stride: int,
            fill_current_index: Optional[int],
            fill_color: Optional[str],
            fill_alpha_current: float,
            zorder_base: int,
        ) -> None:
            T = int(traj11.shape[0])
            for t in range(0, T, int(stride)):
                if not bool(valid_t[t]):
                    continue
                row = traj11[t]
                x = float(row[self.IDX_X]); y = float(row[self.IDX_Y])
                c = float(row[self.IDX_COS]); s = float(row[self.IDX_SIN])
                c, s = _norm_cs(c, s)
                vx = float(row[self.IDX_VX]); vy = float(row[self.IDX_VY])
                W = float(row[self.IDX_W]);  L = float(row[self.IDX_L])

                corners = _oriented_box_corners(x, y, c, s, L, W)

                is_cur = (fill_current_index is not None) and (t == int(fill_current_index))
                face = "none"
                alpha = float(base_alpha)
                if is_cur and (fill_color is not None):
                    face = fill_color
                    alpha = float(fill_alpha_current)

                ax.add_patch(
                    Polygon(
                        corners,
                        closed=True,
                        facecolor=face,
                        edgecolor=edge_color,
                        linewidth=float(line_width),
                        alpha=float(alpha),
                        zorder=int(zorder_base + (2 if is_cur else 0)),
                    )
                )

                # heading + velocity는 "항상 고정 길이 화살표"
                _add_heading_line(ax, x, y, c, s, L, edge_color, line_width, base_alpha, zorder_base + 1)
                _add_vel_arrow_unit(ax, x, y, vx, vy, edge_color, max(0.5, float(line_width)), base_alpha, zorder_base + 2)

        def _extract_scene_np(
            *,
            src_inputs: Dict[str, Any],
            src_outputs: Dict[str, Any],
            batch_idx: Optional[int],
        ) -> Dict[str, Any]:
            """src에서 필요한 텐서를 numpy로 뽑습니다.
            batch_idx가 None이면 이미 (샘플 단위)로 들어온 것으로 간주합니다.
            """
            def _get_in(k: str) -> Optional[Tensor]:
                v = src_inputs.get(k, None)
                if not isinstance(v, torch.Tensor):
                    return None
                return v if batch_idx is None else v[int(batch_idx)]

            def _get_out(k: str) -> Optional[Tensor]:
                v = src_outputs.get(k, None)
                if not isinstance(v, torch.Tensor):
                    return None
                return v if batch_idx is None else v[int(batch_idx)]

            out: Dict[str, Any] = {}
            # agents
            out["ego_past"] = _get_in("ego_agent_past")
            out["ego_past_valid"] = _get_in("ego_agent_past_is_valid")
            out["ego_future"] = _get_in("ego_future_gt_11_dim")
            out["ego_future_valid"] = _get_out("ego_future_gt_is_valid")

            out["nbr_past"] = _get_in("neighbor_agents_past")
            out["nbr_past_valid"] = _get_in("neighbor_agents_past_is_valid")
            out["nbr_future"] = _get_in("neighbor_future_gt_11_dim")
            out["nbr_future_valid"] = _get_in("neighbor_future_gt_is_valid")

            # map / points
            out["lanes"] = _get_in("lanes")
            out["lanes_valid"] = _get_in("lanes_len_is_valid")

            for pts_k, v_k in [
                ("stop_sign_points", "stop_sign_is_valid"),
                ("crosswalk_points", "crosswalk_is_valid"),
                ("speed_bump_points", "speed_bump_is_valid"),
                ("driveway_points", "driveway_is_valid"),
                ("road_edge", "road_edge_is_valid"),
            ]:
                out[pts_k] = _get_in(pts_k)
                out[v_k] = _get_in(v_k)

            out["static_objects"] = _get_in("static_objects")
            out["static_objects_is_valid"] = _get_in("static_objects_is_valid")

            # numpy 변환
            def _to_np(v: Optional[Tensor]) -> Optional[np.ndarray]:
                if v is None:
                    return None
                if v.dtype == torch.bool:
                    return _to_np_bool(v)
                return _to_np_f32(v)

            for k in list(out.keys()):
                out[k] = _to_np(out[k])

            return out

        def _draw_scene(ax: plt.Axes, scene: Dict[str, Any], *, title: str, aug_ego_flag: bool, aug_nbr_np: Optional[np.ndarray]) -> Tuple[List[float], List[float]]:
            ax.set_facecolor("#000000")
            ax.set_title(title, color="#FFFFFF", fontsize=10)

            xs: List[float] = []
            ys: List[float] = []

            # lanes (빠르게 LineCollection)
            lanes_np = scene.get("lanes", None)
            lanes_v = scene.get("lanes_valid", None)
            if (lanes_np is not None) and (lanes_v is not None):
                segments_center: List[np.ndarray] = []
                segments_left: List[np.ndarray] = []
                segments_right: List[np.ndarray] = []

                L = int(lanes_np.shape[0])
                eps_vec = 1e-6
                for li in range(L):
                    vmask = lanes_v[li].astype(bool)  # (20,)
                    if not bool(np.any(vmask)):
                        continue
                    lane = lanes_np[li]  # (20,12)
                    center = lane[:, 0:2]
                    left_vec = lane[:, 4:6]
                    right_vec = lane[:, 6:8]

                    seg_ok = vmask[:-1] & vmask[1:]
                    if bool(np.any(seg_ok)):
                        p0 = center[:-1][seg_ok]
                        p1 = center[1:][seg_ok]
                        segments_center.append(np.stack([p0, p1], axis=1))

                    left_ok = (np.linalg.norm(left_vec, axis=1) > eps_vec)
                    right_ok = (np.linalg.norm(right_vec, axis=1) > eps_vec)

                    segL = seg_ok & left_ok[:-1] & left_ok[1:]
                    if bool(np.any(segL)):
                        left_xy = center + left_vec
                        p0 = left_xy[:-1][segL]
                        p1 = left_xy[1:][segL]
                        segments_left.append(np.stack([p0, p1], axis=1))

                    segR = seg_ok & right_ok[:-1] & right_ok[1:]
                    if bool(np.any(segR)):
                        right_xy = center + right_vec
                        p0 = right_xy[:-1][segR]
                        p1 = right_xy[1:][segR]
                        segments_right.append(np.stack([p0, p1], axis=1))

                if len(segments_center) > 0:
                    seg = np.concatenate(segments_center, axis=0)
                    ax.add_collection(LineCollection(seg, colors="#808080", linewidths=0.6, linestyles=(0, (4, 4)), zorder=1, alpha=0.35))
                    xs.extend(seg[..., 0].reshape(-1).tolist())
                    ys.extend(seg[..., 1].reshape(-1).tolist())
                if len(segments_left) > 0:
                    seg = np.concatenate(segments_left, axis=0)
                    ax.add_collection(LineCollection(seg, colors="#E6E6FA", linewidths=0.8, linestyles="-", zorder=2, alpha=0.55))
                    xs.extend(seg[..., 0].reshape(-1).tolist())
                    ys.extend(seg[..., 1].reshape(-1).tolist())
                if len(segments_right) > 0:
                    seg = np.concatenate(segments_right, axis=0)
                    ax.add_collection(LineCollection(seg, colors="#E6E6FA", linewidths=0.8, linestyles="-", zorder=2, alpha=0.55))
                    xs.extend(seg[..., 0].reshape(-1).tolist())
                    ys.extend(seg[..., 1].reshape(-1).tolist())

            # road safety polygons
            def _draw_polys(pts: Optional[np.ndarray], v: Optional[np.ndarray], edge_color: str, text: str) -> None:
                if pts is None or v is None:
                    return
                N = int(pts.shape[0])
                for i in range(N):
                    if not bool(v[i]):
                        continue
                    poly = pts[i]
                    if not bool(np.any(np.abs(poly) > 0)):
                        continue
                    ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor=edge_color, linewidth=1.0, zorder=4, alpha=0.9))
                    cx = float(np.mean(poly[:, 0])); cy = float(np.mean(poly[:, 1]))
                    ax.text(cx, cy, text, color=edge_color, fontsize=6, ha="center", va="center", zorder=5)

            _draw_polys(scene.get("stop_sign_points"), scene.get("stop_sign_is_valid"), "#D50000", "stop")
            _draw_polys(scene.get("crosswalk_points"), scene.get("crosswalk_is_valid"), "#FFFFFF", "cross")
            _draw_polys(scene.get("speed_bump_points"), scene.get("speed_bump_is_valid"), "#FFA500", "bump")
            _draw_polys(scene.get("driveway_points"), scene.get("driveway_is_valid"), "#FFFFFF", "drive")

            # road_edge (점)
            re = scene.get("road_edge", None)
            rev = scene.get("road_edge_is_valid", None)
            if re is not None and rev is not None:
                E = int(re.shape[0])
                for i in range(E):
                    if not bool(rev[i]):
                        continue
                    pts = re[i]
                    ok = np.any(np.abs(pts) > 0, axis=1)
                    if not bool(np.any(ok)):
                        continue
                    xy = pts[ok]
                    ax.scatter(xy[:, 0], xy[:, 1], s=8.0, marker="x", linewidths=0.6, c="#D50000", zorder=3, alpha=0.8)
                    xs.extend(xy[:, 0].tolist()); ys.extend(xy[:, 1].tolist())

            # static_objects
            so = scene.get("static_objects", None)
            sov = scene.get("static_objects_is_valid", None)
            if so is not None and sov is not None:
                S = int(so.shape[0])
                for i in range(S):
                    if not bool(sov[i]):
                        continue
                    row = so[i]
                    x = float(row[0]); y = float(row[1])
                    c = float(row[2]); s = float(row[3])
                    c, s = _norm_cs(c, s)
                    W = float(row[4]); L = float(row[5])
                    corners = _oriented_box_corners(x, y, c, s, L, W)
                    ax.add_patch(Polygon(corners, closed=True, fill=False, edgecolor="#FFFFFF", linewidth=0.6, zorder=6, alpha=0.6))
                    _add_heading_line(ax, x, y, c, s, L, "#FFFFFF", 0.6, 0.6, 7)
                    xs.extend(corners[:, 0].tolist()); ys.extend(corners[:, 1].tolist())

            # agents
            ego_past = scene["ego_past"]; ego_past_v = scene["ego_past_valid"]
            ego_fut = scene["ego_future"]; ego_fut_v = scene["ego_future_valid"]
            nbr_past = scene["nbr_past"]; nbr_past_v = scene["nbr_past_valid"]
            nbr_fut = scene["nbr_future"]; nbr_fut_v = scene["nbr_future_valid"]

            Tp = int(ego_past.shape[0])
            t_cur = Tp - 1

            # ego 강조/비강조
            ego_edge = "#FFD700" if aug_ego_flag else "#808080"
            ego_alpha = 1.0 if aug_ego_flag else 0.55
            ego_lw = 1.3 if aug_ego_flag else 0.8

            _draw_traj_rects(
                ax,
                ego_past,
                ego_past_v.astype(bool),
                edge_color=ego_edge,
                base_alpha=ego_alpha,
                line_width=ego_lw,
                stride=int(past_stride),
                fill_current_index=t_cur,
                fill_color="#FFFFFF",
                fill_alpha_current=0.6 if aug_ego_flag else 0.35,
                zorder_base=20,
            )
            _draw_traj_rects(
                ax,
                ego_fut,
                ego_fut_v.astype(bool),
                edge_color=ego_edge,
                base_alpha=ego_alpha,
                line_width=0.6 if aug_ego_flag else 0.4,
                stride=int(future_stride),
                fill_current_index=None,
                fill_color=None,
                fill_alpha_current=0.0,
                zorder_base=18,
            )

            xs.extend(ego_past[ego_past_v.astype(bool), 0].tolist())
            ys.extend(ego_past[ego_past_v.astype(bool), 1].tolist())
            xs.extend(ego_fut[ego_fut_v.astype(bool), 0].tolist())
            ys.extend(ego_fut[ego_fut_v.astype(bool), 1].tolist())

            A = int(nbr_past.shape[0])
            for i in range(A):
                if not bool(nbr_past_v[i, t_cur]):
                    continue

                base_color = _cls_color_from_row(nbr_past[i, t_cur])
                hi = bool(aug_nbr_np[i]) if (aug_nbr_np is not None and i < int(aug_nbr_np.shape[0])) else False

                # ✅ 핵심: 증강된 agent만 진하게 / 나머지는 옅게
                edge_color = "#FFD700" if hi else base_color
                alpha = 1.0 if hi else 0.12
                lw = 1.1 if hi else 0.35

                _draw_traj_rects(
                    ax,
                    nbr_past[i],
                    nbr_past_v[i].astype(bool),
                    edge_color=edge_color,
                    base_alpha=alpha,
                    line_width=lw,
                    stride=int(past_stride),
                    fill_current_index=t_cur,
                    fill_color=base_color,
                    fill_alpha_current=(0.25 if hi else 0.08),
                    zorder_base=12,
                )
                _draw_traj_rects(
                    ax,
                    nbr_fut[i],
                    nbr_fut_v[i].astype(bool),
                    edge_color=edge_color,
                    base_alpha=alpha,
                    line_width=0.55 if hi else 0.25,
                    stride=int(future_stride),
                    fill_current_index=None,
                    fill_color=None,
                    fill_alpha_current=0.0,
                    zorder_base=10,
                )

                # bounds용
                vv0 = nbr_past_v[i].astype(bool)
                vv1 = nbr_fut_v[i].astype(bool)
                if np.any(vv0):
                    xs.extend(nbr_past[i, vv0, 0].tolist()); ys.extend(nbr_past[i, vv0, 1].tolist())
                if np.any(vv1):
                    xs.extend(nbr_fut[i, vv1, 0].tolist()); ys.extend(nbr_fut[i, vv1, 1].tolist())

            ax.axis("off")
            ax.set_aspect("equal", adjustable="box")
            return xs, ys

        # before/after scene 준비
        scene_before = _extract_scene_np(src_inputs=before_inputs, src_outputs=before_outputs, batch_idx=None)
        scene_after = _extract_scene_np(src_inputs=after_inputs, src_outputs=after_outputs, batch_idx=int(batch_index))

        aug_nbr_np = None
        if isinstance(aug_nbr_mask, torch.Tensor):
            aug_nbr_np = _to_np_bool(aug_nbr_mask)

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(24.0, 12.0), dpi=200)
        fig.patch.set_facecolor("#000000")

        xsL, ysL = _draw_scene(axL, scene_before, title="BEFORE", aug_ego_flag=bool(aug_ego), aug_nbr_np=aug_nbr_np)
        xsR, ysR = _draw_scene(axR, scene_after, title="AFTER", aug_ego_flag=bool(aug_ego), aug_nbr_np=aug_nbr_np)

        # 두 그림이 같은 범위를 쓰도록 union bounds 설정
        xs = xsL + xsR
        ys = ysL + ysR
        if len(xs) == 0:
            xmin, xmax, ymin, ymax = -30.0, 30.0, -30.0, 30.0
        else:
            xmin, xmax = float(min(xs)), float(max(xs))
            ymin, ymax = float(min(ys)), float(max(ys))
            # 여백
            margin = 5.0
            xmin -= margin; xmax += margin
            ymin -= margin; ymax += margin

        for ax in (axL, axR):
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        fig.savefig(save_path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)


    def apply_inplace(
            self,
            *,
            inputs: Dict[str, Any],
            outputs: Dict[str, Tensor],
            augment_prob: float,
            agent_prob: float,
            use_body_vel: bool,

            # ---- debug vis (선택) ----
            debug_vis_dir: Optional[str] = None,
            debug_step: Optional[int] = None,
            debug_max_scenes: int = 2,
            debug_every_n_steps: int = 1,
            debug_past_stride: int = 1,
            debug_future_stride: int = 1,
            debug_vel_arrow_len_m: float = 1.0,
    ) -> None:
        """배치(inputs/outputs)를 in-place로 증강합니다.

        Args:
            inputs:
                _prepare_batch_for_device 이후의 입력 dict.
                텐서들은 이미 device(args.device)로 올라와 있어야 합니다.
            outputs:
                _prepare_batch_for_device에서 분리된 출력 dict(정답 텐서들).
            augment_prob:
                샘플 단위 확률. shape: ()
            agent_prob:
                샘플 안에서 agent 단위 확률. shape: ()
            use_body_vel:
                seg control을 (몸체 기준)으로 만들지 여부.

        Returns:
            None
        """
        ego_agent_past = inputs.get("ego_agent_past", None)
        ego_future_11 = inputs.get("ego_future_gt_11_dim", None)
        nbr_past = inputs.get("neighbor_agents_past", None)
        nbr_future_11 = inputs.get("neighbor_future_gt_11_dim", None)

        if not isinstance(ego_agent_past, torch.Tensor):
            return
        if not isinstance(ego_future_11, torch.Tensor):
            return
        if not isinstance(nbr_past, torch.Tensor):
            return
        if not isinstance(nbr_future_11, torch.Tensor):
            return

        device = ego_agent_past.device
        dtype = ego_agent_past.dtype

        B = int(ego_agent_past.shape[0])
        Tp = int(ego_agent_past.shape[1])
        Tf = int(ego_future_11.shape[1])
        t_cur = Tp - 1

        if Tp != int(self._Tp) or Tf != int(self._Tf):
            # args.time_len / args.future_len와 다르면 그냥 스킵(안전)
            return

        A = int(nbr_past.shape[1])

        ego_past_valid = _as_bool_mask(
            inputs["ego_agent_past_is_valid"])  # (B,Tp)
        ego_future_valid = _as_bool_mask(
            outputs["ego_future_gt_is_valid"])  # (B,Tf)

        nbr_past_valid = _as_bool_mask(
            inputs["neighbor_agents_past_is_valid"])  # (B,A,Tp)
        nbr_future_valid = _as_bool_mask(
            inputs["neighbor_future_gt_is_valid"])  # (B,A,Tf)

        # -----------------------------
        # Step 0) 샘플/agent 마스크
        # -----------------------------
        if B <= 0:
            return

        aug_sample_mask = (torch.rand((B,), device=device)
                           < float(augment_prob))  # (B,)

        # current state
        ego_cur = ego_agent_past[:, t_cur, :]  # (B,11)
        nbr_cur = nbr_past[:, :, t_cur, :]  # (B,A,11)

        ego_cur_valid = ego_past_valid[:, t_cur]  # (B,)
        nbr_cur_valid = nbr_past_valid[:, :, t_cur]  # (B,A)

        ego_speed = torch.sqrt(ego_cur[:, self.IDX_VX]**2 +
                               ego_cur[:, self.IDX_VY]**2)
        if A > 0:
            nbr_speed = torch.sqrt(nbr_cur[..., self.IDX_VX]**2 +
                                   nbr_cur[..., self.IDX_VY]**2)
        else:
            nbr_speed = torch.zeros((B, 0), device=device, dtype=dtype)

        ego_is_car = ego_cur[:, self.IDX_OH_CAR] > 0.5
        ego_is_ped = ego_cur[:, self.IDX_OH_PED] > 0.5
        ego_is_cyc = ego_cur[:, self.IDX_OH_CYC] > 0.5

        if A > 0:
            nbr_is_car = nbr_cur[..., self.IDX_OH_CAR] > 0.5
            nbr_is_ped = nbr_cur[..., self.IDX_OH_PED] > 0.5
            nbr_is_cyc = nbr_cur[..., self.IDX_OH_CYC] > 0.5
        else:
            nbr_is_car = torch.zeros((B, 0), device=device, dtype=torch.bool)
            nbr_is_ped = torch.zeros((B, 0), device=device, dtype=torch.bool)
            nbr_is_cyc = torch.zeros((B, 0), device=device, dtype=torch.bool)

        def _update_cos_sin_from_vxvy(
            *,
            vx: Tensor,        # (..., K)
            vy: Tensor,        # (..., K)
            cos_old: Tensor,   # (..., K)
            sin_old: Tensor,   # (..., K)
            speed_eps: float,
        ) -> Tuple[Tensor, Tensor]:
            """속도 방향으로 (cos,sin)을 만들고, 너무 느리면 기존 값을 유지합니다.

            Args:
                vx (Tensor): shape (..., K) x방향 속도.
                vy (Tensor): shape (..., K) y방향 속도.
                cos_old (Tensor): shape (..., K) 기존 cos 값.
                sin_old (Tensor): shape (..., K) 기존 sin 값.
                speed_eps (float): sqrt(vx^2+vy^2)가 이 값보다 작으면 기존 방향을 유지합니다.

            Returns:
                Tuple[Tensor, Tensor]:
                    - cos_new: shape (..., K)
                    - sin_new: shape (..., K)
            """
            speed = torch.sqrt(vx * vx + vy * vy)
            use_new = speed > float(speed_eps)

            yaw = torch.atan2(vy, vx)
            cos_new = torch.cos(yaw)
            sin_new = torch.sin(yaw)

            cos_out = torch.where(use_new, cos_new, cos_old)
            sin_out = torch.where(use_new, sin_new, sin_old)
            cos_out, sin_out = _normalize_cos_sin(cos_out, sin_out)
            return cos_out, sin_out

        # 타입별 보간 길이(step)
        def _N_from_T(T: float) -> int:
            return int(round(float(T) / float(self._dt)))

        Np_car = _N_from_T(self._car.T_past)
        Np_cyc = _N_from_T(self._cyc.T_past)
        Np_ped = _N_from_T(self._ped.T_past)

        Nf_car = _N_from_T(self._car.T_fut)
        Nf_cyc = _N_from_T(self._cyc.T_fut)
        Nf_ped = _N_from_T(self._ped.T_fut)

        # 과거/미래 유효 길이 검사
        def _has_past(valid_bt: Tensor, N: int) -> Tensor:
            # valid_bt: (B,Tp)
            if t_cur - N < 0:
                return torch.zeros((valid_bt.shape[0],),
                                   device=valid_bt.device,
                                   dtype=torch.bool)
            return torch.all(valid_bt[:, (t_cur - N):(t_cur + 1)], dim=-1)

        def _has_future(valid_btf: Tensor, N: int) -> Tensor:
            # valid_btf: (B,Tf)
            if N <= 0:
                return torch.zeros((valid_btf.shape[0],),
                                   device=valid_btf.device,
                                   dtype=torch.bool)
            if N > int(valid_btf.shape[1]):
                return torch.zeros((valid_btf.shape[0],),
                                   device=valid_btf.device,
                                   dtype=torch.bool)
            return torch.all(valid_btf[:, :N], dim=-1)

        ego_has_past = ((ego_is_car & _has_past(ego_past_valid, Np_car)) |
                        (ego_is_cyc & _has_past(ego_past_valid, Np_cyc)) |
                        (ego_is_ped & _has_past(ego_past_valid, Np_ped)))
        ego_has_fut = ((ego_is_car & _has_future(ego_future_valid, Nf_car)) |
                       (ego_is_cyc & _has_future(ego_future_valid, Nf_cyc)) |
                       (ego_is_ped & _has_future(ego_future_valid, Nf_ped)))

        if A > 0:

            def _has_past_nbr(valid_bat: Tensor, N: int) -> Tensor:
                # (B,A,Tp) -> (B,A)
                if t_cur - N < 0:
                    return torch.zeros((valid_bat.shape[0], valid_bat.shape[1]),
                                       device=valid_bat.device,
                                       dtype=torch.bool)
                return torch.all(valid_bat[:, :, (t_cur - N):(t_cur + 1)],
                                 dim=-1)

            def _has_fut_nbr(valid_baf: Tensor, N: int) -> Tensor:
                # (B,A,Tf) -> (B,A)
                if N <= 0 or N > int(valid_baf.shape[2]):
                    return torch.zeros((valid_baf.shape[0], valid_baf.shape[1]),
                                       device=valid_baf.device,
                                       dtype=torch.bool)
                return torch.all(valid_baf[:, :, :N], dim=-1)

            nbr_has_past = (
                (nbr_is_car & _has_past_nbr(nbr_past_valid, Np_car)) |
                (nbr_is_cyc & _has_past_nbr(nbr_past_valid, Np_cyc)) |
                (nbr_is_ped & _has_past_nbr(nbr_past_valid, Np_ped)))
            nbr_has_fut = (
                (nbr_is_car & _has_fut_nbr(nbr_future_valid, Nf_car)) |
                (nbr_is_cyc & _has_fut_nbr(nbr_future_valid, Nf_cyc)) |
                (nbr_is_ped & _has_fut_nbr(nbr_future_valid, Nf_ped)))
        else:
            nbr_has_past = torch.zeros((B, 0), device=device, dtype=torch.bool)
            nbr_has_fut = torch.zeros((B, 0), device=device, dtype=torch.bool)

        # speed 조건
        ego_speed_ok = ((ego_is_car &
                         (ego_speed >= self._car.min_speed_for_aug)) |
                        (ego_is_cyc &
                         (ego_speed >= self._cyc.min_speed_for_aug)) |
                        (ego_is_ped &
                         (ego_speed >= self._ped.min_speed_for_aug)))
        if A > 0:
            nbr_speed_ok = ((nbr_is_car &
                             (nbr_speed >= self._car.min_speed_for_aug)) |
                            (nbr_is_cyc &
                             (nbr_speed >= self._cyc.min_speed_for_aug)) |
                            (nbr_is_ped &
                             (nbr_speed >= self._ped.min_speed_for_aug)))
        else:
            nbr_speed_ok = torch.zeros((B, 0), device=device, dtype=torch.bool)

        # agent 랜덤
        ego_pick = torch.rand((B,), device=device) < float(agent_prob)
        if A > 0:
            nbr_pick = torch.rand((B, A), device=device) < float(agent_prob)
        else:
            nbr_pick = torch.zeros((B, 0), device=device, dtype=torch.bool)

        aug_ego = aug_sample_mask & ego_pick & ego_cur_valid & ego_speed_ok & ego_has_past & ego_has_fut  # (B,)
        aug_nbr = (aug_sample_mask[:, None] & nbr_pick & nbr_cur_valid &
                   nbr_speed_ok & nbr_has_past & nbr_has_fut)  # (B,A)

        any_agent_aug = aug_ego | (
            torch.any(aug_nbr, dim=1) if A > 0 else torch.zeros((B,),
                                                                device=device,
                                                                dtype=torch.bool)
        )
        do_sample_aug = aug_sample_mask & any_agent_aug  # (B,)

        # ✅ CPU가 torch.any(...) 결과를 매번 확인하지 않게, nonzero 결과를 1번만 만듦
        sel_do = torch.nonzero(do_sample_aug, as_tuple=True)[0]  # (Bs,)
        if int(sel_do.numel()) == 0:
            return
        # -----------------------------
        # [DEBUG] 증강 전(before) 스냅샷(최소 비용)
        # - do_sample_aug 후보 중 일부만 CPU로 복사해둠
        # - 나중에 idx_keep(실제 적용 샘플)과 매칭해서 before/after 한 장에 저장
        # -----------------------------
        debug_before_cache: Dict[int, Tuple[Dict[str, Any], Dict[str, Any]]] = {}
        debug_enabled = self._debug_should_run(
            debug_vis_dir=debug_vis_dir,
            debug_step=debug_step,
            debug_max_scenes=int(debug_max_scenes),
            debug_every_n_steps=int(debug_every_n_steps),
        )
        if debug_enabled:
            # collision reject를 감안해 조금 여유 있게 저장 후보를 잡음
            max_cache = max(1, int(debug_max_scenes) * 8)
            debug_before_cache = self._build_debug_before_cache(
                inputs=inputs,
                outputs=outputs,
                candidate_indices=sel_do,
                max_cache=int(max_cache),
            )
        # -----------------------------
        # Step 2~4) 현재 acc, yaw_rate(속도방향)
        # -----------------------------
        # 현재 acc는 past에서 vx 차분으로 계산(현재 포함)
        # -----------------------------
        # Step 2~4) 현재 acc, yaw_rate(속도방향)
        # -----------------------------
        # ✅ 기존: (B,Tp,2)/(B,A,Tp,2) 전체 acc를 만든 뒤 t_cur만 사용
        # ✅ 개선: t_cur 한 시점만 차분으로 계산 (결과 동일, 연산/메모리 크게 절감)

        ego_v_past = ego_agent_past[..., self.IDX_VX:self.IDX_VY + 1]  # (B, Tp, 2)
        ego_cur_acc = _compute_acc_at_t_from_vxvy(
            ego_v_past,          # (B, Tp, 2)
            ego_past_valid,      # (B, Tp)
            dt=self._dt,
            t=t_cur,
        )  # (B, 2)

        if A > 0:
            nbr_v_past = nbr_past[..., self.IDX_VX:self.IDX_VY + 1]  # (B, A, Tp, 2)
            nbr_cur_acc = _compute_acc_at_t_from_vxvy(
                nbr_v_past,       # (B, A, Tp, 2)
                nbr_past_valid,   # (B, A, Tp)
                dt=self._dt,
                t=t_cur,
            )  # (B, A, 2)
        else:
            nbr_cur_acc = torch.zeros((B, 0, 2), device=device, dtype=dtype)


        # -----------------------------
        # Step 5) self 좌표계 변환(현재)
        # -----------------------------
        ego_cos0 = ego_cur[:, self.IDX_COS]
        ego_sin0 = ego_cur[:, self.IDX_SIN]
        ego_cos0, ego_sin0 = _normalize_cos_sin(ego_cos0, ego_sin0)

        if A > 0:
            nbr_cos0 = nbr_cur[..., self.IDX_COS]
            nbr_sin0 = nbr_cur[..., self.IDX_SIN]
            nbr_cos0, nbr_sin0 = _normalize_cos_sin(nbr_cos0, nbr_sin0)
        else:
            nbr_cos0 = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_sin0 = torch.zeros((B, 0), device=device, dtype=dtype)

        # v_self = R(-yaw) v
        ego_vx = ego_cur[:, self.IDX_VX]
        ego_vy = ego_cur[:, self.IDX_VY]
        ego_vx_self, ego_vy_self = _rot_vec_by_minus_yaw(
            ego_vx, ego_vy, ego_cos0, ego_sin0)

        ego_ax = ego_cur_acc[:, 0]
        ego_ay = ego_cur_acc[:, 1]
        ego_ax_self, ego_ay_self = _rot_vec_by_minus_yaw(
            ego_ax, ego_ay, ego_cos0, ego_sin0)

        if A > 0:
            nbr_vx = nbr_cur[..., self.IDX_VX]
            nbr_vy = nbr_cur[..., self.IDX_VY]
            nbr_vx_self, nbr_vy_self = _rot_vec_by_minus_yaw(
                nbr_vx, nbr_vy, nbr_cos0, nbr_sin0)

            nbr_ax = nbr_cur_acc[..., 0]
            nbr_ay = nbr_cur_acc[..., 1]
            nbr_ax_self, nbr_ay_self = _rot_vec_by_minus_yaw(
                nbr_ax, nbr_ay, nbr_cos0, nbr_sin0)
        else:
            nbr_vx_self = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_vy_self = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ax_self = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ay_self = torch.zeros((B, 0), device=device, dtype=dtype)

        # -----------------------------
        # Step 6) self 좌표계 perturb
        # -----------------------------
        # range 텐서 만들기
        def _ranges_for_type(
            is_car: Tensor, is_ped: Tensor, is_cyc: Tensor
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            dy = (is_car.to(dtype) * self._car.dy +
                  is_ped.to(dtype) * self._ped.dy +
                  is_cyc.to(dtype) * self._cyc.dy)
            dyaw = (is_car.to(dtype) * self._car.dyaw +
                    is_ped.to(dtype) * self._ped.dyaw +
                    is_cyc.to(dtype) * self._cyc.dyaw)
            dvx = (is_car.to(dtype) * self._car.dvx +
                   is_ped.to(dtype) * self._ped.dvx +
                   is_cyc.to(dtype) * self._cyc.dvx)
            dvy = (is_car.to(dtype) * self._car.dvy +
                   is_ped.to(dtype) * self._ped.dvy +
                   is_cyc.to(dtype) * self._cyc.dvy)
            dax = (is_car.to(dtype) * self._car.dax +
                   is_ped.to(dtype) * self._ped.dax +
                   is_cyc.to(dtype) * self._cyc.dax)
            day = (is_car.to(dtype) * self._car.day +
                   is_ped.to(dtype) * self._ped.day +
                   is_cyc.to(dtype) * self._cyc.day)
            return dy, dyaw, dvx, dvy, dax, day

        ego_dy, ego_dyaw, ego_dvx, ego_dvy, ego_dax, ego_day = _ranges_for_type(
            ego_is_car, ego_is_ped, ego_is_cyc)
        if A > 0:
            nbr_dy, nbr_dyaw, nbr_dvx, nbr_dvy, nbr_dax, nbr_day = _ranges_for_type(
                nbr_is_car, nbr_is_ped, nbr_is_cyc)
        else:
            nbr_dy = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_dyaw = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_dvx = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_dvy = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_dax = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_day = torch.zeros((B, 0), device=device, dtype=dtype)

        # delta 샘플링(Uniform[-range, +range])
        def _sample_delta(range_t: Tensor) -> Tensor:
            return (torch.rand_like(range_t) * 2.0 - 1.0) * range_t

        ego_delta_y = _sample_delta(ego_dy)
        ego_delta_yaw = _sample_delta(ego_dyaw)
        ego_delta_vx = _sample_delta(ego_dvx)
        ego_delta_vy = _sample_delta(ego_dvy)
        ego_delta_ax = _sample_delta(ego_dax)
        ego_delta_ay = _sample_delta(ego_day)

        if A > 0:
            nbr_delta_y = _sample_delta(nbr_dy)
            nbr_delta_yaw = _sample_delta(nbr_dyaw)
            nbr_delta_vx = _sample_delta(nbr_dvx)
            nbr_delta_vy = _sample_delta(nbr_dvy)
            nbr_delta_ax = _sample_delta(nbr_dax)
            nbr_delta_ay = _sample_delta(nbr_day)
        else:
            nbr_delta_y = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_delta_yaw = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_delta_vx = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_delta_vy = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_delta_ax = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_delta_ay = torch.zeros((B, 0), device=device, dtype=dtype)

        # yaw perturb로 v/a도 같이 회전
        ego_cd = torch.cos(ego_delta_yaw)
        ego_sd = torch.sin(ego_delta_yaw)
        ego_vx_base, ego_vy_base = _rot_vec_by_yaw(ego_vx_self, ego_vy_self,
                                                   ego_cd, ego_sd)
        ego_ax_base, ego_ay_base = _rot_vec_by_yaw(ego_ax_self, ego_ay_self,
                                                   ego_cd, ego_sd)

        ego_vx_self_p = ego_vx_base + ego_delta_vx
        ego_vy_self_p = ego_vy_base + ego_delta_vy
        ego_ax_self_p = ego_ax_base + ego_delta_ax
        ego_ay_self_p = ego_ay_base + ego_delta_ay

        if A > 0:
            nbr_cd = torch.cos(nbr_delta_yaw)
            nbr_sd = torch.sin(nbr_delta_yaw)
            nbr_vx_base, nbr_vy_base = _rot_vec_by_yaw(nbr_vx_self, nbr_vy_self,
                                                       nbr_cd, nbr_sd)
            nbr_ax_base, nbr_ay_base = _rot_vec_by_yaw(nbr_ax_self, nbr_ay_self,
                                                       nbr_cd, nbr_sd)

            nbr_vx_self_p = nbr_vx_base + nbr_delta_vx
            nbr_vy_self_p = nbr_vy_base + nbr_delta_vy
            nbr_ax_self_p = nbr_ax_base + nbr_delta_ax
            nbr_ay_self_p = nbr_ay_base + nbr_delta_ay
        else:
            nbr_vx_self_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_vy_self_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ax_self_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ay_self_p = torch.zeros((B, 0), device=device, dtype=dtype)

        # 안전장치: car/cyc는 vx>=0, lateral clamp
        def _apply_type_safety_self(
                vx: Tensor,
                vy: Tensor,
                ax: Tensor,
                ay: Tensor,
                is_car: Tensor,
                is_cyc: Tensor,
                is_ped: Tensor,
        ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
            """타입별 최소 안전장치만 적용합니다.

            - car/cyc: vx는 0 이상으로 제한
            - 모든 타입: vy/ay는 타입별 범위로 제한
            """
            # vx >= 0 for car/cyc
            vx2 = torch.where((is_car | is_cyc), torch.clamp_min(vx, 0.0), vx)

            # lateral clamp
            car_vy_lim = float(self._car.vy_clamp)
            cyc_vy_lim = float(self._cyc.vy_clamp)
            ped_vy_lim = float(self._ped.vy_clamp)

            car_ay_lim = float(self._car.ay_clamp)
            cyc_ay_lim = float(self._cyc.ay_clamp)
            ped_ay_lim = float(self._ped.ay_clamp)

            vy_lim = (
                    is_car.to(vy.dtype) * car_vy_lim
                    + is_cyc.to(vy.dtype) * cyc_vy_lim
                    + is_ped.to(vy.dtype) * ped_vy_lim
            )
            ay_lim = (
                    is_car.to(ay.dtype) * car_ay_lim
                    + is_cyc.to(ay.dtype) * cyc_ay_lim
                    + is_ped.to(ay.dtype) * ped_ay_lim
            )

            vy2 = torch.clamp(vy, -vy_lim, vy_lim)
            ay2 = torch.clamp(ay, -ay_lim, ay_lim)

            return vx2, vy2, ax, ay2

        ego_vx_self_p, ego_vy_self_p, ego_ax_self_p, ego_ay_self_p = _apply_type_safety_self(
            ego_vx_self_p, ego_vy_self_p, ego_ax_self_p, ego_ay_self_p,
            ego_is_car, ego_is_cyc, ego_is_ped,
        )
        if A > 0:
            nbr_vx_self_p, nbr_vy_self_p, nbr_ax_self_p, nbr_ay_self_p = _apply_type_safety_self(
                nbr_vx_self_p, nbr_vy_self_p, nbr_ax_self_p, nbr_ay_self_p,
                nbr_is_car, nbr_is_cyc, nbr_is_ped,
            )

        # -----------------------------
        # Step 7) self -> old ego 복원 (현재만)
        # -----------------------------
        # 위치 delta: (dx=0, dy=delta_y)
        ego_dx_self = torch.zeros_like(ego_delta_y)
        ego_dy_self = ego_delta_y

        ego_dx_w, ego_dy_w = _rot_vec_by_yaw(ego_dx_self, ego_dy_self, ego_cos0,
                                             ego_sin0)
        ego_x_p = ego_cur[:, self.IDX_X] + ego_dx_w
        ego_y_p = ego_cur[:, self.IDX_Y] + ego_dy_w

        # heading: yaw0 + delta_yaw -> cos/sin (각도 합 공식)
        ego_cos_p = ego_cos0 * ego_cd - ego_sin0 * ego_sd
        ego_sin_p = ego_sin0 * ego_cd + ego_cos0 * ego_sd
        ego_cos_p, ego_sin_p = _normalize_cos_sin(ego_cos_p, ego_sin_p)

        # v,a: R(yaw0) * v_self'
        ego_vx_p, ego_vy_p = _rot_vec_by_yaw(ego_vx_self_p, ego_vy_self_p,
                                             ego_cos0, ego_sin0)
        ego_ax_p, ego_ay_p = _rot_vec_by_yaw(ego_ax_self_p, ego_ay_self_p,
                                             ego_cos0, ego_sin0)

        if A > 0:
            nbr_dx_self = torch.zeros_like(nbr_delta_y)
            nbr_dy_self = nbr_delta_y
            nbr_dx_w, nbr_dy_w = _rot_vec_by_yaw(nbr_dx_self, nbr_dy_self,
                                                 nbr_cos0, nbr_sin0)
            nbr_x_p = nbr_cur[..., self.IDX_X] + nbr_dx_w
            nbr_y_p = nbr_cur[..., self.IDX_Y] + nbr_dy_w

            nbr_cos_p = nbr_cos0 * nbr_cd - nbr_sin0 * nbr_sd
            nbr_sin_p = nbr_sin0 * nbr_cd + nbr_cos0 * nbr_sd
            nbr_cos_p, nbr_sin_p = _normalize_cos_sin(nbr_cos_p, nbr_sin_p)

            nbr_vx_p, nbr_vy_p = _rot_vec_by_yaw(nbr_vx_self_p, nbr_vy_self_p,
                                                 nbr_cos0, nbr_sin0)
            nbr_ax_p, nbr_ay_p = _rot_vec_by_yaw(nbr_ax_self_p, nbr_ay_self_p,
                                                 nbr_cos0, nbr_sin0)
        else:
            nbr_x_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_y_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_cos_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_sin_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_vx_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_vy_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ax_p = torch.zeros((B, 0), device=device, dtype=dtype)
            nbr_ay_p = torch.zeros((B, 0), device=device, dtype=dtype)

        # "현재 state"에 perturb 반영(충돌 검사 전 준비)
        ego_cur_after = ego_cur.clone()
        ego_cur_acc_after = ego_cur_acc.clone()
        ego_cur_after[:, self.IDX_X] = torch.where(aug_ego, ego_x_p,
                                                   ego_cur_after[:, self.IDX_X])
        ego_cur_after[:, self.IDX_Y] = torch.where(aug_ego, ego_y_p,
                                                   ego_cur_after[:, self.IDX_Y])
        ego_cur_after[:, self.IDX_COS] = torch.where(
            aug_ego, ego_cos_p, ego_cur_after[:, self.IDX_COS])
        ego_cur_after[:, self.IDX_SIN] = torch.where(
            aug_ego, ego_sin_p, ego_cur_after[:, self.IDX_SIN])
        ego_cur_after[:,
                      self.IDX_VX] = torch.where(aug_ego, ego_vx_p,
                                                 ego_cur_after[:, self.IDX_VX])
        ego_cur_after[:,
                      self.IDX_VY] = torch.where(aug_ego, ego_vy_p,
                                                 ego_cur_after[:, self.IDX_VY])
        ego_cur_acc_after[:, 0] = torch.where(aug_ego, ego_ax_p,
                                              ego_cur_acc_after[:, 0])
        ego_cur_acc_after[:, 1] = torch.where(aug_ego, ego_ay_p,
                                              ego_cur_acc_after[:, 1])

        if A > 0:
            nbr_cur_after = nbr_cur.clone()
            nbr_cur_acc_after = nbr_cur_acc.clone()
            nbr_cur_after[..., self.IDX_X] = torch.where(
                aug_nbr, nbr_x_p, nbr_cur_after[..., self.IDX_X])
            nbr_cur_after[..., self.IDX_Y] = torch.where(
                aug_nbr, nbr_y_p, nbr_cur_after[..., self.IDX_Y])
            nbr_cur_after[..., self.IDX_COS] = torch.where(
                aug_nbr, nbr_cos_p, nbr_cur_after[..., self.IDX_COS])
            nbr_cur_after[..., self.IDX_SIN] = torch.where(
                aug_nbr, nbr_sin_p, nbr_cur_after[..., self.IDX_SIN])
            nbr_cur_after[..., self.IDX_VX] = torch.where(
                aug_nbr, nbr_vx_p, nbr_cur_after[..., self.IDX_VX])
            nbr_cur_after[..., self.IDX_VY] = torch.where(
                aug_nbr, nbr_vy_p, nbr_cur_after[..., self.IDX_VY])
            nbr_cur_acc_after[..., 0] = torch.where(aug_nbr, nbr_ax_p,
                                                    nbr_cur_acc_after[..., 0])
            nbr_cur_acc_after[..., 1] = torch.where(aug_nbr, nbr_ay_p,
                                                    nbr_cur_acc_after[..., 1])
        else:
            nbr_cur_after = nbr_cur
            nbr_cur_acc_after = nbr_cur_acc

        # -----------------------------
        # Step 8) 현재 충돌 검사
        #   (1) ego vs others
        #   (2) others vs others (neighbor끼리)
        #       - 속도/데이터 손실을 줄이기 위해 "증강된 neighbor가 포함된 pair"만 검사
        # -----------------------------
        sel = sel_do  # ✅ 위에서 만든 걸 재사용

        ego_xy_sel = torch.stack(
            [ego_cur_after[sel, self.IDX_X], ego_cur_after[sel, self.IDX_Y]],
            dim=-1)  # (Bs,2)
        ego_cs_sel = torch.stack([
            ego_cur_after[sel, self.IDX_COS], ego_cur_after[sel, self.IDX_SIN]
        ],
                                 dim=-1)  # (Bs,2)
        ego_wl_sel = torch.stack(
            [ego_cur_after[sel, self.IDX_W], ego_cur_after[sel, self.IDX_L]],
            dim=-1)  # (Bs,2)

        if A > 0:
            oth_xy_sel = torch.stack([
                nbr_cur_after[sel, :, self.IDX_X], nbr_cur_after[sel, :,
                                                                 self.IDX_Y]
            ],
                                     dim=-1)  # (Bs,A,2)
            oth_cs_sel = torch.stack([
                nbr_cur_after[sel, :, self.IDX_COS], nbr_cur_after[sel, :,
                                                                   self.IDX_SIN]
            ],
                                     dim=-1)  # (Bs,A,2)
            oth_wl_sel = torch.stack([
                nbr_cur_after[sel, :, self.IDX_W], nbr_cur_after[sel, :,
                                                                 self.IDX_L]
            ],
                                     dim=-1)  # (Bs,A,2)
            oth_valid_sel = nbr_cur_valid[sel, :]  # (Bs,A)

            collide_sel = _obb_intersects_ego_vs_others(
                ego_xy=ego_xy_sel,
                ego_cs=ego_cs_sel,
                ego_wl=ego_wl_sel,
                oth_xy=oth_xy_sel,
                oth_cs=oth_cs_sel,
                oth_wl=oth_wl_sel,
                oth_valid=oth_valid_sel,
            )  # (Bs,)

            # --- neighbor-neighbor collision (증강된 neighbor가 포함된 pair만) ---
            collide_nbr_sel = torch.zeros_like(collide_sel)  # (Bs,)
            if A > 1:
                aug_nbr_sel_mask = aug_nbr[sel]  # (Bs, A)
                any_aug_nbr_sel = torch.any(aug_nbr_sel_mask, dim=1)  # (Bs,)
                idx_sub = torch.nonzero(any_aug_nbr_sel, as_tuple=True)[
                    0]  # (Bs2,)
                if int(idx_sub.numel()) != 0:
                    collide_sub = _obb_intersects_others_vs_others_augmask_fast(
                        oth_xy=oth_xy_sel[idx_sub],
                        oth_cs=oth_cs_sel[idx_sub],
                        oth_wl=oth_wl_sel[idx_sub],
                        oth_valid=oth_valid_sel[idx_sub],
                        aug_mask=aug_nbr_sel_mask[idx_sub],
                    )  # (Bs2,)
                    collide_nbr_sel[idx_sub] = collide_sub
        else:
            collide_sel = torch.zeros((int(sel.numel()),),
                                      device=device,
                                      dtype=torch.bool)
            collide_nbr_sel = torch.zeros_like(collide_sel)

        final_aug_sample_mask = do_sample_aug.clone()
        final_aug_sample_mask[sel] = final_aug_sample_mask[sel] & (
            ~(collide_sel | collide_nbr_sel))

        # ✅ torch.any(...)로 CPU가 확인하지 말고, 필요한 idx_keep를 바로 만듦
        idx_keep = torch.nonzero(final_aug_sample_mask, as_tuple=True)[
            0]  # (Bk,)
        if int(idx_keep.numel()) == 0:
            return

        # 샘플 단위로 꺼진 것들은 agent도 전부 꺼짐
        aug_ego = aug_ego & final_aug_sample_mask
        aug_nbr = aug_nbr & final_aug_sample_mask[:, None]

        # ego current 업데이트
        # ego current 업데이트 (GPU sync 없는 버전)
        ego_keep_local = torch.nonzero(aug_ego[idx_keep], as_tuple=True)[
            0]  # (Me,)
        if int(ego_keep_local.numel()) != 0:
            idx_ego_keep = idx_keep.index_select(0, ego_keep_local)  # (Me,)
            ego_m = aug_ego[idx_ego_keep]  # (Me,) bool

            ego_agent_past[idx_ego_keep, t_cur, self.IDX_X] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_X],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_X]
            )
            ego_agent_past[idx_ego_keep, t_cur, self.IDX_Y] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_Y],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_Y]
            )
            ego_agent_past[idx_ego_keep, t_cur, self.IDX_COS] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_COS],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_COS]
            )
            ego_agent_past[idx_ego_keep, t_cur, self.IDX_SIN] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_SIN],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_SIN]
            )
            ego_agent_past[idx_ego_keep, t_cur, self.IDX_VX] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_VX],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_VX]
            )
            ego_agent_past[idx_ego_keep, t_cur, self.IDX_VY] = torch.where(
                ego_m, ego_cur_after[idx_ego_keep, self.IDX_VY],
                ego_agent_past[idx_ego_keep, t_cur, self.IDX_VY]
            )
        # neighbor current 업데이트 (GPU sync 없는 버전)
        if A > 0:
            nbr_mask_keep = aug_nbr[idx_keep]  # (Bk, A)
            nbr_rows = torch.nonzero(torch.any(nbr_mask_keep, dim=1), as_tuple=True)[
                0]  # (Mn,)

            if int(nbr_rows.numel()) != 0:
                idx_nbr_keep = idx_keep.index_select(0, nbr_rows)  # (Mn,)
                mask = aug_nbr[idx_nbr_keep]  # (Mn, A)
                m1 = mask.unsqueeze(-1)  # (Mn, A, 1)

                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_X] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_X],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_X],
                )
                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_Y] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_Y],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_Y],
                )
                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_COS] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_COS],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_COS],
                )
                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_SIN] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_SIN],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_SIN],
                )
                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_VX] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_VX],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_VX],
                )
                nbr_past[idx_nbr_keep, :, t_cur, self.IDX_VY] = torch.where(
                    m1[..., 0],
                    nbr_cur_after[idx_nbr_keep, :, self.IDX_VY],
                    nbr_past[idx_nbr_keep, :, t_cur, self.IDX_VY],
                )

        # -----------------------------
        # Step 10) perturbed ego 기준 new ego 좌표계 변환(딱 1번)
        # -----------------------------
        # 기준 ego pose(현재)
        ego_cur2 = ego_agent_past[idx_keep, t_cur, :]  # (Bk,11)
        ego_xe = ego_cur2[:, self.IDX_X]  # (Bk,)
        ego_ye = ego_cur2[:, self.IDX_Y]  # (Bk,)
        ego_ce = ego_cur2[:, self.IDX_COS]
        ego_se = ego_cur2[:, self.IDX_SIN]
        ego_ce, ego_se = _normalize_cos_sin(ego_ce, ego_se)

        # -----------------------------
        # [DEBUG] before를 after frame으로 정렬하기 위한 파라미터 저장
        # - old -> new ego frame 변환: (xe, ye, ce, se)
        # -----------------------------
        debug_frame_params_cache: Dict[int, Tuple[float, float, float, float]] = {}
        if debug_enabled:
            idx_keep_cpu = idx_keep.detach().to("cpu")
            xe_cpu = ego_xe.detach().to(dtype=torch.float32).cpu()
            ye_cpu = ego_ye.detach().to(dtype=torch.float32).cpu()
            ce_cpu = ego_ce.detach().to(dtype=torch.float32).cpu()
            se_cpu = ego_se.detach().to(dtype=torch.float32).cpu()

            # idx_keep의 j번째가 원 배치의 b 인덱스
            for j, b in enumerate(idx_keep_cpu.tolist()):
                bi = int(b)
                debug_frame_params_cache[bi] = (
                    float(xe_cpu[j].item()),
                    float(ye_cpu[j].item()),
                    float(ce_cpu[j].item()),
                    float(se_cpu[j].item()),
                )
        else:
            debug_frame_params_cache = {}

        # (A) ego/neighbor past/future 변환
        def _transform_agent_traj11_inplace(traj: Tensor,
                                            valid: Tensor) -> None:
            # traj: (Bk, ..., T, 11)
            # valid: (Bk, ..., T) bool
            x = traj[..., self.IDX_X]
            y = traj[..., self.IDX_Y]
            dx = x - ego_xe.view(-1, *([1] * (x.ndim - 1)))
            dy = y - ego_ye.view(-1, *([1] * (y.ndim - 1)))

            # R(-yaw_e)
            x_new = ego_ce.view(-1, *([1] * (x.ndim - 1))) * dx + ego_se.view(
                -1, *([1] * (x.ndim - 1))) * dy
            y_new = -ego_se.view(-1, *([1] * (x.ndim - 1))) * dx + ego_ce.view(
                -1, *([1] * (y.ndim - 1))) * dy
            traj[..., self.IDX_X] = x_new
            traj[..., self.IDX_Y] = y_new

            # heading (cos,sin) rotate only
            hc = traj[..., self.IDX_COS]
            hs = traj[..., self.IDX_SIN]
            hc_new = ego_ce.view(-1, *([1] * (hc.ndim - 1))) * hc + ego_se.view(
                -1, *([1] * (hc.ndim - 1))) * hs
            hs_new = -ego_se.view(-1, *([1] *
                                        (hs.ndim - 1))) * hc + ego_ce.view(
                                            -1, *([1] * (hs.ndim - 1))) * hs
            hc_new, hs_new = _normalize_cos_sin(hc_new, hs_new)
            traj[..., self.IDX_COS] = hc_new
            traj[..., self.IDX_SIN] = hs_new

            # velocity rotate only
            vx = traj[..., self.IDX_VX]
            vy = traj[..., self.IDX_VY]
            vx_new = ego_ce.view(-1, *([1] * (vx.ndim - 1))) * vx + ego_se.view(
                -1, *([1] * (vx.ndim - 1))) * vy
            vy_new = -ego_se.view(-1, *([1] *
                                        (vy.ndim - 1))) * vx + ego_ce.view(
                                            -1, *([1] * (vy.ndim - 1))) * vy
            traj[..., self.IDX_VX] = vx_new
            traj[..., self.IDX_VY] = vy_new

            # invalid 프레임은 끝까지 0 유지
            traj.masked_fill_(~valid.unsqueeze(-1), 0.0)

        ego_agent_past_sel = ego_agent_past[idx_keep]  # (Bk,Tp,11)
        nbr_past_sel = nbr_past[idx_keep]  # (Bk,A,Tp,11)
        ego_future_sel = ego_future_11[idx_keep]  # (Bk,Tf,11)
        nbr_future_sel = nbr_future_11[idx_keep]  # (Bk,A,Tf,11)

        ego_past_valid_sel = ego_past_valid[idx_keep]  # (Bk,Tp)
        nbr_past_valid_sel = nbr_past_valid[idx_keep]  # (Bk,A,Tp)
        ego_future_valid_sel = ego_future_valid[idx_keep]  # (Bk,Tf)
        nbr_future_valid_sel = nbr_future_valid[idx_keep]  # (Bk,A,Tf)

        _transform_agent_traj11_inplace(ego_agent_past_sel, ego_past_valid_sel)
        if A > 0:
            _transform_agent_traj11_inplace(nbr_past_sel, nbr_past_valid_sel)
        _transform_agent_traj11_inplace(ego_future_sel, ego_future_valid_sel)
        if A > 0:
            _transform_agent_traj11_inplace(nbr_future_sel,
                                            nbr_future_valid_sel)

        # 결과를 원본 텐서에 반영(copy 대신 view in-place라 이미 반영됨)
        ego_past_work = ego_agent_past_sel      # (Bk, Tp, 11)
        ego_fut_work = ego_future_sel           # (Bk, Tf, 11)
        if A > 0:
            nbr_past_work = nbr_past_sel  # (Bk, A, Tp, 11)
            nbr_fut_work = nbr_future_sel  # (Bk, A, Tf, 11)

        # (B) map / lanes / points 변환
        def _transform_points_inplace(key_pts: str, key_valid: str) -> None:
            pts = inputs.get(key_pts, None)
            v = inputs.get(key_valid, None)
            if not isinstance(pts, torch.Tensor):
                return
            if not isinstance(v, torch.Tensor):
                return
            pts_sel = pts[idx_keep]
            v_sel = _as_bool_mask(v)[idx_keep]  # (Bk,N)
            # pts_sel: (Bk,N,P,2)
            dx = pts_sel[..., 0] - ego_xe[:, None, None]
            dy = pts_sel[..., 1] - ego_ye[:, None, None]
            x_new = ego_ce[:, None, None] * dx + ego_se[:, None, None] * dy
            y_new = -ego_se[:, None, None] * dx + ego_ce[:, None, None] * dy
            pts_sel[..., 0] = x_new
            pts_sel[..., 1] = y_new
            pts_sel.masked_fill_(~v_sel[:, :, None, None], 0.0)
            pts[idx_keep] = pts_sel

        _transform_points_inplace("stop_sign_points", "stop_sign_is_valid")
        _transform_points_inplace("crosswalk_points", "crosswalk_is_valid")
        _transform_points_inplace("speed_bump_points", "speed_bump_is_valid")
        _transform_points_inplace("driveway_points", "driveway_is_valid")
        _transform_points_inplace("road_edge", "road_edge_is_valid")

        lanes = inputs.get("lanes", None)
        lanes_valid = inputs.get("lanes_len_is_valid", None)
        if isinstance(lanes, torch.Tensor) and isinstance(
                lanes_valid, torch.Tensor):
            lanes_sel = lanes[idx_keep]  # (Bk,L,20,12)
            lv_sel = _as_bool_mask(lanes_valid)[idx_keep]  # (Bk,L,20)

            # 0:2 점 -> 회전+이동
            dx = lanes_sel[..., 0] - ego_xe[:, None, None]
            dy = lanes_sel[..., 1] - ego_ye[:, None, None]
            x_new = ego_ce[:, None, None] * dx + ego_se[:, None, None] * dy
            y_new = -ego_se[:, None, None] * dx + ego_ce[:, None, None] * dy
            lanes_sel[..., 0] = x_new
            lanes_sel[..., 1] = y_new

            # 2:8 벡터 -> 회전만
            for s0, s1 in ((2, 4), (4, 6), (6, 8)):
                vx = lanes_sel[..., s0]
                vy = lanes_sel[..., s0 + 1]
                vx_new = ego_ce[:, None, None] * vx + ego_se[:, None, None] * vy
                vy_new = -ego_se[:, None, None] * vx + ego_ce[:, None,
                                                              None] * vy
                lanes_sel[..., s0] = vx_new
                lanes_sel[..., s0 + 1] = vy_new

            lanes_sel.masked_fill_(~lv_sel.unsqueeze(-1), 0.0)
            lanes[idx_keep] = lanes_sel

        static_objects = inputs.get("static_objects", None)
        static_valid = inputs.get("static_objects_is_valid", None)
        if isinstance(static_objects, torch.Tensor) and isinstance(
                static_valid, torch.Tensor):
            so = static_objects[idx_keep]  # (Bk,S,10)
            sv = _as_bool_mask(static_valid)[idx_keep]  # (Bk,S)

            dx = so[..., 0] - ego_xe[:, None]
            dy = so[..., 1] - ego_ye[:, None]
            x_new = ego_ce[:, None] * dx + ego_se[:, None] * dy
            y_new = -ego_se[:, None] * dx + ego_ce[:, None] * dy
            so[..., 0] = x_new
            so[..., 1] = y_new

            # (cos,sin) rotate only
            hc = so[..., 2]
            hs = so[..., 3]
            hc_new = ego_ce[:, None] * hc + ego_se[:, None] * hs
            hs_new = -ego_se[:, None] * hc + ego_ce[:, None] * hs
            hc_new, hs_new = _normalize_cos_sin(hc_new, hs_new)
            so[..., 2] = hc_new
            so[..., 3] = hs_new

            so.masked_fill_(~sv.unsqueeze(-1), 0.0)
            static_objects[idx_keep] = so


        # -----------------------------
        # Step 11/12) past/future quintic 보간 (증강된 agent만)
        # -----------------------------

        # 현재 acc(perturbed)도 new ego frame으로 회전해서 사용
        ego_acc_sel_old = ego_cur_acc_after[idx_keep]  # (Bk, 2)
        ego_ax_new = ego_ce * ego_acc_sel_old[:, 0] + ego_se * ego_acc_sel_old[:, 1]
        ego_ay_new = -ego_se * ego_acc_sel_old[:, 0] + ego_ce * ego_acc_sel_old[:, 1]
        ego_cur_acc_new = torch.stack([ego_ax_new, ego_ay_new], dim=-1)  # (Bk, 2)

        if A > 0:
            nbr_acc_sel_old = nbr_cur_acc_after[idx_keep]  # (Bk, A, 2)
            ax_new = ego_ce[:, None] * nbr_acc_sel_old[..., 0] + ego_se[:, None] * nbr_acc_sel_old[..., 1]
            ay_new = -ego_se[:, None] * nbr_acc_sel_old[..., 0] + ego_ce[:, None] * nbr_acc_sel_old[..., 1]
            nbr_cur_acc_new = torch.stack([ax_new, ay_new], dim=-1)  # (Bk, A, 2)
        else:
            nbr_cur_acc_new = torch.zeros((int(idx_keep.numel()), 0, 2), device=device, dtype=dtype)

        # 보간할 agent 마스크(선택된 샘플 내에서)
        aug_ego_sel = aug_ego[idx_keep]  # (Bk,)
        aug_nbr_sel = aug_nbr[idx_keep]  # (Bk, A)

        # 타입 판정(현재 onehot) - work 기준
        ego_cur_sel = ego_past_work[:, t_cur, :]  # (Bk, 11)
        ego_is_car_sel = ego_cur_sel[:, self.IDX_OH_CAR] > 0.5
        ego_is_ped_sel = ego_cur_sel[:, self.IDX_OH_PED] > 0.5
        ego_is_cyc_sel = ego_cur_sel[:, self.IDX_OH_CYC] > 0.5

        if A > 0:
            nbr_cur_sel = nbr_past_work[:, :, t_cur, :]  # (Bk, A, 11)
            nbr_is_car_sel = nbr_cur_sel[..., self.IDX_OH_CAR] > 0.5
            nbr_is_ped_sel = nbr_cur_sel[..., self.IDX_OH_PED] > 0.5
            nbr_is_cyc_sel = nbr_cur_sel[..., self.IDX_OH_CYC] > 0.5
        else:
            nbr_is_car_sel = torch.zeros((int(idx_keep.numel()), 0), device=device, dtype=torch.bool)
            nbr_is_ped_sel = torch.zeros((int(idx_keep.numel()), 0), device=device, dtype=torch.bool)
            nbr_is_cyc_sel = torch.zeros((int(idx_keep.numel()), 0), device=device, dtype=torch.bool)

        # -----------------------------
        # ✅ (중요 최적화) acc는 "전체 시퀀스"가 아니라, 필요한 tA/kT 1장만 계산
        # -----------------------------
        ego_v_past_sel = ego_past_work[..., self.IDX_VX:self.IDX_VY + 1]  # (Bk, Tp, 2)
        ego_v_fut_sel = ego_fut_work[..., self.IDX_VX:self.IDX_VY + 1]    # (Bk, Tf, 2)

        if A > 0:
            nbr_v_past_sel = nbr_past_work[..., self.IDX_VX:self.IDX_VY + 1]  # (Bk, A, Tp, 2)
            nbr_v_fut_sel = nbr_fut_work[..., self.IDX_VX:self.IDX_VY + 1]    # (Bk, A, Tf, 2)

        # ---- past/future 보간 helper (acc 스냅샷 버전) ----
        def _apply_past_interp_for_sel(
            traj_past: Tensor,      # (Bk, ..., Tp, 11)
            acc_tA: Tensor,         # (Bk, ..., 2)  tA에서의 가속도만
            cur_acc_new: Tensor,    # (Bk, ..., 2)
            sel_i: Tensor,          # (M,) long (flatten lead-dim)
            Np: int,
            Tsec: float,
        ) -> None:
            """과거 구간[t_cur-Np : t_cur]을 quintic으로 다시 만듭니다."""
            if int(Np) <= 0:
                return
            if t_cur - int(Np) < 0:
                return
            if int(sel_i.numel()) == 0:
                return

            Tloc = int(traj_past.shape[-2])
            traj_f = traj_past.reshape(-1, Tloc, 11)          # (flat, Tp, 11)
            accA_f = acc_tA.reshape(-1, 2)                    # (flat, 2)
            curacc_f = cur_acc_new.reshape(-1, 2)             # (flat, 2)

            tA = t_cur - int(Np)

            s_tr = traj_f.index_select(0, sel_i)              # (M, Tp, 11)
            s_aA = accA_f.index_select(0, sel_i)              # (M, 2)
            s_ca = curacc_f.index_select(0, sel_i)            # (M, 2)

            pA = s_tr[:, tA, 0:2]                             # (M, 2)
            vA = s_tr[:, tA, self.IDX_VX:self.IDX_VY + 1]     # (M, 2)
            aA = s_aA                                         # (M, 2)

            p0 = s_tr[:, t_cur, 0:2]                          # (M, 2)
            v0 = s_tr[:, t_cur, self.IDX_VX:self.IDX_VY + 1]  # (M, 2)
            a0 = s_ca                                         # (M, 2)

            p_seq, v_seq = self._quintic_2d(
                p0=pA, v0=vA, a0=aA,
                p1=p0, v1=v0, a1=a0,
                T=float(Tsec), N=int(Np),
            )  # p_seq/v_seq: (M, Np+1, 2)

            s_tr[:, tA:(t_cur + 1), 0:2] = p_seq
            s_tr[:, tA:(t_cur + 1), self.IDX_VX:self.IDX_VY + 1] = v_seq

            # 중간 프레임 heading만 속도 방향으로 맞춤 (끝점은 유지)
            if int(Np) >= 2:
                cos_old = s_tr[:, (tA + 1):t_cur, self.IDX_COS]  # (M, Np-1)
                sin_old = s_tr[:, (tA + 1):t_cur, self.IDX_SIN]  # (M, Np-1)
                vx_mid = v_seq[:, 1:-1, 0]                       # (M, Np-1)
                vy_mid = v_seq[:, 1:-1, 1]                       # (M, Np-1)

                cos_new, sin_new = _update_cos_sin_from_vxvy(
                    vx=vx_mid,
                    vy=vy_mid,
                    cos_old=cos_old,
                    sin_old=sin_old,
                    speed_eps=1e-3,
                )
                s_tr[:, (tA + 1):t_cur, self.IDX_COS] = cos_new
                s_tr[:, (tA + 1):t_cur, self.IDX_SIN] = sin_new

            traj_f.index_copy_(0, sel_i, s_tr)

        def _apply_future_interp_for_sel(
            traj_future: Tensor,    # (Bk, ..., Tf, 11)
            acc_kT: Tensor,         # (Bk, ..., 2)  kT에서의 가속도만
            cur_past: Tensor,       # (Bk, ..., 11)
            cur_acc_new: Tensor,    # (Bk, ..., 2)
            sel_i: Tensor,          # (M,) long (flatten lead-dim)
            Nf: int,
            Tsec: float,
        ) -> None:
            """미래 구간[0 : Nf-1]을 quintic으로 다시 만듭니다."""
            if int(Nf) <= 0:
                return
            if int(Nf) > int(traj_future.shape[-2]):
                return
            if int(sel_i.numel()) == 0:
                return

            Tloc = int(traj_future.shape[-2])
            fut_f = traj_future.reshape(-1, Tloc, 11)         # (flat, Tf, 11)
            accT_f = acc_kT.reshape(-1, 2)                    # (flat, 2)
            cur_f = cur_past.reshape(-1, 11)                  # (flat, 11)
            curacc_f = cur_acc_new.reshape(-1, 2)             # (flat, 2)

            kT = int(Nf) - 1

            s_fut = fut_f.index_select(0, sel_i)              # (M, Tf, 11)
            s_aT = accT_f.index_select(0, sel_i)              # (M, 2)
            s_cur = cur_f.index_select(0, sel_i)              # (M, 11)
            s_ca = curacc_f.index_select(0, sel_i)            # (M, 2)

            p0 = s_cur[:, 0:2]                                # (M, 2)
            v0 = s_cur[:, self.IDX_VX:self.IDX_VY + 1]        # (M, 2)
            a0 = s_ca                                         # (M, 2)

            pT = s_fut[:, kT, 0:2]                            # (M, 2)
            vT = s_fut[:, kT, self.IDX_VX:self.IDX_VY + 1]    # (M, 2)
            aT = s_aT                                         # (M, 2)

            p_seq, v_seq = self._quintic_2d(
                p0=p0, v0=v0, a0=a0,
                p1=pT, v1=vT, a1=aT,
                T=float(Tsec), N=int(Nf),
            )  # p_seq/v_seq: (M, Nf+1, 2)

            # future에는 현재(t=0)가 없으므로, 1..Nf를 0..Nf-1에 저장
            s_fut[:, :int(Nf), 0:2] = p_seq[:, 1:]
            s_fut[:, :int(Nf), self.IDX_VX:self.IDX_VY + 1] = v_seq[:, 1:]

            # 중간 프레임 heading만 속도 방향으로 맞춤 (마지막 프레임은 유지)
            if int(Nf) >= 2:
                cos_old = s_fut[:, :(int(Nf) - 1), self.IDX_COS]  # (M, Nf-1)
                sin_old = s_fut[:, :(int(Nf) - 1), self.IDX_SIN]  # (M, Nf-1)
                vx_mid = v_seq[:, 1:-1, 0]                        # (M, Nf-1)
                vy_mid = v_seq[:, 1:-1, 1]                        # (M, Nf-1)

                cos_new, sin_new = _update_cos_sin_from_vxvy(
                    vx=vx_mid,
                    vy=vy_mid,
                    cos_old=cos_old,
                    sin_old=sin_old,
                    speed_eps=1e-3,
                )
                s_fut[:, :(int(Nf) - 1), self.IDX_COS] = cos_new
                s_fut[:, :(int(Nf) - 1), self.IDX_SIN] = sin_new

            fut_f.index_copy_(0, sel_i, s_fut)

        # ---- (1) ego: 증강된 ego 인덱스를 1번만 뽑고, 타입으로 나눠서 재사용 ----
        sel_ego_all = torch.nonzero(aug_ego_sel, as_tuple=True)[0]  # (Me,)
        sel_ego_car = sel_ego_all[ego_is_car_sel[sel_ego_all]]
        sel_ego_cyc = sel_ego_all[ego_is_cyc_sel[sel_ego_all]]
        sel_ego_ped = sel_ego_all[ego_is_ped_sel[sel_ego_all]]

        # 타입별 past tA / future kT (인덱스)
        tA_car = t_cur - int(Np_car)
        tA_cyc = t_cur - int(Np_cyc)
        tA_ped = t_cur - int(Np_ped)

        kT_car = int(Nf_car) - 1
        kT_cyc = int(Nf_cyc) - 1
        kT_ped = int(Nf_ped) - 1

        # ✅ ego past tA 가속도(필요한 타입만)
        ego_acc_tA_car = _compute_acc_at_t_from_vxvy(ego_v_past_sel, ego_past_valid_sel, dt=self._dt, t=int(tA_car)) \
            if int(sel_ego_car.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)
        ego_acc_tA_cyc = _compute_acc_at_t_from_vxvy(ego_v_past_sel, ego_past_valid_sel, dt=self._dt, t=int(tA_cyc)) \
            if int(sel_ego_cyc.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)
        ego_acc_tA_ped = _compute_acc_at_t_from_vxvy(ego_v_past_sel, ego_past_valid_sel, dt=self._dt, t=int(tA_ped)) \
            if int(sel_ego_ped.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)

        _apply_past_interp_for_sel(
            traj_past=ego_past_work,
            acc_tA=ego_acc_tA_car,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_car,
            Np=Np_car,
            Tsec=float(Np_car) * self._dt,
        )
        _apply_past_interp_for_sel(
            traj_past=ego_past_work,
            acc_tA=ego_acc_tA_cyc,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_cyc,
            Np=Np_cyc,
            Tsec=float(Np_cyc) * self._dt,
        )
        _apply_past_interp_for_sel(
            traj_past=ego_past_work,
            acc_tA=ego_acc_tA_ped,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_ped,
            Np=Np_ped,
            Tsec=float(Np_ped) * self._dt,
        )

        ego_cur_new_frame = ego_past_work[:, t_cur, :]  # (Bk, 11)

        # ✅ ego future kT 가속도(필요한 타입만)
        ego_acc_kT_car = _compute_acc_at_t_from_vxvy(ego_v_fut_sel, ego_future_valid_sel, dt=self._dt, t=int(kT_car)) \
            if int(sel_ego_car.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)
        ego_acc_kT_cyc = _compute_acc_at_t_from_vxvy(ego_v_fut_sel, ego_future_valid_sel, dt=self._dt, t=int(kT_cyc)) \
            if int(sel_ego_cyc.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)
        ego_acc_kT_ped = _compute_acc_at_t_from_vxvy(ego_v_fut_sel, ego_future_valid_sel, dt=self._dt, t=int(kT_ped)) \
            if int(sel_ego_ped.numel()) != 0 else torch.zeros((int(idx_keep.numel()), 2), device=device, dtype=dtype)

        _apply_future_interp_for_sel(
            traj_future=ego_fut_work,
            acc_kT=ego_acc_kT_car,
            cur_past=ego_cur_new_frame,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_car,
            Nf=Nf_car,
            Tsec=float(Nf_car) * self._dt,
        )
        _apply_future_interp_for_sel(
            traj_future=ego_fut_work,
            acc_kT=ego_acc_kT_cyc,
            cur_past=ego_cur_new_frame,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_cyc,
            Nf=Nf_cyc,
            Tsec=float(Nf_cyc) * self._dt,
        )
        _apply_future_interp_for_sel(
            traj_future=ego_fut_work,
            acc_kT=ego_acc_kT_ped,
            cur_past=ego_cur_new_frame,
            cur_acc_new=ego_cur_acc_new,
            sel_i=sel_ego_ped,
            Nf=Nf_ped,
            Tsec=float(Nf_ped) * self._dt,
        )

        # ---- (2) neighbor: 증강된 neighbor 인덱스를 1번만 뽑고, 타입으로 나눠서 재사용 ----
        if A > 0:
            aug_nbr_flat = aug_nbr_sel.reshape(-1)  # (Bk*A,)
            sel_nbr_all = torch.nonzero(aug_nbr_flat, as_tuple=True)[0]  # (Mn,)

            nbr_is_car_flat = nbr_is_car_sel.reshape(-1)
            nbr_is_cyc_flat = nbr_is_cyc_sel.reshape(-1)
            nbr_is_ped_flat = nbr_is_ped_sel.reshape(-1)

            sel_nbr_car = sel_nbr_all[nbr_is_car_flat[sel_nbr_all]]
            sel_nbr_cyc = sel_nbr_all[nbr_is_cyc_flat[sel_nbr_all]]
            sel_nbr_ped = sel_nbr_all[nbr_is_ped_flat[sel_nbr_all]]

            # neighbor past tA 가속도(필요한 타입만)
            nbr_acc_tA_car = _compute_acc_at_t_from_vxvy(nbr_v_past_sel, nbr_past_valid_sel, dt=self._dt, t=int(tA_car)) \
                if int(sel_nbr_car.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)
            nbr_acc_tA_cyc = _compute_acc_at_t_from_vxvy(nbr_v_past_sel, nbr_past_valid_sel, dt=self._dt, t=int(tA_cyc)) \
                if int(sel_nbr_cyc.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)
            nbr_acc_tA_ped = _compute_acc_at_t_from_vxvy(nbr_v_past_sel, nbr_past_valid_sel, dt=self._dt, t=int(tA_ped)) \
                if int(sel_nbr_ped.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)

            _apply_past_interp_for_sel(
                traj_past=nbr_past_work,
                acc_tA=nbr_acc_tA_car,
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_car,
                Np=Np_car,
                Tsec=float(Np_car) * self._dt,
            )
            _apply_past_interp_for_sel(
                traj_past=nbr_past_work,
                acc_tA=nbr_acc_tA_cyc,
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_cyc,
                Np=Np_cyc,
                Tsec=float(Np_cyc) * self._dt,
            )
            _apply_past_interp_for_sel(
                traj_past=nbr_past_work,
                acc_tA=nbr_acc_tA_ped,
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_ped,
                Np=Np_ped,
                Tsec=float(Np_ped) * self._dt,
            )

            # neighbor future kT 가속도(필요한 타입만)
            nbr_acc_kT_car = _compute_acc_at_t_from_vxvy(nbr_v_fut_sel, nbr_future_valid_sel, dt=self._dt, t=int(kT_car)) \
                if int(sel_nbr_car.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)
            nbr_acc_kT_cyc = _compute_acc_at_t_from_vxvy(nbr_v_fut_sel, nbr_future_valid_sel, dt=self._dt, t=int(kT_cyc)) \
                if int(sel_nbr_cyc.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)
            nbr_acc_kT_ped = _compute_acc_at_t_from_vxvy(nbr_v_fut_sel, nbr_future_valid_sel, dt=self._dt, t=int(kT_ped)) \
                if int(sel_nbr_ped.numel()) != 0 else torch.zeros((int(idx_keep.numel()), int(A), 2), device=device, dtype=dtype)

            _apply_future_interp_for_sel(
                traj_future=nbr_fut_work,
                acc_kT=nbr_acc_kT_car,
                cur_past=nbr_past_work[:, :, t_cur, :],
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_car,
                Nf=Nf_car,
                Tsec=float(Nf_car) * self._dt,
            )
            _apply_future_interp_for_sel(
                traj_future=nbr_fut_work,
                acc_kT=nbr_acc_kT_cyc,
                cur_past=nbr_past_work[:, :, t_cur, :],
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_cyc,
                Nf=Nf_cyc,
                Tsec=float(Nf_cyc) * self._dt,
            )
            _apply_future_interp_for_sel(
                traj_future=nbr_fut_work,
                acc_kT=nbr_acc_kT_ped,
                cur_past=nbr_past_work[:, :, t_cur, :],
                cur_acc_new=nbr_cur_acc_new,
                sel_i=sel_nbr_ped,
                Nf=Nf_ped,
                Tsec=float(Nf_ped) * self._dt,
            )

        # ego current 정렬 강제(안전) - ✅ work에 적용 후 마지막에 write-back
        ego_past_work[:, t_cur, self.IDX_X] = 0.0
        ego_past_work[:, t_cur, self.IDX_Y] = 0.0
        ego_past_work[:, t_cur, self.IDX_COS] = 1.0
        ego_past_work[:, t_cur, self.IDX_SIN] = 0.0

        # ✅ 보간 결과를 원본 텐서에 "반드시" 반영 (여기서 1번만)
        ego_agent_past[idx_keep] = ego_past_work
        ego_future_11[idx_keep] = ego_fut_work
        if A > 0:
            nbr_past[idx_keep] = nbr_past_work
            nbr_future_11[idx_keep] = nbr_fut_work
        # -----------------------------
        # Step 14) 파생 데이터 재생성 (neighbor=near 강제)
        # -----------------------------
        # ego_future_gt_4_dim (outputs)
        ego_gt4 = ego_future_11[..., :4].clone()  # (B,Tf,4)
        ego_gt4.masked_fill_(~ego_future_valid.unsqueeze(-1), 0.0)
        outputs["ego_future_gt_4_dim"] = ego_gt4

        # near/neighbor future 3/4
        # neighbor_future_gt_3_dim / near_future_gt_3_dim 갱신
        if "neighbor_future_gt_3_dim" in inputs and isinstance(
                inputs["neighbor_future_gt_3_dim"], torch.Tensor):
            nbr_yaw = torch.atan2(nbr_future_11[..., self.IDX_SIN],
                                  nbr_future_11[..., self.IDX_COS])
            nbr_gt3 = torch.stack([
                nbr_future_11[..., self.IDX_X], nbr_future_11[..., self.IDX_Y],
                nbr_yaw
            ],
                                  dim=-1)  # (B,A,Tf,3)
            nbr_gt3.masked_fill_(~nbr_future_valid.unsqueeze(-1), 0.0)
            inputs["neighbor_future_gt_3_dim"] = nbr_gt3

        if "near_future_gt_3_dim" in inputs and isinstance(
                inputs["near_future_gt_3_dim"], torch.Tensor):
            # neighbor = near
            inputs["near_future_gt_3_dim"] = inputs.get(
                "neighbor_future_gt_3_dim", inputs["near_future_gt_3_dim"])

        # near_future_gt_4_dim (outputs)
        near_gt4 = nbr_future_11[..., :4].clone()  # (B,A,Tf,4)
        near_valid = _as_bool_mask(
            outputs["near_future_gt_is_valid"])  # (B,A,Tf)
        near_gt4.masked_fill_(~near_valid.unsqueeze(-1), 0.0)
        outputs["near_future_gt_4_dim"] = near_gt4

        # ego_future_gt_3_dim이 inputs에 있으면 갱신
        if "ego_future_gt_3_dim" in inputs and isinstance(
                inputs["ego_future_gt_3_dim"], torch.Tensor):
            ego_yaw = torch.atan2(ego_future_11[..., self.IDX_SIN],
                                  ego_future_11[..., self.IDX_COS])
            ego_gt3 = torch.stack([
                ego_future_11[..., self.IDX_X], ego_future_11[..., self.IDX_Y],
                ego_yaw
            ],
                                  dim=-1)
            ego_gt3.masked_fill_(~ego_future_valid.unsqueeze(-1), 0.0)
            inputs["ego_future_gt_3_dim"] = ego_gt3

        # near_agents_past = neighbor_agents_past (reference로 맞춤)
        inputs["near_agents_past"] = inputs["neighbor_agents_past"]
        if "near_agents_past_is_valid" in inputs and "neighbor_agents_past_is_valid" in inputs:
            inputs["near_agents_past_is_valid"] = inputs[
                "neighbor_agents_past_is_valid"]
        if "near_agents_is_valid" in inputs and "neighbor_agents_is_valid" in inputs:
            inputs["near_agents_is_valid"] = inputs["neighbor_agents_is_valid"]
        if "near_future_gt_is_valid" in outputs and "neighbor_future_gt_is_valid" in inputs:
            # outputs에는 이미 near_future_gt_is_valid가 있으므로 그대로 둠
            pass
        # -----------------------------
        # Step 14-3) seg control 재계산(torch) + valid 재생성
        #   ✅ 전제: past/future seg control은 항상 ego 포함 (P = 1 + A)
        #   ✅ 최적화: 증강이 실제 적용된 샘플(idx_keep)만 재계산 후 write-back
        # -----------------------------
        past_ctrl = inputs.get("past_seg_control_gt_3_dim", None)
        past_ctrl_valid = inputs.get("past_seg_control_is_valid", None)

        if isinstance(past_ctrl, torch.Tensor) and isinstance(
                past_ctrl_valid, torch.Tensor):
            # 기대 shape:
            #   past_ctrl       : (B, 1+A, Tp-1, 3)
            #   past_ctrl_valid : (B, 1+A, Tp-1)
            expected_p = 1 + int(A)
            if int(past_ctrl.shape[1]) != expected_p:
                raise ValueError(
                    f"past_seg_control_gt_3_dim agent dim must be 1+agent_num. "
                    f"got P={int(past_ctrl.shape[1])}, expected {expected_p}")
            if int(past_ctrl_valid.shape[1]) != expected_p:
                raise ValueError(
                    f"past_seg_control_is_valid agent dim must be 1+agent_num. "
                    f"got P={int(past_ctrl_valid.shape[1])}, expected {expected_p}"
                )

            # ✅ idx_keep 샘플만 재계산
            ego_past11_sel = ego_agent_past[idx_keep]  # (Bk, Tp, 11)
            nbr_past11_sel = nbr_past[idx_keep]  # (Bk, A, Tp, 11)
            ego_past_valid_sel2 = ego_past_valid[idx_keep]  # (Bk, Tp)
            nbr_past_valid_sel2 = nbr_past_valid[idx_keep]  # (Bk, A, Tp)

            all_past11_sel = torch.cat(
                [ego_past11_sel.unsqueeze(1), nbr_past11_sel],
                dim=1)  # (Bk, 1+A, Tp, 11)
            all_past_valid_sel = torch.cat(
                [ego_past_valid_sel2.unsqueeze(1), nbr_past_valid_sel2],
                dim=1)  # (Bk, 1+A, Tp)

            all_pose3_sel = self._traj11_to_pose3(
                all_past11_sel)  # (Bk, 1+A, Tp, 3)
            controls_sel, seg_valid_sel = self._pose3_to_control3(
                all_pose3_sel,
                frame_valid=_as_bool_mask(all_past_valid_sel),
                dt=self._dt,
                use_body_vel=bool(use_body_vel),
            )  # controls_sel: (Bk, 1+A, Tp-1, 3), seg_valid_sel: (Bk, 1+A, Tp-1)

            # write-back (dtype 유지)
            past_ctrl[idx_keep] = controls_sel.to(dtype=past_ctrl.dtype)
            if past_ctrl_valid.dtype == torch.bool:
                past_ctrl_valid[idx_keep] = seg_valid_sel
            else:
                past_ctrl_valid[idx_keep] = seg_valid_sel.to(dtype=past_ctrl_valid.dtype)

        # future_seg_control (outputs)
        fut_ctrl = outputs.get("future_seg_control_gt_3_dim", None)
        fut_ctrl_valid = outputs.get("future_seg_control_is_valid", None)

        if isinstance(fut_ctrl, torch.Tensor) and isinstance(
                fut_ctrl_valid, torch.Tensor):
            # 기대 shape:
            #   fut_ctrl       : (B, 1+A, Tf, 3)
            #   fut_ctrl_valid : (B, 1+A, Tf)
            expected_p = 1 + int(A)
            if int(fut_ctrl.shape[1]) != expected_p:
                raise ValueError(
                    f"future_seg_control_gt_3_dim agent dim must be 1+agent_num. "
                    f"got P={int(fut_ctrl.shape[1])}, expected {expected_p}")
            if int(fut_ctrl_valid.shape[1]) != expected_p:
                raise ValueError(
                    f"future_seg_control_is_valid agent dim must be 1+agent_num. "
                    f"got P={int(fut_ctrl_valid.shape[1])}, expected {expected_p}"
                )

            # ✅ idx_keep 샘플만 재계산
            ego_past11_sel = ego_agent_past[idx_keep]  # (Bk, Tp, 11)
            ego_fut11_sel = ego_future_11[idx_keep]  # (Bk, Tf, 11)
            nbr_past11_sel = nbr_past[idx_keep]  # (Bk, A, Tp, 11)
            nbr_fut11_sel = nbr_future_11[idx_keep]  # (Bk, A, Tf, 11)

            ego_past_valid_sel2 = ego_past_valid[idx_keep]  # (Bk, Tp)
            ego_fut_valid_sel2 = ego_future_valid[idx_keep]  # (Bk, Tf)
            nbr_past_valid_sel2 = nbr_past_valid[idx_keep]  # (Bk, A, Tp)
            nbr_fut_valid_sel2 = nbr_future_valid[idx_keep]  # (Bk, A, Tf)

            # (current + future) node sequence: length = 1+Tf
            ego_all11_sel = torch.cat(
                [ego_past11_sel[:, -1:, :], ego_fut11_sel],
                dim=1)  # (Bk, 1+Tf, 11)
            ego_all_valid_sel = torch.cat(
                [ego_past_valid_sel2[:, -1:], ego_fut_valid_sel2],
                dim=1)  # (Bk, 1+Tf)

            if int(nbr_past11_sel.shape[1]) > 0:
                nbr_all11_sel = torch.cat(
                    [nbr_past11_sel[:, :, -1:, :], nbr_fut11_sel],
                    dim=2)  # (Bk, A, 1+Tf, 11)
                nbr_all_valid_sel = torch.cat(
                    [nbr_past_valid_sel2[:, :, -1:], nbr_fut_valid_sel2],
                    dim=2)  # (Bk, A, 1+Tf)
            else:
                # A=0 케이스 방어 (실제로는 거의 없겠지만 안전)
                Bk = int(idx_keep.numel())
                nbr_all11_sel = torch.zeros(
                    (Bk, 0, 1 + int(ego_fut11_sel.shape[1]), 11),
                    device=ego_all11_sel.device,
                    dtype=ego_all11_sel.dtype)
                nbr_all_valid_sel = torch.zeros(
                    (Bk, 0, 1 + int(ego_fut11_sel.shape[1])),
                    device=ego_all11_sel.device,
                    dtype=torch.bool)

            all11_sel = torch.cat([ego_all11_sel.unsqueeze(1), nbr_all11_sel],
                                  dim=1)  # (Bk, 1+A, 1+Tf, 11)
            allv_sel = torch.cat(
                [ego_all_valid_sel.unsqueeze(1), nbr_all_valid_sel],
                dim=1)  # (Bk, 1+A, 1+Tf)

            pose3_sel = self._traj11_to_pose3(all11_sel)  # (Bk, 1+A, 1+Tf, 3)
            controls_sel, seg_valid_sel = self._pose3_to_control3(
                pose3_sel,
                frame_valid=_as_bool_mask(allv_sel),
                dt=self._dt,
                use_body_vel=bool(use_body_vel),
            )  # controls_sel: (Bk, 1+A, Tf, 3)

            fut_ctrl[idx_keep] = controls_sel.to(dtype=fut_ctrl.dtype)
            if fut_ctrl_valid.dtype == torch.bool:
                fut_ctrl_valid[idx_keep] = seg_valid_sel
            else:
                fut_ctrl_valid[idx_keep] = seg_valid_sel.to(dtype=fut_ctrl_valid.dtype)
        # -----------------------------
        # Step 15) origin_world_pose 업데이트(ego가 증강된 샘플만)
        # -----------------------------
        origin_world_pose = inputs.get("origin_world_pose", None)
        if isinstance(origin_world_pose, torch.Tensor):
            ego_aug_samples = aug_ego  # (B,)

            delta_px = ego_cur_after[:, self.IDX_X] - ego_cur[:, self.IDX_X]
            delta_py = ego_cur_after[:, self.IDX_Y] - ego_cur[:, self.IDX_Y]

            yaw_old = torch.atan2(ego_cur[:, self.IDX_SIN],
                                  ego_cur[:, self.IDX_COS])
            yaw_new = torch.atan2(ego_cur_after[:, self.IDX_SIN],
                                  ego_cur_after[:, self.IDX_COS])
            delta_yaw = _wrap_to_pi(yaw_new - yaw_old)

            twx = origin_world_pose[:, 0]
            twy = origin_world_pose[:, 1]
            cw = origin_world_pose[:, 2]
            sw = origin_world_pose[:, 3]
            cw, sw = _normalize_cos_sin(cw, sw)

            dtx, dty = _rot_vec_by_yaw(delta_px, delta_py, cw, sw)
            twx_new = twx + dtx
            twy_new = twy + dty

            cdy = torch.cos(delta_yaw)
            sdy = torch.sin(delta_yaw)
            cw_new = cw * cdy - sw * sdy
            sw_new = sw * cdy + cw * sdy
            cw_new, sw_new = _normalize_cos_sin(cw_new, sw_new)

            origin_world_pose[:, 0] = torch.where(ego_aug_samples, twx_new,
                                                  origin_world_pose[:, 0])
            origin_world_pose[:, 1] = torch.where(ego_aug_samples, twy_new,
                                                  origin_world_pose[:, 1])
            origin_world_pose[:, 2] = torch.where(ego_aug_samples, cw_new,
                                                  origin_world_pose[:, 2])
            origin_world_pose[:, 3] = torch.where(ego_aug_samples, sw_new,
                                                  origin_world_pose[:, 3])
        # 끝: inputs/outputs in-place 갱신 완료
        # 끝: inputs/outputs in-place 갱신 완료 직전(맨 마지막 return 직전)에 아래 한 줄 추가
        # ✅ debug_enabled가 True일 때만 저장 루틴 호출 (False면 호출 자체가 없음)
        if debug_enabled:
            self._maybe_save_augmented_debug_png(
                inputs=inputs,
                outputs=outputs,
                idx_keep=idx_keep,  # (Bk,)
                aug_ego=aug_ego,  # (B,)
                aug_nbr=aug_nbr,  # (B,A)
                debug_vis_dir=debug_vis_dir,
                debug_step=debug_step,
                debug_max_scenes=int(debug_max_scenes),
                debug_every_n_steps=int(debug_every_n_steps),
                past_stride=int(debug_past_stride),
                future_stride=int(debug_future_stride),
                vel_arrow_len_m=float(debug_vel_arrow_len_m),
                debug_before_cache=debug_before_cache,
                debug_frame_params_cache=debug_frame_params_cache,
            )
        return
