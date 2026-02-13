from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

ArrayF = NDArray[np.floating]


def _to_scalar_dt(value: Union[float, np.ndarray], ref: NDArray[np.generic]) -> np.floating:
    """dt를 ref와 같은 dtype의 스칼라로 정리합니다.

    Args:
        value (float | np.ndarray):
            시간 간격 dt (스칼라).
            - float 또는 shape=() 혹은 size=1인 배열도 허용합니다.
        ref (np.ndarray):
            dtype 기준 배열.

    Returns:
        np.floating:
            ref.dtype로 맞춘 스칼라 dt.

    Raises:
        ValueError:
            dt가 스칼라가 아닐 때 발생합니다.
    """
    dt_arr = np.asarray(value, dtype=ref.dtype)
    if dt_arr.size != 1:
        raise ValueError(f"dt는 스칼라여야 합니다. got shape={dt_arr.shape}, size={dt_arr.size}")
    return dt_arr.reshape(()).item()


def _wrap_to_pi(delta: ArrayF) -> ArrayF:
    """각도 차이를 (-pi, pi] 범위로 접습니다.

    Args:
        delta (np.ndarray): 각도 차이. shape 자유.

    Returns:
        np.ndarray: (-pi, pi] 범위로 접힌 각도 차이. shape는 입력과 동일.
    """
    return np.arctan2(np.sin(delta), np.cos(delta)).astype(delta.dtype, copy=False)


def _normalize_cos_sin(cos_seq: ArrayF, sin_seq: ArrayF, eps: float) -> Tuple[ArrayF, ArrayF]:
    """(cos, sin) 쌍을 길이 1이 되도록 정리합니다.

    Args:
        cos_seq (np.ndarray): cos 값들. shape 자유.
        sin_seq (np.ndarray): sin 값들. shape는 cos_seq와 동일.
        eps (float): 0으로 나누는 것을 피하기 위한 작은 값.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            (cos_norm, sin_norm) - 입력과 동일한 shape
    """
    r = np.sqrt(cos_seq * cos_seq + sin_seq * sin_seq + eps).astype(cos_seq.dtype, copy=False)
    return (cos_seq / r).astype(cos_seq.dtype, copy=False), (sin_seq / r).astype(sin_seq.dtype, copy=False)


def differentiate_numpy_pose3_to_control3(
    cur_future_pose_gt_3_dim: ArrayF,  # (P, 1+T, 3) = (x, y, heading)
    dt: Union[float, np.ndarray],
    *,
    eps: float = 1e-8,
    normalize_yaw: bool = True,
    wrap_heading: bool = True,
) -> ArrayF:
    """(x,y,heading) 궤적에서 구간별 제어(vx_b, vy_b, omega)를 차분으로 복원합니다.

    입력:
        cur_future_pose_gt_3_dim: (P, 1+T, 3)
            - 마지막 3은 (x, y, heading[rad]) 입니다.
            - 시간축은 k=0..T (총 1+T개 상태)

    출력:
        future_seg_control_gt_3_dim: (P, T, 3)
            - 마지막 3은 (v_x^b, v_y^b, omega) 입니다.
            - 시간축은 구간 k=0..T-1 (총 T개 구간)
    """
    pose = np.asarray(cur_future_pose_gt_3_dim)
    if pose.ndim != 3 or int(pose.shape[-1]) != 3:
        raise ValueError(
            "cur_future_pose_gt_3_dim은 (P, 1+T, 3) 3D 배열이어야 합니다. "
            f"got shape={pose.shape}"
        )

    # float dtype 강제(삼각함수/나눗셈 안정)
    pose = pose.astype(np.float32 if pose.dtype.kind != "f" else pose.dtype, copy=False)

    _, time_len, _ = pose.shape  # last dim=3
    T = int(time_len - 1)
    if T <= 0:
        raise ValueError(f"time_len(=1+T)은 최소 2여야 합니다. got time_len={time_len}")

    dt_s = _to_scalar_dt(dt, ref=pose)
    if (not np.isfinite(dt_s)) or float(dt_s) <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt_s}")

    # 분해: (P, 1+T)
    x = pose[..., 0]
    y = pose[..., 1]
    heading = pose[..., 2]

    # 구간별 slice: (P, T)
    x0, x1 = x[..., :-1], x[..., 1:]
    y0, y1 = y[..., :-1], y[..., 1:]
    th0, th1 = heading[..., :-1], heading[..., 1:]

    # 1) Δθ, omega
    delta_theta = (th1 - th0).astype(pose.dtype, copy=False)  # (P, T)
    if wrap_heading:
        delta_theta = _wrap_to_pi(delta_theta)  # (P, T)
    omega = (delta_theta / dt_s).astype(pose.dtype, copy=False)  # (P, T)

    # 2) 중간 방향 -> (cos, sin)
    th_mid = (th0 + 0.5 * delta_theta).astype(pose.dtype, copy=False)  # (P, T)
    cos_mid = np.cos(th_mid).astype(pose.dtype, copy=False)            # (P, T)
    sin_mid = np.sin(th_mid).astype(pose.dtype, copy=False)            # (P, T)

    if normalize_yaw:
        cos_mid, sin_mid = _normalize_cos_sin(cos_mid, sin_mid, eps=float(eps))

    # 3) 지도 기준 속도
    vwx = ((x1 - x0) / dt_s).astype(pose.dtype, copy=False)  # (P, T)
    vwy = ((y1 - y0) / dt_s).astype(pose.dtype, copy=False)  # (P, T)

    # 4) 지도 -> 차량(몸체) (중간 방향으로 거꾸로 회전)
    vx_b = (cos_mid * vwx + sin_mid * vwy).astype(pose.dtype, copy=False)     # (P, T)
    vy_b = (-sin_mid * vwx + cos_mid * vwy).astype(pose.dtype, copy=False)   # (P, T)

    # 출력: (P, T, 3)
    future_seg_control_gt_3_dim = np.stack([vx_b, vy_b, omega], axis=-1).astype(pose.dtype, copy=False)
    return future_seg_control_gt_3_dim


def _traj11_to_traj3_heading(traj_11: ArrayF) -> ArrayF:
    """11차원 궤적을 (x, y, heading) 3차원으로 바꿉니다.

    Args:
        traj_11 (np.ndarray):
            - (T, 11) 또는 (N, T, 11)
            - 11차원 = [x, y, cos, sin, vx, vy, width, length, onehot(3)]

    Returns:
        np.ndarray:
            - (T, 3) 또는 (N, T, 3)
            - 3차원 = [x, y, heading]
    """
    arr = np.asarray(traj_11)
    if arr.ndim == 2:
        if arr.shape[-1] != 11:
            raise ValueError(f"traj_11 마지막 차원은 11이어야 합니다. got {arr.shape}")
        heading = np.arctan2(arr[:, 3], arr[:, 2]).astype(np.float32, copy=False)
        return np.stack([arr[:, 0], arr[:, 1], heading], axis=-1).astype(np.float32, copy=False)
    if arr.ndim == 3:
        if arr.shape[-1] != 11:
            raise ValueError(f"traj_11 마지막 차원은 11이어야 합니다. got {arr.shape}")
        heading = np.arctan2(arr[:, :, 3], arr[:, :, 2]).astype(np.float32, copy=False)
        return np.stack([arr[:, :, 0], arr[:, :, 1], heading], axis=-1).astype(np.float32, copy=False)
    raise ValueError(f"traj_11은 (T,11) 또는 (N,T,11) 이어야 합니다. got {arr.shape}")


def _get_ego_cur_future_gt_3_dim(
    ego_current_4_dim: ArrayF,  # (4,) = x,y,cos,sin
    ego_future_gt_3_dim: ArrayF,  # (Tf, 3) = x,y,heading
    *,
    eps: float = 1e-8,
) -> ArrayF:
    """ego의 현재(x,y,cos,sin)와 미래(x,y,heading)를 이어붙여 (1+Tf,3)을 만듭니다.

    규칙:
        - 현재가 무효(전부 0이면)면, 결과 전체를 0으로 둡니다.

    Args:
        ego_current_4_dim (np.ndarray): shape (4,)
        ego_future_gt_3_dim (np.ndarray): shape (Tf, 3)
        eps (float): 0 판정 기준(아주 작은 값)

    Returns:
        np.ndarray: shape (1+Tf, 3)
    """
    cur4 = np.asarray(ego_current_4_dim).astype(np.float32, copy=False)
    fut3 = np.asarray(ego_future_gt_3_dim).astype(np.float32, copy=False)

    if cur4.shape != (4,):
        raise ValueError(f"ego_current_4_dim shape는 (4,) 이어야 합니다. got {cur4.shape}")
    if fut3.ndim != 2 or fut3.shape[-1] != 3:
        raise ValueError(f"ego_future_gt_3_dim shape는 (Tf,3) 이어야 합니다. got {fut3.shape}")

    Tf = int(fut3.shape[0])
    out = np.zeros((1 + Tf, 3), dtype=np.float32)

    is_valid = bool((np.abs(cur4) > eps).any())
    if not is_valid:
        return out

    heading0 = float(np.arctan2(cur4[3], cur4[2]))
    out[0, 0] = float(cur4[0])
    out[0, 1] = float(cur4[1])
    out[0, 2] = heading0

    if Tf > 0:
        out[1:, :] = fut3
    return out


def _get_neighbor_cur_future_gt_3_dim(
    neighbor_agents_current_4_dim: ArrayF,  # (N,4)=x,y,cos,sin
    neighbor_future_gt_3_dim: ArrayF,  # (N,Tf,3)=x,y,heading
    *,
    eps: float = 1e-8,
) -> ArrayF:
    """neighbor의 현재(x,y,cos,sin)와 미래(x,y,heading)를 이어붙여 (N,1+Tf,3)을 만듭니다.

    규칙:
        - 각 agent별로 현재가 무효(전부 0)이면, 그 agent의 결과 전체를 0으로 둡니다.

    Args:
        neighbor_agents_current_4_dim (np.ndarray): shape (N,4)
        neighbor_future_gt_3_dim (np.ndarray): shape (N,Tf,3)
        eps (float): 0 판정 기준(아주 작은 값)

    Returns:
        np.ndarray: shape (N, 1+Tf, 3)
    """
    cur4 = np.asarray(neighbor_agents_current_4_dim).astype(np.float32, copy=False)
    fut3 = np.asarray(neighbor_future_gt_3_dim).astype(np.float32, copy=False)

    if cur4.ndim != 2 or cur4.shape[-1] != 4:
        raise ValueError(f"neighbor_agents_current_4_dim shape는 (N,4) 이어야 합니다. got {cur4.shape}")
    if fut3.ndim != 3 or fut3.shape[-1] != 3:
        raise ValueError(f"neighbor_future_gt_3_dim shape는 (N,Tf,3) 이어야 합니다. got {fut3.shape}")
    if cur4.shape[0] != fut3.shape[0]:
        raise ValueError(f"N 차원이 일치해야 합니다. got {cur4.shape[0]} vs {fut3.shape[0]}")

    N = int(cur4.shape[0])
    Tf = int(fut3.shape[1])
    out = np.zeros((N, 1 + Tf, 3), dtype=np.float32)

    if N == 0:
        return out

    valid_mask = (np.abs(cur4) > eps).any(axis=1)  # (N,)
    if not np.any(valid_mask):
        return out

    heading0 = np.arctan2(cur4[:, 3], cur4[:, 2]).astype(np.float32, copy=False)  # (N,)
    out[:, 0, 0] = cur4[:, 0]
    out[:, 0, 1] = cur4[:, 1]
    out[:, 0, 2] = heading0
    out[:, 1:, :] = fut3

    out[~valid_mask, :, :] = 0.0
    return out


def _get_near_future_segment_valid(
    ego_cur_future_gt_11_dim: ArrayF,  # (1+Tf,11)
    neighbor_cur_future_gt_11_dim: ArrayF,  # (N,1+Tf,11)
    *,
    eps: float = 1e-8,
) -> Tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """현재~미래 프레임의 유효/무효를 보고, '구간별(프레임-프레임 사이)' 유효 마스크를 만듭니다.

    유효 기준:
        - 11차원 중 앞 8개([x,y,cos,sin,vx,vy,width,length])가 전부 0이면 무효입니다.

    Args:
        ego_cur_future_gt_11_dim (np.ndarray): shape (1+Tf,11)
        neighbor_cur_future_gt_11_dim (np.ndarray): shape (N,1+Tf,11)
        eps (float): 0 판정 기준(아주 작은 값)

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - near_cur_future_valid: shape (1+N, 1+Tf)  (프레임 유효)
            - near_future_segment_valid: shape (1+N, Tf) (구간 유효: k와 k+1 둘 다 유효일 때 True)
    """
    ego11 = np.asarray(ego_cur_future_gt_11_dim)
    nbr11 = np.asarray(neighbor_cur_future_gt_11_dim)

    if ego11.ndim != 2 or ego11.shape[-1] != 11:
        raise ValueError(f"ego_cur_future_gt_11_dim shape는 (1+Tf,11) 이어야 합니다. got {ego11.shape}")
    if nbr11.ndim != 3 or nbr11.shape[-1] != 11:
        raise ValueError(f"neighbor_cur_future_gt_11_dim shape는 (N,1+Tf,11) 이어야 합니다. got {nbr11.shape}")
    if nbr11.shape[1] != ego11.shape[0]:
        raise ValueError(
            "시간축(1+Tf)이 일치해야 합니다. "
            f"got ego time={ego11.shape[0]}, neighbor time={nbr11.shape[1]}"
        )

    ego_valid = (np.abs(ego11[:, :8]) > eps).any(axis=1)  # (1+Tf,)
    nbr_valid = (np.abs(nbr11[:, :, :8]) > eps).any(axis=2)  # (N,1+Tf)

    near_cur_future_valid = np.concatenate([ego_valid[None, :], nbr_valid], axis=0).astype(bool)  # (1+N,1+Tf)
    near_future_segment_valid = (near_cur_future_valid[:, :-1] & near_cur_future_valid[:, 1:]).astype(bool)  # (1+N,Tf)
    return near_cur_future_valid, near_future_segment_valid


def build_cur_future_control_gt_3_dim_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)
    ego_future_gt_11_dim: ArrayF,  # (Tf,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    neighbor_future_gt_11_dim: ArrayF,  # (N,Tf,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> ArrayF:
    """npz 내부의 11차원 궤적들로부터 cur_future_control_gt_3_dim을 만듭니다.

    Args:
        ego_agent_past (np.ndarray): (Tp,11)
        ego_future_gt_11_dim (np.ndarray): (Tf,11)
        neighbor_agents_past (np.ndarray): (N,Tp,11)
        neighbor_future_gt_11_dim (np.ndarray): (N,Tf,11)
        dt (float): 시간 간격(예: 0.1)
        eps (float): 0 판정 기준(아주 작은 값)

    Returns:
        np.ndarray:
            future_seg_control_gt_3_dim, shape (1+N, Tf, 3)
            - 마지막 3: (v_x^b, v_y^b, omega)
    """
    ego_past = np.asarray(ego_agent_past)
    ego_fut11 = np.asarray(ego_future_gt_11_dim)
    nbr_past = np.asarray(neighbor_agents_past)
    nbr_fut11 = np.asarray(neighbor_future_gt_11_dim)

    if ego_past.ndim != 2 or ego_past.shape[-1] != 11:
        raise ValueError(f"ego_agent_past shape는 (Tp,11)이어야 합니다. got {ego_past.shape}")
    if ego_fut11.ndim != 2 or ego_fut11.shape[-1] != 11:
        raise ValueError(f"ego_future_gt_11_dim shape는 (Tf,11)이어야 합니다. got {ego_fut11.shape}")
    if nbr_past.ndim != 3 or nbr_past.shape[-1] != 11:
        raise ValueError(f"neighbor_agents_past shape는 (N,Tp,11)이어야 합니다. got {nbr_past.shape}")
    if nbr_fut11.ndim != 3 or nbr_fut11.shape[-1] != 11:
        raise ValueError(f"neighbor_future_gt_11_dim shape는 (N,Tf,11)이어야 합니다. got {nbr_fut11.shape}")

    Tp = int(ego_past.shape[0])
    Tf = int(ego_fut11.shape[0])
    N = int(nbr_past.shape[0])

    if Tp <= 0:
        raise ValueError(f"ego_agent_past Tp는 최소 1이어야 합니다. got Tp={Tp}")
    if nbr_past.shape[1] != Tp:
        raise ValueError(f"neighbor_agents_past의 Tp가 ego와 같아야 합니다. got {nbr_past.shape[1]} vs {Tp}")
    if nbr_fut11.shape[0] != N:
        raise ValueError(f"neighbor_future_gt_11_dim의 N이 neighbor_agents_past와 같아야 합니다. got {nbr_fut11.shape[0]} vs {N}")
    if nbr_fut11.shape[1] != Tf:
        raise ValueError(f"neighbor_future_gt_11_dim의 Tf가 ego_future와 같아야 합니다. got {nbr_fut11.shape[1]} vs {Tf}")

    # 11 -> 3 (x,y,heading)
    ego_future_gt_3_dim = _traj11_to_traj3_heading(ego_fut11)          # (Tf,3)
    neighbor_future_gt_3_dim = _traj11_to_traj3_heading(nbr_fut11)     # (N,Tf,3)

    # 현재+미래 3D 만들기
    ego_cur_future_gt_3_dim = _get_ego_cur_future_gt_3_dim(
        ego_current_4_dim=ego_past[-1, :4],          # (4,)
        ego_future_gt_3_dim=ego_future_gt_3_dim,     # (Tf,3)
        eps=eps,
    )  # (1+Tf,3)

    neighbor_cur_future_gt_3_dim = _get_neighbor_cur_future_gt_3_dim(
        neighbor_agents_current_4_dim=nbr_past[:, -1, :4],   # (N,4)
        neighbor_future_gt_3_dim=neighbor_future_gt_3_dim,   # (N,Tf,3)
        eps=eps,
    )  # (N,1+Tf,3)

    # 합치기: (1+N,1+Tf,3)
    all_cur_future_gt_3_dim = np.concatenate(
        [ego_cur_future_gt_3_dim[None, ...], neighbor_cur_future_gt_3_dim],
        axis=0,
    ).astype(np.float32, copy=False)

    # 유효 구간 마스크 (앞 8차원이 전부 0이면 무효)
    ego_cur_future_gt_11_dim = np.concatenate([ego_past[-1:, :], ego_fut11], axis=0)  # (1+Tf,11)
    neighbor_cur_future_gt_11_dim = np.concatenate([nbr_past[:, -1:, :], nbr_fut11], axis=1)  # (N,1+Tf,11)
    _, near_future_segment_valid = _get_near_future_segment_valid(
        ego_cur_future_gt_11_dim=ego_cur_future_gt_11_dim,
        neighbor_cur_future_gt_11_dim=neighbor_cur_future_gt_11_dim,
        eps=eps,
    )  # (1+N,Tf)

    # 제어 복원
    future_seg_control_gt_3_dim = differentiate_numpy_pose3_to_control3(
        all_cur_future_gt_3_dim,
        dt=float(dt),
    )  # (1+N,Tf,3)

    # 무효 구간은 0
    future_seg_control_gt_3_dim[~near_future_segment_valid] = 0.0
    return future_seg_control_gt_3_dim.astype(np.float32, copy=False)


def _load_training_file_list(json_path: str) -> List[str]:
    """학습에 쓰는 npz 파일명 리스트(json)를 읽습니다."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"json은 '문자열 리스트' 형식이어야 합니다. got type={type(data)}")
    return data


def _read_npz_as_dict(npz_path: str) -> Dict[str, np.ndarray]:
    """npz 파일을 전부 읽어서 dict로 만듭니다.

    기본은 allow_pickle=False로 읽고,
    만약 object 배열 때문에 실패하면 allow_pickle=True로 한 번 더 시도합니다.
    """
    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            return {k: npz[k] for k in npz.files}
    except ValueError as e:
        msg = str(e)
        if "allow_pickle=False" in msg or "Object arrays cannot be loaded" in msg:
            with np.load(npz_path, allow_pickle=True) as npz:
                return {k: npz[k] for k in npz.files}
        raise


def _atomic_save_npz(npz_path: str, data: Dict[str, np.ndarray], *, compress: bool) -> None:
    """npz를 임시 파일(.tmp)에 쓴 뒤 원래 이름으로 교체합니다."""
    tmp_path = npz_path + ".tmp"
    try:
        with open(tmp_path, "wb") as f:
            if compress:
                np.savez_compressed(f, **data)
            else:
                np.savez(f, **data)
        os.replace(tmp_path, npz_path)
    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def _build_future_gt_4_dim_from_3_dim(
    future_gt_3_dim: NDArray[np.floating],
    future_gt_is_valid: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """(…, 3) 미래 GT를 (…, 4)로 바꿉니다.

    변환 규칙
    --------
    - 입력 마지막 축 3개는 (x, y, 방향각)이라고 가정합니다.
    - 출력 마지막 축 4개는 (x, y, cos(방향각), sin(방향각)) 입니다.
    - future_gt_is_valid가 False인 위치는 출력 값을 0.0으로 만듭니다.

    Args:
        future_gt_3_dim:
            미래 GT. shape: (..., 3)
            예:
              - ego: (future_len, 3)
              - near: (predicted_neighbor_num, future_len, 3)
        future_gt_is_valid:
            유효 마스크. shape: (...)  (마지막 차원(3)은 제외한 shape)
            예:
              - ego: (future_len,)
              - near: (predicted_neighbor_num, future_len)

    Returns:
        future_gt_4_dim:
            변환된 미래 GT. shape: (..., 4)
    """
    gt3 = np.asarray(future_gt_3_dim)
    valid = np.asarray(future_gt_is_valid).astype(bool)

    if gt3.ndim < 1 or int(gt3.shape[-1]) != 3:
        raise ValueError(f"future_gt_3_dim last dim must be 3. got shape={gt3.shape}")

    heading = gt3[..., 2:3]  # (..., 1)
    cos_heading = np.cos(heading)  # (..., 1)
    sin_heading = np.sin(heading)  # (..., 1)

    future_gt_4_dim = np.concatenate(
        [gt3[..., :2], cos_heading, sin_heading],
        axis=-1,
    )  # (..., 4)

    future_gt_4_dim[~valid] = 0.0
    return future_gt_4_dim


def _add_future_gt_4_dim_keys_inplace(sample: Dict[str, Any]) -> None:
    """sample dict에 ego/near의 *_future_gt_4_dim 키를 추가합니다.

    - sample을 새로 만들지 않고, 들어온 sample dict를 그대로 수정합니다.
    - 필요한 입력 키:
      - ego: "ego_future_gt_3_dim", "ego_future_gt_is_valid"
      - near: "near_future_gt_3_dim", "near_future_gt_is_valid"

    Args:
        sample:
            __getitem__에서 만드는 샘플 dict와 같은 형태의 dict.
    """
    ego_future_gt_3_dim = sample.get("ego_future_gt_3_dim", None)
    ego_future_gt_is_valid = sample.get("ego_future_gt_is_valid", None)

    if isinstance(ego_future_gt_3_dim, np.ndarray) and isinstance(ego_future_gt_is_valid, np.ndarray):
        sample["ego_future_gt_4_dim"] = _build_future_gt_4_dim_from_3_dim(
            ego_future_gt_3_dim,
            ego_future_gt_is_valid,
        )

    near_future_gt_3_dim = sample.get("near_future_gt_3_dim", None)
    near_future_gt_is_valid = sample.get("near_future_gt_is_valid", None)

    if isinstance(near_future_gt_3_dim, np.ndarray) and isinstance(near_future_gt_is_valid, np.ndarray):
        sample["near_future_gt_4_dim"] = _build_future_gt_4_dim_from_3_dim(
            near_future_gt_3_dim,
            near_future_gt_is_valid,
        )


def _get_dataset_npz_keys(
    *,
    eval_method: str,
    use_agent_route_lane_order: bool,
) -> Tuple[List[str], Dict[str, str]]:
    """dataset.py의 __getitem__이 읽던 key 목록과 rename 규칙을 제공합니다."""
    both_keys: List[str] = [
        "origin_world_pose",
        "ego_agent_past",
        "ego_future_gt_3_dim",
        "ego_future_gt_11_dim",
        "neighbor_agents_past",
        "neighbor_future_gt_3_dim",
        "neighbor_future_gt_11_dim",
        "stop_sign_points",
        "crosswalk_points",
        "lanes",
        "lanes_speed_limit",
        "lanes_has_speed_limit",
    ]

    nuplan_only_keys: List[str] = [
        "static_objects",
        "route_lanes",
        "route_lanes_speed_limit",
        "route_lanes_has_speed_limit",
    ]
    if bool(use_agent_route_lane_order):
        nuplan_only_keys.append("agent_route_lane_order")

    womd_only_keys: List[str] = [
        "speed_bump_points",
        "driveway_points",
        "lane_type",
        "left_line_type",
        "right_line_type",
        "road_edge",
        "road_edge_type",
    ]

    wosac_only_keys: List[str] = []
    if str(eval_method) in ("validation", "test"):
        wosac_only_keys = [
            "target_id",
            "target_z",
        ]

    npz_keys: List[str] = both_keys + nuplan_only_keys + womd_only_keys + wosac_only_keys

    npz_key_to_new_key: Dict[str, str] = {
        "ego_future_gt_11_dim": "planner_future_11_dim",
        "driveway": "driveway_points",
    }
    return npz_keys, npz_key_to_new_key


def _build_sample_dict_like_dataset_getitem(
    npz_data: Mapping[str, Any],
    *,
    file_name: str,
    predicted_neighbor_num: int,
    eval_method: str,
    use_agent_route_lane_order: bool,
) -> Tuple[Dict[str, Any], Sequence[str], Sequence[str]]:
    """DiffusionPlannerData.__getitem__과 같은 방식으로 sample dict를 만듭니다.

    차이점
    ------
    - "tfrecord_path"는 어떤 경우에도 넣지 않습니다.

    Args:
        npz_data:
            npz에서 읽은 key->value 매핑.
        file_name:
            data_list에 있던 파일명(예: "abc.npz"). scenario_id를 만들 때 사용합니다.
        predicted_neighbor_num:
            near로 자를 neighbor 수.
        eval_method:
            "train" / "validation" / "test". 일부 key 선택에만 사용합니다.
        use_agent_route_lane_order:
            True면 agent_route_lane_order를 sample에 포함합니다.

    Returns:
        Tuple[Dict[str, Any], Sequence[str], Sequence[str]]:
            - sample: __getitem__이 반환하던 것과 같은 형태의 dict
            - base_output_keys: npz 원본에서 읽어서 채운 key들(rename 반영)
            - alias_output_keys: rename 때문에 새로 생긴 key들(예: planner_future_11_dim)
    """
    npz_keys, npz_key_to_new_key = _get_dataset_npz_keys(
        eval_method=str(eval_method),
        use_agent_route_lane_order=bool(use_agent_route_lane_order),
    )

    base_output_keys: List[str] = []
    alias_output_keys: List[str] = []

    sample: Dict[str, Any] = {}
    for npz_key in npz_keys:
        value = npz_data.get(npz_key, None)

        # dataset.py와 동일: agent_route_lane_order는 int64로 정리
        if value is not None and npz_key == "agent_route_lane_order":
            try:
                value = np.asarray(value).astype("int64")
            except Exception:
                pass

        out_key = npz_key_to_new_key.get(npz_key, npz_key)
        sample[out_key] = value
        base_output_keys.append(out_key)

        if out_key != npz_key:
            alias_output_keys.append(out_key)

    # validity / near split 로직은 기존 유틸을 그대로 사용
    try:
        from diffusion_planner.utils.validity import add_validity_keys_inplace
    except Exception as e:
        raise RuntimeError(
            "diffusion_planner.utils.validity.add_validity_keys_inplace import 실패. "
            "프로젝트 루트가 PYTHONPATH에 잡혀있는지 확인해 주세요."
        ) from e

    try:
        from nuplan_extent.planning.training.preprocessing.utils.near_agents import (
            add_near_agents_info_inplace,
        )
    except Exception as e:
        raise RuntimeError(
            "nuplan_extent...add_near_agents_info_inplace import 실패. "
            "프로젝트/의존성이 정상 설치되어 있는지 확인해 주세요."
        ) from e

    add_validity_keys_inplace(sample, missing_policy="none")
    add_near_agents_info_inplace(sample, predicted_neighbor_num=int(predicted_neighbor_num))

    scenario_id = str(os.path.splitext(str(file_name))[0])
    sample["scenario_id"] = scenario_id

    _add_future_gt_4_dim_keys_inplace(sample)
    return sample, base_output_keys, alias_output_keys


def _to_np_array_or_skip(value: Any) -> Union[np.ndarray, None]:
    """np.savez에 넣을 수 있는 값이면 np.ndarray로 바꾸고, 아니면 None을 반환합니다.

    Args:
        value: sample dict의 값.

    Returns:
        np.ndarray | None:
            - 저장 가능한 값이면 np.ndarray
            - 저장이 애매하면 None (스킵)
    """
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (np.number, int, float, bool, str, bytes)):
        return np.asarray(value)
    return None


def _merge_sample_keys_into_npz_data(
    data: Dict[str, np.ndarray],
    sample: Mapping[str, Any],
    *,
    base_output_keys: Sequence[str],
    alias_output_keys: Sequence[str],
    overwrite: bool,
) -> bool:
    """sample dict에서 만든 파생 key들을 npz dict에 합칩니다.

    규칙
    ----
    - dataset __getitem__이 "원본에서 읽은 key"들은 저장하지 않습니다.
      (큰 배열을 중복 저장하지 않기 위함)
    - 단, rename 때문에 생긴 alias key(예: planner_future_11_dim)는 저장합니다.
    - overwrite=False면:
        이미 존재하는 파생 key는 건드리지 않습니다.
    - overwrite=True면:
        파생 key를 다시 계산한 값으로 덮어씁니다.

    Args:
        data:
            npz에 저장할 dict (수정 대상).
        sample:
            __getitem__ 형태로 만든 sample dict.
        base_output_keys:
            원본에서 읽어서 채운 key들(rename 반영).
        alias_output_keys:
            rename으로 새로 생긴 key들.
        overwrite:
            True면 파생 key를 덮어씁니다.

    Returns:
        bool:
            data가 실제로 바뀌었으면 True, 아니면 False.
    """
    base_set = set(map(str, base_output_keys))
    alias_set = set(map(str, alias_output_keys))

    changed = False
    for key, value in sample.items():
        k = str(key)

        # 원본에서 읽어온 값은 저장하지 않음(중복 방지).
        # 단, alias key는 예외로 저장.
        if (k in base_set) and (k not in alias_set):
            continue

        arr = _to_np_array_or_skip(value)
        if arr is None:
            continue

        if (not overwrite) and (k in data):
            continue

        data[k] = arr
        changed = True

    return changed


def _process_one_file(
    npz_path: str,
    *,
    dt: float,
    overwrite: bool,
    compress: bool,
    add_sample_keys: bool,
    overwrite_sample_keys: bool,
    predicted_neighbor_num: int,
    eval_method: str,
    use_agent_route_lane_order: bool,
) -> Tuple[bool, str]:
    """npz 하나를 읽고 필요한 key들을 추가해 저장합니다.

    Returns:
        (success, message)
    """
    if not os.path.exists(npz_path):
        return False, f"missing: {npz_path}"

    data = _read_npz_as_dict(npz_path)

    # (A) control key 처리
    need_control = bool(overwrite) or ("future_seg_control_gt_3_dim" not in data)

    changed = False
    if need_control:
        required_keys = [
            "ego_agent_past",
            "ego_future_gt_11_dim",
            "neighbor_agents_past",
            "neighbor_future_gt_11_dim",
        ]
        for k in required_keys:
            if k not in data:
                return False, f"missing key '{k}'"

        control = build_cur_future_control_gt_3_dim_from_npz_arrays(
            ego_agent_past=data["ego_agent_past"],
            ego_future_gt_11_dim=data["ego_future_gt_11_dim"],
            neighbor_agents_past=data["neighbor_agents_past"],
            neighbor_future_gt_11_dim=data["neighbor_future_gt_11_dim"],
            dt=float(dt),
        )

        data["future_seg_control_gt_3_dim"] = control
        changed = True

    # (B) dataset.py __getitem__의 sample dict 로직을 동일하게 수행 (tfrecord_path 제외)
    if bool(add_sample_keys):
        try:
            sample, base_output_keys, alias_output_keys = _build_sample_dict_like_dataset_getitem(
                data,
                file_name=os.path.basename(npz_path),
                predicted_neighbor_num=int(predicted_neighbor_num),
                eval_method=str(eval_method),
                use_agent_route_lane_order=bool(use_agent_route_lane_order),
            )
        except Exception as e:
            return False, f"build_sample_failed: {type(e).__name__}: {e}"

        # sample에서 계산된 "파생 key"만 npz에 추가
        changed |= _merge_sample_keys_into_npz_data(
            data,
            sample,
            base_output_keys=base_output_keys,
            alias_output_keys=alias_output_keys,
            overwrite=bool(overwrite_sample_keys),
        )

    if not changed:
        return True, "skip(no change)"

    _atomic_save_npz(npz_path, data, compress=compress)
    return True, "ok"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add future_seg_control_gt_3_dim and (optionally) dataset sample keys to existing npz files."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/workspace/local_shards_v1",
        help="npz 파일들이 들어있는 폴더(내부 폴더 없음)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="/workspace/local_shards_v1/diffusion_planner_training.json",
        help="학습에 쓰는 npz 파일명 리스트(json)",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="시간 간격 dt (예: 0.1)")
    parser.add_argument("--overwrite", action="store_true", help="이미 control 키가 있어도 다시 계산해서 덮어씁니다.")
    parser.add_argument("--no_compress", action="store_true", help="저장할 때 압축을 끕니다(더 빠르지만 파일이 커짐).")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체, 양수면 앞에서 N개만 처리")

    # --- dataset.py __getitem__ sample dict 생성 관련 옵션 ---
    parser.add_argument(
        "--skip_sample_keys",
        action="store_true",
        help="dataset.py __getitem__에서 만들던 sample 파생키(validity/near/gt_4_dim/scenario_id)를 npz에 저장하지 않습니다.",
    )
    parser.add_argument(
        "--overwrite_sample_keys",
        action="store_true",
        help="파생키가 이미 있어도 다시 계산해서 덮어씁니다.",
    )
    parser.add_argument(
        "--predicted_neighbor_num",
        type=int,
        default=32,
        help="near로 뽑을 neighbor 수(predicted_neighbor_num).",
    )
    parser.add_argument(
        "--eval_method",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="dataset.py __getitem__에서 일부 key 선택에 쓰는 모드. tfrecord_path는 어떤 모드에서도 추가하지 않습니다.",
    )
    parser.add_argument(
        "--use_agent_route_lane_order",
        action="store_true",
        help="dataset.py와 동일하게 agent_route_lane_order 키도 sample에 포함합니다.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    dataset_dir: str = args.dataset_dir
    train_json: str = args.train_json
    dt: float = float(args.dt)
    overwrite: bool = bool(args.overwrite)
    compress: bool = (not bool(args.no_compress))
    limit: int = int(args.limit)

    add_sample_keys: bool = (not bool(args.skip_sample_keys))
    overwrite_sample_keys: bool = bool(args.overwrite_sample_keys)
    predicted_neighbor_num: int = int(args.predicted_neighbor_num)
    eval_method: str = str(args.eval_method)
    use_agent_route_lane_order: bool = bool(args.use_agent_route_lane_order)

    file_names = _load_training_file_list(train_json)
    if limit > 0:
        file_names = file_names[:limit]

    ok_count = 0
    fail_count = 0

    for fname in tqdm(file_names, desc="add_control_to_npz"):
        npz_path = os.path.join(dataset_dir, fname)
        try:
            ok, msg = _process_one_file(
                npz_path,
                dt=dt,
                overwrite=overwrite,
                compress=compress,
                add_sample_keys=add_sample_keys,
                overwrite_sample_keys=overwrite_sample_keys,
                predicted_neighbor_num=predicted_neighbor_num,
                eval_method=eval_method,
                use_agent_route_lane_order=use_agent_route_lane_order,
            )
            if ok:
                ok_count += 1
            else:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")
        except Exception as e:
            fail_count += 1
            tqdm.write(f"[EXCEPTION] {fname}: {type(e).__name__}: {e}")

    print(f"done. ok={ok_count}, fail={fail_count}, total={len(file_names)}")


if __name__ == "__main__":
    main()

"""
python /mnt/nuplan/projects/Diffusion-Planner/add_control_to_npz.py

python /mnt/nuplan/projects/Diffusion-Planner/add_control_to_npz.py --dt 0.1

# sample 파생키(near/validity/gt_4_dim/scenario_id)를 저장하지 않고 control만 추가
python /mnt/nuplan/projects/Diffusion-Planner/add_control_to_npz.py --skip_sample_keys

# 이미 저장된 파생키도 다시 계산해서 갱신
python /mnt/nuplan/projects/Diffusion-Planner/add_control_to_npz.py --overwrite_sample_keys

"""
