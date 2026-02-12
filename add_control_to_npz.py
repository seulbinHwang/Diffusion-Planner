from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple, Union

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
        cur_future_control_gt_3_dim: (P, T, 3)
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
    cur_future_control_gt_3_dim = np.stack([vx_b, vy_b, omega], axis=-1).astype(pose.dtype, copy=False)
    return cur_future_control_gt_3_dim


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
            cur_future_control_gt_3_dim, shape (1+N, Tf, 3)
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
    cur_future_control_gt_3_dim = differentiate_numpy_pose3_to_control3(
        all_cur_future_gt_3_dim,
        dt=float(dt),
    )  # (1+N,Tf,3)

    # 무효 구간은 0
    cur_future_control_gt_3_dim[~near_future_segment_valid] = 0.0
    return cur_future_control_gt_3_dim.astype(np.float32, copy=False)


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


def _process_one_file(
    npz_path: str,
    *,
    dt: float,
    overwrite: bool,
    compress: bool,
) -> Tuple[bool, str]:
    """npz 하나를 읽고 cur_future_control_gt_3_dim을 추가해 저장합니다.

    Returns:
        (success, message)
    """
    if not os.path.exists(npz_path):
        return False, f"missing: {npz_path}"

    data = _read_npz_as_dict(npz_path)

    if (not overwrite) and ("cur_future_control_gt_3_dim" in data):
        return True, "skip(existing key)"

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

    data["cur_future_control_gt_3_dim"] = control
    _atomic_save_npz(npz_path, data, compress=compress)
    return True, "ok"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add cur_future_control_gt_3_dim to existing npz files.")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/mnt/nuplan/dataset/processed",
        help="npz 파일들이 들어있는 폴더(내부 폴더 없음)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        default="/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json",
        help="학습에 쓰는 npz 파일명 리스트(json)",
    )
    parser.add_argument("--dt", type=float, default=0.1, help="시간 간격 dt (예: 0.1)")
    parser.add_argument("--overwrite", action="store_true", help="이미 키가 있어도 다시 계산해서 덮어씁니다.")
    parser.add_argument("--no_compress", action="store_true", help="저장할 때 압축을 끕니다(더 빠르지만 파일이 커짐).")
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체, 양수면 앞에서 N개만 처리")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    dataset_dir: str = args.dataset_dir
    train_json: str = args.train_json
    dt: float = float(args.dt)
    overwrite: bool = bool(args.overwrite)
    compress: bool = (not bool(args.no_compress))
    limit: int = int(args.limit)

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


"""