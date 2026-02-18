import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter
import draw_machine_fast
import traceback
import signal
import faulthandler
import time
import sys  # 추가
import contextlib
import multiprocessing.pool as mppool
from types import SimpleNamespace
from diffusion_planner.model.module.feasible import FeasibleProjector
_LOG_ENABLED = True  # 메인 프로세스는 기본 출력
_LOGGER_PID_VAL = None  # multiprocessing.Value (pid 저장)
# =========================
# 맵/정적 객체 필터 반경(ego 기준)
# =========================
# =========================
# 맵/정적 객체 필터 반경(ego 기준)
# =========================
_USE_FILTER_RADIUS: bool = True
_FILTER_RADIUS_M: float = 150.0

from typing import Tuple
import numpy as np
from numpy.typing import NDArray
ArrayF = NDArray[np.floating]
from diffusion_planner.loss import _compute_xy_yaw_losses



def _compute_valid_mask_from_feat_11(
    feat_11: np.ndarray,  # shape: (T, 11)
    eps: float = 0.0,
) -> np.ndarray:
    """(T,11) 에이전트 시계열에서 각 시점이 '유효한 점'인지 마스크를 만듭니다.

    여기서 유효/무효 판단은 아주 단순하게 합니다.
    - 해당 시점의 11차원 값이 전부 0이면 -> 무효(False)
    - 하나라도 0이 아니면 -> 유효(True)

    이 방식은 캐시에 저장되는 최종 11차원 데이터가
    "무효 시점은 11차원 전부 0"이라는 전제를 갖기 때문에,
    이후 단계(학습/로딩 등)에서도 같은 기준을 그대로 쓸 수 있습니다.

    Args:
        feat_11: (T,11) float32. [x,y,cos,sin,vx,vy,width,length,one_hot(3)] 형태.
        eps: 0과 비교할 때 아주 작은 오차를 무시하고 싶으면 양수로 줄 수 있습니다.
             (보통은 0.0으로 두면 됩니다)

    Returns:
        valid_mask: (T,) bool. True면 유효, False면 무효.
    """
    # feat_11: np.ndarray, shape (T, 11)
    if feat_11.ndim != 2 or int(feat_11.shape[1]) != 11:
        return np.zeros((int(feat_11.shape[0]),), dtype=bool)

    abs_max = np.max(np.abs(feat_11.astype(np.float32)), axis=1)  # shape: (T,)
    return (abs_max > float(eps))





def _build_future_3_from_future_11(
    future_feat_11: np.ndarray,  # shape: (F, 11)
) -> np.ndarray:
    """(F,11) 미래 11차원에서 (F,3) 미래 GT([x,y,heading])를 만듭니다.

    - x, y는 그대로 사용합니다.
    - heading은 (cos, sin)에서 atan2(sin, cos)로 복원합니다.
    - 무효 시점은 (0,0,0)으로 유지합니다.

    Args:
        future_feat_11: (F,11) float32.

    Returns:
        future_feat_3: (F,3) float32. [x, y, heading]
    """
    # future_feat_11: np.ndarray, shape (F, 11)
    future_len = int(future_feat_11.shape[0])
    out = np.zeros((future_len, 3), dtype=np.float32)  # shape: (F,3)
    if future_len == 0:
        return out

    valid_mask = _compute_valid_mask_from_feat_11(future_feat_11)  # shape: (F,)
    if not bool(np.any(valid_mask)):
        return out

    xy = future_feat_11[:, 0:2].astype(np.float32)  # shape: (F,2)
    cos_yaw = future_feat_11[:, 2].astype(np.float32)  # shape: (F,)
    sin_yaw = future_feat_11[:, 3].astype(np.float32)  # shape: (F,)

    heading = np.arctan2(sin_yaw, cos_yaw).astype(np.float32)  # shape: (F,)
    heading = wrap_angle_np(heading)  # [-pi, pi)

    out[valid_mask, 0:2] = xy[valid_mask]
    out[valid_mask, 2] = heading[valid_mask]
    return out

def _apply_current_validity_and_fill_gaps_with_interpolation(
    pos_xy_raw: np.ndarray,          # shape: (T, 2)
    vel_xy_raw: np.ndarray,          # shape: (T, 2)
    yaw_raw: np.ndarray,             # shape: (T,)
    width_length_raw: np.ndarray,    # shape: (T, 2)
    valid_raw: np.ndarray,           # shape: (T,)
    current_index: int,
    enforce_all_invalid_when_current_invalid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """현재 시점을 기준으로 유효/무효 규칙을 적용하고, 유효점 사이의 빈 칸을 보간으로 채웁니다.

    이 함수의 목적은 “현재가 유효한 트랙”에 대해서
    `valid=True`인 시점들 사이에 끼어 있는 `valid=False` 시점들을 그대로 0으로 두지 않고,
    **직선 보간으로 채워서** “유효-무효-유효” 같은 끊김 패턴이 나오지 않게 만드는 것입니다.

    처리 규칙:
    1) 현재 시점(current_index)이 무효(valid_raw[current_index]=False)라면
       - (enforce_all_invalid_when_current_invalid 값과 관계없이)
         안전하게 전체를 0으로 비우고 valid도 전부 False로 반환합니다.
       - 이유: 현재가 무효인데 과거/미래에 값이 섞여 있으면 downstream에서 해석이 애매해지고,
         원치 않는 “점프”가 생기기 쉽습니다.

    2) 현재 시점이 유효라면
       - valid_raw에서 True로 표시된 시점들을 “실제로 값이 있는 시점”으로 보고,
         그 사이에 끼어 있는 빈 칸(valid=False)을 보간으로 채웁니다.
       - 위치/속도/크기(width,length)는 직선 보간,
         yaw(각도)는 -pi/pi 경계에서 튀지 않도록 한 번 자연스럽게 이어서(unwarp) 보간한 뒤,
         다시 [-pi, pi) 범위로 정리합니다.
       - 결과 valid_mask는 “첫 유효 시점 ~ 마지막 유효 시점” 구간이 전부 True가 됩니다.
         (구간 밖은 False 유지)

    Args:
        pos_xy_raw: (T,2) 원본 위치 시계열. 무효 시점은 (0,0)일 수 있습니다.
        vel_xy_raw: (T,2) 원본 속도 시계열.
        yaw_raw: (T,) 원본 yaw 시계열(rad).
        width_length_raw: (T,2) 원본 (width, length) 시계열.
        valid_raw: (T,) 원본 유효 여부.
        current_index: 현재 시점이 시계열에서 몇 번째인지.
        enforce_all_invalid_when_current_invalid: 현재가 무효일 때 “전체 무효 처리”를 강제할지.
            - 현재 구현에서는 안전을 위해 True/False와 상관없이 전체 무효로 반환합니다.
              (neighbor는 현재 유효한 트랙만 뽑으므로 보통 이 케이스 자체가 발생하지 않습니다.)

    Returns:
        pos_xy_filled: (T,2) float32. 보간으로 채워진 위치.
        vel_xy_filled: (T,2) float32. 보간으로 채워진 속도.
        yaw_filled: (T,) float32. 보간으로 채워진 yaw.
        width_length_filled: (T,2) float32. 보간으로 채워진 (width,length).
        valid_filled: (T,) bool. “첫 유효~마지막 유효” 구간은 True.
    """
    time_len = int(pos_xy_raw.shape[0])

    pos_xy_out = np.zeros((time_len, 2), dtype=np.float32)          # shape: (T,2)
    vel_xy_out = np.zeros((time_len, 2), dtype=np.float32)          # shape: (T,2)
    yaw_out = np.zeros((time_len,), dtype=np.float32)               # shape: (T,)
    width_length_out = np.zeros((time_len, 2), dtype=np.float32)    # shape: (T,2)
    valid_out = np.zeros((time_len,), dtype=bool)                   # shape: (T,)

    if time_len == 0:
        return pos_xy_out, vel_xy_out, yaw_out, width_length_out, valid_out

    cur = int(current_index)
    cur = max(0, min(cur, time_len - 1))

    # 현재가 무효면: 전체 무효(안전)
    if not bool(valid_raw.astype(bool)[cur]):
        # enforce_all_invalid_when_current_invalid 플래그는 유지하되,
        # 현재가 무효인 케이스는 해석이 애매해서 안전하게 전부 0으로 처리합니다.
        _ = bool(enforce_all_invalid_when_current_invalid)
        return pos_xy_out, vel_xy_out, yaw_out, width_length_out, valid_out

    # 현재가 유효면: 유효점 사이의 빈 칸을 보간으로 채움
    pos_xy_filled, vel_xy_filled, yaw_filled, width_length_filled, valid_filled = build_interpolated_agent_traj(
        pos_xy_raw=pos_xy_raw,
        vel_xy_raw=vel_xy_raw,
        yaw_raw=yaw_raw,
        width_length_raw=width_length_raw,
        valid_raw=valid_raw,
    )

    return (
        pos_xy_filled.astype(np.float32),
        vel_xy_filled.astype(np.float32),
        yaw_filled.astype(np.float32),
        width_length_filled.astype(np.float32),
        valid_filled.astype(bool),
    )

def build_agent_past_future_cache_arrays_from_full_trajectory(
    agent_idx: int,
    past_indices: np.ndarray,  # shape: (time_len,)
    future_indices: np.ndarray,  # shape: (future_len,)
    pos_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    vel_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    yaw_local_all: np.ndarray,  # shape: (N, S)
    width_length_all: np.ndarray,  # shape: (N, S, 2)
    valid_all: np.ndarray,  # shape: (N, S)
    agent_one_hot: np.ndarray,  # shape: (3,)
    enforce_all_invalid_when_current_invalid: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """에이전트 1명의 past/future 출력 배열을, '현재 기준 유효 규칙'을 만족하도록 만들어 반환합니다.

    이 함수는 과거(past)와 미래(future)를 먼저 하나로 합친 뒤(T=time_len+future_len),
    아래 규칙을 전체 길이 기준으로 적용하고,
    마지막에 past / future로 다시 나눠 반환합니다.

    규칙:
    1) 현재 시점이 무효라면:
       - 전체를 전부 무효(전부 0)로 반환합니다.
         (neighbor는 보통 “현재 유효한 트랙만 뽑기” 때문에 이 케이스가 거의 없어야 합니다.)

    2) 현재 시점이 유효라면:
       - 유효점(True)과 유효점(True) 사이에 끼어 있는 무효점(False)은
         직선 보간으로 채웁니다.
       - 그 결과 “유효-무효-유효” 같은 끊김 패턴이 없어집니다.
       - 첫 유효 이전/마지막 유효 이후 구간은 0(무효)로 유지합니다.

    반환 포맷:
      - past 11차원: (time_len, 11)
      - future 3차원: (future_len, 3)
      - future 11차원: (future_len, 11)

    Args:
        agent_idx: 대상 트랙 인덱스.
        past_indices: (time_len,) 과거(현재 포함) 인덱스. 없는 곳은 -1.
        future_indices: (future_len,) 미래 인덱스. 없는 곳은 -1.
        pos_xy_local_all: (N,S,2)
        vel_xy_local_all: (N,S,2)
        yaw_local_all: (N,S)
        width_length_all: (N,S,2)
        valid_all: (N,S)
        agent_one_hot: (3,)
        enforce_all_invalid_when_current_invalid: 현재가 무효일 때 전체 무효로 처리할지(방어용).

    Returns:
        agent_past_11: (time_len,11) float32
        agent_future_3: (future_len,3) float32
        agent_future_11: (future_len,11) float32
    """
    past_len = int(past_indices.shape[0])
    future_len = int(future_indices.shape[0])

    # full_indices: (time_len + future_len,)
    full_indices = np.concatenate([past_indices, future_indices], axis=0).astype(np.int64)

    # 원본 값 뽑기
    # pos_raw: (T,2), vel_raw: (T,2), yaw_raw: (T,), wl_raw: (T,2), valid_raw: (T,)
    pos_raw, vel_raw, yaw_raw, wl_raw, valid_raw = gather_agent_sequence_by_indices(
        agent_idx=int(agent_idx),
        time_indices=full_indices,
        pos_xy_local_all=pos_xy_local_all,
        vel_xy_local_all=vel_xy_local_all,
        yaw_local_all=yaw_local_all,
        width_length_all=width_length_all,
        valid_all=valid_all,
    )

    # 현재는 past의 마지막
    current_index_in_full = max(0, past_len - 1)

    # ✅ (b) 요구: 유효-무효-유효 패턴이 생기면, 중간 무효를 보간으로 채우기
    pos_filled, vel_filled, yaw_filled, wl_filled, valid_filled = _apply_current_validity_and_fill_gaps_with_interpolation(
        pos_xy_raw=pos_raw,                 # (T,2)
        vel_xy_raw=vel_raw,                 # (T,2)
        yaw_raw=yaw_raw,                    # (T,)
        width_length_raw=wl_raw,            # (T,2)
        valid_raw=valid_raw,                # (T,)
        current_index=int(current_index_in_full),
        enforce_all_invalid_when_current_invalid=bool(enforce_all_invalid_when_current_invalid),
    )

    # full_feat_11_clean: (T,11)
    full_feat_11_clean = pack_agent_features_11(
        pos_xy=pos_filled,                  # (T,2)
        vel_xy=vel_filled,                  # (T,2)
        yaw_rad=yaw_filled,                 # (T,)
        width_length=wl_filled,             # (T,2)
        agent_one_hot=agent_one_hot,        # (3,)
        valid_mask=valid_filled,            # (T,)
    ).astype(np.float32)

    # past / future 분리
    agent_past_11 = full_feat_11_clean[:past_len].astype(np.float32)  # (time_len,11)
    agent_future_11 = full_feat_11_clean[past_len:past_len + future_len].astype(np.float32)  # (future_len,11)

    # future_3 만들기 (future_11 기반)
    agent_future_3 = _build_future_3_from_future_11(agent_future_11).astype(np.float32)  # (future_len,3)

    return agent_past_11, agent_future_3, agent_future_11


def _require_filter_radius_args(args: Any) -> None:
    """args에 필터링 관련 설정이 있는지 확인합니다.

    이 코드에서는 오직 args_util.py에서 선언한 아래 두 값만 사용하도록 강제합니다.
      - use_filter_radius (bool)
      - filter_radius (float, meter)

    위 두 값이 없으면, 잘못된 설정/환경에서 "조용히 다른 값"을 쓰는 일을 막기 위해
    즉시 에러를 내고 실행을 중단합니다.

    Args:
        args: args_util.get_args()가 반환한 인자 객체(보통 argparse.Namespace)

    Raises:
        RuntimeError: use_filter_radius 또는 filter_radius가 args에 없을 때
    """
    missing: List[str] = []
    if not hasattr(args, "use_filter_radius"):
        missing.append("use_filter_radius")
    if not hasattr(args, "filter_radius"):
        missing.append("filter_radius")

    if len(missing) > 0:
        raise RuntimeError(
            "필수 인자가 args에 없습니다. "
            "args_util.py에 `--use_filter_radius`(bool)와 `--filter_radius`(float)를 반드시 선언해야 합니다. "
            f"missing={missing}"
        )


def _set_filter_radius_settings_from_args(args: Any) -> None:
    """args에서 필터링 설정을 읽어 전역 변수에 반영합니다.

    규칙:
      - use_filter_radius=True:
          _FILTER_RADIUS_M = args.filter_radius (meter)
          이후 맵/정적 객체 필터링을 실제로 수행합니다.
      - use_filter_radius=False:
          _FILTER_RADIUS_M 값은 읽어 두되(로깅/재현성을 위해),
          실제 필터링은 수행하지 않도록 _USE_FILTER_RADIUS=False로 둡니다.

    Args:
        args: args_util.get_args() 결과

    Raises:
        RuntimeError: 필수 인자가 args에 없을 때
        ValueError: use_filter_radius=True인데 filter_radius가 음수인 경우
    """
    global _USE_FILTER_RADIUS, _FILTER_RADIUS_M

    _require_filter_radius_args(args)

    _USE_FILTER_RADIUS = bool(getattr(args, "use_filter_radius"))
    _FILTER_RADIUS_M = float(getattr(args, "filter_radius"))

    # 필터링을 켜는 경우에만 값 검증을 더 강하게 합니다.
    if _USE_FILTER_RADIUS:
        # inf는 "사실상 필터 없음"으로도 쓸 수 있으니 허용합니다.
        # 다만 음수는 의미가 애매해서 바로 막습니다.
        if _FILTER_RADIUS_M < 0.0:
            raise ValueError(
                "use_filter_radius=True인데 filter_radius가 음수입니다. "
                "필터를 끄고 싶으면 use_filter_radius=False를 사용하세요. "
                f"filter_radius={_FILTER_RADIUS_M}"
            )




def _ping() -> str:
    return f"ping pid={os.getpid()}"


def _claim_logger_if_needed() -> None:
    """처음 일을 시작한 워커 1개만 로거로 '선점'해서 출력하도록 함."""
    global _LOG_ENABLED
    if _LOGGER_PID_VAL is None:
        _LOG_ENABLED = True
        return

    with _LOGGER_PID_VAL.get_lock():
        if _LOGGER_PID_VAL.value == 0:
            _LOGGER_PID_VAL.value = os.getpid()

    _LOG_ENABLED = (_LOGGER_PID_VAL.value == os.getpid())


_WORKER_STOP_EVENT = None  # 추가 (spawn 워커에서 init으로 채움)


def _log(msg: str) -> None:
    if not _LOG_ENABLED:
        return
    print(f"[{time.strftime('%H:%M:%S')}] pid={os.getpid()} {msg}",
          file=sys.stderr,
          flush=True)

_FEASIBLE_PROJECTOR_CACHE: Optional[FeasibleProjector] = None

def compute_past_future_yaw_rate_from_cs_yaw_via_feasible_projector(
    ego_past_future_gt_cs_yaw: np.ndarray,          # shape: (point_len, 2)
    neighbor_past_future_gt_cs_yaw: np.ndarray,     # shape: (A, point_len, 2)
    ego_pf_valid: np.ndarray,                       # shape: (point_len,)
    neighbor_pf_valid: np.ndarray,                  # shape: (A, point_len)
    *,
    dt_sec: float,
    polyorder: int = 2,
    max_window_len_yaw: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    """(cos(yaw), sin(yaw))을 그대로 사용해 yaw_rate를 계산합니다.

    목적
    ----
    - 이미 계산해 둔 cs_yaw(=cos/sin) 시계열을 그대로 사용해,
      FeasibleProjector의 SG(yaw_rate) 로직만 재사용합니다.
    - 이렇게 하면 points(x,y,cos,sin)를 다시 만들거나,
      그 안에서 다시 cos/sin을 뽑는 중복 계산을 줄일 수 있습니다.

    Args:
        ego_past_future_gt_cs_yaw:
            shape: (point_len, 2), dtype float32 권장.
            마지막 축 2는 [cos(yaw), sin(yaw)] 입니다.
        neighbor_past_future_gt_cs_yaw:
            shape: (A, point_len, 2), dtype float32 권장.
        ego_pf_valid:
            shape: (point_len,), dtype bool.
            True면 유효 시점입니다.
        neighbor_pf_valid:
            shape: (A, point_len), dtype bool.
        dt_sec:
            샘플 간 시간 간격(초).
        polyorder:
            SG 계산 차수(기본 2).
        max_window_len_yaw:
            SG 창 최대 길이(기본 7).

    Returns:
        ego_yaw_rate:
            shape: (point_len,), dtype float32
        neighbor_yaw_rate:
            shape: (A, point_len), dtype float32
    """
    ego_cs = np.asarray(ego_past_future_gt_cs_yaw, dtype=np.float32)
    neigh_cs = np.asarray(neighbor_past_future_gt_cs_yaw, dtype=np.float32)

    ego_valid = np.asarray(ego_pf_valid, dtype=bool)
    neigh_valid = np.asarray(neighbor_pf_valid, dtype=bool)

    if ego_cs.ndim != 2 or int(ego_cs.shape[1]) != 2:
        raise ValueError(f"ego_past_future_gt_cs_yaw must be (point_len,2). got={ego_cs.shape}")
    if neigh_cs.ndim != 3 or int(neigh_cs.shape[2]) != 2:
        raise ValueError(f"neighbor_past_future_gt_cs_yaw must be (A,point_len,2). got={neigh_cs.shape}")

    point_len = int(ego_cs.shape[0])
    if int(ego_valid.shape[0]) != point_len:
        raise ValueError(
            "ego_pf_valid length mismatch. "
            f"ego_pf_valid.shape={ego_valid.shape}, point_len={point_len}"
        )

    A = int(neigh_cs.shape[0])
    if int(neigh_cs.shape[1]) != point_len:
        raise ValueError(
            "neighbor_past_future_gt_cs_yaw point_len mismatch. "
            f"neighbor.shape={neigh_cs.shape}, point_len={point_len}"
        )
    if neigh_valid.shape != (A, point_len):
        raise ValueError(
            "neighbor_pf_valid shape mismatch. "
            f"neighbor_pf_valid.shape={neigh_valid.shape}, expected={(A, point_len)}"
        )

    # (1) ego+neighbor를 (Pnn=1+A)로 묶기
    cs_yaw_pnn = np.concatenate(
        [ego_cs[None, ...], neigh_cs],
        axis=0,
    ).astype(np.float32)  # shape: (1+A, point_len, 2)

    valid_pnn = np.concatenate(
        [ego_valid[None, ...], neigh_valid],
        axis=0,
    ).astype(bool)  # shape: (1+A, point_len)

    # (2) torch 입력 (B=1, Pnn=1+A)
    # cos_y/sin_y: (1, 1+A, point_len)
    cos_y_t = torch.from_numpy(np.ascontiguousarray(cs_yaw_pnn[..., 0][None, ...]))
    sin_y_t = torch.from_numpy(np.ascontiguousarray(cs_yaw_pnn[..., 1][None, ...]))
    points_valid_t = torch.from_numpy(np.ascontiguousarray(valid_pnn[None, ...]))  # bool

    # (3) SG(yaw_rate) 계산 (FeasibleProjector SG 로직 재사용)
    fp = _get_feasible_projector_for_cache()
    with torch.no_grad():
        yaw_rate_t = fp._compute_yaw_rate_via_sg(
            cos_y=cos_y_t,
            sin_y=sin_y_t,
            points_valid=points_valid_t,
            dt=float(dt_sec),
            polyorder=int(polyorder),
            max_window_len_yaw=int(max_window_len_yaw),
        )  # shape: (1, 1+A, point_len)

    yaw_rate_pnn = yaw_rate_t.squeeze(0).cpu().numpy().astype(np.float32)  # (1+A, point_len)
    ego_yaw_rate = yaw_rate_pnn[0]         # (point_len,)
    neighbor_yaw_rate = yaw_rate_pnn[1:]   # (A, point_len)

    return ego_yaw_rate, neighbor_yaw_rate


def _get_feasible_projector_for_cache() -> FeasibleProjector:
    """캐싱 스크립트에서 yaw_rate 계산에만 쓸 FeasibleProjector를 워커당 1개만 만든 뒤 재사용합니다.

    - FeasibleProjector의 SG(yaw_rate 계산) 관련 함수들을 그대로 쓰기 위한 목적입니다.
    - 신경망(Feasible DL)은 쓰지 않으므로 use_feasible_dl=False로 만들어 초기화 비용을 줄입니다.

    Returns:
        FeasibleProjector: 워커 프로세스 내에서 재사용되는 인스턴스.
    """
    global _FEASIBLE_PROJECTOR_CACHE
    if _FEASIBLE_PROJECTOR_CACHE is not None:
        return _FEASIBLE_PROJECTOR_CACHE

    # FeasibleProjector __init__에서 필요한 최소 config 필드만 둡니다.
    cfg = SimpleNamespace(
        use_batch_integration=False,
        feasible_debug_check_mask=False,
    )

    # hidden_dim은 Feasible DL을 끄면 사실상 쓰이지 않지만, 시그니처 상 필요합니다.
    fp = FeasibleProjector(
        config=cfg,
        hidden_dim=1,
        use_feasible_dl=False,
        use_feasible_filter=False,
    )
    fp.eval()
    _FEASIBLE_PROJECTOR_CACHE = fp
    return fp


import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
import math
import args_util
import contextlib
import multiprocessing.pool as mppool
import time

def _get_dead_worker_info_from_pool(
    pool: Optional[mppool.Pool],
) -> List[Tuple[int, Optional[int]]]:
    """Pool 안의 워커들 중 '이미 죽어있는 것'을 찾아서 (pid, exitcode) 목록으로 반환합니다.

    이 함수는 "워커가 살아있는지"만 확인합니다.
    - 살아있으면 목록에 넣지 않습니다.
    - 죽어있으면 (pid, exitcode)를 기록합니다.
      exitcode는 종료 원인을 나타내는데,
      None이거나 0이 아닐 수도 있습니다(강제 종료/에러 종료 등).

    Args:
        pool: multiprocessing Pool. None이면 빈 리스트를 반환합니다.

    Returns:
        dead: List[Tuple[int, Optional[int]]]
            - 각 원소는 (pid, exitcode)
            - pid를 알 수 없으면 -1
    """
    if pool is None:
        return []

    procs = getattr(pool, "_pool", None) or []
    dead: List[Tuple[int, Optional[int]]] = []
    for p in procs:
        if p is None:
            continue
        # p: multiprocessing.Process 류
        if not p.is_alive():
            pid_val = int(p.pid) if p.pid is not None else -1
            exit_code = p.exitcode  # Optional[int]
            dead.append((pid_val, exit_code))
    return dead


def _die_all_if_any_worker_dead(
    pool: Optional[mppool.Pool],
    stop_event: Any,
    enabled: bool,
    exit_code: int = 1,
) -> None:
    """enabled=True일 때, 워커가 하나라도 죽어있으면 전체 워커를 즉시 종료합니다.

    목적(쉽게 설명)
    -------------
    - 데이터 캐싱처럼 긴 작업에서, 워커 하나가 갑자기 죽으면(메모리 문제/프로세스 크래시 등)
      남은 워커들이 계속 돌거나 메인이 멈춰서 기다리는 상황이 생길 수 있습니다.
    - args.die_all=True인 경우에는 이런 상황을 "즉시 전체 종료"로 바꿉니다.

    동작 순서
    --------
    1) stop_event가 아직 꺼져 있고(pool이 정상 실행 중) enabled=True이면,
       pool 안의 워커 목록을 확인합니다.
    2) 죽은 워커가 발견되면:
       - stop_event.set() 해서 "나도 멈출 거다"를 표시하고,
       - _terminate_pool_hard(pool)로 전체 워커를 강제 종료하고,
       - SystemExit(exit_code)로 메인도 즉시 종료합니다.

    Args:
        pool: multiprocessing Pool.
        stop_event: 메인/워커가 공유하는 종료 이벤트.
        enabled: die_all 기능 사용 여부(args.die_all).
        exit_code: 종료 코드(기본 1).
    """
    if not bool(enabled):
        return
    if pool is None:
        return
    if stop_event is not None and stop_event.is_set():
        return

    # pool이 이미 close/terminate 상태라면 감시할 필요 없음
    pool_state = getattr(pool, "_state", None)
    if pool_state is not None and pool_state != mppool.RUN:
        return

    dead = _get_dead_worker_info_from_pool(pool)
    if len(dead) == 0:
        return

    # 여기로 왔다는 건 "워커 하나 이상이 죽어있음"
    if stop_event is not None:
        stop_event.set()

    _log(f"[die_all] dead worker detected -> {dead}. terminate all workers now.")
    _terminate_pool_hard(pool, timeout_sec=1.0)
    raise SystemExit(int(exit_code))


def _terminate_pool_hard(
    pool: Optional[mppool.Pool],
    timeout_sec: float = 3.0,
) -> None:
    """Pool을 Ctrl+C 시 '확실하게' 종료합니다.

    1) pool.terminate()로 SIGTERM 전송
    2) timeout_sec 동안 join 시도
    3) 아직 살아있는 워커는 SIGKILL로 강제 종료

    Args:
        pool: multiprocessing Pool (None이면 무시)
        timeout_sec: SIGTERM 후 기다릴 최대 시간(초)
    """
    if pool is None:
        return

    # 1) SIGTERM
    with contextlib.suppress(Exception):
        pool.terminate()

    procs = getattr(pool, "_pool", None) or []
    deadline = time.monotonic() + float(timeout_sec)

    # 2) 일정 시간 기다리기
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(Exception):
            proc.join(timeout=remaining)

    # 3) 남아있으면 SIGKILL
    for proc in procs:
        if proc.is_alive():
            # Python 3.7+ : Process.kill() 지원
            with contextlib.suppress(Exception):
                proc.kill()
            # kill()이 없거나 실패한 환경 대비
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    os.kill(proc.pid, signal.SIGKILL)

    # 마무리 join (짧게)
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.join(timeout=1.0)


def _join_pool_soft(
    pool: Optional[mppool.Pool],
    timeout_sec: float = 10.0,
) -> bool:
    """pool.close() 이후 워커들이 timeout 안에 종료되는지 기다립니다.

    Args:
        pool: multiprocessing Pool
        timeout_sec: 기다릴 최대 시간(초)

    Returns:
        모두 종료되면 True, 일부라도 살아있으면 False
    """
    if pool is None:
        return True
    procs = getattr(pool, "_pool", None) or []
    deadline = time.monotonic() + float(timeout_sec)
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(Exception):
            proc.join(timeout=remaining)
    return not any(p.is_alive() for p in procs)


def _install_main_signal_handlers(
    stop_event: Any,
    pool_ref: Dict[str, Optional[mppool.Pool]],
    grace_sec: float = 12.0,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """메인에서 Ctrl+C를 '확실히' 잡아서 stop_event로 전파 + 단계적 종료를 수행합니다.

    - Ctrl+C 1회: stop_event set -> 우아한 종료(워커가 체크 지점에서 빠져나오게)
    - Ctrl+C 2회: 즉시 워커 강제 종료(SIGKILL 포함) 후 종료
    """
    state: Dict[str, Any] = {
        "count": 0,
        "deadline": None,
        "grace_sec": float(grace_sec)
    }
    old = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def _handler(signum: int, frame: Any) -> None:
        state["count"] += 1
        stop_event.set()

        if state["deadline"] is None:
            state["deadline"] = time.monotonic() + state["grace_sec"]

        pool = pool_ref.get("pool", None)

        if state["count"] == 1:
            _log(
                "Ctrl+C 감지 -> stop_event 전파(우아한 종료 시도). 한 번 더 누르면 즉시 강제 종료합니다.")
            if pool is not None:
                with contextlib.suppress(Exception):
                    pool.close()
            return

        _log("Ctrl+C 2회 감지 -> 워커 강제 종료(SIGKILL 포함) 후 즉시 종료합니다.")
        _terminate_pool_hard(pool, timeout_sec=1.0)
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # 환경에 따라 block syscall에서 더 잘 빠져나오게
    with contextlib.suppress(Exception):
        signal.siginterrupt(signal.SIGINT, True)
        signal.siginterrupt(signal.SIGTERM, True)

    return old, state


def wrap_angle(angle: torch.Tensor,
               min_val: float = -math.pi,
               max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


# =========================
# 설정값 (요구사항 고정)
# =========================

SPLITS: Tuple[str, ...] = ("training", "validation", "testing")

# args_util.py에서 주입받는 길이들(워커 init에서 설정됨)
TIME_LEN: int = 0
FUTURE_LEN: int = 0
SAFETY_LEN: int = 0
LANE_LEN: int = 0

DT_SEC: float = 0.1
EPS: float = 1e-6


def _set_womd_lengths_from_args(args: Any) -> None:
    """args_util.py 인자에서 길이 관련 설정을 가져와 전역 변수에 주입합니다."""
    global TIME_LEN, FUTURE_LEN, SAFETY_LEN, LANE_LEN

    TIME_LEN = int(getattr(args, "time_len"))
    FUTURE_LEN = int(getattr(args, "future_len"))
    SAFETY_LEN = int(getattr(args, "safety_len"))
    LANE_LEN = int(getattr(args, "lane_len"))

    # ✅ 필터 반경은 args_util.py의 (use_filter_radius, filter_radius)만 사용
    _set_filter_radius_settings_from_args(args)

    if TIME_LEN <= 0 or FUTURE_LEN <= 0 or SAFETY_LEN <= 0 or LANE_LEN <= 0:
        raise ValueError(
            f"Invalid lengths: time_len={TIME_LEN}, future_len={FUTURE_LEN}, "
            f"safety_len={SAFETY_LEN}, lane_len={LANE_LEN}"
        )


def _require_womd_lengths_initialized() -> None:
    """워커 init에서 길이값이 세팅되었는지 확인합니다."""
    if TIME_LEN <= 0 or FUTURE_LEN <= 0 or SAFETY_LEN <= 0 or LANE_LEN <= 0:
        raise RuntimeError(
            "WOMD lengths are not initialized. "
            "Make sure Pool initializer calls _set_womd_lengths_from_args().")


# =========================
# 데이터 구조 (맵 파싱용)
# =========================
# =========================
# 데이터 구조 (맵 파싱용)
# =========================
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundarySegmentInfo:
    """Lane 경계 구간 1개를 필요한 정보만 뽑아 저장합니다."""
    lane_start_index: int
    lane_end_index: int
    boundary_feature_id: int


@dataclass(frozen=True)
class LaneInfo:
    """차선 1개에 대한 원본 정보 묶음."""
    lane_id: int
    centerline_xy_global: np.ndarray  # shape: (P, 2), float32/float64

    # ✅ 변경: boundary_feature_id만 저장하지 말고, lane_start/end_index까지 같이 저장
    left_boundary_segments: List[BoundarySegmentInfo]
    right_boundary_segments: List[BoundarySegmentInfo]

    speed_limit_mph: float
    lane_type: int  # lane의 큰 분류(고속도로/일반도로/자전거/미정)

@dataclass(frozen=True)
class ParsedMap:
    """시나리오 맵을 필요한 형태로 미리 모아둔 결과."""
    stop_sign_xy_global: List[np.ndarray]  # each shape: (2,)
    crosswalk_polygons_xy_global: List[np.ndarray]  # each shape: (M,2)
    speed_bump_polygons_xy_global: List[np.ndarray]

    # ✅ driveway는 polygon(닫힌 영역) 형태로만 들어오는 것으로 확인되어 polygon만 저장합니다.
    driveway_polygons_xy_global: List[np.ndarray]   # each shape: (M,2)

    lanes: List[LaneInfo]

    # boundary (road_line / road_edge) polyline
    boundary_polylines_xy_global: Dict[int, np.ndarray]  # id -> shape: (K,2)

    # boundary id가 road_line인지 road_edge인지 구분
    boundary_id_to_kind: Dict[int, str]  # id -> "road_line" or "road_edge"

    # road_line / road_edge 타입(정수 enum) 저장
    road_line_type_by_id: Dict[int, int]  # road_line id -> type int
    road_edge_type_by_id: Dict[int, int]  # road_edge id -> type int

    # road_edge id 목록(road_edge 출력용)
    road_edge_ids: List[int]



# =========================
# 기본 유틸
# =========================
def ensure_dir(path: Path) -> None:
    """폴더가 없으면 생성합니다.

    Args:
        path: 만들고 싶은 폴더 경로
    """
    path.mkdir(parents=True, exist_ok=True)


def _proto_points_to_xy_array(points: Iterable[Any]) -> np.ndarray:
    """proto의 점 목록을 (N,2) numpy 배열로 바꿉니다.

    Args:
        points: 각 원소가 x, y 값을 가진 점들의 목록(반복 가능한 형태)

    Returns:
        xy: shape (N,2) float32
    """
    points_list = [(float(p.x), float(p.y)) for p in points]
    if len(points_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points_list, dtype=np.float32)  # shape (N,2)


def lane_type_value_to_one_hot_4(lane_type_value: int) -> np.ndarray:
    """차선의 큰 종류를 4칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(4칸):
        0: FREEWAY
        1: SURFACE_STREET
        2: BIKE_LANE
        3: UNDEFINED

    Args:
        lane_type_value: Waymo map proto의 lane.type 값(정수)

    Returns:
        one_hot: shape (4,) float32
    """
    # one_hot: np.ndarray, shape (4,)
    one_hot = np.zeros((4,), dtype=np.float32)

    # Waymo map.proto(또는 tf.Example 문서)에서 일반적으로 쓰이는 값:
    # 0: TYPE_UNDEFINED
    # 1: TYPE_FREEWAY
    # 2: TYPE_SURFACE_STREET
    # 3: TYPE_BIKE_LANE
    if lane_type_value == 1:
        one_hot[0] = 1.0
    elif lane_type_value == 2:
        one_hot[1] = 1.0
    elif lane_type_value == 3:
        one_hot[2] = 1.0
    else:
        one_hot[3] = 1.0
    return one_hot


def road_line_type_value_to_one_hot_10(
    road_line_type_value: int,
    has_line: bool,
) -> np.ndarray:
    """차선 경계 '선'의 종류를 10칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(10칸):
        0: BROKEN_SINGLE_WHITE
        1: SOLID_SINGLE_WHITE
        2: SOLID_DOUBLE_WHITE
        3: BROKEN_SINGLE_YELLOW
        4: BROKEN_DOUBLE_YELLOW
        5: SOLID_SINGLE_YELLOW
        6: SOLID_DOUBLE_YELLOW
        7: PASSING_DOUBLE_YELLOW
        8: UNKNOWN
        9: INVALID (해당 방향 선이 아예 없는 경우)

    Args:
        road_line_type_value: Waymo map proto의 road_line.type 값(정수)
        has_line: True면 "선이 있다", False면 "선이 없다(=INVALID)"로 처리

    Returns:
        one_hot: shape (10,) float32
    """
    # one_hot: np.ndarray, shape (10,)
    one_hot = np.zeros((10,), dtype=np.float32)

    if not has_line:
        one_hot[9] = 1.0
        return one_hot

    # Waymo map.proto에서 일반적으로 쓰이는 값:
    # 0: TYPE_UNKNOWN
    # 1..8: 각 선 종류
    mapping = {
        1: 0,  # BROKEN_SINGLE_WHITE
        2: 1,  # SOLID_SINGLE_WHITE
        3: 2,  # SOLID_DOUBLE_WHITE
        4: 3,  # BROKEN_SINGLE_YELLOW
        5: 4,  # BROKEN_DOUBLE_YELLOW
        6: 5,  # SOLID_SINGLE_YELLOW
        7: 6,  # SOLID_DOUBLE_YELLOW
        8: 7,  # PASSING_DOUBLE_YELLOW
        0: 8,  # UNKNOWN
    }
    idx = mapping.get(int(road_line_type_value), 8)  # 모르는 값은 UNKNOWN
    one_hot[idx] = 1.0
    return one_hot


from typing import Dict, List
import numpy as np


def _lane_boundary_type_index_to_one_hot_13(type_index: int) -> np.ndarray:
    """13칸짜리 0/1 벡터에서, 주어진 칸만 1로 켭니다.

    이 함수는 "왼쪽/오른쪽 경계의 대표 타입"을 저장할 때 쓰는 13칸짜리 벡터를 만듭니다.
    벡터는 길이가 13이고, 전부 0으로 시작한 뒤 type_index 위치만 1이 됩니다.

    Args:
        type_index: 0~12 범위의 정수.
            - 0~7  : 선(road_line)에서 타입이 확실한 8가지
            - 8~10 : 도로 가장자리(road_edge) 타입 3가지
            - 11   : 선(road_line)은 있는데 타입을 모르는 경우
            - 12   : 선/가장자리 자체가 아예 없는 경우(완전 없음)

    Returns:
        one_hot: shape (13,) float32
    """
    one_hot = np.zeros((13,), dtype=np.float32)  # shape: (13,)
    idx = int(type_index)
    idx = max(0, min(idx, 12))
    one_hot[idx] = 1.0
    return one_hot


def _boundary_feature_id_to_lane_boundary_type_index_13(
    boundary_feature_id: int,
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> int:
    """boundary_feature_id 1개를 13칸 타입 벡터의 '칸 번호(인덱스)'로 바꿉니다.

    이 함수는 "이 경계가 선(road_line)인지, 도로 가장자리(road_edge)인지"를 먼저 판단하고,
    그 다음에 type 값을 보고 13칸 중 어디에 해당하는지 정합니다.

    13칸 정의(0부터 시작):
        - 0~7:
            선(road_line)의 구체 타입 8가지
            [BROKEN_SINGLE_WHITE, SOLID_SINGLE_WHITE, SOLID_DOUBLE_WHITE,
             BROKEN_SINGLE_YELLOW, BROKEN_DOUBLE_YELLOW, SOLID_SINGLE_YELLOW,
             SOLID_DOUBLE_YELLOW, PASSING_DOUBLE_YELLOW]
        - 8:
            도로 가장자리(road_edge)가 있는데, 어떤 타입인지 모르는 경우
        - 9:
            도로 가장자리(road_edge) - ROAD_EDGE_BOUNDARY
        - 10:
            도로 가장자리(road_edge) - ROAD_EDGE_MEDIAN
        - 11:
            선(road_line)이 있는데, 어떤 타입인지 모르는 경우(UNKNOWN 또는 모르는 값)
        - 12:
            "아예 없음"은 여기서 만들지 않습니다(상위 함수에서 처리)

    Args:
        boundary_feature_id: 경계 feature id (정수)
        boundary_id_to_kind: id -> "road_line" 또는 "road_edge"
        road_line_type_by_id: road_line id -> type 값(정수)
        road_edge_type_by_id: road_edge id -> type 값(정수)

    Returns:
        type_index: 0~11 중 하나.
            (12=invalid은 "경계 자체가 없는 경우"라서 상위 함수에서 따로 처리합니다.)
    """
    fid = int(boundary_feature_id)

    # kind가 없을 수도 있으니(데이터가 이상하거나 일부만 파싱된 경우),
    # type dict를 한 번 더 보고 kind를 추정합니다.
    kind = boundary_id_to_kind.get(fid, "")
    if kind == "":
        if fid in road_edge_type_by_id:
            kind = "road_edge"
        elif fid in road_line_type_by_id:
            kind = "road_line"

    if kind == "road_edge":
        edge_type_value = int(road_edge_type_by_id.get(fid, 0))
        # road_edge 타입:
        # 0: UNKNOWN, 1: BOUNDARY, 2: MEDIAN
        if edge_type_value == 1:
            return 9
        if edge_type_value == 2:
            return 10
        return 8  # edge_TYPE_UNKNOWN

    # 나머지는 "선(road_line)"로 취급 (kind가 비어있어도 여기로 옴)
    line_type_value = int(road_line_type_by_id.get(fid, 0))
    # road_line 타입:
    # 1..8: 구체 타입, 0: UNKNOWN
    mapping_line = {
        1: 0,  # BROKEN_SINGLE_WHITE
        2: 1,  # SOLID_SINGLE_WHITE
        3: 2,  # SOLID_DOUBLE_WHITE
        4: 3,  # BROKEN_SINGLE_YELLOW
        5: 4,  # BROKEN_DOUBLE_YELLOW
        6: 5,  # SOLID_SINGLE_YELLOW
        7: 6,  # SOLID_DOUBLE_YELLOW
        8: 7,  # PASSING_DOUBLE_YELLOW
    }
    return int(mapping_line.get(line_type_value, 11))  # unknown(line)


def _choose_lane_side_boundary_type_one_hot_13(
    boundary_segments: List[BoundarySegmentInfo],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> np.ndarray:
    """lane 한쪽(left/right) 경계의 대표 타입을 13칸 0/1 벡터로 만듭니다.

    한 lane의 한쪽 경계는 "조각(segments)" 여러 개로 들어올 수 있습니다.
    그런데 left_line_type / right_line_type는 (lane_num, 13)처럼
    lane마다 한 줄로 저장해야 하므로, 조각들 중에서 대표 1개를 골라야 합니다.

    대표를 고르는 방법(데이터가 부분적으로만 있는 경우를 최대한 안전하게 처리):
    1) boundary_feature_id가 0 이하이거나, end_index <= start_index인 조각은 무시합니다.
    2) 남은 조각마다 "이 조각이 얼마나 긴 구간을 덮는지"를 (end-start)로 계산해서 가중치로 씁니다.
       - 즉, 더 긴 구간을 덮는 조각이 대표로 뽑히기 쉽습니다.
    3) 각 조각을 13칸 중 어디에 해당하는지 분류해서(선/가장자리 + 타입),
       그 칸의 가중치를 누적합니다.
    4) 누적 가중치가 가장 큰 칸을 대표로 선택합니다.
       - 만약 동점이면, "타입을 아는 쪽"을 우선합니다.
         (예: edge_UNKNOWN(8) vs edge_BOUNDARY(9) 동점이면 9를 선택)
    5) 유효한 조각이 하나도 없으면 "완전 없음"으로 보고 invalid(12)를 1로 합니다.
       - 단, lane_len=10 점 중 일부 구간만 경계가 없는 건 여기서 invalid가 아닙니다.
         (그건 lanes 벡터에서 일부 점이 0으로 되는 '부분 없음' 케이스입니다.)

    Args:
        boundary_segments: lane.left_boundary_segments 또는 lane.right_boundary_segments
        boundary_id_to_kind: id -> "road_line" / "road_edge"
        road_line_type_by_id: road_line id -> type 값
        road_edge_type_by_id: road_edge id -> type 값

    Returns:
        one_hot_13: shape (13,) float32
    """
    # weights: np.ndarray, shape (13,)
    weights = np.zeros((13,), dtype=np.float32)

    for seg in boundary_segments:
        fid = int(seg.boundary_feature_id)
        if fid <= 0:
            continue

        start_idx = int(seg.lane_start_index)
        end_idx = int(seg.lane_end_index)
        if end_idx <= start_idx:
            continue

        weight = float(end_idx - start_idx)  # "덮는 길이" 가중치
        type_index = _boundary_feature_id_to_lane_boundary_type_index_13(
            boundary_feature_id=fid,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )
        weights[int(type_index)] += weight

    # 유효한 조각이 1개도 없으면 -> invalid(12)
    if float(weights.sum()) <= 0.0:
        return _lane_boundary_type_index_to_one_hot_13(12)

    max_w = float(weights.max())
    candidate_indices = np.where(weights == max_w)[0].astype(np.int64).tolist()

    # 동점 처리 우선순위:
    # - "타입이 확실한 칸"(0~7, 9~10)을 먼저
    # - 그 다음 "타입을 모르는 칸"(8, 11)
    # - invalid(12)은 여기 후보에 보통 안 들어오지만, 방어적으로 가장 뒤
    def _priority_rank(idx: int) -> int:
        idx_i = int(idx)
        if 0 <= idx_i <= 7:
            return 0
        if idx_i in (9, 10):
            return 0
        if idx_i in (8, 11):
            return 1
        return 2

    chosen = sorted(candidate_indices,
                    key=lambda x: (_priority_rank(int(x)), int(x)))[0]
    return _lane_boundary_type_index_to_one_hot_13(int(chosen))


def road_edge_type_value_to_one_hot_3(road_edge_type_value: int) -> np.ndarray:
    """road_edge의 종류를 3칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(3칸):
        0: TYPE_UNKNOWN
        1: TYPE_ROAD_EDGE_BOUNDARY
        2: TYPE_ROAD_EDGE_MEDIAN

    Args:
        road_edge_type_value: Waymo map proto의 road_edge.type 값(정수)

    Returns:
        one_hot: shape (3,) float32
    """
    # one_hot: np.ndarray, shape (3,)
    one_hot = np.zeros((3,), dtype=np.float32)

    # 보통 값:
    # 0: UNKNOWN
    # 1: BOUNDARY
    # 2: MEDIAN
    if road_edge_type_value == 1:
        one_hot[1] = 1.0
    elif road_edge_type_value == 2:
        one_hot[2] = 1.0
    else:
        one_hot[0] = 1.0
    return one_hot


def _choose_road_line_type_for_lane_side(
    boundary_feature_ids: List[int],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
) -> Tuple[int, bool]:
    """lane의 한쪽(왼쪽/오른쪽)에서 '대표 선 종류'를 하나 고릅니다.

    처리 규칙(한 lane의 한쪽에 boundary_feature_id가 여러 개 있을 수 있어서 필요):
    1) boundary_feature_id 중 "road_line"인 것만 후보로 모읍니다.
    2) 후보가 하나도 없으면 => has_line=False (INVALID)
    3) 후보가 있는데, UNKNOWN(0) 말고 다른 값이 있으면:
       - UNKNOWN이 아닌 값들 중 "가장 많이 나온 값"을 대표로 씁니다.
    4) 후보가 전부 UNKNOWN(0)이면 => 대표값=0, has_line=True (UNKNOWN)

    Args:
        boundary_feature_ids: lane의 왼쪽 또는 오른쪽 boundary_feature_id 목록
        boundary_id_to_kind: boundary id가 road_line인지 road_edge인지 구분하는 dict
        road_line_type_by_id: road_line id -> road_line.type(정수) dict

    Returns:
        chosen_type_value: 대표 road_line.type 값(정수). 선이 없으면 0
        has_line: True면 선이 있음, False면 선이 없음(INVALID)
    """
    # 후보 타입들: List[int]
    candidate_types: List[int] = []
    for fid in boundary_feature_ids:
        if boundary_id_to_kind.get(int(fid), "") != "road_line":
            continue
        if int(fid) not in road_line_type_by_id:
            continue
        candidate_types.append(int(road_line_type_by_id[int(fid)]))

    if len(candidate_types) == 0:
        return 0, False  # 선이 아예 없음(INVALID)

    non_unknown = [t for t in candidate_types if t != 0]
    if len(non_unknown) == 0:
        return 0, True  # 선은 있는데 타입을 모름(UNKNOWN)

    # 가장 많이 나온 값 선택(동점이면 Counter가 먼저 본 것 기준으로 안정적으로 뽑힘)
    chosen = Counter(non_unknown).most_common(1)[0][0]
    return int(chosen), True


def build_lane_type_one_hot_array(lanes: List[LaneInfo]) -> np.ndarray:
    """lane 목록을 (lane_num, 4) lane_type 배열로 만듭니다.

    Args:
        lanes: LaneInfo 리스트

    Returns:
        lane_type: shape (lane_num, 4) float32
    """
    lane_num = len(lanes)
    lane_type = np.zeros((lane_num, 4), dtype=np.float32)  # shape (L,4)
    for i, lane in enumerate(lanes):
        lane_type[i] = lane_type_value_to_one_hot_4(int(lane.lane_type))
    return lane_type


def build_lane_line_type_arrays(
    lanes: List[LaneInfo],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """각 lane의 왼쪽/오른쪽 경계 타입을 (lane_num, 13)으로 만듭니다.

    left_line_type / right_line_type는 lanes에서 쓰는 "왼쪽/오른쪽 경계"가
    선(road_line)인지 도로 가장자리(road_edge)인지까지 포함해서 대표 타입을 저장합니다.

    13칸 정의(0부터 시작):
        0~7:
            선(road_line)의 8가지 타입
            [BROKEN_SINGLE_WHITE, SOLID_SINGLE_WHITE, SOLID_DOUBLE_WHITE,
             BROKEN_SINGLE_YELLOW, BROKEN_DOUBLE_YELLOW, SOLID_SINGLE_YELLOW,
             SOLID_DOUBLE_YELLOW, PASSING_DOUBLE_YELLOW]
        8:
            도로 가장자리(road_edge)가 있으나 타입을 모름
        9:
            도로 가장자리(road_edge) - ROAD_EDGE_BOUNDARY
        10:
            도로 가장자리(road_edge) - ROAD_EDGE_MEDIAN
        11:
            선(road_line)은 있으나 타입을 모름(UNKNOWN 또는 모르는 값)
        12:
            경계 자체가 아예 없음(선/가장자리 모두 없음)
            ※ lane_len=10 점 중 일부만 경계가 없어서 lanes 벡터가 0이 되는 경우는 여기에 해당하지 않습니다.

    Args:
        lanes: LaneInfo 리스트
        boundary_id_to_kind: boundary id -> "road_line" / "road_edge"
        road_line_type_by_id: road_line id -> type 값(정수)
        road_edge_type_by_id: road_edge id -> type 값(정수)

    Returns:
        left_line_type: shape (lane_num, 13) float32
        right_line_type: shape (lane_num, 13) float32
    """
    lane_num = int(len(lanes))

    left_line_type = np.zeros((lane_num, 13), dtype=np.float32)  # shape: (L,13)
    right_line_type = np.zeros((lane_num, 13),
                               dtype=np.float32)  # shape: (L,13)

    for i, lane in enumerate(lanes):
        left_line_type[i] = _choose_lane_side_boundary_type_one_hot_13(
            boundary_segments=lane.left_boundary_segments,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )
        right_line_type[i] = _choose_lane_side_boundary_type_one_hot_13(
            boundary_segments=lane.right_boundary_segments,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )

    return left_line_type, right_line_type


def build_road_edge_points_and_types(
    road_edge_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    road_edge_type_by_id: Dict[int, int],
    ego_xy_global: np.ndarray,  # shape (2,)
    ego_yaw_global: float,
    safety_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """road_edge를 (점들, 타입)으로 캐싱용 배열로 만듭니다.

    - road_edge는 "점들의 줄"로 주어지므로, 전체 길이를 따라 같은 간격으로 safety_len개 점을 뽑습니다.
    - 좌표는 ego 기준으로 바꿉니다.

    Args:
        road_edge_ids: road_edge feature id 목록
        boundary_polylines_xy_global: boundary id -> polyline (K,2)
        road_edge_type_by_id: road_edge id -> type 값(정수)
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 방향(라디안)
        safety_len: 출력 점 개수(요구사항: 10)

    Returns:
        road_edge_points: shape (n_road_edge, safety_len, 2) float32
        road_edge_type: shape (n_road_edge, 3) float32 (one-hot)
    """
    n_road_edge = len(road_edge_ids)
    road_edge_points = np.zeros((n_road_edge, safety_len, 2),
                                dtype=np.float32)  # (E,10,2)
    road_edge_type = np.zeros((n_road_edge, 3), dtype=np.float32)  # (E,3)

    for i, edge_id in enumerate(road_edge_ids):
        poly_xy_g = boundary_polylines_xy_global.get(
            int(edge_id), np.zeros((0, 2), dtype=np.float32))  # shape (K,2)

        poly_xy_l = transform_points_global_to_ego_local(
            poly_xy_g, ego_xy_global, ego_yaw_global)  # shape (K,2)

        sampled = resample_polyline_equal_distance(
            poly_xy_l, num_samples=safety_len,
            closed=False)  # shape (safety_len,2)

        road_edge_points[i] = sampled
        edge_type_value = int(road_edge_type_by_id.get(int(edge_id), 0))
        road_edge_type[i] = road_edge_type_value_to_one_hot_3(edge_type_value)

    return road_edge_points, road_edge_type





def list_tfrecord_files(split_dir: Path) -> List[Path]:
    """split 폴더 안의 모든 tfrecord 파일을 모읍니다.

    파일 확장자가 다양할 수 있어서(예: .tfrecord, .tfrecords, .gz),
    "파일"이면 모두 대상으로 잡되, 폴더는 제외합니다.

    Args:
        split_dir: 예) /home/user/womd_v1_3/scenario/training

    Returns:
        tfrecord 파일 경로 리스트
    """
    all_paths: List[Path] = sorted(
        [p for p in split_dir.glob("*") if p.is_file()])
    return all_paths


def guess_tfrecord_compression(file_path: Path) -> str:
    """tfrecord 파일의 압축 여부를 파일명으로 추정합니다.

    Args:
        file_path: tfrecord 파일 경로

    Returns:
        TensorFlow TFRecordDataset에 넣을 compression_type 문자열.
        - ".gz" 로 끝나면 "GZIP"
        - 아니면 "" (압축 없음)
    """
    if file_path.name.endswith(".gz"):
        return "GZIP"
    return ""


def parse_scenario_from_bytes(record_bytes: bytes) -> scenario_pb2.Scenario:
    """TFRecord의 한 레코드(bytes)를 Scenario proto로 변환합니다.

    Args:
        record_bytes: TFRecord에서 꺼낸 1개 레코드의 raw bytes

    Returns:
        Scenario proto 객체
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(record_bytes)
    return scenario


# =========================
# 좌표 변환 (ego 기준)
# =========================
def transform_points_global_to_ego_local(
    points_xy_global: np.ndarray,  # shape: (..., 2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
) -> np.ndarray:
    """전역 좌표계의 (x,y) 점들을 ego 기준 좌표계로 바꿉니다.

    변환 규칙:
    1) ego 위치를 원점으로 만들기 위해 (point - ego_pos)를 합니다.
    2) ego가 바라보는 방향이 +x가 되도록, -ego_yaw 만큼 회전합니다.

    Args:
        points_xy_global: 전역 좌표계 점들, 마지막 차원은 (x,y)
        ego_xy_global: ego의 전역 위치 (x,y)
        ego_yaw_global: ego의 전역 yaw (라디안)

    Returns:
        ego 기준 좌표계 점들, shape은 입력과 같고 마지막 차원은 (x,y)
    """
    # points_xy_global: np.ndarray, shape (..., 2)
    # ego_xy_global: np.ndarray, shape (2,)
    delta_xy = points_xy_global - ego_xy_global  # shape (..., 2)

    cos_yaw = float(np.cos(ego_yaw_global))
    sin_yaw = float(np.sin(ego_yaw_global))

    # local_x =  dx*cos + dy*sin
    # local_y = -dx*sin + dy*cos
    local_x = delta_xy[..., 0] * cos_yaw + delta_xy[..., 1] * sin_yaw
    local_y = -delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw

    out = np.stack([local_x, local_y], axis=-1).astype(np.float32)
    return out


from typing import Dict, List, Tuple
import numpy as np


def resample_polyline_equal_distance_with_segment_indices(
    points_xy: np.ndarray,  # shape: (N, 2)
    num_samples: int,
    closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """점들의 줄을 같은 간격으로 num_samples개로 만들면서, 각 샘플이 '원본의 어느 구간'에 있었는지도 함께 반환합니다.

    '구간(segment index)' 정의:
      - 원본 points_xy의 i번째 점과 (i+1)번째 점 사이를 i번째 구간이라고 봅니다.
      - seg_indices[j] = i 라면, sampled_xy[j]는 원본 (i ~ i+1) 사이에 위치했다는 뜻입니다.

    Args:
        points_xy: (N,2) 입력 점들.
        num_samples: 뽑을 점 개수.
        closed: 닫힌 모양이면 True.

    Returns:
        sampled_xy: (num_samples,2) float32. 샘플링된 점들.
        seg_indices: (num_samples,) int64.
            sampled_xy의 각 점이 원본 polyline의 몇 번째 구간(i~i+1)에 있었는지.
            points_xy가 비어있으면 전부 -1.
    """
    n = int(points_xy.shape[0])
    if n == 0:
        return (
            np.zeros((int(num_samples), 2), dtype=np.float32),
            -np.ones((int(num_samples),), dtype=np.int64),
        )

    if n == 1:
        sampled = np.repeat(points_xy.astype(np.float32),
                            repeats=int(num_samples),
                            axis=0)
        seg_indices = np.zeros((int(num_samples),), dtype=np.int64)
        return sampled, seg_indices

    points_work = points_xy
    if bool(closed):
        if not np.allclose(points_work[0], points_work[-1]):
            points_work = np.vstack([points_work, points_work[0]])

    diffs = points_work[1:] - points_work[:-1]  # (M,2)
    seg_lens = np.linalg.norm(diffs, axis=1).astype(np.float32)  # (M,)
    total_len = float(seg_lens.sum())

    if total_len < EPS:
        sampled = np.repeat(points_work[:1].astype(np.float32),
                            repeats=int(num_samples),
                            axis=0)
        seg_indices = np.zeros((int(num_samples),), dtype=np.int64)
        return sampled, seg_indices

    cum_len = np.concatenate(
        [np.array([0.0], dtype=np.float32),
         np.cumsum(seg_lens)],
        axis=0,
    )  # (M+1,)

    if bool(closed):
        target = np.linspace(0.0,
                             total_len,
                             int(num_samples),
                             endpoint=False,
                             dtype=np.float32)
    else:
        target = np.linspace(0.0,
                             total_len,
                             int(num_samples),
                             endpoint=True,
                             dtype=np.float32)

    seg_idx = np.searchsorted(cum_len, target, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, int(seg_lens.shape[0]) - 1).astype(np.int64)

    seg_start = points_work[seg_idx].astype(np.float32)  # (num_samples,2)
    seg_vec = diffs[seg_idx].astype(np.float32)  # (num_samples,2)
    seg_len = seg_lens[seg_idx].astype(np.float32)  # (num_samples,)

    alpha = (target - cum_len[seg_idx]) / np.maximum(seg_len,
                                                     EPS)  # (num_samples,)
    sampled = seg_start + seg_vec * alpha[:, None]  # (num_samples,2)

    return sampled.astype(np.float32), seg_idx


def map_center_segment_indices_to_boundary_feature_ids(
    center_seg_indices: np.ndarray,  # shape: (lane_len,)
    boundary_segments: List[BoundarySegmentInfo],
) -> np.ndarray:
    """센터 샘플이 속한 구간 인덱스로, 해당하는 boundary_feature_id를 찾습니다.

    매칭 규칙:
      - center_seg_idx = i 일 때,
        boundary segment가 lane_start_index <= i < lane_end_index 범위를 덮으면 매칭으로 봅니다.

    Args:
        center_seg_indices: (lane_len,) int64.
        boundary_segments: BoundarySegmentInfo 리스트.

    Returns:
        feature_ids: (lane_len,) int64.
            없으면 -1.
    """
    lane_len = int(center_seg_indices.shape[0])
    out = -np.ones((lane_len,), dtype=np.int64)
    if lane_len == 0 or len(boundary_segments) == 0:
        return out

    starts = np.asarray([int(s.lane_start_index) for s in boundary_segments],
                        dtype=np.int64)  # (S,)
    ends = np.asarray([int(s.lane_end_index) for s in boundary_segments],
                      dtype=np.int64)  # (S,)
    fids = np.asarray([int(s.boundary_feature_id) for s in boundary_segments],
                      dtype=np.int64)  # (S,)

    valid_seg = (fids > 0) & (ends > starts)
    if not bool(np.any(valid_seg)):
        return out

    starts = starts[valid_seg]
    ends = ends[valid_seg]
    fids = fids[valid_seg]

    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = ends[order]
    fids = fids[order]

    idx = center_seg_indices.astype(np.int64)  # (lane_len,)
    pos = np.searchsorted(starts, idx, side="right") - 1  # (lane_len,)

    valid_pos = (pos >= 0) & (idx >= 0)
    pos_clip = np.clip(pos, 0, int(starts.shape[0]) - 1)

    in_range = valid_pos & (idx < ends[pos_clip])
    out[in_range] = fids[pos_clip[in_range]]
    return out


def compute_center_tangent_and_left_normals(
        center_sampled_xy: np.ndarray,  # shape: (lane_len,2)
) -> Tuple[np.ndarray, np.ndarray]:
    """센터 샘플 점들에서 '진행 방향'과 '왼쪽 방향' 단위벡터를 계산합니다.

    - 진행 방향: 인접한 점 차이로 근사합니다.
      * 마지막 점은 뒤쪽 차이를 사용합니다.
    - 왼쪽 방향: 진행 방향 (dx,dy)를 90도 회전한 (-dy, dx) 입니다.

    Args:
        center_sampled_xy: (lane_len,2) float32.

    Returns:
        tangent_unit: (lane_len,2) float32. 진행 방향 단위벡터.
        left_normal_unit: (lane_len,2) float32. 왼쪽 방향 단위벡터.
    """
    lane_len = int(center_sampled_xy.shape[0])
    tangent = np.zeros((lane_len, 2), dtype=np.float32)

    if lane_len == 0:
        return tangent, tangent

    if lane_len == 1:
        tangent[0] = np.array([1.0, 0.0], dtype=np.float32)
    else:
        tangent[:-1] = center_sampled_xy[1:] - center_sampled_xy[:-1]
        tangent[-1] = center_sampled_xy[-1] - center_sampled_xy[-2]

    norm = np.linalg.norm(tangent, axis=1).astype(np.float32)  # (lane_len,)
    good = norm > EPS

    tangent_unit = np.zeros_like(tangent, dtype=np.float32)
    tangent_unit[good] = tangent[good] / norm[good, None]
    tangent_unit[~good] = np.array([1.0, 0.0], dtype=np.float32)

    left_normal_unit = np.stack([-tangent_unit[:, 1], tangent_unit[:, 0]],
                                axis=1).astype(np.float32)
    return tangent_unit, left_normal_unit


def closest_points_on_polyline(
        query_points_xy: np.ndarray,  # shape: (Q,2)
        polyline_xy: np.ndarray,  # shape: (K,2)
) -> Tuple[np.ndarray, np.ndarray]:
    """여러 점에서 polyline(선분들의 모음)까지의 가장 가까운 위치를 한 번에 구합니다.

    Args:
        query_points_xy: (Q,2) float32.
        polyline_xy: (K,2) float32.

    Returns:
        closest_xy: (Q,2) float32. polyline 위의 가장 가까운 점.
        dist2: (Q,) float32. 거리^2.
            polyline이 비어있으면 dist2는 inf.
    """
    q = query_points_xy.astype(np.float32)
    k = int(polyline_xy.shape[0])

    if k == 0:
        return (
            np.zeros_like(q, dtype=np.float32),
            np.full((int(q.shape[0]),), np.inf, dtype=np.float32),
        )

    if k == 1:
        p = polyline_xy[0].astype(np.float32)
        closest = np.repeat(p[None, :], repeats=int(q.shape[0]), axis=0)
        diff = q - closest
        dist2 = np.sum(diff * diff, axis=1)
        return closest.astype(np.float32), dist2.astype(np.float32)

    p0 = polyline_xy[:-1].astype(np.float32)  # (M,2)
    p1 = polyline_xy[1:].astype(np.float32)  # (M,2)
    seg = (p1 - p0).astype(np.float32)  # (M,2)
    seg_len2 = np.sum(seg * seg, axis=1).astype(np.float32)  # (M,)
    seg_len2_safe = np.maximum(seg_len2, EPS).astype(np.float32)

    diff = q[:, None, :] - p0[None, :, :]  # (Q,M,2)
    t = (np.sum(diff * seg[None, :, :], axis=2) /
         seg_len2_safe[None, :]).astype(np.float32)  # (Q,M)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    closest_all = p0[None, :, :] + t[:, :, None] * seg[None, :, :]  # (Q,M,2)
    d = q[:, None, :] - closest_all
    dist2_all = np.sum(d * d, axis=2).astype(np.float32)  # (Q,M)

    best = np.argmin(dist2_all, axis=1)  # (Q,)
    closest = closest_all[np.arange(int(q.shape[0])), best]
    dist2 = dist2_all[np.arange(int(q.shape[0])), best]
    return closest.astype(np.float32), dist2.astype(np.float32)


def compute_lane_boundary_vectors_for_side(
    center_sampled_xy_local: np.ndarray,  # shape: (lane_len,2)
    center_seg_indices: np.ndarray,  # shape: (lane_len,)
    center_left_normals_unit: np.ndarray,  # shape: (lane_len,2)
    boundary_segments: List[BoundarySegmentInfo],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
    boundary_polyline_local_cache: Dict[int, np.ndarray],
    side: str,
    max_dist_road_line_m: float = 8.0,
    max_dist_road_edge_m: float = 15.0,
) -> np.ndarray:
    """lane의 한쪽(left/right)에 대해 center 샘플 점 -> 경계 polyline 최소거리 벡터를 구합니다.

    Args:
        center_sampled_xy_local: (lane_len,2) float32. center 샘플 점(ego 기준).
        center_seg_indices: (lane_len,) int64. 각 샘플이 원본 center polyline의 어느 구간(i~i+1)인지.
        center_left_normals_unit: (lane_len,2) float32. 각 샘플에서의 '왼쪽 방향' 단위벡터.
        boundary_segments: left_boundaries 또는 right_boundaries 리스트.
        boundary_polylines_xy_global: boundary_feature_id -> (K,2) global polyline dict.
        boundary_id_to_kind: boundary_feature_id -> "road_line" or "road_edge".
        ego_xy_global: (2,) ego 전역 위치.
        ego_yaw_global: ego 전역 yaw(rad).
        boundary_polyline_local_cache: boundary_feature_id -> (K,2) ego local cache.
        side: "left" 또는 "right".
        max_dist_road_line_m: road_line 허용 최대 거리.
        max_dist_road_edge_m: road_edge 허용 최대 거리(더 넉넉).

    Returns:
        vec_xy: (lane_len,2) float32.
            못 찾으면 (0,0).
    """
    lane_len = int(center_sampled_xy_local.shape[0])
    out = np.zeros((lane_len, 2), dtype=np.float32)
    if lane_len == 0:
        return out

    feature_ids = map_center_segment_indices_to_boundary_feature_ids(
        center_seg_indices=center_seg_indices,
        boundary_segments=boundary_segments,
    )  # (lane_len,)

    valid_mask = feature_ids > 0
    if not bool(np.any(valid_mask)):
        return out

    # 성능: 같은 boundary_feature_id를 공유하는 점들을 묶어서 계산
    unique_fids = np.unique(feature_ids[valid_mask]).astype(np.int64)
    for fid in unique_fids.tolist():
        idxs = np.where(feature_ids == int(fid))[0]
        if idxs.size == 0:
            continue

        # polyline ego 변환 캐시
        poly_local = boundary_polyline_local_cache.get(int(fid), None)
        if poly_local is None:
            poly_global = boundary_polylines_xy_global.get(int(fid), None)
            if poly_global is None:
                continue
            poly_local = transform_points_global_to_ego_local(
                poly_global, ego_xy_global, ego_yaw_global)  # (K,2)
            boundary_polyline_local_cache[int(fid)] = poly_local.astype(
                np.float32)

        if poly_local.ndim != 2 or poly_local.shape[1] != 2 or poly_local.shape[
                0] == 0:
            continue

        query = center_sampled_xy_local[idxs].astype(np.float32)  # (Q,2)
        closest, dist2 = closest_points_on_polyline(query,
                                                    poly_local)  # (Q,2), (Q,)
        vec = (closest - query).astype(np.float32)  # (Q,2)
        dist = np.sqrt(dist2).astype(np.float32)  # (Q,)

        kind = boundary_id_to_kind.get(int(fid), "road_line")
        max_dist = float(
            max_dist_road_edge_m) if kind == "road_edge" else float(
                max_dist_road_line_m)

        invalid = dist > max_dist

        # 방향 체크: dot(vec, left_normal) 부호로 left/right 판별
        normals = center_left_normals_unit[idxs].astype(np.float32)  # (Q,2)
        dot = np.sum(vec * normals, axis=1).astype(np.float32)  # (Q,)

        if side == "left":
            invalid = np.logical_or(invalid, dot <= 0.0)
        elif side == "right":
            invalid = np.logical_or(invalid, dot >= 0.0)
        else:
            raise ValueError(f"side must be 'left' or 'right'. got={side}")

        if bool(np.any(invalid)):
            vec[invalid] = 0.0

        out[idxs] = vec

    return out


def transform_vectors_global_to_ego_local(
    vectors_xy_global: np.ndarray,  # shape: (..., 2)
    ego_yaw_global: float,
) -> np.ndarray:
    """전역 좌표계의 (vx,vy) 같은 '방향/속도 벡터'를 ego 기준으로 바꿉니다.

    점 변환과 달리, 벡터는 위치 이동(빼기)을 하지 않고 회전만 합니다.

    Args:
        vectors_xy_global: 전역 좌표계 벡터들, 마지막 차원은 (x,y)
        ego_yaw_global: ego의 전역 yaw (라디안)

    Returns:
        ego 기준 좌표계 벡터들, shape은 입력과 같고 마지막 차원은 (x,y)
    """
    # vectors_xy_global: np.ndarray, shape (..., 2)
    cos_yaw = float(np.cos(ego_yaw_global))
    sin_yaw = float(np.sin(ego_yaw_global))

    local_x = vectors_xy_global[..., 0] * cos_yaw + vectors_xy_global[
        ..., 1] * sin_yaw
    local_y = -vectors_xy_global[..., 0] * sin_yaw + vectors_xy_global[
        ..., 1] * cos_yaw

    out = np.stack([local_x, local_y], axis=-1).astype(np.float32)
    return out


def wrap_angle_np(angles_rad: np.ndarray) -> np.ndarray:
    """각도(rad)를 [-pi, pi) 범위로 정리합니다.

    Args:
        angles_rad: 각도 배열 (어떤 shape이든 가능)

    Returns:
        같은 shape의 각도 배열 (float32)
    """
    wrapped = (angles_rad + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped.astype(np.float32)


def wrap_angle_with_repo_fn(angles_rad: np.ndarray) -> np.ndarray:
    """레포에 있는 wrap_angle(torch 기반)을 이용해 각도를 정리합니다.

    Args:
        angles_rad: 각도 배열 (shape 자유)

    Returns:
        [-pi, pi) 범위로 정리된 각도 (float32)
    """
    # angles_rad_t: torch.Tensor, shape same as angles_rad
    angles_rad_t = torch.from_numpy(angles_rad.astype(np.float32))
    wrapped_t = wrap_angle(angles_rad_t)
    return wrapped_t.numpy().astype(np.float32)


# =========================
# 보간(빈 프레임 채우기)
# =========================
from typing import Tuple
import numpy as np


def interpolate_linear_sequence(
    values: np.ndarray,  # shape: (T, D)
    valid_mask: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray]:
    """유효한 시점들 사이의 빈 칸을 '직선으로' 채웁니다. (numpy만 사용)

    이 함수는 길이가 짧은 시계열(T=21/80 같은)에서, 무거운 외부 보간 도구를 매번 만드는 대신
    numpy의 간단한 방식으로 빠르게 채우는 버전입니다.

    규칙(기존과 동일):
    - 유효한 값이 2개 이상이면:
      - 첫 유효 시점 ~ 마지막 유효 시점 사이를 직선으로 연결해 채웁니다.
      - 그 구간만 유효(True)로 표시하고, 구간 밖은 0으로 둡니다.
    - 유효한 값이 1개뿐이면:
      - 그 시점만 채우고 나머지는 0으로 둡니다.
    - 유효한 값이 0개면:
      - 전부 0으로 둡니다.

    Args:
        values: (T, D) 값 배열.
        valid_mask: (T,) 유효 여부. True인 시점만 '실제 값이 있다'고 봅니다.

    Returns:
        filled_values: (T, D) float32. 채워진 결과.
        filled_valid_mask: (T,) bool. 채워진 구간을 True로 표시한 마스크.
    """
    # values: np.ndarray, shape (T, D)
    # valid_mask: np.ndarray, shape (T,)
    time_len = int(values.shape[0])
    feat_dim = int(values.shape[1]) if values.ndim == 2 else 0

    out = np.zeros((time_len, feat_dim), dtype=np.float32)  # shape: (T, D)
    out_valid = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    if time_len == 0 or feat_dim == 0:
        return out, out_valid

    valid_indices = np.flatnonzero(valid_mask.astype(bool)).astype(np.int64)  # shape: (K,)
    if valid_indices.size == 0:
        return out, out_valid

    if valid_indices.size == 1:
        t = int(valid_indices[0])
        out[t] = values[t].astype(np.float32)
        out_valid[t] = True
        return out, out_valid

    t_start = int(valid_indices[0])
    t_end = int(valid_indices[-1])

    # 보간할 시간 구간
    t_query = np.arange(t_start, t_end + 1, dtype=np.float32)  # shape: (M,)
    t_known = valid_indices.astype(np.float32)  # shape: (K,)

    known_values = values[valid_mask.astype(bool)].astype(np.float32)  # shape: (K, D)

    # D가 보통 2~3처럼 작기 때문에, D만큼의 작은 루프는 부담이 거의 없습니다.
    out_slice = np.zeros((int(t_query.shape[0]), feat_dim), dtype=np.float32)  # shape: (M, D)
    for d in range(feat_dim):
        out_slice[:, d] = np.interp(t_query, t_known, known_values[:, d]).astype(np.float32)

    out[t_start:t_end + 1] = out_slice
    out_valid[t_start:t_end + 1] = True
    return out, out_valid



from typing import Tuple
import numpy as np


def interpolate_linear_angle(
    angles_rad: np.ndarray,  # shape: (T,)
    valid_mask: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray]:
    """각도 시계열의 빈 칸을 직선으로 채웁니다. (numpy만 사용)

    각도는 -pi와 pi 근처에서 값이 갑자기 튀어 보일 수 있습니다.
    그래서 유효한 각도들만 뽑아서 "자연스럽게 이어지도록" 한 번 펼친 뒤,
    직선으로 채우고 마지막에 [-pi, pi) 범위로 다시 정리합니다.

    Args:
        angles_rad: (T,) 각도 배열(라디안).
        valid_mask: (T,) 유효 여부.

    Returns:
        filled_angles: (T,) float32. 채워진 각도.
        filled_valid_mask: (T,) bool. 채워진 구간을 True로 표시한 마스크.
    """
    # angles_rad: np.ndarray, shape (T,)
    # valid_mask: np.ndarray, shape (T,)
    time_len = int(angles_rad.shape[0])

    out = np.zeros((time_len,), dtype=np.float32)  # shape: (T,)
    out_valid = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    if time_len == 0:
        return out, out_valid

    valid_indices = np.flatnonzero(valid_mask.astype(bool)).astype(np.int64)  # shape: (K,)
    if valid_indices.size == 0:
        return out, out_valid

    if valid_indices.size == 1:
        t = int(valid_indices[0])
        out[t] = float(angles_rad[t])
        out_valid[t] = True
        return out, out_valid

    t_start = int(valid_indices[0])
    t_end = int(valid_indices[-1])

    valid_angles = angles_rad[valid_mask.astype(bool)].astype(np.float32)  # shape: (K,)
    valid_angles_unwrapped = np.unwrap(valid_angles).astype(np.float32)  # shape: (K,)

    t_query = np.arange(t_start, t_end + 1, dtype=np.float32)  # shape: (M,)
    t_known = valid_indices.astype(np.float32)  # shape: (K,)

    out[t_start:t_end + 1] = np.interp(t_query, t_known, valid_angles_unwrapped).astype(np.float32)
    out_valid[t_start:t_end + 1] = True

    # torch를 거치지 않고 numpy로 바로 [-pi, pi) 정리
    out = wrap_angle_np(out)
    return out, out_valid



# =========================
# 에이전트(ego/neighbor) 특징 만들기
# =========================
def make_agent_type_one_hot(object_type_minus1: int) -> np.ndarray:
    """에이전트 타입을 one-hot(3개)로 만듭니다.

    매핑 규칙:
    - 0: VEHICLE
    - 1: PEDESTRIAN
    - 2: CYCLIST
    그 외는 [0,0,0]으로 둡니다.

    Args:
        object_type_minus1: scenario Track의 object_type에서 1을 뺀 값

    Returns:
        one_hot: shape (3,), float32
    """
    # one_hot: np.ndarray, shape (3,)
    one_hot = np.zeros((3,), dtype=np.float32)
    if object_type_minus1 == 0:
        one_hot[0] = 1.0
    elif object_type_minus1 == 1:
        one_hot[1] = 1.0
    elif object_type_minus1 == 2:
        one_hot[2] = 1.0
    return one_hot


def get_num_steps_from_scenario(scenario: Any) -> int:
    """시나리오의 “전체 시간 길이(step 수)”를 안전하게 구합니다.

    어떤 시나리오는 track마다 길이가 다를 수 있습니다.
    그래서 한 트랙(track0 등)의 길이에 기대면 전체가 잘릴 수 있습니다.
    이 함수는 다음 순서로 전체 길이를 정합니다.

    1) 시나리오에 timestamps_seconds가 있으면 그 길이를 사용
    2) 없으면 모든 트랙의 states 길이 중 “가장 긴 길이”를 사용
    3) 그래도 0이면 에러

    Args:
        scenario: WOMD 시나리오 객체

    Returns:
        num_steps: 전체 step 수(정수)

    Raises:
        ValueError: step 수를 0으로밖에 구할 수 없을 때
    """
    timestamps = getattr(scenario, "timestamps_seconds", None)
    if timestamps is not None:
        num_steps = int(len(timestamps))
        if num_steps > 0:
            return num_steps

    tracks = getattr(scenario, "tracks", [])
    if len(tracks) > 0:
        lengths = [int(len(getattr(tr, "states", []))) for tr in tracks]
        num_steps = int(max(lengths)) if len(lengths) > 0 else 0
        if num_steps > 0:
            return num_steps

    raise ValueError("Scenario has zero steps (invalid WOMD record).")


def build_time_indices(
    current_time_index: int,
    num_total_steps: int,
    desired_past_len: int,
    desired_future_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """현재 시점을 기준으로 “최근 과거 / 바로 다음 미래” 인덱스를 만듭니다.

    이 함수는 과거 인덱스의 “끝이 항상 현재 시점”이 되도록 만듭니다.
    예를 들어 현재가 10이고 과거 길이가 5면, [6,7,8,9,10]처럼 “최근 구간”을 씁니다.
    데이터가 부족한 앞쪽은 -1로 채웁니다.

    Args:
        current_time_index: 현재 시점 인덱스(정수)
        num_total_steps: 전체 시점 개수(정수)
        desired_past_len: 과거 길이(TIME_LEN)
        desired_future_len: 미래 길이(FUTURE_LEN)

    Returns:
        past_indices: shape (desired_past_len,), dtype int64. 없으면 -1.
        future_indices: shape (desired_future_len,), dtype int64. 없으면 -1.
    """
    current_t = int(current_time_index)
    num_steps = int(num_total_steps)
    past_len = int(desired_past_len)
    future_len = int(desired_future_len)

    past_indices = -np.ones((past_len,), dtype=np.int64)  # shape (past_len,)
    future_indices = -np.ones(
        (future_len,), dtype=np.int64)  # shape (future_len,)

    if num_steps <= 0:
        return past_indices, future_indices

    current_t = max(0, min(current_t, num_steps - 1))

    # 과거: "끝이 current_t"가 되도록 최근 구간 선택
    start_past = max(0, current_t - past_len + 1)
    selected_past = np.arange(start_past, current_t + 1,
                              dtype=np.int64)  # shape (K,)
    past_indices[-selected_past.shape[0]:] = selected_past

    # 미래: current_t+1부터 채우고 부족분은 -1 유지
    start_future = current_t + 1
    end_future = min(num_steps, start_future + future_len)
    if start_future < num_steps:
        future_indices[:(end_future - start_future)] = np.arange(start_future,
                                                                 end_future,
                                                                 dtype=np.int64)

    return past_indices, future_indices


from typing import Tuple
import numpy as np


def gather_agent_sequence_by_indices(
    agent_idx: int,
    time_indices: np.ndarray,  # shape: (T,), -1이면 없음
    pos_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    vel_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    yaw_local_all: np.ndarray,  # shape: (N, S)
    width_length_all: np.ndarray,  # shape: (N, S, 2)  (width, length)
    valid_all: np.ndarray,  # shape: (N, S)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """원하는 시간 인덱스들만 뽑아서 (x,y, yaw, v, w/l, valid)를 만듭니다. (numpy로 한 번에 처리)

    기존 구현은 T(21/80) 길이만큼 파이썬 for문을 돌면서 한 칸씩 복사했습니다.
    여기서는 time_indices에서 "실제로 존재하는 인덱스"만 골라서 numpy가 한 번에 가져오도록 바꿉니다.

    Args:
        agent_idx: 에이전트 인덱스.
        time_indices: (T,) 시간 인덱스 배열. -1은 데이터 없음.
        pos_xy_local_all: (N, S, 2) ego 기준 위치.
        vel_xy_local_all: (N, S, 2) ego 기준 속도.
        yaw_local_all: (N, S) ego 기준 yaw.
        width_length_all: (N, S, 2) (width, length).
        valid_all: (N, S) 유효 여부.

    Returns:
        pos_xy_seq: (T, 2) float32
        vel_xy_seq: (T, 2) float32
        yaw_seq: (T,) float32
        width_length_seq: (T, 2) float32
        valid_seq: (T,) bool
    """
    # time_indices: np.ndarray, shape (T,)
    time_len = int(time_indices.shape[0])
    num_steps = int(pos_xy_local_all.shape[1])  # S

    pos_xy_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    vel_xy_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    yaw_seq = np.zeros((time_len,), dtype=np.float32)  # shape: (T,)
    width_length_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    valid_seq = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    # 유효한(0 이상, 범위 안) 인덱스만 사용
    src_mask = (time_indices >= 0) & (time_indices < num_steps)  # shape: (T,)
    if not bool(np.any(src_mask)):
        return pos_xy_seq, vel_xy_seq, yaw_seq, width_length_seq, valid_seq

    src_idx = time_indices[src_mask].astype(np.int64)  # shape: (K,)

    pos_xy_seq[src_mask] = pos_xy_local_all[int(agent_idx), src_idx]  # (K,2)
    vel_xy_seq[src_mask] = vel_xy_local_all[int(agent_idx), src_idx]  # (K,2)
    yaw_seq[src_mask] = yaw_local_all[int(agent_idx), src_idx].astype(np.float32)  # (K,)
    width_length_seq[src_mask] = width_length_all[int(agent_idx), src_idx]  # (K,2)
    valid_seq[src_mask] = valid_all[int(agent_idx), src_idx].astype(bool)  # (K,)

    return pos_xy_seq, vel_xy_seq, yaw_seq, width_length_seq, valid_seq



import numpy as np

def build_past_current_and_future_vxy_from_feat_11(
    past_feat_11: np.ndarray,   # shape: (T, 11) or (A, T, 11)
    future_feat_11: np.ndarray, # shape: (F, 11) or (A, F, 11)
) -> np.ndarray:
    """11차원 특징에서 v_x, v_y만 뽑아 (과거+현재+미래)로 이어붙입니다.

    이 프로젝트의 11차원 agent 특징은 아래 순서를 가집니다.
      [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]
    따라서 v_x, v_y는 항상 [:, 4:6] 위치에 이미 들어 있습니다.

    이 함수는
      - 과거+현재(past_feat_11)에서 v_x,v_y를 뽑고
      - 미래(future_feat_11)에서 v_x,v_y를 뽑아서
      - 시간축으로 그대로 이어붙여 반환합니다.

    중요한 점:
      - v_x, v_y는 "데이터셋에 라벨로 들어있는 속도"를 그대로 사용합니다.
      - 위치를 미분해서 속도를 새로 계산하지 않습니다.
      - 입력에서 무효 시점이 0으로 채워져 있으면, 출력도 그 시점은 0으로 유지됩니다.

    Args:
        past_feat_11 (np.ndarray):
            과거+현재 11차원 특징.
            shape: (T, 11) 또는 (A, T, 11)
        future_feat_11 (np.ndarray):
            미래 11차원 특징(현재 점 제외).
            shape: (F, 11) 또는 (A, F, 11)

    Returns:
        np.ndarray:
            vxy 시계열.
            shape: (T+F, 2) 또는 (A, T+F, 2)
            dtype: float32
    """
    past = np.asarray(past_feat_11, dtype=np.float32)
    future = np.asarray(future_feat_11, dtype=np.float32)

    if past.ndim not in (2, 3):
        raise ValueError(
            f"past_feat_11 must be (T,11) or (A,T,11). got shape={past.shape}"
        )
    if future.ndim != past.ndim:
        raise ValueError(
            f"future_feat_11 ndim mismatch. past={past.shape}, future={future.shape}"
        )

    if past.shape[-1] < 6 or future.shape[-1] < 6:
        raise ValueError(
            f"feat_11 last dim must be >= 6 (to include v_x,v_y). "
            f"past.shape={past.shape}, future.shape={future.shape}"
        )

    # past/future의 "시간축(-2)과 feature축(-1)을 제외한 앞쪽 shape"가 같아야 합니다.
    # - (T,11)에서는 past.shape[:-2] == () 이고 future도 동일
    # - (A,T,11)에서는 past.shape[:-2] == (A,) 이고 future도 (A,) 이어야 함
    if past.shape[:-2] != future.shape[:-2]:
        raise ValueError(
            f"leading dims mismatch (excluding time/feat). "
            f"past.shape={past.shape}, future.shape={future.shape}"
        )

    past_vxy = past[..., 4:6]    # shape: (T,2) or (A,T,2)
    future_vxy = future[..., 4:6]  # shape: (F,2) or (A,F,2)

    # 시간축은 항상 -2 입니다. (마지막 축은 2)
    vxy = np.concatenate([past_vxy, future_vxy], axis=-2).astype(np.float32)
    return vxy


def pack_agent_features_11(
    pos_xy: np.ndarray,  # shape: (T, 2)
    vel_xy: np.ndarray,  # shape: (T, 2)
    yaw_rad: np.ndarray,  # shape: (T,)
    width_length: np.ndarray,  # shape: (T, 2)
    agent_one_hot: np.ndarray,  # shape: (3,)
    valid_mask: np.ndarray,  # shape: (T,)
) -> np.ndarray:
    """요구사항의 11차원 포맷으로 묶습니다. (불필요한 복사 최소화)

    [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]

    Args:
        pos_xy: (T,2) 위치(ego 기준).
        vel_xy: (T,2) 속도(ego 기준).
        yaw_rad: (T,) yaw(ego 기준).
        width_length: (T,2) (width,length).
        agent_one_hot: (3,) 타입 one-hot.
        valid_mask: (T,) 유효 여부.

    Returns:
        feat_11: (T,11) float32
    """
    time_len = int(pos_xy.shape[0])
    feat_11 = np.zeros((time_len, 11), dtype=np.float32)  # shape: (T, 11)

    idx = valid_mask.astype(bool)  # shape: (T,)
    if not bool(np.any(idx)):
        return feat_11

    # 필요한 시점만 cos/sin 계산
    yaw_valid = yaw_rad[idx].astype(np.float32)  # shape: (K,)
    cos_yaw = np.cos(yaw_valid).astype(np.float32)  # shape: (K,)
    sin_yaw = np.sin(yaw_valid).astype(np.float32)  # shape: (K,)

    feat_11[idx, 0:2] = pos_xy[idx]
    feat_11[idx, 2] = cos_yaw
    feat_11[idx, 3] = sin_yaw
    feat_11[idx, 4:6] = vel_xy[idx]
    feat_11[idx, 6:8] = width_length[idx]

    # repeat로 큰 배열을 만들지 않고, (3,) 값을 (K,3)에 자연스럽게 채움
    feat_11[idx, 8:11] = agent_one_hot.astype(np.float32)  # shape: (K,3)

    return feat_11

from typing import List
import numpy as np


def build_agent_type_one_hot_all(
    object_type_minus1_all: np.ndarray,  # shape: (N,)
    ego_track_index: int,
) -> np.ndarray:
    """모든 트랙의 타입 one-hot(3)을 한 번에 만듭니다.

    기존 방식은 트랙 수(N)만큼 파이썬 for문을 돌며 one-hot을 채웠습니다.
    여기서는 numpy로 한 번에 채워서 불필요한 파이썬 반복을 줄입니다.

    Args:
        object_type_minus1_all: (N,) 각 트랙의 타입 값.
            - 0: VEHICLE
            - 1: PEDESTRIAN
            - 2: CYCLIST
            - 그 외 값은 "모름"으로 보고 [0,0,0] 유지
        ego_track_index: ego 트랙 인덱스(정수). ego는 항상 VEHICLE로 고정합니다.

    Returns:
        one_hot_all: (N, 3) float32.
            각 트랙마다 타입 one-hot. (ego는 [1,0,0]으로 강제)
    """
    # object_type_minus1_all: np.ndarray, shape (N,)
    num_tracks = int(object_type_minus1_all.shape[0])
    one_hot_all = np.zeros((num_tracks, 3), dtype=np.float32)  # shape: (N,3)

    valid_mask = (object_type_minus1_all >= 0) & (object_type_minus1_all <= 2)  # shape: (N,)
    if bool(np.any(valid_mask)):
        rows = np.flatnonzero(valid_mask).astype(np.int64)  # shape: (K,)
        cols = object_type_minus1_all[valid_mask].astype(np.int64)  # shape: (K,)
        one_hot_all[rows, cols] = 1.0

    if 0 <= int(ego_track_index) < num_tracks:
        one_hot_all[int(ego_track_index)] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return one_hot_all


def collect_neighbor_indices_at_current(
    valid_all: np.ndarray,  # shape: (N, S)
    current_time_index: int,
    ego_track_index: int,
) -> np.ndarray:
    """현재 시점에서 "존재하는 neighbor" 트랙 인덱스를 모읍니다.

    규칙:
    - valid_all[:, current_time_index]가 True인 트랙만 선택합니다.
    - ego 트랙은 제외합니다.

    Args:
        valid_all: (N, S) bool. 각 트랙의 각 시점 유효 여부.
        current_time_index: 현재 시점 인덱스(정수).
        ego_track_index: ego 트랙 인덱스(정수).

    Returns:
        neighbor_indices: (A,) int64.
            현재 시점에 존재하는 neighbor 트랙 인덱스들.
    """
    # valid_all: np.ndarray, shape (N, S)
    current_t = int(current_time_index)
    is_valid_now = valid_all[:, current_t].astype(bool)  # shape: (N,)

    indices = np.flatnonzero(is_valid_now).astype(np.int64)  # shape: (A_with_ego,)
    indices = indices[indices != int(ego_track_index)]  # shape: (A,)
    return indices


def _compute_valid_mask_from_prefix_nonzero(
    features: np.ndarray,
    prefix_dim: int,
) -> np.ndarray:
    """마지막 차원의 앞쪽(prefix) 값이 '전부 0인지'로 유효 마스크를 만듭니다.

    규칙:
      - 마지막 차원(feature_dim) 중 앞 prefix_dim개가 전부 0이면 invalid(False)
      - 하나라도 0이 아닌 값이 있으면 valid(True)

    예시:
      - ego_agent_past: (time_len, 11) -> (time_len,) bool
      - neighbor_agents_past: (A, time_len, 11) -> (A, time_len) bool
      - lanes: (L, lane_len, 12) -> (L, lane_len) bool

    Args:
        features: numpy 배열. shape: (..., feature_dim)
            - 마지막 차원이 feature_dim이어야 합니다.
        prefix_dim: 마지막 차원에서 앞쪽으로 검사할 길이.
            - agent feature(11)에서는 보통 8
            - lane/route feature(12)에서도 보통 8
            - (x,y,heading) 3차원에서는 3

    Returns:
        valid_mask: bool 배열. shape: features.shape[:-1]
    """
    # features: np.ndarray, shape (..., feature_dim)
    if features.ndim < 1:
        raise ValueError(
            f"features must have at least 1 dim. got shape={features.shape}")

    feature_dim = int(features.shape[-1])
    k = int(prefix_dim)
    if k <= 0 or k > feature_dim:
        raise ValueError(
            f"prefix_dim must be in [1, feature_dim]. got prefix_dim={k}, feature_dim={feature_dim}"
        )

    prefix = features[..., :k]  # shape (..., k)
    # abs_sum: shape (...,)
    abs_sum = np.sum(np.abs(prefix), axis=-1)
    valid_mask = abs_sum > 0.0
    return valid_mask.astype(bool)

def assert_cur_future_valid_mask_np(
    valid_bpt,
    *,
    context: str = "savgol_filter_for_control",
) -> None:
    """유효 마스크가 행마다 True*False* (단조 감소)인지 검사합니다(NumPy 버전).

    의미
    ----
    시간축을 왼쪽→오른쪽으로 볼 때,
    한 번 False(무효)가 된 이후에는 다시 True(유효)로 돌아오면 안 됩니다.
    즉, 각 (b, p) 행이 아래 패턴만 허용됩니다.

      - 허용: [True, True, True, False, False]
      - 금지: [True, False, True, False]  (중간에 구멍)
      - 금지: [False, True, True, ...]   (무효였다가 다시 유효)

    Args:
        valid_bpt (np.ndarray):
            유효 마스크.
            shape: (B, Pnn, T1)
            dtype: bool (또는 0/1 같은 값이면 bool로 해석됨)
            - True: 유효
            - False: 무효
        context (str):
            에러 메시지에 표시할 호출 위치 문자열.

    Raises:
        ValueError:
            0→1 전이(False→True)가 하나라도 발견되면 발생합니다.
    """
    v0 = np.asarray(valid_bpt)
    if v0.ndim != 3:
        raise ValueError(
            f"valid_bpt must be (B,Pnn,T1). got shape={v0.shape}")

    B, Pnn, T1 = v0.shape
    v = v0.astype(np.bool_).reshape(-1, T1).astype(np.int8)  # (B*Pnn, T1)

    # d[t] = v[t+1] - v[t]  ->  (0→1) 이면 +1
    d = v[:, 1:] - v[:, :-1]  # (B*Pnn, T1-1)
    has_01 = (d > 0).any(axis=1)  # (B*Pnn,)

    if np.any(has_01):
        bad_idx = np.nonzero(has_01)[0]  # (N_bad,)
        max_show = min(int(bad_idx.size), 8)
        bad_idx_sample = bad_idx[:max_show]

        b_list = (bad_idx_sample // Pnn).tolist()
        p_list = (bad_idx_sample % Pnn).tolist()

        raise ValueError(
            f"[{context}] near_cur_future_valid violates the per-row monotonic constraint (True* then False*).\n"
            f"A 0→1 transition was detected. Number of invalid rows={int(bad_idx.size)},\n"
            f"example (b,p)={list(zip(b_list, p_list))}.\n"
            f"Internal holes (1→0→1) or becoming valid after being invalid (0→1) are not allowed."
        )

def pack_agent_future_3(
        pos_xy: np.ndarray,  # shape: (T,2)
        yaw_rad: np.ndarray,  # shape: (T,)
        valid_mask: np.ndarray,  # shape: (T,)
) -> np.ndarray:
    """요구사항의 미래 GT 3차원 포맷으로 묶습니다.

    [x, y, heading]

    Args:
        pos_xy: (T,2) 위치(ego 기준)
        yaw_rad: (T,) heading(ego 기준)
        valid_mask: (T,) 유효 여부

    Returns:
        feat_3: (T,3) float32
    """
    time_len = int(pos_xy.shape[0])
    feat_3 = np.zeros((time_len, 3), dtype=np.float32)  # shape (T,3)
    if not np.any(valid_mask):
        return feat_3

    idx = valid_mask
    feat_3[idx, 0:2] = pos_xy[idx]
    feat_3[idx, 2] = yaw_rad[idx]
    return feat_3


def build_interpolated_agent_traj(
    pos_xy_raw: np.ndarray,  # shape: (T,2)
    vel_xy_raw: np.ndarray,  # shape: (T,2)
    yaw_raw: np.ndarray,  # shape: (T,)
    width_length_raw: np.ndarray,  # shape: (T,2)
    valid_raw: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x,y, v, yaw, width/length) 시계열을 보간 규칙에 맞게 정리합니다.

    Args:
        pos_xy_raw: (T,2) 원본 위치 (없는 곳은 0)
        vel_xy_raw: (T,2) 원본 속도 (없는 곳은 0)
        yaw_raw: (T,) 원본 yaw (없는 곳은 0)
        width_length_raw: (T,2) 원본 (width,length)
        valid_raw: (T,) 유효 여부

    Returns:
        pos_xy: (T,2) 보간 후 위치
        vel_xy: (T,2) 보간 후 속도
        yaw: (T,) 보간 후 yaw
        width_length: (T,2) 보간 후 (width,length)
        valid_mask: (T,) 보간으로 채워진 구간 True
    """
    # pos_xy: (T,2)
    pos_xy, valid_filled = interpolate_linear_sequence(pos_xy_raw, valid_raw)
    vel_xy, _ = interpolate_linear_sequence(vel_xy_raw, valid_raw)
    width_length, _ = interpolate_linear_sequence(width_length_raw, valid_raw)

    # yaw: (T,)
    yaw, valid_filled_yaw = interpolate_linear_angle(yaw_raw, valid_raw)

    # 위치/각도 보간 구간이 다르면 이상하므로, 둘 다 True인 곳만 최종 valid로 사용
    valid_mask = np.logical_and(valid_filled, valid_filled_yaw)  # shape (T,)
    return pos_xy, vel_xy, yaw, width_length, valid_mask


# =========================
# 트랙/역할 디코딩
# =========================


def _track_states_to_numpy(
    track: Any,
    num_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track 1개의 states를 numpy로 빠르게 변환합니다.

    Args:
        track: scenario.tracks의 원소 (proto Track)
        num_steps: 시나리오의 전체 step 수 (보통 91)

    Returns:
        states_9: (S, 9) float32
            [center_x, center_y, center_z, length, width, height, heading, v_x, v_y]
        valid: (S,) bool
    """
    track_states = track.states
    n = len(track_states)

    # (S,9)/(S,) 기본값은 0/False로 패딩
    states_9 = np.zeros((num_steps, 9), dtype=np.float32)
    valid = np.zeros((num_steps,), dtype=bool)
    if n == 0:
        return states_9, valid

    # 한 번의 리스트 컴프리헨션으로 (n, 10) = 9개 상태 + valid(0/1)을 같이 만들기
    # -> 내부 for문에서 states[i,t,k]를 9번 찍는 것보다 보통 더 빠릅니다.
    packed_10 = np.asarray(
        [
            (
                float(st.center_x),
                float(st.center_y),
                float(st.center_z),
                float(st.length),
                float(st.width),
                float(st.height),
                float(st.heading),
                float(st.velocity_x),
                float(st.velocity_y),
                float(st.valid),  # bool -> 0.0/1.0
            ) for st in track_states
        ],
        dtype=np.float32,
    )  # shape: (n, 10)

    # 혹시 track마다 states 길이가 다를 수 있어 안전하게 min으로
    m = min(n, num_steps)
    states_9[:m] = packed_10[:m, :9]
    valid[:m] = packed_10[:m, 9] != 0.0
    return states_9, valid


from typing import Any, Dict, Set
import numpy as np


def _collect_track_indices_to_predict(scenario: Any,
                                      num_tracks: int) -> Set[int]:
    """tracks_to_predict에서 'tracks 인덱스'들을 안전하게 모읍니다.

    Args:
        scenario: WOMD scenario 객체.
        num_tracks: scenario.tracks의 길이.

    Returns:
        track_indices: tracks 인덱스들의 집합.
    """
    track_indices: Set[int] = set()
    for rp in getattr(scenario, "tracks_to_predict", []):
        idx = int(getattr(rp, "track_index"))
        if 0 <= idx < num_tracks:
            track_indices.add(idx)
    return track_indices


def _collect_track_indices_of_interest(scenario: Any,
                                       num_tracks: int) -> Set[int]:
    """objects_of_interest에서 'tracks 인덱스'들을 안전하게 모읍니다.

    Waymo Motion 문서에 따르면 objects_of_interest는 tracks 필드에 대한 인덱스입니다. :contentReference[oaicite:1]{index=1}

    Args:
        scenario: WOMD scenario 객체.
        num_tracks: scenario.tracks의 길이.

    Returns:
        interest_indices: tracks 인덱스들의 집합.
    """
    interest_indices: Set[int] = set()
    for x in getattr(scenario, "objects_of_interest", []):
        idx = int(x)
        if 0 <= idx < num_tracks:
            interest_indices.add(idx)
    return interest_indices


def decode_tracks_and_roles_from_scenario(
        scenario: Any) -> Dict[str, np.ndarray]:
    """Scenario에서 트랙 정보와 role(interest/predict)을 numpy로 만듭니다.

    - tracks_to_predict: tracks 안의 인덱스 :contentReference[oaicite:2]{index=2}
    - objects_of_interest: tracks 안의 인덱스 :contentReference[oaicite:3]{index=3}

    Returns:
        object_id: (N,) int64
        object_type: (N,) int32
        states: (N,S,9) float32
        valid: (N,S) bool
        role_interest: (N,) bool
        role_predict: (N,) bool
        ego_index: (1,) int64
    """
    tracks = scenario.tracks
    num_tracks = int(len(tracks))
    num_steps = get_num_steps_from_scenario(scenario)

    object_id = np.zeros((num_tracks,), dtype=np.int64)  # (N,)
    object_type = np.zeros((num_tracks,), dtype=np.int32)  # (N,)
    states = np.zeros((num_tracks, num_steps, 9), dtype=np.float32)  # (N,S,9)
    valid = np.zeros((num_tracks, num_steps), dtype=bool)  # (N,S)

    predict_track_indices = _collect_track_indices_to_predict(
        scenario, num_tracks)
    interest_track_indices = _collect_track_indices_of_interest(
        scenario, num_tracks)

    role_interest = np.zeros((num_tracks,), dtype=bool)  # (N,)
    role_predict = np.zeros((num_tracks,), dtype=bool)  # (N,)

    for i, tr in enumerate(tracks):
        object_id[i] = int(tr.id)
        object_type[i] = int(tr.object_type) - 1

        role_predict[i] = (i in predict_track_indices)
        role_interest[i] = (i in interest_track_indices)

        states_i, valid_i = _track_states_to_numpy(tr, num_steps)
        states[i] = states_i
        valid[i] = valid_i

    return {
        "object_id": object_id,
        "object_type": object_type,
        "states": states,
        "valid": valid,
        "role_interest": role_interest,
        "role_predict": role_predict,
        "ego_index": np.array([int(scenario.sdc_track_index)], dtype=np.int64),
    }


def require_valid_ego_at_current(valid_all: np.ndarray, ego_idx: int,
                                 current_t: int) -> None:
    """현재 시점에서 ego가 유효한지 확인합니다.

    ego가 현재 프레임에서 유효하지 않으면, 좌표계 기준(ego 위치/방향)이 잘못 잡혀서
    이후에 만들어지는 모든 값이 “조용히” 틀어질 수 있습니다.
    그래서 이런 경우는 바로 에러로 중단시키는 게 안전합니다.

    Args:
        valid_all: shape (N,S), bool. 각 트랙의 각 시점 유효 여부
        ego_idx: ego 트랙 인덱스
        current_t: 현재 시점 인덱스

    Raises:
        ValueError: ego가 현재 시점에서 유효하지 않거나, 인덱스가 범위를 벗어날 때
    """
    # valid_all: np.ndarray, shape (N,S)
    if valid_all.ndim != 2:
        raise ValueError(
            f"valid_all must be 2D (N,S). got shape={valid_all.shape}")

    n, s = int(valid_all.shape[0]), int(valid_all.shape[1])
    if not (0 <= ego_idx < n) or not (0 <= current_t < s):
        raise ValueError(
            f"Index out of range: ego_idx={ego_idx}/{n}, t={current_t}/{s}")

    if not bool(valid_all[ego_idx, current_t]):
        raise ValueError(
            f"Ego invalid at current frame: ego_idx={ego_idx}, t={current_t}")

def build_origin_world_pose(
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
) -> np.ndarray:
    """ego의 현재 위치/방향을 "전역 좌표 기준"으로 저장하기 좋은 4차원 벡터로 만듭니다.

    지금 코드에서는 모든 출력이 "ego의 현재 위치를 원점"으로 하는 좌표 기준(ego local)으로 저장됩니다.
    그런데 학습/후처리에서 가끔
    "이 local 좌표의 원점이 전역(지도) 기준으로 어디였는지"
    "local 좌표의 x축이 전역 기준으로 어느 방향이었는지"
    를 다시 알아야 할 때가 있습니다.

    그래서 ego의 현재 pose를 아래 4개 값으로 저장합니다.

      1) x: 전역 좌표 기준 ego의 x 위치
      2) y: 전역 좌표 기준 ego의 y 위치
      3) cos_yaw: 전역 좌표 기준 ego yaw의 cos 값
      4) sin_yaw: 전역 좌표 기준 ego yaw의 sin 값

    yaw를 각도(rad) 그대로 저장해도 되지만,
    각도는 -pi/pi 경계에서 값이 튀는 문제가 생기기 쉬워서
    cos/sin 형태로 저장하면 더 안정적으로 복원할 수 있습니다.
    (필요하면 나중에 atan2(sin_yaw, cos_yaw)로 yaw를 다시 만들 수 있습니다)

    Args:
        ego_xy_global: (2,) float32/float64.
            전역 좌표 기준 ego의 (x, y) 위치.
        ego_yaw_global: float.
            전역 좌표 기준 ego의 yaw(라디안).

    Returns:
        origin_world_pose: (4,) float32.
            [x, y, cos(yaw), sin(yaw)] 형태의 벡터.
    """
    ego_xy = np.asarray(ego_xy_global, dtype=np.float32).reshape(-1)  # shape: (2,)
    if int(ego_xy.shape[0]) != 2:
        raise ValueError(
            "ego_xy_global은 (2,) 형태여야 합니다. "
            f"got shape={tuple(ego_xy_global.shape)}"
        )

    yaw = float(ego_yaw_global)
    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))

    origin_world_pose = np.zeros((4,), dtype=np.float32)  # shape: (4,)
    origin_world_pose[0] = ego_xy[0]
    origin_world_pose[1] = ego_xy[1]
    origin_world_pose[2] = np.float32(cos_yaw)
    origin_world_pose[3] = np.float32(sin_yaw)
    return origin_world_pose


def compute_ego_pose_at_current(
    scenario: Any,
    track_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, float, float]:
    """현재 시점의 ego 위치/방향을 구합니다.

    반환되는 ego 위치/방향은 이후 모든 좌표 변환의 기준이 됩니다.
    그래서 ego가 현재 시점에서 유효하지 않으면 즉시 에러로 중단합니다.

    Args:
        scenario: WOMD 시나리오 객체
        track_dict: decode_tracks_and_roles_from_scenario의 결과

    Returns:
        ego_xy_global: shape (2,), float32
        ego_yaw_global: float
        ego_z_global: float
    """
    ego_idx = int(track_dict["ego_index"][0])
    current_t = int(scenario.current_time_index)

    valid_all = track_dict["valid"]  # shape (N,S)
    require_valid_ego_at_current(valid_all, ego_idx, current_t)

    ego_state = track_dict["states"][ego_idx, current_t]  # shape (9,)
    ego_xy = ego_state[0:2].astype(np.float32)  # shape (2,)
    ego_yaw = float(ego_state[6])
    ego_z = float(ego_state[2])
    return ego_xy, ego_yaw, ego_z


def is_valid_polygon_xy(points_xy: np.ndarray, min_points: int = 3) -> bool:
    """polygon 점 목록이 “의미 있게” 존재하는지 검사합니다.

    점이 너무 적으면(예: 0개, 1개, 2개) polygon이라고 보기 어렵고,
    이런 값이 캐시에 들어가면 이후 단계에서 전부 0으로 된 결과가 생겨서
    데이터가 더러워질 수 있습니다.

    Args:
        points_xy: shape (M,2), float32. polygon의 점 목록
        min_points: 최소 점 개수(기본 3)

    Returns:
        True면 저장할 가치가 있는 polygon, False면 버립니다.
    """
    # points_xy: np.ndarray, shape (M,2)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return False
    return int(points_xy.shape[0]) >= int(min_points)


# =========================
# 교통 신호(차선 신호) 파싱
# =========================
def lane_signal_state_to_one_hot(state_value: int) -> np.ndarray:
    """WOMD의 lane signal state 값을 4차원 one-hot으로 바꿉니다.

    출력 one-hot 순서:
        [GO, CAUTION, STOP, UNKNOWN]

    Args:
        state_value: TrafficSignalLaneState.State (정수)

    Returns:
        one_hot: (4,) float32
    """
    # 참고: state_value는 map_pb2.TrafficSignalLaneState.State 값이 들어옵니다.
    # 여기서는 '큰 분류'만 사용합니다.
    go_states = {3, 6}  # ARROW_GO(3), GO(6)
    caution_states = {2, 5,
                      8}  # ARROW_CAUTION(2), CAUTION(5), FLASHING_CAUTION(8)
    stop_states = {1, 4, 7}  # ARROW_STOP(1), STOP(4), FLASHING_STOP(7)
    unknown_states = {0}  # UNKNOWN(0)

    one_hot = np.zeros((4,), dtype=np.float32)
    if state_value in go_states:
        one_hot[0] = 1.0
    elif state_value in caution_states:
        one_hot[1] = 1.0
    elif state_value in stop_states:
        one_hot[2] = 1.0
    elif state_value in unknown_states:
        one_hot[3] = 1.0
    else:
        # 혹시 모르는 값은 UNKNOWN 처리
        one_hot[3] = 1.0
    return one_hot


def extract_lane_signals_at_current(
    scenario: scenario_pb2.Scenario,) -> Dict[int, np.ndarray]:
    """현재 시점의 차선 신호를 lane_id -> one-hot(4)로 뽑습니다.

    Args:
        scenario: Scenario proto

    Returns:
        lane_id_to_light: lane_id(int) -> one_hot(4,) float32
    """
    lane_id_to_light: Dict[int, np.ndarray] = {}
    current_t = int(scenario.current_time_index)

    if len(scenario.dynamic_map_states) <= current_t:
        return lane_id_to_light

    dyn = scenario.dynamic_map_states[current_t]
    for lane_state in dyn.lane_states:
        lane_id = int(lane_state.lane)
        state_value = int(lane_state.state)
        lane_id_to_light[lane_id] = lane_signal_state_to_one_hot(state_value)

    return lane_id_to_light

def _min_dist2_point_to_polyline_xy(
    points_xy: np.ndarray,  # shape: (K, 2)
    query_xy: np.ndarray,  # shape: (2,)
    closed: bool,
) -> float:
    """한 점(query)이 '점들이 줄로 이어진 형태'까지의 최소 거리^2를 계산합니다.

    - points_xy가 (K,2)로 주어졌을 때, 인접한 점들을 이은 선분들(segments)을 만들고,
      query_xy에서 그 선분들까지의 거리를 계산해 가장 작은 값을 반환합니다.
    - closed=True이면 마지막 점과 첫 점도 이어서(닫힌 모양) 선분을 하나 더 만든 것으로 봅니다.
    - sqrt를 하지 않고 거리의 제곱(dist^2)으로 반환해서 빠르게 비교할 수 있게 합니다.

    Args:
        points_xy: (K,2) float32/float64. 선을 구성하는 점들.
        query_xy: (2,) float32/float64. 기준 점(여기서는 ego 위치).
        closed: 닫힌 모양 여부.

    Returns:
        min_dist2: float. 최소 거리의 제곱.
            points_xy가 비어 있으면 inf.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return float("inf")

    k = int(points_xy.shape[0])
    if k == 0:
        return float("inf")

    q = query_xy.astype(np.float32).reshape(2,)
    p = points_xy.astype(np.float32)

    if k == 1:
        d = p[0] - q
        return float(d[0] * d[0] + d[1] * d[1])

    if bool(closed):
        p0 = p
        p1 = np.roll(p, shift=-1, axis=0)
    else:
        p0 = p[:-1]
        p1 = p[1:]

    seg = (p1 - p0).astype(np.float32)  # (M,2)
    v = (q[None, :] - p0).astype(np.float32)  # (M,2)

    seg_len2 = (seg[:, 0] * seg[:, 0] + seg[:, 1] * seg[:, 1]).astype(np.float32)  # (M,)
    seg_len2_safe = np.maximum(seg_len2, float(EPS)).astype(np.float32)

    t = ((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2_safe).astype(np.float32)  # (M,)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    closest = (p0 + seg * t[:, None]).astype(np.float32)  # (M,2)
    d = (q[None, :] - closest).astype(np.float32)  # (M,2)
    dist2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).astype(np.float32)  # (M,)

    return float(np.min(dist2))


def _is_point_inside_polygon_xy(
    point_xy: np.ndarray,  # shape: (2,)
    polygon_xy: np.ndarray,  # shape: (M, 2)
) -> bool:
    """점(point)이 다각형(polygon) '안쪽'에 있는지 빠르게 검사합니다.

    - crosswalk/driveway/speed_bump 같은 것은 '영역'으로 보는 것이 자연스럽습니다.
      그래서 ego가 polygon 내부에 있으면, 거리는 0이라고 보고 무조건 포함시키는 게 맞습니다.
    - 이 함수는 다각형의 변을 기준으로, 가로선이 몇 번 교차하는지 세는 방식으로 판정합니다.

    Args:
        point_xy: (2,) float32/float64. 검사할 점(여기서는 ego 위치).
        polygon_xy: (M,2) float32/float64. 다각형 꼭짓점들.

    Returns:
        inside: bool. 내부면 True.
    """
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        return False

    m = int(polygon_xy.shape[0])
    if m < 3:
        return False

    p = polygon_xy.astype(np.float32)
    px = float(point_xy[0])
    py = float(point_xy[1])

    x0 = p[:, 0]
    y0 = p[:, 1]
    x1 = np.roll(x0, shift=-1)
    y1 = np.roll(y0, shift=-1)

    cond1 = (y0 > py) != (y1 > py)
    # y1-y0가 0에 가까운 경우 분모가 0이 될 수 있어 EPS를 더합니다.
    x_intersect = (x1 - x0) * (py - y0) / (y1 - y0 + float(EPS)) + x0
    cond2 = px < x_intersect

    crossings = int(np.count_nonzero(cond1 & cond2))
    return (crossings % 2) == 1


def _keep_polyline_if_within_radius(
    polyline_xy_global: np.ndarray,  # shape: (K,2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    radius_m: float,
) -> bool:
    """polyline(점들이 줄로 이어진 것)이 ego 반경 안에 '조금이라도' 들어오면 True를 반환합니다.

    판정 기준:
      - ego 위치에서 polyline 선분들까지의 최소 거리가 radius_m 이하이면 포함합니다.
      - 즉, 꼭짓점이 반경 밖에 있어도 선분이 원을 스치면 포함됩니다.

    Args:
        polyline_xy_global: (K,2) 전역 좌표 점들.
        ego_xy_global: (2,) ego 전역 위치.
        radius_m: 반경(m).

    Returns:
        keep: bool
    """
    r = float(radius_m)
    if not np.isfinite(r):
        return True
    if r < 0.0:
        return True

    r2 = r * r
    d2 = _min_dist2_point_to_polyline_xy(
        points_xy=polyline_xy_global,
        query_xy=ego_xy_global,
        closed=False,
    )
    return bool(d2 <= r2)


def _keep_polygon_area_if_within_radius(
    polygon_xy_global: np.ndarray,  # shape: (M,2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    radius_m: float,
) -> bool:
    """polygon(영역)이 ego 반경 안에 '조금이라도' 들어오면 True를 반환합니다.

    판정 기준:
      1) ego가 polygon 내부에 있으면 거리=0으로 보고 무조건 포함
      2) 아니면 ego에서 polygon 테두리(선분)까지의 최소 거리가 radius_m 이하이면 포함

    Args:
        polygon_xy_global: (M,2) 전역 좌표 꼭짓점들.
        ego_xy_global: (2,) ego 전역 위치.
        radius_m: 반경(m).

    Returns:
        keep: bool
    """
    r = float(radius_m)
    if not np.isfinite(r):
        return True
    if r < 0.0:
        return True

    if _is_point_inside_polygon_xy(ego_xy_global, polygon_xy_global):
        return True

    r2 = r * r
    d2 = _min_dist2_point_to_polyline_xy(
        points_xy=polygon_xy_global,
        query_xy=ego_xy_global,
        closed=True,
    )
    return bool(d2 <= r2)

def filter_parsed_map_by_radius(
    parsed_map: ParsedMap,
    ego_xy_global: np.ndarray,  # shape: (2,)
    filter_radius_m: float,
) -> ParsedMap:
    """ParsedMap에서 ego 반경 안에 들어오는 것만 남겨서 새 ParsedMap을 만듭니다.

    이 함수는 캐싱 대상 중 아래 항목들만 필터링합니다.
      - lanes (centerline polyline 기준)
      - road_edge (polyline 기준)
      - stop_sign (점 기준)
      - crosswalk (polygon 영역 기준)
      - speed_bump (polygon 영역 기준)
      - driveway_polygon (polygon 영역 기준)

    Args:
        parsed_map: parse_map_from_scenario() 결과.
        ego_xy_global: (2,) ego 전역 위치.
        filter_radius_m: 반경(m). inf면 필터 없이 그대로 반환합니다.

    Returns:
        filtered_map: 필터가 적용된 ParsedMap
    """
    r = float(filter_radius_m)
    if not np.isfinite(r):
        return parsed_map
    if r < 0.0:
        return parsed_map

    # 1) stop sign (점)
    if len(parsed_map.stop_sign_xy_global) == 0:
        stop_sign_xy = []
    else:
        pts = np.stack(parsed_map.stop_sign_xy_global, axis=0).astype(np.float32)  # (N,2)
        d = pts - ego_xy_global.astype(np.float32)[None, :]  # (N,2)
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).astype(np.float32)  # (N,)
        keep = d2 <= (r * r)
        idxs = np.flatnonzero(keep).astype(np.int64)
        stop_sign_xy = [parsed_map.stop_sign_xy_global[int(i)] for i in idxs.tolist()]

    # 2) polygons (영역)
    crosswalk_polys: List[np.ndarray] = []
    for poly in parsed_map.crosswalk_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            crosswalk_polys.append(poly)

    speed_bump_polys: List[np.ndarray] = []
    for poly in parsed_map.speed_bump_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            speed_bump_polys.append(poly)

    driveway_polys: List[np.ndarray] = []
    for poly in parsed_map.driveway_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            driveway_polys.append(poly)

    # 3) lanes (centerline polyline 기준)
    lanes_filtered: List[LaneInfo] = []
    for lane in parsed_map.lanes:
        if _keep_polyline_if_within_radius(lane.centerline_xy_global, ego_xy_global, r):
            lanes_filtered.append(lane)

    # 4) road_edge ids (polyline 기준)
    road_edge_ids_filtered: List[int] = []
    for edge_id in parsed_map.road_edge_ids:
        poly = parsed_map.boundary_polylines_xy_global.get(int(edge_id), None)
        if poly is None:
            continue
        if _keep_polyline_if_within_radius(poly, ego_xy_global, r):
            road_edge_ids_filtered.append(int(edge_id))

    return ParsedMap(
        stop_sign_xy_global=stop_sign_xy,
        crosswalk_polygons_xy_global=crosswalk_polys,
        speed_bump_polygons_xy_global=speed_bump_polys,
        driveway_polygons_xy_global=driveway_polys,
        lanes=lanes_filtered,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        road_line_type_by_id=parsed_map.road_line_type_by_id,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
        road_edge_ids=road_edge_ids_filtered,
    )




# =========================
# 맵 파싱 + 샘플링
# =========================
def polyline_length(points_xy: np.ndarray) -> float:
    """폴리라인(점들의 줄)의 전체 길이를 계산합니다.

    Args:
        points_xy: (N,2) 점들

    Returns:
        길이(float)
    """
    if points_xy.shape[0] < 2:
        return 0.0
    diffs = points_xy[1:] - points_xy[:-1]  # shape (N-1,2)
    seg_lens = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
    return float(seg_lens.sum())


def resample_polyline_equal_distance(
    points_xy: np.ndarray,  # shape: (N,2)
    num_samples: int,
    closed: bool,
) -> np.ndarray:
    """점들의 줄(또는 닫힌 다각형)을 '같은 간격'으로 num_samples개 점으로 만듭니다.

    - points_xy가 1개뿐이면 그 점을 반복합니다.
    - 길이가 0에 가깝다면 첫 점을 반복합니다.
    - closed=True이면 마지막에서 첫 점으로 이어진다고 보고 둘레를 따라 샘플링합니다.

    Args:
        points_xy: (N,2) 입력 점들
        num_samples: 뽑을 점 개수
        closed: 닫힌 모양인지 여부

    Returns:
        sampled: (num_samples,2) float32
    """
    if points_xy.shape[0] == 0:
        return np.zeros((num_samples, 2), dtype=np.float32)

    if points_xy.shape[0] == 1:
        return np.repeat(points_xy.astype(np.float32),
                         repeats=num_samples,
                         axis=0)

    if closed:
        # 닫힌 모양이면 끝에 첫 점을 붙여서 둘레를 만듭니다.
        if not np.allclose(points_xy[0], points_xy[-1]):
            points_xy = np.vstack([points_xy, points_xy[0]])  # shape (N+1,2)

    diffs = points_xy[1:] - points_xy[:-1]  # shape (M,2)
    seg_lens = np.linalg.norm(diffs, axis=1)  # shape (M,)
    total_len = float(seg_lens.sum())

    if total_len < EPS:
        return np.repeat(points_xy[:1].astype(np.float32),
                         repeats=num_samples,
                         axis=0)

    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])  # shape (M+1,)

    if closed:
        # 닫힌 모양은 시작점 중복을 피하려고 endpoint=False
        target = np.linspace(0.0,
                             total_len,
                             num_samples,
                             endpoint=False,
                             dtype=np.float32)
    else:
        target = np.linspace(0.0,
                             total_len,
                             num_samples,
                             endpoint=True,
                             dtype=np.float32)

    # 각 target이 어느 구간에 속하는지 찾기
    seg_idx = np.searchsorted(cum_len, target, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lens) - 1)  # shape (num_samples,)

    seg_start = points_xy[seg_idx]  # shape (num_samples,2)
    seg_vec = diffs[seg_idx]  # shape (num_samples,2)
    seg_len = seg_lens[seg_idx]  # shape (num_samples,)

    alpha = (target - cum_len[seg_idx]) / np.maximum(
        seg_len, EPS)  # shape (num_samples,)
    sampled = seg_start + seg_vec * alpha[:, None]  # shape (num_samples,2)
    return sampled.astype(np.float32)


from typing import Any, Iterable, List


def convert_proto_boundary_segments(
    boundary_segments_proto: Iterable[Any],) -> List[BoundarySegmentInfo]:
    """Waymo lane boundary segment(proto) 목록을 BoundarySegmentInfo 리스트로 변환합니다.

    Waymo 버전/환경에 따라 필드명이 lane_end_index 또는 lane_end_idx처럼
    약간 다를 수 있어서, getattr로 안전하게 읽습니다.

    Args:
        boundary_segments_proto: 반복 가능한 boundary segment proto 목록.
            각 원소는 보통 다음 필드를 가집니다.
            - lane_start_index (또는 lane_start_idx)
            - lane_end_index (또는 lane_end_idx)
            - boundary_feature_id

    Returns:
        segments: BoundarySegmentInfo 리스트.
            boundary_feature_id가 0 이하이거나, end <= start 같은 값도
            일단 담아두고, 실제 사용 단계에서 자동으로 무시되도록 합니다.
    """
    segments: List[BoundarySegmentInfo] = []
    for seg in boundary_segments_proto:
        lane_start_index = int(
            getattr(seg, "lane_start_index", getattr(seg, "lane_start_idx", 0)))
        lane_end_index = int(
            getattr(seg, "lane_end_index", getattr(seg, "lane_end_idx", 0)))
        boundary_feature_id = int(getattr(seg, "boundary_feature_id", 0))

        segments.append(
            BoundarySegmentInfo(
                lane_start_index=lane_start_index,
                lane_end_index=lane_end_index,
                boundary_feature_id=boundary_feature_id,
            ))
    return segments

def parse_map_from_scenario(scenario: scenario_pb2.Scenario) -> ParsedMap:
    """Scenario의 map_features를 요구사항에 맞게 필요한 것만 모읍니다.

    Returns:
        ParsedMap
    """
    stop_sign_xy_global: List[np.ndarray] = []
    crosswalk_polygons_xy_global: List[np.ndarray] = []
    speed_bump_polygons_xy_global: List[np.ndarray] = []

    # ✅ driveway는 polygon만 저장
    driveway_polygons_xy_global: List[np.ndarray] = []

    lanes: List[LaneInfo] = []

    boundary_polylines_xy_global: Dict[int, np.ndarray] = {}
    boundary_id_to_kind: Dict[int, str] = {}

    road_line_type_by_id: Dict[int, int] = {}
    road_edge_type_by_id: Dict[int, int] = {}
    road_edge_ids: List[int] = []

    for mf in scenario.map_features:
        feature_type = mf.WhichOneof("feature_data")

        if feature_type == "stop_sign":
            pos = mf.stop_sign.position
            stop_sign_xy_global.append(np.array([pos.x, pos.y], dtype=np.float32))

        elif feature_type == "crosswalk":
            polygon_xy = _proto_points_to_xy_array(mf.crosswalk.polygon)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=3):
                crosswalk_polygons_xy_global.append(polygon_xy)

        elif feature_type == "speed_bump":
            polygon_xy = _proto_points_to_xy_array(mf.speed_bump.polygon)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=3):
                speed_bump_polygons_xy_global.append(polygon_xy)

        elif feature_type == "driveway":
            polygon_xy = _proto_points_to_xy_array(mf.driveway.polygon)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=3):
                driveway_polygons_xy_global.append(polygon_xy)

        elif feature_type == "road_line":
            poly_xy = _proto_points_to_xy_array(mf.road_line.polyline)  # (K,2)
            if poly_xy.shape[0] >= 1:
                fid = int(mf.id)
                boundary_polylines_xy_global[fid] = poly_xy
                boundary_id_to_kind[fid] = "road_line"
                road_line_type_by_id[fid] = int(getattr(mf.road_line, "type", 0))

        elif feature_type == "road_edge":
            poly_xy = _proto_points_to_xy_array(mf.road_edge.polyline)  # (K,2)
            if poly_xy.shape[0] >= 1:
                fid = int(mf.id)
                boundary_polylines_xy_global[fid] = poly_xy
                boundary_id_to_kind[fid] = "road_edge"
                road_edge_ids.append(fid)
                road_edge_type_by_id[fid] = int(getattr(mf.road_edge, "type", 0))

        elif feature_type == "lane":
            centerline_xy = _proto_points_to_xy_array(mf.lane.polyline)  # (P,2)
            if centerline_xy.shape[0] == 0:
                continue

            left_segments = convert_proto_boundary_segments(mf.lane.left_boundaries)
            right_segments = convert_proto_boundary_segments(mf.lane.right_boundaries)

            speed_limit_mph = float(getattr(mf.lane, "speed_limit_mph", 0.0))
            lane_type_value = int(getattr(mf.lane, "type", 0))

            lanes.append(
                LaneInfo(
                    lane_id=int(mf.id),
                    centerline_xy_global=centerline_xy,
                    left_boundary_segments=left_segments,
                    right_boundary_segments=right_segments,
                    speed_limit_mph=speed_limit_mph,
                    lane_type=lane_type_value,
                )
            )

    return ParsedMap(
        stop_sign_xy_global=stop_sign_xy_global,
        crosswalk_polygons_xy_global=crosswalk_polygons_xy_global,
        speed_bump_polygons_xy_global=speed_bump_polygons_xy_global,
        driveway_polygons_xy_global=driveway_polygons_xy_global,
        lanes=lanes,
        boundary_polylines_xy_global=boundary_polylines_xy_global,
        boundary_id_to_kind=boundary_id_to_kind,
        road_line_type_by_id=road_line_type_by_id,
        road_edge_type_by_id=road_edge_type_by_id,
        road_edge_ids=road_edge_ids,
    )


def atomic_pickle_dump(obj: Any, out_path: Path) -> None:
    """pickle 파일을 “깨지지 않게” 저장합니다.

    저장 중에 프로세스가 죽으면, 파일이 중간까지만 써진 상태로 남을 수 있습니다.
    그러면 다음 실행에서 그 파일을 읽다가 오류가 나거나, 더 나쁘게는 조용히 잘못 읽힐 수 있습니다.

    이 함수는
    1) 임시 파일에 먼저 끝까지 저장하고
    2) 디스크에 실제로 기록되도록 강제로 한 번 밀어넣은 다음
    3) 마지막에 이름만 바꿔서(out_path로 교체) “완성된 파일만 보이게” 합니다.

    Args:
        obj: 저장할 파이썬 객체
        out_path: 최종 pkl 경로
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists() and (not out_path.exists()):
            with contextlib.suppress(Exception):
                tmp_path.unlink()


from typing import Any, Dict, Iterable, Tuple
from pathlib import Path
import contextlib
import os
import numpy as np

EXCLUDE_KEYS_FOR_NPZ: Tuple[str, ...] = ("scenario_id", "neighbor_track_token")
from typing import Tuple
import numpy as np


def gather_z_global_at_time(
    z_global_all: np.ndarray,        # shape: (N, S)
    track_indices: np.ndarray,       # shape: (A,)
    time_index: int,
) -> np.ndarray:
    """여러 트랙의 '절대 z(전역 좌표계 기준)'를 특정 시점에서 한 번에 뽑아 (A,)로 반환합니다.

    이 함수는 target_z를 "절대 좌표계 기준"으로 만들 때 쓰기 위한 유틸입니다.

    입력으로 들어오는 z_global_all은 WOMD track states의 center_z를 모아둔 값으로,
    ego 기준으로 빼기(상대화)를 하지 않은 "원본 전역 z"라고 가정합니다.

    안전을 위해 아래를 같이 처리합니다.
    - track_indices 안에 범위를 벗어난 값이 섞여 있으면 그 위치는 0.0으로 채웁니다.
    - time_index가 범위를 벗어나면 가능한 범위로 clamp 합니다.

    Args:
        z_global_all: (N,S) float32/float64.
            모든 트랙의 모든 시점에 대한 전역 z 값 배열.
        track_indices: (A,) int64/int32.
            z를 뽑고 싶은 트랙 인덱스들.
        time_index: int.
            z를 뽑을 시점 인덱스.

    Returns:
        z_global_at_t: (A,) float32.
            track_indices 순서대로 뽑은 전역 z 값.
            잘못된 트랙 인덱스는 0.0으로 반환됩니다.
    """
    # z_global_all: (N,S)
    if z_global_all.ndim != 2:
        raise ValueError(f"z_global_all must be (N,S). got shape={z_global_all.shape}")

    num_tracks = int(z_global_all.shape[0])  # N
    num_steps = int(z_global_all.shape[1])   # S

    idx = np.asarray(track_indices, dtype=np.int64).reshape(-1)  # shape: (A,)
    a = int(idx.shape[0])

    if a == 0:
        return np.zeros((0,), dtype=np.float32)

    t = int(time_index)
    if num_steps <= 0:
        return np.zeros((a,), dtype=np.float32)
    t = max(0, min(t, num_steps - 1))

    out = np.zeros((a,), dtype=np.float32)  # shape: (A,)

    valid = (idx >= 0) & (idx < num_tracks)  # shape: (A,)
    if bool(np.any(valid)):
        out[valid] = z_global_all[idx[valid], t].astype(np.float32)  # shape: (A_valid,)

    return out

def build_target_id_and_z_from_ego_and_neighbors(
    ego_object_id: int,
    ego_z_value: float,
    neighbor_object_id: np.ndarray,
    neighbor_z_value: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """ego 1개와 neighbor 여러 개의 id/z를 같은 순서로 묶어 target_id/target_z를 만듭니다.

    이 스크립트는 원래 ego는 ego_*로 따로 저장되고,
    neighbor는 neighbor_*로 따로 저장됩니다.

    그런데 어떤 후처리/학습 코드에서는
    "ego + neighbor를 한 줄로 쭉" 같은 순서로 쓰는 게 편할 때가 있습니다.
    예를 들면 아래처럼요.

      - target_id: [ego_id, neighbor_id_0, neighbor_id_1, ...]
      - target_z : [ego_z , neighbor_z_0 , neighbor_z_1 , ...]

    이 함수는 딱 그 형태로 두 배열을 만듭니다.

    주의
    ----
    현재 data_process_womd.py의 neighbor_z는
    "ego보다 얼마나 위/아래인지" (즉 ego 기준 상대값)으로 만들어집니다.
    그래서 ego의 상대 z는 보통 0.0 입니다.

    Args:
        ego_object_id: ego의 고유 id(정수).
        ego_z_value: ego의 z 값(실수).
            - 현재 코드 기준으로는 보통 0.0(ego 기준 상대값)입니다.
        neighbor_object_id: (A,) int64. neighbor들의 id 배열.
        neighbor_z_value: (A,) float32. neighbor들의 z 배열.

    Returns:
        target_id: (A+1,) int64. 첫 칸은 ego id, 그 뒤는 neighbor id.
        target_z: (A+1,) float32. 첫 칸은 ego z, 그 뒤는 neighbor z.
    """
    neighbor_object_id_arr = np.asarray(neighbor_object_id, dtype=np.int64).reshape(-1)  # shape: (A,)
    neighbor_z_value_arr = np.asarray(neighbor_z_value, dtype=np.float32).reshape(-1)   # shape: (A,)

    if int(neighbor_object_id_arr.shape[0]) != int(neighbor_z_value_arr.shape[0]):
        raise ValueError(
            "neighbor_object_id와 neighbor_z_value의 길이가 다릅니다. "
            f"id_len={int(neighbor_object_id_arr.shape[0])}, "
            f"z_len={int(neighbor_z_value_arr.shape[0])}"
        )

    neighbor_count = int(neighbor_object_id_arr.shape[0])

    target_id = np.empty((neighbor_count + 1,), dtype=np.int64)    # shape: (A+1,)
    target_z = np.empty((neighbor_count + 1,), dtype=np.float32)   # shape: (A+1,)

    target_id[0] = int(ego_object_id)
    target_z[0] = float(ego_z_value)

    if neighbor_count > 0:
        target_id[1:] = neighbor_object_id_arr
        target_z[1:] = neighbor_z_value_arr

    return target_id, target_z


def build_npz_payload_from_cache_dict(
    cache_dict: Dict[str, Any],
    excluded_keys: Iterable[str] = EXCLUDE_KEYS_FOR_NPZ,
) -> Dict[str, np.ndarray]:
    """캐시 dict에서 npz로 저장할 '배열들'만 골라 dict로 만듭니다.

    npz 파일은 "이름(key) + numpy 배열(value)" 여러 개를 한 파일에 담는 방식입니다.
    그래서 문자열(예: scenario_id)이나 문자열 리스트(예: neighbor_track_token)는
    그대로 저장하기가 애매하고, 이번 요구사항에서도 저장 대상에서 제외해야 합니다.

    이 함수는 다음 순서로 처리합니다.
    1) excluded_keys에 들어있는 키는 통째로 제외합니다.
       - 예: "scenario_id", "neighbor_track_token"
    2) 나머지 값은 numpy 배열인지 확인합니다.
       - 이미 np.ndarray면 그대로 사용합니다.
       - np.ndarray가 아니면 np.asarray로 바꿀 수 있으면 바꿉니다.
    3) 그래도 "저장하기 애매한 값"(dtype=object 같은)이면 에러로 막습니다.
       - 저장 후 읽을 때 예기치 않은 문제가 생기는 것을 미리 막기 위함입니다.

    Args:
        cache_dict: 시나리오 1개에서 만든 캐시 dict.
            - 대부분 value는 np.ndarray이고,
              예를 들면 (21,11), (80,3), (A,21,11), (L,10,12) 같은 shape을 가집니다.
        excluded_keys: npz 파일에 넣지 않을 키 목록.

    Returns:
        npz_payload: np.savez_compressed(**npz_payload)로 바로 저장 가능한 dict.
            - key: str
            - value: np.ndarray (shape은 cache_dict의 원본과 동일)
    """
    excluded = set(str(k) for k in excluded_keys)
    npz_payload: Dict[str, np.ndarray]
    npz_payload = {}

    for key, value in cache_dict.items():
        if str(key) in excluded:
            continue

        if isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.asarray(value)

        # npz에는 "진짜 숫자/불리언 배열" 형태로 담기는 게 안전합니다.
        if arr.dtype == object:
            raise TypeError(
                f"npz로 저장하기 어려운 값이 있습니다. key={key}, dtype=object"
            )

        npz_payload[str(key)] = arr

    return npz_payload


def atomic_npz_save(
    arrays: Dict[str, np.ndarray],
    out_path: Path,
    compress: bool = True,
) -> None:
    """npz 파일을 저장 도중 끊겨도 '깨진 파일'이 남지 않게 저장합니다.

    저장 중에 프로세스가 갑자기 종료되면 파일이 중간까지만 써질 수 있습니다.
    그러면 다음 실행에서 읽다가 실패하거나, 더 나쁘게는 이상한 값이 섞일 수 있습니다.

    이 함수는 아래 방식으로 안전하게 저장합니다.
    1) 같은 폴더에 임시 파일(.tmp)에 먼저 끝까지 저장
    2) 디스크에 실제로 기록되도록 강제로 한 번 반영(fsync)
    3) 마지막에 파일 이름만 바꿔서(out_path로 교체) "완성본만 보이게" 처리

    Args:
        arrays: npz에 담을 배열 dict.
            - 각 value는 np.ndarray (shape은 원본과 동일)
        out_path: 최종 .npz 경로
        compress: True면 용량을 줄여 저장합니다(대신 저장 시간이 약간 늘 수 있음).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, out_path)

    finally:
        if tmp_path.exists() and (not out_path.exists()):
            with contextlib.suppress(Exception):
                tmp_path.unlink()


def save_cache_dict_to_npz(
    cache_dict: Dict[str, Any],
    out_npz_path: Path,
    excluded_keys: Iterable[str] = EXCLUDE_KEYS_FOR_NPZ,
    compress: bool = True,
) -> None:
    """cache_dict를 npz로 저장하되, 특정 키는 제외하고 저장합니다.

    Args:
        cache_dict: build_cache_dict_for_scenario()의 결과 dict.
        out_npz_path: 저장할 .npz 파일 경로.
        excluded_keys: npz에 넣지 않을 키 목록.
            - 기본값은 ("scenario_id", "neighbor_track_token")
        compress: True면 용량을 줄여 저장합니다.
    """
    npz_payload = build_npz_payload_from_cache_dict(
        cache_dict=cache_dict,
        excluded_keys=excluded_keys,
    )
    atomic_npz_save(
        arrays=npz_payload,
        out_path=out_npz_path,
        compress=compress,
    )


def build_stop_sign_points(
    stop_sign_xy_global: List[np.ndarray],
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    safety_len: int,
) -> np.ndarray:
    """stop sign 위치를 (n_stop_sign, safety_len, 2)로 만듭니다.

    WOMD에서는 stop sign이 점 1개로 표현되므로,
    그 점을 safety_len번 복사합니다.

    Args:
        stop_sign_xy_global: stop sign 점들의 리스트 (각각 (2,))
        ego_xy_global: ego 전역 위치 (2,)
        ego_yaw_global: ego 전역 yaw
        safety_len: 출력 점 개수(10)

    Returns:
        stop_sign_points: (n_stop_sign, safety_len, 2) float32 (ego 기준)
    """
    n = len(stop_sign_xy_global)
    out = np.zeros((n, safety_len, 2), dtype=np.float32)  # shape (n,10,2)
    for i, xy in enumerate(stop_sign_xy_global):
        # xy_local: (2,)
        xy_local = transform_points_global_to_ego_local(xy[None, :],
                                                        ego_xy_global,
                                                        ego_yaw_global)[0]
        out[i] = np.repeat(xy_local[None, :], repeats=safety_len, axis=0)
    return out



def build_polygon_points(
    polygons_xy_global: List[np.ndarray],  # each (M,2)
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    safety_len: int,
) -> np.ndarray:
    """polygon 리스트를 (n_poly, safety_len, 2)로 바꿉니다.

    요구사항대로 polygon을 둘레를 따라 같은 간격으로 safety_len개로 쪼갭니다.

    Args:
        polygons_xy_global: polygon 꼭짓점 리스트 (각각 (M,2))
        ego_xy_global: ego 전역 위치 (2,)
        ego_yaw_global: ego 전역 yaw
        safety_len: 출력 점 개수(10)

    Returns:
        points: (n_poly, safety_len, 2) float32 (ego 기준)
    """
    n = len(polygons_xy_global)
    out = np.zeros((n, safety_len, 2), dtype=np.float32)  # shape (n,10,2)

    for i, poly_xy in enumerate(polygons_xy_global):
        # poly_xy: (M,2)
        if poly_xy.shape[0] == 0:
            continue

        # 먼저 ego 기준으로 바꾼 뒤, 그 좌표에서 샘플링합니다.
        poly_local = transform_points_global_to_ego_local(
            poly_xy, ego_xy_global, ego_yaw_global)  # (M,2)
        sampled = resample_polyline_equal_distance(poly_local,
                                                   num_samples=safety_len,
                                                   closed=True)  # (10,2)
        out[i] = sampled

    return out


def nearest_points_vectors(
        centers_xy: np.ndarray,  # shape: (L,2)
        boundary_xy: np.ndarray,  # shape: (K,2)
) -> np.ndarray:
    """center 점 각각에 대해 boundary에서 가장 가까운 점을 찾아 (boundary-center) 벡터를 만듭니다.

    Args:
        centers_xy: (L,2) 중심선 점들
        boundary_xy: (K,2) 경계선 점들

    Returns:
        vecs: (L,2) (가까운 boundary점 - center점)
    """
    lane_len = int(centers_xy.shape[0])
    if boundary_xy.shape[0] == 0:
        return np.zeros((lane_len, 2), dtype=np.float32)

    vecs = np.zeros((lane_len, 2), dtype=np.float32)  # shape (L,2)
    for i in range(lane_len):
        c = centers_xy[i]  # (2,)
        diffs = boundary_xy - c[None, :]  # (K,2)
        d2 = np.sum(diffs * diffs, axis=1)  # (K,)
        j = int(np.argmin(d2))
        vecs[i] = boundary_xy[j] - c
    return vecs


def _collect_road_line_boundary_points_local(
    boundary_feature_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
) -> np.ndarray:
    """lane 한쪽(왼쪽/오른쪽)에 해당하는 road_line 점들을 ego 기준으로 모읍니다.

    이 함수는 "road_edge(연석 등)"는 완전히 제외하고,
    "road_line(차선 페인트 선)"에 해당하는 점들만 모읍니다.

    Args:
        boundary_feature_ids: lane의 한쪽에 연결된 boundary feature id 목록
        boundary_polylines_xy_global: boundary id -> polyline 점들, shape (K,2)
        boundary_id_to_kind: boundary id -> "road_line" 또는 "road_edge"
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 yaw (라디안)

    Returns:
        points_local: shape (K,2) float32
            - road_line 점들을 전부 모아 ego 기준 좌표계로 바꾼 결과
            - road_line이 하나도 없으면 shape (0,2)
    """
    points_list: List[np.ndarray] = []

    for fid in boundary_feature_ids:
        fid_int = int(fid)

        if boundary_id_to_kind.get(fid_int, "") != "road_line":
            continue

        poly_xy_g = boundary_polylines_xy_global.get(fid_int, None)
        if poly_xy_g is None:
            continue
        if poly_xy_g.ndim != 2 or poly_xy_g.shape[1] != 2 or poly_xy_g.shape[
                0] == 0:
            continue

        points_list.append(poly_xy_g.astype(np.float32))  # shape: (Ki,2)

    if len(points_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points_xy_global = np.concatenate(points_list,
                                      axis=0).astype(np.float32)  # shape: (K,2)
    points_xy_local = transform_points_global_to_ego_local(
        points_xy_global, ego_xy_global, ego_yaw_global)  # shape: (K,2)
    return points_xy_local


def _align_boundary_points_to_center_indices(
    center_xy: np.ndarray,  # shape: (lane_len,2)
    boundary_xy: np.ndarray,  # shape: (K,2)
    max_valid_dist_m: float = 6.0,
    center_index_window: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """중심선의 각 점(j)에 대응되는 경계선 점(j)을 고릅니다.

    핵심 아이디어는 NuPlan처럼 "center[j]와 boundary[j]를 대응"시키는 것입니다.
    다만 WOMD는 경계선이 부분적으로만 있을 수 있으니,
    다음 규칙으로 '없는 구간'을 0 벡터로 처리합니다.

    처리 규칙:
    1) 중심선은 (lane_len,2)로 이미 샘플링되어 있다고 가정합니다.
    2) boundary_xy는 road_line 점들의 모음입니다(순서/연결 여부는 신뢰하지 않습니다).
    3) boundary 점마다 "가장 가까운 center 인덱스"를 구해둡니다.
    4) center[j]에서는, 그 j 주변(center_index_window 범위)으로 분류된 boundary 점들만 후보로 씁니다.
       - 이렇게 하면 lane의 다른 구간에 있는 점을 '우연히 가까워서' 끌어오는 것을 줄일 수 있습니다.
    5) 후보 중 가장 가까운 점이 max_valid_dist_m보다 멀면
       - 그 j 위치에는 경계가 없다고 보고 boundary_aligned[j] = center[j]로 둡니다.
       - 그러면 (boundary_aligned - center)가 (0,0)이 됩니다.

    Args:
        center_xy: 중심선 점들, shape (lane_len,2)
        boundary_xy: road_line 점들, shape (K,2)
        max_valid_dist_m: 이 거리보다 멀면 "경계가 없다"고 판단(보수적으로 6m 권장)
        center_index_window: center 인덱스 기준 후보를 고를 때 허용하는 인덱스 차이(기본 ±2)

    Returns:
        boundary_aligned: shape (lane_len,2) float32
            - 각 j에 대응되는 경계선 점
            - 없다고 판단되면 center[j]와 동일(그래서 벡터는 0)
        valid_mask: shape (lane_len,) bool
            - True면 그 j에서 경계점을 찾았고, False면 못 찾았다는 뜻
    """
    lane_len = int(center_xy.shape[0])

    boundary_aligned = center_xy.astype(
        np.float32).copy()  # shape: (lane_len,2)
    valid_mask = np.zeros((lane_len,), dtype=bool)  # shape: (lane_len,)

    if boundary_xy.ndim != 2 or boundary_xy.shape[1] != 2 or boundary_xy.shape[
            0] == 0:
        return boundary_aligned, valid_mask

    # boundary 점마다 가장 가까운 center 인덱스 구하기
    # d2_bc: (K, lane_len)
    diffs_bc = boundary_xy[:, None, :] - center_xy[
        None, :, :]  # shape: (K,lane_len,2)
    d2_bc = np.sum(diffs_bc * diffs_bc, axis=-1)  # shape: (K,lane_len)
    nearest_center_idx = np.argmin(d2_bc,
                                   axis=1).astype(np.int64)  # shape: (K,)

    win = int(center_index_window)
    max_d2 = float(max_valid_dist_m)**2

    for j in range(lane_len):
        # "이 center[j] 주변 구간"으로 보이는 boundary 점만 후보로 사용
        mask = np.abs(nearest_center_idx - int(j)) <= win  # shape: (K,)
        if not bool(np.any(mask)):
            continue

        cand = boundary_xy[mask]  # shape: (M,2)
        # center[j]에 가장 가까운 후보 선택
        diffs = cand - center_xy[j][None, :]  # shape: (M,2)
        d2 = np.sum(diffs * diffs, axis=1)  # shape: (M,)
        m = int(np.argmin(d2))
        if float(d2[m]) <= max_d2:
            boundary_aligned[j] = cand[m].astype(np.float32)  # shape: (2,)
            valid_mask[j] = True

    return boundary_aligned, valid_mask


def _compute_nearest_center_indices(
        points_xy: np.ndarray,  # shape: (K, 2)
        center_xy: np.ndarray,  # shape: (lane_len, 2)
) -> np.ndarray:
    """각 점이 중심선의 몇 번째 점과 가장 가까운지 인덱스를 구합니다.

    Args:
        points_xy: 경계(또는 다른 점들) 좌표, shape (K, 2)
        center_xy: 중심선 좌표(이미 lane_len개로 맞춰진 상태), shape (lane_len, 2)

    Returns:
        nearest_idx: points_xy 각 점마다 가장 가까운 center_xy 인덱스, shape (K,), dtype int64
    """
    # points_xy: (K,2), center_xy: (L,2)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if center_xy.ndim != 2 or center_xy.shape[1] != 2 or center_xy.shape[0] == 0:
        return np.zeros((points_xy.shape[0],), dtype=np.int64)

    diffs = points_xy[:, None, :] - center_xy[None, :, :]  # shape: (K, L, 2)
    d2 = np.sum(diffs * diffs, axis=-1)  # shape: (K, L)
    nearest_idx = np.argmin(d2, axis=1).astype(np.int64)  # shape: (K,)
    return nearest_idx


def _orient_and_sort_boundary_polylines_along_center(
        boundary_polylines_local: List[np.ndarray],  # each shape: (Ki, 2)
        center_sampled: np.ndarray,  # shape: (lane_len, 2)
) -> List[np.ndarray]:
    """경계선 조각(여러 개일 수 있음)을 '중심선 진행 방향'에 맞게 정리합니다.

    WOMD에서는 한쪽 경계선이 한 덩어리가 아니라 여러 조각으로 나뉘어 들어올 수 있습니다.
    NuPlan 방식(같은 인덱스끼리 빼기)을 하려면, 경계선도 "한 줄"처럼 정리한 다음 샘플링하는 게 편합니다.

    이 함수는 아래를 합니다.
    1) 각 조각이 거꾸로 들어온 경우가 있어서, 조각의 앞/뒤를 뒤집을지 판단합니다.
       - 방법: 조각의 첫 점과 끝 점이 중심선에서 가까운 인덱스를 비교합니다.
       - 끝점 쪽 인덱스가 더 작으면(=거꾸로), 조각을 뒤집습니다.
    2) 여러 조각이 있다면, 중심선 인덱스 기준으로 앞쪽 조각부터 오도록 정렬합니다.
       - 방법: 각 조각 점들이 가까운 중심선 인덱스들의 평균값을 구해 그 값으로 정렬합니다.

    Args:
        boundary_polylines_local: 경계선 조각들의 리스트(ego 기준), 각 원소 shape (Ki, 2)
        center_sampled: 중심선(ego 기준, lane_len개), shape (lane_len, 2)

    Returns:
        sorted_polylines: 방향이 맞춰지고 정렬된 경계선 조각 리스트
    """
    processed: List[Tuple[float, np.ndarray]] = []

    for poly in boundary_polylines_local:
        # poly: (Ki,2)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] == 0:
            continue

        nearest_idx = _compute_nearest_center_indices(poly,
                                                      center_sampled)  # (Ki,)
        if nearest_idx.size == 0:
            continue

        # 조각 방향 정리(거꾸로면 뒤집기)
        start_idx = int(nearest_idx[0])
        end_idx = int(nearest_idx[-1])
        poly_oriented = poly
        nearest_idx_oriented = nearest_idx
        if end_idx < start_idx:
            poly_oriented = poly[::-1].copy()
            nearest_idx_oriented = nearest_idx[::-1].copy()

        # 이 조각이 lane에서 대략 어디쯤인지(정렬용 스코어)
        score = float(nearest_idx_oriented.astype(np.float32).mean())
        processed.append((score, poly_oriented.astype(np.float32)))

    processed.sort(key=lambda x: x[0])
    return [p for _, p in processed]


def _maybe_flip_points_to_match_center_start(
        boundary_points: np.ndarray,  # shape: (K, 2)
        center_sampled: np.ndarray,  # shape: (lane_len, 2)
) -> np.ndarray:
    """NuPlan의 '플립'과 같은 의도로, 경계선의 시작점이 center[0]에 가깝게 만듭니다.

    NuPlan에서는 경계선 점들의 순서가 거꾸로 들어오는 경우를 대비해,
    'center[0]에 더 가까운 쪽이 boundary[0]'이 되도록 뒤집습니다.

    Args:
        boundary_points: 경계선 점들의 줄(ego 기준), shape (K, 2)
        center_sampled: 중심선 점들(ego 기준, lane_len개), shape (lane_len, 2)

    Returns:
        boundary_points_fixed: 필요하면 뒤집힌 boundary_points, shape (K, 2)
    """
    if boundary_points.ndim != 2 or boundary_points.shape[
            1] != 2 or boundary_points.shape[0] == 0:
        return boundary_points
    if center_sampled.ndim != 2 or center_sampled.shape[
            1] != 2 or center_sampled.shape[0] == 0:
        return boundary_points

    # d0 = boundary[0]이 center[0]과 얼마나 가까운지
    # d1 = boundary[-1]이 center[0]과 얼마나 가까운지
    d0 = float(np.linalg.norm(boundary_points[0] - center_sampled[0]))
    d1 = float(np.linalg.norm(boundary_points[-1] - center_sampled[0]))
    if d1 < d0:
        return boundary_points[::-1].copy()
    return boundary_points


def _build_lane_side_boundary_sampled_like_nuplan(
    center_sampled: np.ndarray,  # shape: (lane_len, 2), ego 기준
    boundary_feature_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
    lane_len: int,
    max_valid_dist_m: float = 6.0,
) -> np.ndarray:
    """WOMD 경계선을 NuPlan 스타일로 'lane_len개 점'으로 만든 뒤, 없는 구간은 안전하게 비웁니다.

    NuPlan 방식의 핵심은:
    - 중심선(center)과 경계선(left/right)을 각각 같은 개수(lane_len=10)로 점을 맞춘 다음,
    - 같은 인덱스끼리 빼서 (dx, dy)를 만든다는 점입니다.

    WOMD는 한쪽 경계가 여러 조각으로 나뉘어 있을 수 있고,
    어떤 구간은 경계가 아예 없을 수도 있습니다.
    그래서 이 함수는 아래 순서로 처리합니다.

    1) boundary_feature_ids 중 road_line인 것만 골라서, 각 조각의 점들을 가져옵니다. (global 좌표)
    2) ego 기준 좌표로 바꿉니다. (local 좌표)
    3) 조각들의 방향을 중심선 진행 방향에 맞게 뒤집고, 앞쪽 조각부터 오도록 정렬합니다.
    4) 정렬된 조각들을 하나의 "점들의 줄"로 이어 붙입니다.
    5) NuPlan의 플립 로직처럼, boundary[0]이 center[0]에 가깝도록 한 번 더 뒤집을 수 있습니다.
    6) 이어 붙인 경계선을 lane_len개 점으로 같은 간격으로 샘플링합니다.
    7) 샘플링된 boundary[j]가 center[j]에서 너무 멀면(> max_valid_dist_m),
       그 j는 "경계가 없는 구간"이라고 보고 boundary[j]를 center[j]로 바꿉니다.
       → 그러면 (boundary - center)가 (0,0)이 됩니다.

    Args:
        center_sampled: 중심선 점들(ego 기준), shape (lane_len, 2)
        boundary_feature_ids: lane 한쪽(left/right)에 연결된 boundary feature id 목록
        boundary_polylines_xy_global: boundary id -> 점들(global), shape (Ki, 2)
        boundary_id_to_kind: boundary id -> "road_line" / "road_edge"
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 yaw (라디안)
        lane_len: 출력 점 개수(예: 10)
        max_valid_dist_m: center[j]와 boundary[j]의 거리가 이 값보다 크면 "없음" 처리

    Returns:
        boundary_sampled: NuPlan 스타일로 만든 경계선 점들(ego 기준), shape (lane_len, 2), float32
            - 경계가 없다고 판단된 인덱스는 center_sampled와 동일하게 들어갑니다.
    """
    # center_sampled: (lane_len,2)
    if center_sampled.ndim != 2 or center_sampled.shape[
            1] != 2 or center_sampled.shape[0] != int(lane_len):
        return np.zeros((int(lane_len), 2), dtype=np.float32)

    # 1) road_line 조각 모으기 (global)
    polylines_global: List[np.ndarray] = []
    for fid in boundary_feature_ids:
        fid_int = int(fid)
        if boundary_id_to_kind.get(fid_int, "") != "road_line":
            continue
        poly_g = boundary_polylines_xy_global.get(fid_int, None)
        if poly_g is None:
            continue
        if poly_g.ndim != 2 or poly_g.shape[1] != 2 or poly_g.shape[0] == 0:
            continue
        polylines_global.append(poly_g.astype(np.float32))  # shape: (Ki,2)

    # 경계가 아예 없으면: NuPlan에서도 "없음"은 결국 벡터 0으로 처리하는 게 안전함
    if len(polylines_global) == 0:
        return center_sampled.astype(np.float32).copy()

    # 2) ego 기준(local)로 변환
    polylines_local: List[np.ndarray] = []
    for poly_g in polylines_global:
        poly_l = transform_points_global_to_ego_local(poly_g, ego_xy_global,
                                                      ego_yaw_global)  # (Ki,2)
        if poly_l.shape[0] > 0:
            polylines_local.append(poly_l.astype(np.float32))

    if len(polylines_local) == 0:
        return center_sampled.astype(np.float32).copy()

    # 3) 중심선 방향 기준으로 조각 방향/순서 정리
    polylines_sorted = _orient_and_sort_boundary_polylines_along_center(
        boundary_polylines_local=polylines_local,
        center_sampled=center_sampled,
    )
    if len(polylines_sorted) == 0:
        return center_sampled.astype(np.float32).copy()

    # 4) 이어 붙여 하나의 줄로 만들기
    boundary_points = np.concatenate(polylines_sorted,
                                     axis=0).astype(np.float32)  # shape: (K,2)
    if boundary_points.shape[0] == 0:
        return center_sampled.astype(np.float32).copy()

    # 5) NuPlan 플립(안전)
    boundary_points = _maybe_flip_points_to_match_center_start(
        boundary_points, center_sampled)  # (K,2)

    # 6) lane_len개로 샘플링
    boundary_sampled = resample_polyline_equal_distance(
        boundary_points,
        num_samples=int(lane_len),
        closed=False,
    ).astype(np.float32)  # shape: (lane_len,2)

    # 7) 너무 멀면 "없음" 처리 -> boundary[j]=center[j]
    diffs = boundary_sampled - center_sampled  # shape: (lane_len,2)
    dists = np.linalg.norm(diffs,
                           axis=1).astype(np.float32)  # shape: (lane_len,)
    far_mask = dists > float(max_valid_dist_m)  # shape: (lane_len,)
    if np.any(far_mask):
        boundary_sampled[far_mask] = center_sampled[far_mask]

    return boundary_sampled.astype(np.float32)


def build_lane_arrays(
    lanes: List[LaneInfo],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    lane_id_to_light: Dict[int, np.ndarray],
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    lane_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """차선 정보를 요구사항의 lanes 포맷으로 만듭니다.

    lanes: (lane_num, lane_len=10, 12)
        0-1  : center (x,y)
        2-3  : center vector (dx,dy)
        4-5  : left boundary vector (dx,dy)   = (closest_on_left_boundary - center)
        6-7  : right boundary vector (dx,dy)  = (closest_on_right_boundary - center)
        8-11 : lane light one-hot [GO, CAUTION, STOP, UNKNOWN]

    새 요구사항 반영 포인트:
      - left_boundaries/right_boundaries(=BoundarySegment)에서
        center 샘플 점이 속한 구간을 덮는 boundary_feature_id를 찾음
      - 그 boundary_feature_id의 RoadLine/RoadEdge polyline에 대해
        점→폴리라인 최소거리(가장 가까운 위치)를 구해서 dx,dy 계산
      - 너무 멀거나(left/right 방향 틀리면) 0으로 비움

    Returns:
        lanes_arr: (lane_num, lane_len, 12) float32
        lanes_speed_limit: (lane_num, 1) float32 (m/s)
        lanes_has_speed_limit: (lane_num, 1) bool
        lane_light: (lane_num, 4) float32
    """
    lane_num = len(lanes)

    lanes_arr = np.zeros((lane_num, int(lane_len), 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((lane_num, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((lane_num, 1), dtype=bool)
    lane_light = np.zeros((lane_num, 4), dtype=np.float32)

    # ✅ 거리 제한(“널널한” 기준)
    # - road_line: 보통 center에서 멀리 갈 일이 적어서 조금 타이트
    # - road_edge: 도로 가장자리라 더 멀 수 있어 더 넉넉
    max_dist_road_line_m = 8.0
    max_dist_road_edge_m = 15.0

    # ✅ 성능: boundary polyline ego 변환 캐시(시나리오 내에서 재사용)
    boundary_polyline_local_cache: Dict[int, np.ndarray] = {}

    for i, lane in enumerate(lanes):
        # 1) centerline 샘플링 (ego 기준) + "원본 구간 index" 같이 얻기
        center_xy_g = lane.centerline_xy_global  # (P,2)
        center_xy_l = transform_points_global_to_ego_local(
            center_xy_g, ego_xy_global, ego_yaw_global)  # (P,2)

        center_sampled, center_seg_idx = resample_polyline_equal_distance_with_segment_indices(
            center_xy_l,
            num_samples=int(lane_len),
            closed=False,
        )  # (lane_len,2), (lane_len,)

        # 2) center vector (dx,dy) (마지막은 0)
        center_vec = np.zeros((int(lane_len), 2),
                              dtype=np.float32)  # (lane_len,2)
        if int(lane_len) >= 2:
            center_vec[:-1] = center_sampled[1:] - center_sampled[:-1]
        center_vec[-1] = 0.0

        # 3) 진행방향 기반 left_normal 계산 (방향 체크용)
        _, left_normals = compute_center_tangent_and_left_normals(
            center_sampled)  # (lane_len,2)

        # 4) 새 요구사항 방식으로 left/right boundary 벡터 계산
        left_vec = compute_lane_boundary_vectors_for_side(
            center_sampled_xy_local=center_sampled,
            center_seg_indices=center_seg_idx,
            center_left_normals_unit=left_normals,
            boundary_segments=lane.left_boundary_segments,
            boundary_polylines_xy_global=boundary_polylines_xy_global,
            boundary_id_to_kind=boundary_id_to_kind,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
            boundary_polyline_local_cache=boundary_polyline_local_cache,
            side="left",
            max_dist_road_line_m=float(max_dist_road_line_m),
            max_dist_road_edge_m=float(max_dist_road_edge_m),
        )  # (lane_len,2)

        right_vec = compute_lane_boundary_vectors_for_side(
            center_sampled_xy_local=center_sampled,
            center_seg_indices=center_seg_idx,
            center_left_normals_unit=left_normals,
            boundary_segments=lane.right_boundary_segments,
            boundary_polylines_xy_global=boundary_polylines_xy_global,
            boundary_id_to_kind=boundary_id_to_kind,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
            boundary_polyline_local_cache=boundary_polyline_local_cache,
            side="right",
            max_dist_road_line_m=float(max_dist_road_line_m),
            max_dist_road_edge_m=float(max_dist_road_edge_m),
        )  # (lane_len,2)

        # 5) lane light (기존 유지)
        light = lane_id_to_light.get(
            lane.lane_id,
            np.array([0, 0, 0, 1], dtype=np.float32),
        )  # (4,)
        lane_light[i] = light
        lanes_arr[i, :, 8:12] = light[None, :]  # broadcast

        # 6) 속도제한(m/s) (기존 유지)
        if lane.speed_limit_mph > 0.0:
            lanes_has_speed_limit[i, 0] = True
            lanes_speed_limit[i, 0] = float(lane.speed_limit_mph) * 0.44704
        else:
            lanes_has_speed_limit[i, 0] = False
            lanes_speed_limit[i, 0] = 0.0

        # 7) 최종 lanes 채우기
        lanes_arr[i, :, 0:2] = center_sampled
        lanes_arr[i, :, 2:4] = center_vec
        lanes_arr[i, :, 4:6] = left_vec
        lanes_arr[i, :, 6:8] = right_vec

    return lanes_arr, lanes_speed_limit, lanes_has_speed_limit, lane_light


def get_womd_track_token(track: Any) -> str:
    """트랙(에이전트 1개)을 구분하는 고유 문자열을 얻습니다.

    - WOMD에서는 트랙을 구분하는 고유 값으로 track.id(숫자)를 제공합니다.
    - 따라서 이 함수는 track.id를 정수로 바꾼 뒤, 문자열로 바꾼 값만 반환합니다.
    - 다른 값은 보지 않고, 조합도 하지 않습니다.

    Args:
        track: scenario.tracks의 원소(트랙 1개)

    Returns:
        token: track.id를 문자열로 바꾼 값
    """
    if hasattr(track, "id"):
        return str(int(getattr(track, "id")))
    return ""


from typing import Dict, List, Tuple
import numpy as np
from typing import Any


def tfrecord_element_to_bytes(element: Any) -> bytes:
    """TFRecordDataset에서 나온 원소를 bytes로 안전하게 바꿉니다.

    환경에 따라 dataset 반복에서 나오는 값이
    - bytes
    - numpy의 bytes 스칼라
    - (혹시) tf.Tensor
    처럼 다를 수 있어서, 어느 경우든 bytes로 통일해 반환합니다.

    Args:
        element: dataset에서 나온 원소(한 레코드).

    Returns:
        record_bytes: bytes. Scenario.ParseFromString에 바로 넣을 수 있는 형태.
    """
    if isinstance(element, (bytes, bytearray)):
        return bytes(element)

    # tf.Tensor 같은 경우
    if hasattr(element, "numpy"):
        return bytes(element.numpy())

    # numpy 스칼라/배열 같은 경우
    if hasattr(element, "tobytes"):
        return element.tobytes()

    return bytes(element)

def _build_past_future_vxy_for_control_cache(
    ego_agent_past: np.ndarray,             # shape: (T, 11)
    ego_future_gt_11_dim: np.ndarray,       # shape: (F, 11)
    neighbor_agents_past: np.ndarray,       # shape: (A, T, 11)
    neighbor_future_gt_11_dim: np.ndarray,  # shape: (A, F, 11)
) -> Tuple[np.ndarray, np.ndarray]:
    """ego/neighbor의 v_x,v_y 시계열을 (past+future)로 이어붙여 만듭니다.

    Args:
        ego_agent_past:
            shape: (T, 11). 과거+현재(현재 포함).
        ego_future_gt_11_dim:
            shape: (F, 11). 미래(현재 제외).
        neighbor_agents_past:
            shape: (A, T, 11). 이웃 과거+현재.
        neighbor_future_gt_11_dim:
            shape: (A, F, 11). 이웃 미래.

    Returns:
        ego_future_vxy:
            shape: (T+F, 2). [v_x, v_y]
        neighbor_future_vxy:
            shape: (A, T+F, 2). [v_x, v_y]
    """
    ego_future_vxy = build_past_current_and_future_vxy_from_feat_11(
        past_feat_11=ego_agent_past,          # (T,11)
        future_feat_11=ego_future_gt_11_dim,  # (F,11)
    ).astype(np.float32)  # (T+F,2)

    neighbor_future_vxy = build_past_current_and_future_vxy_from_feat_11(
        past_feat_11=neighbor_agents_past,         # (A,T,11)
        future_feat_11=neighbor_future_gt_11_dim,  # (A,F,11)
    ).astype(np.float32)  # (A,T+F,2)

    return ego_future_vxy, neighbor_future_vxy


def _build_cs_yaw_and_valid_for_control_cache(
    ego_agent_past: np.ndarray,             # shape: (T, 11)
    ego_future_gt_11_dim: np.ndarray,       # shape: (F, 11)
    neighbor_agents_past: np.ndarray,       # shape: (A, T, 11)
    neighbor_future_gt_11_dim: np.ndarray,  # shape: (A, F, 11)
    *,
    prefix_dim: int = 8,
) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(past+future) 11차원에서 cs_yaw와 valid 마스크를 만듭니다.

    - cs_yaw는 11차원의 [cos(yaw), sin(yaw)]를 그대로 씁니다.
    - valid는 "앞 prefix_dim개 값이 전부 0이면 무효" 규칙을 씁니다.

    Args:
        ego_agent_past:
            shape: (T,11)
        ego_future_gt_11_dim:
            shape: (F,11)
        neighbor_agents_past:
            shape: (A,T,11)
        neighbor_future_gt_11_dim:
            shape: (A,F,11)
        prefix_dim:
            유효/무효 판단에 쓸 앞쪽 차원 길이(기본 8).

    Returns:
        point_len:
            T+F (예: 101)
        ego_cs_yaw:
            shape: (point_len, 2)  -> [cos(yaw), sin(yaw)]
        neighbor_cs_yaw:
            shape: (A, point_len, 2)
        ego_valid:
            shape: (point_len,) bool
        neighbor_valid:
            shape: (A, point_len) bool
    """
    ego_past = np.asarray(ego_agent_past, dtype=np.float32)
    ego_fut = np.asarray(ego_future_gt_11_dim, dtype=np.float32)
    neigh_past = np.asarray(neighbor_agents_past, dtype=np.float32)
    neigh_fut = np.asarray(neighbor_future_gt_11_dim, dtype=np.float32)

    if ego_past.ndim != 2 or int(ego_past.shape[1]) != 11:
        raise ValueError(f"ego_agent_past must be (T,11). got={ego_past.shape}")
    if ego_fut.ndim != 2 or int(ego_fut.shape[1]) != 11:
        raise ValueError(f"ego_future_gt_11_dim must be (F,11). got={ego_fut.shape}")
    if neigh_past.ndim != 3 or int(neigh_past.shape[2]) != 11:
        raise ValueError(f"neighbor_agents_past must be (A,T,11). got={neigh_past.shape}")
    if neigh_fut.ndim != 3 or int(neigh_fut.shape[2]) != 11:
        raise ValueError(f"neighbor_future_gt_11_dim must be (A,F,11). got={neigh_fut.shape}")

    # (1) past+future 11차원
    ego_pf_11 = np.concatenate([ego_past, ego_fut], axis=0).astype(np.float32)      # (T+F,11)
    neighbor_pf_11 = np.concatenate([neigh_past, neigh_fut], axis=1).astype(np.float32)  # (A,T+F,11)

    point_len = int(ego_pf_11.shape[0])
    if int(neighbor_pf_11.shape[1]) != point_len:
        raise ValueError(
            "past+future length mismatch between ego and neighbor. "
            f"ego point_len={point_len}, neighbor.shape={neighbor_pf_11.shape}"
        )

    # (2) cs_yaw
    ego_cs_yaw = ego_pf_11[:, 2:4].astype(np.float32)            # (point_len,2)
    neighbor_cs_yaw = neighbor_pf_11[:, :, 2:4].astype(np.float32)  # (A,point_len,2)

    # (3) valid 마스크
    ego_valid = _compute_valid_mask_from_prefix_nonzero(
        ego_pf_11,
        prefix_dim=int(prefix_dim),
    ).astype(bool)  # (point_len,)

    neighbor_valid = _compute_valid_mask_from_prefix_nonzero(
        neighbor_pf_11,
        prefix_dim=int(prefix_dim),
    ).astype(bool)  # (A,point_len)

    return point_len, ego_cs_yaw, neighbor_cs_yaw, ego_valid, neighbor_valid


def build_past_future_control_for_cache(
    ego_agent_past: np.ndarray,             # shape: (T, 11)
    ego_future_gt_11_dim: np.ndarray,       # shape: (F, 11)
    neighbor_agents_past: np.ndarray,       # shape: (A, T, 11)
    neighbor_future_gt_11_dim: np.ndarray,  # shape: (A, F, 11)
    *,
    dt_sec: float,
    polyorder: int = 2,
    max_window_len_yaw: int = 7,
    prefix_dim: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """캐싱용 control([v_x,v_y,yaw_rate])을 ego/neighbor에 대해 만듭니다.

    Args:
        ego_agent_past:
            shape: (T,11)
        ego_future_gt_11_dim:
            shape: (F,11)
        neighbor_agents_past:
            shape: (A,T,11)
        neighbor_future_gt_11_dim:
            shape: (A,F,11)
        dt_sec:
            샘플 사이 시간 간격(초).
        polyorder:
            yaw_rate를 만들 때 쓰는 계산 차수(기본 2).
        max_window_len_yaw:
            yaw_rate를 만들 때 쓰는 창 최대 길이(기본 7).
        prefix_dim:
            valid 판단 기준(기본 8).

    Returns:
        ego_past_future_control:
            shape: (point_len,3) -> [v_x,v_y,yaw_rate]
        neighbor_past_future_control:
            shape: (A,point_len,3)
    """
    # (1) vxy
    ego_future_vxy, neighbor_future_vxy = _build_past_future_vxy_for_control_cache(
        ego_agent_past=ego_agent_past,
        ego_future_gt_11_dim=ego_future_gt_11_dim,
        neighbor_agents_past=neighbor_agents_past,
        neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,
    )  # (point_len,2), (A,point_len,2)

    # (2) cs_yaw + valid
    point_len, ego_cs_yaw, neighbor_cs_yaw, ego_valid, neighbor_valid = _build_cs_yaw_and_valid_for_control_cache(
        ego_agent_past=ego_agent_past,
        ego_future_gt_11_dim=ego_future_gt_11_dim,
        neighbor_agents_past=neighbor_agents_past,
        neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,
        prefix_dim=int(prefix_dim),
    )

    # (3) yaw_rate
    ego_yaw_rate, neighbor_yaw_rate = compute_past_future_yaw_rate_from_cs_yaw_via_feasible_projector(
        ego_past_future_gt_cs_yaw=ego_cs_yaw,                 # (point_len,2)
        neighbor_past_future_gt_cs_yaw=neighbor_cs_yaw,       # (A,point_len,2)
        ego_pf_valid=ego_valid,                               # (point_len,)
        neighbor_pf_valid=neighbor_valid,                     # (A,point_len)
        dt_sec=float(dt_sec),
        polyorder=int(polyorder),
        max_window_len_yaw=int(max_window_len_yaw),
    )  # (point_len,), (A,point_len)

    # (4) control = [v_x, v_y, yaw_rate]
    ego_past_future_control = np.concatenate(
        [ego_future_vxy, ego_yaw_rate[:, None]],
        axis=1,
    ).astype(np.float32)  # (point_len,3)

    if int(neighbor_future_vxy.shape[0]) > 0:
        neighbor_past_future_control = np.concatenate(
            [neighbor_future_vxy, neighbor_yaw_rate[..., None]],
            axis=2,
        ).astype(np.float32)  # (A,point_len,3)
    else:
        neighbor_past_future_control = np.zeros((0, int(point_len), 3), dtype=np.float32)

    return (
        ego_past_future_control,
        neighbor_past_future_control,
    )

def build_target_past_future_body_seg_control_for_cache(
    ego_agent_past: np.ndarray,                 # shape: (TIME_LEN, 11)
    ego_future_gt_11_dim: np.ndarray,           # shape: (FUTURE_LEN, 11)
    neighbor_agents_past: np.ndarray,           # shape: (A, TIME_LEN, 11)
    neighbor_future_gt_11_dim: np.ndarray,      # shape: (A, FUTURE_LEN, 11)
    ego_past_future_control: np.ndarray,        # shape: (point_len, 3)
    neighbor_past_future_control: np.ndarray,   # shape: (A, point_len, 3)
    *,
    prefix_dim: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """ego/neighbor control을 합쳐 midpoint body segment control을 만듭니다.

    이 함수는 캐싱 과정에서 이미 만든
      - ego_past_future_control: (point_len, 3)
      - neighbor_past_future_control: (A, point_len, 3)
    를 이용해

      1) target_past_future_control: (1+A, point_len, 3)
         - 첫 번째는 ego, 그 뒤는 neighbor 순서

      2) FeasibleProjector.compute_midpoint_controls()를 호출해서
         target_past_future_body_seg_control: (1+A, point_len-1, 3)
         - 각 구간 [t_k, t_{k+1})의 “중간 시점 기준” 제어
         - 출력의 3채널은 [v_x^b, v_y^b, yaw_rate] 입니다.
         - 여기서 b는 “그 객체의 진행 방향(heading) 기준”으로 회전한 좌표를 뜻합니다.

    주의:
      - compute_midpoint_controls는 control만 받는 게 아니라,
        같은 시간축의 (x,y,cos,sin)과 유효 마스크도 필요합니다.
      - 그래서 이 함수 내부에서 ego/neighbor의 11차원(past/future)에서
        (x,y,cos,sin)과 valid 마스크를 같이 만들어 넣습니다.

    Args:
        ego_agent_past:
            shape: (TIME_LEN, 11). 과거+현재(현재 포함).
        ego_future_gt_11_dim:
            shape: (FUTURE_LEN, 11). 미래(현재 제외).
        neighbor_agents_past:
            shape: (A, TIME_LEN, 11).
        neighbor_future_gt_11_dim:
            shape: (A, FUTURE_LEN, 11).
        ego_past_future_control:
            shape: (point_len, 3) = (TIME_LEN+FUTURE_LEN, 3)
        neighbor_past_future_control:
            shape: (A, point_len, 3)
        prefix_dim:
            valid 마스크를 만들 때 “앞 prefix_dim개가 전부 0이면 무효” 규칙에 쓰는 값(기본 8).

    Returns:
        target_past_future_control:
            shape: (1+A, point_len, 3)
        target_past_future_body_seg_control:
            shape: (1+A, point_len-1, 3) = (1+A, past_len+future_len, 3)
    """
    ego_past = np.asarray(ego_agent_past, dtype=np.float32)
    ego_fut = np.asarray(ego_future_gt_11_dim, dtype=np.float32)
    neigh_past = np.asarray(neighbor_agents_past, dtype=np.float32)
    neigh_fut = np.asarray(neighbor_future_gt_11_dim, dtype=np.float32)

    ego_ctrl = np.asarray(ego_past_future_control, dtype=np.float32)
    neigh_ctrl = np.asarray(neighbor_past_future_control, dtype=np.float32)

    if ego_past.ndim != 2 or int(ego_past.shape[1]) != 11:
        raise ValueError(f"ego_agent_past must be (TIME_LEN,11). got={ego_past.shape}")
    if ego_fut.ndim != 2 or int(ego_fut.shape[1]) != 11:
        raise ValueError(f"ego_future_gt_11_dim must be (FUTURE_LEN,11). got={ego_fut.shape}")
    if neigh_past.ndim != 3 or int(neigh_past.shape[2]) != 11:
        raise ValueError(f"neighbor_agents_past must be (A,TIME_LEN,11). got={neigh_past.shape}")
    if neigh_fut.ndim != 3 or int(neigh_fut.shape[2]) != 11:
        raise ValueError(f"neighbor_future_gt_11_dim must be (A,FUTURE_LEN,11). got={neigh_fut.shape}")

    past_len = int(ego_past.shape[0]) - 1
    if past_len < 0:
        raise ValueError(f"TIME_LEN must be >= 1. got TIME_LEN={int(ego_past.shape[0])}")
    future_len = int(ego_fut.shape[0])
    point_len = int(past_len + 1 + future_len)

    if ego_ctrl.ndim != 2 or int(ego_ctrl.shape[1]) != 3:
        raise ValueError(f"ego_past_future_control must be (point_len,3). got={ego_ctrl.shape}")
    if int(ego_ctrl.shape[0]) != point_len:
        raise ValueError(
            "ego_past_future_control length mismatch. "
            f"expected point_len={point_len}, got={int(ego_ctrl.shape[0])}"
        )

    if neigh_ctrl.ndim != 3 or int(neigh_ctrl.shape[2]) != 3:
        raise ValueError(f"neighbor_past_future_control must be (A,point_len,3). got={neigh_ctrl.shape}")
    if int(neigh_ctrl.shape[1]) != point_len:
        raise ValueError(
            "neighbor_past_future_control point_len mismatch. "
            f"expected point_len={point_len}, got={int(neigh_ctrl.shape[1])}"
        )

    # ------------------------------------------------------------
    # (1) target_past_future_control: (1+A, point_len, 3)
    # ------------------------------------------------------------
    target_past_future_control = np.concatenate(
        [ego_ctrl[None, ...], neigh_ctrl],
        axis=0,
    ).astype(np.float32)  # (1+A, point_len, 3)

    # ------------------------------------------------------------
    # (2) compute_midpoint_controls 입력 만들기
    #   - unnorm_diffusion_trajectory: (B=1, Pnn=1+A, 1+future_len, 4)
    #   - unnorm_near_past_xyyaw:      (B=1, Pnn=1+A, past_len, 4) or None
    #   - target_past_cur_future_valid:(B=1, Pnn=1+A, point_len) bool
    # ------------------------------------------------------------
    ego_near_past_xyyaw = ego_past[:past_len, 0:4].astype(np.float32)  # (past_len,4)
    ego_cur_future_xyyaw = np.concatenate(
        [ego_past[past_len:past_len + 1, 0:4], ego_fut[:, 0:4]],
        axis=0,
    ).astype(np.float32)  # (1+future_len,4)

    neighbor_near_past_xyyaw = neigh_past[:, :past_len, 0:4].astype(np.float32)  # (A,past_len,4)
    neighbor_cur_future_xyyaw = np.concatenate(
        [neigh_past[:, past_len:past_len + 1, 0:4], neigh_fut[:, :, 0:4]],
        axis=1,
    ).astype(np.float32)  # (A,1+future_len,4)

    target_near_past_xyyaw_pnn = np.concatenate(
        [ego_near_past_xyyaw[None, ...], neighbor_near_past_xyyaw],
        axis=0,
    ).astype(np.float32)  # (1+A, past_len, 4)

    target_cur_future_xyyaw_pnn = np.concatenate(
        [ego_cur_future_xyyaw[None, ...], neighbor_cur_future_xyyaw],
        axis=0,
    ).astype(np.float32)  # (1+A, 1+future_len, 4)

    ego_pf_11 = np.concatenate([ego_past, ego_fut], axis=0).astype(np.float32)  # (point_len,11)
    neighbor_pf_11 = np.concatenate([neigh_past, neigh_fut], axis=1).astype(np.float32)  # (A,point_len,11)

    ego_valid = _compute_valid_mask_from_prefix_nonzero(
        ego_pf_11,
        prefix_dim=int(prefix_dim),
    ).astype(bool)  # (point_len,)

    neighbor_valid = _compute_valid_mask_from_prefix_nonzero(
        neighbor_pf_11,
        prefix_dim=int(prefix_dim),
    ).astype(bool)  # (A,point_len)

    target_valid_pnn = np.concatenate(
        [ego_valid[None, ...], neighbor_valid],
        axis=0,
    ).astype(bool)  # (1+A, point_len)

    # torch 입력으로 변환 (B=1)
    target_past_future_control_t = torch.from_numpy(
        np.ascontiguousarray(target_past_future_control[None, ...])
    )  # (1, 1+A, point_len, 3)

    unnorm_diffusion_trajectory_t = torch.from_numpy(
        np.ascontiguousarray(target_cur_future_xyyaw_pnn[None, ...])
    )  # (1, 1+A, 1+future_len, 4)

    if past_len > 0:
        unnorm_near_past_xyyaw_t: Optional[torch.Tensor] = torch.from_numpy(
            np.ascontiguousarray(target_near_past_xyyaw_pnn[None, ...])
        )  # (1, 1+A, past_len, 4)
    else:
        unnorm_near_past_xyyaw_t = None

    target_past_cur_future_valid_t = torch.from_numpy(
        np.ascontiguousarray(target_valid_pnn[None, ...])
    )  # (1, 1+A, point_len) bool

    # ------------------------------------------------------------
    # (3) compute_midpoint_controls 호출
    # ------------------------------------------------------------
    fp = _get_feasible_projector_for_cache()
    with torch.no_grad():
        target_body_seg_control_t = fp.compute_midpoint_controls(
            unnorm_diffusion_trajectory=unnorm_diffusion_trajectory_t.float(),
            unnorm_near_past_xyyaw=None if unnorm_near_past_xyyaw_t is None else unnorm_near_past_xyyaw_t.float(),
            unnorm_points_world_control=target_past_future_control_t.float(),
            target_past_cur_future_valid=target_past_cur_future_valid_t.to(torch.bool),
        )  # (1, 1+A, point_len-1, 3)

    target_past_future_body_seg_control = (
        target_body_seg_control_t.squeeze(0).cpu().numpy().astype(np.float32)
    )  # (1+A, point_len-1, 3)

    return target_past_future_control, target_past_future_body_seg_control

# =========================
# [추가] target control -> filter_and_integrate 입력/출력 만들기
# =========================

_FEASIBLE_PROJECTOR_INTEGRATE_CACHE: Optional[FeasibleProjector] = None


def _get_feasible_projector_for_integration_cache() -> FeasibleProjector:
    """캐싱 스크립트에서 filter_and_integrate 용 FeasibleProjector를 워커당 1개만 만든 뒤 재사용합니다.

    - 신경망(use_feasible_dl)은 쓰지 않습니다(False).
    - 제약 기반 필터(use_feasible_filter)는 켭니다(True).
    - 배치 적분(use_batch_integration)은 켭니다(True) -> CPU에서도 보통 더 빠릅니다.

    Returns:
        FeasibleProjector: 워커 프로세스 내에서 재사용되는 인스턴스.
    """
    global _FEASIBLE_PROJECTOR_INTEGRATE_CACHE
    if _FEASIBLE_PROJECTOR_INTEGRATE_CACHE is not None:
        return _FEASIBLE_PROJECTOR_INTEGRATE_CACHE

    cfg = SimpleNamespace(
        use_batch_integration=True,
        feasible_debug_check_mask=False,
    )

    fp = FeasibleProjector(
        config=cfg,
        hidden_dim=1,              # use_feasible_dl=False면 사실상 사용되지 않음
        use_feasible_dl=False,
        use_feasible_filter=True,  # ✅ 제약 필터 ON
    )
    fp.eval()
    _FEASIBLE_PROJECTOR_INTEGRATE_CACHE = fp
    return fp


def build_target_current_state_for_cache(
    ego_agent_past: np.ndarray,          # shape: (TIME_LEN, 11)
    neighbor_agents_past: np.ndarray,    # shape: (A, TIME_LEN, 11)
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """ego/neighbor의 '현재 상태'(x,y,cos,sin)를 (1+A,4)로 만듭니다.

    Args:
        ego_agent_past:
            shape: (TIME_LEN, 11)
            마지막 시점이 현재입니다.
        neighbor_agents_past:
            shape: (A, TIME_LEN, 11)
            마지막 시점이 현재입니다.
        eps:
            cos/sin 정규화에서 0 나눗셈을 막는 작은 값입니다.

    Returns:
        target_current_state:
            shape: (1+A, 4), dtype float32
            [x, y, cos(yaw), sin(yaw)]
    """
    ego_past = np.asarray(ego_agent_past, dtype=np.float32)
    neigh_past = np.asarray(neighbor_agents_past, dtype=np.float32)

    if ego_past.ndim != 2 or int(ego_past.shape[1]) < 4:
        raise ValueError(f"ego_agent_past must be (TIME_LEN, >=4). got={ego_past.shape}")
    if neigh_past.ndim != 3 or int(neigh_past.shape[2]) < 4:
        raise ValueError(f"neighbor_agents_past must be (A, TIME_LEN, >=4). got={neigh_past.shape}")

    ego_cur = ego_past[-1, 0:4].astype(np.float32)  # shape: (4,)

    A = int(neigh_past.shape[0])
    if A > 0:
        neigh_cur = neigh_past[:, -1, 0:4].astype(np.float32)  # shape: (A,4)
        out = np.concatenate([ego_cur[None, :], neigh_cur], axis=0).astype(np.float32)  # (1+A,4)
    else:
        out = ego_cur[None, :].astype(np.float32)  # (1,4)

    # cos/sin 정규화(수치 안전): (cos,sin)이 (0,0)이면 그대로 0 유지
    cs = out[:, 2:4]  # shape: (1+A,2)
    norm = np.sqrt(np.sum(cs * cs, axis=1, keepdims=True)).astype(np.float32)  # (1+A,1)
    norm = np.maximum(norm, float(eps)).astype(np.float32)
    out[:, 2:4] = cs / norm
    return out


def build_target_cur_future_valid_for_cache(
    ego_agent_past: np.ndarray,              # shape: (TIME_LEN, 11)
    ego_future_gt_11_dim: np.ndarray,        # shape: (FUTURE_LEN, 11)
    neighbor_agents_past: np.ndarray,        # shape: (A, TIME_LEN, 11)
    neighbor_future_gt_11_dim: np.ndarray,   # shape: (A, FUTURE_LEN, 11)
    *,
    prefix_dim: int = 8,
) -> np.ndarray:
    """ego/neighbor의 (현재+미래) 유효 마스크를 (1+A, 1+future_len)로 만듭니다.

    유효 판단 규칙:
      - 마지막 차원에서 앞 prefix_dim개가 전부 0이면 무효(False)
      - 하나라도 0이 아니면 유효(True)

    Args:
        ego_agent_past: (TIME_LEN,11)
        ego_future_gt_11_dim: (FUTURE_LEN,11)
        neighbor_agents_past: (A,TIME_LEN,11)
        neighbor_future_gt_11_dim: (A,FUTURE_LEN,11)
        prefix_dim: 유효/무효 판단에 쓸 앞쪽 차원 길이(기본 8)

    Returns:
        target_cur_future_valid:
            shape: (1+A, 1+future_len), dtype bool
    """
    ego_past = np.asarray(ego_agent_past, dtype=np.float32)
    ego_fut = np.asarray(ego_future_gt_11_dim, dtype=np.float32)
    neigh_past = np.asarray(neighbor_agents_past, dtype=np.float32)
    neigh_fut = np.asarray(neighbor_future_gt_11_dim, dtype=np.float32)

    if ego_past.ndim != 2 or int(ego_past.shape[1]) != 11:
        raise ValueError(f"ego_agent_past must be (TIME_LEN,11). got={ego_past.shape}")
    if ego_fut.ndim != 2 or int(ego_fut.shape[1]) != 11:
        raise ValueError(f"ego_future_gt_11_dim must be (FUTURE_LEN,11). got={ego_fut.shape}")
    if neigh_past.ndim != 3 or int(neigh_past.shape[2]) != 11:
        raise ValueError(f"neighbor_agents_past must be (A,TIME_LEN,11). got={neigh_past.shape}")
    if neigh_fut.ndim != 3 or int(neigh_fut.shape[2]) != 11:
        raise ValueError(f"neighbor_future_gt_11_dim must be (A,FUTURE_LEN,11). got={neigh_fut.shape}")

    time_len = int(ego_past.shape[0])              # = past_len + 1
    past_len = int(time_len - 1)
    future_len = int(ego_fut.shape[0])
    point_len = int(time_len + future_len)

    ego_pf_11 = np.concatenate([ego_past, ego_fut], axis=0).astype(np.float32)            # (point_len,11)
    A = int(neigh_past.shape[0])
    if A > 0:
        neigh_pf_11 = np.concatenate([neigh_past, neigh_fut], axis=1).astype(np.float32) # (A,point_len,11)
        if int(neigh_pf_11.shape[1]) != point_len:
            raise ValueError(f"neighbor point_len mismatch. expected={point_len}, got={neigh_pf_11.shape}")
    else:
        neigh_pf_11 = np.zeros((0, point_len, 11), dtype=np.float32)

    ego_valid = _compute_valid_mask_from_prefix_nonzero(
        ego_pf_11, prefix_dim=int(prefix_dim)
    ).astype(bool)  # (point_len,)

    if A > 0:
        neigh_valid = _compute_valid_mask_from_prefix_nonzero(
            neigh_pf_11, prefix_dim=int(prefix_dim)
        ).astype(bool)  # (A,point_len)
        valid_pnn = np.concatenate([ego_valid[None, :], neigh_valid], axis=0).astype(bool)  # (1+A,point_len)
    else:
        valid_pnn = ego_valid[None, :].astype(bool)  # (1,point_len)

    cur_future_valid = valid_pnn[:, past_len:]  # (1+A, 1+future_len)

    # 안전 체크: (1, Pnn, T1) 형태로 만들어 검사
    assert_cur_future_valid_mask_np(
        cur_future_valid[None, ...],
        context="build_target_cur_future_valid_for_cache",
    )
    return cur_future_valid.astype(bool)

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch


@dataclass
class NeighborXYYawLossRunningSums:
    """시나리오 단위로 계산한 neighbor xy/yaw loss를 누적해서 평균/분산을 만들기 위한 합계들.

    - 여기서 “loss 1개”는 시나리오 1개에 대해 `_compute_xy_yaw_losses`가 만든 스칼라 값입니다.
    - 이 값들을 시나리오마다 모아:
      - 평균 = sum / count
      - 분산(모집단 분산) = (sum_sq / count) - mean^2
      로 계산합니다.

    Attributes:
        count (int): 유효한 neighbor loss를 계산할 수 있었던 시나리오 개수. shape: ()
        sum_xy (float): xy loss 합. shape: ()
        sum_sq_xy (float): xy loss 제곱 합. shape: ()
        sum_yaw (float): yaw loss 합. shape: ()
        sum_sq_yaw (float): yaw loss 제곱 합. shape: ()
        skipped_no_valid (int): “유효 neighbor 프레임이 0개”라서 스킵된 시나리오 수. shape: ()
    """
    count: int = 0
    sum_xy: float = 0.0
    sum_sq_xy: float = 0.0
    sum_yaw: float = 0.0
    sum_sq_yaw: float = 0.0
    skipped_no_valid: int = 0


# 워커 프로세스 내부에서만 쓰는 “파일 1개 처리용” 로컬 누적
_WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS: Optional[NeighborXYYawLossRunningSums] = None

# 워커->메인 공유(파일 단위로만 합산)
_NEIGHBOR_XY_YAW_SUMS_PROXY: Optional[Any] = None
_NEIGHBOR_XY_YAW_SUMS_LOCK: Optional[Any] = None

from typing import Any, Optional, Tuple
import os
import math
import uuid
from pathlib import Path

import numpy as np


def build_target_current_wl_from_current_feat11_for_cache(
    ego_agents_past: np.ndarray,          # shape: (TIME_LEN, 11)
    neighbor_agents_past: np.ndarray,     # shape: (N, TIME_LEN, 11)
) -> np.ndarray:
    """ego/neighbor의 현재 시점 width/length를 (1+N,2)로 만듭니다.

    11차원 agent 특징 포맷(요약):
      [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]
    따라서 width/length는 항상 [6:8]에 있습니다.

    Args:
        ego_agents_past: (TIME_LEN, 11) float32.
            마지막 시점(-1)이 현재입니다.
        neighbor_agents_past: (N, TIME_LEN, 11) float32.
            마지막 시점(-1)이 현재입니다.

    Returns:
        target_current_wl: (1+N, 2) float32.
            [:,0] = width, [:,1] = length
            첫 행은 ego, 그 뒤는 neighbor 순서입니다.
    """
    ego = np.asarray(ego_agents_past, dtype=np.float32)
    neigh = np.asarray(neighbor_agents_past, dtype=np.float32)

    if ego.ndim != 2 or int(ego.shape[1]) < 8:
        raise ValueError(f"ego_agents_past must be (TIME_LEN, >=8). got={ego.shape}")
    if neigh.ndim != 3 or int(neigh.shape[2]) < 8:
        raise ValueError(f"neighbor_agents_past must be (N, TIME_LEN, >=8). got={neigh.shape}")

    ego_wl = ego[-1, 6:8].reshape(1, 2).astype(np.float32)  # (1,2)

    n = int(neigh.shape[0])
    if n > 0:
        neigh_wl = neigh[:, -1, 6:8].astype(np.float32)     # (N,2)
        out = np.concatenate([ego_wl, neigh_wl], axis=0).astype(np.float32)  # (1+N,2)
    else:
        out = ego_wl.astype(np.float32)  # (1,2)

    return out


def _rect_poly_and_heading_xy(
    center_xy: np.ndarray,  # shape: (2,)
    cos_yaw: float,
    sin_yaw: float,
    width: float,
    length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """중심/방향/가로세로로 “방향 표시가 되는” 사각형 외곽선(poly)과 전방 포인트를 만듭니다.

    Args:
        center_xy: (2,) [x,y]
        cos_yaw: cos(yaw)
        sin_yaw: sin(yaw)
        width: 가로(옆 폭)
        length: 세로(앞뒤 길이)

    Returns:
        poly_xy: (5,2) float32. 사각형 외곽선(마지막은 첫 점으로 닫음).
        head_xy: (2,) float32. 중심에서 “앞 방향”을 가리키는 점(전방 중심).
    """
    c = np.asarray(center_xy, dtype=np.float32).reshape(2,)
    w = float(width)
    l = float(length)

    hl = 0.5 * l
    hw = 0.5 * w

    # local corners: 앞(+x) 기준으로 길이, 옆(+y) 기준으로 폭
    local = np.array(
        [
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ],
        dtype=np.float32,
    )  # (4,2)

    # cos/sin이 정규화가 아니어도 atan2로 yaw 복원 가능하지만,
    # 회전엔 “정규화된” 방향이 더 안전합니다(0 나눗셈 방지).
    norm = math.sqrt(float(cos_yaw) * float(cos_yaw) + float(sin_yaw) * float(sin_yaw))
    if norm <= 1e-6:
        cy = 1.0
        sy = 0.0
    else:
        cy = float(cos_yaw) / norm
        sy = float(sin_yaw) / norm

    rot = np.array([[cy, -sy], [sy, cy]], dtype=np.float32)  # (2,2)

    world = (local @ rot.T) + c[None, :]  # (4,2)
    poly = np.vstack([world, world[:1]]).astype(np.float32)  # (5,2)

    head_local = np.array([hl, 0.0], dtype=np.float32)       # (2,)
    head_xy = (head_local @ rot.T) + c                        # (2,)

    return poly, head_xy.astype(np.float32)


def save_rect_trajectory_comparison_png(
    out_dir: str,
    *,
    scenario_id: str,
    target_integrated_trajectory: np.ndarray,  # shape: (1+N, F, 4)
    target_future_gt_4_dim: np.ndarray,        # shape: (1+N, F, 4)
    target_current_wl: np.ndarray,             # shape: (1+N, 2)
    target_cur_future_valid: np.ndarray,       # shape: (1+N, 1+F)
    dpi: int = 150,
    step_stride: int = 1,
) -> str:
    """integrated vs GT를 사각형(방향 포함) 궤적으로 겹쳐 그린 png를 저장합니다.

    Args:
        out_dir: 저장 폴더 경로.
        scenario_id: 파일명에 포함할 시나리오 id.
        target_integrated_trajectory: (1+N, F, 4) [x,y,cos,sin]
        target_future_gt_4_dim: (1+N, F, 4) [x,y,cos,sin]
        target_current_wl: (1+N, 2) [width,length]
        target_cur_future_valid: (1+N, 1+F) bool. 현재+미래 유효 마스크.
        dpi: 저장 dpi.
        step_stride: 몇 스텝 간격으로 사각형을 그릴지(1이면 전부).

    Returns:
        saved_path: 저장된 png 경로(str). 실패하면 "".
    """
    integ = np.asarray(target_integrated_trajectory, dtype=np.float32)
    gt = np.asarray(target_future_gt_4_dim, dtype=np.float32)
    wl = np.asarray(target_current_wl, dtype=np.float32)
    valid_cf = np.asarray(target_cur_future_valid, dtype=bool)

    if integ.ndim != 3 or int(integ.shape[2]) != 4:
        raise ValueError(f"target_integrated_trajectory must be (P,F,4). got={integ.shape}")
    if gt.shape != integ.shape:
        raise ValueError(f"target_future_gt_4_dim shape mismatch. gt={gt.shape}, integ={integ.shape}")
    if wl.shape != (int(integ.shape[0]), 2):
        raise ValueError(f"target_current_wl must be (P,2). got={wl.shape}, P={int(integ.shape[0])}")
    if valid_cf.shape != (int(integ.shape[0]), int(integ.shape[1]) + 1):
        raise ValueError(
            "target_cur_future_valid must be (P,1+F). "
            f"got={valid_cf.shape}, expected={(int(integ.shape[0]), int(integ.shape[1]) + 1)}"
        )

    # 미래 유효 마스크: (P,F)
    valid_f = valid_cf[:, 1:]  # (P,F)

    # matplotlib는 환경에 따라 없을 수 있어, 없으면 조용히 스킵
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception:
        return ""

    # 저장 경로 준비(파일명 충돌 방지: pid + uuid)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    fname = f"{scenario_id}_pid{os.getpid()}_{uuid.uuid4().hex}.png"
    save_path = str(Path(out_dir) / fname)

    # 보기 범위 계산(유효한 xy만)
    def _collect_valid_xy(arr_4: np.ndarray) -> np.ndarray:
        # arr_4: (P,F,4)
        xy = arr_4[:, :, 0:2]  # (P,F,2)
        m = valid_f
        if not bool(np.any(m)):
            return np.zeros((0, 2), dtype=np.float32)
        pts = xy[m].reshape(-1, 2).astype(np.float32)  # (K,2)
        return pts

    pts_all = []
    pts_i = _collect_valid_xy(integ)
    pts_g = _collect_valid_xy(gt)
    if pts_i.shape[0] > 0:
        pts_all.append(pts_i)
    if pts_g.shape[0] > 0:
        pts_all.append(pts_g)

    if len(pts_all) > 0:
        pts = np.concatenate(pts_all, axis=0)  # (K,2)
        x_min = float(np.min(pts[:, 0]))
        x_max = float(np.max(pts[:, 0]))
        y_min = float(np.min(pts[:, 1]))
        y_max = float(np.max(pts[:, 1]))
    else:
        x_min, x_max, y_min, y_max = -10.0, 10.0, -10.0, 10.0

    # 여유 마진(차 크기 고려)
    max_len = float(np.max(wl[:, 1])) if wl.size > 0 else 4.0
    max_wid = float(np.max(wl[:, 0])) if wl.size > 0 else 2.0
    margin = max(10.0, 2.0 * max(max_len, max_wid))
    x_min -= margin
    x_max += margin
    y_min -= margin
    y_max += margin

    fig, ax = plt.subplots(figsize=(8, 8), dpi=int(dpi))

    def _draw_one(traj_4: np.ndarray, color: str, step_stride_i: int) -> None:
        P = int(traj_4.shape[0])
        F = int(traj_4.shape[1])
        stride = max(1, int(step_stride_i))

        # 중심 궤적(라인)도 같이(가독성)
        for p in range(P):
            m = valid_f[p]  # (F,)
            if not bool(np.any(m)):
                continue
            xy = traj_4[p, :, 0:2]  # (F,2)
            xy_v = xy[m]
            if xy_v.shape[0] >= 2:
                ax.plot(xy_v[:, 0], xy_v[:, 1], color=color, alpha=0.35, linewidth=1.0)

        # 사각형(방향 포함)
        for p in range(P):
            width = float(wl[p, 0])
            length = float(wl[p, 1])
            if not (width > 0.0 and length > 0.0):
                continue

            for t in range(0, F, stride):
                if not bool(valid_f[p, t]):
                    continue

                x = float(traj_4[p, t, 0])
                y = float(traj_4[p, t, 1])
                cy = float(traj_4[p, t, 2])
                sy = float(traj_4[p, t, 3])

                poly, head = _rect_poly_and_heading_xy(
                    center_xy=np.array([x, y], dtype=np.float32),
                    cos_yaw=cy,
                    sin_yaw=sy,
                    width=width,
                    length=length,
                )  # poly:(5,2), head:(2,)

                ax.plot(poly[:, 0], poly[:, 1], color=color, alpha=0.18, linewidth=0.8)
                ax.plot([x, float(head[0])], [y, float(head[1])], color=color, alpha=0.25, linewidth=0.8)

    # GT(파란색) vs integrated(빨간색)
    _draw_one(gt, color="tab:blue", step_stride_i=step_stride)
    _draw_one(integ, color="tab:red", step_stride_i=step_stride)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"rect traj compare | scenario={scenario_id}")

    # 범례(간단히 2개만)
    legend_lines = [
        Line2D([0], [0], color="tab:blue", lw=2, label="GT (future_gt_4_dim)"),
        Line2D([0], [0], color="tab:red", lw=2, label="Integrated (filter_and_integrate)"),
    ]
    ax.legend(handles=legend_lines, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def build_target_future_gt_4_dim_from_future_11_dim(
    ego_future_gt_11_dim: np.ndarray,          # shape: (F, 11)
    neighbor_future_gt_11_dim: np.ndarray,     # shape: (A, F, 11)
) -> np.ndarray:
    """ego/neighbor의 미래 11차원에서 [x,y,cos,sin]만 뽑아 (1+A,F,4) GT를 만듭니다.

    Args:
        ego_future_gt_11_dim:
            shape: (F,11). ego의 미래 GT(현재 제외).
        neighbor_future_gt_11_dim:
            shape: (A,F,11). neighbor들의 미래 GT.

    Returns:
        target_future_gt_4_dim:
            shape: (1+A, F, 4), dtype float32
            - 첫 번째 행은 ego
            - 그 뒤는 neighbor 순서
            - 마지막 차원 4는 [x, y, cos(yaw), sin(yaw)]
    """
    ego_f = np.asarray(ego_future_gt_11_dim, dtype=np.float32)
    neigh_f = np.asarray(neighbor_future_gt_11_dim, dtype=np.float32)

    if ego_f.ndim != 2 or int(ego_f.shape[1]) < 4:
        raise ValueError(f"ego_future_gt_11_dim must be (F,>=4). got={ego_f.shape}")
    if neigh_f.ndim != 3 or int(neigh_f.shape[2]) < 4:
        raise ValueError(f"neighbor_future_gt_11_dim must be (A,F,>=4). got={neigh_f.shape}")

    F = int(ego_f.shape[0])
    A = int(neigh_f.shape[0])
    if A > 0 and int(neigh_f.shape[1]) != F:
        raise ValueError(
            "future_len mismatch between ego and neighbor. "
            f"ego F={F}, neighbor.shape={neigh_f.shape}"
        )

    ego_4 = ego_f[:, 0:4].astype(np.float32)            # (F,4)
    if A > 0:
        neigh_4 = neigh_f[:, :, 0:4].astype(np.float32) # (A,F,4)
        out = np.concatenate([ego_4[None, ...], neigh_4], axis=0).astype(np.float32)  # (1+A,F,4)
    else:
        out = ego_4[None, ...].astype(np.float32)  # (1,F,4)

    return out


def compute_neighbor_xy_yaw_losses_for_integrated_trajectory(
    target_integrated_trajectory: np.ndarray,  # shape: (1+A, F, 4)
    target_future_gt_4_dim: np.ndarray,        # shape: (1+A, F, 4)
    target_cur_future_valid: np.ndarray,       # shape: (1+A, 1+F)
    *,
    exclude_ego: bool = True,
) -> Tuple[float, float, int]:
    """integrated trajectory와 GT future 간의 xy/yaw 오차를 `_compute_xy_yaw_losses`로 계산합니다.

    주의:
    - `_compute_xy_yaw_losses`는 (B,P,T,4) 형태를 기대하므로, 여기서는 B=1을 붙여 호출합니다.
    - “neighbor만” 보려면 ego(첫 행)는 valid를 False로 꺼서 제외합니다.

    Args:
        target_integrated_trajectory:
            shape: (1+A, F, 4). filter_and_integrate 결과.
        target_future_gt_4_dim:
            shape: (1+A, F, 4). GT future [x,y,cos,sin].
        target_cur_future_valid:
            shape: (1+A, 1+F). 현재+미래 유효 마스크.
        exclude_ego:
            True면 ego(0번 행)를 valid에서 제외해서 “neighbor loss”만 계산합니다.

    Returns:
        xy_loss:
            float. dict 키 "neighbor_prediction_loss_xy" 값.
        yaw_loss:
            float. dict 키 "neighbor_prediction_loss_yaw" 값.
        num_valid:
            int. (ego 제외 후) 유효한 (agent,time) 개수.
            0이면 loss는 의미가 없어서 보통 누적에서 스킵합니다.
    """
    integ = np.asarray(target_integrated_trajectory, dtype=np.float32)
    gt = np.asarray(target_future_gt_4_dim, dtype=np.float32)
    valid_cf = np.asarray(target_cur_future_valid, dtype=bool)

    if integ.ndim != 3 or int(integ.shape[2]) != 4:
        raise ValueError(f"target_integrated_trajectory must be (Pnn,F,4). got={integ.shape}")
    if gt.shape != integ.shape:
        raise ValueError(
            "target_future_gt_4_dim shape mismatch. "
            f"gt={gt.shape}, integ={integ.shape}"
        )
    if valid_cf.ndim != 2 or int(valid_cf.shape[0]) != int(integ.shape[0]):
        raise ValueError(
            "target_cur_future_valid must be (Pnn,1+F). "
            f"got={valid_cf.shape}, Pnn={int(integ.shape[0])}"
        )

    Pnn = int(integ.shape[0])
    F = int(integ.shape[1])
    if int(valid_cf.shape[1]) != int(1 + F):
        raise ValueError(
            "target_cur_future_valid length mismatch. "
            f"valid_cf.shape={valid_cf.shape}, expected second dim={1+F}"
        )

    # 미래 노드 유효 마스크: (Pnn,F)
    valid_future = valid_cf[:, 1:].copy()
    if bool(exclude_ego) and Pnn > 0:
        valid_future[0, :] = False

    num_valid = int(np.count_nonzero(valid_future))
    if num_valid <= 0:
        return 0.0, 0.0, 0

    # torch 입력(B=1)
    score_t = torch.from_numpy(np.ascontiguousarray(integ[None, ...])).float()  # (1,Pnn,F,4)
    gt_t = torch.from_numpy(np.ascontiguousarray(gt[None, ...])).float()        # (1,Pnn,F,4)
    valid_t = torch.from_numpy(np.ascontiguousarray(valid_future[None, ...])).to(torch.bool)  # (1,Pnn,F)

    with torch.no_grad():
        losses = _compute_xy_yaw_losses(
            score_denorm=score_t,
            target_future_gt=gt_t,
            target_future_valid=valid_t,
            prefix="neighbor_prediction_loss",
        )

    xy_loss = float(losses["neighbor_prediction_loss_xy"].item())
    yaw_loss = float(losses["neighbor_prediction_loss_yaw"].item())
    return xy_loss, yaw_loss, num_valid


def _reset_worker_local_neighbor_xy_yaw_sums() -> None:
    """워커 프로세스에서, 현재 파일(tfrecord 1개) 처리용 로컬 누적치를 초기화합니다."""
    global _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS
    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS = NeighborXYYawLossRunningSums()


def _update_worker_local_neighbor_xy_yaw_sums(
    xy_loss: float,
    yaw_loss: float,
    num_valid: int,
) -> None:
    """워커 로컬 누적치에 (시나리오 1개) loss를 더합니다.

    Args:
        xy_loss: 시나리오 1개에 대한 neighbor xy loss. shape: ()
        yaw_loss: 시나리오 1개에 대한 neighbor yaw loss. shape: ()
        num_valid: 유효한 (agent,time) 개수. 0이면 스킵합니다. shape: ()
    """
    global _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS
    if _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS is None:
        _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS = NeighborXYYawLossRunningSums()

    if int(num_valid) <= 0:
        _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.skipped_no_valid += 1
        return

    x = float(xy_loss)
    y = float(yaw_loss)

    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.count += 1
    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.sum_xy += x
    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.sum_sq_xy += (x * x)
    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.sum_yaw += y
    _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS.sum_sq_yaw += (y * y)


def _flush_worker_local_neighbor_xy_yaw_sums_to_shared() -> None:
    """워커 로컬 누적치를 공유 누적치에 “한 번에” 합산합니다(파일 1개 끝날 때 1회)."""
    global _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS, _NEIGHBOR_XY_YAW_SUMS_PROXY, _NEIGHBOR_XY_YAW_SUMS_LOCK
    if _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS is None:
        return
    if _NEIGHBOR_XY_YAW_SUMS_PROXY is None or _NEIGHBOR_XY_YAW_SUMS_LOCK is None:
        return

    local = _WORKER_LOCAL_NEIGHBOR_XY_YAW_SUMS
    with _NEIGHBOR_XY_YAW_SUMS_LOCK:
        d = _NEIGHBOR_XY_YAW_SUMS_PROXY

        d["count"] = int(d.get("count", 0)) + int(local.count)
        d["sum_xy"] = float(d.get("sum_xy", 0.0)) + float(local.sum_xy)
        d["sum_sq_xy"] = float(d.get("sum_sq_xy", 0.0)) + float(local.sum_sq_xy)
        d["sum_yaw"] = float(d.get("sum_yaw", 0.0)) + float(local.sum_yaw)
        d["sum_sq_yaw"] = float(d.get("sum_sq_yaw", 0.0)) + float(local.sum_sq_yaw)
        d["skipped_no_valid"] = int(d.get("skipped_no_valid", 0)) + int(local.skipped_no_valid)


def _mean_and_variance_from_sums(
    count: int,
    sum_val: float,
    sum_sq_val: float,
) -> Tuple[float, float]:
    """count/sum/sum_sq로 평균과 분산(모집단 분산)을 계산합니다."""
    c = int(count)
    if c <= 0:
        return 0.0, 0.0
    mean = float(sum_val) / float(c)
    var = float(sum_sq_val) / float(c) - mean * mean
    if var < 0.0:
        var = 0.0
    return mean, var


def build_target_class_one_hot_from_current_feat11_for_cache(
    ego_agent_past: np.ndarray,          # shape: (TIME_LEN, 11)
    neighbor_agents_past: np.ndarray,    # shape: (A, TIME_LEN, 11)
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    """ego/neighbor의 '현재 시점' 타입 벡터(세 칸)를 (1+A,3)으로 만듭니다.

    이 프로젝트의 11차원 agent 특징은 아래 순서를 가집니다.
      [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]
    따라서 타입(one_hot 3칸)은 항상 feature 인덱스 8:11에 들어 있습니다.

    Args:
        ego_agent_past:
            shape: (TIME_LEN, 11)
            마지막 시점(-1)이 현재입니다.
        neighbor_agents_past:
            shape: (A, TIME_LEN, 11)
            마지막 시점(-1)이 현재입니다.
        eps:
            합이 0이 아닌 경우에만 “합이 1이 되도록” 아주 작은 오차를 정리할 때 쓰는 값입니다.

    Returns:
        target_class_one_hot:
            shape: (1+A, 3), dtype float32
            첫 행은 ego, 그 뒤는 neighbor 순서입니다.
    """
    ego_past = np.asarray(ego_agent_past, dtype=np.float32)
    neigh_past = np.asarray(neighbor_agents_past, dtype=np.float32)

    if ego_past.ndim != 2 or int(ego_past.shape[1]) != 11:
        raise ValueError(f"ego_agent_past must be (TIME_LEN,11). got={ego_past.shape}")
    if neigh_past.ndim != 3 or int(neigh_past.shape[2]) != 11:
        raise ValueError(f"neighbor_agents_past must be (A,TIME_LEN,11). got={neigh_past.shape}")

    # ego 현재 타입: (1,3)
    ego_oh = ego_past[-1, 8:11].reshape(1, 3).astype(np.float32)

    # neighbor 현재 타입: (A,3)
    A = int(neigh_past.shape[0])
    if A > 0:
        neigh_oh = neigh_past[:, -1, 8:11].astype(np.float32)
        out = np.concatenate([ego_oh, neigh_oh], axis=0).astype(np.float32)  # (1+A,3)
    else:
        out = ego_oh.astype(np.float32)  # (1,3)

    # (선택) 혹시 합이 1이 아닌 미세 오차가 있으면 정리
    s = np.sum(out, axis=1, keepdims=True).astype(np.float32)  # (1+A,1)
    good = s > float(eps)
    out[good[:, 0]] = out[good[:, 0]] / np.maximum(s[good[:, 0]], float(eps))

    return out.astype(np.float32)


def compute_target_integrated_trajectory_and_constraint_diff_for_cache(
    target_past_future_body_seg_control: np.ndarray,  # shape: (1+A, past_len+future_len, 3)
    target_current_state: np.ndarray,                 # shape: (1+A, 4)
    target_cur_future_valid: np.ndarray,              # shape: (1+A, 1+future_len)
    target_class_one_hot: np.ndarray,                 # shape: (1+A, 3)
    *,
    dt_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """FeasibleProjector.filter_and_integrate를 사용해 GT 적분 궤적과 제약 보정량을 만듭니다.

    Args:
        target_past_future_body_seg_control:
            shape: (1+A, past_len+future_len, 3)
            [v_x^b, v_y^b, yaw_rate] (구간 중점 제어) 입니다.
            여기서 past_len은 (TIME_LEN-1), future_len은 FUTURE_LEN 입니다.
        target_current_state:
            shape: (1+A, 4) = [x, y, cos(yaw), sin(yaw)]
        target_cur_future_valid:
            shape: (1+A, 1+future_len) bool
            현재 포함 노드 유효 마스크.
        target_class_one_hot:
            shape: (1+A, 3) float32
        dt_sec:
            샘플 간 시간 간격(초). FeasibleProjector 내부 dt와 맞춰야 합니다.

    Returns:
        target_integrated_trajectory:
            shape: (1+A, future_len, 4) float32
            현재에서 시작해 future_len 스텝 적분한 노드 상태 [x,y,cos,sin] 입니다.
        target_control_constraint_diff:
            shape: (1+A, future_len, 3) float32
            (제약 적용 후 제어 - 입력 제어) 입니다.
            제약이 꺼져 있으면 거의 0이 됩니다.
    """
    seg_all = np.asarray(target_past_future_body_seg_control, dtype=np.float32)
    cur_state = np.asarray(target_current_state, dtype=np.float32)
    cur_fut_valid = np.asarray(target_cur_future_valid, dtype=bool)
    cls_oh = np.asarray(target_class_one_hot, dtype=np.float32)

    if seg_all.ndim != 3 or int(seg_all.shape[2]) != 3:
        raise ValueError(f"target_past_future_body_seg_control must be (Pnn,seg_len,3). got={seg_all.shape}")
    if cur_state.shape != (int(seg_all.shape[0]), 4):
        raise ValueError(
            f"target_current_state must be (Pnn,4). got={cur_state.shape}, Pnn={int(seg_all.shape[0])}"
        )
    if cls_oh.shape != (int(seg_all.shape[0]), 3):
        raise ValueError(
            f"target_class_one_hot must be (Pnn,3). got={cls_oh.shape}, Pnn={int(seg_all.shape[0])}"
        )

    Pnn = int(seg_all.shape[0])
    future_len = int(cur_fut_valid.shape[1]) - 1
    if future_len <= 0:
        raise ValueError(f"future_len must be > 0. got future_len={future_len}")

    if cur_fut_valid.shape[0] != Pnn:
        raise ValueError(
            f"target_cur_future_valid must be (Pnn,1+future_len). got={cur_fut_valid.shape}, Pnn={Pnn}"
        )

    seg_len_all = int(seg_all.shape[1])  # = past_len + future_len
    past_len = int(seg_len_all - future_len)
    if past_len < 0:
        raise ValueError(
            "segment length mismatch. "
            f"seg_len_all={seg_len_all}, future_len={future_len} -> past_len={past_len}"
        )

    # 현재->미래 구간 제어만 사용: (Pnn, future_len, 3)
    cur_future_seg_body_control = seg_all[:, past_len:, :].astype(np.float32)
    if cur_future_seg_body_control.shape != (Pnn, future_len, 3):
        raise ValueError(
            "cur_future_seg_body_control shape mismatch. "
            f"got={cur_future_seg_body_control.shape}, expected={(Pnn, future_len, 3)}"
        )

    # filter_and_integrate 입력: 배치 차원(B=1) 추가
    unnorm_near_current_state_t = torch.from_numpy(
        np.ascontiguousarray(cur_state[None, ...])
    ).float()  # (1, Pnn, 4)

    near_cur_future_valid_t = torch.from_numpy(
        np.ascontiguousarray(cur_fut_valid[None, ...])
    ).to(torch.bool)  # (1, Pnn, 1+future_len)

    unnorm_cur_future_seg_body_control_t = torch.from_numpy(
        np.ascontiguousarray(cur_future_seg_body_control[None, ...])
    ).float()  # (1, Pnn, future_len, 3)

    near_class_one_hot_t = torch.from_numpy(
        np.ascontiguousarray(cls_oh[None, ...])
    ).float()  # (1, Pnn, 3)

    fp = _get_feasible_projector_for_integration_cache()
    # dt 동기화(혹시 DT가 바뀌는 경우 대비)
    with contextlib.suppress(Exception):
        fp.constraints_h_params.dt = float(dt_sec)

    with torch.no_grad():
        traj_t, diff_t = fp.filter_and_integrate(
            unnorm_near_current_state=unnorm_near_current_state_t,
            near_cur_future_valid=near_cur_future_valid_t,
            unnorm_cur_future_seg_body_control=unnorm_cur_future_seg_body_control_t,
            near_class_one_hot=near_class_one_hot_t,
        )  # traj:(1,Pnn,future_len,4), diff:(1,Pnn,future_len,3)

    target_integrated_trajectory = traj_t.squeeze(0).cpu().numpy().astype(np.float32)  # (Pnn,future_len,4)
    target_control_constraint_diff = diff_t.squeeze(0).cpu().numpy().astype(np.float32)  # (Pnn,future_len,3)

    return target_integrated_trajectory, target_control_constraint_diff


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

from typing import Union

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

def _build_seg_control_gt_and_seg_valid_from_all11(
    ego_all11: ArrayF,  # (T,11)
    neighbor_all11: ArrayF,  # (N,T,11)
    *,
    current_index: int,
    dt: float,
    eps: float = 1e-8,
) -> Tuple[ArrayF, NDArray[np.bool_]]:
    """11차원 궤적에서 구간 제어와 구간 유효 마스크를 계산합니다.

    동작 요약:
        - (x,y,cos,sin,...) 값이 모두 0에 가깝다면 그 프레임은 무효로 봅니다.
        - 두 프레임이 연속으로 유효일 때만 그 사이 "구간"을 유효로 봅니다(seg_valid=True).
        - current_index 위치의 프레임이 무효라면(현재가 무효),
          해당 에이전트의 전체 궤적을 0으로 만들어 결과도 전부 0이 되게 합니다.

    Args:
        ego_all11 (np.ndarray): ego 궤적. shape: (T, 11)
        neighbor_all11 (np.ndarray): neighbor 궤적. shape: (N, T, 11)
        current_index (int): "현재 프레임"이 들어있는 인덱스
        dt (float): 시간 간격
        eps (float): 0 판정 기준

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            controls: shape (1+N, T-1, 3)
                - 마지막 3은 (v_x^b, v_y^b, yaw_rate)
            seg_valid: shape (1+N, T-1) bool
                - 구간이 유효하면 True
    """
    ego11 = np.asarray(ego_all11)
    nbr11 = np.asarray(neighbor_all11)

    if ego11.ndim != 2 or int(ego11.shape[-1]) != 11:
        raise ValueError(f"ego_all11 shape는 (T,11) 이어야 합니다. got {ego11.shape}")
    if nbr11.ndim != 3 or int(nbr11.shape[-1]) != 11:
        raise ValueError(f"neighbor_all11 shape는 (N,T,11) 이어야 합니다. got {nbr11.shape}")
    if int(nbr11.shape[1]) != int(ego11.shape[0]):
        raise ValueError(f"T 차원이 일치해야 합니다. got ego T={ego11.shape[0]} vs nbr T={nbr11.shape[1]}")

    T = int(ego11.shape[0])
    if T <= 1:
        raise ValueError(f"T는 최소 2 이상이어야 합니다. got T={T}")

    dt = float(dt)
    if (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt}")

    cur_idx = int(current_index)
    if cur_idx < 0:
        cur_idx += T
    if not (0 <= cur_idx < T):
        raise ValueError(f"current_index 범위가 잘못되었습니다. got {current_index}, T={T}")

    # float dtype 강제(삼각함수/나눗셈 안정)
    ego11 = ego11.astype(np.float32 if ego11.dtype.kind != "f" else ego11.dtype, copy=False)
    nbr11 = nbr11.astype(ego11.dtype, copy=False)

    # (안전) "현재"가 무효면 그 에이전트 전체를 0으로
    ego_cur_valid = bool((np.abs(ego11[cur_idx, :8]) > eps).any())
    if not ego_cur_valid:
        ego11 = np.zeros_like(ego11)

    N = int(nbr11.shape[0])
    if N > 0:
        nbr_cur_valid_mask = (np.abs(nbr11[:, cur_idx, :8]) > eps).any(axis=1)  # (N,)
        if not np.all(nbr_cur_valid_mask):
            nbr11 = np.array(nbr11, copy=True)
            nbr11[~nbr_cur_valid_mask, :, :] = 0.0

    # 11D -> pose3(x,y,heading)
    ego_pose3 = _traj11_to_traj3_heading(ego11)  # (T,3)
    nbr_pose3 = _traj11_to_traj3_heading(nbr11)  # (N,T,3)
    all_pose3 = np.concatenate([ego_pose3[None, ...], nbr_pose3], axis=0).astype(np.float32, copy=False)
    # all_pose3: (1+N, T, 3)

    # frame valid -> seg valid
    ego_valid = (np.abs(ego11[:, :8]) > eps).any(axis=1)            # (T,)
    nbr_valid = (np.abs(nbr11[:, :, :8]) > eps).any(axis=2)         # (N,T)
    all_valid = np.concatenate([ego_valid[None, :], nbr_valid], axis=0).astype(bool)  # (1+N,T)
    seg_valid = (all_valid[:, :-1] & all_valid[:, 1:]).astype(bool)  # (1+N,T-1)

    # controls
    controls = differentiate_numpy_pose3_to_control3(all_pose3, dt=dt).astype(np.float32, copy=False)
    # controls: (1+N, T-1, 3)

    return controls, seg_valid

def _build_past_seg_control_gt_and_seg_valid_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> Tuple[ArrayF, NDArray[np.bool_]]:
    """past 구간 제어(past_len=Tp-1)와 seg_valid를 계산합니다."""
    ego_past = np.asarray(ego_agent_past)
    nbr_past = np.asarray(neighbor_agents_past)

    if ego_past.ndim != 2 or ego_past.shape[-1] != 11:
        raise ValueError(f"ego_agent_past shape는 (Tp,11)이어야 합니다. got {ego_past.shape}")
    if nbr_past.ndim != 3 or nbr_past.shape[-1] != 11:
        raise ValueError(f"neighbor_agents_past shape는 (N,Tp,11)이어야 합니다. got {nbr_past.shape}")
    if int(nbr_past.shape[1]) != int(ego_past.shape[0]):
        raise ValueError(f"Tp 차원이 일치해야 합니다. got ego Tp={ego_past.shape[0]} vs nbr Tp={nbr_past.shape[1]}")

    Tp = int(ego_past.shape[0])
    return _build_seg_control_gt_and_seg_valid_from_all11(
        ego_all11=ego_past,                 # (Tp,11)
        neighbor_all11=nbr_past,            # (N,Tp,11)
        current_index=Tp - 1,               # past에서 현재는 마지막 프레임
        dt=float(dt),
        eps=float(eps),
    )


def build_past_seg_control_gt_3_dim_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> ArrayF:
    """npz 내부의 past 11차원 궤적으로부터 past 구간 제어를 만듭니다.

    Returns:
        np.ndarray:
            past_seg_control_gt_3_dim, shape (1+N, Tp-1, 3)
            - 마지막 3: (v_x^b, v_y^b, yaw_rate)
            - 무효 구간은 0.0
    """
    controls, seg_valid = _build_past_seg_control_gt_and_seg_valid_from_npz_arrays(
        ego_agent_past=ego_agent_past,
        neighbor_agents_past=neighbor_agents_past,
        dt=float(dt),
        eps=float(eps),
    )
    controls[~seg_valid] = 0.0
    return controls


def _build_future_seg_control_gt_and_seg_valid_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)  현재 프레임을 얻기 위해 필요
    ego_future_gt_11_dim: ArrayF,  # (Tf,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11) 현재 프레임을 얻기 위해 필요
    neighbor_future_gt_11_dim: ArrayF,  # (N,Tf,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> Tuple[ArrayF, NDArray[np.bool_]]:
    """future 구간 제어(future_len=Tf)와 seg_valid를 계산합니다.

    future 구간은 "현재 -> 첫 미래"부터 시작하므로,
    (현재 1프레임 + 미래 Tf프레임) = (1+Tf) 상태를 만든 뒤,
    그 사이 구간 Tf개에 대한 제어를 만듭니다.
    """
    ego_past = np.asarray(ego_agent_past)
    ego_fut = np.asarray(ego_future_gt_11_dim)
    nbr_past = np.asarray(neighbor_agents_past)
    nbr_fut = np.asarray(neighbor_future_gt_11_dim)

    if ego_past.ndim != 2 or ego_past.shape[-1] != 11:
        raise ValueError(f"ego_agent_past shape는 (Tp,11)이어야 합니다. got {ego_past.shape}")
    if ego_fut.ndim != 2 or ego_fut.shape[-1] != 11:
        raise ValueError(f"ego_future_gt_11_dim shape는 (Tf,11)이어야 합니다. got {ego_fut.shape}")
    if nbr_past.ndim != 3 or nbr_past.shape[-1] != 11:
        raise ValueError(f"neighbor_agents_past shape는 (N,Tp,11)이어야 합니다. got {nbr_past.shape}")
    if nbr_fut.ndim != 3 or nbr_fut.shape[-1] != 11:
        raise ValueError(f"neighbor_future_gt_11_dim shape는 (N,Tf,11)이어야 합니다. got {nbr_fut.shape}")

    Tp = int(ego_past.shape[0])
    Tf = int(ego_fut.shape[0])
    N = int(nbr_past.shape[0])

    if Tp <= 0:
        raise ValueError(f"Tp는 1 이상이어야 합니다. got Tp={Tp}")
    if int(nbr_past.shape[1]) != Tp:
        raise ValueError(f"neighbor_agents_past의 Tp가 ego와 같아야 합니다. got {nbr_past.shape[1]} vs {Tp}")
    if int(nbr_fut.shape[0]) != N:
        raise ValueError(f"neighbor_future_gt_11_dim의 N이 neighbor_agents_past와 같아야 합니다. got {nbr_fut.shape[0]} vs {N}")
    if int(nbr_fut.shape[1]) != Tf:
        raise ValueError(f"neighbor_future_gt_11_dim의 Tf가 ego_future와 같아야 합니다. got {nbr_fut.shape[1]} vs {Tf}")

    # (현재 1프레임 + 미래 Tf프레임) 만들기
    ego_cur = ego_past[-1:, :]  # (1,11)
    ego_all11 = np.concatenate([ego_cur, ego_fut], axis=0)  # (1+Tf,11)

    if N > 0:
        nbr_cur = nbr_past[:, -1:, :]  # (N,1,11)
        nbr_all11 = np.concatenate([nbr_cur, nbr_fut], axis=1)  # (N,1+Tf,11)
    else:
        nbr_all11 = np.zeros((0, 1 + Tf, 11), dtype=ego_all11.dtype)

    return _build_seg_control_gt_and_seg_valid_from_all11(
        ego_all11=ego_all11,              # (1+Tf,11)
        neighbor_all11=nbr_all11,         # (N,1+Tf,11)
        current_index=0,                  # future에서 현재는 첫 프레임
        dt=float(dt),
        eps=float(eps),
    )

def build_future_seg_control_gt_3_dim_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)
    ego_future_gt_11_dim: ArrayF,  # (Tf,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    neighbor_future_gt_11_dim: ArrayF,  # (N,Tf,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> ArrayF:
    """npz 내부의 future 11차원 궤적으로부터 future 구간 제어를 만듭니다.

    Returns:
        np.ndarray:
            future_seg_control_gt_3_dim, shape (1+N, Tf, 3)
            - 마지막 3: (v_x^b, v_y^b, yaw_rate)
            - 무효 구간은 0.0
    """
    controls, seg_valid = _build_future_seg_control_gt_and_seg_valid_from_npz_arrays(
        ego_agent_past=ego_agent_past,
        ego_future_gt_11_dim=ego_future_gt_11_dim,
        neighbor_agents_past=neighbor_agents_past,
        neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,
        dt=float(dt),
        eps=float(eps),
    )
    controls[~seg_valid] = 0.0
    return controls

# =========================
# 시나리오 -> pkl dict 만들기
# =========================
def build_cache_dict_for_scenario(
    scenario: scenario_pb2.Scenario,
    *,
    traj_rect_compare_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Scenario 1개를 요구사항 포맷의 dict로 바꿉니다.

    이 dict가 그대로 pickle로 저장됩니다.

    Returns dict keys(요구사항):
        - scenario_id: str
        - ego_agent_past: (21,11) float32
        - ego_future_gt_3_dim: (80,3) float32
        - ego_future_gt_11_dim: (80,11) float32

        - neighbor_role: (A,2) bool
        - target_id: (A+1,) int64
        - target_z: (A+1,) float32
        - neighbor_shape: (A,3) float32  [length,width,height] 평균
        - neighbor_agents_past: (A,21,11) float32
        - neighbor_future_gt_3_dim: (A,80,3) float32
        - neighbor_future_gt_11_dim: (A,80,11) float32
        - neighbor_track_token: List[str] (길이 A)

        - stop_sign_points: (Ns,10,2) float32
        - crosswalk_points: (Nc,10,2) float32
        - speed_bump_points: (Nb,10,2) float32

        - lanes: (L,10,12) float32
        - lanes_speed_limit: (L,1) float32
        - lanes_has_speed_limit: (L,1) bool
        - lane_light: (L,4) float32

        lane_type: (L,4) float32

        left_line_type: (L,13) float32

        right_line_type: (L,13) float32

        road_edge: (E,10,2) float32

        road_edge_type: (E,3) float32

        driveway_points: (D,10,2) float32

    Args:
        scenario: Scenario proto

    Returns:
        cache_dict: pickle로 저장할 dict
    """
    _require_womd_lengths_initialized()
    track_key_to_array = decode_tracks_and_roles_from_scenario(scenario)

    object_id_all = track_key_to_array["object_id"]  # (N,)
    object_type_all = track_key_to_array["object_type"]  # (N,)
    states_all = track_key_to_array["states"]  # (N,S,9)
    valid_all = track_key_to_array["valid"]  # (N,S)
    role_interest_all = track_key_to_array["role_interest"]  # (N,)
    role_predict_all = track_key_to_array["role_predict"]  # (N,)

    ego_idx = int(track_key_to_array["ego_index"][0])
    current_t = int(scenario.current_time_index)
    num_tracks = int(states_all.shape[0])
    num_steps = int(states_all.shape[1])

    ego_xy_global, ego_yaw_global, ego_z_global = compute_ego_pose_at_current(
        scenario, track_key_to_array)
    origin_world_pose = build_origin_world_pose(
        ego_xy_global=ego_xy_global,  # shape: (2,)
        ego_yaw_global=ego_yaw_global,
    )  # shape: (4,)

    # -------------------------
    # 모든 트랙을 ego 좌표계로 미리 변환
    # -------------------------
    pos_xy_global_all = states_all[:, :, 0:2]  # shape (N,S,2)
    vel_xy_global_all = states_all[:, :, 7:9]  # shape (N,S,2)
    yaw_global_all = states_all[:, :, 6]  # shape (N,S)
    z_global_all = states_all[:, :, 2]  # shape (N,S)

    # local 변환
    pos_xy_local_all = transform_points_global_to_ego_local(
        pos_xy_global_all, ego_xy_global, ego_yaw_global)  # (N,S,2)
    vel_xy_local_all = transform_vectors_global_to_ego_local(
        vel_xy_global_all, ego_yaw_global)  # (N,S,2)

    # heading은 ego_yaw 기준으로 상대값
    yaw_local_all = wrap_angle_np(yaw_global_all - ego_yaw_global)  # (N,S)

    # width/length: 요구사항 11차원에서 (width, length) 순서
    width_all = states_all[:, :, 4]  # (N,S)
    length_all = states_all[:, :, 3]  # (N,S)
    width_length_all = np.stack([width_all, length_all],
                                axis=-1).astype(np.float32)  # (N,S,2)

    # z_rel_all: ego 기준 상대 z (필요하면 다른 곳에서 사용 가능)
    z_rel_all = (z_global_all - ego_z_global).astype(np.float32)  # (N,S)

    # one-hot 타입 미리 생성
    one_hot_all = build_agent_type_one_hot_all(
        object_type_minus1_all=object_type_all,  # shape: (N,)
        ego_track_index=ego_idx,
    )
    # ego는 무조건 VEHICLE로 고정
    one_hot_all[ego_idx] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # -------------------------
    # time index 구성
    # -------------------------
    past_indices, future_indices = build_time_indices(
        current_time_index=current_t,
        num_total_steps=num_steps,
        desired_past_len=TIME_LEN,
        desired_future_len=FUTURE_LEN,
    )

    # -------------------------
    # ego: past/future 만들기
    # -------------------------
    # -------------------------
    # ego: past/future 만들기
    # (과거~현재~미래를 한 번에 이어 붙인 뒤, "유효-무효-유효" 끊김이 없도록 정리)
    # -------------------------
    ego_agent_past, ego_future_gt_3_dim, ego_future_gt_11_dim = build_agent_past_future_cache_arrays_from_full_trajectory(
        agent_idx=ego_idx,
        past_indices=past_indices,          # (TIME_LEN,)
        future_indices=future_indices,      # (FUTURE_LEN,)
        pos_xy_local_all=pos_xy_local_all,
        vel_xy_local_all=vel_xy_local_all,
        yaw_local_all=yaw_local_all,
        width_length_all=width_length_all,
        valid_all=valid_all,
        agent_one_hot=one_hot_all[ego_idx],  # (3,)
        enforce_all_invalid_when_current_invalid=False,  # ego는 현재가 무조건 유효(위에서 검증)
    )

    # -------------------------
    # neighbors: 현재 시점에 존재하는(agent valid at current)만
    # -------------------------
    neighbor_indices = collect_neighbor_indices_at_current(
        valid_all=valid_all,  # shape: (N,S)
        current_time_index=current_t,
        ego_track_index=ego_idx,
    )
    agent_num = int(neighbor_indices.shape[0])

    neighbor_role = np.zeros((agent_num, 2),
                             dtype=bool)  # (A,2) [interest, predict]
    neighbor_id = np.zeros((agent_num,), dtype=np.int64)  # (A,)
    neighbor_z = np.zeros((agent_num,), dtype=np.float32)  # (A,)
    neighbor_shape = np.zeros((agent_num, 3),
                              dtype=np.float32)  # (A,3) [length,width,height]
    neighbor_agents_past = np.zeros((agent_num, TIME_LEN, 11),
                                    dtype=np.float32)  # (A,21,11)
    neighbor_future_gt_3_dim = np.zeros((agent_num, FUTURE_LEN, 3),
                                        dtype=np.float32)  # (A,80,3)

    # ✅ 추가: neighbor 미래 11차원도 저장
    neighbor_future_gt_11_dim = np.zeros((agent_num, FUTURE_LEN, 11),
                                         dtype=np.float32)  # (A,80,11)
    neighbor_track_token: List[str] = [""] * agent_num  # ✅ 추가: 길이 A

    for out_i, tr_i in enumerate(neighbor_indices.tolist()):
        tr_i = int(tr_i)
        neighbor_track_token[out_i] = get_womd_track_token(
            scenario.tracks[tr_i])  # ✅ 추가

        neighbor_id[out_i] = object_id_all[tr_i]
        neighbor_role[out_i, 0] = bool(role_interest_all[tr_i])
        neighbor_role[out_i, 1] = bool(role_predict_all[tr_i])
        neighbor_z[out_i] = float(z_rel_all[tr_i, current_t])

        # shape 평균: states[...,3:6] = [length,width,height]
        valid_mask_full = valid_all[tr_i]  # (S,)
        if np.any(valid_mask_full):
            shape_vals = states_all[tr_i, valid_mask_full, 3:6]  # (K,3)
            neighbor_shape[out_i] = shape_vals.mean(axis=0).astype(np.float32)
        else:
            neighbor_shape[out_i] = 0.0
        # past/future: (과거~현재~미래를 한 번에 이어 붙인 뒤 규칙 적용)
        n_past_11, n_future_3, n_future_11 = build_agent_past_future_cache_arrays_from_full_trajectory(
            agent_idx=tr_i,
            past_indices=past_indices,            # (TIME_LEN,)
            future_indices=future_indices,        # (FUTURE_LEN,)
            pos_xy_local_all=pos_xy_local_all,
            vel_xy_local_all=vel_xy_local_all,
            yaw_local_all=yaw_local_all,
            width_length_all=width_length_all,
            valid_all=valid_all,
            agent_one_hot=one_hot_all[tr_i],      # (3,)
            enforce_all_invalid_when_current_invalid=False,  # ✅ 요구사항 a: 현재 무효면 101개 전부 무효
        )

        neighbor_agents_past[out_i] = n_past_11
        neighbor_future_gt_3_dim[out_i] = n_future_3
        neighbor_future_gt_11_dim[out_i] = n_future_11

    # past_seg_control_gt_3_dim : (1+Pnn, past_len, 3)
    past_seg_control_gt_3_dim = build_past_seg_control_gt_3_dim_from_npz_arrays(
        ego_agent_past=ego_agent_past,
        neighbor_agents_past=neighbor_agents_past,
        dt=float(0.1),
    )
    # future_seg_control_gt_3_dim : (1+Pnn, future_len, 3)
    future_seg_control_gt_3_dim = build_future_seg_control_gt_3_dim_from_npz_arrays(
        ego_agent_past=ego_agent_past,
        ego_future_gt_11_dim=ego_future_gt_11_dim,
        neighbor_agents_past=neighbor_agents_past,
        neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,
        dt=float(0.1),
    )
    # -------------------------
    # ✅ (추가) vxy + cs_yaw 기반 yaw_rate + control을 한 번에 계산(함수화 버전)
    # -------------------------
    """
    ego_past_future_control: (time_len(past_len+1)+future_len,3)
    neighbor_past_future_control: (time_len(past_len+1)+future_len,point_len,3)
    """
    # (
    #     ego_past_future_control,
    #     neighbor_past_future_control,
    # ) = build_past_future_control_for_cache(
    #     ego_agent_past=ego_agent_past,                     # (TIME_LEN,11)
    #     ego_future_gt_11_dim=ego_future_gt_11_dim,         # (FUTURE_LEN,11)
    #     neighbor_agents_past=neighbor_agents_past,         # (A,TIME_LEN,11)
    #     neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,  # (A,FUTURE_LEN,11)
    #     dt_sec=float(DT_SEC),
    #     polyorder=2,
    #     max_window_len_yaw=7,
    #     prefix_dim=8,
    # )
    # (
    #     target_past_future_control,
    #     target_past_future_body_seg_control,
    # ) = build_target_past_future_body_seg_control_for_cache(
    #     ego_agent_past=ego_agent_past,                     # (TIME_LEN,11)
    #     ego_future_gt_11_dim=ego_future_gt_11_dim,         # (FUTURE_LEN,11)
    #     neighbor_agents_past=neighbor_agents_past,         # (A,TIME_LEN,11)
    #     neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,  # (A,FUTURE_LEN,11)
    #     ego_past_future_control=ego_past_future_control,   # (point_len,3)
    #     neighbor_past_future_control=neighbor_past_future_control,  # (A,point_len,3)
    #     prefix_dim=8,
    # )
    # # -------------------------
    # # ✅ (추가) filter_and_integrate 입력들 만들기 + 적분 GT 만들기
    # # -------------------------
    # target_current_state = build_target_current_state_for_cache(
    #     ego_agent_past=ego_agent_past,                 # (TIME_LEN,11)
    #     neighbor_agents_past=neighbor_agents_past,     # (A,TIME_LEN,11)
    # )  # (1+A,4)
    #
    # target_cur_future_valid = build_target_cur_future_valid_for_cache(
    #     ego_agent_past=ego_agent_past,                       # (TIME_LEN,11)
    #     ego_future_gt_11_dim=ego_future_gt_11_dim,           # (FUTURE_LEN,11)
    #     neighbor_agents_past=neighbor_agents_past,           # (A,TIME_LEN,11)
    #     neighbor_future_gt_11_dim=neighbor_future_gt_11_dim, # (A,FUTURE_LEN,11)
    #     prefix_dim=8,
    # )  # (1+A,1+FUTURE_LEN) bool
    #
    # target_class_one_hot = build_target_class_one_hot_from_current_feat11_for_cache(
    #     ego_agent_past=ego_agent_past,               # (TIME_LEN,11)
    #     neighbor_agents_past=neighbor_agents_past,   # (A,TIME_LEN,11)
    # )  # (1+A,3)
    #
    # # filter_and_integrate 실행 -> (1+A,future_len,4), (1+A,future_len,3)
    # target_integrated_trajectory, target_control_constraint_diff = compute_target_integrated_trajectory_and_constraint_diff_for_cache(
    #     target_past_future_body_seg_control=target_past_future_body_seg_control,  # (1+A,past_len+future_len,3)
    #     target_current_state=target_current_state,                                # (1+A,4)
    #     target_cur_future_valid=target_cur_future_valid,                          # (1+A,1+future_len)
    #     target_class_one_hot=target_class_one_hot,                                # (1+A,3)
    #     dt_sec=float(DT_SEC),
    # )
    #
    # # ------------------------------------------------------------
    # # (추가) integrated_trajectory vs GT future(ego+neighbor) 오차를 xy/yaw로 계산하고 누적
    # # ------------------------------------------------------------
    # target_future_gt_4_dim = build_target_future_gt_4_dim_from_future_11_dim(
    #     ego_future_gt_11_dim=ego_future_gt_11_dim,                 # (F,11)
    #     neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,       # (A,F,11)
    # )  # (1+A,F,4)
    #
    # # ------------------------------------------------------------
    # # (추가) 현재 width/length로 사각형 궤적 비교 그림 저장
    # # ------------------------------------------------------------
    # if traj_rect_compare_dir is not None:
    #     target_current_wl = build_target_current_wl_from_current_feat11_for_cache(
    #         ego_agents_past=ego_agent_past,               # (TIME_LEN,11)
    #         neighbor_agents_past=neighbor_agents_past,    # (N,TIME_LEN,11)
    #     )  # (1+N,2)
    #
    #     _ = save_rect_trajectory_comparison_png(
    #         traj_rect_compare_dir,
    #         scenario_id=str(scenario.scenario_id),
    #         target_integrated_trajectory=target_integrated_trajectory,  # (1+N,F,4)
    #         target_future_gt_4_dim=target_future_gt_4_dim,              # (1+N,F,4)
    #         target_current_wl=target_current_wl,                         # (1+N,2)
    #         target_cur_future_valid=target_cur_future_valid,             # (1+N,1+F)
    #         dpi=150,
    #         step_stride=1,
    #     )
    #
    #
    # xy_loss, yaw_loss, num_valid = compute_neighbor_xy_yaw_losses_for_integrated_trajectory(
    #     target_integrated_trajectory=target_integrated_trajectory,  # (1+A,F,4)
    #     target_future_gt_4_dim=target_future_gt_4_dim,              # (1+A,F,4)
    #     target_cur_future_valid=target_cur_future_valid,            # (1+A,1+F)
    #     exclude_ego=False,
    # )
    #
    # _update_worker_local_neighbor_xy_yaw_sums(
    #     xy_loss=xy_loss,
    #     yaw_loss=yaw_loss,
    #     num_valid=num_valid,
    # )


    # -------------------------
    # ✅ ego + neighbor를 한 줄로 묶은 target_id / target_z 만들기
    # -------------------------
    ego_object_id_value = int(object_id_all[ego_idx])

    # ✅ target_z는 "절대 좌표계(전역) z"로 저장
    # ego_z_global은 compute_ego_pose_at_current()에서 얻은 전역 z 입니다.
    ego_z_value_for_target = float(ego_z_global)

    # neighbor들도 현재 시점의 전역 z를 그대로 사용합니다.
    neighbor_z_value_for_target = gather_z_global_at_time(
        z_global_all=z_global_all,          # shape: (N,S)
        track_indices=neighbor_indices,     # shape: (A,)
        time_index=current_t,
    )  # shape: (A,)

    target_id, target_z = build_target_id_and_z_from_ego_and_neighbors(
        ego_object_id=ego_object_id_value,
        ego_z_value=ego_z_value_for_target,
        neighbor_object_id=neighbor_id,                 # shape: (A,)
        neighbor_z_value=neighbor_z_value_for_target,   # shape: (A,)
    )

    # -------------------------
    # map: stop/crosswalk/speed_bump/lanes
    # -------------------------
    parsed_map_all = parse_map_from_scenario(scenario)
    # ✅ use_filter_radius=True일 때만 반경 필터 적용
    if _USE_FILTER_RADIUS:
        parsed_map = filter_parsed_map_by_radius(
            parsed_map=parsed_map_all,
            ego_xy_global=ego_xy_global,
            filter_radius_m=float(_FILTER_RADIUS_M),
        )
    else:
        parsed_map = parsed_map_all

    lane_id_to_light = extract_lane_signals_at_current(scenario)

    stop_sign_points = build_stop_sign_points(
        parsed_map.stop_sign_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    crosswalk_points = build_polygon_points(
        parsed_map.crosswalk_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    speed_bump_points = build_polygon_points(
        parsed_map.speed_bump_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    lanes_arr, lanes_speed_limit, lanes_has_speed_limit, lane_light = build_lane_arrays(
        lanes=parsed_map.lanes,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        lane_id_to_light=lane_id_to_light,
        ego_xy_global=ego_xy_global,
        ego_yaw_global=ego_yaw_global,
        lane_len=LANE_LEN,
    )

    lane_type = build_lane_type_one_hot_array(parsed_map.lanes)

    left_line_type, right_line_type = build_lane_line_type_arrays(
        lanes=parsed_map.lanes,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        road_line_type_by_id=parsed_map.road_line_type_by_id,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
    )

    road_edge, road_edge_type = build_road_edge_points_and_types(
        road_edge_ids=parsed_map.road_edge_ids,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
        ego_xy_global=ego_xy_global,
        ego_yaw_global=ego_yaw_global,
        safety_len=SAFETY_LEN,
    )
    # ✅ driveway: polygon만 샘플링
    driveway_points = build_polygon_points(
        parsed_map.driveway_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )  # shape: (D, SAFETY_LEN, 2)
    # ✅ 기존 키(driveway_points) 호환 + 타입 구분용 bool도 같이 저장



    # ego_agent_past_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     ego_agent_past, prefix_dim=8)
    # ego_agent_past_is_valid = np.expand_dims(ego_agent_past_is_valid, axis=0)
    # assert_cur_future_valid_mask_np(
    #     ego_agent_past_is_valid,
    #     context=f" - ego_agent_past_is_valid",
    # )
    # ego_future_gt_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     ego_future_gt_3_dim, prefix_dim=3)
    # ego_future_gt_is_valid = np.expand_dims(ego_future_gt_is_valid, axis=0)
    # ego_future_gt_is_valid = np.expand_dims(ego_future_gt_is_valid, axis=0)
    # print("ego_future_gt_is_valid.shape:", ego_future_gt_is_valid.shape)
    # assert_cur_future_valid_mask_np(
    #     ego_future_gt_is_valid,
    #     context=f" - ego_future_gt_is_valid",
    # )
    # ego_future_gt_11_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     ego_future_gt_11_dim, prefix_dim=8)
    # ego_future_gt_11_is_valid = np.expand_dims(ego_future_gt_11_is_valid, axis=0)
    # ego_future_gt_11_is_valid = np.expand_dims(ego_future_gt_11_is_valid, axis=0)
    # assert_cur_future_valid_mask_np(
    #     ego_future_gt_11_is_valid,
    #     context=f" - ego_future_gt_11_is_valid",
    # )

    # neighbor_agents_past_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     neighbor_agents_past, prefix_dim=8)
    # neighbor_agents_past_is_valid = np.expand_dims(
    #     neighbor_agents_past_is_valid, axis=0)
    # assert_cur_future_valid_mask_np(
    #     neighbor_agents_past_is_valid,
    #     context=f" - neighbor_agents_past_is_valid",
    # )
    # neighbor_future_gt_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     neighbor_future_gt_3_dim, prefix_dim=3)
    # neighbor_future_gt_is_valid = np.expand_dims(
    #     neighbor_future_gt_is_valid, axis=0)
    # assert_cur_future_valid_mask_np(
    #     neighbor_future_gt_is_valid,
    #     context=f" - neighbor_future_gt_is_valid",
    # )
    # neighbor_future_gt_11_is_valid = _compute_valid_mask_from_prefix_nonzero(
    #     neighbor_future_gt_11_dim, prefix_dim=8)
    # neighbor_future_gt_11_is_valid = np.expand_dims(
    #     neighbor_future_gt_11_is_valid, axis=0)
    # assert_cur_future_valid_mask_np(
    #     neighbor_future_gt_11_is_valid,
    #     context=f" - neighbor_future_gt_11_is_valid",
    # )

    cache_dict: Dict[str, Any] = {
        "scenario_id": str(scenario.scenario_id),
        "origin_world_pose": origin_world_pose,
        # (4,) float32. [x,y,cos(yaw),sin(yaw)]
        "ego_agent_past": ego_agent_past,  # (21,11)  # womd
        "ego_future_gt_3_dim": ego_future_gt_3_dim,  # (80,3)  # womd
        "ego_future_gt_11_dim": ego_future_gt_11_dim,  # (80,11)  # womd
        "past_seg_control_gt_3_dim": past_seg_control_gt_3_dim, #  (1+Pnn, past_len, 3)
        "future_seg_control_gt_3_dim": future_seg_control_gt_3_dim, # (1+Pnn, future_len, 3)

        "neighbor_role": neighbor_role,  # (A,2) bool
        "target_id": target_id,  # (1+A,) int64. [ego_id, neighbor_id...]
        "target_z": target_z,    # (1+A,) float32. [ego_z, neighbor_z...]

        "neighbor_shape": neighbor_shape,  # (A,3)

        "neighbor_agents_past": neighbor_agents_past,  # (A,21,11)  # womd
        "neighbor_future_gt_3_dim": neighbor_future_gt_3_dim,  # (A,80,3)  # womd
        "neighbor_future_gt_11_dim": neighbor_future_gt_11_dim,
        # ✅ 추가: (A,80,11)
        "neighbor_track_token": neighbor_track_token,  # ✅ 추가: List[str], 길이 A

        "stop_sign_points": stop_sign_points,  # (Ns,10,2)  # womd
        "crosswalk_points": crosswalk_points,  # (Nc,10,2)  # womd
        "speed_bump_points": speed_bump_points,  # (Nb,10,2)  # womd
        "driveway_points": driveway_points,  # shape: (D,10,2)
        "lanes": lanes_arr,  # (L,10,12)  # womd
        "lanes_speed_limit": lanes_speed_limit,  # (L,1) # womd
        "lanes_has_speed_limit": lanes_has_speed_limit,  # (L,1) # womd
        "lane_light": lane_light,  # (L,4)
        # ✅ 추가 캐싱ㅇ
        "lane_type": lane_type,  # (L,4)
        "left_line_type": left_line_type,  # (L,13)

        "right_line_type": right_line_type,  # (L,13)

        "road_edge": road_edge,  # (E,10,2)
        "road_edge_type": road_edge_type,  # (E,3)
    }
    return cache_dict


# =========================
# TFRecord 파일 1개 처리(멀티프로세스 워커)
# =========================
def process_one_tfrecord_file(
    tfrecord_path: str,
    split: str,
    caching_dir: str,
    overwrite: bool,
    save_image: bool,
) -> Tuple[str, int, int, int, str]:
    """TFRecord 파일 1개를 읽어서, 안에 들어있는 시나리오들을 캐싱합니다.

    병렬 처리를 위해 "파일 단위"로 처리합니다.

    Args:
        tfrecord_path: 입력 tfrecord 경로
        split: training/validation/testing
        caching_dir: /home/user/womd_v1_3/cache
        overwrite: True면 이미 npz가 있어도 다시 만듭니다.

    Returns:
        (tfrecord_path, processed, skipped, failed, message)
    """
    if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
        return tfrecord_path, 0, 0, 0, "ABORTED_BY_USER"
    _claim_logger_if_needed()
    _log(f"START file={tfrecord_path} split={split}")
    t0 = time.perf_counter()
    last_hb = time.monotonic()
    hb_sec = 10.0
    processed = 0
    skipped = 0
    failed = 0
    message = "OK"

    tfrecord_p = Path(tfrecord_path)
    caching_p = Path(caching_dir)

    out_split_dir = caching_p / split
    ensure_dir(out_split_dir)

    out_validation_split_dir: Optional[Path] = None
    if split == "validation":
        out_validation_split_dir = caching_p / "validation_tfrecords_splitted"
        ensure_dir(out_validation_split_dir)

    compression_type = guess_tfrecord_compression(tfrecord_p)
    dataset = tf.data.TFRecordDataset(
        tfrecord_p.as_posix(),
        compression_type=compression_type,
        num_parallel_reads=1,
    ).prefetch(1)
    _log("TFRecordDataset created")
    _reset_worker_local_neighbor_xy_yaw_sums()
    for k, record_elem in enumerate(dataset.as_numpy_iterator()):
        if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
            message = "ABORTED_BY_USER"
            break

        try:
            if k == 0:
                _log("FIRST record fetched (as_numpy_iterator)")

            record_bytes = tfrecord_element_to_bytes(record_elem)
            if k == 0:
                _log("FIRST record -> numpy done")

            scenario = parse_scenario_from_bytes(record_bytes)
            scenario_id = str(scenario.scenario_id)
            if processed == 0:
                _log(f"FIRST scenario parsed id={scenario_id}")

            # ✅ pkl -> npz
            out_npz_path = out_split_dir / f"{scenario_id}.npz"
            out_tfrecord_path = None
            if out_validation_split_dir is not None:
                out_tfrecord_path = out_validation_split_dir / f"{scenario_id}.tfrecords"

            # 이미 있으면 skip (overwrite=False일 때)
            if (not overwrite) and out_npz_path.exists():
                if split == "validation" and out_tfrecord_path is not None and (
                        not out_tfrecord_path.exists()):
                    with tf.io.TFRecordWriter(out_tfrecord_path.as_posix()) as w:
                        w.write(record_bytes)
                skipped += 1
                continue

            if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
                message = "ABORTED_BY_USER"
                break

            traj_rect_dir: Optional[str] = None
            if bool(save_image):
                traj_rect_dir = os.path.join(
                    out_split_dir.as_posix() if hasattr(out_split_dir,
                                                        "as_posix") else str(
                        out_split_dir),
                    "traj_rect_compare")

            cache_dict = build_cache_dict_for_scenario(
                scenario,
                traj_rect_compare_dir=traj_rect_dir,
            )

            # ✅ npz 저장 (scenario_id, neighbor_track_token 제외)
            save_cache_dict_to_npz(
                cache_dict=cache_dict,
                out_npz_path=out_npz_path,
                excluded_keys=EXCLUDE_KEYS_FOR_NPZ,
                compress=True,
            )

            if processed == 0:
                _log(f"FIRST npz written: {out_npz_path}")

            if split == "validation" and out_tfrecord_path is not None:
                with tf.io.TFRecordWriter(out_tfrecord_path.as_posix()) as w:
                    w.write(record_bytes)

            now = time.monotonic()
            if now - last_hb >= hb_sec:
                elapsed = time.perf_counter() - t0
                _log(
                    f"PROGRESS file={tfrecord_p.name} rec={k+1} "
                    f"processed={processed} skipped={skipped} failed={failed} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_hb = now

            processed += 1

            if save_image:
                save_dir = os.path.join(out_split_dir, "debug_vis")
                save_path = os.path.join(save_dir, f"{scenario_id}.png")
                os.makedirs(save_dir, exist_ok=True)
                cache_dict["token_to_future_traj_wrt_ego"] = None
                # draw_machine_fast.draw_world_model_to_png(
                #     cache_dict,
                #     output_data={},
                #     save_path=save_path
                # )

        except Exception as e:
            failed += 1
            _log(
                f"[FAIL] scenario_id={scenario_id if 'scenario_id' in locals() else 'unknown'} err={repr(e)}"
            )
            _log(traceback.format_exc())
            message = f"FAILED: {repr(e)}"
    _flush_worker_local_neighbor_xy_yaw_sums_to_shared()

    return tfrecord_path, processed, skipped, failed, message


def _worker_init(stop_event, logger_pid_val, args, neighbor_xy_yaw_sums_proxy, neighbor_xy_yaw_sums_lock) -> None:
    global _WORKER_STOP_EVENT, _LOG_ENABLED, _LOGGER_PID_VAL
    global _NEIGHBOR_XY_YAW_SUMS_PROXY, _NEIGHBOR_XY_YAW_SUMS_LOCK

    _WORKER_STOP_EVENT = stop_event
    _LOGGER_PID_VAL = logger_pid_val
    _LOG_ENABLED = False

    _NEIGHBOR_XY_YAW_SUMS_PROXY = neighbor_xy_yaw_sums_proxy
    _NEIGHBOR_XY_YAW_SUMS_LOCK = neighbor_xy_yaw_sums_lock

    # ✅ args에서 길이 설정 주입
    _set_womd_lengths_from_args(args)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    faulthandler.enable()

from dataclasses import dataclass
from typing import Tuple
import time
from tqdm import tqdm


@dataclass
class CacheProgress:
    """메인 프로세스에서 캐싱 진행 상황을 누적해서 관리하는 상태 묶음입니다.

    이 스크립트는 워커가 tfrecord 파일 1개를 통째로 처리하고,
    메인 프로세스는 그 결과를 파일 단위로 받습니다.

    그래서 진행률(%)과 ETA는 "파일 개수 기준"으로 정확하게 계산하고,
    추가로 "시나리오 개수(레코드 개수)"는 파일 결과(processed/skipped/failed)를
    누적해서 같이 보여줍니다.

    Attributes:
        split: 현재 처리 중인 split 이름 (예: "training")
        total_files: 전체 tfrecord 파일 개수
        start_ts: 진행 시작 시각(time.monotonic() 기준)
        log_every_files: 몇 개 파일마다 한 번씩 진행 로그를 찍을지(대략적인 빈도)
        min_print_interval_sec: 로그가 너무 자주 찍히는 걸 막기 위한 최소 시간 간격(초)

        done_files: 완료된 tfrecord 파일 수(성공/실패 상관없이 “결과를 받은” 파일)
        scenarios_created: 새로 npz를 만든 시나리오 수(processed 누적)
        scenarios_skipped: 이미 npz가 있어서 건너뛴 시나리오 수(skipped 누적)
        scenarios_failed: 처리 중 예외가 난 시나리오 수(failed 누적)

        last_print_ts: 마지막으로 진행 로그를 찍은 시각(time.monotonic())
    """
    split: str
    total_files: int
    start_ts: float
    log_every_files: int
    min_print_interval_sec: float = 5.0

    done_files: int = 0
    scenarios_created: int = 0
    scenarios_skipped: int = 0
    scenarios_failed: int = 0

    last_print_ts: float = 0.0


def _format_hhmm(seconds: float) -> str:
    """초 단위 시간을 '00h00m' 형태로 바꿔서 보기 좋게 만듭니다.

    Args:
        seconds: 초 단위 시간(실수 가능)

    Returns:
        '00h00m' 형태 문자열.
        seconds가 0 이하이거나 계산이 어렵다면 '--:--'를 반환합니다.
    """
    sec = float(seconds)
    if not (sec > 0.0):
        return "--:--"
    hours = int(sec // 3600.0)
    minutes = int((sec % 3600.0) // 60.0)
    return f"{hours:02d}h{minutes:02d}m"


def _compute_eta_seconds(done: int, total: int, elapsed_sec: float) -> float:
    """현재 속도를 기준으로 남은 시간을 초 단위로 추정합니다.

    Args:
        done: 완료된 개수
        total: 전체 개수
        elapsed_sec: 경과 시간(초)

    Returns:
        남은 시간(초). 추정이 불가능하면 0.0을 반환합니다.
    """
    done_i = int(done)
    total_i = int(total)
    elapsed = float(elapsed_sec)

    if total_i <= 0 or done_i <= 0:
        return 0.0
    if done_i >= total_i:
        return 0.0
    if not (elapsed > 0.0):
        return 0.0

    speed = float(done_i) / elapsed  # files/sec
    if not (speed > 0.0):
        return 0.0

    remain = total_i - done_i
    return float(remain) / speed


def _should_print_progress(
    done: int,
    total: int,
    log_every: int,
    now_ts: float,
    last_print_ts: float,
    min_interval_sec: float,
) -> bool:
    """진행 로그를 지금 찍을지 여부를 결정합니다.

    너무 많은 로그가 찍히면 보기 어려워지므로,
    아래 조건 중 하나를 만족할 때만 찍습니다.

    - 첫 결과(done==1)
    - 마지막(done==total)
    - log_every 간격마다(예: 0.1% 정도 간격)
    - 단, 위 조건을 만족해도 min_interval_sec보다 너무 빠르면 잠깐 기다립니다.
      (마지막 로그는 무조건 찍습니다.)

    Args:
        done: 현재 완료 개수
        total: 전체 개수
        log_every: 몇 개마다 찍을지
        now_ts: 현재 시각(time.monotonic())
        last_print_ts: 마지막 출력 시각(time.monotonic())
        min_interval_sec: 최소 출력 간격(초)

    Returns:
        출력하면 True, 아니면 False
    """
    done_i = int(done)
    total_i = int(total)
    every = max(1, int(log_every))

    if total_i <= 0:
        return False

    if done_i <= 1:
        return True
    if done_i >= total_i:
        return True
    if done_i % every != 0:
        return False

    # 너무 빠른 연속 출력 방지
    if float(now_ts - last_print_ts) < float(min_interval_sec):
        return False
    return True


def _print_cache_line(line: str) -> None:
    """tqdm 진행바가 깨지지 않게 한 줄 로그를 출력합니다.

    Args:
        line: 출력할 한 줄 문자열
    """
    # tqdm 환경에서도 줄바꿈 출력이 깔끔하도록 tqdm.write 사용
    tqdm.write(line)
    try:
        # tqdm.write는 내부적으로 flush를 하지만, 로그 파일로 리다이렉트할 때를 대비해 한 번 더.
        sys.stdout.flush()
    except Exception:
        pass


def print_cache_start(
    progress: CacheProgress,
    num_workers: int,
    overwrite: bool,
    caching_dir: str,
) -> None:
    """split 캐싱 시작 시점에 '총량'과 설정을 한 번 출력합니다.

    Args:
        progress: 진행 상태(초기화된 상태)
        num_workers: 워커 프로세스 수
        overwrite: overwrite 여부
        caching_dir: 캐시 저장 루트 폴더 경로(문자열)
    """
    _print_cache_line(
        f"[CACHE] start split={progress.split}: "
        f"{progress.total_files:,} tfrecord files "
        f"(workers={int(num_workers)}, overwrite={bool(overwrite)}) | "
        f"out={caching_dir}"
    )


def update_and_maybe_print_cache_progress(
    progress: CacheProgress,
    file_result: Tuple[str, int, int, int, str],
) -> None:
    """tfrecord 파일 1개 처리 결과를 누적하고, 필요할 때 진행 로그를 출력합니다.

    Args:
        progress: 누적 상태
        file_result: process_one_tfrecord_file()의 반환값
            (tfrecord_path, processed, skipped, failed, message)
            - processed/skipped/failed는 "시나리오 개수"입니다.
    """
    _, processed, skipped, failed, _ = file_result

    progress.done_files += 1
    progress.scenarios_created += int(processed)
    progress.scenarios_skipped += int(skipped)
    progress.scenarios_failed += int(failed)

    now_ts = time.monotonic()
    elapsed = float(now_ts - progress.start_ts)
    eta_sec = _compute_eta_seconds(
        done=progress.done_files,
        total=progress.total_files,
        elapsed_sec=elapsed,
    )

    if not _should_print_progress(
        done=progress.done_files,
        total=progress.total_files,
        log_every=progress.log_every_files,
        now_ts=now_ts,
        last_print_ts=progress.last_print_ts,
        min_interval_sec=progress.min_print_interval_sec,
    ):
        return

    progress.last_print_ts = now_ts

    pct = 0.0
    if progress.total_files > 0:
        pct = 100.0 * float(progress.done_files) / float(progress.total_files)

    # 시나리오 처리량(속도)도 같이 보여주면 전체 시간 감이 빨리 옵니다.
    scenarios_seen = int(progress.scenarios_created + progress.scenarios_skipped + progress.scenarios_failed)
    scen_per_sec = float(scenarios_seen) / elapsed if elapsed > 0.0 else 0.0

    eta_str = "--:--" if progress.done_files >= progress.total_files else _format_hhmm(eta_sec)

    _print_cache_line(
        f"[CACHE] {progress.split} "
        f"{progress.done_files:,}/{progress.total_files:,} "
        f"({pct:5.1f}%) | "
        f"elapsed {_format_hhmm(elapsed)}, "
        f"ETA {eta_str} | "
        f"scenarios seen {scenarios_seen:,} "
        f"(new {progress.scenarios_created:,}, "
        f"skip {progress.scenarios_skipped:,}, "
        f"fail {progress.scenarios_failed:,}) | "
        f"scen/s {scen_per_sec:.2f}"
    )

from pathlib import Path
from typing import Any


def _require_save_path_args(args: Any) -> None:
    """save_path 인자가 반드시 제공되었는지 검사합니다.

    이 스크립트는 결과 캐시(.npz)를 저장할 루트 폴더를
    오직 `--save_path` 인자에서만 받도록 강제합니다.

    - save_folder 같은 '추가 폴더 이름' 인자는 사용하지 않습니다.
    - save_path가 비어 있거나 기본값처럼 보이는 값이면, 실수로 다른 위치에 저장되는 걸 막기 위해
      즉시 에러를 냅니다.

    Args:
        args: args_util.get_args()가 반환한 인자 객체(보통 argparse.Namespace)

    Raises:
        RuntimeError: save_path가 없거나, 비어 있거나, 기본값("./cache")인 경우
    """
    if not hasattr(args, "save_path"):
        raise RuntimeError(
            "필수 인자가 args에 없습니다: save_path. "
            "data_process_womd.py는 반드시 `--save_path <OUTPUT_DIR>` 를 받아야 합니다."
        )

    save_path = str(getattr(args, "save_path")).strip()
    if save_path == "":
        raise RuntimeError(
            "`--save_path`가 비어 있습니다. 출력 폴더를 반드시 지정해야 합니다."
        )

    # args_util.py 기본값(현재 ./cache)을 사실상 '미지정'으로 간주해서 막습니다.
    # 정말 ./cache에 저장하고 싶으면, 의도적으로 다른 경로로 바꾸거나 이 체크를 제거하세요.
    if save_path == "./cache":
        raise RuntimeError(
            "`--save_path`가 기본값('./cache')입니다. "
            "실수 방지를 위해 data_process_womd.py에서는 기본 저장 위치를 허용하지 않습니다. "
            "원하는 출력 폴더를 `--save_path`로 명시해 주세요."
        )


def _get_cache_root_dir_from_args(args: Any) -> Path:
    """save_path를 Path로 정규화해서 반환합니다.

    Args:
        args: args_util.get_args() 결과

    Returns:
        cache_root_dir: 캐시 출력 루트 폴더 경로(Path)
    """
    _require_save_path_args(args)
    return Path(str(getattr(args, "save_path"))).expanduser()


# =========================
# split 전체 캐싱
# =========================
def cache_all_splits(
    args,
    data_path: str,
    num_workers: int,
    overwrite: bool,
    splits: Tuple[str, ...] = SPLITS,
) -> None:
    dataset_p = Path(data_path)
    scenario_dir = dataset_p / "scenario"
    # ✅ 출력 루트는 오직 save_path만 사용
    caching_dir = _get_cache_root_dir_from_args(args)
    ensure_dir(caching_dir)

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()

    stop_event = ctx.Event()

    # 메인 시그널 핸들링(1회: graceful, 2회: hard kill)
    pool_ref: Dict[str, Optional[mppool.Pool]] = {"pool": None}
    old_handlers, shutdown_state = _install_main_signal_handlers(
        stop_event=stop_event,
        pool_ref=pool_ref,
        grace_sec=12.0,  # 워커가 stop_event 보고 빠져나올 시간(필요하면 늘리기)
    )

    try:
        for split in splits:
            split_input_dir = scenario_dir / split
            split_output_dir = caching_dir / split
            ensure_dir(split_output_dir)

            if split == "validation":
                ensure_dir(caching_dir / "validation_tfrecords_splitted")

            tfrecord_files: List[Path] = list_tfrecord_files(split_input_dir)
            if len(tfrecord_files) == 0:
                print(f"[WARN] No files found: {split_input_dir}")
                continue

            process_one_tfrecord_file_fn = partial(
                process_one_tfrecord_file,
                split=split,
                caching_dir=caching_dir.as_posix(),
                overwrite=overwrite,
                save_image=bool(args.save_image),
            )
            pool: Optional[mppool.Pool] = None
            results: List[Tuple[str, int, int, int, str]] = []
            tfrecord_paths = [p.as_posix() for p in tfrecord_files]

            # ✅ [추가] split 단위 진행률(파일 기준 %/ETA + 시나리오 누적량) 초기화/시작 로그
            progress = CacheProgress(
                split=str(split),
                total_files=int(len(tfrecord_paths)),
                start_ts=time.monotonic(),
                log_every_files=max(1, int(len(tfrecord_paths)) // 3000),  # data_process.py의 감도(0.1% 수준)와 유사
                min_print_interval_sec=5.0,  # 너무 잦은 출력 방지(필요하면 10~30으로 늘려도 됨)
                last_print_ts=time.monotonic(),
            )
            print_cache_start(
                progress=progress,
                num_workers=int(num_workers),
                overwrite=bool(overwrite),
                caching_dir=caching_dir.as_posix(),
            )
            try:
                neighbor_xy_yaw_sums_proxy = manager.dict({
                    "count": 0,
                    "sum_xy": 0.0,
                    "sum_sq_xy": 0.0,
                    "sum_yaw": 0.0,
                    "sum_sq_yaw": 0.0,
                    "skipped_no_valid": 0,
                })
                neighbor_xy_yaw_sums_lock = manager.Lock()
                logger_pid_val = ctx.Value('i', 0)  # 0이면 아직 아무도 로거 선점 안 함
                pool = ctx.Pool(
                    processes=num_workers,
                    initializer=_worker_init,
                    initargs=(stop_event, logger_pid_val, args, neighbor_xy_yaw_sums_proxy, neighbor_xy_yaw_sums_lock),
                )
                pool_ref["pool"] = pool
                try:
                    msg = pool.apply_async(_ping).get(timeout=10)
                    _log(f"WORKER READY: {msg}")
                except Exception as e:
                    _log(
                        f"WORKER NOT READY (likely importing TF/torch/etc): {repr(e)}"
                    )
                die_all_enabled: bool = bool(getattr(args, "die_all", False))


                iterator = pool.imap_unordered(
                    process_one_tfrecord_file_fn,
                    tfrecord_paths,
                    chunksize=1,
                )

                # 무기한 block 대신 timeout polling (Ctrl+C 반응성 확보)
                pbar = tqdm(total=len(tfrecord_paths), desc=f"cache-{split}")
                try:
                    while True:
                        # ✅ die_all=True면: 워커가 하나라도 죽었는지 먼저 확인
                        _die_all_if_any_worker_dead(
                            pool=pool,
                            stop_event=stop_event,
                            enabled=die_all_enabled,
                            exit_code=1,
                        )

                        if stop_event.is_set():
                            deadline = shutdown_state.get("deadline", None)
                            if deadline is not None and time.monotonic(
                            ) > float(deadline):
                                _log("grace 기간 초과 -> 워커 강제 종료(SIGKILL 포함)합니다.")
                                _terminate_pool_hard(pool, timeout_sec=1.0)
                                raise SystemExit(130)

                        try:
                            res = iterator.next(timeout=0.5)
                        except mp.TimeoutError:
                            # ✅ 기다리는 동안에도 워커가 죽을 수 있으니 한 번 더 체크
                            _die_all_if_any_worker_dead(
                                pool=pool,
                                stop_event=stop_event,
                                enabled=die_all_enabled,
                                exit_code=1,
                            )
                            continue
                        except StopIteration:
                            break

                        results.append(res)
                        pbar.update(1)
                        # ✅ [추가] 파일 1개 결과를 누적하고, 종종 [CACHE] 진행 로그 출력
                        update_and_maybe_print_cache_progress(
                            progress=progress,
                            file_result=res,
                        )
                finally:
                    pbar.close()

                # 정리
                with contextlib.suppress(Exception):
                    pool.close()

                if not _join_pool_soft(pool, timeout_sec=8.0):
                    _terminate_pool_hard(pool, timeout_sec=1.0)

                pool_ref["pool"] = None
                # ------------------------------------------------------------
                # (추가) split 전체 누적 통계 출력
                # ------------------------------------------------------------
                stats = dict(neighbor_xy_yaw_sums_proxy)
                cnt = int(stats.get("count", 0))
                skipped_no_valid = int(stats.get("skipped_no_valid", 0))

                mean_xy, var_xy = _mean_and_variance_from_sums(
                    count=cnt,
                    sum_val=float(stats.get("sum_xy", 0.0)),
                    sum_sq_val=float(stats.get("sum_sq_xy", 0.0)),
                )
                mean_yaw, var_yaw = _mean_and_variance_from_sums(
                    count=cnt,
                    sum_val=float(stats.get("sum_yaw", 0.0)),
                    sum_sq_val=float(stats.get("sum_sq_yaw", 0.0)),
                )

                _print_cache_line(
                    f"[LOSS_STATS] split={split} "
                    f"count={cnt} skipped_no_valid={skipped_no_valid} | "
                    f"neighbor_prediction_loss_xy mean={mean_xy:.6f} var={var_xy:.6f} | "
                    f"neighbor_prediction_loss_yaw mean={mean_yaw:.6f} var={var_yaw:.6f}"
                )

                # Ctrl+C로 stop_event가 켜졌다면 여기서 종료
                if stop_event.is_set():
                    raise SystemExit(130)

            except KeyboardInterrupt:
                # 혹시라도 KeyboardInterrupt로 들어오는 환경 대비
                stop_event.set()
                _log("KeyboardInterrupt -> 워커 종료 후 종료합니다.")
                _terminate_pool_hard(pool, timeout_sec=1.0)
                raise SystemExit(130)

            except Exception:
                stop_event.set()
                _terminate_pool_hard(pool, timeout_sec=1.0)
                raise

            # 실패 요약 출력 (기존 로직 유지)
            num_failed_files = sum(
                1 for _, _, _, failed, _ in results if failed > 0)
            if num_failed_files > 0:
                print(
                    f"[WARN] split={split} failed_files={num_failed_files}/{len(results)}"
                )
                for tfp, processed, skipped, failed, msg in results:
                    if failed > 0:
                        print(
                            f"  - {tfp} | processed={processed} skipped={skipped} failed={failed} | {msg}"
                        )

    finally:
        # 시그널 핸들러 원복
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, old_handlers[signal.SIGINT])
            signal.signal(signal.SIGTERM, old_handlers[signal.SIGTERM])


# =========================
# CLI
# =========================
def _str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "t", "yes", "y")


if __name__ == "__main__":
    args = args_util.get_args()
    _require_filter_radius_args(args)
    _require_save_path_args(args)  # ✅ 추가: save_path 강제


    splits = tuple(
        [s.strip() for s in args.womd_splits.split(",") if len(s.strip()) > 0])
    cache_all_splits(
        args,
        data_path=args.womd_data_path,
        num_workers=int(args.num_workers),
        overwrite=_str2bool(args.overwrite_womd_cache),
        splits=splits,
    )