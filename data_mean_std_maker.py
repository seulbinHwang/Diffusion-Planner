# data_mean_std_maker.py
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

ArrayF = NDArray[np.floating]

# ----------------------------
# key 이름들(요청에서 준 그대로)
# ----------------------------
EGO_PAST_KEY = "ego_agent_past"
EGO_FUTURE_KEY = "ego_future_gt_11_dim"
NBR_PAST_KEY = "neighbor_agents_past"
NBR_FUTURE_KEY = "neighbor_future_gt_11_dim"

PAST_CONTROL_KEY = "past_seg_control_gt_3_dim"
FUTURE_CONTROL_KEY = "future_seg_control_gt_3_dim"

STATIC_OBJECTS_KEY = "static_objects"

LANES_KEY = "lanes"
LANES_SPEED_LIMIT_KEY = "lanes_speed_limit"
LANES_HAS_SPEED_LIMIT_KEY = "lanes_has_speed_limit"  # 있으면 사용(없어도 됨)

POINTSET_KEYS = (
    "stop_sign_points",
    "crosswalk_points",
    "speed_bump_points",
    "driveway_points",
    "road_edge",
)

# worker에서 공통으로 쓰는 설정 저장용
_WORKER_CONFIG: Dict[str, Any] = {}


# ============================================================
# 1) 실행 중 누적 통계: count / sum / sumsq
# ============================================================
class RunningStats:
    """값들을 저장하지 않고 mean/std를 구하기 위한 누적기입니다.

    이 누적기는 아래 3가지만 쌓습니다.
      - count: 유효한 샘플 개수
      - sum:   채널별 합
      - sumsq: 채널별 제곱의 합

    이렇게 하면 파일이 매우 많아도 메모리를 거의 쓰지 않습니다.

    Attributes:
        dim (int): 마지막 차원의 길이(예: 6채널이면 6).
        count (int): 누적된 유효 샘플 수.
        sum (np.ndarray): shape (dim,) 합.
        sumsq (np.ndarray): shape (dim,) 제곱합.
    """

    def __init__(self, dim: int) -> None:
        self.dim: int = int(dim)
        self.count: int = 0
        self.sum: NDArray[np.float64] = np.zeros((self.dim,), dtype=np.float64)  # (dim,)
        self.sumsq: NDArray[np.float64] = np.zeros((self.dim,), dtype=np.float64)  # (dim,)

    def update(self, values: np.ndarray) -> None:
        """(M, dim) 값들을 누적합니다.

        Args:
            values (np.ndarray): shape (M, dim)
                - M은 샘플 개수(0일 수도 있음)
                - dim은 self.dim과 같아야 함

        Returns:
            None
        """
        v = np.asarray(values)
        if v.size == 0:
            return
        if v.ndim != 2 or int(v.shape[1]) != int(self.dim):
            raise ValueError(f"values는 (M,{self.dim}) 이어야 합니다. got shape={v.shape}")

        v64 = v.astype(np.float64, copy=False)

        # 행 단위로 유한값 체크(각 행의 모든 채널이 유한한 경우만 사용)
        finite_row = np.isfinite(v64).all(axis=1)  # (M,)
        if not np.any(finite_row):
            return

        v_sel = v64[finite_row]  # (K, dim)
        if v_sel.size == 0:
            return

        self.count += int(v_sel.shape[0])
        self.sum += v_sel.sum(axis=0)
        self.sumsq += (v_sel * v_sel).sum(axis=0)

    def merge_from_parts(self, count: int, sum_list: Sequence[float], sumsq_list: Sequence[float]) -> None:
        """멀티프로세스 워커가 보내준 (count, sum, sumsq)을 합칩니다.

        Args:
            count (int): 샘플 수
            sum_list (Sequence[float]): 길이 dim
            sumsq_list (Sequence[float]): 길이 dim

        Returns:
            None
        """
        self.count += int(count)
        self.sum += np.asarray(sum_list, dtype=np.float64).reshape((self.dim,))
        self.sumsq += np.asarray(sumsq_list, dtype=np.float64).reshape((self.dim,))

    def to_parts(self) -> Tuple[int, List[float], List[float]]:
        """현재 누적값을 (count, sum_list, sumsq_list)로 꺼냅니다."""
        return (
            int(self.count),
            [float(x) for x in self.sum.reshape((self.dim,)).tolist()],
            [float(x) for x in self.sumsq.reshape((self.dim,)).tolist()],
        )

    def compute_mean_std(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """누적된 값으로 mean/std를 계산합니다.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mean: shape (dim,)
                - std:  shape (dim,)
        """
        if int(self.count) <= 0:
            nan = np.full((self.dim,), np.nan, dtype=np.float64)
            return nan, nan

        n = float(self.count)
        mean = self.sum / n
        var = (self.sumsq / n) - (mean * mean)
        var = np.maximum(var, 0.0)  # 수치 오차로 음수가 되는 것 방지
        std = np.sqrt(var)
        return mean, std


# ============================================================
# 2) npz 로딩/파일 리스트/병렬 도우미
# ============================================================
def _load_training_file_list(json_path: str) -> List[str]:
    """train_json(문자열 리스트)을 읽습니다."""
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("train_json은 '문자열 리스트' 형식이어야 합니다.")
    return data


def _load_npz_subset_as_dict(npz_path: str, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """npz에서 필요한 key만 골라 dict로 읽습니다(없는 key는 결과에 없음).

    Args:
        npz_path (str): npz 경로
        keys (Sequence[str]): 읽을 key 목록

    Returns:
        Dict[str, np.ndarray]: key -> 배열
    """
    key_list = [str(k) for k in keys]
    key_set = set(key_list)

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            files = set(npz.files)
            return {k: npz[k] for k in key_set if k in files}
    except ValueError as e:
        msg = str(e)
        if "allow_pickle=False" in msg or "Object arrays cannot be loaded" in msg:
            with np.load(npz_path, allow_pickle=True) as npz:
                files = set(npz.files)
                return {k: npz[k] for k in key_set if k in files}
        raise


def _get_auto_worker_count(total_files: int) -> int:
    """자동 worker 개수를 정합니다(최대 16, 파일 수보다 크지 않게)."""
    cpu = int(os.cpu_count() or 1)
    upper = min(16, cpu)
    if int(total_files) <= 0:
        return 1
    return max(1, min(int(total_files), int(upper)))


def _init_worker_process(config: Dict[str, Any]) -> None:
    """워커 프로세스 시작 시 공통 설정을 저장합니다."""
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)


def _format_seconds_to_hh_mm(seconds: float) -> str:
    """초를 HH:MM으로 바꿉니다(보기용)."""
    s = float(seconds)
    if (not np.isfinite(s)) or s < 0.0:
        s = 0.0
    total_minutes = int((s + 30.0) // 60.0)
    hours = total_minutes // 60
    minutes = total_minutes % 60
    return f"{hours:02d}:{minutes:02d}"


def _maybe_print_progress_every_5_min(
    *,
    start_time_s: float,
    last_print_time_s: float,
    processed: int,
    total: int,
    interval_s: float = 300.0,
) -> float:
    """일정 시간마다 진행률을 출력합니다."""
    now = time.monotonic()
    if (now - float(last_print_time_s)) < float(interval_s):
        return last_print_time_s
    if total <= 0 or processed <= 0:
        return now

    elapsed_s = now - float(start_time_s)
    pct = (float(processed) / float(total)) * 100.0
    avg_s_per_file = elapsed_s / float(processed)
    remaining_files = max(int(total - processed), 0)
    remaining_s = avg_s_per_file * float(remaining_files)

    tqdm.write(
        f"[진행] {pct:.2f}% ({processed}/{total}) | "
        f"경과 {_format_seconds_to_hh_mm(elapsed_s)} | "
        f"남은시간(예상) {_format_seconds_to_hh_mm(remaining_s)}"
    )
    return now


# ============================================================
# 3) 무효점(valid) 마스크 만들기
# ============================================================
def _valid_mask_by_first_k_nonzero(arr: np.ndarray, *, k: int, eps: float) -> NDArray[np.bool_]:
    """마지막 축의 앞 k개 값 중 하나라도 0이 아니면 True(=유효)로 보는 마스크를 만듭니다.

    Args:
        arr (np.ndarray): shape (..., D)
        k (int): 앞에서 볼 개수
        eps (float): 0 판정 기준(아주 작은 값)

    Returns:
        np.ndarray: shape (...) bool
    """
    a = np.asarray(arr)
    if a.ndim < 1 or int(a.shape[-1]) < int(k):
        raise ValueError(f"arr 마지막 차원은 최소 {k} 이상이어야 합니다. got shape={a.shape}")
    return (np.abs(a[..., :k]) > float(eps)).any(axis=-1)


def _derive_control_valid_from_state_valid(state_valid: NDArray[np.bool_], control_len: int) -> NDArray[np.bool_]:
    """상태 유효 마스크(state_valid)로부터 control 유효 마스크를 만듭니다.

    여기서 "상태"는 (x,y,...) 같은 시점별 값이고,
    "control"은 시점과 시점 사이(구간)의 값이라고 가정합니다.

    자동 정렬 규칙(현실 데이터에 맞춰 최대한 안전하게):
      - control_len == T-1 이면: state_valid[t] & state_valid[t+1] (길이 T-1)
      - control_len == T 이면: 0번째는 False, 나머지는 state_valid[t-1] & state_valid[t]
      - 그 외 길이면: 가능한 구간만 채우고 나머지는 False로 둡니다.

    Args:
        state_valid (np.ndarray): shape (A, T)
            A는 객체 수(ego 포함), T는 시간 길이
        control_len (int): control의 시간 길이

    Returns:
        np.ndarray: shape (A, control_len) bool
    """
    sv = np.asarray(state_valid).astype(bool)
    if sv.ndim != 2:
        raise ValueError(f"state_valid는 (A,T) 이어야 합니다. got shape={sv.shape}")

    A, T = int(sv.shape[0]), int(sv.shape[1])
    L = int(control_len)
    if L <= 0:
        return np.zeros((A, 0), dtype=bool)

    out = np.zeros((A, L), dtype=bool)

    if T <= 1:
        return out

    seg = (sv[:, :-1] & sv[:, 1:])  # (A, T-1)

    if L == (T - 1):
        out[:, :] = seg
        return out

    if L == T:
        out[:, 0] = False
        out[:, 1:] = seg
        return out

    # 길이가 다르면: 가능한 만큼만 앞에서부터 채움
    fill = min(int(seg.shape[1]), L)
    if fill > 0:
        out[:, :fill] = seg[:, :fill]
    return out


# ============================================================
# 4) 각 key별 통계 업데이트 로직(핵심)
# ============================================================
def _update_agent_stats(
    *,
    acc_agent_dyn6: RunningStats,
    acc_agent_size2: RunningStats,
    ego_past: Optional[np.ndarray],
    ego_future: Optional[np.ndarray],
    nbr_past: Optional[np.ndarray],
    nbr_future: Optional[np.ndarray],
    eps: float,
) -> Tuple[
    Optional[NDArray[np.bool_]],
    Optional[NDArray[np.bool_]],
    Optional[NDArray[np.bool_]],
    Optional[NDArray[np.bool_]],
]:
    """agent(11차원)에서 통계를 업데이트합니다.

    업데이트하는 값:
      - dyn6: (x,y,cos,sin,vx,vy) 6개는 "프레임 단위"로 전부 집계
      - size2: (width,length) 2개는 "agent 트랙 단위"로 첫 valid 프레임 1개만 집계

    Args:
        acc_agent_dyn6: dyn6 누적기 (dim=6)
        acc_agent_size2: size2 누적기 (dim=2)
        ego_past: (Tp,11) 또는 None
        ego_future: (Tf,11) 또는 None
        nbr_past: (N,Tp,11) 또는 None
        nbr_future: (N,Tf,11) 또는 None
        eps: 0 판정 기준

    Returns:
        (ego_past_valid, ego_future_valid, nbr_past_valid, nbr_future_valid)
        - ego_past_valid: shape (Tp,) 또는 None
        - ego_future_valid: shape (Tf,) 또는 None
        - nbr_past_valid: shape (N,Tp) 또는 None
        - nbr_future_valid: shape (N,Tf) 또는 None
    """
    ego_past_valid: Optional[NDArray[np.bool_]] = None
    ego_future_valid: Optional[NDArray[np.bool_]] = None
    nbr_past_valid: Optional[NDArray[np.bool_]] = None
    nbr_future_valid: Optional[NDArray[np.bool_]] = None

    # -----------------------
    # (1) dyn6: 프레임 단위
    # -----------------------
    if ego_past is not None:
        ep = np.asarray(ego_past)
        if ep.ndim == 2 and int(ep.shape[1]) == 11:
            ego_past_valid = _valid_mask_by_first_k_nonzero(ep, k=8, eps=eps)  # (Tp,)
            if np.any(ego_past_valid):
                acc_agent_dyn6.update(ep[ego_past_valid, :6])  # (K,6)

    if ego_future is not None:
        ef = np.asarray(ego_future)
        if ef.ndim == 2 and int(ef.shape[1]) == 11:
            ego_future_valid = _valid_mask_by_first_k_nonzero(ef, k=8, eps=eps)  # (Tf,)
            if np.any(ego_future_valid):
                acc_agent_dyn6.update(ef[ego_future_valid, :6])  # (K,6)

    if nbr_past is not None:
        npast = np.asarray(nbr_past)
        if npast.ndim == 3 and int(npast.shape[2]) == 11:
            nbr_past_valid = _valid_mask_by_first_k_nonzero(npast, k=8, eps=eps)  # (N,Tp)
            if np.any(nbr_past_valid):
                flat = npast.reshape((-1, 11))  # (N*Tp,11)
                mask = nbr_past_valid.reshape((-1,))  # (N*Tp,)
                acc_agent_dyn6.update(flat[mask, :6])  # (K,6)

    if nbr_future is not None:
        nfut = np.asarray(nbr_future)
        if nfut.ndim == 3 and int(nfut.shape[2]) == 11:
            nbr_future_valid = _valid_mask_by_first_k_nonzero(nfut, k=8, eps=eps)  # (N,Tf)
            if np.any(nbr_future_valid):
                flat = nfut.reshape((-1, 11))  # (N*Tf,11)
                mask = nbr_future_valid.reshape((-1,))
                acc_agent_dyn6.update(flat[mask, :6])  # (K,6)

    # -----------------------
    # (2) size2: agent 단위(첫 valid 프레임 1개)
    # -----------------------
    # ego 1개 샘플
    ego_size_samples: List[np.ndarray] = []
    if ego_past is not None and ego_past_valid is not None and np.any(ego_past_valid):
        idx = int(np.argmax(ego_past_valid))
        ego_size_samples.append(np.asarray(ego_past)[idx, 6:8].reshape((1, 2)))
    elif ego_future is not None and ego_future_valid is not None and np.any(ego_future_valid):
        idx = int(np.argmax(ego_future_valid))
        ego_size_samples.append(np.asarray(ego_future)[idx, 6:8].reshape((1, 2)))

    if ego_size_samples:
        acc_agent_size2.update(np.concatenate(ego_size_samples, axis=0))

    # neighbor 여러 개 샘플
    if nbr_past is not None or nbr_future is not None:
        # 둘 다 없으면 종료
        if nbr_past is None and nbr_future is None:
            pass
        else:
            # N 추정
            N: int = 0
            if nbr_past is not None:
                N = int(np.asarray(nbr_past).shape[0])
            elif nbr_future is not None:
                N = int(np.asarray(nbr_future).shape[0])

            if N > 0:
                has_past = None
                has_fut = None
                idx_past = None
                idx_fut = None

                if nbr_past_valid is not None:
                    has_past = nbr_past_valid.any(axis=1)  # (N,)
                    idx_past = np.argmax(nbr_past_valid, axis=1)  # (N,)
                if nbr_future_valid is not None:
                    has_fut = nbr_future_valid.any(axis=1)  # (N,)
                    idx_fut = np.argmax(nbr_future_valid, axis=1)  # (N,)

                samples_list: List[np.ndarray] = []

                if (nbr_past is not None) and (has_past is not None) and np.any(has_past):
                    npast = np.asarray(nbr_past)
                    ar = np.arange(N, dtype=np.int64)
                    size_all = npast[ar, idx_past.astype(np.int64), 6:8]  # (N,2)
                    samples_list.append(size_all[has_past])  # (K,2)

                # past가 없거나 past에 valid가 없는 agent는 future에서 가져오기
                if nbr_future is not None and (has_fut is not None) and np.any(has_fut):
                    nfut = np.asarray(nbr_future)
                    ar = np.arange(N, dtype=np.int64)

                    if has_past is None:
                        need_fut = has_fut
                    else:
                        need_fut = (~has_past) & has_fut

                    if np.any(need_fut):
                        size_all = nfut[ar, idx_fut.astype(np.int64), 6:8]  # (N,2)
                        samples_list.append(size_all[need_fut])  # (K,2)

                if samples_list:
                    acc_agent_size2.update(np.concatenate(samples_list, axis=0))

    return ego_past_valid, ego_future_valid, nbr_past_valid, nbr_future_valid


def _update_control_stats(
    *,
    acc_control3: RunningStats,
    past_control: Optional[np.ndarray],
    future_control: Optional[np.ndarray],
    ego_past_valid: Optional[NDArray[np.bool_]],
    ego_future_valid: Optional[NDArray[np.bool_]],
    nbr_past_valid: Optional[NDArray[np.bool_]],
    nbr_future_valid: Optional[NDArray[np.bool_]],
) -> None:
    """past/future control(3차원) 통계를 업데이트합니다.

    padding/무효 구간을 빼기 위해, control 자체 값(0인지 여부)이 아니라
    "상태(ego/neighbor 궤적)의 유효 마스크"로 control 유효 마스크를 만듭니다.

    Args:
        acc_control3: control 누적기 (dim=3)
        past_control: shape (1+N, Lp, 3) 또는 None
        future_control: shape (1+N, Lf, 3) 또는 None
        ego_past_valid: shape (Tp,) 또는 None
        ego_future_valid: shape (Tf,) 또는 None
        nbr_past_valid: shape (N,Tp) 또는 None
        nbr_future_valid: shape (N,Tf) 또는 None

    Returns:
        None
    """
    # (A) past control
    if past_control is not None and ego_past_valid is not None and nbr_past_valid is not None:
        pc = np.asarray(past_control)
        if pc.ndim == 3 and int(pc.shape[2]) == 3:
            # state_valid_past: (1+N, Tp)
            state_valid_past = np.concatenate(
                [ego_past_valid[None, :], nbr_past_valid.astype(bool)], axis=0
            ).astype(bool)

            control_valid = _derive_control_valid_from_state_valid(state_valid_past, int(pc.shape[1]))  # (1+N,Lp)
            if np.any(control_valid):
                acc_control3.update(pc[control_valid])  # (K,3)

    # (B) future control
    # future는 "현재(ego past 마지막) + ego future" 형태로 state_valid를 구성
    if (
        future_control is not None
        and ego_past_valid is not None
        and ego_future_valid is not None
        and nbr_past_valid is not None
        and nbr_future_valid is not None
    ):
        fc = np.asarray(future_control)
        if fc.ndim == 3 and int(fc.shape[2]) == 3:
            ego_cur = np.asarray([bool(ego_past_valid[-1])], dtype=bool)  # (1,)
            ego_state_valid = np.concatenate([ego_cur, ego_future_valid.astype(bool)], axis=0)  # (1+Tf,)

            nbr_cur = nbr_past_valid[:, -1].astype(bool)  # (N,)
            nbr_state_valid = np.concatenate(
                [nbr_cur[:, None], nbr_future_valid.astype(bool)], axis=1
            )  # (N,1+Tf)

            state_valid_future = np.concatenate(
                [ego_state_valid[None, :], nbr_state_valid], axis=0
            ).astype(bool)  # (1+N,1+Tf)

            control_valid = _derive_control_valid_from_state_valid(state_valid_future, int(fc.shape[1]))
            if np.any(control_valid):
                acc_control3.update(fc[control_valid])  # (K,3)


def _update_static_objects_stats(*, acc_static6: RunningStats, static_objects: Optional[np.ndarray], eps: float) -> None:
    """static_objects(10차원) 중 연속값 6개(x,y,cos,sin,width,length)만 통계를 냅니다.

    one-hot 4개는 정규화하지 않을 예정이라 통계에서 제외합니다.

    Args:
        acc_static6: dim=6 누적기
        static_objects: shape (S,10) 또는 None
        eps: 0 판정 기준

    Returns:
        None
    """
    if static_objects is None:
        return
    so = np.asarray(static_objects)
    if so.ndim != 2 or int(so.shape[1]) != 10:
        return

    valid = _valid_mask_by_first_k_nonzero(so, k=6, eps=eps)  # (S,)
    if not np.any(valid):
        return
    acc_static6.update(so[valid, :6])  # (K,6)


def _update_lanes_stats(
    *,
    acc_lanes_xy2: RunningStats,
    acc_lanes_vec6: RunningStats,
    acc_speed_limit1: RunningStats,
    lanes: Optional[np.ndarray],
    lanes_speed_limit: Optional[np.ndarray],
    lanes_has_speed_limit: Optional[np.ndarray],
    eps: float,
) -> None:
    """lanes(12차원) + lanes_speed_limit 통계를 업데이트합니다.

    lanes:
      - 좌표(0~1): (x,y)
      - 벡터(2~7): (dx,dy) 3쌍 = 총 6개
      - 신호 one-hot(8~11): 정규화 안 함

    lanes 무효점:
      - 앞 8개(0~7)가 전부 0이면 그 점은 무효

    speed_limit:
      - lane 단위로 유효 lane만 사용
      - lanes_has_speed_limit가 있으면 True인 lane만 사용(있을 때만)

    Args:
        acc_lanes_xy2: dim=2 누적기
        acc_lanes_vec6: dim=6 누적기
        acc_speed_limit1: dim=1 누적기
        lanes: (L,P,12) 또는 None
        lanes_speed_limit: (L,) 또는 None
        lanes_has_speed_limit: (L,) 또는 None
        eps: 0 판정 기준

    Returns:
        None
    """
    if lanes is None:
        return
    ln = np.asarray(lanes)
    if ln.ndim != 3 or int(ln.shape[2]) != 12:
        return

    # valid_point: (L,P)
    valid_point = _valid_mask_by_first_k_nonzero(ln, k=8, eps=eps)

    if np.any(valid_point):
        xy = ln[..., 0:2][valid_point]  # (K,2)
        vec = ln[..., 2:8][valid_point]  # (K,6)
        if xy.size > 0:
            acc_lanes_xy2.update(xy.reshape((-1, 2)))
        if vec.size > 0:
            acc_lanes_vec6.update(vec.reshape((-1, 6)))

    # speed limit은 lane 단위
    if lanes_speed_limit is None:
        return
    sl = np.asarray(lanes_speed_limit)
    if sl.ndim != 1 or int(sl.shape[0]) != int(ln.shape[0]):
        return

    lane_valid = valid_point.any(axis=1)  # (L,)
    mask = lane_valid.astype(bool)

    if lanes_has_speed_limit is not None:
        hs = np.asarray(lanes_has_speed_limit)
        if hs.ndim == 1 and int(hs.shape[0]) == int(sl.shape[0]):
            mask = mask & hs.astype(bool)

    if np.any(mask):
        sel = sl[mask].reshape((-1, 1))  # (K,1)
        acc_speed_limit1.update(sel)


def _update_pointset_xy_stats(*, acc_xy2: RunningStats, points: Optional[np.ndarray], eps: float) -> None:
    """(num, ..., 2) 형태 점 묶음 key의 (x,y) 통계를 업데이트합니다.

    무효 객체 규칙:
      - 객체 하나의 모든 점(x,y)이 전부 0이면 그 객체는 무효

    Args:
        acc_xy2: dim=2 누적기
        points: shape (N, ..., 2) 또는 None
        eps: 0 판정 기준

    Returns:
        None
    """
    if points is None:
        return
    p = np.asarray(points)
    if p.ndim < 2 or int(p.shape[-1]) != 2:
        return

    # valid_obj: (N,)
    # axis는 1..ndim-1 전체(좌표 2차원 포함)
    reduce_axes = tuple(range(1, p.ndim))
    valid_obj = (np.abs(p) > float(eps)).any(axis=reduce_axes)

    if not np.any(valid_obj):
        return

    pts = p[valid_obj].reshape((-1, 2))  # (K,2)
    if pts.size == 0:
        return
    acc_xy2.update(pts)


# ============================================================
# 5) 파일 1개 처리 -> 작은 통계 묶음(parts) 반환
# ============================================================
def _calculate_statistics_from_npz_path(npz_path: str, *, eps: float) -> Tuple[bool, str, Dict[str, Tuple[int, List[float], List[float]]]]:
    """npz 파일 1개에서 필요한 통계를 계산해 parts로 반환합니다.

    parts는 group 이름별로 (count, sum_list, sumsq_list)만 담습니다.
    """
    if not os.path.exists(npz_path):
        return False, f"missing: {npz_path}", {}

    # 필요한 key들만 로드(없으면 dict에 없음)
    keys_to_load = [
        EGO_PAST_KEY,
        EGO_FUTURE_KEY,
        NBR_PAST_KEY,
        NBR_FUTURE_KEY,
        PAST_CONTROL_KEY,
        FUTURE_CONTROL_KEY,
        STATIC_OBJECTS_KEY,
        LANES_KEY,
        LANES_SPEED_LIMIT_KEY,
        LANES_HAS_SPEED_LIMIT_KEY,
        *list(POINTSET_KEYS),
    ]

    try:
        npz_data = _load_npz_subset_as_dict(npz_path, keys_to_load)
    except Exception as e:
        return False, f"load_failed: {type(e).__name__}: {e}", {}

    # 그룹 누적기들
    acc_agent_dyn6 = RunningStats(dim=6)
    acc_agent_size2 = RunningStats(dim=2)
    acc_control3 = RunningStats(dim=3)

    acc_static6 = RunningStats(dim=6)

    acc_lanes_xy2 = RunningStats(dim=2)
    acc_lanes_vec6 = RunningStats(dim=6)
    acc_speed_limit1 = RunningStats(dim=1)

    acc_points: Dict[str, RunningStats] = {k: RunningStats(dim=2) for k in POINTSET_KEYS}

    # (1) agent
    ego_past = npz_data.get(EGO_PAST_KEY, None)
    ego_future = npz_data.get(EGO_FUTURE_KEY, None)
    nbr_past = npz_data.get(NBR_PAST_KEY, None)
    nbr_future = npz_data.get(NBR_FUTURE_KEY, None)

    ego_past_valid, ego_future_valid, nbr_past_valid, nbr_future_valid = _update_agent_stats(
        acc_agent_dyn6=acc_agent_dyn6,
        acc_agent_size2=acc_agent_size2,
        ego_past=ego_past,
        ego_future=ego_future,
        nbr_past=nbr_past,
        nbr_future=nbr_future,
        eps=eps,
    )

    # (2) control
    _update_control_stats(
        acc_control3=acc_control3,
        past_control=npz_data.get(PAST_CONTROL_KEY, None),
        future_control=npz_data.get(FUTURE_CONTROL_KEY, None),
        ego_past_valid=ego_past_valid,
        ego_future_valid=ego_future_valid,
        nbr_past_valid=nbr_past_valid,
        nbr_future_valid=nbr_future_valid,
    )

    # (3) static_objects
    _update_static_objects_stats(
        acc_static6=acc_static6,
        static_objects=npz_data.get(STATIC_OBJECTS_KEY, None),
        eps=eps,
    )

    # (4) lanes + speed limit
    _update_lanes_stats(
        acc_lanes_xy2=acc_lanes_xy2,
        acc_lanes_vec6=acc_lanes_vec6,
        acc_speed_limit1=acc_speed_limit1,
        lanes=npz_data.get(LANES_KEY, None),
        lanes_speed_limit=npz_data.get(LANES_SPEED_LIMIT_KEY, None),
        lanes_has_speed_limit=npz_data.get(LANES_HAS_SPEED_LIMIT_KEY, None),
        eps=eps,
    )

    # (5) point sets
    for k in POINTSET_KEYS:
        _update_pointset_xy_stats(acc_xy2=acc_points[k], points=npz_data.get(k, None), eps=eps)

    parts: Dict[str, Tuple[int, List[float], List[float]]] = {
        "agent_dyn6": acc_agent_dyn6.to_parts(),
        "agent_size2": acc_agent_size2.to_parts(),
        "control3": acc_control3.to_parts(),
        "static6": acc_static6.to_parts(),
        "lanes_xy2": acc_lanes_xy2.to_parts(),
        "lanes_vec6": acc_lanes_vec6.to_parts(),
        "lanes_speed1": acc_speed_limit1.to_parts(),
    }
    for k in POINTSET_KEYS:
        parts[f"points_{k}"] = acc_points[k].to_parts()

    return True, "ok", parts


def _worker_calculate_statistics_one_fname(fname: str) -> Tuple[str, bool, str, Dict[str, Tuple[int, List[float], List[float]]]]:
    """멀티프로세스 워커: 파일 1개 통계 parts를 계산해 반환합니다."""
    cfg = _WORKER_CONFIG
    dataset_dir = str(cfg["dataset_dir"])
    eps = float(cfg["eps"])

    npz_path = os.path.join(dataset_dir, str(fname))
    ok, msg, parts = _calculate_statistics_from_npz_path(npz_path, eps=eps)
    return str(fname), bool(ok), str(msg), parts


# ============================================================
# 6) 최종 mean/std 만들기(각 key 차원에 맞춰서)
# ============================================================
def _safe_mean_std(stats: RunningStats) -> Tuple[List[float], List[float]]:
    """RunningStats에서 mean/std를 list로 뽑습니다."""
    mean, std = stats.compute_mean_std()
    return [float(x) for x in mean.tolist()], [float(x) for x in std.tolist()]


def _build_output_mean_std(
    *,
    global_stats: Dict[str, RunningStats],
    files_total: int,
    files_failed: int,
    eps: float,
) -> Dict[str, Any]:
    """누적 통계를 key별 mean/std 형식으로 정리합니다."""
    # 그룹별 mean/std
    agent_dyn_mean, agent_dyn_std = _safe_mean_std(global_stats["agent_dyn6"])  # len 6
    agent_size_mean, agent_size_std = _safe_mean_std(global_stats["agent_size2"])  # len 2
    control_mean, control_std = _safe_mean_std(global_stats["control3"])  # len 3
    static_mean6, static_std6 = _safe_mean_std(global_stats["static6"])  # len 6
    lanes_xy_mean, lanes_xy_std = _safe_mean_std(global_stats["lanes_xy2"])  # len 2
    lanes_vec_mean, lanes_vec_std = _safe_mean_std(global_stats["lanes_vec6"])  # len 6
    lanes_speed_mean1, lanes_speed_std1 = _safe_mean_std(global_stats["lanes_speed1"])  # len 1

    # (A) agent 11차원 mean/std 만들기
    # 0~5: dyn6, 6~7: size2, 8~10: one-hot(정규화 안 함 => mean=0,std=1)
    agent_mean_11 = agent_dyn_mean + agent_size_mean + [0.0, 0.0, 0.0]
    agent_std_11 = agent_dyn_std + agent_size_std + [1.0, 1.0, 1.0]

    # (B) static_objects 10차원: 0~5 연속값, 6~9 one-hot
    static_mean_10 = static_mean6 + [0.0, 0.0, 0.0, 0.0]
    static_std_10 = static_std6 + [1.0, 1.0, 1.0, 1.0]

    # (C) lanes 12차원: 0~1 좌표, 2~7 벡터, 8~11 one-hot
    lanes_mean_12 = lanes_xy_mean + lanes_vec_mean + [0.0, 0.0, 0.0, 0.0]
    lanes_std_12 = lanes_xy_std + lanes_vec_std + [1.0, 1.0, 1.0, 1.0]

    # (D) point sets(2차원)
    point_out: Dict[str, Any] = {}
    for k in POINTSET_KEYS:
        m, s = _safe_mean_std(global_stats[f"points_{k}"])
        point_out[k] = {
            "mean": m,
            "std": s,
            "count": int(global_stats[f"points_{k}"].count),
        }

    out: Dict[str, Any] = {
        "meta": {
            "files_total": int(files_total),
            "files_failed": int(files_failed),
            "eps_for_zero_check": float(eps),
            "note": (
                "One-hot channels are not normalized; we store mean=0 and std=1 for them."
            ),
        },
        # agent 계열 4개 key는 같은 mean/std를 쓰도록 출력
        EGO_PAST_KEY: {"mean": agent_mean_11, "std": agent_std_11},
        EGO_FUTURE_KEY: {"mean": agent_mean_11, "std": agent_std_11},
        NBR_PAST_KEY: {"mean": agent_mean_11, "std": agent_std_11},
        NBR_FUTURE_KEY: {"mean": agent_mean_11, "std": agent_std_11},
        # control도 past/future를 합쳐서 하나로 계산한 값을 둘 다에 적용
        PAST_CONTROL_KEY: {"mean": control_mean, "std": control_std},
        FUTURE_CONTROL_KEY: {"mean": control_mean, "std": control_std},
        STATIC_OBJECTS_KEY: {"mean": static_mean_10, "std": static_std_10},
        LANES_KEY: {"mean": lanes_mean_12, "std": lanes_std_12},
        LANES_SPEED_LIMIT_KEY: {"mean": float(lanes_speed_mean1[0]), "std": float(lanes_speed_std1[0])},
        # 점 묶음들
        "point_sets": point_out,
        # count도 같이(디버깅/검증용)
        "counts": {
            "agent_dyn6_frames": int(global_stats["agent_dyn6"].count),
            "agent_size2_tracks": int(global_stats["agent_size2"].count),
            "control3_segments": int(global_stats["control3"].count),
            "static6_objects": int(global_stats["static6"].count),
            "lanes_points": int(global_stats["lanes_xy2"].count),
            "lanes_speed_limit_lanes": int(global_stats["lanes_speed1"].count),
        },
    }
    return out


# ============================================================
# 7) 실행(only_calculate_statistics)
# ============================================================
def _run_only_calculate_statistics(
    *,
    dataset_dir: str,
    train_json: str,
    limit: int,
    workers_arg: int,
    eps: float,
    out_json: Optional[str],
) -> None:
    """npz를 수정하지 않고 mean/std만 계산합니다."""
    file_names = _load_training_file_list(train_json)
    if int(limit) > 0:
        file_names = file_names[: int(limit)]

    total_files = len(file_names)
    if total_files <= 0:
        print("done. files_total=0, files_failed=0")
        return

    workers: int = int(workers_arg) if int(workers_arg) > 0 else _get_auto_worker_count(total_files)
    workers = max(1, min(int(workers), int(total_files)))

    # global 누적기들
    global_stats: Dict[str, RunningStats] = {
        "agent_dyn6": RunningStats(dim=6),
        "agent_size2": RunningStats(dim=2),
        "control3": RunningStats(dim=3),
        "static6": RunningStats(dim=6),
        "lanes_xy2": RunningStats(dim=2),
        "lanes_vec6": RunningStats(dim=6),
        "lanes_speed1": RunningStats(dim=1),
    }
    for k in POINTSET_KEYS:
        global_stats[f"points_{k}"] = RunningStats(dim=2)

    fail_count = 0
    start_time_s = time.monotonic()
    last_print_time_s = start_time_s

    # -------------------
    # (1) 순차 처리
    # -------------------
    if workers <= 1:
        pbar = tqdm(file_names, desc="only_calculate_statistics")
        for idx, fname in enumerate(pbar, start=1):
            npz_path = os.path.join(dataset_dir, str(fname))
            ok, msg, parts = _calculate_statistics_from_npz_path(npz_path, eps=float(eps))
            if not ok:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")
            else:
                for group_name, (cnt, s, ss) in parts.items():
                    if group_name in global_stats:
                        global_stats[group_name].merge_from_parts(cnt, s, ss)
                    else:
                        # points_XXX 형태
                        if group_name.startswith("points_") and group_name in global_stats:
                            global_stats[group_name].merge_from_parts(cnt, s, ss)

            last_print_time_s = _maybe_print_progress_every_5_min(
                start_time_s=start_time_s,
                last_print_time_s=last_print_time_s,
                processed=idx,
                total=total_files,
                interval_s=300.0,
            )

        out = _build_output_mean_std(
            global_stats=global_stats,
            files_total=total_files,
            files_failed=fail_count,
            eps=float(eps),
        )
        _emit_output(out, out_json)
        return

    # -------------------
    # (2) 멀티프로세스 처리
    # -------------------
    worker_config: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "eps": float(eps),
    }

    ctx = mp.get_context()
    processed_count = 0

    pbar = tqdm(total=total_files, desc=f"only_calculate_statistics (workers={workers})")
    pool = ctx.Pool(
        processes=workers,
        initializer=_init_worker_process,
        initargs=(worker_config,),
    )

    try:
        chunksize = 4
        for fname, ok, msg, parts in pool.imap_unordered(_worker_calculate_statistics_one_fname, file_names, chunksize=chunksize):
            processed_count += 1
            pbar.update(1)

            if not ok:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")
            else:
                for group_name, (cnt, s, ss) in parts.items():
                    if group_name in global_stats:
                        global_stats[group_name].merge_from_parts(cnt, s, ss)
                    else:
                        if group_name.startswith("points_") and group_name in global_stats:
                            global_stats[group_name].merge_from_parts(cnt, s, ss)

            last_print_time_s = _maybe_print_progress_every_5_min(
                start_time_s=start_time_s,
                last_print_time_s=last_print_time_s,
                processed=processed_count,
                total=total_files,
                interval_s=300.0,
            )

        pool.close()
        pool.join()

    except KeyboardInterrupt:
        tqdm.write("[STOP] Ctrl+C")
        pool.terminate()
        pool.join()

    finally:
        pbar.close()

    out = _build_output_mean_std(
        global_stats=global_stats,
        files_total=total_files,
        files_failed=fail_count,
        eps=float(eps),
    )
    _emit_output(out, out_json)


def _emit_output(out: Dict[str, Any], out_json: Optional[str]) -> None:
    """결과를 콘솔에 출력하고, out_json이 있으면 파일로 저장합니다."""
    print("========== data_mean_std_maker result ==========")
    meta = out.get("meta", {})
    print(f"files_total={meta.get('files_total')}, files_failed={meta.get('files_failed')}")
    print(f"eps_for_zero_check={meta.get('eps_for_zero_check')}")

    text = json.dumps(out, ensure_ascii=False, indent=2)

    if out_json is not None and str(out_json).strip() != "":
        out_path = str(out_json)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
        print(f"[SAVED] {out_path}")

    print(text)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="npz를 수정하지 않고(mean/std만) 통계를 계산합니다."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="npz 파일들이 들어있는 폴더(내부 폴더 없음)",
    )
    parser.add_argument(
        "--train_json",
        type=str,
        required=True,
        help="학습에 쓰는 npz 파일명 리스트(json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0이면 전체, 양수면 앞에서 N개만",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="프로세스 개수. 0이면 자동(최대 16), 1이면 순차",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="0 판정 기준(무효점 판단용)",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="",
        help="결과 json 저장 경로(비우면 저장 안 함)",
    )
    parser.add_argument(
        "--only_calculate_statistics",
        action="store_true",
        help="이 옵션이 있어야 실행됩니다(안전장치).",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not bool(getattr(args, "only_calculate_statistics", False)):
        print("이 스크립트는 npz를 수정하지 않고 mean/std만 계산합니다.")
        print("실행하려면 --only_calculate_statistics 옵션을 넣어 주세요.")
        return

    _run_only_calculate_statistics(
        dataset_dir=str(args.dataset_dir),
        train_json=str(args.train_json),
        limit=int(args.limit),
        workers_arg=int(args.workers),
        eps=float(args.eps),
        out_json=(str(args.out_json) if str(args.out_json).strip() != "" else None),
    )


if __name__ == "__main__":
    main()

"""
예시 실행:

python data_mean_std_maker.py \
  --dataset_dir /workspace/local_shards_v_world \
  --train_json /workspace/local_shards_v_world/diffusion_planner_training.json \
  --only_calculate_statistics \
  --workers 28 \
  --out_json /workspace/mean_std.json

"""