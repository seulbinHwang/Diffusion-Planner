from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union, Optional
import multiprocessing as mp

import zipfile
import shutil

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
import time

ArrayF = NDArray[np.floating]

# --------------------
# [1] 상수 교체
# --------------------
PAST_CONTROL_KEY = "past_seg_control_gt_3_dim"
FUTURE_CONTROL_KEY = "future_seg_control_gt_3_dim"
_CONTROL_KEYS = (PAST_CONTROL_KEY, FUTURE_CONTROL_KEY)

_WORKER_CONFIG: Dict[str, Any] = {}

class ControlStatsAccumulator:
    """(v_x^b, v_y^b, yaw_rate) mean/std 계산을 위한 누적기입니다.

    원본 값을 전부 저장하지 않고, 아래 3가지만 누적합니다.
      1) count: 유효 샘플 개수
      2) sum:   값의 합 (채널 3개)
      3) sumsq: 값^2의 합 (채널 3개)

    Notes:
        - update()에 들어오는 values는 shape (M, 3) 이어야 합니다.
        - 누적은 float64로 합니다. (오차가 덜 쌓이도록)
    """

    def __init__(self) -> None:
        self.count: int = 0
        self.sum: NDArray[np.float64] = np.zeros((3,), dtype=np.float64)    # (3,)
        self.sumsq: NDArray[np.float64] = np.zeros((3,), dtype=np.float64)  # (3,)

    def update(self, values: np.ndarray) -> None:
        """유효한 값들을 누적합니다.

        Args:
            values (np.ndarray): shape (M, 3)
                [v_x^b, v_y^b, yaw_rate] 값들.

        Returns:
            None
        """
        v = np.asarray(values)
        if v.size == 0:
            return
        if v.ndim != 2 or int(v.shape[1]) != 3:
            raise ValueError(f"values는 (M,3) 이어야 합니다. got shape={v.shape}")

        # v: (M,3)
        v64 = v.astype(np.float64, copy=False)

        # (M,) 행 단위로 finite 체크 (3개 채널이 모두 유한한 경우만 사용)
        finite_row = np.isfinite(v64).all(axis=1)
        if not np.any(finite_row):
            return

        v_sel = v64[finite_row]  # (K,3)
        if v_sel.size == 0:
            return

        self.count += int(v_sel.shape[0])
        self.sum += v_sel.sum(axis=0)             # (3,)
        self.sumsq += (v_sel * v_sel).sum(axis=0) # (3,)

    def merge(self, other: "ControlStatsAccumulator") -> None:
        """다른 누적기의 값을 합칩니다.

        Args:
            other (ControlStatsAccumulator): 합칠 대상.

        Returns:
            None
        """
        if not isinstance(other, ControlStatsAccumulator):
            raise ValueError(f"other는 ControlStatsAccumulator 이어야 합니다. got={type(other)}")
        self.count += int(other.count)
        self.sum += np.asarray(other.sum, dtype=np.float64)
        self.sumsq += np.asarray(other.sumsq, dtype=np.float64)

    def merge_from_parts(
        self,
        *,
        count: int,
        sum_list: Sequence[float],
        sumsq_list: Sequence[float],
    ) -> None:
        """(count, sum, sumsq) 형태의 작은 값 묶음을 합칩니다.

        멀티프로세스에서 워커가 보내주는 값 합치기에 사용합니다.

        Args:
            count (int): 유효 샘플 개수
            sum_list (Sequence[float]): shape (3,)
            sumsq_list (Sequence[float]): shape (3,)

        Returns:
            None
        """
        self.count += int(count)
        self.sum += np.asarray(sum_list, dtype=np.float64).reshape((3,))
        self.sumsq += np.asarray(sumsq_list, dtype=np.float64).reshape((3,))

    def compute_mean_std(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """누적된 값으로 mean/std를 계산합니다.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - mean: shape (3,)
                - std:  shape (3,)
        """
        if int(self.count) <= 0:
            nan3 = np.full((3,), np.nan, dtype=np.float64)
            return nan3, nan3

        n = float(self.count)
        mean = self.sum / n                      # (3,)
        var = (self.sumsq / n) - (mean * mean)   # (3,)

        # 수치 오차로 -0.xx 가 나오는 것을 방지
        var = np.maximum(var, 0.0)
        std = np.sqrt(var)
        return mean, std


def _get_auto_worker_count(total_files: int) -> int:
    """자동으로 사용할 프로세스 개수를 정합니다.

    Args:
        total_files (int): 처리할 파일 개수.

    Returns:
        int:
            사용할 프로세스 개수.
            - CPU 코어 수를 기준으로 하되, 너무 큰 값은 제한합니다.
            - 파일 개수보다 많게 잡지 않습니다.
    """
    cpu = int(os.cpu_count() or 1)

    # 너무 과하게 늘리면 디스크가 버거울 수 있어서 상한을 둡니다.
    upper = min(16, cpu)

    if int(total_files) <= 0:
        return 1
    return max(1, min(int(total_files), int(upper)))


def _init_worker_process(config: Dict[str, Any]) -> None:
    """각 워커 프로세스 시작 시 설정을 저장합니다.

    Args:
        config (Dict[str, Any]): 워커가 공통으로 쓸 설정 값들.
    """
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)


def _worker_process_one_fname(fname: str) -> Tuple[str, bool, str, bool]:
    """워커 프로세스에서 npz 파일 1개를 처리합니다.

    Args:
        fname (str): train_json에 들어있는 파일명(예: "abc.npz")

    Returns:
        Tuple[str, bool, str, bool]:
            - fname: 입력 파일명
            - ok: 성공(True) / 실패(False)
            - msg: "ok" / "skip(...)" / 실패 원인 메시지
            - skipped: 이미 처리 완료라서 스킵이면 True, 아니면 False
    """
    cfg = _WORKER_CONFIG
    dataset_dir = str(cfg["dataset_dir"])

    dt = float(cfg["dt"])
    overwrite = bool(cfg["overwrite"])
    compress = bool(cfg["compress"])
    add_sample_keys = bool(cfg["add_sample_keys"])
    overwrite_sample_keys = bool(cfg["overwrite_sample_keys"])
    predicted_neighbor_num = int(cfg["predicted_neighbor_num"])
    eval_method = str(cfg["eval_method"])
    use_agent_route_lane_order = bool(cfg["use_agent_route_lane_order"])

    npz_path = os.path.join(dataset_dir, str(fname))

    try:
        keys: Optional[set[str]] = None
        try:
            keys = _read_npz_key_set(npz_path)
        except Exception:
            keys = None

        if _is_already_processed_npz(
            npz_path,
            overwrite=overwrite,
            add_sample_keys=add_sample_keys,
            overwrite_sample_keys=overwrite_sample_keys,
            use_agent_route_lane_order=use_agent_route_lane_order,
            existing_keys=keys,
        ):
            return str(fname), True, "skip(already processed)", True

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
            existing_keys=keys,
        )
        return str(fname), bool(ok), str(msg), False

    except Exception as e:
        return str(fname), False, f"{type(e).__name__}: {e}", False

def _load_npz_subset_as_dict(npz_path: str, keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """npz에서 '필요한 key들만' 골라서 dict로 읽습니다.

    Args:
        npz_path (str): npz 파일 경로.
        keys (Sequence[str]): 읽고 싶은 key 목록.

    Returns:
        Dict[str, np.ndarray]:
            - key -> np.ndarray
            - npz에 없는 key는 결과 dict에 포함되지 않습니다.

    Notes:
        - 기존 `_read_npz_as_dict()`처럼 파일 전체를 다 읽지 않습니다.
        - 기본은 allow_pickle=False로 읽고,
          object 배열 때문에 실패하면 allow_pickle=True로 한 번 더 시도합니다.
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


def _atomic_update_npz_by_copy_and_append(
    npz_path: str,
    new_arrays: Mapping[str, np.ndarray],
    *,
    compress: bool,
    compresslevel: int = 1,
) -> None:
    """원본 npz를 그대로 복사한 뒤, 새 key들만 '추가로' 붙여서 원자적으로 교체합니다.

    이 방식의 핵심:
        - 기존 npz의 모든 배열을 다시 저장(재압축)하지 않습니다.
        - 실제로 새로 만든 배열들만 zip 끝에 추가로 기록합니다.
        - 마지막에 os.replace로 한 번에 교체합니다.

    Args:
        npz_path (str): 대상 npz 경로.
        new_arrays (Mapping[str, np.ndarray]): 추가/갱신할 key -> 배열.
        compress (bool): True면 새로 추가하는 entry는 deflate 압축으로 저장합니다.
        compresslevel (int): 압축 강도(낮을수록 보통 더 빠름). 기본 1.

    Notes:
        - overwrite 모드에서 기존 key를 "갱신"하려면 zip에 같은 이름을 다시 쓰게 됩니다.
          이 스크립트에서는 overwrite 계열일 때는 전체 재저장(full rewrite)로 처리해서,
          기본 모드에서는 중복 entry가 생기지 않게 설계합니다.
    """
    tmp_path = npz_path + ".tmp"
    try:
        # 1) 원본을 그대로 복사 (기존 데이터는 재압축/재저장 안 함)
        shutil.copyfile(npz_path, tmp_path)

        compression = zipfile.ZIP_DEFLATED if bool(compress) else zipfile.ZIP_STORED

        zip_kwargs: Dict[str, Any] = {"mode": "a", "compression": compression}
        if bool(compress):
            zip_kwargs["compresslevel"] = int(max(0, int(compresslevel)))

        def _append_with_kwargs(kwargs: Dict[str, Any]) -> None:
            with zipfile.ZipFile(tmp_path, **kwargs) as zf:
                # key 순서를 고정하면, 실행마다 결과가 더 일정해집니다(보기/검증용).
                for key in sorted(new_arrays.keys()):
                    arr = np.asarray(new_arrays[key])
                    # npz 내부에서는 "<key>.npy" 형태로 저장됩니다.
                    with zf.open(f"{key}.npy", mode="w") as f:
                        np.save(f, arr, allow_pickle=False)

        try:
            _append_with_kwargs(zip_kwargs)
        except TypeError:
            # 일부 파이썬/환경에서 compresslevel을 지원하지 않을 수 있어 안전 처리
            zip_kwargs.pop("compresslevel", None)
            _append_with_kwargs(zip_kwargs)

        # 2) 원자적 교체
        os.replace(tmp_path, npz_path)

    except BaseException:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def _collect_npz_arrays_from_sample(
    sample: Mapping[str, Any],
    *,
    base_output_keys: Sequence[str],
    alias_output_keys: Sequence[str],
    overwrite: bool,
    existing_keys: set[str],
) -> Dict[str, np.ndarray]:
    """sample dict에서 '실제로 npz에 저장할 파생 key'만 골라 dict로 뽑습니다.

    Args:
        sample (Mapping[str, Any]): __getitem__ 형태로 만든 sample dict.
        base_output_keys (Sequence[str]): 원본에서 읽어서 채운 key들(rename 반영).
        alias_output_keys (Sequence[str]): rename으로 새로 생긴 key들(예: planner_future_11_dim).
        overwrite (bool): True면 이미 존재하는 파생키도 덮어쓴다고 가정합니다.
        existing_keys (set[str]): 현재 npz에 이미 들어있는 key 이름 집합.

    Returns:
        Dict[str, np.ndarray]: npz에 추가로 저장할 key -> array dict.
    """
    base_set = set(map(str, base_output_keys))
    alias_set = set(map(str, alias_output_keys))

    out: Dict[str, np.ndarray] = {}
    for key, value in sample.items():
        k = str(key)

        # 원본에서 읽어온 값은 저장하지 않음(중복 방지).
        # 단, alias key는 예외로 저장.
        if (k in base_set) and (k not in alias_set):
            continue

        arr = _to_np_array_or_skip(value)
        if arr is None:
            continue

        # overwrite가 아니면: 이미 있는 키는 "추가 저장"하지 않아서 zip 중복을 방지
        if (not bool(overwrite)) and (k in existing_keys):
            continue

        out[k] = arr

    return out


def _read_npz_key_set(npz_path: str) -> set[str]:
    """npz 파일에 들어있는 key 이름 목록만 빠르게 읽습니다.

    Args:
        npz_path (str): npz 파일 경로.

    Returns:
        set[str]: npz 내부 key 이름 집합.

    Notes:
        - 배열 데이터(값)는 읽지 않습니다.
        - zip 파일 목록만 확인합니다.
    """
    with zipfile.ZipFile(npz_path, "r") as zf:
        names = zf.namelist()

    keys: set[str] = set()
    for name in names:
        if name.endswith(".npy"):
            keys.add(name[:-4])  # ".npy" 제거
    return keys


# --------------------
# [2] 완료 판정 expected key 교체
# --------------------
def _build_expected_keys_for_done(
    existing_keys: set[str],
    *,
    add_sample_keys: bool,
    use_agent_route_lane_order: bool,
) -> set[str]:
    """현재 옵션에서 '처리 완료'라고 보기 위해 필요한 key 집합을 만듭니다.

    기준:
        - control key 2개는 항상 필요(past/future)
        - sample 파생키 모드면:
          - 항상 추가되는 key + (원본에 해당 입력 key가 있을 때만) 추가되는 key를 포함

    Args:
        existing_keys (set[str]):
            현재 npz에 있는 key 이름들.
        add_sample_keys (bool):
            sample 파생키를 저장하는 모드인지 여부(--skip_sample_keys 반대).
        use_agent_route_lane_order (bool):
            agent_route_lane_order를 sample에 포함하는 모드인지 여부.

    Returns:
        set[str]: 완료 판정에 필요한 key 이름 집합.
    """
    expected: set[str] = {PAST_CONTROL_KEY, FUTURE_CONTROL_KEY}

    if not bool(add_sample_keys):
        return expected

    # sample 파생키 모드에서 "항상" 추가되는 것들
    expected.update(
        {
            "scenario_id",
            "planner_future_11_dim",  # ego_future_gt_11_dim의 alias 저장
        }
    )

    # --- validity 관련(입력 key가 있을 때만 저장되는 것들) ---
    validity_by_source: Dict[str, List[str]] = {
        "ego_agent_past": ["ego_agent_past_is_valid"],
        "ego_future_gt_11_dim": ["ego_future_gt_is_valid"],
        "neighbor_agents_past": ["neighbor_agents_past_is_valid", "neighbor_agents_is_valid"],
        "neighbor_future_gt_11_dim": ["neighbor_future_gt_is_valid"],
        "stop_sign_points": ["stop_sign_is_valid"],
        "crosswalk_points": ["crosswalk_is_valid"],
        "lanes": ["lanes_len_is_valid", "lanes_is_valid"],
        "static_objects": ["static_objects_is_valid"],
        "route_lanes": ["route_lanes_len_is_valid", "route_lanes_is_valid"],
        "speed_bump_points": ["speed_bump_is_valid"],
        "driveway_points": ["driveway_is_valid"],
        "road_edge": ["road_edge_is_valid"],
    }
    for src_key, out_keys in validity_by_source.items():
        if src_key in existing_keys:
            expected.update(out_keys)

    if bool(use_agent_route_lane_order) and ("agent_route_lane_order" in existing_keys):
        expected.add("agent_route_lane_order_is_valid")

    # --- near split 관련(입력 key가 있을 때만 저장되는 것들) ---
    if "neighbor_agents_past" in existing_keys:
        expected.update(
            {
                "near_agents_past",
                "non_near_agents_past",
                "near_agents_past_is_valid",
                "non_near_agents_past_is_valid",
                "near_agents_is_valid",
                "non_near_agents_is_valid",
            }
        )

    if "neighbor_future_gt_11_dim" in existing_keys:
        expected.update(
            {
                "near_future_gt_is_valid",
                "non_near_future_gt_is_valid",
            }
        )

    if "neighbor_future_gt_3_dim" in existing_keys:
        expected.add("near_future_gt_3_dim")

    # --- gt_4_dim 관련(입력 key가 있을 때만 저장되는 것들) ---
    if ("ego_future_gt_3_dim" in existing_keys) and ("ego_future_gt_11_dim" in existing_keys):
        expected.add("ego_future_gt_4_dim")

    if ("neighbor_future_gt_3_dim" in existing_keys) and ("neighbor_future_gt_11_dim" in existing_keys):
        expected.add("near_future_gt_4_dim")

    return expected

# --------------------
# [3] past/future control 생성 함수 추가(기존 past_future 관련 함수는 더 이상 사용 안 해도 됨)
# --------------------
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


def _is_already_processed_npz(
    npz_path: str,
    *,
    overwrite: bool,
    add_sample_keys: bool,
    overwrite_sample_keys: bool,
    use_agent_route_lane_order: bool,
    existing_keys: Optional[set[str]] = None,
) -> bool:
    """이 npz를 '이미 완료'로 보고 바로 스킵해도 되는지 판단합니다.

    Args:
        npz_path (str): npz 파일 경로.
        overwrite (bool): True면 항상 다시 처리해야 하므로 False 반환.
        add_sample_keys (bool): sample 파생키 모드인지 여부.
        overwrite_sample_keys (bool): True면 sample 파생키도 다시 계산해야 하므로 False 반환.
        use_agent_route_lane_order (bool): agent_route_lane_order 포함 모드인지 여부.
        existing_keys (Optional[set[str]]): 이미 읽어둔 key set이 있으면 재사용합니다.

    Returns:
        bool: 이미 완료 상태면 True, 아니면 False.
    """
    if not os.path.exists(npz_path):
        return False
    if bool(overwrite):
        return False
    if bool(add_sample_keys) and bool(overwrite_sample_keys):
        return False

    keys: set[str]
    if existing_keys is not None:
        keys = set(existing_keys)
    else:
        try:
            keys = _read_npz_key_set(npz_path)
        except Exception:
            return False

    expected = _build_expected_keys_for_done(
        keys,
        add_sample_keys=bool(add_sample_keys),
        use_agent_route_lane_order=bool(use_agent_route_lane_order),
    )
    return expected.issubset(keys)



def _format_seconds_to_hh_mm(seconds: float) -> str:
    """초 단위 시간을 '시간:분' 문자열(HH:MM)로 바꿉니다.

    Args:
        seconds (float): 초 단위 시간(음수면 0으로 처리).

    Returns:
        str: 'HH:MM' 형태 문자열.
    """
    s = float(seconds)
    if (not np.isfinite(s)) or s < 0.0:
        s = 0.0

    # 분 단위로 반올림(보기용)
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
    """5분(기본)마다 진행률/경과/남은시간(예상)을 출력합니다.

    출력 내용:
        - 진행률(%)
        - 경과 시간(HH:MM)
        - 남은 시간(예상, HH:MM)

    Args:
        start_time_s (float): 시작 시각(time.monotonic()).
        last_print_time_s (float): 마지막 출력 시각(time.monotonic()).
        processed (int): 지금까지 처리한 파일 수.
        total (int): 전체 파일 수.
        interval_s (float): 출력 간격(초). 기본 300초(=5분).

    Returns:
        float: 업데이트된 마지막 출력 시각(출력했으면 now, 아니면 기존값).
    """
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


def _build_past_future_seg_control_gt_and_seg_valid_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)
    ego_future_gt_11_dim: ArrayF,  # (Tf,11)
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    neighbor_future_gt_11_dim: ArrayF,  # (N,Tf,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> Tuple[ArrayF, NDArray[np.bool_]]:
    """(과거~미래) 구간 제어와 seg_valid를 함께 계산합니다.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            controls: shape (1+N, (Tp-1)+Tf, 3)
                마지막 3은 (v_x^b, v_y^b, yaw_rate)
            seg_valid: shape (1+N, (Tp-1)+Tf) bool
                구간이 유효하면 True

    Notes:
        - 통계 모드에서는 seg_valid=True인 controls만 모아서 mean/std를 계산합니다.
        - 여기서는 controls를 '그대로' 반환합니다(유효하지 않은 구간을 0으로 만들지 않음).
          (필요하면 호출자가 seg_valid로 걸러서 쓰면 됩니다.)
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

    Tp = int(ego_past.shape[0])   # (past+cur) = 21
    Tf = int(ego_fut11.shape[0])  # 80
    N = int(nbr_past.shape[0])

    if Tp <= 1:
        raise ValueError(f"Tp는 최소 2 이상이어야 합니다. got Tp={Tp}")
    if int(nbr_past.shape[1]) != Tp:
        raise ValueError(f"neighbor_agents_past의 Tp가 ego와 같아야 합니다. got {nbr_past.shape[1]} vs {Tp}")
    if int(nbr_fut11.shape[0]) != N:
        raise ValueError(f"neighbor_future_gt_11_dim의 N이 neighbor_agents_past와 같아야 합니다. got {nbr_fut11.shape[0]} vs {N}")
    if int(nbr_fut11.shape[1]) != Tf:
        raise ValueError(f"neighbor_future_gt_11_dim의 Tf가 ego_future와 같아야 합니다. got {nbr_fut11.shape[1]} vs {Tf}")

    dt = float(dt)
    if (not np.isfinite(dt)) or dt <= 0.0:
        raise ValueError(f"dt는 0보다 큰 유한한 값이어야 합니다. got dt={dt}")

    # float32로 정리 (삼각함수/나눗셈 안정)
    ego_past = ego_past.astype(np.float32 if ego_past.dtype.kind != "f" else ego_past.dtype, copy=False)
    ego_fut11 = ego_fut11.astype(ego_past.dtype, copy=False)
    nbr_past = nbr_past.astype(ego_past.dtype, copy=False)
    nbr_fut11 = nbr_fut11.astype(ego_past.dtype, copy=False)

    # (1) 과거+현재+미래 11D 타임라인 만들기
    ego_all11 = np.concatenate([ego_past, ego_fut11], axis=0).astype(np.float32, copy=False)  # (Tp+Tf,11)
    nbr_all11 = np.concatenate([nbr_past, nbr_fut11], axis=1).astype(np.float32, copy=False)  # (N,Tp+Tf,11)

    # (안전) 현재가 무효면 그 에이전트 전체를 0으로
    ego_cur_valid = bool((np.abs(ego_past[-1, :8]) > eps).any())
    if not ego_cur_valid:
        ego_all11[:] = 0.0

    if N > 0:
        nbr_cur_valid_mask = (np.abs(nbr_past[:, -1, :8]) > eps).any(axis=1)  # (N,)
        if not np.all(nbr_cur_valid_mask):
            nbr_all11 = np.array(nbr_all11, copy=True)
            nbr_all11[~nbr_cur_valid_mask, :, :] = 0.0

    # (2) 11D -> pose3(x,y,heading)
    ego_pose_all3 = _traj11_to_traj3_heading(ego_all11)  # (Tp+Tf,3)
    nbr_pose_all3 = _traj11_to_traj3_heading(nbr_all11)  # (N,Tp+Tf,3)

    all_pose = np.concatenate([ego_pose_all3[None, ...], nbr_pose_all3], axis=0).astype(np.float32, copy=False)
    # all_pose: (1+N, Tp+Tf, 3)

    # (3) seg_valid 만들기
    ego_valid = (np.abs(ego_all11[:, :8]) > eps).any(axis=1)          # (Tp+Tf,)
    nbr_valid = (np.abs(nbr_all11[:, :, :8]) > eps).any(axis=2)       # (N,Tp+Tf)
    all_valid = np.concatenate([ego_valid[None, :], nbr_valid], axis=0).astype(bool)  # (1+N,Tp+Tf)

    seg_valid = (all_valid[:, :-1] & all_valid[:, 1:]).astype(bool)   # (1+N,Tp+Tf-1)

    # (4) controls 계산
    controls = differentiate_numpy_pose3_to_control3(all_pose, dt=dt).astype(np.float32, copy=False)
    # controls: (1+N, Tp+Tf-1, 3)

    return controls, seg_valid


def build_past_future_seg_control_gt_3_dim_from_npz_arrays(
    ego_agent_past: ArrayF,  # (Tp,11)  Tp=20+1
    ego_future_gt_11_dim: ArrayF,  # (Tf,11)  Tf=80
    neighbor_agents_past: ArrayF,  # (N,Tp,11)
    neighbor_future_gt_11_dim: ArrayF,  # (N,Tf,11)
    *,
    dt: float,
    eps: float = 1e-8,
) -> ArrayF:
    """npz 내부의 11차원 궤적들로부터 (과거~미래) 구간 제어를 만듭니다.

    Returns:
        np.ndarray:
            past_future_seg_control_gt_3_dim, shape (1+N, (Tp-1)+Tf, 3)
            - 마지막 3: (v_x^b, v_y^b, yaw_rate)
            - 무효 구간은 0.0
    """
    controls, seg_valid = _build_past_future_seg_control_gt_and_seg_valid_from_npz_arrays(
        ego_agent_past=ego_agent_past,
        ego_future_gt_11_dim=ego_future_gt_11_dim,
        neighbor_agents_past=neighbor_agents_past,
        neighbor_future_gt_11_dim=neighbor_future_gt_11_dim,
        dt=float(dt),
        eps=float(eps),
    )

    # controls: (1+N, Tseg, 3)
    # seg_valid: (1+N, Tseg)
    controls[~seg_valid] = 0.0
    return controls




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


# --------------------
# [4] _process_one_file 교체
# --------------------
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
    existing_keys: Optional[set[str]] = None,
) -> Tuple[bool, str]:
    """npz 하나를 읽고 필요한 key들을 추가해 저장합니다.

    변경점:
        - control key를 1개(past_future)에서 2개(past/future)로 분리 저장합니다.
        - past는 past만으로, future는 (현재+future)로 계산합니다.
        - 무효 구간은 0.0 처리합니다.

    Returns:
        (success, message)
    """
    if not os.path.exists(npz_path):
        return False, f"missing: {npz_path}"

    # 기존 key 목록(가능하면 main에서 읽어온 것을 재사용)
    keys: set[str]
    if existing_keys is not None:
        keys = set(existing_keys)
    else:
        try:
            keys = _read_npz_key_set(npz_path)
        except Exception:
            keys = set()

    expected = _build_expected_keys_for_done(
        keys,
        add_sample_keys=bool(add_sample_keys),
        use_agent_route_lane_order=bool(use_agent_route_lane_order),
    )
    missing_expected = expected.difference(keys)

    # control 필요 여부(각각)
    need_past_control = bool(overwrite) or (PAST_CONTROL_KEY in missing_expected)
    need_future_control = bool(overwrite) or (FUTURE_CONTROL_KEY in missing_expected)
    need_control = bool(need_past_control or need_future_control)

    control_key_set = set(_CONTROL_KEYS)

    need_sample = bool(add_sample_keys) and (
        bool(overwrite_sample_keys) or (len(missing_expected.difference(control_key_set)) > 0)
    )

    if (not need_control) and (not need_sample):
        return True, "skip(no change)"

    # overwrite 계열이면: 중복 entry가 쌓이지 않도록 기존 방식(전체 재저장) 유지
    must_full_rewrite = bool(overwrite) or (bool(add_sample_keys) and bool(overwrite_sample_keys))

    # ------------------------------------------------------------
    # (A) 기본 모드(덮어쓰기 없음): 필요한 입력만 로드 + 새 key만 append 저장
    # ------------------------------------------------------------
    if not must_full_rewrite:
        keys_to_load: set[str] = set()

        if need_control:
            # past는 past만 필요하지만, future는 (현재=past[-1])이 필요해서 past도 같이 필요
            keys_to_load.update({"ego_agent_past", "neighbor_agents_past"})
            if bool(need_future_control):
                keys_to_load.update({"ego_future_gt_11_dim", "neighbor_future_gt_11_dim"})

        if need_sample:
            keys_to_load.update(
                {
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
                    "static_objects",
                    "route_lanes",
                    "route_lanes_speed_limit",
                    "route_lanes_has_speed_limit",
                    "speed_bump_points",
                    "driveway_points",
                    "lane_type",
                    "left_line_type",
                    "right_line_type",
                    "road_edge",
                    "road_edge_type",
                    "target_id",
                    "target_z",
                }
            )
            if bool(use_agent_route_lane_order):
                keys_to_load.add("agent_route_lane_order")

        npz_data = _load_npz_subset_as_dict(npz_path, sorted(keys_to_load))

        new_arrays: Dict[str, np.ndarray] = {}

        # (A-1) control keys (past / future)
        if need_control:
            if "ego_agent_past" not in npz_data or "neighbor_agents_past" not in npz_data:
                return False, "missing key 'ego_agent_past' or 'neighbor_agents_past'"

            if bool(need_past_control):
                past_control = build_past_seg_control_gt_3_dim_from_npz_arrays(
                    ego_agent_past=npz_data["ego_agent_past"],
                    neighbor_agents_past=npz_data["neighbor_agents_past"],
                    dt=float(dt),
                )
                new_arrays[PAST_CONTROL_KEY] = past_control

            if bool(need_future_control):
                for k in ["ego_future_gt_11_dim", "neighbor_future_gt_11_dim"]:
                    if k not in npz_data:
                        return False, f"missing key '{k}'"

                future_control = build_future_seg_control_gt_3_dim_from_npz_arrays(
                    ego_agent_past=npz_data["ego_agent_past"],
                    ego_future_gt_11_dim=npz_data["ego_future_gt_11_dim"],
                    neighbor_agents_past=npz_data["neighbor_agents_past"],
                    neighbor_future_gt_11_dim=npz_data["neighbor_future_gt_11_dim"],
                    dt=float(dt),
                )
                new_arrays[FUTURE_CONTROL_KEY] = future_control

        # (A-2) sample 파생키
        if need_sample:
            try:
                sample, base_output_keys, alias_output_keys = _build_sample_dict_like_dataset_getitem(
                    npz_data,
                    file_name=os.path.basename(npz_path),
                    predicted_neighbor_num=int(predicted_neighbor_num),
                    eval_method=str(eval_method),
                    use_agent_route_lane_order=bool(use_agent_route_lane_order),
                )
            except Exception as e:
                return False, f"build_sample_failed: {type(e).__name__}: {e}"

            sample_new = _collect_npz_arrays_from_sample(
                sample,
                base_output_keys=base_output_keys,
                alias_output_keys=alias_output_keys,
                overwrite=False,
                existing_keys=keys,
            )
            new_arrays.update(sample_new)

        if not new_arrays:
            return True, "skip(no change)"

        _atomic_update_npz_by_copy_and_append(
            npz_path,
            new_arrays,
            compress=bool(compress),
            compresslevel=1,
        )
        return True, "ok"

    # ------------------------------------------------------------
    # (B) overwrite 계열: 전체 재저장(중복 entry 방지)
    # ------------------------------------------------------------
    data = _read_npz_as_dict(npz_path)
    changed = False

    # (B-1) control keys
    need_control_full = bool(overwrite) or (PAST_CONTROL_KEY not in data) or (FUTURE_CONTROL_KEY not in data)
    if need_control_full:
        for k in ["ego_agent_past", "neighbor_agents_past", "ego_future_gt_11_dim", "neighbor_future_gt_11_dim"]:
            if k not in data:
                return False, f"missing key '{k}'"

        data[PAST_CONTROL_KEY] = build_past_seg_control_gt_3_dim_from_npz_arrays(
            ego_agent_past=data["ego_agent_past"],
            neighbor_agents_past=data["neighbor_agents_past"],
            dt=float(dt),
        )
        data[FUTURE_CONTROL_KEY] = build_future_seg_control_gt_3_dim_from_npz_arrays(
            ego_agent_past=data["ego_agent_past"],
            ego_future_gt_11_dim=data["ego_future_gt_11_dim"],
            neighbor_agents_past=data["neighbor_agents_past"],
            neighbor_future_gt_11_dim=data["neighbor_future_gt_11_dim"],
            dt=float(dt),
        )
        changed = True

    # (B-2) sample 파생키
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

        changed |= _merge_sample_keys_into_npz_data(
            data,
            sample,
            base_output_keys=base_output_keys,
            alias_output_keys=alias_output_keys,
            overwrite=bool(overwrite_sample_keys),
        )

    if not changed:
        return True, "skip(no change)"

    _atomic_save_npz(npz_path, data, compress=bool(compress))
    return True, "ok"



# --------------------
# [5] 통계 계산 함수 교체(only_calculate_statistics 모드)
# --------------------
def _calculate_control_statistics_from_npz_path(
    npz_path: str,
    *,
    dt: float,
) -> Tuple[bool, str, ControlStatsAccumulator]:
    """npz 파일 1개에서 (v_x^b, v_y^b, yaw_rate) 통계 누적값을 계산합니다.

    변경점:
        - past/future control을 각각 계산하고,
          seg_valid=True인 값들만 모아서 누적합니다.

    Args:
        npz_path (str): npz 파일 경로
        dt (float): 시간 간격

    Returns:
        Tuple[bool, str, ControlStatsAccumulator]:
            - ok: 성공 여부
            - msg: "ok" 또는 실패 원인
            - acc: 해당 파일에서 얻은 누적기(성공 시 값 포함)
    """
    acc = ControlStatsAccumulator()

    if not os.path.exists(npz_path):
        return False, f"missing: {npz_path}", acc

    keys_to_load = [
        "ego_agent_past",
        "ego_future_gt_11_dim",
        "neighbor_agents_past",
        "neighbor_future_gt_11_dim",
    ]

    try:
        npz_data = _load_npz_subset_as_dict(npz_path, keys_to_load)
    except Exception as e:
        return False, f"load_failed: {type(e).__name__}: {e}", acc

    for k in keys_to_load:
        if k not in npz_data:
            return False, f"missing key '{k}'", acc

    try:
        past_controls, past_seg_valid = _build_past_seg_control_gt_and_seg_valid_from_npz_arrays(
            ego_agent_past=npz_data["ego_agent_past"],
            neighbor_agents_past=npz_data["neighbor_agents_past"],
            dt=float(dt),
        )
        future_controls, future_seg_valid = _build_future_seg_control_gt_and_seg_valid_from_npz_arrays(
            ego_agent_past=npz_data["ego_agent_past"],
            ego_future_gt_11_dim=npz_data["ego_future_gt_11_dim"],
            neighbor_agents_past=npz_data["neighbor_agents_past"],
            neighbor_future_gt_11_dim=npz_data["neighbor_future_gt_11_dim"],
            dt=float(dt),
        )
    except Exception as e:
        return False, f"control_build_failed: {type(e).__name__}: {e}", acc

    if past_controls.ndim != 3 or int(past_controls.shape[-1]) != 3:
        return False, f"past_controls shape mismatch: {past_controls.shape}", acc
    if past_seg_valid.shape != past_controls.shape[:2]:
        return False, f"past_seg_valid shape mismatch: {past_seg_valid.shape} vs {past_controls.shape}", acc

    if future_controls.ndim != 3 or int(future_controls.shape[-1]) != 3:
        return False, f"future_controls shape mismatch: {future_controls.shape}", acc
    if future_seg_valid.shape != future_controls.shape[:2]:
        return False, f"future_seg_valid shape mismatch: {future_seg_valid.shape} vs {future_controls.shape}", acc

    acc.update(past_controls[past_seg_valid])
    acc.update(future_controls[future_seg_valid])
    return True, "ok", acc



def _worker_calculate_statistics_one_fname(
    fname: str,
) -> Tuple[str, bool, str, int, List[float], List[float]]:
    """멀티프로세스 워커: 파일 1개 통계를 계산해 (count,sum,sumsq)만 반환합니다."""
    cfg = _WORKER_CONFIG
    dataset_dir = str(cfg["dataset_dir"])
    dt = float(cfg["dt"])

    npz_path = os.path.join(dataset_dir, str(fname))

    ok, msg, acc = _calculate_control_statistics_from_npz_path(npz_path, dt=dt)
    if not ok:
        return str(fname), False, str(msg), 0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    return (
        str(fname),
        True,
        "ok",
        int(acc.count),
        [float(x) for x in acc.sum.reshape((3,)).tolist()],
        [float(x) for x in acc.sumsq.reshape((3,)).tolist()],
    )


def _run_only_calculate_statistics(
    *,
    dataset_dir: str,
    train_json: str,
    dt: float,
    limit: int,
    workers_arg: int,
) -> None:
    """npz를 수정하지 않고 mean/std만 계산하는 실행 함수입니다.

    Args:
        dataset_dir (str): npz 폴더
        train_json (str): 파일명 리스트(json)
        dt (float): 시간 간격
        limit (int): 0이면 전체, 양수면 앞에서 N개만
        workers_arg (int): 0이면 자동, 1이면 순차, 2 이상이면 멀티프로세스

    Returns:
        None
    """
    file_names = _load_training_file_list(train_json)
    if int(limit) > 0:
        file_names = file_names[: int(limit)]

    total_files = len(file_names)
    if total_files <= 0:
        print("done. files_total=0, files_failed=0, samples_valid=0")
        return

    workers: int = int(workers_arg) if int(workers_arg) > 0 else _get_auto_worker_count(total_files)
    workers = max(1, min(int(workers), int(total_files)))

    global_acc = ControlStatsAccumulator()
    fail_count = 0

    start_time_s = time.monotonic()
    last_print_time_s = start_time_s

    # (1) 순차
    if workers <= 1:
        pbar = tqdm(file_names, desc="only_calculate_statistics")
        for idx, fname in enumerate(pbar, start=1):
            npz_path = os.path.join(dataset_dir, str(fname))
            ok, msg, acc = _calculate_control_statistics_from_npz_path(npz_path, dt=float(dt))
            if ok:
                global_acc.merge(acc)
            else:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")

            last_print_time_s = _maybe_print_progress_every_5_min(
                start_time_s=start_time_s,
                last_print_time_s=last_print_time_s,
                processed=idx,
                total=total_files,
                interval_s=300.0,
            )

        mean, std = global_acc.compute_mean_std()
        _print_control_statistics_summary(
            total_files=total_files,
            fail_count=fail_count,
            acc=global_acc,
            mean=mean,
            std=std,
        )
        return

    # (2) 멀티프로세스
    worker_config: Dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "dt": float(dt),
        "only_calculate_statistics": True,
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
        for fname, ok, msg, count, sum_list, sumsq_list in pool.imap_unordered(
            _worker_calculate_statistics_one_fname,
            file_names,
            chunksize=chunksize,
        ):
            processed_count += 1
            pbar.update(1)

            if ok:
                global_acc.merge_from_parts(
                    count=int(count),
                    sum_list=sum_list,
                    sumsq_list=sumsq_list,
                )
            else:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")

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

    mean, std = global_acc.compute_mean_std()
    _print_control_statistics_summary(
        total_files=total_files,
        fail_count=fail_count,
        acc=global_acc,
        mean=mean,
        std=std,
    )


def _print_control_statistics_summary(
    *,
    total_files: int,
    fail_count: int,
    acc: ControlStatsAccumulator,
    mean: NDArray[np.float64],
    std: NDArray[np.float64],
) -> None:
    """콘솔에 결과를 보기 좋게 출력합니다."""
    labels = ["v_x^b", "v_y^b", "yaw_rate"]

    print("========== only_calculate_statistics result ==========")
    print(f"files_total={int(total_files)}, files_failed={int(fail_count)}")
    print(f"samples_valid={int(acc.count)}")

    for i, name in enumerate(labels):
        m = float(mean[i]) if np.isfinite(mean[i]) else float("nan")
        s = float(std[i]) if np.isfinite(std[i]) else float("nan")
        print(f"{name}: mean={m:.6g}, std={s:.6g}")

    # 복사/붙여넣기 편하게 json도 같이 출력
    out = {
        "v_x_b": {"mean": float(mean[0]), "std": float(std[0])},
        "v_y_b": {"mean": float(mean[1]), "std": float(std[1])},
        "yaw_rate": {"mean": float(mean[2]), "std": float(std[2])},
        "samples_valid": int(acc.count),
        "files_total": int(total_files),
        "files_failed": int(fail_count),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


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
    parser.add_argument(
        "--only_calculate_statistics",
        action="store_true",
        help="npz를 저장/수정하지 않고 (v_x^b, v_y^b, yaw_rate)의 mean/std만 계산합니다.",
    )

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
        default=448,
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
        type=bool,
        default=False,
        help="dataset.py와 동일하게 agent_route_lane_order 키도 sample에 포함합니다.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="프로세스 개수. 0이면 자동(최대 16), 1이면 순차 처리",
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

    only_calculate_statistics: bool = bool(getattr(args, "only_calculate_statistics", False))
    workers_arg: int = int(getattr(args, "workers", 0))

    if only_calculate_statistics:
        # ✅ 이 모드에서는 npz를 절대 저장/수정하지 않습니다.
        _run_only_calculate_statistics(
            dataset_dir=str(args.dataset_dir),
            train_json=str(args.train_json),
            dt=float(args.dt),
            limit=int(args.limit),
            workers_arg=int(workers_arg),
        )
        return
    file_names = _load_training_file_list(train_json)
    if limit > 0:
        file_names = file_names[:limit]

    total_files = len(file_names)
    if total_files <= 0:
        print("done. ok=0, fail=0, skip=0, total=0")
        return

    workers: int = workers_arg if workers_arg > 0 else _get_auto_worker_count(total_files)
    workers = max(1, min(int(workers), int(total_files)))

    ok_count = 0
    fail_count = 0
    skip_count = 0

    start_time_s = time.monotonic()
    last_print_time_s = start_time_s

    # -----------------------------
    # (1) 순차 모드
    # -----------------------------
    if workers <= 1:
        pbar = tqdm(file_names, desc="add_control_to_npz")
        for idx, fname in enumerate(pbar, start=1):
            npz_path = os.path.join(dataset_dir, fname)

            try:
                keys: Optional[set[str]] = None
                try:
                    keys = _read_npz_key_set(npz_path)
                except Exception:
                    keys = None

                if _is_already_processed_npz(
                    npz_path,
                    overwrite=overwrite,
                    add_sample_keys=add_sample_keys,
                    overwrite_sample_keys=overwrite_sample_keys,
                    use_agent_route_lane_order=use_agent_route_lane_order,
                    existing_keys=keys,
                ):
                    ok = True
                    msg = "skip(already processed)"
                    skip_count += 1
                else:
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
                        existing_keys=keys,
                    )

                if ok:
                    ok_count += 1
                else:
                    fail_count += 1
                    tqdm.write(f"[FAIL] {fname}: {msg}")

            except Exception as e:
                fail_count += 1
                tqdm.write(f"[EXCEPTION] {fname}: {type(e).__name__}: {e}")

            last_print_time_s = _maybe_print_progress_every_5_min(
                start_time_s=start_time_s,
                last_print_time_s=last_print_time_s,
                processed=idx,
                total=total_files,
                interval_s=300.0,
            )

        print(f"done. ok={ok_count}, fail={fail_count}, skip={skip_count}, total={total_files}")
        return

    # -----------------------------
    # (2) 병렬 모드
    # -----------------------------
    worker_config: Dict[str, Any] = {
        "dataset_dir": dataset_dir,
        "dt": dt,
        "overwrite": overwrite,
        "compress": compress,
        "add_sample_keys": add_sample_keys,
        "overwrite_sample_keys": overwrite_sample_keys,
        "predicted_neighbor_num": predicted_neighbor_num,
        "eval_method": eval_method,
        "use_agent_route_lane_order": use_agent_route_lane_order,
    }

    ctx = mp.get_context()  # 기본 시작 방식 사용
    processed_count = 0

    pbar = tqdm(total=total_files, desc=f"add_control_to_npz (workers={workers})")

    pool = ctx.Pool(
        processes=workers,
        initializer=_init_worker_process,
        initargs=(worker_config,),
    )

    try:
        # chunksize는 너무 작으면 오버헤드가 커질 수 있어 적당히 잡습니다.
        chunksize = 4

        for fname, ok, msg, skipped in pool.imap_unordered(_worker_process_one_fname, file_names, chunksize=chunksize):
            processed_count += 1
            pbar.update(1)

            if ok:
                ok_count += 1
                if bool(skipped):
                    skip_count += 1
            else:
                fail_count += 1
                tqdm.write(f"[FAIL] {fname}: {msg}")

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

    print(f"done. ok={ok_count}, fail={fail_count}, skip={skip_count}, total={total_files}")




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
