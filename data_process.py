# ==== CPU-ONLY & THREADING GUARD (must be first) ==============================
import os as _os
import args_util
from typing import Dict, List, Union


_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import time
# ---- 멀티프로세싱은 spawn으로 (fork로 인한 상태 상속 이슈 회피) ----
try:
    import multiprocessing as _mp
    _mp.set_start_method("spawn", force=True)
except Exception:
    pass

# ================================================================================



import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Any, Tuple, Dict, List
import sqlite3
from pathlib import Path

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_pool import Task
from concurrent.futures import as_completed  # NEW
import wandb
from datetime import datetime
import json
import os
from scenario_utils import get_or_load_scenarios  # 새 유틸리티 함수
# 임시 몽키패치
from nuplan.database.nuplan_db.lidar_pc import LidarPc
import sqlite3
from typing import Set

def _safe_from_db_row(row: sqlite3.Row) -> LidarPc:
    keys: Set[str] = set(row.keys())  # type: ignore
    def _hex_or_none(field: str):
        if field not in keys:
            return None
        val = row[field]
        return val.hex() if val is not None else None

    return LidarPc(
        token=_hex_or_none("token"),
        next_token=_hex_or_none("next_token"),
        prev_token=_hex_or_none("prev_token"),
        ego_pose_token=_hex_or_none("ego_pose_token"),
        lidar_token=_hex_or_none("lidar_token"),
        scene_token=_hex_or_none("scene_token"),
        filename=row["filename"] if "filename" in keys else None,
        timestamp=row["timestamp"] if "timestamp" in keys else None,
    )

LidarPc.from_db_row = staticmethod(_safe_from_db_row)

"""
<DB에서 처음 추출 + 캐시 저장>
python preprocess.py \
  --scenarios_cache_out my_scenarios.pkl \
  --scenarios_cache_in ""              # 비워두거나 생략

<이미 저장된 캐시 사용(빠르게 실행)>
python preprocess.py \
  --scenarios_cache_in my_scenarios.pkl

"""
# data_process.py (상단 import 아래 어울리는 곳)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json


def _load_all_sample_stats(save_dir: str) -> Dict[str, List[float]]:
    """save_path 안의 *.stats.json 사이드카를 모두 읽어 리스트로 병합한다.

    Returns:
        Dict[str, List[float]]:
            {
              "vehicle_count": [...],
              "pedestrian_count": [...],
              "bicycle_count": [...],
              "lane_speed_limit_ratio_percent": [...],
              "mean_speed_limit_kmh": [...],  # None/결측 제외
            }
    """
    vehicle, ped, bike = [], [], []
    ratio_pct, mean_kmh = [], []
    save_dir = os.path.join(save_dir, "json_temp")
    for name in os.listdir(save_dir):
        if not name.endswith(".stats.json"):
            continue
        path = os.path.join(save_dir, name)
        try:
            with open(path, "r") as f:
                d = json.load(f)
            vehicle.append(int(d.get("vehicle_count", 0)))
            ped.append(int(d.get("pedestrian_count", 0)))
            bike.append(int(d.get("bicycle_count", 0)))
            ratio = float(d.get("lane_speed_limit_ratio_percent", 0.0))
            ratio_pct.append(ratio)
            mk = d.get("mean_speed_limit_kmh", None)
            if mk is not None:
                mean_kmh.append(float(mk))
        except Exception as e:
            print(f"[Warn] stats load error: {name} ({e})")
            continue

    return {
        "vehicle_count": vehicle,
        "pedestrian_count": ped,
        "bicycle_count": bike,
        "lane_speed_limit_ratio_percent": ratio_pct,
        "mean_speed_limit_kmh": mean_kmh,
    }


def _int_bin_edges(data: np.ndarray) -> np.ndarray:
    """정수형 데이터 히스토그램용 bin 경계값을 계산한다.

    Args:
        data (np.ndarray): shape (N,), 정수/실수 값이 들어 있는 1차원 배열.

    Returns:
        np.ndarray: shape (K,), 막대 경계값 1차원 배열.
    """
    if data.size == 0:
        # shape (2,)
        return np.array([-0.5, 0.5], dtype=float)
    vmin = max(0, int(np.floor(np.nanmin(data))))
    vmax = int(np.ceil(np.nanmax(data)))
    if vmax < vmin:
        vmax = vmin
    # 정수 중앙에 막대가 오도록 -0.5, +0.5 간격
    # shape (K,)
    return np.arange(vmin - 0.5, vmax + 1.5, 1.0)


def _auto_continuous_edges(data: np.ndarray, max_bins: int = 50) -> np.ndarray:
    """연속값 데이터에 대해 자동으로 bin 경계값을 계산한다.

    Args:
        data (np.ndarray): shape (N,), 실수 값이 들어 있는 1차원 배열.
        max_bins (int): 최대 bin 개수.

    Returns:
        np.ndarray: shape (K,), 막대 경계값 1차원 배열.
    """
    # shape (M,), NaN/무한대 제거
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0, 1.0])
    dmin, dmax = float(np.min(data)), float(np.max(data))
    if not np.isfinite(dmin) or not np.isfinite(dmax):
        return np.array([0.0, 1.0])
    if dmax <= dmin:
        eps = 1.0 if dmax == 0 else abs(dmax) * 0.1
        return np.array([dmin - eps, dmax + eps])

    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    n = data.size
    if iqr > 0:
        bw = 2.0 * iqr * (n ** (-1.0 / 3.0))
    else:
        sd = np.std(data)
        bw = 3.5 * sd * (n ** (-1.0 / 3.0)) if sd > 0 else (dmax - dmin) / 10.0
    bw = max(bw, (dmax - dmin) / 100.0)  # 너무 촘촘/빈약 방지

    nbins = int(np.ceil((dmax - dmin) / bw))
    nbins = max(5, min(nbins, max_bins))
    # shape (K,)
    return np.linspace(dmin, dmax, nbins + 1)


def _hist(
    ax: plt.Axes,
    data: Union[List[float], np.ndarray],
    title: str,
    xlabel: str,
    bins: Union[int, str] = "auto",
    integer_bins: bool = False,
) -> None:
    """하나의 히스토그램을 그리고, y축을 % 단위로 맞춘다.

    Args:
        ax (plt.Axes): 히스토그램을 그릴 축 객체.
        data (List[float] | np.ndarray): shape (N,), 원본 데이터.
        title (str): 그래프 제목.
        xlabel (str): x축 라벨.
        bins (int | str): bin 개수 또는 모드 문자열("auto" 등).
        integer_bins (bool): 정수형 bin을 사용할지 여부.
                             True인 경우 x축 눈금이 항상 정수가 되도록 설정한다.
    """
    # shape (N,)
    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        ax.text(0.5, 0.5, "No Data", ha="center", va="center")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Percentage of samples (%)")
        ax.grid(True, alpha=0.3)
        return

    # y축을 %로: 각 샘플에 100/N 가중치
    # weights shape (N,)
    weights = np.full_like(data, 100.0 / data.size, dtype=float)

    # bin 엣지 결정
    if integer_bins:
        # 정수 개수용 bin → x축 눈금도 정수만 나오도록 설정
        edges = _int_bin_edges(data)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        if isinstance(bins, int):
            # 명시적 bin 개수 → 엣지 균등 분할
            dmin, dmax = float(np.min(data)), float(np.max(data))
            if dmax <= dmin:
                edges = np.array([dmin - 0.5, dmax + 0.5])
            else:
                # shape (bins+1,)
                edges = np.linspace(dmin, dmax, bins + 1)
        else:
            # 'auto' 등 문자열 → 직접 엣지 계산
            edges = _auto_continuous_edges(data)

    ax.hist(data, bins=edges, weights=weights)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Percentage of samples (%)")
    ax.grid(True, alpha=0.3)


def _plot_and_save_histograms(
    stats: Dict[str, List[float]],
    out_path: str,
    title_prefix: str = "Dataset Statistics",
) -> None:
    """5개의 히스토그램을 한 그림에 그린 뒤 PNG 파일로 저장한다.

    - vehicle/pedestrian/bicycle 는 개수 히스토그램이고,
      이 세 개 그래프의 x축 눈금은 항상 정수 값만 사용된다.
    - 나머지 두 그래프는 비율(%)과 속도(km/h)를 표시한다.

    Args:
        stats (Dict[str, List[float]]): 각 항목별 값 리스트.
            - "vehicle_count": List[float], 길이 N
            - "pedestrian_count": List[float], 길이 N
            - "bicycle_count": List[float], 길이 N
            - "lane_speed_limit_ratio_percent": List[float], 길이 N
            - "mean_speed_limit_kmh": List[float], 길이 N
        out_path (str): PNG를 저장할 파일 경로.
        title_prefix (str): 각 서브플롯 제목 앞에 붙일 문자열.
    """
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    # axes shape (6,)
    axes = axes.ravel()

    # 정수 개수 계열(대수): 정수 bin + % y축
    _hist(
        axes[0],
        stats["vehicle_count"],
        f"{title_prefix}: Vehicles / sample",
        "Vehicles per sample",
        integer_bins=True,
    )

    _hist(
        axes[1],
        stats["pedestrian_count"],
        f"{title_prefix}: Pedestrians / sample",
        "Pedestrians per sample",
        integer_bins=True,
    )

    _hist(
        axes[2],
        stats["bicycle_count"],
        f"{title_prefix}: Bicycles / sample",
        "Bicycles per sample",
        integer_bins=True,
    )

    # 비율(%): 명시적인 bin 개수(20) → 엣지 생성 + % y축
    _hist(
        axes[3],
        stats["lane_speed_limit_ratio_percent"],
        f"{title_prefix}: % Lanes with speed limit",
        "% (per sample)",
        bins=20,
        integer_bins=False,
    )

    # 연속값: 자동 엣지 계산 + % y축
    _hist(
        axes[4],
        stats["mean_speed_limit_kmh"],
        f"{title_prefix}: Mean speed limit (km/h) on limited lanes",
        "km/h",
        bins="auto",  # 내부에서 엣지 직접 계산
        integer_bins=False,
    )

    axes[5].axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# [추가] 전역 CPU 고정값(기본 128). 환경변수 DP_MAX_CPUS로 덮어쓰기 가능
# DP_MAX_CPUS = int(os.environ.get("DP_MAX_CPUS", "96"))

# [추가] 과다 스레딩 방지(각 워커 프로세스 내부 스레드 1로 고정)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")


def available_cpu_count() -> int:
    """
    컨테이너/호스트 어디서 실행해도
    현재 프로세스에 **실제로 할당된 논리 CPU 개수**를 반환.
    1) Linux & Python 3.9+ : os.sched_getaffinity(0)
    2) 그 외 : os.cpu_count()  (fallback)
    """
    # return DP_MAX_CPUS
    try:
        return_ = len(os.sched_getaffinity(0))  # 현재 프로세스에 할당된 CPU 개수
        print(f"Available CPUs: {return_}")  # 디버그용
        return return_  # cgroup cpuset 존중
    except AttributeError:
        print("Using os.cpu_count() as fallback for CPU count.")
        return_ = os.cpu_count()  # 전체 CPU 개수
        print(f"Total CPUs: {return_}")  # 디버그용
        return return_ or 1  # 최소 1


import shutil

_PROCESSOR = None  # 워커‑프로세스 전역 캐시
_CFG_NS = None  # cfg 를 다시 만들지 않도록 캐시

def _is_valid_npz_quick(path: str) -> bool:
    try:
        with np.load(path, allow_pickle=False) as z:
            need = {"map_name","token","ego_agent_past","neighbor_agents_past"}
            if not need <= set(z.files):
                return False
            eg, nb = z["ego_agent_past"], z["neighbor_agents_past"]
            return (eg.ndim == 2 and eg.shape[-1] == 11 and
                    nb.ndim == 3 and nb.shape[-1] == 11)
    except Exception:
        return False


def _is_valid_npz(path: str) -> bool:
    # quick보다 조금 더 엄격하게 하고 싶으면 여기에서 추가 검사
    return _is_valid_npz_quick(path)


def run_scenario(
    scn,  # NuPlan 시나리오 객체   (executor.map 의 1st iterable)
    cfg_dict: Dict  # config 를 dict 로 직렬화한 것 (2nd iterable)
) -> None:
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    """
    • 각 워커 프로세스에서 여러 번 호출된다.
    • 최초 호출 시에만 DataProcessor 를 만들어 전역에 저장하고 재사용한다.
    • 원래 `process_single_scenario` 와 동일한 예외‑안전 로직 포함.
    """

    global _PROCESSOR, _CFG_NS

    # ── 0) Lazy‑initialization (프로세스당 1회) ─────────────────
    if _PROCESSOR is None:
        _CFG_NS = argparse.Namespace(**cfg_dict)
        _PROCESSOR = DataProcessor(_CFG_NS)

    cfg = _CFG_NS  # 가독성용 얼라이어스

    # ── 1) 저장 파일 경로 ──────────────────────────────────────
    file_name = f"{scn._map_name}_{scn.token}.npz"
    final_filepath = os.path.join(cfg.save_path, file_name)

    try:
        # ── 2) 실제 전처리 (DataProcessor 내부에서 .npz 저장) ──
        _PROCESSOR.work([scn])

        # ── 3) 생성된 파일 무결성 체크 ────────────────────────
        if os.path.exists(final_filepath):
            ok_size = os.path.getsize(final_filepath) > 0
            ok_npz = _is_valid_npz(final_filepath)
            if not (ok_size and ok_npz):
                os.remove(final_filepath)
                raise RuntimeError(
                    f"{file_name}: invalid npz (size={ok_size}, npz={ok_npz})")

    except Exception:
        # ── 4) 오류 발생 시 불완전 파일 제거 후 예외 전파 ──────
        if os.path.exists(final_filepath):
            os.remove(final_filepath)
        raise


# ─── 1단계: 필요한 모듈 import 및 원본 함수 백업 ───
import nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_filter_utils as sf

_ORIG_GET = sf.get_scenarios_from_log_file  # 원본 함수


# ─── 2단계: Top-level 래퍼 함수 정의 (Pickle 가능) ───
def safe_get_scenarios_from_log_file(params):
    """
    params는
      • GetScenariosFromDbFileParams  단일 객체
      • 또는 그 객체들의 list (worker_map이 chunking해서 넘김)
    반환: List[ScenarioDict]  ← 원본 함수와 동일
    """

    def _call_orig(param_list):
        # _ORIG_GET은 "list"를 받아서 "List[ScenarioDict]"를 반환
        return _ORIG_GET(param_list)

    # 1) always list 로 맞추기
    param_list = params if isinstance(params, list) else [params]

    try:
        # 배치 전체 먼저 시도
        return _call_orig(param_list)

    except (sqlite3.DatabaseError, sqlite3.OperationalError):
        # 배치 내 개별 DB를 순차 검사
        merged: list = []
        for p in param_list:
            try:
                merged.extend(_call_orig([p]))  # 성공하면 그대로 추가
            except (sqlite3.DatabaseError, sqlite3.OperationalError) as e:
                db_path = p.log_file_absolute_path
                print(f"[Warning] Skip corrupt DB: {db_path}\n         └─ {e}")

                # data_root 경로는 params 안에 이미 들어 있음
                bad_db_path = Path(p.data_root) / "bad_db.json"
                try:
                    bad_list = json.loads(bad_db_path.read_text()
                                         ) if bad_db_path.exists() else []
                    bad_list.append(db_path)
                    bad_db_path.write_text(json.dumps(bad_list, indent=2))
                except Exception as io_err:
                    print(f"[Warning] Could not update bad_db.json: {io_err}")
                # 손상된 DB는 simply skip
        return merged


# ─── 3단계: 모든 모듈에서 같은 함수 객체를 보도록 패치 ───
sf.get_scenarios_from_log_file = safe_get_scenarios_from_log_file

# 이미 함수 핸들이 캐시된 모듈에도 덮어쓰기
import nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder as sb

sb.get_scenarios_from_log_file = safe_get_scenarios_from_log_file


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_filter_parameters(num_scenarios_per_type=None,
                          limit_total_scenarios=None,
                          shuffle=True,
                          scenario_tokens=None,
                          log_names=None):

    scenario_types = None
    scenario_tokens  # List of scenario tokens to include
    log_names = log_names  # Filter scenarios by log names
    map_names = None  # Filter scenarios by map names

    num_scenarios_per_type
    limit_total_scenarios
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None

    expand_scenarios = True
    remove_invalid_goals = False
    shuffle

    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (scenario_types, scenario_tokens, log_names, map_names,
            num_scenarios_per_type, limit_total_scenarios,
            timestamp_threshold_s, ego_displacement_minimum_m, expand_scenarios,
            remove_invalid_goals, shuffle, ego_start_speed_threshold,
            ego_stop_speed_threshold, speed_noise_tolerance)

def process_single_scenario(config_and_scenario: Tuple[Any, Any]) -> None:
    """
    한 시나리오를 처리할 때 사용되는 함수.
    처리 도중 에러가 나면 해당 시나리오의 .npz 파일을 삭제합니다.
    """
    config, scenario = config_and_scenario

    # 1) 저장 폴더와 파일명 미리 정해두기
    #    save_to_disk 메서드에 맞춰 map_name과 token을 합친 파일명
    filename = f"{scenario._map_name}_{scenario.token}.npz"
    final_filepath = os.path.join(config.save_path, filename)

    try:
        # 2) 실제 시나리오 처리
        processor = DataProcessor(config)
        processor.work([scenario])  # 내부에서 save_to_disk가 호출되어 .npz 파일 생성됨

        # 3) 파일이 정상적으로 만들어졌는지 간단히 검사
        if os.path.exists(final_filepath) and os.path.getsize(
                final_filepath) == 0:
            # 완전히 쓰이지 않은 빈 파일이므로 삭제
            os.remove(final_filepath)
            raise RuntimeError(
                f"Scenario {scenario._map_name}_{scenario.token}: 생성된 파일이 비어있음")

    except Exception:
        # 4) 예외 발생 시, 미완성된 .npz 파일이 있으면 삭제하고 예외를 전파
        if os.path.exists(final_filepath):
            os.remove(final_filepath)
        raise


# ====================== main()용 헬퍼 함수들 ======================

def prepare_save_path(args: argparse.Namespace) -> None:
    """캐시 파일을 저장할 폴더를 준비한다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자. args.save_path를 사용한다.
    """
    # 1) 저장 폴더
    if args.reset_save_path:
        # 기존 폴더 삭제 후 새로 생성
        if os.path.exists(args.save_path):
            ans = input(f"\n[data_process.py] Delete files at '{args.save_path}'? [y/N]: ").strip().lower()
            if ans == 'y':
                shutil.rmtree(args.save_path)
                print(f"Removed existing save path: {args.save_path}")
    print("save_path:", args.save_path)
    os.makedirs(args.save_path, exist_ok=True)


def get_processed_npz_set(args: argparse.Namespace) -> Set[str]:
    """이미 생성된 .npz 파일 목록을 읽어 집합으로 만든다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자. args.save_path를 사용한다.

    Returns:
        Set[str]: 이미 처리된 샘플 ID 집합. 원소 개수 = 저장된 .npz 개수.
    """
    processed_npz_set: Set[str] = set()
    # 2) 이미 생성된 .npz 확인
    with os.scandir(args.save_path) as it:
        for entry in it:
            name = entry.name
            # .npz 끝나는 것만
            if name.endswith('.npz'):
                # replace 대신 슬라이싱: 조금 더 빠름
                processed_npz_set.add(name[:-4])
    return processed_npz_set


def load_train_log_names(args: argparse.Namespace) -> List[str]:
    """학습에 사용할 로그 이름 리스트를 읽고, 깨진 DB 로그는 제외한다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자. args.data_path를 사용한다.

    Returns:
        List[str]: 최종적으로 사용할 로그 이름 리스트.
    """
    # 3) 학습에 쓸 로그 이름 읽기
    with open('./nuplan_train.json', encoding="utf-8") as f:
        log_names: List[str] = json.load(f)

    # 3-1) 깨진 로그 목록 읽어 제외
    from pathlib import Path
    bad_db_path = os.path.join(args.data_path, "bad_db.json")
    if os.path.exists(bad_db_path):
        with open(bad_db_path) as f:
            # JSON 에 저장된 전체 경로에서 파일명(stem)만 추출
            bad_logs = {Path(p).stem for p in json.load(f)}
        # 원래 로그 리스트에서 깨진 것들만 걸러냄
        log_names = [ln for ln in log_names if ln not in bad_logs]
        print(f"제외한 깨진 로그 개수: {len(bad_logs)}")
    else:
        print("bad_db.json이 없어 모든 로그를 사용합니다.")

    return log_names


def build_scenarios_from_args(args: argparse.Namespace,
                              log_names: List[str]) -> List[Any]:
    """설정과 로그 이름을 이용해, nuPlan 시나리오 리스트를 만든다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자.
        log_names (List[str]): 사용할 로그 이름 리스트.

    Returns:
        List[Any]: 로딩된 시나리오 객체 리스트. 길이 = 전체 시나리오 개수.
    """
    # 4) 시나리오 빌더
    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(
        args.data_path,  # "/media/user/E/dataset/nuplan-v1.1/splits/trainval"
        args.map_path,   # "/media/user/E/dataset/maps"
        sensor_root=None,
        db_files=None,
        map_version=map_version,
    )
    scenario_filter = ScenarioFilter(*get_filter_parameters(
        args.scenarios_per_type,
        args.total_scenarios,
        args.shuffle_scenarios,
        log_names=log_names,  # 깨진 로그가 빠진 목록
    ))
    # 5) 시나리오 생성
    loader_pool = SingleMachineParallelExecutor(
        use_process_pool=False, max_workers=available_cpu_count())
    scenarios = get_or_load_scenarios(
        builder=builder,
        scenario_filter=scenario_filter,
        loader_pool=loader_pool,
        cache_in=args.scenarios_cache_in,
        cache_out=args.scenarios_cache_out,
    )

    # scenarios = builder.get_scenarios(scenario_filter, loader_pool)  # 내부에서 병렬 로딩
    print(f"Total scenarios: {len(scenarios)}")
    loader_pool._executor.shutdown(wait=True)
    return list(scenarios)


def create_proc_pool() -> SingleMachineParallelExecutor:
    """시나리오 캐싱에 사용할 프로세스 풀을 만든다.

    Returns:
        SingleMachineParallelExecutor: use_process_pool=True 로 만든 실행기.
    """
    proc_pool = SingleMachineParallelExecutor(
        use_process_pool=True,
        max_workers=available_cpu_count(),
    )
    return proc_pool


def compute_remaining_scenarios(
    scenarios: List[Any],
    processed_npz_set: Set[str],
) -> List[Any]:
    """이미 처리된 시나리오를 제외하고 남은 시나리오 목록을 만든다.

    Args:
        scenarios (List[Any]): 전체 시나리오 리스트. 길이 = 전체 시나리오 수.
        processed_npz_set (Set[str]): 이미 처리된 `<map>_<token>` ID 집합.

    Returns:
        List[Any]: 새로 처리해야 할 시나리오 리스트.
    """
    #######
    # 6) 아직 안 한 시나리오만 (차집합 + 한 번만 포맷팅)
    print(f"processed_npz_set: {len(processed_npz_set)}")
    # 6-1) ID → 시나리오 객체 매핑
    scenario_id_map: Dict[str, Any] = {
        f"{s._map_name}_{s.token}": s for s in scenarios
    }
    # 6-2) processed와 차집합 연산
    remaining_ids = scenario_id_map.keys() - processed_npz_set
    # 6-3) 최종 리스트
    remaining: List[Any] = [scenario_id_map[token] for token in remaining_ids]
    remaining = remaining
    print(f"Remaining to process: {len(remaining)}")
    return remaining

def run_parallel_caching(
    remaining: List[Any],
    args: argparse.Namespace,
    proc_pool: SingleMachineParallelExecutor,
) -> None:
    """남은 시나리오들을 병렬로 캐싱하고, 전체 진행률과 ETA를 출력한다.

    SingleMachineParallelExecutor.map 대신
    submit + as_completed 를 써서
    각 시나리오가 끝날 때마다 바로 진행률을 찍는다.
    """
    if not remaining:
        print("새로 처리할 시나리오가 없습니다.")
        return

    cfg_dict = vars(args)
    total = len(remaining)

    start_ts = time.time()
    # 0.1% 단위로 로그 (최소 1개)
    log_every = max(1, total // 3000)

    print(f"[CACHE] start: {total:,} scenarios to process")

    # 각 future -> 시나리오 ID 매핑 (에러 메시지용)
    future_to_id: Dict[Any, str] = {}

    # 1) 모든 작업을 한 번에 submit
    for scn in remaining:
        scen_id = f"{scn._map_name}_{scn.token}"
        fut = proc_pool.submit(Task(run_scenario), scn, cfg_dict)
        future_to_id[fut] = scen_id

    try:
        done = 0

        # 2) 끝나는 작업부터 하나씩 받아서 진행률 출력
        for fut in as_completed(future_to_id):
            scen_id = future_to_id[fut]

            # 내부 예외를 여기서 다시 꺼내서 확인
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] scenario failed: {scen_id} ({e})")
                # 예전 map()처럼, 하나라도 실패하면 전체 중단
                raise

            done += 1

            # 0.1% 단위 / 처음 / 마지막만 출력
            if done == 1 or done == total or done % log_every == 0:
                now = time.time()
                elapsed = now - start_ts
                speed = done / elapsed if elapsed > 0 else 0.0
                remain = total - done
                eta_sec = remain / speed if speed > 0 else 0.0

                def _fmt_hhmm(sec: float) -> str:
                    if not (sec > 0):
                        return "--:--"
                    h = int(sec // 3600)
                    m = int((sec % 3600) // 60)
                    return f"{h:02d}h{m:02d}m"

                print(
                    f"[CACHE] {done:,}/{total:,} "
                    f"({done * 100 / total:5.1f}%) | "
                    f"elapsed {_fmt_hhmm(elapsed)}, "
                    f"ETA {_fmt_hhmm(eta_sec)}"
                )
    finally:
        # 기존 코드와 동일하게 풀 정리
        proc_pool._executor.shutdown(wait=True)


def run_parallel_caching2(
    remaining: List[Any],
    args: argparse.Namespace,
    proc_pool: SingleMachineParallelExecutor,
) -> None:
    """남은 시나리오들을 병렬로 캐싱하고, 전체 진행률과 ETA를 출력한다.

    Args:
        remaining (List[Any]): 새로 처리해야 할 시나리오 리스트.
        args (argparse.Namespace): 커맨드라인 인자. vars(args)를 그대로 넘긴다.
        proc_pool (SingleMachineParallelExecutor): 프로세스 풀 실행기.
    """
    # 7) 배치 단위로 병렬 처리 + 실시간 완료율 표시
    if remaining:
        cfg_dict = vars(args)
        total = len(remaining)

        # 전체 진행률 계산용
        start_ts = time.time()
        # 1% 단위로만 찍기 (최소 1개)
        log_every = max(1, total // 1000)

        print(f"[CACHE] start: {total:,} scenarios to process")

        try:
            results = proc_pool.map(
                Task(run_scenario),
                remaining,
                [cfg_dict] * total,
                verbose=False,  # ✅ 내부 tqdm 끄기
            )
            print("len(results):", len(results))

            def _fmt_hhmm(sec: float) -> str:
                """초 단위를 '00h00m' 형태 문자열로 바꾼다."""
                if not (sec > 0):
                    return "--:--"
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                return f"{h:02d}h{m:02d}m"

            # ✅ 완료된 시나리오 수 기준으로 전체 진행률 출력
            for i, _ in enumerate(results, start=1):
                print("i:", i)
                # 1% 단위 / 처음 / 끝에서만 찍기 → 로그 과하지 않게
                if i == 1 or i == total or i % log_every == 0:
                    now = time.time()
                    elapsed = now - start_ts
                    done_ratio = i / total
                    speed = i / elapsed if elapsed > 0 else 0.0
                    remain = total - i
                    eta_sec = remain / speed if speed > 0 else 0.0

                    print(
                        f"[CACHE] {i:,}/{total:,} "
                        f"({done_ratio*100:5.1f}%) | "
                        f"elapsed {_fmt_hhmm(elapsed)}, "
                        f"ETA {_fmt_hhmm(eta_sec)}"
                    )

        finally:
            proc_pool._executor.shutdown(wait=True)
    else:
        print("새로 처리할 시나리오가 없습니다.")


def save_npz_index(args: argparse.Namespace) -> None:
    """저장된 .npz 파일 이름 리스트를 json으로 저장한다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자. args.save_path를 사용한다.
    """
    # 8) 결과 파일 목록 저장(동일)
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    with open('./diffusion_planner_training.json', 'w') as jf:
        json.dump(npz_files, jf, indent=4)
    print(f"Saved {len(npz_files)} .npz file names")


def maybe_save_statistics(args: argparse.Namespace) -> None:
    """통계 플래그가 켜져 있으면 히스토그램 이미지를 생성해 저장한다.

    Args:
        args (argparse.Namespace): 커맨드라인 인자. args.save_path와 플래그를 사용한다.
    """
    # 집계 & 히스토그램 저장
    if args.make_statistics_when_caching:
        stats = _load_all_sample_stats(args.save_path)
        save_path = os.path.join(args.save_path, "histograms")
        os.makedirs(save_path, exist_ok=True)
        hist_png = os.path.join(save_path, "dataset_statistics_histograms.png")
        _plot_and_save_histograms(
            stats, hist_png, title_prefix="Diffusion-world model")
        print(f"Saved histogram PNG: {hist_png}")


def main() -> None:
    """data_process.py의 전체 흐름을 단계별로 실행한다."""
    args = args_util.get_args()

    sf.get_scenarios_from_log_file = safe_get_scenarios_from_log_file
    ctrl_run = None

    prepare_save_path(args)
    processed_npz_set = get_processed_npz_set(args)
    log_names = load_train_log_names(args)
    scenarios = build_scenarios_from_args(args, log_names)
    proc_pool = create_proc_pool()
    remaining = compute_remaining_scenarios(scenarios, processed_npz_set)
    run_parallel_caching(remaining, args, proc_pool)

    if ctrl_run is not None:
        ctrl_run.finish()

    save_npz_index(args)
    maybe_save_statistics(args)


if __name__ == "__main__":
    main()

