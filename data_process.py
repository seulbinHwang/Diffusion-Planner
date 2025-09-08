import os
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


def _plot_and_save_histograms(
    stats: Dict[str, List[float]],
    out_path: str,
    title_prefix: str = "Dataset Statistics",
) -> None:
    """수집된 리스트로 5개 히스토그램을 그리고 하나의 PNG로 저장한다."""
    # 5개 subplot (3x2 레이아웃; 마지막 한 칸은 비워둠)
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    axes = axes.ravel()

    def _hist(ax, data, title, xlabel, bins="auto"):
        if len(data) == 0:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center")
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            return
        ax.hist(data, bins=bins)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)

    _hist(axes[0], stats["vehicle_count"], f"{title_prefix}: Vehicles / sample",
          "Vehicles per sample")
    _hist(axes[1], stats["pedestrian_count"],
          f"{title_prefix}: Pedestrians / sample", "Pedestrians per sample")
    _hist(axes[2], stats["bicycle_count"], f"{title_prefix}: Bicycles / sample",
          "Bicycles per sample")
    _hist(axes[3],
          stats["lane_speed_limit_ratio_percent"],
          f"{title_prefix}: % Lanes with speed limit",
          "% (per sample)",
          bins=20)
    _hist(axes[4],
          stats["mean_speed_limit_kmh"],
          f"{title_prefix}: Mean speed limit (km/h) on limited lanes",
          "km/h",
          bins=20)

    # 마지막 subplot 비우기
    axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def available_cpu_count() -> int:
    """
    컨테이너/호스트 어디서 실행해도
    현재 프로세스에 **실제로 할당된 논리 CPU 개수**를 반환.
    1) Linux & Python 3.9+ : os.sched_getaffinity(0)
    2) 그 외 : os.cpu_count()  (fallback)
    """
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


def run_scenario(
    scn,  # NuPlan 시나리오 객체   (executor.map 의 1st iterable)
    cfg_dict: Dict  # config 를 dict 로 직렬화한 것 (2nd iterable)
) -> None:
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
        if os.path.exists(final_filepath) and os.path.getsize(
                final_filepath) == 0:
            os.remove(final_filepath)
            raise RuntimeError(f"{file_name}: 파일이 비어 있습니다.")

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument(
        '--scenarios_cache_in',  # 2) 불러올 파일
        type=str,
        default='scenarios_cache.pkl',  #None,
        help='미리 저장해둔 시나리오 *.pkl 경로 (지정 시 DB 로딩 건너뜀)',
    )
    parser.add_argument(
        '--scenarios_cache_out',  # 1) 저장할 파일
        type=str,
        default=None,  #'scenarios_cache.pkl',
        help='새로 추출한 시나리오를 저장할 *.pkl 경로',
    )
    parser.add_argument('--data_path',
                        default='/data/nuplan-v1.1/trainval',
                        type=str,
                        help='path to raw data')
    parser.add_argument('--map_path',
                        default='/data/nuplan-v1.1/maps',
                        type=str,
                        help='path to map data')
    parser.add_argument('--save_path',
                        default='./cache',
                        type=str,
                        help='path to save processed data')
    parser.add_argument('--scenarios_per_type',
                        type=int,
                        default=None,
                        help='number of scenarios per type')
    parser.add_argument('--total_scenarios',
                        type=int,
                        default=10,
                        help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios',
                        type=bool,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--reset_save_path',
                        type=bool,
                        default=True,
                        help='shuffle scenarios')
    parser.add_argument('--agent_num',
                        type=int,
                        default=448,
                        help='number of agents')
    parser.add_argument('--static_objects_num',
                        type=int,
                        default=5,
                        help='number of static objects')
    parser.add_argument('--lane_len',
                        type=int,
                        default=20,
                        help='number of lane point')
    parser.add_argument('--lane_num',
                        type=int,
                        default=150,
                        help='number of lanes')
    parser.add_argument('--route_len',
                        type=int,
                        default=20,
                        help='number of route lane point')
    parser.add_argument('--route_num',
                        type=int,
                        default=25,
                        help='number of route lanes')
    # ────── WandB 옵션 추가 ──────
    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--save_image', default=True, type=boolean)

    parser.add_argument('--wandb_project',
                        type=str,
                        default='Diffusion-Planner',
                        help='wandb project')
    parser.add_argument('--wandb_entity',
                        type=str,
                        default=None,
                        help='wandb entity (team or user)')
    parser.add_argument('--name',
                        type=str,
                        help='log name (default: "diffusion-planner-training")',
                        default="test_0727")  # npc_current_state_aug_0.5
    # (인자 정의는 동일)
    args = parser.parse_args()
    sf.get_scenarios_from_log_file = safe_get_scenarios_from_log_file
    if args.use_wandb:
        os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
        ctrl_run = wandb.init(
            project=args.wandb_project,
            name=args.name,
            entity=args.wandb_entity,
            settings=wandb.Settings(start_method="fork"),
        )
    else:
        ctrl_run = None
    args.wandb_group = ctrl_run.id if ctrl_run else None
    # 1) 저장 폴더
    if args.reset_save_path:
        # 기존 폴더 삭제 후 새로 생성
        if os.path.exists(args.save_path):
            shutil.rmtree(args.save_path)
            print(f"Removed existing save path: {args.save_path}")

    os.makedirs(args.save_path, exist_ok=True)

    # 2) 이미 생성된 .npz 확인
    processed = set()
    with os.scandir(args.save_path) as it:
        for entry in it:
            name = entry.name
            # .npz 끝나는 것만
            if name.endswith('.npz'):
                # replace 대신 슬라이싱: 조금 더 빠름
                processed.add(name[:-4])

    # 3) 학습에 쓸 로그 이름 읽기
    with open('./nuplan_train.json', encoding="utf-8") as f:
        log_names = json.load(f)

    # 3-1) 깨진 로그 목록 읽어 제외  ### NEW
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

    # 4) 시나리오 빌더
    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path,
                                    args.map_path,
                                    sensor_root=None,
                                    db_files=None,
                                    map_version=map_version)
    scenario_filter = ScenarioFilter(*get_filter_parameters(
        args.scenarios_per_type,
        args.total_scenarios,
        args.shuffle_scenarios,
        log_names=None,#log_names  # 깨진 로그가 빠진 목록
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

    proc_pool = SingleMachineParallelExecutor(use_process_pool=True,
                                              max_workers=available_cpu_count())

    #######
    # 6) 아직 안 한 시나리오만 (차집합 + 한 번만 포맷팅)
    print(f"processed: {len(processed)}")
    # 6-1) ID → 시나리오 객체 매핑
    scenario_id_map = {f"{s._map_name}_{s.token}": s for s in scenarios}
    # 6-2) processed와 차집합 연산
    remaining_ids = scenario_id_map.keys() - processed
    # 6-3) 최종 리스트
    remaining = [scenario_id_map[token] for token in remaining_ids]
    print(f"Remaining to process: {len(remaining)}")

    # 7) 배치 단위로 병렬 처리 + 실시간 완료율 표시 ──────────────────────
    if remaining:
        # 전체 배치 개수
        cfg_dict = vars(args)  # Namespace -> dict (pickle friendly)

        # map: iterable 인자들을 “열” 단위로 넘긴다.
        # 1st iterable  → remaining 시나리오들
        # 2nd iterable  → cfg_dict 를 시나리오 수 만큼 반복
        try:
            results = proc_pool.map(
                Task(run_scenario),
                remaining,
                [cfg_dict] * len(remaining),
                verbose=True,  # tqdm 진행률 표시
            )
            # 결과 소비(예외 전파용) ─ 이미 _map 내부에서 tqdm 으로 진행률 출력
            for _ in results:
                pass
        finally:
            proc_pool._executor.shutdown(wait=True)
    else:
        print("새로 처리할 시나리오가 없습니다.")
    if ctrl_run is not None:
        ctrl_run.finish()
    # 8) 결과 파일 목록 저장(동일)  ───────────────────────────
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    with open('./diffusion_planner_training.json', 'w') as jf:
        json.dump(npz_files, jf, indent=4)
    print(f"Saved {len(npz_files)} .npz file names")

    # 집계 & 히스토그램 저장
    stats = _load_all_sample_stats(args.save_path)
    save_path = os.path.join(args.save_path, "histograms")
    os.makedirs(save_path, exist_ok=True)
    hist_png = os.path.join(args.save_path, "dataset_statistics_histograms.png")
    _plot_and_save_histograms(stats, hist_png, title_prefix="Diffusion-Planner")
    print(f"Saved histogram PNG: {hist_png}")
