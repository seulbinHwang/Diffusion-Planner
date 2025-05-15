import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from typing import Any, Tuple

from diffusion_planner.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.utils.multithreading.worker_pool import Task
from concurrent.futures import as_completed  # NEW


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
    parser.add_argument('--agent_num',
                        type=int,
                        default=32,
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
                        default=70,
                        help='number of lanes')
    parser.add_argument('--route_len',
                        type=int,
                        default=20,
                        help='number of route lane point')
    parser.add_argument('--route_num',
                        type=int,
                        default=25,
                        help='number of route lanes')

    # (인자 정의는 동일)
    args = parser.parse_args()

    # 1) 저장 폴더
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
        log_names=log_names  # 깨진 로그가 빠진 목록
    ))
    # 5) 시나리오 생성
    worker = SingleMachineParallelExecutor(use_process_pool=True, max_workers=24)
    scenarios = builder.get_scenarios(scenario_filter, worker)  # 내부에서 병렬 로딩
    print(f"Total scenarios after filtering: {len(scenarios)}")

    #######
    # 6) 아직 안 한 시나리오만 (차집합 + 한 번만 포맷팅)
    print(f"processed: {len(processed)}")
    # 6-1) ID → 시나리오 객체 매핑
    scenario_id_map = {
        f"{s._map_name}_{s.token}": s
        for s in scenarios
    }
    # 6-2) processed와 차집합 연산
    remaining_ids = scenario_id_map.keys() - processed
    # 6-3) 최종 리스트
    remaining = [scenario_id_map[token] for token in remaining_ids]
    print(f"Remaining to process: {len(remaining)}")


    # 7) 배치 단위로 병렬 처리 + 실시간 완료율 표시 ──────────────────────
    if remaining:
        args_list = [(args, sc) for sc in remaining]
        batch_size = 24
        # 전체 배치 개수
        num_batches = (len(args_list) + batch_size - 1) // batch_size

        # 배치 단위로 진행 상황 표시
        for batch_idx in tqdm(
                range(num_batches),
                total=num_batches,
                desc="Processing batches",
                unit="batch",
        ):
            start = batch_idx * batch_size
            batch = args_list[start: start + batch_size]

            # 1) 현재 배치 태스크 예약
            futures = [
                worker.submit(Task(process_single_scenario), cfg_and_scn)
                for cfg_and_scn in batch
            ]

            # 2) 배치 완료까지 대기
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    # 로그 남기고 다음 시나리오로 넘어감
                    print(f"[Error] {e}")

    else:
        print("새로 처리할 시나리오가 없습니다.")

    # 8) 결과 파일 목록 저장(동일)  ───────────────────────────
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    with open('./diffusion_planner_training.json', 'w') as jf:
        json.dump(npz_files, jf, indent=4)
    print(f"Saved {len(npz_files)} .npz file names")
