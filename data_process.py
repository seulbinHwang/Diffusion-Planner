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
            timestamp_threshold_s, ego_displacement_minimum_m,
            expand_scenarios, remove_invalid_goals, shuffle,
            ego_start_speed_threshold, ego_stop_speed_threshold,
            speed_noise_tolerance)


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
        if os.path.exists(final_filepath) and os.path.getsize(final_filepath) == 0:
            # 완전히 쓰이지 않은 빈 파일이므로 삭제
            os.remove(final_filepath)
            raise RuntimeError(f"Scenario {scenario.map_name}_{scenario.token}: 생성된 파일이 비어있음")

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
                        default=True,
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

    args = parser.parse_args()

    # 5) 저장 폴더 만들기
    os.makedirs(args.save_path, exist_ok=True)

    # 6) 이미 처리된 파일(.npz) 목록을 모아서, 중복 처리 방지
    processed = set()
    for fname in os.listdir(args.save_path):
        if fname.endswith('.npz'):
            processed.add(fname.replace('.npz', ""))  # map_name_token 형태

    # 7) 시나리오 빌더 초기화
    with open('./nuplan_train.json', "r", encoding="utf-8") as file:
        log_names = json.load(file)

    map_version = "nuplan-maps-v1.0"
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path,
                                    sensor_root=None, db_files=None,
                                    map_version=map_version)
    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            args.scenarios_per_type,
            args.total_scenarios,
            args.shuffle_scenarios,
            log_names=log_names
        )
    )

    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")

    # 8) 아직 처리되지 않은 시나리오만 모아서 병렬 처리 리스트 구성
    remaining_scenarios = []
    for a_scenario in scenarios:
        key = f"{a_scenario._map_name}_{a_scenario.token}"
        if key not in processed:
            remaining_scenarios.append(a_scenario)

    print(f"Remaining scenarios to process: {len(remaining_scenarios)}")

    args_list = [(args, a_scenario) for a_scenario in remaining_scenarios]

    # 9) 병렬 처리 (중간에 실패해도 process_single_scenario에서 미완성 파일 삭제 후 재시도 가능)
    if args_list:
        worker = SingleMachineParallelExecutor(use_process_pool=True)
        list(
            tqdm(
                worker.map(Task(process_single_scenario), args_list),
                total=len(args_list),
                desc="Processing scenarios"
            )
        )
    else:
        print("모두 처리된 상태입니다. 새로운 작업이 없습니다.")

    # 10) 최종적으로 만들어진 파일 목록 저장
    npz_files = [f for f in os.listdir(args.save_path) if f.endswith('.npz')]
    with open(
            '/mnt/nuplan/projects/Diffusion-Planner/diffusion_planner_training.json',
            'w') as json_file:
        json.dump(npz_files, json_file, indent=4)
    print(f"Saved {len(npz_files)} .npz file names")
