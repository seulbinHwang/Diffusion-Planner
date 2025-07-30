# ────────────────────────────────────────────────────────────────
# scenario_utils.py
# ────────────────────────────────────────────────────────────────
"""시나리오 빌드 또는 캐시 로드 결정 함수."""

from __future__ import annotations

from pathlib import Path
from typing import List

from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import (
    NuPlanScenarioBuilder,
)
from nuplan.planning.utils.multithreading.worker_parallel import (
    SingleMachineParallelExecutor,
)

from scenario_io import load_scenarios, save_scenarios  # 위에서 작성한 모듈


def get_or_load_scenarios(
    builder: NuPlanScenarioBuilder,
    scenario_filter: ScenarioFilter,
    loader_pool: SingleMachineParallelExecutor,
    cache_in: str | None,
    cache_out: str | None,
) -> List['AbstractScenario']:  # noqa: WPS110
    """시나리오를 로드/빌드 후 리턴한다.

    Args:
        builder: `NuPlanScenarioBuilder` 인스턴스.
        scenario_filter: 시나리오 필터 객체.
        loader_pool: 시나리오 로드용 워커 풀.
        cache_in: 이미 저장된 *.pkl 경로. 존재하면 이 파일을 로드하고 빌드를 건너뜀.
        cache_out: 새로 빌드한 시나리오를 저장할 *.pkl 경로. ``None`` 이면 저장 안 함.

    Returns:
        시나리오 객체 리스트.
    """
    # 1) 미리 저장된 캐시가 있으면 우선 사용
    if cache_in and Path(cache_in).exists():
        print(f'[Info] 캐시에서 시나리오를 불러옵니다 → {cache_in}')
        return load_scenarios(cache_in)

    # 2) 그렇지 않으면 새로 빌드
    print('[Info] NuPlan DB에서 시나리오를 추출합니다…')
    scenarios = builder.get_scenarios(scenario_filter, loader_pool)

    # 3) 원한다면 캐시로 저장
    if cache_out:
        save_scenarios(scenarios, cache_out)
        print(f'[Info] 시나리오 {len(scenarios):,}개를 캐시에 저장했습니다 → {cache_out}')

    return scenarios
