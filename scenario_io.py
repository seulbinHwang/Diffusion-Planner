# ────────────────────────────────────────────────────────────────
# scenario_io.py
# ────────────────────────────────────────────────────────────────
"""시나리오(Scenario) 객체 입‧출력 유틸리티.

Pickle(기본)·dill(선택)을 사용해 파이썬 객체 그대로 직렬화/역직렬화한다.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Sequence

try:
    import dill as _pkl  # dill 이 설치돼 있으면 더 넓은 객체 지원
except ModuleNotFoundError:          # noqa: WPS440
    _pkl = pickle                    # dill 미설치 시 표준 pickle 사용

# 타입 힌트를 위해 import‑time 타입만 가져옴 (런타임 영향 無)
from nuplan.planning.scenario_builder.abstract_scenario import (  # type: ignore
    AbstractScenario
)


def save_scenarios(
    scenarios: Sequence[AbstractScenario],
    file_path: str | Path,
) -> None:
    """시나리오 객체들을 바이너리 파일로 저장한다.

    Args:
        scenarios: NuPlan `Scenario` 인스턴스들의 시퀀스.
        file_path: 저장할 *.pkl 파일 경로.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open('wb') as fp:
        # protocol=HIGHEST 로 최대 압축·호환
        _pkl.dump(list(scenarios), fp, protocol=_pkl.HIGHEST_PROTOCOL)


def load_scenarios(file_path: str | Path) -> List[AbstractScenario]:
    """저장된 *.pkl 파일에서 시나리오 객체들을 복원한다.

    Args:
        file_path: `save_scenarios()` 로 만든 *.pkl 경로.

    Returns:
        시나리오 객체 리스트.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f'시나리오 파일을 찾을 수 없습니다: {file_path}')

    with file_path.open('rb') as fp:
        scenarios: List[AbstractScenario] = _pkl.load(fp)
    return scenarios
