#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NuBoard 런처 스크립트 (Jupyter → .py 변환)

- 하는 일
  - nuPlan 실험 결과 폴더 아래의 모든 `.nuboard` 파일을 자동으로 수집
  - Hydra 설정을 합성하여 NuBoard(웹 UI)를 지정 포트로 실행
  - diffusion_planner가 설치되어 있으면 해당 Hydra 설정 경로도 자동 포함

- 사용 순서
  1) 아래 '사용자 설정(필수/선택)' 값을 실제 경로/값으로 교체
  2) `python launch_nuboard.py`
  3) 브라우저에서 http://localhost:<포트> 접속 (기본 6599)

필수 전제:
- nuplan-devkit, hydra-core, (선택) diffusion_planner 가 설치/임포트 가능한 상태여야 합니다.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import hydra
from hydra.core.global_hydra import GlobalHydra

# =============================================================================
# 사용자 설정 (실행 전 반드시 확인/수정)
# =============================================================================

# [필수] 시뮬레이션 결과 최상위 폴더
# - 설명: 이 폴더 아래에 여러 실험 폴더가 있고, 각 실험 폴더 안에 `.nuboard` 파일이 있어야 합니다.
# - 예시: "/data/nuplan-v1.1/exp/exp/simulation/closed_loop_nonreactive_agents/diffusion_planner/val14/diffusion_planner_release/model_2025-01-25-18-29-09"
RESULT_FOLDER: str = "/home/user/nuplan/exp/simulation/log_replay_reactive_diffusion_agents/val14/CHALLENGE/npc_model_2025-09-14-21-02-39"

# [필수] nuPlan/데이터/맵/실험 경로 환경변수
# - NUPLAN_DEVKIT_ROOT: nuplan-devkit 저장소의 **로컬 절대 경로**
#   예: "/home/user/nuplan-devkit"
# - NUPLAN_DATA_ROOT: nuPlan dataset 루트 경로 (로그/DB 등)
#   예: "/data"
# - NUPLAN_MAPS_ROOT: nuPlan 맵 파일 루트 경로
#   예: "/data/nuplan-v1.1/maps"
# - NUPLAN_EXP_ROOT: nuPlan 실험 결과(실행 산출물) 루트 경로
#   예: "/data/nuplan-v1.1/exp"
# - NUPLAN_SIMULATION_ALLOW_ANY_BUILDER (선택): 커스텀 빌더 허용 플래그 (확장 코드 있을 때 편리)
HOME_DIR="/home/user/nuplan"
ENV_VARS: Dict[str, str] = {
    "NUPLAN_DEVKIT_ROOT": "/home/user/PycharmProjects/nuplan-devkit",
    "NUPLAN_DATA_ROOT":  f"{HOME_DIR}/dataset",
    "NUPLAN_MAPS_ROOT": f"{HOME_DIR}/dataset/maps",
    "NUPLAN_EXP_ROOT":  f"{HOME_DIR}",
    "NUPLAN_SIMULATION_ALLOW_ANY_BUILDER": "1",  # 선택사항, 필요 없으면 삭제 가능
}

# [선택] NuBoard 웹 UI 포트 번호 (이미 사용 중이면 다른 번호로 바꾸세요)
DEFAULT_PORT: int = 6599

# =============================================================================
# 내부 구현 (수정 불필요)
# =============================================================================

logger = logging.getLogger("nuboard_launcher")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def set_env_variables(env_vars: Dict[str, str]) -> None:
    """환경변수를 일괄 설정합니다.

    Args:
        env_vars: 설정할 환경변수 딕셔너리 (키=이름, 값=문자열)

    Raises:
        ValueError: 값이 비어 있거나 'REPLACE_WITH' 플레이스홀더 그대로인 경우
    """
    for key, value in env_vars.items():
        if not isinstance(value, str) or not value or "REPLACE_WITH" in value:
            raise ValueError(
                f"[환경변수 누락] {key} 값을 실제 경로/값으로 교체하세요. 현재: {value!r}"
            )
        os.environ[key] = value
        logger.info(f"환경변수 설정: {key} = {value}")


def validate_required_settings(result_folder: str, env_vars: Dict[str, str]) -> None:
    """사용자 설정 값의 존재 여부 및 경로 유효성을 대략 점검합니다.

    Args:
        result_folder: 시뮬레이션 결과 최상위 폴더 경로
        env_vars: 환경변수 딕셔너리

    Raises:
        FileNotFoundError: 경로가 존재하지 않는 경우
        ValueError: 필수 값이 비어 있거나 플레이스홀더인 경우
    """
    if not result_folder or "REPLACE_WITH" in result_folder:
        raise ValueError("[설정 누락] RESULT_FOLDER 를 실제 결과 루트 경로로 교체하세요.")
    if not Path(result_folder).exists():
        raise FileNotFoundError(f"[경로 오류] RESULT_FOLDER가 존재하지 않습니다: {result_folder}")

    # 필수 경로 유효성 (devkit/maps/exp는 실제로 존재해야 정상 동작)
    required_keys = ["NUPLAN_DEVKIT_ROOT", "NUPLAN_DATA_ROOT", "NUPLAN_MAPS_ROOT", "NUPLAN_EXP_ROOT"]
    for k in required_keys:
        v = env_vars.get(k, "")
        if not v or "REPLACE_WITH" in v:
            raise ValueError(f"[설정 누락] {k} 값을 실제 경로로 교체하세요.")
        if k != "NUPLAN_DATA_ROOT" and not Path(v).exists():
            # 데이터 루트는 네트워크 마운트/가상 경로일 수 있어 엄격히 체크하지 않습니다.
            raise FileNotFoundError(f"[경로 오류] {k} 경로가 존재하지 않습니다: {v}")


def resolve_config_path(env_vars: Dict[str, str]) -> Path:
    """Hydra가 참조할 NuBoard 설정 디렉터리 경로를 계산합니다.

    - nuplan-devkit 기준 경로: <NUPLAN_DEVKIT_ROOT>/nuplan/planning/script/config/nuboard

    Args:
        env_vars: 환경변수 딕셔너리

    Returns:
        NuBoard용 Hydra config 디렉터리 절대 경로
    """
    devkit_root = Path(env_vars["NUPLAN_DEVKIT_ROOT"]).resolve()
    config_path = devkit_root / "nuplan" / "planning" / "script" / "config" / "nuboard"
    return config_path


def _diffusion_planner_available() -> bool:
    """diffusion_planner 패키지 임포트 가능 여부를 반환합니다."""
    try:
        import importlib

        importlib.import_module("diffusion_planner")
        return True
    except Exception:
        return False


def build_hydra_searchpaths() -> List[str]:
    """Hydra searchpath 목록을 구성합니다.

    Returns:
        'pkg://...' 형태의 경로 문자열 리스트 (Hydra override에서 그대로 사용)
    """
    paths: List[str] = [
        "pkg://nuplan.planning.script.config.common",
        "pkg://nuplan.planning.script.config.experiments",
    ]
    if _diffusion_planner_available():
        # diffusion_planner가 설치되어 있을 때만 추가 (없으면 Hydra가 찾지 못해 에러)
        paths = [
            "pkg://diffusion_planner.config.scenario_filter",
            "pkg://diffusion_planner.config",
            *paths,
        ]
    return paths


def find_nuboard_files(result_root: str | Path) -> List[str]:
    """루트 폴더 아래에서 `.nuboard` 파일들의 **절대 경로**를 모두 수집합니다.

    Args:
        result_root: 시뮬레이션 결과 루트 폴더

    Returns:
        `.nuboard` 파일 경로 문자열 리스트(정렬됨)
    """
    root = Path(result_root).resolve()
    nuboard_files: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".nuboard"):
                nuboard_files.append(str((Path(dirpath) / fn).resolve()))
    nuboard_files.sort()
    return nuboard_files


def _initialize_hydra(config_path: Path) -> None:
    from os import getcwd
    from os.path import relpath

    GlobalHydra.instance().clear()
    config_path_rel = relpath(str(config_path), start=getcwd())  # <-- 절대→상대

    # Hydra 1.1~1.3 모두 안전하게 동작
    hydra.initialize(version_base=None, config_path=config_path_rel)



def run_nuboard_with_overrides(
    config_path: Path,
    nuboard_paths: List[str],
    port: int,
    hydra_searchpaths: List[str],
) -> None:
    """Hydra로 설정을 합성한 뒤 NuBoard를 실행합니다.

    Args:
        config_path: NuBoard Hydra config 디렉터리 절대 경로
        nuboard_paths: 로드할 `.nuboard` 파일 경로 리스트 (절대 경로)
        port: NuBoard 웹 UI 포트
        hydra_searchpaths: Hydra searchpath 목록 (pkg://...)
    """
    if not nuboard_paths:
        raise FileNotFoundError(
            "[검색 결과 없음] RESULT_FOLDER 하위에서 '.nuboard' 파일을 찾지 못했습니다.\n"
            "- 실험이 정상 완료되어 .nuboard가 생성되었는지 확인하세요.\n"
            "- RESULT_FOLDER 경로가 올바른지 확인하세요."
        )

    # Hydra 초기화 및 override 구성
    _initialize_hydra(config_path)

    # 리스트/문자열을 안전하게 넘기기 위해 JSON 포맷 사용
    overrides: List[str] = [
        "scenario_builder=nuplan",
        f"simulation_path={json.dumps(nuboard_paths)}",  # 예: ["/a/b/run1.nuboard", "/a/b/run2.nuboard"]
        f"hydra.searchpath=[{', '.join(hydra_searchpaths)}]",
        f"port_number={port}",
    ]

    cfg = hydra.compose(config_name="default_nuboard", overrides=overrides)

    # NuBoard 실행 (nuplan-devkit의 run_nuboard.py 내부 main 함수 사용)
    from nuplan_extent.planning.script.run_nuboard import main as main_nuboard

    logger.info("NuBoard 시작 준비 완료")
    logger.info(" - Nuboard 파일 개수: %d", len(nuboard_paths))
    logger.info(" - 포트: %d", port)
    main_nuboard(cfg)


def main() -> None:
    """엔트리포인트: 환경변수 설정 → 유효성 검사 → .nuboard 수집 → NuBoard 실행."""
    # 1) 환경변수 설정 및 점검
    set_env_variables(ENV_VARS)
    validate_required_settings(RESULT_FOLDER, ENV_VARS)

    # 2) NuBoard Hydra 설정 경로 계산
    config_path = resolve_config_path(ENV_VARS)
    if not config_path.exists():
        raise FileNotFoundError(
            "[경로 오류] NuBoard Hydra config 경로가 존재하지 않습니다: "
            f"{str(config_path)}\n"
            "- NUPLAN_DEVKIT_ROOT가 올바른지 확인하세요.\n"
            "- devkit이 최신인지(폴더 구조 동일) 확인하세요."
        )

    # 3) .nuboard 파일 수집
    nuboard_files = find_nuboard_files(RESULT_FOLDER)
    for i, p in enumerate(nuboard_files[:5]):
        logger.info("찾은 .nuboard 파일 [%d]: %s", i, p)
    if len(nuboard_files) > 5:
        logger.info("... (총 %d개)", len(nuboard_files))

    # 4) Hydra searchpath 구성 (diffusion_planner가 있을 때만 자동 포함)
    searchpaths = build_hydra_searchpaths()

    # 5) NuBoard 실행
    run_nuboard_with_overrides(
        config_path=config_path,
        nuboard_paths=nuboard_files,
        port=DEFAULT_PORT,
        hydra_searchpaths=searchpaths,
    )


if __name__ == "__main__":
    main()
