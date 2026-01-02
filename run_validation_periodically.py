#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

DEFAULT_MINIFORGE_CONDA_PATH: str = "~/miniforge3/bin/conda"


@dataclass(frozen=True)
class _Config:
    """주기 실행에 필요한 설정값 묶음.

    Attributes:
        interval_min (int):
            몇 분마다 실행할지. shape: ()
        env_name (str):
            실행에 사용할 가상환경 이름. shape: ()
        conda_exe_path (Path):
            conda 실행 파일 경로. shape: ()
        run_sh_path (Path):
            실행할 .sh 파일 경로. shape: ()
        working_dir (Path):
            .sh를 실행할 때 기준 폴더. 보통 프로젝트 폴더. shape: ()
    """
    interval_min: int
    env_name: str
    conda_exe_path: Path
    run_sh_path: Path
    working_dir: Path


def _now_string() -> str:
    """현재 시간을 사람이 읽기 쉬운 문자열로 만든다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _find_conda_executable(user_given_path: Optional[str]) -> Path:
    """conda 실행 파일 경로를 찾는다.

    이 파일이 필요한 이유
    -------------------
    우리는 validation_notebook.sh를 "diffusion_planner" 환경에서 실행해야 한다.
    그때 conda라는 실행 프로그램이 환경을 골라서 실행해준다.

    찾는 순서(우선순위)
    ------------------
    1) 사용자가 --conda_path 로 직접 준 경로
    2) 기본 경로: ~/miniforge3/bin/conda
    3) PATH에서 conda 검색

    Args:
        user_given_path (Optional[str]):
            사용자가 직접 준 conda 경로. shape: ()

    Returns:
        Path:
            conda 실행 파일 경로. shape: ()

    Raises:
        FileNotFoundError:
            conda 실행 파일을 찾지 못했을 때.
    """
    candidates = []

    if user_given_path is not None and user_given_path.strip() != "":
        candidates.append(Path(user_given_path).expanduser())

    candidates.append(Path(DEFAULT_MINIFORGE_CONDA_PATH).expanduser())

    # PATH에서 찾기
    conda_in_path = shutil_which("conda")
    if conda_in_path is not None:
        candidates.append(Path(conda_in_path))

    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            p = p.expanduser()

        if p.exists() and p.is_file():
            return p

    raise FileNotFoundError("conda 실행 파일을 찾지 못했습니다.\n"
                            f"- 시도한 기본 경로: {DEFAULT_MINIFORGE_CONDA_PATH}\n"
                            "- 해결: --conda_path 로 conda 경로를 직접 지정해 주세요.\n"
                            "  예) --conda_path /home/user/miniforge3/bin/conda")


def shutil_which(command_name: str) -> Optional[str]:
    """PATH에서 실행 파일을 찾아 경로 문자열로 돌려준다."""
    import shutil
    return shutil.which(command_name)


def _run_validation_once(cfg: _Config, run_count: int) -> int:
    """validation_notebook.sh를 딱 1번 실행한다.

    Args:
        cfg (_Config): 실행 설정값. shape: ()
        run_count (int): 실제 실행 횟수(1부터 증가). shape: ()

    Returns:
        int: 실행 결과 코드. shape: ()
    """
    cmd = [
        str(cfg.conda_exe_path),
        "run",
        "--no-capture-output",
        "-n",
        cfg.env_name,
        "bash",
        str(cfg.run_sh_path),
        str(run_count),  # ✅ 실행 횟수 주입
    ]

    print(f"[{_now_string()}] 실행 시작 (run_count={run_count})")
    print("  실행 명령:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cfg.working_dir),
            check=False,
        )
        code = int(result.returncode)
    except KeyboardInterrupt:
        # 사용자가 멈추고 싶을 때(Ctrl+C)
        print(f"[{_now_string()}] 사용자가 중단했습니다.")
        raise
    except Exception as e:
        print(f"[{_now_string()}] 실행 중 오류: {e}")
        code = 1

    print(f"[{_now_string()}] 실행 종료 (결과 코드: {code})")
    return code


def _compute_next_run_index(
    start_monotonic: float,
    interval_sec: float,
    now_monotonic: float,
) -> int:
    """다음 실행 순서(index)를 계산한다.

    중요한 동작(요청 조건 2번)
    ------------------------
    - 실행 시간이 interval보다 길어지면,
      그 사이에 "원래 실행됐어야 할 횟수"는 따라잡지 않는다.
    - 즉, 지나간 시도는 건너뛰고, 다음 시간에 1번만 실행한다.

    Args:
        start_monotonic (float):
            이 프로그램이 처음 시작한 시점(초 단위). shape: ()
        interval_sec (float):
            실행 간격(초 단위). shape: ()
        now_monotonic (float):
            현재 시점(초 단위). shape: ()

    Returns:
        int:
            다음 실행 index. shape: ()
    """
    elapsed = float(now_monotonic - start_monotonic)
    if interval_sec <= 0.0:
        return 1
    # 지금 시점이 이미 몇 구간을 지났는지 계산하고, 그 다음 구간으로 맞춘다.
    next_index = int(math.floor(elapsed / interval_sec) + 1)
    return max(0, next_index)


def _sleep_until(target_monotonic: float) -> None:
    """target 시점까지 쉬었다가(sleep) 깨어난다.

    기다리는 동안 CPU를 거의 쓰지 않아서, 불필요한 자원 사용이 적다.
    """
    while True:
        now = time.monotonic()
        remain = float(target_monotonic - now)
        if remain <= 0.0:
            return
        # 너무 잘게 자주 깨지 않게 조금 길게 자도 된다.
        time.sleep(min(remain, 5.0))


def _parse_args() -> argparse.Namespace:
    """입력값(몇 분마다 실행할지 등)을 읽는다."""
    parser = argparse.ArgumentParser(
        description="validation_pnc.sh를 N분마다 1번씩 실행합니다(겹치면 건너뜀).")
    parser.add_argument(
        "--interval_min",
        type=int,
        default=60,
        help=f"몇 분마다 실행할지)",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="diffusion_planner",
        help=f"사용할 가상환경 이름",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default="",
        help="conda 실행 파일 경로(비우면 자동으로 찾아봄)",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="~/PycharmProjects/Diffusion-Planner/validation_pnc.sh",
        help="실행할 validation_pnc.sh 경로",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    conda_exe_path = _find_conda_executable(
        args.conda_path if args.conda_path.strip() != "" else None)
    """
    Path(...)
        문자열로 된 경로(예: "~/.../validation_pnc.sh")를
        **“경로 전용 객체”**로 바꿔줘요.
    .expanduser()
        경로에 ~가 있으면 그걸 내 홈 폴더 경로로 바꿔줘요.
        예: ~/PycharmProjects/...
        → /home/user/PycharmProjects/... 처럼 바뀜
    .resolve()
        경로를 더 “확정된 형태”로 정리해줘요.
        .(현재 폴더), ..(상위 폴더) 같은 걸 정리해서 없애고
        가능하면 실제 파일 위치 기준으로 깔끔한 절대경로로 바꿔줘요.
        결과적으로 “어디서 실행하든” 같은 파일을 가리키게 만들기 좋아요.
    """
    run_sh_path = Path(args.script_path).expanduser().resolve()
    if not run_sh_path.exists():
        print(f"실행할 .sh 파일을 찾지 못했습니다: {run_sh_path}")
        return 2

    # 보통은 프로젝트 폴더에서 실행하는 게 안전함(상대 경로 파일을 쓸 수 있어서)
    working_dir = run_sh_path.parent

    cfg = _Config(
        interval_min=int(args.interval_min),
        env_name=str(args.env_name),
        conda_exe_path=conda_exe_path,
        run_sh_path=run_sh_path,
        working_dir=working_dir,
    )

    interval_sec = float(cfg.interval_min) * 60.0

    print("======================================")
    print(f"시작 시간: {_now_string()}")
    print(f"실행 간격: {cfg.interval_min} 분")
    print(f"가상환경: {cfg.env_name}")
    print(f"conda 경로: {cfg.conda_exe_path}")
    print(f"실행할 파일: {cfg.run_sh_path}")
    print("중단: Ctrl+C")
    print("======================================")

    start_t = time.monotonic()
    run_index = 0
    actual_run_count = 0  # ✅ "진짜 실행 횟수" (스킵된 시간은 포함 안 함)
    try:
        while True:
            scheduled_t = start_t + float(run_index) * interval_sec
            _sleep_until(scheduled_t)

            actual_run_count += 1
            _run_validation_once(cfg, actual_run_count)
            # 실행이 오래 걸렸으면, 지나간 횟수는 건너뛰고 다음 시간으로 맞춤
            now_t = time.monotonic()
            run_index = _compute_next_run_index(start_t, interval_sec, now_t)
    except KeyboardInterrupt:
        print(f"\n[{_now_string()}] 프로그램을 종료합니다.")
        return 0


if __name__ == "__main__":
    """ main() 의 return 값 -> 숫자(0, 1, 2 같은 값)
    
    정상 종료인지(0) / 문제가 있었는지(0이 아님)”를 나타내는 값이에요.
    
    main()이 돌려준 숫자를 그대로 종료 코드로 써서 프로그램을 끝내기
    """
    raise SystemExit(main())
