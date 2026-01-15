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

DEFAULT_MINIFORGE_CONDA_PATH: str = "~/miniforge/bin/conda"


@dataclass(frozen=True)
class _Config:
    """A bundle of settings needed for periodic execution.

    Attributes:
        interval_min (int):
            How many minutes between runs. shape: ()
        env_name (str):
            Name of the virtual environment to use. shape: ()
        conda_exe_path (Path):
            Path to the conda executable. shape: ()
        run_sh_path (Path):
            Path to the .sh script to run. shape: ()
        working_dir (Path):
            Working directory for running the .sh script (usually the project dir). shape: ()
    """
    interval_min: int
    env_name: str
    conda_exe_path: Path
    run_sh_path: Path
    working_dir: Path


def _now_string() -> str:
    """Return the current time as a human-readable string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _find_conda_executable(user_given_path: Optional[str]) -> Path:
    """Find the path to the conda executable.

    Why this file is needed
    -----------------------
    We must run validation_notebook.sh in the "diffusion_planner" environment.
    The conda executable selects the environment and runs the command.

    Search order (priority)
    -----------------------
    1) Path provided by the user via --conda_path
    2) Default path: ~/miniforge3/bin/conda
    3) Search for conda in PATH

    Args:
        user_given_path (Optional[str]):
            User-provided conda path. shape: ()

    Returns:
        Path:
            Path to the conda executable. shape: ()

    Raises:
        FileNotFoundError:
            If the conda executable cannot be found.
    """
    candidates = []

    if user_given_path is not None and user_given_path.strip() != "":
        candidates.append(Path(user_given_path).expanduser())

    candidates.append(Path(DEFAULT_MINIFORGE_CONDA_PATH).expanduser())

    # Search in PATH
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

    raise FileNotFoundError(
        "Could not find the conda executable.\n"
        f"- Default path tried: {DEFAULT_MINIFORGE_CONDA_PATH}\n"
        "- Fix: Provide the conda path explicitly via --conda_path.\n"
        "  Example) --conda_path /home/user/miniforge3/bin/conda"
    )


def shutil_which(command_name: str) -> Optional[str]:
    """Find an executable in PATH and return its path as a string."""
    import shutil
    return shutil.which(command_name)


def _run_validation_once(cfg: _Config, run_count: int) -> int:
    """Run validation_notebook.sh exactly once.

    Args:
        cfg (_Config): Run configuration. shape: ()
        run_count (int): Actual run count (starts at 1). shape: ()

    Returns:
        int: Process exit code. shape: ()
    """
    cmd = [
        str(cfg.conda_exe_path),
        "run",
        "--no-capture-output",
        "-n",
        cfg.env_name,
        "bash",
        str(cfg.run_sh_path),
        str(run_count),  # ✅ inject run count
    ]

    print(f"[{_now_string()}] Run started (run_count={run_count})")
    print("  Command:", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cfg.working_dir),
            check=False,
        )
        code = int(result.returncode)
    except KeyboardInterrupt:
        # User interruption (Ctrl+C)
        print(f"[{_now_string()}] Interrupted by user.")
        raise
    except Exception as e:
        print(f"[{_now_string()}] Error during execution: {e}")
        code = 1

    print(f"[{_now_string()}] Run finished (exit code: {code})")
    return code


def _compute_next_run_index(
    start_monotonic: float,
    interval_sec: float,
    now_monotonic: float,
) -> int:
    """Compute the next run index.

    Important behavior (requested condition #2)
    -------------------------------------------
    - If a run takes longer than the interval,
      we do NOT "catch up" missed runs.
    - We skip missed slots and run only once at the next scheduled time.

    Args:
        start_monotonic (float):
            Program start time (seconds, monotonic). shape: ()
        interval_sec (float):
            Interval (seconds). shape: ()
        now_monotonic (float):
            Current time (seconds, monotonic). shape: ()

    Returns:
        int:
            Next run index. shape: ()
    """
    elapsed = float(now_monotonic - start_monotonic)
    if interval_sec <= 0.0:
        return 1
    # Determine how many intervals have passed and move to the next one.
    next_index = int(math.floor(elapsed / interval_sec) + 1)
    return max(0, next_index)


def _sleep_until(target_monotonic: float) -> None:
    """Sleep until target time (monotonic seconds).

    This uses very little CPU while waiting.
    """
    while True:
        now = time.monotonic()
        remain = float(target_monotonic - now)
        if remain <= 0.0:
            return
        # Don't wake up too frequently.
        time.sleep(min(remain, 5.0))


def _parse_args() -> argparse.Namespace:
    """Parse input arguments (interval, etc.)."""
    parser = argparse.ArgumentParser(
        description="Run validation_pnc.sh once every N minutes (skip overlaps)."
    )
    parser.add_argument(
        "--interval_min",
        type=int,
        default=60,
        help="How many minutes between runs",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="diffusion_planner",
        help="Virtual environment name to use",
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default="",
        help="Path to the conda executable (leave empty to auto-detect)",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default="/media/user/E/projects/Diffusion-Planner/validation_pnc.sh",
        help="Path to validation_pnc.sh to run",
    )
    # parser.add_argument(
    #     "--script_path",
    #     type=str,
    #     default="/mnt/nuplan/projects/Diffusion-Planner/validation_mlx.sh",
    #     help="Path to validation_pnc.sh to run",
    # )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    conda_exe_path = _find_conda_executable(
        args.conda_path if args.conda_path.strip() != "" else None
    )
    """
    Path(...)
        Converts a string path (e.g., "~/.../validation_pnc.sh")
        into a "path object".
    .expanduser()
        Expands ~ into the user's home directory.
        Example: ~/PycharmProjects/... -> /home/user/PycharmProjects/...
    .resolve()
        Normalizes the path into a more "final" form:
        removes '.' and '..' and, if possible, produces a clean absolute path.
        This helps ensure the same file is targeted regardless of the current directory.
    """
    run_sh_path = Path(args.script_path).expanduser().resolve()
    if not run_sh_path.exists():
        print(f"Could not find the .sh script to run: {run_sh_path}")
        return 2

    # Usually safest to run from the project directory (relative paths may be used).
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
    print(f"Start time: {_now_string()}")
    print(f"Interval: {cfg.interval_min} minutes")
    print(f"Environment: {cfg.env_name}")
    print(f"Conda path: {cfg.conda_exe_path}")
    print(f"Script to run: {cfg.run_sh_path}")
    print("Stop: Ctrl+C")
    print("======================================")

    start_t = time.monotonic()
    run_index = 0
    actual_run_count = 0  # ✅ actual run count (does not include skipped slots)
    try:
        while True:
            scheduled_t = start_t + float(run_index) * interval_sec
            _sleep_until(scheduled_t)

            actual_run_count += 1
            _run_validation_once(cfg, actual_run_count)
            # If the run took too long, skip missed slots and align to the next one.
            now_t = time.monotonic()
            run_index = _compute_next_run_index(start_t, interval_sec, now_t)
    except KeyboardInterrupt:
        print(f"\n[{_now_string()}] Exiting the program.")
        return 0


if __name__ == "__main__":
    """main() return value -> an integer (e.g., 0, 1, 2).

    This represents whether the program ended successfully (0) or not (non-zero).

    Use the return value from main() as the process exit code.
    """
    raise SystemExit(main())
