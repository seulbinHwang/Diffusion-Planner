#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import subprocess
from pathlib import Path


def apply_k8s_yaml(yaml_path: Path, namespace: str) -> None:
    """kubectl apply 명령으로 YAML을 적용합니다.

    이 함수는 지정한 YAML 파일이 실제로 존재하는지 먼저 확인한 뒤,
    `kubectl -n <namespace> apply -f <yaml_path>` 를 실행합니다.
    실행에 실패하면(권한/클러스터 접속/파일 오류 등) 에러 메시지를 그대로 보여주고 예외를 발생시킵니다.

    Args:
        yaml_path (Path): 적용할 YAML 파일 경로.
        namespace (str): 적용할 네임스페이스 이름.

    Raises:
        FileNotFoundError: YAML 파일이 없을 때 발생합니다.
        RuntimeError: kubectl 실행이 실패했을 때 발생합니다.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 파일이 없습니다: {yaml_path}")

    cmd = ["kubectl", "-n", namespace, "apply", "-f", str(yaml_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        msg = "\n".join([s for s in [stdout, stderr] if s])
        raise RuntimeError(f"kubectl apply 실패:\n{msg}")

    out = result.stdout.strip()
    if out:
        print(out)


def main() -> None:
    """스크립트 진입점입니다."""
    yaml_path = Path("/media/user/E/projects/Diffusion-Planner/diffusion_planner_npc_training2.yaml")
    namespace = "p-pnc"
    apply_k8s_yaml(yaml_path=yaml_path, namespace=namespace)


if __name__ == "__main__":
    main()
