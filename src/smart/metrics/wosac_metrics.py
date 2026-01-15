# Not a contribution
# Changes made by NVIDIA CORPORATION & AFFILIATES enabling <CAT-K> or otherwise documented as
# NVIDIA-proprietary are not a contribution and subject to the following terms and conditions:
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import numpy as np

import itertools
import torch.multiprocessing as mp

import os
from pathlib import Path
from typing import Dict, List, Tuple
import time
import tensorflow as tf
import waymo_open_dataset.wdl_limited.sim_agents_metrics.metrics as wosac_metrics
from google.protobuf import text_format
from torch import Tensor, tensor
from torchmetrics import Metric
from waymo_open_dataset.protos import (
    scenario_pb2,
    sim_agents_metrics_pb2,
    sim_agents_submission_pb2,
)
import queue as py_queue
import torch
from typing import Optional
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
import atexit
import multiprocessing.pool as mp_pool
from multiprocessing.pool import Pool as MpPool
from typing import Any

def _to_shared_cpu_tensor_for_wosac(value: Tensor) -> Tensor:
    """WOSAC worker에 넘길 텐서를 'CPU 공유 공간'에 올려서 반환합니다.

    목적
    ----
    - worker에 데이터를 전달할 때, 같은 값이 여러 번 복사되면 메모리가 급격히 늘 수 있습니다.
    - 이 함수는 텐서를 CPU로 옮긴 뒤, 여러 프로세스가 같이 볼 수 있는 공간(공유 공간)에
      올려서 "한 번만 만들어 공유"되는 형태로 만듭니다.

    처리 내용
    --------
    1) GPU 텐서면 CPU로 옮깁니다.
    2) numpy가 잘 처리하지 못하는 bfloat16이면 float32로 바꿉니다.
    3) 메모리 모양을 연속(contiguous)으로 맞춥니다.
    4) 공유 공간에 올립니다.

    Args:
        value (Tensor):
            변환할 입력 텐서.
            - shape: (N, ...)  (예: agent_id: (N2,), pred_traj: (N2, R, T, 2) 등)

    Returns:
        Tensor:
            CPU에 있고, 공유 공간에 올라간 텐서.
            - shape: 입력과 동일
    """
    v = value.detach()

    if v.is_cuda:
        v = v.cpu()

    if v.dtype == torch.bfloat16:
        v = v.to(dtype=torch.float32)

    if not v.is_contiguous():
        v = v.contiguous()

    try:
        # 이미 공유 텐서면 그대로 사용
        if hasattr(v, "is_shared") and bool(v.is_shared()):
            return v
    except Exception:
        pass

    # 공유 공간으로 이동(여러 프로세스가 같은 데이터를 보게 됨)
    v.share_memory_()
    return v


def _torch_cpu_tensor_to_numpy_view_for_wosac(value: Tensor) -> np.ndarray:
    """CPU 텐서를 numpy '뷰(view)'로 바꿉니다(가능하면 복사 없이).

    목적
    ----
    - worker 쪽에서 기존 로직(ScenarioRollouts 생성 등)은 numpy 배열을 사용합니다.
    - 다만 여기서는 "복사본을 만들지 않고" 같은 메모리를 그대로 보게 하고 싶습니다.
    - CPU 텐서가 연속 메모리라면 .numpy()는 보통 같은 메모리를 그대로 보는 형태로 동작합니다.

    Args:
        value (Tensor):
            CPU 텐서.
            - shape: (A, ...) (예: pred_traj: (A, R, T, 2))

    Returns:
        np.ndarray:
            value와 같은 메모리를 보는 numpy 배열.
            - shape: 입력과 동일
    """
    v = value.detach()
    if v.is_cuda:
        raise ValueError("value는 CPU 텐서여야 합니다.")
    if v.dtype == torch.bfloat16:
        v = v.to(dtype=torch.float32)
    if not v.is_contiguous():
        v = v.contiguous()
    return v.numpy()


def _to_cpu_numpy_for_wosac(value: Tensor) -> np.ndarray:
    """torch 텐서를 WOSAC worker로 보내기 좋은 numpy로 바꿉니다.

    이 함수가 하는 일
    ---------------
    - multiprocessing worker로 데이터를 보낼 때는, GPU 텐서를 그대로 보내기 어렵습니다.
      (pickle 과정에서 문제가 나거나, 불필요한 큰 이동이 생길 수 있습니다)
    - 그래서 텐서를 CPU로 옮긴 뒤 numpy로 바꿉니다.
    - 단, torch.bfloat16은 numpy 변환이 막히는 경우가 많아서,
      그 경우만 안전하게 float32로 바꿉니다.

    Args:
        value (Tensor):
            변환할 텐서.
            shape: (N, ...) 또는 빈 텐서도 가능

    Returns:
        np.ndarray:
            CPU에 있는 numpy 배열.
            shape: value와 동일
    """
    v = value.detach()
    if v.is_cuda:
        v = v.cpu()

    # numpy가 bfloat16을 직접 못 받는 환경 대비
    if v.dtype == torch.bfloat16:
        v = v.to(dtype=torch.float32)

    return v.contiguous().numpy()


def _compute_scenario_metrics_from_raw_star(
    args: Tuple[
        Any,
        str,
        str,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        bool,
        bool,
    ],
) -> Tuple[sim_agents_metrics_pb2.SimAgentMetrics, Dict[str, float]]:
    """Pool에서 1개 시나리오에 대해 '포장 만들기 + 점수 계산'을 같이 수행합니다.

    변경점(핵심)
    ----------
    - 기존에는 numpy 배열을 넘겼는데, 그 과정에서 데이터 복사가 늘기 쉬웠습니다.
    - 이제는 CPU 공유 텐서(torch.Tensor)를 넘기고,
      worker 안에서 numpy '뷰'로 바꿔 기존 계산을 그대로 돌립니다.

    Args:
        args (Tuple[...]):
            (config, scenario_file, scenario_id,
             agent_id_cpu, pred_traj_cpu, pred_z_cpu, pred_head_cpu,
             ego_only, should_validate)
            - agent_id_cpu:  shape (A,)
            - pred_traj_cpu: shape (A, R, T, 2)
            - pred_z_cpu:    shape (A, R, T)
            - pred_head_cpu: shape (A, R, T)

    Returns:
        Tuple[SimAgentMetrics, Dict[str, float]]:
            (scenario_metrics, z_only_metrics)
    """
    return WOSACMetrics._compute_scenario_metrics_from_raw(*args)


def _compute_scenario_metrics_star(
    args: Tuple[Any, str, sim_agents_submission_pb2.ScenarioRollouts, bool, bool],
) -> Tuple[sim_agents_metrics_pb2.SimAgentMetrics, Dict[str, float]]:
    """Pool에서 1개 시나리오 계산을 수행합니다.

    Pool은 함수 입력을 1개 값으로 받는 경우가 많아서,
    (config, file, rollout, ego_only) 4개를 튜플로 묶어 전달하고,
    내부에서는 기존 함수를 그대로 호출합니다.

    Args:
        args (Tuple[Any, str, ScenarioRollouts, bool]):
            (config, scenario_file, scenario_rollout, ego_only) 묶음. shape: ()

    Returns:
        Tuple[SimAgentMetrics, Dict[str, float]]:
            - scenario_metrics: WOSAC 결과
            - z_only_metrics: z만으로 계산한 추가 결과
    """
    return WOSACMetrics._compute_scenario_metrics(*args)


def _read_int_env(env_key: str, default: int) -> int:
    """환경변수에서 정수 값을 안전하게 읽습니다.

    이 함수가 필요한 이유
    --------------------
    쉘에서 export로 넣는 값은 문자열이라서,
    잘못된 값이 들어오면 int(...) 변환에서 에러가 날 수 있습니다.
    그래서 "읽기 실패하면 기본값"으로 돌아가게 해서 실행이 끊기지 않게 합니다.

    Args:
        env_key (str):
            읽을 환경변수 이름. shape: ()
        default (int):
            값이 없거나 변환이 실패하면 사용할 기본값. shape: ()

    Returns:
        int:
            읽어온 정수 값(없으면 default). shape: ()
    """
    raw = os.environ.get(env_key, "")
    if str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip())
    except ValueError:
        return int(default)


def _read_float_env(env_key: str, default: float) -> float:
    """환경변수에서 실수 값을 안전하게 읽습니다.

    Args:
        env_key (str):
            읽을 환경변수 이름. shape: ()
        default (float):
            값이 없거나 변환이 실패하면 사용할 기본값. shape: ()

    Returns:
        float:
            읽어온 실수 값(없으면 default). shape: ()
    """
    raw = os.environ.get(env_key, "")
    if str(raw).strip() == "":
        return float(default)
    try:
        return float(str(raw).strip())
    except ValueError:
        return float(default)


def _get_available_cpu_core_count() -> int:
    """현재 프로세스가 '실제로 쓸 수 있는' CPU 코어 수를 구합니다.

    왜 필요한가?
    ------------
    어떤 서버/클러스터 환경에서는 CPU 전체 코어가 아니라,
    현재 작업에 할당된 일부 코어만 사용 가능할 수 있습니다.
    이때 os.cpu_count()만 쓰면 과하게 잡힐 수 있어,
    가능한 경우(리눅스)에는 "현재 프로세스가 배정받은 코어 수"를 우선 사용합니다.

    Returns:
        int:
            사용 가능한 CPU 코어 수. shape: ()
    """
    try:
        # 리눅스에서 cpuset/affinity가 걸린 경우 실제 사용 가능 코어 수가 더 정확합니다.
        core_count = len(os.sched_getaffinity(0))
        return int(max(1, core_count))
    except Exception:
        core_count = os.cpu_count()
        if core_count is None:
            return 1
        return int(max(1, core_count))


def _get_gpu_process_count_per_node() -> int:
    """한 머신(노드)에서 동시에 도는 GPU 프로세스 수를 추정합니다.

    의미
    ----
    - 보통 torchrun/torch.distributed.run을 쓰면,
      GPU 1개당 프로세스 1개가 뜹니다.
    - 이 값이 크면, CPU 자원을 여러 프로세스가 나눠써야 하므로
      worker 수(P)를 더 줄여야 안정적입니다.

    우선순위
    --------
    1) LOCAL_WORLD_SIZE (torchrun이 주는 값: "이 노드에서 프로세스 몇 개?")
    2) WORLD_SIZE (전체 프로세스 수)
    3) 없으면 1

    Returns:
        int:
            GPU 프로세스 수(최소 1). shape: ()
    """
    local_world_size = _read_int_env("LOCAL_WORLD_SIZE", 0)
    if local_world_size > 0:
        return int(local_world_size)

    world_size = _read_int_env("WORLD_SIZE", 0)
    if world_size > 0:
        return int(world_size)

    return 1


def _get_wosac_tf_num_threads() -> int:
    """WOSAC 계산에서 TensorFlow가 쓰는 CPU 스레드 수(T)를 결정합니다.

    설정 방법(권장)
    --------------
    - 환경변수로 직접 지정:
      export DP_WOSAC_TF_THREADS=1  (또는 2,3,4)

    fallback(없으면)
    ----------------
    - TF_NUM_INTRAOP_THREADS
    - OMP_NUM_THREADS
    - 최종 기본값: 2

    Returns:
        int:
            TensorFlow 스레드 수 T (최소 1). shape: ()
    """
    t = _read_int_env("DP_WOSAC_TF_THREADS", 0)
    if t <= 0:
        t = _read_int_env("TF_NUM_INTRAOP_THREADS", 0)
    if t <= 0:
        t = _read_int_env("OMP_NUM_THREADS", 0)
    if t <= 0:
        t = 2

    return int(max(1, t))


def _get_wosac_cpu_fraction() -> float:
    """CPU 자원 중 WOSAC에 쓰겠다고 '목표로 삼는 비율'을 결정합니다.

    의미
    ----
    전체 CPU를 100% 다 쓰면,
    - 파이썬 자체,
    - 데이터 로딩,
    - OS,
    - 다른 라이브러리
    쪽이 흔들리면서 전체가 불안정해질 수 있습니다.

    기본은 0.75(= 75%)로 두고,
    필요하면 환경변수로 바꿀 수 있게 합니다.

    환경변수
    --------
    - DP_WOSAC_CPU_FRACTION (예: 0.7 ~ 0.8 권장)

    Returns:
        float:
            0.1 ~ 1.0 범위의 값. shape: ()
    """
    frac = float(_read_float_env("DP_WOSAC_CPU_FRACTION", 0.75))
    frac = max(0.1, min(1.0, frac))
    return float(frac)


def _recommend_wosac_mp_processes(
    batch_size: int,
    tf_num_threads: int,
) -> int:
    """Option A 규칙으로 multiprocessing worker 수(P)를 추천합니다.

    구현 규칙(요청 그대로)
    ---------------------
    1) TensorFlow 스레드 수(T)는 tf_num_threads 로 제한되어 있다고 가정합니다.
    2) worker 수(P)는 아래로 계산합니다.

        P ≈ (CPU코어수 * cpu_fraction) / (GPU프로세스수) / (T)

    3) P는 "배치 크기보다 클 필요 없음" → P <= batch_size
    4) 최소 1

    Args:
        batch_size (int):
            현재 update()에서 처리할 시나리오 개수(= scenario_rollouts 길이). shape: ()
        tf_num_threads (int):
            TensorFlow가 한 작업에서 쓸 CPU 스레드 수 T. shape: ()

    Returns:
        int:
            추천 worker 프로세스 수 P (최소 1). shape: ()
    """
    safe_batch = int(max(1, batch_size))
    safe_t = int(max(1, tf_num_threads))

    cpu_cores = int(_get_available_cpu_core_count())
    cpu_fraction = float(_get_wosac_cpu_fraction())
    gpu_proc = int(max(1, _get_gpu_process_count_per_node()))
    print("\n\n\n\n[WOSAC] Recommended WOSAC multiprocessing settings:")
    print("[WOSAC] available CPU cores:", cpu_cores,
            ", cpu_fraction:", cpu_fraction,
            ", gpu_processes_per_node:", gpu_proc,
            ", TF num threads:", safe_t)

    raw_p = (float(cpu_cores) *
             float(cpu_fraction)) / float(gpu_proc) / float(safe_t)

    # "대략 규칙"이므로 단순하게 내림 후 최소 1
    p = int(max(1, int(raw_p)))

    # P는 배치 크기보다 클 필요 없음
    p = int(min(p, safe_batch))

    # 혹시 모를 안전장치(코어 수 이상은 굳이 의미 없음)
    p = int(min(p, cpu_cores))

    return int(max(1, p))


def _configure_tensorflow_for_wosac(tf_num_threads: int) -> None:
    """WOSAC 계산용으로 TensorFlow 설정을 안정적으로 맞춥니다.

    하는 일
    ------
    1) TensorFlow가 GPU를 잡지 않게 막습니다.
       (PyTorch가 GPU를 쓰는데 TF까지 GPU를 잡으면 메모리/충돌 문제가 생기기 쉬움)
    2) TensorFlow가 CPU를 너무 많이 동시에 쓰지 않도록 스레드 수를 제한합니다.

    주의
    ----
    TensorFlow는 "이미 내부 준비가 끝난 뒤"에는 일부 설정 변경이 막힐 수 있습니다.
    그래서 이 함수는 실패해도 죽지 않게(try/except) 처리합니다.

    Args:
        tf_num_threads (int):
            TensorFlow 스레드 수 T. shape: ()

    Returns:
        None
    """
    t = int(max(1, tf_num_threads))

    # 1) TF GPU 비활성화
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    # 2) TF CPU 스레드 제한
    try:
        tf.config.threading.set_intra_op_parallelism_threads(t)
    except Exception:
        pass

    # inter_op은 너무 크게 두면 스레드가 늘어날 수 있어서 보수적으로 1로 둡니다.
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except Exception:
        pass


def _parse_cpuset(s: str) -> set[int]:
    cpus = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            cpus.update(range(int(a), int(b) + 1))
        else:
            cpus.add(int(part))
    return cpus

def _init_wosac_mp_worker(tf_num_threads: int) -> None:
    cpuset = os.environ.get("DP_CPUSET", "").strip()
    if cpuset:
        try:
            os.sched_setaffinity(0, _parse_cpuset(cpuset))
        except Exception:
            pass

    _configure_tensorflow_for_wosac(tf_num_threads=int(tf_num_threads))


def _get_wosac_mp_maxtasksperchild() -> Optional[int]:
    """Pool worker를 몇 번의 작업 후 새로 갈아끼울지(선택)를 읽습니다.

    왜 필요한가?
    ------------
    worker 프로세스가 오래 살아있으면, 일부 환경에서 메모리가 조금씩 늘어날 수 있습니다.
    이 값을 설정하면 "작업을 N번 처리한 worker는 자동으로 새 worker로 교체"되도록 할 수 있습니다.

    사용 방법
    --------
    - 환경변수 DP_WOSAC_MP_MAXTASKS_PER_CHILD 를 정수로 설정합니다.
      예) export DP_WOSAC_MP_MAXTASKS_PER_CHILD=50
    - 0 또는 미설정이면 worker를 교체하지 않습니다(None 반환).

    Returns:
        Optional[int]:
            - None: 교체하지 않음
            - int: 한 worker가 처리할 최대 작업 개수
    """
    v = int(_read_int_env("DP_WOSAC_MP_MAXTASKS_PER_CHILD", 3))
    if v <= 0:
        return None
    return int(max(1, v))


def _create_wosac_mp_pool(
    processes: int,
    tf_num_threads: int,
    maxtasksperchild: Optional[int],
) -> MpPool:
    """WOSAC 계산용 Pool을 생성합니다.

    하는 일
    ------
    1) 가능한 경우 'forkserver' 방식으로 worker 프로세스를 띄웁니다.
       - PyTorch 학습/추론과 섞일 때 상대적으로 안전하게 동작하는 편이라서 기본으로 사용합니다.
       - 만약 환경이 지원하지 않으면 기본 방식으로 fallback 합니다.
    2) 각 worker 시작 시 _init_wosac_mp_worker()를 1회 실행해,
       TensorFlow 설정(스레드 제한, GPU 비활성화)을 worker 쪽에서도 확실히 적용합니다.

    Args:
        processes (int):
            worker 프로세스 개수. shape: ()
        tf_num_threads (int):
            worker에서 TensorFlow가 쓸 스레드 수. shape: ()
        maxtasksperchild (Optional[int]):
            worker를 몇 번 작업 후 교체할지. shape: ()

    Returns:
        MpPool:
            생성된 Pool 객체
    """
    safe_p = int(max(1, processes))
    safe_t = int(max(1, tf_num_threads))

    try:
        ctx = mp.get_context("forkserver")
    except Exception:
        ctx = mp.get_context()

    pool_kwargs = {}
    if maxtasksperchild is not None:
        pool_kwargs["maxtasksperchild"] = int(max(1, maxtasksperchild))

    return ctx.Pool(
        processes=safe_p,
        initializer=_init_wosac_mp_worker,
        initargs=(safe_t,),
        **pool_kwargs,
    )


def _is_mp_pool_running(pool: MpPool) -> bool:
    """Pool이 현재 실행 가능한 상태인지 확인합니다.

    Args:
        pool (MpPool):
            검사 대상 Pool.

    Returns:
        bool:
            True이면 starmap/apply 같은 작업을 받을 수 있는 상태입니다.
    """
    try:
        return getattr(pool, "_state", None) == mp_pool.RUN
    except Exception:
        return False


def _safe_close_mp_pool(pool: MpPool) -> None:
    """Pool을 안전하게 닫습니다.

    닫는 순서
    ---------
    1) close + join: "더 이상 새 작업은 안 받지만, 진행 중인 작업은 끝내고 정리"
    2) 위가 실패하면 terminate + join: "강제로 종료"

    Args:
        pool (MpPool):
            닫을 Pool.

    Returns:
        None
    """
    try:
        pool.close()
        pool.join()
        return
    except Exception:
        pass

    try:
        pool.terminate()
        pool.join()
    except Exception:
        pass


def _assert_scenario_id_matches_rollouts(
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
) -> None:
    """원본 시나리오 ID와, 예측 결과에 적힌 시나리오 ID가 같은지 확인합니다.

    왜 필요한가?
    ----------
    - 원본 데이터(정답)와 예측 결과가 서로 다른 시나리오로 섞이면,
      점수가 정상적으로 계산되지 않거나, 엉뚱한 점수가 나올 수 있습니다.

    Args:
        scenario (scenario_pb2.Scenario):
            원본 시나리오(정답이 들어있는 데이터).
            - scenario.scenario_id: str
        scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
            예측 결과 묶음.
            - scenario_rollouts.scenario_id: str

    Returns:
        None

    Raises:
        ValueError:
            두 ID가 다르면 발생합니다.
    """
    scenario_id_from_file: str = str(getattr(scenario, "scenario_id", ""))
    scenario_id_from_rollouts: str = str(
        getattr(scenario_rollouts, "scenario_id", ""))
    if not scenario_id_from_file or not scenario_id_from_rollouts:
        raise ValueError("시나리오 ID가 비어 있습니다. "
                         f"(파일에서 읽은 ID='{scenario_id_from_file}', "
                         f"예측 결과 ID='{scenario_id_from_rollouts}')")

    if scenario_id_from_file and scenario_id_from_rollouts:
        if scenario_id_from_file != scenario_id_from_rollouts:
            raise ValueError("시나리오 ID가 서로 다릅니다. "
                             f"(파일에서 읽은 ID='{scenario_id_from_file}', "
                             f"예측 결과 ID='{scenario_id_from_rollouts}')")


def validate_wosac_rollouts_or_raise(
        should_validate: bool,
    scenario: scenario_pb2.Scenario,
    scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
    *,
    check_scenario_id: bool = True,
) -> None:
    """예측 결과가 '필수 대상만' 그리고 '필수 대상 전부' 들어있는지 공식 규칙으로 검사합니다.

    이 함수가 확인하는 것(공식 API가 검사)
    ------------------------------
    1) "반드시 포함해야 하는 대상"이 하나도 빠지지 않았는지
    2) "포함하면 안 되는 대상"이 섞이지 않았는지
    3) 각 대상의 예측 길이가 규칙(보통 80칸)과 맞는지
    4) 예측 결과 묶음 개수가 규칙(보통 32개)과 맞는지

    Args:
        scenario (scenario_pb2.Scenario):
            원본 시나리오(정답이 들어있는 데이터).
        scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
            예측 결과 묶음.
        check_scenario_id (bool):
            True이면, 시나리오 ID도 같이 비교합니다.

    Returns:
        None

    Raises:
        ValueError:
            규칙을 하나라도 어기면 발생합니다.
            (어떤 대상이 빠졌는지/섞였는지 등이 메시지에 들어갑니다.)
    """
    if check_scenario_id:
        _assert_scenario_id_matches_rollouts(
            scenario=scenario,
            scenario_rollouts=scenario_rollouts,
        )
    # ✅ 규칙 검사 OFF면 여기서 종료
    if not should_validate:
        return
    try:
        submission_specs.validate_scenario_rollouts(
            scenario_rollouts=scenario_rollouts,
            original_scenario=scenario,
            challenge_type=submission_specs.ChallengeType.SIM_AGENTS,
        )
    except ValueError as e:
        sid = str(getattr(scenario, "scenario_id", ""))
        raise ValueError(f"[WOSAC 규칙 위반] scenario_id='{sid}' | {e}") from e


def _format_duration_hms(duration_sec: float) -> str:
    """초 단위 시간을 'Hh Mm Ss' 문자열로 바꿉니다."""
    total_sec = int(max(0.0, float(duration_sec)))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    seconds = total_sec % 60
    return f"{hours}h {minutes:02d}m {seconds:02d}s"

def _is_rank_zero() -> bool:
    """현재 프로세스가 rank 0인지 확인합니다.

    torchrun 환경에서는 RANK가 설정됩니다.
    없으면 단일 프로세스로 보고 rank 0으로 취급합니다.
    """
    return str(os.environ.get("RANK", "0")).strip() == "0"


class WOSACMetrics(Metric):
    """
    validation metrics based on ground truth trajectory, using waymo_open_dataset api
    """

    def __init__(self,
                 prefix: str,
                 is_active: bool,
                 ego_only: bool = False) -> None:
        super().__init__()
        self.is_active = is_active
        self.is_mp_init = False
        self.prefix = prefix
        self.ego_only = ego_only
        self.wosac_config = self.load_metrics_config()

        self.field_names = [
            "metametric",
            "average_displacement_error",
            "linear_speed_likelihood",
            "linear_acceleration_likelihood",
            "angular_speed_likelihood",
            "angular_acceleration_likelihood",
            "distance_to_nearest_object_likelihood",
            "collision_indication_likelihood",
            "time_to_collision_likelihood",
            "distance_to_road_edge_likelihood",
            "offroad_indication_likelihood",
            "traffic_light_violation_likelihood",
            "min_average_displacement_error",
            "simulated_collision_rate",
            "simulated_offroad_rate",
            "simulated_traffic_light_violation_rate",
        ]
        # 이건 “나중에 여러 GPU 프로세스의 값을 합쳐서 한 덩어리로 만들 수 있게” 설계된 표시야.
        for k in self.field_names:
            self.add_state(k, default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("scenario_counter",
                       default=tensor(0.0),
                       dist_reduce_fx="sum")
        self.add_state(
            "z_only_average_displacement_error",
            default=tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "z_only_min_average_displacement_error",
            default=tensor(0.0),
            dist_reduce_fx="sum",
        )
        # ✅ Option A의 전제: TensorFlow가 한 작업에서 쓰는 CPU 스레드 수(T) 제한
        #    + TF가 GPU를 잡지 않도록 강제
        self._tf_num_threads: int = _get_wosac_tf_num_threads()
        _configure_tensorflow_for_wosac(tf_num_threads=self._tf_num_threads)
        # --- multiprocessing Pool 재사용을 위한 상태 ---
        self._mp_pool: Optional[MpPool] = None
        self._mp_pool_processes: int = 0
        self._mp_pool_tf_threads: int = int(self._tf_num_threads)
        self._mp_pool_cleanup_registered: bool = False

    @staticmethod
    def _compute_z_only_displacement_metrics(
        scenario: scenario_pb2.Scenario,
        scenario_rollouts: sim_agents_submission_pb2.ScenarioRollouts,
        challenge_type: submission_specs.ChallengeType = submission_specs.
        ChallengeType.SIM_AGENTS,
    ) -> Dict[str, float]:
        """z 값만으로 '얼마나 틀렸는지'를 계산합니다.

        왜 필요한가?
        ------------
        현재 로그에서
        - val_closed/ADE 는 (x, y)만 보고 계산합니다.
        - WOSAC의 average_displacement_error / min_ade 는 (x, y, z)를 같이 씁니다.

        그래서 z가 잘못 들어가면 WOSAC 쪽 값이 크게 나빠질 수 있는데,
        그 원인이 정말 z 때문인지 빠르게 확인하려면
        'z만' 따로 떼어서 오차를 출력하는 게 가장 확실합니다.

        계산 방법(정확히)
        ----------------
        1) 시나리오(정답)에서 평가 대상 물체들의 미래 z와 valid(유효/무효)를 읽습니다.
           - 정답 z: shape (M, T)
           - 정답 valid: shape (M, T)
           - M: 평가 대상 물체 개수
           - T: 미래 길이(보통 80)

        2) 예측(rollouts)에서 같은 물체들의 미래 z를 읽습니다.
           - 예측 z: rollout마다 (M, T)
           - rollout 개수 R(보통 32)

        3) 각 rollout r, 물체 m에 대해
           valid인 시간만 골라 |z_pred - z_gt| 의 평균을 냅니다.
           - 물체별 z 평균오차: shape (R, M)

        4) 아래 두 값을 만듭니다.
           - average_displacement_error_z_only:
               모든 rollout과 모든 물체에 대해 평균낸 값 (스칼라)
           - min_average_displacement_error_z_only:
               rollout마다 물체 평균을 낸 뒤, 그 중 가장 작은 값 (스칼라)

        Args:
            scenario (scenario_pb2.Scenario):
                정답이 들어있는 시나리오.
            scenario_rollouts (sim_agents_submission_pb2.ScenarioRollouts):
                예측이 들어있는 rollouts.
            challenge_type (submission_specs.ChallengeType):
                기본은 SIM_AGENTS.

        Returns:
            Dict[str, float]:
                - "average_displacement_error_z_only": float
                - "min_average_displacement_error_z_only": float
        """
        config = submission_specs.get_submission_config(challenge_type)
        start_idx = int(config.current_time_index) + 1
        end_idx = start_idx + int(config.n_simulation_steps)
        num_steps = int(config.n_simulation_steps)

        # 평가 대상 물체 id들 (AV + tracks_to_predict)
        eval_object_ids = submission_specs.get_evaluation_sim_agent_ids(
            scenario, challenge_type)

        # 빠른 조회를 위한 track 사전
        track_by_id: Dict[int, scenario_pb2.Track] = {
            int(track.id): track for track in scenario.tracks
        }

        # 정답 z/valid 준비: object_id -> (z[T], valid[T])
        gt_z_by_id: Dict[int, np.ndarray] = {}
        gt_valid_by_id: Dict[int, np.ndarray] = {}

        for obj_id in eval_object_ids:
            obj_id_int = int(obj_id)
            track = track_by_id.get(obj_id_int, None)
            if track is None:
                continue

            states = track.states[start_idx:end_idx]
            if len(states) != num_steps:
                # 길이가 기대와 다르면 안전하게 제외
                continue

            gt_z = np.asarray([s.center_z for s in states],
                              dtype=np.float32)  # (T,)
            gt_valid = np.asarray([s.valid for s in states],
                                  dtype=np.bool_)  # (T,)

            gt_z_by_id[obj_id_int] = gt_z
            gt_valid_by_id[obj_id_int] = gt_valid

        if len(gt_z_by_id) == 0 or len(scenario_rollouts.joint_scenes) == 0:
            return {
                "average_displacement_error_z_only": 0.0,
                "min_average_displacement_error_z_only": 0.0,
            }

        total_sum = 0.0
        total_count = 0
        rollout_means: List[float] = []

        # rollout(R) 반복
        for joint_scene in scenario_rollouts.joint_scenes:
            pred_z_by_id: Dict[int, np.ndarray] = {}

            for traj in joint_scene.simulated_trajectories:
                obj_id_int = int(traj.object_id)
                if obj_id_int not in gt_z_by_id:
                    continue

                pred_z = np.asarray(traj.center_z, dtype=np.float32)  # (T,)
                if pred_z.shape[0] != num_steps:
                    continue
                pred_z_by_id[obj_id_int] = pred_z

            # 이번 rollout에서의 물체별 z 평균오차들
            per_object_errors: List[float] = []

            for obj_id_int, gt_z in gt_z_by_id.items():
                pred_z = pred_z_by_id.get(obj_id_int, None)
                if pred_z is None:
                    continue

                gt_valid = gt_valid_by_id[obj_id_int]  # (T,)
                valid_cnt = int(gt_valid.sum())
                if valid_cnt == 0:
                    continue

                abs_err = np.abs(pred_z - gt_z)  # (T,)
                valid_f = gt_valid.astype(np.float32)  # (T,)
                z_mean_err = float((abs_err * valid_f).sum() / float(valid_cnt))

                per_object_errors.append(z_mean_err)
                total_sum += z_mean_err
                total_count += 1

            if len(per_object_errors) > 0:
                rollout_means.append(float(np.mean(per_object_errors)))

        avg_z_only = float(total_sum /
                           float(total_count)) if total_count > 0 else 0.0
        min_z_only = float(
            min(rollout_means)) if len(rollout_means) > 0 else 0.0

        return {
            "average_displacement_error_z_only": avg_z_only,
            "min_average_displacement_error_z_only": min_z_only,
        }

    @staticmethod
    def _compute_scenario_metrics(
        config,
        scenario_file,
        scenario_rollout,  # sim_agents_submission_pb2.ScenarioRollouts
        ego_only,
            should_validate,
    ) -> Tuple[sim_agents_metrics_pb2.SimAgentMetrics, Dict[str, float]]:
        scenario = scenario_pb2.Scenario()

        debug = False
        first_id = None
        second_id = None
        if debug:
            dataset = tf.data.TFRecordDataset([scenario_file],
                                              compression_type="")
            for idx, data in enumerate(dataset):
                tmp = scenario_pb2.Scenario()
                tmp.ParseFromString(bytes(data.numpy()))

                if idx == 0:
                    scenario.CopyFrom(tmp)
                    first_id = tmp.scenario_id
                    if not debug:
                        break
                elif idx == 1:
                    second_id = tmp.scenario_id
                    break

        if debug:
            rollout_id = getattr(scenario_rollout, "scenario_id", "")
            print(f"[WOSAC_DEBUG] file={scenario_file}")
            print(
                f"[WOSAC_DEBUG] first_id={first_id} second_id={second_id} rollout_id={rollout_id}"
            )

            if second_id is not None:
                print(
                    "[WOSAC_DEBUG][WARNING] 이 파일 안에 시나리오가 2개 이상 들어있습니다(샤드일 가능성)."
                )
                raise RuntimeError("시나리오가 2개 이상 들어있는 파일은 처리할 수 없습니다.")

            if rollout_id and first_id and (rollout_id != first_id):
                print(
                    "[WOSAC_DEBUG][ERROR] rollout_id != 파일의 첫 scenario_id 입니다. (정답-예측 매칭이 깨졌을 가능성 큼)"
                )
                raise RuntimeError("rollout_id와 파일의 scenario_id가 다릅니다.")

        for data in tf.data.TFRecordDataset([scenario_file],
                                            compression_type=""):
            scenario.ParseFromString(bytes(data.numpy()))
            break

        if ego_only:
            for i in range(len(scenario.tracks)):
                if i != scenario.sdc_track_index:
                    for t in range(91):
                        scenario.tracks[i].states[t].valid = False
            while len(scenario.tracks_to_predict) > 1:
                scenario.tracks_to_predict.pop()
            scenario.tracks_to_predict[0].track_index = scenario.sdc_track_index

        validate_wosac_rollouts_or_raise(
            should_validate=should_validate,
            scenario=scenario,
            scenario_rollouts=scenario_rollout,
            # 이 함수 인자 이름이 scenario_rollout(단수)라서 이렇게 넣는 게 맞아요
        )
        # 기존 WOSAC metric (그대로)
        scenario_metrics = wosac_metrics.compute_scenario_metrics_for_bundle(
            config, scenario, scenario_rollout)

        # z만 따로 계산한 metric (추가)
        z_only_metrics = WOSACMetrics._compute_z_only_displacement_metrics(
            scenario=scenario,
            scenario_rollouts=scenario_rollout,
            challenge_type=submission_specs.ChallengeType.SIM_AGENTS,
        )

        return scenario_metrics, z_only_metrics

    def _register_mp_pool_cleanup_at_exit(self) -> None:
        """프로그램 종료 시 Pool을 자동으로 정리하도록 등록합니다.

        왜 필요한가?
        ------------
        Pool을 재사용하면 worker 프로세스들이 계속 살아있습니다.
        이 상태로 프로그램이 끝나면 환경에 따라 worker가 남거나 종료가 늦어질 수 있어,
        종료 시점에 정리 코드를 한 번 실행하도록 등록해 둡니다.

        Returns:
            None
        """
        if self._mp_pool_cleanup_registered:
            return
        atexit.register(self.close_mp_pool)
        self._mp_pool_cleanup_registered = True

    def _get_or_create_mp_pool(
        self,
        *,
        min_processes: int,
        tf_threads: int,
    ) -> MpPool:
        """필요할 때만 Pool을 만들고, 이미 있으면 재사용합니다.

        동작 규칙
        ---------
        1) Pool이 없으면 새로 만듭니다.
        2) Pool이 있는데 이미 죽었거나(tf 설정이 바뀌었거나) worker 수가 부족하면
           기존 Pool을 닫고 새로 만듭니다.
        3) Pool이 있고 worker 수도 충분하면 그대로 재사용합니다.
           - 여기서는 "줄이는 것"은 하지 않습니다.
             이유: 배치 크기가 들쭉날쭉해도 매번 다시 만들지 않게 하려는 목적입니다.

        Args:
            min_processes (int):
                이번 update에서 최소한 필요하다고 보는 worker 개수. shape: ()
            tf_threads (int):
                worker에서 쓸 TensorFlow 스레드 수. shape: ()

        Returns:
            MpPool:
                사용할 Pool
        """
        safe_min_p = int(max(1, min_processes))
        safe_t = int(max(1, tf_threads))

        # 1) 기존 Pool이 있고, 그대로 써도 되는지 판단
        if self._mp_pool is not None:
            pool_ok = _is_mp_pool_running(self._mp_pool)
            tf_ok = (int(self._mp_pool_tf_threads) == int(safe_t))
            size_ok = (int(self._mp_pool_processes) >= int(safe_min_p))

            if pool_ok and tf_ok and size_ok:
                return self._mp_pool

            # 조건이 하나라도 깨지면 정리 후 재생성
            self.close_mp_pool()

        # 2) 새 Pool 생성
        maxtasksperchild = _get_wosac_mp_maxtasksperchild()
        new_pool = _create_wosac_mp_pool(
            processes=int(safe_min_p),
            tf_num_threads=int(safe_t),
            maxtasksperchild=maxtasksperchild,
        )
        self._mp_pool = new_pool
        self._mp_pool_processes = int(safe_min_p)
        self._mp_pool_tf_threads = int(safe_t)

        # 3) 종료 시 정리 등록(1회만)
        self._register_mp_pool_cleanup_at_exit()

        return new_pool

    def close_mp_pool(self) -> None:
        """재사용 중인 Pool이 있으면 닫고, 내부 상태를 초기화합니다.

        언제 쓰나?
        ----------
        - 검증/평가 루프가 끝났는데 프로세스를 빨리 정리하고 싶을 때
        - mp 관련 오류로 Pool을 다시 만들기 전에 안전하게 정리하고 싶을 때

        Returns:
            None
        """
        if self._mp_pool is None:
            self._mp_pool_processes = 0
            return

        try:
            _safe_close_mp_pool(self._mp_pool)
        finally:
            self._mp_pool = None
            self._mp_pool_processes = 0

    def __del__(self) -> None:
        """객체가 사라질 때 Pool을 정리합니다(안전장치).

        주의
        ----
        파이썬 종료 과정에서는 이 함수가 항상 호출된다고 보장할 수 없습니다.
        그래서 atexit 등록도 같이 사용합니다.
        """
        try:
            self.close_mp_pool()
        except Exception:
            pass

    @staticmethod
    def _build_scenario_rollouts_from_raw_arrays(
        scenario_id: str,
        agent_id: np.ndarray,   # (A,)
        pred_traj: np.ndarray,  # (A, R, T, 2)
        pred_z: np.ndarray,     # (A, R, T)
        pred_head: np.ndarray,  # (A, R, T)
    ) -> sim_agents_submission_pb2.ScenarioRollouts:
        """raw 예측 배열들로 ScenarioRollouts(포장)를 1개 시나리오 단위로 만듭니다.

        왜 필요한가?
        -----------
        (C) 방식의 핵심은:
        - 메인 프로세스에서 ScenarioRollouts(큰 포장 객체)를 미리 만들지 않고,
        - worker가 "시나리오 1개"를 맡아서 포장까지 만든 뒤,
        - 바로 점수 계산까지 끝내는 것입니다.

        입력/출력 shape
        -------------
        - agent_id: (A,)
        - pred_traj: (A, R, T, 2)  # 마지막 2는 (x, y)
        - pred_z: (A, R, T)
        - pred_head: (A, R, T)
        - 반환: ScenarioRollouts 1개 (scenario_id 포함)

        Args:
            scenario_id (str):
                시나리오 id 문자열. shape: ()
            agent_id (np.ndarray):
                object_id 목록.
                shape: (A,)
            pred_traj (np.ndarray):
                예측 xy 궤적.
                shape: (A, R, T, 2)
            pred_z (np.ndarray):
                예측 z(높이).
                shape: (A, R, T)
            pred_head (np.ndarray):
                예측 heading(각도).
                shape: (A, R, T)

        Returns:
            sim_agents_submission_pb2.ScenarioRollouts:
                시나리오 1개에 대한 rollouts(포장).
        """
        scenario_id = str(scenario_id)

        if pred_traj.ndim != 4 or int(pred_traj.shape[-1]) != 2:
            raise ValueError(
                "pred_traj는 (A, R, T, 2) 형태여야 합니다. "
                f"현재 shape={tuple(pred_traj.shape)}"
            )
        a = int(pred_traj.shape[0])
        r = int(pred_traj.shape[1])
        t = int(pred_traj.shape[2])

        if agent_id.ndim != 1 or int(agent_id.shape[0]) != a:
            raise ValueError(
                "agent_id는 (A,) 이고 pred_traj의 A와 같아야 합니다. "
                f"A(pred_traj)={a}, agent_id.shape={tuple(agent_id.shape)}"
            )
        if pred_z.shape != (a, r, t):
            raise ValueError(
                "pred_z는 (A, R, T) 형태여야 합니다. "
                f"expected={(a, r, t)}, got={tuple(pred_z.shape)}"
            )
        if pred_head.shape != (a, r, t):
            raise ValueError(
                "pred_head는 (A, R, T) 형태여야 합니다. "
                f"expected={(a, r, t)}, got={tuple(pred_head.shape)}"
            )

        joint_scenes: List[sim_agents_submission_pb2.JointScene] = []

        # rollout(R) 단위로 JointScene 생성
        for i_rollout in range(r):
            simulated_trajectories: List[
                sim_agents_submission_pb2.SimulatedTrajectory
            ] = []

            # agent(A) 단위로 SimulatedTrajectory 생성
            for i_agent in range(a):
                obj_id_int = int(agent_id[i_agent])

                # numpy 1D 배열을 그대로 넣어도(반복 필드에 대한 iterable) 동작하는 경우가 많습니다.
                # 기존 get_scenario_rollouts와 동일한 성격을 유지하기 위해 list 변환을 강제하지 않습니다.
                traj_xy = pred_traj[i_agent, i_rollout]       # (T, 2)
                traj_z = pred_z[i_agent, i_rollout]          # (T,)
                traj_head = pred_head[i_agent, i_rollout]    # (T,)

                simulated_trajectories.append(
                    sim_agents_submission_pb2.SimulatedTrajectory(
                        center_x=traj_xy[:, 0],
                        center_y=traj_xy[:, 1],
                        center_z=traj_z,
                        heading=traj_head,
                        object_id=obj_id_int,
                    )
                )

            joint_scenes.append(
                sim_agents_submission_pb2.JointScene(
                    simulated_trajectories=simulated_trajectories
                )
            )

        return sim_agents_submission_pb2.ScenarioRollouts(
            joint_scenes=joint_scenes,
            scenario_id=scenario_id,
        )

    @staticmethod
    def _compute_scenario_metrics_from_raw(
            config: Any,
            scenario_file: str,
            scenario_id: str,
            agent_id: Tensor,  # (A,)
            pred_traj: Tensor,  # (A, R, T, 2)
            pred_z: Tensor,  # (A, R, T)
            pred_head: Tensor,  # (A, R, T)
            ego_only: bool,
            should_validate: bool,
    ) -> Tuple[sim_agents_metrics_pb2.SimAgentMetrics, Dict[str, float]]:
        """worker 안에서 '포장 만들기 + 점수 계산'을 한 번에 수행합니다.

        변경점(핵심)
        ----------
        - 입력을 numpy가 아니라 CPU 텐서로 받습니다.
        - 그리고 numpy로 바꿀 때도 "복사본"을 만들지 않고,
          같은 메모리를 그대로 보는 형태(뷰)로 바꿉니다.

        Args:
            config (Any):
                WOSAC 설정. shape: ()
            scenario_file (str):
                TFRecord 파일 경로. shape: ()
            scenario_id (str):
                시나리오 id. shape: ()
            agent_id (Tensor):
                object_id 목록. shape: (A,)
            pred_traj (Tensor):
                예측 xy. shape: (A, R, T, 2)
            pred_z (Tensor):
                예측 z. shape: (A, R, T)
            pred_head (Tensor):
                예측 heading. shape: (A, R, T)
            ego_only (bool):
                True면 ego만 평가. shape: ()
            should_validate (bool):
                True면 규칙 검사. shape: ()

        Returns:
            Tuple[SimAgentMetrics, Dict[str, float]]:
                (scenario_metrics, z_only_metrics)
        """
        agent_id_np = _torch_cpu_tensor_to_numpy_view_for_wosac(
            agent_id)  # (A,)
        pred_traj_np = _torch_cpu_tensor_to_numpy_view_for_wosac(
            pred_traj)  # (A, R, T, 2)
        pred_z_np = _torch_cpu_tensor_to_numpy_view_for_wosac(
            pred_z)  # (A, R, T)
        pred_head_np = _torch_cpu_tensor_to_numpy_view_for_wosac(
            pred_head)  # (A, R, T)

        scenario_rollout = WOSACMetrics._build_scenario_rollouts_from_raw_arrays(
            scenario_id=str(scenario_id),
            agent_id=agent_id_np,
            pred_traj=pred_traj_np,
            pred_z=pred_z_np,
            pred_head=pred_head_np,
        )

        return WOSACMetrics._compute_scenario_metrics(
            config,
            scenario_file,
            scenario_rollout,
            ego_only,
            should_validate,
        )

    def update_from_rollout_tensors(
        self,
        scenario_files: List[str],
        scenario_ids: List[str],
        agent_id: Tensor,      # (N2,)
        agent_batch: Tensor,   # (N2,)
        pred_traj: Tensor,     # (N2, R, T, 2)
        pred_z: Tensor,        # (N2, R, T)
        pred_head: Tensor,     # (N2, R, T)
        should_validate: bool = True,
    ) -> None:
        """(C) 방식으로 WOSAC metric을 업데이트합니다.

        핵심 변화
        --------
        - 메인 프로세스에서 ScenarioRollouts(큰 포장 객체) 리스트를 만들지 않습니다.
        - 대신,
          1) 배치를 시나리오 단위로 나눕니다.
          2) 각 시나리오 조각(작은 raw 배열들)을 worker로 보냅니다.
          3) worker가 포장 만들기 + 점수 계산을 한 번에 끝냅니다.
          4) 메인은 결과 숫자만 받아 누적합니다.

        기대 효과(이론적으로)
        -------------------
        - 메모리 피크 감소: 한꺼번에 많은 포장 객체를 들고 있지 않음
        - 속도 개선: 포장 만들기 자체도 여러 CPU에서 같이 처리
        - 큰 객체 이동 감소: protobuf 완성품 대신 raw 배열만 이동

        Args:
            scenario_files (List[str]):
                TFRecord 경로 리스트. len = B
            scenario_ids (List[str]):
                scenario_id 문자열 리스트. len = B
            agent_id (Tensor):
                전체 배치의 agent object_id를 펼친 값(필터 후).
                shape: (N2,)
            agent_batch (Tensor):
                agent_id의 각 원소가 어느 시나리오(0~B-1)에 속하는지.
                shape: (N2,)
            pred_traj (Tensor):
                예측 xy. shape: (N2, R, T, 2)
            pred_z (Tensor):
                예측 z. shape: (N2, R, T)
            pred_head (Tensor):
                예측 heading. shape: (N2, R, T)
            should_validate (bool):
                True면 공식 규칙 검사 수행.

        Returns:
            None
        """
        batch_size_now: int = int(len(scenario_files))
        if int(len(scenario_ids)) != batch_size_now:
            raise ValueError(
                "scenario_files와 scenario_ids 길이가 다릅니다. "
                f"len(files)={len(scenario_files)}, len(ids)={len(scenario_ids)}"
            )

        # worker 추천 수(기존 update와 동일 규칙)
        tf_threads: int = int(
            getattr(self, "_tf_num_threads", _get_wosac_tf_num_threads())
        )
        recommended_p: int = _recommend_wosac_mp_processes(
            batch_size=batch_size_now,
            tf_num_threads=tf_threads,
        )

        disable_mp: bool = str(os.environ.get("DP_WOSAC_DISABLE_MP", "0")).strip() == "1"
        use_mp_pool: bool = (not disable_mp) and (recommended_p > 1)

        progress_sec_raw = _read_float_env("DP_WOSAC_PROGRESS_SEC", 60.0)
        progress_sec: float = float(progress_sec_raw)
        if progress_sec <= 0.0:
            progress_sec = 0.0

        if _is_rank_zero():
            print(
                f"[WOSACMetrics] update_from_rollout_tensors(): batch_size={batch_size_now}, "
                f"tf_threads={tf_threads}, "
                f"recommended_p={recommended_p}, "
                f"use_mp_pool={use_mp_pool}, "
                f"disable_mp={disable_mp}",
                flush=True,
            )

        if disable_mp:
            self.close_mp_pool()

        total = int(batch_size_now)
        start_t = time.perf_counter()
        last_print_t = start_t

        def _maybe_print(done: int, force: bool = False) -> None:
            nonlocal last_print_t
            if not _is_rank_zero():
                return
            if progress_sec <= 0.0:
                return
            now = time.perf_counter()
            if force or (now - last_print_t) >= float(progress_sec) or done >= total:
                elapsed_str = _format_duration_hms(now - start_t)
                print(f"[WOSACMetrics] progress: {done}/{total} (elapsed {elapsed_str})", flush=True)
                last_print_t = now

        _maybe_print(0, force=True)

        # ------------------------------------------------------------
        # 1) agent_batch 기준으로 시나리오별 크기(sizes) 계산
        #    전제: agent_batch는 시나리오 순서대로 정렬되어 있어야 split이 의미가 유지됩니다.
        # ------------------------------------------------------------
        agent_batch_cpu = agent_batch.detach().to("cpu", dtype=torch.long)  # (N2,)
        if agent_batch_cpu.numel() > 1:
            is_sorted = bool(torch.all(agent_batch_cpu[1:] >= agent_batch_cpu[:-1]))
            if not is_sorted:
                raise ValueError(
                    "agent_batch가 시나리오 순서로 정렬되어 있지 않습니다. "
                    "현재 방식(split)은 '시나리오별로 연속 구간'이라는 전제가 필요합니다."
                )

        sizes: List[int] = torch.bincount(
            agent_batch_cpu, minlength=int(batch_size_now)
        ).to(dtype=torch.long).tolist()

        if int(sum(sizes)) != int(agent_batch_cpu.numel()):
            raise ValueError(
                "시나리오별 sizes 합이 agent_batch 길이와 다릅니다. "
                f"sum(sizes)={sum(sizes)}, N2={int(agent_batch_cpu.numel())}"
            )

        # (기존 sizes 계산까지는 그대로)

        # ✅ (추가) 큰 텐서 전체를 CPU 공유 텐서로 한 번만 변환
        agent_id_cpu_shared = _to_shared_cpu_tensor_for_wosac(agent_id)  # (N2,)
        pred_traj_cpu_shared = _to_shared_cpu_tensor_for_wosac(
            pred_traj)  # (N2, R, T, 2)
        pred_z_cpu_shared = _to_shared_cpu_tensor_for_wosac(
            pred_z)  # (N2, R, T)
        pred_head_cpu_shared = _to_shared_cpu_tensor_for_wosac(
            pred_head)  # (N2, R, T)

        # ✅ 공유 텐서를 시나리오 단위로 split (뷰라서 추가 복사 최소)
        agent_id_splits = torch.split(agent_id_cpu_shared, sizes,
                                      dim=0)  # len=B, each (Ai,)
        pred_traj_splits = torch.split(pred_traj_cpu_shared, sizes,
                                       dim=0)  # len=B, each (Ai,R,T,2)
        pred_z_splits = torch.split(pred_z_cpu_shared, sizes,
                                    dim=0)  # len=B, each (Ai,R,T)
        pred_head_splits = torch.split(pred_head_cpu_shared, sizes,
                                       dim=0)  # len=B, each (Ai,R,T)

        # ------------------------------------------------------------
        # 2) multiprocessing: "포장+점수"를 worker가 수행, 메인은 숫자만 누적
        # ------------------------------------------------------------
        if use_mp_pool:
            pool = self._get_or_create_mp_pool(
                min_processes=int(recommended_p),
                tf_threads=int(tf_threads),
            )

            # 메모리/큐 폭주를 막기 위해 in-flight 작업 수를 제한합니다.
            # 기본: worker 수 * 2
            inflight_limit = int(max(1, int(recommended_p) * 2))

            result_q: py_queue.Queue = py_queue.Queue()

            def _on_success(res: Any) -> None:
                result_q.put(("ok", res))

            def _on_error(err: BaseException) -> None:
                result_q.put(("err", err))

            inflight = 0
            done = 0

            for i in range(total):
                args = (
                    self.wosac_config,
                    str(scenario_files[i]),
                    str(scenario_ids[i]),
                    agent_id_splits[i],
                    pred_traj_splits[i],
                    pred_z_splits[i],
                    pred_head_splits[i],
                    bool(self.ego_only),
                    bool(should_validate),
                )

                pool.apply_async(
                    _compute_scenario_metrics_from_raw_star,
                    args=(args,),
                    callback=_on_success,
                    error_callback=_on_error,
                )
                inflight += 1

                # in-flight가 너무 많아지면 결과를 하나 이상 처리하고 진행
                while inflight >= inflight_limit:
                    status, payload = result_q.get()
                    inflight -= 1
                    if status == "err":
                        raise payload  # worker 예외를 그대로 전파

                    scenario_metrics, z_only = payload
                    done += 1
                    _maybe_print(done, force=False)

                    # 누적(기존 update와 동일)
                    self.scenario_counter += 1
                    self.metametric += scenario_metrics.metametric
                    self.average_displacement_error += scenario_metrics.average_displacement_error
                    self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
                    self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
                    self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
                    self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
                    self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
                    self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
                    self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
                    self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
                    self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
                    self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
                    self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
                    self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
                    self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
                    self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

                    self.z_only_average_displacement_error += tensor(
                        float(z_only.get("average_displacement_error_z_only", 0.0))
                    )
                    self.z_only_min_average_displacement_error += tensor(
                        float(z_only.get("min_average_displacement_error_z_only", 0.0))
                    )

            # 남은 결과 모두 수거
            while inflight > 0:
                status, payload = result_q.get()
                inflight -= 1
                if status == "err":
                    raise payload

                scenario_metrics, z_only = payload
                done += 1
                _maybe_print(done, force=False)

                self.scenario_counter += 1
                self.metametric += scenario_metrics.metametric
                self.average_displacement_error += scenario_metrics.average_displacement_error
                self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
                self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
                self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
                self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
                self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
                self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
                self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
                self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
                self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
                self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
                self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
                self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
                self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
                self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

                self.z_only_average_displacement_error += tensor(
                    float(z_only.get("average_displacement_error_z_only", 0.0))
                )
                self.z_only_min_average_displacement_error += tensor(
                    float(z_only.get("min_average_displacement_error_z_only", 0.0))
                )

            _maybe_print(done, force=True)

        else:
            # --------------------------------------------------------
            # 3) multiprocessing을 안 쓰는 경우: 시나리오를 1개씩 순차 처리
            #    (그래도 포장을 한꺼번에 만들지 않아서 메모리 피크는 줄어듭니다)
            # --------------------------------------------------------
            done = 0
            for i in range(total):
                scenario_metrics, z_only = WOSACMetrics._compute_scenario_metrics_from_raw(
                    self.wosac_config,
                    str(scenario_files[i]),
                    str(scenario_ids[i]),
                    agent_id_splits[i],
                    pred_traj_splits[i],
                    pred_z_splits[i],
                    pred_head_splits[i],
                    bool(self.ego_only),
                    bool(should_validate),
                )

                done += 1
                _maybe_print(done, force=False)

                self.scenario_counter += 1
                self.metametric += scenario_metrics.metametric
                self.average_displacement_error += scenario_metrics.average_displacement_error
                self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
                self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
                self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
                self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
                self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
                self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
                self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
                self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
                self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
                self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
                self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
                self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
                self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
                self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

                self.z_only_average_displacement_error += tensor(
                    float(z_only.get("average_displacement_error_z_only", 0.0))
                )
                self.z_only_min_average_displacement_error += tensor(
                    float(z_only.get("min_average_displacement_error_z_only", 0.0))
                )

            _maybe_print(done, force=True)

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
            should_validate: bool = True,
        ) -> None:
        batch_size_now: int = int(len(scenario_rollouts))

        tf_threads: int = int(
            getattr(self, "_tf_num_threads", _get_wosac_tf_num_threads()))
        recommended_p: int = _recommend_wosac_mp_processes(
            batch_size=batch_size_now,
            tf_num_threads=tf_threads,
        )

        disable_mp: bool = str(os.environ.get("DP_WOSAC_DISABLE_MP",
                                              "0")).strip() == "1"
        use_mp_pool: bool = (not disable_mp) and (recommended_p > 1)

        progress_sec_raw = _read_float_env("DP_WOSAC_PROGRESS_SEC", 60.0)
        progress_sec: float = float(progress_sec_raw)
        progress_sec = float(progress_sec)  # 안전
        if progress_sec <= 0.0:
            progress_sec = 0.0

        if _is_rank_zero():
            print(
                f"[WOSACMetrics] update(): batch_size={batch_size_now}, "
                f"tf_threads={tf_threads}, "
                f"recommended_p={recommended_p}, "
                f"use_mp_pool={use_mp_pool}, "
                f"disable_mp={disable_mp}",
                flush=True,
            )

        if disable_mp:
            self.close_mp_pool()

        total = int(batch_size_now)
        start_t = time.perf_counter()
        last_print_t = start_t

        def _maybe_print(done: int, force: bool = False) -> None:
            nonlocal last_print_t

            if not _is_rank_zero():
                return
            if progress_sec <= 0.0:
                return
            now = time.perf_counter()
            if force or (now -
                         last_print_t) >= float(progress_sec) or done >= total:
                elapsed = now - start_t
                elapsed_str = _format_duration_hms(elapsed)
                print(
                    f"[WOSACMetrics] progress: {done}/{total} (elapsed {elapsed_str})",
                    flush=True)
                last_print_t = now

        _maybe_print(0, force=True)

        if use_mp_pool:
            pool = self._get_or_create_mp_pool(
                min_processes=int(recommended_p),
                tf_threads=int(tf_threads),
            )

            args_iter = zip(
                itertools.repeat(self.wosac_config),
                scenario_files,
                scenario_rollouts,
                itertools.repeat(self.ego_only),
                itertools.repeat(should_validate),  # should_validate
            )

            done = 0
            for scenario_metrics, z_only in pool.imap_unordered(
                    _compute_scenario_metrics_star,
                    args_iter,
                    chunksize=1,
            ):
                done += 1
                _maybe_print(done, force=False)

                self.scenario_counter += 1
                self.metametric += scenario_metrics.metametric
                self.average_displacement_error += scenario_metrics.average_displacement_error
                self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
                self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
                self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
                self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
                self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
                self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
                self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
                self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
                self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
                self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
                self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
                self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
                self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
                self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

                self.z_only_average_displacement_error += tensor(
                    float(z_only.get("average_displacement_error_z_only", 0.0)))
                self.z_only_min_average_displacement_error += tensor(
                    float(
                        z_only.get("min_average_displacement_error_z_only",
                                   0.0)))

            _maybe_print(done, force=True)

        else:
            done = 0
            for _scenario_file, _scenario_rollout in zip(
                    scenario_files, scenario_rollouts):
                scenario_metrics, z_only = self._compute_scenario_metrics(
                    self.wosac_config,
                    _scenario_file,
                    _scenario_rollout,
                    self.ego_only,
                    should_validate,
                )
                done += 1
                _maybe_print(done, force=False)

                self.scenario_counter += 1
                self.metametric += scenario_metrics.metametric
                self.average_displacement_error += scenario_metrics.average_displacement_error
                self.linear_speed_likelihood += scenario_metrics.linear_speed_likelihood
                self.linear_acceleration_likelihood += scenario_metrics.linear_acceleration_likelihood
                self.angular_speed_likelihood += scenario_metrics.angular_speed_likelihood
                self.angular_acceleration_likelihood += scenario_metrics.angular_acceleration_likelihood
                self.distance_to_nearest_object_likelihood += scenario_metrics.distance_to_nearest_object_likelihood
                self.collision_indication_likelihood += scenario_metrics.collision_indication_likelihood
                self.time_to_collision_likelihood += scenario_metrics.time_to_collision_likelihood
                self.distance_to_road_edge_likelihood += scenario_metrics.distance_to_road_edge_likelihood
                self.offroad_indication_likelihood += scenario_metrics.offroad_indication_likelihood
                self.min_average_displacement_error += scenario_metrics.min_average_displacement_error
                self.simulated_collision_rate += scenario_metrics.simulated_collision_rate
                self.simulated_offroad_rate += scenario_metrics.simulated_offroad_rate
                self.traffic_light_violation_likelihood += scenario_metrics.traffic_light_violation_likelihood
                self.simulated_traffic_light_violation_rate += scenario_metrics.simulated_traffic_light_violation_rate

                self.z_only_average_displacement_error += tensor(
                    float(z_only.get("average_displacement_error_z_only", 0.0)))
                self.z_only_min_average_displacement_error += tensor(
                    float(
                        z_only.get("min_average_displacement_error_z_only",
                                   0.0)))

            _maybe_print(done, force=True)

    def compute(self) -> Dict[str, Tensor]:
        metrics_dict = {}
        for k in self.field_names:
            metrics_dict[k] = getattr(self, k) / self.scenario_counter

        mean_metrics = sim_agents_metrics_pb2.SimAgentMetrics(scenario_id="",
                                                              **metrics_dict)
        final_metrics = wosac_metrics.aggregate_metrics_to_buckets(
            self.wosac_config, mean_metrics)

        out_dict = {
            f"{self.prefix}/wosac/realism_meta_metric":
                final_metrics.realism_meta_metric,
            f"{self.prefix}/wosac/kinematic_metrics":
                final_metrics.kinematic_metrics,
            f"{self.prefix}/wosac/interactive_metrics":
                final_metrics.interactive_metrics,
            f"{self.prefix}/wosac/map_based_metrics":
                final_metrics.map_based_metrics,
            f"{self.prefix}/wosac/min_ade":
                final_metrics.min_ade,
            f"{self.prefix}/wosac/scenario_counter":
                self.scenario_counter,
        }
        for k in self.field_names:
            out_dict[f"{self.prefix}/wosac_likelihood/{k}"] = metrics_dict[k]

        out_dict[
            f"{self.prefix}/wosac_likelihood/average_displacement_error_z_only"] = (
                self.z_only_average_displacement_error / self.scenario_counter)
        out_dict[f"{self.prefix}/wosac/min_ade_z_only"] = (
            self.z_only_min_average_displacement_error / self.scenario_counter)
        return out_dict

    @staticmethod
    def load_metrics_config() -> sim_agents_metrics_pb2.SimAgentMetricsConfig:
        """WOSAC 2025 Sim Agents 평가 설정 파일을 읽습니다.

        - 2024 설정 파일(challenge_2024_config.textproto)이 아니라,
          2025 설정 파일(challenge_2025_sim_agents_config.textproto)을 사용합니다.
        - 이렇게 해야 2025 점수 계산 규칙(예: 신호등 위반 항목 포함)이 제대로 반영됩니다.
        """
        config_path = (Path(wosac_metrics.__file__).parent /
                       "challenge_2025_sim_agents_config.textproto")
        with open(config_path, "r") as f:
            config = sim_agents_metrics_pb2.SimAgentMetricsConfig()
            text_format.Parse(f.read(), config)
        return config
