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
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
from typing import Optional
from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.protos import scenario_pb2, sim_agents_submission_pb2
import atexit
import multiprocessing.pool as mp_pool
from multiprocessing.pool import Pool as MpPool


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


def _init_wosac_mp_worker(tf_num_threads: int) -> None:
    """multiprocessing worker 프로세스가 시작될 때 1번 실행되는 초기화 함수입니다.

    목적
    ----
    Pool의 각 worker는 별도 프로세스이므로,
    TensorFlow 설정(T, GPU 비활성화)을 worker 안에서도 확실히 적용하기 위해 씁니다.

    Args:
        tf_num_threads (int):
            TensorFlow 스레드 수 T. shape: ()

    Returns:
        None
    """
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
    v = int(_read_int_env("DP_WOSAC_MP_MAXTASKS_PER_CHILD", 0))
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

    try:
        submission_specs.validate_scenario_rollouts(
            scenario_rollouts=scenario_rollouts,
            original_scenario=scenario,
            challenge_type=submission_specs.ChallengeType.SIM_AGENTS,
        )
    except ValueError as e:
        sid = str(getattr(scenario, "scenario_id", ""))
        raise ValueError(f"[WOSAC 규칙 위반] scenario_id='{sid}' | {e}") from e


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

    def update(
        self,
        scenario_files: List[str],
        scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts],
    ) -> None:

        # ✅ 현재 배치 크기(= 처리할 시나리오 개수)
        batch_size_now: int = int(len(scenario_rollouts))

        # ✅ Option A 규칙으로 worker 수(P) 계산
        tf_threads: int = int(
            getattr(self, "_tf_num_threads", _get_wosac_tf_num_threads()))
        recommended_p: int = _recommend_wosac_mp_processes(
            batch_size=batch_size_now,
            tf_num_threads=tf_threads,
        )

        # 필요하면 환경변수로 mp 자체를 끌 수 있게(안전장치)
        disable_mp: bool = str(os.environ.get("DP_WOSAC_DISABLE_MP",
                                              "0")).strip() == "1"

        use_mp_pool: bool = (not disable_mp) and (recommended_p > 1)
        print(f"[WOSACMetrics] update(): batch_size={batch_size_now}, "
              f"tf_threads={tf_threads}, "
              f"recommended_p={recommended_p}, "
              f"use_mp_pool={use_mp_pool}, "
              f"disable_mp={disable_mp} ")
        # disable_mp가 켜졌으면, 이미 떠 있는 Pool도 정리(원하면)
        if disable_mp:
            self.close_mp_pool()

        if use_mp_pool:
            # ✅ 핵심 변경점:
            # - Pool을 매 update마다 새로 만들지 않고,
            # - self._mp_pool로 "재사용"합니다.
            pool = self._get_or_create_mp_pool(
                min_processes=int(recommended_p),
                tf_threads=int(tf_threads),
            )

            pool_scenario_metrics = pool.starmap(
                self._compute_scenario_metrics,
                zip(
                    itertools.repeat(self.wosac_config),
                    scenario_files,
                    scenario_rollouts,
                    itertools.repeat(self.ego_only),
                ),
            )
        else:
            pool_scenario_metrics = []
            for _scenario, _scenario_rollout in zip(scenario_files,
                                                    scenario_rollouts):
                pool_scenario_metrics.append(
                    self._compute_scenario_metrics(
                        self.wosac_config,
                        _scenario,
                        _scenario_rollout,
                        self.ego_only,
                    ))

        for scenario_metrics, z_only in pool_scenario_metrics:
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
                float(z_only.get("min_average_displacement_error_z_only", 0.0)))

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
