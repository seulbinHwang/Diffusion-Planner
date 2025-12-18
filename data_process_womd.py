import multiprocessing as mp
import os
import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from collections import Counter
import draw_machine
import traceback
import signal
import faulthandler
import time
import sys  # 추가
import contextlib
import multiprocessing.pool as mppool

_LOG_ENABLED = True  # 메인 프로세스는 기본 출력
_LOGGER_PID_VAL = None  # multiprocessing.Value (pid 저장)
# =========================
# 맵/정적 객체 필터 반경(ego 기준)
# =========================
# =========================
# 맵/정적 객체 필터 반경(ego 기준)
# =========================
_USE_FILTER_RADIUS: bool = True
_FILTER_RADIUS_M: float = 150.0


def _require_filter_radius_args(args: Any) -> None:
    """args에 필터링 관련 설정이 있는지 확인합니다.

    이 코드에서는 오직 args_util.py에서 선언한 아래 두 값만 사용하도록 강제합니다.
      - use_filter_radius (bool)
      - filter_radius (float, meter)

    위 두 값이 없으면, 잘못된 설정/환경에서 "조용히 다른 값"을 쓰는 일을 막기 위해
    즉시 에러를 내고 실행을 중단합니다.

    Args:
        args: args_util.get_args()가 반환한 인자 객체(보통 argparse.Namespace)

    Raises:
        RuntimeError: use_filter_radius 또는 filter_radius가 args에 없을 때
    """
    missing: List[str] = []
    if not hasattr(args, "use_filter_radius"):
        missing.append("use_filter_radius")
    if not hasattr(args, "filter_radius"):
        missing.append("filter_radius")

    if len(missing) > 0:
        raise RuntimeError(
            "필수 인자가 args에 없습니다. "
            "args_util.py에 `--use_filter_radius`(bool)와 `--filter_radius`(float)를 반드시 선언해야 합니다. "
            f"missing={missing}"
        )


def _set_filter_radius_settings_from_args(args: Any) -> None:
    """args에서 필터링 설정을 읽어 전역 변수에 반영합니다.

    규칙:
      - use_filter_radius=True:
          _FILTER_RADIUS_M = args.filter_radius (meter)
          이후 맵/정적 객체 필터링을 실제로 수행합니다.
      - use_filter_radius=False:
          _FILTER_RADIUS_M 값은 읽어 두되(로깅/재현성을 위해),
          실제 필터링은 수행하지 않도록 _USE_FILTER_RADIUS=False로 둡니다.

    Args:
        args: args_util.get_args() 결과

    Raises:
        RuntimeError: 필수 인자가 args에 없을 때
        ValueError: use_filter_radius=True인데 filter_radius가 음수인 경우
    """
    global _USE_FILTER_RADIUS, _FILTER_RADIUS_M

    _require_filter_radius_args(args)

    _USE_FILTER_RADIUS = bool(getattr(args, "use_filter_radius"))
    _FILTER_RADIUS_M = float(getattr(args, "filter_radius"))

    # 필터링을 켜는 경우에만 값 검증을 더 강하게 합니다.
    if _USE_FILTER_RADIUS:
        # inf는 "사실상 필터 없음"으로도 쓸 수 있으니 허용합니다.
        # 다만 음수는 의미가 애매해서 바로 막습니다.
        if _FILTER_RADIUS_M < 0.0:
            raise ValueError(
                "use_filter_radius=True인데 filter_radius가 음수입니다. "
                "필터를 끄고 싶으면 use_filter_radius=False를 사용하세요. "
                f"filter_radius={_FILTER_RADIUS_M}"
            )




def _ping() -> str:
    return f"ping pid={os.getpid()}"


def _claim_logger_if_needed() -> None:
    """처음 일을 시작한 워커 1개만 로거로 '선점'해서 출력하도록 함."""
    global _LOG_ENABLED
    if _LOGGER_PID_VAL is None:
        _LOG_ENABLED = True
        return

    with _LOGGER_PID_VAL.get_lock():
        if _LOGGER_PID_VAL.value == 0:
            _LOGGER_PID_VAL.value = os.getpid()

    _LOG_ENABLED = (_LOGGER_PID_VAL.value == os.getpid())


_WORKER_STOP_EVENT = None  # 추가 (spawn 워커에서 init으로 채움)


def _log(msg: str) -> None:
    if not _LOG_ENABLED:
        return
    print(f"[{time.strftime('%H:%M:%S')}] pid={os.getpid()} {msg}",
          file=sys.stderr,
          flush=True)


import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm
from waymo_open_dataset.protos import scenario_pb2
import math
import args_util
import contextlib
import multiprocessing.pool as mppool
import time


def _terminate_pool_hard(
    pool: Optional[mppool.Pool],
    timeout_sec: float = 3.0,
) -> None:
    """Pool을 Ctrl+C 시 '확실하게' 종료합니다.

    1) pool.terminate()로 SIGTERM 전송
    2) timeout_sec 동안 join 시도
    3) 아직 살아있는 워커는 SIGKILL로 강제 종료

    Args:
        pool: multiprocessing Pool (None이면 무시)
        timeout_sec: SIGTERM 후 기다릴 최대 시간(초)
    """
    if pool is None:
        return

    # 1) SIGTERM
    with contextlib.suppress(Exception):
        pool.terminate()

    procs = getattr(pool, "_pool", None) or []
    deadline = time.monotonic() + float(timeout_sec)

    # 2) 일정 시간 기다리기
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(Exception):
            proc.join(timeout=remaining)

    # 3) 남아있으면 SIGKILL
    for proc in procs:
        if proc.is_alive():
            # Python 3.7+ : Process.kill() 지원
            with contextlib.suppress(Exception):
                proc.kill()
            # kill()이 없거나 실패한 환경 대비
            if proc.is_alive():
                with contextlib.suppress(Exception):
                    os.kill(proc.pid, signal.SIGKILL)

    # 마무리 join (짧게)
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.join(timeout=1.0)


def _join_pool_soft(
    pool: Optional[mppool.Pool],
    timeout_sec: float = 10.0,
) -> bool:
    """pool.close() 이후 워커들이 timeout 안에 종료되는지 기다립니다.

    Args:
        pool: multiprocessing Pool
        timeout_sec: 기다릴 최대 시간(초)

    Returns:
        모두 종료되면 True, 일부라도 살아있으면 False
    """
    if pool is None:
        return True
    procs = getattr(pool, "_pool", None) or []
    deadline = time.monotonic() + float(timeout_sec)
    for proc in procs:
        remaining = max(0.0, deadline - time.monotonic())
        with contextlib.suppress(Exception):
            proc.join(timeout=remaining)
    return not any(p.is_alive() for p in procs)


def _install_main_signal_handlers(
    stop_event: Any,
    pool_ref: Dict[str, Optional[mppool.Pool]],
    grace_sec: float = 12.0,
) -> Tuple[Dict[int, Any], Dict[str, Any]]:
    """메인에서 Ctrl+C를 '확실히' 잡아서 stop_event로 전파 + 단계적 종료를 수행합니다.

    - Ctrl+C 1회: stop_event set -> 우아한 종료(워커가 체크 지점에서 빠져나오게)
    - Ctrl+C 2회: 즉시 워커 강제 종료(SIGKILL 포함) 후 종료
    """
    state: Dict[str, Any] = {
        "count": 0,
        "deadline": None,
        "grace_sec": float(grace_sec)
    }
    old = {
        signal.SIGINT: signal.getsignal(signal.SIGINT),
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
    }

    def _handler(signum: int, frame: Any) -> None:
        state["count"] += 1
        stop_event.set()

        if state["deadline"] is None:
            state["deadline"] = time.monotonic() + state["grace_sec"]

        pool = pool_ref.get("pool", None)

        if state["count"] == 1:
            _log(
                "Ctrl+C 감지 -> stop_event 전파(우아한 종료 시도). 한 번 더 누르면 즉시 강제 종료합니다.")
            if pool is not None:
                with contextlib.suppress(Exception):
                    pool.close()
            return

        _log("Ctrl+C 2회 감지 -> 워커 강제 종료(SIGKILL 포함) 후 즉시 종료합니다.")
        _terminate_pool_hard(pool, timeout_sec=1.0)
        os._exit(130)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    # 환경에 따라 block syscall에서 더 잘 빠져나오게
    with contextlib.suppress(Exception):
        signal.siginterrupt(signal.SIGINT, True)
        signal.siginterrupt(signal.SIGTERM, True)

    return old, state


def wrap_angle(angle: torch.Tensor,
               min_val: float = -math.pi,
               max_val: float = math.pi) -> torch.Tensor:
    return min_val + (angle + max_val) % (max_val - min_val)


# =========================
# 설정값 (요구사항 고정)
# =========================

SPLITS: Tuple[str, ...] = ("training", "validation", "testing")

# args_util.py에서 주입받는 길이들(워커 init에서 설정됨)
TIME_LEN: int = 0
FUTURE_LEN: int = 0
SAFETY_LEN: int = 0
LANE_LEN: int = 0

DT_SEC: float = 0.1
EPS: float = 1e-6


def _set_womd_lengths_from_args(args: Any) -> None:
    """args_util.py 인자에서 길이 관련 설정을 가져와 전역 변수에 주입합니다."""
    global TIME_LEN, FUTURE_LEN, SAFETY_LEN, LANE_LEN

    TIME_LEN = int(getattr(args, "time_len"))
    FUTURE_LEN = int(getattr(args, "future_len"))
    SAFETY_LEN = int(getattr(args, "safety_len"))
    LANE_LEN = int(getattr(args, "lane_len"))

    # ✅ 필터 반경은 args_util.py의 (use_filter_radius, filter_radius)만 사용
    _set_filter_radius_settings_from_args(args)

    if TIME_LEN <= 0 or FUTURE_LEN <= 0 or SAFETY_LEN <= 0 or LANE_LEN <= 0:
        raise ValueError(
            f"Invalid lengths: time_len={TIME_LEN}, future_len={FUTURE_LEN}, "
            f"safety_len={SAFETY_LEN}, lane_len={LANE_LEN}"
        )


def _require_womd_lengths_initialized() -> None:
    """워커 init에서 길이값이 세팅되었는지 확인합니다."""
    if TIME_LEN <= 0 or FUTURE_LEN <= 0 or SAFETY_LEN <= 0 or LANE_LEN <= 0:
        raise RuntimeError(
            "WOMD lengths are not initialized. "
            "Make sure Pool initializer calls _set_womd_lengths_from_args().")


# =========================
# 데이터 구조 (맵 파싱용)
# =========================
# =========================
# 데이터 구조 (맵 파싱용)
# =========================
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class BoundarySegmentInfo:
    """Lane 경계 구간 1개를 필요한 정보만 뽑아 저장합니다."""
    lane_start_index: int
    lane_end_index: int
    boundary_feature_id: int


@dataclass(frozen=True)
class LaneInfo:
    """차선 1개에 대한 원본 정보 묶음."""
    lane_id: int
    centerline_xy_global: np.ndarray  # shape: (P, 2), float32/float64

    # ✅ 변경: boundary_feature_id만 저장하지 말고, lane_start/end_index까지 같이 저장
    left_boundary_segments: List[BoundarySegmentInfo]
    right_boundary_segments: List[BoundarySegmentInfo]

    speed_limit_mph: float
    lane_type: int  # lane의 큰 분류(고속도로/일반도로/자전거/미정)


@dataclass(frozen=True)
class ParsedMap:
    """시나리오 맵을 필요한 형태로 미리 모아둔 결과."""
    stop_sign_xy_global: List[np.ndarray]  # each shape: (2,)
    crosswalk_polygons_xy_global: List[np.ndarray]  # each shape: (M,2)
    speed_bump_polygons_xy_global: List[np.ndarray]

    # ✅ 추가: driveway polygon (각각 (M,2))
    driveway_polygons_xy_global: List[np.ndarray]

    lanes: List[LaneInfo]

    # boundary (road_line / road_edge) polyline
    boundary_polylines_xy_global: Dict[int, np.ndarray]  # id -> shape: (K,2)

    # ✅ 추가: boundary id가 road_line인지 road_edge인지 구분
    boundary_id_to_kind: Dict[int, str]  # id -> "road_line" or "road_edge"

    # ✅ 추가: road_line / road_edge 타입(정수 enum) 저장
    road_line_type_by_id: Dict[int, int]  # road_line id -> type int
    road_edge_type_by_id: Dict[int, int]  # road_edge id -> type int

    # ✅ 추가: road_edge id 목록(road_edge 출력용)
    road_edge_ids: List[int]


# =========================
# 기본 유틸
# =========================
def ensure_dir(path: Path) -> None:
    """폴더가 없으면 생성합니다.

    Args:
        path: 만들고 싶은 폴더 경로
    """
    path.mkdir(parents=True, exist_ok=True)


def _proto_points_to_xy_array(points: Iterable[Any]) -> np.ndarray:
    """proto의 점 목록을 (N,2) numpy 배열로 바꿉니다.

    Args:
        points: 각 원소가 x, y 값을 가진 점들의 목록(반복 가능한 형태)

    Returns:
        xy: shape (N,2) float32
    """
    points_list = [(float(p.x), float(p.y)) for p in points]
    if len(points_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(points_list, dtype=np.float32)  # shape (N,2)


def lane_type_value_to_one_hot_4(lane_type_value: int) -> np.ndarray:
    """차선의 큰 종류를 4칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(4칸):
        0: FREEWAY
        1: SURFACE_STREET
        2: BIKE_LANE
        3: UNDEFINED

    Args:
        lane_type_value: Waymo map proto의 lane.type 값(정수)

    Returns:
        one_hot: shape (4,) float32
    """
    # one_hot: np.ndarray, shape (4,)
    one_hot = np.zeros((4,), dtype=np.float32)

    # Waymo map.proto(또는 tf.Example 문서)에서 일반적으로 쓰이는 값:
    # 0: TYPE_UNDEFINED
    # 1: TYPE_FREEWAY
    # 2: TYPE_SURFACE_STREET
    # 3: TYPE_BIKE_LANE
    if lane_type_value == 1:
        one_hot[0] = 1.0
    elif lane_type_value == 2:
        one_hot[1] = 1.0
    elif lane_type_value == 3:
        one_hot[2] = 1.0
    else:
        one_hot[3] = 1.0
    return one_hot


def road_line_type_value_to_one_hot_10(
    road_line_type_value: int,
    has_line: bool,
) -> np.ndarray:
    """차선 경계 '선'의 종류를 10칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(10칸):
        0: BROKEN_SINGLE_WHITE
        1: SOLID_SINGLE_WHITE
        2: SOLID_DOUBLE_WHITE
        3: BROKEN_SINGLE_YELLOW
        4: BROKEN_DOUBLE_YELLOW
        5: SOLID_SINGLE_YELLOW
        6: SOLID_DOUBLE_YELLOW
        7: PASSING_DOUBLE_YELLOW
        8: UNKNOWN
        9: INVALID (해당 방향 선이 아예 없는 경우)

    Args:
        road_line_type_value: Waymo map proto의 road_line.type 값(정수)
        has_line: True면 "선이 있다", False면 "선이 없다(=INVALID)"로 처리

    Returns:
        one_hot: shape (10,) float32
    """
    # one_hot: np.ndarray, shape (10,)
    one_hot = np.zeros((10,), dtype=np.float32)

    if not has_line:
        one_hot[9] = 1.0
        return one_hot

    # Waymo map.proto에서 일반적으로 쓰이는 값:
    # 0: TYPE_UNKNOWN
    # 1..8: 각 선 종류
    mapping = {
        1: 0,  # BROKEN_SINGLE_WHITE
        2: 1,  # SOLID_SINGLE_WHITE
        3: 2,  # SOLID_DOUBLE_WHITE
        4: 3,  # BROKEN_SINGLE_YELLOW
        5: 4,  # BROKEN_DOUBLE_YELLOW
        6: 5,  # SOLID_SINGLE_YELLOW
        7: 6,  # SOLID_DOUBLE_YELLOW
        8: 7,  # PASSING_DOUBLE_YELLOW
        0: 8,  # UNKNOWN
    }
    idx = mapping.get(int(road_line_type_value), 8)  # 모르는 값은 UNKNOWN
    one_hot[idx] = 1.0
    return one_hot


from typing import Dict, List
import numpy as np


def _lane_boundary_type_index_to_one_hot_13(type_index: int) -> np.ndarray:
    """13칸짜리 0/1 벡터에서, 주어진 칸만 1로 켭니다.

    이 함수는 "왼쪽/오른쪽 경계의 대표 타입"을 저장할 때 쓰는 13칸짜리 벡터를 만듭니다.
    벡터는 길이가 13이고, 전부 0으로 시작한 뒤 type_index 위치만 1이 됩니다.

    Args:
        type_index: 0~12 범위의 정수.
            - 0~7  : 선(road_line)에서 타입이 확실한 8가지
            - 8~10 : 도로 가장자리(road_edge) 타입 3가지
            - 11   : 선(road_line)은 있는데 타입을 모르는 경우
            - 12   : 선/가장자리 자체가 아예 없는 경우(완전 없음)

    Returns:
        one_hot: shape (13,) float32
    """
    one_hot = np.zeros((13,), dtype=np.float32)  # shape: (13,)
    idx = int(type_index)
    idx = max(0, min(idx, 12))
    one_hot[idx] = 1.0
    return one_hot


def _boundary_feature_id_to_lane_boundary_type_index_13(
    boundary_feature_id: int,
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> int:
    """boundary_feature_id 1개를 13칸 타입 벡터의 '칸 번호(인덱스)'로 바꿉니다.

    이 함수는 "이 경계가 선(road_line)인지, 도로 가장자리(road_edge)인지"를 먼저 판단하고,
    그 다음에 type 값을 보고 13칸 중 어디에 해당하는지 정합니다.

    13칸 정의(0부터 시작):
        - 0~7:
            선(road_line)의 구체 타입 8가지
            [BROKEN_SINGLE_WHITE, SOLID_SINGLE_WHITE, SOLID_DOUBLE_WHITE,
             BROKEN_SINGLE_YELLOW, BROKEN_DOUBLE_YELLOW, SOLID_SINGLE_YELLOW,
             SOLID_DOUBLE_YELLOW, PASSING_DOUBLE_YELLOW]
        - 8:
            도로 가장자리(road_edge)가 있는데, 어떤 타입인지 모르는 경우
        - 9:
            도로 가장자리(road_edge) - ROAD_EDGE_BOUNDARY
        - 10:
            도로 가장자리(road_edge) - ROAD_EDGE_MEDIAN
        - 11:
            선(road_line)이 있는데, 어떤 타입인지 모르는 경우(UNKNOWN 또는 모르는 값)
        - 12:
            "아예 없음"은 여기서 만들지 않습니다(상위 함수에서 처리)

    Args:
        boundary_feature_id: 경계 feature id (정수)
        boundary_id_to_kind: id -> "road_line" 또는 "road_edge"
        road_line_type_by_id: road_line id -> type 값(정수)
        road_edge_type_by_id: road_edge id -> type 값(정수)

    Returns:
        type_index: 0~11 중 하나.
            (12=invalid은 "경계 자체가 없는 경우"라서 상위 함수에서 따로 처리합니다.)
    """
    fid = int(boundary_feature_id)

    # kind가 없을 수도 있으니(데이터가 이상하거나 일부만 파싱된 경우),
    # type dict를 한 번 더 보고 kind를 추정합니다.
    kind = boundary_id_to_kind.get(fid, "")
    if kind == "":
        if fid in road_edge_type_by_id:
            kind = "road_edge"
        elif fid in road_line_type_by_id:
            kind = "road_line"

    if kind == "road_edge":
        edge_type_value = int(road_edge_type_by_id.get(fid, 0))
        # road_edge 타입:
        # 0: UNKNOWN, 1: BOUNDARY, 2: MEDIAN
        if edge_type_value == 1:
            return 9
        if edge_type_value == 2:
            return 10
        return 8  # edge_TYPE_UNKNOWN

    # 나머지는 "선(road_line)"로 취급 (kind가 비어있어도 여기로 옴)
    line_type_value = int(road_line_type_by_id.get(fid, 0))
    # road_line 타입:
    # 1..8: 구체 타입, 0: UNKNOWN
    mapping_line = {
        1: 0,  # BROKEN_SINGLE_WHITE
        2: 1,  # SOLID_SINGLE_WHITE
        3: 2,  # SOLID_DOUBLE_WHITE
        4: 3,  # BROKEN_SINGLE_YELLOW
        5: 4,  # BROKEN_DOUBLE_YELLOW
        6: 5,  # SOLID_SINGLE_YELLOW
        7: 6,  # SOLID_DOUBLE_YELLOW
        8: 7,  # PASSING_DOUBLE_YELLOW
    }
    return int(mapping_line.get(line_type_value, 11))  # unknown(line)


def _choose_lane_side_boundary_type_one_hot_13(
    boundary_segments: List[BoundarySegmentInfo],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> np.ndarray:
    """lane 한쪽(left/right) 경계의 대표 타입을 13칸 0/1 벡터로 만듭니다.

    한 lane의 한쪽 경계는 "조각(segments)" 여러 개로 들어올 수 있습니다.
    그런데 left_line_type / right_line_type는 (lane_num, 13)처럼
    lane마다 한 줄로 저장해야 하므로, 조각들 중에서 대표 1개를 골라야 합니다.

    대표를 고르는 방법(데이터가 부분적으로만 있는 경우를 최대한 안전하게 처리):
    1) boundary_feature_id가 0 이하이거나, end_index <= start_index인 조각은 무시합니다.
    2) 남은 조각마다 "이 조각이 얼마나 긴 구간을 덮는지"를 (end-start)로 계산해서 가중치로 씁니다.
       - 즉, 더 긴 구간을 덮는 조각이 대표로 뽑히기 쉽습니다.
    3) 각 조각을 13칸 중 어디에 해당하는지 분류해서(선/가장자리 + 타입),
       그 칸의 가중치를 누적합니다.
    4) 누적 가중치가 가장 큰 칸을 대표로 선택합니다.
       - 만약 동점이면, "타입을 아는 쪽"을 우선합니다.
         (예: edge_UNKNOWN(8) vs edge_BOUNDARY(9) 동점이면 9를 선택)
    5) 유효한 조각이 하나도 없으면 "완전 없음"으로 보고 invalid(12)를 1로 합니다.
       - 단, lane_len=10 점 중 일부 구간만 경계가 없는 건 여기서 invalid가 아닙니다.
         (그건 lanes 벡터에서 일부 점이 0으로 되는 '부분 없음' 케이스입니다.)

    Args:
        boundary_segments: lane.left_boundary_segments 또는 lane.right_boundary_segments
        boundary_id_to_kind: id -> "road_line" / "road_edge"
        road_line_type_by_id: road_line id -> type 값
        road_edge_type_by_id: road_edge id -> type 값

    Returns:
        one_hot_13: shape (13,) float32
    """
    # weights: np.ndarray, shape (13,)
    weights = np.zeros((13,), dtype=np.float32)

    for seg in boundary_segments:
        fid = int(seg.boundary_feature_id)
        if fid <= 0:
            continue

        start_idx = int(seg.lane_start_index)
        end_idx = int(seg.lane_end_index)
        if end_idx <= start_idx:
            continue

        weight = float(end_idx - start_idx)  # "덮는 길이" 가중치
        type_index = _boundary_feature_id_to_lane_boundary_type_index_13(
            boundary_feature_id=fid,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )
        weights[int(type_index)] += weight

    # 유효한 조각이 1개도 없으면 -> invalid(12)
    if float(weights.sum()) <= 0.0:
        return _lane_boundary_type_index_to_one_hot_13(12)

    max_w = float(weights.max())
    candidate_indices = np.where(weights == max_w)[0].astype(np.int64).tolist()

    # 동점 처리 우선순위:
    # - "타입이 확실한 칸"(0~7, 9~10)을 먼저
    # - 그 다음 "타입을 모르는 칸"(8, 11)
    # - invalid(12)은 여기 후보에 보통 안 들어오지만, 방어적으로 가장 뒤
    def _priority_rank(idx: int) -> int:
        idx_i = int(idx)
        if 0 <= idx_i <= 7:
            return 0
        if idx_i in (9, 10):
            return 0
        if idx_i in (8, 11):
            return 1
        return 2

    chosen = sorted(candidate_indices,
                    key=lambda x: (_priority_rank(int(x)), int(x)))[0]
    return _lane_boundary_type_index_to_one_hot_13(int(chosen))


def road_edge_type_value_to_one_hot_3(road_edge_type_value: int) -> np.ndarray:
    """road_edge의 종류를 3칸짜리 0/1 벡터로 바꿉니다.

    출력 순서(3칸):
        0: TYPE_UNKNOWN
        1: TYPE_ROAD_EDGE_BOUNDARY
        2: TYPE_ROAD_EDGE_MEDIAN

    Args:
        road_edge_type_value: Waymo map proto의 road_edge.type 값(정수)

    Returns:
        one_hot: shape (3,) float32
    """
    # one_hot: np.ndarray, shape (3,)
    one_hot = np.zeros((3,), dtype=np.float32)

    # 보통 값:
    # 0: UNKNOWN
    # 1: BOUNDARY
    # 2: MEDIAN
    if road_edge_type_value == 1:
        one_hot[1] = 1.0
    elif road_edge_type_value == 2:
        one_hot[2] = 1.0
    else:
        one_hot[0] = 1.0
    return one_hot


def _choose_road_line_type_for_lane_side(
    boundary_feature_ids: List[int],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
) -> Tuple[int, bool]:
    """lane의 한쪽(왼쪽/오른쪽)에서 '대표 선 종류'를 하나 고릅니다.

    처리 규칙(한 lane의 한쪽에 boundary_feature_id가 여러 개 있을 수 있어서 필요):
    1) boundary_feature_id 중 "road_line"인 것만 후보로 모읍니다.
    2) 후보가 하나도 없으면 => has_line=False (INVALID)
    3) 후보가 있는데, UNKNOWN(0) 말고 다른 값이 있으면:
       - UNKNOWN이 아닌 값들 중 "가장 많이 나온 값"을 대표로 씁니다.
    4) 후보가 전부 UNKNOWN(0)이면 => 대표값=0, has_line=True (UNKNOWN)

    Args:
        boundary_feature_ids: lane의 왼쪽 또는 오른쪽 boundary_feature_id 목록
        boundary_id_to_kind: boundary id가 road_line인지 road_edge인지 구분하는 dict
        road_line_type_by_id: road_line id -> road_line.type(정수) dict

    Returns:
        chosen_type_value: 대표 road_line.type 값(정수). 선이 없으면 0
        has_line: True면 선이 있음, False면 선이 없음(INVALID)
    """
    # 후보 타입들: List[int]
    candidate_types: List[int] = []
    for fid in boundary_feature_ids:
        if boundary_id_to_kind.get(int(fid), "") != "road_line":
            continue
        if int(fid) not in road_line_type_by_id:
            continue
        candidate_types.append(int(road_line_type_by_id[int(fid)]))

    if len(candidate_types) == 0:
        return 0, False  # 선이 아예 없음(INVALID)

    non_unknown = [t for t in candidate_types if t != 0]
    if len(non_unknown) == 0:
        return 0, True  # 선은 있는데 타입을 모름(UNKNOWN)

    # 가장 많이 나온 값 선택(동점이면 Counter가 먼저 본 것 기준으로 안정적으로 뽑힘)
    chosen = Counter(non_unknown).most_common(1)[0][0]
    return int(chosen), True


def build_lane_type_one_hot_array(lanes: List[LaneInfo]) -> np.ndarray:
    """lane 목록을 (lane_num, 4) lane_type 배열로 만듭니다.

    Args:
        lanes: LaneInfo 리스트

    Returns:
        lane_type: shape (lane_num, 4) float32
    """
    lane_num = len(lanes)
    lane_type = np.zeros((lane_num, 4), dtype=np.float32)  # shape (L,4)
    for i, lane in enumerate(lanes):
        lane_type[i] = lane_type_value_to_one_hot_4(int(lane.lane_type))
    return lane_type


def build_lane_line_type_arrays(
    lanes: List[LaneInfo],
    boundary_id_to_kind: Dict[int, str],
    road_line_type_by_id: Dict[int, int],
    road_edge_type_by_id: Dict[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """각 lane의 왼쪽/오른쪽 경계 타입을 (lane_num, 13)으로 만듭니다.

    left_line_type / right_line_type는 lanes에서 쓰는 "왼쪽/오른쪽 경계"가
    선(road_line)인지 도로 가장자리(road_edge)인지까지 포함해서 대표 타입을 저장합니다.

    13칸 정의(0부터 시작):
        0~7:
            선(road_line)의 8가지 타입
            [BROKEN_SINGLE_WHITE, SOLID_SINGLE_WHITE, SOLID_DOUBLE_WHITE,
             BROKEN_SINGLE_YELLOW, BROKEN_DOUBLE_YELLOW, SOLID_SINGLE_YELLOW,
             SOLID_DOUBLE_YELLOW, PASSING_DOUBLE_YELLOW]
        8:
            도로 가장자리(road_edge)가 있으나 타입을 모름
        9:
            도로 가장자리(road_edge) - ROAD_EDGE_BOUNDARY
        10:
            도로 가장자리(road_edge) - ROAD_EDGE_MEDIAN
        11:
            선(road_line)은 있으나 타입을 모름(UNKNOWN 또는 모르는 값)
        12:
            경계 자체가 아예 없음(선/가장자리 모두 없음)
            ※ lane_len=10 점 중 일부만 경계가 없어서 lanes 벡터가 0이 되는 경우는 여기에 해당하지 않습니다.

    Args:
        lanes: LaneInfo 리스트
        boundary_id_to_kind: boundary id -> "road_line" / "road_edge"
        road_line_type_by_id: road_line id -> type 값(정수)
        road_edge_type_by_id: road_edge id -> type 값(정수)

    Returns:
        left_line_type: shape (lane_num, 13) float32
        right_line_type: shape (lane_num, 13) float32
    """
    lane_num = int(len(lanes))

    left_line_type = np.zeros((lane_num, 13), dtype=np.float32)  # shape: (L,13)
    right_line_type = np.zeros((lane_num, 13),
                               dtype=np.float32)  # shape: (L,13)

    for i, lane in enumerate(lanes):
        left_line_type[i] = _choose_lane_side_boundary_type_one_hot_13(
            boundary_segments=lane.left_boundary_segments,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )
        right_line_type[i] = _choose_lane_side_boundary_type_one_hot_13(
            boundary_segments=lane.right_boundary_segments,
            boundary_id_to_kind=boundary_id_to_kind,
            road_line_type_by_id=road_line_type_by_id,
            road_edge_type_by_id=road_edge_type_by_id,
        )

    return left_line_type, right_line_type


def build_road_edge_points_and_types(
    road_edge_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    road_edge_type_by_id: Dict[int, int],
    ego_xy_global: np.ndarray,  # shape (2,)
    ego_yaw_global: float,
    safety_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """road_edge를 (점들, 타입)으로 캐싱용 배열로 만듭니다.

    - road_edge는 "점들의 줄"로 주어지므로, 전체 길이를 따라 같은 간격으로 safety_len개 점을 뽑습니다.
    - 좌표는 ego 기준으로 바꿉니다.

    Args:
        road_edge_ids: road_edge feature id 목록
        boundary_polylines_xy_global: boundary id -> polyline (K,2)
        road_edge_type_by_id: road_edge id -> type 값(정수)
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 방향(라디안)
        safety_len: 출력 점 개수(요구사항: 10)

    Returns:
        road_edge_points: shape (n_road_edge, safety_len, 2) float32
        road_edge_type: shape (n_road_edge, 3) float32 (one-hot)
    """
    n_road_edge = len(road_edge_ids)
    road_edge_points = np.zeros((n_road_edge, safety_len, 2),
                                dtype=np.float32)  # (E,10,2)
    road_edge_type = np.zeros((n_road_edge, 3), dtype=np.float32)  # (E,3)

    for i, edge_id in enumerate(road_edge_ids):
        poly_xy_g = boundary_polylines_xy_global.get(
            int(edge_id), np.zeros((0, 2), dtype=np.float32))  # shape (K,2)

        poly_xy_l = transform_points_global_to_ego_local(
            poly_xy_g, ego_xy_global, ego_yaw_global)  # shape (K,2)

        sampled = resample_polyline_equal_distance(
            poly_xy_l, num_samples=safety_len,
            closed=False)  # shape (safety_len,2)

        road_edge_points[i] = sampled
        edge_type_value = int(road_edge_type_by_id.get(int(edge_id), 0))
        road_edge_type[i] = road_edge_type_value_to_one_hot_3(edge_type_value)

    return road_edge_points, road_edge_type


def _extract_driveway_polygon_xy_global(driveway_msg: Any) -> np.ndarray:
    """driveway 메시지에서 (M,2) polygon 점들을 뽑습니다.

    데이터/버전에 따라 내부 필드명이 약간 다를 수 있어,
    polygon이 있으면 polygon을 우선 사용하고,
    없으면 polyline을 대체로 사용합니다.

    Args:
        driveway_msg: mf.driveway (proto 메시지)

    Returns:
        polygon_xy: shape (M,2) float32
    """
    if hasattr(driveway_msg, "polygon") and len(getattr(driveway_msg,
                                                        "polygon")) > 0:
        return _proto_points_to_xy_array(getattr(driveway_msg, "polygon"))
    if hasattr(driveway_msg, "polyline") and len(
            getattr(driveway_msg, "polyline")) > 0:
        return _proto_points_to_xy_array(getattr(driveway_msg, "polyline"))
    return np.zeros((0, 2), dtype=np.float32)


def list_tfrecord_files(split_dir: Path) -> List[Path]:
    """split 폴더 안의 모든 tfrecord 파일을 모읍니다.

    파일 확장자가 다양할 수 있어서(예: .tfrecord, .tfrecords, .gz),
    "파일"이면 모두 대상으로 잡되, 폴더는 제외합니다.

    Args:
        split_dir: 예) /home/user/womd_v1_3/scenario/training

    Returns:
        tfrecord 파일 경로 리스트
    """
    all_paths: List[Path] = sorted(
        [p for p in split_dir.glob("*") if p.is_file()])
    return all_paths


def guess_tfrecord_compression(file_path: Path) -> str:
    """tfrecord 파일의 압축 여부를 파일명으로 추정합니다.

    Args:
        file_path: tfrecord 파일 경로

    Returns:
        TensorFlow TFRecordDataset에 넣을 compression_type 문자열.
        - ".gz" 로 끝나면 "GZIP"
        - 아니면 "" (압축 없음)
    """
    if file_path.name.endswith(".gz"):
        return "GZIP"
    return ""


def parse_scenario_from_bytes(record_bytes: bytes) -> scenario_pb2.Scenario:
    """TFRecord의 한 레코드(bytes)를 Scenario proto로 변환합니다.

    Args:
        record_bytes: TFRecord에서 꺼낸 1개 레코드의 raw bytes

    Returns:
        Scenario proto 객체
    """
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(record_bytes)
    return scenario


# =========================
# 좌표 변환 (ego 기준)
# =========================
def transform_points_global_to_ego_local(
    points_xy_global: np.ndarray,  # shape: (..., 2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
) -> np.ndarray:
    """전역 좌표계의 (x,y) 점들을 ego 기준 좌표계로 바꿉니다.

    변환 규칙:
    1) ego 위치를 원점으로 만들기 위해 (point - ego_pos)를 합니다.
    2) ego가 바라보는 방향이 +x가 되도록, -ego_yaw 만큼 회전합니다.

    Args:
        points_xy_global: 전역 좌표계 점들, 마지막 차원은 (x,y)
        ego_xy_global: ego의 전역 위치 (x,y)
        ego_yaw_global: ego의 전역 yaw (라디안)

    Returns:
        ego 기준 좌표계 점들, shape은 입력과 같고 마지막 차원은 (x,y)
    """
    # points_xy_global: np.ndarray, shape (..., 2)
    # ego_xy_global: np.ndarray, shape (2,)
    delta_xy = points_xy_global - ego_xy_global  # shape (..., 2)

    cos_yaw = float(np.cos(ego_yaw_global))
    sin_yaw = float(np.sin(ego_yaw_global))

    # local_x =  dx*cos + dy*sin
    # local_y = -dx*sin + dy*cos
    local_x = delta_xy[..., 0] * cos_yaw + delta_xy[..., 1] * sin_yaw
    local_y = -delta_xy[..., 0] * sin_yaw + delta_xy[..., 1] * cos_yaw

    out = np.stack([local_x, local_y], axis=-1).astype(np.float32)
    return out


from typing import Dict, List, Tuple
import numpy as np


def resample_polyline_equal_distance_with_segment_indices(
    points_xy: np.ndarray,  # shape: (N, 2)
    num_samples: int,
    closed: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """점들의 줄을 같은 간격으로 num_samples개로 만들면서, 각 샘플이 '원본의 어느 구간'에 있었는지도 함께 반환합니다.

    '구간(segment index)' 정의:
      - 원본 points_xy의 i번째 점과 (i+1)번째 점 사이를 i번째 구간이라고 봅니다.
      - seg_indices[j] = i 라면, sampled_xy[j]는 원본 (i ~ i+1) 사이에 위치했다는 뜻입니다.

    Args:
        points_xy: (N,2) 입력 점들.
        num_samples: 뽑을 점 개수.
        closed: 닫힌 모양이면 True.

    Returns:
        sampled_xy: (num_samples,2) float32. 샘플링된 점들.
        seg_indices: (num_samples,) int64.
            sampled_xy의 각 점이 원본 polyline의 몇 번째 구간(i~i+1)에 있었는지.
            points_xy가 비어있으면 전부 -1.
    """
    n = int(points_xy.shape[0])
    if n == 0:
        return (
            np.zeros((int(num_samples), 2), dtype=np.float32),
            -np.ones((int(num_samples),), dtype=np.int64),
        )

    if n == 1:
        sampled = np.repeat(points_xy.astype(np.float32),
                            repeats=int(num_samples),
                            axis=0)
        seg_indices = np.zeros((int(num_samples),), dtype=np.int64)
        return sampled, seg_indices

    points_work = points_xy
    if bool(closed):
        if not np.allclose(points_work[0], points_work[-1]):
            points_work = np.vstack([points_work, points_work[0]])

    diffs = points_work[1:] - points_work[:-1]  # (M,2)
    seg_lens = np.linalg.norm(diffs, axis=1).astype(np.float32)  # (M,)
    total_len = float(seg_lens.sum())

    if total_len < EPS:
        sampled = np.repeat(points_work[:1].astype(np.float32),
                            repeats=int(num_samples),
                            axis=0)
        seg_indices = np.zeros((int(num_samples),), dtype=np.int64)
        return sampled, seg_indices

    cum_len = np.concatenate(
        [np.array([0.0], dtype=np.float32),
         np.cumsum(seg_lens)],
        axis=0,
    )  # (M+1,)

    if bool(closed):
        target = np.linspace(0.0,
                             total_len,
                             int(num_samples),
                             endpoint=False,
                             dtype=np.float32)
    else:
        target = np.linspace(0.0,
                             total_len,
                             int(num_samples),
                             endpoint=True,
                             dtype=np.float32)

    seg_idx = np.searchsorted(cum_len, target, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, int(seg_lens.shape[0]) - 1).astype(np.int64)

    seg_start = points_work[seg_idx].astype(np.float32)  # (num_samples,2)
    seg_vec = diffs[seg_idx].astype(np.float32)  # (num_samples,2)
    seg_len = seg_lens[seg_idx].astype(np.float32)  # (num_samples,)

    alpha = (target - cum_len[seg_idx]) / np.maximum(seg_len,
                                                     EPS)  # (num_samples,)
    sampled = seg_start + seg_vec * alpha[:, None]  # (num_samples,2)

    return sampled.astype(np.float32), seg_idx


def map_center_segment_indices_to_boundary_feature_ids(
    center_seg_indices: np.ndarray,  # shape: (lane_len,)
    boundary_segments: List[BoundarySegmentInfo],
) -> np.ndarray:
    """센터 샘플이 속한 구간 인덱스로, 해당하는 boundary_feature_id를 찾습니다.

    매칭 규칙:
      - center_seg_idx = i 일 때,
        boundary segment가 lane_start_index <= i < lane_end_index 범위를 덮으면 매칭으로 봅니다.

    Args:
        center_seg_indices: (lane_len,) int64.
        boundary_segments: BoundarySegmentInfo 리스트.

    Returns:
        feature_ids: (lane_len,) int64.
            없으면 -1.
    """
    lane_len = int(center_seg_indices.shape[0])
    out = -np.ones((lane_len,), dtype=np.int64)
    if lane_len == 0 or len(boundary_segments) == 0:
        return out

    starts = np.asarray([int(s.lane_start_index) for s in boundary_segments],
                        dtype=np.int64)  # (S,)
    ends = np.asarray([int(s.lane_end_index) for s in boundary_segments],
                      dtype=np.int64)  # (S,)
    fids = np.asarray([int(s.boundary_feature_id) for s in boundary_segments],
                      dtype=np.int64)  # (S,)

    valid_seg = (fids > 0) & (ends > starts)
    if not bool(np.any(valid_seg)):
        return out

    starts = starts[valid_seg]
    ends = ends[valid_seg]
    fids = fids[valid_seg]

    order = np.argsort(starts, kind="mergesort")
    starts = starts[order]
    ends = ends[order]
    fids = fids[order]

    idx = center_seg_indices.astype(np.int64)  # (lane_len,)
    pos = np.searchsorted(starts, idx, side="right") - 1  # (lane_len,)

    valid_pos = (pos >= 0) & (idx >= 0)
    pos_clip = np.clip(pos, 0, int(starts.shape[0]) - 1)

    in_range = valid_pos & (idx < ends[pos_clip])
    out[in_range] = fids[pos_clip[in_range]]
    return out


def compute_center_tangent_and_left_normals(
        center_sampled_xy: np.ndarray,  # shape: (lane_len,2)
) -> Tuple[np.ndarray, np.ndarray]:
    """센터 샘플 점들에서 '진행 방향'과 '왼쪽 방향' 단위벡터를 계산합니다.

    - 진행 방향: 인접한 점 차이로 근사합니다.
      * 마지막 점은 뒤쪽 차이를 사용합니다.
    - 왼쪽 방향: 진행 방향 (dx,dy)를 90도 회전한 (-dy, dx) 입니다.

    Args:
        center_sampled_xy: (lane_len,2) float32.

    Returns:
        tangent_unit: (lane_len,2) float32. 진행 방향 단위벡터.
        left_normal_unit: (lane_len,2) float32. 왼쪽 방향 단위벡터.
    """
    lane_len = int(center_sampled_xy.shape[0])
    tangent = np.zeros((lane_len, 2), dtype=np.float32)

    if lane_len == 0:
        return tangent, tangent

    if lane_len == 1:
        tangent[0] = np.array([1.0, 0.0], dtype=np.float32)
    else:
        tangent[:-1] = center_sampled_xy[1:] - center_sampled_xy[:-1]
        tangent[-1] = center_sampled_xy[-1] - center_sampled_xy[-2]

    norm = np.linalg.norm(tangent, axis=1).astype(np.float32)  # (lane_len,)
    good = norm > EPS

    tangent_unit = np.zeros_like(tangent, dtype=np.float32)
    tangent_unit[good] = tangent[good] / norm[good, None]
    tangent_unit[~good] = np.array([1.0, 0.0], dtype=np.float32)

    left_normal_unit = np.stack([-tangent_unit[:, 1], tangent_unit[:, 0]],
                                axis=1).astype(np.float32)
    return tangent_unit, left_normal_unit


def closest_points_on_polyline(
        query_points_xy: np.ndarray,  # shape: (Q,2)
        polyline_xy: np.ndarray,  # shape: (K,2)
) -> Tuple[np.ndarray, np.ndarray]:
    """여러 점에서 polyline(선분들의 모음)까지의 가장 가까운 위치를 한 번에 구합니다.

    Args:
        query_points_xy: (Q,2) float32.
        polyline_xy: (K,2) float32.

    Returns:
        closest_xy: (Q,2) float32. polyline 위의 가장 가까운 점.
        dist2: (Q,) float32. 거리^2.
            polyline이 비어있으면 dist2는 inf.
    """
    q = query_points_xy.astype(np.float32)
    k = int(polyline_xy.shape[0])

    if k == 0:
        return (
            np.zeros_like(q, dtype=np.float32),
            np.full((int(q.shape[0]),), np.inf, dtype=np.float32),
        )

    if k == 1:
        p = polyline_xy[0].astype(np.float32)
        closest = np.repeat(p[None, :], repeats=int(q.shape[0]), axis=0)
        diff = q - closest
        dist2 = np.sum(diff * diff, axis=1)
        return closest.astype(np.float32), dist2.astype(np.float32)

    p0 = polyline_xy[:-1].astype(np.float32)  # (M,2)
    p1 = polyline_xy[1:].astype(np.float32)  # (M,2)
    seg = (p1 - p0).astype(np.float32)  # (M,2)
    seg_len2 = np.sum(seg * seg, axis=1).astype(np.float32)  # (M,)
    seg_len2_safe = np.maximum(seg_len2, EPS).astype(np.float32)

    diff = q[:, None, :] - p0[None, :, :]  # (Q,M,2)
    t = (np.sum(diff * seg[None, :, :], axis=2) /
         seg_len2_safe[None, :]).astype(np.float32)  # (Q,M)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    closest_all = p0[None, :, :] + t[:, :, None] * seg[None, :, :]  # (Q,M,2)
    d = q[:, None, :] - closest_all
    dist2_all = np.sum(d * d, axis=2).astype(np.float32)  # (Q,M)

    best = np.argmin(dist2_all, axis=1)  # (Q,)
    closest = closest_all[np.arange(int(q.shape[0])), best]
    dist2 = dist2_all[np.arange(int(q.shape[0])), best]
    return closest.astype(np.float32), dist2.astype(np.float32)


def compute_lane_boundary_vectors_for_side(
    center_sampled_xy_local: np.ndarray,  # shape: (lane_len,2)
    center_seg_indices: np.ndarray,  # shape: (lane_len,)
    center_left_normals_unit: np.ndarray,  # shape: (lane_len,2)
    boundary_segments: List[BoundarySegmentInfo],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
    boundary_polyline_local_cache: Dict[int, np.ndarray],
    side: str,
    max_dist_road_line_m: float = 8.0,
    max_dist_road_edge_m: float = 15.0,
) -> np.ndarray:
    """lane의 한쪽(left/right)에 대해 center 샘플 점 -> 경계 polyline 최소거리 벡터를 구합니다.

    Args:
        center_sampled_xy_local: (lane_len,2) float32. center 샘플 점(ego 기준).
        center_seg_indices: (lane_len,) int64. 각 샘플이 원본 center polyline의 어느 구간(i~i+1)인지.
        center_left_normals_unit: (lane_len,2) float32. 각 샘플에서의 '왼쪽 방향' 단위벡터.
        boundary_segments: left_boundaries 또는 right_boundaries 리스트.
        boundary_polylines_xy_global: boundary_feature_id -> (K,2) global polyline dict.
        boundary_id_to_kind: boundary_feature_id -> "road_line" or "road_edge".
        ego_xy_global: (2,) ego 전역 위치.
        ego_yaw_global: ego 전역 yaw(rad).
        boundary_polyline_local_cache: boundary_feature_id -> (K,2) ego local cache.
        side: "left" 또는 "right".
        max_dist_road_line_m: road_line 허용 최대 거리.
        max_dist_road_edge_m: road_edge 허용 최대 거리(더 넉넉).

    Returns:
        vec_xy: (lane_len,2) float32.
            못 찾으면 (0,0).
    """
    lane_len = int(center_sampled_xy_local.shape[0])
    out = np.zeros((lane_len, 2), dtype=np.float32)
    if lane_len == 0:
        return out

    feature_ids = map_center_segment_indices_to_boundary_feature_ids(
        center_seg_indices=center_seg_indices,
        boundary_segments=boundary_segments,
    )  # (lane_len,)

    valid_mask = feature_ids > 0
    if not bool(np.any(valid_mask)):
        return out

    # 성능: 같은 boundary_feature_id를 공유하는 점들을 묶어서 계산
    unique_fids = np.unique(feature_ids[valid_mask]).astype(np.int64)
    for fid in unique_fids.tolist():
        idxs = np.where(feature_ids == int(fid))[0]
        if idxs.size == 0:
            continue

        # polyline ego 변환 캐시
        poly_local = boundary_polyline_local_cache.get(int(fid), None)
        if poly_local is None:
            poly_global = boundary_polylines_xy_global.get(int(fid), None)
            if poly_global is None:
                continue
            poly_local = transform_points_global_to_ego_local(
                poly_global, ego_xy_global, ego_yaw_global)  # (K,2)
            boundary_polyline_local_cache[int(fid)] = poly_local.astype(
                np.float32)

        if poly_local.ndim != 2 or poly_local.shape[1] != 2 or poly_local.shape[
                0] == 0:
            continue

        query = center_sampled_xy_local[idxs].astype(np.float32)  # (Q,2)
        closest, dist2 = closest_points_on_polyline(query,
                                                    poly_local)  # (Q,2), (Q,)
        vec = (closest - query).astype(np.float32)  # (Q,2)
        dist = np.sqrt(dist2).astype(np.float32)  # (Q,)

        kind = boundary_id_to_kind.get(int(fid), "road_line")
        max_dist = float(
            max_dist_road_edge_m) if kind == "road_edge" else float(
                max_dist_road_line_m)

        invalid = dist > max_dist

        # 방향 체크: dot(vec, left_normal) 부호로 left/right 판별
        normals = center_left_normals_unit[idxs].astype(np.float32)  # (Q,2)
        dot = np.sum(vec * normals, axis=1).astype(np.float32)  # (Q,)

        if side == "left":
            invalid = np.logical_or(invalid, dot <= 0.0)
        elif side == "right":
            invalid = np.logical_or(invalid, dot >= 0.0)
        else:
            raise ValueError(f"side must be 'left' or 'right'. got={side}")

        if bool(np.any(invalid)):
            vec[invalid] = 0.0

        out[idxs] = vec

    return out


def transform_vectors_global_to_ego_local(
    vectors_xy_global: np.ndarray,  # shape: (..., 2)
    ego_yaw_global: float,
) -> np.ndarray:
    """전역 좌표계의 (vx,vy) 같은 '방향/속도 벡터'를 ego 기준으로 바꿉니다.

    점 변환과 달리, 벡터는 위치 이동(빼기)을 하지 않고 회전만 합니다.

    Args:
        vectors_xy_global: 전역 좌표계 벡터들, 마지막 차원은 (x,y)
        ego_yaw_global: ego의 전역 yaw (라디안)

    Returns:
        ego 기준 좌표계 벡터들, shape은 입력과 같고 마지막 차원은 (x,y)
    """
    # vectors_xy_global: np.ndarray, shape (..., 2)
    cos_yaw = float(np.cos(ego_yaw_global))
    sin_yaw = float(np.sin(ego_yaw_global))

    local_x = vectors_xy_global[..., 0] * cos_yaw + vectors_xy_global[
        ..., 1] * sin_yaw
    local_y = -vectors_xy_global[..., 0] * sin_yaw + vectors_xy_global[
        ..., 1] * cos_yaw

    out = np.stack([local_x, local_y], axis=-1).astype(np.float32)
    return out


def wrap_angle_np(angles_rad: np.ndarray) -> np.ndarray:
    """각도(rad)를 [-pi, pi) 범위로 정리합니다.

    Args:
        angles_rad: 각도 배열 (어떤 shape이든 가능)

    Returns:
        같은 shape의 각도 배열 (float32)
    """
    wrapped = (angles_rad + np.pi) % (2.0 * np.pi) - np.pi
    return wrapped.astype(np.float32)


def wrap_angle_with_repo_fn(angles_rad: np.ndarray) -> np.ndarray:
    """레포에 있는 wrap_angle(torch 기반)을 이용해 각도를 정리합니다.

    Args:
        angles_rad: 각도 배열 (shape 자유)

    Returns:
        [-pi, pi) 범위로 정리된 각도 (float32)
    """
    # angles_rad_t: torch.Tensor, shape same as angles_rad
    angles_rad_t = torch.from_numpy(angles_rad.astype(np.float32))
    wrapped_t = wrap_angle(angles_rad_t)
    return wrapped_t.numpy().astype(np.float32)


# =========================
# 보간(빈 프레임 채우기)
# =========================
from typing import Tuple
import numpy as np


def interpolate_linear_sequence(
    values: np.ndarray,  # shape: (T, D)
    valid_mask: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray]:
    """유효한 시점들 사이의 빈 칸을 '직선으로' 채웁니다. (numpy만 사용)

    이 함수는 길이가 짧은 시계열(T=21/80 같은)에서, 무거운 외부 보간 도구를 매번 만드는 대신
    numpy의 간단한 방식으로 빠르게 채우는 버전입니다.

    규칙(기존과 동일):
    - 유효한 값이 2개 이상이면:
      - 첫 유효 시점 ~ 마지막 유효 시점 사이를 직선으로 연결해 채웁니다.
      - 그 구간만 유효(True)로 표시하고, 구간 밖은 0으로 둡니다.
    - 유효한 값이 1개뿐이면:
      - 그 시점만 채우고 나머지는 0으로 둡니다.
    - 유효한 값이 0개면:
      - 전부 0으로 둡니다.

    Args:
        values: (T, D) 값 배열.
        valid_mask: (T,) 유효 여부. True인 시점만 '실제 값이 있다'고 봅니다.

    Returns:
        filled_values: (T, D) float32. 채워진 결과.
        filled_valid_mask: (T,) bool. 채워진 구간을 True로 표시한 마스크.
    """
    # values: np.ndarray, shape (T, D)
    # valid_mask: np.ndarray, shape (T,)
    time_len = int(values.shape[0])
    feat_dim = int(values.shape[1]) if values.ndim == 2 else 0

    out = np.zeros((time_len, feat_dim), dtype=np.float32)  # shape: (T, D)
    out_valid = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    if time_len == 0 or feat_dim == 0:
        return out, out_valid

    valid_indices = np.flatnonzero(valid_mask.astype(bool)).astype(np.int64)  # shape: (K,)
    if valid_indices.size == 0:
        return out, out_valid

    if valid_indices.size == 1:
        t = int(valid_indices[0])
        out[t] = values[t].astype(np.float32)
        out_valid[t] = True
        return out, out_valid

    t_start = int(valid_indices[0])
    t_end = int(valid_indices[-1])

    # 보간할 시간 구간
    t_query = np.arange(t_start, t_end + 1, dtype=np.float32)  # shape: (M,)
    t_known = valid_indices.astype(np.float32)  # shape: (K,)

    known_values = values[valid_mask.astype(bool)].astype(np.float32)  # shape: (K, D)

    # D가 보통 2~3처럼 작기 때문에, D만큼의 작은 루프는 부담이 거의 없습니다.
    out_slice = np.zeros((int(t_query.shape[0]), feat_dim), dtype=np.float32)  # shape: (M, D)
    for d in range(feat_dim):
        out_slice[:, d] = np.interp(t_query, t_known, known_values[:, d]).astype(np.float32)

    out[t_start:t_end + 1] = out_slice
    out_valid[t_start:t_end + 1] = True
    return out, out_valid



from typing import Tuple
import numpy as np


def interpolate_linear_angle(
    angles_rad: np.ndarray,  # shape: (T,)
    valid_mask: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray]:
    """각도 시계열의 빈 칸을 직선으로 채웁니다. (numpy만 사용)

    각도는 -pi와 pi 근처에서 값이 갑자기 튀어 보일 수 있습니다.
    그래서 유효한 각도들만 뽑아서 "자연스럽게 이어지도록" 한 번 펼친 뒤,
    직선으로 채우고 마지막에 [-pi, pi) 범위로 다시 정리합니다.

    Args:
        angles_rad: (T,) 각도 배열(라디안).
        valid_mask: (T,) 유효 여부.

    Returns:
        filled_angles: (T,) float32. 채워진 각도.
        filled_valid_mask: (T,) bool. 채워진 구간을 True로 표시한 마스크.
    """
    # angles_rad: np.ndarray, shape (T,)
    # valid_mask: np.ndarray, shape (T,)
    time_len = int(angles_rad.shape[0])

    out = np.zeros((time_len,), dtype=np.float32)  # shape: (T,)
    out_valid = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    if time_len == 0:
        return out, out_valid

    valid_indices = np.flatnonzero(valid_mask.astype(bool)).astype(np.int64)  # shape: (K,)
    if valid_indices.size == 0:
        return out, out_valid

    if valid_indices.size == 1:
        t = int(valid_indices[0])
        out[t] = float(angles_rad[t])
        out_valid[t] = True
        return out, out_valid

    t_start = int(valid_indices[0])
    t_end = int(valid_indices[-1])

    valid_angles = angles_rad[valid_mask.astype(bool)].astype(np.float32)  # shape: (K,)
    valid_angles_unwrapped = np.unwrap(valid_angles).astype(np.float32)  # shape: (K,)

    t_query = np.arange(t_start, t_end + 1, dtype=np.float32)  # shape: (M,)
    t_known = valid_indices.astype(np.float32)  # shape: (K,)

    out[t_start:t_end + 1] = np.interp(t_query, t_known, valid_angles_unwrapped).astype(np.float32)
    out_valid[t_start:t_end + 1] = True

    # torch를 거치지 않고 numpy로 바로 [-pi, pi) 정리
    out = wrap_angle_np(out)
    return out, out_valid



# =========================
# 에이전트(ego/neighbor) 특징 만들기
# =========================
def make_agent_type_one_hot(object_type_minus1: int) -> np.ndarray:
    """에이전트 타입을 one-hot(3개)로 만듭니다.

    매핑 규칙:
    - 0: VEHICLE
    - 1: PEDESTRIAN
    - 2: CYCLIST
    그 외는 [0,0,0]으로 둡니다.

    Args:
        object_type_minus1: scenario Track의 object_type에서 1을 뺀 값

    Returns:
        one_hot: shape (3,), float32
    """
    # one_hot: np.ndarray, shape (3,)
    one_hot = np.zeros((3,), dtype=np.float32)
    if object_type_minus1 == 0:
        one_hot[0] = 1.0
    elif object_type_minus1 == 1:
        one_hot[1] = 1.0
    elif object_type_minus1 == 2:
        one_hot[2] = 1.0
    return one_hot


def get_num_steps_from_scenario(scenario: Any) -> int:
    """시나리오의 “전체 시간 길이(step 수)”를 안전하게 구합니다.

    어떤 시나리오는 track마다 길이가 다를 수 있습니다.
    그래서 한 트랙(track0 등)의 길이에 기대면 전체가 잘릴 수 있습니다.
    이 함수는 다음 순서로 전체 길이를 정합니다.

    1) 시나리오에 timestamps_seconds가 있으면 그 길이를 사용
    2) 없으면 모든 트랙의 states 길이 중 “가장 긴 길이”를 사용
    3) 그래도 0이면 에러

    Args:
        scenario: WOMD 시나리오 객체

    Returns:
        num_steps: 전체 step 수(정수)

    Raises:
        ValueError: step 수를 0으로밖에 구할 수 없을 때
    """
    timestamps = getattr(scenario, "timestamps_seconds", None)
    if timestamps is not None:
        num_steps = int(len(timestamps))
        if num_steps > 0:
            return num_steps

    tracks = getattr(scenario, "tracks", [])
    if len(tracks) > 0:
        lengths = [int(len(getattr(tr, "states", []))) for tr in tracks]
        num_steps = int(max(lengths)) if len(lengths) > 0 else 0
        if num_steps > 0:
            return num_steps

    raise ValueError("Scenario has zero steps (invalid WOMD record).")


def build_time_indices(
    current_time_index: int,
    num_total_steps: int,
    desired_past_len: int,
    desired_future_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """현재 시점을 기준으로 “최근 과거 / 바로 다음 미래” 인덱스를 만듭니다.

    이 함수는 과거 인덱스의 “끝이 항상 현재 시점”이 되도록 만듭니다.
    예를 들어 현재가 10이고 과거 길이가 5면, [6,7,8,9,10]처럼 “최근 구간”을 씁니다.
    데이터가 부족한 앞쪽은 -1로 채웁니다.

    Args:
        current_time_index: 현재 시점 인덱스(정수)
        num_total_steps: 전체 시점 개수(정수)
        desired_past_len: 과거 길이(TIME_LEN)
        desired_future_len: 미래 길이(FUTURE_LEN)

    Returns:
        past_indices: shape (desired_past_len,), dtype int64. 없으면 -1.
        future_indices: shape (desired_future_len,), dtype int64. 없으면 -1.
    """
    current_t = int(current_time_index)
    num_steps = int(num_total_steps)
    past_len = int(desired_past_len)
    future_len = int(desired_future_len)

    past_indices = -np.ones((past_len,), dtype=np.int64)  # shape (past_len,)
    future_indices = -np.ones(
        (future_len,), dtype=np.int64)  # shape (future_len,)

    if num_steps <= 0:
        return past_indices, future_indices

    current_t = max(0, min(current_t, num_steps - 1))

    # 과거: "끝이 current_t"가 되도록 최근 구간 선택
    start_past = max(0, current_t - past_len + 1)
    selected_past = np.arange(start_past, current_t + 1,
                              dtype=np.int64)  # shape (K,)
    past_indices[-selected_past.shape[0]:] = selected_past

    # 미래: current_t+1부터 채우고 부족분은 -1 유지
    start_future = current_t + 1
    end_future = min(num_steps, start_future + future_len)
    if start_future < num_steps:
        future_indices[:(end_future - start_future)] = np.arange(start_future,
                                                                 end_future,
                                                                 dtype=np.int64)

    return past_indices, future_indices


from typing import Tuple
import numpy as np


def gather_agent_sequence_by_indices(
    agent_idx: int,
    time_indices: np.ndarray,  # shape: (T,), -1이면 없음
    pos_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    vel_xy_local_all: np.ndarray,  # shape: (N, S, 2)
    yaw_local_all: np.ndarray,  # shape: (N, S)
    width_length_all: np.ndarray,  # shape: (N, S, 2)  (width, length)
    valid_all: np.ndarray,  # shape: (N, S)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """원하는 시간 인덱스들만 뽑아서 (x,y, yaw, v, w/l, valid)를 만듭니다. (numpy로 한 번에 처리)

    기존 구현은 T(21/80) 길이만큼 파이썬 for문을 돌면서 한 칸씩 복사했습니다.
    여기서는 time_indices에서 "실제로 존재하는 인덱스"만 골라서 numpy가 한 번에 가져오도록 바꿉니다.

    Args:
        agent_idx: 에이전트 인덱스.
        time_indices: (T,) 시간 인덱스 배열. -1은 데이터 없음.
        pos_xy_local_all: (N, S, 2) ego 기준 위치.
        vel_xy_local_all: (N, S, 2) ego 기준 속도.
        yaw_local_all: (N, S) ego 기준 yaw.
        width_length_all: (N, S, 2) (width, length).
        valid_all: (N, S) 유효 여부.

    Returns:
        pos_xy_seq: (T, 2) float32
        vel_xy_seq: (T, 2) float32
        yaw_seq: (T,) float32
        width_length_seq: (T, 2) float32
        valid_seq: (T,) bool
    """
    # time_indices: np.ndarray, shape (T,)
    time_len = int(time_indices.shape[0])
    num_steps = int(pos_xy_local_all.shape[1])  # S

    pos_xy_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    vel_xy_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    yaw_seq = np.zeros((time_len,), dtype=np.float32)  # shape: (T,)
    width_length_seq = np.zeros((time_len, 2), dtype=np.float32)  # shape: (T, 2)
    valid_seq = np.zeros((time_len,), dtype=bool)  # shape: (T,)

    # 유효한(0 이상, 범위 안) 인덱스만 사용
    src_mask = (time_indices >= 0) & (time_indices < num_steps)  # shape: (T,)
    if not bool(np.any(src_mask)):
        return pos_xy_seq, vel_xy_seq, yaw_seq, width_length_seq, valid_seq

    src_idx = time_indices[src_mask].astype(np.int64)  # shape: (K,)

    pos_xy_seq[src_mask] = pos_xy_local_all[int(agent_idx), src_idx]  # (K,2)
    vel_xy_seq[src_mask] = vel_xy_local_all[int(agent_idx), src_idx]  # (K,2)
    yaw_seq[src_mask] = yaw_local_all[int(agent_idx), src_idx].astype(np.float32)  # (K,)
    width_length_seq[src_mask] = width_length_all[int(agent_idx), src_idx]  # (K,2)
    valid_seq[src_mask] = valid_all[int(agent_idx), src_idx].astype(bool)  # (K,)

    return pos_xy_seq, vel_xy_seq, yaw_seq, width_length_seq, valid_seq



import numpy as np


def pack_agent_features_11(
    pos_xy: np.ndarray,  # shape: (T, 2)
    vel_xy: np.ndarray,  # shape: (T, 2)
    yaw_rad: np.ndarray,  # shape: (T,)
    width_length: np.ndarray,  # shape: (T, 2)
    agent_one_hot: np.ndarray,  # shape: (3,)
    valid_mask: np.ndarray,  # shape: (T,)
) -> np.ndarray:
    """요구사항의 11차원 포맷으로 묶습니다. (불필요한 복사 최소화)

    [x, y, cos(yaw), sin(yaw), v_x, v_y, width, length, one_hot(3)]

    Args:
        pos_xy: (T,2) 위치(ego 기준).
        vel_xy: (T,2) 속도(ego 기준).
        yaw_rad: (T,) yaw(ego 기준).
        width_length: (T,2) (width,length).
        agent_one_hot: (3,) 타입 one-hot.
        valid_mask: (T,) 유효 여부.

    Returns:
        feat_11: (T,11) float32
    """
    time_len = int(pos_xy.shape[0])
    feat_11 = np.zeros((time_len, 11), dtype=np.float32)  # shape: (T, 11)

    idx = valid_mask.astype(bool)  # shape: (T,)
    if not bool(np.any(idx)):
        return feat_11

    # 필요한 시점만 cos/sin 계산
    yaw_valid = yaw_rad[idx].astype(np.float32)  # shape: (K,)
    cos_yaw = np.cos(yaw_valid).astype(np.float32)  # shape: (K,)
    sin_yaw = np.sin(yaw_valid).astype(np.float32)  # shape: (K,)

    feat_11[idx, 0:2] = pos_xy[idx]
    feat_11[idx, 2] = cos_yaw
    feat_11[idx, 3] = sin_yaw
    feat_11[idx, 4:6] = vel_xy[idx]
    feat_11[idx, 6:8] = width_length[idx]

    # repeat로 큰 배열을 만들지 않고, (3,) 값을 (K,3)에 자연스럽게 채움
    feat_11[idx, 8:11] = agent_one_hot.astype(np.float32)  # shape: (K,3)

    return feat_11

from typing import List
import numpy as np


def build_agent_type_one_hot_all(
    object_type_minus1_all: np.ndarray,  # shape: (N,)
    ego_track_index: int,
) -> np.ndarray:
    """모든 트랙의 타입 one-hot(3)을 한 번에 만듭니다.

    기존 방식은 트랙 수(N)만큼 파이썬 for문을 돌며 one-hot을 채웠습니다.
    여기서는 numpy로 한 번에 채워서 불필요한 파이썬 반복을 줄입니다.

    Args:
        object_type_minus1_all: (N,) 각 트랙의 타입 값.
            - 0: VEHICLE
            - 1: PEDESTRIAN
            - 2: CYCLIST
            - 그 외 값은 "모름"으로 보고 [0,0,0] 유지
        ego_track_index: ego 트랙 인덱스(정수). ego는 항상 VEHICLE로 고정합니다.

    Returns:
        one_hot_all: (N, 3) float32.
            각 트랙마다 타입 one-hot. (ego는 [1,0,0]으로 강제)
    """
    # object_type_minus1_all: np.ndarray, shape (N,)
    num_tracks = int(object_type_minus1_all.shape[0])
    one_hot_all = np.zeros((num_tracks, 3), dtype=np.float32)  # shape: (N,3)

    valid_mask = (object_type_minus1_all >= 0) & (object_type_minus1_all <= 2)  # shape: (N,)
    if bool(np.any(valid_mask)):
        rows = np.flatnonzero(valid_mask).astype(np.int64)  # shape: (K,)
        cols = object_type_minus1_all[valid_mask].astype(np.int64)  # shape: (K,)
        one_hot_all[rows, cols] = 1.0

    if 0 <= int(ego_track_index) < num_tracks:
        one_hot_all[int(ego_track_index)] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    return one_hot_all


def collect_neighbor_indices_at_current(
    valid_all: np.ndarray,  # shape: (N, S)
    current_time_index: int,
    ego_track_index: int,
) -> np.ndarray:
    """현재 시점에서 "존재하는 neighbor" 트랙 인덱스를 모읍니다.

    규칙:
    - valid_all[:, current_time_index]가 True인 트랙만 선택합니다.
    - ego 트랙은 제외합니다.

    Args:
        valid_all: (N, S) bool. 각 트랙의 각 시점 유효 여부.
        current_time_index: 현재 시점 인덱스(정수).
        ego_track_index: ego 트랙 인덱스(정수).

    Returns:
        neighbor_indices: (A,) int64.
            현재 시점에 존재하는 neighbor 트랙 인덱스들.
    """
    # valid_all: np.ndarray, shape (N, S)
    current_t = int(current_time_index)
    is_valid_now = valid_all[:, current_t].astype(bool)  # shape: (N,)

    indices = np.flatnonzero(is_valid_now).astype(np.int64)  # shape: (A_with_ego,)
    indices = indices[indices != int(ego_track_index)]  # shape: (A,)
    return indices


def pack_agent_future_3(
        pos_xy: np.ndarray,  # shape: (T,2)
        yaw_rad: np.ndarray,  # shape: (T,)
        valid_mask: np.ndarray,  # shape: (T,)
) -> np.ndarray:
    """요구사항의 미래 GT 3차원 포맷으로 묶습니다.

    [x, y, heading]

    Args:
        pos_xy: (T,2) 위치(ego 기준)
        yaw_rad: (T,) heading(ego 기준)
        valid_mask: (T,) 유효 여부

    Returns:
        feat_3: (T,3) float32
    """
    time_len = int(pos_xy.shape[0])
    feat_3 = np.zeros((time_len, 3), dtype=np.float32)  # shape (T,3)
    if not np.any(valid_mask):
        return feat_3

    idx = valid_mask
    feat_3[idx, 0:2] = pos_xy[idx]
    feat_3[idx, 2] = yaw_rad[idx]
    return feat_3


def build_interpolated_agent_traj(
    pos_xy_raw: np.ndarray,  # shape: (T,2)
    vel_xy_raw: np.ndarray,  # shape: (T,2)
    yaw_raw: np.ndarray,  # shape: (T,)
    width_length_raw: np.ndarray,  # shape: (T,2)
    valid_raw: np.ndarray,  # shape: (T,)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(x,y, v, yaw, width/length) 시계열을 보간 규칙에 맞게 정리합니다.

    Args:
        pos_xy_raw: (T,2) 원본 위치 (없는 곳은 0)
        vel_xy_raw: (T,2) 원본 속도 (없는 곳은 0)
        yaw_raw: (T,) 원본 yaw (없는 곳은 0)
        width_length_raw: (T,2) 원본 (width,length)
        valid_raw: (T,) 유효 여부

    Returns:
        pos_xy: (T,2) 보간 후 위치
        vel_xy: (T,2) 보간 후 속도
        yaw: (T,) 보간 후 yaw
        width_length: (T,2) 보간 후 (width,length)
        valid_mask: (T,) 보간으로 채워진 구간 True
    """
    # pos_xy: (T,2)
    pos_xy, valid_filled = interpolate_linear_sequence(pos_xy_raw, valid_raw)
    vel_xy, _ = interpolate_linear_sequence(vel_xy_raw, valid_raw)
    width_length, _ = interpolate_linear_sequence(width_length_raw, valid_raw)

    # yaw: (T,)
    yaw, valid_filled_yaw = interpolate_linear_angle(yaw_raw, valid_raw)

    # 위치/각도 보간 구간이 다르면 이상하므로, 둘 다 True인 곳만 최종 valid로 사용
    valid_mask = np.logical_and(valid_filled, valid_filled_yaw)  # shape (T,)
    return pos_xy, vel_xy, yaw, width_length, valid_mask


# =========================
# 트랙/역할 디코딩
# =========================


def _track_states_to_numpy(
    track: Any,
    num_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Track 1개의 states를 numpy로 빠르게 변환합니다.

    Args:
        track: scenario.tracks의 원소 (proto Track)
        num_steps: 시나리오의 전체 step 수 (보통 91)

    Returns:
        states_9: (S, 9) float32
            [center_x, center_y, center_z, length, width, height, heading, v_x, v_y]
        valid: (S,) bool
    """
    track_states = track.states
    n = len(track_states)

    # (S,9)/(S,) 기본값은 0/False로 패딩
    states_9 = np.zeros((num_steps, 9), dtype=np.float32)
    valid = np.zeros((num_steps,), dtype=bool)
    if n == 0:
        return states_9, valid

    # 한 번의 리스트 컴프리헨션으로 (n, 10) = 9개 상태 + valid(0/1)을 같이 만들기
    # -> 내부 for문에서 states[i,t,k]를 9번 찍는 것보다 보통 더 빠릅니다.
    packed_10 = np.asarray(
        [
            (
                float(st.center_x),
                float(st.center_y),
                float(st.center_z),
                float(st.length),
                float(st.width),
                float(st.height),
                float(st.heading),
                float(st.velocity_x),
                float(st.velocity_y),
                float(st.valid),  # bool -> 0.0/1.0
            ) for st in track_states
        ],
        dtype=np.float32,
    )  # shape: (n, 10)

    # 혹시 track마다 states 길이가 다를 수 있어 안전하게 min으로
    m = min(n, num_steps)
    states_9[:m] = packed_10[:m, :9]
    valid[:m] = packed_10[:m, 9] != 0.0
    return states_9, valid


from typing import Any, Dict, Set
import numpy as np


def _collect_track_indices_to_predict(scenario: Any,
                                      num_tracks: int) -> Set[int]:
    """tracks_to_predict에서 'tracks 인덱스'들을 안전하게 모읍니다.

    Args:
        scenario: WOMD scenario 객체.
        num_tracks: scenario.tracks의 길이.

    Returns:
        track_indices: tracks 인덱스들의 집합.
    """
    track_indices: Set[int] = set()
    for rp in getattr(scenario, "tracks_to_predict", []):
        idx = int(getattr(rp, "track_index"))
        if 0 <= idx < num_tracks:
            track_indices.add(idx)
    return track_indices


def _collect_track_indices_of_interest(scenario: Any,
                                       num_tracks: int) -> Set[int]:
    """objects_of_interest에서 'tracks 인덱스'들을 안전하게 모읍니다.

    Waymo Motion 문서에 따르면 objects_of_interest는 tracks 필드에 대한 인덱스입니다. :contentReference[oaicite:1]{index=1}

    Args:
        scenario: WOMD scenario 객체.
        num_tracks: scenario.tracks의 길이.

    Returns:
        interest_indices: tracks 인덱스들의 집합.
    """
    interest_indices: Set[int] = set()
    for x in getattr(scenario, "objects_of_interest", []):
        idx = int(x)
        if 0 <= idx < num_tracks:
            interest_indices.add(idx)
    return interest_indices


def decode_tracks_and_roles_from_scenario(
        scenario: Any) -> Dict[str, np.ndarray]:
    """Scenario에서 트랙 정보와 role(interest/predict)을 numpy로 만듭니다.

    - tracks_to_predict: tracks 안의 인덱스 :contentReference[oaicite:2]{index=2}
    - objects_of_interest: tracks 안의 인덱스 :contentReference[oaicite:3]{index=3}

    Returns:
        object_id: (N,) int64
        object_type: (N,) int32
        states: (N,S,9) float32
        valid: (N,S) bool
        role_interest: (N,) bool
        role_predict: (N,) bool
        ego_index: (1,) int64
    """
    tracks = scenario.tracks
    num_tracks = int(len(tracks))
    num_steps = get_num_steps_from_scenario(scenario)

    object_id = np.zeros((num_tracks,), dtype=np.int64)  # (N,)
    object_type = np.zeros((num_tracks,), dtype=np.int32)  # (N,)
    states = np.zeros((num_tracks, num_steps, 9), dtype=np.float32)  # (N,S,9)
    valid = np.zeros((num_tracks, num_steps), dtype=bool)  # (N,S)

    predict_track_indices = _collect_track_indices_to_predict(
        scenario, num_tracks)
    interest_track_indices = _collect_track_indices_of_interest(
        scenario, num_tracks)

    role_interest = np.zeros((num_tracks,), dtype=bool)  # (N,)
    role_predict = np.zeros((num_tracks,), dtype=bool)  # (N,)

    for i, tr in enumerate(tracks):
        object_id[i] = int(tr.id)
        object_type[i] = int(tr.object_type) - 1

        role_predict[i] = (i in predict_track_indices)
        role_interest[i] = (i in interest_track_indices)

        states_i, valid_i = _track_states_to_numpy(tr, num_steps)
        states[i] = states_i
        valid[i] = valid_i

    return {
        "object_id": object_id,
        "object_type": object_type,
        "states": states,
        "valid": valid,
        "role_interest": role_interest,
        "role_predict": role_predict,
        "ego_index": np.array([int(scenario.sdc_track_index)], dtype=np.int64),
    }


def require_valid_ego_at_current(valid_all: np.ndarray, ego_idx: int,
                                 current_t: int) -> None:
    """현재 시점에서 ego가 유효한지 확인합니다.

    ego가 현재 프레임에서 유효하지 않으면, 좌표계 기준(ego 위치/방향)이 잘못 잡혀서
    이후에 만들어지는 모든 값이 “조용히” 틀어질 수 있습니다.
    그래서 이런 경우는 바로 에러로 중단시키는 게 안전합니다.

    Args:
        valid_all: shape (N,S), bool. 각 트랙의 각 시점 유효 여부
        ego_idx: ego 트랙 인덱스
        current_t: 현재 시점 인덱스

    Raises:
        ValueError: ego가 현재 시점에서 유효하지 않거나, 인덱스가 범위를 벗어날 때
    """
    # valid_all: np.ndarray, shape (N,S)
    if valid_all.ndim != 2:
        raise ValueError(
            f"valid_all must be 2D (N,S). got shape={valid_all.shape}")

    n, s = int(valid_all.shape[0]), int(valid_all.shape[1])
    if not (0 <= ego_idx < n) or not (0 <= current_t < s):
        raise ValueError(
            f"Index out of range: ego_idx={ego_idx}/{n}, t={current_t}/{s}")

    if not bool(valid_all[ego_idx, current_t]):
        raise ValueError(
            f"Ego invalid at current frame: ego_idx={ego_idx}, t={current_t}")


def compute_ego_pose_at_current(
    scenario: Any,
    track_dict: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, float, float]:
    """현재 시점의 ego 위치/방향을 구합니다.

    반환되는 ego 위치/방향은 이후 모든 좌표 변환의 기준이 됩니다.
    그래서 ego가 현재 시점에서 유효하지 않으면 즉시 에러로 중단합니다.

    Args:
        scenario: WOMD 시나리오 객체
        track_dict: decode_tracks_and_roles_from_scenario의 결과

    Returns:
        ego_xy_global: shape (2,), float32
        ego_yaw_global: float
        ego_z_global: float
    """
    ego_idx = int(track_dict["ego_index"][0])
    current_t = int(scenario.current_time_index)

    valid_all = track_dict["valid"]  # shape (N,S)
    require_valid_ego_at_current(valid_all, ego_idx, current_t)

    ego_state = track_dict["states"][ego_idx, current_t]  # shape (9,)
    ego_xy = ego_state[0:2].astype(np.float32)  # shape (2,)
    ego_yaw = float(ego_state[6])
    ego_z = float(ego_state[2])
    return ego_xy, ego_yaw, ego_z


def is_valid_polygon_xy(points_xy: np.ndarray, min_points: int = 3) -> bool:
    """polygon 점 목록이 “의미 있게” 존재하는지 검사합니다.

    점이 너무 적으면(예: 0개, 1개, 2개) polygon이라고 보기 어렵고,
    이런 값이 캐시에 들어가면 이후 단계에서 전부 0으로 된 결과가 생겨서
    데이터가 더러워질 수 있습니다.

    Args:
        points_xy: shape (M,2), float32. polygon의 점 목록
        min_points: 최소 점 개수(기본 3)

    Returns:
        True면 저장할 가치가 있는 polygon, False면 버립니다.
    """
    # points_xy: np.ndarray, shape (M,2)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return False
    return int(points_xy.shape[0]) >= int(min_points)


# =========================
# 교통 신호(차선 신호) 파싱
# =========================
def lane_signal_state_to_one_hot(state_value: int) -> np.ndarray:
    """WOMD의 lane signal state 값을 4차원 one-hot으로 바꿉니다.

    출력 one-hot 순서:
        [GO, CAUTION, STOP, UNKNOWN]

    Args:
        state_value: TrafficSignalLaneState.State (정수)

    Returns:
        one_hot: (4,) float32
    """
    # 참고: state_value는 map_pb2.TrafficSignalLaneState.State 값이 들어옵니다.
    # 여기서는 '큰 분류'만 사용합니다.
    go_states = {3, 6}  # ARROW_GO(3), GO(6)
    caution_states = {2, 5,
                      8}  # ARROW_CAUTION(2), CAUTION(5), FLASHING_CAUTION(8)
    stop_states = {1, 4, 7}  # ARROW_STOP(1), STOP(4), FLASHING_STOP(7)
    unknown_states = {0}  # UNKNOWN(0)

    one_hot = np.zeros((4,), dtype=np.float32)
    if state_value in go_states:
        one_hot[0] = 1.0
    elif state_value in caution_states:
        one_hot[1] = 1.0
    elif state_value in stop_states:
        one_hot[2] = 1.0
    elif state_value in unknown_states:
        one_hot[3] = 1.0
    else:
        # 혹시 모르는 값은 UNKNOWN 처리
        one_hot[3] = 1.0
    return one_hot


def extract_lane_signals_at_current(
    scenario: scenario_pb2.Scenario,) -> Dict[int, np.ndarray]:
    """현재 시점의 차선 신호를 lane_id -> one-hot(4)로 뽑습니다.

    Args:
        scenario: Scenario proto

    Returns:
        lane_id_to_light: lane_id(int) -> one_hot(4,) float32
    """
    lane_id_to_light: Dict[int, np.ndarray] = {}
    current_t = int(scenario.current_time_index)

    if len(scenario.dynamic_map_states) <= current_t:
        return lane_id_to_light

    dyn = scenario.dynamic_map_states[current_t]
    for lane_state in dyn.lane_states:
        lane_id = int(lane_state.lane)
        state_value = int(lane_state.state)
        lane_id_to_light[lane_id] = lane_signal_state_to_one_hot(state_value)

    return lane_id_to_light

def _min_dist2_point_to_polyline_xy(
    points_xy: np.ndarray,  # shape: (K, 2)
    query_xy: np.ndarray,  # shape: (2,)
    closed: bool,
) -> float:
    """한 점(query)이 '점들이 줄로 이어진 형태'까지의 최소 거리^2를 계산합니다.

    - points_xy가 (K,2)로 주어졌을 때, 인접한 점들을 이은 선분들(segments)을 만들고,
      query_xy에서 그 선분들까지의 거리를 계산해 가장 작은 값을 반환합니다.
    - closed=True이면 마지막 점과 첫 점도 이어서(닫힌 모양) 선분을 하나 더 만든 것으로 봅니다.
    - sqrt를 하지 않고 거리의 제곱(dist^2)으로 반환해서 빠르게 비교할 수 있게 합니다.

    Args:
        points_xy: (K,2) float32/float64. 선을 구성하는 점들.
        query_xy: (2,) float32/float64. 기준 점(여기서는 ego 위치).
        closed: 닫힌 모양 여부.

    Returns:
        min_dist2: float. 최소 거리의 제곱.
            points_xy가 비어 있으면 inf.
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        return float("inf")

    k = int(points_xy.shape[0])
    if k == 0:
        return float("inf")

    q = query_xy.astype(np.float32).reshape(2,)
    p = points_xy.astype(np.float32)

    if k == 1:
        d = p[0] - q
        return float(d[0] * d[0] + d[1] * d[1])

    if bool(closed):
        p0 = p
        p1 = np.roll(p, shift=-1, axis=0)
    else:
        p0 = p[:-1]
        p1 = p[1:]

    seg = (p1 - p0).astype(np.float32)  # (M,2)
    v = (q[None, :] - p0).astype(np.float32)  # (M,2)

    seg_len2 = (seg[:, 0] * seg[:, 0] + seg[:, 1] * seg[:, 1]).astype(np.float32)  # (M,)
    seg_len2_safe = np.maximum(seg_len2, float(EPS)).astype(np.float32)

    t = ((v[:, 0] * seg[:, 0] + v[:, 1] * seg[:, 1]) / seg_len2_safe).astype(np.float32)  # (M,)
    t = np.clip(t, 0.0, 1.0).astype(np.float32)

    closest = (p0 + seg * t[:, None]).astype(np.float32)  # (M,2)
    d = (q[None, :] - closest).astype(np.float32)  # (M,2)
    dist2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).astype(np.float32)  # (M,)

    return float(np.min(dist2))


def _is_point_inside_polygon_xy(
    point_xy: np.ndarray,  # shape: (2,)
    polygon_xy: np.ndarray,  # shape: (M, 2)
) -> bool:
    """점(point)이 다각형(polygon) '안쪽'에 있는지 빠르게 검사합니다.

    - crosswalk/driveway/speed_bump 같은 것은 '영역'으로 보는 것이 자연스럽습니다.
      그래서 ego가 polygon 내부에 있으면, 거리는 0이라고 보고 무조건 포함시키는 게 맞습니다.
    - 이 함수는 다각형의 변을 기준으로, 가로선이 몇 번 교차하는지 세는 방식으로 판정합니다.

    Args:
        point_xy: (2,) float32/float64. 검사할 점(여기서는 ego 위치).
        polygon_xy: (M,2) float32/float64. 다각형 꼭짓점들.

    Returns:
        inside: bool. 내부면 True.
    """
    if polygon_xy.ndim != 2 or polygon_xy.shape[1] != 2:
        return False

    m = int(polygon_xy.shape[0])
    if m < 3:
        return False

    p = polygon_xy.astype(np.float32)
    px = float(point_xy[0])
    py = float(point_xy[1])

    x0 = p[:, 0]
    y0 = p[:, 1]
    x1 = np.roll(x0, shift=-1)
    y1 = np.roll(y0, shift=-1)

    cond1 = (y0 > py) != (y1 > py)
    # y1-y0가 0에 가까운 경우 분모가 0이 될 수 있어 EPS를 더합니다.
    x_intersect = (x1 - x0) * (py - y0) / (y1 - y0 + float(EPS)) + x0
    cond2 = px < x_intersect

    crossings = int(np.count_nonzero(cond1 & cond2))
    return (crossings % 2) == 1


def _keep_polyline_if_within_radius(
    polyline_xy_global: np.ndarray,  # shape: (K,2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    radius_m: float,
) -> bool:
    """polyline(점들이 줄로 이어진 것)이 ego 반경 안에 '조금이라도' 들어오면 True를 반환합니다.

    판정 기준:
      - ego 위치에서 polyline 선분들까지의 최소 거리가 radius_m 이하이면 포함합니다.
      - 즉, 꼭짓점이 반경 밖에 있어도 선분이 원을 스치면 포함됩니다.

    Args:
        polyline_xy_global: (K,2) 전역 좌표 점들.
        ego_xy_global: (2,) ego 전역 위치.
        radius_m: 반경(m).

    Returns:
        keep: bool
    """
    r = float(radius_m)
    if not np.isfinite(r):
        return True
    if r < 0.0:
        return True

    r2 = r * r
    d2 = _min_dist2_point_to_polyline_xy(
        points_xy=polyline_xy_global,
        query_xy=ego_xy_global,
        closed=False,
    )
    return bool(d2 <= r2)


def _keep_polygon_area_if_within_radius(
    polygon_xy_global: np.ndarray,  # shape: (M,2)
    ego_xy_global: np.ndarray,  # shape: (2,)
    radius_m: float,
) -> bool:
    """polygon(영역)이 ego 반경 안에 '조금이라도' 들어오면 True를 반환합니다.

    판정 기준:
      1) ego가 polygon 내부에 있으면 거리=0으로 보고 무조건 포함
      2) 아니면 ego에서 polygon 테두리(선분)까지의 최소 거리가 radius_m 이하이면 포함

    Args:
        polygon_xy_global: (M,2) 전역 좌표 꼭짓점들.
        ego_xy_global: (2,) ego 전역 위치.
        radius_m: 반경(m).

    Returns:
        keep: bool
    """
    r = float(radius_m)
    if not np.isfinite(r):
        return True
    if r < 0.0:
        return True

    if _is_point_inside_polygon_xy(ego_xy_global, polygon_xy_global):
        return True

    r2 = r * r
    d2 = _min_dist2_point_to_polyline_xy(
        points_xy=polygon_xy_global,
        query_xy=ego_xy_global,
        closed=True,
    )
    return bool(d2 <= r2)


def filter_parsed_map_by_radius(
    parsed_map: ParsedMap,
    ego_xy_global: np.ndarray,  # shape: (2,)
    filter_radius_m: float,
) -> ParsedMap:
    """ParsedMap에서 ego 반경 안에 들어오는 것만 남겨서 새 ParsedMap을 만듭니다.

    이 함수는 캐싱 대상 중 아래 항목들만 필터링합니다.
      - lanes (→ lanes_speed_limit, lanes_has_speed_limit, lane_light, lane_type,
               left_line_type, right_line_type는 lanes에 종속이라 자동으로 같이 줄어듭니다)
      - road_edge (→ road_edge_type도 같이 줄어듭니다)
      - driveway
      - stop_sign_points(=stop_sign_xy_global 기반)
      - crosswalk_points
      - speed_bump_points

    “일부라도 영역 내에 들어오면 포함” 조건을 만족시키기 위해,
      - polyline: 선분까지의 최소거리로 판정
      - polygon: (내부 포함) + (테두리 선분까지 최소거리)로 판정
    을 사용합니다.

    Args:
        parsed_map: parse_map_from_scenario() 결과.
        ego_xy_global: (2,) ego 전역 위치.
        filter_radius_m: 반경(m). inf면 필터 없이 그대로 반환합니다.

    Returns:
        filtered_map: 필터가 적용된 ParsedMap
    """
    r = float(filter_radius_m)
    if not np.isfinite(r):
        return parsed_map
    if r < 0.0:
        return parsed_map

    # 1) stop sign (점)
    if len(parsed_map.stop_sign_xy_global) == 0:
        stop_sign_xy = []
    else:
        pts = np.stack(parsed_map.stop_sign_xy_global, axis=0).astype(np.float32)  # (N,2)
        d = pts - ego_xy_global.astype(np.float32)[None, :]  # (N,2)
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]).astype(np.float32)  # (N,)
        keep = d2 <= (r * r)
        idxs = np.flatnonzero(keep).astype(np.int64)
        stop_sign_xy = [parsed_map.stop_sign_xy_global[int(i)] for i in idxs.tolist()]

    # 2) polygons (영역)
    crosswalk_polys: List[np.ndarray] = []
    for poly in parsed_map.crosswalk_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            crosswalk_polys.append(poly)

    speed_bump_polys: List[np.ndarray] = []
    for poly in parsed_map.speed_bump_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            speed_bump_polys.append(poly)

    driveway_polys: List[np.ndarray] = []
    for poly in parsed_map.driveway_polygons_xy_global:
        if _keep_polygon_area_if_within_radius(poly, ego_xy_global, r):
            driveway_polys.append(poly)

    # 3) lanes (centerline polyline 기준)
    lanes_filtered: List[LaneInfo] = []
    for lane in parsed_map.lanes:
        if _keep_polyline_if_within_radius(lane.centerline_xy_global, ego_xy_global, r):
            lanes_filtered.append(lane)

    # 4) road_edge ids (polyline 기준)
    road_edge_ids_filtered: List[int] = []
    for edge_id in parsed_map.road_edge_ids:
        poly = parsed_map.boundary_polylines_xy_global.get(int(edge_id), None)
        if poly is None:
            continue
        if _keep_polyline_if_within_radius(poly, ego_xy_global, r):
            road_edge_ids_filtered.append(int(edge_id))

    # dict류(경계 polyline/type 정보)는 그대로 참조해도 됨 (캐싱 대상이 아니고, lanes 계산에 필요)
    return ParsedMap(
        stop_sign_xy_global=stop_sign_xy,
        crosswalk_polygons_xy_global=crosswalk_polys,
        speed_bump_polygons_xy_global=speed_bump_polys,
        driveway_polygons_xy_global=driveway_polys,
        lanes=lanes_filtered,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        road_line_type_by_id=parsed_map.road_line_type_by_id,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
        road_edge_ids=road_edge_ids_filtered,
    )


# =========================
# 맵 파싱 + 샘플링
# =========================
def polyline_length(points_xy: np.ndarray) -> float:
    """폴리라인(점들의 줄)의 전체 길이를 계산합니다.

    Args:
        points_xy: (N,2) 점들

    Returns:
        길이(float)
    """
    if points_xy.shape[0] < 2:
        return 0.0
    diffs = points_xy[1:] - points_xy[:-1]  # shape (N-1,2)
    seg_lens = np.linalg.norm(diffs, axis=1)  # shape (N-1,)
    return float(seg_lens.sum())


def resample_polyline_equal_distance(
    points_xy: np.ndarray,  # shape: (N,2)
    num_samples: int,
    closed: bool,
) -> np.ndarray:
    """점들의 줄(또는 닫힌 다각형)을 '같은 간격'으로 num_samples개 점으로 만듭니다.

    - points_xy가 1개뿐이면 그 점을 반복합니다.
    - 길이가 0에 가깝다면 첫 점을 반복합니다.
    - closed=True이면 마지막에서 첫 점으로 이어진다고 보고 둘레를 따라 샘플링합니다.

    Args:
        points_xy: (N,2) 입력 점들
        num_samples: 뽑을 점 개수
        closed: 닫힌 모양인지 여부

    Returns:
        sampled: (num_samples,2) float32
    """
    if points_xy.shape[0] == 0:
        return np.zeros((num_samples, 2), dtype=np.float32)

    if points_xy.shape[0] == 1:
        return np.repeat(points_xy.astype(np.float32),
                         repeats=num_samples,
                         axis=0)

    if closed:
        # 닫힌 모양이면 끝에 첫 점을 붙여서 둘레를 만듭니다.
        if not np.allclose(points_xy[0], points_xy[-1]):
            points_xy = np.vstack([points_xy, points_xy[0]])  # shape (N+1,2)

    diffs = points_xy[1:] - points_xy[:-1]  # shape (M,2)
    seg_lens = np.linalg.norm(diffs, axis=1)  # shape (M,)
    total_len = float(seg_lens.sum())

    if total_len < EPS:
        return np.repeat(points_xy[:1].astype(np.float32),
                         repeats=num_samples,
                         axis=0)

    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])  # shape (M+1,)

    if closed:
        # 닫힌 모양은 시작점 중복을 피하려고 endpoint=False
        target = np.linspace(0.0,
                             total_len,
                             num_samples,
                             endpoint=False,
                             dtype=np.float32)
    else:
        target = np.linspace(0.0,
                             total_len,
                             num_samples,
                             endpoint=True,
                             dtype=np.float32)

    # 각 target이 어느 구간에 속하는지 찾기
    seg_idx = np.searchsorted(cum_len, target, side="right") - 1
    seg_idx = np.clip(seg_idx, 0, len(seg_lens) - 1)  # shape (num_samples,)

    seg_start = points_xy[seg_idx]  # shape (num_samples,2)
    seg_vec = diffs[seg_idx]  # shape (num_samples,2)
    seg_len = seg_lens[seg_idx]  # shape (num_samples,)

    alpha = (target - cum_len[seg_idx]) / np.maximum(
        seg_len, EPS)  # shape (num_samples,)
    sampled = seg_start + seg_vec * alpha[:, None]  # shape (num_samples,2)
    return sampled.astype(np.float32)


from typing import Any, Iterable, List


def convert_proto_boundary_segments(
    boundary_segments_proto: Iterable[Any],) -> List[BoundarySegmentInfo]:
    """Waymo lane boundary segment(proto) 목록을 BoundarySegmentInfo 리스트로 변환합니다.

    Waymo 버전/환경에 따라 필드명이 lane_end_index 또는 lane_end_idx처럼
    약간 다를 수 있어서, getattr로 안전하게 읽습니다.

    Args:
        boundary_segments_proto: 반복 가능한 boundary segment proto 목록.
            각 원소는 보통 다음 필드를 가집니다.
            - lane_start_index (또는 lane_start_idx)
            - lane_end_index (또는 lane_end_idx)
            - boundary_feature_id

    Returns:
        segments: BoundarySegmentInfo 리스트.
            boundary_feature_id가 0 이하이거나, end <= start 같은 값도
            일단 담아두고, 실제 사용 단계에서 자동으로 무시되도록 합니다.
    """
    segments: List[BoundarySegmentInfo] = []
    for seg in boundary_segments_proto:
        lane_start_index = int(
            getattr(seg, "lane_start_index", getattr(seg, "lane_start_idx", 0)))
        lane_end_index = int(
            getattr(seg, "lane_end_index", getattr(seg, "lane_end_idx", 0)))
        boundary_feature_id = int(getattr(seg, "boundary_feature_id", 0))

        segments.append(
            BoundarySegmentInfo(
                lane_start_index=lane_start_index,
                lane_end_index=lane_end_index,
                boundary_feature_id=boundary_feature_id,
            ))
    return segments


def parse_map_from_scenario(scenario: scenario_pb2.Scenario) -> ParsedMap:
    """Scenario의 map_features를 요구사항에 맞게 필요한 것만 모읍니다.

    Returns:
        ParsedMap
    """
    stop_sign_xy_global: List[np.ndarray] = []
    crosswalk_polygons_xy_global: List[np.ndarray] = []
    speed_bump_polygons_xy_global: List[np.ndarray] = []
    driveway_polygons_xy_global: List[np.ndarray] = []

    lanes: List[LaneInfo] = []

    boundary_polylines_xy_global: Dict[int, np.ndarray] = {}
    boundary_id_to_kind: Dict[int, str] = {}

    road_line_type_by_id: Dict[int, int] = {}
    road_edge_type_by_id: Dict[int, int] = {}
    road_edge_ids: List[int] = []

    for mf in scenario.map_features:
        feature_type = mf.WhichOneof("feature_data")

        if feature_type == "stop_sign":
            pos = mf.stop_sign.position
            stop_sign_xy_global.append(
                np.array([pos.x, pos.y], dtype=np.float32))

        elif feature_type == "crosswalk":
            polygon_xy = _proto_points_to_xy_array(
                mf.crosswalk.polygon)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=3):
                crosswalk_polygons_xy_global.append(polygon_xy)

        elif feature_type == "speed_bump":
            polygon_xy = _proto_points_to_xy_array(
                mf.speed_bump.polygon)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=3):
                speed_bump_polygons_xy_global.append(polygon_xy)

        elif feature_type == "driveway":
            polygon_xy = _extract_driveway_polygon_xy_global(
                mf.driveway)  # (M,2)
            if is_valid_polygon_xy(polygon_xy, min_points=2):
                driveway_polygons_xy_global.append(polygon_xy)

        elif feature_type == "road_line":
            poly_xy = _proto_points_to_xy_array(mf.road_line.polyline)  # (K,2)
            if poly_xy.shape[0] >= 1:
                fid = int(mf.id)
                boundary_polylines_xy_global[fid] = poly_xy
                boundary_id_to_kind[fid] = "road_line"
                road_line_type_by_id[fid] = int(getattr(mf.road_line, "type",
                                                        0))

        elif feature_type == "road_edge":
            poly_xy = _proto_points_to_xy_array(mf.road_edge.polyline)  # (K,2)
            if poly_xy.shape[0] >= 1:
                fid = int(mf.id)
                boundary_polylines_xy_global[fid] = poly_xy
                boundary_id_to_kind[fid] = "road_edge"
                road_edge_ids.append(fid)
                road_edge_type_by_id[fid] = int(getattr(mf.road_edge, "type",
                                                        0))

        elif feature_type == "lane":
            centerline_xy = _proto_points_to_xy_array(mf.lane.polyline)  # (P,2)
            if centerline_xy.shape[0] == 0:
                continue

            # ✅ 변경: boundary_feature_id만 뽑지 않고, start/end index까지 같이 저장
            left_segments = convert_proto_boundary_segments(
                mf.lane.left_boundaries)
            right_segments = convert_proto_boundary_segments(
                mf.lane.right_boundaries)

            speed_limit_mph = float(getattr(mf.lane, "speed_limit_mph", 0.0))
            lane_type_value = int(getattr(mf.lane, "type", 0))

            lanes.append(
                LaneInfo(
                    lane_id=int(mf.id),
                    centerline_xy_global=centerline_xy,
                    left_boundary_segments=left_segments,
                    right_boundary_segments=right_segments,
                    speed_limit_mph=speed_limit_mph,
                    lane_type=lane_type_value,
                ))

    return ParsedMap(
        stop_sign_xy_global=stop_sign_xy_global,
        crosswalk_polygons_xy_global=crosswalk_polygons_xy_global,
        speed_bump_polygons_xy_global=speed_bump_polygons_xy_global,
        driveway_polygons_xy_global=driveway_polygons_xy_global,
        lanes=lanes,
        boundary_polylines_xy_global=boundary_polylines_xy_global,
        boundary_id_to_kind=boundary_id_to_kind,
        road_line_type_by_id=road_line_type_by_id,
        road_edge_type_by_id=road_edge_type_by_id,
        road_edge_ids=road_edge_ids,
    )


def atomic_pickle_dump(obj: Any, out_path: Path) -> None:
    """pickle 파일을 “깨지지 않게” 저장합니다.

    저장 중에 프로세스가 죽으면, 파일이 중간까지만 써진 상태로 남을 수 있습니다.
    그러면 다음 실행에서 그 파일을 읽다가 오류가 나거나, 더 나쁘게는 조용히 잘못 읽힐 수 있습니다.

    이 함수는
    1) 임시 파일에 먼저 끝까지 저장하고
    2) 디스크에 실제로 기록되도록 강제로 한 번 밀어넣은 다음
    3) 마지막에 이름만 바꿔서(out_path로 교체) “완성된 파일만 보이게” 합니다.

    Args:
        obj: 저장할 파이썬 객체
        out_path: 최종 pkl 경로
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with open(tmp_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists() and (not out_path.exists()):
            with contextlib.suppress(Exception):
                tmp_path.unlink()


from typing import Any, Dict, Iterable, Tuple
from pathlib import Path
import contextlib
import os
import numpy as np

EXCLUDE_KEYS_FOR_NPZ: Tuple[str, ...] = ("scenario_id", "neighbor_track_token")


def build_npz_payload_from_cache_dict(
    cache_dict: Dict[str, Any],
    excluded_keys: Iterable[str] = EXCLUDE_KEYS_FOR_NPZ,
) -> Dict[str, np.ndarray]:
    """캐시 dict에서 npz로 저장할 '배열들'만 골라 dict로 만듭니다.

    npz 파일은 "이름(key) + numpy 배열(value)" 여러 개를 한 파일에 담는 방식입니다.
    그래서 문자열(예: scenario_id)이나 문자열 리스트(예: neighbor_track_token)는
    그대로 저장하기가 애매하고, 이번 요구사항에서도 저장 대상에서 제외해야 합니다.

    이 함수는 다음 순서로 처리합니다.
    1) excluded_keys에 들어있는 키는 통째로 제외합니다.
       - 예: "scenario_id", "neighbor_track_token"
    2) 나머지 값은 numpy 배열인지 확인합니다.
       - 이미 np.ndarray면 그대로 사용합니다.
       - np.ndarray가 아니면 np.asarray로 바꿀 수 있으면 바꿉니다.
    3) 그래도 "저장하기 애매한 값"(dtype=object 같은)이면 에러로 막습니다.
       - 저장 후 읽을 때 예기치 않은 문제가 생기는 것을 미리 막기 위함입니다.

    Args:
        cache_dict: 시나리오 1개에서 만든 캐시 dict.
            - 대부분 value는 np.ndarray이고,
              예를 들면 (21,11), (80,3), (A,21,11), (L,10,12) 같은 shape을 가집니다.
        excluded_keys: npz 파일에 넣지 않을 키 목록.

    Returns:
        npz_payload: np.savez_compressed(**npz_payload)로 바로 저장 가능한 dict.
            - key: str
            - value: np.ndarray (shape은 cache_dict의 원본과 동일)
    """
    excluded = set(str(k) for k in excluded_keys)
    npz_payload: Dict[str, np.ndarray]
    npz_payload = {}

    for key, value in cache_dict.items():
        if str(key) in excluded:
            continue

        if isinstance(value, np.ndarray):
            arr = value
        else:
            arr = np.asarray(value)

        # npz에는 "진짜 숫자/불리언 배열" 형태로 담기는 게 안전합니다.
        if arr.dtype == object:
            raise TypeError(
                f"npz로 저장하기 어려운 값이 있습니다. key={key}, dtype=object"
            )

        npz_payload[str(key)] = arr

    return npz_payload


def atomic_npz_save(
    arrays: Dict[str, np.ndarray],
    out_path: Path,
    compress: bool = True,
) -> None:
    """npz 파일을 저장 도중 끊겨도 '깨진 파일'이 남지 않게 저장합니다.

    저장 중에 프로세스가 갑자기 종료되면 파일이 중간까지만 써질 수 있습니다.
    그러면 다음 실행에서 읽다가 실패하거나, 더 나쁘게는 이상한 값이 섞일 수 있습니다.

    이 함수는 아래 방식으로 안전하게 저장합니다.
    1) 같은 폴더에 임시 파일(.tmp)에 먼저 끝까지 저장
    2) 디스크에 실제로 기록되도록 강제로 한 번 반영(fsync)
    3) 마지막에 파일 이름만 바꿔서(out_path로 교체) "완성본만 보이게" 처리

    Args:
        arrays: npz에 담을 배열 dict.
            - 각 value는 np.ndarray (shape은 원본과 동일)
        out_path: 최종 .npz 경로
        compress: True면 용량을 줄여 저장합니다(대신 저장 시간이 약간 늘 수 있음).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        with open(tmp_path, "wb") as f:
            if compress:
                np.savez_compressed(f, **arrays)
            else:
                np.savez(f, **arrays)

            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, out_path)

    finally:
        if tmp_path.exists() and (not out_path.exists()):
            with contextlib.suppress(Exception):
                tmp_path.unlink()


def save_cache_dict_to_npz(
    cache_dict: Dict[str, Any],
    out_npz_path: Path,
    excluded_keys: Iterable[str] = EXCLUDE_KEYS_FOR_NPZ,
    compress: bool = True,
) -> None:
    """cache_dict를 npz로 저장하되, 특정 키는 제외하고 저장합니다.

    Args:
        cache_dict: build_cache_dict_for_scenario()의 결과 dict.
        out_npz_path: 저장할 .npz 파일 경로.
        excluded_keys: npz에 넣지 않을 키 목록.
            - 기본값은 ("scenario_id", "neighbor_track_token")
        compress: True면 용량을 줄여 저장합니다.
    """
    npz_payload = build_npz_payload_from_cache_dict(
        cache_dict=cache_dict,
        excluded_keys=excluded_keys,
    )
    atomic_npz_save(
        arrays=npz_payload,
        out_path=out_npz_path,
        compress=compress,
    )


def build_stop_sign_points(
    stop_sign_xy_global: List[np.ndarray],
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    safety_len: int,
) -> np.ndarray:
    """stop sign 위치를 (n_stop_sign, safety_len, 2)로 만듭니다.

    WOMD에서는 stop sign이 점 1개로 표현되므로,
    그 점을 safety_len번 복사합니다.

    Args:
        stop_sign_xy_global: stop sign 점들의 리스트 (각각 (2,))
        ego_xy_global: ego 전역 위치 (2,)
        ego_yaw_global: ego 전역 yaw
        safety_len: 출력 점 개수(10)

    Returns:
        stop_sign_points: (n_stop_sign, safety_len, 2) float32 (ego 기준)
    """
    n = len(stop_sign_xy_global)
    out = np.zeros((n, safety_len, 2), dtype=np.float32)  # shape (n,10,2)
    for i, xy in enumerate(stop_sign_xy_global):
        # xy_local: (2,)
        xy_local = transform_points_global_to_ego_local(xy[None, :],
                                                        ego_xy_global,
                                                        ego_yaw_global)[0]
        out[i] = np.repeat(xy_local[None, :], repeats=safety_len, axis=0)
    return out


def build_polygon_points(
    polygons_xy_global: List[np.ndarray],  # each (M,2)
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    safety_len: int,
) -> np.ndarray:
    """polygon 리스트를 (n_poly, safety_len, 2)로 바꿉니다.

    요구사항대로 polygon을 둘레를 따라 같은 간격으로 safety_len개로 쪼갭니다.

    Args:
        polygons_xy_global: polygon 꼭짓점 리스트 (각각 (M,2))
        ego_xy_global: ego 전역 위치 (2,)
        ego_yaw_global: ego 전역 yaw
        safety_len: 출력 점 개수(10)

    Returns:
        points: (n_poly, safety_len, 2) float32 (ego 기준)
    """
    n = len(polygons_xy_global)
    out = np.zeros((n, safety_len, 2), dtype=np.float32)  # shape (n,10,2)

    for i, poly_xy in enumerate(polygons_xy_global):
        # poly_xy: (M,2)
        if poly_xy.shape[0] == 0:
            continue

        # 먼저 ego 기준으로 바꾼 뒤, 그 좌표에서 샘플링합니다.
        poly_local = transform_points_global_to_ego_local(
            poly_xy, ego_xy_global, ego_yaw_global)  # (M,2)
        sampled = resample_polyline_equal_distance(poly_local,
                                                   num_samples=safety_len,
                                                   closed=True)  # (10,2)
        out[i] = sampled

    return out


def nearest_points_vectors(
        centers_xy: np.ndarray,  # shape: (L,2)
        boundary_xy: np.ndarray,  # shape: (K,2)
) -> np.ndarray:
    """center 점 각각에 대해 boundary에서 가장 가까운 점을 찾아 (boundary-center) 벡터를 만듭니다.

    Args:
        centers_xy: (L,2) 중심선 점들
        boundary_xy: (K,2) 경계선 점들

    Returns:
        vecs: (L,2) (가까운 boundary점 - center점)
    """
    lane_len = int(centers_xy.shape[0])
    if boundary_xy.shape[0] == 0:
        return np.zeros((lane_len, 2), dtype=np.float32)

    vecs = np.zeros((lane_len, 2), dtype=np.float32)  # shape (L,2)
    for i in range(lane_len):
        c = centers_xy[i]  # (2,)
        diffs = boundary_xy - c[None, :]  # (K,2)
        d2 = np.sum(diffs * diffs, axis=1)  # (K,)
        j = int(np.argmin(d2))
        vecs[i] = boundary_xy[j] - c
    return vecs


def _collect_road_line_boundary_points_local(
    boundary_feature_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
) -> np.ndarray:
    """lane 한쪽(왼쪽/오른쪽)에 해당하는 road_line 점들을 ego 기준으로 모읍니다.

    이 함수는 "road_edge(연석 등)"는 완전히 제외하고,
    "road_line(차선 페인트 선)"에 해당하는 점들만 모읍니다.

    Args:
        boundary_feature_ids: lane의 한쪽에 연결된 boundary feature id 목록
        boundary_polylines_xy_global: boundary id -> polyline 점들, shape (K,2)
        boundary_id_to_kind: boundary id -> "road_line" 또는 "road_edge"
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 yaw (라디안)

    Returns:
        points_local: shape (K,2) float32
            - road_line 점들을 전부 모아 ego 기준 좌표계로 바꾼 결과
            - road_line이 하나도 없으면 shape (0,2)
    """
    points_list: List[np.ndarray] = []

    for fid in boundary_feature_ids:
        fid_int = int(fid)

        if boundary_id_to_kind.get(fid_int, "") != "road_line":
            continue

        poly_xy_g = boundary_polylines_xy_global.get(fid_int, None)
        if poly_xy_g is None:
            continue
        if poly_xy_g.ndim != 2 or poly_xy_g.shape[1] != 2 or poly_xy_g.shape[
                0] == 0:
            continue

        points_list.append(poly_xy_g.astype(np.float32))  # shape: (Ki,2)

    if len(points_list) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    points_xy_global = np.concatenate(points_list,
                                      axis=0).astype(np.float32)  # shape: (K,2)
    points_xy_local = transform_points_global_to_ego_local(
        points_xy_global, ego_xy_global, ego_yaw_global)  # shape: (K,2)
    return points_xy_local


def _align_boundary_points_to_center_indices(
    center_xy: np.ndarray,  # shape: (lane_len,2)
    boundary_xy: np.ndarray,  # shape: (K,2)
    max_valid_dist_m: float = 6.0,
    center_index_window: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """중심선의 각 점(j)에 대응되는 경계선 점(j)을 고릅니다.

    핵심 아이디어는 NuPlan처럼 "center[j]와 boundary[j]를 대응"시키는 것입니다.
    다만 WOMD는 경계선이 부분적으로만 있을 수 있으니,
    다음 규칙으로 '없는 구간'을 0 벡터로 처리합니다.

    처리 규칙:
    1) 중심선은 (lane_len,2)로 이미 샘플링되어 있다고 가정합니다.
    2) boundary_xy는 road_line 점들의 모음입니다(순서/연결 여부는 신뢰하지 않습니다).
    3) boundary 점마다 "가장 가까운 center 인덱스"를 구해둡니다.
    4) center[j]에서는, 그 j 주변(center_index_window 범위)으로 분류된 boundary 점들만 후보로 씁니다.
       - 이렇게 하면 lane의 다른 구간에 있는 점을 '우연히 가까워서' 끌어오는 것을 줄일 수 있습니다.
    5) 후보 중 가장 가까운 점이 max_valid_dist_m보다 멀면
       - 그 j 위치에는 경계가 없다고 보고 boundary_aligned[j] = center[j]로 둡니다.
       - 그러면 (boundary_aligned - center)가 (0,0)이 됩니다.

    Args:
        center_xy: 중심선 점들, shape (lane_len,2)
        boundary_xy: road_line 점들, shape (K,2)
        max_valid_dist_m: 이 거리보다 멀면 "경계가 없다"고 판단(보수적으로 6m 권장)
        center_index_window: center 인덱스 기준 후보를 고를 때 허용하는 인덱스 차이(기본 ±2)

    Returns:
        boundary_aligned: shape (lane_len,2) float32
            - 각 j에 대응되는 경계선 점
            - 없다고 판단되면 center[j]와 동일(그래서 벡터는 0)
        valid_mask: shape (lane_len,) bool
            - True면 그 j에서 경계점을 찾았고, False면 못 찾았다는 뜻
    """
    lane_len = int(center_xy.shape[0])

    boundary_aligned = center_xy.astype(
        np.float32).copy()  # shape: (lane_len,2)
    valid_mask = np.zeros((lane_len,), dtype=bool)  # shape: (lane_len,)

    if boundary_xy.ndim != 2 or boundary_xy.shape[1] != 2 or boundary_xy.shape[
            0] == 0:
        return boundary_aligned, valid_mask

    # boundary 점마다 가장 가까운 center 인덱스 구하기
    # d2_bc: (K, lane_len)
    diffs_bc = boundary_xy[:, None, :] - center_xy[
        None, :, :]  # shape: (K,lane_len,2)
    d2_bc = np.sum(diffs_bc * diffs_bc, axis=-1)  # shape: (K,lane_len)
    nearest_center_idx = np.argmin(d2_bc,
                                   axis=1).astype(np.int64)  # shape: (K,)

    win = int(center_index_window)
    max_d2 = float(max_valid_dist_m)**2

    for j in range(lane_len):
        # "이 center[j] 주변 구간"으로 보이는 boundary 점만 후보로 사용
        mask = np.abs(nearest_center_idx - int(j)) <= win  # shape: (K,)
        if not bool(np.any(mask)):
            continue

        cand = boundary_xy[mask]  # shape: (M,2)
        # center[j]에 가장 가까운 후보 선택
        diffs = cand - center_xy[j][None, :]  # shape: (M,2)
        d2 = np.sum(diffs * diffs, axis=1)  # shape: (M,)
        m = int(np.argmin(d2))
        if float(d2[m]) <= max_d2:
            boundary_aligned[j] = cand[m].astype(np.float32)  # shape: (2,)
            valid_mask[j] = True

    return boundary_aligned, valid_mask


def _compute_nearest_center_indices(
        points_xy: np.ndarray,  # shape: (K, 2)
        center_xy: np.ndarray,  # shape: (lane_len, 2)
) -> np.ndarray:
    """각 점이 중심선의 몇 번째 점과 가장 가까운지 인덱스를 구합니다.

    Args:
        points_xy: 경계(또는 다른 점들) 좌표, shape (K, 2)
        center_xy: 중심선 좌표(이미 lane_len개로 맞춰진 상태), shape (lane_len, 2)

    Returns:
        nearest_idx: points_xy 각 점마다 가장 가까운 center_xy 인덱스, shape (K,), dtype int64
    """
    # points_xy: (K,2), center_xy: (L,2)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2 or points_xy.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    if center_xy.ndim != 2 or center_xy.shape[1] != 2 or center_xy.shape[0] == 0:
        return np.zeros((points_xy.shape[0],), dtype=np.int64)

    diffs = points_xy[:, None, :] - center_xy[None, :, :]  # shape: (K, L, 2)
    d2 = np.sum(diffs * diffs, axis=-1)  # shape: (K, L)
    nearest_idx = np.argmin(d2, axis=1).astype(np.int64)  # shape: (K,)
    return nearest_idx


def _orient_and_sort_boundary_polylines_along_center(
        boundary_polylines_local: List[np.ndarray],  # each shape: (Ki, 2)
        center_sampled: np.ndarray,  # shape: (lane_len, 2)
) -> List[np.ndarray]:
    """경계선 조각(여러 개일 수 있음)을 '중심선 진행 방향'에 맞게 정리합니다.

    WOMD에서는 한쪽 경계선이 한 덩어리가 아니라 여러 조각으로 나뉘어 들어올 수 있습니다.
    NuPlan 방식(같은 인덱스끼리 빼기)을 하려면, 경계선도 "한 줄"처럼 정리한 다음 샘플링하는 게 편합니다.

    이 함수는 아래를 합니다.
    1) 각 조각이 거꾸로 들어온 경우가 있어서, 조각의 앞/뒤를 뒤집을지 판단합니다.
       - 방법: 조각의 첫 점과 끝 점이 중심선에서 가까운 인덱스를 비교합니다.
       - 끝점 쪽 인덱스가 더 작으면(=거꾸로), 조각을 뒤집습니다.
    2) 여러 조각이 있다면, 중심선 인덱스 기준으로 앞쪽 조각부터 오도록 정렬합니다.
       - 방법: 각 조각 점들이 가까운 중심선 인덱스들의 평균값을 구해 그 값으로 정렬합니다.

    Args:
        boundary_polylines_local: 경계선 조각들의 리스트(ego 기준), 각 원소 shape (Ki, 2)
        center_sampled: 중심선(ego 기준, lane_len개), shape (lane_len, 2)

    Returns:
        sorted_polylines: 방향이 맞춰지고 정렬된 경계선 조각 리스트
    """
    processed: List[Tuple[float, np.ndarray]] = []

    for poly in boundary_polylines_local:
        # poly: (Ki,2)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] == 0:
            continue

        nearest_idx = _compute_nearest_center_indices(poly,
                                                      center_sampled)  # (Ki,)
        if nearest_idx.size == 0:
            continue

        # 조각 방향 정리(거꾸로면 뒤집기)
        start_idx = int(nearest_idx[0])
        end_idx = int(nearest_idx[-1])
        poly_oriented = poly
        nearest_idx_oriented = nearest_idx
        if end_idx < start_idx:
            poly_oriented = poly[::-1].copy()
            nearest_idx_oriented = nearest_idx[::-1].copy()

        # 이 조각이 lane에서 대략 어디쯤인지(정렬용 스코어)
        score = float(nearest_idx_oriented.astype(np.float32).mean())
        processed.append((score, poly_oriented.astype(np.float32)))

    processed.sort(key=lambda x: x[0])
    return [p for _, p in processed]


def _maybe_flip_points_to_match_center_start(
        boundary_points: np.ndarray,  # shape: (K, 2)
        center_sampled: np.ndarray,  # shape: (lane_len, 2)
) -> np.ndarray:
    """NuPlan의 '플립'과 같은 의도로, 경계선의 시작점이 center[0]에 가깝게 만듭니다.

    NuPlan에서는 경계선 점들의 순서가 거꾸로 들어오는 경우를 대비해,
    'center[0]에 더 가까운 쪽이 boundary[0]'이 되도록 뒤집습니다.

    Args:
        boundary_points: 경계선 점들의 줄(ego 기준), shape (K, 2)
        center_sampled: 중심선 점들(ego 기준, lane_len개), shape (lane_len, 2)

    Returns:
        boundary_points_fixed: 필요하면 뒤집힌 boundary_points, shape (K, 2)
    """
    if boundary_points.ndim != 2 or boundary_points.shape[
            1] != 2 or boundary_points.shape[0] == 0:
        return boundary_points
    if center_sampled.ndim != 2 or center_sampled.shape[
            1] != 2 or center_sampled.shape[0] == 0:
        return boundary_points

    # d0 = boundary[0]이 center[0]과 얼마나 가까운지
    # d1 = boundary[-1]이 center[0]과 얼마나 가까운지
    d0 = float(np.linalg.norm(boundary_points[0] - center_sampled[0]))
    d1 = float(np.linalg.norm(boundary_points[-1] - center_sampled[0]))
    if d1 < d0:
        return boundary_points[::-1].copy()
    return boundary_points


def _build_lane_side_boundary_sampled_like_nuplan(
    center_sampled: np.ndarray,  # shape: (lane_len, 2), ego 기준
    boundary_feature_ids: List[int],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    ego_xy_global: np.ndarray,  # shape: (2,)
    ego_yaw_global: float,
    lane_len: int,
    max_valid_dist_m: float = 6.0,
) -> np.ndarray:
    """WOMD 경계선을 NuPlan 스타일로 'lane_len개 점'으로 만든 뒤, 없는 구간은 안전하게 비웁니다.

    NuPlan 방식의 핵심은:
    - 중심선(center)과 경계선(left/right)을 각각 같은 개수(lane_len=10)로 점을 맞춘 다음,
    - 같은 인덱스끼리 빼서 (dx, dy)를 만든다는 점입니다.

    WOMD는 한쪽 경계가 여러 조각으로 나뉘어 있을 수 있고,
    어떤 구간은 경계가 아예 없을 수도 있습니다.
    그래서 이 함수는 아래 순서로 처리합니다.

    1) boundary_feature_ids 중 road_line인 것만 골라서, 각 조각의 점들을 가져옵니다. (global 좌표)
    2) ego 기준 좌표로 바꿉니다. (local 좌표)
    3) 조각들의 방향을 중심선 진행 방향에 맞게 뒤집고, 앞쪽 조각부터 오도록 정렬합니다.
    4) 정렬된 조각들을 하나의 "점들의 줄"로 이어 붙입니다.
    5) NuPlan의 플립 로직처럼, boundary[0]이 center[0]에 가깝도록 한 번 더 뒤집을 수 있습니다.
    6) 이어 붙인 경계선을 lane_len개 점으로 같은 간격으로 샘플링합니다.
    7) 샘플링된 boundary[j]가 center[j]에서 너무 멀면(> max_valid_dist_m),
       그 j는 "경계가 없는 구간"이라고 보고 boundary[j]를 center[j]로 바꿉니다.
       → 그러면 (boundary - center)가 (0,0)이 됩니다.

    Args:
        center_sampled: 중심선 점들(ego 기준), shape (lane_len, 2)
        boundary_feature_ids: lane 한쪽(left/right)에 연결된 boundary feature id 목록
        boundary_polylines_xy_global: boundary id -> 점들(global), shape (Ki, 2)
        boundary_id_to_kind: boundary id -> "road_line" / "road_edge"
        ego_xy_global: ego 전역 위치, shape (2,)
        ego_yaw_global: ego 전역 yaw (라디안)
        lane_len: 출력 점 개수(예: 10)
        max_valid_dist_m: center[j]와 boundary[j]의 거리가 이 값보다 크면 "없음" 처리

    Returns:
        boundary_sampled: NuPlan 스타일로 만든 경계선 점들(ego 기준), shape (lane_len, 2), float32
            - 경계가 없다고 판단된 인덱스는 center_sampled와 동일하게 들어갑니다.
    """
    # center_sampled: (lane_len,2)
    if center_sampled.ndim != 2 or center_sampled.shape[
            1] != 2 or center_sampled.shape[0] != int(lane_len):
        return np.zeros((int(lane_len), 2), dtype=np.float32)

    # 1) road_line 조각 모으기 (global)
    polylines_global: List[np.ndarray] = []
    for fid in boundary_feature_ids:
        fid_int = int(fid)
        if boundary_id_to_kind.get(fid_int, "") != "road_line":
            continue
        poly_g = boundary_polylines_xy_global.get(fid_int, None)
        if poly_g is None:
            continue
        if poly_g.ndim != 2 or poly_g.shape[1] != 2 or poly_g.shape[0] == 0:
            continue
        polylines_global.append(poly_g.astype(np.float32))  # shape: (Ki,2)

    # 경계가 아예 없으면: NuPlan에서도 "없음"은 결국 벡터 0으로 처리하는 게 안전함
    if len(polylines_global) == 0:
        return center_sampled.astype(np.float32).copy()

    # 2) ego 기준(local)로 변환
    polylines_local: List[np.ndarray] = []
    for poly_g in polylines_global:
        poly_l = transform_points_global_to_ego_local(poly_g, ego_xy_global,
                                                      ego_yaw_global)  # (Ki,2)
        if poly_l.shape[0] > 0:
            polylines_local.append(poly_l.astype(np.float32))

    if len(polylines_local) == 0:
        return center_sampled.astype(np.float32).copy()

    # 3) 중심선 방향 기준으로 조각 방향/순서 정리
    polylines_sorted = _orient_and_sort_boundary_polylines_along_center(
        boundary_polylines_local=polylines_local,
        center_sampled=center_sampled,
    )
    if len(polylines_sorted) == 0:
        return center_sampled.astype(np.float32).copy()

    # 4) 이어 붙여 하나의 줄로 만들기
    boundary_points = np.concatenate(polylines_sorted,
                                     axis=0).astype(np.float32)  # shape: (K,2)
    if boundary_points.shape[0] == 0:
        return center_sampled.astype(np.float32).copy()

    # 5) NuPlan 플립(안전)
    boundary_points = _maybe_flip_points_to_match_center_start(
        boundary_points, center_sampled)  # (K,2)

    # 6) lane_len개로 샘플링
    boundary_sampled = resample_polyline_equal_distance(
        boundary_points,
        num_samples=int(lane_len),
        closed=False,
    ).astype(np.float32)  # shape: (lane_len,2)

    # 7) 너무 멀면 "없음" 처리 -> boundary[j]=center[j]
    diffs = boundary_sampled - center_sampled  # shape: (lane_len,2)
    dists = np.linalg.norm(diffs,
                           axis=1).astype(np.float32)  # shape: (lane_len,)
    far_mask = dists > float(max_valid_dist_m)  # shape: (lane_len,)
    if np.any(far_mask):
        boundary_sampled[far_mask] = center_sampled[far_mask]

    return boundary_sampled.astype(np.float32)


def build_lane_arrays(
    lanes: List[LaneInfo],
    boundary_polylines_xy_global: Dict[int, np.ndarray],
    boundary_id_to_kind: Dict[int, str],
    lane_id_to_light: Dict[int, np.ndarray],
    ego_xy_global: np.ndarray,
    ego_yaw_global: float,
    lane_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """차선 정보를 요구사항의 lanes 포맷으로 만듭니다.

    lanes: (lane_num, lane_len=10, 12)
        0-1  : center (x,y)
        2-3  : center vector (dx,dy)
        4-5  : left boundary vector (dx,dy)   = (closest_on_left_boundary - center)
        6-7  : right boundary vector (dx,dy)  = (closest_on_right_boundary - center)
        8-11 : lane light one-hot [GO, CAUTION, STOP, UNKNOWN]

    새 요구사항 반영 포인트:
      - left_boundaries/right_boundaries(=BoundarySegment)에서
        center 샘플 점이 속한 구간을 덮는 boundary_feature_id를 찾음
      - 그 boundary_feature_id의 RoadLine/RoadEdge polyline에 대해
        점→폴리라인 최소거리(가장 가까운 위치)를 구해서 dx,dy 계산
      - 너무 멀거나(left/right 방향 틀리면) 0으로 비움

    Returns:
        lanes_arr: (lane_num, lane_len, 12) float32
        lanes_speed_limit: (lane_num, 1) float32 (m/s)
        lanes_has_speed_limit: (lane_num, 1) bool
        lane_light: (lane_num, 4) float32
    """
    lane_num = len(lanes)

    lanes_arr = np.zeros((lane_num, int(lane_len), 12), dtype=np.float32)
    lanes_speed_limit = np.zeros((lane_num, 1), dtype=np.float32)
    lanes_has_speed_limit = np.zeros((lane_num, 1), dtype=bool)
    lane_light = np.zeros((lane_num, 4), dtype=np.float32)

    # ✅ 거리 제한(“널널한” 기준)
    # - road_line: 보통 center에서 멀리 갈 일이 적어서 조금 타이트
    # - road_edge: 도로 가장자리라 더 멀 수 있어 더 넉넉
    max_dist_road_line_m = 8.0
    max_dist_road_edge_m = 15.0

    # ✅ 성능: boundary polyline ego 변환 캐시(시나리오 내에서 재사용)
    boundary_polyline_local_cache: Dict[int, np.ndarray] = {}

    for i, lane in enumerate(lanes):
        # 1) centerline 샘플링 (ego 기준) + "원본 구간 index" 같이 얻기
        center_xy_g = lane.centerline_xy_global  # (P,2)
        center_xy_l = transform_points_global_to_ego_local(
            center_xy_g, ego_xy_global, ego_yaw_global)  # (P,2)

        center_sampled, center_seg_idx = resample_polyline_equal_distance_with_segment_indices(
            center_xy_l,
            num_samples=int(lane_len),
            closed=False,
        )  # (lane_len,2), (lane_len,)

        # 2) center vector (dx,dy) (마지막은 0)
        center_vec = np.zeros((int(lane_len), 2),
                              dtype=np.float32)  # (lane_len,2)
        if int(lane_len) >= 2:
            center_vec[:-1] = center_sampled[1:] - center_sampled[:-1]
        center_vec[-1] = 0.0

        # 3) 진행방향 기반 left_normal 계산 (방향 체크용)
        _, left_normals = compute_center_tangent_and_left_normals(
            center_sampled)  # (lane_len,2)

        # 4) 새 요구사항 방식으로 left/right boundary 벡터 계산
        left_vec = compute_lane_boundary_vectors_for_side(
            center_sampled_xy_local=center_sampled,
            center_seg_indices=center_seg_idx,
            center_left_normals_unit=left_normals,
            boundary_segments=lane.left_boundary_segments,
            boundary_polylines_xy_global=boundary_polylines_xy_global,
            boundary_id_to_kind=boundary_id_to_kind,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
            boundary_polyline_local_cache=boundary_polyline_local_cache,
            side="left",
            max_dist_road_line_m=float(max_dist_road_line_m),
            max_dist_road_edge_m=float(max_dist_road_edge_m),
        )  # (lane_len,2)

        right_vec = compute_lane_boundary_vectors_for_side(
            center_sampled_xy_local=center_sampled,
            center_seg_indices=center_seg_idx,
            center_left_normals_unit=left_normals,
            boundary_segments=lane.right_boundary_segments,
            boundary_polylines_xy_global=boundary_polylines_xy_global,
            boundary_id_to_kind=boundary_id_to_kind,
            ego_xy_global=ego_xy_global,
            ego_yaw_global=ego_yaw_global,
            boundary_polyline_local_cache=boundary_polyline_local_cache,
            side="right",
            max_dist_road_line_m=float(max_dist_road_line_m),
            max_dist_road_edge_m=float(max_dist_road_edge_m),
        )  # (lane_len,2)

        # 5) lane light (기존 유지)
        light = lane_id_to_light.get(
            lane.lane_id,
            np.array([0, 0, 0, 1], dtype=np.float32),
        )  # (4,)
        lane_light[i] = light
        lanes_arr[i, :, 8:12] = light[None, :]  # broadcast

        # 6) 속도제한(m/s) (기존 유지)
        if lane.speed_limit_mph > 0.0:
            lanes_has_speed_limit[i, 0] = True
            lanes_speed_limit[i, 0] = float(lane.speed_limit_mph) * 0.44704
        else:
            lanes_has_speed_limit[i, 0] = False
            lanes_speed_limit[i, 0] = 0.0

        # 7) 최종 lanes 채우기
        lanes_arr[i, :, 0:2] = center_sampled
        lanes_arr[i, :, 2:4] = center_vec
        lanes_arr[i, :, 4:6] = left_vec
        lanes_arr[i, :, 6:8] = right_vec

    return lanes_arr, lanes_speed_limit, lanes_has_speed_limit, lane_light


def get_womd_track_token(track: Any) -> str:
    """트랙(에이전트 1개)을 구분하는 고유 문자열을 얻습니다.

    - WOMD에서는 트랙을 구분하는 고유 값으로 track.id(숫자)를 제공합니다.
    - 따라서 이 함수는 track.id를 정수로 바꾼 뒤, 문자열로 바꾼 값만 반환합니다.
    - 다른 값은 보지 않고, 조합도 하지 않습니다.

    Args:
        track: scenario.tracks의 원소(트랙 1개)

    Returns:
        token: track.id를 문자열로 바꾼 값
    """
    if hasattr(track, "id"):
        return str(int(getattr(track, "id")))
    return ""


from typing import Dict, List, Tuple
import numpy as np
from typing import Any


def tfrecord_element_to_bytes(element: Any) -> bytes:
    """TFRecordDataset에서 나온 원소를 bytes로 안전하게 바꿉니다.

    환경에 따라 dataset 반복에서 나오는 값이
    - bytes
    - numpy의 bytes 스칼라
    - (혹시) tf.Tensor
    처럼 다를 수 있어서, 어느 경우든 bytes로 통일해 반환합니다.

    Args:
        element: dataset에서 나온 원소(한 레코드).

    Returns:
        record_bytes: bytes. Scenario.ParseFromString에 바로 넣을 수 있는 형태.
    """
    if isinstance(element, (bytes, bytearray)):
        return bytes(element)

    # tf.Tensor 같은 경우
    if hasattr(element, "numpy"):
        return bytes(element.numpy())

    # numpy 스칼라/배열 같은 경우
    if hasattr(element, "tobytes"):
        return element.tobytes()

    return bytes(element)


# =========================
# 시나리오 -> pkl dict 만들기
# =========================
def build_cache_dict_for_scenario(
    scenario: scenario_pb2.Scenario,) -> Dict[str, Any]:
    """Scenario 1개를 요구사항 포맷의 dict로 바꿉니다.

    이 dict가 그대로 pickle로 저장됩니다.

    Returns dict keys(요구사항):
        - scenario_id: str
        - ego_agent_past: (21,11) float32
        - ego_future_gt_3_dim: (80,3) float32
        - ego_future_gt_11_dim: (80,11) float32

        - neighbor_role: (A,2) bool
        - neighbor_id: (A,) int64
        - neighbor_z: (A,) float32
        - neighbor_shape: (A,3) float32  [length,width,height] 평균
        - neighbor_agents_past: (A,21,11) float32
        - neighbor_future_gt_3_dim: (A,80,3) float32
        - neighbor_track_token: List[str] (길이 A)

        - stop_sign_points: (Ns,10,2) float32
        - crosswalk_points: (Nc,10,2) float32
        - speed_bump_points: (Nb,10,2) float32

        - lanes: (L,10,12) float32
        - lanes_speed_limit: (L,1) float32
        - lanes_has_speed_limit: (L,1) bool
        - lane_light: (L,4) float32

        lane_type: (L,4) float32

        left_line_type: (L,10) float32

        right_line_type: (L,10) float32

        road_edge: (E,10,2) float32

        road_edge_type: (E,3) float32

        driveway: (D,10,2) float32

    Args:
        scenario: Scenario proto

    Returns:
        cache_dict: pickle로 저장할 dict
    """
    _require_womd_lengths_initialized()
    track_key_to_array = decode_tracks_and_roles_from_scenario(scenario)

    object_id_all = track_key_to_array["object_id"]  # (N,)
    object_type_all = track_key_to_array["object_type"]  # (N,)
    states_all = track_key_to_array["states"]  # (N,S,9)
    valid_all = track_key_to_array["valid"]  # (N,S)
    role_interest_all = track_key_to_array["role_interest"]  # (N,)
    role_predict_all = track_key_to_array["role_predict"]  # (N,)

    ego_idx = int(track_key_to_array["ego_index"][0])
    current_t = int(scenario.current_time_index)
    num_tracks = int(states_all.shape[0])
    num_steps = int(states_all.shape[1])

    ego_xy_global, ego_yaw_global, ego_z_global = compute_ego_pose_at_current(
        scenario, track_key_to_array)

    # -------------------------
    # 모든 트랙을 ego 좌표계로 미리 변환
    # -------------------------
    pos_xy_global_all = states_all[:, :, 0:2]  # shape (N,S,2)
    vel_xy_global_all = states_all[:, :, 7:9]  # shape (N,S,2)
    yaw_global_all = states_all[:, :, 6]  # shape (N,S)
    z_global_all = states_all[:, :, 2]  # shape (N,S)

    # local 변환
    pos_xy_local_all = transform_points_global_to_ego_local(
        pos_xy_global_all, ego_xy_global, ego_yaw_global)  # (N,S,2)
    vel_xy_local_all = transform_vectors_global_to_ego_local(
        vel_xy_global_all, ego_yaw_global)  # (N,S,2)

    # heading은 ego_yaw 기준으로 상대값
    yaw_local_all = wrap_angle_np(yaw_global_all - ego_yaw_global)  # (N,S)

    # width/length: 요구사항 11차원에서 (width, length) 순서
    width_all = states_all[:, :, 4]  # (N,S)
    length_all = states_all[:, :, 3]  # (N,S)
    width_length_all = np.stack([width_all, length_all],
                                axis=-1).astype(np.float32)  # (N,S,2)

    # z는 ego 기준으로 상대값으로 저장
    z_rel_all = (z_global_all - ego_z_global).astype(np.float32)  # (N,S)

    # one-hot 타입 미리 생성
    one_hot_all = build_agent_type_one_hot_all(
        object_type_minus1_all=object_type_all,  # shape: (N,)
        ego_track_index=ego_idx,
    )
    # ego는 무조건 VEHICLE로 고정
    one_hot_all[ego_idx] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # -------------------------
    # time index 구성
    # -------------------------
    past_indices, future_indices = build_time_indices(
        current_time_index=current_t,
        num_total_steps=num_steps,
        desired_past_len=TIME_LEN,
        desired_future_len=FUTURE_LEN,
    )

    # -------------------------
    # ego: past/future 만들기
    # -------------------------
    ego_p_pos, ego_p_vel, ego_p_yaw, ego_p_wl, ego_p_valid = gather_agent_sequence_by_indices(
        agent_idx=ego_idx,
        time_indices=past_indices,
        pos_xy_local_all=pos_xy_local_all,
        vel_xy_local_all=vel_xy_local_all,
        yaw_local_all=yaw_local_all,
        width_length_all=width_length_all,
        valid_all=valid_all,
    )
    ego_p_pos_i, ego_p_vel_i, ego_p_yaw_i, ego_p_wl_i, ego_p_valid_i = build_interpolated_agent_traj(
        ego_p_pos, ego_p_vel, ego_p_yaw, ego_p_wl, ego_p_valid)
    ego_agent_past = pack_agent_features_11(
        pos_xy=ego_p_pos_i,
        vel_xy=ego_p_vel_i,
        yaw_rad=ego_p_yaw_i,
        width_length=ego_p_wl_i,
        agent_one_hot=one_hot_all[ego_idx],
        valid_mask=ego_p_valid_i,
    )  # (21,11)

    ego_f_pos, ego_f_vel, ego_f_yaw, ego_f_wl, ego_f_valid = gather_agent_sequence_by_indices(
        agent_idx=ego_idx,
        time_indices=future_indices,
        pos_xy_local_all=pos_xy_local_all,
        vel_xy_local_all=vel_xy_local_all,
        yaw_local_all=yaw_local_all,
        width_length_all=width_length_all,
        valid_all=valid_all,
    )
    ego_f_pos_i, ego_f_vel_i, ego_f_yaw_i, ego_f_wl_i, ego_f_valid_i = build_interpolated_agent_traj(
        ego_f_pos, ego_f_vel, ego_f_yaw, ego_f_wl, ego_f_valid)

    ego_future_gt_3_dim = pack_agent_future_3(
        pos_xy=ego_f_pos_i,
        yaw_rad=ego_f_yaw_i,
        valid_mask=ego_f_valid_i,
    )  # (80,3)

    ego_future_gt_11_dim = pack_agent_features_11(
        pos_xy=ego_f_pos_i,
        vel_xy=ego_f_vel_i,
        yaw_rad=ego_f_yaw_i,
        width_length=ego_f_wl_i,
        agent_one_hot=one_hot_all[ego_idx],
        valid_mask=ego_f_valid_i,
    )  # (80,11)

    # -------------------------
    # neighbors: 현재 시점에 존재하는(agent valid at current)만
    # -------------------------
    neighbor_indices = collect_neighbor_indices_at_current(
        valid_all=valid_all,  # shape: (N,S)
        current_time_index=current_t,
        ego_track_index=ego_idx,
    )
    agent_num = int(neighbor_indices.shape[0])

    neighbor_role = np.zeros((agent_num, 2),
                             dtype=bool)  # (A,2) [interest, predict]
    neighbor_id = np.zeros((agent_num,), dtype=np.int64)  # (A,)
    neighbor_z = np.zeros((agent_num,), dtype=np.float32)  # (A,)
    neighbor_shape = np.zeros((agent_num, 3),
                              dtype=np.float32)  # (A,3) [length,width,height]
    neighbor_agents_past = np.zeros((agent_num, TIME_LEN, 11),
                                    dtype=np.float32)  # (A,21,11)
    neighbor_future_gt_3_dim = np.zeros((agent_num, FUTURE_LEN, 3),
                                        dtype=np.float32)  # (A,80,3)
    neighbor_track_token: List[str] = [""] * agent_num  # ✅ 추가: 길이 A

    for out_i, tr_i in enumerate(neighbor_indices.tolist()):
        tr_i = int(tr_i)
        neighbor_track_token[out_i] = get_womd_track_token(
            scenario.tracks[tr_i])  # ✅ 추가

        neighbor_id[out_i] = object_id_all[tr_i]
        neighbor_role[out_i, 0] = bool(role_interest_all[tr_i])
        neighbor_role[out_i, 1] = bool(role_predict_all[tr_i])
        neighbor_z[out_i] = float(z_rel_all[tr_i, current_t])

        # shape 평균: states[...,3:6] = [length,width,height]
        valid_mask_full = valid_all[tr_i]  # (S,)
        if np.any(valid_mask_full):
            shape_vals = states_all[tr_i, valid_mask_full, 3:6]  # (K,3)
            neighbor_shape[out_i] = shape_vals.mean(axis=0).astype(np.float32)
        else:
            neighbor_shape[out_i] = 0.0

        # past 11-dim
        p_pos, p_vel, p_yaw, p_wl, p_valid = gather_agent_sequence_by_indices(
            agent_idx=tr_i,
            time_indices=past_indices,
            pos_xy_local_all=pos_xy_local_all,
            vel_xy_local_all=vel_xy_local_all,
            yaw_local_all=yaw_local_all,
            width_length_all=width_length_all,
            valid_all=valid_all,
        )
        p_pos_i, p_vel_i, p_yaw_i, p_wl_i, p_valid_i = build_interpolated_agent_traj(
            p_pos, p_vel, p_yaw, p_wl, p_valid)
        neighbor_agents_past[out_i] = pack_agent_features_11(
            pos_xy=p_pos_i,
            vel_xy=p_vel_i,
            yaw_rad=p_yaw_i,
            width_length=p_wl_i,
            agent_one_hot=one_hot_all[tr_i],
            valid_mask=p_valid_i,
        )

        # future 3-dim
        f_pos, f_vel, f_yaw, f_wl, f_valid = gather_agent_sequence_by_indices(
            agent_idx=tr_i,
            time_indices=future_indices,
            pos_xy_local_all=pos_xy_local_all,
            vel_xy_local_all=vel_xy_local_all,
            yaw_local_all=yaw_local_all,
            width_length_all=width_length_all,
            valid_all=valid_all,
        )
        f_pos_i, _, f_yaw_i, _, f_valid_i = build_interpolated_agent_traj(
            f_pos, f_vel, f_yaw, f_wl, f_valid)
        neighbor_future_gt_3_dim[out_i] = pack_agent_future_3(
            pos_xy=f_pos_i,
            yaw_rad=f_yaw_i,
            valid_mask=f_valid_i,
        )

    # -------------------------
    # map: stop/crosswalk/speed_bump/lanes
    # -------------------------
    parsed_map_all = parse_map_from_scenario(scenario)
    # ✅ use_filter_radius=True일 때만 반경 필터 적용
    if _USE_FILTER_RADIUS:
        parsed_map = filter_parsed_map_by_radius(
            parsed_map=parsed_map_all,
            ego_xy_global=ego_xy_global,
            filter_radius_m=float(_FILTER_RADIUS_M),
        )
    else:
        parsed_map = parsed_map_all

    lane_id_to_light = extract_lane_signals_at_current(scenario)

    stop_sign_points = build_stop_sign_points(
        parsed_map.stop_sign_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    crosswalk_points = build_polygon_points(
        parsed_map.crosswalk_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    speed_bump_points = build_polygon_points(
        parsed_map.speed_bump_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    lanes_arr, lanes_speed_limit, lanes_has_speed_limit, lane_light = build_lane_arrays(
        lanes=parsed_map.lanes,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        lane_id_to_light=lane_id_to_light,
        ego_xy_global=ego_xy_global,
        ego_yaw_global=ego_yaw_global,
        lane_len=LANE_LEN,
    )

    lane_type = build_lane_type_one_hot_array(parsed_map.lanes)

    left_line_type, right_line_type = build_lane_line_type_arrays(
        lanes=parsed_map.lanes,
        boundary_id_to_kind=parsed_map.boundary_id_to_kind,
        road_line_type_by_id=parsed_map.road_line_type_by_id,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
    )

    road_edge, road_edge_type = build_road_edge_points_and_types(
        road_edge_ids=parsed_map.road_edge_ids,
        boundary_polylines_xy_global=parsed_map.boundary_polylines_xy_global,
        road_edge_type_by_id=parsed_map.road_edge_type_by_id,
        ego_xy_global=ego_xy_global,
        ego_yaw_global=ego_yaw_global,
        safety_len=SAFETY_LEN,
    )

    driveway = build_polygon_points(
        parsed_map.driveway_polygons_xy_global,
        ego_xy_global,
        ego_yaw_global,
        SAFETY_LEN,
    )

    cache_dict: Dict[str, Any] = {
        "scenario_id": str(scenario.scenario_id),
        "ego_agent_past": ego_agent_past,  # (21,11)
        "ego_future_gt_3_dim": ego_future_gt_3_dim,  # (80,3)
        "ego_future_gt_11_dim": ego_future_gt_11_dim,  # (80,11)
        "neighbor_role": neighbor_role,  # (A,2) bool
        "neighbor_id": neighbor_id,  # (A,)
        "neighbor_z": neighbor_z,  # (A,)
        "neighbor_shape": neighbor_shape,  # (A,3)
        "neighbor_agents_past": neighbor_agents_past,  # (A,21,11)
        "neighbor_future_gt_3_dim": neighbor_future_gt_3_dim,  # (A,80,3)
        "neighbor_track_token": neighbor_track_token,  # ✅ 추가: List[str], 길이 A
        "stop_sign_points": stop_sign_points,  # (Ns,10,2)
        "crosswalk_points": crosswalk_points,  # (Nc,10,2)
        "speed_bump_points": speed_bump_points,  # (Nb,10,2)
        "lanes": lanes_arr,  # (L,10,12)
        "lanes_speed_limit": lanes_speed_limit,  # (L,1)
        "lanes_has_speed_limit": lanes_has_speed_limit,  # (L,1)
        "lane_light": lane_light,  # (L,4)
        # ✅ 추가 캐싱
        "lane_type": lane_type,  # (L,4)
        "left_line_type": left_line_type,  # (L,10)

        # 요구사항 이름이 right_lne_type로 되어 있어서 key는 그렇게 저장
        "right_line_type": right_line_type,  # (L,10)
        "road_edge": road_edge,  # (E,10,2)
        "road_edge_type": road_edge_type,  # (E,3)
        "driveway": driveway,  # (D,10,2)
    }
    return cache_dict


# =========================
# TFRecord 파일 1개 처리(멀티프로세스 워커)
# =========================
def process_one_tfrecord_file(
    tfrecord_path: str,
    split: str,
    caching_dir: str,
    overwrite: bool,
    save_image: bool,
) -> Tuple[str, int, int, int, str]:
    """TFRecord 파일 1개를 읽어서, 안에 들어있는 시나리오들을 캐싱합니다.

    병렬 처리를 위해 "파일 단위"로 처리합니다.

    Args:
        tfrecord_path: 입력 tfrecord 경로
        split: training/validation/testing
        caching_dir: /home/user/womd_v1_3/cache
        overwrite: True면 이미 npz가 있어도 다시 만듭니다.

    Returns:
        (tfrecord_path, processed, skipped, failed, message)
    """
    if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
        return tfrecord_path, 0, 0, 0, "ABORTED_BY_USER"
    _claim_logger_if_needed()
    _log(f"START file={tfrecord_path} split={split}")
    t0 = time.perf_counter()
    last_hb = time.monotonic()
    hb_sec = 10.0
    processed = 0
    skipped = 0
    failed = 0
    message = "OK"

    tfrecord_p = Path(tfrecord_path)
    caching_p = Path(caching_dir)

    out_split_dir = caching_p / split
    ensure_dir(out_split_dir)

    out_validation_split_dir: Optional[Path] = None
    if split == "validation":
        out_validation_split_dir = caching_p / "validation_tfrecords_splitted"
        ensure_dir(out_validation_split_dir)

    compression_type = guess_tfrecord_compression(tfrecord_p)
    dataset = tf.data.TFRecordDataset(
        tfrecord_p.as_posix(),
        compression_type=compression_type,
        num_parallel_reads=1,
    ).prefetch(1)
    _log("TFRecordDataset created")

    for k, record_elem in enumerate(dataset.as_numpy_iterator()):
        if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
            message = "ABORTED_BY_USER"
            break

        try:
            if k == 0:
                _log("FIRST record fetched (as_numpy_iterator)")

            record_bytes = tfrecord_element_to_bytes(record_elem)
            if k == 0:
                _log("FIRST record -> numpy done")

            scenario = parse_scenario_from_bytes(record_bytes)
            scenario_id = str(scenario.scenario_id)
            if processed == 0:
                _log(f"FIRST scenario parsed id={scenario_id}")

            # ✅ pkl -> npz
            out_npz_path = out_split_dir / f"{scenario_id}.npz"
            out_tfrecord_path = None
            if out_validation_split_dir is not None:
                out_tfrecord_path = out_validation_split_dir / f"{scenario_id}.tfrecords"

            # 이미 있으면 skip (overwrite=False일 때)
            if (not overwrite) and out_npz_path.exists():
                if split == "validation" and out_tfrecord_path is not None and (
                        not out_tfrecord_path.exists()):
                    with tf.io.TFRecordWriter(out_tfrecord_path.as_posix()) as w:
                        w.write(record_bytes)
                skipped += 1
                continue

            if _WORKER_STOP_EVENT is not None and _WORKER_STOP_EVENT.is_set():
                message = "ABORTED_BY_USER"
                break

            cache_dict = build_cache_dict_for_scenario(scenario)

            # ✅ npz 저장 (scenario_id, neighbor_track_token 제외)
            save_cache_dict_to_npz(
                cache_dict=cache_dict,
                out_npz_path=out_npz_path,
                excluded_keys=EXCLUDE_KEYS_FOR_NPZ,
                compress=True,
            )

            if processed == 0:
                _log(f"FIRST npz written: {out_npz_path}")

            if split == "validation" and out_tfrecord_path is not None:
                with tf.io.TFRecordWriter(out_tfrecord_path.as_posix()) as w:
                    w.write(record_bytes)

            now = time.monotonic()
            if now - last_hb >= hb_sec:
                elapsed = time.perf_counter() - t0
                _log(
                    f"PROGRESS file={tfrecord_p.name} rec={k+1} "
                    f"processed={processed} skipped={skipped} failed={failed} "
                    f"elapsed={elapsed:.1f}s"
                )
                last_hb = now

            processed += 1

            if save_image:
                save_dir = os.path.join(out_split_dir, "debug_vis")
                save_path = os.path.join(save_dir, f"{scenario_id}.png")
                os.makedirs(save_dir, exist_ok=True)
                cache_dict["token_to_future_traj_wrt_ego"] = None
                draw_machine.draw_world_model_to_png(
                    cache_dict,
                    output_data={},
                    save_path=save_path
                )

        except Exception as e:
            failed += 1
            _log(
                f"[FAIL] scenario_id={scenario_id if 'scenario_id' in locals() else 'unknown'} err={repr(e)}"
            )
            _log(traceback.format_exc())
            message = f"FAILED: {repr(e)}"

    return tfrecord_path, processed, skipped, failed, message


def _worker_init(stop_event, logger_pid_val, args) -> None:
    global _WORKER_STOP_EVENT, _LOG_ENABLED, _LOGGER_PID_VAL
    _WORKER_STOP_EVENT = stop_event
    _LOGGER_PID_VAL = logger_pid_val
    _LOG_ENABLED = False

    # ✅ args에서 길이 설정 주입
    _set_womd_lengths_from_args(args)

    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    faulthandler.enable()


# =========================
# split 전체 캐싱
# =========================
def cache_all_splits(
    args,
    data_path: str,
    num_workers: int,
    overwrite: bool,
    splits: Tuple[str, ...] = SPLITS,
) -> None:
    dataset_p = Path(data_path)
    scenario_dir = dataset_p / "scenario"
    caching_dir = dataset_p / "cache" / args.save_folder
    ensure_dir(caching_dir)

    ctx = mp.get_context("spawn")
    stop_event = ctx.Event()

    # 메인 시그널 핸들링(1회: graceful, 2회: hard kill)
    pool_ref: Dict[str, Optional[mppool.Pool]] = {"pool": None}
    old_handlers, shutdown_state = _install_main_signal_handlers(
        stop_event=stop_event,
        pool_ref=pool_ref,
        grace_sec=12.0,  # 워커가 stop_event 보고 빠져나올 시간(필요하면 늘리기)
    )

    try:
        for split in splits:
            split_input_dir = scenario_dir / split
            split_output_dir = caching_dir / split
            ensure_dir(split_output_dir)

            if split == "validation":
                ensure_dir(caching_dir / "validation_tfrecords_splitted")

            tfrecord_files: List[Path] = list_tfrecord_files(split_input_dir)
            if len(tfrecord_files) == 0:
                print(f"[WARN] No files found: {split_input_dir}")
                continue

            process_one_tfrecord_file_fn = partial(
                process_one_tfrecord_file,
                split=split,
                caching_dir=caching_dir.as_posix(),
                overwrite=overwrite,
                save_image=bool(args.save_image),
            )
            pool: Optional[mppool.Pool] = None
            results: List[Tuple[str, int, int, int, str]] = []
            tfrecord_paths = [p.as_posix() for p in tfrecord_files]

            try:
                logger_pid_val = ctx.Value('i', 0)  # 0이면 아직 아무도 로거 선점 안 함
                pool = ctx.Pool(
                    processes=num_workers,
                    initializer=_worker_init,
                    initargs=(stop_event, logger_pid_val, args),
                )
                pool_ref["pool"] = pool
                try:
                    msg = pool.apply_async(_ping).get(timeout=10)
                    _log(f"WORKER READY: {msg}")
                except Exception as e:
                    _log(
                        f"WORKER NOT READY (likely importing TF/torch/etc): {repr(e)}"
                    )

                iterator = pool.imap_unordered(
                    process_one_tfrecord_file_fn,
                    tfrecord_paths,
                    chunksize=1,
                )

                # 무기한 block 대신 timeout polling (Ctrl+C 반응성 확보)
                pbar = tqdm(total=len(tfrecord_paths), desc=f"cache-{split}")
                try:
                    while True:
                        if stop_event.is_set():
                            deadline = shutdown_state.get("deadline", None)
                            if deadline is not None and time.monotonic(
                            ) > float(deadline):
                                _log("grace 기간 초과 -> 워커 강제 종료(SIGKILL 포함)합니다.")
                                _terminate_pool_hard(pool, timeout_sec=1.0)
                                raise SystemExit(130)

                        try:
                            res = iterator.next(timeout=0.5)
                        except mp.TimeoutError:
                            continue
                        except StopIteration:
                            break

                        results.append(res)
                        pbar.update(1)
                finally:
                    pbar.close()

                # 정리
                with contextlib.suppress(Exception):
                    pool.close()

                if not _join_pool_soft(pool, timeout_sec=8.0):
                    _terminate_pool_hard(pool, timeout_sec=1.0)

                pool_ref["pool"] = None

                # Ctrl+C로 stop_event가 켜졌다면 여기서 종료
                if stop_event.is_set():
                    raise SystemExit(130)

            except KeyboardInterrupt:
                # 혹시라도 KeyboardInterrupt로 들어오는 환경 대비
                stop_event.set()
                _log("KeyboardInterrupt -> 워커 종료 후 종료합니다.")
                _terminate_pool_hard(pool, timeout_sec=1.0)
                raise SystemExit(130)

            except Exception:
                stop_event.set()
                _terminate_pool_hard(pool, timeout_sec=1.0)
                raise

            # 실패 요약 출력 (기존 로직 유지)
            num_failed_files = sum(
                1 for _, _, _, failed, _ in results if failed > 0)
            if num_failed_files > 0:
                print(
                    f"[WARN] split={split} failed_files={num_failed_files}/{len(results)}"
                )
                for tfp, processed, skipped, failed, msg in results:
                    if failed > 0:
                        print(
                            f"  - {tfp} | processed={processed} skipped={skipped} failed={failed} | {msg}"
                        )

    finally:
        # 시그널 핸들러 원복
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, old_handlers[signal.SIGINT])
            signal.signal(signal.SIGTERM, old_handlers[signal.SIGTERM])


# =========================
# CLI
# =========================
def _str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "t", "yes", "y")


if __name__ == "__main__":
    args = args_util.get_args()
    _require_filter_radius_args(args)

    splits = tuple(
        [s.strip() for s in args.womd_splits.split(",") if len(s.strip()) > 0])
    cache_all_splits(
        args,
        data_path=args.womd_data_path,
        num_workers=int(args.num_workers),
        overwrite=_str2bool(args.overwrite_womd_cache),
        splits=splits,
    )