from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.collections import LineCollection, PolyCollection

# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib 성능 설정 (아주 자세한 주석)
# ─────────────────────────────────────────────────────────────────────────────
# 1) path.simplify
#    - "경로 단순화" 기능 ON. 수많은 꼭짓점 중에서 육안으로 거의 차이가 없는
#      중간 점들을 자동으로 생략합니다.
#    - 화면 해상도에서 구별하기 어려운 zig-zag, 미세 진동을 쳐내기 때문에
#      라인/패치가 매우 많을 때 렌더링 속도가 크게 빨라집니다.
mpl.rcParams['path.simplify'] = True

# 2) path.simplify_threshold
#    - 경로 단순화의 "공격성(강도)". 0.0(보수적) ~ 1.0(공격적).
#    - 0.5는 적당히 단순화: 세세한 점을 과감히 줄여 그리기량을 감소시킵니다.
#    - 숫자가 클수록 더 많은 중간 점을 생략하므로 빠르지만, 너무 크면 라인이 너무
#      단순해질 수 있습니다.
mpl.rcParams['path.simplify_threshold'] = 0.5

# 3) agg.path.chunksize
#    - "아주 긴 경로"를 내부적으로 몇 개의 덩어리로 쪼개서 그립니다.
#    - 메모리/CPU의 부담을 줄여 긴 폴리라인/폴리곤을 그릴 때 멈칫거림(hiccup)을 완화.
#    - 20000이면, 꼭짓점 2만 개 단위로 나눠 처리합니다.
mpl.rcParams['agg.path.chunksize'] = 20000

# 4) lines.antialiased / patch.antialiased
#    - 안티앨리어싱(가장자리 부드럽게)을 꺼서 CPU 부담을 줄입니다.
#    - 살짝 거칠게 보일 수 있지만, 객체 수가 매우 많을 때 속도 향상 효과가 큽니다.
mpl.rcParams['lines.antialiased'] = False
mpl.rcParams['patch.antialiased'] = False


from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import to_rgba

Array = np.ndarray
ND_f32 = npt.NDArray[np.float32]
ND_f64 = npt.NDArray[np.float64]


def _valid_mask_agent_rows(rows11: ND_f32, *, invalid_eps: float) -> ND_f32:
    """(N,11)에서 앞 8차원 절대값 중 하나라도 eps보다 크면 유효(True).

    Args:
        rows11 (ND_f32): shape (N, 11).
        invalid_eps (float): 유효성 판정 epsilon.

    Returns:
        ND_f32: shape (N,), bool mask.
    """
    return np.any(np.abs(rows11[:, :8]) > invalid_eps, axis=1)


def _make_rotation_mats(cos_yaw: ND_f32, sin_yaw: ND_f32) -> ND_f32:
    """cos/sin으로 (N,2,2) 회전행렬 생성."""
    n = cos_yaw.shape[0]
    rot = np.empty((n, 2, 2), dtype=np.float32)
    rot[:, 0, 0] = cos_yaw
    rot[:, 0, 1] = -sin_yaw
    rot[:, 1, 0] = sin_yaw
    rot[:, 1, 1] = cos_yaw
    return rot


def _batch_oriented_box_corners(
    x: ND_f32, y: ND_f32, cos_yaw: ND_f32, sin_yaw: ND_f32, length: ND_f32, width: ND_f32
) -> ND_f32:
    """여러 사각형의 꼭짓점 좌표를 벡터화로 생성.

    Args:
        x, y, cos_yaw, sin_yaw, length, width: shape (N,)

    Returns:
        ND_f32: shape (N, 4, 2)  (각 사각형 꼭짓점 시계방향)
    """
    n = x.shape[0]
    half_L = length * 0.5
    half_W = width * 0.5
    # 로컬 코너 (N,4,2)
    corners_local = np.stack(
        [
            np.stack([+half_L, +half_W], axis=1),
            np.stack([+half_L, -half_W], axis=1),
            np.stack([-half_L, -half_W], axis=1),
            np.stack([-half_L, +half_W], axis=1),
        ],
        axis=1,
    ).astype(np.float32)

    rot = _make_rotation_mats(cos_yaw, sin_yaw)              # (N,2,2)
    rotated = corners_local @ np.swapaxes(rot, 1, 2)         # (N,4,2)
    translated = rotated + np.stack([x, y], axis=1)[:, None, :]
    return translated


def apply_rasterization(ax: plt.Axes, *, rasterization_zorder: int = 10) -> None:
    """Axes에 래스터화 기준 zorder를 설정합니다.

    zorder가 지정값 이상인 아티스트는 벡터 대신 비트맵으로 저장되어(특히 PDF/EPS),
    파일 크기와 렌더 시간이 줄어듭니다.

    Args:
        ax (plt.Axes): 대상 축.
        rasterization_zorder (int): 이 값 이상인 아티스트를 래스터화합니다.
    """
    ax.set_rasterization_zorder(rasterization_zorder)


def apply_agg_speed_toggles(ax: plt.Axes, *, rasterization_zorder: int = 10) -> None:
    """Axes에 래스터화 기준 zorder를 설정합니다.

    그림 속 **zorder가 지정값 이상인 아티스트**는 **벡터가 아닌 비트맵**으로
    저장되므로(특히 PDF/EPS), 파일 크기와 렌더 시간이 줄어듭니다.

    Args:
        ax (plt.Axes): 대상 Axes.
        rasterization_zorder (int): 이 값 이상인 아티스트를 래스터화(bitmap)합니다.
            예: 10으로 두면 zorder >= 10 에 대해 래스터화.
    """
    # 예: 작은 선/점이 수만 개면 벡터는 무겁습니다 → 비트맵으로 내려 CPU/파일 크기 절약
    ax.set_rasterization_zorder(rasterization_zorder)


# ─────────────────────────────────────────────────────────────────────────────
# 타입 별칭
# ─────────────────────────────────────────────────────────────────────────────
ND_f32 = npt.NDArray[np.float32]
ND_f64 = npt.NDArray[np.float64]
ColorArray = Union[str, ND_f32, ND_f64]  # 단색 또는 (N,3/4) 색상 배열


# ─────────────────────────────────────────────────────────────────────────────
# 기본 벡터화 유틸
# ─────────────────────────────────────────────────────────────────────────────
def boxes_to_polycollection(
    centers_x: ND_f32,
    centers_y: ND_f32,
    cos_yaws: ND_f32,
    sin_yaws: ND_f32,
    lengths: ND_f32,
    widths: ND_f32,
    *,
    edgecolor: str,
    linewidth: float,
    fillcolor: Optional[str] = None,
    fillalpha: Optional[float] = None,
    zorder: int = 5,
    rasterize: bool = True,
) -> PolyCollection:
    """사각형 박스 N개를 한 번에 그리는 PolyCollection을 생성합니다.

    배열 shape
    ----------
    - centers_x, centers_y : (N,)
    - cos_yaws, sin_yaws   : (N,)
    - lengths, widths      : (N,)
    - 반환 PolyCollection 의 각 아이템 꼭짓점: (4, 2)

    Args:
        centers_x (ND_f32): 중심 x 좌표 배열. shape (N,)
        centers_y (ND_f32): 중심 y 좌표 배열. shape (N,)
        cos_yaws (ND_f32):   cos(yaw) 배열.      shape (N,)
        sin_yaws (ND_f32):   sin(yaw) 배열.      shape (N,)
        lengths (ND_f32):    길이(차량 길이) 배열. shape (N,)
        widths (ND_f32):     폭(차량 폭) 배열.    shape (N,)
        edgecolor (str):     테두리 색상.
        linewidth (float):   테두리 선 두께.
        fillcolor (Optional[str]): 내부 채움 색상. None이면 내부 투명.
        fillalpha (Optional[float]): 내부 채움 알파. None이면 내부 투명.
        zorder (int):        그리기 순서.
        rasterize (bool):    래스터화(bitmap) 여부.

    Returns:
        PolyCollection: N개의 사각형을 담은 PolyCollection.
    """
    centers_x = np.asarray(centers_x, dtype=np.float32)
    centers_y = np.asarray(centers_y, dtype=np.float32)
    cos_yaws = np.asarray(cos_yaws, dtype=np.float32)
    sin_yaws = np.asarray(sin_yaws, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)

    # (N, 1) 스케일 벡터
    half_L = (0.5 * lengths)[:, None]  # (N,1)
    half_W = (0.5 * widths)[:, None]   # (N,1)

    # 로컬 단위 사각형(정방형) 템플릿 (4,2) → (N, 4, 2)로 브로드캐스트
    corners_template: ND_f32 = np.array(
        [[+1, +1], [+1, -1], [-1, -1], [-1, +1]], dtype=np.float32
    )
    local_corners: ND_f32 = np.empty((lengths.shape[0], 4, 2), dtype=np.float32)
    # x축은 길이, y축은 폭 방향으로 스케일
    local_corners[:, :, 0] = corners_template[:, 0] * half_L  # (N,4)
    local_corners[:, :, 1] = corners_template[:, 1] * half_W  # (N,4)

    # 회전 행렬 R = [[c,-s],[s,c]]  (N,2,2)
    R: ND_f32 = np.stack(
        [
            np.stack([cos_yaws, -sin_yaws], axis=1),
            np.stack([sin_yaws,  cos_yaws], axis=1),
        ],
        axis=1,
    )
    # 회전 + 평행이동 → 전역 꼭짓점 (N,4,2)
    rotated: ND_f32 = local_corners @ np.transpose(R, (0, 2, 1))
    centers: ND_f32 = np.stack([centers_x, centers_y], axis=1)[:, None, :]  # (N,1,2)
    corners_global: ND_f32 = rotated + centers  # (N,4,2)

    # 내부 채우기 색
    face = 'none' if (fillcolor is None or fillalpha is None) else to_rgba(fillcolor, fillalpha)

    pc = PolyCollection(
        corners_global,
        closed=True,
        edgecolors=edgecolor,
        facecolors=face,
        linewidths=linewidth,
        antialiased=False,
    )
    pc.set_zorder(zorder)
    if rasterize:
        pc.set_rasterized(True)
    return pc


def segments_to_linecollection(
    p0: ND_f32,
    p1: ND_f32,
    *,
    color: ColorArray,
    linewidth: float,
    linestyle: Optional[Tuple[int, Tuple[int, int]]] = None,
    zorder: int = 5,
    rasterize: bool = True,
) -> LineCollection:
    """선분 N개를 한 번에 그리는 LineCollection을 생성합니다.

    배열 shape
    ----------
    - p0, p1: (N, 2)
    - 결과 내부 세그먼트: (N, 2, 2)

    Args:
        p0 (ND_f32): 각 선분의 시작점 (x,y). shape (N,2)
        p1 (ND_f32): 각 선분의 끝점 (x,y).   shape (N,2)
        color (ColorArray): 단색 문자열 또는 (N,3|4) RGBA 배열.
        linewidth (float): 선 두께.
        linestyle (Optional[Tuple[int, Tuple[int,int]]]): matplotlib 커스텀 dash 스타일.
            예) 점선: (0, (4, 4))
        zorder (int): 그리기 순서.
        rasterize (bool): 래스터화 여부.

    Returns:
        LineCollection: N개의 선분을 담은 LineCollection.
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    segments: ND_f32 = np.stack([p0, p1], axis=1)  # (N,2,2)

    lc = LineCollection(segments, colors=color, linewidths=linewidth, antialiased=False)
    if linestyle is not None:
        lc.set_linestyle(linestyle)
    lc.set_zorder(zorder)
    if rasterize:
        lc.set_rasterized(True)
    return lc


def heading_lines_linecollection(
    centers_x: ND_f32,
    centers_y: ND_f32,
    cos_yaws: ND_f32,
    sin_yaws: ND_f32,
    lengths: ND_f32,
    *,
    scale: float,
    color: str,
    linewidth: float,
    zorder: int = 6,
) -> LineCollection:
    """박스의 '헤딩 방향' 선을 한 번에 생성합니다.

    배열 shape
    ----------
    - centers_x, centers_y, cos_yaws, sin_yaws, lengths: (N,)

    Args:
        centers_x (ND_f32): 중심 x 좌표. shape (N,)
        centers_y (ND_f32): 중심 y 좌표. shape (N,)
        cos_yaws  (ND_f32): cos(yaw). shape (N,)
        sin_yaws  (ND_f32): sin(yaw). shape (N,)
        lengths   (ND_f32): 박스 길이. shape (N,)
        scale (float): 헤딩 선 길이 스케일(보통 길이 * scale).
        color (str): 선 색상.
        linewidth (float): 선 두께.
        zorder (int): 그리기 순서.

    Returns:
        LineCollection: 헤딩 선 모음.
    """
    centers_x = np.asarray(centers_x, dtype=np.float32)
    centers_y = np.asarray(centers_y, dtype=np.float32)
    cos_yaws  = np.asarray(cos_yaws,  dtype=np.float32)
    sin_yaws  = np.asarray(sin_yaws,  dtype=np.float32)
    lengths   = np.asarray(lengths,   dtype=np.float32)

    head_len: ND_f32 = (lengths * scale).astype(np.float32)  # (N,)
    head_x: ND_f32 = centers_x + head_len * cos_yaws
    head_y: ND_f32 = centers_y + head_len * sin_yaws

    p0: ND_f32 = np.stack([centers_x, centers_y], axis=1)
    p1: ND_f32 = np.stack([head_x,    head_y],    axis=1)
    return segments_to_linecollection(p0, p1, color=color, linewidth=linewidth, zorder=zorder, rasterize=True)


def quiver_fixed_length(
    ax: plt.Axes,
    origins_x: ND_f32,
    origins_y: ND_f32,
    vec_x: ND_f32,
    vec_y: ND_f32,
    *,
    length_m: float,
    color: str,
    width: float = 0.001,          # ⬅ 샤프트 두께도 살짝 줄임
    headlength: float = 3.0,       # ⬅ 머리 길이(포인트)
    headwidth: float = 2.0,        # ⬅ 머리 너비(포인트)
    headaxislength: float = 2.0,   # ⬅ 머리 축 길이(포인트)
    pivot: str = "tail",           # ⬅ 화살표 pivot (tail/mid)
    zorder: int = 15,
) -> None:
    """벡터(방향/속도)를 길이 고정 화살표로 한 번에 그립니다."""
    origins_x = np.asarray(origins_x, dtype=np.float32)
    origins_y = np.asarray(origins_y, dtype=np.float32)
    vec_x     = np.asarray(vec_x,     dtype=np.float32)
    vec_y     = np.asarray(vec_y,     dtype=np.float32)

    magnitude: ND_f32 = np.hypot(vec_x, vec_y)
    mask: ND_f32 = magnitude > 1e-6  # 영벡터 제외
    if not np.any(mask):
        return

    # 방향만 유지하고 크기는 length_m로 정규화
    u = length_m * (vec_x[mask] / magnitude[mask])
    v = length_m * (vec_y[mask] / magnitude[mask])

    ax.quiver(
        origins_x[mask],
        origins_y[mask],
        u,
        v,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=color,
        width=width,
        headlength=headlength,
        headwidth=headwidth,
        headaxislength=headaxislength,
        pivot=pivot,
        zorder=zorder,
    )



# ─────────────────────────────────────────────────────────────────────────────
# 빠른 드로잉 함수(기존 반복 plot/patch → Collection/Quiver/Scatter)
# ─────────────────────────────────────────────────────────────────────────────


def draw_lane_centerlines_fast(
    ax: plt.Axes,
    lanes: ND_f32,
    *,
    signal_colors: Dict[int, str],
    draw_agent_route_lane_order: bool,
    agent_route_lane_order: Optional[npt.NDArray[np.int64]] = None,  # (max_agent_num, lane_num)
    draw_token_int_list: Optional[List[int]] = None,
    label_stride: int = 4,
    route_label_color: str = "#00FFFF",  # CYAN
    route_label_fontsize: int = 7,
    route_label_vstep_m: float = 0.3,
    dashed_linewidth: float = 1.2,
    dashed_pattern: Tuple[int, Tuple[int, int]] = (0, (4, 4)),
    invalid_eps: float = 0.0,
) -> None:
    """센터라인을 빠르게 렌더링하되, **원래 역할과 모드**를 그대로 지킵니다.

    동작
    ----
    1) draw_agent_route_lane_order=False 또는 agent_route_lane_order=None 인 경우:
       - **센터라인 점선**을 상태별 색상으로 **LineCollection** 한 번에 그립니다(빠름).

    2) draw_agent_route_lane_order=True 이고 agent_route_lane_order가 주어진 경우:
       - **선을 그리지 않고**, 각 차선의 유효 포인트 위치에
         에이전트‑차선 랭크 텍스트를 표기합니다(원래 동작 유지).
       - 텍스트는 Matplotlib에서 배치 최적화가 어려워 per‑item이지만,
         위치 계산은 벡터화하고 `label_stride`로 샘플링하여 부담을 줄입니다.

    배열 shape
    ----------
    lanes : (lane_num, lane_len, 12) float32
        · [:, :, 0:2]  : center xy
        · [:, :, 2:4]  : center diff (미사용)
        · [:, :, 4:6]  : left offset (dx, dy)
        · [:, :, 6:8]  : right offset (dx, dy)
        · [:, :, 8:12] : signal one-hot [green, yellow, red, unknown]
    agent_route_lane_order : (max_agent_num, lane_num) int
        · 값 >=0: 해당 agent가 이 lane을 경로에 포함, 값은 "가까운 순서 랭크"
        · 값 <0: 경로에 포함되지 않음

    Args:
        ax (plt.Axes): 대상 축.
        lanes (ND_f32): 차선 배열. shape (L, P, 12)
        signal_colors (Dict[int, str]): 신호 상태별 색상 매핑.
        draw_agent_route_lane_order (bool): True면 텍스트 모드, False면 점선 모드.
        agent_route_lane_order (Optional[np.ndarray]): 에이전트‑차선 랭크 배열. shape (A, L)
        draw_token_int_list (Optional[List[int]]): 텍스트 표기할 agent 인덱스 제한(None이면 전체).
        label_stride (int): 텍스트를 찍을 포인트 간격(기존 코드의 `if point_idx % 4 != 0` 유지).
        route_label_color (str): 텍스트 색.
        route_label_fontsize (int): 텍스트 크기.
        route_label_vstep_m (float): 같은 위치에 여러 텍스트가 겹칠 때 세로로 띄울 간격[m].
        dashed_linewidth (float): 점선 두께.
        dashed_pattern (Tuple[int, Tuple[int, int]]): 점선 패턴.
        invalid_eps (float): 유효성 판정 epsilon.
    """
    if lanes is None or lanes.size == 0:
        return

    lane_num: int = int(lanes.shape[0])
    lane_len: int = int(lanes.shape[1])

    # ─────────────────────────────────────────────────────────────────────────
    # (A) 점선 모드: centerline을 상태별 색상으로 LineCollection 한 번에 렌더
    # ─────────────────────────────────────────────────────────────────────────
    if (not draw_agent_route_lane_order) or (agent_route_lane_order is None):
        all_segments: List[ND_f32] = []
        all_colors: List[str] = []

        for lane_idx in range(lane_num):
            lane_i = lanes[lane_idx]                            # (P, 12)
            center_xy: ND_f32 = lane_i[:, 0:2]                  # (P, 2)
            signals:   ND_f32 = lane_i[:, 8:12]                 # (P, 4)
            valid_mask: ND_f32 = np.any(np.abs(lane_i[:, :8]) > invalid_eps, axis=1)  # (P,)

            if lane_len < 2:
                continue

            state_idx: npt.NDArray[np.int64] = np.argmax(signals, axis=1)  # (P,)
            # 양 끝점이 모두 유효한 구간만 채택
            idx: npt.NDArray[np.int64] = np.where(valid_mask[:-1] & valid_mask[1:])[0]
            if idx.size == 0:
                continue

            c0 = center_xy[idx]       # (K, 2)
            c1 = center_xy[idx + 1]   # (K, 2)
            seg = np.stack([c0, c1], axis=1)  # (K, 2, 2)
            all_segments.append(seg)

            # 세그먼트별 색상(신호 상태에 맞춰 색 결정)
            for k in idx:
                all_colors.append(signal_colors.get(int(state_idx[k]), "#808080"))

        if all_segments:
            segs = np.concatenate(all_segments, axis=0)  # (N, 2, 2)
            lc = LineCollection(segs, colors=all_colors, linewidths=dashed_linewidth, antialiased=False)
            lc.set_linestyle(dashed_pattern)  # 점선
            lc.set_zorder(2)
            lc.set_rasterized(True)           # 래스터화로 저장 성능/용량 개선
            ax.add_collection(lc)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # (B) 텍스트 모드: 점선은 그리지 않고, 랭크 텍스트만 렌더(원래 동작 유지)
    # ─────────────────────────────────────────────────────────────────────────
    # draw_token_int_list가 None이면 모든 agent 대상으로 표기
    draw_all_agents: bool = draw_token_int_list is None

    # 라벨링할 포인트 인덱스(샘플링 stride 적용: 원본의 point_idx % 4 == 0)
    sample_indices: npt.NDArray[np.int64] = np.arange(0, lane_len, max(1, int(label_stride)), dtype=np.int64)

    for lane_idx in range(lane_num):
        lane_j = lanes[lane_idx]                             # (P, 12)
        lane_j_center_xy: ND_f32 = lane_j[:, 0:2]            # (P, 2)
        valid_mask: ND_f32 = np.any(np.abs(lane_j[:, :8]) > invalid_eps, axis=1)  # (P,)

        # 이 차선을 자신의 경로에 포함하는 모든 agent와 rank
        ranks_j: npt.NDArray[np.int64] = agent_route_lane_order[:, lane_idx]  # (A,)
        valid_agent_indices: npt.NDArray[np.int64] = np.nonzero(ranks_j >= 0)[0]  # (K,)

        if valid_agent_indices.size == 0:
            continue

        # 유효 포인트들만 샘플링된 인덱스로 필터
        for point_idx in sample_indices:
            if point_idx >= lane_len:
                break
            if not valid_mask[point_idx]:
                continue

            point_x: float = float(lane_j_center_xy[point_idx, 0])
            point_y: float = float(lane_j_center_xy[point_idx, 1])

            # 여러 agent 텍스트가 한 위치에 겹치지 않도록 세로로 살짝씩 적층
            stack_count: int = 0
            for agent_idx in valid_agent_indices:
                if (not draw_all_agents) and (int(agent_idx) not in draw_token_int_list):  # 선택된 agent만
                    continue
                rank_ij: int = int(ranks_j[int(agent_idx)])

                # label = f"{agent_idx}:{rank_ij}"  # (원본 코드에서 rank만 표기하므로 주석)
                label: str = f"{rank_ij}"

                # Matplotlib는 텍스트를 batch로 묶을 수 없어 per-item 렌더가 불가피합니다.
                # 다만 위치계산은 위에서 벡터화/stride로 줄였고, zorder/안티앨리어싱 설정은 전체 성능에 도움이 됩니다.
                ax.text(
                    point_x,
                    point_y + route_label_vstep_m * stack_count,  # 위로 적층
                    label,
                    color=route_label_color,
                    fontsize=route_label_fontsize,
                    ha="center",
                    va="bottom",
                    zorder=3,
                )
                stack_count += 1


def draw_lane_boundaries_fast(
    ax: plt.Axes,
    lanes: ND_f32,
    *,
    boundary_color: str,
    linewidth: float = 1.0,
    invalid_eps: float = 0.0,
) -> None:
    """좌/우 차선 경계를 LineCollection으로 한 번에 그립니다. (역할 동일, 성능↑)

    배열 shape
    ----------
    lanes : (lane_num, lane_len, 12)
        · [:, :, 0:2] : center xy
        · [:, :, 4:6] : left offset (dx, dy)
        · [:, :, 6:8] : right offset (dx, dy)

    Args:
        ax (plt.Axes): 대상 축.
        lanes (ND_f32): 차선 배열. shape (L, P, 12)
        boundary_color (str): 선 색.
        linewidth (float): 선 두께.
        invalid_eps (float): 유효성 판정 epsilon.
    """
    if lanes is None or lanes.size == 0:
        return

    left_segments: List[ND_f32] = []
    right_segments: List[ND_f32] = []

    for lane_i in lanes:
        center_xy: ND_f32 = lane_i[:, 0:2]  # (P,2)
        left_vec:  ND_f32 = lane_i[:, 4:6]  # (P,2)
        right_vec: ND_f32 = lane_i[:, 6:8]  # (P,2)

        valid_mask: ND_f32 = np.any(np.abs(lane_i[:, :8]) > invalid_eps, axis=1)  # (P,)
        if center_xy.shape[0] < 2:
            continue

        idx: npt.NDArray[np.int64] = np.where(valid_mask[:-1] & valid_mask[1:])[0]
        if idx.size == 0:
            continue

        l0 = center_xy[idx]     + left_vec[idx]
        l1 = center_xy[idx + 1] + left_vec[idx + 1]
        r0 = center_xy[idx]     + right_vec[idx]
        r1 = center_xy[idx + 1] + right_vec[idx + 1]

        left_segments.append(np.stack([l0, l1], axis=1))   # (K,2,2)
        right_segments.append(np.stack([r0, r1], axis=1))  # (K,2,2)

    if left_segments:
        segs = np.concatenate(left_segments, axis=0)
        lc_left = LineCollection(segs, colors=boundary_color, linewidths=linewidth, antialiased=False)
        lc_left.set_zorder(1); lc_left.set_rasterized(True)
        ax.add_collection(lc_left)

    if right_segments:
        segs = np.concatenate(right_segments, axis=0)
        lc_right = LineCollection(segs, colors=boundary_color, linewidths=linewidth, antialiased=False)
        lc_right.set_zorder(1); lc_right.set_rasterized(True)
        ax.add_collection(lc_right)


def draw_boxes_and_headings_fast(
    ax: plt.Axes,
    xycsLw: Tuple[ND_f32, ND_f32, ND_f32, ND_f32, ND_f32, ND_f32],
    *,
    edge_color: str,
    edge_linewidth: float,
    heading_color: str,
    heading_linewidth: float,
    heading_scale_of_length: float,
    fill_color: Optional[str] = None,
    fill_alpha: Optional[float] = None,
    box_zorder: int = 5,
    heading_zorder: int = 6,
) -> None:
    """박스(PolyCollection)와 헤딩선(LineCollection)을 한 번에 그립니다.

    배열 shape
    ----------
    - 각 배열: shape (N,)

    Args:
        ax (plt.Axes): 대상 Axes.
        xycsLw (Tuple[ND_f32,...]): (x, y, cos, sin, length, width) 6개 배열.
        edge_color (str): 박스 테두리 색상.
        edge_linewidth (float): 박스 테두리 두께.
        heading_color (str): 헤딩선 색상.
        heading_linewidth (float): 헤딩선 두께.
        heading_scale_of_length (float): 헤딩선 길이 = length * scale.
        fill_color (Optional[str]): 내부 채움 색상(None이면 투명).
        fill_alpha (Optional[float]): 내부 채움 알파(None이면 투명).
        box_zorder (int): 박스 zorder.
        heading_zorder (int): 헤딩선 zorder.
    """
    x, y, c, s, L, W = xycsLw
    if x.size == 0:
        return

    pc = boxes_to_polycollection(
        x, y, c, s, L, W,
        edgecolor=edge_color,
        linewidth=edge_linewidth,
        fillcolor=fill_color,
        fillalpha=fill_alpha,
        zorder=box_zorder,
        rasterize=True,
    )
    ax.add_collection(pc)

    lc = heading_lines_linecollection(
        x, y, c, s, L,
        scale=heading_scale_of_length,
        color=heading_color,
        linewidth=heading_linewidth,
        zorder=heading_zorder,
    )
    ax.add_collection(lc)


def draw_future_points_scatter_fast(
    ax: plt.Axes,
    xs: ND_f32,
    ys: ND_f32,
    *,
    marker: str,
    size: float,
    color: str,
    zorder: int,
) -> None:
    """미래 포인트 묶음을 한 번에 scatter(PathCollection)로 그립니다.

    배열 shape
    ----------
    - xs, ys: (N,)

    Args:
        ax (plt.Axes): 대상 Axes.
        xs (ND_f32): x 좌표. shape (N,)
        ys (ND_f32): y 좌표. shape (N,)
        marker (str): 마커 형태(예: 'x', 'o').
        size (float): 사이즈(포인트 면적, point^2 단위).
        color (str): 색상.
        zorder (int): 그리기 순서.
    """
    if xs.size == 0:
        return
    ax.scatter(xs, ys, marker=marker, s=size, c=color, zorder=zorder)

def draw_neighbor_past_fast(
    ax: plt.Axes,
    neighbor_agents_past: ND_f32,                    # (max_agent_num, T, 11)
    options,                                          # DrawingOptions
    draw_token_int_list: Optional[List[int]] = None,
) -> None:
    """이웃 과거 시퀀스를 Poly/Line/Quiver로 배치 렌더(역할/표현 동일).

    원본 규칙
    ----------
    - 각 시점 박스/헤딩선을 그림. 현재 스텝(t==T-1)은 채움(fill), 과거는 테두리만.
    - 속도 화살표: `options.NEI_draw_velocity_arrow`가 True이거나 t==현재 스텝이면 그림.
    - 클래스별 색상/두께 유지: options.NEI_neighbor_style 활용.
    - `draw_token_int_list`가 주어지면 해당 agent만 **현재 스텝**만 그림(원본과 동일).

    Args:
        ax (plt.Axes): 축.
        neighbor_agents_past (ND_f32): (A, T, 11)
        options: DrawingOptions.
        draw_token_int_list (Optional[List[int]]): 인덱스 목록(None이면 전체/모든 시점).
    """
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return

    invalid_eps = float(options.invalid_eps)
    max_agent_num, time_len, feat_dim = neighbor_agents_past.shape
    assert feat_dim == 11, "neighbor_agents_past의 마지막 차원은 11이어야 합니다."
    current_t = time_len - 1

    # 누적 버퍼
    polys_past: List[ND_f32] = []      # 과거(테두리만)
    colors_past: List[str] = []
    polys_curr: List[ND_f32] = []      # 현재(채움 + 테두리)
    edge_curr: List[str] = []
    face_curr: List[Tuple[float, float, float, float]] = []

    # 헤딩 라인 (과거/현재 분리)
    lines_past: List[ND_f32] = []      # (N, 2, 2)
    lines_curr: List[ND_f32] = []
    line_colors_past: List[str] = []
    line_colors_curr: List[str] = []

    # 속도 화살표(quiver): 클래스별로 모아서 한번에 그린다.
    quiver_by_class = {
        "vehicles": {"x": [], "y": [], "u": [], "v": [], "z": []},
        "pedestrians": {"x": [], "y": [], "u": [], "v": [], "z": []},
        "bicycles": {"x": [], "y": [], "u": [], "v": [], "z": []},
    }

    # agent 루프 (그림 수를 줄이기 위해 데이터만 수집)
    for agent_idx in range(max_agent_num):
        track = neighbor_agents_past[agent_idx]                  # (T,11)
        # draw_all_time = True이면 모든 시점, False면 현재 시점만(원본과 동일)
        draw_all_time = not (draw_token_int_list and agent_idx not in draw_token_int_list)

        # 유효 마스크
        valid_mask = _valid_mask_agent_rows(track, invalid_eps=invalid_eps)  # (T,)
        if not np.any(valid_mask):
            continue

        # 시점 선택
        if draw_all_time:
            time_indices = np.nonzero(valid_mask)[0]
        else:
            time_indices = np.array([current_t], dtype=np.int64) if valid_mask[current_t] else np.array([], dtype=np.int64)
        if time_indices.size == 0:
            continue

        # 행 데이터 추출
        rows = track[time_indices]                                # (K,11)
        x, y = rows[:, 0].astype(np.float32), rows[:, 1].astype(np.float32)
        c, s = rows[:, 2].astype(np.float32), rows[:, 3].astype(np.float32)
        vx, vy = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
        W, L = rows[:, 6].astype(np.float32), rows[:, 7].astype(np.float32)
        onehot = rows[:, 8:11]                                    # (K,3)
        cls_idx = np.argmax(onehot, axis=1)                       # (K,)
        cls_list = np.take(["vehicles", "pedestrians", "bicycles"], cls_idx)

        # 코너/헤딩
        corners = _batch_oriented_box_corners(x, y, c, s, L, W)   # (K,4,2)
        hx = x + (L * options.COMMON_heading_line_scale) * c
        hy = y + (L * options.COMMON_heading_line_scale) * s
        headings = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)  # (K,2,2)

        # 현재/과거 분리
        is_current = (time_indices == current_t)
        for idx_local, cls_name in enumerate(cls_list):
            style = options.NEI_neighbor_style[cls_name]
            edge_col = style["line_color"]
            face_col = to_rgba(style["fill_color"], style["fill_alpha"])

            if is_current[idx_local]:
                polys_curr.append(corners[idx_local])
                edge_curr.append(edge_col)
                face_curr.append(face_col)
                lines_curr.append(headings[idx_local])
                line_colors_curr.append(edge_col)
            else:
                polys_past.append(corners[idx_local])
                colors_past.append(edge_col)
                lines_past.append(headings[idx_local])
                line_colors_past.append(edge_col)

            # 속도 화살표: 옵션이 True이거나, 현재 스텝이면 그림(원본과 동일)
            if options.NEI_draw_velocity_arrow and is_current[idx_local]:
                quiver_by_class[cls_name]["x"].append(float(x[idx_local]))
                quiver_by_class[cls_name]["y"].append(float(y[idx_local]))
                quiver_by_class[cls_name]["u"].append(float(vx[idx_local]))
                quiver_by_class[cls_name]["v"].append(float(vy[idx_local]))
                quiver_by_class[cls_name]["z"].append(7 if is_current[idx_local] else 4)

    # ── PolyCollection: 과거(테두리만), 현재(채움)
    if polys_past:
        pc_past = PolyCollection(polys_past, facecolors=(0, 0, 0, 0), edgecolors=colors_past, linewidths=0.2, antialiased=False)
        pc_past.set_zorder(4)
        pc_past.set_rasterized(True)
        ax.add_collection(pc_past)

    if polys_curr:
        pc_curr = PolyCollection(polys_curr, facecolors=face_curr, edgecolors=edge_curr, linewidths=0.2, antialiased=False)
        pc_curr.set_zorder(5)
        pc_curr.set_rasterized(True)
        ax.add_collection(pc_curr)

    # ── LineCollection: 헤딩선
    if lines_past:
        lc_past = LineCollection(lines_past, colors=line_colors_past, linewidths=0.2, antialiased=False)
        lc_past.set_zorder(4)
        lc_past.set_rasterized(True)
        ax.add_collection(lc_past)

    if lines_curr:
        lc_curr = LineCollection(lines_curr, colors=line_colors_curr, linewidths=0.2, antialiased=False)
        lc_curr.set_zorder(6)
        lc_curr.set_rasterized(True)
        ax.add_collection(lc_curr)

    # ── Quiver: 클래스별 한 번씩
    for cls_name, bucket in quiver_by_class.items():
        if not bucket["x"]:
            continue
        style = options.NEI_neighbor_style[cls_name]
        # 고정 길이 화살표가 아니라, 속도 벡터 방향/크기 그대로(원본 add_velocity_arrow와 동일하게 '방향' 표시).
        q = ax.quiver(
            np.array(bucket["x"]),
            np.array(bucket["y"]),
            np.array(bucket["u"]),
            np.array(bucket["v"]),
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=style["velocity_line_color"],
            linewidth=style["velocity_line_width"],
            zorder=7,  # 현재/과거 혼재 → 상한 zorder
        )
        q.set_rasterized(True)



def annotate_neighbor_indices_for_past_fast(
    ax: plt.Axes,
    neighbor_track_token: List[Optional[str]],
    neighbor_agents_past: ND_f32,    # (A,T,11)
    options,
) -> None:
    """현재 스텝 위치 근처에 토큰 텍스트 표기(원본과 동일, 벡터화로 좌표만 수집)."""
    if neighbor_agents_past is None or neighbor_agents_past.size == 0:
        return
    invalid_eps = float(options.invalid_eps)
    max_agent_num, time_len, feat_dim = neighbor_agents_past.shape
    assert feat_dim == 11
    current_t = time_len - 1

    rows = neighbor_agents_past[:, current_t, :]                   # (A,11)
    valid = _valid_mask_agent_rows(rows, invalid_eps=invalid_eps)  # (A,)
    xs = rows[valid, 0]
    ys = rows[valid, 1]
    tokens = [neighbor_track_token[i] for i in np.nonzero(valid)[0]]
    for (x, y), token in zip(zip(xs, ys), tokens):
        ax.text(
            float(x) + options.NEI_past_token_place_offset_m,
            float(y) + options.NEI_past_token_place_offset_m,
            str(token)[:5],
            color=options.NEI_past_token_color,
            fontsize=options.NEI_past_token_fontsize,
            ha="left",
            va="bottom",
            zorder=30,
        )


def draw_ego_past_fast(ax: plt.Axes, ego_agent_past: ND_f32, options) -> None:
    """에고 과거 시퀀스(현재만 채움/과거 테두리) 배치 렌더.

    원본 규칙:
    - 현재 스텝만 채움(fill_color + alpha), 과거는 테두리만.
    - 속도 화살표: `options.EGO_draw_ego_past_vel`이 True이거나 t==현재면 그림.
    """
    if ego_agent_past is None or ego_agent_past.size == 0:
        return
    invalid_eps = float(options.invalid_eps)
    time_len, feat_dim = ego_agent_past.shape
    assert feat_dim == 11
    current_t = time_len - 1

    valid = _valid_mask_agent_rows(ego_agent_past, invalid_eps=invalid_eps)  # (T,)
    if not np.any(valid):
        return
    t_idx = np.nonzero(valid)[0]
    rows = ego_agent_past[t_idx]

    x, y = rows[:, 0].astype(np.float32), rows[:, 1].astype(np.float32)
    c, s = rows[:, 2].astype(np.float32), rows[:, 3].astype(np.float32)
    vx, vy = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
    W, L = rows[:, 6].astype(np.float32), rows[:, 7].astype(np.float32)

    corners = _batch_oriented_box_corners(x, y, c, s, L, W)
    hx = x + (L * options.COMMON_heading_line_scale) * c
    hy = y + (L * options.COMMON_heading_line_scale) * s
    headings = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)

    is_current = (t_idx == current_t)

    # 과거(테두리), 현재(채움)
    if np.any(~is_current):
        pc_past = PolyCollection(
            corners[~is_current], facecolors=(0, 0, 0, 0),
            edgecolors=options.EGO_past_style["line_color"], linewidths=options.EGO_past_style["line_width"],
            antialiased=False
        )
        pc_past.set_zorder(9)
        pc_past.set_rasterized(True)
        ax.add_collection(pc_past)
        lc_past = LineCollection(
            headings[~is_current], colors=options.EGO_past_style["line_color"],
            linewidths=options.EGO_past_style["line_width"], antialiased=False
        )
        lc_past.set_zorder(9)
        lc_past.set_rasterized(True)
        ax.add_collection(lc_past)

    if np.any(is_current):
        face = to_rgba(options.EGO_past_style["fill_color"], options.EGO_past_style["fill_alpha_current"])
        pc_curr = PolyCollection(
            corners[is_current], facecolors=[face]*int(np.sum(is_current)),
            edgecolors=options.EGO_past_style["line_color"], linewidths=options.EGO_past_style["line_width"],
            antialiased=False
        )
        pc_curr.set_zorder(20)
        pc_curr.set_rasterized(True)
        ax.add_collection(pc_curr)
        lc_curr = LineCollection(
            headings[is_current], colors=options.EGO_past_style["line_color"],
            linewidths=options.EGO_past_style["line_width"], antialiased=False
        )
        lc_curr.set_zorder(21)
        lc_curr.set_rasterized(True)
        ax.add_collection(lc_curr)

    # 속도 화살표
    draw_mask = is_current & bool(options.EGO_draw_ego_past_vel)
    if np.any(draw_mask):
        q = ax.quiver(
            x[draw_mask], y[draw_mask], vx[draw_mask], vy[draw_mask],
            angles="xy", scale_units="xy", scale=1.0,
            color=options.EGO_past_style["line_color"],
            linewidth=options.EGO_past_style["line_width"],
            zorder=22,
        )
        q.set_rasterized(True)


def _draw_ego_seq_boxes_lines_quiver(
    ax: plt.Axes,
    seq11: ND_f32,                 # (N,11)
    line_color: str,
    line_width: float,
    draw_velocity: bool,
    velocity_line_color: str,
    velocity_line_width: float,
    velocity_line_alpha: float,
    options,
    z_base: int,
) -> None:
    """에고 시퀀스(테두리만) 박스/헤딩/속도화살표 배치 렌더 공통부."""
    if seq11 is None or seq11.size == 0:
        return
    invalid_eps = float(options.invalid_eps)
    valid = _valid_mask_agent_rows(seq11, invalid_eps=invalid_eps)
    if not np.any(valid):
        return
    rows = seq11[valid]
    x, y = rows[:, 0].astype(np.float32), rows[:, 1].astype(np.float32)
    c, s = rows[:, 2].astype(np.float32), rows[:, 3].astype(np.float32)
    vx, vy = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
    W, L = rows[:, 6].astype(np.float32), rows[:, 7].astype(np.float32)
    corners = _batch_oriented_box_corners(x, y, c, s, L, W)
    hx = x + (L * options.COMMON_heading_line_scale) * c
    hy = y + (L * options.COMMON_heading_line_scale) * s
    headings = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)

    pc = PolyCollection(corners, facecolors=(0, 0, 0, 0), edgecolors=line_color, linewidths=line_width, antialiased=False)
    pc.set_zorder(z_base)
    pc.set_rasterized(True)
    ax.add_collection(pc)

    lc = LineCollection(headings, colors=line_color, linewidths=line_width, antialiased=False)
    lc.set_zorder(z_base)
    lc.set_rasterized(True)
    ax.add_collection(lc)

    if draw_velocity:
        q = ax.quiver(
            x, y, vx, vy,
            angles="xy", scale_units="xy", scale=1.0,
            color=velocity_line_color, linewidth=velocity_line_width,
            alpha=velocity_line_alpha, zorder=z_base
        )
        q.set_rasterized(True)


def draw_ego_agent_next_11_dim_fast(ax: plt.Axes, ego_agent_next_11_dim: ND_f32, options) -> None:
    """에고 **예측(인터폴)** 시퀀스: 박스 테두리/헤딩/옵션 속도."""
    _draw_ego_seq_boxes_lines_quiver(
        ax,
        ego_agent_next_11_dim,
        line_color=options.EGO_next_11_dim_style["line_color"],
        line_width=options.EGO_next_11_dim_style["line_width"],
        draw_velocity=bool(options.EGO_draw_ego_agent_next_11_vel),
        velocity_line_color=options.EGO_next_11_dim_style["velocity_line_color"],
        velocity_line_width=options.EGO_next_11_dim_style["velocity_line_width"],
        velocity_line_alpha=options.EGO_next_11_dim_style["velocity_line_alpha"],
        options=options,
        z_base=25,
    )


def draw_planner_future_11_dim_fast(ax: plt.Axes, planner_future_11_dim: ND_f32, options) -> None:
    """에고 **Planner 미래** 시퀀스: 박스 테두리/헤딩/옵션 속도."""
    _draw_ego_seq_boxes_lines_quiver(
        ax,
        planner_future_11_dim,
        line_color=options.EGO_planner_future_11_style["line_color"],
        line_width=options.EGO_planner_future_11_style["line_width"],
        draw_velocity=bool(options.EGO_draw_planner_velocity),
        velocity_line_color=options.EGO_planner_future_11_style["velocity_line_color"],
        velocity_line_width=options.EGO_planner_future_11_style["velocity_line_width"],
        velocity_line_alpha=options.EGO_planner_future_11_style["velocity_line_alpha"],
        options=options,
        z_base=24,
    )


def draw_ego_future_gt_11_dim_fast(ax: plt.Axes, ego_future_gt_11_dim: ND_f32, options) -> None:
    """에고 **GT 미래(11D)** 시퀀스: 박스 테두리/헤딩/옵션 속도."""
    _draw_ego_seq_boxes_lines_quiver(
        ax,
        ego_future_gt_11_dim,
        line_color=options.EGO_future_gt_11_style["line_color"],
        line_width=options.EGO_future_gt_11_style["line_width"],
        draw_velocity=bool(options.EGO_draw_future_11_velocity),
        velocity_line_color=options.EGO_future_gt_11_style["velocity_line_color"],
        velocity_line_width=options.EGO_future_gt_11_style["velocity_line_width"],
        velocity_line_alpha=options.EGO_future_gt_11_style["velocity_line_alpha"],
        options=options,
        z_base=24,
    )


def draw_diff_future_gen_traj_fast(
    ax: plt.Axes,
    diff_token_to_np_gen_traj_wrt_ego: Optional[Dict[str, ND_f32]],  # value (T,4)
    options,
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """토큰별 미래 포즈를 quiver/PathCollection로 한 번에 렌더(역할 동일).

    - 'arrow' 모드: (x,y)에서 (cos,sin) 방향으로 **고정 길이** 화살표
    - 'point' 모드: (x,y)만 점으로 표시
    - t==0에 토큰 라벨(옵션 활성 시) 표기
    """
    token_to_traj = diff_token_to_np_gen_traj_wrt_ego
    if not token_to_traj:
        return

    invalid_eps = float(options.invalid_eps)
    mode = options.DIFF_future_gen_traj_mode
    arrow_len = float(options.DIFF_future_gen_traj_arrow_len_m)

    xs_all: List[float]; ys_all: List[float]
    xs_all, ys_all = [], []
    us_all: List[float]; vs_all: List[float]
    us_all, vs_all = [], []

    # t==0 라벨 좌표
    first_xy_for_label: List[Tuple[float, float, str]] = []

    for token, traj in token_to_traj.items():
        if draw_token_list is not None and token not in draw_token_list:
            continue
        if traj is None or traj.size == 0:
            continue
        if traj.ndim != 2 or traj.shape[1] != 4:
            raise ValueError("각 value는 (future_len, 4) 여야 합니다.")

        # 유효 포인트
        valid = np.any(np.abs(traj[:, :4]) > invalid_eps, axis=1)
        if not np.any(valid):
            continue
        rows = traj[valid]
        x, y = rows[:, 0], rows[:, 1]
        c, s = rows[:, 2], rows[:, 3]

        xs_all.extend(x.tolist())
        ys_all.extend(y.tolist())

        if mode == "arrow":
            # 원하는 **데이터 단위 고정 길이** 화살표: (U,V)를 길이*방향으로 만들어 scale=1 사용
            us_all.extend((arrow_len * c).tolist())
            vs_all.extend((arrow_len * s).tolist())

        # t==0(원래 시계열 첫 시점) 라벨은, 해당 시점이 valid일 때만
        if options.DIFF_draw_diff_future_gen_traj_token and np.any(valid):
            first_valid_global_idx = int(np.nonzero(valid)[0][0])
            fx, fy = float(traj[first_valid_global_idx, 0]), float(traj[first_valid_global_idx, 1])
            first_xy_for_label.append((fx, fy, str(token)))

    if mode == "arrow" and xs_all:
        q = ax.quiver(
            np.array(xs_all), np.array(ys_all),
            np.array(us_all), np.array(vs_all),
            angles="xy", scale_units="xy", scale=1.0,
            color=options.DIFF_future_gen_traj_style["line_color"],
            linewidth=options.DIFF_future_gen_traj_style["line_width"],
            zorder=23,
        )
        q.set_rasterized(True)
    elif mode == "point" and xs_all:
        sc = ax.scatter(
            np.array(xs_all), np.array(ys_all),
            marker=options.DIFF_future_gen_traj_point_marker,
            s=(options.DIFF_future_gen_traj_point_marker_size ** 2) * 10.0,  # px^2 스케일
            c=options.DIFF_future_gen_traj_style["line_color"],
            zorder=23,
        )
        sc.set_rasterized(True)

    # 라벨
    if options.DIFF_draw_diff_future_gen_traj_token:
        for fx, fy, token in first_xy_for_label:
            ax.text(
                fx, fy - options.DIFF_future_gen_trak_token_text_y_offset_m,
                token,
                color=options.DIFF_future_gen_traj_style["index_color"],
                fontsize=options.LANE_AGENT_index_fontsize,
                ha="center", va="top", zorder=24,
            )


def draw_neighbor_future_gt_3_dim_fast(
    ax: plt.Axes,
    diff_token_to_future_gt_3_dim: Dict[str, ND_f32],  # value (T,3)
    options,
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """(x,y,yaw)들의 (x,y)만 'x' 마커로 PathCollection 1회 렌더."""
    invalid_eps = float(options.invalid_eps)
    xs: List[float]; ys: List[float]
    xs, ys = [], []
    first_labels: List[Tuple[float, float, str]] = []

    for token, fut in diff_token_to_future_gt_3_dim.items():
        if draw_token_list is not None and token not in draw_token_list:
            continue
        if fut is None or fut.size == 0:
            continue
        valid = (np.abs(fut[:, 0]) > invalid_eps) | (np.abs(fut[:, 1]) > invalid_eps)
        if not np.any(valid):
            continue
        rows = fut[valid]
        xs.extend(rows[:, 0].tolist())
        ys.extend(rows[:, 1].tolist())

        if options.DIFF_draw_diff_future_gt_3_dim_token:
            first_idx = int(np.nonzero(valid)[0][0])
            fx, fy = float(fut[first_idx, 0]), float(fut[first_idx, 1])
            first_labels.append((fx, fy, str(token)[:5]))

    if xs:
        sc = ax.scatter(
            np.array(xs), np.array(ys),
            marker='x',
            s=(options.DIFF_future_gt_3_dim_marker_size ** 2) * 10.0,
            c=options.DIFF_future_gt_3_dim_COLOR,
            zorder=26,
        )
        sc.set_rasterized(True)

    for fx, fy, lab in first_labels:
        ax.text(
            fx + options.DIFF_future_gt_3_dim_text_offset_m,
            fy + options.DIFF_future_gt_3_dim_text_offset_m,
            lab,
            color=options.DIFF_future_gt_3_dim_token_color,
            fontsize=options.DIFF_future_gt_3_dim_token_fontsize,
            ha="left", va="bottom", zorder=30
        )


def draw_diff_future_gen_refined_traj_fast(
    ax: plt.Axes,
    diff_token_to_interp_np_traj_wrt_ego: Optional[Dict[str, ND_f32]],  # (N,11)
    options,
    diff_token_to_next_wp_wrt_ego: Optional[Dict[str, ND_f32]],
    draw_token_list: Optional[List[str]] = None,
) -> None:
    """Refined 연속 궤적(박스/헤딩) + 신규 waypoint(박스/헤딩)를 배치 렌더.

    - Refined: 테두리만(빨강), 헤딩선. (옵션) 속도 화살표/속도 숫자.
    - Waypoint: 테두리만(주황), 헤딩선. (옵션) 속도 화살표/속도 숫자.
    """
    invalid_eps = float(options.invalid_eps)

    # ── Refined (모든 토큰 합쳐서 한 번에)
    if diff_token_to_interp_np_traj_wrt_ego:
        rows_all: List[ND_f32] = []
        for token, arr in diff_token_to_interp_np_traj_wrt_ego.items():
            if draw_token_list is not None and token not in draw_token_list:
                continue
            if arr is None or arr.size == 0:
                continue
            if arr.ndim != 2 or arr.shape[1] != 11:
                continue
            rows_all.append(arr)
        if rows_all:
            rows = np.concatenate(rows_all, axis=0)                       # (K,11)
            valid = _valid_mask_agent_rows(rows, invalid_eps=invalid_eps)
            rows = rows[valid]
            if rows.size:
                x, y = rows[:, 0].astype(np.float32), rows[:, 1].astype(np.float32)
                c, s = rows[:, 2].astype(np.float32), rows[:, 3].astype(np.float32)
                vx, vy = rows[:, 4].astype(np.float32), rows[:, 5].astype(np.float32)
                W, L = rows[:, 6].astype(np.float32), rows[:, 7].astype(np.float32)

                corners = _batch_oriented_box_corners(x, y, c, s, L, W)
                hx = x + (L * options.COMMON_heading_line_scale) * c
                hy = y + (L * options.COMMON_heading_line_scale) * s
                headings = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)

                pc = PolyCollection(
                    corners, facecolors=(0, 0, 0, 0),
                    edgecolors=options.DIFF_future_gen_refined_style["line_color"],
                    linewidths=options.DIFF_future_gen_refined_style["line_width"],
                    antialiased=False
                )
                pc.set_zorder(24); pc.set_rasterized(True); ax.add_collection(pc)

                lc = LineCollection(
                    headings, colors=options.DIFF_future_gen_refined_style["line_color"],
                    linewidths=options.DIFF_future_gen_refined_style["line_width"],
                    antialiased=False
                )
                lc.set_zorder(24); lc.set_rasterized(True); ax.add_collection(lc)

                if options.DIFF_draw_future_gen_refined_velocity:
                    q = ax.quiver(
                        x, y, vx, vy,
                        angles="xy", scale_units="xy", scale=1.0,
                        color=options.DIFF_future_gen_refined_style["velocity_line_color"],
                        linewidth=options.DIFF_future_gen_refined_style["velocity_line_width"],
                        alpha=options.DIFF_future_gen_refined_style["velocity_line_alpha"],
                        zorder=24
                    )
                    q.set_rasterized(True)
                    # 속도 숫자(원본 유지: 짝수/홀수 오프셋)
                    for i in range(rows.shape[0]):
                        speed_kmh = float(np.hypot(vx[i], vy[i])) * 3.6
                        offset = 5 if (i % 2 == 0) else 3
                        ax.text(
                            float(x[i]),
                            float(y[i]) + options.DIFF_future_gen_refined_velocity_offset_m * offset,
                            f"{speed_kmh:.1f}",
                            color=options.DIFF_future_gen_refined_style["line_color"],
                            fontsize=options.DIFF_future_gen_refined_velocity_font_size,
                            ha="center", va="bottom", zorder=25, clip_on=True
                        )

    # ── Waypoint (모든 토큰 합쳐서 한 번에)
    if diff_token_to_next_wp_wrt_ego:
        wps: List[ND_f32] = []
        for token, wp in diff_token_to_next_wp_wrt_ego.items():
            if draw_token_list is not None and token not in draw_token_list:
                continue
            if wp is None:
                continue
            arr = np.asarray(wp)
            if arr.ndim == 2:
                arr = arr.squeeze()
            if arr.ndim != 1 or arr.shape[0] != 11:
                continue
            wps.append(arr.astype(np.float32))
        if wps:
            rows = np.stack(wps, axis=0)                                 # (M,11)
            valid = _valid_mask_agent_rows(rows, invalid_eps=invalid_eps)
            rows = rows[valid]
            if rows.size:
                x, y = rows[:, 0], rows[:, 1]
                c, s = rows[:, 2], rows[:, 3]
                vx, vy = rows[:, 4], rows[:, 5]
                W, L = rows[:, 6], rows[:, 7]
                corners = _batch_oriented_box_corners(x, y, c, s, L, W)
                hx = x + (L * options.COMMON_heading_line_scale) * c
                hy = y + (L * options.COMMON_heading_line_scale) * s
                headings = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)

                pc = PolyCollection(
                    corners, facecolors=(0, 0, 0, 0),
                    edgecolors=options.DIFF_new_waypoint_style["line_color"],
                    linewidths=options.DIFF_new_waypoint_style["line_width"],
                    antialiased=False
                )
                pc.set_zorder(28); pc.set_rasterized(True); ax.add_collection(pc)

                lc = LineCollection(
                    headings, colors=options.DIFF_new_waypoint_style["line_color"],
                    linewidths=options.DIFF_new_waypoint_style["line_width"],
                    antialiased=False
                )
                lc.set_zorder(28); lc.set_rasterized(True); ax.add_collection(lc)

                if options.DIFF_draw_future_gen_refined_velocity:
                    q = ax.quiver(
                        x, y, vx, vy,
                        angles="xy", scale_units="xy", scale=1.0,
                        color=options.DIFF_new_waypoint_style["line_color"],
                        linewidth=options.DIFF_new_waypoint_style["line_width"],
                        zorder=28
                    )
                    q.set_rasterized(True)
                    for i in range(rows.shape[0]):
                        speed_kmh = float(np.hypot(vx[i], vy[i])) * 3.6
                        ax.text(
                            float(x[i]),
                            float(y[i]) + options.DIFF_new_waypoint_vel_text_y_offset_m,
                            f"{speed_kmh:.1f}",
                            color=options.DIFF_new_waypoint_style["line_color"],
                            fontsize=options.DIFF_future_gen_refined_velocity_font_size,
                            ha="center", va="bottom", zorder=29, clip_on=True
                        )


def draw_neighbor_past_output_fast(
    ax: plt.Axes,
    current_token_to_np_history: Optional[Dict[str, ND_f32]],  # value (H,11)
    options,
) -> None:
    """토큰별 과거 히스토리(주황) 박스/헤딩/옵션 속도 화살표를 배치 렌더."""
    if not current_token_to_np_history:
        return

    invalid_eps = float(options.invalid_eps)
    line_color = options.NEI_neighbor_past_output_style["line_color"]
    line_width = options.NEI_neighbor_past_output_style["line_width"]

    polys: List[ND_f32] = []
    headings: List[ND_f32] = []
    xs_q: List[float]; ys_q: List[float]; us_q: List[float]; vs_q: List[float]
    xs_q, ys_q, us_q, vs_q = [], [], [], []

    # 텍스트(속도 숫자)는 per‑item
    texts: List[Tuple[float, float, str, int]] = []

    for token, hist in current_token_to_np_history.items():
        if hist is None or hist.size == 0:
            continue
        if hist.ndim != 2 or hist.shape[1] != 11:
            continue
        valid = _valid_mask_agent_rows(hist.astype(np.float32), invalid_eps=invalid_eps)
        if not np.any(valid):
            continue
        rows = hist[valid].astype(np.float32)

        x, y = rows[:, 0], rows[:, 1]
        c, s = rows[:, 2], rows[:, 3]
        vx, vy = rows[:, 4], rows[:, 5]
        W, L = rows[:, 6], rows[:, 7]
        corners = _batch_oriented_box_corners(x, y, c, s, L, W)
        polys.extend(list(corners))

        hx = x + (L * options.COMMON_heading_line_scale) * c
        hy = y + (L * options.COMMON_heading_line_scale) * s
        hd = np.stack([np.stack([x, y], axis=1), np.stack([hx, hy], axis=1)], axis=1)
        headings.extend(list(hd))

        if options.NEI_draw_neighbor_past_output_vel:
            xs_q.extend(x.tolist()); ys_q.extend(y.tolist())
            us_q.extend(vx.tolist()); vs_q.extend(vy.tolist())
            # 속도 숫자(짝/홀 오프셋)
            for i in range(rows.shape[0]):
                speed_kmh = float(np.hypot(vx[i], vy[i])) * 3.6
                offset = 4 if (i % 2 == 0) else 3
                texts.append((float(x[i]), float(y[i]) - options.NEI_neighbor_past_output_vel_offset_m * offset, f"{speed_kmh:.1f}", 20))

    if polys:
        pc = PolyCollection(polys, facecolors=(0, 0, 0, 0), edgecolors=line_color, linewidths=line_width, antialiased=False)
        pc.set_zorder(18); pc.set_rasterized(True); ax.add_collection(pc)
    if headings:
        lc = LineCollection(headings, colors=line_color, linewidths=line_width, antialiased=False)
        lc.set_zorder(18); lc.set_rasterized(True); ax.add_collection(lc)
    if xs_q:
        q = ax.quiver(np.array(xs_q), np.array(ys_q), np.array(us_q), np.array(vs_q),
                      angles="xy", scale_units="xy", scale=1.0,
                      color=line_color, linewidth=line_width, zorder=20)
        q.set_rasterized(True)
    for x, y, txt, z in texts:
        ax.text(x, y, txt, color=line_color,
                fontsize=options.NEI_neighbor_past_output_vel_fontsize,
                ha="center", va="bottom", zorder=z, clip_on=True)
