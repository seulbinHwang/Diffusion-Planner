from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class _RunningMeanRatio:
    """여러 시나리오에서 비율을 평균 내기 위한 누적기.

    같은 종류의 항목(예: lane_type)을 시나리오마다 '비율'로 만든 뒤,
    그 비율을 시나리오 단위로 평균내고 싶을 때 사용한다.
    (즉, 항목이 많은 시나리오가 과하게 영향 주지 않도록 한다.)
    """
    sum_ratio: np.ndarray  # shape: (K,)
    scenario_count: int

    @staticmethod
    def create(num_bins: int) -> "_RunningMeanRatio":
        return _RunningMeanRatio(
            sum_ratio=np.zeros((num_bins,), dtype=np.float64),  # shape: (K,)
            scenario_count=0,
        )

    def update_with_counts(self, counts: np.ndarray) -> None:
        """counts를 비율로 바꾼 뒤 평균 누적기에 더한다.

        Args:
            counts: shape (K,) 각 칸은 해당 종류의 '개수'
        """
        counts = np.asarray(counts).reshape(-1)  # shape: (K,)
        total = int(np.sum(counts))
        if total <= 0:
            return
        ratio = (counts.astype(np.float64) / float(total))  # shape: (K,)
        self.sum_ratio += ratio
        self.scenario_count += 1

    def mean(self) -> np.ndarray:
        """누적된 비율의 평균을 반환한다.

        Returns:
            mean_ratio: shape (K,)
        """
        if self.scenario_count <= 0:
            return np.zeros_like(self.sum_ratio, dtype=np.float64)
        return self.sum_ratio / float(self.scenario_count)


def _load_cache_pkl(pkl_path: Path) -> Dict[str, Any]:
    """캐시 pkl 파일 1개를 읽어 딕셔너리로 반환한다.

    Args:
        pkl_path: 읽을 pkl 파일 경로.

    Returns:
        cache_dict: pkl 안에 저장된 딕셔너리.
    """
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"pkl 내용이 dict가 아닙니다: {pkl_path}")
    return obj


def _safe_argmax_row(row: np.ndarray, default_index: int) -> int:
    """row가 전부 0이면 default_index를, 아니면 가장 큰 값의 칸 번호를 반환한다.

    Args:
        row: shape (K,)
        default_index: 전부 0일 때 돌려줄 칸 번호

    Returns:
        index: 0..K-1
    """
    row = np.asarray(row).reshape(-1)  # shape: (K,)
    if float(np.sum(np.abs(row))) == 0.0:
        return int(default_index)
    return int(np.argmax(row))


def compute_neighbor_role_counts(neighbor_role: np.ndarray) -> np.ndarray:
    """neighbor_role을 4가지 경우로 나눠 개수를 센다.

    4가지 경우(순서 고정):
        0: 둘 다 아님(interest=False, predict=False)
        1: 관심만(interest=True,  predict=False)
        2: 예측만(interest=False, predict=True)
        3: 둘 다(interest=True,  predict=True)

    Args:
        neighbor_role: shape (A, 2) bool 배열.
            - [:,0] = 관심 표시
            - [:,1] = 예측 표시

    Returns:
        counts: shape (4,) int64
    """
    role = np.asarray(neighbor_role)
    # role: np.ndarray, shape (A,2)
    if role.ndim != 2 or role.shape[1] < 2:
        return np.zeros((4,), dtype=np.int64)

    interest = role[:, 0].astype(bool)  # shape: (A,)
    predict = role[:, 1].astype(bool)   # shape: (A,)

    both = interest & predict
    interest_only = interest & (~predict)
    predict_only = (~interest) & predict
    none = (~interest) & (~predict)

    counts = np.array(
        [
            int(np.sum(none)),
            int(np.sum(interest_only)),
            int(np.sum(predict_only)),
            int(np.sum(both)),
        ],
        dtype=np.int64,
    )  # shape: (4,)
    return counts


def compute_lane_type_counts(lane_type: np.ndarray) -> np.ndarray:
    """lane_type(4칸 표시)를 세어서 종류별 개수를 만든다.

    lane_type의 각 행은 한 차선을 뜻한다.
    4칸 중 1이 들어있는 칸이 그 차선의 종류다.
    (혹시 전부 0이면 '미정(UNDEFINED)'으로 본다.)

    종류 순서(순서 고정):
        0: FREEWAY
        1: SURFACE_STREET
        2: BIKE_LANE
        3: UNDEFINED

    Args:
        lane_type: shape (L,4)

    Returns:
        counts: shape (4,) int64
    """
    lt = np.asarray(lane_type)
    # lt: np.ndarray, shape (L,4)
    if lt.ndim != 2 or lt.shape[1] != 4:
        return np.zeros((4,), dtype=np.int64)

    L = int(lt.shape[0])
    counts = np.zeros((4,), dtype=np.int64)  # shape: (4,)
    for i in range(L):
        idx = _safe_argmax_row(lt[i], default_index=3)
        counts[idx] += 1
    return counts


def compute_line_type_counts(line_type: np.ndarray) -> np.ndarray:
    """left/right_line_type(10칸 표시)를 세어서 종류별 개수를 만든다.

    10칸 중 1이 들어있는 칸이 해당 선의 종류다.
    (혹시 전부 0이면 'UNKNOWN(8번 칸)'으로 본다.)

    Args:
        line_type: shape (L,10)

    Returns:
        counts: shape (10,) int64
    """
    t = np.asarray(line_type)
    # t: np.ndarray, shape (L,10)
    if t.ndim != 2 or t.shape[1] != 10:
        return np.zeros((10,), dtype=np.int64)

    L = int(t.shape[0])
    counts = np.zeros((10,), dtype=np.int64)  # shape: (10,)
    for i in range(L):
        idx = _safe_argmax_row(t[i], default_index=8)
        counts[idx] += 1
    return counts


def compute_road_edge_type_counts(road_edge_type: np.ndarray) -> np.ndarray:
    """road_edge_type(3칸 표시)를 세어서 종류별 개수를 만든다.

    종류 순서(순서 고정):
        0: UNKNOWN
        1: BOUNDARY
        2: MEDIAN

    Args:
        road_edge_type: shape (E,3)

    Returns:
        counts: shape (3,) int64
    """
    t = np.asarray(road_edge_type)
    # t: np.ndarray, shape (E,3)
    if t.ndim != 2 or t.shape[1] != 3:
        return np.zeros((3,), dtype=np.int64)

    E = int(t.shape[0])
    counts = np.zeros((3,), dtype=np.int64)  # shape: (3,)
    for i in range(E):
        idx = _safe_argmax_row(t[i], default_index=0)
        counts[idx] += 1
    return counts


def _save_bar_png(
    values: np.ndarray,
    labels: Sequence[str],
    title: str,
    save_path: Path,
) -> None:
    """값들을 막대 그래프로 그려 PNG로 저장한다.

    Args:
        values: shape (K,) 값 배열(보통 0~1 비율).
        labels: 막대 이름 리스트(길이 K).
        title: 그래프 제목.
        save_path: 저장할 png 경로.
    """
    v = np.asarray(values).reshape(-1)  # shape: (K,)
    if len(labels) != int(v.shape[0]):
        raise ValueError("labels 길이와 values 길이가 다릅니다.")

    fig = plt.figure(figsize=(10.0, 4.0), dpi=200)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels), dtype=np.int64)  # shape: (K,)

    ax.bar(x, v)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, float(np.max(v)) * 1.2))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path.as_posix())
    plt.close(fig)


def analyze_cache_dir(
    cache_dir: Path,
    splits: List[str],
    max_scenarios_per_split: int,
    out_dir: Path,
) -> None:
    """캐시 폴더를 훑어서 타입/역할 분포를 평균으로 계산하고 그래프를 저장한다.

    Args:
        cache_dir: 예) /home/user/womd_v1_3/cache
        splits: 예) ["training", "validation"]
        max_scenarios_per_split: split마다 최대 몇 개 pkl까지 볼지
        out_dir: 결과 png를 저장할 폴더
    """
    # 평균(시나리오 단위) 누적기들
    role_mean = _RunningMeanRatio.create(num_bins=4)
    lane_type_mean = _RunningMeanRatio.create(num_bins=4)
    left_line_mean = _RunningMeanRatio.create(num_bins=10)
    right_line_mean = _RunningMeanRatio.create(num_bins=10)
    road_edge_mean = _RunningMeanRatio.create(num_bins=3)

    # 전체 개수 합(“아예 0인지”를 보기 좋게)
    role_sum = np.zeros((4,), dtype=np.int64)
    lane_type_sum = np.zeros((4,), dtype=np.int64)
    left_line_sum = np.zeros((10,), dtype=np.int64)
    right_line_sum = np.zeros((10,), dtype=np.int64)
    road_edge_sum = np.zeros((3,), dtype=np.int64)

    total_seen = 0

    for split in splits:
        split_dir = cache_dir / split
        if not split_dir.exists():
            continue

        pkl_paths = sorted(split_dir.glob("*.pkl"))
        if max_scenarios_per_split > 0:
            pkl_paths = pkl_paths[:max_scenarios_per_split]

        for pkl_path in pkl_paths:
            cache_dict = _load_cache_pkl(pkl_path)

            # --- neighbor_role ---
            if "neighbor_role" in cache_dict:
                counts = compute_neighbor_role_counts(cache_dict["neighbor_role"])
                role_sum += counts
                role_mean.update_with_counts(counts)

            # --- lane_type ---
            if "lane_type" in cache_dict:
                counts = compute_lane_type_counts(cache_dict["lane_type"])
                lane_type_sum += counts
                lane_type_mean.update_with_counts(counts)

            # --- left/right line type ---
            if "left_line_type" in cache_dict:
                counts = compute_line_type_counts(cache_dict["left_line_type"])
                left_line_sum += counts
                left_line_mean.update_with_counts(counts)

            if "right_lane_type" in cache_dict:
                counts = compute_line_type_counts(cache_dict["right_lane_type"])
                right_line_sum += counts
                right_line_mean.update_with_counts(counts)

            # --- road_edge_type ---
            if "road_edge_type" in cache_dict:
                counts = compute_road_edge_type_counts(cache_dict["road_edge_type"])
                road_edge_sum += counts
                road_edge_mean.update_with_counts(counts)

            total_seen += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # 텍스트 요약(콘솔)
    print(f"[INFO] analyzed_scenarios={total_seen}")
    print(f"[SUM] neighbor_role={role_sum.tolist()} (none, interest_only, predict_only, both)")
    print(f"[SUM] lane_type={lane_type_sum.tolist()} (FREEWAY, SURFACE_STREET, BIKE_LANE, UNDEFINED)")
    print(f"[SUM] road_edge_type={road_edge_sum.tolist()} (UNKNOWN, BOUNDARY, MEDIAN)")
    print(f"[SUM] left_line_type={left_line_sum.tolist()} (0..9)")
    print(f"[SUM] right_line_type={right_line_sum.tolist()} (0..9)")

    # 평균 비율 그래프 저장
    _save_bar_png(
        values=role_mean.mean(),
        labels=["none", "interest_only", "predict_only", "both"],
        title="neighbor_role 평균 비율(시나리오 단위 평균)",
        save_path=out_dir / "neighbor_role_mean_ratio.png",
    )
    _save_bar_png(
        values=lane_type_mean.mean(),
        labels=["FREEWAY", "SURFACE_STREET", "BIKE_LANE", "UNDEFINED"],
        title="lane_type 평균 비율(시나리오 단위 평균)",
        save_path=out_dir / "lane_type_mean_ratio.png",
    )
    _save_bar_png(
        values=left_line_mean.mean(),
        labels=[str(i) for i in range(10)],
        title="left_line_type 평균 비율(시나리오 단위 평균)",
        save_path=out_dir / "left_line_type_mean_ratio.png",
    )
    _save_bar_png(
        values=right_line_mean.mean(),
        labels=[str(i) for i in range(10)],
        title="right_line_type 평균 비율(시나리오 단위 평균)",
        save_path=out_dir / "right_line_type_mean_ratio.png",
    )
    _save_bar_png(
        values=road_edge_mean.mean(),
        labels=["UNKNOWN", "BOUNDARY", "MEDIAN"],
        title="road_edge_type 평균 비율(시나리오 단위 평균)",
        save_path=out_dir / "road_edge_type_mean_ratio.png",
    )

    print(f"[DONE] saved pngs to: {out_dir.as_posix()}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--splits", type=str, default="training,validation")
    parser.add_argument("--max_scenarios_per_split", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache_dir = Path(args.cache_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    out_dir = Path(args.out_dir) if args.out_dir else (cache_dir / "debug_stats")

    analyze_cache_dir(
        cache_dir=cache_dir,
        splits=splits,
        max_scenarios_per_split=int(args.max_scenarios_per_split),
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
