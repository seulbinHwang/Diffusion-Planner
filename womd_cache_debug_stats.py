from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import matplotlib.pyplot as plt


# =========================
# Label dictionaries (EN)
# =========================
LINE_TYPE_INDEX_TO_NAME_EN: List[str] = [
    "0: BROKEN_SINGLE_WHITE",
    "1: SOLID_SINGLE_WHITE",
    "2: SOLID_DOUBLE_WHITE",
    "3: BROKEN_SINGLE_YELLOW",
    "4: BROKEN_DOUBLE_YELLOW",
    "5: SOLID_SINGLE_YELLOW",
    "6: SOLID_DOUBLE_YELLOW",
    "7: PASSING_DOUBLE_YELLOW",
    "8: UNKNOWN",
    "9: INVALID(no line)",
]


@dataclass
class _RunningMeanRatio:
    """Running mean accumulator for per-scenario category ratios.

    For each scenario, we convert category counts into ratios and accumulate them.
    This prevents scenarios with many items from dominating the final average.

    Attributes:
        sum_ratio: shape (K,) float64, accumulated ratios.
        scenario_count: number of scenarios that contributed.
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
        """Convert counts to ratio and add to the running sum.

        Args:
            counts: shape (K,) int-like. Each bin is the count for a category.
        """
        counts = np.asarray(counts).reshape(-1)  # shape: (K,)
        total = int(np.sum(counts))
        if total <= 0:
            return
        ratio = counts.astype(np.float64) / float(total)  # shape: (K,)
        self.sum_ratio += ratio
        self.scenario_count += 1

    def mean(self) -> np.ndarray:
        """Return mean ratio over scenarios.

        Returns:
            mean_ratio: shape (K,) float64
        """
        if self.scenario_count <= 0:
            return np.zeros_like(self.sum_ratio, dtype=np.float64)
        return self.sum_ratio / float(self.scenario_count)


def _load_cache_pkl(pkl_path: Path) -> Dict[str, Any]:
    """Load one cache pkl and return as a dict."""
    with pkl_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"PKL content is not a dict: {pkl_path}")
    return obj


def _safe_argmax_row(row: np.ndarray, default_index: int) -> int:
    """Return argmax index, or default_index if the row is all zeros."""
    row = np.asarray(row).reshape(-1)  # shape: (K,)
    if float(np.sum(np.abs(row))) == 0.0:
        return int(default_index)
    return int(np.argmax(row))


def compute_neighbor_role_counts(neighbor_role: np.ndarray) -> np.ndarray:
    """Count neighbor roles into 4 buckets.

    Bucket order:
        0: none (interest=False, predict=False)
        1: interest_only (interest=True,  predict=False)
        2: predict_only  (interest=False, predict=True)
        3: both          (interest=True,  predict=True)

    Args:
        neighbor_role: shape (A, 2) bool array.
            - [:,0] = interest flag
            - [:,1] = predict flag

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
    """Count lane_type one-hot rows into 4 categories.

    Category order:
        0: FREEWAY
        1: SURFACE_STREET
        2: BIKE_LANE
        3: UNDEFINED (also used if all-zero)

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
    """Count left/right line type (10-dim one-hot) into 10 categories.

    If a row is all-zero, it is treated as UNKNOWN (index 8).

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
    """Count road_edge_type (3-dim one-hot) into 3 categories.

    Category order:
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
    x_label: str = "Category",
    y_label: str = "Mean ratio",
) -> None:
    """Save bar chart to PNG (English only)."""
    v = np.asarray(values).reshape(-1)  # shape: (K,)
    if len(labels) != int(v.shape[0]):
        raise ValueError("labels length must match values length.")

    fig = plt.figure(figsize=(12.0, 4.8), dpi=200)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels), dtype=np.int64)  # shape: (K,)

    ax.bar(x, v)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
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
    """Analyze cached pkls and save English-only summary plots."""
    role_mean = _RunningMeanRatio.create(num_bins=4)
    lane_type_mean = _RunningMeanRatio.create(num_bins=4)
    left_line_mean = _RunningMeanRatio.create(num_bins=10)
    right_line_mean = _RunningMeanRatio.create(num_bins=10)
    road_edge_mean = _RunningMeanRatio.create(num_bins=3)

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

            if "neighbor_role" in cache_dict:
                counts = compute_neighbor_role_counts(cache_dict["neighbor_role"])
                role_sum += counts
                role_mean.update_with_counts(counts)

            if "lane_type" in cache_dict:
                counts = compute_lane_type_counts(cache_dict["lane_type"])
                lane_type_sum += counts
                lane_type_mean.update_with_counts(counts)

            if "left_line_type" in cache_dict:
                counts = compute_line_type_counts(cache_dict["left_line_type"])
                left_line_sum += counts
                left_line_mean.update_with_counts(counts)

            if "right_lane_type" in cache_dict:
                counts = compute_line_type_counts(cache_dict["right_lane_type"])
                right_line_sum += counts
                right_line_mean.update_with_counts(counts)

            if "road_edge_type" in cache_dict:
                counts = compute_road_edge_type_counts(cache_dict["road_edge_type"])
                road_edge_sum += counts
                road_edge_mean.update_with_counts(counts)

            total_seen += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    # Console summary (English)
    print(f"[INFO] analyzed_scenarios={total_seen}")
    print(f"[SUM] neighbor_role={role_sum.tolist()} (none, interest_only, predict_only, both)")
    print(f"[SUM] lane_type={lane_type_sum.tolist()} (FREEWAY, SURFACE_STREET, BIKE_LANE, UNDEFINED)")
    print(f"[SUM] road_edge_type={road_edge_sum.tolist()} (UNKNOWN, BOUNDARY, MEDIAN)")
    print(f"[SUM] left_line_type={left_line_sum.tolist()} (0..9)")
    print(f"[SUM] right_line_type={right_line_sum.tolist()} (0..9)")

    # Save mean-ratio plots (English titles)
    _save_bar_png(
        values=role_mean.mean(),
        labels=["none", "interest_only", "predict_only", "both"],
        title="Neighbor role: mean ratio (scenario-wise average)",
        save_path=out_dir / "neighbor_role_mean_ratio.png",
        x_label="Role",
        y_label="Mean ratio",
    )
    _save_bar_png(
        values=lane_type_mean.mean(),
        labels=["FREEWAY", "SURFACE_STREET", "BIKE_LANE", "UNDEFINED"],
        title="Lane type: mean ratio (scenario-wise average)",
        save_path=out_dir / "lane_type_mean_ratio.png",
        x_label="Lane type",
        y_label="Mean ratio",
    )
    _save_bar_png(
        values=left_line_mean.mean(),
        labels=LINE_TYPE_INDEX_TO_NAME_EN,
        title="Left line type: mean ratio (scenario-wise average)",
        save_path=out_dir / "left_line_type_mean_ratio.png",
        x_label="Road line type",
        y_label="Mean ratio",
    )
    _save_bar_png(
        values=right_line_mean.mean(),
        labels=LINE_TYPE_INDEX_TO_NAME_EN,
        title="Right line type: mean ratio (scenario-wise average)",
        save_path=out_dir / "right_line_type_mean_ratio.png",
        x_label="Road line type",
        y_label="Mean ratio",
    )
    _save_bar_png(
        values=road_edge_mean.mean(),
        labels=["UNKNOWN", "BOUNDARY", "MEDIAN"],
        title="Road edge type: mean ratio (scenario-wise average)",
        save_path=out_dir / "road_edge_type_mean_ratio.png",
        x_label="Road edge type",
        y_label="Mean ratio",
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
