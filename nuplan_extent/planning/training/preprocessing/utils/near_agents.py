from __future__ import annotations

from collections.abc import MutableMapping, Sequence
from typing import Any, Optional, Union, List

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore[assignment]

SampleDict = MutableMapping[str, Any]
SampleOrBatch = Union[SampleDict, Sequence[SampleDict]]


def get_near_track_token(
    neighbor_track_token: List[str],
    predicted_neighbor_num: int,
) -> List[str]:
    near_track_token = neighbor_track_token[:predicted_neighbor_num]
    return near_track_token


def add_near_agents_info_inplace(
    samples: SampleOrBatch,
    predicted_neighbor_num: int,
) -> None:
    """neighbor_agents를 near / non-near로 나누는 정보를 입력 dict에 "그 자리에서" 추가합니다.

    이 함수가 하는 일
    --------------
    - 입력이 샘플 1개(dict)면: 그 샘플 dict를 바로 수정합니다.
    - 입력이 샘플 여러 개(list[dict] 등)면: 각 샘플 dict를 순회하며 같은 작업을 합니다.

    near를 고르는 규칙
    --------------
    - `neighbor_agents_past`의 "앞쪽"부터 `predicted_neighbor_num`개를 near로 봅니다.
    - 다만, 같이 잘라야 하는 값들의 길이가 더 짧으면(예: agent_route_lane_order가 더 짧음)
      그에 맞춰 near 개수를 더 줄여서, 서로 길이가 어긋나지 않게 맞춥니다.

    추가/수정되는 key
    --------------
    - near_agents_past: neighbor_agents_past에서 near 부분만 따로 떼어둔 값
    - non_near_agents_past: neighbor_agents_past에서 near를 제외한 나머지
    - (있을 때만) near_future_gt_3_dim: neighbor_future_gt_3_dim의 near 부분
    - (있을 때만) agent_route_lane_order: agent 축을 near 개수만 남기도록 "덮어씀"
    - (있을 때만) agent_route_lane_order_is_valid: agent 축을 near 개수만 남기도록 "덮어씀"
    - (있을 때만) near_agents_past_is_valid / non_near_agents_past_is_valid
    - (있을 때만) near_agents_is_valid / non_near_agents_is_valid
    - (있을 때만) near_future_gt_is_valid / non_near_future_gt_is_valid

    입력에서 기대하는 대표 shape (샘플 1개 기준)
    ----------------------------------------
    - 배치가 없는 형태(학습 데이터 샘플 dict에서 흔함)
      - neighbor_agents_past: (A, time_len, F)
      - neighbor_future_gt_3_dim: (A, T_fut, 3)
      - agent_route_lane_order: (A, L)
      - agent_route_lane_order_is_valid: (A,)

    - 배치가 포함된 형태(추론/시뮬레이션에서 B=1로 들어오는 경우를 대비)
      - neighbor_agents_past: (B, A, time_len, F)
      - neighbor_future_gt_3_dim: (B, A, T_fut, 3)
      - agent_route_lane_order: (B, A, L)
      - agent_route_lane_order_is_valid: (B, A)

    Args:
        samples:
            - 샘플 1개 dict 또는 샘플 dict의 리스트/시퀀스.
            - 각 샘플 dict 안에는 최소한 "neighbor_agents_past"가 있어야 의미 있게 동작합니다.
        predicted_neighbor_num:
            - near로 뽑을 최대 agent 개수. (스칼라 int)

    Returns:
        None:
            - 입력 dict(또는 그 리스트) 내용을 그 자리에서 수정만 합니다.
    """
    if isinstance(samples, MutableMapping):
        _add_near_agents_info_for_one_sample_inplace(samples,
                                                     predicted_neighbor_num)
        return

    if not isinstance(samples, Sequence):
        return

    for sample in samples:
        if not isinstance(sample, MutableMapping):
            continue
        _add_near_agents_info_for_one_sample_inplace(sample,
                                                     predicted_neighbor_num)


def _add_near_agents_info_for_one_sample_inplace(
    sample: SampleDict,
    predicted_neighbor_num: int,
) -> None:
    """샘플 1개 dict에 near / non-near 정보를 그 자리에서 추가합니다.

    Args:
        sample:
            - 단일 샘플 dict.
            - 최소 "neighbor_agents_past" 키가 있어야 동작합니다.
        predicted_neighbor_num:
            - near로 뽑을 최대 agent 개수.

    Returns:
        None
    """
    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    if neighbor_agents_past is None:
        return

    agent_dim = _infer_agent_dim_index_from_neighbor_agents_past(
        neighbor_agents_past)
    if agent_dim is None:
        # neighbor_agents_past shape가 (A, T, F) 또는 (B, A, T, F) 형태가 아니면 안전하게 스킵
        return

    has_batch_dim: bool = (agent_dim == 1)

    near_num: int = _compute_near_num_for_sample(
        sample=sample,
        predicted_neighbor_num=predicted_neighbor_num,
        agent_dim=agent_dim,
        has_batch_dim=has_batch_dim,
    )

    # neighbor_agents_past:
    # - 배치 없음: (A, time_len, F)
    # - 배치 있음: (B, A, time_len, F)
    sample["near_agents_past"] = _slice_along_dim(
        value=neighbor_agents_past,
        dim=agent_dim,
        start=0,
        end=near_num,
    )
    sample["non_near_agents_past"] = _slice_along_dim(
        value=neighbor_agents_past,
        dim=agent_dim,
        start=near_num,
        end=None,
    )

    # --- (추가 1) neighbor_agents_past_is_valid -> near/non-near split ---
    neighbor_agents_past_is_valid = sample.get("neighbor_agents_past_is_valid",
                                               None)
    if neighbor_agents_past_is_valid is not None:
        expected_ndim = 3 if has_batch_dim else 2
        if _get_ndim(neighbor_agents_past_is_valid) == expected_ndim:
            sample["near_agents_past_is_valid"] = _slice_along_dim(
                value=neighbor_agents_past_is_valid,
                dim=agent_dim,
                start=0,
                end=near_num,
            )
            sample["non_near_agents_past_is_valid"] = _slice_along_dim(
                value=neighbor_agents_past_is_valid,
                dim=agent_dim,
                start=near_num,
                end=None,
            )

    # --- (추가 2) neighbor_agents_is_valid -> near/non-near split ---
    neighbor_agents_is_valid = sample.get("neighbor_agents_is_valid", None)
    if neighbor_agents_is_valid is not None:
        expected_ndim = 2 if has_batch_dim else 1
        if _get_ndim(neighbor_agents_is_valid) == expected_ndim:
            sample["near_agents_is_valid"] = _slice_along_dim(
                value=neighbor_agents_is_valid,
                dim=agent_dim,
                start=0,
                end=near_num,
            )
            sample["non_near_agents_is_valid"] = _slice_along_dim(
                value=neighbor_agents_is_valid,
                dim=agent_dim,
                start=near_num,
                end=None,
            )

    neighbor_future = sample.get("neighbor_future_gt_3_dim", None)
    if neighbor_future is not None:
        expected_ndim = 4 if has_batch_dim else 3
        if _get_ndim(neighbor_future) == expected_ndim:
            # neighbor_future_gt_3_dim:
            # - 배치 없음: (A, T_fut, 3)
            # - 배치 있음: (B, A, T_fut, 3)
            sample["near_future_gt_3_dim"] = _slice_along_dim(
                value=neighbor_future,
                dim=agent_dim,
                start=0,
                end=near_num,
            )

    # --- (추가 3) neighbor_future_gt_is_valid -> near/non-near split ---
    neighbor_future_gt_is_valid = sample.get("neighbor_future_gt_is_valid", None)
    if neighbor_future_gt_is_valid is not None:
        expected_ndim = 3 if has_batch_dim else 2
        if _get_ndim(neighbor_future_gt_is_valid) == expected_ndim:
            sample["near_future_gt_is_valid"] = _slice_along_dim(
                value=neighbor_future_gt_is_valid,
                dim=agent_dim,
                start=0,
                end=near_num,
            )
            sample["non_near_future_gt_is_valid"] = _slice_along_dim(
                value=neighbor_future_gt_is_valid,
                dim=agent_dim,
                start=near_num,
                end=None,
            )

    agent_route_lane_order = sample.get("agent_route_lane_order", None)
    if agent_route_lane_order is not None:
        expected_ndim = 3 if has_batch_dim else 2
        if _get_ndim(agent_route_lane_order) == expected_ndim:
            # agent_route_lane_order:
            # - 배치 없음: (A, L)
            # - 배치 있음: (B, A, L)
            sample["agent_route_lane_order"] = _slice_along_dim(
                value=agent_route_lane_order,
                dim=agent_dim,
                start=0,
                end=near_num,
            )


    agent_route_lane_order_is_valid = sample.get(
        "agent_route_lane_order_is_valid", None)
    if agent_route_lane_order_is_valid is not None:
        expected_ndim = 2 if has_batch_dim else 1
        if _get_ndim(agent_route_lane_order_is_valid) == expected_ndim:
            # agent_route_lane_order_is_valid:
            # - 배치 없음: (A,)
            # - 배치 있음: (B, A)
            sample["agent_route_lane_order_is_valid"] = _slice_along_dim(
                value=agent_route_lane_order_is_valid,
                dim=agent_dim,
                start=0,
                end=near_num,
            )


def _infer_agent_dim_index_from_neighbor_agents_past(
    neighbor_agents_past: Any,) -> Optional[int]:
    ndim = _get_ndim(neighbor_agents_past)
    if ndim is None:
        return None

    if int(ndim) == 3:
        return 0
    if int(ndim) == 4:
        return 1
    return None


def _compute_near_num_for_sample(
    sample: SampleDict,
    predicted_neighbor_num: int,
    agent_dim: int,
    has_batch_dim: bool,
) -> int:
    requested_near_num: int = max(0, int(predicted_neighbor_num))

    neighbor_agents_past = sample.get("neighbor_agents_past", None)
    a_len = _get_length_along_dim(neighbor_agents_past, agent_dim)
    if a_len is None:
        return 0

    near_num: int = min(requested_near_num, int(a_len))

    neighbor_future = sample.get("neighbor_future_gt_3_dim", None)
    if neighbor_future is not None:
        expected_ndim = 4 if has_batch_dim else 3
        if _get_ndim(neighbor_future) == expected_ndim:
            nf_len = _get_length_along_dim(neighbor_future, agent_dim)
            if nf_len is not None:
                near_num = min(near_num, int(nf_len))

    aro = sample.get("agent_route_lane_order", None)
    if aro is not None:
        expected_ndim = 3 if has_batch_dim else 2
        if _get_ndim(aro) == expected_ndim:
            aro_len = _get_length_along_dim(aro, agent_dim)
            if aro_len is not None:
                near_num = min(near_num, int(aro_len))

    aro_valid = sample.get("agent_route_lane_order_is_valid", None)
    if aro_valid is not None:
        expected_ndim = 2 if has_batch_dim else 1
        if _get_ndim(aro_valid) == expected_ndim:
            aro_valid_len = _get_length_along_dim(aro_valid, agent_dim)
            if aro_valid_len is not None:
                near_num = min(near_num, int(aro_valid_len))

    return int(near_num)


def _get_ndim(value: Any) -> Optional[int]:
    if value is None:
        return None

    if torch is not None and isinstance(value, torch.Tensor):
        return int(value.ndim)

    if isinstance(value, np.ndarray):
        return int(value.ndim)

    try:
        return int(np.asarray(value).ndim)
    except Exception:
        return None


def _get_length_along_dim(value: Any, dim: int) -> Optional[int]:
    if value is None:
        return None

    if torch is not None and isinstance(value, torch.Tensor):
        if int(value.ndim) <= int(dim):
            return None
        return int(value.shape[int(dim)])

    if isinstance(value, np.ndarray):
        if int(value.ndim) <= int(dim):
            return None
        return int(value.shape[int(dim)])

    try:
        arr = np.asarray(value)
        if int(arr.ndim) <= int(dim):
            return None
        return int(arr.shape[int(dim)])
    except Exception:
        return None


def _slice_along_dim(
    value: Any,
    dim: int,
    start: int,
    end: Optional[int],
) -> Any:
    if value is None:
        return None

    start_i = max(0, int(start))
    end_i = None if end is None else int(end)

    if torch is not None and isinstance(value, torch.Tensor):
        ndim = int(value.ndim)
        if ndim <= int(dim):
            return value
        slices = [slice(None)] * ndim
        slices[int(dim)] = slice(start_i, end_i)
        return value[tuple(slices)]

    if isinstance(value, np.ndarray):
        ndim = int(value.ndim)
        if ndim <= int(dim):
            return value
        slices = [slice(None)] * ndim
        slices[int(dim)] = slice(start_i, end_i)
        return value[tuple(slices)]

    arr = np.asarray(value)
    ndim = int(arr.ndim)
    if ndim <= int(dim):
        return arr
    slices = [slice(None)] * ndim
    slices[int(dim)] = slice(start_i, end_i)
    return arr[tuple(slices)]
