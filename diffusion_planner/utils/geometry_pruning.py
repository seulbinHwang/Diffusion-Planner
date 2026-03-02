# diffusion_planner/utils/geometry_pruning.py
"""
거리 기반 pruning(선택) 유틸.

이 모듈은 "인코더에 넣기 전에" lane / road-safety 입력 개수를 줄이기 위해
거리(제곱거리) 기준 Top-K 선택을 제공합니다.

- sqrt를 쓰지 않습니다. (정렬/Top-K 결과는 제곱거리로도 동일)
- 파이썬 루프 없이, 텐서 연산으로 한 번에 처리하는 형태를 기본으로 합니다.
- 입력이 None이거나 topk<=0이면 원본을 그대로 반환합니다.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch


def build_agents_xy_and_valid_from_past(
    ego_agent_past: torch.Tensor,
    near_agents_past: torch.Tensor,
    ego_agent_past_is_valid: Optional[torch.Tensor] = None,
    near_agents_past_is_valid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """현재 시점 agent (x, y)와 유효 마스크를 만듭니다.

    ego와 주변 agent의 "마지막 시점" 위치를 사용합니다.

    Args:
        ego_agent_past (torch.Tensor):
            shape: (B, T, D_ego)
            마지막 시점의 [:2]가 (x,y)라고 가정합니다.
        near_agents_past (torch.Tensor):
            shape: (B, N, T, D_near)
            마지막 시점의 [:2]가 (x,y)라고 가정합니다.
        ego_agent_past_is_valid (Optional[torch.Tensor]):
            shape: (B, T) 또는 (B, T, 1)
            마지막 시점이 True면 ego 위치가 유효.
            None이면 전부 유효로 간주합니다.
        near_agents_past_is_valid (Optional[torch.Tensor]):
            shape: (B, N, T) 또는 (B, N, T, 1)
            마지막 시점이 True면 해당 agent 위치가 유효.
            None이면 전부 유효로 간주합니다.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            agents_xy:
                shape: (B, P, 2)
                P = 1 + N (ego 1개 + 주변 N개)
            agents_valid:
                shape: (B, P) bool

    Raises:
        ValueError: 입력 shape이 기대와 다르면 발생합니다.
    """
    if ego_agent_past.dim() != 3:
        raise ValueError(
            f"ego_agent_past must be (B,T,D). got {tuple(ego_agent_past.shape)}"
        )
    if near_agents_past.dim() != 4:
        raise ValueError(
            f"near_agents_past must be (B,N,T,D). got {tuple(near_agents_past.shape)}"
        )

    B: int = int(ego_agent_past.shape[0])
    if int(near_agents_past.shape[0]) != B:
        raise ValueError("ego/near batch size mismatch")

    # (B, 2)
    ego_xy: torch.Tensor = ego_agent_past[:, -1, :2]
    # (B, N, 2)
    near_xy: torch.Tensor = near_agents_past[:, :, -1, :2]

    # (B, 1+N, 2)
    agents_xy: torch.Tensor = torch.cat([ego_xy.unsqueeze(1), near_xy], dim=1)

    # -------------------------
    # valid 마스크 (bool)
    # -------------------------
    if ego_agent_past_is_valid is None:
        ego_valid_last = torch.ones((B,), device=agents_xy.device, dtype=torch.bool)
    else:
        if ego_agent_past_is_valid.dim() == 3 and int(ego_agent_past_is_valid.shape[-1]) == 1:
            ego_agent_past_is_valid = ego_agent_past_is_valid.squeeze(-1)
        if ego_agent_past_is_valid.dim() != 2:
            raise ValueError(
                "ego_agent_past_is_valid must be (B,T) or (B,T,1). "
                f"got {tuple(ego_agent_past_is_valid.shape)}"
            )
        ego_valid_last = ego_agent_past_is_valid[:, -1].to(torch.bool)

    if near_agents_past_is_valid is None:
        N: int = int(near_agents_past.shape[1])
        near_valid_last = torch.ones((B, N), device=agents_xy.device, dtype=torch.bool)
    else:
        if near_agents_past_is_valid.dim() == 4 and int(near_agents_past_is_valid.shape[-1]) == 1:
            near_agents_past_is_valid = near_agents_past_is_valid.squeeze(-1)
        if near_agents_past_is_valid.dim() != 3:
            raise ValueError(
                "near_agents_past_is_valid must be (B,N,T) or (B,N,T,1). "
                f"got {tuple(near_agents_past_is_valid.shape)}"
            )
        near_valid_last = near_agents_past_is_valid[:, :, -1].to(torch.bool)

    # (B, 1+N)
    agents_valid: torch.Tensor = torch.cat(
        [ego_valid_last.unsqueeze(1), near_valid_last], dim=1
    )

    return agents_xy, agents_valid


def _gather_dim1(x: Optional[torch.Tensor], index_bk: torch.Tensor) -> Optional[torch.Tensor]:
    """x의 dim=1 축을 (B,K) 인덱스로 gather 합니다.

    Args:
        x: shape (B, L, ...) 또는 None
        index_bk: shape (B, K) int64

    Returns:
        shape (B, K, ...) 또는 None

    Note:
        - x가 2D이면 torch.gather로 바로 처리합니다.
        - x가 3D 이상이면 index를 expand한 뒤 gather 합니다.
    """
    if x is None:
        return None
    if x.dim() < 2:
        raise ValueError(f"x must have dim >= 2. got {tuple(x.shape)}")

    if index_bk.dtype != torch.long:
        index_bk = index_bk.to(torch.long)

    if x.dim() == 2:
        # (B, L) -> (B, K)
        return torch.gather(x, dim=1, index=index_bk)

    # x: (B, L, d2, d3, ...)
    B: int = int(x.shape[0])
    K: int = int(index_bk.shape[1])
    if int(index_bk.shape[0]) != B:
        raise ValueError("batch size mismatch in gather")

    # (B, K, 1, 1, ...)
    idx = index_bk
    for _ in range(x.dim() - 2):
        idx = idx.unsqueeze(-1)

    # (B, K, d2, d3, ...)
    idx = idx.expand((B, K, *x.shape[2:]))
    return torch.gather(x, dim=1, index=idx)


def _min_dist2_agents_to_points(
    agents_xy: torch.Tensor,
    agents_valid: torch.Tensor,
    points_xy: torch.Tensor,
    points_valid: torch.Tensor,
) -> torch.Tensor:
    """agents와 points 사이의 "가장 가까운 제곱거리"를 point마다 계산합니다.

    Args:
        agents_xy: shape (B, P, 2)
        agents_valid: shape (B, P) bool
        points_xy: shape (B, N, 2)
        points_valid: shape (B, N) bool

    Returns:
        torch.Tensor:
            min_dist2_per_point:
                shape (B, N) float32
                각 point에 대해, 유효한 agent들 중 가장 가까운 제곱거리.
                유효 agent/point가 하나도 없으면 +inf가 됩니다.

    Note:
        - sqrt를 쓰지 않습니다.
        - 계산은 float32로 수행합니다(정렬/Top-K 안정성 목적).
    """
    if agents_xy.dim() != 3 or int(agents_xy.shape[-1]) != 2:
        raise ValueError(f"agents_xy must be (B,P,2). got {tuple(agents_xy.shape)}")
    if agents_valid.dim() != 2:
        raise ValueError(
            f"agents_valid must be (B,P). got {tuple(agents_valid.shape)}"
        )
    if points_xy.dim() != 3 or int(points_xy.shape[-1]) != 2:
        raise ValueError(f"points_xy must be (B,N,2). got {tuple(points_xy.shape)}")
    if points_valid.dim() != 2:
        raise ValueError(
            f"points_valid must be (B,N). got {tuple(points_valid.shape)}"
        )

    B: int = int(agents_xy.shape[0])
    if int(points_xy.shape[0]) != B:
        raise ValueError("batch size mismatch (agents vs points)")

    P: int = int(agents_xy.shape[1])
    N: int = int(points_xy.shape[1])

    # N==0이면 바로 빈 결과
    if N == 0:
        return torch.empty((B, 0), device=agents_xy.device, dtype=torch.float32)

    # P==0이면 전부 inf
    if P == 0:
        return torch.full((B, N), float("inf"), device=agents_xy.device, dtype=torch.float32)

    # float32로 계산
    a: torch.Tensor = agents_xy.to(dtype=torch.float32)  # (B,P,2)
    b: torch.Tensor = points_xy.to(dtype=torch.float32)  # (B,N,2)

    # (B,P,1)
    a2: torch.Tensor = (a * a).sum(dim=-1, keepdim=True)
    # (B,1,N)
    b2: torch.Tensor = (b * b).sum(dim=-1).unsqueeze(1)

    # (B,P,N) = a @ b^T
    # a: (B,P,2), bT: (B,2,N)
    ab: torch.Tensor = torch.bmm(a, b.transpose(1, 2))

    dist2: torch.Tensor = a2 + b2 - 2.0 * ab  # (B,P,N)
    dist2 = torch.clamp(dist2, min=0.0)

    # 마스킹: invalid는 inf
    inf = float("inf")
    invalid_agents = (~agents_valid.to(torch.bool)).unsqueeze(-1)  # (B,P,1)
    invalid_points = (~points_valid.to(torch.bool)).unsqueeze(1)   # (B,1,N)
    dist2 = dist2.masked_fill(invalid_agents, inf)
    dist2 = dist2.masked_fill(invalid_points, inf)

    # point마다 agent 축 최소
    min_dist2_per_point: torch.Tensor = dist2.amin(dim=1)  # (B,N)
    return min_dist2_per_point


def _lane_min_dist2(
    agents_xy: torch.Tensor,
    agents_valid: torch.Tensor,
    lanes_xy: torch.Tensor,
    lane_point_valid: torch.Tensor,
) -> torch.Tensor:
    """lane별 "가장 가까운 제곱거리"를 계산합니다.

    정의:
        - 한 lane은 S개의 점(보통 20개)을 가집니다.
        - lane_min_dist2[b, l] =
            유효한 agent들의 현재 위치와
            유효한 lane 점들 사이 제곱거리의 최소값

    Args:
        agents_xy: shape (B,P,2)
        agents_valid: shape (B,P) bool
        lanes_xy: shape (B,L,S,2)
        lane_point_valid: shape (B,L,S) bool

    Returns:
        torch.Tensor: shape (B,L) float32
    """
    if lanes_xy.dim() != 4 or int(lanes_xy.shape[-1]) != 2:
        raise ValueError(f"lanes_xy must be (B,L,S,2). got {tuple(lanes_xy.shape)}")
    if lane_point_valid.shape[:3] != lanes_xy.shape[:3]:
        raise ValueError(
            "lane_point_valid must match lanes_xy (B,L,S). "
            f"got {tuple(lane_point_valid.shape)} vs {tuple(lanes_xy.shape)}"
        )

    B, L, S, _ = lanes_xy.shape

    # (B, L*S, 2)
    points_xy = lanes_xy.reshape(B, L * S, 2)
    # (B, L*S)
    points_valid = lane_point_valid.reshape(B, L * S)

    # (B, L*S)
    min_dist2_per_point = _min_dist2_agents_to_points(
        agents_xy=agents_xy,
        agents_valid=agents_valid,
        points_xy=points_xy,
        points_valid=points_valid,
    )

    # (B, L, S) -> (B, L)
    lane_min_dist2 = min_dist2_per_point.view(B, L, S).amin(dim=2)
    return lane_min_dist2


def prune_lanes_by_distance_to_agents(
    lanes: torch.Tensor,
    lanes_is_valid: torch.Tensor,
    agents_xy: torch.Tensor,
    agents_valid: torch.Tensor,
    topk: int,
    *,
    lanes_len_is_valid: Optional[torch.Tensor] = None,
    lanes_speed_limit: Optional[torch.Tensor] = None,
    lanes_has_speed_limit: Optional[torch.Tensor] = None,
    lane_type: Optional[torch.Tensor] = None,
    left_line_type: Optional[torch.Tensor] = None,
    right_line_type: Optional[torch.Tensor] = None,
    use_batch_max_k: bool = True,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    torch.Tensor,
]:
    """lane을 거리 기준으로 Top-K만 남기고 나머지는 버립니다.

    Args:
        lanes:
            shape (B, L, S, D_lane)
        lanes_is_valid:
            shape (B, L)  (bool 또는 0/1)
        agents_xy:
            shape (B, P, 2)
        agents_valid:
            shape (B, P) bool
        topk:
            남길 최대 lane 개수(A)
        lanes_len_is_valid:
            shape (B, L, S) 또는 None
            점 단위 유효 마스크. None이면 lane_is_valid만으로 처리합니다.
        lanes_speed_limit:
            shape (B, L, 1) 또는 None
        lanes_has_speed_limit:
            shape (B, L, 1) 또는 None
        lane_type:
            shape (B, L, 4) 또는 None
        left_line_type:
            shape (B, L, 13) 또는 None
        right_line_type:
            shape (B, L, 13) 또는 None
        use_batch_max_k:
            True면 K를 "배치에서 가장 많이 유효한 개수"까지로 줄입니다.
            예) A=75, 배치 최대 유효 lane=40 -> K=40

    Returns:
        Tuple:
            lanes_pruned: (B, K, S, D_lane)
            lanes_is_valid_pruned: (B, K)
            lanes_speed_limit_pruned: (B, K, 1) 또는 None
            lanes_has_speed_limit_pruned: (B, K, 1) 또는 None
            lane_type_pruned: (B, K, 4) 또는 None
            left_line_type_pruned: (B, K, 13) 또는 None
            right_line_type_pruned: (B, K, 13) 또는 None
            lanes_len_is_valid_pruned: (B, K, S) 또는 None
            selected_indices: (B, K) int64

    Note:
        - topk<=0 이면 입력을 그대로 반환합니다.
        - 유효 lane이 부족한 샘플은 invalid lane이 K에 포함될 수 있습니다.
          이때 lanes_is_valid_pruned가 False로 들어가므로 이후 마스크로 처리됩니다.
    """
    if lanes.dim() != 4:
        raise ValueError(f"lanes must be (B,L,S,D). got {tuple(lanes.shape)}")
    if lanes_is_valid.dim() != 2:
        raise ValueError(
            f"lanes_is_valid must be (B,L). got {tuple(lanes_is_valid.shape)}"
        )

    B, L, S, _ = lanes.shape

    lanes_is_valid_bool = lanes_is_valid.to(torch.bool)

    # pruning 비활성
    if topk <= 0 or L == 0:
        selected = torch.arange(L, device=lanes.device, dtype=torch.long).view(1, L).expand(B, L)
        return (
            lanes,
            lanes_is_valid_bool,
            lanes_speed_limit,
            lanes_has_speed_limit,
            lane_type,
            left_line_type,
            right_line_type,
            lanes_len_is_valid,
            selected,
        )

    A = min(int(topk), int(L))

    if use_batch_max_k:
        # (B,) -> 스칼라 int (GPU sync 1회)
        batch_max_valid = int(lanes_is_valid_bool.sum(dim=1).max().item())
        K = min(A, batch_max_valid)
    else:
        K = A

    # K==0이면 빈 텐서 반환
    if K <= 0:
        empty_idx = torch.zeros((B, 0), device=lanes.device, dtype=torch.long)
        lanes_empty = lanes[:, :0]
        lanes_is_valid_empty = lanes_is_valid_bool[:, :0]
        return (
            lanes_empty,
            lanes_is_valid_empty,
            None if lanes_speed_limit is None else lanes_speed_limit[:, :0],
            None if lanes_has_speed_limit is None else lanes_has_speed_limit[:, :0],
            None if lane_type is None else lane_type[:, :0],
            None if left_line_type is None else left_line_type[:, :0],
            None if right_line_type is None else right_line_type[:, :0],
            None if lanes_len_is_valid is None else lanes_len_is_valid[:, :0],
            empty_idx,
        )

    # lane 점(x,y): (B,L,S,2)
    lanes_xy = lanes[..., :2]

    if lanes_len_is_valid is None:
        lane_point_valid = lanes_is_valid_bool.unsqueeze(-1).expand(B, L, S)
    else:
        if lanes_len_is_valid.shape[:3] != (B, L, S):
            raise ValueError(
                f"lanes_len_is_valid must be (B,L,S)={B,L,S}. got {tuple(lanes_len_is_valid.shape)}"
            )
        lane_point_valid = lanes_len_is_valid.to(torch.bool) & lanes_is_valid_bool.unsqueeze(-1)

    # (B,L) float32
    lane_min_dist2 = _lane_min_dist2(
        agents_xy=agents_xy,
        agents_valid=agents_valid,
        lanes_xy=lanes_xy,
        lane_point_valid=lane_point_valid,
    )

    # Top-K (가까운 순)
    _, selected_idx = torch.topk(
        lane_min_dist2,
        k=int(K),
        dim=1,
        largest=False,
        sorted=True,
    )  # (B,K)

    lanes_pruned = _gather_dim1(lanes, selected_idx)
    lanes_is_valid_pruned = _gather_dim1(lanes_is_valid_bool, selected_idx)

    lanes_speed_limit_pruned = _gather_dim1(lanes_speed_limit, selected_idx)
    lanes_has_speed_limit_pruned = _gather_dim1(lanes_has_speed_limit, selected_idx)
    lane_type_pruned = _gather_dim1(lane_type, selected_idx)
    left_line_type_pruned = _gather_dim1(left_line_type, selected_idx)
    right_line_type_pruned = _gather_dim1(right_line_type, selected_idx)
    lanes_len_is_valid_pruned = _gather_dim1(lanes_len_is_valid, selected_idx)

    return (
        lanes_pruned,
        lanes_is_valid_pruned,
        lanes_speed_limit_pruned,
        lanes_has_speed_limit_pruned,
        lane_type_pruned,
        left_line_type_pruned,
        right_line_type_pruned,
        lanes_len_is_valid_pruned,
        selected_idx,
    )


def _empty_points(ref: torch.Tensor, batch_size: int, points_per_object: int) -> torch.Tensor:
    """(B, 0, S, 2) empty points 텐서를 만듭니다."""
    return ref.new_zeros((int(batch_size), 0, int(points_per_object), 2))


def _empty_valid(ref: torch.Tensor, batch_size: int) -> torch.Tensor:
    """(B, 0) empty bool 텐서를 만듭니다."""
    return torch.zeros((int(batch_size), 0), device=ref.device, dtype=torch.bool)


def _pack_selected_by_type(
    selected_points: torch.Tensor,
    selected_valid: torch.Tensor,
    selected_type: torch.Tensor,
    target_type: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """선택된 (B,K,...) 항목을 타입별로 모아 (B, K_type_max, ...)로 pack 합니다.

    Args:
        selected_points: (B, K, S, 2)
        selected_valid: (B, K) bool
        selected_type: (B, K) int64
        target_type: 모을 타입 id (0..)

    Returns:
        out_points: (B, Kt, S, 2)
        out_valid:  (B, Kt) bool
        여기서 Kt는 "이 배치에서 target_type으로 선택된 최대 개수"입니다.
        (샘플마다 개수는 다를 수 있고, 부족한 부분은 out_valid=False로 패딩)
    """
    if selected_points.dim() != 4:
        raise ValueError(
            f"selected_points must be (B,K,S,2). got {tuple(selected_points.shape)}"
        )
    B, K, S, _ = selected_points.shape

    mask = (selected_type == int(target_type)) & selected_valid.to(torch.bool)  # (B,K)
    if int(K) == 0:
        return selected_points[:, :0], selected_valid[:, :0]

    # Kt = 배치 내 최대 선택 수 (GPU sync 1회)
    Kt = int(mask.sum(dim=1).max().item())
    if Kt <= 0:
        return selected_points[:, :0], selected_valid[:, :0]

    out_points = selected_points.new_zeros((B, Kt, S, 2))
    out_valid = torch.zeros((B, Kt), device=selected_points.device, dtype=torch.bool)

    # 각 샘플에서 mask가 True인 항목의 "타입 내 순번"(0..count-1)
    pos = torch.cumsum(mask.to(torch.int64), dim=1) - 1  # (B,K)

    mask_flat = mask.reshape(B * K)
    if int(mask_flat.sum().item()) == 0:
        return out_points, out_valid

    b_ids = (
        torch.arange(B, device=selected_points.device, dtype=torch.long)
        .unsqueeze(1)
        .expand(B, K)
        .reshape(B * K)
    )
    b_sel = b_ids[mask_flat]  # (M,)
    pos_sel = pos.reshape(B * K)[mask_flat]  # (M,)
    pts_sel = selected_points.reshape(B * K, S, 2)[mask_flat]  # (M,S,2)

    out_points[b_sel, pos_sel] = pts_sel
    out_valid[b_sel, pos_sel] = True
    return out_points, out_valid


def prune_road_safety_by_distance_to_agents(
    agents_xy: torch.Tensor,
    agents_valid: torch.Tensor,
    stop_sign_points: Optional[torch.Tensor],
    stop_sign_is_valid: Optional[torch.Tensor],
    crosswalk_points: Optional[torch.Tensor],
    crosswalk_is_valid: Optional[torch.Tensor],
    speed_bump_points: Optional[torch.Tensor],
    speed_bump_is_valid: Optional[torch.Tensor],
    driveway_points: Optional[torch.Tensor],
    driveway_is_valid: Optional[torch.Tensor],
    road_edge: Optional[torch.Tensor],
    road_edge_is_valid: Optional[torch.Tensor],
    road_edge_type: Optional[torch.Tensor],
    topk_total: int,
    topk_road_edge: int,
    *,
    use_batch_max_k: bool = True,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """road-safety 요소를 agent와의 거리 기준으로 pruning 합니다.

    처리 규칙(요청 반영)
    --------------------
    1) stop/cross/speed/driveway 4종은 합쳐서 Top-(topk_total)을 뽑습니다.
    2) road_edge는 별도로 Top-(topk_road_edge)을 뽑습니다.
    3) 거리 계산은 "agent 현재 (x,y)" vs "각 요소의 10개 점"이며,
       요소별 최소 제곱거리로 순위를 매깁니다.

    Args:
        agents_xy: (B, P, 2)
        agents_valid: (B, P)
        *_points: (B, N, 10, 2) 또는 None
        *_is_valid: (B, N) 또는 None
        road_edge_type: (B, N, 3) 또는 None
        topk_total: 4종 합쳐서 남길 최대 개수
        topk_road_edge: road_edge에서 남길 최대 개수
        use_batch_max_k: True면 K를 배치 최대 유효 개수까지로 줄입니다.

    Returns:
        pruning된 입력들을 원래 포맷으로 돌려줍니다.
        각 텐서는 원래와 같은 의미의 "pad 자리"를 is_valid=False로 둡니다.

        순서:
            stop_sign_points, stop_sign_is_valid,
            crosswalk_points, crosswalk_is_valid,
            speed_bump_points, speed_bump_is_valid,
            driveway_points, driveway_is_valid,
            road_edge, road_edge_is_valid, road_edge_type

    Note:
        - topk_total/topk_road_edge <= 0이면 해당 그룹은 pruning 하지 않습니다.
        - 입력이 None인 카테고리는 None 그대로 반환합니다.
    """
    B: int = int(agents_xy.shape[0])
    device = agents_xy.device

    do_total = int(topk_total) > 0

    def _as_points_and_valid(
        points: Optional[torch.Tensor],
        is_valid: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if points is None:
            return _empty_points(ref, B, 10), _empty_valid(ref, B), False
        if points.dim() != 4 or int(points.shape[0]) != B or int(points.shape[2]) != 10 or int(points.shape[3]) != 2:
            raise ValueError(
                f"points must be (B,N,10,2). got {tuple(points.shape)}"
            )
        N = int(points.shape[1])
        if is_valid is None:
            valid = torch.ones((B, N), device=points.device, dtype=torch.bool)
        else:
            if is_valid.dim() != 2:
                raise ValueError(
                    f"is_valid must be (B,N). got {tuple(is_valid.shape)}"
                )
            valid = is_valid.to(torch.bool).to(device=points.device)
        return points, valid, True

    ref_for_empty = (
        stop_sign_points
        if stop_sign_points is not None
        else crosswalk_points
        if crosswalk_points is not None
        else speed_bump_points
        if speed_bump_points is not None
        else driveway_points
        if driveway_points is not None
        else road_edge
        if road_edge is not None
        else agents_xy
    )

    stop_p, stop_v, stop_exist = _as_points_and_valid(stop_sign_points, stop_sign_is_valid, ref_for_empty)
    cross_p, cross_v, cross_exist = _as_points_and_valid(crosswalk_points, crosswalk_is_valid, ref_for_empty)
    bump_p, bump_v, bump_exist = _as_points_and_valid(speed_bump_points, speed_bump_is_valid, ref_for_empty)
    drive_p, drive_v, drive_exist = _as_points_and_valid(driveway_points, driveway_is_valid, ref_for_empty)

    if do_total:
        points_all = torch.cat([stop_p, cross_p, bump_p, drive_p], dim=1)  # (B, Ntot, 10, 2)
        valid_all = torch.cat([stop_v, cross_v, bump_v, drive_v], dim=1)   # (B, Ntot)

        N_stop = int(stop_p.shape[1])
        N_cross = int(cross_p.shape[1])
        N_bump = int(bump_p.shape[1])
        N_drive = int(drive_p.shape[1])
        N_total = int(points_all.shape[1])

        if N_total > 0:
            type_ids_1d = torch.cat(
                [
                    torch.full((N_stop,), 0, device=device, dtype=torch.long),
                    torch.full((N_cross,), 1, device=device, dtype=torch.long),
                    torch.full((N_bump,), 2, device=device, dtype=torch.long),
                    torch.full((N_drive,), 3, device=device, dtype=torch.long),
                ],
                dim=0,
            )  # (N_total,)
            type_ids = type_ids_1d.unsqueeze(0).expand(B, N_total)  # (B, N_total)

            pts_flat = points_all.reshape(B, N_total * 10, 2)
            pts_valid_flat = valid_all.unsqueeze(-1).expand(B, N_total, 10).reshape(B, N_total * 10)

            min_d2_pts = _min_dist2_agents_to_points(
                agents_xy=agents_xy,
                agents_valid=agents_valid,
                points_xy=pts_flat,
                points_valid=pts_valid_flat,
            )  # (B, N_total*10)

            obj_min_d2 = min_d2_pts.view(B, N_total, 10).amin(dim=2)  # (B, N_total)

            A = min(int(topk_total), int(N_total))
            if use_batch_max_k:
                batch_max_valid = int(valid_all.sum(dim=1).max().item())
                K = min(A, batch_max_valid)
            else:
                K = A

            if K > 0:
                _, sel_idx = torch.topk(obj_min_d2, k=int(K), dim=1, largest=False, sorted=True)

                sel_points = _gather_dim1(points_all, sel_idx)  # (B,K,10,2)
                sel_valid = _gather_dim1(valid_all, sel_idx)    # (B,K)
                sel_type = _gather_dim1(type_ids, sel_idx)      # (B,K)

                stop_p2, stop_v2 = _pack_selected_by_type(sel_points, sel_valid, sel_type, 0)
                cross_p2, cross_v2 = _pack_selected_by_type(sel_points, sel_valid, sel_type, 1)
                bump_p2, bump_v2 = _pack_selected_by_type(sel_points, sel_valid, sel_type, 2)
                drive_p2, drive_v2 = _pack_selected_by_type(sel_points, sel_valid, sel_type, 3)

                stop_sign_points = None if not stop_exist else stop_p2
                stop_sign_is_valid = None if not stop_exist else stop_v2

                crosswalk_points = None if not cross_exist else cross_p2
                crosswalk_is_valid = None if not cross_exist else cross_v2

                speed_bump_points = None if not bump_exist else bump_p2
                speed_bump_is_valid = None if not bump_exist else bump_v2

                driveway_points = None if not drive_exist else drive_p2
                driveway_is_valid = None if not drive_exist else drive_v2
            else:
                stop_sign_points = None if not stop_exist else stop_p[:, :0]
                stop_sign_is_valid = None if not stop_exist else stop_v[:, :0]

                crosswalk_points = None if not cross_exist else cross_p[:, :0]
                crosswalk_is_valid = None if not cross_exist else cross_v[:, :0]

                speed_bump_points = None if not bump_exist else bump_p[:, :0]
                speed_bump_is_valid = None if not bump_exist else bump_v[:, :0]

                driveway_points = None if not drive_exist else drive_p[:, :0]
                driveway_is_valid = None if not drive_exist else drive_v[:, :0]
        else:
            stop_sign_points = None if not stop_exist else stop_p
            stop_sign_is_valid = None if not stop_exist else stop_v

            crosswalk_points = None if not cross_exist else cross_p
            crosswalk_is_valid = None if not cross_exist else cross_v

            speed_bump_points = None if not bump_exist else bump_p
            speed_bump_is_valid = None if not bump_exist else bump_v

            driveway_points = None if not drive_exist else drive_p
            driveway_is_valid = None if not drive_exist else drive_v

    # road_edge pruning
    do_edge = (road_edge is not None) and (int(topk_road_edge) > 0)
    if do_edge:
        assert road_edge is not None
        if road_edge.dim() != 4 or int(road_edge.shape[0]) != B or int(road_edge.shape[2]) != 10 or int(road_edge.shape[3]) != 2:
            raise ValueError(f"road_edge must be (B,N,10,2). got {tuple(road_edge.shape)}")

        Ne = int(road_edge.shape[1])
        if road_edge_is_valid is None:
            edge_valid = torch.ones((B, Ne), device=road_edge.device, dtype=torch.bool)
        else:
            edge_valid = road_edge_is_valid.to(torch.bool).to(device=road_edge.device)

        if Ne > 0:
            pts_flat = road_edge.reshape(B, Ne * 10, 2)
            pts_valid_flat = edge_valid.unsqueeze(-1).expand(B, Ne, 10).reshape(B, Ne * 10)

            min_d2_pts = _min_dist2_agents_to_points(
                agents_xy=agents_xy,
                agents_valid=agents_valid,
                points_xy=pts_flat,
                points_valid=pts_valid_flat,
            )  # (B, Ne*10)

            edge_min_d2 = min_d2_pts.view(B, Ne, 10).amin(dim=2)  # (B, Ne)

            A = min(int(topk_road_edge), int(Ne))
            if use_batch_max_k:
                batch_max_valid = int(edge_valid.sum(dim=1).max().item())
                K = min(A, batch_max_valid)
            else:
                K = A

            if K > 0:
                _, sel_idx = torch.topk(edge_min_d2, k=int(K), dim=1, largest=False, sorted=True)
                road_edge = _gather_dim1(road_edge, sel_idx)
                road_edge_is_valid = _gather_dim1(edge_valid, sel_idx)
                if road_edge_type is not None:
                    road_edge_type = _gather_dim1(road_edge_type, sel_idx)
            else:
                road_edge = road_edge[:, :0]
                road_edge_is_valid = edge_valid[:, :0]
                if road_edge_type is not None:
                    road_edge_type = road_edge_type[:, :0]
        else:
            road_edge_is_valid = edge_valid

    return (
        stop_sign_points,
        stop_sign_is_valid,
        crosswalk_points,
        crosswalk_is_valid,
        speed_bump_points,
        speed_bump_is_valid,
        driveway_points,
        driveway_is_valid,
        road_edge,
        road_edge_is_valid,
        road_edge_type,
    )