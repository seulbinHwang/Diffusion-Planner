import torch
from typing import Dict, Tuple, Optional

# 동역학 가이던스 비용의 비율과, 언제부터 가이던스를 켤지 결정하는 기본 값들
DEFAULT_PROJECTION_COST_WEIGHT: float = 1.0  # projection 거리 비용 비율
DEFAULT_CONSTRAINT_COST_WEIGHT: float = 0.  # 제약 위반량 비용 비율


def feasible_guidance_fn(
    x_dit: torch.Tensor,  # 정규화된 궤적 텐서 (B, Pnn, T, 4) 또는 (B, Pnn, (1+T), 4)
    t: torch.Tensor,
    cond: Optional[torch.Tensor],
    *args,
    **kwargs,
) -> torch.Tensor:
    """
    한 타임스텝에서 '동역학적으로 자연스러운 궤적' 쪽으로 당기는 에너지 값을 계산하는 함수.

    - 입력 궤적이 동역학 보정 궤적에서 멀수록, 그리고 제약을 많이 어길수록 큰 값을 내고,
    - 둘 다 괜찮으면 작은 값을 내도록 만든다.
    이 값은 나중에 미분되어, 궤적을 조금씩 더 자연스러운 방향으로 움직이는 데 사용된다.
    """
    # x_dit: (use_current_input = F (B, Pnn, T, 4) or use_current_input = T (B, Pnn, (1+T), 4)
    # t: (B,) 또는 (B, 1)       — 각 샘플의 현재 시간 값

    integrated_trajectory = kwargs["integrated_trajectory"]  # (B, Pnn, T, 4)
    control_constraint_diff = kwargs[
        "control_constraint_diff"]  # (B, Pnn, T, 3)
    model_condition: Dict = kwargs.get("model_condition", {})
    config = kwargs["config"]
    if config.use_current_input:
        # exclude current input.
        x_dit = x_dit[:, :, 1:, :]  # (B, Pnn, T, 4)
    _, _, x_len, _ = x_dit.shape
    _, _, int_traj_len, _ = integrated_trajectory.shape
    _, _, constraint_len, _ = control_constraint_diff.shape
    assert x_len == int_traj_len == constraint_len, (
        f"입력 궤적 길이({x_len})와 동역학 보정 궤적 길이({int_traj_len}), "
        f"제어 제약 차이 길이({constraint_len})가 서로 맞지 않습니다.")
    # 시간 텐서를 배치 크기에 맞는 1차원 벡터로 정리
    # current_time_vector: (B,)
    current_time_vector = _reshape_time_to_batch_vector(t, x_dit)

    # 현재 시간이 어느 구간인지에 따라, 동역학 가이던스를 얼마나 쓸지 결정
    # guidance_strength : (B,)
    guidance_strength = _compute_guidance_strength(
        current_time_vector,  # (B,)
        config,
    )

    # 아직 너무 이른 단계라서 가이던스를 완전히 끄는 경우
    if torch.all(guidance_strength == 0):
        batch_size: int = x_dit.shape[0]
        return torch.zeros(batch_size, device=x_dit.device,
                           dtype=x_dit.dtype)  # (B,)

    # projection 비용 / 제약 위반 비용에 쓸 유효 시간 마스크 만들기

    # near_future_valid : (B, Pnn, T)
    near_future_valid = _build_valid_masks_for_costs(
        x_dit,  # (B, Pnn, T, 4)
        model_condition,
    )

    # (1) 현재 궤적과 보정된 궤적 사이의 거리 기반 비용
    # projection_cost_per_batch: (B,)
    projection_cost_per_batch = _compute_projection_based_cost(
        x_dit,  # (B, Pnn, T, 4)
        integrated_trajectory,  # (B, Pnn, T, 4)
        near_future_valid,  # (B, Pnn, T)
    )

    # (2) 제어 값이 제약 때문에 얼마나 많이 잘렸는지 기반 비용
    # constraint_cost_per_batch = _compute_constraint_based_cost(
    #     control_constraint_diff,  # (B, Pnn, T, 3)
    #     near_future_valid,  # (B, Pnn, T)
    # )  # (B,)

    # 두 비용을 얼마나 비중 있게 볼지에 대한 가중치
    projection_cost_weight: float = kwargs.get(
        "feasible_projection_cost_weight",
        DEFAULT_PROJECTION_COST_WEIGHT,
    )
    # constraint_cost_weight: float = kwargs.get(
    #     "feasible_constraint_cost_weight",
    #     DEFAULT_CONSTRAINT_COST_WEIGHT,
    # )

    # 최종 동역학 비용: projection + 제약 위반 비용을 섞어서 만듦
    # (B,)
    total_feasible_cost = (projection_cost_weight * projection_cost_per_batch
                          )  #+
    # constraint_cost_weight * constraint_cost_per_batch
    #)

    # 아직 노이즈가 많은 단계라면 가이던스 세기를 줄이기 위해 시간 기반 가중치 곱하기
    total_feasible_cost = guidance_strength * total_feasible_cost  # (B,)

    # DPM 쪽에서는 '로그 확률'처럼 쓰기 때문에, 비용의 부호를 반대로 돌려서 넘겨준다.
    log_probability_like = -total_feasible_cost  # (B,)

    return log_probability_like


def _reshape_time_to_batch_vector(
    time_tensor: torch.Tensor,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    시간 t 값이 어떤 모양으로 들어오더라도, 샘플 개수와 맞는 1차원 벡터로 정리해 주는 함수.

    예를 들어 t가 스칼라이든, (1,) 이든, (B, 1) 이든 상관없이,
    궤적 묶음(배치) 크기 B에 맞춰서 (B,) 모양으로 만들어 준다.
    """
    # time_tensor: 스칼라 또는 (B,), (B, 1), (?, ...)
    # reference_tensor: (B, ...)  — 여기서 배치 크기를 가져온다.
    batch_size: int = reference_tensor.shape[0]

    if time_tensor.dim() == 0:
        time_tensor = time_tensor.expand(batch_size)  # (B,)
    else:
        time_tensor = time_tensor.view(-1)  # (?,)
        if time_tensor.shape[0] == 1 and batch_size > 1:
            time_tensor = time_tensor.expand(batch_size)  # (B,)

    return time_tensor  # (B,)


def _compute_guidance_strength(
    current_time_vector: torch.Tensor,
    config,
) -> torch.Tensor:
    """
    현재 시간 t 값에 따라 '동역학 가이던스를 얼마나 쓸지' 결정하는 함수.

    아직 노이즈가 많은 구간(t가 큰 쪽)에서는 거의 0에 가깝게,
    노이즈가 많이 줄어든 구간(t가 작아진 뒤)에는 1에 가깝게 값을 만들어 준다.
    """
    # current_time_vector: (B,)
    batch_size: int = current_time_vector.shape[0]
    feasible_learn_noise_thresh: float = config.feasible_learn_noise_thresh

    start_time_tensor = current_time_vector.new_full(
        (batch_size,),
        feasible_learn_noise_thresh,
    )  # (B,)

    # t <= feasible_learn_noise_thresh 인 구간만 가이던스를 켠다. (<=: 조금 여유 있게 켜기 위함)
    guidance_strength = (current_time_vector <= start_time_tensor).to(
        dtype=current_time_vector.dtype)  # (B,)

    return guidance_strength  # (B,)


def _build_valid_masks_for_costs(
    dit_trajectory: torch.Tensor,  # (B, Pnn, T, 4)
    model_condition: Dict,
) -> torch.Tensor:
    """
    궤적과 제어 값들 중에서 '실제로 유효한 시점'만 골라내는 마스크를 만드는 함수.

    과거·현재·미래 유효 여부를 담은 마스크에서,
    우리가 쓰려는 미래 구간만을 잘라서 projection / 제약 비용에 각각 맞춰준다.
    """
    # dit_trajectory: (B, Pnn, T, 4)
    batch_size, Pnn, future_len, _ = dit_trajectory.shape
    near_past_cur_future_valid: Optional[torch.Tensor] = model_condition.get(
        "near_past_cur_future_valid", None)  # (B, Pnn, T_full)
    if near_past_cur_future_valid is None:
        raise ValueError("near_past_cur_future_valid 마스크가 제공되지 않았습니다.")
    near_future_valid = near_past_cur_future_valid[:, :,
                                                   -future_len:]  # (B, Pnn, T)
    return near_future_valid


def _compute_projection_based_cost(
        dit_trajectory: torch.Tensor,  # (B, Pnn, T, 4)
        integrated_world_trajectory: torch.Tensor,  # (B, Pnn, T, 4)
        near_future_valid: torch.Tensor,  # (B, Pnn, T)
) -> torch.Tensor:
    """
    '현재 궤적'과 '동역학 보정 궤적' 사이의 거리로부터 비용을 계산하는 함수.

    두 궤적이 서로 비슷하게 움직이면 비용이 작고,
    서로 많이 다르면 비용이 커지도록 만든다.
    """
    # dit_trajectory: (B, Pnn, T, 4)
    # integrated_world_trajectory: (B, Pnn, T, 4)
    # near_future_valid: (B, Pnn, T)
    batch_size, Pnn, future_len, _ = dit_trajectory.shape
    near_future_valid_float = near_future_valid.to(
        dtype=dit_trajectory.dtype)  # (B, Pnn, T)

    # 궤적 차이의 제곱 거리 계산
    squared_difference = (dit_trajectory - integrated_world_trajectory).pow(
        2.0)  # (B, Pnn, T 4)
    squared_difference_sum = squared_difference.sum(dim=-1)  # (B, Pnn, T)

    masked_squared_difference = squared_difference_sum * near_future_valid_float  # (B, Pnn, T)

    # 유효한 시점 개수 (0으로 나누지 않기 위해 최소 1로 보정)
    valid_counts = near_future_valid_float.sum(dim=(1, 2))  # (B,)
    valid_counts = torch.clamp(valid_counts, min=1.0)  # (B,)

    # 배치별 평균 거리 비용
    cost_per_batch = masked_squared_difference.sum(
        dim=(1, 2)) / valid_counts  # (B,)

    return cost_per_batch  # (B,)


def _compute_constraint_based_cost(
    control_constraint_difference: torch.Tensor,
    near_future_valid: torch.Tensor,
) -> torch.Tensor:
    """
    제어 값이 제약 조건에 의해 얼마나 많이 잘렸는지를 비용으로 계산하는 함수.

    규칙을 거의 안 어기면 비용이 작고,
    제어 값을 크게 깎아야 했다면 비용이 커지도록 만든다.
    """
    # control_constraint_difference: (B, Pnn, T, 3)
    # near_future_valid: (B, Pnn, T)
    batch_size, Pnn, future_len, _ = control_constraint_difference.shape

    near_future_valid_float = near_future_valid.to(
        dtype=control_constraint_difference.dtype)  # (B, Pnn, T)

    # 제약 전/후 제어 차이의 제곱합
    squared_diff = control_constraint_difference.pow(2.0).sum(
        dim=-1)  # (B, Pnn, T)
    masked_squared_diff = squared_diff * near_future_valid_float  # (B, Pnn, T)

    valid_counts = near_future_valid_float.sum(dim=(1, 2))  # (B,)
    valid_counts = torch.clamp(valid_counts, min=1.0)  # (B,)

    cost_per_batch = masked_squared_diff.sum(dim=(1, 2)) / valid_counts  # (B,)

    return cost_per_batch  # (B,)
