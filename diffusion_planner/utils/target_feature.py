from typing import Any, Dict, Tuple
import torch


def build_target_future_tensors_and_masks_for_inference(
    args,
    norm_inputs: Dict[str, torch.Tensor],
):
    near_agents_past = norm_inputs["near_agents_past"]  # (B, Pnn, time_len, 11)
    near_agents_current = near_agents_past[:, :, -1, :]  # (B, Pnn, 11)
    #  near_agents_current_valid: (B, Pnn) 가 필요함. ->  # TODO 만드는 방식 바꾸기
    near_agents_current_valid = torch.sum(torch.ne(near_agents_current[..., :8],
                                                   0),
                                          dim=-1) != 0  # (B, Pnn) True=유효
    future_len = norm_inputs["planner_future_11_dim"].shape[1]
    if not args.do_ego_predict:
        # target_future_valid: (B, Pnn, future_len)  True=유효
        """ target_future_valid 만드는 법
        near_agents_current_valid 에서 True인 agent는 미래 예측을 수행하고, (valid=True)
        False인 agent는 미래 예측을 수행하지 않습니다. (valid=False)
        """
        target_future_valid = near_agents_current_valid.unsqueeze(-1).repeat(
            1, 1, future_len)  # (B, Pnn, future_len)  True=유효
    else:
        # target_future_valid: (B, 1 + Pnn, future_len)  True=유효
        """
        이번에는 ego를 포함합니다. ego는 항상 유효하다고 가정합니다.
        """
        B = norm_inputs["ego_agent_past"].shape[0]
        ego_current_valid = torch.ones(
            (B, 1),
            dtype=near_agents_current_valid.dtype,
            device=near_agents_current_valid.device)  # (B, 1) True=유효
        target_agents_current_valid = torch.cat(
            [ego_current_valid, near_agents_current_valid],
            dim=1,
        )  # (B, 1 + Pnn) True=유효
        target_future_valid = target_agents_current_valid.unsqueeze(-1).repeat(
            1, 1, future_len)  # (B, 1 + Pnn, future_len)  True=유효
    return target_future_valid  # (B, (1+)Pnn, future_len)  True=유효


def build_target_future_tensors_and_masks(
    args: Any,
    norm_inputs: Dict[str, torch.Tensor],
    ego_future_gt_4_dim: torch.Tensor,  # (B, future_len, 4)
    near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    near_cur_future_mask: torch.Tensor,  # (B, Pnn, 1 + future_len)
    near_current_xyyaw_norm: torch.Tensor,  # (B, Pnn, 4)
    near_future_valid: torch.Tensor,  # (B, Pnn, future_len)
) -> Tuple[
        torch.Tensor,  # target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
        torch.Tensor,  # target_cur_future_mask: (B, (1+)Pnn, 1 + future_len)
        torch.Tensor,  # target_current_xyyaw_norm: (B, (1+)Pnn, 4)
        torch.Tensor,  # target_future_valid: (B, (1+)Pnn, future_len) ##
]:
    """ego를 예측 대상에 포함할지 여부에 따라, 학습에 쓸 target 텐서/마스크를 만든다.

    이 함수는 학습 손실에서 “미래 정답(ground truth)”과 “유효/무효 마스크”를
    이웃만 쓸지, ego+이웃을 같이 쓸지에 따라 한 번에 만들어 줍니다.

    동작 규칙
    ----------
    - args.do_ego_predict == True:
        * target의 0번째 축(에이전트 축 맨 앞)에 ego를 붙입니다.
        * ego의 현재+미래 마스크는 `ego_cur_future_gt_11_dim[..., :8]`가 전부 0인 프레임을
          무효(True)로 처리합니다.
    - args.do_ego_predict == False:
        * 이웃(neighbor)만 그대로 사용합니다.

    Args:
        args: 설정 객체. `args.do_ego_predict`를 사용합니다.
        norm_inputs: 정규화된 입력 dict.
            - "ego_future_gt_11_dim": (B, future_len, 11)
            - "ego_agent_past": (B, time_len, 11)
        ego_future_gt_4_dim: (B, future_len, 4) ego 미래 정답(x,y,cos,sin).
        near_future_gt_4_dim: (B, Pnn, future_len, 4) 이웃 미래 정답.
        near_cur_future_mask: (B, Pnn, 1+future_len) 이웃 현재+미래 마스크(True=무효).
        near_current_xyyaw_norm: (B, Pnn, 4) 이웃 현재 상태(정규화).
        near_future_valid: (B, Pnn, future_len) 이웃 미래 유효 마스크(True=유효).

    Returns:
        target_future_gt_4_dim:
            (B, (1+)Pnn, future_len, 4)
            - ego를 포함하면 (B, 1+Pnn, future_len, 4)
            - 아니면 (B, Pnn, future_len, 4)
        target_cur_future_mask:
            (B, (1+)Pnn, 1+future_len)  True=무효(값 없음)
        target_current_xyyaw_norm:
            (B, (1+)Pnn, 4)  현재 상태(정규화)
        target_future_valid:
            (B, (1+)Pnn, future_len)  True=유효
    """
    # near_future_gt_4_dim: (B, Pnn, future_len, 4)
    B, Pnn, future_len, D4 = near_future_gt_4_dim.shape
    if int(ego_future_gt_4_dim.shape[0]) != B or int(
            ego_future_gt_4_dim.shape[1]) != future_len or int(
                ego_future_gt_4_dim.shape[2]) != 4:
        raise ValueError(
            f"ego_future_gt_4_dim must be (B, future_len, 4). got {tuple(ego_future_gt_4_dim.shape)}, "
            f"expected ({B}, {future_len}, 4)")

    if near_cur_future_mask.shape != (B, Pnn, 1 + future_len):

        raise ValueError(
            f"near_cur_future_mask must be (B, Pnn, 1+future_len). got {tuple(near_cur_future_mask.shape)}, "
            f"expected ({B}, {Pnn}, {1 + future_len})")

    if near_current_xyyaw_norm.shape != (B, Pnn, 4):
        raise ValueError(
            f"near_current_xyyaw_norm must be (B, Pnn, 4). got {tuple(near_current_xyyaw_norm.shape)}, "
            f"expected ({B}, {Pnn}, 4)")
    if near_future_valid.shape != (B, Pnn, future_len):
        raise ValueError(
            f"near_future_valid must be (B, Pnn, future_len). got {tuple(near_future_valid.shape)}, "
            f"expected ({B}, {Pnn}, {future_len})")

    if not args.do_ego_predict:
        # 이웃만 사용
        target_future_gt_4_dim = near_future_gt_4_dim  # (B, Pnn, future_len, 4)
        target_cur_future_mask = near_cur_future_mask  # (B, Pnn, 1+future_len)
        target_current_xyyaw_norm = near_current_xyyaw_norm  # (B, Pnn, 4)
        target_future_valid = near_future_valid  # (B, Pnn, future_len)
        return (
            target_future_gt_4_dim,
            target_cur_future_mask,
            target_current_xyyaw_norm,
            target_future_valid,
        )

    # ----------------------------
    # ego + 이웃을 함께 학습 대상에 포함
    # ----------------------------

    # (B, 1, future_len, 4)
    ego_future_gt_4_dim_ = ego_future_gt_4_dim.unsqueeze(1).to(
        dtype=near_future_gt_4_dim.dtype,
        device=near_future_gt_4_dim.device,
    )
    # target_future_gt_4_dim: (B, 1 + Pnn, future_len, 4)
    target_future_gt_4_dim = torch.cat(
        [ego_future_gt_4_dim_, near_future_gt_4_dim],
        dim=1,
    )

    # ego_future_gt_11_dim: (B, future_len, 11)
    ego_future_gt_11_dim: torch.Tensor = norm_inputs["planner_future_11_dim"]
    # ego_agent_past: (B, time_len, 11)
    ego_agent_past: torch.Tensor = norm_inputs["ego_agent_past"]
    # ego_cur_gt_11_dim: (B, 1, 11)
    ego_cur_gt_11_dim: torch.Tensor = ego_agent_past[:, -1:, :]

    # ego_cur_future_gt_11_dim: (B, 1 + future_len, 11)
    ego_cur_future_gt_11_dim: torch.Tensor = torch.cat(
        [ego_cur_gt_11_dim, ego_future_gt_11_dim],
        dim=1,
    )

    # ego_cur_future_mask: (B, 1 + future_len)  True=무효

    # ego_cur_future_gt_11_dim: (B, 1 + future_len, 11)
    ego_cur_future_mask: torch.Tensor = (torch.sum(torch.ne(
        ego_cur_future_gt_11_dim[..., :8], 0),
                                                   dim=-1) == 0)
    # (B, 1, 1 + future_len)
    ego_cur_future_mask = ego_cur_future_mask.unsqueeze(1)

    # target_cur_future_mask: (B, 1 + Pnn, 1 + future_len)
    target_cur_future_mask: torch.Tensor = torch.cat(
        [
            ego_cur_future_mask.to(device=near_cur_future_mask.device),
            near_cur_future_mask
        ],
        dim=1,
    )

    # ego_current_xyyaw_norm: (B, 1, 4)
    ego_current_xyyaw_norm: torch.Tensor = ego_cur_gt_11_dim[:, :, :4].to(
        dtype=near_current_xyyaw_norm.dtype,
        device=near_current_xyyaw_norm.device,
    )
    # target_current_xyyaw_norm: (B, 1 + Pnn, 4)
    target_current_xyyaw_norm: torch.Tensor = torch.cat(
        [ego_current_xyyaw_norm, near_current_xyyaw_norm],
        dim=1,
    )

    # target_future_mask: (B, 1 + Pnn, future_len)  True=무효
    target_future_mask: torch.Tensor = target_cur_future_mask[:, :, 1:]
    # target_future_valid: (B, 1 + Pnn, future_len)  True=유효
    target_future_valid: torch.Tensor = ~target_future_mask

    return (
        target_future_gt_4_dim,
        target_cur_future_mask,
        target_current_xyyaw_norm,
        target_future_valid,
    )
