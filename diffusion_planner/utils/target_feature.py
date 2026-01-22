from typing import Any, Dict, Tuple
import torch


def build_target_future_tensors_and_masks_for_inference(
    args,
    norm_inputs: Dict[str, torch.Tensor],
    future_len: int,
):
    near_agents_past = norm_inputs["near_agents_past"]  # (B, Pnn, time_len, 11)
    near_agents_current = near_agents_past[:, :, -1, :]  # (B, Pnn, 11)
    #  near_agents_current_valid: (B, Pnn) 가 필요함. ->  # TODO 만드는 방식 바꾸기
    near_agents_current_valid = torch.sum(torch.ne(near_agents_current[..., :8],
                                                   0),
                                          dim=-1) != 0  # (B, Pnn) True=유효
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
    normed_ego_future_gt_4_dim: torch.Tensor,  # (B, future_len, 4)
    normed_near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
    near_cur_future_gt_is_valid: torch.Tensor,  # (B, Pnn, 1 + future_len)
    norm_near_current_4_dim: torch.Tensor,  # (B, Pnn, 4)
) -> Tuple[
        torch.
        Tensor,  # normed_target_cur_future_gt_4_dim: (B, (1+)Pnn, 1+future_len, 4)
        torch.
        Tensor,  # target_cur_future_is_valid: (B, (1+)Pnn, 1 + future_len)
]:
    """ego를 예측 대상에 포함할지 여부에 따라, 학습에 쓸 target 텐서/마스크를 만든다.

    이 함수는 학습 손실에서 “미래 정답(ground truth)”과 “유효/무효 마스크”를
    이웃만 쓸지, ego+이웃을 같이 쓸지에 따라 한 번에 만들어 줍니다.

    동작 규칙
    ----------
    - args.do_ego_predict == True:
        * target의 0번째 축(에이전트 축 맨 앞)에 ego를 붙입니다.
    - args.do_ego_predict == False:
        * 이웃(neighbor)만 그대로 사용합니다.

    Returns:
        normed_target_cur_future_gt_4_dim: (B, (1+)Pnn, 1+future_len, 4)
        target_cur_future_is_valid:
            (B, (1+)Pnn, 1+future_len)  True=유효


    """
    # normed_near_future_gt_4_dim: (B, Pnn, future_len, 4)
    if not args.do_ego_predict:
        normed_target_cur_future_gt_4_dim = torch.cat(
            [
                norm_near_current_4_dim.unsqueeze(2),  # (B, Pnn, 1, 4)
                normed_near_future_gt_4_dim
            ],  # (B, Pnn, future_len, 4)
            dim=2,
        )
        target_cur_future_is_valid = near_cur_future_gt_is_valid  # (B, Pnn, 1+future_len)

        return (
            normed_target_cur_future_gt_4_dim,  # (B, Pnn, 1+future_len, 4)
            target_cur_future_is_valid,  # (B, Pnn, 1+future_len)
        )

    # ----------------------------
    # ego + 이웃을 함께 학습 대상에 포함
    # ----------------------------

    ############## 미래 위치 텐서 만들기 ##############
    # (B, 1, future_len, 4)
    normed_ego_future_gt_4_dim_ = normed_ego_future_gt_4_dim.unsqueeze(1).to(
        dtype=normed_near_future_gt_4_dim.dtype,
        device=normed_near_future_gt_4_dim.device,
    )
    # normed_target_future_gt_4_dim: (B, 1 + Pnn, future_len, 4)
    normed_target_future_gt_4_dim = torch.cat(
        [normed_ego_future_gt_4_dim_, normed_near_future_gt_4_dim],
        dim=1,
    )
    ############## 현재 위치 텐서 만들기 ##############
    # ego_agent_past: (B, time_len, 11)
    ego_agent_past: torch.Tensor = norm_inputs["ego_agent_past"]
    # ego_cur_gt_11_dim: (B, 1, 11)
    ego_cur_gt_11_dim: torch.Tensor = ego_agent_past[:, -1:, :]
    # ego_current_xyyaw_norm: (B, 1, 4)
    ego_current_xyyaw_norm: torch.Tensor = ego_cur_gt_11_dim[:, :, :4].to(
        dtype=norm_near_current_4_dim.dtype,
        device=norm_near_current_4_dim.device,
    )
    # target_current_xyyaw_norm: (B, 1 + Pnn, 4)
    target_current_xyyaw_norm: torch.Tensor = torch.cat(
        [ego_current_xyyaw_norm, norm_near_current_4_dim],
        dim=1,
    )
    ########### 현재 + 미래 위치 텐서 만들기 ##############
    normed_target_cur_future_gt_4_dim = torch.cat(
        [
            target_current_xyyaw_norm.unsqueeze(2),  # (B, 1 + Pnn, 1, 4)
            normed_target_future_gt_4_dim,  # (B, 1 + Pnn, future_len, 4)
        ],
        dim=2,
    )
    ############### 유효 마스크 만들기 ##############
    # ego_future_gt_is_valid: (B, future_len)  True=유효
    ego_future_gt_is_valid = norm_inputs["ego_future_gt_is_valid"]
    # ego_agent_past_is_valid: (B, time_len)  True=유효
    ego_agent_past_is_valid = norm_inputs["ego_agent_past_is_valid"]
    ego_agent_current_is_valid: torch.Tensor = ego_agent_past_is_valid[:,
                                                                       -1:]  # (B, 1)  True=유효
    # ego_cur_future_is_valid: (B, 1 + future_len)  True=무효
    ego_cur_future_is_valid = torch.cat(
        [ego_agent_current_is_valid, ego_future_gt_is_valid],
        dim=1,
    ).to(device=near_cur_future_gt_is_valid.device)
    # target_cur_future_is_valid: (B, 1 + Pnn, 1 + future_len)
    target_cur_future_is_valid: torch.Tensor = torch.cat(
        [ego_cur_future_is_valid, near_cur_future_gt_is_valid],
        dim=1,
    )

    return (
        normed_target_cur_future_gt_4_dim,
        target_cur_future_is_valid,
    )
