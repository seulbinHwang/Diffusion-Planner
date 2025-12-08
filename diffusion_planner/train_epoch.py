from tqdm import tqdm
import torch
from torch import nn
from typing import Tuple
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.model.module.feasible import FeasibleProjector
# =====================================================================
from typing import Dict, Tuple, Optional, List, Any
import argparse

import torch
from torch import nn
from typing import Tuple
...
from diffusion_planner.model.module.feasible import FeasibleProjector

# =====================================================================


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: str,
) -> Dict[str, torch.Tensor]:
    """배치 dict의 모든 텐서를 지정한 device 로 옮긴다.

    Args:
        batch: DataLoader 에서 온 배치 dict. 각 값은 torch.Tensor.
        device: "cuda:0", "cpu" 등.

    Returns:
        Dict[str, torch.Tensor]:
            동일 key 를 가지되, 텐서가 모두 device 상에 놓인 dict.
    """
    return {
        key: tensor.to(device, non_blocking=True)
        for key, tensor in batch.items()
    }


def _clip_input_axes_by_args(
    batch_on_device: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """모델 설정 상한에 맞춰 입력 텐서들의 agent / lane 축을 in-place 로 자른다.

    처리 규칙 (do_not_clip=False 일 때만 적용):
      - neighbor_agents_past      : agent 축 → args.max_agent_num
      - lanes / lanes_*           : lane  축 → args.max_use_lane_num
      - route_lanes / route_lanes_* : lane 축 → args.max_use_lane_num
      - agent_route_lane_order    : (agent, lane) 축 → (args.predicted_neighbor_num, args.max_use_lane_num)
    """
    if getattr(args, "do_not_clip", False):
        return

    # ---- neighbor_agents_past: (B, A, T, 11) → A ≤ max_agent_num ----
    max_agent_num = int(getattr(args, "max_agent_num", 0))
    if "neighbor_agents_past" in batch_on_device and max_agent_num > 0:
        neighbor_agents_past = batch_on_device["neighbor_agents_past"]
        if neighbor_agents_past.dim() == 4:
            # neighbor_agents_past: (B, A_c, T, 11)
            keep_agents = min(max_agent_num, neighbor_agents_past.shape[1])
            batch_on_device["neighbor_agents_past"] = \
                neighbor_agents_past[:, :keep_agents, :, :]

    # ---- lanes / lanes_* : lane 축 → max_use_lane_num ----
    max_use_lane_num = int(getattr(args, "max_use_lane_num", 0))
    if "lanes" in batch_on_device and max_use_lane_num > 0:
        lanes = batch_on_device["lanes"]
        if lanes.dim() == 4:
            # lanes: (B, L_c, lane_len, 12)
            keep_lanes = min(max_use_lane_num, lanes.shape[1])
            if keep_lanes < lanes.shape[1]:
                batch_on_device["lanes"] = lanes[:, :keep_lanes, :, :]
                if "lanes_speed_limit" in batch_on_device:
                    # (B, L_c, 1)
                    batch_on_device["lanes_speed_limit"] = \
                        batch_on_device["lanes_speed_limit"][:, :keep_lanes, :]
                if "lanes_has_speed_limit" in batch_on_device:
                    # (B, L_c, 1)
                    batch_on_device["lanes_has_speed_limit"] = \
                        batch_on_device["lanes_has_speed_limit"][:, :keep_lanes, :]

    # ---- route_lanes / route_lanes_* : lane 축 → max_use_lane_num ----
    if "route_lanes" in batch_on_device and max_use_lane_num > 0:
        route_lanes = batch_on_device["route_lanes"]
        if route_lanes.dim() == 4:
            # route_lanes: (B, R_c, route_len, 12)
            keep_route_lanes = min(max_use_lane_num, route_lanes.shape[1])
            if keep_route_lanes < route_lanes.shape[1]:
                batch_on_device["route_lanes"] = \
                    route_lanes[:, :keep_route_lanes, :, :]
                if "route_lanes_speed_limit" in batch_on_device:
                    # (B, R_c, 1)
                    batch_on_device["route_lanes_speed_limit"] = \
                        batch_on_device["route_lanes_speed_limit"][:, :keep_route_lanes, :]
                if "route_lanes_has_speed_limit" in batch_on_device:
                    # (B, R_c, 1)
                    batch_on_device["route_lanes_has_speed_limit"] = \
                        batch_on_device["route_lanes_has_speed_limit"][:, :keep_route_lanes, :]

    # ---- agent_route_lane_order: (B, A_c, L_c) → (A', L') ----
    predicted_neighbor_num = int(getattr(args, "predicted_neighbor_num", 0))
    if "agent_route_lane_order" in batch_on_device:
        arl = batch_on_device["agent_route_lane_order"]
        if arl.dim() == 3:
            # arl: (B, A_c, L_c)
            bsz, c_agent, c_lane = arl.shape

            keep_agents_for_route = c_agent
            if predicted_neighbor_num > 0:
                keep_agents_for_route = min(predicted_neighbor_num, c_agent)

            lane_dim_input = c_lane
            if "lanes" in batch_on_device:
                # lanes: (B, L', lane_len, 12)
                lane_dim_input = batch_on_device["lanes"].shape[1]
            keep_lanes_for_route = min(
                max_use_lane_num or lane_dim_input,
                lane_dim_input,
            )

            batch_on_device["agent_route_lane_order"] = \
                arl[:, :keep_agents_for_route, :keep_lanes_for_route]


def _clip_targets_by_predicted_neighbors(
    outputs: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    """정답 텐서 중 neighbor 축(predicted_neighbor_num)만 자른다.

    현재는 near_future_gt_3_dim 만 처리한다.
      - near_future_gt_3_dim: (B, A_c, Tf, 3) → A_c ≤ predicted_neighbor_num

    do_not_clip=True 이면 아무 것도 자르지 않는다.
    """
    if getattr(args, "do_not_clip", False):
        return

    if "near_future_gt_3_dim" not in outputs:
        return

    predicted_neighbor_num = int(getattr(args, "predicted_neighbor_num", 0))
    near_future_gt_3_dim = outputs["near_future_gt_3_dim"]
    if near_future_gt_3_dim.dim() != 4 or predicted_neighbor_num <= 0:
        return

    # near_future_gt_3_dim: (B, A_c, Tf, 3) → (B, A', Tf, 3)
    keep_agents = min(predicted_neighbor_num, near_future_gt_3_dim.shape[1])
    outputs["near_future_gt_3_dim"] = \
        near_future_gt_3_dim[:, :keep_agents, :, :]


def _prepare_batch_for_device(
    batch: Dict[str, torch.Tensor],
    device: str,
    args: Optional[argparse.Namespace] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """배치 dict를 GPU/CPU로 옮기고, 모델 상한에 맞게 축을 잘라 입력/정답을 나눈다.

    전제: collate_fn(DiffusionPlannerCollate)에서 이미
      - neighbor_agents_past         : (B, data_max_agent_num, time_len, 11)
      - near_future_gt_3_dim         : (B, data_max_agent_num, future_len, 3)
      - static_objects               : (B, data_max_static_num, 10)
      - lanes / lanes_*              : (B, data_max_lane_num, lane_len, ·)
      - route_lanes / route_lanes_*  : (B, data_max_route_num, route_len, ·)
      - agent_route_lane_order       : (B, data_max_agent_num, data_max_lane_num)
    형태로 패딩이 끝난 상태라고 가정한다.

    여기서는 (do_not_clip=False 인 경우에만)
      - neighbor_agents_past / near_future_gt_3_dim: agent 축 → max_agent_num / predicted_neighbor_num
      - lanes / lanes_*                            : lane  축 → max_use_lane_num
      - route_lanes / route_lanes_*                : lane  축 → max_use_lane_num
      - agent_route_lane_order                     : (agent, lane) 축 → (predicted_neighbor_num, max_use_lane_num)
      - static_objects                              : 그대로 유지

    를 수행한다.
    """
    # 1) device 로 옮기기
    batch_on_device: Dict[str,
                          torch.Tensor] = _move_batch_to_device(batch, device)

    # 2) dtype 특수 처리: route lane order는 항상 long 형으로 맞춘다.
    if "agent_route_lane_order" in batch_on_device:
        # (B, A_c, L_c)
        batch_on_device["agent_route_lane_order"] = batch_on_device[
            "agent_route_lane_order"].long()

    # 3) 입력 텐서들의 agent / lane 축을 모델 상한에 맞게 자르기
    if args is not None:
        _clip_input_axes_by_args(batch_on_device, args)

    # 4) 정답 키만 outputs 로 분리
    target_keys = {"ego_future_gt_3_dim", "near_future_gt_3_dim"}
    outputs: Dict[str, torch.Tensor] = {}
    for key in list(batch_on_device.keys()):
        if key in target_keys:
            outputs[key] = batch_on_device.pop(key)

    # 5) 정답 중 neighbor 축(predicted_neighbor_num)만 안쪽에서 추가 클리핑
    if args is not None:
        _clip_targets_by_predicted_neighbors(outputs, args)

    inputs: Dict[str, torch.Tensor] = batch_on_device
    return inputs, outputs


def _get_stage_for_step(args: argparse.Namespace) -> int:
    """현재 전역 스텝에 따라 사용할 Stage 번호를 계산한다.

    전역 스텝은 학습 중 전체 업데이트 횟수를 0부터 센 값이다.
    이 값과 미리 계산해 둔 경계 스텝을 비교해서 Stage1/2/3을 나눈다.

    사용되는 값:
        - args._global_update_step (int):
            지금까지 진행된 전체 update step 수.
        - args._stage1_end_step (int):
            Stage1 이 끝나는 step 인덱스 (해당 값은 Stage1에 포함되지 않음).
        - args._stage2_end_step (int):
            Stage2 가 끝나는 step 인덱스 (해당 값은 Stage2에 포함되지 않음).
        - args._total_update_steps (int):
            전체 학습 동안 사용할 총 update step 수.

    구간 정의(0 기반 step):
        - 0 <= step < _stage1_end_step         → Stage 1
        - _stage1_end_step <= step < _stage2_end_step → Stage 2
        - _stage2_end_step <= step < _total_update_steps → Stage 3

    Args:
        args: 학습 설정과 상태를 담고 있는 Namespace.

    Returns:
        int: 현재 step 에서 사용할 Stage 번호 (1, 2, 3 중 하나).
    """
    cur_step: int = int(getattr(args, "_global_update_step", 0))
    stage1_end: int = int(getattr(args, "_stage1_end_step", 0))
    stage2_end: int = int(getattr(args, "_stage2_end_step", 0))
    total_steps: int = int(getattr(args, "_total_update_steps", 0))

    if total_steps <= 0 or stage2_end <= 0:
        return 1

    if cur_step < stage1_end:
        return 1
    if cur_step < stage2_end:
        return 2
    return 3


def _get_stage_group_lr_scales(
    stage: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """현재 Stage에서 각 파라미터 그룹에 곱해 줄 학습률 비율을 만든다.

    그룹 의미(build_adamw_with_param_groups 에서 붙인 stage_group):
        - "encoder_local"  : Group A (로컬 인코더)
        - "encoder_global" : Group B (Fusion 등, 여러 객체를 함께 보는 인코더)
        - "decoder"        : Group C (디코더 + PRAM/Feasible)
        - "others"         : 그 외 나머지

    정책:
        * Stage 1:
            - 모든 그룹 lr_scale = 1.0
              (Stage1 max lr를 그대로 사용)
        * Stage 2:
            - encoder_local: freeze → lr_scale = 0.0
            - 나머지(encoder_global / decoder / others): lr_scale = 1.0
              (Stage2 max lr는 Stage1 대비 0.333배 정도로
               LR 스케줄러에서 이미 줄어든 값을 사용)
        * Stage 3:
            - encoder_local: decoder 계열보다 더 작게 → lr_scale = stage3_encoder_local_lr_scale
            - 나머지(encoder_global / decoder / others): lr_scale = 1.0

    Args:
        stage (int): 현재 Stage 번호 (1, 2, 3).
        args (argparse.Namespace):
            - stage3_encoder_local_lr_scale (float):
                Stage3에서 로컬 인코더 그룹에만 추가로 줄일 비율.

    Returns:
        Dict[str, float]: 그룹 이름 → lr_scale 값.
            예: {"encoder_local": 0.0, "encoder_global": 1.0, ...}
    """
    s3_local: float = float(getattr(args, "stage3_encoder_local_lr_scale", 0.1))

    if stage <= 1:
        return {
            "encoder_local": 1.0,
            "encoder_global": 1.0,
            "decoder": 1.0,
            "others": 1.0,
        }
    elif stage == 2:
        # Stage2: 로컬 인코더만 freeze, 나머지는 Stage2 max lr 그대로 사용
        return {
            "encoder_local": 0.0,
            "encoder_global": 1.0,
            "decoder": 1.0,
            "others": 1.0,
        }
    else:
        # Stage3:
        #   - encoder_local: Stage3 max lr * s3_local
        #   - 나머지: Stage3 max lr 그대로 사용
        return {
            "encoder_local": s3_local,
            "encoder_global": 1.0,
            "decoder": 1.0,
            "others": 1.0,
        }


def _update_model_stage_and_lr_scale(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    """현재 전역 스텝에 맞춰 Stage 전환과 그룹별 lr_scale 값을 갱신한다.

    동작 요약:
        1) _get_stage_for_step(...) 으로 이번 스텝에 사용할 Stage(1/2/3)를 구한다.
        2) Stage 번호가 이전과 다르면
           - model.set_stage(stage)를 호출해 Encoder 쪽 freeze/unfreeze 를 반영한다.
           - optimizer.param_groups[*]["lr_scale"] 에
             해당 Stage 에 맞는 비율을 다시 써 넣는다.

    주의:
        - model 은 DDP/DeepSpeed 래퍼일 수 있으므로,
          실제 모듈은 getattr(model, "module", model) 로 꺼낸다.
        - Diffusion_Planner.set_stage(...) 안에서는
          Encoder 로컬 인코더 파라미터의 requires_grad 를 조정한다.
          Decoder 쪽은 항상 requires_grad=True 로 유지된다.
        - lr_scale 값은 _apply_stage_lr_scale_to_optimizer(...) 가
          scheduler.step() 이후에 실제 lr 에 곱해 사용하는 값이다.

    Args:
        model: 학습 중인 모델(nn.Module 또는 DDP/DeepSpeed 래퍼).
        optimizer: torch.optim.Optimizer 인스턴스.
            각 param_group 에 "stage_group" 키가 있어야 하고,
            여기에 "encoder_local"/"encoder_global"/"decoder"/"others" 중 하나가 들어간다.
        args: 학습 설정/상태 Namespace.
            - _global_update_step
            - _stage1_end_step / _stage2_end_step
            - stage2_lr_scale / stage3_encoder_local_lr_scale
            - _current_stage (내부적으로 현재 Stage 번호를 저장).
    """
    new_stage: int = _get_stage_for_step(args)
    old_stage: int = int(getattr(args, "_current_stage", 0))

    if new_stage == old_stage:
        return

    base_model: nn.Module = getattr(model, "module", model)
    if hasattr(base_model, "set_stage"):
        base_model.set_stage(new_stage)

    args._current_stage = int(new_stage)

    scales: Dict[str, float] = _get_stage_group_lr_scales(new_stage, args)
    for group in optimizer.param_groups:
        group_name: str = str(group.get("stage_group", "others"))
        lr_scale: float = float(
            scales.get(group_name, scales.get("others", 1.0)))
        group["lr_scale"] = lr_scale


def _apply_stage_lr_scale_for_deepspeed(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
) -> None:
    """DeepSpeed 엔진 사용 시에도 Stage별 lr_scale을 실제 lr 값에 반영한다.

    동작 요약:
      1) model 이 DeepSpeedEngine 이면 model.optimizer 를 우선 사용한다.
      2) 그렇지 않으면 인자로 넘어온 optimizer 를 사용한다.
      3) 선택된 옵티마이저에 대해 _apply_stage_lr_scale_to_optimizer(...) 를 호출한다.

    주의:
        - DeepSpeedEngine.step() 안에서 이미 scheduler.step() 이 호출된 뒤이므로,
          여기서의 lr_scale 적용은 "다음 step부터" 사용할 lr 값에 반영된다.
          일반 PyTorch 경로와 비교하면 한 step 정도 시점 차이는 있지만,
          Stage1/2/3 구간별 lr 비율 자체는 동일하게 유지된다.
    """
    ds_optimizer: Optional[torch.optim.Optimizer] = None

    # 1) DeepSpeedEngine 내부 optimizer(model.optimizer)가 있으면 우선 사용
    inner_optimizer: Any = getattr(model, "optimizer", None)
    if inner_optimizer is not None and hasattr(inner_optimizer, "param_groups"):
        ds_optimizer = inner_optimizer  # type: ignore[assignment]
    elif optimizer is not None and hasattr(optimizer, "param_groups"):
        # fallback: 인자로 넘어온 optimizer 사용
        ds_optimizer = optimizer

    if ds_optimizer is None:
        # optimizer 정보를 찾을 수 없으면 조용히 패스
        return

    _apply_stage_lr_scale_to_optimizer(ds_optimizer)  # type: ignore[arg-type]


def _apply_stage_lr_scale_to_optimizer(
    optimizer: torch.optim.Optimizer,) -> None:
    """param_group 의 lr 값에 lr_scale 을 곱해, Stage 반영 최종 lr 로 만든다.

    사용 시점:
        - scheduler.step() 이 각 그룹의 "기본 lr" 을 갱신한 뒤
        - 그 값에 param_group["lr_scale"] 을 곱해 Stage2/3 비율을 반영하기 위해 호출한다.

    전제:
        - optimizer.param_groups[*]["lr"] : 현재 스케줄러가 정한 기본 lr (스칼라).
        - optimizer.param_groups[*]["lr_scale"] : Stage/그룹별 비율(0.0 ~ 1.0 이상).

    처리:
        - lr_scale == 1.0 인 그룹은 그대로 두고,
        - 그 외 그룹은 lr ← lr * lr_scale 로 덮어쓴다.

    Args:
        optimizer: torch.optim.Optimizer 인스턴스.
            각 param_group 의 "params" 리스트 안에는
            다양한 shape 의 파라미터 텐서들
            (예: (out_dim, in_dim), (dim,), (C_out, C_in, kH, kW)) 이 들어있다.
    """
    for group in optimizer.param_groups:
        lr_scale: float = float(group.get("lr_scale", 1.0))
        if lr_scale == 1.0:
            continue
        group["lr"] = group["lr"] * lr_scale


# =====================================================================
# 아래부터 train_epoch 내부를 역할별 함수로 분리
# =====================================================================


def _init_global_step_and_total_updates(
    args: argparse.Namespace,
    batch_num_in_epoch: int,
) -> int:
    """전역 스텝 상태를 초기화하고, 전체 업데이트 스텝 수를 계산한다.

    우선 순서:
      1) Stage 스케줄 설정(_setup_stage_schedule)에서
         이미 전체 step 수(args._total_update_steps)를 계산해둔 경우
         그 값을 그대로 사용한다.
      2) 그 값이 없다면, 이전 방식대로
         train_epochs * batch_num_in_epoch 를 이용해 근사한다.

    Args:
        args:
            학습 설정/상태를 담고 있는 Namespace.
            - _global_update_step: 없으면 0으로 새로 만든다.
            - _total_update_steps: 있으면 전체 step 수로 사용.
        batch_num_in_epoch:
            현재 epoch에서 배치 개수. len(data_loader).

    Returns:
        int:
            전체 학습 동안의 총 update step 수.
    """
    if not hasattr(args, "_global_update_step"):
        args._global_update_step = 0

    total_updates_from_stage: int = int(getattr(args, "_total_update_steps", 0))
    if total_updates_from_stage > 0:
        return total_updates_from_stage

    batch_num_in_all_epoch: int = max(
        1,
        int(args.train_epochs) * int(batch_num_in_epoch))
    return batch_num_in_all_epoch


def _apply_augmentation(
    inputs: Dict[str, torch.Tensor],
    ego_future_gt_3_dim: torch.Tensor,  # (B, Tf, 3)
    near_future_gt_3_dim: torch.Tensor,  # (B, A, Tf, 3)
    aug: Optional[StatePerturbation],
    args: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """ego / 이웃 궤적에 대해 augmentation 을 적용한다.

    Args:
        inputs:
            모델 입력용 배치 dict. 각 값은 (B, ...) 텐서.
        ego_future_gt_3_dim:
            이고 미래 궤적. shape: (B, future_len, 3).
        near_future_gt_3_dim:
            이웃 미래 궤적. shape: (B, agent_num, future_len, 3).
        aug:
            - StatePerturbation 인스턴스이면 ego/neighbor 둘 다에 적용.
            - NPCStatePerturbation 인스턴스이면 neighbor 에만 적용.
        args:
            augmentation 이 NPCStatePerturbation 인 경우에 필요한 설정.

    Returns:
        Tuple[inputs, ego_future_gt_3_dim, near_future_gt_3_dim]:
            augmentation 이 반영된 새 텐서들.
    """
    # ✅ augmentation 전에 패딩 위치를 미리 기억해 둔다.  # shape: (B, A, Tf, 1)
    pad_mask_near: torch.Tensor = (near_future_gt_3_dim == 0).all(
        dim=-1,
        keepdim=True,
    )

    if isinstance(aug, StatePerturbation):
        inputs, ego_future_gt_3_dim, near_future_gt_3_dim = aug(
            inputs, ego_future_gt_3_dim, near_future_gt_3_dim)

    if isinstance(aug, NPCStatePerturbation):
        inputs, near_future_gt_3_dim = aug(inputs, near_future_gt_3_dim, args)

    # augmentation 과정에서 값이 바뀌었더라도,
    # 원래 패딩이었던 위치는 다시 전부 0으로 되돌린다.
    near_future_gt_3_dim = near_future_gt_3_dim.masked_fill(pad_mask_near, 0.0)

    return inputs, ego_future_gt_3_dim, near_future_gt_3_dim


def _build_near_future_4dim_and_mask(
    near_future_gt_3_dim: torch.Tensor,) -> Tuple[torch.Tensor, torch.Tensor]:
    """yaw를 cos/sin으로 확장하고, 비어 있는 구간 마스크를 만든다.

    처리 내용:
      1) (x, y, yaw) 궤적에서 yaw를 cos/sin 두 값으로 나눠
         (x, y, cos, sin) 4차원 궤적을 만든다.
      2) (x, y, yaw)가 모두 0인 프레임을 "비어 있는 프레임"으로 보고
         near_future_mask 를 True/False로 만든다.
      3) 비어 있는 위치의 4차원 값은 0으로 덮어쓴다.

    Args:
        near_future_gt_3_dim:
            - shape: (B, agent_num, future_len, 3)
            - [..., 0:2] = (x, y), [..., 2] = yaw.

    Returns:
        near_future_gt_4_dim:
            - shape: (B, agent_num, future_len, 4)
            - [x, y, cos(yaw), sin(yaw)].
        near_future_mask:
            - shape: (B, agent_num, future_len)
            - True 이면 해당 프레임은 완전히 비어 있는 프레임.
    """
    # near_future_gt_3_dim: (B, A, Tf, 3)
    near_future_mask: torch.Tensor = torch.sum(
        torch.ne(near_future_gt_3_dim[..., :3], 0),
        dim=-1,
    ) == 0  # (B, A, Tf)

    # yaw: (B, A, Tf)
    yaw: torch.Tensor = near_future_gt_3_dim[..., 2]

    # near_future_gt_4_dim: (B, A, Tf, 4)
    near_future_gt_4_dim: torch.Tensor = torch.cat(
        [
            near_future_gt_3_dim[..., :2],  # (B, A, Tf, 2)
            torch.stack(
                [
                    yaw.cos(),  # (B, A, Tf)
                    yaw.sin(),  # (B, A, Tf)
                ],
                dim=-1,  # -> (B, A, Tf, 2)
            ),
        ],
        dim=-1,
    )

    near_future_gt_4_dim[near_future_mask] = 0.0

    if not torch.isfinite(near_future_gt_4_dim).all():
        raise ValueError("Non-finite values detected in near_future_gt_4_dim")

    return near_future_gt_4_dim, near_future_mask


def _compute_loss_dict(
    loss_dict: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    model: nn.Module,
    batch_num_in_all_epoch: int,
) -> Dict[str, torch.Tensor]:
    """diffusion 손실과 feasible 가중합까지 포함한 loss_dict dict 를 계산한다.

    여기서 batch_num_in_all_epoch 인자는
    이제 "전체 update step 수"를 뜻한다고 보면 된다.
    (_setup_stage_schedule 에서 계산된 args._total_update_steps 를 넘겨주는 용도)

    Args:
        loss_dict:
            diffusion_loss_func 가 채운 기본 손실 딕셔너리.
        args:
            - _global_update_step: 지금까지 진행된 전체 step 수.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        norm_inputs:
            관측값 정규화가 적용된 입력 dict. 각 텐서 shape: (B, ...).
        batch_num_in_all_epoch:
            전체 update step 수. 보통 args._total_update_steps 값을 그대로 사용한다.

    Returns:
        loss_dict:
            진행도/가중치/최종 loss("loss")까지 포함된 dict.
    """

    total_steps_for_progress: int = max(1, int(batch_num_in_all_epoch))

    # 진행도 0~1 계산 (0 step → 0, 마지막 step 근처 → 1)
    progress: float = min(
        1.0,
        args._global_update_step / float(max(1, total_steps_for_progress - 1)),
    )
    w_dir, w_int, w_const = FeasibleProjector.loss_weights_by_progress(progress)

    if not args.use_direct_loss:
        w_dir = 0.0
        w_int = 1.0

    # 진행도/가중치 기록(평균 로그용)
    device = next(model.parameters()).device
    loss_dict["learn_progress"] = torch.tensor(float(progress), device=device)
    loss_dict["direct_loss_weight"] = torch.tensor(float(w_dir), device=device)
    loss_dict["int_loss_weight"] = torch.tensor(float(w_int), device=device)
    loss_dict["const_loss_weight"] = torch.tensor(float(w_const), device=device)

    # 개별 손실이 없을 수도 있으니 기본값 0 텐서로 처리
    l_dir = loss_dict.get("neighbor_prediction_loss",
                          torch.tensor(0.0, device=device))
    l_int = loss_dict.get("integration_loss", torch.tensor(0.0, device=device))
    l_con = loss_dict.get("constraint_loss", torch.tensor(0.0, device=device))

    # 최종 손실 합성
    loss_dict["loss"] = w_dir * l_dir + w_int * l_int + w_const * l_con

    return loss_dict


def _backward_and_step(
    loss_dict: Dict[str, torch.Tensor],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    args: argparse.Namespace,
) -> float:
    """손실에 대해 역전파를 수행하고, 한 번의 optimizer step 을 끝낸다.

    크게 두 가지 경우를 나눠 처리한다.

    1) DeepSpeed 엔진 사용 시 (use_deepspeed=True):
        - model.backward(loss) 를 호출해 DeepSpeed가 관리하는 방식으로
          기울기 계산과 통신을 진행한다.
        - model.step() 을 호출해 내부 옵티마이저, 스케줄러, grad clipping 을 처리한다.
        - 이 경우 Stage별 lr_scale 은 build_deepspeed_config 의
          gradient_clipping, lr_scheduler 설정에 의존한다.

    2) 일반 PyTorch / DDP 사용 시:
        - loss.backward() 로 기울기를 계산한다.
        - args.max_grad_norm > 0 일 때 nn.utils.clip_grad_norm_ 로
          모든 파라미터에 대해 클리핑을 적용한다.
        - scheduler.step() 으로 각 param_group 의 기본 lr 을 갱신한다.
        - _apply_stage_lr_scale_to_optimizer(...) 로
          그룹별 lr_scale 을 곱해 최종 lr 을 만든다.
        - optimizer.step() 으로 파라미터를 한 스텝 업데이트한다.

    Args:
        loss_dict:
            "loss" 키를 포함하는 손실 딕셔너리.
            loss_dict["loss"] 텐서 shape: () (스칼라).
        model:
            학습 중인 모델.
            - 일반 모드: nn.Module 또는 DDP 래퍼.
            - DeepSpeed 모드: deepspeed.DeepSpeedEngine.
        optimizer:
            torch.optim.Optimizer 인스턴스.
        scheduler:
            학습률 스케줄러 객체. 일반 모드에서만 step() 을 직접 호출한다.
        args:
            학습 설정/상태 Namespace.
            - use_deepspeed (bool)
            - max_grad_norm (float)

    Returns:
        float: loss_dict["loss"].item() 값 (로깅용 스칼라).
    """
    # total_loss는 scalar float 값
    total_loss: float = float(loss_dict["loss"].item())
    loss_tensor: torch.Tensor = loss_dict["loss"]

    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False)) \
        and hasattr(model, "backward") and hasattr(model, "step")

    # 한 곳에서만 클리핑 기준을 정한다.
    max_grad_norm: float = float(getattr(args, "max_grad_norm", 0.0))

    if use_deepspeed:
        # DeepSpeed 엔진을 사용하는 경우:
        # gradient_clipping 값은 build_deepspeed_config()에서
        # ds_config["gradient_clipping"] = max_grad_norm 으로 넘어간다.
        # 여기서는 따로 clip_grad_norm을 호출하지 않는다.
        """ model.backward(loss_tensor) # deepseed 전용 역전파
        각 GPU가 자기 배치에 대한 기울기를 먼저 계산한다.
            GPU0: g0(W1), g0(W2), g0(W3), g0(W4)
            GPU1: g1(W1), g1(W2), g1(W3), g1(W4)
        여기까진 “각자 전체 기울기를 한 번씩 계산했다”고 보면 된다.
            (이 단계는 ZeRO-2라도 어쩔 수 없이 한 번 거치는 단계)

        2. 그 다음 “기울기를 나누고 합치는 통신 단계”가 들어간다.
        
           * GPU0와 GPU1이 서로 기울기 조각을 주고받아서:
        
             * GPU0는 W1, W2에 대한
               **(g0 + g1)의 합**만 남기고 W3, W4에 대한 기울기는 버린다.
             * GPU1은 W3, W4에 대한
               **(g0 + g1)의 합**만 남기고 W1, W2에 대한 기울기는 버린다.
        
           즉, **최종적으로**:
        
           * GPU0: (합쳐진 기울기) g(W1), g(W2) 만 보관
           * GPU1: (합쳐진 기울기) g(W3), g(W4) 만 보관
        
           → 이게 “기울기를 GPU 사이에 나눠서 가진다”는 뜻이다.
           (옵티마 상태도 비슷하게 “나눠서 저장”한다.)

# algather_partitions = False 일 때
    1. GPU0, GPU1이 각각 **자기 배치에 대해 전체 기울기**를 계산 (g0, g1).
    
    2. GPU0와 GPU1이 통신해서:
       * **먼저 전체 기울기를 서로 합친다.**
         * 결과적으로 GPU0, GPU1 둘 다
           * g_sum(W1), g_sum(W2), g_sum(W3), g_sum(W4)
             를 잠깐씩 다 들고 있을 수 있다.
       * 그 다음,
         * “나는 W1,W2만 쥐고 있을게” / “나는 W3,W4만 쥐고 있을게” 식으로
           기울기와 옵티마 상태를 다시 나누고 정리.
    
       → 즉, **중간에 “모든 기울기 합본을 한 번씩 다 들고 있는 순간”이 있을 수 있다.**
       그래서 메모리 사용량 관점에서 조금 덜 효율적인 쪽.

# allgather_partitions = True 일 때
    2. 그런데 여기서는, **“전체 합본을 두 군데 다 오래 들고 있게 만들지 않고”**
       바로 “나눠진 형태” 위주로 유지하려고 한다.
    
       예를 들어 개념적으로는:
    
       * 단계 1: GPU0/1이 서로 기울기를 교환하면서,
    
         * GPU0는 W1,W2 부분에 대한 `g0+g1`만 남기고,
         * GPU1은 W3,W4 부분에 대한 `g0+g1`만 남긴다.
       * 이때 “모든 파라미터에 대한 합본”을 각 GPU가 오래 들고 있는 순간을 줄이고,
         * 바로 “나눠진 합본”만 남기는 쪽으로 통신을 설계하는 것.
    
       → 실제 구현은 더 복잡하지만,
       **“합친 전체 기울기를 각 GPU가 길게 들고 있지 않는다”**는 방향으로 생각하면 된다.
        """
        model.backward(loss_tensor)

        _apply_stage_lr_scale_for_deepspeed(
            model=model,
            optimizer=optimizer,
        )
        """ model.step()
3. 옵티마 단계에서:
   * GPU0는 자신이 가진 파라미터에 대해서만 갱신
     * W1, W2 를 g(W1), g(W2) 와 자기 쪽 옵티마 상태를 써서 업데이트
   * GPU1은 W3, W4 를 자기 쪽 기울기/옵티마로 업데이트

4. 업데이트가 끝나면,
   * GPU0와 GPU1이 서로 **업데이트된 W1~W4 전체를 다시 맞춘다**.
     (브로드캐스트 혹은 비슷한 방식으로 동기화)
   * 그래서 **step이 끝난 후에는** 다시
     * GPU0: W1~W4 전체 최신 버전
     * GPU1: W1~W4 전체 최신 버전

        """
        model.step()
    else:
        # 일반 PyTorch / DDP 모드
        """ loss_tensor.backward()
이때 DDP가 **파라미터별 gradient가 만들어지는 순간마다 후크를 걸어** 다음을 수행:

* GPU0는 자기 gradient `g0`를 가지고 있음.
* GPU1는 자기 gradient `g1`를 가지고 있음.
* PyTorch DDP 내부에서:
  * 각 파라미터마다
    `g_avg = (g0 + g1) / 2` 를 만들기 위해
    GPU0와 GPU1이 서로 값을 주고받고, 더하고, 나눔.
* 그 결과:
  * GPU0의 해당 파라미터 gradient = `g_avg`
  * GPU1의 해당 파라미터 gradient = `g_avg`
즉, **backward가 끝났을 때, 두 GPU의 gradient는 완전히 동일**하게 맞춰져 있음.
이 과정이 이 설정에서 **가장 큰 통신 비용**이야.        
        """
        loss_tensor.backward()
        """
        아래 3줄 코드에서는 GPU끼리 통신하지 않음.
        """
        # max_grad_norm > 0 일 때만 클리핑 수행
        if max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # 1) 스케줄러로 기본 lr 갱신
        scheduler.step()
        # 2) Stage/그룹별 lr_scale 적용
        _apply_stage_lr_scale_to_optimizer(optimizer)
        optimizer.step()

    return total_loss


def _apply_weight_decay_warmdown(optimizer: torch.optim.Optimizer) -> None:
    """각 파라미터 그룹의 lr 비율에 맞춰 weight_decay 를 선형으로 조정한다.

    Args:
        optimizer:
            AdamW 옵티마이저. 각 param_group 에 "wd_max", "lr_max" 키가 설정되어 있어야 한다.
    """
    for pg in optimizer.param_groups:
        wd_max = pg.get("wd_max", None)
        lr_max = pg.get("lr_max", None)
        if wd_max is None or lr_max is None:
            continue
        if wd_max > 0.0 and lr_max > 0.0:
            pg["weight_decay"] = wd_max * (pg["lr"] / lr_max)


def _update_ema_if_needed(ema: Optional[object], model: nn.Module) -> None:
    """EMA 객체가 있으면 한 스텝 업데이트한다.

    Args:
        ema:
            timm.utils.ModelEma 와 같은 EMA 래퍼 또는 None.
        model:
            현재 학습 중인 모델. ema.update(model) 의 인자로 사용된다.
    """
    if ema is not None:
        # DDP / DeepSpeed 래퍼가 씌워져 있으면 .module을 사용해 실제 모듈 기준으로 EMA를 업데이트한다.
        src_model: nn.Module = getattr(model, "module", model)
        ema.update(src_model)


# =====================================================================


def train_epoch(
    data_loader,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    ema: Optional[object],
    scheduler,
    aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행한다.

    처리 흐름:
        1) 모델을 train 모드로 두고, DDP 사용 시 CUDA 동기화.
        2) 현재 전역 스텝 값에 맞춰 Stage / lr_scale 을 한 번 초기화한다.
        3) 전체 업데이트 스텝 수(batch_num_in_all_epoch)를 계산한다.
        4) 각 배치에 대해 다음 순서를 수행한다.
            a) _update_model_stage_and_lr_scale(...) 로
               Stage1→2→3 전환 시점에 맞춰 Encoder freeze + lr_scale 갱신.
            b) _prepare_batch_for_device(...) 로
               배치를 device 로 옮기고, agent/lane 축을 상한에 맞게 자름.
               · ego_future_gt_3_dim: (B, future_len, 3)
               · near_future_gt_3_dim: (B, agent_num, future_len, 3)
            c) _apply_augmentation(...) 으로 ego / neighbor 궤적에 증강 적용.
            d) _build_near_future_4dim_and_mask(...) 로
               (x, y, yaw) → (x, y, cos, sin) + mask 생성.
            e) args.observation_normalizer(...) 로 입력을 정규화.
            f) diffusion_loss_func(...) 로 기본 손실(neighbor / integration / constraint 등) 계산.
            g) _compute_loss_dict(...) 으로 진행도에 따라
               direct / integration / constraint 가중치를 섞어 최종 loss 만들기.
            h) _backward_and_step(...) 으로 역전파 + optimizer/scheduler step 수행.
            i) _apply_weight_decay_warmdown(...) 으로 lr 비율에 맞춰 weight_decay 보정.
            j) _update_ema_if_needed(...) 로 EMA 모델이 있을 경우 한 스텝 업데이트.
            k) loss_dict 를 epoch_loss_dict_list 에 모으고,
               args._global_update_step 을 1 증가시킨다.
        5) get_epoch_mean_loss(...) 로 배치별 loss 를 평균낸다.
        6) DDP 사용 시 ddp.reduce_and_average_losses(...) 로
           rank 0 기준 전체 평균을 계산한다.

    Args:
        data_loader: PyTorch DataLoader.
            각 요소는 collate_fn(DiffusionPlannerCollate)이 만든
            배치 dict (key: str, value: Tensor) 이다.
        model: 학습 중인 모델(nn.Module 또는 DDP/DeepSpeed 래퍼).
        optimizer: torch.optim.Optimizer 인스턴스.
        args: 학습 설정/상태 Namespace.
        ema: timm.utils.ModelEma 와 같은 EMA 래퍼 또는 None.
        scheduler: 학습률 스케줄러 객체.
        aug: StatePerturbation 또는 NPCStatePerturbation, 또는 None.

    Returns:
        Tuple[Dict[str, float], float]:
            - epoch_mean_loss:
                손실 항목별 평균 값 딕셔너리.
                예: {"loss": 0.42, "neighbor_prediction_loss": 0.3, ...}
            - epoch_mean_loss["loss"]:
                최종 학습 손실 스칼라(float).
    """
    epoch_loss_dict_list: List[Dict[str, torch.Tensor]] = []

    model.train()

    if args.ddp:
        torch.cuda.synchronize()

    # ✅ 1) 현재 전역 step에 맞춰 Stage / lr_scale 한 번만 정리
    _update_model_stage_and_lr_scale(
        model=model,
        optimizer=optimizer,
        args=args,
    )

    # ✅ 2) 진행도(progress) 계산에 사용할 전체 update step 수 결정
    #    - Stage 스케줄에서 미리 계산된 값이 있으면 그대로 사용
    #    - 없으면 과거 방식대로 train_epochs * len(data_loader) 로 근사
    total_update_steps: int = int(getattr(args, "_total_update_steps", 0))
    if total_update_steps <= 0:
        total_update_steps = max(
            1,
            int(args.train_epochs) * max(1, len(data_loader)),
        )

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            _update_model_stage_and_lr_scale(
                model=model,
                optimizer=optimizer,
                args=args,
            )
            # 1) device 이동 + 상한 클리핑 + 정답 분리
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )
            # ego_future_gt_3_dim: (B, future_len, 3)
            ego_future_gt_3_dim: torch.Tensor = outputs["ego_future_gt_3_dim"]
            ego_future_len = ego_future_gt_3_dim.shape[1]
            assert ego_future_len == args.future_len, \
                f"ego future len mismatch: {ego_future_len} vs {args.future_len}"
            # near_future_gt_3_dim: (B, agent_num, future_len, 3)
            near_future_gt_3_dim: torch.Tensor = outputs["near_future_gt_3_dim"]

            # 2) augmentation 적용
            inputs, ego_future_gt_3_dim, near_future_gt_3_dim = \
                _apply_augmentation(
                    inputs=inputs,
                    ego_future_gt_3_dim=ego_future_gt_3_dim,
                    near_future_gt_3_dim=near_future_gt_3_dim,
                    aug=aug,
                    args=args,
                )

            # 3) near future 4차원 궤적 + mask 생성
            # near_future_gt_4_dim: (B, agent_num, future_len, 4)
            # near_future_mask:    (B, agent_num, future_len)
            near_future_gt_4_dim, near_future_mask = \
                _build_near_future_4dim_and_mask(near_future_gt_3_dim)

            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)

            # 5) loss 계산 + 역전파 + optimizer/scheduler step
            if getattr(args, "use_deepspeed", False) and hasattr(
                    model, "zero_grad"):
                model.zero_grad()
            else:
                optimizer.zero_grad(set_to_none=True)

            # base_model: DDP/DeepSpeed 래퍼 벗긴 실제 모델
            base_model = ddp.get_model(model, args.ddp)
            sde_marginal_prob = base_model.sde.marginal_prob

            # 5-1) diffusion 기본 loss 계산 (neighbor / integration / constraint 등)
            raw_loss_dict: Dict[str, torch.Tensor] = {}
            raw_loss_dict, _ = diffusion_loss_func(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                marginal_prob=sde_marginal_prob,
                near_future_gt_4_dim=near_future_gt_4_dim,
                near_future_mask=near_future_mask,
                state_normalizer=args.state_normalizer,
                loss_dict=raw_loss_dict,
                model_type=args.diffusion_model_type,  # 보통 "x_start" 또는 "score"
                observation_normalizer=args.observation_normalizer,
            )

            # 5-2) feasible weight 로 최종 loss 합성
            loss_dict: Dict[str, torch.Tensor] = _compute_loss_dict(
                loss_dict=raw_loss_dict,
                args=args,
                model=model,
                batch_num_in_all_epoch=total_update_steps,
            )

            total_loss: float = _backward_and_step(
                loss_dict=loss_dict,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                args=args,
            )
            # 6) WD warmdown, EMA 업데이트
            _apply_weight_decay_warmdown(optimizer)
            _update_ema_if_needed(ema, model)

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss_dict_list.append(loss_dict)

            # 전역 스텝 누적 (진행도 계산에 사용)
            args._global_update_step += 1

    # --- 에폭 평균 손실 계산 ---
    """ epoch_mean_loss (각 GPU마다 계산해 둠)
    {"loss": 0.42, "neighbor_prediction_loss": 0.3, ...}
    """
    epoch_mean_loss: Dict[str,
                          float] = get_epoch_mean_loss(epoch_loss_dict_list)

    if args.ddp:
        """ ddp.reduce_and_average_losses
            두 GPU가 서로 값을 더해서 합을 만든 뒤(world_size로 나눔)
            → 모든 GPU가 동일한 평균 값을 가지게 함
        통신하는 내용
            스칼라 몇 개밖에 안 되는 작은 숫자
        """
        epoch_mean_loss = ddp.reduce_and_average_losses(
            epoch_mean_loss,
            torch.device(args.device),
        )
    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")

    return epoch_mean_loss, epoch_mean_loss["loss"]
