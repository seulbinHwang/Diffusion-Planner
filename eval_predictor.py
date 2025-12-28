import argparse
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
from typing import Tuple, Any, Dict, Optional, List, Callable
import torch
import torch.nn as nn
from diffusion_planner.loss import _sanitize_norm_inputs
from tools.predictor_utils import (
    build_dataset_and_sampler,
    build_data_loader,
    create_diffusion_planner_and_ema,
    build_deepspeed_inference_config,
    init_distributed,
    get_save_path,
    prepare_wandb_resume,
    set_distributed_flag_from_env,
)
import args_util
import os
import sys, faulthandler, traceback
from diffusion_planner.utils.train_utils import set_seed
from timm.utils import ModelEma
from diffusion_planner.utils import ddp
import time
from tqdm import tqdm
from diffusion_planner.utils.target_feature import build_target_future_tensors_and_masks_for_inference

from diffusion_planner.train_epoch import _prepare_batch_for_device
from diffusion_planner.utils.normalizer import StateNormalizer
from waymo_open_dataset.protos import sim_agents_submission_pb2

AMP_DTYPE = torch.bfloat16  # A100 권장 dtype
from src.utils.wosac_utils import get_scenario_id_int_tensor, get_scenario_rollouts
from src.smart.metrics import WOSACMetrics, minADE
from torch import optim
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger


def model_validation(
    args: argparse.Namespace,
    global_rank: int,
    rank: int,
    world_size: int,
    use_deepspeed: bool,
) -> None:
    """전체 학습 파이프라인을 실행하는 상위 함수."""
    torch.cuda.empty_cache()

    # 2) seed 고정
    set_seed(args.seed + global_rank)

    # 3) augmentation, Dataset, Sampler
    batch_size = args.batch_size
    """
    validation_set: DiffusionPlannerData
    validation_sampler: DistributedSampler
    """
    validation_set, validation_sampler = build_dataset_and_sampler(
        args,
        args.validation_set,
        args.validation_set_list,
        "validation",
        world_size,
        global_rank,
    )
    validation_loader = build_data_loader(
        args,
        validation_set,
        validation_sampler,
        batch_size,
        world_size,
    )
    if args.ddp and not use_deepspeed:
        torch.distributed.barrier()

    diffusion_planner, model_ema, base_model = create_diffusion_planner_and_ema(
        args=args,
        rank=rank,
        use_deepspeed=use_deepspeed,
    )
    if use_deepspeed:
        import deepspeed
        ds_config = build_deepspeed_inference_config(args)
        ds_engine = deepspeed.init_inference(model=diffusion_planner,
                                             config=ds_config)
        diffusion_planner = ds_engine.module  # 이후 diffusion_planner로 그대로 추론

    # 6) 체크포인트 재개
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id,
     train_epochs, allow_val_change) = _maybe_resume_from_checkpoint(
         args=args,
         diffusion_planner=diffusion_planner,
         optimizer=optimizer,
         scheduler=scheduler,
         model_ema=model_ema,
         global_rank=global_rank,
         use_deepspeed=use_deepspeed,
     )
    args._global_update_step = 0

    min_ade = minADE()
    wosac_metrics = WOSACMetrics("val_closed")

    run_validation_loop(
        args=args,
        diffusion_planner=diffusion_planner,
        model_ema=model_ema,
        validation_loader=validation_loader,
        wosac_metrics=wosac_metrics,
        min_ade=min_ade,
    )

    _finalize_training_cleanup()


def _finalize_training_cleanup():
    # 1) 분산 학습일 때만 barrier 호출
    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    if ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()


def run_validation_loop(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    validation_loader: DataLoader,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
) -> None:
    """전체 epoch 루프를 돌면서 학습, 속도 측정, 로깅, 체크포인트 저장을 수행한다."""
    elapsed_training_time_hour: float = 0.0
    # 전체 업데이트 스텝 수 설정 및 global step 초기화 보장
    batch_num_in_one_val_epoch: int = max(1, len(validation_loader))
    # ✅ (중요) epoch 시작 전에 sampler epoch를 먼저 세팅
    # - resume(init_epoch>0) 시에도 첫 epoch부터 올바른 shuffle이 나오도록 함
    # Dict[str, torch.Tensor]
    (epoch_wosac_metrics, epoch_elapsed_time_sec) = validate_one_epoch(
        epoch=0,
        total_epochs=1,
        validation_loader=validation_loader,
        diffusion_planner=diffusion_planner,
        args=args,
        model_ema=model_ema,
        batch_num_in_one_val_epoch=batch_num_in_one_val_epoch,
        wosac_metrics=wosac_metrics,
        min_ade=min_ade,
    )
    # print "epoch_elapsed_time_sec"
    print(f"[Validation] completed in "
          f"{epoch_elapsed_time_sec:.2f} sec.")


def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
    batch_num_in_one_val_epoch: int,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
) -> Tuple[Dict[str, torch.Tensor], float]:
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{total_epochs}")

    epoch_t0 = time.perf_counter()
    epoch_wosac_metrics: Dict[str, torch.Tensor] = validation_epoch(
        validation_loader,
        diffusion_planner,
        args,
        model_ema,
        batch_num_in_one_val_epoch,
        wosac_metrics,
        min_ade,
    )
    """
CUDA를 쓰는 경우, PyTorch가 캐시로 잡아둔 GPU 메모리(다음 연산 재사용용)를 OS/GPU 쪽에 반납
메모리 피크를 낮춰 OOM 위험을 줄이려는 목적이지만, 너무 자주 하면 약간의 성능 손해가 날 수 있어요.
    """
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    """
torch.cuda.synchronize() : 현재 GPU에서 큐에 쌓인 비동기 연산이 다 끝날 때까지 대기
torch.distributed.barrier() :
    모든 rank(프로세스)가 이 지점에 도착할 때까지 서로 기다리게 해서, 어떤 rank만 앞서가거나(예: 저장/로그/다음 epoch) 뒤처지는 상황을 막습니다.
    """
    if args.ddp:
        torch.cuda.synchronize()
        torch.distributed.barrier()

    epoch_elapsed_time_sec = time.perf_counter() - epoch_t0
    return epoch_wosac_metrics, epoch_elapsed_time_sec


def validation_epoch(
    data_loader,
    model: nn.Module,
    args: argparse.Namespace,
    ema: Optional[object],
    batch_num_in_one_val_epoch: int,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
) -> Dict[str, torch.Tensor]:
    model.eval()
    if args.ddp:
        torch.cuda.synchronize()
    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # 1) device 이동 + 상한 클리핑 + 정답 분리
            inputs, outputs = _prepare_batch_for_device(
                batch,
                device=args.device,
                args=args,
            )
            # 4) 관측 정규화
            # norm_inputs: 각 value shape = (B, ...)
            norm_inputs: Dict[str, torch.Tensor] = \
                args.observation_normalizer(inputs)

            # 5-1) diffusion 기본 loss 계산 (neighbor / integration / constraint 등)
            validate_func(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                state_normalizer=args.state_normalizer,
                observation_normalizer=args.observation_normalizer,
                wosac_metrics=wosac_metrics,
                min_ade=min_ade,
            )
            if args.ddp:
                torch.cuda.synchronize()

            # 전역 스텝 누적 (진행도 계산에 사용)
    """
    시나리오 개수로 나눠 평균을 내고,
    “큰 항목별 점수(종합/움직임/상호작용/지도 등)”로 정리합니다.
    """
    epoch_wosac_metrics: Dict[str, torch.Tensor] = wosac_metrics.compute()
    epoch_wosac_metrics["val_closed/ADE"] = min_ade.compute()
    wosac_metrics.reset()
    min_ade.reset()

    return epoch_wosac_metrics


def validate_func(
    args: Any,
    model: nn.Module,
    norm_inputs: Dict[str, torch.Tensor],
    state_normalizer: StateNormalizer,
    observation_normalizer: Any,
    wosac_metrics: WOSACMetrics,
    min_ade: minADE,
) -> None:
    # TODO: ema 모델로 inference 하고 싶음
    # norm_inputs 의 각 텐서 NaN/Inf 체크
    norm_inputs = _sanitize_norm_inputs(norm_inputs)
    # target_future_valid : # (B, (1+)Pnn, future_len)  True=유효
    target_future_valid = build_target_future_tensors_and_masks_for_inference(
        args,
        norm_inputs,
        args.future_len,
    )
    # target_future_valid 에 단 하나도 False가 있으면 안됨.
    assert torch.all(target_future_valid), "target_future_valid에 False 값이 있습니다."
    norm_inputs["target_future_valid"] = target_future_valid
    batch_size = norm_inputs["ego_agent_past"].shape[0]
    is_ds_engine = hasattr(model, "backward") and hasattr(
        model, "step") and hasattr(model, "module")

    ROLLOUT_NUMBER = 32
    # target_scenario_rollouts_np : (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 4)
    target_scenario_rollouts_np = []  # 원소 하나씩은 : (B, (1+)Pnn, future_len, 4)
    for rollout_idx in range(ROLLOUT_NUMBER):
        """
        TODO
            rollout_idx 마다 다른 샘플링 noise 를 사용하여 예측을 수행해야 합니다.
        """
        norm_inputs_copy = _clone_norm_inputs_for_rollout(
            norm_inputs,
            sanity_check_tensor_storage=False,
        )
        unnorm_origin_world_pose = norm_inputs_copy[
            "origin_world_pose"]  # (B, 4)

        target_joint_scene = []  # 최종 모습 (B, (1+)Pnn, future_len, 4)
        for step_idx in range(args.future_len):
            if args.use_deepspeed:
                assert is_ds_engine, "use_deepspeed=True 인데 model 이 DS engine 아님"
                _, decoder_output = model(
                    norm_inputs_copy)  # DS가 torch_autocast로 처리
            else:
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    _, decoder_output = model(norm_inputs_copy)
            assert args.do_ego_predict, "현재 ego 예측은 항상 활성화되어 있어야 합니다."
            # TODO: 나중에 더 고급진 방법으로 업데이트할 수도? (지금 반영하고 싶은건 아님)
            normed_trajectories = decoder_output.get(
                "integrated_trajectory")  # (B, (1+)Pnn, 1+T, 4)
            # (B, 1+T, 4)
            normed_ego_trajectories = normed_trajectories[:, 0, :, :]
            # (B, Pnn, 1+T, 4)
            normed_near_trajectories = normed_trajectories[:, 1:, :, :]
            # TODO: 시간 간격에 따라 제대로 표시하기 (지금 반영하고 싶은건 아님)
            normed_ego_next_pose = normed_ego_trajectories[:, 1, :]  # (B, 4)
            normed_near_next_pose = normed_near_trajectories[:, :,
                                                             1, :]  # (B, Pnn, 4)
            norm_inputs_copy = _update_merged_inputs(norm_inputs_copy,
                                                     normed_ego_next_pose,
                                                     normed_near_next_pose)

            # (B, (1+)Pnn, 4)
            normed_target_next_pose = torch.cat(
                [normed_ego_next_pose[:, None, :], normed_near_next_pose],
                dim=1)
            # (B, (1+)Pnn, 4)
            unnorm_target_next_pose = state_normalizer.inverse(
                normed_target_next_pose)
            # target_future_valid : (B, (1+)Pnn, future_len)  True=유효
            # target_step_valid : (B, (1+)Pnn)
            # target_step_valid = target_future_valid[:, :, step_idx]
            # unnorm_target_next_pose[~target_step_valid] = 0.
            """
            unnorm_target_next_pose_flat: (B * (1+)Pnn, 4) # 지금 ego 좌표계 기준인데, world 좌표계로 바꿔줘야 함.

            unnorm_origin_world_pose: (B, 4)

            unnorm_target_next_pose_world_flat: (B * (1+)Pnn, 4)
            """
            unnorm_target_next_pose_flat = unnorm_target_next_pose.reshape(
                -1, unnorm_target_next_pose.shape[2])  # (B * (1+)Pnn, 4)
            unnorm_target_next_pose_world_flat = _covert_from_ego_to_world(
                unnorm_target_next_pose_flat, unnorm_origin_world_pose)
            unnorm_target_next_pose_world = unnorm_target_next_pose_world_flat.reshape(
                unnorm_target_next_pose.shape[0],
                unnorm_target_next_pose.shape[1],
                unnorm_target_next_pose.shape[2],
            )  # (B, (1+)Pnn, 4)
            # unnorm_target_next_pose_world[~target_future_valid] = 0.
            target_joint_scene.append(unnorm_target_next_pose_world)
            unnorm_origin_world_pose = unnorm_target_next_pose_world[:, 0, :]

        target_joint_scene = torch.stack(target_joint_scene,
                                         dim=2)  # (B, (1+)Pnn, future_len, 4)
        target_scenario_rollouts_np.append(target_joint_scene)

    target_scenario_rollouts_np = torch.stack(
        target_scenario_rollouts_np,
        dim=2)  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 4)
    """
    target_scenario_rollouts : List[sim_agents_submission_pb2.ScenarioRollouts] # 길이: B
    """
    pred_traj = target_scenario_rollouts_np[:, :, :, :, :
                                            2]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 2)
    pred_traj = pred_traj.reshape(
        pred_traj.shape[0] * pred_traj.shape[1],
        pred_traj.shape[2],
        pred_traj.shape[3],
        pred_traj.shape[4],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step, 2]
    pred_head_cos_yaw = target_scenario_rollouts_np[:, :, :, :,
                                                    2]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head_sin_yaw = target_scenario_rollouts_np[:, :, :, :,
                                                    3]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head = torch.atan2(
        pred_head_sin_yaw,
        pred_head_cos_yaw)  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head = pred_head.reshape(
        pred_head.shape[0] * pred_head.shape[1],
        pred_head.shape[2],
        pred_head.shape[3],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step]
    # List[ScenarioRollouts]

    scenario_id: List[str] = norm_inputs[
        "scenario_id"]  # List[str], 길이 Batch size
    assert isinstance(scenario_id, list), "scenario_id는 List[str] 타입이어야 합니다."
    assert batch_size == len(
        scenario_id), "batch_size와 scenario_id 길이가 맞지 않습니다."
    target_id = norm_inputs["target_id"]  # (B, (1+)Pnn)
    target_id = target_id.reshape(-1)  # [n_agent = B * (1+)Pnn]
    """
    agent_batch 를 구합니다. (n_agent 길이, 각 agent 가 속한 batch 인덱스)
    예: batch_size=2, (1+)Pnn=3 이면,
        agent_batch = [0,0,0,1,1,1]
    """
    agent_batch = []
    one_or_Pnn = target_future_valid.shape[1]
    for batch_idx in range(batch_size):
        agent_batch.extend([batch_idx] * one_or_Pnn)
    agent_batch = torch.tensor(agent_batch,
                               dtype=torch.long,
                               device=pred_traj.device)  # [n_agent]

    target_z = norm_inputs["target_z"]  # (B, (1+)Pnn)
    target_z = target_z.reshape(-1)  # [n_agent = B * (1+)Pnn]

    device = pred_traj.device
    # scenario_rollouts: List[sim_agents_submission_pb2.ScenarioRollouts]
    scenario_rollouts = get_scenario_rollouts(
        scenario_id=get_scenario_id_int_tensor(
            scenario_id, device),  # [n_scenario, n_str_length]
        agent_id=target_id,  # [n_agent]
        agent_batch=agent_batch,  # [n_agent]
        pred_traj=pred_traj,  # [n_agent, n_rollout, n_step, 2]
        pred_z=target_z,  # [n_agent, n_rollout, n_step]
        pred_head=pred_head,  # [n_agent, n_rollout, n_step]
    )
    unnorm_ego_future_gt_4_dim = norm_inputs[
        "ego_future_gt_4_dim"]  # (B, future_len, 4)
    # (B, Pnn, future_len, 4)
    unnorm_near_future_gt_4_dim = norm_inputs["near_future_gt_4_dim"]
    unnorm_target_future_gt_4_dim = torch.cat(
        [
            unnorm_ego_future_gt_4_dim[:, None, :, :],
            unnorm_near_future_gt_4_dim
        ],
        dim=1)  # (B, (1+)Pnn, future_len, 4)

    # unnorm_target_future_gt_4_dim_flat: [n_agent = B * (1+)Pnn, future_len, 4]
    unnorm_target_future_gt_4_dim_flat = unnorm_target_future_gt_4_dim.reshape(
        -1, unnorm_target_future_gt_4_dim.shape[2]
    )  # [n_agent = B * (1+)Pnn, future_len, 4]
    unnorm_origin_world_pose = norm_inputs["origin_world_pose"]  # (B, 4)
    """
    unnorm_target_future_gt_4_dim_flat: (n_agent = B * (1+)Pnn, future_len, 4)
    unnorm_origin_world_pose: (B, 4)
    
    unnorm_target_future_gt_4_dim_world: (n_agent = B * (1+)Pnn, future_len, 4)
    """
    unnorm_target_future_gt_4_dim_world = _covert_from_ego_to_world(
        unnorm_target_future_gt_4_dim_flat, unnorm_origin_world_pose)
    # unnorm_target_future_gt_xy_world: [n_agent = B * (1+)Pnn, future_len, 2]
    unnorm_target_future_gt_xy_world = unnorm_target_future_gt_4_dim_world[:, :, :
                                                                           2]

    ######
    """
    unnorm_target_future_gt_4_dim_flat :[n_agent = B * (1+)Pnn, future_len, 4]
    
    이 중, 마지막 차원 (4)의 값이 전부 0. 인 프레임은 무효 프레임입니다.
    """
    target_future_valid_flat = torch.any(
        unnorm_target_future_gt_4_dim_flat[:, :, :4] != 0,
        dim=-1)  # [n_agent = B * (1+)Pnn, future_len]
    min_ade.update(
        pred=pred_traj,  # # [n_ag, n_rollout, n_step, 2]
        target=unnorm_target_future_gt_xy_world,  # [n_ag, n_step, 2]
        target_valid=target_future_valid_flat,  # [n_ag, n_step]
    )

    # data["tfrecord_path"] : List[str]
    # scenario_rollouts : List[sim_agents_submission_pb2.ScenarioRollouts]
    tfrecord_path = norm_inputs["tfrecord_path"]
    assert isinstance(tfrecord_path,
                      list), "tfrecord_path는 List[str] 타입이어야 합니다."
    assert batch_size == len(
        tfrecord_path), "batch_size와 tfrecord_path 길이가 맞지 않습니다."
    wosac_metrics.update(tfrecord_path, scenario_rollouts)


def _update_merged_inputs(
        norm_inputs_copy: Dict[str, torch.Tensor],
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
        normed_near_next_pose: torch.Tensor,  # (B, Pnn, 4)
) -> Dict[str, torch.Tensor]:

    ego_agent_past = norm_inputs_copy["ego_agent_past"]  # (B, time_len, 11)
    ego_next_11_dim = ego_agent_past[:, -1, :].clone()  # (B, 11)
    ego_next_11_dim[:, :4] = normed_ego_next_pose  # (B, 11)
    norm_inputs_copy["ego_agent_past"] = torch.cat(
        [ego_agent_past[:, 1:, :], ego_next_11_dim[:, None, :]],
        dim=1)  # (B, time_len, 11)

    near_agents_past = norm_inputs_copy[
        "near_agents_past"]  # (B, Pnn, T_past, 11)
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B, Pnn, 11)
    near_next_11_dim[:, :4] = normed_near_next_pose  # (B, Pnn, 11)
    norm_inputs_copy["near_agents_past"] = torch.cat(
        [near_agents_past[:, :, 1:, :], near_next_11_dim[:, None, :]],
        dim=2)  # (B, Pnn, T_past, 11)

    non_near_agents_past = norm_inputs_copy[
        "non_near_agents_past"]  # (B, Nnn, T_past, 11)
    assert non_near_agents_past.shape[1] == 0, "현재 non-near agent는 처리하지 않습니다."

    neighbor_agents_past = norm_inputs_copy[
        "neighbor_agents_past"]  # (B, agent_num, T_past, 11)
    neighbor_agents_num = neighbor_agents_past.shape[1]
    assert neighbor_agents_num == near_agents_past.shape[
        1], "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B, Pnn, 11)
    near_next_11_dim[:, :4] = normed_near_next_pose  # (B, Pnn, 11)
    norm_inputs_copy["neighbor_agents_past"] = torch.cat(
        [neighbor_agents_past[:, :, 1:, :], near_next_11_dim[:, None, :]],
        dim=2)  # (B, agent_num, T_past, 11)

    norm_inputs_copy = _transform_origin(norm_inputs_copy, normed_ego_next_pose)
    return norm_inputs_copy


def _clone_norm_inputs_for_rollout(
    norm_inputs: Dict[str, Any],
    sanity_check_tensor_storage: bool = False,
) -> Dict[str, Any]:
    """rollout에서 입력 dict를 안전하게 복사합니다.

    목적은 2가지입니다.

    1) rollout 중에 `norm_inputs_copy`를 계속 수정하더라도,
       원본 `norm_inputs`가 같이 바뀌지 않게 만들기
    2) 값이 `torch.Tensor`인 경우에는 실제 메모리까지 새로 만들어서(copy),
       같은 저장공간을 공유하지 않게 만들기

    복사 규칙:
        - 값이 torch.Tensor 이면: `clone()`으로 새 텐서를 만듭니다.
        - 값이 dict/list/tuple 이면: 안쪽 원소까지 재귀적으로 같은 규칙을 적용합니다.
        - 그 외 타입(int/float/str 등)은: 보통 rollout 중에 수정하지 않으므로 그대로 둡니다.

    Args:
        norm_inputs: 모델 입력 dict.
        sanity_check_tensor_storage: True이면 최상위 key 기준으로
            원본 텐서와 복사 텐서가 같은 저장공간을 공유하지 않는지 확인합니다.

    Returns:
        rollout_inputs: 복사된 입력 dict.
    """
    rollout_inputs: Dict[str, Any] = {
        key: _clone_nested_value(value) for key, value in norm_inputs.items()
    }

    if sanity_check_tensor_storage:
        _assert_no_shared_tensor_storage(norm_inputs, rollout_inputs)

    return rollout_inputs


def _clone_nested_value(value: Any) -> Any:
    """중첩 구조 안의 값을 복사합니다.

    Args:
        value: dict/list/tuple/torch.Tensor 등 어떤 값이든 들어올 수 있습니다.

    Returns:
        copied_value: 복사된 값.
    """
    if isinstance(value, torch.Tensor):
        return value.clone()

    if isinstance(value, dict):
        return {k: _clone_nested_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_clone_nested_value(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_clone_nested_value(v) for v in value)

    # 그 외 타입은 보통 불변이거나, rollout 중에 수정하지 않는다고 가정합니다.
    return value


def _assert_no_shared_tensor_storage(
    original: Dict[str, Any],
    copied: Dict[str, Any],
) -> None:
    """최상위 dict에서 텐서 저장공간 공유 여부를 확인합니다.

    Args:
        original: 원본 dict.
        copied: 복사본 dict.

    Raises:
        AssertionError: 같은 텐서 저장공간을 공유하는 경우.
    """
    for key, original_value in original.items():
        copied_value = copied.get(key, None)
        if not isinstance(original_value, torch.Tensor):
            continue
        if not isinstance(copied_value, torch.Tensor):
            raise AssertionError(
                f"[copy check] key='{key}'는 원본이 Tensor인데 복사본이 Tensor가 아닙니다.")

        original_ptr = _get_tensor_storage_ptr(original_value)
        copied_ptr = _get_tensor_storage_ptr(copied_value)
        if original_ptr == copied_ptr:
            raise AssertionError(
                f"[copy check] key='{key}' 텐서가 같은 저장공간을 공유합니다. "
                f"(원본/복사 storage ptr 동일)")


def _get_tensor_storage_ptr(tensor: torch.Tensor) -> int:
    """텐서의 저장공간 포인터(정수)를 얻습니다.

    Args:
        tensor: torch.Tensor

    Returns:
        storage_ptr: 저장공간 시작 주소(정수).
    """
    if hasattr(tensor, "untyped_storage"):
        return int(tensor.untyped_storage().data_ptr())
    return int(tensor.storage().data_ptr())


def _normalize_cos_sin_for_rotation(
    cos_values: torch.Tensor,
    sin_values: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin)을 회전에 쓰기 좋게 길이 1로 맞춥니다.

    - 길이가 충분히 크면: (cos, sin)을 길이 1로 나눠 정리합니다.
    - 길이가 거의 0이면: 회전을 못하므로 (1, 0)으로 둡니다(회전 없음).

    Args:
        cos_values: shape = (...)
        sin_values: shape = (...)
        eps: 0 나눗셈 방지 값.

    Returns:
        cos_normalized: shape = (...)
        sin_normalized: shape = (...)
    """
    raw_norm = torch.sqrt(cos_values**2 + sin_values**2)
    safe_norm = torch.clamp(raw_norm, min=eps)

    cos_normalized = cos_values / safe_norm
    sin_normalized = sin_values / safe_norm

    too_small = raw_norm < eps
    cos_normalized = torch.where(too_small, torch.ones_like(cos_normalized),
                                 cos_normalized)
    sin_normalized = torch.where(too_small, torch.zeros_like(sin_normalized),
                                 sin_normalized)
    return cos_normalized, sin_normalized


def _transform_origin(
        norm_inputs_copy: Dict[str, torch.Tensor],
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
) -> Dict[str, torch.Tensor]:
    """normed_ego_next_pose 기준으로 입력 전체의 좌표 기준을 바꿉니다.

    핵심 아이디어는 간단합니다.

    - ego의 다음 위치/방향을 "새 기준"으로 삼습니다.
    - 그러면 ego의 다음 위치는 (0, 0)이 되고,
      ego의 다음 방향은 정면(각도 0)이 되도록 입력 전체를 같이 바꿔줘야 합니다.

    변환은 모든 좌표에 대해 아래 2단계로 이뤄집니다.
        1) 이동: 모든 (x, y)에서 ego의 다음 위치 (dx, dy)를 뺍니다.
        2) 회전: ego의 다음 방향만큼 반대로 회전합니다.

    무효 데이터:
        각 텐서마다 정해진 규칙(0 패딩 또는 별도 valid mask)에 따라
        무효인 값은 변환하지 않고 원래 값을 유지합니다.

    Args:
        norm_inputs_copy: 모델 입력 dict.
        normed_ego_next_pose: (B, 4) ego 다음 프레임 (x, y, cos, sin)

    Returns:
        norm_inputs_copy: 좌표 기준이 변환된 입력 dict.
    """
    (delta_xy, cos_delta, sin_delta,
     yaw_delta) = _extract_delta_pose_params(normed_ego_next_pose)

    # 1) ego_agent_past: (B, time_len, 11)
    if "ego_agent_past" in norm_inputs_copy:
        ego_agent_past = norm_inputs_copy["ego_agent_past"]
        if isinstance(ego_agent_past,
                      torch.Tensor) and ego_agent_past.numel() > 0:
            valid_mask = torch.any(ego_agent_past[..., :8] != 0.0,
                                   dim=-1)  # (B, time_len)
            _transform_state_11_dim_inplace(
                state_11=ego_agent_past,
                delta_xy=delta_xy,
                cos_delta=cos_delta,
                sin_delta=sin_delta,
                valid_mask=valid_mask,
            )

    # 3) agent past 11 dim: near/non_near/neighbor
    for key in [
            "near_agents_past", "non_near_agents_past", "neighbor_agents_past"
    ]:
        if key not in norm_inputs_copy:
            continue
        agents_past = norm_inputs_copy[key]
        if not isinstance(agents_past,
                          torch.Tensor) or agents_past.numel() == 0:
            continue

        valid_mask = torch.any(agents_past[..., :8] != 0.0,
                               dim=-1)  # (B, A, time_len)
        _transform_state_11_dim_inplace(
            state_11=agents_past,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    # 5) points + explicit valid mask
    point_and_mask_keys = [
        ("stop_sign_points", "stop_sign_points_is_valid"),
        ("crosswalk_points", "crosswalk_points_is_valid"),
        ("speed_bump_points", "speed_bump_points_is_valid"),
        ("driveway_points", "driveway_points_is_valid"),
        ("road_edge", "road_edge_is_valid"),
    ]
    for point_key, mask_key in point_and_mask_keys:
        if point_key not in norm_inputs_copy or mask_key not in norm_inputs_copy:
            continue
        points_xy = norm_inputs_copy[point_key]  # (B, N, L, 2)
        points_valid = norm_inputs_copy[mask_key]  # (B, N, L)
        if (not isinstance(points_xy, torch.Tensor)) or points_xy.numel() == 0:
            continue
        if not isinstance(points_valid, torch.Tensor):
            continue

        _transform_points_2d_inplace(
            points_xy=points_xy,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=points_valid.to(dtype=torch.bool),
        )

    # 6) lanes / route_lanes: (B, lane_num, lane_len, 12)
    for lane_key in ["lanes", "route_lanes"]:
        if lane_key not in norm_inputs_copy:
            continue
        lane_12 = norm_inputs_copy[lane_key]
        if (not isinstance(lane_12, torch.Tensor)) or lane_12.numel() == 0:
            continue

        valid_mask = torch.any(lane_12[..., :8] != 0.0,
                               dim=-1)  # (B, lane_num, lane_len)
        _transform_lane_12_dim_inplace(
            lane_12=lane_12,
            delta_xy=delta_xy,
            cos_delta=cos_delta,
            sin_delta=sin_delta,
            valid_mask=valid_mask,
        )

    # 7) static_objects: (B, static_num, 10)
    if "static_objects" in norm_inputs_copy:
        static_objects = norm_inputs_copy["static_objects"]
        if isinstance(static_objects,
                      torch.Tensor) and static_objects.numel() > 0:
            valid_mask = torch.any(static_objects[..., :4] != 0.0,
                                   dim=-1)  # (B, static_num)
            _transform_static_object_10_dim_inplace(
                static_10=static_objects,
                delta_xy=delta_xy,
                cos_delta=cos_delta,
                sin_delta=sin_delta,
                valid_mask=valid_mask,
            )

    return norm_inputs_copy


def _extract_delta_pose_params(
        normed_ego_next_pose: torch.Tensor,  # (B, 4)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ego 다음 포즈에서 이동/회전에 필요한 값들을 뽑습니다.

    Args:
        normed_ego_next_pose: (B, 4) = (x, y, cos(각도), sin(각도))

    Returns:
        delta_xy: (B, 2)
        cos_delta: (B,)
        sin_delta: (B,)
        yaw_delta: (B,)
    """
    if normed_ego_next_pose.dim() != 2 or normed_ego_next_pose.shape[-1] != 4:
        raise ValueError("normed_ego_next_pose는 (B, 4) 여야 합니다. "
                         f"현재 shape={tuple(normed_ego_next_pose.shape)}")

    delta_xy = normed_ego_next_pose[:, 0:2]  # (B, 2)
    cos_raw = normed_ego_next_pose[:, 2]  # (B,)
    sin_raw = normed_ego_next_pose[:, 3]  # (B,)

    cos_delta, sin_delta = _normalize_cos_sin_for_rotation(cos_raw, sin_raw)
    yaw_delta = torch.atan2(sin_delta, cos_delta)
    return delta_xy, cos_delta, sin_delta, yaw_delta


def _transform_points_to_new_origin(
        points_xy: torch.Tensor,  # (B, ..., 2)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """(x, y) 점들을 새 원점 기준으로 변환합니다(이동+회전)."""
    if points_xy.shape[-1] != 2:
        raise ValueError(
            f"points_xy의 마지막 차원은 2여야 합니다. shape={tuple(points_xy.shape)}")

    delta_xy_b = delta_xy
    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(points_xy.dim() - 2):
        delta_xy_b = delta_xy_b.unsqueeze(1)
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    dx = points_xy[..., 0] - delta_xy_b[..., 0]
    dy = points_xy[..., 1] - delta_xy_b[..., 1]

    x_new = cos_b * dx + sin_b * dy
    y_new = -sin_b * dx + cos_b * dy
    return torch.stack([x_new, y_new], dim=-1)


def _rotate_vectors_to_new_origin(
        vectors_xy: torch.Tensor,  # (B, ..., 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """(vx, vy) 같은 벡터를 새 기준으로 회전만 적용합니다."""
    if vectors_xy.shape[-1] != 2:
        raise ValueError(
            f"vectors_xy의 마지막 차원은 2여야 합니다. shape={tuple(vectors_xy.shape)}")

    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(vectors_xy.dim() - 2):
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    vx = vectors_xy[..., 0]
    vy = vectors_xy[..., 1]
    vx_new = cos_b * vx + sin_b * vy
    vy_new = -sin_b * vx + cos_b * vy
    return torch.stack([vx_new, vy_new], dim=-1)


def _rotate_heading_cos_sin_to_new_origin(
        cos_heading: torch.Tensor,  # (B, ...)
        sin_heading: torch.Tensor,  # (B, ...)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) 방향을 새 기준으로 바꿉니다(각도 빼기)."""
    cos_b = cos_delta
    sin_b = sin_delta
    for _ in range(cos_heading.dim() - 1):
        cos_b = cos_b.unsqueeze(1)
        sin_b = sin_b.unsqueeze(1)

    cos_new = cos_heading * cos_b + sin_heading * sin_b
    sin_new = sin_heading * cos_b - cos_heading * sin_b
    return cos_new, sin_new


def _wrap_angle_to_pi(angle_rad: torch.Tensor) -> torch.Tensor:
    """각도를 [-pi, pi] 범위로 정리합니다."""
    return torch.atan2(torch.sin(angle_rad), torch.cos(angle_rad))


def _transform_state_11_dim_inplace(
        state_11: torch.Tensor,  # (B, ..., 11)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """11차원 상태의 (x,y,cos,sin,vx,vy)를 새 기준으로 변환합니다."""
    if state_11.shape[-1] != 11:
        raise ValueError(
            f"state_11 마지막 차원은 11이어야 합니다. shape={tuple(state_11.shape)}")

    pos_xy = state_11[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    state_11[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    cos_h = state_11[..., 2]
    sin_h = state_11[..., 3]
    cos_new, sin_new = _rotate_heading_cos_sin_to_new_origin(
        cos_h, sin_h, cos_delta, sin_delta)
    state_11[..., 2] = torch.where(valid_mask, cos_new, cos_h)
    state_11[..., 3] = torch.where(valid_mask, sin_new, sin_h)

    vel_xy = state_11[..., 4:6]
    vel_xy_new = _rotate_vectors_to_new_origin(vel_xy, cos_delta, sin_delta)
    state_11[..., 4:6] = torch.where(valid_mask[..., None], vel_xy_new, vel_xy)


def _transform_future_3_dim_inplace(
        future_3: torch.Tensor,  # (B, ..., 3)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        yaw_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """3차원 미래 정답(x,y,각도)을 새 기준으로 변환합니다."""
    if future_3.shape[-1] != 3:
        raise ValueError(
            f"future_3 마지막 차원은 3이어야 합니다. shape={tuple(future_3.shape)}")

    pos_xy = future_3[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    future_3[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    heading = future_3[..., 2]  # (B, ...)
    yaw_b = yaw_delta
    for _ in range(heading.dim() - 1):
        yaw_b = yaw_b.unsqueeze(1)

    heading_new = _wrap_angle_to_pi(heading - yaw_b)
    future_3[..., 2] = torch.where(valid_mask, heading_new, heading)


def _transform_points_2d_inplace(
        points_xy: torch.Tensor,  # (B, ..., 2)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """2D 점(x,y)을 valid_mask가 True인 곳만 새 기준으로 변환합니다."""
    if points_xy.shape[-1] != 2:
        raise ValueError(
            f"points_xy 마지막 차원은 2이어야 합니다. shape={tuple(points_xy.shape)}")

    points_new = _transform_points_to_new_origin(points_xy, delta_xy, cos_delta,
                                                 sin_delta)
    points_xy[...] = torch.where(valid_mask[..., None], points_new, points_xy)


def _transform_lane_12_dim_inplace(
        lane_12: torch.Tensor,  # (B, ..., 12)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """12차원 차선 텐서의 (x,y,dx,dy,dx_left,dy_left,dx_right,dy_right)를 새 기준으로 변환합니다."""
    if lane_12.shape[-1] != 12:
        raise ValueError(
            f"lane_12 마지막 차원은 12이어야 합니다. shape={tuple(lane_12.shape)}")

    pos_xy = lane_12[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    lane_12[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    for start_idx in [2, 4, 6]:
        vec_xy = lane_12[..., start_idx:start_idx + 2]
        vec_new = _rotate_vectors_to_new_origin(vec_xy, cos_delta, sin_delta)
        lane_12[...,
                start_idx:start_idx + 2] = torch.where(valid_mask[..., None],
                                                       vec_new, vec_xy)


def _transform_static_object_10_dim_inplace(
        static_10: torch.Tensor,  # (B, ..., 10)
        delta_xy: torch.Tensor,  # (B, 2)
        cos_delta: torch.Tensor,  # (B,)
        sin_delta: torch.Tensor,  # (B,)
        valid_mask: torch.Tensor,  # (B, ...) bool
) -> None:
    """10차원 고정 물체의 (x,y,cos,sin)을 새 기준으로 변환합니다."""
    if static_10.shape[-1] != 10:
        raise ValueError(
            f"static_10 마지막 차원은 10이어야 합니다. shape={tuple(static_10.shape)}")

    pos_xy = static_10[..., 0:2]
    pos_xy_new = _transform_points_to_new_origin(pos_xy, delta_xy, cos_delta,
                                                 sin_delta)
    static_10[..., 0:2] = torch.where(valid_mask[..., None], pos_xy_new, pos_xy)

    cos_h = static_10[..., 2]
    sin_h = static_10[..., 3]
    cos_new, sin_new = _rotate_heading_cos_sin_to_new_origin(
        cos_h, sin_h, cos_delta, sin_delta)
    static_10[..., 2] = torch.where(valid_mask, cos_new, cos_h)
    static_10[..., 3] = torch.where(valid_mask, sin_new, sin_h)


def _expand_origin_world_pose_for_agents(
    origin_world_pose: torch.Tensor,  # (B, 4) 또는 (N, 4) 또는 (1, 4)
    n_agent: int,
) -> torch.Tensor:
    """agent 개수에 맞게 origin_world_pose를 늘려서 (N, 4)로 만듭니다.

    이 함수는 "배치 단위로 주어진 world 기준 원점/방향"을,
    "agent 단위(1+Pnn까지 펼친 개수)"로 맞춰주는 역할을 합니다.

    예를 들어,
    - origin_world_pose가 (B, 4)이고,
    - ego_pose가 (N, ..., 4)이며 N = B * (1+Pnn) 라면,
    배치의 origin_world_pose를 각 배치의 agent 수만큼 반복해서 (N, 4)로 확장합니다.

    Args:
        origin_world_pose: (B, 4) 또는 (N, 4) 또는 (1, 4).
            마지막 차원 4는 (x, y, cos, sin) 입니다.
        n_agent: 펼친 agent 개수. (예: B * (1+Pnn))

    Returns:
        origin_world_pose_per_agent: (N, 4)
            각 agent에 대응되는 world 기준 원점/방향.
    """
    if origin_world_pose.dim() != 2 or origin_world_pose.shape[-1] != 4:
        raise ValueError(
            "origin_world_pose는 (B, 4) 또는 (N, 4) 또는 (1, 4) 여야 합니다. "
            f"현재 shape={tuple(origin_world_pose.shape)}")

    origin_count = int(origin_world_pose.shape[0])
    if origin_count == n_agent:
        return origin_world_pose

    if origin_count == 1:
        return origin_world_pose.repeat(n_agent, 1)

    if n_agent % origin_count != 0:
        raise ValueError(
            "n_agent가 origin_world_pose의 첫 번째 차원(B)로 나누어 떨어져야 합니다. "
            f"n_agent={n_agent}, origin_count(B)={origin_count}")

    repeat_factor = n_agent // origin_count
    return origin_world_pose.repeat_interleave(repeat_factor, dim=0)


def _covert_from_ego_to_world(
        ego_pose: torch.Tensor,  # (N, 4) 또는 (N, T, 4)
        origin_world_pose: torch.Tensor,  # (B, 4) 또는 (N, 4) 또는 (1, 4)
) -> torch.Tensor:
    """ego 기준 포즈를 world 기준 포즈로 바꿉니다.

    이 함수는 (x, y, cos, sin) 형태의 포즈를 변환합니다.

    - ego_pose는 "ego를 원점(0,0)으로 보는 좌표"에서의 값입니다.
    - origin_world_pose는 "ego 원점이 world에서 어디에 있고, 어느 방향을 보고 있는지"를 나타냅니다.

    변환 방법은 다음 순서로 진행합니다.

    1) 위치(x, y) 변환
       - ego 기준 (x, y)를 origin_world_pose의 방향만큼 돌린 뒤,
         origin_world_pose의 (x, y)를 더해서 world 위치로 만듭니다.

    2) 방향(cos, sin) 변환
       - ego 기준 방향과 origin_world_pose의 방향을 합쳐서 world 방향으로 만듭니다.
       - (cos, sin)이 길이 1이 아니게 흔들릴 수 있으니, 변환 전에 길이를 1로 정리합니다.

    3) 무효 프레임 처리
       - 마지막 차원 4개 값이 모두 0인 경우는 "패딩(없는 데이터)"로 보고,
         변환 결과도 0을 유지합니다.
         (이 처리를 안 하면, (0,0,0,0)이 origin 위치/방향으로 바뀌어버리는 문제가 생깁니다.)

    Args:
        ego_pose: (N, 4) 또는 (N, T, 4)
            - N: agent 개수(예: B*(1+Pnn))
            - T: 시간 길이(future_len 등)
            - 마지막 4: (x, y, cos, sin)  (ego 기준)
        origin_world_pose: (B, 4) 또는 (N, 4) 또는 (1, 4)
            - B: 배치 크기
            - 마지막 4: (x, y, cos, sin)  (world 기준)

    Returns:
        world_pose: ego_pose와 같은 shape
            - (N, 4) 또는 (N, T, 4)
            - 마지막 4: (x, y, cos, sin)  (world 기준)
    """
    if ego_pose.dim() not in (2, 3) or ego_pose.shape[-1] != 4:
        raise ValueError("ego_pose는 (N, 4) 또는 (N, T, 4) 여야 합니다. "
                         f"현재 shape={tuple(ego_pose.shape)}")

    n_agent = int(ego_pose.shape[0])
    origin_world_pose_per_agent = _expand_origin_world_pose_for_agents(
        origin_world_pose=origin_world_pose,
        n_agent=n_agent,
    )  # (N, 4)

    # origin pose (world)
    origin_xy = origin_world_pose_per_agent[:, 0:2]  # (N, 2)
    origin_cos_raw = origin_world_pose_per_agent[:, 2]  # (N,)
    origin_sin_raw = origin_world_pose_per_agent[:, 3]  # (N,)
    origin_cos, origin_sin = _normalize_cos_sin_for_rotation(
        origin_cos_raw, origin_sin_raw)  # (N,), (N,)

    # ego pose (ego frame)
    ego_xy = ego_pose[..., 0:2]  # (N, 2) 또는 (N, T, 2)
    ego_cos_raw = ego_pose[..., 2]  # (N,) 또는 (N, T)
    ego_sin_raw = ego_pose[..., 3]  # (N,) 또는 (N, T)
    ego_cos, ego_sin = _normalize_cos_sin_for_rotation(ego_cos_raw, ego_sin_raw)

    # 브로드캐스팅을 위해 (N, 2)/(N,) -> (N, 1, 2)/(N, 1) 로 확장 (T가 있는 경우)
    if ego_pose.dim() == 2:
        origin_xy_b = origin_xy  # (N, 2)
        origin_cos_b = origin_cos  # (N,)
        origin_sin_b = origin_sin  # (N,)
    else:
        origin_xy_b = origin_xy[:, None, :]  # (N, 1, 2)
        origin_cos_b = origin_cos[:, None]  # (N, 1)
        origin_sin_b = origin_sin[:, None]  # (N, 1)

    # 1) 위치 변환: world_xy = origin_xy + R(origin_yaw) * ego_xy
    ego_x = ego_xy[..., 0]
    ego_y = ego_xy[..., 1]
    world_x = origin_xy_b[..., 0] + origin_cos_b * ego_x - origin_sin_b * ego_y
    world_y = origin_xy_b[..., 1] + origin_sin_b * ego_x + origin_cos_b * ego_y

    # 2) 방향 변환: yaw_world = yaw_origin + yaw_ego
    # cos(yaw_o + yaw_e) = cos_o*cos_e - sin_o*sin_e
    # sin(yaw_o + yaw_e) = sin_o*cos_e + cos_o*sin_e
    world_cos = origin_cos_b * ego_cos - origin_sin_b * ego_sin
    world_sin = origin_sin_b * ego_cos + origin_cos_b * ego_sin

    world_pose = torch.stack([world_x, world_y, world_cos, world_sin], dim=-1)

    # 3) 무효 프레임(전부 0)인 경우는 그대로 0을 유지
    valid_mask = torch.any(ego_pose != 0.0, dim=-1)  # (N,) 또는 (N, T)
    world_pose = torch.where(valid_mask[..., None], world_pose,
                             torch.zeros_like(world_pose))
    return world_pose


def main() -> None:
    """train_predictor 진입점.

    처리 순서:
      1) 명령행 인자를 읽고(args_util.get_args), 필요 시 W&B 아티팩트에서
         체크포인트 파일(latest.pth 등)을 내려받아 args.save_path 를 채운다.
      2) 환경변수 WORLD_SIZE 를 기준으로 args.distributed 를 설정해,
         이후 ddp 설정이 올바르게 동작하도록 만든다.
      3) model_validation(args) 를 호출해 전체 학습 파이프라인을 수행하고,
         예외가 발생하면 rank 정보를 찍고 전체 스택을 출력한다.
    """
    # 1) 분산 초기화 및 rank 정보
    args = args_util.get_args()
    global_rank, rank, world_size, use_deepspeed = init_distributed(args)
    get_save_path(
        args=args,
        global_rank=global_rank,
    )
    prepare_wandb_resume(args)
    set_distributed_flag_from_env(args)

    # Run
    try:
        model_validation(args, global_rank, rank, world_size, use_deepspeed)
    except BaseException:
        rank = int(os.environ.get("RANK", -1))
        print(
            f"\n[rank{rank}] Unhandled exception (printing full traceback):",
            file=sys.stderr,
            flush=True,
        )
        traceback.print_exc()  # <-- 표준에러로 자세한 스택
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
