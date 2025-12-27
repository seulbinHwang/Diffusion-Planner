import argparse
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
from train_predictor import DiffusionPlannerCollate
from typing import Tuple, Any, Dict, Optional, List, Callable
import torch
import torch.nn as nn
from torch import optim
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

def validate_one_epoch(
    epoch: int,
    total_epochs: int,
    validation_loader: DataLoader,
    diffusion_planner: nn.Module,
    args: argparse.Namespace,
    model_ema: Optional[ModelEma],
batch_num_in_one_val_epoch: int,
) -> Tuple[Dict[str, float], float, float]:
    """하나의 epoch 동안 학습을 실행하고, 손실과 걸린 시간을 계산한다.

    Args:
        epoch: 현재 epoch 인덱스(0 기반).
        total_epochs: 전체 epoch 수.
        validation_loader: 학습용 DataLoader. 배치 텐서는 (B, ·) shape.
        diffusion_planner: 학습 중인 모델(nn.Module 또는 DDP/DeepSpeed 래퍼).

        args: 학습 설정.
        model_ema: EMA 래퍼 또는 None.
        aug: augmentation 객체(StatePerturbation/NPCStatePerturbation 등) 또는 None.

    Returns:
        validation_loss: key별 epoch 평균 손실 딕셔너리. 각 값은 scalar float.
        validation_total_loss: 최종 합 손실 scalar float.
        epoch_elapsed_time_sec: 해당 epoch에 걸린 시간(초).
    """
    # train_epoch 안에서 각 step 배치 텐서는 대략
    #   ego_agent_past: (B, T_past, 11)
    #   neighbor_agents_past: (B, A, T_past, 11)
    #   lanes: (B, L, lane_len, 12)
    #   static_objects: (B, S, 10)
    #   neighbor_future_gt_3_dim: (B, A, T_future, 3)
    # 와 같은 shape로 사용된다.
    if args.ddp and ddp.get_rank() == 0:
        print(f"Epoch {epoch + 1}/{total_epochs}")

    epoch_t0 = time.perf_counter()
    """ validation_loss: Dict[str, float]
    <validate_func 가 출력해주는 loss_dict>
        - neighbor_prediction_loss : 이웃 예측 손실 텐서
        - integration_loss : 통합 손실 텐서
        - constraint_loss : 제약 손실 텐서

        - neighbor_prediction_loss_xy / integration_loss_xy
        - neighbor_prediction_loss_yaw / integration_loss_yaw
        - neighbor_prediction_loss_xy_early / integration_loss_xy_early
        - neighbor_prediction_loss_yaw_early / integration_loss_yaw_early
        - constraint_diff_vx_b / constraint_diff_vy_b / constraint_diff_yaw_rate

    <validate_func 출력 후 _compute_loss_dict 가 추가하는 항목>
    - learn_progress / direct_loss_weight / int_loss_weight / const_loss_weight :
        진행도 및 가중치 기록용 텐서
    - loss_dict : 최종 합 손실 텐서.
    """

    validation_loss, validation_total_loss = validation_epoch(
        validation_loader,
        diffusion_planner,
        args,
        model_ema,
        batch_num_in_one_val_epoch,
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
    return validation_loss, validation_total_loss, epoch_elapsed_time_sec


def validation_epoch(
        data_loader,
        model: nn.Module,
        args: argparse.Namespace,
        ema: Optional[object],
        batch_num_in_one_val_epoch: int,
        aug: Optional[StatePerturbation] = None,
) -> Tuple[Dict[str, float], float]:
    """하나의 epoch 동안 DataLoader 전체를 돌며 학습을 수행한다.

    처리 순서:
      1) 모델을 train 모드로 두고, DDP 사용 시 CUDA 동기화.
      2) 전체 업데이트 스텝 수(batch_num_in_one_val_epoch)를 계산해 진행도(progress) 기준을 잡는다.
      3) 각 배치에 대해
         - device 로 이동 및 상한 클리핑(_prepare_batch_for_device)
         - augmentation, near future mask/ near future 4차원 궤적 생성
         - 관측 normalization
         - diffusion 손실 / feasible 손실 합성
         - 역전파, optimizer/scheduler step, EMA 업데이트
         - 배치별 loss 를 epoch_loss_dict_list 리스트에 모은다.
      4) epoch 종료 후, 배치별 loss 를 평균(get_epoch_mean_loss).
      5) DDP 사용 시 rank 0 기준으로 평균 후 출력/반환.

    Args:
        data_loader:
            PyTorch DataLoader. 각 요소는 collate_fn 이 만든 배치 dict.
        model:
            학습 중인 모델(nn.Module 또는 DDP 래퍼).
        optimizer:
            torch.optim.Optimizer 인스턴스.
        args:
            학습 설정/상태 Namespace.
        ema:
            EMA 래퍼(ModelEma 등) 또는 None.
        scheduler:
            학습률 스케줄러. 배치마다 step() 이 호출된다.
        aug:
            StatePerturbation 또는 NPCStatePerturbation, 또는 None.

    Returns:
        - epoch_mean_loss: 손실 항목별 평균 dict. ( Dict[str, float] )
        - epoch_mean_loss["loss"]: 최종 스칼라 손실 값. float
    """
    epoch_loss_dict_list: List[Dict[str, torch.Tensor]] = []

    model.eval()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
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

            # 5) loss 계산 + 역전파 + optimizer/scheduler step

            # base_model: DDP/DeepSpeed 래퍼 벗긴 실제 모델
            base_model = ddp.get_model(model, args.ddp)
            sde_marginal_prob = base_model.sde.marginal_prob

            # 5-1) diffusion 기본 loss 계산 (neighbor / integration / constraint 등)
            raw_loss_dict: Dict[str, torch.Tensor] = {}
            raw_loss_dict, _ = validate_func(
                args=args,
                model=model,
                norm_inputs=norm_inputs,
                marginal_prob=sde_marginal_prob,
                ego_future_gt_4_dim=outputs["ego_future_gt_4_dim"],
                near_future_gt_4_dim=outputs["near_future_gt_4_dim"],
                near_future_mask=outputs["near_future_mask"],
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
                norm_inputs=norm_inputs,
                batch_num_in_all_epoch=batch_num_in_one_val_epoch,
            )
            total_loss: float = float(loss_dict["loss"].item())

            if args.ddp:
                torch.cuda.synchronize()

            data_epoch.set_postfix(loss="{:.4f}".format(total_loss))
            epoch_loss_dict_list.append(loss_dict)

            # 전역 스텝 누적 (진행도 계산에 사용)

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

def _normalize_futures_and_build_xT(
    target_future_gt_4_dim: torch.Tensor,  # (B, (1+)Pnn, future_len, 4)
    target_current_xyyaw_norm: torch.Tensor,  # (B, (1+)Pnn, 4)
    target_cur_future_mask: torch.Tensor,  # (B, (1+)Pnn, 1+future_len)
    batch_diffusion_time: torch.Tensor,  # (B,)
    state_normalizer: StateNormalizer,
    marginal_prob: Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
                                                                torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 궤적을 정규화하고, x_T 샘플과 std 를 만든다.

    Args:
        target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4) 미래 궤적(denorm).
        target_current_xyyaw_norm: (B, (1+)Pnn, 4) 현재 상태(정규화).
        target_cur_future_mask: (B, (1+)Pnn, 1+future_len) 현재+미래 마스크.
        batch_diffusion_time: (B,) diffusion 시간.
        random_noise: (B, (1+)Pnn, future_len, 4) 노이즈.
        state_normalizer: 상태 정규화/역정규화 도우미.
        marginal_prob: SDE 의 marginal_prob 함수.

    Returns:
        target_future_norm_gt: (B, (1+)Pnn, future_len, 4) 정규화된 미래 궤적.
        cond_last_pos_norm: (B, (1+)Pnn, 4) 미래 마지막 프레임(정규화).
        std: (B, 1, 1, 1) 노이즈 표준편차.
    """
    # target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
    B, one_or_Pnn, future_len, _ = target_future_gt_4_dim.shape

    # normed_target_future_gt_4_dim: (B, (1+)Pnn, future_len, 4)
    normed_target_future_gt_4_dim: torch.Tensor = state_normalizer(
        target_future_gt_4_dim)
    target_future_mask = target_cur_future_mask[:, :,
                                                1:]  # (B, (1+)Pnn, future_len)
    normed_target_future_gt_4_dim[target_future_mask] = 0.0
    # cond_last_pos_norm: (B, (1+)Pnn, 4)
    cond_last_pos_norm: torch.Tensor = normed_target_future_gt_4_dim[:, :,
                                                                     -1, :]

    normed_target_future_gt_4_dim = _require_finite(
        "state_normalizer(near_future_gt_4_dim)", normed_target_future_gt_4_dim)

    # target_cur_future_norm_gt: (B, (1+)Pnn, 1+future_len, 4)
    target_cur_future_norm_gt: torch.Tensor = torch.cat(
        [
            target_current_xyyaw_norm[:, :, None, :],
            normed_target_future_gt_4_dim
        ],
        dim=2,
    )
    target_cur_future_norm_gt[target_cur_future_mask] = 0.0
    # target_future_norm_gt: (B, (1+)Pnn, future_len, 4)
    target_future_norm_gt: torch.Tensor = target_cur_future_norm_gt[:, :, 1:, :]

    # mean, std_raw: 각각 (B, (1+)Pnn, future_len, 4) 또는 (B,) 등 marginal_prob 설계에 맞게 반환
    mean, std_raw = marginal_prob(target_future_norm_gt, batch_diffusion_time)
    std_raw = _require_finite("marginal_prob std", std_raw)

    # std: (B, 1, 1, 1) 로 reshape
    std: torch.Tensor = std_raw.view(
        -1,
        *([1] * (len(target_future_norm_gt.shape) - 1)),
    )


    return target_future_norm_gt, cond_last_pos_norm, std


def validate_func(
        args: Any,
        model: nn.Module,
        norm_inputs: Dict[str, torch.Tensor],
        marginal_prob: Callable[
            [torch.Tensor, torch.Tensor], Tuple[torch.Tensor,
            torch.Tensor]],
        ego_future_gt_4_dim: torch.Tensor,  # (B, future_len, 4)
        near_future_gt_4_dim: torch.Tensor,  # (B, Pnn, future_len, 4)
        near_future_mask: torch.Tensor,  # (B, Pnn, future_len)
        state_normalizer: StateNormalizer,
        loss_dict: Dict[str, Any],
        model_type: str,
        observation_normalizer: Any,
        eps: float = 1e-3,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """diffusion 학습에 쓰이는 전체 손실을 계산한다.

    처리 흐름:
      1) 미래 궤적/마스크, 정규화 입력을 안전하게 준비.
      2) 현재 상태와 미래를 이어 붙인 시퀀스를 만들고, diffusion time/노이즈 샘플링.
      3) SDE marginal_prob 를 통해 x_T 샘플 생성 후, 모델 forward.
      4) 기본 diffusion 손실(dpm_loss) + half-life 시간 가중 평균.
      5) x_start 모드일 때 통합 궤적/제어 제약 손실, xy/yaw 통계까지 loss dict 에 추가.

    Args:
        args: 학습 설정/옵션이 들어 있는 객체.
        model: 학습 중인 모델(nn.Module 또는 DDP 래퍼).
        norm_inputs: 정규화된 관측 dict.
        marginal_prob: SDE marginal_prob 함수.
        near_future_gt_4_dim: (B, Pnn, future_len, 4) 미래 궤적 (denorm).
        near_future_mask: (B, Pnn, future_len) True면 값이 없는 프레임.
        state_normalizer: 상태 정규화/역정규화 도우미.
        loss_dict: 손실/통계를 쌓아갈 dict (in-place 업데이트).
        model_type: "score" 또는 "x_start".
        observation_normalizer: 제어 역정규화 도우미.
        eps: diffusion time 샘플링 하한.

    Returns:
        loss_dict: 다양한 손실 항목이 담긴 dict.
        decoder_output: 모델 decoder 의 출력 dict.
    """
    # norm_inputs 의 각 텐서 NaN/Inf 체크
    norm_inputs = _sanitize_norm_inputs(norm_inputs)
    # 모델 forward + decoder_output 생성
    decoder_output: Dict[str, torch.Tensor] = _forward_model_with_autocast(args,
                                                                           state_normalizer,
        model=model,  #
        norm_inputs=norm_inputs,  #
        use_deepspeed=getattr(args, "use_deepspeed", False),
    )


    return loss_dict, decoder_output


def _forward_model_with_autocast(
args,
state_normalizer,
    model: nn.Module,

    norm_inputs: Dict[str, torch.Tensor],
    use_deepspeed: bool,
) -> Dict[str, torch.Tensor]:
    """모델 입력 dict 를 만들고 AMP 로 forward 를 수행한다.

    Args:
        model: 학습 중인 모델.
        norm_inputs: 정규화된 관측 dict.
            ego_agent_past : (B, time_len, 11) #
            ego_future_gt_3_dim : (B, future_len, 3)
            neighbor_agents_past : (B, agent_num, time_len, 11) #
            lanes : (B, lane_num, lane_len, 12) #
            lanes_speed_limit : (B, lane_num, 1) #
            lanes_has_speed_limit : (B, lane_num, 1) #
            route_lanes : (B, route_num, route_len, 12)
            route_lanes_speed_limit : (B, route_num, 1)
            route_lanes_has_speed_limit : (B, route_num, 1)
            static_objects : (B, static_num, 10) #
            near_future_gt_3_dim: (B, Pnn, future_len, 3)
            planner_future_11_dim: (B, future_len, 11) #
            agent_route_lane_order: (B, agent_num, lane_num)

        target_future_valid: (B, (1 +) Pnn, future_len) 미래 유효 마스크.

    Returns:
        decoder_output: model 의 두 번째 반환값 dict.


        "near_future_valid": near_future_valid,  # (B, Pnn, future_len)
        "near_cur_future_norm_xT":
            near_cur_future_norm_xT,  # (B, Pnn, 1+future_len, 4)
    """
    merged_inputs: Dict[str, torch.Tensor] = {
        **norm_inputs,
    }
    target_future_valid = build_target_future_tensors_and_masks_for_inference(
        args,
        merged_inputs,
        args.future_len,
    )  # (B, (1+)Pnn, future_len)  True=유효
    merged_inputs["target_future_valid"] = target_future_valid
    batch_size =  merged_inputs["ego_agent_past"].shape[0]
    is_ds_engine = hasattr(model, "backward") and hasattr(
        model, "step") and hasattr(model, "module")
    ROLLOUT_NUMBER = 32
    target_scenario_rollouts_np = [] # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 4)
    for rollout_idx in range(ROLLOUT_NUMBER):
        target_joint_scene = [] # (B, (1+)Pnn, future_len, 4)
        for step_idx in range(args.future_len):
            if use_deepspeed:
                assert is_ds_engine, "use_deepspeed=True 인데 model 이 DS engine 아님"
                _, decoder_output = model(merged_inputs)  # DS가 torch_autocast로 처리
            else:
                with torch.autocast("cuda", dtype=AMP_DTYPE):
                    _, decoder_output = model(merged_inputs)
            assert args.do_ego_predict, "현재 ego 예측은 항상 활성화되어 있어야 합니다."
            # TODO: 나중에 더 고급진 방법으로 업데이트할 수도?
            integrated_trajectories = decoder_output.get(
                "integrated_trajectory")  # (B, (1+)Pnn, 1+T, 4)
            ego_trajectories = integrated_trajectories[:, 0, :, :]  # (B, 1+T, 4)
            near_trajectories = integrated_trajectories[
                :, 1:, :, :]  # (B, Pnn, 1+T, 4)
            # TODO: 시간 간격에 따라 제대로 표시하기
            ego_next_pose = ego_trajectories[:, 1, :]  # (B, 4)
            near_next_pose = near_trajectories[:, :, 1, :]  # (B, Pnn, 4)
            merged_inputs = _update_merged_inputs(merged_inputs, ego_next_pose, near_next_pose)
            target_next_pose = torch.cat(
                [ego_next_pose[:, None, :], near_next_pose], dim=1
            ) # (B, (1+)Pnn, 4)
            target_next_pose_unnorm = state_normalizer.inverse(target_next_pose)
            target_next_pose_unnorm[target_future_valid] = 0.0
            target_joint_scene.append(target_next_pose_unnorm)
        target_joint_scene = torch.stack(
            target_joint_scene, dim=2)  # (B, (1+)Pnn, future_len, 4)
        target_scenario_rollouts_np.append(target_joint_scene)
    target_scenario_rollouts_np = torch.stack(
        target_scenario_rollouts_np, dim=2)  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 4)
    """ TODO
    target_scenario_rollouts_np 는 ego 좌표계 기준 값(unnorm)이므로, 이를 세계 좌표계로 변환해야 합니다.
    이에 걸맞는 함수를 작성해야 합니다. (무효점은 그대로 남겨둡니다.)
    """
    """
    target_scenario_rollouts : List[sim_agents_submission_pb2.ScenarioRollouts] # 길이: B
    """
    pred_traj = target_scenario_rollouts_np[:, :, :, :, :2]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len, 2)
    pred_traj = pred_traj.reshape(
        pred_traj.shape[0] * pred_traj.shape[1],
        pred_traj.shape[2],
        pred_traj.shape[3],
        pred_traj.shape[4],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step, 2]
    pred_head_cos_yaw = target_scenario_rollouts_np[:, :, :, :, 2]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head_sin_yaw = target_scenario_rollouts_np[:, :, :, :, 3]  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head = torch.atan2(pred_head_sin_yaw, pred_head_cos_yaw)  # (B, (1+)Pnn, ROLLOUT_NUMBER, future_len)
    pred_head = pred_head.reshape(
        pred_head.shape[0] * pred_head.shape[1],
        pred_head.shape[2],
        pred_head.shape[3],
    )  # [n_agent = B * (1+)Pnn, n_rollout, n_step]
    # List[ScenarioRollouts]

    scenario_id:List[str] = merged_inputs["scenario_id"]  # List[str], 길이 Batch size
    assert isinstance(scenario_id, list), "scenario_id는 List[str] 타입이어야 합니다."
    assert batch_size == len(scenario_id), "batch_size와 scenario_id 길이가 맞지 않습니다."
    target_id = merged_inputs["target_id"]  # (B, (1+)Pnn)
    target_id = target_id.reshape(-1)  # [n_agent = B * (1+)Pnn]
    """
    agent_batch 를 구합니다. (n_agent 길이, 각 agent 가 속한 batch 인덱스)
    예: batch_size=2, (1+)Pnn=3 이면,
        agent_batch = [0,0,0,1,1,1]
    """
    agent_batch = []
    for batch_idx in range(batch_size):
        one_or_Pnn = merged_inputs["target_future_valid"].shape[1]
        agent_batch.extend([batch_idx] * one_or_Pnn)
    agent_batch = torch.tensor(agent_batch, dtype=torch.long, device=pred_traj.device)  # [n_agent]

    target_z = merged_inputs["target_z"] #  (B, (1+)Pnn)
    target_z = target_z.reshape(-1)  # [n_agent = B * (1+)Pnn]

    device = pred_traj.device
    scenario_rollouts = get_scenario_rollouts(
        scenario_id=get_scenario_id_int_tensor(
            scenario_id, device
        ), # [n_scenario, n_str_length]
        agent_id=target_id, # [n_agent]
        agent_batch=agent_batch, # [n_agent]
        pred_traj=pred_traj, # [n_agent, n_rollout, n_step, 2]
        pred_z=target_z, # [n_agent, n_rollout, n_step]
        pred_head=pred_head, # [n_agent, n_rollout, n_step]
    )


    return decoder_output  # decoder_output["score"], ["integrated_trajectory"], ...


def _update_merged_inputs(
    merged_inputs: Dict[str, torch.Tensor],
    ego_next_pose: torch.Tensor,  # (B, 4)
    near_next_pose: torch.Tensor,  # (B, Pnn, 4)
) -> Dict[str, torch.Tensor]:

    ego_agent_past = merged_inputs["ego_agent_past"]  # (B, time_len, 11)
    ego_next_11_dim = ego_agent_past[:, -1, :].clone()  # (B, 11)
    ego_next_11_dim[:, :4] = ego_next_pose  # (B, 11)
    merged_inputs["ego_agent_past"] = torch.cat(
            [ego_agent_past[:, 1:, :], ego_next_11_dim[:, None, :]], dim=1
        )  # (B, time_len, 11)

    near_agents_past = merged_inputs["near_agents_past"]  # (B, Pnn, T_past, 11)
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B, Pnn, 11)
    near_next_11_dim[:, :4] = near_next_pose  # (B, Pnn, 11)
    merged_inputs["near_agents_past"] = torch.cat(
            [near_agents_past[:, :, 1:, :], near_next_11_dim[:, None, :]], dim=2
        )  # (B, Pnn, T_past, 11)

    non_near_agents_past = merged_inputs["non_near_agents_past"]  # (B, Nnn, T_past, 11)
    assert non_near_agents_past.shape[1] == 0, "현재 non-near agent는 처리하지 않습니다."

    neighbor_agents_past = merged_inputs["neighbor_agents_past"]  # (B, agent_num, T_past, 11)
    neighbor_agents_num = neighbor_agents_past.shape[1]
    assert neighbor_agents_num == near_agents_past.shape[1], "neighbor_agents_past의 agent 수가 near_agents_past와 다릅니다."
    near_next_11_dim = near_agents_past[:, :, -1, :].clone()  # (B, Pnn, 11)
    near_next_11_dim[:, :4] = near_next_pose  # (B, Pnn, 11)
    merged_inputs["neighbor_agents_past"] = torch.cat(
            [neighbor_agents_past[:, :, 1:, :], near_next_11_dim[:, None, :]], dim=2
        )  # (B, agent_num, T_past, 11)
    merged_input = _transform_origin(merged_inputs, ego_next_pose)
    return merged_inputs

def _transform_origin(
    merged_inputs: Dict[str, torch.Tensor],
    ego_next_pose: torch.Tensor,  # (B, 4)
) -> Dict[str, torch.Tensor]:
    """ego_next_pose 기준으로 좌표계를 변환합니다.

    Args:
        merged_inputs: 모델 입력 dict. 변환이 필요한 key 와 tensor들은 전부 변환합니다.
        ego_next_pose: (B, 4) ego 다음 프레임 위치/방향.

    중요:
        무효인 데이터는 변환하지 않습니다.

    Returns:
        transformed_inputs: 좌표계가 변환된 모델 입력 dict.
    """
    # TODO:
    return merged_inputs

