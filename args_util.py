import argparse
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
import os
from typing import Any, Dict
from mmengine.fileio import load as mmengine_load  # stage config 로딩용


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def _override_args_with_stage_config(
        args: argparse.Namespace) -> argparse.Namespace:
    """stage_config_path에 지정된 json/yaml 파일을 읽어 args 값을 덮어쓴다.

    이 함수는 "한 번의 학습(run)을 하나의 stage 설정 파일로 제어"하기 위한 역할을 한다.
    stage_config_path가 비어 있으면 아무 것도 하지 않고 그대로 돌려주고,
    지정되어 있으면 해당 파일을 로드해서 args 안의 값을 덮어쓴다.

    동작 규칙:
      * 파일 최상단이 dict 이면 그대로 사용하고,
        {"args": {...}} 형태이면 내부 args dict만 꺼내서 쓴다.
      * 파일 안에 있는 key들은 모두 args의 속성으로 복사한다.
        - 이미 존재하는 key  → 값 덮어쓰기 (override 로그 출력)
        - 존재하지 않는 key → 새 속성으로 추가 (unknown key 로그 출력)

    Args:
        args (argparse.Namespace):
            - argparse로 파싱된 원본 인자 객체.
            - 이 함수 안에서 in-place로 수정되며,
              config_dict: Dict[str, Any]  # json/yaml에서 읽은 일반적인 키-값 매핑
              overrides: Dict[str, Any]    # 실제로 덮어쓸 키-값 집합

    Returns:
        argparse.Namespace:
            - stage_config_path가 설정되지 않은 경우: 입력으로 받은 args 그대로.
            - stage_config_path가 유효한 파일인 경우:
              파일 내용으로 값이 덮어쓰기된 동일 args 객체.
    """
    # 우선순위: stage_config_path > stage_config
    stage_config_path = getattr(args, "stage_config_path", None)

    if not stage_config_path:
        return args

    config_path = os.path.expanduser(stage_config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"stage_config_path로 지정한 파일을 찾을 수 없습니다: {config_path}")

    config_dict: Dict[str, Any] = mmengine_load(config_path)
    if not isinstance(config_dict, dict):
        raise TypeError(f"stage 설정 파일은 dict 형태여야 합니다. type={type(config_dict)}")

    overrides: Dict[str, Any] = config_dict.get("args", config_dict)

    for key, value in overrides.items():
        if hasattr(args, key):
            print(f"[StageConfig] override: {key}={value!r}", flush=True)
        else:
            print(f"[StageConfig] unknown key 추가: {key}={value!r}", flush=True)
        setattr(args, key, value)

    print(f"[StageConfig] '{config_path}' 적용 완료.", flush=True)
    return args


def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--name',
        type=str,
        help='log name (default: "diffusion-planner-training")',
        default="world-model-small")  # npc_current_state_aug_0.5
    parser.add_argument('--save_dir',
                        type=str,
                        help='save dir for model ckpt',
                        default=".")
    parser.add_argument('--load_name',
                        type=str,
                        help='prev name for model ckpt',
                        default=None)
    parser.add_argument('--resume_local_path_model_path',
                        type=str,
                        help='path to resume model',
                        default=None)
    parser.add_argument(
        '--resume_wandb_model_name',
        type=str,
        help='wandb artifact version to resume from (e.g., "latest")',
        default=None)
    parser.add_argument(
        '--resume_model_only',
        type=boolean,
        default=False,
        help=("True이면 체크포인트에서 모델(또는 EMA) 파라미터만 불러오고, "
              "optimizer / scheduler / EMA 상태, epoch, wandb run 은 모두 새로 시작합니다. "
              "즉, pretrained weight 로만 초기화된 새 학습 run 으로 동작합니다."),
    )
    # Data
    parser.add_argument('--train_set',
                        type=str,
                        help='path to train data',
                        default=None)
    parser.add_argument('--train_set_list',
                        type=str,
                        help='data list of train data',
                        default=None)
    parser.add_argument('--eval_set',
                        type=str,
                        help='path to validation data',
                        default=None)
    parser.add_argument('--eval_set_list',
                        type=str,
                        help='data list of validation data',
                        default=None)
    parser.add_argument('--eval_method',
                        type=str,
                        help='data list of validation data',
                        default="validation") # or "test"
    parser.add_argument('--run_count',
                        type=int,
                        help='number of run_count',
                        default=0)
    parser.add_argument('--total_save_image_trial_num',
                        type=int,
                        help='number of run_count',
                        default=-1)
    parser.add_argument('--rollout_number',
                        type=int,
                        default=32)
    parser.add_argument('--future_len',
                        type=int,
                        help='number of time point',
                        default=80)
    parser.add_argument('--time_len',
                        type=int,
                        help='number of time point',
                        default=21)

    parser.add_argument('--agent_state_dim',
                        type=int,
                        help='past state dim for agents',
                        default=11)
    parser.add_argument('--caching_max_agent_num',
                        type=int,
                        help='number of agents',
                        default=448)
    parser.add_argument('--max_agent_num',
                        type=int,
                        help='number of agents',
                        default=448)
    parser.add_argument('--predicted_neighbor_num',
                        type=int,
                        help='number of neighbor agents to predict',
                        default=448)
    parser.add_argument('--center_crop_radius_m',
                        type=float,
                        help='center_crop_radius_m',
                        default=-1.0)
    parser.add_argument('--use_data_percent',
                        type=float,
                        help='use_data_percent',
                        default=-100.0)
    parser.add_argument('--center_crop_mode',
                        type=str,
                        choices=['none', 'npc', 'ego'],
                        default='none',
                        help='center crop 기준: none / npc / ego 중 하나만 선택')
    parser.add_argument('--wosac_sub_is_active',
                        default=False,
                        type=boolean,
                        help='True이면 WOSAC 서브미션을 수행함.')
    parser.add_argument('--min_ade_is_active',
                        default=False,
                        type=boolean,
                        help='True이면 WOSAC 서브미션을 수행함.')
    parser.add_argument('--wosac_metric_is_active',
                        default=False,
                        type=boolean,
                        help='True이면 WOSAC 서브미션을 수행함.')
    parser.add_argument('--do_not_clip',
                        default=True,
                        type=boolean,
                        help='True이면 neighbor/lanes/route_lanes 축 클리핑을 수행하지 않음')
    parser.add_argument('--save_inference_data',
                        default=False,
                        type=boolean,
                        help='save_inference_data')
    parser.add_argument('--save_integration_traj',
                        default=False,
                        type=boolean)
    parser.add_argument('--rollout_step_count_for_save',
                        type=int,
                        default=-1)
    parser.add_argument('--static_objects_state_dim',
                        type=int,
                        help='state dim for static objects',
                        default=10)
    parser.add_argument('--caching_max_static_num',
                        type=int,
                        help='number of static objects',
                        default=50)
    parser.add_argument('--max_static_num',
                        type=int,
                        help='number of static objects',
                        default=50)
    parser.add_argument('--lane_len',
                        type=int,
                        help='number of lane point',
                        default=20)
    parser.add_argument('--lane_state_dim',
                        type=int,
                        help='state dim for lane point',
                        default=12)
    parser.add_argument('--caching_max_lane_num',
                        type=int,
                        help='number of lanes',
                        default=250)
    parser.add_argument('--max_lane_num',
                        type=int,
                        help='number of lanes',
                        default=250)
    parser.add_argument('--lane_summary_num',
                        type=int,
                        help='number of lanes',
                        default=100)  # 0 이면 요약 안함
    parser.add_argument('--route_num',
                        type=int,
                        help='number of route lanes',
                        default=25)
    parser.add_argument('--safety_len',
                        type=int,
                        help='number of route lane point',
                        default=10)
    parser.add_argument('--p_sat', type=float, help='p_sat', default=0.6)
    parser.add_argument('--w_dir', type=float, help='w_dir', default=1.0)
    parser.add_argument('--w_int_min',
                        type=float,
                        help='w_int_min',
                        default=0.05)
    parser.add_argument('--w_int_max',
                        type=float,
                        help='w_int_max',
                        default=2.0)
    parser.add_argument('--w_const', type=float, help='w_const', default=0.02)
    parser.add_argument(
        "--resume_wosac_submission",
        default=False,
        type=boolean,
        help=(
            "True이면, validation/test에서 이전 실행으로 만들어진 "
            "wosac_submission 결과 폴더를 지우지 않고 이어서 저장합니다."
        ),
    )

    #########################
    parser.add_argument(
        "--auto_tune_aux_weights",
        default=False,
        type=boolean,
        help=(
            "True이면 epoch 평균(가중치 없는) loss를 보고 w_int/w_const를 자동으로 천천히 조절합니다. "
            "neighbor_prediction_loss는 기준(앵커)으로 고정해두고, "
            "integration/constraint가 best-so-far(여유 포함)에서 크게 나빠지지 않게 합니다."
        ),
    )

    # -------------------------
    # auto_tune_aux_weights용 하이퍼파라미터 (모두 기본값 포함)
    # -------------------------
    parser.add_argument(
        "--auto_aux_ema_alpha",
        type=float,
        default=0.1,
        help="epoch 평균 loss를 부드럽게 만들 때 쓰는 '이동 평균'의 새 값 반영 비율(0~1).",
    )
    parser.add_argument(
        "--auto_aux_delta_int",
        type=float,
        default=0.2,
        help="integration 목표를 best 대비 얼마나 여유 있게 둘지(예: 0.2 = 20%).",
    )
    parser.add_argument(
        "--auto_aux_delta_const",
        type=float,
        default=0.2,
        help="constraint 목표를 best 대비 얼마나 여유 있게 둘지(예: 0.2 = 20%).",
    )
    parser.add_argument(
        "--auto_aux_delta_dir",
        type=float,
        default=0.05,
        help="diffusion(앵커) 목표 여유(예: 0.05 = 5%).",
    )
    parser.add_argument(
        "--auto_aux_deadzone",
        type=float,
        default=0.05,
        help="ratio가 1±deadzone 안이면 가중치 업데이트를 하지 않습니다(출렁임 방지).",
    )
    parser.add_argument(
        "--auto_aux_eta",
        type=float,
        default=0.1,
        help="가중치 곱셈 업데이트 강도(작게 둘수록 '천천히' 변합니다).",
    )
    parser.add_argument(
        "--auto_aux_anchor_gamma",
        type=float,
        default=0.1,
        help="diffusion(앵커)이 best 대비 확실히 나빠지면(aux 가중치 '증가'만 금지) 기준 비율.",
    )
    parser.add_argument(
        "--auto_aux_eps_int",
        type=float,
        default=1e-4,
        help="integration 목표(target)의 최소 바닥값(0 근처로 망가지는 것 방지).",
    )
    parser.add_argument(
        "--auto_aux_eps_const",
        type=float,
        default=1e-4,
        help="constraint 목표(target)의 최소 바닥값(0 근처로 망가지는 것 방지).",
    )
    parser.add_argument(
        "--auto_aux_eps_dir",
        type=float,
        default=1e-4,
        help="diffusion(앵커) 목표(target)의 최소 바닥값.",
    )

    # constraint 가중치 범위(곱셈 업데이트를 쓰므로 min은 0보다 커야 함)
    parser.add_argument(
        "--w_const_min",
        type=float,
        default=0.01,
        help="constraint 가중치 최솟값(0이면 곱셈 업데이트가 영원히 0이라 반드시 양수 권장).",
    )
    parser.add_argument(
        "--w_const_max",
        type=float,
        default=1.0,
        help="constraint 가중치 최댓값(폭주 방지).",
    )


    # DataLoader parameters
    # DataLoader parameters
    parser.add_argument('--augment_prob',
                        type=float,
                        help='augmentation probability',
                        default=0.5)
    parser.add_argument(
        '--diffusion_time_eps',
        type=float,
        default=1e-3,
        help='diffusion 시간 t가 0/1에 너무 붙지 않게 피하는 작은 값(학습/추론 공통).',
    )

    parser.add_argument('--feasible_learn_noise_thresh',
                        type=float,
                        help='feasible_learn_noise_thresh',
                        default=0.33)
    parser.add_argument('--normalization_file_path',
                        default='normalization.json',
                        help='filepath of normalization.json',
                        type=str)
    parser.add_argument('--use_amortized_diffusion', default=True, type=boolean)
    parser.add_argument('--validate_scenario_rollouts', default=False,
                        type=boolean)
    parser.add_argument('--do_ego_predict', default=True, type=boolean)
    parser.add_argument('--use_ego_data_augment', default=False, type=boolean)
    parser.add_argument('--use_npc_data_augment', default=False, type=boolean)
    parser.add_argument('--scenario_finish_step', default=-1, type=int)
    parser.add_argument('--rollout_time_chunk_size', default=1, type=int)
    parser.add_argument('--num_workers', default=3, type=int)
    parser.add_argument('--prefetch_factor', type=int, default=8)

    parser.add_argument(
        '--use_agent_route_lane_order',
        default=False,
        type=boolean,
        help=(
            "True이면 agent_route_lane_order를 dataset/배치에 포함합니다. "
            "학습에서 사용하지 않으면 False로 두면 CPU/RAM 사용을 줄일 수 있습니다."
        ),
    ) # use_past_for_feasible = True / use_past_dit_input = True

    parser.add_argument('--use_fallback', default=True, type=boolean)
    parser.add_argument('--pin_mem', default=True, type=boolean)
    parser.add_argument('--set_coord_as_center', default=True, type=boolean)
    parser.add_argument('--use_feasible', default=True, type=boolean)
    parser.add_argument('--use_feasible_dl', default=True, type=boolean)
    parser.add_argument('--use_feasible_filter', default=True, type=boolean)
    parser.add_argument('--use_past_for_feasible', default=True, type=boolean)
    parser.add_argument('--use_batch_integration', default=True, type=boolean)
    parser.add_argument('--use_past_dit_input', default=True, type=boolean)
    parser.add_argument('--use_current_input', default=True, type=boolean)
    parser.add_argument('--feasible_grad_to_dit', default=False, type=boolean)
    parser.add_argument('--use_huber_loss', default=True, type=boolean)
    parser.add_argument('--profile_feasible', default=False, type=boolean)
    parser.add_argument('--use_vel_input', default=False, type=boolean)
    parser.add_argument('--use_pram', default=True, type=boolean)
    parser.add_argument('--use_route_lanes', default=True, type=boolean)
    parser.add_argument('--use_ego_plan', default=False, type=boolean)
    parser.add_argument('--use_integration_trajectory',
                        default=True,
                        type=boolean)
    parser.add_argument('--use_guidance', default=False, type=boolean)
    parser.add_argument('--use_feasible_blend', default=False, type=boolean)

    # Training
    parser.add_argument('--seed',
                        type=int,
                        help='fix random seed',
                        default=3407)
    parser.add_argument('--sampler_epoch_offset',
                        type=int,
                        help='fix random sampler_epoch_offset',
                        default=0)
    parser.add_argument('--weight_decay',
                        type=float,
                        default=1e-2,
                        help='AdamW weight decay for decayed params')
    parser.add_argument('--filter_radius',
                        type=float,
                        default=150.,
                        help='filter_radius meter')
    parser.add_argument(
        '--use_filter_radius',
        default=True,
        type=boolean,
        help=("True이면 filter_radius 반경(m) 안의 요소만 남기도록 필터링을 수행합니다. "
              "False이면 filter_radius 값과 상관없이 필터링을 하지 않습니다."),
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        default=True,
        type=boolean,
        help='True이면 AdamW 옵티마이저 상태를 8비트로 저장해 GPU 메모리 사용을 줄입니다.')

    parser.add_argument('--train_epochs',
                        type=int,
                        help='epochs of training',
                        default=550)
    parser.add_argument('--save_utd',
                        type=int,
                        help='save frequency',
                        default=15)
    parser.add_argument('--xy_yaw_metric_interval_steps',
                        type=int,
                        help='save frequency',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size (default: 2048)',
                        default=1024)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate (default: 5e-4)',
                        default=5e-4)
    parser.add_argument(
        '--min_learning_rate',
        type=float,
        default=None,
        help='cosine 스케줄에서 사용할 최소 learning rate. '
        'None이면 max lr의 0.2배를 사용합니다.',
    )

    parser.add_argument('--warm_up_epoch',
                        type=int,
                        help='number of warm up',
                        default=10)
    # ===== Stage config / LR scheduler / 모듈 그룹 설정 =====
    parser.add_argument(
        '--stage_config_path',
        type=str,
        default=None,
        help='단일 stage 학습 설정이 들어 있는 json/yaml 파일 경로.',
    )
    parser.add_argument(
        '--lr_scheduler_type',
        type=str,
        choices=['cosine', 'uniform'],
        default='cosine',
        help='learning rate 스케줄 형태 ("cosine" 또는 "uniform").',
    )

    parser.add_argument(
        '--use_lr_warmup',
        type=boolean,
        default=True,
        help='True이면 학습 초기에 learning rate 선형 warmup을 사용합니다.',
    )

    parser.add_argument(
        '--freeze_encoder_local',
        type=boolean,
        default=False,
        help='True이면 encoder_local(Group A)을 학습에서 제외합니다.',
    )

    parser.add_argument(
        '--encoder_local_lr_scale',
        type=float,
        default=1.0,
        help='encoder_local(Group A)에 곱해질 lr 배율 (예: 0.1).',
    )

    parser.add_argument(
        '--encoder_global_lr_scale',
        type=float,
        default=1.0,
        help='encoder_global(Group B)에 곱해질 lr 배율.',
    )

    parser.add_argument(
        '--decoder_lr_scale',
        type=float,
        default=1.0,
        help='decoder(Group C)에 곱해질 lr 배율.',
    )


    parser.add_argument('--encoder_drop_path_rate',
                        type=float,
                        help='encoder drop out rate',
                        default=0.1)

    parser.add_argument('--decoder_drop_path_rate',
                        type=float,
                        help='decoder drop out rate',
                        default=0.1)
    parser.add_argument('--feasible_stride_dt',
                        type=float,
                        help='feasible_stride_dt',
                        default=0.1)

    parser.add_argument('--device',
                        type=str,
                        help='run on which device (default: cuda)',
                        default='cuda')

    parser.add_argument('--use_ema', default=True, type=boolean)
    parser.add_argument('--load_ema_for_model', default=True, type=boolean)
    parser.add_argument('--load_ema_for_ema_model', default=True, type=boolean)
    parser.add_argument('--remove_existing_wb_weight',
                        default=False,
                        type=boolean)
    parser.add_argument('--delete_wb_weight_when_running',
                        default=False,
                        type=boolean)

    # ===== DeepSpeed / ZeRO-2 관련 설정 =====
    parser.add_argument("--use_deepspeed",
                        type=boolean,
                        default=True,
                        help="DeepSpeed ZeRO-2로 학습할지 여부")

    parser.add_argument("--grad_accum_steps",
                        type=int,
                        default=1,
                        help="optimizer.step 한 번 전에 몇 step을 모아서 사용할지")

    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=1.0,
                        help="gradient clipping 기준값")
    parser.add_argument("--stats_accel_v_stop_vehicle_mps",
                        type=float,
                        default=1.5,
                        help="gradient clipping 기준값")
    parser.add_argument("--stats_accel_v_stop_bicycle_mps",
                        type=float,
                        default=0.5,
                        help="gradient clipping 기준값")
    parser.add_argument("--stats_accel_v_stop_pedestrian_mps",
                        type=float,
                        default=0.1,
                        help="gradient clipping 기준값")

    parser.add_argument("--zero_offload_optimizer",
                        type=boolean,
                        default=False,
                        help="ZeRO-2 옵티마 상태를 CPU 메모리로 일부 옮겨서 GPU 메모리를 더 아낄지 여부")
    parser.add_argument("--do_data_statistics",
                        type=boolean,
                        default=False)
    parser.add_argument("--ds_steps_per_print",
                        type=int,
                        default=10000,
                        help="DeepSpeed가 내부 로그를 몇 step마다 출력할지")

    parser.add_argument("--ds_allgather_bucket_size",
                        type=float,
                        default=2e8,
                        help="ZeRO allgather bucket 크기(바이트 단위)")

    parser.add_argument("--ds_reduce_bucket_size",
                        type=float,
                        default=2e8,
                        help="ZeRO reduce-scatter bucket 크기(바이트 단위)")
    # Model
    parser.add_argument('--encoder_depth',
                        type=int,
                        help='number of encoding layers',
                        default=3)
    parser.add_argument('--decoder_depth',
                        type=int,
                        help='number of decoding layers',
                        default=4)
    parser.add_argument('--num_heads',
                        type=int,
                        help='number of multi-head',
                        default=8)
    parser.add_argument('--hidden_dim',
                        type=int,
                        help='hidden dimension',
                        default=192)
    parser.add_argument('--diffusion_model_type',
                        type=str,
                        help='type of diffusion model [x_start, score]',
                        choices=['score', 'x_start', "v"],
                        default='x_start')

    parser.add_argument('--use_wandb', default=True, type=boolean)
    parser.add_argument('--verbose', default=True, type=boolean)
    parser.add_argument('--notes', default='', type=str)
    parser.add_argument('--entity', default='jksg01019-naver-labs', type=str)
    parser.add_argument('--project', default='Diffusion-Planner', type=str)
    parser.add_argument('--histogram_output_dir', default='histograms', type=str)

    # distributed training parameters
    parser.add_argument('--ddp',
                        default=True,
                        type=boolean,
                        help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    # caching
    parser.add_argument(
        '--scenarios_cache_in',  # 2) 불러올 파일
        type=str,
        default='scenarios_cache.pkl',  #None,
        help='미리 저장해둔 시나리오 *.pkl 경로 (지정 시 DB 로딩 건너뜀)',
    )
    parser.add_argument(
        '--scenarios_cache_out',  # 1) 저장할 파일
        type=str,
        default="scenarios_cache.pkl",  #'scenarios_cache.pkl',
        help='새로 추출한 시나리오를 저장할 *.pkl 경로',
    )
    parser.add_argument('--womd_data_path',
                        default="/home/user/womd_v1_3",
                        type=str,
                        help='path to raw data')
    parser.add_argument("--overwrite_womd_cache",
                        type=str,
                        default="false",
                        help="true/false")
    parser.add_argument(
        "--womd_splits",
        type=str,
        default="validation,testing",  #,validation,testing",#"
        help="예: training,validation (콤마로 구분)",
    )

    parser.add_argument('--data_path',
                        default='/data/nuplan-v1.1/trainval',
                        type=str,
                        help='path to raw data')
    parser.add_argument('--map_path',
                        default='/data/nuplan-v1.1/maps',
                        type=str,
                        help='path to map data')
    parser.add_argument('--save_path',
                        default='./cache',
                        type=str,
                        help='path to save processed data')
    parser.add_argument('--save_cache_path',
                        default='./cache',
                        type=str,
                        help='path to save processed data')
    parser.add_argument('--scenarios_per_type',
                        type=int,
                        default=None,
                        help='number of scenarios per type')
    parser.add_argument('--total_scenarios',
                        type=int,
                        default=1,
                        help='limit total number of scenarios')
    parser.add_argument('--fine_tune_gen_k',
                        type=int,
                        default=64)
    parser.add_argument('--eval_temperature',
                        type=float,
                        default=1.0)
    parser.add_argument('--fine_tune_temperature',
                        type=float,
                        default=0.5)
    parser.add_argument('--time_step_for_compare',
                        type=int,
                        default=20)
    parser.add_argument('--time_step_for_recover',
                        type=int,
                        default=30)
    parser.add_argument('--shuffle_scenarios',
                        type=boolean,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--use_timestep_weight_loss',
                        type=boolean,
                        default=False)
    parser.add_argument('--reset_save_path',
                        type=boolean,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--save_image',
                        type=boolean,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--save_video',
                        type=boolean,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--finish_when_no_updated_pt',
                        type=boolean,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--make_statistics_when_caching',
                        default=False,
                        type=boolean)
    parser.add_argument('--pose_based', default=True, type=boolean)
    parser.add_argument('--die_all', default=True, type=boolean)
    parser.add_argument('--select_jointly', default=True, type=boolean)
    parser.add_argument('--use_recovery', default=False, type=boolean)
    parser.add_argument('--recovery_threshold_m',
                        type=float,
                        default=3.)
    parser.add_argument('--guidance_scale',
                        type=float,
                        default=1.)
    parser.add_argument('--dp_sample_k_candidate_batch_size',
                        type=float,
                        default=-1.)
    args = parser.parse_args()
    # ★ stage config(json/yaml)로 CLI 인자 덮어쓰기
    args = _override_args_with_stage_config(args)
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args
