import argparse
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
import os
from typing import Any, Dict


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


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
    parser.add_argument('--resume_local_path_model_path',
                        type=str,
                        help='path to resume model',
                        default=None)
    parser.add_argument(
        '--resume_model_from_wandb',
        type=str,
        help='wandb artifact version to resume from (e.g., "latest")',
        default=None)

    # Data
    parser.add_argument('--train_set',
                        type=str,
                        help='path to train data',
                        default=None)
    parser.add_argument('--train_set_list',
                        type=str,
                        help='data list of train data',
                        default=None)

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
    parser.add_argument('--max_agent_num',
                        type=int,
                        help='number of agents',
                        default=448)
    parser.add_argument('--predicted_neighbor_num',
                        type=int,
                        help='number of neighbor agents to predict',
                        default=448)

    parser.add_argument('--caching_max_agent_num',
                        type=int,
                        help='number of agents',
                        default=448)
    parser.add_argument('--do_not_clip',
                        default=True,
                        type=boolean,
                        help='True이면 neighbor/lanes/route_lanes 축 클리핑을 수행하지 않음')
    parser.add_argument('--static_objects_state_dim',
                        type=int,
                        help='state dim for static objects',
                        default=10)
    parser.add_argument('--static_objects_num',
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
    parser.add_argument('--lane_num',
                        type=int,
                        help='number of lanes',
                        default=250)
    parser.add_argument('--max_use_lane_num',
                        type=int,
                        help='number of lanes',
                        default=250)

    parser.add_argument('--route_len',
                        type=int,
                        help='number of route lane point',
                        default=20)
    parser.add_argument('--route_state_dim',
                        type=int,
                        help='state dim for route lane point',
                        default=12)
    parser.add_argument('--route_num',
                        type=int,
                        help='number of route lanes',
                        default=250)

    # DataLoader parameters
    parser.add_argument('--augment_prob',
                        type=float,
                        help='augmentation probability',
                        default=0.5)
    parser.add_argument('--feasible_learn_noise_thresh',
                        type=float,
                        help='feasible_learn_noise_thresh',
                        default=0.3)
    parser.add_argument('--normalization_file_path',
                        default='normalization.json',
                        help='filepath of normalization.json',
                        type=str)
    parser.add_argument('--use_ego_data_augment', default=False, type=boolean)
    parser.add_argument('--use_npc_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', default=True, type=boolean)
    parser.add_argument('--use_feasible', default=True, type=boolean)
    parser.add_argument('--use_feasible_dl', default=True, type=boolean)
    parser.add_argument('--use_feasible_filter', default=True, type=boolean)
    parser.add_argument('--use_past_for_feasible', default=True, type=boolean)
    parser.add_argument('--use_batch_integration', default=True, type=boolean)
    parser.add_argument('--use_current_input', default=True, type=boolean)
    parser.add_argument('--feasible_grad_to_dit', default=False, type=boolean)
    parser.add_argument('--use_direct_loss', default=True, type=boolean)
    parser.add_argument('--use_huber_loss', default=True, type=boolean)
    parser.add_argument('--profile_feasible', default=False, type=boolean)
    parser.add_argument('--use_vel_input', default=False, type=boolean)
    parser.add_argument('--use_pram', default=True, type=boolean)
    parser.add_argument('--use_route_lanes', default=True, type=boolean)
    parser.add_argument('--use_ego_plan', default=True, type=boolean)
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
    parser.add_argument(
        '--use_8bit_optimizer',
        default=True,
        type=boolean,
        help='True이면 AdamW 옵티마이저 상태를 8비트로 저장해 GPU 메모리 사용을 줄입니다.')

    parser.add_argument('--train_epochs',
                        type=int,
                        help='epochs of training',
                        default=360)
    parser.add_argument('--save_utd',
                        type=int,
                        help='save frequency',
                        default=20)
    parser.add_argument('--batch_size',
                        type=int,
                        help='batch size (default: 2048)',
                        default=1024)
    parser.add_argument('--learning_rate',
                        type=float,
                        help='learning rate (default: 5e-4)',
                        default=5e-4)
    parser.add_argument('--warm_up_epoch',
                        type=int,
                        help='number of warm up',
                        default=10)
    parser.add_argument('--prefetch_factor',
                        type=int,
                        help='number of warm up',
                        default=6)
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
    parser.add_argument('--remove_existing_wb_weight',
                        default=False,
                        type=boolean)
    parser.add_argument('--delete_wb_weight_when_running',
                        default=False,
                        type=boolean)

    # ===== DeepSpeed / ZeRO-2 관련 설정 =====
    parser.add_argument("--use_deepspeed",
                        type=boolean,
                        default=False,
                        help="DeepSpeed ZeRO-2로 학습할지 여부")

    parser.add_argument("--grad_accum_steps",
                        type=int,
                        default=1,
                        help="optimizer.step 한 번 전에 몇 step을 모아서 사용할지")

    parser.add_argument("--max_grad_norm",
                        type=float,
                        default=5.0,
                        help="gradient clipping 기준값")

    parser.add_argument("--zero_offload_optimizer",
                        type=boolean,
                        default=False,
                        help="ZeRO-2 옵티마 상태를 CPU 메모리로 일부 옮겨서 GPU 메모리를 더 아낄지 여부")

    parser.add_argument("--ds_steps_per_print",
                        type=int,
                        default=100,
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
                        default=3)
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
                        choices=['score', 'x_start'],
                        default='x_start')

    parser.add_argument('--use_wandb', default=True, type=boolean)
    parser.add_argument('--notes', default='', type=str)

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
    parser.add_argument('--scenarios_per_type',
                        type=int,
                        default=None,
                        help='number of scenarios per type')
    parser.add_argument('--total_scenarios',
                        type=int,
                        default=1,
                        help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios',
                        type=bool,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--reset_save_path',
                        type=bool,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--save_image',
                        type=bool,
                        default=False,
                        help='shuffle scenarios')
    parser.add_argument('--make_statistics_when_caching',
                        default=True,
                        type=boolean)
    args = parser.parse_args()
    if not args.use_direct_loss:
        assert not args.use_guidance and not args.use_feasible_blend, \
            "use_direct_loss가 False인 경우, use_guidance와 use_feasible_blend는 모두 False여야 합니다."
        assert args.feasible_grad_to_dit, \
            "use_direct_loss가 False인 경우, feasible_grad_to_dit는 True여야 합니다."
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    if getattr(args, "use_deepspeed", False):
        args.deepspeed_config = build_deepspeed_config(args)
    else:
        args.deepspeed_config = None

    return args
