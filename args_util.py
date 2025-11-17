import argparse
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer


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
    parser.add_argument('--agent_num',
                        type=int,
                        help='number of agents',
                        default=32)

    parser.add_argument('--static_objects_state_dim',
                        type=int,
                        help='state dim for static objects',
                        default=10)
    parser.add_argument('--static_objects_num',
                        type=int,
                        help='number of static objects',
                        default=5)

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
                        default=100)

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
                        default=25)

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
    parser.add_argument('--use_feasible_train', default=True, type=boolean)
    parser.add_argument('--use_feasible_filter', default=True, type=boolean)
    parser.add_argument('--use_past_for_feasible', default=False, type=boolean)
    parser.add_argument('--use_current_input', default=True, type=boolean)
    parser.add_argument('--use_huber_loss', default=True, type=boolean)
    parser.add_argument('--profile_feasible', default=True, type=boolean)
    parser.add_argument('--use_vel_input', default=False, type=boolean)
    parser.add_argument('--use_pram', default=True, type=boolean)
    parser.add_argument('--use_integration_trajectory', default=False, type=boolean)

    # parser.add_argument(
    #     '--pin-mem',
    #     action='store_true',
    #     help=
    #     'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    # )
    # parser.add_argument('--no-pin-mem',
    #                     action='store_false',
    #                     dest='pin_mem',
    #                     help='')
    # parser.set_defaults(pin_mem=True)

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
                        default=4)
    parser.add_argument('--encoder_drop_path_rate',
                        type=float,
                        help='encoder drop out rate',
                        default=0.1)
    parser.add_argument('--decoder_drop_path_rate',
                        type=float,
                        help='decoder drop out rate',
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

    # decoder
    parser.add_argument('--predicted_neighbor_num',
                        type=int,
                        help='number of neighbor agents to predict',
                        default=32)

    parser.add_argument('--use_wandb', default=True, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    parser.add_argument('--ddp',
                        default=True,
                        type=boolean,
                        help='use ddp or not')
    parser.add_argument('--port', default='22323', type=str, help='port')

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)

    return args
