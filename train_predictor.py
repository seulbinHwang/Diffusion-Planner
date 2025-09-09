import os
# 128 MiB 단위로 메모리 청크를 잘라서 할당하도록 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
# DDP 디버깅을 위해 사용되지 않은 파라미터 정보를 상세히 출력
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")
import torch
# TensorFloat-32(TF32) 연산을 허용하여
#   - Ampere(A100 등) GPU에서 matmul/cuDNN 연산을 FP32보다 빠르게 처리하고
#   - 눈에 띄는 정밀도 손실 없이 학습·추론 속도를 높이기 위한 설정입니다.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision('high')  # PyTorch>=2.0

import argparse
import shutil
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
from requests.exceptions import HTTPError
from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.npc_data_augmentation import NPCStatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp

from diffusion_planner.train_epoch import train_epoch


import math

def pnn_schedule(epoch: int,
                 total_epochs: int,
                 *,
                 base: int = 32,           # 시작 Pnn
                 cap: int,                 # 최종 Pnn (보통 args.predicted_neighbor_num)
                 step: int = 32,           # Pnn 증분 단위(캐시/모델 구조와 맞추기)
                 ramp_fraction: float = 0.30  # 총 학습의 몇 % 지점에서 cap 도달할지
                 ) -> int:
    """
    커리큘럼: [0 .. ramp_end-1] 구간에서 base→cap을 'step' 단위로 균등 분할해서 올리고,
             ramp_end 이후엔 cap으로 고정한다.
    - epoch: 0-indexed
    - total_epochs: 전체 학습 epoch 수(예: 500)
    - base/cap/step: (base + k*step)이 cap을 넘지 않도록 설계
    - ramp_fraction: cap에 도달할 시점(비율). 예: 0.30 → 500×0.30≈150epoch에 cap 도달

    장점:
    - 학습 초기 충분한 '쉬운 구간'을 확보 (안정·속도 유리)
    - 전체의 60~80%는 풀 난이도에서 수렴 (성능 유지)
    """
    cap = int(cap)
    base = int(base)
    step = int(step)
    total_epochs = int(total_epochs)
    ramp_end = max(1, int(round(total_epochs * ramp_fraction)))  # cap 도달 epoch (0-index 기준 ramp_end-1)

    if cap <= base:
        return cap  # 방어: 이미 최대 이하라면 그대로

    steps_needed = math.ceil((cap - base) / step)  # 몇 번 올려야 cap에 도달하는지
    # 0..(ramp_end-1) 구간을 steps_needed 칸으로 균등 분할하여 인덱스 계산
    # epoch=ramp_end-1일 때 idx=steps_needed가 되도록 설계
    idx = 0 if ramp_end == 0 else min(
        steps_needed,
        math.floor(((epoch + 1) / ramp_end) * steps_needed)
    )

    pnn = base + idx * step
    if pnn > cap:
        pnn = cap
    return int(pnn)

# === [ADD] Pnn 기반 배치 크기 스케줄러 ==================================
def bs_schedule_by_pnn(pnn_curr: int,
                       pnn_base: int,
                       bs_base: int,
                       *,
                       world_size: int = 1,
                       alpha: float = 1.5,
                       min_per_rank: int = 1,
                       max_global: int = None):
    """
    메모리 예산을 대략 일정하게 유지하기 위해, Pnn(=시퀀스 길이)↑ 시 global batch를 줄입니다.
      - pnn_base: pnn_schedule의 시작값(예: 32)과 일치시킬 것
      - bs_base : 기준 '글로벌' 배치 크기(args.batch_size)
      - alpha   : 메모리 증가 근사 지수(1~2 권장; self-attn 지배면 2에 가까움)
    반환:
      - (bs_global_now, bs_per_rank_now)
    """
    if pnn_curr <= 0:
        pnn_curr = pnn_base
    scale = (float(pnn_base) / float(pnn_curr)) ** float(alpha)
    bs_global = int(max(world_size, round(bs_base * scale)))
    if max_global is not None:
        bs_global = min(bs_global, int(max_global))
    # DDP 정합: world_size 배수로 내림
    bs_global = (bs_global // world_size) * world_size
    bs_per_rank = max(min_per_rank, bs_global // world_size)
    return bs_global, bs_per_rank
# =======================================================================


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_parameter_index_mapping(model):
    print("Parameter index mapping:")
    for idx, (name, _) in enumerate(model.named_parameters()):
        print(f"{idx}: {name}")


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
    parser.add_argument('--normalization_file_path',
                        default='normalization.json',
                        help='filepath of normalization.json',
                        type=str)
    parser.add_argument('--use_ego_data_augment', default=False, type=boolean)
    parser.add_argument('--use_npc_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument(
        '--pin-mem',
        action='store_true',
        help=
        'Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.'
    )
    parser.add_argument('--no-pin-mem',
                        action='store_false',
                        dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # Training
    parser.add_argument('--seed',
                        type=int,
                        help='fix random seed',
                        default=3407)
    parser.add_argument('--train_epochs',
                        type=int,
                        help='epochs of training',
                        default=500)
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
                        default=5)
    parser.add_argument('--prefetch_factor',
                        type=int,
                        help='number of warm up',
                        default=2)
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


def safe_get_artifacts(api, type_name, path):
    try:
        return list(api.artifacts(type_name, path))
    except (wandb.errors.CommError, HTTPError) as e:
        # 404 또는 권한 오류 → 컬렉션이 아직 없다고 판단
        print(f"[SKIP] '{path}' 컬렉션 없음/권한 문제: {e}")
        return []


def purge_collection(api, entity, project, coll_name):
    path = f"{entity}/{project}/{coll_name}"
    versions = safe_get_artifacts(api, "model", path)  # 신규 API 사용
    if not versions:
        print(f"[PURGE] {coll_name}: 삭제할 버전이 없습니다.")
        return

    deleted_count = 0
    failed_count = 0

    for art in versions:
        try:
            # alias가 있는 artifact의 경우 alias를 먼저 제거
            if hasattr(art, 'aliases') and art.aliases:
                print(
                    f"[PURGE] {coll_name}: Artifact {art.id}에 alias가 있어 alias를 먼저 제거합니다: {art.aliases}"
                )
                for alias in art.aliases:
                    try:
                        art.delete_alias(alias)
                        print(f"[PURGE] {coll_name}: Alias '{alias}' 제거 완료")
                    except Exception as alias_err:
                        print(
                            f"[PURGE] {coll_name}: Alias '{alias}' 제거 실패: {alias_err}"
                        )

            # artifact 삭제 시도
            art.delete()
            deleted_count += 1
            print(f"[PURGE] {coll_name}: Artifact {art.id} 삭제 완료")

        except Exception as e:
            failed_count += 1
            print(f"[PURGE] {coll_name}: Artifact {art.id} 삭제 실패 - {str(e)}")
            # alias가 있는 경우의 오류는 경고로만 처리하고 계속 진행
            if "due to existing alias" in str(e):
                print(
                    f"[PURGE] {coll_name}: Alias로 인한 삭제 실패는 정상적인 상황입니다. 계속 진행합니다."
                )

    print(f"[PURGE] {coll_name}: 삭제 완료 {deleted_count}개, 실패 {failed_count}개")


def model_training(args):
    best_loss = float('inf')
    torch.cuda.empty_cache()
    # init ddp
    global_rank, rank, _ = ddp.ddp_setup_universal(True, args)

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_local_path_model_path is not None:
            # resume_local_path_model_path: ./training_log/npc_aug_n_ego_past/2025-08-06-07:23:04/
            save_path = args.resume_local_path_model_path
            # 폴더명 자체가 타임스탬프이므로 그대로 재사용
            time_str = os.path.basename(
                os.path.normpath(args.resume_local_path_model_path))
        else:
            from datetime import datetime
            time = datetime.now()
            time = time.strftime("%Y-%m-%d-%H:%M:%S")
            time_str = datetime.now().strftime("%Y-%m-%d-%H-%M")

            save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)
        args.save_path = save_path
        # Save args
        args_dict = vars(args)
        args_dict = {
            k:
                v if not isinstance(v, (StateNormalizer, ObservationNormalizer))
                else v.to_dict() for k, v in args_dict.items()
        }

        from mmengine.fileio import dump
        dump(args_dict,
             os.path.join(save_path, 'args.json'),
             file_format='json',
             indent=4)
    else:
        save_path = None
        args.save_path = None

    # set seed
    set_seed(args.seed + global_rank)
    # Enable anomaly detection to trace NaN/Inf origins during training
    # torch.autograd.set_detect_anomaly(True)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    # set up data loaders
    if args.use_ego_data_augment and args.use_npc_data_augment:
        raise ValueError(
            "You cannot use both ego and npc data augmentation at the same time. "
        )
    if args.use_ego_data_augment:
        aug = StatePerturbation(augment_prob=args.augment_prob,
                                device=args.device)
    elif args.use_npc_data_augment:
        aug = NPCStatePerturbation(augment_prob=args.augment_prob,
                                   device=args.device)
    else:
        aug = None
    train_set = DiffusionPlannerData(args.train_set, args.train_set_list,
                                     args.agent_num,
                                     args.predicted_neighbor_num,
                                     args.future_len)
    train_sampler = DistributedSampler(train_set,
                                       num_replicas=ddp.get_world_size(),
                                       rank=global_rank,
                                       shuffle=True)
    train_loader = DataLoader(train_set,
                              sampler=train_sampler,
                              batch_size=batch_size // ddp.get_world_size(),
                              num_workers=args.num_workers,
                              prefetch_factor=args.prefetch_factor,
                              pin_memory=args.pin_mem,
                              drop_last=True)

    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(rank if args.device ==
                                             'cuda' else args.device)

    if args.ddp:
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])
    model_ema = None
    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    if global_rank == 0:
        model_to_print = ddp.get_model(diffusion_planner, args.ddp)
        print("Model Params: {}".format(
            sum(p.numel() for p in model_to_print.parameters())))
        print_parameter_index_mapping(model_to_print)

    # optimizer
    params = [{
        'params': ddp.get_model(diffusion_planner, args.ddp).parameters(),
        'lr': args.learning_rate
    }]

    try:
        optimizer = optim.AdamW(params, fused=True)
    except (TypeError, RuntimeError):  # fused 미지원 환경
        optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs,
                                              args.warm_up_epoch)

    allow_val_change = False
    if args.resume_local_path_model_path is not None:
        print(f"Model loaded from {args.resume_local_path_model_path}")
        (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id,
         model_ema) = resume_model(args.resume_local_path_model_path,
                                   diffusion_planner, optimizer, scheduler,
                                   model_ema, args.device)
        if args.resume_model_from_wandb:
            allow_val_change = True
    else:
        init_epoch = 0
        wandb_id = None

    # logger
    wandb_logger = Logger(args.name,
                          args.notes,
                          args,
                          wandb_resume_id=wandb_id,
                          save_path=save_path,
                          rank=global_rank,
                          allow_val_change=allow_val_change)
    if global_rank == 0 and args.remove_existing_wb_weight:
        api = wandb.Api()

        # 현재 run의 entity / project
        entity = wandb.run.entity
        project = wandb.run.project

        # 우리가 새로 만들 컬렉션 이름 (time_str는 앞에서 한 번 계산됨)
        purge_collection(api, entity, project, f"{args.name}_latest-model")
        purge_collection(api, entity, project, f"{args.name}_best-model")
    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        # === [ADD] 이 에폭에서 사용할 예측 대상 수(Pnn_curr)와 컨텍스트 클립 수 ===
        # args.curr_predicted_neighbor_num = pnn_schedule(
        #     epoch, train_epochs,
        #     base=32,
        #     cap=args.predicted_neighbor_num,  # 보통 448
        #     step=32,
        #     ramp_fraction=0.30,  # 0.20~0.40 사이에서 취향과 자원에 맞게 조절
        # )
        # args.curr_predicted_neighbor_num = args.predicted_neighbor_num
        # === [ADD] Pnn에 맞춘 동적 배치 크기 계산 & DataLoader 재생성 ======
        # bs_global_now, bs_per_rank_now = bs_schedule_by_pnn(
        #     pnn_curr=args.curr_predicted_neighbor_num,
        #     pnn_base=32,                     # pnn_schedule 시작값과 일치
        #     bs_base=args.batch_size,         # "글로벌" 배치 크기 기준
        #     alpha=getattr(args, "bs_pnn_alpha", 1.5),
        #     world_size=ddp.get_world_size(),
        #     min_per_rank=1,
        # )

        # DDP sampler에 epoch 설정(셔플 시드 고정은 '현재 에폭' 직전에 호출)
        # train_sampler.set_epoch(epoch)

        # 이 에폭에 사용할 DataLoader를 '그때그때' 생성
        # train_loader = DataLoader(
        #     train_set,
        #     sampler=train_sampler,
        #     batch_size=bs_per_rank_now,
        #     num_workers=args.num_workers,
        #     prefetch_factor=args.prefetch_factor,
        #     pin_memory=args.pin_mem,
        #     drop_last=True,
        # )
        # ===================================================================


        # if global_rank == 0:
        #     clip_num = min(args.agent_num, args.curr_predicted_neighbor_num * 2)
        #     print(
        #         f"[Curriculum] Pnn_curr={args.curr_predicted_neighbor_num} | "
        #         f"context_clip={clip_num} (<= agent_num={args.agent_num}) | "
        #         f"batch(per-rank/global)={bs_per_rank_now}/{bs_global_now} | "
        #         f"alpha={getattr(args,'bs_pnn_alpha',1.5):.2f}"
        #     )

        train_loss, train_total_loss = train_epoch(train_loader,
                                                   diffusion_planner, optimizer,
                                                   args, model_ema, aug)
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
        if global_rank == 0:
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            wandb_logger.log_metrics(
                {
                    f"train_loss/{k}": v for k, v in train_loss.items()
                },
                step=epoch + 1)
            wandb_logger.log_metrics({
                f"lr/{k}": v for k, v in lr_dict.items()
            },
                                     step=epoch + 1)

            if (epoch + 1) % args.save_utd == 1:
                save_best = False
                if train_total_loss < best_loss:
                    best_loss = train_total_loss
                    save_best = True
                # save model at the end of epoch
                save_model(diffusion_planner, optimizer, scheduler, save_path,
                           epoch, train_total_loss, wandb_logger.id,
                           model_ema.ema if model_ema is not None else None, save_best)
                print(f"Model saved in {save_path}\n")
                # ── latest-model 아티팩트 (매번 덮어쓰기) ──
                # save_path = f"{args.save_dir}/training_log/{args.name}/{time}/"
                # f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth'
                latest_coll = f"{args.name}_latest-model"
                latest_art = wandb.Artifact(
                    name=latest_coll,  # latest-model 라는 이름의 데이터 묶음
                    type="model",  # "dataset" 이 될수도 있음
                    metadata={
                        "time_str": time_str,
                        "epoch": epoch + 1,
                        "loss": train_total_loss
                    })
                # 로컬에 저장된 latest.pth 파일을 담아 넣는 동작
                latest_art.add_file(os.path.join(save_path, "latest.pth"))
                """
이 상자(Artifact)를 W&B 서버로 전송하고, ["latest"]라는 별명을 붙여 줘요.
별명을 쓰면, 다음에 다시 “latest”라는 이름으로 덮어써 가며 하나의 모델만 관리할 수 있습니다.
                """
                wandb.log_artifact(latest_art, aliases=["latest"])
                latest_art.wait()  # 업로드 완료 보장

                if args.delete_wb_weight_when_running:
                    # 이전 버전 삭제
                    api = wandb.Api()
                    # wandb.run.entity: jksg01019-naver-labs
                    # wandb.run.project: Diffusion-Planner
                    entity = wandb.run.entity
                    project = wandb.run.project
                    # ':latest' alias로 가져오면 방금 올린 버전이 리턴됩니다

                    current = api.artifact(
                        f"{entity}/{project}/{latest_coll}:latest")

                    # 3) 모든 버전 목록 중, 이 버전이 아닌 나머지를 삭제
                    for v in api.artifacts("model",
                                           f"{entity}/{project}/{latest_coll}"):
                        if v.id != current.id:
                            v.delete()
                # ── best-model 아티팩트 (조건부 덮어쓰기) ──
                if save_best:
                    best_coll = f"{args.name}_best-model"
                    best_art = wandb.Artifact(name=best_coll,
                                              type="model",
                                              metadata={
                                                  "time_str": time_str,
                                                  "epoch": epoch + 1,
                                                  "loss": train_total_loss
                                              })
                    best_art.add_file(os.path.join(save_path, "best.pth"))
                    wandb.log_artifact(best_art, aliases=["best"])
                    best_art.wait()  # 업로드 완료 보장

                    # 이전 버전 삭제
                    if args.delete_wb_weight_when_running:
                        entity = wandb.run.entity
                        project = wandb.run.project
                        # ':latest' alias로 가져오면 방금 올린 버전이 리턴됩니다
                        current = api.artifact(
                            f"{entity}/{project}/{best_coll}:best")

                        # 3) 모든 버전 목록 중, 이 버전이 아닌 나머지를 삭제
                        for v in api.artifacts(
                                "model", f"{entity}/{project}/{best_coll}"):
                            if v.id != current.id:
                                v.delete()

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)

    # ── 모든 훈련 종료 후 정리 ─
    torch.distributed.barrier()  # ① 모든 rank의 학습 루프 종료 동기화

    # ② 모든 rank에서 wandb 종료 (사용 시)
    if args.use_wandb:
        wandb.finish()
        if global_rank == 0:
            wandb_logger.finish()  # TensorBoard Logger의 writer.close() 호출

    torch.distributed.barrier()  # ③ 모든 rank의 wandb 종료 동기화

    # ④ Rank 0에서만 로컬 파일 정리
    if global_rank == 0 and args.save_path:
        print("[CLEANUP] 훈련 종료 후 로컬 체크포인트 및 로그 정리 시작")
        # TensorBoard 로그·체크포인트 일괄 삭제
        for f in ["latest.pth", "best.pth", "args.json"]:
            p = os.path.join(args.save_path, f)
            try:
                os.remove(p)
                print(f"[CLEANUP] 파일 삭제: {p}")
            except FileNotFoundError:
                print(f"[CLEANUP] 파일 없음 (이미 삭제됨): {p}")
            except Exception as e:
                print(f"[CLEANUP] 파일 삭제 오류: {p}, {e}")

        tb_dir = os.path.join(args.save_path, "tb")
        try:
            shutil.rmtree(tb_dir)
            print(f"[CLEANUP] 디렉터리 삭제: {tb_dir}")
        except FileNotFoundError:
            print(f"[CLEANUP] 디렉터리 없음 (이미 삭제됨): {tb_dir}")
        except Exception as e:
            print(f"[CLEANUP] 디렉터리 삭제 오류: {tb_dir}, {e}")


if __name__ == "__main__":

    args = get_args()

    if args.resume_model_from_wandb:
        # resume_model_from_wandb 가 "latest"가 아니면 에러를 발생시킵니다.
        if args.resume_model_from_wandb not in ['latest']:
            raise ValueError("args.resume_model_from_wandb must be 'latest.")
        if not args.name:
            raise ValueError(
                "args.name must be provided to resume from a wandb artifact.")

        print(
            f"Resuming from wandb artifact: {args.name}:{args.resume_model_from_wandb}"
        )

        # W&B Public API를 사용하여 아티팩트와 원본 Run의 config를 가져옵니다.
        api = wandb.Api()

        try:
            # resume_model_from_wandb 값('best', 'latest' 등)에 따라 컬렉션과 파일 이름을 결정합니다.
            resume_alias = args.resume_model_from_wandb
            if resume_alias == 'best':
                collection_name = f"{args.name}_best-model"
                checkpoint_filename = "best.pth"
            else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
                collection_name = f"{args.name}_latest-model"
                checkpoint_filename = "latest.pth"

            # Public API를 사용해 아티팩트를 가져오려면 entity와 project 정보가 필요합니다.
            # 임시 Run을 생성하여 컨텍스트(entity, project)를 얻어옵니다.
            temp_run_for_context = wandb.init(project="Diffusion-Planner",
                                              name=f"temp_api_run_{args.name}",
                                              job_type="api_access")
            # wandb.run.entity: jksg01019-naver-labs
            # wandb.run.project: Diffusion-Planner
            entity = temp_run_for_context.entity
            project = temp_run_for_context.project
            temp_run_for_context.finish()

            artifact_path = f"{entity}/{project}/{collection_name}:{resume_alias}"
            print(f"아티팩트 경로에서 가져오는 중: {artifact_path}")
            # Public API를 통해 아티팩트 객체를 가져옵니다.
            artifact = api.artifact(artifact_path, type='model')

            # 이 아티팩트를 생성한 원본 Run을 가져옵니다.
            source_run = artifact.logged_by()

            # 원본 Run의 config에서 save_path를 가져옵니다.
            if 'save_path' in source_run.config:
                save_path = source_run.config['save_path']
                print(f"원본 Run의 config에서 save_path를 찾았습니다: {save_path}")
                os.makedirs(save_path, exist_ok=True)

                # 다운로드될 체크포인트 파일의 전체 경로를 지정합니다.
                target_file_path = os.path.join(save_path, checkpoint_filename)

                # 만약 해당 경로에 파일이 이미 존재하면, 덮어쓰기를 위해 삭제합니다.
                if os.path.exists(target_file_path):
                    print(f"기존 파일 '{target_file_path}'가 존재하여 삭제하고 새로 다운로드합니다.")
                    os.remove(target_file_path)

                # 아티팩트 파일을 다운로드하기 위해 임시 Run이 필요합니다.
                download_run = wandb.init(
                    project=project,
                    name=f"resume_run_download_{args.name}",
                    resume="allow")
                # 현재 Run에서 사용할 아티팩트를 지정합니다.
                artifact_for_download = download_run.use_artifact(
                    artifact, aliases=[resume_alias])
                # 원본 save_path에 아티팩트 파일을 다운로드합니다.
                artifact_for_download.download(root=save_path)
                download_run.finish()
                print(f"아티팩트 '{artifact_path}'을(를) {save_path}에 다운로드했습니다.")

                # 다운로드된 체크포인트의 전체 경로를 설정합니다.
                model_path = os.path.join(save_path, checkpoint_filename)
                if not os.path.exists(model_path):
                    downloaded_files = os.listdir(save_path)
                    raise FileNotFoundError(
                        f"다운로드된 아티팩트 디렉터리에서 '{checkpoint_filename}'을(를) 찾을 수 없습니다: {save_path}. "
                        f"사용 가능한 파일: {downloaded_files}")
                save_path = os.path.dirname(os.path.abspath(model_path))
                args.resume_local_path_model_path = save_path

                print(
                    f"아티팩트를 {save_path}에 다운로드했습니다. 체크포인트에서 학습을 재개합니다: {args.resume_local_path_model_path}"
                )
            else:
                raise ValueError("원본 Run의 config에 'save_path'가 없습니다.")

        except Exception as e:
            print(f"W&B에서 재개하는 동안 오류 발생: {e}")
            # 오류 발생 시 임시 Run이 종료되도록 보장합니다.
            if wandb.run:
                wandb.finish()
            raise e

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

    # Run
    model_training(args)
