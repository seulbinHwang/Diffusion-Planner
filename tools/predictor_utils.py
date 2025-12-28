import argparse
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
import os
import wandb
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_planner.utils import ddp
from timm.utils import ModelEma
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
import argparse
import os
from typing import Any, Dict
from datetime import datetime

import torch

def set_distributed_flag_from_env(args: argparse.Namespace) -> None:
    """환경변수 WORLD_SIZE를 읽어 args.distributed 플래그를 설정한다.

    WORLD_SIZE 가 존재하면
      - WORLD_SIZE > 1 이면 여러 프로세스를 쓰는 분산 학습으로 보고 True,
      - 그렇지 않으면 False 로 둔다.
    WORLD_SIZE 가 없으면 단일 프로세스 학습으로 간주하고 False 를 넣는다.
    """
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    else:
        args.distributed = False

def _download_wandb_checkpoint_to_local(
    args: argparse.Namespace,
    api: wandb.Api,
    entity: str,
    project: str,
    collection_name: str,
    resume_alias: str,
    checkpoint_filename: str,
) -> None:
    """지정한 W&B 아티팩트에서 체크포인트 파일을 내려받고, 최종 디렉터리 경로를 돌려준다."""
    # rank 0만 별도 run을 만들어서 use_artifact 호출
    rank_str = os.environ.get("RANK", "0")
    try:
        rank = int(rank_str)
    except ValueError:
        rank = 0
    """
    resume_alias : str
        - 실제로 사용할 별칭. 예: 'latest', 'best'.
    collection_name : str
        - W&B 상에서 모델 묶음 이름. 예: f"{args.past_name}_latest-model".
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    """
    entity: str 예: 'jksg01019-naver-labs'
    project: str 예: 'Diffusion-Planner'
    """
    artifact_wandb_path = f"{entity}/{project}/{collection_name}:{resume_alias}"
    print(
        f"[WANDB->local] loading from artifact_wandb_path: {artifact_wandb_path}"
    )
    artifact = api.artifact(artifact_wandb_path, type='model')

    source_run = artifact.logged_by()
    if 'save_path' not in source_run.config:
        raise ValueError("NO save path in WANDB run config")

    # "./training_log/feasible_full_time_use_vel_gpu_2_exp_A/2025-12-06-06:56:58/"
    past_save_path = source_run.config['save_path']
    print(
        f"[WANDB->local] find save_path in WANDB run config past_save_path: {past_save_path}"
    )
    if args.save_path is None:  #
        experiment_is_same = args.name == args.past_name
        assert experiment_is_same, (
            "args.save_path None -> , args.name과 args.past_name should be same."
        )
        assert args.resume_wandb_model_name is not None, (
            "args.resume_wandb_model_name must be set when resuming the same experiment."
        )
        args.save_path = past_save_path

    # target_local_ckpt_path: ./training_log/.../2025-12-06-06:56:58/latest.pth
    target_local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
    if rank == 0:
        os.makedirs(past_save_path, exist_ok=True)
        artifacts_root = os.path.join(past_save_path, "artifacts")
        artifact_dir_past = artifact.download(root=artifacts_root)
        print(
            f"[WANDB->local] {artifact_wandb_path} -> {artifact_dir_past} [download]. "
            f"FYI, past_save_path: {past_save_path}")

        # 1) checkpoint 파일을 past_save_path 루트로 복사
        """
        checkpoint_filename: 'latest.pth' 또는 'best.pth'
        src_ckpt_path_past: ./training_log/.../2025-12-06-06:56:58/artifacts/latest.pth
        target_local_ckpt_path: ./training_log/.../2025-12-06-06:56:58/latest.pth
        """

        if os.path.exists(target_local_ckpt_path):
            print(
                f"[local->local] target_local_ckpt_path '{target_local_ckpt_path}' already exist so delete and re download ."
            )
            os.remove(target_local_ckpt_path)
        src_ckpt_path_past = os.path.join(artifact_dir_past,
                                          checkpoint_filename)
        if os.path.exists(src_ckpt_path_past):
            shutil.copy2(src_ckpt_path_past, target_local_ckpt_path)
            print(
                f"[local->local] copy checkpoint: src_ckpt_path_past {src_ckpt_path_past} -> target_local_ckpt_path {target_local_ckpt_path}"
            )
        else:
            raise FileNotFoundError(
                f"[local->local] cannot find checkpoint in downloaded artifact src_ckpt_path_past: {src_ckpt_path_past}"
            )

        # 2) DeepSpeed용 latest / best 디렉터리도 있으면 같이 복사
        for tag in ("latest", "best"):
            """
            artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            artifact_tag_dir_past : ./training_log/.../2025-12-06-06:56:58/artifacts/latest/
            save_path_tag_dir : ./training_log/.../2025-12-06-06:56:58/latest/
            """
            """
            artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            안에서, 이름에 tag가 포함된 서브 폴더를 찾는다.
            예: latest_epoch-000001, best_epoch-000010 등
            """
            if not os.path.isdir(artifact_dir_past):
                continue

            # tag 문자열이 포함된 서브 디렉터리들 후보
            candidate_dirs = [
                d for d in os.listdir(artifact_dir_past) if
                os.path.isdir(os.path.join(artifact_dir_past, d)) and tag in d
            ]
            if not candidate_dirs:
                continue

            # 첫 번째(정렬 기준) 후보를 최종 tag 디렉터리 이름으로 사용
            chosen_dir_name = sorted(candidate_dirs)[0]

            artifact_tag_dir_past = os.path.join(artifact_dir_past,
                                                 chosen_dir_name)
            save_path_tag_dir = os.path.join(args.save_path, chosen_dir_name)

            if os.path.isdir(save_path_tag_dir):
                shutil.rmtree(save_path_tag_dir)
            shutil.copytree(artifact_tag_dir_past, save_path_tag_dir)
            print(
                "[local->local] COpy DeepSpeed checkpoint directory: "
                f"artifact_tag_dir_past   {artifact_tag_dir_past} -> save_path_tag_dir  {save_path_tag_dir}"
            )
        if not os.path.exists(target_local_ckpt_path):
            downloaded_files = []
            # artifact_dir_past: ./training_log/.../2025-12-06-06:56:58/artifacts/
            if os.path.isdir(artifact_dir_past):
                for root, dirs, files in os.walk(artifact_dir_past):
                    for name in files:
                        rel = os.path.relpath(os.path.join(root, name),
                                              artifact_dir_past)
                        downloaded_files.append(rel)
            raise FileNotFoundError(
                f"[local->local] cannot find checkpoint_filename   '{checkpoint_filename}' from downloaded artifact: "
                f"artifact_dir_past={artifact_dir_past}. example files: {downloaded_files[:10]}"
            )
    else:
        # rank 0이 다운로드해서 파일이 생길 때까지 대기
        print(
            f"rank {rank} is waiting for creating checkpoint: target_local_ckpt_path {target_local_ckpt_path}"
        )
        waited_seconds = 0
        max_wait_seconds = 600  # 10분 정도 대기 (필요하면 조정)

        while not os.path.exists(target_local_ckpt_path):
            if waited_seconds >= max_wait_seconds:
                raise TimeoutError(
                    f"rank 0이 {max_wait_seconds}second fail: target_local_ckpt_path {target_local_ckpt_path}"
                )
            time.sleep(1)
            waited_seconds += 1

        print(
            f"rank {rank} already checked downloaded checkpoint! : target_local_ckpt_path {target_local_ckpt_path}"
        )



def _determine_wandb_artifact_config(
    args: argparse.Namespace,) -> Tuple[str, str, str]:
    """재개하려는 W&B 아티팩트의 이름과 파일 이름을 결정한다.

    Args:
        args: argparse.Namespace
            - args.resume_wandb_model_name (str): 'latest' 또는 'best' 등을 기대.
            - args.past_name (str): 실험 이름. 컬렉션 이름의 접두사로 사용된다.

    Returns:
        resume_alias: str
            - 실제로 사용할 별칭. 예: 'latest', 'best'.
        collection_name: str
            - W&B 상에서 모델 묶음 이름. 예: f"{args.past_name}_latest-model".
        checkpoint_filename: str
            - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    resume_alias = args.resume_wandb_model_name
    if resume_alias == 'best':
        collection_name = f"{args.past_name}_best-model"
        checkpoint_filename = "best.pth"
    else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
        collection_name = f"{args.past_name}_latest-model"
        checkpoint_filename = "latest.pth"
    return resume_alias, collection_name, checkpoint_filename


def _validate_wandb_resume_args(args: argparse.Namespace) -> None:
    """W&B에서 체크포인트를 받아올 때 필요한 기본 인자들을 점검한다.

    처리 내용:
      - args.resume_wandb_model_name 가 'latest' 인지 확인한다.
      - args.past_name 이 비어 있지 않은지 확인한다.
    둘 중 하나라도 조건을 만족하지 않으면 바로 ValueError 를 던져서
    잘못된 설정으로 내려받기를 시도하지 않도록 막는다.
    """
    # resume_wandb_model_name 가 "latest"가 아니면 에러를 발생시킵니다.
    if args.resume_wandb_model_name not in ['latest']:
        raise ValueError("args.resume_wandb_model_name must be 'latest.")
    if not args.past_name:
        raise ValueError(
            "args.past_name must be provided to resume from a wandb artifact.")



def prepare_wandb_resume(args: argparse.Namespace,) -> None:
    """명령행 인자를 읽고, 필요하다면 W&B 아티팩트에서 체크포인트를 내려받아 args를 보정한다.

    처리 흐름:
      2) args.resume_wandb_model_name 가 설정되지 않았다면 그대로 반환한다.
      3) 설정된 경우
         - _validate_wandb_resume_args() 로 기본 인자 검사를 하고
         - wandb.Api() 를 만든 뒤
         - _determine_wandb_artifact_config() 으로 컬렉션/파일 이름을 정하고
         - _get_wandb_entity_and_project_for_resume() 로 entity, project 를 얻고
         - _download_wandb_checkpoint_to_local() 로 체크포인트를 내려받는다.
      4) 내려받은 디렉터리 경로를 args.save_path 에 저장해
         이후 model_training() 의 resume 로직이 그대로 동작하도록 만든다.

    Returns:
        args: argparse.Namespace
            - 체크포인트를 쓰지 않을 때는 원본 get_args() 결과와 동일.
            - W&B에서 내려받기를 한 경우에는
              args.save_path 가 실제 디렉터리 경로로 채워진 상태.
    """

    if not args.resume_wandb_model_name:
        return

    _validate_wandb_resume_args(args)
    print(
        f"Resuming from wandb artifact: {args.past_name}:{args.resume_wandb_model_name}"
    )
    """
    resume_alias : str
        - 실제로 사용할 별칭. 예: 'latest', 'best'.
    collection_name : str
        - W&B 상에서 모델 묶음 이름. 예: f"{args.name}_latest-model".
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    api = wandb.Api()
    resume_alias, collection_name, checkpoint_filename = \
        _determine_wandb_artifact_config(args)
    if getattr(args, "save_path", None) is not None:
        local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
        if os.path.exists(local_ckpt_path):
            print(
                f"[WANDB->local] found existing local checkpoint: {local_ckpt_path} "
                f"(alias={resume_alias}), skip wandb download.")
            return
    """
    entity: str 예: 'jksg01019-naver-labs'
    project: str 예: 'Diffusion-Planner'
    """
    try:
        entity = args.entity
        project = args.project
        _download_wandb_checkpoint_to_local(
            args,
            api=api,
            entity=entity,
            project=project,
            collection_name=collection_name,
            resume_alias=resume_alias,
            checkpoint_filename=checkpoint_filename,
        )

    except Exception as e:
        raise RuntimeError(f"W&B 아티팩트에서 체크포인트를 내려받는 중 오류가 발생했습니다: {e}") from e


def get_save_path(args: argparse.Namespace, global_rank: int) -> None:
    """실험 이름과 past_name을 기준으로 save_path를 만들고, 모든 rank에 공유한다.

    동작 요약:
      * (stage1 처럼) 새로운 실험(name != past_name)인 경우
        - rank 0에서만 새로운 디렉터리를 만들고
          예: {args.save_dir}/training_log/{args.name}/{YYYY-MM-DD-HH:MM:SS}/
        - 그 경로를 torch.distributed.broadcast_object_list 로 모든 rank에 전파한다.
      * (같은 실험을 이어서 학습하는) name == past_name 인 경우
        - 여기서는 save_path를 정하지 않고(None 유지)
        - 이후 W&B 아티팩트에서 가져온 past_save_path를 args.save_path에 채운다.
      * 분산이 초기화되어 있지 않은(single GPU / single process) 경우
        - rank 0 로직만 실행되고, broadcast는 생략된다.

    Args:
        args (argparse.Namespace):
            학습 및 실험 설정이 들어 있는 argparse 인자 모음.
            - name (str): 현재 stage에서 사용할 실험 이름.
            - past_name (str): 이전 stage 실험 이름. 다른 stage의 weight를 재사용할 때 사용.
            - save_dir (str): training_log 루트 디렉터리 상위 경로.
            - save_path (Optional[str]): 이 함수 안에서 최종 로그/체크포인트 경로로 설정된다.
        global_rank (int):
            DDP에서 현재 프로세스의 global rank.
            - 0: 메인 프로세스. 디렉터리 생성 및 초기 save_path 결정 담당.
            - >0: rank 0이 만든 save_path를 broadcast로 전달받기만 한다.

    Returns:
        None:
            args.save_path를 in-place로 수정하고, 모든 rank에서 동일한 값을 갖도록 만든다.
    """
    past_name: str = getattr(args, "past_name", "")
    experiment_is_same: bool = bool(past_name) and (args.name == past_name)

    save_path: Optional[str] = None
    if global_rank == 0:
        if experiment_is_same:
            # 동일 실험 계속 학습: save_path는 나중에 W&B에서 past_save_path로 채움
            save_path = None
        else:
            time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{args.save_dir}/training_log/{args.name}/{time_str}/"
            os.makedirs(save_path, exist_ok=True)

    # DDP / 멀티 GPU에서 rank0가 만든 save_path를 모든 rank에 공유
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        obj_list = [save_path]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        save_path = obj_list[0]

    args.save_path = save_path


def init_distributed(args: argparse.Namespace,) -> Tuple[int, int, int, bool]:
    """여러 GPU를 쓸 때 필요한 분산 학습 환경을 준비하고, 내 위치 정보를 정리한다.

    DeepSpeed를 쓰는 경우에도 torch.distributed를 미리 초기화해 두어서
    get_save_path() 안의 broadcast가 항상 동작하도록 만든다.
    """
    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    if use_deepspeed:
        # torchrun이 설정한 환경변수만 사용해서 rank/world_size를 계산한다.
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            global_rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # 단일 프로세스 fallback
            global_rank = 0
            world_size = 1
            rank = 0

        if args.device.startswith("cuda"):
            torch.cuda.set_device(rank)

        # ✅ DeepSpeed에서도 torch.distributed를 미리 초기화
        #    → _get_save_path의 broadcast_object_list가 항상 동작
        if (world_size > 1 and torch.distributed.is_available() and
                not torch.distributed.is_initialized()):
            backend = "nccl" if args.device.startswith("cuda") else "gloo"
            torch.distributed.init_process_group(
                backend=backend,
                init_method="env://",
                world_size=world_size,
                rank=global_rank,
            )
    else:
        # 기존 DDP 초기화 흐름과 동일
        global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
        world_size = ddp.get_world_size()

    return global_rank, rank, world_size, use_deepspeed


def build_deepspeed_inference_config(args: argparse.Namespace) -> Dict[str, Any]:
    """추론용 DeepSpeed 설정 dict를 만든다(학습용 옵션은 제외)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = max(1, world_size)

    return {
        "dtype": torch.bfloat16,  # 필요하면 torch.float16 로 바꿔도 됨
        "tensor_parallel": {"tp_size": world_size},  # 여러 GPU에 모델을 나눠 올릴 때
        "replace_with_kernel_inject": False,  # 일반 모델이면 보통 꺼두는 게 안전
    }


def build_deepspeed_config(args: argparse.Namespace,
                           current_global_batch: int) -> Dict[str, Any]:
    """args 값을 조합해서 DeepSpeed ZeRO-2 설정 dict를 만든다.

    처리 흐름:
      1) WORLD_SIZE(프로세스 수)를 읽어서, GPU 한 장당 미니배치 크기를 정한다.
      2) args.grad_accum_steps, args.max_grad_norm 같은 값을 읽어서
         한 번에 얼마나 gradient를 모을지, 얼마나 잘라낼지 결정한다.
      3) ZeRO-2 단계(stage=2)를 켜고,
         · gradient와 옵티마 상태를 GPU끼리 나눠 들도록 설정한다.
      4) 필요하면 zero_offload_optimizer 를 써서
         옵티마 상태 일부를 CPU 메모리로 옮겨 GPU 메모리를 더 아낀다.

    이렇게 만든 ds_config dict는 train_predictor.py에서
    deepspeed.initialize(..., config=ds_config) 에 그대로 넘겨서 사용한다.
    """
    # --- 1) world_size 계산 (torchrun이 넘겨준 값 우선 사용) ---
    world_size_str = os.environ.get("WORLD_SIZE")
    if world_size_str is not None:
        try:
            world_size = max(1, int(world_size_str))
        except ValueError:
            world_size = 1
    else:
        world_size = int(getattr(args, "world_size", 1))
        if world_size <= 0:
            world_size = 1
    micro_batch_size = max(1, current_global_batch // world_size)

    # --- 3) args에서 세부 설정값 읽기 ---
    """ grad_accum_steps
    실질 배치 크기 = micro_batch_size * world_size * grad_accum_steps 

    grad_accum_steps 커지면 좋은 점 (배치 사이즈를 키우는 것과 같은 효과)

    batch size를 키우는 것에 비해, grad_accum_steps를 키우는 것의 단점
      - 학습 속도가 느리다.
    """
    grad_accum_steps = int(getattr(args, "grad_accum_steps", 1))
    max_grad_norm = float(getattr(args, "max_grad_norm", 1.0))

    zero_offload_optimizer = bool(getattr(args, "zero_offload_optimizer",
                                          False))
    """
    allgather_bucket_size : 
        weight 조각들을 다른 GPU에서 가져올 때
        한 번에 얼마나 큰 묶음으로 모아서 보내고 받을지 크기.
    reduce_bucket_size
        기울기 값을 여러 GPU에서 합칠 때 한 번에 묶는 크기.

    둘 다 크게 설정할수록
        통신 횟수 ↓
        한 번에 처리하는 메모리 ↑ (피크 메모리 조금 증가 가능)
    """
    allgather_bucket_size = float(getattr(args, "ds_allgather_bucket_size",
                                          2e8))
    reduce_bucket_size = float(getattr(args, "ds_reduce_bucket_size", 2e8))

    steps_per_print = int(getattr(args, "ds_steps_per_print", 100))

    # --- 4) ZeRO-2 중심 DeepSpeed 설정 dict ---
    """        
        stage: 2 -> ZeRO 단계 2.
            기울기와 옵티마 상태를 GPU 사이에 나눠 들고,
            weight 자체는 여전히 GPU마다 풀 복사본을 유지하는 단계.

        allgather_partitions: True
            “기울기/옵티마 나누고 모으는 방식”을 미세하게 조절하는 옵션
            True
            False
                “나눠진 기울기/옵티마를 꽤 자주 전체 형태로 모아서 쓰는 쪽”

        allgather_bucket_size: 
            weight 조각들을 다른 GPU에서 가져올 때
            한 번에 얼마나 큰 묶음으로 모아서 보내고 받을지 크기.

        reduce_scatter: True

            기울기를 각 GPU에서 계산한 뒤
            합치고 다시 나누는 과정을 한 번에 처리하는 모드.
            기울기 통신 비용을 줄이기 위한 방식이라고 보면 된다.

        reduce_bucket_size: reduce_bucket_size
                    기울기 값을 여러 GPU에서 합칠 때 한 번에 묶는 크기.

        overlap_comm: True

            통신(다른 GPU와 데이터 주고받기)과
            연산(다음 레이어 계산)을 가능한 한 동시에 진행하도록 시도한다.

        속도 향상을 노리는 옵션.

        contiguous_gradients: True

            기울기 텐서들을 메모리 상에서 연속된 큰 덩어리로 재배치한다.

            이렇게 하면 통신이나 클리핑 같은 작업에서
            여러 텐서를 따로따로 다루지 않아도 되어서 효율이 좋아진다.
    """
    ds_config: Dict[str, Any] = {
        "train_micro_batch_size_per_gpu": micro_batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_clipping": max_grad_norm,
        "torch_autocast": {
            "enabled": True,
            # 둘 중 하나 선택:
            "dtype": "bfloat16",  # 또는 "float16"
            # (선택) 이 목록을 안 주면 DeepSpeed 기본 목록을 씀 :contentReference[oaicite:1]{index=1}
            # "lower_precision_safe_modules": ["torch.nn.Linear", "torch.nn.Conv2d"],
        },
        "fp16": {
            "enabled": False,
        },
        "bf16": {
            "enabled": False,
        },
        "zero_optimization": {
            "stage": 2,  # ZeRO-2
            "allgather_partitions": True,
            "allgather_bucket_size": allgather_bucket_size,
            "reduce_scatter": True,
            "reduce_bucket_size": reduce_bucket_size,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "zero_allow_untested_optimizer": True,
        "steps_per_print": steps_per_print,
    }

    # 옵티마 상태를 CPU로 일부 넘겨서 GPU 메모리를 더 줄이고 싶을 때
    if zero_offload_optimizer:
        ds_config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
        }

    return ds_config


def effective_global_batch(batch_size: int, world_size: int) -> int:
    """DataLoader가 사용하는 실제 글로벌 배치(DDP 정합)"""
    world_size = max(1, int(world_size))
    per_rank = max(1, batch_size // world_size)
    return per_rank * world_size  # drop_last 정합 반영

def build_dataset_and_sampler(
    args: argparse.Namespace,
    set_: str,
    set_list: str,
    role: str,
    world_size: int,
    global_rank: int,
) -> Tuple[DiffusionPlannerData, DistributedSampler]:
    """학습용 Dataset과 DistributedSampler를 만든다.

    DiffusionPlannerData 는
      - 학습에 쓸 파일 목록(JSON)을 읽고
      - 각 항목에 대응하는 npz 파일을 열어서
      - 하나의 샘플을 이름 기반 dict로 돌려준다.
        예: "ego_agent_past": (time_len, 11),
            "neighbor_agents_past": (agent_num, time_len, 11),
            "lanes": (lane_num, lane_len, 12) 등.

    DistributedSampler 는 전체 샘플 인덱스를
      - world_size 개만큼 나누고
      - 각 rank(global_rank)에 서로 다른 구간을 배정해서
        여러 GPU가 겹치지 않는 데이터를 보게 만든다.
    이렇게 하면 모든 GPU가 동시에 다른 샘플을 보면서도,
    한 epoch 안에서 전체 데이터가 고르게 사용되도록 맞출 수 있다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        world_size: 전체 프로세스 개수.
        global_rank: 전체 프로세스 기준 번호.

    Returns:
        data_set: DiffusionPlannerData 인스턴스.
        data_sampler: 각 rank에 샘플을 나눠주는 DistributedSampler.
    """
    # data_set 내부에서 개별 샘플은 npz를 읽어 (agent, lane 등) 배열을
    #   모델 입력 shape에 맞는 Tensor로 바꿔준다. (예: (time_len, 11), (A, T, 11) 등)
    data_set = DiffusionPlannerData(
        set_,  # 예: "/mnt/nuplan/dataset/processed"
        set_list,  # 예: diffusion_planner_training.json
        args.predicted_neighbor_num,
        role)

    data_sampler = DistributedSampler(
        data_set,
        num_replicas=world_size,
        rank=global_rank,
        shuffle=True,
        seed=args.seed,
    )

    return data_set, data_sampler


def build_data_loader(
    args: argparse.Namespace,
    data_set: DiffusionPlannerData,
    data_sampler: DistributedSampler,
    batch_size: int,
    world_size: int,
) -> DataLoader:
    """분산 학습에 맞는 DataLoader를 만든다.

    Args:
        args: 학습 설정이 들어 있는 argparse.Namespace.
        data_set: 학습용 DiffusionPlannerData.
        data_sampler: rank별로 인덱스를 나누는 DistributedSampler.
        batch_size: 전체 글로벌 배치 크기.
        world_size: 전체 프로세스 개수.

    Returns:
        loader: 각 step마다 배치 dict를 반환하는 DataLoader.
                      각 배치의 텐서는 (B, ·) shape를 가진다.
    """
    # 각 rank가 보는 per-rank 배치 크기
    batch_size_per_rank = batch_size // world_size

    loader = DataLoader(
        data_set,  # DiffusionPlannerData
        sampler=data_sampler,  # DistributedSampler
        batch_size=batch_size_per_rank,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_mem,
        persistent_workers=True,  # 에폭이 바뀌어도 워커 유지
        drop_last=True,
        collate_fn=DiffusionPlannerCollate(args),  # 배치 텐서 shape 맞춤
    )
    return loader


def create_diffusion_planner_and_ema(
    args: argparse.Namespace,
    rank: int,
    use_deepspeed: bool,
) -> Tuple[nn.Module, Optional[ModelEma], nn.Module]:
    """Diffusion_Planner, EMA, base_model(DDP unwrap)을 한 번에 만든다.

    처리 순서:
      1) Diffusion_Planner(args) 로 원본 모델을 만들고,
         rank 또는 args.device 로 to() 호출한다.
      2) DDP 모드이면 torch.nn.parallel.DistributedDataParallel 로 래핑한다.
      3) use_ema=True 이면 timm.utils.ModelEma 로 EMA 복사본을 만든다.
      4) ddp.get_model(...) 로 DDP/DeepSpeed 래퍼를 벗긴 실제 모듈(base_model)을 얻는다.

    Args:
        args (argparse.Namespace):
            - device(str): "cuda" 또는 "cpu" 등.
            - ddp(bool): DDP 사용 여부.
            - use_ema(bool): EMA 사용 여부.
        rank (int):
            - 이 프로세스에서 사용할 GPU 인덱스.  # shape: ()
        use_deepspeed (bool):
            - DeepSpeed 모드 사용 여부. True 이면 DDP 래퍼를 만들지 않는다.

    Returns:
        Tuple[nn.Module, Optional[ModelEma], nn.Module]:
            - diffusion_planner:
                · 원본 모델 또는 DDP/DeepSpeed 래핑된 모델.
            - model_ema:
                · EMA 래퍼 또는 None.
            - base_model:
                · ddp.get_model 으로 래퍼를 벗긴 실제 nn.Module.
                  base_model.parameters() 텐서들의 shape 는
                  레이어 종류에 따라 (out_dim, in_dim), (dim,), (C_out, C_in, kH, kW) 등이다.
    """
    diffusion_planner = Diffusion_Planner(args)
    device = rank if args.device == 'cuda' else args.device
    diffusion_planner = diffusion_planner.to(device)

    # DeepSpeed를 쓸 때는 torch.nn.parallel.DDP 래퍼는 사용하지 않는다.
    if args.ddp and (not use_deepspeed):
        diffusion_planner = DDP(diffusion_planner, device_ids=[rank])

    model_ema: Optional[ModelEma] = None
    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )

    # DDP를 쓴 경우에만 래퍼를 벗겨서 실제 모듈 기준으로 param_group을 만든다.
    base_model = ddp.get_model(
        diffusion_planner,
        args.ddp and (not use_deepspeed),
    )
    return diffusion_planner, model_ema, base_model


class DiffusionPlannerCollate:
    """DiffusionPlannerData 샘플들을 배치 텐서로 묶는 collate_fn.

    각 샘플은 agent / lane / route / static 개수에 상한이 걸려 있고,
    이 collate_fn 은
      1) (선택) 중심 기준 거리 크로핑
      2) 배치 내 최대 길이 계산
      3) 그 길이에 맞춰 0 패딩
    순서로 고정 shape 배치를 만든다.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """collate 설정을 초기화한다.

        Args:
            args (argparse.Namespace):
                - caching_max_agent_num (int)
                - caching_max_lane_num (int)
                - caching_max_static_num (int)
                - center_crop_radius_m (float, 선택)
                - center_crop_mode (str, 'npc' / 'ego' / 'none')
        """
        # 중심 기준 크로핑 옵션
        # - center_crop_radius_m <= 0: 크로핑 사용 안 함
        # - center_crop_mode: "npc" → 임의 NPC 1대를 기준, "ego" → ego 기준
        self.args = args
        self.center_crop_radius_m: float = float(
            getattr(args, "center_crop_radius_m", 0.0))
        self.center_crop_mode: str = str(
            getattr(args, "center_crop_mode", "none")).lower()

    def _is_string_container_value(self, value: Any) -> bool:
        """값이 '문자열(또는 문자열 묶음)'인지 판별합니다.

        목적
        ----
        배치를 만들 때 숫자/배열 데이터는 torch.Tensor로 묶어야 하지만,
        문자열(str)은 torch.Tensor로 바꿀 수 없어 에러가 납니다.
        그래서 문자열은 "그대로 리스트로 모아서" 반환하기 위해 미리 구분합니다.

        문자열로 취급하는 케이스
        ----------------------
        1) value가 str(또는 numpy의 문자열 타입)인 경우
           - 예: "scenario_000123"  # shape: ()
        2) value가 list/tuple이고, 안의 원소가 전부 str인 경우
           - 예: ["a", "b"]  # 원소 개수는 샘플마다 다를 수 있음
        3) value가 numpy 배열이고 dtype이 문자열인 경우
           - 예: np.array(["a","b"])  # shape: (N,)
           - 예: np.array("abc")      # shape: ()

        Args:
            value: 샘플 dict 안의 어떤 값.

        Returns:
            bool:
                - True: 문자열(또는 문자열 묶음)
                - False: 숫자/배열 등(텐서로 묶을 수 있는 쪽)
        """
        if value is None:
            return False

        # 1) 단일 문자열
        if isinstance(value, (str, np.str_)):
            return True

        # 2) list/tuple 안이 전부 문자열인 경우
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return False
            return all(isinstance(x, (str, np.str_)) for x in value)

        # 3) numpy 배열이 문자열 dtype인 경우
        if isinstance(value, np.ndarray):
            # dtype.kind:
            #   'U' unicode string, 'S' byte string, 'O' object
            if value.dtype.kind in ("U", "S"):
                return True
            if value.dtype.kind == "O" and value.size > 0:
                # object 배열이라도 실제가 문자열이면 문자열로 취급
                first = value.flat[0]
                return isinstance(first, (str, np.str_))

        return False

    def _should_return_string_list_for_key(self, key: str,
                                           values: List[Any]) -> bool:
        """특정 key가 '문자열 배치(List[str])'로 반환되어야 하는지 결정합니다.

        규칙
        ----
        - values(길이 B) 중 None이 아닌 값들이 전부 문자열 계열이면:
            -> 이 key는 텐서로 묶지 않고 List[str]로 반환합니다.
        - 문자열 값과 숫자/배열 값이 섞여 있으면:
            -> 데이터 자체가 섞인 상태라 이후 로직이 예측 불가능해지므로
               명확히 에러를 냅니다.

        Args:
            key (str): 배치 dict의 key 이름. shape: ()
            values (List[Any]): 길이 B의 샘플 값 목록.
                - 문자열이면 보통: str (shape: ())
                - 또는 문자열 묶음(list[str], np.ndarray[str])도 가능

        Returns:
            bool:
                - True: 이 key는 List[str]로 반환
                - False: 이 key는 기존대로 torch.Tensor/None으로 반환

        Raises:
            ValueError: 문자열 값과 숫자/배열 값이 같은 key에서 섞여 있을 때
        """
        non_none = [v for v in values if v is not None]
        if len(non_none) == 0:
            return False

        is_str_flags = [self._is_string_container_value(v) for v in non_none]
        if all(is_str_flags):
            return True

        if any(is_str_flags) and (not all(is_str_flags)):
            raise ValueError(
                f"[Collate] key='{key}' 에 문자열 값과 숫자/배열 값이 섞여 있습니다. "
                f"한 key는 한 종류로만 들어오게 정리해야 합니다.")

        return False

    def _collate_string_values_to_list(self, values: List[Any]) -> List[str]:
        """문자열(또는 문자열 묶음)을 배치 단위 List[str]로 모아 반환합니다.

        반환 형태
        --------
        - 항상 길이 B의 리스트를 반환합니다.
        - 각 원소는 str 입니다.
        - None은 빈 문자열 "" 로 바꿉니다.

        예시
        ----
        values = ["a", None, "c"]  -> ["a", "", "c"]

        Args:
            values (List[Any]):
                길이 B 리스트.
                각 원소는 보통 str(shape: ()) 또는 None 입니다.

        Returns:
            List[str]:
                길이 B의 문자열 리스트.
                - length: B
        """
        out: List[str] = []
        for v in values:
            if v is None:
                out.append("")
            elif isinstance(v, (str, np.str_)):
                out.append(str(v))
            else:
                # list[str], np.ndarray[str] 같은 경우도 "문자열 1개"로 만들어 담습니다.
                # (요구사항이 List[str] 이므로, 샘플 1개당 문자열 1개로 표현)
                out.append(str(v))
        return out

    def _infer_max_sizes_for_agent_route_lane_order(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """agent_route_lane_order를 패딩하기 위한 (최대 agent 수, 최대 lane 수)를 계산합니다.

        왜 이 함수가 필요한가
        --------------------
        agent_route_lane_order는 (행=agent, 열=lane) 관계를 담고 있습니다.
        현재 로직에서는 agent_route_lane_order의 "행"이
        neighbor 전체가 아니라 near agent 기준으로 잘려서 들어올 수 있습니다.

        그래서 이 key를 배치로 패딩할 때,
        agent 축 최대값(max_agent_num)은 neighbor_agents_past 같은 다른 텐서가 아니라
        **agent_route_lane_order 자체의 행 개수**만 보고 정하는 것이 안전합니다.
        (그래야 agent_route_lane_order_is_valid 같은 agent 축 마스크와도 길이가 맞습니다.)

        lane 축(max_lane_num)은 lanes.shape[0]와 agent_route_lane_order.shape[1] 중
        더 큰 값을 사용해, lane 텐서 패딩과도 잘 맞도록 합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            Tuple[int, int]:
                - max_agent_num: shape (), int
                    agent_route_lane_order의 행 수(=agent 수) 최대값.
                - max_lane_num: shape (), int
                    lanes의 lane 개수 또는 agent_route_lane_order의 열 수 최대값.
        """
        max_agent_num: int = 0
        max_lane_num: int = 0

        for sample in batch:
            aro = sample.get("agent_route_lane_order", None)
            if aro is not None:
                aro_arr = np.asarray(aro)  # (agent_num, lane_num)
                # aro_arr: (A_i, L_i)
                if aro_arr.ndim == 2:
                    max_agent_num = max(max_agent_num, int(aro_arr.shape[0]))
                    max_lane_num = max(max_lane_num, int(aro_arr.shape[1]))

            lanes = sample.get("lanes", None)
            if lanes is not None:
                lanes_arr = np.asarray(lanes)  #
                # lanes_arr: (L_i, lane_len, feat) 또는 최소 (L_i, ...)
                if lanes_arr.ndim >= 1:
                    max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))

        return int(max_agent_num), int(max_lane_num)

    def _get_special_padding_category_spec(
        self,
        key: str,
    ) -> Optional[Tuple[int, int]]:
        """특정 key에 대해, '패딩을 어떤 카테고리로 채울지' 규칙을 돌려줍니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / left_line_type / right_line_type 같은 값은
        보통 (카테고리 개수)만큼의 길이를 가진 벡터(대부분 one-hot)로 들어옵니다.

        그런데 패딩을 전부 0으로 채우면,
        나중에 argmax 같은 방식으로 카테고리를 뽑을 때
        "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.

        그래서 아래 key들만은 패딩을 0이 아니라
        '모름/미정'에 해당하는 카테고리로 채우도록 규칙을 따로 둡니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()

        Returns:
            Optional[Tuple[int, int]]:
                - (unknown_category_index, num_categories)
                - lane_type: (3, 4)  -> (L, 4)에서 마지막(3번)을 1로 채움
                - left/right_line_type: (11, 13) -> (L, 13)에서 11번을 1로 채움
                - 그 외 key는 None
        """
        key_str = str(key)
        if key_str == "lane_type":
            return 3, 4
        if key_str in ("left_line_type", "right_line_type"):
            return 11, 13
        return None

    def _infer_max_lane_num_from_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> int:
        """배치에서 'lane 개수의 최대값'을 안전하게 구합니다.

        이 함수가 필요한 이유
        --------------------
        lane_type / line_type 같은 값이 어떤 샘플에서는 None일 수 있습니다.
        그런데 그 샘플의 lanes 개수는 클 수 있어서,
        단순히 non-None 값들만 보고 max 길이를 정하면
        lanes 텐서와 lane_type 텐서의 lane 차원이 달라질 수 있습니다.

        그래서 lane 관련 key들은 lane 차원을 항상 lanes 기준으로 맞추기 위해
        batch 전체의 lanes.shape[0] 최대값을 구합니다.

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            int:
                - max_lane_num: 배치 내 lanes의 첫 번째 축 최대값. shape: ()
        """
        max_lane_num: int = 0
        for sample in batch:
            lanes_value = sample.get("lanes", None)
            if lanes_value is None:
                continue
            lanes_arr = np.asarray(lanes_value)
            if lanes_arr.ndim >= 1:
                max_lane_num = max(max_lane_num, int(lanes_arr.shape[0]))
        return int(max_lane_num)

    def _build_default_padded_tensor_for_key(
        self,
        key: str,
        out_shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """key 성격에 맞는 '기본 패딩 텐서'를 만듭니다.

        기본 규칙
        --------
        - 대부분의 key: 0으로 채운 텐서를 만듭니다.
        - lane_type: (…, 4) 형태라면 마지막 축에서 3번 인덱스만 1로 채웁니다.
            즉, [0,0,0,1] 형태의 "미정" one-hot 패딩입니다.
        - left_line_type / right_line_type: (…, 13) 형태라면 마지막 축에서 11번만 1로 채웁니다.
            즉, "선은 있으나 타입 모름" one-hot 패딩입니다.

        주의
        ----
        - 만약 어떤 이유로 one-hot 형태가 아니라 (…, ) 라벨 형태로 들어온다면,
          그 경우는 텐서 전체를 unknown 카테고리 숫자(예: 3, 11)로 채웁니다.

        Args:
            key (str):
                샘플 dict의 key 이름. shape: ()
            out_shape (Tuple[int, ...]):
                출력 텐서 shape.
                예:
                  - lane_type: (B, L_max, 4)
                  - left_line_type: (B, L_max, 13)
            dtype (torch.dtype):
                출력 텐서 dtype.

        Returns:
            torch.Tensor:
                out: out_shape 그대로의 텐서.
        """
        spec = self._get_special_padding_category_spec(key)
        if spec is None:
            return torch.zeros(out_shape, dtype=dtype)

        unknown_index, num_categories = int(spec[0]), int(spec[1])

        # out_shape가 충분히 길고, 마지막 축이 카테고리 개수와 맞으면 one-hot 패딩
        if len(out_shape) >= 2 and int(out_shape[-1]) == int(num_categories):
            out = torch.zeros(out_shape, dtype=dtype)
            if 0 <= unknown_index < num_categories:
                out[..., unknown_index] = torch.as_tensor(1, dtype=dtype)
            return out

        # one-hot이 아니라 라벨 형태(예: (B, L_max))로 들어오는 경우를 대비한 fallback
        if len(out_shape) >= 2:
            return torch.full(out_shape, fill_value=unknown_index, dtype=dtype)

        # 기대 형태가 너무 이상하면 안전하게 0으로
        return torch.zeros(out_shape, dtype=dtype)

    # ------------------------------------------------------------------
    # 0) 크로핑 여부 판단
    # ------------------------------------------------------------------
    def _should_center_crop(self) -> bool:
        """중심 기준 크로핑을 쓸지 여부를 반환한다.

        Returns:
            bool: True면 크로핑을 적용한다.
        """
        return (self.center_crop_radius_m > 0.0 and
                self.center_crop_mode in ("npc", "ego"))

    # ------------------------------------------------------------------
    # 0-1) 샘플별 개수 통계 계산
    # ------------------------------------------------------------------
    def _compute_object_lengths(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """각 샘플의 agent / lane / route_lane / static 개수를 모은다.

        Args:
            batch: 길이 B 리스트, 각 원소는 단일 샘플 dict.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - agent_len_arr: (B,) 각 샘플의 agent 수
                - lane_len_arr: (B,) 각 샘플의 lane 수
                - route_lane_len_arr: (B,) 각 샘플의 route lane 수
                - static_len_arr: (B,) 각 샘플의 static 수
        """
        agent_len_arr = np.array(
            [sample["neighbor_agents_past"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        lane_len_arr = np.array(
            [sample["lanes"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        route_lane_len_arr = np.array(
            [sample["route_lanes"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        static_len_arr = np.array(
            [sample["static_objects"].shape[0] for sample in batch],
            dtype=np.int64,
        )
        return agent_len_arr, lane_len_arr, route_lane_len_arr, static_len_arr

    # ------------------------------------------------------------------
    # 0-2) 최대 개수 및 lane_len 계산
    # ------------------------------------------------------------------
    def _compute_max_counts_and_lane_len(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
    ) -> Tuple[int, int, int, int, int, int]:
        """배치 안에서 agent / lane / route_lane / static 최대 개수와 길이를 계산한다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 개수.
            lane_len_arr: (B,) lane 개수.
            route_lane_len_arr: (B,) route lane 개수.
            static_len_arr: (B,) static 개수.

        Returns:
            Tuple[int, int, int, int, int, int]:
                - max_agent_num: 배치 내 최대 agent 수
                - max_lane_num: 배치 내 최대 lane 수
                - max_route_lane_num: 배치 내 최대 route lane 수
                - caching_max_static_num: 배치 내 최대 static 수
                - lane_len: lane 한 개당 점 개수
                - route_lane_len: route lane 한 개당 점 개수
        """
        batch_size: int = len(batch)
        max_agent_num: int = int(agent_len_arr.max()) if batch_size > 0 else 0
        max_lane_num: int = int(lane_len_arr.max()) if batch_size > 0 else 0
        max_route_lane_num: int = (int(route_lane_len_arr.max())
                                   if batch_size > 0 else 0)
        max_static_num: int = (int(static_len_arr.max())
                               if batch_size > 0 else 0)

        lane_len: int = (batch[0]["lanes"].shape[1] if max_lane_num > 0 else 0)
        route_lane_len: int = (batch[0]["route_lanes"].shape[1]
                               if max_route_lane_num > 0 else 0)

        return (
            max_agent_num,
            max_lane_num,
            max_route_lane_num,
            max_static_num,
            lane_len,
            route_lane_len,
        )

    # ------------------------------------------------------------------
    # 0-3) ego / 중심 좌표 계산
    # ------------------------------------------------------------------
    def _build_ego_and_center_xy(
        self,
        batch: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ego 현재 위치와 초기 중심 좌표를 만든다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - ego_xy: (B, 2) ego 현재 위치
                - center_xy: (B, 2) 초기 중심(ego와 동일)
        """
        ego_xy = np.stack(
            [sample["ego_agent_past"][-1, 0:2] for sample in batch],
            axis=0,
        ).astype(np.float32)  # (B, 2)
        center_xy = ego_xy.copy()  # (B, 2)
        return ego_xy, center_xy

    def _compute_keep_route_lane_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        route_lanes_xy: np.
        ndarray,  # (B, max_route_lane_num, route_lane_len, 2)
        route_lane_len_arr: np.ndarray,  # (B,)
        max_route_lane_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 route lane 을 남길지 여부를 계산한다.

        lane 과 동일한 규칙이지만, route_lanes 만 따로 본다.

        Args:
            center_xy: (B, 2) 중심 좌표.
            route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2) route lane 좌표.
            route_lane_len_arr: (B,) 각 샘플의 route lane 개수.
            max_route_lane_num: 배치 내 최대 route lane 수.

        Returns:
            np.ndarray:
                keep_route_lane_mask: (B, max_route_lane_num)
        """
        batch_size: int = route_lane_len_arr.shape[0]
        if max_route_lane_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        # route_lane_valid_mask: (B, max_route_lane_num)
        route_lane_valid_mask = (np.arange(max_route_lane_num)[None, :]
                                 < route_lane_len_arr[:, None])

        # diff_route_lane: (B, max_route_lane_num, route_lane_len, 2)
        diff_route_lane = route_lanes_xy - center_xy[:, None, None, :]
        # dist2_route_lane: (B, max_route_lane_num, route_lane_len)
        dist2_route_lane = (diff_route_lane**2).sum(axis=-1)
        # min_dist2_route_lane: (B, max_route_lane_num)
        min_dist2_route_lane = dist2_route_lane.min(axis=-1)

        keep_route_lane_mask = route_lane_valid_mask & (min_dist2_route_lane
                                                        <= radius_sq)
        return keep_route_lane_mask

    # ------------------------------------------------------------------
    # 0-4) 큰 좌표 배열 만들기 (batch 연산용)
    # ------------------------------------------------------------------
    def _build_positions_arrays(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
        max_agent_num: int,
        max_lane_num: int,
        max_route_lane_num: int,
        max_static_num: int,
        lane_len: int,
        route_lane_len: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """agent / lane / route_lane / static 좌표를 큰 배열로 모은다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 개수.
            lane_len_arr: (B,) lane 개수.
            route_lane_len_arr: (B,) route lane 개수.
            static_len_arr: (B,) static 개수.
            max_agent_num: 배치 내 최대 agent 수.
            max_lane_num: 배치 내 최대 lane 수.
            max_route_lane_num: 배치 내 최대 route lane 수.
            max_static_num: 배치 내 최대 static 수.
            lane_len: lane 한 개당 점 개수.
            route_lane_len: route lane 한 개당 점 개수.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - agents_xy: (B, max_agent_num, 2)
                - lanes_xy: (B, max_lane_num, lane_len, 2)
                - route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2)
                - static_xy: (B, max_static_num, 2)
        """
        batch_size: int = len(batch)

        # agents_xy: (B, max_agent_num, 2)
        agents_xy = np.zeros(
            (batch_size, max_agent_num, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_agent_num = int(agent_len_arr[b_idx])
            if real_agent_num > 0:
                agents_xy[b_idx, :real_agent_num, :] = sample[
                    "neighbor_agents_past"][real_agent_num * 0:real_agent_num,
                                            -1, 0:2]

        # lanes_xy: (B, max_lane_num, lane_len, 2)
        lanes_xy = np.zeros(
            (batch_size, max_lane_num, lane_len, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_lane_num = int(lane_len_arr[b_idx])
            if real_lane_num > 0:
                lanes_xy[b_idx, :real_lane_num, :, :] = sample[
                    "lanes"][:real_lane_num, :, 0:2]

        # route_lanes_xy: (B, max_route_lane_num, route_lane_len, 2)
        route_lanes_xy = np.zeros(
            (batch_size, max_route_lane_num, route_lane_len, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_route_lane_num = int(route_lane_len_arr[b_idx])
            if real_route_lane_num > 0:
                route_lanes_xy[b_idx, :real_route_lane_num, :, :] = sample[
                    "route_lanes"][:real_route_lane_num, :, 0:2]

        # static_xy: (B, max_static_num, 2)
        static_xy = np.zeros(
            (batch_size, max_static_num, 2),
            dtype=np.float32,
        )
        for b_idx, sample in enumerate(batch):
            real_static_num = int(static_len_arr[b_idx])
            if real_static_num > 0:
                static_xy[b_idx, :real_static_num, :] = sample[
                    "static_objects"][real_static_num * 0:real_static_num, 0:2]

        return agents_xy, lanes_xy, route_lanes_xy, static_xy

    # ------------------------------------------------------------------
    # 0-5) agent 유효 마스크
    # ------------------------------------------------------------------
    def _build_agent_valid_mask(
        self,
        agent_len_arr: np.ndarray,
        max_agent_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """실제 agent 위치가 있는 칸을 표시하는 마스크를 만든다.

        Args:
            agent_len_arr (np.ndarray): (B,) 각 샘플 agent 수.
            max_agent_num (int): 배치 내 최대 agent 수.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - agent_valid_mask: (B, max_agent_num) True=해당 칸에 agent 있음
                - has_any_agent_mask: (B,) True=해당 샘플에 agent 하나 이상
        """
        batch_size: int = agent_len_arr.shape[0]
        if max_agent_num > 0:
            idx = np.arange(max_agent_num)[None, :]  # (1, max_agent_num)
            agent_valid_mask = idx < agent_len_arr[:,
                                                   None]  # (B, max_agent_num)
        else:
            agent_valid_mask = np.zeros((batch_size, 0), dtype=bool)
        has_any_agent_mask = agent_len_arr > 0  # (B,)
        return agent_valid_mask, has_any_agent_mask

    # ------------------------------------------------------------------
    # 0-6) 중심을 NPC 기준으로 바꿀지 결정
    # ------------------------------------------------------------------
    def _maybe_update_center_by_npc(
        self,
        center_xy: np.ndarray,  # (B, 2)
        agents_xy: np.ndarray,  # (B, max_agent_num, 2)
        agent_len_arr: np.ndarray,  # (B,)
        agent_valid_mask: np.ndarray,  # (B, max_agent_num)
        max_agent_num: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """mode가 'npc'일 때, 랜덤 NPC 한 대를 중심으로 바꾼다.

        Args:
            center_xy (np.ndarray): (B, 2) 기존 중심 좌표.
            agents_xy (np.ndarray): (B, max_agent_num, 2) agent 현재 위치.
            agent_len_arr (np.ndarray): (B,) 각 샘플 agent 수.
            agent_valid_mask (np.ndarray): (B, max_agent_num) True=유효 위치.
            max_agent_num (int): 배치 내 최대 agent 수.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - center_xy_new: (B, 2) 새 중심 좌표.
                - center_agent_idx: (B,) 샘플마다 선택된 중심 agent 인덱스.
        """
        batch_size: int = agent_len_arr.shape[0]
        has_any_agent_mask = agent_len_arr > 0  # (B,)
        center_agent_idx = np.zeros(batch_size, dtype=np.int64)

        if self.center_crop_mode != "npc" or max_agent_num <= 0:
            return center_xy, center_agent_idx

        rand_vals = np.random.rand(batch_size, max_agent_num).astype(
            np.float32)  # (B, max_agent_num)
        rand_vals[~agent_valid_mask] = -1.0

        center_agent_idx = rand_vals.argmax(axis=1)  # (B,)
        center_npc_xy = agents_xy[
            np.arange(batch_size),
            np.clip(center_agent_idx, 0, max_agent_num - 1),
        ]  # (B, 2)

        center_xy_new = center_xy.copy()
        center_xy_new[has_any_agent_mask, :] = center_npc_xy[
            has_any_agent_mask, :]
        return center_xy_new, center_agent_idx

    # ------------------------------------------------------------------
    # 0-7-1) agent keep 마스크 계산
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # 0-7-1) agent keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_agent_mask(
            self,
            center_xy: np.ndarray,  # (B, 2)
            agents_xy: np.ndarray,  # (B, max_agent_num, 2)
            agent_len_arr: np.ndarray,  # (B,)
            max_agent_num: int,
            agent_valid_mask: np.ndarray,  # (B, max_agent_num)
    ) -> np.ndarray:
        """중심 기준 거리로 agent 를 남길지 여부를 계산한다.

        동작 요약:
          1) 각 agent 의 현재 위치와 중심 좌표(center_xy) 사이의 거리^2 을 계산한다.
          2) 거리^2 <= (반경 K)^2 인 agent 만 True 로 표시한다.
          3) agent_valid_mask 가 False 인 자리는 무조건 False 로 둔다.

        즉, “중심에서 K m 안에 실제로 존재하는 agent만 남긴다”는 규칙을
        배치 단위로 한 번에 적용하는 함수이다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점(ego 또는 임의 NPC)의 x,y.
            agents_xy (np.ndarray):
                - shape: (B, max_agent_num, 2)
                - 각 배치에서 agent 현재 위치의 x,y.
                - 실제 agent 수보다 큰 부분은 0 으로 채워져 있음.
            agent_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 agent 수.
            max_agent_num (int):
                - 배치 전체에서 가장 많은 agent 개수.
            agent_valid_mask (np.ndarray):
                - shape: (B, max_agent_num)
                - True 이면 해당 자리에는 실제 agent 가 있고,
                  False 이면 패딩(없는 자리).

        Returns:
            np.ndarray:
                - keep_agent_mask
                - shape: (B, max_agent_num)
                - True 인 위치의 agent 만 이후에 남기고,
                  False 인 agent 는 잘라낸다.
        """
        batch_size: int = agent_len_arr.shape[0]
        if max_agent_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        # diff_agent: (B, max_agent_num, 2)
        diff_agent = agents_xy - center_xy[:, None, :]
        # dist2_agent: (B, max_agent_num)
        dist2_agent = (diff_agent**2).sum(axis=-1)

        # 유효한 자리 + 반경 안에 있는 agent 만 남김
        keep_agent_mask = agent_valid_mask & (dist2_agent <= radius_sq)
        return keep_agent_mask

    # ------------------------------------------------------------------
    # 0-7-2) lane keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_lane_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        lanes_xy: np.ndarray,  # (B, max_lane_num, lane_len, 2)
        lane_len_arr: np.ndarray,  # (B,)
        max_lane_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 lane 을 남길지 여부를 계산한다.

        동작 요약:
          1) lane 하나는 여러 점으로 이루어진 선이다.
             · lanes_xy[b, l, p, :] 가 "l번째 lane 의 p번째 점" 좌표.
          2) 각 lane 에 대해, 모든 점과 중심 사이 거리^2 을 구한다.
          3) 그 중 "가장 가까운 점의 거리^2" 이 (반경 K)^2 이하이면
             그 lane 을 keep=True 로 표시한다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점(ego 또는 NPC)의 x,y.
            lanes_xy (np.ndarray):
                - shape: (B, max_lane_num, lane_len, 2)
                - lane 의 각 점 위치 x,y.
                - 실제 lane 수보다 큰 부분은 0 으로 채워져 있음.
            lane_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 lane 수.
            max_lane_num (int):
                - 배치 전체에서 가장 많은 lane 개수.

        Returns:
            np.ndarray:
                - keep_lane_mask
                - shape: (B, max_lane_num)
                - True 인 lane 은 그대로 유지하고,
                  False 인 lane 은 이후 슬라이스에서 제거된다.
        """
        batch_size: int = lane_len_arr.shape[0]
        if max_lane_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        lane_valid_mask = (np.arange(max_lane_num)[None, :]
                           < lane_len_arr[:, None])  # (B, max_lane_num)
        diff_lane = lanes_xy - center_xy[:, None,
                                         None, :]  # (B, max_lane_num, lane_len, 2)
        dist2_lane = (diff_lane**2).sum(axis=-1)  # (B, max_lane_num, lane_len)
        min_dist2_lane = dist2_lane.min(axis=-1)  # (B, max_lane_num)
        keep_lane_mask = lane_valid_mask & (min_dist2_lane <= radius_sq)
        return keep_lane_mask

    # ------------------------------------------------------------------
    # 0-7-3) static keep 마스크 계산
    # ------------------------------------------------------------------
    def _compute_keep_static_mask(
        self,
        center_xy: np.ndarray,  # (B, 2)
        static_xy: np.ndarray,  # (B, max_static_num, 2)
        static_len_arr: np.ndarray,  # (B,)
        max_static_num: int,
    ) -> np.ndarray:
        """중심 기준 거리로 static object 를 남길지 여부를 계산한다.

        동작 요약:
          1) 각 static object 의 위치와 중심 좌표 사이의 거리^2 을 계산한다.
          2) 거리^2 <= (반경 K)^2 인 static 만 keep=True 로 표시한다.
          3) 실제 static 이 없는 칸은 False 로 남는다.

        Args:
            center_xy (np.ndarray):
                - shape: (B, 2)
                - 각 배치에서 기준이 되는 점의 x,y.
            static_xy (np.ndarray):
                - shape: (B, max_static_num, 2)
                - 각 static 의 위치 x,y.
                - 실제 개수보다 큰 부분은 0 으로 채워져 있음.
            static_len_arr (np.ndarray):
                - shape: (B,)
                - 각 샘플의 실제 static 수.
            max_static_num (int):
                - 배치 전체에서 가장 많은 static 개수.

        Returns:
            np.ndarray:
                - keep_static_mask
                - shape: (B, max_static_num)
                - True 인 static 만 이후에 남기고,
                  False 인 static 은 잘라낸다.
        """
        batch_size: int = static_len_arr.shape[0]
        if max_static_num <= 0:
            return np.zeros((batch_size, 0), dtype=bool)

        radius_sq: float = float(self.center_crop_radius_m)**2

        static_valid_mask = (np.arange(max_static_num)[None, :]
                             < static_len_arr[:, None])  # (B, max_static_num)
        diff_static = static_xy - center_xy[:,
                                            None, :]  # (B, max_static_num, 2)
        dist2_static = (diff_static**2).sum(axis=-1)  # (B, max_static_num)
        keep_static_mask = static_valid_mask & (dist2_static <= radius_sq)
        return keep_static_mask

    # ------------------------------------------------------------------
    # 0-8) keep 마스크를 샘플 dict에 적용 (얇은 for 루프)
    # ------------------------------------------------------------------
    def _apply_keep_masks_to_batch(
        self,
        batch: List[Dict[str, Any]],
        agent_len_arr: np.ndarray,
        lane_len_arr: np.ndarray,
        route_lane_len_arr: np.ndarray,
        static_len_arr: np.ndarray,
        keep_agent_mask: np.ndarray,
        keep_lane_mask: np.ndarray,
        keep_route_lane_mask: np.ndarray,
        keep_static_mask: np.ndarray,
    ) -> None:
        """계산된 keep 마스크를 이용해 각 샘플 dict를 잘라낸다.

        Args:
            batch: 길이 B 샘플 리스트.
            agent_len_arr: (B,) agent 수.
            lane_len_arr: (B,) lane 수.
            route_lane_len_arr: (B,) route lane 수.
            static_len_arr: (B,) static 수.
            keep_agent_mask: (B, max_agent_num) agent keep 마스크.
            keep_lane_mask: (B, max_lane_num) lane keep 마스크.
            keep_route_lane_mask: (B, max_route_lane_num) route lane keep 마스크.
            keep_static_mask: (B, max_static_num) static keep 마스크.
        """
        batch_size: int = len(batch)
        for b_idx in range(batch_size):
            sample = batch[b_idx]

            # 1) agent / future
            if int(agent_len_arr[b_idx]) > 0:
                keep_agent_idx = np.nonzero(keep_agent_mask[b_idx])[0]
                sample["neighbor_agents_past"] = sample["neighbor_agents_past"][
                    keep_agent_idx]
                sample["neighbor_future_gt_3_dim"] = sample[
                    "neighbor_future_gt_3_dim"][keep_agent_idx]

            # 2) lane 관련 (superset)
            if int(lane_len_arr[b_idx]) > 0:
                keep_lane_idx = np.nonzero(keep_lane_mask[b_idx])[0]
                sample["lanes"] = sample["lanes"][keep_lane_idx]
                sample["lanes_speed_limit"] = sample["lanes_speed_limit"][
                    keep_lane_idx]
                sample["lanes_has_speed_limit"] = sample[
                    "lanes_has_speed_limit"][keep_lane_idx]

            # 2-1) route_lanes 관련 (lanes 와 독립 마스크)
            if int(route_lane_len_arr[b_idx]) > 0:
                keep_route_lane_idx = np.nonzero(keep_route_lane_mask[b_idx])[0]

                if "route_lanes" in sample:
                    sample["route_lanes"] = sample["route_lanes"][
                        keep_route_lane_idx]
                if "route_lanes_speed_limit" in sample:
                    sample["route_lanes_speed_limit"] = sample[
                        "route_lanes_speed_limit"][keep_route_lane_idx]
                if "route_lanes_has_speed_limit" in sample:
                    sample["route_lanes_has_speed_limit"] = sample[
                        "route_lanes_has_speed_limit"][keep_route_lane_idx]

            # 3) agent_route_lane_order (행: agent, 열: lane → lanes 기준)
            if int(agent_len_arr[b_idx]) > 0 and int(lane_len_arr[b_idx]) > 0:
                keep_agent_idx = np.nonzero(keep_agent_mask[b_idx])[0]
                keep_lane_idx = np.nonzero(keep_lane_mask[b_idx])[0]
                aro = sample["agent_route_lane_order"]  # (A_i, L_i)
                sample["agent_route_lane_order"] = aro[np.ix_(
                    keep_agent_idx, keep_lane_idx)]

            # 4) static_objects
            if int(static_len_arr[b_idx]) > 0:
                keep_static_idx = np.nonzero(keep_static_mask[b_idx])[0]
                sample["static_objects"] = sample["static_objects"][
                    keep_static_idx]

    # ------------------------------------------------------------------
    # 0) 중심 기준 배치 크로핑 (배치 연산 + 얇은 for 루프)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 1) 고정 shape 텐서들 스택
    # ------------------------------------------------------------------
    def _stack_fixed(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """모든 샘플에서 shape가 같은 키를 (B, ...) 텐서로 쌓는다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.
            key (str): 예: "ego_agent_past".
            dtype (torch.dtype): 출력 dtype.

        Returns:
            torch.Tensor: shape (B, *sample_shape)
        """
        return torch.stack(
            [torch.as_tensor(sample[key], dtype=dtype) for sample in batch],
            dim=0,
        )

    # ------------------------------------------------------------------
    # 2) (N_i, ...) → (B, target_len, ...) 패딩
    # ------------------------------------------------------------------
    def _pad_first_dim_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """샘플마다 길이가 다른 1D 축을 (B, target_len, ...)으로 0 패딩한다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 리스트.
            key (str): 예: "neighbor_agents_past".
            target_len (int): 두 번째 축 길이.
            dtype (torch.dtype): 출력 dtype.

        Returns:
            torch.Tensor: shape (B, target_len, *rest_shape)
        """
        batch_size: int = len(batch)
        if batch_size == 0:
            return torch.zeros((0, target_len), dtype=dtype)

        first_arr = batch[0][key]
        rest_shape = first_arr.shape[1:]  # 예: (time_len, 11)
        out_shape = (batch_size, target_len, *rest_shape)
        padded = torch.zeros(out_shape, dtype=dtype)

        for b_idx, sample in enumerate(batch):
            arr = sample[key]
            n_i = min(arr.shape[0], target_len)
            if n_i > 0:
                padded[b_idx, :n_i, ...] = torch.as_tensor(arr[:n_i],
                                                           dtype=dtype)

        return padded

    # ------------------------------------------------------------------
    # 3) (N_i, M_i) → (B, target0, target1) 패딩
    # ------------------------------------------------------------------
    def _pad_two_dims_to(
        self,
        batch: List[Dict[str, Any]],
        key: str,
        target_len0: int,
        target_len1: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """샘플마다 (행,열) 길이가 다른 배열을 (B, target0, target1)으로 패딩한다.

        현재는 agent_route_lane_order에 사용하며,
        패딩 구간은 -1로 채워서 "연결 없음"을 뜻하게 한다.
        """
        batch_size: int = len(batch)
        out = torch.full(
            (batch_size, target_len0, target_len1),
            fill_value=-1,
            dtype=dtype,
        )

        for b_idx, sample in enumerate(batch):
            arr = sample[key]
            n0 = min(arr.shape[0], target_len0)
            n1 = min(arr.shape[1], target_len1)
            if n0 > 0 and n1 > 0:
                out[b_idx, :n0, :n1] = torch.as_tensor(arr[:n0, :n1],
                                                       dtype=dtype)

        return out

    _FIXED_STACK_KEYS: Tuple[str, ...] = (
        "ego_agent_past",
        "ego_future_gt_3_dim",
        "planner_future_11_dim",
        "ego_agent_past_is_valid",
        "ego_future_gt_is_valid",
    )

    def _is_collatable_value(self, value: Any) -> bool:
        """이 값이 '텐서로 묶을 수 있는 값'인지 판별합니다.

        변경 포인트
        ----------
        - 문자열(str) 또는 문자열 배열/리스트는 텐서로 묶을 수 없으므로 False
          (대신 별도 경로로 List[str]로 모아서 반환합니다)

        Args:
            value: 샘플 dict 안의 어떤 값.

        Returns:
            bool:
                - True: torch.Tensor로 묶을 수 있는 값
                - False: 문자열 계열(또는 텐서화 불가 값)
        """
        if value is None:
            return True

        # ✅ 문자열 계열은 텐서로 묶지 않는다.
        if self._is_string_container_value(value):
            return False

        if isinstance(value, torch.Tensor):
            return True

        if isinstance(value, np.ndarray):
            # numpy 배열도 문자열 dtype이면 제외
            if value.dtype.kind in ("U", "S"):
                return False
            if value.dtype.kind == "O" and value.size > 0 and isinstance(
                    value.flat[0], (str, np.str_)):
                return False
            return True

        if isinstance(value, (bool, int, float, np.number)):
            return True

        if isinstance(value, (list, tuple)):
            # list/tuple이라도 전부 문자열이면 제외
            if len(value) > 0 and all(
                    isinstance(x, (str, np.str_)) for x in value):
                return False
            return True

        return False

    def _get_value_shape(self, value: Any) -> Tuple[int, ...]:
        """입력 값의 shape를 튜플로 얻습니다.

        Args:
            value:
                - torch.Tensor 또는 numpy.ndarray 또는 숫자 리스트/튜플/스칼라.

        Returns:
            Tuple[int, ...]:
                - 예: (A, T, 11), (L, P, 2), (A, L), (T,), ()(스칼라)
        """
        if isinstance(value, torch.Tensor):
            return tuple(int(x) for x in value.shape)
        if isinstance(value, np.ndarray):
            return tuple(int(x) for x in value.shape)

        # 리스트/튜플/스칼라 등: numpy로 shape만 확인
        arr = np.asarray(value)
        return tuple(int(x) for x in arr.shape)

    def _choose_output_dtype(self, value: Any) -> torch.dtype:
        """배치 텐서를 만들 때 사용할 dtype을 결정합니다.

        규칙(속도/안정성 목적)
        -------------------
        - 입력이 float 계열이면: torch.float32 로 통일 (float64 방지)
        - bool은: torch.bool 유지
        - 정수는: torch dtype 그대로 유지(보통 int64)

        Args:
            value:
                - None이 아닌 샘플 값(배치 내 최소 1개는 실제 값이 있어야 dtype을 정할 수 있음)

        Returns:
            torch.dtype:
                - 출력 텐서의 dtype
        """
        t = torch.as_tensor(value)
        if t.is_floating_point():
            return torch.float32
        return t.dtype

    def _collect_batch_keys(self, batch: List[Dict[str, Any]]) -> List[str]:
        """배치 안에 등장한 key를 모아 collate 대상으로 삼을 key 목록을 만듭니다.

        변경 포인트
        ----------
        - 기존에는 문자열(str)은 제외했는데,
          이제는 문자열도 "List[str]"로 반환하기 위해 key 목록에 포함합니다.

        Args:
            batch:
                길이 B의 샘플 dict 리스트.

        Returns:
            List[str]:
                collate 대상으로 삼을 key 목록(중복 제거, 등장 순서 최대한 유지)
        """
        seen = set()
        keys: List[str] = []

        for sample in batch:
            for k, v in sample.items():
                if k in seen:
                    continue

                # ✅ 텐서로 묶을 수 있거나, 문자열 계열이면 key를 포함
                if self._is_collatable_value(
                        v) or self._is_string_container_value(v):
                    seen.add(k)
                    keys.append(k)

        fixed_front: List[str] = []
        for k in self._FIXED_STACK_KEYS:
            if k in seen and k in keys:
                fixed_front.append(k)

        if fixed_front:
            rest = [k for k in keys if k not in fixed_front]
            return fixed_front + rest

        return keys

    def _stack_fixed_key_for_named_key(
        self,
        key: str,
        values: List[Any],
    ) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다. validity면 bool로 강제합니다.

        동작
        ----
        - ref_shape: 첫 번째 non-None 샘플 shape를 기준으로 (B, *ref_shape) 텐서를 만듭니다.
        - None인 샘플은 0/False로 남습니다.
        - key가 "*_is_valid"면 dtype을 torch.bool로 강제합니다.
          (입력이 int(0/1)이어도 자동으로 False/True로 변환됩니다)

        Args:
            key (str): key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)

        # 고정 key이므로 shape가 다르면 에러
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"key={key}, ref_shape={ref_shape}, got={self._get_value_shape(v)}"
                )

        # out: shape (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _stack_fixed_key(self, values: List[Any]) -> Optional[torch.Tensor]:
        """(고정 길이 key) 배치 텐서를 stack으로 바로 만듭니다.

        이 함수가 담당하는 경우
        ----------------------
        - ego_agent_past
        - ego_future_gt_3_dim
        - planner_future_11_dim
        - ego_agent_past_is_valid
        - ego_future_gt_is_valid

        위 5개는 "모든 샘플에서 길이가 항상 동일"하다는 전제이므로
        max length를 찾거나 0 padding으로 늘릴 필요가 없습니다.

        단, 값이 None일 수도 있으므로:
        - 배치 안에 실제 값이 하나라도 있으면 그 shape를 기준으로
          None인 샘플은 0으로 채운 텐서를 만들어 stack합니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *S) 텐서
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)  # 예: (T_past, 11)

        # non-None 값들의 shape가 동일한지 검사 (고정 key이므로 같아야 정상)
        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            if self._get_value_shape(v) != ref_shape:
                raise ValueError(
                    f"[Collate] 고정 key인데 shape가 샘플마다 다릅니다. "
                    f"ref_shape={ref_shape}, got={self._get_value_shape(v)}")

        # (B, *ref_shape)
        out = torch.zeros((batch_size, *ref_shape), dtype=out_dtype)

        # 값이 있는 샘플만 복사 (None은 0 그대로)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            out[b_idx] = torch.as_tensor(v, dtype=out_dtype)

        return out

    def _pad_and_stack_agent_route_lane_order(
        self,
        batch: List[Dict[str, Any]],
        max_agent_num: int,
        max_lane_num: int,
    ) -> torch.Tensor:
        """agent_route_lane_order를 (B, A_max, L_max) 텐서로 만들고 -1로 패딩합니다.

        핵심 규칙(중요)
        -------------
        - A_max(최대 agent 수)는 **agent_route_lane_order의 행 개수만** 보고 결정합니다.

        - L_max(최대 lane 수)는
          lanes.shape[0] 와 agent_route_lane_order.shape[1] 중 더 큰 값을 사용합니다.
          (lane 텐서 패딩과 같이 쓰기 쉬운 형태를 유지)

        패딩 값
        -------
        - 없는 값은 전부 -1로 채웁니다. (연결/매칭 없음 의미)

        Args:
            batch (List[Dict[str, Any]]):
                - 길이 B의 샘플 dict 리스트.

        Returns:
            torch.Tensor:
                - out: shape (B, A_max, L_max)
                - dtype: torch.int64
                - padding: -1
        """
        batch_size: int = int(len(batch))

        # max_agent_num, max_lane_num = self._infer_max_sizes_for_agent_route_lane_order(
        #     batch)
        out = torch.full(
            (batch_size, max_agent_num, max_lane_num),
            fill_value=-1,
            dtype=torch.int64,
        )

        for b_idx, sample in enumerate(batch):
            aro = sample.get("agent_route_lane_order", None)
            if aro is None:
                continue

            aro_arr = np.asarray(aro)
            # aro_arr: (A_i, L_i)
            if aro_arr.ndim != 2:
                continue

            a_i: int = min(int(aro_arr.shape[0]), max_agent_num)
            l_i: int = min(int(aro_arr.shape[1]), max_lane_num)

            if a_i <= 0 or l_i <= 0:
                continue

            out[b_idx, :a_i, :l_i] = torch.as_tensor(
                aro_arr[:a_i, :l_i],
                dtype=torch.int64,
            )

        return out

    def _pad_and_stack_variable_key_for_named_key(
        self,
        key: str,
        values: List[Any],
        batch: List[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape 기준으로 padding 후 쌓습니다. validity면 bool로 강제합니다.

        추가로 반영된 규칙
        ----------------
        1) lane_type (shape: (lane_num, 4))
           - 패딩을 0으로 하면 "전부 0 → 0번 카테고리"로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 (0,0,0,1)로 둡니다. (index 3이 1인 one-hot)

        2) left_line_type / right_line_type (shape: (lane_num, 13))
           - 패딩을 0으로 하면 특정 선 타입(0번)으로 잘못 해석될 수 있습니다.
           - 그래서 패딩 기본값을 index 11이 1인 one-hot로 둡니다.
             (선은 있으나 타입을 모름)

        3) lane 관련 key는 lanes의 최대 lane_num에 맞춰 첫 번째 축 길이를 보정합니다.
           - 어떤 샘플에서 lane_type이 None이라도,
             lanes 개수는 클 수 있어서 lane 차원이 어긋나지 않게 보강합니다.

        Args:
            key (str):
                key 이름. shape: ()
            values (List[Any]):
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None
            batch (List[Dict[str, Any]]):
                - 원본 샘플 dict 리스트(길이 B).
                - lanes 개수 등을 참고하기 위해 사용합니다.

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size: int = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype: torch.dtype = self._choose_output_dtype_for_key(
            key, ref_value)
        ref_shape: Tuple[int, ...] = self._get_value_shape(ref_value)
        ndim: int = int(len(ref_shape))

        max_shape = list(ref_shape)
        all_same_shape: bool = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)

            if len(shape_i) != ndim:
                raise ValueError(
                    f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. key={key}, "
                    f"ref_ndim={ndim}, got_ndim={len(shape_i)}")

            if shape_i != ref_shape:
                all_same_shape = False

            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # ✅ lane_type / line_type 계열은 lanes 최대 lane_num에 맞춰 첫 축을 보정
        special_spec = self._get_special_padding_category_spec(key)
        if special_spec is not None and ndim >= 1:
            max_lane_num_from_lanes: int = self._infer_max_lane_num_from_batch(
                batch)
            if max_lane_num_from_lanes > int(max_shape[0]):
                max_shape[0] = int(max_lane_num_from_lanes)

        # 빠른 경로: 패딩이 전혀 필요 없으면 stack
        # (special_spec가 있어도 max_shape가 ref_shape와 같으면 패딩 영역이 없으므로 stack 가능)
        if all_same_shape and (len(non_none_idx)
                               == batch_size) and (tuple(max_shape)
                                                   == tuple(ref_shape)):
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values],
                dim=0,
            )

        # out: shape (B, *max_shape)
        out_shape: Tuple[int,
                         ...] = (batch_size, *tuple(int(x) for x in max_shape))
        out: torch.Tensor = self._build_default_padded_tensor_for_key(
            key=key,
            out_shape=out_shape,
            dtype=out_dtype,
        )

        # 값 복사 (있는 샘플만 앞쪽부터 채움)
        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                out[b_idx] = t
                continue

            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *slices)] = t[slices]

        return out

    def _pad_and_stack_variable_key(
            self, values: List[Any]) -> Optional[torch.Tensor]:
        """(가변 길이 key) 배치 내 최대 shape를 기준으로 0 padding 배치 텐서를 만듭니다.

        처리 방식(핵심)
        -------------
        - key 이름을 보고 분기하지 않습니다.
        - 값이 ndarray/tensor 라면 그 shape를 보고,
          배치 내에서 각 차원별 최대값(max)을 구해 큰 텐서를 한 번만 만들고,
          샘플 값은 "앞쪽부터" 그대로 복사합니다.
        - padding 영역은 전부 0 입니다.
          (bool이면 False, float이면 0.0, int면 0)

        이 로직이 자연스럽게 커버하는 예시 shape
        ----------------------------------------
        - neighbor_agents_past:        (A_i, T_past, 11)  -> (B, A_max, T_past, 11)
        - neighbor_future_gt_3_dim:        (A_i, T_fut, 3)    -> (B, A_max, T_fut, 3)
        - lanes:                       (L_i, P_lane, 12) -> (B, L_max, P_max, 12)
        - lanes_len_is_valid:          (L_i, P_lane)      -> (B, L_max, P_max)
        - agent_route_lane_order:      (A_i, L_i)         -> (B, A_max, L_max)
        - road_edge:                   (E_i, P_edge, 2)   -> (B, E_max, P_max, 2)
        - stop_sign_points:            (S_i, P, 2)        -> (B, S_max, P_max, 2)
        - 그리고 그 외 "첫 번째 축이 개수"인 모든 값들

        None 처리
        --------
        - 어떤 샘플에서 None이면 그 샘플은 전부 0으로 남습니다.
        - 배치 전체가 None이면 None을 반환해서 상위에서 key 자체를 제외합니다.

        Args:
            values:
                - 길이 B 리스트
                - 각 원소는 ndarray/tensor 또는 None

        Returns:
            Optional[torch.Tensor]:
                - 성공 시: shape (B, *max_shape)
                - 전부 None이면: None
        """
        batch_size = int(len(values))
        non_none_idx = [i for i, v in enumerate(values) if v is not None]
        if not non_none_idx:
            return None

        ref_value = values[non_none_idx[0]]
        assert ref_value is not None

        out_dtype = self._choose_output_dtype(ref_value)
        ref_shape = self._get_value_shape(ref_value)
        ndim = int(len(ref_shape))

        # max_shape 계산: 배치 내 각 차원별 최대 크기
        max_shape = list(ref_shape)
        all_same_shape = True

        for i in non_none_idx[1:]:
            v = values[i]
            assert v is not None
            shape_i = self._get_value_shape(v)
            if len(shape_i) != ndim:
                raise ValueError(f"[Collate] 같은 key인데 ndim이 샘플마다 다릅니다. "
                                 f"ref_ndim={ndim}, got_ndim={len(shape_i)}")
            if shape_i != tuple(ref_shape):
                all_same_shape = False
            for d in range(ndim):
                if int(shape_i[d]) > int(max_shape[d]):
                    max_shape[d] = int(shape_i[d])

        # (빠른 경로) None도 없고, 모든 shape가 동일하면 stack이 제일 빠름
        if all_same_shape and (len(non_none_idx) == batch_size):
            # shape: (B, *ref_shape)
            return torch.stack(
                [torch.as_tensor(v, dtype=out_dtype) for v in values], dim=0)

        # 일반 경로: 0 padding 텐서 만들고 값 복사
        # out shape: (B, *max_shape)
        out = torch.zeros((batch_size, *max_shape), dtype=out_dtype)

        for b_idx in non_none_idx:
            v = values[b_idx]
            assert v is not None
            t = torch.as_tensor(v, dtype=out_dtype)

            if ndim == 0:
                # 스칼라: out[b_idx]에 바로 대입
                out[b_idx] = t
                continue

            # 각 차원별로 실제 길이만큼만 복사
            # 예: t.shape == (A_i, T, 11) 이면 slice(0,A_i), slice(0,T), slice(0,11)
            slices = tuple(slice(0, int(s)) for s in t.shape)
            out[(b_idx, *([slice(None)] * 0))]  # (형태 힌트용; 실제론 아래 라인만으로 충분)

            out[(b_idx, *slices)] = t[slices]

        return out

    def _build_collated_batch_tensors(
        self,
        batch: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """배치(dict 리스트)를 최종 배치 텐서(dict)로 변환합니다.

        변경 포인트
        ----------
        - 문자열(str) 계열 key는 텐서로 만들지 않고 List[str]로 반환합니다.
          (길이 B, None은 ""로 치환)
        - 나머지 숫자/배열 key는 기존대로 torch.Tensor 또는 None을 반환합니다.

        Args:
            batch (List[Dict[str, Any]]): 길이 B 샘플 dict 리스트

        Returns:
            Dict[str, Any]:
                - 텐서로 만들 수 있으면 torch.Tensor
                - 배치 전체가 None이면 None
                - 문자열 계열이면 List[str] (length=B)
        """
        batch_size: int = int(len(batch))
        if batch_size <= 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        keys: List[str] = self._collect_batch_keys(batch)
        batch_out: Dict[str, Any] = {}

        # 1) 고정 5개 key (항상 텐서로)
        for k in self._FIXED_STACK_KEYS:
            if k not in keys:
                continue
            values = [sample.get(k, None) for sample in batch]
            t = self._stack_fixed_key_for_named_key(k, values)
            batch_out[k] = t  # torch.Tensor 또는 None

        # 2) 나머지 key
        fixed_set = set(self._FIXED_STACK_KEYS)
        for k in keys:
            if k in fixed_set:
                continue

            values = [sample.get(k, None) for sample in batch]

            # 전부 None이면 None 유지
            if all(v is None for v in values):
                batch_out[k] = None
                continue

            # ✅ 문자열 계열이면 List[str]로 반환
            if self._should_return_string_list_for_key(k, values):
                batch_out[k] = self._collate_string_values_to_list(values)
                continue

            # agent_route_lane_order는 마지막에 전용 처리
            if k == "agent_route_lane_order":
                continue

            # 그 외는 기존 padding+stack 경로
            t = self._pad_and_stack_variable_key_for_named_key(k, values, batch)
            batch_out[k] = t  # torch.Tensor 또는 None

        # 3) agent_route_lane_order는 -1 padding 전용 처리
        neighbor_agents_past = batch_out.get("neighbor_agents_past", None)
        lanes = batch_out.get("lanes", None)

        if not isinstance(neighbor_agents_past, torch.Tensor):
            raise ValueError(
                "[Collate] neighbor_agents_past가 torch.Tensor가 아닙니다. "
                "문자열 key 처리와 무관하게, 입력 데이터가 깨졌을 가능성이 큽니다.")
        if not isinstance(lanes, torch.Tensor):
            raise ValueError("[Collate] lanes가 torch.Tensor가 아닙니다. "
                             "문자열 key 처리와 무관하게, 입력 데이터가 깨졌을 가능성이 큽니다.")

        max_agent_num = int(neighbor_agents_past.shape[1])  # shape: ()
        max_lane_num = int(lanes.shape[1])  # shape: ()

        batch_out[
            "agent_route_lane_order"] = self._pad_and_stack_agent_route_lane_order(
                batch=batch,
                max_agent_num=max_agent_num,
                max_lane_num=max_lane_num,
            )

        # 간단 정합 체크
        agent_route_lane_order = batch_out.get("agent_route_lane_order", None)
        if isinstance(agent_route_lane_order, torch.Tensor):
            assert int(agent_route_lane_order.shape[1]) == int(
                neighbor_agents_past.shape[1]
            ), ("agent_route_lane_order의 agent 수와 neighbor_agents_past의 agent 수가 "
                "일치하지 않습니다.")

        return batch_out

    def _is_validity_key_name(self, key: str) -> bool:
        """key 이름이 validity 마스크인지 빠르게 판별합니다.

        규칙
        ----
        - key가 "_is_valid" 로 끝나면 validity 마스크로 봅니다.
          예) ego_agent_past_is_valid, lanes_len_is_valid, lanes_is_valid, ...

        Args:
            key (str): sample dict의 key 이름. shape: ()

        Returns:
            bool:
                - True: validity 마스크 key
                - False: 그 외 key
        """
        return str(key).endswith("_is_valid")

    def _choose_output_dtype_for_key(self, key: str, value: Any) -> torch.dtype:
        """특정 key에 대해, collate 출력 dtype을 결정합니다.

        핵심 규칙
        --------
        - "*_is_valid" 류 key는 입력이 int(0/1) 이 섞여 있어도 **항상 torch.bool**로 강제합니다.
        - 그 외 key는 기존 규칙을 그대로 사용합니다.
          (float -> float32, 나머지 -> 입력 dtype 유지)

        Args:
            key (str): sample dict의 key 이름. shape: ()
            value (Any): None이 아닌 샘플 값(배치에서 dtype 기준으로 삼을 값)

        Returns:
            torch.dtype: 출력 텐서 dtype
        """
        if self._is_validity_key_name(key):
            return torch.bool
        return self._choose_output_dtype(value)

    def _as_bool_1d(
        self,
        mask: Any,
        expected_len: int,
    ) -> Optional[np.ndarray]:
        """입력 마스크를 (expected_len,) bool 배열로 안전 변환합니다.

        이 함수가 필요한 이유
        --------------------
        validity key는 데이터가 없으면 None일 수 있고,
        또 어떤 실수로 shape가 어긋난 값이 들어올 수도 있습니다.
        그럴 때 바로 예외를 내기보다, "안전하게 None 처리"해서
        나머지 crop 로직은 계속 돌도록 만드는 목적입니다.

        Args:
            mask: 원본 마스크 값. None 또는 numpy 배열/리스트 형태.
            expected_len: 기대하는 길이. shape: (expected_len,)

        Returns:
            Optional[np.ndarray]:
                - 정상 변환 시: np.ndarray(dtype=bool), shape (expected_len,)
                - 변환 불가/None이면: None
        """
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 1:
            return None
        if int(arr.shape[0]) != int(expected_len):
            return None
        return arr.astype(bool)

    def _as_bool_2d(
        self,
        mask: Any,
        expected_shape: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        """입력 마스크를 expected_shape=(N,P) bool 배열로 안전 변환합니다.

        Args:
            mask: 원본 마스크 값. None 또는 numpy 배열/리스트 형태.
            expected_shape: (N, P)
                - N: 객체 개수
                - P: 한 객체를 이루는 점 개수

        Returns:
            Optional[np.ndarray]:
                - 정상 변환 시: np.ndarray(dtype=bool), shape (N, P)
                - 변환 불가/None이면: None
        """
        if mask is None:
            return None
        arr = np.asarray(mask)
        if arr.ndim != 2:
            return None
        if (int(arr.shape[0]) != int(expected_shape[0])) or (int(
                arr.shape[1]) != int(expected_shape[1])):
            return None
        return arr.astype(bool)

    def _slice_first_dim_inplace(
            self,
            sample: Dict[str, Any],
            keys: List[str],
            keep_idx: np.ndarray,  # shape: (K,)
    ) -> None:
        """sample의 여러 key를 첫 번째 축 기준으로 같은 인덱스로 같이 자릅니다.

        Args:
            sample: 단일 샘플 dict.
            keys: 같이 잘라야 하는 key 목록.
            keep_idx: 남길 인덱스. shape: (K,)
        """
        for k in keys:
            if k not in sample:
                continue
            v = sample.get(k, None)
            if v is None:
                continue
            sample[k] = np.asarray(v)[keep_idx]

    def _sort_indices_by_distance(
            self,
            keep_idx: np.ndarray,  # shape: (K,)
            dist2_all: np.ndarray,  # shape: (N,)
    ) -> np.ndarray:
        """keep_idx를 '가까운 순'으로 재정렬합니다.

        Args:
            keep_idx: 남길 인덱스. shape: (K,)
            dist2_all: 전체 객체의 거리^2 배열. shape: (N,)

        Returns:
            sorted_keep_idx: 가까운 순으로 정렬된 keep_idx. shape: (K,)
        """
        if keep_idx.size <= 1:
            return keep_idx
        d = dist2_all[keep_idx]  # (K,)
        order = np.argsort(d, kind="mergesort")
        return keep_idx[order].astype(np.int64)

    def _keep_indices_from_points_xy(
            self,
            points_xy: np.ndarray,  # shape: (N, 2)
            center_xy: np.ndarray,  # shape: (2,)
            radius_sq: float,
            invalid_fill: np.float32,
            object_valid_1d: Optional[np.ndarray],  # shape: (N,)
    ) -> np.ndarray:
        """(N,2) 점 기준으로 반경 안에 들어오는 객체 인덱스를 만듭니다.

        규칙
        ----
        - 각 점의 (x,y)와 center_xy 사이 거리^2 <= radius_sq 이면 keep
        - object_valid_1d가 있으면 False인 객체는 무조건 제외
        - 반환 인덱스는 "가까운 순"으로 정렬해서 돌려줍니다.

        Args:
            points_xy: shape (N,2)
            center_xy: shape (2,)
            radius_sq: 반경^2 (스칼라)
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            object_valid_1d: shape (N,) 또는 None

        Returns:
            keep_idx: shape (K,), dtype int64
        """
        n = int(points_xy.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)

        pts = np.asarray(points_xy, dtype=np.float32)  # (N,2)
        diff = pts - center_xy[None, :]  # (N,2)
        dist2 = (diff * diff).sum(axis=-1)  # (N,)

        if object_valid_1d is not None:
            dist2 = np.where(object_valid_1d, dist2, invalid_fill)

        keep_mask = dist2 <= np.float32(radius_sq)
        keep_idx = np.nonzero(keep_mask)[0].astype(np.int64)
        return self._sort_indices_by_distance(keep_idx, dist2)

    def _keep_indices_from_polyline_xy(
            self,
            poly_xy: np.ndarray,  # shape: (N, P, 2)
            center_xy: np.ndarray,  # shape: (2,)
            radius_sq: float,
            invalid_fill: np.float32,
            point_valid_2d: Optional[np.ndarray],  # shape: (N, P)
            object_valid_1d: Optional[np.ndarray],  # shape: (N,)
    ) -> np.ndarray:
        """(N,P,2) 선/영역 형태에서 반경 안에 들어오는 객체 인덱스를 만듭니다.

        규칙
        ----
        - 한 객체는 P개의 점으로 이루어져 있다고 보고,
          그 중 center_xy에 가장 가까운 점의 거리^2이 radius_sq 이하이면 keep
        - point_valid_2d가 있으면 False인 점은 거리 계산에서 제외(매우 멀리 취급)
        - point_valid_2d가 없으면, (x,y) 둘 다 0인 점을 "없는 점"으로 보고 제외
        - object_valid_1d가 있으면 False인 객체는 무조건 제외
        - 반환 인덱스는 "가까운 순"으로 정렬해서 돌려줍니다.

        Args:
            poly_xy: shape (N,P,2)
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            point_valid_2d: shape (N,P) 또는 None
            object_valid_1d: shape (N,) 또는 None

        Returns:
            keep_idx: shape (K,), dtype int64
        """
        n = int(poly_xy.shape[0])
        if n <= 0:
            return np.zeros((0,), dtype=np.int64)
        p = int(poly_xy.shape[1])
        if p <= 0:
            return np.zeros((0,), dtype=np.int64)

        pts = np.asarray(poly_xy, dtype=np.float32)  # (N,P,2)
        diff = pts - center_xy[None, None, :]  # (N,P,2)
        dist2_pts = (diff * diff).sum(axis=-1)  # (N,P)

        if point_valid_2d is None:
            fallback_valid = np.any(np.abs(pts) > 0.0, axis=-1)  # (N,P)
            dist2_pts = np.where(fallback_valid, dist2_pts, invalid_fill)
        else:
            dist2_pts = np.where(point_valid_2d, dist2_pts, invalid_fill)

        min_dist2 = dist2_pts.min(axis=-1)  # (N,)

        if object_valid_1d is not None:
            min_dist2 = np.where(object_valid_1d, min_dist2, invalid_fill)

        keep_mask = min_dist2 <= np.float32(radius_sq)
        keep_idx = np.nonzero(keep_mask)[0].astype(np.int64)
        return self._sort_indices_by_distance(keep_idx, min_dist2)

    def _compute_center_xy_for_sample(
        self,
        sample: Dict[str, Any],
    ) -> Optional[np.ndarray]:
        """샘플에서 중심점(center_xy)을 계산합니다.

        규칙
        ----
        - 기본은 ego 기준:
            center_xy = ego_agent_past[-1, 0:2]  # shape (2,)
        - center_crop_mode == "npc" 이면:
            neighbor_agents_is_valid==True인 agent들 중 하나를 랜덤 선택해
            center_xy = neighbor_agents_past[selected, -1, 0:2]
          유효 agent가 없으면 ego 기준으로 fallback 합니다.

        Args:
            sample: 단일 샘플 dict.

        Returns:
            Optional[np.ndarray]:
                - 성공 시: center_xy, shape (2,), dtype float32
                - ego 좌표를 뽑을 수 없으면: None
        """
        ego_past = sample.get("ego_agent_past", None)
        if ego_past is None:
            return None

        ego_past_arr = np.asarray(ego_past)
        if ego_past_arr.ndim < 2 or int(ego_past_arr.shape[0]) <= 0:
            return None

        ego_xy = np.asarray(ego_past_arr[-1, 0:2], dtype=np.float32)  # (2,)
        center_xy = ego_xy

        if self.center_crop_mode != "npc":
            return center_xy

        neighbor_past = sample.get("neighbor_agents_past", None)
        if neighbor_past is None:
            return center_xy

        neighbor_past_arr = np.asarray(neighbor_past)
        if neighbor_past_arr.ndim < 3 or int(neighbor_past_arr.shape[0]) <= 0:
            return center_xy

        a = int(neighbor_past_arr.shape[0])
        neighbor_valid_1d = self._as_bool_1d(sample.get(
            "neighbor_agents_is_valid", None),
                                             expected_len=a)
        if neighbor_valid_1d is None:
            candidate_idx = np.arange(a, dtype=np.int64)
        else:
            candidate_idx = np.nonzero(neighbor_valid_1d)[0].astype(np.int64)

        if int(candidate_idx.shape[0]) <= 0:
            return center_xy

        chosen = int(candidate_idx[np.random.randint(int(
            candidate_idx.shape[0]))])
        center_xy = np.asarray(neighbor_past_arr[chosen, -1, 0:2],
                               dtype=np.float32)
        return center_xy

    def _crop_poly_object_group_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape (2,)
        radius_sq: float,
        invalid_fill: np.float32,
        points_key: str,
        is_valid_key: str,
    ) -> None:
        """(N,P,2) + (N,) 형태인 지도/영역 요소를 반경 기준으로 crop 합니다.

        예:
          - stop_sign_points / stop_sign_is_valid
          - crosswalk_points / crosswalk_is_valid
          - speed_bump_points / speed_bump_is_valid
          - driveway_points / driveway_is_valid

        Args:
            sample: 단일 샘플 dict.
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
            points_key: 예) "crosswalk_points"
            is_valid_key: 예) "crosswalk_is_valid"
        """
        pts = sample.get(points_key, None)
        if pts is None:
            return
        pts_arr = np.asarray(pts)
        if pts_arr.ndim != 3:
            return
        n = int(pts_arr.shape[0])
        if n <= 0:
            return

        obj_valid_1d = self._as_bool_1d(sample.get(is_valid_key, None),
                                        expected_len=n)
        keep_idx = self._keep_indices_from_polyline_xy(
            poly_xy=np.asarray(pts_arr, dtype=np.float32),
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=None,
            object_valid_1d=obj_valid_1d,
        )
        self._slice_first_dim_inplace(sample,
                                      keys=[points_key, is_valid_key],
                                      keep_idx=keep_idx)

    def _crop_road_edge_group_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """road_edge 관련 key들을 반경 기준으로 함께 crop 합니다.

        함께 자르는 key
        --------------
        - road_edge: shape (E, safety_len, 2)
        - road_edge_type: shape (E, 3)
        - road_edge_is_valid: shape (E,) (존재하면 같이)

        Args:
            sample: 단일 샘플 dict.
            center_xy: shape (2,)
            radius_sq: 반경^2
            invalid_fill: invalid를 멀리 보내기 위한 큰 값
        """
        road_edge = sample.get("road_edge", None)
        if road_edge is None:
            return

        road_edge_arr = np.asarray(road_edge)
        if road_edge_arr.ndim != 3:
            return
        e = int(road_edge_arr.shape[0])
        if e <= 0:
            return

        road_edge_valid_1d = self._as_bool_1d(sample.get(
            "road_edge_is_valid", None),
                                              expected_len=e)

        keep_idx = self._keep_indices_from_polyline_xy(
            poly_xy=np.asarray(road_edge_arr, dtype=np.float32),
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=None,
            object_valid_1d=road_edge_valid_1d,
        )
        self._slice_first_dim_inplace(
            sample,
            keys=["road_edge", "road_edge_type", "road_edge_is_valid"],
            keep_idx=keep_idx,
        )

    def _crop_agent_route_lane_order_inplace(
        self,
        sample: Dict[str, Any],
        keep_agent_idx: Optional[np.ndarray],
        keep_lane_idx: Optional[np.ndarray],
    ) -> None:
        """agent_route_lane_order를 (행=agent, 열=lane) 기준으로 동시에 crop 합니다.

        Args:
            sample: 단일 샘플 dict.
            keep_agent_idx: shape (K_a,) 또는 None (None이면 agent는 유지)
            keep_lane_idx: shape (K_l,) 또는 None (None이면 lane은 유지)
        """
        aro = sample.get("agent_route_lane_order", None)
        if aro is None:
            return

        aro_arr = np.asarray(aro)
        if aro_arr.ndim != 2:
            return

        if keep_agent_idx is None:
            keep_agent_idx_eff = np.arange(int(aro_arr.shape[0]),
                                           dtype=np.int64)
        else:
            keep_agent_idx_eff = keep_agent_idx.astype(np.int64)

        if keep_lane_idx is None:
            keep_lane_idx_eff = np.arange(int(aro_arr.shape[1]), dtype=np.int64)
        else:
            keep_lane_idx_eff = keep_lane_idx.astype(np.int64)

        sample["agent_route_lane_order"] = aro_arr[np.ix_(
            keep_agent_idx_eff, keep_lane_idx_eff)]

        # agent_route_lane_order_is_valid 는 (agent 축)만 맞추면 됨
        aro_valid = sample.get("agent_route_lane_order_is_valid", None)
        if aro_valid is not None:
            aro_valid_arr = np.asarray(aro_valid)
            if aro_valid_arr.ndim == 1 and int(aro_valid_arr.shape[0]) == int(
                    keep_agent_idx_eff.shape[0]):
                # 이미 앞에서 agent 기준으로 잘려져 들어온 경우라면 그대로 둔다.
                return
            if aro_valid_arr.ndim == 1 and int(aro_valid_arr.shape[0]) >= int(
                    keep_agent_idx_eff.max(initial=-1) + 1):
                sample["agent_route_lane_order_is_valid"] = aro_valid_arr[
                    keep_agent_idx_eff]

    def _build_center_crop_constants(self) -> Tuple[float, np.float32]:
        """center crop에 필요한 상수들을 한 번에 만든다.

        Args:
            없음

        Returns:
            Tuple[float, np.float32]:
                - radius_sq: float, 반경의 제곱(= (center_crop_radius_m)^2). shape: ()
                - invalid_fill: np.float32, invalid를 멀리 보내기 위한 매우 큰 값. shape: ()
        """
        radius_sq: float = float(self.center_crop_radius_m)**2
        invalid_fill: np.float32 = np.float32(1.0e12)
        return radius_sq, invalid_fill

    def _crop_agents_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> Optional[np.ndarray]:
        """Agent 기준으로 crop 하고, agent 관련 의존 key들도 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: neighbor_agents_past (현재 위치: neighbor_agents_past[:, -1, 0:2])
        - 같이 자름:
            neighbor_agents_past
            neighbor_future_gt_3_dim
            neighbor_agents_past_is_valid
            neighbor_agents_is_valid
            neighbor_future_gt_is_valid
            agent_route_lane_order_is_valid  (agent 축과 길이 맞추기)

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()

        Returns:
            Optional[np.ndarray]:
                - keep_agent_idx: np.ndarray(int64), shape (K_a,)
                - agent 정보를 못 자르면 None
        """
        neighbor_past = sample.get("neighbor_agents_past", None)
        if neighbor_past is None:
            return None

        neighbor_past_arr = np.asarray(neighbor_past)
        if neighbor_past_arr.ndim < 3:
            return None

        a = int(neighbor_past_arr.shape[0])
        if a <= 0:
            return np.zeros((0,), dtype=np.int64)

        agents_xy = np.asarray(neighbor_past_arr[:, -1, 0:2],
                               dtype=np.float32)  # (A,2)
        neighbor_valid_1d = self._as_bool_1d(
            sample.get("neighbor_agents_is_valid", None),
            expected_len=a,
        )

        keep_agent_idx = self._keep_indices_from_points_xy(
            points_xy=agents_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            object_valid_1d=neighbor_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "neighbor_agents_past",
                "neighbor_future_gt_3_dim",
                "neighbor_agents_past_is_valid",
                "neighbor_agents_is_valid",
                "neighbor_future_gt_is_valid",
                "agent_route_lane_order_is_valid",
            ],
            keep_idx=keep_agent_idx,
        )
        return keep_agent_idx

    def _crop_lanes_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> Optional[np.ndarray]:
        """Lane 기준으로 crop 하고, lane 관련 의존 key들도 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: lanes (좌표: lanes[:, :, 0:2])
        - 같이 자름:
            lanes
            lanes_speed_limit
            lanes_has_speed_limit
            lane_type
            left_line_type
            right_line_type
            lanes_len_is_valid
            lanes_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()

        Returns:
            Optional[np.ndarray]:
                - keep_lane_idx: np.ndarray(int64), shape (K_l,)
                - lanes를 못 자르면 None
        """
        lanes = sample.get("lanes", None)
        if lanes is None:
            return None

        lanes_arr = np.asarray(lanes)
        if lanes_arr.ndim < 3:
            return None

        l = int(lanes_arr.shape[0])
        if l <= 0:
            return np.zeros((0,), dtype=np.int64)

        lane_len = int(lanes_arr.shape[1])
        lanes_xy = np.asarray(lanes_arr[:, :, 0:2],
                              dtype=np.float32)  # (L, lane_len, 2)

        lanes_len_is_valid_2d = self._as_bool_2d(
            sample.get("lanes_len_is_valid", None),
            expected_shape=(l, lane_len),
        )
        lanes_is_valid_1d = self._as_bool_1d(
            sample.get("lanes_is_valid", None),
            expected_len=l,
        )

        keep_lane_idx = self._keep_indices_from_polyline_xy(
            poly_xy=lanes_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=lanes_len_is_valid_2d,
            object_valid_1d=lanes_is_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "lanes",
                "lanes_speed_limit",
                "lanes_has_speed_limit",
                "lane_type",
                "left_line_type",
                "right_line_type",
                "lanes_len_is_valid",
                "lanes_is_valid",
            ],
            keep_idx=keep_lane_idx,
        )
        return keep_lane_idx

    def _crop_route_lanes_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Route lane 기준으로 crop 하고, route lane 관련 의존 key들을 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: route_lanes (좌표: route_lanes[:, :, 0:2])
        - 같이 자름:
            route_lanes
            route_lanes_speed_limit
            route_lanes_has_speed_limit
            route_lanes_len_is_valid
            route_lanes_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        route_lanes = sample.get("route_lanes", None)
        if route_lanes is None:
            return

        route_arr = np.asarray(route_lanes)
        if route_arr.ndim < 3:
            return

        r = int(route_arr.shape[0])
        if r <= 0:
            return

        route_len = int(route_arr.shape[1])
        route_xy = np.asarray(route_arr[:, :, 0:2],
                              dtype=np.float32)  # (R, route_len, 2)

        route_len_is_valid_2d = self._as_bool_2d(
            sample.get("route_lanes_len_is_valid", None),
            expected_shape=(r, route_len),
        )
        route_is_valid_1d = self._as_bool_1d(
            sample.get("route_lanes_is_valid", None),
            expected_len=r,
        )

        keep_route_idx = self._keep_indices_from_polyline_xy(
            poly_xy=route_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            point_valid_2d=route_len_is_valid_2d,
            object_valid_1d=route_is_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=[
                "route_lanes",
                "route_lanes_speed_limit",
                "route_lanes_has_speed_limit",
                "route_lanes_len_is_valid",
                "route_lanes_is_valid",
            ],
            keep_idx=keep_route_idx,
        )

    def _crop_static_objects_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Static object 기준으로 crop 하고, static 관련 의존 key들을 같이 자른다.

        기준/의존 관계
        ------------
        - 기준: static_objects (좌표: static_objects[:, 0:2])
        - 같이 자름:
            static_objects
            static_objects_is_valid

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        static_objects = sample.get("static_objects", None)
        if static_objects is None:
            return

        static_arr = np.asarray(static_objects)
        if static_arr.ndim < 2:
            return

        s = int(static_arr.shape[0])
        if s <= 0:
            return

        static_xy = np.asarray(static_arr[:, 0:2], dtype=np.float32)  # (S,2)
        static_valid_1d = self._as_bool_1d(
            sample.get("static_objects_is_valid", None),
            expected_len=s,
        )

        keep_static_idx = self._keep_indices_from_points_xy(
            points_xy=static_xy,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            object_valid_1d=static_valid_1d,
        )

        self._slice_first_dim_inplace(
            sample,
            keys=["static_objects", "static_objects_is_valid"],
            keep_idx=keep_static_idx,
        )

    def _crop_safety_and_map_objects_inplace(
        self,
        sample: Dict[str, Any],
        center_xy: np.ndarray,  # shape: (2,)
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """Stop/Crosswalk/SpeedBump/Driveway/RoadEdge 그룹을 한 번에 crop 한다.

        Args:
            sample: 단일 샘플 dict
            center_xy: 중심 좌표, shape (2,)
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        # (N,P,2) + (N,) 쌍들
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="stop_sign_points",
            is_valid_key="stop_sign_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="crosswalk_points",
            is_valid_key="crosswalk_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="speed_bump_points",
            is_valid_key="speed_bump_is_valid",
        )
        self._crop_poly_object_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
            points_key="driveway_points",
            is_valid_key="driveway_is_valid",
        )

        # road_edge (+ type + (있으면) road_edge_is_valid)
        self._crop_road_edge_group_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

    def _center_crop_one_sample_inplace(
        self,
        sample: Dict[str, Any],
        radius_sq: float,
        invalid_fill: np.float32,
    ) -> None:
        """샘플 1개를 center 기준 거리로 crop 하는 전체 절차를 수행한다.

        Args:
            sample: 단일 샘플 dict
            radius_sq: 반경^2, shape ()
            invalid_fill: invalid 마스킹용 큰 값, shape ()
        """
        center_xy = self._compute_center_xy_for_sample(sample)
        if center_xy is None:
            return

        keep_agent_idx = self._crop_agents_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        keep_lane_idx = self._crop_lanes_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

        self._crop_route_lanes_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        self._crop_static_objects_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )
        self._crop_safety_and_map_objects_inplace(
            sample=sample,
            center_xy=center_xy,
            radius_sq=radius_sq,
            invalid_fill=invalid_fill,
        )

        # (행=agent, 열=lane) 동시 crop
        self._crop_agent_route_lane_order_inplace(
            sample=sample,
            keep_agent_idx=keep_agent_idx,
            keep_lane_idx=keep_lane_idx,
        )

    def _center_crop_batch_batched(
        self,
        batch: List[Dict[str, Any]],
    ) -> None:
        """배치 샘플들을 중심점(center_xy) 기준 거리로 crop 합니다.

        - batch는 샘플마다 (agent/lane/route/static 개수)가 달라 완전 벡터화가 이득이 크지 않아,
          샘플 단위 루프는 유지하되, 각 샘플 안에서 계산은 numpy 연산으로 처리합니다.
        """
        if not batch:
            return

        radius_sq, invalid_fill = self._build_center_crop_constants()

        for sample in batch:
            self._center_crop_one_sample_inplace(
                sample=sample,
                radius_sq=radius_sq,
                invalid_fill=invalid_fill,
            )

    # ------------------------------------------------------------------
    # 4) collate 본체
    # ------------------------------------------------------------------
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """단일 샘플 dict 리스트를 (B, ·) 배치 dict로 변환한다.

        Returns:
            Dict[str, Any]:
                - 숫자/배열 값: torch.Tensor 또는 None
                - 문자열 값: List[str] (length=B)
        """
        if len(batch) == 0:
            raise ValueError("빈 batch가 들어왔습니다.")

        if self._should_center_crop():
            raise NotImplementedError("현재 구현에서는 center crop을 사용할 수 없습니다.")
            # self._center_crop_batch_batched(batch)

        return self._build_collated_batch_tensors(batch)
