import argparse
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from requests.exceptions import HTTPError

import shutil
import time
import wandb
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusion_planner.utils import ddp
from timm.utils import ModelEma
from diffusion_planner.model.diffusion_planner import Diffusion_Planner
import argparse
import os
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger

from datetime import datetime
from torch import optim
from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model

import torch
from typing import Any, Dict, List, Optional, Tuple, Iterator

def setup_logger_and_purge(
    args: argparse.Namespace,
    global_rank: int,
    wandb_id: Optional[str],
    allow_val_change: bool,
) -> Logger:
    """TensorBoard / W&B 로거를 만들고, 기존 아티팩트를 정리한다."""
    wandb_logger = Logger(
        args,
        wandb_resume_id=wandb_id,
        rank=global_rank,
        allow_val_change=allow_val_change,
    )

    if global_rank == 0 and args.remove_existing_wb_weight:
        api = wandb.Api()
        entity = wandb.run.entity
        project = wandb.run.project
        purge_collection(api, entity, project, f"{args.name}_latest-model")
        purge_collection(api, entity, project, f"{args.name}_best-model")

    # ✅ 분산이 "실제로 초기화된 경우"에만 barrier 호출
    if bool(getattr(args, "ddp", False)) and ddp.is_dist_avail_and_initialized():
        torch.distributed.barrier()

    return wandb_logger


def purge_collection(api, entity, project, coll_name):
    """
    entity/project/coll_name 경로의 해당 컬렉션에 들어있는 아티팩트 버전 목록을 삭제

    jksg01019-naver-labs/Diffusion-Planner/nuplan_womd_latest-model
    jksg01019-naver-labs/Diffusion-Planner/nuplan_womd_best-model
    """
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


def safe_get_artifacts(api, type_name, path):
    try:
        return list(api.artifacts(type_name, path))
    except (wandb.errors.CommError, HTTPError) as e:
        # 404 또는 권한 오류 → 컬렉션이 아직 없다고 판단
        print(f"[SKIP] '{path}' 컬렉션 없음/권한 문제: {e}")
        return []


def _select_latest_like_tag(save_path: str) -> str:
    """DeepSpeed 체크포인트 루트에서 '가장 최신 latest 계열 태그'를 고른다.

    동작 규칙:
    1) save_path 아래에 'latest'라는 디렉터리가 정확히 있으면 그걸 최우선으로 사용한다.
       (사람이 심볼릭 링크/복사로 만들어둔 경우를 고려)
    2) 그 외에는 이름에 "latest"가 들어간 디렉터리들(예: latest_epoch-000010) 중
       정렬 기준 가장 마지막(=가장 큰 숫자) 것을 선택한다.
       - epoch를 6자리 0패딩으로 쓰고 있으므로 문자열 정렬로도 최신 선택이 가능하다.

    Args:
        save_path: 체크포인트 루트 디렉터리.

    Returns:
        str: load_checkpoint에 넣을 tag 문자열.
    """
    tag: str = "latest"
    if not os.path.isdir(save_path):
        return tag

    try:
        children = [
            d for d in os.listdir(save_path)
            if os.path.isdir(os.path.join(save_path, d))
        ]
    except OSError as e:
        raise RuntimeError(
            f"[DeepSpeed] 체크포인트 디렉터리 목록을 읽는 중 오류 발생: {save_path}, {e}") from e

    # 1) 정확히 "latest" 디렉터리가 있으면 우선 사용
    if "latest" in children:
        return "latest"

    # 2) "latest"가 포함된 태그 디렉터리들 중 가장 최신(정렬 마지막) 선택
    candidate_tags = [d for d in children if "latest" in d]
    if candidate_tags:
        return sorted(candidate_tags)[-1]

    return tag


def _load_model_and_ema_state_only_from_deepspeed_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    global_rank: int,
) -> Optional[ModelEma]:
    """DeepSpeed 폴더에서 **모델/EMA 가중치만** 읽어오는 함수.

    이 함수는 optimizer나 스케줄러 상태는 전혀 건드리지 않고,
    디스크에 저장된 DeepSpeed 체크포인트 폴더에서
    순수하게 모델 값(가중치)만 불러와 현재 모델에 채워 넣는다.

    사용 의도:
        * stage1에서 이미 학습한 모델 값을 가져오되,
          stage2에서는 새로운 학습 설정(optimizer, 학습률 등)을 그대로 쓰고 싶을 때.
        * encoder_local을 고정(freeze)해서 메모리를 아끼고 싶지만,
          DeepSpeed의 복잡한 재개(load) 로직은 피하고 싶을 때.

    동작 요약:
        1) save_path 안에서 "latest"가 들어간 하위 폴더 이름을 고른다.
           예: latest_epoch-000001
        2) 그 안의 mp_rank_XX_model_states.pt 파일을 열어
           저장된 모델 값들을 읽어온다.
        3) 읽어온 모델 값을 현재 모델(diffusion_planner)에 넣는다.
        4) EMA 모델이 있으면, 현재 모델 값으로 EMA 모델도 맞춰 준다.

    Args:
        save_path (str):
            DeepSpeed 체크포인트가 들어 있는 루트 폴더 경로.
            예: "./training_log/실험이름/2025-12-10-12:52:12/"
        diffusion_planner (nn.Module):
            DeepSpeed로 감싸진 주행 계획 모델 또는 그와 같은 구조의 모델.
        model_ema (Optional[ModelEma]):
            EMA(지수 이동 평균)를 추적하는 보조 모델 래퍼.
            없으면 None.
        global_rank (int):
            분산 학습에서 이 프로세스의 전체 순번.
            주로 로그 출력에만 사용된다.

    Returns:
        Optional[ModelEma]:
            모델 값이 동기화된 EMA 래퍼를 돌려준다.
            EMA를 쓰지 않는 경우에는 원래대로 None을 돌려준다.
    """
    tag = _select_latest_like_tag(save_path)
    tag_dir = os.path.join(save_path, tag)

    # tag 디렉터리 안에서 *_model_states.pt 중 첫 번째를 찾는 방식
    candidates = [
        f for f in os.listdir(tag_dir) if f.endswith("_model_states.pt")
    ]
    if not candidates:
        raise FileNotFoundError(f"No *_model_states.pt found in {tag_dir}")

    mp_rank_str = sorted(candidates)[0]  # 보통 mp_rank_00_model_states.pt
    ckpt_path = os.path.join(tag_dir, mp_rank_str)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_like = ckpt.get("ema_state_dict", ckpt["module"])

    base_model: nn.Module = getattr(diffusion_planner, "module",
                                    diffusion_planner)
    incompatible = base_model.load_state_dict(state_like, strict=False)

    if global_rank == 0:
        print(f"[ModelOnly<DeepSpeed>] LOAD MODEL PARAM from {ckpt_path}")
        if getattr(incompatible, "missing_keys", None):
            print(
                f"[ModelOnly<DeepSpeed>] missing_keys: {incompatible.missing_keys}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            print(
                f"[ModelOnly<DeepSpeed>] unexpected_keys: {incompatible.unexpected_keys}"
            )

    if model_ema is not None:
        ema_model = getattr(model_ema, "ema", model_ema)
        ema_model.load_state_dict(base_model.state_dict(), strict=False)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)

    return model_ema


def _is_deepspeed_checkpoint_dir(save_path: str) -> bool:
    """주어진 경로가 DeepSpeed checkpoint 디렉터리인지 대략 판별한다.

    규칙(간단한 휴리스틱):
      - save_path 하위 폴더들 중 이름에 "latest" 또는 "best" 가 들어가는
        디렉터리를 찾고
      - 그 디렉터리 안에 "mp_rank_" 로 시작하는 하위 파일/디렉터리가 있으면
        DeepSpeed 가 저장한 구조라고 본다.

    Args:
        save_path:
            체크포인트 루트 디렉터리 경로.

    Returns:
        bool:
            True  → DeepSpeed 형식으로 저장된 디렉터리라고 판단.
            False → PyTorch 단일 .pth 형식이라고 보고 처리한다.
    """
    if not os.path.isdir(save_path):
        return False

    try:
        # save_path 하위의 1차 자식 디렉터리 목록
        top_children = os.listdir(save_path)
    except OSError:
        return False

    # 이름에 "latest" 또는 "best" 가 들어가는 하위 디렉터리만 후보로 사용
    candidate_tag_dirs = []
    for name in sorted(top_children):
        if ("latest" in name) or ("best" in name):
            full = os.path.join(save_path, name)
            if os.path.isdir(full):
                candidate_tag_dirs.append(full)

    # 후보 tag 디렉터리들 중 하나라도 mp_rank_* 를 포함하면 DeepSpeed 형식으로 판단
    for tag_dir in candidate_tag_dirs:
        try:
            children = os.listdir(tag_dir)
        except OSError:
            continue
        for name in children:
            # 예: mp_rank_00_model_states.pt, mp_rank_00/
            if name.startswith("mp_rank_"):
                return True

    return False


def _restore_ema_and_model_from_client_state_for_deepspeed(
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    client_state: Dict[str, Any],
    global_rank: int,
    load_ema_for_model: bool,
    load_ema_for_ema_model: bool,
) -> Optional[ModelEma]:
    """client_state 정보를 이용해 EMA와 (필요하다면) 모델 가중치를 복원한다.

    Args:
        diffusion_planner (nn.Module): DeepSpeed 엔진 객체.
        model_ema (Optional[ModelEma]): EMA 래퍼. 없으면 None.
        client_state (Dict[str, Any]): 체크포인트에 저장된 작은 상태 dict.
            - client_state.get("ema_state_dict"): Dict[str, Tensor]
              각 텐서의 shape 는 원본 모델 파라미터와 동일
              (예: (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등).

            save_checkpoint 때 넘겨준 client_state 그대로 돌아온 dict.
            epoch, loss, wandb_id, ema_state_dict 가 그 안에 들어있음
            ( client_state 내용을 포함한 메타 정보를 별도의
            체크포인트 파일(예: mp_rank_00_model_states.pt 안의 딕셔너리 키)로 같이 저장해 둔다. )

        global_rank (int): 전역 rank. 0일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]: EMA 상태가 반영된 ModelEma 또는 None.
    """
    ema_state_dict: Optional[Dict[str, Any]] = client_state.get(
        "ema_state_dict", None)

    # 1) EMA 객체가 있고, ema_state_dict 가 있으면 EMA 가중치 복원
    if load_ema_for_ema_model:
        assert (model_ema is not None) and (ema_state_dict is not None), \
            "[DeepSpeed LOAD MODEL PARAM] Restoring an EMA model requires both `model_ema` and `ema_state_dict`."
        try:
            ema_model: nn.Module = getattr(model_ema, "ema", model_ema)
            # ema_model.state_dict() 의 각 텐서 shape:
            #   (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW) 등
            ema_model.load_state_dict(ema_state_dict, strict=False)
            ema_model.eval()
            for p in ema_model.parameters():
                p.requires_grad_(False)
            if global_rank == 0:
                print("[DeepSpeed LOAD MODEL PARAM]EMA state load done")
        except Exception as e:
            if global_rank == 0:
                print(f"[DeepSpeed LOAD MODEL PARAM]EMA state load FAIL: {e}")

    if load_ema_for_model:
        assert ema_state_dict is not None, \
            "[DeepSpeed LOAD MODEL PARAM] Initializing EMA → base_model requires an EMA `state_dict`."
        try:
            base_model: nn.Module = getattr(diffusion_planner, "module",
                                            diffusion_planner)
            incompatible = base_model.load_state_dict(ema_state_dict,
                                                      strict=False)
            if global_rank == 0:
                if getattr(incompatible, "missing_keys", None):
                    print(
                        f"[DeepSpeed LOAD MODEL PARAM] model-only(EMA) missing_keys: {incompatible.missing_keys}"
                    )
                if getattr(incompatible, "unexpected_keys", None):
                    print(
                        f"[DeepSpeed LOAD MODEL PARAM] model-only(EMA) unexpected_keys: {incompatible.unexpected_keys}"
                    )
                print(
                    "[DeepSpeed LOAD MODEL PARAM] model-only resume: base model initialized from EMA weights"
                )
        except Exception as e:
            if global_rank == 0:
                print(f"... 실패: {e}")
            raise RuntimeError(
                "[DeepSpeed LOAD MODEL PARAM] Error occurred during EMA → base_model initialization in model-only mode."
            ) from e
            if global_rank == 0:
                print(
                    f"[DeepSpeed LOAD MODEL PARAM] EMA→base_model 초기화 실패: {e}")

    return model_ema


def _update_deepspeed_optimizer_param_group_meta(
    diffusion_planner: nn.Module,) -> None:
    """DeepSpeed 엔진 내부 옵티마이저 param_group 메타 정보를 보정한다.

    DeepSpeed 모드에서는
      - diffusion_planner.optimizer.param_groups[*]["lr"]
      - diffusion_planner.optimizer.param_groups[*]["weight_decay"]
    값들만 채워져 있는 경우가 있어, 이후 WD warmdown 등의 처리를 위해
    각 그룹에
      - "lr_max": 기준 학습률 (shape: ())
      - "wd_max": 기준 weight decay (shape: ())
    필드를 추가로 넣어 준다.

    Args:
        diffusion_planner (nn.Module):
            - deepspeed.DeepSpeedEngine 인스턴스를 기대한다.
            - diffusion_planner.optimizer.param_groups 는
              각 그룹별로 "params": List[nn.Parameter] 를 가진다.
              각 파라미터 텐서 shape 는 (out_dim, in_dim), (dim,),
              (C_out, C_in, kH, kW) 등 계층 종류에 따라 다양하다.
    """
    if not hasattr(diffusion_planner, "optimizer"):
        return

    for pg in diffusion_planner.optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))


def _resume_from_deepspeed_checkpoint_format(
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    save_path: str,
    global_rank: int,
    load_optimizer_states: bool = True,
    load_lr_scheduler_states: bool = True,
    load_ema_for_model: bool = True,
    load_ema_for_ema_model: bool = True,
) -> Tuple[int, Optional[str], Optional[ModelEma]]:
    """DeepSpeed 형식으로 저장된 체크포인트를 읽어와 학습을 이어갈 준비를 한다.

    Args:
        diffusion_planner (nn.Module): DeepSpeed 엔진 객체.
        model_ema (Optional[ModelEma]): EMA 래퍼. 없으면 None.
        save_path (str): 체크포인트 루트 디렉터리 경로.
        global_rank (int): 전역 rank. 0일 때만 로그를 출력한다.
        load_optimizer_states (bool): 옵티마 상태를 함께 불러올지 여부.
        load_lr_scheduler_states (bool): 스케줄러 상태를 함께 불러올지 여부.

    Returns:
        Tuple[int, Optional[str], Optional[ModelEma]]:
            - init_epoch (int): 재개할 epoch 인덱스(0 기반).
            - wandb_id (Optional[str]): 이어서 사용할 W&B run id 또는 None.
            - model_ema (Optional[ModelEma]): EMA 상태가 반영된 EMA 래퍼 또는 None.
    """
    # 0) tag 자동 선택: save_path 하위 폴더 중 이름에 "latest"가 포함된 디렉터리들 중
    #    정렬 기준 첫 번째 것을 tag로 사용. 없으면 기본값 "latest".
    tag: str = _select_latest_like_tag(save_path)

    if global_rank == 0:
        print(
            f"[DeepSpeed LOAD MODEL PARAM] load_checkpoint: save_path={save_path}, tag='{tag}'"
        )

        # 1) DeepSpeedEngine.load_checkpoint 호출 (신/구 버전 둘 다 지원)
        """
load_path: str
    실제로 로드에 사용된 체크포인트 디렉터리 경로.
    보통 save_path/tag 에 해당하는 문자열 
    (예: /.../training_log/.../latest_epoch-000010)

client_state: Optional[Dict[str, Any]]
    save_checkpoint 때 넘겨준 client_state 그대로 돌아온 dict.
    epoch, loss, wandb_id, ema_state_dict 가 그 안에 들어있음
    ( client_state 내용을 포함한 메타 정보를 별도의 
    체크포인트 파일(예: mp_rank_00_model_states.pt 안의 딕셔너리 키)로 같이 저장해 둔다. )
        """
    load_path, client_state = diffusion_planner.load_checkpoint(
        save_path,
        tag=tag,
        load_optimizer_states=load_optimizer_states,
        load_lr_scheduler_states=load_lr_scheduler_states,
    )

    if global_rank == 0:
        print(
            f"[DeepSpeed LOAD MODEL PARAM] load_checkpoint returned load_path={load_path}, "
            f"client_state keys={list((client_state or {}).keys())}")

    # client_state 가 None 인 경우에도 이후 로직이 동일하게 동작하도록 빈 dict 로 대체
    client_state = client_state or {}
    init_epoch: int = int(client_state.get("epoch", 0))
    wandb_id: Optional[str] = client_state.get("wandb_id", None)

    # EMA 및 (필요 시) base model 복원
    model_ema = _restore_ema_and_model_from_client_state_for_deepspeed(
        diffusion_planner=diffusion_planner,
        model_ema=model_ema,
        client_state=client_state,
        global_rank=global_rank,
        load_ema_for_model=load_ema_for_model,
        load_ema_for_ema_model=load_ema_for_ema_model,
    )

    # DeepSpeed 엔진 내부 옵티마 param_group 메타 보정
    _update_deepspeed_optimizer_param_group_meta(diffusion_planner)

    return init_epoch, wandb_id, model_ema


def _resume_from_checkpoint_with_deepspeed(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    global_rank: int,
    resume_model_only: bool,
) -> Tuple[nn.Module, Optional[ModelEma], int, Optional[str]]:
    """DeepSpeed 모드에서 args.save_path 기준으로 모델/EMA를 불러온다.

    처리 규칙:
      - args.save_path 이 DeepSpeed 전용 디렉터리 구조이면
        _resume_from_deepspeed_checkpoint_format(...) 을 사용한다.
        · resume_model_only=False → optimizer/scheduler 상태까지 함께 복원.
        · resume_model_only=True  → optimizer/scheduler 는 건드리지 않고,
          모델/EMA만 로드한다.
      - args.save_path 이 PyTorch latest.pth 형식이면
        _resume_deepspeed_from_pytorch_checkpoint(...) 를 사용해
        모델/EMA를 불러온다. (optimizer/scheduler는 그대로)

      - resume_model_only=True 인 경우:
        · init_epoch=0, wandb_id=None 으로 돌려준다.
      - resume_model_only=False 인 경우:
        · ckpt 에서 읽은 init_epoch / wandb_id 를 그대로 사용한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace. device 정보 등을 포함한다.
        diffusion_planner (nn.Module):
            deepspeed.DeepSpeedEngine 또는 DDP 래퍼가 씌워진 모델.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 상세 로그를 출력한다.
        resume_model_only (bool):
            True 이면 모델/EMA 파라미터만 불러오는 모드.

    Returns:
        Tuple[nn.Module, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 모델/EMA 로딩이 반영된 모델.
            - model_ema: EMA 상태가 복원된 ModelEma 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스. resume_model_only=True 이면 0.
            - wandb_id: 이어서 사용할 W&B run id 또는 None.
    """
    # ✅ 1. model-only + DeepSpeed인 경우: DeepSpeedEngine.load_checkpoint 안 쓴다
    if resume_model_only:
        model_ema = _load_model_and_ema_state_only_from_deepspeed_checkpoint(
            save_path=args.save_path,
            diffusion_planner=diffusion_planner,
            model_ema=model_ema,
            global_rank=global_rank,
        )
        init_epoch = 0
        wandb_id = None
        return diffusion_planner, model_ema, init_epoch, wandb_id
    elif _is_deepspeed_checkpoint_dir(args.save_path):
        init_epoch_loaded, wandb_id_loaded, model_ema = \
            _resume_from_deepspeed_checkpoint_format(
                diffusion_planner=diffusion_planner,
                model_ema=model_ema,
                save_path=args.save_path,
                global_rank=global_rank,
                load_optimizer_states=True,
                load_lr_scheduler_states=True,
                load_ema_for_model=args.load_ema_for_model,
                load_ema_for_ema_model=args.load_ema_for_ema_model,
            )
    else:
        raise ValueError(
            "The save_path does not contain a valid DeepSpeed checkpoint directory."
        )

    if resume_model_only:
        init_epoch = 0
        wandb_id = None
    else:
        init_epoch = init_epoch_loaded
        wandb_id = wandb_id_loaded

    return diffusion_planner, model_ema, init_epoch, wandb_id


def _select_pytorch_checkpoint_path_for_model_only(ckpt_dir: str) -> str:
    """model-only 로딩에 사용할 latest.pth 또는 best.pth 경로를 고른다.

    우선순위:
      1) ckpt_dir/latest.pth 가 있으면 그 경로를 사용한다.
      2) 없으면 ckpt_dir/best.pth 를 시도한다.
      3) 둘 다 없으면 FileNotFoundError 를 발생시킨다.

    Args:
        ckpt_dir (str):
            PyTorch 체크포인트가 들어 있는 디렉터리 경로.

    Returns:
        str:
            실제로 torch.load 에 사용할 체크포인트 파일 경로.
    """
    ckpt_path = os.path.join(ckpt_dir, "latest.pth")
    if os.path.exists(ckpt_path):
        return ckpt_path

    alt_ckpt_path = os.path.join(ckpt_dir, "best.pth")
    if os.path.exists(alt_ckpt_path):
        return alt_ckpt_path

    raise FileNotFoundError(
        f"PyTorch 체크포인트 latest.pth / best.pth 를 찾을 수 없습니다: {ckpt_dir}")


def _load_pytorch_checkpoint_dict_for_model_only(
    ckpt_path: str,
    device: str,
) -> Dict[str, Any]:
    """model-only 로딩용 PyTorch 체크포인트 파일을 dict 로 읽어온다.

    Args:
        ckpt_path (str):
            latest.pth 또는 best.pth 전체 경로.
        device (str):
            "cuda" / "cuda:0" / "cpu" 등. torch.load map_location 에 사용된다.

    Returns:
        Dict[str, Any]:
            torch.load 로 읽은 체크포인트 dict.
            - ckpt["model"]: 모델 state_dict (각 value 텐서 shape는
              (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW) 등).
            - ckpt["ema_state_dict"] (선택): EMA state_dict.
    """
    map_location = "cpu" if device.startswith("cuda") else device
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=map_location)
    return ckpt


def _choose_state_dict_for_model_only(
    ckpt: Dict[str, Any],) -> Tuple[Dict[str, torch.Tensor], str]:
    """EMA state 가 있으면 EMA를, 없으면 model state 를 선택한다.

    Args:
        ckpt (Dict[str, Any]):
            torch.load 로 읽어온 체크포인트 dict.

    Returns:
        Tuple[Dict[str, torch.Tensor], str]:
            - state_like:
                모델에 실제로 넣을 state_dict.
            - source_name:
                "EMA" 또는 "model" 문자열. 로그용.
    """
    raw_model_state: Dict[str, torch.Tensor] = ckpt.get("model", ckpt)
    raw_ema_state: Optional[Dict[str, torch.Tensor]] = ckpt.get(
        "ema_state_dict", None)

    if raw_ema_state is not None and isinstance(
            raw_ema_state, dict) and len(raw_ema_state) > 0:
        return raw_ema_state, "EMA"
    else:
        return raw_model_state, "model"


def _strip_or_add_module_prefix_for_model_only(
    src_state: Dict[str, torch.Tensor],
    want_module_prefix: bool,
) -> Dict[str, torch.Tensor]:
    """state_dict 키에 "module." 접두사를 붙이거나 떼는 작은 도우미 함수.

    DDP/DeepSpeed 래핑 여부에 따라
    - 모델 쪽 키는 "module.xxx" 또는 "xxx" 형태일 수 있고,
    - 체크포인트 쪽 키도 두 형태 중 하나일 수 있다.
    이 함수는 원하는 형태(want_module_prefix)에 맞춰
    키 문자열만 정리해 새 dict를 만들어 준다.

    Args:
        src_state (Dict[str, torch.Tensor]):
            체크포인트에서 가져온 state_dict.
            각 value 텐서의 shape 는 레이어 종류에 따라
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등 다양할 수 있다.
        want_module_prefix (bool):
            True  이면 모든 키가 "module." 으로 시작하도록 맞춘다.
            False 이면 "module." 으로 시작하는 키는 접두사를 떼고,
            나머지는 그대로 둔다.

    Returns:
        Dict[str, torch.Tensor]:
            키 문자열만 정리된 새 state_dict.
    """
    new_state: Dict[str, torch.Tensor] = {}
    for k, v in src_state.items():
        if want_module_prefix:
            # 이미 "module." 로 시작하면 그대로, 아니면 앞에 붙인다.
            new_k = k if k.startswith("module.") else f"module.{k}"
        else:
            # "module." 로 시작하면 떼고, 아니면 그대로 둔다.
            new_k = k[7:] if k.startswith("module.") else k
        new_state[new_k] = v
    return new_state


def _align_state_dict_keys_to_base_model_for_model_only(
    state_like: Dict[str, torch.Tensor],
    base_model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """base_model.state_dict() 와 키 형태(module. prefix 유무)를 맞춘 state_dict 를 만든다.

    DeepSpeed/DDP 래핑 여부에 따라
      - 현재 모델 키는 "module.xxx" 또는 "xxx" 형태일 수 있고,
      - ckpt 키도 "module.xxx" 또는 "xxx" 형태일 수 있다.
    이 함수는 두 쪽의 키 패턴을 비교해서,
    필요한 경우 "module." 접두사를 붙이거나 떼서
    base_model.load_state_dict(...) 에 바로 넣을 수 있는 state_dict 로 맞춰준다.

    Args:
        state_like (Dict[str, torch.Tensor]):
            ckpt 에서 선택한 state_dict.
            각 value 텐서는 레이어에 따라
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등의 shape 를 가진다.
        base_model (nn.Module):
            실제 파라미터를 가진 nn.Module (DDP 래퍼를 벗긴 모델).

    Returns:
        Dict[str, torch.Tensor]:
            base_model.load_state_dict(...) 에 바로 사용할 수 있는 state_dict.
    """
    current_keys = list(base_model.state_dict().keys())
    ckpt_keys = list(state_like.keys())

    has_module_in_ckpt = all(k.startswith("module.") for k in ckpt_keys)
    has_module_in_model = all(k.startswith("module.") for k in current_keys)

    # ckpt: module.* / 모델: 평범한 키 → prefix 제거
    if has_module_in_ckpt and (not has_module_in_model):
        return _strip_or_add_module_prefix_for_model_only(
            state_like,
            want_module_prefix=False,
        )
    # ckpt: 평범한 키 / 모델: module.* → prefix 추가
    elif (not has_module_in_ckpt) and has_module_in_model:
        return _strip_or_add_module_prefix_for_model_only(
            state_like,
            want_module_prefix=True,
        )
    else:
        # 둘 다 module.* 이거나, 둘 다 평범한 키인 경우 그대로 사용
        return state_like


def _load_state_dict_into_base_and_ema_model_only(
    state_dict: Dict[str, torch.Tensor],
    source_name: str,
    base_model: nn.Module,
    model_ema: Optional[ModelEma],
    ckpt_path: str,
    global_rank: int,
) -> Optional[ModelEma]:
    """선택된 state_dict 를 base_model 과 EMA 모델에 주입하고 로그를 남긴다.

    처리 순서:
      1) base_model.load_state_dict(..., strict=False) 로 파라미터 로드.
      2) global_rank==0 인 경우 missing_keys / unexpected_keys 를 출력.
      3) model_ema 가 있으면 base_model.state_dict() 로 EMA 모델도 초기화하고,
         eval 모드 + requires_grad=False 로 고정한다.

    Args:
        state_dict (Dict[str, torch.Tensor]):
            base_model.load_state_dict 에 넣을 state_dict.
        source_name (str):
            "EMA" 또는 "model". 어떤 state 를 사용했는지 로그용.
        base_model (nn.Module):
            실제 파라미터를 가진 nn.Module.
            base_model.state_dict() 의 각 텐서는
              (out_dim, in_dim), (hidden_dim,),
              (C_out, C_in, kH, kW) 등 shape 를 가진다.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        ckpt_path (str):
            사용한 체크포인트 파일 경로. 로그용.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]:
            EMA 상태가 덮어써진 model_ema (또는 원래 None).
    """
    incompatible = base_model.load_state_dict(state_dict, strict=False)
    if global_rank == 0:
        print(
            f"[ModelOnly<Pytorch>] {source_name} state_dict 로 LOAD MODEL PARAM 완료 "
            f"(ckpt_path={ckpt_path})")
        if getattr(incompatible, "missing_keys", None):
            print(
                f"[ModelOnly<Pytorch>] missing_keys: {incompatible.missing_keys}"
            )
        if getattr(incompatible, "unexpected_keys", None):
            print(
                f"[ModelOnly<Pytorch>] unexpected_keys: {incompatible.unexpected_keys}"
            )

    # EMA 객체도 있으면 base_model 과 동일한 파라미터로 맞춰준다.
    if model_ema is not None:
        try:
            ema_model: nn.Module = getattr(model_ema, "ema", model_ema)
            ema_incompatible = ema_model.load_state_dict(
                base_model.state_dict(),
                strict=False,
            )
            ema_model.eval()
            for p in ema_model.parameters():
                # 각 파라미터 텐서 shape: (out_dim, in_dim), (hidden_dim,), ...
                p.requires_grad_(False)
            if global_rank == 0:
                if getattr(ema_incompatible, "missing_keys", None):
                    print(
                        f"[ModelOnly<Pytorch>] EMA missing_keys: {ema_incompatible.missing_keys}"
                    )
                if getattr(ema_incompatible, "unexpected_keys", None):
                    print(
                        f"[ModelOnly<Pytorch>] EMA unexpected_keys: {ema_incompatible.unexpected_keys}"
                    )
                print("[ModelOnly<Pytorch>] EMA state 동기화 완료")
        except Exception as e:
            if global_rank == 0:
                print(f"[ModelOnly<Pytorch>] EMA state 동기화 실패: {e}")

    return model_ema


def _load_model_and_ema_state_only_from_pytorch_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    model_ema: Optional[ModelEma],
    device: str,
    global_rank: int,
) -> Optional[ModelEma]:
    """PyTorch latest.pth 에서 모델/EMA 파라미터만 불러와 새 학습을 시작할 때 사용한다.

    전체 흐름:
      1) save_path 안에서 latest.pth 또는 best.pth 경로를 고른다.
      2) 선택한 파일을 torch.load 로 읽어 체크포인트 dict 를 얻는다.
      3) EMA state_dict 가 있으면 EMA를, 없으면 model state_dict 를 선택한다.
      4) base_model(state_dict) 의 키 형태와 맞도록 module. prefix 를 조정한다.
      5) base_model.load_state_dict(...) 로 파라미터를 주입하고,
         model_ema 가 있으면 base_model.state_dict() 로 EMA도 초기화한다.

    Args:
        save_path (str):
            latest.pth / best.pth 가 들어 있는 디렉터리 경로.
        diffusion_planner (nn.Module):
            현재 학습에 사용할 모델 또는 DDP 래퍼.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        device (str):
            "cuda" / "cpu" 등. torch.load 의 map_location 에 사용된다.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Optional[ModelEma]:
            EMA 상태가 덮어써진 model_ema (또는 원래 None).
    """
    ckpt_path: str = _select_pytorch_checkpoint_path_for_model_only(save_path)
    ckpt: Dict[str, Any] = _load_pytorch_checkpoint_dict_for_model_only(
        ckpt_path=ckpt_path,
        device=device,
    )

    # base_model: 실제 nn.Module (DDP 래퍼 벗겨진 상태)
    base_model: nn.Module = getattr(diffusion_planner, "module",
                                    diffusion_planner)

    state_like, source_name = _choose_state_dict_for_model_only(ckpt)
    state_dict = _align_state_dict_keys_to_base_model_for_model_only(
        state_like=state_like,
        base_model=base_model,
    )

    model_ema = _load_state_dict_into_base_and_ema_model_only(
        state_dict=state_dict,
        source_name=source_name,
        base_model=base_model,
        model_ema=model_ema,
        ckpt_path=ckpt_path,
        global_rank=global_rank,
    )
    return model_ema


def _resume_from_pytorch_checkpoint(
    save_path: str,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str]]:
    """기존 PyTorch/DDP 형식 latest.pth 를 그대로 불러와서 학습을 재개한다.

    처리 흐름:
      1) diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema =
         resume_model(save_path, ...) 를 호출해
         - 모델 파라미터
         - 옵티마이저 상태
         - 스케줄러 상태
         - EMA 상태
         를 한 번에 복원한다.
      2) optimizer.param_groups[*] 에 대해
         - "lr_max": 기준 학습률 (shape: ())
         - "wd_max": 기준 weight_decay (shape: ())
         를 채워 이후 WD warmdown 에서 사용할 수 있게 한다.

    Args:
        save_path (str):
            - latest.pth 가 들어 있는 PyTorch 체크포인트 디렉터리.
        diffusion_planner (nn.Module):
            - nn.Module 또는 DDP 래핑된 모델.
        optimizer (optim.Optimizer):
            - AdamW/AdamW8bit 옵티마이저.
        scheduler (Any):
            - 학습률 스케줄러 객체.
        model_ema (Optional[ModelEma]):
            - EMA 래퍼 또는 None.
        device (str):
            - "cuda" / "cpu" 등. resume_model 내부에서 map_location 에 사용된다.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 복원된 모델.
            - optimizer: 복원된 옵티마이저.
            - scheduler: 복원된 스케줄러.
            - model_ema: 복원된 EMA 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스(0 기반). shape: ()
            - wandb_id: W&B run id 또는 None.
    """
    (diffusion_planner, optimizer, scheduler, init_epoch, wandb_id,
     model_ema) = resume_model(
         save_path,
         diffusion_planner,
         optimizer,
         scheduler,
         model_ema,
     )

    # resume 후에도 wd_max / lr_max 기본값이 유지되도록 보강
    for pg in optimizer.param_groups:
        pg.setdefault("lr_max", float(pg["lr"]))
        pg.setdefault("wd_max", float(pg.get("weight_decay", 0.0)))

    return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id


def _resume_from_checkpoint_with_pytorch(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    model_ema: Optional[ModelEma],
    save_path: str,
    global_rank: int,
    resume_model_only: bool,
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str]]:
    """일반 PyTorch/DDP 모드에서 save_path 기준으로 모델/옵티마를 불러온다.

    처리 규칙:
      - resume_model_only=True:
          · _load_model_and_ema_state_only_from_pytorch_checkpoint(...) 를 사용해
            모델/EMA 파라미터만 불러온다.
          · optimizer / scheduler 상태는 그대로 둔다.
          · init_epoch=0, wandb_id=None 으로 돌려준다.
      - resume_model_only=False:
          · _resume_from_pytorch_checkpoint(...) 로
            모델/optimizer/scheduler/EMA/epoch/wandb_id 를 모두 복원한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace. device 정보 포함.
        diffusion_planner (nn.Module):
            현재 학습 중인 모델 또는 DDP 래퍼.
        optimizer (optim.Optimizer):
            AdamW 또는 8bit AdamW 옵티마이저.
        scheduler (Any):
            학습률 스케줄러 객체.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        save_path (str):
            PyTorch latest.pth / best.pth 가 들어 있는 디렉터리 경로.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 상세 로그 출력.
        resume_model_only (bool):
            True 이면 모델/EMA 파라미터만 불러오는 모드.

    Returns:
        Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int, Optional[str]]:
            - diffusion_planner: 복원된 모델.
            - optimizer: 복원된(또는 기존) 옵티마이저.
            - scheduler: 복원된(또는 기존) 스케줄러.
            - model_ema: EMA 상태가 복원된 ModelEma 또는 None.
            - init_epoch: 재개 시작 epoch 인덱스. model-only 이면 0.
            - wandb_id: 이어서 사용할 W&B run id 또는 None.
    """
    if resume_model_only:
        # 모델(또는 EMA) weight만 불러오고 optimizer/scheduler 는 그대로 유지
        model_ema = _load_model_and_ema_state_only_from_pytorch_checkpoint(
            save_path=save_path,
            diffusion_planner=diffusion_planner,
            model_ema=model_ema,
            device=args.device,
            global_rank=global_rank,
        )
        init_epoch = 0
        wandb_id = None
        return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id

    # 전체 상태 복원 (기존 PyTorch 경로)
    (diffusion_planner, optimizer, scheduler, model_ema, init_epoch,
     wandb_id) = _resume_from_pytorch_checkpoint(
         save_path=save_path,
         diffusion_planner=diffusion_planner,
         optimizer=optimizer,
         scheduler=scheduler,
         model_ema=model_ema,
     )
    return diffusion_planner, optimizer, scheduler, model_ema, init_epoch, wandb_id


def _finalize_train_epochs_and_wandb_after_resume(
    args: argparse.Namespace,
    init_epoch: int,
    resume_model_only: bool,
    global_rank: int,
) -> Tuple[int, bool]:
    """체크포인트 로드 이후 train_epochs 와 allow_val_change 값을 최종 결정한다.

    처리 규칙:
      - resume_model_only=True:
          · epoch 정보는 모두 무시하고,
            args.train_epochs 를 그대로 사용한다.
          · allow_val_change=False 로 둔다 (새 run 개념).
      - resume_model_only=False:
          · init_epoch 가 이미 지난 epoch 수이므로,
            args.train_epochs <= init_epoch 이면 init_epoch+1 로 늘린다.
          · args.resume_wandb_model_name 가 설정되어 있으면
            allow_val_change=True 로 W&B config 변경을 허용한다.

    Args:
        args (argparse.Namespace):
            학습 설정 Namespace.
        init_epoch (int):
            체크포인트에서 읽어온 재개 시작 epoch 인덱스(0 기반).
        resume_model_only (bool):
            True 이면 model-only 재시작 모드.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 로그를 출력한다.

    Returns:
        Tuple[int, bool]:
            - train_epochs: 최종 사용할 train_epochs 값.
            - allow_val_change: W&B config 변경 허용 여부.
    """
    allow_val_change: bool = False

    if resume_model_only:
        # model-only 재시작: epoch 정보는 모두 무시하고 config 값 그대로 사용
        train_epochs = int(args.train_epochs)
        return train_epochs, allow_val_change

    # 기존 full-resume 경로: init_epoch 를 기준으로 train_epochs 보정
    if args.train_epochs <= init_epoch:
        old = int(args.train_epochs)
        args.train_epochs = int(init_epoch) + 1
        if global_rank == 0:
            print(
                f"[ADJUST] scaled train_epochs ({old}) <= init_epoch ({init_epoch}). "
                f"Bumping to {args.train_epochs}.")
    train_epochs = int(args.train_epochs)

    # W&B 아티팩트에서 재개하는 경우 config 값 변경 허용
    if args.resume_wandb_model_name:
        allow_val_change = True

    return train_epochs, allow_val_change


def maybe_resume_from_checkpoint(
    args: argparse.Namespace,
    diffusion_planner: nn.Module,
    optimizer: Optional[optim.Optimizer],
    scheduler: Optional[Any],
    model_ema: Optional[ModelEma],
    global_rank: int,
    use_deepspeed: bool,
) -> Tuple[nn.Module, optim.Optimizer, Any, Optional[ModelEma], int,
           Optional[str], int, bool]:
    """local 또는 W&B에서 지정한 체크포인트가 있으면 학습 상태를 복원한다.

    이 함수는 "재개할 체크포인트가 있는지"를 먼저 확인한 뒤,
    상황에 따라 다음과 같이 동작한다.

      * resume_model_only=False (기존 resume 모드):
        - DeepSpeed 모드 + DeepSpeed 형식 디렉터리:
            · 모델/옵티마이저/스케줄러/EMA/epoch/wandb_id 를 모두 복원.
        - DeepSpeed 모드 + PyTorch latest.pth:
            · 모델/EMA 를 불러오고, epoch/wandb_id 를 사용해 계속 진행
              (옵티마이저/스케줄러는 새 상태로 시작).
        - 일반 PyTorch/DDP 모드:
            · resume_model(...) 을 통해 모델/옵티마이저/스케줄러/EMA/epoch/wandb_id 복원.

      * resume_model_only=True (model-only 재시작 모드):
        - 어떤 형식이든 모델/EMA 파라미터만 불러오고,
          optimizer/scheduler/epoch/wandb_id 는 모두 새로 시작한다.
          · DeepSpeed: load_optimizer_states=False, load_lr_scheduler_states=False 로 로드.
          · PyTorch: _load_model_and_ema_state_only_from_pytorch_checkpoint(...) 사용.
        - init_epoch=0, train_epochs 는 config 값 그대로 사용.
        - wandb_id=None, allow_val_change=False 로 처리한다.

    Args:
        args (argparse.Namespace):
            학습 설정 및 resume 관련 인자를 담고 있는 Namespace.
        diffusion_planner (nn.Module):
            현재 학습 중인 모델 또는 DeepSpeed/ DDP 래퍼.
        optimizer (optim.Optimizer):
            AdamW 또는 8bit AdamW 옵티마이저.
        scheduler (Any):
            학습률 스케줄러 인스턴스.
        model_ema (Optional[ModelEma]):
            EMA 래퍼 또는 None.
        global_rank (int):
            전체 프로세스 기준 rank. 0 일 때만 주요 로그를 출력한다.
        use_deepspeed (bool):
            DeepSpeed 모드 사용 여부.

    """

    if args.resume_wandb_model_name is not None:
        print(f"[LOAD MODEL PARAM] Model loaded from {args.save_path}")

        # -------- DeepSpeed 경로 --------
        if use_deepspeed and (args.resume_model_only or
                              hasattr(diffusion_planner, "load_checkpoint")):
            diffusion_planner, model_ema, init_epoch, wandb_id = \
                _resume_from_checkpoint_with_deepspeed(
                    args=args,
                    diffusion_planner=diffusion_planner,
                    model_ema=model_ema,
                    global_rank=global_rank,
                    resume_model_only=args.resume_model_only,
                )

        # -------- 일반 PyTorch / DDP 경로 --------
        else:
            assert optimizer is not None and scheduler is not None, \
                "optimizer 및 scheduler 는 None 이 될 수 없습니다."
            (diffusion_planner, optimizer, scheduler, model_ema, init_epoch,
             wandb_id) = _resume_from_checkpoint_with_pytorch(
                 args=args,
                 diffusion_planner=diffusion_planner,
                 optimizer=optimizer,
                 scheduler=scheduler,
                 model_ema=model_ema,
                 save_path=args.save_path,
                 global_rank=global_rank,
                 resume_model_only=args.resume_model_only,
             )

        # -------- train_epochs / allow_val_change 최종 결정 --------
        train_epochs, allow_val_change = _finalize_train_epochs_and_wandb_after_resume(
            args=args,
            init_epoch=init_epoch,
            resume_model_only=args.resume_model_only,
            global_rank=global_rank,
        )

    else:
        # 체크포인트 지정이 전혀 없는 경우: 완전 새 학습
        init_epoch = 0
        wandb_id = None
        train_epochs = int(args.train_epochs)
        allow_val_change = False

    return (
        diffusion_planner,
        optimizer,
        scheduler,
        model_ema,
        init_epoch,
        wandb_id,
        train_epochs,
        allow_val_change,
    )


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


def _does_file_exist(file_path: str) -> bool:
    """주어진 경로에 '파일'이 실제로 존재하는지 확인합니다.

    폴더(디렉터리)가 아니라 "파일"인지까지 확인해서,
    같은 이름의 폴더가 우연히 있을 때를 안전하게 피합니다.

    Args:
        file_path (str): 확인할 파일 경로. shape: ()

    Returns:
        bool:
            - True: file_path가 존재하고, 그 대상이 파일임
            - False: 존재하지 않거나, 파일이 아님
    """
    if not isinstance(file_path, str) or not file_path:
        return False
    return os.path.isfile(file_path)


def _find_file_name_of_latest_or_best_dir(
    save_path: str,
    checkpoint_filename: str,
) -> Optional[Tuple[str, str]]:
    """save_path 안에서 'latest' 또는 'best' 글자가 들어간 하위 폴더를 찾아,
    그 폴더 안에 checkpoint_filename 파일이 있으면 그 파일 경로를 돌려줍니다.

    이 함수는 DeepSpeed를 쓸 때,
    "추가로 생기는 체크포인트 폴더(예: latest_epoch-000001 같은 폴더)가
    이미 존재하는지"를 확인하는 용도로 사용합니다.

    Args:
        save_path (str): 상위 저장 폴더 경로. shape: ()
        checkpoint_filename (str): 하위 폴더 안에서 찾을 파일 이름. shape: ()

    Returns:
        Optional[str]:
            - 찾으면: 파일의 전체 경로(str)
            - 못 찾으면: None
    """
    if not isinstance(save_path, str) or not save_path:
        return None
    if not os.path.isdir(save_path):
        return None
    if not isinstance(checkpoint_filename, str) or not checkpoint_filename:
        return None

    try:
        entries = os.listdir(save_path)
    except OSError:
        return None

    for entry in entries:
        # entry_path: './training_log/.../2025-12-06-06:56:58/latest_epoch-000001'
        # entry: 'latest_epoch-000001', 'best_epoch-000010', ...
        entry_path = os.path.join(save_path, entry)
        if not os.path.isdir(entry_path):
            continue
        """
        entry_path 에 .pt 확장자의 파일이 1개라도 있는지 확인합니다.(이름 상관없음) 없으면, 다음 항목으로 넘어갑니다.
        """
        # entry_path 안에 있는 파일 목록을 얻습니다.
        try:
            sub_entries = os.listdir(entry_path)
        except OSError:
            continue
        has_pt_file = any(
            sub_entry.lower().endswith(".pt") for sub_entry in sub_entries)
        if not has_pt_file:
            continue

        entry_lower = entry.lower()
        if ("latest" not in entry_lower) and ("best" not in entry_lower):
            continue
        return entry_path, entry

    return None


def _should_skip_wandb_checkpoint_download(
        args: argparse.Namespace,
        wandb_new_folder_name: str,
        target_local_ckpt_path:
    str,  # ./training_log/.../2025-12-06-06:56:58/latest.pth
        checkpoint_filename: str,  # 'latest.pth'
) -> Tuple[bool, str]:
    """로컬에 이미 필요한 체크포인트가 갖춰져 있으면 W&B 다운로드를 생략할지 판단합니다.

    규칙
    ----
    1) args.use_deepspeed == True 인 경우 (AND 조건, 둘 다 만족해야 생략)
       - (조건 1) target_local_ckpt_path 파일이 이미 존재
       - (조건 2) args.save_path 아래에 이름에 "latest" 또는 "best" 가 들어간 폴더가 있고,
                 그 폴더 안에 checkpoint_filename 파일이 존재

    2) args.use_deepspeed == False 인 경우
       - target_local_ckpt_path 파일이 이미 존재하면 생략

    Args:
        args (argparse.Namespace):
            - use_deepspeed (bool): DeepSpeed 사용 여부
            - save_path (Optional[str]): 체크포인트 저장 상위 폴더 경로
        target_local_ckpt_path (str): 최종 체크포인트 파일 경로. shape: ()
        checkpoint_filename (str): 체크포인트 파일 이름. shape: ()

    Returns:
        Tuple[bool, str]:
            - should_skip (bool): True면 다운로드 생략
            - reason (str): 로그에 찍기 위한 간단한 이유 설명
    """
    has_main_ckpt: bool = _does_file_exist(target_local_ckpt_path)
    use_deepspeed: bool = bool(getattr(args, "use_deepspeed", False))

    if not use_deepspeed:
        if has_main_ckpt:
            return True, "DeepSpeed 미사용: target_local_ckpt_path 파일이 이미 존재합니다."
        return False, "DeepSpeed 미사용: target_local_ckpt_path 파일이 없습니다."

    save_path = getattr(args, "save_path", None)
    if not isinstance(save_path, str) or not save_path:
        return False, "DeepSpeed 사용: (조건 2) args.save_path가 비어 있어 하위 폴더를 확인할 수 없습니다."

    output_: Optional[str] = _find_file_name_of_latest_or_best_dir(
        save_path=save_path,
        checkpoint_filename=checkpoint_filename,
    )
    if output_ is None:
        return False, (
            "DeepSpeed 사용: (조건 2) save_path 아래 'latest/best' 이름을 가지면서,"
            "안에 .pt 파일이 있는 폴더를 찾지 못했습니다.")
    # DeepSpeed 사용 시: AND 조건
    if not has_main_ckpt:
        return False, "DeepSpeed 사용: (조건 1) target_local_ckpt_path 파일이 없습니다."

    # found_dir_path: './training_log/.../2025-12-06-06:56:58/latest_epoch-000001'
    # found_in_tag_dir: 'latest_epoch-000001' 또는 'best_epoch-000010' 등
    # wandb_new_folder_name: 'latest_epoch-000220' 등
    found_dir_path, found_in_tag_dir = output_
    if wandb_new_folder_name != found_in_tag_dir:
        """ 아래 코드 내용
        found_dir_path 을 삭제한다.
        save_path 에 있는 checkpoint_filename 도 삭제한다.
        """
        shutil.rmtree(found_dir_path, ignore_errors=True)
        print(
            f"[CLEANUP] Removed mismatched DeepSpeed checkpoint dir: {found_dir_path}"
        )
        main_ckpt_path = os.path.join(save_path, checkpoint_filename)
        try:
            if _does_file_exist(main_ckpt_path):
                os.remove(main_ckpt_path)
                print(
                    f"[CLEANUP] Removed mismatched main checkpoint file: {main_ckpt_path}"
                )
        except OSError:
            pass
        return False, (
            "DeepSpeed 사용: (조건 2) W&B에서 찾은 폴더 이름과 실제 존재하는 폴더 이름이 다릅니다. "
            f"(wandb: {wandb_new_folder_name}, found: {found_in_tag_dir})")

    return True, (
        "DeepSpeed 사용: (조건 1) 메인 파일 + (조건 2) 'latest/best' 폴더 안 파일 "
        f"둘 다 확인했습니다.  ( at wandb: {wandb_new_folder_name},  found: {found_in_tag_dir})"
    )


def _extract_first_pt_parent_dir(file_names: list[str]) -> str:
    """파일 목록에서 첫 번째 .pt 파일의 '상위 폴더 이름'을 뽑습니다.

    동작 방식
    ----------
    1) file_names를 앞에서부터 훑습니다.
    2) ".pt"로 끝나는 항목을 처음 발견하면,
       그 경로에서 마지막 "/" 앞의 폴더 경로를 가져옵니다.
       예) "latest_epoch-000220/mp_rank_00_model_states.pt" -> "latest_epoch-000220"
    3) 상위 폴더가 없으면(예: "latest.pth" 같이 루트에 있는 경우) 에러를 냅니다.

    Args:
        file_names (list[str]):
            artifact.files()에서 얻은 파일 경로 리스트. length: (K,)

    Returns:
        str:
            첫 번째 .pt 파일의 상위 폴더 이름. shape: ()

    Raises:
        ValueError:
            - .pt 파일이 없거나
            - .pt 파일이 루트에만 있어서 상위 폴더를 못 뽑는 경우
    """
    for name in file_names:
        if not isinstance(name, str):
            continue
        if not name.endswith(".pt"):
            continue

        parent = name.rsplit("/", 1)[0]  # "a/b.pt" -> "a"
        if parent == name:
            raise ValueError(f".pt 파일이 루트에만 있습니다: {name}")
        return parent

    raise ValueError(".pt 파일을 찾지 못했습니다.")


def _download_wandb_checkpoint_to_local(
        args: argparse.Namespace,
        api: wandb.Api,
        entity: str,  # 'jksg01019-naver-labs'
        project: str,  # 'Diffusion-Planner'
        collection_name: str,  # W&B 상에서 모델 묶음 이름. 예: f"{args.name}_latest-model".
        resume_alias: str,  # 'latest'
        checkpoint_filename: str,  # 'latest.pth'
) -> bool:
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
        - W&B 상에서 모델 묶음 이름. 예: f"{args.load_name}_latest-model".
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
    file_names = [f.name for f in artifact.files()]
    # wandb_new_folder_name: 'latest_epoch-000220' 등
    wandb_new_folder_name = _extract_first_pt_parent_dir(file_names)

    if "save_path" not in source_run.config:
        raise ValueError("NO save path in WANDB run config")

    # "./training_log/feasible_full_time_use_vel_gpu_2_exp_A/2025-12-06-06:56:58/"
    past_save_path = source_run.config["save_path"]
    print(
        f"[WANDB->local] find save_path in WANDB run config past_save_path: {past_save_path}"
    )
    experiment_is_same = (args.name == args.load_name)
    if args.save_path is None:  #
        assert experiment_is_same, (
            "args.save_path None -> , args.name과 args.load_name should be same."
        )
        assert args.resume_wandb_model_name is not None, (
            "args.resume_wandb_model_name must be set when resuming the same experiment."
        )
        args.save_path = past_save_path
    """
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """

    # target_local_ckpt_path: ./training_log/.../2025-12-06-06:56:58/latest.pth
    target_local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
    """
    args.use_deepspeed일 떄, 다운로드 생략 조건 (AND 조건임. 둘 다 만족해야함)
        1) target_local_ckpt_path 가 이미 존재하는 경우
        2) args.save_path 에 "latest"나 "best" 글자가 들어가있는 디렉터리가 이미 존재하고, 그 안에 checkpoint_filename 파일이 존재하는 경우 (pt)
    not args.use_deepspeed일 떄, 다운로드 생략 조건
        1) target_local_ckpt_path 가 이미 존재하는 경우
    """
    # =========================
    # ✅ 다운로드 생략 조건 구현
    # =========================
    should_skip, skip_reason = _should_skip_wandb_checkpoint_download(
        args=args,
        wandb_new_folder_name=wandb_new_folder_name,
        target_local_ckpt_path=
        target_local_ckpt_path,  # ./training_log/.../2025-12-06-06:56:58/latest.pth
        checkpoint_filename=checkpoint_filename,  # 'latest.pth'
    )
    if should_skip:
        print(f"[WANDB->local] skip download. rank={rank}, "
              f"target_local_ckpt_path='{target_local_ckpt_path}', "
              f"checkpoint_filename='{checkpoint_filename}'. "
              f"reason: {skip_reason}")
        if args.finish_when_no_updated_pt:
            print("[EXIT] finish_when_no_updated_pt is set. exit now.")

            return True
        return False
    if rank == 0:
        os.makedirs(past_save_path, exist_ok=True)
        artifacts_root = os.path.join(past_save_path, "artifacts")
        # 이미 artifacts_root 가 있으면, 지우고 다시 받음
        if os.path.exists(artifacts_root):
            print(
                f"[WANDB->local] artifacts_root '{artifacts_root}' already exist so delete and re download ."
            )
            shutil.rmtree(artifacts_root)
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
            len_candidate_dirs = len(candidate_dirs)
            assert len_candidate_dirs == 1, (
                f"[local->local] expect only one candidate dir for tag '{tag}', "
                f"but found {len_candidate_dirs} dirs: {candidate_dirs}")

            # 첫 번째(정렬 기준) 후보를 최종 tag 디렉터리 이름으로 사용
            chosen_dir_name = sorted(candidate_dirs)[0]

            artifact_tag_dir_past = os.path.join(artifact_dir_past,
                                                 chosen_dir_name)
            save_path_tag_dir = os.path.join(args.save_path, chosen_dir_name)

            if os.path.isdir(save_path_tag_dir):
                shutil.rmtree(save_path_tag_dir)
            shutil.copytree(artifact_tag_dir_past, save_path_tag_dir)
            print(
                "[local->local] Copy DeepSpeed checkpoint directory: "
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
    return False

def _determine_wandb_artifact_config(
    args: argparse.Namespace,) -> Tuple[str, str, str]:
    """재개하려는 W&B 아티팩트의 이름과 파일 이름을 결정한다.

    Args:
        args: argparse.Namespace
            - args.resume_wandb_model_name (str): 'latest' 또는 'best' 등을 기대.
            - args.load_name (str): 실험 이름. 컬렉션 이름의 접두사로 사용된다.

    Returns:
        resume_alias: str
            - 실제로 사용할 별칭. 예: 'latest', 'best'.
        collection_name: str
            - W&B 상에서 모델 묶음 이름. 예: f"{args.load_name}_latest-model".
        checkpoint_filename: str
            - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    resume_alias = args.resume_wandb_model_name
    if resume_alias == 'best':
        collection_name = f"{args.load_name}_best-model"
        checkpoint_filename = "best.pth"
    else:  # 'latest' 또는 'v10'과 같은 특정 버전을 처리합니다.
        collection_name = f"{args.load_name}_latest-model"
        checkpoint_filename = "latest.pth"
    return resume_alias, collection_name, checkpoint_filename


def _validate_wandb_resume_args(args: argparse.Namespace) -> None:
    """W&B에서 체크포인트를 받아올 때 필요한 기본 인자들을 점검한다.

    처리 내용:
      - args.resume_wandb_model_name 가 'latest' 인지 확인한다.
      - args.load_name 이 비어 있지 않은지 확인한다.
    둘 중 하나라도 조건을 만족하지 않으면 바로 ValueError 를 던져서
    잘못된 설정으로 내려받기를 시도하지 않도록 막는다.
    """
    # resume_wandb_model_name 가 "latest"가 아니면 에러를 발생시킵니다.
    if args.resume_wandb_model_name not in ['latest']:
        raise ValueError("args.resume_wandb_model_name must be 'latest.")
    if not args.load_name:
        raise ValueError(
            "args.load_name must be provided to resume from a wandb artifact.")


def prepare_wandb_resume(args: argparse.Namespace,) -> bool:
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

    if not args.resume_wandb_model_name:  # "latest"
        return False

    _validate_wandb_resume_args(args)
    print(
        f"Resuming from wandb artifact: {args.load_name}:{args.resume_wandb_model_name}"
    )
    """
    resume_alias : str
        - 실제로 사용할 별칭. 예: 'latest', 'best'.
    collection_name : str
        - W&B 상에서 모델 묶음 이름. 예: f"{args.load_name}_latest-model".
    checkpoint_filename : str
        - local에 내려 받을 파일 이름. 예: 'latest.pth', 'best.pth'.
    """
    api = wandb.Api()
    resume_alias, collection_name, checkpoint_filename = \
        _determine_wandb_artifact_config(args)
    if args.save_path is not None:
        local_ckpt_path = os.path.join(args.save_path, checkpoint_filename)
        if not args.use_deepspeed and os.path.exists(local_ckpt_path):
            print(
                f"[WANDB->local] found existing local checkpoint: {local_ckpt_path} "
                f"(alias={resume_alias}), skip wandb download.")
            if args.finish_when_no_updated_pt:
                print("[EXIT] finish_when_no_updated_pt is set. exit now.")
                return True
            return False
    """
    entity: str 예: 'jksg01019-naver-labs'
    project: str 예: 'Diffusion-Planner'
    """
    try:
        entity = args.entity
        project = args.project
        """ 수동으로 다운로드 받는 방법
        args.name을 args.load_name 과 똑같이 설정한 후, 
            training_log/args.load_name~~~~/ 디렉터리에
                latest.pth 를 다운받으면 된다. (not args.use_deepspeed 일 때)
                latest_epoch-0000xx/ 디렉터리도 같이 복사해야 한다. (args.use_deepspeed 일 때)
        """
        should_finish = _download_wandb_checkpoint_to_local(
            args,
            api=api,
            entity=entity,
            project=project,
            collection_name=collection_name,
            resume_alias=resume_alias,
            checkpoint_filename=checkpoint_filename,
        )
        if should_finish:
            # finish code right now.
            SystemExit(int(0))


        return should_finish
    except Exception as e:
        raise RuntimeError(f"W&B 아티팩트에서 체크포인트를 내려받는 중 오류가 발생했습니다: {e}") from e


def set_save_path(args: argparse.Namespace, global_rank: int) -> None:
    """실험 이름과 past_name을 기준으로 save_path를 만들고, 모든 rank에 공유한다.

    동작 요약:
      * (stage1 처럼) 새로운 실험(name != load_name)인 경우
        - rank 0에서만 새로운 디렉터리를 만들고
          예: {args.save_dir}/training_log/{args.name}/{YYYY-MM-DD-HH:MM:SS}/
        - 그 경로를 torch.distributed.broadcast_object_list 로 모든 rank에 전파한다.
      * (같은 실험을 이어서 학습하는) name == load_name 인 경우
        - 여기서는 save_path를 정하지 않고(None 유지)
        - 이후 W&B 아티팩트에서 가져온 past_save_path를 args.save_path에 채운다.
      * 분산이 초기화되어 있지 않은(single GPU / single process) 경우
        - rank 0 로직만 실행되고, broadcast는 생략된다.

    Args:
        args (argparse.Namespace):
            학습 및 실험 설정이 들어 있는 argparse 인자 모음.
            - name (str): 현재 stage에서 사용할 실험 이름.
            - load_name (str): 이전 stage 실험 이름. 다른 stage의 weight를 재사용할 때 사용.
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
    load_name: str = getattr(args, "load_name", "")
    experiment_is_same: bool = bool(load_name) and (args.name == load_name)

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
    set_save_path() 안의 broadcast가 항상 동작하도록 만든다.
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


def build_deepspeed_inference_config(
        args: argparse.Namespace) -> Dict[str, Any]:
    """추론용 DeepSpeed 설정 dict를 만든다(학습용 옵션은 제외)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    world_size = max(1, world_size)

    return {
        "dtype": torch.bfloat16,  # 필요하면 torch.float16 로 바꿔도 됨
        "tensor_parallel": {
            "tp_size": world_size
        },  # 여러 GPU에 모델을 나눠 올릴 때
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


class NoPaddingDistributedEvalSampler(torch.utils.data.Sampler[int]):
    """평가용으로 "패딩/드랍 없이" 데이터 인덱스를 rank별로 나눠주는 도구입니다.

    왜 필요한가
    ----------
    PyTorch 기본 DistributedSampler는 데이터 개수가 world_size로 딱 나누어 떨어지지 않으면
    - (옵션에 따라) 끝을 잘라버리거나(drop),
    - 또는 몇 개를 복제해서 길이를 맞추는(padding) 동작을 할 수 있습니다.

    평가/추론에서는 샘플이 "하나도 빠지지 않고, 중복도 없이" 정확히 1번씩 처리되어야 하는 경우가 많습니다.
    그래서 이 클래스는 아래 규칙으로만 인덱스를 나눕니다.

    규칙
    ----
    - 전체 인덱스가 [0, 1, 2, ..., N-1]일 때,
      rank r 는 [r, r+world_size, r+2*world_size, ...] 만 처리합니다.
      즉, 각 rank는 `indices[rank::world_size]`만 처리합니다.
    - 추가로 붙이는 인덱스(복제)도 없고, 잘라내는 것도 없습니다.

    반환/shape
    ---------
    - __iter__가 만들어내는 인덱스 개수는 rank마다 최대 1개 차이입니다.
    - 각 rank의 인덱스 리스트 길이: (N_rank,)  # shape: (N_rank,)

    Args:
        dataset: __len__을 지원하는 Dataset. 전체 길이 N을 가진다고 가정. shape: ()
        num_replicas: 전체 프로세스 수(world_size). shape: ()
        rank: 현재 프로세스의 global rank. shape: ()
        shuffle: True면 epoch마다 같은 규칙으로 섞되, 그래도 "패딩/드랍"은 하지 않습니다.
        seed: shuffle=True일 때 섞기 시드. shape: ()

    Note:
        평가에서는 보통 shuffle=False를 권장합니다.
    """

    def __init__(
        self,
        dataset: Any,
        num_replicas: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.num_replicas = int(max(1, num_replicas))
        self.rank = int(rank)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if self.rank < 0 or self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, num_replicas-1]. got rank={self.rank}, num_replicas={self.num_replicas}"
            )

    def set_epoch(self, epoch: int) -> None:
        """epoch 값을 저장합니다.

        목적
        ----
        - shuffle=True인 경우에만 epoch에 따라 섞이는 순서를 바꾸기 위해 사용합니다.
        - shuffle=False면 호출되어도 동작에 영향은 없습니다.

        Args:
            epoch (int): 현재 epoch 번호. shape: ()
        """
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        """현재 rank가 처리할 인덱스를 순서대로 반환합니다.

        Returns:
            Iterator[int]:
                - 이 rank가 처리할 데이터 인덱스 흐름
                - 총 개수: __len__()와 동일
        """
        dataset_size: int = int(len(self.dataset))
        if dataset_size <= 0:
            return iter(())

        # shuffle을 쓰지 않으면, 메모리 추가 없이 range로 바로 처리 가능
        if not self.shuffle:
            # indices: rank, rank+world_size, rank+2*world_size, ...
            return iter(range(self.rank, dataset_size, self.num_replicas))

        # shuffle=True: 전체 인덱스를 한 번 섞고, 그 중에서 rank::world_size만 선택
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        # perm: (N,) 형태의 섞인 인덱스
        perm = torch.randperm(dataset_size, generator=generator).tolist()
        # 이 rank 몫만 슬라이스 (padding/드랍 없음)
        per_rank_indices = perm[self.rank:dataset_size:self.num_replicas]
        return iter(per_rank_indices)

    def __len__(self) -> int:
        """현재 rank가 처리할 샘플 개수를 반환합니다.

        Returns:
            int:
                - 이 rank가 처리할 샘플 개수. shape: ()
        """
        dataset_size: int = int(len(self.dataset))
        if dataset_size <= self.rank:
            return 0
        # range(rank, dataset_size, world_size)의 길이
        return int((dataset_size - self.rank + self.num_replicas - 1) //
                   self.num_replicas)


def effective_global_batch(batch_size: int, world_size: int) -> int:
    """DataLoader가 사용하는 실제 글로벌 배치(DDP 정합)"""
    world_size = max(1, int(world_size))
    per_rank = max(1, batch_size // world_size)
    return per_rank * world_size  # drop_last 정합 반영


def build_dataset_and_sampler(
    args: argparse.Namespace,
    set_: str,
    set_list: str,
    eval_method: str,
    world_size: int,
    global_rank: int,
) -> Tuple[DiffusionPlannerData, torch.utils.data.Sampler[int]]:
    """Dataset과 Sampler를 만든다.

    변경 목표(평가 시)
    ----------------
    - 평가(eval_method이 train이 아닐 때)는
      1) 샘플 중복 없이
      2) 샘플 누락 없이
      각 rank가 자기 몫만 처리하도록 Sampler를 바꾼다.
      (indices[rank::world_size] 방식)

    학습 시
    ------
    - 기존 로직(DistributedSampler + shuffle)을 그대로 유지한다.

    Args:
        args: 학습/평가 설정 Namespace.
        set_: 데이터 루트 경로. shape: ()
        set_list: 파일 리스트(json) 경로. shape: ()
        eval_method: "train" 또는 "validation" 등. shape: ()
        world_size: 전체 프로세스 수. shape: ()
        global_rank: 현재 프로세스의 global rank. shape: ()

    Returns:
        Tuple[DiffusionPlannerData, torch.utils.data.Sampler[int]]:
            - data_set: DiffusionPlannerData
            - data_sampler:
                · train: torch.utils.data.DistributedSampler
                · eval : NoPaddingDistributedEvalSampler
    """
    data_set = DiffusionPlannerData(
        set_,
        set_list,
        args.predicted_neighbor_num,
        eval_method,
        args.use_data_percent,
    )

    eval_method_lower = str(eval_method).lower()

    if eval_method_lower == "train":
        data_sampler = DistributedSampler(
            data_set,
            num_replicas=int(max(1, world_size)),
            rank=int(global_rank),
            shuffle=True,
            seed=int(args.seed),
        )
        return data_set, data_sampler

    # ✅ eval/validation/test: 패딩/드랍 없는 방식
    else:
        data_sampler = NoPaddingDistributedEvalSampler(
            dataset=data_set,
            num_replicas=int(max(1, world_size)),
            rank=int(global_rank),
            shuffle=False,  # 평가에서는 보통 고정 순서 권장
            seed=int(args.seed),
        )
        return data_set, data_sampler
    raise ValueError(
        f"Unsupported eval_method for building dataset and sampler: eval_method='{eval_method}'"
    )


def build_data_loader(
    args: argparse.Namespace,
    data_set: DiffusionPlannerData,
    data_sampler: torch.utils.data.Sampler[int],
    batch_size: int,
    world_size: int,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """분산 환경에서 DataLoader를 만든다.

    변경 목표(평가 시)
    ----------------
    - eval/validation/test에서는 drop_last=False로 두어
      마지막 남은 샘플이 배치 크기보다 작아도 버리지 않게 한다.

    동작 규칙
    --------
    - drop_last 인자를 명시하면 그 값을 그대로 쓴다.
    - drop_last=None이면:
        · NoPaddingDistributedEvalSampler 사용 시: drop_last=False
        · 그 외(학습): drop_last=True

    Args:
        args: 설정 Namespace.
        data_set: Dataset.
        data_sampler: Sampler.
        batch_size: 전체(global) 배치 크기. shape: ()
        world_size: 전체 프로세스 수. shape: ()
        drop_last: 마지막 배치를 버릴지 여부. shape: ()

    Returns:
        DataLoader:
            각 step마다 배치 dict를 반환.
            배치 텐서는 보통 (B, ...) shape.
    """
    world_size_i = int(max(1, world_size))
    batch_size_per_rank = int(batch_size // world_size_i)

    # 안전장치(원래 코드 구조 유지하면서 0 배치만 방지)
    if batch_size_per_rank <= 0:
        batch_size_per_rank = 1

    if drop_last is None:
        # ✅ eval sampler면 drop_last=False, train이면 drop_last=True
        drop_last = not isinstance(data_sampler,
                                   NoPaddingDistributedEvalSampler)

    loader = DataLoader(
        data_set,
        sampler=data_sampler,
        batch_size=batch_size_per_rank,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_mem,
        persistent_workers=True,
        drop_last=bool(drop_last),
        collate_fn=DiffusionPlannerCollate(args),
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
