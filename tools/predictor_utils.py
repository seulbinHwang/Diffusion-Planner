import argparse
from diffusion_planner.utils.dataset import DiffusionPlannerData
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
from requests.exceptions import HTTPError
from tools.diffusion_planner_collate import DiffusionPlannerCollate
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
                raise SystemExit(int(0))
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
            print("[EXIT] finish_when_no_updated_pt is set. ")
            # finish code right now.
            raise SystemExit(int(0))
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
        # "tensor_parallel": {
        #     "tp_size": world_size
        # },  # 여러 GPU에 모델을 나눠 올릴 때
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

