import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json
from pickle import UnpicklingError
from typing import Dict, Tuple, Optional, List
from typing import Dict, Tuple, Optional, List, Any


def openjson(path):
    value = fileio.get_text(path)
    dict = json.loads(value)
    return dict


def opendata(path):

    npz_bytes = fileio.get(path)
    buff = io.BytesIO(npz_bytes)
    try:
        npz_data = np.load(buff, allow_pickle=True)

        return npz_data
    except (EOFError, UnpicklingError):
        print(f"[경고] 잘못된 파일 건너뜁니다: {path}")
        # 샘플 스킵용 빈 배열을 리턴하거나,
        # raise IndexError로 DataLoader가 다음 샘플로 넘어가게 할 수 있음
        return None


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_epoch_mean_loss(
        epoch_loss_dict_list: List[Dict[str,
                                        torch.Tensor]]) -> Dict[str, float]:
    """한 에폭 동안 step별 loss 딕셔너리 리스트에서, key별 평균 값을 계산한다.

    사용 예:
        - train_epoch 안에서 매 step마다 loss 딕셔너리(예: {"loss": tensor(0.5), "integration_loss": tensor(0.1), ...})
          를 epoch_loss_dict_list 리스트에 append 해 놓고,
        - 에폭 끝에서 이 함수를 호출하면 각 key마다 스칼라 평균값을 돌려준다.

    Args:
        epoch_loss_dict_list (List[Dict[str, torch.Tensor]]):
            - 길이: num_steps (에폭 내 step 수).
            - 각 원소: step별 loss 딕셔너리.
              · 예: {"loss": tensor(...), "neighbor_prediction_loss": tensor(...), ...}
              · 각 value 는 보통 shape=() 인 스칼라 텐서이지만,
                int/float 형태로 들어와도 처리 가능하다.

    Returns:
        Dict[str, float]:
            - key: loss 이름 (예: "loss", "neighbor_prediction_loss", "integration_loss_xy" 등)
            - value: 한 에폭 동안 해당 key에 대한 평균 스칼라 값.

    Note:
        - 내부에서 key별로 [step0, step1, ...] 리스트를 만든 뒤,
          np.mean(np.array(values)) 로 평균을 구한다.
        - 텐서는 value.item() 으로 float 로 변환해서 저장한다.
    """
    epoch_mean_loss = {}
    for current_loss in epoch_loss_dict_list:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(
                    value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [
                    value if isinstance(value, (int, float)) else value.item()
                ]

    for key, values in epoch_mean_loss.items():
        # values: List[float]  → np.array(values).shape = (num_steps,)
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss


def save_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    save_path: str,
    epoch: int,
    train_loss: float,
    wandb_id: Optional[str],
    ema: Optional[Any],
    save_best: bool,
) -> None:
    """모델/옵티마이저/스케줄러/EMA 상태를 하나의 체크포인트로 저장한다.

    이 함수는 PyTorch 스타일의 단일 .pth 파일(latest.pth / best.pth)을 만든다.
    DeepSpeed 를 쓰지 않는 일반 PyTorch / DDP 학습에서 사용하는 저장 경로이다.

    저장되는 내용(save_model dict):
      - "epoch": int
          · 다음 학습에 사용할 epoch 인덱스 (현재 epoch + 1).  # shape: ()
      - "model": Dict[str, torch.Tensor]
          · model.state_dict() 결과.
          · key: 파라미터 이름, value: 파라미터 텐서.
            예) Linear weight: (out_dim, in_dim), LayerNorm weight: (hidden_dim,),
                Conv weight: (C_out, C_in, kH, kW) 등.
      - "ema_state_dict": Optional[Dict[str, torch.Tensor]]
          · ema 가 있으면 ema.state_dict() 결과.
          · value 텐서 shape 는 원본 모델 파라미터와 동일.
      - "optimizer": 옵티마이저 state_dict
          · torch.optim.AdamW / AdamW8bit 상태가 그대로 들어간다.
      - "schedule": 스케줄러 state_dict
          · step, last_epoch, lr 배열 등 스케줄러 내부 상태.
      - "loss": float
          · 이번 epoch 의 train_loss 스칼라 값.  # shape: ()
      - "wandb_id": Optional[str]
          · 이어서 사용할 W&B run id. 없으면 None.

    파일 저장 규칙:
      1) 항상 save_path/latest.pth 를 덮어쓴다.
      2) save_best=True 이면 save_path/best.pth 도 같이 덮어쓴다.

    Args:
        model (torch.nn.Module):
            - 학습 중인 모델 또는 DDP 래퍼(DDP일 경우 내부 .module 의 state_dict가 저장됨).
        optimizer (torch.optim.Optimizer):
            - AdamW/AdamW8bit 등의 옵티마이저 인스턴스.
        scheduler (Any):
            - torch.optim.lr_scheduler._LRScheduler 와 유사한 객체.
            - scheduler.state_dict() 가 가능한 타입이어야 한다.
        save_path (str):
            - 체크포인트를 저장할 디렉터리 경로.
            - latest.pth / best.pth 파일이 이 경로 아래에 생성된다.
        epoch (int):
            - 현재 epoch 인덱스(0부터 시작). 저장 시에는 epoch+1 로 기록한다.  # shape: ()
        train_loss (float):
            - 이번 epoch 의 train loss 스칼라 값.  # shape: ()
        wandb_id (Optional[str]):
            - 이 체크포인트와 연결할 W&B run id. 없으면 None.
        ema (Optional[Any]):
            - EMA 래퍼 또는 None.
            - ema 가 있으면 ema.state_dict() 를 "ema_state_dict" 로 함께 저장한다.
        save_best (bool):
            - True 이면 best.pth 도 갱신한다.
              False 이면 latest.pth 만 덮어쓴다.

    Returns:
        None:
            - 파일 시스템에 latest.pth / best.pth 를 쓰기만 하고 값을 반환하지 않는다.
    """
    save_model = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'ema_state_dict': ema.state_dict() if ema is not None else None,
        'optimizer': optimizer.state_dict(),
        'schedule': scheduler.state_dict(),
        'loss': train_loss,
        'wandb_id': wandb_id
    }

    with io.BytesIO() as f:
        torch.save(save_model, f)
        if save_best:
            fileio.put(f.getvalue(), f"{save_path}/best.pth")
        # fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")


def resume_model(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    ema: Optional[Any],
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Any, int, Optional[str],
           Optional[Any]]:
    """저장된 latest.pth 체크포인트를 읽어서 모델/옵티마이저/스케줄러/EMA 상태를 복원한다.

    이 함수는 `save_model(...)` 이 만든 PyTorch 체크포인트를 기준으로 동작한다.
    디렉터리 경로 `path` 를 입력으로 받고, 그 안의 `latest.pth` 파일을 열어서
    다음과 같은 정보를 순서대로 복구한다.

    * 모델 파라미터
      - ckpt["model"] 에서 state_dict 를 읽어 model.load_state_dict(...) 로 주입한다.
      - 각 텐서의 shape 는 계층 종류에 따라 다양하다.
        · Linear weight: (out_dim, in_dim)
        · LayerNorm weight/bias: (hidden_dim,)
        · Conv weight: (C_out, C_in, kH, kW) 등.
      - 예전 포맷(ckpt 자체가 state_dict)도 지원하기 위해, KeyError 시에는
        ckpt 를 그대로 state_dict 로 보고 다시 시도한다.

    * 옵티마이저 상태
      - ckpt["optimizer"] 를 optimizer.load_state_dict(...) 로 복원한다.
      - 8bit ↔ 32bit AdamW 처럼 타입이 달라져서 로드가 실패하면 예외를 잡고
        경고만 출력한 뒤, 옵티마이저는 새 상태로 계속 사용한다.

    * 스케줄러 상태
      - ckpt["schedule"] 를 scheduler.load_state_dict(...) 로 복원한다.
      - 키가 없거나 포맷이 다르면 경고만 출력하고 스케줄러는 새 상태로 둔다.

    * 학습 진행 위치 / wandb id
      - ckpt["epoch"] 를 init_epoch 로 읽어온다. (이미 학습이 끝난 epoch 수)  # shape: ()
      - 없으면 init_epoch=0 으로 본다.
      - ckpt["wandb_id"] 를 wandb_id 로 가져온다. 없으면 None.

    * EMA 상태
      - ckpt["ema_state_dict"] 를 ema.ema.load_state_dict(...) 에 넣어서 복원한다.
      - 이후 ema.ema.eval() 로 평가 모드로 바꾸고,
        모든 파라미터에 대해 requires_grad=False 로 고정한다.
      - 키가 없거나 ema 가 None 이면 "no ema shadow found" 만 출력하고 넘어간다.
      - ema.ema 의 각 파라미터 텐서 shape 는 원본 모델 파라미터와 동일하다.
        (예: (out_dim, in_dim), (hidden_dim,), (C_out, C_in, kH, kW))

    Note:
        device 인자는 현재 이 함수 내부에서는 직접 사용하지 않지만,
        상위 코드와의 인터페이스를 맞추기 위해 인자로 유지한다.

    Args:
        path (str):
            - 체크포인트 디렉터리 경로.
            - 실제로는 `os.path.join(path, "latest.pth")` 파일을 읽는다.
        model (torch.nn.Module):
            - 복원 대상 모델 또는 DDP 래퍼.
            - load_state_dict 를 통해 파라미터 텐서들을 갱신한다.
        optimizer (torch.optim.Optimizer):
            - AdamW/AdamW8bit 등 옵티마이저 인스턴스.
            - state_dict 가 존재하면 학습 중이던 모멘트, step 등을 그대로 이어받는다.
        scheduler (Any):
            - 학습률 스케줄러 객체.
            - scheduler.state_dict() / load_state_dict(...) 를 지원해야 한다.
        ema (Optional[Any]):
            - EMA 래퍼. 보통 timm.utils.ModelEma 와 같이
              내부에 ema.ema (nn.Module) 를 가지고 있다고 가정한다.
            - None 인 경우 EMA 로드는 건너뛴다.
        device (str):
            - "cuda" / "cpu" 등 장치 정보.
            - 현재 구현에서는 사용하지 않지만, 호출부와 시그니처를 맞추기 위해 남겨둔다.

    Returns:
        Tuple[torch.nn.Module, torch.optim.Optimizer, Any, int, Optional[str], Optional[Any]]:
            - model:
                복원된 파라미터를 가진 모델 객체.
            - optimizer:
                가능한 경우 state_dict 가 복원된 옵티마이저.
            - scheduler:
                가능한 경우 state_dict 가 복원된 스케줄러.
            - init_epoch:
                체크포인트에 저장된 학습 재개 epoch 인덱스 (보통 이미 끝난 epoch 수).  # shape: ()
            - wandb_id:
                이어서 사용할 W&B run id. 없으면 None.
            - ema:
                EMA shadow 가 복원된 EMA 래퍼 또는 원래 ema 값.
    """
    path = os.path.join(path, 'latest.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f)

    # load model
    try:
        model.load_state_dict(ckpt['model'])
    except:
        model.load_state_dict(ckpt)
    print("Model load done")

    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except Exception as e:
        # 8bit ↔ 32bit AdamW 같이 타입이 바뀐 경우에도 여기로 온다.
        print(f"no pretrained optimizer found or optimizer state mismatch: {e}")

    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")

    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0

    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None

    try:
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')

    return model, optimizer, scheduler, init_epoch, wandb_id, ema
