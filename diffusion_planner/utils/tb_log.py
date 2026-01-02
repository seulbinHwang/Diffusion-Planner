import os
from torch.utils.tensorboard import SummaryWriter

import wandb
from typing import Any, Dict, Optional
import torch

def _to_float_for_logging(value: Any) -> float:
    """W&B/TensorBoard에 안전하게 기록할 수 있도록 값을 float로 바꿉니다.

    규칙
    ----
    - torch.Tensor(스칼라)면: value.item()으로 숫자 1개로 바꿉니다. (shape: ())
    - 숫자면: float(...)로 바꿉니다.
    - 그 외 타입이면: 에러를 막기 위해 0.0을 반환합니다.

    Args:
        value (Any): 기록할 값. 보통 스칼라 텐서(shape: ()) 또는 숫자(shape: ()) 입니다.

    Returns:
        float: 기록 가능한 숫자 1개. shape: ()
    """
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().cpu().reshape(-1)[0].item())
    try:
        return float(value)
    except Exception:
        return 0.0


def _build_scalar_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """metrics dict를 '문자열 -> float' 형태로 정리합니다.

    Args:
        metrics (Dict[str, Any]): key는 문자열, value는 스칼라 텐서/숫자일 수 있습니다.

    Returns:
        Dict[str, float]: value가 모두 float인 dict.
    """
    return {str(k): _to_float_for_logging(v) for k, v in metrics.items()}


class TensorBoardLogger():

    def __init__(self, args, wandb_resume_id, rank=0, allow_val_change=False):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """
        self.args = args
        self.writer = None
        self.id = None

        if rank == 0:
            os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"
            """
 “이 실험(런)을 W\&B에 등록하고, 실시간으로 기록을 보내줘”

* `project='Diffusion-Planner'`
  이 실험을 묶어 놓을 **프로젝트 이름**
  W\&B 웹에서 ‘Diffusion-Planner’라는 폴더 안에 결과가 저장돼요.

* `config=args`
  실험에 사용된 모든 설정(hyperparameter)을 `args` 변수에서 가져와 저장해요.
  나중에 실험 결과를 분석할 때, 어떤 설정값으로 얻은 결과인지 쉽게 알 수 있어요.

* `name=run_name`
  이번 실험의 **별명**이에요.
  예를 들어 `diffusion-planner-training` 같은 식으로, 여러 번 돌린 실험을 구분할 때 씁니다.

* `notes=notes`
  이 실험에 대한 **짧은 설명**을 남겨요.

* `resume="allow"`
  이전에 중단된 같은 실험이 있으면, 이어서 기록을 붙여 쌓도록 허용해 줍니다.

* `id=wandb_resume_id`
  이어 붙일 때 쓸 **기존 실험의 고유번호**예요.
  보통은 중단 후 재시작할 때 내부적으로 사용되고, 처음엔 `None`이라 새로 만듭니다.

* `sync_tensorboard=True`
  코드가 TensorBoard로 남기는 로그(그래프, 손실 곡선 등)를
  **자동으로 W\&B로 가져가서** 똑같이 보여 달라고 요청하는 옵션

* `dir=f'{save_path}'`
  W&B가 자체 로그 파일(메트릭, 설정 등)을 **저장할 로컬 폴더** 경로
  보통 `save_path` 안에 `.wandb/` 폴더가 생김
            """
            self.run = wandb.init(
                project=args.project,
                config=args,
                name=args.name,
                notes=args.notes,
                resume='allow',
                id=wandb_resume_id,  # str
                sync_tensorboard=True)  # TensorBoard 로그를 W&B와 자동 동기화
            self.id = self.run.id
            """
            이미 run에 저장된 설정값(config)을 args로 갱신합니다. 
            (재시작/재개 시 “현재 args를 config에 다시 반영”하려는 용도)
    
    allow_val_change는 이미 저장된 설정값을 다른 값으로 바꾸는 걸 허용할지를 정합니다. 
    False면 보통 “기존 값과 다르게 바꾸는 것”을 막고, True면 바꾸는 걸 허용합니다
            """
            wandb.config.update(args, allow_val_change=allow_val_change)

            log_dir = os.path.join(args.save_path, "tb")
            """
    이 코드에서는 wandb.log(...)를 직접 안 쓰고, 
    대신 TensorBoard 로그를 쓰면(SummaryWriter) 
    → W&B가 그 로그를 자동으로 가져가서(sync_tensorboard=True) W&B 화면에 표시하는 구조
            """
            self.writer = SummaryWriter(log_dir=log_dir)

    def log_metrics(self, metrics: dict, step: int):
        """
       metrics (dict):
       step (int, optional): epoch or step
       """
        scalar_metrics = _build_scalar_metrics(metrics)

        if self.writer is not None:
            for key, value in scalar_metrics.items():
                """ 그래프 이름: key
                x축: step
                y축: value
                """
                self.writer.add_scalar(key, value, step)
            self.writer.flush()
            # if wandb.run:  # W&B 대시보드에서도 동일한 메트릭을 보기 위해 추가
            #     wandb.log(metrics, step=step)

    def finish(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
