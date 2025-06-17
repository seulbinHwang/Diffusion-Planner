import os
from torch.utils.tensorboard import SummaryWriter

import wandb


class TensorBoardLogger():

    def __init__(self,
                 run_name,
                 notes,
                 args,
                 wandb_resume_id,
                 save_path,
                 rank=0):
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
            wandb_writer = wandb.init(project='Diffusion-Planner', # 폴더
                                      name=run_name, # 별명: diffusion-planner-training
                                      notes=notes, # 메모: " "
                                      resume="allow",
                                      id=wandb_resume_id, # None
                                      sync_tensorboard=True,
                                      dir=f'{save_path}')
            wandb.config.update(args)
            self.id = wandb_writer.id

            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')

    def log_metrics(self, metrics: dict, step: int):
        """
       metrics (dict):
       step (int, optional): epoch or step
       """
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
        if self.writer is not None:
            self.writer.close()
