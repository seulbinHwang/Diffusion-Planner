import torch
import random
import numpy as np
from mmengine import fileio
import io
import os
import json
from pickle import UnpicklingError


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


def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(
                    value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [
                    value if isinstance(value, (int, float)) else value.item()
                ]

    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss


def save_model(model, optimizer, scheduler, save_path, epoch, train_loss,
               wandb_id, ema, save_best):
    """
    save the model to path
    """
    save_model = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'ema_state_dict': ema.state_dict(),
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


def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    load ckpt from path
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
    except:
        print("no pretrained optimizer found")

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
