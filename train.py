import os
import time
import copy
import gc
from collections import defaultdict
import numpy as np
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.cuda import amp

# For colored terminal text
from colorama import Fore, Style

from config import CFG
from helpers import setup_criterion, setup_optimizer, setup_scheduler
from metrics import dice_coef, iou_coef, recall_coef, precision_coef
from utils import seed_everything, save_metrics_plot
from model.unet import UNet
from model.utils import save_model
from dataset.dataloader import prepare_loader
from dataset.augmentations import get_train_transforms, get_valid_transforms

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

c_  = Fore.GREEN
sr_ = Style.RESET_ALL


def train_epoch(model, criterion, optimizer, scheduler, dataloader, device):
    model.train()
    scaler = amp.GradScaler()

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        scaler.scale(loss).backward()

        if (step + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')
        torch.cuda.empty_cache()
        gc.collect()

    return epoch_loss, criterion, optimizer, scheduler


@torch.no_grad()
def valid_epoch(model, criterion, optimizer, dataloader, device):
    model.eval()

    dataset_size = 0
    epoch_loss = 0.0
    running_loss = 0.0

    val_scores = []

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        y_pred = model(images)
        loss = criterion(y_pred, masks)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_recall = recall_coef(masks, y_pred).cpu().detach().numpy()
        val_precision = precision_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard, val_recall, val_precision])

        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')
    val_scores = np.mean(val_scores, axis=0)
    torch.cuda.empty_cache()
    gc.collect()

    return epoch_loss, val_scores


def run_training(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf     # Dice metric has the same formula as F1 metric
    best_epoch = -1
    history = defaultdict(list)

    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')

        train_loss, criterion, optimizer, scheduler = train_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            dataloader=train_loader,
            device=CFG.device
        )

        val_loss, val_scores = valid_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            dataloader=valid_loader,
            device=CFG.device
        )
        val_dice, val_jaccard, val_recall, val_precision = val_scores

        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)
        history['Valid Recall'].append(val_recall)
        history['Valid Precision'].append(val_precision)

        # Log the metrics
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')
        print(f'Valid Recall: {val_recall:0.4f} | Valid Precision: {val_precision:0.4f}')
        print("\n")

        # deep copy the model
        if val_dice >= best_dice:
            print(f"{c_}Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f}){sr_}")
            best_dice = val_dice
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())

            # Save a model file from the current directory
            filename = f"{CFG.model_name}_{CFG.backbone}_best_epoch.pth"
            save_model(model, epoch, optimizer, train_loss, val_loss, filename)
            print(f"Model Saved: {filename}")

        filename = f"{CFG.model_name}_{CFG.backbone}_epoch_{epoch}.pth"
        save_model(model, epoch, optimizer, train_loss, val_loss, filename)

        save_metrics_plot(history, epoch)
        print("\n")

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score (Dice): {:.4f}".format(best_dice))
    print(f"Best Epoch: {best_epoch}")

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, history


def main():
    seed_everything(CFG.seed)

    model = UNet(
        encoder_name=CFG.backbone,
        encoder_pretrained=True,
        n_classes=CFG.num_classes,
        activation=None,
    )
    model.to(CFG.device)

    train_transforms = get_train_transforms()
    valid_transforms = get_valid_transforms()
    train_loader = prepare_loader(
        dataset_path=CFG.train_data,
        transforms=train_transforms,
        loader_type="train",
        debug=CFG.debug
    )
    valid_loader = prepare_loader(
        dataset_path=CFG.valid_data,
        transforms=valid_transforms,
        loader_type="valid",
        debug=CFG.debug
    )

    critetion = setup_criterion()
    optimizer = setup_optimizer(model=model)
    scheduler = setup_scheduler(optimizer=optimizer)

    model, history = run_training(
        model=model,
        criterion=critetion,
        optimizer=optimizer, scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=CFG.epochs
    )

    save_metrics_plot(history, CFG.epochs)


if __name__ == "__main__":
    main()