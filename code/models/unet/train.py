# train.py

import os
import cv2
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim

# from your_model_def import get_model  # where you defined get_model()

# 1) Adjust these paths to your dataset layout
TRAIN_IMG_DIR = "/path/to/train/images"
TRAIN_MASK_DIR = "/path/to/train/masks"
VAL_IMG_DIR   = "/path/to/val/images"
VAL_MASK_DIR  = "/path/to/val/masks"
CHECKPOINT_PATH = "./best_unet.pth"

# 2) Define your color→class-index mapping
COLOR2IDX = {
    (  0,   0,   0): 0,  # background
    (255,   0,   0): 1,  # muscle 1
    (  0, 255,   0): 2,  # muscle 2
    (  0,   0, 255): 3,  # muscle 3
    (255, 255,   0): 4,  # muscle 4
    (255,   0, 255): 5,  # muscle 5
    (  0, 255, 255): 6,  # muscle 6
    # add/remove entries to match your 7 classes
}

IDX2COLOR = {v: k for k, v in COLOR2IDX.items()}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)                   # (N, classes, H, W)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        print(f"  Batch loss: {loss.item():.4f}")

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val  "):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)

    return val_loss / len(loader.dataset)

class UNetTrainer:
