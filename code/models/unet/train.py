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


# 3) Custom Dataset
class USSegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transforms=None):
        self.img_paths  = sorted(glob(os.path.join(img_dir, "*")))
        self.mask_paths = sorted(glob(os.path.join(mask_dir, "*")))
        self.transforms = transforms

        assert len(self.img_paths) == len(self.mask_paths), \
            "Image and mask counts do not match"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # load grayscale image
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        # load color mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_COLOR)

        # apply any albumentations transforms
        if self.transforms:
            augmented = self.transforms(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        # normalize & add channel dim
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)  # (1, H, W)

        # convert mask from RGB to class indices
        h, w, _ = mask.shape
        label = np.zeros((h, w), dtype=np.int64)
        for rgb, idx_cls in COLOR2IDX.items():
            matches = np.all(mask == rgb, axis=-1)
            label[matches] = idx_cls

        return torch.tensor(img), torch.tensor(label)
    
# 4) Transforms (you can add flips, rotations, etc.) # TODO: Add as function
train_transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
])
val_transform = A.Compose([
    A.Resize(256, 256),
])


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)                   # (N, classes, H, W)
                                                # segmentation-models-pytorch with activation="softmax" returns probabilities;
                                                # CrossEntropyLoss expects raw logits. So either:
                                                #  - change activation to None and rely on logits + CE, or
                                                #  - here, wrap it manually:
        outputs = torch.log(outputs + 1e-8)     # to get log-probs

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val  "):
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            outputs = torch.log(outputs + 1e-8)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)

    return val_loss / len(loader.dataset)


def main():
    # Parameters
    training_epochs = 50
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create datasets & loaders
    train_ds = USSegmentationDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR, transforms=train_transform)
    val_ds   = USSegmentationDataset(VAL_IMG_DIR, VAL_MASK_DIR, transforms=val_transform)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=4)

    # Model, loss, optimizer
    model = get_model().to(device)
    # Use NLLLoss since we converted to log-probs; it’s equivalent to CE on logits
    criterion = nn.NLLLoss()                                # TODO Check loss correctness
    optimizer = optim.Adam(model.parameters(), lr=1e-4)     # TODO Check optimizer correctness

    best_val_loss = float("inf")
    
    for epoch in range(1, training_epochs + 1):
        print(f"\nEpoch {epoch}/{training_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss   = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val   Loss: {val_loss:.4f}")

        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("  Saved new best model.")

    print("Training complete.")


if __name__ == "__main__":
    main()