# USPrediction - Ultrasound Muscle Segmentation

Deep learning pipeline for automatic muscle segmentation in ultrasound videos using U-Net with ResNet34 encoder.

## Overview

This project implements a segmentation model to identify and segment different muscle groups in ultrasound imaging. The model is trained on annotated ultrasound videos and can predict pixel-wise segmentation masks in real-time.

**Key Features:**
- U-Net architecture with ImageNet-pretrained ResNet34 encoder
- Subject-based train/validation split to prevent data leakage
- Multi-class segmentation (up to 8 classes including background)
- Comprehensive data augmentation (speckle noise, gamma correction, random crops)
- Video inference with per-frame loss tracking

**Reference:** [Automatic Muscle Segmentation in Ultrasound Videos](https://arxiv.org/pdf/2202.05204)

## Project Structure

```
US_Project/
├── code/                    # Source code (see code/README.md)
│   ├── models/unet/         # U-Net model, training, inference
│   └── utils/               # Video/image utilities, augmentations
├── dataset/                 # Raw videos and mask videos
│   ├── raw/                 # Raw ultrasound videos
│   └── masks/               # Annotated mask videos
└── visual_results/          # Generated comparison videos

```

## Visual Results

Side-by-side comparison showing **Raw Video | Ground Truth Masks | Model Predictions**:

https://github.com/user-attachments/assets/3e881822-265a-44e2-bc76-7496bb4ed673

The comparison video includes:
- Frame counter at the top
- Three panels: raw ultrasound, ground truth segmentation, model prediction
- Real-time visualization of model performance

## Authors

- Adir Bruchim (adirbru1@gmail.com)
- Eyal Amdur (eyalamdur@gmail.com)

Technion - Israel Institute of Technology
