# USPrediction - Ultrasound Muscle Segmentation

A PyTorch-based U-Net model for predicting muscle segmentation in ultrasound videos using k-fold cross-validation.

## Project Overview

This project implements a deep learning pipeline for ultrasound image segmentation, specifically designed to identify and segment different muscle groups in ultrasound videos. The model uses a U-Net architecture with a ResNet34 encoder pre-trained on ImageNet.

## Features

- **U-Net Architecture**: ResNet34 encoder with ImageNet pre-trained weights
- **Subject-Based Validation**: Train on 75% of subjects, validate on 25% to prevent data leakage
- **Video Processing**: Automatic extraction of frames from video files
- **Multi-Class Segmentation**: Supports up to 7 muscle groups
- **Data Augmentation**: Configurable image augmentation during training
- **GPU Support**: Automatic CUDA detection and utilization

## Requirements

- Python >= 3.11
- PyTorch
- OpenCV
- Segmentation Models PyTorch
- scikit-learn
- NumPy
- tqdm

## Installation

### Option 1: Using uv (Recommended)

If you have `uv` installed:

```bash
# Navigate to the code directory
cd code

# Install the package in editable mode
uv pip install -e .
```

### Option 2: Using pip

```bash
# Navigate to the code directory
cd code

# Install the package in editable mode
pip install -e .
```

### Option 3: Manual Installation

```bash
# Navigate to the code directory
cd code

# Install dependencies from pyproject.toml
pip install numpy~=1.26 opencv-python~=4.10 torch~=2.4 segmentation-models-pytorch~=0.3 scikit-learn~=1.5 tqdm~=4.66
```

## Project Structure

```
code/
├── models/
│   └── unet/
│       ├── main.py              # Main training script
│       ├── train.py             # Training logic and cross-validation
│       ├── dataset.py           # Dataset class for video processing
│       ├── loader.py            # Model loading utilities
│       └── configs/
│           └── config.json      # Default configuration file
├── utils/
│   ├── video_utils.py           # Video processing utilities
│   └── prepare_data.py          # Data preparation utilities
├── dataset/                     # Dataset directories
└── pyproject.toml              # Project configuration and dependencies
```

## Configuration

The training process can be configured using a JSON configuration file. Here's the structure and default values:

### Configuration File Structure (`configs/config.json`)

```json
{
  "data": {
    "raw_video_dir": "dataset/raw_test/",
    "masked_video_dir": "dataset/processed/videos/masks",
    "temp_dir": "./temp_frames"
  },
  "split": {
    "train_ratio": 0.75,
    "random_seed": 42
  },
  "training": {
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "num_workers": 8
  },
  "model": {
    "checkpoint_path": "./best_unet.pth"
  }
}
```

### Configuration Parameters

#### Data Parameters
- `raw_video_dir`: Path to directory containing raw ultrasound videos
- `masked_video_dir`: Path to directory containing masked/annotated videos
- `temp_dir`: Temporary directory for storing extracted frames

#### Split Parameters
- `train_ratio`: Ratio of subjects to use for training (default: 0.75)
- `random_seed`: Random seed for reproducible subject splits (default: 42)

#### Training Parameters
- `epochs`: Number of training epochs (default: 1)
- `batch_size`: Batch size for training (default: 8)
- `learning_rate`: Learning rate for Adam optimizer (default: 0.0001)
- `num_workers`: Number of workers for data loading (default: 8)

#### Model Parameters
- `checkpoint_path`: Path to save the best model checkpoint

## Data Preparation

### Video File Structure

The model expects paired video files:

```
dataset/
├── raw/                          # Raw ultrasound videos
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── processed/
    └── videos/
        └── masks/                # Corresponding masked videos
            ├── video1.mp4
            ├── video2.mp4
            └── ...
```

### Mask Format

The masked videos should contain colored segmentation masks where:
- **Black (0,0,0)**: Background
- **Red (255,0,0)**: Muscle group 1
- **Green (0,255,0)**: Muscle group 2
- **Blue (0,0,255)**: Muscle group 3
- **Yellow (255,255,0)**: Muscle group 4
- **Magenta (255,0,255)**: Muscle group 5
- **Cyan (0,255,255)**: Muscle group 6

## Training the Model

### Using Configuration File (Recommended)

```bash
# Navigate to the models/unet directory
cd models/unet

# Train using the default configuration
python main.py --config configs/config.json

# Train with custom configuration
python main.py --config path/to/your/config.json
```

### Using Command Line Arguments

```bash
# Train with command line arguments
python main.py \
    --raw_video_dir "dataset/raw/" \
    --masked_video_dir "dataset/processed/videos/masks" \
    --train_ratio 0.75 \
    --epochs 50 \
    --batch_size 16 \
    --learning_rate 0.001 \
    --checkpoint_path "best_model.pth" \
    --num_workers 4
```

### Mixed Configuration (Config File + Command Line Overrides)

```bash
# Use config file but override specific parameters
python main.py \
    --config configs/config.json \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.0005
```

## Training Process

The training process follows these steps:

1. **Data Loading**: Extracts frames from paired video files and organizes by subject
2. **Subject Split**: Splits subjects into training (75%) and validation (25%) sets
3. **Model Training**: 
   - Creates a U-Net model with ResNet34 encoder
   - Trains on training subjects for the specified number of epochs
   - Validates on validation subjects after each epoch
   - Saves the best model based on validation loss
4. **Results Summary**: Provides detailed training statistics and subject information

### Output

The training process will output:

- **Subject Information**: Which subjects are used for training vs validation
- **Real-time Progress**: Training and validation losses for each epoch
- **Best Model Tracking**: Automatic saving of the best model based on validation loss
- **Final Summary**: Training statistics including final losses and subject breakdown
- **Saved Model**: Best model checkpoint saved to the specified path

### Example Output

```
Using device: cuda
Training parameters: epochs=50, batch_size=16, lr=0.001
Validation strategy: Subject-based split (75% train, 25% validation)

Subject split:
  Train subjects (15): ['01', '04', '05', '06', '10', '17', '19', '20', '21', '23', '24', '25', '26', '27', '28']
  Validation subjects (5): ['02', '03', '07', '08', '09']

============================================================
SUBJECT-BASED TRAINING
============================================================
Train samples: 2400 (from 15 subjects)
Validation samples: 800 (from 5 subjects)
============================================================

Epoch 1/50
Train: 100%|██████████| 150/150 [02:15<00:00,  2.70s/it]
  Batch loss: 1.8234
Val  : 100%|██████████| 50/50 [00:08<00:00,  1.58it/s]
  Train Loss: 1.8234
  Val   Loss: 1.6543
  New best model saved (val_loss: 1.6543)

...

============================================================
TRAINING SUMMARY
============================================================
Training subjects: ['01', '04', '05', '06', '10', '17', '19', '20', '21', '23', '24', '25', '26', '27', '28'] (15 subjects)
Validation subjects: ['02', '03', '07', '08', '09'] (5 subjects)
Training samples: 2400
Validation samples: 800
Final training loss: 0.4523
Final validation loss: 0.4123
Best validation loss: 0.4123

Saved best model (val_loss: 0.4123) to ./best_unet.pth

Training complete!
```

## Model Architecture

The model uses a U-Net architecture with the following specifications:

- **Encoder**: ResNet34 with ImageNet pre-trained weights
- **Input Channels**: 1 (grayscale ultrasound images)
- **Output Classes**: 7 (background + 6 muscle groups)
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: Adam

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `batch_size` in configuration
   - Reduce `resize_height` and `resize_width`

2. **Video Processing Errors**
   - Ensure video files are in supported formats (MP4, AVI, etc.)
   - Check that paired videos have matching names
   - Verify video files are not corrupted

3. **Data Loading Issues**
   - Ensure `num_workers` is appropriate for your system
   - Check that video directories exist and contain files
   - Verify temporary directory has sufficient disk space

### Performance Tips

1. **GPU Utilization**: Ensure CUDA is available for faster training
2. **Data Loading**: Adjust `num_workers` based on your CPU cores
3. **Memory Management**: Monitor GPU memory usage and adjust batch size accordingly
4. **Disk Space**: Ensure sufficient space for temporary frame storage

## Dependencies

The project requires the following Python packages:

- `numpy~=1.26`
- `opencv-python~=4.10`
- `torch~=2.4`
- `segmentation-models-pytorch~=0.3`
- `scikit-learn~=1.5`
- `tqdm~=4.66`

## Authors

- Adir Bruchim (adir.bruchim@campus.technion.ac.il)
- Eyal Amdur (eyal.amdur@campus.technion.ac.il)

## License

This project is part of a university research project at the Technion - Israel Institute of Technology.
