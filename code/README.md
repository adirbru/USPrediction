# USPrediction - Code Documentation

U-Net model for ultrasound muscle segmentation.

## Installation

```bash
cd code
uv pip install -e .
# or
pip install -e .
```

## Project Structure

```
code/
├── models/unet/
│   ├── main.py                      # Training entry point
│   ├── train.py                     # Training logic
│   ├── dataset.py                   # Dataset class
│   ├── loader.py                    # Model loader
│   ├── infer.py                     # Single frame inference
│   ├── infer_video.py               # Video inference
│   ├── create_comparison_video.py   # Comparison video generator
│   └── configs/config.json          # Configuration file
├── utils/
│   ├── video_utils.py               # Video processing
│   ├── image_utils.py               # Image utilities & color palette
│   ├── augmentations.py             # Data augmentation classes
│   └── prepare_data.py              # Data preparation
└── pyproject.toml
```

## Configuration (`configs/config.json`)

```json
{
  "data": {
    "raw_video_dir": "dataset/raw/",
    "masked_video_dir": "dataset/masks/",
    "temp_dir": "./temp_frames"
  },
  "augmentations": {
    "in_place_augmentations": ["quantize", "resize", "speckle", "gamma"],
    "enrichment_augmentations": ["random_resize_crop", "flip"],
    "random_resize_crop_percent": 20,
    "resize_to": [240, 240],
    "speckle_variance": 0.1,
    "gamma_range": [0.5, 2.0]
  },
  "split": {
    "train_ratio": 0.9,
    "random_seed": 42
  },
  "training": {
    "epochs": 100,
    "batch_size": 8,
    "learning_rate": 0.0001,
    "num_workers": 8
  },
  "model": {
    "checkpoint_path": "./best_unet.pth"
  }
}
```

### Augmentations

**In-place** (on-the-fly, no disk space):
- `quantize` - Convert colored masks to class indices
- `resize` - Resize to `resize_to` dimensions
- `gamma` - Brightness variation (range: `gamma_range`)
- `speckle` - Ultrasound noise (variance: `speckle_variance`)

**Enrichment** (creates additional samples on disk):
- `random_resize_crop` - Random crop (percent: `random_resize_crop_percent`)
- `flip` - Horizontal flip

## Data Format

**File naming**: `recordings_<subject_id>_enrollment<enroll_id>_multi_playing<play_id>.mp4`

```
dataset/
├── raw/
│   └── recordings_06_enrollment01_multi_playing01.mp4
└── masks/
    └── recordings_06_enrollment01_multi_playing01_mask.mp4
```

**Color palette** (BGR):

| Class | Color | Value |
|-------|-------|-------|
| 0 | Black | (0,0,0) |
| 1 | Green | (0,128,0) |
| 2 | Blue | (0,0,128) |
| 3 | Red | (128,0,0) |
| 4 | Teal | (0,128,128) |
| 5 | Purple | (128,0,128) |
| 6 | Yellow | (128,128,0) |
| 7 | Gray | (128,128,128) |

## Usage

### Training

```bash
cd models/unet
python main.py --config configs/config.json
```

Override config values:
```bash
python main.py --config configs/config.json --epochs 50 --batch_size 16
```

### Inference

**Single frame:**
```bash
python infer.py --checkpoint_path ./best_unet.pth --raw_frame_path /path/to/frame.png
```

**Video:**
```bash
python infer_video.py \
    --checkpoint_path ./best_unet.pth \
    --raw_frames_dir ./temp_frames/raw/recordings_06_enrollment01_multi_playing01 \
    --output_video_path ./output.mp4
```

With loss computation:
```bash
python infer_video.py \
    --checkpoint_path ./best_unet.pth \
    --raw_frames_dir ./temp_frames/raw/recordings_06_enrollment01_multi_playing01 \
    --masked_frames_dir ./temp_frames/masked/recordings_06_enrollment01_multi_playing01 \
    --output_video_path ./output.mp4 \
    --loss_graph_path ./loss_graph.png
```

**Comparison video** (raw | mask | prediction side-by-side):
```bash
python create_comparison_video.py \
    --subject_id 06 \
    --exp_dir experiments/4 \
    --output_dir ./comparison_videos
```

Options: `--video_name` (specific video), `--cpu` (force CPU), `--fps` (default: 30)

## Model

- **Architecture**: U-Net with ResNet34 encoder (ImageNet pretrained)
- **Input**: 1 channel (grayscale)
- **Output**: 8 classes
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
