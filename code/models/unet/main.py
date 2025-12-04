import argparse
import json

from train import UNetTrainer

def load_config(config_path):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in configuration file: {e}")


def parse_args():
    """
    Parse command line arguments with support for config files.
    Either provide all arguments or use --config with a JSON file.
    If both are provided, the provided command line arguments will override the config file ones.
    """
    parser = argparse.ArgumentParser(description='Train U-Net for ultrasound segmentation')
    
    # Configuration file option
    parser.add_argument('--config', type=str, 
                        help='Path to JSON configuration file (overrides all other arguments)')
    
    # Data paths
    parser.add_argument('--raw_video_dir', type=str,
                        help='Path to raw video files directory')
    parser.add_argument('--masked_video_dir', type=str,
                        help='Path to masked video files directory')
    parser.add_argument('--temp_dir', type=str,
                        help='Temporary directory to store extracted frames')
    
    # Subject-based split
    parser.add_argument('--train_ratio', type=float,
                        help='Ratio of subjects to use for training')
    parser.add_argument('--random_seed', type=int,
                        help='Random seed for reproducible subject splits')
    
    # Training parameters
    parser.add_argument('--epochs', type=int,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float,
                        help='Learning rate for optimizer')
    parser.add_argument('--num_workers', type=int,
                        help='Number of workers for data loading')
    
    # Model and checkpoint
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to save the best model checkpoint')
    parser.add_argument('--preprocess', action='store_true', default=False,
                        help='Force re-extraction of frames from videos and re-creation of all augmentations. Without this flag, existing frames and augmentations are loaded (augmentations are created only if missing).')

    # Augmentations
    parser.add_argument('--in_place_augmentations', type=str, nargs='*', default=None,
                        help='space-separated list of augmentation names to apply on-the-fly during training (e.g. "quantize"). These augmentations are applied during __getitem__ and do not create new files.')

    parser.add_argument('--enrichment_augmentations', type=str, nargs='*', default=None,
                        help='space-separated list of augmentation names to apply during preprocessing and enrich the dataset (e.g. "flip random_resize_crop"). These augmentations will create new augmented frames in addition to the original ones.')

    parser.add_argument('--random_resize_crop_percent', type=int, default=20,
                        help='For random_resize_crop augmentation, how much of the original width/height to potentially remove (0–100)')

    parser.add_argument('--resize_to', type=int, nargs=2, default=(240, 240),
                        help='Target size (width height) to resize frames to during preprocessing')
    
    # Regularization and training improvements
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 regularization) for optimizer (default: 1e-5)')
    parser.add_argument('--scheduler_patience', type=int, default=5,
                        help='Number of epochs with no improvement before reducing learning rate (default: 5)')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor by which learning rate is reduced (default: 0.5)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Number of epochs with no improvement before early stopping (default: 10, set to 0 to disable)')
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4,
                        help='Minimum change to qualify as an improvement for early stopping (default: 1e-4)')
    
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU training even if CUDA is available (useful for CUDA compatibility issues)')

    args = parser.parse_args()
    
    # If config file is provided, load it and override args
    if args.config:
        config = load_config(args.config)

        config_args = {}
        
        # Override args with config values
        for sub_dict in config:
            config_args.update(config[sub_dict])

        # Give priority to command line arguments
        for key, value in args.__dict__.items():
            if value is not None:
                config_args[key] = value
    
        args.__dict__.update(config_args)

    # Validate required arguments
    required_args = ['raw_video_dir', 'masked_video_dir', 'epochs', 'batch_size', 
                     'learning_rate', 'checkpoint_path']

    missing_args = [arg for arg in required_args if getattr(args, arg) is None]
    
    if missing_args:
            raise ValueError(f"Missing required arguments: {missing_args}. ")
    
    return args


def main():
    """Main entry point - parse arguments and run training"""
    args = parse_args()
    trainer = UNetTrainer(args)
    trainer.run()


if __name__ == "__main__":
    main()