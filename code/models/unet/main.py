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
    
    # Cross-validation
    parser.add_argument('--k_folds', type=int,
                        help='Number of folds for k-fold cross-validation')
    parser.add_argument('--random_seed', type=int,
                        help='Random seed for reproducible cross-validation splits')
    
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
    required_args = ['raw_video_dir', 'masked_video_dir', 'k_folds', 'epochs', 'batch_size', 
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