#!/usr/bin/env python3
"""
Leave-one-subject-out cross-validation experiment.
For each subject, train on all other subjects for 1 epoch,
validate on the held-out subject, and save results.
"""

import sys
from pathlib import Path

# Add parent directories to path to import modules
unet_dir = Path(__file__).resolve().parent.parent.parent
code_dir = unet_dir.parent.parent
sys.path.insert(0, str(unet_dir))
sys.path.insert(0, str(code_dir))

import torch
import logging
import json
import argparse
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim

from loader import get_model
from dataset import USSegmentationDataset
from train import train_one_epoch, validate
from infer import run_inference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LeaveOneOutExperiment:
    """Leave-one-subject-out cross-validation experiment"""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Create output directory for experiment results
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup dataset
        logging.info("Loading dataset...")
        self.dataset = USSegmentationDataset(
            raw_video_dir=self.args.raw_video_dir,
            masked_video_dir=self.args.masked_video_dir,
            temp_dir=self.args.temp_dir,
            preprocess=self.args.preprocess,
            in_place_augmentations=self.args.in_place_augmentations,
            enrichment_augmentations=self.args.enrichment_augmentations,
            augmentation_params=vars(self.args)
        )

        self.all_subjects = sorted(self.dataset.get_subject_ids())
        logging.info(f"Found {len(self.all_subjects)} subjects: {self.all_subjects}")

    def run_experiment(self):
        """Run leave-one-subject-out cross-validation"""
        all_results = {}

        for held_out_subject in self.all_subjects:
            logging.info(f"\n{'='*80}")
            logging.info(f"EXPERIMENT: Holding out subject {held_out_subject}")
            logging.info(f"{'='*80}\n")

            # Split subjects
            train_subjects = [s for s in self.all_subjects if s != held_out_subject]
            val_subjects = [held_out_subject]

            logging.info(f"Training subjects: {train_subjects}")
            logging.info(f"Validation subject: {val_subjects}")

            # Get indices for train/val split
            train_indices = self.dataset.get_video_indices_for_subjects(train_subjects)
            val_indices = self.dataset.get_video_indices_for_subjects(val_subjects)

            # Create subsets
            train_subset = Subset(self.dataset, train_indices)
            val_subset = Subset(self.dataset, val_indices)

            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers
            )

            logging.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")

            # Initialize model
            model = get_model().to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)

            # Train for 1 epoch
            logging.info(f"\nTraining on subjects {train_subjects} for 1 epoch...")
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, self.device)

            logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Accuracy: {train_metrics['accuracy']:.4f}, "
                        f"Precision: {train_metrics['precision']:.4f}, "
                        f"Recall: {train_metrics['recall']:.4f}")

            # Validate on held-out subject
            logging.info(f"\nValidating on subject {held_out_subject}...")
            val_metrics = validate(model, val_loader, criterion, self.device)

            logging.info(f"Val - Loss: {val_metrics['loss']:.4f}, "
                        f"Accuracy: {val_metrics['accuracy']:.4f}, "
                        f"Precision: {val_metrics['precision']:.4f}, "
                        f"Recall: {val_metrics['recall']:.4f}")

            # Save model checkpoint
            checkpoint_path = self.output_dir / "current_unet.pth"
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved model to {checkpoint_path}")

            # Save metrics to JSON
            results = {
                'subject_id': held_out_subject,
                'train_subjects': train_subjects,
                'val_subject': held_out_subject,
                'train_samples': len(train_subset),
                'val_samples': len(val_subset),
                'train_metrics': {
                    'loss': float(train_metrics['loss']),
                    'accuracy': float(train_metrics['accuracy']),
                    'precision': float(train_metrics['precision']),
                    'recall': float(train_metrics['recall'])
                },
                'val_metrics': {
                    'loss': float(val_metrics['loss']),
                    'accuracy': float(val_metrics['accuracy']),
                    'precision': float(val_metrics['precision']),
                    'recall': float(val_metrics['recall'])
                }
            }

            # Save JSON results
            json_path = self.output_dir / f"subject_{held_out_subject}_results.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            logging.info(f"Saved results to {json_path}")

            # Run inference on first frame of held-out subject
            self._run_inference_on_subject(held_out_subject, checkpoint_path)

            all_results[held_out_subject] = results

        # Save summary of all results
        summary_path = self.output_dir / "all_subjects_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info(f"\nSaved summary of all results to {summary_path}")

        logging.info(f"\n{'='*80}")
        logging.info("EXPERIMENT COMPLETE")
        logging.info(f"{'='*80}")

    def _run_inference_on_subject(self, subject_id, checkpoint_path):
        """Run inference on the first frame of a subject"""
        # Find the first frame for this subject
        subject_indices = self.dataset.get_video_indices_for_subjects([subject_id])
        if not subject_indices:
            logging.warning(f"No frames found for subject {subject_id}")
            return

        first_frame_index = subject_indices[0]

        # Get the frame path from dataset
        video_index = 0
        frame_index = first_frame_index
        for video in self.dataset.videos:
            if frame_index < len(video):
                raw_frame_path, _ = video[frame_index]
                break
            frame_index -= len(video)
            video_index += 1

        # Run inference
        output_name = f"subject_{subject_id}_inference.png"
        logging.info(f"Running inference on first frame for subject {subject_id}")

        try:
            run_inference(
                checkpoint_path=str(checkpoint_path),
                raw_frame_path=str(raw_frame_path),
                output_dir=str(self.output_dir),
                use_cpu=False,
                output_name=output_name
            )
        except Exception as e:
            logging.error(f"Error running inference for subject {subject_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Leave-one-subject-out cross-validation experiment')

    # Data paths
    parser.add_argument('--raw_video_dir', type=str,
                       default='/home/crl05/USPrediction/dataset/raw',
                       help='Directory containing raw video files')
    parser.add_argument('--masked_video_dir', type=str,
                       default='/home/crl05/USPrediction/dataset/masks',
                       help='Directory containing masked video files')
    parser.add_argument('--temp_dir', type=str, default='../../temp_frames',
                       help='Temporary directory for extracted frames')
    parser.add_argument('--output_dir', type=str, default='./results',
                       help='Directory to save experiment results')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=10,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=10,
                       help='Number of data loader workers')

    # Data processing
    parser.add_argument('--preprocess', action='store_true',
                       help='Force re-extraction of frames')

    # Augmentations (from config)
    parser.add_argument('--in_place_augmentations', nargs='*',
                       default=['quantize', 'resize', 'speckle', 'gamma'],
                       help='In-place augmentations to apply')
    parser.add_argument('--enrichment_augmentations', nargs='*',
                       default=['random_resize_crop'],
                       help='Enrichment augmentations to apply')

    # Augmentation parameters
    parser.add_argument('--resize_to', nargs=2, type=int, default=[240, 240],
                       help='Target size for resize augmentation')
    parser.add_argument('--speckle_variance', type=float, default=0.1,
                       help='Variance for speckle noise')
    parser.add_argument('--gamma_range', nargs=2, type=float, default=[0.5, 2.0],
                       help='Range for gamma correction')
    parser.add_argument('--random_resize_crop_percent', type=int, default=5,
                       help='Percentage for random resize crop')

    args = parser.parse_args()

    # Run experiment
    experiment = LeaveOneOutExperiment(args)
    experiment.run_experiment()


if __name__ == "__main__":
    main()
