# train.py

import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np

from loader import get_model
from dataset import USSegmentationDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_pixel_metrics(outputs, masks):
    """
    Compute pixel-level accuracy, precision, and recall for multi-class segmentation.

    Args:
        outputs: Model outputs (N, num_classes, H, W) - logits
        masks: Ground truth masks (N, H, W) - class indices

    Returns:
        dict: Dictionary containing overall accuracy and per-class precision and recall
    """
    # Get predicted class for each pixel
    preds = torch.argmax(outputs, dim=1)  # (N, H, W)

    # Flatten predictions and masks for easier computation
    preds_flat = preds.view(-1)
    masks_flat = masks.view(-1)

    # Overall pixel accuracy (all classes)
    correct_pixels = (preds_flat == masks_flat).sum().float()
    total_pixels = preds_flat.numel()
    overall_accuracy = (correct_pixels / total_pixels).item()

    # Get number of classes from outputs
    num_classes = outputs.shape[1]

    # Compute per-class precision and recall
    per_class_precision = []
    per_class_recall = []

    for class_idx in range(num_classes):
        # True Positives: predicted class_idx AND ground truth class_idx
        tp = ((preds_flat == class_idx) & (masks_flat == class_idx)).sum().float()

        # False Positives: predicted class_idx BUT ground truth is NOT class_idx
        fp = ((preds_flat == class_idx) & (masks_flat != class_idx)).sum().float()

        # False Negatives: predicted NOT class_idx BUT ground truth IS class_idx
        fn = ((preds_flat != class_idx) & (masks_flat == class_idx)).sum().float()

        # Precision: TP / (TP + FP)
        precision = (tp / (tp + fp + 1e-7)).item()
        per_class_precision.append(precision)

        # Recall: TP / (TP + FN)
        recall = (tp / (tp + fn + 1e-7)).item()
        per_class_recall.append(recall)

    # Macro-averaged precision and recall (average across classes)
    avg_precision = sum(per_class_precision) / num_classes
    avg_recall = sum(per_class_recall) / num_classes

    return {
        'accuracy': overall_accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0

    for imgs, masks in tqdm(loader, desc="Train"):
        # The DataLoader output for images is (N, C, H, W) if RawVideo.get_frame_matrix returns (C, H, W).
        # Assuming RawVideo returns (1, H, W), the DataLoader output is (N, 1, H, W).
        # We REMOVE the redundant .unsqueeze(1) call.
        imgs  = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)                   # (N, classes, H, W)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # Compute metrics
        with torch.no_grad():
            metrics = compute_pixel_metrics(outputs, masks)
            total_accuracy += metrics['accuracy']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            num_batches += 1

        # logging.info(f"  Batch loss: {loss.item():.4f}") # Disabled to prevent excessive logging

    avg_loss = running_loss / len(loader.dataset)
    avg_accuracy = total_accuracy / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall
    }


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
    total_accuracy = 0.0
    total_precision = 0.0
    total_recall = 0.0
    num_batches = 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val  "):
            # The DataLoader output for images is (N, C, H, W) if RawVideo.get_frame_matrix returns (C, H, W).
            # Assuming RawVideo returns (1, H, W), the DataLoader output is (N, 1, H, W).
            # We REMOVE the redundant .unsqueeze(1) call.
            imgs  = imgs.to(device)
            masks = masks.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.item() * imgs.size(0)

            # Compute metrics
            metrics = compute_pixel_metrics(outputs, masks)
            total_accuracy += metrics['accuracy']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            num_batches += 1

    avg_loss = val_loss / len(loader.dataset)
    avg_accuracy = total_accuracy / num_batches
    avg_precision = total_precision / num_batches
    avg_recall = total_recall / num_batches

    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'precision': avg_precision,
        'recall': avg_recall
    }

class UNetTrainer:
    """U-Net trainer with subject-based train/validation split"""
    def __init__(self, args):
        self.args = args
        # Check if CPU is forced
        use_cpu = getattr(args, 'cpu', False)
        if use_cpu:
            self.device = torch.device("cpu")
            logging.info("CPU mode forced by --cpu flag")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # --- Data Setup ---
        self.dataset = USSegmentationDataset(
            raw_video_dir=self.args.raw_video_dir,
            masked_video_dir=self.args.masked_video_dir,
            temp_dir=self.args.temp_dir,
            preprocess=self.args.preprocess,
            in_place_augmentations=self.args.in_place_augmentations,
            enrichment_augmentations=self.args.enrichment_augmentations,
            augmentation_params=vars(self.args)
        )
        self.train_subjects, self.val_subjects = self._split_subjects()
        self.train_loader, self.val_loader = self._setup_loaders()
        
        # --- Model and Weights Setup ---
        self.model = get_model().to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Add weight decay for L2 regularization (helps prevent overfitting)
        weight_decay = getattr(self.args, 'weight_decay', 1e-5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler - reduces LR when validation loss plateaus
        scheduler_patience = getattr(self.args, 'scheduler_patience', 5)
        scheduler_factor = getattr(self.args, 'scheduler_factor', 0.5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )
        
        # Early stopping parameters
        self.early_stop_patience = getattr(self.args, 'early_stop_patience', 10)
        self.early_stop_min_delta = getattr(self.args, 'early_stop_min_delta', 1e-4)
        self.early_stop_counter = 0
        
        self.best_val_loss = float('inf')
        self.best_model = None
        self.training_results = {}

    def _calculate_class_weights(self):
        """
        Calculate class weights for CrossEntropyLoss to address class imbalance.
        Since Black (index 0) is the background and dominant, we give it a lower weight.
        """
        # Define a base weight array: 8 classes (0 to 7)
        weights = np.ones(8, dtype=np.float32)
        
        # Class 0 (Black - Background) is typically dominant and should be penalized less
        # We can set its weight significantly lower than 1.0
        weights[0] = 0.1 # Example: 10% of the normal weight
        
        # Other classes (1-7, your target segments) keep higher weight
        weights[1:] = 1.0 # Standard weight for foreground classes
        
        logging.info(f"Using class weights: {weights}")
        return torch.from_numpy(weights)


    def _split_subjects(self):
        """Splits the dataset's subject IDs into train and validation sets."""
        all_subjects = self.dataset.get_subject_ids()
        
        # Ensure subject IDs are used for splitting to maintain independence
        train_subjects, val_subjects = train_test_split(
            all_subjects, 
            train_size=self.args.train_ratio, 
            random_state=self.args.random_seed
        )
        return train_subjects, val_subjects

    def _setup_loaders(self):
        """Creates DataLoader objects for training and validation."""
        
        # Get all frame indices corresponding to the selected subjects
        train_indices = self.dataset.get_video_indices_for_subjects(self.train_subjects)
        val_indices = self.dataset.get_video_indices_for_subjects(self.val_subjects)
        
        # Create subsets based on indices
        train_subset = Subset(self.dataset, train_indices)
        val_subset = Subset(self.dataset, val_indices)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.args.batch_size,
            shuffle=False, # No need to shuffle validation data
            num_workers=self.args.num_workers
        )
        
        logging.info(f"Train samples: {len(train_subset)}, Val samples: {len(val_subset)}")
        return train_loader, val_loader

    def run(self):
        """Runs the main training loop."""
        train_metrics_history = []
        val_metrics_history = []

        logging.info(f"\n{'='*60}\nStarting Training for {self.args.epochs} Epochs\n{'='*60}")

        for epoch in range(self.args.epochs):
            logging.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")

            # Training phase
            train_metrics = train_one_epoch(
                self.model, self.train_loader, self.criterion, self.optimizer, self.device
            )
            train_metrics_history.append(train_metrics)
            logging.info(f"Epoch {epoch+1} Train - Loss: {train_metrics['loss']:.4f}, "
                        f"Accuracy: {train_metrics['accuracy']:.4f}, "
                        f"Precision: {train_metrics['precision']:.4f}, "
                        f"Recall: {train_metrics['recall']:.4f}")

            # Validation phase
            val_metrics = validate(
                self.model, self.val_loader, self.criterion, self.device
            )
            val_metrics_history.append(val_metrics)
            logging.info(f"Epoch {epoch+1} Val   - Loss: {val_metrics['loss']:.4f}, "
                        f"Accuracy: {val_metrics['accuracy']:.4f}, "
                        f"Precision: {val_metrics['precision']:.4f}, "
                        f"Recall: {val_metrics['recall']:.4f}")

            # Update learning rate scheduler based on validation loss
            self.scheduler.step(val_metrics['loss'])
            current_lr = self.optimizer.param_groups[0]['lr']
            logging.info(f"Current learning rate: {current_lr:.6f}")

            # Check for best model
            improvement = self.best_val_loss - val_metrics['loss']
            if improvement > self.early_stop_min_delta:
                # Significant improvement - reset early stop counter
                self.best_val_loss = val_metrics['loss']
                self.best_model = self.model.state_dict()
                self.early_stop_counter = 0
                logging.info(f"New best model saved. (Improvement: {improvement:.6f})")
            else:
                # No significant improvement
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.early_stop_patience:
                    logging.info(f"\nEarly stopping triggered! No improvement for {self.early_stop_patience} epochs.")
                    logging.info(f"Best validation loss: {self.best_val_loss:.4f} at epoch {epoch+1 - self.early_stop_patience}")
                    break

        self._record_results(train_metrics_history, val_metrics_history)
        self.save_best_model()
        self.print_summary()
        
    def _record_results(self, train_metrics_history, val_metrics_history):
        """Record final results and training history"""
        final_train = train_metrics_history[-1] if train_metrics_history else {}
        best_val = min(val_metrics_history or [{}], key=lambda history: history.get('loss'))

        self.training_results = {
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'train_size': len(self.train_loader.dataset),
            'val_size': len(self.val_loader.dataset),
            'best_val_loss': self.best_val_loss,

            # Final metrics
            'final_train_loss': final_train.get('loss', float('nan')),
            'final_train_accuracy': final_train.get('accuracy', float('nan')),
            'final_train_precision': final_train.get('precision', float('nan')),
            'final_train_recall': final_train.get('recall', float('nan')),

            'best_val_loss': best_val.get('loss', self.best_val_loss),
            'best_val_accuracy': best_val.get('accuracy', float('nan')),
            'best_val_precision': best_val.get('precision', float('nan')),
            'best_val_recall': best_val.get('recall', float('nan')),

            # Full history
            'train_metrics_history': train_metrics_history,
            'val_metrics_history': val_metrics_history
        }
    
    def save_best_model(self):
        """Save the best model"""
        if self.best_model is not None:
            torch.save(self.best_model, self.args.checkpoint_path)
            logging.info(f"\nSaved best model (val_loss: {self.best_val_loss:.4f}) to {self.args.checkpoint_path}")
    
    def print_summary(self):
        """logging.info training summary"""
        logging.info(f"\n{'='*60}")
        logging.info("TRAINING SUMMARY")
        logging.info(f"{'='*60}")

        results = self.training_results

        logging.info(f"Training subjects: {sorted(results['train_subjects'])} ({len(results['train_subjects'])} subjects)")
        logging.info(f"Validation subjects: {sorted(results['val_subjects'])} ({len(results['val_subjects'])} subjects)")
        logging.info(f"Training samples: {results['train_size']}")
        logging.info(f"Validation samples: {results['val_size']}")

        logging.info(f"\nFinal Training Metrics:")
        logging.info(f"  Loss:      {results['final_train_loss']:.4f}")
        logging.info(f"  Accuracy:  {results['final_train_accuracy']:.4f}")
        logging.info(f"  Precision: {results['final_train_precision']:.4f}")
        logging.info(f"  Recall:    {results['final_train_recall']:.4f}")

        logging.info(f"\nBest Validation Metrics:")
        logging.info(f"  Loss:      {results['best_val_loss']:.4f}")
        logging.info(f"  Accuracy:  {results['best_val_accuracy']:.4f}")
        logging.info(f"  Precision: {results['best_val_precision']:.4f}")
        logging.info(f"  Recall:    {results['best_val_recall']:.4f}")

        logging.info(f"\nBest validation loss: {results['best_val_loss']:.4f}")

        logging.info("\nTraining complete!")
