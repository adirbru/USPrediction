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


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
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
        # logging.info(f"  Batch loss: {loss.item():.4f}") # Disabled to prevent excessive logging

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    val_loss = 0.0
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

    return val_loss / len(loader.dataset)

class UNetTrainer:
    """U-Net trainer with subject-based train/validation split"""
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # --- Data Setup ---
        self.dataset = USSegmentationDataset(
            raw_video_dir=self.args.raw_video_dir,
            masked_video_dir=self.args.masked_video_dir,
            temp_dir=self.args.temp_dir
        )
        self.train_subjects, self.val_subjects = self._split_subjects()
        self.train_loader, self.val_loader = self._setup_loaders()
        
        # --- Model and Weights Setup ---
        self.model = get_model().to(self.device)
        
        # --- NEW: Calculate and assign class weights ---
        self.class_weights = self._calculate_class_weights()
        
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        
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
        train_losses = []
        val_losses = []
        
        logging.info(f"\n{'='*60}\nStarting Training for {self.args.epochs} Epochs\n{'='*60}")
        
        for epoch in range(self.args.epochs):
            logging.info(f"\n--- Epoch {epoch+1}/{self.args.epochs} ---")
            
            # Training phase
            train_loss = train_one_epoch(
                self.model, self.train_loader, self.criterion, self.optimizer, self.device
            )
            train_losses.append(train_loss)
            logging.info(f"Epoch {epoch+1} Train Loss: {train_loss:.4f}")

            # Validation phase
            val_loss = validate(
                self.model, self.val_loader, self.criterion, self.device
            )
            val_losses.append(val_loss)
            logging.info(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model.state_dict()
                logging.info(f"New best model saved.")
        
        self._record_results(train_losses, val_losses)
        self.save_best_model()
        self.print_summary()
        
    def _record_results(self, train_losses, val_losses):
        """Record final results and training history"""
        self.training_results = {
            'train_subjects': self.train_subjects,
            'val_subjects': self.val_subjects,
            'train_size': len(self.train_loader.dataset),
            'val_size': len(self.val_loader.dataset),
            'best_val_loss': self.best_val_loss,
            'final_train_loss': train_losses[-1] if train_losses else float('nan'),
            'final_val_loss': val_losses[-1] if val_losses else self.best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
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
        logging.info(f"Final training loss: {results['final_train_loss']:.4f}")
        logging.info(f"Final validation loss: {results['final_val_loss']:.4f}")
        logging.info(f"Best validation loss: {results['best_val_loss']:.4f}")
        
        logging.info("\nTraining complete!")
