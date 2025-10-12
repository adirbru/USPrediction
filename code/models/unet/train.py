# train.py

import torch
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

from loader import get_model
from dataset import USSegmentationDataset


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
        logging.info(f"  Batch loss: {loss.item():.4f}")

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
    """U-Net trainer with subject-based train/validation split"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.best_val_loss = float("inf")
        self.best_model = None
        
        logging.info(f"Using device: {self.device}")
        logging.info(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
    
    def create_dataset(self):
        """Create the full dataset from video files"""
        dataset = USSegmentationDataset(
            raw_video_dir=self.args.raw_video_dir,
            masked_video_dir=self.args.masked_video_dir,
            temp_dir=self.args.temp_dir,
        )
        return dataset

    def setup_subject_split(self, dataset):
        """Setup subject-based train/validation split"""
        subject_ids = dataset.get_subject_ids()

        # Use args.train_ratio for training, 1 - args.train_ratio for validation
        train_subjects, val_subjects = train_test_split(
            subject_ids, 
            test_size=1 - self.args.train_ratio, 
            random_state=self.args.random_seed,
            shuffle=True
        )
        
        logging.info("Subject split:")
        logging.info(f"  Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
        logging.info(f"  Validation subjects ({len(val_subjects)}): {sorted(val_subjects)}")
        
        # Get frame indices for each split
        train_indices = dataset.get_video_indices_for_subjects(train_subjects)
        val_indices = dataset.get_video_indices_for_subjects(val_subjects)
        
        return train_indices, val_indices, train_subjects, val_subjects
    
    def train_model(self, model, train_loader, val_loader, criterion, optimizer, epochs):
        """Train model with subject-based validation"""
        train_losses = []
        val_losses = []
        
        for epoch in range(1, epochs + 1):
            logging.info(f"Epoch {epoch}/{epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss = validate(model, val_loader, criterion, self.device)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            logging.info(f"  Train Loss: {train_loss:.4f}")
            logging.info(f"  Val   Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = model.state_dict().copy()
                logging.info(f"  New best model saved (val_loss: {val_loss:.4f})")
        
        return train_losses, val_losses
    
    def run_training(self, dataset: USSegmentationDataset):
        """Run the complete subject-based training process"""
        # Setup subject-based split
        train_indices, val_indices, train_subjects, val_subjects = self.setup_subject_split(dataset)
        
        logging.info(f"\n{'='*60}")
        logging.info("SUBJECT-BASED TRAINING")
        logging.info(f"{'='*60}")
        logging.info(f"Train samples: {len(train_indices)} (from {len(train_subjects)} subjects)")
        logging.info(f"Validation samples: {len(val_indices)} (from {len(val_subjects)} subjects)")
        logging.info(f"{'='*60}")
        
        # Create subsets
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
        
        # Create model
        model = get_model().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        
        # Train model
        train_losses, val_losses = self.train_model(
            model, train_loader, val_loader, criterion, optimizer, self.args.epochs
        )
        
        # Store results
        self.training_results = {
            'train_subjects': train_subjects,
            'val_subjects': val_subjects,
            'train_size': len(train_indices),
            'val_size': len(val_indices),
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': self.best_val_loss,
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
    
    def run(self):
        """Main training pipeline"""
        dataset = self.create_dataset()
        self.run_training(dataset)
        self.save_best_model()
        self.print_summary()