# train.py

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
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
        print(f"  Batch loss: {loss.item():.4f}")

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
    """U-Net trainer with k-fold cross-validation support"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.all_fold_results = []
        self.best_overall_val_loss = float("inf")
        self.best_fold_model = None
        
        print(f"Using device: {self.device}")
        print(f"Training parameters: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.learning_rate}")
        print(f"Cross-validation: {args.k_folds}-fold CV")
    
    def create_dataset(self):
        """Create the full dataset from video files"""
        dataset = USSegmentationDataset(
            raw_video_dir=self.args.raw_video_dir,
            masked_video_dir=self.args.masked_video_dir,
            temp_dir=self.args.temp_dir,
        )
        return dataset

    def setup_cross_validation(self, dataset_size):
        """Setup k-fold cross-validation"""
        kfold = KFold(n_splits=self.args.k_folds, shuffle=True, random_state=self.args.random_seed)
        indices = np.arange(dataset_size)
        return kfold, indices
    
    def train_fold(self, model, train_loader, val_loader, criterion, optimizer, epochs, fold_num):
        """Train model for one fold of cross-validation"""
        best_val_loss = float("inf")
        fold_train_losses = []
        fold_val_losses = []
        
        for epoch in range(1, epochs + 1):
            print(f"  Fold {fold_num} - Epoch {epoch}/{epochs}")
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, self.device)
            val_loss = validate(model, val_loader, criterion, self.device)
            
            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)
            
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Val   Loss: {val_loss:.4f}")
            
            # Save best model for this fold
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        return best_val_loss, fold_train_losses, fold_val_losses
    
    def run_cross_validation(self, dataset: USSegmentationDataset):
        """Run the complete k-fold cross-validation process"""
        # Setup cross-validation
        kfold, indices = self.setup_cross_validation(len(dataset))

        print(f"\nStarting {self.args.k_folds}-fold cross-validation...")
        
        for fold, (train_indices, val_indices) in enumerate(kfold.split(indices), 1):
            print(f"\n{'='*50}")
            print(f"FOLD {fold}/{self.args.k_folds}")
            print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
            print(f"{'='*50}")
            
            # Create subsets
            train_subset = Subset(dataset, train_indices.tolist())
            val_subset = Subset(dataset, val_indices.tolist())

            # Create data loaders
            train_loader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
            val_loader = DataLoader(val_subset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)
            
            # Create fresh model for this fold
            model = get_model().to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=self.args.learning_rate)
            
            # Train this fold
            best_val_loss, train_losses, val_losses = self.train_fold(
                model, train_loader, val_loader, criterion, optimizer, self.args.epochs, fold
            )
            
            # Store results
            fold_result = {
                'fold': fold,
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_size': len(train_indices),
                'val_size': len(val_indices)
            }
            self.all_fold_results.append(fold_result)
            
            # Track best overall model
            if best_val_loss < self.best_overall_val_loss:
                self.best_overall_val_loss = best_val_loss
                self.best_fold_model = model.state_dict().copy()
            
            print(f"Fold {fold} completed - Best validation loss: {best_val_loss:.4f}")
    
    def save_best_model(self):
        """Save the best model across all folds"""
        if self.best_fold_model is not None:
            torch.save(self.best_fold_model, self.args.checkpoint_path)
            print(f"\nSaved best model (val_loss: {self.best_overall_val_loss:.4f}) to {self.args.checkpoint_path}")
    
    def print_summary(self):
        """Print cross-validation summary"""
        print(f"\n{'='*60}")
        print("CROSS-VALIDATION SUMMARY")
        print(f"{'='*60}")
        
        val_losses = [result['best_val_loss'] for result in self.all_fold_results]
        mean_val_loss = np.mean(val_losses)
        std_val_loss = np.std(val_losses)
        
        print(f"Number of folds: {self.args.k_folds}")
        print(f"Mean validation loss: {mean_val_loss:.4f} ± {std_val_loss:.4f}")
        print(f"Best validation loss: {min(val_losses):.4f}")
        print(f"Worst validation loss: {max(val_losses):.4f}")
        
        print("Per-fold results:")
        for result in self.all_fold_results:
            print(f"  Fold {result['fold']}: {result['best_val_loss']:.4f} "
                  f"(train: {result['train_size']}, val: {result['val_size']})")
        
        print("\nCross-validation complete!")
    
    def run(self):
        """Main training pipeline"""
        dataset = self.create_dataset()
        self.run_cross_validation(dataset)
        self.save_best_model()
        self.print_summary()