"""
Training logic for Anomaly OV fine-tuning.
"""

import os
import sys
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple

# Setup paths for imports
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.training.losses import AnomalyStage1Loss
from src.training.visualization import visualize_predictions


class Trainer:
    """Trainer class for Anomaly OV fine-tuning."""
    
    def __init__(
        self,
        model,
        config,
        train_loader,
        eval_loader,
        test_dataset=None,
        image_processor=None
    ):
        """
        Initialize the trainer.
        
        Args:
            model: OVAnomalyDetector model
            config: Configuration object
            train_loader: Training data loader
            eval_loader: Evaluation data loader
            test_dataset: Test dataset for visualization (optional)
            image_processor: Image processor for visualization (optional)
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.test_dataset = test_dataset
        self.image_processor = image_processor
        
        # Setup model for training
        self._setup_model()
        
        # Initialize loss
        self.criterion = AnomalyStage1Loss(
            tau=config.loss_tau,
            pos_weight=config.loss_pos_weight,
            margin=config.loss_margin
        )
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        steps_per_epoch = len(train_loader)
        T_0 = max(1, steps_per_epoch // 2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=T_0,
            T_mult=1,
            eta_min=1e-6
        )
        
        # Initialize TensorBoard
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(config.tensorboard_log_dir, f"anomaly_finetuning_{timestamp}")
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs: {log_dir}")
        
        # Training state
        self.start_epoch = 0
        self.best_eval_loss = float('inf')
        self.best_eval_acc = 0.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'eval_loss': [],
            'eval_acc': []
        }
        
        # Load checkpoint if resuming
        if config.resume_from_checkpoint and config.resume_checkpoint_path:
            self._load_checkpoint(config.resume_checkpoint_path)
    
    def _setup_model(self):
        """Setup model for training: freeze vision encoder, enable anomaly expert."""
        # Freeze vision encoder
        self.model.vision_encoder.requires_grad_(False)
        self.model.vision_encoder.eval()
        
        # Enable anomaly expert
        self.model.anomaly_expert.requires_grad_(True)
        self.model.anomaly_expert.train()
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nModel setup:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Percentage trainable: {100 * trainable_params / total_params:.2f}%")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        import re
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device)
        
        # Try to get epoch from checkpoint data first, then fall back to filename parsing
        epoch_from_checkpoint = checkpoint.get('epoch', None)
        
        # Parse epoch from filename (e.g., checkpoint_epoch_1.pt -> epoch 1)
        epoch_from_filename = None
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'checkpoint_epoch_(\d+)\.pt', filename)
        if match:
            epoch_from_filename = int(match.group(1))
        
        # Use filename-based epoch as primary (since it's what user sees), 
        # fall back to checkpoint data, then default to 0
        if epoch_from_filename is not None:
            self.start_epoch = epoch_from_filename
            print(f"  Epoch parsed from filename: {epoch_from_filename}")
        elif epoch_from_checkpoint is not None:
            self.start_epoch = epoch_from_checkpoint
            print(f"  Epoch loaded from checkpoint data: {epoch_from_checkpoint}")
        else:
            self.start_epoch = 0
            print(f"  Warning: Could not determine epoch, starting from 0")
        
        self.best_eval_loss = checkpoint.get('best_eval_loss', float('inf'))
        self.best_eval_acc = checkpoint.get('best_eval_acc', 0.0)
        self.epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        
        if 'history' in checkpoint and checkpoint['history']:
            self.history = checkpoint['history']
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("  Optimizer state loaded")
        
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("  Scheduler state loaded")
        
        print(f"  Resuming training from epoch {self.start_epoch + 1}")
        if self.best_eval_loss != float('inf'):
            print(f"  Best eval loss so far: {self.best_eval_loss:.4f}")
    
    def _save_checkpoint(self, epoch: int, checkpoint_path: str, is_best: bool = False):
        """Save training checkpoint."""
        self.model.save_checkpoint(
            checkpoint_path,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            best_eval_loss=self.best_eval_loss,
            best_eval_acc=self.best_eval_acc,
            epochs_without_improvement=self.epochs_without_improvement,
            history=self.history
        )
        if is_best:
            print(f"  ✓ New best model saved! (Eval Acc: {self.best_eval_acc:.2f}%, Loss: {self.best_eval_loss:.4f})")
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.anomaly_expert.train()
        self.model.vision_encoder.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} - Training")
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            patches_list = [patches.to(self.config.device) for patches in batch["patches"]]
            labels = batch["label"].to(self.config.device)
            num_patches = batch["num_patches"]
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model.get_anomaly_fetures_from_images(
                patches_list,
                with_attention_map=False
            )
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            predicted = (predictions > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Clear intermediate tensors to free memory
            del patches_list, predictions
            
            # Periodically clear MPS cache to prevent memory fragmentation
            if self.config.device == "mps" and (batch_idx + 1) % 20 == 0:
                torch.mps.empty_cache()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            self.writer.add_scalar('Train/LR', self.scheduler.get_last_lr()[0], global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.eval_loader, desc="Validation")
            for batch in pbar:
                patches_list = [patches.to(self.config.device) for patches in batch["patches"]]
                labels = batch["label"].to(self.config.device)
                
                predictions = self.model.get_anomaly_fetures_from_images(
                    patches_list,
                    with_attention_map=False
                )
                
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        avg_loss = total_loss / len(self.eval_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self):
        """Run the full training loop."""
        print("=" * 80)
        print("Starting training...")
        print(f"  Epochs: {self.start_epoch + 1} to {self.config.num_epochs}")
        print("=" * 80)
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 80)
            
            # Clear MPS cache before each epoch to prevent memory fragmentation
            if self.config.device == "mps":
                import torch
                torch.mps.empty_cache()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            eval_loss, eval_acc = self.validate()
            
            # Log to TensorBoard
            self.writer.add_scalar('Epoch/TrainLoss', train_loss, epoch)
            self.writer.add_scalar('Epoch/TrainAcc', train_acc, epoch)
            self.writer.add_scalar('Epoch/EvalLoss', eval_loss, epoch)
            self.writer.add_scalar('Epoch/EvalAcc', eval_acc, epoch)
            
            # Visualize predictions
            if self.config.visualize_every_epoch and self.test_dataset:
                visualize_predictions(
                    self.model, 
                    self.test_dataset, 
                    self.config,
                    num_samples=self.config.num_visualization_samples,
                    split_name=f"test_epoch{epoch+1}"
                )
            
            # Print summary
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['eval_loss'].append(eval_loss)
            self.history['eval_acc'].append(eval_acc)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                checkpoint_path = os.path.join(
                    self.config.checkpoint_dir, 
                    f"checkpoint_epoch_{epoch+1}.pt"
                )
                self._save_checkpoint(epoch + 1, checkpoint_path)
                print(f"  Checkpoint saved: {checkpoint_path}")
            
            # Save best model (based on eval accuracy)
            if eval_acc > self.best_eval_acc + self.config.early_stopping_min_delta:
                self.best_eval_acc = eval_acc
                self.best_eval_loss = eval_loss
                self.best_epoch = epoch + 1
                self.epochs_without_improvement = 0
                best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                self._save_checkpoint(epoch + 1, best_model_path, is_best=True)
            else:
                self.epochs_without_improvement += 1
                print(f"  No improvement for {self.epochs_without_improvement} epoch(s)")
            
            # Early stopping check
            if self.config.early_stopping and self.epochs_without_improvement >= self.config.early_stopping_patience:
                print(f"\n" + "=" * 80)
                print(f"Early stopping triggered! No improvement for {self.config.early_stopping_patience} epochs.")
                print(f"Best model at epoch {self.best_epoch} with eval acc: {self.best_eval_acc:.2f}%")
                break
        
        print("\n" + "=" * 80)
        print("Training completed!")
        print(f"Best model at epoch {self.best_epoch} with eval acc: {self.best_eval_acc:.2f}% (loss: {self.best_eval_loss:.4f})")
        
        # Close TensorBoard writer
        self.writer.close()
        
        return self.history
    
    def save_training_config(self, test_metrics: Optional[Dict] = None, dataset_sizes: Optional[Dict] = None):
        """Save training configuration to JSON."""
        config_dict = self.config.to_dict()
        config_dict.update({
            "best_epoch": self.best_epoch,
            "best_eval_loss": float(self.best_eval_loss),
        })
        
        if test_metrics:
            config_dict["test_auc"] = float(test_metrics.get('auc', 0))
        
        if dataset_sizes:
            config_dict["dataset_splits"] = dataset_sizes
        
        config_path = os.path.join(self.config.checkpoint_dir, "training_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
        
        print(f"Training configuration saved to: {config_path}")
