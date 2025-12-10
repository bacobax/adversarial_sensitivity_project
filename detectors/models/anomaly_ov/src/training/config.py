"""
Configuration for Anomaly OV fine-tuning.
"""

import json
import os
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

# Determine project root (parent of src folder)
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
PROJECT_ROOT = os.path.dirname(_src_dir)


def _resolve_path(path: str) -> str:
    """Resolve path relative to project root if not absolute."""
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


@dataclass
class Config:
    """Configuration class for fine-tuning the Anomaly OV model."""
    
    # Dataset paths (relative to project root)
    data_root: str = "./finetune_dataset"
    
    # Model configuration
    vision_tower_name: str = "google/siglip-so400m-patch14-384"
    anomaly_expert_path: str = "./weights/pretrained_expert_05b.pth"
    initial_checkpoint: str = "./weights/zs_checkpoint.pt"
    
    # Training hyperparameters
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Loss hyperparameters
    loss_tau: float = 1.0
    loss_pos_weight: float = 1.5
    loss_margin: float = 0.0
    
    # Logging
    tensorboard_log_dir: str = "./outputs/logs"
    
    # Device configuration
    device: str = field(default_factory=lambda: "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32
    
    # Checkpointing
    checkpoint_dir: str = "./outputs/checkpoints"
    save_every: int = 1
    
    # Early stopping (based on eval accuracy)
    early_stopping: bool = True
    early_stopping_patience: int = 5  # Number of epochs without improvement before stopping
    early_stopping_min_delta: float = 0.1  # Minimum accuracy improvement to count as improvement (in %)
    
    # Resume training
    resume_from_checkpoint: bool = False
    resume_checkpoint_path: Optional[str] = None
    
    # Dataset limits (number of image pairs, each pair = 2 samples: real + inpainted)
    train_max_pairs: Optional[int] = 300
    eval_max_pairs: Optional[int] = 50
    test_max_pairs: Optional[int] = 50
    
    # Image configuration
    image_size: int = 384
    
    # AnyRes configuration (loaded from config.json)
    config_path: str = "./config.json"
    image_aspect_ratio: str = "anyres_max_9"
    image_grid_pinpoints: List = field(default_factory=list)
    
    # Visualization
    visualize_every_epoch: bool = True
    num_visualization_samples: int = 8
    
    # DataLoader settings
    num_workers: int = 0  # Set to 0 for MPS compatibility
    pin_memory: bool = False
    
    # Data augmentation settings (to reduce brightness/color bias)
    use_augmentation: bool = True
    augment_config: dict = field(default_factory=lambda: {
        'brightness_range': (0.5, 1.5),
        'contrast_range': (0.5, 1.5),
        'saturation_range': (0.5, 1.5),
        'gamma_range': (0.5, 2.0),
        'invert_prob': 0.1,
        'grayscale_prob': 0.1,
        'channel_shuffle_prob': 0.1,
        'solarize_prob': 0.1,
        'solarize_threshold': 128,
        'equalize_prob': 0.1,
        'flip_prob': 0.5,
        'rotation_prob': 0.5,
        'rotation_degrees': 90
    })
    
    def __post_init__(self):
        """Load additional configuration after initialization and resolve paths."""
        # Resolve all paths relative to project root
        self.data_root = _resolve_path(self.data_root)
        self.anomaly_expert_path = _resolve_path(self.anomaly_expert_path)
        self.initial_checkpoint = _resolve_path(self.initial_checkpoint)
        self.tensorboard_log_dir = _resolve_path(self.tensorboard_log_dir)
        self.checkpoint_dir = _resolve_path(self.checkpoint_dir)
        self.config_path = _resolve_path(self.config_path)
        
        if self.resume_checkpoint_path:
            self.resume_checkpoint_path = _resolve_path(self.resume_checkpoint_path)
        
        # Load anyres configuration from config.json if it exists
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                model_config = json.load(f)
            self.image_aspect_ratio = model_config.get("image_aspect_ratio", self.image_aspect_ratio)
            self.image_grid_pinpoints = model_config.get("image_grid_pinpoints", self.image_grid_pinpoints)
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
    
    def verify_dataset(self):
        """Verify dataset structure and print statistics."""
        print("\nDataset structure:")
        for split in ['train', 'eval', 'test']:
            split_path = Path(self.data_root) / split
            if split_path.exists():
                inpainted_count = len(list((split_path / "inpainted").glob("*.png"))) + \
                                  len(list((split_path / "inpainted").glob("*.jpg")))
                print(f"  {split}/: {inpainted_count} images")
            else:
                print(f"  {split}/: NOT FOUND")
    
    def print_config(self):
        """Print configuration summary."""
        print("=" * 80)
        print("Configuration")
        print("=" * 80)
        print(f"Device: {self.device}")
        print(f"Data root: {self.data_root}")
        print(f"Batch size: {self.batch_size}")
        print(f"Num epochs: {self.num_epochs}")
        print(f"Learning rate: {self.learning_rate}")
        print(f"Weight decay: {self.weight_decay}")
        print(f"Image aspect ratio: {self.image_aspect_ratio}")
        print(f"Grid pinpoints: {len(self.image_grid_pinpoints)} configurations")
        print(f"Resume from checkpoint: {self.resume_from_checkpoint}")
        if self.resume_from_checkpoint and self.resume_checkpoint_path:
            print(f"  Checkpoint path: {self.resume_checkpoint_path}")
        print("=" * 80)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for saving."""
        return {
            "vision_tower_name": self.vision_tower_name,
            "anomaly_expert_path": self.anomaly_expert_path,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "image_size": self.image_size,
            "image_aspect_ratio": self.image_aspect_ratio,
            "num_grid_pinpoints": len(self.image_grid_pinpoints),
            "loss_tau": self.loss_tau,
            "loss_pos_weight": self.loss_pos_weight,
            "loss_margin": self.loss_margin,
        }


def load_config_from_args():
    """Load configuration from command line arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune Anomaly OV model")
    
    # Dataset
    parser.add_argument("--data-root", type=str, default="./finetune_dataset",
                        help="Root directory for dataset")
    
    # Model
    parser.add_argument("--initial-checkpoint", type=str, default="./zs_checkpoint.pt",
                        help="Path to initial model checkpoint")
    parser.add_argument("--vision-tower", type=str, default="google/siglip-so400m-patch14-384",
                        help="Vision tower name")
    
    # Training
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    
    # Loss
    parser.add_argument("--loss-tau", type=float, default=1.0, help="Temperature for BCE loss")
    parser.add_argument("--loss-pos-weight", type=float, default=1.5, help="Positive class weight")
    parser.add_argument("--loss-margin", type=float, default=0.0, help="Margin for loss")
    
    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory to save checkpoints")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    
    # Early stopping
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.1,
                        help="Minimum accuracy improvement to count as improvement (in %%)")
    
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--resume-path", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Dataset limits (image pairs, each pair = 2 samples)
    parser.add_argument("--train-pairs", type=int, default=250,
                        help="Max training image pairs (each pair = 2 samples)")
    parser.add_argument("--eval-pairs", type=int, default=50,
                        help="Max eval image pairs")
    parser.add_argument("--test-pairs", type=int, default=50,
                        help="Max test image pairs")
    
    # Visualization
    parser.add_argument("--no-visualize", action="store_true",
                        help="Disable visualization during training")
    parser.add_argument("--vis-samples", type=int, default=8,
                        help="Number of visualization samples")
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="TensorBoard log directory")
    
    # Config file
    parser.add_argument("--config-json", type=str, default="./config.json",
                        help="Path to model config JSON")
    
    # Data augmentation
    parser.add_argument("--no-augmentation", action="store_true",
                        help="Disable data augmentation during training")
    
    args = parser.parse_args()
    
    # Create config from args
    config = Config(
        data_root=args.data_root,
        vision_tower_name=args.vision_tower,
        initial_checkpoint=args.initial_checkpoint,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        loss_tau=args.loss_tau,
        loss_pos_weight=args.loss_pos_weight,
        loss_margin=args.loss_margin,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.patience,
        early_stopping_min_delta=args.min_delta,
        resume_from_checkpoint=args.resume,
        resume_checkpoint_path=args.resume_path,
        train_max_pairs=args.train_pairs if args.train_pairs > 0 else None,
        eval_max_pairs=args.eval_pairs if args.eval_pairs > 0 else None,
        test_max_pairs=args.test_pairs if args.test_pairs > 0 else None,
        visualize_every_epoch=not args.no_visualize,
        num_visualization_samples=args.vis_samples,
        tensorboard_log_dir=args.log_dir,
        config_path=args.config_json,
        use_augmentation=not args.no_augmentation,
    )
    
    return config
