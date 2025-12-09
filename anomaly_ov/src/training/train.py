#!/usr/bin/env python3
"""
Main training script for Anomaly OV fine-tuning.

Usage:
    python train.py [options]                    # From project root
    python src/training/train.py [options]       # Direct execution

Examples:
    # Basic training
    python train.py

    # Resume from checkpoint
    python train.py --resume --resume-path ./outputs/checkpoints/checkpoint_epoch_1.pt

    # Custom settings
    python train.py --epochs 10 --batch-size 8 --lr 5e-5
"""

import os
import sys

# Determine the project root and add to path
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
_project_root = os.path.dirname(_src_dir)

# Add project root to path for imports
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add src dir to path for local imports when running directly
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)

import torch
from src.models.custom_anomaly_detector import OVAnomalyDetector

# Use absolute imports that work from any location
from src.training.config import Config, load_config_from_args
from src.training.dataset import create_dataloaders
from src.training.trainer import Trainer
from src.training.evaluation import evaluate_model
from src.training.visualization import plot_training_history, visualize_predictions


def main():
    """Main training function."""
    print("=" * 80)
    print("Anomaly OV Fine-tuning")
    print("=" * 80)
    
    # Load configuration
    try:
        config = load_config_from_args()
    except SystemExit:
        # If no args provided, use defaults
        config = Config()
    
    config.print_config()
    config.verify_dataset()
    
    # Load model
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    
    if config.resume_from_checkpoint and config.resume_checkpoint_path and os.path.exists(config.resume_checkpoint_path):
        print(f"Resuming from checkpoint: {config.resume_checkpoint_path}")
        detector, image_processor = OVAnomalyDetector.load_from_checkpoint(
            config.resume_checkpoint_path,
            device=config.device
        )
    else:
        print(f"Loading from initial checkpoint: {config.initial_checkpoint}")
        detector, image_processor = OVAnomalyDetector.load_from_checkpoint(
            config.initial_checkpoint,
            device=config.device
        )
    
    detector = detector.to(config.device)
    print(f"Model loaded on {config.device}")
    
    # Create dataloaders
    print("\n" + "=" * 80)
    print("Creating datasets...")
    print("=" * 80)
    
    train_loader, eval_loader, test_loader, train_dataset, eval_dataset, test_dataset = \
        create_dataloaders(config, image_processor)
    
    # Create trainer
    trainer = Trainer(
        model=detector,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        test_dataset=test_dataset,
        image_processor=image_processor
    )
    
    # Train
    history = trainer.train()
    
    # Plot training history
    history_path = os.path.join(config.checkpoint_dir, 'training_history.png')
    plot_training_history(history, history_path)
    
    # Load best model for evaluation
    print("\n" + "=" * 80)
    print("Loading best model for final evaluation...")
    print("=" * 80)
    
    best_model_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    best_detector, _ = OVAnomalyDetector.load_from_checkpoint(
        best_model_path,
        device=config.device
    )
    best_detector.eval()
    
    # Evaluate on test set
    test_metrics = evaluate_model(best_detector, test_loader, config)
    
    # Visualize predictions on test set
    print("\nGenerating final visualizations...")
    visualize_predictions(
        best_detector, 
        test_dataset, 
        config,
        num_samples=config.num_visualization_samples,
        split_name="test_final"
    )
    
    # Save training config with results
    trainer.save_training_config(
        test_metrics=test_metrics,
        dataset_sizes={
            "train": len(train_dataset),
            "eval": len(eval_dataset),
            "test": len(test_dataset)
        }
    )
    
    # Export final model
    final_model_path = os.path.join(config.checkpoint_dir, "finetuned_anomaly_detector.pt")
    best_detector.save_checkpoint(final_model_path)
    print(f"\nFinal model exported to: {final_model_path}")
    print(f"File size: {os.path.getsize(final_model_path) / (1024**2):.2f} MB")
    
    print("\n" + "=" * 80)
    print("Fine-tuning complete!")
    print(f"  Best model: {best_model_path}")
    print(f"  Test AUC: {test_metrics['auc']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
