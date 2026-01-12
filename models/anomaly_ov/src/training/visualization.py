"""
Visualization utilities for Anomaly OV fine-tuning.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Setup paths for imports
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.utils.visualization import visualize_anomaly_gradcam


def visualize_predictions(
    model, 
    dataset, 
    config,
    num_samples: int = 8, 
    split_name: str = "test",
    save_dir: Optional[str] = None
):
    """
    Visualize model predictions with anomaly maps using anyres preprocessing.
    
    Args:
        model: OVAnomalyDetector model
        dataset: Dataset to sample from
        config: Configuration object
        num_samples: Number of samples to visualize
        split_name: Name of the split (for saving)
        save_dir: Directory to save visualizations (default: config.checkpoint_dir)
    """
    model.eval()
    device = config.device
    save_dir = save_dir or config.checkpoint_dir
    
    with torch.no_grad():
        for i in range(num_samples):
            # Get random sample
            idx = np.random.randint(0, len(dataset))
            sample = dataset[idx]
            
            # Prepare input
            patches = sample["patches"].to(device)
            label = sample["label"].item()
            num_patches = sample["num_patches"]
            gt_mask = sample["mask"]
            
            # Get prediction and anomaly map
            patches_list = [patches]
            pred, _, anomaly_map = model.get_anomaly_fetures_from_images(
                patches_list,
                with_attention_map=True,
                anomaly_map_size=(384, 384)
            )
            
            pred_value = pred.item()
            pred_label = 1 if pred_value > 0.5 else 0
            
            # Prepare data for visualize_anomaly_gradcam
            image_tensor = patches.unsqueeze(0)
            anomaly_maps = anomaly_map[:num_patches]
            
            # Resize ground truth mask
            gt_mask_resized = F.interpolate(
                gt_mask.unsqueeze(0).unsqueeze(0),
                size=(384, 384),
                mode='bilinear',
                align_corners=False
            )
            mask_views = gt_mask_resized.repeat(1, num_patches, 1, 1, 1)
            
            # Create metrics label
            gt_str = "Anomaly" if label == 1 else "Normal"
            pred_str = f"Pred: {pred_value:.3f} ({'✓' if pred_label == label else '✗'})"
            metrics_label = f"GT: {gt_str} | {pred_str}"
            
            # Call visualize_anomaly_gradcam
            fig = visualize_anomaly_gradcam(
                image_tensor=image_tensor,
                anomaly_maps=anomaly_maps,
                mask_image=mask_views,
                metrics_label=metrics_label,
                alpha=0.5,
                figsize=(20, 12)
            )
            
            # Save the figure
            save_path = os.path.join(save_dir, f'predictions_{split_name}_sample{i}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Visualizations saved to: {save_dir}/predictions_{split_name}_sample*.png")


def plot_training_history(history: dict, save_path: str):
    """
    Plot training history curves.
    
    Args:
        history: Dictionary with train_loss, train_acc, eval_loss, eval_acc lists
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['eval_loss'], label='Eval Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Evaluation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(epochs, history['eval_acc'], label='Eval Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Evaluation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Training history plot saved to: {save_path}")


def plot_roc_curve(fpr, tpr, auc, save_path: str, split_name: str = "test"):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc: Area under curve
        save_path: Path to save the plot
        split_name: Name of the split
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {split_name.upper()} Set (AnyRes)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {save_path}")
