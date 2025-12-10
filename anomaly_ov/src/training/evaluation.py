"""
Evaluation utilities for Anomaly OV fine-tuning.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, Any, Optional
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Setup paths for imports
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from src.utils.image_processing import process_anyres_image
from src.training.visualization import plot_roc_curve


def compute_metrics(
    model, 
    dataloader, 
    device: str, 
    split_name: str = "test",
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compute detailed metrics on a dataset.
    
    Args:
        model: OVAnomalyDetector model
        dataloader: DataLoader for the dataset
        device: Device to run inference on
        split_name: Name of the split
        save_dir: Directory to save ROC curve (optional)
    
    Returns:
        Dictionary with predictions, probabilities, labels, AUC, and confusion matrix
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Computing metrics on {split_name}"):
            patches_list = [patches.to(device) for patches in batch["patches"]]
            labels = batch["label"].to(device)
            
            predictions = model.get_anomaly_fetures_from_images(
                patches_list,
                with_attention_map=False
            )
            
            all_probs.extend(predictions.cpu().numpy().flatten())
            all_preds.extend((predictions > 0.5).float().cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    # Classification report
    print("\n" + "=" * 80)
    print(f"Classification Report ({split_name.upper()}):")
    print("=" * 80)
    print(classification_report(
        all_labels, all_preds, 
        target_names=['Normal', 'Anomaly'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("=" * 80)
    print(f"                 Predicted Normal  Predicted Anomaly")
    print(f"Actual Normal    {cm[0,0]:>16d}  {cm[0,1]:>17d}")
    print(f"Actual Anomaly   {cm[1,0]:>16d}  {cm[1,1]:>17d}")
    
    # ROC AUC
    auc = roc_auc_score(all_labels, all_probs)
    print(f"\nROC AUC Score: {auc:.4f}")
    
    # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    
    if save_dir:
        roc_path = os.path.join(save_dir, f'roc_curve_{split_name}.png')
        plot_roc_curve(fpr, tpr, auc, roc_path, split_name)
    
    return {
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels,
        'auc': auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }


def infer_single_image(
    model, 
    image_path: str, 
    image_processor, 
    image_grid_pinpoints: list,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Run inference on a single image using anyres preprocessing.
    
    Args:
        model: The OVAnomalyDetector model
        image_path: Path to the image file
        image_processor: Image processor for preprocessing
        image_grid_pinpoints: Grid pinpoints for anyres
        device: Device to run inference on
    
    Returns:
        Dictionary with prediction results and anomaly map
    """
    model.eval()
    
    # Load and preprocess image with anyres
    image = Image.open(image_path).convert("RGB")
    
    # Process with anyres strategy
    patches = process_anyres_image(image, image_processor, image_grid_pinpoints)
    patches = patches.to(device)
    num_patches = patches.shape[0]
    
    with torch.no_grad():
        # Get prediction and anomaly map
        patches_list = [patches]
        pred, _, anomaly_map = model.get_anomaly_fetures_from_images(
            patches_list,
            with_attention_map=True,
            anomaly_map_size=(384, 384)
        )
    
    pred_value = pred.item()
    pred_label = "Anomaly" if pred_value > 0.5 else "Normal"
    confidence = pred_value if pred_value > 0.5 else (1 - pred_value)
    
    # Get anomaly map - average across all patches
    anom_map = anomaly_map[:num_patches, 0].mean(dim=0).cpu().numpy()
    
    return {
        'prediction': pred_label,
        'confidence': confidence,
        'score': pred_value,
        'anomaly_map': anom_map,
        'num_patches': num_patches,
        'image': np.array(image)
    }


def evaluate_model(model, test_loader, config, save_results: bool = True) -> Dict[str, Any]:
    """
    Full evaluation of a model on the test set.
    
    Args:
        model: OVAnomalyDetector model
        test_loader: Test data loader
        config: Configuration object
        save_results: Whether to save results to disk
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Model Evaluation")
    print("=" * 80)
    
    save_dir = config.checkpoint_dir if save_results else None
    
    metrics = compute_metrics(
        model, 
        test_loader, 
        config.device, 
        split_name="test",
        save_dir=save_dir
    )
    
    return metrics
