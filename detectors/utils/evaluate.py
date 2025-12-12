"""
Metrics computation for anomaly-map vs inpainting mask correlation.

This module provides functions to compute metrics comparing explanation/vulnerability
maps against ground truth inpainting masks.
"""
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_mask_anomaly_metrics(
    anomaly_map: Any,
    mask_image: Any,
    top_k: float = 0.10,
    white_threshold: Optional[float] = None,
    inpainted_is_white: bool = True,
    eps: float = 1e-8,
) -> Dict[str, float]:
    """
    Compute correlation-style metrics between an anomaly/explanation map and an inpainting mask.

    Metrics computed:
      - IoU (Intersection over Union) between mask and top-k anomalous regions (thresholded)
      - Anomaly mass fraction inside mask: sum(A * M) / sum(A)
      - ROC-AUC treating mask pixels as positives and raw anomaly as scores (no threshold)
      - PR-AUC (Average Precision)

    Args:
        anomaly_map: Anomaly/explanation map. Can be:
            - torch.Tensor with shape [H, W], [1, H, W], [C, H, W], [1, C, H, W]
            - numpy array with similar shapes
        mask_image: Ground truth inpainting mask. Can be:
            - torch.Tensor or numpy array
            - Shape [H, W], [1, H, W], [H, W, 3], [1, H, W, 3], etc.
            - Values in [0, 1] or [0, 255]
        top_k: Fraction of top anomalous pixels to consider for IoU (default: 0.10)
        white_threshold: Threshold to binarize mask. If None, uses Otsu's method.
        inpainted_is_white: If True, inpainted region is white (high values) in mask.
            If False, inpainted region is black (low values). Default: True
        eps: Small epsilon for numerical stability

    Returns:
        Dict with keys: 'iou_topk', 'mass_frac', 'roc_auc', 'pr_auc'
        Values are float or np.nan if not computable.
    """
    metrics = {"iou_topk": np.nan, "mass_frac": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    
    if anomaly_map is None or mask_image is None:
        return metrics
    
    # Import torch lazily to avoid hard dependency
    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        HAS_TORCH = False
    
    # --------------------------------------------------
    # Helper functions
    # --------------------------------------------------
    def to_numpy_2d(arr: Any) -> Optional[np.ndarray]:
        """Convert anomaly map to 2D numpy array [H, W]."""
        if HAS_TORCH and isinstance(arr, torch.Tensor):
            arr = arr.detach().float().cpu().numpy()
        else:
            arr = np.asarray(arr)
        
        arr = arr.astype(np.float32)
        
        # Handle various shapes
        while arr.ndim > 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[-1] == 3:
                # RGB -> grayscale
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            elif arr.shape[0] == 3:
                # [3, H, W] -> grayscale
                arr = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
            else:
                arr = arr[0]
        
        if arr.ndim != 2:
            return None
        
        return arr.astype(np.float32)
    
    def to_01(x: np.ndarray) -> np.ndarray:
        """Scale array to [0, 1] range."""
        x = x.astype(np.float32)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= 1.5 and xmin >= 0.0:
            return x
        if xmax <= 1.0 and xmin >= -1.0:  # likely [-1,1]
            return (x + 1.0) / 2.0
        if xmax > 1.5:  # likely [0,255]
            return x / 255.0
        rng = xmax - xmin
        return (x - xmin) / (rng + eps) if rng > eps else x - xmin
    
    def standardize_mask(m: Any) -> Optional[np.ndarray]:
        """Return mask as [H, W] float32 in [0, 1] grayscale."""
        if HAS_TORCH and isinstance(m, torch.Tensor):
            arr = m.detach().cpu().numpy()
        else:
            arr = np.asarray(m)
        
        arr = arr.astype(np.float32)
        
        # Squeeze batch dimensions
        while arr.ndim > 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            else:
                break
        
        # Handle channel dimension
        if arr.ndim == 3:
            if arr.shape[-1] == 3:
                # [H, W, 3] -> grayscale
                arr = to_01(arr)
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[0] == 3:
                # [3, H, W] -> grayscale
                arr = to_01(arr)
                arr = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
            elif arr.shape[0] == 1:
                arr = arr[0]
            else:
                arr = arr[0]
        
        if arr.ndim != 2:
            return None
        
        return to_01(arr).astype(np.float32)
    
    def compute_otsu_threshold(mv01: np.ndarray) -> float:
        """Compute Otsu threshold on a [0, 1] image.

        For binary masks (only 0 and 1 values), returns 0.5 to avoid
        Otsu returning 0.0 which would make all pixels positive.
        """
        # Check if mask is already binary (only contains ~0 and ~1 values)
        unique_vals = np.unique(mv01)
        if len(unique_vals) <= 2:
            # Binary mask - use fixed threshold at 0.5
            return 0.5
        
        thr_val, _ = cv2.threshold(
            (mv01 * 255.0).astype(np.uint8),
            0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return float(thr_val) / 255.0
    
    def auto_scale_threshold(th: float) -> float:
        """Scale threshold to [0, 1] if given in [0, 255]."""
        return float(th / 255.0) if th is not None and th > 1.5 else float(th)
    
    # --------------------------------------------------
    # Process inputs
    # --------------------------------------------------
    anom = to_numpy_2d(anomaly_map)
    mask = standardize_mask(mask_image)
    
    if anom is None or mask is None:
        return metrics
    
    # Resize anomaly map to mask size if different
    h_m, w_m = mask.shape
    h_a, w_a = anom.shape
    
    if h_m == 0 or w_m == 0 or h_a == 0 or w_a == 0:
        return metrics
    
    # Check aspect ratio - if very different, might be misaligned
    ratio_m = h_m / float(w_m)
    ratio_a = h_a / float(w_a) if w_a != 0 else 1.0
    if abs(ratio_m - ratio_a) / max(eps, ratio_m) > 0.25:
        # Still try, but might be problematic
        pass
    
    # Resize anomaly to mask resolution
    if (h_a, w_a) != (h_m, w_m):
        anom = cv2.resize(anom, (w_m, h_m), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    
    # Normalize anomaly map to [0, 1]
    min_v, max_v = float(anom.min()), float(anom.max())
    if max_v - min_v > eps:
        anom_norm = (anom - min_v) / (max_v - min_v)
    else:
        anom_norm = np.zeros_like(anom, dtype=np.float32)
    
    # Binarize mask
    thr01 = auto_scale_threshold(white_threshold) if white_threshold is not None else compute_otsu_threshold(mask)
    
    if inpainted_is_white:
        mask_bin = (mask >= thr01).astype(np.uint8)
    else:
        mask_bin = (mask < thr01).astype(np.uint8)
    
    flat_scores = anom_norm.reshape(-1)
    flat_mask = mask_bin.reshape(-1)
    
    # --------------------------------------------------
    # Compute metrics
    # --------------------------------------------------
    
    # 1. IoU with top-k of anomaly
    top_k = float(top_k)
    if not (0.0 < top_k <= 1.0):
        top_k = 0.10
    
    k = max(1, int(top_k * flat_scores.size))
    sorted_scores = np.sort(flat_scores)[::-1]
    thr_k = sorted_scores[k - 1] if k <= sorted_scores.size else sorted_scores[-1]
    anom_bin = (anom_norm >= thr_k).astype(np.uint8)
    
    intersection = int(np.sum(anom_bin * mask_bin))
    union = int(np.sum(((anom_bin + mask_bin) > 0).astype(np.uint8)))
    iou = (intersection / union) if union > 0 else 0.0
    metrics["iou_topk"] = float(iou)
    
    # 2. Mass fraction inside mask
    total_mass = float(np.sum(anom_norm))
    mass_frac = float(np.sum(anom_norm * mask_bin) / total_mass) if total_mass > eps else 0.0
    metrics["mass_frac"] = float(mass_frac)
    
    # 3. ROC-AUC & PR-AUC (only if both classes exist in mask)
    if np.any(flat_mask == 1) and np.any(flat_mask == 0):
        try:
            metrics["roc_auc"] = float(roc_auc_score(flat_mask, flat_scores))
        except Exception:
            metrics["roc_auc"] = np.nan
        
        try:
            metrics["pr_auc"] = float(average_precision_score(flat_mask, flat_scores))
        except Exception:
            metrics["pr_auc"] = np.nan
    
    return metrics


class MetricsAggregator:
    """
    Aggregator for collecting and averaging metrics across multiple samples.

    Usage:
        agg = MetricsAggregator()
        for sample in samples:
            metrics = compute_mask_anomaly_metrics(...)
            agg.update(metrics)

        print(agg.averages())
        agg.to_csv("metrics.csv")
    """
    
    def __init__(self):
        self.records = []  # List of individual metric dicts
        self.sums = {}
        self.counts = {}
    
    def update(self, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """
        Add a new set of metrics.

        Args:
            metrics: Dict with metric values (can contain np.nan)
            metadata: Optional dict with additional info (filename, image_type, etc.)
        """
        record = dict(metrics)
        if metadata:
            record.update(metadata)
        self.records.append(record)
        
        # Update running sums for averages
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.sums[k] = self.sums.get(k, 0.0) + float(v)
                self.counts[k] = self.counts.get(k, 0) + 1
    
    def averages(self) -> Dict[str, float]:
        """Return average values for all metrics."""
        return {
            k: (self.sums[k] / self.counts[k]) if self.counts.get(k, 0) > 0 else float('nan')
            for k in self.sums
        }
    
    def to_dataframe(self):
        """Convert records to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.records)
    
    def to_csv(self, path: str):
        """Save all records to CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.records)
        # Keep NaN as is (pandas will write them as empty strings in CSV)
        # But we can also write them explicitly if preferred
        df.to_csv(path, index=False, na_rep='NaN')  # Use 'NaN' representation for clarity
        return df
    
    def summary_str(self) -> str:
        """Return a formatted summary string of averages."""
        avg = self.averages()
        parts = []
        for k in sorted(avg.keys()):
            v = avg[k]
            if np.isnan(v):
                parts.append(f"{k}=nan")
            else:
                parts.append(f"{k}={v:.4f}")
        return ", ".join(parts)
    
    def __len__(self):
        return len(self.records)
    
    def __str__(self):
        return f"MetricsAggregator({len(self)} samples): {self.summary_str()}"


def compute_metrics(
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    topk_percents: List[float],
) -> Dict[str, float]:
    """
    Compute spatial correlation metrics between heatmap and ground truth mask.

    Args:
        heatmap: 2D float32 array (H, W) in [0, 1]
        gt_mask: 2D float32 array (H, W) in [0, 1]
        topk_percents: List of top-k percentages to compute IoU for

    Returns:
        Dict with metric values
    """
    all_metrics = {}
    
    for topk in topk_percents:
        topk_frac = topk / 100.0  # Convert percent to fraction
        
        metrics = compute_mask_anomaly_metrics(
            anomaly_map=heatmap,
            mask_image=gt_mask,
            top_k=topk_frac,
            inpainted_is_white=True,
        )
        
        # Add topk suffix if multiple topk values
        if len(topk_percents) > 1:
            for key, value in metrics.items():
                all_metrics[f"{key}@{topk}"] = value
        else:
            all_metrics.update(metrics)
    
    return all_metrics


# ============================================================================
# VISUALIZATION
# ============================================================================


def colorize_cam(cam: np.ndarray) -> np.ndarray:
    """
    Simple red colormap: map cam [1,H,W] -> [3,H,W] with red channel = cam, others = (1-cam)*0.
    """
    # Robust per-image normalization to [-1,1] using PyTorch
    max_abs = np.quantile(np.abs(cam).flatten(), 0.99)
    if max_abs > 0:
        cam /= max_abs
    
    red = np.clip(cam, 0, 1)
    blue = -np.clip(cam, -1, 0)
    # make red where cam > 0 and blue where < 0
    rgb = np.stack([red, (blue + red) / 3, blue]).transpose(1, 2, 0)
    return rgb


def create_visualization_grid(
    images: Dict[str, np.ndarray],
    exp_orig: Dict[str, np.ndarray],
    exp_adv: Dict[str, np.ndarray],
    vuln_maps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
    filename: str,
    attack_type: str,
    output_path: str,
    dpi: int = 150,
    detector_name: str = "",
) -> None:
    """
    Create and save a grid visualization.

    Grid layout:
        Columns = {real, samecat, diffcat}
        Rows = {image, exp_orig, exp_adv, vuln_map, gt_mask}

    Args:
        images: Dict mapping image_type to RGB image array
        exp_orig: Dict mapping image_type to original explanation map
        exp_adv: Dict mapping image_type to attacked explanation map
        vuln_maps: Dict mapping image_type to vulnerability map
        gt_masks: Dict mapping image_type to ground truth mask
        filename: Sample filename
        attack_type: Attack type name
        output_path: Path to save visualization
        dpi: DPI for saved image
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Column order
    columns = ['real', 'samecat', 'diffcat']
    # Row names
    row_names = ['Image', 'Exp Original', 'Exp Attacked', 'Vulnerability', 'GT Mask']
    
    # Create figure
    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.1, wspace=0.05)
    
    for col_idx, img_type in enumerate(columns):
        if img_type not in images:
            continue
        
        # Row 0: Original image
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(images[img_type])
        if col_idx == 0:
            ax.set_ylabel(row_names[0], fontsize=10)
        ax.set_title(img_type.capitalize(), fontsize=12)
        ax.axis('off')
        
        # Row 1: Original explanation
        ax = fig.add_subplot(gs[1, col_idx])
        if img_type in exp_orig and exp_orig[img_type] is not None:
            if detector_name == "WaveRep":
                ax.imshow(colorize_cam(exp_orig[img_type]))
            else:
                ax.imshow(exp_orig[img_type], cmap='jet', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[1], fontsize=10)
        ax.axis('off')
        
        # Row 2: Attacked explanation
        ax = fig.add_subplot(gs[2, col_idx])
        if img_type in exp_adv and exp_adv[img_type] is not None:
            if detector_name == "WaveRep":
                ax.imshow(colorize_cam(exp_adv[img_type]))
            else:
                ax.imshow(exp_adv[img_type], cmap='jet', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[2], fontsize=10)
        ax.axis('off')
        
        # Row 3: Vulnerability map
        ax = fig.add_subplot(gs[3, col_idx])
        if img_type in vuln_maps and vuln_maps[img_type] is not None:
            if detector_name == "WaveRep":
                ax.imshow(colorize_cam(vuln_maps[img_type]))
            else:
                ax.imshow(vuln_maps[img_type], cmap='jet', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[3], fontsize=10)
        ax.axis('off')
        
        # Row 4: Ground truth mask
        ax = fig.add_subplot(gs[4, col_idx])
        if img_type in gt_masks and gt_masks[img_type] is not None:
            ax.imshow(gt_masks[img_type], cmap='gray', vmin=0, vmax=1)
        else:
            # Black mask for real images
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[4], fontsize=10)
        ax.axis('off')
    
    # Add title
    fig.suptitle(f'{filename} - {attack_type.upper()}', fontsize=14, y=0.98)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
