"""
Metrics computation for anomaly-map vs inpainting mask correlation.

This module provides functions to compute metrics comparing explanation/vulnerability
maps against ground truth inpainting masks.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Any
from sklearn.metrics import roc_auc_score, average_precision_score


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
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
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


def load_mask_image(mask_path: str) -> Optional[np.ndarray]:
    """
    Load a mask image from file.
    
    Args:
        mask_path: Path to mask image file
        
    Returns:
        numpy array [H, W] in [0, 1] or None if failed
    """
    if not mask_path or not isinstance(mask_path, str):
        return None
    
    import os
    if not os.path.exists(mask_path):
        return None
    
    try:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        return mask.astype(np.float32) / 255.0
    except Exception:
        return None
