# Metrics computation for anomaly-map vs inpainting mask correlation


import numpy as np


import cv2


from typing import Dict, Optional, Tuple


from sklearn.metrics import roc_auc_score, average_precision_score




def compute_mask_anomaly_metrics(
    anomaly_maps: Optional["torch.Tensor"],
    mask_image,
    top_k: float = 0.10,
    aggregate: str = "mean",  # aggregation across crops: 'mean' | 'max' | 'view0'
    white_threshold: Optional[float] = None,
    inpainted_is_white: bool = False,
    eps: float = 1e-8,
 ) -> Dict[str, float]:
    """
    Compute correlation-style metrics between per-crop anomaly maps and per-crop inpainting masks.
    For each crop (view), compute:
      - IoU (Intersection over Union) between mask and top-k anomalous regions (thresholded)
      - Anomaly mass fraction inside mask: sum(A * M) / sum(A)
      - ROC-AUC treating mask pixels as positives and raw anomaly as scores (no threshold)
      - PR-AUC (Average Precision)
    Then aggregate across crops (views) by 'mean' (default), 'max', or use 'view0'.
    
    Robust handling:
    - Accepts mask formats [1,V,C,H,W], [V,C,H,W], [V,H,W,3], [V,H,W] and converts to [V,H,W] grayscale in [0,1].
    - Accepts anomaly maps in shapes like [V,1,h,w], [V,h,w], or [1,V,1,h,w] and converts to [V,h,w].
    - Uses Otsu threshold when `white_threshold` is None; otherwise respects the provided threshold (auto-scaled).
    - Resizes anomaly to mask resolution after an aspect-ratio sanity check.
    - Avoids NaNs for IoU/mass (falls back to 0 when undefined). ROC/PR remain NaN when not computable.
    - Polarity is determined only by `inpainted_is_white` (no auto-detection).
    """
    metrics = {"iou_topk": np.nan, "mass_frac": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    if anomaly_maps is None or mask_image is None:
        return metrics
    
    import torch  # local import to avoid hard dependency at import-time


    # ------------------------------
    # Helpers
    # ------------------------------
    def to_numpy_anom(am) -> Optional[np.ndarray]:
        # Convert anomaly maps to [V, h, w] float32
        if isinstance(am, torch.Tensor):
            arr = am.detach().float().cpu().numpy()
        else:
            arr = np.asarray(am)
        if arr.ndim == 5 and arr.shape[0] == 1:   # [1, V, 1, h, w] -> [V, 1, h, w]
            arr = arr[0]
        if arr.ndim == 4 and arr.shape[1] == 1:   # [V, 1, h, w] -> [V, h, w]
            arr = arr[:, 0]
        elif arr.ndim == 3:                       # [V, h, w]
            pass
        else:
            return None
        return arr.astype(np.float32)


    def to_01(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax <= 1.5 and xmin >= 0.0:
            return x
        if xmax <= 1.0 and xmin >= -1.0:  # likely [-1,1]
            return (x + 1.0) / 2.0
        if xmax > 1.5:  # likely [0,255]
            return x / 255.0
        rng = xmax - xmin
        return (x - xmin) / (rng + 1e-8)


    def standardize_masks(m) -> Optional[np.ndarray]:
        """Return masks as [V,H,W] float32 in [0,1] grayscale."""
        if isinstance(m, torch.Tensor):
            arr = m.detach().cpu().numpy()
        else:
            arr = np.asarray(m)
        # Squeeze batch dim if present: [1,V,C,H,W] -> [V,C,H,W]
        if arr.ndim == 5 and arr.shape[0] == 1:
            arr = arr[0]
        # [V,C,H,W]
        if arr.ndim == 4:
            V, C = arr.shape[0], arr.shape[1]
            if C == 3:  # RGB -> grayscale
                ch = to_01(arr)  # scale to [0,1] per-array
                gray = 0.299 * ch[:, 0] + 0.587 * ch[:, 1] + 0.114 * ch[:, 2]  # [V,H,W]
                return gray.astype(np.float32)
            elif C == 1:  # single channel
                return to_01(arr[:, 0]).astype(np.float32)
            else:  # unknown channels -> take first and scale
                return to_01(arr[:, 0]).astype(np.float32)
        # [V,H,W,3]
        if arr.ndim == 4 and arr.shape[-1] == 3:
            ch = to_01(arr)
            gray = 0.299 * ch[..., 0] + 0.587 * ch[..., 1] + 0.114 * ch[..., 2]  # [V,H,W]
            return gray.astype(np.float32)
        # [V,H,W]
        if arr.ndim == 3:
            return to_01(arr).astype(np.float32)
        return None


    def auto_scale_threshold(th: float) -> float:
        # Accept thresholds provided in either [0,1] or [0,255]
        return float(th / 255.0) if th is not None and th > 1.5 else float(th)


    def compute_otsu_threshold(mv01: np.ndarray) -> float:
        # Otsu on uint8 returns (ret, thresh_img)
        thr_val, _ = cv2.threshold((mv01 * 255.0).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return float(thr_val) / 255.0


    # ------------------------------
    # Standardize inputs
    # ------------------------------
    anom = to_numpy_anom(anomaly_maps)  # [V,h,w]
    mask_views = standardize_masks(mask_image)  # [V,H,W] in [0,1]
    if anom is None or mask_views is None:
        return metrics


    V = anom.shape[0]
    if mask_views.shape[0] != V:
        # If a broadcast-like mismatch happens, try simple repeat/slice safeguards
        if mask_views.shape[0] == 1:
            mask_views = np.repeat(mask_views, V, axis=0)
        else:
            V = min(V, mask_views.shape[0])
            anom = anom[:V]
            mask_views = mask_views[:V]


    # Prepare aggregation lists
    iou_list, mass_list, roc_list, pr_list = [], [], [], []


    # Normalize top_k and aggregation option
    top_k = float(top_k)
    if not (0.0 < top_k <= 1.0):
        top_k = 0.10
    aggregate = str(aggregate).lower()


    # Precompute provided threshold if any (scaled to [0,1])
    thr01_global = auto_scale_threshold(white_threshold) if white_threshold is not None else None


    is_white = bool(inpainted_is_white)


    for v in range(V):
        mask_np = mask_views[v]  # [H,W] in [0,1]
        anom_v  = anom[v]       # [h,w]
        # Simple aspect check before resizing to avoid severe warping
        h_m, w_m = mask_np.shape
        h_a, w_a = anom_v.shape[-2], anom_v.shape[-1]
        ratio_m, ratio_a = h_m / float(w_m), h_a / float(w_a) if w_a != 0 else 1.0
        if w_m == 0 or h_m == 0 or w_a == 0 or h_a == 0:
            continue  # skip invalid
        if abs(ratio_m - ratio_a) / max(1e-6, ratio_m) > 0.20:
            # Likely misalignment; skip this view
            continue


        # Resize anomaly to mask size for per-pixel ops
        anom_resized = cv2.resize(anom_v, (w_m, h_m), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        # Normalize anomaly map to [0,1]
        min_v, max_v = float(anom_resized.min()), float(anom_resized.max())
        if max_v - min_v > eps:
            anom_norm = (anom_resized - min_v) / (max_v - min_v)
        else:
            anom_norm = np.zeros_like(anom_resized, dtype=np.float32)


        # Threshold mask: use provided threshold or Otsu; apply polarity via inpainted_is_white
        thr01 = thr01_global if thr01_global is not None else compute_otsu_threshold(mask_np)
        mask_bin = (mask_np >= thr01).astype(np.uint8) if is_white else (mask_np < thr01).astype(np.uint8)
        # Ensure mask has both classes for ROC/PR checks
        flat_scores = anom_norm.reshape(-1)
        flat_mask = mask_bin.reshape(-1)


        # IoU with top-k of anomaly
        k = max(1, int(top_k * flat_scores.size))
        # Deterministic threshold selection for top-k
        sorted_scores = np.sort(flat_scores)[::-1]
        thr_k = sorted_scores[k - 1] if k <= sorted_scores.size else sorted_scores[-1]
        anom_bin = (anom_norm >= thr_k).astype(np.uint8)
        intersection = int(np.sum(anom_bin * mask_bin))
        union = int(np.sum(((anom_bin + mask_bin) > 0).astype(np.uint8)))
        iou = (intersection / union) if union > 0 else 0.0  # avoid NaN
        iou_list.append(float(iou))


        # Mass fraction inside mask
        total_mass = float(np.sum(anom_norm))
        mass_frac = float(np.sum(anom_norm * mask_bin) / total_mass) if total_mass > eps else 0.0
        mass_list.append(mass_frac)


        # ROC-AUC & PR-AUC only if both classes exist in mask
        if np.any(flat_mask == 1) and np.any(flat_mask == 0):
            try:
                roc_list.append(float(roc_auc_score(flat_mask, flat_scores)))
            except Exception:
                roc_list.append(np.nan)
            try:
                pr_list.append(float(average_precision_score(flat_mask, flat_scores)))
            except Exception:
                pr_list.append(np.nan)
        else:
            roc_list.append(np.nan)
            pr_list.append(np.nan)


    # Aggregate across views
    def agg(vals):
        arr = np.array(vals, dtype=np.float32)
        if aggregate == 'view0':
            return float(arr[0]) if arr.size > 0 and not np.isnan(arr[0]) else np.nan
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            return np.nan
        if aggregate == 'max':
            return float(np.max(valid))
        return float(np.mean(valid))


    metrics["iou_topk"] = agg(iou_list)
    metrics["mass_frac"] = agg(mass_list)
    metrics["roc_auc"] = agg(roc_list)
    metrics["pr_auc"] = agg(pr_list)
    return metrics

class MetricsAverages:
    def __init__(self):
        self.sums = {}
        self.counts = {}
    def update(self, metrics: dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                self.sums[k] = self.sums.get(k, 0.0) + float(v)
                self.counts[k] = self.counts.get(k, 0) + 1
    def averages(self):
        return {k: (self.sums[k] / self.counts[k]) if self.counts.get(k,0) > 0 else float('nan') for k in self.sums}
    def __str__(self):
        avg = self.averages()
        return ", ".join(f"{k}={avg[k]:.4f}" if not np.isnan(avg[k]) else f"{k}=nan" for k in sorted(avg))