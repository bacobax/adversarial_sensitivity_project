import os
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from visualize_explain import colorize_cam


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


def to_numpy_2d(arr: Any, target_size: Tuple[int, int]) -> np.ndarray:
    """Convert array to 2D float32 numpy array normalized to [0, 1]."""
    # Handle torch tensors
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    
    arr = np.asarray(arr, dtype=np.float32)
    
    # Squeeze extra dimensions
    while arr.ndim > 2:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        elif arr.shape[-1] == 3:
            # RGB to grayscale
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        elif arr.shape[0] == 3:
            arr = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
        else:
            arr = arr[0]
    
    if arr.ndim != 2:
        raise ValueError(f"Could not convert to 2D array, got shape {arr.shape}")
    
    # Normalize to [0, 1]
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    
    # Resize to target size if needed
    if arr.shape != target_size:
        arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
    
    return arr.astype(np.float32)


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


