import os

import numpy as np
import torch
from PIL import Image

from support.base_detector import BaseDetector
from utils.image_loader import load_image


def get_attack_cache_path(
    root_dataset: str,
    model_name: str,
    attack_type: str,
    image_type: str,
    filename: str,
) -> str:
    """Get the path for a cached attacked image."""
    base_name = os.path.splitext(filename)[0]
    return os.path.join(
        root_dataset,
        'adv_attacks',
        model_name,
        attack_type,
        image_type,
        f"{base_name}.png",
    )


def denormalize_chw(x, mean, std):
    # x: (C,H,W) float tensor (normalized)
    mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    return x * std + mean


def _to_uint8_hwc(detector: BaseDetector, adv) -> np.ndarray:
    if isinstance(adv, torch.Tensor):
        t = adv.detach().cpu().float()
        if t.dim() == 4:
            t = t[0]
        if t.dim() == 2:
            t = t.unsqueeze(0)
        
        if t.dim() != 3:
            raise TypeError(f"Adversarial output must be 2D/3D/4D tensor; got shape {tuple(t.shape)}")
        
        if t.shape[0] in (1, 3, 4):
            t = denormalize_chw(t, detector.mean, detector.std)
            t = t.clamp(0.0, 1.0)
            arr = (t * 255.0).byte().permute(1, 2, 0).numpy()
        elif t.shape[-1] in (1, 3, 4):
            t = t.clamp(0.0, 1.0)
            arr = (t * 255.0).byte().numpy()
        else:
            raise TypeError(f"Adversarial tensor has unsupported shape {tuple(t.shape)}")
    
    else:
        arr = np.asarray(adv)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise TypeError(f"Adversarial output must be 2D/3D/4D array; got shape {arr.shape}")
        
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        
        if arr.shape[-1] not in (1, 3, 4):
            raise TypeError(f"Adversarial array has unsupported shape {arr.shape}")
        
        if arr.dtype != np.uint8:
            arr_f = arr.astype(np.float32)
            mn, mx = float(np.min(arr_f)), float(np.max(arr_f))
            if mn >= -1.0 and mx <= 1.0 and (mn < 0.0 or mx <= 1.0):
                if mn < 0.0:
                    arr_f = (arr_f + 1.0) / 2.0
                arr_f = np.clip(arr_f, 0.0, 1.0) * 255.0
            else:
                arr_f = np.clip(arr_f, 0.0, 255.0)
            arr = arr_f.astype(np.uint8)
    
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return arr


def get_or_generate_attacked_image(
    detector: BaseDetector,
    image: Image.Image,
    attack_type: str,
    cache_path: str,
    overwrite: bool = False,
) -> np.ndarray:
    """
    Get attacked image from cache or generate if not exists.

    Args:
        detector: Detector wrapper instance
        image: Original RGB image
        attack_type: Type of attack
        cache_path: Path to cache the attacked image
        overwrite: If True, regenerate even if cached
        attack_obj: Attack object

    Returns:
        Attacked RGB image as np.ndarray (H, W, 3) uint8
    """
    if os.path.exists(cache_path) and not overwrite:
        return load_image(cache_path)
    
    # Generate attacked image
    adv_image = detector.generate_adversarial(image, attack_type)
    
    adv_image = _to_uint8_hwc(detector, adv_image)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to cache
    Image.fromarray(adv_image).save(cache_path)
    
    return adv_image
