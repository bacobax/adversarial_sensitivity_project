import os

import numpy as np
from PIL import Image

from utils.detector_wrapper import DetectorWrapper
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

def get_or_generate_attacked_image(
    detector: DetectorWrapper,
    image: np.ndarray,
    attack_type: str,
    cache_path: str,
    overwrite: bool = False,
    filename: str = "",
    image_type: str = ""
) -> np.ndarray:
    """
    Get attacked image from cache or generate if not exists.

    Args:
        detector: Detector wrapper instance
        image: Original RGB image
        attack_type: Type of attack
        cache_path: Path to cache the attacked image
        overwrite: If True, regenerate even if cached
        filename: Filename for error messages
        image_type: Image type for error messages

    Returns:
        Attacked RGB image as np.ndarray (H, W, 3) uint8
    """
    if os.path.exists(cache_path) and not overwrite:
        return load_image(cache_path)
    
    # Generate attacked image
    adv_image = detector.attack(
        image,
        attack_type,
        filename=filename,
        image_type=image_type
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to cache
    Image.fromarray(adv_image).save(cache_path)
    
    return adv_image