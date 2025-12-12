import os
from dataclasses import dataclass, field
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image


@dataclass
class DetectorWrapper:
    """
    Unified interface wrapper for detectors.

    Ensures all detectors conform to the same external API:
        - name: str
        - explain(image: np.ndarray) -> np.ndarray
        - attack(image: np.ndarray, attack_type: str) -> np.ndarray
    """
    detector: Any
    _name: str = field(init=False)
    
    def __post_init__(self):
        self._name = getattr(self.detector, 'name', self.detector.__class__.__name__)
    
    @property
    def name(self) -> str:
        return self._name
    
    def attack(
        self,
        image: np.ndarray,
        attack_type: str,
        filename: str = "",
        image_type: str = "",
    ) -> np.ndarray:
        """
        Generate adversarial attack on image.

        Args:
            image: RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]
            attack_type: Type of attack ('pgd', 'fgsm', 'deepfool')
            filename: Source filename for error messages
            image_type: Image type for error messages

        Returns:
            Attacked RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]

        Raises:
            ValueError: If required args are None or invalid
        """
        if image is None:
            raise ValueError(
                f"attack() received None image. "
                f"model_name={self.name}, attack_type={attack_type}, "
                f"filename={filename}, image_type={image_type}",
            )
        
        if attack_type is None:
            raise ValueError(
                f"attack() received None attack_type. "
                f"model_name={self.name}, filename={filename}, image_type={image_type}",
            )
        
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"attack() expected np.ndarray, got {type(image)}. "
                f"model_name={self.name}, attack_type={attack_type}, "
                f"filename={filename}, image_type={image_type}",
            )
        
        import tempfile
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_input = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_output = f.name
        
        try:
            # Save input image
            Image.fromarray(image).save(temp_input)
            
            # Generate adversarial image
            if hasattr(self.detector, 'generate_adversarial'):
                self.detector.generate_adversarial(
                    image_path=temp_input,
                    output_path=temp_output,
                    attack_type=attack_type,
                    true_label=1,  # Assume fake images (samecat/diffcat)
                )
            elif hasattr(self.detector, '_generate_adversarial_image'):
                self.detector._generate_adversarial_image(
                    image_path=temp_input,
                    output_path=temp_output,
                    attack_type=attack_type,
                    true_label=1,
                )
            else:
                raise NotImplementedError(
                    f"Detector {self.name} does not implement generate_adversarial() or "
                    f"_generate_adversarial_image()",
                )
            
            # Load attacked image
            adv_image = np.array(Image.open(temp_output).convert('RGB'))
            
            # Ensure same size as input
            if adv_image.shape[:2] != image.shape[:2]:
                adv_pil = Image.fromarray(adv_image)
                adv_pil = adv_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
                adv_image = np.array(adv_pil)
            
            return adv_image
        
        finally:
            # Cleanup temp files
            for f in [temp_input, temp_output]:
                try:
                    os.unlink(f)
                except:
                    pass


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
