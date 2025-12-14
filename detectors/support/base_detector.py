"""
Base Detector Module

This module provides the abstract base class for all detectors in the framework.
It defines the interface for prediction, explainability maps, vulnerability analysis,
and adversarial attack generation.

Subclasses should implement the required abstract methods to provide detector-specific
behavior while inheriting common functionality.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torchattacks as ta
from PIL import Image
from torch import nn
from torchvision import transforms

from .detect_utils import get_device


class BaseDetector(ABC):
    """
    Abstract base class that unifies I/O, batching, and explainability for detectors.
    
    This class provides:
    - Core prediction interface
    - Explainability map generation (optional, detector-specific)
    - Vulnerability analysis (optional, detector-specific)
    - Adversarial attack generation (optional, detector-specific)
    - Batch processing utilities
    
    Required implementations:
        - name: Unique model name string (class attribute)
        - load(): Load model weights
        - predict(): Single image prediction
    
    Optional implementations (for explainability/vulnerability):
        - _compute_explanation_map(): Generate explanation/saliency map
        - _compute_vulnerability_map(): Generate vulnerability map after attack
        - _generate_adversarial_image(): Generate adversarial perturbation
        - visualize_vulnerability_grid(): Create grid visualization
    
    Attributes:
        name (str): Unique identifier for the detector
        device (torch.device): Device for computation
        model: The underlying model (type depends on detector)
    """
    
    name: str = "base"
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Feature flags - subclasses should set these to True if they implement the feature
    supports_explainability: bool = False
    supports_vulnerability: bool = False
    supports_adversarial: bool = False
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the base detector.
        
        Args:
            device: Torch device for computation. If None, auto-detects best available.
        """
        self.device = device or get_device()
        self.model = None
    
    # =========================================================================
    # REQUIRED ABSTRACT METHODS - Must be implemented by all subclasses
    # =========================================================================
    
    @abstractmethod
    def load(self, model_id: Optional[str] = None) -> None:
        """
        Load model weights from disk or hub.
        
        Args:
            model_id: Path to weights file or model identifier.
                     If None, uses default weights location.
        
        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If model loading fails
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, images):
        """
        Predict whether an image is fake.
        
        Args:
            images: Preprocessed image tensor
        
        Returns:
            float: Confidence score in [0, 1] where higher = more likely fake
        """
        raise NotImplementedError
    
    @abstractmethod
    def explain(self, images: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError
    
    def transform(self, x):
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])
        return t(x)
    
    def generate_adversarial(
        self,
        image: Image.Image,
        attack_type: str = 'fgsm',
    ):
        """If attack is requested, generate adversarial images for WaveRep and return folders pointing to adv images.

        Creates a mirrored folder tree under <out_dir>/adv_inputs/<attack_name>/ and writes PNG files there.
        """
        attack_label = 1
        attack_type = attack_type.lower()
        labels_tensor = torch.as_tensor([attack_label], device=self.device, dtype=torch.long)
        
        attack_detector = AttackDetector(self, self.device)
        attack = build_attack(attack_type, attack_detector)
        print(f"Len images before prepare_batch: {len(image) if isinstance(image, (list, tuple)) else '1, non array'}")
        image = self.prepare_batch(image, self.transform)
        print(f"Prepared batch shape: {image.shape}")
        image.requires_grad_(True)
        
        with torch.enable_grad():
            if self.device.type == 'cuda':
                from torch.amp import autocast
                with autocast('cuda'):
                    adv = attack(image, labels_tensor)
                del attack_detector
                del attack
                torch.cuda.empty_cache()
            else:
                adv = attack(image, labels_tensor)
                del attack_detector
                del attack
        return adv.detach().cpu()[0]
    
    def prepare_batch(self, images, transform=None):
        # Normalize input into a list of PIL Images
        frames = []
        
        # Treat list/tuple/ndarray as a batch
        if isinstance(images, (list, tuple, np.ndarray)):
            iterable = images
        else:
            iterable = [images]
        
        for img in iterable:
            # Allow passing paths, NumPy arrays, or PIL Images
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            elif isinstance(img, np.ndarray):
                # Expect HWC, uint8 or float in [0, 255] / [0, 1]
                if img.dtype != np.uint8:
                    img = np.clip(img, 0, 255).astype(np.uint8)
                img = Image.fromarray(img)
            # If it's already a PIL.Image.Image or tensor, let the transform handle it
            if transform is not None:
                img = transform(img)
            frames.append(img)

        
        return torch.stack(frames, 0).to(self.device)


class AttackDetector(nn.Module):
    def __init__(self, base: BaseDetector, device: torch.device):
        super().__init__()
        self.base = base
        self.device = device
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base.forward(x)
        # Convert single-logit output to 2-class logits for CE-based attacks
        if out.dim() == 2 and out.size(1) > 1:
            z = out[:, -1]
        else:
            z = out.reshape(-1)  # NEW: handles scalar / [B] / [B,1] -> [B]
        logits2 = torch.stack([-z, z], dim=1)
        return logits2


def build_attack(name: str, model: AttackDetector):
    name = name.lower()
    if name == 'fgsm':
        atk = ta.FGSM(model)
    elif name == 'pgd':
        atk = ta.PGD(model)
    elif name == 'deepfool':
        atk = ta.DeepFool(model)
    else:
        raise ValueError(f"Unsupported attack '{name}'. Choose from: fgsm, pgd, deepfool")
    atk.set_normalization_used(model.base.mean, model.base.std)
    return atk
