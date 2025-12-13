"""
Anomaly-OV Detector wrapper for the unified detection framework.

This detector uses the Anomaly-OV model for image anomaly detection, 
implementing the BaseDetector interface for compatibility with detect.py.
"""

import os
import sys
from typing import Optional

import numpy as np
import torch

# Set up path for internal imports within anomaly_ov
DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure anomaly_ov directory is in path for internal llava/src imports
if DETECTOR_DIR not in sys.path:
    sys.path.insert(0, DETECTOR_DIR)

# Import BaseDetector from the support module (parent directory)
from support.base_detector import BaseDetector

# Internal imports from the anomaly_ov module
from llava.model.anomaly_expert import AnomalyOV
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower


class AnomalyOVDetector(BaseDetector):
    """
    Anomaly-OV Detector implementing the BaseDetector interface.
    
    This detector uses a SigLip vision encoder combined with an 
    anomaly expert module for zero-shot anomaly detection.
    """
    
    name = 'AnomalyOV'
    
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    # Feature flags - this detector supports all features
    supports_explainability = True
    supports_vulnerability = True
    supports_adversarial = True
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.vision_tower = None
        self.anomaly_expert = None
        self.image_processor = None
        self._dtype = torch.float32
        
        # Handle MPS/CUDA/CPU device-specific dtype
        if self.device.type == 'cuda':
            self._dtype = torch.float16
        elif self.device.type == 'mps':
            self._dtype = torch.float32  # MPS has limited dtype support
        else:
            self._dtype = torch.float32
    
    def load(self, model_id: Optional[str] = None) -> None:
        """
        Load the Anomaly-OV model weights.
        
        Args:
            model_id: Path to a checkpoint file (.pt) or to the anomaly expert weights.
                     If None, uses the default weights in the weights folder.
                     Can be:
                       - A full checkpoint from save_checkpoint() method
                       - An anomaly expert weights file (pretrained_expert_*.pth)
                       - A directory containing 'best.pt' or 'zs_checkpoint.pt'
        """
        device = self.device
        
        # Determine weights path
        if model_id is None:
            # Try default checkpoint locations in order of preference
            default_paths = [
                os.path.join(DETECTOR_DIR, 'weights', 'best.pt'),
                os.path.join(DETECTOR_DIR, 'weights', 'zs_checkpoint.pt'),
            ]
            weights = None
            for path in default_paths:
                if os.path.exists(path):
                    weights = path
                    break
            if weights is None:
                raise FileNotFoundError(
                    f"AnomalyOV weights not found. Tried: {default_paths}. "
                    f"Please place weights in {os.path.join(DETECTOR_DIR, 'weights')}/",
                )
        elif os.path.isdir(model_id):
            # If directory provided, look for standard checkpoint names
            for name in ['best.pt', 'zs_checkpoint.pt']:
                path = os.path.join(model_id, name)
                if os.path.exists(path):
                    weights = path
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {model_id}")
        else:
            weights = model_id
            if not os.path.exists(weights):
                raise FileNotFoundError(f"AnomalyOV weights not found: {weights}")
        
        print(f"Loading AnomalyOV from: {weights}")
        
        # Check if this is a full checkpoint or just anomaly expert weights
        checkpoint = torch.load(weights, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'vision_encoder_state_dict' in checkpoint:
            # Full checkpoint from OVAnomalyDetector.save_checkpoint()
            self._load_from_full_checkpoint(checkpoint, device)
        else:
            # Assume it's anomaly expert weights or a state dict
            self._load_from_expert_weights(weights, device)
        
        print(f"AnomalyOV loaded successfully on {device}")
    
    def _load_from_full_checkpoint(self, checkpoint: dict, device: torch.device) -> None:
        """Load from a full checkpoint containing both vision encoder and anomaly expert."""
        # Parse dtype
        dtype_str = checkpoint.get('dtype', 'torch.float32')
        if isinstance(dtype_str, str):
            self._dtype = getattr(torch, dtype_str.replace('torch.', ''), torch.float32)
        
        # Fallback on CPU/MPS if unsupported dtype
        if device.type in ('cpu', 'mps') and self._dtype in (torch.float16, torch.bfloat16):
            self._dtype = torch.float32
        
        # Build vision tower
        vision_tower_name = checkpoint.get('vision_tower_name', 'google/siglip-so400m-patch14-384')
        print(f"Loading vision tower: {vision_tower_name}")
        
        self.vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
        self.vision_tower.load_state_dict(checkpoint['vision_encoder_state_dict'])
        self.vision_tower.to(device)
        self.vision_tower.to(dtype=self._dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        
        # Build anomaly expert if present
        if checkpoint.get('anomaly_expert_state_dict') is not None:
            print("Loading anomaly expert...")
            self.anomaly_expert = AnomalyOV()
            self.anomaly_expert.load_state_dict(checkpoint['anomaly_expert_state_dict'])
            self.anomaly_expert.to(dtype=self._dtype, device=device)
            self.anomaly_expert.requires_grad_(False)
            self.anomaly_expert.eval()
        
        self.image_processor = self.vision_tower.image_processor
    
    def _load_from_expert_weights(self, weights_path: str, device: torch.device) -> None:
        """Load vision tower from pretrained and anomaly expert from weights file."""
        # Fallback dtype for CPU/MPS
        if device.type in ('cpu', 'mps'):
            self._dtype = torch.float32
        
        # Build vision tower (downloads from HuggingFace if needed)
        vision_tower_name = 'google/siglip-so400m-patch14-384'
        print(f"Loading vision tower: {vision_tower_name}")
        
        self.vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
        self.vision_tower.to(device)
        self.vision_tower.to(dtype=self._dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        
        # Build anomaly expert
        print(f"Loading anomaly expert from: {weights_path}")
        self.anomaly_expert = AnomalyOV()
        self.anomaly_expert.load_zero_shot_weights(path=weights_path, device=str(device))
        self.anomaly_expert.freeze_layers()
        self.anomaly_expert.to(dtype=self._dtype, device=device)
        self.anomaly_expert.requires_grad_(False)
        self.anomaly_expert.eval()
        
        self.image_processor = self.vision_tower.image_processor
    
    def transform(self, img):
        # Prepare a transform that uses the Anomaly-OV image processor
        processed = self.image_processor.preprocess([img], return_tensors='pt')
        pixel_values = processed['pixel_values'][0]
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        return pixel_values
    
    def forward(self, images) -> torch.Tensor:
        """Predict whether image(s) are real or fake.
        
        This mirrors the batch-style structure of WaveRepDetector.forward:
        it accepts a single image or a batch and returns a CPU tensor of
        confidences.
        
        Args:
            images: Single image or iterable of images. Each element can be:
                - str (path to image)
                - PIL.Image.Image
                - np.ndarray
        
        Returns:
            torch.Tensor: 1D tensor of confidence scores on CPU, where higher
                values indicate the image is more likely to be FAKE (anomalous).
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if not isinstance(images, torch.Tensor):
            pixel_values = self.prepare_batch(images, self.transform)
        else:
            pixel_values = images
        
        # Move to target device/dtype once (SigLipVisionTower.forward would also do this,
        # but doing it here prevents repeated conversions and enables channels_last).
        pixel_values = pixel_values.to(self.device, dtype=self._dtype, non_blocking=True)
        
        # Ensure shape is [B, V, 3, H, W]; most common case is V=1
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)
        
        b, v, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * v, c, h, w)
        
        # Encode images through vision tower
        image_features, image_level_features = self.vision_tower(pixel_values)
        
        # split_sizes represents number of patches per image (for batching)
        patches_per_image = image_features.shape[0] // (b * v)
        split_sizes = [patches_per_image] * (b * v)
        
        _, _, final_prediction = self.anomaly_expert(
            image_features,
            image_level_features,
            split_sizes,
            return_anomaly_map=False,
            return_probabilities=False,
        )
        
        # final_prediction is in [0, 1] where 1 = anomalous (fake)
        final_prediction = final_prediction.view(b, v, -1).mean(dim=(1, 2))
        return final_prediction
    
    def explain(self, image: np.ndarray, anomaly_map_size: tuple = (224, 224)) -> None:
        """
        Explain the prediction for a given image.
        
        Args:
            image: Input image as a numpy array.
            anomaly_map_size: Size of the anomaly map to generate.
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        pixel_values = self.transform(image)
        pixel_values = pixel_values.unsqueeze(0)
        
        # Move to target device/dtype once (SigLipVisionTower.forward would also do this,
        # but doing it here prevents repeated conversions and enables channels_last).
        pixel_values = pixel_values.to(self.device, dtype=self._dtype, non_blocking=True)
        
        if self.device.type == 'cuda':
            pixel_values = pixel_values.contiguous(memory_format=torch.channels_last)
        
        # Encode images through vision tower
        if self.device.type == 'cuda':
            from torch.amp import autocast
            with torch.inference_mode(), autocast('cuda'):
                image_features, image_level_features = self.vision_tower(pixel_values)
                split_sizes = [image_features.shape[0]]
                _, _, final_prediction, anomaly_map = self.anomaly_expert(
                    image_features,
                    image_level_features,
                    split_sizes,
                    return_anomaly_map=True,
                    anomaly_map_size=anomaly_map_size,
                )
        else:
            with torch.inference_mode():
                image_features, image_level_features = self.vision_tower(pixel_values)
                split_sizes = [image_features.shape[0]]
                _, _, final_prediction, anomaly_map = self.anomaly_expert(
                    image_features,
                    image_level_features,
                    split_sizes,
                    return_anomaly_map=True,
                    anomaly_map_size=anomaly_map_size,
                )

        return anomaly_map.detach().cpu()
