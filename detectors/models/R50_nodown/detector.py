"""
R50_nodown Detector with GradCAM-based vulnerability visualization.

This detector uses ResNet50 without downsampling and generates saliency maps
using GradCAM for vulnerability analysis.
"""
import contextlib
import os
import sys
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from support.base_detector import BaseDetector
from utils.visualize import to_numpy_2d

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from networks.utils import create_architecture

# Default image size for R50_nodown
DEFAULT_IMAGE_SIZE = 512  # FIXME


class R50NoDownDetector(BaseDetector):
    """
    R50_nodown Detector implementing the BaseDetector interface with GradCAM support.
    
    This detector uses ResNet50 without downsampling and provides GradCAM-based
    saliency maps for vulnerability analysis.
    """
    
    name = 'R50_nodown'
    
    # Feature flags - this detector supports all features
    supports_explainability = True
    supports_vulnerability = True
    supports_adversarial = True
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.cam = None
        self.image_size = DEFAULT_IMAGE_SIZE
        self.use_amp = True  # NEW: mixed precision on CUDA to save memory
    
    def load(self, model_id: Optional[str] = None) -> None:
        """Load model weights and initialize GradCAM."""
        device = self.device
        weights = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"R50_nodown weights not found: {weights}")
        checkpoint = torch.load(weights, map_location=device)
        model = create_architecture("res50nodown", pretrained=True, num_classes=1).to(device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        
        # Initialize GradCAM with the last convolutional layer
        try:
            from pytorch_grad_cam import GradCAM
            target_layer = self.model.layer4[-1].conv3
            self.cam = GradCAM(model=self.model, target_layers=[target_layer])
        except ImportError:
            print("[warn] pytorch_grad_cam not installed. GradCAM features disabled.")
            self.cam = None
    
    def forward(self, image_tensor: torch.Tensor) -> float:
        """Predict whether image is fake. Returns confidence [0,1] where higher = more fake."""
        return self.model(image_tensor)
    
    def _amp_ctx(self):  # NEW
        if self.device is not None and self.device.type == "cuda" and self.use_amp:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        return contextlib.nullcontext()
    
    def explain(
        self,
        image: np.ndarray,
        map_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Compute GradCAM explanation map for an image.
        
        Args:
            image: Input image as path string or PIL Image
            map_size: Size of the returned map (H, W). Default: (512, 512)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            tuple: (confidence, gradcam_map)
                - confidence: float in [0, 1], higher = more likely fake
                - gradcam_map: np.ndarray of shape (H, W) normalized to [0, 1]
        """
        if map_size is None:
            map_size = (self.image_size, self.image_size)
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.cam is None:
            raise RuntimeError("GradCAM not initialized. Install pytorch_grad_cam.")
        
        image_pil = Image.fromarray(image)  # NEW (avoid overwriting `image`)
        
        # CAM at reduced resolution to avoid OOM
        img_tensor = self.transform(image_pil).to(self.device)  # NEW
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        self.model.eval()
        
        # GradCAM needs grads: do NOT wrap the whole forward in no_grad()
        with self._amp_ctx():  # NEW
            output = self.model(img_tensor)
        
        # Target index: for single-logit binary, GradCAM can just target that logit
        target_idx = 0 if output.ndim == 1 or output.shape[-1] == 1 else 1
        
        with self._amp_ctx():  # NEW
            grayscale_cam = self.cam(
                input_tensor=img_tensor,
                targets=[ClassifierOutputTarget(target_idx)],
            )
        
        
        cam_map = grayscale_cam[0, :].astype(np.float32)  # NEW
        
        # Resize CAM map to requested map_size
        if cam_map.shape != map_size:
            # cam_pil = Image.Image.fromarray((cam_map * 255).astype(np.uint8), mode='L')
            cam_pil = Image.fromarray((cam_map * 255).astype(np.uint8), mode="L")  # NEW
            cam_pil = cam_pil.resize((map_size[1], map_size[0]), Image.BILINEAR)
            cam_map = np.array(cam_pil).astype(np.float32) / 255.0
        
        cam_map = to_numpy_2d(cam_map, image.shape[:2])
        
        # best-effort memory cleanup on CUDA
        if self.device is not None and self.device.type == "cuda":  # NEW
            self.model.zero_grad(set_to_none=True)  # NEW
            torch.cuda.empty_cache()  # NEW (optional but helps in loops)
        
        return cam_map
