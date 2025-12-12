import os
import sys
from typing import Optional

import torch
from torch import nn

from support.base_detector import BaseDetector, prepare_batch

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)

from utils import create_transform, create_model


class WaveRepDetector(BaseDetector):
    name = 'WaveRep'
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.transform = None
        self.cropping = 512
        self.arc = 'vit_base_patch14_reg4_dinov2.lvd142m'
        # Inference knobs
        self.use_amp = True  # CUDA autocast
        self._gradcam_target_layer: Optional[nn.Module] = None
        # cache for attention rollout (not strictly required but handy)
        self._attn_hook_handles = []
    
    @staticmethod
    def _default_weights() -> str:
        # Default weights location within WaveRep repo
        weights_path = os.path.abspath(os.path.join(DETECTOR_DIR, 'weights', 'weights_dinov2_G4.ckpt'))
        return weights_path
    
    def load(self, model_id: Optional[str] = None) -> None:
        device = self.device
        weights = model_id if model_id else self._default_weights()
        if not os.path.exists(weights):
            raise FileNotFoundError(f"WaveRep weights not found: {weights}")
        self.transform = create_transform(self.cropping)
        self.model = create_model(weights, self.arc, self.cropping, device)
        self.model.eval()
    
    def forward(self, images) -> float:
        assert self.model is not None, "Model not loaded"
        assert self.transform is not None, "Transform not initialized"
        
        batch = prepare_batch(images, self.device, self.transform)
        
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    logits = self.model(batch)[:, -1]
            else:
                logits = self.model(batch)[:, -1]
        return logits.detach().cpu()
