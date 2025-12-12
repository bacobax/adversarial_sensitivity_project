import os
from typing import Optional

import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from torch import nn

from support.base_detector import BaseDetector, prepare_batch
from support.lime_explain import lime_explain

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))


class WaveRepDetector(BaseDetector):
    name = 'WaveRep'
    
    supports_explainability = True
    
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
        self.transform = self.create_transform(self.cropping)
        self.model = self.create_model(weights, self.arc, self.cropping, device)
        self.model.eval()
    
    @staticmethod
    def create_model(weights, arc, cropping, device):
        # create model
        model = timm.create_model(arc, num_classes=1, pretrained=True, img_size=cropping)
        model = model.to(device)
        
        # load weights
        # print('loading the model from %s' % weights)
        dat = torch.load(weights, map_location=device)
        if 'state_dict' in dat:
            dat = {k[6:]: dat['state_dict'][k] for k in dat['state_dict'] if k.startswith('model')}
        model.load_state_dict(dat)
        del dat
        
        return model
    
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
    
    def explain(
        self,
        images: np.ndarray,
        batch_size: int = 80,
        num_samples: int = 150,
    ) -> np.ndarray:
        """
        Compute explainability maps for a batch of already-normalized images.

        Args:
            images: (B, 3, H, W) tensor normalized with ImageNet stats.
            batch_size: batch size for explainability maps.
            num_samples: number of samples for explainability maps.

        Returns:
            cam: (B, 1, H, W) maps in [0, 1].
            logits: (B, K) raw logits (no sigmoid).
        """
    
        cam = lime_explain(
            logits_fn=self.forward,
            images=images,
            batch_size=batch_size,
            num_samples=num_samples,
        )
        return cam.detach().cpu().numpy()
