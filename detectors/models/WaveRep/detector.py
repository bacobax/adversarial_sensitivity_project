import os
from typing import Optional, Sequence, Union

import numpy as np
import timm
import torch

from support.base_detector import BaseDetector
from support.lime_explain import lime_explain

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))


class WaveRepDetector(BaseDetector):
    name = 'WaveRep'
    
    supports_explainability = True
    supports_adversarial = True
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.cropping = 512
        self.arc = 'vit_base_patch14_reg4_dinov2.lvd142m'
        self.use_amp = True  # CUDA autocast
    
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
        
        if not isinstance(images, torch.Tensor):
            batch = self.prepare_batch(images, self.transform)
        else:
            batch = images
        
        if self.device.type == 'cuda' and self.use_amp:
            from torch.amp import autocast
            with autocast('cuda'):
                logits = self.model(batch)[:, -1]
        else:
            logits = self.model(batch)[:, -1]
        return logits
    
    def explain(
        self,
        images: np.ndarray,
        batch_size: int = 64,
        num_samples: int = 150,
        fixed_segments: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
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
            fixed_segments=fixed_segments,
        )
        return cam.detach().cpu().numpy()
    