import os
import sys
from typing import Optional

import torch
from PIL import Image

from support.base_detector import BaseDetector

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)

from utils import create_transform, create_model  # type: ignore


class WaveRepDetector(BaseDetector):
    name = 'WaveRep'
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.transform = None
        self.cropping = 512
        self.arc = 'vit_base_patch14_reg4_dinov2.lvd142m'
        # Inference knobs
        self.use_tta = True  # horizontal flip TTA
        self.use_amp = True  # CUDA autocast
    
    def _default_weights(self) -> str:
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
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        # WaveRep expects its own crop/normalization; load from path and ignore generic tensor
        assert self.model is not None, "Model not loaded"
        assert self.transform is not None, "Transform not initialized"
        img = Image.open(image_path).convert('RGB')
        frame = self.transform(img)
        frames = [frame]
        if self.use_tta:
            frames.append(torch.flip(frame, dims=[2]))  # horizontal flip (C,H,W) -> flip W
        batch = torch.stack(frames, 0).to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    logits = self.model(batch)[:, -1]
            else:
                logits = self.model(batch)[:, -1]
            probs = torch.sigmoid(logits).detach().cpu()
            prob = float(probs.mean().item())
        return prob
    
    def batch_predict(self, image_paths):
        assert self.model is not None, "Model not loaded"
        assert self.transform is not None, "Transform not initialized"
        frames = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            frames.append(self.transform(img))
        if not frames:
            return []
        batch = torch.stack(frames, 0).to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    logits_main = self.model(batch)[:, -1]
            else:
                logits_main = self.model(batch)[:, -1]
            probs_main = torch.sigmoid(logits_main)
            if self.use_tta:
                batch_flip = torch.flip(batch, dims=[3])  # flip width
                if self.device.type == 'cuda' and self.use_amp:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        logits_flip = self.model(batch_flip)[:, -1]
                else:
                    logits_flip = self.model(batch_flip)[:, -1]
                probs_flip = torch.sigmoid(logits_flip)
                probs = (probs_main + probs_flip) * 0.5
            else:
                probs = probs_main
            probs = probs.detach().cpu().tolist()
        return [float(x) for x in probs]
