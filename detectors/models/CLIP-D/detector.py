import os
import sys
from typing import Optional

import torch

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from networks.utils import create_architecture
from support.base_detector import BaseDetector


class CLIPDDetector(BaseDetector):
    name = 'CLIP-D'
    
    def load(self, model_id: Optional[str] = None) -> None:
        device = self.device
        weights = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"CLIP-D weights not found: {weights}")
        checkpoint = torch.load(weights, map_location=device)
        model = create_architecture("opencliplinearnext_clipL14commonpool", pretrained=False, num_classes=1).to(device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        out = self.model(image_tensor)
        return float(torch.sigmoid(out).item())
