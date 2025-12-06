import os
import sys
from typing import Optional

import torch

from support.base_detector import BaseDetector

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from networks.resnet import resnet50


class NPRDetector(BaseDetector):
    name = 'NPR'
    
    def load(self, model_id: Optional[str] = None) -> None:
        device = self.device
        weights = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"NPR weights not found: {weights}")
        checkpoint = torch.load(weights, map_location=device)
        model = resnet50(num_classes=1).to(device)
        model.load_state_dict(checkpoint, strict=True)
        model.eval()
        self.model = model
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        out = self.model(image_tensor)
        return float(torch.sigmoid(out).item())
