import os
import sys
from typing import Optional

import torch
import yaml

from support.base_detector import BaseDetector

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from network import ImageClassifier


class R50TFDetector(BaseDetector):
    name = 'R50_TF'
    
    def __init__(self, device: Optional[torch.device] = None, config_path: Optional[str] = None):
        super().__init__(device)
        self.config_path = config_path or os.path.join('configs', 'R50_TF.yaml')
    
    def _parse_detector_args(self, detector_args, default_num_centers=1):
        class Settings:
            def __init__(self):
                self.arch = "nodown"
                self.freeze = False
                self.prototype = False
                self.num_centers = default_num_centers
        
        settings = Settings()
        i = 0
        while i < len(detector_args):
            arg = detector_args[i]
            if arg == "--arch" and i + 1 < len(detector_args):
                settings.arch = detector_args[i + 1]
                i += 2
            elif arg == "--freeze":
                settings.freeze = True
                i += 1
            elif arg == "--prototype":
                settings.prototype = True
                i += 1
            elif arg == "--num_centers" and i + 1 < len(detector_args):
                settings.num_centers = int(detector_args[i + 1])
                i += 2
            else:
                i += 1
        return settings
    
    def load(self, model_id: Optional[str] = None) -> None:
        device = self.device
        weights = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"R50_TF config not found: {self.config_path}")
        with open(self.config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        settings = self._parse_detector_args(cfg.get('detector_args', []))
        model = ImageClassifier(settings)
        model.load_state_dict(torch.load(weights, map_location=device))
        model.to(device)
        model.eval()
        self.model = model
    
    def forward(self, image_tensor: torch.Tensor) -> float:
        out = self.model(image_tensor).squeeze(1)
        return float(torch.sigmoid(out).item())
