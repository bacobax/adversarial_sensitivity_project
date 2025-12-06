import json
import os
import pickle
import sys
from typing import Optional

import torch

from support.base_detector import BaseDetector

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from src.models.slinet_det import SliNet  # type: ignore


class P2GDetector(BaseDetector):
    name = 'P2G'
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.config = None
        self.object_labels_dict = None
    
    def load(self, model_id: Optional[str] = None) -> None:
        if SliNet is None:
            raise RuntimeError("P2G dependencies not available")
        device = self.device
        model_path = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        config_path = os.path.join(DETECTOR_DIR, 'configs', 'test.json')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"P2G checkpoint not found: {model_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        checkpoint = torch.load(model_path, map_location=device)
        # Populate config fields from checkpoint
        try:
            config['K'] = checkpoint.get('K', config.get('K', 5))
            config['topk_classes'] = checkpoint.get('topk_classes', config.get('topk_classes', 1))
            config['ensembling'] = checkpoint.get('ensembling_flags', config.get('ensembling', [False, False, False, False]))
            if 'tasks' in checkpoint:
                config['num_tasks'] = checkpoint['tasks'] + 1
                config['task_name'] = range(config['num_tasks'])
        except Exception:
            pass
        config['device'] = device
        model = SliNet(config)
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        self.model = model
        self.config = config
        # load object labels
        pkl_path = os.path.join(DETECTOR_DIR, 'src', 'utils', 'classes.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                self.object_labels_dict = pickle.load(f)
        else:
            self.object_labels_dict = {}
    
    def _lookup_object_label(self, image_path: str):
        def ensure_topk_tuples(label_list, topk=5):
            tuples = [(lbl, 1.0) if isinstance(lbl, str) else lbl for lbl in label_list]
            seen = set()
            unique = []
            for t in tuples:
                if t[0] not in seen:
                    unique.append(t)
                    seen.add(t[0])
            while len(unique) < topk:
                unique.append(('unknown', 1.0))
            return unique[:topk]
        
        if not self.object_labels_dict:
            return ensure_topk_tuples(['unknown'] * 5, topk=5)
        rel_path = os.path.relpath(image_path, self.config.get('data_path', '')).replace(os.sep, '/')
        candidates = [rel_path, rel_path.lstrip('/'), '/' + rel_path]
        found_key = None
        for k in candidates:
            if k in self.object_labels_dict:
                found_key = k
                break
        if found_key is None:
            basename = os.path.basename(rel_path)
            for k in self.object_labels_dict.keys():
                if k.endswith('/' + basename) or k.endswith(basename):
                    found_key = k
                    break
        if found_key is None:
            try:
                fallback_val = next(iter(self.object_labels_dict.values()))
            except StopIteration:
                fallback_val = [('unknown', 1.0)] * 5
            return ensure_topk_tuples(fallback_val, topk=5)
        val = self.object_labels_dict[found_key]
        return ensure_topk_tuples(val, topk=5)
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        object_label = self._lookup_object_label(image_path)
        outputs = self.model(image_tensor, object_label)
        if isinstance(outputs, dict) and 'logits' in outputs:
            out = torch.as_tensor(outputs['logits']).detach().cpu()
            if out.ndim == 2 and out.shape[1] == 2:
                probs = torch.softmax(out, dim=1)
                return float(probs[0, 1])
            return float(torch.sigmoid(out.mean()).item())
        elif torch.is_tensor(outputs):
            out = outputs.detach().cpu()
            if out.ndim == 0:
                return float(torch.sigmoid(out).item())
            if out.ndim == 1:
                if out.numel() == 2:
                    probs = torch.softmax(out, dim=0)
                    return float(probs[1])
                return float(torch.sigmoid(out.mean()).item())
            if out.ndim == 2 and out.shape[1] == 2:
                probs = torch.softmax(out, dim=1)
                return float(probs[0, 1])
            return float(torch.sigmoid(out.mean()).item())
        try:
            return float(outputs)
        except Exception:
            return 0.0
