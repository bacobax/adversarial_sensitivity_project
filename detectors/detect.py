import argparse
import importlib.util
import os
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import torch

from support.base_detector import BaseDetector
from support.detect_utils import get_device

mapping = {
    'CLIP-D': ('CLIPDDetector', os.path.join('models', 'CLIP-D', 'detector.py')),
    'NPR': ('NPRDetector', os.path.join('models', 'NPR', 'detector.py')),
    'R50_nodown': ('R50NoDownDetector', os.path.join('models', 'R50_nodown', 'detector.py')),
    'P2G': ('P2GDetector', os.path.join('models', 'P2G', 'detector.py')),
    'R50_TF': ('R50TFDetector', os.path.join('models', 'R50_TF', 'detector.py')),
    'WaveRep': ('WaveRepDetector', os.path.join('models', 'WaveRep', 'detector.py')),
    'AnomalyOV': ('AnomalyOVDetector', os.path.join('models', 'anomaly_ov', 'detector.py')),
}

def _load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _detector_class_by_name(name: str):
    if name not in mapping:
        raise KeyError(name)
    class_name, path = mapping[name]
    module = _load_module_from_path(f"detector_{name}", path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in {path}")
    return getattr(module, class_name)


def parse_weights_map(weights: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Parse a weights mapping from command-line arguments.
    
    Input format: List of strings where each string can be:
        - "model_name:checkpoint_path" (model paired with checkpoint)
        - "model_name" (model with no checkpoint)
    
    Output format: Dictionary with model names as keys and checkpoint paths (or None) as values.
        Example: {"CLIP-D": "/path/to/checkpoint", "NPR": None}
    """
    mapping: Dict[str, Optional[str]] = {}
    if not weights:
        return mapping
    pairs = weights
    for p in pairs:
        if ':' in p:
            k, v = p.split(':', 1)
            mapping[k.strip()] = v.strip() if v.strip() else None
        else:
            mapping[p.strip()] = None
    return mapping


def available_detectors(device: torch.device) -> Dict[str, BaseDetector]:
    names = list(mapping.keys())
    instances: Dict[str, BaseDetector] = {}
    for n in names:
        try:
            cls = _detector_class_by_name(n)
            instances[n] = cls(device)
        except Exception as e:
            print(f"[warn] Could not register detector {n}: {e}")
    return instances


def main():
    parser = argparse.ArgumentParser(description='Batch detection over folders into a single results.csv')
    parser.add_argument('--folders', nargs='+', required=True, help='List of folders with images')
    parser.add_argument('--limit', type=int, default=0, help='Max number of sorted images per folder (0 = all)')
    parser.add_argument('--models', nargs='+', default=['all'], help='Comma list of models to use, or "all"')
    parser.add_argument('--weights', nargs='+', default=[], help='Optional mappings model:checkpoint_dir')
    parser.add_argument('--output', type=str, default='results.csv', help='Path to output CSV')
    parser.add_argument('--device', type=str, default='', help='Device override like cuda:0 or cpu')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else get_device()
    
    all_dets = available_detectors(device)
    
    selected = list(all_dets.keys()) if 'all' in args.models else args.models
    detectors_list: List[Tuple[BaseDetector, Optional[str]]] = []
    
    weights_map = parse_weights_map(args.weights)
    
    for name in selected:
        if name not in all_dets:
            print(f"[warn] Unknown detector '{name}', skipping")
            continue
        det = all_dets[name]
        model_id = weights_map.get(name)
        # Do not load here; BaseDetector.run_batch will call load for each detector
        detectors_list.append((det, model_id))
    
    if not detectors_list:
        raise RuntimeError('No detectors loaded. Check --models and --weights.')
    
    BaseDetector.run_batch(detectors_list, args.folders, args.limit, args.output)
    print(f"Done. Wrote {args.output}")


if __name__ == '__main__':
    main()
