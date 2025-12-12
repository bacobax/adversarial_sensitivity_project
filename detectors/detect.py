import argparse
import importlib.util
import os
from types import ModuleType
from typing import Dict, List, Optional, Tuple

import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from support.base_detector import BaseDetector
from support.detect_utils import get_device

# Optional: attacks for WaveRep
try:
    from models.WaveRep.attack import attack_image_paths
    
    _WAVEREP_ATTACKS_AVAILABLE = True
except Exception as e:
    print(f"[warn] {e}")
    _WAVEREP_ATTACKS_AVAILABLE = False

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


def _parse_attack_args(pairs: List[str]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for p in pairs:
        if '=' not in p:
            continue
        k, v = p.split('=', 1)
        k = k.strip()
        v = v.strip()
        # try int, then float, then bool, else string
        if v.lower() in {'true', 'false'}:
            out[k] = v.lower() == 'true'
            continue
        try:
            out[k] = int(v)
            continue
        except ValueError:
            pass
        try:
            out[k] = float(v)
            continue
        except ValueError:
            pass
        out[k] = v
    return out


def _maybe_attack_inputs(args, selected: List[str], folders: List[str], device: torch.device) -> List[str]:
    """If attack is requested, generate adversarial images for WaveRep and return folders pointing to adv images.

    Creates a mirrored folder tree under <out_dir>/adv_inputs/<attack_name>/ and writes PNG files there.
    """
    if not args.attack:
        return folders
    if 'WaveRep' not in selected:
        print('[warn] --attack is currently supported only for WaveRep. Skipping adversarial generation.')
        return folders
    if not _WAVEREP_ATTACKS_AVAILABLE:
        print('[warn] WaveRep attack module not available. Skipping adversarial generation.')
        return folders
    if args.attack_label not in (0, 1):
        raise ValueError('--attack_label must be 0 or 1 when using --attack')
    
    atk_name = args.attack.lower()
    atk_kwargs = _parse_attack_args(args.attack_args)
    # Ensure common defaults if not present
    if atk_name in {'fgsm', 'pgd'}:
        # Expect eps/alpha in 0..1 range. If provided as integer like 8, assume 8/255.
        def _fix_eps(key: str):
            if key in atk_kwargs:
                val = atk_kwargs[key]
                if isinstance(val, (int, float)) and val > 1:
                    atk_kwargs[key] = float(val) / 255.0
        
        _fix_eps('eps')
        _fix_eps('alpha')
    
    # Build adversarials per folder
    to_pil = ToPILImage()
    adv_root = os.path.join(os.path.dirname(args.output), 'adv_inputs', atk_name)
    os.makedirs(adv_root, exist_ok=True)
    
    adv_folders: List[str] = []
    for folder in folders:
        # enumerate images
        images = BaseDetector.list_images(folder)
        if args.limit > 0:
            images = images[:args.limit]
        if not images:
            adv_folders.append(folder)
            continue
        labels = [int(args.attack_label)] * len(images)
        # Generate adversarials
        advs = attack_image_paths(
            images,
            labels,
            attack=atk_name,
            attack_kwargs=atk_kwargs,
            device=device,
            batch_size=max(1, int(args.attack_batch)),
        )
        # Write to mirrored folder
        rel_name = os.path.basename(os.path.normpath(folder))
        out_folder = os.path.join(adv_root, rel_name)
        os.makedirs(out_folder, exist_ok=True)
        for src, adv in tqdm(zip(images, advs), total=len(images)):
            fname = os.path.basename(src)
            dst = os.path.join(out_folder, fname)
            to_pil(adv).save(dst)
        adv_folders.append(out_folder)
    
    print(f"[info] Generated adversarial inputs under: {adv_root}")
    return adv_folders


def main():
    parser = argparse.ArgumentParser(description='Batch detection over folders into a single results.csv')
    parser.add_argument('--folders', nargs='+', required=True, help='List of folders with images')
    parser.add_argument('--limit', type=int, default=0, help='Max number of sorted images per folder (0 = all)')
    parser.add_argument('--detectors', nargs='+', default=['all'], help='Comma list of detectors to use, or "all"')
    parser.add_argument('--weights', nargs='+', default=[], help='Optional mappings model:checkpoint_dir')
    parser.add_argument('--output', type=str, default='out/results.csv', help='Path to output CSV')
    parser.add_argument('--device', type=str, default='', help='Device override like cuda:0 or cpu')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    # Adversarial options (WaveRep only for now)
    parser.add_argument('--attack', type=str, default='', help='Adversarial attack name: fgsm | pgd | deepfool')
    parser.add_argument('--attack_label', type=int, default=1, help='Label to use during attack (0=real,1=fake)')
    parser.add_argument('--attack_args', nargs='*', default=[], help='Extra attack args as key=val e.g. eps=8 alpha=2 steps=10')
    parser.add_argument('--attack_batch', type=int, default=2, help='Batch size when generating adversarial inputs')
    
    args = parser.parse_args()
    
    device = torch.device(args.device) if args.device else get_device()
    
    all_dets = available_detectors(device)
    
    selected = list(all_dets.keys()) if 'all' in args.detectors else args.detectors
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
        raise RuntimeError('No detectors loaded. Check --detectors and --weights.')
    
    # Optionally produce adversarial inputs for WaveRep
    run_folders = _maybe_attack_inputs(args, selected, args.folders, device)
    
    BaseDetector.run_batch(detectors_list, run_folders, args.limit, args.output)
    print(f"Done. Wrote {args.output}")


if __name__ == '__main__':
    main()
