
from models.WaveRep.detector import WaveRepDetector
from models.R50_nodown.detector import R50NoDownDetector
from support.detect_utils import load_np_image
from detect import parse_weights_map
import os
import importlib.util
from types import ModuleType

from tqdm import tqdm
import torch
from typing import Dict, List

from support.base_detector import BaseDetector

device = "mps"

mapping = {
    'CLIP-D': ('CLIPDDetector', os.path.join('models', 'CLIP-D', 'detector.py')),
    'NPR': ('NPRDetector', os.path.join('models', 'NPR', 'detector.py')),
    'R50_nodown': ('R50NoDownDetector', os.path.join('models', 'R50_nodown', 'detector.py')),
    'P2G': ('P2GDetector', os.path.join('models', 'P2G', 'detector.py')),
    'R50_TF': ('R50TFDetector', os.path.join('models', 'R50_TF', 'detector.py')),
    'WaveRep': ('WaveRepDetector', os.path.join('models', 'WaveRep', 'detector.py')),
}
def _load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

def list_images(folder: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    files.sort()
    return files

def _detector_class_by_name(name: str):
    if name not in mapping:
        raise KeyError(name)
    class_name, path = mapping[name]
    module = _load_module_from_path(f"detector_{name}", path)
    if not hasattr(module, class_name):
        raise AttributeError(f"Class {class_name} not found in {path}")
    return getattr(module, class_name)

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


limit_per_folder = 10
batch_size = 4
models = ["CLIP-D"]
available_d = available_detectors(device)
selected = list(available_d.keys()) if 'all' in models else models


w_map = parse_weights_map(["WaveRep:./models/WaveRep/weights", "R50_nodown:./models/R50_nodown/weights/best.pt" , "CLIP-D:./models/CLIP-D/weights/best.pt",])
# detectors = [(WaveRepDetector(device=device), w_map.get("WaveRep"))]
detectors = []
for name in selected:
    if name not in available_d:
        print(f"[warn] Unknown detector '{name}', skipping")
        continue
    det = available_d[name]
    model_id = w_map.get(name)
    # Do not load here; BaseDetector.run_batch will call load for each detector
    detectors.append((det, model_id))

for det, model_id in detectors:
    det.load(model_id)


for folder in ["../b-free/samecat"]:
    images = list_images(folder)
    if limit_per_folder > 0:
        images = images[:limit_per_folder]

    # For better throughput, iterate per detector and use batching if available
    for det, _ in detectors:
        # If detector exposes batch_predict, use it in chunks
        if hasattr(det, 'batch_predict') and callable(getattr(det, 'batch_predict')):
            pbar = tqdm(total=len(images), desc=f"{det.name} on {os.path.basename(folder)}")
            for i in range(0, len(images), max(1, batch_size)):
                chunk = images[i:i + batch_size]
                with torch.no_grad():
                    confs = getattr(det, 'batch_predict')(chunk)  # type: ignore
                print(confs)
                for img_path, conf in zip(chunk, confs):
                    pred_flag = det.label_from_conf(float(conf))
                    # writer.writerow([
                    #     folder,
                    #     os.path.relpath(img_path, folder),
                    #     det.name,
                    #     f"{float(conf):.6f}",
                    #     str(pred_flag),
                    # ])
                    # f.flush()
                pbar.update(len(chunk))
            pbar.close()
        else:
            # Fallback to single-image predictions
            for img_path in tqdm(images, desc=f"{det.name} on {os.path.basename(folder)}"):
                img_tensor_cpu, _ = load_np_image(img_path, size=224)
                img_tensor = img_tensor_cpu.to(det.device)
                with torch.no_grad():
                    conf = float(det.forward(img_tensor, img_path))
                    print(conf)
                    pred_flag = det.label_from_conf(conf)
                        # writer.writerow([
                        #     folder,
                        #     os.path.relpath(img_path, folder),
                        #     det.name,
                        #     f"{conf:.6f}",
                        #     str(pred_flag),
                        # ])
                        # f.flush()
