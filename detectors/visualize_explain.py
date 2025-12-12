import argparse
import os
from typing import List

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from detect import _detector_class_by_name
from support.base_detector import BaseDetector
from support.detect_utils import get_device

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Root directory for cached explainability results
CACHE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'datasets', 'outputs')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def list_images(folder: str) -> List[str]:
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    files: List[str] = []
    for root, _, fnames in os.walk(folder):
        for f in fnames:
            if os.path.splitext(f)[1].lower() in exts:
                files.append(os.path.join(root, f))
    files.sort()
    return files


def load_batch(paths: np.ndarray, size: int) -> torch.Tensor:
    tfm = T.Compose([
        T.Resize(size),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    imgs = [tfm(Image.open(p).convert('RGB')) for p in paths]
    return torch.stack(imgs, dim=0)


def denormalize(batch: torch.Tensor) -> torch.Tensor:
    return (batch * IMAGENET_STD.to(batch.device)) + IMAGENET_MEAN.to(batch.device)


def colorize_cam(cam: torch.Tensor) -> torch.Tensor:
    """
    Simple red colormap: map cam [1,H,W] -> [3,H,W] with red channel = cam, others = (1-cam)*0.
    """
    # Robust per-image normalization to [-1,1] using PyTorch
    max_abs = torch.quantile(cam.abs().flatten(), 0.99)
    if max_abs > 0:
        cam /= max_abs
    
    red = cam.clamp(0, 1)
    blue = -cam.clamp(-1, 0)
    zeros = torch.zeros_like(cam)
    # make red where cam > 0 and blue where < 0
    bgr = torch.cat([blue, (blue + red) / 3, red], dim=0)
    return bgr


def overlay(img: torch.Tensor, cam_rgb: torch.Tensor, alpha: float):
    over = (1 - alpha) * img + alpha * cam_rgb
    over = over.clamp(0, 1)
    return (over.mul(255).add(0.5)).byte().cpu().permute(1, 2, 0).numpy()


def save_tensor_image(t: torch.Tensor, path: str) -> None:
    t = t.clamp(0, 1)
    nd = (t.mul(255).add(0.5)).byte().cpu().permute(1, 2, 0).numpy()
    Image.fromarray(nd).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize explainability overlays for detector batches")
    parser.add_argument('--categories', nargs='+', default=['real', 'samecat', 'diffcat'], help='Folder with input folders')
    parser.add_argument('--dataset', type=str, default='../datasets/b-free', help='Folders with corresponding adversarial images for vulnerability analysis')
    parser.add_argument('--attacked-dataset', type=str, default='../datasets/adv_attacks',
                        help='Folders with corresponding adversarial images for vulnerability analysis')
    parser.add_argument('--detector', type=str, default='WaveRep', help='Detector name (see detect.py mapping)')
    parser.add_argument('--attack', type=str, default=None, help='Folders with corresponding adversarial images for vulnerability analysis')
    parser.add_argument('--model-id', type=str, default=None, help='Optional model identifier/path for detector.load')
    parser.add_argument('--output', type=str, default='out', help='Output directory for overlays')
    parser.add_argument('--size', type=int, default=512, help='Input resize/crop size')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--method', type=str, default='lime', help='Explainability method')
    parser.add_argument('--alpha', type=float, default=1, help='Overlay alpha in [0,1]')
    parser.add_argument('--class-idx', type=int, default=None, help='Target class index for multi-class models')
    parser.add_argument('--metrics', type=str, default=None, help='Metrics to compute for vulnerability analysis')
    parser.add_argument('--limit', type=int, default=0, help='Metrics to compute for vulnerability analysis')
    parser.add_argument('--recompute-logits', action='store_true', help='Recompute logits for all images and update NPZ cache, even if cached entries exist')
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(os.path.join(args.output, args.method), exist_ok=True)
    os.makedirs(CACHE_ROOT, exist_ok=True)
    
    device = get_device()
    
    # Instantiate detector
    DetClass = _detector_class_by_name(args.detector)
    detector: BaseDetector = DetClass(device=device)  # type: ignore
    detector.load(args.model_id)
    
    # Sanity check that this detector supports explain()
    if not hasattr(detector, "explain"):
        raise RuntimeError(f"Detector '{args.detector}' does not implement explain()")
    
    # Gather images
    images = {}
    folders = []
    for folder in args.categories:
        folders.append(os.path.join(args.dataset, folder))
        if args.attack is not None and args.attacked_dataset is not None:
            folders.append(os.path.join(args.attacked_dataset, args.attack, folder))
    for folder in folders:
        paths = list_images(folder)
        if args.limit > 0:
            paths = paths[:args.limit]
        if not paths:
            print(f"Warning: No images found in {folder}")
            continue
        for path in paths:
            base = os.path.basename(path)
            images.setdefault(base, []).append(path)
    images = np.array(list(images.values()))  #: group by filename
    
    # Process in batches
    bs = max(1, int(args.batch_size))
    cache_dir = os.path.join(CACHE_ROOT, args.detector, args.method)
    os.makedirs(cache_dir, exist_ok=True)
    
    for start in tqdm(range(0, len(images), bs)):
        end = min(start + bs, len(images))
        batch_paths = images[start:end].flatten()
        batch = np.array([Image.open(p).convert('RGB') for p in batch_paths], dtype=np.uint8)

        # Prepare grouped structure and load any existing cache per base image
        bases = [os.path.basename(p) for p in batch_paths]
        grouped_images = {}
        
        for base in set(bases):
            cache_path = os.path.join(cache_dir, f"{os.path.splitext(base)[0]}.npz")
            if os.path.exists(cache_path):
                try:
                    data = np.load(cache_path, allow_pickle=True)
                    folder_data = data['folder_data'].item()
                    grouped_images[base] = folder_data
                except Exception as e:
                    print(f"[warn] Failed to load cache '{cache_path}': {e}. Recomputing this entry.")
                    # Best effort to remove corrupted cache so future runs don't keep failing.
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                    grouped_images[base] = {}
            else:
                grouped_images[base] = {}
        
        # Decide which (attack, category) entries still need computation
        idx_to_compute = []
        key_per_index = []
        for i, p in enumerate(batch_paths):
            base = bases[i]
            category = p.split('/')[-2]
            attack = p.split('/')[-3]
            key = (attack, category)
            key_per_index.append(key)
            if key not in grouped_images[base]:
                idx_to_compute.append(i)

        logits_all = None
        if idx_to_compute or args.recompute_logits:
            with torch.no_grad():
                logits_all = detector.forward(batch)

        if idx_to_compute:
            sub_batch = batch[idx_to_compute]

            # Compute explainability maps only for uncached entries
            cams_sub = detector.explain(
                sub_batch, method=args.method, class_idx=args.class_idx,
            )

            # Fill grouped_images for newly computed samples
            for local_idx, global_idx in enumerate(idx_to_compute):
                p = batch_paths[global_idx]
                base = bases[global_idx]
                cam = cams_sub[local_idx]
                attack, category = key_per_index[global_idx]
                folder_data = grouped_images.setdefault(base, {})
                logit_val = logits_all[global_idx] if logits_all is not None else None
                entry = {
                    'img': p,
                    'cam': cam,
                }
                if logit_val is not None:
                    entry['logit'] = logit_val
                folder_data[(attack, category)] = entry

        # Optionally recompute logits for all images in this batch and update cache entries
        if args.recompute_logits and logits_all is not None:
            for i, p in enumerate(batch_paths):
                base = bases[i]
                attack, category = key_per_index[i]
                folder_data = grouped_images.setdefault(base, {})
                entry = folder_data.get((attack, category))
                if entry is None:
                    entry = {'img': p}
                    folder_data[(attack, category)] = entry
                entry['logit'] = logits_all[i]

        # Save/extend cache for all bases touched in this batch
        for base, folder_data in grouped_images.items():
            cache_path = os.path.join(cache_dir, f"{os.path.splitext(base)[0]}.npz")
            meta = {
                'model': args.detector,
                'method': args.method,
            }
            np.savez_compressed(
                cache_path,
                folder_data=np.array(folder_data, dtype=object),
                meta=np.array(meta, dtype=object),
            )
        
        gt_masks = {
            'real': None,
            'samecat': 'mask',
            'diffcat': 'bbox',
        }
        
        # Save collages
        for base, folder_data in grouped_images.items():
            out_path = os.path.join(args.output, args.method, base)
            attacks = [a for a in ['b-free', args.attack] if (a, 'real') in folder_data]
            categories = [c for c in args.categories if ('b-free', c) in folder_data]
            out = [[] for _ in range(len(attacks) + (1 if args.attack is None else 2))]  # first row for ground truth, last for diff
            
            for i, attack in enumerate(attacks):
                for cat in categories:
                    cam = folder_data[(attack, cat)]['cam']
                    heat = colorize_cam(cam).detach().cpu()
                    img = torch.from_numpy(cv2.imread(folder_data[(attack, cat)]['img']).astype(np.float32))
                    img = img.permute(2, 0, 1) / 255.
                    over = overlay(img, heat, alpha=args.alpha)
                    out[i + 1].append(over)
            
            if args.attack is not None:
                for cat in categories:  # fixme
                    cam_orig = folder_data[('b-free', cat)]['cam']
                    cam_adv = folder_data[(args.attack, cat)]['cam']
                    eps = 1e-8
                    cam_orig_norm = cam_orig / (cam_orig.abs().sum() + eps)
                    cam_adv_norm = cam_adv / (cam_adv.abs().sum() + eps)
                    cam_diff = cam_orig_norm - cam_adv_norm
                    # vulnerability = (cam_orig_norm + cam_diff) * cam_diff.abs()  # FIXME
                    vulnerability = cam_diff.abs()  # FIXME
                    vulnerability = vulnerability.clamp(min=0)
                    
                    folder_data[('diff', cat)] = {'cam': vulnerability}
                    heat = colorize_cam(vulnerability).detach().cpu()
                    img = torch.from_numpy(cv2.imread(folder_data[('b-free', cat)]['img']).astype(np.float32))
                    img = img.permute(2, 0, 1) / 255.
                    over = overlay(img, heat, alpha=args.alpha).astype(np.float32)
                    # print(folder_data[(args.attack, cat)]['logit'].item())
                    sigm = torch.sigmoid(folder_data[(args.attack, cat)]['logit']).item()
                    print(sigm)
                    # over *= sigm
                    out[-1].append(over.astype(np.uint8))
            
            for i, cat in enumerate(categories):
                im = cv2.imread(os.path.join(args.dataset, cat, base), cv2.IMREAD_COLOR_BGR)
                if gt_masks[cat] is not None:
                    mask = cv2.imread(os.path.join(args.dataset, gt_masks[cat], base), cv2.IMREAD_GRAYSCALE)
                    im = cv2.bitwise_and(im, im, mask=mask)
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    for j in range(1, len(out)):
                        cv2.drawContours(out[j][i], contours, -1, [255, 255, 255], 2)
                out[0].append(im)
            
            # out = np.array(out, dtype=np.uint8)
            for i in range(len(out)):
                out[i] = np.hstack(out[i])
            out = np.vstack(out)
            out = cv2.resize(out, (out.shape[1] // 3, out.shape[0] // 3))
            cv2.imwrite(out_path, out)


if __name__ == '__main__':
    main()
