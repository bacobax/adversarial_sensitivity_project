import os
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torchattacks as ta
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .utils import create_model

# Defaults aligned with WaveRepDetector
_DEFAULT_ARC = 'vit_base_patch14_reg4_dinov2.lvd142m'
_DEFAULT_CROP = 512


def _default_weights(detector_dir: str) -> str:
    return os.path.abspath(os.path.join(detector_dir, 'weights', 'weights_dinov2_G4.ckpt'))


class _Normalize(nn.Module):
    def __init__(self, mean: Sequence[float], std: Sequence[float]):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, -1, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class WaveRepForAttack(nn.Module):
    """Wrap WaveRep to provide 2-class logits and internal normalization for attacks.

    Expects inputs in [0,1] range with shape (N,3,H,W) already center-cropped to _DEFAULT_CROP.
    """
    
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base.eval()
        self.norm = _Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        out = self.base(x)
        # Convert single-logit output to 2-class logits for CE-based attacks
        if out.dim() == 2 and out.size(1) > 1:
            z = out[:, -1]
        else:
            z = out.squeeze(-1)
        logits2 = torch.stack([-z, z], dim=1)
        return logits2


def _preprocess_transform(cropping: int = _DEFAULT_CROP) -> transforms.Compose:
    return transforms.Compose([
        transforms.CenterCrop((cropping, cropping)),
        transforms.ToTensor(),  # [0,1]
    ])


def _build_attack(name: str, model: nn.Module, **kwargs):
    name = name.lower()
    if name == 'fgsm':
        return ta.FGSM(model, **kwargs)
    if name == 'pgd':
        return ta.PGD(model, **kwargs)
    if name == 'deepfool':
        return ta.DeepFool(model, **kwargs)
    raise ValueError(f"Unsupported attack '{name}'. Choose from: fgsm, pgd, deepfool")


def load_waverep_for_attack(
    device: torch.device,
    detector_dir: Optional[str] = None,
    weights: Optional[str] = None,
    arc: str = _DEFAULT_ARC,
    cropping: int = _DEFAULT_CROP,
) -> Tuple[WaveRepForAttack, transforms.Compose]:
    """Load WaveRep base model and return wrapped model for attacks + preprocessing transform."""
    if detector_dir is None:
        detector_dir = os.path.dirname(os.path.abspath(__file__))
    if weights is None:
        weights = _default_weights(detector_dir)
    base = create_model(weights, arc, cropping, device)
    model = WaveRepForAttack(base).to(device)
    tfm = _preprocess_transform(cropping)
    return model, tfm


class _PathsDataset(Dataset):
    def __init__(self, paths: Sequence[str], tfm: transforms.Compose):
        self.paths = list(paths)
        self.tfm = tfm
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        x = self.tfm(img)
        return x, idx


@torch.no_grad()
def load_images(paths: Sequence[str], tfm: transforms.Compose, device: torch.device) -> torch.Tensor:
    frames: List[torch.Tensor] = []
    for p in paths:
        img = Image.open(p).convert('RGB')
        frames.append(tfm(img))
    if not frames:
        return torch.empty(0, 3, _DEFAULT_CROP, _DEFAULT_CROP, device=device)
    batch = torch.stack(frames, 0).to(device)
    return batch


def attack_image_paths(
    image_paths: Sequence[str],
    labels: Sequence[int],
    attack: str,
    attack_kwargs: Optional[dict] = None,
    device: Optional[torch.device] = None,
    batch_size: int = 8,
    data_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    amp: bool = False,
    detector_dir: Optional[str] = None,
    weights: Optional[str] = None,
    arc: str = _DEFAULT_ARC,
    cropping: int = _DEFAULT_CROP,
) -> List[torch.Tensor]:
    """Run an adversarial attack on WaveRep for given image paths.

    Returns a list of adversarial tensors in [0,1] with shape (3,cropping,cropping), matching input order.
    """
    if attack_kwargs is None:
        attack_kwargs = {}
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(True)

    # NEW: precompute labels as a single tensor on the target device
    labels_tensor = torch.as_tensor(labels, device=device, dtype=torch.long)  # NEW
    
    model, tfm = load_waverep_for_attack(device, detector_dir, weights, arc, cropping)
    atk = _build_attack(attack, model, **attack_kwargs)
    
    N = len(image_paths)
    adv_all = torch.empty(N, 3, cropping, cropping, dtype=torch.float32)  # NEW
    
    ds = _PathsDataset(image_paths, tfm)
    if data_workers < 0:
        data_workers = 0
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=data_workers,
        pin_memory=pin_memory if device.type == 'cuda' else False,
        prefetch_factor=prefetch_factor if data_workers > 0 else None,
        persistent_workers=True if data_workers > 0 else False,
    )
    use_amp = bool(amp) and device.type == 'cuda'
    for batch, idxs in tqdm(loader, total=len(loader), desc=f"{attack.upper()} attack", leave=False):
        if device.type == 'cuda' and pin_memory:
            batch = batch.pin_memory()
        batch = batch.to(device, non_blocking=True)
        
        # lbs = torch.tensor([labels[i] for i in idxs.tolist()], device=device, dtype=torch.long)
        lbs = labels_tensor[idxs.to(device, non_blocking=True)]  # NEW
        
        if use_amp:
            from torch.cuda.amp import autocast
            with autocast():
                adv = atk(batch, lbs)
        else:
            adv = atk(batch, lbs)
        adv = adv.detach().cpu()
        
        # for j, i in enumerate(idxs.tolist()):
        #     adv_images[i] = adv[j]
        adv_all[idxs] = adv  # NEW
    
    return [adv_all[i] for i in range(N)]  # NEW
