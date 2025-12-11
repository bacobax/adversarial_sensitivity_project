from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime import lime_image
from PIL import Image
from torchvision import transforms as T

# --- Utilities for LIME over tensors normalized with ImageNet stats ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _denorm_to_pil(img: torch.Tensor) -> Image.Image:
    """
    Convert a single normalized tensor (3,H,W) in ImageNet space to a PIL image (RGB).
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(3, 1, 1)
    x = img * std + mean
    x = x.clamp(0, 1)
    x = (x * 255.0).round().byte().cpu()
    nd = x.permute(1, 2, 0).numpy()
    return Image.fromarray(nd)


def _denorm_to_numpy(img: torch.Tensor) -> np.ndarray:
    """
    Convert a single normalized tensor (3,H,W) to uint8 numpy array (H,W,3).
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype, device=img.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=img.dtype, device=img.device).view(3, 1, 1)
    x = img * std + mean
    x = x.clamp(0, 1)
    x = (x * 255.0).round().byte().cpu()
    return x.permute(1, 2, 0).numpy()


def _get_preprocess_transform() -> T.Compose:
    """
    Transform PIL -> normalized tensor matching ImageNet stats without resizing.
    """
    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return T.Compose([T.ToTensor(), normalize])


def _make_batch_predict_fn(model: nn.Module, device: torch.device):
    preprocess = _get_preprocess_transform()
    
    def batch_predict(images: List[Image.Image]) -> np.ndarray:
        model.eval()
        batch = torch.stack([preprocess(Image.fromarray(i) if isinstance(i, np.ndarray) else i) for i in images], dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
        # SPECIALIZED: treat last logit as "fake" score and map to 2-class probs via sigmoid
        if logits.ndim == 1:
            last = logits
        elif logits.ndim == 2:
            last = logits[:, -1]
        else:
            last = logits.view(logits.size(0), -1)[:, -1]
        p1 = torch.sigmoid(last)
        probs = torch.stack([1.0 - p1, p1], dim=1)
        return probs.detach().cpu().numpy()
    
    return batch_predict


def _find_last_conv(module: nn.Module) -> nn.Module:
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("No Conv2d layer found to use as target for Grad-based explainability.")
    return last_conv


class _HookStore:
    def __init__(self, layer: nn.Module):
        self.activ: Optional[torch.Tensor] = None
        self.grads: Optional[torch.Tensor] = None
        self.h1 = layer.register_forward_hook(self._fh)
        self.h2 = layer.register_full_backward_hook(self._bh)
    
    def _fh(self, m, i, o):
        self.activ = o.detach()
    
    def _bh(self, m, gi, go):
        self.grads = go[0].detach()
    
    def close(self):
        self.h1.remove()
        self.h2.remove()


def _select_target(logits: torch.Tensor, class_idx: Optional[int]) -> torch.Tensor:
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)
    if logits.size(1) == 1:
        return logits[:, 0]
    if class_idx is None:
        class_idx = min(1, logits.size(1) - 1)
    return logits[:, class_idx]


def _normalize_heatmap(cam: torch.Tensor, out_size: Tuple[int, int]) -> torch.Tensor:
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=out_size, mode="bilinear", align_corners=False)
    vmin = cam.amin(dim=(2, 3), keepdim=True)
    vmax = cam.amax(dim=(2, 3), keepdim=True)
    cam = (cam - vmin) / (vmax - vmin + 1e-8)
    return cam


@torch.no_grad()
def _ensure_eval_on_device(model: nn.Module, device: Optional[torch.device]) -> nn.Module:
    if device is not None:
        model = model.to(device)
    model.eval()
    return model


def gradcam_map(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model = _ensure_eval_on_device(model, device)
    if device is not None:
        image = image.to(device)
    if target_layer is None:
        target_layer = _find_last_conv(model)
    
    hook = _HookStore(target_layer)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        target = _select_target(logits, class_idx)
        # backward with grad
        target.sum().backward()
        if hook.activ is None or hook.grads is None:
            raise RuntimeError("Failed to capture activations/gradients for GradCAM.")
        # weights: channel-wise mean of gradients
        weights = hook.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * hook.activ).sum(dim=1, keepdim=True)
        cam = _normalize_heatmap(cam, image.shape[-2:])
        return cam.detach().cpu(), logits.detach().cpu()
    finally:
        hook.close()


def lime_explain(
    model: nn.Module,
    image: torch.Tensor,
    device: Optional[torch.device] = None,
    class_idx: Optional[int] = None,
    top_labels: Optional[int] = None,
    num_samples: int = 400,
    positive_only: bool = True,
    n_segments: int = 24,
    n_features: int = 100000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LIME masks for a batch tensor normalized with ImageNet stats.

    Returns:
        cam: (B,1,H,W) float tensor in [0,1]
        logits: (B,K) raw model logits for the original batch
    """
    assert image.ndim == 4, "Expected batch tensor (B,3,H,W)"
    model = _ensure_eval_on_device(model, device)
    if device is not None:
        image = image.to(device)
    
    B, C, H, W = image.shape
    
    # Prepare predictor and explainer
    batch_predict = _make_batch_predict_fn(model, device if device is not None else torch.device('cpu'))
    explainer = lime_image.LimeImageExplainer()
    
    # Fast SLIC segmentation to reduce superpixels and speed up
    try:
        from skimage.segmentation import slic
        
        def segmentation_fn(x):
            return slic(x, n_segments=n_segments, compactness=15, start_label=0)
    except Exception:
        segmentation_fn = None  # fall back to default
    
    cams: List[torch.Tensor] = []
    for b in range(B):
        np_img = _denorm_to_numpy(image[b])
        explanation = explainer.explain_instance(
            np_img,
            batch_predict,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn,
            random_seed=0,
            batch_size=52,
            # num_features=n_features,
            # fudged_image=cv2.cvtColor(fudged_image, cv2.COLOR_BGR2RGB),
        )
        
        # Determine which class to visualize
        if class_idx is None:
            # For single-logit detectors we visualize the positive class (index 1)
            target_cls = 1
        else:
            target_cls = int(class_idx)
        
        temp, mask = explanation.get_image_and_mask(
            label=target_cls,
            positive_only=positive_only,
            num_features=n_features,
            hide_rest=False,
        )
        m = mask
        
        # Convert mask (H,W) -> (1,H,W) float in [0,1]
        m = torch.from_numpy(m.astype(np.float32))[None, ...]
        cams.append(m)
    
    cam_batch = torch.stack(cams, dim=0)  # (B,1,H,W)
    return cam_batch


def gradsam_map(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gradient-activation saliency map: element-wise product between gradients and activations
    aggregated over channels (no global pooling weights), then ReLU and normalize.
    This provides a sharper alternative to GradCAM, often dubbed as Grad-SAM.
    """
    model = _ensure_eval_on_device(model, device)
    if device is not None:
        image = image.to(device)
    if target_layer is None:
        target_layer = _find_last_conv(model)
    
    hook = _HookStore(target_layer)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        target = _select_target(logits, class_idx)
        target.sum().backward()
        if hook.activ is None or hook.grads is None:
            raise RuntimeError("Failed to capture activations/gradients for GradSAM.")
        # element-wise gradient * activation
        elem = hook.grads * hook.activ
        cam = elem.sum(dim=1, keepdim=True)
        cam = _normalize_heatmap(cam, image.shape[-2:])
        return cam.detach().cpu(), logits.detach().cpu()
    finally:
        hook.close()


def grad_input_explain(
    model: nn.Module,
    x: torch.Tensor,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
    n_samples: int = 1,
    sigma: float = 0.1,
    aggregate_channels: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Gradient × Input saliency maps with optional SmoothGrad.
    
    Args:
        model: PyTorch model that outputs a single logit per image.
        x: Input tensor of shape (B, C, H, W), already preprocessed.
        class_idx: Target class index. If None, uses the last logit.
        device: Device to run the model on.
        n_samples: Number of noise samples for SmoothGrad. If <=1, uses vanilla Gradient × Input.
        sigma: Standard deviation of Gaussian noise for SmoothGrad, relative to input range.
        aggregate_channels: If True, returns (B, 1, H, W); else (B, C, H, W).
        
    Returns:
        saliency: Saliency map tensor.
        logits: Raw model logits for the input batch.
    """
    model = _ensure_eval_on_device(model, device)
    if device is not None:
        x = x.to(device)
    
    B, C, H, W = x.shape
    x_orig = x.detach().clone()
    
    # Initialize accumulator for SmoothGrad
    grad_accum = torch.zeros_like(x_orig)
    
    for _ in range(max(1, n_samples)):  # At least 1 iteration
        if n_samples > 1:
            # Add noise for SmoothGrad
            noise = torch.randn_like(x_orig) * sigma
            x_noisy = (x_orig + noise).detach().requires_grad_(True)
        else:
            x_noisy = x_orig.clone().requires_grad_(True)
        
        # Forward pass
        logits = model(x_noisy).view(-1)  # (B,)
        
        # Select target
        if class_idx is not None and logits.dim() > 1:
            target = logits[:, class_idx].sum()
        else:
            target = logits.sum()  # Sum over batch for efficiency
        
        # Backward pass
        model.zero_grad()
        if x_noisy.grad is not None:
            x_noisy.grad.zero_()
        target.backward()
        
        grad_accum += x_noisy.grad.detach()
    
    # Average gradients if using SmoothGrad
    if n_samples > 1:
        grads = grad_accum / n_samples
    else:
        grads = grad_accum
    
    # Gradient × Input
    saliency = grads * x_orig
    
    # Aggregate channels if requested
    if aggregate_channels:
        # L1 norm over channels
        saliency = saliency.abs().sum(dim=1, keepdim=True)  # (B, 1, H, W)
    
    # Normalize per image for visualization
    saliency = _normalize_heatmap(saliency, (H, W))
    
    # Get final logits without computing gradients
    with torch.no_grad():
        logits = model(x_orig)
    
    return saliency.detach().cpu(), logits.detach().cpu()
