import math
import os
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)

from utils import create_transform, create_model  # type: ignore
from explain import gradsam_map, lime_explain  # type: ignore


def _auto_select_gradcam_layer(model: nn.Module) -> nn.Module:  # NEW
    """
    Try to select a reasonable target layer for Grad-CAM.
    - Prefer the last Conv2d if present.
    - Otherwise, for ViT-like models, prefer the last block.
    This is heuristic but robust for most backbones.
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return last_conv
    
    # Fallback: try to find a 'blocks' attribute (ViT-style)
    if hasattr(model, "blocks") and len(getattr(model, "blocks")) > 0:
        return model.blocks[-1]
    
    # Sometimes the actual backbone is nested in .model
    if hasattr(model, "model"):
        backbone = model.model
        if hasattr(backbone, "blocks") and len(getattr(backbone, "blocks")) > 0:
            return backbone.blocks[-1]
    
    # As a last resort, just use the whole model (not ideal, but avoids crash)
    return model


def gradcam_map(  # NEW
    model: nn.Module,
    x: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    class_idx: Optional[int] = None,
    device: Optional[torch.device] = None,
):  # NEW
    """
    Compute Grad-CAM for a batch.

    Args:
        model: backbone with logits output.
        x: (B, 3, H, W) normalized input.
        target_layer: module to hook; if None, pick a sensible default.
        class_idx: which logit to target; if None, use last logit (e.g. 'fake').
        device: optional device override.

    Returns:
        cam: (B, 1, H, W) in [0, 1]
        logits_all: (B, K) raw logits.
    """
    if device is not None:
        x = x.to(device)
        model = model.to(device)
    
    model.eval()
    
    if target_layer is None:
        target_layer = _auto_select_gradcam_layer(model)
    
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    
    def fwd_hook(_, __, output):
        activations.append(output)
    
    def bwd_hook(_, grad_in, grad_out):
        # grad_out is a tuple; we want the gradient w.r.t. the output of the layer
        gradients.append(grad_out[0])
    
    # Register hooks
    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)
    
    try:
        # Forward pass
        logits_all = model(x)
        if logits_all.dim() == 1:
            logits_all = logits_all.unsqueeze(1)  # (B,) -> (B,1)
        
        B = logits_all.shape[0]
        
        # Pick target class/logit
        if class_idx is None:
            target = logits_all[:, -1]  # assume last logit is 'fake'
        else:
            target = logits_all[:, class_idx]
        
        # Backward pass
        model.zero_grad()
        # sum over batch so backward has a scalar
        target_sum = target.sum()
        target_sum.backward(retain_graph=True)
        
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")
        
        act = activations[-1]
        grad = gradients[-1]
        
        # Handle different shapes:
        # - CNN: act, grad: (B, C, H, W)
        # - ViT: act, grad: (B, N, C) or (B, C, N)
        if act.dim() == 4:
            # Standard CNN case
            # global average pool gradients over spatial dims
            weights = grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            cam = (weights * act).sum(dim=1, keepdim=True)  # (B, 1, H, W)
            cam = F.relu(cam)
            # Normalize and upsample to input size
            cam = _normalize_cam(cam)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        elif act.dim() == 3:
            # Token-based (ViT) case
            # Try to infer layout: (B, N, C) vs (B, C, N)
            if act.shape[1] == grad.shape[1]:  # (B, N, C)
                tokens = act
                grad_tokens = grad
                # Drop CLS token if present
                if tokens.shape[1] > 1 and (tokens.shape[1] - 1) in (196, 256, 324, 400, 576):
                    tokens = tokens[:, 1:, :]
                    grad_tokens = grad_tokens[:, 1:, :]
                # weights: mean over tokens
                weights = grad_tokens.mean(dim=1, keepdim=True)  # (B, 1, C)
                cam_tokens = (weights * tokens).sum(dim=2)  # (B, N_tokens)
            else:
                # Assume (B, C, N)
                tokens = act
                grad_tokens = grad
                weights = grad_tokens.mean(dim=2, keepdim=True)  # (B, C, 1)
                cam_tokens = (weights * tokens).sum(dim=1)  # (B, N_tokens)
            
            # Map tokens to square grid
            N_tokens = cam_tokens.shape[1]
            side = int(math.sqrt(N_tokens))
            if side * side != N_tokens:
                # Fallback: pad or truncate to nearest square
                side = int(math.sqrt(N_tokens))
                N_square = side * side
                cam_tokens = cam_tokens[:, :N_square]
            cam = cam_tokens.view(B, 1, side, side)  # (B, 1, Ht, Wt)
            cam = F.relu(cam)
            cam = _normalize_cam(cam)
            cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            raise RuntimeError(f"Unsupported activation shape for Grad-CAM: {act.shape}")
        
        return cam.detach(), logits_all.detach()
    finally:
        # Always clean up hooks
        handle_fwd.remove()
        handle_bwd.remove()


def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:  # NEW
    """
    Normalize CAM per image to [0, 1].
    cam: (B, 1, H, W)
    """
    B = cam.shape[0]
    cam_reshaped = cam.view(B, -1)
    cam_min = cam_reshaped.min(dim=1, keepdim=True)[0]
    cam_max = cam_reshaped.max(dim=1, keepdim=True)[0]
    denom = (cam_max - cam_min).clamp(min=1e-8)
    cam_norm = (cam_reshaped - cam_min) / denom
    return cam_norm.view_as(cam)
