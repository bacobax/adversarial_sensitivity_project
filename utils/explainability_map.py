import torch
import torch.nn.functional as F
from typing import Any, Tuple

from .vulnerability_map import _to_device_tensor


class _GradCAMExtractor:
    """Utility class that collects activations and gradients for Grad-CAM."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._forward_hook),
            target_layer.register_full_backward_hook(self._backward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple with a single tensor for conv layers
        self.gradients = grad_output[0].detach()

    def cleanup(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(self, image: torch.Tensor, class_idx: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run forward/backward pass and return (cam, logits)."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)

        if logits.size(1) == 1:
            target = logits[:, 0]
        else:
            if class_idx is None:
                class_idx = min(1, logits.size(1) - 1)
            target = logits[:, class_idx]

        target.sum().backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Gradients or activations were not captured for Grad-CAM.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)

        vmin = cam.amin(dim=(2, 3), keepdim=True)
        vmax = cam.amax(dim=(2, 3), keepdim=True)
        cam = (cam - vmin) / (vmax - vmin + 1e-8)

        return cam, logits


def _resolve_target_layer(detector_model: torch.nn.Module, is_resnet: bool) -> torch.nn.Module:
    if is_resnet and hasattr(detector_model, "layer4"):
        return detector_model.layer4[-1]

    clip_backbone = getattr(detector_model, "bb", None)
    if clip_backbone and isinstance(clip_backbone, (list, tuple)) and clip_backbone:
        backbone = clip_backbone[0]
        conv = getattr(backbone, "conv1", None)
        if conv is not None:
            return conv

    raise ValueError("Unable to locate a suitable layer for Grad-CAM.")


def get_explainability_map(
    image: Any,
    detector_model: torch.nn.Module,
    is_resnet: bool,
    device: str = "cpu",
    target_class: int | None = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Grad-CAM explainability map for the provided detector."""
    image_tensor = _to_device_tensor(image, device).float()
    detector_model = detector_model.to(device)
    detector_model.eval()

    target_layer = _resolve_target_layer(detector_model, is_resnet)
    grad_cam = _GradCAMExtractor(detector_model, target_layer)
    try:
        cam, logits = grad_cam(image_tensor, class_idx=target_class)
    finally:
        grad_cam.cleanup()

    return cam.detach().cpu(), logits.detach().cpu()
