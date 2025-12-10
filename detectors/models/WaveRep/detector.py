import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pywt
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from support.base_detector import BaseDetector

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)

from utils import create_transform, create_model  # type: ignore
from explain import grad_input_explain, gradsam_map, lime_explain  # type: ignore
from gradcam import gradcam_map


class WaveRepDetector(BaseDetector):
    name = 'WaveRep'
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device=device)
        self.transform = None
        self.cropping = 512
        self.arc = 'vit_base_patch14_reg4_dinov2.lvd142m'
        # Inference knobs
        self.use_tta = True  # horizontal flip TTA
        self.use_amp = True  # CUDA autocast
        self._gradcam_target_layer: Optional[nn.Module] = None
        # cache for attention rollout (not strictly required but handy)
        self._attn_hook_handles = []
    
    def _clear_attn_hooks(self) -> None:
        """Remove any previously registered attention hooks."""
        for h in self._attn_hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._attn_hook_handles = []
    
    def _attention_rollout(
        self,
        x: torch.Tensor,
        head_fusion: str = "mean",
        discard_ratio: float = 0.9,
        class_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate attention rollout visualization for the input.
        Enhanced with detailed debugging for attention map collection.
        """
        assert self.model is not None, "Model not loaded"
        self.model.eval()
        attn_maps: List[torch.Tensor] = []  # NEW

        def _attn_hook(module: nn.Module, inputs, output):  # NEW
            """
            Forward hook on attention module:
            - Recompute Q,K from the input using module.qkv
            - Build attention weights: softmax(q k^T / sqrt(d_head))
            - Store as (B, heads, N, N)
            """  # NEW
            if not inputs:  # NEW
                return  # NEW
            x_in = inputs[0]  # expected shape: (B, N, C)  # NEW
            if not isinstance(x_in, torch.Tensor) or x_in.dim() != 3:  # NEW
                return  # NEW
            B, N, C = x_in.shape  # NEW

            if not hasattr(module, "qkv") or not hasattr(module, "num_heads"):  # NEW
                return  # NEW

            # qkv projection: (B, N, 3*C)  # NEW
            qkv = module.qkv(x_in)  # type: ignore[attr-defined]  # NEW
            # reshape to (3, B, heads, N, d_head)  # NEW
            num_heads = module.num_heads  # type: ignore[attr-defined]  # NEW
            head_dim = C // num_heads  # NEW
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 3, 1, 4)  # NEW
            q, k, _v = qkv[0], qkv[1], qkv[2]  # (B, heads, N, d_head)  # NEW

            # Compute scaled dot-product attention: (B, heads, N, N)  # NEW
            scale = getattr(module, "scale", None)  # NEW
            if scale is None:  # NEW
                scale = 1.0 / math.sqrt(head_dim)  # NEW
            attn = (q @ k.transpose(-2, -1)) * float(scale)  # NEW
            attn = attn.softmax(dim=-1)  # NEW

            attn_maps.append(attn.detach())  # NEW

        # Register hooks on all modules that look like attention (have qkv & num_heads).  # NEW
        self._clear_attn_hooks()  # NEW
        for name, module in self.model.named_modules():  # NEW
            if hasattr(module, "qkv") and hasattr(module, "num_heads"):  # NEW
                handle = module.register_forward_hook(_attn_hook)  # NEW
                self._attn_hook_handles.append(handle)  # NEW

        # Forward pass to collect attention + logits.  # NEW
        with torch.no_grad():  # NEW
            logits_all = self.model(x)  # shape: (B, K)  # NEW

        # Clean hooks as soon as we are done.  # NEW
        self._clear_attn_hooks()  # NEW

        if not attn_maps:  # NEW
            print("[warn] Attention Rollout: no attention maps collected; returning uniform CAM.")  # NEW
            B, _, H, W = x.shape  # NEW
            cam = torch.ones((B, 1, H, W), device=x.device)  # NEW
            return cam, logits_all  # NEW

        
        # --- Fuse heads and build residual-augmented attention for each layer ---
        fused_attn: List[torch.Tensor] = []
        for attn in attn_maps:  # each: (B, H, N, N)
            if head_fusion == "mean":
                A = attn.mean(dim=1)  # (B, N, N)
            elif head_fusion == "max":
                A = attn.max(dim=1).values
            elif head_fusion == "min":
                A = attn.min(dim=1).values
            else:
                raise ValueError(f"Unknown head_fusion mode: {head_fusion}")
            
            # Optionally discard lowest attentions as in "attention rollout" variants.
            if discard_ratio > 0.0:
                B, N, _ = A.shape
                flat = A.view(B, -1)  # (B, N*N)
                num_discard = int(flat.shape[1] * discard_ratio)
                if num_discard > 0:
                    thresh, _ = torch.kthvalue(flat, num_discard, dim=1)
                    thresh = thresh.view(B, 1, 1)
                    A = torch.where(A < thresh, torch.zeros_like(A), A)
            
            # Add identity (residual) and renormalize rows.
            I = torch.eye(A.size(-1), device=A.device).unsqueeze(0)  # (1, N, N)
            A = A + I
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            fused_attn.append(A)
        
        # --- Rollout: combine attention matrices across layers ---
        # Start from identity: each token attends to itself.
        B, N, _ = fused_attn[0].shape
        joint = torch.eye(N, device=x.device).unsqueeze(0).expand(B, -1, -1)  # (B, N, N)
        for A in fused_attn:
            joint = A @ joint
        
        # joint[b, 0] is the attention of the CLS token over all tokens after rollout.
        cls_attn = joint[:, 0]  # (B, N)
        
        # --- Map token attentions to spatial grid (patch tokens only) ---
        # Infer grid from input size & patch size (14 for vit_base_patch14_...).
        _, _, H, W = x.shape
        patch_size = 14
        gh, gw = H // patch_size, W // patch_size
        num_patches = gh * gw
        
        if N <= num_patches:
            # Fallback: assume first token is CLS and the rest are patches.
            n_special = 1
        else:
            # Tokens beyond CLS + registers are assumed to be patches.
            n_special = N - num_patches
            if n_special < 1:
                n_special = 1
        
        patch_attn = cls_attn[:, n_special:]  # (B, num_patches)
        if patch_attn.shape[1] != num_patches:
            # Shape mismatch: fall back to using as many tokens as we can reshape.
            usable = min(patch_attn.shape[1], num_patches)
            patch_attn = patch_attn[:, :usable]
            gh = int(np.sqrt(usable))  # type: ignore[name-defined]
            gw = usable // gh
        
        cam_small = patch_attn.view(B, 1, gh, gw)  # (B, 1, gh, gw)
        
        # Upsample CAM to full image resolution.
        cam = torch.nn.functional.interpolate(
            cam_small,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, H, W)
        
        # Normalize each CAM to [0, 1].
        cam_min = cam.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = cam.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        
        return cam, logits_all
    
    def _default_weights(self) -> str:
        # Default weights location within WaveRep repo
        weights_path = os.path.abspath(os.path.join(DETECTOR_DIR, 'weights', 'weights_dinov2_G4.ckpt'))
        return weights_path
    
    def load(self, model_id: Optional[str] = None) -> None:
        device = self.device
        weights = model_id if model_id else self._default_weights()
        if not os.path.exists(weights):
            raise FileNotFoundError(f"WaveRep weights not found: {weights}")
        self.transform = create_transform(self.cropping)
        self.model = create_model(weights, self.arc, self.cropping, device)
        self.model.eval()
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        # WaveRep expects its own crop/normalization; load from path and ignore generic tensor
        assert self.model is not None, "Model not loaded"
        assert self.transform is not None, "Transform not initialized"
        img = Image.open(image_path).convert('RGB')
        frame = self.transform(img)
        frames = [frame]
        if self.use_tta:
            frames.append(torch.flip(frame, dims=[2]))  # horizontal flip (C,H,W) -> flip W
        batch = torch.stack(frames, 0).to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    logits = self.model(batch)[:, -1]
            else:
                logits = self.model(batch)[:, -1]
            probs = torch.sigmoid(logits).detach().cpu()
            prob = float(probs.mean().item())
        return prob
    
    def batch_predict(self, image_paths):
        assert self.model is not None, "Model not loaded"
        assert self.transform is not None, "Transform not initialized"
        frames = []
        for p in image_paths:
            img = Image.open(p).convert('RGB')
            frames.append(self.transform(img))
        if not frames:
            return []
        batch = torch.stack(frames, 0).to(self.device)
        with torch.no_grad():
            if self.device.type == 'cuda' and self.use_amp:
                from torch.amp import autocast
                with autocast('cuda'):
                    logits_main = self.model(batch)[:, -1]
            else:
                logits_main = self.model(batch)[:, -1]
            probs_main = torch.sigmoid(logits_main)
            if self.use_tta:
                batch_flip = torch.flip(batch, dims=[3])  # flip width
                if self.device.type == 'cuda' and self.use_amp:
                    from torch.amp import autocast
                    with autocast('cuda'):
                        logits_flip = self.model(batch_flip)[:, -1]
                else:
                    logits_flip = self.model(batch_flip)[:, -1]
                probs_flip = torch.sigmoid(logits_flip)
                probs = (probs_main + probs_flip) * 0.5
            else:
                probs = probs_main
            probs = probs.detach().cpu().tolist()
        return [float(x) for x in probs]
    
    def explain(
        self,
        batch: torch.Tensor,
        method: str = "gradcam",
        class_idx: Optional[int] = None,
    ):  # -> Tuple[torch.Tensor, torch.Tensor]
        """
        Compute explainability maps for a batch of already-normalized images.

        Args:
            batch: (B, 3, H, W) tensor normalized with ImageNet stats.
            method: 'gradcam', 'gradsam' or 'smoothgrad' or 'integrated_gradients'.
            class_idx: target logit index; if None, use last logit (fake).

        Returns:
            cam: (B, 1, H, W) maps in [0, 1].
            logits: (B, K) raw logits (no sigmoid).
        """
        assert self.model is not None, "Model not loaded"
        x = batch.detach().clone().to(self.device)
        m = method.lower()
        
        if m == "lime":
            cam = lime_explain(
                model=self.model,
                image=x,
                device=self.device,
                class_idx=class_idx,
                positive_only=False,
            )
            return cam
        if m == "gradcam":
            cam, logits_all = gradcam_map(self.model, x, target_layer=None, class_idx=class_idx, device=self.device)
            return cam
        if m == "gradsam":
            cam, logits_all = gradsam_map(self.model, x, target_layer=None, class_idx=class_idx, device=self.device)
            return cam
        if m == "smoothgrad":
            self.model.eval()
            cam = self._smoothgrad_input(x, class_idx=class_idx, n_samples=8, sigma=0.1)
            return cam.detach()
        if m == "ig" or m == "integrated_gradients":
            self.model.eval()
            cam = self._integrated_gradients_input(x, class_idx=class_idx, steps=16)
            return cam.detach()
        if m in ("wps", "wavelet_perturbation"):
            cam, logits_all = self._wps_input(x, class_idx=class_idx, n_cells=8, wavelet="haar")
            return cam.detach()
        if m in ("gradcam_backbone", "gradcam_spatial"):
            cam, logits_all = self._gradcam_spatial_backbone(x, class_idx=class_idx)
            return cam.detach()
        if m == "attention_rollout":
            cam, logits_all = self._attention_rollout(x, class_idx=class_idx)
            return cam.detach()
        if m == "grad":
            cam, logits_all = grad_input_explain(
                self.model, x,
                class_idx=class_idx,
                device=self.device,
                n_samples=1,  # or 1 for vanilla Gradient × Input
                sigma=0.1,
                aggregate_channels=True
            )
            return cam
        raise ValueError(f"Unknown explainability method '{method}'")
    
    def _wps_input(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
        n_cells: int = 8,
        wavelet: str = "haar",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Wavelet-Perturbation Sensitivity (WPS) on the input frames.

        For each image and for each coarse cell in the wavelet high-frequency bands,
        we zero those coefficients, run the model, and measure the logit drop.
        The resulting grid of drops is upsampled to (H, W) to form a saliency map.

        Args:
            x: (B, 3, H, W) tensor on self.device.
            class_idx: which logit to explain; if None, use last logit (fake).
            n_cells: number of cells along each side in wavelet domain.
            wavelet: wavelet name for pywt.dwt2 / idwt2.

        Returns:
            cam_batch: (B, 1, H, W) saliency maps in [0, 1].
            logits_all: (B, K) baseline logits for original x.
        """
        assert self.model is not None, "Model not loaded"
        self.model.eval()
        
        with torch.no_grad():
            logits_all = self.model(x)  # (B, K)
        
        if class_idx is None:
            target_idx = logits_all.shape[-1] - 1
        else:
            target_idx = int(class_idx)
        
        base_scores = logits_all[:, target_idx]  # (B,)
        B, C, H, W = x.shape
        cams: list[torch.Tensor] = []
        
        for b in range(B):
            base = float(base_scores[b].item())
            # (C, H, W) -> numpy on CPU
            x_b = x[b].detach().cpu().numpy().astype(np.float32)
            
            # Use first channel to define wavelet grid geometry
            cA0, (cH0, cV0, cD0) = pywt.dwt2(x_b[0], wavelet)
            hH, wH = cH0.shape
            cell_h = max(hH // n_cells, 1)
            cell_w = max(wH // n_cells, 1)
            
            heat_grid = np.zeros((n_cells, n_cells), dtype=np.float32)
            
            for gy in range(n_cells):
                for gx in range(n_cells):
                    y0 = gy * cell_h
                    x0 = gx * cell_w
                    y1 = hH if gy == n_cells - 1 else min((gy + 1) * cell_h, hH)
                    x1 = wH if gx == n_cells - 1 else min((gx + 1) * cell_w, wH)
                    if y0 >= y1 or x0 >= x1:
                        continue
                    
                    # Build perturbed image in numpy
                    pert_img = np.empty_like(x_b)
                    for c in range(C):
                        cA, (cH, cV, cD) = pywt.dwt2(x_b[c], wavelet)
                        # zero high-frequency coeffs in this cell
                        cH_pert = cH.copy()
                        cV_pert = cV.copy()
                        cD_pert = cD.copy()
                        cH_pert[y0:y1, x0:x1] = 0.0
                        cV_pert[y0:y1, x0:x1] = 0.0
                        cD_pert[y0:y1, x0:x1] = 0.0
                        rec = pywt.idwt2((cA, (cH_pert, cV_pert, cD_pert)), wavelet)
                        rec = rec[:H, :W]
                        pert_img[c] = rec
                    
                    pert_tensor = torch.from_numpy(pert_img).to(
                        device=self.device, dtype=x.dtype,
                    ).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits_pert = self.model(pert_tensor)
                        score_pert = float(logits_pert[0, target_idx].item())
                    
                    drop = max(base - score_pert, 0.0)
                    heat_grid[gy, gx] = drop
            
            # Convert grid to full-resolution heatmap
            heat = torch.from_numpy(heat_grid).view(1, 1, n_cells, n_cells).to(self.device)
            heat = F.interpolate(heat, size=(H, W), mode="bilinear", align_corners=False)[0]  # (1,H,W)
            
            maxv = torch.max(heat)
            if maxv > 0:
                heat = heat / maxv
            
            cams.append(heat)
        
        cam_batch = torch.stack(cams, dim=0)  # (B, 1, H, W)
        return cam_batch, logits_all
    
    def _smoothgrad_input(self, x: torch.Tensor, class_idx: Optional[int] = None, n_samples: int = 8, sigma: float = 0.1) -> torch.Tensor:
        assert self.model is not None, "Model not loaded"
        B, C, H, W = x.shape
        acc = torch.zeros((B, 1, H, W), device=x.device, dtype=x.dtype)
        base = x.detach()
        for _ in range(n_samples):
            noise = torch.randn_like(base) * sigma
            xn = (base + noise).clone().requires_grad_(True)
            self.model.zero_grad()
            logits_all = self.model(xn)
            if class_idx is None:
                logits = logits_all[:, -1]
            else:
                logits = logits_all[:, class_idx]
            logits.sum().backward()
            g = xn.grad.abs().mean(dim=1, keepdim=True)
            acc += g
        acc = acc / float(n_samples)
        cams = []
        for i in range(B):
            c = acc[i:i + 1]
            c = c - c.min()
            m = c.max()
            if m > 0:
                c = c / m
            cams.append(c)
        return torch.cat(cams, dim=0)
    
    def _integrated_gradients_input(self, x: torch.Tensor, class_idx: Optional[int] = None, steps: int = 32) -> torch.Tensor:
        assert self.model is not None, "Model not loaded"
        B, C, H, W = x.shape
        baseline = torch.zeros_like(x)
        total_grad = torch.zeros_like(x)
        for k in range(1, steps + 1):
            alpha = float(k) / steps
            xi = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
            self.model.zero_grad()
            logits_all = self.model(xi)
            if class_idx is None:
                logits = logits_all[:, -1]
            else:
                logits = logits_all[:, class_idx]
            logits.sum().backward()
            if xi.grad is not None:
                total_grad += xi.grad
        avg_grad = total_grad / float(steps)
        attr = (x - baseline) * avg_grad
        raw = attr.abs().mean(dim=1, keepdim=True)
        cams = []
        for i in range(B):
            c = raw[i:i + 1]
            c = c - c.min()
            m = c.max()
            if m > 0:
                c = c / m
            cams.append(c)
        return torch.cat(cams, dim=0)
    
    def _find_spatial_backbone_layer(self, x: torch.Tensor):  # old def stays but body changes
        """
        Find the most 'spatial' 4D feature map (B,C,H,W) in the model:
        pick the layer with the largest H*W.
        """
        assert self.model is not None, "Model not loaded"
        
        activations: dict[str, torch.Size] = {}
        modules: dict[str, torch.nn.Module] = {}
        hooks = []
        
        def make_hook(name: str):
            def _hook(_m, _inp, out):
                if isinstance(out, torch.Tensor) and out.dim() == 4:
                    activations[name] = out.shape
            
            return _hook
        
        for name, m in self.model.named_modules():
            modules[name] = m
            h = m.register_forward_hook(make_hook(name))
            hooks.append(h)
        
        with torch.no_grad():
            _ = self.model(x[:1])
        
        for h in hooks:
            h.remove()
        
        if not activations:
            raise RuntimeError("No 4D feature map found for Grad-CAM spatial backbone")
        
        # --- pick layer with largest spatial size ---------------------------
        best_name = None
        best_score = -1
        for name, shape in activations.items():
            if len(shape) != 4:
                continue
            _, _, h, w = shape
            score = h * w
            if score >= best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise RuntimeError("No suitable spatial 4D feature map found")
        
        best_module = modules[best_name]
        # Optional: print once to see what we picked
        # print(f"[WaveRepDetector] Grad-CAM spatial layer: {best_name}, shape={activations[best_name]}")
        
        return best_module
    
    def _get_spatial_backbone_layer(self, x: torch.Tensor) -> nn.Module:
        """
        Get (and cache) the spatial backbone layer for Grad-CAM.
        """
        if self._gradcam_target_layer is None:
            self._gradcam_target_layer = self._find_spatial_backbone_layer(x)
        return self._gradcam_target_layer
    
    def _gradcam_spatial_backbone(
        self,
        x: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Grad-CAM on the spatial backbone feature map.

        Args:
            x: (B,3,H,W) normalized input tensor on self.device.
            class_idx: target logit index (None -> last logit / 'fake').

        Returns:
            cam_batch: (B,1,H,W) Grad-CAM maps in [0,1].
            logits_all: (B,K) logits for the original batch.
        """
        assert self.model is not None, "Model not loaded"
        self.model.eval()
        
        B, _, H, W = x.shape
        target_layer = self._get_spatial_backbone_layer(x)
        
        feats: list[torch.Tensor] = []
        grads: list[torch.Tensor] = []
        
        def fwd_hook(_m, _inp, out):
            feats.clear()
            feats.append(out)
        
        def bwd_hook(_m, grad_in, grad_out):
            grads.clear()
            grads.append(grad_out[0])
        
        h_fwd = target_layer.register_forward_hook(fwd_hook)
        h_bwd = target_layer.register_full_backward_hook(bwd_hook)
        
        x = x.requires_grad_(True)
        
        logits_all = self.model(x)  # (B,K)
        if class_idx is None:
            target_idx = logits_all.shape[-1] - 1
        else:
            target_idx = int(class_idx)
        
        scores = logits_all[:, target_idx].sum()
        self.model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        scores.backward(retain_graph=False)
        
        h_fwd.remove()
        h_bwd.remove()
        
        if not feats or not grads:
            raise RuntimeError("Grad-CAM hooks did not capture features or gradients")
        
        feat = feats[0]  # (B,C,h,w)
        grad = grads[0]  # (B,C,h,w)
        B_f, C, h, w = feat.shape
        assert B_f == B, "Mismatch between batch size of features and input"
        
        # GAP on gradients → weights per channel
        alpha = grad.contiguous().view(B, C, -1).mean(dim=2)  # (B,C)
        cam_list = []
        
        for b in range(B):
            w_ch = alpha[b].view(C, 1, 1)  # (C,1,1)
            cam = torch.relu((w_ch * feat[b]).sum(dim=0))  # (h,w)
            if cam.max() > 0:
                cam = cam / cam.max()
            cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,h,w)
            cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
            cam_list.append(cam_up[0])  # (1,H,W)
        
        cam_batch = torch.stack(cam_list, dim=0)  # (B,1,H,W)
        
        # Normalize per-batch to [0,1] (optional but handy)
        cam_min = cam_batch.view(B, -1).min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = cam_batch.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        denom = (cam_max - cam_min).clamp(min=1e-6)
        cam_batch = (cam_batch - cam_min) / denom
        
        return cam_batch, logits_all
