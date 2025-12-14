from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from lime import lime_image
from PIL import Image


def _make_batch_predict_fn(forward):
    def batch_predict(images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            logits = forward(images)  # make sure model returns raw logits
            logits = logits.detach().cpu()
        # SPECIALIZED: treat last logit as "fake" score and map to 2-class probs via sigmoid
        if logits.ndim == 1:
            last = logits
        elif logits.ndim == 2:
            last = logits[:, -1]
        else:
            last = logits.view(logits.size(0), -1)[:, -1]
        last_np = last.detach().cpu().numpy()
        zeros = np.zeros_like(last_np)
        scores = np.stack([zeros, last_np], axis=1)
        return scores
    
    return batch_predict


def lime_explain(
    logits_fn,
    images: np.ndarray,
    class_idx: Optional[int] = None,
    top_labels: Optional[int] = None,
    num_samples: int = 150,
    n_segments: int = 24,
    batch_size: int = 64,
    fixed_segments: Optional[Union[np.ndarray, Sequence[np.ndarray]]] = None,
) -> torch.Tensor:
    """
    Compute LIME masks for a batch tensor normalized with ImageNet stats.

    Returns:
        cam: (B,1,H,W) float tensor in [0,1]
        logits: (B,K) raw model logits for the original batch
    """
    # Prepare predictor and explainer
    batch_predict = _make_batch_predict_fn(logits_fn)
    explainer = lime_image.LimeImageExplainer()
    
    # Fast SLIC segmentation to reduce superpixels and speed up
    try:
        from skimage.segmentation import slic
        
        def default_segmentation_fn(x):
            return slic(x, n_segments=n_segments, compactness=20, start_label=0)
    except Exception:
        default_segmentation_fn = None  # fall back to default
    
    # Determine which class to visualize
    if class_idx is None:
        # For single-logit detectors we visualize the positive class (index 1)
        target_cls = 1
    else:
        target_cls = int(class_idx)
    
    single_image = False
    if isinstance(images, Image.Image):
        images = np.array([images])
        single_image = True
    
    if len(images.shape) == 3:
        images = np.expand_dims(images, axis=0)
        single_image = True
    
    cams: List[torch.Tensor] = []
    for i, img in enumerate(images):
        if fixed_segments is None:
            segmentation_fn = default_segmentation_fn
        else:
            seg_i = fixed_segments[i] if isinstance(fixed_segments, (list, tuple)) else fixed_segments
            if seg_i.shape != img.shape[:2]:
                raise ValueError(
                    f"fixed_segments shape {seg_i.shape} must match image spatial shape {img.shape[:2]}",
                )
            
            def segmentation_fn(_x, _seg=seg_i):
                return _seg
        
        explanation = explainer.explain_instance(
            image=img,
            classifier_fn=batch_predict,
            top_labels=top_labels,
            hide_color=0,
            num_samples=num_samples,
            segmentation_fn=segmentation_fn,
            random_seed=0,
            batch_size=batch_size,
        )
        
        # --- Build per-pixel heatmap from segment weights -------------
        segments = explanation.segments
        weights = dict(explanation.local_exp[target_cls])
        
        hm = np.zeros_like(segments, dtype=np.float32)
        for sp_id, w in weights.items():
            hm[segments == sp_id] = w
        
        m = torch.from_numpy(hm)[None, ...]
        cams.append(m)
    
    if single_image:
        return cams[0][0]
    
    cam_batch = torch.stack(cams, dim=0)  # (B,1,H,W)
    return cam_batch
