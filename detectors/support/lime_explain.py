from typing import List, Optional, Tuple

import numpy as np
import torch
from lime import lime_image
from PIL import Image
from skimage.segmentation import slic
from tqdm import tqdm


def _make_batch_predict_fn(forward):
    def batch_predict(images: List[Image.Image]) -> np.ndarray:
        with torch.no_grad():
            logits = forward(images)  # make sure model returns raw logits
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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute LIME masks for a batch tensor normalized with ImageNet stats.

    Returns:
        cam: (B,1,H,W) float tensor in [0,1]
        logits: (B,K) raw model logits for the original batch
    """
    # Prepare predictor and explainer
    batch_predict = _make_batch_predict_fn(logits_fn)
    explainer = lime_image.LimeImageExplainer()
    
    # Determine which class to visualize
    if class_idx is None:
        # For single-logit detectors we visualize the positive class (index 1)
        target_cls = 1
    else:
        target_cls = int(class_idx)
    
    cams: List[torch.Tensor] = []
    orig_img = None
    for i, img in tqdm(enumerate(images), leave=False):
        def segmentation_fn(_):  # FIXME !!! assumes images are in pairs (original, adversarial)
            nonlocal orig_img
            if i % 2 == 0:
                orig_img = img
            im = orig_img
            return slic(im, n_segments=n_segments, compactness=20, start_label=0)
        
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
    
    cam_batch = torch.stack(cams, dim=0)  # (B,1,H,W)
    return cam_batch
