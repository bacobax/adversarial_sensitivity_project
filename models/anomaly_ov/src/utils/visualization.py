import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from matplotlib import cm
import cv2

def visualize_anomaly_gradcam(
        image_tensor,
        anomaly_maps=None,
        vuln_maps=None,
        mask_image=None,
        anomaly_maps_adv=None,
        output_maps=None,
        metrics_label: str | None = None,
        alpha=0.5,
        figsize=(20, 12)):
    """
    Visualize original crops plus overlays:
      1) Original crops
      2) Anomaly Grad-CAM (if anomaly_maps)
      3) Adv Anomaly Grad-CAM (if anomaly_maps_adv)
      4) Output |Δ| Map (if output_maps)
      5) Vulnerability maps (if vuln_maps)
      6) Inpainted masks per crop (if mask_image)

    All per-view inputs expected as tensors/numpy in shapes similar to:
    - image_tensor: [1, V, 3, H, W]
    - anomaly_maps/anomaly_maps_adv/output_maps: [V, 1, h, w]
    - vuln_maps: [1, V, 1, h, w] or [V, 1, h, w]
    - mask_image: [1, V, C, Hm, Wm] or [V, C, Hm, Wm] or [V, Hm, Wm]
    """
    # Convert tensors to numpy
    if isinstance(image_tensor, torch.Tensor):
        image_tensor = image_tensor.detach().float().cpu().numpy()

    def to_numpy_opt(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().float().cpu().numpy()
        return x

    anomaly_maps = to_numpy_opt(anomaly_maps)
    anomaly_maps_adv = to_numpy_opt(anomaly_maps_adv)
    output_maps = to_numpy_opt(output_maps)
    vuln_maps = to_numpy_opt(vuln_maps)

    # Convert mask tensor to numpy and simplify shape to [V, H, W] (grayscale)
    mask_views = None
    if mask_image is not None:
        if isinstance(mask_image, torch.Tensor):
            m = mask_image.detach().float().cpu().numpy()  # [1, V, C, H, W]
            if m.ndim == 5 and m.shape[0] == 1:
                m = m[0]  # [V, C, H, W]
            if m.ndim == 4:
                if m.shape[1] == 3:
                    m_norm = (m - m.min()) / (m.max() - m.min() + 1e-8)
                    mask_views = 0.299 * m_norm[:,0] + 0.587 * m_norm[:,1] + 0.114 * m_norm[:,2]
                elif m.shape[1] == 1:
                    mask_views = m[:,0]
                else:
                    mask_views = m[:,0]
            else:
                mask_views = None
        else:
            try:
                m = np.array(mask_image)
                if m.ndim == 4:
                    if m.shape[-1] == 3:
                        mask_views = 0.299*m[...,0] + 0.587*m[...,1] + 0.114*m[...,2]
                    elif m.shape[-1] == 1:
                        mask_views = m[...,0]
                    else:
                        mask_views = m[...,0]
                elif m.ndim == 3:
                    mask_views = m
                else:
                    mask_views = None
            except Exception:
                mask_views = None

    num_views = image_tensor.shape[1]

    # Determine number of rows in desired order
    rows = 1  # Original images
    if anomaly_maps is not None:
        rows += 1
    if anomaly_maps_adv is not None:
        rows += 1
    if output_maps is not None:
        rows += 1
    if vuln_maps is not None:
        rows += 1
    if mask_views is not None:
        rows += 1

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(rows, num_views, figure=fig, hspace=0.25, wspace=0.15)

    # Draw metrics label ONCE at figure level (top-right) if provided
    if metrics_label:
        fig.text(0.98, 0.965, metrics_label,
                 transform=fig.transFigure,
                 fontsize=18, color='white', va='top', ha='right',
                 bbox=dict(facecolor='black', alpha=0.5, pad=3, edgecolor='none'))

    current_row = 0

    # Row 1: Original images
    for view_idx in range(num_views):
        img = image_tensor[0, view_idx]
        img = np.transpose(img, (1, 2, 0))
        img = img * 0.5 + 0.5
        img = np.clip(img, 0, 1)
        ax_orig = fig.add_subplot(gs[current_row, view_idx])
        ax_orig.imshow(img)
        title = f'View {view_idx}\n(Base/Global)' if view_idx == 0 else f'View {view_idx}\n(Crop {view_idx})'
        ax_orig.set_title(title, fontsize=10, fontweight='bold')
        ax_orig.axis('off')
    current_row += 1

    def overlay_heat(ax, base_img, heat, title):
        anom_resized = cv2.resize(heat, (base_img.shape[1], base_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        anom_resized = (anom_resized - anom_resized.min()) / (anom_resized.max() - anom_resized.min() + 1e-8)
        colormap_hot = cm.get_cmap('hot')
        anom_colored = colormap_hot(anom_resized)[:, :, :3]
        overlay = base_img * (1 - alpha) + anom_colored * alpha
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Row 2: Anomaly overlay
    if anomaly_maps is not None:
        for view_idx in range(num_views):
            img = image_tensor[0, view_idx]
            img = np.transpose(img, (1, 2, 0))
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ax_anom = fig.add_subplot(gs[current_row, view_idx])
            anom = anomaly_maps[view_idx, 0]
            overlay_heat(ax_anom, img, anom, 'Anomaly Grad-CAM')
        current_row += 1

    # Row 3: Adv Anomaly overlay
    if anomaly_maps_adv is not None:
        for view_idx in range(num_views):
            img = image_tensor[0, view_idx]
            img = np.transpose(img, (1, 2, 0))
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ax_adv = fig.add_subplot(gs[current_row, view_idx])
            anom_adv = anomaly_maps_adv[view_idx, 0]
            overlay_heat(ax_adv, img, anom_adv, 'Adv Anomaly')
        current_row += 1

    # Row 4: Output |Δ| overlay
    if output_maps is not None:
        for view_idx in range(num_views):
            img = image_tensor[0, view_idx]
            img = np.transpose(img, (1, 2, 0))
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ax_out = fig.add_subplot(gs[current_row, view_idx])
            outm = output_maps[view_idx, 0]
            overlay_heat(ax_out, img, outm, 'Output |Δ| Map')
        current_row += 1

    # Next row: Vulnerability maps
    if vuln_maps is not None:
        for view_idx in range(num_views):
            img = image_tensor[0, view_idx]
            img = np.transpose(img, (1, 2, 0))
            img = img * 0.5 + 0.5
            img = np.clip(img, 0, 1)
            ax_vuln = fig.add_subplot(gs[current_row, view_idx])
            if isinstance(vuln_maps, np.ndarray) and vuln_maps.ndim == 5:
                vuln = vuln_maps[0, view_idx, 0]
            elif isinstance(vuln_maps, np.ndarray) and vuln_maps.ndim == 4:
                vuln = vuln_maps[view_idx, 0]
            else:
                vuln = vuln_maps[view_idx]
            vuln_resized = cv2.resize(vuln, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
            vuln_resized = (vuln_resized - vuln_resized.min()) / (vuln_resized.max() - vuln_resized.min() + 1e-8)
            colormap_jet = cm.get_cmap('jet')
            vuln_colored = colormap_jet(vuln_resized)[:, :, :3]
            vuln_overlay = img * (1 - alpha) + vuln_colored * alpha
            vuln_overlay = np.clip(vuln_overlay, 0, 1)
            ax_vuln.imshow(vuln_overlay)
            ax_vuln.set_title('Vulnerability Map', fontsize=10)
            ax_vuln.axis('off')
        current_row += 1

    # Final row: Masks per view
    if mask_views is not None:
        for view_idx in range(num_views):
            ax_mask = fig.add_subplot(gs[current_row, view_idx])
            mv = mask_views[view_idx]
            mv_norm = (mv - mv.min()) / (mv.max() - mv.min() + 1e-8)
            ax_mask.imshow(mv_norm, cmap='gray')
            ax_mask.set_title('Mask', fontsize=10, fontweight='bold')
            ax_mask.axis('off')

    # Row labels (left-side text). Predefined positions for up to 6 rows.
    y_positions = []
    if rows == 2:
        y_positions = [0.75, 0.25]
    elif rows == 3:
        y_positions = [0.83, 0.52, 0.20]
    elif rows == 4:
        y_positions = [0.87, 0.62, 0.37, 0.12]
    elif rows == 5:
        y_positions = [0.90, 0.72, 0.54, 0.36, 0.18]
    elif rows >= 6:
        y_positions = [0.92, 0.77, 0.62, 0.47, 0.32, 0.17]

    labels = ['Original\nCrops']
    if anomaly_maps is not None:
        labels.append('Anomaly\nGrad-CAM')
    if anomaly_maps_adv is not None:
        labels.append('Adv\nAnomaly')
    if output_maps is not None:
        labels.append('Output\n|Δ| Map')
    if vuln_maps is not None:
        labels.append('Vulnerability\nMap')
    if mask_views is not None:
        labels.append('Mask')

    for i, lab in enumerate(labels):
        if i < len(y_positions):
            fig.text(0.02, y_positions[i], lab, fontsize=12, fontweight='bold', va='center', ha='center')

    plt.suptitle(f'Anomaly Detection Visualization (Alpha: {alpha})', fontsize=14, fontweight='bold', y=0.99)
    return fig