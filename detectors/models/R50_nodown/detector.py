"""
R50_nodown Detector with GradCAM-based vulnerability visualization.

This detector uses ResNet50 without downsampling and generates saliency maps
using GradCAM for vulnerability analysis.
"""

import os
import sys
from typing import Optional, Dict, Any, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from support.base_detector import BaseDetector
from support.detect_utils import load_image

DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))
if DETECTOR_DIR in sys.path:
    sys.path.remove(DETECTOR_DIR)
sys.path.insert(0, DETECTOR_DIR)
from networks.utils import create_architecture

# Default image size for R50_nodown
DEFAULT_IMAGE_SIZE = 512


class R50NoDownDetector(BaseDetector):
    """
    R50_nodown Detector implementing the BaseDetector interface with GradCAM support.
    
    This detector uses ResNet50 without downsampling and provides GradCAM-based
    saliency maps for vulnerability analysis.
    """
    
    name = 'R50_nodown'
    
    # Feature flags - this detector supports all features
    supports_explainability = True
    supports_vulnerability = True
    supports_adversarial = True
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.cam = None
        self.image_size = DEFAULT_IMAGE_SIZE
    
    def load(self, model_id: Optional[str] = None) -> None:
        """Load model weights and initialize GradCAM."""
        device = self.device
        weights = model_id or os.path.join(DETECTOR_DIR, 'weights', 'best.pt')
        if not os.path.exists(weights):
            raise FileNotFoundError(f"R50_nodown weights not found: {weights}")
        checkpoint = torch.load(weights, map_location=device)
        model = create_architecture("res50nodown", pretrained=True, num_classes=1).to(device)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        self.model = model
        
        # Initialize GradCAM with the last convolutional layer
        try:
            from pytorch_grad_cam import GradCAM
            target_layer = self.model.layer4[-1].conv3
            self.cam = GradCAM(model=self.model, target_layers=[target_layer])
        except ImportError:
            print("[warn] pytorch_grad_cam not installed. GradCAM features disabled.")
            self.cam = None
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        """Predict whether image is fake. Returns confidence [0,1] where higher = more fake."""
        out = self.model(image_tensor)
        return float(torch.sigmoid(out).item())
    
    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================
    
    def _compute_explanation_map(
        self,
        image: Union[str, Image.Image],
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        """
        Compute GradCAM explanation map for an image.
        
        Args:
            image: Input image as path string or PIL Image
            map_size: Size of the returned map (H, W). Default: (512, 512)
            **kwargs: Additional arguments (ignored)
            
        Returns:
            tuple: (confidence, gradcam_map)
                - confidence: float in [0, 1], higher = more likely fake
                - gradcam_map: np.ndarray of shape (H, W) normalized to [0, 1]
        """
        if map_size is None:
            map_size = (self.image_size, self.image_size)
            
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.cam is None:
            raise RuntimeError("GradCAM not initialized. Install pytorch_grad_cam.")
        
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        # Get image path
        image_path = self._ensure_image_path(image)
        
        # Load and preprocess image
        img_tensor, rgb_img = load_image(image_path, size=self.image_size)
        img_tensor = img_tensor.to(self.device)
        
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            confidence = float(torch.sigmoid(output).item())
        
        # Target index: 0 for single output model
        target_idx = 0 if output.ndim == 1 or output.shape[-1] == 1 else 1
        
        # Generate GradCAM map
        print(f"generating gradcam for image: {image_path}")
        grayscale_cam = self.cam(input_tensor=img_tensor, targets=[ClassifierOutputTarget(target_idx)])
        cam_map = grayscale_cam[0, :]  # Shape: (H, W)
        
        # Resize if needed
        if cam_map.shape != map_size:
            from PIL import Image as PILImage
            cam_pil = PILImage.fromarray((cam_map * 255).astype(np.uint8), mode='L')
            cam_pil = cam_pil.resize((map_size[1], map_size[0]), PILImage.BILINEAR)
            cam_map = np.array(cam_pil).astype(np.float32) / 255.0
        
        return confidence, cam_map
    
    def _compute_vulnerability_map(
        self, 
        image: Union[str, Image.Image],
        attack_type: str = "fgsm",
        epsilon: float = 0.03,
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute vulnerability map by comparing GradCAM before/after attack.
        
        Args:
            image: Input image (PIL Image or path string)
            attack_type: Type of adversarial attack
            epsilon: Attack strength
            map_size: Size of output maps. Default: (512, 512)
            **kwargs: Additional arguments (e.g., true_label)
        
        Returns:
            dict with keys:
                - 'prediction': float, original prediction score
                - 'prediction_attacked': float, prediction after attack
                - 'explanation_map': np.ndarray, original GradCAM map
                - 'explanation_map_attacked': np.ndarray, GradCAM after attack
                - 'vulnerability_map': np.ndarray, |original - attacked|
                - 'input_tensor': torch.Tensor, preprocessed input
                - 'anomaly_map': np.ndarray (legacy alias for explanation_map)
        """
        if map_size is None:
            map_size = (self.image_size, self.image_size)
            
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Get image path
        image_path = self._ensure_image_path(image)
        
        # Get original prediction and explanation map
        confidence, cam_map = self._compute_explanation_map(image_path, map_size=map_size)
        
        # Generate adversarial image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            adv_path = f.name
        
        true_label = kwargs.get('true_label', 1)
        self._generate_adversarial_image(
            image_path=image_path,
            output_path=adv_path,
            attack_type=attack_type,
            epsilon=epsilon,
            true_label=true_label
        )
        
        # Get attacked prediction and explanation map
        confidence_attacked, cam_map_attacked = self._compute_explanation_map(adv_path, map_size=map_size)
        
        # Compute vulnerability map
        vulnerability_map = np.abs(cam_map - cam_map_attacked)
        
        # Load tensor for return
        img_tensor, _ = load_image(image_path, size=self.image_size)
        img_tensor = img_tensor.to(self.device)
        
        # Clean up temp file
        try:
            os.unlink(adv_path)
        except:
            pass
        
        return {
            'prediction': confidence,
            'prediction_attacked': confidence_attacked,
            'explanation_map': cam_map,
            'explanation_map_attacked': cam_map_attacked,
            'vulnerability_map': vulnerability_map,
            'input_tensor': img_tensor.unsqueeze(0),
            # Legacy alias
            'anomaly_map': cam_map,
        }
    
    def _generate_adversarial_image(
        self,
        image_path: str,
        output_path: str,
        attack_type: str = "pgd",
        epsilon: float = 0.03,
        true_label: int = 1,
        **kwargs
    ) -> str:
        """
        Generate adversarial image using the specified attack type.
        
        Args:
            image_path: Path to the original image
            output_path: Path to save the adversarial image
            attack_type: Type of attack ('pgd', 'fgsm', 'deepfool')
            epsilon: Attack strength
            true_label: True label (0=real, 1=fake). Attack targets opposite class.
            **kwargs: Additional attack parameters
        
        Returns:
            Path to the saved adversarial image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        try:
            import torchattacks as ta
        except ImportError:
            raise RuntimeError("torchattacks not installed. Run: pip install torchattacks")
        
        # Load original image to get size
        original_img = Image.open(image_path).convert('RGB')
        original_size = original_img.size  # (W, H)
        
        # Load and preprocess image
        img_tensor, rgb_img = load_image(image_path, size=self.image_size)
        img_tensor = img_tensor.to(self.device)
        
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        # Create a wrapper model for torchattacks that outputs 2-class logits
        wrapper = _R50NoDownAttackWrapper(self.model, self.device)
        
        # Create attack
        attack_type_lower = attack_type.lower()
        if attack_type_lower == "fgsm":
            attack = ta.FGSM(wrapper, eps=epsilon)
        elif attack_type_lower == "pgd":
            attack = ta.PGD(wrapper, eps=epsilon, alpha=epsilon/4, steps=10)
        elif attack_type_lower == "deepfool":
            attack = ta.DeepFool(wrapper, steps=50, overshoot=0.02)
        else:
            raise ValueError(f"Unsupported attack type: {attack_type}. Supported: fgsm, pgd, deepfool")
        
        # Target the OPPOSITE class to flip the prediction
        # true_label=1 (fake) -> target=0 (real), true_label=0 (real) -> target=1 (fake)
        target = torch.tensor([1 - true_label], device=self.device)
        
        # Generate adversarial example
        wrapper.train()  # Some attacks require train mode
        x_adv = attack(img_tensor, target)
        wrapper.eval()
        
        # Convert back to image and save
        x_adv_np = x_adv.squeeze(0).cpu().numpy()
        # Denormalize: assuming input was normalized to [0, 1]
        x_adv_np = np.clip(x_adv_np, 0, 1)
        x_adv_np = (x_adv_np * 255).astype(np.uint8)
        x_adv_np = np.transpose(x_adv_np, (1, 2, 0))  # CHW -> HWC
        
        # Create and resize back to original size
        adv_img = Image.fromarray(x_adv_np)
        if adv_img.size != original_size:
            adv_img = adv_img.resize(original_size, Image.BILINEAR)
        
        # Save image
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        adv_img.save(output_path)
        print(f"[R50_nodown] Generated {attack_type} adversarial image (label={true_label}): {output_path}")
        
        return output_path
    
    # =========================================================================
    # PUBLIC API - Wrappers around abstract methods (for backward compatibility)
    # =========================================================================
    
    def predict_with_map(self, image_path: str, map_size: tuple = (512, 512)) -> tuple:
        """
        Predict whether an image is fake and return the GradCAM saliency map.
        
        This is the public API that wraps _compute_explanation_map.
        
        Args:
            image_path: Path to the image
            map_size: Size of the returned map (H, W)
            
        Returns:
            tuple: (confidence, gradcam_map)
        """
        return self._compute_explanation_map(image_path, map_size=map_size)
    
    def predict_with_vulnerability(
        self, 
        image: Union[str, Image.Image],
        device: str = "cpu",
        map_size: tuple = (512, 512),
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict and compute vulnerability information using GradCAM.
        
        This is the public API that wraps _compute_vulnerability_map.
        For backward compatibility, also accepts 'device' parameter (ignored).
        
        Args:
            image: Input image (PIL Image or path string)
            device: Device for computation (ignored, uses self.device)
            map_size: Size of output maps
            **kwargs: Additional arguments
        
        Returns:
            dict with prediction and map information
        """
        return self._compute_vulnerability_map(image, map_size=map_size, **kwargs)
    
    def generate_adversarial(
        self,
        image_path: str,
        output_path: str,
        attack_type: str = "pgd",
        epsilon: float = 0.03,
        true_label: int = 1,
        **kwargs
    ) -> str:
        """
        Generate adversarial image using the specified attack type.
        
        This is the public API that wraps _generate_adversarial_image.
        
        Args:
            image_path: Path to the original image
            output_path: Path to save the adversarial image
            attack_type: Type of attack ('pgd', 'fgsm', 'deepfool')
            epsilon: Attack strength
            true_label: True label (0=real, 1=fake). Attack targets opposite class.
            **kwargs: Additional attack parameters
        
        Returns:
            Path to the saved adversarial image
        """
        return self._generate_adversarial_image(
            image_path=image_path,
            output_path=output_path,
            attack_type=attack_type,
            epsilon=epsilon,
            true_label=true_label,
            **kwargs
        )
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    def visualize_vulnerability_grid(
        self,
        image_set,
        output_path: str,
        attack_type: str = "pgd",
        dpi: int = 150,
        overlay_alpha: float = 0.4,
        epsilon: float = 0.03,
        map_cache: Optional[Dict[str, Tuple[float, np.ndarray]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize vulnerability analysis in a grid format using GradCAM.
        
        If adversarial images don't exist, they will be generated on-the-fly
        and saved to the expected paths in the image_set.
        
        Creates a 5x3 grid:
            - Columns: Real (C1), Samecat (C2), Diffcat (C3)
            - Rows: R1: Image, R2: Image + GradCAM, R3: Attacked + GradCAM, 
                    R4: Image + Vulnerability Map, R5: Mask/BBox
        
        Args:
            image_set: ImageSet dataclass containing all image paths
            output_path: Path to save the visualization
            attack_type: Name of the attack type (for labeling and generation)
            dpi: DPI for saved image
            overlay_alpha: Alpha for map overlays
            epsilon: Attack strength for adversarial generation
            **kwargs: Additional arguments
        
        Returns:
            dict with 'generated_adversarial': True if any adversarial images were generated
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self.cam is None:
            raise RuntimeError("GradCAM not initialized. Install pytorch_grad_cam.")
        
        generated_any = False
        
        # Get or generate adversarial images
        # real images have true_label=0, fake images (samecat/diffcat) have true_label=1
        real_adv_path, gen1 = self._get_or_generate_adversarial(
            image_set.real, image_set.real_adv, 
            getattr(image_set, 'real_adv_expected', None),
            attack_type, epsilon,
            true_label=0,  # Real image
        )
        samecat_adv_path, gen2 = self._get_or_generate_adversarial(
            image_set.samecat, image_set.samecat_adv,
            getattr(image_set, 'samecat_adv_expected', None),
            attack_type, epsilon,
            true_label=1,  # Fake image (inpainted)
        )
        diffcat_adv_path, gen3 = self._get_or_generate_adversarial(
            image_set.diffcat, image_set.diffcat_adv,
            getattr(image_set, 'diffcat_adv_expected', None),
            attack_type, epsilon,
            true_label=1,  # Fake image (inpainted)
        )
        generated_any = gen1 or gen2 or gen3
        
        # Load images
        img_real = Image.open(image_set.real).convert('RGB')
        img_samecat = Image.open(image_set.samecat).convert('RGB')
        img_diffcat = Image.open(image_set.diffcat).convert('RGB')
        
        img_real_adv = Image.open(real_adv_path).convert('RGB')
        img_samecat_adv = Image.open(samecat_adv_path).convert('RGB')
        img_diffcat_adv = Image.open(diffcat_adv_path).convert('RGB')
        
        # Load mask and bbox
        mask_np = self._load_mask(image_set.mask)
        bbox_np = self._load_mask(image_set.bbox)
        black_mask_np = np.zeros_like(mask_np) if mask_np is not None else np.zeros((512, 512))
        if mask_np is None:
            mask_np = np.zeros((512, 512))
        if bbox_np is None:
            bbox_np = np.zeros((512, 512))
        
        # Helper function to get map from cache or compute
        def get_map_cached(image_path: str) -> Tuple[float, np.ndarray]:
            """Get map from cache or compute if not cached."""
            if map_cache is not None and image_path in map_cache:
                return map_cache[image_path]
            pred, cam = self.predict_with_map(image_path)
            if map_cache is not None:
                map_cache[image_path] = (pred, cam)
            return pred, cam
        
        # Get predictions and GradCAM maps for benign images (using cache if available)
        pred_real, cam_real = get_map_cached(image_set.real)
        pred_samecat, cam_samecat = get_map_cached(image_set.samecat)
        pred_diffcat, cam_diffcat = get_map_cached(image_set.diffcat)
        
        # Get predictions and GradCAM maps for adversarial images (using cache if available)
        pred_real_adv, cam_real_adv = get_map_cached(real_adv_path)
        pred_samecat_adv, cam_samecat_adv = get_map_cached(samecat_adv_path)
        pred_diffcat_adv, cam_diffcat_adv = get_map_cached(diffcat_adv_path)
        
        # Compute vulnerability maps (|original - attacked|)
        vuln_real = np.abs(cam_real - cam_real_adv)
        vuln_samecat = np.abs(cam_samecat - cam_samecat_adv)
        vuln_diffcat = np.abs(cam_diffcat - cam_diffcat_adv)
        
        # Create figure with 5 rows x 3 columns
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        
        # Column and row titles
        col_titles = ['Real (C1)', 'Samecat (C2)', 'Diffcat (C3)']
        row_titles = [
            'R1: Image',
            'R2: Image + GradCAM',
            'R3: Attacked + GradCAM',
            'R4: Image + Vulnerability Map',
            'R5: Mask/BBox'
        ]
        
        # Data for each column
        columns_data = [
            {
                'image': img_real,
                'image_adv': img_real_adv,
                'mask': black_mask_np,
                'cam': cam_real,
                'cam_adv': cam_real_adv,
                'vuln': vuln_real,
                'pred': pred_real,
                'pred_adv': pred_real_adv,
            },
            {
                'image': img_samecat,
                'image_adv': img_samecat_adv,
                'mask': mask_np,
                'cam': cam_samecat,
                'cam_adv': cam_samecat_adv,
                'vuln': vuln_samecat,
                'pred': pred_samecat,
                'pred_adv': pred_samecat_adv,
            },
            {
                'image': img_diffcat,
                'image_adv': img_diffcat_adv,
                'mask': bbox_np,
                'cam': cam_diffcat,
                'cam_adv': cam_diffcat_adv,
                'vuln': vuln_diffcat,
                'pred': pred_diffcat,
                'pred_adv': pred_diffcat_adv,
            },
        ]
        
        for col_idx, col_data in enumerate(columns_data):
            img = col_data['image']
            img_adv = col_data['image_adv']
            mask = col_data['mask']
            cam_map = col_data['cam']
            cam_map_adv = col_data['cam_adv']
            vuln = col_data['vuln']
            pred = col_data['pred']
            pred_adv = col_data['pred_adv']
            
            # Resize maps to match image size if needed
            img_size = (img.size[1], img.size[0])  # (H, W)
            cam_resized = self._resize_map(cam_map, img_size)
            cam_adv_resized = self._resize_map(cam_map_adv, img_size)
            vuln_resized = self._resize_map(vuln, img_size)
            mask_resized = self._resize_map(mask, img_size) if mask is not None else np.zeros(img_size)
            
            # R1: Original image with prediction
            axes[0, col_idx].imshow(img)
            pred_label = 'FAKE' if pred > 0.5 else 'REAL'
            axes[0, col_idx].set_title(f'{col_titles[col_idx]}\nPred: {pred_label} ({pred:.3f})', fontsize=10)
            axes[0, col_idx].axis('off')
            
            # R2: Original image with GradCAM overlay
            axes[1, col_idx].imshow(img)
            axes[1, col_idx].imshow(cam_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
            axes[1, col_idx].axis('off')
            
            # R3: Attacked image with GradCAM overlay
            pred_adv_label = 'FAKE' if pred_adv > 0.5 else 'REAL'
            axes[2, col_idx].imshow(img_adv)
            axes[2, col_idx].imshow(cam_adv_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
            axes[2, col_idx].set_title(f'{attack_type.upper()}\nPred: {pred_adv_label} ({pred_adv:.3f})', fontsize=9)
            axes[2, col_idx].axis('off')
            
            # R4: Original image with vulnerability map overlay
            axes[3, col_idx].imshow(img)
            axes[3, col_idx].imshow(vuln_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
            axes[3, col_idx].axis('off')
            
            # R5: Mask/BBox
            axes[4, col_idx].imshow(mask_resized, cmap='gray', vmin=0, vmax=1)
            axes[4, col_idx].axis('off')
        
        # Add row labels on the left
        for row_idx, row_title in enumerate(row_titles):
            axes[row_idx, 0].set_ylabel(row_title, fontsize=10, rotation=0, ha='right', va='center')
        
        plt.suptitle(f'R50_nodown Vulnerability Analysis - {image_set.filename} ({attack_type})', fontsize=12, y=1.01)
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        return {'generated_adversarial': generated_any}
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    @staticmethod
    def _resize_map(map_np: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize a map to target size (H, W)."""
        from PIL import Image as PILImage
        if map_np.shape == target_size:
            return map_np
        # Convert to PIL, resize, convert back
        map_img = PILImage.fromarray((map_np * 255).astype(np.uint8), mode='L')
        map_img = map_img.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
        return np.array(map_img).astype(np.float32) / 255.0


class _R50NoDownAttackWrapper(torch.nn.Module):
    """
    Wrapper to make R50_nodown compatible with torchattacks library.
    
    torchattacks expects a model that:
    - Is a nn.Module
    - Takes image tensor [B, C, H, W] and returns logits [B, num_classes]
    
    R50_nodown outputs a single value (sigmoid), so we convert it to 2-class logits.
    """
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        
    def forward(self, x):
        """
        Forward pass for torchattacks compatibility.
        
        Args:
            x: Image tensor of shape [B, C, H, W]
            
        Returns:
            Logits tensor of shape [B, 2] (binary classification as 2-class)
        """
        # Get model output (raw logits before sigmoid)
        output = self.model(x)  # [B, 1] or [B]
        
        # Convert to [B, 1] if needed
        if output.ndim == 1:
            output = output.unsqueeze(1)
        
        # Apply sigmoid to get probability
        prob_fake = torch.sigmoid(output)  # [B, 1]
        prob_real = 1.0 - prob_fake  # [B, 1]
        
        # Convert probabilities to logits for 2-class output
        # Use logit transform: logit(p) = log(p / (1-p))
        prob_fake = torch.clamp(prob_fake, 1e-7, 1 - 1e-7)
        prob_real = torch.clamp(prob_real, 1e-7, 1 - 1e-7)
        
        logit_real = torch.log(prob_real / (1 - prob_real))
        logit_fake = torch.log(prob_fake / (1 - prob_fake))
        
        logits = torch.cat([logit_real, logit_fake], dim=1)  # [B, 2]
        
        return logits
