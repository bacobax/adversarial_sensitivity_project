"""
Anomaly-OV Detector wrapper for the unified detection framework.

This detector uses the Anomaly-OV model for image anomaly detection, 
implementing the BaseDetector interface for compatibility with detect.py.
"""

import os
import sys
from typing import Optional, Dict, Any

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Set up path for internal imports within anomaly_ov
DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure anomaly_ov directory is in path for internal llava/src imports
if DETECTOR_DIR not in sys.path:
    sys.path.insert(0, DETECTOR_DIR)

# Import BaseDetector from the support module (parent directory)
from support.base_detector import BaseDetector

# Internal imports from the anomaly_ov module
from llava.model.anomaly_expert import AnomalyOV
from llava.model.multimodal_encoder.siglip_encoder import SigLipVisionTower


class AnomalyOVDetector(BaseDetector):
    """
    Anomaly-OV Detector implementing the BaseDetector interface.
    
    This detector uses a SigLip vision encoder combined with an 
    anomaly expert module for zero-shot anomaly detection.
    """
    
    name = 'AnomalyOV'
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.vision_tower = None
        self.anomaly_expert = None
        self.image_processor = None
        self._dtype = torch.float32
        
        # Handle MPS/CUDA/CPU device-specific dtype
        if self.device.type == 'cuda':
            self._dtype = torch.float16
        elif self.device.type == 'mps':
            self._dtype = torch.float32  # MPS has limited dtype support
        else:
            self._dtype = torch.float32
    
    def load(self, model_id: Optional[str] = None) -> None:
        """
        Load the Anomaly-OV model weights.
        
        Args:
            model_id: Path to a checkpoint file (.pt) or to the anomaly expert weights.
                     If None, uses the default weights in the weights folder.
                     Can be:
                       - A full checkpoint from save_checkpoint() method
                       - An anomaly expert weights file (pretrained_expert_*.pth)
                       - A directory containing 'best.pt' or 'zs_checkpoint.pt'
        """
        device = self.device
        
        # Determine weights path
        if model_id is None:
            # Try default checkpoint locations in order of preference
            default_paths = [
                os.path.join(DETECTOR_DIR, 'weights', 'best.pt'),
                os.path.join(DETECTOR_DIR, 'weights', 'zs_checkpoint.pt'),
            ]
            weights = None
            for path in default_paths:
                if os.path.exists(path):
                    weights = path
                    break
            if weights is None:
                raise FileNotFoundError(
                    f"AnomalyOV weights not found. Tried: {default_paths}. "
                    f"Please place weights in {os.path.join(DETECTOR_DIR, 'weights')}/"
                )
        elif os.path.isdir(model_id):
            # If directory provided, look for standard checkpoint names
            for name in ['best.pt', 'zs_checkpoint.pt']:
                path = os.path.join(model_id, name)
                if os.path.exists(path):
                    weights = path
                    break
            else:
                raise FileNotFoundError(f"No checkpoint found in directory: {model_id}")
        else:
            weights = model_id
            if not os.path.exists(weights):
                raise FileNotFoundError(f"AnomalyOV weights not found: {weights}")
        
        print(f"Loading AnomalyOV from: {weights}")
        
        # Check if this is a full checkpoint or just anomaly expert weights
        checkpoint = torch.load(weights, map_location='cpu')
        
        if isinstance(checkpoint, dict) and 'vision_encoder_state_dict' in checkpoint:
            # Full checkpoint from OVAnomalyDetector.save_checkpoint()
            self._load_from_full_checkpoint(checkpoint, device)
        else:
            # Assume it's anomaly expert weights or a state dict
            self._load_from_expert_weights(weights, device)
        
        print(f"AnomalyOV loaded successfully on {device}")
    
    def _load_from_full_checkpoint(self, checkpoint: dict, device: torch.device) -> None:
        """Load from a full checkpoint containing both vision encoder and anomaly expert."""
        # Parse dtype
        dtype_str = checkpoint.get('dtype', 'torch.float32')
        if isinstance(dtype_str, str):
            self._dtype = getattr(torch, dtype_str.replace('torch.', ''), torch.float32)
        
        # Fallback on CPU/MPS if unsupported dtype
        if device.type in ('cpu', 'mps') and self._dtype in (torch.float16, torch.bfloat16):
            self._dtype = torch.float32
        
        # Build vision tower
        vision_tower_name = checkpoint.get('vision_tower_name', 'google/siglip-so400m-patch14-384')
        print(f"Loading vision tower: {vision_tower_name}")
        
        self.vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
        self.vision_tower.load_state_dict(checkpoint['vision_encoder_state_dict'])
        self.vision_tower.to(device)
        self.vision_tower.to(dtype=self._dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        
        # Build anomaly expert if present
        if checkpoint.get('anomaly_expert_state_dict') is not None:
            print("Loading anomaly expert...")
            self.anomaly_expert = AnomalyOV()
            self.anomaly_expert.load_state_dict(checkpoint['anomaly_expert_state_dict'])
            self.anomaly_expert.to(dtype=self._dtype, device=device)
            self.anomaly_expert.requires_grad_(False)
            self.anomaly_expert.eval()
        
        self.image_processor = self.vision_tower.image_processor
    
    def _load_from_expert_weights(self, weights_path: str, device: torch.device) -> None:
        """Load vision tower from pretrained and anomaly expert from weights file."""
        # Fallback dtype for CPU/MPS
        if device.type in ('cpu', 'mps'):
            self._dtype = torch.float32
        
        # Build vision tower (downloads from HuggingFace if needed)
        vision_tower_name = 'google/siglip-so400m-patch14-384'
        print(f"Loading vision tower: {vision_tower_name}")
        
        self.vision_tower = SigLipVisionTower(vision_tower_name, vision_tower_cfg={}, delay_load=False)
        self.vision_tower.to(device)
        self.vision_tower.to(dtype=self._dtype)
        self.vision_tower.requires_grad_(False)
        self.vision_tower.eval()
        
        # Build anomaly expert
        print(f"Loading anomaly expert from: {weights_path}")
        self.anomaly_expert = AnomalyOV()
        self.anomaly_expert.load_zero_shot_weights(path=weights_path, device=str(device))
        self.anomaly_expert.freeze_layers()
        self.anomaly_expert.to(dtype=self._dtype, device=device)
        self.anomaly_expert.requires_grad_(False)
        self.anomaly_expert.eval()
        
        self.image_processor = self.vision_tower.image_processor
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        """
        Predict whether an image is real or fake.
        
        Note: For AnomalyOV, we need to use the internal image processor
        rather than the standard preprocessing, so we load from image_path.
        
        Args:
            image_tensor: Preprocessed image tensor (may be ignored for this detector)
            image_path: Path to the original image
            
        Returns:
            float: Confidence score in [0, 1], where higher values indicate 
                   the image is more likely to be FAKE (anomalous).
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load and preprocess image using the internal processor
        image = Image.open(image_path).convert('RGB')
        processed = self.image_processor.preprocess([image], return_tensors='pt')
        pixel_values = processed['pixel_values'][0]  # Get the tensor
        
        # Convert to tensor and move to device
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self._dtype)
        
        with torch.no_grad():
            # Encode images through vision tower
            image_features, image_level_features = self.vision_tower(pixel_values)
            
            # Get anomaly prediction
            # split_sizes represents number of patches per image (for batching)
            split_sizes = [image_features.shape[0]]
            
            _, _, final_prediction = self.anomaly_expert(
                image_features, 
                image_level_features, 
                split_sizes,
                return_anomaly_map=False
            )
            
            # final_prediction is in [0, 1] where 1 = anomalous (fake)
            confidence = float(final_prediction.squeeze().cpu().item())
        
        return confidence
    
    def predict_with_map(self, image_path: str, anomaly_map_size: tuple = (224, 224)):
        """
        Predict whether an image is fake and return the anomaly map.
        
        Args:
            image_path: Path to the image
            anomaly_map_size: Size of the returned anomaly map
            
        Returns:
            tuple: (confidence, anomaly_map)
                - confidence: float in [0, 1], higher = more likely fake
                - anomaly_map: torch.Tensor of shape (1, 1, H, W)
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        processed = self.image_processor.preprocess([image], return_tensors='pt')
        pixel_values = processed['pixel_values'][0]
        
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self._dtype)
        
        with torch.no_grad():
            image_features, image_level_features = self.vision_tower(pixel_values)
            split_sizes = [image_features.shape[0]]
            
            _, _, final_prediction, anomaly_map = self.anomaly_expert(
                image_features,
                image_level_features,
                split_sizes,
                return_anomaly_map=True,
                anomaly_map_size=anomaly_map_size
            )
            
            confidence = float(final_prediction.squeeze().cpu().item())
            
        return confidence, anomaly_map

    def predict_with_vulnerability(
        self, 
        image, 
        device="cpu", 
        epsilon: float = 0.05,
        top_k: float = 0.5,
        noise_mode: str = "random",
        anomaly_map_size: tuple = (384, 384),  # SigLip native resolution
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict and compute vulnerability information for AnomalyOV.
        
        This method performs prediction and computes vulnerability analysis
        by comparing the anomaly map before and after adversarial attack.
        
        Args:
            image: Input image (PIL Image or path string)
            device: Device for computation (can be overridden by self.device)
            epsilon: Attack strength for adversarial perturbation
            top_k: Fraction of top anomaly regions to target in attack
            noise_mode: Attack type - 'random', 'structured', 'fgsm', 'pgd', 'deepfool'
            anomaly_map_size: Size of the output anomaly maps
            **kwargs: Additional arguments (ignored)
        
        Returns:
            dict with keys:
                - 'prediction': float, the model's anomaly score
                - 'input_tensor': torch.Tensor, the preprocessed input [1, V, 3, H, W]
                - 'anomaly_map': torch.Tensor, the anomaly map [V, 1, H, W]
                - 'anomaly_map_attacked': torch.Tensor, anomaly map of attacked input [V, 1, H, W]
                - 'vulnerability_map': torch.Tensor, |original_map - attacked_map| [V, 1, H, W]
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Import vulnerability functions
        from src.utils.vulnerability_map import adversarial_recompute
        
        # Use self.device if available
        device = self.device if self.device is not None else device
        
        # Load and preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or a file path string")
        
        processed = self.image_processor.preprocess([image], return_tensors='pt')
        pixel_values = processed['pixel_values'][0]
        
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        
        # Ensure shape is [1, V, 3, H, W] for the vulnerability map functions
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0).to(device, dtype=self._dtype)
        # pixel_values shape: [1, 1, 3, H, W] where V=1 (single view)
        
        # Create a wrapper model compatible with vulnerability_map functions
        model_wrapper = _AnomalyOVWrapper(
            vision_tower=self.vision_tower,
            anomaly_expert=self.anomaly_expert,
            dtype=self._dtype,
            device=device
        )
        
        # Get original prediction and anomaly map
        with torch.no_grad():
            # Forward pass for original prediction
            image_features, image_level_features = self.vision_tower(
                pixel_values.squeeze(0).squeeze(0).unsqueeze(0)  # [1, 3, H, W]
            )
            split_sizes = [image_features.shape[0]]
            
            _, _, final_prediction, anomaly_map_orig = self.anomaly_expert(
                image_features,
                image_level_features,
                split_sizes,
                return_anomaly_map=True,
                anomaly_map_size=anomaly_map_size
            )
            prediction = float(final_prediction.squeeze().cpu().item())
        
        # Compute adversarial attack and difference map
        x_adv, am_adv, vulnerability_map, adv_score = adversarial_recompute(
            model=model_wrapper,
            image_tensor=pixel_values,
            original_anomaly_maps=anomaly_map_orig,
            device=str(device),
            epsilon=epsilon,
            top_k=top_k,
            noise_mode=noise_mode,
        )
        
        # Normalize anomaly_map to [V, 1, H, W] format
        if anomaly_map_orig.ndim == 4 and anomaly_map_orig.shape[0] == 1:
            anomaly_map = anomaly_map_orig  # Already [1, 1, H, W] -> treat as [V, 1, H, W]
        else:
            anomaly_map = anomaly_map_orig
        
        return {
            'prediction': prediction,
            'input_tensor': pixel_values,
            'anomaly_map': anomaly_map,
            'anomaly_map_attacked': am_adv,
            'vulnerability_map': vulnerability_map,
        }
    
    def visualize_vulnerability(
        self,
        image,
        output_path: str,
        device="cpu",
        dpi: int = 150,
        mask_path: Optional[str] = None,
        metrics_top_k: float = 0.1,
        **kwargs
    ) -> None:
        """
        Visualize vulnerability analysis for a single image.
        
        Creates a visualization with panels:
        1. Original image
        2. Anomaly map (original)
        3. Anomaly map (attacked)
        4. Vulnerability map (|original - attacked|)
        5. Ground truth mask (if provided)
        6. Prediction & metrics panel
        
        Args:
            image: Input image (PIL Image or path string)
            output_path: Path to save the visualization
            device: Device for computation
            dpi: DPI for saved image
            mask_path: Optional path to ground truth mask
            metrics_top_k: Top-k fraction for IoU metric computation
            **kwargs: Additional arguments passed to predict_with_vulnerability
        """
        # Get image path for loading original
        if isinstance(image, str):
            image_path = image
        else:
            image_path = None
        
        # Get vulnerability data
        result = self.predict_with_vulnerability(image, device=device, **kwargs)
        
        prediction = result['prediction']
        anomaly_map = result['anomaly_map']
        anomaly_map_attacked = result['anomaly_map_attacked']
        vulnerability_map = result['vulnerability_map']
        
        # Load mask and compute metrics if provided
        mask_np = None
        anomaly_metrics = None
        vuln_metrics = None
        
        if mask_path and os.path.exists(mask_path):
            mask_np = self._load_mask(mask_path)
            if mask_np is not None:
                anomaly_metrics, vuln_metrics = self._compute_metrics(
                    anomaly_map, vulnerability_map, mask_np, top_k=metrics_top_k
                )
        
        # Create visualization
        self._create_visualization(
            image_path=image_path,
            image=image if image_path is None else None,
            anomaly_map=anomaly_map,
            anomaly_map_attacked=anomaly_map_attacked,
            vulnerability_map=vulnerability_map,
            prediction=prediction,
            output_path=output_path,
            dpi=dpi,
            mask_np=mask_np,
            anomaly_metrics=anomaly_metrics,
            vuln_metrics=vuln_metrics,
        )
    
    def visualize_vulnerability_grid(
        self,
        filename: str,
        data_folder: str,
        output_path: str,
        device="cpu",
        dpi: int = 150,
        overlay_alpha: float = 0.4,
        **kwargs
    ) -> None:
        """
        Visualize vulnerability analysis in a grid format for the new dataset structure.
        
        Dataset structure expected under data_folder:
        - original/: Original unmodified images
        - samecat/: Inpainted images with same category
        - diffcat/: Inpainted images with different category  
        - masks/: Inpainting masks for samecat (b&w)
        - bboxs/: Inpainting bboxes for diffcat (b&w)
        
        Grid layout (5 rows x 3 columns):
        Columns: C1=Original, C2=Samecat, C3=Diffcat
        Rows:
          R1: Original image
          R2: Original image with mask overlay
          R3: Attacked image with mask overlay
          R4: Original image with vulnerability map (mask subtraction) overlay
          R5: Mask/BBox image
        
        Args:
            filename: The filename (without folder path) to process
            data_folder: Parent folder containing original/, samecat/, diffcat/, masks/, bboxs/
            output_path: Path to save the visualization
            device: Device for computation
            dpi: DPI for saved image
            overlay_alpha: Alpha for mask/map overlays
            **kwargs: Additional arguments passed to predict_with_vulnerability
        """
        # Build paths for all images
        original_path = os.path.join(data_folder, 'original', filename)
        samecat_path = os.path.join(data_folder, 'samecat', filename)
        diffcat_path = os.path.join(data_folder, 'diffcat', filename)
        mask_path = os.path.join(data_folder, 'masks', filename)
        bbox_path = os.path.join(data_folder, 'bboxs', filename)
        
        # Try different extensions for mask/bbox if exact filename doesn't exist
        mask_path = self._find_file_with_extensions(mask_path)
        bbox_path = self._find_file_with_extensions(bbox_path)
        
        # Verify required files exist
        missing = []
        if not os.path.exists(original_path):
            missing.append(f"original/{filename}")
        if not os.path.exists(samecat_path):
            missing.append(f"samecat/{filename}")
        if not os.path.exists(diffcat_path):
            missing.append(f"diffcat/{filename}")
        if mask_path is None:
            missing.append(f"masks/{filename}")
        if bbox_path is None:
            missing.append(f"bboxs/{filename}")
        
        if missing:
            raise FileNotFoundError(f"Missing files in {data_folder}: {missing}")
        
        # Load all images
        img_original = Image.open(original_path).convert('RGB')
        img_samecat = Image.open(samecat_path).convert('RGB')
        img_diffcat = Image.open(diffcat_path).convert('RGB')
        mask_np = self._load_mask(mask_path)
        bbox_np = self._load_mask(bbox_path)
        
        # For original image, create an all-black mask (no inpainting)
        black_mask_np = np.zeros_like(mask_np)
        
        # Get vulnerability data for each image type
        result_original = self.predict_with_vulnerability(original_path, device=device, **kwargs)
        result_samecat = self.predict_with_vulnerability(samecat_path, device=device, **kwargs)
        result_diffcat = self.predict_with_vulnerability(diffcat_path, device=device, **kwargs)
        
        # Extract anomaly maps and vulnerability maps
        am_original = self._map_to_numpy(result_original['anomaly_map'])
        am_original_attacked = self._map_to_numpy(result_original['anomaly_map_attacked'])
        vuln_original = self._map_to_numpy(result_original['vulnerability_map'])
        
        am_samecat = self._map_to_numpy(result_samecat['anomaly_map'])
        am_samecat_attacked = self._map_to_numpy(result_samecat['anomaly_map_attacked'])
        vuln_samecat = self._map_to_numpy(result_samecat['vulnerability_map'])
        
        am_diffcat = self._map_to_numpy(result_diffcat['anomaly_map'])
        am_diffcat_attacked = self._map_to_numpy(result_diffcat['anomaly_map_attacked'])
        vuln_diffcat = self._map_to_numpy(result_diffcat['vulnerability_map'])
        
        # Create figure with 5 rows x 3 columns
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        
        # Column titles
        col_titles = ['Original (C1)', 'Samecat (C2)', 'Diffcat (C3)']
        row_titles = [
            'R1: Image',
            'R2: Image + Anomaly Map',
            'R3: Attacked + Anomaly Map',
            'R4: Image + Vulnerability Map',
            'R5: Mask/BBox'
        ]
        
        # Data for each column
        columns_data = [
            {
                'image': img_original,
                'mask': black_mask_np,
                'am': am_original,
                'am_attacked': am_original_attacked,
                'vuln': vuln_original,
                'prediction': result_original['prediction'],
            },
            {
                'image': img_samecat,
                'mask': mask_np,
                'am': am_samecat,
                'am_attacked': am_samecat_attacked,
                'vuln': vuln_samecat,
                'prediction': result_samecat['prediction'],
            },
            {
                'image': img_diffcat,
                'mask': bbox_np,
                'am': am_diffcat,
                'am_attacked': am_diffcat_attacked,
                'vuln': vuln_diffcat,
                'prediction': result_diffcat['prediction'],
            },
        ]
        
        for col_idx, col_data in enumerate(columns_data):
            img = col_data['image']
            mask = col_data['mask']
            am = col_data['am']
            am_attacked = col_data['am_attacked']
            vuln = col_data['vuln']
            pred = col_data['prediction']
            
            # Resize maps to match image size if needed
            img_size = (img.size[1], img.size[0])  # (H, W)
            am_resized = self._resize_map(am, img_size)
            am_attacked_resized = self._resize_map(am_attacked, img_size)
            vuln_resized = self._resize_map(vuln, img_size)
            mask_resized = self._resize_map(mask, img_size)
            
            # R1: Original image
            axes[0, col_idx].imshow(img)
            pred_label = 'FAKE' if pred > 0.5 else 'REAL'
            axes[0, col_idx].set_title(f'{col_titles[col_idx]}\nPred: {pred_label} ({pred:.3f})', fontsize=10)
            axes[0, col_idx].axis('off')
            
            # R2: Original image with anomaly map overlay
            axes[1, col_idx].imshow(img)
            axes[1, col_idx].imshow(am_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
            axes[1, col_idx].axis('off')
            
            # R3: Attacked image with anomaly map overlay
            axes[2, col_idx].imshow(img)
            axes[2, col_idx].imshow(am_attacked_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
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
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    @staticmethod
    def _find_file_with_extensions(base_path: str) -> Optional[str]:
        """Try to find a file with different image extensions."""
        if os.path.exists(base_path):
            return base_path
        
        base, ext = os.path.splitext(base_path)
        for new_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            new_path = base + new_ext
            if os.path.exists(new_path):
                return new_path
        return None
    
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
    
    @staticmethod
    def _load_mask(mask_path: str) -> Optional[np.ndarray]:
        """Load a mask image and convert to grayscale numpy array."""
        try:
            mask = Image.open(mask_path).convert('L')
            mask_np = np.array(mask).astype(np.float32) / 255.0
            return mask_np
        except Exception as e:
            print(f"Warning: Failed to load mask {mask_path}: {e}")
            return None
    
    def _compute_metrics(
        self,
        anomaly_map: torch.Tensor,
        vulnerability_map: torch.Tensor,
        mask_np: np.ndarray,
        top_k: float = 0.1,
    ) -> tuple:
        """Compute metrics between anomaly/vulnerability maps and ground truth mask."""
        try:
            from src.utils.metrics import compute_mask_anomaly_metrics
        except ImportError:
            empty = {"iou_topk": float('nan'), "mass_frac": float('nan'), 
                     "roc_auc": float('nan'), "pr_auc": float('nan')}
            return empty, empty
        
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)
        
        anomaly_metrics = compute_mask_anomaly_metrics(
            anomaly_maps=anomaly_map,
            mask_image=mask_tensor,
            top_k=top_k,
            aggregate="mean",
            inpainted_is_white=True,
        )
        
        vuln_metrics = compute_mask_anomaly_metrics(
            anomaly_maps=vulnerability_map,
            mask_image=mask_tensor,
            top_k=top_k,
            aggregate="mean",
            inpainted_is_white=True,
        )
        
        return anomaly_metrics, vuln_metrics
    
    @staticmethod
    def _map_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """Convert a map tensor to numpy for visualization."""
        t = tensor.detach().cpu().float()
        while t.ndim > 2:
            t = t.squeeze(0)
        t = t - t.min()
        if t.max() > 0:
            t = t / t.max()
        return t.numpy()
    
    def _create_visualization(
        self,
        image_path: Optional[str],
        image: Optional[Image.Image],
        anomaly_map: torch.Tensor,
        anomaly_map_attacked: torch.Tensor,
        vulnerability_map: torch.Tensor,
        prediction: float,
        output_path: str,
        dpi: int = 150,
        mask_np: Optional[np.ndarray] = None,
        anomaly_metrics: Optional[dict] = None,
        vuln_metrics: Optional[dict] = None,
    ) -> None:
        """Create and save the visualization figure."""
        # Load original image for display
        if image_path:
            original_image = Image.open(image_path).convert('RGB')
        else:
            original_image = image
        
        # Convert tensors to numpy
        anomaly_np = self._map_to_numpy(anomaly_map)
        anomaly_attacked_np = self._map_to_numpy(anomaly_map_attacked)
        vuln_np = self._map_to_numpy(vulnerability_map)
        
        # Determine number of panels
        num_panels = 5 if mask_np is None else 6
        
        # Create figure
        fig, axes = plt.subplots(1, num_panels, figsize=(4 * num_panels, 4))
        
        # Panel 1: Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Panel 2: Anomaly map (original)
        im1 = axes[1].imshow(anomaly_np, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Anomaly Map\n(Original)', fontsize=12)
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Panel 3: Anomaly map (attacked)
        im2 = axes[2].imshow(anomaly_attacked_np, cmap='hot', vmin=0, vmax=1)
        axes[2].set_title('Anomaly Map\n(Attacked)', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        
        # Panel 4: Vulnerability map
        im3 = axes[3].imshow(vuln_np, cmap='hot', vmin=0, vmax=1)
        axes[3].set_title('Vulnerability Map\n(|orig - attacked|)', fontsize=12)
        axes[3].axis('off')
        plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
        
        # Panel 5 (or 6): Prediction and metrics info
        if mask_np is not None:
            axes[4].imshow(mask_np, cmap='gray', vmin=0, vmax=1)
            axes[4].set_title('Ground Truth Mask', fontsize=12)
            axes[4].axis('off')
            info_panel = axes[5]
        else:
            info_panel = axes[4]
        
        # Build info text
        info_panel.axis('off')
        pred_label = 'FAKE' if prediction > 0.5 else 'REAL'
        pred_color = 'red' if prediction > 0.5 else 'green'
        
        info_lines = [
            f"Prediction: {pred_label}",
            f"Score: {prediction:.4f}",
            "",
            "(0 = Real, 1 = Fake)",
        ]
        
        if anomaly_metrics is not None and vuln_metrics is not None:
            info_lines.extend([
                "",
                "─" * 20,
                "Anomaly Map Metrics:",
                f"  IoU: {anomaly_metrics.get('iou_topk', float('nan')):.4f}",
                f"  Mass: {anomaly_metrics.get('mass_frac', float('nan')):.4f}",
                f"  ROC-AUC: {anomaly_metrics.get('roc_auc', float('nan')):.4f}",
                f"  PR-AUC: {anomaly_metrics.get('pr_auc', float('nan')):.4f}",
                "",
                "Vuln Map Metrics:",
                f"  IoU: {vuln_metrics.get('iou_topk', float('nan')):.4f}",
                f"  Mass: {vuln_metrics.get('mass_frac', float('nan')):.4f}",
                f"  ROC-AUC: {vuln_metrics.get('roc_auc', float('nan')):.4f}",
                f"  PR-AUC: {vuln_metrics.get('pr_auc', float('nan')):.4f}",
            ])
        
        info_text = "\n".join(info_lines)
        info_panel.text(
            0.5, 0.5, info_text,
            transform=info_panel.transAxes,
            fontsize=10,
            verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor=pred_color, alpha=0.3),
            family='monospace',
        )
        info_panel.set_title('Prediction & Metrics', fontsize=12)
        
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # Save
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


class _AnomalyOVWrapper:
    """
    Internal wrapper to make AnomalyOVDetector compatible with vulnerability_map functions.
    
    The vulnerability_map module expects a model with get_anomaly_fetures_from_images method.
    """
    
    def __init__(self, vision_tower, anomaly_expert, dtype, device):
        self.vision_tower = vision_tower
        self.anomaly_expert = anomaly_expert
        self.dtype = dtype
        self.device = device
    
    def eval(self):
        """Set models to eval mode."""
        self.vision_tower.eval()
        self.anomaly_expert.eval()
        return self
    
    def get_anomaly_fetures_from_images(
        self, 
        image_tensor, 
        with_attention_map=True, 
        anomaly_map_size=(224, 224)
    ):
        """
        Process images and return anomaly features.
        
        Args:
            image_tensor: Tensor of shape [1, V, 3, H, W]
            with_attention_map: Whether to return attention maps
            anomaly_map_size: Size of the output anomaly map
            
        Returns:
            final_prediction, attn_maps, anomaly_map
        """
        # image_tensor: [1, V, 3, H, W]
        B, V, C, H, W = image_tensor.shape
        
        # Flatten to [B*V, C, H, W] for vision tower
        pixel_values = image_tensor.view(B * V, C, H, W).to(self.device, dtype=self.dtype)
        
        # Forward through vision tower
        image_features, image_level_features = self.vision_tower(pixel_values)
        
        # Forward through anomaly expert
        split_sizes = [image_features.shape[0]]
        
        patch_pred, image_pred, final_prediction, anomaly_map = self.anomaly_expert(
            image_features,
            image_level_features,
            split_sizes,
            return_anomaly_map=True,
            anomaly_map_size=anomaly_map_size
        )
        
        # For attention maps, use the patch predictions reshaped
        # attn_maps shape expected: [V, num_patches, H, W] or similar
        # We'll return a placeholder since the vulnerability_map mainly uses anomaly_map
        attn_maps = patch_pred.view(V, -1, 1, 1) if patch_pred is not None else torch.zeros(V, 1, 1, 1, device=self.device)
        
        return final_prediction, attn_maps, anomaly_map
