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

from utils.detector_wrapper import to_numpy_2d

# Set up path for internal imports within anomaly_ov
DETECTOR_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure anomaly_ov directory is in path for internal llava/src imports
if DETECTOR_DIR not in sys.path:
    sys.path.insert(0, DETECTOR_DIR)

# Import BaseDetector from the support module (parent directory)
from support.base_detector import BaseDetector, prepare_batch

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
    
    # Feature flags - this detector supports all features
    supports_explainability = True
    supports_vulnerability = True
    supports_adversarial = True
    
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
    
    def forward(self, images) -> torch.Tensor:
        """Predict whether image(s) are real or fake.
        
        This mirrors the batch-style structure of WaveRepDetector.forward:
        it accepts a single image or a batch and returns a CPU tensor of
        confidences.
        
        Args:
            images: Single image or iterable of images. Each element can be:
                - str (path to image)
                - PIL.Image.Image
                - np.ndarray
        
        Returns:
            torch.Tensor: 1D tensor of confidence scores on CPU, where higher
                values indicate the image is more likely to be FAKE (anomalous).
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Prepare a transform that uses the Anomaly-OV image processor
        def _transform(img):
            processed = self.image_processor.preprocess([img], return_tensors='pt')
            pixel_values = processed['pixel_values'][0]
            if not isinstance(pixel_values, torch.Tensor):
                pixel_values = torch.tensor(pixel_values)
            return pixel_values

        # Use shared batch preparation utility (mirrors WaveRepDetector.forward)
        pixel_values = prepare_batch(images, self.device, _transform)

        # Ensure shape is [B, V, 3, H, W]; most common case is V=1
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(1)

        b, v, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(b * v, c, h, w)

        with torch.no_grad():
            # Encode images through vision tower
            image_features, image_level_features = self.vision_tower(pixel_values)

            # split_sizes represents number of patches per image (for batching)
            patches_per_image = image_features.shape[0] // (b * v)
            split_sizes = [patches_per_image] * (b * v)

            _, _, final_prediction = self.anomaly_expert(
                image_features,
                image_level_features,
                split_sizes,
                return_anomaly_map=False,
                return_probabilities=False,
            )

        # final_prediction is in [0, 1] where 1 = anomalous (fake)
        final_prediction = final_prediction.view(b, v, -1).mean(dim=(1, 2))
        return final_prediction.detach().cpu().float()
    
    def explain(self, image: np.ndarray, anomaly_map_size: tuple = (224, 224)):
        """
        Predict whether an image is fake and return the anomaly map.
        
        Args:
            image: np.ndarray image
            anomaly_map_size: Size of the returned anomaly map
            
        Returns:
            tuple: (confidence, anomaly_map)
                - confidence: float in [0, 1], higher = more likely fake
                - anomaly_map: torch.Tensor of shape (1, 1, H, W)
        """
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
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
        
        anomaly_map = to_numpy_2d(anomaly_map, image.shape[:2])
        return anomaly_map

    # def explain(self, image: np.ndarray, map_size: tuple = (384, 384)) -> np.ndarray:
    #     """
    #     Generate explanation/saliency map for an image.
    #
    #     This method provides a unified interface for generating explanation maps.
    #     For AnomalyOV, the explanation map is the anomaly map produced by the model.
    #
    #     Args:
    #         image: RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]
    #         map_size: Size of the output map (H, W). Default: (384, 384) for SigLip resolution.
    #
    #     Returns:
    #         Explanation map as np.ndarray (H, W) float32 in [0, 1]
    #
    #     Raises:
    #         ValueError: If image is None or invalid
    #         RuntimeError: If model is not loaded
    #     """
    #     if image is None:
    #         raise ValueError(f"explain() received None image. model_name={self.name}")
    #
    #     if not isinstance(image, np.ndarray):
    #         raise ValueError(
    #             f"explain() expected np.ndarray, got {type(image)}. model_name={self.name}"
    #         )
    #
    #     if self.vision_tower is None or self.anomaly_expert is None:
    #         raise RuntimeError("Model not loaded. Call load() first.")
    #
    #     # Convert numpy array to PIL Image
    #     pil_image = Image.fromarray(image)
    #
    #     # Use image processor to preprocess
    #     processed = self.image_processor.preprocess([pil_image], return_tensors='pt')
    #     pixel_values = processed['pixel_values'][0]
    #
    #     if not isinstance(pixel_values, torch.Tensor):
    #         pixel_values = torch.tensor(pixel_values)
    #     pixel_values = pixel_values.unsqueeze(0).to(self.device, dtype=self._dtype)
    #
    #     with torch.no_grad():
    #         image_features, image_level_features = self.vision_tower(pixel_values)
    #         split_sizes = [image_features.shape[0]]
    #
    #         _, _, _, anomaly_map = self.anomaly_expert(
    #             image_features,
    #             image_level_features,
    #             split_sizes,
    #             return_anomaly_map=True,
    #             anomaly_map_size=map_size
    #         )
    #
    #     # Convert to numpy and normalize to [0, 1]
    #     exp_map = anomaly_map.detach().cpu().float()
    #
    #     # Squeeze extra dimensions: [1, 1, H, W] -> [H, W]
    #     while exp_map.ndim > 2:
    #         exp_map = exp_map.squeeze(0)
    #
    #     exp_map = exp_map.numpy()
    #
    #     # Normalize to [0, 1]
    #     exp_min, exp_max = exp_map.min(), exp_map.max()
    #     if exp_max > exp_min:
    #         exp_map = (exp_map - exp_min) / (exp_max - exp_min)
    #     else:
    #         exp_map = np.zeros_like(exp_map)
    #
    #     return exp_map.astype(np.float32)

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
        
        # Import vulnerability functions (use explicit import to avoid path conflicts)
        import importlib.util
        vuln_spec = importlib.util.spec_from_file_location(
            "vulnerability_map",
            os.path.join(DETECTOR_DIR, "src", "utils", "vulnerability_map.py")
        )
        vuln_module = importlib.util.module_from_spec(vuln_spec)
        vuln_spec.loader.exec_module(vuln_module)
        adversarial_recompute = vuln_module.adversarial_recompute
        
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
    
    def generate_adversarial_attack(
        self, 
        image_path: str, 
        attack_type: str, 
        epsilon: float = 0.05,
        output_path: Optional[str] = None,
        true_label: int = 1,
    ) -> str:
        """
        Generate an adversarial attack for a given image and save it.
        
        Uses the adversarial_recompute function from vulnerability_map module
        which applies top-k masked attacks based on anomaly maps.
        
        Args:
            image_path: Path to the input image
            attack_type: Type of attack ('fgsm', 'pgd', 'deepfool', 'random', 'structured')
            epsilon: Attack strength
            output_path: Path to save the adversarial image
            true_label: True label of the image (0=real, 1=fake). Attack targets opposite class.
            
        Returns:
            Path to the saved adversarial image
        """
        # Import vulnerability functions (use explicit import to avoid path conflicts)
        import importlib.util
        vuln_spec = importlib.util.spec_from_file_location(
            "vulnerability_map",
            os.path.join(DETECTOR_DIR, "src", "utils", "vulnerability_map.py")
        )
        vuln_module = importlib.util.module_from_spec(vuln_spec)
        vuln_spec.loader.exec_module(vuln_module)
        adversarial_recompute = vuln_module.adversarial_recompute
        
        if self.vision_tower is None or self.anomaly_expert is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Load original image to get its size
        img = Image.open(image_path).convert('RGB')
        original_size = img.size  # (W, H)
        
        # Preprocess image
        processed = self.image_processor.preprocess([img], return_tensors='pt')
        pixel_values = processed['pixel_values'][0]
        
        if not isinstance(pixel_values, torch.Tensor):
            pixel_values = torch.tensor(pixel_values)
        
        # Shape: [1, V, 3, H, W] where V=1 (single view)
        pixel_values = pixel_values.unsqueeze(0).unsqueeze(0).to(self.device, dtype=self._dtype)
        
        # Create model wrapper for vulnerability_map functions
        model_wrapper = _AnomalyOVWrapper(
            vision_tower=self.vision_tower,
            anomaly_expert=self.anomaly_expert,
            dtype=self._dtype,
            device=self.device
        )
        
        # Get original anomaly map first
        with torch.no_grad():
            image_features, image_level_features = self.vision_tower(
                pixel_values.squeeze(0).squeeze(0).unsqueeze(0)  # [1, 3, H, W]
            )
            split_sizes = [image_features.shape[0]]
            _, _, _, anomaly_map_orig = self.anomaly_expert(
                image_features,
                image_level_features,
                split_sizes,
                return_anomaly_map=True,
                anomaly_map_size=(384, 384)  # SigLip native resolution
            )
        
        # Generate adversarial image using adversarial_recompute
        # Pass true_label so the attack targets the opposite class
        x_adv, _, _, _ = adversarial_recompute(
            model=model_wrapper,
            image_tensor=pixel_values,
            original_anomaly_maps=anomaly_map_orig,
            device=str(self.device),
            epsilon=epsilon,
            top_k=0.5,
            noise_mode=attack_type,  # 'fgsm', 'pgd', 'deepfool', 'random', 'structured'
            true_label=true_label,
        )
        
        # Convert adversarial tensor back to image
        # x_adv shape: [1, V, 3, H, W] -> [3, H, W]
        x_adv_np = x_adv.squeeze(0).squeeze(0).cpu().float().numpy()
        
        # Denormalize from model's normalization to [0, 255]
        # SigLip typically normalizes to approximately [-1, 1] or [0, 1]
        x_adv_np = np.clip(x_adv_np, -1, 1)
        x_adv_np = ((x_adv_np + 1) / 2 * 255).astype(np.uint8)  # Assuming [-1, 1] normalization
        x_adv_np = np.transpose(x_adv_np, (1, 2, 0))  # CHW -> HWC
        
        # Create adversarial image and resize back to original size
        adv_img = Image.fromarray(x_adv_np)
        if adv_img.size != original_size:
            adv_img = adv_img.resize(original_size, Image.BILINEAR)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            adv_img.save(output_path)
            # print(f"[AnomalyOV] Generated {attack_type} adversarial image (label={true_label}): {output_path}")
            return output_path
        else:
            # Save to a temp location and return path
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                adv_img.save(f.name)
                return f.name
    
    def generate_adversarial(
        self,
        image_path: str,
        output_path: str,
        attack_type: str = "fgsm",
        epsilon: float = 0.05,
        true_label: int = 1,
        **kwargs
    ) -> str:
        """
        Alias for generate_adversarial_attack to maintain interface consistency with other detectors.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the adversarial image
            attack_type: Type of attack ('fgsm', 'pgd', 'deepfool', 'random', 'structured')
            epsilon: Attack strength
            true_label: True label of the image (0=real, 1=fake). Attack targets opposite class.
            **kwargs: Additional arguments (ignored)
            
        Returns:
            Path to the saved adversarial image
        """
        return self.generate_adversarial_attack(
            image_path=image_path,
            attack_type=attack_type,
            epsilon=epsilon,
            output_path=output_path,
            true_label=true_label,
        )
    
    def _get_or_generate_adversarial(
        self,
        benign_path: str,
        existing_adv_path: Optional[str],
        expected_adv_path: Optional[str],
        attack_type: str,
        epsilon: float,
        true_label: int = 1,
    ) -> tuple:
        """
        Get or generate an adversarial image.
        
        Args:
            benign_path: Path to the benign image
            existing_adv_path: Path to existing adversarial (may be None or not exist)
            expected_adv_path: Where to save generated adversarial if needed
            attack_type: Type of attack
            epsilon: Attack strength
            true_label: True label (0=real, 1=fake). Attack targets opposite class.
            
        Returns:
            Tuple of (adversarial_path, was_generated)
        """
        # First try existing adversarial
        if existing_adv_path and os.path.exists(existing_adv_path):
            return existing_adv_path, False
        
        # Try finding with different extensions
        if existing_adv_path:
            found = self._find_file_with_extensions(existing_adv_path)
            if found:
                return found, False
        
        # Need to generate - determine output path
        if expected_adv_path:
            output_path = expected_adv_path
        else:
            # Fallback: save next to benign image with attack suffix
            base, ext = os.path.splitext(benign_path)
            output_path = f"{base}_{attack_type}_adv.png"
        
        # Generate the adversarial image with correct label
        self.generate_adversarial_attack(
            image_path=benign_path,
            attack_type=attack_type,
            epsilon=epsilon,
            output_path=output_path,
            true_label=true_label,
        )
        
        return output_path, True
    
    def visualize_vulnerability_grid(
        self,
        image_set,
        output_path: str,
        attack_type: str = "pgd",
        dpi: int = 150,
        overlay_alpha: float = 0.4,
        epsilon: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize vulnerability analysis in a grid format.
        
        If adversarial images don't exist, they will be generated on-the-fly
        and saved to the expected paths in the image_set.
        
        Dataset structure (ImageSet contains paths for):
        - real, samecat, diffcat: Benign images
        - real_adv, samecat_adv, diffcat_adv: Pre-computed adversarial images (may be None)
        - real_adv_expected, samecat_adv_expected, diffcat_adv_expected: Where to save generated images
        - mask, bbox: Inpainting masks/bboxes
        
        Grid layout (5 rows x 3 columns):
        Columns: C1=Real, C2=Samecat, C3=Diffcat
        Rows:
          R1: Original image
          R2: Original image with anomaly map overlay
          R3: Attacked image with anomaly map overlay
          R4: Original image with vulnerability map (|orig_am - attacked_am|) overlay
          R5: Mask/BBox image
        
        Args:
            image_set: ImageSet dataclass containing all image paths
            output_path: Path to save the visualization
            attack_type: Name of the attack type (for labeling and generation)
            dpi: DPI for saved image
            overlay_alpha: Alpha for map overlays
            epsilon: Attack strength for adversarial generation
            **kwargs: Additional arguments (ignored)
        
        Returns:
            dict with 'generated_adversarial': True if any adversarial images were generated
        """
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
        
        # Load all images
        img_real = Image.open(image_set.real).convert('RGB')
        img_samecat = Image.open(image_set.samecat).convert('RGB')
        img_diffcat = Image.open(image_set.diffcat).convert('RGB')
        
        img_real_adv = Image.open(real_adv_path).convert('RGB')
        img_samecat_adv = Image.open(samecat_adv_path).convert('RGB')
        img_diffcat_adv = Image.open(diffcat_adv_path).convert('RGB')
        
        mask_np = self._load_mask(image_set.mask)
        bbox_np = self._load_mask(image_set.bbox)
        
        # For real image, create an all-black mask (no inpainting)
        black_mask_np = np.zeros_like(mask_np) if mask_np is not None else np.zeros((224, 224))
        
        # Get anomaly maps for benign images
        pred_real, am_real = self.predict_with_map(image_set.real)
        pred_samecat, am_samecat = self.predict_with_map(image_set.samecat)
        pred_diffcat, am_diffcat = self.predict_with_map(image_set.diffcat)
        
        # Get anomaly maps for adversarial images
        pred_real_adv, am_real_adv = self.predict_with_map(real_adv_path)
        pred_samecat_adv, am_samecat_adv = self.predict_with_map(samecat_adv_path)
        pred_diffcat_adv, am_diffcat_adv = self.predict_with_map(diffcat_adv_path)
        
        # Convert to numpy
        am_real_np = self._map_to_numpy(am_real)
        am_samecat_np = self._map_to_numpy(am_samecat)
        am_diffcat_np = self._map_to_numpy(am_diffcat)
        
        am_real_adv_np = self._map_to_numpy(am_real_adv)
        am_samecat_adv_np = self._map_to_numpy(am_samecat_adv)
        am_diffcat_adv_np = self._map_to_numpy(am_diffcat_adv)
        
        # Compute vulnerability maps (|original - attacked|)
        #vuln_real_np = np.abs(am_real_np - am_real_adv_np)
        #vuln_samecat_np = np.abs(am_samecat_np - am_samecat_adv_np)
        #vuln_diffcat_np = np.abs(am_diffcat_np - am_diffcat_adv_np)

        vuln_real_np = am_real_np - am_real_adv_np
        vuln_samecat_np = am_samecat_np - am_samecat_adv_np
        vuln_diffcat_np = am_diffcat_np - am_diffcat_adv_np
        
        # Create figure with 5 rows x 3 columns
        fig, axes = plt.subplots(5, 3, figsize=(12, 20))
        
        # Column titles
        col_titles = ['Real (C1)', 'Samecat (C2)', 'Diffcat (C3)']
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
                'image': img_real,
                'image_adv': img_real_adv,
                'mask': black_mask_np,
                'am': am_real_np,
                'am_adv': am_real_adv_np,
                'vuln': vuln_real_np,
                'pred': pred_real,
                'pred_adv': pred_real_adv,
            },
            {
                'image': img_samecat,
                'image_adv': img_samecat_adv,
                'mask': mask_np,
                'am': am_samecat_np,
                'am_adv': am_samecat_adv_np,
                'vuln': vuln_samecat_np,
                'pred': pred_samecat,
                'pred_adv': pred_samecat_adv,
            },
            {
                'image': img_diffcat,
                'image_adv': img_diffcat_adv,
                'mask': bbox_np,
                'am': am_diffcat_np,
                'am_adv': am_diffcat_adv_np,
                'vuln': vuln_diffcat_np,
                'pred': pred_diffcat,
                'pred_adv': pred_diffcat_adv,
            },
        ]
        
        for col_idx, col_data in enumerate(columns_data):
            img = col_data['image']
            img_adv = col_data['image_adv']
            mask = col_data['mask']
            am = col_data['am']
            am_adv = col_data['am_adv']
            vuln = col_data['vuln']
            pred = col_data['pred']
            pred_adv = col_data['pred_adv']
            
            # Resize maps to match image size if needed
            img_size = (img.size[1], img.size[0])  # (H, W)
            am_resized = self._resize_map(am, img_size)
            am_adv_resized = self._resize_map(am_adv, img_size)
            vuln_resized = self._resize_map(vuln, img_size)
            mask_resized = self._resize_map(mask, img_size) if mask is not None else np.zeros(img_size)
            
            # R1: Original image with prediction
            axes[0, col_idx].imshow(img)
            pred_label = 'FAKE' if pred > 0.5 else 'REAL'
            axes[0, col_idx].set_title(f'{col_titles[col_idx]}\nPred: {pred_label} ({pred:.3f})', fontsize=10)
            axes[0, col_idx].axis('off')
            
            # R2: Original image with anomaly map overlay
            axes[1, col_idx].imshow(img)
            axes[1, col_idx].imshow(am_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
            axes[1, col_idx].axis('off')
            
            # R3: Attacked image with anomaly map overlay
            pred_adv_label = 'FAKE' if pred_adv > 0.5 else 'REAL'
            axes[2, col_idx].imshow(img_adv)
            axes[2, col_idx].imshow(am_adv_resized, cmap='hot', alpha=overlay_alpha, vmin=0, vmax=1)
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
        
        plt.suptitle(f'Vulnerability Analysis - {image_set.filename} ({attack_type})', fontsize=12, y=1.01)
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
            # Import metrics (use explicit import to avoid path conflicts)
            import importlib.util
            metrics_spec = importlib.util.spec_from_file_location(
                "metrics",
                os.path.join(DETECTOR_DIR, "src", "utils", "metrics.py")
            )
            metrics_module = importlib.util.module_from_spec(metrics_spec)
            metrics_spec.loader.exec_module(metrics_module)
            compute_mask_anomaly_metrics = metrics_module.compute_mask_anomaly_metrics
        except (ImportError, Exception):
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
