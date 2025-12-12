"""
Base Detector Module

This module provides the abstract base class for all detectors in the framework.
It defines the interface for prediction, explainability maps, vulnerability analysis,
and adversarial attack generation.

Subclasses should implement the required abstract methods to provide detector-specific
behavior while inheriting common functionality.
"""

import csv
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .detect_utils import get_device, load_image
from .lime_explain import lime_explain


class BaseDetector(ABC):
    """
    Abstract base class that unifies I/O, batching, and explainability for detectors.
    
    This class provides:
    - Core prediction interface
    - Explainability map generation (optional, detector-specific)
    - Vulnerability analysis (optional, detector-specific)
    - Adversarial attack generation (optional, detector-specific)
    - Batch processing utilities
    
    Required implementations:
        - name: Unique model name string (class attribute)
        - load(): Load model weights
        - predict(): Single image prediction
    
    Optional implementations (for explainability/vulnerability):
        - _compute_explanation_map(): Generate explanation/saliency map
        - _compute_vulnerability_map(): Generate vulnerability map after attack
        - _generate_adversarial_image(): Generate adversarial perturbation
        - visualize_vulnerability_grid(): Create grid visualization
    
    Attributes:
        name (str): Unique identifier for the detector
        device (torch.device): Device for computation
        model: The underlying model (type depends on detector)
    """
    
    name: str = "base"
    
    # Feature flags - subclasses should set these to True if they implement the feature
    supports_explainability: bool = False
    supports_vulnerability: bool = False
    supports_adversarial: bool = False
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize the base detector.
        
        Args:
            device: Torch device for computation. If None, auto-detects best available.
        """
        self.device = device or get_device()
        self.model = None
    
    # =========================================================================
    # REQUIRED ABSTRACT METHODS - Must be implemented by all subclasses
    # =========================================================================
    
    @abstractmethod
    def load(self, model_id: Optional[str] = None) -> None:
        """
        Load model weights from disk or hub.
        
        Args:
            model_id: Path to weights file or model identifier.
                     If None, uses default weights location.
        
        Raises:
            FileNotFoundError: If weights file not found
            RuntimeError: If model loading fails
        """
        raise NotImplementedError
    
    @abstractmethod
    def forward(self, images):
        """
        Predict whether an image is fake.
        
        Args:
            images: Preprocessed image tensor
        
        Returns:
            float: Confidence score in [0, 1] where higher = more likely fake
        """
        raise NotImplementedError
    
    def __call__(self, images) -> float:
        """
        Predict whether an image is fake.
        
        Args:
            images: Preprocessed image tensor
        
        Returns:
            float: Confidence score in [0, 1] where higher = more likely fake
        """
        return self.forward(images)
    
    # =========================================================================
    # OPTIONAL ABSTRACT METHODS - Implement for explainability/vulnerability
    # =========================================================================
    
    def _compute_explanation_map(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        """
        Compute explanation/saliency map for an image.
        
        This is the core method for generating model explanations. Subclasses
        should override this to provide detector-specific explanation methods
        (e.g., GradCAM, attention maps, anomaly maps).
        
        Args:
            image: Input image as path, PIL Image, or tensor
            map_size: Desired output map size (H, W). If None, uses detector default.
            **kwargs: Detector-specific arguments
        
        Returns:
            Tuple of:
                - confidence (float): Model's prediction confidence [0, 1]
                - explanation_map (np.ndarray): 2D array of shape (H, W) in [0, 1]
        
        Raises:
            NotImplementedError: If detector doesn't support explainability
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _compute_explanation_map(). "
            f"Set supports_explainability=True and implement this method to enable "
            f"explanation map generation."
        )
    
    def _compute_vulnerability_map(
        self,
        image: Union[str, Image.Image, torch.Tensor],
        attack_type: str = "fgsm",
        epsilon: float = 0.03,
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute vulnerability map by comparing explanation before/after attack.
        
        This method generates an adversarial perturbation and computes how
        the explanation map changes, revealing model vulnerabilities.
        
        Args:
            image: Input image as path, PIL Image, or tensor
            attack_type: Type of adversarial attack ('fgsm', 'pgd', 'deepfool', etc.)
            epsilon: Attack strength/perturbation budget
            map_size: Desired output map size (H, W)
            **kwargs: Detector-specific arguments (e.g., true_label, top_k)
        
        Returns:
            Dict containing:
                - 'prediction': float, original prediction confidence
                - 'prediction_attacked': float, prediction after attack
                - 'explanation_map': np.ndarray, original explanation map
                - 'explanation_map_attacked': np.ndarray, explanation map after attack
                - 'vulnerability_map': np.ndarray, |original - attacked| difference
                - 'input_tensor': torch.Tensor, preprocessed input (optional)
        
        Raises:
            NotImplementedError: If detector doesn't support vulnerability analysis
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _compute_vulnerability_map(). "
            f"Set supports_vulnerability=True and implement this method to enable "
            f"vulnerability analysis."
        )
    
    def _generate_adversarial_image(
        self,
        image_path: str,
        output_path: str,
        attack_type: str = "fgsm",
        epsilon: float = 0.03,
        true_label: int = 1,
        **kwargs
    ) -> str:
        """
        Generate and save an adversarial image.
        
        This method applies an adversarial attack to the input image and
        saves the result. The attack should target flipping the prediction.
        
        Args:
            image_path: Path to the original image
            output_path: Path to save the adversarial image
            attack_type: Type of attack ('fgsm', 'pgd', 'deepfool', etc.)
            epsilon: Attack strength/perturbation budget
            true_label: True label (0=real, 1=fake). Attack targets opposite class.
            **kwargs: Detector-specific attack parameters
        
        Returns:
            str: Path to the saved adversarial image
        
        Raises:
            NotImplementedError: If detector doesn't support adversarial generation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement _generate_adversarial_image(). "
            f"Set supports_adversarial=True and implement this method to enable "
            f"adversarial image generation."
        )
    
    # =========================================================================
    # PUBLIC API METHODS - High-level interface for users
    # =========================================================================
    
    def predict_with_map(
        self,
        image: Union[str, Image.Image],
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Tuple[float, np.ndarray]:
        """
        Predict and return explanation map.
        
        This is the primary public API for getting predictions with explanations.
        
        Args:
            image: Input image as path or PIL Image
            map_size: Desired output map size (H, W)
            **kwargs: Additional arguments passed to _compute_explanation_map
        
        Returns:
            Tuple of (confidence, explanation_map)
        
        Raises:
            NotImplementedError: If detector doesn't support explainability
            RuntimeError: If model not loaded
        """
        if not self.supports_explainability:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support explanation maps. "
                f"Check detector.supports_explainability before calling this method."
            )
        return self._compute_explanation_map(image, map_size=map_size, **kwargs)
    
    def predict_with_vulnerability(
        self,
        image: Union[str, Image.Image],
        attack_type: str = "fgsm",
        epsilon: float = 0.03,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Predict and compute vulnerability analysis.
        
        This method returns comprehensive vulnerability information including
        original and attacked explanations.
        
        Args:
            image: Input image as path or PIL Image
            attack_type: Type of adversarial attack
            epsilon: Attack strength
            **kwargs: Additional arguments passed to _compute_vulnerability_map
        
        Returns:
            Dict with prediction, explanation maps, and vulnerability map
        
        Raises:
            NotImplementedError: If detector doesn't support vulnerability analysis
            RuntimeError: If model not loaded
        """
        if not self.supports_vulnerability:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support vulnerability analysis. "
                f"Check detector.supports_vulnerability before calling this method."
            )
        return self._compute_vulnerability_map(
            image, attack_type=attack_type, epsilon=epsilon, **kwargs
        )
    
    def generate_adversarial(
        self,
        image_path: str,
        output_path: str,
        attack_type: str = "fgsm",
        epsilon: float = 0.03,
        true_label: int = 1,
        **kwargs
    ) -> str:
        """
        Generate adversarial image.
        
        Public API for generating adversarial perturbations.
        
        Args:
            image_path: Path to original image
            output_path: Path to save adversarial image
            attack_type: Type of attack
            epsilon: Attack strength
            true_label: True label (0=real, 1=fake)
            **kwargs: Additional attack parameters
        
        Returns:
            Path to saved adversarial image
        
        Raises:
            NotImplementedError: If detector doesn't support adversarial generation
            RuntimeError: If model not loaded
        """
        if not self.supports_adversarial:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support adversarial generation. "
                f"Check detector.supports_adversarial before calling this method."
            )
        return self._generate_adversarial_image(
            image_path=image_path,
            output_path=output_path,
            attack_type=attack_type,
            epsilon=epsilon,
            true_label=true_label,
            **kwargs
        )
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def visualize_vulnerability(
        self,
        image: Union[str, Image.Image],
        output_path: str,
        dpi: int = 150,
        mask_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """
        Visualize vulnerability analysis for a single image.
        
        Creates and saves a visualization showing the original image,
        explanation map, and vulnerability analysis.
        
        Args:
            image: Input image (PIL Image or path string)
            output_path: Path to save the visualization
            dpi: DPI for saved image
            mask_path: Optional path to ground truth mask for comparison
            **kwargs: Additional arguments passed to vulnerability computation
        
        Raises:
            NotImplementedError: If detector doesn't support visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visualize_vulnerability(). "
            f"Override this method in subclasses that support visualization."
        )
    
    def visualize_vulnerability_grid(
        self,
        image_set,
        output_path: str,
        attack_type: str = "pgd",
        dpi: int = 150,
        overlay_alpha: float = 0.4,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Visualize vulnerability analysis in a grid format.
        
        Creates a grid visualization comparing benign and adversarial
        images across different image types (real, samecat, diffcat).
        
        Args:
            image_set: ImageSet dataclass containing all image paths
            output_path: Path to save the visualization
            attack_type: Name of the attack type (for labeling)
            dpi: DPI for saved image
            overlay_alpha: Alpha for map overlays
            **kwargs: Additional arguments
        
        Returns:
            Dict with metadata (e.g., 'generated_adversarial': bool)
        
        Raises:
            NotImplementedError: If detector doesn't support grid visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visualize_vulnerability_grid(). "
            f"Override this method in subclasses that support grid visualization."
        )
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_or_generate_adversarial(
        self,
        benign_path: str,
        existing_adv_path: Optional[str],
        expected_adv_path: Optional[str],
        attack_type: str,
        epsilon: float = 0.03,
        true_label: int = 1,
    ) -> Tuple[str, bool]:
        """
        Get existing adversarial image or generate if not found.
        
        Helper method for grid visualization that handles caching of
        adversarial images.
        
        Args:
            benign_path: Path to benign image
            existing_adv_path: Path to existing adversarial image (may be None)
            expected_adv_path: Path where to save generated adversarial image
            attack_type: Type of attack
            epsilon: Attack strength
            true_label: True label (0=real, 1=fake)
        
        Returns:
            Tuple of (adversarial_image_path, was_generated)
        """
        # If adversarial image exists, use it
        if existing_adv_path is not None and os.path.exists(existing_adv_path):
            return existing_adv_path, False
        
        # Generate adversarial image
        if expected_adv_path is None:
            raise ValueError("No adversarial path and no expected path provided")
        
        self.generate_adversarial(
            image_path=benign_path,
            output_path=expected_adv_path,
            attack_type=attack_type,
            epsilon=epsilon,
            true_label=true_label,
        )
        
        return expected_adv_path, True
    
    def _load_mask(self, mask_path: Optional[str]) -> Optional[np.ndarray]:
        """
        Load a mask image from file.
        
        Args:
            mask_path: Path to mask image (can be None)
        
        Returns:
            Grayscale numpy array normalized to [0, 1], or None if path is None
        """
        if mask_path is None or not os.path.exists(mask_path):
            return None
        
        mask = Image.open(mask_path).convert('L')
        return np.array(mask).astype(np.float32) / 255.0
    
    @staticmethod
    def _ensure_image_path(image: Union[str, Image.Image]) -> str:
        """
        Ensure we have a file path for the image.
        
        If image is a PIL Image without a filename, saves to temp file.
        
        Args:
            image: Image path or PIL Image
        
        Returns:
            Path to the image file
        """
        if isinstance(image, str):
            return image
        elif hasattr(image, 'filename') and image.filename:
            return image.filename
        else:
            # Save PIL image to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                image.save(f.name)
                return f.name
    
    # =========================================================================
    # STATIC UTILITY METHODS
    # =========================================================================
    
    @staticmethod
    def label_from_conf(conf: float) -> int:
        """
        Convert confidence to binary label.
        
        Args:
            conf: Confidence score [0, 1]
        
        Returns:
            int: 1 if conf >= 0.5 (fake), 0 otherwise (real)
        """
        return int(np.round(conf))
    
    @staticmethod
    def list_images(folder: str) -> List[str]:
        """
        List all image files in a folder recursively.
        
        Args:
            folder: Path to folder
        
        Returns:
            Sorted list of image file paths
        """
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = []
        for root, _, fnames in os.walk(folder):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in exts:
                    files.append(os.path.join(root, f))
        files.sort()
        return files
    
    @staticmethod
    def ensure_parent(path: str) -> None:
        """
        Ensure parent directory exists.
        
        Args:
            path: File path whose parent directory should exist
        """
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    
    # =========================================================================
    # BATCH PROCESSING
    # =========================================================================
    
    @classmethod
    def run_batch(
        cls,
        detectors: List[Tuple['BaseDetector', Optional[str]]],
        input_folders: List[str],
        limit_per_folder: int,
        output_csv: str,
        batch_size: int = 16,
    ) -> None:
        """
        Run batch detection on images in input folders using specified detectors.
        
        Args:
            detectors: List of (detector_instance, model_id) to load and evaluate
            input_folders: List of folder paths containing images (recursively)
            limit_per_folder: Max number of sorted images to process per folder
            output_csv: Path to write results.csv with columns:
                folder, image, model, confidence, prediction
            batch_size: Batch size for detectors that support batch processing
        """
        # Load all detectors first
        for det, model_id in detectors:
            det.load(model_id)
        
        cls.ensure_parent(output_csv)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "image", "model", "confidence", "prediction"])
            
            # Process each folder
            for folder in input_folders:
                images = cls.list_images(folder)
                if limit_per_folder > 0:
                    images = images[:limit_per_folder]
                
                # For better throughput, iterate per detector
                for det, _ in detectors:
                    # If detector exposes batch_predict, use it in chunks
                    if hasattr(det, 'batch_predict') and callable(getattr(det, 'batch_predict')):
                        pbar = tqdm(total=len(images), desc=f"{det.name} on {os.path.basename(folder)}")
                        for i in range(0, len(images), max(1, batch_size)):
                            chunk = images[i:i + batch_size]
                            with torch.no_grad():
                                confs = getattr(det, 'batch_predict')(chunk)
                            for img_path, conf in zip(chunk, confs):
                                pred_flag = det.label_from_conf(float(conf))
                                writer.writerow([
                                    folder,
                                    os.path.relpath(img_path, folder),
                                    det.name,
                                    f"{float(conf):.6f}",
                                    str(pred_flag),
                                ])
                                f.flush()
                            pbar.update(len(chunk))
                        pbar.close()
                    else:
                        raise NotImplementedError('detector.predict() not implemented')
                        # Fallback to single-image predictions
                        # for img_path in tqdm(images, desc=f"{det.name} on {os.path.basename(folder)}"):
                        #     img_tensor_cpu, _ = load_image(img_path, size=224)
                        #     img_tensor = img_tensor_cpu.to(det.device)
                        #     with torch.no_grad():
                        #         conf = float(det.predict(img_tensor, img_path))
                        #     pred_flag = det.label_from_conf(conf)
                        #     writer.writerow([
                        #         folder,
                        #         os.path.relpath(img_path, folder),
                        #         det.name,
                        #         f"{conf:.6f}",
                        #         str(pred_flag),
                        #     ])
                        #     f.flush()
    
    def explain(
        self,
        batch: np.ndarray,
        method: str = "lime",
        class_idx: Optional[int] = None,
        batch_size: int = 1,
    ):  # -> Tuple[torch.Tensor, torch.Tensor]
        """
        Compute explainability maps for a batch of already-normalized images.

        Args:
            batch: (B, 3, H, W) tensor normalized with ImageNet stats.
            method: 'gradcam', 'gradsam' or 'smoothgrad' or 'integrated_gradients'.
            class_idx: target logit index; if None, use last logit (fake).
            batch_size: batch size for explainability maps.

        Returns:
            cam: (B, 1, H, W) maps in [0, 1].
            logits: (B, K) raw logits (no sigmoid).
        """
        m = method.lower()
        
        if m == "lime":
            cam = lime_explain(
                logits_fn=self.forward,
                images=batch,
                class_idx=class_idx,
                batch_size=batch_size,
            )
            return cam
        
        raise ValueError(f"Unknown explainability method '{method}'")


def prepare_batch(images, device, transform: Optional[Callable] = None):
    # Normalize input into a list of PIL Images
    frames = []
    
    # Treat list/tuple/ndarray as a batch
    if isinstance(images, (list, tuple, np.ndarray)):
        iterable = images
    else:
        iterable = [images]
    
    for img in iterable:
        # Allow passing paths, NumPy arrays, or PIL Images
        if isinstance(img, str):
            img = Image.open(img).convert('RGB')
        elif isinstance(img, np.ndarray):
            # Expect HWC, uint8 or float in [0, 255] / [0, 1]
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
        # If it's already a PIL.Image.Image or tensor, let the transform handle it
        if transform is not None:
            img = transform(img)
        frames.append(img)
    
    return torch.stack(frames, 0).to(device)
