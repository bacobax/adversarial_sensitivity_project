import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .detect_utils import get_device, load_image


class BaseDetector:
    """
    Base class that unifies I/O and batching for detectors.
    Subclasses should implement:
      - name: unique model name string
      - load(self, model_id: Optional[str], device: torch.device) -> None
      - predict(self, image_tensor: torch.Tensor, image_path: str) -> float
        Returns a confidence (float in [0,1]) that the image is fake.
    """
    
    name: str = "base"
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = None
    
    def load(self, model_id: Optional[str] = None) -> None:
        raise NotImplementedError
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        raise NotImplementedError
    
    def predict_with_vulnerability(self, image, device="cpu", **kwargs):
        """Predict and return basic vulnerability information.
        
        This method performs prediction and returns the basic outputs needed
        for vulnerability visualization. Each detector can override this to
        provide model-specific outputs.
        
        Args:
            image: Input image (PIL Image, path, or tensor)
            device: Device for computation
            **kwargs: Additional arguments (e.g., epsilon, top_k, noise_mode)
        
        Returns:
            dict with keys:
                - 'prediction': float, the model's prediction score
                - 'input_tensor': torch.Tensor, the preprocessed input tensor
                - 'anomaly_map': torch.Tensor, the anomaly/saliency map
                Additional model-specific keys may be included.
        
        Raises:
            NotImplementedError: If the detector does not support vulnerability analysis
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement predict_with_vulnerability. "
            "This method is only available for detectors that support vulnerability analysis."
        )
    
    def visualize_vulnerability(
        self,
        image,
        output_path: str,
        device="cpu",
        dpi: int = 150,
        mask_path: Optional[str] = None,
        **kwargs
    ) -> None:
        """Visualize vulnerability analysis for a single image.
        
        This method creates and saves a visualization of the vulnerability
        analysis. Each detector should override this to provide model-specific
        visualizations.
        
        Args:
            image: Input image (PIL Image or path string)
            output_path: Path to save the visualization
            device: Device for computation
            dpi: DPI for saved image
            mask_path: Optional path to ground truth mask
            **kwargs: Additional arguments passed to predict_with_vulnerability
        
        Raises:
            NotImplementedError: If the detector does not support visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visualize_vulnerability. "
            "This method is only available for detectors that support vulnerability visualization."
        )
    
    def visualize_vulnerability_grid(
        self,
        image_set,
        output_path: str,
        attack_type: str = "pgd",
        dpi: int = 150,
        overlay_alpha: float = 0.4,
        **kwargs
    ) -> None:
        """Visualize vulnerability analysis in a grid format using pre-computed adversarial images.
        
        This method creates a grid visualization comparing benign and adversarial
        images across different image types (real, samecat, diffcat).
        
        Args:
            image_set: ImageSet dataclass containing all image paths
            output_path: Path to save the visualization
            attack_type: Name of the attack type (for labeling)
            dpi: DPI for saved image
            overlay_alpha: Alpha for map overlays
            **kwargs: Additional arguments
        
        Raises:
            NotImplementedError: If the detector does not support grid visualization
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement visualize_vulnerability_grid. "
            "This method is only available for detectors that support grid visualization."
        )
    
    @staticmethod
    def label_from_conf(conf: float) -> int:
        # True => real, False => fake (per user spec: prediction true for real, false for fake)
        return int(np.round(conf))
    
    @staticmethod
    def list_images(folder: str) -> List[str]:
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
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    
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
        """
        # Load all detectors first
        for det, model_id in detectors:
            det.load(model_id)
        
        cls.ensure_parent(output_csv)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "image", "model", "confidence", "prediction"])  # prediction: True for real, False for fake
            
            # Process each folder
            for folder in input_folders:
                images = cls.list_images(folder)
                if limit_per_folder > 0:
                    images = images[:limit_per_folder]
                
                # For better throughput, iterate per detector and use batching if available
                for det, _ in detectors:
                    # If detector exposes batch_predict, use it in chunks
                    if hasattr(det, 'batch_predict') and callable(getattr(det, 'batch_predict')):
                        pbar = tqdm(total=len(images), desc=f"{det.name} on {os.path.basename(folder)}")
                        for i in range(0, len(images), max(1, batch_size)):
                            chunk = images[i:i + batch_size]
                            with torch.no_grad():
                                confs = getattr(det, 'batch_predict')(chunk)  # type: ignore
                            for img_path, conf in zip(chunk, confs):
                                pred_flag = det.label_from_conf(float(conf))
                                writer.writerow([folder, os.path.basename(img_path), det.name, float(conf), pred_flag])
                            pbar.update(len(chunk))
                        pbar.close()
                    else:
                        # Fallback: single-image predict
                        for img_path in tqdm(images, desc=f"{det.name} on {os.path.basename(folder)}"):
                            try:
                                img_tensor, _ = load_image(img_path)
                                img_tensor = img_tensor.to(det.device)
                                with torch.no_grad():
                                    conf = det.predict(img_tensor, img_path)
                                pred_flag = det.label_from_conf(float(conf))
                                writer.writerow([folder, os.path.basename(img_path), det.name, float(conf), pred_flag])
                            except Exception as e:
                                print(f"[warn] {det.name} failed on {img_path}: {e}")
                                writer.writerow([folder, os.path.basename(img_path), det.name, -1, -1])
import csv
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from .detect_utils import get_device, load_image


class BaseDetector:
    """
    Base class that unifies I/O and batching for detectors.
    Subclasses should implement:
      - name: unique model name string
      - load(self, model_id: Optional[str], device: torch.device) -> None
      - predict(self, image_tensor: torch.Tensor, image_path: str) -> float
        Returns a confidence (float in [0,1]) that the image is fake.
    """
    
    name: str = "base"
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or get_device()
        self.model = None
    
    def load(self, model_id: Optional[str] = None) -> None:
        raise NotImplementedError
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        raise NotImplementedError
    
    @staticmethod
    def label_from_conf(conf: float) -> int:
        # True => real, False => fake (per user spec: prediction true for real, false for fake)
        return int(np.round(conf))
    
    @staticmethod
    def list_images(folder: str) -> List[str]:
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
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    
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
        """
        # Load all detectors first
        for det, model_id in detectors:
            det.load(model_id)
        
        cls.ensure_parent(output_csv)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["folder", "image", "model", "confidence", "prediction"])  # prediction: True for real, False for fake
            
            # Process each folder
            for folder in input_folders:
                images = cls.list_images(folder)
                if limit_per_folder > 0:
                    images = images[:limit_per_folder]
                
                # For better throughput, iterate per detector and use batching if available
                for det, _ in detectors:
                    # If detector exposes batch_predict, use it in chunks
                    if hasattr(det, 'batch_predict') and callable(getattr(det, 'batch_predict')):
                        pbar = tqdm(total=len(images), desc=f"{det.name} on {os.path.basename(folder)}")
                        for i in range(0, len(images), max(1, batch_size)):
                            chunk = images[i:i + batch_size]
                            with torch.no_grad():
                                confs = getattr(det, 'batch_predict')(chunk)  # type: ignore
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
                        # Fallback to single-image predictions
                        for img_path in tqdm(images, desc=f"{det.name} on {os.path.basename(folder)}"):
                            img_tensor_cpu, _ = load_image(img_path, size=224)
                            img_tensor = img_tensor_cpu.to(det.device)
                            with torch.no_grad():
                                conf = float(det.predict(img_tensor, img_path))
                            pred_flag = det.label_from_conf(conf)
                            writer.writerow([
                                folder,
                                os.path.relpath(img_path, folder),
                                det.name,
                                f"{conf:.6f}",
                                str(pred_flag),
                            ])
                            f.flush()
    
    def supports_explainability(self) -> bool:  # NEW
        """
        Return True if this detector implements explain_image(), False otherwise.
        This lets callers check capability generically.  # NEW
        """
        return hasattr(self, "explain_image")  # NEW

    def explain(  # NEW
        self,
        batch: torch.Tensor,
        method: str = "gradcam",
        class_idx: Optional[int] = None,
    ):
        """
        Explainability API: detectors that support it must override this.  # NEW
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement explain()"
        )  # NEW