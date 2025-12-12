#!/usr/bin/env python3
"""
Spatial Correlation Test Script

This script computes explainability alignment and vulnerability alignment
with ground-truth manipulated regions for detector models under adversarial attacks.

For each detector model and attack type, it:
1. Computes explanation metrics (exp_orig vs gt_mask)
2. Computes vulnerability metrics (vuln_map vs gt_mask)
3. Outputs CSV metric files and grid visualizations

Dataset structure expected:
    ROOT_DATASET/
    ├── b-free/
    │   ├── real/        # Original real images
    │   ├── samecat/     # Inpainted with same category
    │   ├── diffcat/     # Inpainted with different category
    │   ├── mask/        # Binary masks aligned with samecat images
    │   └── bbox/        # Bounding box annotations for diffcat images
    └── adv_attacks/
        └── <model_name>/
            └── <attack_type>/
                ├── real/
                ├── samecat/
                └── diffcat/

Usage:
    python spatial_corr_test.py \
        --root_dataset ./datasets \
        --detectors AnomalyOV R50_nodown \
        --weights AnomalyOV:/path/to/weights.pt,R50_nodown:/path/to/weights.pt \
        --attacks pgd fgsm deepfool \
        --image_types samecat diffcat \
        --output_dir outputs/ \
        --max_visualizations 10
"""

import os
import sys
import argparse
import json
import logging
from typing import Optional, List, Dict, Tuple, Any, Union
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
import cv2
from tqdm import tqdm

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Import detector utilities and metrics
from support.detect_utils import get_device
from support.metrics import compute_mask_anomaly_metrics, MetricsAggregator

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_ATTACKS = {'pgd', 'fgsm', 'deepfool'}
SUPPORTED_IMAGE_TYPES = {'real', 'samecat', 'diffcat'}
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

DETECTOR_MAP = {
    'AnomalyOV': ('AnomalyOVDetector', os.path.join('models', 'anomaly_ov', 'detector.py')),
    'CLIP-D': ('CLIPDDetector', os.path.join('models', 'CLIP-D', 'detector.py')),
    'NPR': ('NPRDetector', os.path.join('models', 'NPR', 'detector.py')),
    'R50_nodown': ('R50NoDownDetector', os.path.join('models', 'R50_nodown', 'detector.py')),
    'R50_TF': ('R50TFDetector', os.path.join('models', 'R50_TF', 'detector.py')),
    'P2G': ('P2GDetector', os.path.join('models', 'P2G', 'detector.py')),
    'WaveRep': ('WaveRepDetector', os.path.join('models', 'WaveRep', 'detector.py')),
}

# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class SamplePaths:
    """Container for all image paths related to a single sample."""
    filename: str
    real: str
    samecat: str
    diffcat: str
    mask_samecat: str
    mask_diffcat: str  # Derived from bbox


@dataclass
class DetectorWrapper:
    """
    Unified interface wrapper for detectors.
    
    Ensures all detectors conform to the same external API:
        - name: str
        - explain(image: np.ndarray) -> np.ndarray
        - attack(image: np.ndarray, attack_type: str) -> np.ndarray
    """
    detector: Any
    _name: str = field(init=False)
    
    def __post_init__(self):
        self._name = getattr(self.detector, 'name', self.detector.__class__.__name__)
    
    @property
    def name(self) -> str:
        return self._name
    
    def explain(self, image: np.ndarray) -> np.ndarray:
        """
        Generate explanation/saliency map for an image.
        
        Args:
            image: RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]
        
        Returns:
            Explanation map as np.ndarray (H, W) float32 in [0, 1]
        
        Raises:
            ValueError: If image is None or invalid
        """
        if image is None:
            raise ValueError(
                f"explain() received None image. "
                f"model_name={self.name}"
            )
        
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"explain() expected np.ndarray, got {type(image)}. "
                f"model_name={self.name}"
            )
        
        # Convert to PIL for detector
        pil_image = Image.fromarray(image)
        
        # Get explanation map
        if hasattr(self.detector, 'predict_with_map'):
            _, exp_map = self.detector.predict_with_map(pil_image)
        elif hasattr(self.detector, '_compute_explanation_map'):
            _, exp_map = self.detector._compute_explanation_map(pil_image)
        else:
            raise NotImplementedError(
                f"Detector {self.name} does not implement predict_with_map() or "
                f"_compute_explanation_map()"
            )
        
        # Normalize to 2D float32 [0, 1]
        exp_map = self._to_numpy_2d(exp_map, image.shape[:2])
        return exp_map
    
    def attack(
        self, 
        image: np.ndarray, 
        attack_type: str,
        filename: str = "",
        image_type: str = ""
    ) -> np.ndarray:
        """
        Generate adversarial attack on image.
        
        Args:
            image: RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]
            attack_type: Type of attack ('pgd', 'fgsm', 'deepfool')
            filename: Source filename for error messages
            image_type: Image type for error messages
        
        Returns:
            Attacked RGB image as np.ndarray (H, W, 3) uint8 in [0, 255]
        
        Raises:
            ValueError: If required args are None or invalid
        """
        if image is None:
            raise ValueError(
                f"attack() received None image. "
                f"model_name={self.name}, attack_type={attack_type}, "
                f"filename={filename}, image_type={image_type}"
            )
        
        if attack_type is None:
            raise ValueError(
                f"attack() received None attack_type. "
                f"model_name={self.name}, filename={filename}, image_type={image_type}"
            )
        
        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"attack() expected np.ndarray, got {type(image)}. "
                f"model_name={self.name}, attack_type={attack_type}, "
                f"filename={filename}, image_type={image_type}"
            )
        
        import tempfile
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_input = f.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_output = f.name
        
        try:
            # Save input image
            Image.fromarray(image).save(temp_input)
            
            # Generate adversarial image
            if hasattr(self.detector, 'generate_adversarial'):
                self.detector.generate_adversarial(
                    image_path=temp_input,
                    output_path=temp_output,
                    attack_type=attack_type,
                    true_label=1,  # Assume fake images (samecat/diffcat)
                )
            elif hasattr(self.detector, '_generate_adversarial_image'):
                self.detector._generate_adversarial_image(
                    image_path=temp_input,
                    output_path=temp_output,
                    attack_type=attack_type,
                    true_label=1,
                )
            else:
                raise NotImplementedError(
                    f"Detector {self.name} does not implement generate_adversarial() or "
                    f"_generate_adversarial_image()"
                )
            
            # Load attacked image
            adv_image = np.array(Image.open(temp_output).convert('RGB'))
            
            # Ensure same size as input
            if adv_image.shape[:2] != image.shape[:2]:
                adv_pil = Image.fromarray(adv_image)
                adv_pil = adv_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
                adv_image = np.array(adv_pil)
            
            return adv_image
            
        finally:
            # Cleanup temp files
            for f in [temp_input, temp_output]:
                try:
                    os.unlink(f)
                except:
                    pass
    
    def _to_numpy_2d(
        self, 
        arr: Any, 
        target_size: Tuple[int, int]
    ) -> np.ndarray:
        """Convert array to 2D float32 numpy array normalized to [0, 1]."""
        # Handle torch tensors
        if hasattr(arr, 'detach'):
            arr = arr.detach().cpu().numpy()
        
        arr = np.asarray(arr, dtype=np.float32)
        
        # Squeeze extra dimensions
        while arr.ndim > 2:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
            elif arr.shape[-1] == 3:
                # RGB to grayscale
                arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            elif arr.shape[0] == 3:
                arr = 0.299 * arr[0] + 0.587 * arr[1] + 0.114 * arr[2]
            else:
                arr = arr[0]
        
        if arr.ndim != 2:
            raise ValueError(f"Could not convert to 2D array, got shape {arr.shape}")
        
        # Normalize to [0, 1]
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min)
        else:
            arr = np.zeros_like(arr)
        
        # Resize to target size if needed
        if arr.shape != target_size:
            arr = cv2.resize(arr, (target_size[1], target_size[0]), interpolation=cv2.INTER_CUBIC)
        
        return arr.astype(np.float32)


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_weights(weights_str: str) -> Dict[str, str]:
    """
    Parse detector weights specification string.
    
    Format: "detector1:/path/to/weights1.pt,detector2:/path/to/weights2.pt"
    
    Returns:
        Dict mapping detector names to weight paths
    """
    if not weights_str:
        return {}
    
    weights_dict = {}
    pairs = weights_str.split(',')
    
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        
        if ':' not in pair:
            raise ValueError(
                f"Invalid weights specification: '{pair}'. "
                f"Expected format: 'detector_name:/path/to/weights.pt'"
            )
        
        # Handle Windows paths with drive letters (e.g., C:\path)
        first_colon = pair.find(':')
        if len(pair) > first_colon + 1 and pair[first_colon + 1] in ('/', '\\'):
            # This might be a Windows path, look for second colon
            second_colon = pair.find(':', first_colon + 1)
            if second_colon != -1:
                detector_name = pair[:first_colon]
                path = pair[first_colon + 1:]
            else:
                detector_name = pair[:first_colon]
                path = pair[first_colon + 1:]
        else:
            detector_name = pair[:first_colon]
            path = pair[first_colon + 1:]
        
        weights_dict[detector_name] = path
    
    return weights_dict


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute spatial correlation metrics for detector explainability and vulnerability.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--root_dataset',
        type=str,
        required=True,
        help='Path to the dataset root containing b-free/ and adv_attacks/'
    )
    parser.add_argument(
        '--detectors',
        type=str,
        nargs='+',
        required=True,
        help=f'List of detector identifiers to evaluate. Available: {", ".join(DETECTOR_MAP.keys())}'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Mapping between detector names and weight paths. Format: "det1:/path1,det2:/path2"'
    )
    parser.add_argument(
        '--attacks',
        type=str,
        nargs='+',
        required=True,
        help=f'List of attack types to run. Supported: {", ".join(SUPPORTED_ATTACKS)}'
    )
    parser.add_argument(
        '--image_types',
        type=str,
        nargs='+',
        required=True,
        help=f'Subset of image types to process. Supported: {", ".join(SUPPORTED_IMAGE_TYPES)}'
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--topk_percent',
        type=float,
        nargs='+',
        default=[1.0],
        help='Percentage(s) of pixels considered as high-saliency for IoU (default: 1)'
    )
    parser.add_argument(
        '--max_visualizations',
        type=int,
        default=10,
        help='Maximum number of samples for which to generate grid visualizations (default: 10)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/',
        help='Root directory for all outputs (default: outputs/)'
    )
    parser.add_argument(
        '--overwrite_attacks',
        action='store_true',
        help='If set, recompute and overwrite attacked images even if they already exist'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for computation (default: auto-detect)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved visualization images (default: 150)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    validate_arguments(args)
    
    return args


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate all parsed arguments and raise errors for invalid inputs."""
    errors = []
    
    # Validate root_dataset exists
    if not os.path.isdir(args.root_dataset):
        errors.append(f"root_dataset does not exist: {args.root_dataset}")
    else:
        bfree = os.path.join(args.root_dataset, 'b-free')
        if not os.path.isdir(bfree):
            errors.append(f"b-free folder not found in root_dataset: {bfree}")
        else:
            for folder in ['real', 'samecat', 'diffcat', 'mask', 'bbox']:
                folder_path = os.path.join(bfree, folder)
                if not os.path.isdir(folder_path):
                    errors.append(f"Required b-free subfolder missing: {folder_path}")
    
    # Validate detectors
    for detector in args.detectors:
        if detector not in DETECTOR_MAP:
            errors.append(
                f"Unknown detector: {detector}. "
                f"Available: {', '.join(DETECTOR_MAP.keys())}"
            )
    
    # Parse and validate weights
    weights_dict = parse_weights(args.weights)
    for detector in args.detectors:
        if detector not in weights_dict:
            errors.append(
                f"Detector '{detector}' specified but no weight path provided. "
                f"Add '{detector}:/path/to/weights.pt' to --weights"
            )
        else:
            weight_path = weights_dict[detector]
            if not os.path.exists(weight_path):
                errors.append(f"Weight file not found for {detector}: {weight_path}")
    
    # Validate attacks
    for attack in args.attacks:
        if attack.lower() not in SUPPORTED_ATTACKS:
            errors.append(
                f"Unknown attack type: {attack}. "
                f"Supported: {', '.join(SUPPORTED_ATTACKS)}"
            )
    
    # Validate image_types
    for img_type in args.image_types:
        if img_type.lower() not in SUPPORTED_IMAGE_TYPES:
            errors.append(
                f"Unknown image type: {img_type}. "
                f"Supported: {', '.join(SUPPORTED_IMAGE_TYPES)}"
            )
    
    # Warn about 'real' in image_types
    if 'real' in [t.lower() for t in args.image_types]:
        logger.warning(
            "'real' is included in image_types but has no ground truth mask. "
            "Metrics will be skipped for real images."
        )
    
    # Validate topk_percent
    for topk in args.topk_percent:
        if not (0 < topk <= 100):
            errors.append(f"topk_percent must be in (0, 100], got: {topk}")
    
    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError(f"Argument validation failed with {len(errors)} error(s)")


def log_configuration(args: argparse.Namespace) -> None:
    """Log all parsed arguments at the beginning of execution."""
    logger.info("=" * 60)
    logger.info("Spatial Correlation Test Configuration")
    logger.info("=" * 60)
    logger.info(f"Root Dataset: {args.root_dataset}")
    logger.info(f"Detectors: {', '.join(args.detectors)}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Attacks: {', '.join(args.attacks)}")
    logger.info(f"Image Types: {', '.join(args.image_types)}")
    logger.info(f"Top-K Percent: {args.topk_percent}")
    logger.info(f"Max Visualizations: {args.max_visualizations}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Overwrite Attacks: {args.overwrite_attacks}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"DPI: {args.dpi}")
    logger.info("=" * 60)


# ============================================================================
# DATASET LOADING
# ============================================================================

def list_images(folder: str) -> List[str]:
    """List all image files in a folder (non-recursive), sorted."""
    files = []
    if not os.path.isdir(folder):
        return files
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return files


def find_file_with_extensions(base_path: str) -> Optional[str]:
    """Try to find a file with different image extensions."""
    if os.path.exists(base_path):
        return base_path
    
    base, ext = os.path.splitext(base_path)
    for new_ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']:
        new_path = base + new_ext
        if os.path.exists(new_path):
            return new_path
    return None


def load_image(path: str) -> np.ndarray:
    """
    Load an image as RGB numpy array.
    
    Args:
        path: Path to image file
    
    Returns:
        np.ndarray (H, W, 3) uint8 in [0, 255]
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If image can't be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image file not found: {path}")
    
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_mask(path: str) -> np.ndarray:
    """
    Load a binary mask as 2D numpy array.
    
    Args:
        path: Path to mask image file
    
    Returns:
        np.ndarray (H, W) float32 in [0, 1]
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If mask can't be loaded
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")
    
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to load mask: {path}")
    
    return mask.astype(np.float32) / 255.0


def bbox_to_mask(bbox_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Convert a bounding box annotation to a binary mask.
    
    The bbox file can be:
    1. An image file where the bbox region is marked (white on black or vice versa)
    2. A text file with bbox coordinates (x, y, w, h or x1, y1, x2, y2)
    
    Args:
        bbox_path: Path to bbox annotation file
        target_size: (H, W) size for output mask
    
    Returns:
        np.ndarray (H, W) float32 in [0, 1] where 1 = inside bbox
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If bbox format is not recognized
    """
    if not os.path.exists(bbox_path):
        raise FileNotFoundError(f"Bbox file not found: {bbox_path}")
    
    # Check if it's an image file
    ext = os.path.splitext(bbox_path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        # Load as image - assume it's a mask-like image
        mask = cv2.imread(bbox_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load bbox image: {bbox_path}")
        
        # Resize to target size
        if mask.shape != target_size:
            mask = cv2.resize(mask, (target_size[1], target_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Normalize to [0, 1]
        mask = mask.astype(np.float32) / 255.0
        
        # Binarize using Otsu or fixed threshold
        if mask.max() > 0:
            threshold = 0.5
            mask = (mask >= threshold).astype(np.float32)
        
        return mask
    
    # Try parsing as text file with coordinates
    elif ext in ['.txt', '.json', '.xml']:
        with open(bbox_path, 'r') as f:
            content = f.read().strip()
        
        # Create empty mask
        mask = np.zeros(target_size, dtype=np.float32)
        
        if ext == '.json':
            import json
            data = json.loads(content)
            # Assume format: {"x": x, "y": y, "width": w, "height": h}
            # or {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if 'x' in data and 'width' in data:
                x, y = int(data['x']), int(data['y'])
                w, h = int(data['width']), int(data['height'])
                mask[y:y+h, x:x+w] = 1.0
            elif 'x1' in data:
                x1, y1 = int(data['x1']), int(data['y1'])
                x2, y2 = int(data['x2']), int(data['y2'])
                mask[y1:y2, x1:x2] = 1.0
        else:
            # Try to parse as space or comma separated coordinates
            # Format: x y w h  or  x1 y1 x2 y2
            values = [float(v) for v in content.replace(',', ' ').split()]
            if len(values) == 4:
                if max(values) <= 1.0:
                    # Normalized coordinates
                    x, y, w, h = values
                    x = int(x * target_size[1])
                    y = int(y * target_size[0])
                    w = int(w * target_size[1])
                    h = int(h * target_size[0])
                else:
                    x, y, w, h = [int(v) for v in values]
                mask[y:y+h, x:x+w] = 1.0
        
        return mask
    
    else:
        raise ValueError(f"Unknown bbox file format: {bbox_path}")


def collect_sample_paths(
    root_dataset: str,
    image_types: List[str]
) -> List[SamplePaths]:
    """
    Collect all sample paths from the dataset.
    
    Args:
        root_dataset: Root dataset directory
        image_types: List of image types to process
    
    Returns:
        List of SamplePaths objects
    
    Raises:
        FileNotFoundError: If expected files are missing
    """
    logger.info("Collecting sample paths from dataset...")
    bfree = os.path.join(root_dataset, 'b-free')
    real_folder = os.path.join(bfree, 'real')
    
    filenames = list_images(real_folder)
    if not filenames:
        raise FileNotFoundError(f"No images found in {real_folder}")
    
    logger.info(f"Found {len(filenames)} image files in {real_folder}")
    
    samples = []
    errors = []
    
    logger.info("Validating sample paths...")
    for filename in tqdm(filenames, desc="Validating samples"):
        # Build paths for all required files
        real_path = os.path.join(bfree, 'real', filename)
        samecat_path = find_file_with_extensions(os.path.join(bfree, 'samecat', filename))
        diffcat_path = find_file_with_extensions(os.path.join(bfree, 'diffcat', filename))
        mask_path = find_file_with_extensions(os.path.join(bfree, 'mask', filename))
        bbox_path = find_file_with_extensions(os.path.join(bfree, 'bbox', filename))
        
        # Check for missing required files based on image_types
        missing = []
        
        if not os.path.exists(real_path):
            missing.append('real')
        
        if 'samecat' in image_types:
            if samecat_path is None:
                missing.append('samecat')
            if mask_path is None:
                missing.append('mask')
        
        if 'diffcat' in image_types:
            if diffcat_path is None:
                missing.append('diffcat')
            if bbox_path is None:
                missing.append('bbox')
        
        if missing:
            errors.append(f"{filename}: missing {', '.join(missing)}")
            continue
        
        samples.append(SamplePaths(
            filename=filename,
            real=real_path,
            samecat=samecat_path or "",
            diffcat=diffcat_path or "",
            mask_samecat=mask_path or "",
            mask_diffcat=bbox_path or "",  # Will be converted to mask later
        ))
    
    if errors:
        logger.error(f"Missing files for {len(errors)} samples:")
        for error in errors[:10]:
            logger.error(f"  {error}")
        if len(errors) > 10:
            logger.error(f"  ... and {len(errors) - 10} more")
        raise FileNotFoundError(
            f"Missing required files for {len(errors)} samples. "
            f"See log for details."
        )
    
    logger.info(f"Collected {len(samples)} sample paths")
    return samples


# ============================================================================
# DETECTOR LOADING
# ============================================================================

def load_detector_class(detector_name: str):
    """
    Dynamically load a detector class by name.
    
    Args:
        detector_name: Name of the detector (e.g., 'AnomalyOV')
    
    Returns:
        Detector class
    """
    if detector_name not in DETECTOR_MAP:
        raise ValueError(f"Unknown detector: {detector_name}")
    
    class_name, module_path = DETECTOR_MAP[detector_name]
    full_path = os.path.join(SCRIPT_DIR, module_path)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Detector module not found: {full_path}")
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(detector_name, full_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[detector_name] = module
    spec.loader.exec_module(module)
    
    return getattr(module, class_name)


def load_detector(
    detector_name: str,
    weights_path: str,
    device: torch.device
) -> DetectorWrapper:
    """
    Load a detector with its weights and wrap in unified interface.
    
    Args:
        detector_name: Name of the detector
        weights_path: Path to weights file
        device: Device for computation
    
    Returns:
        DetectorWrapper instance
    """
    logger.info(f"Loading detector: {detector_name}")
    logger.info(f"  Weights: {weights_path}")
    logger.info(f"  Device: {device}")
    
    DetectorClass = load_detector_class(detector_name)
    detector = DetectorClass(device=device)
    logger.info(f"  Loading model weights...")
    detector.load(weights_path)
    logger.info(f"  Model loaded successfully!")
    
    # Verify required capabilities
    logger.info(f"  Verifying detector capabilities...")
    if not (hasattr(detector, 'predict_with_map') or 
            hasattr(detector, '_compute_explanation_map')):
        raise ValueError(
            f"Detector {detector_name} does not support explanation maps"
        )
    
    if not (hasattr(detector, 'generate_adversarial') or 
            hasattr(detector, '_generate_adversarial_image')):
        raise ValueError(
            f"Detector {detector_name} does not support adversarial attacks"
        )
    
    logger.info(f"  ✓ Detector {detector_name} ready")
    return DetectorWrapper(detector=detector)


# ============================================================================
# ATTACK CACHING
# ============================================================================

def get_attack_cache_path(
    root_dataset: str,
    model_name: str,
    attack_type: str,
    image_type: str,
    filename: str
) -> str:
    """Get the path for a cached attacked image."""
    base_name = os.path.splitext(filename)[0]
    return os.path.join(
        root_dataset, 
        'adv_attacks', 
        model_name, 
        attack_type, 
        image_type,
        f"{base_name}.png"
    )


def get_or_generate_attacked_image(
    detector: DetectorWrapper,
    image: np.ndarray,
    attack_type: str,
    cache_path: str,
    overwrite: bool = False,
    filename: str = "",
    image_type: str = ""
) -> np.ndarray:
    """
    Get attacked image from cache or generate if not exists.
    
    Args:
        detector: Detector wrapper instance
        image: Original RGB image
        attack_type: Type of attack
        cache_path: Path to cache the attacked image
        overwrite: If True, regenerate even if cached
        filename: Filename for error messages
        image_type: Image type for error messages
    
    Returns:
        Attacked RGB image as np.ndarray (H, W, 3) uint8
    """
    if os.path.exists(cache_path) and not overwrite:
        return load_image(cache_path)
    
    # Generate attacked image
    adv_image = detector.attack(
        image, 
        attack_type,
        filename=filename,
        image_type=image_type
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    # Save to cache
    Image.fromarray(adv_image).save(cache_path)
    
    return adv_image


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(
    heatmap: np.ndarray,
    gt_mask: np.ndarray,
    topk_percents: List[float]
) -> Dict[str, float]:
    """
    Compute spatial correlation metrics between heatmap and ground truth mask.
    
    Args:
        heatmap: 2D float32 array (H, W) in [0, 1]
        gt_mask: 2D float32 array (H, W) in [0, 1]
        topk_percents: List of top-k percentages to compute IoU for
    
    Returns:
        Dict with metric values
    """
    all_metrics = {}
    
    for topk in topk_percents:
        topk_frac = topk / 100.0  # Convert percent to fraction
        
        metrics = compute_mask_anomaly_metrics(
            anomaly_map=heatmap,
            mask_image=gt_mask,
            top_k=topk_frac,
            inpainted_is_white=True,
        )
        
        # Add topk suffix if multiple topk values
        if len(topk_percents) > 1:
            for key, value in metrics.items():
                all_metrics[f"{key}@{topk}"] = value
        else:
            all_metrics.update(metrics)
    
    return all_metrics


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization_grid(
    images: Dict[str, np.ndarray],
    exp_orig: Dict[str, np.ndarray],
    exp_adv: Dict[str, np.ndarray],
    vuln_maps: Dict[str, np.ndarray],
    gt_masks: Dict[str, np.ndarray],
    filename: str,
    attack_type: str,
    output_path: str,
    dpi: int = 150
) -> None:
    """
    Create and save a grid visualization.
    
    Grid layout:
        Columns = {real, samecat, diffcat}
        Rows = {image, exp_orig, exp_adv, vuln_map, gt_mask}
    
    Args:
        images: Dict mapping image_type to RGB image array
        exp_orig: Dict mapping image_type to original explanation map
        exp_adv: Dict mapping image_type to attacked explanation map
        vuln_maps: Dict mapping image_type to vulnerability map
        gt_masks: Dict mapping image_type to ground truth mask
        filename: Sample filename
        attack_type: Attack type name
        output_path: Path to save visualization
        dpi: DPI for saved image
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    # Column order
    columns = ['real', 'samecat', 'diffcat']
    # Row names
    row_names = ['Image', 'Exp Original', 'Exp Attacked', 'Vulnerability', 'GT Mask']
    
    # Create figure
    fig = plt.figure(figsize=(12, 15))
    gs = gridspec.GridSpec(5, 3, figure=fig, hspace=0.1, wspace=0.05)
    
    for col_idx, img_type in enumerate(columns):
        if img_type not in images:
            continue
        
        # Row 0: Original image
        ax = fig.add_subplot(gs[0, col_idx])
        ax.imshow(images[img_type])
        if col_idx == 0:
            ax.set_ylabel(row_names[0], fontsize=10)
        ax.set_title(img_type.capitalize(), fontsize=12)
        ax.axis('off')
        
        # Row 1: Original explanation
        ax = fig.add_subplot(gs[1, col_idx])
        if img_type in exp_orig and exp_orig[img_type] is not None:
            ax.imshow(exp_orig[img_type], cmap='jet', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[1], fontsize=10)
        ax.axis('off')
        
        # Row 2: Attacked explanation
        ax = fig.add_subplot(gs[2, col_idx])
        if img_type in exp_adv and exp_adv[img_type] is not None:
            ax.imshow(exp_adv[img_type], cmap='jet', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[2], fontsize=10)
        ax.axis('off')
        
        # Row 3: Vulnerability map
        ax = fig.add_subplot(gs[3, col_idx])
        if img_type in vuln_maps and vuln_maps[img_type] is not None:
            ax.imshow(vuln_maps[img_type], cmap='hot', vmin=0, vmax=1)
        else:
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[3], fontsize=10)
        ax.axis('off')
        
        # Row 4: Ground truth mask
        ax = fig.add_subplot(gs[4, col_idx])
        if img_type in gt_masks and gt_masks[img_type] is not None:
            ax.imshow(gt_masks[img_type], cmap='gray', vmin=0, vmax=1)
        else:
            # Black mask for real images
            ax.imshow(np.zeros((10, 10)), cmap='gray')
        if col_idx == 0:
            ax.set_ylabel(row_names[4], fontsize=10)
        ax.axis('off')
    
    # Add title
    fig.suptitle(f'{filename} - {attack_type.upper()}', fontsize=14, y=0.98)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

def process_sample(
    sample: SamplePaths,
    detector: DetectorWrapper,
    attack_type: str,
    image_types: List[str],
    root_dataset: str,
    topk_percents: List[float],
    overwrite_attacks: bool,
    exp_cache: Dict[str, np.ndarray]
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Process a single sample for all requested image types.
    
    Args:
        sample: Sample paths
        detector: Detector wrapper
        attack_type: Attack type
        image_types: List of image types to process
        root_dataset: Root dataset path
        topk_percents: List of top-k percentages
        overwrite_attacks: Whether to overwrite cached attacks
        exp_cache: Cache for original explanation maps
    
    Returns:
        Tuple of:
            - explanation_metrics: Dict[image_type, metrics_dict]
            - vulnerability_metrics: Dict[image_type, metrics_dict]
            - visualization_data: Dict with data for visualization
    """
    explanation_metrics = {}
    vulnerability_metrics = {}
    vis_data = {
        'images': {},
        'exp_orig': {},
        'exp_adv': {},
        'vuln_maps': {},
        'gt_masks': {},
    }
    
    for img_type in image_types:
        # Skip metrics for 'real' (no ground truth mask)
        if img_type == 'real':
            # Still load for visualization
            try:
                vis_data['images']['real'] = load_image(sample.real)
            except:
                pass
            continue
        
        # Get image path
        if img_type == 'samecat':
            img_path = sample.samecat
            mask_path = sample.mask_samecat
        elif img_type == 'diffcat':
            img_path = sample.diffcat
            mask_path = sample.mask_diffcat
        else:
            continue
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        # Load image
        image = load_image(img_path)
        vis_data['images'][img_type] = image
        
        # Load/compute ground truth mask
        if img_type == 'samecat':
            gt_mask = load_mask(mask_path)
        else:  # diffcat
            gt_mask = bbox_to_mask(mask_path, image.shape[:2])
        vis_data['gt_masks'][img_type] = gt_mask
        
        # Compute or get cached original explanation
        cache_key = (detector.name, img_type, sample.filename)
        if cache_key in exp_cache:
            exp_orig = exp_cache[cache_key]
        else:
            exp_orig = detector.explain(image)
            exp_cache[cache_key] = exp_orig
        vis_data['exp_orig'][img_type] = exp_orig
        
        # Get or generate attacked image
        attack_cache_path = get_attack_cache_path(
            root_dataset, detector.name, attack_type, img_type, sample.filename
        )
        adv_image = get_or_generate_attacked_image(
            detector=detector,
            image=image,
            attack_type=attack_type,
            cache_path=attack_cache_path,
            overwrite=overwrite_attacks,
            filename=sample.filename,
            image_type=img_type
        )
        
        # Compute explanation on attacked image
        exp_adv = detector.explain(adv_image)
        vis_data['exp_adv'][img_type] = exp_adv
        
        # Compute vulnerability map
        #vuln_map = np.abs(exp_orig - exp_adv)
        vuln_map = exp_orig - exp_adv
        # Normalize vulnerability map to [0, 1]
        if vuln_map.max() > 0:
            vuln_map = vuln_map / vuln_map.max()
        vis_data['vuln_maps'][img_type] = vuln_map
        
        # Compute metrics
        # Explanation metrics: exp_orig vs gt_mask
        exp_metrics = compute_metrics(exp_orig, gt_mask, topk_percents)
        explanation_metrics[img_type] = exp_metrics
        
        # Vulnerability metrics: vuln_map vs gt_mask
        vuln_metrics = compute_metrics(vuln_map, gt_mask, topk_percents)
        vulnerability_metrics[img_type] = vuln_metrics
    
    return explanation_metrics, vulnerability_metrics, vis_data


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Log configuration
    log_configuration(args)
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Parse weights
    weights_dict = parse_weights(args.weights)
    
    # Collect sample paths
    logger.info("\n" + "="*60)
    logger.info("Stage 1: Collecting and validating sample paths")
    logger.info("="*60)
    samples = collect_sample_paths(args.root_dataset, args.image_types)
    
    # Sort samples for deterministic ordering
    samples.sort(key=lambda s: s.filename)
    logger.info(f"✓ {len(samples)} samples ready for processing")
    
    # Normalize image_types to lowercase
    image_types = [t.lower() for t in args.image_types]
    
    # Normalize attacks to lowercase
    attacks = [a.lower() for a in args.attacks]
    
    # Process each detector
    logger.info("\n" + "="*60)
    logger.info(f"Stage 2: Processing {len(args.detectors)} detector(s) with {len(attacks)} attack(s)")
    logger.info("="*60)
    
    for detector_idx, detector_name in enumerate(args.detectors, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing detector: {detector_name}")
        logger.info(f"{'='*60}")
        
        # Load detector
        try:
            detector = load_detector(
                detector_name,
                weights_dict[detector_name],
                device
            )
        except Exception as e:
            logger.error(f"Failed to load detector {detector_name}: {e}")
            continue
        
        # Create output directory for this detector
        detector_output = os.path.join(args.output_dir, detector_name)
        os.makedirs(detector_output, exist_ok=True)
        
        # Explanation metrics aggregator (computed once, attack-independent)
        exp_aggregator = MetricsAggregator()
        exp_processed = set()  # Track which (filename, image_type) have been computed
        
        # Cache for original explanation maps
        exp_cache: Dict[str, np.ndarray] = {}
        
        # Process each attack type
        for attack_idx, attack_type in enumerate(attacks, 1):
            logger.info(f"\n--- [{attack_idx}/{len(attacks)}] Processing attack: {attack_type.upper()} ---")
            logger.info(f"Samples to process: {len(samples)}")
            logger.info(f"Image types: {', '.join(image_types)}")
            
            # Vulnerability metrics aggregator (per attack)
            vuln_aggregator = MetricsAggregator()
            
            # Visualization counter
            vis_count = 0
            
            # Create attack-specific output directory
            attack_output = os.path.join(detector_output, attack_type)
            os.makedirs(attack_output, exist_ok=True)
            vis_output = os.path.join(detector_output, 'vis', attack_type)
            os.makedirs(vis_output, exist_ok=True)
            
            # Process samples
            pbar = tqdm(
                samples, 
                desc=f"Processing {detector_name}/{attack_type}",
                unit="sample",
                leave=True
            )
            for sample_idx, sample in enumerate(pbar, 1):
                # Update progress bar with current file
                pbar.set_postfix_str(f"{sample.filename[:30]}...", refresh=True)
                try:
                    exp_metrics, vuln_metrics, vis_data = process_sample(
                        sample=sample,
                        detector=detector,
                        attack_type=attack_type,
                        image_types=image_types,
                        root_dataset=args.root_dataset,
                        topk_percents=args.topk_percent,
                        overwrite_attacks=args.overwrite_attacks,
                        exp_cache=exp_cache
                    )
                    
                    # Update explanation metrics (only once per sample/image_type)
                    for img_type, metrics in exp_metrics.items():
                        key = (sample.filename, img_type)
                        if key not in exp_processed:
                            metadata = {
                                'filename': sample.filename,
                                'image_type': img_type,
                            }
                            if len(args.topk_percent) > 1:
                                metadata['topk_percent'] = args.topk_percent[0]
                            exp_aggregator.update(metrics, metadata=metadata)
                            exp_processed.add(key)
                    
                    # Update vulnerability metrics
                    for img_type, metrics in vuln_metrics.items():
                        metadata = {
                            'filename': sample.filename,
                            'image_type': img_type,
                            'attack_type': attack_type,
                        }
                        if len(args.topk_percent) > 1:
                            metadata['topk_percent'] = args.topk_percent[0]
                        vuln_aggregator.update(metrics, metadata=metadata)
                    
                    # Generate visualization if within limit
                    if vis_count < args.max_visualizations:
                        pbar.set_postfix_str(f"Generating visualization {vis_count+1}/{args.max_visualizations}", refresh=True)
                        base_name = os.path.splitext(sample.filename)[0]
                        vis_path = os.path.join(vis_output, f"{base_name}_grid.png")
                        
                        try:
                            create_visualization_grid(
                                images=vis_data['images'],
                                exp_orig=vis_data['exp_orig'],
                                exp_adv=vis_data['exp_adv'],
                                vuln_maps=vis_data['vuln_maps'],
                                gt_masks=vis_data['gt_masks'],
                                filename=sample.filename,
                                attack_type=attack_type,
                                output_path=vis_path,
                                dpi=args.dpi
                            )
                            vis_count += 1
                        except Exception as e:
                            logger.warning(f"Visualization failed for {sample.filename}: {e}")
                    
                except Exception as e:
                    pbar.set_postfix_str(f"ERROR: {str(e)[:30]}", refresh=True)
                    logger.error(f"Failed to process {sample.filename}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Close progress bar
            pbar.close()
            
            logger.info(f"Completed processing {len(samples)} samples")
            logger.info(f"Generated {vis_count} visualizations")
            
            # Save vulnerability metrics CSV
            if len(vuln_aggregator) > 0:
                logger.info(f"Saving vulnerability metrics...")
                # Ensure directory exists before saving
                os.makedirs(attack_output, exist_ok=True)
                vuln_csv = os.path.join(attack_output, "metrics_vulnerability.csv")
                vuln_aggregator.to_csv(vuln_csv)
                logger.info(f"✓ Vulnerability metrics saved to: {vuln_csv}")
                logger.info(f"  Samples: {len(vuln_aggregator)}")
                logger.info(f"  Averages: {vuln_aggregator.summary_str()}")
            else:
                logger.warning(f"No vulnerability metrics collected for {attack_type}")
        
        # Save explanation metrics CSV (once per detector, attack-independent)
        logger.info(f"\n{'='*60}")
        logger.info(f"Saving explanation metrics for {detector_name}...")
        if len(exp_aggregator) > 0:
            # Ensure directory exists before saving
            os.makedirs(detector_output, exist_ok=True)
            exp_csv = os.path.join(detector_output, "metrics_explanation.csv")
            exp_aggregator.to_csv(exp_csv)
            logger.info(f"✓ Explanation metrics saved to: {exp_csv}")
            logger.info(f"  Samples: {len(exp_aggregator)}")
            logger.info(f"  Averages: {exp_aggregator.summary_str()}")
        else:
            logger.warning(f"No explanation metrics collected for {detector_name}")
        logger.info(f"{'='*60}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("✓ ALL PROCESSING COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Detectors processed: {len(args.detectors)}")
    logger.info(f"Attack types: {len(attacks)}")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nOutput structure:")
    logger.info(f"  [output]/<model>/metrics_explanation.csv")
    logger.info(f"  [output]/<model>/<attack>/metrics_vulnerability.csv")
    logger.info(f"  [output]/<model>/vis/<attack>/<filename>_grid.png")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
