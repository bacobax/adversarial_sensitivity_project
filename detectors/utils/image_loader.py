from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.consts import *
from utils.logging import logger
from utils.sample_paths import SamplePaths


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


def load_image(path: str) -> Image.Image:
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
    return img


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
                mask[y:y + h, x:x + w] = 1.0
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
                mask[y:y + h, x:x + w] = 1.0
        
        return mask
    
    else:
        raise ValueError(f"Unknown bbox file format: {bbox_path}")


def collect_sample_paths(
    root_dataset: str,
    image_types: List[str],
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
            f"See log for details.",
        )
    
    logger.info(f"Collected {len(samples)} sample paths")
    return samples
