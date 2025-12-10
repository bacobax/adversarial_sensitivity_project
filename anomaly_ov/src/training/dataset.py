"""
Dataset classes for Anomaly OV fine-tuning.
"""

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import random

# Setup paths for imports
_current_file = os.path.abspath(__file__)
_training_dir = os.path.dirname(_current_file)
_src_dir = os.path.dirname(_training_dir)
_project_root = os.path.dirname(_src_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.image_processing import process_anyres_image


class ColorAugmentation:
    """
    Color augmentation to reduce bias towards bright/white regions.
    
    Applies various color transformations to help the model focus on 
    structural anomalies rather than brightness patterns.
    """
    
    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.6, 1.4),
        contrast_range: Tuple[float, float] = (0.6, 1.4),
        saturation_range: Tuple[float, float] = (0.6, 1.4),
        hue_shift_range: Tuple[float, float] = (-0.1, 0.1),
        gamma_range: Tuple[float, float] = (0.7, 1.5),
        invert_prob: float = 0.1,
        grayscale_prob: float = 0.05,
        channel_shuffle_prob: float = 0.1,
        solarize_prob: float = 0.05,
        equalize_prob: float = 0.1,
    ):
        """
        Args:
            brightness_range: Range for brightness adjustment (1.0 = no change)
            contrast_range: Range for contrast adjustment (1.0 = no change)
            saturation_range: Range for saturation adjustment (1.0 = no change)
            hue_shift_range: Range for hue shift (-0.5 to 0.5)
            gamma_range: Range for gamma correction (1.0 = no change)
            invert_prob: Probability of inverting colors
            grayscale_prob: Probability of converting to grayscale
            channel_shuffle_prob: Probability of shuffling RGB channels
            solarize_prob: Probability of solarization
            equalize_prob: Probability of histogram equalization
        """
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_shift_range = hue_shift_range
        self.gamma_range = gamma_range
        self.invert_prob = invert_prob
        self.grayscale_prob = grayscale_prob
        self.channel_shuffle_prob = channel_shuffle_prob
        self.solarize_prob = solarize_prob
        self.equalize_prob = equalize_prob
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """Apply random color augmentations to the image."""
        
        # Random brightness adjustment
        if random.random() < 0.8:
            factor = random.uniform(*self.brightness_range)
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(factor)
        
        # Random contrast adjustment
        if random.random() < 0.8:
            factor = random.uniform(*self.contrast_range)
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(factor)
        
        # Random saturation adjustment
        if random.random() < 0.7:
            factor = random.uniform(*self.saturation_range)
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(factor)
        
        # Gamma correction (helps with bright region bias)
        if random.random() < 0.5:
            gamma = random.uniform(*self.gamma_range)
            image = self._apply_gamma(image, gamma)
        
        # Histogram equalization (reduces brightness dominance)
        if random.random() < self.equalize_prob:
            image = self._equalize(image)
        
        # Solarization (inverts pixels above threshold)
        if random.random() < self.solarize_prob:
            threshold = random.randint(128, 230)
            image = self._solarize(image, threshold)
        
        # Color inversion
        if random.random() < self.invert_prob:
            image = self._invert(image)
        
        # Grayscale conversion (forces model to focus on structure)
        if random.random() < self.grayscale_prob:
            image = self._to_grayscale(image)
        
        # Channel shuffle (prevents channel-specific biases)
        if random.random() < self.channel_shuffle_prob:
            image = self._shuffle_channels(image)
        
        return image
    
    def _apply_gamma(self, image: Image.Image, gamma: float) -> Image.Image:
        """Apply gamma correction."""
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def _equalize(self, image: Image.Image) -> Image.Image:
        """Apply histogram equalization to each channel."""
        from PIL import ImageOps
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Split channels, equalize each, merge back
        channels = image.split()
        eq_channels = [ImageOps.equalize(c) for c in channels]
        return Image.merge('RGB', eq_channels)
    
    def _solarize(self, image: Image.Image, threshold: int) -> Image.Image:
        """Solarize pixels above threshold."""
        from PIL import ImageOps
        return ImageOps.solarize(image, threshold)
    
    def _invert(self, image: Image.Image) -> Image.Image:
        """Invert image colors."""
        from PIL import ImageOps
        return ImageOps.invert(image)
    
    def _to_grayscale(self, image: Image.Image) -> Image.Image:
        """Convert to grayscale and back to RGB."""
        gray = image.convert('L')
        return gray.convert('RGB')
    
    def _shuffle_channels(self, image: Image.Image) -> Image.Image:
        """Randomly shuffle RGB channels."""
        channels = list(image.split())
        random.shuffle(channels)
        return Image.merge('RGB', channels)


class GeometricAugmentation:
    """
    Geometric augmentation that preserves mask alignment.
    """
    
    def __init__(
        self,
        horizontal_flip_prob: float = 0.5,
        vertical_flip_prob: float = 0.1,
        rotation_prob: float = 0.3,
        rotation_range: Tuple[int, int] = (-15, 15),
    ):
        self.horizontal_flip_prob = horizontal_flip_prob
        self.vertical_flip_prob = vertical_flip_prob
        self.rotation_prob = rotation_prob
        self.rotation_range = rotation_range
    
    def __call__(self, image: Image.Image, mask: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Apply geometric augmentations to both image and mask."""
        
        # Horizontal flip
        if random.random() < self.horizontal_flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = np.fliplr(mask).copy()
        
        # Vertical flip
        if random.random() < self.vertical_flip_prob:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = np.flipud(mask).copy()
        
        # Rotation
        if random.random() < self.rotation_prob:
            angle = random.randint(*self.rotation_range)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))
            # Rotate mask using scipy or manual rotation
            mask = self._rotate_mask(mask, angle)
        
        return image, mask
    
    def _rotate_mask(self, mask: np.ndarray, angle: float) -> np.ndarray:
        """Rotate mask by given angle."""
        from PIL import Image as PILImage
        mask_img = PILImage.fromarray((mask * 255).astype(np.uint8))
        mask_img = mask_img.rotate(angle, resample=PILImage.BILINEAR, fillcolor=0)
        return np.array(mask_img) / 255.0


class AnomalyDataset(Dataset):
    """
    Dataset for anomaly detection with real images, inpainted images, and masks.
    Uses anyres preprocessing to handle variable resolutions and aspect ratios.
    
    Includes data augmentation to reduce bias towards bright regions.
    
    Supports pre-split dataset structure: train/eval/test folders.
    
    Returns:
        - patches: Multi-patch tensor [num_patches+1, 3, 384, 384]
        - label: Binary label (0=normal, 1=anomaly)
        - num_patches: Number of patches for this image
        - mask: Binary mask indicating anomaly regions
    """
    
    def __init__(
        self, 
        data_root: str, 
        image_processor, 
        image_grid_pinpoints: List,
        split: str = "train", 
        max_pairs: Optional[int] = 1000,
        image_size: int = 384,
        mask_size: int = 27,
        augment: bool = True,
        augment_config: Optional[Dict] = None
    ):
        """
        Args:
            data_root: Root directory containing train/eval/test folders
            image_processor: SigLip image processor
            image_grid_pinpoints: Grid pinpoints for anyres processing
            split: One of 'train', 'eval', or 'test'
            max_pairs: Maximum number of image pairs to use (each pair = 2 samples: real + inpainted)
            image_size: Size of input images (default: 384)
            mask_size: Size of output mask (default: 27 for SigLip 384)
            augment: Whether to apply data augmentation (default: True for train, False for eval/test)
            augment_config: Optional dict to override augmentation parameters
        """
        self.data_root = Path(data_root)
        self.image_processor = image_processor
        self.image_grid_pinpoints = image_grid_pinpoints
        self.split = split
        self.image_size = image_size
        self.mask_size = mask_size
        
        # Only augment training data by default
        self.augment = augment and (split == "train")
        
        # Initialize augmentations
        if self.augment:
            aug_config = augment_config or {}
            self.color_aug = ColorAugmentation(
                brightness_range=aug_config.get('brightness_range', (0.5, 1.5)),
                contrast_range=aug_config.get('contrast_range', (0.5, 1.5)),
                saturation_range=aug_config.get('saturation_range', (0.5, 1.5)),
                gamma_range=aug_config.get('gamma_range', (0.6, 1.6)),
                invert_prob=aug_config.get('invert_prob', 0.1),
                grayscale_prob=aug_config.get('grayscale_prob', 0.05),
                channel_shuffle_prob=aug_config.get('channel_shuffle_prob', 0.1),
                solarize_prob=aug_config.get('solarize_prob', 0.05),
                equalize_prob=aug_config.get('equalize_prob', 0.15),
            )
            self.geo_aug = GeometricAugmentation(
                horizontal_flip_prob=aug_config.get('horizontal_flip_prob', 0.5),
                vertical_flip_prob=aug_config.get('vertical_flip_prob', 0.1),
                rotation_prob=aug_config.get('rotation_prob', 0.2),
            )
            print(f"  Data augmentation: ENABLED (color + geometric)")
        else:
            self.color_aug = None
            self.geo_aug = None
            if split == "train":
                print(f"  Data augmentation: DISABLED")
        
        # Set up paths for this split
        split_dir = self.data_root / split
        self.inpainted_dir = split_dir / "inpainted"
        self.real_dir = split_dir / "COCO_real"
        self.masks_dir = split_dir / "masks"
        
        # Verify directories exist
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Get all inpainted images (these determine the dataset size)
        all_inpainted_files = sorted(
            list(self.inpainted_dir.glob("*.png")) + 
            list(self.inpainted_dir.glob("*.jpg"))
        )
        
        if len(all_inpainted_files) == 0:
            raise ValueError(f"No images found in {self.inpainted_dir}")
        
        # Limit dataset size if max_pairs is specified
        if max_pairs is not None and max_pairs > 0:
            self.inpainted_files = all_inpainted_files[:max_pairs]
            print(f"{split} dataset: {len(self.inpainted_files)} image pairs (limited from {len(all_inpainted_files)})")
        else:
            self.inpainted_files = all_inpainted_files
            print(f"{split} dataset: {len(self.inpainted_files)} image pairs")
    
    def __len__(self) -> int:
        return len(self.inpainted_files) * 2  # Each pair provides 2 samples (real + inpainted)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # Determine if this is a real or inpainted sample
        pair_idx = idx // 2
        is_inpainted = idx % 2 == 1
        
        inpainted_file = self.inpainted_files[pair_idx]
        filename = inpainted_file.name
        
        # Load images
        if is_inpainted:
            # Load inpainted image (anomaly)
            image = Image.open(inpainted_file).convert("RGB")
            label = 1.0  # Anomaly present
            
            # Load mask
            mask_file = self.masks_dir / filename
            if mask_file.exists():
                mask = Image.open(mask_file).convert("L")
                mask = np.array(mask) / 255.0  # Normalize to [0, 1]
            else:
                mask = np.zeros((self.image_size, self.image_size))
        else:
            # Load real image (no anomaly)
            real_file = self.real_dir / filename
            if not real_file.exists():
                # Fallback: if exact match doesn't exist, use first available real image
                real_files = list(self.real_dir.glob("*.png")) + list(self.real_dir.glob("*.jpg"))
                if real_files:
                    real_file = real_files[pair_idx % len(real_files)]
                else:
                    raise ValueError(f"No real images found in {self.real_dir}")
            
            image = Image.open(real_file).convert("RGB")
            label = 0.0  # No anomaly
            mask = np.zeros((self.image_size, self.image_size))
        
        # Apply augmentations (only for training)
        if self.augment:
            # Apply geometric augmentation first (affects both image and mask)
            if self.geo_aug is not None:
                image, mask = self.geo_aug(image, mask)
            
            # Apply color augmentation (only affects image)
            if self.color_aug is not None:
                image = self.color_aug(image)
        
        # Process image using anyres strategy
        patches = process_anyres_image(image, self.image_processor, self.image_grid_pinpoints)
        num_patches = patches.shape[0]
        
        # Resize mask to match anomaly map output size
        mask_tensor = torch.from_numpy(mask).float()
        if mask_tensor.shape[0] != self.mask_size or mask_tensor.shape[1] != self.mask_size:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor.unsqueeze(0).unsqueeze(0), 
                size=(self.mask_size, self.mask_size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        
        return {
            "patches": patches,  # [num_patches, 3, 384, 384]
            "label": torch.tensor([label], dtype=torch.float32),
            "num_patches": num_patches,
            "mask": mask_tensor,
            "filename": filename
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function to handle variable number of patches per image.
    Returns batched data where images are processed individually.
    """
    return {
        "patches": [item["patches"] for item in batch],  # List of tensors
        "label": torch.stack([item["label"] for item in batch]),
        "num_patches": [item["num_patches"] for item in batch],
        "mask": torch.stack([item["mask"] for item in batch]),
        "filename": [item["filename"] for item in batch]
    }


def create_dataloaders(config, image_processor):
    """
    Create train, eval, and test dataloaders.
    
    Args:
        config: Configuration object
        image_processor: Image processor from vision tower
    
    Returns:
        Tuple of (train_loader, eval_loader, test_loader, train_dataset, eval_dataset, test_dataset)
    """
    print("Creating datasets...")
    
    # Get augmentation config from config object if available
    augment_config = getattr(config, 'augment_config', None)
    use_augmentation = getattr(config, 'use_augmentation', True)
    
    train_dataset = AnomalyDataset(
        data_root=config.data_root,
        image_processor=image_processor,
        image_grid_pinpoints=config.image_grid_pinpoints,
        split="train",
        max_pairs=config.train_max_pairs,
        image_size=config.image_size,
        augment=use_augmentation,
        augment_config=augment_config,
    )
    
    eval_dataset = AnomalyDataset(
        data_root=config.data_root,
        image_processor=image_processor,
        image_grid_pinpoints=config.image_grid_pinpoints,
        split="eval",
        max_pairs=config.eval_max_pairs,
        image_size=config.image_size,
        augment=False,  # Never augment eval
    )
    
    test_dataset = AnomalyDataset(
        data_root=config.data_root,
        image_processor=image_processor,
        image_grid_pinpoints=config.image_grid_pinpoints,
        split="test",
        max_pairs=config.test_max_pairs,
        image_size=config.image_size,
        augment=False,  # Never augment test
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn
    )
    
    print(f"\nDataset Summary:")
    print(f"  Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"  Eval:  {len(eval_dataset)} samples ({len(eval_loader)} batches)")
    print(f"  Test:  {len(test_dataset)} samples ({len(test_loader)} batches)")
    
    return train_loader, eval_loader, test_loader, train_dataset, eval_dataset, test_dataset
