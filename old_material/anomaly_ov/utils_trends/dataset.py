import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from glob import glob
from typing import Tuple, Optional, List, Dict
import random
from utils_trends.image_processing import process_images


def collate_fn(batch):
    """
    Custom collate function that handles:
    - Variable-sized image tensors (different number of views)
    - Optional masks (can be None)
    """
    images = [item['image'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    masks = [item['mask'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    
    return {
        'image': images,  # List of tensors
        'label': labels,
        'mask': masks,    # List of tensors or None
        'image_path': image_paths
    }


class AnomalyFinetuneDataset(Dataset):
    """
    Dataset for anomaly detection finetuning.
    
    Structure expected in data_root:
    - inpainted/: Contains inpainted images (label=1, anomalous)
    - masks/: Contains masks for inpainted images (same filenames)
    - [other folders]/: Real images (label=0, normal)
    
    Args:
        data_root: Root directory containing inpainted/, masks/, and other image folders
        image_processor: Image processor from the model (vision tower)
        image_aspect_ratio: Aspect ratio setting for processing (e.g., "anyres")
        image_grid_pinpoints: Grid pinpoints for anyres processing
        split: 'train', 'test_inpainted', 'test_real', or 'test_mixed'
        train_ratio: Ratio of images to use for training (default: 0.8)
        balance_train: If True, balance training set to have equal inpainted/real samples
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        data_root: str,
        image_processor,
        image_aspect_ratio: str,
        image_grid_pinpoints,
        split: str = 'train',
        train_ratio: float = 0.8,
        balance_train: bool = True,
        seed: int = 42
    ):
        self.data_root = data_root
        self.image_processor = image_processor
        self.image_aspect_ratio = image_aspect_ratio
        self.image_grid_pinpoints = image_grid_pinpoints
        self.split = split
        self.train_ratio = train_ratio
        self.balance_train = balance_train
        self.seed = seed
        
        random.seed(seed)
        
        # Validate data_root structure
        self.inpainted_dir = os.path.join(data_root, 'inpainted')
        self.masks_dir = os.path.join(data_root, 'masks')
        
        if not os.path.isdir(self.inpainted_dir):
            raise ValueError(f"Inpainted directory not found: {self.inpainted_dir}")
        if not os.path.isdir(self.masks_dir):
            print(f"Warning: Masks directory not found: {self.masks_dir}")
        
        # Collect all image paths and labels
        self.samples = self._collect_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{split}' in {data_root}")
        
        print(f"Dataset split '{split}': {len(self.samples)} samples")
        if split == 'train':
            inpainted_count = sum(1 for _, label, _ in self.samples if label == 1)
            real_count = len(self.samples) - inpainted_count
            print(f"  - Inpainted (anomalous): {inpainted_count}")
            print(f"  - Real (normal): {real_count}")
    
    def _collect_samples(self) -> List[Tuple[str, int, Optional[str]]]:
        """
        Collect all samples: (image_path, label, mask_path).
        Returns list of tuples: (image_path, label, mask_path or None)
        """
        inpainted_samples = []
        real_samples = []
        
        # Collect inpainted images (label=1)
        inpainted_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        for pattern in inpainted_patterns:
            for img_path in glob(os.path.join(self.inpainted_dir, '**', pattern), recursive=True):
                if os.path.isfile(img_path):
                    # Find corresponding mask
                    mask_filename = os.path.basename(img_path)
                    mask_path = os.path.join(self.masks_dir, mask_filename)
                    mask_path = mask_path if os.path.isfile(mask_path) else None
                    inpainted_samples.append((img_path, 1, mask_path))
        
        # Collect real images (label=0) from other directories
        for item in os.listdir(self.data_root):
            item_path = os.path.join(self.data_root, item)
            # Skip inpainted and masks directories
            if not os.path.isdir(item_path) or item in ['inpainted', 'masks']:
                continue
            
            # Collect images from this directory
            for pattern in inpainted_patterns:
                for img_path in glob(os.path.join(item_path, '**', pattern), recursive=True):
                    if os.path.isfile(img_path):
                        real_samples.append((img_path, 0, None))
        
        # Shuffle samples
        random.shuffle(inpainted_samples)
        random.shuffle(real_samples)
        
        # Split into train/test based on split parameter
        if self.split == 'train':
            # Training set: take train_ratio of each category
            n_inpainted_train = int(len(inpainted_samples) * self.train_ratio)
            n_real_train = int(len(real_samples) * self.train_ratio)
            
            train_inpainted = inpainted_samples[:n_inpainted_train]
            train_real = real_samples[:n_real_train]
            
            # Balance if requested
            if self.balance_train:
                min_count = min(len(train_inpainted), len(train_real))
                train_inpainted = train_inpainted[:min_count]
                train_real = train_real[:min_count]
            
            samples = train_inpainted + train_real
            random.shuffle(samples)
            return samples
        
        elif self.split == 'test_inpainted':
            # Test set: only inpainted images (remaining after train split)
            n_inpainted_train = int(len(inpainted_samples) * self.train_ratio)
            return inpainted_samples[n_inpainted_train:]
        
        elif self.split == 'test_real':
            # Test set: only real images (remaining after train split)
            n_real_train = int(len(real_samples) * self.train_ratio)
            return real_samples[n_real_train:]
        
        elif self.split == 'test_mixed':
            # Test set: mixed inpainted and real images
            n_inpainted_train = int(len(inpainted_samples) * self.train_ratio)
            n_real_train = int(len(real_samples) * self.train_ratio)
            
            test_samples = (inpainted_samples[n_inpainted_train:] + 
                          real_samples[n_real_train:])
            random.shuffle(test_samples)
            return test_samples
        
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'test_inpainted', 'test_real', or 'test_mixed'")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a dictionary containing:
        - 'image': Processed image tensor [num_views, C, H, W]
        - 'label': Binary label (0=normal, 1=anomalous)
        - 'mask': Processed mask tensor or None [num_views, C, H, W]
        - 'image_path': Path to the original image
        """
        img_path, label, mask_path = self.samples[idx]
        
        # Load and process image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy sample (you might want to handle this differently)
            return self.__getitem__((idx + 1) % len(self))
        
        # Process image using the provided function
        image_tensor = process_images(
            [image], 
            self.image_processor, 
            self.image_aspect_ratio, 
            self.image_grid_pinpoints
        )
        
        # Handle stacked vs single tensor output
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.ndim == 4:  # [batch, num_views, C, H, W]
                image_tensor = image_tensor[0]  # Remove batch dimension
        
        # Load and process mask if available
        mask_tensor = None
        if mask_path is not None and os.path.isfile(mask_path):
            try:
                mask = Image.open(mask_path).convert('L')  # Load as grayscale
                mask_tensor = process_images(
                    [mask], 
                    self.image_processor, 
                    self.image_aspect_ratio, 
                    self.image_grid_pinpoints
                )
                if isinstance(mask_tensor, torch.Tensor):
                    if mask_tensor.ndim == 4:
                        mask_tensor = mask_tensor[0]
            except Exception as e:
                print(f"Error loading mask {mask_path}: {e}")
                mask_tensor = None
        
        return {
            'image': image_tensor,
            'label': torch.tensor(label, dtype=torch.long),
            'mask': mask_tensor,
            'image_path': img_path
        }


def create_dataloaders(
    data_root: str,
    image_processor,
    image_aspect_ratio: str,
    image_grid_pinpoints,
    batch_size: int = 8,
    train_ratio: float = 0.8,
    balance_train: bool = True,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Factory function to create all dataloaders at once.
    
    Args:
        data_root: Root directory containing the dataset
        image_processor: Image processor from the model
        image_aspect_ratio: Aspect ratio setting (e.g., "anyres")
        image_grid_pinpoints: Grid pinpoints for processing
        batch_size: Batch size for dataloaders
        train_ratio: Ratio of data to use for training
        balance_train: Whether to balance the training set
        num_workers: Number of workers for data loading
        seed: Random seed
    
    Returns:
        Tuple of (train_loader, test_inpainted_loader, test_real_loader, test_mixed_loader)
    """
    
    # Create datasets
    train_dataset = AnomalyFinetuneDataset(
        data_root=data_root,
        image_processor=image_processor,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        split='train',
        train_ratio=train_ratio,
        balance_train=balance_train,
        seed=seed
    )
    
    test_inpainted_dataset = AnomalyFinetuneDataset(
        data_root=data_root,
        image_processor=image_processor,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        split='test_inpainted',
        train_ratio=train_ratio,
        seed=seed
    )
    
    test_real_dataset = AnomalyFinetuneDataset(
        data_root=data_root,
        image_processor=image_processor,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        split='test_real',
        train_ratio=train_ratio,
        seed=seed
    )
    
    test_mixed_dataset = AnomalyFinetuneDataset(
        data_root=data_root,
        image_processor=image_processor,
        image_aspect_ratio=image_aspect_ratio,
        image_grid_pinpoints=image_grid_pinpoints,
        split='test_mixed',
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_inpainted_loader = DataLoader(
        test_inpainted_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_real_loader = DataLoader(
        test_real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    test_mixed_loader = DataLoader(
        test_mixed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, test_inpainted_loader, test_real_loader, test_mixed_loader


if __name__ == "__main__":
    """
    Example usage of the dataset.
    """
    # This example shows how to use the dataset with a real model
    import json
    from custom_anomaly_detector import OVAnomalyDetector
    
    # Load config
    with open('../config.json') as f:
        config = json.load(f)
    
    # Load model to get image_processor
    model, image_processor = OVAnomalyDetector.load_from_checkpoint(
        "../zs_checkpoint.pt", 
        device="cpu"
    )
    
    # Create dataloaders
    data_root = "../finetune_dataset"  # or "../trends_cv_data"
    
    train_loader, test_inpainted_loader, test_real_loader, test_mixed_loader = create_dataloaders(
        data_root=data_root,
        image_processor=image_processor,
        image_aspect_ratio=config["image_aspect_ratio"],
        image_grid_pinpoints=config["image_grid_pinpoints"],
        batch_size=4,
        train_ratio=0.8,
        balance_train=True,
        num_workers=0,  # Set to 0 for debugging
        seed=42
    )
    
    # Test the train loader
    print("\n=== Testing Train Loader ===")
    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        labels = batch['label']
        masks = batch['mask']
        paths = batch['image_path']
        
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Number of samples: {len(images)}")
        print(f"  Labels: {labels}")
        print(f"  Image shapes: {[img.shape for img in images]}")
        print(f"  Mask availability: {[mask is not None for mask in masks]}")
        
        if batch_idx >= 2:  # Show first 3 batches
            break
    
    # Test the test loaders
    print("\n=== Testing Test Loaders ===")
    print(f"Test Inpainted: {len(test_inpainted_loader.dataset)} samples")
    print(f"Test Real: {len(test_real_loader.dataset)} samples")
    print(f"Test Mixed: {len(test_mixed_loader.dataset)} samples")
