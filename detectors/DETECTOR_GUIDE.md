# Detector Framework Guide

This guide explains how to use the detector framework for vulnerability analysis and visualization, and how to create new detectors or add explainability/adversarial features to existing ones.

## Table of Contents

1. [Overview](#overview)
2. [Using `visualize_vulnerability.py`](#using-visualize_vulnerabilitypy)
3. [Creating a New Detector](#creating-a-new-detector)
4. [Adding Explainability & Vulnerability Features](#adding-explainability--vulnerability-features)
5. [Understanding the Framework](#understanding-the-framework)

---

## Overview

The detector framework provides a unified interface for:
- **Prediction**: Classify images as real or fake
- **Explainability**: Generate saliency/explanation maps
- **Vulnerability Analysis**: Analyze model robustness to adversarial attacks
- **Visualization**: Create comprehensive vulnerability visualizations

### Detector Capabilities

Each detector declares which features it supports:

```python
class MyDetector(BaseDetector):
    name = 'MyDetector'
    
    supports_explainability = True   # Can generate explanation maps
    supports_vulnerability = True    # Can perform vulnerability analysis
    supports_adversarial = True      # Can generate adversarial images
```

---

## Using `visualize_vulnerability.py`

### Purpose

This script generates vulnerability visualizations by analyzing how model explanations change under adversarial attacks. It supports batch processing of datasets and automatic adversarial image generation.

### Dataset Structure

The script expects images organized as follows:

```
data_folder/
├── b-free/                    # Benign (non-attacked) images
│   ├── real/                  # Original real images
│   ├── samecat/               # Inpainted images (same category)
│   ├── diffcat/               # Inpainted images (different category)
│   ├── mask/                  # Masks for samecat inpainting (grayscale)
│   └── bbox/                  # Bounding boxes for diffcat (grayscale)
└── adv_attacks/               # Adversarial attacked images (auto-generated if missing)
    └── [detector_name]/       # e.g., AnomalyOV, R50_nodown
        └── [attack_type]/     # e.g., pgd, fgsm, deepfool
            ├── real/          # Attacked real images
            ├── samecat/       # Attacked inpainted images (samecat)
            └── diffcat/       # Attacked inpainted images (diffcat)
```

### Basic Usage

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets/my_dataset \
    --output_folder ./vulnerability_viz \
    --detector AnomalyOV \
    --attack_type pgd
```

### Command-Line Arguments

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--data_folder` | str | **Required**. Parent data folder containing `b-free/` and `adv_attacks/` subfolders. Must have the structure shown above. |
| `--output_folder` / `-o` | str | **Required**. Where to save vulnerability visualizations. Will be created if it doesn't exist. |
| `--attack_type` / `-a` | str | **Required**. Adversarial attack type to analyze (e.g., `pgd`, `fgsm`, `deepfool`). Must match a folder name in `adv_attacks/[detector_name]/`. |

#### Optional Arguments - Model & Device

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detector` / `-d` | str | `AnomalyOV` | Detector name(s) to use. Can specify multiple: `AnomalyOV R50_nodown`. Available: AnomalyOV, R50_nodown, R50_TF, NPR, CLIP-D, P2G, WaveRep. |
| `--weights` / `-w` | str | None | Path to model weights. For single detector: `/path/to/weights.pt`. For multiple: `detector1:/path1,detector2:/path2`. If not specified, uses default weights from `models/[detector]/weights/`. |
| `--device` | str | auto-detect | Device to use for computation. Options: `cpu`, `cuda:0`, `cuda:1`, `mps` (Apple Silicon). Defaults to best available (GPU > CPU). |

#### Optional Arguments - Image Processing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` / `-l` | int | `0` | Maximum number of images to process per image type. Use `0` to process all images. Useful for quick testing: `--limit 5` processes only first 5 samples. |
| `--image_types` | str (list) | `samecat diffcat` | Which image types to process. Options: `real`, `samecat`, `diffcat`. Can specify multiple: `--image_types samecat diffcat`. Note: `real` has no ground truth mask for metrics. |
| `--visualize_limit` | int | `5` | Number of samples to create visualizations for. Use `0` to visualize all. Set to `-1` or `0` for very large datasets to just compute metrics without creating PNG files. |
| `--skip_visualization` | flag | False | Skip creating visualization PNG files entirely. Useful when you only want to compute metrics via `--save_csv`. Saves time and disk space. |

#### Optional Arguments - Visualization Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dpi` | int | `150` | DPI (dots per inch) for saved visualization images. Higher DPI = larger file size but higher quality. Typical values: `150` (normal), `300` (high quality). |
| `--overlay_alpha` | float | `0.4` | Transparency of map overlays on images (range: 0.0-1.0). Lower values make overlays more transparent, higher values make them more opaque. Default `0.4` works well for most cases. |

#### Optional Arguments - Metrics & Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| (none - CSV is auto-generated) | - | - | **CSV metrics are automatically saved** to `output_folder/vulnerability.csv` containing: filename, image_type, detector_name, attack_type, predictions, IoU scores, ROC-AUC scores, and more. |

### Detailed Argument Examples

#### Example 1: Basic Usage (Single Detector)

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets/my_dataset \
    --output_folder ./viz_output \
    --attack_type pgd \
    --detector AnomalyOV
```

- Processes all images in `./datasets/my_dataset`
- Uses AnomalyOV detector with default weights
- Creates visualizations for first 5 samples
- Saves to `./viz_output/`

#### Example 2: Multiple Detectors with Custom Weights

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets \
    --output_folder ./analysis \
    --attack_type fgsm \
    --detector AnomalyOV R50_nodown \
    --weights "AnomalyOV:models/anomaly_ov/weights/checkpoint_epoch_3.pt,R50_nodown:models/R50_nodown/weights/best.pt" \
    --device [DEVICE] \
    --limit 2
```

- Compares vulnerability across two detectors
- Each detector uses specified custom weights
- Results organized as: `analysis/AnomalyOV/` and `analysis/R50_nodown/`

#### Example 3: Large Dataset with Metrics Only

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets/large_dataset \
    --output_folder ./metrics_output \
    --attack_type fgsm \
    --detector AnomalyOV \
    --limit 1000 \
    --visualize_limit 0 \
    --skip_visualization
```

- Process 1000 images but skip PNG visualization (fast)
- Only computes and saves metrics to CSV
- Ideal for statistical analysis without disk space concerns

#### Example 4: Testing with Small Sample

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets/test \
    --output_folder ./quick_test \
    --attack_type deepfool \
    --detector R50_nodown \
    --limit 5 \
    --visualize_limit 5 \
    --image_types samecat \
    --dpi 100 \
    --device cuda:0
```

- Quick test: only 5 images
- Single image type (samecat) to speed up processing
- GPU acceleration with `cuda:0`
- Lower DPI (100) to reduce file sizes

#### Example 5: Comparing Different Attacks

```bash
# First run with PGD attack
python visualize_vulnerability.py \
    --data_folder ./datasets/data \
    --output_folder ./pgd_analysis \
    --attack_type pgd \
    --detector AnomalyOV \
    --limit 20

# Then run with FGSM attack
python visualize_vulnerability.py \
    --data_folder ./datasets/data \
    --output_folder ./fgsm_analysis \
    --attack_type fgsm \
    --detector AnomalyOV \
    --limit 20
```

- Compares how detector reacts to different attacks
- Same images processed with different attacks
- Results can be compared: `pgd_analysis/` vs `fgsm_analysis/`

### Output Structure

```
output_folder/
├── [image_id]_[image_type].png      # Vulnerability visualizations
├── vulnerability.csv                 # Metrics (if --save_csv enabled)
└── adversarial_images.csv            # List of generated adversarial images
```

### Example: Process 10 Images with Metrics

```bash
python visualize_vulnerability.py \
    --data_folder ./datasets/coco \
    --output_folder ./viz_output \
    --detector AnomalyOV \
    --attack_type pgd \
    --limit 10 \
    --save_csv \
    --device cuda:0
```

### Visualization Output

Each visualization shows:
- **Column 1 (Real)**: Original real image, explanation maps, and vulnerability
- **Column 2 (Samecat)**: Inpainted image (same category), analysis
- **Column 3 (Diffcat)**: Inpainted image (different category), analysis

Each row displays:
- **R1**: Original image with prediction
- **R2**: Image + explanation map overlay
- **R3**: Attacked image + explanation map overlay
- **R4**: Image + vulnerability map (how explanation changed)
- **R5**: Ground truth mask/bbox

### Metrics Output (CSV)

**A CSV file is automatically generated** at `output_folder/vulnerability.csv` with metrics for each processed image. This happens automatically without needing extra arguments.

#### CSV Columns Explained

| Column | Type | Description | Interpretation |
|--------|------|-------------|-----------------|
| `filename` | str | Base filename of the image | Identifier for the sample |
| `image_type` | str | Type: `real`, `samecat`, or `diffcat` | What category of image was processed |
| `detector_name` | str | Detector used (e.g., AnomalyOV) | Which detector generated the maps |
| `attack_type` | str | Attack type (e.g., pgd, fgsm) | What adversarial attack was applied |
| `prediction_benign` | float | Model confidence on clean image [0, 1] | Higher = more likely fake (detector-dependent) |
| `prediction_attacked` | float | Model confidence on adversarial image [0, 1] | How attack changes prediction |
| `prediction_changed` | bool | True if attack flipped prediction | Whether attack was successful in fooling detector |
| `explanation_map` | str | Path or indicator for explanation map | Source of explanation (e.g., GradCAM) |
| `iou_anomaly_mask` | float | IoU between explanation map and ground truth [0, 1] | How well explanation matches ground truth. 1.0 = perfect match, 0.0 = no overlap |
| `iou_vulnerability_mask` | float | IoU between vulnerability map and ground truth [0, 1] | How well vulnerability map highlights important regions |
| `auc_roc_anomaly_mask` | float | Area under ROC curve for explanation map [0, 1] | Classification performance. 0.5 = random, 1.0 = perfect |
| `auc_roc_vulnerability_mask` | float | Area under ROC curve for vulnerability map [0, 1] | How well vulnerability predicts anomalous regions |

#### Understanding the Metrics

**Explanation Map Metrics (vs Ground Truth Mask/BBox):**
- **IoU (Intersection over Union)**: Measures spatial overlap. Range [0, 1]:
  - 1.0 = Explanation perfectly matches ground truth
  - 0.5 = 50% overlap
  - 0.0 = No overlap
  - **Better for**: Localization accuracy

- **ROC-AUC**: Measures ranking quality. Range [0.5, 1.0]:
  - 1.0 = Perfect ranking (high saliency on anomaly, low elsewhere)
  - 0.5 = Random (no better than guessing)
  - **Better for**: Overall discriminative power

**Vulnerability Map Metrics:**
- Shows how explanation changes under attack
- High metrics indicate detector explanations are unstable
- Can reveal whether attention mechanisms are robust
- Used to assess adversarial vulnerability

#### Example Interpretation

```
Sample: COCO_0001_samecat.jpg
├─ prediction_benign: 0.87        # Model predicts "fake" (87% confidence)
├─ prediction_attacked: 0.42      # Attack reduces confidence to 42%
├─ prediction_changed: True       # Attack flipped prediction (fake→real)
├─ iou_anomaly_mask: 0.65         # Explanation map 65% matches ground truth
├─ iou_vulnerability_mask: 0.58   # Vulnerability map 58% matches
├─ auc_roc_anomaly_mask: 0.78     # Good ranking of anomalous regions
└─ auc_roc_vulnerability_mask: 0.72 # Shows unstable explanation under attack
```

**Interpretation:**
- Model makes correct prediction initially
- Attack successfully fools the model
- Explanation maps align reasonably well with ground truth (65% IoU)
- But explanation becomes less reliable under attack (vulnerability detected)

#### Aggregated Statistics

The CSV allows computing summary statistics:

```python
import pandas as pd

df = pd.read_csv('vulnerability.csv')

# Average metrics per detector
print(df.groupby('detector_name')[['iou_anomaly_mask', 'auc_roc_anomaly_mask']].mean())

# How many attacks succeeded?
print(f"Attack success rate: {df['prediction_changed'].mean():.2%}")

# Which image types are more vulnerable?
print(df.groupby('image_type')['iou_vulnerability_mask'].mean())
```

---

## Using `detect.py` for Batch Predictions

### Purpose

This script performs batch inference across multiple folders using multiple detectors. It's useful for:
- Processing large datasets quickly
- Comparing predictions across multiple detectors
- Using custom checkpoint paths for each detector
- Optional adversarial evaluation (WaveRep only)

### Output Structure

```
output_path (specified by --output)
├── results.csv                    # Main output with predictions
```

CSV columns: `folder`, `image`, `model`, `confidence`, `prediction`

### Command-Line Arguments

#### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--folders` | str (list) | **Required**. Paths to image folders to process. Can specify multiple: `folder1 folder2 folder3`. Images are recursively collected from each folder. |

#### Optional Arguments - Model Selection & Weights

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--detectors` | str (list) | `all` | Which detectors to use. Options: `all` (use all available), or specify names like `AnomalyOV R50_nodown`. Available: AnomalyOV, R50_nodown, R50_TF, NPR, CLIP-D, P2G, WaveRep. |
| `--weights` | str (list) | [] | Custom model weights. Format: `model_name:path_to_weights`. Examples:<br/>`--weights AnomalyOV:/path/to/weights.pt`<br/>`--weights "AnomalyOV:/path1.pt" "R50_nodown:/path2.pt"`<br/>If not specified, uses default weights from `models/[model]/weights/best.pt`. |

#### Optional Arguments - Processing

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--limit` | int | `0` | Maximum number of images per folder to process. Use `0` to process all images. Sorted alphabetically. Example: `--limit 100` processes only first 100 images per folder. |
| `--device` | str | auto-detect | Device for computation: `cpu`, `cuda:0`, `cuda:1`, `mps`. Defaults to best available. |
| `--batch_size` | int | `16` | Number of images to process in parallel (if detector supports batch). Higher values = faster but more memory. Adjust based on GPU memory: 8-16 for modest GPUs, 32-64 for high-end. |

#### Optional Arguments - Output

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output` | str | `out/results.csv` | Path to save results CSV file. Parent directory is created if needed. |

#### Optional Arguments - Adversarial Attacks (WaveRep only)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--attack` | str | '' (none) | Adversarial attack type. **Only supported for WaveRep detector.** Options: `fgsm`, `pgd`, `deepfool`. If specified, generates adversarial images and evaluates detector on them. |
| `--attack_label` | int | `1` | True label for attack target. `0` = real image, `1` = fake image. The attack will try to flip the prediction. |
| `--attack_args` | str (list) | [] | Extra attack parameters as `key=value` pairs. Examples:<br/>`eps=8` - perturbation budget (8/255)<br/>`alpha=2` - step size (2/255 for PGD)<br/>`steps=10` - number of steps (for PGD)<br/>Usage: `--attack_args eps=8 alpha=2 steps=10` |
| `--attack_batch` | int | `2` | Batch size when generating adversarial images. Lower if you run out of memory. |

### Usage Examples

#### Example 1: Simple Batch Prediction

```bash
python detect.py \
    --folders ./datasets/folder1 ./datasets/folder2 \
    --detectors all \
    --output ./results.csv
```

- Processes all images in both folders
- Uses all available detectors with default weights
- Saves predictions to `results.csv`

#### Example 2: Specific Detectors with Custom Weights

```bash
python detect.py \
    --folders ./datasets/test_images \
    --detectors AnomalyOV R50_nodown \
    --weights "AnomalyOV:/path/to/anomaly.pt" "R50_nodown:/path/to/r50.pt" \
    --output ./predictions.csv \
    --device cuda:0
```

- Only uses AnomalyOV and R50_nodown
- Each with custom weights
- GPU acceleration with CUDA

#### Example 3: Large Dataset with Sampling

```bash
python detect.py \
    --folders ./datasets/large \
    --detectors AnomalyOV \
    --limit 1000 \
    --batch_size 64 \
    --output ./sampled_results.csv
```

- Processes only first 1000 images per folder (quick sampling)
- Large batch size (64) for faster processing
- Good for quick statistics on large datasets

#### Example 4: Adversarial Evaluation (WaveRep Only)

```bash
python detect.py \
    --folders ./datasets/images \
    --detectors WaveRep \
    --attack pgd \
    --attack_args eps=8 alpha=2 steps=10 \
    --attack_label 1 \
    --attack_batch 4 \
    --output ./adversarial_results.csv
```

- Generates PGD adversarial examples
- Evaluates WaveRep on both clean and adversarial images
- Results show impact of adversarial perturbations

---

## Creating a New Detector

### Step 1: Create Detector File

Create a new file in `models/[YourDetectorName]/detector.py`:

```python
"""
YourDetector - Brief description.

This detector uses [method/model] for deepfake detection.
"""

import os
from typing import Optional, Tuple
import torch
from PIL import Image
import numpy as np

from support.base_detector import BaseDetector
from support.detect_utils import load_image


class YourDetectorName(BaseDetector):
    """
    YourDetector implementing the BaseDetector interface.
    
    Brief description of the detector approach.
    """
    
    name = 'YourDetectorName'
    
    # Declare which features are supported
    supports_explainability = False  # Set to True if implementing maps
    supports_vulnerability = False   # Set to True if implementing vulnerability analysis
    supports_adversarial = False     # Set to True if implementing adversarial generation
    
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__(device)
        self.model = None
    
    def load(self, model_id: Optional[str] = None) -> None:
        """
        Load model weights from disk.
        
        Args:
            model_id: Path to weights file or None for default
        """
        # Load your model here
        weights = model_id or os.path.join(
            os.path.dirname(__file__), 'weights', 'best.pt'
        )
        if not os.path.exists(weights):
            raise FileNotFoundError(f"Weights not found: {weights}")
        
        # Load model
        self.model = torch.load(weights, map_location=self.device)
        self.model.eval()
        print(f"Model loaded from {weights}")
    
    def predict(self, image_tensor: torch.Tensor, image_path: str) -> float:
        """
        Predict whether image is fake.
        
        Args:
            image_tensor: Preprocessed image tensor [1, C, H, W]
            image_path: Path to original image (for reference)
            
        Returns:
            float: Confidence in [0, 1], higher = more likely fake
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        with torch.no_grad():
            output = self.model(image_tensor)
        
        # Convert to confidence in [0, 1]
        confidence = torch.sigmoid(output).item()
        return float(confidence)
```

### Step 2: Register Detector

Update `detect.py` to register your detector in the `mapping` dictionary:

```python
mapping = {
    # ... existing detectors ...
    'YourDetectorName': ('YourDetectorName', os.path.join('models', 'YourDetectorName', 'detector.py')),
}
```

### Step 3: Test the Detector

```python
from support.base_detector import BaseDetector
from models.YourDetectorName.detector import YourDetectorName

# Create and load detector
detector = YourDetectorName(device='cpu')
detector.load()

# Test prediction
image_tensor, _ = load_image('path/to/image.jpg', size=224)
confidence = detector.forward(image_tensor, 'path/to/image.jpg')
print(f"Prediction: {confidence:.3f}")
```

---

## Adding Explainability & Vulnerability Features

### Overview

The framework uses an abstract method pattern where detectors implement private `_compute_*` methods and expose public API methods:

```
Public API          Abstract Methods      Detector Implementation
predict_with_map()  ->  _compute_explanation_map()      [Detector-specific]
                    ↓
predict_with_vulnerability() -> _compute_vulnerability_map() [Detector-specific]
                              ↓
                    _generate_adversarial_image() [Detector-specific]
```

### Adding Explanation Maps

#### Step 1: Set Feature Flag

```python
class MyDetector(BaseDetector):
    name = 'MyDetector'
    supports_explainability = True  # Enable explainability
```

#### Step 2: Implement `_compute_explanation_map()`

```python
def _compute_explanation_map(
        self,
        image,
        map_size: Optional[Tuple[int, int]] = None,
        **kwargs
) -> Tuple[float, np.ndarray]:
  """
  Compute explanation map for the image.
  
  Args:
      image: Image path (str) or PIL Image
      map_size: Output map size (H, W). If None, use detector default.
      **kwargs: Additional detector-specific arguments
      
  Returns:
      Tuple of (confidence, explanation_map)
          - confidence: float in [0, 1]
          - explanation_map: np.ndarray of shape (H, W) in [0, 1]
  """
  if map_size is None:
    map_size = (self.image_size, self.image_size)

  # Get image path
  image_path = self._ensure_image_path(image)

  # Load and preprocess image
  img_tensor, _ = load_image(image_path, size=self.image_size)
  img_tensor = img_tensor.to(self.device)

  # Get prediction
  with torch.no_grad():
    confidence = self.forward(img_tensor, image_path)

  # Generate explanation map using your method
  # Examples: GradCAM, attention maps, anomaly maps, etc.
  explanation_map = self._my_explanation_method(img_tensor)

  # Resize to desired size
  if explanation_map.shape != map_size:
    explanation_map = self._resize_map(explanation_map, map_size)

  return confidence, explanation_map


def _my_explanation_method(self, img_tensor):
  """Your custom explanation generation logic."""
  # Example: GradCAM, saliency, attention visualization, etc.
  pass


@staticmethod
def _resize_map(map_np: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
  """Resize explanation map to target size."""
  from PIL import Image as PILImage
  if map_np.shape == target_size:
    return map_np
  map_img = PILImage.fromarray((map_np * 255).astype(np.uint8), mode='L')
  map_img = map_img.resize((target_size[1], target_size[0]), PILImage.BILINEAR)
  return np.array(map_img).astype(np.float32) / 255.0
```

#### Step 3: Use the Public API

```python
detector.load()
confidence, explanation_map = detector.predict_with_map('image.jpg')
print(f"Confidence: {confidence:.3f}")
print(f"Explanation map shape: {explanation_map.shape}")
```

### Adding Vulnerability Analysis

#### Step 1: Set Feature Flags

```python
class MyDetector(BaseDetector):
    name = 'MyDetector'
    supports_explainability = True
    supports_vulnerability = True      # Enable vulnerability analysis
    supports_adversarial = True        # Enable adversarial generation
```

#### Step 2: Implement `_generate_adversarial_image()`

```python
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
    Generate adversarial perturbation.
    
    Args:
        image_path: Path to original image
        output_path: Where to save adversarial image
        attack_type: Attack type ('pgd', 'fgsm', 'deepfool', etc.)
        epsilon: Attack strength (detector-dependent)
        true_label: True label (0=real, 1=fake). Attack targets opposite class.
        **kwargs: Additional attack parameters
        
    Returns:
        Path to saved adversarial image
    """
    # Load original image
    original_img = Image.open(image_path).convert('RGB')
    original_size = original_img.size
    
    # Load and preprocess
    img_tensor, _ = load_image(image_path, size=self.image_size)
    img_tensor = img_tensor.to(self.device)
    if img_tensor.ndim == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # Create adversarial perturbation using your method
    # Options:
    # - Use torchattacks library
    # - Use your custom attack method
    # - Use gradient-based attacks
    x_adv = self._apply_attack(img_tensor, attack_type, epsilon, true_label)
    
    # Convert to image
    x_adv_np = x_adv.squeeze(0).cpu().numpy()
    x_adv_np = np.clip(x_adv_np, 0, 1)  # Clip to [0, 1]
    x_adv_np = (x_adv_np * 255).astype(np.uint8)
    x_adv_np = np.transpose(x_adv_np, (1, 2, 0))  # CHW -> HWC
    
    # Create and save
    adv_img = Image.fromarray(x_adv_np)
    if adv_img.size != original_size:
        adv_img = adv_img.resize(original_size, Image.BILINEAR)
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    adv_img.save(output_path)
    print(f"Adversarial image saved: {output_path}")
    
    return output_path


def _apply_attack(self, img_tensor, attack_type, epsilon, true_label):
    """Your attack implementation."""
    # Example using torchattacks:
    # import torchattacks as ta
    # attack = ta.PGD(self.model, eps=epsilon, alpha=epsilon/4, steps=10)
    # return attack(img_tensor, torch.tensor([1 - true_label]))
    pass
```

#### Step 3: Implement `_compute_vulnerability_map()`

```python
def _compute_vulnerability_map(
    self,
    image,
    attack_type: str = "pgd",
    epsilon: float = 0.03,
    map_size: Optional[Tuple[int, int]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Compute vulnerability by comparing explanations before/after attack.
    
    Args:
        image: Image path or PIL Image
        attack_type: Attack type
        epsilon: Attack strength
        map_size: Output map size
        **kwargs: Additional arguments (e.g., true_label)
        
    Returns:
        Dict with keys:
            - 'prediction': float, original prediction
            - 'prediction_attacked': float, attacked prediction
            - 'explanation_map': np.ndarray, original explanation
            - 'explanation_map_attacked': np.ndarray, attacked explanation
            - 'vulnerability_map': np.ndarray, |original - attacked|
            - 'input_tensor': torch.Tensor, preprocessed input
    """
    if map_size is None:
        map_size = (self.image_size, self.image_size)
    
    image_path = self._ensure_image_path(image)
    
    # Get original prediction and map
    confidence, explanation_map = self._compute_explanation_map(image_path, map_size)
    
    # Generate adversarial
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        adv_path = f.name
    
    true_label = kwargs.get('true_label', 1)
    self._generate_adversarial_image(
        image_path, adv_path, attack_type, epsilon, true_label
    )
    
    # Get attacked prediction and map
    confidence_attacked, explanation_map_attacked = self._compute_explanation_map(
        adv_path, map_size
    )
    
    # Compute vulnerability map
    vulnerability_map = np.abs(explanation_map - explanation_map_attacked)
    
    # Load tensor
    img_tensor, _ = load_image(image_path, size=self.image_size)
    img_tensor = img_tensor.to(self.device)
    
    # Cleanup
    try:
        os.unlink(adv_path)
    except:
        pass
    
    return {
        'prediction': confidence,
        'prediction_attacked': confidence_attacked,
        'explanation_map': explanation_map,
        'explanation_map_attacked': explanation_map_attacked,
        'vulnerability_map': vulnerability_map,
        'input_tensor': img_tensor.unsqueeze(0),
    }
```

#### Step 4: Implement Grid Visualization (Optional)

For comprehensive visualizations, implement `visualize_vulnerability_grid()`:

```python
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
    Create a grid visualization of vulnerability analysis.
    
    Args:
        image_set: ImageSet dataclass with image paths
        output_path: Where to save visualization
        attack_type: Attack type for labeling
        dpi: DPI for saved image
        overlay_alpha: Transparency for map overlays
        **kwargs: Additional arguments
        
    Returns:
        Dict with 'generated_adversarial': bool
    """
    # See R50_nodown or AnomalyOV implementations for examples
    pass
```

### Example: Full Detector with Explainability

```python
class MyGradCAMDetector(BaseDetector):
    name = 'MyGradCAM'
    supports_explainability = True
    supports_vulnerability = True
    supports_adversarial = True
    
    def __init__(self, device=None):
        super().__init__(device)
        self.model = None
        self.cam = None  # For GradCAM
    
    def load(self, model_id=None):
        # Load model and initialize GradCAM
        weights = model_id or 'weights/best.pt'
        self.model = torch.load(weights, map_location=self.device)
        
        from pytorch_grad_cam import GradCAM
        target_layer = self.model.layer4[-1]
        self.cam = GradCAM(model=self.model, target_layers=[target_layer])
    
    def predict(self, image_tensor, image_path):
        out = self.model(image_tensor)
        return float(torch.sigmoid(out).item())
    
    def _compute_explanation_map(self, image, map_size=None, **kwargs):
        if map_size is None:
            map_size = (512, 512)
        
        image_path = self._ensure_image_path(image)
        img_tensor, _ = load_image(image_path, size=512)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        confidence = self.predict(img_tensor, image_path)
        
        # Get GradCAM
        grayscale_cam = self.cam(img_tensor)
        explanation_map = grayscale_cam[0]
        
        # Resize if needed
        if explanation_map.shape != map_size:
            explanation_map = self._resize_map(explanation_map, map_size)
        
        return confidence, explanation_map
    
    def _generate_adversarial_image(self, image_path, output_path, 
                                   attack_type="pgd", epsilon=0.03, 
                                   true_label=1, **kwargs):
        import torchattacks as ta
        
        original_img = Image.open(image_path).convert('RGB')
        img_tensor, _ = load_image(image_path, size=512)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        attack = ta.PGD(self.model, eps=epsilon, alpha=epsilon/4, steps=10)
        target = torch.tensor([1 - true_label], device=self.device)
        x_adv = attack(img_tensor, target)
        
        # Save adversarial
        x_adv_np = x_adv.squeeze(0).cpu().numpy()
        x_adv_np = (np.clip(x_adv_np, 0, 1) * 255).astype(np.uint8)
        x_adv_np = np.transpose(x_adv_np, (1, 2, 0))
        
        adv_img = Image.fromarray(x_adv_np)
        adv_img.save(output_path)
        return output_path
    
    def _compute_vulnerability_map(self, image, attack_type="pgd", 
                                  epsilon=0.03, map_size=None, **kwargs):
        # Use base class helper implementation
        return super()._compute_vulnerability_map(
            image, attack_type, epsilon, map_size, **kwargs
        )
```

---

## Understanding the Framework

### Architecture Overview

```
BaseDetector (abstract)
├── load()                           [Abstract - must implement]
├── predict()                        [Abstract - must implement]
├── _compute_explanation_map()       [Optional - implement for explainability]
├── _compute_vulnerability_map()     [Optional - implement for vulnerability]
├── _generate_adversarial_image()    [Optional - implement for adversarial]
│
└── Public API (implemented by BaseDetector)
    ├── predict_with_map()           [Uses _compute_explanation_map()]
    ├── predict_with_vulnerability() [Uses _compute_vulnerability_map()]
    ├── generate_adversarial()       [Uses _generate_adversarial_image()]
    └── visualize_vulnerability_grid() [Optional - override for custom viz]
```

### Feature Flags

Detectors declare capabilities:

```python
supports_explainability = True   # Can call predict_with_map()
supports_vulnerability = True    # Can call predict_with_vulnerability()
supports_adversarial = True      # Can call generate_adversarial()
```

If a feature is not supported, calling the corresponding public method raises `NotImplementedError`.

### Available Detectors

```
✓ R50_nodown       - GradCAM-based explainability
✓ AnomalyOV        - Anomaly map-based explainability
✗ NPR              - Basic prediction only
✗ R50_TF           - Basic prediction only
? CLIP-D           - Requires special setup
? P2G              - Requires einops
? WaveRep          - Requires pywt
```

### Helper Methods

The `BaseDetector` provides several utilities:

```python
detector._ensure_image_path(image)     # Convert PIL Image to path
detector._load_mask(mask_path)         # Load mask as [0, 1] array
detector._get_or_generate_adversarial(...) # Smart caching for advs
detector.get_capabilities()            # Get feature dict
detector.label_from_conf(conf)         # Convert confidence to label
```

---

## FAQ

**Q: How do I test my new detector?**

```python
from models.MyDetector.detector import MyDetector

detector = MyDetector(device='cpu')
detector.load()
conf = detector.forward(tensor, 'image.jpg')
```

**Q: Can I use a different attack library?**
A: Yes! `_generate_adversarial_image()` is fully detector-specific. Use torchattacks, foolbox, or your own implementation.

**Q: What if my model outputs different confidence ranges?**
A: Normalize in `predict()` to return [0, 1]. Use sigmoid, softmax, or manual scaling as needed.

**Q: Can I customize the visualization?**
A: Yes, override `visualize_vulnerability_grid()` for complete control over the output.

**Q: How do I support multiple explanation types?**
A: Pass `explanation_type` via `**kwargs` in `_compute_explanation_map()` and dispatch accordingly.

---

## Contributing

To add a new detector:
1. Create `models/[YourName]/detector.py`
2. Implement `BaseDetector` interface
3. Register in `detect.py`
4. Test with `visualize_vulnerability.py`
5. Document capabilities in README

---

## References

- **GradCAM**: https://github.com/jacobgil/pytorch-grad-cam
- **torchattacks**: https://github.com/Harry24k/adversarial-attacks-pytorch
- **foolbox**: https://foolbox.jonasrauber.de/
