import os
import sys

import torch

from support.base_detector import BaseDetector
from utils.consts import DETECTOR_MAP, SCRIPT_DIR
from utils.logging import logger


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
    device: torch.device,
) -> BaseDetector:
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
    
    logger.info(f"  ✓ Detector {detector_name} ready")
    return detector
