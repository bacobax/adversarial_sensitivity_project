import argparse
import os
from typing import Dict, List, Tuple

from utils.consts import DETECTOR_MAP, SUPPORTED_ATTACKS, SUPPORTED_IMAGE_TYPES
from utils.logging import logger


def parse_models(models: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse detector models and optional weights from command line arguments.

    Args:
        models: List of strings in format "model_name[:/path/to/weights.pt]"

    Returns:
        Tuple of (detector_names, weights_dict) where:
            - detector_names: List of detector names
            - weights_dict: Dict mapping detector names to weight paths (may be empty)
    """
    detector_names = []
    weights_dict = {}
    
    for model_spec in models:
        if ':' in model_spec:
            first_colon = model_spec.find(':')
            detector_name = model_spec[:first_colon]
            path = model_spec[first_colon + 1:]
            detector_names.append(detector_name)
            weights_dict[detector_name] = path
        else:
            detector_names.append(model_spec)
    
    return detector_names, weights_dict


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Compute spatial correlation metrics for detector explainability and vulnerability.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Required arguments
    parser.add_argument(
        '--root_dataset',
        type=str,
        required=True,
        help='Path to the dataset root containing b-free/ and adv_attacks/',
    )
    parser.add_argument(
        '--detectors',
        type=str,
        nargs='+',
        required=True,
        help=f'List of detector models with optional weights. Format: "model1[:/path/to/weights1.pt] model2[:/path/to/weights2.pt]"',
    )
    parser.add_argument(
        '--attacks',
        type=str,
        nargs='+',
        required=True,
        help=f'List of attack types to run. Supported: {", ".join(SUPPORTED_ATTACKS)}',
    )
    parser.add_argument(
        '--image_types',
        type=str,
        nargs='+',
        required=True,
        help=f'Subset of image types to process. Supported: {", ".join(SUPPORTED_IMAGE_TYPES)}',
    )
    
    # Optional arguments with defaults
    parser.add_argument(
        '--topk_percent',
        type=float,
        nargs='+',
        default=[1.0],
        help='Percentage(s) of pixels considered as high-saliency for IoU (default: 1)',
    )
    parser.add_argument(
        '--max_visualizations',
        type=int,
        default=10,
        help='Maximum number of samples for which to generate grid visualizations (default: 10)',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/',
        help='Root directory for all outputs (default: outputs/)',
    )
    parser.add_argument(
        '--overwrite_attacks',
        action='store_true',
        help='If set, recompute and overwrite attacked images even if they already exist',
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use for computation (default: auto-detect)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=150,
        help='DPI for saved visualization images (default: 150)',
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit processing to first N samples (for testing). Default: process all.',
    )

    parser.add_argument(
        '--attack_processes',
        type=int,
        default=None,
        help='Number of parallel processes to use across attacks (default: number of attacks).',
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
    
    # Parse models and weights
    try:
        detectors, weights_dict = parse_models(args.detectors)
    except ValueError as e:
        errors.append(f"Invalid detector specification: {e}")
        detectors = []
    except Exception as e:
        errors.append(f"Error parsing detectors: {str(e)}")
        detectors = []
    
    # Validate detectors
    if not hasattr(args, 'detectors') or not detectors:
        errors.append("No detectors specified. Use --detectors to specify one or more detectors.")
    else:
        for detector in detectors:
            if detector not in DETECTOR_MAP:
                errors.append(
                    f"Unknown detector: {detector}. "
                    f"Available: {', '.join(sorted(DETECTOR_MAP.keys()))}"
                )
    
    # Validate attacks
    if not hasattr(args, 'attacks') or not args.attacks:
        errors.append("No attack types specified. Use --attacks to specify one or more attack types.")
    else:
        for attack in args.attacks:
            if attack.lower() not in SUPPORTED_ATTACKS:
                errors.append(
                    f"Unknown attack type: {attack}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_ATTACKS))}"
                )
    
    # Validate image_types
    if not hasattr(args, 'image_types') or not args.image_types:
        errors.append("No image types specified. Use --image_types to specify one or more image types.")
    else:
        for img_type in args.image_types:
            if img_type.lower() not in SUPPORTED_IMAGE_TYPES:
                errors.append(
                    f"Unknown image type: {img_type}. "
                    f"Supported: {', '.join(sorted(SUPPORTED_IMAGE_TYPES))}"
                )
    
    # Warn about 'real' in image_types
    if 'real' in [t.lower() for t in args.image_types]:
        logger.warning(
            "'real' is included in image_types but has no ground truth mask. "
            "Metrics will be skipped for real images.",
        )
    
    # Validate topk_percent
    for topk in args.topk_percent:
        if not (0 < topk <= 100):
            errors.append(f"topk_percent must be in (0, 100], got: {topk}")

    # Validate attack_processes
    if hasattr(args, 'attack_processes') and args.attack_processes is not None:
        if args.attack_processes <= 0:
            errors.append(f"attack_processes must be a positive integer, got: {args.attack_processes}")
    
    if errors:
        for error in errors:
            logger.error(error)
        raise ValueError(f"Argument validation failed with {len(errors)} error(s)")
