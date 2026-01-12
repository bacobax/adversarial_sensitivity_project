import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def log_configuration(args: argparse.Namespace) -> None:
    """Log all parsed arguments at the beginning of execution."""
    logger.info("=" * 60)
    logger.info("Spatial Correlation Test Configuration")
    logger.info("=" * 60)
    logger.info(f"Root Dataset: {args.root_dataset}")
    logger.info("Models and weights:")
    for det in args.detectors:
        # weight_info = args.weights_dict.get(det, "(using default weights)")
        logger.info(f"  {det}")
    logger.info(f"Attacks: {', '.join(args.attacks)}")
    logger.info(f"Image Types: {', '.join(args.image_types)}")
    logger.info(f"Top-K Percent: {args.topk_percent}")
    logger.info(f"Max Visualizations: {args.max_visualizations}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Overwrite Attacks: {args.overwrite_attacks}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"DPI: {args.dpi}")
    logger.info("=" * 60)
