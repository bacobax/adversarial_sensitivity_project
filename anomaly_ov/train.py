#!/usr/bin/env python3
"""
Convenience script to run Anomaly OV fine-tuning.

Usage:
    python train.py [options]

Examples:
    # Basic training
    python train.py

    # Resume from checkpoint
    python train.py --resume --resume-path ./outputs/checkpoints/checkpoint_epoch_1.pt

    # Custom settings
    python train.py --epochs 10 --batch-size 8 --lr 5e-5

    # Full options
    python train.py --help
"""

import os
import sys

# Add project root to path
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Add src directories to path
_src_dir = os.path.join(_project_root, 'src')
_training_dir = os.path.join(_src_dir, 'training')
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)
if _training_dir not in sys.path:
    sys.path.insert(0, _training_dir)

from src.training.train import main

if __name__ == "__main__":
    main()
