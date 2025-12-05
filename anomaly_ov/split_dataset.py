#!/usr/bin/env python3
"""
Dataset Splitting Script for Anomaly Detection

This script reorganizes the dataset from:
  data_folder/
    - inpainted/
    - masks/
    - COCO_real/

To:
  data_folder/
    - train/inpainted/
    - train/masks/
    - train/COCO_real/
    - eval/inpainted/
    - eval/masks/
    - eval/COCO_real/
    - test/inpainted/
    - test/masks/
    - test/COCO_real/

The split balances both real and inpainted classes across all splits.
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict
import argparse


def get_matching_files(inpainted_dir, real_dir, masks_dir):
    """
    Find all matching files across inpainted, real, and masks directories.
    
    Returns:
        List of filenames that exist in all three directories
    """
    inpainted_files = set(f.name for f in Path(inpainted_dir).glob("*") if f.is_file())
    real_files = set(f.name for f in Path(real_dir).glob("*") if f.is_file())
    masks_files = set(f.name for f in Path(masks_dir).glob("*") if f.is_file())
    
    # Find common files
    common_files = inpainted_files & real_files & masks_files
    
    print(f"Found {len(inpainted_files)} inpainted files")
    print(f"Found {len(real_files)} real files")
    print(f"Found {len(masks_files)} mask files")
    print(f"Found {len(common_files)} matching files across all directories")
    
    return sorted(list(common_files))


def split_dataset(data_root, train_ratio=0.7, eval_ratio=0.15, test_ratio=0.15, 
                  real_folder="COCO_real", seed=42, dry_run=False):
    """
    Split dataset into train/eval/test sets.
    
    Args:
        data_root: Root directory containing inpainted/, masks/, and real_folder/
        train_ratio: Proportion for training set (default: 0.7)
        eval_ratio: Proportion for evaluation set (default: 0.15)
        test_ratio: Proportion for test set (default: 0.15)
        real_folder: Name of the real images folder (default: "COCO_real")
        seed: Random seed for reproducibility
        dry_run: If True, only print what would be done without copying files
    """
    # Validate ratios
    assert abs(train_ratio + eval_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + eval_ratio + test_ratio}"
    
    data_root = Path(data_root)
    
    # Define source directories
    inpainted_dir = data_root / "inpainted"
    real_dir = data_root / real_folder
    masks_dir = data_root / "masks"
    
    # Verify source directories exist
    for d in [inpainted_dir, real_dir, masks_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Directory not found: {d}")
    
    # Get matching files
    matching_files = get_matching_files(inpainted_dir, real_dir, masks_dir)
    
    if len(matching_files) == 0:
        raise ValueError("No matching files found across all directories!")
    
    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(matching_files)
    
    # Calculate split indices
    n_total = len(matching_files)
    n_train = int(n_total * train_ratio)
    n_eval = int(n_total * eval_ratio)
    n_test = n_total - n_train - n_eval  # Use remaining to avoid rounding issues
    
    # Split files
    train_files = matching_files[:n_train]
    eval_files = matching_files[n_train:n_train + n_eval]
    test_files = matching_files[n_train + n_eval:]
    
    print("\n" + "="*80)
    print("Dataset Split Summary:")
    print("="*80)
    print(f"Total files: {n_total}")
    print(f"Train files: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    print(f"Eval files:  {len(eval_files)} ({len(eval_files)/n_total*100:.1f}%)")
    print(f"Test files:  {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    print("="*80)
    
    if dry_run:
        print("\nDRY RUN MODE - No files will be copied")
        print("\nExample files for each split:")
        print(f"Train: {train_files[:3]}")
        print(f"Eval:  {eval_files[:3]}")
        print(f"Test:  {test_files[:3]}")
        return
    
    # Create target directory structure
    splits = {
        'train': train_files,
        'eval': eval_files,
        'test': test_files
    }
    
    subdirs = ['inpainted', 'masks', real_folder]
    
    print("\nCreating directory structure...")
    for split_name in splits.keys():
        for subdir in subdirs:
            target_dir = data_root / split_name / subdir
            target_dir.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {target_dir}")
    
    # Copy files to respective splits
    print("\nCopying files...")
    stats = defaultdict(int)
    
    for split_name, files in splits.items():
        print(f"\nProcessing {split_name} split ({len(files)} files)...")
        
        for i, filename in enumerate(files, 1):
            # Copy from each source directory
            for source_name, source_dir in [
                ('inpainted', inpainted_dir),
                ('masks', masks_dir),
                (real_folder, real_dir)
            ]:
                src_file = source_dir / filename
                dst_file = data_root / split_name / source_name / filename
                
                if src_file.exists():
                    shutil.copy2(src_file, dst_file)
                    stats[f"{split_name}_{source_name}"] += 1
                else:
                    print(f"  Warning: Source file not found: {src_file}")
            
            # Progress update every 100 files
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(files)} files copied")
    
    print("\n" + "="*80)
    print("Copy Statistics:")
    print("="*80)
    for split_name in splits.keys():
        print(f"\n{split_name.upper()}:")
        for subdir in subdirs:
            key = f"{split_name}_{subdir}"
            print(f"  {subdir}: {stats[key]} files")
    
    print("\n" + "="*80)
    print("Dataset splitting complete!")
    print("="*80)
    print(f"\nNew structure created in: {data_root}")
    print("\nDirectory structure:")
    for split_name in splits.keys():
        print(f"  {split_name}/")
        for subdir in subdirs:
            count = stats[f"{split_name}_{subdir}"]
            print(f"    {subdir}/ ({count} files)")


def main():
    parser = argparse.ArgumentParser(
        description="Split anomaly detection dataset into train/eval/test sets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split with default ratios (70/15/15)
  python split_dataset.py ./finetune_dataset

  # Custom ratios
  python split_dataset.py ./finetune_dataset --train 0.6 --eval 0.2 --test 0.2

  # Different real folder name
  python split_dataset.py ./finetune_dataset --real-folder real_images

  # Dry run to see what would happen
  python split_dataset.py ./finetune_dataset --dry-run
        """
    )
    
    parser.add_argument(
        'data_root',
        type=str,
        help='Path to the dataset root directory'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--eval',
        type=float,
        default=0.15,
        help='Evaluation set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='Test set ratio (default: 0.15)'
    )
    
    parser.add_argument(
        '--real-folder',
        type=str,
        default='COCO_real',
        help='Name of the real images folder (default: COCO_real)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            data_root=args.data_root,
            train_ratio=args.train,
            eval_ratio=args.eval,
            test_ratio=args.test,
            real_folder=args.real_folder,
            seed=args.seed,
            dry_run=args.dry_run
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
