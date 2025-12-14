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
import gc
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from skimage.segmentation import slic
from sklearn.metrics import average_precision_score
from tqdm import tqdm

from support.base_detector import BaseDetector
from support.detect_utils import get_device
from utils import arg_parse, attack, evaluate, image_loader, visualize
from utils.detector_loader import load_detector
from utils.image_loader import load_image
from utils.logging import log_configuration, logger
from utils.sample_paths import SamplePaths

LIME_BATCH_SIZE = 256
LIME_NUM_SAMPLES = 256

cvs_file = 'outputs/results.csv'
cvs_header = 'detector,attack,category,image,logit,sigmoid,ap,mim'

if not os.path.exists(cvs_file):
    with open(cvs_file, 'w') as f:
        f.write(cvs_header + '\n')

CSV_COLS = cvs_header.split(',')
INDEX_COLS = ["detector", "attack", "category", "image"]
DATA_COLS = [c for c in CSV_COLS if c not in INDEX_COLS]

df = pd.DataFrame(columns=DATA_COLS)
df.index = pd.MultiIndex.from_tuples([], names=INDEX_COLS)


def _ensure_row(df_: pd.DataFrame, key: tuple) -> None:
    """Ensure MultiIndex row exists so df.loc[...] assignments work."""
    if key not in df_.index:
        df_.loc[key, :] = np.nan


def _set_cell(df_: pd.DataFrame, key: tuple, col: str, val) -> None:
    _ensure_row(df_, key)
    df_.loc[key, col] = val


def to_numpy(arr):
    """Convert tensor or array to numpy array on CPU."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def process_sample(
    sample: SamplePaths,
    detector: BaseDetector,
    attack_type: str,
    image_types: List[str],
    root_dataset: str,
    topk_percents: List[float],
    overwrite_attacks: bool,
    cache_paths: Dict[Tuple[str, str], str],
    # results: Dict[str, Dict[str, List[float]]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Any]]:
    """
    Process a single sample for all requested image types.

    Args:
        sample: Sample paths
        detector: Base Detector
        attack_type: Attack type
        image_types: List of image types to process
        root_dataset: Root dataset path
        topk_percents: List of top-k percentages
        overwrite_attacks: Whether to overwrite cached attacks
        cache_paths: Cache for original explanation maps
        results: Dict to store results

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
                vis_data['images']['real'] = np.array(load_image(sample.real))
            except:
                pass
            continue
        
        exp_orig = None
        exp_adv = None
        cache_path_orig = cache_paths[('orig', img_type)]
        cache_path_adv = cache_paths[(attack_type, img_type)]
        if os.path.exists(cache_path_orig):
            exp_orig = np.load(cache_path_orig)
        if os.path.exists(cache_path_adv):
            exp_adv = np.load(cache_path_adv)
        
        # Get image path
        if img_type == 'samecat':
            img_path = sample.samecat
            mask_path = sample.mask_samecat
        elif img_type == 'diffcat':
            img_path = sample.diffcat
            mask_path = sample.mask_diffcat
        else:
            continue
        
        eps = 1e-8
        
        # Load image
        image_pil = load_image(img_path)
        image_np = np.array(image_pil)
        
        df_key_orig = (detector.name, 'orig', img_type, os.path.basename(sample.filename))
        df_key_adv = (detector.name, attack_type, img_type, os.path.basename(sample.filename))
        _ensure_row(df, df_key_orig)
        _ensure_row(df, df_key_adv)
        
        needs_maps = (exp_orig is None) or (exp_adv is None)
        needs_logits = (
            pd.isna(df.loc[df_key_orig, "logit"]) or
            pd.isna(df.loc[df_key_adv, "logit"])
        )
        
        if needs_maps or needs_logits:
            if not img_path or not os.path.exists(img_path):
                continue
                
            # Get or generate attacked image
            attack_cache_path = attack.get_attack_cache_path(
                root_dataset, detector.name, attack_type, img_type, sample.filename,
            )
            adv_image = attack.get_or_generate_attacked_image(
                detector=detector,
                image=image_pil,
                attack_type=attack_type,
                cache_path=attack_cache_path,
                overwrite=overwrite_attacks,
            )
            
        
        # compute only if exp_orig or exp_adv are missing
        if needs_maps:
            # Load/compute ground truth mask
            # if img_type == 'samecat':
            #     gt_mask = image_loader.load_mask(mask_path)
            # else:  # diffcat
            #     gt_mask = image_loader.bbox_to_mask(mask_path, image_np.shape[:2])
            
            kwargs = {}
            if detector.name == 'WaveRep':
                fixed_segments = slic(image_np, n_segments=24, compactness=20, start_label=0)
                kwargs = {
                    'batch_size': LIME_BATCH_SIZE,
                    'num_samples': LIME_NUM_SAMPLES,
                    'fixed_segments': fixed_segments,
                }
            
            # Compute or get cached original explanation
            if exp_orig is None:
                exp_orig = to_numpy(detector.explain(image_np, **kwargs))
                np.save(cache_path_orig, exp_orig)
            
            # Compute explanation on attacked image
            if exp_adv is None:
                exp_adv = to_numpy(detector.explain(adv_image, **kwargs))
                np.save(cache_path_adv, exp_adv)
        
        if needs_logits:
            with torch.inference_mode():
                if pd.isna(df.loc[df_key_orig, "logit"]):
                    logit_orig = detector.forward(detector.transform(image_np).unsqueeze(0).to('cuda', non_blocking=True)).detach().cpu()
                    sigmoid_orig = torch.sigmoid(logit_orig).item()
                    _set_cell(df, df_key_orig, "logit", float(logit_orig))
                    _set_cell(df, df_key_orig, "sigmoid", float(sigmoid_orig))
                
                if pd.isna(df.loc[df_key_adv, "logit"]):
                    logit_adv = detector.forward(detector.transform(adv_image).unsqueeze(0).to('cuda', non_blocking=True)).detach().cpu()
                    sigmoid_adv = torch.sigmoid(logit_adv).item()
                    _set_cell(df, df_key_adv, "logit", float(logit_adv))
                    _set_cell(df, df_key_adv, "sigmoid", float(sigmoid_adv))
            
            torch.cuda.empty_cache()
            
            # Compute vulnerability map
            exp_orig_norm = exp_orig / (np.abs(exp_orig).sum() + eps)
            exp_adv_norm = exp_adv / (np.abs(exp_adv).sum() + eps)
            vuln_map = exp_orig_norm - exp_adv_norm
            
            # Normalize vulnerability map to [0, 1]
            # if vuln_map.max() > 0:
            #     vuln_map = vuln_map / vuln_map.max()
            
            # # Compute metrics
            # # Explanation metrics: exp_orig vs gt_mask
            # exp_metrics = evaluate.compute_metrics(exp_orig, gt_mask, topk_percents)
            # explanation_metrics[img_type] = exp_metrics
            #
            # # Vulnerability metrics: vuln_map vs gt_mask
            # vuln_metrics = evaluate.compute_metrics(vuln_map, gt_mask, topk_percents)
            # vulnerability_metrics[img_type] = vuln_metrics
        
        if detector.name == 'AnomalyOV':
            exp_orig_norm = (exp_orig - exp_orig.min()) / (exp_orig.max() - exp_orig.min() + 1e-9)
            exp_adv_norm = (exp_adv - exp_adv.min()) / (exp_adv.max() - exp_adv.min() + 1e-9)
        else:
            exp_orig_norm = exp_orig / (np.abs(exp_orig).sum() + eps)
            exp_adv_norm = exp_adv / (np.abs(exp_adv).sum() + eps)
        
        if detector.name == 'WaveRep':
            vuln = exp_orig_norm + np.abs(-exp_adv_norm)
        else:
            vuln = exp_orig_norm - exp_adv_norm
        
        orig = np.clip(exp_orig_norm, 0, None)**2
        vuln = np.clip(vuln, 0, None)**2
        
        while len(orig.shape) > 2:
            orig = orig[0]
        while len(vuln.shape) > 2:
            vuln = vuln[0]
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if orig.shape != mask.shape:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_AREA)
        if vuln.shape != mask.shape:
            mask = cv2.resize(mask, (vuln.shape[1], vuln.shape[0]), interpolation=cv2.INTER_AREA)
        mask = np.array(mask, dtype=bool)
        
        flat_mask = mask.reshape(-1).astype(np.uint8)
        ap_orig = float(average_precision_score(flat_mask, orig.reshape(-1)))
        ap_vuln = float(average_precision_score(flat_mask, vuln.reshape(-1)))
        # results['ap_orig'][img_type].append(ap_orig)
        # results['ap_vuln'][img_type].append(ap_vuln)
        
        if np.sum(orig) > 0:
            mass_in_mask_orig = np.sum(orig[mask]) / np.sum(orig)
            # results['mim_orig'][img_type].append(mass_in_mask_orig)
        if np.sum(vuln) > 0:
            mass_in_mask_vuln = np.sum(vuln[mask]) / np.sum(vuln)
            # results['mim_vuln'][img_type].append(mass_in_mask_vuln)
        
        _set_cell(df, df_key_orig, "ap", ap_orig)
        _set_cell(df, df_key_adv, "ap", ap_vuln)
        _set_cell(df, df_key_orig, "mim", float(mass_in_mask_orig) if np.sum(orig) > 0 else -1.0)
        _set_cell(df, df_key_adv, "mim", float(mass_in_mask_vuln) if np.sum(vuln) > 0 else -1.0)
        
        # vis_data['exp_orig'][img_type] = exp_orig
        # vis_data['exp_adv'][img_type] = exp_adv
        # vis_data['vuln_maps'][img_type] = vuln / vuln.max()
        # vis_data['gt_masks'][img_type] = mask.astype(np.uint8) * 255
        # vis_data['images'][img_type] = image_np
    
    return explanation_metrics, vulnerability_metrics, vis_data


def main():
    """Main entry point."""
    # Parse arguments
    args = arg_parse.parse_arguments()
    
    # Log configuration
    log_configuration(args)
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()
    logger.info(f"Using device: {device}")
    
    # Parse weights
    detectors_names, weights_dict = arg_parse.parse_models(args.detectors)
    print(detectors_names, weights_dict)
    
    # Collect sample paths
    logger.info("\n" + "=" * 60)
    logger.info("Stage 1: Collecting and validating sample paths")
    logger.info("=" * 60)
    samples = image_loader.collect_sample_paths(args.root_dataset, args.image_types)
    
    # Sort samples for deterministic ordering
    samples.sort(key=lambda s: s.filename)
    
    # Apply limit if specified (for testing)
    if args.limit is not None and args.limit > 0:
        samples = samples[:args.limit]
        logger.info(f"⚠ Limited to first {args.limit} sample(s) for testing")
    
    logger.info(f"✓ {len(samples)} samples ready for processing")
    
    # Normalize image_types to lowercase
    image_types = [t.lower() for t in args.image_types]
    
    # Normalize attacks to lowercase
    attacks = [a.lower() for a in args.attacks]
    
    # Process each detector
    logger.info("\n" + "=" * 60)
    logger.info(f"Stage 2: Processing {len(args.detectors)} detector(s) with {len(attacks)} attack(s)")
    logger.info("=" * 60)
    
    for detector_name in detectors_names:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing detector: {detector_name}")
        logger.info(f"{'=' * 60}")
        
        # Load detector
        try:
            detector = load_detector(detector_name, weights_dict[detector_name], device)
        except Exception as e:
            logger.error(f"Failed to load detector {detector_name}: {e}")
            continue
        
        # Create output directory for this detector
        detector_output = os.path.join(args.output_dir, detector_name)
        os.makedirs(detector_output, exist_ok=True)
        
        # Explanation metrics aggregator (computed once, attack-independent)
        exp_aggregator = evaluate.MetricsAggregator()
        exp_processed = set()  # Track which (filename, image_type) have been computed
        
        # Process each attack type
        for attack_idx, attack_type in enumerate(attacks, 1):
            logger.info(f"\n--- [{attack_idx}/{len(attacks)}] Processing attack: {attack_type.upper()} ---")
            logger.info(f"Samples to process: {len(samples)}")
            logger.info(f"Image types: {', '.join(image_types)}")
            
            # Vulnerability metrics aggregator (per attack)
            vuln_aggregator = evaluate.MetricsAggregator()
            
            # Visualization counter
            vis_count = 0
            
            # Create attack-specific output directory
            attack_output = os.path.join(detector_output, attack_type)
            os.makedirs(attack_output, exist_ok=True)
            vis_output = os.path.join(detector_output, 'vis', attack_type)
            os.makedirs(vis_output, exist_ok=True)
            pt_output = os.path.join(detector_output, 'maps')
            os.makedirs(pt_output, exist_ok=True)
            
            # results = {
            #     'ap_orig': {t: [] for t in image_types},
            #     'ap_vuln': {t: [] for t in image_types},
            #     'mim_orig': {t: [] for t in image_types},
            #     'mim_vuln': {t: [] for t in image_types},
            # }
            
            # Process samples
            pbar = tqdm(
                samples,
                desc=f"Processing {detector_name}/{attack_type}",
                unit="sample",
                leave=True,
            )
            for sample in pbar:
                # Update progress bar with current file
                pbar.set_postfix_str(f"{sample.filename[:30]}...", refresh=True)
                base_name = os.path.splitext(sample.filename)[0]
                
                cache_paths = {}
                for img_type in image_types:
                    cache_paths[('orig', img_type)] = os.path.join(pt_output, f"{base_name}_orig_{img_type}.npy")
                    cache_paths[(attack_type, img_type)] = os.path.join(pt_output, f"{base_name}_{attack_type}_{img_type}.npy")
                
                try:
                    exp_metrics, vuln_metrics, vis_data, = process_sample(
                        sample=sample,
                        detector=detector,
                        attack_type=attack_type,
                        image_types=image_types,
                        root_dataset=args.root_dataset,
                        topk_percents=args.topk_percent,
                        overwrite_attacks=args.overwrite_attacks,
                        cache_paths=cache_paths,
                        # results=results,
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
                        pbar.set_postfix_str(f"Generating visualization {vis_count + 1}/{args.max_visualizations}", refresh=True)
                        vis_path = os.path.join(vis_output, f"{base_name}_grid.png")
                        
                        try:
                            visualize.create_visualization_grid(
                                images=vis_data['images'],
                                exp_orig=vis_data['exp_orig'],
                                exp_adv=vis_data['exp_adv'],
                                vuln_maps=vis_data['vuln_maps'],
                                gt_masks=vis_data['gt_masks'],
                                filename=sample.filename,
                                attack_type=attack_type,
                                output_path=vis_path,
                                dpi=args.dpi,
                                detector_name=detector.name,
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
                
            #     print()
            # for img_type in image_types:
            #     print(img_type)
            #     print('*' * 80)
            #     print(f'orig: ap={np.mean(results['ap_orig'][img_type]):.4f} mim={np.mean(results['mim_orig'][img_type]):.4f}')
            #     print(f'vuln: ap={np.mean(results['ap_vuln'][img_type]):.4f} mim={np.mean(results['mim_vuln'][img_type]):.4f}')
            #     print('*' * 80)
            #     print()
            
            # Save df
            logger.info(f"Saving df to csv...")
            df.reset_index().to_csv(cvs_file, index=False)  # NEW
            logger.info(f"✓ df saved to: {cvs_file}")
        
        # Save explanation metrics CSV (once per detector, attack-independent)
        logger.info(f"\n{'=' * 60}")
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
        logger.info(f"{'=' * 60}")
    
    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("✓ ALL PROCESSING COMPLETE!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Detectors processed: {len(args.detectors)}")
    logger.info(f"Attack types: {len(attacks)}")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"\nOutput structure:")
    logger.info(f"  [output]/<model>/metrics_explanation.csv")
    logger.info(f"  [output]/<model>/<attack>/metrics_vulnerability.csv")
    logger.info(f"  [output]/<model>/vis/<attack>/<filename>_grid.png")
    logger.info(f"{'=' * 60}")
    
if __name__ == '__main__':
    main()
