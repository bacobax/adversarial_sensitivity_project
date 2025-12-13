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

import os
from typing import Any, Dict, List, Optional, Tuple

import concurrent.futures
import multiprocessing as mp
import numpy as np
import torch
from skimage.segmentation import slic
from tqdm import tqdm

from support.base_detector import BaseDetector
from support.detect_utils import get_device
from utils import arg_parse, attack, evaluate, image_loader, visualize
from utils.detector_loader import load_detector
from utils.logging import log_configuration, logger
from utils.sample_paths import SamplePaths
from utils.image_loader import load_image

LIME_BATCH_SIZE = 256
LIME_NUM_SAMPLES = 256


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
        
        # Get image path
        if img_type == 'samecat':
            img_path = sample.samecat
            mask_path = sample.mask_samecat
        elif img_type == 'diffcat':
            img_path = sample.diffcat
            mask_path = sample.mask_diffcat
        else:
            continue
        
        if not img_path or not os.path.exists(img_path):
            continue
        
        # Load image
        image_pil = load_image(img_path)
        image_np = np.array(image_pil)
        vis_data['images'][img_type] = image_np
        
        # Load/compute ground truth mask
        if img_type == 'samecat':
            gt_mask = image_loader.load_mask(mask_path)
        else:  # diffcat
            gt_mask = image_loader.bbox_to_mask(mask_path, image_np.shape[:2])
        vis_data['gt_masks'][img_type] = gt_mask
        
        kwargs = {}
        if detector.name == 'WaveRep':
            fixed_segments = slic(image_np, n_segments=24, compactness=20, start_label=0)
            kwargs = {
                'batch_size': LIME_BATCH_SIZE,
                'num_samples': LIME_NUM_SAMPLES,
                'fixed_segments': fixed_segments,
            }
        
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
        
        # Compute or get cached original explanation
        cache_path_orig = cache_paths[('orig', img_type)]
        if os.path.exists(cache_path_orig):
            exp_orig = np.load(cache_path_orig)
        else:
            exp_orig = to_numpy(detector.explain(image_np, **kwargs))
            np.save(cache_path_orig, exp_orig)
        vis_data['exp_orig'][img_type] = exp_orig
        
        # Compute explanation on attacked image
        cache_path_adv = cache_paths[(attack_type, img_type)]
        if os.path.exists(cache_path_adv):
            exp_adv = np.load(cache_path_adv)
        else:
            exp_adv = to_numpy(detector.explain(adv_image, **kwargs))
            np.save(cache_path_adv, exp_adv)
        vis_data['exp_adv'][img_type] = exp_adv
        
        # Compute vulnerability map
        # vuln_map = np.abs(exp_orig - exp_adv)
        eps = 1e-8
        exp_orig_norm = exp_orig / (np.abs(exp_orig).sum() + eps)
        exp_adv_norm = exp_adv / (np.abs(exp_adv).sum() + eps)
        vuln_map = exp_orig_norm - exp_adv_norm
        
        # Normalize vulnerability map to [0, 1]
        if vuln_map.max() > 0:
            vuln_map = vuln_map / vuln_map.max()
        vis_data['vuln_maps'][img_type] = vuln_map
        
        # Compute metrics
        # Explanation metrics: exp_orig vs gt_mask
        exp_metrics = evaluate.compute_metrics(exp_orig, gt_mask, topk_percents)
        explanation_metrics[img_type] = exp_metrics
        
        # Vulnerability metrics: vuln_map vs gt_mask
        vuln_metrics = evaluate.compute_metrics(vuln_map, gt_mask, topk_percents)
        vulnerability_metrics[img_type] = vuln_metrics
    
    return explanation_metrics, vulnerability_metrics, vis_data


def _precompute_explanations_for_detector(
    *,
    detector: BaseDetector,
    samples: List[SamplePaths],
    image_types: List[str],
    topk_percents: List[float],
    pt_output: str,
) -> evaluate.MetricsAggregator:
    exp_aggregator = evaluate.MetricsAggregator()
    exp_processed = set()

    pbar = tqdm(
        samples,
        desc=f"Precomputing explanations ({detector.name})",
        unit="sample",
        leave=True,
    )
    for sample in pbar:
        pbar.set_postfix_str(f"{sample.filename[:30]}...", refresh=True)
        base_name = os.path.splitext(sample.filename)[0]

        cache_paths: Dict[Tuple[str, str], str] = {}
        for img_type in image_types:
            cache_paths[('orig', img_type)] = os.path.join(pt_output, f"{base_name}_orig_{img_type}.npy")
            cache_paths[('__noop__', img_type)] = os.path.join(pt_output, f"{base_name}___noop___{img_type}.npy")

        try:
            exp_metrics, _, _ = process_sample(
                sample=sample,
                detector=detector,
                attack_type='__noop__',
                image_types=image_types,
                root_dataset='',
                topk_percents=topk_percents,
                overwrite_attacks=False,
                cache_paths=cache_paths,
            )
        except Exception:
            continue

        for img_type, metrics in exp_metrics.items():
            key = (sample.filename, img_type)
            if key in exp_processed:
                continue
            metadata = {
                'filename': sample.filename,
                'image_type': img_type,
            }
            if len(topk_percents) > 1:
                metadata['topk_percent'] = topk_percents[0]
            exp_aggregator.update(metrics, metadata=metadata)
            exp_processed.add(key)

    pbar.close()
    return exp_aggregator


def _run_single_attack_process(
    *,
    detector_name: str,
    weights_path: Optional[str],
    device_str: str,
    attack_type: str,
    samples: List[SamplePaths],
    image_types: List[str],
    root_dataset: str,
    topk_percents: List[float],
    overwrite_attacks: bool,
    output_dir: str,
    max_visualizations: int,
    dpi: int,
) -> Tuple[str, int, int]:
    device = torch.device(device_str)
    detector = load_detector(detector_name, weights_path, device)

    detector_output = os.path.join(output_dir, detector_name)
    os.makedirs(detector_output, exist_ok=True)

    vuln_aggregator = evaluate.MetricsAggregator()
    vis_count = 0

    attack_output = os.path.join(detector_output, attack_type)
    os.makedirs(attack_output, exist_ok=True)
    vis_output = os.path.join(detector_output, 'vis', attack_type)
    os.makedirs(vis_output, exist_ok=True)
    pt_output = os.path.join(detector_output, 'maps')
    os.makedirs(pt_output, exist_ok=True)

    pbar = tqdm(
        samples,
        desc=f"Processing {detector_name}/{attack_type}",
        unit="sample",
        leave=True,
    )
    for sample in pbar:
        pbar.set_postfix_str(f"{sample.filename[:30]}...", refresh=True)
        base_name = os.path.splitext(sample.filename)[0]

        cache_paths: Dict[Tuple[str, str], str] = {}
        for img_type in image_types:
            cache_paths[('orig', img_type)] = os.path.join(pt_output, f"{base_name}_orig_{img_type}.npy")
            cache_paths[(attack_type, img_type)] = os.path.join(pt_output, f"{base_name}_{attack_type}_{img_type}.npy")

        try:
            _, vuln_metrics, vis_data = process_sample(
                sample=sample,
                detector=detector,
                attack_type=attack_type,
                image_types=image_types,
                root_dataset=root_dataset,
                topk_percents=topk_percents,
                overwrite_attacks=overwrite_attacks,
                cache_paths=cache_paths,
            )

            for img_type, metrics in vuln_metrics.items():
                metadata = {
                    'filename': sample.filename,
                    'image_type': img_type,
                    'attack_type': attack_type,
                }
                if len(topk_percents) > 1:
                    metadata['topk_percent'] = topk_percents[0]
                vuln_aggregator.update(metrics, metadata=metadata)

            if vis_count < max_visualizations:
                pbar.set_postfix_str(
                    f"Generating visualization {vis_count + 1}/{max_visualizations}",
                    refresh=True,
                )
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
                        dpi=dpi,
                        detector_name=detector.name,
                    )
                    vis_count += 1
                except Exception as e:
                    logger.warning(f"Visualization failed for {sample.filename}: {e}")

        except Exception as e:
            pbar.set_postfix_str(f"ERROR: {str(e)[:30]}", refresh=True)
            logger.error(f"Failed to process {sample.filename}: {e}")
            continue

    pbar.close()

    if len(vuln_aggregator) > 0:
        vuln_csv = os.path.join(attack_output, "metrics_vulnerability.csv")
        vuln_aggregator.to_csv(vuln_csv)
        logger.info(f"✓ Vulnerability metrics saved to: {vuln_csv}")
    else:
        logger.warning(f"No vulnerability metrics collected for {attack_type}")

    return attack_type, len(vuln_aggregator), vis_count


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
        
        try:
            detector = load_detector(detector_name, weights_dict.get(detector_name), device)
        except Exception as e:
            logger.error(f"Failed to load detector {detector_name}: {e}")
            continue

        detector_output = os.path.join(args.output_dir, detector_name)
        os.makedirs(detector_output, exist_ok=True)
        pt_output = os.path.join(detector_output, 'maps')
        os.makedirs(pt_output, exist_ok=True)

        exp_aggregator = _precompute_explanations_for_detector(
            detector=detector,
            samples=samples,
            image_types=image_types,
            topk_percents=args.topk_percent,
            pt_output=pt_output,
        )

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

        # Run each attack in a separate process
        logger.info(f"\n--- Running {len(attacks)} attack(s) in parallel processes ---")
        max_workers = args.attack_processes or len(attacks)
        max_workers = min(max_workers, len(attacks))

        ctx = mp.get_context('spawn')
        futures: Dict[concurrent.futures.Future, str] = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
            for attack_type in attacks:
                fut = executor.submit(
                    _run_single_attack_process,
                    detector_name=detector_name,
                    weights_path=weights_dict.get(detector_name),
                    device_str=str(device),
                    attack_type=attack_type,
                    samples=samples,
                    image_types=image_types,
                    root_dataset=args.root_dataset,
                    topk_percents=args.topk_percent,
                    overwrite_attacks=args.overwrite_attacks,
                    output_dir=args.output_dir,
                    max_visualizations=args.max_visualizations,
                    dpi=args.dpi,
                )
                futures[fut] = attack_type

            for fut in concurrent.futures.as_completed(futures):
                attack_type = futures[fut]
                try:
                    atk, n_metrics, n_vis = fut.result()
                    logger.info(
                        f"✓ Completed attack {atk}: vulnerability rows={n_metrics}, visualizations={n_vis}",
                    )
                except Exception as e:
                    logger.error(f"Attack process failed for {attack_type}: {e}")
    
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
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    main()
