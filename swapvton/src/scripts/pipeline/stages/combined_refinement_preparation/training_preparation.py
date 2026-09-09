import os
import re
from pathlib import Path
from typing import List, Dict
import numpy as np
from PIL import Image
from loguru import logger


import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from scripts.head_mask_extraction.utils.grid_parser import (
    detect_image_resolution,
    get_camera_names_from_dataset,
    parse_segmentation_grid
)
from scripts.head_mask_extraction.utils.mask_creator import create_head_mask
from scripts.head_mask_extraction.utils.neck_mask_creator import create_neck_mask_from_segmentation
from scripts.head_mask_extraction.utils.image_combination import (
    combine_rgb_images,
    compute_and_crop_portrait
)


def discover_restored_subjects(config: dict) -> List[Dict[str, str]]:

    avatarrex_dir = config['paths']['avatarrex_output']
    swapped_back_dir = os.path.join(avatarrex_dir, "head_swapped_back_renders")

    if not os.path.exists(swapped_back_dir):
        logger.warning(f"Second-swap renders directory not found: {swapped_back_dir}")
        return []

    restored_subjects = []

    for item in os.listdir(swapped_back_dir):
        if not item.startswith("restored_"):
            continue

        restored_path = os.path.join(swapped_back_dir, item)
        if not os.path.isdir(restored_path):
            continue


        match = re.match(r'restored_(\d+)_from_', item)
        if not match:
            logger.warning(f"Could not parse subject ID from directory name: {item}")
            continue

        subject_id = match.group(1)


        gt_path = os.path.join(avatarrex_dir, subject_id)
        if not os.path.exists(gt_path):
            logger.warning(f"GT directory not found for subject {subject_id}: {gt_path}")
            continue


        seg_grid_path = os.path.join(gt_path, "Semantic", "process", "parser", "parser-f0000.png")
        if not os.path.exists(seg_grid_path):
            logger.warning(f"GT segmentation not found for subject {subject_id}: {seg_grid_path}")
            continue

        restored_subjects.append({
            'subject_id': subject_id,
            'restored_path': restored_path,
            'gt_path': gt_path,
            'restored_name': item
        })

        logger.debug(f"Found restored subject: {subject_id} ({item})")

    logger.info(f"Discovered {len(restored_subjects)} restored subjects for combined portrait creation")

    return restored_subjects


def process_subject_portraits(subject_info: Dict[str, str], config: dict) -> Dict[str, int]:

    subject_id = subject_info['subject_id']
    gt_path = subject_info['gt_path']
    restored_path = subject_info['restored_path']
    restored_name = subject_info['restored_name']

    logger.info(f"Processing subject {subject_id}: {restored_name}")


    stage_config = config['pipeline_stages'].get('29_create_combined_portraits', {})
    tolerance = stage_config.get('tolerance', 5)
    neck_selection_ratio = stage_config.get('neck_selection_ratio', 1/3)
    lower_context_ratio = stage_config.get('lower_context_ratio', 0.2)
    skip_existing = stage_config.get('skip_existing', False)
    save_visualization_portraits = stage_config.get('save_visualization_portraits', False)
    neck_source_mode = stage_config.get('neck_source_mode', 'gt')
    dataset_type = config.get('data_type', 'thuman2')

    logger.info(f"  Neck source mode: {neck_source_mode}")


    seg_grid_path = os.path.join(gt_path, "Semantic", "process", "parser", "parser-f0000.png")

    try:

        image_resolution = detect_image_resolution(gt_path, dataset_type)


        camera_names = get_camera_names_from_dataset(gt_path, dataset_type)


        camera_seg_images = parse_segmentation_grid(seg_grid_path, image_resolution, camera_names)

        logger.info(f"  GT segmentation: {len(camera_seg_images)} camera views")

    except Exception as e:
        logger.error(f"Failed to load GT segmentation for subject {subject_id}: {e}")
        return {'processed': 0, 'failed': 0, 'skipped': 0, 'warnings': 0}


    stats = {
        'processed': 0,
        'failed': 0,
        'skipped': 0,
        'warnings': 0
    }


    neck_warnings_path = os.path.join(restored_path, "neck_warnings.txt")
    neck_warnings = []


    neck_modes_to_process = []
    if neck_source_mode in ['gt', 'both']:
        neck_modes_to_process.append(('gt', 'combined_gt_neck'))
    if neck_source_mode in ['swapped', 'both']:
        neck_modes_to_process.append(('swapped', 'combined_swapped_neck'))


    for camera_idx, (camera_name, seg_image) in enumerate(zip(camera_names, camera_seg_images)):
        try:

            camera_dir = os.path.join(restored_path, camera_name)


            gt_rgb_path = os.path.join(gt_path, camera_name, "0000.jpg")
            swapped_rgb_path = os.path.join(camera_dir, "0000.jpg")

            if not os.path.exists(gt_rgb_path):
                logger.warning(f"  GT RGB not found: {camera_name}")
                stats['failed'] += 1
                continue

            if not os.path.exists(swapped_rgb_path):
                logger.warning(f"  Swapped RGB not found: {camera_name}")
                stats['failed'] += 1
                continue

            gt_rgb = np.array(Image.open(gt_rgb_path))
            swapped_rgb = np.array(Image.open(swapped_rgb_path))


            if gt_rgb.shape != swapped_rgb.shape:
                logger.error(f"  Shape mismatch {camera_name}: GT {gt_rgb.shape} != swapped {swapped_rgb.shape}")
                stats['failed'] += 1
                continue


            head_mask = create_head_mask(seg_image, tolerance=tolerance)


            gt_neck_mask, _ = create_neck_mask_from_segmentation(
                seg_image, head_mask, tolerance=tolerance, neck_selection_ratio=neck_selection_ratio
            )


            swapped_seg = None
            if neck_source_mode in ['swapped', 'both']:
                swapped_seg_path = os.path.join(camera_dir, "segmentation.png")
                if not os.path.exists(swapped_seg_path):
                    raise FileNotFoundError(
                        f"Swapped segmentation not found: {swapped_seg_path}\n"
                        f"Please run Stage 28b (segment_restored_renders) before Stage 29 with swapped neck mode.\n"
                        f"Ensure Stage 28b completed successfully for all restored subjects."
                    )
                swapped_seg = np.array(Image.open(swapped_seg_path))


            for mode_name, output_subdir_name in neck_modes_to_process:
                logger.debug(f"    Processing neck mode: {mode_name}")


                combined_dir = os.path.join(camera_dir, output_subdir_name)


                if skip_existing and os.path.exists(combined_dir):
                    combined_jpg = os.path.join(combined_dir, "combined.jpg")
                    seg_png = os.path.join(combined_dir, "segmentation.png")
                    if os.path.exists(combined_jpg) and os.path.exists(seg_png):
                        logger.debug(f"    Skipping {mode_name}: outputs already exist")
                        stats['skipped'] += 1
                        continue


                if mode_name == 'gt':
                    neck_seg_source = seg_image
                elif mode_name == 'swapped':
                    neck_seg_source = swapped_seg
                else:
                    raise ValueError(f"Unknown neck mode: {mode_name}")


                neck_mask, neck_stats = create_neck_mask_from_segmentation(
                    neck_seg_source, head_mask, tolerance=tolerance, neck_selection_ratio=neck_selection_ratio
                )


                dilated_neck_mask = None
                if lower_context_ratio > 0:
                    from scripts.head_mask_extraction.utils.neck_mask_creator import dilate_neck_mask_for_bbox
                    dilated_neck_mask = dilate_neck_mask_for_bbox(
                        neck_mask, head_mask, lower_context_ratio=lower_context_ratio
                    )


                if neck_stats['is_large_neck']:
                    neck_pixels = neck_stats['neck_pixels']
                    head_pixels = neck_stats['head_pixels']
                    ratio = neck_pixels / head_pixels if head_pixels > 0 else 0
                    warning_msg = f"Camera {camera_name} ({mode_name}): neck={neck_pixels}, head={head_pixels}, ratio={ratio:.2f}"
                    neck_warnings.append(warning_msg)
                    stats['warnings'] += 1


                if mode_name == 'gt':

                    combined_rgb = combine_rgb_images(gt_rgb, swapped_rgb, neck_mask, gt_neck_mask=None)
                elif mode_name == 'swapped':


                    combined_rgb = combine_rgb_images(gt_rgb, swapped_rgb, neck_mask, gt_neck_mask=gt_neck_mask)
                else:
                    raise ValueError(f"Unknown neck mode: {mode_name}")


                cropped_combined_rgb, cropped_seg, cropped_gt_rgb, bbox_metadata = compute_and_crop_portrait(
                    combined_rgb, seg_image, head_mask, neck_mask,
                    gt_rgb=gt_rgb,
                    dilated_neck_mask=dilated_neck_mask,
                    return_bbox=True
                )


                x = bbox_metadata['x']
                y = bbox_metadata['y']
                size = bbox_metadata['size']


                cropped_neck_mask_debug = neck_mask[y:y+size, x:x+size]


                if save_visualization_portraits:

                    cropped_swapped_rgb = swapped_rgb[y:y+size, x:x+size, :]


                    non_neck_mask = np.where(neck_mask == 255, 0, 255).astype(np.uint8)
                    cropped_non_neck_mask = non_neck_mask[y:y+size, x:x+size]

                    logger.debug(f"    Cropped visualization portraits: swapped={cropped_swapped_rgb.shape}, "
                                f"neck_mask={cropped_neck_mask_debug.shape}, non_neck_mask={cropped_non_neck_mask.shape}")
                else:
                    cropped_swapped_rgb = None
                    cropped_non_neck_mask = None


                os.makedirs(combined_dir, exist_ok=True)


                combined_jpg_path = os.path.join(combined_dir, "combined.jpg")
                Image.fromarray(cropped_combined_rgb).save(combined_jpg_path, quality=95)


                gt_jpg_path = os.path.join(combined_dir, "gt.jpg")
                Image.fromarray(cropped_gt_rgb).save(gt_jpg_path, quality=95)


                seg_png_path = os.path.join(combined_dir, "segmentation.png")
                Image.fromarray(cropped_seg).save(seg_png_path)


                neck_mask_png_path = os.path.join(combined_dir, "neck_mask.png")
                Image.fromarray(cropped_neck_mask_debug).save(neck_mask_png_path)


                cropped_gt_neck_mask = gt_neck_mask[y:y+size, x:x+size]
                gt_neck_mask_png_path = os.path.join(combined_dir, "gt_neck_mask.png")
                Image.fromarray(cropped_gt_neck_mask).save(gt_neck_mask_png_path)


                if mode_name == 'swapped':

                    cropped_swapped_seg = swapped_seg[y:y+size, x:x+size, :]
                    swapped_seg_png_path = os.path.join(combined_dir, "swapped_segmentation.png")
                    Image.fromarray(cropped_swapped_seg).save(swapped_seg_png_path)


                if save_visualization_portraits:

                    portrait_swapped_path = os.path.join(combined_dir, "portrait_swapped.jpg")
                    Image.fromarray(cropped_swapped_rgb).save(portrait_swapped_path, quality=95)


                    neck_mask_portrait_path = os.path.join(combined_dir, "neck_mask_portrait.png")
                    Image.fromarray(cropped_neck_mask_debug).save(neck_mask_portrait_path)


                    non_neck_mask_path = os.path.join(combined_dir, "non_neck_mask_portrait.png")
                    Image.fromarray(cropped_non_neck_mask).save(non_neck_mask_path)

                    logger.debug(f"    Saved {3} visualization files for {camera_name}")


                base_files = 6 if mode_name == 'swapped' else 5
                file_count = base_files + (3 if save_visualization_portraits else 0)
                logger.debug(f"    Camera {camera_name} ({mode_name}): {file_count} files saved, "
                            f"head={neck_stats['head_pixels']}, neck={neck_stats['neck_pixels']}, "
                            f"portrait={cropped_combined_rgb.shape}")


            stats['processed'] += 1

        except Exception as e:
            logger.error(f"  Failed to process camera {camera_name}: {e}")
            stats['failed'] += 1
            continue


    if neck_warnings:
        with open(neck_warnings_path, 'w') as f:
            for warning in neck_warnings:
                f.write(warning + '\n')
        logger.info(f"  Warnings: {len(neck_warnings)} large neck detections (see neck_warnings.txt)")


    total_modes = len(neck_modes_to_process)
    expected_outputs = len(camera_names) * total_modes
    logger.info(f"  Processed: {stats['processed']}/{len(camera_names)} cameras successfully")
    if total_modes > 1:
        logger.info(f"  Generated {total_modes} neck modes per camera: {[mode for mode, _ in neck_modes_to_process]}")
    if stats['skipped'] > 0:
        logger.info(f"  Skipped: {stats['skipped']} camera-mode pairs (already exist)")
    if stats['failed'] > 0:
        logger.warning(f"  Failed: {stats['failed']} cameras")

    return stats


def _normalize_subject_id(subject_id) -> str:

    return str(int(subject_id))


def stage_create_combined_portraits(config: dict, subjects: List[str]) -> None:

    logger.info("="*80)
    logger.info("STAGE 29: CREATE COMBINED PORTRAITS")
    logger.info("="*80)


    if isinstance(subjects, list) and len(subjects) > 0:

        normalized_subjects = set()
        for subj in subjects:
            if isinstance(subj, (list, tuple)):

                normalized_subjects.update(_normalize_subject_id(s) for s in subj)
            else:

                normalized_subjects.add(_normalize_subject_id(subj))

        logger.info(f"Processing subjects: {sorted(normalized_subjects)}")
    else:
        logger.warning("No subjects specified - will process all discovered subjects")
        normalized_subjects = None


    all_restored_subjects = discover_restored_subjects(config)

    if not all_restored_subjects:
        logger.warning("No restored subjects found - nothing to process")
        return


    if normalized_subjects is not None:
        restored_subjects = [
            subj_info for subj_info in all_restored_subjects
            if _normalize_subject_id(subj_info['subject_id']) in normalized_subjects
        ]

        if not restored_subjects:
            logger.warning(f"None of the specified subjects found in restored subjects")
            logger.info(f"  Specified: {sorted(normalized_subjects)}")
            logger.info(f"  Available: {[s['subject_id'] for s in all_restored_subjects]}")
            return

        logger.info(f"Filtered to {len(restored_subjects)}/{len(all_restored_subjects)} restored subjects matching specified subjects")
    else:
        restored_subjects = all_restored_subjects


    total_stats = {
        'total_subjects': len(restored_subjects),
        'total_processed': 0,
        'total_failed': 0,
        'total_skipped': 0,
        'total_warnings': 0
    }

    for idx, subject_info in enumerate(restored_subjects, 1):
        logger.info(f"\n[Subject {idx}/{len(restored_subjects)}] Processing: {subject_info['subject_id']}")

        stats = process_subject_portraits(subject_info, config)

        total_stats['total_processed'] += stats['processed']
        total_stats['total_failed'] += stats['failed']
        total_stats['total_skipped'] += stats['skipped']
        total_stats['total_warnings'] += stats['warnings']


    logger.info(f"\n{'='*80}")
    logger.info("COMBINED PORTRAIT CREATION COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"Subjects processed: {total_stats['total_subjects']}")
    logger.info(f"Total cameras processed: {total_stats['total_processed']}")
    if total_stats['total_skipped'] > 0:
        logger.info(f"Total cameras skipped: {total_stats['total_skipped']}")
    if total_stats['total_failed'] > 0:
        logger.warning(f"Total cameras failed: {total_stats['total_failed']}")
    if total_stats['total_warnings'] > 0:
        logger.info(f"Total neck warnings: {total_stats['total_warnings']}")
    logger.info(f"{'='*80}\n")
