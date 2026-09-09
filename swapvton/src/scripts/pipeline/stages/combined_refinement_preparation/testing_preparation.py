import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict
from loguru import logger

from scripts.head_mask_extraction.utils.mask_creator import create_head_mask
from scripts.head_mask_extraction.utils.body_mask_creator import create_body_mask
from scripts.head_mask_extraction.utils.testing_composition import (
    create_neck_mask_with_mode, combine_three_way_rgb
)
from scripts.head_mask_extraction.utils.image_combination import compute_and_crop_portrait
from scripts.head_mask_extraction.utils.neck_mask_creator import dilate_neck_mask_for_bbox
from scripts.head_mask_extraction.utils.testing_portrait import (
    crop_segmentations_with_bbox, crop_masks_with_bbox, crop_rgb_images_with_bbox,
    save_validation_outputs, verify_outputs_exist
)
from scripts.head_mask_extraction.utils.grid_parser import (
    parse_segmentation_grid, detect_image_resolution, get_camera_names_from_dataset
)
from scripts.pipeline.sampling.subject_discovery import first_swap_reshape_enabled, discover_swapped_directories


def stage_create_validation_data_portraits(config: Dict, subjects: List[str], debug_subprocess: bool = False):

    logger.info("="*80)
    logger.info("Stage 30: Create Validation Data Portraits")
    logger.info("="*80)


    stage_config = config['pipeline_stages']['30_create_validation_data_portraits']
    tolerance = stage_config.get('tolerance', 5)
    neck_selection_ratio = stage_config.get('neck_selection_ratio', 0.333)
    neck_dilation_ratio = stage_config.get('neck_dilation_ratio', 0.05)
    lower_context_ratio = stage_config.get('lower_context_ratio', 0.2)
    neck_modes = stage_config.get('neck_modes', ['first_swap_only', 'donator_only', 'union'])
    skip_existing = stage_config.get('skip_existing', False)
    save_visualization_portraits = stage_config.get('save_visualization_portraits', True)


    use_no_sam = stage_config.get('use_label_no_sam', False)


    head_aligned_subdir = 'head_aligned_no_sam' if use_no_sam else 'head_aligned'
    body_donator_subdir = 'full_body_donator_no_sam' if use_no_sam else 'full_body_donator'


    avatarrex_dir = Path(config['paths']['avatarrex_output'])
    head_swapped_renders_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_renders'

    logger.info(f"Configuration:")
    logger.info(f"  - Tolerance: {tolerance}")
    logger.info(f"  - Neck selection ratio: {neck_selection_ratio}")
    logger.info(f"  - Neck dilation ratio: {neck_dilation_ratio}")
    logger.info(f"  - Lower context ratio: {lower_context_ratio}")
    logger.info(f"  - Neck modes: {neck_modes}")
    logger.info(f"  - Skip existing: {skip_existing}")
    logger.info(f"  - Save visualization portraits: {save_visualization_portraits}")
    logger.info(f"  - Use no_sam aligned renders: {use_no_sam}")
    logger.info(f"  - Head aligned subdirectory: {head_aligned_subdir}")
    logger.info(f"  - Body donator subdirectory: {body_donator_subdir}")
    logger.info(f"  - Avatar-Rex directory: {avatarrex_dir}")
    logger.info(f"  - Head swapped renders directory: {head_swapped_renders_dir}")


    for subject_id in subjects:
        user_id, model_id = subject_id, (set(subjects) - {subject_id}).pop()
        logger.info(f"\nProcessing subject pair: {user_id} (user) - {model_id} (model)")


        swapped_dirs = discover_swapped_directories(config, [user_id, model_id])

        if not swapped_dirs:
            logger.warning(f"No swapped directories found for subject pair: {subject_id}")
            continue

        swapped_dir = swapped_dirs[0]
        logger.info(f"Found swapped directory: {swapped_dir.name}")


        user_data_dir = avatarrex_dir / f"{user_id.zfill(4)}"
        if not user_data_dir.exists():
            logger.error(f"User data directory not found: {user_data_dir}")
            continue


        gt_seg_grid_path = user_data_dir / "Semantic" / "process" / "parser" / "parser-f0000.png"
        if not gt_seg_grid_path.exists():
            logger.error(f"GT segmentation grid not found: {gt_seg_grid_path}")
            continue


        image_resolution = detect_image_resolution(str(user_data_dir), dataset_type="thuman2")
        camera_names = get_camera_names_from_dataset(str(user_data_dir), dataset_type="thuman2")

        logger.info(f"Detected image resolution: {image_resolution}, {len(camera_names)} cameras")


        gt_seg_list = parse_segmentation_grid(str(gt_seg_grid_path), image_resolution, camera_names)
        logger.info(f"Parsed GT segmentation grid: {len(gt_seg_list)} camera views")


        for cam_idx, cam_name in enumerate(camera_names):
            logger.info(f"\n  Processing camera {cam_idx+1}/{len(camera_names)}: {cam_name}")


            gt_seg = gt_seg_list[cam_idx]


            gt_rgb_path = user_data_dir / cam_name / "0000.jpg"
            if not gt_rgb_path.exists():
                logger.warning(f"GT RGB not found: {gt_rgb_path}, skipping camera")
                continue

            gt_rgb = cv2.imread(str(gt_rgb_path))
            gt_rgb = cv2.cvtColor(gt_rgb, cv2.COLOR_BGR2RGB)


            first_swap_rgb_path = swapped_dir / cam_name / head_aligned_subdir / "0000.png"
            first_swap_seg_path = swapped_dir / cam_name / head_aligned_subdir / "segmentation.png"

            if not first_swap_rgb_path.exists() or not first_swap_seg_path.exists():
                if use_no_sam:
                    logger.error(f"use_label_no_sam=true but _no_sam aligned renders not found for {cam_name}")
                    logger.error(f"Expected paths: {first_swap_rgb_path}, {first_swap_seg_path}")
                    raise FileNotFoundError(f"Missing _no_sam aligned renders - ensure Stages 26, 27 ran with use_label_no_sam=true")
                logger.warning(f"First-swap data not found for {cam_name}, skipping")
                continue

            first_swap_rgb = cv2.imread(str(first_swap_rgb_path))
            first_swap_rgb = cv2.cvtColor(first_swap_rgb, cv2.COLOR_BGR2RGB)

            first_swap_seg = cv2.imread(str(first_swap_seg_path))
            first_swap_seg = cv2.cvtColor(first_swap_seg, cv2.COLOR_BGR2RGB)


            donator_rgb_path = swapped_dir / cam_name / body_donator_subdir / "0000.png"
            donator_seg_path = swapped_dir / cam_name / body_donator_subdir / "segmentation.png"

            if not donator_rgb_path.exists() or not donator_seg_path.exists():
                if use_no_sam:
                    logger.error(f"use_label_no_sam=true but _no_sam aligned renders not found for {cam_name}")
                    logger.error(f"Expected paths: {donator_rgb_path}, {donator_seg_path}")
                    raise FileNotFoundError(f"Missing _no_sam aligned renders - ensure Stages 26, 27 ran with use_label_no_sam=true")
                logger.warning(f"Donator data not found for {cam_name}, skipping")
                continue

            donator_rgb = cv2.imread(str(donator_rgb_path))
            donator_rgb = cv2.cvtColor(donator_rgb, cv2.COLOR_BGR2RGB)

            donator_seg = cv2.imread(str(donator_seg_path))
            donator_seg = cv2.cvtColor(donator_seg, cv2.COLOR_BGR2RGB)


            head_mask = create_head_mask(gt_seg, tolerance=tolerance)
            logger.debug(f"Created head mask: {np.sum(head_mask > 0)} pixels")


            donator_body_mask = create_body_mask(donator_seg, tolerance=tolerance)
            logger.debug(f"Created donator body mask: {np.sum(donator_body_mask > 0)} pixels")


            for neck_mode in neck_modes:
                logger.info(f"    Processing neck mode: {neck_mode}")


                output_dir = swapped_dir / cam_name / "validation_data" / f"{neck_mode}_neck"
                if skip_existing and verify_outputs_exist(output_dir):
                    logger.info(f"    Skipping {neck_mode}: outputs already exist")
                    continue


                neck_mask, neck_stats = create_neck_mask_with_mode(
                    first_swap_seg, donator_seg, head_mask, donator_body_mask,
                    mode=neck_mode, tolerance=tolerance, neck_selection_ratio=neck_selection_ratio,
                    neck_dilation_ratio=neck_dilation_ratio
                )
                logger.debug(f"Created {neck_mode} neck mask: {neck_stats['neck_pixels']} pixels")


                fullbody_combined = combine_three_way_rgb(
                    gt_rgb, first_swap_rgb, donator_rgb,
                    head_mask, neck_mask, donator_body_mask
                )


                if lower_context_ratio > 0:
                    dilated_neck_mask = dilate_neck_mask_for_bbox(
                        neck_mask, head_mask, lower_context_ratio
                    )
                else:
                    dilated_neck_mask = None


                portrait_combined, portrait_seg_gt, portrait_gt, bbox_metadata = compute_and_crop_portrait(
                    fullbody_combined, gt_seg, head_mask, neck_mask, gt_rgb,
                    dilated_neck_mask=dilated_neck_mask, return_bbox=True
                )


                (segmentation_gt_portrait,
                 segmentation_first_swap_portrait,
                 segmentation_body_portrait) = crop_segmentations_with_bbox(
                    gt_seg, first_swap_seg, donator_seg, bbox_metadata
                )


                (head_mask_portrait,
                 neck_mask_portrait,
                 body_mask_portrait) = crop_masks_with_bbox(
                    head_mask, neck_mask, donator_body_mask, bbox_metadata
                )


                if save_visualization_portraits:
                    portrait_first_swap, portrait_donator = crop_rgb_images_with_bbox(
                        first_swap_rgb, donator_rgb, bbox_metadata
                    )
                    logger.debug(f"Cropped visualization portraits: first_swap={portrait_first_swap.shape}, "
                                f"donator={portrait_donator.shape}")
                else:
                    portrait_first_swap, portrait_donator = None, None


                save_validation_outputs(
                    output_dir, neck_mode,
                    fullbody_combined, donator_body_mask,
                    portrait_combined, portrait_gt,
                    head_mask_portrait, neck_mask_portrait, body_mask_portrait,
                    segmentation_gt_portrait, segmentation_first_swap_portrait, segmentation_body_portrait,
                    bbox_metadata,
                    portrait_first_swap=portrait_first_swap,
                    portrait_donator=portrait_donator
                )

                file_count = 11 + (2 if save_visualization_portraits else 0)
                logger.info(f"    Completed {neck_mode}: saved {file_count} files to {output_dir}")

    logger.info("="*80)
    logger.info("Stage 30 completed successfully")
    logger.info("="*80)
