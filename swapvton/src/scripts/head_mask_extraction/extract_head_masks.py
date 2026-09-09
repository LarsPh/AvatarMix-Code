#!/usr/bin/env python3
"""
Extract head masks from segmentation grids for swapped or ground-truth avatars.
Use --data_root for a single subject, or --batch_root for a dataset directory.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger


project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.scripts.head_mask_extraction.utils.grid_parser import (
    parse_segmentation_grid,
    detect_image_resolution,
    get_camera_names_from_dataset,
    validate_camera_count
)
from src.scripts.head_mask_extraction.utils.mask_creator import (
    create_head_mask,
    validate_head_mask,
    get_head_mask_stats
)
from src.scripts.head_mask_extraction.utils.rgb_processor import (
    apply_mask_to_rgb,
    create_masked_rgb_batch
)


def setup_logging(verbose: bool = False) -> None:

    log_level = "DEBUG" if verbose else "INFO"


    logger.remove()


    logger.add(
        sys.stderr,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>"
    )


def find_segmentation_grid(data_root: str) -> str:

    seg_path = os.path.join(data_root, "Semantic", "process", "parser", "parser-f0000.png")

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation grid not found: {seg_path}")

    logger.debug(f"Found segmentation grid: {seg_path}")
    return seg_path


def process_subject_data(data_root: str, data_type: str, tolerance: int = 5,
                        background_color: Tuple[int, int, int] = (0, 0, 0),
                        dataset_type: str = "thuman2") -> None:

    logger.info(f"Processing subject data: {data_root}")
    logger.info(f"Data type: {data_type}, dataset type: {dataset_type}, tolerance: {tolerance}")


    seg_grid_path = find_segmentation_grid(data_root)


    logger.info("Detecting image resolution from dataset...")
    image_resolution = detect_image_resolution(data_root, dataset_type)


    logger.info("Discovering camera names from dataset...")
    camera_names = get_camera_names_from_dataset(data_root, dataset_type)


    logger.info("Parsing segmentation grid...")
    camera_images = parse_segmentation_grid(seg_grid_path, image_resolution, camera_names)


    validate_camera_count(len(camera_images), camera_names)


    if skip_existing and len(camera_names) > 0:
        first_camera_mask = os.path.join(data_root, camera_names[0], "mask", "head", "0000.png")
        if os.path.exists(first_camera_mask):
            logger.info(f"Head masks already exist, skipping: {data_root}")
            return


    logger.info(f"Creating head masks for {len(camera_images)} camera views...")

    masks = []
    processed_count = 0
    failed_count = 0

    for i, (camera_image, camera_name) in enumerate(zip(camera_images, camera_names)):
        try:

            head_mask = create_head_mask(camera_image, tolerance=tolerance)


            if not validate_head_mask(head_mask, min_pixels=50):
                logger.warning(f"Invalid head mask for camera {camera_name}")
                failed_count += 1
                continue


            camera_dir = os.path.join(data_root, camera_name)
            mask_dir = os.path.join(camera_dir, "mask", "head")
            os.makedirs(mask_dir, exist_ok=True)


            mask_path = os.path.join(mask_dir, "0000.png")
            mask_image = Image.fromarray(head_mask)
            mask_image.save(mask_path)

            masks.append(head_mask)
            processed_count += 1


            if i < 5 or (i + 1) % 20 == 0:
                stats = get_head_mask_stats(head_mask)
                logger.debug(f"Camera {camera_name}: {stats['head_pixels']} head pixels "
                           f"({stats['head_percentage']:.1f}%)")

        except Exception as e:
            logger.error(f"Failed to process camera {camera_name}: {e}")
            failed_count += 1

    logger.info(f"Head mask extraction complete: {processed_count} successful, {failed_count} failed")


    if data_type in ['gt', 'both'] and processed_count > 0:
        logger.info("Applying masks to RGB images...")

        try:

            valid_camera_names = camera_names[:len(masks)]
            create_masked_rgb_batch(
                data_root=data_root,
                masks=masks,
                camera_names=valid_camera_names,
                background_color=background_color
            )
            logger.info("RGB masking complete")

        except Exception as e:
            logger.error(f"Failed to process RGB images: {e}")


def find_subjects_for_batch_processing(batch_root: str, data_type: str) -> List[str]:

    subjects = []

    if data_type in ['gt', 'both']:

        for item in os.listdir(batch_root):
            item_path = os.path.join(batch_root, item)
            if os.path.isdir(item_path) and item.isdigit():

                seg_path = os.path.join(item_path, "Semantic", "process", "parser", "parser-f0000.png")
                if os.path.exists(seg_path):
                    subjects.append(item_path)
                    logger.debug(f"Found GT subject: {item}")

    if data_type in ['swapped', 'both']:

        swapped_root = os.path.join(batch_root, "head_swapped_renders_refined")
        if os.path.exists(swapped_root):
            for item in os.listdir(swapped_root):
                if item.startswith("swapped_"):
                    item_path = os.path.join(swapped_root, item)
                    if os.path.isdir(item_path):

                        seg_path = os.path.join(item_path, "Semantic", "process", "parser", "parser-f0000.png")
                        if os.path.exists(seg_path):
                            subjects.append(item_path)
                            logger.debug(f"Found swapped subject: {item}")

    logger.info(f"Found {len(subjects)} subjects for batch processing")
    return subjects


def main():


    parser = argparse.ArgumentParser(
        description="Extract head masks from segmentation grids for SwapVTON pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )


    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data_root",
        type=str,
        help="Root directory of single subject data"
    )
    group.add_argument(
        "--batch_root",
        type=str,
        help="Root directory for batch processing multiple subjects"
    )


    parser.add_argument(
        "--data_type",
        choices=['swapped', 'gt', 'both'],
        default='swapped',
        help="Type of data to process (default: swapped)"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="thuman2",
        help="Type of dataset (default: thuman2)"
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Color matching tolerance (0-255, default: 5)"
    )
    parser.add_argument(
        "--background_color",
        type=int,
        nargs=3,
        default=[0, 0, 0],
        help="Background color for masked RGB images (R G B, default: 0 0 0)"
    )


    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )

    args = parser.parse_args()


    setup_logging(args.verbose)

    logger.info("=" * 80)
    logger.info("SwapVTON Head Mask Extraction")
    logger.info("=" * 80)

    try:
        background_color = tuple(args.background_color)

        if args.data_root:

            if args.dry_run:
                logger.info(f"[DRY RUN] Would process: {args.data_root}")
                seg_path = find_segmentation_grid(args.data_root)
                logger.info(f"[DRY RUN] Segmentation grid found: {seg_path}")
            else:
                process_subject_data(
                    data_root=args.data_root,
                    data_type=args.data_type,
                    tolerance=args.tolerance,
                    background_color=background_color,
                    dataset_type=args.dataset_type
                )

        elif args.batch_root:

            subjects = find_subjects_for_batch_processing(args.batch_root, args.data_type)

            if not subjects:
                logger.warning("No subjects found for batch processing")
                return

            if args.dry_run:
                logger.info(f"[DRY RUN] Would process {len(subjects)} subjects:")
                for subject in subjects[:10]:
                    logger.info(f"[DRY RUN]   {subject}")
                if len(subjects) > 10:
                    logger.info(f"[DRY RUN]   ... and {len(subjects) - 10} more")
            else:
                logger.info(f"Starting batch processing of {len(subjects)} subjects...")

                for i, subject_path in enumerate(subjects, 1):
                    logger.info(f"Processing subject {i}/{len(subjects)}: {os.path.basename(subject_path)}")

                    try:
                        process_subject_data(
                            data_root=subject_path,
                            data_type=args.data_type,
                            tolerance=args.tolerance,
                            background_color=background_color,
                            dataset_type=args.dataset_type
                        )
                    except Exception as e:
                        logger.error(f"Failed to process subject {subject_path}: {e}")
                        continue

                logger.info("Batch processing complete")

        logger.info("Head mask extraction finished successfully!")

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Head mask extraction failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
