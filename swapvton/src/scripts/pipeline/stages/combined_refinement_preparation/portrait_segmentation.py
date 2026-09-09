import os
from pathlib import Path
from typing import List
from loguru import logger

from scripts.pipeline.sampling.subject_discovery import discover_swapped_directories
from scripts.pipeline.execution.subprocess_runner import run_command
from scripts.pipeline.execution.env_utils import build_python_command
from scripts.pipeline.execution.debug_support import get_debug_port


def stage_refine_portrait_body_segmentation(config: dict, subjects: List[str], debug_subprocess: bool = False):

    logger.info("="*80)
    logger.info("Stage 31: Refine Portrait Body Segmentation")
    logger.info("="*80)


    stage_config = config['pipeline_stages'].get('31_refine_portrait_body_segmentation', {})
    skip_missing_portraits = stage_config.get('skip_missing_portraits', False)


    avatarrex_dir = Path(config['paths']['avatarrex_output'])
    head_swapped_renders_dir = avatarrex_dir / 'head_swapped_renders'
    fourd_dress_root = config['paths'].get('4d_dress_root')
    checkpoint_dir = os.path.join(fourd_dress_root, '4dhumanparsing', 'checkpoints', 'graphonomy')
    conda_env = config['conda_envs'].get('fourd_dress', '4ddress')
    script_path = os.path.join('4dhumanparsing', 'scripts', 'segment_portrait_validation_data.py')

    logger.info(f"Configuration:")
    logger.info(f"  - Skip missing portraits: {skip_missing_portraits}")
    logger.info(f"  - Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  - Head swapped renders: {head_swapped_renders_dir}")


    if not os.path.exists(str(Path(fourd_dress_root) / script_path)):
        raise FileNotFoundError(f"Segmentation script not found: {script_path}")


    swapped_dirs = discover_swapped_directories(config, subjects)
    if not swapped_dirs:
        logger.warning("No swapped rendering directories found for specified subjects")
        return

    logger.info(f"Found {len(swapped_dirs)} swapped rendering directories")


    for swapped_dir in swapped_dirs:
        logger.info(f"\nProcessing: {swapped_dir.name}")


        camera_dirs = sorted([d for d in swapped_dir.iterdir()
                            if d.is_dir() and (d / 'validation_data').exists()])

        if not camera_dirs:
            logger.warning(f"No validation_data found in any camera directories")
            continue

        logger.info(f"Found {len(camera_dirs)} cameras with validation_data")


        for cam_dir in camera_dirs:
            cam_name = cam_dir.name
            validation_data_dir = str(cam_dir / 'validation_data')

            logger.info(f"  Camera {cam_name}: processing validation_data")


            python_prefix = build_python_command(
                script_path,
                conda_env_name=conda_env
            )
            cmd = python_prefix + [
                script_path,
                '--input_dir', validation_data_dir, '--checkpoint_dir', checkpoint_dir, '--device', 'cuda:0'
            ] + (['--skip_missing'] if skip_missing_portraits else [])


            debug_port = get_debug_port('refine_portrait_body_segmentation') if debug_subprocess else None

            try:

                logger.debug(f"Executing: {cmd}")
                run_command(cmd, cwd=fourd_dress_root, debug_port=debug_port)
                logger.info(f"  ✓ {cam_name}: segmentation complete")

            except Exception as e:
                logger.error(f"Failed to run script for {cam_name}: {e}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 31 complete: Portrait body segmentation refined")
    logger.info(f"Output files: segmentation_body_portrait_from_portrait.png in each validation_data mode")
    logger.info(f"{'='*80}")
