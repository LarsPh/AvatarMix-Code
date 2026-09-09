from pathlib import Path
from loguru import logger
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
import os
import re


def stage_segment_restored_renders(config, subjects, debug_subprocess=False):

    logger.info("\n" + "="*80)
    logger.info("STAGE 28b: SEGMENT RESTORED (SECOND-SWAP) RENDERS")
    logger.info("="*80)


    avatarrex_dir = config['paths']['avatarrex_output']
    head_swapped_back_renders_dir = os.path.join(avatarrex_dir, 'head_swapped_back_renders')
    fourd_dress_root = config['paths'].get('4d_dress_root')
    checkpoint_dir = os.path.join(fourd_dress_root, '4dhumanparsing', 'checkpoints', 'graphonomy')
    conda_env = config['conda_envs'].get('fourd_dress', '4ddress')


    stage_cfg = config['pipeline_stages'].get('28b_segment_restored_renders', {})
    resize_to = stage_cfg.get('resize_to', 512)


    script_path = os.path.join('4dhumanparsing', 'scripts', 'segment_swapvton_renders.py')


    if not checkpoint_dir:
        logger.error("Graphonomy checkpoint path not configured in 'paths.4d_dress_root'")
        logger.error("Please add to config YAML:")
        logger.error("  paths:")
        logger.error("    4d_dress_root: '${AVATARMIX_ROOT}/4d-dress'")
        raise ValueError("Missing 4d_dress_root in configuration")

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Graphonomy checkpoint directory not found: {checkpoint_dir}")
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if not os.path.exists(head_swapped_back_renders_dir):
        logger.warning(f"Head swapped back renders directory not found: {head_swapped_back_renders_dir}")
        logger.warning("Please run Stage 25 (render_swapped_back) first")
        return

    logger.info(f"Configuration:")
    logger.info(f"  Renders directory: {head_swapped_back_renders_dir}")
    logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  Conda environment: {conda_env}")
    logger.info(f"  4D-Dress script: {script_path}")
    logger.info(f"  Resize to: {resize_to}x{resize_to}" if resize_to else "  Resize: disabled (original size)")


    restored_dirs = []

    if not os.path.exists(head_swapped_back_renders_dir):
        logger.warning(f"No restored directories found in: {head_swapped_back_renders_dir}")
        return


    for item in os.listdir(head_swapped_back_renders_dir):
        item_path = os.path.join(head_swapped_back_renders_dir, item)
        if not os.path.isdir(item_path):
            continue


        if not item.startswith('restored_'):
            continue


        match = re.match(r'restored_(\d+)_from_', item)
        if not match:
            logger.debug(f"Skipping directory (no subject ID match): {item}")
            continue

        restored_subject_id = match.group(1)


        normalized_subjects = [str(int(s)) if isinstance(s, str) and s.isdigit() else str(s) for s in subjects]
        restored_subject_normalized = str(int(restored_subject_id))

        if restored_subject_normalized in normalized_subjects:
            restored_dirs.append(item_path)
            logger.debug(f"Found restored directory for subject {restored_subject_id}: {item}")
        else:
            logger.debug(f"Skipping directory (subject {restored_subject_id} not in list): {item}")

    if not restored_dirs:
        logger.warning("No restored directories found for specified subject pairs")
        logger.warning(f"Head swapped back renders directory: {head_swapped_back_renders_dir}")
        logger.info(f"Subject list: {subjects}")
        logger.info(f"Normalized subjects: {[str(int(s)) if isinstance(s, str) and s.isdigit() else str(s) for s in subjects]}")
        return

    logger.info(f"Found {len(restored_dirs)} restored directories to segment")


    success_count = 0
    error_count = 0

    for restored_dir in restored_dirs:
        restored_name = os.path.basename(restored_dir)
        logger.info(f"\nProcessing restored directory: {restored_name}")


        camera_dirs = [d for d in os.listdir(restored_dir)
                      if os.path.isdir(os.path.join(restored_dir, d))]

        if not camera_dirs:
            logger.warning(f"  No camera directories found in {restored_name}")
            error_count += 1
            continue

        logger.info(f"  Found {len(camera_dirs)} camera directories")


        python_prefix = build_python_command(
            fourd_dress_root,
            conda_env_name=conda_env
        )

        cmd = python_prefix + [
            script_path,
            '--input_root_dir', restored_dir,
            '--target_subdir', '',
            '--checkpoint_dir', checkpoint_dir,
            '--device', 'cuda:0',
            '--resize_to', str(resize_to) if resize_to else '0',
            '--skip_existing'
        ]


        try:


            if debug_subprocess:
                logger.warning("Debug subprocess not supported for external project (4D-Dress)")
                logger.warning("Debug the script directly in 4D-Dress environment if needed")

            logger.debug(f"Executing: {' '.join(cmd)}")
            run_command(cmd, cwd=fourd_dress_root)

            logger.info(f"  ✓ Successfully segmented {restored_name}")
            success_count += 1

        except Exception as e:
            logger.error(f"  ✗ Error segmenting {restored_name}: {e}")
            error_count += 1


    logger.info("\n" + "="*80)
    logger.info("STAGE 28b COMPLETED")
    logger.info("="*80)
    logger.info(f"Total restored directories processed: {len(restored_dirs)}")
    logger.info(f"  Successful: {success_count}")
    if error_count > 0:
        logger.warning(f"  Errors: {error_count}")
    logger.info("="*80 + "\n")

    if error_count == len(restored_dirs):
        raise RuntimeError(f"All {len(restored_dirs)} restored directories failed segmentation")
