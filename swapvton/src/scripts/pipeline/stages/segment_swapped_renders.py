from pathlib import Path
from loguru import logger
from pipeline.core.file_operations import find_swapped_directories
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.sampling.subject_discovery import get_reshape_suffix_from_config
import os


def stage_segment_swapped_renders(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 28: Segment Swapped Renders ---")


    cfg = config['pipeline_stages'].get('28_segment_swapped_renders', {})
    avatarrex_dir = config['paths']['avatarrex_output']
    head_swapped_renders_dir = os.path.join(avatarrex_dir, 'head_swapped_renders')
    fourd_dress_root = config['paths'].get('4d_dress_root')
    checkpoint_dir = os.path.join(fourd_dress_root, '4dhumanparsing', 'checkpoints', 'graphonomy')
    conda_env = config['conda_envs'].get('fourd_dress', '4ddress')


    stage_cfg = config['pipeline_stages'].get('28_segment_swapped_renders', {})
    resize_to = stage_cfg.get('resize_to', 512)


    script_path = os.path.join('4dhumanparsing', 'scripts', 'segment_swapvton_renders.py')


    if not checkpoint_dir:
        logger.error("Graphonomy checkpoint path not configured in 'paths.graphonomy_checkpoint'")
        logger.error("Please add to config YAML:")
        logger.error("  paths:")
        logger.error("    graphonomy_checkpoint: '${AVATARMIX_ROOT}/4d-dress/4dhumanparsing/checkpoints/graphonomy'")
        raise ValueError("Missing graphonomy_checkpoint in configuration")

    if not os.path.exists(checkpoint_dir):
        logger.error(f"Graphonomy checkpoint directory not found: {checkpoint_dir}")
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if not os.path.exists(head_swapped_renders_dir):
        logger.warning(f"Head swapped renders directory not found: {head_swapped_renders_dir}")
        logger.warning("Please run the swapped render stage first (Stage 13 for normal, Stage 26/27 for head_aligned)")
        return

    logger.info(f"Configuration:")
    logger.info(f"  Renders directory: {head_swapped_renders_dir}")
    logger.info(f"  Checkpoint directory: {checkpoint_dir}")
    logger.info(f"  Conda environment: {conda_env}")
    logger.info(f"  4D-Dress script: {script_path}")
    logger.info(f"  Resize to: {resize_to}x{resize_to}" if resize_to else "  Resize: disabled (original size)")


    swapped_dirs = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            pair_dirs = find_swapped_directories(config, subject_a, subject_b)

            if not pair_dirs:
                logger.warning(f"No swapped directories found for pair {subject_a} -> {subject_b}")
                reshape_suffix = get_reshape_suffix_from_config(config)
                expected_base = f"A{subject_a}_B{subject_b}"
                expected_with_suffix = f"{expected_base}{reshape_suffix}" if reshape_suffix else expected_base
                logger.info(f"  Expected: {os.path.join(head_swapped_renders_dir, expected_with_suffix)}")
                continue


            for swapped_dir in pair_dirs:

                swapped_name = os.path.basename(swapped_dir)


                pass


    if not os.path.exists(head_swapped_renders_dir):
        logger.warning(f"No rendered directories found in: {head_swapped_renders_dir}")
        return


    from pipeline.core.config import get_refinement_render_mode
    render_mode = get_refinement_render_mode(config, "28_segment_swapped_renders")


    render_dirs = []
    for item in os.listdir(head_swapped_renders_dir):
        item_path = os.path.join(head_swapped_renders_dir, item)
        if not os.path.isdir(item_path):
            continue


        if not item.startswith('swapped_'):
            continue


        import re
        match = re.match(r'^swapped_(.+?)head_on_(.+?)body', item)
        if not match:
            continue

        head_id = match.group(1)
        body_id = match.group(2)


        for i, subject_a in enumerate(subjects):
            for j, subject_b in enumerate(subjects):
                if i == j:
                    continue


                if head_id == str(subject_a) and body_id == str(subject_b):
                    render_dirs.append(item_path)
                    break

    if not render_dirs:
        logger.warning("No rendered directories found for specified subject pairs")
        logger.warning(f"Head swapped renders directory: {head_swapped_renders_dir}")
        logger.info(f"Subject pairs: {[(subjects[i], subjects[j]) for i in range(len(subjects)) for j in range(len(subjects)) if i != j]}")
        return

    logger.info(f"Found {len(render_dirs)} rendered directories to segment")


    if render_mode == "head_aligned":
        if cfg.get('use_label_no_sam', False):
            target_subdirs = ['head_aligned_no_sam', 'full_body_donator_no_sam']
        else:
            target_subdirs = ['head_aligned', 'full_body_donator']
    else:
        target_subdirs = ['']


    for render_dir in render_dirs:
        render_name = os.path.basename(render_dir)
        logger.info(f"Processing rendered directory: {render_name}")


        cam_dirs = [d for d in os.listdir(render_dir) if os.path.isdir(os.path.join(render_dir, d))]
        if not cam_dirs:
            logger.warning(f"  No camera directories found under: {render_dir}")
            continue


        for target_subdir in target_subdirs:
            logger.info(f"  Segmenting target_subdir='{target_subdir}' (render_mode={render_mode})...")


            python_prefix = build_python_command(
                fourd_dress_root,
                conda_env_name=conda_env
            )
            cmd = python_prefix + [
                script_path,
                '--input_root_dir', render_dir,
                '--target_subdir', target_subdir,
                '--checkpoint_dir', checkpoint_dir,
                '--device', 'cuda:0',
                '--resize_to', str(resize_to) if resize_to else '0',

            ]


            try:


                if debug_subprocess:
                    logger.warning("Debug subprocess not supported for external project (4D-Dress)")
                    logger.warning("Debug the script directly in 4D-Dress environment if needed")

                logger.debug(f"Executing: {cmd}")
                run_command(cmd, cwd=fourd_dress_root)
                logger.info(f"  Successfully segmented {target_subdir}")

            except Exception as e:
                logger.error(f"  Error segmenting {target_subdir}: {e}")
                raise

        logger.info(f"  Completed segmentation for {render_name}")

    logger.info("="*60)
    logger.info("Stage 28 complete: All rendered images segmented")
    logger.info(f"  Total directories processed: {len(render_dirs)}")
    logger.info(f"  Subdirectories per directory: {len(target_subdirs)}")
    logger.info("="*60)
