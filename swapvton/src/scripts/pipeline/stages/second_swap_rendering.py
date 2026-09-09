from loguru import logger
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.sampling.subject_discovery import get_padded_subject_id, get_reshape_suffix_from_config, second_swap_reshape_enabled
from pipeline.core.file_operations import copy_calibration_file_to_dataset, copy_non_refined_masks, extract_masks_from_refined_directory
import os
import glob
import fnmatch


def stage_render_swapped_back_gaussians(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 24: Render Swapped-Back Gaussians ---")

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    conda_env = config['conda_envs']['splatting']


    render_cfg = config['pipeline_stages'].get('13_render_swapped_gaussians', {})
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']

    render_splits = render_cfg.get('render_splits', 'train,test,val')
    frame_id = render_cfg.get('frame_id', 0)


    reshape_suffix = get_reshape_suffix_from_config(config)
    swapped_back_dirname = f"swapped_back{reshape_suffix}"
    swapped_back_base_dir = os.path.join(avatarrex_dir, swapped_back_dirname)

    logger.info(f"Looking for swapped-back directory: {swapped_back_base_dir}")

    if not os.path.exists(swapped_back_base_dir):
        logger.error(f"Error: Swapped-back directory not found: {swapped_back_base_dir}")
        logger.error("Please run Stage 23 (swap_back) first.")
        return

    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            ply_pattern = os.path.join(swapped_back_base_dir, f"restored_{int(subject_a):04d}_from_swapped_{int(subject_a):04d}head_on_{int(subject_b):04d}body_*_swap_back.ply")
            restored_ply_files = glob.glob(ply_pattern)

            if not restored_ply_files:
                logger.warning(f"Warning: No restored PLY files found in: {swapped_back_base_dir}")
                logger.warning(f"Expected files matching pattern: {ply_pattern}")
                return


            reshaped_ply_files = [ply_file for ply_file in restored_ply_files if "reshaped_" in ply_file and "_with_color_transfer" in ply_file]
            non_reshaped_ply_files = [ply_file for ply_file in restored_ply_files if "reshaped_" not in ply_file and "_with_color_transfer" in ply_file]
            reshape_enabled = second_swap_reshape_enabled(config)
            if reshape_enabled:
                restored_ply_files = reshaped_ply_files
            else:
                restored_ply_files = non_reshaped_ply_files

            logger.info(f"Found {len(restored_ply_files)} restored PLY files for rendering")


            for ply_file in restored_ply_files:
                ply_filename = os.path.basename(ply_file)
                ply_name = os.path.splitext(ply_filename)[0]

                logger.info(f"Rendering restored PLY: {ply_filename}")


                if ply_filename.startswith("restored_"):

                    parts = ply_filename.split("_")
                    if len(parts) >= 2:
                        restored_subject_id = parts[1]
                    else:
                        logger.warning(f"Warning: Cannot extract subject ID from filename: {ply_filename}")
                        continue
                else:
                    logger.warning(f"Warning: Unexpected filename format: {ply_filename}")
                    continue


                output_base_dir = os.path.join(avatarrex_dir, "head_swapped_back_renders")
                output_dir = os.path.join(output_base_dir, ply_name)
                os.makedirs(output_dir, exist_ok=True)


                copy_calibration_file_to_dataset(config, restored_subject_id, output_dir)


                dat_dir = os.path.join(avatarrex_dir, get_padded_subject_id(restored_subject_id))

                if not os.path.exists(dat_dir):
                    logger.warning(f"Warning: Original subject directory not found: {dat_dir}")
                    logger.warning(f"Cannot render restored subject {restored_subject_id} without original camera data")
                    continue


                python_prefix = build_python_command(splatting_avatar_project, conda_env)


                cmd = python_prefix + [
                    'render_static_gaussians.py',
                    '--input_gs_ply', ply_file,
                    '--dat_dir', dat_dir,
                    '--configs', splatting_cfg['configs'],
                    '--output_dir', output_base_dir,
                    '--frame_id', str(frame_id),
                    '--render_splits', render_splits
                ]

                logger.info(f"Running SplattingAvatar rendering for: {ply_filename}")
                logger.info(f"Using camera data from: {dat_dir}")
                logger.info(f"Output directory: {output_dir}")

                try:
                    debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
                    run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
                    logger.info(f"Successfully rendered: {ply_filename}")
                except Exception as e:
                    logger.error(f"Error rendering {ply_filename}: {e}")
                    continue

    logger.info("Swapped-back Gaussian rendering completed.")

def stage_refine_swapped_back_images(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 25: Refine Swapped-Back Images ---")

    avatarrex_dir = config['paths']['avatarrex_output']
    difix3d_project = config['paths']['difix3d_project']


    refine_cfg = config['pipeline_stages'].get('14_refine_rendered_images', {})
    batch_size = refine_cfg.get('batch_size', 2)
    create_separate_dataset = refine_cfg.get('create_separate_dataset', True)
    extract_masks = refine_cfg.get('extract_masks', False)


    rendered_back_base_dir = os.path.join(avatarrex_dir, "head_swapped_back_renders")

    if not os.path.exists(rendered_back_base_dir):
        logger.error(f"Error: Rendered swapped-back images directory not found: {rendered_back_base_dir}")
        logger.error("Please run Stage 24 (render_swapped_back_gaussians) first.")
        return


    restored_dirs = []
    for subject in subjects:
        padded_subject = get_padded_subject_id(subject)


        restored_pattern = f"restored_{padded_subject}_from_*"
        for item in os.listdir(rendered_back_base_dir):
            if fnmatch.fnmatch(item, restored_pattern):
                item_path = os.path.join(rendered_back_base_dir, item)
                if os.path.isdir(item_path):
                    restored_dirs.append(item_path)

    if not restored_dirs:
        subject_list = ", ".join(subjects)
        logger.warning(f"Warning: No restored directories found for subjects: {subject_list}")
        logger.warning(f"Searched in: {rendered_back_base_dir}")
        logger.warning("Expected directories matching pattern: restored_<subject>_from_*")
        return


    reshaped_dirs = [dir for dir in restored_dirs if "reshaped_" in dir]
    non_reshaped_dirs = [dir for dir in restored_dirs if "reshaped_" not in dir]
    reshape_enabled = second_swap_reshape_enabled(config)
    if reshape_enabled:
        restored_dirs = reshaped_dirs
    else:
        restored_dirs = non_reshaped_dirs
    logger.info(f"Found {len(restored_dirs)} restored directories for specified subject pairs")


    refined_base_dir = os.path.join(avatarrex_dir, "head_swapped_back_renders_refined")
    logger.info(f"Refined images will be saved to: {refined_base_dir}")


    total_dirs = len(restored_dirs)
    for batch_start in range(0, total_dirs, batch_size):
        batch_end = min(batch_start + batch_size, total_dirs)
        batch_dirs = restored_dirs[batch_start:batch_end]

        logger.info(f"\n--- Processing Batch {batch_start//batch_size + 1} ({batch_start + 1}-{batch_end} of {total_dirs}) ---")

        for restored_dir in batch_dirs:
            restored_name = os.path.basename(restored_dir)
            logger.info(f"Processing restored directory: {restored_name}")


            if not os.path.exists(restored_dir):
                logger.warning(f"Warning: Restored directory not found: {restored_dir}")
                continue


            refined_base_dir = os.path.join(avatarrex_dir, "head_swapped_back_renders_refined")
            refined_dir = os.path.join(refined_base_dir, restored_name)


            cmd = [
                'uv', 'run', 'python', 'scripts/refine_batch_images_lightning.py',
                '--input_dir', restored_dir,
                '--output_suffix', '_refined',
                '--batch_size', str(batch_size),
                '--skip_existing'
            ]

            logger.info(f"Running batch refinement for directory: {restored_name}")
            logger.info(f"Input: {restored_dir}")

            try:
                debug_port = get_debug_port('refine_swapped_back_images') if debug_subprocess else None
                run_command(cmd, cwd=difix3d_project, debug_port=debug_port)
                logger.info(f"Successfully completed batch refinement for: {restored_name}")


                if extract_masks:
                    logger.info("Extracting masks from refined images")
                    extract_masks_from_refined_directory(refined_dir)
                    logger.info(f"Masks extracted to: {refined_dir}")
                else:
                    logger.info("Copying non-refined masks from original rendered directory")
                    copy_non_refined_masks(refined_dir)
                    logger.info(f"Non-refined masks copied to: {refined_dir}")

            except Exception as e:
                logger.error(f"Error refining {restored_name}: {e}")
                continue

    logger.info("Swapped-back image refinement completed.")
