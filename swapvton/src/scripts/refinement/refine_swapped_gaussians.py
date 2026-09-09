#!/usr/bin/env python3


import argparse
import os
import glob
import subprocess
import shutil
from pathlib import Path
from loguru import logger
import yaml
import numpy as np
from PIL import Image


def load_config(config_path):

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_padded_subject_id(subject_id):

    return f"{int(subject_id):04d}"

def get_reshape_suffix_from_config(config):

    reposing_cfg = config['pipeline_stages']['11_reposing']
    if not reposing_cfg.get('enable_body_reshape', False):
        return ""

    samples = reposing_cfg.get('body_reshape_smoothing_samples', 64)
    std_scale = reposing_cfg.get('body_reshape_sample_std_scale', 0.1)
    suffix = f"_reshaped_{samples}samples_std{std_scale}"


    scale_factor = reposing_cfg.get('body_reshape_scale_factor', 1.0)
    if scale_factor != 1.0:
        suffix += f"_scale{scale_factor}"


    if reposing_cfg.get('body_reshape_distance_weighting', False):
        suffix += "_distweight"

    return suffix

def find_swapped_directories(config, user_A_id, model_B_id):

    avatarrex_dir = config['paths']['avatarrex_output']
    swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

    if not os.path.exists(swapped_base_dir):
        return []

    base_pattern = f"A{user_A_id}_B{model_B_id}"


    reshape_suffix = get_reshape_suffix_from_config(config)
    if reshape_suffix:
        suffixed_dir = os.path.join(swapped_base_dir, f"{base_pattern}{reshape_suffix}")
        if os.path.exists(suffixed_dir):
            return [suffixed_dir]
    else:
        exact_dir = os.path.join(swapped_base_dir, base_pattern)
        if os.path.exists(exact_dir):
            return [exact_dir]

    raise ValueError(f"No swapped directories found for pair {user_A_id} -> {model_B_id}")


def find_swapped_ply_files(swapped_dir):

    ply_pattern = os.path.join(swapped_dir, "*.ply")
    ply_files = glob.glob(ply_pattern)


    swapped_files = []
    for ply_file in ply_files:
        filename = os.path.basename(ply_file)
        if "swapped_" in filename and ("_direct" in filename or "_with_color_transfer" in filename):
            swapped_files.append(ply_file)

    return swapped_files


def get_dat_dir_for_subject(config, subject_id):

    avatarrex_dir = config['paths']['avatarrex_output']
    sub_padded = get_padded_subject_id(subject_id)
    return os.path.join(avatarrex_dir, sub_padded)


def extract_mask_from_refined_image(image_path, mask_output_path, bg_mode="black", white_threshold=250, black_threshold=5):

    try:

        image = Image.open(image_path)
        image_array = np.array(image)

        extract_fn_rgb = lambda x: np.all(x >= white_threshold, axis=-1) if bg_mode == "white" else np.all(x <= black_threshold, axis=-1)
        extract_fn_gray = lambda x: x >= white_threshold if bg_mode == "white" else x <= black_threshold


        if len(image_array.shape) == 3:
            white_pixels = extract_fn_rgb(image_array)
        else:

            white_pixels = extract_fn_gray(image_array)


        mask = np.where(white_pixels, 0, 255).astype(np.uint8)


        os.makedirs(os.path.dirname(mask_output_path), exist_ok=True)
        mask_image = Image.fromarray(mask, mode='L')
        mask_image.save(mask_output_path)

        logger.debug(f"Extracted mask saved to: {mask_output_path}")

    except Exception as e:
        logger.error(f"Failed to extract mask from {image_path}: {e}")
        raise


def create_refined_dataset_structure(original_rendered_dir, refined_base_dir):


    ply_name = os.path.basename(original_rendered_dir)
    refined_dir = os.path.join(refined_base_dir, ply_name)


    os.makedirs(refined_dir, exist_ok=True)

    return refined_dir

def copy_non_refined_masks(refined_dir):

    non_refined_dir_parent, non_refined_dir_name = os.path.dirname(refined_dir), os.path.basename(refined_dir)
    non_refined_mask_pattern = os.path.join(non_refined_dir_parent.replace("_refined", ""), non_refined_dir_name, "**", "mask", "pha", "*.png")
    non_refined_mask_files = glob.glob(non_refined_mask_pattern, recursive=False)

    logger.info(f"Found {len(non_refined_mask_files)} non-refined masks to copy from {non_refined_mask_pattern}")
    for non_refined_mask_path in non_refined_mask_files:

        relative_path = os.path.relpath(non_refined_mask_path, refined_dir.replace("_refined", ""))
        refined_mask_path = os.path.join(refined_dir, relative_path)

        Path(refined_mask_path).parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(non_refined_mask_path, refined_mask_path)
        logger.debug(f"Copied non-refined mask: {refined_mask_path}")

def extract_masks_from_refined_directory(refined_dir):

    from loguru import logger
    import glob

    logger.info(f"Extracting masks from refined directory: {refined_dir}")


    refined_image_pattern = os.path.join(refined_dir, "**", "*.jpg")
    all_image_files = glob.glob(refined_image_pattern, recursive=True)


    refined_image_files = []
    for image_path in all_image_files:

        if 'mask' not in image_path:
            refined_image_files.append(image_path)

    logger.info(f"Found {len(refined_image_files)} refined images to extract masks from")

    mask_count = 0

    for refined_image_path in refined_image_files:
        try:

            rel_path = os.path.relpath(refined_image_path, refined_dir)


            rel_dir = os.path.dirname(rel_path)
            image_name = os.path.splitext(os.path.basename(rel_path))[0]
            mask_rel_path = os.path.join(rel_dir, "mask", "pha", f"{image_name}.png")
            mask_dest = os.path.join(refined_dir, mask_rel_path)


            extract_mask_from_refined_image(refined_image_path, mask_dest, bg_mode="black")
            mask_count += 1
            logger.debug(f"Extracted mask: {mask_rel_path}")

        except Exception as e:
            logger.error(f"Failed to extract mask from {refined_image_path}: {e}")
            continue

    logger.info(f"Successfully extracted {mask_count} masks")


def copy_refined_images_and_extract_masks(original_rendered_dir, refined_dir, extract_masks=True):

    logger.info("Copying refined images to separate dataset and extracting masks")


    refined_image_pattern = os.path.join(original_rendered_dir, "**", "*_refined.jpg")
    refined_image_files = glob.glob(refined_image_pattern, recursive=True)

    logger.info(f"Found {len(refined_image_files)} refined images to process")

    copied_count = 0
    mask_count = 0

    for refined_image_path in refined_image_files:
        try:

            rel_path = os.path.relpath(refined_image_path, original_rendered_dir)


            rel_dir = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            name_without_ext = os.path.splitext(filename)[0]


            if name_without_ext.endswith("_refined"):
                name_without_ext = name_without_ext[:-8]


            clean_filename = f"{name_without_ext}.jpg"
            clean_rel_path = os.path.join(rel_dir, clean_filename)


            refined_image_dest = os.path.join(refined_dir, clean_rel_path)
            os.makedirs(os.path.dirname(refined_image_dest), exist_ok=True)
            shutil.copy2(refined_image_path, refined_image_dest)
            copied_count += 1
            logger.debug(f"Copied refined image: {clean_rel_path}")


            if extract_masks:


                cam_name = os.path.dirname(clean_rel_path)
                image_name = os.path.splitext(os.path.basename(clean_rel_path))[0]
                mask_rel_path = os.path.join(cam_name, "mask", "pha", f"{image_name}.png")
                mask_dest = os.path.join(refined_dir, mask_rel_path)


                extract_mask_from_refined_image(refined_image_dest, mask_dest)
                mask_count += 1
                logger.debug(f"Extracted mask: {mask_rel_path}")

        except Exception as e:
            logger.error(f"Failed to process refined image {refined_image_path}: {e}")
            continue

    logger.info(f"Successfully copied {copied_count} refined images")
    if extract_masks:
        logger.info(f"Successfully extracted {mask_count} masks")


def render_swapped_gaussians(config, input_ply_path, user_A_id, model_B_id):

    logger.info(f"Starting render for swapped PLY: {input_ply_path}")


    splatting_avatar_project = config['paths']['splatting_avatar_project']
    conda_env = config['conda_envs']['splatting']
    avatarrex_dir = config['paths']['avatarrex_output']


    dat_dir = get_dat_dir_for_subject(config, user_A_id)


    ply_filename = os.path.basename(input_ply_path)
    ply_name = os.path.splitext(ply_filename)[0]
    output_base_dir = os.path.join(avatarrex_dir, "head_swapped_renders")


    splatting_configs = config['pipeline_stages']['10_splatting_avatar']['configs']


    cmd = [
        'conda', 'run', '-n', conda_env, '--no-capture-output', 'python', 'render_static_gaussians.py',
        '--input_gs_ply', input_ply_path,
        '--dat_dir', dat_dir,
        '--configs', splatting_configs,
        '--output_dir', output_base_dir,
        '--frame_id', '0',
    ]

    logger.info(f"Running SplattingAvatar rendering command: {' '.join(cmd)}")

    output_dir = os.path.join(output_base_dir, ply_name)
    logger.info(f"Output directory: {output_dir}")


    try:
        result = subprocess.Popen(cmd, cwd=splatting_avatar_project, shell=False)
        return_code = result.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)
        logger.info("SplattingAvatar rendering completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"SplattingAvatar rendering failed: {e}")
        raise

    return output_dir


def refine_rendered_images(config, rendered_dir, create_separate_dataset=True, extract_masks=True):

    logger.info(f"Starting batch refinement for rendered images in: {rendered_dir}")


    difix3d_project = config['paths']['difix3d_project']
    avatarrex_dir = config['paths']['avatarrex_output']


    if create_separate_dataset:
        refined_base_dir = os.path.join(avatarrex_dir, "head_swapped_renders_refined")
        refined_dir = create_refined_dataset_structure(rendered_dir, refined_base_dir)
        logger.info(f"Created refined dataset directory: {refined_dir}")
    else:
        refined_dir = rendered_dir


    image_pattern = os.path.join(rendered_dir, "**", "*.jpg")
    image_files = glob.glob(image_pattern, recursive=True)


    images_to_process = []
    for image_path in image_files:
        if "_refined" in image_path:
            continue


        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        name_without_ext = os.path.splitext(image_name)[0]

        if create_separate_dataset:

            rel_path = os.path.relpath(image_path, rendered_dir)
            refined_image_path = os.path.join(refined_dir, rel_path)
            if os.path.exists(refined_image_path):
                logger.debug(f"Refined image already exists in separate dataset: {refined_image_path}")
                continue
        else:

            refined_path = os.path.join(image_dir, f"{name_without_ext}_refined.jpg")
            if os.path.exists(refined_path):
                logger.debug(f"Refined image already exists: {refined_path}")
                continue

        images_to_process.append(image_path)

    logger.info(f"Found {len(images_to_process)} images to refine")

    if not images_to_process:
        logger.info("No images to refine - all images already processed")
        return refined_dir if create_separate_dataset else rendered_dir


    try:

        cmd = [
            'uv', 'run', 'python', 'refine_batch_images.py',
            '--input_dir', rendered_dir,
            '--output_suffix', '_refined',
            '--batch_size', '4',
            '--skip_existing'
        ]

        logger.info(f"Running batch refinement with {len(images_to_process)} images")
        logger.debug(f"Command: {' '.join(cmd)}")

        result = subprocess.Popen(cmd, cwd=difix3d_project, shell=False)
        return_code = result.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

        logger.info("Batch refinement completed successfully")

    except subprocess.CalledProcessError as e:
        logger.error(f"Batch refinement failed: {e}")


        logger.warning("Falling back to single image processing")
        refined_count = 0

        for image_path in images_to_process:
            image_dir = os.path.dirname(image_path)
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            refined_path = os.path.join(image_dir, f"{name_without_ext}_refined.jpg")


            fallback_cmd = [
                'uv', 'run', 'python', 'refine_single_image.py',
                '--input_image', image_path,
                '--output_image', refined_path
            ]

            logger.debug(f"Refining image: {image_path}")

            try:
                result = subprocess.Popen(fallback_cmd, cwd=difix3d_project, shell=False)
                return_code = result.wait()
                if return_code != 0:
                    raise subprocess.CalledProcessError(return_code, fallback_cmd)
                refined_count += 1
                logger.debug(f"Successfully refined: {image_path}")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to refine image {image_path}: {e}")
                continue

        logger.info(f"Fallback processing completed: {refined_count} images refined")


    if create_separate_dataset:
        logger.info("Creating separate refined dataset with extracted masks")
        copy_refined_images_and_extract_masks(rendered_dir, refined_dir, extract_masks)

    return refined_dir if create_separate_dataset else rendered_dir


def process_single_pair(config, user_A_id, model_B_id, render_only=False, create_separate_dataset=True, extract_masks=True):

    logger.info(f"Processing pair: User A ({user_A_id}) -> Model B ({model_B_id})")


    swapped_dirs = find_swapped_directories(config, user_A_id, model_B_id)

    if not swapped_dirs:
        logger.error(f"No swapped directories found for pair {user_A_id} -> {model_B_id}")


        avatarrex_dir = config['paths']['avatarrex_output']
        swapped_base_dir = os.path.join(avatarrex_dir, "swapped")
        reshape_suffix = get_reshape_suffix_from_config(config)
        expected_base = f"A{user_A_id}_B{model_B_id}"
        expected_with_suffix = f"{expected_base}{reshape_suffix}" if reshape_suffix else expected_base

        logger.info(f"Expected directory: {os.path.join(swapped_base_dir, expected_with_suffix)}")
        if reshape_suffix:
            logger.info(f"Also checked: {os.path.join(swapped_base_dir, expected_base)}")

        return []

    logger.info(f"Found {len(swapped_dirs)} swapped directories to process")

    output_dirs = []


    for swapped_dir in swapped_dirs:
        logger.info(f"Processing directory: {os.path.basename(swapped_dir)}")

        swapped_ply_files = find_swapped_ply_files(swapped_dir)

        if not swapped_ply_files:
            logger.warning(f"No swapped PLY files found in: {swapped_dir}")
            continue

        logger.info(f"Found {len(swapped_ply_files)} swapped PLY files to process")

        for ply_file in swapped_ply_files:
            logger.info(f"Processing PLY file: {os.path.basename(ply_file)}")


            try:
                rendered_dir = render_swapped_gaussians(config, ply_file, user_A_id, model_B_id)
                output_dirs.append(rendered_dir)
                logger.info(f"Rendering completed for: {os.path.basename(ply_file)}")
            except Exception as e:
                logger.error(f"Rendering failed for {ply_file}: {e}")
                continue


            if not render_only:
                try:
                    refined_dir = refine_rendered_images(config, rendered_dir, create_separate_dataset, extract_masks)
                    logger.info(f"Refinement completed for: {os.path.basename(ply_file)}")

                    output_dirs[-1] = refined_dir
                except Exception as e:
                    logger.error(f"Refinement failed for {ply_file}: {e}")
                    continue

    return output_dirs


def main():

    parser = argparse.ArgumentParser(description='Refine swapped Gaussians')
    parser.add_argument('--config', required=True, help='Configuration YAML file')
    parser.add_argument('--user_A_id', required=True, help='User A subject ID')
    parser.add_argument('--model_B_id', required=True, help='Model B subject ID')
    parser.add_argument('--render_only', action='store_true',
                       help='Only render images, skip refinement')
    parser.add_argument('--create_separate_dataset', action='store_true', default=True,
                       help='Create separate directory for refined dataset (default: True)')
    parser.add_argument('--no_separate_dataset', action='store_true',
                       help='Disable separate dataset creation (save refined images in same directory)')
    parser.add_argument('--extract_masks', action='store_true', default=True,
                       help='Extract masks from refined images (default: True)')
    parser.add_argument('--no_extract_masks', action='store_true',
                       help='Disable mask extraction from refined images')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()


    create_separate_dataset = args.create_separate_dataset and not args.no_separate_dataset
    extract_masks = args.extract_masks and not args.no_extract_masks


    if args.verbose:
        logger.add(lambda msg: print(msg, end=''), level="DEBUG")
    else:
        logger.add(lambda msg: print(msg, end=''), level="INFO")


    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1


    try:
        output_dirs = process_single_pair(config, args.user_A_id, args.model_B_id,
                                         render_only=args.render_only,
                                         create_separate_dataset=create_separate_dataset,
                                         extract_masks=extract_masks)

        if output_dirs:
            logger.info(f"Successfully processed {len(output_dirs)} PLY files")
            logger.info("Output directories:")
            for output_dir in output_dirs:
                logger.info(f"  - {output_dir}")
        else:
            logger.error("No PLY files were successfully processed")
            return 1

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
