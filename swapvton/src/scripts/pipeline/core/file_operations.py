import os
import shutil
import glob
from loguru import logger
from pathlib import Path
import numpy as np
from PIL import Image
from ..sampling.subject_discovery import get_padded_subject_id


def copy_calibration_file_to_dataset(config, subject_id, target_dir):

    avatarrex_dir = config['paths']['avatarrex_output']
    padded_subject_id = get_padded_subject_id(subject_id)
    source_calib = os.path.join(avatarrex_dir, padded_subject_id, "calibration_full.json")
    target_calib = os.path.join(target_dir, "calibration_full.json")

    if os.path.exists(source_calib):
        if not os.path.exists(target_calib):

            if not os.path.exists(os.path.dirname(target_calib)):
                os.makedirs(os.path.dirname(target_calib), exist_ok=True)
            shutil.copy2(source_calib, target_calib)
            logger.info(f"Copied calibration file from {padded_subject_id} to {os.path.basename(target_dir)}")
        else:
            logger.info(f"Calibration file already exists in {os.path.basename(target_dir)}")
    else:
        logger.warning(f"Calibration file not found for subject {padded_subject_id}: {source_calib}")


def find_swapped_directories(config, user_A_id, model_B_id):

    avatarrex_dir = config['paths']['avatarrex_output']
    swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

    if not os.path.exists(swapped_base_dir):
        logger.warning(f"Swapped base directory not found: {swapped_base_dir}")
        return []


    user_A_padded = get_padded_subject_id(user_A_id)
    model_B_padded = get_padded_subject_id(model_B_id)

    pattern = f"A{user_A_padded}_B{model_B_padded}*"
    search_pattern = os.path.join(swapped_base_dir, pattern)

    matching_dirs = glob.glob(search_pattern)
    matching_dirs = [d for d in matching_dirs if os.path.isdir(d)]

    if not matching_dirs:
        logger.warning(f"No swapped directories found matching pattern: {pattern}")

    return matching_dirs


def find_swapped_subject_directories(swapped_base_dir, subjects):

    subject_dirs = []

    if not os.path.exists(swapped_base_dir):
        logger.warning(f"Swapped base directory not found: {swapped_base_dir}")
        return subject_dirs


    all_dirs = [d for d in os.listdir(swapped_base_dir)
               if os.path.isdir(os.path.join(swapped_base_dir, d))]


    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue

            user_A_padded = get_padded_subject_id(subject_a)
            model_B_padded = get_padded_subject_id(subject_b)


            pattern_prefix = f"A{user_A_padded}_B{model_B_padded}"
            matching_dirs = [d for d in all_dirs if d.startswith(pattern_prefix)]

            for dir_name in matching_dirs:
                full_path = os.path.join(swapped_base_dir, dir_name)
                subject_dirs.append(full_path)

    return subject_dirs

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
