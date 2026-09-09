import numpy as np
from PIL import Image
import os
from typing import Optional, Tuple
from loguru import logger


def load_rgb_image(rgb_path: str) -> np.ndarray:

    try:
        rgb_image = Image.open(rgb_path)


        if rgb_image.mode != 'RGB':
            logger.debug(f"Converting image from {rgb_image.mode} to RGB")
            rgb_image = rgb_image.convert('RGB')

        rgb_array = np.array(rgb_image)
        logger.debug(f"Loaded RGB image: {rgb_path}, shape={rgb_array.shape}")

        return rgb_array

    except FileNotFoundError:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    except Exception as e:
        raise ValueError(f"Failed to load RGB image {rgb_path}: {e}")


def validate_mask_rgb_compatibility(mask: np.ndarray, rgb_image: np.ndarray) -> None:

    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")

    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {rgb_image.shape}")

    mask_height, mask_width = mask.shape
    rgb_height, rgb_width = rgb_image.shape[:2]

    if mask_height != rgb_height or mask_width != rgb_width:
        raise ValueError(
            f"Mask dimensions ({mask_height}, {mask_width}) don't match "
            f"RGB image dimensions ({rgb_height}, {rgb_width})"
        )


def apply_mask_to_rgb(rgb_path: str, mask: np.ndarray, output_path: str,
                     background_color: Tuple[int, int, int] = (0, 0, 0)) -> None:

    logger.debug(f"Applying mask to RGB image: {rgb_path} -> {output_path}")


    rgb_image = load_rgb_image(rgb_path)


    validate_mask_rgb_compatibility(mask, rgb_image)


    masked_rgb = apply_mask_to_rgb_array(rgb_image, mask, background_color)


    save_masked_rgb(masked_rgb, output_path)

    logger.info(f"Applied head mask to RGB image: {rgb_path} -> {output_path}")


def apply_mask_to_rgb_array(rgb_image: np.ndarray, mask: np.ndarray,
                           background_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:


    validate_mask_rgb_compatibility(mask, rgb_image)


    masked_rgb = rgb_image.copy()


    head_mask_bool = mask == 255


    for channel in range(3):
        masked_rgb[:, :, channel][~head_mask_bool] = background_color[channel]


    head_pixels = np.sum(head_mask_bool)
    total_pixels = head_mask_bool.size
    preserved_percentage = (head_pixels / total_pixels) * 100

    logger.debug(f"Applied mask: preserved {head_pixels}/{total_pixels} pixels ({preserved_percentage:.1f}%)")

    return masked_rgb


def save_masked_rgb(masked_rgb: np.ndarray, output_path: str, quality: int = 95) -> None:


    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    masked_image = Image.fromarray(masked_rgb.astype(np.uint8))

    try:

        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            masked_image.save(output_path, 'JPEG', quality=quality)
        else:
            masked_image.save(output_path)

        logger.debug(f"Saved masked RGB image: {output_path}")

    except Exception as e:
        raise ValueError(f"Failed to save masked RGB image to {output_path}: {e}")


def create_masked_rgb_batch(data_root: str, masks: list, camera_names: list,
                           background_color: Tuple[int, int, int] = (0, 0, 0)) -> None:

    if len(masks) != len(camera_names):
        raise ValueError(f"Number of masks ({len(masks)}) doesn't match camera names ({len(camera_names)})")

    logger.info(f"Processing batch of {len(masks)} RGB images with per-camera structure")
    processed_count = 0
    skipped_count = 0

    for mask, camera_name in zip(masks, camera_names):

        camera_dir = os.path.join(data_root, camera_name)
        rgb_path = os.path.join(camera_dir, "0000.jpg")


        output_dir = os.path.join(camera_dir, "masked_rgb", "head")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "0000.png")


        if not os.path.exists(rgb_path):
            logger.warning(f"RGB image not found, skipping: {rgb_path}")
            skipped_count += 1
            continue

        try:

            apply_mask_to_rgb(rgb_path, mask, output_path, background_color)
            processed_count += 1

        except Exception as e:
            logger.error(f"Failed to process {rgb_path}: {e}")
            skipped_count += 1

    logger.info(f"Batch processing complete: {processed_count} processed, {skipped_count} skipped")


def get_masked_rgb_stats(masked_rgb: np.ndarray, mask: np.ndarray) -> dict:

    height, width = masked_rgb.shape[:2]
    total_pixels = height * width


    head_mask_bool = mask == 255
    head_pixels = np.sum(head_mask_bool)
    preserved_percentage = (head_pixels / total_pixels) * 100


    head_region = masked_rgb[head_mask_bool]
    if len(head_region) > 0:
        mean_rgb = np.mean(head_region, axis=0)
        std_rgb = np.std(head_region, axis=0)
    else:
        mean_rgb = np.array([0, 0, 0])
        std_rgb = np.array([0, 0, 0])

    return {
        'total_pixels': total_pixels,
        'head_pixels': head_pixels,
        'preserved_percentage': preserved_percentage,
        'mean_rgb': mean_rgb.tolist(),
        'std_rgb': std_rgb.tolist(),
        'image_size': {'width': width, 'height': height}
    }
