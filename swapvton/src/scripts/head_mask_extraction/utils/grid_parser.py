import os
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from loguru import logger


def detect_image_resolution(data_root: str, dataset_type: str = "thuman2") -> Tuple[int, int]:

    logger.debug(f"Detecting image resolution for {dataset_type} dataset: {data_root}")


    sample_image_path = None


    if os.path.exists(data_root):
        for item in os.listdir(data_root):
            item_path = os.path.join(data_root, item)
            if os.path.isdir(item_path):

                for img_name in ["0000.jpg", "0000.png"]:
                    img_path = os.path.join(item_path, img_name)
                    if os.path.exists(img_path):
                        sample_image_path = img_path
                        break
                if sample_image_path:
                    break

    if not sample_image_path:

        if dataset_type == "thuman2":
            logger.warning(f"No sample images found, using default 1024x1024 for {dataset_type}")
            return (1024, 1024)
        else:
            logger.warning(f"No sample images found, using default 1024x1024 for unknown dataset type")
            return (1024, 1024)

    try:
        with Image.open(sample_image_path) as img:
            width, height = img.size
            logger.debug(f"Detected resolution from {sample_image_path}: {height}x{width}")
            return (height, width)
    except Exception as e:
        logger.error(f"Failed to read sample image {sample_image_path}: {e}")
        logger.warning(f"Using default 1024x1024 resolution")
        return (1024, 1024)


def get_camera_names_from_dataset(data_root: str, dataset_type: str = "thuman2") -> List[str]:

    logger.debug(f"Discovering camera names for {dataset_type} dataset: {data_root}")

    camera_names = []

    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")


    for item in os.listdir(data_root):
        item_path = os.path.join(data_root, item)
        if os.path.isdir(item_path):


            has_images = any(
                os.path.exists(os.path.join(item_path, img_name))
                for img_name in ["0000.jpg", "0000.png"]
            )
            if has_images:
                camera_names.append(item)

    if not camera_names:
        raise FileNotFoundError(f"No camera directories with images found in {data_root}")


    if dataset_type == "thuman2":

        def extract_yaw_angle(camera_name: str) -> int:
            try:

                yaw_str = ''.join(filter(str.isdigit, camera_name))[:3]
                return int(yaw_str) if yaw_str else 0
            except (ValueError, IndexError):
                return 0

        camera_names.sort(key=extract_yaw_angle)
    else:

        camera_names.sort()

    logger.debug(f"Found {len(camera_names)} cameras: {camera_names[:4]}...{camera_names[-4:] if len(camera_names) > 4 else camera_names}")
    return camera_names


def detect_grid_layout(image_shape: Tuple[int, int], image_resolution: Tuple[int, int]) -> Tuple[int, int]:

    grid_height, grid_width = image_shape
    img_height, img_width = image_resolution

    if grid_height % img_height != 0 or grid_width % img_width != 0:
        raise ValueError(
            f"Grid dimensions ({grid_height}, {grid_width}) are not divisible by "
            f"image resolution ({img_height}, {img_width}). "
            f"Expected format: (rows*{img_height}, cols*{img_width})"
        )

    rows = grid_height // img_height
    cols = grid_width // img_width

    logger.debug(f"Detected grid layout: {rows}×{cols} ({rows*cols} total images) "
                f"with {img_height}×{img_width} resolution per image")
    return rows, cols


def parse_segmentation_grid(grid_path: str, image_resolution: Tuple[int, int],
                          camera_names: List[str]) -> List[np.ndarray]:

    logger.debug(f"Parsing segmentation grid: {grid_path}")
    logger.debug(f"Using image resolution: {image_resolution}, expected cameras: {len(camera_names)}")


    try:
        grid_image = Image.open(grid_path)
        grid_array = np.array(grid_image)
    except FileNotFoundError:
        raise FileNotFoundError(f"Segmentation grid not found: {grid_path}")
    except Exception as e:
        raise ValueError(f"Failed to load segmentation grid: {e}")

    logger.debug(f"Grid image shape: {grid_array.shape}")


    if len(grid_array.shape) == 3:
        height, width, channels = grid_array.shape
    else:
        height, width = grid_array.shape
        channels = 1

    rows, cols = detect_grid_layout((height, width), image_resolution)
    img_height, img_width = image_resolution


    total_views = rows * cols
    expected_cameras = len(camera_names)

    if total_views != expected_cameras:
        logger.warning(
            f"Grid contains {total_views} views but expected {expected_cameras} cameras. "
            f"Proceeding with available views."
        )


    camera_images = []

    for row in range(rows):
        for col in range(cols):

            y_start = row * img_height
            y_end = y_start + img_height
            x_start = col * img_width
            x_end = x_start + img_width


            if len(grid_array.shape) == 3:
                camera_image = grid_array[y_start:y_end, x_start:x_end, :]
            else:
                camera_image = grid_array[y_start:y_end, x_start:x_end]

                camera_image = np.stack([camera_image] * 3, axis=-1)

            camera_images.append(camera_image)

            logger.debug(f"Extracted camera view {len(camera_images)}: "
                        f"grid_pos=({row},{col}), shape={camera_image.shape}")

    logger.info(f"Successfully parsed segmentation grid into {len(camera_images)} camera views")
    return camera_images


def validate_camera_count(num_images: int, camera_names: List[str]) -> None:

    if num_images != len(camera_names):
        raise ValueError(
            f"Mismatch between extracted images ({num_images}) and discovered cameras ({len(camera_names)}). "
            f"Grid layout may not match actual dataset camera configuration."
        )
