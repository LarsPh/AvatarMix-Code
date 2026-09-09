import numpy as np
from typing import Tuple
from loguru import logger


def match_color(
    image: np.ndarray,
    target_color: Tuple[int, int, int],
    tolerance: int = 5
) -> np.ndarray:

    if image.shape[2] != 3:
        raise ValueError(f"Image must have 3 channels (RGB), got {image.shape[2]}")


    image_float = image.astype(np.float32)
    target_color_arr = np.array(target_color, dtype=np.float32)


    distance = np.sqrt(np.sum((image_float - target_color_arr) ** 2, axis=2))


    mask = distance <= tolerance

    return mask


def create_region_mask(
    segmentation: np.ndarray,
    color_list: list[Tuple[int, int, int]],
    tolerance: int = 5
) -> np.ndarray:

    if not color_list:
        raise ValueError("color_list cannot be empty")


    combined_mask = np.zeros(segmentation.shape[:2], dtype=bool)


    for color in color_list:
        color_mask = match_color(segmentation, color, tolerance)
        combined_mask = combined_mask | color_mask

    return combined_mask


def create_neck_hair_mask(
    segmentation: np.ndarray,
    neck_color: Tuple[int, int, int] = (85, 51, 0),
    hair_color: Tuple[int, int, int] = (255, 0, 0),
    tolerance: int = 5
) -> np.ndarray:

    return create_region_mask(
        segmentation,
        color_list=[neck_color, hair_color],
        tolerance=tolerance
    )


def create_neck_hair_mask_with_separate_dilation(
    segmentation: np.ndarray,
    neck_color: Tuple[int, int, int] = (85, 51, 0),
    hair_color: Tuple[int, int, int] = (255, 0, 0),
    tolerance: int = 5,
    dilate_neck: bool = False,
    neck_dilation_kernel_size: int = 5,
    dilate_hair: bool = False,
    hair_dilation_kernel_size: int = 3
) -> np.ndarray:


    neck_mask = match_color(segmentation, neck_color, tolerance)
    hair_mask = match_color(segmentation, hair_color, tolerance)


    if dilate_neck:
        neck_mask = dilate_mask(neck_mask, neck_dilation_kernel_size, iterations=1)
    if dilate_hair:
        hair_mask = dilate_mask(hair_mask, hair_dilation_kernel_size, iterations=1)


    combined_mask = neck_mask | hair_mask

    return combined_mask


def dilate_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping mask dilation")
        return mask


    mask_uint8 = mask.astype(np.uint8)


    kernel = np.ones((kernel_size, kernel_size), np.uint8)


    dilated = cv2.dilate(mask_uint8, kernel, iterations=iterations)


    return dilated.astype(bool)


def erode_mask(
    mask: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:

    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not available, skipping mask erosion")
        return mask


    mask_uint8 = mask.astype(np.uint8)


    kernel = np.ones((kernel_size, kernel_size), np.uint8)


    eroded = cv2.erode(mask_uint8, kernel, iterations=iterations)


    return eroded.astype(bool)


def visualize_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5
) -> np.ndarray:


    overlay = image.copy()
    overlay[mask] = color


    result = (alpha * overlay + (1 - alpha) * image).astype(np.uint8)

    return result
