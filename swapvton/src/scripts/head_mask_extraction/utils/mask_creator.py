import numpy as np
from typing import Tuple, List
from loguru import logger


HEAD_COLORS = {
    'face': (0, 0, 255),
    'hair': (255, 0, 0),
    'hat': (128, 0, 0),
    'sunglasses': (170, 0, 51),
}


def color_distance(color1: Tuple[int, int, int], color2: Tuple[int, int, int]) -> float:

    r1, g1, b1 = color1
    r2, g2, b2 = color2
    return np.sqrt((r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2)


def find_matching_pixels(seg_image: np.ndarray, target_color: Tuple[int, int, int],
                        tolerance: int = 5) -> np.ndarray:

    if len(seg_image.shape) != 3 or seg_image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {seg_image.shape}")


    r_diff = seg_image[:, :, 0].astype(np.float32) - target_color[0]
    g_diff = seg_image[:, :, 1].astype(np.float32) - target_color[1]
    b_diff = seg_image[:, :, 2].astype(np.float32) - target_color[2]

    distances = np.sqrt(r_diff**2 + g_diff**2 + b_diff**2)


    matching_pixels = distances <= tolerance

    return matching_pixels


def create_head_mask(seg_image: np.ndarray, tolerance: int = 5) -> np.ndarray:

    logger.debug(f"Creating head mask from segmentation image: shape={seg_image.shape}, tolerance={tolerance}")

    if len(seg_image.shape) != 3 or seg_image.shape[2] != 3:
        raise ValueError(f"Expected RGB segmentation image (H, W, 3), got shape {seg_image.shape}")

    height, width = seg_image.shape[:2]


    head_mask = np.zeros((height, width), dtype=bool)


    found_components = []


    for component_name, target_color in HEAD_COLORS.items():

        component_mask = find_matching_pixels(seg_image, target_color, tolerance)
        component_pixel_count = np.sum(component_mask)

        if component_pixel_count > 0:
            found_components.append(component_name)
            head_mask = head_mask | component_mask
            logger.debug(f"Found {component_pixel_count} pixels for {component_name} {target_color}")
        else:
            logger.debug(f"No pixels found for {component_name} {target_color}")


    binary_mask = (head_mask * 255).astype(np.uint8)

    total_head_pixels = np.sum(head_mask)
    total_pixels = height * width
    head_percentage = (total_head_pixels / total_pixels) * 100

    logger.info(f"Created head mask: {total_head_pixels}/{total_pixels} pixels ({head_percentage:.1f}%)")
    logger.info(f"Head components found: {found_components}")

    if total_head_pixels == 0:
        logger.warning("No head pixels found! Check segmentation image and color definitions.")
    elif head_percentage > 50:
        logger.warning(f"Head region covers {head_percentage:.1f}% of image - unusually large!")

    return binary_mask


def validate_head_mask(mask: np.ndarray, min_pixels: int = 100) -> bool:

    if len(mask.shape) != 2:
        logger.error(f"Expected 2D mask, got shape {mask.shape}")
        return False


    unique_values = np.unique(mask)
    if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [0, 255]):
        logger.error(f"Expected binary mask with values [0] or [0, 255], got {unique_values}")
        return False


    head_pixels = np.sum(mask == 255)
    if head_pixels < min_pixels:
        logger.error(f"Head mask has only {head_pixels} pixels (minimum: {min_pixels})")
        return False

    return True


def get_head_mask_stats(mask: np.ndarray) -> dict:

    height, width = mask.shape
    total_pixels = height * width
    head_pixels = np.sum(mask == 255)
    head_percentage = (head_pixels / total_pixels) * 100


    head_coords = np.where(mask == 255)
    if len(head_coords[0]) > 0:
        min_y, max_y = np.min(head_coords[0]), np.max(head_coords[0])
        min_x, max_x = np.min(head_coords[1]), np.max(head_coords[1])
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
    else:
        min_x = min_y = max_x = max_y = 0
        bbox_width = bbox_height = 0

    return {
        'total_pixels': total_pixels,
        'head_pixels': head_pixels,
        'head_percentage': head_percentage,
        'bbox': {'x': min_x, 'y': min_y, 'width': bbox_width, 'height': bbox_height},
        'image_size': {'width': width, 'height': height}
    }
