import numpy as np
import cv2
from typing import Tuple
from loguru import logger


NECK_COLOR = (85, 51, 0)


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


def create_neck_mask_from_segmentation(seg_image: np.ndarray, head_mask: np.ndarray,
                                       tolerance: int = 5, neck_selection_ratio: float = 1/3) -> Tuple[np.ndarray, dict]:

    logger.debug(f"Creating neck mask from segmentation: shape={seg_image.shape}, tolerance={tolerance}, neck_selection_ratio={neck_selection_ratio}")

    if len(seg_image.shape) != 3 or seg_image.shape[2] != 3:
        raise ValueError(f"Expected RGB segmentation image (H, W, 3), got shape {seg_image.shape}")

    if len(head_mask.shape) != 2:
        raise ValueError(f"Expected 2D head mask (H, W), got shape {head_mask.shape}")

    H, W = seg_image.shape[:2]


    torso_skin_matches = find_matching_pixels(seg_image, NECK_COLOR, tolerance)
    torso_skin_mask = (torso_skin_matches * 255).astype(np.uint8)

    logger.debug(f"Found {np.sum(torso_skin_matches)} torso-skin pixels matching {NECK_COLOR}")


    head_coords = np.where(head_mask == 255)

    if len(head_coords[0]) == 0:
        logger.warning("Empty head mask provided - cannot compute neck region")
        return np.zeros((H, W), dtype=np.uint8), {
            'neck_pixels': 0,
            'head_pixels': 0,
            'num_components': 0,
            'distance_threshold': 0,
            'is_large_neck': False
        }

    head_min_y = np.min(head_coords[0])
    head_max_y = np.max(head_coords[0])
    head_bottom_y = head_max_y
    head_height = head_max_y - head_min_y
    head_pixels = len(head_coords[0])

    distance_threshold = head_height * neck_selection_ratio

    logger.debug(f"Head bbox: y=[{head_min_y}, {head_max_y}], height={head_height}")
    logger.debug(f"Distance threshold: {distance_threshold:.1f} pixels (ratio={neck_selection_ratio})")


    num_labels, labels = cv2.connectedComponents(torso_skin_mask)

    logger.debug(f"Found {num_labels - 1} torso-skin components")


    selected_components = []

    for label_id in range(1, num_labels):
        component_mask = (labels == label_id)
        component_coords = np.where(component_mask)

        if len(component_coords[0]) == 0:
            continue

        component_top_y = np.min(component_coords[0])
        component_size = np.sum(component_mask)


        distance = component_top_y - head_bottom_y

        if distance <= distance_threshold:
            selected_components.append(component_mask)
            logger.debug(f"Selected component {label_id}: size={component_size}, "
                        f"top_y={component_top_y}, distance={distance:.1f}")


    neck_mask = np.zeros((H, W), dtype=bool)
    for comp_mask in selected_components:
        neck_mask |= comp_mask

    neck_mask = (neck_mask.astype(np.uint8) * 255)


    neck_pixels = np.sum(neck_mask > 0)
    is_large_neck = neck_pixels > 2 * head_pixels

    stats = {
        'neck_pixels': int(neck_pixels),
        'head_pixels': int(head_pixels),
        'num_components': len(selected_components),
        'distance_threshold': float(distance_threshold),
        'is_large_neck': bool(is_large_neck)
    }

    logger.debug(f"Created neck mask: {neck_pixels} pixels from {len(selected_components)} components")

    if is_large_neck:
        ratio = neck_pixels / head_pixels if head_pixels > 0 else 0
        logger.debug(f"Large neck detected: {neck_pixels} pixels (head: {head_pixels}, ratio: {ratio:.2f})")

    if neck_pixels == 0:
        logger.debug("No neck components found within threshold")

    return neck_mask, stats


def dilate_neck_mask_for_composition(neck_mask: np.ndarray, head_mask: np.ndarray,
                                     neck_dilation_ratio: float = 0.05) -> np.ndarray:


    if neck_dilation_ratio == 0:
        logger.debug("Skipping neck composition dilation (neck_dilation_ratio=0)")
        return neck_mask.copy()


    head_coords = np.where(head_mask > 0)
    if len(head_coords[0]) == 0:
        logger.warning("Empty head mask - cannot calculate head height for composition dilation")
        return neck_mask.copy()

    head_height = np.max(head_coords[0]) - np.min(head_coords[0])


    kernel_size = int(head_height * neck_dilation_ratio)
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


    dilated_mask = cv2.dilate(neck_mask, kernel, iterations=1)

    original_pixels = np.sum(neck_mask > 0)
    dilated_pixels = np.sum(dilated_mask > 0)

    logger.debug(f"Dilated neck mask for composition: kernel_size={kernel_size}, "
                f"original_pixels={original_pixels}, dilated_pixels={dilated_pixels}, "
                f"expansion={dilated_pixels - original_pixels} pixels")

    return dilated_mask


def dilate_neck_mask_for_bbox(neck_mask: np.ndarray, head_mask: np.ndarray,
                               lower_context_ratio: float = 0.2) -> np.ndarray:


    if lower_context_ratio == 0:
        logger.debug("Skipping neck mask dilation (lower_context_ratio=0)")
        return neck_mask.copy()


    head_coords = np.where(head_mask > 0)
    if len(head_coords[0]) == 0:
        logger.warning("Empty head mask - cannot calculate head height for dilation")
        return neck_mask.copy()

    head_height = np.max(head_coords[0]) - np.min(head_coords[0])


    kernel_size = int(head_height * lower_context_ratio)
    kernel_size = max(3, kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))


    dilated_mask = cv2.dilate(neck_mask, kernel, iterations=1)

    original_pixels = np.sum(neck_mask > 0)
    dilated_pixels = np.sum(dilated_mask > 0)

    logger.debug(f"Dilated neck mask for bbox: kernel_size={kernel_size}, "
                f"original_pixels={original_pixels}, dilated_pixels={dilated_pixels}, "
                f"expansion={dilated_pixels - original_pixels} pixels")

    return dilated_mask


def validate_neck_mask(neck_mask: np.ndarray, head_mask: np.ndarray) -> Tuple[bool, str]:

    if len(neck_mask.shape) != 2:
        return False, f"Expected 2D neck mask, got shape {neck_mask.shape}"

    if len(head_mask.shape) != 2:
        return False, f"Expected 2D head mask, got shape {head_mask.shape}"


    unique_values = np.unique(neck_mask)
    if not (np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [0, 255])):
        return False, f"Expected binary mask with values [0] or [0, 255], got {unique_values}"

    neck_pixels = np.sum(neck_mask == 255)
    head_pixels = np.sum(head_mask == 255)


    if neck_pixels == 0:
        return True, "Empty neck mask (will use GT-only portrait)"


    if neck_pixels > 3 * head_pixels:
        ratio = neck_pixels / head_pixels if head_pixels > 0 else 0
        return False, f"Neck extremely large: {neck_pixels} pixels (head: {head_pixels}, ratio: {ratio:.2f})"

    return True, f"Valid neck mask with {neck_pixels} pixels"


def get_neck_mask_stats(neck_mask: np.ndarray, head_mask: np.ndarray) -> dict:

    height, width = neck_mask.shape
    total_pixels = height * width
    neck_pixels = np.sum(neck_mask == 255)
    head_pixels = np.sum(head_mask == 255)
    neck_percentage = (neck_pixels / total_pixels) * 100


    neck_coords = np.where(neck_mask == 255)
    if len(neck_coords[0]) > 0:
        min_y, max_y = np.min(neck_coords[0]), np.max(neck_coords[0])
        min_x, max_x = np.min(neck_coords[1]), np.max(neck_coords[1])
        bbox_width = max_x - min_x + 1
        bbox_height = max_y - min_y + 1
    else:
        min_x = min_y = max_x = max_y = 0
        bbox_width = bbox_height = 0

    ratio = neck_pixels / head_pixels if head_pixels > 0 else 0

    return {
        'total_pixels': total_pixels,
        'neck_pixels': neck_pixels,
        'head_pixels': head_pixels,
        'neck_percentage': neck_percentage,
        'neck_to_head_ratio': ratio,
        'bbox': {'x': min_x, 'y': min_y, 'width': bbox_width, 'height': bbox_height},
        'image_size': {'width': width, 'height': height}
    }
