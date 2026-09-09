import numpy as np
from typing import Tuple, Optional, Union, Dict
from loguru import logger


def combine_rgb_images(gt_rgb: np.ndarray, swapped_rgb: np.ndarray,
                      neck_mask: np.ndarray, gt_neck_mask: Optional[np.ndarray] = None) -> np.ndarray:

    if len(gt_rgb.shape) != 3 or gt_rgb.shape[2] != 3:
        raise ValueError(f"Expected GT RGB (H, W, 3), got shape {gt_rgb.shape}")

    if len(swapped_rgb.shape) != 3 or swapped_rgb.shape[2] != 3:
        raise ValueError(f"Expected swapped RGB (H, W, 3), got shape {swapped_rgb.shape}")

    if len(neck_mask.shape) != 2:
        raise ValueError(f"Expected 2D neck mask (H, W), got shape {neck_mask.shape}")

    if gt_rgb.shape[:2] != swapped_rgb.shape[:2]:
        raise ValueError(f"GT RGB shape {gt_rgb.shape[:2]} != swapped RGB shape {swapped_rgb.shape[:2]}")

    if gt_rgb.shape[:2] != neck_mask.shape:
        raise ValueError(f"GT RGB shape {gt_rgb.shape[:2]} != neck mask shape {neck_mask.shape}")

    if gt_neck_mask is not None:
        if len(gt_neck_mask.shape) != 2:
            raise ValueError(f"Expected 2D GT neck mask (H, W), got shape {gt_neck_mask.shape}")
        if gt_rgb.shape[:2] != gt_neck_mask.shape:
            raise ValueError(f"GT RGB shape {gt_rgb.shape[:2]} != GT neck mask shape {gt_neck_mask.shape}")

    H, W = gt_rgb.shape[:2]


    combined = np.zeros((H, W, 3), dtype=np.uint8)


    if gt_neck_mask is None:


        neck_bool = neck_mask > 0
        combined[neck_bool] = swapped_rgb[neck_bool]


        gt_foreground = np.any(gt_rgb > 0, axis=2)


        gt_region = gt_foreground & (neck_mask == 0)
        combined[gt_region] = gt_rgb[gt_region]

        logger.debug(f"Combined RGB (perfect alignment): GT pixels={np.sum(gt_region)}, neck pixels={np.sum(neck_bool)}")

    else:


        gt_no_neck_bool = (gt_neck_mask == 0)
        gt_foreground = np.any(gt_rgb > 0, axis=2)
        gt_region = gt_foreground & gt_no_neck_bool
        combined[gt_region] = gt_rgb[gt_region]


        swapped_neck_bool = (neck_mask > 0)
        combined[swapped_neck_bool] = swapped_rgb[swapped_neck_bool]


        overlap_region = gt_region & swapped_neck_bool
        gap_region = (gt_neck_mask > 0) & (~swapped_neck_bool)

        logger.debug(f"Combined RGB (with misalignment): GT no-neck pixels={np.sum(gt_region)}, "
                    f"swapped neck pixels={np.sum(swapped_neck_bool)}, "
                    f"overlap={np.sum(overlap_region)}, gap={np.sum(gap_region)}")

    return combined


def compute_and_crop_portrait(combined_rgb: np.ndarray, gt_seg: np.ndarray,
                              head_mask: np.ndarray, neck_mask: np.ndarray,
                              gt_rgb: np.ndarray,
                              dilated_neck_mask: Optional[np.ndarray] = None,
                              return_bbox: bool = False) -> Union[
                                  Tuple[np.ndarray, np.ndarray, np.ndarray],
                                  Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]
                              ]:

    if len(combined_rgb.shape) != 3 or combined_rgb.shape[2] != 3:
        raise ValueError(f"Expected combined RGB (H, W, 3), got shape {combined_rgb.shape}")

    if len(gt_seg.shape) != 3 or gt_seg.shape[2] != 3:
        raise ValueError(f"Expected GT segmentation (H, W, 3), got shape {gt_seg.shape}")

    if len(head_mask.shape) != 2:
        raise ValueError(f"Expected 2D head mask (H, W), got shape {head_mask.shape}")

    if len(neck_mask.shape) != 2:
        raise ValueError(f"Expected 2D neck mask (H, W), got shape {neck_mask.shape}")

    if len(gt_rgb.shape) != 3 or gt_rgb.shape[2] != 3:
        raise ValueError(f"Expected GT RGB (H, W, 3), got shape {gt_rgb.shape}")

    if combined_rgb.shape[:2] != gt_seg.shape[:2]:
        raise ValueError(f"Combined RGB shape {combined_rgb.shape[:2]} != seg shape {gt_seg.shape[:2]}")

    if gt_rgb.shape != combined_rgb.shape:
        raise ValueError(f"GT RGB shape {gt_rgb.shape} != combined RGB shape {combined_rgb.shape}")

    H, W = combined_rgb.shape[:2]


    neck_for_bbox = dilated_neck_mask if dilated_neck_mask is not None else neck_mask


    combined_mask = ((head_mask > 0) | (neck_for_bbox > 0)).astype(np.uint8) * 255


    coords = np.where(combined_mask > 0)

    if len(coords[0]) == 0:

        logger.warning("Empty combined mask - using head mask only")
        coords = np.where(head_mask > 0)

        if len(coords[0]) == 0:

            raise ValueError("Both head and neck masks are empty - cannot crop portrait")

    min_y, max_y = np.min(coords[0]), np.max(coords[0])
    min_x, max_x = np.min(coords[1]), np.max(coords[1])

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    logger.debug(f"Bbox: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}], width={width}, height={height}")


    size = max(width, height)


    if width > height:

        center_y = (min_y + max_y) // 2
        new_y = center_y - size // 2
        new_x = min_x
    else:

        center_x = (min_x + max_x) // 2
        new_x = center_x - size // 2
        new_y = min_y


    new_x = max(0, min(new_x, W - size))
    new_y = max(0, min(new_y, H - size))


    actual_size_x = min(size, W - new_x)
    actual_size_y = min(size, H - new_y)
    actual_size = min(actual_size_x, actual_size_y)

    logger.debug(f"Square bbox: x={new_x}, y={new_y}, size={actual_size}")


    cropped_combined_rgb = combined_rgb[new_y:new_y+actual_size, new_x:new_x+actual_size]
    cropped_seg = gt_seg[new_y:new_y+actual_size, new_x:new_x+actual_size]
    cropped_gt_rgb = gt_rgb[new_y:new_y+actual_size, new_x:new_x+actual_size]

    logger.debug(f"Cropped portrait: combined={cropped_combined_rgb.shape}, "
                f"seg={cropped_seg.shape}, gt={cropped_gt_rgb.shape}")


    if return_bbox:
        bbox_metadata = {
            'x': int(new_x),
            'y': int(new_y),
            'size': int(actual_size),
            'original_width': int(W),
            'original_height': int(H)
        }
        logger.debug(f"Bbox metadata: {bbox_metadata}")
        return cropped_combined_rgb, cropped_seg, cropped_gt_rgb, bbox_metadata
    else:
        return cropped_combined_rgb, cropped_seg, cropped_gt_rgb


def get_combined_image_stats(combined_rgb: np.ndarray, gt_rgb: np.ndarray,
                             swapped_rgb: np.ndarray, neck_mask: np.ndarray) -> dict:

    H, W = combined_rgb.shape[:2]
    total_pixels = H * W


    combined_pixels = np.sum(np.any(combined_rgb > 0, axis=2))
    gt_pixels = np.sum(np.any(gt_rgb > 0, axis=2))
    swapped_pixels = np.sum(np.any(swapped_rgb > 0, axis=2))
    neck_pixels = np.sum(neck_mask > 0)


    gt_foreground = np.any(gt_rgb > 0, axis=2)
    gt_contribution = np.sum(gt_foreground & (neck_mask == 0))

    return {
        'total_pixels': total_pixels,
        'combined_pixels': combined_pixels,
        'gt_contribution': gt_contribution,
        'neck_contribution': neck_pixels,
        'combined_percentage': (combined_pixels / total_pixels) * 100,
        'gt_percentage': (gt_contribution / total_pixels) * 100,
        'neck_percentage': (neck_pixels / total_pixels) * 100,
        'image_size': {'width': W, 'height': H}
    }
