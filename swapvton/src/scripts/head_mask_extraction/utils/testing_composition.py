import numpy as np
import cv2
from typing import Tuple, Literal
from loguru import logger

from .neck_mask_creator import create_neck_mask_from_segmentation, dilate_neck_mask_for_composition


NeckMode = Literal["first_swap_only", "donator_only", "union"]


def create_neck_mask_with_mode(
    first_swap_seg: np.ndarray,
    donator_seg: np.ndarray,
    head_mask: np.ndarray,
    donator_body_mask: np.ndarray,
    mode: NeckMode,
    tolerance: int = 5,
    neck_selection_ratio: float = 1/3,
    neck_dilation_ratio: float = 0.05
) -> Tuple[np.ndarray, dict]:

    if mode not in ["first_swap_only", "donator_only", "union"]:
        raise ValueError(f"Invalid neck mode: {mode}. Must be one of: first_swap_only, donator_only, union")

    H, W = first_swap_seg.shape[:2]
    if donator_seg.shape[:2] != (H, W) or head_mask.shape != (H, W) or donator_body_mask.shape != (H, W):
        raise ValueError(f"Shape mismatch: first_swap={first_swap_seg.shape}, "
                        f"donator={donator_seg.shape}, head_mask={head_mask.shape}, "
                        f"body_mask={donator_body_mask.shape}")

    logger.debug(f"Creating neck mask with mode={mode}, tolerance={tolerance}, ratio={neck_selection_ratio}")

    if mode == "first_swap_only":

        base_neck, stats = create_neck_mask_from_segmentation(
            first_swap_seg, head_mask, tolerance, neck_selection_ratio
        )
        logger.debug(f"Base first_swap neck: {stats['neck_pixels']} pixels")


        dilated_neck = dilate_neck_mask_for_composition(base_neck, head_mask, neck_dilation_ratio)


        head_binary = head_mask > 0
        body_binary = donator_body_mask > 0
        dilated_binary = dilated_neck > 0
        final_neck = dilated_binary & ~head_binary & ~body_binary
        neck_mask = (final_neck * 255).astype(np.uint8)


        stats['mode'] = 'first_swap_only'
        stats['source'] = 'first_swap'
        stats['neck_pixels'] = int(np.sum(final_neck))
        stats['base_neck_pixels'] = int(np.sum(base_neck > 0))
        stats['dilated_neck_pixels'] = int(np.sum(dilated_neck > 0))
        logger.debug(f"Created first_swap_only neck: base={stats['base_neck_pixels']}, "
                    f"dilated={stats['dilated_neck_pixels']}, final={stats['neck_pixels']} pixels")

    elif mode == "donator_only":

        base_neck, stats = create_neck_mask_from_segmentation(
            donator_seg, head_mask, tolerance, neck_selection_ratio
        )
        logger.debug(f"Base donator neck: {stats['neck_pixels']} pixels")


        dilated_neck = dilate_neck_mask_for_composition(base_neck, head_mask, neck_dilation_ratio)


        head_binary = head_mask > 0
        body_binary = donator_body_mask > 0
        dilated_binary = dilated_neck > 0
        final_neck = dilated_binary & ~head_binary & ~body_binary
        neck_mask = (final_neck * 255).astype(np.uint8)


        stats['mode'] = 'donator_only'
        stats['source'] = 'donator'
        stats['neck_pixels'] = int(np.sum(final_neck))
        stats['base_neck_pixels'] = int(np.sum(base_neck > 0))
        stats['dilated_neck_pixels'] = int(np.sum(dilated_neck > 0))
        logger.debug(f"Created donator_only neck: base={stats['base_neck_pixels']}, "
                    f"dilated={stats['dilated_neck_pixels']}, final={stats['neck_pixels']} pixels")

    elif mode == "union":

        neck_mask, stats = _create_union_neck_mask(
            first_swap_seg, donator_seg, head_mask, donator_body_mask,
            tolerance, neck_selection_ratio, neck_dilation_ratio
        )

    return neck_mask, stats


def _create_union_neck_mask(
    first_swap_seg: np.ndarray,
    donator_seg: np.ndarray,
    head_mask: np.ndarray,
    donator_body_mask: np.ndarray,
    tolerance: int,
    neck_selection_ratio: float,
    neck_dilation_ratio: float
) -> Tuple[np.ndarray, dict]:

    logger.debug("Creating union neck mask from both sources")


    neck_first, stats_first = create_neck_mask_from_segmentation(
        first_swap_seg, head_mask, tolerance, neck_selection_ratio
    )
    logger.debug(f"First-swap neck: {stats_first['neck_pixels']} pixels "
                f"from {stats_first['num_components']} components")


    neck_donator, stats_donator = create_neck_mask_from_segmentation(
        donator_seg, head_mask, tolerance, neck_selection_ratio
    )
    logger.debug(f"Donator neck: {stats_donator['neck_pixels']} pixels "
                f"from {stats_donator['num_components']} components")


    neck_union = (neck_first > 0) | (neck_donator > 0)
    neck_union_uint8 = (neck_union * 255).astype(np.uint8)
    union_pixels = np.sum(neck_union)
    logger.debug(f"Union neck: {union_pixels} pixels")


    dilated_union = dilate_neck_mask_for_composition(neck_union_uint8, head_mask, neck_dilation_ratio)
    dilated_union_binary = dilated_union > 0
    dilated_pixels = np.sum(dilated_union_binary)
    logger.debug(f"Dilated union: {dilated_pixels} pixels (expansion: {dilated_pixels - union_pixels})")


    head_binary = head_mask > 0
    neck_no_head = dilated_union_binary & ~head_binary
    no_head_pixels = np.sum(neck_no_head)
    excluded_head_pixels = dilated_pixels - no_head_pixels
    logger.debug(f"After excluding head overlaps: {no_head_pixels} pixels "
                f"(excluded {excluded_head_pixels} pixels)")


    body_binary = donator_body_mask > 0
    neck_final = neck_no_head & ~body_binary
    final_pixels = np.sum(neck_final)
    excluded_body_pixels = no_head_pixels - final_pixels
    logger.debug(f"After excluding donator body overlaps: {final_pixels} pixels "
                f"(excluded {excluded_body_pixels} pixels)")


    neck_mask = (neck_final * 255).astype(np.uint8)


    stats = {
        'mode': 'union',
        'source': 'union',
        'neck_pixels': int(final_pixels),
        'head_pixels': int(stats_first['head_pixels']),
        'num_components': stats_first['num_components'] + stats_donator['num_components'],
        'distance_threshold': stats_first['distance_threshold'],
        'is_large_neck': final_pixels > 2 * stats_first['head_pixels'],
        'union_details': {
            'first_swap_pixels': int(stats_first['neck_pixels']),
            'donator_pixels': int(stats_donator['neck_pixels']),
            'union_pixels': int(union_pixels),
            'dilated_union_pixels': int(dilated_pixels),
            'excluded_head_pixels': int(excluded_head_pixels),
            'excluded_body_pixels': int(excluded_body_pixels),
            'final_pixels': int(final_pixels)
        }
    }

    logger.debug(f"Union neck final: {final_pixels} pixels "
                f"(first={stats_first['neck_pixels']}, donator={stats_donator['neck_pixels']}, "
                f"union={union_pixels})")

    return neck_mask, stats


def combine_three_way_rgb(
    gt_head_rgb: np.ndarray,
    first_swap_rgb: np.ndarray,
    donator_body_rgb: np.ndarray,
    head_mask: np.ndarray,
    neck_mask: np.ndarray,
    body_mask: np.ndarray
) -> np.ndarray:

    H, W = gt_head_rgb.shape[:2]


    if (first_swap_rgb.shape[:2] != (H, W) or donator_body_rgb.shape[:2] != (H, W) or
        head_mask.shape != (H, W) or neck_mask.shape != (H, W) or body_mask.shape != (H, W)):
        raise ValueError(f"Shape mismatch in inputs: gt={gt_head_rgb.shape}, "
                        f"swap={first_swap_rgb.shape}, donator={donator_body_rgb.shape}, "
                        f"head_mask={head_mask.shape}, neck_mask={neck_mask.shape}, "
                        f"body_mask={body_mask.shape}")

    logger.debug(f"Combining three RGB sources with BODY > HEAD > NECK priority: shape={gt_head_rgb.shape}")


    combined = np.zeros((H, W, 3), dtype=np.uint8)


    head_bool = head_mask > 0
    neck_bool = neck_mask > 0
    body_bool = body_mask > 0


    head_pixels = np.sum(head_bool)
    neck_pixels = np.sum(neck_bool)
    body_pixels = np.sum(body_bool)

    logger.debug(f"Region pixel counts: head={head_pixels}, neck={neck_pixels}, body={body_pixels}")


    combined[neck_bool] = first_swap_rgb[neck_bool]


    combined[head_bool] = gt_head_rgb[head_bool]


    combined[body_bool] = donator_body_rgb[body_bool]


    head_neck_overlap = np.sum(head_bool & neck_bool)
    head_body_overlap = np.sum(head_bool & body_bool)
    neck_body_overlap = np.sum(neck_bool & body_bool)

    if head_neck_overlap > 0:
        logger.debug(f"Head-neck overlap: {head_neck_overlap} pixels (head wins)")
    if head_body_overlap > 0:
        logger.debug(f"Head-body overlap: {head_body_overlap} pixels (BODY WINS - physical occlusion preserved)")
    if neck_body_overlap > 0:
        logger.debug(f"Neck-body overlap: {neck_body_overlap} pixels (body wins)")

    total_combined_pixels = np.sum(np.any(combined > 0, axis=2))
    logger.debug(f"Combined image: {total_combined_pixels} non-black pixels")

    return combined
