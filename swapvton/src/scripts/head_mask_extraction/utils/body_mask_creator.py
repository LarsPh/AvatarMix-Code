import numpy as np
from typing import Tuple

from .mask_creator import HEAD_COLORS
from .neck_mask_creator import NECK_COLOR


def find_matching_pixels(seg_image: np.ndarray, target_color: Tuple[int, int, int],
                        tolerance: int = 5) -> np.ndarray:

    r, g, b = target_color


    r_match = np.abs(seg_image[:, :, 0].astype(int) - r) <= tolerance
    g_match = np.abs(seg_image[:, :, 1].astype(int) - g) <= tolerance
    b_match = np.abs(seg_image[:, :, 2].astype(int) - b) <= tolerance


    return r_match & g_match & b_match


def create_body_mask(seg_image: np.ndarray, tolerance: int = 5) -> np.ndarray:


    non_bg = np.any(seg_image > 0, axis=2)


    for color_name, color in HEAD_COLORS.items():
        head_pixels = find_matching_pixels(seg_image, color, tolerance)
        non_bg = non_bg & ~head_pixels


    neck_pixels = find_matching_pixels(seg_image, NECK_COLOR, tolerance)
    non_bg = non_bg & ~neck_pixels


    body_mask = (non_bg * 255).astype(np.uint8)

    return body_mask
