import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Tuple, List, Dict
from loguru import logger


def crop_with_bbox(image: np.ndarray, bbox_metadata: Dict) -> np.ndarray:

    x = bbox_metadata['x']
    y = bbox_metadata['y']
    size = bbox_metadata['size']

    if len(image.shape) == 2:

        cropped = image[y:y+size, x:x+size]
    elif len(image.shape) == 3:

        cropped = image[y:y+size, x:x+size, :]
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    return cropped


def crop_segmentations_with_bbox(
    gt_seg: np.ndarray,
    first_swap_seg: np.ndarray,
    body_seg: np.ndarray,
    bbox_metadata: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    logger.debug(f"Cropping 3 segmentations with bbox: x={bbox_metadata['x']}, "
                f"y={bbox_metadata['y']}, size={bbox_metadata['size']}")

    cropped_gt_seg = crop_with_bbox(gt_seg, bbox_metadata)
    cropped_first_swap_seg = crop_with_bbox(first_swap_seg, bbox_metadata)
    cropped_body_seg = crop_with_bbox(body_seg, bbox_metadata)

    logger.debug(f"Cropped segmentations: gt={cropped_gt_seg.shape}, "
                f"first_swap={cropped_first_swap_seg.shape}, body={cropped_body_seg.shape}")

    return cropped_gt_seg, cropped_first_swap_seg, cropped_body_seg


def crop_masks_with_bbox(
    head_mask: np.ndarray,
    neck_mask: np.ndarray,
    body_mask: np.ndarray,
    bbox_metadata: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    logger.debug(f"Cropping 3 masks with bbox: x={bbox_metadata['x']}, "
                f"y={bbox_metadata['y']}, size={bbox_metadata['size']}")

    cropped_head_mask = crop_with_bbox(head_mask, bbox_metadata)
    cropped_neck_mask = crop_with_bbox(neck_mask, bbox_metadata)
    cropped_body_mask = crop_with_bbox(body_mask, bbox_metadata)

    logger.debug(f"Cropped masks: head={cropped_head_mask.shape}, "
                f"neck={cropped_neck_mask.shape}, body={cropped_body_mask.shape}")

    return cropped_head_mask, cropped_neck_mask, cropped_body_mask


def crop_rgb_images_with_bbox(
    first_swap_rgb: np.ndarray,
    donator_rgb: np.ndarray,
    bbox_metadata: Dict
) -> Tuple[np.ndarray, np.ndarray]:

    logger.debug(f"Cropping 2 RGB images for visualization with bbox: x={bbox_metadata['x']}, "
                f"y={bbox_metadata['y']}, size={bbox_metadata['size']}")

    cropped_first_swap_rgb = crop_with_bbox(first_swap_rgb, bbox_metadata)
    cropped_donator_rgb = crop_with_bbox(donator_rgb, bbox_metadata)

    logger.debug(f"Cropped RGB images: first_swap={cropped_first_swap_rgb.shape}, "
                f"donator={cropped_donator_rgb.shape}")

    return cropped_first_swap_rgb, cropped_donator_rgb


def save_validation_outputs(
    output_dir: Path,
    neck_mode: str,
    fullbody_combined: np.ndarray,
    body_mask_fullbody: np.ndarray,
    portrait_combined: np.ndarray,
    portrait_gt: np.ndarray,
    head_mask_portrait: np.ndarray,
    neck_mask_portrait: np.ndarray,
    body_mask_portrait: np.ndarray,
    segmentation_gt_portrait: np.ndarray,
    segmentation_first_swap_portrait: np.ndarray,
    segmentation_body_portrait: np.ndarray,
    bbox_metadata: Dict,
    portrait_first_swap: np.ndarray = None,
    portrait_donator: np.ndarray = None
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)


    file_count = 11
    if portrait_first_swap is not None:
        file_count += 1
    if portrait_donator is not None:
        file_count += 1

    logger.info(f"Saving {file_count} validation outputs to: {output_dir}")


    cv2.imwrite(str(output_dir / "fullbody_combined.jpg"),
                cv2.cvtColor(fullbody_combined, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    cv2.imwrite(str(output_dir / "portrait_combined.jpg"),
                cv2.cvtColor(portrait_combined, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    cv2.imwrite(str(output_dir / "portrait_gt.jpg"),
                cv2.cvtColor(portrait_gt, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])


    if portrait_first_swap is not None:
        cv2.imwrite(str(output_dir / "portrait_first_swap.jpg"),
                    cv2.cvtColor(portrait_first_swap, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.debug("Saved portrait_first_swap.jpg for visualization")

    if portrait_donator is not None:
        cv2.imwrite(str(output_dir / "portrait_donator.jpg"),
                    cv2.cvtColor(portrait_donator, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        logger.debug("Saved portrait_donator.jpg for visualization")


    cv2.imwrite(str(output_dir / "body_mask_fullbody.png"), body_mask_fullbody)
    cv2.imwrite(str(output_dir / "head_mask_portrait.png"), head_mask_portrait)
    cv2.imwrite(str(output_dir / "neck_mask_portrait.png"), neck_mask_portrait)
    cv2.imwrite(str(output_dir / "body_mask_portrait.png"), body_mask_portrait)


    cv2.imwrite(str(output_dir / "segmentation_gt_portrait.png"),
                cv2.cvtColor(segmentation_gt_portrait, cv2.COLOR_RGB2BGR))

    cv2.imwrite(str(output_dir / "segmentation_first_swap_portrait.png"),
                cv2.cvtColor(segmentation_first_swap_portrait, cv2.COLOR_RGB2BGR))

    cv2.imwrite(str(output_dir / "segmentation_body_portrait.png"),
                cv2.cvtColor(segmentation_body_portrait, cv2.COLOR_RGB2BGR))


    bbox_with_mode = bbox_metadata.copy()
    bbox_with_mode['neck_mode'] = neck_mode

    with open(output_dir / "bbox_metadata.yaml", 'w') as f:
        yaml.dump(bbox_with_mode, f, default_flow_style=False, sort_keys=False)


    rgb_count = 3 + (1 if portrait_first_swap is not None else 0) + (1 if portrait_donator is not None else 0)
    logger.info(f"Successfully saved {file_count} files: {rgb_count} RGB, 4 masks, 3 segmentations, 1 metadata")


def verify_outputs_exist(output_dir: Path) -> bool:

    required_files = [
        "fullbody_combined.jpg",
        "body_mask_fullbody.png",
        "portrait_combined.jpg",
        "portrait_gt.jpg",
        "head_mask_portrait.png",
        "neck_mask_portrait.png",
        "body_mask_portrait.png",
        "segmentation_gt_portrait.png",
        "segmentation_first_swap_portrait.png",
        "segmentation_body_portrait.png",
        "bbox_metadata.yaml"
    ]

    for filename in required_files:
        if not (output_dir / filename).exists():
            logger.warning(f"Missing file: {output_dir / filename}")
            return False

    return True
