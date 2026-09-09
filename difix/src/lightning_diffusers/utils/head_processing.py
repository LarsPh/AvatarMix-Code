import numpy as np
import torch
from typing import Tuple, Optional
from PIL import Image
from loguru import logger


def crop_head_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    target_resolution: int = 224,
    padding_ratio: float = 0.1
) -> Tuple[np.ndarray, bool]:

    try:

        if mask.max() > 1:
            binary_mask = (mask > 128).astype(np.uint8)
        else:
            binary_mask = mask.astype(np.uint8)


        if len(image.shape) == 3:

            masked_image = image.copy()
            for c in range(image.shape[2]):
                masked_image[:, :, c] = masked_image[:, :, c] * binary_mask
        else:

            masked_image = image * binary_mask


        nonzero_indices = np.nonzero(binary_mask)

        if len(nonzero_indices[0]) == 0:
            logger.warning("No non-zero pixels found in head mask")
            return _create_fallback_crop(masked_image, target_resolution), False


        y_min, y_max = nonzero_indices[0].min(), nonzero_indices[0].max()
        x_min, x_max = nonzero_indices[1].min(), nonzero_indices[1].max()

        x, y = x_min, y_min
        w, h = x_max - x_min + 1, y_max - y_min + 1


        padding_x = int(w * padding_ratio)
        padding_y = int(h * padding_ratio)


        x_start = max(0, x - padding_x)
        y_start = max(0, y - padding_y)
        x_end = min(masked_image.shape[1], x + w + padding_x)
        y_end = min(masked_image.shape[0], y + h + padding_y)


        head_crop = masked_image[y_start:y_end, x_start:x_end]

        if head_crop.size == 0:
            logger.warning("Empty crop resulted from bounding box")
            return _create_fallback_crop(masked_image, target_resolution), False


        resized_head = _resize_with_padding(head_crop, target_resolution)

        logger.debug(f"Successfully cropped head: bbox=({x},{y},{w},{h}), "
                    f"padded=({x_start},{y_start},{x_end-x_start},{y_end-y_start}), "
                    f"final_size={resized_head.shape}")

        return resized_head, True

    except Exception as e:
        logger.warning(f"Error during head cropping: {e}")
        return _create_fallback_crop(image, target_resolution), False


def _create_fallback_crop(image: np.ndarray, target_resolution: int) -> np.ndarray:

    h, w = image.shape[:2]
    center_x, center_y = w // 2, h // 2


    crop_size = min(h, w) // 2

    x_start = max(0, center_x - crop_size)
    y_start = max(0, center_y - crop_size)
    x_end = min(w, center_x + crop_size)
    y_end = min(h, center_y + crop_size)

    crop = image[y_start:y_end, x_start:x_end]
    return _resize_with_padding(crop, target_resolution)


def _resize_with_padding(image: np.ndarray, target_size: int) -> np.ndarray:

    h, w = image.shape[:2]


    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)


    if len(image.shape) == 3:
        pil_image = Image.fromarray(image)
    else:
        pil_image = Image.fromarray(image, mode='L')


    resized_pil = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    resized = np.array(resized_pil)


    if len(image.shape) == 3:
        padded = np.zeros((target_size, target_size, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=image.dtype)


    start_y = (target_size - new_h) // 2
    start_x = (target_size - new_w) // 2

    if len(image.shape) == 3:
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    else:
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized

    return padded


def load_and_crop_head(
    image_path: str,
    mask_path: str,
    target_resolution: int = 224
) -> Tuple[torch.Tensor, bool]:

    try:

        image_pil = Image.open(image_path).convert('RGB')
        if image_pil is None:
            raise ValueError(f"Could not load image: {image_path}")

        image = np.array(image_pil)

        mask_pil = Image.open(mask_path).convert('L')
        if mask_pil is None:
            raise ValueError(f"Could not load mask: {mask_path}")

        mask = np.array(mask_pil)


        head_crop, success = crop_head_with_mask(image, mask, target_resolution)


        head_tensor = torch.from_numpy(head_crop).permute(2, 0, 1).float() / 255.0

        return head_tensor, success

    except Exception as e:
        logger.error(f"Error loading and cropping head: {e}")

        dummy_tensor = torch.zeros(3, target_resolution, target_resolution)
        return dummy_tensor, False


def batch_crop_heads(
    image_paths: list,
    mask_paths: list,
    target_resolution: int = 224
) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size = len(image_paths)
    head_batch = torch.zeros(batch_size, 3, target_resolution, target_resolution)
    success_mask = torch.zeros(batch_size, dtype=torch.bool)

    for i, (img_path, mask_path) in enumerate(zip(image_paths, mask_paths)):
        head_tensor, success = load_and_crop_head(img_path, mask_path, target_resolution)
        head_batch[i] = head_tensor
        success_mask[i] = success

    successful_crops = success_mask.sum().item()
    logger.info(f"Successfully cropped {successful_crops}/{batch_size} heads")

    return head_batch, success_mask


def validate_head_mask_data(
    image_path: str,
    mask_path: str,
    min_mask_area: int = 1000
) -> bool:

    try:

        import os
        if not os.path.exists(image_path):
            logger.debug(f"Image file not found: {image_path}")
            return False

        if not os.path.exists(mask_path):
            logger.debug(f"Mask file not found: {mask_path}")
            return False


        mask_pil = Image.open(mask_path).convert('L')
        if mask_pil is None:
            logger.debug(f"Could not load mask: {mask_path}")
            return False

        mask = np.array(mask_pil)


        if mask.max() > 1:
            binary_mask = (mask > 128).astype(np.uint8)
        else:
            binary_mask = mask

        mask_area = np.sum(binary_mask > 0)

        if mask_area < min_mask_area:
            logger.debug(f"Mask area too small: {mask_area} < {min_mask_area}")
            return False

        return True

    except Exception as e:
        logger.debug(f"Validation error for {image_path}: {e}")
        return False


def compute_head_crop_statistics(
    image_paths: list,
    mask_paths: list
) -> dict:

    stats = {
        "total_images": len(image_paths),
        "successful_crops": 0,
        "failed_crops": 0,
        "average_mask_area": 0,
        "bbox_sizes": [],
        "aspect_ratios": []
    }

    total_mask_area = 0

    for img_path, mask_path in zip(image_paths, mask_paths):
        try:

            mask_pil = Image.open(mask_path).convert('L')
            if mask_pil is None:
                stats["failed_crops"] += 1
                continue

            mask = np.array(mask_pil)


            if mask.max() > 1:
                binary_mask = (mask > 128).astype(np.uint8)
            else:
                binary_mask = mask.astype(np.uint8)


            nonzero_indices = np.nonzero(binary_mask)

            if len(nonzero_indices[0]) == 0:
                stats["failed_crops"] += 1
                continue


            y_min, y_max = nonzero_indices[0].min(), nonzero_indices[0].max()
            x_min, x_max = nonzero_indices[1].min(), nonzero_indices[1].max()

            x, y = x_min, y_min
            w, h = x_max - x_min + 1, y_max - y_min + 1


            stats["successful_crops"] += 1
            stats["bbox_sizes"].append((w, h))
            stats["aspect_ratios"].append(w / h if h > 0 else 1.0)

            mask_area = np.sum(binary_mask > 0)
            total_mask_area += mask_area

        except Exception as e:
            logger.debug(f"Error processing {img_path}: {e}")
            stats["failed_crops"] += 1

    if stats["successful_crops"] > 0:
        stats["average_mask_area"] = total_mask_area / stats["successful_crops"]
        stats["average_bbox_size"] = (
            np.mean([w for w, h in stats["bbox_sizes"]]),
            np.mean([h for w, h in stats["bbox_sizes"]])
        )
        stats["average_aspect_ratio"] = np.mean(stats["aspect_ratios"])

    logger.info(f"Head crop statistics: {stats['successful_crops']}/{stats['total_images']} successful")

    return stats


def find_bounding_box_tensor(mask: torch.Tensor, padding_ratio: float = 0.1) -> Tuple[int, int, int, int, bool]:

    try:

        if mask.max() > 1:
            binary_mask = (mask > 128).float()
        else:
            binary_mask = mask.float()


        nonzero_coords = torch.nonzero(binary_mask, as_tuple=False)

        if len(nonzero_coords) == 0:
            logger.warning("No non-zero pixels found in head mask tensor")
            return 0, mask.shape[0], 0, mask.shape[1], False


        y_coords, x_coords = nonzero_coords[:, 0], nonzero_coords[:, 1]
        y_min, y_max = y_coords.min().item(), y_coords.max().item()
        x_min, x_max = x_coords.min().item(), x_coords.max().item()


        h, w = y_max - y_min + 1, x_max - x_min + 1
        padding_y = int(h * padding_ratio)
        padding_x = int(w * padding_ratio)


        H, W = mask.shape[0], mask.shape[1]
        y_start = max(0, y_min - padding_y)
        y_end = min(H, y_max + 1 + padding_y)
        x_start = max(0, x_min - padding_x)
        x_end = min(W, x_max + 1 + padding_x)

        return y_start, y_end, x_start, x_end, True

    except Exception as e:
        logger.warning(f"Error finding bounding box in tensor: {e}")
        return 0, mask.shape[0], 0, mask.shape[1], False


def crop_head_with_mask_tensor(
    image: torch.Tensor,
    mask: torch.Tensor,
    target_resolution: int = 224,
    padding_ratio: float = 0.1,
    padding_value: float = 0.0
) -> Tuple[torch.Tensor, bool]:

    try:
        import torch.nn.functional as F


        if mask.max() > 1:
            binary_mask = (mask > 128).float()
        else:
            binary_mask = mask.float()


        masked_image = image.clone()
        for c in range(image.shape[0]):
            masked_image[c] = masked_image[c] * binary_mask


        y_start, y_end, x_start, x_end, bbox_success = find_bounding_box_tensor(mask, padding_ratio)

        if not bbox_success:

            H, W = masked_image.shape[1], masked_image.shape[2]
            center_y, center_x = H // 2, W // 2
            crop_size = min(H, W) // 2

            y_start = max(0, center_y - crop_size)
            y_end = min(H, center_y + crop_size)
            x_start = max(0, center_x - crop_size)
            x_end = min(W, center_x + crop_size)


        head_crop = masked_image[:, y_start:y_end, x_start:x_end]

        if head_crop.numel() == 0:
            logger.warning("Empty crop resulted from bounding box")

            dummy_crop = torch.zeros(image.shape[0], target_resolution, target_resolution,
                                   device=image.device, dtype=image.dtype)
            return dummy_crop, False


        _, crop_height, crop_width = head_crop.shape
        crop_aspect_ratio = crop_width / crop_height


        if crop_aspect_ratio > 1.0:

            padding_needed = crop_width - crop_height
            padding_top = padding_needed // 2
            padding_bottom = padding_needed - padding_top
            padded_head = F.pad(head_crop, (0, 0, padding_top, padding_bottom),
                               mode='constant', value=padding_value)
        elif crop_aspect_ratio < 1.0:

            padding_needed = crop_height - crop_width
            padding_left = padding_needed // 2
            padding_right = padding_needed - padding_left
            padded_head = F.pad(head_crop, (padding_left, padding_right, 0, 0),
                               mode='constant', value=padding_value)
        else:

            padded_head = head_crop


        resized_head = F.interpolate(
            padded_head.unsqueeze(0),
            size=(target_resolution, target_resolution),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        return resized_head, bbox_success

    except Exception as e:
        logger.warning(f"Error during tensor head cropping: {e}")

        dummy_crop = torch.zeros(image.shape[0], target_resolution, target_resolution,
                               device=image.device, dtype=image.dtype)
        return dummy_crop, False


def batch_crop_heads_tensor(
    images: torch.Tensor,
    masks: torch.Tensor,
    target_resolution: int = 224,
    padding_ratio: float = 0.1,
    padding_value: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:

    try:
        batch_size = images.shape[0]
        cropped_heads = []
        success_flags = []


        for i in range(batch_size):
            image_i = images[i]
            mask_i = masks[i]

            cropped_head, success = crop_head_with_mask_tensor(
                image_i, mask_i, target_resolution, padding_ratio, padding_value
            )

            cropped_heads.append(cropped_head)
            success_flags.append(success)


        cropped_heads_batch = torch.stack(cropped_heads, dim=0)
        success_mask = torch.tensor(success_flags, device=images.device, dtype=torch.bool)

        successful_crops = success_mask.sum().item()
        logger.debug(f"Tensor batch crop: {successful_crops}/{batch_size} successful")

        return cropped_heads_batch, success_mask

    except Exception as e:
        logger.error(f"Error in batch tensor head cropping: {e}")

        dummy_batch = torch.zeros(images.shape[0], images.shape[1], target_resolution, target_resolution,
                                device=images.device, dtype=images.dtype)
        dummy_success = torch.zeros(images.shape[0], device=images.device, dtype=torch.bool)
        return dummy_batch, dummy_success
