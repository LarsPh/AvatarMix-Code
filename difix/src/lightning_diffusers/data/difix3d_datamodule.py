import os
import re
import glob
import random
from pathlib import Path
from typing import List, Optional, Union, Any, Dict, Tuple

import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
import numpy as np
from diffusers.utils import load_image
from torchvision import transforms


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


def extract_subject_id_from_restored_dir(dir_path: str, pattern: str = r"restored_(\d+)_from_") -> Optional[str]:


    dir_name = os.path.basename(dir_path.rstrip('/'))


    match = re.search(pattern, dir_name)
    if match:
        return match.group(1)

    logger.warning(f"Could not extract subject ID from directory: {dir_name}")
    return None


def construct_gt_path(degraded_path: str, subject_id: str, gt_dataset_root: str) -> str:


    path_obj = Path(degraded_path)


    camera_dir = None
    filename = path_obj.name

    for part in path_obj.parts:
        if re.match(r'\d{3}_p[+-]\d{2}', part):
            camera_dir = part
            break

    if camera_dir is None:
        logger.warning(f"Could not find camera directory in path: {degraded_path}")
        return ""


    gt_path = str(Path(gt_dataset_root) / subject_id / camera_dir / filename)
    return gt_path


def map_degraded_to_gt_paths(
    degraded_paths: List[str],
    gt_dataset_root: str,
    subject_pattern: str = r"restored_(\d+)_from_",
    validate_existence: bool = True
) -> List[Tuple[str, str]]:

    valid_pairs = []
    missing_gt = []

    for degraded_path in degraded_paths:


        path_parts = Path(degraded_path).parts
        restored_dir = None

        for part in path_parts:
            if part.startswith('restored_'):
                restored_dir = part
                break

        if not restored_dir:
            logger.debug(f"No restored directory found in path: {degraded_path}")
            continue


        subject_id = extract_subject_id_from_restored_dir(restored_dir, subject_pattern)

        if subject_id is None:
            continue


        gt_path = construct_gt_path(degraded_path, subject_id, gt_dataset_root)

        if not gt_path:
            continue


        if validate_existence:
            if os.path.exists(gt_path):
                valid_pairs.append((degraded_path, gt_path))
            else:
                missing_gt.append((degraded_path, gt_path))
        else:
            valid_pairs.append((degraded_path, gt_path))


    logger.info(f"Mapped {len(valid_pairs)} degraded-GT pairs from {len(degraded_paths)} degraded images")
    if missing_gt:
        logger.warning(f"Found {len(missing_gt)} degraded images without corresponding GT images")
        if len(missing_gt) <= 5:
            for deg, gt in missing_gt:
                logger.debug(f"Missing GT: {deg} -> {gt}")

    return valid_pairs


def filter_paths_by_subject_pairs(
    degraded_paths: List[str],
    subject_pairs: Optional[List[str]] = None,
    subject_a: Optional[str] = None,
    subject_b: Optional[str] = None,
    filter_enabled: bool = True,
    extract_pattern: str = r"restored_(\d+)_from_",
) -> List[str]:

    if not filter_enabled:
        logger.debug("Subject filtering disabled, processing all found paths")
        return degraded_paths


    target_subjects = set()

    if subject_pairs:

        for pair_str in subject_pairs:
            if ',' in pair_str:
                subj_1, subj_2 = pair_str.split(',', 1)
                target_subjects.add(subj_1.strip())
                target_subjects.add(subj_2.strip())
            else:
                logger.warning(f"Invalid subject pair format: '{pair_str}' (expected 'subj_a,subj_b')")
        logger.info(f"Multi-pair filtering: targeting {len(target_subjects)} subjects from {len(subject_pairs)} pairs")

    elif subject_a and subject_b:

        target_subjects.add(subject_a.strip())
        target_subjects.add(subject_b.strip())
        logger.info(f"Single-pair filtering: targeting subjects {subject_a} and {subject_b}")

    else:
        logger.warning("No subject filtering configuration provided, processing all paths")
        return degraded_paths

    if not target_subjects:
        logger.warning("No valid target subjects found, processing all paths")
        return degraded_paths


    filtered_paths = []
    subject_counts = {}

    for degraded_path in degraded_paths:

        path_parts = Path(degraded_path).parts
        restored_dir = None

        for part in path_parts:
            if part.startswith('restored_'):
                restored_dir = part
                break

        if not restored_dir:
            logger.debug(f"No restored directory found in path: {degraded_path}")
            continue


        subject_id = extract_subject_id_from_restored_dir(restored_dir, extract_pattern)

        if subject_id in target_subjects:
            filtered_paths.append(degraded_path)
            subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1


    logger.info(f"Subject filtering results:")
    logger.info(f"  Original paths: {len(degraded_paths)}")
    logger.info(f"  Filtered paths: {len(filtered_paths)}")
    logger.info(f"  Target subjects: {sorted(target_subjects)}")
    logger.info(f"  Found subjects: {sorted(subject_counts.keys())}")

    for subject_id, count in sorted(subject_counts.items()):
        logger.info(f"    {subject_id}: {count} images")

    missing_subjects = target_subjects - set(subject_counts.keys())
    if missing_subjects:
        logger.warning(f"Target subjects not found in data: {sorted(missing_subjects)}")

    return filtered_paths


def split_subjects_train_val(
    subject_pairs: List[Tuple[str, str]],
    train_ratio: float = 0.8,
    seed: int = 42,
    use_same_subjects_for_val: bool = False
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:


    subject_groups = {}
    for degraded_path, gt_path in subject_pairs:

        path_parts = Path(degraded_path).parts
        restored_dir = None

        for part in path_parts:
            if part.startswith('restored_'):
                restored_dir = part
                break

        if restored_dir:
            subject_id = extract_subject_id_from_restored_dir(restored_dir)
            if subject_id:
                if subject_id not in subject_groups:
                    subject_groups[subject_id] = []
                subject_groups[subject_id].append((degraded_path, gt_path))


    subject_ids = list(subject_groups.keys())

    if use_same_subjects_for_val:

        train_subject_ids = subject_ids
        val_subject_ids = subject_ids


        all_pairs = []
        for subject_id in subject_ids:
            all_pairs.extend(subject_groups[subject_id])

        train_pairs = all_pairs.copy()
        val_pairs = all_pairs.copy()

        logger.info(f"Same-subject split: {len(subject_ids)} subjects used for both train and val ({len(all_pairs)} pairs each)")

    else:

        random.seed(seed)
        random.shuffle(subject_ids)


        n_train_subjects = int(len(subject_ids) * train_ratio)
        train_subject_ids = subject_ids[:n_train_subjects]
        val_subject_ids = subject_ids[n_train_subjects:]


        train_pairs = []
        val_pairs = []

        for subject_id in train_subject_ids:
            train_pairs.extend(subject_groups[subject_id])

        for subject_id in val_subject_ids:
            val_pairs.extend(subject_groups[subject_id])

        logger.info(f"Random split: {len(train_subject_ids)} train subjects ({len(train_pairs)} pairs), "
                    f"{len(val_subject_ids)} val subjects ({len(val_pairs)} pairs)")

    return train_pairs, val_pairs


class DiFix3DTrainingDataset(Dataset):


    def __init__(
        self,
        degraded_gt_pairs: List[Tuple[str, str]],
        prompts: Optional[Union[str, List[str]]] = None,
        reference_paths: Optional[List[str]] = None,
        reference_strategy: str = "placeholder",
        adaptive_resolution_enabled: bool = False,
        target_resolution: Tuple[int, int] = (448, 896),
        min_bbox_size: Tuple[int, int] = (64, 64),
    ):

        self.degraded_gt_pairs = degraded_gt_pairs
        self.reference_strategy = reference_strategy


        self.adaptive_resolution_enabled = adaptive_resolution_enabled
        self.target_resolution = target_resolution
        self.min_bbox_size = min_bbox_size


        if prompts is None:
            self.prompts = ["remove head-swapping artifacts"] * len(degraded_gt_pairs)
        elif isinstance(prompts, str):
            self.prompts = [prompts] * len(degraded_gt_pairs)
        else:
            self.prompts = prompts


        if reference_strategy == "placeholder":
            self.reference_paths = [None] * len(degraded_gt_pairs)
        else:
            self.reference_paths = reference_paths or [None] * len(degraded_gt_pairs)

    def __len__(self) -> int:
        return len(self.degraded_gt_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        degraded_path, gt_path = self.degraded_gt_pairs[idx]
        prompt = self.prompts[idx]
        reference_path = self.reference_paths[idx]


        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:

            degraded_pil = load_image(degraded_path).convert('RGB')
            target_pil = load_image(gt_path).convert('RGB')


            if self.adaptive_resolution_enabled:

                gt_mask_path = construct_gt_mask_path(gt_path)
                bbox = calculate_bounding_box_from_mask(gt_mask_path, self.min_bbox_size)


                degraded_pil = apply_adaptive_resolution_transform(
                    degraded_pil, target_resolution=self.target_resolution,
                    min_bbox_size=self.min_bbox_size, bbox=bbox
                )

                target_pil = apply_adaptive_resolution_transform(
                    target_pil, target_resolution=self.target_resolution,
                    min_bbox_size=self.min_bbox_size, bbox=bbox
                )


            degraded_tensor = transform(degraded_pil)
            target_tensor = transform(target_pil)


            reference_tensor = None
            if reference_path and os.path.exists(reference_path):
                reference_pil = load_image(reference_path).convert('RGB')


                if self.adaptive_resolution_enabled:

                    gt_mask_path = construct_gt_mask_path(gt_path)
                    bbox = calculate_bounding_box_from_mask(gt_mask_path, self.min_bbox_size)
                    reference_pil = apply_adaptive_resolution_transform(
                        reference_pil, target_resolution=self.target_resolution,
                        min_bbox_size=self.min_bbox_size, bbox=bbox
                    )

                reference_tensor = transform(reference_pil)


            for name, tensor in [("degraded", degraded_tensor), ("target", target_tensor)]:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    logger.warning(f"NaN/inf detected in {name} tensor from {degraded_path if name == 'degraded' else gt_path}, cleaning...")
                    tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0)
                    if name == "degraded":
                        degraded_tensor = tensor
                    else:
                        target_tensor = tensor

            if reference_tensor is not None and (torch.isnan(reference_tensor).any() or torch.isinf(reference_tensor).any()):
                logger.warning(f"NaN/inf detected in reference tensor from {reference_path}, cleaning...")
                reference_tensor = torch.nan_to_num(reference_tensor, nan=0.0, posinf=1.0, neginf=0.0)

            return {
                "degraded": degraded_tensor,
                "target": target_tensor,
                "reference": reference_tensor,
                "prompt": prompt,
                "metadata": {
                    "degraded_path": degraded_path,
                    "target_path": gt_path,
                    "reference_path": reference_path,
                    "index": idx,
                }
            }

        except Exception as e:

            logger.error(f"Failed to load training images at index {idx}: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "degraded_path": degraded_path,
                    "target_path": gt_path,
                    "reference_path": reference_path,
                    "index": idx,
                }
            }


class DiFix3DDataset(Dataset):


    def __init__(
        self,
        image_paths: List[str],
        prompts: Optional[Union[str, List[str]]] = None,
        output_paths: Optional[List[str]] = None,
        skip_existing: bool = True,
        adaptive_resolution_enabled: bool = False,
        target_resolution: Tuple[int, int] = (448, 896),
        min_bbox_size: Tuple[int, int] = (64, 64),
    ):

        self.image_paths = image_paths
        self.output_paths = output_paths or [None] * len(image_paths)


        self.adaptive_resolution_enabled = adaptive_resolution_enabled
        self.target_resolution = target_resolution
        self.min_bbox_size = min_bbox_size


        if prompts is None:
            self.prompts = ["remove degradation"] * len(image_paths)
        elif isinstance(prompts, str):
            self.prompts = [prompts] * len(image_paths)
        else:
            self.prompts = prompts


        if skip_existing and output_paths:
            original_count = len(self.image_paths)
            self._filter_existing()
            filtered_count = len(self.image_paths)
            if original_count != filtered_count:
                print(f"Skip existing: filtered {original_count - filtered_count} images with existing outputs ({filtered_count} remaining)")
            else:
                print(f"Skip existing: no existing outputs found, processing all {original_count} images")

    def _filter_existing(self):

        filtered_paths = []
        filtered_prompts = []
        filtered_outputs = []

        for img_path, prompt, out_path in zip(self.image_paths, self.prompts, self.output_paths):
            if out_path is None or not os.path.exists(out_path):
                filtered_paths.append(img_path)
                filtered_prompts.append(prompt)
                filtered_outputs.append(out_path)

        self.image_paths = filtered_paths
        self.prompts = filtered_prompts
        self.output_paths = filtered_outputs

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]
        output_path = self.output_paths[idx]


        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:

            image_pil = load_image(image_path).convert('RGB')


            if self.adaptive_resolution_enabled:

                mask_path = construct_mask_path(image_path)
                image_pil = apply_adaptive_resolution_transform(
                    image_pil, mask_path=mask_path, target_resolution=self.target_resolution,
                    min_bbox_size=self.min_bbox_size
                )


            image_tensor = transform(image_pil)


            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                logger.warning(f"NaN/inf detected in image tensor from {image_path}, cleaning...")
                image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)

            return {
                "image": image_tensor,
                "prompt": prompt,
                "metadata": {
                    "input_path": image_path,
                    "output_path": output_path,
                    "index": idx,
                }
            }

        except Exception as e:

            logger.error(f"Failed to load image at index {idx}: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "input_path": image_path,
                    "output_path": output_path,
                    "index": idx,
                }
            }


def construct_mask_path(image_path: str) -> str:

    path_obj = Path(image_path)


    filename_stem = path_obj.stem
    cam_dir = path_obj.parent


    mask_path = cam_dir / "mask" / "pha" / f"{filename_stem}.png"


    if not mask_path.exists():
        mask_path = cam_dir / "mask" / "pha" / f"{filename_stem}.jpg"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found: {mask_path}")

    return str(mask_path)


def construct_gt_mask_path(gt_image_path: str) -> str:

    path_obj = Path(gt_image_path)


    filename_stem = path_obj.stem
    cam_dir = path_obj.parent


    mask_path_png = cam_dir / "mask" / "pha" / f"{filename_stem}.png"
    if mask_path_png.exists():
        return str(mask_path_png)


    mask_path_jpg = cam_dir / "mask" / "pha" / f"{filename_stem}.jpg"
    if mask_path_jpg.exists():
        return str(mask_path_jpg)


    raise FileNotFoundError(
        f"GT mask file not found. Tried: {mask_path_png} and {mask_path_jpg}"
    )


def calculate_bounding_box_from_mask(
    mask_path: str,
    min_bbox_size: Tuple[int, int] = (64, 64)
) -> Tuple[int, int, int, int]:

    if not os.path.exists(mask_path):
        mask_path = mask_path.replace(".png", ".jpg")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")


    mask = Image.open(mask_path).convert('L')
    mask_array = np.array(mask)


    non_zero_coords = np.where(mask_array > 0)

    if len(non_zero_coords[0]) == 0:
        raise ValueError(f"Empty mask (no subject pixels): {mask_path}")


    y_min, y_max = non_zero_coords[0].min(), non_zero_coords[0].max()
    x_min, x_max = non_zero_coords[1].min(), non_zero_coords[1].max()


    bbox_width = x_max - x_min + 1
    bbox_height = y_max - y_min + 1

    min_width, min_height = min_bbox_size
    if bbox_width < min_width or bbox_height < min_height:
        raise ValueError(
            f"Bounding box too small: {bbox_width}x{bbox_height} < {min_width}x{min_height} "
            f"(mask: {mask_path})"
        )

    return x_min, y_min, x_max, y_max


def apply_adaptive_resolution_transform(
    image: Image.Image,
    mask_path: Optional[str] = None,
    target_resolution: Tuple[int, int] = (448, 896),
    min_bbox_size: Tuple[int, int] = (64, 64),
    bbox: Optional[Tuple[int, int, int, int]] = None
) -> Image.Image:


    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox

        bbox_width = x_max - x_min + 1
        bbox_height = y_max - y_min + 1
        min_width, min_height = min_bbox_size
        if bbox_width < min_width or bbox_height < min_height:
            raise ValueError(
                f"Provided bounding box too small: {bbox_width}x{bbox_height} < {min_width}x{min_height}"
            )
    elif mask_path is not None:
        x_min, y_min, x_max, y_max = calculate_bounding_box_from_mask(mask_path, min_bbox_size)
    else:
        raise ValueError("Either mask_path or bbox must be provided")


    cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))
    crop_width, crop_height = cropped.size


    target_aspect_ratio = float(target_resolution[0]) / float(target_resolution[1])
    current_aspect_ratio = crop_width / crop_height

    if abs(current_aspect_ratio - target_aspect_ratio) < 1e-6:

        padded = cropped
    elif current_aspect_ratio < target_aspect_ratio:

        target_width = int(crop_height * target_aspect_ratio)
        pad_width = target_width - crop_width


        left_pad = pad_width // 2
        right_pad = pad_width - left_pad
        padded = ImageOps.expand(cropped, border=(left_pad, 0, right_pad, 0), fill=0)
    else:

        target_height = int(crop_width / target_aspect_ratio)
        pad_height = target_height - crop_height


        top_pad = pad_height // 2
        bottom_pad = pad_height - top_pad
        padded = ImageOps.expand(cropped, border=(0, top_pad, 0, bottom_pad), fill=0)


    target_width, target_height = target_resolution
    resized = padded.resize((target_width, target_height), Image.LANCZOS)

    return resized


def apply_adaptive_resolution_transform_to_mask(
    mask_array: np.ndarray,
    bbox: Tuple[int, int, int, int],
    target_resolution: Tuple[int, int] = (448, 896)
) -> np.ndarray:

    try:

        x_min, y_min, x_max, y_max = bbox
        cropped_mask = mask_array[y_min:y_max+1, x_min:x_max+1]
        crop_height, crop_width = cropped_mask.shape


        target_aspect_ratio = float(target_resolution[0]) / float(target_resolution[1])
        current_aspect_ratio = crop_width / crop_height

        if abs(current_aspect_ratio - target_aspect_ratio) < 1e-6:

            padded_mask = cropped_mask
        elif current_aspect_ratio < target_aspect_ratio:

            target_width = int(crop_height * target_aspect_ratio)
            pad_width = target_width - crop_width


            left_pad = pad_width // 2
            right_pad = pad_width - left_pad
            padded_mask = np.pad(cropped_mask, ((0, 0), (left_pad, right_pad)), mode='constant', constant_values=0)
        else:

            target_height = int(crop_width / target_aspect_ratio)
            pad_height = target_height - crop_height


            top_pad = pad_height // 2
            bottom_pad = pad_height - top_pad
            padded_mask = np.pad(cropped_mask, ((top_pad, bottom_pad), (0, 0)), mode='constant', constant_values=0)


        target_width, target_height = target_resolution


        mask_pil = Image.fromarray(padded_mask, mode='L')
        resized_mask_pil = mask_pil.resize((target_width, target_height), Image.NEAREST)
        resized_mask = np.array(resized_mask_pil)

        return resized_mask

    except Exception as e:
        logger.error(f"Error transforming mask with adaptive resolution: {e}")

        target_width, target_height = target_resolution
        return np.zeros((target_height, target_width), dtype=np.uint8)


def collect_image_paths(
    input_dir: Optional[str] = None,
    input_images: Optional[List[str]] = None,
    supported_extensions: Optional[List[str]] = None,
    recursive: bool = True,
    filter_mask_dirs: bool = True,
) -> List[str]:

    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    image_paths = []


    if input_dir:
        search_pattern = "**" if recursive else "*"
        for ext in supported_extensions:
            pattern = os.path.join(input_dir, search_pattern, f"*{ext}")
            image_paths.extend(glob.glob(pattern, recursive=recursive))

            pattern = os.path.join(input_dir, search_pattern, f"*{ext.upper()}")
            image_paths.extend(glob.glob(pattern, recursive=recursive))


    if input_images:
        for img_path in input_images:
            if os.path.exists(img_path):
                image_paths.append(img_path)
            else:
                print(f"Warning: Image not found: {img_path}")


    image_paths = sorted(list(set(image_paths)))


    filtered_paths = []
    for path in image_paths:
        basename = os.path.basename(path)
        dirname_parts = os.path.dirname(path).split(os.sep)


        if "_refined" in basename:
            continue


        if filter_mask_dirs and "mask" in dirname_parts:
            continue

        filtered_paths.append(path)

    return filtered_paths


def generate_output_paths(
    image_paths: List[str],
    output_dir: Optional[str] = None,
    output_suffix: str = "_refined",
) -> List[str]:

    output_paths = []

    for input_path in image_paths:
        if output_dir:


            input_path_obj = Path(input_path)
            input_filename = input_path_obj.name
            parts = input_path_obj.parts


            relative_parts = []


            for i, part in enumerate(parts):
                if part.startswith('restored_') or part.startswith('swapped_'):

                    relative_parts = parts[i:-1]
                    break

            if relative_parts:

                relative_path = os.path.join(*relative_parts)
                output_path = os.path.join(output_dir, relative_path, input_filename)
            else:

                camera_dir = None
                for part in parts:
                    if 'p+' in part or 'p-' in part:
                        camera_dir = part
                        break

                if camera_dir:

                    output_path = os.path.join(output_dir, camera_dir, input_filename)
                else:

                    output_path = os.path.join(output_dir, input_filename)
        else:

            input_dir = os.path.dirname(input_path)
            input_name = os.path.basename(input_path)
            name_without_ext = os.path.splitext(input_name)[0]
            ext = os.path.splitext(input_name)[1]

            output_name = f"{name_without_ext}{output_suffix}{ext}"
            output_path = os.path.join(input_dir, output_name)

        output_paths.append(output_path)

    return output_paths


class DiFix3DDataModule(L.LightningDataModule):


    def __init__(
        self,

        training_mode: bool = False,


        input_dir: Optional[str] = None,
        input_images: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
        output_suffix: str = "_refined",
        skip_existing: bool = True,


        degraded_dir: Optional[str] = None,
        gt_dataset_root: Optional[str] = None,
        train_split_ratio: float = 0.8,
        split_by_subjects: bool = True,
        use_same_subjects_for_val: bool = False,
        ensure_pair_consistency: bool = True,
        validate_gt_existence: bool = True,
        skip_missing_pairs: bool = True,
        min_pairs_per_subject: int = 1,
        extract_subject_pattern: str = r"restored_(\d+)_from_",
        reference_strategy: str = "placeholder",


        filter_by_subjects: bool = True,
        subject_pairs: Optional[List[str]] = None,
        subject_a: Optional[str] = None,
        subject_b: Optional[str] = None,


        prompts: Optional[Union[str, List[str]]] = None,
        batch_size: int = 2,
        num_workers: int = 0,
        supported_extensions: Optional[List[str]] = None,


        adaptive_resolution_enabled: bool = False,
        target_resolution: Tuple[int, int] = (448, 896),
        min_bbox_size: Tuple[int, int] = (64, 64),


        swapped_validation_enabled: bool = False,
        swapped_validation_dir: Optional[str] = None,
        max_validation_cameras: Optional[int] = None,
        visualize_restored_images: bool = False,


        verbose_mapping: bool = False,
        log_split_statistics: bool = True,
        seed: int = 42,
    ):

        super().__init__()


        self.training_mode = training_mode


        self.input_dir = input_dir
        self.input_images = input_images
        self.output_dir = output_dir
        self.output_suffix = output_suffix
        self.skip_existing = skip_existing


        self.degraded_dir = degraded_dir
        self.gt_dataset_root = gt_dataset_root
        self.train_split_ratio = train_split_ratio
        self.split_by_subjects = split_by_subjects
        self.use_same_subjects_for_val = use_same_subjects_for_val
        self.ensure_pair_consistency = ensure_pair_consistency
        self.validate_gt_existence = validate_gt_existence
        self.skip_missing_pairs = skip_missing_pairs
        self.min_pairs_per_subject = min_pairs_per_subject
        self.extract_subject_pattern = extract_subject_pattern
        self.reference_strategy = reference_strategy


        self.filter_by_subjects = filter_by_subjects
        self.subject_pairs = subject_pairs
        self.subject_a = subject_a
        self.subject_b = subject_b


        self.prompts = prompts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.supported_extensions = supported_extensions


        self.adaptive_resolution_enabled = adaptive_resolution_enabled
        self.target_resolution = target_resolution
        self.min_bbox_size = min_bbox_size


        self.swapped_validation_enabled = swapped_validation_enabled
        self.swapped_validation_dir = swapped_validation_dir
        self.max_validation_cameras = max_validation_cameras
        self.visualize_restored_images = visualize_restored_images


        self.verbose_mapping = verbose_mapping
        self.log_split_statistics = log_split_statistics
        self.seed = seed


        self.image_paths = []
        self.output_paths = []
        self.predict_dataset = None


        self.train_pairs = []
        self.val_pairs = []
        self.train_dataset = None
        self.val_dataset = None


        self.swapped_val_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:

        if self.training_mode:

            if stage == "fit" or stage is None:
                self._setup_training_datasets()


                if self.swapped_validation_enabled:
                    self._setup_swapped_validation_dataset()
        else:

            if stage == "predict" or stage is None:
                self._setup_inference_datasets()

    def _setup_inference_datasets(self) -> None:


        self.image_paths = collect_image_paths(
            input_dir=self.input_dir,
            input_images=self.input_images,
            supported_extensions=self.supported_extensions,
        )

        if not self.image_paths:
            raise ValueError("No images found to process")


        self.output_paths = generate_output_paths(
            image_paths=self.image_paths,
            output_dir=self.output_dir,
            output_suffix=self.output_suffix,
        )


        self.predict_dataset = DiFix3DDataset(
            image_paths=self.image_paths,
            prompts=self.prompts,
            output_paths=self.output_paths,
            skip_existing=self.skip_existing,
            adaptive_resolution_enabled=self.adaptive_resolution_enabled,
            target_resolution=self.target_resolution,
            min_bbox_size=self.min_bbox_size,
        )

        logger.info(f"Inference setup complete: {len(self.predict_dataset)} images to process")

    def _setup_training_datasets(self) -> None:

        if not self.degraded_dir or not self.gt_dataset_root:
            raise ValueError("Training mode requires both degraded_dir and gt_dataset_root")


        if self.filter_by_subjects:
            if self.subject_pairs and (self.subject_a or self.subject_b):
                raise ValueError("Cannot use both subject_pairs and subject_a/subject_b simultaneously")

            if not self.subject_pairs and not (self.subject_a and self.subject_b):
                logger.warning("Subject filtering enabled but no subjects specified. Processing all subjects.")
                self.filter_by_subjects = False

            if self.subject_a and not self.subject_b:
                raise ValueError("Single-pair mode requires both subject_a and subject_b")
            if self.subject_b and not self.subject_a:
                raise ValueError("Single-pair mode requires both subject_a and subject_b")

        logger.info(f"Setting up training datasets from:")
        logger.info(f"  Degraded dir: {self.degraded_dir}")
        logger.info(f"  GT dataset root: {self.gt_dataset_root}")

        if self.filter_by_subjects:
            if self.subject_pairs:
                logger.info(f"  Subject filtering: Multi-pair mode with {len(self.subject_pairs)} pairs")
            elif self.subject_a and self.subject_b:
                logger.info(f"  Subject filtering: Single-pair mode ({self.subject_a}, {self.subject_b})")
        else:
            logger.info("  Subject filtering: Disabled (processing all subjects)")


        degraded_paths = collect_image_paths(
            input_dir=self.degraded_dir,
            supported_extensions=self.supported_extensions,
            recursive=True,
        )

        if not degraded_paths:
            raise ValueError(f"No degraded images found in {self.degraded_dir}")

        logger.info(f"Found {len(degraded_paths)} degraded images")


        if self.filter_by_subjects:
            degraded_paths = filter_paths_by_subject_pairs(
                degraded_paths=degraded_paths,
                subject_pairs=self.subject_pairs,
                subject_a=self.subject_a,
                subject_b=self.subject_b,
                filter_enabled=self.filter_by_subjects,
                extract_pattern=self.extract_subject_pattern,
            )

            if not degraded_paths:
                raise ValueError("No degraded images found matching specified subject pairs")
        else:
            logger.info("Subject filtering disabled, processing all found subjects")


        all_pairs = map_degraded_to_gt_paths(
            degraded_paths=degraded_paths,
            gt_dataset_root=self.gt_dataset_root,
            subject_pattern=self.extract_subject_pattern,
            validate_existence=self.validate_gt_existence,
        )

        if not all_pairs:
            raise ValueError("No valid degraded-GT pairs found")


        if self.min_pairs_per_subject > 1:
            filtered_pairs = self._filter_pairs_by_subject_count(all_pairs)
            logger.info(f"Filtered to {len(filtered_pairs)} pairs meeting minimum count requirement")
            all_pairs = filtered_pairs


        self.train_pairs, self.val_pairs = split_subjects_train_val(
            subject_pairs=all_pairs,
            train_ratio=self.train_split_ratio,
            seed=self.seed,
            use_same_subjects_for_val=self.use_same_subjects_for_val,
        )


        self.train_dataset = DiFix3DTrainingDataset(
            degraded_gt_pairs=self.train_pairs,
            prompts=self.prompts,
            reference_strategy=self.reference_strategy,
            adaptive_resolution_enabled=self.adaptive_resolution_enabled,
            target_resolution=self.target_resolution,
            min_bbox_size=self.min_bbox_size,
        )

        self.val_dataset = DiFix3DTrainingDataset(
            degraded_gt_pairs=self.val_pairs,
            prompts=self.prompts,
            reference_strategy=self.reference_strategy,
            adaptive_resolution_enabled=self.adaptive_resolution_enabled,
            target_resolution=self.target_resolution,
            min_bbox_size=self.min_bbox_size,
        )

        if self.log_split_statistics:
            self._log_split_statistics()

        logger.info(f"Training setup complete: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def _setup_swapped_validation_dataset(self) -> None:

        if not self.swapped_validation_dir:
            logger.warning("Swapped validation enabled but no swapped_validation_dir provided")
            return

        if not os.path.exists(self.swapped_validation_dir):
            logger.warning(f"Swapped validation directory does not exist: {self.swapped_validation_dir}")
            return


        if not (self.subject_a and self.subject_b):
            logger.warning("Swapped validation requires both subject_a and subject_b to be specified")
            return

        logger.info(f"Setting up swapped validation dataset from: {self.swapped_validation_dir}")
        logger.info(f"Looking for subject pair: ({self.subject_a}, {self.subject_b})")


        swapped_image_paths = discover_swapped_validation_data(
            swapped_dir=self.swapped_validation_dir,
            subject_a=self.subject_a,
            subject_b=self.subject_b,
            supported_extensions=self.supported_extensions,
        )

        if not swapped_image_paths:
            logger.warning(f"No swapped validation images found for subjects ({self.subject_a}, {self.subject_b})")
            return


        self.swapped_val_dataset = SwappedValidationDataset(
            image_paths=swapped_image_paths,
            prompts=self.prompts,
            adaptive_resolution_enabled=self.adaptive_resolution_enabled,
            target_resolution=self.target_resolution,
            min_bbox_size=self.min_bbox_size,
            max_validation_cameras=self.max_validation_cameras,
            use_clean_masks=True,
            visualize_restored_images=self.visualize_restored_images,
        )

        logger.info(f"Swapped validation setup complete: {len(self.swapped_val_dataset)} images")

    def _filter_pairs_by_subject_count(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

        subject_counts = {}
        for degraded_path, gt_path in pairs:

            path_parts = Path(degraded_path).parts
            restored_dir = None

            for part in path_parts:
                if part.startswith('restored_'):
                    restored_dir = part
                    break

            if restored_dir:
                subject_id = extract_subject_id_from_restored_dir(restored_dir, self.extract_subject_pattern)
                if subject_id:
                    subject_counts[subject_id] = subject_counts.get(subject_id, 0) + 1

        valid_subjects = {subj for subj, count in subject_counts.items() if count >= self.min_pairs_per_subject}

        filtered_pairs = []
        for degraded_path, gt_path in pairs:
            path_parts = Path(degraded_path).parts
            restored_dir = None

            for part in path_parts:
                if part.startswith('restored_'):
                    restored_dir = part
                    break

            if restored_dir:
                subject_id = extract_subject_id_from_restored_dir(restored_dir, self.extract_subject_pattern)
                if subject_id in valid_subjects:
                    filtered_pairs.append((degraded_path, gt_path))

        logger.info(f"Filtered subjects: {len(valid_subjects)}/{len(subject_counts)} subjects meet minimum count requirement")
        return filtered_pairs

    def _log_split_statistics(self) -> None:

        train_subjects = set()
        val_subjects = set()

        for degraded_path, _ in self.train_pairs:

            path_parts = Path(degraded_path).parts
            restored_dir = None

            for part in path_parts:
                if part.startswith('restored_'):
                    restored_dir = part
                    break

            if restored_dir:
                subject_id = extract_subject_id_from_restored_dir(restored_dir, self.extract_subject_pattern)
                if subject_id:
                    train_subjects.add(subject_id)

        for degraded_path, _ in self.val_pairs:

            path_parts = Path(degraded_path).parts
            restored_dir = None

            for part in path_parts:
                if part.startswith('restored_'):
                    restored_dir = part
                    break

            if restored_dir:
                subject_id = extract_subject_id_from_restored_dir(restored_dir, self.extract_subject_pattern)
                if subject_id:
                    val_subjects.add(subject_id)

        logger.info("=== Training Split Statistics ===")
        logger.info(f"Train: {len(train_subjects)} subjects, {len(self.train_pairs)} pairs")
        logger.info(f"Val: {len(val_subjects)} subjects, {len(self.val_pairs)} pairs")
        logger.info(f"Total: {len(train_subjects) + len(val_subjects)} subjects, {len(self.train_pairs) + len(self.val_pairs)} pairs")

        if self.verbose_mapping:
            logger.debug(f"Train subjects: {sorted(train_subjects)}")
            logger.debug(f"Val subjects: {sorted(val_subjects)}")

    def train_dataloader(self) -> DataLoader:

        if not self.training_mode:
            raise RuntimeError("DataModule not in training mode")
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self._collate_training_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        if not self.training_mode:
            raise RuntimeError("DataModule not in training mode")
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")


        training_val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_training_fn,
            pin_memory=True,
        )


        if self.swapped_validation_enabled and self.swapped_val_dataset is not None:
            swapped_val_loader = DataLoader(
                self.swapped_val_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
                collate_fn=self._collate_swapped_validation_fn,
                pin_memory=True,
            )

            logger.info("Returning dual validation dataloaders: [training_validation, swapped_validation]")
            return [training_val_loader, swapped_val_loader]
        else:
            logger.info("Returning single training validation dataloader")
            return training_val_loader

    def predict_dataloader(self) -> DataLoader:

        if self.training_mode:
            raise RuntimeError("DataModule in training mode, use train_dataloader() instead")
        if self.predict_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self._collate_inference_fn,
        )

    def _collate_inference_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:


        valid_items = [item for item in batch if "error" not in item]
        error_items = [item for item in batch if "error" in item]

        if not valid_items:

            return {
                "errors": error_items,
                "metadata": [item["metadata"] for item in error_items],
            }


        image_tensors = [item["image"] for item in valid_items]
        prompts = [item["prompt"] for item in valid_items]
        metadata = [item["metadata"] for item in valid_items]


        image_batch = torch.stack(image_tensors, dim=0)


        if error_items:
            metadata.append({"errors": error_items})

        return {
            "images": image_batch,
            "prompts": prompts,
            "metadata": metadata,
        }

    def _collate_training_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:


        valid_items = [item for item in batch if "error" not in item]
        error_items = [item for item in batch if "error" in item]

        if not valid_items:

            return {
                "errors": error_items,
                "metadata": [item["metadata"] for item in error_items],
            }


        degraded_tensors = [item["degraded"] for item in valid_items]
        target_tensors = [item["target"] for item in valid_items]
        reference_tensors = [item["reference"] for item in valid_items if item.get("reference") is not None]
        prompts = [item["prompt"] for item in valid_items]
        metadata = [item["metadata"] for item in valid_items]


        degraded_batch = torch.stack(degraded_tensors, dim=0)
        target_batch = torch.stack(target_tensors, dim=0)
        reference_batch = torch.stack(reference_tensors, dim=0) if reference_tensors else None


        if error_items:
            metadata.append({"errors": error_items})

        return {
            "degraded": degraded_batch,
            "target": target_batch,
            "reference": reference_batch,
            "prompts": prompts,
            "metadata": metadata,
        }

    def _collate_swapped_validation_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:


        valid_items = [item for item in batch if "error" not in item]
        error_items = [item for item in batch if "error" in item]

        if not valid_items:

            return {
                "errors": error_items,
                "metadata": [item["metadata"] for item in error_items],
                "is_swapped_validation": True,
            }


        image_tensors = [item["image"] for item in valid_items]
        prompts = [item["prompt"] for item in valid_items]
        metadata = [item["metadata"] for item in valid_items]


        image_batch = torch.stack(image_tensors, dim=0)


        if error_items:
            metadata.append({"errors": error_items})

        return {
            "images": image_batch,
            "prompts": prompts,
            "metadata": metadata,
            "is_swapped_validation": True,
        }


def parse_swapped_directory_name(dir_name: str) -> Optional[Tuple[str, str]]:


    pattern = r"swapped_(\d+)head_on_(\d+)body"
    match = re.search(pattern, dir_name)

    if match:
        head_id = match.group(1)
        body_id = match.group(2)
        return (head_id, body_id)

    logger.warning(f"Could not parse swapped directory name: {dir_name}")
    return None


def discover_swapped_validation_data(
    swapped_dir: str,
    subject_a: str,
    subject_b: str,
    supported_extensions: Optional[List[str]] = None
) -> List[str]:

    if supported_extensions is None:
        supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    swapped_paths = []

    if not os.path.exists(swapped_dir):
        logger.warning(f"Swapped directory does not exist: {swapped_dir}")
        return swapped_paths


    swap_patterns = [
        f"swapped_{subject_a}head_on_{subject_b}body*",
        f"swapped_{subject_b}head_on_{subject_a}body*"
    ]

    for pattern in swap_patterns:
        pattern_dirs = glob.glob(os.path.join(swapped_dir, pattern))

        for swap_dir in pattern_dirs:
            if not os.path.isdir(swap_dir):
                continue

            logger.info(f"Found swapped directory: {os.path.basename(swap_dir)}")


            for camera_dir in os.listdir(swap_dir):
                camera_path = os.path.join(swap_dir, camera_dir)

                if not os.path.isdir(camera_path):
                    continue


                if 'mask' in camera_dir.lower():
                    continue


                for filename in os.listdir(camera_path):
                    file_path = os.path.join(camera_path, filename)

                    if os.path.isfile(file_path):

                        _, ext = os.path.splitext(filename.lower())
                        if ext in supported_extensions:
                            swapped_paths.append(file_path)

    logger.info(f"Found {len(swapped_paths)} swapped validation images for subjects ({subject_a}, {subject_b})")
    return swapped_paths


def get_clean_mask_directory(data_dir: str) -> str:

    if "_refined" in data_dir:
        clean_dir = data_dir.replace("_refined", "")
        logger.debug(f"Detected refined directory, using clean masks from: {clean_dir}")
        return clean_dir
    return data_dir


def construct_swapped_mask_path(image_path: str, clean_mask_dir: Optional[str] = None) -> str:

    if clean_mask_dir is not None:

        path_obj = Path(image_path)

        parts = path_obj.parts


        swap_start_idx = None
        for i, part in enumerate(parts):
            if part.startswith('swapped_'):
                swap_start_idx = i
                break

        if swap_start_idx is not None:

            relative_parts = parts[swap_start_idx:]
            clean_path = os.path.join(clean_mask_dir, *relative_parts)
            return construct_mask_path(clean_path)
        else:
            logger.warning(f"Could not find swapped directory in path: {image_path}")

            return construct_mask_path(image_path)
    else:

        return construct_mask_path(image_path)


def reverse_adaptive_resolution_metadata(
    bbox: Tuple[int, int, int, int],
    original_size: Tuple[int, int],
    target_resolution: Tuple[int, int]
) -> Dict[str, Any]:

    crop_width = bbox[2] - bbox[0]
    crop_height = bbox[3] - bbox[1]


    crop_aspect = crop_width / crop_height
    target_aspect = target_resolution[0] / target_resolution[1]


    if crop_aspect > target_aspect:

        padded_width = crop_width
        padded_height = int(crop_width / target_aspect)
        pad_x = 0
        pad_y = (padded_height - crop_height) // 2
    else:

        padded_height = crop_height
        padded_width = int(crop_height * target_aspect)
        pad_x = (padded_width - crop_width) // 2
        pad_y = 0

    return {
        "original_size": original_size,
        "bbox": bbox,
        "crop_size": (crop_width, crop_height),
        "padded_size": (padded_width, padded_height),
        "padding": (pad_x, pad_y),
        "target_resolution": target_resolution
    }


def apply_reverse_adaptive_resolution_transform(
    processed_image: Image.Image,
    metadata: Dict[str, Any]
) -> Image.Image:


    original_size = metadata["original_size"]
    bbox = metadata["bbox"]
    crop_size = metadata["crop_size"]
    padded_size = metadata["padded_size"]
    padding = metadata["padding"]


    processed_resized = processed_image.resize(padded_size, Image.LANCZOS)


    if isinstance(padding, torch.Tensor):
        padding = padding.tolist()
    if isinstance(crop_size, torch.Tensor):
        crop_size = crop_size.tolist()
    left = padding[0]
    top = padding[1]
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    processed_cropped = processed_resized.crop((left.item(), top.item(), right.item(), bottom.item()))


    restored_image = Image.new('RGB', original_size, (0, 0, 0))
    restored_image.paste(processed_cropped, (bbox[0], bbox[1]))

    return restored_image


class SwappedValidationDataset(Dataset):


    def __init__(
        self,
        image_paths: List[str],
        prompts: Optional[Union[str, List[str]]] = None,
        adaptive_resolution_enabled: bool = False,
        target_resolution: Tuple[int, int] = (448, 896),
        min_bbox_size: Tuple[int, int] = (64, 64),
        max_validation_cameras: Optional[int] = None,
        use_clean_masks: bool = True,
        visualize_restored_images: bool = False,
    ):

        self.image_paths = image_paths
        self.adaptive_resolution_enabled = adaptive_resolution_enabled
        self.target_resolution = target_resolution
        self.min_bbox_size = min_bbox_size
        self.max_validation_cameras = max_validation_cameras
        self.use_clean_masks = use_clean_masks
        self.visualize_restored_images = visualize_restored_images


        self.clean_mask_dir = None
        if use_clean_masks and image_paths:

            first_image_path = image_paths[0]
            base_dir = str(Path(first_image_path).parent.parent.parent)
            self.clean_mask_dir = get_clean_mask_directory(base_dir)

            if self.clean_mask_dir != base_dir:
                logger.info(f"Using clean masks from: {self.clean_mask_dir}")
            else:
                logger.debug("Using masks from same directory (no refinement detected)")
                self.clean_mask_dir = None


        if prompts is None:
            self.prompts = ["remove degradation"] * len(image_paths)
        elif isinstance(prompts, str):
            self.prompts = [prompts] * len(image_paths)
        elif isinstance(prompts, list):
            if len(prompts) != len(image_paths):
                raise ValueError(f"Prompts length ({len(prompts)}) must match image_paths length ({len(image_paths)})")
            self.prompts = prompts
        else:
            raise ValueError("Prompts must be None, str, or list of str")


        if max_validation_cameras and len(image_paths) > max_validation_cameras:

            sampled_indices = self._sample_camera_indices(max_validation_cameras)
            self.image_paths = [self.image_paths[i] for i in sampled_indices]
            self.prompts = [self.prompts[i] for i in sampled_indices]

        logger.info(f"SwappedValidationDataset initialized with {len(self.image_paths)} images")

    def _sample_camera_indices(self, max_cameras: int) -> List[int]:


        a_to_b_indices = []
        b_to_a_indices = []

        for i, path in enumerate(self.image_paths):

            path_parts = Path(path).parts
            swap_dir = None
            for part in path_parts:
                if part.startswith('swapped_'):
                    swap_dir = part
                    break

            if swap_dir:
                parsed = parse_swapped_directory_name(swap_dir)
                if parsed:
                    head_id, body_id = parsed

                    if head_id < body_id:
                        a_to_b_indices.append(i)
                    else:
                        b_to_a_indices.append(i)


        cameras_per_direction = max_cameras // 2
        remaining = max_cameras % 2

        sampled_indices = []


        if a_to_b_indices:
            sampled_a_to_b = random.sample(
                a_to_b_indices,
                min(cameras_per_direction + remaining, len(a_to_b_indices))
            )
            sampled_indices.extend(sampled_a_to_b)


        if b_to_a_indices:
            remaining_slots = max_cameras - len(sampled_indices)
            sampled_b_to_a = random.sample(
                b_to_a_indices,
                min(remaining_slots, len(b_to_a_indices))
            )
            sampled_indices.extend(sampled_b_to_a)


        if len(sampled_indices) < max_cameras:
            all_indices = set(range(len(self.image_paths)))
            remaining_indices = list(all_indices - set(sampled_indices))
            if remaining_indices:
                additional_needed = max_cameras - len(sampled_indices)
                additional_samples = random.sample(
                    remaining_indices,
                    min(additional_needed, len(remaining_indices))
                )
                sampled_indices.extend(additional_samples)

        return sorted(sampled_indices)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        image_path = self.image_paths[idx]
        prompt = self.prompts[idx]


        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        try:

            image_pil = load_image(image_path).convert('RGB')
            original_size = image_pil.size


            reverse_metadata = None


            if self.adaptive_resolution_enabled:

                mask_path = construct_swapped_mask_path(image_path, clean_mask_dir=self.clean_mask_dir)

                if os.path.exists(mask_path):

                    bbox = calculate_bounding_box_from_mask(mask_path, self.min_bbox_size)
                elif self.clean_mask_dir:

                    fallback_mask_path = construct_swapped_mask_path(image_path)
                    if os.path.exists(fallback_mask_path):
                        logger.warning(f"Clean mask not found, using fallback: {fallback_mask_path}")
                        mask_path = fallback_mask_path
                        bbox = calculate_bounding_box_from_mask(mask_path, self.min_bbox_size)
                    else:
                        logger.warning(f"Neither clean nor fallback mask found for: {image_path}")
                        bbox = None
                else:
                    bbox = None

                if bbox is not None:


                    reverse_metadata = reverse_adaptive_resolution_metadata(
                        bbox=bbox,
                        original_size=original_size,
                        target_resolution=self.target_resolution
                    )


                    image_pil = apply_adaptive_resolution_transform(
                        image_pil,
                        target_resolution=self.target_resolution,
                        min_bbox_size=self.min_bbox_size,
                        bbox=bbox
                    )
                else:
                    logger.warning(f"Mask not found for swapped validation image: {mask_path}")


            image_tensor = transform(image_pil)


            if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
                logger.warning(f"NaN/inf detected in swapped validation tensor from {image_path}, cleaning...")
                image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=1.0, neginf=0.0)


            path_parts = Path(image_path).parts
            swap_dir = None
            camera_dir = None

            for part in path_parts:
                if part.startswith('swapped_'):
                    swap_dir = part
                elif re.match(r'\d{3}_p[+-]\d{2}', part):
                    camera_dir = part

            swap_info = None
            if swap_dir:
                parsed = parse_swapped_directory_name(swap_dir)
                if parsed:
                    head_id, body_id = parsed
                    swap_info = {
                        "head_subject": head_id,
                        "body_subject": body_id,
                        "swap_direction": f"{head_id}→{body_id}"
                    }

            return {
                "image": image_tensor,
                "prompt": prompt,
                "metadata": {
                    "input_path": image_path,
                    "index": idx,
                    "swap_info": swap_info,
                    "camera_dir": camera_dir,
                    "reverse_metadata": reverse_metadata,
                    "is_swapped_validation": True,
                    "visualize_restored_images": self.visualize_restored_images,
                }
            }

        except Exception as e:

            logger.error(f"Failed to load swapped validation image at index {idx}: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "input_path": image_path,
                    "index": idx,
                    "is_swapped_validation": True,
                    "visualize_restored_images": self.visualize_restored_images,
                }
            }
