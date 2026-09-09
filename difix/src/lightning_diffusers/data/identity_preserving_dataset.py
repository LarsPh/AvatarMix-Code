import os
import re
import random
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any

import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from loguru import logger

from ..utils.camera_selection import CameraSelector
from ..utils.head_processing import load_and_crop_head, validate_head_mask_data
from .difix3d_datamodule import construct_gt_mask_path, calculate_bounding_box_from_mask, apply_adaptive_resolution_transform, apply_adaptive_resolution_transform_to_mask


class IdentityPreservingDataset(Dataset):


    def __init__(
        self,

        degraded_data_root: str,
        gt_data_root: str,
        first_swapped_data_root: str,


        camera_selector: CameraSelector,


        prompts: Optional[Union[str, List[str]]] = None,
        head_crop_resolution: int = 224,
        adaptive_resolution_enabled: bool = False,
        target_resolution: Tuple[int, int] = (448, 896),
        min_bbox_size: Tuple[int, int] = (64, 64),


        subject_a: Optional[str] = None,
        subject_b: Optional[str] = None,


        validate_data_existence: bool = True,
        min_mask_area: int = 1000,


        use_no_reshape_data: bool = False,


        alignment_mode: str = "standard",
    ):

        super().__init__()

        self.degraded_data_root = Path(degraded_data_root)
        self.gt_data_root = Path(gt_data_root)
        self.camera_selector = camera_selector
        self.head_crop_resolution = head_crop_resolution
        self.adaptive_resolution_enabled = adaptive_resolution_enabled
        self.target_resolution = target_resolution
        self.min_bbox_size = min_bbox_size
        self.min_mask_area = min_mask_area
        self.first_swapped_data_root = first_swapped_data_root


        self.use_no_reshape_data = use_no_reshape_data


        self.alignment_mode = alignment_mode


        self.subject_a = subject_a
        self.subject_b = subject_b


        if isinstance(prompts, str):
            self.prompts = [prompts]
        elif prompts is None:
            self.prompts = ["remove degradation"]
        else:
            self.prompts = list(prompts)


        if self.alignment_mode not in ["standard", "gt_aligned"]:
            raise ValueError(f"Invalid alignment_mode: {self.alignment_mode}. Must be 'standard' or 'gt_aligned'")


        logger.info(f"IdentityPreservingDataset constructor called:")
        logger.info(f"  Degraded data root: {self.degraded_data_root}")
        logger.info(f"  GT data root: {self.gt_data_root}")
        logger.info(f"  Subject A: {subject_a}, Subject B: {subject_b}")
        logger.info(f"  Head crop resolution: {head_crop_resolution}")
        logger.info(f"  Adaptive resolution: {adaptive_resolution_enabled}")
        logger.info(f"  No reshape data: {use_no_reshape_data}")
        logger.info(f"  Alignment mode: {self.alignment_mode}")

        self.camera_pairs = self._generate_valid_camera_pairs(validate_data_existence)

        if not self.camera_pairs:
            logger.warning(f"No camera pairs found - dataset will be empty")

        logger.info(f"IdentityPreservingDataset initialized with {len(self.camera_pairs)} camera pairs")


        self.transform = self._setup_transforms()


        self.current_mode = "identity"

    def set_mode(self, mode: str) -> None:

        if mode not in ["normal", "identity"]:
            raise ValueError(f"Invalid mode: {mode}. Use 'normal' or 'identity'")

        self.current_mode = mode
        logger.debug(f"Dataset mode set to: {mode}")

    def _discover_all_subject_pairs(self) -> List[Tuple[str, str]]:

        refined_renders_dir = self.degraded_data_root.parent / "head_swapped_renders_refined"
        subject_pairs = []

        logger.info(f"Discovering subject pairs from both data sources:")
        logger.info(f"  First-swapped data: {refined_renders_dir}")
        logger.info(f"  Twice-swapped data: {self.degraded_data_root}")

        if not refined_renders_dir.exists():
            logger.warning(f"Head swapped renders directory not found: {refined_renders_dir}")
            return subject_pairs


        candidate_pairs = []
        for subdir in refined_renders_dir.iterdir():
            if not subdir.is_dir():
                continue


            match = re.search(r'swapped_(\d+)head_on_(\d+)body', subdir.name)


            if self.use_no_reshape_data:

                if "reshaped" in subdir.name:
                    logger.debug(f"Skipping reshaped directory (no-reshape mode): {subdir.name}")
                    continue
            else:

                if "reshaped" not in subdir.name:
                    logger.debug(f"Skipping non-reshaped directory (reshape mode): {subdir.name}")
                    continue

            if match:
                head_subject = match.group(1)
                body_subject = match.group(2)
                candidate_pairs.append((head_subject, subdir.name))


        reshape_mode_desc = "no-reshape" if self.use_no_reshape_data else "reshape"
        expected_pattern_desc = "without 'reshaped'" if self.use_no_reshape_data else "with 'reshaped'"

        logger.info(f"Found {len(candidate_pairs)} first-swapped candidates in {reshape_mode_desc} mode ({expected_pattern_desc})")


        if not candidate_pairs:
            available_dirs = [subdir.name for subdir in refined_renders_dir.iterdir() if subdir.is_dir() and re.search(r'swapped_(\d+)head_on_(\d+)body', subdir.name)]
            if self.use_no_reshape_data:
                opposite_available = [d for d in available_dirs if "reshaped" not in d]
                error_suggestion = f"Found {len([d for d in available_dirs if 'reshaped' in d])} reshaped directories. Set use_no_reshape_data=False to use them."
            else:
                opposite_available = [d for d in available_dirs if "reshaped" in d]
                error_suggestion = f"Found {len([d for d in available_dirs if 'reshaped' not in d])} non-reshaped directories. Set use_no_reshape_data=True to use them."

            raise FileNotFoundError(
                f"No first-swapped directories found matching {reshape_mode_desc} criteria ({expected_pattern_desc}). "
                f"Available swapped directories: {len(available_dirs)}. {error_suggestion}"
            )


        for head_subject, swapped_dir_name in candidate_pairs:

            twice_swapped_pattern = f"restored_{head_subject}_from_refined_*"
            twice_swapped_matches = list(self.degraded_data_root.glob(twice_swapped_pattern))

            if twice_swapped_matches:

                valid_restored_matches = []
                for restored_match in twice_swapped_matches:
                    if self.use_no_reshape_data:

                        if "reshaped" not in restored_match.name:
                            valid_restored_matches.append(restored_match)
                        else:
                            logger.debug(f"Skipping reshaped restored directory (no-reshape mode): {restored_match.name}")
                    else:

                        if "reshaped" in restored_match.name:
                            valid_restored_matches.append(restored_match)
                        else:
                            logger.debug(f"Skipping non-reshaped restored directory (reshape mode): {restored_match.name}")

                if valid_restored_matches:
                    subject_pairs.append((head_subject, swapped_dir_name))
                    logger.debug(f"Validated pair: GT={head_subject} has both first and twice-swapped data matching {reshape_mode_desc} criteria")
                else:
                    available_restored = [m.name for m in twice_swapped_matches]
                    logger.debug(f"Skipping GT={head_subject}: restored directories found but none match {reshape_mode_desc} criteria. Available: {available_restored}")
            else:
                logger.debug(f"Skipping GT={head_subject}: missing twice-swapped data (pattern: {twice_swapped_pattern})")

        logger.info(f"Validated {len(subject_pairs)} complete subject pairs:")
        for gt_subject, swapped_dir in subject_pairs:
            logger.info(f"  GT subject {gt_subject} with swapped directory: {swapped_dir[:50]}...")

        return subject_pairs

    def _parse_subject_ids(self, gt_subject_id: str, swapped_subject_id: str) -> Tuple[str, str]:


        source_subject = gt_subject_id


        match = re.search(r'swapped_(\d+)head_on_(\d+)body', swapped_subject_id)
        if match:
            parsed_source = match.group(1)
            target_subject = match.group(2)


            if parsed_source != source_subject:
                logger.warning(f"Source subject mismatch: gt_subject_id={source_subject}, parsed from swapped={parsed_source}")

            logger.debug(f"Parsed subject pair: {source_subject} → {target_subject}")
            return source_subject, target_subject
        else:
            raise ValueError(
                f"Cannot parse swapped subject ID: {swapped_subject_id}. "
                f"Expected format: 'swapped_XXXXhead_on_YYYYbody_...'"
            )

    def _get_twice_swapped_data_root(self, gt_subject: str) -> Path:


        pattern = f"restored_{gt_subject}_from_refined_*"
        matches = list(self.degraded_data_root.glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No twice-swapped data found for subject {gt_subject} (pattern: {pattern})")

        if len(matches) > 1:
            logger.warning(f"Multiple twice-swapped directories found for {gt_subject}, using first: {matches[0].name}")

        return matches[0]

    def _build_swapped_image_path(self, swapped_data_root: Path, camera: str) -> Path:

        if self.alignment_mode == "gt_aligned":
            return swapped_data_root / camera / "head_aligned" / "0000.png"
        else:
            return swapped_data_root / camera / "0000.jpg"

    def _build_head_mask_path(self, data_root: Path, subject: str, camera: str, for_swapped: bool = False) -> Path:

        if self.alignment_mode == "gt_aligned":

            return self.gt_data_root / subject / camera / "mask" / "head" / "0000.png"
        else:

            if for_swapped:
                return data_root / camera / "mask" / "head" / "0000.png"
            else:
                return self.gt_data_root / subject / camera / "mask" / "head" / "0000.png"

    def _build_fullbody_mask_path(self, swapped_data_root: Path, camera: str) -> Path:

        if self.alignment_mode == "gt_aligned":
            return swapped_data_root / camera / "head_aligned" / "mask" / "pha" / "0000.png"
        else:

            non_refined_root = Path(str(swapped_data_root).replace("_refined", ""))
            return non_refined_root / camera / "mask" / "pha" / "0000.png"

    def _calculate_union_bbox(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:

        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        union_x_min = min(x1_min, x2_min)
        union_y_min = min(y1_min, y2_min)
        union_x_max = max(x1_max, x2_max)
        union_y_max = max(y1_max, y2_max)

        return (union_x_min, union_y_min, union_x_max, union_y_max)

    def _calculate_adaptive_resolution_bboxes(self, gt_fullbody_mask_path: Path, degraded_fullbody_mask_path: Path) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:

        if not self.adaptive_resolution_enabled:
            return None, None

        if self.current_mode == "normal":

            try:
                gt_adaptive_bbox = calculate_bounding_box_from_mask(str(gt_fullbody_mask_path), self.min_bbox_size)
                degraded_adaptive_bbox = gt_adaptive_bbox
                logger.debug(f"Normal mode: using GT bbox for both images")
                return gt_adaptive_bbox, degraded_adaptive_bbox
            except Exception as e:
                logger.warning(f"GT bbox calculation failed in normal mode: {e}")
                return None, None

        else:
            if self.alignment_mode == "gt_aligned":

                gt_bbox = None
                swapped_bbox = None


                try:
                    gt_bbox = calculate_bounding_box_from_mask(str(gt_fullbody_mask_path), self.min_bbox_size)
                    logger.debug(f"GT-aligned mode: calculated GT bbox successfully")
                except Exception as e:
                    logger.warning(f"GT bbox calculation failed in GT-aligned mode: {e}")

                try:
                    swapped_bbox = calculate_bounding_box_from_mask(str(degraded_fullbody_mask_path), self.min_bbox_size)
                    logger.debug(f"GT-aligned mode: calculated swapped bbox successfully")
                except Exception as e:
                    logger.warning(f"Swapped bbox calculation failed in GT-aligned mode: {e}")


                if gt_bbox is not None and swapped_bbox is not None:

                    union_bbox = self._calculate_union_bbox(gt_bbox, swapped_bbox)
                    logger.debug(f"GT-aligned mode: using union bbox {union_bbox} for both images")
                    return union_bbox, union_bbox
                elif gt_bbox is not None:

                    logger.debug(f"GT-aligned mode: using GT bbox {gt_bbox} for both images (swapped bbox failed)")
                    return gt_bbox, gt_bbox
                elif swapped_bbox is not None:

                    logger.debug(f"GT-aligned mode: using swapped bbox {swapped_bbox} for both images (GT bbox failed)")
                    return swapped_bbox, swapped_bbox
                else:

                    logger.warning("GT-aligned mode: both bbox calculations failed, disabling adaptive resolution")
                    return None, None
            else:

                gt_adaptive_bbox = None
                degraded_adaptive_bbox = None

                try:
                    gt_adaptive_bbox = calculate_bounding_box_from_mask(str(gt_fullbody_mask_path), self.min_bbox_size)
                    logger.debug(f"Standard identity mode: calculated GT bbox successfully")
                except Exception as e:
                    logger.warning(f"GT bbox calculation failed in standard identity mode: {e}")

                try:
                    degraded_adaptive_bbox = calculate_bounding_box_from_mask(str(degraded_fullbody_mask_path), self.min_bbox_size)
                    logger.debug(f"Standard identity mode: calculated degraded bbox successfully")
                except Exception as e:
                    logger.warning(f"Degraded bbox calculation failed in standard identity mode: {e}")

                return gt_adaptive_bbox, degraded_adaptive_bbox

    def _validate_gt_aligned_data_exists(self, swapped_data_root: Path, camera: str, gt_subject: str) -> None:

        if self.alignment_mode == "gt_aligned":

            aligned_image_path = self._build_swapped_image_path(swapped_data_root, camera)
            if not aligned_image_path.exists():
                raise FileNotFoundError(f"GT-aligned swapped image missing: {aligned_image_path}")


            aligned_fullbody_mask_path = self._build_fullbody_mask_path(swapped_data_root, camera)
            if not aligned_fullbody_mask_path.exists():
                raise FileNotFoundError(f"GT-aligned swapped fullbody mask missing: {aligned_fullbody_mask_path}")


            gt_mask_path = self.gt_data_root / gt_subject / camera / "mask" / "head" / "0000.png"
            if not gt_mask_path.exists():
                raise FileNotFoundError(f"GT head mask missing: {gt_mask_path}")

    def _get_first_swapped_data_root(self, swapped_subject_id: str) -> Path:


        refined_renders_dir = Path(self.first_swapped_data_root)

        if refined_renders_dir.exists():
            for subdir in refined_renders_dir.iterdir():
                if subdir.is_dir() and subdir.name.startswith(swapped_subject_id):

                    return subdir

        raise ValueError(
            f"Could not find first-swapped directory for {swapped_subject_id} "
            f"in {refined_renders_dir}"
        )

    def _get_subject_type(self, gt_subject: str) -> str:

        if not (self.subject_a and self.subject_b):
            raise ValueError("subject_a and subject_b must be provided from config for subject type determination")

        if gt_subject == self.subject_a:
            return 'a'
        elif gt_subject == self.subject_b:
            return 'b'
        else:
            logger.debug(f"GT subject {gt_subject} not found in config subjects: A={self.subject_a}, B={self.subject_b}")
            return None


    def _generate_valid_camera_pairs(self, validate_existence: bool) -> List[Tuple[str, str, str, str]]:


        subject_pairs = self._discover_all_subject_pairs()

        if not subject_pairs:
            logger.warning("No subject pairs discovered - no camera pairs generated")
            return []

        logger.info(f"Generating camera pairs for {len(subject_pairs)} subject pairs...")

        all_camera_pairs = []


        for gt_subject, swapped_directory in subject_pairs:
            try:

                source_subject, target_subject = self._parse_subject_ids(gt_subject, swapped_directory)


                subject_type = self._get_subject_type(gt_subject)
                if subject_type is None:
                    continue

                logger.info(f"Processing GT subject {gt_subject} → type {subject_type} (swapped: {source_subject} → {target_subject})")


                swapped_data_root = self._get_first_swapped_data_root(swapped_directory)


                camera_pairs = self.camera_selector.generate_camera_pairs(
                    gt_data_root=self.gt_data_root,
                    swapped_data_root=swapped_data_root,
                    source_subject_id=source_subject,
                    target_subject_id=target_subject,
                    subject_type=subject_type
                )


                for gt_camera, swapped_camera in camera_pairs:
                    enhanced_pair = (gt_camera, swapped_camera, gt_subject, swapped_directory)

                    if validate_existence:
                        if self._validate_camera_pair_data(gt_camera, swapped_camera, gt_subject, swapped_directory):
                            all_camera_pairs.append(enhanced_pair)
                    else:
                        all_camera_pairs.append(enhanced_pair)

                logger.info(f"  Generated {len([p for p in all_camera_pairs if p[2] == gt_subject])} camera pairs for subject {gt_subject}")

            except Exception as e:
                logger.error(f"Error processing subject pair {gt_subject}: {e}")
                continue

        logger.info(f"Total enhanced camera pairs generated: {len(all_camera_pairs)}")
        return all_camera_pairs

    def _validate_camera_pair_data(self, gt_camera: str, swapped_camera: str, gt_subject: str, swapped_directory: str) -> bool:

        try:

            gt_image_path = self.gt_data_root / gt_subject / gt_camera / "0000.jpg"
            gt_mask_path = self._build_head_mask_path(self.gt_data_root, gt_subject, gt_camera, for_swapped=False)
            gt_fullbody_mask_path = self.gt_data_root / gt_subject / gt_camera / "mask" / "pha" / "0000.png"


            first_swapped_data_root = self._get_first_swapped_data_root(swapped_directory)


            first_swapped_image_path = self._build_swapped_image_path(first_swapped_data_root, swapped_camera)
            first_swapped_mask_path = self._build_head_mask_path(first_swapped_data_root, gt_subject, swapped_camera, for_swapped=True)
            first_swapped_fullbody_mask_path = self._build_fullbody_mask_path(first_swapped_data_root, swapped_camera)


            if self.alignment_mode == "gt_aligned":
                self._validate_gt_aligned_data_exists(first_swapped_data_root, swapped_camera, gt_subject)


            twice_swapped_data_root = self._get_twice_swapped_data_root(gt_subject)
            twice_swapped_image_path = twice_swapped_data_root / gt_camera / "0000.jpg"
            twice_swapped_fullbody_mask_path = twice_swapped_data_root / gt_camera / "mask" / "pha" / "0000.png"


            required_files = [
                gt_image_path, gt_mask_path,
                first_swapped_image_path, first_swapped_mask_path,
                twice_swapped_image_path
            ]

            required_files_fullbody_mask = [
                twice_swapped_fullbody_mask_path,
                first_swapped_fullbody_mask_path,
                gt_fullbody_mask_path
            ]

            for file_path in required_files:
                if not file_path.exists():
                    logger.debug(f"Missing file: {file_path}")
                    return False

            for file_path in required_files_fullbody_mask:
                if not file_path.exists():
                    file_path = file_path.with_suffix(".jpg")
                    if not file_path.exists():
                        logger.debug(f"Missing file: {file_path}")
                        return False


            gt_mask_valid = validate_head_mask_data(
                str(gt_image_path), str(gt_mask_path), self.min_mask_area
            )
            first_swapped_mask_valid = validate_head_mask_data(
                str(first_swapped_image_path), str(first_swapped_mask_path), self.min_mask_area
            )

            if not (gt_mask_valid and first_swapped_mask_valid):
                logger.debug(f"Invalid head mask data for pair {gt_camera} <-> {swapped_camera}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Validation error for pair {gt_camera} <-> {swapped_camera}: {e}")
            return False

    def _setup_transforms(self) -> transforms.Compose:

        transform_list = [
            transforms.ToTensor(),
        ]
        return transforms.Compose(transform_list)

    def __len__(self) -> int:

        return len(self.camera_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:


        gt_camera, swapped_camera, gt_subject, swapped_directory = self.camera_pairs[idx]

        try:

            gt_image_path = self.gt_data_root / gt_subject / gt_camera / "0000.jpg"
            gt_head_mask_path = self._build_head_mask_path(self.gt_data_root, gt_subject, gt_camera, for_swapped=False)

            gt_fullbody_mask_path = self.gt_data_root / gt_subject / gt_camera / "mask" / "pha" / "0000.png"


            first_swapped_data_root = self._get_first_swapped_data_root(swapped_directory)


            self._validate_gt_aligned_data_exists(first_swapped_data_root, swapped_camera, gt_subject)

            first_swapped_image_path = self._build_swapped_image_path(first_swapped_data_root, swapped_camera)
            first_swapped_head_mask_path = self._build_head_mask_path(first_swapped_data_root, gt_subject, swapped_camera, for_swapped=True)

            first_swapped_fullbody_mask_path = self._build_fullbody_mask_path(first_swapped_data_root, swapped_camera)


            twice_swapped_data_root = self._get_twice_swapped_data_root(gt_subject)
            twice_swapped_image_path = twice_swapped_data_root / gt_camera / "0000.jpg"

            twice_swapped_fullbody_mask_path = Path(str(twice_swapped_data_root).replace("_refined", "")) / gt_camera / "mask" / "pha" / "0000.png"


            gt_image = Image.open(gt_image_path).convert("RGB")


            if self.current_mode == "normal":

                degraded_image = Image.open(twice_swapped_image_path).convert("RGB")
                degraded_fullbody_mask_path = twice_swapped_fullbody_mask_path
                logger.debug(f"Mode '{self.current_mode}': loaded twice-swapped image")
            else:

                degraded_image = Image.open(first_swapped_image_path).convert("RGB")
                degraded_fullbody_mask_path = first_swapped_fullbody_mask_path
                logger.debug(f"Mode '{self.current_mode}': loaded first-swapped image")


            gt_adaptive_bbox = None
            degraded_adaptive_bbox = None

            if self.adaptive_resolution_enabled:

                gt_adaptive_bbox, degraded_adaptive_bbox = self._calculate_adaptive_resolution_bboxes(
                    gt_fullbody_mask_path, degraded_fullbody_mask_path
                )


                try:
                    if gt_adaptive_bbox is not None:
                        gt_image = apply_adaptive_resolution_transform(
                            gt_image, target_resolution=self.target_resolution,
                            min_bbox_size=self.min_bbox_size, bbox=gt_adaptive_bbox
                        )
                        logger.debug(f"Applied GT adaptive resolution transform successfully")
                    else:
                        logger.debug(f"Skipping GT adaptive resolution transform (bbox unavailable)")
                except Exception as e:
                    logger.warning(f"GT adaptive resolution transform failed: {e}")
                    gt_adaptive_bbox = None

                try:
                    if degraded_adaptive_bbox is not None:
                        degraded_image = apply_adaptive_resolution_transform(
                            degraded_image, target_resolution=self.target_resolution,
                            min_bbox_size=self.min_bbox_size, bbox=degraded_adaptive_bbox
                        )
                        logger.debug(f"Applied degraded adaptive resolution transform successfully")
                    else:
                        logger.debug(f"Skipping degraded adaptive resolution transform (bbox unavailable)")
                except Exception as e:
                    logger.warning(f"Degraded adaptive resolution transform failed: {e}")
                    degraded_adaptive_bbox = None


            gt_tensor = self.transform(gt_image)
            degraded_tensor = self.transform(degraded_image)


            gt_head_crop, gt_head_success = load_and_crop_head(
                str(gt_image_path), str(gt_head_mask_path), self.head_crop_resolution
            )


            if self.current_mode == "identity":

                head_mask_pil = Image.open(first_swapped_head_mask_path).convert('L')
                head_mask_array = np.array(head_mask_pil)


                if self.adaptive_resolution_enabled and degraded_adaptive_bbox is not None:
                    try:
                        head_mask_array = apply_adaptive_resolution_transform_to_mask(
                            head_mask_array, degraded_adaptive_bbox, self.target_resolution
                        )
                        logger.debug(f"Applied adaptive resolution to head mask using degraded bbox: {head_mask_array.shape}")
                    except Exception as e:
                        logger.warning(f"Failed to apply adaptive resolution to head mask: {e}")
                        logger.warning("Using original resolution head mask")

                head_mask_tensor = torch.from_numpy(head_mask_array).float()
                head_mask_success = True
            else:


                if self.adaptive_resolution_enabled:
                    mask_height, mask_width = self.target_resolution[1], self.target_resolution[0]
                else:
                    mask_height, mask_width = 512, 512
                head_mask_tensor = torch.zeros(mask_height, mask_width)
                head_mask_success = False


            prompt = random.choice(self.prompts)


            metadata = {
                "gt_camera": gt_camera,
                "swapped_camera": swapped_camera,
                "gt_subject_id": gt_subject,
                "swapped_subject_id": swapped_directory,
                "gt_image_path": str(gt_image_path),
                "degraded_image_path": str(first_swapped_image_path if self.current_mode == "identity" else twice_swapped_image_path),
                "current_mode": self.current_mode,
                "gt_head_success": gt_head_success,
                "head_mask_success": head_mask_success,
                "index": idx
            }

            return {
                "degraded": degraded_tensor,
                "target": gt_tensor,
                "gt_head_crop": gt_head_crop,
                "head_mask": head_mask_tensor,
                "prompt": prompt,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx} (cameras: {gt_camera}, {swapped_camera}): {e}")


            if self.adaptive_resolution_enabled:
                img_height, img_width = self.target_resolution[1], self.target_resolution[0]
                dummy_image = torch.zeros(3, img_height, img_width)
                dummy_mask = torch.zeros(img_height, img_width)
            else:
                dummy_image = torch.zeros(3, 512, 512)
                dummy_mask = torch.zeros(512, 512)

            dummy_head = torch.zeros(3, self.head_crop_resolution, self.head_crop_resolution)

            return {
                "degraded": dummy_image,
                "target": dummy_image,
                "gt_head_crop": dummy_head,
                "head_mask": dummy_mask,
                "prompt": self.prompts[0],
                "metadata": {
                    "gt_camera": gt_camera,
                    "swapped_camera": swapped_camera,
                    "error": str(e),
                    "index": idx
                }
            }
