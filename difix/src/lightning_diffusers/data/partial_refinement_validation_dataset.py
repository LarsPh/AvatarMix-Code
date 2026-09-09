import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import cv2
import yaml
import numpy as np
from loguru import logger
try:

    from omegaconf import ListConfig  # type: ignore
except Exception:  # pragma: no cover
    ListConfig = ()  # type: ignore[misc,assignment]

from ..utils.color_matching import dilate_mask, match_color
from .difix3d_datamodule import (
    calculate_bounding_box_from_mask,
    apply_adaptive_resolution_transform,
    reverse_adaptive_resolution_metadata,
)


class PartialRefinementValidationDataset(Dataset):


    def __init__(
        self,
        data_root: str,

        subject_pairs: Optional[List[Tuple[str, str]]] = None,
        subject_a: Optional[str] = None,
        subject_b: Optional[str] = None,
        neck_mode: str = "union",

        refinement_mode: str = "partial",

        is_test: bool = False,

        validation_source: str = "combined_portraits",
        first_swap_dir_filter: str = "from_point_cloud",

        use_head_alignment: bool = True,

        test_output_root: Optional[str] = None,

        neck_color: tuple[int, int, int] = (85, 51, 0),
        hair_color: tuple[int, int, int] = (255, 0, 0),
        color_tolerance: int = 5,
        dilate_neck: bool = True,
        neck_dilation_kernel_size: int = 7,
        dilate_hair: bool = False,
        hair_dilation_kernel_size: int = 3,

        target_resolution: int = 512,

        fullbody_target_resolution: Optional[Union[Tuple[int, int], List[int]]] = None,
        fullbody_min_bbox_size: Optional[Union[Tuple[int, int], List[int]]] = None,


        fullbody_target_resolution_order: str = "wh",
        skip_missing: bool = False,
        **kwargs
    ):

        super().__init__()

        self.data_root = Path(data_root)
        self.neck_mode = neck_mode
        self.is_test = is_test

        self.validation_source = validation_source
        self.first_swap_dir_filter = first_swap_dir_filter
        self.use_head_alignment = bool(use_head_alignment)

        self.neck_color = neck_color
        self.hair_color = hair_color
        self.color_tolerance = color_tolerance
        self.dilate_neck = dilate_neck
        self.neck_dilation_kernel_size = neck_dilation_kernel_size
        self.dilate_hair = dilate_hair
        self.hair_dilation_kernel_size = hair_dilation_kernel_size

        self.target_resolution = target_resolution
        self.skip_missing = skip_missing


        valid_refinement_modes = ["partial", "full"]
        self.test_output_root = Path(test_output_root) if test_output_root else None
        if refinement_mode not in valid_refinement_modes:
            raise ValueError(
                f"refinement_mode must be one of {valid_refinement_modes}, got {refinement_mode}"
            )
        self.refinement_mode = refinement_mode


        self.adaptive_resolution_enabled = (refinement_mode == "full")

        self.adaptive_target_resolution = (448, 896)
        if fullbody_target_resolution is not None:
            ftr = fullbody_target_resolution
            if isinstance(ftr, ListConfig):
                ftr = list(ftr)
            if isinstance(ftr, (list, tuple)) and len(ftr) == 2:

                order = str(fullbody_target_resolution_order).lower().strip()
                if order not in ("wh", "hw"):
                    raise ValueError(f"fullbody_target_resolution_order must be 'wh' or 'hw', got: {fullbody_target_resolution_order}")
                if order == "wh":
                    self.adaptive_target_resolution = (int(ftr[0]), int(ftr[1]))
                else:
                    self.adaptive_target_resolution = (int(ftr[1]), int(ftr[0]))
            else:
                raise ValueError(
                    "fullbody_target_resolution must be a tuple/list/ListConfig of two ints, "
                    f"got: {fullbody_target_resolution}"
                )

        self.adaptive_min_bbox_size = (64, 64)
        if fullbody_min_bbox_size is not None:
            fmb = fullbody_min_bbox_size
            if isinstance(fmb, ListConfig):
                fmb = list(fmb)
            if isinstance(fmb, (list, tuple)) and len(fmb) == 2:
                order = str(fullbody_target_resolution_order).lower().strip()
                if order not in ("wh", "hw"):
                    raise ValueError(f"fullbody_target_resolution_order must be 'wh' or 'hw', got: {fullbody_target_resolution_order}")
                if order == "wh":
                    self.adaptive_min_bbox_size = (int(fmb[0]), int(fmb[1]))
                else:
                    self.adaptive_min_bbox_size = (int(fmb[1]), int(fmb[0]))
            else:
                raise ValueError(
                    "fullbody_min_bbox_size must be a tuple/list/ListConfig of two ints, "
                    f"got: {fullbody_min_bbox_size}"
                )

        if self.adaptive_resolution_enabled:

            if self.adaptive_target_resolution[0] % 8 != 0 or self.adaptive_target_resolution[1] % 8 != 0:
                raise ValueError(
                    "fullbody_target_resolution must be divisible by 8, got: "
                    f"{self.adaptive_target_resolution}"
                )
            logger.info(
                f"Full-body mode: adaptive resolution enabled ({self.adaptive_target_resolution[0]}×"
                f"{self.adaptive_target_resolution[1]}) (W×H)"
            )
        else:
            logger.info(f"Partial mode: simple resize ({target_resolution}×{target_resolution})")


        if subject_pairs is not None:

            self.subject_pairs = subject_pairs
            logger.info(f"Multi-pair mode: {len(subject_pairs)} pairs")
        elif subject_a is not None and subject_b is not None:

            self.subject_pairs = [(subject_a, subject_b)]
            logger.info(f"Single-pair mode: ({subject_a}, {subject_b})")
        else:

            self.subject_pairs = None
            logger.info("No subject filtering (discover all validation data)")


        if target_resolution % 8 != 0:
            raise ValueError(f"target_resolution must be divisible by 8, got {target_resolution}")


        valid_modes = ["union", "first_swap_only", "donator_only"]
        if neck_mode not in valid_modes:
            raise ValueError(f"neck_mode must be one of {valid_modes}, got {neck_mode}")


        valid_sources = ["combined_portraits", "first_swap"]
        if validation_source not in valid_sources:
            raise ValueError(f"validation_source must be one of {valid_sources}, got {validation_source}")


        self.samples = self._discover_validation_samples()

        logger.info(f"PartialRefinementValidationDataset initialized with {len(self.samples)} samples")
        logger.info(f"Validation source: {validation_source}")
        if validation_source == "first_swap":
            logger.info(f"First-swap directory filter: '{first_swap_dir_filter}'")
        logger.info(f"Target resolution: {target_resolution}x{target_resolution} (for inference)")
        logger.info(f"Neck mode: {neck_mode}, Dilate: {dilate_neck} (kernel_size={neck_dilation_kernel_size})")
        if self.dilate_hair:
            logger.info(f"Hair dilation: enabled (kernel_size={hair_dilation_kernel_size})")
        if self.subject_pairs:
            logger.info(f"From {len(self.subject_pairs)} subject pairs")

    def _discover_validation_samples(self) -> List[Dict[str, Any]]:

        if self.refinement_mode == "full":


            return self._discover_fullbody_validation_samples()
        else:

            if self.subject_pairs is None:

                return self._discover_all_validation_samples()
            else:

                return self._discover_multi_pair_validation_samples()

    def _discover_all_validation_samples(self) -> List[Dict[str, Any]]:

        samples = []

        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return samples


        if self.validation_source == "first_swap":

            swapped_dirs = sorted([d for d in self.data_root.iterdir()
                                  if d.is_dir()
                                  and d.name.startswith("swapped_")
                                  and "head_on" in d.name
                                  and self.first_swap_dir_filter in d.name])
        else:

            swapped_dirs = sorted([d for d in self.data_root.iterdir()
                                  if d.is_dir()
                                  and d.name.startswith("swapped_")
                                  and "head_on" in d.name])

        logger.info(f"Found {len(swapped_dirs)} swapped directories in {self.data_root}")

        for swapped_dir in swapped_dirs:
            dir_samples = self._discover_cameras_in_swapped_dir(swapped_dir)
            samples.extend(dir_samples)

        logger.info(f"Discovered {len(samples)} valid validation data samples")

        return samples

    def _discover_multi_pair_validation_samples(self) -> List[Dict[str, Any]]:

        all_samples = []

        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return all_samples


        if self.validation_source == "first_swap":
            all_swapped_dirs = sorted([d for d in self.data_root.iterdir()
                                      if d.is_dir()
                                      and d.name.startswith("swapped_")
                                      and "head_on" in d.name
                                      and self.first_swap_dir_filter in d.name])
        else:
            all_swapped_dirs = sorted([d for d in self.data_root.iterdir()
                                      if d.is_dir()
                                      and d.name.startswith("swapped_")
                                      and "head_on" in d.name])

        logger.info(f"Discovering validation data for {len(self.subject_pairs)} pairs...")
        logger.info(f"Found {len(all_swapped_dirs)} total swapped directories")

        for pair_idx, pair in enumerate(self.subject_pairs):
            subject_a, subject_b = pair


            pair_dirs = self._filter_swapped_dirs_by_pair(all_swapped_dirs, subject_a, subject_b)


            for swapped_dir in pair_dirs:
                dir_samples = self._discover_cameras_in_swapped_dir(swapped_dir)

                for sample in dir_samples:
                    sample["pair_subjects"] = pair
                all_samples.extend(dir_samples)


            if (pair_idx + 1) % 20 == 0:
                logger.info(
                    f"Progress: {pair_idx + 1}/{len(self.subject_pairs)} pairs, "
                    f"{len(all_samples)} samples so far"
                )

        logger.info(f"Discovered {len(all_samples)} validation samples from {len(self.subject_pairs)} pairs")

        return all_samples

    def _discover_fullbody_validation_samples(self) -> List[Dict[str, Any]]:

        samples = []

        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return samples


        swapped_dirs = sorted([
            d for d in self.data_root.iterdir()
            if d.is_dir()
            and d.name.startswith("swapped_")
            and "head_on" in d.name
        ])

        logger.info(f"Found {len(swapped_dirs)} swapped directories for full-body validation")


        if self.subject_pairs is not None:
            filtered_dirs = []
            for swapped_dir in swapped_dirs:
                head_id, body_id = self._extract_subject_ids(swapped_dir.name)
                if head_id is None or body_id is None:
                    continue


                if any(
                    (head_id == pair[0] and body_id == pair[1]) or
                    (head_id == pair[1] and body_id == pair[0])
                    for pair in self.subject_pairs
                ):
                    filtered_dirs.append(swapped_dir)

            swapped_dirs = filtered_dirs
            logger.info(f"Filtered to {len(swapped_dirs)} directories matching subject pairs")


        for swapped_dir in swapped_dirs:
            camera_dirs = sorted([
                d for d in swapped_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

            for camera_dir in camera_dirs:

                if (camera_dir / "0000.jpg").exists():
                    samples.append({
                        "swapped_dir": str(swapped_dir),
                        "camera_name": camera_dir.name,
                    })

        logger.info(f"Discovered {len(samples)} full-body validation samples")

        return samples

    def _discover_fullbody_test_samples(self) -> List[Dict[str, Any]]:

        samples = []

        if not self.test_output_root.exists():
            logger.warning(f"Test output root does not exist: {self.test_output_root}")
            return samples


        subject_dirs = sorted([
            d for d in self.test_output_root.iterdir()
            if d.is_dir()
            and d.name.startswith("swapped_")
            and "head_on" in d.name
        ])

        logger.info(f"Found {len(subject_dirs)} subject directories for full-body testing")


        if self.subject_pairs is not None:
            filtered_dirs = []
            for subject_dir in subject_dirs:
                head_id, body_id = self._extract_subject_ids(subject_dir.name)
                if head_id is None or body_id is None:
                    continue


                if any(
                    (head_id == pair[0] and body_id == pair[1]) or
                    (head_id == pair[1] and body_id == pair[0])
                    for pair in self.subject_pairs
                ):
                    filtered_dirs.append(subject_dir)

            subject_dirs = filtered_dirs
            logger.info(f"Filtered to {len(subject_dirs)} directories matching subject pairs")


        for subject_dir in subject_dirs:
            camera_dirs = sorted([
                d for d in subject_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ])

            for camera_dir in camera_dirs:

                metadata_path = camera_dir / "difix_intermediates" / "metadata.json"
                if metadata_path.exists():
                    samples.append({
                        "subject_dir": str(subject_dir),
                        "camera_name": camera_dir.name,
                    })

        logger.info(f"Discovered {len(samples)} full-body test samples from partial test outputs")

        return samples

    def _filter_swapped_dirs_by_pair(
        self,
        swapped_dirs: List[Path],
        subject_a: str,
        subject_b: str
    ) -> List[Path]:

        filtered_dirs = []

        for swapped_dir in swapped_dirs:
            head_id, body_id = self._extract_subject_ids(swapped_dir.name)


            if ((head_id == subject_a and body_id == subject_b) or
                (head_id == subject_b and body_id == subject_a)):
                filtered_dirs.append(swapped_dir)

        return filtered_dirs

    def _discover_cameras_in_swapped_dir(self, swapped_dir: Path) -> List[Dict[str, Any]]:

        samples = []


        head_id, body_id = self._extract_subject_ids(swapped_dir.name)


        camera_dirs = sorted([d for d in swapped_dir.iterdir()
                            if d.is_dir() and not d.name.startswith(".")])

        for camera_dir in camera_dirs:
            validation_data_dir = camera_dir / "validation_data" / f"{self.neck_mode}_neck"

            if not validation_data_dir.exists():
                if not self.skip_missing:
                    logger.debug(f"Missing validation_data directory: {validation_data_dir}")
                continue


            required_files = [
                "portrait_combined.jpg",
                "neck_mask_portrait.png",
                "bbox_metadata.yaml",
                "fullbody_combined.jpg",
                "portrait_gt.jpg",
                "head_mask_portrait.png",
                "body_mask_portrait.png",
            ]

            missing = [f for f in required_files if not (validation_data_dir / f).exists()]
            if missing:
                if not self.skip_missing:
                    logger.debug(f"Missing files in {validation_data_dir}: {missing}")
                continue


            sample = {
                "swapped_dir": str(swapped_dir),
                "camera_dir": str(camera_dir),
                "validation_data_dir": str(validation_data_dir),
                "head_id": head_id,
                "body_id": body_id,
                "camera_name": camera_dir.name,
            }

            samples.append(sample)

        return samples

    @staticmethod
    def _extract_subject_ids(directory_name: str) -> Tuple[Optional[str], Optional[str]]:


        match = re.search(r"swapped_(.+?)head_on_(.+?)body", directory_name)
        if match:
            return match.group(1), match.group(2)
        else:
            logger.warning(f"Could not extract subject IDs from: {directory_name}")
            return None, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:

        if self.refinement_mode == "full":


            if self.is_test and self.test_output_root is not None:
                try:
                    sample = self.samples[idx]
                    swapped_dir = Path(sample.get("swapped_dir", ""))
                    camera_name = sample.get("camera_name", "")

                    subj_dir = self.test_output_root / swapped_dir.name
                    meta = subj_dir / camera_name / "difix_intermediates" / "metadata.json"
                    if meta.exists():
                        return self._getitem_fullbody_test(idx)
                except Exception:

                    pass
            return self._getitem_fullbody(idx)
        else:
            return self._getitem_partial(idx)

    def _getitem_fullbody(self, idx: int) -> Dict:

        sample = self.samples[idx]

        try:

            swapped_dir = Path(sample["swapped_dir"])
            head_id, body_id = self._extract_subject_ids(swapped_dir.name)
            camera_name = sample["camera_name"]


            camera_dir = swapped_dir / camera_name
            degraded_path = camera_dir / "0000.jpg"
            degraded_pil = Image.open(degraded_path).convert("RGB")
            original_size = degraded_pil.size


            mask_path = camera_dir / "mask" / "pha" / "0000.png"
            if not mask_path.exists():
                logger.warning(f"Mask not found for bbox calculation: {mask_path}, using full image")

                bbox = (0, 0, original_size[0], original_size[1])
            else:
                bbox = calculate_bounding_box_from_mask(
                    str(mask_path),
                    self.adaptive_min_bbox_size
                )


            degraded_pil = apply_adaptive_resolution_transform(
                degraded_pil,
                bbox=bbox,
                target_resolution=self.adaptive_target_resolution
            )


            reverse_metadata = reverse_adaptive_resolution_metadata(
                bbox=bbox,
                original_size=original_size,
                target_resolution=self.adaptive_target_resolution
            )


            degraded_tensor = transforms.ToTensor()(degraded_pil)


            H, W = degraded_tensor.shape[1:]
            mask_tensor = torch.ones((1, H, W), dtype=torch.float32)


            return {
                "degraded": degraded_tensor,
                "mask": mask_tensor,
                "reverse_metadata": reverse_metadata,
                "metadata": {
                    "head_id": head_id,
                    "body_id": body_id,
                    "camera_name": camera_name,
                    "swapped_dir": str(swapped_dir),
                    "degraded_path": str(degraded_path),
                    "original_size": original_size,
                }
            }

        except Exception as e:
            logger.error(f"Error loading full-body validation sample {idx}: {e}")
            raise

    def _getitem_fullbody_test(self, idx: int) -> Dict:

        import json

        sample = self.samples[idx]

        try:

            camera_dir = Path(sample["subject_dir"]) / sample["camera_name"]


            fullbody_path = camera_dir / "0000.jpg"
            fullbody_pil = Image.open(fullbody_path).convert("RGB")
            original_size = fullbody_pil.size


            mask_path = camera_dir / "mask" / "pha" / "0000.png"
            if not mask_path.exists():
                logger.warning(f"Mask not found for bbox calculation: {mask_path}, using full image")
                bbox = (0, 0, original_size[0], original_size[1])
            else:
                bbox = calculate_bounding_box_from_mask(
                    str(mask_path),
                    self.adaptive_min_bbox_size
                )


            fullbody_transformed = apply_adaptive_resolution_transform(
                fullbody_pil,
                bbox=bbox,
                target_resolution=self.adaptive_target_resolution
            )


            reverse_metadata = reverse_adaptive_resolution_metadata(
                bbox=bbox,
                original_size=original_size,
                target_resolution=self.adaptive_target_resolution
            )


            metadata_path = camera_dir / "difix_intermediates" / "metadata.json"
            with open(metadata_path, "r") as f:
                restoration_metadata = json.load(f)


            degraded_tensor = transforms.ToTensor()(fullbody_transformed)


            H, W = degraded_tensor.shape[1:]
            mask_tensor = torch.ones((1, H, W), dtype=torch.float32)

            return {
                "degraded": degraded_tensor,
                "mask": mask_tensor,
                "reverse_metadata": reverse_metadata,
                "restoration_metadata": restoration_metadata,
                "metadata": {
                    "camera_dir": str(camera_dir),
                    "camera_name": sample["camera_name"],
                    "subject_dir": sample["subject_dir"],
                }
            }

        except Exception as e:
            logger.error(f"Error loading full-body test sample {idx}: {e}")
            raise

    def _getitem_partial(self, idx: int) -> Dict:

        sample = self.samples[idx]
        val_dir = Path(sample["validation_data_dir"])
        camera_dir = Path(sample["camera_dir"])

        try:

            if self.validation_source == "first_swap":

                portrait_combined = cv2.imread(str(val_dir / "portrait_first_swap.jpg"))
                if portrait_combined is None:
                    raise IOError(f"Failed to read portrait_first_swap.jpg from {val_dir}")
                portrait_combined = cv2.cvtColor(portrait_combined, cv2.COLOR_BGR2RGB)


                if self.use_head_alignment:
                    fullbody_combined = self._load_head_aligned_fullbody(camera_dir)
                else:
                    fullbody_combined = self._load_standard_fullbody(camera_dir)
            else:

                fullbody_combined = cv2.imread(str(val_dir / "fullbody_combined.jpg"))
                if fullbody_combined is None:
                    raise IOError(f"Failed to read fullbody_combined.jpg from {val_dir}")
                fullbody_combined = cv2.cvtColor(fullbody_combined, cv2.COLOR_BGR2RGB)

                portrait_combined = cv2.imread(str(val_dir / "portrait_combined.jpg"))
                if portrait_combined is None:
                    raise IOError(f"Failed to read portrait_combined.jpg from {val_dir}")
                portrait_combined = cv2.cvtColor(portrait_combined, cv2.COLOR_BGR2RGB)

            portrait_gt = cv2.imread(str(val_dir / "portrait_gt.jpg"))
            if portrait_gt is None:
                raise IOError(f"Failed to read portrait_gt.jpg from {val_dir}")
            portrait_gt = cv2.cvtColor(portrait_gt, cv2.COLOR_BGR2RGB)


            segmentation_gt_portrait = cv2.imread(str(val_dir / "segmentation_gt_portrait.png"))
            if segmentation_gt_portrait is None:
                raise IOError(f"Failed to read segmentation_gt_portrait.png from {val_dir}")
            segmentation_gt_portrait = cv2.cvtColor(segmentation_gt_portrait, cv2.COLOR_BGR2RGB)


            neck_mask = cv2.imread(str(val_dir / "neck_mask_portrait.png"), cv2.IMREAD_GRAYSCALE)
            if neck_mask is None:
                raise IOError(f"Failed to read neck_mask_portrait.png from {val_dir}")


            head_mask = cv2.imread(str(val_dir / "head_mask_portrait.png"), cv2.IMREAD_GRAYSCALE)
            if head_mask is None:
                raise IOError(f"Failed to read head_mask_portrait.png from {val_dir}")

            body_mask = cv2.imread(str(val_dir / "body_mask_portrait.png"), cv2.IMREAD_GRAYSCALE)
            if body_mask is None:
                raise IOError(f"Failed to read body_mask_portrait.png from {val_dir}")


            original_portrait_height, original_portrait_width = portrait_combined.shape[:2]


            portrait_combined_resized = cv2.resize(
                portrait_combined,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_LINEAR
            )
            portrait_gt_resized = cv2.resize(
                portrait_gt,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_LINEAR
            )


            neck_mask_resized = cv2.resize(
                neck_mask,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_NEAREST
            )
            head_mask_resized = cv2.resize(
                head_mask,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_NEAREST
            )
            body_mask_resized = cv2.resize(
                body_mask,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_NEAREST
            )


            neck_mask_binary = neck_mask > 0


            hair_mask_binary = match_color(
                segmentation_gt_portrait,
                target_color=self.hair_color,
                tolerance=self.color_tolerance
            )


            if self.dilate_neck:
                neck_mask_binary = dilate_mask(
                    neck_mask_binary,
                    kernel_size=self.neck_dilation_kernel_size,
                    iterations=1
                )

            if self.dilate_hair:
                hair_mask_binary = dilate_mask(
                    hair_mask_binary,
                    kernel_size=self.hair_dilation_kernel_size,
                    iterations=1
                )


            neck_hair_mask_combined = neck_mask_binary | hair_mask_binary


            neck_hair_mask = (neck_hair_mask_combined.astype(np.uint8) * 255)


            neck_hair_mask_resized = cv2.resize(
                neck_hair_mask,
                (self.target_resolution, self.target_resolution),
                interpolation=cv2.INTER_NEAREST
            )


            with open(val_dir / "bbox_metadata.yaml") as f:
                bbox_metadata = yaml.safe_load(f)


            fullbody_tensor = torch.from_numpy(fullbody_combined).permute(2, 0, 1).float() / 255.0
            portrait_combined_tensor = torch.from_numpy(portrait_combined_resized).permute(2, 0, 1).float() / 255.0
            portrait_gt_tensor = torch.from_numpy(portrait_gt_resized).permute(2, 0, 1).float() / 255.0


            neck_mask_tensor = torch.from_numpy(neck_mask_resized).unsqueeze(0).float() / 255.0
            head_mask_tensor = torch.from_numpy(head_mask_resized).unsqueeze(0).float() / 255.0
            body_mask_tensor = torch.from_numpy(body_mask_resized).unsqueeze(0).float() / 255.0
            neck_hair_mask_tensor = torch.from_numpy(neck_hair_mask_resized).unsqueeze(0).float() / 255.0

            return {
                "fullbody_combined": fullbody_tensor,
                "portrait_combined": portrait_combined_tensor,
                "portrait_gt": portrait_gt_tensor,
                "neck_mask": neck_mask_tensor,
                "head_mask": head_mask_tensor,
                "body_mask": body_mask_tensor,
                "neck_hair_mask": neck_hair_mask_tensor,
                "bbox_metadata": bbox_metadata,
                "metadata": {
                    "head_id": sample["head_id"],
                    "body_id": sample["body_id"],
                    "camera_name": sample["camera_name"],
                    "swapped_dir": sample["swapped_dir"],
                    "original_portrait_height": original_portrait_height,
                    "original_portrait_width": original_portrait_width,
                }
            }

        except Exception as e:
            logger.error(f"Error loading sample {idx} from {sample['validation_data_dir']}: {e}")
            raise

    def _load_standard_fullbody(self, camera_dir: Path) -> np.ndarray:

        candidates = [
            camera_dir / "0000.jpg",
            camera_dir / "0000.png",
        ]
        for p in candidates:
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raise IOError(
            "Standard fullbody image not found. Tried:\n"
            + "\n".join([f"  - {p}" for p in candidates])
        )

    def _load_head_aligned_fullbody(self, camera_dir: Path) -> np.ndarray:


        fullbody_path_no_sam = camera_dir / "head_aligned_no_sam" / "0000.png"
        if fullbody_path_no_sam.exists():
            fullbody = cv2.imread(str(fullbody_path_no_sam))
            if fullbody is not None:
                logger.debug(f"Loaded head-aligned fullbody from head_aligned_no_sam: {fullbody_path_no_sam}")
                return cv2.cvtColor(fullbody, cv2.COLOR_BGR2RGB)


        fullbody_path = camera_dir / "head_aligned" / "0000.png"
        if fullbody_path.exists():
            fullbody = cv2.imread(str(fullbody_path))
            if fullbody is not None:
                logger.debug(f"Loaded head-aligned fullbody from head_aligned (fallback): {fullbody_path}")
                return cv2.cvtColor(fullbody, cv2.COLOR_BGR2RGB)


        raise IOError(
            f"Head-aligned fullbody image not found at either location:\n"
            f"  Preferred: {fullbody_path_no_sam}\n"
            f"  Fallback: {fullbody_path}"
        )
