import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from loguru import logger

from ..utils.color_matching import create_neck_hair_mask_with_separate_dilation
from ..utils.augmentation import SynchronizedRandomRotation
from .difix3d_datamodule import (
    calculate_bounding_box_from_mask,
    apply_adaptive_resolution_transform,
    apply_adaptive_resolution_transform_to_mask,
    reverse_adaptive_resolution_metadata,
    construct_gt_mask_path,
)


class PartialRefinementDataset(Dataset):


    def __init__(
        self,
        data_root: str,
        prompts: Optional[List[str]] = None,
        neck_color: Tuple[int, int, int] = (85, 51, 0),
        hair_color: Tuple[int, int, int] = (255, 0, 0),
        color_tolerance: int = 5,
        dilate_neck: bool = True,
        neck_dilation_kernel_size: int = 7,
        dilate_hair: bool = False,
        hair_dilation_kernel_size: int = 3,
        validate_data_existence: bool = True,
        transform: Optional[transforms.Compose] = None,

        subject_pairs: Optional[List[Tuple[str, str]]] = None,
        subject_a: Optional[str] = None,
        subject_b: Optional[str] = None,

        strict_mode: bool = True,

        target_resolution: int = 512,

        training_neck_mode: str = "gt",

        both_mode_swapped_ratio: float = 0.5,
        both_mode_random_seed: Optional[int] = None,
        both_mode_max_oversampling: float = 3.0,

        enable_rotation_aug: bool = True,
        rotation_degrees: Tuple[float, float] = (-30, 30),
        rotation_prob: float = 0.8,

        refinement_mode: str = "partial",
        gt_dataset_root: Optional[str] = None,
    ):

        super().__init__()

        self.data_root = Path(data_root)
        self.prompts = prompts or ["remove degradation"]
        self.neck_color = neck_color
        self.hair_color = hair_color
        self.color_tolerance = color_tolerance
        self.dilate_neck = dilate_neck
        self.neck_dilation_kernel_size = neck_dilation_kernel_size
        self.dilate_hair = dilate_hair
        self.hair_dilation_kernel_size = hair_dilation_kernel_size
        self.validate_data_existence = validate_data_existence
        self.strict_mode = strict_mode
        self.target_resolution = target_resolution


        valid_modes = ["gt", "swapped", "both"]
        if training_neck_mode not in valid_modes:
            raise ValueError(f"training_neck_mode must be one of {valid_modes}, got {training_neck_mode}")

        self.training_neck_mode = training_neck_mode
        if training_neck_mode == "gt":
            self.combined_dir_name = "combined"
        else:
            self.combined_dir_name = f"combined_{training_neck_mode}_neck"


        self.both_mode_swapped_ratio = both_mode_swapped_ratio
        self.both_mode_random_seed = both_mode_random_seed
        self.both_mode_max_oversampling = both_mode_max_oversampling


        self.enable_rotation_aug = enable_rotation_aug
        self.rotation_augmentation = None
        if enable_rotation_aug:
            self.rotation_augmentation = SynchronizedRandomRotation(
                degrees=rotation_degrees,
                prob=rotation_prob,
                fill=0
            )
            logger.info(f"Rotation augmentation enabled: {self.rotation_augmentation}")
        else:
            logger.info("Rotation augmentation disabled")


        valid_modes = ["partial", "full"]
        if refinement_mode not in valid_modes:
            raise ValueError(
                f"refinement_mode must be one of {valid_modes}, got {refinement_mode}. "
                f"Use 'partial' for neck+hair refinement, 'full' for full-body refinement."
            )

        self.refinement_mode = refinement_mode


        if refinement_mode == "full" and not gt_dataset_root:
            raise ValueError(
                "gt_dataset_root is required when refinement_mode='full'. "
                "GT dataset root is needed for GT mask lookup for bbox calculation."
            )

        self.gt_dataset_root = Path(gt_dataset_root) if gt_dataset_root else None


        self.adaptive_resolution_enabled = (refinement_mode == "full")
        self.adaptive_target_resolution = (448, 896)
        self.adaptive_min_bbox_size = (64, 64)


        if refinement_mode == "full":
            logger.info("Full-body refinement mode: Will use adaptive resolution (448×896) with all-ones mask")
            logger.info(f"GT dataset root: {gt_dataset_root}")

            if enable_rotation_aug:
                logger.warning("Rotation augmentation is postponed for full-body mode (will be disabled)")
                self.enable_rotation_aug = False
                self.rotation_augmentation = None
        else:
            logger.info("Partial refinement mode: Will use simple resize (512×512) with neck+hair mask from segmentation")


        if subject_pairs is not None:

            self.subject_pairs = subject_pairs
            logger.info(f"Multi-pair mode: {len(subject_pairs)} pairs")
        elif subject_a is not None and subject_b is not None:

            self.subject_pairs = [(subject_a, subject_b)]
            logger.info(f"Single-pair mode: ({subject_a}, {subject_b})")
        else:

            self.subject_pairs = None
            logger.info("No subject filtering (discover all data)")


        if target_resolution % 8 != 0:
            raise ValueError(f"target_resolution must be divisible by 8, got {target_resolution}")


        if self.adaptive_resolution_enabled:

            self.transform = transform or transforms.ToTensor()
            logger.info(f"Adaptive resolution enabled: {self.adaptive_target_resolution}")
        else:

            self.transform = transform or transforms.Compose([
                transforms.Resize((target_resolution, target_resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])
            logger.info(f"Simple resize enabled: {target_resolution}x{target_resolution}")


        if self.refinement_mode == "full":

            logger.info("Full-body mode: discovering samples from root level ([cam]/0000.jpg)")
            self.samples = self._discover_fullbody_samples()
        elif self.training_neck_mode == "both":

            logger.info("Partial mode with mixed sampling: discovering from both combined directories")
            self.samples = self._discover_both_modes_samples()
        else:

            logger.info(f"Partial mode: discovering from {self.combined_dir_name} directory")
            self.samples = self._discover_samples()


        if strict_mode and self.subject_pairs:
            self._validate_strict_mode()

        logger.info(f"PartialRefinementDataset initialized with {len(self.samples)} samples")
        if self.refinement_mode == "full":
            logger.info(f"Full-body mode: loading from root level")
        else:
            logger.info(f"Partial mode: training neck mode = {training_neck_mode} (directory: {self.combined_dir_name})")
        if self.subject_pairs:
            logger.info(f"From {len(self.subject_pairs)} subject pairs")

    def _discover_fullbody_samples(self) -> List[Dict[str, Any]]:

        samples = []

        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return samples


        for restored_dir in sorted(self.data_root.glob("restored_*")):
            if not restored_dir.is_dir():
                continue

            restored_dir_name = restored_dir.name


            if self.subject_pairs is not None:

                import re
                match = re.search(r'restored_(\d+)_from', restored_dir_name)
                if not match:
                    continue

                subject_id = match.group(1)


                subject_in_pairs = any(
                    subject_id == pair[0] or subject_id == pair[1]
                    for pair in self.subject_pairs
                )

                if not subject_in_pairs:
                    continue


            for cam_dir in sorted(restored_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue

                camera_name = cam_dir.name


                degraded_path = cam_dir / "0000.jpg"
                if not degraded_path.exists():
                    continue


                samples.append({
                    "restored_dir": restored_dir_name,
                    "camera_dir": camera_name,
                    "degraded_path": str(degraded_path),
                    "combined_path": str(degraded_path),
                })

        logger.info(f"Discovered {len(samples)} full-body samples from {len(set(s['restored_dir'] for s in samples))} restored directories")

        return samples

    def _discover_samples(self) -> List[Dict[str, Any]]:

        if self.subject_pairs is None:

            return self._discover_all_samples()
        else:

            return self._discover_multi_pair_samples()

    def _discover_all_samples(self) -> List[Dict[str, Any]]:

        samples = []

        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return samples


        restored_dirs = sorted([d for d in self.data_root.iterdir()
                              if d.is_dir() and d.name.startswith("restored_")])

        logger.info(f"Found {len(restored_dirs)} restored directories in {self.data_root}")

        for restored_dir in restored_dirs:
            dir_samples = self._discover_cameras_in_dir(restored_dir)
            samples.extend(dir_samples)

        logger.info(f"Discovered {len(samples)} valid combined portrait samples")

        return samples

    def _discover_multi_pair_samples(self) -> List[Dict[str, Any]]:

        all_samples = []
        self.pair_sample_counts = {}


        if not self.data_root.exists():
            logger.warning(f"Data root does not exist: {self.data_root}")
            return all_samples

        all_restored_dirs = sorted([d for d in self.data_root.iterdir()
                                   if d.is_dir() and d.name.startswith("restored_")])

        logger.info(f"Discovering data for {len(self.subject_pairs)} pairs...")
        logger.info(f"Found {len(all_restored_dirs)} total restored directories")

        for pair_idx, pair in enumerate(self.subject_pairs):
            subject_a, subject_b = pair


            pair_samples = self._discover_samples_for_pair(
                subject_a, subject_b, all_restored_dirs
            )


            self.pair_sample_counts[pair] = len(pair_samples)
            all_samples.extend(pair_samples)


            if (pair_idx + 1) % 20 == 0:
                logger.info(
                    f"Progress: {pair_idx + 1}/{len(self.subject_pairs)} pairs, "
                    f"{len(all_samples)} samples so far"
                )


        if logger._core.min_level <= 10:
            for pair, count in self.pair_sample_counts.items():
                logger.debug(f"Pair {pair}: {count} samples")

        logger.info(f"Discovered {len(all_samples)} samples from {len(self.subject_pairs)} pairs")

        return all_samples

    def _discover_samples_for_pair(
        self,
        subject_a: str,
        subject_b: str,
        all_restored_dirs: List[Path]
    ) -> List[Dict[str, Any]]:


        pair_dirs = self._filter_dirs_by_pair(
            all_restored_dirs, subject_a, subject_b
        )


        pair_samples = []
        for restored_dir in pair_dirs:
            dir_samples = self._discover_cameras_in_dir(restored_dir)

            for sample in dir_samples:
                sample["pair_subjects"] = (subject_a, subject_b)
            pair_samples.extend(dir_samples)

        return pair_samples

    def _filter_dirs_by_pair(
        self,
        restored_dirs: List[Path],
        subject_a: str,
        subject_b: str
    ) -> List[Path]:

        import re

        filtered_dirs = []
        target_subjects = {subject_a, subject_b}

        for restored_dir in restored_dirs:
            dir_name = restored_dir.name


            restored_match = re.search(r'restored_(\d+)_from', dir_name)
            restored_subject = restored_match.group(1) if restored_match else None

            swapped_match = re.search(r'swapped_(\d+)head_on_(\d+)body', dir_name)
            if swapped_match:
                head_subject = swapped_match.group(1)
                body_subject = swapped_match.group(2)
                found_subjects = {restored_subject, head_subject, body_subject}
            else:
                found_subjects = {restored_subject} if restored_subject else set()


            if target_subjects.issubset(found_subjects):
                filtered_dirs.append(restored_dir)

        return filtered_dirs

    def _discover_cameras_in_dir(self, restored_dir: Path) -> List[Dict[str, Any]]:

        samples = []


        camera_dirs = sorted([d for d in restored_dir.iterdir()
                            if d.is_dir() and not d.name.startswith(".")])

        for camera_dir in camera_dirs:
            combined_dir = camera_dir / self.combined_dir_name

            if not combined_dir.exists():
                continue


            combined_path = combined_dir / "combined.jpg"
            gt_path = combined_dir / "gt.jpg"
            segmentation_path = combined_dir / "segmentation.png"


            if self.validate_data_existence:
                if not combined_path.exists():
                    logger.debug(f"Missing combined.jpg: {combined_path}")
                    continue
                if not gt_path.exists():
                    logger.debug(f"Missing gt.jpg: {gt_path}")
                    continue
                if not segmentation_path.exists():
                    logger.debug(f"Missing segmentation.png: {segmentation_path}")
                    continue


            sample = {
                "combined_path": str(combined_path),
                "gt_path": str(gt_path),
                "segmentation_path": str(segmentation_path),
                "restored_dir": restored_dir.name,
                "camera_dir": camera_dir.name,
            }

            samples.append(sample)

        return samples

    def _discover_both_modes_samples(self) -> List[Dict[str, Any]]:


        original_dir_name = self.combined_dir_name


        self.combined_dir_name = "combined"
        samples_gt = self._discover_samples()
        logger.info(f"Discovered {len(samples_gt)} samples from combined_gt_neck")


        self.combined_dir_name = "combined_swapped_neck"
        samples_swapped = self._discover_samples()
        logger.info(f"Discovered {len(samples_swapped)} samples from combined_swapped_neck")


        self.combined_dir_name = original_dir_name


        if len(samples_gt) == 0 and len(samples_swapped) == 0:
            raise ValueError(
                "No samples found in either combined_gt_neck or combined_swapped_neck. "
                "Check data_root path and ensure Stage 29 has generated combined portrait data."
            )

        if len(samples_gt) == 0:
            raise ValueError(
                f"Cannot use training_neck_mode='both' - GT mode data is missing.\n\n"
                f"  Data root: {self.data_root}\n"
                f"  Expected pattern: {{data_root}}/restored_*/*/combined_gt_neck/\n"
                f"  Swapped samples found: {len(samples_swapped)}\n"
                f"  GT samples found: 0\n\n"
                f"  This usually means:\n"
                f"  1. Stage 29 was run with neck_source_mode excluding 'gt'\n"
                f"  2. Data directories were moved or renamed\n"
                f"  3. Wrong data_root path configured\n\n"
                f"  Solutions:\n"
                f"  1. Re-run Stage 29 with neck_source_mode='gt' or 'both'\n"
                f"  2. Use training_neck_mode='swapped' (only swapped available)\n"
                f"  3. Check data_root path in config"
            )

        if len(samples_swapped) == 0:
            raise ValueError(
                f"Cannot use training_neck_mode='both' - Swapped mode data is missing.\n\n"
                f"  Data root: {self.data_root}\n"
                f"  Expected pattern: {{data_root}}/restored_*/*/combined_swapped_neck/\n"
                f"  GT samples found: {len(samples_gt)}\n"
                f"  Swapped samples found: 0\n\n"
                f"  This usually means:\n"
                f"  1. Stage 29 was run with neck_source_mode excluding 'swapped'\n"
                f"  2. Data directories were moved or renamed\n"
                f"  3. Wrong data_root path configured\n\n"
                f"  Solutions:\n"
                f"  1. Re-run Stage 29 with neck_source_mode='swapped' or 'both'\n"
                f"  2. Use training_neck_mode='gt' (only GT available)\n"
                f"  3. Check data_root path in config"
            )


        mixed_samples = self._mix_samples(samples_gt, samples_swapped)

        return mixed_samples

    def _mix_samples(
        self,
        samples_gt: List[Dict[str, Any]],
        samples_swapped: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        import random

        N_gt = len(samples_gt)
        N_swapped = len(samples_swapped)
        total_available = N_gt + N_swapped


        target_swapped = round(total_available * self.both_mode_swapped_ratio)
        target_gt = total_available - target_swapped

        logger.info(f"Target mixing: {target_swapped} swapped ({self.both_mode_swapped_ratio:.1%}) "
                    f"+ {target_gt} gt ({1-self.both_mode_swapped_ratio:.1%})")


        swapped_oversample = target_swapped / N_swapped if N_swapped > 0 else 0
        gt_oversample = target_gt / N_gt if N_gt > 0 else 0

        max_oversample = max(swapped_oversample, gt_oversample)


        if max_oversample > self.both_mode_max_oversampling:
            raise ValueError(
                f"Cannot achieve mixing ratio {self.both_mode_swapped_ratio:.0%}:{1-self.both_mode_swapped_ratio:.0%} with available data.\n\n"
                f"  Available: {N_gt} gt, {N_swapped} swapped\n"
                f"  Target: {target_gt} gt ({1-self.both_mode_swapped_ratio:.0%}), {target_swapped} swapped ({self.both_mode_swapped_ratio:.0%})\n"
                f"  Required oversampling: gt={gt_oversample:.1f}x, swapped={swapped_oversample:.1f}x\n"
                f"  Max oversampling ({max_oversample:.1f}x) exceeds limit ({self.both_mode_max_oversampling}x).\n\n"
                f"  Solutions:\n"
                f"  1. Adjust both_mode_swapped_ratio to reduce {'gt' if gt_oversample > swapped_oversample else 'swapped'} requirement\n"
                f"  2. Increase both_mode_max_oversampling limit (not recommended)\n"
                f"  3. Collect more {'gt' if gt_oversample > swapped_oversample else 'swapped'} mode data"
            )


        if self.both_mode_random_seed is not None:
            random.seed(self.both_mode_random_seed)


        replace_swapped = target_swapped > N_swapped
        replace_gt = target_gt > N_gt

        sampled_swapped = random.choices(samples_swapped, k=target_swapped) if replace_swapped else random.sample(samples_swapped, target_swapped)
        sampled_gt = random.choices(samples_gt, k=target_gt) if replace_gt else random.sample(samples_gt, target_gt)


        sampled_swapped = self._add_neck_mode_label(sampled_swapped, "swapped")
        sampled_gt = self._add_neck_mode_label(sampled_gt, "gt")


        mixed_samples = sampled_swapped + sampled_gt
        random.shuffle(mixed_samples)


        actual_swapped_ratio = len(sampled_swapped) / len(mixed_samples)
        logger.info(f"Mixed dataset created: {len(mixed_samples)} samples total")
        logger.info(f"  Swapped: {len(sampled_swapped)} samples ({actual_swapped_ratio:.1%})")
        logger.info(f"  GT: {len(sampled_gt)} samples ({1-actual_swapped_ratio:.1%})")


        if swapped_oversample > 1.0:
            logger.warning(
                f"Swapped mode oversampling detected\n"
                f"  Available: {N_swapped} swapped samples\n"
                f"  Target: {target_swapped} swapped samples ({self.both_mode_swapped_ratio:.0%} of total)\n"
                f"  Oversampling: {swapped_oversample:.2f}x ({target_swapped - N_swapped} samples will be duplicated)\n"
                f"  This may lead to overfitting on duplicated samples."
            )
        if gt_oversample > 1.0:
            logger.warning(
                f"GT mode oversampling detected\n"
                f"  Available: {N_gt} gt samples\n"
                f"  Target: {target_gt} gt samples ({1-self.both_mode_swapped_ratio:.0%} of total)\n"
                f"  Oversampling: {gt_oversample:.2f}x ({target_gt - N_gt} samples will be duplicated)\n"
                f"  This may lead to overfitting on duplicated samples."
            )


        if swapped_oversample < 1.0:
            logger.warning(
                f"Swapped mode downsampling detected\n"
                f"  Available: {N_swapped} swapped samples\n"
                f"  Target: {target_swapped} swapped samples ({self.both_mode_swapped_ratio:.0%} of total)\n"
                f"  Downsampling: {swapped_oversample:.1%} ({N_swapped - target_swapped} samples unused)"
            )
        if gt_oversample < 1.0:
            logger.warning(
                f"GT mode downsampling detected\n"
                f"  Available: {N_gt} gt samples\n"
                f"  Target: {target_gt} gt samples ({1-self.both_mode_swapped_ratio:.0%} of total)\n"
                f"  Downsampling: {gt_oversample:.1%} ({N_gt - target_gt} samples unused)"
            )

        return mixed_samples

    def _add_neck_mode_label(
        self,
        samples: List[Dict[str, Any]],
        neck_mode: str
    ) -> List[Dict[str, Any]]:

        labeled_samples = []
        for sample in samples:

            labeled_sample = sample.copy()
            labeled_sample["neck_mode"] = neck_mode
            labeled_samples.append(labeled_sample)

        return labeled_samples

    def _validate_strict_mode(self):

        pairs_without_data = []

        for pair in self.subject_pairs:
            if self.pair_sample_counts.get(pair, 0) == 0:
                pairs_without_data.append(pair)

        if pairs_without_data:

            total_pairs = len(self.subject_pairs)
            pairs_with_data = total_pairs - len(pairs_without_data)

            error_msg = (
                f"Strict mode validation failed: {len(pairs_without_data)} pairs have no data.\n\n"
                f"Missing pairs (first 5): {pairs_without_data[:5]}\n"
                f"Total pairs requested: {total_pairs}\n"
                f"Pairs with data: {pairs_with_data}\n"
                f"Pairs without data: {len(pairs_without_data)}\n\n"
                f"Suggestions:\n"
                f"1. Check if data exists for these pairs in: {self.data_root}\n"
                f"2. Use pair filtering to exclude these pairs (exclude_pairs or exclude_pair_ids)\n"
                f"3. Set strict_mode=False to skip validation (not recommended)"
            )

            logger.error(error_msg)
            raise ValueError(error_msg)

    def _extract_subject_id(self, sample: Dict) -> str:

        import re

        restored_dir = sample["restored_dir"]


        match = re.search(r'restored_(\d+)_from', restored_dir)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Cannot extract subject_id from restored_dir: {restored_dir}")

    def _extract_camera_name(self, sample: Dict) -> str:

        return sample["camera_dir"]

    def _create_neck_hair_mask(self, segmentation: np.ndarray) -> np.ndarray:


        mask = create_neck_hair_mask_with_separate_dilation(
            segmentation,
            neck_color=self.neck_color,
            hair_color=self.hair_color,
            tolerance=self.color_tolerance,
            dilate_neck=self.dilate_neck,
            neck_dilation_kernel_size=self.neck_dilation_kernel_size,
            dilate_hair=self.dilate_hair,
            hair_dilation_kernel_size=self.hair_dilation_kernel_size
        )


        mask = mask.astype(np.float32)

        return mask

    def __len__(self) -> int:

        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self.samples[idx]

        if self.refinement_mode == "full":
            return self._getitem_fullbody(sample)
        else:
            return self._getitem_partial(sample)

    def _getitem_fullbody(self, sample: Dict) -> Dict[str, torch.Tensor]:


        subject_id = self._extract_subject_id(sample)
        camera_name = self._extract_camera_name(sample)


        degraded_path = sample["degraded_path"]
        degraded_pil = Image.open(degraded_path).convert("RGB")
        original_size = degraded_pil.size


        gt_path = self.gt_dataset_root / subject_id / camera_name / "0000.jpg"
        gt_pil = Image.open(gt_path).convert("RGB")


        gt_mask_path = construct_gt_mask_path(str(gt_path))
        bbox = calculate_bounding_box_from_mask(
            gt_mask_path,
            self.adaptive_min_bbox_size
        )


        degraded_pil = apply_adaptive_resolution_transform(
            degraded_pil,
            bbox=bbox,
            target_resolution=self.adaptive_target_resolution,
            min_bbox_size=self.adaptive_min_bbox_size
        )

        gt_pil = apply_adaptive_resolution_transform(
            gt_pil,
            bbox=bbox,
            target_resolution=self.adaptive_target_resolution,
            min_bbox_size=self.adaptive_min_bbox_size
        )


        reverse_metadata = reverse_adaptive_resolution_metadata(
            bbox=bbox,
            original_size=original_size,
            target_resolution=self.adaptive_target_resolution
        )


        degraded_tensor = self.transform(degraded_pil)
        gt_tensor = self.transform(gt_pil)


        H, W = degraded_tensor.shape[1:]
        mask_tensor = torch.ones((1, H, W), dtype=torch.float32)


        prompt = self.prompts[0] if self.prompts else "remove degradation"


        metadata = {
            "restored_dir": sample["restored_dir"],
            "camera_dir": sample["camera_dir"],
            "subject_id": subject_id,
            "degraded_path": degraded_path,
            "gt_path": str(gt_path),
            "original_size": original_size,
            "reverse_metadata": reverse_metadata,
        }

        return {
            "degraded": degraded_tensor,
            "target": gt_tensor,
            "mask": mask_tensor,
            "prompt": prompt,
            "metadata": metadata,
        }

    def _getitem_partial(self, sample: Dict) -> Dict[str, torch.Tensor]:


        combined_img = Image.open(sample["combined_path"]).convert("RGB")
        gt_img = Image.open(sample["gt_path"]).convert("RGB")
        segmentation = np.array(Image.open(sample["segmentation_path"]).convert("RGB"))


        if self.rotation_augmentation is not None:
            combined_img, gt_img, segmentation = self.rotation_augmentation(
                combined_img, gt_img, segmentation
            )


        neck_hair_mask = self._create_neck_hair_mask(segmentation)


        combined_tensor = self.transform(combined_img)
        gt_tensor = self.transform(gt_img)


        mask_pil = Image.fromarray((neck_hair_mask * 255).astype(np.uint8), mode='L')
        mask_resized = mask_pil.resize(
            (self.target_resolution, self.target_resolution),
            resample=Image.NEAREST
        )
        mask_resized_array = np.array(mask_resized).astype(np.float32) / 255.0


        mask_tensor = torch.from_numpy(mask_resized_array).unsqueeze(0)


        prompt = self.prompts[0] if self.prompts else "remove degradation"


        metadata = {
            "restored_dir": sample["restored_dir"],
            "camera_dir": sample["camera_dir"],
            "combined_path": sample["combined_path"],
            "gt_path": sample["gt_path"],
            "segmentation_path": sample["segmentation_path"],
        }

        return {
            "degraded": combined_tensor,
            "target": gt_tensor,
            "mask": mask_tensor,
            "prompt": prompt,
            "metadata": metadata,
        }
