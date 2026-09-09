from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
import lightning as L
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .difix3d_datamodule import DiFix3DDataModule
from .partial_refinement_dataset import PartialRefinementDataset
from .partial_refinement_validation_dataset import PartialRefinementValidationDataset


class PartialRefinementDataModule(DiFix3DDataModule):


    def __init__(
        self,

        combined_data_root: str,


        train_pairs_yaml: Optional[str] = None,
        val_pairs_yaml: Optional[str] = None,


        exclude_pairs: Optional[List[Tuple[str, str]]] = None,
        include_only_pairs: Optional[List[Tuple[str, str]]] = None,
        exclude_pair_ids: Optional[List[Union[int, List[int]]]] = None,
        include_only_pair_ids: Optional[List[Union[int, List[int]]]] = None,


        val_exclude_pairs: Optional[List[Tuple[str, str]]] = None,
        val_include_only_pairs: Optional[List[Tuple[str, str]]] = None,
        val_exclude_pair_ids: Optional[List[Union[int, List[int]]]] = None,
        val_include_only_pair_ids: Optional[List[Union[int, List[int]]]] = None,


        strict_mode: bool = True,


        neck_color: tuple[int, int, int] = (85, 51, 0),
        hair_color: tuple[int, int, int] = (255, 0, 0),
        color_tolerance: int = 5,
        dilate_neck: bool = True,
        neck_dilation_kernel_size: int = 7,
        dilate_hair: bool = False,
        hair_dilation_kernel_size: int = 3,


        subject_a: Optional[str] = None,
        subject_b: Optional[str] = None,


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

        fullbody_target_resolution: Optional[Union[Tuple[int, int], List[int]]] = None,
        fullbody_min_bbox_size: Optional[Union[Tuple[int, int], List[int]]] = None,
        fullbody_target_resolution_order: str = "wh",


        validation_source: str = "first_swap",
        first_swap_dir_filter: str = "from_point_cloud",

        use_head_alignment: bool = True,


        validation_data_root: Optional[str] = None,
        validation_neck_mode: str = "union",
        skip_validation: bool = False,
        skip_missing_validation: bool = False,


        validation_neck_color: Optional[tuple[int, int, int]] = None,
        validation_hair_color: Optional[tuple[int, int, int]] = None,
        validation_color_tolerance: Optional[int] = None,
        validation_dilate_neck: Optional[bool] = None,
        validation_neck_dilation_kernel_size: Optional[int] = None,
        validation_dilate_hair: Optional[bool] = None,
        validation_hair_dilation_kernel_size: Optional[int] = None,


        test_pairs_yaml: Optional[str] = None,
        test_exclude_pairs: Optional[List[Tuple[str, str]]] = None,
        test_include_only_pairs: Optional[List[Tuple[str, str]]] = None,
        test_exclude_pair_ids: Optional[List[Union[int, List[int]]]] = None,
        test_include_only_pair_ids: Optional[List[Union[int, List[int]]]] = None,


        test_output_root: Optional[str] = None,
        test_camera_filter: Optional[List[str]] = None,
        test_save_intermediate_outputs: bool = True,


        **kwargs
    ):


        kwargs['training_mode'] = True


        super().__init__(**kwargs)


        train_has_include = include_only_pairs is not None or include_only_pair_ids is not None
        train_has_exclude = exclude_pairs is not None or exclude_pair_ids is not None

        if train_has_include and train_has_exclude:
            raise ValueError(
                "Training pair filtering: include-only and exclude filters are mutually exclusive. "
                "Use whitelist mode (include_only_pairs/include_only_pair_ids) OR "
                "blacklist mode (exclude_pairs/exclude_pair_ids), not both."
            )

        if include_only_pairs is not None and include_only_pair_ids is not None:
            raise ValueError(
                "include_only_pairs and include_only_pair_ids are mutually exclusive. "
                "Use one whitelist method only."
            )


        val_has_include = val_include_only_pairs is not None or val_include_only_pair_ids is not None
        val_has_exclude = val_exclude_pairs is not None or val_exclude_pair_ids is not None

        if val_has_include and val_has_exclude:
            raise ValueError(
                "Validation pair filtering: include-only and exclude filters are mutually exclusive. "
                "Use whitelist mode (val_include_only_pairs/val_include_only_pair_ids) OR "
                "blacklist mode (val_exclude_pairs/val_exclude_pair_ids), not both."
            )

        if val_include_only_pairs is not None and val_include_only_pair_ids is not None:
            raise ValueError(
                "val_include_only_pairs and val_include_only_pair_ids are mutually exclusive. "
                "Use one whitelist method only."
            )


        self.exclude_pairs = exclude_pairs
        self.include_only_pairs = include_only_pairs
        self.exclude_pair_ids = exclude_pair_ids
        self.include_only_pair_ids = include_only_pair_ids
        self.val_exclude_pairs = val_exclude_pairs
        self.val_include_only_pairs = val_include_only_pairs
        self.val_exclude_pair_ids = val_exclude_pair_ids
        self.val_include_only_pair_ids = val_include_only_pair_ids
        self.strict_mode = strict_mode

        self.training_neck_mode = training_neck_mode
        self.validation_source = validation_source
        self.first_swap_dir_filter = first_swap_dir_filter
        self.use_head_alignment = bool(use_head_alignment)


        if train_pairs_yaml is not None:

            logger.info("Multi-pair mode: Loading pairs from YAML files")
            from ..utils.yaml_utils import load_subject_pairs_from_yaml

            self.train_pairs, self.train_pair_to_id = load_subject_pairs_from_yaml(
                train_pairs_yaml,
                exclude_pairs=exclude_pairs,
                include_only_pairs=include_only_pairs,
                exclude_pair_ids=exclude_pair_ids,
                include_only_pair_ids=include_only_pair_ids,
            )
            logger.info(f"Training: {len(self.train_pairs)} pairs")

            if not skip_validation and val_pairs_yaml is not None:

                self.val_pairs, self.val_pair_to_id = load_subject_pairs_from_yaml(
                    val_pairs_yaml,
                    exclude_pairs=val_exclude_pairs,
                    include_only_pairs=val_include_only_pairs,
                    exclude_pair_ids=val_exclude_pair_ids,
                    include_only_pair_ids=val_include_only_pair_ids,
                )
                logger.info(f"Validation: {len(self.val_pairs)} pairs")
            else:
                self.val_pairs = None
                self.val_pair_to_id = {}


            if test_pairs_yaml is not None:
                self.test_pairs, self.test_pair_to_id = load_subject_pairs_from_yaml(
                    test_pairs_yaml,
                    exclude_pairs=test_exclude_pairs,
                    include_only_pairs=test_include_only_pairs,
                    exclude_pair_ids=test_exclude_pair_ids,
                    include_only_pair_ids=test_include_only_pair_ids,
                )
                logger.info(f"Test: {len(self.test_pairs)} pairs")
            else:
                self.test_pairs = None
                self.test_pair_to_id = {}


            self.subject_a = None
            self.subject_b = None

        else:

            logger.info("Single-pair mode: Using subject_a and subject_b")

            if subject_a is None or subject_b is None:
                raise ValueError(
                    "Must provide either (train_pairs_yaml, val_pairs_yaml) "
                    "or (subject_a, subject_b)"
                )

            self.train_pairs = [(subject_a, subject_b)]
            self.val_pairs = [(subject_a, subject_b)] if not skip_validation else None


            self.test_pairs = [(subject_a, subject_b)]
            self.test_pair_to_id = {(subject_a, subject_b): 0}
            self.train_pair_to_id = {(subject_a, subject_b): 0}
            self.val_pair_to_id = {(subject_a, subject_b): 0}
            self.subject_a = subject_a
            self.subject_b = subject_b

            logger.info(f"Single pair: ({subject_a}, {subject_b})")


        self.combined_data_root = combined_data_root
        self.neck_color = neck_color
        self.hair_color = hair_color
        self.color_tolerance = color_tolerance
        self.dilate_neck = dilate_neck
        self.neck_dilation_kernel_size = neck_dilation_kernel_size
        self.dilate_hair = dilate_hair
        self.hair_dilation_kernel_size = hair_dilation_kernel_size
        self.target_resolution = target_resolution


        self.training_neck_mode = training_neck_mode


        self.both_mode_swapped_ratio = both_mode_swapped_ratio
        self.both_mode_random_seed = both_mode_random_seed
        self.both_mode_max_oversampling = both_mode_max_oversampling


        self.enable_rotation_aug = enable_rotation_aug
        self.rotation_degrees = rotation_degrees
        self.rotation_prob = rotation_prob


        self.refinement_mode = refinement_mode
        self.gt_dataset_root = gt_dataset_root
        self.fullbody_target_resolution = fullbody_target_resolution
        self.fullbody_min_bbox_size = fullbody_min_bbox_size
        self.fullbody_target_resolution_order = fullbody_target_resolution_order


        self.validation_source = validation_source
        self.first_swap_dir_filter = first_swap_dir_filter


        self.validation_data_root = validation_data_root
        self.validation_neck_mode = validation_neck_mode
        self.skip_validation = skip_validation
        self.skip_missing_validation = skip_missing_validation


        self.test_pairs_yaml = test_pairs_yaml
        self.test_exclude_pairs = test_exclude_pairs
        self.test_include_only_pairs = test_include_only_pairs
        self.test_exclude_pair_ids = test_exclude_pair_ids
        self.test_include_only_pair_ids = test_include_only_pair_ids


        self.test_output_root = test_output_root
        self.test_camera_filter = test_camera_filter
        self.test_save_intermediate_outputs = test_save_intermediate_outputs


        self.validation_neck_color = validation_neck_color or neck_color
        self.validation_hair_color = validation_hair_color or hair_color
        self.validation_color_tolerance = validation_color_tolerance if validation_color_tolerance is not None else color_tolerance
        self.validation_dilate_neck = validation_dilate_neck if validation_dilate_neck is not None else dilate_neck
        self.validation_neck_dilation_kernel_size = validation_neck_dilation_kernel_size or neck_dilation_kernel_size
        self.validation_dilate_hair = validation_dilate_hair if validation_dilate_hair is not None else dilate_hair
        self.validation_hair_dilation_kernel_size = validation_hair_dilation_kernel_size or hair_dilation_kernel_size

        logger.info(f"PartialRefinementDataModule initialized")
        logger.info(f"Combined data root: {combined_data_root}")
        logger.info(f"Refinement mode: {refinement_mode}")
        if refinement_mode == "full":
            if self.fullbody_target_resolution is not None:
                logger.info(f"  Full-body mode: adaptive resolution ({self.fullbody_target_resolution[0]}×{self.fullbody_target_resolution[1]}), all-ones mask")
            else:
                logger.info(f"  Full-body mode: adaptive resolution (448×896), all-ones mask")
            logger.info(f"  GT dataset root: {gt_dataset_root}")
        else:
            logger.info(f"  Partial mode: simple resize ({target_resolution}×{target_resolution}), neck+hair mask")
            logger.info(f"  Neck color: {neck_color}, Hair color: {hair_color}")
        logger.info(f"Strict mode: {strict_mode}")
        if train_pairs_yaml is not None:
            logger.info(f"Multi-pair mode: {len(self.train_pairs)} training pairs")
            if self.val_pairs:
                logger.info(f"Multi-pair mode: {len(self.val_pairs)} validation pairs")
        else:
            logger.info(f"Single-pair mode: ({subject_a}, {subject_b})")
        logger.info(f"Skip validation: {skip_validation}")
        if not skip_validation and validation_data_root:
            logger.info(f"Validation data root: {validation_data_root}")
            logger.info(f"Validation neck mode: {validation_neck_mode}")
            logger.info(f"Validation neck color: {self.validation_neck_color}, Hair color: {self.validation_hair_color}")
            logger.info(f"Validation dilation: neck={self.validation_dilate_neck} (ks={self.validation_neck_dilation_kernel_size}), "
                       f"hair={self.validation_dilate_hair} (ks={self.validation_hair_dilation_kernel_size})")

    def setup(self, stage: Optional[str] = None) -> None:

        if (stage == "fit" or stage is None) and self.training_mode:
            self._setup_partial_refinement_training_dataset()

            if not self.skip_validation:
                self._setup_partial_refinement_validation_dataset()
            else:
                logger.info("Validation skipped as configured")
                self.val_dataset = None
        else:

            super().setup(stage)

    def _setup_partial_refinement_training_dataset(self) -> None:

        logger.info("Setting up partial refinement training dataset...")

        try:
            self.train_dataset = PartialRefinementDataset(
                data_root=self.combined_data_root,
                prompts=self.prompts,
                neck_color=self.neck_color,
                hair_color=self.hair_color,
                color_tolerance=self.color_tolerance,
                dilate_neck=self.dilate_neck,
                neck_dilation_kernel_size=self.neck_dilation_kernel_size,
                dilate_hair=self.dilate_hair,
                hair_dilation_kernel_size=self.hair_dilation_kernel_size,
                validate_data_existence=getattr(self, 'validate_gt_existence', True),


                subject_pairs=self.train_pairs,
                strict_mode=self.strict_mode,

                target_resolution=self.target_resolution,


                training_neck_mode=self.training_neck_mode,


                both_mode_swapped_ratio=self.both_mode_swapped_ratio,
                both_mode_random_seed=self.both_mode_random_seed,
                both_mode_max_oversampling=self.both_mode_max_oversampling,


                enable_rotation_aug=self.enable_rotation_aug,
                rotation_degrees=self.rotation_degrees,
                rotation_prob=self.rotation_prob,


                refinement_mode=self.refinement_mode,
                gt_dataset_root=self.gt_dataset_root,
            )

            logger.info(f"Training dataset created: {len(self.train_dataset)} samples")
            logger.info(f"  From {len(self.train_pairs)} subject pairs")

        except Exception as e:
            logger.error(f"Failed to create partial refinement training dataset: {e}")
            raise

    def _setup_partial_refinement_validation_dataset(self) -> None:

        if not self.validation_data_root:
            logger.warning("validation_data_root not provided, skipping validation setup")
            self.val_dataset = None
            return

        if self.val_pairs is None:
            logger.info("No validation pairs specified, skipping validation setup")
            self.val_dataset = None
            return

        logger.info("Setting up partial refinement validation dataset...")

        try:
            self.val_dataset = PartialRefinementValidationDataset(
                data_root=self.validation_data_root,


                subject_pairs=self.val_pairs,

                neck_mode=self.validation_neck_mode,


                refinement_mode=self.refinement_mode,
                fullbody_target_resolution=self.fullbody_target_resolution,
                fullbody_min_bbox_size=self.fullbody_min_bbox_size,
                fullbody_target_resolution_order=self.fullbody_target_resolution_order,


                validation_source=self.validation_source,
                first_swap_dir_filter=self.first_swap_dir_filter,
                use_head_alignment=self.use_head_alignment,


                neck_color=self.validation_neck_color,
                hair_color=self.validation_hair_color,
                color_tolerance=self.validation_color_tolerance,
                dilate_neck=self.validation_dilate_neck,
                neck_dilation_kernel_size=self.validation_neck_dilation_kernel_size,
                dilate_hair=self.validation_dilate_hair,
                hair_dilation_kernel_size=self.validation_hair_dilation_kernel_size,

                target_resolution=self.target_resolution,
                skip_missing=self.skip_missing_validation,
            )

            logger.info(f"Validation dataset created: {len(self.val_dataset)} samples")
            logger.info(f"  From {len(self.val_pairs)} subject pairs")

        except Exception as e:
            logger.error(f"Failed to create partial refinement validation dataset: {e}")
            raise

    def train_dataloader(self) -> DataLoader:

        if not hasattr(self, 'train_dataset') or self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

    def val_dataloader(self) -> Optional[DataLoader]:

        if self.skip_validation or not hasattr(self, 'val_dataset') or self.val_dataset is None:
            logger.info("Validation skipped or not initialized")
            return None

        logger.info(f"Creating validation dataloader with {len(self.val_dataset)} samples")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> Optional[DataLoader]:

        if self.test_pairs is None or not self.validation_data_root:
            logger.warning("Skipping test dataloader (no test_pairs or validation_data_root)")
            return None

        logger.info("Setting up partial refinement test dataset...")


        test_dataset = PartialRefinementValidationDataset(
            data_root=self.validation_data_root,
            subject_pairs=self.test_pairs,
            neck_mode=self.validation_neck_mode,
            refinement_mode=self.refinement_mode,
            test_output_root=self.test_output_root,
            is_test=True,
            fullbody_target_resolution=self.fullbody_target_resolution,
            fullbody_min_bbox_size=self.fullbody_min_bbox_size,
            fullbody_target_resolution_order=self.fullbody_target_resolution_order,
            validation_source=self.validation_source,
            first_swap_dir_filter=self.first_swap_dir_filter,
            use_head_alignment=self.use_head_alignment,
            neck_color=self.validation_neck_color,
            hair_color=self.validation_hair_color,
            color_tolerance=self.validation_color_tolerance,
            dilate_neck=self.validation_dilate_neck,
            neck_dilation_kernel_size=self.validation_neck_dilation_kernel_size,
            dilate_hair=self.validation_dilate_hair,
            hair_dilation_kernel_size=self.validation_hair_dilation_kernel_size,
            target_resolution=self.target_resolution,
            skip_missing=self.skip_missing_validation
        )

        logger.info(f"Test dataset created with {len(test_dataset)} samples from {len(self.test_pairs)} pairs")

        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def get_partial_refinement_info(self) -> Dict[str, Any]:

        return {
            "combined_data_root": self.combined_data_root,
            "neck_color": self.neck_color,
            "hair_color": self.hair_color,
            "color_tolerance": self.color_tolerance,
            "dilate_neck": self.dilate_neck,
            "neck_dilation_kernel_size": self.neck_dilation_kernel_size,
            "dilate_hair": self.dilate_hair,
            "hair_dilation_kernel_size": self.hair_dilation_kernel_size,
            "skip_validation": self.skip_validation,
            "training_dataset_initialized": hasattr(self, 'train_dataset') and self.train_dataset is not None,
            "training_dataset_size": len(self.train_dataset) if hasattr(self, 'train_dataset') and self.train_dataset else 0,
            "validation_dataset_initialized": hasattr(self, 'val_dataset') and self.val_dataset is not None,
        }

    @classmethod
    def from_config(cls, config: dict | DictConfig) -> "PartialRefinementDataModule":

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        return cls(**config)
