import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import lightning as L
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig, OmegaConf
from loguru import logger

from .identity_preserving_dataset import IdentityPreservingDataset
from .difix3d_datamodule import DiFix3DDataModule
from ..utils.camera_selection import CameraSelector


class IdentityPreservingDataModule(DiFix3DDataModule):


    def __init__(
        self,

        head_mask_dir: Optional[str] = None,
        camera_config: Optional[Dict[str, Any]] = None,
        head_crop_resolution: int = 224,
        validation_modes: Optional[List[str]] = None,
        max_identity_validation_samples: int = 50,
        use_no_reshape_data: bool = False,
        alignment_mode: str = "standard",


        gt_data_dir: Optional[str] = None,
        image_size: Optional[int] = 512,
        normalize: Optional[bool] = True,

        **kwargs
    ):


        if gt_data_dir is not None and 'gt_dataset_root' not in kwargs:
            kwargs['gt_dataset_root'] = gt_data_dir
            logger.warning("Using deprecated parameter 'gt_data_dir'. Please use 'gt_dataset_root' instead.")


        kwargs['training_mode'] = True


        super().__init__(**kwargs)


        self.head_mask_dir = head_mask_dir
        self.camera_config = camera_config or {}
        self.head_crop_resolution = head_crop_resolution
        self.validation_modes = validation_modes or ['normal']
        self.max_identity_validation_samples = max_identity_validation_samples
        self.use_no_reshape_data = use_no_reshape_data
        self.alignment_mode = alignment_mode
        self.image_size = image_size
        self.normalize = normalize


        self._validate_gt_aligned_config()


        self.identity_validation_datasets = None

        logger.info(f"Adaptive resolution enabled: {getattr(self, 'adaptive_resolution_enabled', False)}")
        logger.info(f"Target resolution: {getattr(self, 'target_resolution', (448, 896))}")

        logger.info(f"IdentityPreservingDataModule initialized with validation modes: {self.validation_modes}")
        if head_mask_dir:
            logger.info(f"Head mask directory: {head_mask_dir}")

    def setup(self, stage: Optional[str] = None) -> None:


        super().setup(stage)


        if (stage == "fit" or stage is None) and self.training_mode:
            self._setup_identity_training_dataset()
            self._setup_identity_validation_datasets()

    def _setup_identity_validation_datasets(self) -> None:

        logger.info("Setting up unified identity validation datasets...")


        if 'identity' not in self.validation_modes:
            logger.info("Identity validation mode not enabled - skipping identity dataset creation")
            return

        if not (self.subject_a and self.subject_b):
            logger.warning("Identity validation requires both subject_a and subject_b to be configured")
            return

        logger.info(f"Setting up unified identity validation for subjects: {self.subject_a}, {self.subject_b}")

        try:

            camera_selector = self._create_camera_selector()


            normal_validation_dataset = IdentityPreservingDataset(
                degraded_data_root=self.degraded_dir,
                gt_data_root=self.gt_dataset_root,
                first_swapped_data_root=self.swapped_validation_dir,
                camera_selector=camera_selector,
                prompts=self.prompts,
                head_crop_resolution=self.head_crop_resolution,
                adaptive_resolution_enabled=getattr(self, 'adaptive_resolution_enabled', False),
                target_resolution=getattr(self, 'target_resolution', (448, 896)),
                min_bbox_size=getattr(self, 'min_bbox_size', (64, 64)),
                subject_a=self.subject_a,
                subject_b=self.subject_b,
                validate_data_existence=getattr(self, 'validate_gt_existence', True),
                min_mask_area=getattr(self, 'min_mask_area', 1000),
                use_no_reshape_data=self.use_no_reshape_data,
                alignment_mode=self.alignment_mode,
            )
            normal_validation_dataset.set_mode("normal")


            identity_validation_dataset = IdentityPreservingDataset(
                degraded_data_root=self.degraded_dir,
                gt_data_root=self.gt_dataset_root,
                first_swapped_data_root=self.swapped_validation_dir,
                camera_selector=camera_selector,
                prompts=self.prompts,
                head_crop_resolution=self.head_crop_resolution,
                adaptive_resolution_enabled=getattr(self, 'adaptive_resolution_enabled', False),
                target_resolution=getattr(self, 'target_resolution', (448, 896)),
                min_bbox_size=getattr(self, 'min_bbox_size', (64, 64)),
                subject_a=self.subject_a,
                subject_b=self.subject_b,
                validate_data_existence=getattr(self, 'validate_gt_existence', True),
                min_mask_area=getattr(self, 'min_mask_area', 1000),
                use_no_reshape_data=self.use_no_reshape_data,
                alignment_mode=self.alignment_mode,
            )
            identity_validation_dataset.set_mode("identity")


            self.identity_validation_datasets = {
                'normal': normal_validation_dataset,
                'identity': identity_validation_dataset
            }

            logger.info(f"Unified validation datasets created:")
            logger.info(f"  Normal validation: {len(normal_validation_dataset)} samples (twice-swapped)")
            logger.info(f"  Identity validation: {len(identity_validation_dataset)} samples (first-swapped)")

        except Exception as e:
            logger.error(f"Failed to create unified identity validation datasets: {e}")
            self.identity_validation_datasets = None

    def _setup_identity_training_dataset(self) -> None:

        logger.info("Setting up dual identity training datasets...")

        if not hasattr(self, 'train_pairs') or not self.train_pairs:
            logger.warning("No train_pairs found - identity training datasets cannot be created")
            return


        subject_info = self._extract_subject_ids_from_pairs()

        if not subject_info:
            logger.warning("Could not extract subject information from train_pairs")
            return

        gt_subject_id, swapped_subject_id = subject_info
        logger.info(f"Extracted subjects: GT={gt_subject_id}, Swapped={swapped_subject_id}")


        camera_selector = self._create_camera_selector()


        try:

            refinement_dataset = IdentityPreservingDataset(
                degraded_data_root=self.degraded_dir,
                gt_data_root=self.gt_dataset_root,
                first_swapped_data_root=self.swapped_validation_dir,
                camera_selector=camera_selector,
                prompts=self.prompts,
                head_crop_resolution=self.head_crop_resolution,
                adaptive_resolution_enabled=getattr(self, 'adaptive_resolution_enabled', False),
                target_resolution=getattr(self, 'target_resolution', (448, 896)),
                min_bbox_size=getattr(self, 'min_bbox_size', (64, 64)),
                subject_a=self.subject_a,
                subject_b=self.subject_b,
                validate_data_existence=getattr(self, 'validate_gt_existence', True),
                min_mask_area=getattr(self, 'min_mask_area', 1000),
                use_no_reshape_data=self.use_no_reshape_data,
                alignment_mode=self.alignment_mode,
            )
            refinement_dataset.set_mode("normal")


            identity_dataset = IdentityPreservingDataset(
                degraded_data_root=self.degraded_dir,
                gt_data_root=self.gt_dataset_root,
                first_swapped_data_root=self.swapped_validation_dir,
                camera_selector=camera_selector,
                prompts=self.prompts,
                head_crop_resolution=self.head_crop_resolution,
                adaptive_resolution_enabled=getattr(self, 'adaptive_resolution_enabled', False),
                target_resolution=getattr(self, 'target_resolution', (448, 896)),
                min_bbox_size=getattr(self, 'min_bbox_size', (64, 64)),
                subject_a=self.subject_a,
                subject_b=self.subject_b,
                validate_data_existence=getattr(self, 'validate_gt_existence', True),
                min_mask_area=getattr(self, 'min_mask_area', 1000),
                use_no_reshape_data=self.use_no_reshape_data,
                alignment_mode=self.alignment_mode,
            )
            identity_dataset.set_mode("identity")


            self.identity_training_datasets = {
                'refinement': refinement_dataset,
                'identity': identity_dataset
            }


            self.train_dataset = refinement_dataset

            logger.info(f"Dual training datasets created:")
            logger.info(f"  Refinement: {len(refinement_dataset)} samples (twice-swapped)")
            logger.info(f"  Identity: {len(identity_dataset)} samples (first-swapped)")

        except Exception as e:
            logger.error(f"Failed to create dual training datasets: {e}")
            logger.warning("Falling back to parent's basic training dataset")
            self.identity_training_datasets = None

    def _extract_subject_ids_from_pairs(self) -> Optional[Tuple[str, str]]:

        try:

            if not self.train_pairs:
                return None

            degraded_path, gt_path = self.train_pairs[0]
            logger.debug(f"Analyzing paths: degraded='{degraded_path}', gt='{gt_path}'")


            gt_path_obj = Path(gt_path)
            gt_subject_id = None


            for part in gt_path_obj.parts:
                if part.isdigit() and len(part) == 4:
                    gt_subject_id = part
                    break
                elif part.startswith(tuple('0123456789')) and '_' not in part and len(part) >= 3:
                    gt_subject_id = part
                    break


            degraded_path_obj = Path(degraded_path)
            swapped_subject_id = None


            for part in degraded_path_obj.parts:
                if 'restored_' in part:

                    import re

                    match = re.search(r'from_refined_(swapped_\d+head_on_\d+body)', part)
                    if match:
                        swapped_subject_id = match.group(1)
                        break

                    fallback_match = re.search(r'restored_(\d+)_from', part)
                    if fallback_match:
                        swapped_subject_id = fallback_match.group(1)
                        break
                elif part.isdigit() and len(part) == 4:
                    swapped_subject_id = part
                    break

            if gt_subject_id and swapped_subject_id:
                return (gt_subject_id, swapped_subject_id)
            else:
                logger.warning(f"Could not extract subject IDs: gt='{gt_subject_id}', swapped='{swapped_subject_id}'")
                return None

        except Exception as e:
            logger.error(f"Error extracting subject IDs: {e}")
            return None

    def _validate_gt_aligned_config(self) -> None:

        if self.alignment_mode not in ["standard", "gt_aligned"]:
            raise ValueError(f"Invalid alignment_mode: {self.alignment_mode}. Must be 'standard' or 'gt_aligned'")

        if self.alignment_mode == "gt_aligned":
            camera_config = self.camera_config or {}


            required_aligned_fields = [
                'forward_view_a', 'forward_view_b',
                'camera_range_a', 'camera_range_b',
                'filtered_pairs_a', 'filtered_pairs_b'
            ]


            missing_fields = [field for field in required_aligned_fields if field not in camera_config]
            if missing_fields:
                raise ValueError(f"GT-aligned mode requires per-subject camera configuration. Missing fields: {missing_fields}")

            logger.info(f"GT-aligned configuration validated successfully")
            logger.info(f"  Subject A camera forward view: {camera_config['forward_view_a']} (same for GT and swapped)")
            logger.info(f"  Subject B camera forward view: {camera_config['forward_view_b']} (same for GT and swapped)")
            logger.info(f"  Subject A camera range: {camera_config['camera_range_a']}")
            logger.info(f"  Subject B camera range: {camera_config['camera_range_b']}")

    def _create_camera_selector(self) -> CameraSelector:

        camera_config = self.camera_config or {}

        if self.alignment_mode == "gt_aligned":


            forward_view_a = camera_config["forward_view_a"]
            forward_view_b = camera_config["forward_view_b"]
            camera_range_a = tuple(camera_config["camera_range_a"])
            camera_range_b = tuple(camera_config["camera_range_b"])
            filtered_pairs_a = camera_config.get("filtered_pairs_a", [])
            filtered_pairs_b = camera_config.get("filtered_pairs_b", [])


            return CameraSelector(
                forward_view_gt_a=forward_view_a,
                forward_view_swapped_a=forward_view_a,
                forward_view_gt_b=forward_view_b,
                forward_view_swapped_b=forward_view_b,
                camera_range_a=camera_range_a,
                camera_range_b=camera_range_b,
                filtered_pairs_a=filtered_pairs_a,
                filtered_pairs_b=filtered_pairs_b
            )
        else:

            if all(key in camera_config for key in ["forward_view_gt_a", "forward_view_swapped_a", "forward_view_gt_b", "forward_view_swapped_b"]):

                return CameraSelector(
                    forward_view_gt_a=camera_config["forward_view_gt_a"],
                    forward_view_swapped_a=camera_config["forward_view_swapped_a"],
                    forward_view_gt_b=camera_config["forward_view_gt_b"],
                    forward_view_swapped_b=camera_config["forward_view_swapped_b"],
                    camera_range_a=tuple(camera_config.get("camera_range_a", [2, 3])),
                    camera_range_b=tuple(camera_config.get("camera_range_b", [2, 3])),
                    filtered_pairs_a=camera_config.get("filtered_pairs_a", []),
                    filtered_pairs_b=camera_config.get("filtered_pairs_b", [])
                )
            else:
                raise ValueError("Full camera configuration must be provided for identity training")

    def _determine_subject_types(self, gt_subject_id: str, swapped_subject_id: str) -> Tuple[str, str]:


        subjects = sorted([gt_subject_id, swapped_subject_id])

        if gt_subject_id == subjects[0]:
            gt_type = 'a'
        else:
            gt_type = 'b'

        if swapped_subject_id == subjects[0]:
            swapped_type = 'a'
        else:
            swapped_type = 'b'

        logger.debug(f"Subject type mapping: {gt_subject_id} -> {gt_type}, {swapped_subject_id} -> {swapped_type}")

        return gt_type, swapped_type

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:


        base_dataloaders = super().val_dataloader()


        if 'identity' not in self.validation_modes or self.identity_validation_datasets is None:
            return base_dataloaders


        if not isinstance(base_dataloaders, list):
            base_dataloaders = [base_dataloaders]


        normal_dataset = self.identity_validation_datasets['normal']
        identity_dataset = self.identity_validation_datasets['identity']

        normal_loader = DataLoader(
            normal_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

        identity_loader = DataLoader(
            identity_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )


        identity_dataloaders = [normal_loader, identity_loader]

        logger.info(f"Returning {len(identity_dataloaders)} unified identity validation dataloaders:")
        logger.info(f"  dataloader_idx=0: Normal validation ({len(normal_dataset)} samples, twice-swapped)")
        logger.info(f"  dataloader_idx=1: Identity validation ({len(identity_dataset)} samples, first-swapped)")
        return identity_dataloaders

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader]]:


        if not hasattr(self, 'identity_training_datasets') or not self.identity_training_datasets:
            logger.info("Dual training datasets not available, using single dataloader")
            return super().train_dataloader()


        refinement_dataset = self.identity_training_datasets['refinement']
        identity_dataset = self.identity_training_datasets['identity']


        refinement_loader = DataLoader(
            refinement_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

        identity_loader = DataLoader(
            identity_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            drop_last=True
        )

        logger.info(f"Returning dual training dataloaders:")
        logger.info(f"  [0] Refinement: {len(refinement_dataset)} samples (twice-swapped)")
        logger.info(f"  [1] Identity: {len(identity_dataset)} samples (first-swapped)")

        return [refinement_loader, identity_loader]

    def get_identity_info(self) -> Dict[str, Any]:

        return {
            "head_mask_dir": self.head_mask_dir,
            "camera_config": self.camera_config,
            "head_crop_resolution": self.head_crop_resolution,
            "validation_modes": self.validation_modes,
            "max_identity_validation_samples": self.max_identity_validation_samples,
            "identity_datasets_initialized": self.identity_validation_datasets is not None,
            "identity_dataset_size": len(self.identity_validation_datasets['identity']) if self.identity_validation_datasets else 0
        }

    @classmethod
    def from_config(cls, config: Union[Dict, DictConfig]) -> "IdentityPreservingDataModule":

        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)

        return cls(**config)
