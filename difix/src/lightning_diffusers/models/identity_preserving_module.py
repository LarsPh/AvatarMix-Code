import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import torch
import torch.nn.functional as F
import lightning as L
from omegaconf import DictConfig
from loguru import logger
import wandb
import numpy as np


from .difix3d_module import DiFix3DModule, DiFix3DConfig, TrainingConfig


from ..utils.head_processing import batch_crop_heads_tensor
from ..losses.id.identity_loss_manager import IdentityLossManager


@dataclass
class IdentityTrainingConfig:


    enabled: bool = True


    normal_steps: int = 2
    identity_steps: int = 1


    model_name_or_path: str = "openai/clip-vit-base-patch32"
    identity_loss_weight: float = 0.1
    similarity_target: float = 0.85


    head_crop_resolution: int = 224
    min_mask_area: int = 1000


    forward_view_gt: str = "100_p+000"
    forward_view_swapped: str = "105_p+000"
    camera_range_gt: List[int] = field(default_factory=lambda: [2, 3])
    camera_range_swapped: List[int] = field(default_factory=lambda: [2, 3])
    filtered_pairs: List[str] = field(default_factory=list)


    identity_validation_enabled: bool = True
    max_identity_validation_samples: int = 50


@dataclass
class EnhancedDiFix3DConfig(DiFix3DConfig):

    identity_training: IdentityTrainingConfig = field(default_factory=IdentityTrainingConfig)


class IdentityPreservingDiFix3DModule(DiFix3DModule):


    def __init__(
        self,

        **base_kwargs
    ):


        super().__init__(**base_kwargs)


        self.identity_config = base_kwargs.get("identity_cfg", {})
        self.identity_enabled = self.identity_config.get("enabled", False)

        if self.identity_enabled:

            self.automatic_optimization = False

            logger.info("Initializing identity-preserving training with manual optimization...")


            self._setup_identity_loss()


            self.head_crop_resolution = self.identity_config.get("head_crop_resolution", 224)


            self.identity_step_count = 0
            self.normal_step_count = 0

            logger.info(f"Identity-preserving training initialized:")
            logger.info(f"  Manual optimization: enabled")
            logger.info(f"  Identity loss available: {self.identity_loss is not None}")
        else:
            logger.info("Identity-preserving training disabled")
            self.identity_loss = None
            self.head_crop_resolution = 224
            self.identity_step_count = 0
            self.normal_step_count = 0


    def _setup_identity_loss(self) -> Optional[IdentityLossManager]:

        try:
            self.identity_loss = IdentityLossManager(self.identity_config)
            self.identity_loss.eval()
            return self.identity_loss

        except Exception as e:
            logger.error(f"Failed to setup identity loss manager: {e}")
            return None

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:

        if not self.identity_enabled:

            return super().training_step(batch, batch_idx)


        opt = self.optimizers()


        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            refinement_batch, identity_batch = batch
        else:

            logger.warning("Expected tuple of (refinement_batch, identity_batch), got single batch. Using same batch for both losses.")
            refinement_batch = identity_batch = batch

        opt.zero_grad()

        refinement_loss = self._compute_refinement_loss(refinement_batch, batch_idx)
        self.manual_backward(refinement_loss)
        self.normal_step_count += 1


        identity_loss = self._compute_identity_loss(identity_batch, batch_idx)
        self.manual_backward(identity_loss)
        self.identity_step_count += 1


        opt.step()


        total_loss = refinement_loss + identity_loss


        log_dict = {
            'train/l2_loss': self._last_refinement_loss_dict.get('l2_loss', 0.0),
            'train/lpips_loss': self._last_refinement_loss_dict.get('lpips_loss', 0.0),
            'train/gram_loss': self._last_refinement_loss_dict.get('gram_loss', 0.0),
            'train/refinement_loss': refinement_loss,
            'train/identity_loss': identity_loss,
            'train/clip_loss': self._last_identity_loss_dict.get('clip_loss', 0.0),
            'train/arcface_loss': self._last_identity_loss_dict.get('arcface_loss', 0.0),
            'train/total_loss': total_loss,
            'train/refinement_step_count': float(self.normal_step_count),
            'train/identity_step_count': float(self.identity_step_count)
        }


        if hasattr(self, '_last_identity_loss_dict'):
            for key, value in self._last_identity_loss_dict.items():
                if key != 'identity_loss' and key != 'l1_error_map' and torch.is_tensor(value):
                    log_dict[f'train/{key}'] = value

        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def _compute_refinement_loss(self, batch: Any, batch_idx: int) -> torch.Tensor:


        if isinstance(batch, dict):
            degraded_images = batch["degraded"]
            target_images = batch["target"]
            reference_images = batch.get("reference", None)
        else:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                degraded_images, target_images = batch[0], batch[1]
                reference_images = batch[2] if len(batch) > 2 else None
            else:
                raise ValueError(f"Unsupported batch format: {type(batch)}")


        degraded_images_tensor = self._validate_and_clean_tensor(degraded_images, "degraded").requires_grad_(True)
        target_images_tensor = self._validate_and_clean_tensor(target_images, "target").requires_grad_(True)
        reference_images_tensor = self._validate_and_clean_tensor(reference_images, "reference").requires_grad_(True) if reference_images is not None else None


        batch_size = degraded_images_tensor.shape[0]

        result = self.pipe(
            prompt=[self.prompt] * batch_size,
            image=degraded_images_tensor,
            reference_image=reference_images_tensor,
            lora_mode="swapfix",
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance_scale=self.guidance_scale,
            return_dict=True,
            disable_progress_bar=True,
            output_type="pt",
        )

        refined_images_tensor = result.images
        refined_images_tensor = self._validate_and_clean_tensor(refined_images_tensor, "refined")


        loss_dict = self.compute_loss(refined_images_tensor, target_images_tensor, degraded_images_tensor)
        refinement_loss = loss_dict["total_loss"]

        self._last_refinement_loss_dict = loss_dict
        return refinement_loss

    def _compute_identity_loss(self, batch: Any, batch_idx: int) -> torch.Tensor:


        if not isinstance(batch, dict):
            raise ValueError("Identity training requires dictionary batch format with head crops")

        degraded_images = batch["degraded"]
        gt_head_crop = batch.get("gt_head_crop")
        reference_images = batch.get("reference", None)
        head_mask = batch.get("head_mask")
        metadata = batch.get("metadata", {})


        if gt_head_crop is None or head_mask is None:
            logger.warning("Missing head crops for identity training, falling back to refinement training")
            return self._compute_refinement_loss(batch, batch_idx)

        if self.identity_loss is None:
            logger.warning("Identity loss not available, falling back to refinement training")
            return self._compute_refinement_loss(batch, batch_idx)


        degraded_images_tensor = self._validate_and_clean_tensor(degraded_images, "degraded").requires_grad_(True)
        reference_images_tensor = self._validate_and_clean_tensor(reference_images, "reference").requires_grad_(True) if reference_images is not None else None


        batch_size = degraded_images_tensor.shape[0]

        result = self.pipe(
            prompt=[self.prompt] * batch_size,
            image=degraded_images_tensor,
            reference_image=reference_images_tensor,
            lora_mode="swapfix",
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance_scale=self.guidance_scale,
            return_dict=True,
            disable_progress_bar=True,
            output_type="pt",
        )

        refined_images_tensor = result.images
        refined_images_tensor = self._validate_and_clean_tensor(refined_images_tensor, "refined")


        identity_loss_dict = self._compute_identity_loss_from_heads(
            refined_images_tensor, gt_head_crop, head_mask, metadata
        )


        self._last_identity_loss_dict = identity_loss_dict


        identity_loss = identity_loss_dict["identity_loss"]

        return identity_loss

    def _compute_identity_loss_from_heads(
        self,
        refined_images: torch.Tensor,
        gt_head_crop: torch.Tensor,
        head_mask: torch.Tensor,
        metadata: Dict
    ) -> Dict[str, torch.Tensor]:

        try:

            refined_images = refined_images.to(self.device)
            gt_head_crop = gt_head_crop.to(self.device)
            head_mask = head_mask.to(self.device)


            refined_head_crops, crop_success_mask = batch_crop_heads_tensor(
                refined_images, head_mask,
                target_resolution=self.head_crop_resolution,
                padding_ratio=0.1
            )


            success_mask = None
            if isinstance(metadata, dict):

                gt_success = metadata.get("gt_head_success", True)
                mask_success = metadata.get("head_mask_success", True)
                crop_success = crop_success_mask[0].item() if len(crop_success_mask) > 0 else True
                success_mask = torch.tensor([gt_success and mask_success and crop_success], device=self.device)
            elif isinstance(metadata, list):

                success_flags = []
                for i, meta in enumerate(metadata):
                    if isinstance(meta, dict):
                        gt_success = meta.get("gt_head_success", True)
                        mask_success = meta.get("head_mask_success", True)
                        crop_success = crop_success_mask[i].item() if i < len(crop_success_mask) else True
                        success_flags.append(gt_success and mask_success and crop_success)
                    else:
                        success_flags.append(True)
                success_mask = torch.tensor(success_flags, device=self.device)


            identity_loss_dict = self.identity_loss.compute_identity_loss(
                refined_heads=refined_head_crops,
                gt_heads=gt_head_crop,
                success_mask=success_mask
            )

            return identity_loss_dict

        except Exception as e:
            logger.error(f"Error computing identity loss from heads: {e}")

            return {
                "identity_loss": torch.zeros(1, device=self.device, requires_grad=True),
                "valid_samples": torch.tensor(0, device=self.device)
            }

    def _log_training_step(
        self,
        loss_dict: Dict[str, torch.Tensor],
        batch_size: int,
        step_type: str
    ) -> None:


        log_dict = {
            "train_loss": loss_dict["total_loss"],
            "train_l2_loss": loss_dict["l2_loss"],
            "train_lpips_loss": loss_dict.get("lpips_loss", 0.0),
            "train_gram_loss": loss_dict.get("gram_loss", 0.0),
            "lr": self.trainer.optimizers[0].param_groups[0]['lr']
        }


        if step_type == "identity":
            log_dict.update({
                "train_identity_loss": loss_dict.get("identity_loss", 0.0),
                "train_clip_similarity": loss_dict.get("clip_similarity", 0.0),
                "train_identity_valid_samples": loss_dict.get("valid_samples", 0)
            })


        log_dict["train_step_type"] = 1.0 if step_type == "identity" else 0.0


        self.log_dict(log_dict, prog_bar=True, on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)


        global_step = self.global_step
        self._log_training_metrics(loss_dict, global_step, step_type=step_type)


    def _log_training_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        global_step: int,
        step_type: Optional[str] = None
    ) -> None:

        try:
            import wandb
            if wandb.run is None:
                return


            log_data = {
                "train/loss_total": loss_dict["total_loss"].item(),
                "train/loss_l2": loss_dict["l2_loss"].item(),
                "train/loss_lpips": loss_dict.get("lpips_loss", torch.tensor(0.0)).item(),
                "train/loss_gram": loss_dict.get("gram_loss", torch.tensor(0.0)).item(),
                "train/global_step": global_step
            }


            if "identity_loss" in loss_dict:
                log_data.update({
                    "train/loss_identity": loss_dict["identity_loss"].item(),
                    "train/clip_similarity": loss_dict.get("clip_similarity", torch.tensor(0.0)).item(),
                    "train/identity_valid_samples": loss_dict.get("valid_samples", torch.tensor(0)).item()
                })


            if step_type:
                log_data["train/step_type"] = step_type


            wandb.log(log_data, step=global_step)

        except Exception as e:
            logger.debug(f"WandB logging failed: {e}")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:

        if not self.identity_enabled:

            return super().validation_step(batch, batch_idx, dataloader_idx)

        if dataloader_idx == 1:

            return self._identity_validation_step(batch, batch_idx)
        else:

            return self._refinement_validation_step(batch, batch_idx)

    def _refinement_validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:

        try:

            if isinstance(batch, dict):
                input_images = batch["degraded"]
                target_images = batch["target"]
                reference_images = batch.get("reference", None)
            else:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    input_images, target_images = batch[0], batch[1]
                    reference_images = batch[2] if len(batch) > 2 else None
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")


            input_images_tensor = self._validate_and_clean_tensor(input_images, "input")
            target_images_tensor = self._validate_and_clean_tensor(target_images, "target") if target_images is not None else None
            reference_images_tensor = self._validate_and_clean_tensor(reference_images, "reference") if reference_images is not None else None

            batch_size = input_images_tensor.shape[0]
            prompts = [self.prompt] * batch_size

            val_metrics = {}


            with torch.no_grad():
                result = self.pipe(
                    prompt=prompts,
                    image=input_images_tensor,
                    reference_image=reference_images_tensor,
                    lora_mode="swapfix",
                    num_inference_steps=self.num_inference_steps,
                    timesteps=self.timesteps,
                    guidance_scale=self.guidance_scale,
                    return_dict=True,
                    disable_progress_bar=True,
                    output_type="pt",
                )

                refined_images_tensor = self._validate_and_clean_tensor(result.images, "refined")


                if target_images_tensor is not None:
                    loss_dict = self.compute_loss(refined_images_tensor, target_images_tensor, input_images_tensor)


                    psnr = self.compute_psnr(refined_images_tensor, target_images_tensor)
                    ssim = self.compute_ssim(refined_images_tensor, target_images_tensor) if hasattr(self, 'compute_ssim') else 0.0


                    val_metrics.update({
                        "val_loss_second_swapped": loss_dict["total_loss"],
                        "val_l2_loss_second_swapped": loss_dict["l2_loss"],
                        "val_lpips_loss_second_swapped": loss_dict.get("lpips_loss", torch.tensor(0.0, device=self.device)),
                        "val_gram_loss_second_swapped": loss_dict.get("gram_loss", torch.tensor(0.0, device=self.device)),
                        "val_psnr_second_swapped": psnr,
                        "val_ssim_second_swapped": torch.tensor(ssim, device=self.device),
                    })


                    batch_dict = {
                        "degraded": input_images_tensor,
                        "target": target_images_tensor
                    }
                    self._log_validation_images(batch_dict, refined_images_tensor, self.global_step)
                else:

                    val_metrics.update({
                        "val_loss_second_swapped": torch.tensor(float('inf'), device=self.device),
                        "val_l2_loss_second_swapped": torch.tensor(0.0, device=self.device),
                        "val_lpips_loss_second_swapped": torch.tensor(0.0, device=self.device),
                        "val_gram_loss_second_swapped": torch.tensor(0.0, device=self.device),
                        "val_psnr_second_swapped": torch.tensor(0.0, device=self.device),
                        "val_ssim_second_swapped": torch.tensor(0.0, device=self.device),
                    })

                return val_metrics

        except Exception as e:
            logger.error(f"Refinement validation failed: {e}")

            return {
                "val_loss_second_swapped": torch.tensor(float('inf'), device=self.device),
                "val_l2_loss_second_swapped": torch.tensor(0.0, device=self.device),
                "val_lpips_loss_second_swapped": torch.tensor(0.0, device=self.device),
                "val_gram_loss_second_swapped": torch.tensor(0.0, device=self.device),
                "val_psnr_second_swapped": torch.tensor(0.0, device=self.device),
                "val_ssim_second_swapped": torch.tensor(0.0, device=self.device),
            }

    def _identity_validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:

        if not isinstance(batch, dict) or self.identity_loss is None:
            return {"val_identity_error_first_swapped": torch.tensor(1.0, device=self.device)}

        try:

            degraded_images = batch["degraded"]
            gt_head_crop = batch["gt_head_crop"]
            head_mask = batch["head_mask"]
            reference_images = batch.get("reference", None)
            metadata = batch.get("metadata", {})


            degraded_images_tensor = self._validate_and_clean_tensor(degraded_images, "degraded")
            reference_images_tensor = self._validate_and_clean_tensor(reference_images, "reference") if reference_images is not None else None


            with torch.no_grad():
                batch_size = degraded_images_tensor.shape[0]

                result = self.pipe(
                    prompt=[self.prompt] * batch_size,
                    image=degraded_images_tensor,
                    reference_image=reference_images_tensor,
                    lora_mode="swapfix",
                    num_inference_steps=self.num_inference_steps,
                    timesteps=self.timesteps,
                    guidance_scale=self.guidance_scale,
                    return_dict=True,
                    disable_progress_bar=True,
                    output_type="pt",
                )

                refined_images_tensor = result.images
                refined_images_tensor = self._validate_and_clean_tensor(refined_images_tensor, "refined")


                identity_loss_dict = self._compute_identity_loss_from_heads(
                    refined_images_tensor, gt_head_crop, head_mask, metadata
                )


                batch_dict = {
                    "degraded": degraded_images_tensor,
                    "refined": refined_images_tensor,
                    "gt_head_crop": gt_head_crop,
                    "head_mask": head_mask
                }
                self._log_identity_validation_images(batch_dict, self.global_step, identity_loss_dict)


            return {
                "val_identity_loss_first_swapped": identity_loss_dict["identity_loss"],
                "val_identity_clip_loss_first_swapped": identity_loss_dict.get("clip_loss", torch.tensor(0.0, device=self.device)),
                "val_identity_arcface_loss_first_swapped": identity_loss_dict.get("arcface_loss", torch.tensor(0.0, device=self.device)),
                "val_identity_l1_loss_first_swapped": identity_loss_dict.get("l1_loss", torch.tensor(0.0, device=self.device)),
                "val_identity_lpips_loss_first_swapped": identity_loss_dict.get("lpips_loss", torch.tensor(0.0, device=self.device)),
            }

        except Exception as e:
            logger.error(f"Identity validation step failed: {e}")
            return {"val_identity_error_first_swapped": torch.tensor(1.0, device=self.device)}

    def _log_identity_validation_images(self, batch_dict: Dict[str, torch.Tensor], global_step: int, identity_loss_dict: Optional[Dict] = None) -> None:

        try:
            import wandb
            if wandb.run is None:
                return


            refinement_images = []
            if "degraded" in batch_dict and "refined" in batch_dict:
                degraded = batch_dict["degraded"]
                refined = batch_dict["refined"]


                max_samples = min(4, degraded.shape[0])
                for i in range(max_samples):

                    degraded_img = self.tensor_to_wandb_image(
                        degraded[i], f"Degraded (First-swapped) - Sample {i+1}"
                    )
                    refinement_images.append(degraded_img)


                    refined_img = self.tensor_to_wandb_image(
                        refined[i], f"Refined (First-swapped) - Sample {i+1}"
                    )
                    refinement_images.append(refined_img)


            head_crop_images = []
            if "gt_head_crop" in batch_dict and "head_mask" in batch_dict and "refined" in batch_dict:
                gt_heads = batch_dict["gt_head_crop"]
                head_masks = batch_dict["head_mask"]
                refined_images = batch_dict["refined"]


                refined_head_crops, _ = batch_crop_heads_tensor(
                    refined_images, head_masks,
                    target_resolution=self.head_crop_resolution,
                    padding_ratio=0.1
                )


                max_samples = min(4, gt_heads.shape[0])
                for i in range(max_samples):

                    gt_head_img = self.tensor_to_wandb_image(
                        gt_heads[i], f"GT Head - Sample {i+1}"
                    )
                    head_crop_images.append(gt_head_img)


                    refined_head_img = self.tensor_to_wandb_image(
                        refined_head_crops[i], f"Refined Head - Sample {i+1}"
                    )
                    head_crop_images.append(refined_head_img)


                    if identity_loss_dict and self.identity_loss:
                        error_heatmaps = self.identity_loss.extract_error_map_data(identity_loss_dict, max_samples=4)
                        if error_heatmaps and i < len(error_heatmaps):
                            error_img = wandb.Image(error_heatmaps[i], caption=f"L1 Error Map - Sample {i+1}")
                            head_crop_images.append(error_img)


            log_data = {}
            if refinement_images:
                log_data["val/first_swapped_images"] = refinement_images
            if head_crop_images:
                log_data["val/head_crops_images"] = head_crop_images

            if log_data:
                wandb.log(log_data)

        except Exception as e:
            logger.debug(f"Identity validation image logging failed: {e}")

    def on_train_epoch_end(self) -> None:

        super().on_train_epoch_end()

        if self.identity_enabled:
            total_steps = self.normal_step_count + self.identity_step_count
            logger.info(f"Epoch {self.current_epoch} dual training summary:")
            logger.info(f"  Refinement steps: {self.normal_step_count}")
            logger.info(f"  Identity steps: {self.identity_step_count}")

    def configure_optimizers(self):

        return super().configure_optimizers()

    def get_identity_training_info(self) -> Dict[str, Any]:

        if not self.identity_enabled:
            return {"enabled": False}

        info = {
            "enabled": True,
            "manual_optimization": True,
            "identity_loss": self.identity_loss.get_loss_info() if self.identity_loss else None,
            "step_counts": {
                "normal": self.normal_step_count,
                "identity": self.identity_step_count,
                "total": self.normal_step_count + self.identity_step_count
            }
        }

        return info
