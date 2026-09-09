from typing import Any, Dict, List, Optional
from pathlib import Path
import torch
import numpy as np
import cv2
import wandb
from loguru import logger

from .difix3d_module import DiFix3DModule
from ..utils.restoration import restore_portrait_to_fullbody
from ..data.difix3d_datamodule import apply_reverse_adaptive_resolution_transform

class PartialRefinementModule(DiFix3DModule):


    def __init__(
        self,

        log_training_images_freq: float = 5.0,
        max_training_images: int = 4,
        log_training_error_images: bool = True,
        **kwargs
    ):

        super().__init__(**kwargs)


        if log_training_images_freq < 0:
            raise ValueError(
                f"log_training_images_freq must be >= 0, got {log_training_images_freq}"
            )


        self.log_training_images_freq = log_training_images_freq
        self.max_training_images = max_training_images
        self.log_training_error_images = log_training_error_images
        self.kwargs = kwargs


        self._last_logged_epoch = -1
        self._last_logged_step = -1


        self._log_interval_steps = None
        self._is_fractional_logging = (
            0 < log_training_images_freq < 1.0
        )


        self._accumulated_images = {
            'degraded': [],
            'refined': [],
            'assembled': [],
            'gt': [],
            'masks': [],
            'metadata': []
        }
        self._accumulated_count = 0
        self._is_logging_epoch = False
        self._accumulation_complete = False

        logger.info("PartialRefinementModule initialized for neck+hair refinement")
        logger.info(
            f"Training image logging: freq={log_training_images_freq}, "
            f"max_images={max_training_images}, error_images={log_training_error_images}, "
            f"mode={'fractional_epoch' if self._is_fractional_logging else 'epoch'}"
        )

    def _reset_image_accumulation(self) -> None:

        self._accumulated_images = {
            'degraded': [],
            'refined': [],
            'assembled': [],
            'gt': [],
            'masks': [],
            'metadata': []
        }
        self._accumulated_count = 0
        self._accumulation_complete = False

    def _accumulate_training_images(
        self,
        degraded: torch.Tensor,
        refined: torch.Tensor,
        assembled: torch.Tensor,
        gt: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[List[Dict]] = None
    ) -> None:


        if self._accumulation_complete:
            return

        batch_size = degraded.shape[0]


        remaining = self.max_training_images - self._accumulated_count
        num_to_add = min(batch_size, remaining)


        self._accumulated_images['degraded'].append(degraded[:num_to_add].cpu())
        self._accumulated_images['refined'].append(refined[:num_to_add].cpu())
        self._accumulated_images['assembled'].append(assembled[:num_to_add].cpu())
        self._accumulated_images['gt'].append(gt[:num_to_add].cpu())
        self._accumulated_images['masks'].append(masks[:num_to_add].cpu())


        if metadata:
            self._accumulated_images['metadata'].extend(metadata)

        self._accumulated_count += num_to_add


        if self._accumulated_count >= self.max_training_images:
            self._accumulation_complete = True
            logger.debug(
                f"Accumulated {self._accumulated_count} images, "
                f"stopping accumulation for epoch {self.current_epoch}"
            )

    def _log_accumulated_images(self) -> None:

        if self._accumulated_count == 0:
            return

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            logger.warning("Logger not available, skipping training image logging")
            return

        try:

            degraded = torch.cat(self._accumulated_images['degraded'], dim=0).to(self.device)
            refined = torch.cat(self._accumulated_images['refined'], dim=0).to(self.device)
            assembled = torch.cat(self._accumulated_images['assembled'], dim=0).to(self.device)
            gt = torch.cat(self._accumulated_images['gt'], dim=0).to(self.device)
            masks = torch.cat(self._accumulated_images['masks'], dim=0).to(self.device)


            metadata = self._accumulated_images['metadata'] if self._accumulated_images['metadata'] else None


            self._log_training_step_images(
                degraded=degraded,
                refined=refined,
                assembled=assembled,
                gt=gt,
                masks=masks,
                metadata=metadata
            )


            if self._is_fractional_logging:
                self._last_logged_step = self.global_step
                logger.debug(
                    f"Logged images at step {self.global_step} "
                    f"(fractional mode, freq={self.log_training_images_freq})"
                )
            else:
                self._last_logged_epoch = self.current_epoch
                logger.debug(
                    f"Logged images at epoch {self.current_epoch} "
                    f"(epoch mode, freq={self.log_training_images_freq})"
                )


            self._reset_image_accumulation()

        except Exception as e:
            logger.warning(f"Failed to log accumulated images: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _tensor_to_heatmap_wandb_image(self, tensor: torch.Tensor, caption: str) -> wandb.Image:


        if tensor.dim() == 3:

            tensor = tensor.mean(dim=0)

        error_map = tensor.detach().cpu().numpy()


        error_map_normalized = (error_map * 255).astype(np.uint8)


        heatmap_bgr = cv2.applyColorMap(error_map_normalized, cv2.COLORMAP_JET)


        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

        return wandb.Image(heatmap_rgb, caption=caption)

    def _log_training_step_images(
        self,
        degraded: torch.Tensor,
        refined: torch.Tensor,
        assembled: torch.Tensor,
        gt: torch.Tensor,
        masks: torch.Tensor,
        metadata: Optional[Dict] = None
    ) -> None:


        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            logger.warning("Logger not available, skipping training image logging")
            return

        try:

            batch_size = min(degraded.shape[0], self.max_training_images)


            all_images = []

            for i in range(batch_size):

                all_images.append(
                    self.tensor_to_wandb_image(degraded[i], f"epoch{self.current_epoch}_sample{i}_degraded")
                )
                all_images.append(
                    self.tensor_to_wandb_image(refined[i], f"epoch{self.current_epoch}_sample{i}_prediction")
                )
                all_images.append(
                    self.tensor_to_wandb_image(assembled[i], f"epoch{self.current_epoch}_sample{i}_assembled")
                )


                all_images.append(
                    self.tensor_to_wandb_image(gt[i], f"epoch{self.current_epoch}_sample{i}_gt")
                )


                if self.log_training_error_images:

                    error_pred = torch.abs(refined[i] - gt[i])
                    all_images.append(
                        self._tensor_to_heatmap_wandb_image(error_pred, f"epoch{self.current_epoch}_sample{i}_error_pred")
                    )


                    error_assembled = torch.abs(assembled[i] - gt[i])
                    all_images.append(
                        self._tensor_to_heatmap_wandb_image(error_assembled, f"epoch{self.current_epoch}_sample{i}_error_assembled")
                    )


            log_dict = {
                "train/training_samples": all_images,
                "trainer/global_step": self.global_step,
                "trainer/current_epoch": self.current_epoch
            }


            self.logger.experiment.log(log_dict)

            logger.info(
                f"Logged {batch_size} training samples at epoch {self.current_epoch} "
                f"(step {self.global_step})"
            )


            if isinstance(metadata, dict) and metadata.get("reverse_metadata") is not None:
                logger.info("Adaptive resolution detected, logging full-resolution images...")
                self._log_fullres_images(
                    refined=refined,
                    assembled=assembled,
                    gt=gt,
                    metadata=metadata,
                    batch_size=batch_size
                )


            self._last_logged_epoch = self.current_epoch

        except Exception as e:
            logger.warning(f"Failed to log training step images: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def _log_fullres_images(
        self,
        refined: torch.Tensor,
        assembled: torch.Tensor,
        gt: torch.Tensor,
        metadata: Dict,
        batch_size: int
    ) -> None:

        try:
            from torchvision import transforms


            fullres_images = []

            for i in range(batch_size):
                reverse_meta = metadata.get("reverse_metadata")
                if reverse_meta is None:
                    continue


                refined_pil = transforms.ToPILImage()(refined[i].cpu())
                assembled_pil = transforms.ToPILImage()(assembled[i].cpu())
                gt_pil = transforms.ToPILImage()(gt[i].cpu())


                refined_fullres = apply_reverse_adaptive_resolution_transform(
                    refined_pil, reverse_meta
                )
                assembled_fullres = apply_reverse_adaptive_resolution_transform(
                    assembled_pil, reverse_meta
                )
                gt_fullres = apply_reverse_adaptive_resolution_transform(
                    gt_pil, reverse_meta
                )


                fullres_images.append(
                    wandb.Image(refined_fullres, caption=f"epoch{self.current_epoch}_sample{i}_refined_fullres")
                )
                fullres_images.append(
                    wandb.Image(assembled_fullres, caption=f"epoch{self.current_epoch}_sample{i}_assembled_fullres")
                )
                fullres_images.append(
                    wandb.Image(gt_fullres, caption=f"epoch{self.current_epoch}_sample{i}_gt_fullres")
                )


            if fullres_images:
                log_dict = {
                    "train/fullres_samples": fullres_images,
                    "trainer/global_step": self.global_step,
                    "trainer/current_epoch": self.current_epoch
                }
                self.logger.experiment.log(log_dict)
                logger.info(f"Logged {len(fullres_images)//3} full-resolution training samples")

        except Exception as e:
            logger.warning(f"Failed to log full-resolution images: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:


        if not isinstance(batch, dict):
            raise ValueError(f"Expected dict batch format, got {type(batch)}")

        degraded_images = batch["degraded"]
        target_images = batch["target"]
        masks = batch["mask"]
        prompts = batch.get("prompt", self.prompt)


        if isinstance(prompts, str):
            prompts = [prompts] * degraded_images.shape[0]


        degraded_images_tensor = self._validate_and_clean_tensor(degraded_images, "degraded").requires_grad_(True)
        target_images_tensor = self._validate_and_clean_tensor(target_images, "target").requires_grad_(True)
        masks_tensor = self._validate_and_clean_tensor(masks, "mask")


        batch_size = degraded_images_tensor.shape[0]

        result = self.pipe(
            prompt=prompts,
            image=degraded_images_tensor,
            reference_image=None,
            lora_mode=self.lora_training_mode,
            num_inference_steps=self.num_inference_steps,
            timesteps=self.timesteps,
            guidance_scale=self.guidance_scale,
            return_dict=True,
            disable_progress_bar=True,
            output_type="pt",
        )

        refined_images_tensor = result.images
        refined_images_tensor = self._validate_and_clean_tensor(refined_images_tensor, "refined")


        assembled_output = self._assemble_selective_output(
            predicted=refined_images_tensor,
            gt=target_images_tensor,
            mask=masks_tensor
        )


        try:
            loss_dict = self.compute_loss(
                refined_images=assembled_output,
                target_images=target_images_tensor,
                degraded_images=degraded_images_tensor
            )
            total_loss = loss_dict["total_loss"]


            self.log_dict({
                "train_loss": total_loss,
                "train_l2_loss": loss_dict["l2_loss"],
                "train_lpips_loss": loss_dict.get("lpips_loss", 0.0),
                "train_gram_loss": loss_dict.get("gram_loss", 0.0),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr']
            }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)


            self._log_training_metrics(loss_dict, self.global_step)


            if self._is_logging_epoch:

                if self._is_fractional_logging:
                    should_log_now = (
                        self.global_step % self._log_interval_steps == 0 and
                        self.global_step != self._last_logged_step
                    )

                    if should_log_now:

                        self._accumulate_training_images(
                            degraded=degraded_images_tensor,
                            refined=refined_images_tensor,
                            assembled=assembled_output,
                            gt=target_images_tensor,
                            masks=masks_tensor,
                            metadata=batch.get("metadata", {})
                        )


                        if self._accumulation_complete:
                            self._log_accumulated_images()
                            self._last_logged_step = self.global_step

                            self._reset_image_accumulation()


                else:
                    self._accumulate_training_images(
                        degraded=degraded_images_tensor,
                        refined=refined_images_tensor,
                        assembled=assembled_output,
                        gt=target_images_tensor,
                        masks=masks_tensor,
                        metadata=batch.get("metadata", {})
                    )

            return total_loss

        except Exception as e:
            if self.log_training_errors:
                self._log_training_error(e, batch_idx, self.global_step)
            logger.error(f"Training step failed: {e}")
            raise

    def _assemble_selective_output(
        self,
        predicted: torch.Tensor,
        gt: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:


        mask_3ch = mask.repeat(1, 3, 1, 1)


        masked_predicted = predicted * mask_3ch
        masked_gt = gt * (1 - mask_3ch)

        assembled = masked_predicted + masked_gt

        return assembled

    def _validation_step_common(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:

        import cv2
        import numpy as np


        fullbody_combined = batch["fullbody_combined"]
        portrait_combined = batch["portrait_combined"]
        neck_hair_mask = batch["neck_hair_mask"]
        bbox_metadata = batch["bbox_metadata"]
        metadata = batch["metadata"]

        batch_size = portrait_combined.shape[0]


        neck_hair_mask_dilated = neck_hair_mask


        with torch.no_grad():

            result = self.pipe(
                prompt="remove degradation",
                image=portrait_combined,
                reference_image=None,
                lora_mode=self.lora_training_mode,
                num_inference_steps=self.num_inference_steps,
                timesteps=self.timesteps,
                guidance_scale=self.guidance_scale,
                return_dict=True,
                disable_progress_bar=True,
                output_type="pt",
            )
            refined_full = result.images


        assembled_portrait = self._assemble_selective_output(
            predicted=refined_full,
            gt=portrait_combined,
            mask=neck_hair_mask_dilated
        )


        assembled_portrait_list = []
        neck_hair_mask_dilated_list = []
        for i in range(batch_size):

            original_h = int(metadata["original_portrait_height"])
            original_w = int(metadata["original_portrait_width"])


            portrait_np = (assembled_portrait[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            neck_hair_mask_dilated_np = (neck_hair_mask_dilated[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


            portrait_resized_np = cv2.resize(
                portrait_np,
                (original_w, original_h),
                interpolation=cv2.INTER_LINEAR
            )
            neck_hair_mask_dilated_resized_np = cv2.resize(
                neck_hair_mask_dilated_np.squeeze(2),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST
            )


            portrait_resized_tensor = torch.from_numpy(portrait_resized_np).permute(2, 0, 1).float() / 255.0
            assembled_portrait_list.append(portrait_resized_tensor)
            neck_hair_mask_dilated_tensor = torch.from_numpy(neck_hair_mask_dilated_resized_np).unsqueeze(0).float() / 255.0
            neck_hair_mask_dilated_list.append(neck_hair_mask_dilated_tensor)


        assembled_portrait_resized = torch.stack(assembled_portrait_list, dim=0)
        neck_hair_mask_dilated_resized = torch.stack(neck_hair_mask_dilated_list, dim=0)


        fullbody_restored_list = []
        for i in range(batch_size):
            fullbody_restored = restore_portrait_to_fullbody(
                portrait=assembled_portrait_resized[i],
                fullbody_original=fullbody_combined[i],
                bbox_metadata=bbox_metadata
            )
            fullbody_restored_list.append(fullbody_restored)

        fullbody_restored = torch.stack(fullbody_restored_list, dim=0)


        error_map = torch.abs(fullbody_combined - fullbody_restored)
        error_map = error_map.mean(dim=1, keepdim=True)

        return {
            "portrait_combined": portrait_combined,
            "refined_full": refined_full,
            "assembled_portrait": assembled_portrait,
            "assembled_portrait_resized": assembled_portrait_resized,
            "fullbody_combined": fullbody_combined,
            "fullbody_restored": fullbody_restored,
            "error_map": error_map,
            "neck_hair_mask_resized": neck_hair_mask_dilated_resized,
            "metadata": metadata,
            "bbox_metadata": bbox_metadata
        }

    def _validation_step_fullbody(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:


        degraded = batch["degraded"]
        masks = batch["mask"]
        reverse_metadata = batch["reverse_metadata"]


        with torch.no_grad():
            result = self.pipe(
                prompt="remove degradation",
                image=degraded,
                reference_image=None,
                lora_mode=self.lora_training_mode,
                num_inference_steps=self.num_inference_steps,
                timesteps=self.timesteps,
                guidance_scale=self.guidance_scale,
                return_dict=True,
                disable_progress_bar=True,
                output_type="pt",
            )
            refined = result.images


        assembled = refined

        return {
            "degraded": degraded,
            "refined": refined,
            "assembled": assembled,
            "masks": masks,
            "reverse_metadata": reverse_metadata,
        }

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:


        is_fullbody_mode = ("reverse_metadata" in batch and
                           "fullbody_combined" not in batch)

        if is_fullbody_mode:

            outputs = self._validation_step_fullbody(batch, batch_idx)


            self._log_validation_images_fullbody(
                degraded=outputs["degraded"],
                refined=outputs["refined"],
                assembled=outputs["assembled"],
                masks=outputs["masks"],
                reverse_metadata=outputs["reverse_metadata"],
                batch_idx=batch_idx
            )
        else:

            outputs = self._validation_step_common(batch, batch_idx)


            self._log_validation_images(
                portrait_combined=outputs["portrait_combined"],
                refined_full=outputs["refined_full"],
                assembled_portrait=outputs["refined_full"],
                fullbody_combined=outputs["fullbody_combined"],
                fullbody_restored=outputs["fullbody_restored"],
                error_map=outputs["error_map"],
                batch_idx=batch_idx
            )


        return {"val_step": torch.tensor(0.0, device=self.device)}

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:


        if self.kwargs.get("refinement_mode", "partial") == "full":

            outputs = self._test_step_fullbody(batch, batch_idx)
            self._save_fullbody_test_outputs(outputs, batch, batch_idx)
        else:

            outputs = self._validation_step_common(batch, batch_idx)
            self._save_test_outputs_to_files(outputs, batch, batch_idx)


        return {"test_step": torch.tensor(0.0, device=self.device)}

    def _simplify_subject_name(self, verbose_name: str) -> str:

        import re

        pattern = r'(swapped_.+?head_on_.+?body).*'
        match = re.match(pattern, verbose_name)
        if match:
            return match.group(1)
        return verbose_name

    def _filter_cameras(self, all_cameras: List[str], camera_filter: Optional[List[str]]) -> List[str]:

        if camera_filter is None:
            return all_cameras

        return [cam for cam in all_cameras if cam in camera_filter]

    def _copy_mask_from_head_aligned_no_sam(
        self,
        original_swapped_dir: Path,
        camera_name: str,
        output_mask_path: Path
    ) -> bool:

        import shutil
        source_mask = original_swapped_dir / camera_name / "head_aligned_no_sam" / "mask" / "pha" / "0000.png"

        if not source_mask.exists():
            raise FileNotFoundError(
                f"head_aligned_no_sam mask not found: {source_mask}\n"
                f"Please run Stage 26 (GT-aligned head rendering) first!"
            )


        output_mask_path.parent.mkdir(parents=True, exist_ok=True)


        shutil.copy2(source_mask, output_mask_path)

        return True

    def _copy_mask_for_refinement(
        self,
        *,
        original_swapped_dir: Path,
        camera_name: str,
        output_mask_path: Path,
        mask_source: str,
    ) -> bool:

        import shutil

        if mask_source == "standard":
            source_mask = original_swapped_dir / camera_name / "mask" / "pha" / "0000.png"
        elif mask_source == "head_aligned_no_sam":
            source_mask = original_swapped_dir / camera_name / "head_aligned_no_sam" / "mask" / "pha" / "0000.png"
        elif mask_source == "head_aligned":
            source_mask = original_swapped_dir / camera_name / "head_aligned" / "mask" / "pha" / "0000.png"
        else:
            raise ValueError(f"Unknown mask_source: {mask_source}")

        if not source_mask.exists():
            raise FileNotFoundError(
                f"Mask not found for mask_source='{mask_source}': {source_mask}\n"
                f"original_swapped_dir={original_swapped_dir}\n"
                f"camera_name={camera_name}"
            )

        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_mask, output_mask_path)
        return True

    def _save_test_outputs_to_files(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
        batch_idx: int
    ) -> None:

        import cv2
        import numpy as np
        from pathlib import Path


        test_output_root = Path(self.kwargs.get("test_output_root", None))
        camera_filter = self.kwargs.get("test_camera_filter", None)
        save_intermediates = self.kwargs.get("test_save_intermediate_outputs", True)
        use_head_alignment = bool(self.kwargs.get("use_head_alignment", True))
        mask_source = self.kwargs.get("mask_source", None)
        if mask_source is None:
            mask_source = "head_aligned_no_sam" if use_head_alignment else "standard"


        fullbody_restored = outputs["fullbody_restored"]
        portrait_combined = outputs["portrait_combined"]
        refined_full = outputs["refined_full"]
        assembled_portrait_resized = outputs["assembled_portrait_resized"]
        neck_hair_mask_resized = outputs["neck_hair_mask_resized"]
        error_map = outputs["error_map"]
        metadata = outputs["metadata"]
        bbox_metadata = outputs["bbox_metadata"]

        batch_size = fullbody_restored.shape[0]

        for i in range(batch_size):

            original_swapped_dir = Path(metadata["swapped_dir"][i])
            camera_name = metadata["camera_name"][i]


            if camera_filter is not None and camera_name not in camera_filter:
                continue


            verbose_subject_name = original_swapped_dir.name
            simplified_subject_name = self._simplify_subject_name(verbose_subject_name)


            subject_output_dir = test_output_root / simplified_subject_name
            camera_output_dir = subject_output_dir / camera_name
            camera_output_dir.mkdir(parents=True, exist_ok=True)


            rgb_output_path = camera_output_dir / "0000.jpg"
            rgb_np = (fullbody_restored[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_output_path), rgb_bgr)


            mask_output_path = camera_output_dir / "mask" / "pha" / "0000.png"
            try:
                self._copy_mask_for_refinement(
                    original_swapped_dir=original_swapped_dir,
                    camera_name=camera_name,
                    output_mask_path=mask_output_path,
                    mask_source=str(mask_source),
                )
            except FileNotFoundError as e:
                logger.error(str(e))
                raise


            if save_intermediates:
                intermediate_dir = camera_output_dir / "difix_intermediates"
                intermediate_dir.mkdir(parents=True, exist_ok=True)


                portrait_np = (portrait_combined[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                portrait_bgr = cv2.cvtColor(portrait_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(intermediate_dir / "portrait_combined.jpg"), portrait_bgr)


                refined_np = (refined_full[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                refined_bgr = cv2.cvtColor(refined_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(intermediate_dir / "refined_full.jpg"), refined_bgr)


                error_np = (error_map[i, 0].float().cpu().numpy() * 255).astype(np.uint8)
                error_heatmap = cv2.applyColorMap(error_np, cv2.COLORMAP_JET)
                cv2.imwrite(str(intermediate_dir / "error_map.png"), error_heatmap)


                assembled_portrait_np = (assembled_portrait_resized[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                assembled_portrait_bgr = cv2.cvtColor(assembled_portrait_np, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(intermediate_dir / "assembled_portrait.jpg"), assembled_portrait_bgr)


                neck_hair_mask_resized_np = (neck_hair_mask_resized[i, 0].float().cpu().numpy() * 255).astype(np.uint8)
                cv2.imwrite(str(intermediate_dir / "neck_mask.png"), neck_hair_mask_resized_np)


                import json
                restoration_metadata = {
                    "bbox_metadata": {
                        "x": int(bbox_metadata["x"]),
                        "y": int(bbox_metadata["y"]),
                        "size": int(bbox_metadata["size"])
                    },
                    "original_portrait_height": int(metadata["original_portrait_height"][i]),
                    "original_portrait_width": int(metadata["original_portrait_width"][i]),
                    "head_id": metadata["head_id"][i],
                    "body_id": metadata["body_id"][i],
                    "camera_name": camera_name,
                    "swapped_dir": str(original_swapped_dir)
                }
                with open(intermediate_dir / "metadata.json", "w") as f:
                    json.dump(restoration_metadata, f, indent=2)

            logger.info(f"Saved test outputs for {simplified_subject_name}/{camera_name}")


            calibration_output_path = subject_output_dir / "calibration_full.json"
            if not calibration_output_path.exists():
                try:
                    self._save_calibration_for_subject(
                        subject_output_dir=subject_output_dir,
                        original_swapped_dir=original_swapped_dir,
                        metadata=metadata,
                        sample_idx=i
                    )
                except Exception as e:
                    logger.error(f"Failed to save calibration for {simplified_subject_name}: {e}")
                    raise

    def _test_step_fullbody(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:


        degraded = batch["degraded"]
        reverse_metadata = batch["reverse_metadata"]


        restoration_metadata = batch.get("restoration_metadata", None)


        with torch.no_grad():
            result = self.pipe(
                prompt="remove degradation",
                image=degraded,
                reference_image=None,
                lora_mode=self.lora_training_mode,
                num_inference_steps=self.num_inference_steps,
                timesteps=self.timesteps,
                guidance_scale=self.guidance_scale,
                return_dict=True,
                disable_progress_bar=True,
                output_type="pt",
            )
            refined = result.images


        from torchvision import transforms

        refined_fullres_list = []
        for i in range(refined.shape[0]):
            refined_pil = transforms.ToPILImage()(refined[i].float().cpu())
            refined_fullres = apply_reverse_adaptive_resolution_transform(refined_pil, reverse_metadata)
            refined_fullres_list.append(transforms.ToTensor()(refined_fullres))
        refined_fullres = torch.stack(refined_fullres_list, dim=0)


        if restoration_metadata is not None:
            restored_list = []
            for i in range(refined.shape[0]):


                md = batch.get("metadata", {})
                if "camera_dir" in md:
                    camera_dir = Path(md["camera_dir"][i])
                else:
                    camera_dir = Path(md["swapped_dir"][i]) / md["camera_name"][i]

                restored = self._restore_partial_into_fullbody(
                    fullbody_refined=refined_fullres[i],
                    restoration_metadata=restoration_metadata,
                    camera_dir=camera_dir,
                )
                restored_list.append(restored)
            restored = torch.stack(restored_list, dim=0)
        else:


            restored = refined_fullres

        return {
            "degraded": degraded,
            "refined": refined,
            "refined_fullres": refined_fullres,
            "restored": restored,
            "metadata": batch["metadata"],
        }

    def _restore_partial_into_fullbody(
        self,
        fullbody_refined: torch.Tensor,
        restoration_metadata: Dict,
        camera_dir: Path
    ) -> torch.Tensor:

        import cv2
        import numpy as np
        from PIL import Image


        assembled_portrait_path = camera_dir / "difix_intermediates" / "assembled_portrait.jpg"
        assembled_portrait = Image.open(assembled_portrait_path).convert("RGB")
        assembled_portrait_np = np.array(assembled_portrait)


        fullbody_np = (fullbody_refined.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


        bbox = restoration_metadata["bbox_metadata"]
        x, y, size = bbox["x"], bbox["y"], bbox["size"]


        feather_radius = self.kwargs.get("restoration_feather_radius", 0)

        if feather_radius == 0:

            fullbody_np[y:y+size, x:x+size] = assembled_portrait_np
        else:


            mask_u8 = np.ones((size, size), dtype=np.uint8) * 255


            center = (x + size // 2, y + size // 2)


            fullbody_np = cv2.seamlessClone(
                assembled_portrait_np,
                fullbody_np,
                mask_u8,
                (center[0].item(), center[1].item()),
                cv2.MIXED_CLONE
            )


        restored_tensor = torch.from_numpy(fullbody_np).permute(2, 0, 1).float() / 255.0

        return restored_tensor

    def _save_fullbody_test_outputs(self, outputs: Dict, batch: Dict, batch_idx: int):

        import cv2
        import numpy as np
        import shutil
        from pathlib import Path

        restored = outputs["restored"]
        metadata = outputs["metadata"]
        refined = outputs["refined"]
        refined_full = outputs["refined_fullres"]

        for i in range(restored.shape[0]):


            if "camera_dir" in metadata:
                output_camera_dir = Path(metadata["camera_dir"][i])
                output_subject_dir = output_camera_dir.parent
                input_camera_dir = output_camera_dir
                original_swapped_dir = Path(metadata.get("swapped_dir", [output_subject_dir])[i])
            else:
                swapped_dir = Path(metadata["swapped_dir"][i])
                camera_name = metadata["camera_name"][i]
                simplified_subject_name = self._simplify_subject_name(swapped_dir.name)
                test_output_root = Path(self.kwargs.get("test_output_root"))
                output_subject_dir = test_output_root / simplified_subject_name
                output_camera_dir = output_subject_dir / camera_name
                input_camera_dir = swapped_dir / camera_name
                original_swapped_dir = swapped_dir

            output_camera_dir.mkdir(parents=True, exist_ok=True)


            body_fixed_dir = output_camera_dir / "body_fixed"
            body_fixed_dir.mkdir(parents=True, exist_ok=True)


            rgb_output_path = body_fixed_dir / "0000.jpg"
            rgb_np = (restored[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_output_path), rgb_bgr)


            source_mask = input_camera_dir / "mask" / "pha" / "0000.png"
            dest_mask = body_fixed_dir / "mask" / "pha" / "0000.png"
            dest_mask.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_mask, dest_mask)


            intermediate_dir = body_fixed_dir / "difix_intermediates"
            intermediate_dir.mkdir(parents=True, exist_ok=True)
            refined_np = (refined[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            refined_full_np = (refined_full[i].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
            refined_bgr = cv2.cvtColor(refined_np, cv2.COLOR_RGB2BGR)
            refined_full_bgr = cv2.cvtColor(refined_full_np, cv2.COLOR_RGB2BGR)
            refined_dir = intermediate_dir / "refined.jpg"
            refined_full_dir = intermediate_dir / "refined_full.jpg"
            cv2.imwrite(str(refined_dir), refined_bgr)
            cv2.imwrite(str(refined_full_dir), refined_full_bgr)
            logger.info(f"Saved fullbody test output: {body_fixed_dir / '0000.jpg'}")


            calib_out = output_subject_dir / "calibration_full.json"
            if not calib_out.exists():
                try:
                    self._save_calibration_for_subject(
                        subject_output_dir=output_subject_dir,
                        original_swapped_dir=original_swapped_dir,
                        metadata=metadata,
                        sample_idx=i,
                    )
                except Exception as e:
                    logger.error(f"Failed to save calibration for {output_subject_dir.name}: {e}")
                    raise

    def _find_head_transform_npz(
        self,
        head_subject_id: str,
        body_subject_id: str,
        frame_id: int = 0,
        use_no_sam: bool = True
    ) -> Path:


        validation_data_root = Path(self.kwargs.get("validation_data_root", None))
        avatarrex_dir = validation_data_root.parent
        first_swap_dir_filter = self.kwargs.get("first_swap_dir_filter", None)


        repose_base_dir = avatarrex_dir / "gs_on_mesh_repose"


        pattern = f"*{head_subject_id}_to_{body_subject_id}"
        matching_dirs = list(repose_base_dir.glob(pattern))

        if not matching_dirs:
            raise FileNotFoundError(
                f"No reposed directory found matching pattern: {pattern}\n"
                f"Expected in: {repose_base_dir}"
            )

        if len(matching_dirs) > 1:
            logger.warning(f"Multiple reposed directories found, using first: {matching_dirs[0]}")

        reposed_dir = matching_dirs[0]


        npz_suffix = "_no_sam" if use_no_sam else ""
        npz_filename = f"head_transform_avg_frame_{frame_id:04d}{npz_suffix}.npz"
        npz_path = reposed_dir / "aligned_head_assets" / npz_filename

        if not npz_path.exists():
            raise FileNotFoundError(
                f"Head transformation NPZ file not found: {npz_path}\n"
                f"Please run reposing stage with head alignment first!"
            )

        return npz_path

    def _adjust_calibration_with_head_transform(
        self,
        calibration_path: Path,
        head_transform_npz_path: Path,
        output_calibration_path: Path
    ) -> None:

        import json
        import numpy as np


        with open(calibration_path, 'r') as f:
            calib_data = json.load(f)


        npz_data = np.load(head_transform_npz_path)
        avg_head_transform = npz_data['avg_head_transform']

        logger.info(f"Loaded head transformation matrix from: {head_transform_npz_path}")


        for cam_name, cam_data in calib_data.items():
            if cam_name in ['meta_info', 'bounds']:
                continue


            R = np.array(cam_data['R'])
            T = np.array(cam_data['T']).reshape(3, 1)


            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3:4] = T


            new_w2c = w2c @ avg_head_transform


            new_R = new_w2c[:3, :3]
            new_T = new_w2c[:3, 3]


            cam_data['R'] = new_R.tolist()
            cam_data['T'] = new_T.tolist()


        output_calibration_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_calibration_path, 'w') as f:
            json.dump(calib_data, f, indent=2)

        logger.info(f"Saved adjusted calibration to: {output_calibration_path}")

    def _save_calibration_for_subject(
        self,
        subject_output_dir: Path,
        original_swapped_dir: Path,
        metadata: Dict[str, Any],
        sample_idx: int
    ) -> None:


        original_calib = original_swapped_dir / "calibration_full.json"

        if not original_calib.exists():
            logger.warning(f"Original calibration not found: {original_calib}")
            return


        adjust_calibration = bool(self.kwargs.get("adjust_calibration", True))
        output_calib = subject_output_dir / "calibration_full.json"
        if not adjust_calibration:
            import shutil
            output_calib.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_calib, output_calib)
            logger.info(f"Copied calibration without head transform to: {output_calib}")
            return


        import re
        dir_name = original_swapped_dir.name
        match = re.match(r'^swapped_(.+?)head_on_(.+?)body', dir_name)
        if not match:
            logger.error(f"Could not extract subject IDs from directory name: {dir_name}")
            return

        head_subject_id = match.group(1)
        body_subject_id = match.group(2)


        try:
            npz_path = self._find_head_transform_npz(
                head_subject_id=head_subject_id,
                body_subject_id=body_subject_id,
                frame_id=0,
                use_no_sam=True
            )
        except FileNotFoundError as e:
            logger.error(str(e))
            raise


        self._adjust_calibration_with_head_transform(
            calibration_path=original_calib,
            head_transform_npz_path=npz_path,
            output_calibration_path=output_calib
        )

    def _log_validation_images_fullbody(
        self,
        degraded: torch.Tensor,
        refined: torch.Tensor,
        assembled: torch.Tensor,
        masks: torch.Tensor,
        reverse_metadata: List[Dict],
        batch_idx: int
    ) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            logger.warning("Logger not available, skipping validation image logging")
            return

        try:
            from torchvision import transforms

            batch_size = degraded.shape[0]


            adaptive_images = []
            for i in range(batch_size):

                adaptive_images.append(
                    self.tensor_to_wandb_image(
                        degraded[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_degraded_adaptive"
                    )
                )

                adaptive_images.append(
                    self.tensor_to_wandb_image(
                        refined[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_refined_adaptive"
                    )
                )

                adaptive_images.append(
                    self.tensor_to_wandb_image(
                        assembled[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_assembled_adaptive"
                    )
                )

                mask_vis = masks[i].repeat(3, 1, 1)
                adaptive_images.append(
                    self.tensor_to_wandb_image(
                        mask_vis,
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_mask"
                    )
                )

            self.logger.experiment.log({
                "val/fullbody_adaptive": adaptive_images,
                "trainer/global_step": self.global_step
            })


            fullres_images = []
            for i in range(batch_size):
                reverse_meta = reverse_metadata


                refined_pil = transforms.ToPILImage()(refined[i].float().cpu())
                assembled_pil = transforms.ToPILImage()(assembled[i].float().cpu())

                refined_fullres = apply_reverse_adaptive_resolution_transform(
                    refined_pil,
                    reverse_meta
                )
                assembled_fullres = apply_reverse_adaptive_resolution_transform(
                    assembled_pil,
                    reverse_meta
                )


                fullres_images.append(
                    wandb.Image(
                        refined_fullres,
                        caption=f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_refined_fullres"
                    )
                )
                fullres_images.append(
                    wandb.Image(
                        assembled_fullres,
                        caption=f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_assembled_fullres"
                    )
                )

            self.logger.experiment.log({
                "val/fullbody_fullres": fullres_images,
                "trainer/global_step": self.global_step
            })

            logger.debug(f"Logged full-body validation images for batch {batch_idx}")

        except Exception as e:
            logger.error(f"Failed to log full-body validation images: {e}")

    def _log_validation_images(
        self,
        portrait_combined: torch.Tensor,
        refined_full: torch.Tensor,
        assembled_portrait: torch.Tensor,
        fullbody_combined: torch.Tensor,
        fullbody_restored: torch.Tensor,
        error_map: torch.Tensor,
        batch_idx: int
    ) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            logger.warning("Logger not available, skipping validation image logging")
            return

        try:
            batch_size = portrait_combined.shape[0]


            portrait_images = []

            for i in range(batch_size):
                portrait_images.append(
                    self.tensor_to_wandb_image(
                        portrait_combined[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_portrait_input"
                    )
                )
                portrait_images.append(
                    self.tensor_to_wandb_image(
                        refined_full[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_portrait_refined"
                    )
                )
                portrait_images.append(
                    self.tensor_to_wandb_image(
                        assembled_portrait[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_portrait_assembled"
                    )
                )


            fullbody_images = []

            for i in range(batch_size):
                fullbody_images.append(
                    self.tensor_to_wandb_image(
                        fullbody_combined[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_fullbody_before"
                    )
                )
                fullbody_images.append(
                    self.tensor_to_wandb_image(
                        fullbody_restored[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_fullbody_after"
                    )
                )
                fullbody_images.append(
                    self._tensor_to_heatmap_wandb_image(
                        error_map[i],
                        f"val_epoch{self.current_epoch}_b{batch_idx}_s{i}_error_heatmap"
                    )
                )


            log_dict = {
                "val/portrait_panel": portrait_images,
                "val/fullbody_panel": fullbody_images,
            }

            self.logger.experiment.log(log_dict)

            logger.info(
                f"Logged validation images: epoch {self.current_epoch}, batch_idx {batch_idx}, "
                f"samples {batch_size}"
            )

        except Exception as e:
            logger.warning(f"Failed to log validation images: {e}")
            import traceback
            logger.debug(traceback.format_exc())

    def on_train_start(self) -> None:

        super().on_train_start()
        logger.info("Starting partial (neck+hair) refinement training...")

    def on_train_epoch_start(self) -> None:

        super().on_train_epoch_start()


        if self.log_training_images_freq == 0:
            self._is_logging_epoch = False
            return


        if self._is_fractional_logging:

            try:
                train_dataloader = self.trainer.train_dataloader
                steps_per_epoch = len(train_dataloader)


                self._log_interval_steps = max(
                    1,
                    int(steps_per_epoch * self.log_training_images_freq)
                )
                logger.info(f"log_interval_steps: {self._log_interval_steps}")

                self._is_logging_epoch = True
                self._reset_image_accumulation()

                logger.info(
                    f"Epoch {self.current_epoch}: Fractional logging enabled, "
                    f"will log every {self._log_interval_steps} steps "
                    f"(freq={self.log_training_images_freq}, total_steps={steps_per_epoch})"
                )

            except Exception as e:
                logger.warning(
                    f"Failed to calculate step interval for fractional logging: {e}. "
                    f"Falling back to epoch-level logging"
                )
                self._is_fractional_logging = False
                self._is_logging_epoch = False


        else:
            should_log = (
                self.current_epoch % int(self.log_training_images_freq) == 0 and
                self.current_epoch != self._last_logged_epoch
            )

            self._is_logging_epoch = should_log

            if should_log:
                self._reset_image_accumulation()
                logger.info(
                    f"Epoch {self.current_epoch}: Epoch-level logging enabled, "
                    f"will accumulate up to {self.max_training_images} samples"
                )

    def on_train_epoch_end(self) -> None:

        super().on_train_epoch_end()


        if self._is_fractional_logging:
            if self._accumulated_count > 0:
                self._log_accumulated_images()
                logger.info(
                    f"Logged {self._accumulated_count} remaining images at epoch end "
                    f"(fractional logging mode)"
                )


        elif self._is_logging_epoch and self._accumulated_count > 0:
            self._log_accumulated_images()
            logger.info(
                f"Logged {self._accumulated_count} accumulated images at epoch end"
            )

    def on_train_end(self) -> None:

        super().on_train_end()
        logger.info("Partial refinement training completed!")
