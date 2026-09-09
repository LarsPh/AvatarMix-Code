import os
from pathlib import Path
from typing import Any, List, Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from PIL import Image

from lightning_diffusers.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DiFix3DPredictionCallback(Callback):


    def __init__(
        self,
        output_format: str = "JPEG",
        quality: int = 95,
        create_dirs: bool = True,
        log_every_n_batches: Optional[int] = 10,
        overwrite: bool = True,
    ):
        super().__init__()
        self.output_format = output_format.upper()
        self.quality = quality if self.output_format == "JPEG" else None
        self.create_dirs = create_dirs
        self.log_every_n_batches = log_every_n_batches
        self.overwrite = overwrite


        self.images_saved = 0
        self.images_failed = 0
        self.batch_count = 0

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str) -> None:

        if stage == "predict":
            log.info("DiFix3D Prediction Callback initialized for image saving")
            self.images_saved = 0
            self.images_failed = 0
            self.batch_count = 0

    def on_predict_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if not trainer.is_global_zero:
            return

        self.batch_count += 1


        if outputs is None:
            log.warning(f"No outputs received for batch {batch_idx}")
            return


        if isinstance(outputs, dict) and "error" in outputs:
            log.error(f"Batch {batch_idx} failed with error: {outputs['error']}")
            self.images_failed += 1
            return

        try:

            refined_images = outputs.get("refined_images", [])
            metadata_list = outputs.get("metadata", [])

            if not refined_images:
                log.warning(f"No refined images in batch {batch_idx}")
                return


            if len(metadata_list) != len(refined_images):
                log.warning(f"Metadata count ({len(metadata_list)}) doesn't match image count ({len(refined_images)}) for batch {batch_idx}")

                while len(metadata_list) < len(refined_images):
                    metadata_list.append({})


            for i, (image, metadata) in enumerate(zip(refined_images, metadata_list)):
                try:
                    output_path = metadata.get("output_path")
                    if not output_path:
                        log.warning(f"No output path for image {i} in batch {batch_idx}")
                        log.debug(f"Available metadata keys: {list(metadata.keys())}")
                        self.images_failed += 1
                        continue


                    log.debug(f"Processing image {i} from batch {batch_idx}")
                    log.debug(f"Input path: {metadata.get('input_path', 'Unknown')}")
                    log.debug(f"Output path: {output_path}")


                    if os.path.exists(output_path) and not self.overwrite:
                        log.debug(f"Skipping existing file: {output_path}")
                        continue


                    output_dir = os.path.dirname(output_path)
                    if self.create_dirs:
                        os.makedirs(output_dir, exist_ok=True)
                        log.debug(f"Created output directory: {output_dir}")


                    if hasattr(image, 'save'):
                        pil_image = image
                    else:

                        log.error(f"Unsupported image type: {type(image)} for {output_path}")
                        self.images_failed += 1
                        continue


                    save_kwargs = {}
                    if self.output_format == "JPEG" and self.quality is not None:
                        save_kwargs["quality"] = self.quality
                        save_kwargs["optimize"] = True

                    pil_image.save(output_path, format=self.output_format, **save_kwargs)
                    self.images_saved += 1


                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        log.debug(f"Successfully saved refined image: {output_path} ({file_size} bytes)")
                    else:
                        log.error(f"File was not created despite successful save call: {output_path}")
                        self.images_failed += 1

                except Exception as e:
                    log.error(f"Failed to save image {i} from batch {batch_idx}: {e}")
                    self.images_failed += 1

        except Exception as e:
            log.error(f"Error processing batch {batch_idx}: {e}")
            self.images_failed += 1


        if (self.log_every_n_batches is not None and
            self.batch_count % self.log_every_n_batches == 0):
            log.info(f"Progress: {self.images_saved} images saved, {self.images_failed} failed (batch {self.batch_count})")

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        if trainer.is_global_zero:
            total_images = self.images_saved + self.images_failed
            success_rate = (self.images_saved / total_images * 100) if total_images > 0 else 0

            log.info("=" * 50)
            log.info("DiFix3D+ Batch Refinement Complete!")
            log.info(f"Images successfully saved: {self.images_saved}")
            log.info(f"Images failed: {self.images_failed}")
            log.info(f"Success rate: {success_rate:.1f}%")
            log.info(f"Total batches processed: {self.batch_count}")
            log.info("=" * 50)
