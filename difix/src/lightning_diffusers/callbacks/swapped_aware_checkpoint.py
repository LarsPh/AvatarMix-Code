from typing import Any, Optional
import torch
from loguru import logger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
import lightning.pytorch as pl


class SwappedAwareModelCheckpoint(ModelCheckpoint):


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_validation_dataloader_idx = None
        logger.info(f"SwappedAwareModelCheckpoint initialized")
        logger.info(f"Monitoring: {self.monitor} | Mode: {self.mode} | Save top k: {self.save_top_k}")

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:


        if hasattr(trainer, 'num_val_batches') and isinstance(trainer.num_val_batches, list):


            current_dataloader_idx = getattr(trainer, 'current_validated_dataloader_idx', None)


            if current_dataloader_idx is None:


                monitor_candidates = self._monitor_candidates(trainer)
                has_training_metrics = self.monitor in monitor_candidates if self.monitor else 'val_loss' in monitor_candidates

                if not has_training_metrics:
                    logger.debug("Skipping checkpoint: No training validation metrics found (likely swapped validation)")
                    return
            else:

                if current_dataloader_idx != 0:
                    logger.debug(f"Skipping checkpoint: dataloader_idx={current_dataloader_idx} (not training validation)")
                    return


        logger.debug("Proceeding with checkpoint logic for training validation")
        super().on_validation_end(trainer, pl_module)

    def _monitor_candidates(self, trainer: "pl.Trainer") -> dict[str, torch.Tensor]:

        return super()._monitor_candidates(trainer)

    def _should_trigger_checkpoint(self, trainer: "pl.Trainer") -> bool:


        monitor_candidates = self._monitor_candidates(trainer)


        if self.monitor:
            has_monitor_metric = self.monitor in monitor_candidates
        else:

            has_monitor_metric = any(
                metric in monitor_candidates
                for metric in ['val_loss', 'val/loss', 'validation_loss']
            )

        if not has_monitor_metric:
            logger.debug(f"Skipping checkpoint: Monitor metric '{self.monitor}' not found in {list(monitor_candidates.keys())}")
            return False


        swapped_prefixes = ['swapped_', 'validation_swapped_']
        training_metrics_count = sum(
            1 for key in monitor_candidates.keys()
            if not any(key.startswith(prefix) for prefix in swapped_prefixes)
        )

        if training_metrics_count == 0:
            logger.debug("Skipping checkpoint: Only swapped validation metrics found")
            return False

        return True
