"""
Prediction script using Lightning Trainer.predict() for DiFix3D image refinement.

This script provides the Lightning-based inference interface that replaces
the original inference scripts with proper configuration management.
"""

from typing import List, Optional, Tuple

import hydra
import lightning as L
from lightning import LightningDataModule, LightningModule
from omegaconf import DictConfig

from lightning_diffusers import utils
from lightning_diffusers.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


@utils.task_wrapper
def predict(cfg: DictConfig) -> Tuple[dict, dict]:
    """Predict with model using Lightning Trainer.predict().

    This function is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns where failure of one experiment shouldn't stop others.

    Args:
        cfg: Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[L.pytorch.loggers.Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "callbacks": callbacks,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting prediction!")
    predictions = trainer.predict(model=model, datamodule=datamodule)

    # Collect prediction results
    metric_dict = {"predictions": len(predictions) if predictions else 0}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="predict.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for prediction.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optional[float]: Optimized metric value.
    """
    # Ensure prediction mode
    cfg.train = False
    cfg.test = False

    # Apply extra utilities:
    # - ask for tags if none are provided in multirun (optional)
    # - enforce tags and name for multiruns (optional)
    # - log extra information about the run (optional)
    utils.extras(cfg)

    # Predict with model
    metric_dict, _ = predict(cfg)

    # Safely retrieve metric value for Hydra-based hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()