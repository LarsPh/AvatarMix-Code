import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf

from lightning_diffusers.utils import pylogger

log = pylogger.get_pylogger(__name__)


class ConfigValidationError(Exception):


    def __init__(self, message: str, suggestions: Optional[List[str]] = None):
        super().__init__(message)
        self.suggestions = suggestions or []


def validate_experiment_config(cfg: DictConfig) -> None:

    try:
        _validate_data_config(cfg.get("data"))
        _validate_model_config(cfg.get("model"))
        _validate_trainer_config(cfg.get("trainer"))
        _validate_paths_config(cfg)
        _validate_system_requirements(cfg)
        log.info("Configuration validation passed!")

    except ConfigValidationError:
        raise
    except Exception as e:
        raise ConfigValidationError(
            f"Unexpected validation error: {str(e)}",
            suggestions=[
                "Check your configuration files for syntax errors",
                "Ensure all required fields are present",
                "Try running with a known-good configuration first"
            ]
        )


def _validate_data_config(data_cfg: Optional[DictConfig]) -> None:

    if not data_cfg:
        raise ConfigValidationError(
            "Data configuration is missing!",
            suggestions=[
                "Add 'data: <config_name>' to your experiment config",
                "Available data configs: mnist, pokemon_caption, lambdalabs_naruto",
                "Example: experiment=mnist_ddpm (includes data config)"
            ]
        )


    if "_target_" not in data_cfg:
        raise ConfigValidationError(
            "Data configuration missing '_target_' field!",
            suggestions=[
                "Add '_target_: path.to.DataModule' to your data config",
                "Example: '_target_: lightning_diffusers.data.MnistDataModule'"
            ]
        )


    target = data_cfg.get("_target_", "")

    if "TextImageDataModule" in target:
        _validate_text_image_data_config(data_cfg)
    elif "MnistDataModule" in target:
        _validate_mnist_data_config(data_cfg)


def _validate_text_image_data_config(data_cfg: DictConfig) -> None:

    dataset_name = data_cfg.get("dataset_name")
    train_data_dir = data_cfg.get("train_data_dir")

    if not dataset_name and not train_data_dir:
        raise ConfigValidationError(
            "Text-image data config requires either 'dataset_name' or 'train_data_dir'!",
            suggestions=[
                "For HuggingFace datasets: set 'dataset_name: \"lambdalabs/pokemon-blip-captions\"'",
                "For local datasets: set 'train_data_dir: \"/path/to/your/dataset\"'",
                "Check available datasets at https://huggingface.co/datasets"
            ]
        )

    if dataset_name and dataset_name.startswith("lambdalabs/"):
        log.warning(
            f"Dataset '{dataset_name}' may require authentication. "
            "If you get authentication errors, run: huggingface-cli login"
        )


def _validate_mnist_data_config(data_cfg: DictConfig) -> None:


    batch_size = data_cfg.get("batch_size", 32)
    if batch_size <= 0:
        raise ConfigValidationError(
            f"Invalid batch_size: {batch_size}. Must be positive integer.",
            suggestions=["Try batch_size: 32 or 64 for MNIST training"]
        )


def _validate_model_config(model_cfg: Optional[DictConfig]) -> None:

    if not model_cfg:
        raise ConfigValidationError(
            "Model configuration is missing!",
            suggestions=[
                "Add 'model: <config_name>' to your experiment config",
                "Available model configs: ddpm, lora_sd15, lora_sd21, lora_sdxl",
                "Example: experiment=mnist_ddpm (includes model config)"
            ]
        )

    if "_target_" not in model_cfg:
        raise ConfigValidationError(
            "Model configuration missing '_target_' field!",
            suggestions=[
                "Add '_target_: path.to.LightningModule' to your model config",
                "Example: '_target_: lightning_diffusers.models.MnistDDPMModule'"
            ]
        )

    target = model_cfg.get("_target_", "")

    if "LoRASDModule" in target:
        _validate_lora_model_config(model_cfg)


def _validate_lora_model_config(model_cfg: DictConfig) -> None:


    model_path = model_cfg.get("pretrained_model_name_or_path")
    if not model_path:
        raise ConfigValidationError(
            "LoRA model config missing 'pretrained_model_name_or_path'!",
            suggestions=[
                "Add: pretrained_model_name_or_path: \"runwayml/stable-diffusion-v1-5\"",
                "Other options: \"stabilityai/stable-diffusion-2-1\", \"stabilityai/stable-diffusion-xl-base-1.0\""
            ]
        )


    lora_config = model_cfg.get("lora_config", {})
    if lora_config:
        rank = lora_config.get("r", 4)
        if rank <= 0 or rank > 128:
            log.warning(f"LoRA rank {rank} may be suboptimal. Recommended range: 4-64")

        alpha = lora_config.get("lora_alpha", rank)
        if alpha != rank:
            log.info(f"LoRA alpha ({alpha}) != rank ({rank}). This affects learning rate scaling.")


def _validate_trainer_config(trainer_cfg: Optional[DictConfig]) -> None:

    if not trainer_cfg:
        raise ConfigValidationError(
            "Trainer configuration is missing!",
            suggestions=[
                "Add 'trainer: default' to your experiment config",
                "Available trainer configs: default, debug"
            ]
        )


    max_epochs = trainer_cfg.get("max_epochs")
    max_steps = trainer_cfg.get("max_steps", -1)

    if max_epochs is None and max_steps <= 0:
        raise ConfigValidationError(
            "Trainer must specify either 'max_epochs' or 'max_steps'!",
            suggestions=[
                "Add: max_epochs: 10 (for epoch-based training)",
                "Or: max_steps: 1000 (for step-based training)",
                "For debugging: max_steps: 10"
            ]
        )


    precision = trainer_cfg.get("precision")
    if precision and precision not in [16, 32, 64, "16-mixed", "bf16-mixed"]:
        log.warning(f"Unusual precision setting: {precision}. Common values: 32, '16-mixed'")


def _validate_paths_config(cfg: DictConfig) -> None:


    try:
        import tempfile
        from pathlib import Path


        output_dir = Path.cwd() / "outputs"
        output_dir.mkdir(exist_ok=True)

        with tempfile.NamedTemporaryFile(dir=output_dir, delete=True):
            pass

    except PermissionError:
        raise ConfigValidationError(
            "Cannot write to output directory!",
            suggestions=[
                "Check write permissions for the current directory",
                "Try running from a directory where you have write access",
                "Consider setting a custom output path in your config"
            ]
        )


def _validate_system_requirements(cfg: DictConfig) -> None:


    if sys.version_info < (3, 11):
        log.warning(f"Python {sys.version} detected. Recommended: Python 3.11+")


    try:
        import torch
        log.info(f"PyTorch {torch.__version__} detected")


        model_cfg = cfg.get("model", {})
        if "LoRASDModule" in str(model_cfg.get("_target_", "")):
            if not torch.cuda.is_available():
                log.warning(
                    "CUDA not available! LoRA training will be very slow on CPU. "
                    "Consider using experiment=lora_debug for testing."
                )
            else:
                gpu_count = torch.cuda.device_count()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                log.info(f"CUDA available: {gpu_count} GPU(s), {gpu_memory:.1f}GB memory")

                if gpu_memory < 8.0:
                    log.warning(
                        f"GPU memory ({gpu_memory:.1f}GB) may be insufficient for LoRA training. "
                        "Consider using gradient_checkpointing=true or smaller batch sizes."
                    )

    except ImportError as e:
        raise ConfigValidationError(
            f"PyTorch not properly installed: {e}",
            suggestions=[
                "Install PyTorch: pip install torch torchvision",
                "Or with CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118",
                "Check installation guide: https://pytorch.org/get-started/"
            ]
        )


    missing_deps = []
    try:
        import diffusers
    except ImportError:
        missing_deps.append("diffusers")

    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")

    try:
        import lightning
    except ImportError:
        missing_deps.append("lightning")

    if missing_deps:
        raise ConfigValidationError(
            f"Missing required dependencies: {', '.join(missing_deps)}",
            suggestions=[
                "Run: uv sync (to install all dependencies)",
                "Or: pip install " + " ".join(missing_deps)
            ]
        )


def get_helpful_error_message(error: Exception, cfg: Optional[DictConfig] = None) -> str:

    error_msg = str(error)
    suggestions = []


    if "DatasetNotFoundError" in str(type(error)):
        suggestions.extend([
            "Check if the dataset name is correct",
            "For private datasets, run: huggingface-cli login",
            "Try a different dataset or use local data with train_data_dir",
            "Available public datasets: lambdalabs/pokemon-blip-captions"
        ])


    elif "CUDA out of memory" in error_msg:
        suggestions.extend([
            "Reduce batch_size in your data config",
            "Enable gradient checkpointing: model.gradient_checkpointing=true",
            "Use mixed precision: trainer.precision='16-mixed'",
            "Try a smaller model or LoRA rank"
        ])


    elif "OSError" in str(type(error)) and ("model" in error_msg.lower() or "checkpoint" in error_msg.lower()):
        suggestions.extend([
            "Check if the model name/path is correct",
            "Ensure you have internet connection for downloading models",
            "Try a different model variant",
            "Check HuggingFace model hub for available models"
        ])


    elif "MissingConfigException" in str(type(error)):
        suggestions.extend([
            "Check if all required config files exist",
            "Verify config names match the file names (without .yaml)",
            "Try using an existing experiment: experiment=mnist_ddpm",
            "List available configs in the configs/ directory"
        ])


    elif "ConfigComposition" in str(type(error)):
        suggestions.extend([
            "Check the 'defaults' section in your experiment config",
            "Ensure all referenced configs exist",
            "Try simplifying the configuration first",
            "Use override syntax: experiment=mnist_ddpm data.batch_size=32"
        ])


    formatted_msg = f"\n{'='*60}\n"
    formatted_msg += f"❌ ERROR: {error_msg}\n"
    formatted_msg += f"{'='*60}\n"

    if suggestions:
        formatted_msg += "\n💡 SUGGESTIONS:\n"
        for i, suggestion in enumerate(suggestions, 1):
            formatted_msg += f"   {i}. {suggestion}\n"

    formatted_msg += f"\n🔧 For more help:\n"
    formatted_msg += f"   - Check the README.md for examples\n"
    formatted_msg += f"   - Try running: python -m lightning_diffusers.check\n"
    formatted_msg += f"   - Use debug mode: experiment=lora_debug\n"
    formatted_msg += f"{'='*60}\n"

    return formatted_msg
