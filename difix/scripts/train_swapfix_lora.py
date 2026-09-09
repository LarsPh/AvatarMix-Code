#!/usr/bin/env python3


import sys
import os
from pathlib import Path


current_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(current_dir.parent))
sys.path.insert(0, str(current_dir.parent / "src"))
os.environ.setdefault("PROJECT_ROOT", str(current_dir.parent))
difix_src_path = current_dir.parent / "src" / "lightning_diffusers"
sys.path.insert(0, str(difix_src_path))

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
import wandb


from src.lightning_diffusers.models.difix3d_module import DiFix3DModule
from src.lightning_diffusers.models.identity_preserving_module import IdentityPreservingDiFix3DModule
from src.lightning_diffusers.models.partial_refinement_module import PartialRefinementModule
from src.lightning_diffusers.data.difix3d_datamodule import DiFix3DDataModule
from src.lightning_diffusers.data.identity_preserving_datamodule import IdentityPreservingDataModule
from src.lightning_diffusers.data.partial_refinement_datamodule import PartialRefinementDataModule
from src.lightning_diffusers.callbacks import SwappedAwareModelCheckpoint


logger.remove()

log_level = os.environ.get("LOG_LEVEL", "INFO")

logger.add(sys.stderr, level=log_level, format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<green>{function}</green>:<blue>{line}</blue> - <level>{message}</level>")

def setup_identity_preserving_training(cfg: DictConfig):

    logger.info("🎯 Setting up identity-preserving training mode")


    difix3d_datamodule_kwargs = {
        "training_mode": True,
        "degraded_dir": cfg.data.degraded_dir,
        "gt_dataset_root": cfg.data.gt_dataset_root,
        "train_split_ratio": cfg.data.get("train_split_ratio", 0.8),
        "batch_size": cfg.data.get("batch_size", 2),
        "num_workers": cfg.data.get("num_workers", 4),
        "prompts": cfg.data.get("prompts", "remove degradation"),
        "validate_gt_existence": cfg.data.get("validate_gt_existence", True),
        "skip_missing_pairs": cfg.data.get("skip_missing_pairs", True),
        "extract_subject_pattern": cfg.data.get("extract_subject_pattern", r"restored_(\d+)_from_"),
        "reference_strategy": cfg.data.get("reference_strategy", "placeholder"),
        "log_split_statistics": cfg.data.get("log_split_statistics", True),
        "use_same_subjects_for_val": cfg.data.get("use_same_subjects_for_val", True),
        "subject_a": cfg.data.get("subject_a", None),
        "subject_b": cfg.data.get("subject_b", None),
        "adaptive_resolution_enabled": cfg.data.get("adaptive_resolution_enabled", True),
        "swapped_validation_enabled": cfg.data.get("swapped_validation_enabled", True),
        "swapped_validation_dir": cfg.data.get("swapped_validation_dir", None),
        "max_validation_cameras": cfg.data.get("max_validation_cameras", 4),
        "visualize_restored_images": cfg.data.get("visualize_restored_images", False),
        "seed": cfg.get("seed", 42),
    }


    datamodule = IdentityPreservingDataModule(
        head_mask_dir=cfg.data.get("head_mask_dir", None),
        filter_by_subjects=cfg.data.get("filter_by_subjects", False),
        subject_pairs=cfg.data.get("subject_pairs", None),
        head_crop_resolution=cfg.data.get("head_crop_resolution", 224),
        camera_config=cfg.data.get("camera_config", {}),
        validation_modes=cfg.data.get("validation_modes", ["normal"]),
        max_identity_validation_samples=cfg.data.get("max_identity_validation_samples", 50),
        alignment_mode=cfg.data.get("alignment_mode", "standard"),
        image_size=cfg.data.get("image_size", 512),
        normalize=cfg.data.get("normalize", True),
        use_no_reshape_data=cfg.data.get("use_no_reshape_data", False),
        **difix3d_datamodule_kwargs
    )


    training_cfg = cfg.model.get("training", {})
    unet_lora_cfg = training_cfg.get("unet_lora", {})
    swapfix_lora_cfg = training_cfg.get("swapfix_lora", {})
    loss_weights_cfg = training_cfg.get("loss_weights", {})
    optimizer_cfg = training_cfg.get("optimizer", {})
    identity_cfg = training_cfg.get("identity", {})


    model = IdentityPreservingDiFix3DModule(
        training_mode=cfg.model.training_mode,
        model_name=cfg.model.get("model_name", "nvidia/difix"),
        prompt=cfg.model.get("prompt", "remove degradation"),
        num_inference_steps=cfg.model.get("num_inference_steps", 1),
        timesteps=cfg.model.get("timesteps", [199]),
        guidance_scale=cfg.model.get("guidance_scale", 0.0),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        torch_dtype=cfg.model.get("torch_dtype", None),
        unet_training_mode="lora" if unet_lora_cfg.get("enabled", False) else "full",
        unet_lora_rank=unet_lora_cfg.get("rank", 4),
        unet_lora_alpha=unet_lora_cfg.get("alpha", 4.0),
        unet_lora_dropout=unet_lora_cfg.get("dropout", 0.1),
        unet_lora_target_modules=unet_lora_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
        lora_training_mode=training_cfg.get("lora_training_mode", "swapfix"),
        difix_weight=training_cfg.get("difix_weight", 0.5),
        swapfix_weight=training_cfg.get("swapfix_weight", 0.5),
        swapfix_lora_rank=swapfix_lora_cfg.get("rank", 8),
        lora_alpha=swapfix_lora_cfg.get("alpha", 32.0),
        lora_dropout=swapfix_lora_cfg.get("dropout", 0.1),
        target_modules=swapfix_lora_cfg.get("target_modules", [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]),
        l2_weight=loss_weights_cfg.get("l2_weight", 1.0),
        lpips_weight=loss_weights_cfg.get("lpips_weight", 1.0),
        gram_weight=loss_weights_cfg.get("gram_weight", 0.5),
        learning_rate=optimizer_cfg.get("learning_rate", 2e-5),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        use_cosine_scheduler=optimizer_cfg.get("use_cosine_scheduler", True),
        scheduler_eta_min=optimizer_cfg.get("scheduler_eta_min", 1e-6),
        identity_cfg=identity_cfg,
    )

    return datamodule, model


def setup_partial_refinement_training(cfg: DictConfig):

    logger.info("🎨 Setting up partial refinement training mode")


    partial_cfg = cfg.model.get("training", {}).get("partial_refinement", {})


    validation_cfg = cfg.model.get("validation", {}).get("partial_refinement", {})


    datamodule = PartialRefinementDataModule(

        combined_data_root=cfg.data.get("combined_data_root"),


        train_pairs_yaml=cfg.data.get("train_pairs_yaml"),
        val_pairs_yaml=cfg.data.get("val_pairs_yaml"),


        exclude_pairs=cfg.data.get("exclude_pairs"),
        include_only_pairs=cfg.data.get("include_only_pairs"),
        exclude_pair_ids=cfg.data.get("exclude_pair_ids"),
        include_only_pair_ids=cfg.data.get("include_only_pair_ids"),
        val_include_only_pair_ids=cfg.data.get("val_include_only_pair_ids"),
        val_exclude_pair_ids=cfg.data.get("val_exclude_pair_ids"),
        val_exclude_pairs=cfg.data.get("val_exclude_pairs"),
        val_include_only_pairs=cfg.data.get("val_include_only_pairs"),


        strict_mode=cfg.data.get("strict_mode", True),


        training_neck_mode=cfg.data.get("training_neck_mode", "gt"),


        both_mode_swapped_ratio=cfg.data.get("both_mode_swapped_ratio", 0.5),
        both_mode_random_seed=cfg.data.get("both_mode_random_seed", None),
        both_mode_max_oversampling=cfg.data.get("both_mode_max_oversampling", 3.0),


        enable_rotation_aug=cfg.data.get("enable_rotation_aug", True),
        rotation_degrees=tuple(cfg.data.get("rotation_degrees", [-30, 30])),
        rotation_prob=cfg.data.get("rotation_prob", 0.8),


        refinement_mode=cfg.data.get("refinement_mode", "partial"),
        gt_dataset_root=cfg.data.get("gt_dataset_root"),
        fullbody_target_resolution=cfg.data.get("fullbody_target_resolution", None),
        fullbody_min_bbox_size=cfg.data.get("fullbody_min_bbox_size", None),
        fullbody_target_resolution_order=cfg.data.get("fullbody_target_resolution_order", "wh"),


        neck_color=tuple(partial_cfg.get("neck_color", [85, 51, 0])),
        hair_color=tuple(partial_cfg.get("hair_color", [255, 0, 0])),
        color_tolerance=partial_cfg.get("color_tolerance", 5),
        dilate_neck=partial_cfg.get("dilate_neck", True),
        neck_dilation_kernel_size=partial_cfg.get("neck_dilation_kernel_size", 7),
        dilate_hair=partial_cfg.get("dilate_hair", False),
        hair_dilation_kernel_size=partial_cfg.get("hair_dilation_kernel_size", 3),


        subject_a=cfg.data.get("subject_a"),
        subject_b=cfg.data.get("subject_b"),


        target_resolution=cfg.data.get("target_resolution", 512),


        validation_data_root=cfg.data.get("validation_data_root"),
        validation_neck_mode=validation_cfg.get("neck_mode", "union"),
        skip_validation=cfg.data.get("skip_validation", validation_cfg.get("skip_validation", False)),
        skip_missing_validation=validation_cfg.get("skip_missing_validation", False),


        validation_source=cfg.data.get("validation_source", "combined_portraits"),
        first_swap_dir_filter=cfg.data.get("first_swap_dir_filter", "from_point_cloud"),

        use_head_alignment=cfg.data.get("use_head_alignment", True),


        validation_neck_color=tuple(validation_cfg.get("neck_color", [85, 51, 0])) if validation_cfg.get("neck_color") else None,
        validation_hair_color=tuple(validation_cfg.get("hair_color", [255, 0, 0])) if validation_cfg.get("hair_color") else None,
        validation_color_tolerance=validation_cfg.get("color_tolerance"),
        validation_dilate_neck=validation_cfg.get("dilate_neck"),
        validation_neck_dilation_kernel_size=validation_cfg.get("neck_dilation_kernel_size"),
        validation_dilate_hair=validation_cfg.get("dilate_hair"),
        validation_hair_dilation_kernel_size=validation_cfg.get("hair_dilation_kernel_size"),


        test_pairs_yaml=cfg.data.get("test_pairs_yaml"),
        test_exclude_pairs=cfg.data.get("test_exclude_pairs"),
        test_include_only_pairs=cfg.data.get("test_include_only_pairs"),
        test_exclude_pair_ids=cfg.data.get("test_exclude_pair_ids"),
        test_include_only_pair_ids=cfg.data.get("test_include_only_pair_ids"),
        test_output_root=cfg.data.get("test_output_root"),
        test_camera_filter=cfg.data.get("test_camera_filter"),
        test_save_intermediate_outputs=cfg.data.get("test_save_intermediate_outputs", True),


        batch_size=cfg.data.get("batch_size", 2),
        num_workers=cfg.data.get("num_workers", 4),
        prompts=cfg.data.get("prompts", "remove degradation"),
        validate_gt_existence=cfg.data.get("validate_gt_existence", True),
        seed=cfg.get("seed", 42),


        training_mode=True,
    )


    training_cfg = cfg.model.get("training", {})
    unet_lora_cfg = training_cfg.get("unet_lora", {})
    swapfix_lora_cfg = training_cfg.get("swapfix_lora", {})
    loss_weights_cfg = training_cfg.get("loss_weights", {})
    optimizer_cfg = training_cfg.get("optimizer", {})


    model = PartialRefinementModule(
        training_mode=cfg.model.training_mode,
        model_name=cfg.model.get("model_name", "nvidia/difix"),
        prompt=cfg.model.get("prompt", "remove degradation"),
        num_inference_steps=cfg.model.get("num_inference_steps", 1),
        timesteps=cfg.model.get("timesteps", [199]),
        guidance_scale=cfg.model.get("guidance_scale", 0.0),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        torch_dtype=cfg.model.get("torch_dtype", None),
        unet_training_mode="lora" if unet_lora_cfg.get("enabled", False) else "full",
        unet_lora_rank=unet_lora_cfg.get("rank", 4),
        unet_lora_alpha=unet_lora_cfg.get("alpha", 4.0),
        unet_lora_dropout=unet_lora_cfg.get("dropout", 0.1),
        unet_lora_target_modules=unet_lora_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
        lora_training_mode=training_cfg.get("lora_training_mode", "swapfix"),
        difix_weight=training_cfg.get("difix_weight", 0.5),
        swapfix_weight=training_cfg.get("swapfix_weight", 0.5),
        swapfix_lora_rank=swapfix_lora_cfg.get("rank", 8),
        lora_alpha=swapfix_lora_cfg.get("alpha", 32.0),
        lora_dropout=swapfix_lora_cfg.get("dropout", 0.1),
        target_modules=swapfix_lora_cfg.get("target_modules", [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]),
        l2_weight=loss_weights_cfg.get("l2_weight", 1.0),
        lpips_weight=loss_weights_cfg.get("lpips_weight", 1.0),
        gram_weight=loss_weights_cfg.get("gram_weight", 0.5),
        learning_rate=optimizer_cfg.get("learning_rate", 2e-5),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        use_cosine_scheduler=optimizer_cfg.get("use_cosine_scheduler", True),
        scheduler_eta_min=optimizer_cfg.get("scheduler_eta_min", 1e-6),

        log_training_images_freq=partial_cfg.get("log_training_images_freq", 5),
        max_training_images=partial_cfg.get("max_training_images", 4),
        log_training_error_images=partial_cfg.get("log_training_error_images", True),

        validation_data_root=cfg.data.get("validation_data_root"),
        first_swap_dir_filter=cfg.data.get("first_swap_dir_filter"),
        test_output_root=cfg.data.get("test_output_root"),
        test_camera_filter=cfg.data.get("test_camera_filter"),
        test_save_intermediate_outputs=cfg.data.get("test_save_intermediate_outputs", True),

        use_head_alignment=cfg.data.get("use_head_alignment", True),
        mask_source=cfg.data.get("mask_source", None),
        adjust_calibration=cfg.data.get("adjust_calibration", True),

        refinement_mode=cfg.data.get("refinement_mode", "partial"),
        restoration_feather_radius=cfg.model.get("restoration_feather_radius", 0),
    )

    return datamodule, model


def setup_vanilla_training(cfg: DictConfig):

    logger.info("📝 Setting up vanilla DiFix3D training mode")


    difix3d_datamodule_kwargs = {
        "training_mode": True,
        "degraded_dir": cfg.data.degraded_dir,
        "gt_dataset_root": cfg.data.gt_dataset_root,
        "train_split_ratio": cfg.data.get("train_split_ratio", 0.8),
        "batch_size": cfg.data.get("batch_size", 2),
        "num_workers": cfg.data.get("num_workers", 4),
        "prompts": cfg.data.get("prompts", "remove degradation"),
        "validate_gt_existence": cfg.data.get("validate_gt_existence", True),
        "skip_missing_pairs": cfg.data.get("skip_missing_pairs", True),
        "extract_subject_pattern": cfg.data.get("extract_subject_pattern", r"restored_(\d+)_from_"),
        "reference_strategy": cfg.data.get("reference_strategy", "placeholder"),
        "log_split_statistics": cfg.data.get("log_split_statistics", True),
        "use_same_subjects_for_val": cfg.data.get("use_same_subjects_for_val", True),
        "subject_a": cfg.data.get("subject_a", None),
        "subject_b": cfg.data.get("subject_b", None),
        "adaptive_resolution_enabled": cfg.data.get("adaptive_resolution_enabled", True),
        "swapped_validation_enabled": cfg.data.get("swapped_validation_enabled", True),
        "swapped_validation_dir": cfg.data.get("swapped_validation_dir", None),
        "max_validation_cameras": cfg.data.get("max_validation_cameras", 4),
        "visualize_restored_images": cfg.data.get("visualize_restored_images", False),
        "seed": cfg.get("seed", 42),
    }


    datamodule = DiFix3DDataModule(**difix3d_datamodule_kwargs)


    training_cfg = cfg.model.get("training", {})
    unet_lora_cfg = training_cfg.get("unet_lora", {})
    swapfix_lora_cfg = training_cfg.get("swapfix_lora", {})
    loss_weights_cfg = training_cfg.get("loss_weights", {})
    optimizer_cfg = training_cfg.get("optimizer", {})


    model = DiFix3DModule(
        training_mode=cfg.model.training_mode,
        model_name=cfg.model.get("model_name", "nvidia/difix"),
        prompt=cfg.model.get("prompt", "remove degradation"),
        num_inference_steps=cfg.model.get("num_inference_steps", 1),
        timesteps=cfg.model.get("timesteps", [199]),
        guidance_scale=cfg.model.get("guidance_scale", 0.0),
        trust_remote_code=cfg.model.get("trust_remote_code", True),
        torch_dtype=cfg.model.get("torch_dtype", None),
        unet_training_mode="lora" if unet_lora_cfg.get("enabled", False) else "full",
        unet_lora_rank=unet_lora_cfg.get("rank", 4),
        unet_lora_alpha=unet_lora_cfg.get("alpha", 4.0),
        unet_lora_dropout=unet_lora_cfg.get("dropout", 0.1),
        unet_lora_target_modules=unet_lora_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"]),
        lora_training_mode=training_cfg.get("lora_training_mode", "swapfix"),
        difix_weight=training_cfg.get("difix_weight", 0.5),
        swapfix_weight=training_cfg.get("swapfix_weight", 0.5),
        swapfix_lora_rank=swapfix_lora_cfg.get("rank", 8),
        lora_alpha=swapfix_lora_cfg.get("alpha", 32.0),
        lora_dropout=swapfix_lora_cfg.get("dropout", 0.1),
        target_modules=swapfix_lora_cfg.get("target_modules", [
            "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]),
        l2_weight=loss_weights_cfg.get("l2_weight", 1.0),
        lpips_weight=loss_weights_cfg.get("lpips_weight", 1.0),
        gram_weight=loss_weights_cfg.get("gram_weight", 0.5),
        learning_rate=optimizer_cfg.get("learning_rate", 2e-5),
        weight_decay=optimizer_cfg.get("weight_decay", 0.01),
        use_cosine_scheduler=optimizer_cfg.get("use_cosine_scheduler", True),
        scheduler_eta_min=optimizer_cfg.get("scheduler_eta_min", 1e-6),
    )

    return datamodule, model


def setup_wandb_auth(api_key: str = None, base_url: str = None) -> bool:

    try:

        if base_url:
            os.environ["WANDB_BASE_URL"] = base_url
            logger.info(f"🔗 Using WandB server at: {base_url}")


        if api_key:
            os.environ["WANDB_API_KEY"] = api_key


        if api_key:
            wandb.login(key=api_key, relogin=True, verify=True)
        else:

            wandb.login(relogin=True, verify=True)

        logger.info("✅ WandB authentication successful!")


        test_run = wandb.init(project="test-connection", mode="disabled")
        if test_run:
            wandb.finish()
            logger.info("✅ WandB connection verified!")

        return True

    except wandb.errors.AuthenticationError as e:
        logger.error(f"❌ WandB authentication failed: {e}")
        logger.error("Please check your API key and server configuration.")
        return False

    except wandb.errors.CommError as e:
        logger.error(f"❌ WandB communication error: {e}")
        logger.error("Please check your network connection and server URL.")
        return False

    except Exception as e:
        logger.warning(f"⚠️  WandB setup failed with unexpected error: {e}")
        logger.warning("Training will continue but wandb logging may not work properly.")
        return False


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:

    logger.info("=" * 80)
    logger.info("DiFix3D+ SwapFix LoRA Training")
    logger.info("=" * 80)


    logger.info("Configuration loaded:")
    logger.info(f"  Task: {cfg.get('task_name', 'N/A')}")
    logger.info(f"  Name: {cfg.get('name', 'N/A')}")
    logger.info(f"  Training enabled: {cfg.get('train', False)}")
    logger.info(f"  Testing enabled: {cfg.get('test', False)}")


    if not cfg.get('train', False) and not cfg.get('test', False):
        logger.error("Neither training nor testing is enabled in configuration.")
        logger.error("Please set either train=true or test=true")
        sys.exit(1)

    if "model" not in cfg or not cfg.model.get("training_mode", False):
        logger.error("Model not configured for training mode")
        sys.exit(1)


    if "data" not in cfg:
        logger.error("Data configuration missing")
        sys.exit(1)

    if (not cfg.data.get("degraded_dir") or not cfg.data.get("gt_dataset_root")) and (not cfg.data.get("combined_data_root")):
        logger.error("Missing required data paths!")
        logger.error("Please provide:")
        logger.error("  +data.degraded_dir='path/to/degraded/images'")
        logger.error("  +data.gt_dataset_root='path/to/gt/dataset'")
        sys.exit(1)

    logger.info("Data configuration:")
    if cfg.data.get("combined_data_root"):
        logger.info(f"  Combined data root: {cfg.data.combined_data_root}")
        refinement_mode = cfg.data.get("refinement_mode", "partial")
        logger.info(f"  Refinement mode: {refinement_mode}")
        if refinement_mode == "full":
            fullbody_tr = cfg.data.get("fullbody_target_resolution", None)
            if fullbody_tr is not None and isinstance(fullbody_tr, (list, tuple)) and len(fullbody_tr) == 2:
                logger.info(f"    Full-body mode: adaptive resolution ({fullbody_tr[0]}×{fullbody_tr[1]})")
            else:
                logger.info(f"    Full-body mode: adaptive resolution (448×896)")
            logger.info(f"    GT dataset root: {cfg.data.get('gt_dataset_root', 'NOT SET')}")
        else:
            logger.info(f"    Partial mode: simple resize ({cfg.data.get('target_resolution', 512)}×{cfg.data.get('target_resolution', 512)})")
    else:
        logger.info(f"  Degraded dir: {cfg.data.degraded_dir}")
        logger.info(f"  GT dataset root: {cfg.data.gt_dataset_root}")
        logger.info(f"  Train split ratio: {cfg.data.get('train_split_ratio', 0.8)}")
    logger.info(f"  Batch size: {cfg.data.get('batch_size', 2)}")

    if cfg.get("wandb", {}).get("local", False):

        logger.info("Setting up WandB authentication in local mode...")


        wandb_api_key = None
        wandb_base_url = None

        if "wandb" in cfg:
            wandb_api_key = cfg.wandb.get("api_key", None)
            wandb_base_url = cfg.wandb.get("base_url", None)


        if not wandb_api_key:
            wandb_api_key = ""

        setup_wandb_auth(api_key=wandb_api_key, base_url=wandb_base_url)
    elif cfg.get("wandb", {}).get("offline", False):
        logger.info("WandB authentication in offline mode...")
    else:
        logger.info("WandB authentication in online mode...")


    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Random seed set to: {cfg.seed}")


    logger.info("Determining training mode...")
    training_cfg = cfg.model.get("training", {})
    identity_enabled = training_cfg.get("identity", {}).get("enabled", False)
    partial_refinement_enabled = training_cfg.get("partial_refinement", {}).get("enabled", False)


    if identity_enabled and partial_refinement_enabled:
        logger.error("Cannot enable both identity and partial refinement modes!")
        logger.error("Please set only one of:")
        logger.error("  model.training.identity.enabled=true")
        logger.error("  model.training.partial_refinement.enabled=true")
        sys.exit(1)


    if identity_enabled:
        datamodule, model = setup_identity_preserving_training(cfg)
    elif partial_refinement_enabled:
        datamodule, model = setup_partial_refinement_training(cfg)
    else:
        datamodule, model = setup_vanilla_training(cfg)


    callbacks = []
    if "callbacks" in cfg and cfg.callbacks:
        if cfg.callbacks.get("model_checkpoint") and cfg.trainer.get("enable_checkpointing", True):
            checkpoint_callback = instantiate(cfg.callbacks.model_checkpoint)
            callbacks.append(checkpoint_callback)
            logger.info(f"  Model checkpoint callback: {type(checkpoint_callback).__name__}")
        if cfg.callbacks.get("learning_rate_monitor"):
            callbacks.append(instantiate(cfg.callbacks.learning_rate_monitor))
            logger.info("  Learning rate monitor callback added")
        if cfg.callbacks.get("early_stopping"):
            callbacks.append(instantiate(cfg.callbacks.early_stopping))
            logger.info("  Early stopping callback added")


    lightning_logger = None
    if "logger" in cfg and cfg.logger:
        lightning_logger = instantiate(cfg.logger)


    logger.info("Initializing Lightning trainer...")
    trainer_config = cfg.get("trainer", {})
    trainer = L.Trainer(
        max_epochs=trainer_config.get("max_epochs", 10),
        max_steps=trainer_config.get("max_steps", -1),
        limit_train_batches=trainer_config.get("limit_train_batches", 1.0),
        limit_val_batches=trainer_config.get("limit_val_batches", 1.0),
        limit_test_batches=trainer_config.get("limit_test_batches", 1.0),
        fast_dev_run=trainer_config.get("fast_dev_run", False),
        precision=trainer_config.get("precision", "32-true"),
        accelerator=trainer_config.get("accelerator", "gpu"),
        devices=trainer_config.get("devices", 1),
        strategy=trainer_config.get("strategy", "auto"),
        check_val_every_n_epoch=trainer_config.get("check_val_every_n_epoch", None),
        val_check_interval=trainer_config.get("val_check_interval", None),
        log_every_n_steps=trainer_config.get("log_every_n_steps", 10),
        enable_progress_bar=trainer_config.get("enable_progress_bar", True),
        enable_checkpointing=trainer_config.get("enable_checkpointing", True),
        default_root_dir=trainer_config.get("default_root_dir", "./outputs/difix3d_swapfix_training"),
        callbacks=callbacks,
        logger=lightning_logger,
        deterministic=False,
        num_sanity_val_steps=trainer_config.get("num_sanity_val_steps", 2),
    )


    training_cfg = cfg.model.get("training", {})
    identity_cfg = training_cfg.get("identity", {})

    logger.info("Training configuration:")
    logger.info(f"  Model: {cfg.model.get('model_name', 'nvidia/difix')}")
    logger.info(f"  Max epochs: {trainer_config.get('max_epochs', 10)}")
    logger.info(f"  Precision: {trainer_config.get('precision', '32-true')}")
    logger.info(f"  Learning rate: {training_cfg.get('optimizer', {}).get('learning_rate', 2e-5)}")


    unet_lora_cfg = training_cfg.get("unet_lora", {})
    if unet_lora_cfg.get("enabled", False):
        logger.info(f"  UNet LoRA: rank={unet_lora_cfg.get('rank', 4)}, alpha={unet_lora_cfg.get('alpha', 4.0)}")
    else:
        logger.info(f"  UNet training: Full fine-tuning")


    swapfix_lora_cfg = training_cfg.get("swapfix_lora", {})
    logger.info(f"  SwapFix LoRA: rank={swapfix_lora_cfg.get('rank', 8)}, alpha={swapfix_lora_cfg.get('alpha', 32.0)}")


    loss_weights_cfg = training_cfg.get("loss_weights", {})
    logger.info(f"  Loss weights: L2={loss_weights_cfg.get('l2_weight', 1.0)}, LPIPS={loss_weights_cfg.get('lpips_weight', 1.0)}, Gram={loss_weights_cfg.get('gram_weight', 0.1)}")


    if identity_cfg.get("enabled", False):
        logger.info(f"  🎯 Identity training: ENABLED")
        logger.info(f"    Pattern: {identity_cfg.get('normal_steps', 2)}:{identity_cfg.get('identity_steps', 1)} (normal:identity)")
        logger.info(f"    CLIP model: {identity_cfg.get('model_name_or_path', 'openai/clip-vit-base-patch32')}")
        logger.info(f"    Identity loss weight: {identity_cfg.get('identity_loss_weight', 0.1)}")
        logger.info(f"    Similarity target: {identity_cfg.get('similarity_target', 0.85)}")
    else:
        logger.info(f"  Identity training: Disabled")


    try:

        if cfg.get('train', False):
            logger.info("=" * 80)
            logger.info("Starting SwapFix LoRA training...")
            logger.info("=" * 80)

            trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path"))
            logger.success("Training completed successfully!")


            if hasattr(model, 'swapfix_lora_layers'):
                logger.info(f"SwapFix LoRA layers trained: {len(model.swapfix_lora_layers)}")


            training_cfg = cfg.model.get("training", {})
            identity_cfg = training_cfg.get("identity", {})
            if identity_cfg.get("enabled", False):
                identity_info = model.get_identity_training_info()
                if identity_info.get("enabled"):
                    logger.info(f"🎯 Identity training completed:")
                    if identity_info.get("alternating_trainer"):
                        trainer_info = identity_info["alternating_trainer"]
                        logger.info(f"  Pattern executed: {trainer_info.get('pattern', 'Unknown')}")
                        logger.info(f"  Total cycles: {trainer_info.get('total_cycles', 'Unknown')}")
                    logger.info(f"  Identity loss: {'Enabled' if identity_info.get('identity_loss') else 'Disabled'}")


            if hasattr(trainer, 'checkpoint_callback') and trainer.checkpoint_callback:
                if hasattr(trainer.checkpoint_callback, 'best_model_path'):
                    logger.info(f"Best model saved to: {trainer.checkpoint_callback.best_model_path}")
                if hasattr(trainer.checkpoint_callback, 'last_model_path'):
                    logger.info(f"Last model saved to: {trainer.checkpoint_callback.last_model_path}")


        if cfg.get('test', False):
            logger.info("=" * 80)
            logger.info("Starting SwapFix LoRA testing...")
            logger.info("=" * 80)


            ckpt_path = cfg.get('ckpt_path', None)
            if not ckpt_path:
                logger.error("Testing requires a checkpoint path!")
                logger.error("Please provide: ckpt_path='path/to/checkpoint.ckpt'")
                sys.exit(1)


            from pathlib import Path
            if not Path(ckpt_path).exists():
                logger.error(f"Checkpoint file not found: {ckpt_path}")
                sys.exit(1)

            logger.info(f"Loading checkpoint: {ckpt_path}")


            logger.info("Test configuration:")
            if cfg.data.get("test_pairs_yaml"):
                logger.info(f"  Test pairs YAML: {cfg.data.test_pairs_yaml}")
            if cfg.data.get("test_include_only_pair_ids"):
                logger.info(f"  Test pair IDs: {cfg.data.test_include_only_pair_ids}")
            if cfg.data.get("test_output_root"):
                logger.info(f"  Output directory: {cfg.data.test_output_root}")
            if cfg.data.get("test_camera_filter"):
                logger.info(f"  Camera filter: {cfg.data.test_camera_filter}")
            else:
                logger.info(f"  Camera filter: ALL cameras")


            test_results = trainer.test(model, datamodule, ckpt_path=ckpt_path)

            logger.success("Testing completed successfully!")
            logger.info(f"Test results: {test_results}")

            if cfg.data.get("test_output_root"):
                logger.info(f"✅ Test outputs saved to: {cfg.data.test_output_root}")
                logger.info(f"📁 Check for:")
                logger.info(f"   - RGB images: {{subject}}/{{cam}}/0000.jpg")
                logger.info(f"   - Masks: {{subject}}/{{cam}}/mask/pha/0000.png")
                logger.info(f"   - Calibration: {{subject}}/calibration_full.json")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    finally:

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
