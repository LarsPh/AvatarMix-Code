import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from loguru import logger

import torch
import torch.nn.functional as F
import lightning as L
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from PIL import Image
from omegaconf import DictConfig, MISSING
from hydra.core.config_store import ConfigStore
import numpy as np
import wandb


try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    logger.warning("PEFT not available. Install with: pip install peft")


difix_src_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(difix_src_path))

try:
    from pipeline_difix import DifixPipeline
    from diffusers.utils import load_image
except ImportError as e:
    raise ImportError(f"Cannot import DiFix3D modules: {e}. Ensure src/ directory is accessible.")


@dataclass
class TrainingConfig:


    learning_rate: float = 1e-5
    batch_size: int = 2
    num_epochs: int = 10
    mixed_precision: str = "bf16"


    unet_training_mode: str = "full"
    unet_lora_rank: int = 16
    unet_lora_alpha: float = 16.0
    unet_lora_dropout: float = 0.1
    unet_lora_target_modules: List[str] = field(default_factory=lambda: ["to_k", "to_q", "to_v", "to_out.0"])


    swapfix_lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["conv", "to_k", "to_q", "to_v", "to_out.0"])


    l2_weight: float = 1.0
    lpips_weight: float = 0.5
    gram_weight: float = 0.05


    weight_decay: float = 0.01
    use_cosine_scheduler: bool = True
    scheduler_eta_min: float = 1e-6

@dataclass
class DiFix3DConfig:


    model_name: str = "nvidia/difix"
    training_mode: bool = False
    training: TrainingConfig = field(default_factory=TrainingConfig)


    prompt: str = "remove degradation"
    num_inference_steps: int = 1
    timesteps: List[int] = field(default_factory=lambda: [199])
    guidance_scale: float = 0.0
    trust_remote_code: bool = True
    torch_dtype_str: str = "auto"


class DiFix3DModule(L.LightningModule):


    def __init__(
        self,

        model_name: str = "nvidia/difix",
        prompt: str = "remove degradation",
        num_inference_steps: int = 1,
        timesteps: List[int] = None,
        guidance_scale: float = 0.0,
        trust_remote_code: bool = True,
        torch_dtype: Optional[torch.dtype] = None,


        training_mode: bool = False,
        learning_rate: float = 2e-5,


        unet_training_mode: str = "full",
        unet_lora_rank: int = 16,
        unet_lora_alpha: float = 16.0,
        unet_lora_dropout: float = 0.1,
        unet_lora_target_modules: List[str] = None,


        lora_training_mode: str = "swapfix",
        difix_weight: float = 0.5,
        swapfix_weight: float = 0.5,


        swapfix_lora_rank: int = 8,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None,


        l2_weight: float = 1.0,
        lpips_weight: float = 1.0,
        gram_weight: float = 0.5,


        weight_decay: float = 0.01,
        use_cosine_scheduler: bool = True,
        scheduler_eta_min: float = 1e-6,


        log_validation_images: bool = True,
        max_validation_images: int = 4,
        log_training_errors: bool = True,
        log_learning_rate: bool = True,


        **kwargs
    ):

        super().__init__()


        self._validate_lightning_version()


        self.model_name = model_name
        self.prompt = prompt
        self.num_inference_steps = num_inference_steps
        self.timesteps = timesteps or [199]
        self.guidance_scale = guidance_scale
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = torch_dtype


        self.training_mode = training_mode
        self.learning_rate = learning_rate


        self.unet_training_mode = unet_training_mode
        self.unet_lora_rank = unet_lora_rank
        self.unet_lora_alpha = unet_lora_alpha
        self.unet_lora_dropout = unet_lora_dropout
        self.unet_lora_target_modules = unet_lora_target_modules or ["to_k", "to_q", "to_v", "to_out.0"]


        self.lora_training_mode = lora_training_mode
        self.difix_weight = difix_weight
        self.swapfix_weight = swapfix_weight


        if self.lora_training_mode not in ["swapfix", "difix", "combined"]:
            raise ValueError(
                f"Invalid lora_training_mode: {self.lora_training_mode}. "
                f"Must be 'swapfix', 'difix', or 'combined'"
            )


        if self.difix_weight <= 0 or self.swapfix_weight <= 0:
            raise ValueError(
                f"LoRA weights must be positive: "
                f"difix_weight={self.difix_weight}, swapfix_weight={self.swapfix_weight}"
            )


        self.swapfix_lora_rank = swapfix_lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or [
                "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]


        self.l2_weight = l2_weight
        self.lpips_weight = lpips_weight
        self.gram_weight = gram_weight


        self.weight_decay = weight_decay
        self.use_cosine_scheduler = use_cosine_scheduler
        self.scheduler_eta_min = scheduler_eta_min


        self.log_validation_images = log_validation_images
        self.max_validation_images = max_validation_images
        self.log_training_errors = log_training_errors
        self.log_learning_rate = log_learning_rate


        self.pipe = None


        self.lpips_fn = None
        self.vgg_features = None


    def configure_model(self) -> None:


        if hasattr(self, 'pipe') and self.pipe is not None:
            logger.info("Pipeline already configured, skipping configure_model()")
            return

        logger.info(f"=== configure_model(): Creating pipeline for checkpoint loading ===")
        logger.info(f"Training mode: {self.training_mode}")

        if self.training_mode:

            self._setup_pipeline_with_lora()
            self._setup_loss_functions()
        else:

            self._setup_pipeline_inference_only()

    def setup(self, stage: str) -> None:

        logger.info(f"=== setup(stage={stage}): Validating configuration ===")


        self._validate_stage_training_mode_combination(stage)


        if not hasattr(self, 'pipe') or self.pipe is None:
            logger.warning("Pipeline not found in setup(), creating now (fallback for Lightning <2.0)")
            logger.warning("Note: Lightning 2.0+ is required for proper checkpoint loading with LoRA!")


            if self.training_mode:

                logger.info(f"Creating pipeline WITH LoRA (training_mode=True, stage={stage})")
                self._setup_pipeline_with_lora()
                self._setup_loss_functions()
            else:

                logger.info(f"Creating inference pipeline WITHOUT LoRA (training_mode=False, stage={stage})")
                self._setup_pipeline_inference_only()


    def _validate_lightning_version(self) -> None:

        try:
            import lightning as L
            from packaging import version

            lightning_version = version.parse(L.__version__)
            min_version = version.parse("2.0.0")

            if lightning_version < min_version:
                raise RuntimeError(
                    f"PyTorch Lightning {min_version} or higher is required for proper checkpoint loading. "
                    f"Found version {L.__version__}. "
                    f"The configure_model() hook (required for LoRA checkpoint loading) was added in Lightning 2.0. "
                    f"Please upgrade: pip install 'lightning>=2.0.0'"
                )

            logger.info(f"Lightning version {L.__version__} OK (>= 2.0.0)")

        except ImportError:
            raise RuntimeError(
                "Could not import 'packaging' module. "
                "Please install: pip install packaging"
            )

    def _validate_stage_training_mode_combination(self, stage: str) -> None:


        if stage == "predict" and self.training_mode:
            raise ValueError(
                f"Cannot run predict stage with training_mode=True. "
                f"Predict mode is for pure inference without LoRA. "
                f"Use test stage (test_step) for testing with trained LoRA weights. "
                f"Set training_mode=False for predict mode."
            )


        if stage in ["fit", "train", "validate", "test"] and not self.training_mode:
            raise ValueError(
                f"Cannot run {stage} stage with training_mode=False. "
                f"Training stages require training_mode=True to enable LoRA modules. "
                f"Use predict stage for inference without LoRA, or set training_mode=True."
            )

    def _setup_pipeline_inference_only(self) -> None:

        try:
            logger.info(f"Loading DiFix3D+ pipeline for inference from {self.model_name}...")
            self.pipe = DifixPipeline.from_pretrained(
                self.model_name,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
            )


            self.pipe = self.pipe.to(self.device)


            self.pipe.set_progress_bar_config(disable=True)

            logger.info("DiFix3D+ pipeline loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to setup DiFix3D+ pipeline: {e}")

    def _setup_pipeline_with_lora(self) -> None:

        try:
            logger.info(f"Loading DiFix3D+ pipeline for training from {self.model_name}...")


            self.pipe = DifixPipeline.from_pretrained(
                self.model_name,
                model_type="auto",
                lora_mode=self.lora_training_mode,
                trust_remote_code=self.trust_remote_code,
                torch_dtype=self.torch_dtype,
            )


            self.pipe = self.pipe.to(self.device)


            self._ensure_swapfix_lora()


            if self.unet_training_mode == "lora":
                self._setup_unet_lora()


            self._setup_training_mode()


            self._register_pipeline_modules()


            logger.info("DiFix3D+ training pipeline loaded successfully!")

        except Exception as e:
            raise RuntimeError(f"Failed to setup training pipeline: {e}")

    def _ensure_swapfix_lora(self) -> None:

        try:

            if hasattr(self.pipe.vae, 'peft_config') and 'vae_swapfix' in self.pipe.vae.peft_config:
                logger.info("SwapFix LoRA already present.")
                return


            logger.info(f"Adding SwapFix LoRA (rank={self.swapfix_lora_rank})...")


            base_target_modules = [
                "conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
                "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
                "to_k", "to_q", "to_v", "to_out.0",
            ]

            full_target_modules = []
            for name, module in self.pipe.vae.named_modules():
                if 'decoder' in name and any(name.endswith(x) for x in base_target_modules):
                    full_target_modules.append(name)

            logger.info(f"SwapFix LoRA will target {len(full_target_modules)} decoder modules")

            swapfix_config = LoraConfig(
                r=self.swapfix_lora_rank,
                lora_alpha=self.lora_alpha,
                target_modules=full_target_modules,
                lora_dropout=self.lora_dropout,
            )


            self.pipe.vae.add_adapter(swapfix_config, adapter_name="vae_swapfix")
            self.pipe.vae.encoder.requires_grad_(False)


            swapfix_params = sum(
                p.numel() for name, p in self.pipe.vae.named_parameters()
                if 'vae_swapfix' in name and 'lora' in name
            )
            logger.info(f"SwapFix LoRA added successfully with {swapfix_params:,} parameters!")


            if getattr(self, 'lora_training_mode', None) == 'combined':
                self._clone_vae_lora_from_to(
                    getattr(self.pipe, 'difix_adapter_name', 'vae_skip'),
                    'vae_swapfix',
                )

        except Exception as e:
            logger.warning(f"Could not add SwapFix LoRA: {e}")
            logger.warning("Continuing without SwapFix LoRA...")

    def _clone_vae_lora_from_to(self, src_adapter_name: str, dst_adapter_name: str) -> None:

        try:
            from peft.utils import get_peft_model_state_dict
        except Exception as e:
            logger.warning(f"PEFT utilities unavailable for cloning LoRA weights: {e}")
            return

        try:

            src_state = get_peft_model_state_dict(self.pipe.vae, adapter_name=src_adapter_name)
            if not src_state:
                logger.warning(f"No LoRA state found for adapter '{src_adapter_name}'. Skipping clone.")
                return


            self.pipe.vae.load_adapter(src_state, adapter_name=dst_adapter_name)
            logger.info(f"Cloned VAE LoRA weights from '{src_adapter_name}' to '{dst_adapter_name}'.")
        except Exception as e:
            logger.warning(f"Failed to clone VAE LoRA weights from '{src_adapter_name}' to '{dst_adapter_name}': {e}")

    def _setup_unet_lora(self) -> None:

        try:

            if hasattr(self.pipe.unet, 'peft_config') and 'unet_swapfix' in self.pipe.unet.peft_config:
                logger.info("UNet LoRA already present.")
                return


            logger.info(f"Adding UNet LoRA (rank={self.unet_lora_rank}, target_modules={self.unet_lora_target_modules})...")

            unet_lora_config = LoraConfig(
                r=self.unet_lora_rank,
                lora_alpha=self.unet_lora_alpha,
                target_modules=self.unet_lora_target_modules,
                lora_dropout=self.unet_lora_dropout,
            )


            self.pipe.unet.add_adapter(unet_lora_config, adapter_name="unet_swapfix")

            logger.info("UNet LoRA added successfully!")

        except Exception as e:
            logger.warning(f"Could not add UNet LoRA: {e}")
            logger.warning("Continuing without UNet LoRA...")

    def _setup_training_mode(self) -> None:

        logger.info(f"Configuring training mode (UNet: {self.unet_training_mode})...")


        self._freeze_difix_lora()


        if self.unet_training_mode == "full":

            logger.info("Enabling full UNet fine-tuning...")
            self.pipe.unet.train()
            for param in self.pipe.unet.parameters():
                param.requires_grad = True
        elif self.unet_training_mode == "lora":

            logger.info("Enabling UNet LoRA training...")
            self.pipe.unet.train()


            for param in self.pipe.unet.parameters():
                param.requires_grad = False


            if hasattr(self.pipe.unet, 'peft_config') and 'unet_swapfix' in self.pipe.unet.peft_config:
                for name, param in self.pipe.unet.named_parameters():
                    if "unet_swapfix" in name:
                        param.requires_grad = True

        else:
            raise ValueError(f"Invalid unet_training_mode: {self.unet_training_mode}. Must be 'full' or 'lora'.")


        self.pipe.vae.decoder.train()
        for name, param in self.pipe.vae.decoder.named_parameters():
            if "skip_conv" in name:
                param.requires_grad = True
            elif "vae_swapfix" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False


        self.pipe.text_encoder.eval()
        for param in self.pipe.text_encoder.parameters():
            param.requires_grad = False

        self.pipe.vae.encoder.eval()
        for param in self.pipe.vae.encoder.parameters():
            param.requires_grad = False


        logger.info(f"Configuring LoRA training mode: {self.lora_training_mode}")
        if self.lora_training_mode == "combined":

            self.pipe.set_lora_mode(
                mode="combined",
                difix_weight=self.difix_weight,
                swapfix_weight=self.swapfix_weight
            )
            logger.info(
                f"LoRA training mode set to 'combined' "
                f"(difix_weight={self.difix_weight}, swapfix_weight={self.swapfix_weight})"
            )
        elif self.lora_training_mode == "swapfix":

            self.pipe.set_lora_mode(mode="swapfix")
            logger.info("LoRA training mode set to 'swapfix' (SwapFix LoRA only)")
        elif self.lora_training_mode == "difix":

            self.pipe.set_lora_mode(mode="difix")
            logger.info("LoRA training mode set to 'difix' (pretrained LoRA only)")


        self._log_training_parameter_summary()

        logger.info("Training mode configured successfully!")

    def _log_training_parameter_summary(self) -> None:

        unet_trainable = sum(p.numel() for p in self.pipe.unet.parameters() if p.requires_grad)
        unet_total = sum(p.numel() for p in self.pipe.unet.parameters())

        vae_trainable = sum(p.numel() for p in self.pipe.vae.parameters() if p.requires_grad)
        vae_total = sum(p.numel() for p in self.pipe.vae.parameters())

        text_trainable = sum(p.numel() for p in self.pipe.text_encoder.parameters() if p.requires_grad)
        text_total = sum(p.numel() for p in self.pipe.text_encoder.parameters())

        logger.info("=== Training Parameter Summary ===")
        logger.info(f"UNet: {unet_trainable:,}/{unet_total:,} trainable ({unet_trainable/unet_total*100:.1f}%)")
        logger.info(f"VAE: {vae_trainable:,}/{vae_total:,} trainable ({vae_trainable/vae_total*100:.1f}%)")
        logger.info(f"Text Encoder: {text_trainable:,}/{text_total:,} trainable ({text_trainable/text_total*100:.1f}%)")
        logger.info(f"Total trainable: {unet_trainable + vae_trainable + text_trainable:,}")
        logger.info("==================================")

    def _freeze_difix_lora(self) -> None:

        try:
            for name, param in self.pipe.vae.decoder.named_parameters():
                if "vae_skip" in name:
                    param.requires_grad = False
            logger.info("Original DiFix LoRA frozen.")
        except Exception as e:
            logger.warning(f"Could not freeze DiFix LoRA: {e}")

    def _register_pipeline_modules(self) -> None:

        if self.pipe is None:
            logger.error("Cannot register modules: pipeline is None")
            return

        try:
            logger.info("Registering pipeline nn.Module components for Lightning parameter discovery...")

            modules_registered = []


            known_components = [
                'unet', 'vae', 'text_encoder', 'scheduler',
                'safety_checker', 'feature_extractor', 'tokenizer'
            ]

            for attr_name in known_components:
                try:
                    if hasattr(self.pipe, attr_name):
                        attr_value = getattr(self.pipe, attr_name)


                        if isinstance(attr_value, torch.nn.Module):

                            setattr(self, f"pipe_{attr_name}", attr_value)
                            modules_registered.append(f"pipe_{attr_name}")
                            logger.debug(f"Registered nn.Module: {attr_name} -> pipe_{attr_name}")
                        else:
                            logger.debug(f"Skipped non-module: {attr_name} ({type(attr_value)})")

                except Exception as e:
                    logger.debug(f"Skipping attribute {attr_name}: {e}")
                    continue


            for attr_name in dir(self.pipe):

                if attr_name.startswith('_') or attr_name in known_components:
                    continue

                try:
                    if not hasattr(self.pipe, attr_name):
                        continue

                    attr_value = getattr(self.pipe, attr_name)


                    if callable(attr_value):
                        continue


                    if isinstance(attr_value, torch.nn.Module):

                        setattr(self, f"pipe_{attr_name}", attr_value)
                        modules_registered.append(f"pipe_{attr_name}")
                        logger.debug(f"Registered additional nn.Module: {attr_name} -> pipe_{attr_name}")

                except Exception as e:

                    logger.debug(f"Skipping attribute {attr_name}: {e}")
                    continue


            logger.info(f"Registered {len(modules_registered)} pipeline modules: {modules_registered}")


            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

            logger.info(f"Lightning parameter discovery validation:")
            logger.info(f"  Total parameters discovered: {total_params:,}")
            logger.info(f"  Trainable parameters discovered: {trainable_params:,}")

            if trainable_params == 0:
                logger.error("⚠️  No trainable parameters discovered by Lightning!")
                logger.error("This indicates the parameter registration failed.")
            else:
                logger.info("✅ Parameter discovery successful!")

        except Exception as e:
            logger.error(f"Failed to register pipeline modules: {e}")
            logger.error("Parameter discovery may fail, causing training issues")

    def _setup_loss_functions(self) -> None:

        try:


            self.lpips_fn = LPIPS(net_type='vgg', normalize=True)
            self.lpips_fn.eval()

            self._setup_vgg_features()

        except Exception as e:
            logger.warning(f"Could not initialize loss functions: {e}")

    def _setup_vgg_features(self) -> None:

        try:
            import torchvision.models as models


            vgg = models.vgg16(pretrained=True).features
            vgg = vgg.to(self.device).eval()


            for param in vgg.parameters():
                param.requires_grad = False

            self.vgg_features = vgg
            logger.info("VGG features for Gram loss initialized.")

        except Exception as e:
            logger.warning(f"Could not initialize VGG features: {e}")

    def predict_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:

        if self.pipe is None:
            self._setup_inference_pipeline()


        if isinstance(batch, dict):
            images = batch.get("images", batch.get("image"))
            prompts = batch.get("prompts", batch.get("prompt", [self.prompt]))
            metadata_list = batch.get("metadata", [])
        else:

            images = batch
            prompts = [self.prompt]
            metadata_list = []


        if isinstance(prompts, str):
            prompts = [prompts]
        if len(prompts) == 1 and (isinstance(images, list) and len(images) > 1):
            prompts = prompts * len(images)
        elif isinstance(images, torch.Tensor) and images.dim() == 4:
            batch_size = images.shape[0]
            if len(prompts) == 1:
                prompts = prompts * batch_size


        try:
            with torch.no_grad():
                result = self.pipe(
                    prompt=prompts,
                    image=images,
                    num_inference_steps=self.num_inference_steps,
                    timesteps=self.timesteps,
                    guidance_scale=self.guidance_scale,
                    disable_progress_bar=True,
                )

                refined_images = result.images


            output = {
                "refined_images": refined_images,
                "prompts": prompts,
                "metadata": metadata_list,
                "batch_idx": batch_idx,
            }

            return output

        except Exception as e:
            logger.error(f"Error in predict_step for batch {batch_idx}: {e}")

            return {
                "error": str(e),
                "batch_idx": batch_idx,
                "metadata": metadata_list,
            }

    def forward(self, x: Any) -> Any:

        return self.predict_step(x, 0)


    def _log_training_metrics(self, losses: Dict[str, torch.Tensor], step: int) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            return


        log_dict = {}
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_dict[f"train/{loss_name}"] = loss_value.item()
            else:
                log_dict[f"train/{loss_name}"] = loss_value


        if self.log_learning_rate:
            optimizer = self.optimizers()
            if optimizer is not None:
                log_dict["train/learning_rate"] = optimizer.param_groups[0]['lr']


        log_dict["trainer/global_step"] = step


        self.logger.experiment.log(log_dict)

    def _log_validation_metrics(self, metrics: Dict[str, torch.Tensor], step: int) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            return

        log_dict = {}
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor):
                log_dict[f"val/{metric_name}"] = metric_value.item()
            else:
                log_dict[f"val/{metric_name}"] = metric_value


        log_dict["trainer/global_step"] = step


        self.logger.experiment.log(log_dict)

    def tensor_to_wandb_image(self, tensor: torch.Tensor, caption: str) -> wandb.Image:

        import numpy as np

        tensor = torch.clamp(tensor, 0, 1).float()

        if tensor.dim() == 4:
            tensor = tensor[0]
        if tensor.dim() == 3:
            tensor = tensor.permute(1, 2, 0)

        numpy_image = (tensor.detach().cpu().numpy() * 255).astype(np.uint8)
        return wandb.Image(numpy_image, caption=caption)

    def _log_validation_images(self, batch: Dict[str, torch.Tensor], refined_images: torch.Tensor, step: int, max_images: int = None) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            return


        if not self.log_validation_images:
            return


        if max_images is None:
            max_images = self.max_validation_images

        try:

            batch_size = min(refined_images.shape[0], max_images)

            degraded_images = batch["degraded"][:batch_size]
            target_images = batch["target"][:batch_size]
            refined_images = refined_images[:batch_size]


            wandb_images = []
            for i in range(batch_size):
                degraded_img = self.tensor_to_wandb_image(degraded_images[i], caption=f"degraded_{i}")
                target_img = self.tensor_to_wandb_image(target_images[i], caption=f"target_{i}")
                refined_img = self.tensor_to_wandb_image(refined_images[i], caption=f"refined_{i}")

                wandb_images.extend([
                    degraded_img,
                    refined_img,
                    target_img
                ])


            log_dict = {
                "val/second_swapped_images": wandb_images,
            }


            log_dict["trainer/global_step"] = step


            self.logger.experiment.log(log_dict)

        except Exception as e:
            logger.warning(f"Failed to log validation images to wandb: {e}")

    def _log_swapped_validation_images(
        self,
        original_images: torch.Tensor,
        refined_images: torch.Tensor,
        metadata: List[Dict[str, Any]],
        global_step: int,
        max_images: int = None,
        visualize_restored_images: bool = False
    ) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            return


        if not self.log_validation_images:
            return


        if max_images is None:
            max_images = self.max_validation_images

        try:

            batch_size = min(refined_images.shape[0], max_images, len(metadata))

            original_images = original_images[:batch_size]
            refined_images = refined_images[:batch_size]
            metadata = metadata[:batch_size]


            from ..data.difix3d_datamodule import apply_reverse_adaptive_resolution_transform

            def tensor_to_wandb_image(tensor: torch.Tensor, caption: str, reverse_metadata: Dict = None) -> wandb.Image:


                tensor = torch.clamp(tensor, 0, 1).float()

                if tensor.dim() == 4:
                    tensor = tensor[0]
                if tensor.dim() == 3:
                    tensor = tensor.permute(1, 2, 0)


                numpy_image = (tensor.cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(numpy_image)


                if visualize_restored_images and reverse_metadata:
                    try:
                        pil_image = apply_reverse_adaptive_resolution_transform(pil_image, reverse_metadata)
                        caption = f"{caption}_restored"
                    except Exception as e:
                        logger.warning(f"Failed to reverse adaptive resolution for {caption}: {e}")

                return wandb.Image(pil_image, caption=caption)


            wandb_images = []
            for i in range(batch_size):

                img_metadata = metadata[i] if i < len(metadata) else {}
                swap_info = img_metadata.get("swap_info", {})
                camera_dir = img_metadata.get("camera_dir", f"cam_{i}")
                reverse_metadata = img_metadata.get("reverse_metadata")


                swap_direction = swap_info.get("swap_direction", f"swap_{i}")
                original_caption = f"original_{swap_direction}_{camera_dir}"
                refined_caption = f"refined_{swap_direction}_{camera_dir}"

                original_img = tensor_to_wandb_image(
                    original_images[i],
                    caption=original_caption,
                    reverse_metadata=reverse_metadata
                )
                refined_img = tensor_to_wandb_image(
                    refined_images[i],
                    caption=refined_caption,
                    reverse_metadata=reverse_metadata
                )

                wandb_images.extend([original_img, refined_img])


            log_dict = {
                "val/swapped_images": wandb_images,
                "trainer/global_step": global_step
            }


            swap_directions = set()
            for img_metadata in metadata[:batch_size]:
                swap_info = img_metadata.get("swap_info", {})
                if "swap_direction" in swap_info:
                    swap_directions.add(swap_info["swap_direction"])


            self.logger.experiment.log(log_dict)

        except Exception as e:
            logger.warning(f"Failed to log swapped validation images to wandb: {e}")

    def _log_training_error(self, error: Exception, batch_idx: int, step: int) -> None:

        if not hasattr(self.logger, 'experiment') or self.logger.experiment is None:
            return


        if not self.log_training_errors:
            return

        try:
            log_dict = {
                "train/error_count": 1,
                "train/error_type": type(error).__name__,
                "train/error_message": str(error)[:500],
                "train/error_batch_idx": batch_idx
            }


            log_dict["trainer/global_step"] = step


            self.logger.experiment.log(log_dict)

        except Exception as log_error:
            logger.warning(f"Failed to log training error to wandb: {log_error}")


    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:


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


        try:
            loss_dict = self.compute_loss(refined_images_tensor, target_images_tensor, degraded_images_tensor)
            total_loss = loss_dict["total_loss"]


            if "prompt" in batch:
                batch_size = len(batch["prompt"]) if isinstance(batch["prompt"], list) else 1
            elif "degraded" in batch:
                batch_size = batch["degraded"].shape[0] if hasattr(batch["degraded"], 'shape') else len(batch["degraded"])
            elif "image" in batch:
                batch_size = batch["image"].shape[0] if hasattr(batch["image"], 'shape') else len(batch["image"])
            else:
                batch_size = 1


            self.log_dict({
                "train_loss": total_loss,
                "train_l2_loss": loss_dict["l2_loss"],
                "train_lpips_loss": loss_dict.get("lpips_loss", 0.0),
                "train_gram_loss": loss_dict.get("gram_loss", 0.0),
                "lr": self.trainer.optimizers[0].param_groups[0]['lr']
            }, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)


            global_step = self.global_step
            self._log_training_metrics(loss_dict, global_step)

            return total_loss

        except Exception as e:

            if self.log_training_errors:
                self._log_training_error(e, batch_idx, self.global_step)
            logger.error(f"Training step failed: {e}")
            raise

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Dict[str, torch.Tensor]:


        is_swapped_validation = (dataloader_idx == 1) or (isinstance(batch, dict) and batch.get("is_swapped_validation", False))

        val_metrics = {}

        try:

            if is_swapped_validation:

                if isinstance(batch, dict) and batch.get("is_swapped_validation", False):
                    input_images = batch["images"]
                    prompts = batch["prompts"]
                    metadata = batch["metadata"]
                else:
                    logger.warning(f"Invalid swapped validation batch format at batch_idx {batch_idx}")
                    return {"val_swapped_processed": torch.tensor(0.0, device=self.device)}

                target_images = None
                reference_images = None
            else:

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

                batch_size = input_images.shape[0] if hasattr(input_images, 'shape') else len(input_images)
                prompts = [self.prompt] * batch_size
                metadata = batch.get("metadata", []) if isinstance(batch, dict) else []


            input_images_tensor = self._validate_and_clean_tensor(input_images, "input")
            target_images_tensor = self._validate_and_clean_tensor(target_images, "target") if target_images is not None else None
            reference_images_tensor = self._validate_and_clean_tensor(reference_images, "reference") if reference_images is not None else None


            with torch.no_grad():

                batch_size = input_images_tensor.shape[0]


                lora_modes = [self.lora_training_mode]

                for mode in lora_modes:
                    try:

                        result = self.pipe(
                            prompt=prompts,
                            image=input_images_tensor,
                            reference_image=reference_images_tensor,
                            lora_mode=mode,
                            num_inference_steps=self.num_inference_steps,
                            timesteps=self.timesteps,
                            guidance_scale=self.guidance_scale,
                            return_dict=True,
                            disable_progress_bar=True,
                            output_type="pt",
                        )

                        refined_images_tensor = self._validate_and_clean_tensor(result.images, "refined")


                        if is_swapped_validation:

                            val_metrics.update({
                                "val_swapped_processed": torch.tensor(float(batch_size), device=self.device),
                                "val_swapped_batch_idx": torch.tensor(float(batch_idx), device=self.device),
                            })


                            visualize_restored = metadata[0].get("visualize_restored_images", False) if metadata else False

                            self._log_swapped_validation_images(
                                original_images=input_images_tensor,
                                refined_images=refined_images_tensor,
                                metadata=metadata,
                                global_step=self.global_step,
                                visualize_restored_images=visualize_restored
                            )
                        else:

                            loss_dict = self.compute_loss(refined_images_tensor, target_images_tensor, input_images_tensor)


                            psnr = self.compute_psnr(refined_images_tensor, target_images_tensor)
                            ssim = self.compute_ssim(refined_images_tensor, target_images_tensor) if hasattr(self, 'compute_ssim') else 0.0


                            val_metrics.update({
                                "val_loss": loss_dict["total_loss"],
                                "val_l2_loss": loss_dict["l2_loss"],
                                "val_lpips_loss": loss_dict.get("lpips_loss", torch.tensor(0.0, device=self.device)),
                                "val_gram_loss": loss_dict.get("gram_loss", torch.tensor(0.0, device=self.device)),
                                "val_psnr": psnr,
                                "val_ssim": torch.tensor(ssim, device=self.device),
                                f"val_loss_{mode}": loss_dict["total_loss"],
                                f"val_psnr_{mode}": psnr,
                            })


                            if 'refined_images_tensor' in locals() and refined_images_tensor is not None:
                                batch_dict = {
                                    "degraded": input_images_tensor,
                                    "target": target_images_tensor
                                }
                                self._log_validation_images(batch_dict, refined_images_tensor, self.global_step)

                        break

                    except Exception as mode_error:
                        logger.warning(f"Validation failed for mode {mode}: {mode_error}")
                        if is_swapped_validation:
                            val_metrics.update({
                                "val_swapped_processed": torch.tensor(0.0, device=self.device),
                                "val_swapped_error": torch.tensor(1.0, device=self.device),
                            })
                        else:
                            val_metrics.update({
                                "val_loss": torch.tensor(float('inf'), device=self.device),
                                "val_l2_loss": torch.tensor(0.0, device=self.device),
                                "val_lpips_loss": torch.tensor(0.0, device=self.device),
                                "val_gram_loss": torch.tensor(0.0, device=self.device),
                                "val_psnr": torch.tensor(0.0, device=self.device),
                                "val_ssim": torch.tensor(0.0, device=self.device),
                                f"val_loss_{mode}": torch.tensor(float('inf'), device=self.device),
                                f"val_psnr_{mode}": torch.tensor(0.0, device=self.device),
                            })

        except Exception as e:
            logger.error(f"Validation step failed at batch_idx {batch_idx}: {e}")
            if is_swapped_validation:
                val_metrics = {
                    "val_swapped_processed": torch.tensor(0.0, device=self.device),
                    "val_swapped_error": torch.tensor(1.0, device=self.device),
                }
            else:
                val_metrics = {
                    "val_loss": torch.tensor(1.0, device=self.device),
                    "val_l2_loss": torch.tensor(1.0, device=self.device),
                    "val_lpips_loss": torch.tensor(0.0, device=self.device),
                    "val_gram_loss": torch.tensor(0.0, device=self.device),
                    "val_psnr": torch.tensor(10.0, device=self.device),
                    "val_ssim": torch.tensor(0.5, device=self.device),
                }


        if val_metrics:

            batch_size = input_images_tensor.shape[0] if 'input_images_tensor' in locals() else 1

            if is_swapped_validation:

                prefixed_metrics = {f"swapped_{k}": v for k, v in val_metrics.items()}
                self.log_dict(prefixed_metrics, prog_bar=False, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)


                return None

            else:

                self.log_dict(val_metrics, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

                self._log_validation_metrics(val_metrics, self.global_step)
                return val_metrics


        if is_swapped_validation:
            logger.debug("Swapped validation with empty metrics, returning None")
            return None
        else:
            logger.warning("Training validation with empty metrics, returning fallback")
            return val_metrics or {}


    def configure_optimizers(self):

        if not self.training_mode:

            return torch.optim.Adam(self.parameters(), lr=1e-4)


        trainable_params = [p for p in self.parameters() if p.requires_grad]

        logger.info(f"Total trainable parameters discovered by Lightning: {sum(p.numel() for p in trainable_params):,}")


        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        if not self.use_cosine_scheduler:
            return optimizer


        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches,
            eta_min=self.scheduler_eta_min
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "cosine_annealing_lr"
            }
        }


    def compute_loss(self, refined_images: torch.Tensor, target_images: torch.Tensor,
                     degraded_images: torch.Tensor) -> Dict[str, torch.Tensor]:


        loss_dict = {}


        l2_loss = F.mse_loss(refined_images, target_images)
        loss_dict["l2_loss"] = l2_loss


        if self.lpips_fn is not None and self.lpips_weight > 0:
            try:


                if torch.isnan(refined_images).any() or torch.isinf(refined_images).any():
                    logger.warning(f"NaN/inf detected in refined_images for LPIPS")
                    logger.warning(f"  refined_images stats: min={refined_images.min():.4f}, max={refined_images.max():.4f}, nan_count={torch.isnan(refined_images).sum()}")
                    refined_clean = torch.nan_to_num(refined_images, nan=0.0, posinf=1.0, neginf=0.0)
                else:
                    refined_clean = refined_images

                if torch.isnan(target_images).any() or torch.isinf(target_images).any():
                    logger.warning(f"NaN/inf detected in target_images for LPIPS")
                    logger.warning(f"  target_images stats: min={target_images.min():.4f}, max={target_images.max():.4f}, nan_count={torch.isnan(target_images).sum()}")
                    target_clean = torch.nan_to_num(target_images, nan=0.0, posinf=1.0, neginf=0.0)
                else:
                    target_clean = target_images


                if not hasattr(self, '_lpips_debug_printed'):
                    logger.info(f"LPIPS model device: {next(self.lpips_fn.parameters()).device}")
                    logger.info(f"Input tensors device: refined={refined_clean.device}, target={target_clean.device}")
                    print(f"LPIPS model training mode: {self.lpips_fn.training}")
                    self._lpips_debug_printed = True


                lpips_raw = self.lpips_fn(refined_clean, target_clean)


                if torch.isnan(lpips_raw).any() or torch.isinf(lpips_raw).any():
                    print("Warning: NaN/inf detected in LPIPS output, setting LPIPS weight to zero")
                    print(f"  lpips_raw stats: shape={lpips_raw.shape}, min={lpips_raw.min():.4f}, max={lpips_raw.max():.4f}")
                    print(f"  nan_count={torch.isnan(lpips_raw).sum()}, inf_count={torch.isinf(lpips_raw).sum()}")
                    print("  This often happens with multiprocessing - investigating...")
                    lpips_loss = lpips_raw.mean()
                    self.lpips_weight = 0.0
                else:
                    lpips_loss = lpips_raw.mean()

                    if torch.isnan(lpips_loss) or torch.isinf(lpips_loss):
                        print("Warning: NaN/inf in LPIPS mean, setting LPIPS weight to zero")
                        self.lpips_weight = 0.0

                loss_dict["lpips_loss"] = lpips_loss

            except Exception as e:
                print(f"Warning: LPIPS loss computation failed: {e}, setting LPIPS weight to zero")

                loss_dict["lpips_loss"] = F.mse_loss(refined_images, target_images)
                self.lpips_weight = 0.0
        else:

            loss_dict["lpips_loss"] = F.mse_loss(refined_images, target_images) * 0.0


        if self.vgg_features is not None and self.gram_weight > 0:
            try:
                gram_loss = self.compute_gram_loss(refined_images, target_images)


                if torch.isnan(gram_loss) or torch.isinf(gram_loss):
                    print("Warning: NaN/inf detected in Gram loss, setting Gram weight to zero")
                    self.gram_weight = 0.0

                loss_dict["gram_loss"] = gram_loss

            except Exception as e:
                print(f"Warning: Gram loss computation failed: {e}, setting Gram weight to zero")

                loss_dict["gram_loss"] = F.mse_loss(refined_images, target_images)
                self.gram_weight = 0.0
        else:

            loss_dict["gram_loss"] = F.mse_loss(refined_images, target_images) * 0.0


        total_loss = (
            self.l2_weight * loss_dict["l2_loss"] +
            self.lpips_weight * loss_dict["lpips_loss"] +
            self.gram_weight * loss_dict["gram_loss"]
        )


        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("Warning: NaN/inf detected in total_loss, using L2 loss only")
            total_loss = self.l2_weight * loss_dict["l2_loss"]

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Critical: L2 loss also has NaN/inf, this should not happen with normalized inputs")

                total_loss = F.mse_loss(refined_images, refined_images) + 1e-6

        loss_dict["total_loss"] = total_loss

        return loss_dict

    def compute_gram_loss(self, refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        def gram_matrix(features):

            b, c, h, w = features.size()


            if torch.isnan(features).any() or torch.isinf(features).any():
                logger.warning("NaN/inf detected in VGG features, cleaning...")
                features = torch.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            features = features.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))


            normalizer = c * h * w
            if normalizer > 0:
                gram = gram / normalizer
            else:
                logger.warning("Zero normalizer in Gram matrix, using fallback")
                gram = gram * 0.0

            return gram


        refined_features = self.vgg_features(refined)
        target_features = self.vgg_features(target)


        refined_gram = gram_matrix(refined_features)
        target_gram = gram_matrix(target_features)


        if torch.isnan(refined_gram).any() or torch.isinf(refined_gram).any():
            logger.warning("NaN/inf in refined Gram matrix")
            refined_gram = torch.nan_to_num(refined_gram, nan=0.0)

        if torch.isnan(target_gram).any() or torch.isinf(target_gram).any():
            logger.warning("NaN/inf in target Gram matrix")
            target_gram = torch.nan_to_num(target_gram, nan=0.0)


        gram_loss = F.mse_loss(refined_gram, target_gram)

        return gram_loss

    def compute_psnr(self, refined: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        mse = F.mse_loss(refined, target)
        if mse == 0:
            return torch.tensor(float('inf'), device=refined.device)

        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def _validate_and_clean_tensor(self, images: torch.Tensor, name: str = "tensor") -> torch.Tensor:

        if not isinstance(images, torch.Tensor):
            raise ValueError(f"Expected tensor input, got {type(images)}")


        if torch.isnan(images).any() or torch.isinf(images).any():
            logger.warning(f"NaN/inf detected in {name}, cleaning...")
            images = torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0)


        images = images.clamp(0, 1)

        return images

    def on_train_start(self) -> None:

        if self.training_mode:
            logger.info("Starting DiFix3D+ SwapFix LoRA training...")
            logger.info(f"Training parameters: lr={self.learning_rate}, rank={self.swapfix_lora_rank}")
            logger.info(f"Loss weights: L2={self.l2_weight}, LPIPS={self.lpips_weight}, Gram={self.gram_weight}")

    def on_train_end(self) -> None:

        if self.training_mode:
            logger.info("DiFix3D+ SwapFix LoRA training completed!")


            self._save_dual_lora_checkpoint()

    def on_predict_start(self) -> None:

        logger.info("Starting DiFix3D+ batch prediction...")

    def on_predict_end(self) -> None:

        logger.info("DiFix3D+ batch prediction completed!")


        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _diagnose_pipeline_state(self, context: str) -> None:

        import os
        import threading

        worker_info = torch.utils.data.get_worker_info()
        process_id = os.getpid()
        thread_id = threading.get_ident()

        logger.info(f"=== PIPELINE DIAGNOSTICS ({context}) ===")
        logger.info(f"Process ID: {process_id}")
        logger.info(f"Thread ID: {thread_id}")

        if worker_info is not None:
            logger.info(f"DataLoader Worker ID: {worker_info.id}")
            logger.info(f"Worker Dataset: {type(worker_info.dataset).__name__}")
            logger.warning("RUNNING IN DATALOADER WORKER - This may cause issues!")
        else:
            logger.info("Running in main process (no DataLoader worker)")


        if hasattr(self, 'pipe') and self.pipe is not None:
            try:

                logger.info(f"Pipeline type: {type(self.pipe).__name__}")
                logger.info(f"Pipeline device: {self.pipe.device}")


                if hasattr(self.pipe, 'unet'):
                    unet_params = sum(p.numel() for p in self.pipe.unet.parameters())
                    unet_device = next(self.pipe.unet.parameters()).device
                    logger.info(f"UNet parameters: {unet_params:,}")
                    logger.info(f"UNet device: {unet_device}")
                    logger.info(f"UNet training mode: {self.pipe.unet.training}")


                    first_param = next(self.pipe.unet.parameters())
                    param_stats = {
                        'min': first_param.min().item(),
                        'max': first_param.max().item(),
                        'mean': first_param.mean().item(),
                        'std': first_param.std().item()
                    }
                    logger.info(f"UNet first param stats: {param_stats}")


                    if abs(param_stats['std']) < 1e-8:
                        logger.error("UNet parameters appear corrupted (no variation)!")


                    if hasattr(self.pipe.unet, 'peft_config'):
                        unet_lora_configs = list(self.pipe.unet.peft_config.keys())
                        logger.info(f"UNet LoRA adapters: {unet_lora_configs}")
                    else:
                        logger.info("No LoRA adapters found in UNet")


                if hasattr(self.pipe, 'vae'):
                    vae_params = sum(p.numel() for p in self.pipe.vae.parameters())
                    vae_device = next(self.pipe.vae.parameters()).device
                    logger.info(f"VAE parameters: {vae_params:,}")
                    logger.info(f"VAE device: {vae_device}")
                    logger.info(f"VAE training mode: {self.pipe.vae.training}")


                    if hasattr(self.pipe.vae, 'decoder'):
                        logger.info(f"VAE decoder training mode: {self.pipe.vae.decoder.training}")


                        if hasattr(self.pipe.vae, 'peft_config'):
                            lora_configs = list(self.pipe.vae.peft_config.keys())
                            logger.info(f"VAE LoRA adapters: {lora_configs}")
                        else:
                            logger.warning("No LoRA adapters found in VAE decoder")


                if hasattr(self.pipe, 'scheduler'):
                    logger.info(f"Scheduler type: {type(self.pipe.scheduler).__name__}")
                    if hasattr(self.pipe.scheduler, 'timesteps'):
                        logger.info(f"Scheduler timesteps: {len(self.pipe.scheduler.timesteps)}")


                if hasattr(self.pipe, 'model_type'):
                    logger.info(f"Pipeline model type: {self.pipe.model_type}")
                if hasattr(self.pipe, 'lora_mode'):
                    logger.info(f"Pipeline LoRA mode: {self.pipe.lora_mode}")

            except Exception as e:
                logger.error(f"Error during pipeline diagnostics: {e}")
                logger.error(f"Pipeline may be in corrupted state!")
        else:
            logger.error("Pipeline is None or not initialized!")

        logger.info("=== END PIPELINE DIAGNOSTICS ===")

    def _diagnose_refined_images(self, refined_images: torch.Tensor, context: str) -> None:

        logger.info(f"=== REFINED IMAGE DIAGNOSTICS ({context}) ===")


        logger.info(f"Refined images shape: {refined_images.shape}")
        logger.info(f"Refined images dtype: {refined_images.dtype}")
        logger.info(f"Refined images device: {refined_images.device}")


        stats = {
            'min': refined_images.min().item(),
            'max': refined_images.max().item(),
            'mean': refined_images.mean().item(),
            'std': refined_images.std().item()
        }
        logger.info(f"Refined images stats: {stats}")


        num_zeros = (refined_images == 0).sum().item()
        num_ones = (refined_images == 1).sum().item()
        total_pixels = refined_images.numel()

        zero_percentage = (num_zeros / total_pixels) * 100
        ones_percentage = (num_ones / total_pixels) * 100

        logger.info(f"Zero pixels: {num_zeros:,} ({zero_percentage:.2f}%)")
        logger.info(f"One pixels: {num_ones:,} ({ones_percentage:.2f}%)")


        if zero_percentage > 90:
            logger.error(f"BLACK IMAGE DETECTED! {zero_percentage:.1f}% pixels are zero")
            logger.error("This indicates pipeline corruption in multiprocessing!")


        if ones_percentage > 90:
            logger.error(f"SATURATED IMAGE DETECTED! {ones_percentage:.1f}% pixels are one")


        if stats['std'] < 1e-6:
            logger.error("No variation in refined images - likely corrupted!")


        if refined_images.shape[1] == 3:
            for c, channel_name in enumerate(['R', 'G', 'B']):
                channel_data = refined_images[:, c, :, :]
                channel_mean = channel_data.mean().item()
                channel_std = channel_data.std().item()
                logger.info(f"Channel {channel_name}: mean={channel_mean:.4f}, std={channel_std:.4f}")

        logger.info("=== END REFINED IMAGE DIAGNOSTICS ===")

    def _save_dual_lora_checkpoint(self) -> None:

        try:
            save_path = Path(self.trainer.log_dir) / "final_dual_lora_model"
            save_path.mkdir(exist_ok=True)


            self.pipe.save_pretrained(save_path)

            logger.info(f"Dual LoRA model saved to: {save_path}")

        except Exception as e:
            logger.warning(f"Could not save dual LoRA checkpoint: {e}")


cs = ConfigStore.instance()
cs.store(name="difix3d_config", node=DiFix3DConfig)
cs.store(group="training", name="swapfix", node=TrainingConfig)
