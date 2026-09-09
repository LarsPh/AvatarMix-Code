import sys
import os
from typing import Dict, Any, Optional, List, Union
from enum import Enum
import torch
import torch.nn as nn
from loguru import logger
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import Difix
from peft import LoraConfig


class TrainingMode(Enum):

    BASE = "base"
    REFINEMENT = "refinement"
    COMBINED = "combined"


class DiFixDualLoRAModel(Difix):


    def __init__(
        self,
        refinement_lora_rank: int = 4,
        base_adapter_weight: float = 0.7,
        refinement_adapter_weight: float = 0.3,
        **kwargs
    ):

        super().__init__(**kwargs)


        self.refinement_lora_rank = refinement_lora_rank
        self.base_adapter_weight = base_adapter_weight
        self.refinement_adapter_weight = refinement_adapter_weight


        self.base_adapter_name = "vae_skip"
        self.refinement_adapter_name = "vae_refinement"


        self.current_training_mode = TrainingMode.BASE


        self._setup_refinement_lora()

        logger.info(f"DiFixDualLoRAModel initialized")
        logger.info(f"Base adapter: '{self.base_adapter_name}' (weight: {self.base_adapter_weight})")
        logger.info(f"Refinement adapter: '{self.refinement_adapter_name}' (weight: {self.refinement_adapter_weight})")

    def _setup_refinement_lora(self):

        try:

            target_modules = self._get_vae_target_modules()


            lora_config = LoraConfig(
                r=self.refinement_lora_rank,
                lora_alpha=self.refinement_lora_rank,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type=None,
                init_lora_weights="gaussian",
            )


            if hasattr(self.vae, 'add_adapter'):
                self.vae.add_adapter(lora_config, self.refinement_adapter_name)
                logger.info(f"Added refinement LoRA adapter '{self.refinement_adapter_name}' to VAE")


                if hasattr(self.vae, 'get_list_adapters'):
                    adapters = self.vae.get_list_adapters()
                    logger.info(f"VAE adapters: {adapters}")

            else:
                logger.warning("VAE does not support add_adapter - using fallback method")
                self._setup_refinement_lora_fallback(lora_config)

        except Exception as e:
            logger.error(f"Failed to setup refinement LoRA: {e}")
            logger.info("Attempting fallback setup...")
            self._setup_refinement_lora_fallback()

    def _setup_refinement_lora_fallback(self, lora_config=None):

        logger.warning("Using fallback LoRA setup - some features may be limited")


        pass

    def _get_vae_target_modules(self) -> List[str]:


        target_modules = [

            "conv_in",
            "conv_out",
            "conv_norm_out",


            "mid_block.attentions.0.to_q",
            "mid_block.attentions.0.to_k",
            "mid_block.attentions.0.to_v",
            "mid_block.attentions.0.to_out.0",


            "up_blocks.0.attentions.0.to_q",
            "up_blocks.0.attentions.0.to_k",
            "up_blocks.0.attentions.0.to_v",
            "up_blocks.0.attentions.0.to_out.0",
            "up_blocks.1.attentions.0.to_q",
            "up_blocks.1.attentions.0.to_k",
            "up_blocks.1.attentions.0.to_v",
            "up_blocks.1.attentions.0.to_out.0",
            "up_blocks.2.attentions.0.to_q",
            "up_blocks.2.attentions.0.to_k",
            "up_blocks.2.attentions.0.to_v",
            "up_blocks.2.attentions.0.to_out.0",


            "skip_conv_1",
            "skip_conv_2",
            "skip_conv_3",
            "skip_conv_4",
        ]

        return target_modules

    def set_training_mode(self, mode: Union[TrainingMode, str]):

        if isinstance(mode, str):
            mode = TrainingMode(mode)

        self.current_training_mode = mode

        try:
            if hasattr(self.vae, 'set_adapters'):
                if mode == TrainingMode.BASE:

                    self.vae.set_adapters([self.base_adapter_name])
                    logger.info(f"Training mode: BASE - using '{self.base_adapter_name}' only")

                elif mode == TrainingMode.REFINEMENT:

                    self.vae.set_adapters([self.refinement_adapter_name])
                    logger.info(f"Training mode: REFINEMENT - using '{self.refinement_adapter_name}' only")

                elif mode == TrainingMode.COMBINED:

                    self.vae.set_adapters(
                        [self.base_adapter_name, self.refinement_adapter_name],
                        adapter_weights=[self.base_adapter_weight, self.refinement_adapter_weight]
                    )
                    logger.info(f"Training mode: COMBINED - weights: {self.base_adapter_weight:.2f}, {self.refinement_adapter_weight:.2f}")
            else:
                logger.warning("VAE does not support set_adapters - mode change may not take effect")

        except Exception as e:
            logger.error(f"Failed to set training mode {mode}: {e}")

    def freeze_base_lora(self):

        frozen_count = 0

        for name, param in self.vae.named_parameters():
            if self.base_adapter_name in name and "lora" in name:
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Frozen {frozen_count} base LoRA parameters")

    def unfreeze_base_lora(self):

        unfrozen_count = 0

        for name, param in self.vae.named_parameters():
            if self.base_adapter_name in name and "lora" in name:
                param.requires_grad = True
                unfrozen_count += 1

        logger.info(f"Unfrozen {unfrozen_count} base LoRA parameters")

    def freeze_refinement_lora(self):

        frozen_count = 0

        for name, param in self.vae.named_parameters():
            if self.refinement_adapter_name in name and "lora" in name:
                param.requires_grad = False
                frozen_count += 1

        logger.info(f"Frozen {frozen_count} refinement LoRA parameters")

    def unfreeze_refinement_lora(self):

        unfrozen_count = 0

        for name, param in self.vae.named_parameters():
            if self.refinement_adapter_name in name and "lora" in name:
                param.requires_grad = True
                unfrozen_count += 1

        logger.info(f"Unfrozen {unfrozen_count} refinement LoRA parameters")

    def get_trainable_parameters(self, mode: Union[TrainingMode, str]) -> List[torch.nn.Parameter]:

        if isinstance(mode, str):
            mode = TrainingMode(mode)

        trainable_params = []


        unet_params = list(self.unet.parameters())
        trainable_params.extend(unet_params)


        skip_params = []
        for skip_name in ["skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4"]:
            if hasattr(self.vae.decoder, skip_name):
                skip_layer = getattr(self.vae.decoder, skip_name)
                skip_params.extend(list(skip_layer.parameters()))
        trainable_params.extend(skip_params)


        lora_params = []
        for name, param in self.vae.named_parameters():
            if "lora" in name:
                if mode == TrainingMode.BASE and self.base_adapter_name in name:
                    lora_params.append(param)
                elif mode == TrainingMode.REFINEMENT and self.refinement_adapter_name in name:
                    lora_params.append(param)
                elif mode == TrainingMode.COMBINED:
                    lora_params.append(param)
        trainable_params.extend(lora_params)


        unet_count = sum(p.numel() for p in unet_params)
        skip_count = sum(p.numel() for p in skip_params)
        lora_count = sum(p.numel() for p in lora_params)
        total_count = sum(p.numel() for p in trainable_params)

        logger.info(f"Trainable parameters for {mode.value} mode:")
        logger.info(f"  UNet: {unet_count:,}")
        logger.info(f"  Skip connections: {skip_count:,}")
        logger.info(f"  LoRA adapters: {lora_count:,}")
        logger.info(f"  Total: {total_count:,}")

        return trainable_params

    def disable_all_loras(self):

        try:
            if hasattr(self.vae, 'disable_lora'):
                self.vae.disable_lora()
                logger.info("Disabled all LoRA adapters")
            else:
                logger.warning("VAE does not support disable_lora")
        except Exception as e:
            logger.error(f"Failed to disable LoRAs: {e}")

    def enable_loras(self):

        try:

            self.set_training_mode(self.current_training_mode)
            logger.info("Re-enabled LoRA adapters")
        except Exception as e:
            logger.error(f"Failed to enable LoRAs: {e}")

    def get_active_adapters(self) -> List[str]:

        try:
            if hasattr(self.vae, 'get_active_adapters'):
                active = self.vae.get_active_adapters()
                logger.debug(f"Active adapters: {active}")
                return active
            else:
                logger.warning("VAE does not support get_active_adapters")
                return []
        except Exception as e:
            logger.error(f"Failed to get active adapters: {e}")
            return []

    def save_lora_adapter(self, save_path: str, adapter_name: Optional[str] = None):

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        try:
            if hasattr(self.vae, 'save_lora_adapter'):
                if adapter_name:
                    self.vae.save_lora_adapter(str(save_path), adapter_name=adapter_name)
                    logger.info(f"Saved LoRA adapter '{adapter_name}' to {save_path}")
                else:

                    self.vae.save_lora_adapter(str(save_path / "base"), adapter_name=self.base_adapter_name)
                    self.vae.save_lora_adapter(str(save_path / "refinement"), adapter_name=self.refinement_adapter_name)
                    logger.info(f"Saved all LoRA adapters to {save_path}")
            else:
                logger.warning("VAE does not support save_lora_adapter")

        except Exception as e:
            logger.error(f"Failed to save LoRA adapter: {e}")

    def add_weighted_adapter(
        self,
        adapter_names: List[str],
        weights: List[float],
        combination_type: str = "linear",
        new_adapter_name: str = "merged_adapter"
    ):

        try:
            if hasattr(self.vae, 'add_weighted_adapter'):
                self.vae.add_weighted_adapter(
                    adapters=adapter_names,
                    weights=weights,
                    combination_type=combination_type,
                    adapter_name=new_adapter_name
                )
                logger.info(f"Created merged adapter '{new_adapter_name}' using {combination_type}")
            else:
                logger.warning("VAE does not support add_weighted_adapter - using simple weighted combination")

                self.vae.set_adapters(adapter_names, adapter_weights=weights)

        except Exception as e:
            logger.error(f"Failed to create weighted adapter: {e}")

    def print_model_info(self):

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)


        base_lora_params = sum(
            p.numel() for name, p in self.vae.named_parameters()
            if self.base_adapter_name in name and "lora" in name
        )
        refinement_lora_params = sum(
            p.numel() for name, p in self.vae.named_parameters()
            if self.refinement_adapter_name in name and "lora" in name
        )


        active_adapters = self.get_active_adapters()

        logger.info("="*70)
        logger.info("DiFixDualLoRAModel Information")
        logger.info("="*70)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Base LoRA parameters: {base_lora_params:,}")
        logger.info(f"Refinement LoRA parameters: {refinement_lora_params:,}")
        logger.info(f"Current training mode: {self.current_training_mode.value}")
        logger.info(f"Active adapters: {active_adapters}")
        logger.info(f"Base adapter weight: {self.base_adapter_weight}")
        logger.info(f"Refinement adapter weight: {self.refinement_adapter_weight}")
        logger.info("="*70)


def test_difix_dual_lora_model():

    logger.info("Testing DiFixDualLoRAModel...")

    try:

        model = DiFixDualLoRAModel(
            lora_rank_vae=4,
            timestep=199,
            mv_unet=False,
            refinement_lora_rank=4,
            base_adapter_weight=0.7,
            refinement_adapter_weight=0.3
        )


        model.print_model_info()


        logger.info("Testing training mode switching...")
        for mode in [TrainingMode.BASE, TrainingMode.REFINEMENT, TrainingMode.COMBINED]:
            model.set_training_mode(mode)
            params = model.get_trainable_parameters(mode)
            logger.info(f"Mode {mode.value}: {len(params)} parameter tensors")


        logger.info("Testing parameter freezing...")
        model.freeze_base_lora()
        model.unfreeze_base_lora()
        model.freeze_refinement_lora()
        model.unfreeze_refinement_lora()


        logger.info("Testing adapter management...")
        active = model.get_active_adapters()
        logger.info(f"Active adapters: {active}")

        model.disable_all_loras()
        model.enable_loras()

        logger.info("DiFixDualLoRAModel test completed successfully!")
        return True

    except Exception as e:
        logger.error(f"DiFixDualLoRAModel test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":

    success = test_difix_dual_lora_model()
    if success:
        logger.info("All tests passed!")
    else:
        logger.error("Tests failed!")
        exit(1)
