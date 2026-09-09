import torch
import torch.nn as nn
from peft import LoraConfig
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


def my_vae_encoder_fwd(self, sample):

    sample = self.conv_in(sample)
    l_blocks = []

    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)

    sample = self.mid_block(sample)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd(self, sample, latent_embeds=None):

    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]

        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)

            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    return sample


class SerializableAutoencoderKL(AutoencoderKL):


    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types = ("DownEncoderBlock2D",),
        up_block_types = ("UpDecoderBlock2D",),
        block_out_channels = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,

        lora_rank: int = 4,
        gamma: float = 1.0,
        ignore_skip: bool = False,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
        )


        self.decoder.skip_conv_1 = torch.nn.Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.decoder.skip_conv_2 = torch.nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.decoder.skip_conv_3 = torch.nn.Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.decoder.skip_conv_4 = torch.nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.decoder.ignore_skip = ignore_skip
        self.decoder.gamma = gamma


        target_modules_vae = ["conv1", "conv2", "conv_in", "conv_shortcut", "conv", "conv_out",
            "skip_conv_1", "skip_conv_2", "skip_conv_3", "skip_conv_4",
            "to_k", "to_q", "to_v", "to_out.0",
        ]
        target_modules = []
        for id, (name, param) in enumerate(self.named_modules()):
            if 'decoder' in name and any(name.endswith(x) for x in target_modules_vae):
                target_modules.append(name)
        target_modules_vae = target_modules

        vae_lora_config = LoraConfig(r=lora_rank, init_lora_weights="gaussian", target_modules=target_modules_vae)
        self.add_adapter(vae_lora_config, adapter_name="vae_skip")


        self._methods_bound = False

    def _ensure_methods_bound(self):

        if self._methods_bound:
            return


        self.encoder.forward = my_vae_encoder_fwd.__get__(self.encoder, self.encoder.__class__)
        self.decoder.forward = my_vae_decoder_fwd.__get__(self.decoder, self.decoder.__class__)

        self._methods_bound = True

    def encode(self, x, return_dict=True):

        self._ensure_methods_bound()
        return super().encode(x, return_dict)

    def _decode(self, z, return_dict=True):

        self._ensure_methods_bound()
        return super()._decode(z, return_dict)

    def tiled_encode(self, x, return_dict=True):

        self._ensure_methods_bound()
        return super().tiled_encode(x, return_dict)

    def tiled_decode(self, z, return_dict=True):

        self._ensure_methods_bound()
        return super().tiled_decode(z, return_dict)

    def set_adapters(
        self,
        adapter_names,
        adapter_weights = None,
    ):

        from typing import List, Optional, Union
        from diffusers.utils import USE_PEFT_BACKEND, set_weights_and_activate_adapters

        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")


        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]


        if adapter_weights is None:
            adapter_weights = [1.0] * len(adapter_names)
        elif isinstance(adapter_weights, (int, float)):
            adapter_weights = [float(adapter_weights)] * len(adapter_names)


        if len(adapter_names) != len(adapter_weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(adapter_weights)}."
            )


        set_weights_and_activate_adapters(self, adapter_names, adapter_weights)
