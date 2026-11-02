from pathlib import Path
import re

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


FORMAT = "avatarmix-refiner-v1"
VARIANTS = {"seamfix", "fullbodyfix"}


def is_refiner_weight(key: str) -> bool:
    if key.startswith("pipe_unet."):
        return ".lora_" in key and "unet_swapfix" in key
    if key.startswith("pipe_vae."):
        return (".lora_" in key and ("vae_swapfix" in key or "vae_skip" in key)) or key.startswith(
            "pipe_vae.decoder.skip_conv_"
        )
    return False


def read_adapter_metadata(path: str | Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        metadata = reader.metadata() or {}
    if metadata.get("format") != FORMAT:
        raise ValueError(f"Unsupported AvatarMix adapter format: {path}")
    if metadata.get("variant") not in VARIANTS:
        raise ValueError(f"Unknown AvatarMix adapter variant: {path}")
    if metadata.get("base_model") != "nvidia/difix" or not re.fullmatch(
        r"[0-9a-f]{40}", metadata.get("base_revision", "")
    ):
        raise ValueError(f"Missing pinned Difix base model: {path}")
    return metadata


def export_adapter(
    checkpoint: str | Path, output: str | Path, variant: str, base_revision: str
) -> dict[str, int]:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    if not re.fullmatch(r"[0-9a-f]{40}", base_revision):
        raise ValueError(f"Expected a 40-character base revision, got {base_revision}")
    source = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    state = source["state_dict"]
    weights = {key: value.contiguous() for key, value in state.items() if is_refiner_weight(key)}
    counts = {
        "unet_lora": sum(key.startswith("pipe_unet.") for key in weights),
        "vae_lora": sum("vae_swapfix" in key for key in weights),
        "vae_skip_lora": sum("vae_skip" in key for key in weights),
        "vae_skip_conv": sum("skip_conv" in key and ".lora_" not in key for key in weights),
    }
    if not all(counts.values()):
        raise ValueError(f"Incomplete refiner weights in {checkpoint}: {counts}")
    save_file(
        weights,
        str(output),
        metadata={
            "format": FORMAT,
            "variant": variant,
            "base_model": "nvidia/difix",
            "base_revision": base_revision,
        },
    )
    return counts


def load_adapter(model: torch.nn.Module, path: str | Path, variant: str) -> None:
    metadata = read_adapter_metadata(path)
    if metadata["variant"] != variant or metadata["base_model"] != model.model_name:
        raise ValueError(f"Adapter does not match {variant} / {model.model_name}: {path}")
    weights = load_file(str(path), device="cpu")
    expected = {key: value for key, value in model.state_dict().items() if is_refiner_weight(key)}
    if weights.keys() != expected.keys():
        missing = sorted(expected.keys() - weights.keys())
        extra = sorted(weights.keys() - expected.keys())
        raise ValueError(f"Adapter key mismatch: missing={missing[:5]}, extra={extra[:5]}")
    for key, value in weights.items():
        if value.shape != expected[key].shape or value.dtype != expected[key].dtype:
            raise ValueError(f"Adapter tensor mismatch: {key}")
    model.load_state_dict(weights, strict=False)
