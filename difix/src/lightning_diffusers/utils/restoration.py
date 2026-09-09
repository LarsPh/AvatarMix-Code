import torch
from typing import Dict
from loguru import logger


def restore_portrait_to_fullbody(
    portrait: torch.Tensor,
    fullbody_original: torch.Tensor,
    bbox_metadata: Dict
) -> torch.Tensor:

    x = int(bbox_metadata['x'])
    y = int(bbox_metadata['y'])
    size = int(bbox_metadata['size'])


    fullbody_restored = fullbody_original.clone()


    fullbody_restored[:, y:y+size, x:x+size] = portrait

    return fullbody_restored
