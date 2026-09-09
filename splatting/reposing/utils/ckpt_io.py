from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import re

import torch


@dataclass(frozen=True)
class DeformCheckpointInfo:
    ckpt_path: str
    iteration: Optional[int]
    gs_model_state_dict: Dict[str, Any]
    latent_num_embeddings: Optional[int]
    latent_dim: Optional[int]


def load_deform_checkpoint(ckpt_path: str, *, map_location: str | torch.device = "cpu") -> DeformCheckpointInfo:

    ckpt_path = str(ckpt_path)


    try:
        obj = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except TypeError:

        obj = torch.load(ckpt_path, map_location=map_location)
    if not isinstance(obj, dict):
        raise ValueError(f"Checkpoint must be a dict, got: {type(obj)}")

    state = obj.get("gs_model_state_dict", None)
    if state is None or not isinstance(state, dict):
        raise ValueError(f"Checkpoint missing 'gs_model_state_dict': keys={list(obj.keys())}")

    it = obj.get("iteration", None)
    if it is not None:
        try:
            it = int(it)
        except Exception:
            it = None

    n_embed, dim = _infer_latent_embedding_shape_from_state_dict(state)
    return DeformCheckpointInfo(
        ckpt_path=ckpt_path,
        iteration=it,
        gs_model_state_dict=state,
        latent_num_embeddings=n_embed,
        latent_dim=dim,
    )


def _infer_latent_embedding_shape_from_state_dict(state_dict: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:

    key = None
    for k in state_dict.keys():
        if k.endswith("frame_latent_codes.weight"):
            key = k
            break
    if key is None:
        return None, None
    w = state_dict.get(key)
    if not isinstance(w, torch.Tensor) or w.ndim != 2:
        return None, None
    return int(w.shape[0]), int(w.shape[1])


_REF_FRAME_RE = re.compile(r"_(\d{3,})$")


def infer_ref_frame_idx_from_pretrained_gs_path(pretrained_gs_from: str) -> Optional[int]:

    p = Path(str(pretrained_gs_from))

    names = [p.name] + [pp.name for pp in p.parents]
    for nm in names:
        m = _REF_FRAME_RE.search(nm)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None
