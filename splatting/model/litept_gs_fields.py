from __future__ import annotations

import torch

from utils.data_utils import retrieve_verts_barycentric


def interp_token_field_to_gs(
    *,
    field_vc: torch.Tensor,
    faces_f3: torch.Tensor,
    sample_fidxs_ng: torch.Tensor,
    sample_bary_ng3: torch.Tensor,
) -> torch.Tensor:
    """
    Interpolate a per-vertex token field to mesh-attached Gaussians using existing barycentric metadata.

    Args:
        field_vc: (Nv,C) token field on canonical mesh vertices.
        faces_f3: (F,3) mesh faces.
        sample_fidxs_ng: (Ng,) face index per Gaussian.
        sample_bary_ng3: (Ng,3) barycentric weights per Gaussian on that face.
    Returns:
        (Ng,C) interpolated field at Gaussians.
    """
    if not isinstance(field_vc, torch.Tensor) or field_vc.ndim != 2:
        raise ValueError(f"field_vc must be Tensor [Nv,C], got {type(field_vc)} shape={getattr(field_vc, 'shape', None)}")
    if not isinstance(faces_f3, torch.Tensor) or faces_f3.ndim != 2 or faces_f3.shape[-1] != 3:
        raise ValueError(f"faces_f3 must be Tensor [F,3], got shape={getattr(faces_f3, 'shape', None)}")
    if not isinstance(sample_fidxs_ng, torch.Tensor) or sample_fidxs_ng.ndim != 1:
        raise ValueError(f"sample_fidxs_ng must be Tensor [Ng], got shape={getattr(sample_fidxs_ng, 'shape', None)}")
    if not isinstance(sample_bary_ng3, torch.Tensor) or sample_bary_ng3.ndim != 2 or sample_bary_ng3.shape[-1] != 3:
        raise ValueError(f"sample_bary_ng3 must be Tensor [Ng,3], got shape={getattr(sample_bary_ng3, 'shape', None)}")

    Nv = int(field_vc.shape[0])
    if faces_f3.max().item() >= Nv or faces_f3.min().item() < 0:
        raise ValueError("faces_f3 contains vertex indices out of range for field_vc.")

    # retrieve_verts_barycentric supports generic vertex channels (Nv,C) via einsum.
    out = retrieve_verts_barycentric(
        field_vc,
        faces_f3.to(device=field_vc.device, dtype=torch.long),
        sample_fidxs_ng.to(device=field_vc.device, dtype=torch.long),
        sample_bary_ng3.to(device=field_vc.device, dtype=field_vc.dtype),
    )
    if out.ndim != 2 or out.shape[0] != sample_fidxs_ng.shape[0] or out.shape[1] != field_vc.shape[1]:
        raise RuntimeError(f"interp output shape mismatch: got {tuple(out.shape)}, expected ({int(sample_fidxs_ng.shape[0])},{int(field_vc.shape[1])})")
    return out

