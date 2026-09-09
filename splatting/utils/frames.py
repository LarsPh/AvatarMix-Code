import torch
import torch.nn.functional as F


def _safe_normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))


def build_ntb_frame_from_tri_and_smoothN(
    v1: torch.Tensor,
    v2: torch.Tensor,
    v3: torch.Tensor,
    N: torch.Tensor,
    eps: float = 1e-8,
):
    """Build a per-point orthonormal frame (N,T,B) aligned to smooth normal N.

    Inputs are batched per Gaussian:
    - v1,v2,v3: (G,3) posed triangle vertices for the parent face
    - N: (G,3) smooth normal at the barycentric point (already world-space)

    Returns:
    - Nn, T, B: each (G,3), unit vectors.
    """
    Nn = _safe_normalize(N, eps=eps)

    e1 = v2 - v1
    e2 = v3 - v1
    e1n = _safe_normalize(e1, eps=eps)
    e2n = _safe_normalize(e2, eps=eps)

    # Prefer the edge less parallel to N
    dot1 = (e1n * Nn).sum(dim=-1).abs()
    dot2 = (e2n * Nn).sum(dim=-1).abs()
    use_e1 = dot1 < dot2
    e = torch.where(use_e1[:, None], e1, e2)

    # Gram-Schmidt: remove normal component
    T_raw = e - (e * Nn).sum(dim=-1, keepdim=True) * Nn
    T_norm = T_raw.norm(dim=-1, keepdim=True)

    # Degenerate fallback: pick a global axis least aligned with N
    bad = (T_norm.squeeze(-1) < eps)
    if bad.any():
        absN = Nn.abs()
        # choose axis with smallest abs component => least aligned
        idx = absN.argmin(dim=-1)  # (G,)
        axis = torch.zeros_like(Nn)
        axis.scatter_(1, idx[:, None], 1.0)
        e_fallback = axis - (axis * Nn).sum(dim=-1, keepdim=True) * Nn
        T_raw = torch.where(bad[:, None], e_fallback, T_raw)

    T = _safe_normalize(T_raw, eps=eps)
    B = torch.cross(Nn, T, dim=-1)
    B = _safe_normalize(B, eps=eps)

    # Re-orthogonalize T for numerical stability (optional but cheap)
    T = torch.cross(B, Nn, dim=-1)
    T = _safe_normalize(T, eps=eps)

    return Nn, T, B


def apply_ntb_delta(N: torch.Tensor, T: torch.Tensor, B: torch.Tensor, delta_u_local_ntb: torch.Tensor) -> torch.Tensor:
    """Convert δu_local (in NTB coords) to world delta.

    Inputs:
    - N,T,B: (G,3) frame vectors
    - delta_u_local_ntb: (G,3) where components correspond to (N,T,B) axes
    Returns:
    - delta_world: (G,3)
    """
    du0 = delta_u_local_ntb[:, 0:1]
    du1 = delta_u_local_ntb[:, 1:2]
    du2 = delta_u_local_ntb[:, 2:3]
    return N * du0 + T * du1 + B * du2

