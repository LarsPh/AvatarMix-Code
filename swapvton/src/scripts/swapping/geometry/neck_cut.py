from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


SMPLX_J_SPINE2 = 6
SMPLX_J_NECK = 12
SMPLX_J_L_COLLAR = 13
SMPLX_J_R_COLLAR = 14
SMPLX_J_HEAD = 15


@dataclass(frozen=True)
class NeckCutParams:


    radius_ratio: float = 0.35
    axial_ratio: float = 1.0
    eps_ratio: float = 0.02
    plane_lift_ratio: float = 0.20


def _as_numpy_xyz(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a.astype(np.float32)


def compute_neck_axis_and_masks(
    scan_vertices: np.ndarray,
    joints_xyz,
    params: Optional[NeckCutParams] = None,
) -> Dict[str, np.ndarray]:

    if params is None:
        params = NeckCutParams()

    v = _as_numpy_xyz(scan_vertices)
    j = _as_numpy_xyz(joints_xyz)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"scan_vertices must be (N,3), got {v.shape}")
    if j.ndim != 2 or j.shape[1] != 3:
        raise ValueError(f"joints_xyz must be (J,3), got {j.shape}")

    neck = j[SMPLX_J_NECK]
    head = j[SMPLX_J_HEAD]
    spine2 = j[SMPLX_J_SPINE2]
    lcol = j[SMPLX_J_L_COLLAR]
    rcol = j[SMPLX_J_R_COLLAR]

    axis = head - neck
    axis_len = float(np.linalg.norm(axis))
    if not np.isfinite(axis_len) or axis_len < 1e-6:

        axis_u = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        axis_len = 1.0
    else:
        axis_u = axis / axis_len

    collar_width = float(np.linalg.norm(lcol - rcol))
    if not np.isfinite(collar_width) or collar_width < 1e-6:
        collar_width = axis_len

    radius = float(params.radius_ratio * collar_width)
    eps = float(params.eps_ratio * collar_width)
    axial_max = float(params.axial_ratio * axis_len)


    plane_point = neck + float(params.plane_lift_ratio) * (head - neck)
    rel = v - plane_point[None, :]
    t = rel @ axis_u
    above_plane = t >= -eps
    within_axial = (t >= -eps) & (t <= axial_max + eps)


    closest = neck[None, :] + (rel @ axis_u)[:, None] * axis_u[None, :]
    radial = np.linalg.norm(v - closest, axis=1)
    within_radius = radial <= radius

    neck_tube = above_plane & within_axial & within_radius

    return {
        "above_neck_plane_mask": above_plane.astype(bool),
        "neck_tube_mask": neck_tube.astype(bool),
        "debug": {
            "neck": neck,
            "head": head,
            "spine2": spine2,
            "plane_point": plane_point.astype(np.float32),
            "plane_lift_ratio": float(params.plane_lift_ratio),
            "axis_unit": axis_u.astype(np.float32),
            "axis_len": axis_len,
            "collar_width": collar_width,
            "radius": radius,
            "eps": eps,
            "axial_max": axial_max,
        },
    }
