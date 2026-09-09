from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


def save_ply_mesh(
    path: str | Path,
    verts: torch.Tensor,
    faces: torch.Tensor,
    *,
    vert_colors: Optional[torch.Tensor] = None,
    vert_quality: Optional[torch.Tensor] = None,
) -> None:
    """Save a triangle mesh to an ASCII PLY file.

    - verts: (V,3) float
    - faces: (F,3) int
    - vert_colors: optional (V,3) float in [0,1] or uint8 in [0,255]
    - vert_quality: optional (V,) float scalar stored as property 'quality'
    """
    path = Path(path)
    v = verts.detach().cpu().numpy()
    f = faces.detach().cpu().numpy()

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"verts must be (V,3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {f.shape}")

    v = v.astype(np.float32, copy=False)
    f = f.astype(np.int64, copy=False)

    use_c = vert_colors is not None
    c_u8 = None
    if use_c:
        c = vert_colors.detach().cpu().numpy()
        if c.shape != (v.shape[0], 3):
            raise ValueError(f"vert_colors must be (V,3), got {c.shape} vs V={v.shape[0]}")
        if c.dtype == np.uint8:
            c_u8 = c
        else:
            # assume float
            c_u8 = np.clip(c, 0.0, 1.0)
            c_u8 = (c_u8 * 255.0 + 0.5).astype(np.uint8)

    use_q = vert_quality is not None
    q_f = None
    if use_q:
        q = vert_quality.detach().cpu().numpy()
        if q.shape not in [(v.shape[0],), (v.shape[0], 1)]:
            raise ValueError(f"vert_quality must be (V,) or (V,1), got {q.shape} vs V={v.shape[0]}")
        q = q.reshape(v.shape[0]).astype(np.float32, copy=False)
        q_f = q

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fp:
        fp.write("ply\n")
        fp.write("format ascii 1.0\n")
        fp.write(f"element vertex {v.shape[0]}\n")
        fp.write("property float x\n")
        fp.write("property float y\n")
        fp.write("property float z\n")
        if use_q:
            fp.write("property float quality\n")
        if use_c:
            fp.write("property uchar red\n")
            fp.write("property uchar green\n")
            fp.write("property uchar blue\n")
        fp.write(f"element face {f.shape[0]}\n")
        fp.write("property list uchar int vertex_indices\n")
        fp.write("end_header\n")

        if use_q and use_c:
            for (x, y, z), qq, (r, g, b) in zip(v, q_f, c_u8):
                fp.write(f"{x:.6f} {y:.6f} {z:.6f} {float(qq):.6f} {int(r)} {int(g)} {int(b)}\n")
        elif use_q and (not use_c):
            for (x, y, z), qq in zip(v, q_f):
                fp.write(f"{x:.6f} {y:.6f} {z:.6f} {float(qq):.6f}\n")
        elif (not use_q) and use_c:
            for (x, y, z), (r, g, b) in zip(v, c_u8):
                fp.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
        else:
            for x, y, z in v:
                fp.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

        for i0, i1, i2 in f:
            fp.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")

