from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from model import libcore


_FRAME_DIR_RE = re.compile(r"^(?P<prefix>.+)_(?P<frame>\d+)$")


@dataclass(frozen=True)
class GeoFrameSpec:
    frame_id_int: int
    frame_id_str: str
    mesh_path: str


def _pad_frame_id(frame_id_int: int, *, digits: int) -> str:
    d = int(digits)
    return f"{int(frame_id_int):0{d}d}" if d > 0 else str(int(frame_id_int))


def discover_talkbody4d_geo_frames(
    *,
    mesh_subject_root: str,
    subject_prefix: str,
    mesh_relpath: str = "mesh/processed/nerf_simp_cleaned.obj",
    k_frames: int = 10,
    exclude_frame_id_int: Optional[int] = None,
    frame_id_list: Optional[Iterable[int | str]] = None,
    frame_format_digits: int = 4,
) -> List[GeoFrameSpec]:
    """Discover K geometry-supervised frames from TalkBody4D-style per-frame directories.

    Expected folder pattern under mesh_subject_root:
      {subject_prefix}_{frame:0Nd}/...

    Example:
      mesh_subject_root=/.../talkbody4d_avatarrex_2k
      subject_prefix=XG_01
      -> /.../talkbody4d_avatarrex_2k/XG_01_000034/mesh/processed/nerf_simp_cleaned.obj
    """
    root = Path(str(mesh_subject_root))
    if not root.exists():
        raise FileNotFoundError(f"pilot_geo_ft.mesh_subject_root does not exist: {root}")

    prefix = str(subject_prefix)
    rel = Path(str(mesh_relpath))

    chosen: List[int] = []
    if frame_id_list is not None:
        for x in list(frame_id_list):
            if x is None:
                continue
            try:
                if isinstance(x, str) and x.isdigit():
                    chosen.append(int(x))
                else:
                    chosen.append(int(x))
            except Exception:
                continue
    else:
        # Glob all dirs like XG_01_000034
        cands: List[int] = []
        for p in sorted(root.glob(f"{prefix}_*")):
            if not p.is_dir():
                continue
            m = _FRAME_DIR_RE.match(p.name)
            if not m:
                continue
            if m.group("prefix") != prefix:
                continue
            try:
                fid = int(m.group("frame"))
            except Exception:
                continue
            if exclude_frame_id_int is not None and int(fid) == int(exclude_frame_id_int):
                continue
            cands.append(fid)
        cands = sorted(set(cands))
        chosen = cands[: max(0, int(k_frames))]

    out: List[GeoFrameSpec] = []
    seen = set()
    for fid in chosen:
        if exclude_frame_id_int is not None and int(fid) == int(exclude_frame_id_int):
            continue
        if int(fid) in seen:
            continue
        seen.add(int(fid))
        frame_str = _pad_frame_id(int(fid), digits=int(frame_format_digits))
        mesh_dir = root / f"{prefix}_{frame_str}"
        mesh_path = mesh_dir / rel
        if not mesh_path.exists():
            raise FileNotFoundError(f"GT mesh not found for frame {frame_str}: {mesh_path}")
        out.append(
            GeoFrameSpec(
                frame_id_int=int(fid),
                frame_id_str=frame_str,
                mesh_path=str(mesh_path),
            )
        )
        if len(out) >= int(k_frames):
            break
    if not out:
        raise RuntimeError(
            "No geometry frames discovered. "
            f"mesh_subject_root={root} subject_prefix={prefix} exclude={exclude_frame_id_int} k_frames={k_frames}"
        )
    return out


class GeoMeshGtCache:
    """Cache for small-K posed GT meshes."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def get_mesh_vf_cpu(self, mesh_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (V,F) on CPU as float32/long tensors."""
        key = str(mesh_path)
        hit = self._cache.get(key, None)
        if hit is not None:
            return hit
        m = libcore.MeshCpu(key)
        V = torch.tensor(m.V, dtype=torch.float32, device="cpu")
        F = torch.tensor(m.F, dtype=torch.long, device="cpu")
        self._cache[key] = (V, F)
        return V, F

    def get_mesh_vf(self, mesh_path: str, *, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        Vc, Fc = self.get_mesh_vf_cpu(mesh_path)
        return Vc.to(device=device, dtype=dtype), Fc.to(device=device)


def build_pair_to_dataset_index(frameset) -> Dict[Tuple[str, str], int]:
    """Build (frame_id_str, cam_id_str) -> dataset index lookup for an AvatarRexDataset frameset."""
    out: Dict[Tuple[str, str], int] = {}
    for i, (fid, cid) in enumerate(getattr(frameset, "data_samples", [])):
        out[(str(fid), str(cid))] = int(i)
    return out

