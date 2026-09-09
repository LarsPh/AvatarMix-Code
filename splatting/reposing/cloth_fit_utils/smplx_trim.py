from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import trimesh
from loguru import logger

from .reposed_smpl_cleanup import load_smplx_cloth_fit_removal_indices


@dataclass(frozen=True)
class SmplxTrimResult:
    vertices: np.ndarray
    faces: np.ndarray
    keep_vertex_mask: np.ndarray
    full_to_trim_vertex: np.ndarray
    full_to_trim_face: np.ndarray


def _ensure_trimesh(mesh: trimesh.Trimesh | trimesh.Scene, *, path: Path) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)} for {path}")
    return mesh


def load_smplx_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:

    path = Path(path)
    mesh = trimesh.load_mesh(str(path), process=False)
    mesh = _ensure_trimesh(mesh, path=path)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise ValueError(f"Invalid mesh shapes for {path}: V{V.shape} F{F.shape}")
    return V, F


def build_keep_vertex_mask(
    *,
    n_vertices: int,
    smplx_segmentation_json_path: Path,
    cloth_fit_remove_feet: bool,
    cloth_fit_keep_palm: bool = False,
) -> np.ndarray:

    removal = load_smplx_cloth_fit_removal_indices(
        segmentation_json_path=Path(smplx_segmentation_json_path),
        remove_feet=bool(cloth_fit_remove_feet),
        keep_palm=bool(cloth_fit_keep_palm),
    )
    keep = np.ones(int(n_vertices), dtype=bool)
    keep[removal] = False
    return keep


def build_full_to_trim_face_map(*, faces_full: np.ndarray, keep_vertex_mask: np.ndarray) -> np.ndarray:

    faces_full = np.asarray(faces_full, dtype=np.int64)
    keep_vertex_mask = np.asarray(keep_vertex_mask, dtype=bool)
    keep_face = keep_vertex_mask[faces_full].all(axis=1)
    full_to_trim_face = np.full((faces_full.shape[0],), -1, dtype=np.int64)
    full_to_trim_face[keep_face] = np.arange(int(keep_face.sum()), dtype=np.int64)
    return full_to_trim_face


def trim_smplx_mesh(
    *,
    vertices_full: np.ndarray,
    faces_full: np.ndarray,
    keep_vertex_mask: np.ndarray,
) -> SmplxTrimResult:

    V = np.asarray(vertices_full, dtype=np.float64)
    F = np.asarray(faces_full, dtype=np.int64)
    keep_v = np.asarray(keep_vertex_mask, dtype=bool)

    if V.shape[0] != keep_v.shape[0]:
        raise ValueError(f"keep_vertex_mask length {keep_v.shape[0]} != n_vertices {V.shape[0]}")


    keep_f = keep_v[F].all(axis=1)
    F_kept_full_vid = F[keep_f]


    full_to_trim_v = np.full((V.shape[0],), -1, dtype=np.int64)
    kept_vids = np.nonzero(keep_v)[0]
    full_to_trim_v[kept_vids] = np.arange(kept_vids.shape[0], dtype=np.int64)

    F_trim = full_to_trim_v[F_kept_full_vid]
    if (F_trim < 0).any():
        raise RuntimeError("Trimmed face contains removed vertex after filtering (bug).")

    V_trim = V[kept_vids]

    full_to_trim_f = build_full_to_trim_face_map(faces_full=F, keep_vertex_mask=keep_v)

    return SmplxTrimResult(
        vertices=V_trim,
        faces=F_trim,
        keep_vertex_mask=keep_v,
        full_to_trim_vertex=full_to_trim_v,
        full_to_trim_face=full_to_trim_f,
    )


def save_trimmed_mesh(path: Path, *, vertices: np.ndarray, faces: np.ndarray, overwrite: bool) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        logger.info(f"Trim mesh exists and overwrite disabled, skipping: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    m = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    m.export(str(path))
    logger.info(f"Saved trimmed SMPL-X mesh: {path} (V={len(vertices)} F={len(faces)})")


def assert_faces_identical(*, faces_a: np.ndarray, faces_b: np.ndarray, label_a: str, label_b: str) -> None:
    fa = np.asarray(faces_a, dtype=np.int64)
    fb = np.asarray(faces_b, dtype=np.int64)
    if fa.shape != fb.shape or not np.array_equal(fa, fb):
        raise ValueError(
            "SMPL-X face topology mismatch (required identical ordering for A2). "
            f"{label_a}.faces shape={fa.shape}, {label_b}.faces shape={fb.shape}."
        )
