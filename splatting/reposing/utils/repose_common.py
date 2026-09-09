from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


AVATARREX_SMPLX_CONFIG = {
    "use_pca": False,
    "num_pca_comps": 45,
    "flat_hand_mean": True,
}

THUMAN2_SMPLX_CONFIG = {
    "use_pca": True,
    "num_pca_comps": 12,
    "flat_hand_mean": False,
}

ACTORSHQ_SMPLX_CONFIG = {
    "use_pca": True,
    "num_pca_comps": 6,
    "flat_hand_mean": True,
}


def _load_obj_vertices(path: Path) -> np.ndarray:
    import trimesh

    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)} for {path}")
    return np.asarray(mesh.vertices, dtype=np.float32)


def _load_obj_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import trimesh

    mesh = trimesh.load_mesh(str(path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected Trimesh, got {type(mesh)} for {path}")
    return np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces, dtype=np.int64)


def _color_from_id(idx: int) -> tuple[int, int, int]:

    import colorsys

    if idx < 0:
        return (140, 140, 140)
    h = (idx * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def _vertex_colors_from_tri_ids(tri_ids: np.ndarray) -> np.ndarray:
    tri_ids = np.asarray(tri_ids, dtype=np.int64)
    rgb = np.zeros((tri_ids.shape[0], 3), dtype=np.uint8)
    for i, tid in enumerate(tri_ids.tolist()):
        rgb[i] = np.array(_color_from_id(int(tid)), dtype=np.uint8)
    return rgb


def _vertex_colors_from_face_ids(*, n_vertices: int, faces: np.ndarray, face_ids: np.ndarray) -> np.ndarray:

    faces = np.asarray(faces, dtype=np.int64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    accum = np.zeros((int(n_vertices), 3), dtype=np.float64)
    counts = np.zeros((int(n_vertices),), dtype=np.float64)
    for f_idx in range(faces.shape[0]):
        c = np.array(_color_from_id(int(face_ids[f_idx])), dtype=np.float64)
        for v in faces[f_idx]:
            accum[int(v)] += c
            counts[int(v)] += 1.0
    counts = np.maximum(counts, 1.0)[:, None]
    rgb = (accum / counts).clip(0, 255).astype(np.uint8)
    return rgb


def _read_gender_txt(subject_dir: Path) -> Optional[str]:

    try:
        gpath = Path(subject_dir) / "gender.txt"
        if not gpath.exists():
            return None
        txt = gpath.read_text().strip().lower()
        if txt in {"m", "male"}:
            return "male"
        if txt in {"f", "female"}:
            return "female"
        if txt in {"n", "neutral"}:
            return "neutral"
    except Exception:
        return None
    return None


def _resolve_gender_for_dir(subject_dir: Path, requested: str, *, force_neutral: bool) -> str:

    if force_neutral:
        return "neutral"
    req = str(requested).strip().lower()


    sd_str = str(subject_dir).lower()
    if "actorshq" in sd_str or req in {"auto", ""}:
        from_file = _read_gender_txt(Path(subject_dir))
        if from_file is not None:
            return from_file

    if req in {"male", "female", "neutral"}:
        return req
    return "neutral"
