import numpy as np
import torch


def bbox_size(vertices: np.ndarray) -> np.ndarray:

    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    return vmax - vmin


def compute_source_params_from_raw_skeleton(skeleton_raw: np.ndarray):

    if skeleton_raw.ndim != 2 or skeleton_raw.shape[1] != 3:
        raise ValueError("skeleton_raw must be (N, 3)")

    center_offset = skeleton_raw.mean(axis=0)
    skel_centered = skeleton_raw - center_offset
    bbox = bbox_size(skel_centered)
    max_extent = float(bbox.max())

    if max_extent <= 0.0:
        raise ValueError("Skeleton has zero bbox extent after centering; cannot compute scale.")

    source_scaling = 2.0 / max_extent

    return {
        "center_offset": center_offset.astype(np.float32),
        "source_scaling": float(source_scaling)
    }


def restore_source(vertices_normalized: np.ndarray, center_offset: np.ndarray, source_scaling: float):

    return (vertices_normalized / float(source_scaling)) + center_offset
