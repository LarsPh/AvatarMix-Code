from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from src.utils.color.sh import SH2RGB

from src.scripts.swapping.geometry.analysis import get_labeled_part_verts_and_faces
from src.scripts.swapping.color.processing import robust_lab_stats_from_rgb


def _faces_from_vertex_mask(mesh_faces: np.ndarray, vert_mask: np.ndarray) -> np.ndarray:

    if vert_mask.dtype != bool:
        vert_mask = vert_mask.astype(bool)
    face_mask = np.all(vert_mask[mesh_faces], axis=1)
    return np.where(face_mask)[0]


def _union_face_indices(*arrays: np.ndarray) -> np.ndarray:
    a = [x for x in arrays if isinstance(x, np.ndarray) and x.size > 0]
    if not a:
        return np.array([], dtype=int)
    return np.unique(np.concatenate(a)).astype(int)


def gaussian_mask_from_face_indices(sample_fidxs: np.ndarray, face_indices: np.ndarray) -> np.ndarray:

    if face_indices is None or face_indices.size == 0:
        return np.zeros(sample_fidxs.shape[0], dtype=bool)
    return np.isin(sample_fidxs, face_indices)


def build_pooled_face_index_maps(
    segmentation_labels: np.ndarray,
    scan_mesh_faces: np.ndarray,
    current_surface_labels: list[str],
    neck_tube_mask: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:

    out: Dict[str, np.ndarray] = {}


    for label in ("left_arm", "right_arm", "left_leg", "right_leg", "torso_skin", "hands"):
        _, fidx = get_labeled_part_verts_and_faces(
            segmentation_labels, scan_mesh_faces, current_surface_labels, label
        )
        out[label] = fidx.astype(int) if isinstance(fidx, np.ndarray) else np.array([], dtype=int)

    out["arms_both"] = _union_face_indices(out.get("left_arm"), out.get("right_arm"))
    out["legs_both"] = _union_face_indices(out.get("left_leg"), out.get("right_leg"))

    out["hands_both"] = out.get("hands", np.array([], dtype=int))


    try:
        torso_label_idx = current_surface_labels.index("torso_skin")
        torso_vert_indices = np.where(segmentation_labels == torso_label_idx)[0]
    except ValueError:
        torso_vert_indices = np.array([], dtype=int)

    if torso_vert_indices.size > 0 and neck_tube_mask is not None:
        n_verts = segmentation_labels.shape[0]
        neck_tube_mask = np.asarray(neck_tube_mask).astype(bool)
        if neck_tube_mask.shape[0] == n_verts:
            torso_mask = np.zeros(n_verts, dtype=bool)
            torso_mask[torso_vert_indices] = True

            neck_mask = torso_mask & neck_tube_mask
            torso_non_neck_mask = torso_mask & (~neck_tube_mask)

            out["neck"] = _faces_from_vertex_mask(scan_mesh_faces, neck_mask).astype(int)
            out["torso_non_neck"] = _faces_from_vertex_mask(scan_mesh_faces, torso_non_neck_mask).astype(int)
        else:
            out["neck"] = np.array([], dtype=int)
            out["torso_non_neck"] = out.get("torso_skin", np.array([], dtype=int))
    else:
        out["neck"] = np.array([], dtype=int)
        out["torso_non_neck"] = out.get("torso_skin", np.array([], dtype=int))

    return out


def robust_source_lab_stats_from_gaussians_canon(
    part_gaussians_canon: dict,
    *,
    opacity_threshold: float,
    use_all_gaussians_for_stats: bool,
    mad_k: float = 4.0,
    min_neff: float = 100.0,
    min_ratio: float = 0.01,
) -> dict:

    if not part_gaussians_canon:
        return {"status": "missing", "lab_stats": None, "mean_rgb": None, "neff": 0.0, "n_selected": 0, "n_inlier": 0}
    if not all(k in part_gaussians_canon for k in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity")):
        return {"status": "missing", "lab_stats": None, "mean_rgb": None, "neff": 0.0, "n_selected": 0, "n_inlier": 0}

    op = part_gaussians_canon["opacity"]
    if not isinstance(op, torch.Tensor):
        op = torch.from_numpy(np.array(op))
    op_act = torch.sigmoid(op).float()

    if use_all_gaussians_for_stats:
        sel = torch.ones_like(op_act, dtype=torch.bool)
    else:
        sel = op_act >= float(opacity_threshold)

    if sel.numel() == 0 or not bool(torch.any(sel).item()):
        return {"status": "missing", "lab_stats": None, "mean_rgb": None, "neff": 0.0, "n_selected": 0, "n_inlier": 0}

    sh_dc = torch.stack(
        [
            part_gaussians_canon["f_dc_0"][sel],
            part_gaussians_canon["f_dc_1"][sel],
            part_gaussians_canon["f_dc_2"][sel],
        ],
        dim=-1,
    )
    rgb = torch.clamp(SH2RGB(sh_dc), 0.0, 1.0)
    w = op_act[sel]

    mean_lab, std_lab, inlier_mask, neff = robust_lab_stats_from_rgb(
        rgb, w, mad_k=mad_k, min_neff=min_neff, min_ratio=min_ratio
    )
    n_sel = int(rgb.shape[0])
    n_in = int(inlier_mask.sum().item()) if inlier_mask is not None else n_sel
    if mean_lab is None or std_lab is None:
        return {
            "status": "too_small_or_noisy",
            "lab_stats": None,
            "mean_rgb": rgb.mean(dim=0).detach().cpu().tolist() if n_sel > 0 else None,
            "neff": float(neff),
            "n_selected": n_sel,
            "n_inlier": n_in,
        }

    if inlier_mask is not None and n_in > 0:
        mean_rgb = rgb[inlier_mask].mean(dim=0).detach().cpu().tolist()
    else:
        mean_rgb = rgb.mean(dim=0).detach().cpu().tolist() if n_sel > 0 else None
    return {
        "status": "ok",
        "lab_stats": (mean_lab, std_lab),
        "mean_rgb": mean_rgb,
        "neff": float(neff),
        "n_selected": n_sel,
        "n_inlier": n_in,
    }
