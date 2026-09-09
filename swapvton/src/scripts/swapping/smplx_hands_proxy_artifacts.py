from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import trimesh
from loguru import logger

from src.scripts.swapping.data.constants import DETAILED_SURFACE_LABELS
from src.scripts.swapping.geometry.transformations import _transform_gaussian_positions


def emit_smplx_mesh_hands_proxy_artifacts(
    *,
    output_dir: str,
    device: str,
    smplx_hands_proxy: Optional[dict],
    A_smpl_params_path: str,
    A_transl: torch.Tensor,
    A_scale: torch.Tensor,
    A_global_orient_matrix: torch.Tensor,
    A_pelvis_joint_j0: Optional[torch.Tensor],
    B_transl: torch.Tensor,
    B_scale: torch.Tensor,
    B_global_orient_matrix: torch.Tensor,
) -> None:

    if smplx_hands_proxy is None:
        logger.warning("swap_hands_source=smplx_mesh: missing smplx_hands_proxy cache; skipping proxy artifacts.")
        return

    V_hB = np.asarray(smplx_hands_proxy["V_hands_world_B"], dtype=np.float32)
    F_h = np.asarray(smplx_hands_proxy["F_hands"], dtype=np.int64)
    used_vids = np.asarray(smplx_hands_proxy["used_vids_full"], dtype=np.int64).reshape(-1)
    sample_fidxs = np.asarray(smplx_hands_proxy["sample_fidxs"], dtype=np.int64).reshape(-1)
    sample_bary = np.asarray(smplx_hands_proxy["sample_bary"], dtype=np.float32).reshape(-1, 3)


    hands_B_obj = os.path.join(output_dir, "hands_world_B_body_donor.obj")
    trimesh.Trimesh(vertices=V_hB, faces=F_h, process=False).export(hands_B_obj)


    Vb_t = torch.from_numpy(V_hB).to(device=device, dtype=torch.float32)
    Vc_t = _transform_gaussian_positions(
        Vb_t, B_transl, B_scale, B_global_orient_matrix, to_canonical=True, pelvis_joint_j0=None
    )
    Va_t = _transform_gaussian_positions(
        Vc_t, A_transl, A_scale, A_global_orient_matrix, to_canonical=False, pelvis_joint_j0=A_pelvis_joint_j0
    )
    V_hA = Va_t.detach().cpu().numpy().astype(np.float32)
    hands_A_obj = os.path.join(output_dir, "hands_world_A.obj")
    trimesh.Trimesh(vertices=V_hA, faces=F_h, process=False).export(hands_A_obj)


    Ng = int(sample_fidxs.shape[0])
    embed = {
        "cano_mesh": "hands_world_A.obj",
        "sample_fidxs": sample_fidxs.astype(int).tolist(),
        "sample_bary": sample_bary.astype(np.float32).tolist(),
        "_xyz": np.zeros((Ng, 3), dtype=np.float32).tolist(),
        "_rotation": np.tile(np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)[None, :], (Ng, 1)).tolist(),
        "part": "hands",
        "source": "smplx_mesh_dummy_gray",
    }
    with open(os.path.join(output_dir, "embedding_hands.json"), "w") as f:
        json.dump(embed, f, indent=2)


    try:
        hands_label_idx = int(DETAILED_SURFACE_LABELS.index("hands"))
    except Exception:
        hands_label_idx = 7
    label_obj = {"scan_labels": np.full((V_hA.shape[0],), hands_label_idx, dtype=np.int64)}
    with open(os.path.join(output_dir, "label_A_hands.pkl"), "wb") as f:
        pickle.dump(label_obj, f)


    w_cands = [
        Path(str(A_smpl_params_path)).parent / "mesh" / "processed" / "smpl_lbs_weights.npy",
        Path(str(A_smpl_params_path)).parent / "mesh" / "processed" / "smplx_lbs_weights.npy",
    ]
    w_path = next((p for p in w_cands if p.exists()), None)
    if w_path is None:
        logger.warning(f"swap_hands_source=smplx_mesh: missing SMPL LBS weights (tried: {w_cands}); skipping hands weights.")
    else:
        W_full = np.asarray(np.load(str(w_path)), dtype=np.float32)
        max_used = int(used_vids.max()) if used_vids.size > 0 else 0
        if W_full.ndim != 2 or W_full.shape[0] <= max_used:
            logger.warning(
                f"swap_hands_source=smplx_mesh: SMPL LBS weights shape mismatch: {getattr(W_full, 'shape', None)} "
                f"vs max used_vid {max_used}; skipping."
            )
        else:
            W_h = W_full[used_vids]
            np.save(os.path.join(output_dir, "smoothed_inpainted_weights_A_hands.npy"), W_h.astype(np.float32))

    logger.info("swap_hands_source=smplx_mesh: wrote hands proxy mesh/embedding/labels/LBS artifacts.")
