from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
import trimesh
from loguru import logger

from src.utils.color.sh import RGB2SH
from src.scripts.swapping.data.loaders import load_smplx_vert_segmentation
from src.scripts.swapping.geometry.transformations import transform_gaussians


def _infer_dataset_type_from_any_path(p: str) -> str:
    s = str(p).lower()
    if "talkbody4d" in s:
        return "talkbody4d"
    if "mvhumannet" in s:
        return "mvhumannet"
    if "actorshq" in s:
        return "actorshq"
    if "thuman2" in s:
        return "thuman2"
    return "avatarrex"


def init_dummy_gray_hands_gaussians_from_smplx_mesh(
    *,
    args: Any,
    device: str,
    smplx_vert_seg_path: str,
    smplx_vert_seg: Optional[dict],
    A_smpl_params_path: str,
    user_A_id: str,
    model_B_id: str,
    A_prop_names_np: Tuple[str, ...] | list[str],
    B_transl: torch.Tensor,
    B_scale: torch.Tensor,
    B_global_orient_matrix: torch.Tensor,
) -> Tuple[dict, dict, int, Optional[dict]]:

    if smplx_vert_seg is None:
        logger.info("swap_hands_source=smplx_mesh: loading SMPLX vertex segmentation JSON.")
        smplx_vert_seg = load_smplx_vert_segmentation(smplx_vert_seg_path)

    dataset_root = Path(str(A_smpl_params_path)).parent.parent
    dataset_prefix = _infer_dataset_type_from_any_path(str(dataset_root))
    smpl_reposed_dir = (
        dataset_root
        / "gs_on_mesh_repose"
        / f"{dataset_prefix}_{user_A_id}_to_{model_B_id}"
        / "anim_0000_to_targets_meshes"
    )
    smpl_cands = [
        smpl_reposed_dir / "smpl_reposed_frame_0000.obj",
        smpl_reposed_dir / "smpl_reposed_frame_0000_cleaned.obj",
    ]
    smpl_reposed_mesh_path = next((p for p in smpl_cands if p.exists()), None)
    if smpl_reposed_mesh_path is None:
        raise FileNotFoundError(f"SMPL reposed mesh not found (tried: {smpl_cands})")

    smpl_reposed_mesh = trimesh.load(str(smpl_reposed_mesh_path), process=False, force="mesh")
    if not isinstance(smpl_reposed_mesh, trimesh.Trimesh):
        raise ValueError(f"Expected trimesh.Trimesh from {smpl_reposed_mesh_path}, got {type(smpl_reposed_mesh)}")
    V = np.asarray(smpl_reposed_mesh.vertices, dtype=np.float32)
    F = np.asarray(smpl_reposed_mesh.faces, dtype=np.int64)

    def _ids_for_keys(keys: list[str]) -> np.ndarray:
        out = []
        for k in keys:
            ids = smplx_vert_seg.get(k, [])
            if isinstance(ids, list):
                out.extend([int(x) for x in ids if isinstance(x, (int, float))])
        if not out:
            return np.zeros((0,), dtype=np.int64)
        arr = np.unique(np.asarray(out, dtype=np.int64))
        arr = arr[(arr >= 0) & (arr < V.shape[0])]
        return arr

    left_vids = _ids_for_keys(["leftHand", "leftHandIndex1"])
    right_vids = _ids_for_keys(["rightHand", "rightHandIndex1"])
    if left_vids.size == 0 or right_vids.size == 0:


        if smpl_reposed_mesh_path.name.endswith("_cleaned.obj"):
            alt = smpl_reposed_dir / "smpl_reposed_frame_0000.obj"
            if alt.exists():
                logger.warning(
                    "swap_hands_source=smplx_mesh: segmentation indices out of range for cleaned mesh; "
                    f"reloading full-topology mesh: {alt}"
                )
                smpl_reposed_mesh = trimesh.load(str(alt), process=False, force="mesh")
                V = np.asarray(smpl_reposed_mesh.vertices, dtype=np.float32)
                F = np.asarray(smpl_reposed_mesh.faces, dtype=np.int64)
                left_vids = _ids_for_keys(["leftHand", "leftHandIndex1"])
                right_vids = _ids_for_keys(["rightHand", "rightHandIndex1"])

        if left_vids.size == 0 or right_vids.size == 0:
            raise ValueError(
                "SMPLX segmentation missing left/right hand vertex ids. "
                "Expected keys: leftHand/rightHand (+Index1)."
            )

    keep_v_left = np.zeros((V.shape[0],), dtype=bool)
    keep_v_right = np.zeros((V.shape[0],), dtype=bool)
    keep_v_left[left_vids] = True
    keep_v_right[right_vids] = True
    fmask_left = np.all(keep_v_left[F], axis=1)
    fmask_right = np.all(keep_v_right[F], axis=1)
    fmask_union = fmask_left | fmask_right
    if not np.any(fmask_union):
        raise ValueError("No faces selected for SMPLX hands (union mask empty).")
    side_L = fmask_left[fmask_union]
    side_R = fmask_right[fmask_union]
    if not np.any(side_L) or not np.any(side_R):
        raise ValueError("SMPLX hands face selection missing one side (left or right).")


    face_mask_union = np.asarray(fmask_union).astype(bool)
    Fk = F[face_mask_union]
    used = np.unique(Fk.reshape(-1))
    used.sort()
    vmap = -np.ones((V.shape[0],), dtype=np.int64)
    vmap[used] = np.arange(used.shape[0], dtype=np.int64)
    V_hB = V[used]
    F_hB = vmap[Fk].astype(np.int64)
    if V_hB.shape[0] < 3 or F_hB.shape[0] < 1:
        raise ValueError("Degenerate SMPLX hands submesh after compaction.")

    faces_left_idx = np.where(np.asarray(side_L).astype(bool))[0].astype(np.int64)
    faces_right_idx = np.where(np.asarray(side_R).astype(bool))[0].astype(np.int64)
    mL = trimesh.Trimesh(vertices=V_hB, faces=F_hB[faces_left_idx], process=False)
    mR = trimesh.Trimesh(vertices=V_hB, faces=F_hB[faces_right_idx], process=False)

    n_per_hand = int(max(0, int(getattr(args, "smplx_hands_gaussians_per_hand", 2000))))
    if n_per_hand <= 0:
        raise ValueError("smplx_hands_gaussians_per_hand must be > 0")

    ptsL, fidL = trimesh.sample.sample_surface(mL, n_per_hand)
    ptsR, fidR = trimesh.sample.sample_surface(mR, n_per_hand)
    ptsL = np.asarray(ptsL, dtype=np.float32)
    ptsR = np.asarray(ptsR, dtype=np.float32)
    fidL = np.asarray(fidL, dtype=np.int64).reshape(-1)
    fidR = np.asarray(fidR, dtype=np.int64).reshape(-1)

    fidL_full = faces_left_idx[np.clip(fidL, 0, faces_left_idx.shape[0] - 1)]
    fidR_full = faces_right_idx[np.clip(fidR, 0, faces_right_idx.shape[0] - 1)]
    pts = np.concatenate([ptsL, ptsR], axis=0)
    fidx_full = np.concatenate([fidL_full, fidR_full], axis=0).astype(np.int64)


    tri = V_hB[F_hB[fidx_full]]
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    v0p = pts - v0
    d00 = np.sum(v0v1 * v0v1, axis=1)
    d01 = np.sum(v0v1 * v0v2, axis=1)
    d11 = np.sum(v0v2 * v0v2, axis=1)
    d20 = np.sum(v0p * v0v1, axis=1)
    d21 = np.sum(v0p * v0v2, axis=1)
    denom = (d00 * d11 - d01 * d01)
    denom = np.where(np.abs(denom) < 1e-20, 1.0, denom)
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary = np.stack([u, v, w], axis=1).astype(np.float32)
    bary = np.clip(bary, -1e-3, 1.0 + 1e-3).astype(np.float32)

    gray = float(getattr(args, "smplx_hands_gray", 0.6))
    gray = float(np.clip(gray, 0.0, 1.0))
    sh = RGB2SH(torch.tensor([[gray, gray, gray]], dtype=torch.float32, device=device)).view(-1)
    if int(sh.numel()) < 3:
        raise ValueError(f"RGB2SH returned unexpected shape: {tuple(sh.shape)}")
    log_s = float(np.log(max(float(getattr(args, "smplx_hands_scale_m", 0.004)), 1e-8)))

    hands_worldB: dict = {}
    Np = int(pts.shape[0])
    for name in A_prop_names_np:
        if name == "x":
            hands_worldB[name] = torch.from_numpy(pts[:, 0]).to(device=device, dtype=torch.float32)
        elif name == "y":
            hands_worldB[name] = torch.from_numpy(pts[:, 1]).to(device=device, dtype=torch.float32)
        elif name == "z":
            hands_worldB[name] = torch.from_numpy(pts[:, 2]).to(device=device, dtype=torch.float32)
        elif name == "f_dc_0":
            hands_worldB[name] = torch.full((Np,), float(sh[0].item()), device=device, dtype=torch.float32)
        elif name == "f_dc_1":
            hands_worldB[name] = torch.full((Np,), float(sh[1].item()), device=device, dtype=torch.float32)
        elif name == "f_dc_2":
            hands_worldB[name] = torch.full((Np,), float(sh[2].item()), device=device, dtype=torch.float32)
        elif name == "opacity":
            hands_worldB[name] = torch.full((Np,), 1.0, device=device, dtype=torch.float32)
        elif name in ("scale_0", "scale_1", "scale_2"):
            hands_worldB[name] = torch.full((Np,), float(log_s), device=device, dtype=torch.float32)
        elif name == "rot_0":
            hands_worldB[name] = torch.full((Np,), 1.0, device=device, dtype=torch.float32)
        elif name in ("rot_1", "rot_2", "rot_3"):
            hands_worldB[name] = torch.zeros((Np,), device=device, dtype=torch.float32)
        else:
            hands_worldB[name] = torch.zeros((Np,), device=device, dtype=torch.float32)

    A_hands_final_canon = transform_gaussians(
        hands_worldB,
        tuple(A_prop_names_np),
        B_transl,
        B_scale,
        B_global_orient_matrix,
        to_canonical=True,
        device=device,
        pelvis_joint_j0=None,
    )
    num_A_hands_gaussians = int(A_hands_final_canon.get("x", torch.empty(0)).shape[0])

    smplx_hands_proxy = {
        "smpl_reposed_mesh_path": str(smpl_reposed_mesh_path),
        "V_hands_world_B": V_hB,
        "F_hands": F_hB,
        "used_vids_full": used,
        "sample_fidxs": fidx_full,
        "sample_bary": bary,
        "sample_points_world_B": pts,
    }

    logger.info(
        f"swap_hands_source=smplx_mesh: initialized {num_A_hands_gaussians} dummy hands gaussians "
        f"from {smpl_reposed_mesh_path.name} (in canonical space)."
    )
    return smplx_vert_seg, A_hands_final_canon, num_A_hands_gaussians, smplx_hands_proxy
