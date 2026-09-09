import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import trimesh
from loguru import logger

from src.scripts.swapping.data.loaders import load_embedding_full, load_mesh
from src.scripts.swapping.data.savers import save_ply_gaussians
from src.scripts.swapping.geometry.transformations import _transform_gaussian_positions, transform_gaussians


def _to_np(gs_dict: dict) -> dict:
    return {k: (v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)) for k, v in gs_dict.items()}


def _translate_gaussians_xyz(gaussians: dict, offset_xyz: torch.Tensor, *, sign: float) -> dict:

    if gaussians is None:
        return gaussians
    if not all(k in gaussians for k in ("x", "y", "z")):
        return gaussians
    out = dict(gaussians)
    out["x"] = out["x"] + sign * offset_xyz[0]
    out["y"] = out["y"] + sign * offset_xyz[1]
    out["z"] = out["z"] + sign * offset_xyz[2]
    return out


def _denormalize_gaussians_cf(
    gaussians: dict,
    *,
    offset_xyz: torch.Tensor,
    scale: torch.Tensor,
) -> dict:

    out = dict(gaussians)
    if all(k in out for k in ("x", "y", "z")):
        out["x"] = out["x"] * scale + offset_xyz[0]
        out["y"] = out["y"] * scale + offset_xyz[1]
        out["z"] = out["z"] * scale + offset_xyz[2]
    if all(k in out for k in ("scale_0", "scale_1", "scale_2")):
        log_s = torch.log(scale)
        out["scale_0"] = out["scale_0"] + log_s
        out["scale_1"] = out["scale_1"] + log_s
        out["scale_2"] = out["scale_2"] + log_s
    return out


@dataclass(frozen=True)
class CompactSubmesh:


    vertices: np.ndarray
    faces: np.ndarray
    face_map_old_to_new: dict
    used_vertex_ids: np.ndarray
    vertex_map_old_to_new: dict

    def __iter__(self):

        yield self.vertices
        yield self.faces
        yield self.face_map_old_to_new


def _compact_submesh_from_faces(vertices: np.ndarray, faces: np.ndarray, face_indices: np.ndarray) -> CompactSubmesh:

    fi = np.asarray(face_indices, dtype=np.int64).reshape(-1)
    fi = np.unique(fi)
    fi = fi[(fi >= 0) & (fi < faces.shape[0])]
    fi.sort()
    F_sel = faces[fi]
    used_v = np.unique(F_sel.reshape(-1))
    used_v.sort()
    v_map = {int(old): int(i) for i, old in enumerate(used_v.tolist())}
    F_new = np.vectorize(v_map.get)(F_sel)
    V_new = vertices[used_v]
    face_map = {int(old): int(i) for i, old in enumerate(fi.tolist())}
    return CompactSubmesh(
        vertices=V_new,
        faces=F_new.astype(np.int64),
        face_map_old_to_new=face_map,
        used_vertex_ids=used_v,
        vertex_map_old_to_new=v_map,
    )


def _save_obj(path: str, V: np.ndarray, F: np.ndarray):
    trimesh.Trimesh(vertices=V, faces=F, process=False).export(path)


def _mesh_path_from_reposed_ply(ply_path: str) -> str | None:
    bn = os.path.basename(ply_path)
    m = re.match(r"^reposed_gs_targetframe(\d{4})(.*)\.ply$", bn)
    if not m:
        return None
    frame = m.group(1)
    suf = m.group(2) or ""

    cand0 = os.path.join(os.path.dirname(ply_path), f"reposed_mesh_targetframe{frame}{suf}.obj")
    if os.path.exists(cand0):
        return cand0


    base_dir = Path(ply_path).parent.parent
    cands = [
        base_dir / f"normed_nerf_mesh_src_frame_{frame}.obj",
        base_dir / f"nerf_mesh_target_frame_{frame}.obj",
        base_dir / f"nerf_mesh_targetframe{frame}.obj",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return None


def _mesh_path_from_reshaped_ply(ply_path: str) -> str | None:


    bn = os.path.basename(ply_path)
    m = re.search(r"reshaped_gs_target_shape_(\d{4})", bn)
    if not m:
        return None
    frame = m.group(1)
    base_dir = Path(ply_path).parent.parent
    mesh_dir = base_dir / f"reshape_no_repose_{frame}_meshes"
    is_cf = "cloth_fit_reshaped" in bn
    candidates = []
    if is_cf:
        m2 = re.search(r"cloth_fit_reshaped(.*)\.ply$", bn)
        suf = (m2.group(1) if m2 else "") or ""
        candidates += [
            mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}_cloth_fit_reshaped{suf}.obj",
            mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}_cloth_fit_reshaped.obj",
        ]
    candidates += [
        mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}.obj",
        mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}_cloth_fit_reshaped.obj",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _resolve_joint_names_path(smpl_params_path: str) -> str:
    p = Path(str(smpl_params_path))
    cands = [
        p.parent / "smpl_joint_names.txt",
        p.parent / "mesh" / "processed" / "smpl_joint_names.txt",
        p.parent.parent / "mesh" / "processed" / "smpl_joint_names.txt",
        p.parent.parent / "processed" / "smpl_joint_names.txt",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return str(cands[0])


def _resolve_lbs_weights_path(smpl_params_path: str) -> str:
    p = Path(str(smpl_params_path))
    cands = [
        p.parent / "smoothed_inpainted_weights.npy",
        p.parent / "mesh" / "processed" / "smoothed_inpainted_weights.npy",
        p.parent.parent / "mesh" / "processed" / "smoothed_inpainted_weights.npy",
    ]
    for c in cands:
        if c.exists():
            return str(c)
    return str(cands[1])


def _write_split_embedding(
    *,
    out_json_path: str,
    embed_full: dict,
    gaussian_mask: np.ndarray,
    face_map_old_to_new: dict,
    cano_mesh_filename: str,
    extra_meta: dict,
):
    idx = np.where(np.asarray(gaussian_mask).astype(bool))[0]
    fidx_old = np.asarray(embed_full["sample_fidxs"], dtype=np.int64)[idx]
    fidx_new = []
    for f in fidx_old.tolist():
        nf = face_map_old_to_new.get(int(f), None)
        if nf is None:
            raise KeyError(f"Face id {int(f)} not in submesh face map (cano_mesh={cano_mesh_filename}).")
        fidx_new.append(int(nf))

    out = {
        "cano_mesh": str(cano_mesh_filename),
        "sample_fidxs": fidx_new,
        "sample_bary": np.asarray(embed_full.get("sample_bary", embed_full.get("_sample_bary")), dtype=np.float32)[idx].tolist(),
        "_xyz": np.asarray(embed_full.get("_xyz"), dtype=np.float32)[idx].tolist(),
        "_rotation": np.asarray(embed_full.get("_rotation"), dtype=np.float32)[idx].tolist(),
        **extra_meta,
    }
    with open(out_json_path, "w") as f:
        json.dump(out, f, indent=2)


def _save_sub_lbs(
    *,
    out_path: str,
    weights_path: str,
    used_vids: np.ndarray,
    expected_nv: int,
    who: str,
):
    if not os.path.exists(weights_path):
        logger.warning(f"{who} LBS weights file not found: {weights_path}. Skipping.")
        return
    w = np.load(weights_path)
    if not isinstance(w, np.ndarray) or w.ndim != 2:
        logger.warning(f"{who} LBS weights malformed at {weights_path}: got shape={getattr(w, 'shape', None)}. Skipping.")
        return
    if int(w.shape[0]) != int(expected_nv):
        logger.warning(
            f"{who} LBS weights vertex-count mismatch: weights_nv={int(w.shape[0])} vs mesh_nv={int(expected_nv)}. "
            "Will still subset if indices are in-range."
        )
    used_vids = np.asarray(used_vids, dtype=np.int64).reshape(-1)
    if used_vids.size == 0:
        logger.warning(f"{who} used_vids is empty; skipping sub LBS write.")
        return
    if used_vids.max(initial=0) >= int(w.shape[0]):
        logger.warning(
            f"{who} used_vids out of range for {weights_path}: max_used_vid={int(used_vids.max())} >= weights_nv={int(w.shape[0])}. Skipping."
        )
        return
    sub = w[used_vids]
    np.save(out_path, sub.astype(np.float32))
    logger.info(f"Saved {who} sub LBS weights: {out_path}")


def save_head_body_separation_debug_artifacts(
    *,
    output_dir: str,
    device: str,
    user_A_id: str,
    model_B_id: str,
    A_orig_gaussians_path: str,
    B_body_gaussians_path: str,
    A_scan_mesh_path: str,
    B_scan_mesh_path: str,
    A_smpl_params_path: str,
    B_smpl_params_path: str,
    A_transl: torch.Tensor,
    A_scale: float,
    A_global_orient_matrix: torch.Tensor,
    A_pelvis_joint_j0: Optional[torch.Tensor],
    B_transl: torch.Tensor,
    B_scale: float,
    B_global_orient_matrix: torch.Tensor,
    B_pelvis_joint_j0: Optional[torch.Tensor],
    cloth_fit_offsets: Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]],
    A_prop_names_np: tuple,
    num_A_head_plus_neck_gaussians: int,
    num_B_body_gaussians: int,
    A_head_plus_neck_final_canon: dict,
    B_body_final_in_A_canon_space: dict,
    debug_A_head_plus_neck_cf: Optional[dict],
    debug_B_body_cf: Optional[dict],
    is_A_head_plus_neck_gaussian_mask: np.ndarray,
    is_B_body_gaussian_mask: np.ndarray,
    A_head_plus_neck_face_indices: np.ndarray,
    B_orig_head_face_indices: np.ndarray,
    A_segmentation_path: str,
    B_segmentation_path: str,
    A_embedding_path: str,
    B_embedding_path: str,
    B_neck_tube_mask: Optional[np.ndarray],
    A_joints_world: Optional[torch.Tensor],
    B_joints_world: Optional[torch.Tensor],

    A_hands_face_indices: Optional[np.ndarray] = None,
    is_A_hands_gaussian_mask: Optional[np.ndarray] = None,
    B_hands_face_indices: Optional[np.ndarray] = None,
    is_B_body_no_hands_gaussian_mask: Optional[np.ndarray] = None,
):
    def _safe(step: str, fn):
        try:
            fn()
        except Exception as e:
            logger.warning(f"[debug_artifacts] {step} failed: {e}")

    os.makedirs(output_dir, exist_ok=True)


    def _save_world_A_plys():
        if num_A_head_plus_neck_gaussians > 0:
            head_plus_neck_world_A = transform_gaussians(
                A_head_plus_neck_final_canon,
                A_prop_names_np,
                A_transl,
                A_scale,
                A_global_orient_matrix,
                to_canonical=False,
                device=device,
                pelvis_joint_j0=A_pelvis_joint_j0,
            )
            save_ply_gaussians(os.path.join(output_dir, "head_plus_neck_world_A.ply"), _to_np(head_plus_neck_world_A), A_prop_names_np)
        if num_B_body_gaussians > 0:
            body_world_A = transform_gaussians(
                B_body_final_in_A_canon_space,
                A_prop_names_np,
                A_transl,
                A_scale,
                A_global_orient_matrix,
                to_canonical=False,
                device=device,
                pelvis_joint_j0=A_pelvis_joint_j0,
            )
            save_ply_gaussians(os.path.join(output_dir, "body_only_world_A.ply"), _to_np(body_world_A), A_prop_names_np)

    _safe("save world_A split gaussians plys", _save_world_A_plys)


    def _save_joint_names_and_npz():
        import shutil

        A_joint_names_path = _resolve_joint_names_path(A_smpl_params_path)
        B_joint_names_path = _resolve_joint_names_path(B_smpl_params_path)
        joint_names_src = A_joint_names_path if os.path.exists(A_joint_names_path) else B_joint_names_path
        if os.path.exists(joint_names_src):
            shutil.copyfile(joint_names_src, os.path.join(output_dir, "smpl_joint_names.txt"))
            logger.info(f"Copied joint names to: {os.path.join(output_dir, 'smpl_joint_names.txt')}")
        else:
            logger.warning(
                f"Joint-names file not found at either {A_joint_names_path} or {B_joint_names_path}. Skipping copy."
            )

        def _save_joints_npz(out_name: str, joints_t: torch.Tensor):
            j_np = joints_t.detach().cpu().numpy().astype(np.float32)
            np.savez(os.path.join(output_dir, out_name), joints=j_np)
            logger.info(f"Saved joints npz: {os.path.join(output_dir, out_name)}")

        if A_joints_world is not None:
            _save_joints_npz("smpl_joints_A_world_A.npz", A_joints_world)
        if B_joints_world is not None:
            _save_joints_npz("smpl_joints_B_world_B.npz", B_joints_world)


        if A_joints_world is not None:
            A_j_canon = _transform_gaussian_positions(
                A_joints_world, A_transl, A_scale, A_global_orient_matrix, to_canonical=True, pelvis_joint_j0=A_pelvis_joint_j0
            )
            A_j_in_B = _transform_gaussian_positions(
                A_j_canon, B_transl, B_scale, B_global_orient_matrix, to_canonical=False, pelvis_joint_j0=None
            )
            _save_joints_npz("smpl_joints_A_world_B.npz", A_j_in_B)


        if B_joints_world is not None:
            B_j_for_A = B_joints_world
            if cloth_fit_offsets is not None:
                source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
                so = source_offset.view(1, 3)
                to = target_offset.view(1, 3)
                if (source_scale is not None) and (target_scale is not None):
                    ss = torch.clamp(source_scale.view(1, 1), min=1e-12)
                    ts = target_scale.view(1, 1)
                    B_j_for_A = (B_j_for_A - so) / ss * ts + to
                else:
                    B_j_for_A = (B_j_for_A - so) + to
            B_j_canon = _transform_gaussian_positions(
                B_j_for_A, B_transl, B_scale, B_global_orient_matrix, to_canonical=True, pelvis_joint_j0=B_pelvis_joint_j0
            )
            B_j_in_A = _transform_gaussian_positions(
                B_j_canon, A_transl, A_scale, A_global_orient_matrix, to_canonical=False, pelvis_joint_j0=A_pelvis_joint_j0
            )
            _save_joints_npz("smpl_joints_B_world_A.npz", B_j_in_A)

    _safe("save joint names + joints npz", _save_joint_names_and_npz)


    def _save_aligned_smpl_params_for_body_donor():
        if cloth_fit_offsets is None:
            return


        source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
        so = source_offset.detach().cpu().numpy().reshape(3,).astype(np.float32)
        to = target_offset.detach().cpu().numpy().reshape(3,).astype(np.float32)
        use_scale = (source_scale is not None) and (target_scale is not None)
        if use_scale:
            ss = float(source_scale.detach().cpu().view(-1)[0].item())
            ts = float(target_scale.detach().cpu().view(-1)[0].item())
            k = float(ts) / max(float(ss), 1e-12)
        else:
            k = 1.0
        delta = (to - float(k) * so).astype(np.float32)

        if not os.path.exists(B_smpl_params_path):
            raise FileNotFoundError(f"Missing B_smpl_params_path: {B_smpl_params_path}")
        if not os.path.exists(A_smpl_params_path):
            raise FileNotFoundError(f"Missing A_smpl_params_path: {A_smpl_params_path}")

        B_npz = np.load(B_smpl_params_path, allow_pickle=True)
        A_npz = np.load(A_smpl_params_path, allow_pickle=True)
        B_dict = {k0: B_npz[k0] for k0 in B_npz.files}
        A_dict = {k0: A_npz[k0] for k0 in A_npz.files}

        def _vec3_arr(x: np.ndarray) -> np.ndarray:
            arr = np.asarray(x)
            if arr.ndim == 1:
                return arr.astype(np.float32).reshape(1, 3)
            return arr.astype(np.float32).reshape(-1, 3)

        def _assign_vec3_like(dst: dict, key: str, val_f3: np.ndarray, like: np.ndarray):

            like_arr = np.asarray(like)
            if like_arr.ndim == 1:
                dst[key] = np.asarray(val_f3.reshape(3,), dtype=np.float32)
            else:
                dst[key] = np.asarray(val_f3, dtype=np.float32)


        out_B_space = dict(B_dict)


        if ("Th" in out_B_space) or ("transl" in out_B_space):
            if "Th" in out_B_space:
                tB = _vec3_arr(out_B_space["Th"])
                tB2 = float(k) * tB + delta.reshape(1, 3)
                _assign_vec3_like(out_B_space, "Th", tB2, out_B_space["Th"])
            if "transl" in out_B_space:
                tB = _vec3_arr(out_B_space["transl"])
                tB2 = float(k) * tB + delta.reshape(1, 3)
                _assign_vec3_like(out_B_space, "transl", tB2, out_B_space["transl"])
        else:
            logger.warning("[debug_artifacts] B smpl_params has no Th/transl; cannot apply alignment translation.")


        if "scale" in out_B_space:
            try:
                out_B_space["scale"] = (np.asarray(out_B_space["scale"], dtype=np.float32) * float(k)).astype(np.float32)
            except Exception:
                pass
        elif abs(float(k) - 1.0) > 1e-8:
            out_B_space["scale"] = np.asarray(float(k), dtype=np.float32)

        out_path_B = os.path.join(output_dir, "smpl_params_B_body_donor_aligned_world_B_body_donor.npz")
        np.savez(out_path_B, **out_B_space)


        out_A_space = dict(B_dict)


        if "Rh" in A_dict:
            out_A_space["Rh"] = A_dict["Rh"]
        if "global_orient" in A_dict:
            out_A_space["global_orient"] = A_dict["global_orient"]


        if "Th" in A_dict:
            tA = _vec3_arr(A_dict["Th"])
            tA_like = A_dict["Th"]
            keyA = "Th"
        elif "transl" in A_dict:
            tA = _vec3_arr(A_dict["transl"])
            tA_like = A_dict["transl"]
            keyA = "transl"
        else:

            tA = np.asarray(A_transl.detach().cpu().numpy(), dtype=np.float32).reshape(1, 3)
            tA_like = tA
            keyA = "Th"

        if "Th" in B_dict:
            tB = _vec3_arr(B_dict["Th"])
        elif "transl" in B_dict:
            tB = _vec3_arr(B_dict["transl"])
        else:
            tB = np.asarray(B_transl.detach().cpu().numpy(), dtype=np.float32).reshape(1, 3)

        F = int(max(tA.shape[0], tB.shape[0]))
        if tA.shape[0] == 1 and F > 1:
            tA = np.repeat(tA, F, axis=0)
        if tB.shape[0] == 1 and F > 1:
            tB = np.repeat(tB, F, axis=0)

        R_A = np.asarray(A_global_orient_matrix.detach().cpu().numpy(), dtype=np.float32).reshape(3, 3)
        R_B = np.asarray(B_global_orient_matrix.detach().cpu().numpy(), dtype=np.float32).reshape(3, 3)
        sA = float(A_scale)
        sB = float(B_scale)

        shift_term = (float(k) - 1.0) * tB + delta.reshape(1, 3)

        canon_shift = (shift_term @ R_B).astype(np.float32)
        canon_shift = canon_shift / max(float(sB), 1e-12)

        worldA_shift = (canon_shift @ R_A.T).astype(np.float32) * float(sA)
        tA_new = (tA + worldA_shift).astype(np.float32)


        if ("Th" in out_A_space) or (keyA == "Th"):
            _assign_vec3_like(out_A_space, "Th", tA_new, out_A_space.get("Th", tA_like))
        if ("transl" in out_A_space) or (keyA == "transl"):
            _assign_vec3_like(out_A_space, "transl", tA_new, out_A_space.get("transl", tA_like))


        if "scale" in out_A_space:
            try:
                out_A_space["scale"] = (np.asarray(out_A_space["scale"], dtype=np.float32) * float(k)).astype(np.float32)
            except Exception:
                pass
        elif abs(float(k) - 1.0) > 1e-8:
            out_A_space["scale"] = np.asarray(float(k), dtype=np.float32)

        out_path_A = os.path.join(output_dir, "smpl_params_B_body_donor_aligned_world_A.npz")
        np.savez(out_path_A, **out_A_space)


        meta = {
            "B_smpl_params_path": str(B_smpl_params_path),
            "A_smpl_params_path": str(A_smpl_params_path),
            "use_scale": bool(use_scale),
            "k": float(k),
            "source_offset": so.tolist(),
            "target_offset": to.tolist(),
            "delta": delta.tolist(),
            "out_world_B_body_donor": os.path.basename(out_path_B),
            "out_world_A": os.path.basename(out_path_A),
            "notes": [
                "world_B_body_donor: updates B translation by t' = k*t + delta (delta = target_offset - k*source_offset).",
                "world_A: uses A global orientation and shifts translation by rotating delta through canonical via (R_B, R_A).",
            ],
        }
        Path(os.path.join(output_dir, "smpl_params_B_body_donor_aligned_meta.json")).write_text(json.dumps(meta, indent=2) + "\n")
        logger.info(f"Wrote aligned SMPL params: {out_path_B} and {out_path_A}")

    _safe("save aligned smpl params for body donor", _save_aligned_smpl_params_for_body_donor)


    def _save_joints_pointclouds():
        if A_joints_world is not None:
            trimesh.PointCloud(A_joints_world.detach().cpu().numpy()).export(os.path.join(output_dir, "joints_world_A.ply"))
        if cloth_fit_offsets is not None and B_joints_world is not None:
            source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
            use_scale = (source_scale is not None) and (target_scale is not None)
            jb_np = B_joints_world.detach().cpu().numpy()
            so = source_offset.detach().cpu().numpy().reshape(1, 3)
            to = target_offset.detach().cpu().numpy().reshape(1, 3)
            if use_scale:
                ss = float(source_scale.detach().cpu().view(-1)[0].item())
                ts = float(target_scale.detach().cpu().view(-1)[0].item())
                jb_aligned = (jb_np - so) / max(ss, 1e-12) * ts + to
            else:
                jb_aligned = (jb_np - so) + to
            trimesh.PointCloud(jb_aligned).export(os.path.join(output_dir, "joints_world_B_body_donor.ply"))

    _safe("save joints pointcloud plys", _save_joints_pointclouds)


    def _save_world_B_plys():
        if cloth_fit_offsets is None:
            return
        source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
        use_scale = (source_scale is not None) and (target_scale is not None)

        if debug_A_head_plus_neck_cf is not None:
            head_plus_neck_world_B_aligned = (
                _denormalize_gaussians_cf(debug_A_head_plus_neck_cf, offset_xyz=target_offset, scale=target_scale)
                if use_scale
                else _translate_gaussians_xyz(debug_A_head_plus_neck_cf, target_offset, sign=+1.0)
            )
            save_ply_gaussians(
                os.path.join(output_dir, "head_plus_neck_world_B_body_donor.ply"),
                _to_np(head_plus_neck_world_B_aligned),
                A_prop_names_np,
            )
        if debug_B_body_cf is not None:
            body_world_B_aligned = (
                _denormalize_gaussians_cf(debug_B_body_cf, offset_xyz=target_offset, scale=target_scale)
                if use_scale
                else _translate_gaussians_xyz(debug_B_body_cf, target_offset, sign=+1.0)
            )
            save_ply_gaussians(
                os.path.join(output_dir, "body_only_world_B_body_donor.ply"),
                _to_np(body_world_B_aligned),
                A_prop_names_np,
            )

    _safe("save world_B_body_donor split gaussians plys", _save_world_B_plys)


    def _save_mesh_based_artifacts():
        A_full_mesh_path = _mesh_path_from_reposed_ply(A_orig_gaussians_path) or str(A_scan_mesh_path)
        B_full_mesh_path = _mesh_path_from_reshaped_ply(B_body_gaussians_path) or str(B_scan_mesh_path)
        if not os.path.exists(A_full_mesh_path):
            raise FileNotFoundError(f"Head-donor mesh not found: {A_full_mesh_path}")
        if not os.path.exists(B_full_mesh_path):
            raise FileNotFoundError(f"Body-donor mesh not found: {B_full_mesh_path}")

        A_full_mesh = load_mesh(A_full_mesh_path)
        B_full_mesh = load_mesh(B_full_mesh_path)
        A_faces = np.asarray(A_full_mesh.faces, dtype=np.int64)
        A_verts = np.asarray(A_full_mesh.vertices, dtype=np.float32)
        B_faces = np.asarray(B_full_mesh.faces, dtype=np.int64)
        B_verts = np.asarray(B_full_mesh.vertices, dtype=np.float32)

        A_hn_faces = np.asarray(A_head_plus_neck_face_indices, dtype=np.int64).reshape(-1)
        B_body_faces = np.setdiff1d(
            np.arange(B_faces.shape[0], dtype=np.int64),
            np.asarray(B_orig_head_face_indices, dtype=np.int64).reshape(-1),
            assume_unique=False,
        )

        B_body_no_hands_faces = None
        if B_hands_face_indices is not None:
            try:
                B_hands_faces = np.asarray(B_hands_face_indices, dtype=np.int64).reshape(-1)
                B_body_no_hands_faces = np.setdiff1d(B_body_faces, B_hands_faces, assume_unique=False)
            except Exception:
                B_body_no_hands_faces = None


        B_verts_for_A = B_verts
        if cloth_fit_offsets is not None:
            source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
            so = source_offset.detach().cpu().numpy().reshape(1, 3)
            to = target_offset.detach().cpu().numpy().reshape(1, 3)
            if (source_scale is not None) and (target_scale is not None):
                ss = float(source_scale.detach().cpu().view(-1)[0].item())
                ts = float(target_scale.detach().cpu().view(-1)[0].item())
                B_verts_for_A = (B_verts_for_A - so) / max(ss, 1e-12) * ts + to
            else:
                B_verts_for_A = (B_verts_for_A - so) + to

        B_v = torch.from_numpy(B_verts_for_A).to(device=device, dtype=torch.float32)
        B_canon = _transform_gaussian_positions(
            B_v, B_transl, B_scale, B_global_orient_matrix, to_canonical=True, pelvis_joint_j0=B_pelvis_joint_j0
        )
        B_in_A_world = _transform_gaussian_positions(
            B_canon, A_transl, A_scale, A_global_orient_matrix, to_canonical=False, pelvis_joint_j0=A_pelvis_joint_j0
        ).detach().cpu().numpy()

        hn_A = _compact_submesh_from_faces(A_verts, A_faces, A_hn_faces)
        body_A = _compact_submesh_from_faces(B_in_A_world, B_faces, B_body_faces)
        V_hn_A, F_hn_A, hn_face_map = hn_A
        V_body_A, F_body_A, body_face_map = body_A
        A_hn_used_v = hn_A.used_vertex_ids
        hn_obj_name = "head_plus_neck_world_A.obj"
        body_obj_name = "body_only_world_A.obj"
        _save_obj(os.path.join(output_dir, hn_obj_name), V_hn_A, F_hn_A)
        _save_obj(os.path.join(output_dir, body_obj_name), V_body_A, F_body_A)


        A_embed_full = load_embedding_full(A_embedding_path)
        B_embed_full = load_embedding_full(B_embedding_path)
        _write_split_embedding(
            out_json_path=os.path.join(output_dir, "embedding_head_plus_neck.json"),
            embed_full=A_embed_full,
            gaussian_mask=is_A_head_plus_neck_gaussian_mask,
            face_map_old_to_new=hn_face_map,
            cano_mesh_filename=hn_obj_name,
            extra_meta={
                "part": "head_plus_neck",
                "source_subject_id": str(user_A_id),
                "body_subject_id": str(model_B_id),
                "source_embedding_json": str(A_embedding_path),
            },
        )
        _write_split_embedding(
            out_json_path=os.path.join(output_dir, "embedding_body_only.json"),
            embed_full=B_embed_full,
            gaussian_mask=is_B_body_gaussian_mask,
            face_map_old_to_new=body_face_map,
            cano_mesh_filename=body_obj_name,
            extra_meta={
                "part": "body_only",
                "source_subject_id": str(model_B_id),
                "body_subject_id": str(user_A_id),
                "source_embedding_json": str(B_embedding_path),
            },
        )


        if A_hands_face_indices is not None and is_A_hands_gaussian_mask is not None:
            try:
                A_hands_faces = np.asarray(A_hands_face_indices, dtype=np.int64).reshape(-1)
                hands_A = _compact_submesh_from_faces(A_verts, A_faces, A_hands_faces)
                V_hands_A, F_hands_A, hands_face_map = hands_A
                hands_obj_name = "hands_world_A.obj"
                _save_obj(os.path.join(output_dir, hands_obj_name), V_hands_A, F_hands_A)
                _write_split_embedding(
                    out_json_path=os.path.join(output_dir, "embedding_hands.json"),
                    embed_full=A_embed_full,
                    gaussian_mask=np.asarray(is_A_hands_gaussian_mask).astype(bool),
                    face_map_old_to_new=hands_face_map,
                    cano_mesh_filename=hands_obj_name,
                    extra_meta={
                        "part": "hands",
                        "source_subject_id": str(user_A_id),
                        "body_subject_id": str(model_B_id),
                        "source_embedding_json": str(A_embedding_path),
                    },
                )
                A_hands_used_v = hands_A.used_vertex_ids
            except Exception as e_hands_mesh:
                logger.warning(f"Hands split (A) failed: {e_hands_mesh}")
                A_hands_used_v = None
        else:
            A_hands_used_v = None

        if B_body_no_hands_faces is not None and is_B_body_no_hands_gaussian_mask is not None:
            try:
                body_nh_A = _compact_submesh_from_faces(B_in_A_world, B_faces, np.asarray(B_body_no_hands_faces, dtype=np.int64))
                V_body_nh_A, F_body_nh_A, body_nh_face_map = body_nh_A
                body_nh_obj_name = "body_no_hands_world_A.obj"
                _save_obj(os.path.join(output_dir, body_nh_obj_name), V_body_nh_A, F_body_nh_A)
                _write_split_embedding(
                    out_json_path=os.path.join(output_dir, "embedding_body_no_hands.json"),
                    embed_full=B_embed_full,
                    gaussian_mask=np.asarray(is_B_body_no_hands_gaussian_mask).astype(bool),
                    face_map_old_to_new=body_nh_face_map,
                    cano_mesh_filename=body_nh_obj_name,
                    extra_meta={
                        "part": "body_no_hands",
                        "source_subject_id": str(model_B_id),
                        "body_subject_id": str(user_A_id),
                        "source_embedding_json": str(B_embedding_path),
                    },
                )
                B_body_no_hands_used_v = body_nh_A.used_vertex_ids
            except Exception as e_body_nh_mesh:
                logger.warning(f"Body-no-hands split (B) failed: {e_body_nh_mesh}")
                B_body_no_hands_used_v = None
        else:
            B_body_no_hands_used_v = None


        import pickle

        def _load_pkl(p: str) -> dict:
            with open(p, "rb") as f:
                return pickle.load(f)

        def _subset_label_pkl(label_data: dict, used_vids: np.ndarray) -> dict:
            out = dict(label_data) if isinstance(label_data, dict) else {"scan_labels": label_data}
            if "scan_labels" not in out:
                out = {"scan_labels": out}
            scan_labels = np.asarray(out["scan_labels"])
            n = int(scan_labels.shape[0])
            used_vids = np.asarray(used_vids, dtype=np.int64).reshape(-1)

            def _maybe_subset(v):
                if isinstance(v, np.ndarray) and v.shape[0] == n:
                    return v[used_vids]
                if isinstance(v, list) and len(v) == n:
                    return [v[int(i)] for i in used_vids.tolist()]
                return v

            out2 = {k: _maybe_subset(v) for k, v in out.items()}
            out2["scan_labels"] = scan_labels[used_vids]
            return out2

        def _dump_pkl(path: str, data: dict):
            with open(path, "wb") as f:
                pickle.dump(data, f)
            logger.info(f"Saved split label pkl: {path}")

        A_label_full = _load_pkl(A_segmentation_path)
        A_body_faces = np.setdiff1d(
            np.arange(A_faces.shape[0], dtype=np.int64),
            np.asarray(A_hn_faces, dtype=np.int64).reshape(-1),
            assume_unique=False,
        )
        A_body_used_v = _compact_submesh_from_faces(A_verts, A_faces, A_body_faces).used_vertex_ids
        _dump_pkl(os.path.join(output_dir, "label_A_head_plus_neck.pkl"), _subset_label_pkl(A_label_full, A_hn_used_v))
        _dump_pkl(os.path.join(output_dir, "label_A_body_only.pkl"), _subset_label_pkl(A_label_full, A_body_used_v))
        if A_hands_used_v is not None:
            _dump_pkl(os.path.join(output_dir, "label_A_hands.pkl"), _subset_label_pkl(A_label_full, A_hands_used_v))

        B_label_full = _load_pkl(B_segmentation_path)


        B_head_faces = np.asarray(B_orig_head_face_indices, dtype=np.int64).reshape(-1)
        B_hn_used_v = _compact_submesh_from_faces(B_verts, B_faces, B_head_faces).used_vertex_ids


        B_body_used_v2 = _compact_submesh_from_faces(B_verts, B_faces, B_body_faces).used_vertex_ids
        _dump_pkl(os.path.join(output_dir, "label_B_head_plus_neck.pkl"), _subset_label_pkl(B_label_full, B_hn_used_v))
        _dump_pkl(os.path.join(output_dir, "label_B_body_only.pkl"), _subset_label_pkl(B_label_full, B_body_used_v2))
        if B_body_no_hands_used_v is not None:
            _dump_pkl(os.path.join(output_dir, "label_B_body_no_hands.pkl"), _subset_label_pkl(B_label_full, B_body_no_hands_used_v))


        A_lbs_path = _resolve_lbs_weights_path(A_smpl_params_path)
        B_lbs_path = _resolve_lbs_weights_path(B_smpl_params_path)
        _save_sub_lbs(
            out_path=os.path.join(output_dir, "smoothed_inpainted_weights_A_head_plus_neck.npy"),
            weights_path=A_lbs_path,
            used_vids=A_hn_used_v,
            expected_nv=int(A_verts.shape[0]),
            who="A",
        )
        if A_hands_used_v is not None:
            _save_sub_lbs(
                out_path=os.path.join(output_dir, "smoothed_inpainted_weights_A_hands.npy"),
                weights_path=A_lbs_path,
                used_vids=A_hands_used_v,
                expected_nv=int(A_verts.shape[0]),
                who="A",
            )
        _save_sub_lbs(
            out_path=os.path.join(output_dir, "smoothed_inpainted_weights_B_body_only.npy"),
            weights_path=B_lbs_path,
            used_vids=B_body_used_v2,
            expected_nv=int(B_verts.shape[0]),
            who="B",
        )
        if B_body_no_hands_used_v is not None:
            _save_sub_lbs(
                out_path=os.path.join(output_dir, "smoothed_inpainted_weights_B_body_no_hands.npy"),
                weights_path=B_lbs_path,
                used_vids=B_body_no_hands_used_v,
                expected_nv=int(B_verts.shape[0]),
                who="B",
            )


        if cloth_fit_offsets is not None:
            source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
            use_scale = (source_scale is not None) and (target_scale is not None)

            A_v = torch.from_numpy(A_verts).to(device=device, dtype=torch.float32)
            A_canon = _transform_gaussian_positions(
                A_v, A_transl, A_scale, A_global_orient_matrix, to_canonical=True, pelvis_joint_j0=A_pelvis_joint_j0
            )
            A_in_B_world = _transform_gaussian_positions(
                A_canon, B_transl, B_scale, B_global_orient_matrix, to_canonical=False, pelvis_joint_j0=None
            ).detach().cpu().numpy()

            so = source_offset.detach().cpu().numpy().reshape(1, 3)
            to = target_offset.detach().cpu().numpy().reshape(1, 3)
            if use_scale:
                ss = float(source_scale.detach().cpu().view(-1)[0].item())
                ts = float(target_scale.detach().cpu().view(-1)[0].item())
                B_aligned = (B_verts - so) / max(ss, 1e-12) * ts + to
            else:
                B_aligned = (B_verts - so) + to

            V_hn_B, F_hn_B, _ = _compact_submesh_from_faces(A_in_B_world, A_faces, A_hn_faces)
            V_body_B, F_body_B, _ = _compact_submesh_from_faces(B_aligned, B_faces, B_body_faces)
            _save_obj(os.path.join(output_dir, "head_plus_neck_world_B_body_donor.obj"), V_hn_B, F_hn_B)
            _save_obj(os.path.join(output_dir, "body_only_world_B_body_donor.obj"), V_body_B, F_body_B)


            try:
                if use_scale:
                    ts = float(target_scale.detach().cpu().view(-1)[0].item())
                    ts = max(ts, 1e-12)
                    V_hn_opt = (V_hn_B - to.reshape(1, 3)) / ts
                    V_body_opt = (V_body_B - to.reshape(1, 3)) / ts
                else:
                    V_hn_opt = V_hn_B - to.reshape(1, 3)
                    V_body_opt = V_body_B - to.reshape(1, 3)
                _save_obj(os.path.join(output_dir, "head_plus_neck_clothfit_opt.obj"), V_hn_opt, F_hn_B)
                _save_obj(os.path.join(output_dir, "body_only_clothfit_opt.obj"), V_body_opt, F_body_B)
            except Exception as e_opt:
                logger.warning(f"clothfit opt-space obj write failed: {e_opt}")


            if A_hands_face_indices is not None:
                try:
                    V_hands_B, F_hands_B, _ = _compact_submesh_from_faces(
                        A_in_B_world, A_faces, np.asarray(A_hands_face_indices, dtype=np.int64).reshape(-1)
                    )
                    _save_obj(os.path.join(output_dir, "hands_world_B_body_donor.obj"), V_hands_B, F_hands_B)
                except Exception as e_handsB:
                    logger.warning(f"hands_world_B_body_donor.obj write failed: {e_handsB}")
            if B_body_no_hands_faces is not None:
                try:
                    V_body_nh_B, F_body_nh_B, _ = _compact_submesh_from_faces(
                        B_aligned, B_faces, np.asarray(B_body_no_hands_faces, dtype=np.int64).reshape(-1)
                    )
                    _save_obj(os.path.join(output_dir, "body_no_hands_world_B_body_donor.obj"), V_body_nh_B, F_body_nh_B)
                except Exception as e_bodyNhB:
                    logger.warning(f"body_no_hands_world_B_body_donor.obj write failed: {e_bodyNhB}")

    _safe("save mesh-based artifacts (objs/embeddings/seg/lbs)", _save_mesh_based_artifacts)
