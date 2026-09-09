import numpy as np
import plyfile
import json
import trimesh
from scipy.spatial.distance import cdist
import torch
import pytorch3d.transforms as p3d_transforms
import pickle
from src.utils.color.sh import SH2RGB, RGB2SH
from src.utils.color.color_transfer import transfer_skin_color_pytorch, calculate_weighted_lab_stats
from src.utils.color.color_format import rgb_to_lab, lab_to_rgb
from src.utils.smplx_utils.smplx_utils import create_smplx_model_for_neus2_reposing, compute_reference_transform_data
import os
import argparse
import glob
import re
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger


from src.scripts.swapping.data.constants import SURFACE_LABELS, DETAILED_SURFACE_LABELS
from src.scripts.swapping.data.loaders import (
    load_ply_gaussians, load_segmentation_labels, load_smplx_vert_segmentation,
    load_embedding_json, load_embedding_full, load_mesh, load_smpl_transform_params_for_frame
)
from src.scripts.swapping.data.savers import save_ply_gaussians
from src.scripts.swapping.utils.smpl_utils import get_pelvis_joint_j0, get_body_joints_world
from src.scripts.swapping.geometry.transformations import (
    _transform_gaussian_positions, _transform_gaussian_rotations,
    _transform_gaussian_scales, transform_gaussians
)
from src.scripts.swapping.geometry.neck_cut import NeckCutParams, compute_neck_axis_and_masks
from src.scripts.swapping.geometry.analysis import (
    get_smplx_head_center, get_connected_components, identify_head_region,
    get_labeled_part_verts_and_faces
)
from src.scripts.swapping.color.processing import prepare_skin_gaussian_colors_for_transfer
from src.scripts.swapping.color.skin_parts import (
    build_pooled_face_index_maps,
    gaussian_mask_from_face_indices,
    robust_source_lab_stats_from_gaussians_canon,
)
from src.scripts.swapping.color.fallbacks import fallback_order_for_target, resolve_source_key_for_target
from src.scripts.swapping.config.argument_parser import create_argument_parser
from src.scripts.swapping.config.path_config import resolve_ply_path_with_rigid_head_support
from src.scripts.swapping.debug_artifacts import save_head_body_separation_debug_artifacts
from src.scripts.swapping.smplx_hands_mesh import init_dummy_gray_hands_gaussians_from_smplx_mesh
from src.scripts.swapping.smplx_hands_proxy_artifacts import emit_smplx_mesh_hands_proxy_artifacts


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


def _resolve_splatting_subject_dir_name(output_splatting_root: str, subject_id: str) -> str:

    base = os.path.join(str(output_splatting_root), "output-splatting")
    sid = str(subject_id)
    try:
        if os.path.isdir(base):
            cands = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)) and (sid in d)]
            if len(cands) == 1:
                return cands[0]
            if len(cands) > 1:

                ends = [d for d in cands if d.endswith(sid)]
                if len(ends) == 1:
                    return ends[0]
                return sorted(cands, key=lambda x: (len(x), x))[0]
    except Exception:
        pass
    return f"neusclean_sub{sid}"


def _candidate_padded_ids(subject_id: str) -> list[str]:

    sid = str(subject_id)
    cands = [sid]
    if sid.isdigit():
        cands.append(f"{int(sid):04d}")

    out = []
    for x in cands:
        if x not in out:
            out.append(x)
    return out


def _find_cloth_fit_offsets_json(
    *,
    dataset_root: str,
    avatar_subject_id: str,
    garment_subject_id: str,
    cloth_fit_suffix: str = "",
) -> Optional[Path]:

    cloth_fit_root = Path(dataset_root) / "cloth_fit_output"
    if not cloth_fit_root.exists():
        return None

    avatar_cands = _candidate_padded_ids(avatar_subject_id)
    garment_cands = _candidate_padded_ids(garment_subject_id)

    matches: list[Path] = []
    for a in avatar_cands:
        for g in garment_cands:
            prefix = f"{a}_avatar_{g}_garment"
            for d in cloth_fit_root.glob(f"{prefix}*"):
                if not d.is_dir():
                    continue
                p = d / "normalization_offsets.json"
                if p.exists():
                    matches.append(p)

    if not matches:
        return None

    suf_raw = str(cloth_fit_suffix or "").strip()
    suf_variants: list[str] = []
    if suf_raw:
        suf_variants.append(suf_raw)
        if suf_raw.startswith("_"):
            suf_variants.append(suf_raw.lstrip("_"))
        else:
            suf_variants.append("_" + suf_raw)
    else:
        suf_variants.append("")

    chosen: Optional[Path] = None
    if len(matches) > 1 and suf_raw:

        for suf in suf_variants:
            if not suf:
                continue
            preferred = [p for p in matches if p.parent.name.endswith(f"_garment{suf}")]
            if preferred:
                chosen = sorted(preferred, key=lambda x: str(x))[0]
                break

        if chosen is None:
            preferred = [p for p in matches if suf_raw in p.parent.name]
            if preferred:
                chosen = sorted(preferred, key=lambda x: str(x))[0]

    if chosen is None:
        chosen = sorted(matches, key=lambda x: str(x))[0]

    if len(matches) > 1:
        logger.warning(
            f"Multiple cloth-fit offsets found; cloth_fit_suffix='{suf_raw}' -> using: {chosen}"
        )
    return chosen


def _load_cloth_fit_offsets(
    *,
    dataset_root: str,
    avatar_subject_id: str,
    garment_subject_id: str,
    device: str,
    cloth_fit_suffix: str = "",
) -> Optional[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:

    p = _find_cloth_fit_offsets_json(
        dataset_root=dataset_root,
        avatar_subject_id=avatar_subject_id,
        garment_subject_id=garment_subject_id,
        cloth_fit_suffix=str(cloth_fit_suffix or ""),
    )
    if p is None:
        return None
    try:
        data = json.loads(p.read_text())
        source_offset = torch.tensor(data["source_offset"], dtype=torch.float32, device=device).view(3,)
        target_offset = torch.tensor(data["target_offset"], dtype=torch.float32, device=device).view(3,)
        source_scale = None
        target_scale = None
        if "source_scale" in data and "target_scale" in data:
            source_scale = torch.tensor(float(data["source_scale"]), dtype=torch.float32, device=device)
            target_scale = torch.tensor(float(data["target_scale"]), dtype=torch.float32, device=device)
    except Exception as e:
        logger.warning(f"Failed to load cloth-fit offsets from {p}: {e}")
        return None
    logger.info(f"Loaded cloth-fit offsets from {p}")
    return source_offset, target_offset, source_scale, target_scale


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


def _normalize_gaussians_cf(
    gaussians: dict,
    *,
    offset_xyz: torch.Tensor,
    scale: torch.Tensor,
) -> dict:

    out = dict(gaussians)
    if all(k in out for k in ("x", "y", "z")):
        out["x"] = (out["x"] - offset_xyz[0]) / scale
        out["y"] = (out["y"] - offset_xyz[1]) / scale
        out["z"] = (out["z"] - offset_xyz[2]) / scale

    if all(k in out for k in ("scale_0", "scale_1", "scale_2")):
        log_s = torch.log(scale)
        out["scale_0"] = out["scale_0"] - log_s
        out["scale_1"] = out["scale_1"] - log_s
        out["scale_2"] = out["scale_2"] - log_s
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

def _save_body_donator_full_to_file(
    B_body_gaussians_in_canon_space: dict,
    B_body_prop_names_np: list,
    A_transl,
    A_scale,
    A_global_orient_matrix,
    A_pelvis_joint_j0,
    device,
    output_dir: str,
    swapped_filename_without_ext: str
) -> str:

    logger.info("Saving body donator full Gaussian (B_full) in head donator's world (A's world)...")


    B_full_canon = B_body_gaussians_in_canon_space


    B_full_in_A_world = transform_gaussians(
        B_full_canon, B_body_prop_names_np,
        A_transl, A_scale, A_global_orient_matrix,
        to_canonical=False, device=device, pelvis_joint_j0=A_pelvis_joint_j0
    )


    output_filename_full = f"body_donator_full_{swapped_filename_without_ext}.ply"


    os.makedirs(output_dir, exist_ok=True)
    output_ply_path_full = os.path.join(output_dir, output_filename_full)


    B_full_in_A_world_np = {name: data.cpu().numpy() for name, data in B_full_in_A_world.items()}
    save_ply_gaussians(output_ply_path_full, B_full_in_A_world_np, B_body_prop_names_np)
    logger.info(f"Saved body donator full in head world to: {output_ply_path_full}")

    return output_ply_path_full


def swap_head_gaussians(
    user_A_id: str,
    model_B_id: str,
    paths_config: dict,
    smplx_vert_seg_path: str,
    current_surface_labels: list[str],
    use_direct_head_label_A: bool, direct_head_label_name_A: str, fallback_skin_label_name_A: str, hair_label_name_A: str,
    use_direct_head_label_B: bool, direct_head_label_name_B: str, fallback_skin_label_name_B: str, hair_label_name_B: str,
    skin_label_for_color_transfer: str,
    color_transfer_opacity_threshold: float,
    perform_color_transfer: bool,
    debug_color_transfer_save_mesh: bool,
    debug_color_transfer_force_color: bool,
    debug_hardcoded_lab_mean: tuple,
    debug_hardcoded_lab_std: tuple,
    target_body_parts_for_color_transfer: list[str],
    use_detailed_labels_config: bool,
    color_transfer_use_all_gaussians_for_stats: bool,
    color_transfer_use_opacity_weighting: bool,
    repose_load_dir_subfix: str,
    gender_A_for_pelvis_joint: str,
    gender_B_for_pelvis_joint: str,
    direct_swap_mode: bool,
    swap_back_mode: bool = False,
    enable_rigid_head_reposing: bool = False,
    save_body_donator_full_in_head_world: bool = False,
    save_swap_debug_artifacts: bool = False,
    save_head_body_separation_debug: bool = False,
    swap_hands: bool = False,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")


    swap_hands_enabled = bool(swap_hands)
    if swap_hands_enabled and ("hands" not in list(current_surface_labels)):
        logger.warning(
            "swap_hands requested but 'hands' label is not available in current_surface_labels. "
            "Disable --use_detailed_labels or label set mismatch? Falling back to head-only swap."
        )
        swap_hands_enabled = False
    color_transfer_log: dict = {
        "head_donor_subject_id": str(user_A_id),
        "body_donor_subject_id": str(model_B_id),
        "mode": {
            "perform_color_transfer": bool(perform_color_transfer),
            "use_detailed_labels_config": bool(use_detailed_labels_config),
            "swap_hands": bool(swap_hands_enabled),
            "swap_hands_source": str(getattr(args, "swap_hands_source", "neus_gs")),
            "opacity_threshold": float(color_transfer_opacity_threshold) if color_transfer_opacity_threshold is not None else None,
            "use_all_gaussians_for_stats": bool(color_transfer_use_all_gaussians_for_stats),
            "use_opacity_weighting": bool(color_transfer_use_opacity_weighting),
            "neck_plane_lift_ratio": float(getattr(args, "neck_plane_lift_ratio", 0.2)),
        },
        "pooled_sources_A": {},
        "parts": [],
    }

    smplx_vert_seg = None
    if not use_direct_head_label_A or not use_direct_head_label_B:
        logger.info("Loading SMPLX vertex segmentation as it's needed for fallback head ID.")
        smplx_vert_seg = load_smplx_vert_segmentation(smplx_vert_seg_path)


    if 'smpl_params_npz_map' in paths_config and user_A_id in paths_config['smpl_params_npz_map']:
        A_smpl_params_path = paths_config['smpl_params_npz_map'][user_A_id]
    else:
        A_smpl_params_path = paths_config['smpl_params_npz'].format(id=user_A_id)
    A_dataset_type_for_transform = _infer_dataset_type_from_any_path(A_smpl_params_path)
    A_global_orient_matrix, A_transl, A_scale = load_smpl_transform_params_for_frame(
        A_smpl_params_path, frame_idx=0, device=device, dataset_type=A_dataset_type_for_transform
    )
    logger.info(f"User A ({user_A_id}) SMPL params: scale={A_scale:.4f}, transl={A_transl.cpu().numpy().tolist()}")


    if 'smpl_params_npz_map' in paths_config and model_B_id in paths_config['smpl_params_npz_map']:
        B_smpl_params_path = paths_config['smpl_params_npz_map'][model_B_id]
    else:
        B_smpl_params_path = paths_config['smpl_params_npz'].format(id=model_B_id)
    B_dataset_type_for_transform = _infer_dataset_type_from_any_path(B_smpl_params_path)
    B_global_orient_matrix, B_transl, B_scale = load_smpl_transform_params_for_frame(
        B_smpl_params_path, frame_idx=0, device=device, dataset_type=B_dataset_type_for_transform
    )


    logger.info(f"Processing User A (ID: {user_A_id}) for head extraction...")


    iter_paths = paths_config.get('_iteration_paths', {})
    user_A_iter_path = iter_paths.get('user_A', 'latest')
    splat_subdir_map = paths_config.get("_splat_subdir_map", {}) if isinstance(paths_config, dict) else {}
    A_splat_subdir = splat_subdir_map.get(user_A_id, f"neusclean_sub{user_A_id}")
    B_splat_subdir = splat_subdir_map.get(model_B_id, f"neusclean_sub{model_B_id}")
    A_orig_gaussians_path = resolve_ply_path_with_rigid_head_support(
        paths_config['original_avatar_ply'], enable_rigid_head_reposing,
        id=user_A_id, iter_path=user_A_iter_path, splat_subdir=A_splat_subdir
    )

    A_orig_gaussians_np, A_prop_names_np = load_ply_gaussians(A_orig_gaussians_path)

    A_orig_gaussians_torch = {name: torch.from_numpy(data).to(device) for name, data in A_orig_gaussians_np.items()}

    logger.info(f"Transforming User A's original Gaussians to A's canonical space...")


    A_use_pelvis_centering = A_dataset_type_for_transform not in ("talkbody4d", "mvhumannet", "actorshq")


    A_pelvis_joint_j0 = None
    try:
        if A_use_pelvis_centering:
            A_pelvis_joint_j0 = get_pelvis_joint_j0(
                A_smpl_params_path, frame_idx=0, device=device,
                dataset_type=A_dataset_type_for_transform, gender=gender_A_for_pelvis_joint
            )
            logger.info(f"Using User A pelvis joint j0 for rotation center: {A_pelvis_joint_j0.cpu().numpy()}")
        else:
            logger.info(f"User A dataset_type={A_dataset_type_for_transform}: using origin-centered rigid transform (no pelvis centering).")
    except Exception as e:
        logger.warning(f"Warning: Could not extract User A pelvis joint: {e}")
        logger.warning("Falling back to world origin rotation for User A")

    A_gaussians_canon_candidate = transform_gaussians(
        A_orig_gaussians_torch, A_prop_names_np, A_transl, A_scale, A_global_orient_matrix,
        to_canonical=True, device=device, pelvis_joint_j0=A_pelvis_joint_j0
    )


    if 'segmentation_pkl_map' in paths_config and user_A_id in paths_config['segmentation_pkl_map']:
        A_segmentation_path = paths_config['segmentation_pkl_map'][user_A_id]
    else:
        A_segmentation_path = paths_config['segmentation_pkl'].format(id=user_A_id)

    if 'embedding_json_map' in paths_config and user_A_id in paths_config['embedding_json_map']:
        A_embedding_path = paths_config['embedding_json_map'][user_A_id].format(iter_path=user_A_iter_path, splat_subdir=A_splat_subdir)
    else:
        A_embedding_path = paths_config['embedding_json'].format(id=user_A_id, iter_path=user_A_iter_path, splat_subdir=A_splat_subdir)

    if 'scan_mesh_obj_map' in paths_config and user_A_id in paths_config['scan_mesh_obj_map']:
        A_scan_mesh_path = paths_config['scan_mesh_obj_map'][user_A_id]
    else:
        A_scan_mesh_path = paths_config['scan_mesh_obj'].format(id=user_A_id)
    A_seg_labels = load_segmentation_labels(A_segmentation_path)
    A_sample_fidxs = load_embedding_json(A_embedding_path)
    A_scan_mesh = load_mesh(A_scan_mesh_path)


    A_neck_tube_mask = None
    A_above_neck_plane_mask = None
    A_joints_world = None
    try:
        A_joints_world = get_body_joints_world(
            A_smpl_params_path,
            frame_idx=0,
            device=device,
            dataset_type=A_dataset_type_for_transform,
            gender=gender_A_for_pelvis_joint,
        )
        A_neck_masks = compute_neck_axis_and_masks(
            np.asarray(A_scan_mesh.vertices, dtype=np.float32),
            A_joints_world.detach().cpu().numpy(),
            params=NeckCutParams(
                radius_ratio=0.35,
                plane_lift_ratio=float(getattr(args, "neck_plane_lift_ratio", 0.2)),
            ),
        )
        A_neck_tube_mask = A_neck_masks.get("neck_tube_mask", None)
        A_above_neck_plane_mask = A_neck_masks.get("above_neck_plane_mask", None)
        logger.info(
            f"A neck_tube_mask: {int(np.sum(A_neck_tube_mask)) if A_neck_tube_mask is not None else 0} verts; "
            f"above_plane: {int(np.sum(A_above_neck_plane_mask)) if A_above_neck_plane_mask is not None else 0} verts"
        )
        if debug_color_transfer_save_mesh and A_neck_tube_mask is not None:
            os.makedirs("output/debug", exist_ok=True)
            try:
                pts = np.asarray(A_scan_mesh.vertices, dtype=np.float32)[A_neck_tube_mask]
                trimesh.PointCloud(pts).export(f"output/debug/{user_A_id}_neck_tube_verts.ply")
            except Exception as e_vis:
                logger.warning(f"Failed to save A neck tube point cloud: {e_vis}")
    except Exception as e:
        logger.warning(f"Failed to compute A neck cut masks: {e}")

    A_smpl_mesh = None
    if not use_direct_head_label_A:
        A_smpl_mesh_path_for_head_id = paths_config['smpl_mesh_obj'].format(user_id=user_A_id, model_id=model_B_id)
        A_smpl_mesh = load_mesh(A_smpl_mesh_path_for_head_id)

    A_head_verts_indices, A_head_face_indices = identify_head_region(
        A_scan_mesh.vertices, A_scan_mesh.faces, A_seg_labels, current_surface_labels,
        hair_label_name=hair_label_name_A,
        use_direct_head_label=use_direct_head_label_A,
        direct_head_label_name=direct_head_label_name_A,
        smpl_mesh_vertices=A_smpl_mesh.vertices if A_smpl_mesh else None,
        smplx_vert_seg=smplx_vert_seg,
        fallback_skin_label_name=fallback_skin_label_name_A,
        neck_tube_mask=A_neck_tube_mask,
        above_neck_plane_mask=A_above_neck_plane_mask,
    )
    if not A_head_verts_indices.size:
        logger.error(f"Could not identify head vertices for User A ({user_A_id}). Aborting.")
        return


    is_A_head_gaussian = np.isin(A_sample_fidxs, A_head_face_indices)
    A_head_final_canon = {name: data[is_A_head_gaussian] for name, data in A_gaussians_canon_candidate.items()}
    num_A_head_gaussians = A_head_final_canon['x'].shape[0]
    logger.info(f"Extracted {num_A_head_gaussians} head Gaussians from User A (in A's canonical space).")
    if num_A_head_gaussians == 0:
        logger.warning(f"No head Gaussians found for User A ({user_A_id}). Swapped avatar will have no head from A.")


    is_A_head_plus_neck_gaussian_mask = is_A_head_gaussian
    A_head_plus_neck_face_indices = np.asarray(A_head_face_indices).reshape(-1)
    A_head_plus_neck_final_canon = A_head_final_canon
    num_A_head_plus_neck_gaussians = num_A_head_gaussians
    if save_head_body_separation_debug and A_neck_tube_mask is not None:
        try:
            faces_A = np.asarray(A_scan_mesh.faces, dtype=np.int64)
            neck_vert_mask = np.asarray(A_neck_tube_mask).astype(bool)
            if A_above_neck_plane_mask is not None:
                neck_vert_mask = neck_vert_mask & np.asarray(A_above_neck_plane_mask).astype(bool)
            neck_face_indices = np.where(np.any(neck_vert_mask[faces_A], axis=1))[0]
            head_plus_neck_face_indices = np.unique(
                np.concatenate([np.asarray(A_head_face_indices).reshape(-1), neck_face_indices], axis=0)
            )
            is_A_head_plus_neck_gaussian = np.isin(A_sample_fidxs, head_plus_neck_face_indices)
            is_A_head_plus_neck_gaussian_mask = is_A_head_plus_neck_gaussian
            A_head_plus_neck_face_indices = np.asarray(head_plus_neck_face_indices).reshape(-1)
            A_head_plus_neck_final_canon = {
                name: data[is_A_head_plus_neck_gaussian] for name, data in A_gaussians_canon_candidate.items()
            }
            num_A_head_plus_neck_gaussians = int(A_head_plus_neck_final_canon["x"].shape[0])
            logger.info(
                f"[debug] Extracted {num_A_head_plus_neck_gaussians} head+neck Gaussians from User A "
                f"(head-only={num_A_head_gaussians})."
            )
        except Exception as e_hn:
            logger.warning(f"[debug] Failed to build head+neck Gaussian subset; falling back to head-only. err={e_hn}")
            is_A_head_plus_neck_gaussian_mask = is_A_head_gaussian
            A_head_plus_neck_face_indices = np.asarray(A_head_face_indices).reshape(-1)
            A_head_plus_neck_final_canon = A_head_final_canon
            num_A_head_plus_neck_gaussians = num_A_head_gaussians


    smplx_hands_proxy = None
    A_hands_face_indices = np.array([], dtype=np.int64)
    is_A_hands_gaussian_mask = np.zeros((A_sample_fidxs.shape[0],), dtype=bool)
    A_hands_final_canon: dict = {name: data[:0] for name, data in A_gaussians_canon_candidate.items()}
    num_A_hands_gaussians = 0
    if swap_hands_enabled:
        swap_hands_source = str(getattr(args, "swap_hands_source", "neus_gs")).strip().lower()
        if swap_hands_source not in ("neus_gs", "smplx_mesh"):
            logger.warning(f"Unknown swap_hands_source={swap_hands_source!r}; falling back to 'neus_gs'.")
            swap_hands_source = "neus_gs"

        if swap_hands_source == "smplx_mesh":
            try:
                smplx_vert_seg, A_hands_final_canon, num_A_hands_gaussians, smplx_hands_proxy = (
                    init_dummy_gray_hands_gaussians_from_smplx_mesh(
                        args=args,
                        device=device,
                        smplx_vert_seg_path=str(smplx_vert_seg_path),
                        smplx_vert_seg=smplx_vert_seg,
                        A_smpl_params_path=str(A_smpl_params_path),
                        user_A_id=str(user_A_id),
                        model_B_id=str(model_B_id),
                        A_prop_names_np=tuple(A_prop_names_np),
                        B_transl=B_transl,
                        B_scale=B_scale,
                        B_global_orient_matrix=B_global_orient_matrix,
                    )
                )
                if num_A_hands_gaussians == 0:
                    logger.warning("swap_hands_source=smplx_mesh enabled but initialized 0 hands gaussians; proceeding head-only.")
            except Exception as e_handsA:
                logger.warning(f"swap_hands_source=smplx_mesh failed; proceeding head-only. err={e_handsA}")
                swap_hands_enabled = False
        else:

            try:
                _, A_hands_face_indices = get_labeled_part_verts_and_faces(
                    A_seg_labels, A_scan_mesh.faces, current_surface_labels, "hands"
                )
                A_hands_face_indices = np.asarray(A_hands_face_indices, dtype=np.int64).reshape(-1)
                is_A_hands_gaussian_mask = np.isin(A_sample_fidxs, A_hands_face_indices)

                is_A_hands_gaussian_mask = is_A_hands_gaussian_mask & (~np.asarray(is_A_head_gaussian).astype(bool))
                A_hands_final_canon = {
                    name: data[is_A_hands_gaussian_mask] for name, data in A_gaussians_canon_candidate.items()
                }
                num_A_hands_gaussians = int(A_hands_final_canon["x"].shape[0]) if "x" in A_hands_final_canon else 0
                logger.info(f"Extracted {num_A_hands_gaussians} hands Gaussians from User A (in A's canonical space).")
                if num_A_hands_gaussians == 0:
                    logger.warning("swap_hands enabled but found 0 hands gaussians for User A; proceeding head-only.")
            except Exception as e_handsA:
                logger.warning(f"Failed to extract A hands gaussians; proceeding head-only. err={e_handsA}")
                swap_hands_enabled = False


    A_style_sources = {}

    if perform_color_transfer and use_detailed_labels_config:
        logger.info("--- Preparing pooled skin-part sources for User A (DAG fallback) ---")
        A_face_map = build_pooled_face_index_maps(
            segmentation_labels=A_seg_labels,
            scan_mesh_faces=A_scan_mesh.faces,
            current_surface_labels=current_surface_labels,
            neck_tube_mask=A_neck_tube_mask,
        )

        pooled_source_keys = ["arms_both", "legs_both", "torso_non_neck", "neck", "hands_both"]
        for source_key in pooled_source_keys:
            face_idx = A_face_map.get(source_key, np.array([], dtype=int))
            is_A_src = gaussian_mask_from_face_indices(A_sample_fidxs, face_idx)
            if not np.any(is_A_src):
                A_style_sources[source_key] = {"status": "missing"}
                color_transfer_log["pooled_sources_A"][source_key] = {"status": "missing", "reason": "no_gaussians"}
                logger.warning(f"User A pooled source '{source_key}': no gaussians available.")
                continue

            A_src_gaussians_canon = {name: data[is_A_src] for name, data in A_gaussians_canon_candidate.items()}
            if not A_src_gaussians_canon or A_src_gaussians_canon.get("x", torch.empty(0)).shape[0] == 0:
                A_style_sources[source_key] = {"status": "missing"}
                color_transfer_log["pooled_sources_A"][source_key] = {"status": "missing", "reason": "empty_after_selection"}
                logger.warning(f"User A pooled source '{source_key}': empty gaussian dict after selection.")
                continue

            if color_transfer_use_opacity_weighting:
                logger.info(f"Calculating opacity-weighted color stats for pooled source '{source_key}'...")
                src_rgbs = torch.clamp(SH2RGB(
                    torch.stack(
                        [A_src_gaussians_canon["f_dc_0"], A_src_gaussians_canon["f_dc_1"], A_src_gaussians_canon["f_dc_2"]],
                        dim=-1,
                    )
                ), 0.0, 1.0)
                src_op = torch.sigmoid(A_src_gaussians_canon["opacity"]).float()
                neff = float(torch.sum(src_op).item())
                if neff < 100.0:
                    mean_rgb = src_rgbs.mean(dim=0).detach().cpu().tolist()
                    A_style_sources[source_key] = {"status": "too_small_or_noisy", "neff": neff, "mean_rgb": mean_rgb}
                    color_transfer_log["pooled_sources_A"][source_key] = {"status": "too_small_or_noisy", "neff": neff, "mean_rgb": mean_rgb}
                    logger.warning(f"A pooled source '{source_key}' weighted neff too small: {neff:.1f}")
                    continue
                src_lab_stats = calculate_weighted_lab_stats(src_rgbs, src_op)
                mean_rgb_w = (src_rgbs * src_op.view(-1, 1)).sum(dim=0) / torch.clamp(src_op.sum(), min=1e-6)
                A_style_sources[source_key] = {"status": "ok", "lab_stats": src_lab_stats, "neff": neff, "mean_rgb": mean_rgb_w.detach().cpu().tolist()}
                color_transfer_log["pooled_sources_A"][source_key] = {"status": "ok", "neff": neff, "mean_rgb": A_style_sources[source_key]["mean_rgb"]}
                logger.info(f"Stored weighted Lab stats for A pooled source '{source_key}'.")
            else:
                robust_stats = robust_source_lab_stats_from_gaussians_canon(
                    A_src_gaussians_canon,
                    opacity_threshold=color_transfer_opacity_threshold,
                    use_all_gaussians_for_stats=color_transfer_use_all_gaussians_for_stats,
                    mad_k=4.0,
                    min_neff=100.0,
                    min_ratio=0.01,
                )
                A_style_sources[source_key] = robust_stats
                color_transfer_log["pooled_sources_A"][source_key] = {
                    "status": robust_stats.get("status"),
                    "neff": robust_stats.get("neff"),
                    "n_selected": robust_stats.get("n_selected"),
                    "n_inlier": robust_stats.get("n_inlier"),
                    "mean_rgb": robust_stats.get("mean_rgb"),
                }
                if robust_stats.get("status") == "ok":
                    logger.info(
                        f"A pooled source '{source_key}' robust stats ok: "
                        f"neff={robust_stats['neff']:.1f}, inliers={robust_stats['n_inlier']}/{robust_stats['n_selected']}"
                    )
                else:
                    logger.warning(f"A pooled source '{source_key}': {robust_stats.get('status')}")


        try:
            rel_ratio = float(getattr(args, "color_transfer_source_relative_neff_ratio", 0.2))
            primary_keys = ["arms_both", "legs_both", "torso_non_neck"]
            ok_primary = [
                k for k in primary_keys
                if isinstance(A_style_sources.get(k), dict) and A_style_sources[k].get("status") == "ok"
            ]
            if len(ok_primary) >= 2 and rel_ratio > 0:
                neffs = {k: float(A_style_sources[k].get("neff", 0.0) or 0.0) for k in ok_primary}
                max_key = max(neffs, key=lambda kk: neffs[kk])
                max_neff = float(neffs[max_key])
                if max_neff > 0:
                    for k in ok_primary:
                        if k == max_key:
                            continue
                        neff_k = float(neffs[k])
                        if neff_k < rel_ratio * max_neff:

                            A_style_sources[k]["status"] = "too_small_or_noisy"
                            A_style_sources[k]["reason"] = "relative_neff_gate"

                            if k in color_transfer_log["pooled_sources_A"]:
                                color_transfer_log["pooled_sources_A"][k]["status"] = "too_small_or_noisy"
                                color_transfer_log["pooled_sources_A"][k]["reason"] = "relative_neff_gate"
                                color_transfer_log["pooled_sources_A"][k]["rel_ratio"] = rel_ratio
                                color_transfer_log["pooled_sources_A"][k]["max_key"] = max_key
                                color_transfer_log["pooled_sources_A"][k]["max_neff"] = max_neff
                            logger.warning(
                                f"A pooled source '{k}' downgraded by relative neff gate: "
                                f"neff={neff_k:.1f} < {rel_ratio:.3f} * {max_neff:.1f} (max={max_key})"
                            )
        except Exception as e:
            logger.warning(f"Relative neff gating failed: {e}")
    elif perform_color_transfer and not use_detailed_labels_config:
        logger.info("Skipping color transfer: `USE_DETAILED_SURFACE_LABELS` is False. Color transfer is only supported with detailed labels.")


    logger.info(f"Processing Model B (ID: {model_B_id}) for body extraction...")
    model_B_iter_path = iter_paths.get('model_B', 'latest')


    id_format_kwargs = {'id': model_B_id, 'iter_path': model_B_iter_path} if direct_swap_mode else {'user_id': model_B_id, 'model_id': user_A_id}
    B_body_gaussians_path = paths_config['reposed_avatar_ply'].format(**id_format_kwargs)
    B_body_gaussians_np, B_body_prop_names_np = load_ply_gaussians(B_body_gaussians_path)
    B_body_gaussians_torch = {name: torch.from_numpy(data).to(device) for name, data in B_body_gaussians_np.items()}

    if direct_swap_mode:
        logger.info(f"Transforming Model B's body Gaussians (reshaped, in B's world space) to B's canonical space...")
    else:
        logger.info(f"Transforming Model B's body Gaussians (currently still in B's world space) to B's canonical space...")

    B_use_pelvis_centering = B_dataset_type_for_transform not in ("talkbody4d", "mvhumannet", "actorshq")


    B_pelvis_joint_j0 = None
    try:
        if B_use_pelvis_centering:
            B_pelvis_joint_j0 = get_pelvis_joint_j0(
                B_smpl_params_path, frame_idx=0, device=device,
                dataset_type=B_dataset_type_for_transform, gender=gender_B_for_pelvis_joint
            )
            logger.info(f"Using Model B pelvis joint j0 for rotation center: {B_pelvis_joint_j0.cpu().numpy()}")
        else:
            logger.info(f"Model B dataset_type={B_dataset_type_for_transform}: using origin-centered rigid transform (no pelvis centering).")
    except Exception as e:
        logger.warning(f"Could not extract Model B pelvis joint: {e}")
        logger.warning("Falling back to world origin rotation for Model B")


    B_body_gaussians_in_canon_space = transform_gaussians(
        B_body_gaussians_torch, B_body_prop_names_np, B_transl, B_scale, B_global_orient_matrix,
        to_canonical=True, device=device, pelvis_joint_j0=B_pelvis_joint_j0
    )


    if 'segmentation_pkl_map' in paths_config and model_B_id in paths_config['segmentation_pkl_map']:
        B_segmentation_path = paths_config['segmentation_pkl_map'][model_B_id]
    else:
        B_segmentation_path = paths_config['segmentation_pkl'].format(id=model_B_id)

    if 'embedding_json_map' in paths_config and model_B_id in paths_config['embedding_json_map']:
        B_embedding_path = paths_config['embedding_json_map'][model_B_id].format(iter_path=model_B_iter_path, splat_subdir=B_splat_subdir)
    else:
        B_embedding_path = paths_config['embedding_json'].format(id=model_B_id, iter_path=model_B_iter_path, splat_subdir=B_splat_subdir)

    if 'scan_mesh_obj_map' in paths_config and model_B_id in paths_config['scan_mesh_obj_map']:
        B_scan_mesh_path = paths_config['scan_mesh_obj_map'][model_B_id]
    else:
        B_scan_mesh_path = paths_config['scan_mesh_obj'].format(id=model_B_id)
    B_seg_labels = load_segmentation_labels(B_segmentation_path)
    B_sample_fidxs_orig = load_embedding_json(B_embedding_path)
    B_scan_mesh_orig = load_mesh(B_scan_mesh_path)


    B_neck_tube_mask = None
    B_above_neck_plane_mask = None
    B_joints_world = None
    try:
        B_joints_world = get_body_joints_world(
            B_smpl_params_path,
            frame_idx=0,
            device=device,
            dataset_type=B_dataset_type_for_transform,
            gender=gender_B_for_pelvis_joint,
        )
        B_neck_masks = compute_neck_axis_and_masks(
            np.asarray(B_scan_mesh_orig.vertices, dtype=np.float32),
            B_joints_world.detach().cpu().numpy(),
            params=NeckCutParams(
                radius_ratio=0.35,
                plane_lift_ratio=float(getattr(args, "neck_plane_lift_ratio", 0.2)),
            ),
        )
        B_neck_tube_mask = B_neck_masks.get("neck_tube_mask", None)
        B_above_neck_plane_mask = B_neck_masks.get("above_neck_plane_mask", None)
        logger.info(
            f"B neck_tube_mask: {int(np.sum(B_neck_tube_mask)) if B_neck_tube_mask is not None else 0} verts; "
            f"above_plane: {int(np.sum(B_above_neck_plane_mask)) if B_above_neck_plane_mask is not None else 0} verts"
        )
        if debug_color_transfer_save_mesh and B_neck_tube_mask is not None:
            os.makedirs("output/debug", exist_ok=True)
            try:
                pts = np.asarray(B_scan_mesh_orig.vertices, dtype=np.float32)[B_neck_tube_mask]
                trimesh.PointCloud(pts).export(f"output/debug/{model_B_id}_neck_tube_verts.ply")
            except Exception as e_vis:
                logger.warning(f"Failed to save B neck tube point cloud: {e_vis}")
    except Exception as e:
        logger.warning(f"Failed to compute B neck cut masks: {e}")


    B_smpl_mesh = None
    if not use_direct_head_label_B:
        B_smpl_mesh_path_for_head_id = paths_config['smpl_mesh_obj'].format(user_id=model_B_id, model_id=user_A_id)
        B_smpl_mesh = load_mesh(B_smpl_mesh_path_for_head_id)

    B_orig_head_verts_indices, B_orig_head_face_indices = identify_head_region(
        B_scan_mesh_orig.vertices, B_scan_mesh_orig.faces, B_seg_labels, current_surface_labels,
        hair_label_name=hair_label_name_B,
        use_direct_head_label=use_direct_head_label_B,
        direct_head_label_name=direct_head_label_name_B,
        smpl_mesh_vertices=B_smpl_mesh.vertices if B_smpl_mesh else None,
        smplx_vert_seg=smplx_vert_seg,
        fallback_skin_label_name=fallback_skin_label_name_B,
        neck_tube_mask=B_neck_tube_mask,
        above_neck_plane_mask=B_above_neck_plane_mask,
    )
    if not B_orig_head_verts_indices.size:
        logger.error(f"Could not identify original head vertices for Model B ({model_B_id}). Cannot determine body Gaussians. Aborting.")
        return

    if B_body_gaussians_in_canon_space['x'].shape[0] != B_sample_fidxs_orig.shape[0]:
        logger.error(f"Mismatch in Gaussian count for B body ({B_body_gaussians_in_canon_space['x'].shape[0]}) vs B original embedding ({B_sample_fidxs_orig.shape[0]}).")
        return

    is_B_orig_head_gaussian = np.isin(B_sample_fidxs_orig, B_orig_head_face_indices)
    is_B_body_gaussian_mask = ~is_B_orig_head_gaussian


    B_hands_face_indices = np.array([], dtype=np.int64)
    is_B_hands_gaussian_mask = np.zeros((B_sample_fidxs_orig.shape[0],), dtype=bool)
    is_B_body_no_hands_gaussian_mask = np.asarray(is_B_body_gaussian_mask).astype(bool)
    if swap_hands_enabled:
        try:
            _, B_hands_face_indices = get_labeled_part_verts_and_faces(
                B_seg_labels, B_scan_mesh_orig.faces, current_surface_labels, "hands"
            )
            B_hands_face_indices = np.asarray(B_hands_face_indices, dtype=np.int64).reshape(-1)
            is_B_hands_gaussian_mask = np.isin(B_sample_fidxs_orig, B_hands_face_indices)
            is_B_body_no_hands_gaussian_mask = np.asarray(is_B_body_gaussian_mask).astype(bool) & (~is_B_hands_gaussian_mask)
            logger.info(
                f"swap_hands enabled: removing {int(is_B_hands_gaussian_mask.sum())} B hand gaussians from composed body."
            )
        except Exception as e_handsB:
            logger.warning(f"Failed to extract B hands gaussians; proceeding without removing hands. err={e_handsB}")

            is_B_body_no_hands_gaussian_mask = np.asarray(is_B_body_gaussian_mask).astype(bool)

    is_B_body_for_comp_mask = is_B_body_no_hands_gaussian_mask if swap_hands_enabled else np.asarray(is_B_body_gaussian_mask).astype(bool)


    B_body_final_in_A_canon_space = {name: data[is_B_body_for_comp_mask] for name, data in B_body_gaussians_in_canon_space.items()}
    num_B_body_gaussians = B_body_final_in_A_canon_space['x'].shape[0]
    logger.info(f"Extracted {num_B_body_gaussians} body Gaussians from Model B (in A's canonical space).")

    if perform_color_transfer and use_detailed_labels_config:
        logger.info("--- Preparing and Transferring Model B's specified body part colors ---")
        B_face_map = build_pooled_face_index_maps(
            segmentation_labels=B_seg_labels,
            scan_mesh_faces=B_scan_mesh_orig.faces,
            current_surface_labels=current_surface_labels,
            neck_tube_mask=B_neck_tube_mask,
        )

        if "hands" in current_surface_labels and "hands" not in target_body_parts_for_color_transfer:
            target_body_parts_for_color_transfer = list(target_body_parts_for_color_transfer) + ["hands"]
            logger.info("Detected 'hands' label; added 'hands' to target_body_parts_for_color_transfer.")
        available_A_sources = [k for k, v in A_style_sources.items() if isinstance(v, dict) and v.get("status") == "ok"]

        for part_label_name in target_body_parts_for_color_transfer:
            logger.info(f"Processing Model B, part: {part_label_name} for color transfer...")
            part_log = {
                "source_subject_id": str(user_A_id),
                "target_subject_id": str(model_B_id),
                "target_part": str(part_label_name),
                "fallback_path": fallback_order_for_target(part_label_name),
                "candidates": [],
                "chosen_source_key": None,
                "chosen_source_mean_rgb": None,
                "target_face_count": 0,
                "target_gaussians_total": 0,
                "target_gaussians_selected": 0,
                "target_mean_rgb_before": None,
                "target_mean_rgb_after": None,
                "status": "init",
            }

            if part_label_name in ("torso_skin", "torso_non_neck"):
                B_part_face_indices = B_face_map.get("torso_non_neck", np.array([], dtype=int))
                B_part_verts_indices = np.array([], dtype=int)
            else:
                B_part_verts_indices, B_part_face_indices = get_labeled_part_verts_and_faces(
                    B_seg_labels, B_scan_mesh_orig.faces, current_surface_labels, part_label_name
                )
            B_part_gaussians_canon = {}
            if B_part_face_indices.size > 0:
                is_B_part_gaussian_mask = np.isin(B_sample_fidxs_orig, B_part_face_indices)
                B_part_gaussians_canon = {name: data[is_B_part_gaussian_mask] for name, data in B_body_gaussians_in_canon_space.items()}
                part_log["target_face_count"] = int(B_part_face_indices.size)
            else:
                logger.warning(f"No faces found for Model B, part '{part_label_name}'. Skipping this part for B.")
                part_log["status"] = "skip_target_missing_faces"
                color_transfer_log["parts"].append(part_log)
                continue

            B_part_colors_for_transfer = None
            B_part_high_opacity_mask = None
            if B_part_gaussians_canon and B_part_gaussians_canon.get('x', torch.empty(0)).shape[0] > 0:
                part_log["target_gaussians_total"] = int(B_part_gaussians_canon["x"].shape[0])
                debug_prefix_B_part = f"output/debug/{model_B_id}_{part_label_name}" if debug_color_transfer_save_mesh else None
                if color_transfer_use_opacity_weighting:

                    target_part_rgbs = torch.clamp(SH2RGB(torch.stack([
                        B_part_gaussians_canon['f_dc_0'], B_part_gaussians_canon['f_dc_1'], B_part_gaussians_canon['f_dc_2']
                    ], dim=-1)), 0.0, 1.0)

                    B_part_colors_for_transfer = target_part_rgbs.permute(1,0).unsqueeze(-1)

                    B_part_update_mask = torch.ones(B_part_gaussians_canon['opacity'].shape[0], dtype=torch.bool, device=device)
                    part_log["target_gaussians_selected"] = int(B_part_gaussians_canon["opacity"].shape[0])
                    w = torch.sigmoid(B_part_gaussians_canon["opacity"]).float()
                    mean_rgb_before = (target_part_rgbs * w.view(-1, 1)).sum(dim=0) / torch.clamp(w.sum(), min=1e-6)
                    part_log["target_mean_rgb_before"] = mean_rgb_before.detach().cpu().tolist()
                else:

                    B_part_colors_for_transfer, B_part_high_opacity_mask, _ = prepare_skin_gaussian_colors_for_transfer(
                        B_part_gaussians_canon,
                opacity_threshold=color_transfer_opacity_threshold,
                        device=device,
                        debug_save_path_prefix=debug_prefix_B_part,
                        original_property_names=B_body_prop_names_np,
                        use_all_gaussians_for_stats=color_transfer_use_all_gaussians_for_stats
                    )
                    B_part_update_mask = B_part_high_opacity_mask
                    if B_part_high_opacity_mask is not None:
                        part_log["target_gaussians_selected"] = int(B_part_high_opacity_mask.sum().item())
                    if B_part_colors_for_transfer is not None and B_part_colors_for_transfer.shape[1] > 0:
                        rgb_before = B_part_colors_for_transfer.squeeze(-1).permute(1, 0)
                        part_log["target_mean_rgb_before"] = rgb_before.mean(dim=0).detach().cpu().tolist()

                if B_part_colors_for_transfer is not None:
                    logger.info(f"Model B, part '{part_label_name}' colors prepared for receiving style, shape: {B_part_colors_for_transfer.shape}")
                else:
                    logger.warning(f"Could not prepare Model B, part '{part_label_name}' colors for receiving style.")


                chosen_src_key = None
                for cand_key in fallback_order_for_target(part_label_name):
                    cand = A_style_sources.get(cand_key, {"status": "missing"})
                    cand_status = cand.get("status", "missing") if isinstance(cand, dict) else "missing"
                    part_log["candidates"].append(
                        {
                            "source_key": cand_key,
                            "status": cand_status,
                            "neff": cand.get("neff") if isinstance(cand, dict) else None,
                            "n_selected": cand.get("n_selected") if isinstance(cand, dict) else None,
                            "n_inlier": cand.get("n_inlier") if isinstance(cand, dict) else None,
                            "mean_rgb": cand.get("mean_rgb") if isinstance(cand, dict) else None,
                        }
                    )
                    if cand_status == "ok" and chosen_src_key is None:
                        chosen_src_key = cand_key

                A_style_source = A_style_sources.get(chosen_src_key) if chosen_src_key is not None else None
                part_log["chosen_source_key"] = chosen_src_key
                part_log["chosen_source_mean_rgb"] = (
                    A_style_source.get("mean_rgb") if isinstance(A_style_source, dict) else None
                )
                logger.info(f"DAG source for target '{part_label_name}': {chosen_src_key}")

                if A_style_source is not None and B_part_colors_for_transfer is not None:
                    logger.info(f"Attempting color transfer for part '{part_label_name}': B's part to match A's part style...")

                    debug_stats_to_use = None
                    source_lab_stats_to_use = A_style_source.get('lab_stats') if isinstance(A_style_source, dict) else None
                    source_rgb_to_use = A_style_source.get('prepared_colors') if isinstance(A_style_source, dict) else None

                    if debug_color_transfer_force_color:
                        debug_stats_to_use = {'mean_lab': list(debug_hardcoded_lab_mean), 'std_lab': list(debug_hardcoded_lab_std)}
                        logger.info(f"DEBUG MODE: Using hardcoded source color stats for part '{part_label_name}': {debug_stats_to_use}")

                    transferred_B_part_rgb_prepared = transfer_skin_color_pytorch(
                        source_rgb_prepared=source_rgb_to_use,
                        target_rgb_prepared=B_part_colors_for_transfer,
                        debug_source_stats=debug_stats_to_use,
                        source_lab_stats=source_lab_stats_to_use
                    )

                    if transferred_B_part_rgb_prepared is not None:
                        transferred_B_part_rgb_flat = transferred_B_part_rgb_prepared.squeeze(-1).permute(1, 0)
                        transferred_B_part_sh0 = RGB2SH(transferred_B_part_rgb_flat)
                        part_log["target_mean_rgb_after"] = transferred_B_part_rgb_flat.mean(dim=0).detach().cpu().tolist()

                        part_indices_in_full_b_body_canon = torch.where(torch.from_numpy(is_B_part_gaussian_mask).to(device))[0]

                        if B_part_update_mask is not None and B_part_update_mask.any():
                            indices_to_update_within_part_canon = torch.where(B_part_update_mask)[0]
                            indices_to_update_in_b_body_canon = part_indices_in_full_b_body_canon[indices_to_update_within_part_canon]


                            if len(indices_to_update_in_b_body_canon) == transferred_B_part_sh0.shape[0]:
                                B_body_gaussians_in_canon_space['f_dc_0'][indices_to_update_in_b_body_canon] = transferred_B_part_sh0[:, 0]
                                B_body_gaussians_in_canon_space['f_dc_1'][indices_to_update_in_b_body_canon] = transferred_B_part_sh0[:, 1]
                                B_body_gaussians_in_canon_space['f_dc_2'][indices_to_update_in_b_body_canon] = transferred_B_part_sh0[:, 2]
                                logger.info(f"Updated SH0 colors for {transferred_B_part_sh0.shape[0]} Gaussians of part '{part_label_name}' in B's canonical representation.")
                                part_log["status"] = "ok_transferred"
                            else:
                                logger.error(f"Mismatch between number of colors to apply ({transferred_B_part_sh0.shape[0]}) and indices to update ({len(indices_to_update_in_b_body_canon)}). Aborting color update for this part.")
                                part_log["status"] = "fail_update_count_mismatch"
                        else:
                            logger.warning(f"No Gaussians in B for part '{part_label_name}' to update.")
                            part_log["status"] = "skip_target_no_update_gaussians"
                    else:
                        logger.warning(f"Color transfer for part '{part_label_name}' did not produce output. B's part colors remain unchanged.")
                        part_log["status"] = "fail_transfer_returned_none"
                else:
                    logger.warning(f"Skipping color transfer for part '{part_label_name}' as prepared colors/stats for A or B for this part are missing/empty.")
                    part_log["status"] = "skip_missing_source_or_target"
            else:
                part_log["status"] = "skip_target_no_gaussians"

            color_transfer_log["parts"].append(part_log)


        if getattr(args, "post_align_hands_to_arms", False) and "hands" in current_surface_labels:
            try:
                min_g = int(getattr(args, "post_align_hands_to_arms_min_gaussians", 2000))
                min_delta = float(getattr(args, "post_align_hands_to_arms_min_delta", 0.02))
                q = float(getattr(args, "post_align_hands_to_arms_opacity_quantile", 0.7))
                min_dom = int(getattr(args, "post_align_hands_to_arms_min_dominant_gaussians", 500))
                use_k2 = bool(getattr(args, "post_align_hands_to_arms_k2_lab", False))
                k2_iters = int(getattr(args, "post_align_hands_to_arms_k2_iters", 10))


                hands_faces = B_face_map.get("hands", np.array([], dtype=int))
                left_arm_faces = B_face_map.get("left_arm", np.array([], dtype=int))
                right_arm_faces = B_face_map.get("right_arm", np.array([], dtype=int))
                arms_faces = np.unique(np.concatenate([left_arm_faces, right_arm_faces], axis=0)) if (left_arm_faces.size or right_arm_faces.size) else np.array([], dtype=int)

                if hands_faces.size == 0 or arms_faces.size == 0:
                    logger.info("Post-align hands→arms skipped: missing hands or arms faces on target.")
                else:
                    is_hands = np.isin(B_sample_fidxs_orig, hands_faces)
                    is_arms = np.isin(B_sample_fidxs_orig, arms_faces)
                    n_h = int(is_hands.sum())
                    n_a = int(is_arms.sum())

                    if n_h < min_g or n_a < min_g:
                        logger.info(f"Post-align hands→arms skipped: insufficient gaussians (hands={n_h}, arms={n_a}, min={min_g}).")
                    else:

                        fdc = torch.stack(
                            [
                                B_body_gaussians_in_canon_space["f_dc_0"],
                                B_body_gaussians_in_canon_space["f_dc_1"],
                                B_body_gaussians_in_canon_space["f_dc_2"],
                            ],
                            dim=-1,
                        )
                        rgb = torch.clamp(SH2RGB(fdc), 0.0, 1.0)
                        w = torch.sigmoid(B_body_gaussians_in_canon_space["opacity"]).float()

                        rgb_h = rgb[torch.from_numpy(is_hands).to(device)]
                        w_h = w[torch.from_numpy(is_hands).to(device)]
                        rgb_a = rgb[torch.from_numpy(is_arms).to(device)]
                        w_a = w[torch.from_numpy(is_arms).to(device)]


                        q = float(min(max(q, 0.0), 1.0))
                        thr_h = torch.quantile(w_h, q)
                        thr_a = torch.quantile(w_a, q)
                        dom_h = w_h >= thr_h
                        dom_a = w_a >= thr_a
                        n_dom_h = int(dom_h.sum().item())
                        n_dom_a = int(dom_a.sum().item())
                        if n_dom_h < min_dom or n_dom_a < min_dom:
                            logger.info(
                                f"Post-align hands→arms skipped: insufficient dominant gaussians "
                                f"(hands_dom={n_dom_h}, arms_dom={n_dom_a}, min_dom={min_dom})."
                            )
                            dom_h = torch.ones_like(dom_h, dtype=torch.bool)
                            dom_a = torch.ones_like(dom_a, dtype=torch.bool)
                            n_dom_h = int(dom_h.sum().item())
                            n_dom_a = int(dom_a.sum().item())


                        lab_a = rgb_to_lab(rgb_a[dom_a].permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                        w_dom_a = w_a[dom_a]
                        mu_a_lab = (lab_a * w_dom_a[:, None]).sum(dim=0) / torch.clamp(w_dom_a.sum(), min=1e-6)


                        rgb_hd = rgb_h[dom_h]
                        w_hd = w_h[dom_h]
                        lab_h = rgb_to_lab(rgb_hd.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)

                        if use_k2 and lab_h.shape[0] >= 2 * max(50, min_dom // 10):

                            k2_iters = max(1, min(int(k2_iters), 50))

                            l_vals = lab_h[:, 0]
                            idx_lo = int(torch.argmin(l_vals).item())
                            idx_hi = int(torch.argmax(l_vals).item())
                            c0 = lab_h[idx_lo].clone()
                            c1 = lab_h[idx_hi].clone()
                            for _ in range(k2_iters):
                                d0 = torch.sum((lab_h - c0[None, :]) ** 2, dim=1)
                                d1 = torch.sum((lab_h - c1[None, :]) ** 2, dim=1)
                                a0 = d0 <= d1
                                a1 = ~a0
                                if a0.any():
                                    w0 = w_hd[a0]
                                    c0 = (lab_h[a0] * w0[:, None]).sum(dim=0) / torch.clamp(w0.sum(), min=1e-6)
                                if a1.any():
                                    w1 = w_hd[a1]
                                    c1 = (lab_h[a1] * w1[:, None]).sum(dim=0) / torch.clamp(w1.sum(), min=1e-6)


                            d0 = torch.sum((lab_h - c0[None, :]) ** 2, dim=1)
                            d1 = torch.sum((lab_h - c1[None, :]) ** 2, dim=1)
                            a0 = d0 <= d1
                            a1 = ~a0
                            mu0 = (lab_h[a0] * w_hd[a0][:, None]).sum(dim=0) / torch.clamp(w_hd[a0].sum(), min=1e-6) if a0.any() else c0
                            mu1 = (lab_h[a1] * w_hd[a1][:, None]).sum(dim=0) / torch.clamp(w_hd[a1].sum(), min=1e-6) if a1.any() else c1
                            delta0 = (mu_a_lab - mu0)
                            delta1 = (mu_a_lab - mu1)

                            lab_h_all = rgb_to_lab(rgb_h.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
                            d0_all = torch.sum((lab_h_all - c0[None, :]) ** 2, dim=1)
                            d1_all = torch.sum((lab_h_all - c1[None, :]) ** 2, dim=1)
                            a0_all = d0_all <= d1_all
                            a1_all = ~a0_all
                            lab_h_all_new = lab_h_all.clone()
                            if a0_all.any():
                                lab_h_all_new[a0_all] = lab_h_all[a0_all] + delta0[None, :]
                            if a1_all.any():
                                lab_h_all_new[a1_all] = lab_h_all[a1_all] + delta1[None, :]
                            rgb_h_all_new = lab_to_rgb(lab_h_all_new.permute(1, 0).unsqueeze(-1), clip=True).squeeze(-1).permute(1, 0)
                            rgb_h_new = rgb_h_all_new
                            sh0_new = RGB2SH(rgb_h_new)


                            mu_h_rgb_before = (rgb_hd * w_hd[:, None]).sum(dim=0) / torch.clamp(w_hd.sum(), min=1e-6)
                            mu_h_rgb_after = (rgb_h_new[dom_h] * w_hd[:, None]).sum(dim=0) / torch.clamp(w_hd.sum(), min=1e-6)
                            mu_h_rgb_after_all = (rgb_h_new * w_h[:, None]).sum(dim=0) / torch.clamp(w_h.sum(), min=1e-6)
                            delta_norm = float(torch.linalg.norm(mu_h_rgb_after - mu_h_rgb_before).item())
                            mode_name = "k2_lab"
                            extra_log = {
                                "k2_iters": k2_iters,
                                "cluster0_count": int(a0_all.sum().item()),
                                "cluster1_count": int(a1_all.sum().item()),
                                "mu_arms_lab": mu_a_lab.detach().cpu().tolist(),
                                "mu_hands_rgb_before": mu_h_rgb_before.detach().cpu().tolist(),
                                "mu_hands_rgb_after": mu_h_rgb_after.detach().cpu().tolist(),
                                "mu_hands_rgb_after_all": mu_h_rgb_after_all.detach().cpu().tolist(),
                                "delta0_lab": delta0.detach().cpu().tolist(),
                                "delta1_lab": delta1.detach().cpu().tolist(),
                            }
                        else:

                            mu_h = (rgb_h[dom_h] * w_h[dom_h][:, None]).sum(dim=0) / torch.clamp(w_h[dom_h].sum(), min=1e-6)
                            mu_a = (rgb_a[dom_a] * w_a[dom_a][:, None]).sum(dim=0) / torch.clamp(w_a[dom_a].sum(), min=1e-6)
                            delta = mu_a - mu_h
                            delta_norm = float(torch.linalg.norm(delta).item())
                            if delta_norm < min_delta:
                                logger.info(f"Post-align hands→arms skipped: delta_norm={delta_norm:.4f} < {min_delta:.4f}")

                                sh0_new = None
                            else:
                                rgb_h_new = rgb_h.clone()
                                rgb_h_new[dom_h] = torch.clamp(rgb_h[dom_h] + delta[None, :], 0.0, 1.0)
                                sh0_new = RGB2SH(rgb_h_new)
                                mode_name = "rgb_mean_shift"
                                extra_log = {"delta_rgb": delta.detach().cpu().tolist(), "delta_norm": delta_norm}


                        if sh0_new is None:
                            pass
                        else:
                            idx_h = torch.where(torch.from_numpy(is_hands).to(device))[0]
                            B_body_gaussians_in_canon_space["f_dc_0"][idx_h] = sh0_new[:, 0]
                            B_body_gaussians_in_canon_space["f_dc_1"][idx_h] = sh0_new[:, 1]
                            B_body_gaussians_in_canon_space["f_dc_2"][idx_h] = sh0_new[:, 2]

                            color_transfer_log.setdefault("post_align", {})
                            color_transfer_log["post_align"]["hands_to_arms"] = {
                                "enabled": True,
                                "mode": mode_name,
                                "min_gaussians": min_g,
                                "min_delta": min_delta,
                                "opacity_quantile": q,
                                "min_dominant_gaussians": min_dom,
                                "hands_dominant_gaussians": n_dom_h,
                                "arms_dominant_gaussians": n_dom_a,
                                "hands_gaussians": n_h,
                                "arms_gaussians": n_a,
                                **extra_log,
                            }
                            logger.info(f"Post-align hands→arms applied. mode={mode_name}")
            except Exception as e:
                logger.warning(f"Post-align hands→arms failed: {e}")


        B_body_final_in_A_canon_space = {name: data[is_B_body_for_comp_mask] for name, data in B_body_gaussians_in_canon_space.items()}
        logger.info("Re-extracted B's full body gaussians after processing all parts for color transfer.")
    elif not perform_color_transfer:
        logger.info("Skipping color transfer as per global `perform_color_transfer` configuration.")


    dataset_root_for_offsets = str(Path(A_smpl_params_path).parent.parent)
    cloth_fit_offsets = None
    debug_combined_world_B = None
    debug_A_head_world_B = None
    debug_B_body_world_B = None
    debug_A_head_plus_neck_world_B = None
    debug_A_head_plus_neck_cf = None
    debug_B_body_cf = None
    if getattr(args, "use_cloth_fit_reshaped_gs", False) and getattr(args, "cloth_fit_height_aware", False):
        cloth_fit_offsets = _load_cloth_fit_offsets(
            dataset_root=dataset_root_for_offsets,
            avatar_subject_id=str(user_A_id),
            garment_subject_id=str(model_B_id),
            device=device,
            cloth_fit_suffix=str(getattr(args, "cloth_fit_suffix", "") or ""),
        )

    if cloth_fit_offsets is not None:
        source_offset, target_offset, source_scale, target_scale = cloth_fit_offsets
        logger.info("Cloth-fit offsets found: composing in cloth-fit normalized space (no scale).")


        use_scale_aware_cf = (
            (source_scale is not None)
            and (target_scale is not None)
            and (A_dataset_type_for_transform == "thuman2" or B_dataset_type_for_transform == "thuman2")
        )
        if use_scale_aware_cf:
            logger.info(
                f"Scale-aware cloth-fit mode enabled: source_scale={float(source_scale.item()):.6f}, "
                f"target_scale={float(target_scale.item()):.6f}"
            )


        A_head_world_B = transform_gaussians(
            A_head_final_canon, A_prop_names_np,
            B_transl, B_scale, B_global_orient_matrix,
            to_canonical=False, device=device, pelvis_joint_j0=None
        )
        A_hands_world_B = None
        if swap_hands_enabled and num_A_hands_gaussians > 0:
            A_hands_world_B = transform_gaussians(
                A_hands_final_canon, A_prop_names_np,
                B_transl, B_scale, B_global_orient_matrix,
                to_canonical=False, device=device, pelvis_joint_j0=None
            )
        if save_head_body_separation_debug and num_A_head_plus_neck_gaussians > 0:
            debug_A_head_plus_neck_world_B = transform_gaussians(
                A_head_plus_neck_final_canon,
                A_prop_names_np,
                B_transl,
                B_scale,
                B_global_orient_matrix,
                to_canonical=False,
                device=device,
                pelvis_joint_j0=None,
            )
        B_body_world_B = transform_gaussians(
            B_body_final_in_A_canon_space, A_prop_names_np,
            B_transl, B_scale, B_global_orient_matrix,
            to_canonical=False, device=device, pelvis_joint_j0=None
        )
        debug_A_head_world_B = A_head_world_B
        debug_B_body_world_B = B_body_world_B


        if use_scale_aware_cf:


            A_head_cf = _normalize_gaussians_cf(A_head_world_B, offset_xyz=target_offset, scale=target_scale)
            A_hands_cf = _normalize_gaussians_cf(A_hands_world_B, offset_xyz=target_offset, scale=target_scale) if A_hands_world_B is not None else None
            B_body_cf = _normalize_gaussians_cf(B_body_world_B, offset_xyz=source_offset, scale=source_scale)
            if save_head_body_separation_debug and debug_A_head_plus_neck_world_B is not None:
                debug_A_head_plus_neck_cf = _normalize_gaussians_cf(
                    debug_A_head_plus_neck_world_B, offset_xyz=target_offset, scale=target_scale
                )
            debug_B_body_cf = B_body_cf
        else:


            A_head_cf = _translate_gaussians_xyz(A_head_world_B, target_offset, sign=-1.0)
            A_hands_cf = _translate_gaussians_xyz(A_hands_world_B, target_offset, sign=-1.0) if A_hands_world_B is not None else None
            B_body_cf = _translate_gaussians_xyz(B_body_world_B, source_offset, sign=-1.0)
            if save_head_body_separation_debug and debug_A_head_plus_neck_world_B is not None:
                debug_A_head_plus_neck_cf = _translate_gaussians_xyz(debug_A_head_plus_neck_world_B, target_offset, sign=-1.0)
            debug_B_body_cf = B_body_cf


        combined_gaussians_cf = {}
        for name in A_prop_names_np:
            head_data = A_head_cf.get(name) if A_head_cf is not None else None
            hands_data = (A_hands_cf.get(name) if (A_hands_cf is not None) else None)
            body_data = B_body_cf.get(name) if B_body_cf is not None else None

            def _empty_like(ref: torch.Tensor) -> torch.Tensor:
                return torch.empty((0,) + tuple(ref.shape[1:]), dtype=ref.dtype, device=device)


            if head_data is None:
                if body_data is not None:
                    head_data = _empty_like(body_data)
                else:

                    head_data = torch.empty((0,), dtype=torch.float32, device=device)
            template = head_data if isinstance(head_data, torch.Tensor) else None
            if template is not None and template.ndim >= 1:
                if hands_data is None:
                    hands_data = _empty_like(template)
                if body_data is None:
                    body_data = _empty_like(template)
            else:
                if hands_data is None:
                    hands_data = torch.empty((0,), dtype=torch.float32, device=device)
                if body_data is None:
                    body_data = torch.empty((0,), dtype=torch.float32, device=device)


            if head_data.numel() > 0:
                target_dtype = head_data.dtype
            elif hands_data.numel() > 0:
                target_dtype = hands_data.dtype
            else:
                target_dtype = body_data.dtype if body_data.numel() > 0 else torch.float32
            try:
                if body_data.numel() > 0 and body_data.dtype != target_dtype:
                    body_data = body_data.to(target_dtype)
                if hands_data.numel() > 0 and hands_data.dtype != target_dtype:
                    hands_data = hands_data.to(target_dtype)
                if head_data.numel() > 0 and head_data.dtype != target_dtype:
                    head_data = head_data.to(target_dtype)
            except Exception as e_cast:
                logger.error(f"Error casting data for '{name}' (cloth-fit space): {e_cast}. Proceeding with best-effort concatenation.")

            combined_gaussians_cf[name] = torch.cat((head_data, hands_data, body_data), dim=0)

        if not combined_gaussians_cf or 'x' not in combined_gaussians_cf or combined_gaussians_cf['x'].shape[0] == 0:
            logger.error("No Gaussians to save after combining in cloth-fit space. Output will be empty.")
            return


        if use_scale_aware_cf:
            combined_world_B = _denormalize_gaussians_cf(combined_gaussians_cf, offset_xyz=target_offset, scale=target_scale)
        else:
            combined_world_B = _translate_gaussians_xyz(combined_gaussians_cf, target_offset, sign=+1.0)
        debug_combined_world_B = combined_world_B
        combined_canon = transform_gaussians(
            combined_world_B, A_prop_names_np,
            B_transl, B_scale, B_global_orient_matrix,
            to_canonical=True, device=device, pelvis_joint_j0=None
        )

        logger.info("Transforming combined Gaussians from canonical space back to A's world space...")
        final_swapped_gaussians_world_A = transform_gaussians(
            combined_canon, A_prop_names_np,
            A_transl, A_scale, A_global_orient_matrix,
            to_canonical=False, device=device, pelvis_joint_j0=A_pelvis_joint_j0
        )
    else:
        logger.info("Combining User A's head (+ optional hands) with Model B's body (in canonical space)...")
        combined_gaussians_canon = {}

        for name in A_prop_names_np:
            head_data = A_head_final_canon.get(name)
            hands_data = A_hands_final_canon.get(name) if swap_hands_enabled else None
            body_data = B_body_final_in_A_canon_space.get(name)

            if head_data is None:
                head_data = (
                    torch.empty((0,) + B_body_final_in_A_canon_space[name].shape[1:], dtype=B_body_final_in_A_canon_space[name].dtype, device=device)
                    if num_A_head_gaussians == 0 and B_body_final_in_A_canon_space.get(name) is not None
                    else torch.empty(0, dtype=torch.float32, device=device)
                )
            if hands_data is None:
                hands_data = (
                    torch.empty((0,) + head_data.shape[1:], dtype=head_data.dtype, device=device)
                )
            if body_data is None:
                body_data = (
                    torch.empty((0,) + head_data.shape[1:], dtype=head_data.dtype, device=device)
                    if num_B_body_gaussians == 0 and head_data is not None
                    else torch.empty(0, dtype=torch.float32, device=device)
                )

            if head_data.numel() > 0 and body_data.numel() > 0 and head_data.dtype != body_data.dtype:
                logger.warning(f"Dtype mismatch for '{name}'. head: {head_data.dtype}, body: {body_data.dtype}. Casting body to head dtype.")
                try:
                    body_data = body_data.to(head_data.dtype)
                except Exception as e_cast:
                    logger.error(f"Error casting body_data for '{name}': {e_cast}. Skipping body part.")
                    body_data = torch.empty((0,) + head_data.shape[1:], dtype=head_data.dtype, device=device)
            if head_data.numel() > 0 and hands_data.numel() > 0 and head_data.dtype != hands_data.dtype:
                try:
                    hands_data = hands_data.to(head_data.dtype)
                except Exception:
                    pass

            combined_gaussians_canon[name] = torch.cat((head_data, hands_data, body_data), dim=0)


    if debug_color_transfer_save_mesh:
        os.makedirs("output/debug", exist_ok=True)
        a_gaussian = {}
        b_gaussian = {}
        for name in A_prop_names_np:
            A_data = A_head_final_canon.get(name)
            B_data = B_body_final_in_A_canon_space.get(name)
            a_gaussian[name] = A_data.cpu().numpy() if A_data is not None else np.zeros((0,), dtype=np.float32)
            b_gaussian[name] = B_data.cpu().numpy() if B_data is not None else np.zeros((0,), dtype=np.float32)
        save_ply_gaussians(f"output/debug/{user_A_id}_head_gaussians_canon.ply", a_gaussian, A_prop_names_np)
        save_ply_gaussians(f"output/debug/{model_B_id}_body_gaussians_canon.ply", b_gaussian, A_prop_names_np)

    if cloth_fit_offsets is None:
        if not combined_gaussians_canon or 'x' not in combined_gaussians_canon or combined_gaussians_canon['x'].shape[0] == 0:
            logger.error("No Gaussians to save after combining in canonical space. Output will be empty.")
            return


    if cloth_fit_offsets is None:

        logger.info("Transforming combined Gaussians from canonical space back to A's world space...")
        final_swapped_gaussians_world_A = transform_gaussians(
            combined_gaussians_canon, A_prop_names_np, A_transl, A_scale, A_global_orient_matrix,
            to_canonical=False, device=device, pelvis_joint_j0=A_pelvis_joint_j0
        )


    final_swapped_gaussians_np = {name: data.cpu().numpy() for name, data in final_swapped_gaussians_world_A.items()}


    output_dir = paths_config['output_dir_swapped'].format(user_id=user_A_id, model_id=model_B_id)
    if repose_load_dir_subfix:
        output_dir += f"_{repose_load_dir_subfix}"
    if args.use_cloth_fit_reshaped_gs:
        output_dir += f"_reshaped_cloth_fit{args.cloth_fit_suffix}"
        logger.info(f"Cloth-fit reshaped enabled: Using cloth-fit reshaped PLY: {output_dir}")
    if bool(swap_hands_enabled):
        output_dir += "_hands_swapped"
    os.makedirs(output_dir, exist_ok=True)


    swap_hands_source = str(getattr(args, "swap_hands_source", "neus_gs")).strip().lower()
    if swap_hands_source not in ("neus_gs", "smplx_mesh"):
        swap_hands_source = "neus_gs"

    if save_head_body_separation_debug:
        try:

            if swap_hands_enabled and swap_hands_source == "smplx_mesh":
                emit_smplx_mesh_hands_proxy_artifacts(
                    output_dir=str(output_dir),
                    device=str(device),
                    smplx_hands_proxy=smplx_hands_proxy,
                    A_smpl_params_path=str(A_smpl_params_path),
                    A_transl=A_transl,
                    A_scale=A_scale,
                    A_global_orient_matrix=A_global_orient_matrix,
                    A_pelvis_joint_j0=A_pelvis_joint_j0,
                    B_transl=B_transl,
                    B_scale=B_scale,
                    B_global_orient_matrix=B_global_orient_matrix,
                )


            pass_neus_hands_split = bool(swap_hands_enabled and (swap_hands_source == "neus_gs"))
            save_head_body_separation_debug_artifacts(
                output_dir=output_dir,
                device=device,
                user_A_id=str(user_A_id),
                model_B_id=str(model_B_id),
                A_orig_gaussians_path=str(A_orig_gaussians_path),
                B_body_gaussians_path=str(B_body_gaussians_path),
                A_scan_mesh_path=str(A_scan_mesh_path),
                B_scan_mesh_path=str(B_scan_mesh_path),
                A_smpl_params_path=str(A_smpl_params_path),
                B_smpl_params_path=str(B_smpl_params_path),
                A_transl=A_transl,
                A_scale=A_scale,
                A_global_orient_matrix=A_global_orient_matrix,
                A_pelvis_joint_j0=A_pelvis_joint_j0,
                B_transl=B_transl,
                B_scale=B_scale,
                B_global_orient_matrix=B_global_orient_matrix,
                B_pelvis_joint_j0=B_pelvis_joint_j0,
                cloth_fit_offsets=cloth_fit_offsets,
                A_prop_names_np=tuple(A_prop_names_np),
                num_A_head_plus_neck_gaussians=int(num_A_head_plus_neck_gaussians),
                num_B_body_gaussians=int(num_B_body_gaussians),
                A_head_plus_neck_final_canon=A_head_plus_neck_final_canon,
                B_body_final_in_A_canon_space=B_body_final_in_A_canon_space,
                debug_A_head_plus_neck_cf=debug_A_head_plus_neck_cf,
                debug_B_body_cf=debug_B_body_cf,
                is_A_head_plus_neck_gaussian_mask=is_A_head_plus_neck_gaussian_mask,
                is_B_body_gaussian_mask=is_B_body_gaussian_mask,
                A_head_plus_neck_face_indices=A_head_plus_neck_face_indices,
                B_orig_head_face_indices=B_orig_head_face_indices,
                A_segmentation_path=str(A_segmentation_path),
                B_segmentation_path=str(B_segmentation_path),
                A_embedding_path=str(A_embedding_path),
                B_embedding_path=str(B_embedding_path),
                B_neck_tube_mask=B_neck_tube_mask,
                A_joints_world=A_joints_world,
                B_joints_world=B_joints_world,
                A_hands_face_indices=A_hands_face_indices if pass_neus_hands_split else None,
                is_A_hands_gaussian_mask=is_A_hands_gaussian_mask if pass_neus_hands_split else None,
                B_hands_face_indices=B_hands_face_indices if swap_hands_enabled else None,
                is_B_body_no_hands_gaussian_mask=is_B_body_no_hands_gaussian_mask if swap_hands_enabled else None,
            )
        except Exception as e_sep:
            logger.warning(f"Failed to save head/body separation debug artifacts: {e_sep}")

    base_A = os.path.splitext(os.path.basename(A_orig_gaussians_path))[0]

    base_B_body = os.path.splitext(os.path.basename(B_body_gaussians_path))[0]
    if args.use_cloth_fit_reshaped_gs:

        base_B_body = 'point_cloud'
    color_transfer_suffix = "_with_color_transfer" if perform_color_transfer else ""
    if color_transfer_use_opacity_weighting:
        color_transfer_mode_suffix = "_weighted"
    elif color_transfer_use_all_gaussians_for_stats:
        color_transfer_mode_suffix = "_all_gs"
    elif color_transfer_opacity_threshold is not None:
        color_transfer_mode_suffix = f"_filter_{color_transfer_opacity_threshold}"
    else:
        color_transfer_mode_suffix = ""
    direct_swap_mode_suffix = "_direct" if direct_swap_mode else ""
    swap_back_mode_suffix = "_swap_back" if swap_back_mode else ""


    rigid_head_suffix = "_rigidhead" if "_rigidhead" in A_orig_gaussians_path else ""

    if swap_back_mode:

        reshape_suffix = "_reshaped" if args.enable_body_reshape else ""
        logger.debug(f"Swap-back mode enabled - current reshape suffix: {reshape_suffix}")
        output_filename = f"restored_{user_A_id}_from{reshape_suffix}_swapped_{user_A_id}head_on_{model_B_id}body{color_transfer_suffix}{color_transfer_mode_suffix}{rigid_head_suffix}{direct_swap_mode_suffix}{swap_back_mode_suffix}.ply"
    else:

        output_filename = f"swapped_{user_A_id}head_on_{model_B_id}body_from_{base_B_body}_in_A_world{color_transfer_suffix}{color_transfer_mode_suffix}{rigid_head_suffix}{direct_swap_mode_suffix}.ply"
    output_ply_path = os.path.join(output_dir, output_filename)
    swapped_filename_without_ext = os.path.splitext(output_filename)[0]


    try:
        swapped_world_B = None
        if debug_combined_world_B is not None:
            swapped_world_B = debug_combined_world_B
        else:

            cano_src = None
            if "combined_gaussians_canon" in locals():
                cano_src = combined_gaussians_canon
            elif "combined_canon" in locals():
                cano_src = combined_canon
            if cano_src is not None:
                swapped_world_B = transform_gaussians(
                    cano_src,
                    A_prop_names_np,
                    B_transl,
                    B_scale,
                    B_global_orient_matrix,
                    to_canonical=False,
                    device=device,
                    pelvis_joint_j0=B_pelvis_joint_j0,
                )
        if swapped_world_B is not None:


            swapped_name_b = swapped_filename_without_ext
            if "_in_A_world" in swapped_name_b:
                swapped_name_b = swapped_name_b.replace("_in_A_world", "_in_B_world")
            out_b = os.path.join(output_dir, f"{swapped_name_b}_B.ply")
            swapped_world_B_np = {name: data.detach().cpu().numpy() for name, data in swapped_world_B.items()}
            save_ply_gaussians(out_b, swapped_world_B_np, A_prop_names_np)
            logger.info(f"Saved swapped gaussians in B world (suffix _B): {out_b}")
        else:
            logger.warning("Could not derive swapped gaussians in B world; skipping _B.ply output.")
    except Exception as e_bsave:
        logger.warning(f"Failed to save swapped gaussians in B world (_B.ply): {e_bsave}")


    if save_swap_debug_artifacts and debug_combined_world_B is not None:
        try:
            out_b = os.path.join(output_dir, f"{swapped_filename_without_ext}_world_B_body_donor.ply")
            debug_world_B_np = {name: data.detach().cpu().numpy() for name, data in debug_combined_world_B.items()}
            save_ply_gaussians(out_b, debug_world_B_np, A_prop_names_np)
            logger.info(f"Saved swapped gaussians in world_B_body_donor space: {out_b}")
        except Exception as e_b:
            logger.warning(f"Failed to save body-donor-space debug artifacts: {e_b}")


    if save_body_donator_full_in_head_world:
        _save_body_donator_full_to_file(
            B_body_gaussians_in_canon_space=B_body_gaussians_in_canon_space,
            B_body_prop_names_np=B_body_prop_names_np,
            A_transl=A_transl,
            A_scale=A_scale,
            A_global_orient_matrix=A_global_orient_matrix,
            A_pelvis_joint_j0=A_pelvis_joint_j0,
            device=device,
            output_dir=output_dir,
            swapped_filename_without_ext=swapped_filename_without_ext
        )

    save_ply_gaussians(output_ply_path, final_swapped_gaussians_np, A_prop_names_np)
    logger.info(f"Successfully swapped head and saved to {output_ply_path}")


    try:
        log_name = f"color_transfer_log{color_transfer_suffix}{color_transfer_mode_suffix}{direct_swap_mode_suffix}{swap_back_mode_suffix}.json"
        log_path = os.path.join(output_dir, log_name)
        with open(log_path, "w") as f:
            json.dump(color_transfer_log, f, indent=2)
        logger.info(f"Wrote color transfer log to {log_path}")
    except Exception as e:
        logger.warning(f"Failed to write color transfer log: {e}")


    return os.path.splitext(output_filename)[0]


def main(args):

    SURFACE_LABELS = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']
    DETAILED_SURFACE_LABELS = [
        "torso_skin", "head", "left_arm", "right_arm", "left_leg", "right_leg", "clothes", "hands"
    ]

    thuman2_repose_avatarrex_root = args.data_root
    dataset_prefix = _infer_dataset_type_from_any_path(thuman2_repose_avatarrex_root)

    user_A_subject_id = str(args.user_A_id)
    model_B_subject_id = str(args.model_B_id)
    splat_subdir_map = {
        user_A_subject_id: _resolve_splatting_subject_dir_name(thuman2_repose_avatarrex_root, user_A_subject_id),
        model_B_subject_id: _resolve_splatting_subject_dir_name(thuman2_repose_avatarrex_root, model_B_subject_id),
    }
    logger.info(f"Detected dataset_prefix={dataset_prefix}")
    logger.info(f"Resolved output-splatting dirs: {splat_subdir_map}")


    reposed_gs_base_head_donator = f"{thuman2_repose_avatarrex_root}/gs_on_mesh_repose/{dataset_prefix}_{{user_id}}_to_{{model_id}}"
    reposed_gs_base_body_donator = f"{thuman2_repose_avatarrex_root}/gs_on_mesh_repose/{dataset_prefix}_{{user_id}}_to_{{model_id}}"


    use_cloth_fit_reshape = args.use_cloth_fit_reshaped_gs

    if use_cloth_fit_reshape:

        logger.info("Body donator will use cloth-fit reshaped files (pattern: *_cloth_fit_reshaped.ply)")
        logger.info("Head donator paths will NOT include suffix (use standard repose paths)")
    elif args.repose_load_dir_subfix:

        reposed_gs_base_body_donator += f"_{args.repose_load_dir_subfix}"
        logger.info(f"Body donator paths will include SMPL reshape suffix: '{args.repose_load_dir_subfix}'")
        logger.info(f"Head donator paths will NOT include suffix (use standard repose paths)")
    else:
        logger.info("No reshape suffix specified - body donators will use standard reshape paths")

    label_suffix = "_extended" if args.use_detailed_labels else ""


    user_A_iter_path = f"iteration_{args.user_A_iteration}" if args.user_A_iteration else "latest"
    model_B_iter_path = f"iteration_{args.model_B_iteration}" if args.model_B_iteration else "latest"

    logger.info(f"Using iteration paths: User A -> {user_A_iter_path}, Model B -> {model_B_iter_path}")


    paths_config_example = {
        "original_avatar_ply": f"{thuman2_repose_avatarrex_root}/output-splatting/{{splat_subdir}}/point_cloud/{{iter_path}}/point_cloud.ply",
        "reposed_avatar_ply": f"{reposed_gs_base_head_donator}/reposed_gaussians_ply/reposed_gs_targetframe0000.ply",
        "segmentation_pkl": f"{thuman2_repose_avatarrex_root}/{{id}}/mesh/{args.label_dir_name}/label-f0000{label_suffix}.pkl",
        "embedding_json": f"{thuman2_repose_avatarrex_root}/output-splatting/{{splat_subdir}}/point_cloud/{{iter_path}}/embedding.json",
        "scan_mesh_obj": f"{thuman2_repose_avatarrex_root}/{{id}}/mesh/trimesh_cleaned/0000.obj",
        "smpl_mesh_obj": f"{thuman2_repose_avatarrex_root}/gs_on_mesh_repose/{dataset_prefix}_{{user_id}}_to_{{model_id}}/smpl_mesh_src_frame_0000.obj",
        "smpl_params_npz": f"{thuman2_repose_avatarrex_root}/{{id}}/smpl_params.npz",
        "output_dir_swapped": f"{thuman2_repose_avatarrex_root}/swapped/A{{user_id}}_B{{model_id}}"
    }


    paths_config_example["_splat_subdir_map"] = splat_subdir_map


    def get_subject_paths(subject_id):
        iter_path = user_A_iter_path if subject_id == user_A_subject_id else model_B_iter_path
        splat_subdir = splat_subdir_map.get(subject_id, _resolve_splatting_subject_dir_name(thuman2_repose_avatarrex_root, subject_id))
        return {
            "original_avatar_ply": paths_config_example["original_avatar_ply"].format(id=subject_id, iter_path=iter_path, splat_subdir=splat_subdir),
            "embedding_json": paths_config_example["embedding_json"].format(id=subject_id, iter_path=iter_path, splat_subdir=splat_subdir)
        }


    def construct_refined_swapped_subject_dir(head_id, body_id, data_root,
                                             color_transfer_enabled=True,
                                             color_transfer_mode="filter",
                                             color_transfer_threshold=0.01,
                                             use_reshaped=True):


        base_body = "reshaped_gs_target_shape_0000" if use_reshaped else "point_cloud"

        color_transfer_suffix = "_with_color_transfer" if color_transfer_enabled else ""
        if color_transfer_enabled:
            if color_transfer_mode == "weighted":
                color_transfer_mode_suffix = "_weighted"
            elif color_transfer_mode == "all_gs":
                color_transfer_mode_suffix = "_all_gs"
            else:
                color_transfer_mode_suffix = f"_filter_{color_transfer_threshold}"
            color_transfer_suffix += color_transfer_mode_suffix


        target_world_id = head_id


        refined_subject_name = f"swapped_{head_id.zfill(4)}head_on_{body_id.zfill(4)}body_from_{base_body}_in_A_world{color_transfer_suffix}_direct"


        refined_base = os.path.join(data_root, "head_swapped_renders")
        refined_subject_dir = os.path.join(refined_base, refined_subject_name)


        if os.path.exists(refined_subject_dir):
            return refined_subject_dir
        else:
            logger.warning(f"Refined swapped subject directory not found: {refined_subject_dir}")
            return None


    def construct_refined_gaussian_path(head_id, body_id, data_root,
                                       role="head_donator",
                                       direct_swap_mode=False,
                                       color_transfer_enabled=True,
                                       color_transfer_mode="filter",
                                       color_transfer_threshold=0.01,
                                       use_reshaped=True,
                                       reshape_suffix="",
                                       use_cloth_fit_reshape=False
                                       ):


        base_body = "reshaped_gs_target_shape_0000" if use_reshaped else "point_cloud"

        color_transfer_suffix = "_with_color_transfer" if color_transfer_enabled else ""
        if color_transfer_enabled:
            if color_transfer_mode == "weighted":
                color_transfer_mode_suffix = "_weighted"
            elif color_transfer_mode == "all_gs":
                color_transfer_mode_suffix = "_all_gs"
            elif color_transfer_mode == "filter":
                color_transfer_mode_suffix = f"_filter_{color_transfer_threshold}"
            else:
                color_transfer_mode_suffix = ""
        else:
            color_transfer_mode_suffix = ""

        direct_swap_mode_suffix = "_direct"


        refined_subject_name = (f"swapped_{head_id}head_on_{body_id}body_from_{base_body}_in_A_world"
                               f"{color_transfer_suffix}{color_transfer_mode_suffix}{direct_swap_mode_suffix}")

        if direct_swap_mode:
            if role == "head_donator":


                reposed_refined_base = os.path.join(data_root, "swapped_head_donator_reposed")
                reposed_refined_subject_dir = os.path.join(reposed_refined_base, refined_subject_name, "reposed_gaussians_ply")

                if not os.path.exists(reposed_refined_subject_dir):
                    logger.warning(f"Reposed refined head donator directory not found: {reposed_refined_subject_dir}")
                    return None, None

                reposed_ply_path = os.path.join(reposed_refined_subject_dir, "reposed_gs_targetframe0000.ply")

                if not os.path.exists(reposed_ply_path):
                    logger.warning(f"Reposed refined PLY file not found: {reposed_ply_path}")
                    return None, None

                return reposed_ply_path, "reposed"

            elif role == "body_donator":

                if use_reshaped or use_cloth_fit_reshape:

                    if use_cloth_fit_reshape:


                        reshaped_refined_base = os.path.join(data_root, "gs_on_mesh_repose")
                        dataset_prefix = _infer_dataset_type_from_any_path(data_root)
                        reshaped_refined_subject_dir = os.path.join(
                            reshaped_refined_base,
                            f"{dataset_prefix}_{body_id}_to_{head_id}",
                            "reshaped_gaussians_ply_0000",
                        )

                        if os.path.exists(reshaped_refined_subject_dir):

                            cloth_fit_ply_path = os.path.join(reshaped_refined_subject_dir, f"reshaped_gs_target_shape_0000.ply_cloth_fit_reshaped{args.cloth_fit_suffix}.ply")
                            if os.path.exists(cloth_fit_ply_path):
                                logger.info(f"Using cloth-fit reshaped refined PLY for body donator: {cloth_fit_ply_path}")
                                return cloth_fit_ply_path, "cloth_fit_reshaped"

                        logger.warning(f"Cloth-fit reshaped refined PLY not found for body donator, falling back to SMPL-based or standard")


                    refined_subject_name += reshape_suffix

                    reshaped_refined_base = os.path.join(data_root, "body_donator_reshaped")
                    reshaped_refined_subject_dir = os.path.join(reshaped_refined_base, refined_subject_name, "reshaped_gaussians_ply_0000")

                    if os.path.exists(reshaped_refined_subject_dir):
                        reshaped_ply_path = os.path.join(reshaped_refined_subject_dir, "reshaped_gs_target_shape_0000.ply")
                        if os.path.exists(reshaped_ply_path):
                            logger.info(f"Using SMPL-based reshaped refined PLY for body donator: {reshaped_ply_path}")
                            return reshaped_ply_path, "reshaped"

                    logger.warning(f"Reshaped refined PLY not found for body donator, falling back to standard refined training output")


                logger.info(f"Body donator using standard refined training output")


        refined_splatting_base = os.path.join(data_root, "head_swapped_renders", "output-splatting", "swapped", refined_subject_name, "point_cloud")

        if not os.path.exists(refined_splatting_base):
            logger.warning(f"Refined splatting subject directory not found: {refined_splatting_base}")
            return None, None


        iteration_dirs = glob.glob(os.path.join(refined_splatting_base, "iteration_*"))
        if not iteration_dirs:
            logger.warning(f"No iteration directories found in: {refined_splatting_base}")
            return None, None


        iteration_numbers = []
        for iteration_dir in iteration_dirs:
            match = re.search(r'iteration_(\d+)', iteration_dir)
            if match:
                iteration_numbers.append(int(match.group(1)))

        if not iteration_numbers:
            logger.warning(f"No valid iteration directories found in: {refined_splatting_base}")
            return None, None

        highest_iteration = max(iteration_numbers)
        iteration_dir = os.path.join(refined_splatting_base, f"iteration_{highest_iteration}")
        refined_ply_path = os.path.join(iteration_dir, "point_cloud.ply")

        if not os.path.exists(refined_ply_path):
            logger.warning(f"Refined PLY file not found: {refined_ply_path}")
            return None, None

        return refined_ply_path, str(highest_iteration)


    def get_subject_ply_path(subject_id, target_id, direct_swap_mode, role, enable_body_reshape=True, use_cloth_fit_reshape=False, cloth_fit_suffix=""):

        if direct_swap_mode:
            target_frame = 0
            frame_str = f"{target_frame:04d}"

            if role == 'head_donator':


                repose_output_dir = reposed_gs_base_head_donator.format(user_id=subject_id, model_id=target_id)
                reposed_ply_path = os.path.join(repose_output_dir, "reposed_gaussians_ply", "reposed_gs_targetframe0000.ply")

                if os.path.exists(reposed_ply_path):
                    logger.info(f"Head donator mode: Using reposed PLY: {reposed_ply_path}")
                    return reposed_ply_path, "reposed"
                else:
                    raise FileNotFoundError(f"Reposed PLY not found for head donator {subject_id}: {reposed_ply_path}")

            elif role == 'body_donator':


                reshape_output_dir = reposed_gs_base_body_donator.format(user_id=subject_id, model_id=target_id)
                reshaped_ply_path = os.path.join(reshape_output_dir, f"reshaped_gaussians_ply_{frame_str}", f"reshaped_gs_target_shape_{frame_str}.ply")
                original_ply_path = get_subject_paths(subject_id)['original_avatar_ply']

                if use_cloth_fit_reshape:
                    cloth_fit_ply_path = os.path.join(reshape_output_dir, f"reshaped_gaussians_ply_{frame_str}", f"reshaped_gs_target_shape_{frame_str}.ply_cloth_fit_reshaped{cloth_fit_suffix}.ply")
                    if os.path.exists(cloth_fit_ply_path):
                        logger.info(f"Body donator cloth-fit reshape enabled: Using cloth-fit reshaped PLY: {cloth_fit_ply_path}")
                        return cloth_fit_ply_path, "cloth_fit_reshaped"
                    else:
                        logger.warning(f"Cloth-fit reshaped PLY not found for body donator {subject_id}: {cloth_fit_ply_path}")

                if enable_body_reshape and os.path.exists(reshaped_ply_path):
                    logger.info(f"Body donator reshape enabled: Using reshaped PLY: {reshaped_ply_path}")
                    return reshaped_ply_path, "reshaped"
                elif os.path.exists(original_ply_path):
                    logger.info(f"Body donator default mode: Using original PLY: {original_ply_path}")
                    return original_ply_path, "original"
                else:

                    if os.path.exists(reshaped_ply_path):
                        logger.info(f"Body donator fallback mode: Using reshaped PLY: {reshaped_ply_path}")
                        return reshaped_ply_path, "reshaped"
                    else:
                        raise FileNotFoundError(f"Neither original nor reshaped PLY found for body donator {subject_id}, tried: {reshaped_ply_path} and {original_ply_path}")
            else:
                raise ValueError(f"Invalid role: {role}. Must be 'head_donator' or 'body_donator'")
        else:

            reposed_ply_path = paths_config_example['reposed_avatar_ply'].format(user_id=subject_id, model_id=target_id)
            return reposed_ply_path, "reposed"

    smplx_seg_json_path = args.smplx_seg_path

    user_A_subject_id = args.user_A_id
    model_B_subject_id = args.model_B_id


    cfg_use_direct_label = args.use_detailed_labels
    cfg_head_label_name = [args.head_label]
    if "_no_sam" in args.label_dir_name:
        logger.info(f"Using no_sam label for head alignment, adding torso_skin (neck) to head label")
        cfg_head_label_name.append("torso_skin")
    cfg_fallback_skin_name = args.fallback_skin_label
    cfg_hair_label_name = args.hair_label


    cfg_skin_label_for_color_transfer = args.skin_label_for_color_transfer
    cfg_color_transfer_opacity_threshold = args.color_transfer_opacity_threshold
    cfg_perform_color_transfer = args.perform_color_transfer
    cfg_debug_color_transfer_save_mesh = args.debug_color_transfer_save_mesh
    cfg_debug_color_transfer_force_color = args.debug_color_transfer_force_color
    cfg_debug_hardcoded_lab_mean = tuple(args.debug_lab_mean)
    cfg_debug_hardcoded_lab_std = tuple(args.debug_lab_std)
    cfg_target_body_parts_for_color_transfer = args.target_body_parts_for_color_transfer
    cfg_color_transfer_use_all_gaussians_for_stats = args.color_transfer_use_all_gaussians_for_stats
    cfg_color_transfer_use_opacity_weighting = args.color_transfer_use_opacity_weighting


    user_A_paths = get_subject_paths(user_A_subject_id)
    model_B_paths = get_subject_paths(model_B_subject_id)


    a_head_ply_path, a_head_type = get_subject_ply_path(
        user_A_subject_id, model_B_subject_id, args.direct_swap_mode, 'head_donator',
        enable_body_reshape=args.enable_body_reshape,
    )
    logger.info(f"Using {a_head_type} User A (head donator) PLY: {a_head_ply_path}")

    b_body_ply_path, b_body_type = get_subject_ply_path(
        model_B_subject_id, user_A_subject_id, args.direct_swap_mode, 'body_donator',
        enable_body_reshape=args.enable_body_reshape,
        use_cloth_fit_reshape=use_cloth_fit_reshape,
        cloth_fit_suffix=args.cloth_fit_suffix
    )
    logger.info(f"Using {b_body_type} Model B (body donator) PLY: {b_body_ply_path}")

    a_smpl_params = paths_config_example['smpl_params_npz'].format(id=user_A_subject_id)
    b_smpl_params = paths_config_example['smpl_params_npz'].format(id=model_B_subject_id)

    proceed = True
    if not os.path.exists(a_head_ply_path): logger.error(f"Error: User A head PLY missing: {a_head_ply_path}"); proceed = False
    if not os.path.exists(b_body_ply_path): logger.error(f"Error: Model B body PLY missing: {b_body_ply_path}"); proceed = False
    if not os.path.exists(a_smpl_params): logger.error(f"Error: User A SMPL params missing: {a_smpl_params}"); proceed = False
    if not os.path.exists(b_smpl_params): logger.error(f"Error: Model B SMPL params missing: {b_smpl_params}"); proceed = False

    if not cfg_use_direct_label and not os.path.exists(smplx_seg_json_path):
        logger.error(f"Error: SMPLX segmentation JSON ({smplx_seg_json_path}) not found, which is needed when not using direct head labels.")
        proceed = False

    if proceed:
        head_type_desc = f"{a_head_type} head" if args.direct_swap_mode else "original head"
        body_type_desc = f"{b_body_type} body" if args.direct_swap_mode else "body"
        logger.info(f"--- Running head swap: User A ('{user_A_subject_id}') {head_type_desc} onto Model B ('{model_B_subject_id}') {body_type_desc} ---")


        paths_config_with_iterations = paths_config_example.copy()
        paths_config_with_iterations.update({
            "_iteration_paths": {
                "user_A": user_A_iter_path,
                "model_B": model_B_iter_path
            }
        })


        if args.swap_back_mode:

            logger.info("Swap-back mode enabled - using refined Gaussians for identity restoration")


            color_transfer_mode = "weighted" if cfg_color_transfer_use_opacity_weighting else \
                                "all_gs" if cfg_color_transfer_use_all_gaussians_for_stats else "filter"


            a_head_refined_path, a_head_iter = construct_refined_gaussian_path(
                head_id=user_A_subject_id, body_id=model_B_subject_id,
                role="head_donator", direct_swap_mode=args.direct_swap_mode,
                data_root=thuman2_repose_avatarrex_root,
                color_transfer_enabled=cfg_perform_color_transfer,
                color_transfer_mode=color_transfer_mode,
                color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                use_reshaped=bool(args.repose_load_dir_subfix),
                use_cloth_fit_reshape=use_cloth_fit_reshape
            )


            a_body_refined_path, a_body_iter = construct_refined_gaussian_path(
                head_id=model_B_subject_id, body_id=user_A_subject_id,
                role="body_donator", direct_swap_mode=args.direct_swap_mode,
                data_root=thuman2_repose_avatarrex_root,
                color_transfer_enabled=cfg_perform_color_transfer,
                color_transfer_mode=color_transfer_mode,
                color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                use_reshaped=bool(args.repose_load_dir_subfix),
                reshape_suffix="_"+args.repose_load_dir_subfix,
                use_cloth_fit_reshape=use_cloth_fit_reshape
            )

            if a_head_refined_path is None or a_body_refined_path is None:
                logger.error("Cannot perform swap-back: Missing refined Gaussians for identity restoration")
                logger.error(f"A_head from A→B swap: {'✓' if a_head_refined_path else '✗'}")
                logger.error(f"A_body from B→A swap: {'✓' if a_body_refined_path else '✗'}")
                proceed = False
            else:
                logger.info(f"Swap-back paths for restoring A identity:")
                logger.info(f"  A_head source (from A→B): {a_head_refined_path}")
                logger.info(f"  A_body source (from B→A): {a_body_refined_path}")


                paths_config_with_iterations["original_avatar_ply"] = a_head_refined_path
                paths_config_with_iterations["reposed_avatar_ply"] = a_body_refined_path


                paths_config_with_iterations["output_dir_swapped"] = f"{thuman2_repose_avatarrex_root}/swapped_back"


                ab_swapped_subject_path = construct_refined_swapped_subject_dir(
                    head_id=user_A_subject_id, body_id=model_B_subject_id,
                    data_root=thuman2_repose_avatarrex_root,
                    color_transfer_enabled=cfg_perform_color_transfer,
                    color_transfer_mode=color_transfer_mode,
                    color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                    use_reshaped=bool(args.repose_load_dir_subfix)
                )


                ba_swapped_subject_path = construct_refined_swapped_subject_dir(
                    head_id=model_B_subject_id, body_id=user_A_subject_id,
                    data_root=thuman2_repose_avatarrex_root,
                    color_transfer_enabled=cfg_perform_color_transfer,
                    color_transfer_mode=color_transfer_mode,
                    color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                    use_reshaped=bool(args.repose_load_dir_subfix)
                )

                if ab_swapped_subject_path is None or ba_swapped_subject_path is None:
                    logger.error("Cannot perform swap-back: Missing refined swapped subject directories")
                    logger.error(f"A→B swapped dir: {'✓' if ab_swapped_subject_path else '✗'}")
                    logger.error(f"B→A swapped dir: {'✓' if ba_swapped_subject_path else '✗'}")
                    proceed = False
                else:
                    logger.info(f"Swap-back path overrides for A restoration:")
                    logger.info(f"  A_head paths from: {ab_swapped_subject_path}")
                    logger.info(f"  A_body paths from: {ba_swapped_subject_path}")


                    paths_config_with_iterations["segmentation_pkl_map"] = {
                        user_A_subject_id: f"{ab_swapped_subject_path}/mesh/labeled/label-f0000{label_suffix}.pkl",
                        model_B_subject_id: f"{ba_swapped_subject_path}/mesh/labeled/label-f0000{label_suffix}.pkl"
                    }

                    paths_config_with_iterations["scan_mesh_obj_map"] = {
                        user_A_subject_id: f"{ab_swapped_subject_path}/mesh/trimesh_cleaned/0000.obj",
                        model_B_subject_id: f"{ba_swapped_subject_path}/mesh/trimesh_cleaned/0000.obj"
                    }

                    paths_config_with_iterations["smpl_params_npz_map"] = {
                        user_A_subject_id: f"{ab_swapped_subject_path}/smpl_params.npz",
                        model_B_subject_id: f"{ba_swapped_subject_path}/smpl_params.npz"
                    }


                    ab_refined_subject_name = os.path.basename(ab_swapped_subject_path)

                    ba_refined_subject_name = os.path.basename(ba_swapped_subject_path)


                    paths_config_with_iterations["embedding_json_map"] = {
                        user_A_subject_id: f"{thuman2_repose_avatarrex_root}/head_swapped_renders/output-splatting/swapped/{ab_refined_subject_name}/point_cloud/{{iter_path}}/embedding.json",
                        model_B_subject_id: f"{thuman2_repose_avatarrex_root}/head_swapped_renders/output-splatting/swapped/{ba_refined_subject_name}/point_cloud/{{iter_path}}/embedding.json"
                    }

        elif args.direct_swap_mode:

            paths_config_with_iterations["original_avatar_ply"] = a_head_ply_path
            paths_config_with_iterations["reposed_avatar_ply"] = b_body_ply_path

        a_to_b_filename = swap_head_gaussians(
            user_A_id=user_A_subject_id,
            model_B_id=model_B_subject_id,
            paths_config=paths_config_with_iterations,
            smplx_vert_seg_path=smplx_seg_json_path,
            current_surface_labels=DETAILED_SURFACE_LABELS if args.use_detailed_labels else SURFACE_LABELS,
            use_direct_head_label_A=cfg_use_direct_label,
            direct_head_label_name_A=cfg_head_label_name,
            fallback_skin_label_name_A=cfg_fallback_skin_name,
            hair_label_name_A=cfg_hair_label_name,
            use_direct_head_label_B=cfg_use_direct_label,
            direct_head_label_name_B=cfg_head_label_name,
            fallback_skin_label_name_B=cfg_fallback_skin_name,
            hair_label_name_B=cfg_hair_label_name,
            skin_label_for_color_transfer=cfg_skin_label_for_color_transfer,
            color_transfer_opacity_threshold=cfg_color_transfer_opacity_threshold,
            perform_color_transfer=cfg_perform_color_transfer,
            debug_color_transfer_save_mesh=cfg_debug_color_transfer_save_mesh,
            debug_color_transfer_force_color=cfg_debug_color_transfer_force_color,
            debug_hardcoded_lab_mean=cfg_debug_hardcoded_lab_mean,
            debug_hardcoded_lab_std=cfg_debug_hardcoded_lab_std,
            target_body_parts_for_color_transfer=cfg_target_body_parts_for_color_transfer,
            use_detailed_labels_config=args.use_detailed_labels,
            color_transfer_use_all_gaussians_for_stats=cfg_color_transfer_use_all_gaussians_for_stats,
            color_transfer_use_opacity_weighting=cfg_color_transfer_use_opacity_weighting,
            repose_load_dir_subfix=args.repose_load_dir_subfix,
            gender_A_for_pelvis_joint=args.gender_A_for_pelvis_joint,
            gender_B_for_pelvis_joint=args.gender_B_for_pelvis_joint,
            direct_swap_mode=args.direct_swap_mode,
            swap_back_mode=args.swap_back_mode,
            enable_rigid_head_reposing=args.enable_rigid_head_reposing,
            save_body_donator_full_in_head_world=args.save_body_donator_full_in_head_world,
            save_swap_debug_artifacts=getattr(args, "save_swap_debug_artifacts", False),
            save_head_body_separation_debug=getattr(args, "save_head_body_separation_debug", False),
            swap_hands=getattr(args, "swap_hands", False),
        )

    if args.swap_both_directions:


        b_head_ply_path, b_head_type = get_subject_ply_path(
            model_B_subject_id, user_A_subject_id, args.direct_swap_mode, 'head_donator',
            enable_body_reshape=args.enable_body_reshape,
        )
        logger.info(f"Using {b_head_type} User B (head donator) PLY: {b_head_ply_path}")

        a_body_ply_path, a_body_type = get_subject_ply_path(
            user_A_subject_id, model_B_subject_id, args.direct_swap_mode, 'body_donator',
            enable_body_reshape=args.enable_body_reshape,
            use_cloth_fit_reshape=use_cloth_fit_reshape,
            cloth_fit_suffix=args.cloth_fit_suffix
        )

        head_type_reverse_desc = f"{b_head_type} head" if args.direct_swap_mode else "original head"
        body_type_reverse_desc = f"{a_body_type} body" if args.direct_swap_mode else "body"
        logger.info(f"\n--- Running head swap: User B ('{model_B_subject_id}') {head_type_reverse_desc} onto Model A ('{user_A_subject_id}') {body_type_reverse_desc} ---")
        logger.info(f"Using {a_body_type} Model A (body donator) PLY: {a_body_ply_path}")

        proceed_swap2 = True
        if not os.path.exists(b_head_ply_path): logger.error(f"Error: User B head PLY missing: {b_head_ply_path}"); proceed_swap2=False
        if not os.path.exists(a_body_ply_path): logger.error(f"Error: Model A body PLY missing: {a_body_ply_path}"); proceed_swap2=False


        if proceed_swap2:

            paths_config_reverse = paths_config_example.copy()
            paths_config_reverse.update({
                "_iteration_paths": {
                    "user_A": model_B_iter_path,
                    "model_B": user_A_iter_path
                }
            })


            if args.swap_back_mode:


                b_head_refined_path, b_head_iter = construct_refined_gaussian_path(
                    head_id=model_B_subject_id, body_id=user_A_subject_id,
                    role="head_donator", direct_swap_mode=args.direct_swap_mode,
                    data_root=thuman2_repose_avatarrex_root,
                    color_transfer_enabled=cfg_perform_color_transfer,
                    color_transfer_mode=color_transfer_mode,
                    color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                    use_reshaped=bool(args.repose_load_dir_subfix),
                    use_cloth_fit_reshape=use_cloth_fit_reshape
                )


                b_body_refined_path, b_body_iter = construct_refined_gaussian_path(
                    head_id=user_A_subject_id, body_id=model_B_subject_id,
                    role="body_donator", direct_swap_mode=args.direct_swap_mode,
                    data_root=thuman2_repose_avatarrex_root,
                    color_transfer_enabled=cfg_perform_color_transfer,
                    color_transfer_mode=color_transfer_mode,
                    color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                    use_reshaped=bool(args.repose_load_dir_subfix),
                    reshape_suffix="_"+args.repose_load_dir_subfix,
                    use_cloth_fit_reshape=use_cloth_fit_reshape
                )

                if b_head_refined_path is None or b_body_refined_path is None:
                    logger.error("Cannot perform swap-back (reverse): Missing refined Gaussians for B identity restoration")
                    logger.error(f"B_head from B→A swap: {'✓' if b_head_refined_path else '✗'}")
                    logger.error(f"B_body from A→B swap: {'✓' if b_body_refined_path else '✗'}")
                    proceed_swap2 = False
                else:
                    logger.info(f"Swap-back paths for restoring B identity:")
                    logger.info(f"  B_head source (from B→A): {b_head_refined_path}")
                    logger.info(f"  B_body source (from A→B): {b_body_refined_path}")


                    paths_config_reverse["original_avatar_ply"] = b_head_refined_path
                    paths_config_reverse["reposed_avatar_ply"] = b_body_refined_path


                    paths_config_reverse["output_dir_swapped"] = f"{thuman2_repose_avatarrex_root}/swapped_back"


                    ba_swapped_subject_path_b = construct_refined_swapped_subject_dir(
                        head_id=model_B_subject_id, body_id=user_A_subject_id,
                        data_root=thuman2_repose_avatarrex_root,
                        color_transfer_enabled=cfg_perform_color_transfer,
                        color_transfer_mode=color_transfer_mode,
                        color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                        use_reshaped=bool(args.repose_load_dir_subfix)
                    )


                    ab_swapped_subject_path_b = construct_refined_swapped_subject_dir(
                        head_id=user_A_subject_id, body_id=model_B_subject_id,
                        data_root=thuman2_repose_avatarrex_root,
                        color_transfer_enabled=cfg_perform_color_transfer,
                        color_transfer_mode=color_transfer_mode,
                        color_transfer_threshold=cfg_color_transfer_opacity_threshold,
                        use_reshaped=bool(args.repose_load_dir_subfix)
                    )

                    if ba_swapped_subject_path_b is None or ab_swapped_subject_path_b is None:
                        logger.error("Cannot perform swap-back (reverse): Missing refined swapped subject directories")
                        logger.error(f"B→A swapped dir: {'✓' if ba_swapped_subject_path_b else '✗'}")
                        logger.error(f"A→B swapped dir: {'✓' if ab_swapped_subject_path_b else '✗'}")
                        proceed_swap2 = False
                    else:
                        logger.info(f"Swap-back path overrides for B restoration:")
                        logger.info(f"  B_head paths from: {ba_swapped_subject_path_b}")
                        logger.info(f"  B_body paths from: {ab_swapped_subject_path_b}")


                        paths_config_reverse["segmentation_pkl_map"] = {
                            model_B_subject_id: f"{ba_swapped_subject_path_b}/mesh/labeled/label-f0000{label_suffix}.pkl",
                            user_A_subject_id: f"{ab_swapped_subject_path_b}/mesh/labeled/label-f0000{label_suffix}.pkl"
                        }

                        paths_config_reverse["scan_mesh_obj_map"] = {
                            model_B_subject_id: f"{ba_swapped_subject_path_b}/mesh/trimesh_cleaned/0000.obj",
                            user_A_subject_id: f"{ab_swapped_subject_path_b}/mesh/trimesh_cleaned/0000.obj"
                        }

                        paths_config_reverse["smpl_params_npz_map"] = {
                            model_B_subject_id: f"{ba_swapped_subject_path_b}/smpl_params.npz",
                            user_A_subject_id: f"{ab_swapped_subject_path_b}/smpl_params.npz"
                        }


                        ba_refined_subject_name = os.path.basename(ba_swapped_subject_path_b)

                        ab_refined_subject_name_b = os.path.basename(ab_swapped_subject_path_b)


                        paths_config_reverse["embedding_json_map"] = {
                            model_B_subject_id: f"{thuman2_repose_avatarrex_root}/head_swapped_renders/output-splatting/swapped/{ba_refined_subject_name}/point_cloud/{{iter_path}}/embedding.json",
                            user_A_subject_id: f"{thuman2_repose_avatarrex_root}/head_swapped_renders/output-splatting/swapped/{ab_refined_subject_name_b}/point_cloud/{{iter_path}}/embedding.json"
                        }

            elif args.direct_swap_mode:

                paths_config_reverse["original_avatar_ply"] = b_head_ply_path
                paths_config_reverse["reposed_avatar_ply"] = a_body_ply_path

            b_to_a_filename = swap_head_gaussians(
                user_A_id=model_B_subject_id,
                model_B_id=user_A_subject_id,
                paths_config=paths_config_reverse,
                smplx_vert_seg_path=smplx_seg_json_path,
                current_surface_labels=DETAILED_SURFACE_LABELS if args.use_detailed_labels else SURFACE_LABELS,
                use_direct_head_label_A=cfg_use_direct_label,
                direct_head_label_name_A=cfg_head_label_name,
                fallback_skin_label_name_A=cfg_fallback_skin_name,
                hair_label_name_A=cfg_hair_label_name,
                use_direct_head_label_B=cfg_use_direct_label,
                direct_head_label_name_B=cfg_head_label_name,
                fallback_skin_label_name_B=cfg_fallback_skin_name,
                hair_label_name_B=cfg_hair_label_name,
                skin_label_for_color_transfer=cfg_skin_label_for_color_transfer,
                color_transfer_opacity_threshold=cfg_color_transfer_opacity_threshold,
                perform_color_transfer=cfg_perform_color_transfer,
                debug_color_transfer_save_mesh=cfg_debug_color_transfer_save_mesh,
                debug_color_transfer_force_color=cfg_debug_color_transfer_force_color,
                debug_hardcoded_lab_mean=cfg_debug_hardcoded_lab_mean,
                debug_hardcoded_lab_std=cfg_debug_hardcoded_lab_std,
                target_body_parts_for_color_transfer=cfg_target_body_parts_for_color_transfer,
                use_detailed_labels_config=args.use_detailed_labels,
                color_transfer_use_all_gaussians_for_stats=cfg_color_transfer_use_all_gaussians_for_stats,
                color_transfer_use_opacity_weighting=cfg_color_transfer_use_opacity_weighting,
                repose_load_dir_subfix=args.repose_load_dir_subfix,
                gender_A_for_pelvis_joint=args.gender_B_for_pelvis_joint,
                gender_B_for_pelvis_joint=args.gender_A_for_pelvis_joint,
                direct_swap_mode=args.direct_swap_mode,
                swap_back_mode=args.swap_back_mode,
                enable_rigid_head_reposing=args.enable_rigid_head_reposing,
                save_body_donator_full_in_head_world=args.save_body_donator_full_in_head_world,
                save_swap_debug_artifacts=getattr(args, "save_swap_debug_artifacts", False),
                save_head_body_separation_debug=getattr(args, "save_head_body_separation_debug", False),
                swap_hands=getattr(args, "swap_hands", False),
            )
        else:

            b_to_a_filename = None


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
