import argparse
import sys
import numpy as np
import torch
import cv2
from pathlib import Path
import json
import pickle
import trimesh
from loguru import logger
import re
import os
import pymeshlab
import inspect


sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from src.scripts.swapping.head_swap import (
    load_smpl_transform_params_for_frame,
    get_pelvis_joint_j0,
    _transform_gaussian_positions,
    identify_head_region
)


from src.scripts.mesh.mesh_nonhead_utils import (
    prepare_nerf_mesh_with_cleanup,
    apply_mesh_decimation,
    apply_pymeshlab_cleanup,
    log_cleanup_statistics,
    identify_skin_mask_binary,
    extract_skin_indices_from_mask
)


from src.scripts.mesh.mesh_skeleton_utils import extract_skeleton_mesh, export_skeleton_lbs_weights


sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.smplx_utils.smplx_utils import create_smplx_model_for_neus2_reposing
from utils.smplx_utils.smplx_models.smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES


SMPLX_HEAD_TOP_IDX = 8976
SMPLX_LEFT_HEEL_IDX = 8847
SMPLX_RIGHT_HEEL_IDX = 8635


def _load_joint_names_for_lbs(*, frame_output_dir: Path, n_joints: int) -> list[str]:

    joint_names_path = frame_output_dir / "smpl_joint_names.txt"
    if joint_names_path.exists():
        try:
            names = [ln.strip() for ln in joint_names_path.read_text().splitlines() if ln.strip()]
            if len(names) == int(n_joints):
                return names
            logger.warning(
                f"smpl_joint_names.txt length mismatch: got {len(names)}, expected {int(n_joints)}. "
                "Falling back to built-in SMPL-X joint names."
            )
        except Exception as e:
            logger.warning(f"Failed reading smpl_joint_names.txt ({joint_names_path}): {e}")

    if len(SMPLX_JOINT_NAMES) < int(n_joints):
        logger.warning(
            f"Built-in SMPLX_JOINT_NAMES has only {len(SMPLX_JOINT_NAMES)} entries but LBS has {int(n_joints)} joints. "
            "Semantic weighting will likely be skipped."
        )
    return list(SMPLX_JOINT_NAMES[: int(n_joints)])


def _mask_dilate_vertex_rings(mask: np.ndarray, *, mesh: trimesh.Trimesh, rings: int) -> np.ndarray:

    if rings <= 0:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    neighbors = mesh.vertex_neighbors
    for _ in range(int(rings)):
        idx = np.flatnonzero(m)
        if idx.size == 0:
            break
        nbr_list = []
        for v in idx.tolist():
            nbs = neighbors[int(v)]
            if nbs:
                nbr_list.append(np.asarray(nbs, dtype=np.int64))
        if not nbr_list:
            break
        nbr = np.concatenate(nbr_list, axis=0)
        m[nbr] = True
    return m


def _mask_remove_small_components(
    mask: np.ndarray, *, mesh: trimesh.Trimesh, min_size: int
) -> np.ndarray:

    if min_size <= 0:
        return np.asarray(mask, dtype=bool)
    m = np.asarray(mask, dtype=bool)
    neighbors = mesh.vertex_neighbors
    n = int(m.shape[0])
    visited = np.zeros((n,), dtype=bool)
    out = m.copy()

    for seed in np.flatnonzero(m):
        if visited[int(seed)]:
            continue
        stack = [int(seed)]
        visited[int(seed)] = True
        comp = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in neighbors[v]:
                nb = int(nb)
                if not visited[nb] and m[nb]:
                    visited[nb] = True
                    stack.append(nb)
        if len(comp) < int(min_size):
            out[np.asarray(comp, dtype=np.int64)] = False
    return out


def _estimate_height_m_from_model_input_params(
    *,
    smpl_model,
    model_input_params: dict,
    device: torch.device,
) -> float | None:

    if model_input_params is None or "betas" not in model_input_params:
        logger.warning("Height estimate skipped: missing 'betas' in SMPL params.")
        return None


    forward_param_names = set(inspect.signature(smpl_model.forward).parameters.keys())
    forward_param_names.discard("self")

    canon_params: dict[str, torch.Tensor] = {}
    for k, v in model_input_params.items():
        if k not in forward_param_names:
            continue
        if not isinstance(v, torch.Tensor):
            continue
        canon_params[k] = v.to(device)


    if "betas" not in canon_params:
        logger.warning("Height estimate skipped: SMPL forward does not accept 'betas' or it was filtered out.")
        return None


    zero_keys = (
        "global_orient",
        "transl",
        "body_pose",
        "left_hand_pose",
        "right_hand_pose",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
    )
    for zk in zero_keys:
        if zk in canon_params and isinstance(canon_params[zk], torch.Tensor):
            canon_params[zk] = torch.zeros_like(canon_params[zk])


    batched: dict[str, torch.Tensor] = {}
    for k, t in canon_params.items():
        if not isinstance(t, torch.Tensor):
            continue
        if t.ndim == 0:
            batched[k] = t.view(1, 1)
        elif t.ndim == 1:
            batched[k] = t.unsqueeze(0)
        else:
            batched[k] = t

    with torch.no_grad():
        out = smpl_model(**batched, return_verts=True, return_joints=False)
        if not hasattr(out, "vertices") or out.vertices is None:
            logger.warning("Height estimate skipped: SMPL forward did not return vertices.")
            return None
        verts = out.vertices.detach().squeeze(0)

    n_verts = int(verts.shape[0])
    max_idx = max(SMPLX_HEAD_TOP_IDX, SMPLX_LEFT_HEEL_IDX, SMPLX_RIGHT_HEEL_IDX)
    if n_verts <= max_idx:
        logger.warning(
            f"Height estimate skipped: unexpected SMPL-X vertex count {n_verts} (needs > {max_idx})."
        )
        return None

    head = verts[SMPLX_HEAD_TOP_IDX]
    heel_mid = (verts[SMPLX_LEFT_HEEL_IDX] + verts[SMPLX_RIGHT_HEEL_IDX]) * 0.5
    height_m = torch.linalg.norm(head - heel_mid).item()
    return float(height_m)


SURFACE_LABELS = ["skin", "hair", "shoe", "upper", "lower", "outer"]


DETAILED_SURFACE_LABELS = [
    "torso_skin",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "clothes",
    "hands",
    "shoes",
]

SURFACE_LABEL_COLOR_UINT8 = np.array(
    [
        [128, 128, 128],
        [255, 128, 0],
        [128, 0, 255],
        [180, 50, 50],
        [50, 180, 50],
        [0, 128, 255],
    ],
    dtype=np.uint8,
)


SMPL_BODY_COLOR_UINT8 = (
    torch.tensor([0.7, 0.7, 0.7], dtype=torch.float32).cpu().numpy() * 255
).astype(
    np.uint8
)


def _apply_mvhumannet_external_rigid_transform(
    *,
    smpl_model,
    smpl_params_batch_on_device: dict,
    return_verts: bool,
    return_joints: bool,
):

    if "global_orient" not in smpl_params_batch_on_device or "transl" not in smpl_params_batch_on_device:
        return None

    go = smpl_params_batch_on_device["global_orient"]
    tr = smpl_params_batch_on_device["transl"]
    if not (isinstance(go, torch.Tensor) and isinstance(tr, torch.Tensor)):
        return None


    smpl_params_canon = dict(smpl_params_batch_on_device)
    smpl_params_canon["global_orient"] = torch.zeros_like(go)
    smpl_params_canon["transl"] = torch.zeros_like(tr)

    smpl_output_canon = smpl_model(
        **smpl_params_canon,
        return_verts=return_verts,
        return_joints=return_joints,
    )


    rot_np = cv2.Rodrigues(go.squeeze(0).detach().cpu().numpy())[0]
    rot = torch.from_numpy(rot_np).to(go.device, go.dtype).unsqueeze(0)
    tr_b = tr.unsqueeze(1)

    if return_verts and hasattr(smpl_output_canon, "vertices") and smpl_output_canon.vertices is not None:
        verts_c = smpl_output_canon.vertices
        smpl_output_canon.vertices = torch.einsum("bij,bvj->bvi", rot, verts_c) + tr_b

    if return_joints and hasattr(smpl_output_canon, "joints") and smpl_output_canon.joints is not None:
        joints_c = smpl_output_canon.joints
        smpl_output_canon.joints = torch.einsum("bij,bvj->bvi", rot, joints_c) + tr_b

    return smpl_output_canon


def _apply_post_forward_rigid_transform(
    *,
    smpl_output,
    Rh: torch.Tensor,
    Th: torch.Tensor,
    return_verts: bool,
    return_joints: bool,
):

    if not (isinstance(Rh, torch.Tensor) and isinstance(Th, torch.Tensor)):
        return smpl_output
    rot_np = cv2.Rodrigues(Rh.squeeze(0).detach().cpu().numpy())[0]
    rot = torch.from_numpy(rot_np).to(Rh.device, Rh.dtype).unsqueeze(0)
    tr_b = Th.unsqueeze(1)

    if return_verts and hasattr(smpl_output, "vertices") and smpl_output.vertices is not None:
        smpl_output.vertices = torch.einsum("bij,bvj->bvi", rot, smpl_output.vertices) + tr_b
    if return_joints and hasattr(smpl_output, "joints") and smpl_output.joints is not None:
        smpl_output.joints = torch.einsum("bij,bvj->bvi", rot, smpl_output.joints) + tr_b
    return smpl_output


def is_swapped_subject(subject_name):

    return (subject_name.startswith('swapped_') and
            'head_on' in subject_name and
            'body' in subject_name)


def parse_swapped_subject_name(subject_name):


    pattern = r'swapped_(\d+)head_on_(\d+)body_'
    match = re.match(pattern, subject_name)

    if match:
        head_subject_id = match.group(1)
        body_subject_id = match.group(2)
        logger.debug(f"Parsed swapped subject: head={head_subject_id}, body={body_subject_id}")
        return head_subject_id, body_subject_id
    else:
        logger.warning(f"Failed to parse swapped subject name: {subject_name}")
        return None, None


def get_subject_name_from_path(data_dir):

    return Path(data_dir).name


def determine_subject_gender_from_config(head_subject_id, body_subject_id, config):


    config_subjects = config.get('subjects', [])
    config_gender_a = config.get('smpl_gender_a', 'neutral')
    config_gender_b = config.get('smpl_gender_b', 'neutral')


    head_padded = get_padded_subject_id(head_subject_id)
    body_padded = get_padded_subject_id(body_subject_id)


    head_gender = config_gender_a
    body_gender = config_gender_a

    if len(config_subjects) >= 2:
        subject_a = config_subjects[0]
        subject_b = config_subjects[1]


        if head_padded == subject_a:
            head_gender = config_gender_a
        elif head_padded == subject_b:
            head_gender = config_gender_b
        else:
            logger.warning(f"Head donator {head_padded} not found in config subjects {config_subjects}. Using gender A: {config_gender_a}")


        if body_padded == subject_a:
            body_gender = config_gender_a
        elif body_padded == subject_b:
            body_gender = config_gender_b
        else:
            logger.warning(f"Body donator {body_padded} not found in config subjects {config_subjects}. Using gender A: {config_gender_a}")

        logger.info(f"Gender assignment: head donator {head_padded} -> {head_gender}, body donator {body_padded} -> {body_gender}")
    else:
        logger.warning(f"Insufficient subjects in config ({len(config_subjects)}). Using default gender A for both: {config_gender_a}")

    return head_gender, body_gender


def _resolve_gender_auto(gender: str, subject_dir: Path) -> str:

    g = str(gender).strip().lower()
    if g in {"male", "female", "neutral"}:
        return g
    if g != "auto":
        logger.warning(f"Unknown gender '{gender}', fallback to neutral")
        return "neutral"
    gpath = subject_dir / "gender.txt"
    if not gpath.exists():
        logger.warning(f"gender=auto but {gpath} not found. Fallback to neutral.")
        return "neutral"
    try:
        txt = gpath.read_text().strip().lower()
        if txt in {"m", "male"}:
            return "male"
        if txt in {"f", "female"}:
            return "female"
        if txt in {"n", "neutral"}:
            return "neutral"
        logger.warning(f"Unrecognized contents in {gpath}: '{txt}', fallback to neutral")
        return "neutral"
    except Exception as e:
        logger.warning(f"Failed reading {gpath}: {e}, fallback to neutral")
        return "neutral"


def _maybe_load_smpl_params_for_model_init(subject_dir: Path) -> dict | None:

    sp = subject_dir / "smpl_params.npz"
    if not sp.exists():
        return None
    try:
        npz = np.load(sp)
        out = {k: torch.tensor(v.astype(np.float32)) for k, v in npz.items()}
        return out
    except Exception as e:
        logger.warning(f"Failed to load {sp} for model init: {e}")
        return None


def build_smplx_mesh(
    smpl_model,
    smpl_params_batch,
    device,
    return_pelvis_joint=False,
    debug_save_mesh=False,
    return_vertices=True,
    return_full_joints=False,
    dataset_type: str = "thuman2",
):


    smpl_model_params_on_device = {
        name: param.to(device)
        for name, param in smpl_model.named_parameters()
        if hasattr(param, "to")
    }
    param_dims = {
        name: param.ndim
        for name, param in smpl_model_params_on_device.items()
        if hasattr(param, "ndim")
    }


    forward_param_names = set(inspect.signature(smpl_model.forward).parameters.keys())
    forward_param_names.discard("self")
    smpl_params_batch_on_device = {}
    for k, v_tensor in smpl_params_batch.items():
        if k not in forward_param_names:
            continue

        if not isinstance(v_tensor, torch.Tensor):


            smpl_params_batch_on_device[k] = v_tensor
            continue


        v_on_device = v_tensor.to(device)


        if v_on_device.ndim == 0:
            smpl_params_batch_on_device[k] = v_on_device.view(1, 1)
        elif v_on_device.ndim == 1:
            smpl_params_batch_on_device[k] = v_on_device.unsqueeze(0)
        else:
            smpl_params_batch_on_device[k] = v_on_device


    with torch.no_grad():

        need_joints = return_pelvis_joint or return_full_joints
        return_verts_flag = return_vertices
        return_joints_flag = need_joints


        smpl_output_world = smpl_model(
            **smpl_params_batch_on_device,
                                      return_verts=return_verts_flag,
            return_joints=return_joints_flag,
        )


        if (return_vertices or return_joints_flag):
            if dataset_type in ("mvhumannet", "actorshq"):
                corrected = _apply_mvhumannet_external_rigid_transform(
                    smpl_model=smpl_model,
                    smpl_params_batch_on_device=smpl_params_batch_on_device,
                    return_verts=return_verts_flag,
                    return_joints=return_joints_flag,
                )
                if corrected is not None:
                    smpl_output_world = corrected
                else:
                    logger.warning(
                        f"{dataset_type} external Rh/Th correction skipped (missing/invalid global_orient/transl). "
                        "Mesh may be globally shifted."
                    )
            elif dataset_type == "talkbody4d":

                Rh = smpl_params_batch.get("Rh", None)
                Th = smpl_params_batch.get("Th", None)
                if Rh is None or Th is None:

                    Rh = smpl_params_batch_on_device.get("global_orient", None)
                    Th = smpl_params_batch_on_device.get("transl", None)
                if Rh is not None and Th is not None:
                    Rh_b = Rh.to(device) if isinstance(Rh, torch.Tensor) else Rh
                    Th_b = Th.to(device) if isinstance(Th, torch.Tensor) else Th
                    if isinstance(Rh_b, torch.Tensor) and Rh_b.ndim == 1:
                        Rh_b = Rh_b.unsqueeze(0)
                    if isinstance(Th_b, torch.Tensor) and Th_b.ndim == 1:
                        Th_b = Th_b.unsqueeze(0)
                    smpl_output_world = _apply_post_forward_rigid_transform(
                        smpl_output=smpl_output_world,
                        Rh=Rh_b,
                        Th=Th_b,
                        return_verts=return_verts_flag,
                        return_joints=return_joints_flag,
                    )


        verts_torch = None
        if return_vertices:
            verts_torch = smpl_output_world.vertices.detach().squeeze(0)


        pelvis_joint = None
        all_joints = None

        if return_full_joints:

            all_joints = smpl_output_world.joints.detach().squeeze(0)
            logger.debug(f"Extracted {all_joints.shape[0]} joints for skeleton")

        elif return_pelvis_joint:


            smpl_params_canonical = {k: v for k, v in smpl_params_batch_on_device.items()
                                    if k not in ['global_orient', 'transl', 'scale']}

            smpl_output_canonical = smpl_model(**smpl_params_canonical, return_verts=False, return_joints=True)


            joints_tensor = smpl_output_canonical.joints.detach()
            pelvis_joint = joints_tensor.squeeze(0)[0]

            logger.debug(f"Extracted pelvis joint j0: {pelvis_joint.cpu().numpy()}")


        faces_np = None
        if return_vertices:
            faces_np = smpl_model.faces.astype(np.int64)


    if return_full_joints:

        if all_joints is None:
            raise RuntimeError("SMPL model failed to return joint information for skeleton extraction")
        if all_joints.shape[0] < 22:
            raise RuntimeError(f"Insufficient joints returned: {all_joints.shape[0]} < 22")
        return all_joints


    smpl_trimesh = None
    if return_vertices and verts_torch is not None:

        verts_np = verts_torch.cpu().numpy()

        if smpl_params_batch_on_device.get("scale") is not None:
            assert (
                smpl_output_world.use_scale == True
            )


        num_verts = verts_np.shape[0]

        smpl_vertex_colors_np = np.tile(SMPL_BODY_COLOR_UINT8, (num_verts, 1))

        smpl_trimesh = trimesh.Trimesh(
            vertices=verts_np, faces=faces_np, vertex_colors=smpl_vertex_colors_np
        )


    if return_pelvis_joint:
        return smpl_trimesh, pelvis_joint
    else:
        return smpl_trimesh

def get_padded_subject_id(subject_id):

    return f"{int(subject_id):04d}"

def apply_smpl_parameter_rectification(model_input_params, rectified_scale=1.0, rectified_transl=None, dataset_type="thuman2"):

    rectification_info = {"scale_applied": False, "transl_applied": False}


    if rectified_scale != 1.0 and model_input_params.get("scale") is not None:
        original_scale_val = model_input_params["scale"].clone()
        model_input_params["scale"] = torch.tensor(
            [rectified_scale],
            dtype=model_input_params["scale"].dtype,
            device=model_input_params["scale"].device,
        )
        rectification_info["scale_applied"] = True
        rectification_info["original_scale"] = original_scale_val.item()
        rectification_info["rectified_scale"] = rectified_scale
        logger.debug(f"Applied scale rectification: {original_scale_val.item()} -> {rectified_scale}")


    if rectified_transl is not None and rectified_transl != [0.0, 0.0, 0.0] and model_input_params.get("transl") is not None:
        original_transl_val = model_input_params["transl"].clone()
        new_transl = torch.tensor(
            rectified_transl,
            dtype=original_transl_val.dtype,
            device=original_transl_val.device,
        )
        model_input_params["transl"] = new_transl
        rectification_info["transl_applied"] = True
        rectification_info["original_transl"] = original_transl_val.tolist()
        rectification_info["rectified_transl"] = rectified_transl
        logger.debug(f"Applied translation rectification: {original_transl_val.tolist()} -> {rectified_transl}")

    return rectification_info

def get_head_donator_pelvis_joint(head_params_path, smpl_model, frame_idx=0, device='cpu',
                                 rectified_scale=1.0, rectified_transl=None, dataset_type='thuman2'):

    logger.debug(f"Loading head donator SMPL parameters from {head_params_path}")


    head_smpl_data = load_smpl_params(head_params_path, device)


    if frame_idx >= head_smpl_data["body_pose"].shape[0]:
        raise IndexError(f"Frame index {frame_idx} is out of bounds for head donator SMPL params")

    head_smpl_params = {
        k: v[frame_idx] for k, v in head_smpl_data.items() if v.shape[0] > frame_idx
    }


    if "betas" in head_smpl_data:
        if head_smpl_data["betas"].shape[0] == 1:
            head_smpl_params["betas"] = head_smpl_data["betas"][0]
        else:
            head_smpl_params["betas"] = head_smpl_data["betas"][frame_idx]


    head_model_input_params = {k: v for k, v in head_smpl_params.items()}


    if (dataset_type == "thuman2" and
        head_model_input_params.get("translation") is not None):
        head_model_input_params["transl"] = head_model_input_params.pop("translation")


    rectified_transl_list = rectified_transl if rectified_transl is not None else [0.0, 0.0, 0.0]
    rectification_info = apply_smpl_parameter_rectification(
        head_model_input_params, rectified_scale, rectified_transl_list, dataset_type
    )

    if rectification_info["scale_applied"] or rectification_info["transl_applied"]:
        logger.info(f"Applied rectification to head donator parameters: {rectification_info}")


    _, head_pelvis_joint = build_smplx_mesh(smpl_model, head_model_input_params, device, return_pelvis_joint=True)

    logger.debug(f"Extracted head donator pelvis joint: {head_pelvis_joint.cpu().numpy()}")
    return head_pelvis_joint


def load_smpl_params(smpl_params_path, device='cpu'):

    logger.debug(f"Loading SMPL parameters from {smpl_params_path}")

    if not os.path.exists(smpl_params_path):
        raise FileNotFoundError(f"SMPL parameters file not found: {smpl_params_path}")

    try:
        smpl_data_npz = np.load(smpl_params_path)
        smpl_data = {
            k: torch.tensor(v.astype(np.float32)).to(device)
            for k, v in smpl_data_npz.items()
        }
        logger.debug(f"Loaded SMPL parameters with keys: {list(smpl_data.keys())}")
        return smpl_data
    except Exception as e:
        raise RuntimeError(f"Error loading SMPL parameters from {smpl_params_path}: {e}")

def copy_smpl_params_for_swapped_subject(data_dir, head_subject_id, body_subject_id, avatarrex_base_dir, use_body_reshaping=False):

    import shutil

    data_dir = Path(data_dir)
    avatarrex_base_dir = Path(avatarrex_base_dir)

    logger.info(f"Setting up SMPL parameters for swapped subject {head_subject_id}→{body_subject_id}")


    head_padded = get_padded_subject_id(head_subject_id)
    body_padded = get_padded_subject_id(body_subject_id)

    head_source_path = avatarrex_base_dir / head_padded / "smpl_params.npz"
    body_source_path = avatarrex_base_dir / body_padded / "smpl_params.npz"


    head_dest_path = data_dir / "head_donator_smpl_params.npz"
    body_dest_path = data_dir / "body_donator_smpl_params.npz"
    combined_dest_path = data_dir / "smpl_params.npz"

    logger.debug(f"Head source: {head_source_path}")
    logger.debug(f"Body source: {body_source_path}")
    logger.debug(f"Head dest: {head_dest_path}")
    logger.debug(f"Body dest: {body_dest_path}")
    logger.debug(f"Combined dest: {combined_dest_path}")


    if not head_source_path.exists():
        raise FileNotFoundError(f"Head donator SMPL parameters not found: {head_source_path}")
    if not body_source_path.exists():
        raise FileNotFoundError(f"Body donator SMPL parameters not found: {body_source_path}")


    head_needs_copy = True
    body_needs_copy = True
    combined_needs_create = True

    if head_dest_path.exists() and head_source_path.stat().st_mtime <= head_dest_path.stat().st_mtime:
        logger.debug(f"Head donator SMPL params already up to date: {head_dest_path}")
        head_needs_copy = False

    if body_dest_path.exists() and body_source_path.stat().st_mtime <= body_dest_path.stat().st_mtime:
        logger.debug(f"Body donator SMPL params already up to date: {body_dest_path}")
        body_needs_copy = False


    if combined_dest_path.exists():
        combined_mtime = combined_dest_path.stat().st_mtime
        head_mtime = head_source_path.stat().st_mtime
        body_mtime = body_source_path.stat().st_mtime
        if combined_mtime >= head_mtime and combined_mtime >= body_mtime:
            logger.debug(f"Combined SMPL params already up to date: {combined_dest_path}")
            combined_needs_create = False

    try:

        if head_needs_copy:
            shutil.copy2(str(head_source_path), str(head_dest_path))
            logger.success(f"Copied head donator SMPL params: {head_source_path} -> {head_dest_path}")

        if body_needs_copy:
            shutil.copy2(str(body_source_path), str(body_dest_path))
            logger.success(f"Copied body donator SMPL params: {body_source_path} -> {body_dest_path}")


        if combined_needs_create:
            logger.info(f"Creating combined SMPL parameters for swapped subject")


            head_params = np.load(head_source_path, allow_pickle=True)
            body_params = np.load(body_source_path, allow_pickle=True)


            combined_params = {}


            pose_keys = ['body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose']
            for key in pose_keys:
                if key in body_params:
                    combined_params[key] = body_params[key]
                    logger.debug(f"Using body donator {key}: shape {body_params[key].shape}")
                elif key in head_params:

                    combined_params[key] = head_params[key]
                    logger.debug(f"Fallback to head donator {key}: shape {head_params[key].shape}")


            world_keys = ['global_orient', 'transl', 'scale']
            for key in world_keys:
                if key in head_params:
                    combined_params[key] = head_params[key]
                    logger.debug(f"Using head donator {key}: shape {head_params[key].shape if hasattr(head_params[key], 'shape') else type(head_params[key])}")
                elif key in body_params:

                    combined_params[key] = body_params[key]
                    logger.debug(f"Fallback to body donator {key}: shape {body_params[key].shape if hasattr(body_params[key], 'shape') else type(body_params[key])}")


            shape_keys = ['betas']
            for key in shape_keys:
                if use_body_reshaping:

                    if key in head_params:
                        combined_params[key] = head_params[key]
                        logger.debug(f"Using head donator {key} (body reshaping enabled): shape {head_params[key].shape}")
                    elif key in body_params:

                        combined_params[key] = body_params[key]
                        logger.debug(f"Fallback to body donator {key} (body reshaping enabled, head missing): shape {body_params[key].shape}")
                else:

                    if key in body_params:
                        combined_params[key] = body_params[key]
                        logger.debug(f"Using body donator {key} (no body reshaping): shape {body_params[key].shape}")
                    elif key in head_params:

                        combined_params[key] = head_params[key]
                        logger.debug(f"Fallback to head donator {key} (no body reshaping, body missing): shape {head_params[key].shape}")


            if 'expression' in body_params:
                combined_params['expression'] = body_params['expression']
                logger.debug(f"Using body donator expression: shape {body_params['expression'].shape}")
            elif 'expression' in head_params:
                combined_params['expression'] = head_params['expression']
                logger.debug(f"Using head donator expression: shape {head_params['expression'].shape}")


            essential_keys = ['body_pose', 'global_orient', 'transl', 'betas']
            missing_keys = [key for key in essential_keys if key not in combined_params]
            if missing_keys:
                raise ValueError(f"Missing essential SMPL parameters: {missing_keys}")


            logger.info(f"Saving combined SMPL parameters to: {combined_dest_path}")
            np.savez(combined_dest_path, **combined_params)

            logger.success(f"Created combined SMPL parameters for swapped subject:")
            logger.info(f"  - Pose data from body donator ({body_subject_id})")
            logger.info(f"  - World transform from head donator ({head_subject_id})")
            if use_body_reshaping:
                logger.info(f"  - Body shape from head donator ({head_subject_id}) - reshaping enabled")
            else:
                logger.info(f"  - Body shape from body donator ({body_subject_id}) - no reshaping")

        if not head_needs_copy and not body_needs_copy and not combined_needs_create:
            logger.info("All SMPL parameters already up to date, skipped copying and combining")

        return str(head_dest_path), str(body_dest_path), str(combined_dest_path)

    except Exception as e:
        raise RuntimeError(f"Error setting up SMPL parameters for swapped subject: {e}")

def load_swapped_subject_transform_params(head_subject_id, body_subject_id, avatarrex_base_dir, device='cpu'):


    head_padded = get_padded_subject_id(head_subject_id)
    body_padded = get_padded_subject_id(body_subject_id)

    head_smpl_path = os.path.join(avatarrex_base_dir, head_padded, "smpl_params.npz")
    body_smpl_path = os.path.join(avatarrex_base_dir, body_padded, "smpl_params.npz")


    head_global_orient, head_transl, head_scale = load_smpl_transform_params_for_frame(
        head_smpl_path, frame_idx=0, device=device
    )
    body_global_orient, body_transl, body_scale = load_smpl_transform_params_for_frame(
        body_smpl_path, frame_idx=0, device=device
    )

    logger.debug(f"Head donator ({head_subject_id}) transform: scale={head_scale:.4f}, transl={head_transl.cpu().numpy().tolist()}")
    logger.debug(f"Body donator ({body_subject_id}) transform: scale={body_scale:.4f}, transl={body_transl.cpu().numpy().tolist()}")

    return (head_global_orient, head_transl, head_scale), (body_global_orient, body_transl, body_scale), (head_smpl_path, body_smpl_path)


def transform_mesh_coordinates(mesh, from_transform_params, to_transform_params,
                             from_pelvis_joint, to_pelvis_joint, device='cpu'):

    from_global_orient, from_transl, from_scale = from_transform_params
    to_global_orient, to_transl, to_scale = to_transform_params


    from_pelvis_j0 = from_pelvis_joint
    to_pelvis_j0 = to_pelvis_joint

    logger.debug(f"Using pre-computed source pelvis joint: {from_pelvis_j0.cpu().numpy()}")
    logger.debug(f"Using pre-computed target pelvis joint: {to_pelvis_j0.cpu().numpy()}")


    vertices_tensor = torch.from_numpy(mesh.vertices.copy()).float().to(device)


    logger.debug("Transforming mesh from source world space to canonical space...")
    vertices_canonical = _transform_gaussian_positions(
        vertices_tensor, from_transl, from_scale, from_global_orient,
        to_canonical=True, pelvis_joint_j0=from_pelvis_j0
    )


    logger.debug("Transforming mesh from canonical space to target world space...")
    vertices_target_world = _transform_gaussian_positions(
        vertices_canonical, to_transl, to_scale, to_global_orient,
        to_canonical=False, pelvis_joint_j0=to_pelvis_j0
    )


    transformed_mesh = mesh.copy()
    transformed_mesh.vertices = vertices_target_world.cpu().numpy()

    logger.success(f"Successfully transformed mesh vertices from source to target coordinate system")
    return transformed_mesh


def extract_garment_mesh(
    labeled_ply_path: Path, label_pkl_path: Path, device: torch.device
):

    if not labeled_ply_path.exists():
        logger.warning(
            f"Labeled PLY not found: {labeled_ply_path}. Skipping garment extraction."
        )
        return None, None
    if not label_pkl_path.exists():
        logger.warning(
            f"Label PKL not found: {label_pkl_path}. Skipping garment extraction."
        )
        return None, None

    logger.info(f"Loading labeled PLY: {labeled_ply_path} and PKL: {label_pkl_path}")


    try:


        scan_mesh = trimesh.load_mesh(str(labeled_ply_path))
        verts_np = np.array(scan_mesh.vertices)
        faces_np = np.array(scan_mesh.faces)

    except Exception as e:
        logger.error(f"Error loading PLY geometry with trimesh: {e}")
        return None, None


    try:
        with open(label_pkl_path, "rb") as f:
            label_data = pickle.load(f)

        if "scan_labels" not in label_data:
            logger.error(f"'scan_labels' key not found in {label_pkl_path}.")
            return None, None

        labels = label_data["scan_labels"]

        if labels.shape[0] != verts_np.shape[0]:
            logger.error(
                f"Mismatch in vertex count between PLY ({verts_np.shape[0]}) and labels PKL ({labels.shape[0]}) in {label_pkl_path}."
            )
            return None, None

    except Exception as e:
        logger.error(f"Error loading labels from PKL: {e}")
        return None, None


    garment_labels_indices = [2, 3, 4, 5]


    is_garment_vertex = np.isin(labels, garment_labels_indices)


    is_garment_face_mask = is_garment_vertex[faces_np].any(axis=1)


    garment_faces_original_indices = faces_np[is_garment_face_mask]

    if len(garment_faces_original_indices) == 0:
        logger.warning("No garment faces found based on labels.")
        return None, None


    unique_vertex_indices_in_garment_faces = np.unique(
        garment_faces_original_indices.flatten()
    )


    garment_verts_np = verts_np[unique_vertex_indices_in_garment_faces]


    garment_vertex_labels = labels[unique_vertex_indices_in_garment_faces]

    garment_verts_colors_np = SURFACE_LABEL_COLOR_UINT8[garment_vertex_labels]


    old_to_new_vertex_map = {
        old_idx: new_idx
        for new_idx, old_idx in enumerate(unique_vertex_indices_in_garment_faces)
    }


    garment_faces_reindexed_np = np.vectorize(old_to_new_vertex_map.get)(
        garment_faces_original_indices
    )


    garment_trimesh = trimesh.Trimesh(
        vertices=garment_verts_np,
        faces=garment_faces_reindexed_np,
        vertex_colors=garment_verts_colors_np,
    )

    logger.success(
        f"Extracted garment mesh with {garment_trimesh.vertices.shape[0]} vertices, {garment_trimesh.faces.shape[0]} faces, and {garment_trimesh.visual.vertex_colors.shape[0] if garment_trimesh.visual.vertex_colors is not None else 0} colors."
    )

    return (
        garment_trimesh,
        garment_verts_colors_np,
    )


def save_mesh(mesh: trimesh.Trimesh, path: Path, verts_colors: np.ndarray = None):

    if mesh is None:
        logger.warning(f"No mesh to save at {path}")
        return

    logger.info(f"Saving mesh to {path}")
    try:

        if verts_colors is not None:

            if verts_colors.dtype != np.uint8:
                logger.warning(f"Vertex colors for {path} are not uint8. Converting.")
                if verts_colors.max() <= 1.0:
                    verts_colors = (verts_colors * 255).astype(np.uint8)
                else:
                    verts_colors = verts_colors.astype(np.uint8)

            if verts_colors.shape[0] == mesh.vertices.shape[0]:

                if verts_colors.ndim == 2 and (
                    verts_colors.shape[1] == 3 or verts_colors.shape[1] == 4
                ):
                    mesh.vertex_colors = verts_colors
                else:
                    logger.warning(
                        f"verts_colors for {path} have shape {verts_colors.shape}, expected (N,3) or (N,4). Not applying colors."
                    )
            else:
                logger.warning(
                    f"Mismatch between number of vertices ({mesh.vertices.shape[0]}) and number of vertex colors ({verts_colors.shape[0]}) for {path}. Not applying colors."
                )

        file_type = path.suffix.lower()[1:]

        if file_type == "ply":

            mesh.export(str(path), file_type="ply")
            logger.success(
                f"Mesh saved successfully as PLY {'with colors' if mesh.visual.vertex_colors is not None and mesh.visual.vertex_colors.any() else 'without colors'} using trimesh."
            )
        elif file_type == "obj":
            if (
                mesh.visual.vertex_colors is not None
                and mesh.visual.vertex_colors.any()
            ):
                logger.warning(
                    f"Saving {path} as OBJ with vertex colors. OBJ format has limited support for vertex colors. Trimesh will attempt to save them."
                )
            mesh.export(str(path), file_type="obj")
            logger.success(
                f"Mesh saved successfully as OBJ {'(vertex colors attempted)' if mesh.visual.vertex_colors is not None and mesh.visual.vertex_colors.any() else ''} using trimesh."
            )
        else:
            logger.warning(
                f"Unsupported file extension {path.suffix} for {path}. Attempting to save as PLY."
            )
            mesh.export(str(path.with_suffix(".ply")), file_type="ply")
            logger.info("Mesh saved as .ply instead.")

    except Exception as e:
        logger.error(f"Error saving mesh {path}: {e}")


def process_labeled_ply_mesh(
    data_dir: Path,
    file_frame_id_numeric_part: str,
    args: argparse.Namespace,
    outfix_suffix: str = "",
    seg_method_suffix: str = "",
):


    zero_padded_frame_id = f"{0:0{args.frame_format_digits}d}"
    labled_ply_path_original = (
        data_dir
        / "Semantic"
        / "process"
        / f"labels_auto{outfix_suffix}"
        / f"vis-labeled-mesh-f{zero_padded_frame_id}{seg_method_suffix}.ply"
    )
    label_pkl_path_original = (
        data_dir
        / "Semantic"
        / "process"
        / f"labels_auto{outfix_suffix}"
        / f"label-f{zero_padded_frame_id}{seg_method_suffix}.pkl"
    )


    if args.save_label_with_no_sam_suffix:
        labeled_dir = data_dir / "mesh" / "labeled_no_sam"
    else:
        labeled_dir = data_dir / "mesh" / "labeled"


    labeled_ply_path = (
        labeled_dir
        / f"vis-labeled-mesh-f{zero_padded_frame_id}{outfix_suffix}.ply"
    )
    label_pkl_path = (
        labeled_dir
        / f"label-f{zero_padded_frame_id}{outfix_suffix}.pkl"
    )


    if labled_ply_path_original.exists() and label_pkl_path_original.exists():
        try:
            import shutil

            labeled_ply_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(labled_ply_path_original), str(labeled_ply_path))
            shutil.copy2(str(label_pkl_path_original), str(label_pkl_path))
            logger.success(
                f"Copied semantic files for {file_frame_id_numeric_part} to {labeled_dir}"
            )
        except Exception as e:
            logger.error(
                f"Error copying labeled PLY and label pickle files for {file_frame_id_numeric_part}: {e}. Garment processing may fail."
            )
    else:
        logger.warning(
            f"Original semantic files not found for {file_frame_id_numeric_part}:"
        )
        logger.warning(f"  PLY expected at: {labled_ply_path_original}")
        logger.warning(f"  PKL expected at: {label_pkl_path_original}")
        logger.warning(f"Skipping garment extraction for this frame.")

    return labeled_ply_path, label_pkl_path


def process_simplified_mesh_with_skin_mask(
    data_dir: Path,
    frame_output_dir: Path,
    label_pkl_ext_path: Path,
    args,
    file_frame_id_with_suffix: str,
) -> None:

    logger.info("Extracting non-head NeuS2 mesh with simplification and cleanup...")


    if not label_pkl_ext_path.exists():
        error_msg = f"Segmentation files required: {label_pkl_ext_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:


        if args.generate_cloth_fit_skin_mask:

            zero_padded_frame_id = f"{0:0{args.frame_format_digits}d}"
            label_pkl_no_sam_path = data_dir / "Semantic" / "process" / "labels_auto_extended" / f"label-f{zero_padded_frame_id}_no_sam.pkl"

            if not label_pkl_no_sam_path.exists():
                error_msg = (
                    f"generate_cloth_fit_skin_mask requires _no_sam labels for accurate segmentation, but file not found:\n"
                    f"  Expected: {label_pkl_no_sam_path}\n"
                    f"  Ensure parse_mesh stage was run with Extended outfit and no_sam labels generated."
                )
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)

            logger.info(f"Forcing use of _no_sam labels for skin mask: {label_pkl_no_sam_path.name}")
            seg_labels_path_to_use = label_pkl_no_sam_path
        else:
            seg_labels_path_to_use = label_pkl_ext_path

        with open(seg_labels_path_to_use, "rb") as f:
            label_data = pickle.load(f)

        if "scan_labels" not in label_data:
            raise KeyError("'scan_labels' not found in segmentation data")

        seg_labels = label_data["scan_labels"]
        logger.info(f"Loaded segmentation labels: {seg_labels.shape}")


        zero_padded_frame_id = f"{0:0{args.frame_format_digits}d}"
        nerf_mesh_path = data_dir / "mesh" / "trimesh_cleaned" / f"{zero_padded_frame_id}.obj"

        if not nerf_mesh_path.exists():
            raise FileNotFoundError(f"NeuS2 cleaned mesh not found: {nerf_mesh_path}")


        logger.info(f"Loading NeuS2 mesh from: {nerf_mesh_path}")
        nerf_mesh = trimesh.load_mesh(str(nerf_mesh_path))
        logger.info(f"Loaded NeuS2 mesh: {len(nerf_mesh.vertices)} vertices, {len(nerf_mesh.faces)} faces")

        remove_head = args.remove_head
        if remove_head:
            logger.info("Head removal ENABLED - extracting non-head region")
            head_vert_indices, head_face_indices = identify_head_region(
                nerf_mesh.vertices, nerf_mesh.faces, seg_labels,
                DETAILED_SURFACE_LABELS, hair_label_name="hair",
                use_direct_head_label=True, direct_head_label_name="head",
                smpl_mesh_vertices=None, smplx_vert_seg=None,
                fallback_skin_label_name="skin"
            )

            if not head_vert_indices.size:
                raise ValueError("No head vertices found in NeuS2 mesh")

            logger.info(f"Identified {len(head_vert_indices)} head vertices")
            all_vert_indices = np.arange(len(nerf_mesh.vertices))
            nonhead_vert_indices = np.setdiff1d(all_vert_indices, head_vert_indices)
            nonhead_faces_mask = np.all(np.isin(nerf_mesh.faces, nonhead_vert_indices), axis=1)
            mesh_to_process = nerf_mesh.submesh([nonhead_faces_mask], append=True)
            logger.info(f"Created non-head submesh: {len(mesh_to_process.vertices)} vertices")
        else:
            logger.info("Head removal DISABLED - processing full mesh (head + body)")
            mesh_to_process = nerf_mesh
            nonhead_vert_indices = None

        base_name = "nerf_nonhead" if remove_head else "nerf"


        original_output_path = frame_output_dir / f"{base_name}_original.obj"
        save_mesh(mesh_to_process, original_output_path)
        logger.success(f"Saved original mesh: {original_output_path}")


        logger.info("Creating PyMeshLab MeshSet for simplification + cleanup pipeline...")
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=mesh_to_process.vertices,
            face_matrix=mesh_to_process.faces
        ))

        initial_verts = ms.current_mesh().vertex_number()
        initial_faces = ms.current_mesh().face_number()
        logger.info(f"PyMeshLab mesh: {initial_verts} vertices, {initial_faces} faces")


        skin_mask_stored = False
        hands_mask_stored = False
        feet_mask_stored = False
        manual_feet_idx_preclean = np.array([], dtype=np.int64)
        hands_gate_accept = False
        left_ok = False
        right_ok = False
        semantic_chest_stored = False
        semantic_hip_stored = False
        if args.generate_cloth_fit_skin_mask:
            logger.info("Generating binary skin mask for cloth-fit...")


            if remove_head:
                logger.info("Head removal was applied - subsetting seg_labels to match submesh vertices")
                seg_labels_subset = seg_labels[nonhead_vert_indices]
                logger.info(f"Subset seg_labels: {len(seg_labels)} → {len(seg_labels_subset)}")
            else:
                logger.info("No head removal - using full seg_labels")
                seg_labels_subset = seg_labels


            if len(seg_labels_subset) != len(mesh_to_process.vertices):
                raise ValueError(
                    f"seg_labels length ({len(seg_labels_subset)}) doesn't match "
                    f"mesh vertices ({len(mesh_to_process.vertices)})"
                )


            try:
                skin_mask_binary = identify_skin_mask_binary(
                    seg_labels_subset,
                    DETAILED_SURFACE_LABELS,
                    no_torso_skin=args.no_torso_skin_for_skin_mask,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to build skin_mask (no_torso={args.no_torso_skin_for_skin_mask}): {e}") from e

            try:
                skin_mask_full_binary = identify_skin_mask_binary(
                    seg_labels_subset,
                    DETAILED_SURFACE_LABELS,
                    no_torso_skin=False,
                )
            except Exception:

                skin_mask_full_binary = skin_mask_binary


            skin_mask_stored = False


            hands_mask_binary = np.zeros(len(seg_labels_subset), dtype=np.float64)
            try:
                hands_label_idx = DETAILED_SURFACE_LABELS.index("hands")
                hands_verts = np.where(seg_labels_subset == hands_label_idx)[0]
                hands_mask_binary[hands_verts] = 1.0
                logger.info(f"  hands: {len(hands_verts)} vertices")
                ms.current_mesh().add_vertex_custom_scalar_attribute(
                    hands_mask_binary.astype(np.float64), "hands_mask"
                )
                hands_mask_stored = True
            except ValueError:
                logger.warning("Label 'hands' not found in DETAILED_SURFACE_LABELS; hands indices will be empty.")
                hands_mask_stored = False


            feet_mask_binary = np.zeros(len(seg_labels_subset), dtype=np.float64)
            shoes_verts = np.array([], dtype=np.int64)
            try:
                shoes_label_idx = DETAILED_SURFACE_LABELS.index("shoes")
                shoes_verts = np.where(seg_labels_subset == shoes_label_idx)[0]
                feet_mask_binary[shoes_verts] = 1.0
                logger.info(f"  shoes(feet): {len(shoes_verts)} vertices")
            except ValueError:
                logger.warning(
                    "Label 'shoes' not found in DETAILED_SURFACE_LABELS; "
                    "auto shoes(feet) mask will be empty."
                )


            manual_preclean_path = frame_output_dir / f"{base_name}_original_feet_indices_manual.txt"
            if manual_preclean_path.exists():
                try:
                    if remove_head:
                        logger.warning(
                            "Manual pre-clean feet indices file found while remove_head=True. "
                            "IMPORTANT: indices must refer to the SAME topology as the mesh being processed "
                            f"({base_name}_original.obj), which is a NON-HEAD submesh. If you exported indices "
                            "from a full-head labeled mesh/PLY, they will NOT match and will be ignored as out-of-range "
                            "or (worse) mark wrong vertices."
                        )
                    manual_raw = np.loadtxt(manual_preclean_path)
                    manual_idx = (
                        np.array([], dtype=np.int64)
                        if manual_raw.size == 0
                        else np.atleast_1d(manual_raw).astype(np.int64)
                    )
                    n_pre = int(len(seg_labels_subset))
                    valid = (manual_idx >= 0) & (manual_idx < n_pre)
                    n_invalid = int(np.sum(~valid)) if manual_idx.size > 0 else 0
                    if n_invalid > 0:
                        logger.warning(
                            f"Manual pre-clean feet indices contain {n_invalid} out-of-range entries "
                            f"(valid range: [0, {n_pre - 1}]). They will be ignored."
                        )
                    manual_idx = manual_idx[valid]
                    manual_feet_idx_preclean = manual_idx.astype(np.int64)
                    if manual_idx.size > 0:
                        feet_mask_binary[manual_idx] = 1.0
                    logger.success(
                        f"Loaded manual pre-clean feet indices: {manual_preclean_path} "
                        f"(manual_valid={int(manual_idx.size)})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load manual pre-clean feet indices from {manual_preclean_path}: {e}. "
                        "Ignoring pre-clean manual indices."
                    )
            else:
                logger.info(f"Manual pre-clean feet indices file not found: {manual_preclean_path}")


            if manual_feet_idx_preclean.size > 0:
                if remove_head:
                    logger.warning(
                        "Skipping _manual labeled PLY export because remove_head=True (topology mismatch risk). "
                        "If you need this, export indices from the non-head mesh and implement a dedicated non-head "
                        "labeled export path."
                    )
                else:
                    try:


                        prefer_no_sam_dir = bool(getattr(args, "save_label_with_no_sam_suffix", False))
                        labeled_candidates = []
                        if prefer_no_sam_dir:
                            labeled_candidates.append(data_dir / "mesh" / "labeled_no_sam")
                            labeled_candidates.append(data_dir / "mesh" / "labeled")
                        else:
                            labeled_candidates.append(data_dir / "mesh" / "labeled")
                            labeled_candidates.append(data_dir / "mesh" / "labeled_no_sam")

                        src_ply = None
                        out_dir = None
                        for cand in labeled_candidates:
                            cand_src = cand / f"vis-labeled-mesh-f{zero_padded_frame_id}_extended.ply"
                            if cand_src.exists():
                                src_ply = cand_src
                                out_dir = cand
                                break

                        if src_ply is None or out_dir is None:
                            logger.warning(
                                "Manual labeled PLY export skipped (source labeled PLY not found). "
                                f"Tried: {[str(p / f'vis-labeled-mesh-f{zero_padded_frame_id}_extended.ply') for p in labeled_candidates]}"
                            )
                        else:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            dst_ply = out_dir / f"vis-labeled-mesh-f{zero_padded_frame_id}_extended_manual.ply"

                            src_mesh = trimesh.load_mesh(str(src_ply))
                            vc = None
                            if hasattr(src_mesh, "visual") and hasattr(src_mesh.visual, "vertex_colors"):
                                vc = src_mesh.visual.vertex_colors


                            n_v = int(len(src_mesh.vertices))
                            if vc is None or len(vc) != n_v:
                                vc_rgba = np.zeros((n_v, 4), dtype=np.uint8)
                                vc_rgba[:, 3] = 255
                            else:
                                vc_np = np.asarray(vc)
                                if vc_np.dtype != np.uint8:
                                    vc_np = vc_np.astype(np.uint8)
                                if vc_np.ndim != 2 or vc_np.shape[0] != n_v:
                                    vc_rgba = np.zeros((n_v, 4), dtype=np.uint8)
                                    vc_rgba[:, 3] = 255
                                elif vc_np.shape[1] == 3:
                                    vc_rgba = np.concatenate([vc_np, 255 * np.ones((n_v, 1), dtype=np.uint8)], axis=1)
                                else:
                                    vc_rgba = vc_np[:, :4].copy()


                            shoe_fallback_rgba = np.array([128, 0, 255, 255], dtype=np.uint8)
                            try:
                                shoes_label_idx = DETAILED_SURFACE_LABELS.index("shoes")
                                auto_shoes = np.where(seg_labels_subset == shoes_label_idx)[0]
                            except Exception:
                                auto_shoes = np.array([], dtype=np.int64)

                            if auto_shoes.size > 0 and auto_shoes.max(initial=0) < n_v:
                                shoe_color = np.mean(vc_rgba[auto_shoes].astype(np.float32), axis=0)
                                shoe_color_rgba = np.clip(np.round(shoe_color), 0, 255).astype(np.uint8)
                            else:
                                shoe_color_rgba = shoe_fallback_rgba


                            vc_rgba[manual_feet_idx_preclean] = shoe_color_rgba
                            src_mesh.visual.vertex_colors = vc_rgba
                            src_mesh.export(str(dst_ply))
                            logger.success(f"Saved manual-labeled mesh visualization: {dst_ply}")
                    except Exception as e:
                        logger.warning(f"Failed to export _manual labeled PLY: {e}")


            if float(np.max(feet_mask_binary)) > 0.0:
                try:
                    n_auto = int(shoes_verts.size) if shoes_verts is not None else 0
                except Exception:
                    n_auto = 0
                n_manual = int(manual_feet_idx_preclean.size)
                n_union_pre = int(np.sum(feet_mask_binary > 0.5))
                logger.info(
                    "Feet mask (pre-simplification) built as union of auto+manual: "
                    f"auto_shoes={n_auto}, manual_preclean={n_manual}, union={n_union_pre}. "
                    "This `feet_mask` attribute will be interpolated through PyMeshLab decimation/cleanup."
                )
                ms.current_mesh().add_vertex_custom_scalar_attribute(
                    feet_mask_binary.astype(np.float64), "feet_mask"
                )
                feet_mask_stored = True
            else:
                feet_mask_stored = False


            if bool(getattr(args, "generate_semantic_weighting_masks", False)):
                lbs_path = frame_output_dir / "smoothed_inpainted_weights.npy"
                if not lbs_path.exists():
                    logger.warning(
                        f"Semantic weighting enabled but LBS file not found: {lbs_path}. Skipping semantic masks."
                    )
                else:
                    try:
                        w_full = np.load(str(lbs_path))
                        if w_full.ndim != 2:
                            raise ValueError(f"Expected 2D LBS matrix, got shape {w_full.shape}")
                        if int(w_full.shape[0]) != int(len(nerf_mesh.vertices)):
                            raise ValueError(
                                f"LBS vertex count mismatch: lbs={int(w_full.shape[0])}, nerf_mesh={int(len(nerf_mesh.vertices))}"
                            )
                        w = w_full if not remove_head else w_full[nonhead_vert_indices]
                        if int(w.shape[0]) != int(len(mesh_to_process.vertices)):
                            raise ValueError(
                                f"Subset LBS vertex count mismatch: lbs_subset={int(w.shape[0])}, mesh_to_process={int(len(mesh_to_process.vertices))}"
                            )

                        joint_names = _load_joint_names_for_lbs(
                            frame_output_dir=frame_output_dir, n_joints=int(w.shape[1])
                        )
                        joint_to_idx = {name: i for i, name in enumerate(joint_names)}

                        chest_joints = ["spine3", "left_collar", "right_collar"]
                        hip_joints = ["pelvis", "left_hip", "right_hip"]
                        missing = [j for j in (chest_joints + hip_joints) if j not in joint_to_idx]
                        if missing:
                            raise KeyError(f"Missing joint names for semantic weighting: {missing}")

                        chest_idx = np.array([joint_to_idx[j] for j in chest_joints], dtype=np.int64)
                        hip_idx = np.array([joint_to_idx[j] for j in hip_joints], dtype=np.int64)

                        chest_score = np.sum(w[:, chest_idx], axis=1)
                        hip_score = np.sum(w[:, hip_idx], axis=1)

                        semantic_mode = str(getattr(args, "semantic_mode", "sum_threshold") or "sum_threshold")
                        if semantic_mode not in ("sum_threshold", "argmax"):
                            raise ValueError(f"Invalid semantic_mode={semantic_mode!r}")

                        if semantic_mode == "argmax":


                            w_argmax = np.argmax(w, axis=1)
                            w_max = np.max(w, axis=1)
                            chest_thr = float(getattr(args, "semantic_chest_threshold", 0.30))
                            hip_thr = float(getattr(args, "semantic_hip_threshold", 0.30))
                            chest_mask = np.isin(w_argmax, chest_idx) & (w_max > chest_thr)
                            hip_mask = np.isin(w_argmax, hip_idx) & (w_max > hip_thr)
                        else:

                            chest_mask = chest_score > float(getattr(args, "semantic_chest_threshold", 0.30))
                            hip_mask = hip_score > float(getattr(args, "semantic_hip_threshold", 0.30))


                        hands_ex = hands_mask_binary > 0.5
                        feet_ex = feet_mask_binary > 0.5
                        chest_mask = chest_mask & (~hands_ex) & (~feet_ex)
                        hip_mask = hip_mask & (~hands_ex) & (~feet_ex)


                        min_comp = int(getattr(args, "semantic_min_component_size", 100))
                        rings = int(getattr(args, "semantic_dilate_rings", 1))
                        chest_mask = _mask_remove_small_components(
                            chest_mask, mesh=mesh_to_process, min_size=min_comp
                        )
                        hip_mask = _mask_remove_small_components(
                            hip_mask, mesh=mesh_to_process, min_size=min_comp
                        )
                        chest_mask = _mask_dilate_vertex_rings(chest_mask, mesh=mesh_to_process, rings=rings)
                        hip_mask = _mask_dilate_vertex_rings(hip_mask, mesh=mesh_to_process, rings=rings)


                        chest_mask = chest_mask & (~hands_ex) & (~feet_ex)
                        hip_mask = hip_mask & (~hands_ex) & (~feet_ex)

                        ms.current_mesh().add_vertex_custom_scalar_attribute(
                            chest_mask.astype(np.float64), "semantic_chest_mask"
                        )
                        ms.current_mesh().add_vertex_custom_scalar_attribute(
                            hip_mask.astype(np.float64), "semantic_hip_mask"
                        )
                        semantic_chest_stored = True
                        semantic_hip_stored = True
                        logger.info(
                            "Stored semantic masks as vertex attributes (semantic_chest_mask, semantic_hip_mask): "
                            f"mode={semantic_mode}, chest={int(np.sum(chest_mask))}, hip={int(np.sum(hip_mask))}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to generate semantic masks from LBS weights: {e}")
                        semantic_chest_stored = False
                        semantic_hip_stored = False


            try:
                left_arm_idx = DETAILED_SURFACE_LABELS.index("left_arm")
                right_arm_idx = DETAILED_SURFACE_LABELS.index("right_arm")
                n_left_arm = int(np.sum(seg_labels_subset == left_arm_idx))
                n_right_arm = int(np.sum(seg_labels_subset == right_arm_idx))
                n_hands = (
                    int(np.sum(seg_labels_subset == hands_label_idx))
                    if "hands_label_idx" in locals()
                    else 0
                )
                min_arm_verts = int(args.hands_gate_min_arm_verts)
                min_ratio = float(args.hands_gate_min_arm_to_hands_ratio)


                hands_per_side = n_hands / 2.0

                left_ok = (n_left_arm >= min_arm_verts) and (
                    n_left_arm >= min_ratio * hands_per_side
                )
                right_ok = (n_right_arm >= min_arm_verts) and (
                    n_right_arm >= min_ratio * hands_per_side
                )
                hands_gate_accept = bool(left_ok or right_ok)

                logger.info(
                    "Hands gate stats: "
                    f"n_left_arm={n_left_arm}, n_right_arm={n_right_arm}, n_hands={n_hands}, hands_per_side≈{hands_per_side:.1f}, "
                    f"min_arm_verts={min_arm_verts}, min_arm_to_hands_ratio={min_ratio:.2f} "
                    f"-> left_ok={left_ok}, right_ok={right_ok}, accept={hands_gate_accept}"
                )
            except ValueError as e:
                logger.warning(f"Hands gate skipped due to missing label: {e}")
                hands_gate_accept = False
                left_ok = False
                right_ok = False


            try:
                left_arm_idx = DETAILED_SURFACE_LABELS.index("left_arm")
                right_arm_idx = DETAILED_SURFACE_LABELS.index("right_arm")
                if not left_ok:
                    skin_mask_binary[seg_labels_subset == left_arm_idx] = 0.0
                if not right_ok:
                    skin_mask_binary[seg_labels_subset == right_arm_idx] = 0.0

                if skin_mask_full_binary is not None:
                    if not left_ok:
                        skin_mask_full_binary[seg_labels_subset == left_arm_idx] = 0.0
                    if not right_ok:
                        skin_mask_full_binary[seg_labels_subset == right_arm_idx] = 0.0


                ms.current_mesh().add_vertex_custom_scalar_attribute(
                    skin_mask_binary.astype(np.float64), "skin_mask"
                )
                ms.current_mesh().add_vertex_custom_scalar_attribute(
                    (skin_mask_full_binary if skin_mask_full_binary is not None else skin_mask_binary).astype(np.float64),
                    "skin_mask_full",
                )
                skin_mask_stored = True
                logger.info(
                    "Stored skin masks as vertex attributes (skin_mask, skin_mask_full); preserved through operations"
                )
                logger.info(
                    "Applied arm gate to limb mask: "
                    f"keep_left_arm={left_ok}, keep_right_arm={right_ok}. "
                    "This prevents false arm labels near hands from being treated as limb_no_hands."
                )
            except Exception as e:
                logger.warning(f"Failed to apply arm gate to limb mask: {e}")


        create_simplified = args.create_simplified_mesh
        simp_target_percent = args.simplification_target_percent

        if create_simplified:
            logger.info(f"Applying decimation (target={simp_target_percent})...")
            verts_before = ms.current_mesh().vertex_number()
            faces_before = ms.current_mesh().face_number()

            ms.meshing_decimation_quadric_edge_collapse(targetperc=simp_target_percent)

            verts_after = ms.current_mesh().vertex_number()
            faces_after = ms.current_mesh().face_number()

            logger.info(
                f"Decimation: {verts_before}→{verts_after} vertices ({verts_after/verts_before:.1%}), "
                f"{faces_before}→{faces_after} faces ({faces_after/faces_before:.1%})"
            )
            base_name += "_simp"


        logger.info("Applying PyMeshLab cleanup (intersection removal, hole filling)...")


        logger.info("  Step 1: Computing selection by self-intersections...")
        ms.compute_selection_by_self_intersections_per_face()
        selected_faces_count = ms.current_mesh().selected_face_number()
        logger.info(f"  Found {selected_faces_count} self-intersecting faces")

        if selected_faces_count > 0:
            logger.info("  Step 1b: Removing selected self-intersecting faces...")
            ms.meshing_remove_selected_faces()


        logger.info("  Step 2: Removing unreferenced vertices...")
        verts_before_cleanup = ms.current_mesh().vertex_number()
        ms.meshing_remove_unreferenced_vertices()
        verts_after_cleanup = ms.current_mesh().vertex_number()
        unreferenced_count = verts_before_cleanup - verts_after_cleanup
        if unreferenced_count > 0:
            logger.info(f"  Removed {unreferenced_count} unreferenced vertices")


        logger.info("  Step 3: Repairing non-manifold edges...")
        ms.meshing_repair_non_manifold_edges()


        logger.info("  Step 4: Closing holes (maxholesize=30)...")
        faces_before_hole = ms.current_mesh().face_number()
        ms.meshing_close_holes(maxholesize=30)
        faces_after_hole = ms.current_mesh().face_number()
        faces_added = faces_after_hole - faces_before_hole
        if faces_added > 0:
            logger.info(f"  Hole closing added {faces_added} faces")

        final_verts = ms.current_mesh().vertex_number()
        final_faces = ms.current_mesh().face_number()
        logger.info(f"Cleanup complete: {final_verts} vertices, {final_faces} faces")


        if args.use_nerf_mesh_smoothing:
            logger.info("Applying taubin smoothing...")
            ms.apply_coord_taubin_smoothing(stepsmoothnum=8, lambda_=0.33, mu=-0.34)


        logger.info("  Step 5: Computing selection by self-intersections...")
        ms.compute_selection_by_self_intersections_per_face()
        selected_faces_count = ms.current_mesh().selected_face_number()
        logger.info(f"  Found {selected_faces_count} self-intersecting faces")

        if selected_faces_count > 0:
            logger.info("  Step 5b: Removing selected self-intersecting faces...")
            ms.meshing_remove_selected_faces()


        cleaned_vertices = ms.current_mesh().vertex_matrix()
        cleaned_faces = ms.current_mesh().face_matrix()

        if len(cleaned_vertices) == 0 or len(cleaned_faces) == 0:
            raise RuntimeError("Cleanup resulted in empty mesh")

        cleaned_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces)

        base_name += "_cleaned"
        cleaned_output_path = frame_output_dir / f"{base_name}.obj"
        save_mesh(cleaned_mesh, cleaned_output_path)
        logger.success(f"Saved cleaned NeuS2 mesh: {cleaned_output_path}")


        if skin_mask_stored:
            logger.info("Retrieving interpolated skin mask...")
            skin_mask_interpolated = ms.current_mesh().vertex_custom_scalar_attribute_array("skin_mask")
            skin_vertex_indices = extract_skin_indices_from_mask(
                skin_mask_interpolated, threshold=0.5, allow_empty=True
            )


            if skin_vertex_indices.size == 0:
                try:
                    skin_mask_full_interpolated = ms.current_mesh().vertex_custom_scalar_attribute_array(
                        "skin_mask_full"
                    )
                    _ = extract_skin_indices_from_mask(
                        skin_mask_full_interpolated, threshold=0.5, allow_empty=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to extract skin_mask_full after interpolation: {e}")

            skin_mask_suffix = "_no_torso_skin" if args.no_torso_skin_for_skin_mask else ""
            skin_mask_filename = f"{base_name}_skin_indices{skin_mask_suffix}.txt"
            skin_mask_path = frame_output_dir / skin_mask_filename
            np.savetxt(skin_mask_path, skin_vertex_indices.astype(np.int64), fmt="%d")
            logger.success(f"Saved skin vertex indices: {skin_mask_path}")
            logger.info(
                f"  Total skin vertices (mask='{skin_mask_suffix or 'full'}'): {int(skin_vertex_indices.size)}"
            )


        hands_mask_filename = f"{base_name}_hands_indices.txt"
        hands_mask_path = frame_output_dir / hands_mask_filename
        hands_vertex_indices = np.array([], dtype=np.int64)
        if hands_mask_stored:
            logger.info("Retrieving interpolated hands mask...")
            hands_mask_interpolated = ms.current_mesh().vertex_custom_scalar_attribute_array("hands_mask")
            try:
                hands_vertex_indices = extract_skin_indices_from_mask(hands_mask_interpolated, threshold=0.5)
            except Exception as e:
                logger.warning(
                    f"Failed to extract hands indices after interpolation: {e}. Writing empty hands indices."
                )
                hands_vertex_indices = np.array([], dtype=np.int64)
        else:
            logger.info("Hands mask not stored; writing empty hands indices.")
        np.savetxt(hands_mask_path, hands_vertex_indices.astype(np.int64), fmt="%d")
        logger.success(f"Saved hands vertex indices: {hands_mask_path}")
        logger.info(f"  Total hands vertices: {int(hands_vertex_indices.size)}")


        feet_mask_filename = f"{base_name}_feet_indices.txt"
        feet_mask_path = frame_output_dir / feet_mask_filename
        feet_vertex_indices = np.array([], dtype=np.int64)
        if feet_mask_stored:
            logger.info("Retrieving interpolated feet mask...")
            feet_mask_interpolated = ms.current_mesh().vertex_custom_scalar_attribute_array("feet_mask")
            try:
                feet_vertex_indices = extract_skin_indices_from_mask(feet_mask_interpolated, threshold=0.5)
            except Exception as e:
                logger.warning(
                    f"Failed to extract feet indices after interpolation: {e}. Writing empty feet indices."
                )
                feet_vertex_indices = np.array([], dtype=np.int64)
        else:
            logger.info("Feet mask not stored; writing empty feet indices.")
        np.savetxt(feet_mask_path, feet_vertex_indices.astype(np.int64), fmt="%d")
        logger.success(f"Saved feet vertex indices: {feet_mask_path}")
        logger.info(f"  Total feet vertices: {int(feet_vertex_indices.size)}")


        if bool(getattr(args, "generate_semantic_weighting_masks", False)):
            garment_mesh_path = Path(cleaned_output_path)

            if semantic_chest_stored:
                logger.info("Retrieving interpolated semantic chest mask...")
                try:
                    chest_mask_interp = ms.current_mesh().vertex_custom_scalar_attribute_array(
                        "semantic_chest_mask"
                    )
                    chest_idx = extract_skin_indices_from_mask(
                        chest_mask_interp, threshold=0.5, allow_empty=True
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract semantic chest indices after interpolation: {e}. Writing empty indices."
                    )
                    chest_idx = np.array([], dtype=np.int64)
            else:
                chest_idx = np.array([], dtype=np.int64)

            if semantic_hip_stored:
                logger.info("Retrieving interpolated semantic hip mask...")
                try:
                    hip_mask_interp = ms.current_mesh().vertex_custom_scalar_attribute_array("semantic_hip_mask")
                    hip_idx = extract_skin_indices_from_mask(
                        hip_mask_interp, threshold=0.5, allow_empty=True
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract semantic hip indices after interpolation: {e}. Writing empty indices."
                    )
                    hip_idx = np.array([], dtype=np.int64)
            else:
                hip_idx = np.array([], dtype=np.int64)

            chest_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_semantic_chest_indices.txt"
            hip_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_semantic_hip_indices.txt"
            np.savetxt(chest_path, chest_idx.astype(np.int64), fmt="%d")
            np.savetxt(hip_path, hip_idx.astype(np.int64), fmt="%d")
            logger.success(f"Saved semantic chest indices: {chest_path} ({int(chest_idx.size)} verts)")
            logger.success(f"Saved semantic hip indices: {hip_path} ({int(hip_idx.size)} verts)")

            if bool(getattr(args, "semantic_debug_ply", False)):
                try:
                    n_v = int(cleaned_vertices.shape[0])
                    colors = np.zeros((n_v, 4), dtype=np.uint8)
                    colors[:, 3] = 255
                    if chest_idx.size > 0:
                        colors[chest_idx.astype(np.int64)] = np.array([255, 64, 64, 255], dtype=np.uint8)
                    if hip_idx.size > 0:

                        colors[hip_idx.astype(np.int64), 1] = 255
                        colors[hip_idx.astype(np.int64), 0] = np.maximum(
                            colors[hip_idx.astype(np.int64), 0], 64
                        )
                    vis_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces, process=False)
                    vis_mesh.visual.vertex_colors = colors
                    vis_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_semantic_vis.ply"
                    vis_mesh.export(str(vis_path))
                    logger.success(f"Saved semantic debug PLY: {vis_path}")
                except Exception as e:
                    logger.warning(f"Failed to write semantic debug PLY: {e}")

    except Exception as e:
        error_msg = f"Failed to process NeuS2 mesh for {file_frame_id_with_suffix}: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def main():
    parser = argparse.ArgumentParser(description="Process AvatarREX meshes.")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the AvatarREX subject data directory (e.g., data/thuman2_avatarrex/0365).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the processed meshes.",
    )
    parser.add_argument(
        "--frame_start",
        type=int,
        default=0,
        help="Start index of the frame range (inclusive).",
    )
    parser.add_argument(
        "--frame_end",
        type=int,
        default=0,
        help="End index of the frame range (inclusive).",
    )
    parser.add_argument(
        "--frame_step", type=int, default=1, help="Step for the frame range."
    )
    parser.add_argument(
        "--frame_list",
        nargs="+",
        type=str,
        default=None,
        help="Explicit list of frame IDs (e.g., 00000000 00000010). Overrides frame_start/end/step.",
    )
    parser.add_argument(
        "--frame_format_digits",
        type=int,
        default=8,
        help="Number of digits in frame ID strings (e.g., 8 for '00000000'). Used for 'avatarrex' and 'generic' types when not using frame_list.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix after frame ID for special input mesh processed with meshlab instead of trimesh.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["thuman2", "avatarrex", "generic", "actorshq", "mvhumannet", "talkbody4d"],
        default="generic",
        help="Type of the dataset being processed. Affects SMPL parameters and frame handling.",
    )
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available.")
    parser.add_argument(
        "--rectified_smpl_scale",
        type=float,
        default=1.0,
        help="Rectified SMPL scale to use for SMPL mesh generation.",
    )
    parser.add_argument(
        "--rectified_smpl_transl",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Rectified SMPL translation to use for SMPL mesh generation.",
    )
    parser.add_argument(
        "--gender",
        type=str,
        default="neutral",
        help="Gender to use for SMPL mesh generation (backward compatibility).",
    )
    parser.add_argument(
        "--gender_head",
        type=str,
        default=None,
        help="Gender for head donator SMPL model (swapped subjects only).",
    )
    parser.add_argument(
        "--gender_body",
        type=str,
        default=None,
        help="Gender for body donator SMPL model (swapped subjects only).",
    )
    parser.add_argument(
        "--use_body_reshaping_for_swapped",
        action="store_true",
        help="For swapped subjects: use head donator's body shape (betas) instead of body donator's shape when combining SMPL parameters.",
    )
    parser.add_argument(
        "--save_skeleton_mesh",
        action="store_true",
        help="Save SMPL skeleton as OBJ with joint connections"
    )
    parser.add_argument(
        "--save_smpl_joints",
        action="store_true",
        default=False,
        help="Save SMPL-X joints in world space next to smpl_body.obj (default: False)",
    )
    parser.add_argument(
        "--emit_smpl_only",
        action="store_true",
        default=False,
        help="Emit only smpl_body.obj (and optional smpl_joints.npy + smpl_joint_names.txt), skipping all other mesh artifacts.",
    )
    parser.add_argument(
        "--include_skeleton_fingers",
        action="store_true",
        help="Include finger joints in skeleton (default: False)"
    )
    parser.add_argument(
        "--exclude_foot_joints",
        action="store_true",
        default=True,
        help="Exclude problematic foot joints that intersect with mesh (default: True)"
    )
    parser.add_argument(
        "--simplified_skeleton",
        action="store_true",
        default=True,
        help="Use simplified 16-joint skeleton instead of 20-joint (default: True)"
    )
    parser.add_argument(
        "--full_skeleton",
        dest="simplified_skeleton",
        action="store_false",
        help="Use full 20-joint skeleton instead of simplified 16-joint"
    )
    parser.add_argument(
        "--no_simplified_skeleton",
        dest="simplified_skeleton",
        action="store_false",
        help="Use full 20-joint skeleton instead of simplified 16-joint"
    )


    parser.add_argument(
        "--remove_head",
        action="store_true",
        default=False,
        help="Remove head from NeuS2 mesh using 4D-Dress segmentation (default: False, BREAKING CHANGE from v1.x)"
    )
    parser.add_argument(
        "--no_remove_head",
        action="store_false",
        dest="remove_head",
        help="Keep full mesh including head (default behavior since v2.0)"
    )


    parser.add_argument(
        "--create_simplified_mesh",
        action="store_true",
        default=False,
        help="Create simplified version using decimation for cloth-fit reshaping (default: False)"
    )
    parser.add_argument(
        "--simplification_target_percent",
        type=float,
        default=0.3,
        help="Target decimation percentage (default: 0.4 = 40%% of original)"
    )
    parser.add_argument(
        "--use_no_sam_labels",
        action="store_true",
        default=False,
        help="Use no_sam labels for segmentation (default: False)"
    )
    parser.add_argument(
        "--save_label_with_no_sam_suffix",
        action="store_true",
        default=False,
        help="Save processed labels to mesh/labeled_no_sam/ directory instead of mesh/labeled/. "
             "Used in testing mode to preserve both SAM and no_sam label outputs (default: False)"
    )
    parser.add_argument(
        "--generate_cloth_fit_skin_mask",
        action="store_true",
        default=False,
        help="Generate skin vertex indices mask for cloth-fit fit_weight_masks feature (auto-loads _no_sam labels, default: False)"
    )
    parser.add_argument(
        "--generate_semantic_weighting_masks",
        action="store_true",
        default=False,
        help="Generate v1 semantic masks (chest/hip) from smoothed_inpainted_weights.npy and export semantic index files (default: False)",
    )
    parser.add_argument(
        "--semantic_mode",
        type=str,
        choices=["sum_threshold", "argmax"],
        default="sum_threshold",
        help=(
            "Semantic region assignment mode. "
            "'sum_threshold': sum of region joint weights > threshold. "
            "'argmax': argmax joint in region, gated by max-weight > threshold. "
            "(default: sum_threshold)"
        ),
    )
    parser.add_argument(
        "--semantic_chest_threshold",
        type=float,
        default=0.30,
        help="Threshold on summed LBS weights to select chest vertices (default: 0.30)",
    )
    parser.add_argument(
        "--semantic_hip_threshold",
        type=float,
        default=0.30,
        help="Threshold on summed LBS weights to select hip vertices (default: 0.30)",
    )
    parser.add_argument(
        "--semantic_dilate_rings",
        type=int,
        default=1,
        help="Vertex-neighborhood dilation rings for semantic masks (default: 1)",
    )
    parser.add_argument(
        "--semantic_min_component_size",
        type=int,
        default=100,
        help="Minimum connected-component size to keep in semantic masks (default: 100)",
    )
    parser.add_argument(
        "--semantic_debug_ply",
        action="store_true",
        default=False,
        help="Write a debug PLY coloring semantic masks on the cleaned mesh (default: False)",
    )
    parser.add_argument(
        "--no_torso_skin_for_skin_mask",
        action="store_true",
        default=False,
        help="Exclude torso skin (neck) for skin mask generation (default: False)"
    )
    parser.add_argument(
        "--hands_gate_min_arm_verts",
        type=int,
        default=300,
        help="Gate for writing hands indices: minimum verts in an arm label to consider the arm 'real' (default: 300)",
    )
    parser.add_argument(
        "--hands_gate_min_arm_to_hands_ratio",
        type=float,
        default=0.2,
        help="Gate for considering an arm label 'real': require n_arm >= ratio * (n_hands/2) (default: 0.2)",
    )
    parser.add_argument(
        "--use_nerf_mesh_smoothing",
        action="store_true",
        default=False,
        help="Use taubin smoothing for NeuS2 mesh (default: False)"
    )
    parser.add_argument(
        "--emit_smpl_scale_inv",
        action="store_true",
        default=False,
        help=(
            "For THuman2 only: emit smpl_scale_inv artifacts (smpl_scale_inv.json and smpl_scale_inv_4x4.npy) "
            "into the processed mesh output directory. These are used by scale-aware cloth-fit normalization "
            "(default: False)."
        ),
    )
    parser.add_argument(
        "--estimate_height_from_betas",
        action="store_true",
        default=False,
        help="Estimate canonical SMPL-X height from betas and write <data_dir>/height.txt in meters (default: False)",
    )

    args = parser.parse_args()


    if args.dataset_type == "thuman2":
        use_pca, num_pca_comps = True, 12
        flat_hand_mean = False

    elif args.dataset_type == "avatarrex":
        use_pca, num_pca_comps = False, 45
        flat_hand_mean = True
    elif args.dataset_type == "actorshq" or args.dataset_type == "mvhumannet":
        use_pca, num_pca_comps = True, 6
        flat_hand_mean = True
    elif args.dataset_type == "talkbody4d":

        use_pca, num_pca_comps = False, 45
        flat_hand_mean = True
    else:
        use_pca, num_pca_comps = (
            False,
            45,
        )
        flat_hand_mean = True

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    gender_head = args.gender_head if args.gender_head is not None else args.gender
    gender_body = args.gender_body if args.gender_body is not None else args.gender
    logger.info(f"Gender configuration: head={gender_head}, body={gender_body}")


    if args.dataset_type == "talkbody4d":
        if args.gender_head is None and str(gender_head).strip().lower() == "auto":
            gender_head = "neutral"
        if args.gender_body is None and str(gender_body).strip().lower() == "auto":
            gender_body = "neutral"
    subject_gender = _resolve_gender_auto(args.gender, data_dir)
    if args.gender_head is None:
        gender_head = _resolve_gender_auto(gender_head, data_dir)
    if args.gender_body is None:
        gender_body = _resolve_gender_auto(gender_body, data_dir)
    if args.dataset_type == "talkbody4d" and (gender_head != "neutral" or gender_body != "neutral"):
        logger.warning(
            f"TalkBody4D is typically fit with SMPLX_NEUTRAL; got head={gender_head}, body={gender_body}. "
            "If your meshes look different from the official visualization, use --gender neutral."
        )
    device = torch.device(
        "cuda" if args.use_gpu and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")


    subject_name = get_subject_name_from_path(data_dir)
    is_swapped = is_swapped_subject(subject_name)


    smpl_model = None
    smpl_model_head = None
    smpl_model_body = None
    smpl_lbs_weights = None

    try:
        if is_swapped:

            logger.info(f"Creating dual SMPL-X models for swapped subject: head={gender_head}, body={gender_body}")

            smpl_model_head = create_smplx_model_for_neus2_reposing(
                smpl_params=_maybe_load_smpl_params_for_model_init(data_dir),
                batch_size=1,
                device=device,
                gender=gender_head,
                dataset_type=args.dataset_type
            )

            smpl_model_body = create_smplx_model_for_neus2_reposing(
                smpl_params=_maybe_load_smpl_params_for_model_init(data_dir),
                batch_size=1,
                device=device,
                gender=gender_body,
                dataset_type=args.dataset_type
            )

            logger.success(f"Dual SMPL models created successfully: head ({gender_head}) and body ({gender_body})")


            if hasattr(smpl_model_body, "lbs_weights"):
                smpl_lbs_weights = smpl_model_body.lbs_weights.cpu().numpy()
                logger.debug(f"Accessed SMPL LBS weights from body model: {smpl_lbs_weights.shape}")
        else:

            logger.info(f"Creating single SMPL-X model for non-swapped subject: gender={gender_body}")

            smpl_model = create_smplx_model_for_neus2_reposing(
                smpl_params=_maybe_load_smpl_params_for_model_init(data_dir),
                batch_size=1,
                device=device,
                gender=gender_body,
                dataset_type=args.dataset_type
            )

            logger.success("SMPL model created successfully.")

            if hasattr(smpl_model, "lbs_weights"):
                smpl_lbs_weights = smpl_model.lbs_weights.cpu().numpy()
                logger.debug(f"Accessed SMPL LBS weights: {smpl_lbs_weights.shape}")

        if smpl_lbs_weights is None:
            logger.warning("Could not find 'lbs_weights' attribute on SMPL model(s). LBS weights will not be saved.")

    except Exception as e:
        logger.error(
            f"Error creating SMPL model(s): {e}. Make sure smplx_utils is correctly imported and configured, and model files are accessible. SMPL mesh and LBS weights generation skipped."
        )
        smpl_model = None
        smpl_model_head = None
        smpl_model_body = None
        smpl_lbs_weights = None


    subject_name = get_subject_name_from_path(data_dir)
    swapped_subject_params_loaded = False

    smpl_params_path = data_dir / "smpl_params.npz"
    smpl_data = None


    models_available = (smpl_model is not None) or (smpl_model_head is not None and smpl_model_body is not None)

    if models_available:

        if is_swapped_subject(subject_name):
            logger.info(f"Detected swapped subject: {subject_name}")


            head_subject_id, body_subject_id = parse_swapped_subject_name(subject_name)
            if head_subject_id is None or body_subject_id is None:
                logger.error(f"Failed to parse swapped subject name: {subject_name}")
            else:
                logger.info(f"Head donator: {head_subject_id}, Body donator: {body_subject_id}")


                data_path = Path(args.data_dir)
                if "head_swapped_renders" in str(data_path):


                    avatarrex_base_dir = str(data_path.parent.parent)
                    logger.info(f"Swapped subject detected. Using raw avatarrex directory: {avatarrex_base_dir}")
                else:

                    avatarrex_base_dir = str(data_path.parent) if args.dataset_type == "thuman2" else args.data_dir
                    logger.info(f"Non-swapped subject. Using avatarrex directory: {avatarrex_base_dir}")


                try:
                    head_params_path, body_params_path, combined_params_path = copy_smpl_params_for_swapped_subject(
                        data_dir, head_subject_id, body_subject_id, avatarrex_base_dir,
                        use_body_reshaping=args.use_body_reshaping_for_swapped
                    )


                    logger.info(f"Loading combined SMPL parameters from {combined_params_path}")
                    smpl_data = load_smpl_params(combined_params_path, device)
                    swapped_subject_params_loaded = True

                    logger.success(
                        f"Loaded swapped subject SMPL parameters for {smpl_data['body_pose'].shape[0] if smpl_data and 'body_pose' in smpl_data else 'N/A'} frames."
                    )

                except Exception as e:
                    logger.error(f"Error setting up SMPL parameters for swapped subject: {e}")
                    smpl_data = None


        if not swapped_subject_params_loaded:
            if not smpl_params_path.exists():
                logger.warning(
                    f"SMPL parameters file not found at {smpl_params_path}. SMPL mesh generation will be skipped."
                )

            else:
                logger.info(f"Loading SMPL parameters from {smpl_params_path}")
                try:
                    smpl_data_npz = np.load(smpl_params_path)
                    smpl_data = {
                        k: torch.tensor(v.astype(np.float32))
                        for k, v in smpl_data_npz.items()
                    }
                    logger.success(
                        f"Loaded SMPL parameters for {smpl_data['body_pose'].shape[0] if smpl_data and 'body_pose' in smpl_data else 'N/A'} frames."
                    )
                except Exception as e:
                    logger.error(
                        f"Error loading SMPL parameters: {e}. SMPL mesh generation skipped."
                    )
                    smpl_data = None


    effective_frame_ids_for_files = []
    smpl_indices_to_process = []


    is_single_frame_subject_dir = args.dataset_type in {"thuman2", "mvhumannet"}
    if args.dataset_type == "actorshq" and smpl_data is not None and "body_pose" in smpl_data:
        try:
            is_single_frame_subject_dir = (int(smpl_data["body_pose"].shape[0]) == 1)
        except Exception:
            pass
    if args.dataset_type == "talkbody4d" and smpl_data is not None and "body_pose" in smpl_data:

        try:
            is_single_frame_subject_dir = (int(smpl_data["body_pose"].shape[0]) == 1)
        except Exception:

            is_single_frame_subject_dir = True

    if is_single_frame_subject_dir:
        subject_id_from_path = Path(args.data_dir).name
        effective_frame_ids_for_files.append(subject_id_from_path)
        smpl_indices_to_process.append(0)
        if (
            args.frame_list
            or args.frame_start != 0
            or args.frame_end != 0
            or args.frame_step != 1
        ):
            logger.info(
                f"For dataset_type '{args.dataset_type}', frame_list/start/end/step arguments are ignored "
                f"(single-frame subject dir detected). Processing internal frame 0."
            )
        logger.info(
            f"Processing {args.dataset_type} subject dir: {subject_id_from_path} (as single frame)"
        )
    else:
        temp_frame_id_list = []
        if args.frame_list:
            temp_frame_id_list = [
                f.split(args.suffix)[0] if args.suffix else f for f in args.frame_list
            ]
            logger.info(f"Processing explicit frame list: {temp_frame_id_list}")
        elif args.frame_start is not None and args.frame_end is not None:

            for i in range(
                args.frame_start, args.frame_end + 1, args.frame_step
            ):
                temp_frame_id_list.append(f"{i:0{args.frame_format_digits}d}")
            logger.info(
                f"Processing frame range: {args.frame_start}-{args.frame_end} (inclusive) step {args.frame_step} ({len(temp_frame_id_list)} frames)"
            )

        if not temp_frame_id_list:
            logger.error(
                "No frames specified via --frame_list or --frame_start/--frame_end for non-THuman2 dataset."
            )
            return

        for fid_str in temp_frame_id_list:
            effective_frame_ids_for_files.append(
                fid_str + args.suffix
            )
            smpl_indices_to_process.append(
                int(fid_str)
            )

    if not effective_frame_ids_for_files:
        logger.error("No frames to process. Exiting.")
        return

    did_rectify_scale, did_rectify_transl = False, False
    height_written = False


    for i, file_frame_id_with_suffix in enumerate(effective_frame_ids_for_files):
        smpl_idx = smpl_indices_to_process[i]


        file_frame_id_numeric_part = (
            file_frame_id_with_suffix.split(args.suffix)[0]
            if args.suffix
            else file_frame_id_with_suffix
        )

        if is_single_frame_subject_dir:


            frame_output_dir = output_dir
        else:

            frame_output_dir = output_dir / file_frame_id_with_suffix

        frame_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"\nProcessing: file_frame_id '{file_frame_id_with_suffix}', smpl_idx {smpl_idx}. Saving to {frame_output_dir}"
        )

        smpl_mesh = None
        model_input_params = None
        if (
            smpl_data is not None and models_available
        ):
            try:

                if smpl_idx >= smpl_data["body_pose"].shape[0]:
                    raise IndexError(
                        f"SMPL index {smpl_idx} is out of bounds for loaded SMPL params (max index: {smpl_data['body_pose'].shape[0]-1})"
                    )

                current_smpl_params = {
                    k: v[smpl_idx]
                    for k, v in smpl_data.items()
                    if hasattr(v, "shape") and len(v.shape) >= 1 and v.shape[0] > smpl_idx
                }

                for k, v in smpl_data.items():
                    if k in current_smpl_params:
                        continue
                    if not hasattr(v, "shape") or len(v.shape) < 1:
                        continue
                    if int(v.shape[0]) == 1:
                        current_smpl_params[k] = v[0]


                model_input_params = {
                    k: v for k, v in current_smpl_params.items()
                }

                if (
                    args.dataset_type == "thuman2"
                    and model_input_params.get("translation") is not None
                ):
                    model_input_params["transl"] = model_input_params.pop("translation")


                rectification_info = apply_smpl_parameter_rectification(
                    model_input_params, args.rectified_smpl_scale, args.rectified_smpl_transl, args.dataset_type
                )


                if (
                    args.emit_smpl_scale_inv
                    and args.dataset_type == "thuman2"
                    and model_input_params.get("scale") is not None
                ):
                    try:
                        s_val = model_input_params["scale"]
                        if isinstance(s_val, torch.Tensor):
                            s = float(s_val.detach().cpu().view(-1)[0].item())
                        else:
                            s = float(s_val)
                        if abs(s) < 1e-8:
                            raise ValueError(f"Invalid SMPL scale (too small): {s}")

                        inv_s = 1.0 / s
                        S_inv = np.eye(4, dtype=np.float32)
                        S_inv[0, 0] = inv_s
                        S_inv[1, 1] = inv_s
                        S_inv[2, 2] = inv_s

                        np.save(str(frame_output_dir / "smpl_scale_inv_4x4.npy"), S_inv)
                        (frame_output_dir / "smpl_scale_inv.json").write_text(
                            json.dumps(
                                {
                                    "scale": s,
                                    "inv_scale": inv_s,
                                    "inv_scale_matrix_row_major": S_inv.reshape(-1).tolist(),
                                },
                                indent=2,
                            )
                        )
                        logger.info(
                            f"Saved THuman2 inverse scale artifact: scale={s:.6f} inv={inv_s:.6f} -> {frame_output_dir}"
                        )
                    except Exception as e_inv:
                        logger.warning(f"Failed to emit THuman2 smpl_scale_inv artifact: {e_inv}")


                if rectification_info["scale_applied"]:
                    did_rectify_scale = True
                    smpl_data["scale"][smpl_idx] = model_input_params["scale"]


                    scale_info_path = frame_output_dir / "scale_info.txt"
                    try:
                        with open(scale_info_path, "w") as f:
                            f.write(f"Original scale: {rectification_info['original_scale']}\n")
                            f.write(f"Rectified scale: {rectification_info['rectified_scale']}\n")
                        logger.success(f"Saved scale information to {scale_info_path}")
                    except Exception as e:
                        logger.error(f"Error saving scale information to {scale_info_path}: {e}")

                if rectification_info["transl_applied"]:
                    did_rectify_transl = True
                    smpl_data["transl"][smpl_idx] = model_input_params["transl"]


                    transl_info_path = frame_output_dir / "translation_info.txt"
                    try:
                        with open(transl_info_path, "w") as f:
                            f.write(f"Original translation: {rectification_info['original_transl']}\n")
                            f.write(f"Rectified translation: {rectification_info['rectified_transl']}\n")
                        logger.success(f"Saved translation information to {transl_info_path}")
                    except Exception as e:
                        logger.error(f"Error saving translation information to {transl_info_path}: {e}")


                if getattr(args, "estimate_height_from_betas", False) and not height_written:
                    try:

                        is_swapped_frame = is_swapped_subject(file_frame_id_with_suffix)
                        model_for_height = None
                        if is_swapped_frame:
                            model_for_height = smpl_model_body if smpl_model_body is not None else smpl_model
                        else:
                            model_for_height = smpl_model if smpl_model is not None else smpl_model_body
                        if model_for_height is None:
                            model_for_height = smpl_model_head

                        if model_for_height is None:
                            logger.warning("Height estimate skipped: no SMPL-X model available.")
                        else:
                            height_m = _estimate_height_m_from_model_input_params(
                                smpl_model=model_for_height,
                                model_input_params=model_input_params,
                                device=device,
                            )
                            if height_m is not None:
                                height_path = data_dir / "height.txt"
                                height_path.write_text(f"{height_m:.6f}\n")
                                logger.success(f"Saved estimated height (m) to {height_path}: {height_m:.6f}")
                                height_written = True
                    except Exception as e:
                        logger.warning(f"Height estimate failed (non-fatal): {e}")


                if is_swapped_subject(file_frame_id_with_suffix):
                    logger.info(f"Processing swapped subject with coordinate transformation: {file_frame_id_with_suffix}")


                    head_subject_id, body_subject_id = parse_swapped_subject_name(file_frame_id_with_suffix)
                    if head_subject_id is None or body_subject_id is None:
                        raise ValueError(f"Failed to parse swapped subject name: {file_frame_id_with_suffix}")

                    logger.info(f"Head donator: {head_subject_id}, Body donator: {body_subject_id}")


                    logger.info("Building SMPL mesh using body donator parameters (with pelvis joint extraction)")
                    current_body_model = smpl_model_body if smpl_model_body is not None else smpl_model
                    smpl_mesh, body_pelvis_joint = build_smplx_mesh(
                        current_body_model,
                        model_input_params,
                        device,
                        return_pelvis_joint=True,
                        dataset_type=args.dataset_type,
                    )


                else:

                    logger.debug("Processing non-swapped subject with standard logic")
                    current_model = smpl_model if smpl_model is not None else smpl_model_body
                    smpl_mesh = build_smplx_mesh(
                        current_model,
                        model_input_params,
                        device,
                        dataset_type=args.dataset_type,
                    )
                smpl_output_path = (
                    frame_output_dir / f"smpl_body.obj"
                )
                save_mesh(smpl_mesh, smpl_output_path)


                if getattr(args, "save_smpl_joints", False):
                    try:
                        current_model_for_joints = current_body_model if is_swapped_subject(file_frame_id_with_suffix) else current_model
                        joints_t = build_smplx_mesh(
                            current_model_for_joints,
                            model_input_params,
                            device,
                            return_vertices=False,
                            return_full_joints=True,
                            dataset_type=args.dataset_type,
                        )
                        joints_np = joints_t.detach().cpu().numpy().astype(np.float32)
                        joints_output_path = frame_output_dir / "smpl_joints.npy"
                        np.save(str(joints_output_path), joints_np)
                        logger.success(f"Saved SMPL joints to {joints_output_path} (shape={joints_np.shape})")


                        joint_names_path = frame_output_dir / "smpl_joint_names.txt"
                        try:
                            if isinstance(SMPLX_JOINT_NAMES, (list, tuple)) and len(SMPLX_JOINT_NAMES) == int(joints_np.shape[0]):
                                names = list(SMPLX_JOINT_NAMES)
                            else:
                                logger.warning(
                                    f"SMPLX_JOINT_NAMES length mismatch: names={len(SMPLX_JOINT_NAMES)} vs joints={int(joints_np.shape[0])}. "
                                    "Writing placeholder names joint_000..."
                                )
                                names = [f"joint_{i:03d}" for i in range(int(joints_np.shape[0]))]
                            joint_names_path.write_text("\n".join(names) + "\n")
                            logger.success(f"Saved SMPL joint names to {joint_names_path}")
                        except Exception as e:
                            logger.warning(f"Failed writing joint names file {joint_names_path}: {e}")

                    except Exception as e:
                        logger.error(f"Failed to save SMPL joints for {file_frame_id_with_suffix}: {e}")


                if getattr(args, "emit_smpl_only", False):
                    continue


                if smpl_lbs_weights is not None:
                    lbs_output_path = frame_output_dir / f"smpl_lbs_weights.npy"
                    try:
                        np.save(str(lbs_output_path), smpl_lbs_weights)
                        logger.success(f"Saved SMPL LBS weights to {lbs_output_path}")
                    except Exception as e:
                        logger.error(
                            f"Error saving SMPL LBS weights to {lbs_output_path}: {e}"
                        )

            except IndexError as e:
                logger.error(
                    f"Error indexing SMPL params for smpl_idx {smpl_idx} (file_frame_id: {file_frame_id_with_suffix}): {e}. Skipping SMPL mesh for this frame."
                )
                smpl_mesh = None
            except Exception as e:
                logger.error(
                    f"Error building SMPL mesh for file_frame_id {file_frame_id_with_suffix} (smpl_idx {smpl_idx}): {e}"
                )
                smpl_mesh = None

        garment_mesh = None
        labeled_ply_path, label_pkl_path = process_labeled_ply_mesh(
            data_dir, file_frame_id_numeric_part, args
        )

        _, label_pkl_ext_path = process_labeled_ply_mesh(
            data_dir, file_frame_id_numeric_part, args,
            outfix_suffix="_extended",
            seg_method_suffix="_no_sam" if args.use_no_sam_labels else ""
        )


        if labeled_ply_path.exists() and label_pkl_path.exists():
            garment_mesh, garment_verts_colors = extract_garment_mesh(
                labeled_ply_path, label_pkl_path, device
            )
        else:
            garment_mesh, garment_verts_colors = None, None

        if garment_mesh:
            garment_output_path_ply = (
                frame_output_dir / f"garments.ply"
            )


            save_mesh(
                garment_mesh, garment_output_path_ply, verts_colors=garment_verts_colors
            )


        if args.create_simplified_mesh:
            process_simplified_mesh_with_skin_mask(
                data_dir=data_dir,
                frame_output_dir=frame_output_dir,
                label_pkl_ext_path=label_pkl_ext_path,
                args=args,
                file_frame_id_with_suffix=file_frame_id_with_suffix
            )


        if args.save_skeleton_mesh:
            logger.info("Extracting SMPL skeleton mesh...")

            try:
                if model_input_params is None:
                    raise RuntimeError(
                        "SMPL parameters for this frame are not available (model_input_params is None). "
                        "This usually means SMPL params/model failed to load, the SMPL build path errored early, "
                        "or smpl_idx was out of range. Skeleton extraction requires valid per-frame SMPL params."
                    )
                if smpl_model is None:
                    raise RuntimeError(
                        "SMPL model is not available (smpl_model is None). Skeleton extraction requires a loaded SMPL/SMPL-X model."
                    )

                skeleton_mesh_path = extract_skeleton_mesh(
                    smpl_model=smpl_model,
                    smpl_params_batch=model_input_params,
                    output_dir=frame_output_dir,
                    include_fingers=args.include_skeleton_fingers,
                    exclude_foot_joints=args.exclude_foot_joints,
                    simplified=args.simplified_skeleton,
                    device=device,
                    dataset_type=args.dataset_type,
                )
                logger.success(f"Skeleton mesh saved: {skeleton_mesh_path}")


                logger.info("Auto-exporting SMPL skeleton LBS weights...")
                lbs_weights_npy_path = frame_output_dir / "smpl_lbs_weights.npy"

                if lbs_weights_npy_path.exists():
                    try:

                        lbs_weights_txt_path, export_weights = export_skeleton_lbs_weights(
                            lbs_weights_npy_path=lbs_weights_npy_path,
                            output_dir=frame_output_dir,
                            include_fingers=args.include_skeleton_fingers,
                            exclude_foot_joints=args.exclude_foot_joints,
                            simplified=args.simplified_skeleton
                        )
                        logger.success(f"LBS weights exported: {lbs_weights_txt_path}")


                    except Exception as lbs_error:
                        error_msg = f"Failed to export skeleton LBS weights for {file_frame_id_with_suffix}: {lbs_error}"
                        logger.error(error_msg)

                        raise RuntimeError(error_msg) from lbs_error
                else:
                    error_msg = f"SMPL LBS weights file not found for LBS export: {lbs_weights_npy_path}"
                    logger.error(error_msg)

                    raise FileNotFoundError(error_msg)

            except Exception as e:
                error_msg = f"Failed to extract skeleton mesh for {file_frame_id_with_suffix}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e


        if smpl_mesh and garment_mesh:
            logger.info("Attempting to combine SMPL and garment meshes.")
            try:

                smpl_verts = smpl_mesh.vertices
                smpl_faces = smpl_mesh.faces

                garment_verts = garment_mesh.vertices
                garment_faces = garment_mesh.faces


                garment_faces_offset = garment_faces + smpl_verts.shape[0]

                combined_verts_np = np.concatenate([smpl_verts, garment_verts], axis=0)
                combined_faces_np = np.concatenate(
                    [smpl_faces, garment_faces_offset], axis=0
                )


                smpl_vertex_colors_np = smpl_mesh.visual.vertex_colors[
                    ..., :3
                ]
                if (
                    smpl_vertex_colors_np is None
                    or smpl_vertex_colors_np.shape[0] != smpl_verts.shape[0]
                ):
                    logger.warning(
                        "SMPL mesh vertex colors not found or mismatch, using default."
                    )
                    smpl_vertex_colors_np = np.tile(
                        SMPL_BODY_COLOR_UINT8, (smpl_verts.shape[0], 1)
                    )


                if (
                    garment_verts_colors is None
                    or garment_verts_colors.shape[0] != garment_verts.shape[0]
                ):
                    logger.warning(
                        "Garment vertex colors not found or mismatch. This should not happen if garment mesh was processed correctly."
                    )


                if garment_verts_colors is not None:
                    combined_verts_colors_np = np.concatenate(
                        [
                            smpl_vertex_colors_np.astype(np.uint8),
                            garment_verts_colors.astype(np.uint8),
                        ],
                        axis=0,
                    )
                else:
                    logger.warning(
                        "Garment vertex colors are None during combined mesh creation. Using SMPL colors for all."
                    )
                    combined_verts_colors_np = smpl_vertex_colors_np.astype(np.uint8)


                combined_mesh = trimesh.Trimesh(
                    vertices=combined_verts_np,
                    faces=combined_faces_np,
                    vertex_colors=combined_verts_colors_np,
                )

                combined_output_path_ply = (
                    frame_output_dir / f"combined.ply"
                )


                save_mesh(
                    combined_mesh,
                    combined_output_path_ply,
                    verts_colors=combined_verts_colors_np,
                )


            except Exception as e:
                logger.error(
                    f"Error combining meshes for file_frame_id {file_frame_id_with_suffix} (smpl_idx {smpl_idx}): {e}"
                )
        elif smpl_mesh:
            logger.info(
                "Garment mesh not available, only SMPL body mesh was processed."
            )
        elif garment_mesh:
            logger.info(
                "SMPL body mesh not available, only garment mesh was processed."
            )
        else:
            logger.warning(
                f"No meshes processed for file_frame_id {file_frame_id_with_suffix} (smpl_idx {smpl_idx})."
            )

    if did_rectify_scale or did_rectify_transl:


        original_smpl_params_path = data_dir / "smpl_params.npz"
        backup_smpl_params_path = data_dir / "smpl_params_original.npz"

        try:
            import shutil

            shutil.copy2(str(original_smpl_params_path), str(backup_smpl_params_path))
            logger.success(
                f"Backed up original SMPL parameters to {backup_smpl_params_path}"
            )
        except Exception as e:
            logger.error(f"Error backing up original SMPL parameters: {e}")


        smpl_data_npz = {k: v.cpu().numpy() for k, v in smpl_data.items()}
        smpl_params_path = data_dir / "smpl_params.npz"
        np.savez(smpl_params_path, **smpl_data_npz)

def save_ply_with_lbs_weights_visualization(mesh: trimesh.Trimesh, lbs_weights: np.ndarray, output_path: Path):

    import cv2
    lbs_joint_argmax = np.argmax(lbs_weights, axis=0)
    max_lbs_weight = np.max(lbs_weights, axis=0)


    joint_min = 0
    joint_max = max(1, lbs_weights.shape[0] - 1)
    joint_norm = ((lbs_joint_argmax - joint_min) / (joint_max - joint_min) * 255).astype(np.uint8)
    cv2_cmap = cv2.COLORMAP_JET


    joint_norm_img = joint_norm.reshape(-1, 1)
    joint_colors_bgr = cv2.applyColorMap(joint_norm_img, cv2_cmap)[:, 0, :]


    max_lbs_weight_normalized = np.clip(max_lbs_weight, 0.0, 1.0)[:, None]
    vertex_bgr = np.clip(joint_colors_bgr.astype(np.float32) * max_lbs_weight_normalized, 0, 255).astype(np.uint8)


    vertex_rgb = vertex_bgr[:, ::-1]


    ply_mesh = mesh.copy()
    rgba = np.concatenate(
        [vertex_rgb, np.full((vertex_rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1
    ).astype(np.uint8)
    ply_mesh.visual.vertex_colors = rgba
    ply_mesh.export(str(output_path), file_type="ply")

    return ply_mesh

if __name__ == "__main__":
    main()
