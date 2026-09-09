import numpy as np
import plyfile
import json
import trimesh
import torch
import pytorch3d.transforms as p3d_transforms
import os
from pathlib import Path
from loguru import logger


def load_ply_gaussians(ply_path):

    try:
        plydata = plyfile.PlyData.read(ply_path)
        vertices = plydata['vertex']

        gaussians = {}
        for name in vertices.data.dtype.names:
            gaussians[name] = np.asarray(vertices[name])

        if 'x' not in gaussians or 'y' not in gaussians or 'z' not in gaussians:
            raise ValueError("PLY file must contain x, y, z coordinates for Gaussians.")


        default_properties = {
            'f_dc_0': 0.0, 'f_dc_1': 0.0, 'f_dc_2': 0.0,
            'opacity': 1.0,
            'scale_0': 0.01, 'scale_1': 0.01, 'scale_2': 0.01,
            'rot_0': 1.0, 'rot_1': 0.0, 'rot_2': 0.0, 'rot_3': 0.0
        }
        for prop, default_val in default_properties.items():
            if prop not in gaussians:
                logger.warning(f"Property '{prop}' not found in {ply_path}. Using default value {default_val}.")
                gaussians[prop] = np.full(gaussians['x'].shape, default_val, dtype=np.float32)

        return gaussians, vertices.data.dtype.names
    except Exception as e:
        logger.error(f"Error loading PLY file {ply_path}: {e}")
        raise


def load_segmentation_labels(pkl_path):

    try:
        import pickle
        with open(pkl_path, 'rb') as f:
            label_data = pickle.load(f)
        if 'scan_labels' not in label_data:
            raise ValueError(f"'scan_labels' key not found in {pkl_path}.")
        return label_data['scan_labels']
    except Exception as e:
        logger.error(f"Error loading labels from PKL {pkl_path}: {e}")
        raise


def load_smplx_vert_segmentation(json_path):

    try:
        p = Path(str(json_path))
        if not p.exists():


            if p.name == "smplx_vert_segmentation.json":
                here = Path(__file__).resolve()
                asset_root = Path(os.environ.get("AVATARMIX_ASSET_ROOT", here.parents[5] / "external_assets"))
                fallback = asset_root / "smplx_vert_segmentation.json"
                if fallback.exists():
                    logger.warning(
                        f"SMPLX vert segmentation not found at {p}; using shared asset: {fallback}"
                    )
                    p = fallback

        with open(str(p), 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading SMPLX vertex segmentation from {json_path}: {e}")
        raise


def load_embedding_json(json_path):

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if 'sample_fidxs' not in data:
            raise ValueError(f"'sample_fidxs' key not found in {json_path}.")
        return np.array(data['sample_fidxs'])
    except Exception as e:
        logger.error(f"Error loading embedding JSON {json_path}: {e}")
        raise


def load_embedding_full(json_path: str) -> dict:

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        if "sample_fidxs" not in data:
            raise ValueError(f"'sample_fidxs' key not found in {json_path}.")
        return data
    except Exception as e:
        logger.error(f"Error loading full embedding JSON {json_path}: {e}")
        raise


def load_mesh(obj_path):

    try:
        mesh = trimesh.load_mesh(obj_path, process=False)
        return mesh
    except Exception as e:
        logger.error(f"Error loading mesh {obj_path}: {e}")
        raise


def _rot_to_matrix(rot_value: np.ndarray, frame_idx: int, device: str) -> torch.Tensor:

    rot_arr = rot_value
    if isinstance(rot_arr, torch.Tensor):
        rot_arr = rot_arr.detach().cpu().numpy()
    rot_arr = np.asarray(rot_arr)


    if rot_arr.ndim >= 2 and rot_arr.shape[0] > 1 and rot_arr.shape[-1] != 3:

        rot_f = rot_arr[frame_idx]
    elif rot_arr.ndim >= 2 and rot_arr.shape[0] > 1 and rot_arr.shape[-1] == 3:

        rot_f = rot_arr[frame_idx]
    else:

        rot_f = rot_arr[0] if rot_arr.ndim > 2 or (rot_arr.ndim == 2 and rot_arr.shape[0] == 1) else rot_arr

    rot_f = np.asarray(rot_f, dtype=np.float32)
    if rot_f.shape == (3, 3):
        return torch.from_numpy(rot_f).to(device)
    if rot_f.shape == (3,):
        rot_aa = torch.from_numpy(rot_f).to(device)
        return p3d_transforms.axis_angle_to_matrix(rot_aa)

    raise ValueError(f"Unsupported rotation shape for SMPL transform: {rot_f.shape}")


def _vec3_for_frame(value: np.ndarray, frame_idx: int, device: str) -> torch.Tensor:

    arr = np.asarray(value)
    if arr.ndim == 1:
        v = arr
    else:
        v = arr[frame_idx] if frame_idx < arr.shape[0] else arr[0]
    v = np.asarray(v, dtype=np.float32).reshape(3,)
    return torch.from_numpy(v).to(device)


def load_smpl_transform_params_for_frame(
    smpl_params_path: str,
    frame_idx: int = 0,
    device: str = 'cpu',
    dataset_type: str | None = None,
):

    try:
        smpl_data_npz = np.load(smpl_params_path, allow_pickle=True)


        use_post_forward_rigid = ('Rh' in smpl_data_npz and 'Th' in smpl_data_npz)
        if dataset_type in ('talkbody4d', 'mvhumannet', 'actorshq'):
            use_post_forward_rigid = use_post_forward_rigid or True

        if use_post_forward_rigid and ('Rh' in smpl_data_npz and 'Th' in smpl_data_npz):
            global_orient_matrix = _rot_to_matrix(smpl_data_npz['Rh'], frame_idx=frame_idx, device=device)
            transl = _vec3_for_frame(smpl_data_npz['Th'], frame_idx=frame_idx, device=device)
        else:
            global_orient_matrix = _rot_to_matrix(smpl_data_npz['global_orient'], frame_idx=frame_idx, device=device)
            transl = _vec3_for_frame(smpl_data_npz['transl'], frame_idx=frame_idx, device=device)

        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        if 'scale' in smpl_data_npz:
            raw_scale = smpl_data_npz['scale']
            if isinstance(raw_scale, (int, float)):
                 scale = torch.tensor([float(raw_scale)], device=device, dtype=torch.float32)
            elif isinstance(raw_scale, np.ndarray) and raw_scale.ndim >= 1:

                 scale_val = raw_scale[frame_idx] if raw_scale.ndim > 0 and frame_idx < len(raw_scale) else raw_scale[0]
                 scale = torch.tensor([float(scale_val)], device=device, dtype=torch.float32)
            else:
                print(f"Warning: SMPL 'scale' in {smpl_params_path} has unexpected format: {type(raw_scale)}. Using default 1.0.")
        else:
            print(f"Info: SMPL 'scale' not found in {smpl_params_path}. Using default 1.0.")

        return global_orient_matrix, transl, scale.item()

    except Exception as e:
        print(f"Error loading SMPL transform parameters from {smpl_params_path} for frame {frame_idx}: {e}")

        return torch.eye(3, device=device, dtype=torch.float32), torch.zeros(3, device=device, dtype=torch.float32), 1.0
