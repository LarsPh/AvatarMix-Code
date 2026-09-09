import numpy as np
import torch
import pytorch3d.transforms as p3d_transforms
from src.utils.smplx_utils.smplx_utils import create_smplx_model_for_neus2_reposing, compute_reference_transform_data


def _npz_get_frame(arr: np.ndarray, frame_idx: int) -> np.ndarray:

    a = np.asarray(arr)
    if a.ndim == 0:
        return a
    if a.ndim == 1:
        return a

    if frame_idx < a.shape[0]:
        return a[frame_idx]
    return a[0]


def _as_batch_tensor(x: np.ndarray, device: str) -> torch.Tensor:

    t = torch.from_numpy(np.asarray(x).astype(np.float32)).to(device)
    if t.ndim == 0:
        t = t.view(1, 1)
    elif t.ndim == 1:
        t = t.unsqueeze(0)
    return t


def _axis_angle_to_matrix_batched(aa: torch.Tensor) -> torch.Tensor:

    if aa.ndim == 1:
        aa = aa.unsqueeze(0)
    if aa.shape[-1] != 3:
        raise ValueError(f"Expected axis-angle (...,3), got {tuple(aa.shape)}")
    return p3d_transforms.axis_angle_to_matrix(aa)


def _apply_post_rigid_transform_xyz(
    xyz: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:

    if xyz.ndim == 2:
        xyz_b = xyz.unsqueeze(0)
    else:
        xyz_b = xyz
    if R.ndim == 2:
        R_b = R.unsqueeze(0)
    else:
        R_b = R
    if t.ndim == 1:
        t_b = t.unsqueeze(0)
    else:
        t_b = t
    out = torch.matmul(xyz_b, R_b.transpose(1, 2)) + t_b.unsqueeze(1)
    return out.squeeze(0) if xyz.ndim == 2 else out


def get_pelvis_joint_j0(smpl_params_path, frame_idx=0, device='cpu', dataset_type='thuman2', gender='neutral'):


    smpl_data_npz = np.load(smpl_params_path, allow_pickle=True)


    smpl_params = {}
    for key in ['global_orient', 'body_pose', 'transl', 'betas', 'scale']:
        if key in smpl_data_npz:
            data = smpl_data_npz[key]
            if data.ndim > 1:
                smpl_params[key] = torch.from_numpy(data[frame_idx:frame_idx+1].astype(np.float32)).to(device)
            else:
                smpl_params[key] = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).to(device)


    for key in ['jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'left_hand_pose', 'right_hand_pose']:
        if key in smpl_data_npz:
            data = smpl_data_npz[key]
            if data.ndim > 1:
                smpl_params[key] = torch.from_numpy(data[frame_idx:frame_idx+1].astype(np.float32)).to(device)
            else:
                smpl_params[key] = torch.from_numpy(data.astype(np.float32)).unsqueeze(0).to(device)


    smpl_model = create_smplx_model_for_neus2_reposing(
        smpl_params=smpl_params,
        batch_size=1,
        device=device,
        gender=gender,
        dataset_type=dataset_type
    )


    transform_data = compute_reference_transform_data(smpl_params, smpl_model, return_vertices=False)

    return transform_data['j0']


def get_body_joints_world(
    smpl_params_path: str,
    frame_idx: int = 0,
    device: str = "cpu",
    dataset_type: str = "thuman2",
    gender: str = "neutral",
) -> torch.Tensor:

    smpl_data_npz = np.load(smpl_params_path, allow_pickle=True)


    smpl_params: dict[str, torch.Tensor] = {}
    for key in [
        "global_orient",
        "body_pose",
        "transl",
        "betas",
        "scale",
        "jaw_pose",
        "leye_pose",
        "reye_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "v_shape",
        "v_pose",
    ]:
        if key not in smpl_data_npz:
            continue
        v = _npz_get_frame(smpl_data_npz[key], frame_idx=frame_idx)
        smpl_params[key] = _as_batch_tensor(v, device=device)


    ext_R = None
    ext_t = None
    if dataset_type in ("mvhumannet", "actorshq"):
        if "global_orient" in smpl_params:
            ext_R = _axis_angle_to_matrix_batched(smpl_params["global_orient"])[0]
        if "transl" in smpl_params:
            ext_t = smpl_params["transl"][0]
        smpl_params["global_orient"] = torch.zeros((1, 3), device=device, dtype=torch.float32)
        smpl_params["transl"] = torch.zeros((1, 3), device=device, dtype=torch.float32)


    smpl_model = create_smplx_model_for_neus2_reposing(
        smpl_params=smpl_params,
        batch_size=1,
        device=device,
        gender=gender,
        dataset_type=dataset_type,
    )


    with torch.no_grad():
        smpl_params_filtered = {k: v for k, v in smpl_params.items() if v is not None}
        out = smpl_model.forward(**smpl_params_filtered, return_verts=False)
        joints = out.joints[0]


    if dataset_type == "talkbody4d" and ("Rh" in smpl_data_npz and "Th" in smpl_data_npz):
        Rh_v = _npz_get_frame(smpl_data_npz["Rh"], frame_idx=frame_idx)
        Th_v = _npz_get_frame(smpl_data_npz["Th"], frame_idx=frame_idx)
        Rh_t = torch.from_numpy(np.asarray(Rh_v).astype(np.float32)).to(device)
        Th_t = torch.from_numpy(np.asarray(Th_v).astype(np.float32)).to(device)
        if Rh_t.shape == (3,):
            R = p3d_transforms.axis_angle_to_matrix(Rh_t.unsqueeze(0))[0]
        elif Rh_t.shape == (3, 3):
            R = Rh_t
        else:

            Rh_t = Rh_t.squeeze(0)
            if Rh_t.shape == (3,):
                R = p3d_transforms.axis_angle_to_matrix(Rh_t.unsqueeze(0))[0]
            else:
                R = Rh_t
        t = Th_t.view(-1)[:3]
        joints = _apply_post_rigid_transform_xyz(joints, R, t)


    if dataset_type in ("mvhumannet", "actorshq") and (ext_R is not None) and (ext_t is not None):
        joints = _apply_post_rigid_transform_xyz(joints, ext_R, ext_t)

    return joints
