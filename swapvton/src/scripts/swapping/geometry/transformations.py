import torch
import numpy as np
import pytorch3d.transforms as p3d_transforms
from loguru import logger


def _transform_gaussian_positions(positions_tensor, transl, scale_factor, global_orient_matrix, to_canonical: bool, pelvis_joint_j0=None):


    rotation_center = pelvis_joint_j0.unsqueeze(0) if pelvis_joint_j0 is not None else torch.zeros_like(transl).unsqueeze(0)

    if to_canonical:

        positions_centered = positions_tensor - transl.unsqueeze(0)
        positions_scaled = positions_centered / scale_factor


        positions_rel_pelvis = positions_scaled - rotation_center
        positions_rotated = torch.matmul(positions_rel_pelvis, global_orient_matrix)
        positions_normed = positions_rotated + rotation_center
        return positions_normed
    else:


        positions_rel_pelvis = positions_tensor - rotation_center
        positions_rotated = torch.matmul(positions_rel_pelvis, global_orient_matrix.transpose(0, 1))
        positions_after_rotation = positions_rotated + rotation_center
        positions_scaled = positions_after_rotation * scale_factor
        positions_world = positions_scaled + transl.unsqueeze(0)
        return positions_world


def _transform_gaussian_rotations(quaternions_tensor, global_orient_matrix, to_canonical: bool):


    smpl_rot_quat = p3d_transforms.matrix_to_quaternion(global_orient_matrix.unsqueeze(0))

    if to_canonical:


        smpl_rot_quat_inv = p3d_transforms.quaternion_invert(smpl_rot_quat)
        transformed_quats = p3d_transforms.quaternion_multiply(smpl_rot_quat_inv.expand_as(quaternions_tensor), quaternions_tensor)
    else:

        transformed_quats = p3d_transforms.quaternion_multiply(smpl_rot_quat.expand_as(quaternions_tensor), quaternions_tensor)
    return transformed_quats


def _transform_gaussian_scales(scales_3d_tensor, subject_scale_factor, to_canonical: bool):

    if to_canonical:

        return torch.log(torch.exp(scales_3d_tensor) / subject_scale_factor)
    else:

        return torch.log(torch.exp(scales_3d_tensor) * subject_scale_factor)


def transform_gaussians(gaussians_dict, property_names, smpl_transl, smpl_scale_factor, smpl_global_orient_matrix, to_canonical: bool, device: str = 'cpu', pelvis_joint_j0: torch.Tensor = None):

    transformed_gaussians = {k: v.clone() if isinstance(v, torch.Tensor) else torch.from_numpy(np.array(v)).to(device)
                           for k, v in gaussians_dict.items()}


    if 'x' in transformed_gaussians and 'y' in transformed_gaussians and 'z' in transformed_gaussians:
        positions = torch.stack([
            transformed_gaussians['x'],
            transformed_gaussians['y'],
            transformed_gaussians['z']
        ], dim=-1).to(device)

        transformed_positions = _transform_gaussian_positions(
            positions, smpl_transl.to(device), smpl_scale_factor,
            smpl_global_orient_matrix.to(device), to_canonical, pelvis_joint_j0
        )
        transformed_gaussians['x'] = transformed_positions[:, 0]
        transformed_gaussians['y'] = transformed_positions[:, 1]
        transformed_gaussians['z'] = transformed_positions[:, 2]
    else:
        logger.warning("x,y,z not found in gaussians for position transformation.")


    if all(f'rot_{i}' in transformed_gaussians for i in range(4)):
        quaternions_wxyz = torch.stack([
            transformed_gaussians['rot_0'],
            transformed_gaussians['rot_1'],
            transformed_gaussians['rot_2'],
            transformed_gaussians['rot_3']
        ], dim=-1).to(device)

        transformed_quats = _transform_gaussian_rotations(quaternions_wxyz, smpl_global_orient_matrix.to(device), to_canonical)
        transformed_gaussians['rot_0'] = transformed_quats[:, 0]
        transformed_gaussians['rot_1'] = transformed_quats[:, 1]
        transformed_gaussians['rot_2'] = transformed_quats[:, 2]
        transformed_gaussians['rot_3'] = transformed_quats[:, 3]
    else:
        logger.warning("rot_0 to rot_3 not found for rotation transformation.")


    if all(f'scale_{i}' in transformed_gaussians for i in range(3)):
        scales_3d = torch.stack([
            transformed_gaussians['scale_0'],
            transformed_gaussians['scale_1'],
            transformed_gaussians['scale_2']
        ], dim=-1).to(device)

        transformed_scales = _transform_gaussian_scales(scales_3d, smpl_scale_factor, to_canonical)
        transformed_gaussians['scale_0'] = transformed_scales[:, 0]
        transformed_gaussians['scale_1'] = transformed_scales[:, 1]
        transformed_gaussians['scale_2'] = transformed_scales[:, 2]
    else:
        logger.warning("scale_0 to scale_2 not found for scale transformation.")


    return transformed_gaussians
