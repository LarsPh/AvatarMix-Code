import torch
import numpy as np
from pathlib import Path
from loguru import logger


def get_skeleton_joint_indices(include_fingers=False, exclude_foot_joints=True, simplified=True):


    main_body_indices = list(range(22))


    if exclude_foot_joints:

        main_body_indices = [i for i in main_body_indices if i not in [10, 11]]


    if simplified:

        simplification_removals = [3, 6, 13, 14]
        main_body_indices = [i for i in main_body_indices if i not in simplification_removals]

    if include_fingers:

        finger_indices = list(range(25, 55))
        return main_body_indices + finger_indices
    else:
        return main_body_indices


def find_valid_ancestor(joint_idx, parents, idx_mapping):

    current = joint_idx
    while current != -1:
        current = parents[current].item()
        if current in idx_mapping:
            return idx_mapping[current]
    return -1


def filter_joints_and_parents(joint_positions, parents, keep_indices):


    idx_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_indices)}


    filtered_joints = joint_positions[keep_indices]


    filtered_parents = []
    for old_idx in keep_indices:
        old_parent = parents[old_idx].item()
        if old_parent == -1:
            filtered_parents.append(-1)
        elif old_parent in idx_mapping:
            filtered_parents.append(idx_mapping[old_parent])
        else:
            valid_ancestor = find_valid_ancestor(old_parent, parents, idx_mapping)
            filtered_parents.append(valid_ancestor)

    return filtered_joints, filtered_parents


def validate_skeleton_connectivity(filtered_parents):


    root_count = sum(1 for p in filtered_parents if p == -1)
    if root_count != 1:
        raise ValueError(f"Invalid skeleton: {root_count} root joints (expected 1)")


    visited = set()
    for i in range(len(filtered_parents)):
        current = i
        path = set()
        while current != -1 and current not in visited:
            if current in path:
                raise ValueError(f"Cycle detected in skeleton at joint {current}")
            path.add(current)
            if current < len(filtered_parents):
                current = filtered_parents[current]
            else:
                break
        visited.update(path)


def save_skeleton_obj(file_path, joint_positions, skeleton_lines, include_fingers=False, exclude_foot_joints=True):

    joint_count = len(joint_positions)
    line_count = len(skeleton_lines)

    with open(file_path, 'w') as f:

        for joint in joint_positions:
            if torch.is_tensor(joint):
                joint = joint.detach().cpu().numpy()
            f.write(f"v {joint[0]:.6f} {joint[1]:.6f} {joint[2]:.6f}\n")


        for line in skeleton_lines:
            f.write(f"l {line[0]+1} {line[1]+1}\n")


def save_reposed_smpl_mesh_and_skeleton(smpl_model, smpl_params, output_dir, frame_idx,
                                       save_mesh=True, save_skeleton=True,
                                       target_global_orient=None, target_transl=None, target_scale=None,
                                       target_post_rh=None, target_post_th=None,
                                       include_fingers=False, exclude_foot_joints=True, simplified=True, device='cpu'):

    output_dir = Path(output_dir)
    mesh_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}.obj"
    skeleton_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_skeleton.obj"

    logger.info(f"Saving reposed SMPL (mesh: {save_mesh}, skeleton: {save_skeleton}) for frame {frame_idx}")

    try:

        smpl_params_with_tar_global = smpl_params.copy()


        if target_global_orient is not None:
            smpl_params_with_tar_global['global_orient'] = target_global_orient
            logger.info("Applied target global_orient to reposed SMPL")
        if target_transl is not None:
            smpl_params_with_tar_global['transl'] = target_transl
            logger.info("Applied target transl to reposed SMPL")
        if target_scale is not None:
            smpl_params_with_tar_global['scale'] = target_scale
            logger.info("Applied target scale to reposed SMPL")


        smpl_params_filtered = {k: v for k, v in smpl_params_with_tar_global.items() if v is not None}
        smpl_output = smpl_model.forward(**smpl_params_filtered, return_verts=save_mesh)


        if target_post_rh is not None and target_post_th is not None:
            try:
                import pytorch3d.transforms
                Rh = target_post_rh
                Th = target_post_th
                if torch.is_tensor(Rh) and Rh.ndim == 1:
                    Rh = Rh.unsqueeze(0)
                if torch.is_tensor(Th) and Th.ndim == 1:
                    Th = Th.unsqueeze(0)
                R = pytorch3d.transforms.axis_angle_to_matrix(Rh)
                if hasattr(smpl_output, "vertices") and smpl_output.vertices is not None:
                    smpl_output.vertices = smpl_output.vertices @ R.transpose(1, 2) + Th.unsqueeze(1)
                if hasattr(smpl_output, "joints") and smpl_output.joints is not None:
                    smpl_output.joints = smpl_output.joints @ R.transpose(1, 2) + Th.unsqueeze(1)
                logger.info("Applied target post-forward Rh/Th to reposed SMPL outputs")
            except Exception as e:
                logger.warning(f"Failed applying post-forward Rh/Th: {e}")

        saved_mesh_path = None
        saved_skeleton_path = None


        if save_mesh:
            import pytorch3d.io
            pytorch3d.io.save_obj(str(mesh_path),
                                 smpl_output.vertices[0],
                                 torch.tensor(smpl_model.faces.astype(np.int32), device=device))
            logger.success(f"Reposed SMPL mesh saved to: {mesh_path}")
            saved_mesh_path = mesh_path


        if save_skeleton:
            joint_positions = smpl_output.joints[0]
            logger.info(f"Extracted {joint_positions.shape[0]} joints from SMPL model")


            parents = smpl_model.parents
            keep_indices = get_skeleton_joint_indices(include_fingers, exclude_foot_joints, simplified)
            filtered_joints, filtered_parents = filter_joints_and_parents(
                joint_positions, parents, keep_indices)

            logger.info(f"Filtered to {len(filtered_joints)} skeleton joints")


            validate_skeleton_connectivity(filtered_parents)


            skeleton_lines = []
            for i, parent_idx in enumerate(filtered_parents):
                if parent_idx != -1:
                    skeleton_lines.append([i, parent_idx])

            logger.info(f"Created {len(skeleton_lines)} skeleton connections")


            save_skeleton_obj(skeleton_path, filtered_joints, skeleton_lines, include_fingers, exclude_foot_joints)
            logger.success(f"Reposed SMPL skeleton saved to: {skeleton_path}")
            saved_skeleton_path = skeleton_path

        return saved_mesh_path, saved_skeleton_path

    except Exception as e:
        logger.error(f"Reposed SMPL saving failed: {e}")
        raise RuntimeError(f"Failed to save reposed SMPL mesh and skeleton: {e}") from e
