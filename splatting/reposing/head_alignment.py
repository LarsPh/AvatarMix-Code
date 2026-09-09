import numpy as np
import torch
import pytorch3d.io
import pytorch3d.structures
import pytorch3d.transforms as transforms
from pathlib import Path
import trimesh


SURFACE_LABELS = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']
DETAILED_SURFACE_LABELS = ["torso_skin", "head", "left_arm", "right_arm", "left_leg", "right_leg", "clothes"]


def load_segmentation_labels(pkl_path):

    import pickle
    with open(pkl_path, 'rb') as f:
        label_data = pickle.load(f)
    return label_data['scan_labels']


def identify_head_region(scan_mesh_vertices, scan_mesh_faces, segmentation_labels,
                        current_surface_labels, hair_label_name="hair",
                        use_direct_head_label=True, direct_head_label_name="face"):

    head_label_idx = current_surface_labels.index(direct_head_label_name)
    head_verts = np.where(segmentation_labels == head_label_idx)[0]


    try:
        hair_label_idx = current_surface_labels.index(hair_label_name)
        hair_verts = np.where(segmentation_labels == hair_label_idx)[0]
        head_verts = np.union1d(head_verts, hair_verts)
    except ValueError:
        pass

    return head_verts, None


def load_head_vertices_for_subject(dat_dir, nerf_mesh_vertices, nerf_faces_pt,
                                   use_detailed_labels=True, label_dir_name='labeled'):

    MIN_HEAD_VERTICES = 100


    label_suffix = "_extended" if use_detailed_labels else ""
    seg_path = Path(dat_dir) / "mesh" / label_dir_name / f"label-f0000{label_suffix}.pkl"

    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation file not found: {seg_path}")


    seg_labels = load_segmentation_labels(str(seg_path))


    current_surface_labels = DETAILED_SURFACE_LABELS if use_detailed_labels else SURFACE_LABELS


    if use_detailed_labels:

        head_label_name = "head"
    else:

        head_label_name = "skin"


    head_verts, _ = identify_head_region(
        nerf_mesh_vertices.cpu().numpy(),
        nerf_faces_pt.cpu().numpy(),
        seg_labels,
        current_surface_labels,
        hair_label_name="hair",
        use_direct_head_label=use_detailed_labels,
        direct_head_label_name=head_label_name
    )

    if head_verts is None or len(head_verts) < MIN_HEAD_VERTICES:
        raise ValueError(f"Insufficient head vertices ({len(head_verts) if head_verts is not None else 0} < {MIN_HEAD_VERTICES})")

    return torch.tensor(head_verts, device=nerf_mesh_vertices.device)


def compute_average_transform(transform_matrices):

    if transform_matrices.shape[0] == 0:
        return torch.eye(4, device=transform_matrices.device)

    if transform_matrices.shape[0] == 1:
        return transform_matrices[0]


    translations = transform_matrices[:, :3, 3]
    rotation_matrices = transform_matrices[:, :3, :3]


    avg_translation = translations.mean(dim=0)


    quaternions = transforms.matrix_to_quaternion(rotation_matrices)


    avg_quaternion = quaternions.mean(dim=0)
    avg_quaternion = avg_quaternion / torch.norm(avg_quaternion)


    avg_rotation = transforms.quaternion_to_matrix(avg_quaternion)


    avg_transform = torch.eye(4, device=transform_matrices.device)
    avg_transform[:3, :3] = avg_rotation
    avg_transform[:3, 3] = avg_translation

    return avg_transform


def compute_robust_average_transform(transform_matrices, outlier_threshold=2.0):

    if transform_matrices.shape[0] <= 1:
        return compute_average_transform(transform_matrices)


    translations = transform_matrices[:, :3, 3]


    trans_mean = translations.mean(dim=0)
    trans_distances = torch.norm(translations - trans_mean, dim=1)
    trans_threshold = trans_mean.norm() + outlier_threshold * trans_distances.std()

    valid_mask = trans_distances < trans_threshold
    num_outliers = (~valid_mask).sum().item()

    if num_outliers > 0:
        print(f"Head alignment: filtering {num_outliers}/{transform_matrices.shape[0]} outlier transformations")

    if valid_mask.sum() > 0:
        filtered_transforms = transform_matrices[valid_mask]
        return compute_average_transform(filtered_transforms)
    else:
        raise ValueError("All transformations filtered as outliers - cannot compute valid average")


def extract_head_only_mesh_vis(aligned_vertices, nerf_faces_pt, head_vertex_indices):

    try:

        vertices_np = aligned_vertices.cpu().numpy() if hasattr(aligned_vertices, 'cpu') else aligned_vertices
        faces_np = nerf_faces_pt.cpu().numpy() if hasattr(nerf_faces_pt, 'cpu') else nerf_faces_pt
        head_indices_np = head_vertex_indices.cpu().numpy() if hasattr(head_vertex_indices, 'cpu') else head_vertex_indices


        mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=False)


        head_vertex_set = set(head_indices_np.flatten())
        head_face_mask = np.all(np.isin(faces_np, list(head_vertex_set)), axis=1)
        head_face_indices = np.where(head_face_mask)[0]

        if len(head_face_indices) == 0:
            print("Warning: No faces found containing head vertices")
            return None


        head_submesh = mesh.submesh([head_face_indices], only_watertight=False, append=True)

        return head_submesh

    except Exception as e:
        print(f"Warning: Failed to extract head-only mesh for visualization: {e}")
        return None

def construct_w2cano_inv_matrix(global_orient_matrix, j0_src, src_scale, src_transl):

    device = global_orient_matrix.device


    scaled_rotation = src_scale * global_orient_matrix


    rotation_translation = global_orient_matrix @ j0_src.unsqueeze(-1)
    translation_vector = src_scale * (j0_src - rotation_translation.squeeze(-1)) + src_transl


    w2cano_inv = torch.eye(4, device=device, dtype=torch.float32)
    w2cano_inv[:3, :3] = scaled_rotation
    w2cano_inv[:3, 3] = translation_vector

    return w2cano_inv

def construct_w2cano_matrix(global_orient_matrix, j0_src, src_scale, src_transl):

    device = global_orient_matrix.device


    inv_scale = 1.0 / src_scale
    inv_rotation = global_orient_matrix.transpose(1, 2)


    scaled_inv_rotation = inv_scale * inv_rotation


    neg_transl_j0 = -(src_transl * inv_scale + j0_src)
    translation_vector = (inv_rotation @ neg_transl_j0.unsqueeze(-1)).squeeze(-1) + j0_src


    w2cano = torch.eye(4, device=device, dtype=torch.float32)
    w2cano[:3, :3] = scaled_inv_rotation
    w2cano[:3, 3] = translation_vector

    return w2cano

def save_head_alignment_transform(output_dir, frame_idx, avg_head_transform,
                                global_orient_matrix, j0_src, src_scale, src_transl,
                                posed_nerf_verts_tar, nerf_faces_pt, head_vertex_indices, gs_model,
                                verbose=False, npz_suffix=''):

    output_dir = Path(output_dir)
    aligned_dir = output_dir / "aligned_head_assets"
    aligned_dir.mkdir(parents=True, exist_ok=True)


    T_inv = torch.inverse(avg_head_transform)


    if T_inv.shape == (3, 4):
        T_inv_4x4 = torch.eye(4, device=T_inv.device, dtype=T_inv.dtype)
        T_inv_4x4[:3, :] = T_inv
        T_inv = T_inv_4x4


    w2cano_inv = construct_w2cano_inv_matrix(global_orient_matrix, j0_src, src_scale, src_transl)
    w2cano = construct_w2cano_matrix(global_orient_matrix, j0_src, src_scale, src_transl)


    A = w2cano_inv @ T_inv @ w2cano

    if verbose:
        print(f"avg_head_transform shape: {avg_head_transform.shape}")
        print(f"T_inv shape: {T_inv.shape}")
        print(f"w2cano_inv shape: {w2cano_inv.shape}")
        print(f"w2cano shape: {w2cano.shape}")
        print(f"Final A matrix shape: {A.shape}")


    transform_filename = f"head_transform_avg_frame_{frame_idx:04d}{npz_suffix}.npz"
    transform_path = aligned_dir / transform_filename
    np.savez(str(transform_path),
             avg_head_transform=A.cpu().numpy(),
             frame_idx=frame_idx)

    if verbose:
        print(f"Saved head alignment transformation matrix A: {transform_path}")


    if posed_nerf_verts_tar is not None and verbose:
        print("Creating visualization debugging assets...")


        posed_verts_homo = torch.cat([posed_nerf_verts_tar, torch.ones(posed_nerf_verts_tar.shape[0], 1, device=posed_nerf_verts_tar.device)], dim=1)
        aligned_verts_homo = (A @ posed_verts_homo.T).T
        aligned_world_verts = aligned_verts_homo[:, :3]


        mesh_vis_filename = f"aligned_mesh_vis_frame_{frame_idx:04d}.obj"
        mesh_vis_path = aligned_dir / mesh_vis_filename
        pytorch3d.io.save_obj(str(mesh_vis_path), aligned_world_verts, nerf_faces_pt)

        if verbose:
            print(f"Saved aligned full mesh visualization: {mesh_vis_path}")


        head_only_submesh = extract_head_only_mesh_vis(aligned_world_verts, nerf_faces_pt, head_vertex_indices)
        if head_only_submesh is not None:
            head_only_vis_filename = f"aligned_head_only_vis_frame_{frame_idx:04d}.obj"
            head_only_vis_path = aligned_dir / head_only_vis_filename
            head_only_submesh.export(str(head_only_vis_path))
            if verbose:
                print(f"Saved aligned head-only mesh visualization: {head_only_vis_path}")
                print(f"Head-only mesh: {len(head_only_submesh.vertices)} vertices, {len(head_only_submesh.faces)} faces")


        aligned_mesh_temp = pytorch3d.structures.Meshes(verts=[aligned_world_verts], faces=[nerf_faces_pt])
        aligned_normals = aligned_mesh_temp.verts_normals_packed()

        aligned_mesh_info = {
            'mesh_verts': aligned_world_verts,
            'mesh_norms': aligned_normals,
            'mesh_faces': nerf_faces_pt,
        }


        gs_model.update_to_posed_mesh(aligned_mesh_info)


        gs_vis_filename = f"aligned_gs_vis_frame_{frame_idx:04d}.ply"
        gs_vis_path = aligned_dir / gs_vis_filename
        gs_model.save_ply(str(gs_vis_path))

        if verbose:
            print(f"Saved aligned Gaussians visualization: {gs_vis_path}")
            print("Visualization debugging assets created successfully")
    elif posed_nerf_verts_tar is None and verbose:
        print("Skipping world-space visualization debugging - world-space vertices not provided")


    return str(transform_path)


def process_head_alignment(args, dat_dir, ref_nerf_verts_normed, nerf_faces_pt,
                          fwd_skinning_mats_nerf, gs_model,
                          global_orient_matrix, j0_src, src_scale, src_transl,
                          output_dir_base, frame_idx, posed_nerf_verts_tar=None, verbose=False,
                          label_dir_name='labeled', npz_suffix=''):

    try:

        cache_key = '_head_vertex_indices_cache'
        if not hasattr(args, cache_key):
            use_detailed_labels = getattr(args, 'use_detailed_labels', True)
            head_vertex_indices = load_head_vertices_for_subject(
                dat_dir, ref_nerf_verts_normed, nerf_faces_pt, use_detailed_labels, label_dir_name
            )
            setattr(args, cache_key, head_vertex_indices)
        else:
            head_vertex_indices = getattr(args, cache_key)


        head_transforms = fwd_skinning_mats_nerf[head_vertex_indices]


        if args.enable_head_alignment_filtering:
            avg_head_transform = compute_robust_average_transform(
                head_transforms, args.head_alignment_outlier_threshold
            )
        else:
            avg_head_transform = compute_average_transform(head_transforms)


        transform_path = save_head_alignment_transform(
            output_dir_base, frame_idx, avg_head_transform,
            global_orient_matrix, j0_src, src_scale, src_transl,
            posed_nerf_verts_tar, nerf_faces_pt, head_vertex_indices, gs_model,
            verbose=verbose, npz_suffix=npz_suffix
        )

        return transform_path

    except Exception as e:
        print(f"Head alignment failed for frame {frame_idx}: {e}")
        raise
