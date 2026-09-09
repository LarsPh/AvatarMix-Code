import os
import numpy as np
from loguru import logger


from .head_swap import (
    load_ply_gaussians, save_ply_gaussians, load_segmentation_labels,
    load_embedding_json, load_mesh, identify_head_region, get_labeled_part_verts_and_faces
)


def extract_aligned_head_for_subject(head_donator_id, body_donator_id, args, data_root,
                                   surface_labels, smplx_seg_path, use_direct_head_label,
                                   head_label_name, fallback_skin_name, hair_label_name,
                                   swapped_ply_name):

    logger.info(f"Extracting GT-aligned head: {head_donator_id} (head) from aligned Gaussians")


    aligned_gs_base = f"{data_root}/gs_on_mesh_repose/thuman2_{head_donator_id}_to_{body_donator_id}"
    aligned_gs_path = f"{aligned_gs_base}/aligned_head_assets/aligned_head_gs_frame_0000.ply"

    if not os.path.exists(aligned_gs_path):
        logger.warning(f"GT-aligned head Gaussians not found: {aligned_gs_path}")
        return

    logger.info(f"Loading GT-aligned Gaussians from: {aligned_gs_path}")


    aligned_gaussians, prop_names = load_ply_gaussians(aligned_gs_path)


    label_suffix = "_extended" if args.use_detailed_labels else ""

    segmentation_pkl_path = f"{data_root}/{get_padded_subject_id(head_donator_id)}/mesh/labeled/label-f0000{label_suffix}.pkl"
    if not os.path.exists(segmentation_pkl_path):
        logger.error(f"Segmentation file not found: {segmentation_pkl_path}")
        return


    iter_path = f"iteration_{args.user_A_iteration}" if args.user_A_iteration else "latest"
    embedding_path = f"{data_root}/output-splatting/neusclean_sub{get_padded_subject_id(head_donator_id)}/point_cloud/{iter_path}/embedding.json"

    if not os.path.exists(embedding_path):
        logger.error(f"Embedding file not found (required for head extraction): {embedding_path}")
        raise FileNotFoundError(f"Embedding file required but not found: {embedding_path}")

    scan_mesh_path = f"{data_root}/{get_padded_subject_id(head_donator_id)}/mesh/trimesh_cleaned/0000.obj"
    if not os.path.exists(scan_mesh_path):
        logger.error(f"Scan mesh not found: {scan_mesh_path}")
        return

    try:

        seg_labels = load_segmentation_labels(segmentation_pkl_path)
        sample_fidxs = load_embedding_json(embedding_path)
        scan_mesh = load_mesh(scan_mesh_path)


        smpl_mesh = None
        if not use_direct_head_label:
            smpl_mesh_path = f"{data_root}/gs_on_mesh_repose/thuman2_{head_donator_id}_to_{body_donator_id}/smpl_mesh_src_frame_0000.obj"
            if os.path.exists(smpl_mesh_path):
                smpl_mesh = load_mesh(smpl_mesh_path)


        head_vert_indices, head_face_indices = identify_head_region(
            scan_mesh.vertices, scan_mesh.faces, seg_labels, surface_labels,
            hair_label_name=hair_label_name,
            use_direct_head_label=use_direct_head_label,
            direct_head_label_name=head_label_name,
            smpl_mesh_vertices=smpl_mesh.vertices if smpl_mesh else None,
            smplx_vert_seg=None,
            fallback_skin_label_name=fallback_skin_name
        )

        if not head_vert_indices.size:
            logger.error(f"Could not identify head vertices for subject {head_donator_id}")
            return


        is_head_gaussian = np.isin(sample_fidxs, head_face_indices)
        head_gaussians_extracted = {name: data[is_head_gaussian] for name, data in aligned_gaussians.items()}

        num_head_gaussians = head_gaussians_extracted['x'].shape[0]
        logger.info(f"Extracted {num_head_gaussians} head Gaussians from GT-aligned source")

        if num_head_gaussians == 0:
            logger.warning(f"No head Gaussians found for subject {head_donator_id}")
            return


        output_filename = f"gt_aligned_head_{swapped_ply_name}.ply"


        reshape_suffix = f"_{args.repose_load_dir_subfix}" if args.repose_load_dir_subfix else ""
        swapped_dir = f"{data_root}/swapped/A{head_donator_id}_B{body_donator_id}{reshape_suffix}"
        if not os.path.exists(swapped_dir):
            os.makedirs(swapped_dir, exist_ok=True)

        output_path = os.path.join(swapped_dir, output_filename)


        save_ply_gaussians(output_path, head_gaussians_extracted, prop_names)

        logger.info(f"Successfully extracted GT-aligned head: {output_path}")

    except Exception as e:
        logger.error(f"Error extracting GT-aligned head for subject {head_donator_id}: {e}")
        raise


def get_padded_subject_id(subject_id):

    return f"{int(subject_id):04d}"
