import numpy as np
import trimesh
import pymeshlab
from pathlib import Path
from loguru import logger
from typing import Tuple, Dict, Optional, Union


def resolve_source_lbs_path(source_subject_id: str, dataset_root: Union[str, Path]) -> Path:


    padded_id = source_subject_id.zfill(4)
    lbs_path = Path(dataset_root) / padded_id / "mesh" / "processed" / "smpl_skeleton_lbs_weights.txt"

    logger.debug(f"Resolved source LBS weights path: {lbs_path}")
    return lbs_path


def load_lbs_weights_as_joint_indices(lbs_weights_path: Path) -> np.ndarray:

    if not lbs_weights_path.exists():
        raise FileNotFoundError(f"Source LBS weights file not found: {lbs_weights_path}")

    logger.info(f"Loading LBS weights from: {lbs_weights_path}")


    lbs_weights = np.loadtxt(str(lbs_weights_path))

    if len(lbs_weights.shape) != 2:
        raise ValueError(f"Expected 2D LBS weights array, got shape: {lbs_weights.shape}")

    n_joints, n_vertices = lbs_weights.shape
    logger.info(f"Loaded LBS weights: {n_joints} joints × {n_vertices} vertices")


    joint_indices = np.argmax(lbs_weights, axis=0)


    max_weights = np.max(lbs_weights, axis=0)
    if not np.allclose(max_weights, 1.0, atol=1e-6):
        logger.warning(f"LBS weights don't appear to be max-weight assigned (max values: {np.unique(max_weights)})")


    non_zero_counts = np.count_nonzero(lbs_weights, axis=0)
    if not np.all(non_zero_counts == 1):
        unique_counts = np.unique(non_zero_counts)
        logger.warning(f"LBS weights don't have one-hot property (non-zero counts: {unique_counts})")

    logger.info(f"Converted to joint indices: {len(joint_indices)} vertices, joint range [0, {n_joints-1}]")
    return joint_indices


def store_joint_indices_as_attributes(ms: pymeshlab.MeshSet, joint_indices: np.ndarray,
                                    attribute_name: str = "joint_assignment") -> None:

    logger.info(f"Storing {len(joint_indices)} joint indices as vertex attribute '{attribute_name}'")


    joint_indices_float = joint_indices.astype(np.float64)

    try:

        ms.add_vertex_custom_scalar_attribute(joint_indices_float, attribute_name)
        logger.info(f"Successfully stored joint indices as vertex attribute")

    except Exception as e:
        logger.error(f"Failed to store vertex attributes: {e}")
        raise RuntimeError(f"PyMeshLab vertex attribute storage failed: {e}") from e


def retrieve_joint_indices_from_attributes(ms: pymeshlab.MeshSet,
                                         attribute_name: str = "joint_assignment") -> np.ndarray:

    try:

        joint_indices_float = ms.vertex_custom_scalar_attribute_array(attribute_name)


        joint_indices = np.round(joint_indices_float).astype(int)

        logger.info(f"Retrieved {len(joint_indices)} joint indices from vertex attributes")


        if np.any(joint_indices < 0):
            logger.warning(f"Found negative joint indices: {joint_indices[joint_indices < 0]}")
            joint_indices = np.clip(joint_indices, 0, None)

        return joint_indices

    except Exception as e:
        logger.error(f"Failed to retrieve vertex attributes: {e}")
        raise RuntimeError(f"PyMeshLab vertex attribute retrieval failed: {e}") from e


def load_smplx_skin_indices(
    segmentation_json_path: Path = None
) -> np.ndarray:

    if segmentation_json_path is None:

        segmentation_json_path = Path(__file__).parent.parent.parent / \
            "utils" / "smplx_utils" / "smplx_models" / "smplx" / "smplx_vert_segmentation.json"

    if not segmentation_json_path.exists():
        raise FileNotFoundError(f"SMPL-X segmentation JSON not found: {segmentation_json_path}")

    logger.info(f"Loading SMPL-X segmentation from: {segmentation_json_path}")

    import json
    with open(segmentation_json_path) as f:
        seg_data = json.load(f)


    neck_indices = np.array(seg_data["neck"])
    left_hand_indices = np.array(seg_data["leftHand"])
    right_hand_indices = np.array(seg_data["rightHand"])


    skin_indices = np.unique(np.concatenate([neck_indices, left_hand_indices, right_hand_indices]))

    logger.info(f"Loaded SMPL-X skin regions:")
    logger.info(f"  Neck: {len(neck_indices)} vertices")
    logger.info(f"  Left hand: {len(left_hand_indices)} vertices")
    logger.info(f"  Right hand: {len(right_hand_indices)} vertices")
    logger.info(f"  Total unique: {len(skin_indices)} vertices")

    return skin_indices


def create_smplx_skin_mask_binary(
    n_vertices: int,
    skin_indices: np.ndarray
) -> np.ndarray:


    if np.any(skin_indices < 0) or np.any(skin_indices >= n_vertices):
        invalid = skin_indices[(skin_indices < 0) | (skin_indices >= n_vertices)]
        raise ValueError(f"Skin indices out of bounds [0, {n_vertices}): {invalid}")


    mask = np.zeros(n_vertices, dtype=np.float64)
    mask[skin_indices] = 1.0

    skin_count = np.sum(mask > 0.5)
    logger.info(f"Created SMPL-X binary skin mask: {skin_count} skin vertices (1.0), {n_vertices - skin_count} non-skin (0.0)")

    return mask


def store_skin_mask_as_attributes(
    ms: pymeshlab.MeshSet,
    skin_mask: np.ndarray,
    attribute_name: str = "smplx_skin_mask"
) -> None:

    logger.info(f"Storing {len(skin_mask)} skin mask values as vertex attribute '{attribute_name}'")

    try:

        ms.add_vertex_custom_scalar_attribute(skin_mask, attribute_name)
        skin_count = np.sum(skin_mask > 0.5)
        logger.info(f"Successfully stored skin mask ({skin_count} skin vertices)")

    except Exception as e:
        logger.error(f"Failed to store skin mask attributes: {e}")
        raise RuntimeError(f"PyMeshLab skin mask attribute storage failed: {e}") from e


def retrieve_skin_mask_from_attributes(
    ms: pymeshlab.MeshSet,
    attribute_name: str = "smplx_skin_mask",
    threshold: float = 0.5
) -> np.ndarray:

    try:

        skin_mask_interpolated = ms.vertex_custom_scalar_attribute_array(attribute_name)

        logger.info(f"Retrieved {len(skin_mask_interpolated)} skin mask values from vertex attributes")


        skin_vertex_indices = np.where(skin_mask_interpolated > threshold)[0]

        if len(skin_vertex_indices) == 0:
            raise ValueError(f"No skin vertices found after thresholding (threshold={threshold})")


        mask_min = np.min(skin_mask_interpolated)
        mask_max = np.max(skin_mask_interpolated)
        mask_mean = np.mean(skin_mask_interpolated)
        n_skin = len(skin_vertex_indices)
        n_total = len(skin_mask_interpolated)

        logger.info(f"Mask statistics: min={mask_min:.3f}, max={mask_max:.3f}, mean={mask_mean:.3f}")
        logger.info(f"Extracted {n_skin} skin vertices ({100*n_skin/n_total:.1f}% of total {n_total})")

        return skin_vertex_indices

    except Exception as e:
        logger.error(f"Failed to retrieve skin mask attributes: {e}")
        raise RuntimeError(f"PyMeshLab skin mask attribute retrieval failed: {e}") from e


def apply_smpl_cleanup_pipeline(ms: pymeshlab.MeshSet) -> Dict:

    logger.info("Starting extended SMPL cleanup pipeline...")


    initial_verts = ms.current_mesh().vertex_number()
    initial_faces = ms.current_mesh().face_number()
    logger.info(f"Initial mesh: {initial_verts} vertices, {initial_faces} faces")

    try:

        logger.info("Step 1: Computing selection by self-intersections per face...")
        ms.compute_selection_by_self_intersections_per_face()
        selected_faces_count = ms.current_mesh().selected_face_number()
        logger.info(f"Found {selected_faces_count} self-intersecting faces")

        if selected_faces_count > 0:
            logger.info("Step 1b: Removing selected self-intersecting faces...")
            ms.meshing_remove_selected_faces()
            logger.info(f"Removed {selected_faces_count} self-intersecting faces")
        else:
            logger.info("Step 1b: No self-intersecting faces to remove")


        logger.info("Step 2: Removing unreferenced vertices...")
        verts_before_cleanup = ms.current_mesh().vertex_number()
        ms.meshing_remove_unreferenced_vertices()
        verts_after_cleanup = ms.current_mesh().vertex_number()
        unreferenced_count = verts_before_cleanup - verts_after_cleanup

        if unreferenced_count > 0:
            logger.info(f"Removed {unreferenced_count} unreferenced vertices")
        else:
            logger.info("No unreferenced vertices found")


        logger.info("Step 3: Repairing non-manifold edges...")
        ms.meshing_repair_non_manifold_edges()
        logger.info("Repaired non-manifold edges")


        logger.info("Step 4: Removing small disconnected components...")
        verts_before_components = ms.current_mesh().vertex_number()
        faces_before_components = ms.current_mesh().face_number()

        ms.compute_selection_by_small_disconnected_components_per_face(nbfaceratio=0.9)
        selected_component_faces = ms.current_mesh().selected_face_number()

        if selected_component_faces > 0:
            logger.info(f"Found {selected_component_faces} faces in small components to remove")
            ms.meshing_remove_selected_vertices_and_faces()

            verts_after_components = ms.current_mesh().vertex_number()
            faces_after_components = ms.current_mesh().face_number()

            removed_verts = verts_before_components - verts_after_components
            removed_faces = faces_before_components - faces_after_components
            logger.info(f"Removed small components: {removed_verts} vertices, {removed_faces} faces")
        else:
            logger.info("No small disconnected components found")


        logger.info("Step 5: Closing holes...")
        faces_before_closing = ms.current_mesh().face_number()
        ms.meshing_close_holes(maxholesize=50)
        faces_after_closing = ms.current_mesh().face_number()
        faces_added = faces_after_closing - faces_before_closing
        logger.info(f"Closed holes, added {faces_added} faces")


        logger.info("Step 6: Applying quadric edge collapse decimation...")
        verts_before_decimation = ms.current_mesh().vertex_number()
        faces_before_decimation = ms.current_mesh().face_number()

        ms.meshing_decimation_quadric_edge_collapse(targetperc=0.4)

        verts_after_decimation = ms.current_mesh().vertex_number()
        faces_after_decimation = ms.current_mesh().face_number()

        decimation_vert_reduction = verts_before_decimation - verts_after_decimation
        decimation_face_reduction = faces_before_decimation - faces_after_decimation

        logger.info(f"Decimation completed: removed {decimation_vert_reduction} vertices ({verts_after_decimation}/{verts_before_decimation} = {verts_after_decimation/verts_before_decimation:.1%} remaining)")
        logger.info(f"Decimation completed: removed {decimation_face_reduction} faces ({faces_after_decimation}/{faces_before_decimation} = {faces_after_decimation/faces_before_decimation:.1%} remaining)")


        logger.info("Step 7: Closing holes (after decimation)...")
        faces_before_closing = ms.current_mesh().face_number()
        ms.meshing_close_holes(maxholesize=30)
        faces_after_closing = ms.current_mesh().face_number()
        faces_added = faces_after_closing - faces_before_closing
        logger.info(f"Closed holes, added {faces_added} faces")


        final_verts = ms.current_mesh().vertex_number()
        final_faces = ms.current_mesh().face_number()


        cleanup_stats = {
            'initial_vertices': initial_verts,
            'initial_faces': initial_faces,
            'final_vertices': final_verts,
            'final_faces': final_faces,
            'total_vertices_removed': initial_verts - final_verts,
            'total_faces_removed': initial_faces - final_faces,
            'self_intersecting_faces': selected_faces_count,
            'unreferenced_vertices': unreferenced_count,
            'small_component_faces': selected_component_faces,
            'faces_added_holes': faces_added,
            'decimation_vertices_removed': decimation_vert_reduction,
            'decimation_faces_removed': decimation_face_reduction,
            'final_vertex_ratio': final_verts / initial_verts if initial_verts > 0 else 0,
            'final_face_ratio': final_faces / initial_faces if initial_faces > 0 else 0
        }

        logger.success(f"SMPL cleanup pipeline completed: {initial_verts}→{final_verts} vertices ({cleanup_stats['final_vertex_ratio']:.1%}), {initial_faces}→{final_faces} faces ({cleanup_stats['final_face_ratio']:.1%})")
        return cleanup_stats

    except Exception as e:
        logger.error(f"SMPL cleanup pipeline failed: {e}")
        raise RuntimeError(f"PyMeshLab SMPL cleanup failed: {e}") from e


def reconstruct_lbs_weights_matrix(joint_indices: np.ndarray, n_joints: int) -> np.ndarray:

    n_vertices = len(joint_indices)
    logger.info(f"Reconstructing LBS weights: {n_joints} joints × {n_vertices} vertices")


    if np.any(joint_indices < 0) or np.any(joint_indices >= n_joints):
        invalid_indices = joint_indices[(joint_indices < 0) | (joint_indices >= n_joints)]
        logger.warning(f"Invalid joint indices found: {invalid_indices} (valid range: [0, {n_joints-1}])")

        joint_indices = np.clip(joint_indices, 0, n_joints - 1)


    lbs_weights = np.zeros((n_joints, n_vertices), dtype=np.float64)


    for vertex_idx, joint_idx in enumerate(joint_indices):
        lbs_weights[int(joint_idx), vertex_idx] = 1.0

    logger.info(f"Reconstructed LBS weights matrix shape: {lbs_weights.shape}")
    return lbs_weights


def validate_reconstructed_lbs_weights(lbs_weights: np.ndarray) -> Dict:

    logger.info("Validating reconstructed LBS weights...")

    n_joints, n_vertices = lbs_weights.shape


    vertex_sums = np.sum(lbs_weights, axis=0)
    sum_tolerance = 1e-6
    sum_check = np.allclose(vertex_sums, 1.0, atol=sum_tolerance)


    non_zero_counts = np.count_nonzero(lbs_weights, axis=0)
    one_hot_check = np.all(non_zero_counts == 1)


    mean_vertex_sum = np.mean(vertex_sums)
    vertex_sum_std = np.std(vertex_sums)
    unique_non_zero_counts = np.unique(non_zero_counts)

    validation_results = {
        'vertex_sums_valid': sum_check,
        'one_hot_valid': one_hot_check,
        'mean_vertex_sum': mean_vertex_sum,
        'vertex_sum_std': vertex_sum_std,
        'unique_non_zero_counts': unique_non_zero_counts.tolist(),
        'n_vertices': n_vertices,
        'n_joints': n_joints
    }


    if sum_check:
        logger.info(f"✓ Vertex sums valid: mean={mean_vertex_sum:.6f}, std={vertex_sum_std:.6f}")
    else:
        logger.warning(f"✗ Vertex sums invalid: mean={mean_vertex_sum:.6f}, std={vertex_sum_std:.6f} (expected: mean=1.0, std~0.0)")

    if one_hot_check:
        logger.info(f"✓ One-hot property valid: all vertices have exactly 1 joint assigned")
    else:
        logger.warning(f"✗ One-hot property invalid: non-zero counts per vertex: {unique_non_zero_counts}")

    if sum_check and one_hot_check:
        logger.success("LBS weights validation passed")
    else:
        logger.warning("LBS weights validation failed - results may be inaccurate")

    return validation_results


def save_cleaned_outputs(cleaned_mesh: trimesh.Trimesh, lbs_weights: np.ndarray,
                        original_mesh_path: Path, output_dir: Path, frame_idx: int) -> Tuple[Path, Path]:


    cleaned_mesh_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned.obj"
    cleaned_lbs_path = output_dir / f"smpl_skeleton_lbs_weights_cleaned.txt"

    logger.info(f"Saving cleaned outputs: mesh={cleaned_mesh_path}, lbs={cleaned_lbs_path}")

    try:

        cleaned_mesh.export(str(cleaned_mesh_path))
        logger.info(f"Saved cleaned SMPL mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")


        with open(cleaned_lbs_path, 'w') as f:
            for joint_idx, joint_weights in enumerate(lbs_weights):

                formatted_weights = [f'{weight:.5e}' for weight in joint_weights]
                f.write(' '.join(formatted_weights) + '\n')

        n_joints, n_vertices = lbs_weights.shape
        logger.info(f"Saved cleaned LBS weights: {n_joints} joints × {n_vertices} vertices in scientific notation")

        return cleaned_mesh_path, cleaned_lbs_path

    except Exception as e:
        logger.error(f"Failed to save cleaned outputs: {e}")
        raise IOError(f"Cannot save cleaned files: {e}") from e


def clean_reposed_smpl_mesh_with_lbs_preservation(
    reposed_mesh_path: Path,
    source_subject_id: str,
    dataset_root: Path,
    frame_idx: int,
    output_dir: Path
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict]]:

    logger.info("=== Reposed SMPL Mesh Cleanup with LBS Weight Preservation ===")
    logger.info(f"Input mesh: {reposed_mesh_path}")
    logger.info(f"Source subject: {source_subject_id}")
    logger.info(f"Output directory: {output_dir}")

    try:

        lbs_weights_path = resolve_source_lbs_path(source_subject_id, dataset_root)
        joint_indices = load_lbs_weights_as_joint_indices(lbs_weights_path)
        original_n_joints = np.max(joint_indices) + 1


        logger.info(f"Loading reposed SMPL mesh: {reposed_mesh_path}")
        reposed_mesh = trimesh.load_mesh(str(reposed_mesh_path))
        logger.info(f"Loaded mesh: {len(reposed_mesh.vertices)} vertices, {len(reposed_mesh.faces)} faces")


        if len(reposed_mesh.vertices) != len(joint_indices):
            raise ValueError(f"Mesh vertex count ({len(reposed_mesh.vertices)}) doesn't match LBS vertex count ({len(joint_indices)})")


        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=reposed_mesh.vertices,
            face_matrix=reposed_mesh.faces
        ))

        store_joint_indices_as_attributes(ms, joint_indices)


        skin_mask_stored = False
        try:
            smplx_skin_indices = load_smplx_skin_indices()
            smplx_skin_mask = create_smplx_skin_mask_binary(
                n_vertices=len(reposed_mesh.vertices),
                skin_indices=smplx_skin_indices
            )
            store_skin_mask_as_attributes(ms, smplx_skin_mask)
            skin_mask_stored = True
        except Exception as e:
            logger.warning(f"Failed to store SMPL-X skin mask: {e}")
            logger.warning("Continuing without skin mask (will not generate skin indices file)")
            skin_mask_stored = False


        cleanup_stats = apply_smpl_cleanup_pipeline(ms)


        cleaned_joint_indices = retrieve_joint_indices_from_attributes(ms)
        lbs_weights_cleaned = reconstruct_lbs_weights_matrix(cleaned_joint_indices, original_n_joints)


        if skin_mask_stored:
            try:
                cleaned_skin_indices = retrieve_skin_mask_from_attributes(ms)


                skin_indices_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned_smplx_skin_indices.txt"
                np.savetxt(skin_indices_path, cleaned_skin_indices, fmt='%d')
                logger.success(f"Saved SMPL-X skin indices: {skin_indices_path} ({len(cleaned_skin_indices)} vertices)")

            except Exception as e:
                logger.warning(f"Failed to save SMPL-X skin indices: {e}")
                logger.warning("Continuing (skin indices file not generated)")


        validation_results = validate_reconstructed_lbs_weights(lbs_weights_cleaned)
        cleanup_stats.update(validation_results)


        cleaned_vertices = ms.current_mesh().vertex_matrix()
        cleaned_faces = ms.current_mesh().face_matrix()

        if len(cleaned_vertices) == 0 or len(cleaned_faces) == 0:
            raise RuntimeError("Cleanup resulted in empty mesh")

        cleaned_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces)

        cleaned_mesh_path, cleaned_lbs_path = save_cleaned_outputs(
            cleaned_mesh, lbs_weights_cleaned, reposed_mesh_path, output_dir, frame_idx)


        logger.success("=== SMPL Mesh Cleanup Completed Successfully ===")
        logger.info(f"Original: {cleanup_stats['initial_vertices']} vertices → Cleaned: {cleanup_stats['final_vertices']} vertices ({cleanup_stats['final_vertex_ratio']:.1%})")
        logger.info(f"Original: {cleanup_stats['initial_faces']} faces → Cleaned: {cleanup_stats['final_faces']} faces ({cleanup_stats['final_face_ratio']:.1%})")
        logger.info(f"LBS weights validation: sums_valid={validation_results['vertex_sums_valid']}, one_hot_valid={validation_results['one_hot_valid']}")

        return cleaned_mesh_path, cleaned_lbs_path, cleanup_stats

    except Exception as e:
        logger.error(f"SMPL mesh cleanup with LBS preservation failed: {e}")
        logger.info("Original reposed mesh and skeleton files remain available")
        return None, None, None
