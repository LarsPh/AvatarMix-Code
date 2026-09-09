import cv2
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
    return joint_indices, lbs_weights


def store_joint_indices_as_attributes(ms: pymeshlab.MeshSet, joint_indices: np.ndarray,
                                    attribute_name: str = "joint_assignment") -> None:

    logger.info(f"Storing {len(joint_indices)} joint indices as vertex attribute '{attribute_name}'")


    joint_indices_float = joint_indices.astype(np.float64)

    try:

        ms.current_mesh().add_vertex_custom_scalar_attribute(joint_indices_float, attribute_name)
        logger.info(f"Successfully stored joint indices as vertex attribute")

    except Exception as e:
        logger.error(f"Failed to store vertex attributes: {e}")
        raise RuntimeError(f"PyMeshLab vertex attribute storage failed: {e}") from e


def retrieve_joint_indices_from_attributes(ms: pymeshlab.MeshSet,
                                         attribute_name: str = "joint_assignment") -> np.ndarray:

    try:

        joint_indices_float = ms.current_mesh().vertex_custom_scalar_attribute_array(attribute_name)


        joint_indices = np.round(joint_indices_float).astype(int)

        logger.info(f"Retrieved {len(joint_indices)} joint indices from vertex attributes")


        if np.any(joint_indices < 0):
            logger.warning(f"Found negative joint indices: {joint_indices[joint_indices < 0]}")
            joint_indices = np.clip(joint_indices, 0, None)

        return joint_indices

    except Exception as e:
        logger.error(f"Failed to retrieve vertex attributes: {e}")
        raise RuntimeError(f"PyMeshLab vertex attribute retrieval failed: {e}") from e


def load_smplx_cloth_fit_removal_indices(
    segmentation_json_path: Optional[Path] = None,
    remove_feet: bool = False,
    keep_palm: bool = False,
) -> np.ndarray:

    if segmentation_json_path is None:
        raise FileNotFoundError(
            "SMPL-X segmentation JSON path not provided. "
            "Caller must pass --smplx_segmentation_json_path argument."
        )

    segmentation_json_path = Path(segmentation_json_path)
    if not segmentation_json_path.exists():
        raise FileNotFoundError(f"SMPL-X segmentation JSON not found: {segmentation_json_path}")

    logger.info(f"Loading SMPL-X segmentation from: {segmentation_json_path}")

    import json
    with open(segmentation_json_path) as f:
        seg_data = json.load(f)


    left_hand_indices = np.array(seg_data["leftHand"])
    right_hand_indices = np.array(seg_data["rightHand"])
    left_hand_index1_indices = np.array(seg_data["leftHandIndex1"])
    right_hand_index1_indices = np.array(seg_data["rightHandIndex1"])


    if keep_palm:
        removal_indices = np.unique(
            np.concatenate([left_hand_index1_indices, right_hand_index1_indices])
        )
    else:
        removal_indices = np.unique(
            np.concatenate(
                [
                    left_hand_indices,
                    right_hand_indices,
                    left_hand_index1_indices,
                    right_hand_index1_indices,
                ]
            )
        )

    if remove_feet:
        logger.info(f"Removing feet for cloth-fit restoration")
        left_foot_indices = np.array(seg_data["leftFoot"])
        right_foot_indices = np.array(seg_data["rightFoot"])
        left_toe_base_indices = np.array(seg_data["leftToeBase"])
        right_toe_base_indices = np.array(seg_data["rightToeBase"])
        removal_indices = np.unique(np.concatenate(
            [
                removal_indices,
                left_foot_indices,
                right_foot_indices,
                left_toe_base_indices,
                right_toe_base_indices,
            ]
        ))

    logger.info(f"Loaded SMPL-X cloth fit removal regions (remove_feet={remove_feet}, keep_palm={keep_palm}):")
    logger.info(f"  Total unique: {len(removal_indices)} vertices")
    return removal_indices


def create_smplx_cloth_fit_removal_mask_binary(
    n_vertices: int,
    removal_indices: np.ndarray
) -> np.ndarray:


    if np.any(removal_indices < 0) or np.any(removal_indices >= n_vertices):
        invalid = removal_indices[(removal_indices < 0) | (removal_indices >= n_vertices)]
        raise ValueError(f"Cloth fit removal indices out of bounds [0, {n_vertices}): {invalid}")


    mask = np.zeros(n_vertices, dtype=np.float64)
    mask[removal_indices] = 1.0

    removal_count = np.sum(mask > 0.5)
    logger.info(f"Created SMPL-X binary cloth fit removal mask: {removal_count} cloth fit removal vertices (1.0), {n_vertices - removal_count} non-cloth fit removal (0.0)")

    return mask


def store_cloth_fit_removal_mask_as_attributes(
    ms: pymeshlab.MeshSet,
    removal_mask: np.ndarray,
    attribute_name: str = "smplx_cloth_fit_removal_mask"
) -> None:

    logger.info(f"Storing {len(removal_mask)} cloth fit removal mask values as vertex attribute '{attribute_name}'")

    try:

        ms.current_mesh().add_vertex_custom_scalar_attribute(removal_mask, attribute_name)
        removal_count = np.sum(removal_mask > 0.5)
        logger.info(f"Successfully stored cloth fit removal mask ({removal_count} cloth fit removal vertices)")

    except Exception as e:
        logger.error(f"Failed to store cloth fit removal mask attributes: {e}")
        raise RuntimeError(f"PyMeshLab cloth fit removal mask attribute storage failed: {e}") from e


def retrieve_cloth_fit_removal_mask_from_attributes(
    ms: pymeshlab.MeshSet,
    attribute_name: str = "smplx_cloth_fit_removal_mask",
    threshold: float = 0.5
) -> np.ndarray:

    try:

        removal_mask_interpolated = ms.current_mesh().vertex_custom_scalar_attribute_array(attribute_name)

        logger.info(f"Retrieved {len(removal_mask_interpolated)} cloth fit removal mask values from vertex attributes")


        removal_vertex_indices = np.where(removal_mask_interpolated > threshold)[0]

        if len(removal_vertex_indices) == 0:
            raise ValueError(f"No cloth fit removal vertices found after thresholding (threshold={threshold})")


        mask_min = np.min(removal_mask_interpolated)
        mask_max = np.max(removal_mask_interpolated)
        mask_mean = np.mean(removal_mask_interpolated)
        n_removal = len(removal_vertex_indices)
        n_total = len(removal_mask_interpolated)

        logger.info(f"Mask statistics: min={mask_min:.3f}, max={mask_max:.3f}, mean={mask_mean:.3f}")
        logger.info(f"Extracted {n_removal} cloth fit removal vertices ({100*n_removal/n_total:.1f}% of total {n_total})")

        return removal_vertex_indices

    except Exception as e:
        logger.error(f"Failed to retrieve cloth fit removal mask attributes: {e}")
        raise RuntimeError(f"PyMeshLab cloth fit removal mask attribute retrieval failed: {e}") from e


def apply_smpl_cleanup_pipeline(
    ms: pymeshlab.MeshSet,
    *,
    targetperc: float = 0.3,
    maxholesize: int = 80,
    post_decimation_hole_close: bool = True,
) -> Dict:

    logger.info("Starting extended SMPL cleanup pipeline...")


    initial_verts = ms.current_mesh().vertex_number()
    initial_faces = ms.current_mesh().face_number()
    logger.info(f"Initial mesh: {initial_verts} vertices, {initial_faces} faces")

    try:
        def _rebuild_meshset_for_stable_filters(*, preserve_scalar_attrs: tuple[str, ...]) -> None:

            V_tmp = ms.current_mesh().vertex_matrix()
            F_tmp = ms.current_mesh().face_matrix()

            preserved: dict[str, np.ndarray] = {}
            for name in preserve_scalar_attrs:
                try:
                    preserved[name] = ms.current_mesh().vertex_custom_scalar_attribute_array(name)
                except Exception:

                    pass

            ms.clear()
            ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V_tmp, face_matrix=F_tmp))

            for name, arr in preserved.items():
                try:
                    if len(arr) == ms.current_mesh().vertex_number():
                        ms.current_mesh().add_vertex_custom_scalar_attribute(arr, name)
                    else:
                        logger.warning(
                            f"Skipped preserving scalar attribute '{name}' during rebuild: "
                            f"len={len(arr)} != n_vertices={ms.current_mesh().vertex_number()}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to reattach scalar attribute '{name}' after rebuild: {e}")


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
        ms.meshing_close_holes(maxholesize=maxholesize)
        faces_after_closing = ms.current_mesh().face_number()
        faces_added = faces_after_closing - faces_before_closing
        logger.info(f"Closed holes, added {faces_added} faces")


        logger.info("Step 6: Applying quadric edge collapse decimation...")
        verts_before_decimation = ms.current_mesh().vertex_number()
        faces_before_decimation = ms.current_mesh().face_number()


        ms.meshing_decimation_quadric_edge_collapse(
            targetperc=targetperc,

            qualitythr=0.3,
            selected=False
        )

        verts_after_decimation = ms.current_mesh().vertex_number()
        faces_after_decimation = ms.current_mesh().face_number()

        decimation_vert_reduction = verts_before_decimation - verts_after_decimation
        decimation_face_reduction = faces_before_decimation - faces_after_decimation

        logger.info(f"Decimation completed: removed {decimation_vert_reduction} vertices ({verts_after_decimation}/{verts_before_decimation} = {verts_after_decimation/verts_before_decimation:.1%} remaining)")
        logger.info(f"Decimation completed: removed {decimation_face_reduction} faces ({faces_after_decimation}/{faces_before_decimation} = {faces_after_decimation/faces_before_decimation:.1%} remaining)")


        post_faces_added = 0
        post_unreferenced = 0
        if post_decimation_hole_close:
            logger.info("Step 7: Post-decimation watertight pass (repair + close holes)...")

            ms.meshing_repair_non_manifold_edges()

            verts_before_post_cleanup = ms.current_mesh().vertex_number()
            ms.meshing_remove_unreferenced_vertices()
            verts_after_post_cleanup = ms.current_mesh().vertex_number()
            post_unreferenced = verts_before_post_cleanup - verts_after_post_cleanup
            if post_unreferenced > 0:
                logger.info(f"Post-decimation: removed {post_unreferenced} unreferenced vertices")


            _rebuild_meshset_for_stable_filters(
                preserve_scalar_attrs=(
                    "joint_assignment",
                    "smplx_cloth_fit_removal_mask",
                )
            )

            faces_before_post_close = ms.current_mesh().face_number()


            ms.meshing_close_holes(maxholesize=maxholesize, selfintersection=False)
            faces_after_post_close = ms.current_mesh().face_number()
            post_faces_added = faces_after_post_close - faces_before_post_close
            logger.info(f"Post-decimation: closed holes, added {post_faces_added} faces")


            ms.meshing_remove_unreferenced_vertices()


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
            'post_decimation_faces_added_holes': post_faces_added,
            'post_decimation_unreferenced_vertices': post_unreferenced,
            'final_vertex_ratio': final_verts / initial_verts if initial_verts > 0 else 0,
            'final_face_ratio': final_faces / initial_faces if initial_faces > 0 else 0
        }

        logger.success(f"SMPL cleanup pipeline completed: {initial_verts}→{final_verts} vertices ({cleanup_stats['final_vertex_ratio']:.1%}), {initial_faces}→{final_faces} faces ({cleanup_stats['final_face_ratio']:.1%})")
        return cleanup_stats

    except Exception as e:
        logger.error(f"SMPL cleanup pipeline failed: {e}")
        raise RuntimeError(f"PyMeshLab SMPL cleanup failed: {e}") from e


def clean_smpl_body_mesh_for_volume(
    *,
    subject_root: Path,
    targetperc: float = 0.3,
    maxholesize: int = 50,
    overwrite: bool = True,
) -> Tuple[Optional[Path], Optional[Dict]]:

    subject_root = Path(subject_root)
    in_path = subject_root / "mesh" / "processed" / "smpl_body.obj"
    out_path = subject_root / "mesh" / "processed" / "smpl_body_cleaned.obj"

    if not in_path.exists():
        logger.warning(f"smpl_body.obj not found, skipping volume mesh cleaning: {in_path}")
        return None, None

    if out_path.exists() and not overwrite:
        logger.info(f"smpl_body_cleaned.obj exists and overwrite disabled, skipping: {out_path}")
        return out_path, None

    logger.info("=== Cleaning SMPL body mesh for volume computation ===")
    logger.info(f"Subject root: {subject_root}")
    logger.info(f"Input:  {in_path}")
    logger.info(f"Output: {out_path}")
    logger.info(f"Params: targetperc={targetperc}, maxholesize={maxholesize}, overwrite={overwrite}")

    try:
        body_mesh = trimesh.load_mesh(str(in_path), process=False)
        if not isinstance(body_mesh, trimesh.Trimesh):
            raise ValueError(f"Expected a Trimesh, got: {type(body_mesh)}")
        if len(body_mesh.vertices) == 0 or len(body_mesh.faces) == 0:
            raise ValueError("Input smpl_body.obj is empty (no vertices/faces)")

        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
            vertex_matrix=body_mesh.vertices,
            face_matrix=body_mesh.faces,
        ))

        stats = apply_smpl_cleanup_pipeline(
            ms,
            targetperc=float(targetperc),
            maxholesize=int(maxholesize),
            post_decimation_hole_close=True,
        )

        cleaned_vertices = ms.current_mesh().vertex_matrix()
        cleaned_faces = ms.current_mesh().face_matrix()
        cleaned_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces, process=False)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cleaned_mesh.export(str(out_path))
        logger.success(f"Saved smpl_body_cleaned.obj: {out_path}")
        return out_path, stats

    except Exception as e:
        logger.warning(f"Failed to generate smpl_body_cleaned.obj for {subject_root}: {e}")
        return None, None


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
                        original_mesh_path: Path, output_dir: Path, frame_idx: int,
                        original_mesh: trimesh.Trimesh = None,
                        lbs_weights_ori_one_hot: np.ndarray = None) -> Tuple[Path, Path]:


    cleaned_mesh_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned.obj"
    cleaned_lbs_path = output_dir / f"smpl_skeleton_lbs_weights_cleaned.txt"
    cleaned_mesh_ply_lbsvis_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned_lbsvis.ply"

    logger.info(f"Saving cleaned outputs: mesh={cleaned_mesh_path}, lbs={cleaned_lbs_path}, lbsvis_ply={cleaned_mesh_ply_lbsvis_path}")

    try:

        cleaned_mesh.export(str(cleaned_mesh_path))
        logger.info(f"Saved cleaned SMPL mesh: {len(cleaned_mesh.vertices)} vertices, {len(cleaned_mesh.faces)} faces")


        with open(cleaned_lbs_path, 'w') as f:
            for joint_idx, joint_weights in enumerate(lbs_weights):

                formatted_weights = [f'{weight:.5e}' for weight in joint_weights]
                f.write(' '.join(formatted_weights) + '\n')

        n_joints, n_vertices = lbs_weights.shape
        logger.info(f"Saved cleaned LBS weights: {n_joints} joints × {n_vertices} vertices in scientific notation")


        import cv2

        lbs_joint_argmax = np.argmax(lbs_weights, axis=0)
        max_lbs_weight = np.max(lbs_weights, axis=0)


        logger.info(f"Saved cleaned mesh with LBS weights as vertex color (PLY, cv2 colormap): {cleaned_mesh_ply_lbsvis_path}")

        return cleaned_mesh_path, cleaned_lbs_path

    except Exception as e:
        logger.error(f"Failed to save cleaned outputs: {e}")
        raise IOError(f"Cannot save cleaned files: {e}") from e

def save_ply_with_lbs_weights_visualization(mesh: trimesh.Trimesh, lbs_weights: np.ndarray, output_path: Path):

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

def clean_reposed_smpl_mesh_with_lbs_preservation(
    reposed_mesh_path: Path,
    source_subject_id: str,
    dataset_root: Path,
    frame_idx: int,
    output_dir: Path,
    smplx_segmentation_json_path: Optional[Path] = None,
    cloth_fit_remove_feet: bool = False,
    cloth_fit_keep_palm: bool = False,
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict]]:

    logger.info("=== Reposed SMPL Mesh Cleanup with LBS Weight Preservation ===")
    logger.info(f"Input mesh: {reposed_mesh_path}")
    logger.info(f"Source subject: {source_subject_id}")
    logger.info(f"Output directory: {output_dir}")

    try:


        lbs_weights_path = Path(dataset_root) / "smpl_skeleton_lbs_weights_from_313.txt"
        joint_indices, lbs_weights_ori_one_hot = load_lbs_weights_as_joint_indices(lbs_weights_path)
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


        removal_mask_stored = False
        try:
            smplx_cloth_fit_removal_indices = load_smplx_cloth_fit_removal_indices(
                smplx_segmentation_json_path,
                remove_feet=cloth_fit_remove_feet,
                keep_palm=bool(cloth_fit_keep_palm),
            )
            smplx_cloth_fit_removal_mask = create_smplx_cloth_fit_removal_mask_binary(
                n_vertices=len(reposed_mesh.vertices),
                removal_indices=smplx_cloth_fit_removal_indices
            )
            store_cloth_fit_removal_mask_as_attributes(ms, smplx_cloth_fit_removal_mask)
            removal_mask_stored = True
        except Exception as e:
            logger.warning(f"Failed to store SMPL-X cloth fit removal mask: {e}")
            logger.warning("Continuing without cloth fit removal mask (will not generate cloth fit removal indices file)")
            removal_mask_stored = False


        cleanup_stats = apply_smpl_cleanup_pipeline(ms)


        cleaned_joint_indices = retrieve_joint_indices_from_attributes(ms)
        lbs_weights_cleaned = reconstruct_lbs_weights_matrix(cleaned_joint_indices, original_n_joints)


        if removal_mask_stored:
            try:
                cleaned_removal_indices = retrieve_cloth_fit_removal_mask_from_attributes(ms)


                palm_suffix = "_keep_palm" if bool(cloth_fit_keep_palm) else ""
                removal_indices_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned_smplx_skin_indices{palm_suffix}.txt"
                np.savetxt(removal_indices_path, cleaned_removal_indices, fmt='%d')
                logger.success(f"Saved SMPL-X cloth fit removal indices: {removal_indices_path} ({len(cleaned_removal_indices)} vertices)")

            except Exception as e:
                logger.warning(f"Failed to save SMPL-X cloth fit removal indices: {e}")
                logger.warning("Continuing (cloth fit removal indices file not generated)")


        validation_results = validate_reconstructed_lbs_weights(lbs_weights_cleaned)
        cleanup_stats.update(validation_results)


        cleaned_vertices = ms.current_mesh().vertex_matrix()
        cleaned_faces = ms.current_mesh().face_matrix()

        if len(cleaned_vertices) == 0 or len(cleaned_faces) == 0:
            raise RuntimeError("Cleanup resulted in empty mesh")

        cleaned_mesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces)

        cleaned_mesh_path, cleaned_lbs_path = save_cleaned_outputs(
            cleaned_mesh, lbs_weights_cleaned, reposed_mesh_path, output_dir, frame_idx,
            original_mesh=reposed_mesh, lbs_weights_ori_one_hot=lbs_weights_ori_one_hot)


        logger.success("=== SMPL Mesh Cleanup Completed Successfully ===")
        logger.info(f"Original: {cleanup_stats['initial_vertices']} vertices → Cleaned: {cleanup_stats['final_vertices']} vertices ({cleanup_stats['final_vertex_ratio']:.1%})")
        logger.info(f"Original: {cleanup_stats['initial_faces']} faces → Cleaned: {cleanup_stats['final_faces']} faces ({cleanup_stats['final_face_ratio']:.1%})")
        logger.info(f"LBS weights validation: sums_valid={validation_results['vertex_sums_valid']}, one_hot_valid={validation_results['one_hot_valid']}")

        return cleaned_mesh_path, cleaned_lbs_path, cleanup_stats

    except Exception as e:
        logger.error(f"SMPL mesh cleanup with LBS preservation failed: {e}")
        logger.info("Original reposed mesh and skeleton files remain available")
        return None, None, None
