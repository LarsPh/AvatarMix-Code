import numpy as np
import trimesh
import pymeshlab
from pathlib import Path
from loguru import logger


def log_cleanup_statistics(cleanup_stats):

    logger.info("=== Non-head NeuS2 Mesh Cleanup Statistics ===")
    logger.info(f"Initial: {cleanup_stats['initial_vertices']} vertices, {cleanup_stats['initial_faces']} faces")
    logger.info(f"Final: {cleanup_stats['final_vertices']} vertices, {cleanup_stats['final_faces']} faces")
    logger.info(f"Vertices removed: {cleanup_stats['vertices_removed']}")
    logger.info(f"Faces removed: {cleanup_stats['faces_removed']} (including {cleanup_stats['self_intersecting_faces']} self-intersecting)")

    if cleanup_stats['faces_added'] > 0:
        logger.info(f"Faces added (hole filling): {cleanup_stats['faces_added']}")

    if cleanup_stats.get('unreferenced_vertices', 0) > 0:
        logger.info(f"Unreferenced vertices removed: {cleanup_stats['unreferenced_vertices']}")

    logger.success("Non-head mesh cleanup completed successfully")


def apply_pymeshlab_cleanup(trimesh_mesh):

    logger.info("Starting pymeshlab cleanup process...")


    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=trimesh_mesh.vertices,
        face_matrix=trimesh_mesh.faces
    ))


    initial_verts = ms.current_mesh().vertex_number()
    initial_faces = ms.current_mesh().face_number()

    logger.info(f"Initial mesh: {initial_verts} vertices, {initial_faces} faces")


    try:

        logger.info("Step 1: Computing selection by self-intersections per face...")
        ms.compute_selection_by_self_intersections_per_face()
        selected_faces_count = ms.current_mesh().selected_face_number()
        logger.info(f"Found {selected_faces_count} self-intersecting faces")


        if selected_faces_count > 0:
            logger.info("Step 2: Removing selected self-intersecting faces...")
            ms.meshing_remove_selected_faces()
            logger.info(f"Removed {selected_faces_count} self-intersecting faces")
        else:
            logger.info("Step 2: No self-intersecting faces to remove")


        logger.info("Step 4: Repairing non-manifold edges...")
        ms.meshing_repair_non_manifold_edges()
        logger.info("Repaired non-manifold edges")


        logger.info("Step 5: Closing holes...")
        faces_before_closing = ms.current_mesh().face_number()
        ms.meshing_close_holes(maxholesize=30)

        faces_added = ms.current_mesh().face_number() - faces_before_closing
        logger.info(f"Closed holes. Added {faces_added} faces")

    except Exception as e:
        logger.error(f"Pymeshlab processing error: {e}")
        return None, None


    final_verts = ms.current_mesh().vertex_number()
    final_faces = ms.current_mesh().face_number()


    cleanup_stats = {
        'initial_vertices': initial_verts,
        'initial_faces': initial_faces,
        'final_vertices': final_verts,
        'final_faces': final_faces,
        'vertices_removed': initial_verts - final_verts,
        'faces_removed': initial_faces - final_faces + selected_faces_count,
        'faces_added': max(0, final_faces - initial_faces + selected_faces_count),
        'self_intersecting_faces': selected_faces_count,

    }


    cleaned_vertices = ms.current_mesh().vertex_matrix()
    cleaned_faces = ms.current_mesh().face_matrix()

    if len(cleaned_vertices) == 0 or len(cleaned_faces) == 0:
        logger.error("Pymeshlab cleanup resulted in empty mesh")
        return None, None

    cleaned_trimesh = trimesh.Trimesh(vertices=cleaned_vertices, faces=cleaned_faces)

    logger.info("Pymeshlab cleanup completed successfully")
    return cleaned_trimesh, cleanup_stats


def apply_mesh_decimation(trimesh_mesh, target_percent=0.4):

    logger.info(f"Applying decimation (targetperc={target_percent})...")


    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(
        vertex_matrix=trimesh_mesh.vertices,
        face_matrix=trimesh_mesh.faces
    ))

    verts_before = ms.current_mesh().vertex_number()
    faces_before = ms.current_mesh().face_number()


    ms.meshing_decimation_quadric_edge_collapse(targetperc=target_percent)

    verts_after = ms.current_mesh().vertex_number()
    faces_after = ms.current_mesh().face_number()

    logger.info(f"Decimation: {verts_before}→{verts_after} vertices ({verts_after/verts_before:.1%}), "
                f"{faces_before}→{faces_after} faces ({faces_after/faces_before:.1%})")


    decimated_mesh = trimesh.Trimesh(
        vertices=ms.current_mesh().vertex_matrix(),
        faces=ms.current_mesh().face_matrix()
    )

    if len(decimated_mesh.vertices) == 0 or len(decimated_mesh.faces) == 0:
        raise ValueError("Decimation resulted in empty mesh")

    logger.success("Mesh decimation completed successfully")
    return decimated_mesh


def prepare_nerf_mesh_with_cleanup(nerf_mesh_path, seg_labels, surface_labels, identify_head_region_func, remove_head=False):

    logger.info(f"Loading NeuS2 mesh from: {nerf_mesh_path}")


    nerf_mesh = trimesh.load_mesh(str(nerf_mesh_path))
    logger.info(f"Loaded NeuS2 mesh: {len(nerf_mesh.vertices)} vertices, {len(nerf_mesh.faces)} faces")

    if remove_head:
        logger.info("Head removal ENABLED - extracting non-head region using 4D-Dress segmentation")


        head_vert_indices, head_face_indices = identify_head_region_func(
            nerf_mesh.vertices,
            nerf_mesh.faces,
            seg_labels,
            surface_labels,
            hair_label_name="hair",
            use_direct_head_label=True,
            direct_head_label_name="head",
            smpl_mesh_vertices=None,
            smplx_vert_seg=None,
            fallback_skin_label_name="skin"
        )

        if not head_vert_indices.size:
            raise ValueError("No head vertices found in NeuS2 mesh - cannot extract non-head region")

        logger.info(f"Identified {len(head_vert_indices)} head vertices, {len(head_face_indices)} head faces")


        all_vert_indices = np.arange(len(nerf_mesh.vertices))
        nonhead_vert_indices = np.setdiff1d(all_vert_indices, head_vert_indices)

        if not nonhead_vert_indices.size:
            raise ValueError("No non-head vertices found in NeuS2 mesh")

        logger.info(f"Found {len(nonhead_vert_indices)} non-head vertices")


        nonhead_faces_mask = np.all(np.isin(nerf_mesh.faces, nonhead_vert_indices), axis=1)

        if not np.any(nonhead_faces_mask):
            raise ValueError("No non-head faces found in NeuS2 mesh - all faces contain head vertices")

        logger.info(f"Found {np.sum(nonhead_faces_mask)} non-head faces")


        mesh_to_clean = nerf_mesh.submesh([nonhead_faces_mask], append=True)
        logger.info(f"Created non-head submesh: {len(mesh_to_clean.vertices)} vertices, {len(mesh_to_clean.faces)} faces")

    else:
        logger.info("Head removal DISABLED - cleaning full mesh (head + body)")
        logger.info("Rationale: Avoid 4D-Dress segmentation errors; head can be removed later in 2D if needed")
        mesh_to_clean = nerf_mesh


    cleaned_mesh, cleanup_stats = apply_pymeshlab_cleanup(mesh_to_clean)

    if cleaned_mesh is None:
        raise RuntimeError("Pymeshlab cleanup failed")

    return cleaned_mesh, cleanup_stats


def identify_skin_mask_binary(
    seg_labels: np.ndarray,
    surface_labels: list[str],
    no_torso_skin: bool = False,
) -> np.ndarray:

    skin_label_names = ["left_arm", "right_arm", "left_leg", "right_leg"]
    if not no_torso_skin:
        skin_label_names.append("torso_skin")


    skin_mask = np.zeros(len(seg_labels), dtype=np.float64)

    skin_vertex_count = 0
    for label_name in skin_label_names:
        try:
            label_idx = surface_labels.index(label_name)
            skin_verts = np.where(seg_labels == label_idx)[0]
            skin_mask[skin_verts] = 1.0
            skin_vertex_count += len(skin_verts)
            logger.info(f"  {label_name}: {len(skin_verts)} vertices")
        except ValueError:
            logger.warning(f"Label '{label_name}' not found in surface_labels")

    if skin_vertex_count == 0:
        raise ValueError("No skin vertices found in segmentation labels")

    logger.info(f"Created binary skin mask: {skin_vertex_count} skin vertices (1.0), {len(seg_labels) - skin_vertex_count} non-skin (0.0)")
    return skin_mask


def extract_skin_indices_from_mask(
    skin_mask_interpolated: np.ndarray,
    threshold: float = 0.5,
    allow_empty: bool = False,
) -> np.ndarray:

    logger.info(f"Extracting skin vertices with threshold={threshold} (allow_empty={allow_empty})")


    mask_min = float(np.min(skin_mask_interpolated)) if skin_mask_interpolated.size > 0 else float("nan")
    mask_max = float(np.max(skin_mask_interpolated)) if skin_mask_interpolated.size > 0 else float("nan")
    mask_mean = float(np.mean(skin_mask_interpolated)) if skin_mask_interpolated.size > 0 else float("nan")
    n_total = int(len(skin_mask_interpolated))
    n_pos = int(np.sum(skin_mask_interpolated > 0.0)) if skin_mask_interpolated.size > 0 else 0
    n_ge_01 = int(np.sum(skin_mask_interpolated > 0.1)) if skin_mask_interpolated.size > 0 else 0
    n_ge_05 = int(np.sum(skin_mask_interpolated > 0.5)) if skin_mask_interpolated.size > 0 else 0
    logger.info(
        f"Mask stats: n={n_total} min={mask_min:.3f} max={mask_max:.3f} mean={mask_mean:.3f} "
        f">0:{n_pos} >0.1:{n_ge_01} >0.5:{n_ge_05}"
    )

    skin_vertex_indices = np.where(skin_mask_interpolated > float(threshold))[0]
    n_skin = int(len(skin_vertex_indices))

    if n_skin == 0:
        msg = f"No skin vertices found after thresholding (threshold={threshold})"
        if allow_empty:
            logger.warning(msg + " -> returning empty indices")
            return np.array([], dtype=np.int64)
        raise ValueError(msg)

    logger.info(f"Extracted {n_skin} skin vertices ({100*n_skin/max(n_total,1):.1f}% of total {n_total})")
    return skin_vertex_indices.astype(np.int64)
