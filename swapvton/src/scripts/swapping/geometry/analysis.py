import numpy as np
import trimesh
from scipy.spatial.distance import cdist


def get_smplx_head_center(smpl_mesh_vertices, smplx_vert_seg):

    head_vert_indices = smplx_vert_seg.get('head', [])
    if not head_vert_indices:


        head_vert_indices = smplx_vert_seg.get('face', [])

    if not head_vert_indices:
        raise ValueError("SMPLX head vertex indices not found or empty in segmentation file.")

    head_vertices = smpl_mesh_vertices[np.array(head_vert_indices)]
    return np.mean(head_vertices, axis=0)


def get_connected_components(mesh_vertices, mesh_faces, subset_vert_indices):

    if not isinstance(subset_vert_indices, np.ndarray):
        subset_vert_indices = np.array(subset_vert_indices)

    if subset_vert_indices.size == 0:
        return []


    if subset_vert_indices.dtype == bool:
        subset_vert_indices = np.where(subset_vert_indices)[0]
    else:

        subset_vert_indices = np.unique(np.asarray(subset_vert_indices, dtype=int))

    if not subset_vert_indices.size:
        return []


    try:
        mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    except Exception as e:
        print(f"Warning: Failed to create Trimesh object: {e}. Cannot find connected components via trimesh.")

        return [np.array([v_idx], dtype=int) for v_idx in subset_vert_indices]

    if len(mesh.vertices) == 0:
        print("Warning: Trimesh object created with no vertices. Cannot find connected components.")

        return [np.array([v_idx], dtype=int) for v_idx in subset_vert_indices]


    valid_mask = (subset_vert_indices >= 0) & (subset_vert_indices < len(mesh.vertices))
    current_subset_indices = subset_vert_indices[valid_mask]

    if not current_subset_indices.size:
        print("Warning: No valid subset_vert_indices remain after checking against mesh bounds.")
        return []

    if len(current_subset_indices) < len(subset_vert_indices):
        print("Warning: Some subset_vert_indices were out of bounds for the mesh vertices and were excluded.")

    try:


        edges = mesh.edges_unique


        components = trimesh.graph.connected_components(
            edges,
            nodes=current_subset_indices,
            min_len=1
        )


        return [np.array(c, dtype=int) for c in components]

    except ImportError:

        print("Error: networkx might be required by trimesh.graph.connected_components but not found. " +
              "Falling back to treating each subset vertex as a separate component.")
        return [np.array([v_idx], dtype=int) for v_idx in current_subset_indices]
    except Exception as e:
        print(f"Error using trimesh.graph.connected_components: {e}. " +
              "Falling back to treating each subset vertex as a separate component.")
        return [np.array([v_idx], dtype=int) for v_idx in current_subset_indices]

def build_label_to_indices_map(
    direct_label_names: list[str],
    current_surface_labels: list[str],
    segmentation_labels,
):

    label_to_indices = {}
    any_nonempty = False
    for label in direct_label_names:
        try:
            label_idx = current_surface_labels.index(label)
            indices = np.where(segmentation_labels == label_idx)[0]
            label_to_indices[label] = indices
            if indices.size == 0:
                print(f"Warning: Direct head label '{label}' (index {label_idx}) found no vertices.")
            else:
                any_nonempty = True
                print(f"Identified core part using direct label '{label}' (index {label_idx}): {len(indices)} vertices.")
        except ValueError:
            print(f"Error: Direct head label '{label}' not found in labels: {current_surface_labels}. Core part not identified by direct label.")
            continue
    return label_to_indices, any_nonempty

def compute_head_center_for_refinement(
    scan_mesh_vertices,
    current_surface_labels: list[str],
    segmentation_labels,
    label_to_indices: dict,
):

    head_indices = label_to_indices.get("head", np.array([], dtype=int))
    if head_indices.size > 0:
        return np.mean(scan_mesh_vertices[head_indices], axis=0)

    try:
        head_label_idx_global = current_surface_labels.index("head")
        head_indices_global = np.where(segmentation_labels == head_label_idx_global)[0]
        if head_indices_global.size > 0:
            print("Info: 'head' not in direct labels; used global 'head' label to compute head center.")
            return np.mean(scan_mesh_vertices[head_indices_global], axis=0)
        else:
            print("Warning: Global 'head' label yielded no vertices; cannot compute head center for torso_skin refinement.")
            return None
    except ValueError:
        print("Warning: 'head' label not found globally; cannot compute head center for torso_skin refinement.")
        return None

def select_nearest_component_to_center(
    mesh_vertices,
    mesh_faces,
    candidate_vert_indices,
    target_center,
    label_name_for_logs: str,
):

    if target_center is None or candidate_vert_indices.size == 0:
        return None, None

    components = get_connected_components(mesh_vertices, mesh_faces, candidate_vert_indices)
    if not components and candidate_vert_indices.size > 0:
        print(f"Warning: Could not find connected components for '{label_name_for_logs}' vertices. Treating as single component.")
        components = [candidate_vert_indices]

    if not components:
        return None, None

    closest_idx = -1
    min_dist = float('inf')
    selected_component = None
    for i, comp_indices in enumerate(components):
        if not comp_indices.size:
            continue
        avg_pos = np.mean(mesh_vertices[comp_indices], axis=0)
        dist = np.linalg.norm(avg_pos - target_center)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
            selected_component = comp_indices

    if closest_idx == -1:
        return None, None
    return selected_component, min_dist

def union_label_indices_excluding(
    label_to_indices: dict,
    exclude_labels: list[str],
):

    arrays = [v for k, v in label_to_indices.items() if k not in exclude_labels and isinstance(v, np.ndarray) and v.size > 0]
    if arrays:
        return np.unique(np.concatenate(arrays))
    return np.array([], dtype=int)

def identify_head_region(
    scan_mesh_vertices,
    scan_mesh_faces,
    segmentation_labels,
    current_surface_labels: list[str],
    hair_label_name: str = "hair",
    use_direct_head_label: bool = False,
    direct_head_label_name: list[str] = ["head"],
    smpl_mesh_vertices=None,
    smplx_vert_seg=None,
    fallback_skin_label_name: str = "skin",
    neck_tube_mask: np.ndarray | None = None,
    above_neck_plane_mask: np.ndarray | None = None,
):

    core_head_part_verts = np.array([], dtype=int)


    core_head_non_torso_verts = np.array([], dtype=int)
    core_torso_skin_component_verts = np.array([], dtype=int)


    if use_direct_head_label:
        label_to_indices, any_nonempty = build_label_to_indices_map(
            direct_head_label_name,
            current_surface_labels,
            segmentation_labels,
        )

        if any_nonempty:
            if "torso_skin" in direct_head_label_name:
                head_center = compute_head_center_for_refinement(
                    scan_mesh_vertices,
                    current_surface_labels,
                    segmentation_labels,
                    label_to_indices,
                )

                torso_skin_indices = label_to_indices.get("torso_skin", np.array([], dtype=int))


                selected_torso_component = None
                min_dist = None
                if neck_tube_mask is not None and torso_skin_indices.size > 0:
                    try:
                        tube_subset = torso_skin_indices[neck_tube_mask[torso_skin_indices]]
                    except Exception:
                        tube_subset = np.array([], dtype=int)
                    if tube_subset.size > 0:
                        selected_torso_component = tube_subset
                        min_dist = 0.0
                        print(f"Refined 'torso_skin' to neck_tube_mask: {len(selected_torso_component)} vertices.")


                if selected_torso_component is None:
                    selected_torso_component, min_dist = select_nearest_component_to_center(
                        scan_mesh_vertices,
                        scan_mesh_faces,
                        torso_skin_indices,
                        head_center,
                        "torso_skin",
                    )

                if selected_torso_component is not None and selected_torso_component.size > 0:
                    core_head_non_torso_verts = union_label_indices_excluding(label_to_indices, ["torso_skin"])
                    core_torso_skin_component_verts = np.unique(np.asarray(selected_torso_component, dtype=int))
                    if core_head_non_torso_verts.size > 0:
                        core_head_part_verts = np.unique(
                            np.concatenate([core_head_non_torso_verts, core_torso_skin_component_verts])
                        )
                    else:
                        core_head_part_verts = np.unique(core_torso_skin_component_verts)


                else:
                    print("Warning: Failed to identify nearest torso_skin component or head center unavailable. Falling back to direct union of labels.")
                    core_head_part_verts = union_label_indices_excluding(label_to_indices, [])
            else:
                core_head_part_verts = union_label_indices_excluding(label_to_indices, [])

            if "torso_skin" not in direct_head_label_name:
                core_head_non_torso_verts = core_head_part_verts
        else:

            core_head_part_verts = np.array([], dtype=int)
    else:
        if smpl_mesh_vertices is None or smplx_vert_seg is None:
            print("Error: SMPL mesh data not provided for fallback (SMPLX-based) core part identification.")
        else:
            try:
                skin_label_idx = current_surface_labels.index(fallback_skin_label_name)
                print(f"Using SMPLX-based fallback to identify core part with skin label '{fallback_skin_label_name}' (index {skin_label_idx}).")
                smplx_head_center = get_smplx_head_center(smpl_mesh_vertices, smplx_vert_seg)
                skin_vert_indices = np.where(segmentation_labels == skin_label_idx)[0]

                if not skin_vert_indices.size:
                    print(f"Warning: No skin vertices found for label '{fallback_skin_label_name}' (index {skin_label_idx}).")
                else:
                    skin_components = get_connected_components(scan_mesh_vertices, scan_mesh_faces, skin_vert_indices)
                    if not skin_components:
                        print(f"Warning: Could not find connected components for '{fallback_skin_label_name}' vertices.")
                        if skin_vert_indices.size > 0: skin_components = [skin_vert_indices]

                    if skin_components:
                        closest_component_idx = -1
                        min_dist = float('inf')
                        identified_skin_component_verts = np.array([], dtype=int)
                        for i, component_indices in enumerate(skin_components):
                            if not component_indices.size: continue
                            component_verts_coords = scan_mesh_vertices[component_indices]
                            avg_pos = np.mean(component_verts_coords, axis=0)
                            dist = np.linalg.norm(avg_pos - smplx_head_center)
                            if dist < min_dist:
                                min_dist = dist
                                closest_component_idx = i
                                identified_skin_component_verts = component_indices

                        if closest_component_idx != -1:
                            core_head_part_verts = identified_skin_component_verts
                            core_head_non_torso_verts = core_head_part_verts
                            print(f"Identified core part (skin component via fallback) with {len(core_head_part_verts)} verts, closest to SMPLX head (dist: {min_dist:.4f}).")
                        else:
                            print(f"Warning: Could not identify core part (skin component via fallback). No suitable '{fallback_skin_label_name}' component found.")
                    else:
                         print(f"Warning: No skin components derived from '{fallback_skin_label_name}' vertices for fallback method.")
            except ValueError:
                print(f"Error: Fallback skin label '{fallback_skin_label_name}' not found in surface labels: {current_surface_labels}.")

        if core_head_non_torso_verts.size == 0 and core_head_part_verts.size > 0:
            core_head_non_torso_verts = core_head_part_verts


    hair_vert_indices = np.array([], dtype=int)
    try:
        hair_label_idx = current_surface_labels.index(hair_label_name)
        hair_vert_indices = np.where(segmentation_labels == hair_label_idx)[0]
        if hair_vert_indices.size > 0:
            print(f"Found {len(hair_vert_indices)} vertices for hair label '{hair_label_name}' (index {hair_label_idx}).")


    except ValueError:
        print(f"Info: Hair label '{hair_label_name}' not found in surface labels: {current_surface_labels}. Hair will not be included.")


    final_head_total_verts = np.array([], dtype=int)
    if core_head_part_verts.size > 0 and hair_vert_indices.size > 0:
        final_head_total_verts = np.union1d(core_head_part_verts, hair_vert_indices)
        print(f"Combined core head/skin ({len(core_head_part_verts)}) and hair ({len(hair_vert_indices)}) -> {len(final_head_total_verts)} total vertices.")
    elif core_head_part_verts.size > 0:
        final_head_total_verts = core_head_part_verts
        print(f"Using core head/skin part only ({len(final_head_total_verts)} vertices) as hair was not significantly present or specified.")
    elif hair_vert_indices.size > 0:
        final_head_total_verts = hair_vert_indices
        print(f"Using hair part only ({len(final_head_total_verts)} vertices) as core head/skin part was not identified.")
    else:
        print("Warning: No core head/skin vertices AND no hair vertices were identified. Returning empty head region.")
        return np.array([]), np.array([])


    if above_neck_plane_mask is not None and final_head_total_verts.size > 0:
        try:

            filtered_torso = core_torso_skin_component_verts
            if core_torso_skin_component_verts.size > 0:
                filtered_torso = core_torso_skin_component_verts[
                    above_neck_plane_mask[core_torso_skin_component_verts]
                ]
                if filtered_torso.size == 0:
                    filtered_torso = core_torso_skin_component_verts
                    print(
                        "Warning: above_neck_plane_mask removed all torso_skin-component vertices; "
                        "keeping unfiltered torso_skin component."
                    )


            parts = []
            if core_head_non_torso_verts.size > 0:
                parts.append(core_head_non_torso_verts)
            if filtered_torso.size > 0:
                parts.append(filtered_torso)
            if hair_vert_indices.size > 0:
                parts.append(hair_vert_indices)
            if parts:
                final_head_total_verts = np.unique(np.concatenate(parts))

            print(
                f"Applied above_neck_plane_mask to torso_skin-component only: head verts -> {len(final_head_total_verts)}"
            )
        except Exception as e:
            print(f"Warning: Failed to apply above_neck_plane_mask: {e}")


    final_head_faces_indices = np.array([], dtype=int)
    if final_head_total_verts.size > 0:
        head_face_mask = np.all(np.isin(scan_mesh_faces, final_head_total_verts), axis=1)
        final_head_faces_indices = np.where(head_face_mask)[0]
        if not final_head_faces_indices.size and final_head_total_verts.size > 0:

            print(f"Warning: Combined head parts ({len(final_head_total_verts)} verts) resulted in no faces where all face vertices are within this set.")
        elif final_head_faces_indices.size > 0:
            print(f"Found {len(final_head_faces_indices)} faces for the combined head region.")


    return final_head_total_verts, final_head_faces_indices


def get_labeled_part_verts_and_faces(
    segmentation_labels,
    scan_mesh_faces,
    current_surface_labels: list[str],
    target_label_name: str
):

    try:
        label_idx = current_surface_labels.index(target_label_name)
        part_verts = np.where(segmentation_labels == label_idx)[0]
        if not part_verts.size:
            print(f"Warning: Label '{target_label_name}' (index {label_idx}) found no vertices.")
            return np.array([], dtype=int), np.array([], dtype=int)

        print(f"Found {len(part_verts)} vertices for label '{target_label_name}' (index {label_idx}).")
        part_face_mask = np.all(np.isin(scan_mesh_faces, part_verts), axis=1)
        part_faces = np.where(part_face_mask)[0]

        if not part_faces.size and part_verts.size > 0:
            print(f"Warning: Label '{target_label_name}' found vertices but no corresponding faces (where all face vertices have this label).")
        elif part_faces.size > 0:
            print(f"Found {len(part_faces)} faces for label '{target_label_name}'.")

        return part_verts, part_faces
    except ValueError:
        print(f"Error: Target label '{target_label_name}' not found in surface labels list: {current_surface_labels}.")
        return np.array([], dtype=int), np.array([], dtype=int)
