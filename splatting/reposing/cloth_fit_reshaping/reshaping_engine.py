import torch
import pytorch3d.io
import numpy as np
from pathlib import Path
from loguru import logger

from .coordinate_transforms import compute_source_params_from_raw_skeleton, restore_source
from model.reshaping.body_reshape import calc_nearest_neighbor

DEBUG_CLOTH_FIT_RESHAPING = True


def apply_cloth_fit_reshaping(args, device, nerf_verts_original, nerf_faces_original, head_vertex_indices):

    logger.info("Starting cloth-fit reshaping process...")


    all_indices = torch.arange(nerf_verts_original.shape[0], device=device)
    body_vertex_indices = torch.tensor([i for i in all_indices if i not in head_vertex_indices], device=device)

    logger.info(f"Vertex separation - Total: {nerf_verts_original.shape[0]}, Head: {len(head_vertex_indices)}, Body: {len(body_vertex_indices)}")


    dataset_root = Path(args.dat_dir).parent
    subject_id = Path(args.dat_dir).stem

    original_simplified_path = dataset_root / subject_id / "mesh" / "processed" / args.cloth_fit_original_mesh_basename
    simplified_mesh_path = dataset_root / subject_id / "mesh" / "processed" / args.cloth_fit_simplified_mesh_basename
    raw_skeleton_path = dataset_root / subject_id / "mesh" / "processed" / "smpl_skeleton.obj"
    deformed_simplified_path = Path(args.cloth_fit_deformed_mesh_path)


    required_files = [
        (original_simplified_path, "original simplified mesh"),
        (simplified_mesh_path, "simplified mesh"),
        (raw_skeleton_path, "raw skeleton"),
        (deformed_simplified_path, "deformed simplified mesh")
    ]

    for file_path, description in required_files:
        if not file_path.exists():
            raise FileNotFoundError(f"{description} not found: {file_path}")

    logger.info(f"Loading cloth-fit meshes for subject {subject_id}...")


    orig_body_verts, orig_body_faces_idx, _ = pytorch3d.io.load_obj(str(simplified_mesh_path), device=device)
    orig_body_faces = orig_body_faces_idx.verts_idx
    logger.info(f"Loaded original simplified mesh: {orig_body_verts.shape[0]} vertices")


    orig_body_verts_from_full = nerf_verts_original[body_vertex_indices]


    if orig_body_verts_from_full.shape[0] != len(body_vertex_indices):
        raise ValueError(
            f"Body vertex count mismatch: "
            f"cloth-fit simplified mesh has {orig_body_verts_from_full.shape[0]} vertices, "
            f"but original mesh has {len(body_vertex_indices)} body vertices"
        )


    skeleton_verts, _, _ = pytorch3d.io.load_obj(str(raw_skeleton_path), device=device)
    skeleton_np = skeleton_verts.cpu().numpy()
    logger.info(f"Loaded raw skeleton: {skeleton_verts.shape[0]} vertices")


    def_body_verts_norm, def_body_faces_idx, _ = pytorch3d.io.load_obj(str(deformed_simplified_path), device=device)
    logger.info(f"Loaded deformed simplified mesh: {def_body_verts_norm.shape[0]} vertices")


    if orig_body_verts.shape[0] != def_body_verts_norm.shape[0]:
        raise ValueError(
            f"Vertex count mismatch between simplified meshes: "
            f"original({orig_body_verts.shape[0]}) vs deformed({def_body_verts_norm.shape[0]})"
        )


    src_params = None
    if getattr(args, "cloth_fit_deformed_mesh_already_restored", False):
        logger.info("Cloth-fit deformed mesh marked as already restored: skipping restore_source()")
        def_body_verts_restored = def_body_verts_norm
    else:
        logger.info("Computing coordinate transformation parameters...")
        src_params = compute_source_params_from_raw_skeleton(skeleton_np)
        logger.info(f"Transformation params - center_offset: {src_params['center_offset']}, scaling: {src_params['source_scaling']}")

        def_body_verts_restored_np = restore_source(
            def_body_verts_norm.cpu().numpy(),
            src_params["center_offset"],
            src_params["source_scaling"]
        )
        def_body_verts_restored = torch.from_numpy(def_body_verts_restored_np.astype(np.float32)).to(device)


    body_vertex_offsets = def_body_verts_restored - orig_body_verts
    logger.info(f"Computed body vertex offsets, max magnitude: {torch.norm(body_vertex_offsets, dim=-1).max().item():.6f}")


    logger.info("Propagating offsets to original body vertices using barycentric interpolation...")
    propagated_body_offsets, _ = calc_nearest_neighbor(
        query_pts=orig_body_verts_from_full.unsqueeze(0),
        ref_v=orig_body_verts.unsqueeze(0),
        ref_f=orig_body_faces.unsqueeze(0),
        weights=body_vertex_offsets.unsqueeze(0),
        method='barycentric',
        nn_scale_factor=1000.0
    )

    propagated_body_offsets = propagated_body_offsets.squeeze(0)
    logger.info(f"Propagated offsets to {propagated_body_offsets.shape[0]} body vertices")


    cloth_fit_reshaped_body_verts = orig_body_verts_from_full + propagated_body_offsets


    reshaped_full_verts = nerf_verts_original.clone()
    reshaped_full_verts[body_vertex_indices] = cloth_fit_reshaped_body_verts


    debug_dir = Path(args.output_dir) / "cloth_fit_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if DEBUG_CLOTH_FIT_RESHAPING:

        head_only_verts = nerf_verts_original[head_vertex_indices]
        head_debug_path = debug_dir / f"head_only_{subject_id}.obj"

        pytorch3d.io.save_obj(str(head_debug_path), head_only_verts, torch.empty((0, 3), dtype=torch.long, device=device))
        logger.info(f"Saved head-only vertices: {head_debug_path}")


        body_only_debug_path = debug_dir / f"body_only_original_{subject_id}.obj"
        pytorch3d.io.save_obj(str(body_only_debug_path), orig_body_verts_from_full, torch.empty((0, 3), dtype=torch.long, device=device))
        logger.info(f"Saved original body-only vertices: {body_only_debug_path}")


        body_reshaped_debug_path = debug_dir / f"body_only_cloth_fit_{subject_id}.obj"
        pytorch3d.io.save_obj(str(body_reshaped_debug_path), cloth_fit_reshaped_body_verts, torch.empty((0, 3), dtype=torch.long, device=device))
        logger.info(f"Saved cloth-fit reshaped body-only vertices: {body_reshaped_debug_path}")


        restored_debug_path = debug_dir / f"deformed_simplified_restored_{subject_id}.obj"
        pytorch3d.io.save_obj(str(restored_debug_path), def_body_verts_restored, orig_body_faces)
        logger.info(f"Saved restored deformed simplified mesh: {restored_debug_path}")


        original_full_debug_path = debug_dir / f"full_mesh_original_{subject_id}.obj"
        pytorch3d.io.save_obj(str(original_full_debug_path), nerf_verts_original, nerf_faces_original)
        logger.info(f"Saved original full mesh: {original_full_debug_path}")

        reshaped_full_debug_path = debug_dir / f"full_mesh_cloth_fit_reshaped_{subject_id}.obj"
        pytorch3d.io.save_obj(str(reshaped_full_debug_path), reshaped_full_verts, nerf_faces_original)
        logger.info(f"Saved cloth-fit reshaped full mesh: {reshaped_full_debug_path}")


        if src_params is not None:
            target_skeleton_debug_path = Path(args.cloth_fit_deformed_mesh_path).parent / "target_skeleton.obj"
            target_skeleton_verts, _, _ = pytorch3d.io.load_obj(str(target_skeleton_debug_path), device=device)
            target_skeleton_verts_restored = restore_source(
                target_skeleton_verts.cpu().numpy(),
                src_params["center_offset"],
                src_params["source_scaling"]
            )
            target_skeleton_verts_restored = torch.from_numpy(target_skeleton_verts_restored.astype(np.float32)).to(device)
            target_skeleton_debug_path = debug_dir / f"target_skeleton_restored_{subject_id}.obj"
            pytorch3d.io.save_obj(str(target_skeleton_debug_path), target_skeleton_verts_restored, torch.empty((0, 3), dtype=torch.long, device=device))
            logger.info(f"Saved restored target skeleton: {target_skeleton_debug_path}")
        else:
            logger.info("Skipping target skeleton restoration debug (no src_params in already-restored mode).")


    offset_magnitude = torch.norm(body_vertex_offsets, dim=-1)
    logger.info(f"Body vertex offset magnitudes - min: {offset_magnitude.min().item():.6f}, max: {offset_magnitude.max().item():.6f}, mean: {offset_magnitude.mean().item():.6f}")


    propagated_offset_magnitude = torch.norm(propagated_body_offsets, dim=-1)
    logger.info(f"Propagated offset magnitudes - min: {propagated_offset_magnitude.min().item():.6f}, max: {propagated_offset_magnitude.max().item():.6f}, mean: {propagated_offset_magnitude.mean().item():.6f}")
    logger.info(f"Head/body vertex separation - head: {len(head_vertex_indices)}, body: {len(body_vertex_indices)}, total: {nerf_verts_original.shape[0]}")

    logger.info("Cloth-fit reshaping completed successfully")
    return reshaped_full_verts
