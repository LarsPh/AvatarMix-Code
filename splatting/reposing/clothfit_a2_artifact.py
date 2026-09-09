from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger

from reposing.cloth_fit_utils.smplx_trim import (
    assert_faces_identical,
    build_keep_vertex_mask,
    load_smplx_mesh,
    save_trimmed_mesh,
    trim_smplx_mesh,
)

from reposing.utils.repose_common import (
    _load_obj_vertices,
    _load_obj_mesh,
    _vertex_colors_from_face_ids,
    _vertex_colors_from_tri_ids,
)


def _compute_a2_correspondence_lines(
    *,
    query_points_xyz: np.ndarray,
    body_vertices_full: np.ndarray,
    body_faces_full: np.ndarray,
    full_to_trim_face: np.ndarray,
    nn_scale_factor: float,
    chunk_size: int = 20000,
) -> tuple[list[str], np.ndarray]:

    import torch
    from model.reshaping.reshape_utils import nearest_face_pytorch3d

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("A2 correspondence requires CUDA (reshape_ops.nearest_face_pytorch3d).")

    pts = torch.from_numpy(query_points_xyz).to(device=device, dtype=torch.float32)
    V = torch.from_numpy(np.asarray(body_vertices_full)).to(device=device, dtype=torch.float32)
    F = torch.from_numpy(np.asarray(body_faces_full)).to(device=device, dtype=torch.int64)

    full_to_trim_face = np.asarray(full_to_trim_face, dtype=np.int64)

    lines: list[str] = []
    n = int(pts.shape[0])
    tri_ids_all = np.empty((n,), dtype=np.int64)
    for start in range(0, n, int(chunk_size)):
        end = min(n, start + int(chunk_size))
        pts_chunk = pts[start:end][None, ...]
        Vb = V[None, ...]

        with torch.no_grad():
            dists, face_idx, bc = nearest_face_pytorch3d(
                pts_chunk,
                Vb,
                F,
                scale_factor=float(nn_scale_factor),
            )
        d = dists[0].detach().cpu().numpy()
        fi_full = face_idx[0].detach().cpu().numpy().astype(np.int64)
        bc_np = bc[0].detach().cpu().numpy()

        tri_trim = full_to_trim_face[fi_full]
        tri_ids_all[start:end] = tri_trim
        for tri_id, dist, (b0, b1, b2) in zip(tri_trim.tolist(), d.tolist(), bc_np.tolist()):
            if tri_id < 0:
                lines.append("-1\n")
            else:
                lines.append(f"{tri_id} {dist:.10g} {b0:.10g} {b1:.10g} {b2:.10g}\n")

    if len(lines) != int(n):
        raise RuntimeError(f"A2 correspondence line count mismatch: got {len(lines)}, expected {int(n)}")
    return lines, tri_ids_all


def _write_a2_visualizations(
    *,
    out_dir: Path,
    corr_stem: str,
    garment_mesh_path: Path,
    tri_ids_per_garment_vertex: np.ndarray,
    source_body_trim_vertices: np.ndarray,
    source_body_trim_faces: np.ndarray,
):
    import trimesh
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


    Vg, Fg = _load_obj_mesh(garment_mesh_path)
    rgb_g = _vertex_colors_from_tri_ids(tri_ids_per_garment_vertex)
    if rgb_g.shape[0] != Vg.shape[0]:
        raise ValueError(f"A2 vis: tri_id count {rgb_g.shape[0]} != garment verts {Vg.shape[0]}")
    mg = trimesh.Trimesh(vertices=Vg, faces=Fg, process=False)
    mg.visual.vertex_colors = np.concatenate([rgb_g, np.full((rgb_g.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    garment_vis_path = out_dir / f"{corr_stem}_vis_garment.ply"
    mg.export(str(garment_vis_path))


    Vb = np.asarray(source_body_trim_vertices, dtype=np.float64)
    Fb = np.asarray(source_body_trim_faces, dtype=np.int64)
    face_ids = np.arange(Fb.shape[0], dtype=np.int64)
    rgb_b = _vertex_colors_from_face_ids(n_vertices=Vb.shape[0], faces=Fb, face_ids=face_ids)
    mb = trimesh.Trimesh(vertices=Vb, faces=Fb, process=False)
    mb.visual.vertex_colors = np.concatenate([rgb_b, np.full((rgb_b.shape[0], 1), 255, dtype=np.uint8)], axis=1)
    body_vis_path = out_dir / f"{corr_stem}_vis_source_body_trim.ply"
    mb.export(str(body_vis_path))

    logger.info(f"Saved A2 visualization meshes: {garment_vis_path} and {body_vis_path}")


def generate_a2_artifacts_for_frame(
    *,
    args,
    reposed_mesh_path: Path,
    output_dir: Path,
    frame_idx: int,
) -> None:

    if not (getattr(args, "a2_generate_trim_meshes", False) or getattr(args, "a2_generate_correspondence", False)):
        return

    if not args.smplx_segmentation_json_path:
        raise ValueError("A2 requested but --smplx_segmentation_json_path is missing.")

    garment_root = Path(args.target_pose_dir)
    seg_json = Path(args.smplx_segmentation_json_path)
    remove_feet = bool(getattr(args, "cloth_fit_remove_feet", False))
    keep_palm = bool(getattr(args, "cloth_fit_keep_palm", False))
    overwrite = bool(getattr(args, "a2_overwrite", False))


    smpl_body_full_path = garment_root / "mesh" / "processed" / "smpl_body.obj"
    smpl_body_trim_path = garment_root / "mesh" / "processed" / "smpl_body_trim.obj"


    smpl_reposed_trim_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_trim.obj"


    Vs, Fs = load_smplx_mesh(smpl_body_full_path)
    Vt, Ft = load_smplx_mesh(reposed_mesh_path)
    assert_faces_identical(faces_a=Fs, faces_b=Ft, label_a=str(smpl_body_full_path), label_b=str(reposed_mesh_path))

    keep_mask = build_keep_vertex_mask(
        n_vertices=Vs.shape[0],
        smplx_segmentation_json_path=seg_json,
        cloth_fit_remove_feet=remove_feet,
        cloth_fit_keep_palm=keep_palm,
    )


    trim_src = trim_smplx_mesh(vertices_full=Vs, faces_full=Fs, keep_vertex_mask=keep_mask)
    trim_tgt = trim_smplx_mesh(vertices_full=Vt, faces_full=Ft, keep_vertex_mask=keep_mask)
    if not np.array_equal(trim_src.faces, trim_tgt.faces):
        raise ValueError("A2 requires trimmed source/target meshes to share identical face ordering; got mismatch after trimming.")

    if getattr(args, "a2_generate_trim_meshes", False):
        save_trimmed_mesh(smpl_body_trim_path, vertices=trim_src.vertices, faces=trim_src.faces, overwrite=overwrite)
        save_trimmed_mesh(smpl_reposed_trim_path, vertices=trim_tgt.vertices, faces=trim_tgt.faces, overwrite=overwrite)


    if getattr(args, "a2_generate_correspondence", False):
        garment_mesh_path = garment_root / "mesh" / "processed" / "nerf_simp_cleaned.obj"
        query_xyz = _load_obj_vertices(garment_mesh_path)

        corr_out = str(getattr(args, "a2_corr_output_path", "") or "").strip()
        if corr_out:
            p = Path(corr_out)

            corr_path = (output_dir / p.name) if (not p.is_absolute() and p.parent == Path(".")) else p
        else:
            corr_path = output_dir / "a2_corr.txt"

        if corr_path.exists() and not overwrite:
            logger.info(f"A2 correspondence exists and overwrite disabled, skipping: {corr_path}")
            return

        nn_scale_factor = float(getattr(args, "a2_nn_scale_factor", 1000.0))
        lines, tri_ids = _compute_a2_correspondence_lines(
            query_points_xyz=query_xyz,
            body_vertices_full=Vs,
            body_faces_full=Fs,
            full_to_trim_face=trim_src.full_to_trim_face,
            nn_scale_factor=nn_scale_factor,
        )
        corr_path.parent.mkdir(parents=True, exist_ok=True)
        corr_path.write_text("".join(lines))
        n_disabled = sum(1 for ln in lines if ln.strip() == "-1")
        logger.info(
            f"Saved A2 correspondence: {corr_path} (lines={len(lines)}, disabled={n_disabled}, nn_scale_factor={nn_scale_factor})"
        )

        if getattr(args, "a2_visualize_correspondence", False):
            corr_stem = corr_path.stem
            _write_a2_visualizations(
                out_dir=corr_path.parent,
                corr_stem=corr_stem,
                garment_mesh_path=garment_mesh_path,
                tri_ids_per_garment_vertex=tri_ids,
                source_body_trim_vertices=trim_src.vertices,
                source_body_trim_faces=trim_src.faces,
            )
