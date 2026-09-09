from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import pytorch3d.io
import pytorch3d.structures
from tqdm import tqdm
from loguru import logger

from gaussian_renderer import network_gui
from model.splatting_avatar_model import SplattingAvatarModel
from model.smplx_utils import smplx_ani as smplx
from model.reshaping.body_reshape import (
    NeighborVisialzier,
    calc_nearest_neighbor,
    calc_smoothed_nearest_neighbor,
)

from reposing.cloth_fit_reshaping import apply_cloth_fit_reshaping
from reposing.utils.coordinate_transforms import transform_to_canonical_space, transform_to_world_space
from reposing.utils.smpl_utils import (
    get_betas_for_reposing,
    get_v_pose_for_reposing,
    get_v_shape_for_reposing,
)
from reposing.utils.repose_common import (
    ACTORSHQ_SMPLX_CONFIG,
    AVATARREX_SMPLX_CONFIG,
    THUMAN2_SMPLX_CONFIG,
    _resolve_gender_for_dir,
)

def reshape_nerf_in_source_pose(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    if args.visualize_reshaping:
        visualizer = NeighborVisialzier()

    subject_root = Path(args.target_pose_dir)
    smpl_model_path = args.smpl_model_path
    output_dir_base = Path(args.output_dir)
    output_dir_base.mkdir(parents=True, exist_ok=True)


    logger.info(f"Loading source NeRF mesh from: {args.nerf_mesh_path}")
    try:
        nerf_mesh_path_str = str(args.nerf_mesh_path)
        if nerf_mesh_path_str.endswith('.obj'):
            nerf_verts_pt, faces_idx, _ = pytorch3d.io.load_obj(nerf_mesh_path_str, device=device)
            nerf_faces_pt = faces_idx.verts_idx
        elif nerf_mesh_path_str.endswith('.ply'):
            loaded_ply = pytorch3d.io.load_ply(nerf_mesh_path_str, device=device)
            nerf_verts_pt = loaded_ply[0]
            nerf_faces_pt = loaded_ply[1]
        else:
            raise ValueError(f"Unsupported mesh file format: {nerf_mesh_path_str}. Please use .obj or .ply.")
        logger.info(f"Loaded NeRF mesh: {nerf_verts_pt.shape[0]} verts, {nerf_faces_pt.shape[0]} faces")
    except Exception as e:
        logger.error(f"Error loading NeRF mesh: {e}")
        return


    logger.info(f"Loading SMPL parameters from: {subject_root / 'smpl_params.npz'}")
    try:
        smpl_data_npz_tar = np.load(str(subject_root / 'smpl_params.npz'), allow_pickle=True)
        smpl_data_tar = {k: torch.from_numpy(v.astype(np.float32)).to(device) for k,v in smpl_data_npz_tar.items()}
        smpl_data_npz_src = np.load(str(Path(args.dat_dir) / 'smpl_params.npz'), allow_pickle=True)
        smpl_data_src = {k: torch.from_numpy(v.astype(np.float32)).to(device) for k,v in smpl_data_npz_src.items()}
    except Exception as e:
        logger.error(f"Error loading SMPL parameters: {e}")
        return


    logger.info(f"Initializing SMPL-X models from: {smpl_model_path}")
    is_talkbody4d_tar = "talkbody4d" in str(args.target_pose_dir)
    is_talkbody4d_src = "talkbody4d" in str(args.dat_dir)
    gender_tar = _resolve_gender_for_dir(subject_root, args.smpl_gender_tar, force_neutral=is_talkbody4d_tar)
    gender_src = _resolve_gender_for_dir(Path(args.dat_dir), args.smpl_gender_src, force_neutral=is_talkbody4d_src)

    if "thuman2" in args.target_pose_dir:
        smpl_config_tar = THUMAN2_SMPLX_CONFIG
    elif "actorshq" in args.target_pose_dir or "mvhumannet" in args.target_pose_dir:
        smpl_config_tar = ACTORSHQ_SMPLX_CONFIG
    else:
        smpl_config_tar = AVATARREX_SMPLX_CONFIG

    if "thuman2" in args.dat_dir:
        smpl_config_src = THUMAN2_SMPLX_CONFIG
    elif "actorshq" in args.dat_dir or "mvhumannet" in args.dat_dir:
        smpl_config_src = ACTORSHQ_SMPLX_CONFIG
    else:
        smpl_config_src = AVATARREX_SMPLX_CONFIG

    num_betas_tar = int(smpl_data_tar["betas"].shape[-1]) if "betas" in smpl_data_tar else 10
    num_betas_src = int(smpl_data_src["betas"].shape[-1]) if "betas" in smpl_data_src else 10
    num_expr_tar = int(smpl_data_tar["expression"].shape[-1]) if "expression" in smpl_data_tar else 10
    num_expr_src = int(smpl_data_src["expression"].shape[-1]) if "expression" in smpl_data_src else 10

    try:
        smpl_model_tar = smplx.SMPLX(
            model_path=smpl_model_path,
            gender=gender_tar,
            use_pca=smpl_config_tar['use_pca'],
            num_pca_comps=smpl_config_tar['num_pca_comps'],
            flat_hand_mean=smpl_config_tar['flat_hand_mean'],
            num_betas=num_betas_tar,
            num_expression_coeffs=num_expr_tar,
            batch_size=1,
        ).to(device)
        smpl_model_src = smplx.SMPLX(
            model_path=smpl_model_path,
            gender=gender_src,
            use_pca=smpl_config_src['use_pca'],
            num_pca_comps=smpl_config_src['num_pca_comps'],
            flat_hand_mean=smpl_config_src['flat_hand_mean'],
            num_betas=num_betas_src,
            num_expression_coeffs=num_expr_src,
            batch_size=1,
        ).to(device)
    except Exception as e:
        logger.error(f"Error initializing SMPL-X model: {e}")
        return


    with torch.no_grad():
        src_frame_idx = args.nerf_mesh_source_frame_idx

        if is_talkbody4d_src and ("Rh" in smpl_data_src and "Th" in smpl_data_src):
            src_global_orient = smpl_data_src["Rh"][src_frame_idx:src_frame_idx+1]
            src_transl = smpl_data_src["Th"][src_frame_idx:src_frame_idx+1]
        else:
            src_global_orient = smpl_data_src['global_orient'][src_frame_idx:src_frame_idx+1]
            src_transl = smpl_data_src['transl'][src_frame_idx:src_frame_idx+1]
        src_scale = smpl_data_src.get('scale')
        src_scale = src_scale[src_frame_idx:src_frame_idx+1] if src_scale is not None else torch.ones(1, device=device)

        smpl_params_src_pose = {
            'body_pose': smpl_data_src['body_pose'][src_frame_idx:src_frame_idx+1],
            'jaw_pose': smpl_data_src.get('jaw_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_jaw_pose_coeffs if hasattr(smpl_model_src, 'num_jaw_pose_coeffs') else 3, device=device))[src_frame_idx:src_frame_idx+1],
            'expression': smpl_data_src.get('expression', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_expression_coeffs if hasattr(smpl_model_src, 'num_expression_coeffs') else 10, device=device))[src_frame_idx:src_frame_idx+1],
            'left_hand_pose': smpl_data_src.get('left_hand_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_left_hand_pose_coeffs if hasattr(smpl_model_src, 'num_left_hand_pose_coeffs') else 15*3, device=device))[src_frame_idx:src_frame_idx+1],
            'right_hand_pose': smpl_data_src.get('right_hand_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_right_hand_pose_coeffs if hasattr(smpl_model_src, 'num_right_hand_pose_coeffs') else 15*3, device=device))[src_frame_idx:src_frame_idx+1],
            'v_shape': smpl_data_src.get('v_shape', None)[src_frame_idx:src_frame_idx+1] if smpl_data_src.get('v_shape', None) is not None else None,
            'v_pose': smpl_data_src.get('v_pose', None)[src_frame_idx:src_frame_idx+1] if smpl_data_src.get('v_pose', None) is not None else None,
        }
        smpl_params_src_pose_filtered = {k: v for k, v in smpl_params_src_pose.items() if v is not None}
        smpl_out_src = smpl_model_src.forward(**smpl_params_src_pose_filtered)
        j0_src = smpl_out_src.joints[0, 0]


    gs_model = SplattingAvatarModel(config.model, verbose=True, gaussians_are_frozen=True)
    if not args.input_gs_ply:
        logger.error("Path to input Gaussian Splatting PLY (--input_gs_ply) is required.")
        return
    gs_model.load_ply(args.input_gs_ply)
    if args.input_gs_embed:
        gs_model.load_from_embedding(args.input_gs_embed)
    else:
        logger.warning("No embedding path specified, Gaussians might not be fully initialized.")


    if getattr(args, "cloth_fit_gs_scale_safety_enabled", False):
        gs_model.set_cloth_fit_gs_scale_safety_cfg(
            enabled=True,
            percentile=float(getattr(args, "cloth_fit_gs_scale_safety_percentile", 0.995)),
            hard_max_scale=float(getattr(args, "cloth_fit_gs_scale_safety_hard_max_scale", 0.0)),
            ratio_enabled=bool(getattr(args, "cloth_fit_gs_scale_safety_ratio_enabled", False)),
            ratio_percentile=float(getattr(args, "cloth_fit_gs_scale_safety_ratio_percentile", 0.995)),
            ratio_hard_max=float(getattr(args, "cloth_fit_gs_scale_safety_ratio_hard_max", 0.0)),
            ratio_symmetric=bool(getattr(args, "cloth_fit_gs_scale_safety_ratio_symmetric", False)),
            action=str(getattr(args, "cloth_fit_gs_scale_safety_action", "clamp")),
            eps_opacity=float(getattr(args, "cloth_fit_gs_scale_safety_eps_opacity", 1e-6)),
        )
        logger.info(
            "[cloth_fit_gs_scale_safety] enabled=true percentile={} hard_max_scale={} action={}",
            getattr(args, "cloth_fit_gs_scale_safety_percentile", 0.995),
            getattr(args, "cloth_fit_gs_scale_safety_hard_max_scale", 0.0),
            getattr(args, "cloth_fit_gs_scale_safety_action", "clamp"),
        )

    pipe = config.pipe
    if args.gui_ip != 'none':
        network_gui.init(args.gui_ip, args.gui_port)


    actual_target_frame_end = args.target_frame_end
    if actual_target_frame_end is None:
        actual_target_frame_end = len(smpl_data_tar['global_orient']) - 1

    all_target_frame_indices = list(range(args.target_frame_start, actual_target_frame_end, args.target_frame_step))
    if not all_target_frame_indices:
        logger.warning("No target frames selected based on start, end, and step parameters.")
        return

    out_dir_anim_meshes = output_dir_base / f"reshape_no_repose_{Path(args.nerf_mesh_path).stem}_meshes"
    out_dir_anim_meshes.mkdir(parents=True, exist_ok=True)

    logger.info(f"Processing {len(all_target_frame_indices)} target frames for reshape without reposing...")
    logger.info(f"Saving meshes to: {out_dir_anim_meshes}")

    smpl_faces_src = torch.from_numpy(smpl_model_src.faces.astype(np.int32)).to(device)

    for target_frame_idx in tqdm(all_target_frame_indices, desc="Processing target frames for reshape without reposing"):
        if not (0 <= target_frame_idx < len(smpl_data_tar['global_orient'])):
            logger.warning(f"Target frame index {target_frame_idx} is out of bounds. Skipping.")
            continue

        with torch.no_grad():

            smpl_params_src_shape_source_pose = {
                **smpl_params_src_pose,
                'betas': get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, 'source'),
            }
            smpl_params_src_shape_source_pose_filtered = {k: v for k, v in smpl_params_src_shape_source_pose.items() if v is not None}
            smpl_out_src_shape_source_pose = smpl_model_src.forward(**smpl_params_src_shape_source_pose_filtered)
            smpl_verts_src_shape_source_pose = smpl_out_src_shape_source_pose.vertices[0]


            smpl_params_tar_shape_source_pose = {
                **smpl_params_src_pose,
                'betas': get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, 'target'),
            }
            smpl_params_tar_shape_source_pose_filtered = {k: v for k, v in smpl_params_tar_shape_source_pose.items() if v is not None}
            smpl_out_tar_shape_source_pose = smpl_model_tar.forward(**smpl_params_tar_shape_source_pose_filtered)
            smpl_verts_tar_shape_source_pose = smpl_out_tar_shape_source_pose.vertices[0]


            nerf_verts_canonical = transform_to_canonical_space(nerf_verts_pt, src_global_orient, src_transl, src_scale, j0_src)


            if args.enable_cloth_fit_reshaping:
                logger.info("Applying cloth-fit reshaping...")


                from reposing.head_alignment import load_head_vertices_for_subject
                cache_key = '_head_vertex_indices_cache'
                if not hasattr(args, cache_key):
                    head_vertex_indices = load_head_vertices_for_subject(
                        args.dat_dir, nerf_verts_pt, nerf_faces_pt,
                        getattr(args, 'use_detailed_labels', True),
                        getattr(args, 'label_dir_name', 'labeled')
                    )
                    setattr(args, cache_key, head_vertex_indices)
                else:
                    head_vertex_indices = getattr(args, cache_key)

                reshaped_nerf_verts = apply_cloth_fit_reshaping(
                    args, device, nerf_verts_pt, nerf_faces_pt, head_vertex_indices
                )
                reshape_method = "cloth_fit"
            else:
                logger.info("Applying SMPL-based reshaping...")

                shape_diff_smpl = smpl_verts_tar_shape_source_pose - smpl_verts_src_shape_source_pose


                if args.body_reshape_smoothing_samples > 1 and args.body_reshape_mode == 'barycentric':
                    shape_diff_nerf, sampled_pts, surface_pts = calc_smoothed_nearest_neighbor(
                        query_pts=nerf_verts_canonical.unsqueeze(0),
                        ref_v=smpl_verts_src_shape_source_pose.unsqueeze(0),
                        ref_f=smpl_faces_src.unsqueeze(0),
                        weights=shape_diff_smpl.unsqueeze(0),
                        n_samples=args.body_reshape_smoothing_samples,
                        ret_surface_pts=args.visualize_reshaping,
                        method=args.body_reshape_mode,
                        nn_scale_factor=1000.0,
                        sample_std_scale=args.body_reshape_sample_std_scale,
                        use_distance_weighting=args.body_reshape_distance_weighting
                    )
                else:
                    shape_diff_nerf, surface_pts = calc_nearest_neighbor(
                        query_pts=nerf_verts_canonical.unsqueeze(0),
                        ref_v=smpl_verts_src_shape_source_pose.unsqueeze(0),
                        ref_f=smpl_faces_src.unsqueeze(0),
                        weights=shape_diff_smpl.unsqueeze(0),
                        ret_surface_pts=args.visualize_reshaping,
                        method=args.body_reshape_mode,
                        nn_scale_factor=1000.0,
                    )
                    sampled_pts = None

                if args.visualize_reshaping:
                    visualizer.vis(
                        smpl_verts_src_shape_source_pose.cpu().numpy(),
                        smpl_faces_src.cpu().numpy(),
                        nerf_verts_canonical.cpu().numpy(),
                        nerf_faces_pt.cpu().numpy(),
                        sampled_pts.squeeze(0).cpu().numpy() if sampled_pts is not None else None,
                        surface_pts.squeeze(0).cpu().numpy()
                    )

                shape_diff_nerf = shape_diff_nerf.squeeze(0)


                shape_diff_nerf = shape_diff_nerf * args.body_reshape_scale_factor


                reshaped_nerf_verts_canonical = nerf_verts_canonical + shape_diff_nerf
                reshape_method = "smpl"


                reshaped_nerf_verts = transform_to_world_space(reshaped_nerf_verts_canonical, src_global_orient, src_transl, src_scale, j0_src)


            base_name = f"reshaped_nerf_source_pose_target_shape_{target_frame_idx:04d}"
            if args.enable_cloth_fit_reshaping:
                mesh_filename = f"{base_name}_cloth_fit_reshaped.obj"
            else:
                mesh_filename = f"{base_name}.obj"
            reshaped_nerf_mesh_path = out_dir_anim_meshes / mesh_filename
            if args.force_update_all or args.update_meshes or not reshaped_nerf_mesh_path.exists():
                pytorch3d.io.save_obj(str(reshaped_nerf_mesh_path), reshaped_nerf_verts, nerf_faces_pt)


            if args.save_reshaping_meshes:
                src_smpl_debug_path = out_dir_anim_meshes / f"debug_src_smpl_source_pose_{target_frame_idx:04d}.obj"
                tar_smpl_debug_path = out_dir_anim_meshes / f"debug_tar_smpl_source_pose_{target_frame_idx:04d}.obj"
                pytorch3d.io.save_obj(str(src_smpl_debug_path), smpl_verts_src_shape_source_pose, smpl_faces_src)
                pytorch3d.io.save_obj(str(tar_smpl_debug_path), smpl_verts_tar_shape_source_pose, smpl_faces_src)


            reshaped_mesh_temp = pytorch3d.structures.Meshes(verts=[reshaped_nerf_verts], faces=[nerf_faces_pt])
            reshaped_nerf_normals = reshaped_mesh_temp.verts_normals_packed()

            mesh_info = {
                'mesh_verts': reshaped_nerf_verts,
                'mesh_norms': reshaped_nerf_normals,
                'mesh_faces': nerf_faces_pt,
            }
            gs_model.update_to_posed_mesh(mesh_info)


            if args.save_reposed_gs_ply:
                if not hasattr(args, '_gs_output_dir_created'):
                    gs_output_dir = output_dir_base / f"reshaped_gaussians_ply_{Path(args.nerf_mesh_path).stem}"
                    gs_output_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Saving reshaped Gaussian PLY files to: {gs_output_dir}")
                    args._gs_output_dir_created = True
                    args._gs_output_dir = gs_output_dir

                reshaped_gs_ply_filename = f"reshaped_gs_target_shape_{target_frame_idx:04d}.ply"
                if args.enable_cloth_fit_reshaping:
                    reshaped_gs_ply_filename = f"{reshaped_gs_ply_filename}_cloth_fit_reshaped{args.cloth_fit_restoration_suffix}.ply"
                reshaped_gs_ply_path = args._gs_output_dir / reshaped_gs_ply_filename
                if args.force_update_all or args.update_meshes or args.update_reposed_gs_ply or not reshaped_gs_ply_path.exists():
                    gs_model.save_ply(str(reshaped_gs_ply_path))
                    logger.info(f"Saved reshaped Gaussian PLY: {reshaped_gs_ply_path}")

    logger.info(f"Reshape without reposing completed! Reshaped meshes saved to: {out_dir_anim_meshes}")
