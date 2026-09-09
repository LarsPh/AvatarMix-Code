from __future__ import annotations

import os
import math
import json
import glob
import tempfile
from pathlib import Path

import numpy as np
import torch
import pytorch3d.io
import pytorch3d.structures
from tqdm import tqdm
from loguru import logger

from gaussian_renderer import network_gui
from scene.dataset_readers import make_scene_camera
from dataset.dataset_helper import make_frameset_data
from model import libcore
from model.splatting_avatar_model import SplattingAvatarModel
from model.smplx_utils import smplx_ani as smplx
from model.reshaping.body_reshape import (
    NeighborVisialzier,
    calc_nearest_neighbor,
    calc_smoothed_nearest_neighbor,
)

from reposing.head_alignment import process_head_alignment
from reposing.cloth_fit_utils.reposed_smpl_cleanup import clean_reposed_smpl_mesh_with_lbs_preservation
from reposing.cloth_fit_utils.skeleton_extraction import save_reposed_smpl_mesh_and_skeleton
from reposing.cloth_fit_reshaping import apply_cloth_fit_reshaping
from reposing.clothfit_a2_artifact import generate_a2_artifacts_for_frame
from reposing.utils.coordinate_transforms import transform_to_canonical_space, transform_to_world_space
from reposing.utils.smpl_utils import (
    get_betas_for_reposing,
    get_v_pose_for_reposing,
    get_v_shape_for_reposing,
)
from reposing.utils.camera_utils import load_camera_definitions_from_nerfstudio_path, create_scene_camera_from_definition
from reposing.utils.video_utils import make_video
from reposing.dataset import AnimateDataset

from reposing.utils.repose_common import (
    ACTORSHQ_SMPLX_CONFIG,
    AVATARREX_SMPLX_CONFIG,
    THUMAN2_SMPLX_CONFIG,
    _resolve_gender_for_dir,
)


def repose_gs_on_mesh(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")
    if args.visualize_reshaping:
        visualizer = NeighborVisialzier()

    subject_root = Path(args.target_pose_dir)
    smpl_model_path = args.smpl_model_path
    output_dir_base = Path(args.output_dir)
    output_dir_base.mkdir(parents=True, exist_ok=True)

    out_dir_reposed_gs_ply = None
    if args.save_reposed_gs_ply:
        out_dir_reposed_gs_ply = output_dir_base / "reposed_gaussians_ply"
        out_dir_reposed_gs_ply.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving reposed GS PLY files to: {out_dir_reposed_gs_ply}")


    logger.info(f"Loading canonical NeRF mesh from: {args.nerf_mesh_path}")
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

    logger.info(f"Loading LBS weights from: {args.lbs_weights_path}")
    try:
        verts_w = torch.from_numpy(np.load(args.lbs_weights_path).astype(np.float32)).to(device)
        if verts_w.shape[0] != nerf_verts_pt.shape[0]:
            raise ValueError(f"Vertex count mismatch: NeRF mesh has {nerf_verts_pt.shape[0]} verts, LBS weights have {verts_w.shape[0]} verts.")
        logger.info(f"Loaded LBS weights with shape: {verts_w.shape}")
    except Exception as e:
        logger.error(f"Error loading LBS weights: {e}")
        return


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
    logger.info(f"Loading SMPL parameters from: {subject_root / 'smpl_params.npz'}")
    try:
        smpl_data_npz_tar = np.load(str(subject_root / 'smpl_params.npz'), allow_pickle=True)
        smpl_data_tar = {k: torch.from_numpy(v.astype(np.float32)).to(device) for k,v in smpl_data_npz_tar.items()}
        smpl_data_npz_src = np.load(str(Path(args.dat_dir) / 'smpl_params.npz'), allow_pickle=True)
        smpl_data_src = {k: torch.from_numpy(v.astype(np.float32)).to(device) for k,v in smpl_data_npz_src.items()}
    except Exception as e:
        logger.error(f"Error loading SMPL parameters: {e}")
        return


    logger.info(f"Initializing SMPL-X model from: {smpl_model_path}")
    is_talkbody4d_tar = "talkbody4d" in str(args.target_pose_dir)
    is_talkbody4d_src = "talkbody4d" in str(args.dat_dir)
    gender_tar = _resolve_gender_for_dir(subject_root, args.smpl_gender_tar, force_neutral=is_talkbody4d_tar)
    gender_src = _resolve_gender_for_dir(Path(args.dat_dir), args.smpl_gender_src, force_neutral=is_talkbody4d_src)

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
        smpl_params_src = {


            'body_pose': smpl_data_src['body_pose'][src_frame_idx:src_frame_idx+1],
            'jaw_pose': smpl_data_src.get('jaw_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_jaw_pose_coeffs if hasattr(smpl_model_src, 'num_jaw_pose_coeffs') else 3, device=device))[src_frame_idx:src_frame_idx+1],
            'betas': get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, 'source'),
            'expression': smpl_data_src.get('expression', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_expression_coeffs if hasattr(smpl_model_src, 'num_expression_coeffs') else 10, device=device))[src_frame_idx:src_frame_idx+1],
            'left_hand_pose': smpl_data_src.get('left_hand_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_left_hand_pose_coeffs if hasattr(smpl_model_src, 'num_left_hand_pose_coeffs') else 15*3, device=device))[src_frame_idx:src_frame_idx+1],
            'right_hand_pose': smpl_data_src.get('right_hand_pose', torch.zeros(len(smpl_data_src['global_orient']), smpl_model_src.num_right_hand_pose_coeffs if hasattr(smpl_model_src, 'num_right_hand_pose_coeffs') else 15*3, device=device))[src_frame_idx:src_frame_idx+1],
            'v_shape': smpl_data_src.get('v_shape', None)[src_frame_idx:src_frame_idx+1] if smpl_data_src.get('v_shape', None) is not None else None,
            'v_pose': smpl_data_src.get('v_pose', None)[src_frame_idx:src_frame_idx+1] if smpl_data_src.get('v_pose', None) is not None else None,
        }
        smpl_params_src_filtered = {k: v for k, v in smpl_params_src.items() if v is not None}


        smpl_out_src = smpl_model_src.forward(**smpl_params_src_filtered)


        j0_src = smpl_out_src.joints[:, 0]
        src_scale = smpl_data_src.get('scale')
        src_scale = src_scale[src_frame_idx:src_frame_idx+1] if src_scale is not None else torch.ones(1, device=device)

        inv_src_pose_jnt_mats = torch.linalg.inv(smpl_out_src.A)

        src_smpl_mesh_path = output_dir_base / f"smpl_mesh_src_frame_{src_frame_idx:04d}.obj"
        if args.force_update_all or args.update_meshes or not src_smpl_mesh_path.exists():
            pytorch3d.io.save_obj(str(src_smpl_mesh_path),
                                 smpl_out_src.vertices[0],
                                 torch.tensor(smpl_model_src.faces.astype(np.int32), device=device))
            logger.info(f"Saved source SMPL mesh (frame {src_frame_idx}) to: {src_smpl_mesh_path}")
        else:
            logger.info(f"Source SMPL mesh {src_smpl_mesh_path} already exists. Skipping generation.")


        ref_nerf_verts_centered = nerf_verts_pt - src_transl

        ref_nerf_verts_scaled = ref_nerf_verts_centered / src_scale


        global_orient_matrix = pytorch3d.transforms.axis_angle_to_matrix(src_global_orient)
        global_orient_inverse = global_orient_matrix.transpose(1, 2)


        is_external_rigid_src = (
            ("mvhumannet" in str(args.dat_dir))
            or ("actorshq" in str(args.dat_dir))
            or ("talkbody4d" in str(args.dat_dir))
        )
        if is_external_rigid_src:
            ref_nerf_verts_normed = (global_orient_inverse @ ref_nerf_verts_scaled.unsqueeze(-1)).squeeze(-1)
        else:

            ref_nerf_verts_normed = (global_orient_inverse @ (ref_nerf_verts_scaled.unsqueeze(-1) - j0_src.unsqueeze(-1))).squeeze(-1)
            ref_nerf_verts_normed = ref_nerf_verts_normed + j0_src

        src_nerf_mesh_filename = f"normed_nerf_mesh_src_frame_{Path(args.nerf_mesh_path).stem}.obj"
        src_nerf_mesh_path = output_dir_base / src_nerf_mesh_filename
        if args.force_update_all or args.update_meshes or not src_nerf_mesh_path.exists():
            pytorch3d.io.save_obj(str(src_nerf_mesh_path), ref_nerf_verts_normed, nerf_faces_pt)
            logger.info(f"Saved source NeRF mesh (from {args.nerf_mesh_path}, in source pose) to: {src_nerf_mesh_path}")
        else:
            logger.info(f"Source NeRF mesh {src_nerf_mesh_path} already exists. Skipping generation.")


    gs_model = SplattingAvatarModel(config.model, verbose=True, gaussians_are_frozen=True)
    if not args.input_gs_ply:
        logger.error("Error: Path to input Gaussian Splatting PLY (--input_gs_ply) is required.")
        return
    gs_model.load_ply(args.input_gs_ply)
    if args.input_gs_embed:
        gs_model.load_from_embedding(args.input_gs_embed)
    else:
        logger.warning("Warning: No embedding path specified, Gaussians might not be fully initialized.")

    pipe = config.pipe
    if args.gui_ip != 'none':
        network_gui.init(args.gui_ip, args.gui_port)


    actual_target_frame_end = args.target_frame_end
    if actual_target_frame_end is None:
        if 'global_orient' not in smpl_data_tar or len(smpl_data_tar['global_orient']) == 0:
                logger.error("Error: SMPL data is empty or 'global_orient' is missing. Cannot determine target frame end.")
                return
        actual_target_frame_end = len(smpl_data_tar['global_orient']) - 1

    if args.target_frame_start > actual_target_frame_end:
        logger.error(f"Error: target_frame_start ({args.target_frame_start}) is greater than target_frame_end ({actual_target_frame_end}). No frames to render.")
        return

    all_target_frame_indices = list(range(args.target_frame_start, actual_target_frame_end, args.target_frame_step))
    if not all_target_frame_indices:
        logger.warning("Warning: No target frames selected based on start, end, and step parameters.")
        return


    if args.camera_path_json:
        raise NotImplementedError("This branch is not finished yet")


    else:


        render_cameras = []
        if args.num_render_views == 1:
            try:


                frameset_train_for_cam = make_frameset_data(config.dataset, split='train')
                if not frameset_train_for_cam or len(frameset_train_for_cam) == 0:
                    raise ValueError("Training frameset is empty or could not be loaded.")
                first_train_sample = frameset_train_for_cam[0]

                if 'scene_cameras' not in first_train_sample or not first_train_sample['scene_cameras']:
                     raise ValueError("First train sample does not contain 'scene_cameras'.")
                render_cameras.append(first_train_sample['scene_cameras'][0].to(device))
            except Exception as e:
                logger.warning(f"Warning: Could not get camera from dataset for single view rendering ({e}), creating a default one.")
                dummy_cam_obj = libcore.Camera()
                dummy_cam_obj.w, dummy_cam_obj.h = 512, 512; dummy_cam_obj.fx, dummy_cam_obj.fy = 500,500
                dummy_cam_obj.cx, dummy_cam_obj.cy = 256,256; dummy_cam_obj.R = np.eye(3); dummy_cam_obj.T = np.array([0,0,3.0])
                empty_img = np.zeros((dummy_cam_obj.h, dummy_cam_obj.w, 3), dtype=np.uint8)
                render_cameras.append(make_scene_camera(0, dummy_cam_obj, empty_img, config.dataset).to(device))
        else:
            logger.info(f"Setting up {args.num_render_views} render views from training dataset...")
            try:
                full_train_dataset = make_frameset_data(config.dataset, split='train')
                if len(full_train_dataset) == 0:
                    raise ValueError("Training dataset is empty, cannot select cameras.")

                all_train_cameras_collection = []
                camera_ids_added = set()
                num_samples_to_check = min(len(full_train_dataset), args.num_render_views * 10)

                for i in range(num_samples_to_check):
                    sample = full_train_dataset[i]
                    if 'scene_cameras' in sample:
                        for scene_cam in sample['scene_cameras']:
                            cam_uid_tuple = tuple(np.round(scene_cam.world_view_transform.cpu().numpy().flatten(), 4))
                            if cam_uid_tuple not in camera_ids_added:
                                all_train_cameras_collection.append(scene_cam.to(device))
                                camera_ids_added.add(cam_uid_tuple)
                                if len(all_train_cameras_collection) >= args.num_render_views:
                                    break
                    if len(all_train_cameras_collection) >= args.num_render_views:
                        break

                if len(all_train_cameras_collection) < args.num_render_views:
                    logger.warning(f"Warning: Found only {len(all_train_cameras_collection)} unique cameras, requested {args.num_render_views}. Using available ones.")
                if not all_train_cameras_collection:
                     raise ValueError("No cameras could be collected from training data for multi-view rendering.")
                render_cameras = all_train_cameras_collection[:args.num_render_views]

            except Exception as e:
                logger.error(f"Error setting up multi-view cameras from dataset: {e}. Falling back to default single view.")
                render_cameras = []
                dummy_cam_obj = libcore.Camera()
                dummy_cam_obj.w, dummy_cam_obj.h = 512, 512; dummy_cam_obj.fx, dummy_cam_obj.fy = 500,500
                dummy_cam_obj.cx, dummy_cam_obj.cy = 256,256; dummy_cam_obj.R = np.eye(3); dummy_cam_obj.T = np.array([0,0,3.0])
                empty_img = np.zeros((dummy_cam_obj.h, dummy_cam_obj.w, 3), dtype=np.uint8)
                render_cameras.append(make_scene_camera(0, dummy_cam_obj, empty_img, config.dataset).to(device))


        src2live_tar_jnt_mats_list = []
        posed_smpl_verts_target_shape_list = []
        processed_target_indices_original_mode = []
        out_dir_anim_meshes = output_dir_base / f"anim_{Path(args.nerf_mesh_path).stem}_to_targets_meshes"
        out_dir_anim_meshes.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving reposed SMPL and NeRF meshes (for multi-target mode) to: {out_dir_anim_meshes}")

        for frame_idx in all_target_frame_indices:
            if not (0 <= frame_idx < len(smpl_data_tar['global_orient'])):
                logger.warning(f"Warning: Target frame index {frame_idx} is out of bounds for SMPL data (max {len(smpl_data_tar['global_orient'])-1}). Skipping this frame.")
                continue
            with torch.no_grad():
                smpl_params_tar_no_transf = {


                    'body_pose': smpl_data_tar['body_pose'][frame_idx:frame_idx+1],
                    'jaw_pose': smpl_data_tar.get('jaw_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_jaw_pose_coeffs if hasattr(smpl_model_tar, 'num_jaw_pose_coeffs') else 3, device=device))[frame_idx:frame_idx+1],
                    'betas': get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, 'target'),
                    'expression': smpl_data_tar.get('expression', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_expression_coeffs if hasattr(smpl_model_tar, 'num_expression_coeffs') else 10, device=device))[frame_idx:frame_idx+1],
                    'left_hand_pose': smpl_data_tar.get('left_hand_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_left_hand_pose_coeffs if hasattr(smpl_model_tar, 'num_left_hand_pose_coeffs') else 15*3, device=device))[frame_idx:frame_idx+1],
                    'right_hand_pose': smpl_data_tar.get('right_hand_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_right_hand_pose_coeffs if hasattr(smpl_model_tar, 'num_right_hand_pose_coeffs') else 15*3, device=device))[frame_idx:frame_idx+1],

                    'v_pose': get_v_pose_for_reposing(smpl_data_tar, frame_idx),

                    'v_shape': get_v_shape_for_reposing(args, smpl_data_src, smpl_data_tar, 'target'),
                }


                smpl_params_tar_filtered = {k: v for k, v in smpl_params_tar_no_transf.items() if v is not None}
                smpl_out_tar = smpl_model_tar.forward(**smpl_params_tar_filtered)
                if args.enable_body_reshape:
                    posed_smpl_verts_target_shape_list.append(smpl_out_tar.vertices[0].detach())


                posed_smpl_direct_output_mesh_path = out_dir_anim_meshes / f"smpl_mesh_direct_target_frame_{frame_idx:04d}.obj"
                if (args.force_update_all or args.update_meshes or not posed_smpl_direct_output_mesh_path.exists()) and args.verbose_save_mesh:
                    pytorch3d.io.save_obj(str(posed_smpl_direct_output_mesh_path),
                                         smpl_out_tar.vertices[0],
                                         torch.tensor(smpl_model_tar.faces.astype(np.int32), device=device))


                if args.save_reposed_smpl_skeleton_for_repose:
                    output_dir = posed_smpl_direct_output_mesh_path.parent
                    mesh_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}.obj"
                    skeleton_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_skeleton.obj"
                    skin_mask_path = output_dir / f"smpl_reposed_frame_{frame_idx:04d}_cleaned_smplx_skin_indices.txt"


                    needs_update = (args.force_update_all or args.update_meshes or
                                  not mesh_path.exists() or not skeleton_path.exists() or not skin_mask_path.exists())

                    if needs_update:


                        tar_go = smpl_data_tar["global_orient"][frame_idx:frame_idx+1]
                        tar_tr = smpl_data_tar["transl"][frame_idx:frame_idx+1]
                        tar_post_rh = None
                        tar_post_th = None
                        is_talkbody4d_tar = "talkbody4d" in str(args.target_pose_dir)
                        is_external_rigid_tar = ("mvhumannet" in str(args.target_pose_dir)) or ("actorshq" in str(args.target_pose_dir))

                        if is_talkbody4d_tar and ("Rh" in smpl_data_tar and "Th" in smpl_data_tar):
                            tar_post_rh = smpl_data_tar["Rh"][frame_idx:frame_idx+1]
                            tar_post_th = smpl_data_tar["Th"][frame_idx:frame_idx+1]
                        elif is_external_rigid_tar:


                            tar_post_rh = tar_go
                            tar_post_th = tar_tr
                            tar_go = torch.zeros_like(tar_go)
                            tar_tr = torch.zeros_like(tar_tr)

                        saved_mesh_path, saved_skeleton_path = save_reposed_smpl_mesh_and_skeleton(

                            smpl_model_src, smpl_params_tar_no_transf, output_dir, frame_idx,
                            save_mesh=True, save_skeleton=True,
                            target_global_orient=tar_go,
                            target_transl=tar_tr,
                            target_scale=smpl_data_tar.get('scale'),
                            target_post_rh=tar_post_rh,
                            target_post_th=tar_post_th,
                            simplified=args.simplified_skeleton, device=device)
                        logger.info(f"Saved reposed SMPL mesh and skeleton with target global transformation for frame {frame_idx}")


                        if saved_mesh_path:
                            try:


                                try:
                                    generate_a2_artifacts_for_frame(
                                        args=args,
                                        reposed_mesh_path=Path(saved_mesh_path),
                                        output_dir=Path(output_dir),
                                        frame_idx=frame_idx,
                                    )
                                except Exception as e:
                                    logger.error(f"A2 artifact generation failed for frame {frame_idx}: {e}")
                                    raise

                                cleaned_mesh_path, cleaned_lbs_path, cleanup_stats = clean_reposed_smpl_mesh_with_lbs_preservation(
                                    reposed_mesh_path=saved_mesh_path,
                                    source_subject_id=args.src_subject_id,
                                    dataset_root=Path(args.dataset_root),
                                    frame_idx=frame_idx,
                                    output_dir=output_dir,
                                    smplx_segmentation_json_path=Path(args.smplx_segmentation_json_path) if args.smplx_segmentation_json_path else None,
                                    cloth_fit_remove_feet=args.cloth_fit_remove_feet,
                                    cloth_fit_keep_palm=bool(getattr(args, "cloth_fit_keep_palm", False)),
                                )
                                if cleaned_mesh_path:
                                    logger.info(f"Cleaned reposed SMPL mesh with preserved LBS weights: {cleaned_mesh_path}")
                                    if cleanup_stats:
                                        logger.info(f"  Cleanup: {cleanup_stats['initial_vertices']}→{cleanup_stats['final_vertices']} vertices ({cleanup_stats['final_vertex_ratio']:.1%})")
                                        logger.info(f"  LBS validation: sums_valid={cleanup_stats.get('vertex_sums_valid', 'unknown')}, one_hot_valid={cleanup_stats.get('one_hot_valid', 'unknown')}")
                                else:
                                    logger.warning(f"SMPL mesh cleanup failed for frame {frame_idx}, using original mesh")
                            except Exception as e:
                                logger.warning(f"SMPL mesh cleanup failed for frame {frame_idx}: {e}")
                                logger.warning(f"SMPL mesh cleanup failed, using original mesh: {e}")
                        else:
                            logger.debug("SMPL mesh cleanup skipped - no saved mesh path")

                src2live_tar_jnt_mats_list.append(torch.matmul(smpl_out_tar.A, inv_src_pose_jnt_mats))
                processed_target_indices_original_mode.append(frame_idx)

        if not src2live_tar_jnt_mats_list:
            logger.error("No valid target frames processed to create transformations. Exiting original mode.")
            return
        src2live_tar_jnt_mats = torch.cat(src2live_tar_jnt_mats_list, dim=0)

        base_anim_name = f"anim_{Path(args.nerf_mesh_path).stem}_to_targets"
        smplx_lbs_weights = smpl_model_tar.lbs_weights.to(device)
        if args.enable_body_reshape:
            smpl_faces_src = torch.from_numpy(smpl_model_src.faces.astype(np.int32)).to(device)
        smpl_verts_src_pose = smpl_out_src.vertices[0]
        smpl_verts_src_pose_homo = torch.cat([smpl_verts_src_pose, torch.ones_like(smpl_verts_src_pose[:, :1])], dim=-1).unsqueeze(-1)

        for cam_idx, cam_for_render in enumerate(render_cameras):
            out_dir_cam_anim = output_dir_base / f"{base_anim_name}_view{cam_idx:02d}_renders"
            if not args.disable_renders:
                out_dir_cam_anim.mkdir(parents=True, exist_ok=True)
                logger.info(f"Rendering for View {cam_idx}, saving to: {out_dir_cam_anim}")
            else:


                if out_dir_cam_anim.exists() and len(list(out_dir_cam_anim.iterdir())) == 0:
                    out_dir_cam_anim.rmdir()

            num_valid_frames_rendered_for_view = 0
            for i, current_target_frame_idx in enumerate(tqdm(processed_target_indices_original_mode, desc=f'Reposing & Rendering View {cam_idx}', leave=False)):


                with torch.no_grad():
                    fwd_skinning_mats_nerf = torch.einsum('nj,jxy->nxy', verts_w, src2live_tar_jnt_mats[i])


                    if args.enable_rigid_head_reposing:
                        from reposing.utils.rigid_head_reposing import apply_rigid_head_transformation
                        from reposing.head_alignment import load_head_vertices_for_subject


                        cache_key = '_head_vertex_indices_cache'
                        if not hasattr(args, cache_key):
                            head_vertex_indices = load_head_vertices_for_subject(
                                args.dat_dir, ref_nerf_verts_normed, nerf_faces_pt,
                                getattr(args, 'use_detailed_labels', True),
                                getattr(args, 'label_dir_name', 'labeled')
                            )
                            setattr(args, cache_key, head_vertex_indices)
                        else:
                            head_vertex_indices = getattr(args, cache_key)


                        fwd_skinning_mats_nerf = apply_rigid_head_transformation(
                            fwd_skinning_mats_nerf,
                            head_vertex_indices,
                            enable_filtering=args.enable_head_alignment_filtering,
                            outlier_threshold=args.rigid_head_outlier_threshold
                        )

                    ref_nerf_verts_homo = torch.cat([ref_nerf_verts_normed, torch.ones_like(ref_nerf_verts_normed[:, :1])], dim=-1).unsqueeze(-1)
                    posed_nerf_verts_homo_tar = torch.matmul(fwd_skinning_mats_nerf, ref_nerf_verts_homo)
                    normed_posed_nerf_verts_tar = posed_nerf_verts_homo_tar[:, :3, 0]

                    if args.enable_body_reshape:

                        with torch.no_grad():
                            frame_idx = current_target_frame_idx

                            posed_smpl_verts_target_shape = posed_smpl_verts_target_shape_list[i]


                            smpl_params_for_source_shape = {
                                'body_pose': smpl_data_tar['body_pose'][frame_idx:frame_idx+1],
                                'jaw_pose': smpl_data_tar.get('jaw_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_jaw_pose_coeffs if hasattr(smpl_model_tar, 'num_jaw_pose_coeffs') else 3, device=device))[frame_idx:frame_idx+1],
                                'betas': get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, 'source'),
                                'expression': smpl_data_tar.get('expression', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_expression_coeffs if hasattr(smpl_model_tar, 'num_expression_coeffs') else 10, device=device))[frame_idx:frame_idx+1],
                                'left_hand_pose': smpl_data_tar.get('left_hand_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_left_hand_pose_coeffs if hasattr(smpl_model_tar, 'num_left_hand_pose_coeffs') else 15*3, device=device))[frame_idx:frame_idx+1],
                                'right_hand_pose': smpl_data_tar.get('right_hand_pose', torch.zeros(len(smpl_data_tar['global_orient']), smpl_model_tar.num_right_hand_pose_coeffs if hasattr(smpl_model_tar, 'num_right_hand_pose_coeffs') else 15*3, device=device))[frame_idx:frame_idx+1],
                            }
                            smpl_params_for_source_shape_filtered = {k: v for k, v in smpl_params_for_source_shape.items() if v is not None}
                            smpl_out_src_shape_in_tar_pose = smpl_model_src.forward(**smpl_params_for_source_shape_filtered)
                            posed_smpl_verts_source_shape = smpl_out_src_shape_in_tar_pose.vertices[0]


                            shape_diff_smpl = posed_smpl_verts_target_shape - posed_smpl_verts_source_shape


                            if args.body_reshape_smoothing_samples > 1 and args.body_reshape_mode == 'barycentric':
                                shape_diff_nerf, sampled_pts, surface_pts = calc_smoothed_nearest_neighbor(
                                    query_pts=normed_posed_nerf_verts_tar.unsqueeze(0),
                                    ref_v=posed_smpl_verts_source_shape.unsqueeze(0),
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
                                    query_pts=normed_posed_nerf_verts_tar.unsqueeze(0),
                                    ref_v=posed_smpl_verts_source_shape.unsqueeze(0),
                                    ref_f=smpl_faces_src.unsqueeze(0),
                                    weights=shape_diff_smpl.unsqueeze(0),
                                    ret_surface_pts=args.visualize_reshaping,
                                    method=args.body_reshape_mode,
                                    nn_scale_factor=1000.0,
                                )
                                sampled_pts = None
                            if args.visualize_reshaping:
                                visualizer.vis(
                                    posed_smpl_verts_source_shape.cpu().numpy(),
                                    smpl_faces_src.cpu().numpy(),
                                    normed_posed_nerf_verts_tar.cpu().numpy(),
                                    nerf_faces_pt.cpu().numpy(),
                                    sampled_pts.squeeze(0).cpu().numpy() if sampled_pts is not None else None,
                                    surface_pts.squeeze(0).cpu().numpy()
                                )
                            shape_diff_nerf = shape_diff_nerf.squeeze(0)


                            if cam_idx == 0 and args.save_reshaping_meshes:

                                out_dir_reshape_debug = out_dir_anim_meshes / "shape_reshape_debug"
                                out_dir_reshape_debug.mkdir(parents=True, exist_ok=True)


                                smpl_faces_tar = torch.from_numpy(smpl_model_tar.faces.astype(np.int32)).to(device)
                                posed_target_smplx_path = out_dir_reshape_debug / f"posed_target_smplx_{current_target_frame_idx:04d}.obj"
                                pytorch3d.io.save_obj(str(posed_target_smplx_path), posed_smpl_verts_target_shape, smpl_faces_tar)


                                posed_source_smplx_path = out_dir_reshape_debug / f"posed_source_smplx_{current_target_frame_idx:04d}.obj"
                                pytorch3d.io.save_obj(str(posed_source_smplx_path), posed_smpl_verts_source_shape, smpl_faces_src)


                                if is_external_rigid_src:
                                    posed_nerf_verts_tar_before_reshape = (global_orient_matrix @ normed_posed_nerf_verts_tar.unsqueeze(-1)).squeeze(-1)
                                    posed_nerf_verts_tar_before_reshape = posed_nerf_verts_tar_before_reshape * src_scale
                                    posed_nerf_verts_tar_before_reshape = posed_nerf_verts_tar_before_reshape + src_transl
                                else:
                                    posed_nerf_verts_tar_before_reshape = (global_orient_matrix @ (normed_posed_nerf_verts_tar.unsqueeze(-1) - j0_src.unsqueeze(-1))).squeeze(-1)
                                    posed_nerf_verts_tar_before_reshape = posed_nerf_verts_tar_before_reshape + j0_src
                                    posed_nerf_verts_tar_before_reshape = posed_nerf_verts_tar_before_reshape * src_scale
                                    posed_nerf_verts_tar_before_reshape = posed_nerf_verts_tar_before_reshape + src_transl
                                    reposed_nerf_before_shape_change_path = out_dir_reshape_debug / f"reposed_nerf_before_shape_change_{current_target_frame_idx:04d}.obj"
                                    pytorch3d.io.save_obj(str(reposed_nerf_before_shape_change_path), posed_nerf_verts_tar_before_reshape, nerf_faces_pt)


                            shape_diff_nerf[0] = shape_diff_nerf[0] * args.body_reshape_scale_factor

                            normed_posed_nerf_verts_tar = normed_posed_nerf_verts_tar + shape_diff_nerf


                    if is_external_rigid_src:
                        posed_nerf_verts_tar = (global_orient_matrix @ normed_posed_nerf_verts_tar.unsqueeze(-1)).squeeze(-1)
                        posed_nerf_verts_tar = posed_nerf_verts_tar * src_scale
                        posed_nerf_verts_tar = posed_nerf_verts_tar + src_transl
                    else:

                        posed_nerf_verts_tar = (global_orient_matrix @ (normed_posed_nerf_verts_tar.unsqueeze(-1) - j0_src.unsqueeze(-1))).squeeze(-1)
                        posed_nerf_verts_tar = posed_nerf_verts_tar + j0_src
                        posed_nerf_verts_tar = posed_nerf_verts_tar * src_scale
                        posed_nerf_verts_tar = posed_nerf_verts_tar + src_transl

                    if cam_idx == 0:
                        from reposing.utils.rigid_head_reposing import get_rigid_head_suffix
                        rigid_suffix = get_rigid_head_suffix(args)
                        posed_nerf_mesh_path = out_dir_anim_meshes / f"nerf_mesh_target_frame_{current_target_frame_idx:04d}{rigid_suffix}.obj"
                        if args.force_update_all or args.update_meshes or not posed_nerf_mesh_path.exists():
                            pytorch3d.io.save_obj(str(posed_nerf_mesh_path), posed_nerf_verts_tar, nerf_faces_pt)


                    fwd_skinning_mats_smplx = torch.einsum('vj,jxy->vxy', smplx_lbs_weights, src2live_tar_jnt_mats[i])
                    posed_smplx_verts_homo_manual = torch.matmul(fwd_skinning_mats_smplx, smpl_verts_src_pose_homo)
                    posed_smplx_verts_manual = posed_smplx_verts_homo_manual[:, :3, 0]
                    if cam_idx == 0:
                        manual_posed_smplx_mesh_path = out_dir_anim_meshes / f"smplx_mesh_manual_skin_target_frame_{current_target_frame_idx:04d}.obj"
                        if (args.force_update_all or args.update_meshes or not manual_posed_smplx_mesh_path.exists()) and args.verbose_save_mesh:
                            pytorch3d.io.save_obj(str(manual_posed_smplx_mesh_path),
                                                posed_smplx_verts_manual,
                                                torch.tensor(smpl_model_tar.faces.astype(np.int32), device=device))

                    posed_mesh_temp = pytorch3d.structures.Meshes(verts=[posed_nerf_verts_tar], faces=[nerf_faces_pt])
                    posed_nerf_normals_tar = posed_mesh_temp.verts_normals_packed()

                    mesh_info = {
                        'mesh_verts': posed_nerf_verts_tar,
                        'mesh_norms': posed_nerf_normals_tar,
                        'mesh_faces': nerf_faces_pt,
                    }
                    gs_model.update_to_posed_mesh(mesh_info)

                    if args.save_reposed_gs_ply and out_dir_reposed_gs_ply is not None and cam_idx == 0:
                        from reposing.utils.rigid_head_reposing import get_rigid_head_suffix
                        rigid_suffix = get_rigid_head_suffix(args)
                        reposed_gs_ply_filename = f"reposed_gs_targetframe{current_target_frame_idx:04d}{rigid_suffix}.ply"
                        reposed_gs_ply_path = out_dir_reposed_gs_ply / reposed_gs_ply_filename
                        if args.force_update_all or args.update_reposed_gs_ply or not reposed_gs_ply_path.exists():
                            gs_model.save_ply(str(reposed_gs_ply_path))
                            if args.verbose_save_mesh:
                                logger.info(f"Saved reposed Gaussian Splatting PLY to: {reposed_gs_ply_path}")


                        reposed_mesh_filename = f"reposed_mesh_targetframe{current_target_frame_idx:04d}{rigid_suffix}.obj"
                        reposed_mesh_path = out_dir_reposed_gs_ply / reposed_mesh_filename
                        if args.force_update_all or args.update_meshes or args.update_reposed_gs_ply or not reposed_mesh_path.exists():
                            try:
                                pytorch3d.io.save_obj(str(reposed_mesh_path), posed_nerf_verts_tar, nerf_faces_pt)
                                if args.verbose_save_mesh:
                                    logger.info(f"Saved reposed mesh OBJ to: {reposed_mesh_path}")
                            except Exception as e_save:
                                logger.warning(f"Failed to save reposed mesh OBJ {reposed_mesh_path}: {e_save}")
                    if not args.disable_renders:
                        render_pkg = gs_model.render_to_camera(cam_for_render, pipe, background=torch.tensor(args.render_bg_color, dtype=torch.float32, device=device))
                        image = render_pkg['render']
                    else:
                        image = None

                if args.gui_ip != 'none' and not args.disable_renders:
                    network_gui.send_image_to_network(image, Path(args.dat_dir).stem)

                rendered_image_path = out_dir_cam_anim / f'frame_{current_target_frame_idx:04d}_view_{cam_idx:02d}.jpg'
                if args.force_update_all or args.update_renders or not rendered_image_path.exists() and not args.disable_renders:
                    libcore.write_tensor_image(str(rendered_image_path), image, rgb2bgr=True)
                num_valid_frames_rendered_for_view +=1


                if args.save_gt_aligned_head and cam_idx == 0:

                    label_dir_name = getattr(args, 'label_dir_name', 'labeled')
                    npz_suffix = ''
                    if label_dir_name != 'labeled':

                        npz_suffix = label_dir_name.replace('labeled', '')

                    process_head_alignment(
                        args=args,
                        dat_dir=args.dat_dir,
                        ref_nerf_verts_normed=ref_nerf_verts_normed,
                        nerf_faces_pt=nerf_faces_pt,
                        fwd_skinning_mats_nerf=fwd_skinning_mats_nerf,
                        gs_model=gs_model,
                        global_orient_matrix=global_orient_matrix,
                        j0_src=j0_src,
                        src_scale=src_scale,
                        src_transl=src_transl,
                        output_dir_base=output_dir_base,
                        frame_idx=current_target_frame_idx,
                        posed_nerf_verts_tar=posed_nerf_verts_tar,
                        verbose=args.verbose_save_mesh,
                        label_dir_name=label_dir_name,
                        npz_suffix=npz_suffix
                    )

            if num_valid_frames_rendered_for_view > 0:
                video_path_view = Path(out_dir_cam_anim) / f"{base_anim_name}_view{cam_idx:02d}.mp4"
                video_fps_default = 12
                if args.force_update_all or args.update_video or not video_path_view.exists():
                    make_video(str(out_dir_cam_anim), f"{base_anim_name}_view{cam_idx:02d}", frame_pattern=f"frame_*_view_{cam_idx:02d}.jpg", frame_rate=video_fps_default)
                else:
                    logger.info(f"Video {video_path_view} already exists. Skipping generation.")
            else:
                logger.warning(f"No frames rendered for view {cam_idx}, skipping video generation for this view.")
