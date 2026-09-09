import os
from pathlib import Path

from loguru import logger

from model import libcore
from reposing.config import create_argument_parser
from reposing.cloth_fit_reshaping.config import validate_cloth_fit_arguments
from reposing.cloth_fit_utils.reposed_smpl_cleanup import clean_smpl_body_mesh_for_volume

from reposing.repose_nerf_mesh import repose_gs_on_mesh
from reposing.repose_smpl_sequence import repose_gs_on_smpl_mesh
from reposing.reshape_only import reshape_nerf_in_source_pose
from reposing.repose_deformed import repose_gs_on_mesh_deformed


if __name__ == '__main__':
    parser = create_argument_parser()

    args, extras = parser.parse_known_args()


    if args.save_reposed_smpl_skeleton_for_repose:
        args.reposing_without_beta_change = True
        logger.info("Forced --reposing_without_beta_change due to --save_reposed_smpl_skeleton_for_repose")


        if not args.dataset_root:

            dat_path = Path(args.dat_dir)
            args.dataset_root = str(dat_path.parent)
            logger.info(f"Auto-inferred dataset_root: {args.dataset_root}")

        if not args.src_subject_id:

            args.src_subject_id = Path(args.dat_dir).stem
            logger.info(f"Auto-inferred src_subject_id: {args.src_subject_id}")


    if getattr(args, "generate_smpl_body_cleaned", False):
        roots = {Path(args.dat_dir), Path(args.target_pose_dir)}
        for root in sorted(roots, key=lambda p: str(p)):
            clean_smpl_body_mesh_for_volume(
                subject_root=root,
                targetperc=float(getattr(args, "smpl_body_cleaned_targetperc", 0.3)),
                maxholesize=int(getattr(args, "smpl_body_cleaned_maxholesize", 50)),
                overwrite=bool(getattr(args, "overwrite_smpl_body_cleaned", True)),
            )


    config = libcore.load_from_config(args.configs, cli_args=extras)
    config.dataset.dat_dir = args.dat_dir

    if args.enable_body_reshape:
        args.output_dir = args.output_dir + '_reshaped' + f'_{args.body_reshape_smoothing_samples}samples' + f'_std{args.body_reshape_sample_std_scale}'
        if args.body_reshape_scale_factor != 1.0:
            args.output_dir = args.output_dir + f'_scale{args.body_reshape_scale_factor}'
        if args.body_reshape_distance_weighting:
            args.output_dir = args.output_dir + '_distweight'


    if args.animation_mode == 'smpl_sequence':
        if not args.anim_fn:
            parser.error("--anim_fn is required for animation_mode 'smpl_sequence'.")
        if not args.pc_dir and not args.input_gs_ply:
             parser.error("Either --pc_dir (for point_cloud.ply) or --input_gs_ply must be specified for animation_mode 'smpl_sequence'.")
        if args.pc_dir and not args.input_gs_ply:
            args.input_gs_ply = os.path.join(args.pc_dir, 'point_cloud.ply')
        if args.pc_dir and not args.input_gs_embed:
             args.input_gs_embed = os.path.join(args.pc_dir, 'embedding.json')


        repose_gs_on_smpl_mesh(args, config)
    elif args.animation_mode == 'nerf_mesh_repose':
        if not all([args.smpl_model_path, args.nerf_mesh_path, args.lbs_weights_path]):
            parser.error("--smpl_model_path, --nerf_mesh_path, and --lbs_weights_path are required for 'nerf_mesh_repose' mode.")
        if args.camera_path_json and args.num_render_views > 1 and args.num_render_views != parser.get_default('num_render_views'):
             logger.warning("Warning: --camera_path_json is specified, so --num_render_views will be ignored. Camera path defines the views.")
        repose_gs_on_mesh(args, config)
    elif args.animation_mode == 'deform_ckpt_test':


        if not args.deform_ckpt:
            parser.error("--deform_ckpt is required for animation_mode 'deform_ckpt_test'.")
        repose_gs_on_mesh_deformed(args, config)
    elif args.animation_mode == 'nerf_reshape_no_repose':
        if not all([args.smpl_model_path, args.nerf_mesh_path]):
            parser.error("--smpl_model_path and --nerf_mesh_path are required for 'nerf_reshape_no_repose' mode.")
        if not (args.enable_body_reshape or args.enable_cloth_fit_reshaping):
            parser.error("Either --enable_body_reshape or --enable_cloth_fit_reshaping must be enabled for 'nerf_reshape_no_repose' mode.")
        if args.enable_cloth_fit_reshaping:
            validate_cloth_fit_arguments(args)
        reshape_nerf_in_source_pose(args, config)
    else:
        parser.error(f"Unknown animation_mode: {args.animation_mode}")

        args.output_dir = args.output_dir + '_reshaped'

    logger.info('[done]')
