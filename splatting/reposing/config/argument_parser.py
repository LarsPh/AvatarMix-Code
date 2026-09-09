from argparse import ArgumentParser
from ..cloth_fit_reshaping.config import add_cloth_fit_arguments


def create_argument_parser():

    parser = ArgumentParser(description='SplattingAvatar Animation Evaluation')


    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], required=True, help='Path to config file(s).')
    parser.add_argument('--dat_dir', type=str, required=True, help="Subject data directory containing SMPL parameters.")
    parser.add_argument('--target_pose_dir', type=str, required=True, help="Path to the target pose file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save rendered animations.")
    parser.add_argument('--cpu', action='store_true', help="Force use CPU.")


    parser.add_argument('--gui_ip', type=str, default='none', help="IP for GUI network streaming. 'none' to disable.")
    parser.add_argument('--gui_port', type=int, default=6009, help="Port for GUI network streaming.")


    parser.add_argument('--input_gs_ply', type=str, required=False, default=None, help="Path to the input Gaussian Splatting PLY file.")
    parser.add_argument('--input_gs_embed', type=str, default=None, help="Optional path to the embedding JSON for the GS model.")
    parser.add_argument('--render_bg_color', nargs=3, type=float, default=[1.0, 1.0, 1.0], help='Background color for rendering (R G B values 0-1).')


    parser.add_argument(
        '--animation_mode',
        type=str,
        default='nerf_mesh_repose',
        choices=['smpl_sequence', 'nerf_mesh_repose', 'nerf_reshape_no_repose', 'deform_ckpt_test'],
        help=(
            "'smpl_sequence' to use anim_fn for SMPL, 'nerf_mesh_repose' to repose NeRF mesh with AvatarREX SMPL, "
            "'nerf_reshape_no_repose' to reshape NeRF mesh without reposing, "
            "'deform_ckpt_test' to repose using a trained deformation checkpoint on an AvatarRex motion sequence."
        ),
    )


    parser.add_argument('--anim_fn', type=str, help="Path to animation sequence (.npz file with 'poses' and 'trans'). Required if animation_mode is 'smpl_sequence'.")
    parser.add_argument('--pc_dir', type=str, help="Path to the point cloud directory (used by original InstantAvatar to load GS). Deprecated by --input_gs_ply but kept for compatibility if needed.")


    parser.add_argument('--smpl_model_path', type=str, help="Path to SMPL/SMPL-X model directory. Required for 'nerf_mesh_repose'.")
    parser.add_argument('--smpl_gender_src', type=str, default='neutral', choices=['male', 'female', 'neutral'], help="Gender for SMPL model.")
    parser.add_argument('--smpl_gender_tar', type=str, default='neutral', choices=['male', 'female', 'neutral'], help="Gender for SMPL model.")
    parser.add_argument('--nerf_mesh_path', type=str, help="Path to the canonical NeRF mesh (OBJ/PLY). Required for 'nerf_mesh_repose'.")
    parser.add_argument('--nerf_mesh_source_frame_idx', type=int, default=0, help="Frame index of the source pose for the canonical NeRF mesh.")
    parser.add_argument('--lbs_weights_path', type=str, help="Path to the transferred LBS weights for the NeRF mesh (.npy). Required for 'nerf_mesh_repose'.")
    parser.add_argument('--target_frame_start', type=int, default=0, help="Start frame index of the target frames to repose to.")
    parser.add_argument('--target_frame_end', type=int, default=None, help="End frame index of the target frames to repose to. If not provided, will use all frames from target_frame_start to the end of the animation.")
    parser.add_argument('--target_frame_step', type=int, default=1, help="Step size for the target frames to repose to. Default is 1.")

    parser.add_argument('--num_render_views', type=int, default=1, help="Number of viewpoints to render for 'nerf_mesh_repose' mode (when not using --camera_path_json).")


    parser.add_argument('--deform_ckpt', type=str, default=None, help="Path to deformation training checkpoint (chkpnt_iter_*.pth). Required for animation_mode='deform_ckpt_test'.")
    parser.add_argument('--pretrained_gs_from', type=str, default=None, help="Optional override for config.model.load_pretrained_gs_from (directory containing point_cloud.ply + embedding.json).")
    parser.add_argument('--latent_frame_idx', type=int, default=None, help="Optional fixed latent frame index override (default inferred from pretrained_gs_from name).")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], help="Dataset split to run for deform_ckpt_test.")
    parser.add_argument('--max_frames', type=int, default=None, help="Optional max number of unique frames to process for deform_ckpt_test.")
    parser.add_argument('--frame_stride', type=int, default=1, help="Process every Nth unique frame for deform_ckpt_test.")


    parser.add_argument('--camera_path_json', type=str, default=None, help="Path to nerfstudio camera path JSON for rendering. Overrides --num_render_views. Uses target_frame_start/end/step for SMPL poses.")


    parser.add_argument('--enable_body_reshape', action='store_true', help="Enable reshaping of the NeRF mesh to match the target's body shape.")
    parser.add_argument('--body_reshape_mode', type=str, default='barycentric', choices=['vertex', 'barycentric'], help="Method for transferring shape difference from SMPL to NeRF mesh.")
    parser.add_argument('--body_reshape_near_threshold', type=float, default=0.08, help="Distance threshold for body reshaping nearest neighbor search.")
    parser.add_argument('--body_reshape_smoothing_samples', type=int, default=1, help="Number of samples for smoothed body reshaping. Set to 1 to disable smoothing.")
    parser.add_argument('--body_reshape_sample_std_scale', type=float, default=1.0, help="Scale factor for the standard deviation of the sampling distribution for smoothed body reshaping.")
    parser.add_argument('--body_reshape_scale_factor', type=float, default=1.0, help="Global scaling factor applied to shape differences to amplify reshaping effects.")
    parser.add_argument('--body_reshape_distance_weighting', action='store_true', help="Use distance-weighted averaging in smoothed reshaping instead of simple mean.")
    parser.add_argument('--save_reshaping_meshes', action='store_true', help="Save the reshaping meshes for debugging.")


    parser.add_argument('--cloth_fit_restoration_suffix', type=str, default='', help="Suffix for cloth-fit restoration.")
    parser.add_argument('--cloth_fit_remove_feet', action='store_true', help="Remove feet for cloth-fit restoration.")

    parser.add_argument('--cloth_fit_gs_scale_safety_enabled', action='store_true',
                       help="Enable GS scale safety handling during cloth-fit restoration (clamp or zero opacity for outliers).")
    parser.add_argument('--cloth_fit_gs_scale_safety_percentile', type=float, default=0.995,
                       help="Percentile (0-1) used to detect scale outliers (default: 0.995). Ignored if hard_max_scale is set.")
    parser.add_argument('--cloth_fit_gs_scale_safety_hard_max_scale', type=float, default=0.0,
                       help="If >0, use this as the hard maximum allowed Gaussian scale (in activated/linear scale units). Overrides percentile.")

    parser.add_argument('--cloth_fit_gs_scale_safety_ratio_enabled', action='store_true',
                       help="Also detect outliers by scale change ratio (current / previous). Useful to catch sudden scale blow-ups even when absolute scale is small.")
    parser.add_argument('--cloth_fit_gs_scale_safety_ratio_percentile', type=float, default=0.995,
                       help="Percentile (0-1) used to detect ratio outliers (default: 0.995). Ignored if ratio_hard_max is set.")
    parser.add_argument('--cloth_fit_gs_scale_safety_ratio_hard_max', type=float, default=0.0,
                       help="If >0, use this as the hard maximum allowed ratio (current/previous). Overrides ratio_percentile.")
    parser.add_argument('--cloth_fit_gs_scale_safety_ratio_symmetric', action='store_true',
                       help="Use symmetric ratio: max(r, 1/r) so both blow-ups and collapses can be flagged.")
    parser.add_argument('--cloth_fit_gs_scale_safety_action', type=str, default='clamp',
                       choices=['clamp', 'zero_opacity'],
                       help="Outlier handling: 'clamp' scales down outlier Gaussians; 'zero_opacity' hides them (opacity->0).")
    parser.add_argument('--cloth_fit_gs_scale_safety_eps_opacity', type=float, default=1e-6,
                       help="Opacity value to use for hidden Gaussians when action=zero_opacity (default: 1e-6).")
    parser.add_argument(
        '--cloth_fit_keep_palm',
        action='store_true',
        help="Keep palm (leftHand/rightHand) in SMPL-X cloth-fit removal mask; only remove HandIndex1 regions.",
    )
    add_cloth_fit_arguments(parser)


    parser.add_argument('--update_meshes', action='store_true', help="Force update/regeneration of mesh files even if they exist.")
    parser.add_argument('--update_renders', action='store_true', help="Force update/regeneration of rendered images even if they exist.")
    parser.add_argument('--update_video', action='store_true', help="Force update/regeneration of video files even if they exist.")
    parser.add_argument('--disable_renders', action='store_true', help="Disable rendering.")
    parser.add_argument('--force_update_all', action='store_true', help="Force update/regeneration of all outputs (meshes, renders, video), overriding individual update flags.")
    parser.add_argument('--verbose_save_mesh', action='store_true', help="Verbose save mesh.")
    parser.add_argument('--save_reposed_gs_ply', action='store_true', help="Save the reposed Gaussian Splatting model as a .ply file for each target frame.")
    parser.add_argument('--update_reposed_gs_ply', action='store_true', help="Force update/regeneration of reposed GS PLY files even if they exist.")
    parser.add_argument('--save_variant_plys', action='store_true', help="Save raw_lbs/offset/full mesh+GS PLY variants for deform_ckpt_test (like training visualizations).")
    parser.add_argument('--variant_ply_first_k', type=int, default=0, help="If >0, only dump variant PLYs for the first K unique frames (per run) to save space.")
    parser.add_argument(
        '--disable_finger_movement',
        action='store_true',
        help="Disable finger motion for demo: zero SMPL-X left_hand_pose/right_hand_pose per frame (wrists/body still move).",
    )
    parser.add_argument('--visualize_reshaping', action='store_true', help="Visualize the reshaping process.")


    parser.add_argument('--save_gt_aligned_head', action='store_true',
                       help="Save GT-aligned head mesh and Gaussians by averaging head transformations.")
    parser.add_argument('--head_alignment_outlier_threshold', type=float, default=2.0,
                       help="Standard deviation threshold for filtering outlier transformations.")
    parser.add_argument('--enable_head_alignment_filtering', action='store_true',
                       help="Enable robust outlier filtering for head transformations.")
    parser.add_argument('--use_detailed_labels', action='store_true', default=True,
                       help="Use detailed surface labels (same as head_swap.py). Default: True.")
    parser.add_argument('--label_dir_name', type=str, default='labeled',
                       help="Label directory name (e.g., 'labeled' or 'labeled_no_sam'). Default: 'labeled'.")


    parser.add_argument('--enable_rigid_head_reposing', action='store_true',
                       help="Apply uniform averaged transformation to all head vertices for perfect GT alignment.")
    parser.add_argument('--rigid_head_outlier_threshold', type=float, default=2.0,
                       help="Outlier threshold for robust head transformation averaging (only used if --enable_head_alignment_filtering is also enabled).")


    parser.add_argument('--reposing_without_beta_change', action='store_true',
                       help="Use source betas for all target poses (repose-only mode, no body shape change). Default: False (legacy behavior)")
    parser.add_argument('--save_reposed_smpl_skeleton_for_repose', action='store_true',
                       help="Save skeleton meshes along with reposed SMPL meshes for reshaping method. Forces --reposing_without_beta_change. Includes automatic mesh cleanup with LBS weight preservation.")
    parser.add_argument('--simplified_skeleton', action='store_true', default=True,
                       help="Use simplified 16-joint skeleton instead of 20-joint (default: True)")
    parser.add_argument('--no_simplified_skeleton', dest='simplified_skeleton', action='store_false',
                       help="Use full 20-joint skeleton instead of simplified 16-joint")


    parser.add_argument('--dataset_root', type=str,
                       help="Dataset root directory for LBS weight path resolution. If not provided, inferred from dat_dir.")
    parser.add_argument('--src_subject_id', type=str,
                       help="Source subject ID for reposing (for LBS weight loading). If not provided, inferred from dat_dir.")
    parser.add_argument('--smplx_segmentation_json_path', type=str,
                       help="Path to SMPL-X vertex segmentation JSON file for skin mask generation.")


    parser.add_argument(
        '--a2_generate_trim_meshes',
        action='store_true',
        help="Generate SMPL-X trimmed meshes for A2: smpl_body_trim.obj and smpl_reposed_frame_XXXX_trim.obj (no cleaning/simplification).",
    )
    parser.add_argument(
        '--a2_generate_correspondence',
        action='store_true',
        help="Generate A2 correspondence file (tri_id dist b0 b1 b2; tri_id=-1 for trimmed-face hits) for garment vertices.",
    )
    parser.add_argument(
        '--a2_overwrite',
        action='store_true',
        help="Overwrite existing A2 artifacts (trim meshes / correspondence file).",
    )
    parser.add_argument(
        '--a2_nn_scale_factor',
        type=float,
        default=1000.0,
        help="Scale factor passed to nearest-face CUDA op for A2 correspondence (default: 1000.0).",
    )
    parser.add_argument(
        '--a2_corr_output_path',
        type=str,
        default='',
        help="Optional explicit output path for A2 correspondence file. If empty, writes to <output_dir>/anim_.../a2_corr.txt.",
    )
    parser.add_argument(
        '--a2_visualize_correspondence',
        action='store_true',
        help="Write PLY visualizations for A2 correspondence on garment mesh and source_body_trim mesh.",
    )


    parser.add_argument(
        '--generate_smpl_body_cleaned',
        action='store_true',
        help=(
            "Generate <subject_root>/mesh/processed/smpl_body_cleaned.obj for both --dat_dir and "
            "--target_pose_dir subject roots (if smpl_body.obj exists). Useful for cloth-fit A1 volume-based annealing."
        ),
    )
    parser.add_argument(
        '--no_overwrite_smpl_body_cleaned',
        dest='overwrite_smpl_body_cleaned',
        action='store_false',
        help="Do not overwrite smpl_body_cleaned.obj if it already exists.",
    )
    parser.set_defaults(overwrite_smpl_body_cleaned=True)
    parser.add_argument(
        '--smpl_body_cleaned_targetperc',
        type=float,
        default=0.3,
        help="Target percentage for quadric decimation when generating smpl_body_cleaned.obj (default: 0.3).",
    )
    parser.add_argument(
        '--smpl_body_cleaned_maxholesize',
        type=int,
        default=50,
        help="Max hole size for hole closing when generating smpl_body_cleaned.obj (default: 30).",
    )

    return parser
