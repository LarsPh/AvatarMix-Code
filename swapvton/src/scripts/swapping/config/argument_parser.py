import argparse
import os
from pathlib import Path


def _default_smplx_segmentation_json_path() -> str:

    here = Path(__file__).resolve()

    root = Path(os.environ.get("AVATARMIX_ASSET_ROOT", str(here.parents[5] / "external_assets")))
    return str(root / "smplx_vert_segmentation.json")


def create_argument_parser():


    parser = argparse.ArgumentParser(
        description="Swap head between two 3D Gaussian avatars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    core_group = parser.add_argument_group('Core Arguments')
    core_group.add_argument("--user_A_id", type=str, default="0070", help="Subject ID for User A (head donor).")
    core_group.add_argument("--model_B_id", type=str, default="0083", help="Subject ID for Model B (body donor).")
    core_group.add_argument("--swap_both_directions", action=argparse.BooleanOptionalAction, default=True, help="Perform swap in both directions (A->B and B->A).")

    path_group = parser.add_argument_group('Path Arguments')
    path_group.add_argument("--data_root", type=str, required=True, help="Root directory of the dataset.")
    path_group.add_argument(
        "--smplx_seg_path",
        type=str,
        default=_default_smplx_segmentation_json_path(),
        help="Path to SMPLX vertex segmentation JSON.",
    )

    label_group = parser.add_argument_group('Labeling and Segmentation Arguments')
    label_group.add_argument("--use_detailed_labels", action=argparse.BooleanOptionalAction, default=True, help="Use detailed surface labels.")
    label_group.add_argument("--head_label", type=str, default="head", help="Label name for the head part.")
    label_group.add_argument("--fallback_skin_label", type=str, default="skin", help="Fallback label for skin if head label is not found or used.")
    label_group.add_argument("--hair_label", type=str, default="hair", help="Label name for the hair part.")
    label_group.add_argument("--label_dir_name", type=str, default="labeled", help="Directory name for label files (e.g., 'labeled' or 'labeled_no_sam'). Used for testing pipeline label variant selection.")
    label_group.add_argument(
        "--neck_plane_lift_ratio",
        type=float,
        default=0.4,
        help="Lift neck cut plane point toward head by this ratio (aggressive default).",
    )

    color_group = parser.add_argument_group('Color Transfer Arguments')

    color_group.add_argument("--no_color_transfer", dest="perform_color_transfer", action='store_false', help="Disable skin color transfer.")
    color_group.add_argument("--skin_label_for_color_transfer", type=str, default="skin", help="Label name for skin parts to be used in color transfer.")
    color_group.add_argument("--target_body_parts_for_color_transfer", nargs='+', default=["torso_skin", "left_arm", "right_arm", "left_leg", "right_leg"], help="List of body parts on the target to apply color transfer.")
    color_group.add_argument("--color_transfer_opacity_threshold", type=float, default=0.01, help="Opacity threshold for selecting Gaussians for color transfer stats.")
    color_group.add_argument("--color_transfer_use_all_gaussians_for_stats", action=argparse.BooleanOptionalAction, default=False, help="Use all Gaussians for color stats, ignoring opacity threshold.")
    color_group.add_argument("--color_transfer_use_opacity_weighting", action=argparse.BooleanOptionalAction, default=False, help="Use opacity-weighted statistics for color transfer.")
    color_group.add_argument(
        "--color_transfer_source_relative_neff_ratio",
        type=float,
        default=0.2,
        help=(
            "Relative support gate for A pooled sources: downgrade a source if its neff is below "
            "`ratio * max_neff` among primary pooled sources. Helps avoid choosing tiny/shadowed torso sources "
            "when arms/legs have much larger support."
        ),
    )
    color_group.add_argument(
        "--post_align_hands_to_arms",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After per-part transfer, apply a simple post-pass that shifts HANDS colors to match the "
            "opacity-weighted mean RGB of ARMS on the target (if arms exist). Helps mitigate opacity/color entanglement."
        ),
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_min_gaussians",
        type=int,
        default=2000,
        help="Minimum selected gaussians required for both hands and arms to run the post-align step.",
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_min_delta",
        type=float,
        default=0.02,
        help="Minimum L2 delta between (hands, arms) weighted mean RGB to trigger post-align.",
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_opacity_quantile",
        type=float,
        default=0.7,
        help=(
            "When post-aligning hands to arms, compute means on gaussians with opacity in the top quantile "
            "(e.g., 0.7 keeps top 30%%)."
        ),
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_min_dominant_gaussians",
        type=int,
        default=500,
        help="Minimum number of dominant (top-quantile opacity) gaussians required for hands and arms.",
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_k2_lab",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use a k=2 clustering correction for HANDS (dominant subset) in Lab space, then shift each cluster "
            "toward ARMS, including L (brightness)."
        ),
    )
    color_group.add_argument(
        "--post_align_hands_to_arms_k2_iters",
        type=int,
        default=10,
        help="Number of k-means iterations for k=2 hands clustering.",
    )

    debug_group = parser.add_argument_group('Debug Arguments')
    debug_group.add_argument("--debug_color_transfer_save_mesh", action=argparse.BooleanOptionalAction, default=True, help="Save intermediate meshes for debugging color transfer.")
    debug_group.add_argument("--debug_color_transfer_force_color", action=argparse.BooleanOptionalAction, default=False, help="Force a hardcoded color for color transfer source.")
    debug_group.add_argument("--debug_lab_mean", type=float, nargs=3, default=[50.0, 5.0, 10.0], help="Hardcoded Lab mean values (L*, a*, b*) for debug.")
    debug_group.add_argument("--debug_lab_std", type=float, nargs=3, default=[0.0001, 0.0001, 0.0001], help="Hardcoded Lab std values for debug.")
    debug_group.add_argument(
        "--save_swap_debug_artifacts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save additional debug artifacts under swapped output dir (e.g., swapped gaussians in body-donor space).",
    )
    debug_group.add_argument(
        "--save_head_body_separation_debug",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save head+neck and body Gaussians separately, plus a joints point-cloud, in both output space "
            "(A world) and body-donor space (B world / cloth-fit-aligned when applicable). Produces 3×2 PLYs "
            "under the swapped output dir."
        ),
    )

    body_reshape_group = parser.add_argument_group('Reshape Arguments')
    body_reshape_group.add_argument("--repose_load_dir_subfix", type=str, default="", help="Subfix for the reposed body directory to load, mainly reshaping related.")
    body_reshape_group.add_argument("--use_cloth_fit_reshaped_gs", action='store_true', default=False, help="Use cloth-fit reshaped Gaussians (file suffix: *_cloth_fit_reshaped.ply) instead of SMPL-based reshaping")
    body_reshape_group.add_argument("--cloth_fit_suffix", type=str, default="", help="Suffix for the cloth-fit reshaped Gaussians (file suffix: *_cloth_fit_reshaped.ply)")
    body_reshape_group.add_argument(
        "--cloth_fit_height_aware",
        action="store_true",
        default=False,
        help=(
            "Indicate the cloth-fit output was generated with height-aware normalization "
            "(translation restored, no global scaling). Enables using cloth-fit normalization offsets "
            "for head/body composition when --use_cloth_fit_reshaped_gs is set."
        ),
    )

    iteration_group = parser.add_argument_group('Iteration Arguments')
    iteration_group.add_argument("--user_A_iteration", type=int, default=None, help="Specific iteration number for User A (if not provided, uses 'latest')")
    iteration_group.add_argument("--model_B_iteration", type=int, default=None, help="Specific iteration number for Model B (if not provided, uses 'latest')")
    iteration_group.add_argument("--gender_A_for_pelvis_joint", type=str, default="neutral", help="SMPL gender for User A for pelvis joint extraction.")
    iteration_group.add_argument("--gender_B_for_pelvis_joint", type=str, default="neutral", help="SMPL gender for Model B for pelvis joint extraction.")

    mode_group = parser.add_argument_group('Swap Mode Arguments')
    mode_group.add_argument("--direct_swap_mode", action='store_true', help="Direct swap mode: use original B body instead of reposed B body")
    mode_group.add_argument("--enable_body_reshape", action='store_true', default=False, help="Enable body reshaping (prefer reshaped Gaussians when available, fallback to original)")
    mode_group.add_argument("--swap_back_mode", action='store_true', help="Swap-back mode: restore original identities using refined Gaussians from refinement pipeline")
    mode_group.add_argument("--enable_rigid_head_reposing", action='store_true', help="Use rigid head reposed PLY files with _rigidhead suffix")
    mode_group.add_argument("--save_body_donator_full_in_head_world", action='store_true', help="Save full body donator Gaussian (B_full) transformed to head donator's world for validation data preparation")
    mode_group.add_argument(
        "--swap_hands",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also swap both hands from head donor onto body donor (requires detailed 'hands' label).",
    )
    mode_group.add_argument(
        "--swap_hands_source",
        type=str,
        default="neus_gs",
        choices=["neus_gs", "smplx_mesh"],
        help=(
            "When --swap_hands is enabled, choose where the swapped hands come from:\n"
            "- neus_gs: swap hands Gaussians from A's avatar (default)\n"
            "- smplx_mesh: initialize dummy gray hands Gaussians from A's SMPL-X reposed-to-B mesh "
            "(smpl_reposed_frame_0000.obj; already in B space)"
        ),
    )
    mode_group.add_argument(
        "--smplx_hands_gaussians_per_hand",
        type=int,
        default=2000,
        help="For --swap_hands_source smplx_mesh: number of dummy Gaussians to sample per hand.",
    )
    mode_group.add_argument(
        "--smplx_hands_gray",
        type=float,
        default=0.6,
        help="For --swap_hands_source smplx_mesh: gray RGB value in [0,1] for dummy hand Gaussians.",
    )
    mode_group.add_argument(
        "--smplx_hands_scale_m",
        type=float,
        default=0.004,
        help="For --swap_hands_source smplx_mesh: Gaussian spatial scale (meters) for dummy hands (stored in log-space).",
    )

    return parser


def parse_arguments():

    parser = create_argument_parser()
    return parser.parse_args()
