from loguru import logger
from scripts.pipeline.execution.debug_support import get_debug_port
from scripts.pipeline.execution.subprocess_runner import run_command
from scripts.pipeline.execution.env_utils import build_python_command
from scripts.pipeline.sampling.subject_discovery import get_padded_subject_id, get_subject_iteration_and_batch_size, get_reshape_suffix_from_config, first_swap_reshape_enabled
import os
import shutil
import json
from pathlib import Path
import numpy as np
import trimesh
import pickle
import glob
import re


def get_gender_from_subject_root(subject_root: str) -> str:

    if not os.path.exists(os.path.join(subject_root, 'gender.txt')):
        logger.warning(f"Gender file not found for subject {subject_root}, using neutral")
        return 'neutral'
    else:
        gender = open(os.path.join(subject_root, 'gender.txt')).read().strip()
        if gender not in ['male', 'female', 'neutral']:
            logger.warning(f"Invalid gender {gender} for subject {subject_root}, using neutral")
            return 'neutral'
        return gender

def stage_reposing(config, subjects, debug_subprocess=False, operation_mode=None, subject_directions=None, force_enable_reshape=None, disable_renders=False,
                    cloth_fit_deformed_mesh_path=None, save_skeleton_for_cloth_fit=False, smplx_segmentation_json_path=None, cloth_fit_restoration_suffix=None,
                    cloth_fit_remove_feet=False, cloth_fit_deformed_mesh_already_restored: bool = False,
                    cloth_fit_keep_palm: bool = False,
                    cloth_fit_gs_scale_safety_enabled: bool = False,
                    cloth_fit_gs_scale_safety_percentile: float = 0.995,
                    cloth_fit_gs_scale_safety_hard_max_scale: float = 0.0,
                    cloth_fit_gs_scale_safety_ratio_enabled: bool = False,
                    cloth_fit_gs_scale_safety_ratio_percentile: float = 0.995,
                    cloth_fit_gs_scale_safety_ratio_hard_max: float = 0.0,
                    cloth_fit_gs_scale_safety_ratio_symmetric: bool = False,
                    cloth_fit_gs_scale_safety_action: str = "clamp",
                    cloth_fit_gs_scale_safety_eps_opacity: float = 1e-6,
                    generate_smpl_body_cleaned: bool = False, overwrite_smpl_body_cleaned: bool = True,
                    a2_generate_trim_meshes: bool = False,
                    a2_generate_correspondence: bool = False,
                    a2_overwrite: bool = True,
                    a2_nn_scale_factor: float = 1000.0,
                    a2_corr_output_path: str = "",
                    a2_visualize_correspondence: bool = False):

    direct_swap_mode = config.get('_direct_swap_mode', False)
    enable_reshape = force_enable_reshape if force_enable_reshape is not None else config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)


    if operation_mode == "reshaping":
        reshape_only_mode = True
        mode_desc = "Body Reshaping"
        cloth_fit_restoration_mode = False
    elif operation_mode == "reposing":
        reshape_only_mode = False
        mode_desc = "Head Reposing"
        cloth_fit_restoration_mode = False
    elif operation_mode == "cloth_fit_restoration":
        reshape_only_mode = False
        mode_desc = "Cloth-Fit Restoration"
        cloth_fit_restoration_mode = True
    else:

        reshape_only_mode = direct_swap_mode and enable_reshape
        mode_desc = "Reshape-Only" if reshape_only_mode else "Reposing"
        cloth_fit_restoration_mode = False

    logger.info(f"\n--- Stage 11: {mode_desc} ---")

    cfg = config['pipeline_stages']['11_reposing']
    reposing_cfg = config['pipeline_stages']['10_splatting_avatar']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['splatting']
    smpl_model_path = os.path.join(config['paths']['smpl_model_root'], 'smplx')
    script = 'repose_avatar.py'


    python_prefix = build_python_command(splatting_avatar_project, env)


    if subject_directions:

        subject_pairs = subject_directions
        gender_pairs = []
        for sub_a, sub_b in subject_directions:

            if len(subjects) > 2 or 'random' in str(config.get('_execution_mode', '')):

                gender_a = 'neutral'
                gender_b = 'neutral'
                logger.warning(f"Forcing neutral gender for random sampling pair ({sub_a}, {sub_b}) to avoid config mismatches")
                logger.info("TODO: Add automatic gender detection from subject data for future enhancement")
            else:

                gender_a = config.get('smpl_gender_a', 'neutral') if sub_a == subjects[0] else config.get('smpl_gender_b', 'neutral')
                gender_b = config.get('smpl_gender_a', 'neutral') if sub_b == subjects[0] else config.get('smpl_gender_b', 'neutral')
            gender_pairs.append((gender_a, gender_b))
    else:

        subject_pairs = [(subjects[0], subjects[1]), (subjects[1], subjects[0])]


        execution_mode = config.get('_execution_mode', '')
        if 'random' in execution_mode:
            logger.warning(f"Forcing neutral gender for random sampling ({execution_mode}) to avoid config mismatches")
            logger.info("TODO: Add automatic gender detection from subject data for future enhancement")
            gender_pairs = [('neutral', 'neutral'), ('neutral', 'neutral')]
        else:

            gender_pairs = [(config.get('smpl_gender_a', 'neutral'), config.get('smpl_gender_b', 'neutral')), (config.get('smpl_gender_b', 'neutral'), config.get('smpl_gender_a', 'neutral'))]

    if config.get('read_gender_from_subject_root', False):

        gender_pairs = [(get_gender_from_subject_root(os.path.join(avatarrex_dir, get_padded_subject_id(sub_a))), get_gender_from_subject_root(os.path.join(avatarrex_dir, get_padded_subject_id(sub_b)))) for sub_a, sub_b in subject_pairs]
    logger.info(f"Gender pairs: {gender_pairs}")

    for sub_ab, gender_ab in zip(subject_pairs, gender_pairs):
        sub_a, sub_b = sub_ab
        gender_a, gender_b = gender_ab
        action_desc = "reshaping" if reshape_only_mode else "reposing"
        logger.info(f"\n-- {action_desc.capitalize()} subject {sub_a} to {sub_b} --")
        sub_a_padded = get_padded_subject_id(sub_a)
        sub_b_padded = get_padded_subject_id(sub_b)


        subject_position = 'a' if sub_a == subjects[0] else 'b'
        iteration = get_subject_iteration_and_batch_size(config, sub_a, subject_position)
        model_path_a = reposing_cfg['model_path_template'].format(subject_id_padded=sub_a_padded)

        dat_dir = os.path.join(avatarrex_dir, sub_a_padded)
        target_pose_dir = os.path.join(avatarrex_dir, sub_b_padded)
        data_type = config['data_type']
        output_dir = os.path.join(avatarrex_dir, 'gs_on_mesh_repose', f'{data_type}_{sub_a_padded}_to_{sub_b_padded}')

        point_cloud_dir = os.path.join(avatarrex_dir, 'output-splatting', model_path_a, 'point_cloud', f'iteration_{iteration}')
        input_gs_ply = os.path.join(point_cloud_dir, 'point_cloud.ply')
        input_gs_embed = os.path.join(point_cloud_dir, 'embedding.json')

        nerf_mesh_path = os.path.join(avatarrex_dir, sub_a_padded, 'mesh', 'trimesh_cleaned', '0000.obj')
        lbs_weights_path = os.path.join(avatarrex_dir, sub_a_padded, 'mesh', 'processed', 'smoothed_inpainted_weights.npy')


        cmd = python_prefix + [
            script,
            '--configs', cfg['configs'],
            '--dat_dir', dat_dir,
            '--target_pose_dir', target_pose_dir,
            '--output_dir', output_dir,
            '--input_gs_ply', input_gs_ply,
            '--input_gs_embed', input_gs_embed,
            '--nerf_mesh_path', nerf_mesh_path,
            '--nerf_mesh_source_frame_idx', '0',
            '--target_frame_start', '0', '--target_frame_end', '1',
            '--smpl_model_path', smpl_model_path,
            '--smpl_gender_src', gender_a,
            '--smpl_gender_tgt', gender_b,
            '--gui_ip', 'none',
            '--save_posed_meshes',
            '--num_render_views', str(cfg['num_render_views']),
            '--save_reposed_gs_ply',
        ]
        if cloth_fit_restoration_suffix:
            cmd.append('--cloth_fit_restoration_suffix')
            cmd.append(cloth_fit_restoration_suffix)
        if cloth_fit_remove_feet:
            cmd.append('--cloth_fit_remove_feet')
            logger.info(f"   Removing feet for cloth-fit restoration")
        if cloth_fit_keep_palm:
            cmd.append('--cloth_fit_keep_palm')
            logger.info("   Keeping palm (leftHand/rightHand); only removing HandIndex1 regions for cloth-fit mask")

        if reshape_only_mode or cloth_fit_restoration_mode:
            cmd.extend(['--update_meshes'])
            cmd.extend(['--animation_mode', 'nerf_reshape_no_repose'])
            if reshape_only_mode:

                cmd.extend(['--enable_body_reshape'])
            else:
                cmd.extend(['--enable_cloth_fit_reshaping'])
                logger.info("Forcing update of meshes for cloth-fit restoration")
                if cloth_fit_deformed_mesh_path:
                    remove_head = config['pipeline_stages']['8_process_mesh'].get('remove_head', False)
                    cmd.append('--cloth_fit_deformed_mesh_path')
                    cmd.append(cloth_fit_deformed_mesh_path)
                    if cloth_fit_deformed_mesh_already_restored:
                        cmd.append('--cloth_fit_deformed_mesh_already_restored')
                    cmd.append('--cloth_fit_original_mesh_basename')
                    cmd.append('nerf_original.obj') if not remove_head else cmd.append('nerf_nonhead_original.obj')

                    cmd.append('--cloth_fit_simplified_mesh_basename')
                    cmd.append('nerf_simp_cleaned.obj') if not remove_head else cmd.append('nerf_nonhead_simp_cleaned.obj')
                    logger.info(f"   Adding cloth-fit restoration with deformed mesh: {cloth_fit_deformed_mesh_path}")
                else:
                    logger.warning("Cloth-fit restoration mode enabled but no deformed mesh path provided!")


                if cloth_fit_gs_scale_safety_enabled:
                    cmd.append('--cloth_fit_gs_scale_safety_enabled')
                    cmd.append('--cloth_fit_gs_scale_safety_percentile')
                    cmd.append(str(cloth_fit_gs_scale_safety_percentile))
                    cmd.append('--cloth_fit_gs_scale_safety_hard_max_scale')
                    cmd.append(str(cloth_fit_gs_scale_safety_hard_max_scale))
                    if cloth_fit_gs_scale_safety_ratio_enabled:
                        cmd.append('--cloth_fit_gs_scale_safety_ratio_enabled')
                        cmd.append('--cloth_fit_gs_scale_safety_ratio_percentile')
                        cmd.append(str(cloth_fit_gs_scale_safety_ratio_percentile))
                        cmd.append('--cloth_fit_gs_scale_safety_ratio_hard_max')
                        cmd.append(str(cloth_fit_gs_scale_safety_ratio_hard_max))
                        if cloth_fit_gs_scale_safety_ratio_symmetric:
                            cmd.append('--cloth_fit_gs_scale_safety_ratio_symmetric')
                    cmd.append('--cloth_fit_gs_scale_safety_action')
                    cmd.append(str(cloth_fit_gs_scale_safety_action))
                    cmd.append('--cloth_fit_gs_scale_safety_eps_opacity')
                    cmd.append(str(cloth_fit_gs_scale_safety_eps_opacity))
                    logger.info(
                        f"   Enabled cloth-fit GS scale safety: percentile={cloth_fit_gs_scale_safety_percentile} "
                        f"hard_max_scale={cloth_fit_gs_scale_safety_hard_max_scale} action={cloth_fit_gs_scale_safety_action}"
                    )
            logger.info(f"   Using reshape-only mode: reshaping {sub_a} to {sub_b}'s body shape")
        else:
            cmd.extend(['--animation_mode', 'nerf_mesh_repose'])
            cmd.extend(['--lbs_weights_path', lbs_weights_path])
            logger.info(f"   Using reposing mode: reposing {sub_a} to {sub_b}'s pose")


        if enable_reshape:
            if not reshape_only_mode:
                cmd.append('--enable_body_reshape')
            cmd.append('--body_reshape_smoothing_samples')
            cmd.append(str(cfg.get('body_reshape_smoothing_samples', 1)))
            cmd.append('--body_reshape_sample_std_scale')
            cmd.append(str(cfg.get('body_reshape_sample_std_scale', 1.0)))
            cmd.append('--body_reshape_scale_factor')
            cmd.append(str(cfg.get('body_reshape_scale_factor', 1.0)))
            if cfg.get('body_reshape_distance_weighting', False):
                cmd.append('--body_reshape_distance_weighting')
            if cfg.get('save_reshaping_meshes', False):
                cmd.append('--save_reshaping_meshes')
        if disable_renders:
            cmd.append('--disable_renders')


        if save_skeleton_for_cloth_fit:
            cmd.append('--save_reposed_smpl_skeleton_for_repose')
            cmd.append('--simplified_skeleton')

            if '--update_meshes' not in cmd:
                cmd.append('--update_meshes')
            logger.info("   Saving reposed SMPL skeleton for cloth-fit reshaping")


        if smplx_segmentation_json_path:
            cmd.append('--smplx_segmentation_json_path')
            cmd.append(str(smplx_segmentation_json_path))
            logger.info(f"   Using SMPL-X segmentation JSON: {smplx_segmentation_json_path}")


        if generate_smpl_body_cleaned:
            cmd.append('--generate_smpl_body_cleaned')
            if not overwrite_smpl_body_cleaned:
                cmd.append('--no_overwrite_smpl_body_cleaned')
            logger.info(f"   Generating smpl_body_cleaned.obj (overwrite={overwrite_smpl_body_cleaned})")


        if a2_generate_trim_meshes:
            cmd.append('--a2_generate_trim_meshes')
        if a2_generate_correspondence:
            cmd.append('--a2_generate_correspondence')
        if (a2_generate_trim_meshes or a2_generate_correspondence) and a2_overwrite:
            cmd.append('--a2_overwrite')
        if a2_generate_correspondence:
            cmd.append('--a2_nn_scale_factor')
            cmd.append(str(a2_nn_scale_factor))
            if a2_corr_output_path:
                cmd.append('--a2_corr_output_path')
                cmd.append(str(a2_corr_output_path))
            if a2_visualize_correspondence:
                cmd.append('--a2_visualize_correspondence')
            logger.info(
                f"   A2 enabled: trim={a2_generate_trim_meshes} corr={a2_generate_correspondence} "
                f"overwrite={a2_overwrite} nn_scale_factor={a2_nn_scale_factor}"
            )


        if cfg.get('save_gt_aligned_head', False):
            cmd.append('--save_gt_aligned_head')
            logger.info(f"   Adding --save_gt_aligned_head for GT-aligned head pipeline")


        stage_key = '11a_head_donator_reposing'
        if stage_key in config['pipeline_stages'] and config['pipeline_stages'][stage_key].get('save_gt_aligned_head', False):
            if '--save_gt_aligned_head' not in cmd:
                cmd.append('--save_gt_aligned_head')
            cmd.append('--enable_head_alignment_filtering')
            logger.info(f"   Adding --save_gt_aligned_head --enable_head_alignment_filtering for GT-aligned head pipeline")


            from pipeline.core.config import use_label_no_sam_for_alignment
            if use_label_no_sam_for_alignment(config, stage_key):
                cmd.append('--label_dir_name')
                cmd.append('labeled_no_sam')
                logger.info(f"   Using labeled_no_sam for better head alignment")


        enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)
        if enable_rigid_head_reposing:
            cmd.append('--enable_rigid_head_reposing')
            logger.info(f"   Adding --enable_rigid_head_reposing for rigid head pipeline")

        debug_port = get_debug_port('reposing') if debug_subprocess else None
        run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)

def stage_swapping(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 12: Swapping ---")
    cfg = config['pipeline_stages']['12_swapping']
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']
    reposing_cfg = config['pipeline_stages']['11_reposing']
    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    module_name = 'src.scripts.swapping.head_swap'


    logger.info("\n-- Preparing latest splatting outputs --")
    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        iteration_key = f'iteration_{sub}'
        if iteration_key not in splatting_cfg:
            logger.warning(f"Warning: iteration for subject {sub} not found in config. Skipping latest output preparation.")
            continue


        subject_position = 'a' if sub == subjects[0] else 'b'
        iteration_val, batch_size, iteration = get_subject_iteration_and_batch_size(config, sub, subject_position)
        model_path = splatting_cfg['model_path_template'].format(subject_id_padded=sub_padded)

        src_dir = os.path.join(avatarrex_dir, 'output-splatting', model_path, 'point_cloud', f'iteration_{iteration}')
        dest_dir = os.path.join(avatarrex_dir, 'output-splatting', model_path, 'point_cloud', 'latest')

        if not os.path.exists(src_dir):
            logger.warning(f"Warning: Source directory for latest output does not exist, skipping: {src_dir}")
            continue

        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)

        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dest_dir, item)
            if os.path.isfile(s):
                logger.info(f"Copying {s} to {d}")
                shutil.copy2(s, d)


    subject_a_position = 'a'
    subject_b_position = 'b'
    subject_a_iteration = get_subject_iteration_and_batch_size(config, subjects[0], subject_a_position)
    subject_b_iteration = get_subject_iteration_and_batch_size(config, subjects[1], subject_b_position)


    cmd = [
        'uv', 'run', 'python', '-m', module_name,
        '--user_A_id', get_padded_subject_id(subjects[0]),
        '--model_B_id', get_padded_subject_id(subjects[1]),
        '--user_A_iteration', str(subject_a_iteration),
        '--model_B_iteration', str(subject_b_iteration),
    ]
    if cfg.get('swap_both_directions', False):
        cmd.append('--swap_both_directions')

    if cfg.get('enable_body_reshape', True):
        cmd.append('--repose_load_dir_subfix')
        reshape_subfix = get_reshape_suffix_from_config(config)[1:]
        cmd.append(reshape_subfix)
        cmd.append('--enable_body_reshape')

    enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)
    if enable_rigid_head_reposing:
        cmd.append('--enable_rigid_head_reposing')


    if cfg.get("save_head_body_separation_debug", False):
        cmd.append("--save_head_body_separation_debug")

    if cfg.get("swap_hands", False):
        cmd.append("--swap_hands")

        if "swap_hands_source" in cfg:
            cmd.extend(["--swap_hands_source", str(cfg.get("swap_hands_source"))])
        if "smplx_hands_gaussians_per_hand" in cfg:
            cmd.extend(["--smplx_hands_gaussians_per_hand", str(int(cfg.get("smplx_hands_gaussians_per_hand")))])
        if "smplx_hands_gray" in cfg:
            cmd.extend(["--smplx_hands_gray", str(float(cfg.get("smplx_hands_gray")))])
        if "smplx_hands_scale_m" in cfg:
            cmd.extend(["--smplx_hands_scale_m", str(float(cfg.get("smplx_hands_scale_m")))])

    debug_port = get_debug_port('swapping') if debug_subprocess else None
    run_command(cmd, cwd=project_root, debug_port=debug_port)


def merge_maybe_patch_meshes(
    *,
    project_root: str,
    swap_dir: str,
    stage_cfg: dict,
    debug_subprocess: bool = False,
):

    merge_cfg = stage_cfg.get("merge_maybe_patch_meshes", {}) if isinstance(stage_cfg, dict) else {}
    if not bool(merge_cfg.get("enabled", False)):
        return

    extra_args = merge_cfg.get("extra_args", [])
    if extra_args is None:
        extra_args = []
    if not isinstance(extra_args, list):
        logger.warning("merge_maybe_patch_meshes.extra_args should be a list[str]; ignoring.")
        extra_args = []

    cmd = [
        "uv",
        "run",
        "python",
        "src/scripts/mesh_transplant/smplx_neck_bridge_patch.py",
        "--swap_dir",
        str(swap_dir),
    ] + [str(x) for x in extra_args]

    logger.info(f"Running merge_maybe_patch_meshes: swap_dir={swap_dir}")
    debug_port = get_debug_port("merge_maybe_patch_meshes") if debug_subprocess else None
    run_command(cmd, cwd=project_root, debug_port=debug_port)


def _load_cloth_fit_offsets_json(
    *,
    dataset_root: str,
    avatar_subject_id: str,
    garment_subject_id: str,
    cloth_fit_suffix: str,
) -> dict | None:

    cloth_fit_root = Path(str(dataset_root)) / "cloth_fit_output"
    if not cloth_fit_root.exists():
        return None

    avatar = str(avatar_subject_id).strip()
    garment = str(garment_subject_id).strip()
    suf_raw = str(cloth_fit_suffix or "").strip()


    suf_variants: list[str] = []
    if suf_raw:
        suf_variants.append(suf_raw)
        if suf_raw.startswith("_"):
            suf_variants.append(suf_raw.lstrip("_"))
        else:
            suf_variants.append("_" + suf_raw)
    else:
        suf_variants.append("")

    for suf in suf_variants:
        exact = cloth_fit_root / f"{avatar}_avatar_{garment}_garment{suf}" / "normalization_offsets.json"
        if exact.exists():
            try:
                return json.loads(exact.read_text())
            except Exception:
                return None

    prefix = f"{avatar}_avatar_{garment}_garment"
    cands = sorted([p for p in cloth_fit_root.glob(f"{prefix}*/normalization_offsets.json") if p.is_file()])
    if not cands:
        return None


    chosen = None
    if len(cands) > 1 and suf_raw:

        for suf in suf_variants:
            if not suf:
                continue
            preferred = [p for p in cands if p.parent.name.endswith(f"_garment{suf}")]
            if preferred:
                chosen = sorted(preferred)[0]
                break

        if chosen is None:
            preferred = [p for p in cands if suf_raw in p.parent.name]
            if preferred:
                chosen = sorted(preferred)[0]

    if chosen is None:
        chosen = cands[0]

    if len(cands) > 1:
        logger.warning(
            f"Multiple cloth-fit offsets found for {prefix}; "
            f"cloth_fit_suffix='{suf_raw}' -> using: {chosen}"
        )
    try:
        return json.loads(chosen.read_text())
    except Exception:
        return None


def _compact_submesh_from_keep_mask(V: np.ndarray, F: np.ndarray, keep_v: np.ndarray):

    V = np.asarray(V, dtype=np.float32)
    F = np.asarray(F, dtype=np.int64)
    keep_v = np.asarray(keep_v).astype(bool).reshape(-1)
    if keep_v.shape[0] != V.shape[0]:
        raise ValueError(f"keep_v Nv mismatch: {keep_v.shape[0]} vs {V.shape[0]}")
    face_keep = np.all(keep_v[F], axis=1)
    Fk = F[face_keep]
    if Fk.size == 0:
        raise ValueError("No faces kept after applying keep mask.")
    used = np.unique(Fk.reshape(-1))
    used.sort()
    vmap = -np.ones((V.shape[0],), dtype=np.int64)
    vmap[used] = np.arange(used.shape[0], dtype=np.int64)
    V2 = V[used]
    F2 = vmap[Fk]
    return V2, F2.astype(np.int64), used.astype(np.int64)


def _maybe_transfer_body_lbs_from_avatar_smpl(
    *,
    config: dict,
    stage_cfg: dict,
    project_root: str,
    swap_dir: str,
    head_id_raw: str,
    body_id_raw: str,
    debug_subprocess: bool = False,
):

    tcfg = stage_cfg.get("transfer_body_lbs_from_avatar_smpl", {}) if isinstance(stage_cfg, dict) else {}
    if not bool((tcfg or {}).get("enabled", False)):
        return

    swap_dir_p = Path(str(swap_dir))
    if not swap_dir_p.exists():
        logger.warning(f"[body_lbs_transfer] swap_dir not found: {swap_dir_p}")
        return

    avatarrex_root = str(config["paths"]["avatarrex_output"])
    dataset_type = str(config.get("data_type", ""))


    from pipeline.core.config import use_cloth_fit_reshaped_gs
    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, "12b_direct_swapping")
    if not cloth_fit_enabled:
        logger.warning("[body_lbs_transfer] cloth-fit is disabled; skipping.")
        return


    offsets = _load_cloth_fit_offsets_json(
        dataset_root=avatarrex_root,
        avatar_subject_id=get_padded_subject_id(head_id_raw),
        garment_subject_id=get_padded_subject_id(body_id_raw),
        cloth_fit_suffix=str(cloth_fit_suffix),
    )
    if offsets is None or ("source_offset" not in offsets) or ("target_offset" not in offsets):
        logger.warning("[body_lbs_transfer] normalization_offsets.json not found (or missing keys); skipping.")
        return

    source_offset = np.asarray(offsets["source_offset"], dtype=np.float32).reshape(3,)
    target_offset = np.asarray(offsets["target_offset"], dtype=np.float32).reshape(3,)

    swap_hands_enabled = bool(stage_cfg.get("swap_hands", False))
    transfer_mode = str((tcfg or {}).get("mode", "body_and_hands")).strip().lower()


    body_only_keep_b_hands = (not swap_hands_enabled) and (transfer_mode in {"body_only", "only_body", "body_only_keep_b_hands"})


    swap_hands_transfer_hands = swap_hands_enabled and (transfer_mode in {"body_and_hands", "body_plus_hands", "full"})

    overwrite_hands_weights = bool((tcfg or {}).get("overwrite_hands_weights", True))
    out_name = "smoothed_inpainted_weights_B_body_no_hands.npy" if swap_hands_enabled else "smoothed_inpainted_weights_B_body_only.npy"
    out_path = swap_dir_p / out_name
    backup_path = swap_dir_p / out_name.replace(".npy", "_old.npy")

    work_dir = swap_dir_p / "body_lbs_transfer"
    work_dir.mkdir(parents=True, exist_ok=True)
    marker = work_dir / "run_info.json"
    skip_existing = bool((tcfg or {}).get("skip_existing", True))
    if skip_existing and out_path.exists() and backup_path.exists() and marker.exists():
        logger.info(f"[body_lbs_transfer] skip_existing: found outputs at {out_path}; skipping.")
        return


    W_target_full_orig = None
    if body_only_keep_b_hands:
        if not out_path.exists():
            logger.warning(
                f"[body_lbs_transfer] mode={transfer_mode}: expected existing target weights to keep B hands, "
                f"but missing: {out_path}. Skipping."
            )
            return
        try:
            W_target_full_orig = np.asarray(np.load(str(out_path))).astype(np.float32)
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed to load existing target weights {out_path}: {e}")
            return
    if out_path.exists() and (not backup_path.exists()):
        try:
            shutil.copyfile(str(out_path), str(backup_path))
            logger.info(f"[body_lbs_transfer] backed up existing weights: {backup_path.name}")
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed to backup {out_path}: {e}")


    head_id = get_padded_subject_id(head_id_raw)
    body_id = get_padded_subject_id(body_id_raw)
    smpl_reposed_dir = Path(avatarrex_root) / "gs_on_mesh_repose" / f"{dataset_type}_{head_id}_to_{body_id}" / "anim_0000_to_targets_meshes"
    smpl_mesh_cands = [
        smpl_reposed_dir / "smpl_reposed_frame_0000.obj",
        smpl_reposed_dir / "smpl_reposed_frame_0000_cleaned.obj",
    ]
    smpl_mesh_path = next((p for p in smpl_mesh_cands if p.exists()), None)
    if smpl_mesh_path is None:
        logger.warning(f"[body_lbs_transfer] SMPL reposed mesh not found under: {smpl_reposed_dir}")
        return

    smpl_w_path = Path(avatarrex_root) / head_id / "mesh" / "processed" / "smpl_lbs_weights.npy"
    if not smpl_w_path.exists():
        logger.warning(f"[body_lbs_transfer] SMPL LBS weights not found: {smpl_w_path}")
        return

    smplx_seg_path = Path(str(config.get("paths", {}).get("smplx_segmentation_json", "")))
    if not smplx_seg_path.exists():
        logger.warning(f"[body_lbs_transfer] SMPL-X vert segmentation json not found: {smplx_seg_path}")
        return

    try:
        smpl_mesh = trimesh.load(str(smpl_mesh_path), process=False, force="mesh")
        V_s = np.asarray(smpl_mesh.vertices, dtype=np.float32)
        F_s = np.asarray(smpl_mesh.faces, dtype=np.int64)
        W_s_full = np.asarray(np.load(str(smpl_w_path))).astype(np.float32)
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to load SMPL source mesh/weights: {e}")
        return

    if W_s_full.ndim != 2 or W_s_full.shape[0] != V_s.shape[0]:
        logger.warning(f"[body_lbs_transfer] source weights Nv mismatch: W {getattr(W_s_full, 'shape', None)} vs V {V_s.shape}")
        return

    try:
        seg = json.loads(smplx_seg_path.read_text())


        drop_keys = ["head", "leftEye", "rightEye"]
        if (swap_hands_enabled and (not swap_hands_transfer_hands)) or body_only_keep_b_hands:
            drop_keys += ["leftHand", "rightHand", "leftHandIndex1", "rightHandIndex1"]
        drop_ids = []
        for k in drop_keys:
            ids = seg.get(k, [])
            if isinstance(ids, list):
                drop_ids.extend([int(x) for x in ids if isinstance(x, (int, float))])
        drop_ids = np.unique(np.asarray(drop_ids, dtype=np.int64))
        keep_v = np.ones((V_s.shape[0],), dtype=bool)
        drop_ids = drop_ids[(drop_ids >= 0) & (drop_ids < V_s.shape[0])]
        keep_v[drop_ids] = False
        V_s2, F_s2, used_s = _compact_submesh_from_keep_mask(V_s, F_s, keep_v)
        W_s2 = W_s_full[used_s]
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to build source body-only SMPL submesh: {e}")
        return


    V_s2_cf = (V_s2 - target_offset.reshape(1, 3)).astype(np.float32)


    target_mesh_path = None
    if swap_hands_enabled:
        for n in ("body_no_hands_world_B_body_donor.obj", "body_only_world_B_body_donor.obj"):
            p = swap_dir_p / n
            if p.exists():
                target_mesh_path = p
                break
    else:
        for n in ("body_only_world_B_body_donor.obj",):
            p = swap_dir_p / n
            if p.exists():
                target_mesh_path = p
                break

    if target_mesh_path is None:


        try:
            label_dir_name = str(stage_cfg.get("label_dir_name", "labeled"))
            label_suffix = "_extended"
            label_cands = [
                Path(avatarrex_root) / body_id / "mesh" / label_dir_name / f"label-f0000{label_suffix}.pkl",
                Path(avatarrex_root) / body_id / "mesh" / label_dir_name / "label-f0000.pkl",
            ]
            label_path = next((p for p in label_cands if p.exists()), None)
            if label_path is None:
                raise FileNotFoundError(f"Missing body-donor label pkl (tried: {label_cands})")

            obj_dir = Path(avatarrex_root) / "gs_on_mesh_repose" / f"{dataset_type}_{body_id}_to_{head_id}" / "reshaped_gaussians_ply_0000"
            if not obj_dir.exists():
                raise FileNotFoundError(f"Missing reshaped_gaussians_ply_0000 dir: {obj_dir}")


            ply_exact = obj_dir / f"reshaped_gs_target_shape_0000.ply_cloth_fit_reshaped{cloth_fit_suffix}.ply"
            ply_path = ply_exact if ply_exact.exists() else None
            if ply_path is None:
                plys = sorted(obj_dir.glob("reshaped_gs_target_shape_0000.ply_cloth_fit_reshaped*.ply"))
                if not plys:
                    raise FileNotFoundError(f"No cloth-fit reshaped PLYs found under: {obj_dir}")
                ply_path = plys[0]


            bn = ply_path.name
            m = re.search(r"reshaped_gs_target_shape_(\\d{4})", bn)
            frame = (m.group(1) if m else "0000")
            base_dir = ply_path.parent.parent
            mesh_dir = base_dir / f"reshape_no_repose_{frame}_meshes"
            suf = ""
            m2 = re.search(r"cloth_fit_reshaped(.*)\\.ply$", bn)
            if m2:
                suf = m2.group(1) or ""
            mesh_cands = [
                mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}_cloth_fit_reshaped{suf}.obj",
                mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}_cloth_fit_reshaped.obj",
                mesh_dir / f"reshaped_nerf_source_pose_target_shape_{frame}.obj",
            ]
            full_mesh_path = next((p for p in mesh_cands if p.exists()), None)
            if full_mesh_path is None:
                raise FileNotFoundError(f"Could not infer restored mesh from PLY; tried: {mesh_cands}")


            obj = pickle.loads(label_path.read_bytes())
            if isinstance(obj, dict) and ("scan_labels" in obj):
                labels = np.asarray(obj["scan_labels"], dtype=np.int64).reshape(-1)
            else:
                labels = np.asarray(obj, dtype=np.int64).reshape(-1)


            detailed = ["torso_skin", "head", "left_arm", "right_arm", "left_leg", "right_leg", "clothes", "hands", "shoes"]
            head_idx = detailed.index("head")
            hands_idx = detailed.index("hands")
            drop = {head_idx}
            if swap_hands_enabled:
                drop.add(hands_idx)

            m_full = trimesh.load(str(full_mesh_path), process=False, force="mesh")
            V_full = np.asarray(m_full.vertices, dtype=np.float32)
            F_full = np.asarray(m_full.faces, dtype=np.int64)
            if labels.shape[0] != V_full.shape[0]:
                raise ValueError(f"Label Nv mismatch: {labels.shape[0]} vs mesh Nv {V_full.shape[0]} ({full_mesh_path})")
            keep_v = ~np.isin(labels, np.asarray(sorted(list(drop)), dtype=np.int64))
            V_keep, F_keep, _ = _compact_submesh_from_keep_mask(V_full, F_full, keep_v)


            V_keep_aligned = (V_keep - source_offset.reshape(1, 3)) + target_offset.reshape(1, 3)
            fallback_name = "body_no_hands_world_B_body_donor.obj" if swap_hands_enabled else "body_only_world_B_body_donor.obj"
            target_mesh_path = swap_dir_p / fallback_name
            trimesh.Trimesh(vertices=V_keep_aligned, faces=F_keep, process=False).export(str(target_mesh_path))
            logger.info(f"[body_lbs_transfer] wrote fallback target mesh: {target_mesh_path.name}")
        except Exception as e_fallback:
            logger.warning(
                "[body_lbs_transfer] target body mesh not found under swap_dir and fallback inference failed. "
                f"swap_dir={swap_dir_p} err={e_fallback}"
            )
            return

    try:
        tgt_mesh = trimesh.load(str(target_mesh_path), process=False, force="mesh")
        V_t = np.asarray(tgt_mesh.vertices, dtype=np.float32)
        F_t = np.asarray(tgt_mesh.faces, dtype=np.int64)
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to load target mesh {target_mesh_path}: {e}")
        return
    if body_only_keep_b_hands and (W_target_full_orig is not None):
        if W_target_full_orig.ndim != 2 or int(W_target_full_orig.shape[0]) != int(V_t.shape[0]):
            logger.warning(
                f"[body_lbs_transfer] mode={transfer_mode}: existing target weights Nv mismatch; "
                f"W {getattr(W_target_full_orig, 'shape', None)} vs V {V_t.shape} ({out_path.name}). Skipping."
            )
            return


    used_t = None
    split_meta = None
    if body_only_keep_b_hands:

        label_cands = [
            swap_dir_p / "label_B_body_only.pkl",
            swap_dir_p / "label_B_body_no_hands.pkl",
        ]
        label_path = next((p for p in label_cands if p.exists()), None)
        if label_path is None:
            logger.warning(f"[body_lbs_transfer] mode={transfer_mode}: missing body segmentation pkl under swap_dir (tried: {label_cands}); skipping.")
            return
        try:
            obj = pickle.loads(label_path.read_bytes())
            if isinstance(obj, dict) and ("scan_labels" in obj):
                labels_t = np.asarray(obj["scan_labels"], dtype=np.int64).reshape(-1)
            else:
                labels_t = np.asarray(obj, dtype=np.int64).reshape(-1)
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed to load target seg labels: {label_path} err={e}")
            return
        if labels_t.shape[0] != V_t.shape[0]:
            logger.warning(f"[body_lbs_transfer] target seg Nv mismatch: {labels_t.shape[0]} vs V {V_t.shape[0]} ({label_path})")
            return

        detailed = ["torso_skin", "head", "left_arm", "right_arm", "left_leg", "right_leg", "clothes", "hands", "shoes"]
        hands_idx = int(detailed.index("hands"))
        keep_v_body = labels_t != hands_idx
        try:
            V_t2, F_t2, used_t = _compact_submesh_from_keep_mask(V_t, F_t, keep_v_body)
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] mode={transfer_mode}: failed to build target body-only submesh: {e}")
            return

        V_t = V_t2
        F_t = F_t2
        logger.info(
            f"[body_lbs_transfer] mode={transfer_mode}: target cut hands for transfer: Nv {int(V_t2.shape[0])} (full {int(labels_t.shape[0])})"
        )
    elif swap_hands_transfer_hands:

        body_no_hands_path = swap_dir_p / "body_no_hands_world_B_body_donor.obj"
        hands_path = swap_dir_p / "hands_world_B_body_donor.obj"
        if not body_no_hands_path.exists():
            logger.warning(f"[body_lbs_transfer] mode={transfer_mode}: missing body_no_hands mesh: {body_no_hands_path}")
            return
        if not hands_path.exists():
            logger.warning(f"[body_lbs_transfer] mode={transfer_mode}: missing hands mesh: {hands_path}")
            return
        try:
            m_b = trimesh.load(str(body_no_hands_path), process=False, force="mesh")
            V_b = np.asarray(m_b.vertices, dtype=np.float32)
            F_b = np.asarray(m_b.faces, dtype=np.int64)
            m_h = trimesh.load(str(hands_path), process=False, force="mesh")
            V_h = np.asarray(m_h.vertices, dtype=np.float32)
            F_h = np.asarray(m_h.faces, dtype=np.int64)
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] mode={transfer_mode}: failed to load combined target meshes: {e}")
            return
        off = int(V_b.shape[0])
        V_t = np.concatenate([V_b, V_h], axis=0).astype(np.float32)
        F_t = np.concatenate([F_b, (F_h + off)], axis=0).astype(np.int64)
        split_meta = {"nv_body_no_hands": int(V_b.shape[0]), "nv_hands": int(V_h.shape[0]), "hands_path": str(hands_path)}
        try:
            (work_dir / "target_split_meta.json").write_text(json.dumps(split_meta, indent=2) + "\n")
        except Exception:
            pass

        target_mesh_path = body_no_hands_path
        logger.info(
            f"[body_lbs_transfer] mode={transfer_mode}: combined target Nv={int(V_b.shape[0])}+{int(V_h.shape[0])}={int(V_t.shape[0])}"
        )


    if "_B_body_donor" in str(target_mesh_path.name):
        V_t_cf = (V_t - target_offset.reshape(1, 3)).astype(np.float32)
    else:
        V_t_cf = (V_t - source_offset.reshape(1, 3)).astype(np.float32)


    def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:

        h = np.asarray(h, dtype=np.float32)
        s = np.asarray(s, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        h6 = (h % 1.0) * 6.0
        i = np.floor(h6).astype(np.int32)
        f = (h6 - i).astype(np.float32)
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i_mod = (i % 6).astype(np.int32)
        r = np.zeros_like(v, dtype=np.float32)
        g = np.zeros_like(v, dtype=np.float32)
        b = np.zeros_like(v, dtype=np.float32)
        m0 = i_mod == 0
        m1 = i_mod == 1
        m2 = i_mod == 2
        m3 = i_mod == 3
        m4 = i_mod == 4
        m5 = i_mod == 5
        r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
        r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
        r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
        r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
        r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
        r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
        return np.stack([r, g, b], axis=1)

    def _make_joint_palette(K: int) -> np.ndarray:

        K = int(K)
        if K <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        hues = (np.arange(K, dtype=np.float32) * np.float32(0.61803398875)) % np.float32(1.0)
        s = np.full((K,), 0.85, dtype=np.float32)
        v = np.full((K,), 0.95, dtype=np.float32)
        return _hsv_to_rgb(hues, s, v).astype(np.float32)

    def _lbs_to_vertex_colors_topk_blend(W: np.ndarray, k: int = 4) -> np.ndarray:

        W = np.asarray(W, dtype=np.float32)
        if W.ndim != 2:
            raise ValueError(f"W must be (N,K), got {W.shape}")
        N, K = int(W.shape[0]), int(W.shape[1])
        if N == 0 or K == 0:
            return np.zeros((N, 4), dtype=np.uint8)
        palette = _make_joint_palette(K)

        row_sum = np.maximum(W.sum(axis=1, keepdims=True), 1e-12).astype(np.float32)
        P = (W / row_sum).astype(np.float32)
        k_eff = int(min(max(int(k), 1), K))

        idx = np.argpartition(-P, kth=(k_eff - 1), axis=1)[:, :k_eff]
        w_top = np.take_along_axis(P, idx, axis=1)
        w_top = w_top / np.maximum(w_top.sum(axis=1, keepdims=True), 1e-12)
        rgb = (palette[idx] * w_top[:, :, None]).sum(axis=1)
        rgba = np.concatenate([rgb, np.ones((N, 1), dtype=np.float32)], axis=1)
        rgba_u8 = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)
        return rgba_u8

    src_obj = work_dir / "tmp_source_A_smpl_body_cf.ply"
    src_w = work_dir / "tmp_source_A_smpl_body_lbs.npy"
    tgt_obj = work_dir / (
        "tmp_target_B_body_plus_hands_cf.ply"
        if swap_hands_transfer_hands
        else ("tmp_target_B_body_no_hands_cf.ply" if body_only_keep_b_hands else "tmp_target_B_body_cf.ply")
    )
    out_smoothed = work_dir / "smoothed_inpainted_weights.npy"
    out_inpainted = work_dir / "inpainted_weights.npy"
    viz_dir = work_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    trimesh.Trimesh(vertices=V_s2_cf, faces=F_s2, process=False).export(str(src_obj))
    np.save(str(src_w), W_s2.astype(np.float32))
    trimesh.Trimesh(vertices=V_t_cf, faces=F_t, process=False).export(str(tgt_obj))

    try:
        src_vis = work_dir / "tmp_source_A_smpl_body_cf_lbsvis_topk4.ply"
        src_colors = _lbs_to_vertex_colors_topk_blend(W_s2, k=4)
        trimesh.Trimesh(vertices=V_s2_cf, faces=F_s2, process=False, vertex_colors=src_colors).export(str(src_vis))
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to write source LBS visualization PLY: {e}")


    try:
        seg2 = json.loads(smplx_seg_path.read_text())
        drop_keys_vis = ["head", "leftEye", "rightEye"]
        drop_ids_vis = []
        for k in drop_keys_vis:
            ids = seg2.get(k, [])
            if isinstance(ids, list):
                drop_ids_vis.extend([int(x) for x in ids if isinstance(x, (int, float))])
        drop_ids_vis = np.unique(np.asarray(drop_ids_vis, dtype=np.int64))
        keep_v_vis = np.ones((V_s.shape[0],), dtype=bool)
        drop_ids_vis = drop_ids_vis[(drop_ids_vis >= 0) & (drop_ids_vis < V_s.shape[0])]
        keep_v_vis[drop_ids_vis] = False
        V_vis, F_vis, used_vis = _compact_submesh_from_keep_mask(V_s, F_s, keep_v_vis)
        W_vis = W_s_full[used_vis]
        src_full_ply = work_dir / "tmp_source_A_smpl_no_head_with_hands_world_B_body_donor.ply"
        src_full_vis = work_dir / "tmp_source_A_smpl_no_head_with_hands_world_B_body_donor_lbsvis_topk4.ply"
        trimesh.Trimesh(vertices=V_vis, faces=F_vis, process=False).export(str(src_full_ply))
        C_vis = _lbs_to_vertex_colors_topk_blend(W_vis, k=4)
        trimesh.Trimesh(vertices=V_vis, faces=F_vis, process=False, vertex_colors=C_vis).export(str(src_full_vis))
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to write SMPLX no-head-with-hands viz PLY: {e}")

    dist_thr = float((tcfg or {}).get("distance_threshold_m", 0.01))
    ang_thr = float((tcfg or {}).get("angle_threshold_deg", 30.0))

    robust_lbs_project = config["paths"]["robust_lbs_project"]
    env = config["conda_envs"]["skw_transfer"]
    python_prefix = build_python_command(robust_lbs_project, env)

    cmd = python_prefix + [
        "src/avatarrex_transfer.py",
        "--data_dir", str(work_dir),
        "--output_dir", str(work_dir),
        "--dataset_type", "generic",
        "--source_mesh_path", str(src_obj),
        "--source_lbs_weights_path", str(src_w),
        "--target_mesh_path", str(tgt_obj),
        "--out_inpainted_weights_path", str(out_inpainted),
        "--out_smoothed_weights_path", str(out_smoothed),
        "--distance_threshold_m", str(dist_thr),
        "--angle_threshold_deg", str(ang_thr),
        "--viz_dir", str(viz_dir),
        "--explicit_frame_id", "0000",
    ]

    logger.info(f"[body_lbs_transfer] running robust transfer for head={head_id_raw} body={body_id_raw}")
    logger.info(f"[body_lbs_transfer]  source_smpl_mesh: {smpl_mesh_path}")
    logger.info(f"[body_lbs_transfer]  target_mesh: {target_mesh_path}")
    logger.info(f"[body_lbs_transfer]  out_weights: {out_path.name} (backup={backup_path.name})")
    debug_port = get_debug_port("body_lbs_transfer") if debug_subprocess else None
    run_command(cmd, cwd=robust_lbs_project, debug_port=debug_port)

    if not out_smoothed.exists():
        logger.warning(f"[body_lbs_transfer] expected output not found: {out_smoothed}")
        return

    try:
        if swap_hands_transfer_hands:
            W_split = np.asarray(np.load(str(out_smoothed)), dtype=np.float32)
            if not isinstance(split_meta, dict):
                raise ValueError("Internal error: split_meta missing in swap_hands_transfer_hands mode.")
            nv_body = int(split_meta["nv_body_no_hands"])
            nv_hands = int(split_meta["nv_hands"])
            if W_split.ndim != 2 or int(W_split.shape[0]) != int(nv_body + nv_hands):
                raise ValueError(f"W_split shape mismatch: {W_split.shape} vs expected {(nv_body + nv_hands)}")
            W_body = W_split[:nv_body].astype(np.float32)
            W_hands = W_split[nv_body : nv_body + nv_hands].astype(np.float32)
            np.save(str(out_path), W_body.astype(np.float32))

            hands_w_path = swap_dir_p / "smoothed_inpainted_weights_A_hands.npy"
            hands_backup = swap_dir_p / "smoothed_inpainted_weights_A_hands_old.npy"
            if overwrite_hands_weights:
                if hands_w_path.exists() and (not hands_backup.exists()):
                    try:
                        shutil.copyfile(str(hands_w_path), str(hands_backup))
                    except Exception:
                        pass
                np.save(str(hands_w_path), W_hands.astype(np.float32))
            else:
                np.save(str(swap_dir_p / "smoothed_inpainted_weights_A_hands_transferred.npy"), W_hands.astype(np.float32))
        elif body_only_keep_b_hands:

            W_sub = np.asarray(np.load(str(out_smoothed)), dtype=np.float32)
            if W_sub.ndim != 2:
                raise ValueError(f"W_sub must be 2D, got {W_sub.shape}")
            if used_t is None:
                raise ValueError("Internal error: used_t is None in body_only mode.")
            if W_target_full_orig is None:
                raise ValueError("Internal error: W_target_full_orig is None in body_only mode.")
            W_full = np.asarray(W_target_full_orig, dtype=np.float32).copy()
            if W_full.ndim != 2:
                raise ValueError(f"W_full must be 2D, got {W_full.shape}")
            if W_sub.shape[1] != W_full.shape[1]:
                raise ValueError(f"Joint count mismatch: W_sub {W_sub.shape} vs W_full {W_full.shape}")
            used_t = np.asarray(used_t, dtype=np.int64).reshape(-1)
            if used_t.size != W_sub.shape[0]:
                raise ValueError(f"used_t size mismatch: {used_t.size} vs W_sub Nv {W_sub.shape[0]}")
            if used_t.max(initial=-1) >= W_full.shape[0]:
                raise ValueError(f"used_t out of range for W_full: max {int(used_t.max())} vs Nv {int(W_full.shape[0])}")
            W_full[used_t] = W_sub.astype(np.float32)
            np.save(str(out_path), W_full.astype(np.float32))
        else:
            shutil.copyfile(str(out_smoothed), str(out_path))

        try:
            W_t = np.asarray(np.load(str(out_smoothed)), dtype=np.float32)
            if W_t.ndim == 2 and W_t.shape[0] == V_t_cf.shape[0]:
                tgt_vis = work_dir / "tmp_target_B_body_cf_lbsvis_topk4.ply"
                tgt_colors = _lbs_to_vertex_colors_topk_blend(W_t, k=4)
                trimesh.Trimesh(vertices=V_t_cf, faces=F_t, process=False, vertex_colors=tgt_colors).export(str(tgt_vis))
            else:
                logger.warning(
                    f"[body_lbs_transfer] skip target LBS vis: W shape {getattr(W_t, 'shape', None)} vs V {V_t_cf.shape}"
                )
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed to write target LBS visualization PLY: {e}")

        marker.write_text(
            json.dumps(
                {
                    "head_id": str(head_id_raw),
                    "body_id": str(body_id_raw),
                    "swap_dir": str(swap_dir_p),
                    "mode": str(transfer_mode),
                    "swap_hands_transfer_hands": bool(swap_hands_transfer_hands),
                    "overwrite_hands_weights": bool(overwrite_hands_weights),
                    "smpl_mesh_path": str(smpl_mesh_path),
                    "smpl_weights_path": str(smpl_w_path),
                    "target_mesh_path": str(target_mesh_path),
                    "source_offset": source_offset.tolist(),
                    "target_offset": target_offset.tolist(),
                    "distance_threshold_m": dist_thr,
                    "angle_threshold_deg": ang_thr,
                },
                indent=2,
            )
            + "\n"
        )
        logger.success(f"[body_lbs_transfer] wrote transferred weights: {out_path}")
    except Exception as e:
        logger.warning(f"[body_lbs_transfer] failed to persist transferred weights: {e}")


def stage_direct_swapping(config, subjects, debug_subprocess=False):

    from pipeline.core.config import use_cloth_fit_reshaped_gs

    logger.info("\n--- Stage 12b: Direct Swapping (No Reposing) ---")
    cfg = config['pipeline_stages'].get('12b_direct_swapping', config['pipeline_stages']['12_swapping'])
    project_root = config['paths']['project_root']
    module_name = 'src.scripts.swapping.head_swap'


    merge_cfg = cfg.get("merge_maybe_patch_meshes", {}) if isinstance(cfg, dict) else {}
    merge_enabled = bool(merge_cfg.get("enabled", False)) if isinstance(merge_cfg, dict) else False
    if bool(cfg.get("merge_only", False)):
        if not merge_enabled:
            logger.warning("12b_direct_swapping.merge_only is true but merge_maybe_patch_meshes.enabled is false; nothing to do.")
            return
        avatarrex_dir = config["paths"]["avatarrex_output"]
        a_id = get_padded_subject_id(subjects[0])
        b_id = get_padded_subject_id(subjects[1])

        cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '12b_direct_swapping')


        suffix = ""
        if cloth_fit_enabled:
            suffix = f"_reshaped_cloth_fit{cloth_fit_suffix}"
        else:
            if cfg.get("enable_body_reshape", True):
                reshape_subfix = get_reshape_suffix_from_config(config)[1:]
                if reshape_subfix:
                    suffix = f"_{reshape_subfix}"
        if cfg.get("swap_hands", False):
            suffix += "_hands_swapped"

        swap_dir_ab = os.path.join(avatarrex_dir, "swapped", f"A{a_id}_B{b_id}{suffix}")
        try:
            _maybe_transfer_body_lbs_from_avatar_smpl(
                config=config,
                stage_cfg=cfg,
                project_root=project_root,
                swap_dir=swap_dir_ab,
                head_id_raw=subjects[0],
                body_id_raw=subjects[1],
                debug_subprocess=debug_subprocess,
            )
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed for swap_dir={swap_dir_ab}: {e}")
        merge_maybe_patch_meshes(project_root=project_root, swap_dir=swap_dir_ab, stage_cfg=cfg, debug_subprocess=debug_subprocess)
        if cfg.get("swap_both_directions", False):
            swap_dir_ba = os.path.join(avatarrex_dir, "swapped", f"A{b_id}_B{a_id}{suffix}")
            try:
                _maybe_transfer_body_lbs_from_avatar_smpl(
                    config=config,
                    stage_cfg=cfg,
                    project_root=project_root,
                    swap_dir=swap_dir_ba,
                    head_id_raw=subjects[1],
                    body_id_raw=subjects[0],
                    debug_subprocess=debug_subprocess,
                )
            except Exception as e:
                logger.warning(f"[body_lbs_transfer] failed for swap_dir={swap_dir_ba}: {e}")
            merge_maybe_patch_meshes(project_root=project_root, swap_dir=swap_dir_ba, stage_cfg=cfg, debug_subprocess=debug_subprocess)
        return


    subject_a_position = 'a'
    subject_b_position = 'b'
    subject_a_iteration = get_subject_iteration_and_batch_size(config, subjects[0], subject_a_position)
    subject_b_iteration = get_subject_iteration_and_batch_size(config, subjects[1], subject_b_position)


    cmd = [
        'uv', 'run', 'python', '-m', module_name,
        "--data_root", config['paths']['avatarrex_output'],
        '--user_A_id', get_padded_subject_id(subjects[0]),
        '--model_B_id', get_padded_subject_id(subjects[1]),
        '--user_A_iteration', str(subject_a_iteration),
        '--model_B_iteration', str(subject_b_iteration),
        '--direct_swap_mode',
        "--gender_A_for_pelvis_joint", config.get('smpl_gender_a', 'neutral'),
        "--gender_B_for_pelvis_joint", config.get('smpl_gender_b', 'neutral'),
    ]
    if cfg.get('swap_both_directions', False):
        cmd.append('--swap_both_directions')


    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '12b_direct_swapping')

    if cloth_fit_enabled:
        cmd.append('--use_cloth_fit_reshaped_gs')
        cmd.append('--cloth_fit_suffix')
        cmd.append(cloth_fit_suffix)

        cloth_fit_cfg = config['pipeline_stages'].get('11c_cloth_fit_reshaping', {})
        if cloth_fit_cfg.get('height_aware', False):
            cmd.append('--cloth_fit_height_aware')
        logger.info("Direct swapping will use cloth-fit reshaped Gaussians")
        logger.info(f"Cloth-fit suffix: {cloth_fit_suffix}")


    elif cfg.get('enable_body_reshape', True):

        cmd.append('--repose_load_dir_subfix')
        reshape_subfix = get_reshape_suffix_from_config(config)[1:]
        cmd.append(reshape_subfix)
        cmd.append('--enable_body_reshape')

    if cfg.get('disable_color_transfer', False):
        cmd.append('--no_color_transfer')


    merge_cfg = cfg.get("merge_maybe_patch_meshes", {}) if isinstance(cfg, dict) else {}
    merge_enabled = bool(merge_cfg.get("enabled", False)) if isinstance(merge_cfg, dict) else False
    if cfg.get("save_head_body_separation_debug", False) or merge_enabled:
        cmd.append("--save_head_body_separation_debug")
        if merge_enabled and not cfg.get("save_head_body_separation_debug", False):
            logger.info("merge_maybe_patch_meshes enabled: forcing --save_head_body_separation_debug for head_swap.")


    if cfg.get("swap_hands", False):
        cmd.append("--swap_hands")
        if "swap_hands_source" in cfg:
            cmd.extend(["--swap_hands_source", str(cfg.get("swap_hands_source"))])
        if "smplx_hands_gaussians_per_hand" in cfg:
            cmd.extend(["--smplx_hands_gaussians_per_hand", str(int(cfg.get("smplx_hands_gaussians_per_hand")))])
        if "smplx_hands_gray" in cfg:
            cmd.extend(["--smplx_hands_gray", str(float(cfg.get("smplx_hands_gray")))])
        if "smplx_hands_scale_m" in cfg:
            cmd.extend(["--smplx_hands_scale_m", str(float(cfg.get("smplx_hands_scale_m")))])


    if "neck_plane_lift_ratio" in cfg:
        cmd.extend(["--neck_plane_lift_ratio", str(cfg["neck_plane_lift_ratio"])])
    if "color_transfer_opacity_threshold" in cfg:
        cmd.extend(["--color_transfer_opacity_threshold", str(cfg["color_transfer_opacity_threshold"])])
    if "color_transfer_source_relative_neff_ratio" in cfg:
        cmd.extend(["--color_transfer_source_relative_neff_ratio", str(cfg["color_transfer_source_relative_neff_ratio"])])
    if "color_transfer_use_all_gaussians_for_stats" in cfg:
        cmd.append(
            "--color_transfer_use_all_gaussians_for_stats"
            if cfg["color_transfer_use_all_gaussians_for_stats"]
            else "--no-color_transfer_use_all_gaussians_for_stats"
        )
    if "color_transfer_use_opacity_weighting" in cfg:
        cmd.append(
            "--color_transfer_use_opacity_weighting"
            if cfg["color_transfer_use_opacity_weighting"]
            else "--no-color_transfer_use_opacity_weighting"
        )


    if cfg.get('post_align_hands_to_arms', False):
        cmd.append('--post_align_hands_to_arms')
    if 'post_align_hands_to_arms_min_gaussians' in cfg:
        cmd.extend(['--post_align_hands_to_arms_min_gaussians', str(cfg['post_align_hands_to_arms_min_gaussians'])])
    if 'post_align_hands_to_arms_min_delta' in cfg:
        cmd.extend(['--post_align_hands_to_arms_min_delta', str(cfg['post_align_hands_to_arms_min_delta'])])
    if 'post_align_hands_to_arms_opacity_quantile' in cfg:
        cmd.extend(['--post_align_hands_to_arms_opacity_quantile', str(cfg['post_align_hands_to_arms_opacity_quantile'])])
    if 'post_align_hands_to_arms_min_dominant_gaussians' in cfg:
        cmd.extend(['--post_align_hands_to_arms_min_dominant_gaussians', str(cfg['post_align_hands_to_arms_min_dominant_gaussians'])])
    if 'post_align_hands_to_arms_k2_lab' in cfg:

        if cfg['post_align_hands_to_arms_k2_lab']:
            cmd.extend(['--post_align_hands_to_arms_k2_lab'])
        else:
            cmd.extend(['--no-post_align_hands_to_arms_k2_lab'])
    if 'post_align_hands_to_arms_k2_iters' in cfg:
        cmd.extend(['--post_align_hands_to_arms_k2_iters', str(cfg['post_align_hands_to_arms_k2_iters'])])


    enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)
    if enable_rigid_head_reposing:
        cmd.append('--enable_rigid_head_reposing')


    if cfg.get('save_body_donator_full_in_head_world', False):
        cmd.append('--save_body_donator_full_in_head_world')


    if 'label_dir_name' in cfg:
        cmd.extend(['--label_dir_name', cfg['label_dir_name']])


    debug_port = get_debug_port('swapping') if debug_subprocess else None
    run_command(cmd, cwd=project_root, debug_port=debug_port)


    try:
        avatarrex_dir = config["paths"]["avatarrex_output"]
        a_id = get_padded_subject_id(subjects[0])
        b_id = get_padded_subject_id(subjects[1])


        suffix = ""
        if cloth_fit_enabled:
            suffix = f"_reshaped_cloth_fit{cloth_fit_suffix}"
        else:

            if cfg.get("enable_body_reshape", True):
                reshape_subfix = get_reshape_suffix_from_config(config)[1:]
                if reshape_subfix:
                    suffix = f"_{reshape_subfix}"
        if cfg.get("swap_hands", False):
            suffix += "_hands_swapped"

        swap_dir_ab = os.path.join(avatarrex_dir, "swapped", f"A{a_id}_B{b_id}{suffix}")
        try:
            _maybe_transfer_body_lbs_from_avatar_smpl(
                config=config,
                stage_cfg=cfg,
                project_root=project_root,
                swap_dir=swap_dir_ab,
                head_id_raw=subjects[0],
                body_id_raw=subjects[1],
                debug_subprocess=debug_subprocess,
            )
        except Exception as e:
            logger.warning(f"[body_lbs_transfer] failed for swap_dir={swap_dir_ab}: {e}")
        merge_maybe_patch_meshes(project_root=project_root, swap_dir=swap_dir_ab, stage_cfg=cfg, debug_subprocess=debug_subprocess)

        if cfg.get("swap_both_directions", False):
            swap_dir_ba = os.path.join(avatarrex_dir, "swapped", f"A{b_id}_B{a_id}{suffix}")
            try:
                _maybe_transfer_body_lbs_from_avatar_smpl(
                    config=config,
                    stage_cfg=cfg,
                    project_root=project_root,
                    swap_dir=swap_dir_ba,
                    head_id_raw=subjects[1],
                    body_id_raw=subjects[0],
                    debug_subprocess=debug_subprocess,
                )
            except Exception as e:
                logger.warning(f"[body_lbs_transfer] failed for swap_dir={swap_dir_ba}: {e}")
            merge_maybe_patch_meshes(project_root=project_root, swap_dir=swap_dir_ba, stage_cfg=cfg, debug_subprocess=debug_subprocess)
    except Exception as e:
        logger.warning(f"merge_maybe_patch_meshes post-step failed: {e}")

def stage_head_donator_reposing(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 11a: Head Donator Reposing (Head Alignment) ---")

    disable_renders = config['pipeline_stages']['11a_head_donator_reposing'].get('disable_renders', False)


    subject_directions = [(subjects[0], subjects[1]), (subjects[1], subjects[0])]

    stage_reposing(
        config,
        subjects,
        debug_subprocess=debug_subprocess,
        operation_mode="reposing",
        subject_directions=subject_directions,
        force_enable_reshape=False,
        disable_renders=disable_renders
    )

def stage_body_donator_reshaping(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 11b: Body Donator Reshaping (Body Shape) ---")


    cfg = config['pipeline_stages'].get('11b_body_donator_reshaping', {})
    if not cfg.get('enable_body_reshape', True):
        logger.info("Body donator reshaping disabled in config - skipping stage")
        return

    subject_directions = [(subjects[0], subjects[1]), (subjects[1], subjects[0])]

    stage_reposing(
        config,
        subjects,
        debug_subprocess=debug_subprocess,
        operation_mode="reshaping",
        subject_directions=subject_directions,
        force_enable_reshape=True
    )
