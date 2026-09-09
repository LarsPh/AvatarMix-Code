from loguru import logger
from pipeline.core.config import validate_refinement_config, use_swap_hands
from pipeline.core.file_operations import find_swapped_directories
from pipeline.core.file_operations import copy_calibration_file_to_dataset, copy_non_refined_masks, extract_masks_from_refined_directory, find_swapped_subject_directories
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.sampling.subject_discovery import get_padded_subject_id, get_reshape_suffix_from_config, first_swap_reshape_enabled
from pipeline.core.utils import process_subject_head_masks
import os
import glob
import sys
from pathlib import Path
import json
import math


def stage_render_swapped_gaussians(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 13: Render Swapped Gaussians ---")


    validate_refinement_config(config)

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    conda_env = config['conda_envs']['splatting']


    render_cfg = config['pipeline_stages'].get('13_render_swapped_gaussians', {})
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']

    render_splits = render_cfg.get('render_splits', 'train,test,val')
    frame_id = render_cfg.get('frame_id', 0)

    output_base = str(render_cfg.get('output_base', 'head_swapped_renders'))

    camera_source = str(render_cfg.get('camera_source', 'A')).strip().upper()
    if camera_source not in ('A', 'B'):
        logger.warning(f"13_render_swapped_gaussians.camera_source={camera_source!r} invalid; falling back to 'A'")
        camera_source = 'A'


    canonical_cfg = render_cfg.get("canonical", None)
    if canonical_cfg is None:
        canonical_cfg = render_cfg.get("canonical_front", {}) or {}
    canonical_enabled = bool((canonical_cfg or {}).get("enabled", False))

    from pipeline.core.config import use_cloth_fit_reshaped_gs
    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '13_render_swapped_gaussians')
    swap_hands_enabled = use_swap_hands(config)


    python_prefix = build_python_command(splatting_avatar_project, conda_env)


    swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

    if not os.path.exists(swapped_base_dir):
        logger.warning(f"Swapped directory not found: {swapped_base_dir}")
        return

    def _load_obj_bbox(mesh_path: str):
        xs, ys, zs = [], [], []
        with open(mesh_path, "r") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        xs.append(float(parts[1]))
                        ys.append(float(parts[2]))
                        zs.append(float(parts[3]))
        if not xs:
            raise ValueError(f"No vertices found in OBJ: {mesh_path}")
        return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

    def _compute_canonical_params_for_dataset():

        ref_mesh = canonical_cfg.get("reference_mesh_path", None)
        if not ref_mesh:
            raise ValueError("13_render_swapped_gaussians.canonical.enabled requires canonical.reference_mesh_path (or canonical_front.reference_mesh_path)")
        ref_mesh = str(ref_mesh)
        ref_subject_root = Path(ref_mesh).parents[2]
        calib_path = canonical_cfg.get("reference_calib_path", str(ref_subject_root / "calibration_full.json"))
        if not Path(calib_path).exists():
            raise FileNotFoundError(f"Reference calibration not found: {calib_path}")

        with open(calib_path, "r") as f:
            calib = json.load(f)
        first_cam = calib[next(iter(calib.keys()))]

        out_wh = canonical_cfg.get("output_wh", None)
        if isinstance(out_wh, (list, tuple)) and len(out_wh) == 2:
            W, H = int(out_wh[0]), int(out_wh[1])
        else:
            W, H = int(first_cam["imgSize"][0]), int(first_cam["imgSize"][1])

        vmin, vmax = _load_obj_bbox(ref_mesh)
        x0, y0, z0 = vmin
        x1, y1, z1 = vmax
        target_y = 0.5 * (y0 + y1)
        target_y += float(canonical_cfg.get("target_y_offset", 0.0))

        fov_deg = float(canonical_cfg.get("fov_deg", 10.0))
        margin = float(canonical_cfg.get("distance_margin", 1.1))
        fov_rad = math.radians(fov_deg)
        hfov = fov_rad
        vfov = 2.0 * math.atan(math.tan(0.5 * hfov) * (float(H) / float(W)))


        max_r_xz = max(
            math.sqrt(float(x0) * float(x0) + float(z0) * float(z0)),
            math.sqrt(float(x0) * float(x0) + float(z1) * float(z1)),
            math.sqrt(float(x1) * float(x1) + float(z0) * float(z0)),
            math.sqrt(float(x1) * float(x1) + float(z1) * float(z1)),
        )
        max_dy = max(abs(y0 - target_y), abs(y1 - target_y))
        dist_x = max_r_xz / max(math.tan(0.5 * hfov), 1e-8)
        dist_y = max_dy / max(math.tan(0.5 * vfov), 1e-8)
        dist = max(dist_x, dist_y) * margin
        dist *= float(canonical_cfg.get("distance_scale", 1.0))

        return W, H, target_y, dist


    if canonical_enabled:
        W, H, target_y, dist = _compute_canonical_params_for_dataset()
        output_base_dir = canonical_cfg.get("output_base", "canonical_renders")
        output_base_dir = os.path.join(avatarrex_dir, output_base_dir)
        fov_deg = float(canonical_cfg.get("fov_deg", 10.0))
        flip_yz = (config.get("data_type") == "actorshq")
        z_sign = int(canonical_cfg.get("camera_z_sign", -1))
        y_sign = int(canonical_cfg.get("camera_y_sign", 1))

        pitch_deg = float(canonical_cfg.get("camera_pitch_deg", 0.0))

        views = canonical_cfg.get("views", None)
        if not isinstance(views, list) or not views:
            views = [{"id": "front", "yaw_deg": 0.0}]
        logger.info("[CanonicalFront] enabled")
        logger.info(f"  output_base: {output_base_dir}")
        logger.info(f"  resolution: {W}x{H}")
        logger.info(f"  fov_deg: {fov_deg}")
        logger.info(f"  target_y: {target_y:.4f}")
        logger.info(f"  distance: {dist:.4f}")
        logger.info(f"  camera_z_sign: {z_sign}")
        logger.info(f"  camera_y_sign: {y_sign}")
        logger.info(f"  camera_pitch_deg: {pitch_deg}")
        logger.info(f"  flip_yz (actorshq): {flip_yz}")
        logger.info(f"  views: {[str(v.get('id', '')) for v in views]}")


        iteration = int(canonical_cfg.get("iteration", splatting_cfg.get("iteration_default", 30000)))
        model_path_template = splatting_cfg.get("model_path_template", "{subject_id_padded}")

        for subj in subjects:
            subj_padded = get_padded_subject_id(subj)
            model_path = model_path_template.format(subject_id_padded=subj_padded)
            ply_path = os.path.join(avatarrex_dir, "output-splatting", model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
            if not os.path.exists(ply_path):
                logger.warning(f"[CanonicalFront] Missing original avatar PLY for {subj}: {ply_path}")
                continue

            dat_dir = os.path.join(avatarrex_dir, subj_padded)
            calib_file = os.path.join(dat_dir, "calibration_full.json")
            for v in views:
                cam_id = str(v.get("id", "front"))
                yaw_deg = float(v.get("yaw_deg", 0.0))
                view_pitch_deg = float(v.get("pitch_deg", pitch_deg))
                cmd = python_prefix + [
                    "render_static_gaussians.py",
                    "--input_gs_ply", ply_path,
                    "--dat_dir", dat_dir,
                    "--configs", splatting_cfg["configs"],
                    "--output_dir", output_base_dir,
                    "--frame_id", str(frame_id),
                    "--render_canonical",
                    "--canonical_cam_id", cam_id,
                    "--canonical_fov_deg", str(fov_deg),
                    "--canonical_distance", str(dist),
                    "--canonical_target_y", str(target_y),
                    "--canonical_z_sign", str(z_sign),
                    "--canonical_y_sign", str(y_sign),
                    "--canonical_pitch_deg", str(view_pitch_deg),
                    "--canonical_yaw_deg", str(yaw_deg),
                    "--canonical_width", str(W),
                    "--canonical_height", str(H),
                    "--output_subject_name", subj_padded,
                    "--calib_file", calib_file,
                    "--update_all",
                ]
                if flip_yz:
                    cmd.append("--canonical_flip_yz")

                logger.info(
                    f"[Canonical] Rendering original avatar: {subj_padded} view={cam_id} "
                    f"yaw_deg={yaw_deg} pitch_deg={view_pitch_deg}"
                )
                debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
                run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)


    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            swapped_dirs = find_swapped_directories(config, subject_a, subject_b)

            if not swapped_dirs:
                logger.warning(f"No swapped directories found for pair {subject_a} -> {subject_b}")
                reshape_suffix = get_reshape_suffix_from_config(config)
                expected_base = f"A{subject_a}_B{subject_b}"
                expected_with_suffix = f"{expected_base}{reshape_suffix}" if reshape_suffix else expected_base
                logger.info(f"  Expected: {os.path.join(swapped_base_dir, expected_with_suffix)}")
                continue


            for swapped_dir in swapped_dirs:
                logger.info(f"Processing swapped directory: {os.path.basename(swapped_dir)}")
                if cloth_fit_enabled and f"_reshaped_cloth_fit{cloth_fit_suffix}" not in swapped_dir:
                    continue

                swapped_bn = os.path.basename(swapped_dir)
                has_hands_suffix = ("_hands_swapped" in swapped_bn)
                if swap_hands_enabled and (not has_hands_suffix):
                    continue
                if (not swap_hands_enabled) and has_hands_suffix:
                    continue


                ply_pattern = os.path.join(swapped_dir, "*.ply")
                ply_files = glob.glob(ply_pattern)


                swapped_ply_files = [
                    f for f in ply_files
                    if os.path.basename(f).startswith("swapped_")
                ]


                if camera_source == "B":
                    b_only = [f for f in swapped_ply_files if os.path.basename(f).lower().endswith("_b.ply")]
                    if b_only:
                        swapped_ply_files = b_only
                    else:
                        logger.warning(
                            f"camera_source='B' but no swapped '*_B.ply' found in {swapped_dir}. "
                            "Falling back to legacy swapped PLYs."
                        )
                else:
                    swapped_ply_files = [f for f in swapped_ply_files if not os.path.basename(f).lower().endswith("_b.ply")]

                if not swapped_ply_files:
                    logger.warning(f"No swapped PLY files found in: {swapped_dir}")
                    continue

                logger.info(f"Processing {len(swapped_ply_files)} PLY files for pair {subject_a} -> {subject_b}")


                for ply_file in swapped_ply_files:
                    ply_filename = os.path.basename(ply_file)
                    ply_name = os.path.splitext(ply_filename)[0]

                    logger.info(f"Rendering PLY: {ply_filename}")


                    swapped_dir_basename = os.path.basename(swapped_dir)

                    import re

                    match = re.match(r'^A(.+?)_B(.+?)(.*)$', swapped_dir_basename)
                    if not match:
                        logger.warning(f"Could not parse swapped directory name: {swapped_dir_basename}")
                        logger.warning(f"Expected format: A{{head_id}}_B{{body_id}}[suffix]")
                        continue


                    suffix = match.group(3)


                    suffix = suffix.replace(f"_reshaped_cloth_fit{cloth_fit_suffix}", "")


                    output_base_dir_render = os.path.join(avatarrex_dir, output_base)
                    if canonical_enabled:
                        output_base_dir_render = output_base_dir
                    output_dir = os.path.join(output_base_dir_render, ply_name)
                    os.makedirs(output_dir, exist_ok=True)


                    cam_subject = subject_a if camera_source == 'A' else subject_b
                    dat_dir = os.path.join(avatarrex_dir, get_padded_subject_id(cam_subject))
                    calib_file = os.path.join(dat_dir, "calibration_full.json")


                    if not canonical_enabled:
                        copy_calibration_file_to_dataset(config, cam_subject, output_dir)


                    cmd = python_prefix + [
                        'render_static_gaussians.py',
                        '--input_gs_ply', ply_file,
                        '--dat_dir', dat_dir,
                        '--configs', splatting_cfg['configs'],
                        '--output_dir', output_base_dir_render,
                        '--frame_id', str(frame_id),
                        '--render_splits', render_splits,
                        '--update_all'
                    ]
                    if canonical_enabled:

                        for v in views:
                            cam_id = str(v.get("id", "front"))
                            yaw_deg = float(v.get("yaw_deg", 0.0))
                            view_pitch_deg = float(v.get("pitch_deg", pitch_deg))
                            cmd_view = list(cmd)
                            cmd_view.extend([
                                "--render_canonical",
                                "--canonical_cam_id", cam_id,
                                "--canonical_fov_deg", str(fov_deg),
                                "--canonical_distance", str(dist),
                                "--canonical_target_y", str(target_y),
                                "--canonical_z_sign", str(z_sign),
                                "--canonical_y_sign", str(y_sign),
                                "--canonical_pitch_deg", str(view_pitch_deg),
                                "--canonical_yaw_deg", str(yaw_deg),
                                "--canonical_width", str(W),
                                "--canonical_height", str(H),
                                "--output_subject_name", ply_name,
                                "--calib_file", calib_file,
                            ])
                            if flip_yz:
                                cmd_view.append("--canonical_flip_yz")
                            logger.info(
                                f"[Canonical] Rendering swapped: {ply_name} view={cam_id} "
                                f"yaw_deg={yaw_deg} pitch_deg={view_pitch_deg}"
                            )
                            debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
                            run_command(cmd_view, cwd=splatting_avatar_project, debug_port=debug_port)
                        continue

                    logger.info(f"Running SplattingAvatar rendering for: {ply_filename}")
                    logger.info(f"Output directory: {output_dir}")
                    logger.info(f"Camera source: {camera_source} (dat_dir subject={get_padded_subject_id(cam_subject)})")

                    try:
                        debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
                        run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
                        logger.info(f"Successfully rendered: {ply_filename}")
                    except Exception as e:
                        logger.error(f"Error rendering {ply_filename}: {e}")
                        continue

def stage_render_swapped_gt_head_aligned(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 26: Render Swapped GT-Aligned Head ---")


    from pipeline.core.config import get_refinement_render_mode
    render_mode = get_refinement_render_mode(config, "26_render_swapped_gt_head_aligned")
    if render_mode != "head_aligned":
        logger.info(f"refinement.render_mode={render_mode}: skipping Stage 26 (head-aligned rendering)")
        return

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    conda_env = config['conda_envs']['splatting']


    render_cfg = config['pipeline_stages'].get('26_render_swapped_gt_head_aligned', {})
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']

    render_splits = render_cfg.get('render_splits', 'train,test,val')
    frame_id = render_cfg.get('frame_id', 0)
    output_base = render_cfg.get('output_base', 'head_swapped_renders')


    python_prefix = build_python_command(splatting_avatar_project, conda_env)


    from pipeline.core.config import use_label_no_sam_for_alignment
    use_no_sam = use_label_no_sam_for_alignment(config, '26_render_swapped_gt_head_aligned')
    npz_suffix = '_no_sam' if use_no_sam else ''
    output_subdir = 'head_aligned_no_sam' if use_no_sam else 'head_aligned'

    logger.info(f"Using {'labeled_no_sam' if use_no_sam else 'labeled'} head alignment NPZ")
    logger.info(f"Output subdirectory: {output_subdir}")

    from pipeline.core.config import use_cloth_fit_reshaped_gs
    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '26_render_swapped_gt_head_aligned')
    swap_hands_enabled = use_swap_hands(config)

    swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

    if not os.path.exists(swapped_base_dir):
        logger.warning(f"Swapped directory not found: {swapped_base_dir}")
        return


    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            swapped_dirs = find_swapped_directories(config, subject_a, subject_b)

            if not swapped_dirs:
                logger.warning(f"No swapped directories found for pair {subject_a} -> {subject_b}")
                if cloth_fit_enabled:
                    reshape_suffix = f"_reshaped_cloth_fit{cloth_fit_suffix}"
                else:
                    reshape_suffix = get_reshape_suffix_from_config(config)
                expected_base = f"A{subject_a}_B{subject_b}"
                expected_with_suffix = f"{expected_base}{reshape_suffix}" if reshape_suffix else expected_base
                logger.info(f"  Expected: {os.path.join(swapped_base_dir, expected_with_suffix)}")
                continue


            for swapped_dir in swapped_dirs:
                logger.info(f"Processing swapped directory for GT-aligned heads: {os.path.basename(swapped_dir)}")
                if cloth_fit_enabled and f"_reshaped_cloth_fit{cloth_fit_suffix}" not in swapped_dir:
                    continue

                swapped_bn = os.path.basename(swapped_dir)
                has_hands_suffix = ("_hands_swapped" in swapped_bn)
                if swap_hands_enabled and (not has_hands_suffix):
                    continue
                if (not swap_hands_enabled) and has_hands_suffix:
                    continue


                ply_pattern = os.path.join(swapped_dir, "*.ply")
                all_swapped_ply_files = glob.glob(ply_pattern)


                all_swapped_ply_files = [f for f in all_swapped_ply_files if not (os.path.basename(f).startswith('gt_aligned_head_') or os.path.basename(f).startswith('body_donator_full_'))]

                if not all_swapped_ply_files:
                    logger.warning(f"No swapped PLY files found in: {swapped_dir}")
                    continue


                reshape_enabled = first_swap_reshape_enabled(config)
                direct_swap_config = config.get('_direct_swap_mode', False)


                swapped_ply_files = []


                enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)

                for ply_file in all_swapped_ply_files:
                    ply_basename = os.path.basename(ply_file)


                    has_rigidhead = "_rigidhead" in ply_basename
                    if enable_rigid_head_reposing and not has_rigidhead:
                        continue
                    elif not enable_rigid_head_reposing and has_rigidhead:
                        continue


                    if reshape_enabled and "reshaped_" not in ply_basename:
                        continue
                    elif not reshape_enabled and "reshaped_" in ply_basename:
                        continue


                    if "with_color_transfer" not in ply_basename:
                        continue

                    swapped_ply_files.append(ply_file)

                if not swapped_ply_files:
                    logger.warning(f"No swapped PLY files matching configuration found in: {swapped_dir}")
                    logger.info(f"  Rigid head reposing enabled: {enable_rigid_head_reposing}")
                    logger.info(f"  Reshape enabled: {reshape_enabled}")
                    logger.info(f"  Direct swap mode: {direct_swap_config}")
                    continue

                logger.info(f"Processing {len(swapped_ply_files)} swapped PLY files for GT-aligned head rendering for pair {subject_a} -> {subject_b}")


                for ply_file in swapped_ply_files:
                    ply_filename = os.path.basename(ply_file)
                    ply_name = os.path.splitext(ply_filename)[0]

                    logger.info(f"Rendering GT-aligned head PLY: {ply_filename}")
                    logger.info(f"  Cloth-fit enabled: {cloth_fit_enabled and f'_reshaped_cloth_fit{cloth_fit_suffix}' in ply_file}")


                    swapped_dir_basename = os.path.basename(swapped_dir)

                    import re

                    match = re.match(r'^A(.+?)_B(.+?)(.*)$', swapped_dir_basename)
                    if not match:
                        logger.warning(f"Could not parse swapped directory name: {swapped_dir_basename}")
                        logger.warning(f"Expected format: A{{head_id}}_B{{body_id}}[suffix]")
                        continue

                    head_id = match.group(1)
                    body_id = match.group(2)
                    suffix = match.group(3)


                    suffix = suffix.replace(f"_reshaped_cloth_fit{cloth_fit_suffix}", "")


                    dataset_type = config.get('data_type', 'thuman2')
                    reposed_dir_name = f"{dataset_type}_{head_id}_to_{body_id}{suffix}"

                    logger.info(f"Looking for head transformation in reposed directory: {reposed_dir_name}")


                    reposed_dir_path = os.path.join(avatarrex_dir, "gs_on_mesh_repose", reposed_dir_name, "aligned_head_assets")
                    transform_npz_pattern = os.path.join(reposed_dir_path, f"head_transform_avg_frame_{frame_id:04d}{npz_suffix}.npz")

                    if not os.path.exists(transform_npz_pattern):
                        logger.warning(f"Head transformation file not found: {transform_npz_pattern}")
                        logger.warning(f"Skipping GT-aligned rendering for {ply_filename}")
                        continue

                    logger.info(f"Found head transformation: {transform_npz_pattern}")


                    output_base_dir = os.path.join(avatarrex_dir, output_base)
                    output_dir = os.path.join(output_base_dir, ply_name)
                    os.makedirs(output_dir, exist_ok=True)


                    dat_dir = os.path.join(avatarrex_dir, get_padded_subject_id(head_id))


                    cmd = python_prefix + [
                        'render_static_gaussians.py',
                        '--input_gs_ply', ply_file,
                        '--dat_dir', dat_dir,
                        '--configs', splatting_cfg['configs'],
                        '--output_dir', output_base_dir,
                        '--frame_id', str(frame_id),
                        '--render_splits', render_splits,
                        '--camera_transform_npz', transform_npz_pattern,
                        '--output_subdir', output_subdir,
                    ]

                    logger.info(f"Running SplattingAvatar rendering with camera transformation for PLY: {ply_filename}")
                    logger.info(f"Head transformation file: {transform_npz_pattern}")
                    logger.info(f"Output directory: {output_dir}")

                    try:
                        debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
                        run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
                        logger.info(f"Successfully rendered GT-aligned head: {ply_filename}")
                    except Exception as e:
                        logger.error(f"Error rendering GT-aligned head {ply_filename}: {e}")
                        continue

def stage_refine_rendered_images(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 14: Refine Rendered Images ---")

    avatarrex_dir = config['paths']['avatarrex_output']
    difix3d_project = config['paths']['difix3d_project']


    refine_cfg = config['pipeline_stages'].get('14_refine_rendered_images', {})
    batch_size = refine_cfg.get('batch_size', 2)
    create_separate_dataset = refine_cfg.get('create_separate_dataset', True)
    extract_masks = refine_cfg.get('extract_masks', False)


    render_cfg = config['pipeline_stages'].get('13_render_swapped_gaussians', {}) or {}
    output_base = str(render_cfg.get('output_base', 'head_swapped_renders'))
    rendered_base_dir = os.path.join(avatarrex_dir, output_base)

    if not os.path.exists(rendered_base_dir):
        logger.warning(f"Rendered images directory not found: {rendered_base_dir}")
        return


    expected_directories = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_render_dir = os.path.join(rendered_base_dir, ply_name)
                        if os.path.exists(expected_render_dir):
                            expected_directories.append(expected_render_dir)
            except Exception as e:
                logger.warning(f"Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_directories:
        logger.warning("No rendered directories found for specified subject pairs")
        return

    logger.info(f"Found {len(expected_directories)} rendered directories for specified subject pairs")


    for render_path in expected_directories:
        render_dir = os.path.basename(render_path)
        logger.info(f"Processing rendered directory: {render_dir}")


        if not os.path.exists(render_path):
            logger.warning(f"Warning: Rendered directory not found: {render_path}")
            continue


        refined_base_dir = os.path.join(avatarrex_dir, "head_swapped_renders_refined")
        render_dir_name = os.path.basename(render_path)
        refined_dir = os.path.join(refined_base_dir, render_dir_name)


        if create_separate_dataset and "head_on" in render_dir_name:
            try:
                parts = render_dir_name.split("head_on")
                subject_a_part = parts[0].replace("swapped_", "")
                subject_a = subject_a_part.replace("_", "")[:4]
                copy_calibration_file_to_dataset(config, subject_a, refined_dir)
                logger.info(f"Copied calibration file for subject {subject_a} to: {refined_dir}")
            except Exception as e:
                logger.warning(f"Warning: Could not extract subject ID from {render_dir_name}: {e}")


        try:

            cmd = [
                'uv', 'run', 'python', 'scripts/refine_batch_images_lightning.py',
                '--input_dir', render_path,
                '--output_suffix', '_refined',
                '--batch_size', str(batch_size),

            ]

            logger.info(f"Running batch refinement for directory: {render_dir}")

            debug_port = get_debug_port('refine_rendered_images') if debug_subprocess else None
            run_command(cmd, cwd=difix3d_project, debug_port=debug_port)

            logger.info(f"Successfully completed batch refinement for: {render_dir}")


            if extract_masks:
                logger.info("Extracting masks from refined images")
                extract_masks_from_refined_directory(refined_dir)
                logger.info(f"Masks extracted to: {refined_dir}")
            else:
                logger.info("Copying non-refined masks from original rendered directory")
                try:
                    copy_non_refined_masks(refined_dir)
                except Exception as e:
                    logger.error(f"Error copying non-refined masks: {e}")
                    continue
                logger.info(f"Non-refined masks copied to: {refined_dir}")

        except Exception as e:
            raise Exception(f"Batch refinement failed for {render_dir}: {e}")


def stage_extract_head_masks(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 14a: Extract Head Masks ---")


    sys.path.append(str(Path(__file__).parent))


    extraction_cfg = config['pipeline_stages'].get('14a_extract_head_masks', {})
    always_process_gt = extraction_cfg.get('always_process_gt', True)
    process_swapped_data = extraction_cfg.get('process_swapped_data', 'refined')
    color_tolerance = extraction_cfg.get('color_tolerance', 5)
    background_color = tuple(extraction_cfg.get('background_color', [0, 0, 0]))
    skip_existing = extraction_cfg.get('skip_existing', False)


    dataset_type = config['data_type']
    avatarrex_dir = config['paths']['avatarrex_output']

    logger.info(f"Head mask extraction configuration:")
    logger.info(f"  Dataset type: {dataset_type}")
    logger.info(f"  Always process GT: {always_process_gt}")
    logger.info(f"  Process swapped data: {process_swapped_data}")
    logger.info(f"  Color tolerance: {color_tolerance}")
    logger.info(f"  Background color: {background_color}")

    processed_count = 0
    failed_count = 0


    if always_process_gt:
        logger.info("Processing GT data for subject pair...")

        for subject_id in subjects:
            padded_subject_id = get_padded_subject_id(subject_id)
            gt_data_root = os.path.join(avatarrex_dir, padded_subject_id)

            if not os.path.exists(gt_data_root):
                logger.warning(f"GT data not found for subject {subject_id}: {gt_data_root}")
                failed_count += 1
                continue

            try:
                logger.info(f"Extracting head masks for GT subject {subject_id}...")
                process_subject_head_masks(
                    data_root=gt_data_root,
                    data_type="gt",
                    color_tolerance=color_tolerance,
                    background_color=background_color,
                    skip_existing=skip_existing,
                    dataset_type=dataset_type
                )
                processed_count += 1
                logger.info(f"Successfully processed GT subject {subject_id}")

            except Exception as e:
                logger.error(f"Failed to process GT subject {subject_id}: {e}")
                failed_count += 1


    if process_swapped_data != 'none':
        logger.info(f"Processing swapped data with mode: {process_swapped_data}")

        swapped_directories = []


        if process_swapped_data in ['raw', 'both']:
            raw_swapped_dir = os.path.join(avatarrex_dir, "head_swapped_renders")
            if os.path.exists(raw_swapped_dir):
                swapped_directories.extend(find_swapped_subject_directories(raw_swapped_dir, subjects))
                logger.info(f"Found {len(find_swapped_subject_directories(raw_swapped_dir, subjects))} raw swapped directories")
            else:
                logger.warning(f"Raw swapped directory not found: {raw_swapped_dir}")

        if process_swapped_data in ['refined', 'both']:
            refined_swapped_dir = os.path.join(avatarrex_dir, "head_swapped_renders_refined")
            if os.path.exists(refined_swapped_dir):
                refined_dirs = find_swapped_subject_directories(refined_swapped_dir, subjects)

                reshaped_dirs = [dir for dir in refined_dirs if "reshaped_" in dir and "_with_color_transfer" in dir]
                non_reshaped_dirs = [dir for dir in refined_dirs if "reshaped_" not in dir and "_with_color_transfer" in dir]
                reshape_enabled = first_swap_reshape_enabled(config)
                if reshape_enabled:
                    refined_dirs = reshaped_dirs
                else:
                    refined_dirs = non_reshaped_dirs
                swapped_directories.extend(refined_dirs)
                logger.info(f"Found {len(refined_dirs)} refined swapped directories")
            else:
                logger.warning(f"Refined swapped directory not found: {refined_swapped_dir}")


        for swapped_dir in swapped_directories:
            if not os.path.exists(swapped_dir):
                logger.warning(f"Swapped directory not found: {swapped_dir}")
                failed_count += 1
                continue

            try:
                dir_name = os.path.basename(swapped_dir)
                logger.info(f"Extracting head masks for swapped data: {dir_name}")

                process_subject_head_masks(
                    data_root=swapped_dir,
                    data_type="swapped",
                    color_tolerance=color_tolerance,
                    background_color=background_color,
                    skip_existing=skip_existing,
                    dataset_type=dataset_type
                )
                processed_count += 1
                logger.info(f"Successfully processed swapped directory: {dir_name}")

            except Exception as e:
                logger.error(f"Failed to process swapped directory {swapped_dir}: {e}")
                failed_count += 1


    logger.info(f"Head mask extraction complete: {processed_count} successful, {failed_count} failed")

    if failed_count > 0:
        logger.warning(f"Some head mask extractions failed ({failed_count}). Check logs for details.")
