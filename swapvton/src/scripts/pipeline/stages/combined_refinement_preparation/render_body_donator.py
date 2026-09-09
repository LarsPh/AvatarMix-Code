from loguru import logger
from pipeline.core.file_operations import find_swapped_directories
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.sampling.subject_discovery import (
    get_padded_subject_id,
    get_reshape_suffix_from_config,
    first_swap_reshape_enabled
)
import os
import glob
from pathlib import Path
from typing import Optional, List


def discover_original_body_donator_ply(
    config: dict,
    swapped_dir: Path,
    head_id: str,
    body_id: str
) -> Optional[Path]:


    ply_pattern = f"body_donator_full_swapped_{head_id}head_on_{body_id}body_*.ply"

    plys = list(swapped_dir.glob(ply_pattern))


    plys_without_color = [
        ply for ply in plys
        if '_with_color_transfer' not in ply.name
    ]

    if not plys_without_color:
        logger.warning(
            f"No body_donator_full PLY without color transfer found in: {swapped_dir}\n"
            f"  Expected pattern: {ply_pattern} (without _with_color_transfer)\n"
            f"  To generate this file: Run direct swap with save_body_donator_full_in_head_world=true"
        )
        return None


    filtered = filter_body_donator_full_plys(
        [str(p) for p in plys_without_color],
        config
    )

    if not filtered:
        logger.warning(f"No body_donator_full PLY matching configuration in: {swapped_dir}")
        return None


    logger.info(f"Found original body donator PLY (no color transfer): {Path(filtered[0]).name}")
    return Path(filtered[0])


def filter_body_donator_full_plys(
    body_full_ply_files: List[str],
    config: dict
) -> List[str]:

    enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)
    reshape_enabled = first_swap_reshape_enabled(config)
    direct_swap_config = config.get('_direct_swap_mode', False)

    from pipeline.core.config import use_cloth_fit_reshaped_gs
    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '27_render_body_full_gt_aligned')

    filtered_ply_files = []

    for ply_file in body_full_ply_files:
        ply_basename = os.path.basename(ply_file)


        has_rigidhead = "_rigidhead" in ply_basename
        if enable_rigid_head_reposing and not has_rigidhead:
            continue
        elif not enable_rigid_head_reposing and has_rigidhead:
            continue
        elif reshape_enabled:

            if "reshaped_" not in ply_basename:
                continue
        else:

            if "reshaped_" in ply_basename or f"_cloth_fit_reshaped{cloth_fit_suffix}" in ply_basename:
                continue


        if direct_swap_config and "_direct" not in ply_basename:
            continue
        elif not direct_swap_config and "_direct" in ply_basename:
            continue

        filtered_ply_files.append(ply_file)

    return filtered_ply_files


def render_body_gaussian(
    ply_path: Path,
    transform_npz: str,
    dat_dir: str,
    output_base_dir: str,
    output_subdir: str,
    render_config: dict,
    splatting_cfg: dict,
    python_prefix: list,
    splatting_project: str,
    debug_subprocess: bool,
    description: str = "body Gaussian",
    explicit_subject_name: Optional[str] = None
) -> bool:

    render_splits = render_config.get('render_splits', 'train,test,val')
    frame_id = render_config.get('frame_id', 0)


    cmd = python_prefix + [
        'render_static_gaussians.py',
        '--input_gs_ply', str(ply_path),
        '--dat_dir', dat_dir,
        '--configs', splatting_cfg['configs'],
        '--output_dir', output_base_dir,
        '--frame_id', str(frame_id),
        '--render_splits', render_splits,
        '--camera_transform_npz', transform_npz,
        '--output_subdir', output_subdir,
        "--update_all"
    ]


    if explicit_subject_name:
        cmd.extend(['--output_subject_name', explicit_subject_name])

    logger.info(f"Running SplattingAvatar rendering for {description}")
    logger.info(f"  PLY: {os.path.basename(str(ply_path))}")
    logger.info(f"  Transform: {transform_npz}")
    logger.info(f"  Output subdir: {output_subdir}")

    try:
        debug_port = get_debug_port('render_swapped_gaussians') if debug_subprocess else None
        run_command(cmd, cwd=splatting_project, debug_port=debug_port)
        logger.success(f"Successfully rendered {description}")
        return True
    except Exception as e:
        logger.error(f"Error rendering {description}: {e}")
        return False


def stage_render_body_donator_full_gt_aligned(config, subjects, debug_subprocess=False):

    from pipeline.core.config import use_cloth_fit_reshaped_gs

    logger.info("\n--- Stage 27: Render Body Donator Full (GT-Aligned) ---")


    from pipeline.core.config import get_refinement_render_mode
    render_mode = get_refinement_render_mode(config, "27_render_body_full_gt_aligned")
    if render_mode != "head_aligned":
        logger.info(f"refinement.render_mode={render_mode}: skipping Stage 27 (head-aligned body-donator rendering)")
        return

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    conda_env = config['conda_envs']['splatting']


    render_cfg = config['pipeline_stages'].get('27_render_body_full_gt_aligned', {})
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']

    render_splits = render_cfg.get('render_splits', 'train,test,val')
    frame_id = render_cfg.get('frame_id', 0)


    python_prefix = build_python_command(splatting_avatar_project, conda_env)


    from pipeline.core.config import use_label_no_sam_for_alignment
    use_no_sam = use_label_no_sam_for_alignment(config, '27_render_body_full_gt_aligned')
    npz_suffix = '_no_sam' if use_no_sam else ''
    output_subdir = 'full_body_donator_no_sam' if use_no_sam else 'full_body_donator'

    logger.info(f"Using {'labeled_no_sam' if use_no_sam else 'labeled'} head alignment NPZ")
    logger.info(f"Output subdirectory: {output_subdir}")


    enable_rigid_head_reposing = config.get('enable_rigid_head_reposing', False)
    reshape_enabled = first_swap_reshape_enabled(config)
    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, '27_render_body_full_gt_aligned')
    direct_swap_config = config.get('_direct_swap_mode', False)


    output_base = "head_swapped_renders"


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
                reshape_suffix = get_reshape_suffix_from_config(config)
                expected_base = f"A{subject_a}_B{subject_b}"
                expected_with_suffix = f"{expected_base}{reshape_suffix}" if reshape_suffix else expected_base
                logger.info(f"  Expected: {os.path.join(swapped_base_dir, expected_with_suffix)}")
                continue


            for swapped_dir in swapped_dirs:
                if cloth_fit_enabled and f"_reshaped_cloth_fit{cloth_fit_suffix}" not in swapped_dir:
                    logger.warning(f"Skipping swapped directory for body donator full: {swapped_dir} because it does not contain cloth-fit reshaped files")
                    continue
                logger.info(f"Processing swapped directory for body donator full: {os.path.basename(swapped_dir)}")


                ply_pattern = os.path.join(swapped_dir, "body_donator_full_*.ply")
                body_full_ply_files = glob.glob(ply_pattern)

                if not body_full_ply_files:
                    logger.warning(f"No body_donator_full PLY files found in: {swapped_dir}")
                    continue


                filtered_ply_files = filter_body_donator_full_plys(body_full_ply_files, config)

                if not filtered_ply_files:
                    logger.warning(f"No body_donator_full PLY files matching configuration found in: {swapped_dir}")
                    logger.info(f"  Rigid head reposing enabled: {enable_rigid_head_reposing}")
                    logger.info(f"  Cloth-fit reshape enabled: {cloth_fit_enabled}")
                    logger.info(f"  SMPL-based reshape enabled: {reshape_enabled}")
                    logger.info(f"  Direct swap mode: {direct_swap_config}")
                    continue

                logger.info(f"Processing {len(filtered_ply_files)} body_donator_full PLY files for pair {subject_a} -> {subject_b}")


                for ply_file in filtered_ply_files:
                    ply_filename = os.path.basename(ply_file)
                    ply_name = os.path.splitext(ply_filename)[0]

                    logger.info(f"Rendering body donator full: {ply_filename}")


                    import re
                    match = re.match(r'body_donator_full_swapped_(\d+)head_on_(\d+)body_', ply_filename)
                    if not match:
                        logger.warning(f"Could not extract head/body IDs from: {ply_filename}")
                        logger.warning(f"Expected pattern: body_donator_full_swapped_{{head_id}}head_on_{{body_id}}body_...")
                        continue

                    head_id = match.group(1)
                    body_id = match.group(2)
                    logger.info(f"Extracted head_id: {head_id}, body_id: {body_id}")


                    swapped_dir_basename = os.path.basename(swapped_dir)
                    dir_match = re.match(r'A(\d+)_B(\d+)(.*)', swapped_dir_basename)
                    if not dir_match:
                        logger.warning(f"Could not parse swapped directory name: {swapped_dir_basename}")
                        logger.warning(f"Expected format: A{{head_id}}_B{{body_id}}[suffix]")
                        continue


                    suffix = dir_match.group(3)
                    if cloth_fit_enabled and f"_reshaped_cloth_fit{cloth_fit_suffix}" in suffix:
                        suffix = suffix.replace(f"_reshaped_cloth_fit{cloth_fit_suffix}", "")


                    dataset_type = config.get('data_type', 'thuman2')
                    reposed_dir_name = f"{dataset_type}_{head_id}_to_{body_id}{suffix}"

                    logger.info(f"Looking for head transformation in reposed directory: {reposed_dir_name}")


                    reposed_dir_path = os.path.join(avatarrex_dir, "gs_on_mesh_repose", reposed_dir_name, "aligned_head_assets")
                    transform_npz = os.path.join(reposed_dir_path, f"head_transform_avg_frame_{frame_id:04d}{npz_suffix}.npz")

                    if not os.path.exists(transform_npz):
                        logger.warning(f"Head transformation file not found: {transform_npz}")
                        logger.warning(f"Skipping GT-aligned rendering for {ply_filename}")
                        continue

                    logger.info(f"Found head transformation: {transform_npz}")


                    swapped_ply_pattern = os.path.join(swapped_dir, f"swapped_{head_id}head_on_{body_id}body_*.ply")
                    matching_swapped_plys = glob.glob(swapped_ply_pattern)

                    if not matching_swapped_plys:
                        logger.warning(f"Could not find matching swapped PLY for body_donator_full in: {swapped_dir}")
                        logger.warning(f"Expected pattern: swapped_{head_id}head_on_{body_id}body_*.ply")
                        continue


                    swapped_ply_name = os.path.splitext(os.path.basename(matching_swapped_plys[0]))[0]


                    output_base_dir = os.path.join(avatarrex_dir, output_base)
                    output_dir = os.path.join(output_base_dir, swapped_ply_name)

                    if not os.path.exists(output_dir):
                        logger.warning(f"Swapped output directory not found: {output_dir}")
                        logger.warning(f"Please run Stage 13 (render_swapped_gaussians) first")
                        continue

                    logger.info(f"Rendering to existing swapped directory: {swapped_ply_name}")


                    dat_dir = os.path.join(avatarrex_dir, get_padded_subject_id(head_id))


                    if not render_cfg.get('render_original_body_donator', False):
                        render_body_gaussian(
                            ply_path=Path(ply_file),
                            transform_npz=transform_npz,
                            dat_dir=dat_dir,
                            output_base_dir=output_base_dir,
                            output_subdir=output_subdir,
                            render_config=render_cfg,
                            splatting_cfg=splatting_cfg,
                            python_prefix=python_prefix,
                            splatting_project=splatting_avatar_project,
                            debug_subprocess=debug_subprocess,
                            description=f"reshaped body donator full ({ply_filename})"
                        )

                    else:
                        logger.info(f"\n--- Rendering original body donator (no color transfer) for {head_id}→{body_id} ---")


                        original_swapped_dir = Path(swapped_dir).parent / f"A{head_id}_B{body_id}"
                        logger.info(f"Swapped directory with original body donator: {original_swapped_dir}")
                        original_ply = discover_original_body_donator_ply(
                            config,
                            original_swapped_dir,
                            head_id,
                            body_id
                        )

                        if original_ply:

                            original_subdir = output_subdir.replace('full_body_donator', 'full_body_donator_original')

                            render_body_gaussian(
                                ply_path=original_ply,
                                transform_npz=transform_npz,
                                dat_dir=dat_dir,
                                output_base_dir=output_base_dir,
                                output_subdir=original_subdir,
                                render_config=render_cfg,
                                splatting_cfg=splatting_cfg,
                                python_prefix=python_prefix,
                                splatting_project=splatting_avatar_project,
                                debug_subprocess=debug_subprocess,
                                description=f"original body donator without color transfer ({os.path.basename(original_ply)})",
                                explicit_subject_name=swapped_ply_name
                            )
                        else:
                            logger.warning(
                                f"Skipping original body rendering for {head_id}→{body_id}\n"
                                f"  Run direct swap with save_body_donator_full_in_head_world=true to generate"
                            )
