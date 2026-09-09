from loguru import logger
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.sampling.subject_discovery import get_padded_subject_id, first_swap_reshape_enabled, get_reshape_suffix_from_config

import os


def stage_swapped_head_donator_reposing(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 22b: Swapped Head Donator Reposing (Swap-Back Alignment) ---")

    cfg = config['pipeline_stages']['11_reposing']
    cfg.update(config['pipeline_stages'].get('22b_swapped_head_donator_reposing', {}))
    disable_renders = cfg.get('disable_renders', False)
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['splatting']
    smpl_model_path = os.path.join(config['paths']['smpl_model_root'], 'smplx')
    script = 'repose_avatar.py'


    swapped_splatting_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', 'output-splatting', 'swapped')
    if not os.path.exists(swapped_splatting_dir):
        logger.error(f"Error: Swapped splatting directory not found: {swapped_splatting_dir}")
        logger.error("Please run Stage 22 (splatting_avatar_swapped) first.")
        return

    logger.info("Reposing swapped avatars to original poses for swap-back alignment:")
    logger.info(f"  - Swapped A→B avatar will be reposed to A's original pose")
    logger.info(f"  - Swapped B→A avatar will be reposed to B's original pose")


    expected_swapped_subjects = []
    subject_a = get_padded_subject_id(subjects[0])
    subject_b = get_padded_subject_id(subjects[1])


    enable_reshape = first_swap_reshape_enabled(config)
    from_substring = "_from_reshaped_gs_target_shape_0000" if enable_reshape else "_from_point_cloud"
    expected_swapped_subjects = [
        f"swapped_{subjects[0]}head_on_{subjects[1]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct",
        f"swapped_{subjects[1]}head_on_{subjects[0]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"
    ]


    swapped_to_target_mapping = [
        (expected_swapped_subjects[0], subjects[0]),
        (expected_swapped_subjects[1], subjects[1])
    ]

    for swapped_subject_name, target_subject_id in swapped_to_target_mapping:
        logger.info(f"\n-- Reposing swapped subject {swapped_subject_name} to {target_subject_id}'s original pose --")


        swapped_subject_dir = os.path.join(swapped_splatting_dir, swapped_subject_name)
        if not os.path.exists(swapped_subject_dir):
            logger.warning(f"Warning: Swapped subject not found: {swapped_subject_name}")
            continue


        point_cloud_base = os.path.join(swapped_subject_dir, 'point_cloud')
        if not os.path.exists(point_cloud_base):
            logger.warning(f"Warning: Point cloud directory not found for {swapped_subject_name}")
            continue


        iteration_dirs = [d for d in os.listdir(point_cloud_base) if d.startswith('iteration_')]
        if not iteration_dirs:
            logger.warning(f"Warning: No iteration directories found for {swapped_subject_name}")
            continue


        iterations = [int(d.split('_')[1]) for d in iteration_dirs]
        highest_iteration = max(iterations)


        swapped_point_cloud_dir = os.path.join(point_cloud_base, f'iteration_{highest_iteration}')
        input_gs_ply = os.path.join(swapped_point_cloud_dir, 'point_cloud.ply')
        input_gs_embed = os.path.join(swapped_point_cloud_dir, 'embedding.json')


        swapped_dataset_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject_name)


        if target_subject_id == subjects[0]:

            target_swapped_name = f"swapped_{subjects[1]}head_on_{subjects[0]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"
        else:

            target_swapped_name = f"swapped_{subjects[0]}head_on_{subjects[1]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"

        target_pose_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', target_swapped_name)


        source_swapped_mesh_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject_name)
        nerf_mesh_path = os.path.join(source_swapped_mesh_dir, 'mesh', 'trimesh_cleaned', '0000.obj')
        lbs_weights_path = os.path.join(source_swapped_mesh_dir, 'mesh', 'processed', 'smoothed_inpainted_weights.npy')


        output_dir = os.path.join(avatarrex_dir, 'swapped_head_donator_reposed', swapped_subject_name)


        target_gender = config.get(f'smpl_gender_a' if target_subject_id == subjects[0] else 'smpl_gender_b', 'neutral')

        source_gender = config.get(f'smpl_gender_b' if target_subject_id == subjects[0] else 'smpl_gender_a', 'neutral')


        python_prefix = build_python_command(splatting_avatar_project, env)

        cmd = python_prefix + [
            script,
            '--configs', cfg['configs'],
            '--dat_dir', swapped_dataset_dir,
            '--target_pose_dir', target_pose_dir,
            '--output_dir', output_dir,
            '--input_gs_ply', input_gs_ply,
            '--input_gs_embed', input_gs_embed,
            '--nerf_mesh_path', nerf_mesh_path,
            '--lbs_weights_path', lbs_weights_path,
            '--nerf_mesh_source_frame_idx', '0',
            '--target_frame_start', '0', '--target_frame_end', '1',
            '--animation_mode', 'nerf_mesh_repose',
            '--smpl_model_path', smpl_model_path,
            '--smpl_gender_src', source_gender,
            '--smpl_gender_tgt', target_gender,
            '--gui_ip', 'none',
            '--save_posed_meshes',
            '--num_render_views', str(cfg['num_render_views']),
            '--save_reposed_gs_ply'
        ]
        if disable_renders:
            cmd.append('--disable_renders')

        logger.info(f"Running swapped avatar reposing for: {swapped_subject_name}")
        logger.info(f"Target pose: {target_swapped_name} (combined SMPL params)")
        logger.info(f"Output: {output_dir}")

        try:
            debug_port = get_debug_port('reposing') if debug_subprocess else None
            run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
            logger.info(f"Successfully reposed swapped avatar: {swapped_subject_name}")
        except Exception as e:
            logger.error(f"Error reposing {swapped_subject_name}: {e}")
            continue

    logger.info("Swapped head donator reposing completed - avatars ready for swap-back.")

def stage_swapped_body_donator_reshaping(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 22c: Swapped Body Donator Reshaping (Swap-Back Body Shape) ---")


    reshape_cfg = config['pipeline_stages'].get('22c_swapped_body_donator_reshaping', {})
    if not reshape_cfg.get('enable_body_reshape', True):
        logger.info("Swapped body donator reshaping disabled in config - skipping stage")
        return

    cfg = config['pipeline_stages']['11_reposing']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['splatting']
    smpl_model_path = os.path.join(config['paths']['smpl_model_root'], 'smplx')
    script = 'repose_avatar.py'


    swapped_splatting_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', 'output-splatting', 'swapped')
    if not os.path.exists(swapped_splatting_dir):
        logger.error(f"Error: Swapped splatting directory not found: {swapped_splatting_dir}")
        logger.error("Please run Stage 22 (splatting_avatar_swapped) first.")
        return

    logger.info("Reshaping swapped avatars for consistent body shapes in swap-back:")
    logger.info(f"  - Swapped A→B avatar body will be reshaped to A's body shape")
    logger.info(f"  - Swapped B→A avatar body will be reshaped to B's body shape")


    subject_a = get_padded_subject_id(subjects[0])
    subject_b = get_padded_subject_id(subjects[1])


    enable_reshape = first_swap_reshape_enabled(config)
    from_substring = "_from_reshaped_gs_target_shape_0000" if enable_reshape else "_from_point_cloud"
    expected_swapped_subjects = [
        f"swapped_{subjects[0]}head_on_{subjects[1]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct",
        f"swapped_{subjects[1]}head_on_{subjects[0]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"
    ]


    swapped_to_body_mapping = [
        (expected_swapped_subjects[0], subjects[0]),
        (expected_swapped_subjects[1], subjects[1])
    ]

    for swapped_subject_name, body_shape_subject_id in swapped_to_body_mapping:
        logger.info(f"\n-- Reshaping swapped subject {swapped_subject_name} to {body_shape_subject_id}'s body shape --")


        swapped_subject_dir = os.path.join(swapped_splatting_dir, swapped_subject_name)
        if not os.path.exists(swapped_subject_dir):
            logger.warning(f"Warning: Swapped subject not found: {swapped_subject_name}")
            continue


        point_cloud_base = os.path.join(swapped_subject_dir, 'point_cloud')
        if not os.path.exists(point_cloud_base):
            logger.warning(f"Warning: Point cloud directory not found for {swapped_subject_name}")
            continue


        iteration_dirs = [d for d in os.listdir(point_cloud_base) if d.startswith('iteration_')]
        if not iteration_dirs:
            logger.warning(f"Warning: No iteration directories found for {swapped_subject_name}")
            continue


        iterations = [int(d.split('_')[1]) for d in iteration_dirs]
        highest_iteration = max(iterations)


        swapped_point_cloud_dir = os.path.join(point_cloud_base, f'iteration_{highest_iteration}')
        input_gs_ply = os.path.join(swapped_point_cloud_dir, 'point_cloud.ply')
        input_gs_embed = os.path.join(swapped_point_cloud_dir, 'embedding.json')


        swapped_dataset_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject_name)


        if body_shape_subject_id == subjects[0]:

            target_swapped_name = f"swapped_{subjects[1]}head_on_{subjects[0]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"
        else:

            target_swapped_name = f"swapped_{subjects[0]}head_on_{subjects[1]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"

        target_pose_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', target_swapped_name)


        source_swapped_mesh_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject_name)
        nerf_mesh_path = os.path.join(source_swapped_mesh_dir, 'mesh', 'trimesh_cleaned', '0000.obj')


        output_dir = os.path.join(avatarrex_dir, 'swapped_body_donator_reshaped', swapped_subject_name)


        python_prefix = build_python_command(splatting_avatar_project, env)


        cmd = python_prefix + [
            script,
            '--configs', cfg['configs'],
            '--dat_dir', swapped_dataset_dir,
            '--target_pose_dir', target_pose_dir,
            '--output_dir', output_dir,
            '--input_gs_ply', input_gs_ply,
            '--input_gs_embed', input_gs_embed,
            '--nerf_mesh_path', nerf_mesh_path,
            '--nerf_mesh_source_frame_idx', '0',
            '--target_frame_start', '0', '--target_frame_end', '1',
            '--animation_mode', 'nerf_reshape_no_repose',
            '--enable_body_reshape',
            '--smpl_model_path', smpl_model_path,
            '--smpl_gender_src', 'neutral',
            '--smpl_gender_tgt', config.get(f'smpl_gender_a' if body_shape_subject_id == subjects[0] else 'smpl_gender_b', 'neutral'),
            '--gui_ip', 'none',
            '--save_posed_meshes',
            '--num_render_views', str(cfg['num_render_views']),
            '--save_reposed_gs_ply'
        ]


        cmd.extend(['--body_reshape_smoothing_samples', str(cfg.get('body_reshape_smoothing_samples', 1))])
        cmd.extend(['--body_reshape_sample_std_scale', str(cfg.get('body_reshape_sample_std_scale', 1.0))])
        cmd.extend(['--body_reshape_scale_factor', str(cfg.get('body_reshape_scale_factor', 1.0))])
        if cfg.get('body_reshape_distance_weighting', False):
            cmd.append('--body_reshape_distance_weighting')
        if cfg.get('save_reshaping_meshes', False):
            cmd.append('--save_reshaping_meshes')

        logger.info(f"Running swapped avatar reshaping for: {swapped_subject_name}")
        logger.info(f"Target body shape: {target_swapped_name} (combined SMPL params)")
        logger.info(f"Output: {output_dir}")

        try:
            debug_port = get_debug_port('reposing') if debug_subprocess else None
            run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
            logger.info(f"Successfully reshaped swapped avatar: {swapped_subject_name}")
        except Exception as e:
            logger.error(f"Error reshaping {swapped_subject_name}: {e}")
            continue

    logger.info("Swapped body donator reshaping completed - avatars ready for swap-back.")


def stage_swap_back(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 23: Swap Back (Direct Mode) ---")


    swap_back_cfg = config['pipeline_stages'].get('23_swap_back', {})
    enable_reshape = swap_back_cfg.get('enable_body_reshape', False)

    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    module_name = 'src.scripts.swapping.head_swap'


    reposed_swapped_dir = os.path.join(avatarrex_dir, 'swapped_head_donator_reposed')

    if not os.path.exists(reposed_swapped_dir):
        logger.error(f"Error: Reposed swapped avatars directory not found: {reposed_swapped_dir}")
        logger.error("Please run Stage 22b (swapped_head_donator_reposing) first.")
        logger.error("This stage automatically runs when you specify 'swap_back' in --run-stages.")
        return


    subject_a = get_padded_subject_id(subjects[0])
    subject_b = get_padded_subject_id(subjects[1])


    from_substring = "_from_reshaped_gs_target_shape_0000" if enable_reshape else "_from_point_cloud"
    expected_reposed_names = [
        f"swapped_{subjects[0]}head_on_{subjects[1]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct",
        f"swapped_{subjects[1]}head_on_{subjects[0]}body{from_substring}_in_A_world_with_color_transfer_filter_0.01_direct"
    ]


    missing_subjects = []
    for expected_name in expected_reposed_names:
        subject_dir = os.path.join(reposed_swapped_dir, expected_name)
        if not os.path.exists(subject_dir):
            missing_subjects.append(expected_name)

    if missing_subjects:
        logger.error(f"Error: Missing reposed swapped subjects required for swap-back:")
        for missing in missing_subjects:
            logger.error(f"  - {missing}")
        logger.error("Please ensure Stage 22b (swapped_head_donator_reposing) completed for both swap directions.")
        return

    logger.info(f"Found reposed swapped subjects for swap-back:")
    for name in expected_reposed_names:
        logger.info(f"  ✓ {name}")


    for expected_name in expected_reposed_names:
        subject_dir = os.path.join(reposed_swapped_dir, expected_name)
        reposed_ply_dir = os.path.join(subject_dir, 'reposed_gaussians_ply')
        expected_ply = os.path.join(reposed_ply_dir, 'reposed_gs_targetframe0000.ply')

        if not os.path.exists(expected_ply):
            logger.error(f"Error: Reposed PLY file not found: {expected_ply}")
            logger.error(f"Stage 22b may not have completed successfully for {expected_name}")
            return

    logger.info("All prerequisite reposed swapped PLY files found. Proceeding with swap-back...")


    cmd = [
        'uv', 'run', 'python', '-m', module_name,
        '--user_A_id', subject_a,
        '--model_B_id', subject_b,
        '--swap_back_mode',
        '--swap_both_directions',
        '--direct_swap_mode',
        "--gender_A_for_pelvis_joint", config.get('smpl_gender_a', 'neutral'),
        "--gender_B_for_pelvis_joint", config.get('smpl_gender_b', 'neutral'),
        '--data_root', avatarrex_dir,
        "--user_A_iteration", str(config.get('user_A_iteration', 10000)),
        "--model_B_iteration", str(config.get('model_B_iteration', 10000)),
    ]


    if enable_reshape:
        cmd.append('--repose_load_dir_subfix')
        reshape_subfix = get_reshape_suffix_from_config(config)[1:]
        cmd.append(reshape_subfix)
        cmd.append('--enable_body_reshape')

    logger.info(f"Running swap-back command: {' '.join(cmd)}")
    debug_port = get_debug_port('swapping') if debug_subprocess else None
    run_command(cmd, cwd=project_root, debug_port=debug_port)

    logger.info("Swap-back completed. Restored identities saved in swapped_back/ directory.")
