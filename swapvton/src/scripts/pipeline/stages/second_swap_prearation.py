from loguru import logger
from pipeline.core.constants import NEUS2_HUMAN_SCALE, NEUS2_HUMAN_OFFSET
from pipeline.execution.debug_support import get_debug_port
from pipeline.execution.subprocess_runner import run_command, run_neus2_training_and_export
from pipeline.execution.env_utils import build_python_command
from pipeline.core.file_operations import find_swapped_directories
from pipeline.sampling.subject_discovery import first_swap_reshape_enabled
import os
import glob
import shutil
from pathlib import Path

def stage_convert_swapped_to_neus2(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 15: Convert Swapped Dataset to NeuS2 ---")

    project_root = config['paths']['project_root']
    script = 'src/scripts/data_conversion/convert_to_neus2.py'
    avatarrex_dir = config['paths']['avatarrex_output']


    src_root = os.path.join(avatarrex_dir, "head_swapped_renders")
    output_root = os.path.join(config['paths']['neus2_output'], "swapped")


    os.makedirs(output_root, exist_ok=True)

    data_type = config['data_type']


    debug_port = get_debug_port('convert_to_neus2') if debug_subprocess else None

    human_scale = NEUS2_HUMAN_SCALE[data_type]
    human_offset = NEUS2_HUMAN_OFFSET[data_type]
    human_offset = [str(x) for x in human_offset]


    processing_mode = 'single_frame'


    if not os.path.exists(src_root):
        logger.warning(f"Warning: Swapped dataset directory not found: {src_root}")
        return


    expected_swapped_dirs = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_dir = os.path.join(src_root, ply_name)
                        if os.path.exists(expected_swapped_dir):
                            expected_swapped_dirs.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_dirs:
        logger.warning("Warning: No swapped directories found for specified subject pairs")
        return

    logger.info(f"Found {len(expected_swapped_dirs)} swapped directories for specified subject pairs")


    for swapped_dir in expected_swapped_dirs:
        logger.info(f"Processing swapped directory: {swapped_dir}")

        cmd = [
            'uv', 'run', 'python', script,
            '--src_root', src_root,
            '--output_root', output_root,
            '--mode', processing_mode,
            '--subjects', swapped_dir,
            '--human_scale', str(human_scale),
            '--human_offset', *human_offset,
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            run_command(cmd, cwd=project_root, debug_port=debug_port)
            logger.info(f"Successfully converted swapped directory: {swapped_dir}")
        except Exception as e:
            logger.error(f"Error converting {swapped_dir}: {e}")
            continue

def stage_neus2_train_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 16: Train NeuS2 on Swapped Dataset ---")

    neus2_project = config['paths']['neus2_project']
    neus2_data_dir = os.path.join(config['paths']['neus2_output'], "swapped")
    conda_env = config['conda_envs']['neus2']

    if not os.path.exists(neus2_data_dir):
        logger.warning(f"Warning: Swapped NeuS2 data directory not found: {neus2_data_dir}")
        return


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]

                        expected_subject_dir = os.path.join(neus2_data_dir, ply_name)
                        if os.path.exists(expected_subject_dir):
                            expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for specified subject pairs")
        return

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for specified subject pairs")


    neus2_cfg = config['pipeline_stages'].get('16_neus2_train_swapped', config['pipeline_stages']['4_neus2_train'])


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Training NeuS2 for swapped subject: {swapped_subject}")

        base_dir = os.path.join(neus2_data_dir, swapped_subject)
        output_dir = neus2_cfg.get('output_dir_template', 'swapped_{subject_name}_60k_ek0.02/{subject_name}').format(subject_name=swapped_subject)

        try:

            run_neus2_training_and_export(
                config=config,
                subject_name=swapped_subject,
                base_dir=base_dir,
                output_dir=output_dir,
                stage_config=neus2_cfg,
                is_swapped=True,
                debug_subprocess=debug_subprocess
            )
            logger.info(f"Successfully trained and exported mesh for swapped subject: {swapped_subject}")
        except Exception as e:
            logger.error(f"Error training NeuS2 for {swapped_subject}: {e}")
            continue

def stage_copy_neus2_mesh_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 17: Copy NeuS2 Mesh for Swapped Subjects ---")

    neus2_project = config['paths']['neus2_project']
    neus2_output_dir = os.path.join(neus2_project, 'output')
    avatarrex_dir = config['paths']['avatarrex_output']


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for specified subject pairs")
        return

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for specified subject pairs")


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Copying mesh for swapped subject: {swapped_subject}")


        neus2_cfg = config['pipeline_stages'].get('16_neus2_train_swapped', config['pipeline_stages']['4_neus2_train'])
        output_dir_template = neus2_cfg.get('output_dir_template', 'swapped_{subject_name}_60k_ek0.02/{subject_name}')


        src_mesh_path = os.path.join(
            neus2_output_dir,
            output_dir_template.format(subject_name=swapped_subject),
            'transforms_0000', 'evaluation', 'mesh', 'scene_transforms_0000_frame_0000.obj'
        )


        dest_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject, 'mesh', 'neus2_raw')
        os.makedirs(dest_dir, exist_ok=True)
        dest_mesh_path = os.path.join(dest_dir, '0000.obj')

        logger.info(f"  From: {src_mesh_path}")
        logger.info(f"  To:   {dest_mesh_path}")

        if os.path.exists(src_mesh_path):
            shutil.copy(src_mesh_path, dest_mesh_path)
            logger.info(f"Successfully copied mesh for {swapped_subject}")
        else:
            logger.warning(f"Warning: Source mesh not found: {src_mesh_path}")

def stage_clean_mesh_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 18: Clean Swapped Mesh ---")

    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    script = 'src/scripts/mesh/clean_neus2.py'
    cfg = config['pipeline_stages'].get('6_clean_mesh', {})


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for specified subject pairs")
        return

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for specified subject pairs")


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Cleaning mesh for swapped subject: {swapped_subject}")

        subject_root_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject)

        if not os.path.exists(subject_root_dir):
            logger.warning(f"Warning: Swapped subject directory not found: {subject_root_dir}")
            continue

        cmd = [
            'uv', 'run', 'python', script,
            '--subject_root_dir', subject_root_dir,
            '--mesh_frame_idx', str(cfg.get('mesh_frame_idx', 0)),
            '--dataset_type', cfg.get('dataset_type', 'thuman2')
        ]
        debug_port = get_debug_port('clean_mesh') if debug_subprocess else None
        run_command(cmd, cwd=project_root, debug_port=debug_port)

def stage_4d_dress_parsing_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 19: 4DDress Parsing for Swapped Subjects ---")

    cfg = config['pipeline_stages'].get('19_parse_mesh_swapped', {})
    fourd_dress_project = config['paths']['fourd_dress_project']
    dataset_dir = os.path.join(config['paths']['avatarrex_output'], 'head_swapped_renders')
    env = config['conda_envs']['fourd_dress']
    script = '4dhumanparsing/multi_view_parsing.py'


    python_prefix = build_python_command(fourd_dress_project, env)


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for specified subject pairs")
        return


    reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" in sub and "_with_color_transfer" in sub]
    non_reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" not in sub and "_with_color_transfer" in sub]
    reshape_enabled = first_swap_reshape_enabled(config)
    if reshape_enabled:
        expected_swapped_subjects = reshaped_subjects
    else:
        expected_swapped_subjects = non_reshaped_subjects
    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for specified subject pairs")


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Parsing swapped subject: {swapped_subject}")

        cmd = python_prefix + [
            script,
            '--dataset_dir', dataset_dir,
            '--dataset', 'Avatarrex',
            '--res_scale', str(cfg.get('res_scale', 1)),
            '--subj', swapped_subject,
            '--n_start', '0', '--num', '1',
            '--n_digit_padding', str(cfg.get('n_digit_padding', 4)),
            '--outfit', cfg.get('outfit', 'Extended'),
            '--first_frame', str(cfg.get('first_frame', 0))
        ]


        if cfg.get('dataset_type') == 'actorshq':
            cmd.extend(['--actorshq'])
        if cfg.get('disable_sam_votes', False):
            cmd.append('--disable_sam_votes')
        if cfg.get('disable_sam_inference', False):
            cmd.append('--disable_sam_inference')
        if cfg.get('save_sam_mask', False):
            cmd.append('--save_mask')
        if cfg.get('sam_label_gain_scale', 1.0) != 1.0:
            cmd.append('--sam_label_gain_enable')
            cmd.append('--sam_label_gain_scale')
            cmd.append(str(cfg['sam_label_gain_scale']))

        logger.info(f"Running 4DDress parsing: {' '.join(cmd)}")
        run_command(cmd, cwd=fourd_dress_project, debug_port=None)

def stage_process_mesh_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 20: Process Mesh for Swapped Subjects ---")

    cfg = config['pipeline_stages'].get('20_process_mesh_swapped', {})
    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    script = 'src/scripts/mesh/process_meshes.py'


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for specified subject pairs")
        return


    reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" in sub and "_with_color_transfer" in sub]
    non_reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" not in sub and "_with_color_transfer" in sub]
    reshape_enabled = first_swap_reshape_enabled(config)
    if reshape_enabled:
        expected_swapped_subjects = reshaped_subjects
    else:
        expected_swapped_subjects = non_reshaped_subjects

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for processing")


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Processing swapped subject: {swapped_subject}")


        data_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject)
        output_dir = os.path.join(data_dir, 'mesh', 'processed')


        if not os.path.exists(data_dir):
            logger.warning(f"Warning: Swapped subject directory not found: {data_dir}")
            continue

        gender_head = 'neutral'
        gender_body = 'neutral'


        import re
        pattern = r'swapped_(\d+)head_on_(\d+)body_'
        match = re.match(pattern, swapped_subject)

        if match:
            head_subject_id = match.group(1)
            body_subject_id = match.group(2)
            logger.info(f"Parsed swapped subject: head donator={head_subject_id}, body donator={body_subject_id}")


            config_subjects = config.get('subjects', [])
            gender_a = config.get('smpl_gender_a', 'neutral')
            gender_b = config.get('smpl_gender_b', 'neutral')


            if len(config_subjects) >= 2:
                subject_to_gender = {
                    config_subjects[0]: gender_a,
                    config_subjects[1]: gender_b
                }


                gender_head = subject_to_gender.get(head_subject_id, gender_a)
                gender_body = subject_to_gender.get(body_subject_id, gender_a)

                logger.info(f"Gender assignment: head donator {head_subject_id} -> {gender_head}, body donator {body_subject_id} -> {gender_body}")
            else:
                logger.warning(f"Warning: Insufficient subjects in config ({len(config_subjects)}). Using neutral gender for both donators.")
        else:
            logger.warning(f"Warning: Could not parse swapped subject name '{swapped_subject}'. Using neutral gender for both donators.")


        cmd = [
            'uv', 'run', 'python', script,
            '--data_dir', data_dir,
            '--output_dir', output_dir,
            '--frame_start', '0', '--frame_end', '1',
            '--frame_step', str(cfg.get('frame_step', 1)),
            '--use_gpu',
            '--frame_format_digits', str(cfg.get('frame_format_digits', 4)),
            '--dataset_type', cfg.get('dataset_type', 'thuman2'),
            '--gender_head', gender_head,
            '--gender_body', gender_body
        ]


        if reshape_enabled:
            cmd.append('--use_body_reshaping_for_swapped')


        if cfg.get('use_no_sam_labels', False):
            cmd.append('--use_no_sam_labels')

        debug_port = get_debug_port('process_mesh') if debug_subprocess else None

        logger.info(f"Running process_mesh for swapped subject: {swapped_subject}")
        run_command(cmd, cwd=project_root, debug_port=debug_port)

def stage_lbs_transfer_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 21: LBS Transfer for Swapped Subjects ---")

    cfg = config['pipeline_stages'].get('9_lbs_transfer', {})
    cfg.update(config['pipeline_stages'].get('21_lbs_transfer_swapped', {}))
    robust_lbs_project = config['paths']['robust_lbs_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['skw_transfer']
    script = 'src/avatarrex_transfer.py'
    visualize = cfg.get('visualize', False)


    python_prefix = build_python_command(robust_lbs_project, env)


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for LBS transfer")
        return


    reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" in sub and "_with_color_transfer" in sub]
    non_reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" not in sub and "_with_color_transfer" in sub]
    reshape_enabled = first_swap_reshape_enabled(config)
    if reshape_enabled:
        expected_swapped_subjects = reshaped_subjects
    else:
        expected_swapped_subjects = non_reshaped_subjects

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for LBS transfer")


    for swapped_subject in expected_swapped_subjects:
        logger.info(f"Processing LBS transfer for swapped subject: {swapped_subject}")


        swapped_data_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject)
        data_dir = os.path.join(swapped_data_dir, 'mesh', 'processed')


        if not os.path.exists(data_dir):
            logger.warning(f"Warning: Processed mesh directory not found for swapped subject: {data_dir}")
            continue


        full_nerf_mesh_path_template = os.path.join(swapped_data_dir, 'mesh', 'labeled', 'vis-labeled-mesh-f{frame_id}_extended.ply')

        cmd = python_prefix + [
            script,
            '--data_dir', data_dir,
            '--output_dir', data_dir,
            '--target_mesh', cfg.get('target_mesh', 'smpl_body.obj'),
            '--frame_start', '0', '--frame_end', '1',
            '--full_nerf_mesh_path_template', full_nerf_mesh_path_template,
            '--frame_format_digits', str(cfg.get('frame_format_digits', 4)),
            '--dataset_type', cfg.get('dataset_type', 'thuman2')
        ]


        if visualize:
            cmd.append('--visualize')

        logger.info(f"Running LBS transfer for swapped subject: {swapped_subject}")
        logger.info(f"Command: {' '.join(cmd)}")
        run_command(cmd, cwd=robust_lbs_project, debug_port=None)

def validate_splatting_avatar_swapped_single_subject(config, subject_pattern):

    iteration = config['pipeline_stages']['10_splatting_avatar']['iteration_default']

    subject_dir = Path(config['paths']['avatarrex_output']) / 'head_swapped_renders' / 'output-splatting' / 'swapped' / subject_pattern / 'point_cloud' / f"iteration_{iteration}" / 'point_cloud.ply'
    matches = glob.glob(str(subject_dir))
    if len(matches) == 0:
        return False
    logger.info(f"✅ Skipping SplattingAvatar training for {subject_pattern} because output already exists")
    return True

def stage_splatting_avatar_swapped(config, subjects, debug_subprocess=False):

    logger.info("\n--- Stage 22: SplattingAvatar Training for Swapped Subjects ---")

    cfg = config['pipeline_stages'].get('10_splatting_avatar', {})

    cfg.update(config['pipeline_stages'].get('22_splatting_avatar_swapped', {}))
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['splatting']
    script = 'train_splatting_avatar.py'


    python_prefix = build_python_command(splatting_avatar_project, env)


    expected_swapped_subjects = []
    for i, subject_a in enumerate(subjects):
        for j, subject_b in enumerate(subjects):
            if i == j:
                continue


            try:
                swapped_dirs = find_swapped_directories(config, subject_a, subject_b)
                for swapped_dir in swapped_dirs:

                    ply_pattern = os.path.join(swapped_dir, "swapped_*.ply")
                    ply_files = glob.glob(ply_pattern)
                    swapped_ply_files = [f for f in ply_files if "swapped_" in os.path.basename(f)]

                    for ply_file in swapped_ply_files:
                        ply_filename = os.path.basename(ply_file)
                        ply_name = os.path.splitext(ply_filename)[0]
                        expected_swapped_subjects.append(ply_name)
            except Exception as e:
                logger.warning(f"Warning: Could not find swapped directories for pair {subject_a} -> {subject_b}: {e}")
                continue

    if not expected_swapped_subjects:
        logger.warning("Warning: No swapped subjects found for SplattingAvatar training")
        return


    reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" in sub and "_with_color_transfer" in sub]
    non_reshaped_subjects = [sub for sub in expected_swapped_subjects if "_from_reshaped_" not in sub and "_with_color_transfer" in sub]
    reshape_enabled = first_swap_reshape_enabled(config)
    if reshape_enabled:
        expected_swapped_subjects = reshaped_subjects
    else:
        expected_swapped_subjects = non_reshaped_subjects

    logger.info(f"Found {len(expected_swapped_subjects)} swapped subjects for SplattingAvatar training")


    for swapped_subject in expected_swapped_subjects:
        if validate_splatting_avatar_swapped_single_subject(config, subject_pattern=swapped_subject):
            continue
        logger.info(f"Training SplattingAvatar for swapped subject: {swapped_subject}")


        dat_dir = os.path.join(avatarrex_dir, 'head_swapped_renders', swapped_subject)


        if not os.path.exists(dat_dir):
            logger.warning(f"Warning: Swapped subject directory not found: {dat_dir}")
            continue


        model_path = f"swapped/{swapped_subject}"


        total_iteration = cfg.get('iteration_default', 10000)


        cmd = python_prefix + [
            script,
            '--configs', cfg.get('configs', 'configs/splatting_avatar.yaml;configs/thuman2.yaml'),
            '--dat_dir', dat_dir,
            '--ip', 'none',
            '--model_path', model_path,
            '--total_iteration', str(total_iteration),
            '--batch_size', str(cfg.get('batch_size', 1)),
            '--bg_color', cfg.get('bg_color', 'black'),
            '--num_workers', str(cfg.get('num_workers', 4))
        ]

        logger.info(f"Training SplattingAvatar for swapped subject: {swapped_subject}")
        logger.info(f"Command: {' '.join(cmd)}")
        run_command(cmd, cwd=splatting_avatar_project, debug_port=None)
