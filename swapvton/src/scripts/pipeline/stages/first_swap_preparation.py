from loguru import logger
from scripts.pipeline.core.constants import NEUS2_HUMAN_SCALE, NEUS2_HUMAN_OFFSET, ACTORSHQ_GENDER_MAP
from scripts.pipeline.execution.debug_support import get_debug_port
from scripts.pipeline.execution.subprocess_runner import run_command, run_neus2_training_and_export
from scripts.pipeline.execution.env_utils import build_python_command
from scripts.pipeline.sampling.subject_discovery import get_padded_subject_id
import os
import shutil
from pathlib import Path


def stage_convert_to_neus2(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 3: Converting AvatarRex to NeuS2 format ---")
    project_root = config['paths']['project_root']
    script = 'src/scripts/data_conversion/convert_to_neus2.py'
    src_root = config['paths']['avatarrex_output']
    output_root = config['paths']['neus2_output']
    data_type = config['data_type']


    debug_port = get_debug_port('convert_to_neus2') if debug_subprocess else None

    human_scale = NEUS2_HUMAN_SCALE[data_type]
    human_offset = NEUS2_HUMAN_OFFSET[data_type]
    human_offset = [str(x) for x in human_offset]

    if data_type == 'thuman2':
        processing_mode = 'single_frame'
    elif data_type == 'actorshq':


        actorshq_cfg = config.get('pipeline_stages', {}).get('2_convert_to_avatarrex', {}).get('actorshq', {})
        export_mode = str(actorshq_cfg.get('export_mode', 'single_frame'))
        processing_mode = 'multi_frame' if export_mode == 'sequence' else 'single_frame'
    else:
        processing_mode = 'multi_frame'


    cfg_neus2 = config.get('pipeline_stages', {}).get('3_convert_to_neus2', {}) or {}
    exclude_cams = cfg_neus2.get('exclude_cams', None)
    if exclude_cams is None:

        exclude_cams = config.get('pipeline_stages', {}).get('2_convert_to_avatarrex', {}).get('exclude_cams', None)
    exclude_cams_list = exclude_cams if isinstance(exclude_cams, (list, tuple)) else ([exclude_cams] if exclude_cams is not None else None)

    if processing_mode == 'single_frame':

        for sub in subjects:
            cmd = [
                'uv', 'run', 'python', script,
                '--src_root', src_root,
                '--output_root', output_root,
                '--mode', processing_mode,
                '--subjects', sub,
                '--human_scale', str(human_scale),
                '--human_offset', *human_offset,
            ]
            if exclude_cams_list:
                cmd += ['--exclude_cams', *[str(x) for x in exclude_cams_list]]
            run_command(cmd, cwd=project_root, debug_port=debug_port)
    else:

        cfg = config['pipeline_stages']['3_convert_to_neus2']
        cmd = [
            'uv', 'run', 'python', script,
            '--src_root', src_root,
            '--output_root', output_root,
            '--mode', processing_mode,
            '--human_scale', str(human_scale),
            '--human_offset', *human_offset,
            '--subjects'
        ] + subjects


        if 'frame_padding' in cfg:
            cmd += ['--frame_padding', str(cfg['frame_padding'])]
        if 'max_frames' in cfg:
            cmd += ['--max_frames', str(cfg['max_frames'])]
        if exclude_cams_list:
            cmd += ['--exclude_cams', *[str(x) for x in exclude_cams_list]]

        run_command(cmd, cwd=project_root, debug_port=debug_port)

def stage_neus2_train(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 4: Local NeuS2 Training ---")
    cfg = config['pipeline_stages']['4_neus2_train']
    neus2_data_dir = config['paths']['neus2_output']

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        base_dir = os.path.join(neus2_data_dir, sub_padded)
        output_dir = cfg['output_dir_template'].format(subject_id=sub, subject_id_padded=sub_padded)


        run_neus2_training_and_export(
            config=config,
            subject_name=sub,
            base_dir=base_dir,
            output_dir=output_dir,
            stage_config=cfg,
            is_swapped=False,
            debug_subprocess=debug_subprocess,
        )

def stage_copy_neus2_mesh(config, subjects):
    logger.info("\n--- Stage 5: Copying NeuS2 Mesh ---")
    cfg = config['pipeline_stages']['5_copy_mesh']
    neus2_output_dir = os.path.join(config['paths']['neus2_project'], 'output')
    avatarrex_dir = config['paths']['avatarrex_output']

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        src_mesh_path = os.path.join(neus2_output_dir, cfg['mesh_path_template'].format(subject_id_padded=sub_padded))

        dest_dir = os.path.join(avatarrex_dir, sub_padded, 'mesh', 'neus2_raw')
        os.makedirs(dest_dir, exist_ok=True)
        dest_mesh_path = os.path.join(dest_dir, '0000.obj')

        logger.info(f"Copying mesh for subject {sub}:")
        logger.info(f"  From: {src_mesh_path}")
        logger.info(f"  To:   {dest_mesh_path}")
        shutil.copy(src_mesh_path, dest_mesh_path)

def stage_clean_mesh(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 6: Cleaning Mesh ---")
    cfg = config['pipeline_stages']['6_clean_mesh']
    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    script = 'src/scripts/mesh/clean_neus2.py'

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        subject_root_dir = os.path.join(avatarrex_dir, sub_padded)
        cmd = [
            'uv', 'run', 'python', script,
            '--subject_root_dir', subject_root_dir,
            '--mesh_frame_idx', str(cfg['mesh_frame_idx']),
            '--dataset_type', cfg['dataset_type']
        ]
        debug_port = get_debug_port('clean_mesh') if debug_subprocess else None
        run_command(cmd, cwd=project_root, debug_port=debug_port)

def stage_4d_dress_parsing(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 7: 4DDress Parsing ---")
    cfg = config['pipeline_stages']['7_parse_mesh']
    fourd_dress_project = config['paths']['fourd_dress_project']
    dataset_dir = config['paths']['avatarrex_output']
    sam3_project = config.get('paths', {}).get('sam3_project', None) or config.get('path', {}).get('sam3_project', None)
    env = config['conda_envs']['fourd_dress']
    script = '4dhumanparsing/multi_view_parsing.py'


    python_prefix = build_python_command(fourd_dress_project, env)

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        cmd = python_prefix + [
            script,
            '--dataset_dir', dataset_dir,
            '--dataset', 'Avatarrex',
            '--res_scale', str(cfg['res_scale']),
            '--subj', sub_padded,
            '--n_start', '0', '--num', '1',
            '--n_digit_padding', str(cfg['n_digit_padding']),
            '--outfit', cfg['outfit'],
            '--first_frame', str(cfg['first_frame']),
            '--update_3d',
        ]


        enable_sam3_hands = cfg.get('enable_sam3_hands', False)
        if cfg.get('update_2d', False) or enable_sam3_hands or cfg.get('enable_shoes_label', False):
            cmd.append('--update_2d')
        if cfg.get('dataset_type', 'avatarrex') == 'actorshq':

            cmd.append('--filter_portrait_only')
        if cfg.get('save_sam_mask', False):
            cmd.append('--save_mask')
        if cfg.get('sam_label_gain_scale', 1.0) != 1.0:
            cmd.append('--sam_label_gain_enable')
            cmd.append('--sam_label_gain_scale')
            cmd.append(str(cfg['sam_label_gain_scale']))
        if cfg.get('disable_sam_votes', False):
            cmd.append('--disable_sam_votes')
        if cfg.get('disable_sam_inference', False):
            cmd.append('--disable_sam_inference')
        if cfg.get('enable_shoes_label', False):
            cmd.append('--enable_shoes_label')


        if enable_sam3_hands:
            cmd.append('--enable_sam3_hands')
            if not sam3_project:
                raise ValueError("enable_sam3_hands is true but config['paths']['sam3_project'] is missing")
            cmd += ['--sam3_project', str(sam3_project)]
            if 'sam3_hands_confidence' in cfg:
                cmd += ['--sam3_hands_confidence', str(cfg.get('sam3_hands_confidence', 0.8))]
            if 'sam3_hands_bbox_padding' in cfg:
                cmd += ['--sam3_hands_bbox_padding', str(cfg.get('sam3_hands_bbox_padding', 20))]
            if 'sam3_hands_prompt' in cfg:
                cmd += ['--sam3_hands_prompt', str(cfg.get('sam3_hands_prompt', 'hand'))]
        debug_port = get_debug_port('parse_mesh') if debug_subprocess else None
        run_command(cmd, cwd=fourd_dress_project, debug_port=debug_port)

def stage_process_mesh(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 8: Processing Mesh (separate & combine) ---")
    cfg = config['pipeline_stages']['8_process_mesh']
    project_root = config['paths']['project_root']
    avatarrex_dir = config['paths']['avatarrex_output']
    script = 'src/scripts/mesh/process_meshes.py'


    for i, sub in enumerate(subjects):
        sub_padded = get_padded_subject_id(sub)
        data_dir = os.path.join(avatarrex_dir, sub_padded)
        output_dir = os.path.join(data_dir, 'mesh', 'processed')
        cmd = [
            'uv', 'run', 'python', script,
            '--data_dir', data_dir,
            '--output_dir', output_dir,
            '--frame_start', '0', '--frame_end', '1', #!DEBUG
            '--frame_step', str(cfg['frame_step']),
            '--use_gpu',
            '--frame_format_digits', str(cfg['frame_format_digits']),
            '--dataset_type', cfg['dataset_type']
        ]
        if cfg['dataset_type'] == 'actorshq':
            cmd.append('--gender')
            cmd.append(ACTORSHQ_GENDER_MAP[sub.split('_')[0]])
        elif config.get('read_gender_from_subject_root', True):

            cmd.append('--gender')
            cmd.append(config.get('smpl_gender_a', 'auto') if i == 0 else config.get('smpl_gender_b', 'auto'))
            logger.info(f"Gender for subject {sub}: {config.get('smpl_gender_a', 'auto') if i == 0 else config.get('smpl_gender_b', 'auto')}")
        if cfg.get('create_simplified_mesh', False):
            cmd.append('--create_simplified_mesh')
            cmd.append('--simplification_target_percent')
            cmd.append(str(cfg.get('simplification_target_percent', 0.4)))
        if cfg.get('remove_head', False):
            cmd.append('--remove_head')
        if cfg.get('save_skeleton_mesh', False):
            cmd.append('--save_skeleton_mesh')
        if cfg.get('use_no_sam_mask', False):
            cmd.append('--use_no_sam_mask')
        if cfg.get('use_no_sam_labels', False):
            cmd.append('--use_no_sam_labels')
        if cfg.get('save_label_with_no_sam_suffix', False):
            cmd.append('--save_label_with_no_sam_suffix')
        if cfg.get('generate_cloth_fit_skin_mask', False):
            cmd.append('--generate_cloth_fit_skin_mask')

            if 'hands_gate_min_arm_verts' in cfg:
                cmd += ['--hands_gate_min_arm_verts', str(cfg['hands_gate_min_arm_verts'])]
            if 'hands_gate_min_arm_to_hands_ratio' in cfg:
                cmd += ['--hands_gate_min_arm_to_hands_ratio', str(cfg['hands_gate_min_arm_to_hands_ratio'])]

            sem_cfg = cfg.get("semantic_weighting", {}) or {}
            if bool(sem_cfg.get("enabled", False)):
                cmd.append("--generate_semantic_weighting_masks")
                if "mode" in sem_cfg and sem_cfg.get("mode") is not None:
                    cmd += ["--semantic_mode", str(sem_cfg.get("mode"))]
                thr = sem_cfg.get("thresholds", {}) or {}
                if "chest" in thr:
                    cmd += ["--semantic_chest_threshold", str(thr.get("chest"))]
                if "hip" in thr:
                    cmd += ["--semantic_hip_threshold", str(thr.get("hip"))]
                if "dilate_rings" in sem_cfg:
                    cmd += ["--semantic_dilate_rings", str(sem_cfg.get("dilate_rings"))]
                if "min_component_size" in sem_cfg:
                    cmd += ["--semantic_min_component_size", str(sem_cfg.get("min_component_size"))]
                if bool(sem_cfg.get("debug_ply", False)):
                    cmd.append("--semantic_debug_ply")
        if cfg.get('use_nerf_mesh_smoothing', False):
            cmd.append('--use_nerf_mesh_smoothing')
        if cfg.get('no_torso_skin_for_skin_mask', False):
            cmd.append('--no_torso_skin_for_skin_mask')
        if cfg.get('save_smpl_joints', False):
            cmd.append('--save_smpl_joints')
        if cfg.get('estimate_height_from_betas', False):
            cmd.append('--estimate_height_from_betas')
        if cfg.get('emit_smpl_scale_inv', False):
            cmd.append('--emit_smpl_scale_inv')

        debug_port = get_debug_port('process_mesh') if debug_subprocess else None
        run_command(cmd, cwd=project_root, debug_port=debug_port)


def stage_smplx_fitting(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 7b: SMPL-X fitting to NeuS2 mesh (overwrite smpl_params.npz) ---")
    project_root = config["paths"]["project_root"]
    avatarrex_dir = config["paths"]["avatarrex_output"]

    cfg = config.get("pipeline_stages", {}).get("7b_smplx_fitting", {})
    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        logger.info("SMPL-X fitting stage disabled by config (pipeline_stages.7b_smplx_fitting.enabled=false). Skipping.")
        return


    template_path = cfg.get("fitting_config_template", "src/scripts/smplx_fitting/alt_band_adam.yaml")
    template_path_abs = Path(project_root) / template_path
    if not template_path_abs.exists():
        raise FileNotFoundError(f"SMPL-X fitting config template not found: {template_path_abs}")


    neus_mesh_relpath = cfg.get("neus_mesh_relpath", "mesh/trimesh_cleaned/0000.obj")
    exp_name = str(cfg.get("exp_name", "pipeline")).strip()
    skip_existing = bool(cfg.get("skip_existing", True))
    gender_override = cfg.get("gender", None)


    debug_port = get_debug_port("smplx_fitting") if debug_subprocess else None
    script = "src/scripts/smplx_fitting/run_fitting_from_config.py"

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        subject_root_dir = Path(avatarrex_dir) / sub_padded
        neus_mesh_path = subject_root_dir / neus_mesh_relpath

        if skip_existing:


            if list(subject_root_dir.glob(f"smpl_params_finetuned_*_{exp_name}.npz")):
                logger.info(f"SMPL-X finetune outputs exist for {sub_padded} (exp_name={exp_name}), skipping.")
                continue

        cmd = [
            "uv", "run", "python", script,
            "--config", str(template_path_abs),
            "--subject_dir", str(subject_root_dir),
            "--neus_mesh_path", str(neus_mesh_path),
            "--smpl_model_path", config["paths"]["smpl_model_root"],
        ]
        if exp_name:
            cmd += ["--exp_name", exp_name]
        if gender_override is not None:
            cmd += ["--gender", str(gender_override)]
        run_command(cmd, cwd=project_root, debug_port=debug_port)


def stage_hands_transplant(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 7c: SMPL-X hands transplant (overwrite trimesh_cleaned/0000.obj) ---")
    project_root = config["paths"]["project_root"]
    avatarrex_dir = config["paths"]["avatarrex_output"]
    cfg = config.get("pipeline_stages", {}).get("7c_hands_transplant", {})

    enabled = bool(cfg.get("enabled", True))
    if not enabled:
        logger.info("Hands transplant stage disabled by config (pipeline_stages.7c_hands_transplant.enabled=false). Skipping.")
        return


    pm_cfg = config.get("pipeline_stages", {}).get("8_process_mesh", {})
    dataset_type = str(cfg.get("dataset_type", pm_cfg.get("dataset_type", config.get("data_type", "generic"))))
    frame_format_digits = int(cfg.get("frame_format_digits", pm_cfg.get("frame_format_digits", 8)))

    debug_port_process = get_debug_port("hands_transplant_process_mesh") if debug_subprocess else None
    debug_port_transplant = get_debug_port("hands_transplant") if debug_subprocess else None

    process_mesh_script = "src/scripts/mesh/process_meshes.py"
    transplant_script = "src/scripts/mesh_transplant/smplx_hand_transplant.py"

    for i, sub in enumerate(subjects):
        sub_padded = get_padded_subject_id(sub)
        subject_root_dir = Path(avatarrex_dir) / sub_padded
        body_mesh_path = subject_root_dir / "mesh" / "trimesh_cleaned" / "0000.obj"
        body_mesh_backup_path = subject_root_dir / "mesh" / "trimesh_cleaned" / "0000_old.obj"
        processed_dir = subject_root_dir / "mesh" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        smpl_body_path = processed_dir / "smpl_body.obj"
        smpl_joints_path = processed_dir / "smpl_joints.npy"


        if not smpl_body_path.exists() or not smpl_joints_path.exists():
            logger.info(f"[{sub_padded}] Generating SMPL artifacts via process_meshes.py (SMPL-only)")
            cmd = [
                "uv", "run", "python", process_mesh_script,
                "--data_dir", str(subject_root_dir),
                "--output_dir", str(processed_dir),
                "--frame_start", "0", "--frame_end", "1",
                "--frame_step", "1",
                "--use_gpu",
                "--frame_format_digits", str(frame_format_digits),
                "--dataset_type", str(dataset_type),
                "--save_smpl_joints",
                "--emit_smpl_only",
            ]

            if dataset_type == "actorshq":
                cmd += ["--gender", ACTORSHQ_GENDER_MAP[sub.split("_")[0]]]
            elif config.get("read_gender_from_subject_root", True):
                cmd += ["--gender", str(config.get("smpl_gender_a", "auto") if i == 0 else config.get("smpl_gender_b", "auto"))]
            run_command(cmd, cwd=project_root, debug_port=debug_port_process)

        if not smpl_body_path.exists():
            raise FileNotFoundError(f"[{sub_padded}] Missing SMPL body mesh after process_meshes: {smpl_body_path}")
        if not smpl_joints_path.exists():
            raise FileNotFoundError(f"[{sub_padded}] Missing SMPL joints after process_meshes: {smpl_joints_path}")


        out_dir = subject_root_dir / "mesh" / "hand_transplant_debug"
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[{sub_padded}] Running hands transplant -> {out_dir}")

        tcmd = [
            "uv", "run", "python", transplant_script,
            "--body_mesh_path", str(body_mesh_path),
            "--smplx_mesh_path", str(smpl_body_path),
            "--smplx_joints_path", str(smpl_joints_path),
            "--output_dir", str(out_dir),
        ]

        extra_args = cfg.get("extra_args", None)
        if isinstance(extra_args, list):
            tcmd += [str(x) for x in extra_args]

        run_command(tcmd, cwd=project_root, debug_port=debug_port_transplant)

        transplanted_mesh_path = out_dir / "mesh_unified_with_smplx_hands.obj"
        if not transplanted_mesh_path.exists():
            raise FileNotFoundError(f"[{sub_padded}] Transplant did not produce expected mesh: {transplanted_mesh_path}")


        if not body_mesh_backup_path.exists():
            shutil.copy(str(body_mesh_path), str(body_mesh_backup_path))
            logger.info(f"[{sub_padded}] Backed up cleaned mesh -> {body_mesh_backup_path}")
        else:
            logger.info(f"[{sub_padded}] Backup exists, not overwriting -> {body_mesh_backup_path}")

        shutil.copy(str(transplanted_mesh_path), str(body_mesh_path))
        logger.success(f"[{sub_padded}] Replaced cleaned mesh -> {body_mesh_path}")

def stage_robust_lbs_transfer(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 9: Robust LBS Weight Transfer ---")
    cfg = config['pipeline_stages']['9_lbs_transfer']
    robust_lbs_project = config['paths']['robust_lbs_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['skw_transfer']
    script = 'src/avatarrex_transfer.py'
    visualize = cfg.get('visualize', False)


    python_prefix = build_python_command(robust_lbs_project, env)

    for sub in subjects:
        sub_padded = get_padded_subject_id(sub)
        data_dir = os.path.join(avatarrex_dir, sub_padded, 'mesh', 'processed')
        use_no_sam_labels = config.get('pipeline_stages', {}).get('8_process_mesh', {}).get('use_no_sam_labels', False)
        dir_suffix = '_no_sam' if use_no_sam_labels else ''
        full_nerf_mesh_path_template = os.path.join(avatarrex_dir, sub_padded, 'mesh', f'labeled{dir_suffix}', 'vis-labeled-mesh-f{frame_id}_extended.ply')


        dataset_type_arg = cfg.get('dataset_type', 'avatarrex')
        if config.get('data_type') in {'mvhumannet', 'talkbody4d'}:
            dataset_type_arg = 'avatarrex'

        cmd = python_prefix + [
            script,
            '--data_dir', data_dir,
            '--output_dir', data_dir,
            '--target_mesh', cfg['target_mesh'],
            '--frame_start', '0', '--frame_end', '1',
            '--full_nerf_mesh_path_template', full_nerf_mesh_path_template,
            '--frame_format_digits', str(cfg['frame_format_digits']),
            '--dataset_type', dataset_type_arg
        ]
        if visualize:
            cmd.append('--visualize')
        debug_port = get_debug_port('lbs_transfer') if debug_subprocess else None
        run_command(cmd, cwd=robust_lbs_project, debug_port=debug_port)


def stage_splatting_avatar(config, subjects, debug_subprocess=False):
    logger.info("\n--- Stage 10: Building SplattingAvatar ---")
    cfg = config['pipeline_stages']['10_splatting_avatar']
    splatting_avatar_project = config['paths']['splatting_avatar_project']
    avatarrex_dir = config['paths']['avatarrex_output']
    env = config['conda_envs']['splatting']
    script = 'train_splatting_avatar.py'


    python_prefix = build_python_command(splatting_avatar_project, env)

    for i, sub in enumerate(subjects):
        sub_padded = get_padded_subject_id(sub)
        dat_dir = os.path.join(avatarrex_dir, sub_padded)
        model_path = cfg['model_path_template'].format(subject_id_padded=sub_padded)


        pos = 'a' if i == 0 else 'b'
        total_iteration = cfg.get(f'iteration_{sub}', cfg.get(f'iteration_{pos}', cfg.get('iteration_default', 5000)))
        batch_size = cfg.get(f'batch_size_{sub}', cfg.get(f'batch_size_{pos}', cfg.get('batch_size_default', cfg.get('batch_size', 1))))

        cmd = python_prefix + [
            script,
            '--configs', cfg['configs'],
            '--dat_dir', dat_dir,
            '--ip', 'none',
            '--model_path', model_path,
            '--total_iteration', str(total_iteration),
            '--num_workers', str(cfg.get('num_workers', 4)),
            '--batch_size', str(batch_size)
        ]
        debug_port = get_debug_port('splatting_avatar') if debug_subprocess else None
        run_command(cmd, cwd=splatting_avatar_project, debug_port=debug_port)
