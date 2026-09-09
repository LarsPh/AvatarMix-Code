from loguru import logger
from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from pipeline.core.constants import DATA_TYPES
from pipeline.execution.debug_support import get_debug_port

def stage_render_thuman(config, subjects):
    logger.info("\n--- Stage 1: Rendering THuman2.0 ---")
    data_type = config['data_type']


    if data_type != 'thuman2':
        logger.info(f"Skipping rendering stage for {data_type} dataset (only THuman2 requires rendering)")
        return

    cfg = config['pipeline_stages']['1_render_thuman']
    render_project_dir = config['paths']['thuman2_render_project']
    render_script = 'render_batch.py'
    output_dir = config['paths']['thuman2_render_output']
    env = config['conda_envs']['thuman2_render']


    python_prefix = build_python_command(render_project_dir, env)

    for sub in subjects:
        cmd = python_prefix + [
            render_script,
            '-size', str(cfg['size']),
            '-additional_pitch_views',
            '-no_parent_dir',
            '-start_subject', sub,
            '-end_subject', str(int(sub) + 1),
            '-out_dir', output_dir
        ]
        if cfg.get('headless', False):
            cmd += ['--headless', 'True']
        run_command(cmd, cwd=render_project_dir)

def stage_convert_to_avatarrex(config, subjects, debug_subprocess=False):
    logger.info(f"\n--- Stage 2: Converting {config['data_type']} to AvatarRex format ---")
    project_root = config['paths']['project_root']
    data_type = config['data_type']
    if data_type not in DATA_TYPES:
        raise ValueError(f"Invalid data type: {data_type}")

    script = f'src/scripts/data_conversion/convert_{data_type}_to_avatarrex.py'
    src_dir = config['paths']['dataset_for_conversion']
    avatarrex_dir = config['paths']['avatarrex_output']

    cmd = [
        'uv', 'run', 'python', script,
    ]

    if data_type == 'thuman2':
        cmd += [
            '--thuman2_dir', src_dir,
            '--avatarrex_dir', avatarrex_dir,
            '--subjects', *subjects
        ]
    elif data_type == 'mvhumannet':
        cmd += [
            '--mvhumannet_dir', src_dir,
            '--avatarrex_dir', avatarrex_dir,
            '--subjects', *subjects
        ]
        cmd += ['--no_skip_existing'] if config['pipeline_stages']['2_convert_to_avatarrex'].get('skip_existing', True) else []
        transform_mode = config['pipeline_stages']['2_convert_to_avatarrex'].get('transform_mode', None)
        if transform_mode is not None:
            cmd += ['--transform_mode', transform_mode]
        exclude_cams = config['pipeline_stages']['2_convert_to_avatarrex'].get('exclude_cams', None)
        if exclude_cams is not None:
            cmd += ['--exclude_cams', *exclude_cams]
    elif data_type == 'talkbody4d':


        tb_cfg = config.get('pipeline_stages', {}).get('2_convert_to_avatarrex', {})
        export_mode = str(tb_cfg.get('export_mode', 'single_frame')).strip().lower()


        transform_mode = tb_cfg.get('transform_mode', None)
        downscale_factor = tb_cfg.get('downscale_factor', None)
        crop_dilate_ratio = tb_cfg.get('crop_dilate_ratio', None)
        target_long_side = tb_cfg.get('target_long_side', None)
        analyze_crop_stats_only = bool(tb_cfg.get('analyze_crop_stats_only', False))
        gender = tb_cfg.get('gender', None)
        smplx_only = bool(tb_cfg.get('smplx_only', False))
        exclude_cams = tb_cfg.get('exclude_cams', None)


        frames_list_cfg = tb_cfg.get('frames', None)
        frame_start = tb_cfg.get('frame_start', None)
        frame_end = tb_cfg.get('frame_end', None)
        frame_step = tb_cfg.get('frame_step', None)
        frame_padding = tb_cfg.get('frame_padding', None)
        print_video_stats_only = bool(tb_cfg.get('print_video_stats_only', False))
        overwrite = bool(tb_cfg.get('overwrite', False))

        if export_mode == "sequence":

            base_subjects: list[str] = []
            for tok in subjects:
                tok = str(tok).strip()
                parts = tok.split("_")
                if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6:
                    base_subjects.append("_".join(parts[:-1]))
                else:
                    base_subjects.append(tok)
            base_subjects = sorted(set(base_subjects))

            cmd_sub = cmd + [
                '--talkbody4d_dir', src_dir,
                '--avatarrex_dir', avatarrex_dir,
                '--subjects', *base_subjects,
                '--export_mode', 'sequence',
            ]
            if transform_mode is not None:
                cmd_sub += ['--transform_mode', str(transform_mode)]
            if downscale_factor is not None:
                cmd_sub += ['--downscale_factor', str(downscale_factor)]
            if crop_dilate_ratio is not None:
                cmd_sub += ['--crop_dilate_ratio', str(float(crop_dilate_ratio))]
            if target_long_side is not None:
                cmd_sub += ['--target_long_side', str(int(target_long_side))]
            if frame_padding is not None:
                cmd_sub += ['--frame_padding', str(int(frame_padding))]
            if print_video_stats_only:
                cmd_sub += ['--print_video_stats_only']
            if analyze_crop_stats_only:
                cmd_sub += ['--analyze_crop_stats_only']
            if overwrite:
                cmd_sub += ['--overwrite']
            if gender is not None:
                cmd_sub += ['--gender', str(gender)]
            if smplx_only:
                cmd_sub += ['--smplx_only']
            if exclude_cams is not None:
                cams_list = exclude_cams if isinstance(exclude_cams, (list, tuple)) else [exclude_cams]
                cmd_sub += ['--exclude_cams', *[str(x) for x in cams_list]]


            if frames_list_cfg is not None:
                if not isinstance(frames_list_cfg, (list, tuple)):
                    raise ValueError("talkbody4d: pipeline_stages.2_convert_to_avatarrex.frames must be a list of ints")
                cmd_sub += ['--frames', *[str(int(x)) for x in frames_list_cfg]]
            else:
                if frame_start is not None:
                    cmd_sub += ['--frame_start', str(int(frame_start))]
                if frame_end is not None:
                    cmd_sub += ['--frame_end', str(int(frame_end))]
                if frame_step is not None:
                    cmd_sub += ['--frame_step', str(int(frame_step))]

            run_command(cmd_sub, cwd=project_root, debug_port=get_debug_port('convert_to_avatarrex') if debug_subprocess else None)
            return


        by_subject: dict[str, list[int]] = {}
        for tok in subjects:
            tok = str(tok).strip()
            parts = tok.split("_")
            if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6:
                base = "_".join(parts[:-1])
                frame = int(parts[-1])
            else:
                base = tok
                frame = 0
            by_subject.setdefault(base, []).append(frame)

        for base_subj, frames in by_subject.items():
            cmd_sub = cmd + [
                '--talkbody4d_dir', src_dir,
                '--avatarrex_dir', avatarrex_dir,
                '--subjects', base_subj,
                '--export_mode', 'single_frame',
                '--frames', *[str(f) for f in sorted(set(frames))],
            ]
            if transform_mode is not None:
                cmd_sub += ['--transform_mode', str(transform_mode)]
            if downscale_factor is not None:
                cmd_sub += ['--downscale_factor', str(downscale_factor)]
            if crop_dilate_ratio is not None:
                cmd_sub += ['--crop_dilate_ratio', str(float(crop_dilate_ratio))]
            if target_long_side is not None:
                cmd_sub += ['--target_long_side', str(int(target_long_side))]
            if print_video_stats_only:
                cmd_sub += ['--print_video_stats_only']
            if analyze_crop_stats_only:
                cmd_sub += ['--analyze_crop_stats_only']
            if overwrite:
                cmd_sub += ['--overwrite']
            if gender is not None:
                cmd_sub += ['--gender', str(gender)]
            if smplx_only:
                cmd_sub += ['--smplx_only']
            if exclude_cams is not None:
                cams_list = exclude_cams if isinstance(exclude_cams, (list, tuple)) else [exclude_cams]
                cmd_sub += ['--exclude_cams', *[str(x) for x in cams_list]]
            run_command(cmd_sub, cwd=project_root, debug_port=get_debug_port('convert_to_avatarrex') if debug_subprocess else None)
        return
    elif data_type == 'actorshq':


        actorshq_cfg = config['pipeline_stages']['2_convert_to_avatarrex'].get('actorshq', {})
        export_mode = str(actorshq_cfg.get('export_mode', 'single_frame'))
        default_frame = int(actorshq_cfg.get('frame', 0))
        frames_map = actorshq_cfg.get('frames_by_subject', actorshq_cfg.get('frame_by_subject', {}))
        if not isinstance(frames_map, dict):
            frames_map = {}

        def _frame_for_token(tok: str) -> int:
            tok_s = str(tok).strip()
            if tok_s in frames_map:
                return int(frames_map[tok_s])
            parts = tok_s.split('_')
            if len(parts) >= 2:
                actor = parts[0]
                seq_in = parts[1]

                cands = [f'{actor}_{seq_in}']
                if seq_in.lower().startswith('seq') and seq_in[3:].isdigit():
                    cands.append(f'{actor}_Sequence{int(seq_in[3:])}')
                if seq_in.lower().startswith('sequence') and seq_in[8:].isdigit():
                    cands.append(f'{actor}_Seq{int(seq_in[8:])}')
                for c in cands:
                    if c in frames_map:
                        return int(frames_map[c])
            return int(default_frame)

        per_subject = []
        for sub in subjects:
            parts = str(sub).split('_')
            if len(parts) < 2:
                raise ValueError(f"Invalid ActorHQ subject token '{sub}'. Expected ActorXX_SequenceY.")
            actor = parts[0]
            sequence = parts[1]

            if sequence.lower().startswith("seq") and sequence[3:].isdigit():
                sequence_disk = f"Sequence{int(sequence[3:])}"
            else:
                sequence_disk = sequence

            frame_sel = None
            if len(parts) >= 3 and parts[-1].isdigit() and len(parts[-1]) == 6:
                frame_sel = int(parts[-1])
            else:
                frame_sel = _frame_for_token(str(sub))
            per_subject.append((actor, sequence_disk, int(frame_sel)))


        base_args = [
            '--actorshq_dir', src_dir,
            '--avatarrex_dir', avatarrex_dir,
        ]

        common_opt = []
        if 'export_mode' in actorshq_cfg:
            common_opt += ['--export_mode', str(actorshq_cfg['export_mode'])]
        stage_skip_existing = config.get('pipeline_stages', {}).get('2_convert_to_avatarrex', {}).get('skip_existing', True)
        overwrite = actorshq_cfg.get('overwrite', False) or (stage_skip_existing is False)
        if overwrite:
            common_opt += ['--overwrite']
        if 'resolution' in actorshq_cfg:
            common_opt += ['--resolution', actorshq_cfg['resolution']]
        if 'max_cameras' in actorshq_cfg:
            common_opt += ['--max_cameras', str(actorshq_cfg['max_cameras'])]
        if 'camera_step' in actorshq_cfg:
            common_opt += ['--camera_step', str(actorshq_cfg['camera_step'])]


        unique_frames = sorted(set(f for (_, _, f) in per_subject))
        if export_mode == 'single_frame' and len(unique_frames) > 1:
            for actor, sequence_disk, frame_sel in per_subject:
                cmd_sub = cmd + base_args + [
                    '--actors', actor,
                    '--sequences', sequence_disk,
                    '--frame', str(int(frame_sel)),
                ] + common_opt
                run_command(cmd_sub, cwd=project_root, debug_port=get_debug_port('convert_to_avatarrex') if debug_subprocess else None)
            return


        actors = [a for (a, _, _) in per_subject]
        sequences = [s for (_, s, _) in per_subject]
        cmd += base_args + [
            '--actors', *actors,
            '--sequences', *sequences,
        ] + common_opt
        if export_mode == 'single_frame':

            cmd += ['--frame', str(int(unique_frames[0] if unique_frames else default_frame))]
    else:
        raise ValueError(f"Unsupported data type for conversion: {data_type}")
    debug_port = get_debug_port('convert_to_avatarrex') if debug_subprocess else None
    run_command(cmd, cwd=project_root, debug_port=debug_port)
