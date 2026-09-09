import os
import glob
import re
from loguru import logger


def get_padded_subject_id(subject_id, padding=4):

    return str(subject_id).zfill(padding)

def _actorshq_shorten_sequence_name(sequence_name: str) -> str:
    s = str(sequence_name)
    m = re.match(r"^Sequence(\d+)$", s, flags=re.IGNORECASE)
    if m:
        return f"Seq{m.group(1)}"
    if s.lower().startswith("seq"):
        return s
    return s.replace("Sequence", "Seq")


def _actorshq_frame_selector_for_token(tok: str, cfg: dict, default_frame: int) -> int:

    frames_map = cfg.get("frames_by_subject", None)
    if frames_map is None:
        frames_map = cfg.get("frame_by_subject", None)
    if not isinstance(frames_map, dict):
        frames_map = {}

    tok_s = str(tok).strip()
    if tok_s in frames_map:
        return int(frames_map[tok_s])

    parts = tok_s.split("_")
    if len(parts) >= 2:
        actor = parts[0]
        seq_in = parts[1]
        seq_short = _actorshq_shorten_sequence_name(seq_in)


        cands = []
        cands.append(f"{actor}_{seq_in}")
        cands.append(f"{actor}_{seq_short}")

        if seq_short.lower().startswith("seq") and seq_short[3:].isdigit():
            cands.append(f"{actor}_Sequence{int(seq_short[3:])}")
        for c in cands:
            if c in frames_map:
                return int(frames_map[c])

    return int(default_frame)


def _normalize_actorshq_subject_tokens(subjects: list[str], config: dict) -> list[str]:

    cfg = config.get("pipeline_stages", {}).get("2_convert_to_avatarrex", {}).get("actorshq", {})
    export_mode = str(cfg.get("export_mode", "single_frame"))
    if export_mode != "single_frame":
        return subjects
    default_frame = int(cfg.get("frame", 0))

    out: list[str] = []
    for tok in subjects:
        parts = str(tok).split("_")


        frame_suffix: int | None = None
        if len(parts) >= 3 and parts[-1].isdigit() and 4 <= len(parts[-1]) <= 6:
            frame_suffix = int(parts[-1])
            parts = parts[:-1]
        if len(parts) < 2:
            out.append(tok)
            continue
        actor = parts[0]
        seq = _actorshq_shorten_sequence_name(parts[1])
        frame_sel = frame_suffix if frame_suffix is not None else _actorshq_frame_selector_for_token(str(tok), cfg, default_frame=default_frame)
        out.append(f"{actor}_{seq}_{int(frame_sel):06d}")
    return out


def _normalize_talkbody4d_subject_tokens(subjects: list[str], config: dict) -> list[str]:

    cfg = config.get("pipeline_stages", {}).get("2_convert_to_avatarrex", {})
    export_mode = str(cfg.get("export_mode", "single_frame")).strip().lower()
    if export_mode != "sequence":
        return subjects
    out: list[str] = []
    for tok in subjects:
        tok_s = str(tok).strip()
        parts = tok_s.split("_")
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 6:
            out.append("_".join(parts[:-1]))
        else:
            out.append(tok_s)

    return sorted(set(out))


def get_reshape_suffix_from_config(config):

    reposing_cfg = config['pipeline_stages']['11_reposing']
    if not reposing_cfg.get('enable_body_reshape', False):
        return ""

    samples = reposing_cfg.get('body_reshape_smoothing_samples', 64)
    std_scale = reposing_cfg.get('body_reshape_sample_std_scale', 0.1)
    suffix = f"_reshaped_{samples}samples_std{std_scale}"


    scale_factor = reposing_cfg.get('body_reshape_scale_factor', 1.0)
    if scale_factor != 1.0:
        suffix += f"_scale{scale_factor}"


    if reposing_cfg.get('body_reshape_distance_weighting', False):
        suffix += "_distweight"

    return suffix


def get_subjects(args, config):

    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(',')]
    else:
        subjects = config['subjects']


    if config.get("data_type") == "actorshq":
        subjects = _normalize_actorshq_subject_tokens(subjects, config)
    if config.get("data_type") == "talkbody4d":
        subjects = _normalize_talkbody4d_subject_tokens(subjects, config)
    return subjects


def discover_highest_iteration(config, subject_id):

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']

    sub_padded = get_padded_subject_id(subject_id)
    model_path = splatting_cfg['model_path_template'].format(subject_id_padded=sub_padded)
    point_cloud_base_dir = os.path.join(avatarrex_dir, 'output-splatting', model_path, 'point_cloud')

    if not os.path.exists(point_cloud_base_dir):
        logger.warning(f"Warning: point_cloud directory not found for subject {subject_id}: {point_cloud_base_dir}")
        return None

    highest_iteration = 0
    iteration_pattern = os.path.join(point_cloud_base_dir, 'iteration_*')

    for iteration_dir in glob.glob(iteration_pattern):
        if os.path.isdir(iteration_dir):
            dir_name = os.path.basename(iteration_dir)
            if dir_name.startswith('iteration_'):
                try:
                    iteration_num = int(dir_name.replace('iteration_', ''))
                    highest_iteration = max(highest_iteration, iteration_num)
                except ValueError:
                    logger.warning(f"Warning: Could not parse iteration number from directory: {dir_name}")

    if highest_iteration > 0:
        logger.info(f"Auto-discovered highest iteration for subject {subject_id}: {highest_iteration}")
        return highest_iteration
    else:
        logger.warning(f"Warning: No valid iteration directories found for subject {subject_id}")
        return None


def get_subject_iteration_and_batch_size(config, subject_id, subject_position):

    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']


    discovered_iteration = discover_highest_iteration(config, subject_id)

    if discovered_iteration is not None:
        iteration_val = discovered_iteration
        logger.info(f"Using auto-discovered iteration for subject {subject_id}: {iteration_val}")
    else:

        iteration_key = f'iteration_{subject_position}'
        iteration_val = splatting_cfg.get(iteration_key, splatting_cfg.get('iteration_default', 5000))
        logger.info(f"Using config iteration for subject {subject_id} (position {subject_position}): {iteration_val}")

    logger.info(f"Subject {subject_id}: iteration_val={iteration_val}")

    return iteration_val


def discover_trained_subjects(config):

    avatarrex_dir = config['paths']['avatarrex_output']
    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']
    output_splatting_dir = os.path.join(avatarrex_dir, 'output-splatting')

    if not os.path.exists(output_splatting_dir):
        logger.warning(f"Output splatting directory not found: {output_splatting_dir}")
        return []

    trained_subjects = []
    data_type = config.get('data_type', 'thuman2')

    if data_type == 'thuman2':

        for subject_id in range(526):
            subject_str = str(subject_id)
            sub_padded = get_padded_subject_id(subject_str)
            model_path = splatting_cfg['model_path_template'].format(subject_id_padded=sub_padded)
            subject_output_dir = os.path.join(output_splatting_dir, model_path)

            if os.path.exists(subject_output_dir):
                trained_subjects.append(subject_str)

    elif data_type == 'actorshq':

        for actor_num in range(1, 9):
            subject_str = f"Actor{actor_num:02d}"
            sub_padded = get_padded_subject_id(subject_str)
            model_path = splatting_cfg['model_path_template'].format(subject_id_padded=sub_padded)
            subject_output_dir = os.path.join(output_splatting_dir, model_path)

            if os.path.exists(subject_output_dir):
                trained_subjects.append(subject_str)

    else:
        raise ValueError(f"Trained subject discovery not supported for data_type: {data_type}")

    logger.info(f"Discovered {len(trained_subjects)} trained subjects: {trained_subjects}")
    return trained_subjects


def first_swap_reshape_enabled(config):

    reposing_cfg = config['pipeline_stages']['11_reposing']
    return reposing_cfg.get('enable_body_reshape', False)


def second_swap_reshape_enabled(config):

    return first_swap_reshape_enabled(config)


def check_reposing_output_exists(config, subject_a, subject_b):

    avatarrex_dir = config['paths']['avatarrex_output']
    reposing_base_dir = os.path.join(avatarrex_dir, "reposing")

    user_A_padded = get_padded_subject_id(subject_a)
    model_B_padded = get_padded_subject_id(subject_b)


    pattern_ab = f"A{user_A_padded}_B{model_B_padded}*"
    pattern_ba = f"A{model_B_padded}_B{user_A_padded}*"

    search_ab = os.path.join(reposing_base_dir, pattern_ab)
    search_ba = os.path.join(reposing_base_dir, pattern_ba)

    return len(glob.glob(search_ab)) > 0 and len(glob.glob(search_ba)) > 0


def check_swapping_output_exists(config, subject_a, subject_b):

    avatarrex_dir = config['paths']['avatarrex_output']
    swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

    user_A_padded = get_padded_subject_id(subject_a)
    model_B_padded = get_padded_subject_id(subject_b)


    pattern_ab = f"A{user_A_padded}_B{model_B_padded}*"
    pattern_ba = f"A{model_B_padded}_B{user_A_padded}*"

    search_ab = os.path.join(swapped_base_dir, pattern_ab)
    search_ba = os.path.join(swapped_base_dir, pattern_ba)

    return len(glob.glob(search_ab)) > 0 and len(glob.glob(search_ba)) > 0


def discover_swapped_directories(config, subjects):

    from pathlib import Path


    avatarrex_dir = Path(config['paths']['avatarrex_output'])
    head_swapped_renders_dir = avatarrex_dir / 'head_swapped_renders'

    if not head_swapped_renders_dir.exists():
        logger.warning(f"Head swapped renders directory not found: {head_swapped_renders_dir}")
        return []


    reshape_enabled = first_swap_reshape_enabled(config)

    swapped_dirs = []


    for subject_id in subjects:

        other_subjects = [s for s in subjects if s != subject_id]
        if not other_subjects:
            logger.warning(f"Cannot find model subject for user {subject_id} in subjects: {subjects}")
            continue

        model_id = other_subjects[0]


        swapped_dir_pattern = f"swapped_{subject_id}head_on_{model_id}body*"


        if reshape_enabled:
            swapped_dir_pattern += "from_reshaped_*"
        else:
            swapped_dir_pattern += "from_point_cloud*"


        swapped_dir_pattern += "with_color_transfer*"


        matching_dirs = list(head_swapped_renders_dir.glob(swapped_dir_pattern))

        if matching_dirs:
            swapped_dirs.extend(matching_dirs)
            logger.debug(f"Found {len(matching_dirs)} swapped directories for {subject_id}")
        else:
            logger.warning(f"No swapped directories found matching pattern: {swapped_dir_pattern}")

    return swapped_dirs
