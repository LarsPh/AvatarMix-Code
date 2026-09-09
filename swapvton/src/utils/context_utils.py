from pathlib import Path
from loguru import logger


def get_context_code(config):


    if 'context' not in config:
        logger.warning("No 'context' section in config, defaulting to context 1.1 (normal, normal)")
        return '1.1'

    parse_v = config['context'].get('parse_mesh_version', 'normal')
    parse_swap_v = config['context'].get('parse_mesh_swapped_version', 'normal')


    valid_versions = ['normal', 'no_sam']
    if parse_v not in valid_versions:
        raise ValueError(f"Invalid parse_mesh_version '{parse_v}', must be 'normal' or 'no_sam'")
    if parse_swap_v not in valid_versions:
        raise ValueError(f"Invalid parse_mesh_swapped_version '{parse_swap_v}', must be 'normal' or 'no_sam'")


    if parse_v == 'normal' and parse_swap_v == 'normal':
        return '1.1'
    elif parse_v == 'normal' and parse_swap_v == 'no_sam':
        return '1.2'
    elif parse_v == 'no_sam' and parse_swap_v == 'normal':
        return '2.1'
    else:
        return '2.2'


def get_root_suffix(context_code):

    suffixes = {
        '1.1': '',
        '1.2': '_normal_nosam',
        '2.1': '_nosam_normal',
        '2.2': '_nosam_nosam'
    }

    if context_code not in suffixes:
        raise ValueError(f"Invalid context_code '{context_code}', must be one of {list(suffixes.keys())}")

    return suffixes[context_code]


def resolve_all_roots(config):


    if 'avatarrex_output' in config['paths']:

        base_dataset_root = Path(config['paths']['avatarrex_output'])
    else:
        raise KeyError("Missing 'avatarrex_output' in config['paths']")

    if 'neus2_output' in config['paths']:
        base_neus2_root = Path(config['paths']['neus2_output'])
    else:
        raise KeyError("Missing 'neus2_output' in config['paths']")


    context_code = get_context_code(config)
    parse_v = config.get('context', {}).get('parse_mesh_version', 'normal')
    parse_swap_v = config.get('context', {}).get('parse_mesh_swapped_version', 'normal')


    if parse_v == 'normal':
        original_processing_suffix = ''
    else:
        original_processing_suffix = get_root_suffix('2.1')


    context_suffix = get_root_suffix(context_code)

    resolved = {

        'label_root': base_dataset_root,
        'original_processing_root': Path(f"{base_dataset_root}{original_processing_suffix}"),
        'swapped_input_root': Path(f"{base_dataset_root}{original_processing_suffix}"),
        'swapped_output_root': Path(f"{base_dataset_root}{context_suffix}"),


        'neus2_label_root': base_neus2_root,
        'neus2_original_processing_root': Path(f"{base_neus2_root}{original_processing_suffix}"),
        'neus2_swapped_input_root': Path(f"{base_neus2_root}{original_processing_suffix}"),
        'neus2_swapped_output_root': Path(f"{base_neus2_root}{context_suffix}"),


        'context_code': context_code,
        'parse_mesh_version': parse_v,
        'parse_mesh_swapped_version': parse_swap_v,
    }

    return resolved


def get_label_file_path(subject_dir, frame_id, label_version):

    if label_version not in ['normal', 'no_sam']:
        raise ValueError(f"Invalid label_version '{label_version}', must be 'normal' or 'no_sam'")

    labels_dir = Path(subject_dir) / "Semantic/process/labels_auto_extended"
    suffix = "_no_sam" if label_version == 'no_sam' else ""
    return labels_dir / f"label-f{frame_id:04d}{suffix}.ply"


def validate_context_dependencies(config):

    if 'resolved_paths' not in config:
        raise KeyError("Config must have 'resolved_paths' - call resolve_all_roots() first")

    context_code = config['resolved_paths']['context_code']
    label_root = config['resolved_paths']['label_root']
    subjects = config['subjects']

    if context_code == '1.2':


        import glob

        missing = []
        found_any_swapped = False
        head_swapped_renders = label_root / "head_swapped_renders"

        if not head_swapped_renders.exists():
            raise RuntimeError(
                f"Context 1.2 requires swapped renders directory to exist.\n"
                f"Directory not found: {head_swapped_renders}\n"
                f"Run Context 1.1 stages 1-13 first to create and render swapped subjects."
            )

        for i, subject_a in enumerate(subjects):
            for j, subject_b in enumerate(subjects):
                if i == j:
                    continue


                pattern = str(head_swapped_renders / f"swapped_{subject_a}head_on_{subject_b}body*")
                swapped_dirs = glob.glob(pattern)

                if not swapped_dirs:
                    logger.warning(f"No swapped directories found for pair {subject_a} -> {subject_b}")
                    continue

                found_any_swapped = True


                for swapped_dir in swapped_dirs:
                    label_file = Path(swapped_dir) / "Semantic/process/labels_auto_extended/label-f0000_no_sam.ply"
                    if not label_file.exists():
                        missing.append(f"{Path(swapped_dir).name}/label-f0000_no_sam.ply")

        if not found_any_swapped:
            raise RuntimeError(
                f"Context 1.2 requires swapped subjects to exist.\n"
                f"No swapped directories found in: {head_swapped_renders}\n"
                f"Run Context 1.1 stages 1-19 first to create swapped subjects and labels."
            )

        if missing:
            raise RuntimeError(
                f"Context 1.2 requires Context 1.1 stage 19 (parse_mesh_swapped with no_sam) to be completed.\n"
                f"Missing swapped label files: {', '.join(missing)}\n"
                f"Run the following first:\n"
                f"  python pipeline.py --config <config_1.1> --run-stages \"19_parse_mesh_swapped\"\n"
                f"Ensure 19_parse_mesh_swapped has disable_sam_votes: true to generate no_sam labels"
            )

    elif context_code == '2.1':

        missing = []
        for subject in subjects:
            label_file = label_root / subject / "Semantic/process/labels_auto_extended/label-f0000_no_sam.ply"
            if not label_file.exists():
                missing.append(f"{subject}/label-f0000_no_sam.ply")

        if missing:
            raise RuntimeError(
                f"Context 2.1 requires Context 1.1 stages 1-7 (parse_mesh with no_sam) to be completed.\n"
                f"Missing label files: {', '.join(missing)}\n"
                f"Run the following first:\n"
                f"  python pipeline.py --config <config_1.1> --run-stages \"1_render_thuman,...,7_parse_mesh\"\n"
                f"Ensure 7_parse_mesh has disable_sam_votes: true to generate no_sam labels"
            )

    elif context_code == '2.2':


        missing_labels = []
        for subject in subjects:
            label_file = label_root / subject / "Semantic/process/labels_auto_extended/label-f0000_no_sam.ply"
            if not label_file.exists():
                missing_labels.append(f"{subject}/label-f0000_no_sam.ply")


        processing_root = config['resolved_paths']['original_processing_root']
        missing_processing = []
        for subject in subjects:
            splatting_output = processing_root / subject / "output-splatting"
            if not splatting_output.exists():
                missing_processing.append(f"{subject}/output-splatting")

        if missing_labels or missing_processing:
            error_msg = "Context 2.2 has missing dependencies:\n"
            if missing_labels:
                error_msg += f"  - Missing no_sam labels from Context 1.1: {', '.join(missing_labels)}\n"
            if missing_processing:
                error_msg += f"  - Missing processing from Context 2.1: {', '.join(missing_processing)}\n"
            error_msg += "Run the following in order:\n"
            error_msg += "  1. python pipeline.py --config <config_1.1> --run-stages \"1_render_thuman,...,7_parse_mesh\"\n"
            error_msg += "  2. python pipeline.py --config <config_2.1> --run-stages \"8_process_mesh,...,18_clean_mesh_swapped\""
            raise RuntimeError(error_msg)


def auto_create_context_roots(config):

    if 'resolved_paths' not in config:
        raise KeyError("Config must have 'resolved_paths' - call resolve_all_roots() first")

    resolved = config['resolved_paths']
    label_root = resolved['label_root']


    for key in ['original_processing_root', 'swapped_output_root',
                'neus2_original_processing_root', 'neus2_swapped_output_root']:
        root_path = resolved[key]


        if root_path == label_root or root_path == resolved.get('neus2_label_root'):
            continue

        if not root_path.exists():
            root_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created context root directory: {root_path}")
