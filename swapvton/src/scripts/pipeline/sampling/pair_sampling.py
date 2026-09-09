import random
import os
from loguru import logger

from scripts.pipeline.sampling.subject_discovery import (
    get_padded_subject_id,
    discover_trained_subjects,
    check_reposing_output_exists,
    check_swapping_output_exists,
)


def _sample_subject_pair(config, data_type=None, subject_range=None) -> tuple[str, str]:

    data_type = data_type or config.get('data_type', 'thuman2')


    if subject_range is None:
        subject_range = config.get('pair_sampling', {}).get('subject_range')

    if data_type == 'thuman2':

        if subject_range:
            start, end = subject_range
            subject_a, subject_b = random.sample(range(start, end + 1), 2)
        else:
            subject_a, subject_b = random.sample(range(526), 2)
        return str(subject_a), str(subject_b)
    elif data_type == 'actorshq':

        if subject_range:
            start, end = subject_range
            actor_ids = [f"Actor{i:02d}" for i in range(start, end + 1)]
        else:
            actor_ids = [f"Actor{i:02d}" for i in range(1, 9)]
        subject_a, subject_b = random.sample(actor_ids, 2)
        return subject_a, subject_b
    else:
        raise ValueError(f"Random sampling not supported for data_type: {data_type}")


def _sample_trained_pair(config, trained_subjects=None) -> tuple[str, str]:

    if trained_subjects is None:
        trained_subjects = discover_trained_subjects(config)

    if len(trained_subjects) < 2:
        raise ValueError(f"Need at least 2 trained subjects for pairing, found {len(trained_subjects)}")

    subject_a, subject_b = random.sample(trained_subjects, 2)
    return subject_a, subject_b


def _check_pair_swapping_outputs_exist(config, subject_a, subject_b, stages) -> tuple[bool, list[str]]:

    skip_reasons = []


    if "reposing" in stages:
        reposing_exists_a_to_b, reposing_exists_b_to_a, repose_dir_a_to_b, repose_dir_b_to_a = \
            check_reposing_output_exists(config, subject_a, subject_b)
        if reposing_exists_a_to_b:
            skip_reasons.append(f"reposing {subject_a}->{subject_b} exists at {repose_dir_a_to_b}")
        if reposing_exists_b_to_a:
            skip_reasons.append(f"reposing {subject_b}->{subject_a} exists at {repose_dir_b_to_a}")


    if "swapping" in stages or "direct_swapping" in stages:
        swapping_exists_a_to_b, swapping_exists_b_to_a, swap_dir_a_to_b, swap_dir_b_to_a = \
            check_swapping_output_exists(config, subject_a, subject_b)
        swap_desc = "direct_swapping" if "direct_swapping" in stages else "swapping"
        if swapping_exists_a_to_b:
            skip_reasons.append(f"{swap_desc} {subject_a}->{subject_b} exists at {swap_dir_a_to_b}")
        if swapping_exists_b_to_a:
            skip_reasons.append(f"{swap_desc} {subject_b}->{subject_a} exists at {swap_dir_b_to_a}")

    should_skip = len(skip_reasons) > 0
    return should_skip, skip_reasons


def _is_subject_trained(config, subject_id) -> bool:

    splatting_cfg = config['pipeline_stages']['10_splatting_avatar']
    avatarrex_dir = config['paths']['avatarrex_output']

    sub_padded = get_padded_subject_id(subject_id)
    model_path = splatting_cfg['model_path_template'].format(subject_id_padded=sub_padded)
    output_dir = os.path.join(avatarrex_dir, 'output-splatting', model_path)

    exists = os.path.exists(output_dir)
    if exists:
        logger.debug(f"Subject {subject_id} is trained (found {output_dir})")
    else:
        logger.debug(f"Subject {subject_id} not trained (missing {output_dir})")

    return exists
