import os
import random
import re
import yaml
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from loguru import logger


def generate_pairs_without_replacement(
    num_pairs: int,
    subject_range: Optional[Tuple[int, int]] = None,
    data_type: str = 'thuman2',
    deduplicate: bool = True,
    subject_list: Optional[List[str]] = None
) -> List[Tuple[str, str]]:


    if subject_list is not None:

        available_subjects = [str(s) for s in subject_list]
    elif subject_range is not None:

        start, end = subject_range

        if data_type == 'thuman2':

            available_subjects = [str(i) for i in range(start, end + 1)]

        elif data_type == 'actorshq':

            actor_start = int(re.search(r'\d+', str(start)).group())
            actor_end = int(re.search(r'\d+', str(end)).group())
            available_subjects = [f"Actor{i:02d}" for i in range(actor_start, actor_end + 1)]
        else:
            raise ValueError(f"Unsupported data_type for pair generation: {data_type}")
    else:
        raise ValueError("Either subject_range or subject_list must be provided")

    available = len(available_subjects)
    is_odd = available % 2 == 1

    if is_odd:

        pairing_count = num_pairs
        if pairing_count > available:
            raise ValueError(
                f"Cannot generate {num_pairs} pairs from {available} subjects. "
                f"Maximum pairs: {available}"
            )


        num_to_pair = available - 1
        num_pairs_without_dup = num_to_pair // 2

        if pairing_count < num_pairs_without_dup:

            sampled_subjects = random.sample(available_subjects, pairing_count * 2)
            pairs = [
                (sampled_subjects[i], sampled_subjects[i + 1])
                for i in range(0, len(sampled_subjects), 2)
            ]
        elif pairing_count == num_pairs_without_dup + 1:


            pairing_subjects = available_subjects[:-1]
            sampled_subjects = random.sample(pairing_subjects, num_to_pair)
            pairs = [
                (sampled_subjects[i], sampled_subjects[i + 1])
                for i in range(0, len(sampled_subjects), 2)
            ]

            odd_subject = available_subjects[-1]
            random_partner = random.choice(available_subjects)

            if random_partner == odd_subject and len(available_subjects) > 1:

                available_partners = [s for s in available_subjects if s != odd_subject]
                random_partner = random.choice(available_partners)
            pairs.append((odd_subject, random_partner))
        else:
            raise ValueError(
                f"Cannot generate {pairing_count} pairs without duplication from {available} subjects. "
                f"Maximum pairs without duplication: {num_pairs_without_dup + 1} "
                f"(first {num_to_pair} without duplication, last odd subject paired with random)"
            )
    else:

        if num_pairs * 2 > available:
            raise ValueError(
                f"Cannot generate {num_pairs} pairs without replacement from {available} subjects. "
                f"Maximum pairs: {available // 2}"
            )


        sampled_subjects = random.sample(available_subjects, num_pairs * 2)
        pairs = [
            (sampled_subjects[i], sampled_subjects[i + 1])
            for i in range(0, len(sampled_subjects), 2)
        ]


    if deduplicate:
        pairs = deduplicate_pairs(pairs)

    return pairs


def deduplicate_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:

    seen = set()
    deduplicated = []

    for pair in pairs:

        normalized = tuple(sorted([str(pair[0]), str(pair[1])]))

        if normalized not in seen:
            deduplicated.append(pair)
            seen.add(normalized)

    return deduplicated


def save_pairs_to_yaml(
    pairs: List[Tuple[str, str]],
    filepath: str,
    config: Dict,
    description: Optional[str] = None,
    force_overwrite: bool = False
) -> None:


    if os.path.exists(filepath) and not force_overwrite:
        raise FileExistsError(
            f"Pairs file already exists: {filepath}\n"
            f"Use --force-overwrite to overwrite."
        )


    os.makedirs(os.path.dirname(filepath), exist_ok=True)


    data_type = config.get('data_type', 'unknown')
    subject_range = config.get('pair_sampling', {}).get('subject_range')

    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset_type': data_type,
        'subject_range': subject_range,
        'total_subjects': len(set([p[0] for p in pairs] + [p[1] for p in pairs])),
        'num_pairs': len(pairs),
        'sampling_strategy': config.get('pair_sampling', {}).get('strategy', 'without_replacement'),
        'description': description or '',
        'config_snapshot': {
            'data_type': data_type,
            'direct_swap': config.get('_direct_swap_mode', True),
            'enable_reshape': config.get('pipeline_stages', {}).get('11_reposing', {}).get('enable_body_reshape', False),
        }
    }


    pairs_data = [
        {
            'pair_id': idx,
            'subject_a': str(pair[0]),
            'subject_b': str(pair[1])
        }
        for idx, pair in enumerate(pairs)
    ]


    output = {
        'metadata': metadata,
        'pairs': pairs_data
    }


    class ConsistentDumper(yaml.SafeDumper):

        pass

    def str_representer(dumper, data):


        if data and (data[0] == '0' or data.isdigit()):
            return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
        return dumper.represent_scalar('tag:yaml.org,2002:str', data)

    ConsistentDumper.add_representer(str, str_representer)

    with open(filepath, 'w') as f:
        yaml.dump(output, f, Dumper=ConsistentDumper, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved {len(pairs)} pairs to: {filepath}")


def load_pairs_from_yaml(filepath: str) -> Tuple[List[Tuple[str, str]], Dict]:

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Pairs file not found: {filepath}")

    with open(filepath) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}")


    if 'metadata' not in data or 'pairs' not in data:
        raise ValueError("Invalid pairs file: missing 'metadata' or 'pairs' sections")


    pairs = [
        (pair['subject_a'], pair['subject_b'])
        for pair in data['pairs']
    ]

    metadata = data['metadata']

    return pairs, metadata


def validate_pairs_file(filepath: str, config: Dict) -> Tuple[bool, List[str]]:

    errors = []


    if not os.path.exists(filepath):
        return False, [f"File not found: {filepath}"]


    try:
        with open(filepath) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        return False, [f"Invalid YAML: {e}"]


    if 'metadata' not in data or 'pairs' not in data:
        errors.append("Missing required sections: metadata or pairs")
        return False, errors


    file_data_type = data.get('metadata', {}).get('dataset_type')
    config_data_type = config.get('data_type')
    if file_data_type != config_data_type:
        errors.append(
            f"Dataset type mismatch: file={file_data_type}, config={config_data_type}"
        )


    pairs = data.get('pairs', [])
    data_type = file_data_type or config_data_type

    if data_type == 'thuman2':

        valid_range = range(526)
        for pair in pairs:
            for subj in [pair['subject_a'], pair['subject_b']]:
                try:
                    if int(subj) not in valid_range:
                        errors.append(f"Subject {subj} out of valid range [0-525]")
                except ValueError:
                    errors.append(f"Invalid subject ID format: {subj}")

    elif data_type == 'actorshq':

        for pair in pairs:
            for subj in [pair['subject_a'], pair['subject_b']]:
                if not re.match(r'Actor\d{2}', subj):
                    errors.append(f"Invalid ActorHQ subject format: {subj}")


    if config.get('pair_sampling', {}).get('deduplicate_symmetric', True):
        seen = set()
        for pair in pairs:
            a, b = pair['subject_a'], pair['subject_b']
            pair_key = tuple(sorted([a, b]))
            if pair_key in seen:
                errors.append(f"Duplicate pair detected: ({a}, {b})")
            seen.add(pair_key)

    is_valid = len(errors) == 0
    return is_valid, errors


def parse_pair_indices(indices_str: str) -> List[int]:

    indices = []

    for part in indices_str.split(','):
        part = part.strip()
        if '-' in part:

            start, end = map(int, part.split('-'))
            indices.extend(range(start, end + 1))
        else:

            indices.append(int(part))

    return sorted(set(indices))


def parse_subject_id_filter(subject_ids_str: str) -> List[str]:

    if subject_ids_str is None:
        return []

    subject_ids: List[str] = []
    for part in subject_ids_str.split(","):
        part = part.strip()
        if not part:
            continue

        if "-" in part:

            start_s, end_s = (p.strip() for p in part.split("-", 1))
            if start_s.isdigit() and end_s.isdigit():
                start_i, end_i = int(start_s), int(end_s)
                step = 1 if start_i <= end_i else -1
                subject_ids.extend([str(i) for i in range(start_i, end_i + step, step)])
                continue

        subject_ids.append(part)


    seen = set()
    out: List[str] = []
    for sid in subject_ids:
        if sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def resolve_pair_indices_by_subject_ids(
    pairs: List[Tuple[str, str]],
    subject_ids: List[str],
) -> List[int]:

    if not subject_ids:
        return []

    def _matches(filter_sid: str, candidate_sid: str) -> bool:
        fs = str(filter_sid)
        cs = str(candidate_sid)

        if fs.isdigit() and cs.isdigit():
            try:
                return int(fs) == int(cs)
            except Exception:
                return fs == cs

        if "_" not in fs and "_" in cs:
            prefix = cs.split("_", 1)[0]
            if prefix == fs:
                return True

        return cs == fs

    selected: List[int] = []
    for idx, (a, b) in enumerate(pairs):
        for sid in subject_ids:
            if _matches(sid, a) or _matches(sid, b):
                selected.append(idx)
                break

    return sorted(set(selected))


def filter_pairs_by_indices(
    pairs: List[Tuple[str, str]],
    pair_indices: List[int]
) -> List[Tuple[str, str]]:

    max_idx = len(pairs) - 1

    for idx in pair_indices:
        if idx < 0 or idx > max_idx:
            raise IndexError(
                f"Pair index {idx} out of range [0-{max_idx}]. "
                f"File contains {len(pairs)} pairs."
            )

    return [pairs[idx] for idx in pair_indices]


def resolve_pairs_filepath(
    pairs_name: str,
    config: Dict,
    create_dir: bool = True
) -> str:


    if os.path.isabs(pairs_name):
        return pairs_name


    pairs_dir = config.get('pair_sampling', {}).get('pairs_dir', 'pairs')
    logger.debug(f"Pairs directory: {pairs_dir}")


    data_type = config.get('data_type', 'unknown')


    if not pairs_name.startswith(f"{data_type}_"):
        pairs_name = f"{data_type}_{pairs_name}"


    if not pairs_name.endswith('.yaml'):
        pairs_name = f"{pairs_name}.yaml"


    filepath = os.path.join(pairs_dir, pairs_name)


    if create_dir:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    return filepath
