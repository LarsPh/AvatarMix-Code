from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import yaml
from loguru import logger


def load_subject_pairs_from_yaml(
    yaml_path: str,
    exclude_pairs: Optional[List[Tuple[str, str]]] = None,
    include_only_pairs: Optional[List[Tuple[str, str]]] = None,
    exclude_pair_ids: Optional[List[Union[int, List[int]]]] = None,
    include_only_pair_ids: Optional[List[Union[int, List[int]]]] = None,
) -> Tuple[List[Tuple[str, str]], Dict[Tuple[str, str], int]]:


    has_include_pairs = include_only_pairs is not None
    has_include_ids = include_only_pair_ids is not None
    has_exclude_pairs = exclude_pairs is not None
    has_exclude_ids = exclude_pair_ids is not None


    if (has_include_pairs or has_include_ids) and (has_exclude_pairs or has_exclude_ids):
        raise ValueError(
            "Include-only filtering (include_only_pairs, include_only_pair_ids) "
            "and exclude filtering (exclude_pairs, exclude_pair_ids) are mutually exclusive. "
            "Use whitelist mode OR blacklist mode, not both."
        )


    if has_include_pairs and has_include_ids:
        raise ValueError(
            "include_only_pairs and include_only_pair_ids are mutually exclusive. "
            "Use one whitelist method only."
        )


    if include_only_pair_ids:

        requested_ids = set()
        for item in include_only_pair_ids:
            if isinstance(item, int):
                requested_ids.add(item)
            elif (isinstance(item, (list, tuple)) or isinstance(list(item), list)) and len(item) == 2:

                start, end = item
                requested_ids.update(range(start, end + 1))
            else:
                logger.warning(f"Invalid include_only_pair_ids entry: {item}, skipping")

        logger.info(f"Include-only mode (by ID): filtering for {len(requested_ids)} pair IDs")


        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Pairs YAML file not found: {yaml_path}")

        logger.info(f"Loading subject pairs from: {yaml_path}")

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if 'pairs' not in data:
            raise ValueError(f"Invalid YAML format: missing 'pairs' key in {yaml_path}")


        filtered_pairs = []
        pair_to_id = {}
        found_ids = set()

        for pair_dict in data['pairs']:
            if 'subject_a' not in pair_dict or 'subject_b' not in pair_dict:
                logger.warning(f"Skipping invalid pair entry (missing subject_a/b): {pair_dict}")
                continue

            pair_id = pair_dict.get('pair_id')
            if pair_id is not None and pair_id in requested_ids:
                subject_a = str(pair_dict['subject_a'])
                subject_b = str(pair_dict['subject_b'])
                pair = (subject_a, subject_b)
                filtered_pairs.append(pair)
                pair_to_id[pair] = pair_id
                found_ids.add(pair_id)


        invalid_ids = requested_ids - found_ids
        if invalid_ids:
            logger.warning(
                f"Invalid pair IDs (not found in YAML): {sorted(invalid_ids)[:10]}"
                f"{' ...' if len(invalid_ids) > 10 else ''} "
                f"({len(invalid_ids)} total)"
            )

        logger.info(f"Loaded {len(filtered_pairs)} pairs (from {len(found_ids)} requested IDs)")

        if 'metadata' in data:
            logger.info(f"YAML metadata: {data['metadata']}")

        return filtered_pairs, pair_to_id


    if include_only_pairs:
        logger.info(f"Whitelist mode (by pairs): using {len(include_only_pairs)} specified pairs")
        pair_to_id = {pair: i for i, pair in enumerate(include_only_pairs)}
        return include_only_pairs, pair_to_id


    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Pairs YAML file not found: {yaml_path}")

    logger.info(f"Loading subject pairs from: {yaml_path}")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    if 'pairs' not in data:
        raise ValueError(f"Invalid YAML format: missing 'pairs' key in {yaml_path}")


    excluded_ids = set()
    if exclude_pair_ids:
        for item in exclude_pair_ids:
            if isinstance(item, int):
                excluded_ids.add(item)
            elif isinstance(item, (list, tuple)) or isinstance(list(item), list) and len(item) == 2:

                start, end = item
                excluded_ids.update(range(start, end + 1))
            else:
                logger.warning(f"Invalid exclude_pair_ids entry: {item}, skipping")

        if excluded_ids:
            logger.info(f"Excluding {len(excluded_ids)} pair IDs: {sorted(excluded_ids)[:10]}...")


    excluded_pairs_set = set()
    if exclude_pairs:
        excluded_pairs_set = {tuple(pair) for pair in exclude_pairs}
        logger.info(f"Excluding {len(excluded_pairs_set)} specific pairs")


    filtered_pairs = []
    pair_to_id = {}

    for pair_dict in data['pairs']:
        if 'subject_a' not in pair_dict or 'subject_b' not in pair_dict:
            logger.warning(f"Skipping invalid pair entry (missing subject_a/b): {pair_dict}")
            continue

        pair_id = pair_dict.get('pair_id', None)
        subject_a = str(pair_dict['subject_a'])
        subject_b = str(pair_dict['subject_b'])
        pair = (subject_a, subject_b)


        if pair_id is not None and pair_id in excluded_ids:
            logger.debug(f"Filtered out pair_id {pair_id}: {pair}")
            continue


        if pair in excluded_pairs_set:
            logger.debug(f"Filtered out pair: {pair}")
            continue

        filtered_pairs.append(pair)
        if pair_id is not None:
            pair_to_id[pair] = pair_id

    logger.info(f"Loaded {len(filtered_pairs)} subject pairs from {yaml_path}")

    if 'metadata' in data:
        logger.info(f"YAML metadata: {data['metadata']}")


    total_pairs = len(data['pairs'])
    filtered_count = total_pairs - len(filtered_pairs)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} pairs ({filtered_count/total_pairs*100:.1f}%)")

    return filtered_pairs, pair_to_id
