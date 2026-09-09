from typing import List, Tuple, Dict, Optional
from pathlib import Path
from loguru import logger
import random

from scripts.pipeline.sampling.pair_persistence import (
    resolve_pairs_filepath,
    generate_pairs_without_replacement,
    save_pairs_to_yaml,
    load_pairs_from_yaml,
    validate_pairs_file,
    filter_pairs_by_indices,
    resolve_pair_indices_by_subject_ids,
)
from scripts.pipeline.sampling.pair_sampling import _sample_subject_pair
from scripts.pipeline.core.pipeline_builders import _get_full_random_pair_pipeline
from scripts.pipeline.execution.multi_pair_strategies import _execute_pairs_with_strategy
from scripts.pipeline.execution.core_executors import LoopOrder


def parse_subject_list_from_file(subject_file: str) -> List[str]:

    filepath = Path(subject_file)

    if not filepath.exists():
        raise FileNotFoundError(f"Subject file not found: {subject_file}")


    with open(filepath, 'r') as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"Subject file is empty: {subject_file}")


    subject_ids = [s.strip() for s in content.split(',')]

    if not subject_ids:
        raise ValueError(f"No subjects found in file: {subject_file}")

    logger.info(f"Loaded {len(subject_ids)} subjects from {subject_file}")

    return subject_ids


def _sample_pair_from_subject_file(
    config: Dict,
    subject_file: str,
) -> Tuple[str, str]:

    data_type = config.get("data_type", "thuman2")
    subject_ids = parse_subject_list_from_file(subject_file)

    if data_type == "mvhumannet":
        subj_to_frames: Dict[str, List[str]] = {}
        for sid in subject_ids:
            if "_" not in sid:
                raise ValueError(
                    f"MVHumanNet subject_file must contain subject_frame IDs like '100001_1185', got: {sid}"
                )
            subj, frame = sid.split("_", 1)
            subj_to_frames.setdefault(subj, []).append(frame)

        subjects = list(subj_to_frames.keys())
        if len(subjects) < 2:
            raise ValueError(
                f"Need at least 2 unique subjects in MVHumanNet subject_file for pairing, found {len(subjects)}"
            )

        subj_a, subj_b = random.sample(subjects, 2)
        frame_a = random.choice(subj_to_frames[subj_a])
        frame_b = random.choice(subj_to_frames[subj_b])
        return f"{subj_a}_{frame_a}", f"{subj_b}_{frame_b}"


    if len(subject_ids) < 2:
        raise ValueError(f"Need at least 2 subjects in subject_file for pairing, found {len(subject_ids)}")
    return tuple(random.sample(subject_ids, 2))  # type: ignore[return-value]


def generate_pairs_workflow(
    config: Dict,
    num_pairs: int,
    output_path: str,
    subject_range: Optional[Tuple[int, int]] = None,
    subject_file: Optional[str] = None,
    description: Optional[str] = None,
    force_overwrite: bool = False
) -> None:

    logger.info(f"=== GENERATE PAIRS MODE ===")


    filepath = resolve_pairs_filepath(output_path, config)


    data_type = config.get('data_type', 'thuman2')
    subject_list = None
    actual_subject_range = subject_range

    if subject_file:

        logger.info(f"Loading subjects from file: {subject_file}")
        subject_list = parse_subject_list_from_file(subject_file)
        available = len(subject_list)
        logger.info(f"Loaded {available} subjects from file")
    elif subject_range:

        start, end = subject_range
        available = end - start + 1
        actual_subject_range = (start, end)
    else:

        config_range = config.get('pair_sampling', {}).get('subject_range')
        if config_range:
            start, end = config_range
            available = end - start + 1
            actual_subject_range = (start, end)
        else:

            if data_type == 'thuman2':
                available = 526
                actual_subject_range = (0, 525)
            elif data_type == 'actorshq':
                available = 8
                actual_subject_range = (1, 8)
            else:
                raise ValueError(f"Unknown data_type for capacity calculation: {data_type}")


    is_odd = available % 2 == 1
    if is_odd:

        max_pairs = available
    else:

        max_pairs = available // 2

    if num_pairs > max_pairs:
        raise ValueError(
            f"Cannot generate {num_pairs} pairs from {available} subjects. "
            f"Maximum pairs: {max_pairs}"
        )


    logger.info(f"Generating {num_pairs} pairs from {available} subjects...")
    if subject_list:
        logger.info(f"Using subject list ({len(subject_list)} subjects)")
    else:
        logger.info(f"Using subject range {actual_subject_range}")

    pairs = generate_pairs_without_replacement(
        num_pairs=num_pairs,
        subject_range=actual_subject_range if subject_list is None else None,
        subject_list=subject_list,
        data_type=data_type,
        deduplicate=config.get('pair_sampling', {}).get('deduplicate_symmetric', True)
    )


    save_pairs_to_yaml(
        pairs=pairs,
        filepath=filepath,
        config=config,
        description=description,
        force_overwrite=force_overwrite
    )

    logger.info(f"[OK] Generated {len(pairs)} pairs")
    logger.info(f"[OK] Saved to: {filepath}")
    if subject_file:
        logger.info(f"  Subject file: {subject_file}")
    else:
        logger.info(f"  Subject range: {actual_subject_range}")
    logger.info(f"  Dataset: {data_type}")
    if is_odd:
        logger.info(f"  Note: {available} is odd - last subject may be duplicated in pairs")


def run_loaded_pairs_workflow(
    config: Dict,
    stage_definitions: Dict,
    pairs_file: str,
    pair_filter_indices: Optional[List[int]] = None,
    pair_filter_subject_ids: Optional[List[str]] = None,
    pipeline_mode: Optional[str] = None,
    start_at: Optional[str] = None,
    end_at: Optional[str] = None,
    direct_swap: bool = True,
    legacy_reposed_swap: bool = False,
    run_stages: Optional[List[str]] = None,
    debug_subprocess: bool = False
) -> None:

    logger.info(f"=== LOAD PAIRS MODE ===")


    filepath = resolve_pairs_filepath(pairs_file, config, create_dir=False)


    is_valid, errors = validate_pairs_file(filepath, config)
    if not is_valid:
        logger.error(f"Invalid pairs file: {filepath}")
        for err in errors:
            logger.error(f"  - {err}")
        raise ValueError("Pairs file validation failed")


    pairs, metadata = load_pairs_from_yaml(filepath)
    logger.info(f"✓ Loaded {len(pairs)} pairs from {filepath}")
    logger.info(f"  Dataset: {metadata.get('dataset_type')}")
    logger.info(f"  Created: {metadata.get('created_at')}")
    if metadata.get('description'):
        logger.info(f"  Description: {metadata.get('description')}")


    if pair_filter_indices or pair_filter_subject_ids:
        selected = set(range(len(pairs)))

        if pair_filter_indices:
            selected &= set(pair_filter_indices)

        if pair_filter_subject_ids:
            subject_indices = resolve_pair_indices_by_subject_ids(pairs, pair_filter_subject_ids)
            selected &= set(subject_indices)

        selected_indices = sorted(selected)
        pairs = filter_pairs_by_indices(pairs, selected_indices) if selected_indices else []

        details = []
        if pair_filter_indices:
            details.append(f"indices={pair_filter_indices}")
        if pair_filter_subject_ids:
            details.append(f"subject_ids={pair_filter_subject_ids}")
        logger.info(f"✓ Filtered to {len(pairs)} pairs ({', '.join(details)})")
        logger.info(f"  Resolved pair indices: {selected_indices}")
        if not pairs:
            logger.error("No pairs to process after filtering. Exiting.")
            return


    from scripts.pipeline.core.pipeline_builders import (
        _build_testing_pipeline,
        _build_training_pipeline,
        _build_legacy_pipeline,
        _get_first_swap_stages,
    )

    selected_mode = (pipeline_mode or "testing").strip().lower()
    if selected_mode not in {"testing", "training"}:
        raise ValueError(f"Invalid pipeline_mode: {pipeline_mode}")

    if legacy_reposed_swap or not direct_swap:
        if selected_mode == "testing":
            logger.warning(
                "Testing pipeline is not defined for legacy reposing+swapping; "
                "falling back to LEGACY training pipeline for --load-pairs."
            )
        full_pipeline = _build_legacy_pipeline(config)
    else:
        full_pipeline = (
            _build_training_pipeline(config)
            if selected_mode == "training"
            else _build_testing_pipeline(config)
        )


    if start_at and start_at not in full_pipeline:
        raise ValueError(f"--start-at stage '{start_at}' not found in selected {selected_mode} pipeline")
    if end_at and end_at not in full_pipeline:
        raise ValueError(f"--end-at stage '{end_at}' not found in selected {selected_mode} pipeline")

    start_idx = full_pipeline.index(start_at) if start_at else 0
    end_idx = full_pipeline.index(end_at) if end_at else len(full_pipeline) - 1
    if start_idx > end_idx:
        raise ValueError("--start-at stage must come before or be the same as --end-at stage.")

    stages_to_run = full_pipeline[start_idx : end_idx + 1]

    if run_stages:


        stages_to_run = run_stages
        logger.info(f"Running user-specified stages: {stages_to_run}")

        single_stages = []
        pair_stages = stages_to_run
        post_swap_stages = []
    else:

        enable_reshape = config["pipeline_stages"]["11_reposing"].get("enable_body_reshape", False)
        pair_full = _get_first_swap_stages(legacy_mode=False, enable_legacy_reshape=enable_reshape)
        if not pair_full:
            raise RuntimeError("Internal error: first-swap stage list is empty")

        try:
            pair_start = full_pipeline.index(pair_full[0])
        except ValueError:

            pair_start = len(full_pipeline)
        pair_end = pair_start + len(pair_full)

        def _stage_group(stage_name: str) -> str:


            idx = full_pipeline.index(stage_name)
            if idx < pair_start:
                return "single"
            if pair_start <= idx < pair_end:
                return "pair"
            return "post"

        single_stages = [s for s in stages_to_run if _stage_group(s) == "single"]
        pair_stages = [s for s in stages_to_run if _stage_group(s) == "pair"]
        post_swap_stages = [s for s in stages_to_run if _stage_group(s) == "post"]

        logger.info(f"Selected pipeline mode: {selected_mode}")
        if start_at or end_at:
            logger.info(f"Applied stage slice: start_at={start_at} end_at={end_at}")
        logger.info(
            f"Running stages (grouped): {len(single_stages)} single → {len(pair_stages)} pair → "
            f"{len(post_swap_stages)} post"
        )


    loop_order_str = config.get('execution', {}).get('loop_order', 'stage_first')
    loop_order = LoopOrder(loop_order_str)


    results = _execute_pairs_with_strategy(
        config=config,
        stage_definitions=stage_definitions,
        pairs=pairs,
        single_stages=single_stages,
        pair_stages=pair_stages,
        post_swap_stages=post_swap_stages,
        loop_order=loop_order
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Loaded pairs execution completed!")
    logger.info(f"  ✓ Successful: {results['successful_pairs']}/{len(pairs)} pairs")
    if results['failed_pairs']:
        logger.warning(f"  ✗ Failed pairs: {results['failed_pairs']}")
    logger.info(f"{'='*60}")


def run_random_pairs_with_save(
    config: Dict,
    stage_definitions: Dict,
    save_path: str,
    subject_range: Optional[Tuple[int, int]] = None,
    subject_file: Optional[str] = None,
    force_overwrite: bool = False
) -> None:

    num_pairs = config['random_pairs_n']

    logger.info(f"=== RANDOM PAIRS WITH SAVE MODE ===")
    logger.info(f"Sampling {num_pairs} pairs...")


    sampling_strategy = config.get("pair_sampling", {}).get("strategy", "without_replacement")
    data_type = config.get("data_type", "thuman2")

    if subject_file and sampling_strategy == "without_replacement":


        subject_ids = parse_subject_list_from_file(subject_file)
        pairs = generate_pairs_without_replacement(
            num_pairs=num_pairs,
            subject_range=None,
            subject_list=subject_ids,
            data_type=data_type,
            deduplicate=config.get("pair_sampling", {}).get("deduplicate_symmetric", True),
        )
    else:

        pairs = []
        for _ in range(num_pairs):
            if subject_file:
                pair = _sample_pair_from_subject_file(config=config, subject_file=subject_file)
            else:
                pair = _sample_subject_pair(config, subject_range=subject_range)
            pairs.append(pair)

    logger.info(f"Sampled {len(pairs)} pairs: {pairs}")


    filepath = resolve_pairs_filepath(save_path, config)
    save_pairs_to_yaml(
        pairs=pairs,
        filepath=filepath,
        config=config,
        description=f"Auto-saved from --random-pairs {num_pairs}",
        force_overwrite=force_overwrite
    )
    logger.info(f"✓ Saved pairs to: {filepath}")


    single_stages, pair_stages, post_swap_stages = _get_full_random_pair_pipeline(config)
    loop_order = LoopOrder(config.get('execution', {}).get('loop_order', 'stage_first'))

    logger.info(f"Running full pipeline for {len(pairs)} pairs...")

    results = _execute_pairs_with_strategy(
        config=config,
        stage_definitions=stage_definitions,
        pairs=pairs,
        single_stages=single_stages,
        pair_stages=pair_stages,
        post_swap_stages=post_swap_stages,
        loop_order=loop_order
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"Random pairs with save completed!")
    logger.info(f"  ✓ Pairs saved to: {filepath}")
    logger.info(f"  ✓ Successful: {results['successful_pairs']}/{results['total_pairs']} pairs")
    if results['failed_pairs']:
        logger.warning(f"  ✗ Failed pairs: {results['failed_pairs']}")
    logger.info(f"{'='*60}")
