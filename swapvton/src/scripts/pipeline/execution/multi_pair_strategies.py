from loguru import logger
from scripts.pipeline.execution.core_executors import _run_stage, _run_stages, LoopOrder


def _execute_pairs_pair_first(
    config,
    stage_definitions,
    pairs: list[tuple[str, str]],
    single_stages: list[str],
    pair_stages: list[str],
    post_swap_stages: list[str]
) -> dict:

    results = {
        "total_pairs": len(pairs),
        "successful_pairs": 0,
        "failed_pairs": [],
    }


    total_stages = len(single_stages) + len(pair_stages) + len(post_swap_stages)

    for idx, (subject_a, subject_b) in enumerate(pairs, 1):
        logger.info(f"\n{'#'*20} Processing Pair {idx}/{len(pairs)}: [{subject_a}, {subject_b}] {'#'*20}")
        logger.info(f"Pipeline: {len(single_stages)} single → {len(pair_stages)} pair → {len(post_swap_stages)} post-swap stages (total: {total_stages})")


        for subject in [subject_a, subject_b]:
            if not single_stages:
                break
            logger.info(f"Running single-subject stages for {subject}...")
            success = _run_stages(
                config, stage_definitions, single_stages, [subject],
                stop_on_error=True,
                stage_offset=0, total_stages=total_stages,
                pair_idx=idx, total_pairs=len(pairs)
            )
            if not success:
                logger.error(f"Failed single-subject stages for {subject}. Skipping pair.")
                results["failed_pairs"].append((subject_a, subject_b))
                break
        else:

            pass
        if (subject_a, subject_b) in results["failed_pairs"]:
            continue


        logger.info(f"Running pair stages for [{subject_a}, {subject_b}]...")
        pair_success = _run_stages(
            config, stage_definitions, pair_stages, [subject_a, subject_b],
            stop_on_error=True,
            stage_offset=len(single_stages), total_stages=total_stages,
            pair_idx=idx, total_pairs=len(pairs)
        )

        if not pair_success:
            results["failed_pairs"].append((subject_a, subject_b))
            continue


        logger.info(f"Running post-swap stages for [{subject_a}, {subject_b}]...")
        post_success = _run_stages(
            config, stage_definitions, post_swap_stages, [subject_a, subject_b],
            stop_on_error=True,
            stage_offset=len(single_stages) + len(pair_stages), total_stages=total_stages,
            pair_idx=idx, total_pairs=len(pairs)
        )

        if post_success:
            results["successful_pairs"] += 1
            logger.info(f"✓ Successfully completed pair {idx}/{len(pairs)}: [{subject_a}, {subject_b}]")
            logger.info(f"{'#'*60}\n")
        else:
            results["failed_pairs"].append((subject_a, subject_b))

    return results


def _execute_pairs_stage_first(
    config,
    stage_definitions,
    pairs: list[tuple[str, str]],
    single_stages: list[str],
    pair_stages: list[str],
    post_swap_stages: list[str]
) -> dict:

    results = {
        "total_pairs": len(pairs),
        "successful_pairs": 0,
        "failed_pairs": [],
        "stage_stats": {}
    }


    total_stages = len(single_stages) + len(pair_stages) + len(post_swap_stages)


    active_pairs = set(pairs)


    logger.info(f"\n{'='*50}")
    logger.info(f"STAGE GROUP 1: Single-subject training ({len(single_stages)} stages)")
    logger.info(f"{'='*50}")


    all_subjects = set()
    for subject_a, subject_b in pairs:
        all_subjects.add(subject_a)
        all_subjects.add(subject_b)

    logger.info(f"Total unique subjects to train: {len(all_subjects)}")


    for stage_idx, stage in enumerate(single_stages, 1):
        logger.info(f"\n--- Stage {stage_idx}/{total_stages}: {stage.upper()} (for all {len(all_subjects)} subjects) ---")
        results["stage_stats"][stage] = {"success": 0, "failed": 0}

        for subject in all_subjects:

            success = _run_stage(
                config, stage_definitions, stage, [subject],
                stage_idx=stage_idx, total_stages=total_stages
            )

            if success:
                results["stage_stats"][stage]["success"] += 1
            else:
                results["stage_stats"][stage]["failed"] += 1

                pairs_to_fail = [p for p in active_pairs if subject in p]
                for pair in pairs_to_fail:
                    if pair not in results["failed_pairs"]:
                        results["failed_pairs"].append(pair)
                        logger.error(f"✗ Pair {pair} failed due to subject {subject} failure at stage {stage}")
                active_pairs -= set(pairs_to_fail)


        logger.info(f"✓ Stage {stage_idx}/{total_stages} ({stage}) completed: {results['stage_stats'][stage]['success']} success, {results['stage_stats'][stage]['failed']} failed")
        logger.info(f"  Active pairs remaining: {len(active_pairs)}")

    if not active_pairs:
        logger.error("No active pairs remaining after single-subject training. Stopping.")
        return results


    logger.info(f"\n{'='*50}")
    logger.info(f"STAGE GROUP 2: Pair stages ({len(pair_stages)} stages)")
    logger.info(f"Active pairs: {len(active_pairs)}")
    logger.info(f"{'='*50}")

    for stage_idx_offset, stage in enumerate(pair_stages):
        stage_idx = len(single_stages) + stage_idx_offset + 1
        logger.info(f"\n--- Stage {stage_idx}/{total_stages}: {stage.upper()} (for {len(active_pairs)} pairs) ---")
        results["stage_stats"][stage] = {"success": 0, "failed": 0}

        failed_this_stage = []
        for pair_idx, (subject_a, subject_b) in enumerate(list(active_pairs), 1):
            logger.info(f"  → Pair {pair_idx}/{len(active_pairs)}: [{subject_a}, {subject_b}]")

            success = _run_stage(
                config, stage_definitions, stage, [subject_a, subject_b],
                stage_idx=stage_idx, total_stages=total_stages,
                pair_idx=pair_idx, total_pairs=len(active_pairs)
            )

            if success:
                results["stage_stats"][stage]["success"] += 1
            else:
                results["stage_stats"][stage]["failed"] += 1
                failed_this_stage.append((subject_a, subject_b))
                logger.error(f"✗ Pair [{subject_a}, {subject_b}] failed at stage {stage}")


        for pair in failed_this_stage:
            active_pairs.discard(pair)
            if pair not in results["failed_pairs"]:
                results["failed_pairs"].append(pair)


        logger.info(f"✓ Stage {stage_idx}/{total_stages} ({stage}) completed: {results['stage_stats'][stage]['success']} success, {results['stage_stats'][stage]['failed']} failed")
        logger.info(f"  Active pairs remaining: {len(active_pairs)}")

        if not active_pairs:
            logger.error(f"No active pairs remaining after stage {stage}. Stopping.")
            return results


    logger.info(f"\n{'='*50}")
    logger.info(f"STAGE GROUP 3: Post-swap stages ({len(post_swap_stages)} stages)")
    logger.info(f"Active pairs: {len(active_pairs)}")
    logger.info(f"{'='*50}")

    for stage_idx_offset, stage in enumerate(post_swap_stages):
        stage_idx = len(single_stages) + len(pair_stages) + stage_idx_offset + 1
        logger.info(f"\n--- Stage {stage_idx}/{total_stages}: {stage.upper()} (for {len(active_pairs)} pairs) ---")
        results["stage_stats"][stage] = {"success": 0, "failed": 0}

        failed_this_stage = []
        for pair_idx, (subject_a, subject_b) in enumerate(list(active_pairs), 1):
            logger.info(f"  → Pair {pair_idx}/{len(active_pairs)}: [{subject_a}, {subject_b}]")
            success = _run_stage(
                config, stage_definitions, stage, [subject_a, subject_b],
                stage_idx=stage_idx, total_stages=total_stages,
                pair_idx=pair_idx, total_pairs=len(active_pairs)
            )

            if success:
                results["stage_stats"][stage]["success"] += 1
            else:
                results["stage_stats"][stage]["failed"] += 1
                failed_this_stage.append((subject_a, subject_b))

        for pair in failed_this_stage:
            active_pairs.discard(pair)
            if pair not in results["failed_pairs"]:
                results["failed_pairs"].append(pair)


        logger.info(f"✓ Stage {stage_idx}/{total_stages} ({stage}) completed: {results['stage_stats'][stage]['success']} success, {results['stage_stats'][stage]['failed']} failed")
        logger.info(f"  Active pairs remaining: {len(active_pairs)}")


    results["successful_pairs"] = len(active_pairs)

    logger.info(f"\n{'='*50}")
    logger.info(f"Stage-first execution complete:")
    logger.info(f"  ✓ Successful pairs: {results['successful_pairs']}/{results['total_pairs']}")
    logger.info(f"  ✗ Failed pairs: {len(results['failed_pairs'])}")
    logger.info(f"{'='*50}")

    return results


def _execute_pairs_with_strategy(
    config,
    stage_definitions,
    pairs: list[tuple[str, str]],
    single_stages: list[str],
    pair_stages: list[str],
    post_swap_stages: list[str],
    loop_order: LoopOrder = LoopOrder.PAIR_FIRST
) -> dict:

    logger.info(f"\n{'~'*60}")
    logger.info(f"Execution strategy: {loop_order.value}")
    logger.info(f"  Total pairs to process: {len(pairs)}")
    logger.info(f"  Single-subject stages: {len(single_stages)}")
    logger.info(f"  Pair stages: {len(pair_stages)}")
    logger.info(f"  Post-swap stages: {len(post_swap_stages)}")
    print(f"{'='*60}")

    if loop_order == LoopOrder.PAIR_FIRST:
        return _execute_pairs_pair_first(
            config, stage_definitions, pairs,
            single_stages, pair_stages, post_swap_stages
        )
    elif loop_order == LoopOrder.STAGE_FIRST:
        return _execute_pairs_stage_first(
            config, stage_definitions, pairs,
            single_stages, pair_stages, post_swap_stages
        )
    else:
        raise ValueError(f"Unknown loop order: {loop_order}")
