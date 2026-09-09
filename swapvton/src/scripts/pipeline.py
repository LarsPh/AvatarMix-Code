import argparse
import os
from loguru import logger
from typing import Optional
import random
import sys
from pathlib import Path
from enum import Enum


_SCRIPTS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPTS_DIR.parent
_REPO_ROOT = _SRC_DIR.parent

for _p in (str(_REPO_ROOT), str(_SRC_DIR), str(_SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


from pipeline.core.constants import DEBUG_PORTS
from pipeline.core.config import load_config


from pipeline.sampling.subject_discovery import (
    get_subjects, discover_trained_subjects,
)
from pipeline.cli import setup_argument_parser, validate_arguments, process_arguments

from pipeline.stages.data_preparation import stage_render_thuman, stage_convert_to_avatarrex
from pipeline.stages.first_swap_preparation import (
        stage_convert_to_neus2, stage_neus2_train, stage_copy_neus2_mesh, stage_clean_mesh,
        stage_4d_dress_parsing, stage_smplx_fitting, stage_hands_transplant, stage_process_mesh, stage_robust_lbs_transfer, stage_splatting_avatar
    )
from pipeline.stages.first_swap import stage_reposing, stage_swapping, stage_head_donator_reposing, stage_body_donator_reshaping, stage_direct_swapping
from pipeline.stages.stage_cloth_fit_reshaping import stage_cloth_fit_reshaping
from pipeline.stages.first_swap_rendering import stage_render_swapped_gaussians, stage_render_swapped_gt_head_aligned, stage_refine_rendered_images, stage_extract_head_masks
from pipeline.stages.segment_swapped_renders import stage_segment_swapped_renders
from pipeline.stages.segment_restored_renders import stage_segment_restored_renders
from pipeline.stages.second_swap_prearation import (
    stage_convert_swapped_to_neus2, stage_neus2_train_swapped, stage_copy_neus2_mesh_swapped, stage_clean_mesh_swapped,
    stage_4d_dress_parsing_swapped, stage_process_mesh_swapped, stage_lbs_transfer_swapped, stage_splatting_avatar_swapped
)
from pipeline.stages.second_swap import stage_swapped_head_donator_reposing, stage_swapped_body_donator_reshaping, stage_swap_back
from pipeline.stages.second_swap_rendering import stage_render_swapped_back_gaussians, stage_refine_swapped_back_images
from pipeline.stages.combined_refinement_preparation.training_preparation import stage_create_combined_portraits
from pipeline.stages.combined_refinement_preparation.render_body_donator import stage_render_body_donator_full_gt_aligned
from pipeline.stages.combined_refinement_preparation.testing_preparation import stage_create_validation_data_portraits
from pipeline.stages.combined_refinement_preparation.portrait_segmentation import stage_refine_portrait_body_segmentation
from pipeline.stages.stage_difix_refinement import stage_difix_refinement
from pipeline.stages.stage_splatting_avatar_gs_finetune import (
    stage_splatting_avatar_free_gaussian,
    stage_splatting_avatar_gs_finetune,
)
from pipeline.core.pipeline_builders import (
    _get_pair_stages, _get_full_random_pair_pipeline, _build_default_pipeline, _ensure_swap_back_prerequisites,
    _build_training_pipeline, _build_testing_pipeline, _build_legacy_pipeline, _filter_multi_subject_stages, _auto_configure_gt_aligned_head
)
from pipeline.execution.core_executors import _run_stage, _run_stages, LoopOrder
from pipeline.sampling.pair_sampling import _sample_subject_pair, _sample_trained_pair, _check_pair_swapping_outputs_exist, _is_subject_trained
from pipeline.execution.multi_pair_strategies import _execute_pairs_with_strategy, _execute_pairs_pair_first, _execute_pairs_stage_first


logger.remove()

log_level = os.environ.get("LOG_LEVEL", "DEBUG")

logger.add(sys.stderr, level=log_level, format="<green>{time:MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<green>{function}</green>:<blue>{line}</blue> - <level>{message}</level>")


def swap_random_trained_pairs(config, stage_definitions):

    num_pairs = config['random_trained_pairs_n']


    loop_order_str = config.get('execution', {}).get('loop_order', 'stage_first')
    loop_order = LoopOrder(loop_order_str)


    pair_stages, mode_desc = _get_pair_stages(config)
    logger.info(f"--- Running {mode_desc} for {num_pairs} random trained pairs ---")
    logger.info(f"Stages to run: {pair_stages}")


    trained_subjects = discover_trained_subjects(config)
    if len(trained_subjects) < 2:
        raise ValueError(f"Need at least 2 trained subjects for pairing, found {len(trained_subjects)}")


    pairs = []
    sampled_pairs_set = set()
    max_retries = min(50, len(trained_subjects) * (len(trained_subjects) - 1))
    retries = 0

    while len(pairs) < num_pairs and retries < max_retries:
        retries += 1


        subject_a, subject_b = _sample_trained_pair(config, trained_subjects)


        pair_key = tuple(sorted([subject_a, subject_b]))
        if pair_key in sampled_pairs_set:
            continue


        should_skip, skip_reasons = _check_pair_swapping_outputs_exist(config, subject_a, subject_b, pair_stages)
        if should_skip:
            logger.debug(f"Pair [{subject_a}, {subject_b}] outputs exist, skipping")
            sampled_pairs_set.add(pair_key)
            continue


        pairs.append((subject_a, subject_b))
        sampled_pairs_set.add(pair_key)
        logger.info(f"Sampled pair {len(pairs)}/{num_pairs}: [{subject_a}, {subject_b}]")

    if len(pairs) < num_pairs:
        logger.warning(f"Only found {len(pairs)}/{num_pairs} unique pairs without existing outputs")

    if not pairs:
        logger.error("No pairs to process after skip-existing filtering. Exiting.")
        return

    logger.info(f"Final sampled pairs: {pairs}")


    results = _execute_pairs_with_strategy(
        config=config,
        stage_definitions=stage_definitions,
        pairs=pairs,
        single_stages=[],
        pair_stages=pair_stages,
        post_swap_stages=[],
        loop_order=loop_order
    )


    logger.info(f"\n{'='*60}")
    logger.info(f"Random trained pairs execution completed!")
    logger.info(f"  ✓ Successful: {results['successful_pairs']}/{len(pairs)} pairs")
    if results['failed_pairs']:
        logger.warning(f"  ✗ Failed pairs: {results['failed_pairs']}")
    logger.info(f"{'='*60}")

def run_random_pairs(config, stage_definitions, subject_file: Optional[str] = None):

    num_pairs = config['random_pairs_n']


    loop_order_str = config.get('execution', {}).get('loop_order', '')
    loop_order = LoopOrder(loop_order_str)


    single_stages, pair_stages, post_swap_stages = _get_full_random_pair_pipeline(config)

    logger.info(f"--- Running FULL pipeline for {num_pairs} random pairs ---")
    logger.info(f"Single-subject stages: {single_stages}")
    logger.info(f"Pair stages: {pair_stages}")
    logger.info(f"Post-swap stages: {post_swap_stages}")


    sampling_strategy = config.get("pair_sampling", {}).get("strategy", "without_replacement")
    data_type = config.get("data_type", "thuman2")

    if subject_file and sampling_strategy == "without_replacement":
        from scripts.pipeline.execution.pair_workflows import parse_subject_list_from_file
        from scripts.pipeline.sampling.pair_persistence import generate_pairs_without_replacement

        subject_ids = parse_subject_list_from_file(subject_file)
        pairs = generate_pairs_without_replacement(
            num_pairs=num_pairs,
            subject_range=None,
            subject_list=subject_ids,
            data_type=data_type,
            deduplicate=config.get("pair_sampling", {}).get("deduplicate_symmetric", True),
        )
    elif subject_file:
        from scripts.pipeline.execution.pair_workflows import _sample_pair_from_subject_file
        pairs = [_sample_pair_from_subject_file(config=config, subject_file=subject_file) for _ in range(num_pairs)]
    else:
        pairs = [_sample_subject_pair(config) for _ in range(num_pairs)]
    logger.info(f"Sampled pairs: {pairs}")


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
    logger.info(f"Random pairs execution completed!")
    logger.info(f"  ✓ Successful: {results['successful_pairs']}/{results['total_pairs']} pairs")
    if results['failed_pairs']:
        logger.warning(f"  ✗ Failed pairs: {results['failed_pairs']}")
    logger.info(f"{'='*60}")


def main():

    stage_definitions = {

        "render": stage_render_thuman,
        "convert_to_avatarrex": stage_convert_to_avatarrex,

        "convert_to_neus2": stage_convert_to_neus2,
        "neus2_train": stage_neus2_train,
        "copy_mesh": stage_copy_neus2_mesh,
        "clean_mesh": stage_clean_mesh,
        "splatting_avatar": stage_splatting_avatar,
        "smplx_fitting": stage_smplx_fitting,
        "hands_transplant": stage_hands_transplant,
        "parse_mesh": stage_4d_dress_parsing,
        "lbs_transfer": stage_robust_lbs_transfer,
        "process_mesh": stage_process_mesh,

        "reposing": stage_reposing,
        "swapping": stage_swapping,
        "body_donator_reshaping": stage_body_donator_reshaping,


        "head_donator_reposing": stage_head_donator_reposing,

        "cloth_fit_reshaping": stage_cloth_fit_reshaping,
        "direct_swapping": stage_direct_swapping,

        "render_swapped_gaussians": stage_render_swapped_gaussians,

        "render_swapped_gt_head_aligned": stage_render_swapped_gt_head_aligned,
        "refine_rendered_images": stage_refine_rendered_images,
        "extract_head_masks": stage_extract_head_masks,


        "convert_swapped_to_neus2": stage_convert_swapped_to_neus2,
        "neus2_train_swapped": stage_neus2_train_swapped,
        "copy_mesh_swapped": stage_copy_neus2_mesh_swapped,
        "clean_mesh_swapped": stage_clean_mesh_swapped,
        "parse_mesh_swapped": stage_4d_dress_parsing_swapped,
        "process_mesh_swapped": stage_process_mesh_swapped,
        "lbs_transfer_swapped": stage_lbs_transfer_swapped,
        "splatting_avatar_swapped": stage_splatting_avatar_swapped,

        "swapped_head_donator_reposing": stage_swapped_head_donator_reposing,
        "swapped_body_donator_reshaping": stage_swapped_body_donator_reshaping,
        "swap_back": stage_swap_back,

        "render_swapped_back_gaussians": stage_render_swapped_back_gaussians,
        "refine_swapped_back_images": stage_refine_swapped_back_images,


        "create_combined_portraits": stage_create_combined_portraits,
        "render_body_full_gt_aligned": stage_render_body_donator_full_gt_aligned,
        "segment_swapped_renders": stage_segment_swapped_renders,
        "segment_restored_renders": stage_segment_restored_renders,

        "create_validation_data_portraits": stage_create_validation_data_portraits,

        "refine_portrait_body_segmentation": stage_refine_portrait_body_segmentation,


        "difix_refinement": stage_difix_refinement,


        "splatting_avatar_free_gaussian": stage_splatting_avatar_free_gaussian,

        "splatting_avatar_gs_finetune": stage_splatting_avatar_gs_finetune,
    }
    stage_order = list(stage_definitions.keys())


    parser = setup_argument_parser(stage_order)
    args = parser.parse_args()

    config = load_config(args.config)


    validated = validate_arguments(args, parser)
    config = process_arguments(args, config, validated)


    if args.generate_pairs:

        from scripts.pipeline.execution.pair_workflows import generate_pairs_workflow
        generate_pairs_workflow(
            config=config,
            num_pairs=args.num_pairs,
            output_path=args.generate_pairs,
            subject_range=validated.subject_range,
            subject_file=getattr(args, 'subject_file', None),
            description=args.pairs_description,
            force_overwrite=args.force_overwrite
        )
        return


    if args.load_pairs:
        from scripts.pipeline.execution.pair_workflows import run_loaded_pairs_workflow
        run_loaded_pairs_workflow(
            config=config,
            stage_definitions=stage_definitions,
            pairs_file=args.load_pairs,
            pair_filter_indices=validated.pair_filter_indices,
            pair_filter_subject_ids=validated.pair_filter_subject_ids,
            pipeline_mode=args.pipeline_mode,
            start_at=args.start_at,
            end_at=args.end_at,
            direct_swap=args.direct_swap,
            legacy_reposed_swap=not args.direct_swap,
            run_stages=[s.strip() for s in args.run_stages.split(',')] if args.run_stages else None,
            debug_subprocess=args.debug_subprocess
        )
        return


    if args.random_pairs:
        config['random_pairs_n'] = args.random_pairs

        if args.save_pairs:
            from scripts.pipeline.execution.pair_workflows import run_random_pairs_with_save
            run_random_pairs_with_save(
                config=config,
                stage_definitions=stage_definitions,
                save_path=args.save_pairs,
                subject_range=validated.subject_range,
                subject_file=getattr(args, 'subject_file', None),
                force_overwrite=args.force_overwrite
            )
        else:
            run_random_pairs(config, stage_definitions, subject_file=getattr(args, 'subject_file', None))
        return


    if args.random_trained_pairs:
        config['random_trained_pairs_n'] = args.random_trained_pairs
        swap_random_trained_pairs(config, stage_definitions)
        return

    subjects = get_subjects(args, config)


    if args.run_stages:


        stages_to_run = [s.strip() for s in args.run_stages.split(',')]
        for stage in stages_to_run:
            if stage not in stage_definitions:
                parser.error(f"Invalid stage '{stage}' provided to --run-stages.")

    elif args.start_at or args.end_at:


        selected_mode = args.pipeline_mode or "testing"

        if args.direct_swap:
            if selected_mode == "training":
                full_pipeline = _build_training_pipeline(config)
            elif selected_mode == "testing":
                full_pipeline = _build_testing_pipeline(config)
            else:
                parser.error(f"Invalid pipeline mode: {selected_mode}")
        else:
            if selected_mode == "testing":
                logger.warning(
                    "Testing pipeline is not defined for legacy reposing+swapping; "
                    "falling back to LEGACY training pipeline for --start-at/--end-at slicing."
                )
            full_pipeline = _build_legacy_pipeline(config)


        if args.start_at and args.start_at not in full_pipeline:
            parser.error(f"--start-at stage '{args.start_at}' not found in pipeline.")
        if args.end_at and args.end_at not in full_pipeline:
            parser.error(f"--end-at stage '{args.end_at}' not found in pipeline.")

        start_index = full_pipeline.index(args.start_at) if args.start_at else 0
        end_index = full_pipeline.index(args.end_at) if args.end_at else len(full_pipeline) - 1

        if start_index > end_index:
            parser.error("--start-at stage must come before or be the same as --end-at stage.")

        stages_to_run = full_pipeline[start_index:end_index+1]

    elif args.pipeline_mode:

        if args.pipeline_mode == 'training':
            stages_to_run = _build_training_pipeline(config)
            logger.info("Using TRAINING pipeline")
        elif args.pipeline_mode == 'testing':
            stages_to_run = _build_testing_pipeline(config)
            logger.info("Using TESTING pipeline")
        else:
            parser.error(f"Invalid pipeline mode: {args.pipeline_mode}")

    else:

        if args.direct_swap:
            stages_to_run = _build_testing_pipeline(config)
            logger.info("Using DEFAULT testing pipeline")
        else:
            stages_to_run = _build_legacy_pipeline(config)
            logger.info("Using LEGACY training pipeline with reposing+swapping")

    logger.info(f"Pipeline will run for subjects: {subjects}")
    logger.info(f"Executing stages: {', '.join(stages_to_run)}")


    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)
    stages_to_run = _ensure_swap_back_prerequisites(stages_to_run, enable_reshape)
    stages_to_run = _filter_multi_subject_stages(stages_to_run, subjects)

    if not stages_to_run:
        logger.error("No stages to run after filtering. Exiting.")
        return


    _auto_configure_gt_aligned_head(config, stages_to_run, args)


    success = _run_stages(
        config=config,
        stage_definitions=stage_definitions,
        stages=stages_to_run,
        subjects=subjects,
        debug_subprocess=args.debug_subprocess,
        stop_on_error=True
    )

    if success:
        logger.info(f"\n--- Pipeline execution completed successfully for stages: {', '.join(stages_to_run)} ---")
    else:
        logger.error(f"\n--- Pipeline execution failed ---")

if __name__ == '__main__':
    main()
