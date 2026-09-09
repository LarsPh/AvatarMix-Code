import argparse
from typing import List


def setup_argument_parser(stage_order: List[str]) -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="AvatarMix pipeline. By default, it runs the entire configured pipeline. Use flags to control execution.",
        formatter_class=argparse.RawTextHelpFormatter,
    )


    _add_basic_arguments(parser)


    _add_stage_control_arguments(parser, stage_order)


    _add_execution_mode_arguments(parser)


    _add_pipeline_mode_arguments(parser)


    _add_direct_swap_arguments(parser)


    _add_pair_persistence_arguments(parser)


    _add_debug_arguments(parser)

    return parser


def _add_basic_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the pipeline config file."
    )
    parser.add_argument(
        "--subjects",
        type=str,
        help='Comma-separated list of subject IDs to process (e.g., "70,83"). Overrides config file.',
    )


def _add_stage_control_arguments(
    parser: argparse.ArgumentParser, stage_order: List[str]
):

    parser.add_argument(
        "--run-stages",
        type=str,
        help='A comma-separated list of specific stages to run (e.g., "reposing,swapping").',
    )
    parser.add_argument(
        "--start-at",
        choices=stage_order,
        help="The stage to start execution from (inclusive).",
    )
    parser.add_argument(
        "--end-at",
        choices=stage_order,
        help="The stage to end execution at (inclusive).",
    )


def _add_execution_mode_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--random-pairs",
        type=int,
        metavar="N",
        help="Run the full pipeline for N random pairs of subjects.",
    )
    parser.add_argument(
        "--random-trained-pairs",
        type=int,
        metavar="N",
        help="Sample N pairs from already-trained subjects and run only reposing + swapping stages.",
    )
    parser.add_argument(
        "--pair-direction",
        type=str,
        choices=["a_to_b", "b_to_a", "both"],
        default=None,
        help="Pair direction mode for 2-subject stages that support it (default: both).\n"
        "  a_to_b: only process subjects[0] -> subjects[1]\n"
        "  b_to_a: only process subjects[1] -> subjects[0]\n"
        "  both:  process both directions\n"
        "This is currently used by direction-aware stages (e.g. cloth_fit_reshaping).",
    )
    parser.add_argument(
        "--loop-order",
        type=str,
        choices=["pair-first", "stage-first"],
        default=None,
        help="Multi-pair execution loop order (default: pair-first):\n"
        "  pair-first:  Process each pair completely before next pair\n"
        "  stage-first: Process all pairs through each stage together\n"
        "Overrides config file setting if specified.",
    )


def _add_pipeline_mode_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["training", "testing"],
        default=None,
        help="Select pipeline mode:\n"
        "  training: Full training pipeline (stages 1-29, 35 stages total)\n"
        "            - Data prep + first/second swap + swap-back + training data generation\n"
        "  testing:  Testing pipeline (stages 1-14a, 26-28, 30, 18 stages total)\n"
        "            - Data prep + first swap + validation data generation\n"
        "            - Skips second swap preparation and swap-back (stages 15-25)\n"
        "Default: training (if not specified)\n"
        "Note: --run-stages takes precedence over --pipeline-mode",
    )


def _add_direct_swap_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--direct-swap",
        action="store_true",
        default=True,
        help="Skip reposing and directly swap A's head onto B's body in B's original pose.",
    )
    parser.add_argument(
        "--legacy-reposed-swap",
        dest="direct_swap",
        action="store_false",
        help="Skip direct swap and use reposing + swapping instead.",
    )
    parser.add_argument(
        "--prefer-reshaped-gs",
        action="store_true",
        help="Prefer reshaped Gaussians over original when available in direct swap mode.",
    )
    parser.add_argument(
        "--force-original-gs",
        action="store_true",
        help="Force use of original Gaussians, ignore reshaped versions in direct swap mode.",
    )


def _add_pair_persistence_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--generate-pairs",
        type=str,
        metavar="FILEPATH",
        help="Generate pairs file and save to FILEPATH (just filename → uses config pairs_dir). "
        "Exits after generation, does not run pipeline.",
    )
    parser.add_argument(
        "--load-pairs",
        type=str,
        metavar="FILEPATH",
        help="Load pairs from FILEPATH and run pipeline. "
        "Overrides --random-pairs and subjects from config.",
    )
    parser.add_argument(
        "--subject-range",
        type=str,
        metavar="START-END",
        help='Subject range for sampling (inclusive). Example: "0-499" for first 500 subjects. '
        "Overrides config pair_sampling.subject_range if specified.",
    )
    parser.add_argument(
        "--subject-file",
        type=str,
        metavar="FILEPATH",
        help="Path to comma-separated text file with subject IDs for pair generation. "
        "Example: vton360_train_subjects.txt with contents '0004,0005,0007,...,0525'. "
        "Overrides --subject-range if specified.",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        metavar="N",
        help="Number of pairs to generate (used with --generate-pairs). "
        "For without_replacement: must satisfy N*2 <= available subjects (or N <= available for odd counts).",
    )
    parser.add_argument(
        "--pairs-description",
        type=str,
        help="Description for generated pairs file metadata.",
    )
    parser.add_argument(
        "--force-overwrite",
        action="store_true",
        help="Overwrite existing pairs file without prompting.",
    )
    parser.add_argument(
        "--filter-pairs",
        type=str,
        metavar="INDICES",
        help='Load only specific pairs from file. Format: "0,5,10-15" '
        "(pair_id values, comma-separated or ranges). "
        "Must be used with --load-pairs.",
    )
    parser.add_argument(
        "--filter-pair-subjects",
        type=str,
        metavar="SUBJECT_IDS",
        help='Load only pairs that contain any of the given subject IDs. '
        'Format: "70,83" (comma-separated subject IDs). '
        'For purely-numeric IDs, ranges are allowed: "0-10,70,83". '
        "Must be used with --load-pairs.",
    )
    parser.add_argument(
        "--save-pairs",
        type=str,
        metavar="FILEPATH",
        help="Save sampled pairs to FILEPATH after random sampling. "
        "Used with --random-pairs to persist generated pairs.",
    )


def _add_debug_arguments(parser: argparse.ArgumentParser):

    parser.add_argument(
        "--debug_subprocess",
        action="store_true",
        help="Enable debugpy for subprocess debugging (starts debugpy server on port 5678)",
    )
