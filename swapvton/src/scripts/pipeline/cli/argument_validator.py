from typing import Optional, Tuple, List
from dataclasses import dataclass
from loguru import logger


@dataclass
class ValidationResult:

    subject_range: Optional[Tuple[int, int]] = None
    pair_filter_indices: Optional[List[int]] = None
    pair_filter_subject_ids: Optional[List[str]] = None


def validate_arguments(args, parser) -> ValidationResult:

    result = ValidationResult()


    result.subject_range = _parse_subject_range(args, parser)


    result.pair_filter_indices = _parse_filter_pairs(args, parser)


    result.pair_filter_subject_ids = _parse_filter_pair_subjects(args, parser)


    _validate_mode_exclusivity(args, parser)


    _validate_generate_pairs_mode(args, parser)
    _validate_load_pairs_mode(args, parser)
    _validate_random_pairs_mode(args, parser)
    _validate_random_trained_pairs_mode(args, parser)

    return result


def _parse_subject_range(args, parser) -> Optional[Tuple[int, int]]:

    if not args.subject_range:
        return None

    try:
        start, end = map(int, args.subject_range.split('-'))
        if start > end:
            parser.error(f"Invalid subject range: {args.subject_range} (start > end)")
        logger.info(f"Subject range set to: [{start}, {end}]")
        return (start, end)
    except ValueError:
        parser.error(
            f"Invalid subject range format: {args.subject_range}. "
            f"Use START-END (e.g., '0-499')"
        )


def _parse_filter_pairs(args, parser) -> Optional[List[int]]:

    if not args.filter_pairs:
        return None

    if not args.load_pairs:
        parser.error("--filter-pairs requires --load-pairs")

    from scripts.pipeline.sampling.pair_persistence import parse_pair_indices
    try:
        indices = parse_pair_indices(args.filter_pairs)
        logger.info(f"Will filter to pair indices: {indices}")
        return indices
    except Exception as e:
        parser.error(f"Invalid --filter-pairs format: {e}")


def _parse_filter_pair_subjects(args, parser) -> Optional[List[str]]:

    if not getattr(args, "filter_pair_subjects", None):
        return None

    if not args.load_pairs:
        parser.error("--filter-pair-subjects requires --load-pairs")

    from scripts.pipeline.sampling.pair_persistence import parse_subject_id_filter

    try:
        subject_ids = parse_subject_id_filter(args.filter_pair_subjects)
        if not subject_ids:
            parser.error("--filter-pair-subjects parsed to empty subject list")
        logger.info(f"Will filter to pairs containing subject IDs: {subject_ids}")
        return subject_ids
    except Exception as e:
        parser.error(f"Invalid --filter-pair-subjects format: {e}")


def _validate_mode_exclusivity(args, parser):

    mode_flags = [
        args.random_pairs,
        args.random_trained_pairs,
        args.generate_pairs,
        args.load_pairs
    ]

    if sum(bool(f) for f in mode_flags) > 1:
        parser.error(
            "Cannot use multiple modes: "
            "--random-pairs, --random-trained-pairs, --generate-pairs, --load-pairs"
        )


def _validate_generate_pairs_mode(args, parser):

    if not args.generate_pairs:
        return

    if not args.num_pairs:
        parser.error("--generate-pairs requires --num-pairs")

    if args.run_stages or args.start_at or args.end_at:
        parser.error(
            "--generate-pairs cannot be used with --run-stages, --start-at, --end-at "
            "(generate only, no execution)"
        )


def _validate_load_pairs_mode(args, parser):


    pass


def _validate_random_pairs_mode(args, parser):

    if not args.random_pairs:
        return

    if args.run_stages or args.start_at or args.end_at or args.random_trained_pairs:
        parser.error(
            '--random-pairs cannot be used with '
            '--run-stages, --start-at, --end-at, or --random-trained-pairs.'
        )


def _validate_random_trained_pairs_mode(args, parser):

    if not args.random_trained_pairs:
        return

    if args.run_stages or args.start_at or args.end_at or args.random_pairs:
        parser.error(
            '--random-trained-pairs cannot be used with '
            '--run-stages, --start-at, --end-at, or --random-pairs.'
        )
