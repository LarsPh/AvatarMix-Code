from typing import Dict, Optional, Tuple
from loguru import logger


def process_arguments(args, config: Dict, validation_result) -> Dict:


    if args.direct_swap:
        config = _process_direct_swap_mode(config)


    if args.loop_order:
        config = _process_loop_order(args.loop_order, config)


    if getattr(args, "pair_direction", None):
        config = _process_pair_direction(args.pair_direction, config)


    if validation_result.subject_range:
        config = _process_subject_range(validation_result.subject_range, config)


    config = _process_execution_mode(args, config)

    return config


def _process_direct_swap_mode(config: Dict) -> Dict:

    config['_direct_swap_mode'] = True


    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)
    if not enable_reshape:
        logger.info(
            "Direct swap without body reshaping is enabled"
        )


    logger.info("Direct swap mode enabled")
    return config


def _process_loop_order(loop_order: str, config: Dict) -> Dict:

    if 'execution' not in config:
        config['execution'] = {}

    config['execution']['loop_order'] = loop_order.replace('-', '_')
    logger.info(f"Loop order overridden from CLI: {loop_order}")

    return config


def _process_pair_direction(pair_direction: str, config: Dict) -> Dict:

    if 'execution' not in config:
        config['execution'] = {}
    config['execution']['pair_direction'] = pair_direction
    logger.info(f"Pair direction overridden from CLI: {pair_direction}")
    return config


def _process_subject_range(subject_range: Tuple[int, int], config: Dict) -> Dict:

    start, end = subject_range

    if 'pair_sampling' not in config:
        config['pair_sampling'] = {}

    config['pair_sampling']['subject_range'] = [start, end]

    return config


def _process_execution_mode(args, config: Dict) -> Dict:

    if args.random_pairs:
        config['_execution_mode'] = 'random_pairs'
    elif args.random_trained_pairs:
        config['_execution_mode'] = 'random_trained_pairs'


    return config
