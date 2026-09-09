from enum import Enum
from loguru import logger

from scripts.pipeline.core.constants import DEBUG_PORTS
from scripts.pipeline.core.output_validators import get_validator


class LoopOrder(Enum):

    PAIR_FIRST = "pair_first"
    STAGE_FIRST = "stage_first"


def _get_stage_config_key(config: dict, stage_name: str) -> str:

    pipeline_stages = config.get('pipeline_stages', {})


    for config_key in pipeline_stages.keys():


        key_parts = config_key.split('_', 1)
        if len(key_parts) == 2:
            key_suffix = key_parts[1]

            if stage_name == key_suffix:
                return config_key


    raise KeyError(f"No config key found for stage '{stage_name}'")


def _run_stage(config, stage_definitions, stage_name, subjects, debug_subprocess=False,
               stage_idx=None, total_stages=None, pair_idx=None, total_pairs=None) -> bool:

    if stage_name not in stage_definitions:
        logger.warning(f"Stage '{stage_name}' not found in definitions. Skipping.")
        return False


    try:
        stage_config_key = _get_stage_config_key(config, stage_name)
        stage_config = config['pipeline_stages'].get(stage_config_key, {})
        skip_existing = stage_config.get('skip_existing', False)
    except KeyError as e:

        logger.debug(f"No config found for stage '{stage_name}': {e}")
        skip_existing = False


    if skip_existing:
        validator = get_validator(stage_name)

        if validator is None:

            logger.error(f"❌ skip_existing=True for stage '{stage_name}' but no validator registered!")
            raise RuntimeError(
                f"Cannot skip stage '{stage_name}': no output validator found. "
                f"Either implement a validator or set skip_existing=False in config."
            )

        try:

            subjects_list = subjects if isinstance(subjects, list) else [subjects]


            is_valid, reason = validator.validate(config, subjects_list)

            if is_valid:

                logger.info(f"⏭️  Skipping {stage_name}: {reason}")
                return True
            else:

                logger.debug(f"Output validation failed: {reason}. Running stage.")

        except NotImplementedError:

            logger.error(f"❌ skip_existing=True for stage '{stage_name}' but output validation not implemented!")
            raise RuntimeError(
                f"Cannot skip stage '{stage_name}': output validation not implemented. "
                f"Either implement the validator or set skip_existing=False in config."
            )

    stage_func = stage_definitions[stage_name]


    progress_parts = []
    if stage_idx and total_stages:
        progress_parts.append(f"Stage {stage_idx}/{total_stages}")
    if pair_idx and total_pairs:
        progress_parts.append(f"Pair {pair_idx}/{total_pairs}")

    progress_str = " ".join([f"[{part}]" for part in progress_parts]) if progress_parts else ""
    stage_header = f"{progress_str} EXECUTING: {stage_name.upper()}" if progress_str else f"EXECUTING STAGE: {stage_name.upper()}"

    logger.info(f"\n{'='*20} {stage_header} {'='*20}")

    try:
        if debug_subprocess and stage_name in DEBUG_PORTS:
            stage_func(config, subjects, debug_subprocess=True)
        else:
            stage_func(config, subjects)
        return True
    except Exception as e:
        logger.error(f"Error running stage {stage_name} for subjects {subjects}: {e}")
        return False


def _run_stages(config, stage_definitions, stages, subjects,
                debug_subprocess=False, stop_on_error=True,
                stage_offset=0, total_stages=None, pair_idx=None, total_pairs=None) -> bool:

    all_succeeded = True

    for idx, stage_name in enumerate(stages):

        current_stage_idx = stage_offset + idx + 1 if total_stages else None

        success = _run_stage(
            config, stage_definitions, stage_name, subjects, debug_subprocess,
            stage_idx=current_stage_idx, total_stages=total_stages,
            pair_idx=pair_idx, total_pairs=total_pairs
        )

        if not success:
            all_succeeded = False
            if stop_on_error:
                logger.error(f"Stopping execution due to failure in stage '{stage_name}'")
                return False

    return all_succeeded
