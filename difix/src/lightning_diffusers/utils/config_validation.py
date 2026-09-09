import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import re
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def validate_identity_training_config(config: Union[Dict, DictConfig]) -> Tuple[bool, List[str]]:

    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)

    errors = []


    identity_config = config.get("model", {}).get("identity_training", {})
    if not identity_config.get("enabled", False):
        logger.info("Identity training disabled - skipping validation")
        return True, []

    logger.info("Validating identity training configuration...")


    errors.extend(_validate_alternating_training_config(identity_config))


    errors.extend(_validate_camera_selection_config(identity_config))


    errors.extend(_validate_identity_loss_config(identity_config))


    errors.extend(_validate_head_processing_config(identity_config))


    data_config = config.get("data", {})
    errors.extend(_validate_data_paths_config(data_config, identity_config))


    errors.extend(_validate_validation_config(identity_config, data_config))

    is_valid = len(errors) == 0

    if is_valid:
        logger.info("✅ Identity training configuration is valid")
    else:
        logger.error(f"❌ Identity training configuration has {len(errors)} errors:")
        for error in errors:
            logger.error(f"  - {error}")

    return is_valid, errors


def _validate_alternating_training_config(identity_config: Dict[str, Any]) -> List[str]:

    errors = []


    normal_steps = identity_config.get("normal_steps")
    identity_steps = identity_config.get("identity_steps")

    if normal_steps is None:
        errors.append("normal_steps parameter is required")
    elif not isinstance(normal_steps, int) or normal_steps < 1:
        errors.append("normal_steps must be a positive integer")

    if identity_steps is None:
        errors.append("identity_steps parameter is required")
    elif not isinstance(identity_steps, int) or identity_steps < 1:
        errors.append("identity_steps must be a positive integer")


    if normal_steps is not None and identity_steps is not None:
        total_steps = normal_steps + identity_steps
        if total_steps > 10:
            errors.append(f"Total cycle length {total_steps} seems excessive (normal:{normal_steps} + identity:{identity_steps})")

        identity_ratio = identity_steps / total_steps
        if identity_ratio > 0.7:
            errors.append(f"Identity ratio {identity_ratio:.2f} is very high - consider reducing identity_steps")

    return errors


def _validate_camera_selection_config(identity_config: Dict[str, Any]) -> List[str]:

    errors = []


    forward_view_gt = identity_config.get("forward_view_gt")
    forward_view_swapped = identity_config.get("forward_view_swapped")

    camera_pattern = re.compile(r'^\d{3}_p[+-]\d{3}$')

    if not forward_view_gt:
        errors.append("forward_view_gt is required")
    elif not camera_pattern.match(forward_view_gt):
        errors.append(f"forward_view_gt '{forward_view_gt}' must match format 'XXX_p±XXX' (e.g., '100_p+000')")

    if not forward_view_swapped:
        errors.append("forward_view_swapped is required")
    elif not camera_pattern.match(forward_view_swapped):
        errors.append(f"forward_view_swapped '{forward_view_swapped}' must match format 'XXX_p±XXX' (e.g., '105_p+000')")


    camera_range_gt = identity_config.get("camera_range_gt", [])
    camera_range_swapped = identity_config.get("camera_range_swapped", [])

    if not isinstance(camera_range_gt, list) or len(camera_range_gt) != 2:
        errors.append("camera_range_gt must be a list of 2 integers [before, after]")
    elif not all(isinstance(x, int) and x >= 0 for x in camera_range_gt):
        errors.append("camera_range_gt values must be non-negative integers")

    if not isinstance(camera_range_swapped, list) or len(camera_range_swapped) != 2:
        errors.append("camera_range_swapped must be a list of 2 integers [before, after]")
    elif not all(isinstance(x, int) and x >= 0 for x in camera_range_swapped):
        errors.append("camera_range_swapped values must be non-negative integers")


    filtered_pairs = identity_config.get("filtered_pairs", [])
    if not isinstance(filtered_pairs, list):
        errors.append("filtered_pairs must be a list")
    else:
        for pair in filtered_pairs:
            if not isinstance(pair, str) or ',' not in pair:
                errors.append(f"filtered_pairs entry '{pair}' must be comma-separated string (e.g., '105,110')")

    return errors


def _validate_identity_loss_config(identity_config: Dict[str, Any]) -> List[str]:

    errors = []


    model_path = identity_config.get("model_name_or_path")
    if not model_path:
        errors.append("model_name_or_path is required for CLIP identity loss")
    elif not isinstance(model_path, str):
        errors.append("model_name_or_path must be a string")


    loss_weight = identity_config.get("identity_loss_weight")
    if loss_weight is None:
        errors.append("identity_loss_weight is required")
    elif not isinstance(loss_weight, (int, float)) or loss_weight < 0:
        errors.append("identity_loss_weight must be a non-negative number")
    elif loss_weight > 1.0:
        errors.append("identity_loss_weight > 1.0 may dominate other losses")


    similarity_target = identity_config.get("similarity_target", 0.85)
    if not isinstance(similarity_target, (int, float)) or not 0 <= similarity_target <= 1:
        errors.append("similarity_target must be a number between 0 and 1")


    reduction = identity_config.get("reduction", "mean")
    if reduction not in ["mean", "sum", "none"]:
        errors.append("reduction must be 'mean', 'sum', or 'none'")

    return errors


def _validate_head_processing_config(identity_config: Dict[str, Any]) -> List[str]:

    errors = []


    head_crop_resolution = identity_config.get("head_crop_resolution")
    if head_crop_resolution is None:
        errors.append("head_crop_resolution is required")
    elif not isinstance(head_crop_resolution, int) or head_crop_resolution < 64:
        errors.append("head_crop_resolution must be an integer >= 64")
    elif head_crop_resolution > 512:
        errors.append("head_crop_resolution > 512 may impact performance")


    min_mask_area = identity_config.get("min_mask_area", 1000)
    if not isinstance(min_mask_area, int) or min_mask_area < 100:
        errors.append("min_mask_area must be an integer >= 100")


    padding_ratio = identity_config.get("padding_ratio", 0.1)
    if not isinstance(padding_ratio, (int, float)) or not 0 <= padding_ratio <= 1:
        errors.append("padding_ratio must be a number between 0 and 1")

    return errors


def _validate_data_paths_config(data_config: Dict[str, Any], identity_config: Dict[str, Any]) -> List[str]:

    errors = []


    required_dirs = ["gt_data_dir", "swapped_data_dir", "head_mask_dir"]

    for dir_key in required_dirs:
        dir_path = data_config.get(dir_key)

        if not dir_path:
            errors.append(f"{dir_key} is required for identity training")
            continue

        if not isinstance(dir_path, str):
            errors.append(f"{dir_key} must be a string path")
            continue


        if not Path(dir_path).exists():
            logger.warning(f"{dir_key} path does not exist: {dir_path}")


    camera_config = data_config.get("camera_config", {})
    for key in ["forward_view_gt", "forward_view_swapped", "camera_range_gt", "camera_range_swapped"]:
        data_value = camera_config.get(key)
        identity_value = identity_config.get(key)

        if data_value != identity_value:
            errors.append(f"camera_config.{key} ({data_value}) doesn't match identity_training.{key} ({identity_value})")

    return errors


def _validate_validation_config(identity_config: Dict[str, Any], data_config: Dict[str, Any]) -> List[str]:

    errors = []


    validation_enabled = identity_config.get("identity_validation_enabled", True)
    if not isinstance(validation_enabled, bool):
        errors.append("identity_validation_enabled must be a boolean")


    max_samples = identity_config.get("max_identity_validation_samples", 50)
    if not isinstance(max_samples, int) or max_samples < 1:
        errors.append("max_identity_validation_samples must be a positive integer")
    elif max_samples > 200:
        errors.append("max_identity_validation_samples > 200 may slow validation significantly")


    validation_modes = data_config.get("validation_modes", [])
    expected_modes = ["normal", "swapped", "identity"]

    if not isinstance(validation_modes, list):
        errors.append("validation_modes must be a list")
    else:
        for mode in validation_modes:
            if mode not in expected_modes:
                errors.append(f"Invalid validation mode '{mode}'. Expected: {expected_modes}")

        if validation_enabled and "identity" not in validation_modes:
            errors.append("identity_validation_enabled is True but 'identity' not in validation_modes")

    return errors


def create_identity_training_config_template() -> Dict[str, Any]:

    return {
        "model": {
            "identity_training": {
                "enabled": True,
                "normal_steps": 2,
                "identity_steps": 1,
                "model_name_or_path": "openai/clip-vit-base-patch32",
                "identity_loss_weight": 0.1,
                "similarity_target": 0.85,
                "head_crop_resolution": 224,
                "min_mask_area": 1000,
                "forward_view_gt": "100_p+000",
                "forward_view_swapped": "105_p+000",
                "camera_range_gt": [2, 3],
                "camera_range_swapped": [2, 3],
                "filtered_pairs": [],
                "identity_validation_enabled": True,
                "max_identity_validation_samples": 50
            }
        },
        "data": {
            "gt_data_dir": "/path/to/gt_data",
            "swapped_data_dir": "/path/to/swapped_data",
            "head_mask_dir": "/path/to/head_masks",
            "validation_modes": ["normal", "swapped", "identity"],
            "camera_config": {
                "forward_view_gt": "100_p+000",
                "forward_view_swapped": "105_p+000",
                "camera_range_gt": [2, 3],
                "camera_range_swapped": [2, 3],
                "filtered_pairs": []
            }
        }
    }


def get_config_validation_report(config_path: str) -> Dict[str, Any]:

    try:

        if not Path(config_path).exists():
            return {
                "valid": False,
                "error": f"Configuration file not found: {config_path}",
                "errors": [],
                "warnings": [],
                "summary": {}
            }

        config = OmegaConf.load(config_path)


        is_valid, errors = validate_identity_training_config(config)


        identity_config = OmegaConf.to_container(config.get("model", {}).get("identity_training", {}), resolve=True)
        data_config = OmegaConf.to_container(config.get("data", {}), resolve=True)

        summary = {}
        if identity_config.get("enabled", False):
            summary = {
                "identity_training_enabled": True,
                "alternating_pattern": f"{identity_config.get('normal_steps', '?')}:{identity_config.get('identity_steps', '?')}",
                "forward_views": f"{identity_config.get('forward_view_gt', '?')} → {identity_config.get('forward_view_swapped', '?')}",
                "head_crop_resolution": identity_config.get("head_crop_resolution", "?"),
                "identity_loss_weight": identity_config.get("identity_loss_weight", "?"),
                "validation_modes": data_config.get("validation_modes", [])
            }
        else:
            summary = {"identity_training_enabled": False}

        return {
            "valid": is_valid,
            "errors": errors,
            "warnings": [],
            "summary": summary,
            "config_path": config_path
        }

    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to validate configuration: {e}",
            "errors": [str(e)],
            "warnings": [],
            "summary": {}
        }


if __name__ == "__main__":
    """Command-line validation utility."""
    import sys

    if len(sys.argv) != 2:
        print("Usage: python config_validation.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]
    report = get_config_validation_report(config_path)

    print(f"\n=== Configuration Validation Report ===")
    print(f"Config: {config_path}")
    print(f"Valid: {'✅ Yes' if report['valid'] else '❌ No'}")

    if report.get('error'):
        print(f"Error: {report['error']}")

    if report['errors']:
        print(f"\nErrors ({len(report['errors'])}):")
        for i, error in enumerate(report['errors'], 1):
            print(f"  {i}. {error}")

    if report['warnings']:
        print(f"\nWarnings ({len(report['warnings'])}):")
        for i, warning in enumerate(report['warnings'], 1):
            print(f"  {i}. {warning}")

    print(f"\nSummary:")
    for key, value in report['summary'].items():
        print(f"  {key}: {value}")

    sys.exit(0 if report['valid'] else 1)
