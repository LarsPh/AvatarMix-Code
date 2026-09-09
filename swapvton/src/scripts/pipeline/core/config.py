import os
from pathlib import Path
from string import Template
import yaml
from loguru import logger


def load_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    release_root = Path(__file__).resolve().parents[5]
    variables = {
        "AVATARMIX_ROOT": str(release_root),
        "AVATARMIX_DATA_ROOT": str(release_root / "data"),
        "AVATARMIX_ASSET_ROOT": str(release_root / "external_assets"),
        **os.environ,
    }

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            return Template(value).safe_substitute(variables)
        return value

    return resolve(config)


def use_cloth_fit_reshaped_gs(config, stage_key=None):


    stages = config.get("pipeline_stages", {}) or {}
    cloth_fit_stage_cfg = stages.get('11c_cloth_fit_reshaping', {}) or {}
    if stage_key:
        stage_cfg = stages.get(stage_key, {}) or {}
        stage_value = stage_cfg.get('use_cloth_fit_reshaped_gs')
        if stage_value is not None:

            suffix = stage_cfg.get('cloth_fit_suffix', None)
            if suffix is None:
                suffix = cloth_fit_stage_cfg.get('cloth_fit_suffix', '')
            return stage_value, suffix


    return config.get('use_cloth_fit_reshaped_gs', False), cloth_fit_stage_cfg.get('cloth_fit_suffix', '')


def use_swap_hands(config: dict, stage_key: str | None = None) -> bool:

    stages = (config or {}).get("pipeline_stages", {}) or {}
    if stage_key:
        st = stages.get(stage_key, {}) or {}
        v = st.get("swap_hands", None)
        if v is not None:
            return bool(v)


    for k in ("12b_direct_swapping", "12_swapping", "12_direct_swapping"):
        st = stages.get(k, None)
        if isinstance(st, dict) and bool(st.get("swap_hands", False)):
            return True

    return False


def use_label_no_sam_for_alignment(config, stage_key=None):


    if stage_key:
        stage_cfg = config['pipeline_stages'].get(stage_key, {})
        stage_value = stage_cfg.get('use_label_no_sam')
        if stage_value is not None:
            return stage_value


    return config.get('use_label_no_sam_for_head_alignment', False)


def get_refinement_render_mode(config: dict, stage_key: str | None = None) -> str:


    if stage_key:
        stage_cfg = config.get("pipeline_stages", {}).get(stage_key, {})
        stage_value = stage_cfg.get("render_mode")
        if stage_value is not None:
            mode = str(stage_value).strip().lower()
            return mode


    mode = str(config.get("refinement", {}).get("render_mode", "")).strip().lower()
    if not mode:

        mode = str(config.get("refinement_render_mode", "normal")).strip().lower()

    if mode not in ("normal", "head_aligned"):
        logger.warning(f"Unknown refinement.render_mode='{mode}', falling back to 'normal'")
        return "normal"
    return mode


def validate_refinement_config(config):

    from ..sampling.subject_discovery import get_reshape_suffix_from_config

    logger.info("Validating refinement configuration...")


    reposing_cfg = config['pipeline_stages']['11_reposing']
    enable_reshape = reposing_cfg.get('enable_body_reshape', False)

    if enable_reshape:
        reshape_suffix = get_reshape_suffix_from_config(config)
        logger.info(f"Body reshaping enabled - expecting directories with suffix: '{reshape_suffix}'")


        avatarrex_dir = config['paths']['avatarrex_output']
        swapped_base_dir = os.path.join(avatarrex_dir, "swapped")

        if os.path.exists(swapped_base_dir):

            existing_dirs = [d for d in os.listdir(swapped_base_dir)
                           if os.path.isdir(os.path.join(swapped_base_dir, d)) and d.startswith('A')]

            base_dirs = [d for d in existing_dirs if not any(suffix in d for suffix in ['_reshaped_', '_reposed_'])]
            suffixed_dirs = [d for d in existing_dirs if '_reshaped_' in d or '_reposed_' in d]

            if base_dirs and suffixed_dirs:
                logger.warning(f"WARNING: Found both base and suffixed directories in {swapped_base_dir}")
                logger.warning(f"   Base directories: {len(base_dirs)}")
                logger.warning(f"   Suffixed directories: {len(suffixed_dirs)}")
                logger.warning(f"   This may cause refinement to process unexpected directories")
    else:
        logger.info("Body reshaping disabled - expecting base directory names without suffixes")


    required_paths = ['difix3d_project', 'splatting_avatar_project']
    for path_key in required_paths:
        if path_key not in config['paths']:
            logger.warning(f"WARNING: Missing path configuration: {path_key}")
        elif not os.path.exists(config['paths'][path_key]):
            logger.warning(f"WARNING: Path does not exist: {config['paths'][path_key]}")

    logger.info("Configuration validation completed.")
