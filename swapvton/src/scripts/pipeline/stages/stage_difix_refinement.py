from __future__ import annotations

from loguru import logger
from pathlib import Path
from typing import List

from pipeline.execution.subprocess_runner import run_command
from pipeline.sampling.subject_discovery import get_padded_subject_id


def stage_difix_refinement(config: dict, subjects: List[str], debug_subprocess: bool = False) -> None:
    logger.info("\n--- Stage 32: DiFix Refinement (Inference) ---")

    cfg = config["pipeline_stages"].get("32_difix_refinement", {})
    if not cfg.get("enabled", False):
        logger.info("DiFix refinement disabled in config - skipping stage")
        return

    difix_root = Path(config["paths"]["difix3d_project"])
    avatarrex_root = Path(config["paths"]["avatarrex_output"])
    seamfixed_test_dir = Path(config["paths"]["seamfixed_test_dir"])

    experiment = cfg.get("experiment", None)
    ckpt_path = cfg.get("ckpt_path", None)
    if not experiment:
        raise ValueError("32_difix_refinement requires `experiment` (Hydra experiment name)")
    if not ckpt_path:
        raise ValueError("32_difix_refinement requires `ckpt_path` (checkpoint for test/inference)")


    st13 = (config.get("pipeline_stages", {}).get("13_render_swapped_gaussians", {}) or {})
    st13_output_base = str(st13.get("output_base", "head_swapped_renders"))
    validation_data_root = cfg.get("validation_data_root", str(avatarrex_root / st13_output_base))


    degraded_dir = cfg.get("degraded_dir", validation_data_root)
    gt_dataset_root = cfg.get("gt_dataset_root", str(avatarrex_root))


    run_both = cfg.get("run_both_directions", True)
    directions = [
        (subjects[0], subjects[1]),
        (subjects[1], subjects[0]),
    ] if run_both else [(subjects[0], subjects[1])]


    refinement_mode = cfg.get("refinement_mode", "full")
    use_head_alignment = bool(cfg.get("use_head_alignment", False))
    mask_source = cfg.get("mask_source", None)
    adjust_calibration = bool(cfg.get("adjust_calibration", False))
    first_swap_dir_filter = cfg.get("first_swap_dir_filter", None)
    test_camera_filter = cfg.get("test_camera_filter", None)


    fullbody_target_resolution = cfg.get("fullbody_target_resolution", None)
    fullbody_min_bbox_size = cfg.get("fullbody_min_bbox_size", None)
    fullbody_target_resolution_order = cfg.get("fullbody_target_resolution_order", None)


    canonical_only = bool(cfg.get("canonical_only", cfg.get("canonical_front_only", False)))
    canonical_fullbody_target_resolution = cfg.get("canonical_fullbody_target_resolution", None)
    if canonical_only:

        canonical_root_cfg = cfg.get("canonical_input_root", cfg.get("canonical_front_input_root", None))
        if canonical_root_cfg is None:
            st13 = (config.get("pipeline_stages", {}).get("13_render_swapped_gaussians", {}) or {})
            can_cfg = st13.get("canonical", None)
            if can_cfg is None:
                can_cfg = st13.get("canonical_front", {}) or {}
            canonical_base = (can_cfg or {}).get("output_base", "canonical_renders")
            canonical_root = avatarrex_root / canonical_base
        else:
            canonical_root = Path(str(canonical_root_cfg))
            if not canonical_root.is_absolute():
                canonical_root = avatarrex_root / canonical_root

        validation_data_root = str(canonical_root)
        degraded_dir = str(canonical_root)

        if test_camera_filter is None:
            st13 = (config.get("pipeline_stages", {}).get("13_render_swapped_gaussians", {}) or {})
            can_cfg = st13.get("canonical", None)
            if can_cfg is None:
                can_cfg = st13.get("canonical_front", {}) or {}
            views = (can_cfg or {}).get("views", None)
            if isinstance(views, list) and views:
                test_camera_filter = [str(v.get("id", "front")) for v in views]
            else:
                test_camera_filter = ["front"]


        def _round_to_8(x: int) -> int:
            return int(((int(x) + 7) // 8) * 8)

        if canonical_fullbody_target_resolution is not None:
            fullbody_target_resolution = canonical_fullbody_target_resolution
        else:

            st13 = (config.get("pipeline_stages", {}).get("13_render_swapped_gaussians", {}) or {})
            can_cfg = st13.get("canonical", None)
            if can_cfg is None:
                can_cfg = st13.get("canonical_front", {}) or {}
            out_wh = (can_cfg or {}).get("output_wh", None)
            if isinstance(out_wh, (list, tuple)) and len(out_wh) == 2:
                w_in, h_in = int(out_wh[0]), int(out_wh[1])

                fullbody_target_resolution = [_round_to_8(w_in), _round_to_8(h_in)]

                if fullbody_target_resolution_order is None:
                    fullbody_target_resolution_order = "wh"

    extra_overrides = cfg.get("extra_overrides", []) or []
    if not isinstance(extra_overrides, list):
        raise ValueError("32_difix_refinement.extra_overrides must be a list of Hydra override strings")


    test_output_root_cfg = cfg.get("test_output_root", None)
    if canonical_only:

        test_output_root_cfg = cfg.get("canonical_test_output_root", cfg.get("canonical_front_test_output_root", None))
    if test_output_root_cfg is None:
        test_output_root = seamfixed_test_dir
    else:
        test_output_root = Path(str(test_output_root_cfg))
        if not test_output_root.is_absolute():
            test_output_root = avatarrex_root / test_output_root
    test_output_root.mkdir(parents=True, exist_ok=True)


    def _hydra_list_override(v) -> str:

        if v is None:
            return ""
        if isinstance(v, (list, tuple)):

            if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                return f"[{int(v[0])},{int(v[1])}]"

            if all(isinstance(x, str) for x in v):
                return "[" + ",".join(v) + "]"
        return str(v)

    for head_id_raw, body_id_raw in directions:
        head_id = get_padded_subject_id(head_id_raw)
        body_id = get_padded_subject_id(body_id_raw)

        logger.info(f"Running DiFix inference for swapped direction: head={head_id} body={body_id}")
        logger.info(f"  DiFix experiment: {experiment}")
        logger.info(f"  validation_data_root: {validation_data_root}")
        logger.info(f"  test_output_root: {test_output_root}")
        logger.info(f"  refinement_mode: {refinement_mode}")
        logger.info(f"  use_head_alignment: {use_head_alignment}")
        logger.info(f"  adjust_calibration: {adjust_calibration}")
        logger.info(f"  mask_source: {mask_source}")
        if canonical_only:
            logger.info("  canonical_only: true")
        if fullbody_target_resolution is not None:
            logger.info(f"  fullbody_target_resolution: {fullbody_target_resolution}")
        if fullbody_min_bbox_size is not None:
            logger.info(f"  fullbody_min_bbox_size: {fullbody_min_bbox_size}")

        cmd = [
            "uv", "run", "python", "-m", "scripts.train_swapfix_lora",
            f"experiment={experiment}",
            "train=false",
            "test=true",
            "predict=false",
            f"ckpt_path={ckpt_path}",
            f"data.refinement_mode={refinement_mode}",
            f"data.degraded_dir={degraded_dir}",
            f"data.gt_dataset_root={gt_dataset_root}",
            f"data.validation_data_root={validation_data_root}",
            f"data.test_output_root={str(test_output_root)}",
            f"data.subject_a={head_id}",
            f"data.subject_b={body_id}",

            "data.train_pairs_yaml=null",
            "data.val_pairs_yaml=null",
            "data.test_pairs_yaml=null",

            f"data.use_head_alignment={str(use_head_alignment).lower()}",
            f"data.adjust_calibration={str(adjust_calibration).lower()}",
        ]

        if mask_source is not None:
            cmd.append(f"data.mask_source={mask_source}")
        if first_swap_dir_filter is not None:
            cmd.append(f"data.first_swap_dir_filter={first_swap_dir_filter}")
        if test_camera_filter is not None:

            cmd.append(f"data.test_camera_filter={_hydra_list_override(test_camera_filter)}")
        if fullbody_target_resolution is not None:
            cmd.append(f"data.fullbody_target_resolution={_hydra_list_override(fullbody_target_resolution)}")
        if fullbody_min_bbox_size is not None:
            cmd.append(f"data.fullbody_min_bbox_size={_hydra_list_override(fullbody_min_bbox_size)}")
        if fullbody_target_resolution_order is not None:
            cmd.append(f"data.fullbody_target_resolution_order={fullbody_target_resolution_order}")

        cmd.extend(extra_overrides)


        run_command(cmd, cwd=str(difix_root))

    logger.success("DiFix refinement stage completed")
