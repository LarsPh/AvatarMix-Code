from loguru import logger


def _get_data_preparation_stages(data_type: str = "") -> list[str]:


    if str(data_type).strip() == "thuman2":
        logger.info("Adding render stage for thuman2")
        return ["render", "convert_to_avatarrex"]
    return ["convert_to_avatarrex"]


def _get_first_swap_preparation_stages(data_type: str = "") -> list[str]:

    stages = [
        "convert_to_neus2", "neus2_train", "copy_mesh", "clean_mesh", "splatting_avatar",
        "smplx_fitting", "hands_transplant", "parse_mesh", "process_mesh", "lbs_transfer",
    ]

    if str(data_type).strip() == "talkbody4d":
        stages = [s for s in stages if s != "smplx_fitting"]
        logger.info("Skipping SMPL-X fitting for talkbody4d")
    return stages


def _get_first_swap_stages(legacy_mode: bool, enable_legacy_reshape: bool) -> list[str]:

    if legacy_mode:
        return ["reposing", "swapping"]
    else:
        if enable_legacy_reshape:
            return ["head_donator_reposing", "body_donator_reshaping", "direct_swapping"]
        else:
            return ["head_donator_reposing", "cloth_fit_reshaping", "direct_swapping"]


def _get_first_swap_rendering_stages(include_refinement: bool = False) -> list[str]:

    stages = [
        "render_swapped_gaussians",


    ]
    if include_refinement:
        stages.insert(2, "refine_rendered_images")
    return stages

def _get_first_swap_refinement_stages() -> list[str]:
    return [
        "difix_refinement",
    ]

def _get_first_swap_avatar_update_stages() -> list[str]:
    return [
        "splatting_avatar_gs_finetune",
    ]

def _get_second_swap_preparation_stages() -> list[str]:

    return [
        "convert_swapped_to_neus2", "neus2_train_swapped", "copy_mesh_swapped",
        "clean_mesh_swapped", "parse_mesh_swapped", "process_mesh_swapped",
        "lbs_transfer_swapped", "splatting_avatar_swapped"
    ]


def _get_second_swap_stages(enable_reshape: bool) -> list[str]:

    if enable_reshape:
        return ["swapped_head_donator_reposing", "swapped_body_donator_reshaping", "swap_back"]
    else:
        return ["swapped_head_donator_reposing", "swap_back"]


def _get_second_swap_rendering_stages(include_refinement: bool = False) -> list[str]:

    stages = ["render_swapped_back_gaussians"]
    if include_refinement:
        stages.append("refine_swapped_back_images")
    return stages


def _get_first_swap_preparation_stages_subset() -> list[str]:

    return [
        "convert_to_neus2",
        "neus2_train",
        "copy_mesh",
        "clean_mesh",
        "parse_mesh"
    ]


def _get_mesh_processing_stages(data_type: str = "") -> list[str]:

    stages = [
        "smplx_fitting",
        "splatting_avatar",
        "process_mesh",
        "lbs_transfer",
    ]
    if str(data_type).strip() == "talkbody4d":
        stages = [s for s in stages if s != "smplx_fitting"]
    return stages


def _get_training_specific_stages() -> list[str]:

    return [
        "render_swapped_gt_head_aligned",
        "render_body_full_gt_aligned",
        "segment_swapped_renders",
        "create_combined_portraits"
    ]


def _get_testing_specific_stages() -> list[str]:

    return [
        "render_swapped_gt_head_aligned",
        "render_body_full_gt_aligned",
        "segment_swapped_renders",
        "create_validation_data_portraits"
    ]


def _build_training_pipeline(config: dict) -> list[str]:

    enable_legacy_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)

    stages = []


    stages.extend(_get_data_preparation_stages(config.get("data_type", "")))
    stages.extend(_get_first_swap_preparation_stages(config.get("data_type", "")))


    stages.extend(_get_first_swap_stages(legacy_mode=False, enable_legacy_reshape=enable_legacy_reshape))
    stages.extend(_get_first_swap_rendering_stages(include_refinement=False))


    stages.extend(_get_second_swap_preparation_stages())


    stages.extend(_get_second_swap_stages(enable_reshape=enable_legacy_reshape))
    stages.extend(_get_second_swap_rendering_stages(include_refinement=False))


    stages.extend(_get_training_specific_stages())

    return stages


def _build_testing_pipeline(config: dict) -> list[str]:

    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)

    stages = []


    stages.extend(_get_data_preparation_stages(config.get("data_type", "")))
    stages.extend(_get_first_swap_preparation_stages_subset())


    stages.extend(_get_mesh_processing_stages(config.get("data_type", "")))
    stages.extend(_get_first_swap_stages(legacy_mode=False, enable_legacy_reshape=enable_reshape))
    stages.extend(_get_first_swap_rendering_stages(include_refinement=False))


    stages.extend(_get_first_swap_refinement_stages())


    return stages


def _build_default_pipeline(config: dict) -> list[str]:

    import warnings
    warnings.warn(
        "_build_default_pipeline() is deprecated. Use _build_training_pipeline() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.warning("DEPRECATED: _build_default_pipeline() is deprecated. Use _build_training_pipeline() instead.")


    return _build_training_pipeline(config)


def _build_legacy_pipeline(config: dict) -> list[str]:

    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)

    stages = []
    stages.extend(_get_data_preparation_stages(config.get("data_type", "")))
    stages.extend(_get_first_swap_preparation_stages(config.get("data_type", "")))
    stages.extend(_get_first_swap_stages(legacy_mode=True, enable_legacy_reshape=enable_reshape))
    stages.extend(_get_first_swap_rendering_stages(include_refinement=False))
    stages.extend(_get_second_swap_preparation_stages())
    stages.extend(_get_second_swap_stages(enable_reshape=enable_reshape))
    stages.extend(_get_second_swap_rendering_stages(include_refinement=False))

    return stages


def _ensure_swap_back_prerequisites(stages: list[str], enable_reshape: bool) -> list[str]:

    if 'swap_back' not in stages:
        return stages

    prereq_stages = ['swapped_head_donator_reposing']
    if enable_reshape:
        prereq_stages.append('swapped_body_donator_reshaping')

    new_stages = []
    for stage in stages:
        if stage == 'swap_back':

            for prereq in prereq_stages:
                if prereq not in new_stages:
                    new_stages.append(prereq)
                    logger.info(f"Swap-back mode: Adding prerequisite '{prereq}' before 'swap_back'")
            new_stages.append('swap_back')
        elif stage not in prereq_stages:

            new_stages.append(stage)


    return new_stages


def _filter_multi_subject_stages(stages: list[str], subjects: list[str]) -> list[str]:

    if len(subjects) >= 2:
        return stages

    multi_subject_stages = {
        'reposing', 'head_donator_reposing', 'body_donator_reshaping',
        'swapping', 'direct_swapping', 'swap_back',
        'swapped_head_donator_reposing', 'swapped_body_donator_reshaping'
    }

    filtered_stages = []
    for stage in stages:
        if stage in multi_subject_stages:
            logger.warning(
                f"Warning: Skipping stage '{stage}' because it requires at least "
                f"two subjects and only {len(subjects)} was provided."
            )
        else:
            filtered_stages.append(stage)

    return filtered_stages


def _auto_configure_gt_aligned_head(config: dict, stages: list[str], args) -> None:

    if 'render_swapped_gt_head_aligned' not in stages or not args.direct_swap:
        return

    logger.info("\n--- Auto-Argument Propagation for GT-Aligned Head Pipeline ---")
    logger.info("Detected render_swapped_gt_head_aligned stage with direct_swap mode enabled")


    if 'pipeline_stages' not in config:
        config['pipeline_stages'] = {}


    if '26_render_swapped_gt_head_aligned' not in config['pipeline_stages']:
        config['pipeline_stages']['26_render_swapped_gt_head_aligned'] = {}
    config['pipeline_stages']['26_render_swapped_gt_head_aligned']['enabled'] = True
    logger.info("Auto-enabled stage 26_render_swapped_gt_head_aligned")


    if 'head_donator_reposing' in stages:
        if '11a_head_donator_reposing' not in config['pipeline_stages']:
            config['pipeline_stages']['11a_head_donator_reposing'] = {}

        config['pipeline_stages']['11a_head_donator_reposing']['save_gt_aligned_head'] = True
        config['pipeline_stages']['11a_head_donator_reposing']['enable_head_alignment_filtering'] = True
        logger.info("Auto-added --save_gt_aligned_head --enable_head_alignment_filtering to stage 11a")

    logger.info("GT-aligned head pipeline auto-configuration completed")


def _get_pair_stages(config: dict) -> tuple[list[str], str]:

    direct_swap_mode = config.get('_direct_swap_mode', False)
    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)

    if direct_swap_mode and enable_reshape:
        return (
            ["head_donator_reposing", "body_donator_reshaping", "direct_swapping"],
            "head reposing + body reshaping + direct swap"
        )
    elif direct_swap_mode:
        return (
            ["head_donator_reposing", "direct_swapping"],
            "head reposing + direct swap only"
        )
    else:
        return (
            ["reposing", "swapping"],
            "reposing + swapping"
        )


def _get_full_random_pair_pipeline(config) -> tuple[list[str], list[str], list[str]]:

    enable_reshape = config['pipeline_stages']['11_reposing'].get('enable_body_reshape', False)


    single_stages = []
    single_stages.extend(_get_data_preparation_stages(config.get("data_type", "")))
    single_stages.extend(_get_first_swap_preparation_stages(config.get("data_type", "")))


    pair_stages = _get_first_swap_stages(legacy_mode=False, enable_legacy_reshape=enable_reshape)


    post_swap_stages = []
    post_swap_stages.extend(_get_first_swap_rendering_stages(include_refinement=False))
    post_swap_stages.extend(_get_second_swap_preparation_stages())
    post_swap_stages.extend(_get_second_swap_stages(enable_reshape=enable_reshape))
    post_swap_stages.extend(_get_second_swap_rendering_stages(include_refinement=False))

    return single_stages, pair_stages, post_swap_stages
