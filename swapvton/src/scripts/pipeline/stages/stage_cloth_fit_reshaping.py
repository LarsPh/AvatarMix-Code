from loguru import logger
from pathlib import Path
from scripts.pipeline.sampling.subject_discovery import get_padded_subject_id
from utils.cloth_fit_utils import (
    validate_cloth_fit_inputs,
    generate_cloth_fit_setup,
    execute_polyfem_simulation,
    find_deformed_mesh_output
)

def _normalize_direction_tokens(directions_value):

    if directions_value is None:
        return None

    if isinstance(directions_value, str):
        raw = [directions_value]
    elif isinstance(directions_value, (list, tuple, set)):
        raw = list(directions_value)
    else:

        return None

    toks = set()
    for v in raw:
        s = str(v).strip().lower()
        if not s:
            continue
        s = s.replace(" ", "").replace("-", "_")
        if s in {"both", "all", "true", "yes", "1"}:
            return None
        if s in {"a_to_b", "a2b", "a->b"}:
            toks.add("a_to_b")
        elif s in {"b_to_a", "b2a", "b->a"}:
            toks.add("b_to_a")
        else:

            toks.add(s)
    return toks if toks else None


def _dir_tag_for_pair(subjects, garment_subject, avatar_subject):

    if len(subjects) >= 2 and garment_subject == subjects[0] and avatar_subject == subjects[1]:
        return "a_to_b"
    if len(subjects) >= 2 and garment_subject == subjects[1] and avatar_subject == subjects[0]:
        return "b_to_a"

    return f"{garment_subject}_to_{avatar_subject}".lower()


def _block_enabled_for_direction(block_cfg, *, dir_tag: str) -> bool:

    if not isinstance(block_cfg, dict):
        return False
    if not bool(block_cfg.get("enabled", False)):
        return False
    dirs = _normalize_direction_tokens(block_cfg.get("directions", None))
    if dirs is None:
        return True
    return str(dir_tag).lower() in dirs


def _normalize_dir_key(k: str) -> str:
    s = str(k).strip().lower()
    s = s.replace(" ", "").replace("-", "_")
    if s in {"a_to_b", "a2b", "a->b"}:
        return "a_to_b"
    if s in {"b_to_a", "b2a", "b->a"}:
        return "b_to_a"
    return s


def _deep_merge_dicts(base: dict, override: dict) -> dict:

    out = dict(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge_dicts(out[k], v)
        else:
            out[k] = v
    return out


def _cfg_for_direction(block_cfg: dict, *, dir_tag: str) -> dict:

    if not isinstance(block_cfg, dict):
        return {}
    base = dict(block_cfg)
    per = base.pop("per_direction", None)
    if not isinstance(per, dict):
        return base
    key = _normalize_dir_key(dir_tag)

    merged = base
    if "default" in per and isinstance(per.get("default"), dict):
        merged = _deep_merge_dicts(merged, per["default"])

    for k, v in per.items():
        if _normalize_dir_key(k) == key and isinstance(v, dict):
            merged = _deep_merge_dicts(merged, v)
            break
    return merged


def stage_cloth_fit_reshaping(config, subjects, debug_subprocess=False):


    from pipeline.stages.first_swap import stage_reposing
    from pipeline.core.directions import get_pair_directions
    dataset_type = config['data_type']

    logger.info("\n--- Cloth-Fit Body Reshaping ---")

    cfg = config['pipeline_stages'].get('11c_cloth_fit_reshaping', {})


    if not cfg.get('enabled', False):
        logger.info("Cloth-fit reshaping disabled in config - skipping stage")
        return


    cloth_fit_root = Path(cfg['cloth_fit_project_root'])
    dataset_root = Path(config['paths']['avatarrex_output'])
    smplx_segmentation_json_path = config['paths'].get('smplx_segmentation_json')


    subject_directions = get_pair_directions(
        stage_name="cloth_fit_reshaping",
        config=config,
        subjects=subjects,
    )


    path_a_to_b = dataset_root / "gs_on_mesh_repose" / f"{dataset_type}_{subjects[0]}_to_{subjects[1]}" / "anim_0000_to_targets_meshes" / "smpl_reposed_frame_0000_cleaned.obj"
    path_b_to_a = dataset_root / "gs_on_mesh_repose" / f"{dataset_type}_{subjects[1]}_to_{subjects[0]}" / "anim_0000_to_targets_meshes" / "smpl_reposed_frame_0000_cleaned.obj"

    body_a_cleaned = dataset_root / get_padded_subject_id(subjects[0]) / "mesh" / "processed" / "smpl_body_cleaned.obj"
    body_b_cleaned = dataset_root / get_padded_subject_id(subjects[1]) / "mesh" / "processed" / "smpl_body_cleaned.obj"

    body_a_trim = dataset_root / get_padded_subject_id(subjects[0]) / "mesh" / "processed" / "smpl_body_trim.obj"
    body_b_trim = dataset_root / get_padded_subject_id(subjects[1]) / "mesh" / "processed" / "smpl_body_trim.obj"
    skip_existing_reposing = cfg.get('skip_existing_reposing', False)
    area_cfg = cfg.get("area_aware_fit", {}) if isinstance(cfg.get("area_aware_fit", {}), dict) else {}


    step3_cfg_all = cfg.get("step3_anneal", {}) if isinstance(cfg.get("step3_anneal", {}), dict) else {}
    a2_enabled_any = any(
        _block_enabled_for_direction(area_cfg, dir_tag=_dir_tag_for_pair(subjects, g, a))
        for (g, a) in subject_directions
    )
    a1_enabled_any = any(
        _block_enabled_for_direction(step3_cfg_all, dir_tag=_dir_tag_for_pair(subjects, g, a))
        for (g, a) in subject_directions
    )

    if (not path_a_to_b.exists() or not path_b_to_a.exists() or
        (a1_enabled_any and (not body_a_cleaned.exists() or not body_b_cleaned.exists())) or
        (a2_enabled_any and (not body_a_trim.exists() or not body_b_trim.exists())) or
        not skip_existing_reposing):
        logger.info("\n-- Step 0: Generating reposed SMPL meshes for cloth-fit --")
        stage_reposing(
            config,
            subjects,
            debug_subprocess=debug_subprocess,
            operation_mode="reposing",
            subject_directions=subject_directions,
            force_enable_reshape=False,
            disable_renders=True,
            save_skeleton_for_cloth_fit=True,
            smplx_segmentation_json_path=smplx_segmentation_json_path,
            cloth_fit_remove_feet=cfg.get('cloth_fit_remove_feet', False),
            cloth_fit_keep_palm=cfg.get("cloth_fit_keep_palm", False),

            generate_smpl_body_cleaned=bool(a1_enabled_any),
            overwrite_smpl_body_cleaned=bool(a1_enabled_any),

            a2_generate_trim_meshes=a2_enabled_any and bool(area_cfg.get("generate_trim_meshes", True)),
            a2_generate_correspondence=a2_enabled_any and bool(area_cfg.get("generate_correspondence", True)),
            a2_overwrite=a2_enabled_any and bool(area_cfg.get("overwrite", True)),
            a2_nn_scale_factor=float(area_cfg.get("nn_scale_factor", 1000.0)) if a2_enabled_any else 1000.0,

            a2_corr_output_path=str(area_cfg.get("corr_filename", "area_aware_fit_corr.txt") or "area_aware_fit_corr.txt"),
            a2_visualize_correspondence=a2_enabled_any and bool(area_cfg.get("visualize_correspondence", True)),
        )
        logger.success("Reposed SMPL meshes and skeletons generated successfully")
    else:
        logger.info("Reposed SMPL meshes and skeletons already generated, skipping step 0")


    deformed_meshes = {}


    failed_pairs = []

    for garment_subject, avatar_subject in subject_directions:
        logger.info(f"\n-- Cloth-fit reshaping: {garment_subject} → {avatar_subject} body shape --")
        dir_tag = _dir_tag_for_pair(subjects, garment_subject, avatar_subject)


        logger.info("Step 1: Validating required input files...")
        try:
            required_files = validate_cloth_fit_inputs(
                dataset_type, dataset_root, get_padded_subject_id(avatar_subject), get_padded_subject_id(garment_subject)
            )
        except FileNotFoundError as e:
            logger.error(f"Validation failed: {e}")
            logger.error(f"Ensure process_mesh stage was run with create_simplified_mesh=true")
            raise


        logger.info("Step 2: Generating cloth-fit configuration...")
        garment_padded = get_padded_subject_id(garment_subject)
        avatar_padded = get_padded_subject_id(avatar_subject)
        subdir_name = f"{avatar_padded}_avatar_{garment_padded}_garment"


        subdir_suffix = str(cfg.get("cloth_fit_suffix", "") or "")
        subdir_name += subdir_suffix

        output_dir = dataset_root / "cloth_fit_output" / subdir_name
        output_dir.mkdir(parents=True, exist_ok=True)

        def _append_mask_if_exists(*, masks_list, path: Path, multiplier: float, label: str):
            if path.exists():
                masks_list.append({"indices_path": str(path), "multiplier": float(multiplier)})
                logger.info(f"Enabled {label} mask for cloth-fit:")
                logger.info(f"  Path: {path}")
                logger.info(f"  Multiplier: {multiplier}")
                return True
            logger.warning(f"{label} mask not found: {path}")
            return False

        def _count_obj_vertices(path: Path) -> int:
            n = 0
            with open(path, "r", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        n += 1
            return n

        def _read_int_ids_ascii(path: Path) -> list[int]:
            txt = path.read_text().strip()
            if not txt:
                return []
            ids = []
            for tok in txt.split():
                ids.append(int(tok))
            return ids

        def _write_int_ids_ascii(path: Path, ids: list[int]) -> None:

            path.write_text(" ".join(str(i) for i in ids) + "\n")


        garment_mesh_path = Path(required_files["garment_mesh_simplified_path"])
        semantic_cfg_raw = cfg.get("semantic_weighting", {}) or {}
        semantic_cfg = _cfg_for_direction(semantic_cfg_raw, dir_tag=dir_tag)
        semantic_enabled = _block_enabled_for_direction(semantic_cfg, dir_tag=dir_tag)
        skirt_no_fit_cfg = cfg.get("skirt_no_fit", {}) if isinstance(cfg.get("skirt_no_fit", {}), dict) else {}
        skirt_no_fit_enabled = bool(skirt_no_fit_cfg.get("enabled", False))
        fit_weight_vis = None
        if semantic_enabled:
            vis_cfg = semantic_cfg.get("fit_weight_vis", {}) or {}
            vis_enabled = bool(vis_cfg.get("enabled", True))
            if vis_enabled:
                fit_weight_vis = {
                    "enabled": True,
                    "export_mask_only": True,
                    "export_final": True,
                    "mask_only_filename": str(
                        vis_cfg.get("mask_only_filename", "fit_weight_vertex_masks_only.ply")
                    ),
                    "final_filename": str(
                        vis_cfg.get("final_filename", "fit_weight_vertex_masks_times_area.ply")
                    ),
                }
        fit_weight_masks = []
        enable_fit_weight_mask = cfg.get("enable_fit_weight_mask", cfg.get("enable_skin_weight_mask", False))
        remove_neck_from_fit_weight_mask = cfg.get(
            "remove_neck_from_fit_weight_mask",
            cfg.get("remove_neck_from_skin_weight_mask", cfg.get("remove_neck_from_skin_mask", False)),
        )
        if enable_fit_weight_mask or cfg.get('enable_no_fit_mask', False) or semantic_enabled or skirt_no_fit_enabled:


            skin_mask_suffix = "_no_torso_skin" if remove_neck_from_fit_weight_mask else ""
            skin_mask_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_skin_indices{skin_mask_suffix}.txt"


            limb_mult = cfg.get('limb_no_hands_fit_weight_multiplier')
            if limb_mult is None:
                limb_mult = cfg.get(
                    'limb_no_hands_skin_weight_multiplier',
                    cfg.get('fit_weight_multiplier', cfg.get('skin_weight_multiplier', 0.01)),
                )
            _append_mask_if_exists(
                masks_list=fit_weight_masks,
                path=skin_mask_path,
                multiplier=limb_mult,
                label="limb_no_hands fit_weight",
            )


            hands_mask_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_hands_indices.txt"
            hands_mult = cfg.get('hands_skin_weight_multiplier', 0.01)
            _append_mask_if_exists(
                masks_list=fit_weight_masks,
                path=hands_mask_path,
                multiplier=hands_mult,
                label="hands fit_weight",
            )


            feet_mask_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_feet_indices.txt"
            feet_mult = cfg.get('feet_skin_weight_multiplier', 0.01)
            _append_mask_if_exists(
                masks_list=fit_weight_masks,
                path=feet_mask_path,
                multiplier=feet_mult,
                label="feet fit_weight",
            )


            if semantic_enabled:
                regions = semantic_cfg.get("regions", {}) or {}
                defaults = {
                    "chest": {"fit_mult": 2.5, "sim_mult": 0.5},
                    "hip": {"fit_mult": 1.6, "sim_mult": 0.7},
                }
                for region, d in defaults.items():
                    r_cfg = regions.get(region, {}) or {}
                    fit_mult = float(r_cfg.get("fit_mult", d["fit_mult"]))
                    sem_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_semantic_{region}_indices.txt"
                    _append_mask_if_exists(
                        masks_list=fit_weight_masks,
                        path=sem_path,
                        multiplier=fit_mult,
                        label=f"semantic {region} fit_weight",
                    )


            if skirt_no_fit_enabled:
                ids_path_cfg = str(skirt_no_fit_cfg.get("vertex_ids_path", "") or "").strip()
                ids_filename = str(skirt_no_fit_cfg.get("vertex_ids_filename", "skirt_no_fit_vert_ids.txt") or "skirt_no_fit_vert_ids.txt").strip()
                ids_path = Path(ids_path_cfg) if ids_path_cfg else (garment_mesh_path.parent / ids_filename)
                if not ids_path.exists():
                    raise FileNotFoundError(f"skirt_no_fit.enabled=true but vertex id file missing: {ids_path}")
                index_base = int(skirt_no_fit_cfg.get("index_base", 0))
                if index_base not in (0, 1):
                    raise ValueError(f"skirt_no_fit.index_base must be 0 or 1, got {index_base}")


                n_garment_verts = _count_obj_vertices(garment_mesh_path)
                raw_ids = _read_int_ids_ascii(ids_path)
                if not raw_ids:
                    raise ValueError(f"skirt_no_fit vertex id file is empty: {ids_path}")
                ids0 = [i - index_base for i in raw_ids]
                mn, mx = min(ids0), max(ids0)
                if mn < 0 or mx >= n_garment_verts:
                    raise ValueError(
                        "skirt_no_fit vertex IDs out of range after applying index_base.\n"
                        f"- file: {ids_path}\n"
                        f"- index_base: {index_base}\n"
                        f"- simplified_garment_mesh: {garment_mesh_path} (n_verts={n_garment_verts})\n"
                        f"- id_min/id_max (0-based): {mn}/{mx}\n"
                        "This usually means the IDs were exported from a different mesh (wrong vertex count/order)."
                    )

                ids_path_use = ids_path
                if index_base != 0:
                    suffix = ids_path.suffix if ids_path.suffix else ".txt"
                    ids_path_use = ids_path.with_name(f"{ids_path.stem}_0based{suffix}")
                    _write_int_ids_ascii(ids_path_use, ids0)
                    logger.info(f"Converted skirt_no_fit indices to 0-based: {ids_path_use} (from index_base={index_base})")

                mult = float(skirt_no_fit_cfg.get("multiplier", 0.0))
                fit_weight_masks.append({"indices_path": str(ids_path_use), "multiplier": mult})
                logger.info(f"Added skirt_no_fit fit_weight mask: {ids_path_use} multiplier={mult}")

        similarity_weight_mask = []
        if cfg.get('enable_similarity_mask', False):

            hands_mask_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_hands_indices.txt"
            hands_sim_mult = cfg.get(
                'hands_similarity_mask_weight',
                cfg.get('similarity_mask_weight', 10.0),
            )
            _append_mask_if_exists(
                masks_list=similarity_weight_mask,
                path=hands_mask_path,
                multiplier=hands_sim_mult,
                label="hands similarity_weight",
            )

            feet_mask_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_feet_indices.txt"
            feet_sim_mult = cfg.get('feet_similarity_mask_weight', 1.0)
            _append_mask_if_exists(
                masks_list=similarity_weight_mask,
                path=feet_mask_path,
                multiplier=feet_sim_mult,
                label="feet similarity_weight",
            )


        if semantic_enabled:
            regions = semantic_cfg.get("regions", {}) or {}
            defaults = {
                "chest": {"fit_mult": 2.5, "sim_mult": 0.5},
                "hip": {"fit_mult": 1.6, "sim_mult": 0.7},
            }
            for region, d in defaults.items():
                r_cfg = regions.get(region, {}) or {}
                sim_mult = float(r_cfg.get("sim_mult", d["sim_mult"]))
                sem_path = garment_mesh_path.parent / f"{garment_mesh_path.stem}_semantic_{region}_indices.txt"
                _append_mask_if_exists(
                    masks_list=similarity_weight_mask,
                    path=sem_path,
                    multiplier=sim_mult,
                    label=f"semantic {region} similarity_weight",
                )


        keep_palm = bool(cfg.get("cloth_fit_keep_palm", False))
        palm_suffix = "_keep_palm" if keep_palm else ""
        avatar_hand_removel_mask_path = (
            dataset_root
            / "gs_on_mesh_repose"
            / f"{dataset_type}_{avatar_subject}_to_{garment_subject}"
            / "anim_0000_to_targets_meshes"
            / f"smpl_reposed_frame_0000_cleaned_smplx_skin_indices{palm_suffix}.txt"
        )
        if cfg.get('enable_avatar_hand_removel_mask', False) and not avatar_hand_removel_mask_path.exists():
            logger.warning(f"Avatar hand removel weight mask not found: {avatar_hand_removel_mask_path}")
        avatar_hand_removel_mask_path = str(avatar_hand_removel_mask_path)

        skip_cloth_fit_simulation = cfg.get('skip_cloth_fit_simulation', False)
        if not skip_cloth_fit_simulation:


            cfg_for_setup = dict(cfg)

            cfg_for_setup.pop("area_aware_fit", None)
            cfg_for_setup.pop("semantic_weighting", None)
            cfg_for_setup.pop("hem_boundary", None)
            cfg_for_setup.pop("skirt_no_fit", None)
            cfg_for_setup.pop("ring_constraints", None)


            if skirt_no_fit_enabled:
                cfg_for_setup.setdefault("enable_fit_weight_mask", True)


            step3_anneal_cfg = cfg_for_setup.get("step3_anneal", {}) if isinstance(cfg_for_setup.get("step3_anneal", {}), dict) else {}
            step3_enabled_dir = _block_enabled_for_direction(step3_anneal_cfg, dir_tag=dir_tag)
            if isinstance(step3_anneal_cfg, dict):
                step3_anneal_cfg = dict(step3_anneal_cfg)
                step3_anneal_cfg.pop("directions", None)
                if not step3_enabled_dir:
                    step3_anneal_cfg["enabled"] = False
                cfg_for_setup["step3_anneal"] = step3_anneal_cfg
            if cfg.get("height_aware", False):

                normalization = {
                    "mode": "separate_translation_no_scale",
                    "restore_output_translation": True,
                    "save_offsets": True,
                    "save_offsets_path": "",
                }


                if dataset_type == "thuman2" and bool(cfg.get("height_aware_subject_scale", False)):
                    source_scale_path = dataset_root / garment_padded / "mesh" / "processed" / "smpl_scale_inv.json"


                    target_scale_path = source_scale_path
                    normalization = {
                        "mode": "separate_translation_with_subject_scale",
                        "restore_output_translation": True,
                        "save_offsets": True,
                        "save_offsets_path": "",
                        "source_scale_path": str(source_scale_path),
                        "target_scale_path": str(target_scale_path),
                    }

                cfg_for_setup.setdefault("normalization", normalization)


            source_avatar_mesh_path = dataset_root / garment_padded / "mesh" / "processed" / "smpl_body_cleaned.obj"
            if step3_anneal_cfg.get("enabled", False) and not source_avatar_mesh_path.exists():
                raise FileNotFoundError(
                    f"step3_anneal.enabled=true but source_avatar_mesh_path missing: {source_avatar_mesh_path}. "
                    f"Expected it to be generated during Step 0 reposing."
                )


            hem_boundary_cfg = cfg.get("hem_boundary", {}) if isinstance(cfg.get("hem_boundary", {}), dict) else {}
            hem_boundary_setup = None
            if bool(hem_boundary_cfg.get("enabled", False)):

                hem_ids_path_cfg = str(hem_boundary_cfg.get("vertex_ids_path", "") or "").strip()
                hem_ids_filename = str(hem_boundary_cfg.get("vertex_ids_filename", "hem_vert_ids.txt") or "hem_vert_ids.txt").strip()
                hem_ids_path = Path(hem_ids_path_cfg) if hem_ids_path_cfg else (dataset_root / garment_padded / "mesh" / "processed" / hem_ids_filename)

                hem_boundary_setup = {
                    "enabled": True,
                    "vertex_ids_path": str(hem_ids_path),
                    "index_base": int(hem_boundary_cfg.get("index_base", 1)),
                    "overlap_threshold": float(hem_boundary_cfg.get("overlap_threshold", 0.2)),
                    "max_curves": int(hem_boundary_cfg.get("max_curves", 1)),
                }

                if "dilate_steps" in hem_boundary_cfg:
                    hem_boundary_setup["dilate_steps"] = int(hem_boundary_cfg.get("dilate_steps", -1))
                if "auto_dilate_max_steps" in hem_boundary_cfg:
                    hem_boundary_setup["auto_dilate_max_steps"] = int(hem_boundary_cfg.get("auto_dilate_max_steps", 4))
                if "select_mode" in hem_boundary_cfg:
                    hem_boundary_setup["select_mode"] = str(hem_boundary_cfg.get("select_mode", "overlap") or "overlap").strip()
                if "accept_overlap" in hem_boundary_cfg:
                    hem_boundary_setup["accept_overlap"] = float(hem_boundary_cfg.get("accept_overlap", 0.5))
                if "accept_coverage" in hem_boundary_cfg:
                    hem_boundary_setup["accept_coverage"] = float(hem_boundary_cfg.get("accept_coverage", 0.7))
                if "export_skeleton_debug" in hem_boundary_cfg:
                    hem_boundary_setup["export_skeleton_debug"] = bool(hem_boundary_cfg.get("export_skeleton_debug", False))
                if "skeleton_debug_prefix" in hem_boundary_cfg:
                    hem_boundary_setup["skeleton_debug_prefix"] = str(hem_boundary_cfg.get("skeleton_debug_prefix", "hem_boundary_skeleton") or "hem_boundary_skeleton").strip()
                if "curvature_penalty_weight" in hem_boundary_cfg:
                    hem_boundary_setup["curvature_penalty_weight"] = float(hem_boundary_cfg.get("curvature_penalty_weight", 0.0))
                if "twist_penalty_weight" in hem_boundary_cfg:
                    hem_boundary_setup["twist_penalty_weight"] = float(hem_boundary_cfg.get("twist_penalty_weight", 0.0))


                if "export_skeleton_debug" not in hem_boundary_setup and "export_skeleton_debug" in cfg:
                    hem_boundary_setup["export_skeleton_debug"] = bool(cfg.get("export_skeleton_debug", False))
                if "skeleton_debug_prefix" not in hem_boundary_setup and "skeleton_debug_prefix" in cfg:
                    hem_boundary_setup["skeleton_debug_prefix"] = str(cfg.get("skeleton_debug_prefix", "hem_boundary_skeleton") or "hem_boundary_skeleton").strip()


            ring_constraints_cfg = cfg.get("ring_constraints", {}) if isinstance(cfg.get("ring_constraints", {}), dict) else {}
            ring_constraints_setup = None
            manual_rings_cfg = ring_constraints_cfg.get("manual", []) if isinstance(ring_constraints_cfg.get("manual", []), list) else []
            if manual_rings_cfg:
                manual_setup = []
                for i, item in enumerate(manual_rings_cfg):
                    if not isinstance(item, dict):
                        logger.warning(f"Skip ring_constraints.manual[{i}]: expected dict, got {type(item)}")
                        continue
                    if not bool(item.get("enabled", False)):
                        continue

                    ring_name = str(item.get("name", f"ring_{i}") or f"ring_{i}").strip()
                    ids_path_cfg = str(item.get("vertex_ids_path", "") or "").strip()
                    ids_filename = str(item.get("vertex_ids_filename", f"{ring_name}_vert_ids.txt") or f"{ring_name}_vert_ids.txt").strip()
                    ids_path = Path(ids_path_cfg) if ids_path_cfg else (dataset_root / garment_padded / "mesh" / "processed" / ids_filename)

                    ring_setup = {
                        "name": ring_name,
                        "enabled": True,
                        "vertex_ids_path": str(ids_path),
                        "index_base": int(item.get("index_base", 1)),
                        "max_curves": int(item.get("max_curves", 1)),
                    }

                    if "dilate_steps" in item:
                        ring_setup["dilate_steps"] = int(item.get("dilate_steps", -1))
                    if "auto_dilate_max_steps" in item:
                        ring_setup["auto_dilate_max_steps"] = int(item.get("auto_dilate_max_steps", 4))
                    if "select_mode" in item:
                        ring_setup["select_mode"] = str(item.get("select_mode", "overlap") or "overlap").strip()
                    if "accept_overlap" in item:
                        ring_setup["accept_overlap"] = float(item.get("accept_overlap", 0.5))
                    if "accept_coverage" in item:
                        ring_setup["accept_coverage"] = float(item.get("accept_coverage", 0.7))

                    if "curvature_penalty_weight" in item:
                        ring_setup["curvature_penalty_weight"] = float(item.get("curvature_penalty_weight", 0.0))
                    if "twist_penalty_weight" in item:
                        ring_setup["twist_penalty_weight"] = float(item.get("twist_penalty_weight", 0.0))
                    if "curve_center_target_weight" in item:
                        ring_setup["curve_center_target_weight"] = float(item.get("curve_center_target_weight", 0.0))
                    if "is_skirt" in item:
                        ring_setup["is_skirt"] = bool(item.get("is_skirt", False))

                    if "export_skeleton_debug" in item:
                        ring_setup["export_skeleton_debug"] = bool(item.get("export_skeleton_debug", False))
                    if "skeleton_debug_prefix" in item:
                        ring_setup["skeleton_debug_prefix"] = str(item.get("skeleton_debug_prefix", "") or "").strip()
                    manual_setup.append(ring_setup)

                if manual_setup:
                    ring_constraints_setup = {"manual": manual_setup}


            a2_enabled = _block_enabled_for_direction(area_cfg, dir_tag=dir_tag)
            a2_setup = None
            if a2_enabled:
                source_body_trim_path = dataset_root / garment_padded / "mesh" / "processed" / "smpl_body_trim.obj"
                target_anim_dir = dataset_root / "gs_on_mesh_repose" / f"{dataset_type}_{avatar_subject}_to_{garment_subject}" / "anim_0000_to_targets_meshes"
                target_body_trim_path = target_anim_dir / "smpl_reposed_frame_0000_trim.obj"
                corr_filename = str(area_cfg.get("corr_filename", "area_aware_fit_corr.txt") or "area_aware_fit_corr.txt")
                correspondence_path = target_anim_dir / corr_filename

                missing = [p for p in (source_body_trim_path, target_body_trim_path, correspondence_path) if not p.exists()]
                if missing:
                    raise FileNotFoundError(
                        "A2 enabled but required artifacts are missing:\n" + "\n".join([f"  - {p}" for p in missing])
                    )


                a2_setup = {
                    "enabled": True,
                    "source_body_trim_path": str(source_body_trim_path),
                    "target_body_trim_path": str(target_body_trim_path),
                    "correspondence_path": str(correspondence_path),

                    "lambda": float(area_cfg.get("lambda", 6.0)),
                    "w_max": float(area_cfg.get("w_max", 4.0)),
                    "d_gate": float(area_cfg.get("d_gate", 0.05)),
                    "use_distance_gate": bool(area_cfg.get("use_distance_gate", True)),
                    "final_substep_only": bool(area_cfg.get("final_substep_only", True)),

                    "enable_surf_relax": bool(area_cfg.get("enable_surf_relax", False)),
                    "surf_k": float(area_cfg.get("surf_k", 0.8)),
                    "surf_min": float(area_cfg.get("surf_min", 0.7)),

                    "export_visualization": bool(area_cfg.get("cloth_fit_vis", True)),
                }
            setup_json_path = generate_cloth_fit_setup(
                output_dir=output_dir,
                **required_files,
                fit_weight_masks=fit_weight_masks,
                similarity_weight_masks=similarity_weight_mask,
                avatar_hand_removel_mask_path=avatar_hand_removel_mask_path,
                source_avatar_mesh_path=str(source_avatar_mesh_path),
                a2=a2_setup,
                fit_weight_vis=fit_weight_vis,
                hem_boundary=hem_boundary_setup,
                ring_constraints=ring_constraints_setup,
                **cfg_for_setup
            )

            logger.success(f"Generated setup.json: {setup_json_path}")


            skip_existing = cfg.get('skip_existing_mesh', False)
            if skip_existing:
                try:
                    existing_mesh = find_deformed_mesh_output(output_dir)
                    logger.info(f"Found existing deformed mesh: {existing_mesh}")
                    logger.info("skip_existing=true, skipping PolyFEM simulation")
                    deformed_mesh_path = existing_mesh

                    deformed_meshes[(garment_subject, avatar_subject)] = deformed_mesh_path
                    continue
                except FileNotFoundError:
                    logger.info("No existing output found, running simulation")
            if skip_cloth_fit_simulation:

                logger.info("skip_cloth_fit_simulation=true, skipping PolyFEM simulation")
                continue


            logger.info("Step 3: Executing PolyFEM simulation with fallback retry...")

            simulation_succeeded = False


            setup_json_path = generate_cloth_fit_setup(
                output_dir=output_dir,
                **required_files,
                fit_weight_masks=fit_weight_masks,
                similarity_weight_masks=similarity_weight_mask,
                avatar_hand_removel_mask_path=avatar_hand_removel_mask_path,
                source_avatar_mesh_path=str(source_avatar_mesh_path),
                a2=a2_setup,
                fit_weight_vis=fit_weight_vis,
                hem_boundary=hem_boundary_setup,
                ring_constraints=ring_constraints_setup,
                **cfg_for_setup
            )


            success, stdout, stderr, error_msg = execute_polyfem_simulation(
                setup_json_path=setup_json_path,
                cloth_fit_root=cloth_fit_root,
                build_type=cfg.get('polyfem_build_type', 'Release'),
                max_threads=cfg.get('max_threads', 16),
                max_time=cfg.get('max_time', None)
            )

            if success:
                logger.success(f"Simulation succeeded")
                simulation_succeeded = True
            else:
                logger.error(f"Simulation failed: {error_msg}")


            if not simulation_succeeded:
                logger.error(f"PolyFEM simulation failed for {garment_subject}→{avatar_subject}")
                logger.warning("Skipping this pair and continuing to next...")


                from datetime import datetime
                failure_marker = output_dir / "SIMULATION_FAILED.txt"
                failure_marker.write_text(f"""PolyFEM simulation failed
                Pair: {garment_subject} → {avatar_subject}
                Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Last error: {error_msg}

                """)

                failed_pairs.append((garment_subject, avatar_subject, error_msg))
                continue


        logger.info("Step 4: Retrieving deformed mesh output...")
        deformed_mesh_path = find_deformed_mesh_output(output_dir)
        logger.success(f"Found deformed mesh: {deformed_mesh_path}")


        deformed_meshes[(garment_subject, avatar_subject)] = deformed_mesh_path


    logger.info("\n-- Step 5: Restoring deformed meshes to full Gaussian resolution --")

    if cfg.get('restore_deformed_meshes', False):
        if len(deformed_meshes) == 0:
            logger.warning("No deformed meshes found, skipping restoration")

            failed_pairs.append((subjects[0], subjects[1], "No deformed meshes found"))

        for (garment_subject, avatar_subject), deformed_mesh_path in deformed_meshes.items():
            logger.info(f"Restoring {garment_subject} (reshaped to {avatar_subject}'s shape)...")
            if cfg.get('skip_existing_restoration', False):

                suffix = str(cfg.get("cloth_fit_suffix", "") or "")
                existing_filename = f"reshaped_gs_target_shape_0000.ply_cloth_fit_reshaped{suffix}.ply"
                existing_restoration_output = dataset_root / "gs_on_mesh_repose" / f"{dataset_type}_{garment_subject}_to_{avatar_subject}" / "reshaped_gaussians_ply_0000" / existing_filename
                if existing_restoration_output.exists():
                    logger.info(f"Existing restoration output found: {existing_restoration_output}")
                    logger.info("skip_existing_restoration=true, skipping restoration")
                    continue
                else:
                    logger.info("No existing restoration output found, running restoration")


            cloth_fit_restoration_suffix = str(cfg.get("cloth_fit_suffix", "") or "")
            logger.info(f"Cloth-fit restoration suffix: {cloth_fit_restoration_suffix}")
            gs_safety = cfg.get("gs_scale_safety", {}) if isinstance(cfg.get("gs_scale_safety", {}), dict) else {}
            stage_reposing(
                config,
                subjects,
                debug_subprocess=debug_subprocess,
                operation_mode="cloth_fit_restoration",
                subject_directions=[(garment_subject, avatar_subject)],
                cloth_fit_deformed_mesh_path=str(deformed_mesh_path),
                cloth_fit_deformed_mesh_already_restored=cfg.get("height_aware", False),
                cloth_fit_restoration_suffix=cloth_fit_restoration_suffix,
                cloth_fit_gs_scale_safety_enabled=bool(gs_safety.get("enabled", False)),
                cloth_fit_gs_scale_safety_percentile=float(gs_safety.get("percentile", 0.995)),
                cloth_fit_gs_scale_safety_hard_max_scale=float(gs_safety.get("hard_max_scale", 0.0)),
                cloth_fit_gs_scale_safety_ratio_enabled=bool(gs_safety.get("ratio_enabled", False)),
                cloth_fit_gs_scale_safety_ratio_percentile=float(gs_safety.get("ratio_percentile", 0.995)),
                cloth_fit_gs_scale_safety_ratio_hard_max=float(gs_safety.get("ratio_hard_max", 0.0)),
                cloth_fit_gs_scale_safety_ratio_symmetric=bool(gs_safety.get("ratio_symmetric", False)),
                cloth_fit_gs_scale_safety_action=str(gs_safety.get("action", "clamp") or "clamp"),
                cloth_fit_gs_scale_safety_eps_opacity=float(gs_safety.get("eps_opacity", 1e-6)),
            )

            logger.success(f"Restoration completed for {garment_subject}")


    if failed_pairs:
        logger.warning(f"\n{'='*80}")
        logger.warning(f"CLOTH-FIT SUMMARY: {len(failed_pairs)} pair(s) failed:")
        for garment, avatar, error in failed_pairs:
            logger.warning(f"  - {garment} → {avatar}: {error}")
        logger.warning(f"{'='*80}\n")

    logger.info("=" * 80)
    if failed_pairs:
        logger.warning(f"Cloth-fit reshaping stage completed with {len(failed_pairs)} failed pair(s)")
    else:
        logger.success("Cloth-fit reshaping stage completed successfully")
    logger.info("=" * 80)
