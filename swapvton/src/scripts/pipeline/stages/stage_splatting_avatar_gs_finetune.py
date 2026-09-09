from pathlib import Path
from typing import Optional, Dict, Tuple
from loguru import logger
from PIL import Image

from pipeline.execution.subprocess_runner import run_command
from pipeline.execution.env_utils import build_python_command
from scripts.pipeline.core.config import use_cloth_fit_reshaped_gs, use_swap_hands
from pipeline.sampling.subject_discovery import get_padded_subject_id


def _resolve_original_swapped_dir_for_pair(
    swapped_renders_root: Path,
    head_id: str,
    body_id: str,
) -> Path:

    swapped_root = swapped_renders_root
    if not swapped_root.exists():
        raise FileNotFoundError(f"Missing swapped renders directory: {swapped_root}")

    prefix = f"swapped_{head_id}head_on_{body_id}body"
    matches = sorted([d for d in swapped_root.iterdir() if d.is_dir() and d.name.startswith(prefix)])
    if len(matches) == 0:
        raise FileNotFoundError(
            f"No original swapped renders found for prefix '{prefix}' under: {swapped_root}"
        )
    if len(matches) > 1:
        cand = "\n".join([f"- {m}" for m in matches])
        raise RuntimeError(
            "Ambiguous original swapped renders directory (multiple matches).\n"
            f"Prefix: {prefix}\n"
            f"Candidates:\n{cand}\n"
            "Please clean up duplicates or make directory naming unique."
        )
    return matches[0]


def find_cloth_fit_reshaped_ply(
    avatarrex_root: Path,
    head_id: str,
    body_id: str,
    cloth_fit_suffix: str,
    *,
    world_space: str = "A",
    hands_swapped: bool = False,
) -> Optional[Path]:

    head_padded = get_padded_subject_id(head_id)
    body_padded = get_padded_subject_id(body_id)


    swapped_root = avatarrex_root / "swapped"
    base = f"A{head_padded}_B{body_padded}_reshaped_cloth_fit{cloth_fit_suffix}"
    dir_hands = f"{base}_hands_swapped"

    candidates = [dir_hands, base] if bool(hands_swapped) else [base, dir_hands]
    target_dir = None
    for dn in candidates:
        td = swapped_root / dn
        if td.exists():
            target_dir = td
            break
    if target_dir is None:
        logger.debug(f"Cloth-fit directory not found: tried {candidates}")
        return None

    world_space = str(world_space).strip().upper()
    if world_space not in ("A", "B"):
        world_space = "A"


    ply_pattern = f"swapped_{head_padded}head_on_{body_padded}body_*.ply"
    plys = sorted(list(target_dir.glob(ply_pattern)))

    if plys:

        def _score(p: Path) -> int:
            n = p.name.lower()
            if world_space == "B":
                if n.endswith("_b.ply"):
                    return 3
                return 2 if ("in_b_world" in n or "world_b" in n) else 0

            if ("in_a_world" in n or "world_a" in n):
                return 2
            if n.endswith("_b.ply"):
                return -1
            return 1

        best = max(plys, key=lambda p: (_score(p), p.name))
        if _score(best) == 0 and len(plys) > 1:
            logger.warning(
                f"[Stage33] Could not find an explicit world-{world_space} swapped PLY among {len(plys)} candidates; "
                f"falling back to: {best.name}"
            )
        return best

    logger.debug(f"No PLY found in {target_dir.name}")
    return None


def restore_neck_masks(seamfixed_subject_dir: Path, only_update_neck: bool) -> None:

    if not only_update_neck:
        return

    logger.info("Restoring neck masks to full resolution...")


    cam_dirs = sorted([
        d for d in seamfixed_subject_dir.iterdir()
        if d.is_dir() and d.name.replace('_', '').replace('+', '').replace('-', '')[0].isdigit()
    ])

    if not cam_dirs:
        logger.warning(f"No camera directories found in {seamfixed_subject_dir}")
        return

    processed = 0
    for cam_dir in cam_dirs:

        difix_dir = cam_dir / 'difix_intermediates'


        neck_mask_cropped = difix_dir / 'neck_mask.png'
        metadata_json = difix_dir / 'metadata.json'
        neck_mask_full_path = difix_dir / 'neck_mask_full.png'


        if neck_mask_full_path.exists():
            logger.debug(f"  {cam_dir.name}: neck_mask_full.png exists, skipping")
            continue


        if not neck_mask_cropped.exists():
            raise FileNotFoundError(
                f"Neck mask not found (required for only_update_neck mode): {neck_mask_cropped}"
            )
        if not metadata_json.exists():
            raise FileNotFoundError(
                f"Metadata not found (required for only_update_neck mode): {metadata_json}"
            )


        import json
        with open(metadata_json) as f:
            metadata = json.load(f)

        bbox = metadata['bbox_metadata']
        x, y, size = bbox['x'], bbox['y'], bbox['size']


        ref_mask = cam_dir / 'mask' / 'pha' / '0000.png'
        if not ref_mask.exists():
            raise FileNotFoundError(f"Reference mask not found: {ref_mask}")

        from PIL import Image
        ref_img = Image.open(ref_mask)
        full_w, full_h = ref_img.size


        neck_cropped = Image.open(neck_mask_cropped)
        crop_w, crop_h = neck_cropped.size


        if (crop_w, crop_h) != (size, size):
            logger.warning(
                f"  {cam_dir.name}: Size mismatch - metadata says {size}×{size}, "
                f"image is {crop_w}×{crop_h}"
            )


        neck_full = Image.new('L', (full_w, full_h), 0)


        neck_full.paste(neck_cropped, (x, y))


        neck_full.save(neck_mask_full_path)
        logger.info(
            f"  {cam_dir.name}: Created neck_mask_full.png "
            f"({crop_w}×{crop_h} @ {x},{y} → {full_w}×{full_h})"
        )
        processed += 1

    logger.success(f"Neck mask restoration complete ({processed} cameras processed)")


def generate_rembg_masks(
    seamfixed_subject_dir: Path,
    custom_rgb_relpath: str,
    rembg_model: str = "sam",
    invert_mask: bool = False,
    union_with_existing: bool = False,
    existing_mask_relpath: str = "",
    skip_existing: bool = True
) -> str:

    logger.info(f"Generating rembg masks for custom RGB images (model: {rembg_model})...")


    rgb_path = Path(custom_rgb_relpath)
    rgb_dir = rgb_path.parent
    rembg_dir_name = "rembg_mask"
    mask_relpath = str(rgb_dir / rembg_dir_name)

    logger.info(f"  RGB path: {custom_rgb_relpath}")
    logger.info(f"  Mask directory: {mask_relpath}")


    cam_dirs = sorted([
        d for d in seamfixed_subject_dir.iterdir()
        if d.is_dir() and d.name.replace('_', '').replace('+', '').replace('-', '')[0].isdigit()
    ])

    if not cam_dirs:
        logger.warning(f"No camera directories found in {seamfixed_subject_dir}")
        return mask_relpath

    logger.info(f"  Found {len(cam_dirs)} camera directories")


    first_cam = cam_dirs[0]
    first_mask_dir = first_cam / mask_relpath
    if skip_existing and first_mask_dir.exists():
        logger.info(f"  ⏭ Rembg masks already exist, skipping generation")
        return mask_relpath


    processed = 0
    errors = 0

    for cam_dir in cam_dirs:
        try:

            rgb_full_path = cam_dir / custom_rgb_relpath
            if not rgb_full_path.exists():
                logger.warning(f"  {cam_dir.name}: RGB not found at {custom_rgb_relpath}")
                errors += 1
                continue


            output_dir = cam_dir / mask_relpath
            output_dir.mkdir(parents=True, exist_ok=True)


            input_img = Image.open(rgb_full_path)


            from rembg import new_session, remove
            session = new_session(rembg_model)
            output_img = remove(input_img, session=session)


            if output_img.mode == 'RGBA':
                mask = output_img.split()[3]
            else:

                logger.warning(f"  {cam_dir.name}: Unexpected output mode {output_img.mode}, converting to L")
                mask = output_img.convert('L')


            if invert_mask:
                from PIL import ImageOps
                mask = ImageOps.invert(mask)
                logger.debug(f"  {cam_dir.name}: Inverted mask")


            if union_with_existing and existing_mask_relpath:
                existing_mask_path = cam_dir / existing_mask_relpath
                if existing_mask_path.exists():
                    import numpy as np


                    existing_mask = Image.open(existing_mask_path).convert('L')


                    mask_np = np.array(mask)
                    existing_np = np.array(existing_mask)
                    union_np = np.maximum(mask_np, existing_np)


                    mask = Image.fromarray(union_np)
                    logger.debug(f"  {cam_dir.name}: Union with existing mask")
                else:
                    logger.warning(f"  {cam_dir.name}: Existing mask not found at {existing_mask_relpath}, using SAM only")


            mask_path = output_dir / '0000.png'
            mask.save(mask_path)

            logger.debug(f"  {cam_dir.name}: Generated mask → {mask_path.relative_to(cam_dir)}")
            processed += 1

        except Exception as e:
            logger.error(f"  {cam_dir.name}: Failed to generate mask - {e}")
            errors += 1
            continue


    logger.success(
        f"Rembg mask generation complete: "
        f"{processed} cameras processed"
        + (f", {errors} errors" if errors > 0 else "")
    )

    return str(Path(mask_relpath) / "0000.png")


def _validate_gs_finetune_output(
    *,
    splatting_project: Path,
    subject_name: str,
    model_prefix: str,
    iteration: int,
) -> bool:
    ply_path = splatting_project / 'output-splatting' / model_prefix / subject_name / 'point_cloud' / f'iteration_{iteration}' / 'point_cloud.ply'
    return ply_path.exists()


def stage_splatting_avatar_gs_finetune(
    config: dict,
    subjects: list,
    debug_subprocess: bool = False,
    *,
    _force_mode: str | None = None,
    _cfg_key: str = "33_splatting_avatar_gs_finetune",
) -> dict:

    logger.info("\n" + "="*80)
    logger.info("Stage 33: SplattingAvatar GS Fine-tune")
    logger.info("="*80)

    cfg = (config.get('pipeline_stages', {}).get(_cfg_key, None) or
           config.get('pipeline_stages', {}).get('33_splatting_avatar_gs_finetune', {}) or {})
    seamfixed_dir = Path(config['paths']['seamfixed_test_dir'])
    avatarrex_dir = Path(config['paths']['avatarrex_output'])
    splatting_project = Path(config['paths']['splatting_avatar_project'])

    logger.info(f"Seamfixed directory: {seamfixed_dir}")
    logger.info(f"Avatarrex directory: {avatarrex_dir}")
    logger.info(f"SplattingAvatar project: {splatting_project}")


    mode = str(_force_mode or cfg.get("mode", "free")).strip().lower()
    if mode not in ("auto", "free", "mesh_bound"):
        logger.warning(f"Unknown mode={mode!r}; falling back to 'free'")
        mode = "free"
    world_space = str(cfg.get("world_space", "A")).strip().upper()
    if world_space not in ("A", "B"):
        logger.warning(f"world_space={world_space!r} invalid; falling back to 'A'")
        world_space = "A"


    init_gs_ply_override = str(cfg.get("init_gs_ply_path", "") or "").strip()
    init_embed_override = str(cfg.get("init_embedding_json_path", "") or "").strip()
    init_mesh_override = str(cfg.get("init_proxy_mesh_path", "") or "").strip()


    cloth_fit_enabled, cloth_fit_suffix = use_cloth_fit_reshaped_gs(config, _cfg_key)
    swap_hands_enabled = use_swap_hands(config)

    only_update_neck = bool(cfg.get('only_update_neck', False))
    custom_rgb = cfg.get('custom_rgb_relpath', '')
    custom_mask = cfg.get('custom_mask_relpath', '')
    auto_generate_masks = bool(cfg.get('auto_generate_masks', False))
    rembg_model = cfg.get('rembg_model', 'sam')
    invert_mask = bool(cfg.get('invert_mask', False))
    union_with_existing_mask = bool(cfg.get('union_with_existing_mask', False))
    skip_existing = bool(cfg.get('skip_existing_training', False))
    hybrid_cfg = cfg.get("hybrid_supervision", {}) or {}
    hybrid_enabled = bool(hybrid_cfg.get("enabled", False))

    total_iteration = int(cfg.get('total_iteration', 5000))
    batch_size = int(cfg.get('batch_size', 1))
    num_workers = int(cfg.get('num_workers', 4))
    bg_color = str(cfg.get('bg_color', 'black'))
    splatting_config_name = str(cfg.get('splatting_config_name', 'thuman2_refine.yaml'))

    configs_str = str(cfg.get('configs', f'configs/splatting_avatar.yaml;configs/{splatting_config_name}'))


    if (not str(custom_mask).strip()) and ("gs_ft_fullbodyfix" in configs_str):
        custom_mask = "mask/pha/{frame}.png"
        logger.info("FullbodyFix bake config detected; defaulting custom_mask_relpath to 'mask/pha/{frame}.png'")


    if "model_path_prefix" in cfg:
        model_prefix = str(cfg.get("model_path_prefix"))
    else:
        if _cfg_key == "33_splatting_avatar_gs_finetune":
            model_prefix = "free_gaussian"
        elif mode == "mesh_bound":
            model_prefix = "gs_finetune_meshbound"
        else:
            model_prefix = "gs_finetune_free"

    logger.info(f"mode: {mode} (force={_force_mode})")
    logger.info(f"world_space: {world_space}")
    logger.info(f"model_path_prefix: {model_prefix}")
    logger.info(f"only_update_neck: {only_update_neck}")
    logger.info(f"skip_existing_training: {skip_existing}")

    if mode == "auto":

        pass


    env = config['conda_envs']['splatting']
    python_prefix = build_python_command(splatting_project, env)

    results = {'successful': [], 'failed': []}
    pair_count = 0


    st13 = (config.get("pipeline_stages", {}).get("13_render_swapped_gaussians", {}) or {})
    swapped_renders_root = avatarrex_dir / str(st13.get("output_base", "head_swapped_renders"))


    def _pair_dir(head_raw: str, body_raw: str) -> Path:
        head_padded = get_padded_subject_id(head_raw)
        body_padded = get_padded_subject_id(body_raw)
        base = f"A{head_padded}_B{body_padded}_reshaped_cloth_fit{cloth_fit_suffix}"
        if swap_hands_enabled:
            cand = avatarrex_dir / "swapped" / f"{base}_hands_swapped"
            if cand.exists():
                return cand
        return avatarrex_dir / "swapped" / base

    for i in range(len(subjects)):
        for j in range(len(subjects)):
            if i == j:
                continue
            pair_count += 1
            head_id_raw = subjects[i]
            body_id_raw = subjects[j]
            head_id = get_padded_subject_id(head_id_raw)
            body_id = get_padded_subject_id(body_id_raw)

            logger.info(f"\n{'-'*80}")
            logger.info(f"Processing pair {pair_count}: {head_id_raw} → {body_id_raw}")
            logger.info(f"{'-'*80}")

            seamfixed_subject = f"swapped_{head_id}head_on_{body_id}body"
            seamfixed_path = seamfixed_dir / seamfixed_subject

            try:
                if not seamfixed_path.exists():
                    raise FileNotFoundError(f"Seamfixed directory not found: {seamfixed_path}")


                rgb_relpath = custom_rgb
                mask_relpath = custom_mask
                if only_update_neck:
                    logger.info("🔄 Neck-only mode: restoring neck masks...")
                    restore_neck_masks(seamfixed_path, only_update_neck=True)
                    rgb_relpath = "0000.jpg"
                    mask_relpath = "difix_intermediates/neck_mask_full.png"
                elif auto_generate_masks and custom_rgb:
                    logger.info(f"🎭 Auto-generating rembg masks for custom RGB: {custom_rgb}")
                    mask_relpath = generate_rembg_masks(
                        seamfixed_subject_dir=seamfixed_path,
                        custom_rgb_relpath=custom_rgb,
                        rembg_model=rembg_model,
                        invert_mask=invert_mask,
                        union_with_existing=union_with_existing_mask,
                        existing_mask_relpath=custom_mask,
                        skip_existing=cfg.get('skip_existing_masks', True),
                    )
                    rgb_relpath = custom_rgb


                if init_gs_ply_override:
                    init_gs_ply = Path(init_gs_ply_override)
                    if not init_gs_ply.is_absolute():
                        init_gs_ply = avatarrex_dir / init_gs_ply
                else:
                    if not cloth_fit_enabled:
                        raise RuntimeError(
                            "Cloth-fit is disabled and init_gs_ply_path is not provided. "
                            "Enable cloth-fit or set 33_splatting_avatar_gs_finetune.init_gs_ply_path."
                        )
                    init_gs_ply = find_cloth_fit_reshaped_ply(
                        avatarrex_dir,
                        head_id_raw,
                        body_id_raw,
                        cloth_fit_suffix,
                        world_space=world_space,
                        hands_swapped=swap_hands_enabled,
                    )
                    if init_gs_ply is None:
                        raise FileNotFoundError(f"Init GS PLY not found for {head_id_raw}→{body_id_raw} (cloth_fit_suffix={cloth_fit_suffix!r}).")


                init_embed = None
                init_mesh = None
                if mode in ("mesh_bound", "auto"):

                    if init_embed_override:
                        init_embed = Path(init_embed_override)
                        if not init_embed.is_absolute():
                            init_embed = avatarrex_dir / init_embed
                    if init_mesh_override:
                        init_mesh = Path(init_mesh_override)
                        if not init_mesh.is_absolute():
                            init_mesh = avatarrex_dir / init_mesh

                    if init_embed is None or init_mesh is None:
                        pair_dir = _pair_dir(head_id_raw, body_id_raw)
                        neck_bridge_root = pair_dir / str(cfg.get("neck_bridge_dirname", "neck_bridge")) / world_space
                        if init_embed is None:
                            init_embed = neck_bridge_root / str(cfg.get("embedding_filename", "embedding_composed.json"))
                        if init_mesh is None:
                            init_mesh = neck_bridge_root / str(cfg.get("proxy_mesh_filename", "combined_body_head_no_bridge.ply"))

                    if mode == "auto":

                        if init_embed.exists() and init_mesh.exists():
                            mode_resolved = "mesh_bound"
                        else:
                            mode_resolved = "free"
                    else:
                        mode_resolved = mode
                else:
                    mode_resolved = mode


                if mode_resolved == "mesh_bound":
                    n = str(getattr(init_gs_ply, "name", "")).lower()
                    is_b_suffix = n.endswith("_b.ply")
                    if world_space == "B" and (not is_b_suffix) and ("in_a_world" in n or "world_a" in n):
                        logger.warning(
                            "[Stage33] world_space='B' but init_gs_ply filename looks like A-world. "
                            "This will likely cause silhouette translation vs Stage-13 GT after load_from_embedding(). "
                            f"init_gs_ply={init_gs_ply}"
                        )
                    if world_space == "A" and ("in_b_world" in n or "world_b" in n or is_b_suffix):
                        logger.warning(
                            "[Stage33] world_space='A' but init_gs_ply filename looks like B-world. "
                            f"init_gs_ply={init_gs_ply}"
                        )

                if skip_existing and _validate_gs_finetune_output(
                    splatting_project=splatting_project,
                    subject_name=seamfixed_subject,
                    model_prefix=model_prefix,
                    iteration=total_iteration,
                ):
                    logger.info(f"⏭ Skipping {seamfixed_subject} (output exists)")
                    results['successful'].append((head_id_raw, body_id_raw))
                    continue

                model_path = f"{seamfixed_subject}"
                cmd = python_prefix + [
                    'train_splatting_avatar.py',
                    '--configs', configs_str,
                    '--dat_dir', str(seamfixed_path),
                    '--ip', 'none',
                    '--total_iteration', str(total_iteration),
                    '--batch_size', str(batch_size),
                    '--bg_color', bg_color,
                    '--num_workers', str(num_workers),
                    '--model_path', model_path,
                ]

                if mode_resolved == "free":
                    cmd.extend(['--free_gaussians', '--init_gs_ply', str(init_gs_ply)])
                elif mode_resolved == "mesh_bound":
                    if init_embed is None or init_mesh is None:
                        raise RuntimeError("mesh_bound mode requires init_embedding_json_path/init_proxy_mesh_path (or inferable neck_bridge assets).")
                    if not init_embed.exists():
                        raise FileNotFoundError(f"Embedding json not found: {init_embed}")
                    if not init_mesh.exists():
                        raise FileNotFoundError(f"Proxy mesh not found: {init_mesh}")
                    cmd.extend([
                        '--static_proxy_mesh',
                        '--init_gs_ply', str(init_gs_ply),
                        '--init_gs_embed', str(init_embed),
                        '--init_cano_mesh', str(init_mesh),
                    ])
                else:
                    raise RuntimeError(f"Unhandled resolved mode: {mode_resolved}")

                if rgb_relpath:
                    cmd.extend(['--custom_rgb_relpath', rgb_relpath])
                if mask_relpath:
                    cmd.extend(['--custom_mask_relpath', mask_relpath])
                if only_update_neck:
                    cmd.extend(['--mask_render_in_loss'])

                if hybrid_enabled:
                    orig_swapped_dir = _resolve_original_swapped_dir_for_pair(
                        swapped_renders_root=swapped_renders_root,
                        head_id=head_id,
                        body_id=body_id,
                    )
                    cmd.extend([
                        "--hybrid_supervision",
                        "--hybrid_orig_dat_dir", str(orig_swapped_dir),
                        "--hybrid_orig_rgb_relpath", str(hybrid_cfg.get("orig_rgb_relpath", "0000.jpg")),
                        "--hybrid_refined_rgb_relpath", str(hybrid_cfg.get("refined_rgb_relpath", rgb_relpath or "0000.jpg")),
                        "--hybrid_changed_metric", str(hybrid_cfg.get("changed_metric", "l1")),
                        "--hybrid_changed_blur_ksize", str(hybrid_cfg.get("blur_ksize", 7)),
                        "--hybrid_changed_pool", str(hybrid_cfg.get("pool", 16)),
                        "--hybrid_changed_threshold", str(hybrid_cfg.get("threshold", 0.04)),
                        "--hybrid_changed_dilate_ksize", str(hybrid_cfg.get("dilate_ksize", 15)),
                        "--lambda_refined_inside", str(hybrid_cfg.get("lambda_refined_inside", 1.0)),
                        "--lambda_orig_outside", str(hybrid_cfg.get("lambda_orig_outside", 1.0)),
                        "--lambda_lpips_outside", str(hybrid_cfg.get("lambda_lpips_outside", 1.0)),
                    ])

                logger.info(f"🔥 Training {seamfixed_subject} (mode={mode_resolved})...")
                run_command(cmd, cwd=splatting_project, debug_port=None)
                results['successful'].append((head_id_raw, body_id_raw))
                logger.success(f"✓ Completed {head_id_raw}→{body_id_raw}")

            except Exception as e:
                logger.error(f"✗ Failed {head_id_raw}→{body_id_raw}: {e}")
                results['failed'].append((head_id_raw, body_id_raw, str(e)))
                continue

    logger.info(f"\n{'='*80}")
    logger.info("Stage 33: SplattingAvatar GS Fine-tune - COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"  ✓ Successful: {len(results['successful'])}/{pair_count} pairs")
    if results['failed']:
        logger.warning(f"  ✗ Failed: {len(results['failed'])}/{pair_count} pairs")
        for head, body, error in results['failed']:
            logger.warning(f"    - {head}→{body}: {error}")
    else:
        logger.success("  All pairs completed successfully!")
    logger.info(f"{'='*80}\n")
    return results


def stage_splatting_avatar_free_gaussian(config: dict, subjects: list, debug_subprocess: bool = False) -> dict:

    return stage_splatting_avatar_gs_finetune(
        config=config,
        subjects=subjects,
        debug_subprocess=debug_subprocess,
        _force_mode="free",
        _cfg_key="33_splatting_avatar_gs_finetune",
    )
