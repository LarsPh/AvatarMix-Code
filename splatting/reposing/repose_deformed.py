from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger

from dataset.dataset_helper import make_frameset_data
from model.splatting_avatar_model import SplattingAvatarModel
from model import libcore

from reposing.utils.ckpt_io import load_deform_checkpoint, infer_ref_frame_idx_from_pretrained_gs_path
from reposing.utils.ref_resolve import (
    resolve_reference_assets_from_pretrained_gs_from,
    apply_reference_assets_overrides_to_config,
)
from reposing.utils.video_utils import make_video

from utils.ply_io import save_ply_mesh
from utils.lbs_vis import lbs_weights_to_vertex_colors, scalar_to_heatmap_vertex_colors


def _load_canonical_mesh_from_embedding(pretrained_gs_from: Path) -> Dict[str, torch.Tensor]:

    embed_json = pretrained_gs_from / "embedding.json"
    if not embed_json.exists():
        raise FileNotFoundError(f"embedding.json not found under pretrained_gs_from: {embed_json}")
    import json as _json

    with open(embed_json, "r") as f:
        cc = _json.load(f)
    cano_rel = cc.get("cano_mesh", cc.get("mesh_fn"))
    if not cano_rel:
        raise ValueError(f"embedding.json missing cano_mesh/mesh_fn: {embed_json}")
    cano_path = embed_json.parent / cano_rel
    if not cano_path.exists():
        raise FileNotFoundError(f"Canonical mesh referenced by embedding.json not found: {cano_path}")

    cano_mesh_cpu = libcore.MeshCpu(str(cano_path))
    return {
        "mesh_verts": torch.tensor(cano_mesh_cpu.V).float(),
        "mesh_norms": torch.tensor(cano_mesh_cpu.N).float(),
        "mesh_faces": torch.tensor(cano_mesh_cpu.F).long(),
    }


def _load_canonical_mesh_from_embedding_json(embed_json: Path) -> Dict[str, torch.Tensor]:

    embed_json = Path(embed_json)
    if not embed_json.exists():
        raise FileNotFoundError(f"embedding json not found: {embed_json}")
    import json as _json

    with open(embed_json, "r") as f:
        cc = _json.load(f)
    cano_rel = cc.get("cano_mesh", cc.get("mesh_fn"))
    if not cano_rel:
        raise ValueError(f"embedding json missing cano_mesh/mesh_fn: {embed_json}")
    cano_path = embed_json.parent / str(cano_rel)
    if not cano_path.exists():
        raise FileNotFoundError(f"Canonical mesh referenced by embedding json not found: {cano_path}")
    cano_mesh_cpu = libcore.MeshCpu(str(cano_path))
    return {
        "mesh_verts": torch.tensor(cano_mesh_cpu.V).float(),
        "mesh_norms": torch.tensor(cano_mesh_cpu.N).float(),
        "mesh_faces": torch.tensor(cano_mesh_cpu.F).long(),
    }


def _load_lbs_weights_vj_from_npz(npz_path: Path) -> np.ndarray:

    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"combined LBS weights NPZ not found: {npz_path}")
    obj = np.load(str(npz_path), allow_pickle=True)
    keys = list(obj.keys())
    if not keys:
        raise ValueError(f"Empty NPZ: {npz_path}")
    preferred = ["lbs_weights", "W", "weights", "w", "lbs"]
    arr = None
    for k in preferred:
        if k in obj:
            arr = obj[k]
            break
    if arr is None:
        if len(keys) == 1:
            arr = obj[keys[0]]
        else:
            raise ValueError(f"Cannot infer LBS weights array from NPZ {npz_path}. keys={keys}")
    arr = np.asarray(arr).astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D LBS weights (V,J) or (J,V), got shape={arr.shape} from {npz_path}")

    if arr.shape[0] < 256 and arr.shape[1] > 1000:
        arr = arr.T
    return arr


def _extract_ref_params_from_npz(npz: dict, *, ref_idx: int, device: torch.device) -> Dict[str, torch.Tensor]:

    out: Dict[str, torch.Tensor] = {}
    keys = [
        "betas",
        "body_pose",
        "global_orient",
        "transl",
        "jaw_pose",
        "expression",
        "left_hand_pose",
        "right_hand_pose",
        "scale",
        "Rh",
        "Th",
        "v_shape",
        "v_pose",
    ]
    n = None
    if "body_pose" in npz:
        try:
            n = int(np.asarray(npz["body_pose"]).shape[0])
        except Exception:
            n = None
    idx = int(ref_idx)
    if n is not None:
        idx = max(0, min(idx, n - 1))
    for k in keys:
        if k not in npz:
            continue
        arr = np.asarray(npz[k])
        if k == "betas":
            if arr.ndim == 1:
                out[k] = torch.from_numpy(arr.astype(np.float32)).to(device)
            else:
                out[k] = torch.from_numpy((arr[0] if arr.shape[0] == 1 else arr[idx]).astype(np.float32)).to(device)
        elif k in ("v_shape", "v_pose") and arr.ndim == 3 and arr.shape[0] == 1:
            out[k] = torch.from_numpy(arr[0].astype(np.float32)).to(device)
        else:
            if arr.ndim <= 1:
                out[k] = torch.from_numpy(arr.astype(np.float32)).to(device)
            else:
                out[k] = torch.from_numpy(arr[idx].astype(np.float32)).to(device)
    return out


def _compute_global_align_mapping(
    *,
    orig_npz: dict,
    aligned_npz: dict,
    ref_idx: int,
    diff_thresh: float = 1e-6,
) -> Dict[str, object]:

    def _get_vec(npz_obj, key: str):
        if key not in npz_obj:
            return None
        a = np.asarray(npz_obj[key])
        if a.ndim >= 2:
            return a[int(ref_idx)]
        return a

    def _get_scale(npz_obj):
        if "scale" not in npz_obj:
            return None
        a = np.asarray(npz_obj["scale"])
        if a.ndim >= 2:
            a = a[int(ref_idx)]
        a = np.asarray(a).reshape(-1)
        if a.size == 0:
            return None
        return float(a[0])

    s0 = _get_scale(orig_npz)
    s1 = _get_scale(aligned_npz)
    if s0 is not None and s1 is not None and abs(float(s0)) > 0:
        s_map = float(s1) / float(s0)
    else:
        s_map = 1.0

    cand_fields = ["transl", "Th"]
    use_fields = []
    t_map = {}
    for f in cand_fields:
        v0 = _get_vec(orig_npz, f)
        v1 = _get_vec(aligned_npz, f)
        if v0 is None or v1 is None:
            continue
        v0 = np.asarray(v0, dtype=np.float32).reshape(-1)
        v1 = np.asarray(v1, dtype=np.float32).reshape(-1)
        if v0.size < 3 or v1.size < 3:
            continue
        d = float(np.linalg.norm(v1[:3] - v0[:3]))
        if d > float(diff_thresh):
            use_fields.append(f)
            t_map[f] = (v1[:3] - float(s_map) * v0[:3]).astype(np.float32)

    return {"s_map": float(s_map), "t_map": t_map, "use_fields": use_fields}


def _apply_global_align_mapping_to_params(p: Dict, mapping: Dict[str, object]) -> Dict:

    s_map = float(mapping.get("s_map", 1.0))
    t_map = mapping.get("t_map", {}) or {}
    use_fields = set(mapping.get("use_fields", []) or [])
    out: Dict = {}
    for k, v in p.items():
        out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v

    if "scale" in out and isinstance(out["scale"], torch.Tensor) and abs(s_map - 1.0) > 1e-12:
        out["scale"] = out["scale"].float() * float(s_map)

    for f in use_fields:
        if f not in out or not isinstance(out[f], torch.Tensor):
            continue
        t = torch.as_tensor(np.asarray(t_map.get(f), dtype=np.float32), device=out[f].device, dtype=torch.float32)
        x = out[f].float().reshape(-1)
        if x.numel() >= 3:
            x3 = x[:3] * float(s_map) + t
            x = torch.cat([x3, x[3:]], dim=0) if x.numel() > 3 else x3
            out[f] = x.reshape(out[f].shape).to(dtype=out[f].dtype)
    return out


def _group_indices_by_frame(frameset) -> Dict[str, List[int]]:

    samples = getattr(frameset, "data_samples", None)
    if samples is None:
        raise RuntimeError("Expected frameset.data_samples for AvatarRexDataset-like frameset.")
    frame_to_idxs: Dict[str, List[int]] = {}
    for idx, (frame_id_str, _cam_id) in enumerate(samples):
        frame_to_idxs.setdefault(str(frame_id_str), []).append(int(idx))
    return frame_to_idxs


def _sorted_frames(frame_to_idxs: Dict[str, List[int]]) -> List[str]:

    return sorted(frame_to_idxs.keys(), key=lambda s: int(s))


def repose_gs_on_mesh_deformed(args, config) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() and not getattr(args, "cpu", False) else "cpu")
    logger.info(f"[DeformRepose] Using device: {device}")

    deform_ckpt = getattr(args, "deform_ckpt", None)
    if not deform_ckpt:
        raise ValueError("--deform_ckpt is required for deformation checkpoint testing.")


    merged_cfg = None
    try:
        merged_cfg = getattr(config.model, "merged_assets", None)
    except Exception:
        merged_cfg = None
    try:
        merged_enabled = bool(getattr(merged_cfg, "enabled", merged_cfg.get("enabled", False))) if merged_cfg is not None else False
    except Exception:
        merged_enabled = False


    pretrained_gs_from_spec = getattr(args, "pretrained_gs_from", None) or getattr(getattr(config, "model", {}), "load_pretrained_gs_from", None)
    if (not pretrained_gs_from_spec) and (not merged_enabled):
        raise ValueError("Missing pretrained GS root. Provide --pretrained_gs_from or set config.model.load_pretrained_gs_from.")
    pretrained_gs_from = Path(str(pretrained_gs_from_spec)) if pretrained_gs_from_spec else None
    if pretrained_gs_from is not None and (not pretrained_gs_from.exists()):
        raise FileNotFoundError(f"pretrained_gs_from does not exist: {pretrained_gs_from}")


    config.use_deformation = True
    config.model.use_deformation = True

    config.dataset.use_deformation = (not merged_enabled)
    if pretrained_gs_from is not None:
        config.model.load_pretrained_gs_from = str(pretrained_gs_from)


    try:
        avcfg = getattr(config.dataset, "avatarrex_config", {})
    except Exception:
        avcfg = {}

    def _set_if_missing(obj, key, val):
        if val is None:
            return
        try:
            if key not in obj:
                obj[key] = val
        except Exception:
            if not hasattr(obj, key):
                setattr(obj, key, val)

    def _set_or_override(obj, key, val):

        if val is None:
            return
        try:
            has_key = (key in obj)
        except Exception:
            has_key = hasattr(obj, key)
        if has_key:
            try:
                cur = obj.get(key, None) if hasattr(obj, "get") else getattr(obj, key, None)
            except Exception:
                cur = getattr(obj, key, None)

            if cur != val:
                logger.warning(f"[DeformRepose] Overriding config.model.{key}={cur!r} -> {val!r} to match dataset.avatarrex_config.")
                try:
                    obj[key] = val
                except Exception:
                    setattr(obj, key, val)
            return
        _set_if_missing(obj, key, val)


    _set_or_override(config.model, "smpl_model_type", getattr(avcfg, "model_type", None))
    _set_or_override(config.model, "smpl_gender", getattr(avcfg, "gender", None))
    _set_or_override(config.model, "smpl_model_path", getattr(avcfg, "smpl_model_path", None))
    _set_or_override(
        config.model,
        "smpl_use_pca",
        bool(getattr(avcfg, "smpl_use_pca", False)) if hasattr(avcfg, "smpl_use_pca") else None,
    )
    _set_or_override(
        config.model,
        "smpl_num_pca_comps",
        int(getattr(avcfg, "smpl_num_pca_comps", 6)) if hasattr(avcfg, "smpl_num_pca_comps") else None,
    )
    _set_or_override(
        config.model,
        "smpl_flat_hand_mean",
        bool(getattr(avcfg, "flat_hand_mean", False)) if hasattr(avcfg, "flat_hand_mean") else None,
    )


    sampling_cfg = getattr(avcfg, "sampling", {})
    dataset_flavor = sampling_cfg.get("dataset_flavor", "generic") if sampling_cfg else "generic"
    if dataset_flavor == "actorshq":
        _set_if_missing(config.model, "smpl_global_transform_mode", "external")


    if (not merged_enabled) and (pretrained_gs_from is not None):
        assets = resolve_reference_assets_from_pretrained_gs_from(str(pretrained_gs_from))
        apply_reference_assets_overrides_to_config(config, assets)


    ckpt = load_deform_checkpoint(str(deform_ckpt), map_location="cpu")
    if ckpt.latent_num_embeddings is None:
        logger.warning("[DeformRepose] Could not infer latent embedding size from checkpoint. "
                       "Will fall back to a conservative default (1).")
        num_training_frames_ckpt = 1
    else:
        num_training_frames_ckpt = int(ckpt.latent_num_embeddings)


    latent_override = getattr(args, "latent_frame_idx", None)
    if latent_override is not None:
        fixed_latent_idx = int(latent_override)
    else:
        ref_idx = infer_ref_frame_idx_from_pretrained_gs_path(str(pretrained_gs_from)) if pretrained_gs_from is not None else None
        if ref_idx is None:

            try:
                ref_idx = int(getattr(getattr(config.dataset, "avatarrex_config", {}), "reference_frame_idx_for_smpl", 0))
            except Exception:
                ref_idx = 0
        fixed_latent_idx = int(ref_idx) if ref_idx is not None else 0
    if fixed_latent_idx < 0:
        fixed_latent_idx = 0
    if fixed_latent_idx >= num_training_frames_ckpt:
        logger.warning(
            f"[DeformRepose] fixed latent idx {fixed_latent_idx} >= num_latents {num_training_frames_ckpt}. "
            f"Clamping to {num_training_frames_ckpt - 1}."
        )
        fixed_latent_idx = max(0, num_training_frames_ckpt - 1)


    split = getattr(args, "split", "test")
    config.dataset.dat_dir = args.dat_dir
    frameset = make_frameset_data(config.dataset, split=split)

    smpl_global_mapping = None
    dump_aligned_smplx_meshes = False
    merged_world = "B"

    if merged_enabled:

        try:
            merged_root_dir = Path(str(getattr(merged_cfg, "root_dir", merged_cfg.get("root_dir", "")))).expanduser()
        except Exception:
            merged_root_dir = Path("")
        if not merged_root_dir or (not merged_root_dir.exists()):
            raise FileNotFoundError(f"[DeformRepose] merged_assets.enabled but merged_assets.root_dir not found: {merged_root_dir}")
        try:
            merged_world = str(getattr(merged_cfg, "world_space", merged_cfg.get("world_space", "B"))).strip().upper() or "B"
        except Exception:
            merged_world = "B"
        if merged_world not in ("A", "B"):
            merged_world = "B"

        default_embed_rel = f"neck_bridge/{merged_world}/embedding_composed.json"
        default_lbs_rel = f"neck_bridge/{merged_world}/combined_lbs_weights.npz"
        default_aligned_npz_rel = (
            "smpl_params_B_body_donor_aligned_world_B_body_donor.npz" if merged_world == "B"
            else "smpl_params_B_body_donor_aligned_world_A.npz"
        )
        try:
            embed_rel = str(getattr(merged_cfg, "embed_relpath", merged_cfg.get("embed_relpath", default_embed_rel))) or default_embed_rel
        except Exception:
            embed_rel = default_embed_rel
        try:
            lbs_rel = str(getattr(merged_cfg, "lbs_relpath", merged_cfg.get("lbs_relpath", default_lbs_rel))) or default_lbs_rel
        except Exception:
            lbs_rel = default_lbs_rel
        try:
            aligned_rel = str(getattr(merged_cfg, "aligned_smpl_npz_relpath", merged_cfg.get("aligned_smpl_npz_relpath", default_aligned_npz_rel))) or default_aligned_npz_rel
        except Exception:
            aligned_rel = default_aligned_npz_rel
        try:
            gs_glob = str(getattr(merged_cfg, "gs_ply_glob", merged_cfg.get("gs_ply_glob", ""))).strip()
        except Exception:
            gs_glob = ""
        if not gs_glob:
            raise ValueError("[DeformRepose] merged_assets.gs_ply_glob is required (world-aware).")

        merged_embed_json = (merged_root_dir / embed_rel).resolve()
        merged_lbs_npz = (merged_root_dir / lbs_rel).resolve()
        merged_aligned_npz = (merged_root_dir / aligned_rel).resolve()

        if not merged_embed_json.exists():
            raise FileNotFoundError(f"[DeformRepose] composed embedding json not found: {merged_embed_json}")
        if not merged_lbs_npz.exists():
            raise FileNotFoundError(f"[DeformRepose] combined LBS NPZ not found: {merged_lbs_npz}")
        if not merged_aligned_npz.exists():
            raise FileNotFoundError(f"[DeformRepose] aligned SMPL NPZ not found: {merged_aligned_npz}")

        candidates = sorted([p for p in merged_root_dir.rglob(gs_glob) if p.is_file()])
        if len(candidates) != 1:
            preview = "\n".join([f"  - {str(p)}" for p in candidates[:10]])
            raise RuntimeError(
                f"[DeformRepose] merged_assets.gs_ply_glob={gs_glob!r} matched {len(candidates)} files under {merged_root_dir}.\n"
                f"{preview}\n"
                f"Refine the glob so it matches exactly 1 file."
            )
        merged_gs_ply = candidates[0].resolve()


        try:
            config.model.load_pretrained_gs_embed_json = str(merged_embed_json)
            config.model.load_pretrained_gs_ply = str(merged_gs_ply)
        except Exception:
            try:
                config.model["load_pretrained_gs_embed_json"] = str(merged_embed_json)
                config.model["load_pretrained_gs_ply"] = str(merged_gs_ply)
            except Exception:
                pass


        try:
            deform_cfg_m = getattr(config.model, "deformation", None)
            if deform_cfg_m is not None:
                qcfg_m = getattr(deform_cfg_m, "xyz_query_override", None) if not isinstance(deform_cfg_m, dict) else deform_cfg_m.get("xyz_query_override", None)
                if qcfg_m is not None:
                    if isinstance(qcfg_m, dict):
                        qcfg_m["enabled"] = False
                    else:
                        setattr(qcfg_m, "enabled", False)
        except Exception:
            pass


        cano_mesh_data_for_model = _load_canonical_mesh_from_embedding_json(merged_embed_json)


        W_vj = _load_lbs_weights_vj_from_npz(merged_lbs_npz)
        lbs_weights_for_ref_mesh = torch.from_numpy(W_vj).to(device=device)


        try:
            ref_idx_for_smpl = int(getattr(getattr(config.dataset, "avatarrex_config", {}), "reference_frame_idx_for_smpl", 0))
        except Exception:
            ref_idx_for_smpl = 0
        aligned_npz = dict(np.load(str(merged_aligned_npz), allow_pickle=True))
        ref_frame_smpl_params_for_lbs = _extract_ref_params_from_npz(aligned_npz, ref_idx=ref_idx_for_smpl, device=device)


        orig_seq_npz_path = Path(str(args.dat_dir)) / "smpl_params.npz"
        orig_npz = dict(np.load(str(orig_seq_npz_path), allow_pickle=True))
        smpl_global_mapping = _compute_global_align_mapping(orig_npz=orig_npz, aligned_npz=aligned_npz, ref_idx=ref_idx_for_smpl)
        logger.info(
            f"[DeformRepose] SMPL global align mapping (world={merged_world}) at ref_idx={ref_idx_for_smpl}: "
            f"s_map={smpl_global_mapping.get('s_map')} use_fields={smpl_global_mapping.get('use_fields')}"
        )


        garment_mask_ref = None


        try:
            dump_aligned_smplx_meshes = bool(getattr(merged_cfg, "dump_aligned_smplx_meshes", merged_cfg.get("dump_aligned_smplx_meshes", True)))
        except Exception:
            dump_aligned_smplx_meshes = False
    else:
        ref_frame_smpl_params_for_lbs = getattr(frameset, "ref_frame_smpl_params", None)
        lbs_weights_for_ref_mesh = getattr(frameset, "lbs_weights_for_ref_mesh", None)
        if ref_frame_smpl_params_for_lbs is None or lbs_weights_for_ref_mesh is None:
            raise RuntimeError(
                "[DeformRepose] Dataset did not provide reference assets required for deformation. "
                "Expected frameset.ref_frame_smpl_params and frameset.lbs_weights_for_ref_mesh."
            )
        if pretrained_gs_from is None:
            raise RuntimeError("[DeformRepose] pretrained_gs_from is required in legacy mode.")
        cano_mesh_data_for_model = _load_canonical_mesh_from_embedding(pretrained_gs_from)
        garment_mask_ref = getattr(frameset, "ref_frame_garment_mask", None)

    pipe = config.pipe


    gs_model = SplattingAvatarModel(
        config.model,
        device=device,
        verbose=True,
        num_training_frames=num_training_frames_ckpt,
        ref_frame_smpl_params_for_lbs=ref_frame_smpl_params_for_lbs,
        lbs_weights_for_ref_mesh=lbs_weights_for_ref_mesh,
        static_rendering=False,
    )
    gs_model.create_from_canonical(cano_mesh_data_for_model, garment_mask_ref_frame=garment_mask_ref)


    _model_sd = gs_model.state_dict()
    _ckpt_sd_raw = ckpt.gs_model_state_dict
    _ckpt_sd = {}
    _skipped = []
    for _k, _v in _ckpt_sd_raw.items():


        if _k.startswith("smpl_model_for_lbs."):
            _skipped.append(_k)
            continue
        if _k.startswith("quat_helper.") or _k.startswith("phongsurf."):
            _skipped.append(_k)
            continue
        if _k not in _model_sd:
            _skipped.append(_k)
            continue
        try:
            _mv = _model_sd[_k]
            if isinstance(_v, torch.Tensor) and isinstance(_mv, torch.Tensor) and tuple(_v.shape) == tuple(_mv.shape):
                _ckpt_sd[_k] = _v
            else:
                _skipped.append(_k)
        except Exception:
            _skipped.append(_k)
    incompatible = gs_model.load_state_dict(_ckpt_sd, strict=False)
    if _skipped:
        logger.info(f"[DeformRepose] Filtered checkpoint state_dict: loaded={len(_ckpt_sd)} skipped={len(_skipped)} (shape/key mismatch).")
    if hasattr(incompatible, "missing_keys") or hasattr(incompatible, "unexpected_keys"):
        logger.info(f"[DeformRepose] Loaded checkpoint with strict=False. "
                    f"missing={len(getattr(incompatible, 'missing_keys', []))} "
                    f"unexpected={len(getattr(incompatible, 'unexpected_keys', []))}")


    try:
        deform_cfg = config.model.deformation
    except Exception:
        deform_cfg = getattr(config.model, "deformation", {})
    try:
        input_mode = str(getattr(deform_cfg, "input_mode", deform_cfg.get("input_mode", "xyz"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "xyz"
    except Exception:
        input_mode = "xyz"
    try:
        qcfg = getattr(deform_cfg, "xyz_query_override", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("xyz_query_override", None)
        q_enabled = bool(getattr(qcfg, "enabled", qcfg.get("enabled", False))) if qcfg is not None else False
    except Exception:
        q_enabled = False
    if (not merged_enabled) and input_mode == "xyz" and q_enabled and hasattr(gs_model, "maybe_init_xyz_query_override_buffers"):
        try:
            gs_model.maybe_init_xyz_query_override_buffers(force=True)
            logger.info("[DeformRepose] Re-initialized xyz_query_override buffers (force=True).")
        except Exception as e:
            logger.warning(f"[DeformRepose] Failed to re-init xyz_query_override buffers: {e}")


    output_dir_base = Path(args.output_dir)
    if output_dir_base.name == "repose_deformed":
        output_dir_base = output_dir_base.parent / "repose_deformed_test_clothfit"
        logger.info(f"[DeformRepose] Redirecting output base to: {output_dir_base}")
    output_dir_base.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(deform_ckpt)
    exp_name = ckpt_path.parent.name if ckpt_path.parent.name else "unknown_exp"

    iter_suffix = ""
    ckpt_stem = ckpt_path.stem
    if "iter_" in ckpt_stem:
        try:
            iter_part = ckpt_stem.split("iter_")[-1]
            iter_suffix = f"_{iter_part}"
        except Exception:
            pass
    out_dir_renders = output_dir_base / f"{exp_name}{iter_suffix}_{split}"
    out_dir_renders.mkdir(parents=True, exist_ok=True)

    out_dir_reposed_gs_ply = None
    if getattr(args, "save_reposed_gs_ply", False):
        out_dir_reposed_gs_ply = out_dir_renders / "reposed_gaussians_ply"
        out_dir_reposed_gs_ply.mkdir(parents=True, exist_ok=True)
        logger.info(f"[DeformRepose] Saving reposed GS PLYs to: {out_dir_reposed_gs_ply}")


    variant_first_k = int(getattr(args, "variant_ply_first_k", 0) or 0)
    save_variant_plys = bool(getattr(args, "save_variant_plys", False))

    if (not save_variant_plys) and (variant_first_k > 0):
        save_variant_plys = True
        logger.info(f"[DeformRepose] Enabling --save_variant_plys because --variant_ply_first_k={variant_first_k} > 0.")
    out_dir_variants = None
    if save_variant_plys:
        out_dir_variants = out_dir_renders
        out_dir_variants.mkdir(parents=True, exist_ok=True)
        logger.info(f"[DeformRepose] Saving variant mesh/GS PLYs to: {out_dir_variants}")


    frame_to_idxs = _group_indices_by_frame(frameset)
    frames = _sorted_frames(frame_to_idxs)
    frame_stride = int(getattr(args, "frame_stride", 1) or 1)
    max_frames = getattr(args, "max_frames", None)
    if max_frames is not None:
        max_frames = int(max_frames)

    frames = frames[::frame_stride]
    if max_frames is not None:
        frames = frames[:max_frames]
    if not frames:
        logger.warning("[DeformRepose] No frames selected. Nothing to do.")
        return


    if merged_enabled and dump_aligned_smplx_meshes:
        try:
            faces_np = getattr(getattr(gs_model, "smpl_model_for_lbs", None), "faces", None)
            if faces_np is None:
                raise RuntimeError("smpl_model_for_lbs.faces not available")
            faces_smpl = torch.as_tensor(faces_np, dtype=torch.long, device=device)

            def _clone_disable(p: Dict) -> Dict:
                out: Dict = {}
                for k, v in p.items():
                    out[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
                if bool(getattr(args, "disable_finger_movement", False)):
                    if isinstance(out.get("left_hand_pose", None), torch.Tensor):
                        out["left_hand_pose"] = torch.zeros_like(out["left_hand_pose"])
                    if isinstance(out.get("right_hand_pose", None), torch.Tensor):
                        out["right_hand_pose"] = torch.zeros_like(out["right_hand_pose"])
                return out

            with torch.no_grad():
                v_ref = gs_model._smplx_vertices_world_for_anchor(ref_frame_smpl_params_for_lbs)
            save_ply_mesh(out_dir_renders / "smplx_ref_aligned.ply", v_ref, faces_smpl)

            frame0 = frames[0]
            idxs0 = frame_to_idxs.get(frame0, [])
            if idxs0:
                sample0 = frameset[idxs0[0]]
                mesh_info0 = sample0.get("mesh_info", {})
                p0 = mesh_info0.get("smplx_params_raw_for_deformnet", None)
                if p0 is not None:
                    p0 = _clone_disable(p0)
                    if smpl_global_mapping is not None:
                        p0 = _apply_global_align_mapping_to_params(p0, smpl_global_mapping)
                    with torch.no_grad():
                        v_tgt = gs_model._smplx_vertices_world_for_anchor(p0)
                    save_ply_mesh(out_dir_renders / "smplx_target_aligned_first.ply", v_tgt, faces_smpl)
            logger.info(f"[DeformRepose] Dumped aligned SMPL-X PLYs to: {out_dir_renders}")
        except Exception as e:
            logger.warning(f"[DeformRepose] Failed to dump aligned SMPL-X meshes: {e}")


    bg = torch.tensor(getattr(args, "render_bg_color", [1.0, 1.0, 1.0]), dtype=torch.float32, device=device)

    _printed_xyz_query_override_stats = 0

    _logged_disable_fingers = False

    def _clone_and_maybe_disable_fingers(p: Dict) -> Dict:

        nonlocal _logged_disable_fingers
        out: Dict = {}
        for k, v in p.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.detach().clone()
            else:
                out[k] = v
        if not bool(getattr(args, "disable_finger_movement", False)):
            return out
        lh = out.get("left_hand_pose", None)
        rh = out.get("right_hand_pose", None)
        if isinstance(lh, torch.Tensor):
            out["left_hand_pose"] = torch.zeros_like(lh)
        if isinstance(rh, torch.Tensor):
            out["right_hand_pose"] = torch.zeros_like(rh)
        if not _logged_disable_fingers:
            lh_shape = tuple(lh.shape) if isinstance(lh, torch.Tensor) else None
            rh_shape = tuple(rh.shape) if isinstance(rh, torch.Tensor) else None
            logger.info(f"[DeformRepose] disable_finger_movement: zeroing left/right_hand_pose (shapes: {lh_shape}, {rh_shape}).")
            _logged_disable_fingers = True
        return out

    for frame_id_str in frames:
        idxs = frame_to_idxs.get(frame_id_str, [])
        if not idxs:
            continue


        sample0 = frameset[idxs[0]]
        mesh_info = sample0.get("mesh_info")
        if mesh_info is None or "smplx_params_raw_for_deformnet" not in mesh_info:
            logger.warning(f"[DeformRepose] Missing mesh_info/smplx_params for frame {frame_id_str}. Skipping.")
            continue

        smplx_params_dict = mesh_info["smplx_params_raw_for_deformnet"]
        smplx_params_dict_for_frame = _clone_and_maybe_disable_fingers(smplx_params_dict)
        if merged_enabled and (smpl_global_mapping is not None):
            try:
                smplx_params_dict_for_frame = _apply_global_align_mapping_to_params(smplx_params_dict_for_frame, smpl_global_mapping)
            except Exception as e:
                logger.warning(f"[DeformRepose] Failed to apply SMPL global mapping on frame {frame_id_str}: {e}")
        frm_idx_int = int(frame_id_str)


        gs_model.update_to_posed_mesh(
            raw_mesh_info_for_no_deform=mesh_info,
            current_target_frame_smpl_params=smplx_params_dict_for_frame,
            current_frame_idx=frm_idx_int,
            latent_frame_idx_override=fixed_latent_idx,
            apply_vertex_offsets=True,
            apply_gauss_bs=(not merged_enabled),
        )


        if (not merged_enabled) and (_printed_xyz_query_override_stats < 5):
            try:
                deform_cfg = getattr(config.model, "deformation", {})
                qcfg = getattr(deform_cfg, "xyz_query_override", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("xyz_query_override", None)
                q_enabled = bool(getattr(qcfg, "enabled", qcfg.get("enabled", False))) if qcfg is not None else False
                q_log = bool(getattr(qcfg, "log_stats", qcfg.get("log_stats", False))) if qcfg is not None else False
                q_cmp = bool(getattr(qcfg, "compare_to_target_query", qcfg.get("compare_to_target_query", False))) if qcfg is not None else False
            except Exception:
                q_enabled, q_log, q_cmp = False, False, False
            if q_enabled and q_log and q_cmp:
                try:
                    diff_mean = getattr(gs_model, "_xyz_query_override_offset_diff_l2_mean", None)
                    diff_max = getattr(gs_model, "_xyz_query_override_offset_diff_l2_max", None)
                    off_query = getattr(gs_model, "_last_vertex_offset_l2_mean_query", None)
                    off_tgt = getattr(gs_model, "_xyz_query_override_offset_target_l2_mean", None)
                    if (diff_mean is not None) or (diff_max is not None):
                        logger.info(
                            f"[DeformRepose] xyz_query_override A/B (frame {frm_idx_int:05d}): "
                            f"offset_l2_mean(query)={off_query} offset_l2_mean(target)={off_tgt} "
                            f"diff_l2_mean={diff_mean} diff_l2_max={diff_max}"
                        )
                        _printed_xyz_query_override_stats += 1
                except Exception:
                    pass


        if out_dir_reposed_gs_ply is not None:
            ply_path = out_dir_reposed_gs_ply / f"reposed_gs_frame{frm_idx_int:05d}_latent{fixed_latent_idx:05d}.ply"
            if getattr(args, "force_update_all", False) or getattr(args, "update_reposed_gs_ply", False) or (not ply_path.exists()):
                gs_model.save_ply(str(ply_path))


        if out_dir_variants is not None:
            try:
                dumped = int(getattr(repose_gs_on_mesh_deformed, "_variant_dump_count", 0))

                do_dump = (variant_first_k <= 0) or (dumped < variant_first_k)
                if do_dump and variant_first_k > 0:
                    setattr(repose_gs_on_mesh_deformed, "_variant_dump_count", dumped + 1)
            except Exception:
                do_dump = True

            if do_dump:
                faces = getattr(gs_model, "cano_faces", None)
                variants = [("raw_lbs", False, False), ("offset", True, False)]
                if not merged_enabled:
                    variants.append(("full", True, True))
                for vname, v_apply_vert, v_apply_bs in variants:
                    gs_model.update_to_posed_mesh(
                        raw_mesh_info_for_no_deform=mesh_info,
                        current_target_frame_smpl_params=smplx_params_dict_for_frame,
                        current_frame_idx=frm_idx_int,
                        latent_frame_idx_override=fixed_latent_idx,
                        apply_vertex_offsets=v_apply_vert,
                        apply_gauss_bs=(v_apply_bs and (not merged_enabled)),
                    )

                    if vname in ("raw_lbs", "offset") and getattr(gs_model, "mesh_verts", None) is not None and faces is not None:
                        colors = None
                        try:

                            w_vis = gs_model.get_lbs_weights_for_skinning() if hasattr(gs_model, "get_lbs_weights_for_skinning") else getattr(gs_model, "lbs_weights", None)
                            if w_vis is not None:
                                lbs_mode = "argmax"
                                try:
                                    lbs_mode = str(getattr(getattr(config.model, "mesh_vis", None), "lbs_color_mode", "argmax"))
                                except Exception:
                                    lbs_mode = "argmax"
                                colors = lbs_weights_to_vertex_colors(w_vis, mode=lbs_mode)
                        except Exception:
                            colors = None

                        mesh_p = out_dir_variants / f"frame_{frm_idx_int:05d}_{vname}_mesh_lbs.ply"
                        if getattr(args, "force_update_all", False) or getattr(args, "update_meshes", False) or (not mesh_p.exists()):
                            save_ply_mesh(mesh_p, gs_model.mesh_verts, faces, vert_colors=colors)


                        try:
                            mod = getattr(gs_model, "lbs_weight_residual", None)
                            if mod is not None:
                                metric = "l1"
                                try:
                                    lbs_cfg = getattr(config.model, "lbs_weight_residual", None)
                                    vis_cfg = getattr(lbs_cfg, "vis", None) if lbs_cfg is not None else None
                                    metric = str(getattr(vis_cfg, "delta_norm_metric", "l1")) if vis_cfg is not None else "l1"
                                except Exception:
                                    metric = "l1"
                                q = mod.delta_norm_per_vertex(global_step=None, metric=metric, baseline="projected")
                                q_colors = scalar_to_heatmap_vertex_colors(q)
                                q_p = out_dir_variants / f"frame_{frm_idx_int:05d}_{vname}_mesh_delta_norm.ply"
                                if getattr(args, "force_update_all", False) or getattr(args, "update_meshes", False) or (not q_p.exists()):
                                    save_ply_mesh(q_p, gs_model.mesh_verts, faces, vert_colors=q_colors, vert_quality=q)
                        except Exception:
                            pass


                    gs_p = out_dir_variants / f"frame_{frm_idx_int:05d}_{vname}_gs.ply"
                    if getattr(args, "force_update_all", False) or getattr(args, "update_meshes", False) or (not gs_p.exists()):
                        try:
                            gs_model.save_ply(str(gs_p))
                        except Exception:
                            pass


        for idx in idxs:
            sample = frameset[idx]
            cam_id = str(sample.get("cam_id_str", "cam"))
            scene_cams = sample.get("scene_cameras", [])
            if not scene_cams:
                continue
            viewpoint_cam = scene_cams[0].to(device)

            out_dir_cam = out_dir_renders / cam_id
            if not getattr(args, "disable_renders", False):
                out_dir_cam.mkdir(parents=True, exist_ok=True)

            if getattr(args, "disable_renders", False):
                continue

            render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=bg)
            image = render_pkg["render"]

            out_path = out_dir_cam / f"frame_{frm_idx_int:05d}.jpg"
            if getattr(args, "force_update_all", False) or getattr(args, "update_renders", False) or (not out_path.exists()):
                libcore.write_tensor_image(str(out_path), image, rgb2bgr=True)


    video_fps_default = 12

    for cam_dir in out_dir_renders.iterdir():
        if not cam_dir.is_dir():
            continue
        cam_id = cam_dir.name
        frames_for_cam = list(cam_dir.glob("frame_*.jpg"))
        if not frames_for_cam:
            continue
        video_name_base = f"{cam_id}_{exp_name}{iter_suffix}"
        video_path = cam_dir / f"{video_name_base}.mp4"
        if getattr(args, "force_update_all", False) or getattr(args, "update_video", False) or not video_path.exists():
            make_video(str(cam_dir), video_name_base, frame_pattern="frame_*.jpg", frame_rate=video_fps_default)
        else:
            logger.info(f"[DeformRepose] Video {video_path} already exists. Skipping generation.")

    logger.info(f"[DeformRepose] Done. Outputs at: {out_dir_renders}")
