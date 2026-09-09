import os
import sys # Add sys import
from pathlib import Path
import torch 
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf
from argparse import ArgumentParser # Keep for argument parsing
from loguru import logger

# --- Add project root to sys.path for robust imports ---
# This assumes train_splatting_avatar.py is in the project root (SplattingAvatar/)
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End of sys.path modification ---

from model.splatting_avatar_model import SplattingAvatarModel
from model.splatting_avatar_optim import SplattingAvatarOptimizer
from model.loss_base import run_testing, run_temporal_window_validation
from dataset.dataset_helper import make_frameset_data, make_dataloader # Will use AvatarRexDataset via config
from model import libcore
from utils.general_utils import inverse_sigmoid
from utils.sh_utils import RGB2SH
from dataset.sequence_sampling import MixedDivWinSampler, MixedSamplerConfig, save_sampling_artifacts
from utils.pose_features import DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES
import numpy as np
from PIL import Image
import torch.nn.functional as F
import pytorch3d.io
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.loss.point_mesh_distance import point_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from torch.utils.tensorboard import SummaryWriter
from PIL import Image as PILImage
from utils.lbs_vis import lbs_weights_to_vertex_colors
from utils.ply_io import save_ply_mesh
from utils.lbs_vis import scalar_to_heatmap_vertex_colors
from dataset.geo_ft_cache import (
    GeoMeshGtCache,
    discover_talkbody4d_geo_frames,
    build_pair_to_dataset_index,
)
from reposing.utils.ckpt_io import load_deform_checkpoint

if __name__ == '__main__':
    parser = ArgumentParser(description='SplattingAvatar Training with Deformation Network')
    parser.add_argument('--ip', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--dat_dir', type=str, required=False, help="Root directory of the dataset.")
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], 
                        required=True, help='Path(s) to config file(s), semicolon-separated.')
    parser.add_argument('--model_path', type=str, default=None,
                         help="Custom experiment name suffix for output folder.")
    parser.add_argument('--total_iteration', type=int, default=-1, help="Total number of iterations.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size.")
    parser.add_argument('--bg_color', type=str, default='random', choices=['random', 'black', 'white'], help='Background color')
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument('--data_device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use for data loading.')
    parser.add_argument('--free_gaussians', action='store_true', help='Train standard 3DGS without mesh/deformation (no mesh dependency).')
    parser.add_argument('--init_gs_ply', type=str, default=None, help='Path to initial Gaussians PLY for free-gaussians mode (trainable).')
    # Mesh-bound (static proxy) fine-tune mode: still uses mesh embedding, but does NOT require per-frame meshes from dataset.
    parser.add_argument('--static_proxy_mesh', action='store_true', help='Enable static proxy-mesh mode (no per-frame mesh dependency; Gaussians remain mesh-bound).')
    parser.add_argument('--init_gs_embed', type=str, default=None, help='Path to embedding.json for mesh-bound Gaussians (sample_fidxs/bary + _xyz/_rotation).')
    parser.add_argument('--init_cano_mesh', type=str, default=None, help='Optional explicit canonical mesh path (ply/obj) for mesh-bound mode. If omitted, resolves from embedding.json cano_mesh.')
    parser.add_argument('--custom_rgb_relpath', type=str, default=None, help='Custom relative path under subject root to RGB file(s). Supports {frame} and {cam}.')
    parser.add_argument('--custom_mask_relpath', type=str, default=None, help='Custom relative path under subject root to mask file(s). Supports {frame} and {cam}.')
    parser.add_argument('--mask_render_in_loss', action='store_true', help='Apply alpha mask to render when computing loss.')
    # Hybrid supervision (Stage 33): refined inside changed-region mask, original outside.
    parser.add_argument('--hybrid_supervision', action='store_true', help='Enable hybrid supervision (refined inside M, original outside M).')
    parser.add_argument('--hybrid_orig_dat_dir', type=str, default=None, help='Absolute path to the original swapped renders directory (contains camera subdirs).')
    parser.add_argument('--hybrid_orig_rgb_relpath', type=str, default="0000.jpg", help='RGB relpath under each camera dir for original GT (supports {frame} and {cam}).')
    parser.add_argument('--hybrid_refined_rgb_relpath', type=str, default=None, help='RGB relpath under each camera dir for refined GT (supports {frame} and {cam}). Defaults to --custom_rgb_relpath.')
    parser.add_argument('--hybrid_changed_metric', type=str, default="l1", choices=["l1", "l2"], help='Metric for changed-region mask (computed between refined and original).')
    parser.add_argument('--hybrid_changed_blur_ksize', type=int, default=7, help='Blur kernel size for diff map smoothing (odd int).')
    parser.add_argument('--hybrid_changed_pool', type=int, default=16, help='Average-pooling window to capture large changed regions.')
    parser.add_argument('--hybrid_changed_threshold', type=float, default=0.04, help='Threshold on pooled diff to define changed region.')
    parser.add_argument('--hybrid_changed_dilate_ksize', type=int, default=15, help='Dilation kernel size for changed mask (odd int).')
    parser.add_argument('--lambda_refined_inside', type=float, default=1.0, help='RGB loss weight inside changed region (toward refined).')
    parser.add_argument('--lambda_orig_outside', type=float, default=1.0, help='RGB loss weight outside changed region (toward original).')
    parser.add_argument('--lambda_lpips_outside', type=float, default=1.0, help='LPIPS weight outside changed region (toward original).')

    # Debug visualization control for checkpoint images under point_cloud/iteration_*/{gt,render}_train.jpg
    parser.add_argument('--debug_vis_split', type=str, default="train", choices=["train", "test"], help="Which split to use for debug checkpoint images.")
    parser.add_argument('--debug_vis_frame', type=str, default=None, help="Frame id for debug vis (e.g. '00002000' or '2000'). If omitted, uses first sample.")
    parser.add_argument('--debug_vis_cam', type=str, default=None, help="Camera id for debug vis (e.g. '005') or 1-based camera index among all cameras (e.g. '126').")
    args, extras = parser.parse_known_args()

    # --- Output Directory --- 
    output_base_folder = Path(args.dat_dir).parent / "output-splatting"
    if args.model_path is None:
        model_path = output_base_folder / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    else:
        model_path = output_base_folder / args.model_path
    model_path.mkdir(parents=True, exist_ok=True)
    print(f"Outputting to: {model_path}")

    # --- Load Configs --- 
    config = libcore.load_from_config(args.configs, cli_args=extras)
    use_deformation = config.get('use_deformation', False)
    config.use_deformation = use_deformation
    config.model.use_deformation, config.dataset.use_deformation = use_deformation, use_deformation

    # --- FullbodyFix bake micro-tune (optional) ---
    # Enforces a fixed Gaussian set (no densify/prune/reset) and typically freezes geometry params.
    def _cfg_get(cfg, key: str, default=None):
        if cfg is None:
            return default
        try:
            if isinstance(cfg, dict) or hasattr(cfg, "get"):
                v = cfg.get(key, default)
                return default if v is None else v
        except Exception:
            pass
        try:
            v = getattr(cfg, key)
            return default if v is None else v
        except Exception:
            return default

    # --- Pilot B: geometry-supervised finetune (mixed geo/RGB) ---
    geo_ft_cfg = _cfg_get(config, "pilot_geo_ft", None)
    geo_ft_enabled = bool(_cfg_get(geo_ft_cfg, "enabled", False)) if geo_ft_cfg is not None else False

    fbf_cfg = _cfg_get(_cfg_get(config, "optim", None), "fullbodyfix_bake", None)
    fullbodyfix_bake_enabled = bool(_cfg_get(fbf_cfg, "enabled", False))
    if fullbodyfix_bake_enabled:
        print("[FullbodyFixBake] Enabled: enforcing fixed GS set (no densify/prune/reset) + safe training guards.")
        # Defense-in-depth: disable any topology-changing optimizer behavior even if a code path is reached.
        for k, v in [
            ("densify_until_iter", 0),
            ("densify_from_iter", 10**12),
            ("densification_interval", 10**12),
            ("opacity_reset_interval", 0),
            ("opacity_reset_start_iter", 10**12),
        ]:
            try:
                setattr(config.optim, k, v)
            except Exception:
                pass
        # Safety: triangle-walk should never run in bake mode.
        try:
            config.model.skip_triangle_walk = True
        except Exception:
            pass

    # Wire free-gaussians flag into config
    if args.free_gaussians:
        # Disable all mesh/deformation dependencies
        config.use_deformation = False
        config.model.use_deformation = False
        config.dataset.use_deformation = False
        # Expose free mode to dataset
        config.dataset.free_gaussians = True
        # Ensure absolute XYZ behavior and skip triangle walking
        config.model.xyz_as_uvd = False
        config.model.skip_triangle_walk = True
        # Optional: allow override via config as well
        config.model.free_gaussians = True

    # Mesh-bound static proxy mode: dataset should be meshless like free mode, but model remains mesh-bound.
    mesh_bound_static = bool(args.static_proxy_mesh) or (args.init_gs_embed is not None)
    if mesh_bound_static:
        # Dataset: do NOT require SMPL/OBJ meshes
        config.dataset.free_gaussians = True
        # Model: keep mesh-bound semantics (do NOT set free_gaussians), but force safe correspondence behavior
        try:
            config.model.free_gaussians = False
        except Exception:
            pass
        # Mesh-bound embedding requires uvd semantics
        try:
            config.model.xyz_as_uvd = True
        except Exception:
            pass
        try:
            config.model.skip_triangle_walk = True
        except Exception:
            pass
    if config.dataset.dat_dir:
        print(f"Warning: config.dataset.dat_dir is set. Overriding with args.dat_dir: {args.dat_dir}")
    config.dataset.dat_dir = args.dat_dir
    config.dataset.data_device = args.data_device

    # Ensure model-side SMPLX settings exist for deformation LBS.
    # Dataset has its SMPLX config under dataset.avatarrex_config; model uses model.smpl_* keys.
    if use_deformation:
        avcfg = getattr(config.dataset, 'avatarrex_config', {})

        def _set_if_missing(obj, key, val):
            if val is None:
                return
            # DictConfig supports `key in obj`; fall back to hasattr for plain objects.
            try:
                if key not in obj:
                    obj[key] = val
            except Exception:
                if not hasattr(obj, key):
                    setattr(obj, key, val)

        _set_if_missing(config.model, 'smpl_model_type', getattr(avcfg, 'model_type', None))
        _set_if_missing(config.model, 'smpl_gender', getattr(avcfg, 'gender', None))
        _set_if_missing(config.model, 'smpl_model_path', getattr(avcfg, 'smpl_model_path', None))
        _set_if_missing(config.model, 'smpl_use_pca', bool(getattr(avcfg, 'smpl_use_pca', False)) if hasattr(avcfg, 'smpl_use_pca') else None)
        _set_if_missing(config.model, 'smpl_num_pca_comps', int(getattr(avcfg, 'smpl_num_pca_comps', 6)) if hasattr(avcfg, 'smpl_num_pca_comps') else None)
        # Note: dataset uses `flat_hand_mean`, model expects `smpl_flat_hand_mean`
        _set_if_missing(config.model, 'smpl_flat_hand_mean', bool(getattr(avcfg, 'flat_hand_mean', False)) if hasattr(avcfg, 'flat_hand_mean') else None)

        # ActorsHQ-style global transform: treat global_orient/transl as an external rigid transform.
        sampling_cfg = getattr(avcfg, 'sampling', {})
        dataset_flavor = sampling_cfg.get('dataset_flavor', 'generic') if sampling_cfg else 'generic'
        if dataset_flavor == 'actorshq':
            _set_if_missing(config.model, 'smpl_global_transform_mode', 'external')

    # Allow debug visualization selection from config as well (if CLI left at defaults).
    try:
        dbg_cfg = getattr(getattr(config, 'optim', {}), 'debug_vis', None)
    except Exception:
        dbg_cfg = None
    if dbg_cfg is not None:
        if args.debug_vis_frame is None and getattr(dbg_cfg, 'frame', None) is not None:
            args.debug_vis_frame = str(getattr(dbg_cfg, 'frame'))
        if args.debug_vis_cam is None and getattr(dbg_cfg, 'cam', None) is not None:
            args.debug_vis_cam = str(getattr(dbg_cfg, 'cam'))
        # Only override split if user didn't specify anything else and left default.
        if args.debug_vis_split == "train" and (args.debug_vis_frame is None and args.debug_vis_cam is None) and getattr(dbg_cfg, 'split', None) is not None:
            args.debug_vis_split = str(getattr(dbg_cfg, 'split'))

    def _save_ply_simple(path: str, verts: torch.Tensor, faces: torch.Tensor, colors: torch.Tensor = None):
        save_ply_mesh(path, verts, faces, vert_colors=colors)
    # Wire custom image/mask relative paths (optional)
    if args.custom_rgb_relpath is not None:
        config.dataset.custom_rgb_relpath = args.custom_rgb_relpath
    if args.custom_mask_relpath is not None:
        config.dataset.custom_mask_relpath = args.custom_mask_relpath
    # Wire loss masking flag (only set if explicitly enabled to avoid overriding YAML)
    if args.mask_render_in_loss:
        if not hasattr(config, 'optim'):
            config.optim = {}
        config.optim.apply_alpha_mask_to_render = True
    
    # Ensure garment label file path is in config if specified via CLI or make it absolute
    # The actual path is now expected inside avatarrex_config.garment_label_file
    # Example of how you might set it if it was a global arg (but it should be in avatarrex.yaml):
    # if args.garment_label_file:
    #    if not hasattr(config.dataset, 'avatarrex_config'): config.dataset.avatarrex_config = {}
    #    config.dataset.avatarrex_config.garment_label_file = args.garment_label_file

    # Ensure pre-trained GS path is in model config if specified (e.g. via launch.json override)
    # Example: extras could be like ["model.load_pretrained_gs_from=PATH"]. load_from_config handles this.
    # For your specific path, it should be in your YAML or launch.json.

    OmegaConf.save(config, model_path / 'config.yaml')
    libcore.set_seed(config.get('seed', 9061))

    # --- TensorBoard ---
    tb_cfg = getattr(getattr(config, 'optim', {}), 'tensorboard', None)
    tb_enabled = True if tb_cfg is None else bool(getattr(tb_cfg, 'enabled', True))
    tb_log_every = 10 if tb_cfg is None else int(getattr(tb_cfg, 'log_every', 10))
    tb_ema_beta = 0.98 if tb_cfg is None else float(getattr(tb_cfg, 'ema_beta', 0.98))
    writer = None
    ema = {}
    if tb_enabled:
        try:
            writer = SummaryWriter(log_dir=str(model_path / "tb"))
        except Exception as e:
            print(f"Warning: failed to init TensorBoard writer: {e}")
            writer = None

    def _ema_update(key: str, val: float) -> float:
        prev = ema.get(key, None)
        if prev is None:
            ema[key] = float(val)
        else:
            ema[key] = tb_ema_beta * float(prev) + (1.0 - tb_ema_beta) * float(val)
        return float(ema[key])

    # --- Dataset --- 
    config.dataset.cache_dir = Path(args.dat_dir) / f'cache_deform_{Path(args.configs[0]).stem}'
    frameset_train = make_frameset_data(config.dataset, split='train')
    # Use the 'val' split (configured via val_frames/val_cameras) for intermediate eval during training.
    frameset_test = make_frameset_data(config.dataset, split='val')

    # Resolve an optional deterministic debug visualization sample for checkpoint images.
    debug_vis_idx = None
    debug_frameset = frameset_train if args.debug_vis_split == "train" else frameset_test
    try:
        if debug_frameset is not None and len(debug_frameset) > 0:
            # Default to the very first sample if not specified.
            target_frame = None
            if args.debug_vis_frame is not None:
                if args.debug_vis_frame.isdigit():
                    digits = int(getattr(getattr(config.dataset, 'avatarrex_config', {}), 'frame_format_digits', 8))
                    target_frame = f"{int(args.debug_vis_frame):0{digits}d}"
                else:
                    target_frame = str(args.debug_vis_frame)
            target_cam = None
            if args.debug_vis_cam is not None:
                cam_spec = str(args.debug_vis_cam)
                all_ids = getattr(debug_frameset, "all_camera_ids_sorted", [])
                if cam_spec in all_ids:
                    target_cam = cam_spec
                elif cam_spec.isdigit():
                    idx = int(cam_spec)
                    if 1 <= idx <= len(all_ids):
                        target_cam = all_ids[idx - 1]

            # Search a matching (frame,cam) in split samples.
            if target_frame is not None and target_cam is not None:
                for i, (fid, cid) in enumerate(getattr(debug_frameset, "data_samples", [])):
                    if fid == target_frame and cid == target_cam:
                        debug_vis_idx = i
                        break
            elif target_cam is not None:
                for i, (fid, cid) in enumerate(getattr(debug_frameset, "data_samples", [])):
                    if cid == target_cam:
                        debug_vis_idx = i
                        break
            elif target_frame is not None:
                for i, (fid, cid) in enumerate(getattr(debug_frameset, "data_samples", [])):
                    if fid == target_frame:
                        debug_vis_idx = i
                        break

            if debug_vis_idx is None:
                debug_vis_idx = 0
    except Exception as e:
        print(f"Warning: failed to resolve debug vis sample: {e}")
        debug_vis_idx = None

    # Optional: mixed frame/window sampler for deformation training
    sampler = None
    try:
        sampling_cfg = getattr(getattr(config.dataset, 'avatarrex_config', {}), 'sampling', {})
    except Exception:
        sampling_cfg = {}
    enable_mixed_sampler = bool(sampling_cfg.get('enable_mixed_sampler', False)) if sampling_cfg else False
    if config.use_deformation and enable_mixed_sampler and (not getattr(config.model, 'free_gaussians', False)):
        if args.batch_size != 1:
            logger.warning("Mixed sampler is designed for batch_size=1 (one (frame,cam) per iter). Extra batch items will be ignored by training loop.")
        # Build sampler config
        sampler_cfg = MixedSamplerConfig(
            enable_pose_fps=bool(sampling_cfg.get('enable_pose_fps', True)),
            fps_k=int(sampling_cfg.get('fps_k', 1000)),
            fps_seed_mode=str(sampling_cfg.get('fps_seed_mode', 'min_norm')),
            fps_candidate_stride=int(sampling_cfg.get('fps_candidate_stride', 1)),
            window_len=int(sampling_cfg.get('window_len', 3)),
            win_m=int(sampling_cfg.get('win_m', 200)),
            win_min_gap=int(sampling_cfg.get('win_min_gap', 0)),
            p_div=float(sampling_cfg.get('p_div', 0.7)),
            p_win=float(sampling_cfg.get('p_win', 0.3)),
            sampler_seed=int(sampling_cfg.get('sampler_seed', 0)),
        )

        # Pose features for FPS/windows come from split SMPL params (already loaded by dataset for deformation).
        body_pose_tensor = getattr(getattr(frameset_train, 'smpl_params', {}), 'get', lambda k, d=None: d)('body_pose', None)
        if body_pose_tensor is None:
            raise RuntimeError("Mixed sampler enabled but frameset_train.smpl_params['body_pose'] is missing.")
        body_pose_np = body_pose_tensor.detach().cpu().numpy()

        fullbody_cam_ids = getattr(frameset_train, 'fullbody_cam_ids', None)
        if not fullbody_cam_ids:
            # Fallback to all available cams
            fullbody_cam_ids = sorted(list(getattr(frameset_train, 'all_camera_params', {}).keys()))
        sampler = MixedDivWinSampler(
            data_samples=frameset_train.data_samples,
            frame_id_list=frameset_train.frame_id_list,
            body_pose_np=body_pose_np,
            fullbody_cam_ids=fullbody_cam_ids,
            joint_indices_21=DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES,
            cfg=sampler_cfg,
        )
        # Save artifacts for reproducibility/debugging
        export_pose_diverse_cfg = sampling_cfg.get("export_pose_diverse", {}) if hasattr(sampling_cfg, "get") else {}
        export_info = save_sampling_artifacts(
            out_dir=model_path,
            fullbody_cam_ids=fullbody_cam_ids,
            s_div_frame_ids=sampler.s_div_frame_ids,
            win_start_frame_ids=sampler.win_start_frame_ids,
            cfg_dict={'sampling': dict(sampling_cfg), 'resolved': sampler_cfg.__dict__},
            frame_id_list=frameset_train.frame_id_list,
            body_pose_np=body_pose_np,
            joint_indices_21=DEFORMATION_NET_SMPLX_BODY_JOINT_INDICES,
            export_pose_diverse_cfg=export_pose_diverse_cfg,
        )
        print(f"[Sampling] enable_mixed_sampler=True | FULLBODY_CAMS={len(fullbody_cam_ids)} | |S_div|={len(sampler.s_div_frame_ids)} | |W_win|={len(sampler.win_start_frame_ids)}")

        # Optional: dump Top-K pose-diverse quicklook images (RGBA PNG) from a fixed camera.
        try:
            epd = export_pose_diverse_cfg
            export_enabled = bool(epd.get("enabled", False)) if hasattr(epd, "get") else bool(epd.get("enabled", False)) if isinstance(epd, dict) else False
            if export_enabled and export_info is not None:
                fixed_cam_id = str(export_info.get("fixed_cam_id_used", "cam"))
                topk_entries = list(export_info.get("topk_entries", []))
                out_root = Path(str(export_info.get("out_root", model_path)))
                img_dir_topk = out_root / str(export_info.get("img_dir_topk", "pose_diverse_topk_imgs"))
                img_dir_topk.mkdir(parents=True, exist_ok=True)

                # Build pair->dataset index lookup once.
                pair_to_idx = {}
                for i, (fid, cid) in enumerate(getattr(frameset_train, "data_samples", [])):
                    pair_to_idx[(str(fid), str(cid))] = int(i)

                dumped = 0
                for r, e in enumerate(topk_entries):
                    fid = str(e.get("frame_id", ""))
                    if not fid:
                        continue
                    idx = pair_to_idx.get((fid, fixed_cam_id), None)
                    if idx is None:
                        continue
                    sample = frameset_train[idx]
                    img_bgra = sample.get("gt_image_bgra", None)
                    if img_bgra is None:
                        continue
                    # gt_image_bgra is BGRA in [0,1]; convert to RGBA uint8 PNG.
                    x = (img_bgra.detach().cpu().clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
                    if x.ndim == 3 and x.shape[-1] == 4:
                        x = x[..., [2, 1, 0, 3]]
                    from PIL import Image as PILImage
                    PILImage.fromarray(x.numpy(), mode="RGBA").save(img_dir_topk / f"{r:02d}_{fid}.png")
                    dumped += 1
                if dumped > 0:
                    print(f"[SamplingExport] Dumped {dumped} Top-K RGBA quicklook images to {img_dir_topk}")
        except Exception as e:
            logger.warning(f"[SamplingExport] Failed to dump Top-K quicklook images: {e}")

        # Optional: dump GT RGBs for pose-diverse set S (s_div_frame_ids) at a specified view/camera.
        # This helps visually verify the selected frames are diverse enough.
        # `sampling_cfg` can be an OmegaConf DictConfig; treat any mapping with .get the same.
        dump_cfg = sampling_cfg.get("dump_s_div_rgb", {}) if hasattr(sampling_cfg, "get") else {}
        dump_enabled = bool(dump_cfg.get("enabled", False))
        if dump_enabled:
            try:
                cam_spec = dump_cfg.get("cam", None)
                if cam_spec is None:
                    raise RuntimeError("dump_s_div_rgb.enabled=true but dump_s_div_rgb.cam is not set")
                cam_spec_s = str(cam_spec)
                all_ids = getattr(frameset_train, "all_camera_ids_sorted", [])
                cam_id = cam_spec_s if cam_spec_s in all_ids else None
                if cam_id is None and cam_spec_s.isdigit():
                    idx = int(cam_spec_s)
                    # Backward compatible indexing:
                    # - cam: 1..N  => 1-based index (old behavior)
                    # - cam: 0     => first camera (common expectation)
                    if idx == 0 and len(all_ids) > 0:
                        cam_id = all_ids[0]
                    elif 1 <= idx <= len(all_ids):
                        cam_id = all_ids[idx - 1]
                if cam_id is None:
                    raise RuntimeError(
                        f"Could not resolve dump_s_div_rgb.cam={cam_spec!r} to a camera id. "
                        f"Provide a camera id in {list(all_ids)[:8]}{'...' if len(all_ids) > 8 else ''} "
                        f"or an index (0=first, 1..N=1-based)."
                    )

                max_frames = dump_cfg.get("max_frames", None)
                if max_frames is not None:
                    max_frames = int(max_frames)

                out_dir = model_path / "sampling" / f"s_div_rgb_cam_{cam_id}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Build a quick lookup from frame_id_str -> exists for this cam by scanning data_samples once.
                frame_to_exists = set()
                for fid, cid in getattr(frameset_train, "data_samples", []):
                    if cid == cam_id:
                        frame_to_exists.add(fid)

                # Resolve rgb path function (respects custom_rgb_relpath if set)
                dat_dir = getattr(frameset_train, "dat_dir", None)
                custom_rgb_relpath = getattr(frameset_train, "custom_rgb_relpath", None)
                if dat_dir is None:
                    raise RuntimeError("frameset_train.dat_dir not available; cannot dump RGBs")

                def _rgb_path(fid_str: str) -> Path:
                    if custom_rgb_relpath:
                        rel = custom_rgb_relpath.format(frame=fid_str, cam=cam_id)
                        return Path(dat_dir) / cam_id / rel
                    return Path(dat_dir) / cam_id / f"{fid_str}.jpg"

                dumped = 0
                for j, fid_str in enumerate(sampler.s_div_frame_ids):
                    if (max_frames is not None) and (dumped >= max_frames):
                        break
                    if fid_str not in frame_to_exists:
                        continue
                    p = _rgb_path(fid_str)
                    if not p.exists():
                        continue
                    img = PILImage.open(p).convert("RGB")
                    img.save(out_dir / f"{dumped:04d}_{fid_str}.png")
                    dumped += 1
                print(f"[SamplingDump] Dumped {dumped} RGBs for S_div at cam={cam_id} to {out_dir}")
            except Exception as e:
                logger.warning(f"[SamplingDump] Failed to dump S_div RGBs: {e}")

    dataloader = make_dataloader(frameset_train, shuffle=(sampler is None), sampler=sampler,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    pair_to_idx = build_pair_to_dataset_index(frameset_train)
    
    num_training_frames = len(frameset_train)
    if not hasattr(config.model, 'num_training_frames'): 
        config.model.num_training_frames = num_training_frames
    
    if args.total_iteration != -1:
        config.optim.total_iteration = args.total_iteration
        print(f"Warning: Overriding total_iteration with args.total_iteration: {args.total_iteration}")
    
    if not hasattr(config.optim, 'total_iteration') and args.total_iteration == -1: 
        default_total_iter = config.optim.get('stage1_end_iter', 50000) + config.optim.get('stage2_duration', 100000)
        print(f"Warning: config.optim.total_iteration not set. Inferring from stage durations: {default_total_iter}")
        config.optim.total_iteration = default_total_iter

    # --- Canonical mesh + garment mask for initialization ---
    # In deformation mode we must initialize with the *clothed* reference canonical mesh that matches:
    # - pretrained Gaussian binding (embedding.json cano_mesh)
    # - LBS weights (lbs_weights_path)
    #
    # Do NOT use per-frame SMPL mesh (e.g. 10475 verts) as canonical, otherwise LBS will shape-mismatch.
    cano_mesh_data_for_model = None
    initial_garment_mask_from_dataset = None

    # Mesh-bound static mode: canonical mesh comes from embedding.json (or explicit --init_cano_mesh).
    if mesh_bound_static and args.init_gs_embed is not None:
        try:
            embed_json_path = Path(str(args.init_gs_embed))
            if not embed_json_path.exists():
                raise FileNotFoundError(f"embedding json not found: {embed_json_path}")
            import json as _json
            with open(embed_json_path, "r") as f:
                cc = _json.load(f)
            cano_mesh_rel = cc.get("cano_mesh", cc.get("mesh_fn"))
            if args.init_cano_mesh is not None:
                cano_mesh_path = Path(str(args.init_cano_mesh))
            else:
                if not cano_mesh_rel:
                    raise KeyError(f"embedding json missing 'cano_mesh'/'mesh_fn': {embed_json_path}")
                cano_mesh_path = (embed_json_path.parent / str(cano_mesh_rel))
            if not cano_mesh_path.exists():
                raise FileNotFoundError(f"canonical mesh not found: {cano_mesh_path}")
            cano_mesh_cpu = libcore.MeshCpu(str(cano_mesh_path))
            cano_mesh_data_for_model = {
                "mesh_verts": torch.tensor(cano_mesh_cpu.V).float(),
                "mesh_norms": torch.tensor(cano_mesh_cpu.N).float(),
                "mesh_faces": torch.tensor(cano_mesh_cpu.F).long(),
            }
            print(f"[Init] (mesh_bound_static) Loaded canonical mesh: {cano_mesh_path}")
        except Exception as e:
            raise RuntimeError(f"[Init] mesh_bound_static failed to load canonical mesh from embedding: {e}")
    if config.use_deformation and getattr(config.model, 'load_pretrained_gs_from', None):
        try:
            embed_json_path = Path(str(getattr(config.model, 'load_pretrained_gs_from'))) / "embedding.json"
            if embed_json_path.exists():
                import json as _json
                with open(embed_json_path, "r") as f:
                    cc = _json.load(f)
                cano_mesh_rel = cc.get("cano_mesh", cc.get("mesh_fn"))
                if cano_mesh_rel:
                    cano_mesh_path = embed_json_path.parent / cano_mesh_rel
                    if cano_mesh_path.exists():
                        cano_mesh_cpu = libcore.MeshCpu(str(cano_mesh_path))
                        cano_mesh_data_for_model = {
                            "mesh_verts": torch.tensor(cano_mesh_cpu.V).float(),
                            "mesh_norms": torch.tensor(cano_mesh_cpu.N).float(),
                            "mesh_faces": torch.tensor(cano_mesh_cpu.F).long(),
                        }
                        print(f"[Init] Loaded canonical mesh from embedding: {cano_mesh_path}")
        except Exception as e:
            print(f"[Init] WARNING: failed to load canonical mesh from embedding.json: {e}")

    # Garment mask (optional) comes from dataset reference labels if present.
    initial_garment_mask_from_dataset = getattr(frameset_train, "ref_frame_garment_mask", None)

    # Fallback: use first dataset sample mesh_info (may be SMPL mesh; only safe when not in deformation mode).
    if cano_mesh_data_for_model is None:
        first_batch_sample = frameset_train.__getitem__(0)
        cano_mesh_data_for_model = first_batch_sample["mesh_info"]
        if initial_garment_mask_from_dataset is None:
            initial_garment_mask_from_dataset = first_batch_sample.get("garment_mask_cano")
        if initial_garment_mask_from_dataset is None:
            print("Warning: Initial garment mask not found. Model will use fallback.")

    # --- Model --- 
    pipe = config.pipe

    # Deformation mode needs reference-frame SMPL params + LBS weights to compute inverse ref joint mats.
    ref_frame_smpl_params_for_lbs = None
    lbs_weights_for_ref_mesh = None
    if config.use_deformation and (not getattr(config.model, 'free_gaussians', False)):
        ref_frame_smpl_params_for_lbs = getattr(frameset_train, 'ref_frame_smpl_params', None)
        lbs_weights_for_ref_mesh = getattr(frameset_train, 'lbs_weights_for_ref_mesh', None)
        if ref_frame_smpl_params_for_lbs is None or lbs_weights_for_ref_mesh is None:
            raise RuntimeError(
                "Deformation enabled but dataset did not provide reference assets. "
                "Expected frameset_train.ref_frame_smpl_params and frameset_train.lbs_weights_for_ref_mesh."
            )

    gs_model = SplattingAvatarModel(
        config.model,
        verbose=True,
        num_training_frames=num_training_frames,
        ref_frame_smpl_params_for_lbs=ref_frame_smpl_params_for_lbs,
        lbs_weights_for_ref_mesh=lbs_weights_for_ref_mesh,
        static_rendering=getattr(config.model, 'free_gaussians', False) # In free mode, use static rendering semantics
    )

    # Pilot geo-ft: load checkpoint *after* model initialization (PLY/embedding/caches),
    # and ignore SMPLX model buffers/caches which are dataset-specific and can mismatch (e.g. expr_dirs dim).
    geo_ft_ckpt_info = None
    geo_ft_strict_load = True
    if geo_ft_enabled:
        ckpt_path = _cfg_get(geo_ft_cfg, "load_ckpt_path", None)
        if ckpt_path is None:
            raise ValueError("pilot_geo_ft.enabled=true but pilot_geo_ft.load_ckpt_path is not set")
        geo_ft_strict_load = bool(_cfg_get(geo_ft_cfg, "strict_load", True))
        geo_ft_ckpt_info = load_deform_checkpoint(str(ckpt_path), map_location="cpu")
    if getattr(config.model, 'free_gaussians', False):
        # Initialize trainable Gaussians from a PLY (required for free-gaussians mode)
        init_ply = args.init_gs_ply
        if init_ply is None:
            # Fallback to config.model.load_pretrained_gs_from directory (point_cloud.ply)
            pretrained_dir = getattr(config.model, 'load_pretrained_gs_from', None)
            if pretrained_dir is not None:
                candidate = Path(pretrained_dir) / "point_cloud.ply"
                init_ply = str(candidate)
        if init_ply is None or not Path(init_ply).exists():
            # Random initialization path (no PLY available)
            num_init = getattr(config.model, 'num_init_samples', 10000)
            extent = getattr(config.dataset, 'cameras_extent', 1.0)
            init_center = getattr(config.dataset, 'init_center', [0.0, 0.0, 0.0])
            sh_degree = getattr(config.model, 'sh_degree', 0)
            # xyz in cube [-extent, extent]
            xyz = ((np.random.rand(num_init, 3).astype(np.float32) * 2.0 - 1.0) + np.array(init_center)) * float(extent)
            # DC features: random RGB mapped to SH DC term; extras zeros depending on SH degree
            # rand_rgb = torch.rand((num_init, 3), dtype=torch.float32)
            # features_dc = RGB2SH(rand_rgb).cpu().numpy()[:, :, None]  # (P,3,1)
            features_dc = np.zeros((num_init, 3, 1), dtype=np.float32)
            extra_dim = (sh_degree + 1) ** 2 - 1
            features_extra = np.zeros((num_init, 3, max(0, extra_dim)), dtype=np.float32)
            # Opacity initialized to inverse_sigmoid(0.1)
            opacities = np.full((num_init, 1), float(inverse_sigmoid(0.01)), dtype=np.float32)
            # Scales: small initial size proportional to extent
            base_scale = 0.01 * float(extent)
            scales = np.full((num_init, 3), np.log(base_scale), dtype=np.float32)
            # Rotations: identity quaternions
            rots = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_init, 1))
            gs_model.init_gauss(xyz, features_dc, features_extra, opacities, scales, rots, init_params=True)
            print(f"[Free3DGS] Initialized {num_init} random Gaussians within extent ±{extent}.")
        else:
            # Load as trainable parameters
            gs_model.load_ply(init_ply, init_params=True)
            print(f"[Free3DGS] Loaded trainable Gaussians from: {init_ply}")
    else:
        # Mesh-bound static proxy mode (no per-frame meshes): initialize from canonical mesh + embedding + PLY.
        if mesh_bound_static:
            if args.init_gs_ply is None:
                raise ValueError("--static_proxy_mesh/--init_gs_embed requires --init_gs_ply for parameter initialization.")
            if args.init_gs_embed is None:
                raise ValueError("--static_proxy_mesh requires --init_gs_embed (embedding.json).")

            gs_model.create_from_canonical(cano_mesh_data_for_model, garment_mask_ref_frame=initial_garment_mask_from_dataset)
            # Load trainable Gaussian parameters from PLY
            gs_model.load_ply(str(args.init_gs_ply), init_params=True)
            gs_model.gaussians_are_frozen = False
            # Load mesh binding + embedded parameterization (uvd + rotation + fidx/bary)
            gs_model.load_from_embedding(str(args.init_gs_embed))
            # Ensure fixed posed mesh state exists (canonical) for rendering.
            try:
                gs_model.update_to_cano_mesh()
            except Exception:
                pass
            # Safety: reset per-point caches to correct length
            try:
                gs_model.max_radii2D = torch.zeros((gs_model.get_xyz.shape[0]), device='cuda')
            except Exception:
                pass
            print(f"[Init] Mesh-bound static proxy mode enabled (embed={args.init_gs_embed}).")
        else:
            # create_from_canonical now also loads garment mask and potentially pretrained Gaussians
            gs_model.create_from_canonical(cano_mesh_data_for_model, garment_mask_ref_frame=initial_garment_mask_from_dataset)

    # Apply pilot geo-ft checkpoint now that GS + embedding buffers exist.
    if geo_ft_enabled and geo_ft_ckpt_info is not None:
        try:
            ignore_prefixes = (
                # SMPL-X implementation details / buffers (not something we want to restore from ckpt)
                "smpl_model_for_lbs.",
                "smpl_model.",
                # runtime caches derived from canonical mesh (must match current embedding)
                "quat_helper.",
                "phongsurf.",
            )
            model_sd = gs_model.state_dict()
            ckpt_sd = geo_ft_ckpt_info.gs_model_state_dict
            filtered = {}
            skipped_prefix = 0
            skipped_missing = 0
            skipped_shape = 0
            for k, v in ckpt_sd.items():
                if any(str(k).startswith(p) for p in ignore_prefixes):
                    skipped_prefix += 1
                    continue
                if k not in model_sd:
                    skipped_missing += 1
                    continue
                mv = model_sd.get(k, None)
                if isinstance(v, torch.Tensor) and isinstance(mv, torch.Tensor):
                    if tuple(v.shape) != tuple(mv.shape):
                        skipped_shape += 1
                        continue
                filtered[k] = v

            missing, unexpected = gs_model.load_state_dict(filtered, strict=False)
            print(
                f"[pilot_geo_ft] Loaded ckpt: {geo_ft_ckpt_info.ckpt_path} (iter={geo_ft_ckpt_info.iteration}) "
                f"strict_requested={geo_ft_strict_load} strict_applied=False | "
                f"kept={len(filtered)}/{len(ckpt_sd)} skipped_prefix={skipped_prefix} skipped_not_in_model={skipped_missing} skipped_shape={skipped_shape} "
                f"missing_after={len(missing)} unexpected_after={len(unexpected)}"
            )
        except Exception as e:
            raise RuntimeError(f"[pilot_geo_ft] Failed to load ckpt after init: {e}")

    # --- Optimizer --- 
    gs_optim = SplattingAvatarOptimizer(gs_model, config.optim)

    # Pilot geo-ft: freeze everything except Stage-B vertex offset MLP (+ optional latent codes).
    if geo_ft_enabled:
        train_vertex_only = bool(_cfg_get(geo_ft_cfg, "train_vertex_mlp_only", True))
        train_latent = bool(_cfg_get(geo_ft_cfg, "train_latent_codes", False))
        lr_vert = float(_cfg_get(geo_ft_cfg, "lr_vertex_mlp", 1e-4))
        lr_latent = float(_cfg_get(geo_ft_cfg, "lr_latent", 1e-4))

        # Disable any topology-changing optimizer behavior defensively.
        for k, v in [
            ("densify_until_iter", 0),
            ("densify_from_iter", 10**12),
            ("densification_interval", 10**12),
            ("opacity_reset_interval", 0),
            ("opacity_reset_start_iter", 10**12),
        ]:
            try:
                setattr(config.optim, k, v)
            except Exception:
                pass
        try:
            config.model.skip_triangle_walk = True
        except Exception:
            pass

        if train_vertex_only:
            # Freeze all model params first.
            for p in gs_model.parameters():
                try:
                    p.requires_grad_(False)
                except Exception:
                    pass

            # Collect vertex MLP params from the already-built optimizer groups.
            vert_params = []
            latent_params = []
            for g in gs_optim.optimizer.param_groups:
                nm = str(g.get("name", ""))
                if nm == "deform_vert_mlp":
                    vert_params.extend(list(g.get("params", [])))
                if nm == "deform_latent_codes":
                    latent_params.extend(list(g.get("params", [])))
            if not vert_params:
                raise RuntimeError("[pilot_geo_ft] Could not find deform_vert_mlp params to finetune.")
            for p in vert_params:
                p.requires_grad_(True)
            if train_latent:
                for p in latent_params:
                    p.requires_grad_(True)

            # Replace optimizer with only the desired param groups; keep LossBase methods via gs_optim.
            param_groups = [
                {
                    "params": vert_params,
                    "lr": float(lr_vert),
                    "stage1_lr": float(lr_vert),
                    "stage2_lr": float(lr_vert),
                    "name": "pilot_geo_ft_vert_mlp",
                    "stage1_active": True,
                    "stage2_active": True,
                }
            ]
            if train_latent and latent_params:
                param_groups.append(
                    {
                        "params": latent_params,
                        "lr": float(lr_latent),
                        "stage1_lr": float(lr_latent),
                        "stage2_lr": float(lr_latent),
                        "name": "pilot_geo_ft_latent",
                        "stage1_active": True,
                        "stage2_active": True,
                    }
                )
            eps = float(getattr(config.optim, "adam_eps", 1e-15))
            gs_optim.optimizer = torch.optim.Adam(param_groups, lr=float(lr_vert), eps=eps)
            gs_optim.schedulers = []
            gs_optim.lr_schedulers = {}
            # Make stage logic always keep these active (stage2 always).
            try:
                gs_optim.stage1_end_iter = 0
            except Exception:
                pass

            # Debug print: trainable parameter count.
            n_train = 0
            for p in gs_model.parameters():
                if getattr(p, "requires_grad", False):
                    try:
                        n_train += int(p.numel())
                    except Exception:
                        pass
            print(f"[pilot_geo_ft] Trainable params (numel): {n_train} | train_latent={train_latent}")

    # FullbodyFix bake: runtime fixed-N assertions (count and per-GS tensor lengths).
    fixed_gauss_n0 = None
    fixed_gauss_checked_names = None
    if fullbodyfix_bake_enabled:
        try:
            fixed_gauss_n0 = int(gs_model.get_xyz.shape[0])
        except Exception:
            fixed_gauss_n0 = None
        fixed_gauss_checked_names = [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_opacity",
            "_scaling",
            "_rotation",
            "sample_fidxs",
            "sample_bary",
        ]

        def _assert_fixed_gaussian_set(tag: str) -> None:
            if fixed_gauss_n0 is None:
                return
            try:
                n_now = int(gs_model.get_xyz.shape[0])
            except Exception:
                n_now = None
            if n_now is not None and n_now != fixed_gauss_n0:
                raise RuntimeError(f"[FullbodyFixBake] Gaussian count changed at {tag}: {n_now} != {fixed_gauss_n0}")

            for name in (fixed_gauss_checked_names or []):
                t = getattr(gs_model, name, None)
                if not isinstance(t, torch.Tensor):
                    continue
                if t.ndim < 1:
                    continue
                if int(t.shape[0]) != fixed_gauss_n0:
                    raise RuntimeError(
                        f"[FullbodyFixBake] Per-GS tensor length changed at {tag}: {name}.shape[0]={int(t.shape[0])} != {fixed_gauss_n0}"
                    )

        _assert_fixed_gaussian_set("init")

    # --- GUI --- 
    network_gui = None 
    if args.ip != 'none':
        from gaussian_renderer import network_gui as gui_module 
        network_gui = gui_module 
        network_gui.init(args.ip, args.port)
        verify = args.dat_dir
    else:
        verify = None
    
    # --- Training Loop --- 
    data_iterator = iter(dataloader)
    # TODO: this should be enabled when multi-batch training is actually supported
    # total_iteration = config.optim.total_iteration // args.batch_size
    total_iteration = config.optim.total_iteration
    save_every_iter = config.optim.get('save_every_iter', 10000)
    default_testing_iterations = [i for i in range(save_every_iter, total_iteration + 1, save_every_iter) if i > 0] # Ensure positive
    if total_iteration not in default_testing_iterations and total_iteration > 0 : default_testing_iterations.append(total_iteration)
    if not default_testing_iterations and total_iteration > 0 : default_testing_iterations = [total_iteration] # At least test at the end
    testing_iterations = config.optim.get('testing_iterations', default_testing_iterations)

    print(f"Starting training for {total_iteration} iterations. Stage 1 ends at: {config.optim.get('stage1_end_iter', 0)}")
    print(f"Gaussians frozen: {gs_model.gaussians_are_frozen}")
    if gs_model.gaussians_are_frozen:
        print(f"Pretrained Gaussians loaded from: {gs_model.pretrained_gs_path}")

    # TODO-3: window detection state for temporal losses (DoD + coeff smooth)
    temporal_enabled = (float(getattr(config.optim, "lambda_dod", 0.0)) > 0.0) or (float(getattr(config.optim, "lambda_c_smooth", 0.0)) > 0.0)
    temporal_state = {
        "active": False,
        "cam_id": None,
        "f0": None,
        "prev_f": None,
        "step": -1,
        "bg": None,
        "bg_key": None,
        "window_id": None,
    }

    # --- Pilot geo-ft: discover K geometry frames + build GT mesh cache ---
    geo_frames = None
    geo_mesh_cache = None
    p_geo = 0.0
    geo_cam_sampling = "random_fullbody"
    geo_cam_ids = None
    geo_k = 0
    if geo_ft_enabled:
        # Resolve frame-id padding from dataset config.
        try:
            avcfg = getattr(config.dataset, "avatarrex_config", None)
        except Exception:
            avcfg = None
        frame_digits = 4
        try:
            frame_digits = int(_cfg_get(avcfg, "frame_format_digits", 4) or 4) if avcfg is not None else 4
        except Exception:
            frame_digits = 4

        # Discover geometry-supervised frames from per-frame dirs (TalkBody4D).
        mesh_subject_root = _cfg_get(geo_ft_cfg, "mesh_subject_root", None)
        if mesh_subject_root is None:
            raise ValueError("pilot_geo_ft.enabled=true but pilot_geo_ft.mesh_subject_root is not set")
        subject_prefix = _cfg_get(geo_ft_cfg, "subject_prefix", None)
        if subject_prefix is None:
            # Default: use dataset subject name (e.g., XG_01) inferred from sequence dir basename.
            subject_prefix = Path(str(args.dat_dir)).name
        mesh_relpath = str(_cfg_get(geo_ft_cfg, "mesh_relpath", "mesh/processed/nerf_simp_cleaned.obj"))
        geo_k = int(_cfg_get(geo_ft_cfg, "k_frames", 10) or 10)
        frame_id_list = _cfg_get(geo_ft_cfg, "frame_id_list", None)

        ref_idx = None
        try:
            ref_idx = int(_cfg_get(avcfg, "reference_frame_idx_for_smpl", None)) if avcfg is not None else None
        except Exception:
            ref_idx = None

        
        geo_frames = discover_talkbody4d_geo_frames(
            mesh_subject_root=str(mesh_subject_root),
            subject_prefix=str(subject_prefix),
            mesh_relpath=mesh_relpath,
            k_frames=int(geo_k),
            exclude_frame_id_int=ref_idx,
            frame_id_list=frame_id_list,
            frame_format_digits=6,
        )
        geo_mesh_cache = GeoMeshGtCache()
        try:
            import json as _json
            out_json = model_path / "pilot_geo_ft_frames.json"
            out_json.write_text(
                _json.dumps(
                    {
                        "mesh_subject_root": str(mesh_subject_root),
                        "subject_prefix": str(subject_prefix),
                        "mesh_relpath": mesh_relpath,
                        "k_frames": int(geo_k),
                        "exclude_frame_id_int": ref_idx,
                        "frames": [
                            {"frame_id_int": f.frame_id_int, "frame_id_str": f.frame_id_str, "mesh_path": f.mesh_path}
                            for f in geo_frames
                        ],
                    },
                    indent=2,
                )
            )
        except Exception:
            pass

        p_geo = float(_cfg_get(geo_ft_cfg, "p_geo", 0.3) or 0.0)
        geo_cam_sampling = str(_cfg_get(geo_ft_cfg, "geo_cam_sampling", "random_fullbody"))
        geo_cam_ids = _cfg_get(geo_ft_cfg, "fullbody_cam_ids", None)
        if geo_cam_ids is None:
            geo_cam_ids = getattr(frameset_train, "fullbody_cam_ids", None)
        if not geo_cam_ids:
            geo_cam_ids = sorted(list(getattr(frameset_train, "all_camera_params", {}).keys()))
        geo_cam_ids = [str(x) for x in list(geo_cam_ids)]
        print(f"[pilot_geo_ft] enabled | p_geo={p_geo} | K={len(geo_frames)} | geo_cam_sampling={geo_cam_sampling} | cams={len(geo_cam_ids)}")
        geo_vis_warned_once = False

    pbar = tqdm(range(1, total_iteration + 1))
    for iteration in pbar:
        current_lr = gs_optim.update_learning_rate(iteration)

        # Make training iteration available to model submodules (e.g., LBS alpha schedule).
        try:
            gs_model.global_step = int(iteration)
        except Exception:
            gs_model.global_step = None
        # Optional: expose stage to model for stage-gated behaviors (e.g., xyz_query_override use_in_stage1/2).
        try:
            gs_model.current_stage = int(getattr(gs_optim, "current_stage", 0) or 0)
        except Exception:
            pass

        # Pilot geo-ft mixed sampling: with p_geo, pick from sparse geo frames; otherwise normal dataloader.
        is_geo_step = False
        geo_spec = None
        if geo_ft_enabled and geo_frames is not None and geo_mesh_cache is not None and p_geo > 0:
            try:
                if float(np.random.rand()) < float(p_geo):
                    is_geo_step = True
            except Exception:
                is_geo_step = False

        if is_geo_step:
            # Choose geo frame + camera.
            geo_spec = geo_frames[int(np.random.randint(0, len(geo_frames)))]
            if geo_cam_sampling == "fixed":
                cam_id = str(geo_cam_ids[0])
            else:
                cam_id = str(geo_cam_ids[int(np.random.randint(0, len(geo_cam_ids)))])
            def _pair_lookup_idx(frame_id_str: str, cam_id) -> int | None:
                fid_s = str(frame_id_str)
                cam_s = str(cam_id)
                hit = pair_to_idx.get((fid_s, cam_s), None)
                if hit is not None:
                    return int(hit)
                try:
                    fid_i = int(fid_s)
                except Exception:
                    return None
                # raw int string
                hit = pair_to_idx.get((str(fid_i), cam_s), None)
                if hit is not None:
                    return int(hit)
                # dataset padding
                try:
                    fid_pad = f"{fid_i:0{int(frame_digits)}d}"
                    hit = pair_to_idx.get((fid_pad, cam_s), None)
                    if hit is not None:
                        return int(hit)
                except Exception:
                    pass
                # legacy 6-digit padding
                try:
                    fid_pad6 = f"{fid_i:06d}"
                    hit = pair_to_idx.get((fid_pad6, cam_s), None)
                    if hit is not None:
                        return int(hit)
                except Exception:
                    pass
                return None

            idx = _pair_lookup_idx(str(geo_spec.frame_id_str), cam_id)
            if idx is None:
                # Fallback: if that cam doesn't exist for the frame, try random cams a few times.
                idx = None
                for _ in range(10):
                    cam_id2 = str(geo_cam_ids[int(np.random.randint(0, len(geo_cam_ids)))])
                    idx2 = _pair_lookup_idx(str(geo_spec.frame_id_str), cam_id2)
                    if idx2 is not None:
                        idx = idx2
                        cam_id = cam_id2
                        break
            if idx is None:
                # Fallback to normal batch if lookup fails.
                is_geo_step = False
        if geo_ft_enabled and is_geo_step:
            try:
                pbar.set_description(f"geo_ft geo step | fid={geo_spec.frame_id_str if geo_spec else '?'}")
            except Exception:
                pass

        if not is_geo_step:
            try:
                batches = next(data_iterator)
            except StopIteration: 
                data_iterator = iter(dataloader)
                batches = next(data_iterator)
            if len(batches) > 1:
                logger.warning(f"Only the first batch will be used for training. The rest will be ignored.")
            batch = batches[0]
        else:
            batch = frameset_train[int(idx)]
        frm_idx = batch['frm_idx'] # This should be the integer frame index for latent codes
        scene_cameras = batch['scene_cameras']
        current_mesh_info = batch['mesh_info']
        
        # In mesh-bound static proxy mode, we do not require per-frame meshes from the dataset.
        need_per_frame_mesh = (not getattr(config.model, 'free_gaussians', False)) and (not mesh_bound_static)
        if need_per_frame_mesh:
            if current_mesh_info is None or (config.use_deformation and 'smplx_params_raw_for_deformnet' not in current_mesh_info):
                print(f"Skipping iteration {iteration}: mesh_info or smplx_params_raw_for_deformnet missing.")
                continue
        
        if not getattr(config.model, 'free_gaussians', False):
            if mesh_bound_static:
                # Keep mesh state fixed to canonical proxy mesh.
                gs_model.update_to_cano_mesh()
            else:
                # Deformation mode expects the raw SMPLX parameter dict so it can compute joint transforms for LBS.
                # The 63D pose feature for MLP/BS is extracted inside the model.
                smplx_params_dict = current_mesh_info['smplx_params_raw_for_deformnet'] if config.use_deformation else None
                gs_model.update_to_posed_mesh(
                    raw_mesh_info_for_no_deform=current_mesh_info,
                    current_frame_idx=frm_idx,
                    current_target_frame_smpl_params=smplx_params_dict
                )
            
        viewpoint_cam = scene_cameras[0].cuda()

        if network_gui and network_gui.conn is not None:
            ref_gt_for_gui = viewpoint_cam.original_image if hasattr(viewpoint_cam, 'original_image') else None
            network_gui.render_to_network(gs_model, pipe, verify, gt_image=ref_gt_for_gui)

        # TODO-3: deterministic background within a 3-frame window (only when bg_color is random).
        temporal_window_id = None
        temporal_tag = None
        temporal_bg_key = None
        bg_for_render = args.bg_color
        if temporal_enabled and args.bg_color == "random":
            try:
                cam_id = batch.get("cam_id_str", None)
                frm = int(frm_idx) if frm_idx is not None else None
                if cam_id is not None and frm is not None:
                    # Update window state (detect consecutive frames with same cam)
                    if temporal_state["active"] and (cam_id == temporal_state["cam_id"]) and (temporal_state["prev_f"] is not None) and (frm == int(temporal_state["prev_f"]) + 1) and (int(temporal_state["step"]) < 2):
                        temporal_state["step"] = int(temporal_state["step"]) + 1
                        temporal_state["prev_f"] = frm
                    else:
                        temporal_state["active"] = True
                        temporal_state["cam_id"] = cam_id
                        temporal_state["f0"] = frm
                        temporal_state["prev_f"] = frm
                        temporal_state["step"] = 0
                        seq_name = Path(args.dat_dir).name
                        temporal_state["window_id"] = f"{seq_name}|{cam_id}|{frm}"
                        # Deterministic bg seed from window_id
                        import hashlib
                        h = hashlib.sha1(temporal_state["window_id"].encode("utf-8")).digest()
                        seed = int.from_bytes(h[:4], "little", signed=False)
                        temporal_state["bg_key"] = int(seed)
                        rng = np.random.RandomState(seed)
                        temporal_state["bg"] = torch.tensor(rng.rand(3), device=image.device if "image" in locals() else "cuda", dtype=torch.float32)

                    step = int(temporal_state["step"])
                    if step == 0:
                        temporal_tag = "prev"
                    elif step == 1:
                        temporal_tag = "cur"
                    elif step == 2:
                        temporal_tag = "next"

                    temporal_window_id = temporal_state["window_id"]
                    temporal_bg_key = temporal_state["bg_key"]
                    bg_for_render = temporal_state["bg"]

                    if step == 2:
                        # End of 3-micro-step window
                        temporal_state["active"] = False
            except Exception:
                pass

        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=bg_for_render)
        image = render_pkg['render']
        gt_image = render_pkg['gt_image']
        gt_alpha_mask = render_pkg.get('gt_alpha_mask')
        background = render_pkg.get('background', None)

        # Decide whether to save per-iteration debug visuals (shared between hybrid/non-hybrid paths)
        debug_now = (
            iteration == 1
            or iteration == total_iteration
            or (save_every_iter > 0 and iteration % save_every_iter == 0)
            or (iteration in testing_iterations)
        )

        # Hybrid supervision: load both original and refined GT (foreground + same BG)
        if args.hybrid_supervision:
            if args.hybrid_orig_dat_dir is None:
                raise ValueError("--hybrid_supervision requires --hybrid_orig_dat_dir")
            cam_id = batch.get('cam_id_str')
            frame_id_str = batch.get('frm_idx_str')
            if cam_id is None or frame_id_str is None:
                raise RuntimeError("Hybrid supervision requires batch keys: cam_id_str, frm_idx_str")

            refined_rel = args.hybrid_refined_rgb_relpath or args.custom_rgb_relpath
            if refined_rel is None:
                raise ValueError("--hybrid_supervision requires --hybrid_refined_rgb_relpath or --custom_rgb_relpath")

            refined_path = Path(args.dat_dir) / cam_id / refined_rel.format(frame=frame_id_str, cam=cam_id)
            orig_path = Path(args.hybrid_orig_dat_dir) / cam_id / args.hybrid_orig_rgb_relpath.format(frame=frame_id_str, cam=cam_id)

            if not refined_path.exists():
                raise FileNotFoundError(f"Refined GT not found: {refined_path}")
            if not orig_path.exists():
                raise FileNotFoundError(f"Original GT not found: {orig_path}")

            # Load RGB to torch [3,H,W] in [0,1]
            refined_rgb = torch.from_numpy(np.array(Image.open(refined_path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)
            orig_rgb = torch.from_numpy(np.array(Image.open(orig_path).convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)

            # Move to CUDA and resize to current training resolution
            Ht, Wt = gt_image.shape[-2:]
            refined_rgb = refined_rgb.to(image.device).unsqueeze(0)
            orig_rgb = orig_rgb.to(image.device).unsqueeze(0)
            refined_rgb = F.interpolate(refined_rgb, size=(Ht, Wt), mode="bilinear", align_corners=False)
            orig_rgb = F.interpolate(orig_rgb, size=(Ht, Wt), mode="bilinear", align_corners=False)
            refined_rgb = refined_rgb.squeeze(0)
            orig_rgb = orig_rgb.squeeze(0)

            # Use the same alpha mask as the training camera (already aligned to gt_image).
            if gt_alpha_mask is None:
                # No mask available: treat as fully-foreground
                alpha = torch.ones((1, Ht, Wt), device=image.device, dtype=image.dtype)
            else:
                alpha = gt_alpha_mask

            # Use the same per-iteration background used for rendering, if available.
            # Fallback to black if not present (should not happen after gauss_base patch).
            if background is None:
                background = torch.zeros((3,), device=image.device, dtype=image.dtype)

            bg = background[:, None, None]
            gt_orig_comp = orig_rgb * alpha + bg * (1.0 - alpha)
            gt_refined_comp = refined_rgb * alpha + bg * (1.0 - alpha)

            # Report PSNR vs original (more meaningful for detail preservation)
            gt_image = gt_orig_comp

            loss_dict = gs_optim.collect_loss(
                gt_image,
                image,
                gt_alpha_mask=alpha,
                hybrid_gt_image_orig=gt_orig_comp,
                hybrid_gt_image_refined=gt_refined_comp,
                hybrid_changed_metric=args.hybrid_changed_metric,
                hybrid_changed_blur_ksize=args.hybrid_changed_blur_ksize,
                hybrid_changed_pool=args.hybrid_changed_pool,
                hybrid_changed_threshold=args.hybrid_changed_threshold,
                hybrid_changed_dilate_ksize=args.hybrid_changed_dilate_ksize,
                lambda_refined_inside=args.lambda_refined_inside,
                lambda_orig_outside=args.lambda_orig_outside,
                lambda_lpips_outside=args.lambda_lpips_outside,
                hybrid_debug=bool(debug_now),
                debug_vis=bool(debug_now),
                temporal_window_id=temporal_window_id,
                temporal_tag=temporal_tag,
                temporal_bg_key=temporal_bg_key,
            )
        else:
            loss_dict = gs_optim.collect_loss(
                gt_image,
                image,
                gt_alpha_mask=gt_alpha_mask,
                debug_vis=bool(debug_now),
                temporal_window_id=temporal_window_id,
                temporal_tag=temporal_tag,
                temporal_bg_key=temporal_bg_key,
            )

        loss = loss_dict['loss']

        # Pilot geo-ft: add geometry loss on sparse geo steps.
        if geo_ft_enabled and is_geo_step and (geo_spec is not None) and (geo_mesh_cache is not None):
            try:
                # Predicted posed cloth mesh verts (with offsets applied) from the current forward.
                V_pred = getattr(gs_model, "mesh_verts", None)
                F_pred = getattr(gs_model, "cano_faces", None)
                if not isinstance(V_pred, torch.Tensor):
                    raise RuntimeError("gs_model.mesh_verts not available for geo-ft")
                if V_pred.ndim != 2 or V_pred.shape[-1] != 3:
                    raise RuntimeError(f"Unexpected V_pred shape: {tuple(V_pred.shape)}")

                V_gt, F_gt = geo_mesh_cache.get_mesh_vf(geo_spec.mesh_path, device=V_pred.device, dtype=V_pred.dtype)
                tris = V_gt[F_gt]  # (T,3,3)
                points = V_pred  # (P,3)
                points_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                tris_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                max_points = int(points.shape[0])
                d2 = point_face_distance(points, points_first_idx, tris, tris_first_idx, max_points, 1e-8)  # (P,)
                # NOTE: depending on PyTorch3D op/version, the returned distance may be squared or not.
                # Default assumption: point_face_distance returns squared distances.
                geo_dist_is_squared = bool(_cfg_get(geo_ft_cfg, "geo_dist_is_squared", True))
                if geo_dist_is_squared:
                    d2_raw = d2
                    d = torch.sqrt(d2_raw.clamp_min(0.0) + 1e-12)
                else:
                    d = d2
                    d2_raw = d * d

                # Raw-distance diagnostics (do NOT affect gradients).
                try:
                    with torch.no_grad():
                        d2_det = d2_raw.detach().float()
                        d_det = d.detach().float()
                        d_flat = d_det.reshape(-1)
                        geo_rmse_m = torch.sqrt(d2_det.mean().clamp_min(0.0))
                        geo_p95_mm = torch.quantile(d_flat, 0.95) * 1000.0
                        geo_mean_mm = d_flat.mean() * 1000.0
                    loss_dict["geo_rmse_m"] = geo_rmse_m
                    loss_dict["geo_p95_mm"] = geo_p95_mm
                    loss_dict["geo_mean_mm"] = geo_mean_mm
                except Exception:
                    pass

                robust = str(_cfg_get(geo_ft_cfg, "robust_geo", "huber")).lower()
                if robust == "charbonnier":
                    eps = float(_cfg_get(geo_ft_cfg, "geo_charbonnier_eps", 1e-4))
                    geo_term = torch.sqrt(d * d + float(eps) * float(eps))
                else:
                    # default huber on distance (meters)
                    delta = float(_cfg_get(geo_ft_cfg, "geo_huber_delta", 0.01))
                    if delta <= 0:
                        geo_term = d
                    else:
                        geo_term = torch.where(d <= delta, 0.5 * (d * d) / delta, d - 0.5 * delta)
                loss_geo = geo_term.mean()

                lam_geo = float(_cfg_get(geo_ft_cfg, "lambda_geo", 0.0))
                warm = int(_cfg_get(geo_ft_cfg, "lambda_geo_warmup_steps", 0) or 0)
                if warm > 0:
                    ramp = float(min(1.0, max(0.0, float(iteration) / float(warm))))
                else:
                    ramp = 1.0
                lam_eff = float(lam_geo) * float(ramp)

                lam_rgb = float(_cfg_get(geo_ft_cfg, "lambda_rgb", 1.0))
                if lam_rgb != 1.0:
                    loss = loss * float(lam_rgb)

                loss = loss + lam_eff * loss_geo
                loss_dict["loss"] = loss
                loss_dict["loss_geo"] = loss_geo
                loss_dict["lambda_geo_eff"] = float(lam_eff)

                # Debug mesh dumps + per-vertex distance heatmap PLY (vertex colors + quality).
                #
                # Desired behavior: geo-vis dumps should follow the same cadence as other visualization artifacts:
                # - training vis cadence: save_every_iter
                # - eval vis cadence: testing_iterations
                #
                # `pilot_geo_ft.save_debug_every` is treated as a legacy fallback only when no other vis cadence exists.
                dump_every_cfg = int(_cfg_get(geo_ft_cfg, "save_debug_every", 0) or 0)
                has_vis_cadence = (int(save_every_iter) > 0) or (len(list(testing_iterations)) > 0)
                dump_every = int(dump_every_cfg) if (not has_vis_cadence) else 0
                do_dump = bool(debug_now) or (iteration == 1) or (
                    (int(save_every_iter) > 0 and iteration % int(save_every_iter) == 0)
                    or (iteration in testing_iterations)
                    or (dump_every > 0 and iteration % int(dump_every) == 0)
                )
                if do_dump and isinstance(F_pred, torch.Tensor):
                    try:
                        out_dir = (model_path / f"geo_ft_debug/iter_{iteration:06d}")
                        out_dir.mkdir(parents=True, exist_ok=True)
                        save_ply_mesh(str(out_dir / f"mesh_pred_posed_{geo_spec.frame_id_str}.ply"), V_pred, F_pred)
                        save_ply_mesh(str(out_dir / f"mesh_gt_{geo_spec.frame_id_str}.ply"), V_gt, F_gt)
                        colors = scalar_to_heatmap_vertex_colors(d.detach())
                        save_ply_mesh(
                            str(out_dir / f"mesh_pred_posed_{geo_spec.frame_id_str}_geo_err.ply"),
                            V_pred,
                            F_pred,
                            vert_colors=colors.to(V_pred.device, V_pred.dtype),
                            vert_quality=d.detach(),
                        )
                    except Exception:
                        pass
            except Exception as e:
                # Don't crash training; just skip geo loss for this step.
                loss_dict["pilot_geo_ft_error"] = str(e)
        loss.backward()

        # Hybrid debug dumps (saved alongside eval/checkpoints)
        if args.hybrid_supervision and (
            iteration == 1
            or iteration == total_iteration
            or (save_every_iter > 0 and iteration % save_every_iter == 0)
            or (iteration in testing_iterations)
        ):
            try:
                dbg_dir = model_path / "hybrid_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                libcore.write_tensor_image(str(dbg_dir / f"iter_{iteration:06d}_render.jpg"), image, rgb2bgr=True)
                libcore.write_tensor_image(str(dbg_dir / f"iter_{iteration:06d}_gt_orig.jpg"), loss_dict.get("hybrid_gt_orig_vis", gt_image), rgb2bgr=True)
                if "hybrid_gt_refined_vis" in loss_dict:
                    libcore.write_tensor_image(str(dbg_dir / f"iter_{iteration:06d}_gt_refined.jpg"), loss_dict["hybrid_gt_refined_vis"], rgb2bgr=True)
                if "hybrid_changed_mask_vis" in loss_dict:
                    # save mask as 3ch for convenience
                    m = loss_dict["hybrid_changed_mask_vis"]
                    m3 = m.repeat(3, 1, 1)
                    libcore.write_tensor_image(str(dbg_dir / f"iter_{iteration:06d}_changed_mask.jpg"), m3, rgb2bgr=True)
            except Exception as e:
                logger.warning(f"[HybridDebug] Failed to dump debug images: {e}")

        # FullbodyFix bake debug dumps (masks + pred/gt/diff + silhouette zoom)
        if fullbodyfix_bake_enabled and debug_now:
            try:
                dbg_dir = model_path / "fullbodyfix_bake_debug"
                dbg_dir.mkdir(parents=True, exist_ok=True)
                prefix = f"iter_{iteration:06d}"

                # Save core images
                libcore.write_tensor_image(str(dbg_dir / f"{prefix}_pred.jpg"), image, rgb2bgr=True)
                libcore.write_tensor_image(str(dbg_dir / f"{prefix}_gt.jpg"), gt_image, rgb2bgr=True)
                try:
                    diff = (image - gt_image).abs()
                    libcore.write_tensor_image(str(dbg_dir / f"{prefix}_diff.jpg"), diff, rgb2bgr=True)
                except Exception:
                    pass

                # Save masks (if provided by LossBase)
                def _save_mask_1hw(key: str, out_name: str):
                    m = loss_dict.get(key, None)
                    if isinstance(m, torch.Tensor):
                        if m.ndim == 2:
                            m1 = m.unsqueeze(0)
                        else:
                            m1 = m[:1]
                        m3 = m1.repeat(3, 1, 1)
                        libcore.write_tensor_image(str(dbg_dir / f"{prefix}_{out_name}.jpg"), m3, rgb2bgr=True)
                        return m1
                    return None

                m_inner = _save_mask_1hw("fbf_mask_inner_vis", "mask_inner")
                m_outer = _save_mask_1hw("fbf_mask_outer_vis", "mask_outer")
                m_band = _save_mask_1hw("fbf_mask_band_vis", "mask_band")
                _save_mask_1hw("fbf_mask_W_vis", "mask_W")
                _save_mask_1hw("fbf_mask_inner_patch_vis", "mask_inner_patch")

                # Silhouette zoom crop around band (fallback to outer)
                m_for_bbox = None
                if isinstance(m_band, torch.Tensor) and float(m_band.sum().detach().cpu().item()) > 0:
                    m_for_bbox = m_band
                elif isinstance(m_outer, torch.Tensor) and float(m_outer.sum().detach().cpu().item()) > 0:
                    m_for_bbox = m_outer

                if isinstance(m_for_bbox, torch.Tensor):
                    # bbox in pixel coords
                    yy, xx = torch.where(m_for_bbox.squeeze(0) > 0.5)
                    if yy.numel() > 0 and xx.numel() > 0:
                        y0 = int(yy.min().detach().cpu().item())
                        y1 = int(yy.max().detach().cpu().item()) + 1
                        x0 = int(xx.min().detach().cpu().item())
                        x1 = int(xx.max().detach().cpu().item()) + 1
                        pad = 32
                        H, W = int(image.shape[-2]), int(image.shape[-1])
                        y0 = max(0, y0 - pad)
                        x0 = max(0, x0 - pad)
                        y1 = min(H, y1 + pad)
                        x1 = min(W, x1 + pad)
                        pred_z = image[:, y0:y1, x0:x1]
                        gt_z = gt_image[:, y0:y1, x0:x1]
                        libcore.write_tensor_image(str(dbg_dir / f"{prefix}_zoom_pred.jpg"), pred_z, rgb2bgr=True)
                        libcore.write_tensor_image(str(dbg_dir / f"{prefix}_zoom_gt.jpg"), gt_z, rgb2bgr=True)
            except Exception as e:
                logger.warning(f"[FullbodyFixBake] Failed to dump debug visuals: {e}")

        # Only run densification if Gaussians are not frozen AND deformation is disabled.
        # (Tao/deformation training uses a fixed pretrained GS set.)
        if (not fullbodyfix_bake_enabled) and (not gs_model.gaussians_are_frozen) and (not getattr(gs_model, "use_deformation", False)):
            gs_optim.adaptive_density_control(render_pkg, iteration)
        else: # If frozen, ensure grad accumulators are zeroed (though not used by optimizer)
            if hasattr(gs_model, 'xyz_gradient_accum'): gs_model.xyz_gradient_accum.zero_()
            if hasattr(gs_model, 'denom'): gs_model.denom.zero_()

        if fullbodyfix_bake_enabled:
            _assert_fixed_gaussian_set(f"pre_step_iter_{iteration}")

        gs_optim.step()
        gs_optim.zero_grad(set_to_none=True)

        if fullbodyfix_bake_enabled:
            _assert_fixed_gaussian_set(f"post_step_iter_{iteration}")

        pbar_desc = {
            'L': f"{loss.item():.4f}",
            'PSNR': f"{loss_dict['psnr_full']:.2f}",
            'LR': f"{current_lr:.1e}",
            'Stg': gs_optim.current_stage
        }
        if not gs_model.gaussians_are_frozen: pbar_desc['#G'] = gs_model.num_gauss
        pbar.set_postfix(pbar_desc)

        # TensorBoard logging (lightweight)
        if writer is not None and (tb_log_every > 0) and (iteration % tb_log_every == 0):
            try:
                writer.add_scalar("train/loss", float(loss.item()), iteration)
                writer.add_scalar("train/psnr_full", float(loss_dict.get("psnr_full", 0.0)), iteration)
                writer.add_scalar("train/lr", float(current_lr), iteration)
                writer.add_scalar("train/stage", float(gs_optim.current_stage), iteration)
                if "loss_grad" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_grad", float(loss_dict["loss_grad"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "loss_dod_prev_cur" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_dod_prev_cur", float(loss_dict["loss_dod_prev_cur"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "loss_dod_cur_next" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_dod_cur_next", float(loss_dict["loss_dod_cur_next"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if ("loss_dod_prev_cur" in loss_dict) or ("loss_dod_cur_next" in loss_dict):
                    try:
                        a = float(loss_dict.get("loss_dod_prev_cur", torch.zeros(())).detach().cpu().item()) if "loss_dod_prev_cur" in loss_dict else 0.0
                        b = float(loss_dict.get("loss_dod_cur_next", torch.zeros(())).detach().cpu().item()) if "loss_dod_cur_next" in loss_dict else 0.0
                        writer.add_scalar("train/loss_dod_total", a + b, iteration)
                    except Exception:
                        pass
                if "loss_c_smooth" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_c_smooth", float(loss_dict["loss_c_smooth"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "loss_lbs_w_reg" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_lbs_w_reg", float(loss_dict["loss_lbs_w_reg"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "loss_lbs_w_dev" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_lbs_w_dev", float(loss_dict["loss_lbs_w_dev"].detach().cpu().item()), iteration)
                    except Exception:
                        pass

                # P1 joint-aware offset smoothing (stage1 regularizer)
                if "loss_joint_edge" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_joint_edge", float(loss_dict["loss_joint_edge"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "loss_joint_lap" in loss_dict:
                    try:
                        writer.add_scalar("train/loss_joint_lap", float(loss_dict["loss_joint_lap"].detach().cpu().item()), iteration)
                    except Exception:
                        pass
                if "joint_reg_ramp" in loss_dict:
                    try:
                        writer.add_scalar("joint_reg/ramp", float(loss_dict["joint_reg_ramp"]), iteration)
                    except Exception:
                        pass
                if "joint_reg_M_mean" in loss_dict:
                    try:
                        writer.add_scalar("joint_reg/M_mean", float(loss_dict["joint_reg_M_mean"]), iteration)
                    except Exception:
                        pass
                if "joint_reg_M_p95" in loss_dict:
                    try:
                        writer.add_scalar("joint_reg/M_p95", float(loss_dict["joint_reg_M_p95"]), iteration)
                    except Exception:
                        pass

                if getattr(gs_model, "use_deformation", False):
                    # Mean norms (current iteration sample)
                    v_off = getattr(gs_model, "_last_vertex_offset_l2_mean", None)
                    du = getattr(gs_model, "_last_delta_u_l2_mean", None)
                    if v_off is not None:
                        v_val = float(v_off.detach().cpu().item()) if hasattr(v_off, "detach") else float(v_off)
                        writer.add_scalar("deform/vertex_offset_l2_mean", v_val, iteration)
                        writer.add_scalar("deform/vertex_offset_l2_mean_ema", _ema_update("vertex_offset_l2_mean", v_val), iteration)
                    if du is not None:
                        du_val = float(du.detach().cpu().item()) if hasattr(du, "detach") else float(du)
                        writer.add_scalar("deform/gs_delta_u_l2_mean", du_val, iteration)
                        writer.add_scalar("deform/gs_delta_u_l2_mean_ema", _ema_update("gs_delta_u_l2_mean", du_val), iteration)
                    # Delta_u stats logging (uses same frequency as tensorboard.log_every).
                    try:
                        du_full = getattr(gs_model, "_current_delta_u_local_ntb", None)
                        if isinstance(du_full, torch.Tensor) and du_full.numel() > 0:
                            x = du_full.detach()
                            # subsample for speed if huge
                            n_max = 200_000
                            if x.shape[0] > n_max:
                                idx = torch.randint(0, x.shape[0], (n_max,), device=x.device)
                                x = x.index_select(0, idx)
                            n = torch.sqrt((x.float() * x.float()).sum(dim=-1) + 1e-12)  # [M]
                            du_mean = float(n.mean().cpu().item())
                            du_max = float(n.max().cpu().item())
                            try:
                                du_p99 = float(torch.quantile(n, 0.99).cpu().item())
                            except Exception:
                                du_p99 = float(n.kthvalue(max(int(0.99 * (n.numel() - 1)), 0)).values.cpu().item())
                            writer.add_scalar("deform/delta_u_mean", du_mean, iteration)
                            writer.add_scalar("deform/delta_u_max", du_max, iteration)
                            writer.add_scalar("deform/delta_u_p99", du_p99, iteration)
                            # over-tau fraction if tau is configured
                            tau = None
                            try:
                                deform_cfg = getattr(getattr(config, "model", {}), "deformation", None)
                                tau = getattr(deform_cfg, "delta_u_clamp_tau", None) if deform_cfg is not None else None
                            except Exception:
                                tau = None
                            if tau is not None:
                                tau_f = float(tau)
                                writer.add_scalar("deform/delta_u_over_tau", float((n > tau_f).float().mean().cpu().item()), iteration)
                    except Exception:
                        pass
                    # Clamp stats (if enabled)
                    for name, tag in [
                        ("_last_delta_u_max", "deform/delta_u_max"),
                        ("_last_delta_u_p99", "deform/delta_u_p99"),
                        ("_last_delta_u_frac_clamped", "deform/delta_u_frac_clamped"),
                    ]:
                        v = getattr(gs_model, name, None)
                        if v is not None:
                            try:
                                writer.add_scalar(tag, float(v.detach().cpu().item()), iteration)
                            except Exception:
                                pass

                    # TODO-4: LBS residual stats (support-only) for safety monitoring.
                    try:
                        mod = getattr(gs_model, "lbs_weight_residual", None)
                        if mod is not None:
                            m, mx, frac = mod.support_diff_stats(global_step=iteration)
                            writer.add_scalar("deform/lbs_w_diff_mean_abs", float(m.detach().cpu().item()), iteration)
                            writer.add_scalar("deform/lbs_w_diff_max_abs", float(mx.detach().cpu().item()), iteration)
                            writer.add_scalar("deform/lbs_w_diff_frac_gt_0p05", float(frac.detach().cpu().item()), iteration)
                    except Exception:
                        pass

                    # TAO offset-input enhancement: optional input feature stats (smpl_anchor mode)
                    try:
                        deform_cfg = getattr(getattr(config, "model", {}), "deformation", None)
                        input_mode = str(getattr(deform_cfg, "input_mode", "xyz")).lower() if deform_cfg is not None else "xyz"
                        log_in = bool(getattr(deform_cfg, "log_input_stats", False)) if deform_cfg is not None else False
                        if log_in and input_mode == "smpl_anchor":
                            net = getattr(getattr(gs_model, "deformation_module", None), "vertex_deformation_net", None)
                            st = getattr(net, "_last_anchor_stats", None) if net is not None else None
                            if isinstance(st, dict):
                                for k, v in st.items():
                                    try:
                                        writer.add_scalar(f"deform_anchor/{k}", float(v), iteration)
                                    except Exception:
                                        pass
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[TensorBoard] Failed to log scalars: {e}")
        # Pilot geo-ft TB logging (keep lightweight cadence)
        if writer is not None and geo_ft_enabled and (tb_log_every > 0) and (iteration % tb_log_every == 0):
            try:
                if "loss_geo" in loss_dict:
                    writer.add_scalar("train/loss_geo", float(loss_dict["loss_geo"].detach().cpu().item()), iteration)
                if "lambda_geo_eff" in loss_dict:
                    writer.add_scalar("train/lambda_geo_eff", float(loss_dict["lambda_geo_eff"]), iteration)
                if "geo_rmse_m" in loss_dict:
                    writer.add_scalar("train/geo_rmse_m", float(loss_dict["geo_rmse_m"].detach().cpu().item()), iteration)
                if "geo_p95_mm" in loss_dict:
                    writer.add_scalar("train/geo_p95_mm", float(loss_dict["geo_p95_mm"].detach().cpu().item()), iteration)
                if "geo_mean_mm" in loss_dict:
                    writer.add_scalar("train/geo_mean_mm", float(loss_dict["geo_mean_mm"].detach().cpu().item()), iteration)
            except Exception:
                pass

        # NOTE: We no longer export a separate /lbs_delta_norm/step_*.ply here.
        # Delta-norm meshes are saved alongside existing mesh dumps (train debug + eval ply).

        if (not fullbodyfix_bake_enabled) and (not gs_model.gaussians_are_frozen) and (not getattr(config.model, 'free_gaussians', False)):
            gs_optim.update_trangle_walk(iteration)

        if (iteration in testing_iterations) and (not getattr(config.model, 'free_gaussians', False)):
            bg_color_test = 'black' if args.bg_color == 'random' else args.bg_color
            stats = run_testing(pipe, frameset_test, gs_model, str(model_path), iteration, verify=verify, bg_color=bg_color_test, optimizer_config=config.optim)
            if writer is not None and stats is not None:
                try:
                    writer.add_scalar("val/psnr", float(stats.get("psnr", 0.0)), iteration)
                    writer.add_scalar("val/ssim", float(stats.get("ssim", 0.0)), iteration)
                    writer.add_scalar("val/lpips", float(stats.get("lpips", 0.0)), iteration)
                    if "n_gauss" in stats:
                        writer.add_scalar("val/n_gauss", float(stats["n_gauss"]), iteration)
                except Exception as e:
                    logger.warning(f"[TensorBoard] Failed to log val scalars: {e}")

        # Pilot geo-ft: periodic geo-only evaluation over all K frames (no rendering needed).
        if geo_ft_enabled and (geo_frames is not None) and (geo_mesh_cache is not None):
            try:
                eval_every = int(_cfg_get(geo_ft_cfg, "eval_every", 0) or 0)
            except Exception:
                eval_every = 0
            geo_eval_due = (eval_every > 0) and (iteration % eval_every == 0)
            if geo_eval_due:
                try:
                    def _pair_lookup_idx(frame_id_str: str, cam_id) -> int | None:
                        fid_s = str(frame_id_str)
                        cam_s = str(cam_id)
                        # try exact
                        hit = pair_to_idx.get((fid_s, cam_s), None)
                        if hit is not None:
                            return int(hit)
                        # try de-padded int string
                        try:
                            fid_i = int(fid_s)
                        except Exception:
                            return None
                        # raw int string
                        hit = pair_to_idx.get((str(fid_i), cam_s), None)
                        if hit is not None:
                            return int(hit)
                        # dataset padding (e.g. 4 digits)
                        try:
                            fid_pad = f"{fid_i:0{int(frame_digits)}d}"
                            hit = pair_to_idx.get((fid_pad, cam_s), None)
                            if hit is not None:
                                return int(hit)
                        except Exception:
                            pass
                        # legacy 6-digit padding (geo GT dirs)
                        try:
                            fid_pad6 = f"{fid_i:06d}"
                            hit = pair_to_idx.get((fid_pad6, cam_s), None)
                            if hit is not None:
                                return int(hit)
                        except Exception:
                            pass
                        return None

                    total = 0.0
                    count = 0
                    rmse_sum = 0.0
                    p95_sum_mm = 0.0
                    mean_sum_mm = 0.0
                    with torch.no_grad():
                        for spec in geo_frames:
                            # pick any existing cam for this frame (prefer first geo cam id)
                            idx = None
                            for cam_id in geo_cam_ids:
                                idx = _pair_lookup_idx(str(spec.frame_id_str), cam_id)
                                if idx is not None:
                                    break
                            if idx is None:
                                continue
                            sample = frameset_train[int(idx)]
                            frm_idx = sample["frm_idx"]
                            mi = sample["mesh_info"]
                            if mi is None or "smplx_params_raw_for_deformnet" not in mi:
                                continue
                            smplx_params_dict = mi["smplx_params_raw_for_deformnet"]
                            gs_model.update_to_posed_mesh(
                                raw_mesh_info_for_no_deform=mi,
                                current_frame_idx=frm_idx,
                                current_target_frame_smpl_params=smplx_params_dict,
                            )
                            V_pred = getattr(gs_model, "mesh_verts", None)
                            if not isinstance(V_pred, torch.Tensor):
                                continue
                            V_gt, F_gt = geo_mesh_cache.get_mesh_vf(spec.mesh_path, device=V_pred.device, dtype=V_pred.dtype)
                            tris = V_gt[F_gt]
                            points = V_pred
                            points_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                            tris_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                            d2 = point_face_distance(points, points_first_idx, tris, tris_first_idx, int(points.shape[0]), 1e-8)
                            geo_dist_is_squared = bool(_cfg_get(geo_ft_cfg, "geo_dist_is_squared", True))
                            if geo_dist_is_squared:
                                d2_raw = d2
                                d = torch.sqrt(d2_raw.clamp_min(0.0) + 1e-12)
                            else:
                                d = d2
                                d2_raw = d * d
                            d2_det = d2_raw.detach().float()
                            d_det = d.detach().float()
                            d_flat = d_det.reshape(-1)
                            rmse = torch.sqrt(d2_det.mean().clamp_min(0.0))
                            p95_mm = torch.quantile(d_flat, 0.95) * 1000.0
                            mean_mm = d_flat.mean() * 1000.0

                            total += float(d_det.mean().detach().cpu().item())
                            rmse_sum += float(rmse.detach().cpu().item())
                            p95_sum_mm += float(p95_mm.detach().cpu().item())
                            mean_sum_mm += float(mean_mm.detach().cpu().item())
                            count += 1
                    mean_d = (total / max(count, 1))
                    mean_rmse = (rmse_sum / max(count, 1))
                    mean_p95_mm = (p95_sum_mm / max(count, 1))
                    mean_mean_mm = (mean_sum_mm / max(count, 1))
                    print(f"[pilot_geo_ft] geo_eval@{iteration}: mean_point2mesh={mean_d:.6f} over {count}/{len(geo_frames)} frames")
                    if writer is not None:
                        try:
                            writer.add_scalar("val_geo/mean_point2mesh", float(mean_d), iteration)
                            writer.add_scalar("val_geo/count", float(count), iteration)
                            writer.add_scalar("val_geo/geo_rmse_m", float(mean_rmse), iteration)
                            writer.add_scalar("val_geo/geo_p95_mm", float(mean_p95_mm), iteration)
                            writer.add_scalar("val_geo/geo_mean_mm", float(mean_mean_mm), iteration)
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[pilot_geo_ft] geo_eval failed: {e}")

            # Geometry-vis mesh dumps should follow train/eval vis events, NOT only geo-steps.
            # Otherwise, if the current iteration happens to be an RGB step, no geo meshes are written.
            geo_vis_due = bool(debug_now) or (iteration == 1) or (
                (int(save_every_iter) > 0 and iteration % int(save_every_iter) == 0) or (iteration in testing_iterations)
            )
            if geo_vis_due:
                try:
                    def _pair_lookup_idx(frame_id_str: str, cam_id) -> int | None:
                        fid_s = str(frame_id_str)
                        cam_s = str(cam_id)
                        hit = pair_to_idx.get((fid_s, cam_s), None)
                        if hit is not None:
                            return int(hit)
                        try:
                            fid_i = int(fid_s)
                        except Exception:
                            return None
                        hit = pair_to_idx.get((str(fid_i), cam_s), None)
                        if hit is not None:
                            return int(hit)
                        try:
                            fid_pad = f"{fid_i:0{int(frame_digits)}d}"
                            hit = pair_to_idx.get((fid_pad, cam_s), None)
                            if hit is not None:
                                return int(hit)
                        except Exception:
                            pass
                        try:
                            fid_pad6 = f"{fid_i:06d}"
                            hit = pair_to_idx.get((fid_pad6, cam_s), None)
                            if hit is not None:
                                return int(hit)
                        except Exception:
                            pass
                        return None

                    # Limit dumps to a small number of frames to keep artifacts manageable.
                    first_k = 1
                    try:
                        ev = getattr(getattr(config.model, "eval_variants", {}), "dump_ply_first_k", None)
                        if ev is not None:
                            first_k = int(ev)
                    except Exception:
                        first_k = 1
                    first_k = max(1, min(int(first_k), int(len(geo_frames))))

                    out_dir = (model_path / f"geo_ft_debug/iter_{iteration:06d}")
                    out_dir.mkdir(parents=True, exist_ok=True)
                    wrote = 0
                    with torch.no_grad():
                        for spec in list(geo_frames)[:first_k]:
                            idx = None
                            for cam_id in geo_cam_ids:
                                idx = _pair_lookup_idx(str(spec.frame_id_str), cam_id)
                                if idx is not None:
                                    break
                            if idx is None:
                                continue
                            sample = frameset_train[int(idx)]
                            frm_idx = sample["frm_idx"]
                            mi = sample["mesh_info"]
                            if mi is None or "smplx_params_raw_for_deformnet" not in mi:
                                continue
                            smplx_params_dict = mi["smplx_params_raw_for_deformnet"]
                            gs_model.update_to_posed_mesh(
                                raw_mesh_info_for_no_deform=mi,
                                current_frame_idx=frm_idx,
                                current_target_frame_smpl_params=smplx_params_dict,
                            )
                            V_pred = getattr(gs_model, "mesh_verts", None)
                            F_pred = getattr(gs_model, "cano_faces", None)
                            if (not isinstance(V_pred, torch.Tensor)) or (not isinstance(F_pred, torch.Tensor)):
                                continue
                            V_gt, F_gt = geo_mesh_cache.get_mesh_vf(spec.mesh_path, device=V_pred.device, dtype=V_pred.dtype)
                            tris = V_gt[F_gt]
                            points = V_pred
                            points_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                            tris_first_idx = torch.zeros((1,), device=points.device, dtype=torch.int64)
                            d2 = point_face_distance(points, points_first_idx, tris, tris_first_idx, int(points.shape[0]), 1e-8)
                            d = torch.sqrt(d2.clamp_min(0.0) + 1e-12)
                            colors = scalar_to_heatmap_vertex_colors(d.detach())
                            save_ply_mesh(str(out_dir / f"mesh_pred_posed_{spec.frame_id_str}_vis.ply"), V_pred, F_pred)
                            save_ply_mesh(str(out_dir / f"mesh_gt_{spec.frame_id_str}_vis.ply"), V_gt, F_gt)
                            save_ply_mesh(
                                str(out_dir / f"mesh_pred_posed_{spec.frame_id_str}_geo_err_vis.ply"),
                                V_pred,
                                F_pred,
                                vert_colors=colors.to(V_pred.device, V_pred.dtype),
                                vert_quality=d.detach(),
                            )
                            wrote += 1
                    if wrote == 0:
                        logger.warning("[pilot_geo_ft] geo_vis dump wrote 0 meshes.")
                        try:
                            if not geo_vis_warned_once:
                                geo_vis_warned_once = True
                                # Print a small, high-signal debug payload once.
                                spec0 = list(geo_frames)[0]
                                fid_s = str(spec0.frame_id_str)
                                fid_i = int(fid_s) if str(fid_s).isdigit() else None
                                fid_pad = f"{int(fid_i):0{int(frame_digits)}d}" if fid_i is not None else None
                                fid_pad6 = f"{int(fid_i):06d}" if fid_i is not None else None
                                cams_preview = list(geo_cam_ids)[:5]
                                logger.warning(
                                    f"[pilot_geo_ft][debug] frame_digits={int(frame_digits)} "
                                    f"spec0.frame_id_str={fid_s} fid_i={fid_i} fid_pad={fid_pad} fid_pad6={fid_pad6} "
                                    f"geo_cam_ids[:5]={cams_preview}"
                                )
                                # Show whether any candidate key exists for the preview cams.
                                for c in cams_preview:
                                    keys = [
                                        (fid_s, str(c)),
                                        (str(fid_i), str(c)) if fid_i is not None else None,
                                        (fid_pad, str(c)) if fid_pad is not None else None,
                                        (fid_pad6, str(c)) if fid_pad6 is not None else None,
                                    ]
                                    keys = [k for k in keys if k is not None]
                                    exists = [pair_to_idx.get(k, None) is not None for k in keys]
                                    logger.warning(f"[pilot_geo_ft][debug] cam={c} tried={keys} exists={exists}")
                                # Also show a few sample keys from the dataset index.
                                sample_keys = list(pair_to_idx.keys())[:10]
                                logger.warning(f"[pilot_geo_ft][debug] pair_to_idx sample keys[:10]={sample_keys}")
                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"[pilot_geo_ft] geo_vis dump failed: {e}")

            # Optional: 3-frame temporal window validation on specified starts (deformation mode only).
            if bool(getattr(config, "use_deformation", False)):
                try:
                    # Configured under dataset.avatarrex_config for structural consistency with val/test frames.
                    tw_cfg = None
                    ds = getattr(config, "dataset", None)
                    if ds is not None:
                        try:
                            av_cfg = getattr(ds, "avatarrex_config", None)
                        except Exception:
                            av_cfg = None
                        if av_cfg is None and isinstance(ds, dict):
                            av_cfg = ds.get("avatarrex_config", None)
                        if av_cfg is not None:
                            try:
                                tw_cfg = getattr(av_cfg, "temporal_window_val", None)
                            except Exception:
                                tw_cfg = av_cfg.get("temporal_window_val", None) if isinstance(av_cfg, dict) else None
                    enabled = False
                    if tw_cfg is not None:
                        try:
                            enabled = bool(getattr(tw_cfg, "enabled", False))
                        except Exception:
                            enabled = bool(tw_cfg.get("enabled", False)) if isinstance(tw_cfg, dict) else False
                    # Run temporal validation on a cadence (NOT every training step).
                    # Default: reuse pilot_geo_ft.eval_every unless overridden.
                    tw_every = 0
                    if tw_cfg is not None:
                        try:
                            tw_every = int(getattr(tw_cfg, "every", 0) or 0)
                        except Exception:
                            tw_every = int(tw_cfg.get("every", 0) or 0) if isinstance(tw_cfg, dict) else 0
                        if tw_every <= 0:
                            try:
                                tw_every = int(getattr(tw_cfg, "eval_every", 0) or 0)
                            except Exception:
                                tw_every = int(tw_cfg.get("eval_every", 0) or 0) if isinstance(tw_cfg, dict) else 0
                    if tw_every <= 0:
                        tw_every = int(eval_every) if int(eval_every) > 0 else 0

                    temporal_eval_due = (tw_every > 0) and (iteration % tw_every == 0)
                    if enabled and temporal_eval_due:
                        cam = None
                        starts = []
                        bg_mode = "random"
                        try:
                            cam = getattr(tw_cfg, "cam", None)
                            starts = list(getattr(tw_cfg, "window_starts", []))
                            bg_mode = str(getattr(tw_cfg, "bg", "random"))
                        except Exception:
                            if isinstance(tw_cfg, dict):
                                cam = tw_cfg.get("cam", None)
                                starts = list(tw_cfg.get("window_starts", []))
                                bg_mode = str(tw_cfg.get("bg", "random"))
                        if cam is None:
                            try:
                                cam = getattr(getattr(config.optim, "debug_vis", {}), "cam", 1)
                            except Exception:
                                cam = 1
                        # Temporal window validation is an eval-time artifact.
                        # IMPORTANT: it needs (t,t+1,t+2) frames even if the normal val split filters them out.
                        # So we build a dedicated val-frameset that unions:
                        #   existing val_frames + all window triple frames, and uses only the requested camera.
                        try:
                            ds_tw = OmegaConf.create(OmegaConf.to_container(config.dataset, resolve=False))
                            av = getattr(ds_tw, "avatarrex_config", None)
                            if av is not None:
                                # Extend val frames with all temporal triples.
                                base_val_frames = []
                                try:
                                    vf = getattr(av, "val_frames", None)
                                    if vf is not None:
                                        base_val_frames = list(vf)
                                except Exception:
                                    base_val_frames = []
                                tri = []
                                for s in starts:
                                    try:
                                        s_int = int(s)
                                    except Exception:
                                        continue
                                    tri.extend([s_int, s_int + 1, s_int + 2])
                                merged = []
                                seen = set()
                                for x in (base_val_frames + tri):
                                    try:
                                        xi = int(x)
                                    except Exception:
                                        continue
                                    if xi not in seen:
                                        seen.add(xi)
                                        merged.append(xi)
                                av.val_frames = merged
                                # Force camera set to this one camera.
                                try:
                                    av.val_cameras = [int(cam)]
                                except Exception:
                                    av.val_cameras = [cam]
                            frameset_tw = make_frameset_data(ds_tw, split="val")
                        except Exception:
                            frameset_tw = frameset_test
                        eval_dir = os.path.join(str(model_path), f"eval_{iteration}")
                        tw_stats = run_temporal_window_validation(
                            pipe=pipe,
                            frameset=frameset_tw,
                            gs_model=gs_model,
                            out_dir=eval_dir,
                            optimizer_config=config.optim,
                            window_starts=starts,
                            cam_spec=cam,
                            bg_mode=bg_mode,
                        )
                        if writer is not None and isinstance(tw_stats, dict):
                            try:
                                writer.add_scalar("val_temporal/dod_prev_cur", float(tw_stats.get("dod_prev_cur", 0.0)), iteration)
                                writer.add_scalar("val_temporal/dod_cur_next", float(tw_stats.get("dod_cur_next", 0.0)), iteration)
                                writer.add_scalar("val_temporal/dod_total", float(tw_stats.get("dod_total", 0.0)), iteration)
                                writer.add_scalar("val_temporal/count", float(tw_stats.get("count", 0.0)), iteration)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"[TemporalVal] Failed temporal window validation: {e}")

        if iteration % save_every_iter == 0 or iteration == 1 or iteration == total_iteration:
            print(f"\nSaving checkpoint at iteration {iteration}...")
            # Checkpoint saving now includes model state (which has deformation net) and optimizer state
            checkpoint_data = {
                'iteration': iteration,
                'gs_model_state_dict': gs_model.state_dict(),
                'gs_optim_state_dict': gs_optim.optimizer.state_dict(), # Save Adam state
                # Add any other scheduler states if needed: e.g. 'gs_optim_schedulers_state_dict': [s.state_dict() for s in gs_optim.schedulers]
            }
            checkpoint_path = model_path / f"chkpnt_iter_{iteration}.pth"
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

            # Optionally, save PLY and images as before (from original save_checkpoint)
            if config.get('save_ply_during_training', True):
                pc_dir = model_path / f'point_cloud/iteration_{iteration}'
                pc_dir.mkdir(parents=True, exist_ok=True)
                if not getattr(config.model, 'free_gaussians', False):
                    gs_model.update_to_cano_mesh()  # Canonical Gaussians (used for resume/load)
                gs_model.save_ply(str(pc_dir / 'point_cloud.ply'))  # Not the same as gs_*.ply (those are posed/deformed variants)
                if (not getattr(config.model, 'free_gaussians', False)) and hasattr(gs_model, 'save_embedding_json'):
                    gs_model.save_embedding_json(str(pc_dir / 'embedding.json'))

                # Optional: P1 joint_reg debug PLY dumps (mask + dV norm) on canonical mesh.
                try:
                    jr_cfg = getattr(getattr(config, "optim", {}), "joint_reg", None)
                except Exception:
                    jr_cfg = None
                try:
                    jr_enabled = bool(getattr(jr_cfg, "enabled", False)) if jr_cfg is not None else False
                except Exception:
                    jr_enabled = bool(jr_cfg.get("enabled", False)) if isinstance(jr_cfg, dict) else False
                if jr_enabled:
                    try:
                        dump_every = int(getattr(jr_cfg, "dump_ply_every", 0) or 0) if jr_cfg is not None else 0
                    except Exception:
                        dump_every = int(jr_cfg.get("dump_ply_every", 0) or 0) if isinstance(jr_cfg, dict) else 0
                    if dump_every > 0 and (iteration % dump_every == 0):
                        try:
                            from utils.ply_io import save_ply_mesh
                            from utils.lbs_vis import scalar_to_heatmap_vertex_colors
                            out_jr = pc_dir / "joint_reg"
                            out_jr.mkdir(parents=True, exist_ok=True)
                            faces = getattr(gs_model, "cano_faces", None)
                            V0 = getattr(gs_model, "cano_verts_orig_for_deform", None)
                            if faces is not None and V0 is not None:
                                M = None
                                try:
                                    M = gs_model.ensure_joint_reg_mask(jr_cfg)
                                except Exception:
                                    M = getattr(gs_model, "_joint_reg_mask_v", None)
                                if M is not None:
                                    save_ply_mesh(out_jr / "mask_M.ply", V0, faces, vert_colors=scalar_to_heatmap_vertex_colors(M.to(V0.device, V0.dtype)), vert_quality=M)
                                dV = getattr(gs_model, "_current_vertex_offsets", None)
                                if isinstance(dV, torch.Tensor):
                                    dn = dV.norm(dim=1)
                                    save_ply_mesh(out_jr / "dV_norm.ply", V0, faces, vert_colors=scalar_to_heatmap_vertex_colors(dn.to(V0.device, V0.dtype)), vert_quality=dn)
                        except Exception:
                            pass
                # Save deterministic debug images if requested; otherwise use current training batch.
                # Save deterministic debug images if requested; otherwise use current training batch.
                bg_vis = 'white' if args.bg_color == 'random' else args.bg_color
                saved = False
                # Optional: dump debug meshes/transforms for alignment debugging.
                try:
                    debug_dump_cfg = getattr(getattr(config, 'optim', {}), 'debug_dump_alignment', None)
                except Exception:
                    debug_dump_cfg = None
                debug_dump_enabled = False
                debug_dump_iters = [1]
                if debug_dump_cfg is not None:
                    try:
                        debug_dump_enabled = bool(getattr(debug_dump_cfg, 'enabled', True))
                        iters = getattr(debug_dump_cfg, 'iters', None)
                        if iters is not None:
                            debug_dump_iters = list(iters) if isinstance(iters, (list, tuple)) else [int(iters)]
                    except Exception:
                        debug_dump_enabled = True
                        debug_dump_iters = [1]

                if (debug_vis_idx is not None) and (debug_frameset is not None) and (len(debug_frameset) > 0) and (not getattr(config.model, 'free_gaussians', False)):
                    try:
                        dbg_batch = debug_frameset[debug_vis_idx]
                        dbg_mesh = dbg_batch.get('mesh_info')
                        dbg_scene_cams = dbg_batch.get('scene_cameras')
                        if dbg_mesh is not None and dbg_scene_cams is not None and len(dbg_scene_cams) > 0:
                            smplx_params_dict_dbg = dbg_mesh.get('smplx_params_raw_for_deformnet') if config.use_deformation else None
                            dbg_view_cam = dbg_scene_cams[0].cuda()
                            if not config.use_deformation:
                                # Non-deformation: keep old single render/GT only.
                                gs_model.update_to_posed_mesh(dbg_mesh)
                                dbg_render_pkg = gs_model.render_to_camera(dbg_view_cam, pipe, background=bg_vis)
                                libcore.write_tensor_image(str(pc_dir / 'gt_image_train.jpg'), dbg_render_pkg['gt_image'], rgb2bgr=True)
                                libcore.write_tensor_image(str(pc_dir / 'render_train.jpg'), dbg_render_pkg['render'], rgb2bgr=True)
                                saved = True
                            else:
                                variants = [("raw_lbs", False, False), ("offset", True, False), ("full", True, True)]
                                pred_imgs = {}
                                gt_dbg = None
                                alpha_dbg = None
                                for vname, v_apply_vert, v_apply_bs in variants:
                                    gs_model.update_to_posed_mesh(
                                        raw_mesh_info_for_no_deform=dbg_mesh,
                                        current_frame_idx=dbg_batch.get('frm_idx'),
                                        current_target_frame_smpl_params=smplx_params_dict_dbg,
                                        apply_vertex_offsets=v_apply_vert,
                                        apply_gauss_bs=v_apply_bs,
                                    )
                                    dbg_render_pkg = gs_model.render_to_camera(dbg_view_cam, pipe, background=bg_vis)
                                    pred_imgs[vname] = dbg_render_pkg['render']
                                    gt_dbg = dbg_render_pkg['gt_image'] if gt_dbg is None else gt_dbg
                                    alpha_dbg = dbg_render_pkg.get('gt_alpha_mask', alpha_dbg)

                                    # Save intermediate mesh PLYs only for raw_lbs/offset (BS does not affect mesh),
                                    # but save Gaussian PLYs for all variants (BS affects Gaussians).
                                    if vname in ("raw_lbs", "offset"):
                                        try:
                                            if getattr(gs_model, 'mesh_verts', None) is not None and getattr(gs_model, 'cano_faces', None) is not None:
                                                mesh_vis_cfg = getattr(getattr(config, 'optim', {}), 'mesh_vis', None)
                                                lbs_color_mode = "argmax"
                                                dump_lbs_colors = True
                                                if mesh_vis_cfg is not None:
                                                    lbs_color_mode = str(getattr(mesh_vis_cfg, "lbs_color_mode", "argmax"))
                                                    dump_lbs_colors = bool(getattr(mesh_vis_cfg, "dump_lbs_colors", True))
                                                colors = None
                                                if dump_lbs_colors and getattr(gs_model, "lbs_weights", None) is not None:
                                                    # Use updated LBS weights when residual is enabled (updated-only vis).
                                                    w_vis = gs_model.get_lbs_weights_for_skinning() if hasattr(gs_model, "get_lbs_weights_for_skinning") else gs_model.lbs_weights
                                                    colors = lbs_weights_to_vertex_colors(w_vis, mode=lbs_color_mode)
                                                _save_ply_simple(str(pc_dir / f"mesh_{vname}.ply"), gs_model.mesh_verts, gs_model.cano_faces, colors=colors)
                                                # LBS v1.1: same mesh with vertex quality = delta_norm (|w - w0|)
                                                try:
                                                    mod = getattr(gs_model, "lbs_weight_residual", None)
                                                    if mod is not None:
                                                        lbs_cfg = getattr(getattr(config, "model", config), "lbs_weight_residual", None)
                                                        vis_cfg = getattr(lbs_cfg, "vis", None) if lbs_cfg is not None else None
                                                        metric = str(getattr(vis_cfg, "delta_norm_metric", "l1")) if vis_cfg is not None else "l1"
                                                        # Use projected baseline so q==0 before LBS is enabled/trained.
                                                        q = mod.delta_norm_per_vertex(global_step=iteration, metric=metric, baseline="projected")
                                                        from utils.lbs_vis import scalar_to_heatmap_vertex_colors
                                                        q_colors = scalar_to_heatmap_vertex_colors(q)
                                                        save_ply_mesh(
                                                            str(pc_dir / f"mesh_{vname}_delta_norm.ply"),
                                                            gs_model.mesh_verts,
                                                            gs_model.cano_faces,
                                                            vert_colors=q_colors,
                                                            vert_quality=q,
                                                        )
                                                except Exception:
                                                    pass
                                        except Exception:
                                            pass
                                    # gs_* variant PLYs are posed/deformed (different from point_cloud.ply). Set save_gs_variant_ply: false to skip and save space.
                                    if config.get('save_gs_variant_ply', True):
                                        try:
                                            gs_model.save_ply(str(pc_dir / f"gs_{vname}.ply"))
                                        except Exception:
                                            pass

                            if debug_dump_enabled and (iteration in debug_dump_iters) and config.use_deformation:
                                try:
                                    dump_dir = pc_dir / "align_debug"
                                    dump_dir.mkdir(parents=True, exist_ok=True)
                                    # 1) Dump canonical clothed mesh (the one used as LBS reference basis)
                                    if getattr(gs_model, 'cano_verts_used_for_lbs_basis', None) is not None and getattr(gs_model, 'cano_faces', None) is not None:
                                        _save_ply_simple(
                                            str(dump_dir / "debug_clothed_cano_for_lbs.ply"),
                                            gs_model.cano_verts_used_for_lbs_basis,
                                            gs_model.cano_faces,
                                        )
                                    # 2) Dump current skinned clothed mesh (our posed mesh_verts)
                                    if getattr(gs_model, 'mesh_verts', None) is not None and getattr(gs_model, 'cano_faces', None) is not None:
                                        _save_ply_simple(
                                            str(dump_dir / "debug_clothed_skinned.ply"),
                                            gs_model.mesh_verts,
                                            gs_model.cano_faces,
                                        )

                                    # 3) Dump SMPL-X body meshes (internal, canonical, external-corrected)
                                    if getattr(gs_model, 'smpl_model_for_lbs', None) is not None and smplx_params_dict_dbg is not None:
                                        p = gs_model._batch_smpl_params(smplx_params_dict_dbg)
                                        faces_smpl = torch.tensor(gs_model.smpl_model_for_lbs.faces.astype(np.int64), device=p['body_pose'].device)
                                        # LBS colors for SMPL-X mesh
                                        smpl_lbs = getattr(gs_model.smpl_model_for_lbs, "lbs_weights", None)
                                        smpl_colors = None
                                        if smpl_lbs is not None:
                                            try:
                                                mesh_vis_cfg = getattr(getattr(config, 'optim', {}), 'mesh_vis', None)
                                                lbs_color_mode = "argmax"
                                                dump_lbs_colors = True
                                                if mesh_vis_cfg is not None:
                                                    lbs_color_mode = str(getattr(mesh_vis_cfg, "lbs_color_mode", "argmax"))
                                                    dump_lbs_colors = bool(getattr(mesh_vis_cfg, "dump_lbs_colors", True))
                                                if dump_lbs_colors:
                                                    smpl_colors = lbs_weights_to_vertex_colors(smpl_lbs.to(p['body_pose'].device), mode=lbs_color_mode)
                                            except Exception:
                                                smpl_colors = None

                                        # Internal forward (whatever params carry)
                                        out_int = gs_model.smpl_model_for_lbs(**p, return_verts=True, return_full_pose=False)
                                        if out_int.vertices is not None:
                                            _save_ply_simple(
                                                str(dump_dir / "debug_smplx_body_internal.ply"),
                                                out_int.vertices[0],
                                                faces_smpl,
                                                smpl_colors,
                                            )

                                        # Canonical regen: global_orient=0, transl=0 (keep scale)
                                        p0 = dict(p)
                                        if 'global_orient' in p0:
                                            p0['global_orient'] = torch.zeros_like(p0['global_orient'])
                                        if 'transl' in p0:
                                            p0['transl'] = torch.zeros_like(p0['transl'])
                                        out_c = gs_model.smpl_model_for_lbs(**p0, return_verts=True, return_full_pose=False)
                                        if out_c.vertices is not None:
                                            _save_ply_simple(
                                                str(dump_dir / "debug_smplx_body_canon_go0_tr0.ply"),
                                                out_c.vertices[0],
                                                faces_smpl,
                                                smpl_colors,
                                            )

                                            # External correction (ActorsHQ-style): v_world = R(go)*v_canon + tr
                                            if getattr(gs_model, 'smpl_global_transform_mode', 'internal') == 'external' and ('global_orient' in p) and ('transl' in p):
                                                R = axis_angle_to_matrix(p['global_orient'])  # (1,3,3)
                                                v = out_c.vertices  # (1,V,3)
                                                v_ext = torch.einsum('bij,bvj->bvi', R, v) + p['transl'].unsqueeze(1)
                                                _save_ply_simple(
                                                    str(dump_dir / "debug_smplx_body_external_corrected.ply"),
                                                    v_ext[0],
                                                    faces_smpl,
                                                    smpl_colors,
                                                )

                                        # Dump params too
                                        try:
                                            import json as _json
                                            params_cpu = {}
                                            for k, v in p.items():
                                                if isinstance(v, torch.Tensor):
                                                    params_cpu[k] = v.detach().cpu().tolist()
                                            with open(dump_dir / "debug_smplx_params_used.json", "w") as f:
                                                _json.dump(params_cpu, f, indent=2)
                                        except Exception:
                                            pass
                                except Exception as e:
                                    print(f"Warning: debug dump failed: {e}")

                            if config.use_deformation:
                                # Save images
                                libcore.write_tensor_image(str(pc_dir / 'gt_image_train.jpg'), gt_dbg, rgb2bgr=True)
                                for vname, _, _ in variants:
                                    libcore.write_tensor_image(str(pc_dir / f'render_train_{vname}.jpg'), pred_imgs[vname], rgb2bgr=True)
                                # Back-compat filename points to full
                                libcore.write_tensor_image(str(pc_dir / 'render_train.jpg'), pred_imgs["full"], rgb2bgr=True)
                                # Loss debug maps should respect the configured debug view (cam/frame).
                                try:
                                    dbg_loss_dict = gs_optim.collect_loss(
                                        gt_dbg,
                                        pred_imgs["full"],
                                        gt_alpha_mask=alpha_dbg,
                                        debug_vis=True,
                                    )
                                    if isinstance(dbg_loss_dict, dict) and ("grad_diff_vis" in dbg_loss_dict):
                                        libcore.write_tensor_image(str(pc_dir / "grad_diff_heatmap.jpg"), dbg_loss_dict["grad_diff_vis"], rgb2bgr=True)
                                except Exception as e:
                                    print(f"Warning: failed to compute debug grad heatmap: {e}")
                                saved = True
                    except Exception as e:
                        print(f"Warning: debug vis render failed, falling back to current batch: {e}")
                if not saved:
                    libcore.write_tensor_image(str(pc_dir / 'gt_image_train.jpg'), gt_image, rgb2bgr=True)
                    libcore.write_tensor_image(str(pc_dir / 'render_train.jpg'), image, rgb2bgr=True)
                # Optional debug maps from losses (grad/DoD/etc.)
                try:
                    # NOTE: grad heatmap is computed above for the debug vis sample when available.
                    # Keep DoD maps from training step (they require temporal cache).
                    if isinstance(loss_dict, dict) and ("dod_err_vis_prev_cur" in loss_dict):
                        libcore.write_tensor_image(str(pc_dir / "dod_err_heatmap_prev_cur.jpg"), loss_dict["dod_err_vis_prev_cur"], rgb2bgr=True)
                    if isinstance(loss_dict, dict) and ("dod_err_vis_cur_next" in loss_dict):
                        libcore.write_tensor_image(str(pc_dir / "dod_err_heatmap_cur_next.jpg"), loss_dict["dod_err_vis_cur_next"], rgb2bgr=True)
                except Exception as e:
                    print(f"Warning: failed to save loss debug maps: {e}")
                print(f"Saved debug PLY/images to {pc_dir}")

    ##################################################
    print("\nTraining finished.")
    if network_gui and network_gui.conn is not None:
        print("Holding GUI connection alive. Press Ctrl+C to exit.")
        while network_gui.conn is not None:
            try:
                gs_model.update_to_cano_mesh()
                if 'viewpoint_cam' in locals():
                     network_gui.render_to_network(gs_model, pipe, args.dat_dir, viewpoint_cam_override=viewpoint_cam)
                else: 
                     print("No viewpoint_cam available for final GUI render. GUI might not update.")
                torch.cuda.synchronize()
            except Exception as e:
                print(f"GUI render loop error: {e}")
                if network_gui.conn is not None: network_gui.conn.close()
                network_gui.conn = None
    print('[done]')
