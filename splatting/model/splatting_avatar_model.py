# SplattingAvatar Model.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import torch
import torch.nn.functional as thf
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from model import libcore
from simple_phongsurf import PhongSurfacePy3d
from utils.sh_utils import eval_sh, RGB2SH
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from utils.data_utils import sample_bary_on_triangles, retrieve_verts_barycentric
from utils.map import PerVertQuaternion
from utils.graphics_utils import BasicPointCloud
from utils.frames import build_ntb_frame_from_tri_and_smoothN, apply_ntb_delta
from utils.pose_features import get_smplx_pose_63
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_multiply, axis_angle_to_matrix
from pytorch3d.structures import Meshes
from .gauss_base import GaussianBase, to_abs_path, to_cache_path
from .deformation_networks import DeformationModule
from .litept_gs_fields import interp_token_field_to_gs
from .lbs_weight_residual import LbsWeightResidual
from .smplx_utils import smplx_utils
from .smplx_utils.smplx.joint_names import JOINT_NAMES

# standard 3dgs
class SplattingAvatarModel(GaussianBase):
    def __init__(self, config,
                 device=torch.device('cuda'),
                 verbose=False,
                 num_training_frames=None,
                 ref_frame_smpl_params_for_lbs=None,
                 lbs_weights_for_ref_mesh=None,
                 gaussians_are_frozen=False,
                 static_rendering=False):
        super().__init__(sh_degree=config.get('sh_degree', 0))
        self.config = config
        self.device = device
        self.verbose = verbose
        self.num_training_frames = num_training_frames

        self.use_deformation = self.config.get('use_deformation', False)
        self.deformation_module = None
        self.garment_mask_from_labels = None
        self.n_initial_gaussians = 0
        self.cano_verts_orig_for_deform = None
        self.cano_verts_used_for_lbs_basis = None
        self.cano_norms = None
        self.cano_faces = None

        self._current_delta_g_rgb_reshaped = None
        # Tao-style BS position residual: δu_local in per-Gaussian (N,T,B) frame.
        # IMPORTANT: applied ONLY via NTB→world conversion (do not also send through base_quat path).
        self._current_delta_u_local_ntb = None
        # LitePT Task5 GS field residuals (applied in getters, LitePT mode only).
        self._current_delta_alpha_logit = None   # (Ng,1) or None
        self._current_delta_logscale = None      # (Ng,3) or None
        self._last_delta_alpha_gs = None
        self._last_delta_logscale_gs = None
        # For logging/diagnostics
        self._last_vertex_offset_l2_mean = None
        self._last_delta_u_l2_mean = None
        # Stage-B offset smoothing (P1): store current vertex offsets for joint-aware regularizers.
        self._current_vertex_offsets = None  # (Nv,3) or None

        self.gaussians_are_frozen = gaussians_are_frozen
        self.static_rendering = static_rendering
        self.pretrained_gs_path = self.config.get('load_pretrained_gs_from', None)

        # Cloth-fit restoration safety: scale outlier handling (clamp or hide).
        # Config is typically set by repose_avatar CLI args during cloth-fit restoration.
        self._cloth_fit_gs_scale_safety_cfg = None
        self._cloth_fit_gs_scale_safety_cache = None

        self.max_sh_degree = self.config.get('sh_degree', 0)
        self.active_sh_degree = self.max_sh_degree

        self.smpl_model_for_lbs = None
        self.inv_ref_pose_jnt_mats = None
        self.lbs_weights = None
        self.lbs_weight_residual: LbsWeightResidual | None = None
        self._lbs_joint_pos_cano = None  # (J,3) canonical joint centers for union support
        self.global_step = None          # training iteration; used by LBS alpha schedule
        self.smpl_global_transform_mode = self.config.get('smpl_global_transform_mode', 'internal')

        # Stage-B vertex-MLP input enhancement (tao_offset_input_enhancement.md)
        # Cache is built in the same canonical/world space as `cano_verts_orig_for_deform`.
        self._ref_frame_smpl_params_for_lbs = ref_frame_smpl_params_for_lbs
        self._smpl_anchor_cache = None
        self._smpl_anchor_cache_sig = None

        # Clothfit test hack (tao_clothfit_test.md): query Stage-B MLP on ref-before xyz.
        self.xyz_query_ref_before_v3 = None   # (Nv,3) float32 on device
        self.xyz_query_delta_shape_v3 = None  # (Nv,3) float32 on device (after - before)
        self._xyz_query_override_sig = None

        # Stage-B offset smoothing (P1): cached joint mask + mesh edges/adjacency.
        self._joint_reg_mask_v = None          # (Nv,)
        self._joint_reg_mask_cfg_sig = None   # cache key for mask config
        self._joint_reg_edges_e2 = None        # (Ne,2) int64
        self._joint_reg_e0 = None              # (Ne,) float
        self._joint_reg_lap_src = None         # (2Ne,) int64
        self._joint_reg_lap_dst = None         # (2Ne,) int64
        self._joint_reg_lap_deg = None         # (Nv,) float

        if self.use_deformation:
            if ref_frame_smpl_params_for_lbs is None or lbs_weights_for_ref_mesh is None:
                raise ValueError("Reference SMPL parameters and LBS weights must be provided for deformation mode.")
            
            self.lbs_weights = lbs_weights_for_ref_mesh.to(self.device)

            # TODO-4: Global learnable LBS weight residual (cloth-only).
            # Create the module so checkpoints can load it; usage is gated by config.
            try:
                lbs_cfg = self.config.get("lbs_weight_residual", {}) if hasattr(self.config, "get") else {}
            except Exception:
                lbs_cfg = {}
            topk = int(getattr(lbs_cfg, "topk", lbs_cfg.get("topk", 4))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else 4
            alpha = float(getattr(lbs_cfg, "alpha", lbs_cfg.get("alpha", 0.1))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else 0.1
            clamp_logit = float(getattr(lbs_cfg, "clamp_logit", lbs_cfg.get("clamp_logit", 1.0))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else 1.0
            eps = float(getattr(lbs_cfg, "eps", lbs_cfg.get("eps", 1e-8))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else 1e-8

            # v1.1: union support + alpha schedule (all optional; defaults preserve v1 behavior)
            support_mode = str(getattr(lbs_cfg, "support_mode", lbs_cfg.get("support_mode", "topk_only"))).lower() if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else "topk_only"
            topk_w0 = getattr(lbs_cfg, "topk_w0", lbs_cfg.get("topk_w0", None)) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else None
            topk_joint_dist = int(getattr(lbs_cfg, "topk_joint_dist", lbs_cfg.get("topk_joint_dist", 0))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else 0
            k_total = getattr(lbs_cfg, "k_total", lbs_cfg.get("k_total", None)) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else None
            tanh_on_delta = bool(getattr(lbs_cfg, "tanh_on_delta", lbs_cfg.get("tanh_on_delta", True))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else True
            alpha_schedule = getattr(lbs_cfg, "alpha_schedule", lbs_cfg.get("alpha_schedule", None)) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else None

            # forced joints: accept names or integer ids
            forced_joint_ids = []
            try:
                forced = getattr(lbs_cfg, "forced_joints", lbs_cfg.get("forced_joints", [])) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else []
                if forced is None:
                    forced = []
                # build name->id map for the first J joints (matches LBS weight channels)
                name_to_id = {str(n): int(i) for i, n in enumerate(list(JOINT_NAMES)[: int(self.lbs_weights.shape[1])])}
                for x in list(forced) if isinstance(forced, (list, tuple)) else [forced]:
                    if isinstance(x, (int, np.integer)):
                        forced_joint_ids.append(int(x))
                    else:
                        jn = str(x)
                        if jn in name_to_id:
                            forced_joint_ids.append(int(name_to_id[jn]))
            except Exception:
                forced_joint_ids = []
            try:
                self.lbs_weight_residual = LbsWeightResidual(
                    w0_full_vj=self.lbs_weights,
                    topk=topk,
                    support_mode=support_mode,
                    topk_w0=topk_w0,
                    topk_joint_dist=topk_joint_dist,
                    forced_joint_ids=forced_joint_ids,
                    k_total=k_total,
                    tanh_on_delta=tanh_on_delta,
                    alpha=alpha,
                    alpha_schedule=alpha_schedule,
                    clamp_logit=clamp_logit,
                    eps=eps,
                    device=self.device,
                    dtype=self.lbs_weights.dtype,
                ).to(self.device)
            except Exception as e:
                print(f"[SplattingAvatarModel] Warning: failed to init LBS weight residual module: {e}")
                self.lbs_weight_residual = None

            smpl_config_lbs = {
                'model_type': self.config.get('smpl_model_type', 'smplx'),
                'gender': self.config.get('smpl_gender', 'neutral'),
                'model_path': self.config.get('smpl_model_path'),
                'use_pca': self.config.get('smpl_use_pca', False),
                'num_pca_comps': self.config.get('smpl_num_pca_comps', 6),
                'flat_hand_mean': self.config.get('smpl_flat_hand_mean', False),
                'num_betas': ref_frame_smpl_params_for_lbs.get('betas', torch.zeros(10)).shape[-1],
                'batch_size': 1,
                'create_transl': False, 'create_global_orient': False, 'create_body_pose': False, 'create_betas': False
            }
            # TalkBody4D (and some other datasets) store higher-dim expression coefficients (e.g., 50D).
            # If we don't pass `num_expression_coeffs` here, SMPL-X defaults may be smaller (often 10),
            # which forces us to truncate at runtime and can change the produced vertices.
            try:
                expr0 = ref_frame_smpl_params_for_lbs.get("expression", None)
                if isinstance(expr0, torch.Tensor):
                    smpl_config_lbs["num_expression_coeffs"] = int(expr0.shape[-1])
            except Exception:
                pass
            # Make SMPL-X gender selection deterministic by resolving the exact model file when
            # config.model.smpl_model_path is not provided. This avoids any unexpected fallback
            # behavior from directory-based resolution and guarantees we load the intended gender.
            try:
                mt = str(smpl_config_lbs.get("model_type", "smplx")).strip().lower()
                gd = str(smpl_config_lbs.get("gender", "neutral")).strip().lower()
                mp = smpl_config_lbs.get("model_path", None)
                if mt == "smplx" and (mp is None or str(mp).strip() == ""):
                    model_root = smplx_utils.get_smplx_model_path()
                    cand = os.path.join(model_root, "smplx", f"SMPLX_{gd.upper()}.npz")
                    if os.path.exists(cand):
                        smpl_config_lbs["model_path"] = cand
            except Exception:
                pass
            print(f"[SplattingAvatarModel] smpl_config_lbs: {smpl_config_lbs}")
            for key_suffix in ['jaw_pose', 'leye_pose', 'reye_pose', 'expression', 'left_hand_pose', 'right_hand_pose']:
                 if key_suffix in ref_frame_smpl_params_for_lbs: smpl_config_lbs[f'create_{key_suffix}'] = False

            # TalkBody4D provides per-vertex offsets (v_shape/v_pose) that require the smplx_ani implementation.
            # Auto-enable it when these keys are present; otherwise allow overriding via config.
            try:
                if ("v_shape" in ref_frame_smpl_params_for_lbs) or ("v_pose" in ref_frame_smpl_params_for_lbs):
                    smpl_config_lbs["implementation"] = "smplx_ani"
                    print(f"[SplattingAvatarModel] using smplx_ani implementation")
                else:
                    smpl_impl_cfg = self.config.get("smpl_implementation", None)
                    if smpl_impl_cfg is not None:
                        smpl_config_lbs["implementation"] = str(smpl_impl_cfg)
                        print(f"[SplattingAvatarModel] using {smpl_impl_cfg} implementation")
            except Exception:
                pass

            # TalkBody4D (and any dataset that supplies full hand axis-angle): disable PCA.
            # If left/right hand pose has 45 dims (15 joints * 3 axis-angle), PCA=6 will crash in SMPL-X.
            try:
                lh0 = ref_frame_smpl_params_for_lbs.get("left_hand_pose", None)
                rh0 = ref_frame_smpl_params_for_lbs.get("right_hand_pose", None)
                lh_dim = int(lh0.shape[-1]) if isinstance(lh0, torch.Tensor) and lh0.ndim >= 1 else None
                rh_dim = int(rh0.shape[-1]) if isinstance(rh0, torch.Tensor) and rh0.ndim >= 1 else None
                if lh_dim == 45 or rh_dim == 45:
                    smpl_config_lbs["use_pca"] = False
                    smpl_config_lbs["num_pca_comps"] = 45
                    smpl_config_lbs["flat_hand_mean"] = True
            except Exception:
                pass
            
            self.smpl_model_for_lbs = smplx_utils.create_smplx_model(**smpl_config_lbs).to(self.device)

            with torch.no_grad():
                ref_jnt_mats = self._smplx_joint_mats_world_for_skinning(ref_frame_smpl_params_for_lbs)
                self.inv_ref_pose_jnt_mats = torch.linalg.inv(ref_jnt_mats).detach()
                try:
                    self._lbs_joint_pos_cano = ref_jnt_mats[0, :, :3, 3].detach()
                except Exception:
                    self._lbs_joint_pos_cano = None
                if self.verbose:
                    print(f"[SplattingAvatarModel] Initialized SMPL model for LBS and computed inv_ref_pose_jnt_mats.")

    def _batch_smpl_params(self, params: dict) -> dict:
        """Ensure SMPLX parameter dict tensors are batched (B=1) and on device."""
        out = {}
        for k, v in params.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v, dtype=torch.float32, device=self.device)
            else:
                v = v.to(self.device)
            if v.ndim == 1:
                v = v.unsqueeze(0)
            out[k] = v
        return out

    def _extract_global_sim_transform(self, p: dict):
        """Extract (go, tr, scale) tensors (B,3), (B,3), (B,1) if present."""
        go = p.get('global_orient', None)
        tr = p.get('transl', None)
        sc = p.get('scale', None)
        # Normalize shapes
        if sc is not None:
            if sc.ndim == 1:
                sc = sc.unsqueeze(0)
            if sc.ndim == 2 and sc.shape[-1] != 1:
                # Some datasets store scale as (B,) - handle via unsqueeze
                sc = sc.reshape(sc.shape[0], 1)
        return go, tr, sc

    def _smplx_joint_mats_world_for_skinning(self, params: dict) -> torch.Tensor:
        """Return per-joint transform mats (B,J,4,4) in the SAME world space as SMPL vertices/joints.

        This matches `swapvton/.../process_meshes.py` logic:
        - For our default SMPLX implementation (`implementation='smplx'`): the output `T` does NOT include
          `scale` / `transl`, so we must fold them into the joint transforms we use for LBS.
        - For `implementation='smplx_ani'`: the output affine `A` includes `transl` (but not scale),
          so we only fold `scale` into the translation component (do NOT add `transl` again).
        - For ActorsHQ external mode: run SMPLX with global_orient=0, transl=0 (keep scale),
          then apply the external rigid transform (global_orient/transl) to the joint transforms.
        - For TalkBody4D: apply an additional post-forward rigid transform Rh/Th:
          X_world = R(Rh) * X + Th  (see `swapvton/src/scripts/mesh/process_meshes.py:_apply_post_forward_rigid_transform`)
        """
        p = self._batch_smpl_params(params)
        go, tr, sc = self._extract_global_sim_transform(p)

        # TalkBody4D post-forward rigid (Rh/Th). Keep out of forward kwargs.
        Rh = None
        Th = None
        try:
            Rh = p.get("Rh", p.get("rh", None))
            Th = p.get("Th", p.get("th", None))
            for kk in ("Rh", "Th", "rh", "th"):
                if kk in p:
                    p.pop(kk, None)
        except Exception:
            Rh, Th = None, None

        # Make SMPL-X robust to datasets with higher-dim betas / expression than the model supports.
        # (Same idea as AvatarRexDataset.load_smpl_params()).
        try:
            shapedirs = getattr(self.smpl_model_for_lbs, "shapedirs", None)
            shapedirs_l = int(shapedirs.shape[-1]) if isinstance(shapedirs, torch.Tensor) else None
            if shapedirs_l is not None and isinstance(p.get("betas", None), torch.Tensor):
                b = p["betas"]
                if int(b.shape[-1]) > int(shapedirs_l):
                    p["betas"] = b[..., : int(shapedirs_l)]

            expr_dirs = getattr(self.smpl_model_for_lbs, "expr_dirs", None)
            expr_l = int(expr_dirs.shape[-1]) if isinstance(expr_dirs, torch.Tensor) else None
            if expr_l is None:
                expr_l = int(getattr(self.smpl_model_for_lbs, "num_expression_coeffs", 0) or 0)
            if expr_l is not None and expr_l > 0 and isinstance(p.get("expression", None), torch.Tensor):
                e = p["expression"]
                if int(e.shape[-1]) > int(expr_l):
                    p["expression"] = e[..., : int(expr_l)]
        except Exception:
            pass

        if self.smpl_global_transform_mode == 'external' and (go is not None) and (tr is not None):
            # Canonical forward (keep scale), externalize global_orient/transl.
            p0 = dict(p)
            p0['global_orient'] = torch.zeros_like(go)
            p0['transl'] = torch.zeros_like(tr)
            out = self.smpl_model_for_lbs(**p0, return_verts=False, return_full_pose=False)
            T = out.A if (hasattr(out, "A") and isinstance(getattr(out, "A"), torch.Tensor) and out.A is not None) else out.T
            used_affine_A = bool(hasattr(out, "A") and isinstance(getattr(out, "A"), torch.Tensor) and out.A is not None)

            # Fold scale into translation component (SMPL forward scales verts/joints after LBS).
            if sc is not None:
                T = T.clone()
                T[:, :, :3, 3] = T[:, :, :3, 3] * sc.unsqueeze(1)

            # Apply external rigid transform: X_world = R * X_canon + tr
            Rg = axis_angle_to_matrix(go.to(T.device, T.dtype))  # (B,3,3)
            G = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
            G[:, :3, :3] = Rg
            G[:, :3, 3] = tr.to(T.device, T.dtype)
            T = torch.matmul(G.unsqueeze(1), T)
        else:
            # Internal mode: use SMPL's global_orient in T/A (already), then fold scale (+ transl for T-only) to match verts.
            out = self.smpl_model_for_lbs(**p, return_verts=False, return_full_pose=False)
            T = out.A if (hasattr(out, "A") and isinstance(getattr(out, "A"), torch.Tensor) and out.A is not None) else out.T
            used_affine_A = bool(hasattr(out, "A") and isinstance(getattr(out, "A"), torch.Tensor) and out.A is not None)
            if sc is not None:
                T = T.clone()
                T[:, :, :3, 3] = T[:, :, :3, 3] * sc.unsqueeze(1)
            # Only add transl if we are using the T-only implementation (smplx). For smplx_ani, A already includes transl.
            if (not used_affine_A) and (tr is not None):
                # SMPL forward adds transl after scaling.
                if tr.ndim == 1:
                    tr_b = tr.unsqueeze(0)
                else:
                    tr_b = tr
                T = T.clone()
                T[:, :, :3, 3] = T[:, :, :3, 3] + tr_b.to(T.device, T.dtype).unsqueeze(1)

        # TalkBody4D: post-forward rigid transform using Rh/Th (do NOT regenerate canonical).
        try:
            if isinstance(Rh, torch.Tensor) and isinstance(Th, torch.Tensor):
                Rh_b = Rh.unsqueeze(0) if Rh.ndim == 1 else Rh
                Th_b = Th.unsqueeze(0) if Th.ndim == 1 else Th
                Rh_b = Rh_b.to(T.device, T.dtype)
                Th_b = Th_b.to(T.device, T.dtype)
                R = axis_angle_to_matrix(Rh_b)  # (B,3,3)
                G = torch.eye(4, device=T.device, dtype=T.dtype).unsqueeze(0).repeat(T.shape[0], 1, 1)
                G[:, :3, :3] = R
                G[:, :3, 3] = Th_b
                T = torch.matmul(G.unsqueeze(1), T)
        except Exception:
            pass

        return T

    def _smplx_vertices_world_for_anchor(self, params: dict) -> torch.Tensor:
        """Return SMPL(-X) vertices (Ns,3) in the SAME world/canonical space as LBS.

        This mirrors the transform logic in `_smplx_joint_mats_world_for_skinning`, but returns vertices.
        Used for TAO's `smpl_anchor` Stage-B vertex-MLP input mode.
        """
        if self.smpl_model_for_lbs is None:
            raise RuntimeError("SMPL model not initialized.")

        p = self._batch_smpl_params(params)

        # Make SMPL-X robust to higher-dim betas/expression (same as `_smplx_joint_mats_world_for_skinning`).
        try:
            shapedirs = getattr(self.smpl_model_for_lbs, "shapedirs", None)
            shapedirs_l = int(shapedirs.shape[-1]) if isinstance(shapedirs, torch.Tensor) else None
            if shapedirs_l is not None and isinstance(p.get("betas", None), torch.Tensor):
                b = p["betas"]
                if int(b.shape[-1]) > int(shapedirs_l):
                    p["betas"] = b[..., : int(shapedirs_l)]

            expr_dirs = getattr(self.smpl_model_for_lbs, "expr_dirs", None)
            expr_l = int(expr_dirs.shape[-1]) if isinstance(expr_dirs, torch.Tensor) else None
            if expr_l is None:
                expr_l = int(getattr(self.smpl_model_for_lbs, "num_expression_coeffs", 0) or 0)
            if expr_l is not None and expr_l > 0 and isinstance(p.get("expression", None), torch.Tensor):
                e = p["expression"]
                if int(e.shape[-1]) > int(expr_l):
                    p["expression"] = e[..., : int(expr_l)]
        except Exception:
            pass
        go, tr, sc = self._extract_global_sim_transform(p)

        # TalkBody4D post-forward rigid (Rh/Th). Keep out of forward kwargs.
        Rh = None
        Th = None
        try:
            Rh = p.get("Rh", p.get("rh", None))
            Th = p.get("Th", p.get("th", None))
            for kk in ("Rh", "Th", "rh", "th"):
                if kk in p:
                    p.pop(kk, None)
        except Exception:
            Rh, Th = None, None

        # External mode: canonical forward, then apply external rigid (and keep scale).
        if self.smpl_global_transform_mode == "external" and (go is not None) and (tr is not None):
            p0 = dict(p)
            p0["global_orient"] = torch.zeros_like(go)
            p0["transl"] = torch.zeros_like(tr)
            out = self.smpl_model_for_lbs(**p0, return_full_pose=False)
            V = out.vertices  # includes scale if provided to SMPL forward
            Rg = axis_angle_to_matrix(go.to(V.device, V.dtype))  # (B,3,3)
            V = torch.bmm(V, Rg.transpose(1, 2)) + tr.to(V.device, V.dtype).unsqueeze(1)
        else:
            out = self.smpl_model_for_lbs(**p, return_full_pose=False)
            V = out.vertices
        # if getattr(self, "verbose", False):
        #     print(f"[SplattingAvatarModel] Using gender: {getattr(self.smpl_model_for_lbs, 'gender', 'unknown')}")

        # TalkBody4D: post-forward rigid transform using Rh/Th
        try:
            if isinstance(Rh, torch.Tensor) and isinstance(Th, torch.Tensor):
                Rh_b = Rh.unsqueeze(0) if Rh.ndim == 1 else Rh
                Th_b = Th.unsqueeze(0) if Th.ndim == 1 else Th
                Rh_b = Rh_b.to(V.device, V.dtype)
                Th_b = Th_b.to(V.device, V.dtype)
                R = axis_angle_to_matrix(Rh_b)  # (B,3,3)
                V = torch.bmm(V, R.transpose(1, 2)) + Th_b.unsqueeze(1)
        except Exception:
            pass

        return V.squeeze(0).contiguous()

    def _smplx_faces_tensor_for_anchor(self) -> torch.Tensor:
        """Return SMPL faces as (F,3) long tensor on device."""
        if self.smpl_model_for_lbs is None:
            raise RuntimeError("SMPL model not initialized.")
        F = getattr(self.smpl_model_for_lbs, "faces", None)
        if F is None:
            raise RuntimeError("SMPL model has no `faces`.")
        if isinstance(F, torch.Tensor):
            Ft = F.to(device=self.device, dtype=torch.long)
        else:
            Ft = torch.as_tensor(np.asarray(F), device=self.device, dtype=torch.long)
        if Ft.ndim != 2 or Ft.shape[-1] != 3:
            raise ValueError(f"Unexpected SMPL faces shape: {tuple(Ft.shape)}")
        return Ft.contiguous()

    def _smpl_params_pose_only_for_token_features(self, params: dict) -> dict:
        """Return a params dict with global rigid terms zeroed (pose-only local motion)."""
        p = {}
        for k, v in params.items():
            p[k] = v.detach().clone() if isinstance(v, torch.Tensor) else v
        for k in ("global_orient", "transl", "Rh", "Th", "rh", "th"):
            vv = p.get(k, None)
            if isinstance(vv, torch.Tensor):
                p[k] = torch.zeros_like(vv)
        return p

    def _load_canonical_verts_from_embedding_dir(self, pretrained_gs_from: Path) -> torch.Tensor:
        """Load canonical mesh verts from `pretrained_gs_from/embedding.json` (supports cano_mesh/mesh_fn)."""
        pretrained_gs_from = Path(pretrained_gs_from)
        embed_json = pretrained_gs_from / "embedding.json"
        if not embed_json.exists():
            raise FileNotFoundError(f"embedding.json not found under: {embed_json}")
        with open(embed_json, "r") as f:
            cc = json.load(f)
        rel = cc.get("cano_mesh", cc.get("mesh_fn"))
        if not rel:
            raise ValueError(f"embedding.json missing cano_mesh/mesh_fn: {embed_json}")
        mesh_path = embed_json.parent / str(rel)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Canonical mesh referenced by embedding.json not found: {mesh_path}")
        mesh_cpu = libcore.MeshCpu(str(mesh_path))
        V = torch.tensor(mesh_cpu.V).float().to(self.device)
        return V.contiguous()

    def _resolve_ref_before_pretrained_dir(self) -> Path:
        """Auto-resolve the ref-before pretrained dir for xyz_query_override."""
        if not self.pretrained_gs_path:
            raise ValueError("xyz_query_override requires config.model.load_pretrained_gs_from (pretrained_gs_path).")
        p_after = Path(str(self.pretrained_gs_path))
        # Heuristic: test dirs are named like `test_tao_on_clothfit`; use sibling iteration_10000.
        if p_after.name.startswith("test_"):
            cand = p_after.parent / "iteration_10000"
            return cand
        raise ValueError(
            f"Cannot auto-resolve ref-before dir from load_pretrained_gs_from={p_after}. "
            f"Set deformation.xyz_query_override.ref_before_pretrained_gs_from explicitly or use a test_* dir name."
        )

    def maybe_init_xyz_query_override_buffers(self, *, force: bool = False) -> None:
        """Initialize ref-before xyz query buffers (V_before and delta_shape) if enabled."""
        if not self.use_deformation:
            return
        deform_cfg = self.config.get("deformation", {}) if (isinstance(self.config, dict) or hasattr(self.config, "get")) else {}
        input_mode = str(getattr(deform_cfg, "input_mode", deform_cfg.get("input_mode", "xyz"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "xyz"
        # Only meaningful for xyz mode; smpl_anchor already avoids xyz OOD.
        qcfg = getattr(deform_cfg, "xyz_query_override", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("xyz_query_override", None)
        if qcfg is None:
            return
        enabled = bool(getattr(qcfg, "enabled", qcfg.get("enabled", False))) if (isinstance(qcfg, dict) or hasattr(qcfg, "get")) else False
        if not enabled:
            return
        if input_mode != "xyz":
            if self.verbose:
                print(f"[SplattingAvatarModel] xyz_query_override enabled but input_mode={input_mode}; ignoring (xyz-only).")
            return
        if self.cano_verts_orig_for_deform is None:
            raise RuntimeError("xyz_query_override requires canonical verts (cano_verts_orig_for_deform) to be initialized.")

        ref_before_override = ""
        try:
            ref_before_override = str(getattr(qcfg, "ref_before_pretrained_gs_from", qcfg.get("ref_before_pretrained_gs_from", ""))).strip()
        except Exception:
            ref_before_override = ""
        ref_before_dir = Path(ref_before_override) if ref_before_override else self._resolve_ref_before_pretrained_dir()
        sig = (int(self.cano_verts_orig_for_deform.shape[0]), str(ref_before_dir))
        if (not force) and (self._xyz_query_override_sig == sig) and (self.xyz_query_ref_before_v3 is not None) and (self.xyz_query_delta_shape_v3 is not None):
            return

        if not ref_before_dir.exists():
            raise FileNotFoundError(f"xyz_query_override ref_before_pretrained_gs_from does not exist: {ref_before_dir}")

        V_before = self._load_canonical_verts_from_embedding_dir(ref_before_dir)
        V_after = self.cano_verts_orig_for_deform.detach()
        if int(V_before.shape[0]) != int(V_after.shape[0]):
            raise ValueError(f"xyz_query_override Nv mismatch: before={tuple(V_before.shape)} after={tuple(V_after.shape)}")

        delta = (V_after.to(torch.float32) - V_before.to(torch.float32)).contiguous()
        self.xyz_query_ref_before_v3 = V_before.to(torch.float32).contiguous()
        self.xyz_query_delta_shape_v3 = delta
        self._xyz_query_override_sig = sig

        # Optional one-time stats
        try:
            log_stats = bool(getattr(qcfg, "log_stats", qcfg.get("log_stats", True))) if (isinstance(qcfg, dict) or hasattr(qcfg, "get")) else True
        except Exception:
            log_stats = True
        if log_stats and self.verbose:
            dn = delta.norm(dim=-1)
            print(f"[SplattingAvatarModel] xyz_query_override init: ref_before={ref_before_dir} "
                  f"delta_shape mean={float(dn.mean().item()):.4g} max={float(dn.max().item()):.4g}")

        # Expose for optional TB logging
        try:
            self._xyz_query_delta_shape_l2_mean = float(delta.norm(dim=-1).mean().item())
            self._xyz_query_delta_shape_l2_max = float(delta.norm(dim=-1).max().item())
        except Exception:
            self._xyz_query_delta_shape_l2_mean = None
            self._xyz_query_delta_shape_l2_max = None

    def _maybe_build_smpl_anchor_cache(self, *, force: bool = False) -> None:
        """Build or reuse the cached cloth->SMPL projection for `smpl_anchor` input mode."""
        if not self.use_deformation:
            return
        deform_cfg = self.config.get("deformation", {}) if (isinstance(self.config, dict) or hasattr(self.config, "get")) else {}
        input_mode = str(getattr(deform_cfg, "input_mode", deform_cfg.get("input_mode", "xyz"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "xyz"
        backend = str(getattr(deform_cfg, "backend", deform_cfg.get("backend", "mlp"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "mlp"
        if input_mode != "smpl_anchor" and backend != "litept":
            return
        if self.cano_verts_orig_for_deform is None:
            raise RuntimeError("Canonical cloth mesh not initialized (cano_verts_orig_for_deform is None).")
        if self._ref_frame_smpl_params_for_lbs is None:
            raise RuntimeError("Missing reference SMPL params for anchor cache.")

        # Cache signature: (Nv, Ns, betas hash-ish)
        Nv = int(self.cano_verts_orig_for_deform.shape[0])
        try:
            vt = getattr(self.smpl_model_for_lbs, "v_template", None)
            Ns = int(vt.shape[-2]) if isinstance(vt, torch.Tensor) else -1
        except Exception:
            Ns = -1
        beta_sig = None
        try:
            b = self._ref_frame_smpl_params_for_lbs.get("betas", None)
            if isinstance(b, torch.Tensor):
                bb = b.detach().float().reshape(-1)
                beta_sig = (int(bb.numel()), float(bb.mean().item()), float(bb.std().item()))
        except Exception:
            beta_sig = None
        sig = (Nv, Ns, beta_sig)
        if (not force) and (self._smpl_anchor_cache is not None) and (self._smpl_anchor_cache_sig == sig):
            return

        from model.reshaping.reshape_utils import nearest_face_pytorch3d

        # Inputs
        V_cloth = self.cano_verts_orig_for_deform.detach()
        V_smpl = self._smplx_vertices_world_for_anchor(self._ref_frame_smpl_params_for_lbs).detach()
        F_smpl = self._smplx_faces_tensor_for_anchor().detach()

        # Projection: for each cloth vertex, find nearest SMPL face and barycentric coords.
        scale_factor = float(getattr(deform_cfg, "projection_scale_factor", deform_cfg.get("projection_scale_factor", 1.0))) if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else 1.0
        chunk_size = int(getattr(deform_cfg, "projection_chunk_size", deform_cfg.get("projection_chunk_size", 20000))) if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else 20000

        face_idx = torch.empty((Nv,), device=self.device, dtype=torch.long)
        bary = torch.empty((Nv, 3), device=self.device, dtype=torch.float32)
        V_smpl_b = V_smpl.unsqueeze(0)
        # NOTE: reshape_ops-backed nearest-face projection can be sensitive to varying `N` across calls.
        # To be robust, we keep a constant call batch size (chunk_size) by padding the final chunk.
        step = max(1, chunk_size)
        use_fixed_n = (Nv > step)
        for start in range(0, Nv, step):
            end = min(Nv, start + step)
            cur_n = int(end - start)
            pts_raw = V_cloth[start:end].to(device=self.device, dtype=torch.float32)
            if use_fixed_n and cur_n < step:
                # Pad to (step,3) by repeating the last point (arbitrary; sliced away after projection).
                pad_n = int(step - cur_n)
                pts_pad = torch.cat([pts_raw, pts_raw[-1:].expand(pad_n, -1)], dim=0)
                pts = pts_pad.unsqueeze(0)
            else:
                pts = pts_raw.unsqueeze(0)
            with torch.no_grad():
                _d, idx, bc = nearest_face_pytorch3d(
                    pts,
                    V_smpl_b.to(dtype=torch.float32),
                    F_smpl,
                    scale_factor=scale_factor,
                )
            idx0 = idx[0][:cur_n].to(torch.long)
            bc0 = bc[0][:cur_n].to(torch.float32)
            face_idx[start:end] = idx0
            bary[start:end] = bc0

        tri_vidx = F_smpl[face_idx]  # (Nv,3)

        # SMPL normals in canonical/world space
        with torch.no_grad():
            smpl_mesh = Meshes(verts=[V_smpl.to(torch.float32)], faces=[F_smpl])
            N_smpl = smpl_mesh.verts_normals_packed()  # (Ns,3)

        Vtri = V_smpl[tri_vidx]  # (Nv,3,3)
        Ntri = N_smpl[tri_vidx]  # (Nv,3,3)
        q = (bary.unsqueeze(-1) * Vtri.to(torch.float32)).sum(dim=1)  # (Nv,3)
        Nq = thf.normalize((bary.unsqueeze(-1) * Ntri.to(torch.float32)).sum(dim=1), dim=-1)  # (Nv,3)

        # Build NTB frame at q (columns: N,T,B)
        e0 = (Vtri[:, 1, :] - Vtri[:, 0, :]).to(torch.float32)
        e1 = (Vtri[:, 2, :] - Vtri[:, 0, :]).to(torch.float32)
        T0 = e0 - (e0 * Nq).sum(dim=-1, keepdim=True) * Nq
        T1 = e1 - (e1 * Nq).sum(dim=-1, keepdim=True) * Nq
        use_t1 = (T0.norm(dim=-1) < 1e-8)
        T = torch.where(use_t1.unsqueeze(-1), T1, T0)
        T = thf.normalize(T, dim=-1)
        B = thf.normalize(torch.cross(Nq, T, dim=-1), dim=-1)
        R_ntb = torch.stack([Nq, T, B], dim=-1)  # (Nv,3,3)

        # r_local = R^T (p - q)
        p_minus_q = (V_cloth.to(torch.float32) - q).unsqueeze(-1)  # (Nv,3,1)
        r_local = torch.bmm(R_ntb.transpose(1, 2), p_minus_q).squeeze(-1)  # (Nv,3)

        # Convention sanity check for token-feature local-frame usage:
        # expected: x ~= q + R_ntb @ r_local with R_ntb columns=[N,T,B].
        x_recon = q + torch.bmm(R_ntb, r_local.unsqueeze(-1)).squeeze(-1)
        rec_err = torch.norm(x_recon - V_cloth.to(torch.float32), dim=-1)
        rec_p95 = torch.quantile(rec_err, 0.95)
        if float(rec_p95.item()) > 1e-3:
            alt = q + torch.bmm(R_ntb.transpose(1, 2), r_local.unsqueeze(-1)).squeeze(-1)
            alt_err = torch.norm(alt - V_cloth.to(torch.float32), dim=-1)
            alt_p95 = torch.quantile(alt_err, 0.95)
            raise RuntimeError(
                "smpl_anchor R_ntb convention check failed. "
                f"p95 ||q+Rr-x||={float(rec_p95.item()):.6g}, "
                f"alt p95 ||q+R^Tr-x||={float(alt_p95.item()):.6g}."
            )

        # Store as (persistent) buffers so they travel with checkpoints if desired.
        def _set_buf(name: str, value: torch.Tensor):
            if hasattr(self, name) and isinstance(getattr(self, name), torch.Tensor):
                getattr(self, name).data = value
            else:
                self.register_buffer(name, value, persistent=True)

        _set_buf("smpl_anchor_face_idx", face_idx)
        _set_buf("smpl_anchor_tri_vidx", tri_vidx)
        _set_buf("smpl_anchor_bary", bary)
        _set_buf("smpl_anchor_q", q)
        _set_buf("smpl_anchor_R_ntb", R_ntb)
        _set_buf("smpl_anchor_r_local", r_local)

        self._smpl_anchor_cache = {
            "face_idx": self.smpl_anchor_face_idx,
            "tri_vidx": self.smpl_anchor_tri_vidx,
            "bary": self.smpl_anchor_bary,
            "q": self.smpl_anchor_q,
            "R_ntb": self.smpl_anchor_R_ntb,
            "r_local": self.smpl_anchor_r_local,
        }
        self._smpl_anchor_cache_sig = sig
        if self.verbose:
            try:
                rn = r_local.norm(dim=-1)
                print(f"[SplattingAvatarModel] Built smpl_anchor cache: Nv={Nv}, Ns={int(V_smpl.shape[0])}, "
                      f"r_norm mean={float(rn.mean().item()):.4g}, max={float(rn.max().item()):.4g}, "
                      f"recon p95={float(rec_p95.item()):.4g}")
            except Exception:
                print(f"[SplattingAvatarModel] Built smpl_anchor cache: Nv={Nv}, Ns={int(V_smpl.shape[0])}")

    @property
    def num_gauss(self):
        return self._xyz.shape[0]

    @property
    def get_xyz_cano(self):
        # Canonical-space Gaussian centers.
        # IMPORTANT: keep recursion-free (must not depend on posed-space computation).
        if self.config.xyz_as_uvd:
            # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
            xyz = self.base_normal_cano * self._xyz[..., -1:]
            return self.base_xyz_cano + xyz
        else:
            return self._xyz

    @property
    def get_xyz_cano_w_deform(self):
        if self.config.get('xyz_as_uvd', True) and not self.gaussians_are_frozen:
            base_xyz_cano = retrieve_verts_barycentric(self.cano_verts_used_for_lbs_basis, self.cano_faces,
                                                       self.sample_fidxs, self.sample_bary)
            base_normal_cano = retrieve_verts_barycentric(self.cano_norms, self.cano_faces,
                                                          self.sample_fidxs, self.sample_bary)
            base_normal_cano = thf.normalize(base_normal_cano, dim=-1)
            xyz_offset = base_normal_cano * self._xyz[..., -1:]
            return base_xyz_cano + xyz_offset
        else:
            return self._xyz


    @property
    def get_xyz(self):
        if self.static_rendering:
            return self._xyz  # Direct world coordinates for static rendering
        elif self.use_deformation:
            return self.get_xyz_w_deform
        elif self.config.xyz_as_uvd:
                # uv -> self.sample_bary -> self.base_normal --(d)--> xyz
                xyz = self.base_normal * self._xyz[..., -1:]
                return self.base_xyz + xyz
        else:
            return self.get_xyz_cano

    def get_lbs_weights_for_skinning(self) -> torch.Tensor | None:
        """Return LBS weights (V,J) used for cloth mesh skinning (base or corrected)."""
        if self.lbs_weights is None:
            return None
        # Usage gated by config flag; module may still exist for checkpoint loading.
        try:
            lbs_cfg = self.config.get("lbs_weight_residual", {}) if hasattr(self.config, "get") else {}
        except Exception:
            lbs_cfg = {}
        enabled = bool(getattr(lbs_cfg, "enabled", lbs_cfg.get("enabled", False))) if (isinstance(lbs_cfg, dict) or hasattr(lbs_cfg, "get")) else False
        if enabled and (self.lbs_weight_residual is not None):
            try:
                step = None
                try:
                    step = int(self.global_step) if self.global_step is not None else None
                except Exception:
                    step = None
                # If alpha_schedule has a step_offset (e.g. Stage2 start), keep *exact* base weights before that.
                # This avoids support-projection changing skinning/vis in Stage1 when LBS residual isn't learnable yet.
                try:
                    alpha_sched = getattr(lbs_cfg, "alpha_schedule", None) if not isinstance(lbs_cfg, dict) else lbs_cfg.get("alpha_schedule", None)
                    if alpha_sched is not None:
                        off = alpha_sched.get("step_offset", None) if hasattr(alpha_sched, "get") else getattr(alpha_sched, "step_offset", None)
                        off_i = int(off) if off is not None else None
                        if (off_i is not None) and (step is not None) and (step < off_i):
                            return self.lbs_weights
                except Exception:
                    pass
                return self.lbs_weight_residual.corrected_weights_vj(global_step=step)
            except Exception:
                return self.lbs_weights
        return self.lbs_weights

    @property
    def get_scaling(self):
        """Optional TODO-5 (P2-1): static base scale residual on top of frozen _scaling."""
        base = self.scaling_activation(self._scaling)
        try:
            cfg = self.config.get("deform_scale_base", {}) if hasattr(self.config, "get") else {}
        except Exception:
            cfg = {}
        enabled = bool(getattr(cfg, "enabled", cfg.get("enabled", False))) if (isinstance(cfg, dict) or hasattr(cfg, "get")) else False
        if (not enabled) or (not hasattr(self, "delta_log_s_base")) or (self.delta_log_s_base is None):
            return base
        try:
            clamp_log_s = float(getattr(cfg, "clamp_log_s", cfg.get("clamp_log_s", 0.3)))
        except Exception:
            clamp_log_s = 0.3
        d = self.delta_log_s_base.to(self._scaling.device, self._scaling.dtype).clamp(-clamp_log_s, clamp_log_s)
        log_s_new = self._scaling + d
        return torch.exp(log_s_new)

    @property
    def get_xyz_w_deform(self):
        # 1) Base posed/world center (existing SplattingAvatar-style path):
        # canonical offset vector is rotated by base_quat and added to the posed surface point.
        current_xyz_cano = self.get_xyz_cano
        vec_on_cano_tri = current_xyz_cano - self.base_xyz_cano
        lbs_rotations = build_rotation(self.base_quat)
        posed_vec_offset = torch.bmm(lbs_rotations, vec_on_cano_tri.unsqueeze(-1)).squeeze(-1)
        posed_xyz = self.base_xyz + posed_vec_offset

        # 2) Tao-style BS position residual δu:
        # δu is defined in per-Gaussian (N,T,B) where N is the smooth (bary-interpolated) normal.
        # Convert once to world and add once (do NOT also pass through base_quat).
        if self._current_delta_u_local_ntb is not None:
            if self.mesh_verts is None or self.cano_faces is None or self.sample_fidxs is None:
                raise RuntimeError("Missing mesh/binding data required to apply NTB δu.")

            tri_vidxs = self.cano_faces[self.sample_fidxs]  # (Ng, 3)
            tri_verts = self.mesh_verts[tri_vidxs]          # (Ng, 3, 3)
            v1, v2, v3 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]
            N = self.base_normal
            Nn, T, B = build_ntb_frame_from_tri_and_smoothN(v1, v2, v3, N)
            delta_u = self._current_delta_u_local_ntb.to(posed_xyz.device, posed_xyz.dtype)
            if delta_u.shape[0] != posed_xyz.shape[0]:
                raise RuntimeError(f"delta_u_local_ntb shape mismatch: {tuple(delta_u.shape)} vs posed_xyz {tuple(posed_xyz.shape)}")
            # TODO-2: optional soft clamp on ||delta_u|| to suppress spikes/outliers.
            deform_cfg = self.config.get("deformation", {}) if isinstance(self.config, dict) or hasattr(self.config, "get") else {}
            clamp_enabled = bool(getattr(deform_cfg, "delta_u_clamp_enabled", deform_cfg.get("delta_u_clamp_enabled", False))) if isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get") else False
            if clamp_enabled:
                tau = float(getattr(deform_cfg, "delta_u_clamp_tau", deform_cfg.get("delta_u_clamp_tau", 0.02)))
                eps = 1e-8
                norm = torch.sqrt((delta_u * delta_u).sum(dim=-1) + eps)  # [Ng]
                scale = (tau / norm).clamp(max=1.0)
                delta_u = delta_u * scale.unsqueeze(-1)

                # Stats for logging
                try:
                    self._last_delta_u_max = norm.max().detach()
                    # quantile can fail on some older builds; fallback to sort if needed
                    if hasattr(torch, "quantile"):
                        self._last_delta_u_p99 = torch.quantile(norm.detach(), 0.99)
                    else:
                        k = max(int(0.99 * float(norm.numel())) - 1, 0)
                        self._last_delta_u_p99 = norm.detach().flatten().sort().values[k]
                    self._last_delta_u_frac_clamped = (norm > tau).float().mean().detach()
                except Exception:
                    self._last_delta_u_max = None
                    self._last_delta_u_p99 = None
                    self._last_delta_u_frac_clamped = None

            posed_xyz = posed_xyz + apply_ntb_delta(Nn, T, B, delta_u)

        return posed_xyz

    @property
    def base_normal_cano(self):
        return thf.normalize(retrieve_verts_barycentric(self.cano_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)

    @property
    def base_normal(self):
        return thf.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, 
                                                        self.sample_fidxs, self.sample_bary), 
                                                        dim=-1)
    @property
    def base_xyz_cano(self):
        return retrieve_verts_barycentric(self.cano_verts_used_for_lbs_basis, self.cano_faces,
                                          self.sample_fidxs, self.sample_bary)
    @property
    def base_xyz(self):
        return retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, 
                                          self.sample_fidxs, self.sample_bary)

    @property
    def get_rotation_cano(self):
        return self.rotation_activation(self._rotation)
        
    @property
    def get_rotation(self):
        if self.static_rendering:
            return self.rotation_activation(self._rotation)  # Direct rotation from PLY
        else:
            return self.rotation_activation(quaternion_multiply(self.base_quat, self._rotation))
    
    @property
    def get_rotation_embed(self):
        return self.rotation_activation(self.base_quat)
    
    @property
    def base_quat(self):
        return torch.einsum('bij,bi->bj', self.tri_quats[self.sample_fidxs], self.sample_bary)
    
    @property
    def get_scaling_cano(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_scaling(self):
        # Base (log)scale (existing parameterization in this codebase).
        if self.static_rendering:
            log_s = self._scaling
        elif self.config.get('with_mesh_scaling', False) and hasattr(self, '_face_scaling') and self._face_scaling is not None:
            scaling_alter = self._face_scaling[self.sample_fidxs]
            log_s = self._scaling * scaling_alter
        else:
            log_s = self._scaling

        # LitePT Task5: apply GS logscale residual (log-space) in LitePT mode only.
        deform_backend = None
        try:
            deform_backend = str(getattr(self.deformation_module, "backend", "mlp")).lower() if self.deformation_module is not None else None
        except Exception:
            deform_backend = None
        if deform_backend == "litept" and isinstance(self._current_delta_logscale, torch.Tensor):
            d = self._current_delta_logscale.to(device=log_s.device, dtype=log_s.dtype)
            if d.shape == log_s.shape:
                log_s = log_s + d

        s = self.scaling_activation(log_s)

        cache = getattr(self, "_cloth_fit_gs_scale_safety_cache", None)
        if isinstance(cache, dict) and cache.get("enabled", False):
            s_safe = cache.get("scaling_safe", None)
            if isinstance(s_safe, torch.Tensor) and s_safe.shape == s.shape:
                return s_safe.to(s.device, s.dtype)
        return s

    @property
    def get_features(self):
        current_features_dc = self._features_dc.clone()

        if self._current_delta_g_rgb_reshaped is not None:
            delta_to_apply = self._current_delta_g_rgb_reshaped.to(current_features_dc.device, current_features_dc.dtype)
            if delta_to_apply.shape != current_features_dc.shape:
                raise RuntimeError(f"delta_g_rgb shape mismatch: {tuple(delta_to_apply.shape)} vs features_dc {tuple(current_features_dc.shape)}")
            current_features_dc = current_features_dc + delta_to_apply
        
        return torch.cat((current_features_dc, self._features_rest), dim=1)
    
    @property
    def get_opacity(self):
        deform_backend = None
        try:
            deform_backend = str(getattr(self.deformation_module, "backend", "mlp")).lower() if self.deformation_module is not None else None
        except Exception:
            deform_backend = None
        if deform_backend == "litept" and isinstance(self._current_delta_alpha_logit, torch.Tensor):
            d = self._current_delta_alpha_logit.to(device=self._opacity.device, dtype=self._opacity.dtype)
            if d.shape == self._opacity.shape:
                o = self.opacity_activation(self._opacity + d)
            else:
                o = self.opacity_activation(self._opacity)
        else:
            o = self.opacity_activation(self._opacity)
        cache = getattr(self, "_cloth_fit_gs_scale_safety_cache", None)
        if isinstance(cache, dict) and cache.get("enabled", False) and cache.get("action", "") == "zero_opacity":
            mask = cache.get("outlier_mask", None)
            if isinstance(mask, torch.Tensor) and mask.shape[0] == o.shape[0]:
                return o * (~mask).to(o.device, o.dtype).unsqueeze(-1)
        return o

    def set_cloth_fit_gs_scale_safety_cfg(
        self,
        *,
        enabled: bool,
        percentile: float = 0.995,
        hard_max_scale: float = 0.0,
        ratio_enabled: bool = False,
        ratio_percentile: float = 0.995,
        ratio_hard_max: float = 0.0,
        ratio_symmetric: bool = False,
        action: str = "clamp",
        eps_opacity: float = 1e-6,
    ) -> None:
        """Configure GS scale safety for cloth-fit restoration runs."""
        self._cloth_fit_gs_scale_safety_cfg = {
            "enabled": bool(enabled),
            "percentile": float(percentile),
            "hard_max_scale": float(hard_max_scale),
            "ratio_enabled": bool(ratio_enabled),
            "ratio_percentile": float(ratio_percentile),
            "ratio_hard_max": float(ratio_hard_max),
            "ratio_symmetric": bool(ratio_symmetric),
            "action": str(action).strip().lower(),
            "eps_opacity": float(eps_opacity),
        }
        self._cloth_fit_gs_scale_safety_cache = None
        self._cloth_fit_gs_scale_safety_prev_metric = None

    def _update_cloth_fit_gs_scale_safety_cache(self) -> None:
        cfg = getattr(self, "_cloth_fit_gs_scale_safety_cfg", None)
        if not isinstance(cfg, dict) or not cfg.get("enabled", False):
            self._cloth_fit_gs_scale_safety_cache = None
            return

        action = str(cfg.get("action", "clamp")).strip().lower()
        if action not in {"clamp", "zero_opacity"}:
            action = "clamp"
        p = float(cfg.get("percentile", 0.995))
        p = min(max(p, 0.0), 1.0)
        hard = float(cfg.get("hard_max_scale", 0.0))
        eps_op = float(cfg.get("eps_opacity", 1e-6))
        ratio_enabled = bool(cfg.get("ratio_enabled", False))
        rp = float(cfg.get("ratio_percentile", 0.995))
        rp = min(max(rp, 0.0), 1.0)
        rhard = float(cfg.get("ratio_hard_max", 0.0))
        rsym = bool(cfg.get("ratio_symmetric", False))

        # Compute current activated scales (Ng,3) using the same logic as get_scaling.
        if self.static_rendering:
            s = self.scaling_activation(self._scaling)
        elif self.config.get('with_mesh_scaling', False) and hasattr(self, '_face_scaling') and self._face_scaling is not None:
            scaling_alter = self._face_scaling[self.sample_fidxs]
            s = self.scaling_activation(self._scaling * scaling_alter)
        else:
            s = self.scaling_activation(self._scaling)

        if s.numel() == 0:
            self._cloth_fit_gs_scale_safety_cache = {
                "enabled": True,
                "action": action,
                "outlier_mask": torch.zeros((0,), dtype=torch.bool, device=s.device),
                "scaling_safe": s,
                "eps_opacity": eps_op,
                "threshold": 0.0,
                "frac_outliers": 0.0,
            }
            return

        metric = s.max(dim=1).values  # (Ng,)
        # Absolute scale outliers
        if hard > 0.0:
            thr = torch.tensor(float(hard), device=metric.device, dtype=metric.dtype)
        else:
            if hasattr(torch, "quantile"):
                thr = torch.quantile(metric.detach(), p)
            else:
                vals = metric.detach().flatten().sort().values
                k = int(p * float(vals.numel() - 1))
                thr = vals[max(0, min(k, vals.numel() - 1))]
        out_abs = metric > thr

        # Ratio outliers (current / previous cached metric)
        out_ratio = torch.zeros_like(out_abs)
        rthr_f = 0.0
        if ratio_enabled:
            prev = getattr(self, "_cloth_fit_gs_scale_safety_prev_metric", None)
            if isinstance(prev, torch.Tensor) and prev.shape == metric.shape:
                denom = torch.clamp(prev.to(metric.device, metric.dtype), min=1e-12)
                r = metric / denom
                if rsym:
                    r = torch.maximum(r, 1.0 / torch.clamp(r, min=1e-12))
                if rhard > 0.0:
                    rthr = torch.tensor(float(rhard), device=metric.device, dtype=metric.dtype)
                else:
                    if hasattr(torch, "quantile"):
                        rthr = torch.quantile(r.detach(), rp)
                    else:
                        vals = r.detach().flatten().sort().values
                        k = int(rp * float(vals.numel() - 1))
                        rthr = vals[max(0, min(k, vals.numel() - 1))]
                out_ratio = r > rthr
                rthr_f = float(rthr.detach().float().cpu().item())
            # Update prev metric for next comparison (including the common “baseline then reshaped” two-call pattern).
            self._cloth_fit_gs_scale_safety_prev_metric = metric.detach()
        else:
            self._cloth_fit_gs_scale_safety_prev_metric = metric.detach()

        outlier = out_abs | out_ratio
        frac = float(outlier.detach().float().mean().cpu().item())
        thr_f = float(thr.detach().float().cpu().item())
        out_n = int(outlier.detach().sum().cpu().item())
        total_n = int(outlier.numel())
        abs_n = int(out_abs.detach().sum().cpu().item())
        ratio_n = int(out_ratio.detach().sum().cpu().item()) if ratio_enabled else 0
        if self.verbose:
            if hard > 0.0:
                msg_abs = f"hard_max_scale={hard:g}"
            else:
                msg_abs = f"percentile={p:g} thr={thr_f:g}"
            if ratio_enabled:
                if rhard > 0.0:
                    msg_r = f"ratio_hard_max={rhard:g}"
                else:
                    msg_r = f"ratio_percentile={rp:g} ratio_thr={rthr_f:g}"
                msg_r += f" symmetric={bool(rsym)}"
            else:
                msg_r = "ratio=off"
            print(
                f"[cloth_fit_gs_scale_safety] action={action} {msg_abs} {msg_r} "
                f"outliers={out_n}/{total_n} ({frac*100.0:.3f}%) (abs={abs_n}, ratio={ratio_n})"
            )

        if action == "clamp":
            denom = torch.clamp(metric, min=1e-12)
            factor = (thr / denom).clamp(max=1.0)
            s_safe = s * factor.unsqueeze(-1)
        else:
            s_safe = s

        self._cloth_fit_gs_scale_safety_cache = {
            "enabled": True,
            "action": action,
            "outlier_mask": outlier.detach(),
            "scaling_safe": s_safe.detach(),
            "eps_opacity": eps_op,
            "threshold": thr_f,
            "frac_outliers": frac,
            "ratio_threshold": rthr_f,
        }

    def prepare_to_write(self, f_rest_dims=0):
        """
        Override base writer so cloth-fit scale safety can affect saved PLYs:
        - action=clamp: write log(scales_safe) to scale_* fields
        - action=zero_opacity: write near-zero opacity logits for outliers
        """
        contents = super().prepare_to_write(f_rest_dims=f_rest_dims)
        cache = getattr(self, "_cloth_fit_gs_scale_safety_cache", None)
        if not (isinstance(cache, dict) and cache.get("enabled", False)):
            return contents

        s_safe = cache.get("scaling_safe", None)
        if isinstance(s_safe, torch.Tensor) and s_safe.shape[0] == self._scaling.shape[0]:
            contents["scale"] = torch.log(torch.clamp(s_safe, min=1e-12)).detach().cpu().numpy()

        if cache.get("action", "") == "zero_opacity":
            mask = cache.get("outlier_mask", None)
            if isinstance(mask, torch.Tensor) and mask.shape[0] == self._opacity.shape[0]:
                low = inverse_sigmoid(torch.tensor([float(cache.get("eps_opacity", 1e-6))], device=self._opacity.device, dtype=self._opacity.dtype)).item()
                op = self._opacity.detach().clone()
                op[mask.to(op.device)] = float(low)
                contents["opacities"] = op.detach().cpu().numpy()

        return contents
    
    def get_params(self, device='cpu'):
        return {
            '_xyz': self._xyz.detach().to(device),
            '_rotation': self._rotation.detach().to(device),
            '_scaling': self._scaling.detach().to(device),
            '_features_dc': self._features_dc.detach().to(device),
            '_features_rest': self._features_rest.detach().to(device),
            '_opacity': self._opacity.detach().to(device),
        }
    
    def set_params(self, params):
        def _update_param(attr_name, value):
            current_val = getattr(self, attr_name)
            if isinstance(current_val, nn.Parameter):
                current_val.data = value.to(self.device)
            else:
                setattr(self, attr_name, value.to(self.device))
        
        if '_xyz' in params: _update_param('_xyz', params['_xyz'])
        if '_rotation' in params: _update_param('_rotation', params['_rotation'])
        if '_scaling' in params: _update_param('_scaling', params['_scaling'])
        if '_features_dc' in params: _update_param('_features_dc', params['_features_dc'])
        if '_features_rest' in params: _update_param('_features_rest', params['_features_rest'])
        if '_opacity' in params: _update_param('_opacity', params['_opacity'])
    
    def get_colors_precomp(self, viewpoint_camera=None):
        shs_view = self.get_features.transpose(1, 2).view(-1, 3, (self.max_sh_degree+1)**2)
        if viewpoint_camera is not None:
            dir_pp = (self.get_xyz - viewpoint_camera.camera_center.repeat(self.num_gauss, 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        else:
            dir_pp_normalized = torch.zeros_like(self._xyz)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        return colors_precomp

    def create_from_pcd(self, pcd : BasicPointCloud):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(self.device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(self.device))
        features_dc_init = fused_color.unsqueeze(2)
        features_rest_init = torch.zeros((fused_point_cloud.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1), dtype=torch.float, device=self.device)
        if self.verbose: print("Number of points at initialisation (create_from_pcd): ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud.float().cuda()), 0.0000001)
        scales_init = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots_init = torch.zeros((fused_point_cloud.shape[0], 4), device=self.device); rots_init[:, 0] = 1
        opacities_init = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=self.device))

        self.init_gauss(xyz=fused_point_cloud, 
                        features_dc=features_dc_init,
                        features_extra=features_rest_init,
                        opacities=opacities_init, 
                        scales=scales_init, 
                        rots=rots_init, 
                        init_params=False)
    
    def setup_canonical(self, cano_verts, cano_norms, cano_faces):
        self.cano_verts = cano_verts
        self.cano_norms = cano_norms
        self.cano_faces = cano_faces

        self.quat_helper = PerVertQuaternion(cano_verts, cano_faces).to(self.device)
        self.phongsurf = PhongSurfacePy3d(cano_verts, cano_faces, cano_norms,
                                          outer_loop=2, inner_loop=50, method='uvd').to(self.device)

    def _ensure_joint_reg_geometry_caches(self) -> None:
        """Precompute edge/adjacency caches for joint_reg losses (topology-dependent only)."""
        if self._joint_reg_edges_e2 is not None and self._joint_reg_e0 is not None and self._joint_reg_lap_src is not None:
            return
        if self.cano_faces is None or self.cano_verts_orig_for_deform is None:
            return
        F = self.cano_faces.to(self.device)
        V0 = self.cano_verts_orig_for_deform.to(self.device)
        Nv = int(V0.shape[0])

        e01 = torch.stack([F[:, 0], F[:, 1]], dim=1)
        e12 = torch.stack([F[:, 1], F[:, 2]], dim=1)
        e20 = torch.stack([F[:, 2], F[:, 0]], dim=1)
        E = torch.cat([e01, e12, e20], dim=0)  # (3Nf,2)
        E = torch.sort(E, dim=1).values
        E = torch.unique(E, dim=0)
        self._joint_reg_edges_e2 = E.detach()

        vi = V0[E[:, 0]]
        vj = V0[E[:, 1]]
        e0 = (vi - vj).norm(dim=1)
        self._joint_reg_e0 = e0.detach()

        # Directed adjacency for Laplacian: receiver src gets neighbor dst values
        src = torch.cat([E[:, 0], E[:, 1]], dim=0)
        dst = torch.cat([E[:, 1], E[:, 0]], dim=0)
        self._joint_reg_lap_src = src.detach()
        self._joint_reg_lap_dst = dst.detach()
        deg = torch.zeros((Nv,), device=self.device, dtype=torch.float32)
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float32))
        self._joint_reg_lap_deg = deg.clamp_min(1.0).detach()

    def ensure_joint_reg_mask(self, joint_reg_cfg) -> torch.Tensor | None:
        """Build (or reuse) joint soft mask M[v] from cloth LBS weights and config."""
        if self.lbs_weights is None:
            return None
        W = self.lbs_weights.to(self.device, torch.float32)  # (Nv,J)
        Nv, J = int(W.shape[0]), int(W.shape[1])
        names = [str(n) for n in list(JOINT_NAMES)[:J]]
        name_to_j = {n: i for i, n in enumerate(names)}

        # Read groups + mask settings (with defaults).
        def _cfg_get(obj, key, default=None):
            try:
                if obj is None:
                    return default
                if isinstance(obj, dict) or hasattr(obj, "get"):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        groups = _cfg_get(joint_reg_cfg, "joint_groups", None) or {}
        if not groups:
            groups = {
                "upper_limb": ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
                "lower_limb": ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"],
            }
        mask_mode = str(_cfg_get(joint_reg_cfg, "mask_mode", "sum_groups"))
        mask_pow = float(_cfg_get(joint_reg_cfg, "mask_pow", 0.5))
        mask_min = float(_cfg_get(joint_reg_cfg, "mask_min", 0.0))
        mask_max = float(_cfg_get(joint_reg_cfg, "mask_max", 1.0))

        # Signature to avoid recompute.
        sig = (
            mask_mode,
            float(mask_pow),
            float(mask_min),
            float(mask_max),
            tuple(groups.get("upper_limb", [])),
            tuple(groups.get("lower_limb", [])),
            tuple(groups.get("optional_spine", [])),
        )
        if self._joint_reg_mask_v is not None and self._joint_reg_mask_cfg_sig == sig and int(self._joint_reg_mask_v.shape[0]) == Nv:
            return self._joint_reg_mask_v

        def _idx_list(keys):
            out = []
            for k in keys:
                kk = str(k)
                if kk in name_to_j:
                    out.append(int(name_to_j[kk]))
            return out

        idx_upper = _idx_list(groups.get("upper_limb", []))
        idx_lower = _idx_list(groups.get("lower_limb", []))
        idx_spine = _idx_list(groups.get("optional_spine", []))

        m_upper = W[:, idx_upper].sum(dim=1) if idx_upper else torch.zeros((Nv,), device=self.device)
        m_lower = W[:, idx_lower].sum(dim=1) if idx_lower else torch.zeros((Nv,), device=self.device)
        m_spine = W[:, idx_spine].sum(dim=1) if idx_spine else torch.zeros((Nv,), device=self.device)

        if mask_mode == "max_group":
            M = torch.max(torch.max(m_upper, m_lower), m_spine)
        else:
            M = (m_upper + m_lower + m_spine)
        M = M.clamp(mask_min, mask_max)
        if mask_pow != 1.0:
            M = M.clamp_min(0.0).pow(mask_pow)

        self._joint_reg_mask_v = M.detach()
        self._joint_reg_mask_cfg_sig = sig
        return self._joint_reg_mask_v

    def create_from_canonical(self, ref_frame_mesh_data, garment_mask_ref_frame=None):
        raw_cano_verts = ref_frame_mesh_data['mesh_verts'].float().to(self.device)
        self.cano_verts_orig_for_deform = raw_cano_verts.clone() 
        self.cano_verts_used_for_lbs_basis = raw_cano_verts.clone() 

        raw_cano_norms = ref_frame_mesh_data['mesh_norms'].float().to(self.device)
        self.cano_norms = raw_cano_norms.clone()
        
        cano_faces_data = ref_frame_mesh_data['mesh_faces'].long().to(self.device)
        self.cano_faces = cano_faces_data 

        if self.use_deformation:
            if garment_mask_ref_frame is not None:
                self.garment_mask_from_labels = garment_mask_ref_frame.to(self.device)
                if self.verbose:
                    print(f"[SplattingAvatarModel] Received garment_mask for reference frame. Num garment_verts: {self.garment_mask_from_labels.sum()}.")
            else:
                print("[SplattingAvatarModel] WARNING: Garment mask for reference frame not provided. Using all-false mask.")
                self.garment_mask_from_labels = torch.zeros(self.cano_verts_orig_for_deform.shape[0], dtype=torch.bool, device=self.device)

        self.setup_canonical(self.cano_verts_used_for_lbs_basis, self.cano_norms, self.cano_faces)

        # TAO offset-input enhancement: optional SMPL-anchor cache in canonical/world space.
        try:
            if self.use_deformation:
                deform_cfg = self.config.get("deformation", {}) if (isinstance(self.config, dict) or hasattr(self.config, "get")) else {}
                input_mode = str(getattr(deform_cfg, "input_mode", deform_cfg.get("input_mode", "xyz"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "xyz"
                cache_on_init = bool(getattr(deform_cfg, "cache_projection_on_init", deform_cfg.get("cache_projection_on_init", True))) if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else True
                if input_mode == "smpl_anchor" and cache_on_init:
                    self._maybe_build_smpl_anchor_cache(force=False)
        except Exception as e:
            if self.verbose:
                print(f"[SplattingAvatarModel] Warning: failed to build smpl_anchor cache on init: {e}")

        # Clothfit xyz-query-override: initialize ref-before buffers once canonical mesh is known.
        try:
            self.maybe_init_xyz_query_override_buffers(force=False)
        except Exception as e:
            if self.verbose:
                print(f"[SplattingAvatarModel] Warning: failed to init xyz_query_override buffers: {e}")

        # TODO-4 v1.1: build UNION support set once geometry is available.
        try:
            if self.use_deformation and (self.lbs_weight_residual is not None) and (getattr(self.lbs_weight_residual, "support_mode", "topk_only") == "union"):
                if self._lbs_joint_pos_cano is not None:
                    self.lbs_weight_residual.build_support_from_geometry(
                        verts_cano_v3=self.cano_verts_used_for_lbs_basis,
                        joints_cano_j3=self._lbs_joint_pos_cano,
                    )
        except Exception as e:
            print(f"[SplattingAvatarModel] Warning: failed to build LBS union support: {e}")
        # P1 (joint-aware offset smoothing): precompute geometry caches (edges/adjacency). Mask is built lazily from config.
        try:
            self._ensure_joint_reg_geometry_caches()
        except Exception as e:
            if self.verbose:
                print(f"[SplattingAvatarModel] Warning: failed to precompute joint_reg geometry caches: {e}")
        
        # Allow reposing/testing to override GS ply + embedding.json paths without requiring
        # a legacy `point_cloud/point_cloud.ply` folder structure.
        gs_ply_override = None
        embed_json_override = None
        try:
            gs_ply_override = self.config.get("load_pretrained_gs_ply", None) if hasattr(self.config, "get") else None
        except Exception:
            gs_ply_override = None
        try:
            embed_json_override = self.config.get("load_pretrained_gs_embed_json", None) if hasattr(self.config, "get") else None
        except Exception:
            embed_json_override = None

        if self.pretrained_gs_path or gs_ply_override:
            ply_file_path = Path(str(gs_ply_override)) if gs_ply_override else (Path(self.pretrained_gs_path) / "point_cloud.ply")
            if not ply_file_path.exists():
                raise FileNotFoundError(f"[SplattingAvatarModel] Pre-trained Gaussian PLY not found: {ply_file_path}")
            self.load_ply(str(ply_file_path), init_params=False)
            self.n_initial_gaussians = self._xyz.shape[0]
            self.gaussians_are_frozen = True
            if self.verbose: print(f"[SplattingAvatarModel] Loaded and froze {self.n_initial_gaussians} Gaussians from {ply_file_path}.")

            embed_json_path = Path(str(embed_json_override)) if embed_json_override else (Path(self.pretrained_gs_path) / "embedding.json")
            if embed_json_path.exists():
                if self.verbose: print(f"[SplattingAvatarModel] Found embedding.json, loading: {embed_json_path}")
                self.load_from_embedding(str(embed_json_path)) 
                for p_name in ['_xyz', '_rotation']:
                    p_val = getattr(self, p_name)
                    if isinstance(p_val, nn.Parameter) and p_val.requires_grad:
                        print(f"Warning: {p_name} became grad-requiring Parameter after load_from_embedding. Forcing frozen.")
                        setattr(self, p_name, nn.Parameter(p_val.data, requires_grad=False))
            elif self.config.get('xyz_as_uvd', True) and self.use_deformation:
                print(f"[SplattingAvatarModel] WARNING: embedding.json not for ref Gaussians, but xyz_as_uvd is True for deformation. Creating dummy fidxs/bary.")
                sample_fidxs, sample_bary = sample_bary_on_triangles(self.cano_faces.shape[0], self.n_initial_gaussians)
                self.sample_fidxs = sample_fidxs.to(self.device); self.sample_bary = sample_bary.to(self.device)
        else: 
            self.mesh_verts = self.cano_verts_used_for_lbs_basis.clone()
            self.mesh_norms = self.cano_norms.clone()
            num_samples = self.config.get('num_init_samples', 10000)
            sample_fidxs, sample_bary = sample_bary_on_triangles(self.cano_faces.shape[0], num_samples)
            self.sample_fidxs = sample_fidxs.to(self.device); self.sample_bary = sample_bary.to(self.device)
            sample_verts = retrieve_verts_barycentric(self.mesh_verts, self.cano_faces, self.sample_fidxs, self.sample_bary)
            sample_norms = thf.normalize(retrieve_verts_barycentric(self.mesh_norms, self.cano_faces, self.sample_fidxs, self.sample_bary), dim=-1)
            pcd = BasicPointCloud(points=sample_verts.detach().cpu().numpy(), normals=sample_norms.detach().cpu().numpy(), colors=torch.full_like(sample_verts, 0.5).float().cpu())
            self.create_from_pcd(pcd)
            self.n_initial_gaussians = self._xyz.shape[0]
            if self.config.get('xyz_as_uvd', True):
                 self._xyz = torch.zeros_like(self._xyz)
            self.gaussians_are_frozen = False 

        if self.use_deformation:
            if self.n_initial_gaussians == 0: raise ValueError("n_initial_gaussians is 0. Cannot initialize DeformationModule.")
            if self.num_training_frames is None: raise ValueError("num_training_frames not available for DeformationModule.")
            deformation_config = self.config.get('deformation', {})
            deformation_use_tcnn = self.config.get('use_tcnn_deformation', self.config.get('tcnn_deformation', True))
            smpl_num_verts = None
            try:
                vt = getattr(self.smpl_model_for_lbs, "v_template", None)
                if isinstance(vt, torch.Tensor) and vt.ndim >= 2:
                    smpl_num_verts = int(vt.shape[-2])
            except Exception:
                smpl_num_verts = None
            self.deformation_module = DeformationModule(
                deformation_config,
                self.num_training_frames,
                self.n_initial_gaussians,
                use_tcnn=deformation_use_tcnn,
                smpl_num_verts=smpl_num_verts,
            ).to(self.device)
            if self.verbose:
                backend = getattr(self.deformation_module, "backend", "mlp")
                print(f"[SplattingAvatarModel] DeformationModule initialized (backend={backend}).")

            # TODO-5 (P2-1): static per-Gaussian base scale residual (delta_log_s_base).
            # Create parameter after Gaussians are loaded so Ng is known; keep usage gated by config.
            try:
                Ng = int(self._scaling.shape[0]) if hasattr(self, "_scaling") and self._scaling is not None else int(self.n_initial_gaussians)
            except Exception:
                Ng = int(self.n_initial_gaussians)
            if Ng > 0:
                if (not hasattr(self, "delta_log_s_base")) or (self.delta_log_s_base is None) or (tuple(self.delta_log_s_base.shape) != (Ng, 3)):
                    self.delta_log_s_base = nn.Parameter(torch.zeros((Ng, 3), device=self.device, dtype=torch.float32))
    
    def update_to_posed_mesh(
        self,
        raw_mesh_info_for_no_deform=None,
        current_target_frame_smpl_params=None,
        current_frame_idx=None,
        *,
        apply_vertex_offsets: bool = True,
        apply_gauss_bs: bool = True,
        latent_frame_idx_override: int = None,
    ):
        if not self.use_deformation:
            if raw_mesh_info_for_no_deform is not None:
                self.mesh_verts = raw_mesh_info_for_no_deform['mesh_verts'].float().to(self.device)
                self.mesh_norms = raw_mesh_info_for_no_deform['mesh_norms'].float().to(self.device)
                if self.quat_helper:
                    self.per_vert_quat = self.quat_helper(self.mesh_verts)
                    if self.cano_faces is not None: self.tri_quats = self.per_vert_quat[self.cano_faces]
                    self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)
            else:
                print("Warning: No deformation and no raw_mesh_info provided to update_to_posed_mesh.")
            self._current_delta_g_rgb_reshaped = None
            self._current_delta_u_local_ntb = None
            # Cloth-fit GS scale safety cache is still meaningful in no-deformation mode
            # (scales may be modulated by mesh scaling), so refresh it here.
            try:
                self._update_cloth_fit_gs_scale_safety_cache()
            except Exception as e:
                if self.verbose:
                    print(f"[SplattingAvatarModel] Warning: failed to update cloth-fit GS scale safety cache (no-deform): {e}")
            return

        if current_target_frame_smpl_params is None or current_frame_idx is None or self.deformation_module is None or self.smpl_model_for_lbs is None or self.inv_ref_pose_jnt_mats is None or self.lbs_weights is None:
            print("Warning: Missing components for deformation mode in update_to_posed_mesh. Skipping update.")
            return

        with torch.no_grad():
            target_jnt_mats = self._smplx_joint_mats_world_for_skinning(current_target_frame_smpl_params)
        
        ref_to_target_jnt_mats = torch.matmul(target_jnt_mats, self.inv_ref_pose_jnt_mats)

        smplx_pose_63_for_mlp = get_smplx_pose_63(current_target_frame_smpl_params).to(self.device)
        latent_idx = current_frame_idx if latent_frame_idx_override is None else int(latent_frame_idx_override)
        frame_idx_tensor = torch.tensor([latent_idx], device=self.device, dtype=torch.long)
        latent_code = self.deformation_module.get_frame_latent_code(frame_idx_tensor).squeeze(0)
        deform_backend = str(getattr(self.deformation_module, "backend", "mlp")).lower()
        
        # Vertex deformation variant control
        if apply_vertex_offsets:
            deform_cfg = self.config.get("deformation", {}) if (isinstance(self.config, dict) or hasattr(self.config, "get")) else {}
            input_mode = str(getattr(deform_cfg, "input_mode", deform_cfg.get("input_mode", "xyz"))).lower() if (isinstance(deform_cfg, dict) or hasattr(deform_cfg, "get")) else "xyz"

            # TAO offset-input enhancement: build SMPL-anchor cache lazily (covers init-time cache failures/skips).
            try:
                if (input_mode == "smpl_anchor" or deform_backend == "litept") and self._smpl_anchor_cache is None:
                    self._maybe_build_smpl_anchor_cache(force=False)
            except Exception as e:
                if self.verbose:
                    print(f"[SplattingAvatarModel] Warning: failed to (re)build smpl_anchor cache in update_to_posed_mesh: {e}")

            # Clothfit xyz-query-override (xyz-only): query vertex MLP with ref-before xyz, apply on target-shape base.
            use_xyz_override = False
            qcfg = getattr(deform_cfg, "xyz_query_override", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("xyz_query_override", None)
            if deform_backend != "litept" and input_mode == "xyz" and qcfg is not None:
                try:
                    enabled = bool(getattr(qcfg, "enabled", qcfg.get("enabled", False))) if (isinstance(qcfg, dict) or hasattr(qcfg, "get")) else False
                except Exception:
                    enabled = False
                if enabled:
                    # Optional stage gating if caller sets `self.current_stage` (1 or 2).
                    try:
                        use_s1 = bool(getattr(qcfg, "use_in_stage1", qcfg.get("use_in_stage1", True)))
                        use_s2 = bool(getattr(qcfg, "use_in_stage2", qcfg.get("use_in_stage2", True)))
                    except Exception:
                        use_s1, use_s2 = True, True
                    st = getattr(self, "current_stage", None)
                    if st == 1:
                        enabled = bool(use_s1)
                    elif st == 2:
                        enabled = bool(use_s2)
                    else:
                        # In unknown stage (e.g. reposing), treat as stage2 by default.
                        enabled = bool(use_s2)
                if enabled:
                    try:
                        if (self.xyz_query_ref_before_v3 is None) or (self.xyz_query_delta_shape_v3 is None):
                            self.maybe_init_xyz_query_override_buffers(force=False)
                        use_xyz_override = (self.xyz_query_ref_before_v3 is not None) and (self.xyz_query_delta_shape_v3 is not None)
                    except Exception as e:
                        if self.verbose:
                            print(f"[SplattingAvatarModel] Warning: xyz_query_override init failed: {e}")
                        use_xyz_override = False

            # Compute offsets explicitly so we can log their magnitude.
            V_query = self.xyz_query_ref_before_v3 if use_xyz_override else self.cano_verts_orig_for_deform
            if deform_backend == "litept":
                litept_cfg = getattr(deform_cfg, "litept", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("litept", None)
                token_feat_cfg = getattr(litept_cfg, "token_feat", None) if (litept_cfg is not None and not isinstance(litept_cfg, dict)) else (litept_cfg.get("token_feat", None) if isinstance(litept_cfg, dict) else None)
                pose_mode = str(getattr(token_feat_cfg, "pose_mode", token_feat_cfg.get("pose_mode", "dense_local"))).lower() if (isinstance(token_feat_cfg, dict) or hasattr(token_feat_cfg, "get")) else "dense_local"
                if self._smpl_anchor_cache is None:
                    self._maybe_build_smpl_anchor_cache(force=True)
                if self._smpl_anchor_cache is None:
                    raise RuntimeError("LitePT Task2 requires full-domain smpl_anchor_cache, but cache is missing.")

                face_idx = self._smpl_anchor_cache.get("face_idx", None)
                bary = self._smpl_anchor_cache.get("bary", None)
                q_canon = self._smpl_anchor_cache.get("q", None)
                if not isinstance(face_idx, torch.Tensor) or not isinstance(bary, torch.Tensor) or not isinstance(q_canon, torch.Tensor):
                    raise RuntimeError("LitePT Task2 requires anchor cache keys: face_idx, bary, q.")

                b_pose = None
                n_b_pose = None
                if pose_mode == "dense_local":
                    F_smpl = self._smplx_faces_tensor_for_anchor()
                    tri_vidx = F_smpl[face_idx.to(torch.long)]

                    # Pose-only dense body features: remove global rigid drift by using zeroed global terms.
                    p_ref_pose_only = self._smpl_params_pose_only_for_token_features(self._ref_frame_smpl_params_for_lbs)
                    p_cur_pose_only = self._smpl_params_pose_only_for_token_features(current_target_frame_smpl_params)
                    V_ref_pose_only = self._smplx_vertices_world_for_anchor(p_ref_pose_only)
                    V_cur_pose_only = self._smplx_vertices_world_for_anchor(p_cur_pose_only)
                    N_cur_pose_only = Meshes(verts=[V_cur_pose_only], faces=[F_smpl]).verts_normals_packed().contiguous()

                    Vref_tri = V_ref_pose_only[tri_vidx]
                    Vcur_tri = V_cur_pose_only[tri_vidx]
                    Ncur_tri = N_cur_pose_only[tri_vidx]
                    b_ref_pose_only = (bary.unsqueeze(-1) * Vref_tri).sum(dim=1)
                    b_cur_pose_only = (bary.unsqueeze(-1) * Vcur_tri).sum(dim=1)
                    n_b_pose = thf.normalize((bary.unsqueeze(-1) * Ncur_tri).sum(dim=1), dim=-1)
                    # Re-anchor pose-only delta to canonical cached body point frame.
                    b_pose = q_canon.to(b_cur_pose_only.device, b_cur_pose_only.dtype) + (b_cur_pose_only - b_ref_pose_only)

                betas = current_target_frame_smpl_params.get("betas", None)
                if not isinstance(betas, torch.Tensor):
                    betas = self._ref_frame_smpl_params_for_lbs.get("betas", None)
                if not isinstance(betas, torch.Tensor):
                    raise RuntimeError("LitePT Task2 requires betas tensor for beta embedding path.")
                if betas.ndim > 1:
                    betas = betas[0]

                vertex_offsets = self.deformation_module.vertex_deformation_net(
                    self.cano_verts_orig_for_deform,
                    smplx_pose_63_for_mlp,
                    latent_code,
                    self.garment_mask_from_labels,
                    smpl_anchor_cache=self._smpl_anchor_cache,
                    token_normals_canon=self.cano_norms,
                    b_pose=b_pose,
                    n_b_pose=n_b_pose,
                    betas=betas,
                    current_stage=getattr(self, "current_stage", None),
                )
            else:
                vertex_offsets = self.deformation_module.vertex_deformation_net(
                    V_query,
                    smplx_pose_63_for_mlp,
                    latent_code,
                    self.garment_mask_from_labels,
                    smpl_anchor_cache=self._smpl_anchor_cache,
                )
            self._last_vertex_offset_l2_mean = vertex_offsets.norm(dim=-1).mean()
            self._current_vertex_offsets = vertex_offsets
            if use_xyz_override and deform_backend != "litept":
                try:
                    self._last_vertex_offset_l2_mean_query = float(vertex_offsets.detach().float().norm(dim=-1).mean().cpu().item())
                except Exception:
                    self._last_vertex_offset_l2_mean_query = None
                # Optional A/B diagnostic: compare query-on-ref-before vs query-on-target-shape xyz.
                # This is useful when users see little difference and want to confirm the trick is actually affecting MLP outputs.
                try:
                    log_stats = bool(getattr(qcfg, "log_stats", qcfg.get("log_stats", False)))
                except Exception:
                    log_stats = False
                try:
                    compare_to_target_query = bool(getattr(qcfg, "compare_to_target_query", qcfg.get("compare_to_target_query", False)))
                except Exception:
                    compare_to_target_query = False
                if log_stats and compare_to_target_query:
                    try:
                        vertex_offsets_target_query = self.deformation_module.vertex_deformation_net(
                            self.cano_verts_orig_for_deform,
                            smplx_pose_63_for_mlp,
                            latent_code,
                            self.garment_mask_from_labels,
                            smpl_anchor_cache=self._smpl_anchor_cache,
                        )
                        diff = (vertex_offsets_target_query - vertex_offsets).detach().float()
                        self._xyz_query_override_offset_target_l2_mean = float(vertex_offsets_target_query.detach().float().norm(dim=-1).mean().cpu().item())
                        self._xyz_query_override_offset_diff_l2_mean = float(diff.norm(dim=-1).mean().cpu().item())
                        self._xyz_query_override_offset_diff_l2_max = float(diff.norm(dim=-1).max().cpu().item())
                    except Exception:
                        self._xyz_query_override_offset_target_l2_mean = None
                        self._xyz_query_override_offset_diff_l2_mean = None
                        self._xyz_query_override_offset_diff_l2_max = None
            if use_xyz_override:
                try:
                    apply_shape_delta = bool(getattr(qcfg, "apply_shape_delta", qcfg.get("apply_shape_delta", True)))
                except Exception:
                    apply_shape_delta = True
                V_base = (V_query + self.xyz_query_delta_shape_v3) if apply_shape_delta else self.cano_verts_orig_for_deform
            else:
                V_base = self.cano_verts_orig_for_deform
            deformed_ref_verts = V_base + vertex_offsets
        else:
            self._last_vertex_offset_l2_mean = torch.zeros((), device=self.device)
            self._current_vertex_offsets = None
            deformed_ref_verts = self.cano_verts_orig_for_deform

        lbs_w = self.get_lbs_weights_for_skinning()
        if lbs_w is None:
            raise RuntimeError("LBS weights unavailable for deformation skinning.")
        fwd_skin_mats_mesh = torch.einsum('vj,bjxy->bvxy', lbs_w, ref_to_target_jnt_mats).squeeze(0)
        deformed_ref_verts_homo = torch.cat([deformed_ref_verts, torch.ones_like(deformed_ref_verts[:, :1])], dim=-1).unsqueeze(-1)
        posed_verts_homo = torch.matmul(fwd_skin_mats_mesh, deformed_ref_verts_homo)
        self.mesh_verts = posed_verts_homo[:, :3, 0].contiguous()

        if self.cano_faces is not None and self.mesh_verts is not None:
            temp_posed_mesh_for_normals = Meshes(verts=[self.mesh_verts], faces=[self.cano_faces])
            self.mesh_norms = temp_posed_mesh_for_normals.verts_normals_packed().contiguous()
        else:
            self.mesh_norms = torch.zeros_like(self.mesh_verts)

        if self.quat_helper:
            self.per_vert_quat = self.quat_helper(self.mesh_verts) 
            if self.cano_faces is not None: self.tri_quats = self.per_vert_quat[self.cano_faces]
            self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)

        # Update cloth-fit GS scale safety cache once per frame.
        try:
            self._update_cloth_fit_gs_scale_safety_cache()
        except Exception as e:
            if self.verbose:
                print(f"[SplattingAvatarModel] Warning: failed to update cloth-fit GS scale safety cache: {e}")

        # --- LitePT Task5: GS alpha/logscale fields (Option A) ---
        if deform_backend == "litept":
            # Interpolate token fields (Nv,1)/(Nv,3) to mesh-attached Gaussians (Ng,1)/(Ng,3).
            try:
                # Stage-aware GS field gating (separate from optimizer freeze).
                deform_cfg = self.config.get("deformation", {}) if (isinstance(self.config, dict) or hasattr(self.config, "get")) else {}
                litept_cfg = getattr(deform_cfg, "litept", None) if not isinstance(deform_cfg, dict) else deform_cfg.get("litept", None)
                gs_fields_cfg = getattr(litept_cfg, "gs_fields", None) if (litept_cfg is not None and not isinstance(litept_cfg, dict)) else (litept_cfg.get("gs_fields", None) if isinstance(litept_cfg, dict) else None)
                enabled = True
                use_s1 = True
                use_s2 = True
                if gs_fields_cfg is not None:
                    try:
                        enabled = bool(getattr(gs_fields_cfg, "enabled", True))
                        use_s1 = bool(getattr(gs_fields_cfg, "use_in_stage1", True))
                        use_s2 = bool(getattr(gs_fields_cfg, "use_in_stage2", True))
                    except Exception:
                        if isinstance(gs_fields_cfg, dict):
                            enabled = bool(gs_fields_cfg.get("enabled", True))
                            use_s1 = bool(gs_fields_cfg.get("use_in_stage1", True))
                            use_s2 = bool(gs_fields_cfg.get("use_in_stage2", True))
                st = getattr(self, "current_stage", None)
                if st == 1:
                    enabled = bool(enabled and use_s1)
                elif st == 2:
                    enabled = bool(enabled and use_s2)
                if not enabled:
                    self._current_delta_alpha_logit = None
                    self._current_delta_logscale = None
                    self._last_delta_alpha_gs = None
                    self._last_delta_logscale_gs = None
                    # Skip all GS-field computation in this mode.
                    enabled = False

                if enabled:
                    net = getattr(self.deformation_module, "vertex_deformation_net", None)
                    alpha_field = getattr(net, "current_alpha_field", None) if net is not None else None
                    logscale_field = getattr(net, "current_logscale_field", None) if net is not None else None
                    if (
                        isinstance(alpha_field, torch.Tensor)
                        and isinstance(logscale_field, torch.Tensor)
                        and isinstance(self.cano_faces, torch.Tensor)
                        and isinstance(self.sample_fidxs, torch.Tensor)
                        and isinstance(self.sample_bary, torch.Tensor)
                    ):
                        d_alpha = interp_token_field_to_gs(
                            field_vc=alpha_field,
                            faces_f3=self.cano_faces,
                            sample_fidxs_ng=self.sample_fidxs,
                            sample_bary_ng3=self.sample_bary,
                        )
                        d_logs = interp_token_field_to_gs(
                            field_vc=logscale_field,
                            faces_f3=self.cano_faces,
                            sample_fidxs_ng=self.sample_fidxs,
                            sample_bary_ng3=self.sample_bary,
                        )
                        # Cache non-detached tensors so render losses can backprop to LitePT heads.
                        self._current_delta_alpha_logit = d_alpha
                        self._current_delta_logscale = d_logs
                        # Debug copies.
                        self._last_delta_alpha_gs = d_alpha.detach()
                        self._last_delta_logscale_gs = d_logs.detach()
                    else:
                        self._current_delta_alpha_logit = None
                        self._current_delta_logscale = None
                        self._last_delta_alpha_gs = None
                        self._last_delta_logscale_gs = None
            except Exception as e:
                if self.verbose:
                    print(f"[SplattingAvatarModel] Warning: failed to build LitePT GS field residuals: {e}")
                self._current_delta_alpha_logit = None
                self._current_delta_logscale = None
                self._last_delta_alpha_gs = None
                self._last_delta_logscale_gs = None
        else:
            self._current_delta_alpha_logit = None
            self._current_delta_logscale = None
            self._last_delta_alpha_gs = None
            self._last_delta_logscale_gs = None

        # Gaussian BS variant control
        # IMPORTANT: In LitePT backend mode, legacy Tao MLP-era Gaussian blendshapes MUST NOT be applied.
        # Keep modules/params for coexistence, but bypass runtime application.
        if deform_backend == "litept":
            apply_gauss_bs = False
        if apply_gauss_bs:
            delta_g_xyz, delta_g_rgb, pos_w, color_w = self.deformation_module.forward_gaussian_deform(
                smplx_pose_63_for_mlp, return_weights=True
            )
            # Expose coeff vectors for temporal smoothness losses (TODO-3)
            self._last_pos_blend_weights = pos_w.detach()
            self._last_color_blend_weights = color_w.detach()
            self._last_delta_u_l2_mean = delta_g_xyz.norm(dim=-1).mean()
            # δu_local in (N,T,B) frame (see docs/taoavatar/gs_position_res_design.md).
            self._current_delta_u_local_ntb = delta_g_xyz
            # SH0 DC term is stored as (Ng, 1, 3) in this codebase (coeff-first).
            self._current_delta_g_rgb_reshaped = delta_g_rgb.unsqueeze(1)
        else:
            self._last_delta_u_l2_mean = torch.zeros((), device=self.device)
            self._current_delta_u_local_ntb = None
            self._current_delta_g_rgb_reshaped = None
            self._last_pos_blend_weights = None
            self._last_color_blend_weights = None

        # Cache depends on deformation outputs (face scaling / state), so refresh at end too.
        try:
            self._update_cloth_fit_gs_scale_safety_cache()
        except Exception as e:
            if self.verbose:
                print(f"[SplattingAvatarModel] Warning: failed to update cloth-fit GS scale safety cache (post): {e}")

    def update_to_cano_mesh(self):
        self._current_delta_g_rgb_reshaped = None
        self._current_delta_u_local_ntb = None
        self._current_delta_alpha_logit = None
        self._current_delta_logscale = None
        self._last_delta_alpha_gs = None
        self._last_delta_logscale_gs = None
            
        if self.cano_verts_used_for_lbs_basis is not None and self.cano_norms is not None:
            self.mesh_verts = self.cano_verts_used_for_lbs_basis
            self.mesh_norms = self.cano_norms
            
            self.per_vert_quat = self.quat_helper(self.mesh_verts) 
            self.tri_quats = self.per_vert_quat[self.cano_faces]
            self._face_scaling = self.quat_helper.calc_face_area_change(self.mesh_verts)
        else:
            if self.verbose:
                print("Warning: Original canonical info not fully available for reset in update_to_cano_mesh.")

    def prune_points(self, valid_points_mask, optimizable_tensors):
        if self.gaussians_are_frozen or self.use_deformation:
            return

        self._xyz = optimizable_tensors['_xyz']

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = self._scaling[valid_points_mask]

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = self._rotation[valid_points_mask]

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        if self.config.get('xyz_as_uvd', True):
            self.sample_fidxs = self.sample_fidxs[valid_points_mask]
            self.sample_bary = self.sample_bary[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densification_postfix(self, optimizable_tensors, densify_out):
        if self.gaussians_are_frozen or self.use_deformation:
            return

        self._xyz = optimizable_tensors.get('_xyz', self._xyz)

        if '_scaling' in optimizable_tensors:
            self._scaling = optimizable_tensors['_scaling']
        else:
            self._scaling = torch.cat([self._scaling, densify_out['new_scaling']], dim=0)

        if '_rotation' in optimizable_tensors:
            self._rotation = optimizable_tensors['_rotation']
        else:
            self._rotation = torch.cat([self._rotation, densify_out['new_rotation']], dim=0)

        self._opacity = optimizable_tensors.get('_opacity', self._opacity)
        self._features_dc = optimizable_tensors.get('_features_dc', self._features_dc)
        self._features_rest = optimizable_tensors.get('_features_rest', self._features_rest)

        if self.config.get('xyz_as_uvd', True):
            self.sample_fidxs = torch.cat([self.sample_fidxs, densify_out['new_sample_fidxs']], dim=0)
            self.sample_bary = torch.cat([self.sample_bary, densify_out['new_sample_bary']], dim=0)

        # stats
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device='cuda')

    def prepare_densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        if self.gaussians_are_frozen or self.use_deformation:
            return torch.empty(0, dtype=torch.bool, device=self.device), torch.empty(0, 3, device=self.device)
        n_init_points = self._xyz.shape[0]
        padded_grad = torch.zeros((n_init_points), device='cuda')
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)

        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling_cano, dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling_cano[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device='cuda')
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self.get_rotation_cano[selected_pts_mask]).repeat(N,1,1)
        
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz_cano[selected_pts_mask].repeat(N, 1)
        return selected_pts_mask, new_xyz.detach()
 
    def prepare_split_selected_to_new_xyz(self, selected_pts_mask, new_xyz, N):
        if self.gaussians_are_frozen or self.use_deformation:
            return {}
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling_cano[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)

        splitout = {
            'new_xyz': new_xyz,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        if self.config.get('xyz_as_uvd', True):
            fidx = self.sample_fidxs[selected_pts_mask].repeat(N)
            uv = self.sample_bary[selected_pts_mask, :2].repeat(N, 1)
            d_from_original_uvd = self._xyz[selected_pts_mask, -1:].repeat(N, 1)

            if not self.config.get('skip_triangle_walk', False):
                fidx, uv = self.phongsurf.update_corres_spt(new_xyz, None, fidx, uv)

            bary = torch.concat([uv, 1.0 - uv[:, 0:1] - uv[:, 1:2]], dim=-1)
            new_xyz_uvd = torch.concat([torch.zeros_like(uv), d_from_original_uvd], dim=-1)
        
            splitout.update({
                'new_xyz': new_xyz_uvd,
                'new_sample_fidxs': fidx,
                'new_sample_bary': bary,
            })
    
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        splitout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        return splitout

    def prepare_densify_and_clone(self, grads, grad_threshold, scene_extent):
        if self.gaussians_are_frozen or self.use_deformation:
            return {}
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz_clone = self._xyz[selected_pts_mask]

        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        cloneout = {
            'new_xyz': new_xyz_clone,
            'new_scaling': new_scaling,
            'new_rotation': new_rotation,
        }

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        cloneout.update({
            'new_features_dc': new_features_dc,
            'new_features_rest': new_features_rest,
            'new_opacity': new_opacity,
        })

        if self.config.get('xyz_as_uvd', True):
            cloneout.update({
                'new_sample_fidxs': self.sample_fidxs[selected_pts_mask],
                'new_sample_bary': self.sample_bary[selected_pts_mask],
            })

        return cloneout

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if self.gaussians_are_frozen or self.use_deformation:
            return
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def walking_on_triangles(self):
        if self.use_deformation or self.gaussians_are_frozen or self.config.get('skip_triangle_walk', False) or not self.config.get('xyz_as_uvd', True):
            return
        fidx = self.sample_fidxs.detach().cpu().numpy().astype(np.int32)
        uv = self.sample_bary[..., :2].detach().cpu().numpy().astype(np.double)
        delta_uv_params = self._xyz[..., :2].detach().cpu().numpy().astype(np.double)
        fidx, uv = self.phongsurf.triwalk.updateSurfacePoints(fidx, uv, delta_uv_params)

        self.sample_fidxs = torch.tensor(fidx).long().to(self.device)
        self.sample_bary[..., :2] = torch.tensor(uv).float().to(self.device)
        self.sample_bary[..., 2] = 1.0 - self.sample_bary[..., 0] - self.sample_bary[..., 1]

    def load_from_embedding(self, embed_fn):
        if self.verbose: print(f"[SplattingAvatarModel] Loading embedding from: {embed_fn}")
        with open(embed_fn, 'r') as fp:
            cc = json.load(fp)

        mesh_fn_in_json = cc.get('cano_mesh', cc.get('mesh_fn'))
        if not mesh_fn_in_json:
            print(f"Error: 'cano_mesh' or 'mesh_fn' key not found in embedding file {embed_fn}")
            return
            
        mesh_full_path = Path(embed_fn).parent / mesh_fn_in_json
        if not mesh_full_path.exists():
            print(f"Warning: Embedded mesh {mesh_fn_in_json} not found at {mesh_full_path}.")
        
        if (not hasattr(self, 'cano_verts') or self.cano_verts is None) and mesh_full_path.exists():
            print(f"Canonical mesh not yet set, loading from embedded mesh: {mesh_full_path}")
            try:
                cano_mesh_cpu = libcore.MeshCpu(str(mesh_full_path))
                cano_verts = torch.tensor(cano_mesh_cpu.V).float().to(self.device)
                cano_norms = torch.tensor(cano_mesh_cpu.N).float().to(self.device)
                cano_faces = torch.tensor(cano_mesh_cpu.F).long().to(self.device)

                self.setup_canonical(cano_verts, cano_norms, cano_faces) 
            except Exception as e:
                print(f"Error loading mesh from {mesh_full_path} during embedding load: {e}")

        loaded_xyz = torch.tensor(cc['_xyz']).float().to(self.device)
        loaded_rotation = torch.tensor(cc['_rotation']).float().to(self.device) if '_rotation' in cc else self._rotation
        
        if self.gaussians_are_frozen:
            self._xyz = loaded_xyz
            self._rotation = loaded_rotation
        else:
            # NOTE: Do NOT assign the return value of `.copy_()` back to the module attribute:
            # `copy_()` returns a Tensor, which would break nn.Parameter registration.
            if not isinstance(self._xyz, nn.Parameter):
                self._xyz = nn.Parameter(loaded_xyz)
            else:
                self._xyz.data.copy_(loaded_xyz)

            if not isinstance(self._rotation, nn.Parameter):
                self._rotation = nn.Parameter(loaded_rotation)
            else:
                self._rotation.data.copy_(loaded_rotation)
            if self.verbose: print("Updated _xyz and _rotation from embedding.json for learnable Gaussians.")

        self.sample_fidxs = torch.tensor(cc['sample_fidxs']).long().to(self.device)
        self.sample_bary = torch.tensor(cc.get('sample_bary', cc.get('_sample_bary'))).float().to(self.device)
        if self.verbose: print(f"Loaded sample_fidxs ({self.sample_fidxs.shape}) and sample_bary ({self.sample_bary.shape}) from embedding.")
        
    def save_embedding_json(self, embed_fn):
        if (not hasattr(self, 'cano_verts') or self.cano_verts is None) and (self.cano_verts is None or self.cano_norms is None or self.cano_faces is None):
            print("Warning: Cannot save embedding JSON, canonical mesh info not set.")
            return

        obj_fn = Path(embed_fn).stem + '.obj'
        obj_full_path = Path(embed_fn).parent / obj_fn
        
        temp_mesh_cpu = libcore.MeshCpu()
        temp_mesh_cpu.V = self.cano_verts_used_for_lbs_basis.detach().cpu() 
        temp_mesh_cpu.N = self.cano_norms.detach().cpu()
        temp_mesh_cpu.F = self.cano_faces.detach().cpu()
        temp_mesh_cpu.FN = temp_mesh_cpu.F 
        temp_mesh_cpu.save_to_obj(str(obj_full_path))

        xyz_to_save = self._xyz.data.detach().cpu().tolist() if isinstance(self._xyz, nn.Parameter) else self._xyz.detach().cpu().tolist()
        rot_to_save = self._rotation.data.detach().cpu().tolist() if isinstance(self._rotation, nn.Parameter) else self._rotation.detach().cpu().tolist()

        embedding = {
            'cano_mesh': obj_fn,
            'sample_fidxs': self.sample_fidxs.detach().cpu().tolist(),
            'sample_bary': self.sample_bary.detach().cpu().tolist(),
            '_xyz': xyz_to_save,
            '_rotation': rot_to_save,
        }

        with open(embed_fn, 'w') as f:
            json.dump(embedding, f)
            if self.verbose: print(f"Saved embedding to {embed_fn} and mesh to {obj_full_path}")


