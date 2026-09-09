from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Iterable

import torch
import torch.nn as nn

from .litept_token_features import build_litept_fullbody_tokens, infer_litept_token_feat_dim


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    try:
        if isinstance(cfg, dict) or hasattr(cfg, "get"):
            return cfg.get(key, default)
    except Exception:
        pass
    try:
        return getattr(cfg, key)
    except Exception:
        return default


def _xyz_pe_encode(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if int(num_freqs) <= 0:
        return x
    out = [x]
    freq_bands = 2.0 ** torch.arange(int(num_freqs), device=x.device, dtype=x.dtype)
    for f in freq_bands:
        out.append(torch.sin(x * f))
        out.append(torch.cos(x * f))
    return torch.cat(out, dim=-1)


class LitePTBackboneWrapper(nn.Module):
    """Thin wrapper around vendored LitePT that accepts project tensors."""

    def __init__(self, *, in_channels: int, out_channels: int, litept_cfg=None):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.grid_size = float(_cfg_get(litept_cfg, "grid_size", 0.01))

        # Vendored LitePT lives at splatting/litept/litept/model.py
        repo_root = Path(__file__).resolve().parents[1]
        litept_repo_dir = repo_root / "litept"
        if str(litept_repo_dir) not in sys.path:
            sys.path.insert(0, str(litept_repo_dir))

        try:
            from litept.model import LitePT  # type: ignore
        except Exception as e:
            raise ImportError(
                "Failed to import vendored LitePT from `splatting/litept`. "
                "Install LitePT runtime dependencies in splatting uv env "
                "(flash-attn, spconv, torch-scatter, timm, addict)."
            ) from e

        model_kwargs = {}
        try:
            maybe_kwargs = _cfg_get(litept_cfg, "model_kwargs", {})
            if isinstance(maybe_kwargs, dict):
                model_kwargs = dict(maybe_kwargs)
            elif hasattr(maybe_kwargs, "items"):
                model_kwargs = {k: v for k, v in maybe_kwargs.items()}
        except Exception:
            model_kwargs = {}

        model_kwargs["in_channels"] = int(self.in_channels)
        self.model = LitePT(**model_kwargs)

    def forward(self, *, coord: torch.Tensor, feat: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if coord.ndim != 2 or coord.shape[-1] != 3:
            raise ValueError(f"LitePT expected coord [N,3], got shape={tuple(coord.shape)}")
        if feat.ndim != 2:
            raise ValueError(f"LitePT expected feat [N,C], got shape={tuple(feat.shape)}")
        if batch.ndim != 1 or batch.shape[0] != coord.shape[0]:
            raise ValueError(f"LitePT expected batch [N], got shape={tuple(batch.shape)} for N={coord.shape[0]}")

        data_dict = {
            "coord": coord.float().contiguous(),
            "feat": feat.float().contiguous(),
            "batch": batch.long().contiguous(),
            "grid_size": float(self.grid_size),
        }
        point = self.model(data_dict)
        dense_feat = point.feat
        if dense_feat.shape[0] != coord.shape[0]:
            raise RuntimeError(
                f"LitePT output token mismatch: in={int(coord.shape[0])}, out={int(dense_feat.shape[0])}"
            )
        return dense_feat


class LitePTVertexDeformationNetwork(nn.Module):
    """LitePT vertex deformation backend (Task-3 heads + composition)."""

    def __init__(
        self,
        *,
        num_pose_params: int = 63,
        latent_code_dim: int = 32,
        litept_cfg=None,
        input_mode: str = "xyz",
    ):
        super().__init__()
        self.num_pose_params = int(num_pose_params)
        self.latent_code_dim = int(latent_code_dim)
        self.input_mode = str(input_mode).lower()
        self.token_feat_cfg = _cfg_get(litept_cfg, "token_feat", {}) or {}

        feature_dim = int(_cfg_get(litept_cfg, "feature_dim", 72))
        beta_embed_dim = int(_cfg_get(litept_cfg, "beta_embed_dim", 32))
        head_hidden_dim = int(_cfg_get(litept_cfg, "head_hidden_dim", 128))
        pose_mode = str(_cfg_get(self.token_feat_cfg, "pose_mode", "dense_local")).lower()
        self.pose_mode = pose_mode
        global_theta_proj_cfg = _cfg_get(self.token_feat_cfg, "global_theta_proj", {}) or {}
        theta_hidden_dim = int(_cfg_get(global_theta_proj_cfg, "hidden_dim", 32))
        theta_out_dim = int(_cfg_get(global_theta_proj_cfg, "out_dim", 16))
        self.theta_out_dim = theta_out_dim
        self.in_channels = infer_litept_token_feat_dim(
            token_feat_cfg=self.token_feat_cfg,
            global_theta_out_dim=self.theta_out_dim,
        )
        pose_head_coord_cfg = _cfg_get(litept_cfg, "pose_head_coord", {}) or {}
        self.pose_head_coord_xyz_encoding = str(_cfg_get(pose_head_coord_cfg, "xyz_encoding", "none")).strip().lower()
        self.pose_head_coord_xyz_pe_L = int(_cfg_get(pose_head_coord_cfg, "xyz_pe_L", 2))
        self.pose_head_coord_include_raw_xyz_with_pe = bool(
            _cfg_get(pose_head_coord_cfg, "include_raw_xyz_with_pe", True)
        )
        if self.pose_head_coord_xyz_encoding not in {"none", "raw", "pe"}:
            raise RuntimeError(
                "Invalid litept.pose_head_coord.xyz_encoding="
                f"'{self.pose_head_coord_xyz_encoding}'. Expected none/raw/pe."
            )
        if self.pose_head_coord_xyz_pe_L < 0:
            raise RuntimeError("litept.pose_head_coord.xyz_pe_L must be >= 0.")
        if self.pose_head_coord_xyz_encoding == "none":
            self.pose_head_coord_dim = 0
        elif self.pose_head_coord_xyz_encoding == "raw":
            self.pose_head_coord_dim = 3
        else:
            pe_dim = 3 * (1 + 2 * int(self.pose_head_coord_xyz_pe_L))
            self.pose_head_coord_dim = int(pe_dim + (3 if self.pose_head_coord_include_raw_xyz_with_pe else 0))

        # Small global-theta projection for pose_mode=global_theta.
        if self.pose_mode == "global_theta":
            self.theta_proj = nn.Sequential(
                nn.Linear(self.num_pose_params, theta_hidden_dim),
                nn.SiLU(),
                nn.Linear(theta_hidden_dim, self.theta_out_dim),
            )
        else:
            self.theta_proj = None
        interaction_cfg = _cfg_get(litept_cfg, "interaction", {}) or {}
        self.interaction_enabled = bool(_cfg_get(interaction_cfg, "enabled", False))
        self.interaction_scale = float(_cfg_get(interaction_cfg, "scale", 1.0))
        self.debug_checks = bool(_cfg_get(litept_cfg, "debug_checks", True))
        self.gate_bias_init = float(_cfg_get(litept_cfg, "gate_bias_init", -4.0))
        gs_fields_cfg = _cfg_get(litept_cfg, "gs_fields", {}) or {}
        self.gs_fields_enabled = bool(_cfg_get(gs_fields_cfg, "enabled", True))
        self.gs_fields_use_in_stage1 = bool(_cfg_get(gs_fields_cfg, "use_in_stage1", True))
        self.gs_fields_use_in_stage2 = bool(_cfg_get(gs_fields_cfg, "use_in_stage2", True))
        self.beta_mlp = nn.Sequential(
            nn.LazyLinear(beta_embed_dim),
            nn.SiLU(),
            nn.Linear(beta_embed_dim, beta_embed_dim),
        )
        self.last_e_beta = None
        self.last_token_debug = None
        self.last_token_feat_F = None
        self.last_delta_pose = None
        self.last_delta_int = None
        self.last_gate = None
        self.last_offsets = None
        self.current_alpha_field = None
        self.current_logscale_field = None
        self.last_alpha_field = None
        self.last_logscale_field = None
        self._token_cfg_logged = False
        self._pose_head_coord_cfg_logged = False
        self.last_pose_head_input_dim = None

        self.backbone = LitePTBackboneWrapper(
            in_channels=self.in_channels,
            out_channels=feature_dim,
            litept_cfg=litept_cfg,
        )

        # Task-3 heads: lightweight 2-layer MLPs.
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim + int(self.pose_head_coord_dim), head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 3),
        )
        self.interaction_head = nn.Sequential(
            nn.Linear(feature_dim + beta_embed_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 3),
        )
        self.gate_head = nn.Sequential(
            nn.Linear(feature_dim + beta_embed_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 1),
        )
        with torch.no_grad():
            # identity-like init
            nn.init.zeros_(self.pose_head[-1].weight)
            nn.init.zeros_(self.pose_head[-1].bias)
            nn.init.zeros_(self.interaction_head[-1].weight)
            nn.init.zeros_(self.interaction_head[-1].bias)
            # conservative gate at start
            nn.init.zeros_(self.gate_head[-1].weight)
            nn.init.constant_(self.gate_head[-1].bias, self.gate_bias_init)

        # Task-5: GS field heads (Option A field-only).
        # Predict token-level residual fields that will later be interpolated to Gaussians.
        self.gs_alpha_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 1),
        )
        self.gs_logscale_head = nn.Sequential(
            nn.Linear(feature_dim, head_hidden_dim),
            nn.SiLU(),
            nn.Linear(head_hidden_dim, 3),
        )
        with torch.no_grad():
            nn.init.zeros_(self.gs_alpha_head[-1].weight)
            nn.init.zeros_(self.gs_alpha_head[-1].bias)
            nn.init.zeros_(self.gs_logscale_head[-1].weight)
            nn.init.zeros_(self.gs_logscale_head[-1].bias)

    def _build_pose_head_input(self, *, F_tok: torch.Tensor, x_canon: torch.Tensor) -> torch.Tensor:
        pe = None
        if self.pose_head_coord_xyz_encoding == "none":
            pose_in = F_tok
        elif self.pose_head_coord_xyz_encoding == "raw":
            pose_in = torch.cat([F_tok, x_canon], dim=-1)
        elif self.pose_head_coord_xyz_encoding == "pe":
            pe = _xyz_pe_encode(x_canon, self.pose_head_coord_xyz_pe_L)
            if self.pose_head_coord_include_raw_xyz_with_pe:
                pose_in = torch.cat([F_tok, x_canon, pe], dim=-1)
            else:
                pose_in = torch.cat([F_tok, pe], dim=-1)
        else:
            raise RuntimeError(
                "Invalid pose_head_coord.xyz_encoding="
                f"'{self.pose_head_coord_xyz_encoding}'. Expected none/raw/pe."
            )
        if pose_in.shape[0] != F_tok.shape[0]:
            raise RuntimeError(
                f"pose_head input Nv mismatch: F_tok={int(F_tok.shape[0])}, pose_in={int(pose_in.shape[0])}"
            )
        expected = int(F_tok.shape[-1]) + int(self.pose_head_coord_dim)
        if int(pose_in.shape[-1]) != expected:
            raise RuntimeError(f"pose_head input dim mismatch: got {int(pose_in.shape[-1])}, expected {expected}")
        if self.debug_checks:
            if not torch.isfinite(x_canon).all():
                raise RuntimeError("x_canon for pose_head_coord contains NaN/Inf.")
            if pe is not None and (not torch.isfinite(pe).all()):
                raise RuntimeError("PE(x_canon) for pose_head_coord contains NaN/Inf.")
            if not torch.isfinite(pose_in).all():
                raise RuntimeError("pose_head input contains NaN/Inf.")
        self.last_pose_head_input_dim = int(pose_in.shape[-1])
        return pose_in

    def forward(
        self,
        cano_verts: torch.Tensor,
        pose_params: torch.Tensor,
        frame_latent_code: torch.Tensor,
        garment_mask: torch.Tensor,
        *,
        smpl_anchor_cache=None,
        token_normals_canon: torch.Tensor | None = None,
        b_pose: torch.Tensor | None = None,
        n_b_pose: torch.Tensor | None = None,
        betas: torch.Tensor | None = None,
        current_stage: int | None = None,
    ) -> torch.Tensor:
        del garment_mask  # Reserved for future LitePT garment-aware heads.
        del frame_latent_code
        pose_global_feat = None
        if self.pose_mode == "global_theta":
            pose_in = pose_params.to(cano_verts.device, torch.float32).reshape(1, -1)
            if pose_in.shape[-1] != self.num_pose_params:
                raise RuntimeError(
                    f"pose_mode=global_theta expects pose_params dim={self.num_pose_params}, got {int(pose_in.shape[-1])}"
                )
            if self.theta_proj is None:
                raise RuntimeError("pose_mode=global_theta but theta_proj is not initialized.")
            pose_global_feat = self.theta_proj(pose_in).squeeze(0)

        token_pack = build_litept_fullbody_tokens(
            x_canon=cano_verts.float(),
            n_canon=(token_normals_canon.float() if isinstance(token_normals_canon, torch.Tensor) else None),
            anchor_cache=smpl_anchor_cache,
            b_pose=(b_pose.float() if isinstance(b_pose, torch.Tensor) else None),
            n_b_pose=(n_b_pose.float() if isinstance(n_b_pose, torch.Tensor) else None),
            token_feat_cfg=self.token_feat_cfg,
            pose_global_feat=pose_global_feat,
        )
        coord = token_pack["coord"]
        feat = token_pack["feat"]
        self.last_token_debug = token_pack.get("debug", None)
        if feat.shape[-1] != self.in_channels:
            raise RuntimeError(
                f"LitePT feat dim mismatch: got {int(feat.shape[-1])}, expected in_channels={int(self.in_channels)}"
            )
        if self.debug_checks and (not self._token_cfg_logged):
            dbg = self.last_token_debug if isinstance(self.last_token_debug, dict) else {}
            print(
                "[LitePTToken] "
                f"pose_mode={dbg.get('pose_mode', self.pose_mode)} "
                f"active_blocks={dbg.get('active_blocks', 'n/a')} "
                f"feat_dim={dbg.get('feat_dim', int(feat.shape[-1]))} "
                f"use_token_xyz_in_feat={dbg.get('use_token_xyz_in_feat', _cfg_get(self.token_feat_cfg, 'use_token_xyz_in_feat', True))} "
                f"token_xyz_encoding={dbg.get('token_xyz_encoding', _cfg_get(self.token_feat_cfg, 'token_xyz_encoding', 'raw'))} "
                f"token_xyz_pe_L={dbg.get('token_xyz_pe_L', _cfg_get(self.token_feat_cfg, 'token_xyz_pe_L', 6))}"
            )
            self._token_cfg_logged = True
        if betas is None:
            raise RuntimeError("LitePT Task2 requires betas for beta embedding path.")
        beta_in = betas.to(cano_verts.device, torch.float32).reshape(1, -1)
        self.last_e_beta = self.beta_mlp(beta_in).squeeze(0)
        if not torch.isfinite(self.last_e_beta).all():
            raise RuntimeError("e_beta contains NaN/Inf.")
        batch = torch.zeros((coord.shape[0],), device=coord.device, dtype=torch.long)
        F_tok = self.backbone(coord=coord, feat=feat, batch=batch)
        pose_head_in = self._build_pose_head_input(F_tok=F_tok, x_canon=coord)
        if self.debug_checks and (not self._pose_head_coord_cfg_logged):
            print(
                "[LitePTPoseHeadCoord] "
                f"xyz_encoding={self.pose_head_coord_xyz_encoding} "
                f"xyz_pe_L={int(self.pose_head_coord_xyz_pe_L)} "
                f"include_raw_xyz_with_pe={bool(self.pose_head_coord_include_raw_xyz_with_pe)} "
                f"pose_in_dim={int(pose_head_in.shape[-1])}"
            )
            self._pose_head_coord_cfg_logged = True
        e_beta_tok = self.last_e_beta.view(1, -1).expand(F_tok.shape[0], -1)
        head_in = torch.cat([F_tok, e_beta_tok], dim=-1)

        # Task-5 GS fields (Option A). Can be runtime-disabled (e.g., Stage 1).
        stage_i = None
        try:
            stage_i = int(current_stage) if current_stage is not None else None
        except Exception:
            stage_i = None
        gs_fields_enabled_now = bool(self.gs_fields_enabled)
        if stage_i == 1:
            gs_fields_enabled_now = gs_fields_enabled_now and bool(self.gs_fields_use_in_stage1)
        elif stage_i == 2:
            gs_fields_enabled_now = gs_fields_enabled_now and bool(self.gs_fields_use_in_stage2)
        if gs_fields_enabled_now:
            # Keep non-detached tensors for downstream losses/gradients.
            alpha_field = self.gs_alpha_head(F_tok)
            logscale_field = self.gs_logscale_head(F_tok)
            self.current_alpha_field = alpha_field
            self.current_logscale_field = logscale_field
        else:
            alpha_field = None
            logscale_field = None
            self.current_alpha_field = None
            self.current_logscale_field = None

        delta_pose = self.pose_head(pose_head_in)
        if self.interaction_enabled:
            delta_int = self.interaction_head(head_in) * float(self.interaction_scale)
            gate = torch.sigmoid(self.gate_head(head_in))
        else:
            delta_int = torch.zeros_like(delta_pose)
            gate = torch.zeros((delta_pose.shape[0], 1), device=delta_pose.device, dtype=delta_pose.dtype)
        offsets = delta_pose + gate * delta_int

        if self.debug_checks:
            if gs_fields_enabled_now:
                if alpha_field is None or logscale_field is None:
                    raise RuntimeError("gs_fields_enabled_now is true but alpha/logscale fields are None.")
                if alpha_field.shape != (cano_verts.shape[0], 1):
                    raise RuntimeError(f"alpha_field shape mismatch: {tuple(alpha_field.shape)}")
                if logscale_field.shape != (cano_verts.shape[0], 3):
                    raise RuntimeError(f"logscale_field shape mismatch: {tuple(logscale_field.shape)}")
            if delta_pose.shape != (cano_verts.shape[0], 3):
                raise RuntimeError(f"delta_pose shape mismatch: {tuple(delta_pose.shape)}")
            if delta_int.shape != (cano_verts.shape[0], 3):
                raise RuntimeError(f"delta_int shape mismatch: {tuple(delta_int.shape)}")
            if gate.shape != (cano_verts.shape[0], 1):
                raise RuntimeError(f"gate shape mismatch: {tuple(gate.shape)}")
            if offsets.shape != (cano_verts.shape[0], 3):
                raise RuntimeError(f"offsets shape mismatch: {tuple(offsets.shape)}")
            for name, x in [
                ("F_tok", F_tok),
                ("delta_pose", delta_pose),
                ("delta_int", delta_int),
                ("gate", gate),
                ("offsets", offsets),
            ]:
                if not torch.isfinite(x).all():
                    raise RuntimeError(f"{name} contains NaN/Inf.")
            if gs_fields_enabled_now:
                for name, x in [("alpha_field", alpha_field), ("logscale_field", logscale_field)]:
                    if x is None or (not torch.isfinite(x).all()):
                        raise RuntimeError(f"{name} contains NaN/Inf (or is None).")
            if float(gate.min().item()) < -1e-6 or float(gate.max().item()) > 1.0 + 1e-6:
                raise RuntimeError(f"gate range violation: min={float(gate.min().item())}, max={float(gate.max().item())}")

        self.last_token_feat_F = F_tok.detach()
        self.last_delta_pose = delta_pose.detach()
        self.last_delta_int = delta_int.detach()
        self.last_gate = gate.detach()
        self.last_offsets = offsets.detach()
        self.last_alpha_field = alpha_field.detach() if isinstance(alpha_field, torch.Tensor) else None
        self.last_logscale_field = logscale_field.detach() if isinstance(logscale_field, torch.Tensor) else None
        return offsets.to(cano_verts.dtype)

    def get_litept_param_groups(self) -> Dict[str, Iterable[nn.Parameter]]:
        return {
            "litept_trunk_early": self.backbone.model.embedding.parameters(),
            "litept_trunk_late": self.backbone.model.enc.parameters(),
            "litept_decoder": self.backbone.model.dec.parameters() if hasattr(self.backbone.model, "dec") else [],
            "litept_pose_head": self.pose_head.parameters(),
            "litept_interaction_head": self.interaction_head.parameters(),
            "litept_gate_head": self.gate_head.parameters(),
            "litept_gs_alpha_head": self.gs_alpha_head.parameters(),
            "litept_gs_logscale_head": self.gs_logscale_head.parameters(),
        }

