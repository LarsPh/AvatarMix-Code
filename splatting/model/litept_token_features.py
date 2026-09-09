from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def _require_anchor_key(anchor_cache: dict, key: str, nv: int, shape_tail: Tuple[int, ...]) -> torch.Tensor:
    if not isinstance(anchor_cache, dict) or key not in anchor_cache:
        raise RuntimeError(f"LitePT token feature build requires anchor_cache['{key}'].")
    x = anchor_cache[key]
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"anchor_cache['{key}'] must be a torch.Tensor, got {type(x).__name__}.")
    if x.shape[0] != nv or tuple(x.shape[1:]) != tuple(shape_tail):
        raise RuntimeError(
            f"anchor_cache['{key}'] shape mismatch: got {tuple(x.shape)}, expected ({nv}, {', '.join(map(str, shape_tail))})."
        )
    return x


def _validate_finite(name: str, x: torch.Tensor) -> None:
    if not torch.isfinite(x).all():
        raise RuntimeError(f"{name} contains NaN/Inf.")


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


def _parse_pose_mode(token_feat_cfg) -> str:
    pose_mode = str(_cfg_get(token_feat_cfg, "pose_mode", "dense_local")).strip().lower()
    if pose_mode not in {"dense_local", "global_theta", "none"}:
        raise RuntimeError(f"Invalid litept.token_feat.pose_mode='{pose_mode}'. Expected dense_local/global_theta/none.")
    return pose_mode


def _parse_xyz_encoding(token_feat_cfg) -> str:
    enc = str(_cfg_get(token_feat_cfg, "token_xyz_encoding", "raw")).strip().lower()
    if enc not in {"raw", "pe"}:
        raise RuntimeError(f"Invalid litept.token_feat.token_xyz_encoding='{enc}'. Expected raw/pe.")
    return enc


def _xyz_pe_encode(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    if int(num_freqs) <= 0:
        return x
    out = [x]
    freq_bands = 2.0 ** torch.arange(int(num_freqs), device=x.device, dtype=x.dtype)
    for f in freq_bands:
        out.append(torch.sin(x * f))
        out.append(torch.cos(x * f))
    return torch.cat(out, dim=-1)


def infer_litept_token_feat_dim(*, token_feat_cfg=None, global_theta_out_dim: int = 16) -> int:
    use_token_xyz_in_feat = bool(_cfg_get(token_feat_cfg, "use_token_xyz_in_feat", True))
    use_token_normal = bool(_cfg_get(token_feat_cfg, "use_token_normal", True))
    use_body_normal = bool(_cfg_get(token_feat_cfg, "use_body_normal", True))
    use_offset_local = bool(_cfg_get(token_feat_cfg, "use_offset_local", True))
    token_xyz_encoding = _parse_xyz_encoding(token_feat_cfg)
    token_xyz_pe_L = int(_cfg_get(token_feat_cfg, "token_xyz_pe_L", 6))
    pose_mode = _parse_pose_mode(token_feat_cfg)
    d = 0
    if use_token_xyz_in_feat:
        d += (3 * (1 + 2 * token_xyz_pe_L)) if token_xyz_encoding == "pe" else 3
    if use_token_normal:
        d += 3
    if use_body_normal:
        d += 3
    if use_offset_local:
        d += 3
    if pose_mode == "dense_local":
        d += 6
    elif pose_mode == "global_theta":
        d += int(global_theta_out_dim)
    if d <= 0:
        raise RuntimeError("LitePT token feat dim must be > 0. Enable at least one token feature block.")
    return d


def build_litept_fullbody_tokens(
    *,
    x_canon: torch.Tensor,
    n_canon: torch.Tensor | None,
    anchor_cache: dict | None,
    b_pose: torch.Tensor | None,
    n_b_pose: torch.Tensor | None,
    token_feat_cfg=None,
    pose_global_feat: torch.Tensor | None = None,
    recon_p95_tol: float = 1e-3,
) -> Dict[str, torch.Tensor | dict]:
    """
    Build LitePT token features with configurable ablation blocks.
    Default reproduces Task-2 18D:
      [x_canon, n_canon, n_b_canon, o_tb, pose_disp_local, pose_normal_local]
    """
    if x_canon.ndim != 2 or x_canon.shape[-1] != 3:
        raise RuntimeError(f"x_canon must be [Nv,3], got {tuple(x_canon.shape)}")

    nv = int(x_canon.shape[0])
    x_canon_f = x_canon.to(torch.float32)
    use_token_xyz_in_feat = bool(_cfg_get(token_feat_cfg, "use_token_xyz_in_feat", True))
    use_token_normal = bool(_cfg_get(token_feat_cfg, "use_token_normal", True))
    use_body_normal = bool(_cfg_get(token_feat_cfg, "use_body_normal", True))
    use_offset_local = bool(_cfg_get(token_feat_cfg, "use_offset_local", True))
    token_xyz_encoding = _parse_xyz_encoding(token_feat_cfg)
    token_xyz_pe_L = int(_cfg_get(token_feat_cfg, "token_xyz_pe_L", 6))
    pose_mode = _parse_pose_mode(token_feat_cfg)

    # Local-frame terms require anchor cache.
    need_local_frame = bool(use_body_normal or use_offset_local or pose_mode == "dense_local")
    q = None
    r_local = None
    R_ntb = None
    recon_err = None
    recon_p95 = None
    if need_local_frame:
        q = _require_anchor_key(anchor_cache, "q", nv, (3,)).to(x_canon.device, torch.float32)
        r_local = _require_anchor_key(anchor_cache, "r_local", nv, (3,)).to(x_canon.device, torch.float32)
        R_ntb = _require_anchor_key(anchor_cache, "R_ntb", nv, (3, 3)).to(x_canon.device, torch.float32)
        _ = _require_anchor_key(anchor_cache, "face_idx", nv, ())
        _ = _require_anchor_key(anchor_cache, "bary", nv, (3,))
        _ = _require_anchor_key(anchor_cache, "tri_vidx", nv, (3,))

        # High-priority convention check only when local-frame path is used.
        x_recon = q + torch.bmm(R_ntb, r_local.unsqueeze(-1)).squeeze(-1)
        recon_err = torch.norm(x_recon - x_canon_f, dim=-1)
        recon_p95 = torch.quantile(recon_err, 0.95)
        if float(recon_p95.item()) > float(recon_p95_tol):
            alt = q + torch.bmm(R_ntb.transpose(1, 2), r_local.unsqueeze(-1)).squeeze(-1)
            alt_err = torch.norm(alt - x_canon_f, dim=-1)
            alt_p95 = torch.quantile(alt_err, 0.95)
            raise RuntimeError(
                "R_ntb/r_local convention check failed. "
                f"p95 ||q + R@r - x||={float(recon_p95.item()):.6g}, "
                f"alt p95 ||q + R^T@r - x||={float(alt_p95.item()):.6g}. "
                "Do not proceed until axis/row-column convention is fixed."
            )

    active_blocks = []
    blocks = []

    if use_token_xyz_in_feat:
        if token_xyz_encoding == "pe":
            blocks.append(_xyz_pe_encode(x_canon_f, token_xyz_pe_L))
            active_blocks.append(f"x_t_canon_peL{token_xyz_pe_L}")
        else:
            blocks.append(x_canon_f)
            active_blocks.append("x_t_canon")

    if use_token_normal:
        if n_canon is None:
            raise RuntimeError("litept.token_feat.use_token_normal=true but n_canon is missing.")
        if n_canon.ndim != 2 or n_canon.shape != x_canon.shape:
            raise RuntimeError(f"n_canon must be [Nv,3], got {tuple(n_canon.shape)}")
        n_canon_f = F.normalize(n_canon.to(torch.float32), dim=-1)
        blocks.append(n_canon_f)
        active_blocks.append("n_t_canon")

    if use_body_normal:
        if R_ntb is None:
            raise RuntimeError("litept.token_feat.use_body_normal=true but local-frame cache is unavailable.")
        n_b_canon = F.normalize(R_ntb[:, :, 0], dim=-1)
        blocks.append(n_b_canon)
        active_blocks.append("n_b_canon")

    if use_offset_local:
        if r_local is None:
            raise RuntimeError("litept.token_feat.use_offset_local=true but local-frame cache is unavailable.")
        blocks.append(r_local)
        active_blocks.append("o_tb")

    pose_disp_local = None
    pose_normal_local = None
    if pose_mode == "dense_local":
        if R_ntb is None or q is None:
            raise RuntimeError("pose_mode=dense_local requires local-frame cache (R_ntb,q).")
        if b_pose is None or n_b_pose is None:
            raise RuntimeError("pose_mode=dense_local requires b_pose and n_b_pose.")
        if b_pose.shape != x_canon.shape:
            raise RuntimeError(f"b_pose shape mismatch: got {tuple(b_pose.shape)}, expected {tuple(x_canon.shape)}")
        if n_b_pose.shape != x_canon.shape:
            raise RuntimeError(f"n_b_pose shape mismatch: got {tuple(n_b_pose.shape)}, expected {tuple(x_canon.shape)}")
        b_pose_f = b_pose.to(torch.float32)
        n_b_pose_f = F.normalize(n_b_pose.to(torch.float32), dim=-1)
        pose_disp_local = torch.bmm(R_ntb.transpose(1, 2), (b_pose_f - q).unsqueeze(-1)).squeeze(-1)
        pose_normal_local = torch.bmm(R_ntb.transpose(1, 2), n_b_pose_f.unsqueeze(-1)).squeeze(-1)
        blocks.extend([pose_disp_local, pose_normal_local])
        active_blocks.extend(["pose_disp_local", "pose_normal_local"])
    elif pose_mode == "global_theta":
        if pose_global_feat is None:
            raise RuntimeError("pose_mode=global_theta requires pose_global_feat.")
        if pose_global_feat.ndim == 1:
            pose_global_feat = pose_global_feat.view(1, -1).expand(nv, -1)
        elif pose_global_feat.ndim == 2 and pose_global_feat.shape[0] == 1:
            pose_global_feat = pose_global_feat.expand(nv, -1)
        elif pose_global_feat.ndim != 2 or pose_global_feat.shape[0] != nv:
            raise RuntimeError(
                f"pose_global_feat shape mismatch for global_theta: got {tuple(pose_global_feat.shape)}, expected ({nv}, D)"
            )
        pose_global_feat = pose_global_feat.to(x_canon.device, torch.float32)
        blocks.append(pose_global_feat)
        active_blocks.append("pose_global_theta")
    else:
        # pose_mode == "none"
        pass

    coord = x_canon_f
    if len(blocks) == 0:
        raise RuntimeError("LitePT token feature blocks are empty. Enable at least one token feature.")
    feat = torch.cat(blocks, dim=-1)

    _validate_finite("coord", coord)
    _validate_finite("feat", feat)

    return {
        "coord": coord,
        "feat": feat,
        "pose_disp_local": pose_disp_local,
        "pose_normal_local": pose_normal_local,
        "debug": {
            "active_blocks": active_blocks,
            "feat_dim": int(feat.shape[-1]),
            "pose_mode": pose_mode,
            "use_token_xyz_in_feat": bool(use_token_xyz_in_feat),
            "token_xyz_encoding": token_xyz_encoding,
            "token_xyz_pe_L": int(token_xyz_pe_L),
            "recon_err_mean": (recon_err.mean().detach() if recon_err is not None else torch.tensor(0.0, device=x_canon.device)),
            "recon_err_p95": (recon_p95.detach() if recon_p95 is not None else torch.tensor(0.0, device=x_canon.device)),
            "recon_err_max": (recon_err.max().detach() if recon_err is not None else torch.tensor(0.0, device=x_canon.device)),
        },
    }

