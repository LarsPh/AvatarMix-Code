# The base class for GaussianSplatting Loss.
# Contributer(s): Neil Z. Shao
# All rights reserved. Prometheus 2022-2024.
import os
import cv2
import numpy as np
import json
import torch
import torch.nn.functional as thf
from utils.loss_utils import l1_loss, ssim, LPIPS, erode_mask, gaussian_blur_2d
from utils.image_utils import psnr
from utils.metrics import img_mse, img_ssim, img_psnr, perceptual
from dataset.dataset_helper import make_dataloader
from gaussian_renderer import network_gui
from tqdm import tqdm

def sobel_grad_diff_vis(
    *,
    image: torch.Tensor,         # [3,H,W] in [0,1]
    gt_image: torch.Tensor,      # [3,H,W] in [0,1]
    gt_alpha_mask: torch.Tensor | None,  # [1,H,W] or None
    optimizer_config,
) -> torch.Tensor | None:
    """Compute a normalized Sobel gradient-diff heatmap as a 3xHxW tensor in [0,1].

    This is meant for visualization during eval/validation without needing a LossBase instance.
    Respects `optimizer_config.ssim_masking` erosion when gt_alpha_mask is available.
    Returns None if lambda_grad <= 0.
    """
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

    # Prefer new schema: optim.loss.grad.* ; also support optim.grad.* ; fallback to legacy optim.lambda_grad.
    grad_sub = None
    loss_root = _cfg_get(optimizer_config, "loss", None)
    if loss_root is not None:
        grad_sub = _cfg_get(loss_root, "grad", None)
    if grad_sub is None:
        grad_sub = _cfg_get(optimizer_config, "grad", None)

    enabled = _cfg_get(grad_sub, "enabled", None) if grad_sub is not None else None
    lam = _cfg_get(grad_sub, "lambda", None) if grad_sub is not None else None
    if lam is None:
        lam = _cfg_get(optimizer_config, "lambda_grad", 0.0)
    lambda_grad = float(lam)
    if enabled is None:
        enabled = lambda_grad > 0
    if (not bool(enabled)) or (lambda_grad <= 0):
        return None

    device = image.device
    dtype = image.dtype

    mask_e = None
    if gt_alpha_mask is not None:
        ssim_masking_cfg = getattr(optimizer_config, "ssim_masking", {}) if hasattr(optimizer_config, "ssim_masking") else optimizer_config.get("ssim_masking", {})
        ssim_masking_enabled = bool(getattr(ssim_masking_cfg, "enabled", ssim_masking_cfg.get("enabled", False))) if isinstance(ssim_masking_cfg, dict) or hasattr(ssim_masking_cfg, "get") else False
        if ssim_masking_enabled:
            alpha_thresh = float(getattr(ssim_masking_cfg, "alpha_thresh", ssim_masking_cfg.get("alpha_thresh", 0.5)))
            erode_ksize = int(getattr(ssim_masking_cfg, "erode_ksize", ssim_masking_cfg.get("erode_ksize", 15)))
            erode_iters = int(getattr(ssim_masking_cfg, "erode_iters", ssim_masking_cfg.get("erode_iters", 1)))
            m = (gt_alpha_mask > alpha_thresh).float()
            mask_e = erode_mask(m, ksize=erode_ksize, iters=erode_iters)  # [1,H,W]
        else:
            mask_e = (gt_alpha_mask > 0.5).float()

    # grayscale
    if image.shape[0] >= 3:
        gray_pred = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0).unsqueeze(0)
        gray_gt = (0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]).unsqueeze(0).unsqueeze(0)
    else:
        gray_pred = image.mean(dim=0, keepdim=True).unsqueeze(0)
        gray_gt = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)

    kx = torch.tensor([[-1.0, 0.0, 1.0],
                       [-2.0, 0.0, 2.0],
                       [-1.0, 0.0, 1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)
    ky = torch.tensor([[-1.0, -2.0, -1.0],
                       [ 0.0,  0.0,  0.0],
                       [ 1.0,  2.0,  1.0]], device=device, dtype=dtype).view(1, 1, 3, 3)

    gx_p = thf.conv2d(gray_pred, kx, padding=1)
    gy_p = thf.conv2d(gray_pred, ky, padding=1)
    gx_g = thf.conv2d(gray_gt, kx, padding=1)
    gy_g = thf.conv2d(gray_gt, ky, padding=1)

    diff = (gx_p - gx_g).abs() + (gy_p - gy_g).abs()  # [1,1,H,W]
    if mask_e is not None:
        diff = diff * mask_e.unsqueeze(0)
    v = diff.detach().squeeze(0).squeeze(0)
    denom = v.max().clamp_min(1e-6)
    h = (v / denom).clamp(0.0, 1.0)
    return h.unsqueeze(0).repeat(3, 1, 1)

class LossBase(torch.nn.Module):
    def __init__(self, gs_model, optimizer_config) -> None:
        super().__init__()
        self.gs_model = gs_model
        self.optimizer_config = optimizer_config
        self.lpips = LPIPS(eval=False).cuda()
        # TODO-3 temporal cache for 3-micro-step window losses (DoD + coeff smooth)
        # Structure: {window_id: {"prev": entry, "cur": entry}}
        self.temporal_cache = {}

    @staticmethod
    def _compute_changed_region_mask(
        gt_orig: torch.Tensor,      # [3, H, W]
        gt_refined: torch.Tensor,   # [3, H, W]
        alpha: torch.Tensor,        # [1, H, W] in [0,1]
        metric: str = "l1",
        blur_ksize: int = 7,
        pool: int = 16,
        threshold: float = 0.04,
        dilate_ksize: int = 15,
    ) -> torch.Tensor:
        """Compute a changed-region mask M in [0,1], shape [1,H,W].

        Designed to capture both:
        - small, localized changes (holes/seams)
        - large smooth edits (e.g., whole-arm denoising) via pooling.
        """
        if gt_orig.shape != gt_refined.shape:
            raise ValueError(f"gt_orig and gt_refined shape mismatch: {gt_orig.shape} vs {gt_refined.shape}")
        if alpha.ndim != 3 or alpha.shape[0] != 1:
            raise ValueError(f"alpha must be [1,H,W], got: {alpha.shape}")

        # Foreground gating (avoid random background affecting diff)
        alpha_bin = (alpha > 0.5).float()

        # IMPORTANT: build the change mask on a low-frequency signal to reduce:
        # - false positives from resolution mismatch (orig high-res vs refined upsample)
        # - false positives from texture/style edits by DiFix (even when no artifacts)
        #
        # We do this by first low-pass filtering BOTH images, then computing diff,
        # then applying pooling/dilation for large-region capture.
        k = int(blur_ksize) if blur_ksize else 0
        if k and k > 1:
            if k % 2 == 0:
                k += 1
            # [1,3,H,W] -> low-frequency images
            o = gt_orig.unsqueeze(0)
            r = gt_refined.unsqueeze(0)
            o = thf.avg_pool2d(o, kernel_size=k, stride=1, padding=k // 2)
            r = thf.avg_pool2d(r, kernel_size=k, stride=1, padding=k // 2)
            gt_orig_lf = o.squeeze(0)
            gt_refined_lf = r.squeeze(0)
        else:
            gt_orig_lf = gt_orig
            gt_refined_lf = gt_refined

        if metric == "l2":
            diff = (gt_refined_lf - gt_orig_lf).pow(2).mean(dim=0, keepdim=True).sqrt()
        else:
            diff = (gt_refined_lf - gt_orig_lf).abs().mean(dim=0, keepdim=True)
        diff = diff * alpha_bin

        x = diff.unsqueeze(0)  # [1,1,H,W]

        # Pool to capture large changed regions even if patchy
        if pool and pool > 1:
            p = int(pool)
            pooled = thf.avg_pool2d(x, kernel_size=p, stride=p, padding=0)
            x = thf.interpolate(pooled, size=diff.shape[-2:], mode="bilinear", align_corners=False)

        m = (x > float(threshold)).float()

        # Dilate to cover boundaries / holes neighborhood
        if dilate_ksize and dilate_ksize > 1:
            d = int(dilate_ksize)
            if d % 2 == 0:
                d += 1
            m = thf.max_pool2d(m, kernel_size=d, stride=1, padding=d // 2)

        # Ensure background stays 0
        m = (m.squeeze(0) * alpha_bin).clamp(0.0, 1.0)
        return m

    def collect_loss(self, gt_image, image, gt_alpha_mask=None, **kwargs):
        def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
            return torch.sqrt(x * x + (eps * eps))

        def _cfg_get(cfg, key: str, default=None):
            """OmegaConf-safe getter for dict-like or attribute-like config."""
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

        def _loss_cfg(name: str):
            """Return (subcfg, legacy_lambda_key)."""
            loss_root = _cfg_get(self.optimizer_config, "loss", None)
            sub = _cfg_get(loss_root, name, None)
            # Also support `optim.<name>` (e.g. `optim.grad`, `optim.dod`, `optim.lpips`) for convenience.
            if sub is None:
                sub = _cfg_get(self.optimizer_config, name, None)
            legacy = {
                "grad": "lambda_grad",
                "dod": "lambda_dod",
                "lpips": "lambda_perceptual",
                "delta_u_reg": "lambda_delta_u",
            }.get(name, None)
            return sub, legacy

        def _luma_3chw(x3: torch.Tensor) -> torch.Tensor:
            # x3: [3,H,W] in [0,1] -> [1,H,W]
            if x3.shape[0] >= 3:
                return (0.299 * x3[0] + 0.587 * x3[1] + 0.114 * x3[2]).unsqueeze(0)
            return x3.mean(dim=0, keepdim=True)

        def _compose_fixed_gray(x3: torch.Tensor, mask_e_1hw: torch.Tensor, gray: float = 0.5) -> torch.Tensor:
            # x3: [3,H,W], mask: [1,H,W] in [0,1]
            g = float(gray)
            bg = torch.full_like(x3, g)
            m3 = mask_e_1hw.repeat(x3.shape[0], 1, 1) if x3.shape[0] > 1 else mask_e_1hw
            return x3 * m3 + bg * (1.0 - m3)

        # ---- New config schema (preferred): optim.loss.* with legacy fallbacks ----
        grad_sub, grad_legacy = _loss_cfg("grad")
        dod_sub, dod_legacy = _loss_cfg("dod")
        lpips_sub, lpips_legacy = _loss_cfg("lpips")
        du_sub, du_legacy = _loss_cfg("delta_u_reg")

        def _enabled_and_lambda(subcfg, legacy_key: str | None, default_lambda: float = 0.0):
            # lambda precedence: subcfg.lambda -> legacy -> default
            lam = None
            if subcfg is not None:
                lam = _cfg_get(subcfg, "lambda", None)
            if lam is None and legacy_key is not None:
                lam = _cfg_get(self.optimizer_config, legacy_key, None)
            if lam is None:
                lam = default_lambda
            lam = float(lam)
            en = _cfg_get(subcfg, "enabled", None) if subcfg is not None else None
            if en is None:
                en = lam > 0
            en = bool(en)
            return en, (lam if en else 0.0)

        grad_enabled, lambda_grad = _enabled_and_lambda(grad_sub, grad_legacy, 0.0)
        dod_enabled, lambda_dod = _enabled_and_lambda(dod_sub, dod_legacy, 0.0)
        lpips_enabled, lambda_lpips = _enabled_and_lambda(lpips_sub, lpips_legacy, 0.0)
        du_enabled, lambda_delta_u = _enabled_and_lambda(du_sub, du_legacy, 0.0)

        def _build_mask_e(alpha: torch.Tensor) -> torch.Tensor:
            # alpha: [1,H,W] in [0,1]
            ssim_masking_cfg = (
                getattr(self.optimizer_config, "ssim_masking", {})
                if hasattr(self.optimizer_config, "ssim_masking")
                else self.optimizer_config.get("ssim_masking", {})
            )
            ssim_masking_enabled = bool(
                getattr(ssim_masking_cfg, "enabled", ssim_masking_cfg.get("enabled", False))
            ) if isinstance(ssim_masking_cfg, dict) or hasattr(ssim_masking_cfg, "get") else False

            if ssim_masking_enabled:
                alpha_thresh = float(getattr(ssim_masking_cfg, "alpha_thresh", ssim_masking_cfg.get("alpha_thresh", 0.5)))
                erode_ksize = int(getattr(ssim_masking_cfg, "erode_ksize", ssim_masking_cfg.get("erode_ksize", 15)))
                erode_iters = int(getattr(ssim_masking_cfg, "erode_iters", ssim_masking_cfg.get("erode_iters", 1)))
                m = (alpha > alpha_thresh).float()
                m = erode_mask(m, ksize=erode_ksize, iters=erode_iters)  # [1,H,W]
                return m

            # Fallback: non-eroded binary alpha
            return (alpha > 0.5).float()

        def _maybe_get_coeff_vector() -> torch.Tensor | None:
            # Coeff vector for temporal smooth: concatenate pose-dependent blendshape weights.
            pos_w = getattr(self.gs_model, "_last_pos_blend_weights", None)
            col_w = getattr(self.gs_model, "_last_color_blend_weights", None)
            if isinstance(pos_w, torch.Tensor) and isinstance(col_w, torch.Tensor):
                try:
                    return torch.cat([pos_w.flatten(), col_w.flatten()], dim=0)
                except Exception:
                    return None
            if isinstance(pos_w, torch.Tensor):
                return pos_w.flatten()
            return None

        # Optionally mask both render and GT with the alpha mask before computing losses
        use_mask_for_all = self.optimizer_config.get('apply_alpha_mask_to_render', False) and gt_alpha_mask is not None
        if use_mask_for_all:
            masked_image = image * gt_alpha_mask
            masked_gt = gt_image * gt_alpha_mask
        else:
            masked_image = image
            masked_gt = gt_image

        # Hybrid supervision (optional): refined inside changed region, original outside
        hybrid_gt_orig = kwargs.get("hybrid_gt_image_orig", None)
        hybrid_gt_refined = kwargs.get("hybrid_gt_image_refined", None)
        hybrid_enabled = (hybrid_gt_orig is not None) and (hybrid_gt_refined is not None)

        if hybrid_enabled:
            # Use alpha to define valid/foreground region (even though images are BG-composited)
            if gt_alpha_mask is None:
                alpha = torch.ones((1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
            else:
                alpha = gt_alpha_mask
            alpha_bin = (alpha > 0.5).float()

            M = self._compute_changed_region_mask(
                gt_orig=hybrid_gt_orig,
                gt_refined=hybrid_gt_refined,
                alpha=alpha,
                metric=str(kwargs.get("hybrid_changed_metric", "l1")),
                blur_ksize=int(kwargs.get("hybrid_changed_blur_ksize", 7)),
                pool=int(kwargs.get("hybrid_changed_pool", 16)),
                threshold=float(kwargs.get("hybrid_changed_threshold", 0.04)),
                dilate_ksize=int(kwargs.get("hybrid_changed_dilate_ksize", 15)),
            )  # [1,H,W]

            inside = M * alpha_bin
            outside = (1.0 - M) * alpha_bin

            eps = 1e-6
            c = float(image.shape[0])

            # L1 hybrid
            l1_in = ((image - hybrid_gt_refined).abs() * inside).sum() / (inside.sum() * c + eps)
            l1_out = ((image - hybrid_gt_orig).abs() * outside).sum() / (outside.sum() * c + eps)
            w_in = float(kwargs.get("lambda_refined_inside", 1.0))
            w_out = float(kwargs.get("lambda_orig_outside", 1.0))
            loss = w_in * l1_in + w_out * l1_out

            # Optional L2 hybrid (respects existing lambda_rgb_mse)
            if self.optimizer_config.get('lambda_rgb_mse', 0) > 0:
                mse_in = ((image - hybrid_gt_refined).pow(2) * inside).sum() / (inside.sum() * c + eps)
                mse_out = ((image - hybrid_gt_orig).pow(2) * outside).sum() / (outside.sum() * c + eps)
                loss += self.optimizer_config.lambda_rgb_mse * (w_in * mse_in + w_out * mse_out)

            # LPIPS only outside changed region (per your request)
            if self.optimizer_config.get('lambda_perceptual', 0) > 0:
                if outside.sum() > 0:
                    mask_lp = outside
                    Llpips = self.lpips(hybrid_gt_orig * mask_lp, image * mask_lp).squeeze()
                    loss += self.optimizer_config.lambda_perceptual * float(kwargs.get("lambda_lpips_outside", 1.0)) * Llpips

            # Continue with regularizers
            if self.optimizer_config.get('lambda_sparsity', 0) > 0:
                loss += self.optimizer_config.lambda_sparsity * self.gs_model.get_opacity.mean()

            if self.optimizer_config.get('lambda_scaling', 0) > 0:
                thresh_scaling_max = self.optimizer_config.get('thresh_scaling_max', 0.008)
                thresh_scaling_ratio = self.optimizer_config.get('thresh_scaling_ratio', 10.0)
                max_vals = self.gs_model.get_scaling.max(dim=-1).values
                min_vals = self.gs_model.get_scaling.min(dim=-1).values
                ratio = max_vals / min_vals
                thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
                if thresh_idxs.sum() > 0:
                    loss += self.optimizer_config.lambda_scaling * max_vals[thresh_idxs].mean()

            psnr_full = psnr(image, gt_image).mean().float().item()
            out = {'loss': loss, 'psnr_full': psnr_full}
            if kwargs.get("hybrid_debug", False):
                out.update({
                    "hybrid_gt_orig_vis": hybrid_gt_orig.detach(),
                    "hybrid_gt_refined_vis": hybrid_gt_refined.detach(),
                    "hybrid_changed_mask_vis": M.detach(),
                })
            return out

        # ------------------------------------------------------------
        # FullbodyFix bake micro-tune: fixed GS set + inner-only masking
        # ------------------------------------------------------------
        fbf_cfg = _cfg_get(self.optimizer_config, "fullbodyfix_bake", None)
        fbf_enabled = bool(_cfg_get(fbf_cfg, "enabled", False))
        if fbf_enabled:
            mask_cfg = _cfg_get(fbf_cfg, "mask", None)
            loss_cfg = _cfg_get(fbf_cfg, "loss", None)
            bg_cfg = _cfg_get(fbf_cfg, "background", None)

            # Resolve silhouette source: in current pipeline this is typically `gt_alpha_mask`
            # loaded from `--custom_mask_relpath` (render-derived mask/pha).
            if gt_alpha_mask is None:
                alpha = torch.ones((1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)
            else:
                alpha = gt_alpha_mask
                if alpha.ndim == 2:
                    alpha = alpha.unsqueeze(0)
                if alpha.shape[0] != 1:
                    alpha = alpha[:1]

            alpha_thresh = float(_cfg_get(mask_cfg, "alpha_thresh", 0.5))
            m0 = (alpha > alpha_thresh).float()  # [1,H,W]

            # Morphology helpers (torch-only; deterministic; GPU-friendly).
            def _odd(k: int) -> int:
                k = int(k)
                if k <= 1:
                    return 1
                return (k + 1) if (k % 2 == 0) else k

            def _dilate_mask(m_1hw: torch.Tensor, ksize: int, iters: int = 1) -> torch.Tensor:
                k = _odd(ksize)
                if iters <= 0 or k <= 1:
                    return m_1hw.float()
                x = m_1hw.unsqueeze(0)  # [1,1,H,W]
                for _ in range(int(iters)):
                    x = thf.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
                return x.squeeze(0).clamp(0.0, 1.0)

            erode_ksize = _odd(int(_cfg_get(mask_cfg, "erode_ksize", 15)))
            erode_iters = int(_cfg_get(mask_cfg, "erode_iters", 1))
            dilate_ksize = _odd(int(_cfg_get(mask_cfg, "dilate_ksize", 15)))
            dilate_iters = int(_cfg_get(mask_cfg, "dilate_iters", 1))

            inner = erode_mask(m0, ksize=erode_ksize, iters=erode_iters).clamp(0.0, 1.0)  # [1,H,W]
            outer = _dilate_mask(m0, ksize=dilate_ksize, iters=dilate_iters).clamp(0.0, 1.0)  # [1,H,W]

            band = ((outer > 0.5) & (inner <= 0.5)).float()
            outside = (outer <= 0.5).float()

            inner_w = float(_cfg_get(mask_cfg, "inner_weight", 1.0))
            band_w = float(_cfg_get(mask_cfg, "band_weight", 0.0))
            outside_w = float(_cfg_get(mask_cfg, "outside_weight", 0.0))
            W = (inner_w * inner + band_w * band + outside_w * outside)  # [1,H,W]

            # Extra erosion margin for patch/window losses (LPIPS/SSIM) to reduce receptive-field leakage.
            extra_px = int(_cfg_get(mask_cfg, "extra_inner_erode_for_patch_losses", 0) or 0)
            if extra_px > 0:
                k_extra = _odd(2 * extra_px + 1)
                inner_patch = erode_mask(inner, ksize=k_extra, iters=1).clamp(0.0, 1.0)
            else:
                inner_patch = inner

            # Deterministic fill color for masked pred/GT in patch/window losses.
            fill_rgb = _cfg_get(bg_cfg, "fill_rgb", [0.5, 0.5, 0.5])
            try:
                fr = torch.tensor(fill_rgb, device=image.device, dtype=image.dtype).view(3, 1, 1)
            except Exception:
                fr = torch.tensor([0.5, 0.5, 0.5], device=image.device, dtype=image.dtype).view(3, 1, 1)

            def _fill_outside_const(x3: torch.Tensor, m_1hw: torch.Tensor) -> torch.Tensor:
                # x3: [3,H,W], m: [1,H,W]
                m3 = m_1hw.repeat(x3.shape[0], 1, 1) if x3.shape[0] > 1 else m_1hw
                return x3 * m3 + fr * (1.0 - m3)

            eps = 1e-6
            c = float(image.shape[0])

            def _weighted_l1(img: torch.Tensor, gt: torch.Tensor, w_1hw: torch.Tensor) -> torch.Tensor:
                w3 = w_1hw.repeat(img.shape[0], 1, 1) if img.shape[0] > 1 else w_1hw
                return ((img - gt).abs() * w3).sum() / (w_1hw.sum() * c + eps)

            def _weighted_mse(img: torch.Tensor, gt: torch.Tensor, w_1hw: torch.Tensor) -> torch.Tensor:
                w3 = w_1hw.repeat(img.shape[0], 1, 1) if img.shape[0] > 1 else w_1hw
                return ((img - gt).pow(2) * w3).sum() / (w_1hw.sum() * c + eps)

            # Resolve loss enables/weights (fall back to legacy lambdas if needed).
            l1_sub = _cfg_get(loss_cfg, "rgb_l1", None)
            lp_sub = _cfg_get(loss_cfg, "lpips", None)
            mse_sub = _cfg_get(loss_cfg, "mse", None)
            ssim_sub = _cfg_get(loss_cfg, "ssim", None)

            l1_enabled = bool(_cfg_get(l1_sub, "enabled", True)) if l1_sub is not None else True
            w_l1 = float(_cfg_get(l1_sub, "weight", _cfg_get(self.optimizer_config, "lambda_l1", 1.0))) if l1_enabled else 0.0

            lp_enabled = bool(_cfg_get(lp_sub, "enabled", False)) if lp_sub is not None else False
            w_lp = float(_cfg_get(lp_sub, "weight", _cfg_get(self.optimizer_config, "lambda_perceptual", 0.0))) if lp_enabled else 0.0

            mse_enabled = bool(_cfg_get(mse_sub, "enabled", False)) if mse_sub is not None else False
            w_mse = float(_cfg_get(mse_sub, "weight", _cfg_get(self.optimizer_config, "lambda_rgb_mse", 0.0))) if mse_enabled else 0.0

            ssim_enabled = bool(_cfg_get(ssim_sub, "enabled", False)) if ssim_sub is not None else False
            w_ssim = float(_cfg_get(ssim_sub, "weight", _cfg_get(self.optimizer_config, "lambda_ssim", 0.0))) if ssim_enabled else 0.0

            loss = torch.zeros((), device=image.device, dtype=image.dtype)

            if w_l1 > 0 and l1_enabled:
                loss = loss + float(w_l1) * _weighted_l1(image, gt_image, W)

            if w_mse > 0 and mse_enabled:
                loss = loss + float(w_mse) * _weighted_mse(image, gt_image, W)

            if w_lp > 0 and lp_enabled:
                # Mask both pred and GT by filling outside with constant RGB.
                pred_lp = _fill_outside_const(image, inner_patch)
                gt_lp = _fill_outside_const(gt_image, inner_patch)
                Llpips = self.lpips(gt_lp.unsqueeze(0), pred_lp.unsqueeze(0)).squeeze()
                loss = loss + float(w_lp) * Llpips

            if w_ssim > 0 and ssim_enabled:
                # Mask both pred and GT by filling outside with constant RGB.
                pred_s = _fill_outside_const(image, inner_patch)
                gt_s = _fill_outside_const(gt_image, inner_patch)
                Lssim = 1.0 - ssim(pred_s.unsqueeze(0), gt_s.unsqueeze(0))
                loss = loss + float(w_ssim) * Lssim

            psnr_full = psnr(image, gt_image).mean().float().item()
            out = {
                "loss": loss,
                "psnr_full": psnr_full,
            }
            if bool(kwargs.get("debug_vis", False)):
                out.update({
                    "fbf_mask_inner_vis": inner.detach(),
                    "fbf_mask_outer_vis": outer.detach(),
                    "fbf_mask_band_vis": band.detach(),
                    "fbf_mask_W_vis": W.detach(),
                    "fbf_mask_inner_patch_vis": inner_patch.detach(),
                })
            return out

        Ll1 = l1_loss(masked_image, masked_gt)

        # Base L1 weight (independent of SSIM).
        lambda_l1 = float(self.optimizer_config.get('lambda_l1', 1.0))
        loss = lambda_l1 * Ll1

        # --- TODO-1: gradient loss (Sobel) ---
        grad_diff_vis = None
        loss_grad = None
        if lambda_grad > 0 and grad_enabled:
            # Mask for gradient loss: use the same eroded mask as SSIM masking when available.
            if gt_alpha_mask is not None and (not use_mask_for_all):
                mask_e = _build_mask_e(gt_alpha_mask)  # [1,H,W]
            elif gt_alpha_mask is not None and use_mask_for_all:
                # If user masks all losses, keep gradients consistent with that mask (no erosion here).
                mask_e = (gt_alpha_mask > 0.5).float()
            else:
                mask_e = None

            grad_use_luma = bool(_cfg_get(grad_sub, "use_luma", True))
            grad_char_eps = float(_cfg_get(grad_sub, "charbonnier_eps", 0.0) or 0.0)

            if grad_use_luma:
                # grayscale gradients
                if image.shape[0] >= 3:
                    pred_in = (0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                    gt_in = (0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]).unsqueeze(0).unsqueeze(0)
                else:
                    pred_in = image.mean(dim=0, keepdim=True).unsqueeze(0)  # [1,1,H,W]
                    gt_in = gt_image.mean(dim=0, keepdim=True).unsqueeze(0)
                groups = 1
                n_ch = 1
            else:
                # Per-channel gradients, averaged (RGB)
                pred_in = image.unsqueeze(0)   # [1,3,H,W]
                gt_in = gt_image.unsqueeze(0)  # [1,3,H,W]
                groups = pred_in.shape[1]
                n_ch = groups

            # Sobel kernels
            kx = torch.tensor([[-1.0, 0.0, 1.0],
                               [-2.0, 0.0, 2.0],
                               [-1.0, 0.0, 1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            ky = torch.tensor([[-1.0, -2.0, -1.0],
                               [ 0.0,  0.0,  0.0],
                               [ 1.0,  2.0,  1.0]], device=image.device, dtype=image.dtype).view(1, 1, 3, 3)
            if n_ch > 1:
                kx = kx.repeat(n_ch, 1, 1, 1)  # [C,1,3,3]
                ky = ky.repeat(n_ch, 1, 1, 1)

            gx_p = thf.conv2d(pred_in, kx, padding=1, groups=groups)
            gy_p = thf.conv2d(pred_in, ky, padding=1, groups=groups)
            gx_g = thf.conv2d(gt_in, kx, padding=1, groups=groups)
            gy_g = thf.conv2d(gt_in, ky, padding=1, groups=groups)

            diff_raw = (gx_p - gx_g).abs() + (gy_p - gy_g).abs()  # [1,C,H,W] or [1,1,H,W]
            if diff_raw.shape[1] > 1:
                diff_raw = diff_raw.mean(dim=1, keepdim=True)  # -> [1,1,H,W]
            diff = _charbonnier(diff_raw, eps=grad_char_eps) if grad_char_eps and grad_char_eps > 0 else diff_raw

            if mask_e is not None:
                m = mask_e.unsqueeze(0)  # [1,1,H,W]
                eps = 1e-6
                loss_grad = (diff * m).sum() / (m.sum() + eps)
                diff_for_vis = (diff * m).detach()
            else:
                loss_grad = diff.mean()
                diff_for_vis = diff.detach()

            loss = loss + lambda_grad * loss_grad

            if bool(kwargs.get("debug_vis", False)):
                # Simple normalized heatmap for visualization.
                v = diff_for_vis.squeeze(0).squeeze(0)  # [H,W]
                denom = v.max().clamp_min(1e-6)
                h = (v / denom).clamp(0.0, 1.0)
                grad_diff_vis = h.unsqueeze(0).repeat(3, 1, 1)

        # Optional SSIM term.
        lambda_ssim = float(self.optimizer_config.get('lambda_ssim', 0.0))
        if lambda_ssim > 0:
            # Optional SSIM-only masking with erosion (to reduce silhouette artifacts from windowed SSIM).
            ssim_masking_cfg = getattr(self.optimizer_config, "ssim_masking", {}) if hasattr(self.optimizer_config, "ssim_masking") else self.optimizer_config.get("ssim_masking", {})
            ssim_masking_enabled = bool(getattr(ssim_masking_cfg, "enabled", ssim_masking_cfg.get("enabled", False))) if isinstance(ssim_masking_cfg, dict) or hasattr(ssim_masking_cfg, "get") else False

            if gt_alpha_mask is not None and ssim_masking_enabled and not use_mask_for_all:
                # Build eroded foreground mask.
                alpha = gt_alpha_mask
                alpha_thresh = float(getattr(ssim_masking_cfg, "alpha_thresh", ssim_masking_cfg.get("alpha_thresh", 0.5)))
                erode_ksize = int(getattr(ssim_masking_cfg, "erode_ksize", ssim_masking_cfg.get("erode_ksize", 15)))
                erode_iters = int(getattr(ssim_masking_cfg, "erode_iters", ssim_masking_cfg.get("erode_iters", 1)))

                m = (alpha > alpha_thresh).float()
                m = erode_mask(m, ksize=erode_ksize, iters=erode_iters)  # [1,H,W]
                # Broadcast to channels (e.g. 3,H,W)
                if image.shape[0] > 1:
                    m3 = m.repeat(image.shape[0], 1, 1)
                else:
                    m3 = m
                ssim_img = image * m3
                ssim_gt = gt_image * m3
            else:
                # Default: use the same masked/unmasked pair as L1.
                ssim_img = masked_image
                ssim_gt = masked_gt

            Lssim = 1.0 - ssim(ssim_img, ssim_gt)
            loss = loss + lambda_ssim * Lssim

        if self.optimizer_config.get('lambda_rgb_mse', 0) > 0:
            Ll2 = thf.mse_loss(masked_image, masked_gt)
            loss += self.optimizer_config.lambda_rgb_mse * Ll2

        # --- P1: joint-aware geometry regularization for Stage-B vertex offsets ---
        loss_joint_edge = None
        loss_joint_lap = None
        joint_reg_ramp = None
        try:
            joint_sub = _cfg_get(_cfg_get(self.optimizer_config, "loss", None), "joint_reg", None)
            if joint_sub is None:
                joint_sub = _cfg_get(self.optimizer_config, "joint_reg", None)
            joint_enabled = bool(_cfg_get(joint_sub, "enabled", False)) if joint_sub is not None else False
            # Stage1-only gate (per spec)
            step = getattr(self.gs_model, "global_step", None)
            stage1_end = int(_cfg_get(self.optimizer_config, "stage1_end_iter", 0) or 0)
            in_stage1 = (step is not None) and (stage1_end > 0) and (int(step) <= stage1_end)
            if joint_enabled and getattr(self.gs_model, "use_deformation", False) and in_stage1:
                dV = getattr(self.gs_model, "_current_vertex_offsets", None)
                V0 = getattr(self.gs_model, "cano_verts_orig_for_deform", None)
                F = getattr(self.gs_model, "cano_faces", None)
                if isinstance(dV, torch.Tensor) and isinstance(V0, torch.Tensor) and isinstance(F, torch.Tensor):
                    # Ramp schedule
                    ramp_cfg = _cfg_get(joint_sub, "ramp", None)
                    start_it = int(_cfg_get(ramp_cfg, "start_iter", 0) or 0)
                    end_it = int(_cfg_get(ramp_cfg, "end_iter", start_it) or start_it)
                    s = int(step)
                    if s < start_it:
                        joint_reg_ramp = 0.0
                    elif end_it <= start_it:
                        joint_reg_ramp = 1.0
                    elif s >= end_it:
                        joint_reg_ramp = 1.0
                    else:
                        joint_reg_ramp = float(s - start_it) / float(max(end_it - start_it, 1))

                    lam_edge = float(_cfg_get(joint_sub, "lambda_edge", 0.0) or 0.0)
                    lam_lap = float(_cfg_get(joint_sub, "lambda_lap", 0.0) or 0.0)
                    if joint_reg_ramp > 0 and (lam_edge > 0 or lam_lap > 0):
                        # Ensure model caches exist
                        try:
                            self.gs_model._ensure_joint_reg_geometry_caches()
                        except Exception:
                            pass
                        M = None
                        try:
                            M = self.gs_model.ensure_joint_reg_mask(joint_sub)
                        except Exception:
                            M = getattr(self.gs_model, "_joint_reg_mask_v", None)

                        E = getattr(self.gs_model, "_joint_reg_edges_e2", None)
                        e0 = getattr(self.gs_model, "_joint_reg_e0", None)
                        src = getattr(self.gs_model, "_joint_reg_lap_src", None)
                        dst = getattr(self.gs_model, "_joint_reg_lap_dst", None)
                        deg = getattr(self.gs_model, "_joint_reg_lap_deg", None)

                        if M is not None:
                            M = M.to(dV.device, dV.dtype)
                        V = V0.to(dV.device, dV.dtype) + dV

                        # Edge loss
                        if lam_edge > 0 and isinstance(E, torch.Tensor) and isinstance(e0, torch.Tensor) and M is not None:
                            Ei = E.to(dV.device)
                            e0t = e0.to(dV.device, dV.dtype)
                            vi = V[Ei[:, 0]]
                            vj = V[Ei[:, 1]]
                            e = (vi - vj).norm(dim=1)
                            # weights for edge based on joint mask
                            w_e = torch.maximum(M[Ei[:, 0]], M[Ei[:, 1]])
                            eps = 1e-12
                            denom = (e0t * e0t + eps)
                            valid = (e0t > 1.0e-9).to(dV.dtype)
                            loss_joint_edge = (w_e * valid * ((e - e0t) ** 2) / denom).sum() / (valid.sum().clamp_min(1.0))
                            loss = loss + float(joint_reg_ramp) * lam_edge * loss_joint_edge

                        # Laplacian on dV (B1)
                        if lam_lap > 0 and isinstance(src, torch.Tensor) and isinstance(dst, torch.Tensor) and isinstance(deg, torch.Tensor) and M is not None:
                            src_t = src.to(dV.device)
                            dst_t = dst.to(dV.device)
                            deg_t = deg.to(dV.device, dV.dtype).unsqueeze(1)  # (Nv,1)
                            sum_n = torch.zeros_like(dV)
                            sum_n.index_add_(0, src_t, dV[dst_t])
                            mean_n = sum_n / deg_t.clamp_min(1.0)
                            lap = dV - mean_n
                            loss_joint_lap = (M * (lap.pow(2).sum(dim=1))).mean()
                            loss = loss + float(joint_reg_ramp) * lam_lap * loss_joint_lap

                    # Populate out terms even if ramp=0 for logging
                    out["joint_reg_ramp"] = float(joint_reg_ramp) if joint_reg_ramp is not None else 0.0
                    if M is not None:
                        try:
                            out["joint_reg_M_mean"] = float(M.mean().detach().cpu().item())
                            if hasattr(torch, "quantile"):
                                out["joint_reg_M_p95"] = float(torch.quantile(M, 0.95).detach().cpu().item())
                        except Exception:
                            pass
        except Exception:
            pass

        # --- TODO-4: global learnable LBS weight residual (regularizers) ---
        loss_lbs_w_reg = None
        loss_lbs_w_dev = None
        # Accept both optim.loss.lbs_weight_residual.* and optim.lbs_weight_residual.*.
        lbs_sub = _cfg_get(_cfg_get(self.optimizer_config, "loss", None), "lbs_weight_residual", None)
        if lbs_sub is None:
            lbs_sub = _cfg_get(self.optimizer_config, "lbs_weight_residual", None)
        lambda_w_reg = float(_cfg_get(lbs_sub, "lambda_w_reg", 0.0) or 0.0)
        lambda_w_dev = float(_cfg_get(lbs_sub, "lambda_w_dev", 0.0) or 0.0)
        if getattr(self.gs_model, "use_deformation", False) and getattr(self.gs_model, "lbs_weight_residual", None) is not None:
            mod = self.gs_model.lbs_weight_residual
            if lambda_w_reg > 0:
                try:
                    loss_lbs_w_reg = (mod.delta_l.pow(2)).mean()
                    loss = loss + lambda_w_reg * loss_lbs_w_reg
                except Exception:
                    pass
            if lambda_w_dev > 0:
                try:
                    w = mod.corrected_weights_vj()
                    w0 = mod.w0_full
                    idx = mod.idx_topk
                    w_s = w.gather(1, idx)
                    w0_s = w0.gather(1, idx)
                    loss_lbs_w_dev = (w_s - w0_s).pow(2).mean()
                    loss = loss + lambda_w_dev * loss_lbs_w_dev
                except Exception:
                    pass

        # --- TODO-2: delta_u magnitude regularization ---
        loss_delta_u_reg = None
        if lambda_delta_u > 0 and du_enabled and getattr(self.gs_model, "use_deformation", False):
            du = getattr(self.gs_model, "_current_delta_u_local_ntb", None)
            if isinstance(du, torch.Tensor) and du.numel() > 0:
                # L = mean(||delta_u||^2)
                loss_delta_u_reg = (du.pow(2).sum(dim=-1)).mean()
                loss = loss + lambda_delta_u * loss_delta_u_reg

        # --- TODO-5 (P2-1): base scale residual losses (isotropy + reg) ---
        loss_scale_iso = None
        loss_scale_reg = None
        scale_cfg = _cfg_get(self.gs_model, "config", None)
        try:
            model_scale_cfg = _cfg_get(scale_cfg, "deform_scale_base", None) if scale_cfg is not None else None
        except Exception:
            model_scale_cfg = None
        # Loss weights: accept optim.loss.deform_scale_base.* and optim.deform_scale_base.*.
        scale_loss_sub = _cfg_get(_cfg_get(self.optimizer_config, "loss", None), "deform_scale_base", None)
        if scale_loss_sub is None:
            scale_loss_sub = _cfg_get(self.optimizer_config, "deform_scale_base", None)
        lambda_iso = float(_cfg_get(scale_loss_sub, "lambda_iso", 0.0) or 0.0)
        lambda_s_reg = float(_cfg_get(scale_loss_sub, "lambda_s_reg", 0.0) or 0.0)
        clamp_log_s = float(_cfg_get(model_scale_cfg, "clamp_log_s", 0.3) or 0.3)
        enabled_scale = bool(_cfg_get(model_scale_cfg, "enabled", False)) if model_scale_cfg is not None else False
        if enabled_scale and getattr(self.gs_model, "delta_log_s_base", None) is not None:
            try:
                d = self.gs_model.delta_log_s_base
                loss_scale_reg = (d.pow(2)).mean()
                if lambda_s_reg > 0:
                    loss = loss + lambda_s_reg * loss_scale_reg
            except Exception:
                pass
            if lambda_iso > 0:
                try:
                    d_clamped = self.gs_model.delta_log_s_base.clamp(-clamp_log_s, clamp_log_s)
                    log_s_new = self.gs_model._scaling + d_clamped.to(self.gs_model._scaling.device, self.gs_model._scaling.dtype)
                    loss_scale_iso = log_s_new.var(dim=-1).mean()
                    loss = loss + lambda_iso * loss_scale_iso
                except Exception:
                    pass

        # --- TODO-3: temporal DoD + coeff smooth (3 micro-steps) ---
        # DoD config (new schema: optim.loss.dod.*)
        dod_use_luma = bool(_cfg_get(dod_sub, "use_luma", True))
        dod_blur_cfg = _cfg_get(dod_sub, "blur", None)
        dod_blur_enabled = bool(_cfg_get(dod_blur_cfg, "enabled", False)) if dod_blur_cfg is not None else False
        dod_blur_k = int(_cfg_get(dod_blur_cfg, "kernel_size", 5)) if dod_blur_cfg is not None else 5
        dod_blur_sigma = float(_cfg_get(dod_blur_cfg, "sigma", 1.0)) if dod_blur_cfg is not None else 1.0
        dod_maskpair_cfg = _cfg_get(dod_sub, "mask_pair", None)
        dod_maskpair_enabled = bool(_cfg_get(dod_maskpair_cfg, "enabled", True)) if dod_maskpair_cfg is not None else True
        dod_maskpair_extra = bool(_cfg_get(dod_maskpair_cfg, "extra_erode", False)) if dod_maskpair_cfg is not None else False
        dod_maskpair_ksize = int(_cfg_get(dod_maskpair_cfg, "extra_erode_ksize", 7)) if dod_maskpair_cfg is not None else 7
        dod_maskpair_iters = int(_cfg_get(dod_maskpair_cfg, "extra_erode_iters", 1)) if dod_maskpair_cfg is not None else 1

        # Fixed-gray composition for DoD/LPIPS inputs (pixel losses remain unchanged)
        lpips_bg_mode = str(_cfg_get(lpips_sub, "bg_mode", "random"))
        lpips_fixed_gray = float(_cfg_get(lpips_sub, "fixed_gray", 0.5))
        dod_use_fixed_gray = (lpips_bg_mode == "fixed_gray")

        lambda_c_smooth = float(self.optimizer_config.get("lambda_c_smooth", 0.0))
        loss_dod_prev_cur = None
        loss_dod_cur_next = None
        loss_c_smooth = None
        dod_err_vis_prev_cur = None
        dod_err_vis_cur_next = None

        window_id = kwargs.get("temporal_window_id", None)
        tag = kwargs.get("temporal_tag", None)  # "prev" | "cur" | "next"
        bg_key = kwargs.get("temporal_bg_key", None)
        temporal_enabled = (lambda_dod > 0 and dod_enabled) or (lambda_c_smooth > 0)

        if temporal_enabled and (window_id is not None) and (tag in {"prev", "cur", "next"}):
            # Safety: avoid unbounded growth if windows are interrupted frequently.
            if isinstance(self.temporal_cache, dict) and len(self.temporal_cache) > 32:
                self.temporal_cache.clear()
            # Build mask_e for temporal losses (SSIM-style erosion when available).
            if gt_alpha_mask is not None:
                mask_e = _build_mask_e(gt_alpha_mask)  # [1,H,W]
            else:
                mask_e = torch.ones((1, image.shape[-2], image.shape[-1]), device=image.device, dtype=image.dtype)

            entry = {
                "I_pred": image.detach().to(dtype=torch.float16),
                "I_gt": gt_image.detach().to(dtype=torch.float16),
                "I_pred_fixed": (
                    _compose_fixed_gray(image, mask_e, gray=lpips_fixed_gray).detach().to(dtype=torch.float16)
                    if dod_use_fixed_gray
                    else None
                ),
                "I_gt_fixed": (
                    _compose_fixed_gray(gt_image, mask_e, gray=lpips_fixed_gray).detach().to(dtype=torch.float16)
                    if dod_use_fixed_gray
                    else None
                ),
                "mask_e": mask_e.detach().to(dtype=torch.float16),
                "c": (_maybe_get_coeff_vector().detach().to(dtype=torch.float16) if _maybe_get_coeff_vector() is not None else None),
                "bg_key": bg_key,
            }

            cache = self.temporal_cache.setdefault(str(window_id), {})

            def _mask_pair(a, b):
                if not dod_maskpair_enabled:
                    return torch.ones_like(a["mask_e"]).to(dtype=image.dtype)
                m = (a["mask_e"] * b["mask_e"]).to(dtype=image.dtype)  # [1,H,W]
                if dod_maskpair_extra:
                    try:
                        m = erode_mask(m.float(), ksize=dod_maskpair_ksize, iters=dod_maskpair_iters).to(dtype=image.dtype)
                    except Exception:
                        pass
                return m

            def _dod_loss(a, b):
                if lambda_dod <= 0:
                    return None, None
                # Choose DoD inputs (fixed-gray or raw), then optional luma + blur.
                a_pred_fixed = a.get("I_pred_fixed", None) if dod_use_fixed_gray else None
                b_pred_fixed = b.get("I_pred_fixed", None) if dod_use_fixed_gray else None
                a_gt_fixed = a.get("I_gt_fixed", None) if dod_use_fixed_gray else None
                b_gt_fixed = b.get("I_gt_fixed", None) if dod_use_fixed_gray else None

                a_pred = a_pred_fixed if isinstance(a_pred_fixed, torch.Tensor) else a["I_pred"]
                b_pred = b_pred_fixed if isinstance(b_pred_fixed, torch.Tensor) else b["I_pred"]
                a_gt = a_gt_fixed if isinstance(a_gt_fixed, torch.Tensor) else a["I_gt"]
                b_gt = b_gt_fixed if isinstance(b_gt_fixed, torch.Tensor) else b["I_gt"]

                a_pred = a_pred.to(dtype=image.dtype)
                b_pred = b_pred.to(dtype=image.dtype)
                a_gt = a_gt.to(dtype=image.dtype)
                b_gt = b_gt.to(dtype=image.dtype)

                if dod_use_luma:
                    a_pred_x = _luma_3chw(a_pred)
                    b_pred_x = _luma_3chw(b_pred)
                    a_gt_x = _luma_3chw(a_gt)
                    b_gt_x = _luma_3chw(b_gt)
                else:
                    a_pred_x, b_pred_x, a_gt_x, b_gt_x = a_pred, b_pred, a_gt, b_gt

                if dod_blur_enabled:
                    a_pred_x = gaussian_blur_2d(a_pred_x.unsqueeze(0), kernel_size=dod_blur_k, sigma=dod_blur_sigma).squeeze(0)
                    b_pred_x = gaussian_blur_2d(b_pred_x.unsqueeze(0), kernel_size=dod_blur_k, sigma=dod_blur_sigma).squeeze(0)
                    a_gt_x = gaussian_blur_2d(a_gt_x.unsqueeze(0), kernel_size=dod_blur_k, sigma=dod_blur_sigma).squeeze(0)
                    b_gt_x = gaussian_blur_2d(b_gt_x.unsqueeze(0), kernel_size=dod_blur_k, sigma=dod_blur_sigma).squeeze(0)

                d_pred = (b_pred_x - a_pred_x)
                d_gt = (b_gt_x - a_gt_x)
                diff = d_pred - d_gt
                m = _mask_pair(a, b)  # [1,H,W]
                # Charbonnier per-pixel, average channels
                per = _charbonnier(diff, eps=1e-3).mean(dim=0, keepdim=True)  # [1,H,W]
                eps = 1e-6
                L = (per * m).sum() / (m.sum() + eps)
                # vis heatmap
                if bool(kwargs.get("debug_vis", False)):
                    v = (per * m).detach().squeeze(0)
                    denom = v.max().clamp_min(1e-6)
                    h = (v / denom).clamp(0.0, 1.0)
                    vis = h.unsqueeze(0).repeat(3, 1, 1)
                else:
                    vis = None
                return L, vis

            def _c_smooth(a, b):
                if lambda_c_smooth <= 0:
                    return None
                ca, cb = a.get("c", None), b.get("c", None)
                if ca is None or cb is None:
                    return None
                ca = ca.to(dtype=image.dtype)
                cb = cb.to(dtype=image.dtype)
                return (cb - ca).pow(2).mean()

            if tag == "prev":
                cache["prev"] = entry
            elif tag == "cur":
                prev = cache.get("prev", None)
                if prev is not None and (prev.get("bg_key", None) == bg_key):
                    loss_dod_prev_cur, dod_err_vis_prev_cur = _dod_loss(prev, entry)
                    loss_c_smooth = _c_smooth(prev, entry)
                    if loss_dod_prev_cur is not None:
                        loss = loss + lambda_dod * loss_dod_prev_cur
                    if loss_c_smooth is not None:
                        loss = loss + lambda_c_smooth * loss_c_smooth
                cache["cur"] = entry
            else:  # next
                cur = cache.get("cur", None)
                if cur is not None and (cur.get("bg_key", None) == bg_key):
                    loss_dod_cur_next, dod_err_vis_cur_next = _dod_loss(cur, entry)
                    loss_c_term = _c_smooth(cur, entry)
                    if loss_dod_cur_next is not None:
                        loss = loss + lambda_dod * loss_dod_cur_next
                    if loss_c_term is not None:
                        loss_c_smooth = loss_c_term if loss_c_smooth is None else (loss_c_smooth + loss_c_term)
                        loss = loss + lambda_c_smooth * loss_c_term
                # cleanup window cache
                try:
                    self.temporal_cache.pop(str(window_id), None)
                except Exception:
                    pass

        if lambda_lpips > 0 and lpips_enabled:
            # Optionally compute LPIPS on a fixed-gray composition (recommended for 2K color-bias stability).
            lpips_bg_mode = str(_cfg_get(lpips_sub, "bg_mode", "random"))
            lpips_fixed_gray = float(_cfg_get(lpips_sub, "fixed_gray", 0.5))
            lpips_use_mask = bool(_cfg_get(lpips_sub, "use_mask", True))

            if (lpips_bg_mode == "fixed_gray") and (gt_alpha_mask is not None) and lpips_use_mask and (not use_mask_for_all):
                # Use eroded mask (SSIM-style) for stability on silhouettes.
                try:
                    mask_e_lp = _build_mask_e(gt_alpha_mask)  # [1,H,W]
                except Exception:
                    mask_e_lp = (gt_alpha_mask > 0.5).float()
                pred_fixed = _compose_fixed_gray(image, mask_e_lp, gray=lpips_fixed_gray)
                gt_fixed = _compose_fixed_gray(gt_image, mask_e_lp, gray=lpips_fixed_gray)
                Llpips = self.lpips(gt_fixed, pred_fixed).squeeze()
            else:
                # Legacy behavior
                if gt_alpha_mask is not None:
                    if use_mask_for_all:
                        Llpips = self.lpips(masked_gt, masked_image).squeeze()
                    else:
                        Llpips = self.lpips(gt_image * gt_alpha_mask, image * gt_alpha_mask).squeeze()
                else:
                    Llpips = self.lpips(masked_gt, masked_image).squeeze()

            loss += float(lambda_lpips) * Llpips

        if self.optimizer_config.get('lambda_sparsity', 0) > 0:
            loss += self.optimizer_config.lambda_sparsity * self.gs_model.get_opacity.mean()

        if self.optimizer_config.get('lambda_scaling', 0) > 0:
            thresh_scaling_max = self.optimizer_config.get('thresh_scaling_max', 0.008)
            thresh_scaling_ratio = self.optimizer_config.get('thresh_scaling_ratio', 10.0)
            max_vals = self.gs_model.get_scaling.max(dim=-1).values
            min_vals = self.gs_model.get_scaling.min(dim=-1).values
            ratio = max_vals / min_vals
            thresh_idxs = (max_vals > thresh_scaling_max) & (ratio > thresh_scaling_ratio)
            if thresh_idxs.sum() > 0:
                loss += self.optimizer_config.lambda_scaling * max_vals[thresh_idxs].mean()

        # Report PSNR on unmasked composites to keep continuity, unless user wants masked reporting (future option)
        psnr_full = psnr(image, gt_image).mean().float().item()

        out = {
            'loss': loss,
            'psnr_full': psnr_full,
        }
        if loss_grad is not None:
            out["loss_grad"] = loss_grad
        if grad_diff_vis is not None:
            out["grad_diff_vis"] = grad_diff_vis
        if loss_delta_u_reg is not None:
            out["loss_delta_u_reg"] = loss_delta_u_reg
        if loss_lbs_w_reg is not None:
            out["loss_lbs_w_reg"] = loss_lbs_w_reg
        if loss_lbs_w_dev is not None:
            out["loss_lbs_w_dev"] = loss_lbs_w_dev
        if loss_scale_iso is not None:
            out["loss_scale_iso"] = loss_scale_iso
        if loss_scale_reg is not None:
            out["loss_scale_reg"] = loss_scale_reg
        if loss_joint_edge is not None:
            out["loss_joint_edge"] = loss_joint_edge
        if loss_joint_lap is not None:
            out["loss_joint_lap"] = loss_joint_lap
        if loss_dod_prev_cur is not None:
            out["loss_dod_prev_cur"] = loss_dod_prev_cur
        if loss_dod_cur_next is not None:
            out["loss_dod_cur_next"] = loss_dod_cur_next
        if loss_c_smooth is not None:
            out["loss_c_smooth"] = loss_c_smooth
        if dod_err_vis_prev_cur is not None:
            out["dod_err_vis_prev_cur"] = dod_err_vis_prev_cur
        if dod_err_vis_cur_next is not None:
            out["dod_err_vis_cur_next"] = dod_err_vis_cur_next
        return out

########## testing routine ##########
def visualize_compare(gt_image, image, psnr, ssim, lpips):
    compare = torch.concat([gt_image, image], dim=2)
    compare = (compare.permute([1, 2, 0]) * 255)[:, :, [2, 1, 0]].detach().cpu().numpy()
    compare = cv2.putText(compare, f'psnr/ssim/lpips', (20, compare.shape[0] - 50), 0, 1, (0, 0, 255))
    compare = cv2.putText(compare, f'{psnr:.4f}/{ssim:.4f}/{lpips:.4f}', 
                            (20, compare.shape[0] - 10), 0, 1, (0, 0, 255))
    
    err = (image - gt_image).abs().max(dim=0)[0].clip(0, 1)     
    from model import libcore       
    err_map = libcore.colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
    compare = np.concatenate([compare, err_map], axis=1)
    return compare

def visualize_compare_variants(gt_image, pred_dict, labels_order, metrics_dict=None):
    """Create a 2-row grid: top row shows GT + preds, bottom row shows errors for preds.

    pred_dict: {label: imageTensor[3,H,W]} images in [0,1]
    labels_order: list of labels, e.g. ["raw_lbs","offset","full"]
    metrics_dict: optional {label: (psnr, ssim, lpips)} to annotate.
    """
    import cv2
    from model import libcore
    gt = (gt_image.permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy()[:, :, [2, 1, 0]]
    gt = np.ascontiguousarray(gt.astype(np.uint8))
    H, W = gt.shape[:2]

    def _to_bgr_u8(img):
        x = (img.permute(1, 2, 0).clamp(0, 1) * 255).detach().cpu().numpy()
        x = x[:, :, [2, 1, 0]]
        return np.ascontiguousarray(x.astype(np.uint8))

    top = [gt]
    bottom = [np.zeros_like(gt)]
    # GT label
    gt_l = cv2.putText(gt.copy(), "GT", (10, 30), 0, 1, (0, 0, 255), 2)
    top[0] = gt_l

    for lab in labels_order:
        img = pred_dict[lab]
        bgr = _to_bgr_u8(img)
        txt = lab
        if metrics_dict and lab in metrics_dict:
            p, s, l = metrics_dict[lab]
            txt = f"{lab}  p/s/l={p:.2f}/{s:.3f}/{l:.3f}"
        bgr = cv2.putText(bgr, txt, (10, 30), 0, 0.8, (0, 0, 255), 2)
        top.append(bgr)

        err = (img - gt_image).abs().max(dim=0)[0].clip(0, 1)
        err_map = libcore.colorizeWeightsMap(err.detach().cpu().numpy(), min_val=0, max_val=1)
        err_map = np.ascontiguousarray(err_map)
        bottom.append(err_map)

    top_row = np.concatenate(top, axis=1)
    bot_row = np.concatenate(bottom, axis=1)
    grid = np.concatenate([top_row, bot_row], axis=0)
    return grid

# tensor to image
def write_tensor_image(fn, tensor, rgb2bgr=False):
    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[2] == 3:
            tensor = tensor[:, :, [2, 1, 0]]
        else:
            tensor = tensor[:, :, [2, 1, 0, 3]]
    
    cv2.imwrite(fn, (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8))

# testing routine
def testing_routine(pipe, frameset, gs_model, render_dir=None, compare_dir=None, verify=None, bg_color='white', optimizer_config=None):
    if render_dir is not None:
        os.makedirs(render_dir, exist_ok=True)
    if compare_dir is not None:
        os.makedirs(compare_dir, exist_ok=True)

    psnr_full = 0
    ssim_full = 0
    lpips_full = 0
    count = 0

    dataloader = make_dataloader(frameset, shuffle=False)
    data_iterator = iter(dataloader)

    with torch.no_grad():
        num_frames = len(frameset)
        if num_frames == 0:
            raise RuntimeError(
                "Testing dataset is empty (len(frameset)==0). "
                "If you set val/test cameras, ensure they match calibration camera ids (e.g. '005'), "
                "or pass 1-based camera indices (e.g. 126) which are now supported."
            )
        pbar = tqdm(range(num_frames))
        # Optional PLY dumping control (default: dump only first K frames per cam in deformation mode)
        evcfg = {}
        try:
            evcfg = getattr(gs_model, "config", {}).get("eval_variants", {})
        except Exception:
            evcfg = {}
        dump_ply_enabled = bool(evcfg.get("dump_ply_enabled", True))
        dump_ply_first_k = int(evcfg.get("dump_ply_first_k", 3))
        dump_counts_by_cam = {}

        def _maybe_dump_plys(cam_id: str, frame_tag: str, variants_list, mesh_info_for_variant, smplx_dict_for_variant):
            if (not dump_ply_enabled) or (dump_ply_first_k <= 0):
                return
            if not bool(getattr(gs_model, "use_deformation", False)):
                return
            c = dump_counts_by_cam.get(cam_id, 0)
            if c >= dump_ply_first_k:
                return
            dump_counts_by_cam[cam_id] = c + 1

            # Resolve dump root under eval_<iter>/ply/<cam_id>/
            if render_dir is None:
                return
            eval_root = os.path.dirname(render_dir)
            ply_dir = os.path.join(eval_root, "ply", str(cam_id))
            os.makedirs(ply_dir, exist_ok=True)

            # Helper to save mesh ply (optionally with vertex quality scalar for delta_norm)
            def _save_mesh_ply(path, verts, faces, colors=None, vert_quality=None):
                from utils.ply_io import save_ply_mesh
                save_ply_mesh(path, verts, faces, vert_colors=colors, vert_quality=vert_quality)

            # Need faces for clothed mesh
            faces = getattr(gs_model, "cano_faces", None)
            if faces is None:
                return

            for vname, v_apply_vert, v_apply_bs in variants_list:
                gs_model.update_to_posed_mesh(
                    raw_mesh_info_for_no_deform=mesh_info_for_variant,
                    current_target_frame_smpl_params=smplx_dict_for_variant,
                    current_frame_idx=int(frame_tag),
                    apply_vertex_offsets=v_apply_vert,
                    apply_gauss_bs=v_apply_bs,
                )
                # Save posed cloth mesh only for raw_lbs/offset (BS does not affect mesh).
                if vname in ("raw_lbs", "offset") and getattr(gs_model, "mesh_verts", None) is not None:
                    colors = None
                    try:
                        from utils.lbs_vis import lbs_weights_to_vertex_colors
                        evcfg = {}
                        try:
                            evcfg = getattr(gs_model, "config", {}).get("mesh_vis", {})
                        except Exception:
                            evcfg = {}
                        dump_lbs_colors = bool(evcfg.get("dump_lbs_colors", True))
                        lbs_color_mode = str(evcfg.get("lbs_color_mode", "argmax"))
                        if dump_lbs_colors and getattr(gs_model, "lbs_weights", None) is not None:
                            # Use updated/corrected weights if TODO-4 is enabled (updated-only visualization).
                            w_vis = gs_model.get_lbs_weights_for_skinning() if hasattr(gs_model, "get_lbs_weights_for_skinning") else gs_model.lbs_weights
                            colors = lbs_weights_to_vertex_colors(w_vis, mode=lbs_color_mode)
                    except Exception:
                        colors = None
                    _save_mesh_ply(
                        os.path.join(ply_dir, f"{frame_tag}_{vname}_mesh.ply"),
                        gs_model.mesh_verts,
                        faces,
                        colors=colors,
                    )
                    # LBS v1.1: save same mesh with vertex quality = |w - w0| (delta_norm) for visualization
                    try:
                        mod = getattr(gs_model, "lbs_weight_residual", None)
                        if mod is not None and getattr(gs_model, "config", None) is not None:
                            lbs_cfg = getattr(gs_model.config, "lbs_weight_residual", None) or gs_model.config.get("lbs_weight_residual", {})
                            vis_cfg = getattr(lbs_cfg, "vis", None) or (lbs_cfg.get("vis") if isinstance(lbs_cfg, dict) else None)
                            metric = str(getattr(vis_cfg, "delta_norm_metric", None) or (vis_cfg.get("delta_norm_metric", "l1") if isinstance(vis_cfg, dict) else "l1"))
                            # Use projected baseline so q==0 unless LBS residual changed.
                            q = mod.delta_norm_per_vertex(global_step=None, metric=metric, baseline="projected")
                            from utils.lbs_vis import scalar_to_heatmap_vertex_colors
                            q_colors = scalar_to_heatmap_vertex_colors(q)
                            _save_mesh_ply(
                                os.path.join(ply_dir, f"{frame_tag}_{vname}_mesh_delta_norm.ply"),
                                gs_model.mesh_verts,
                                faces,
                                colors=q_colors,
                                vert_quality=q,
                            )
                    except Exception:
                        pass
                # Save gaussians ply (variant-specific because get_xyz depends on deltas)
                try:
                    gs_model.save_ply(os.path.join(ply_dir, f"{frame_tag}_{vname}_gs.ply"))
                except Exception:
                    pass

        for idx in pbar:
            batch = next(data_iterator)[0]
            frm_idx = batch['frm_idx']
            scene_cameras = batch['scene_cameras']

            # there should be only one camera
            viewpoint_cam = scene_cameras[0].cuda()

            mesh_info = batch.get('mesh_info')
            use_def = bool(getattr(gs_model, 'use_deformation', False))
            if not use_def:
                # Non-deformation eval: keep original single-render behavior and flat filenames.
                gs_model.update_to_posed_mesh(mesh_info)
                render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=bg_color)
                image = render_pkg['render']
                gt_image = render_pkg['gt_image']

                _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
                _ssim = img_ssim(image[None, ...], gt_image[None, ...])
                _psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
                _lpips = perceptual(image[None, ...], gt_image[None, ...], mask=None, use_mask=False)

                #############
                if verify is not None:
                    network_gui.send_image_to_network(image, verify)

                #############
                if render_dir is not None:
                    cam_idx = batch.get('cam_id_str', '')
                    write_tensor_image(os.path.join(render_dir, f'{frm_idx:05d}_{cam_idx}.png'), image, rgb2bgr=True)
                if compare_dir is not None:
                    cam_idx = batch.get('cam_id_str', '')
                    compare = visualize_compare(gt_image, image, _psnr.item(), _ssim.item(), _lpips.item())
                    cv2.imwrite(os.path.join(compare_dir, f'{frm_idx:05d}_{cam_idx}.jpg'), compare)

                psnr_full += _psnr.item()
                ssim_full += _ssim.item()
                lpips_full += _lpips.item()
                count += 1

                pbar.set_postfix({
                    'psnr': f'{(psnr_full / count):.4f}({_psnr.item():.4f})',
                    'ssim': f'{(ssim_full / count):.4f}({_ssim.item():.4f})',
                    'lpips': f'{(lpips_full / count):.4f}({_lpips.item():.4f})',
                })
                continue

            # Deformation eval: render variants + per-cam subdirs + optional PLY dumps.
            if mesh_info is None or 'smplx_params_raw_for_deformnet' not in mesh_info:
                raise RuntimeError("Deformation evaluation requires mesh_info.smplx_params_raw_for_deformnet")
            smplx_dict = mesh_info['smplx_params_raw_for_deformnet']

            variants = [("raw_lbs", False, False), ("offset", True, False), ("full", True, True)]
            pred_imgs = {}
            metrics = {}
            gt_image = None
            for vname, v_apply_vert, v_apply_bs in variants:
                gs_model.update_to_posed_mesh(
                    raw_mesh_info_for_no_deform=mesh_info,
                    current_target_frame_smpl_params=smplx_dict,
                    current_frame_idx=int(frm_idx),
                    apply_vertex_offsets=v_apply_vert,
                    apply_gauss_bs=v_apply_bs,
                )
                render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=bg_color)
                image = render_pkg['render']
                gt_image = render_pkg['gt_image'] if gt_image is None else gt_image
                pred_imgs[vname] = image

                _rmse = img_mse(image[None, ...], gt_image[None, ...], mask=None, error_type='rmse', use_mask=False)
                _ssim = img_ssim(image[None, ...], gt_image[None, ...])
                _psnr = img_psnr(image[None, ...], gt_image[None, ...], rmse=_rmse)
                _lpips = perceptual(image[None, ...], gt_image[None, ...], mask=None, use_mask=False)
                metrics[vname] = (_psnr.item(), _ssim.item(), _lpips.item())

            # Use "full" for aggregate stats
            _psnr, _ssim, _lpips = metrics["full"][0], metrics["full"][1], metrics["full"][2]

            #############
            if verify is not None:
                network_gui.send_image_to_network(pred_imgs["full"], verify)
        
            #############
            if render_dir is not None:
                cam_idx = batch.get('cam_id_str', '')
                cam_dir = os.path.join(render_dir, str(cam_idx))
                os.makedirs(cam_dir, exist_ok=True)
                # save GT once
                write_tensor_image(os.path.join(cam_dir, f'{frm_idx:05d}_gt.png'), gt_image, rgb2bgr=True)
                for vname, _, _ in variants:
                    write_tensor_image(os.path.join(cam_dir, f'{frm_idx:05d}_{vname}.png'), pred_imgs[vname], rgb2bgr=True)
                # Optional grad heatmap on full prediction
                if optimizer_config is not None:
                    try:
                        grad_vis = sobel_grad_diff_vis(
                            image=pred_imgs["full"],
                            gt_image=gt_image,
                            gt_alpha_mask=render_pkg.get("gt_alpha_mask", None),
                            optimizer_config=optimizer_config,
                        )
                        if grad_vis is not None:
                            write_tensor_image(os.path.join(cam_dir, f"{frm_idx:05d}_grad.png"), grad_vis, rgb2bgr=True)
                    except Exception:
                        pass
            if compare_dir is not None:
                compare = visualize_compare_variants(gt_image, pred_imgs, [v[0] for v in variants], metrics_dict=metrics)
                cam_dir = os.path.join(compare_dir, str(cam_idx))
                os.makedirs(cam_dir, exist_ok=True)
                cv2.imwrite(os.path.join(cam_dir, f'{frm_idx:05d}.jpg'), compare)

            # Optional: dump intermediate PLYs for a small subset of frames per camera
            _maybe_dump_plys(str(cam_idx), f"{int(frm_idx):05d}", variants, mesh_info, smplx_dict)
            
                # err_map = cv2.putText(err_map, f'psnr/ssim/lpips', (20, err_map.shape[0] - 50), 0, 1, (255, 255, 255))
                # err_map = cv2.putText(err_map, f'{_psnr.item():.4f}/{_ssim.item():.4f}/{_lpips.item():.4f}', 
                #                       (20, err_map.shape[0] - 10), 0, 1, (255, 255, 255))
                # cv2.imwrite(os.path.join(err_dir, f'{frm_idx:05d}.jpg'), err_map)
            #############

            psnr_full += float(_psnr)
            ssim_full += float(_ssim)
            lpips_full += float(_lpips)
            count += 1

            pbar.set_postfix({
                'psnr': f'{(psnr_full / count):.4f}({float(_psnr):.4f})',
                'ssim': f'{(ssim_full / count):.4f}({float(_ssim):.4f})',
                'lpips': f'{(lpips_full / count):.4f}({float(_lpips):.4f})',
            })

    return {
        'psnr': psnr_full / count,
        'ssim': ssim_full / count,
        'lpips': lpips_full / count,
        'n_gauss': gs_model._xyz.shape[0],
    }

def run_testing(pipe, frameset_test, gs_model, model_path=None, iteration=None, verify=None, bg_color='white', optimizer_config=None):
    if model_path is not None:
        if iteration is not None:
            render_dir = os.path.join(model_path, f'eval_{iteration}/render')
            compare_dir = os.path.join(model_path, f'eval_{iteration}/compare')
            stats_fn = os.path.join(model_path, f'eval_{iteration}/stats.json')
        else:
            render_dir = os.path.join(model_path, f'eval/render')
            compare_dir = os.path.join(model_path, f'eval/compare')
            stats_fn = os.path.join(model_path, f'eval/stats.json')
    else:
        stats_fn = None

    stats = testing_routine(pipe, frameset_test, gs_model,
                            render_dir=render_dir, 
                            compare_dir=compare_dir,
                            verify=verify,
                            bg_color=bg_color,
                            optimizer_config=optimizer_config)
    
    if stats_fn is not None:
        with open(stats_fn, 'w') as fp:
            json.dump(stats, fp)
    return stats


def run_temporal_window_validation(
    *,
    pipe,
    frameset,
    gs_model,
    out_dir: str,
    optimizer_config,
    window_starts: list[int],
    cam_spec: str | int,
    bg_mode: str = "random",
):
    """Validate temporal stability on specific 3-frame windows (t,t+1,t+2) for one camera.

    Writes under:
      <out_dir>/temporal_window/<cam_id>/<start>/{renders,heatmaps}/...
    and returns aggregate scalars.
    """
    import hashlib
    from pathlib import Path

    out_root = Path(out_dir) / "temporal_window"
    out_root.mkdir(parents=True, exist_ok=True)

    window_starts = [int(x) for x in (window_starts or [])]

    # Resolve camera id (supports exact id or 1-based index)
    all_ids = getattr(frameset, "all_camera_ids_sorted", [])
    cam_id = None
    cam_s = str(cam_spec)
    if cam_s in all_ids:
        cam_id = cam_s
    elif cam_s.isdigit() and len(all_ids) > 0:
        idx = int(cam_s)
        if 1 <= idx <= len(all_ids):
            cam_id = all_ids[idx - 1]
    if cam_id is None:
        raise RuntimeError(f"Temporal window val: could not resolve cam '{cam_spec}'. Available: {all_ids[:5]}...")

    # Map (frame_idx_int, cam_id_str) -> dataset idx (robust to padding differences)
    sample_to_idx = {}
    for i, (fid, cid) in enumerate(getattr(frameset, "data_samples", [])):
        try:
            f_int = int(str(fid))
        except Exception:
            continue
        sample_to_idx[(f_int, str(cid))] = int(i)

    def _bg_for_window(start_f: int):
        if bg_mode == "black":
            return "black", "black"
        if bg_mode == "white":
            return "white", "white"
        # deterministic random for this window
        seq_name = Path(str(getattr(frameset, "dat_dir", "seq"))).name
        w_id = f"{seq_name}|{cam_id}|{start_f}"
        h = hashlib.sha1(w_id.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "little", signed=False)
        rng = np.random.RandomState(seed)
        device = getattr(gs_model, "device", "cuda")
        bg = torch.tensor(rng.rand(3), device=device, dtype=torch.float32)
        return bg, str(seed)

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

    loss_root = _cfg_get(optimizer_config, "loss", None)
    dod_cfg = _cfg_get(loss_root, "dod", None)
    lpips_cfg = _cfg_get(loss_root, "lpips", None)
    # Also support `optim.dod` / `optim.lpips` (config convenience)
    if dod_cfg is None:
        dod_cfg = _cfg_get(optimizer_config, "dod", None)
    if lpips_cfg is None:
        lpips_cfg = _cfg_get(optimizer_config, "lpips", None)

    dod_enabled = bool(_cfg_get(dod_cfg, "enabled", True)) if dod_cfg is not None else True
    lambda_dod = _cfg_get(dod_cfg, "lambda", None) if dod_cfg is not None else None
    if lambda_dod is None:
        lambda_dod = getattr(optimizer_config, "lambda_dod", _cfg_get(optimizer_config, "lambda_dod", 0.0))
    lambda_dod = float(lambda_dod)

    dod_use_luma = bool(_cfg_get(dod_cfg, "use_luma", True)) if dod_cfg is not None else True
    blur_cfg = _cfg_get(dod_cfg, "blur", None)
    blur_enabled = bool(_cfg_get(blur_cfg, "enabled", False)) if blur_cfg is not None else False
    blur_k = int(_cfg_get(blur_cfg, "kernel_size", 5)) if blur_cfg is not None else 5
    blur_sigma = float(_cfg_get(blur_cfg, "sigma", 1.0)) if blur_cfg is not None else 1.0
    mp_cfg = _cfg_get(dod_cfg, "mask_pair", None)
    mp_extra = bool(_cfg_get(mp_cfg, "extra_erode", False)) if mp_cfg is not None else False
    mp_ksize = int(_cfg_get(mp_cfg, "extra_erode_ksize", 7)) if mp_cfg is not None else 7
    mp_iters = int(_cfg_get(mp_cfg, "extra_erode_iters", 1)) if mp_cfg is not None else 1

    lpips_bg_mode = str(_cfg_get(lpips_cfg, "bg_mode", "random"))
    fixed_gray = float(_cfg_get(lpips_cfg, "fixed_gray", 0.5))
    use_fixed_gray = (lpips_bg_mode == "fixed_gray")

    def _luma_3chw(x3: torch.Tensor) -> torch.Tensor:
        if x3.shape[0] >= 3:
            return (0.299 * x3[0] + 0.587 * x3[1] + 0.114 * x3[2]).unsqueeze(0)
        return x3.mean(dim=0, keepdim=True)

    def _compose_fixed_gray(x3: torch.Tensor, mask_e_1hw: torch.Tensor) -> torch.Tensor:
        bg = torch.full_like(x3, float(fixed_gray))
        m3 = mask_e_1hw.repeat(x3.shape[0], 1, 1) if x3.shape[0] > 1 else mask_e_1hw
        return x3 * m3 + bg * (1.0 - m3)

    def _mask_e(alpha: torch.Tensor | None, *, hw: tuple[int, int] | None = None, device=None, dtype=None) -> torch.Tensor:
        ssim_masking_cfg = getattr(optimizer_config, "ssim_masking", {}) if hasattr(optimizer_config, "ssim_masking") else optimizer_config.get("ssim_masking", {})
        enabled = bool(getattr(ssim_masking_cfg, "enabled", ssim_masking_cfg.get("enabled", False))) if isinstance(ssim_masking_cfg, dict) or hasattr(ssim_masking_cfg, "get") else False
        if alpha is None:
            if hw is None:
                raise RuntimeError("temporal_window_val: missing alpha and unknown H,W")
            H, W = int(hw[0]), int(hw[1])
            return torch.ones((1, H, W), device=device, dtype=dtype if dtype is not None else torch.float32)
        if not enabled:
            return (alpha > 0.5).float()
        alpha_thresh = float(getattr(ssim_masking_cfg, "alpha_thresh", ssim_masking_cfg.get("alpha_thresh", 0.5)))
        erode_ksize = int(getattr(ssim_masking_cfg, "erode_ksize", ssim_masking_cfg.get("erode_ksize", 15)))
        erode_iters = int(getattr(ssim_masking_cfg, "erode_iters", ssim_masking_cfg.get("erode_iters", 1)))
        m = (alpha > alpha_thresh).float()
        return erode_mask(m, ksize=erode_ksize, iters=erode_iters)

    def _dod_loss_and_vis(a_pred, a_gt, a_m, b_pred, b_gt, b_m):
        H, W = int(a_pred.shape[-2]), int(a_pred.shape[-1])
        ma = _mask_e(a_m, hw=(H, W), device=a_pred.device, dtype=a_pred.dtype)
        mb = _mask_e(b_m, hw=(H, W), device=a_pred.device, dtype=a_pred.dtype)
        m = (ma * mb).to(dtype=a_pred.dtype)  # [1,H,W]
        if mp_extra:
            try:
                m = erode_mask(m.float(), ksize=mp_ksize, iters=mp_iters).to(dtype=a_pred.dtype)
            except Exception:
                pass

        # Fixed-gray composition (optional) uses per-frame eroded mask.
        if use_fixed_gray:
            a_pred = _compose_fixed_gray(a_pred, ma)
            b_pred = _compose_fixed_gray(b_pred, mb)
            a_gt = _compose_fixed_gray(a_gt, ma)
            b_gt = _compose_fixed_gray(b_gt, mb)

        if dod_use_luma:
            a_pred_x = _luma_3chw(a_pred)
            b_pred_x = _luma_3chw(b_pred)
            a_gt_x = _luma_3chw(a_gt)
            b_gt_x = _luma_3chw(b_gt)
        else:
            a_pred_x, b_pred_x, a_gt_x, b_gt_x = a_pred, b_pred, a_gt, b_gt

        if blur_enabled:
            a_pred_x = gaussian_blur_2d(a_pred_x.unsqueeze(0), kernel_size=blur_k, sigma=blur_sigma).squeeze(0)
            b_pred_x = gaussian_blur_2d(b_pred_x.unsqueeze(0), kernel_size=blur_k, sigma=blur_sigma).squeeze(0)
            a_gt_x = gaussian_blur_2d(a_gt_x.unsqueeze(0), kernel_size=blur_k, sigma=blur_sigma).squeeze(0)
            b_gt_x = gaussian_blur_2d(b_gt_x.unsqueeze(0), kernel_size=blur_k, sigma=blur_sigma).squeeze(0)

        d_pred = b_pred_x - a_pred_x
        d_gt = b_gt_x - a_gt_x
        diff = d_pred - d_gt
        per = torch.sqrt(diff * diff + 1e-6).mean(dim=0, keepdim=True)  # [1,H,W]
        L = (per * m).sum() / (m.sum() + 1e-6)
        v = (per * m).detach().squeeze(0)
        denom = v.max().clamp_min(1e-6)
        h = (v / denom).clamp(0.0, 1.0)
        vis = h.unsqueeze(0).repeat(3, 1, 1)
        return L, vis

    if (not dod_enabled) or (lambda_dod <= 0) or len(window_starts) == 0:
        return {"dod_prev_cur": 0.0, "dod_cur_next": 0.0, "dod_total": 0.0, "count": 0}

    agg = {"dod_prev_cur": 0.0, "dod_cur_next": 0.0, "count": 0}

    for start_f in window_starts:
        idxs = [sample_to_idx.get((int(start_f + k), cam_id), None) for k in range(3)]
        if any(i is None for i in idxs):
            continue

        bg, bg_key = _bg_for_window(int(start_f))
        cam_out = out_root / str(cam_id) / f"{int(start_f):05d}"
        (cam_out / "renders").mkdir(parents=True, exist_ok=True)
        (cam_out / "heatmaps").mkdir(parents=True, exist_ok=True)

        preds = []
        gts = []
        alphas = []

        for k in range(3):
            batch = frameset[idxs[k]]
            frm_idx = int(batch["frm_idx"])
            viewpoint_cam = batch["scene_cameras"][0].cuda()
            mesh_info = batch.get("mesh_info")
            if mesh_info is None or "smplx_params_raw_for_deformnet" not in mesh_info:
                continue
            smplx_dict = mesh_info["smplx_params_raw_for_deformnet"]

            gs_model.update_to_posed_mesh(
                raw_mesh_info_for_no_deform=mesh_info,
                current_target_frame_smpl_params=smplx_dict,
                current_frame_idx=frm_idx,
                apply_vertex_offsets=True,
                apply_gauss_bs=True,
            )
            pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=bg)
            preds.append(pkg["render"].detach())
            gts.append(pkg["gt_image"].detach())
            alphas.append(pkg.get("gt_alpha_mask"))

            write_tensor_image(str(cam_out / "renders" / f"{k}_{frm_idx:05d}_pred.png"), preds[-1], rgb2bgr=True)
            write_tensor_image(str(cam_out / "renders" / f"{k}_{frm_idx:05d}_gt.png"), gts[-1], rgb2bgr=True)
            try:
                gv = sobel_grad_diff_vis(image=preds[-1], gt_image=gts[-1], gt_alpha_mask=alphas[-1], optimizer_config=optimizer_config)
                if gv is not None:
                    write_tensor_image(str(cam_out / "renders" / f"{k}_{frm_idx:05d}_grad.png"), gv, rgb2bgr=True)
            except Exception:
                pass

        if len(preds) != 3:
            continue

        L01, vis01 = _dod_loss_and_vis(preds[0], gts[0], alphas[0], preds[1], gts[1], alphas[1])
        L12, vis12 = _dod_loss_and_vis(preds[1], gts[1], alphas[1], preds[2], gts[2], alphas[2])
        agg["dod_prev_cur"] += float(L01.detach().cpu().item())
        agg["dod_cur_next"] += float(L12.detach().cpu().item())
        agg["count"] += 1

        write_tensor_image(str(cam_out / "heatmaps" / "dod_err_prev_cur.png"), vis01, rgb2bgr=True)
        write_tensor_image(str(cam_out / "heatmaps" / "dod_err_cur_next.png"), vis12, rgb2bgr=True)
        try:
            (cam_out / "meta.json").write_text(json.dumps({"cam_id": cam_id, "start": int(start_f), "bg_key": bg_key}, indent=2))
        except Exception:
            pass

    if agg["count"] > 0:
        agg["dod_prev_cur"] /= agg["count"]
        agg["dod_cur_next"] /= agg["count"]
    agg["dod_total"] = agg["dod_prev_cur"] + agg["dod_cur_next"]
    return agg

