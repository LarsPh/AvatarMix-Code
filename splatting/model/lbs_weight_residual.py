from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn


class LbsWeightResidual(nn.Module):
    """Global learnable LBS weight residual on a fixed per-vertex support set.

    Base weights w0 are assumed to be (V,J) and sum to 1 (approx) per vertex.
    We only allow non-zero weights on a fixed per-vertex support set.
    """

    @dataclass(frozen=True)
    class AlphaKnot:
        step: int
        alpha: float

    def __init__(
        self,
        *,
        w0_full_vj: torch.Tensor,  # (V,J)
        # --- Back-compat (topK-by-w0 only) ---
        topk: int = 4,
        # --- New v1.1 options ---
        support_mode: str = "topk_only",  # "topk_only" | "union"
        topk_w0: Optional[int] = None,  # Kw; defaults to `topk`
        topk_joint_dist: int = 0,        # Kd; only used for union
        forced_joint_ids: Optional[Iterable[int]] = None,  # already-resolved indices
        k_total: Optional[int] = None,   # total support slots; if None infer
        tanh_on_delta: bool = True,
        alpha: float = 0.1,
        alpha_schedule: Optional[dict] = None,
        clamp_logit: float = 1.0,
        eps: float = 1e-8,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if w0_full_vj.ndim != 2:
            raise ValueError(f"w0_full_vj must be (V,J), got {tuple(w0_full_vj.shape)}")
        if topk <= 0:
            raise ValueError(f"topk must be > 0, got {topk}")

        w0 = w0_full_vj
        if device is not None:
            w0 = w0.to(device)
        if dtype is not None:
            w0 = w0.to(dtype)

        V, J = int(w0.shape[0]), int(w0.shape[1])
        # Normalize config
        support_mode = str(support_mode).lower()
        if support_mode not in ("topk_only", "union"):
            raise ValueError(f"support_mode must be 'topk_only' or 'union', got {support_mode}")

        Kw = int(topk if topk_w0 is None else topk_w0)
        Kw = int(max(1, min(Kw, J)))
        Kd = int(max(0, min(int(topk_joint_dist), J)))
        forced_ids = list(forced_joint_ids) if forced_joint_ids is not None else []
        forced_ids = [int(i) for i in forced_ids if 0 <= int(i) < J]

        # Total support slots (fixed parameter shape).
        infer_total = int(Kw + Kd + len(forced_ids)) if support_mode == "union" else int(Kw)
        K_total = int(infer_total if k_total is None else int(k_total))
        if K_total <= 0:
            raise ValueError(f"k_total must be > 0, got {K_total}")

        self.register_buffer("w0_full", w0.detach(), persistent=True)  # (V,J)

        # Default support = topK by w0. In union mode this is a placeholder until geometry is provided.
        idx_w = torch.topk(w0, k=Kw, dim=1, largest=True, sorted=False).indices  # (V,Kw)
        idx_support = idx_w
        if K_total != Kw:
            pad = idx_w[:, :1].expand(V, max(0, K_total - Kw))
            idx_support = torch.cat([idx_w, pad], dim=1)[:, :K_total]  # (V,K_total)
        valid_mask = torch.ones((V, K_total), device=w0.device, dtype=torch.bool)
        if K_total > Kw:
            valid_mask[:, Kw:] = False
        w0_support = w0.gather(1, idx_support).clamp_min(float(eps))  # (V,K_total)

        self.register_buffer("idx_support", idx_support.detach(), persistent=True)    # (V,K_total)
        self.register_buffer("valid_mask", valid_mask.detach(), persistent=True)     # (V,K_total)
        self.register_buffer("w0_support", w0_support.detach(), persistent=True)     # (V,K_total)

        # Learn residual logits on support (fixed shape).
        self.delta_l = nn.Parameter(torch.zeros((V, K_total), device=w0.device, dtype=w0.dtype))

        # Store config
        self.support_mode = support_mode
        self.topk_w0 = int(Kw)
        self.topk_joint_dist = int(Kd)
        self.forced_joint_ids = tuple(forced_ids)
        self.k_total = int(K_total)
        self.tanh_on_delta = bool(tanh_on_delta)

        self.alpha = float(alpha)
        self._alpha_knots, self._alpha_step_offset = self._parse_alpha_schedule(alpha_schedule)
        self.clamp_logit = float(clamp_logit)
        self.eps = float(eps)

    @staticmethod
    def _parse_alpha_schedule(alpha_schedule: Optional[dict]) -> tuple[list["LbsWeightResidual.AlphaKnot"], Optional[int]]:
        """Return (knots, step_offset). step_offset: if set, knots are relative to (global_step - step_offset). Before offset, alpha=0 (no blend)."""
        # OmegaConf DictConfig: ensure we can .get(); convert to plain dict if needed
        try:
            from omegaconf import OmegaConf
            if hasattr(alpha_schedule, "__module__") and "omegaconf" in str(type(alpha_schedule).__module__):
                alpha_schedule = OmegaConf.to_container(alpha_schedule, resolve=True)
        except Exception:
            pass
        if not isinstance(alpha_schedule, dict):
            return [], None
        if not bool(alpha_schedule.get("enabled", False)):
            return [], None
        knots = alpha_schedule.get("knots", None)
        if not isinstance(knots, (list, tuple)) or len(knots) == 0:
            return [], None
        out: list[LbsWeightResidual.AlphaKnot] = []
        for k in knots:
            if not isinstance(k, dict):
                try:
                    k = dict(k) if hasattr(k, "items") else {}
                except Exception:
                    continue
            try:
                step = int(k.get("step", 0))
                a = float(k.get("alpha", 0.0))
            except Exception:
                continue
            out.append(LbsWeightResidual.AlphaKnot(step=step, alpha=a))
        out.sort(key=lambda x: x.step)
        step_offset = alpha_schedule.get("step_offset", None)
        if step_offset is not None:
            try:
                step_offset = int(step_offset)
            except Exception:
                step_offset = None
        return out, step_offset

    def alpha_at_step(self, global_step: Optional[int]) -> float:
        # Training: piecewise constant by last knot <= step. Eval/reposing: use final knot.
        if len(self._alpha_knots) == 0:
            return float(self.alpha)
        if global_step is None:
            return float(self._alpha_knots[-1].alpha)
        s = int(global_step)
        # Before step_offset (e.g. before Stage 2): no LBS blend so weights stay at w0
        if self._alpha_step_offset is not None and s < self._alpha_step_offset:
            return 0.0
        if self._alpha_step_offset is not None:
            s = s - self._alpha_step_offset
        a = float(self._alpha_knots[0].alpha)
        for k in self._alpha_knots:
            if k.step <= s:
                a = float(k.alpha)
            else:
                break
        return a

    @torch.no_grad()
    def build_support_from_geometry(
        self,
        *,
        verts_cano_v3: torch.Tensor,   # (V,3)
        joints_cano_j3: torch.Tensor,  # (J,3)
    ) -> None:
        """(Union mode) Build idx_support/valid_mask/w0_support using joint distances + forced joints."""
        if self.support_mode != "union":
            return

        w0 = self.w0_full
        V, J = int(w0.shape[0]), int(w0.shape[1])
        if verts_cano_v3.shape[0] != V:
            raise ValueError(f"verts_cano_v3 must have V={V} vertices, got {int(verts_cano_v3.shape[0])}")
        if joints_cano_j3.shape[0] != J:
            raise ValueError(f"joints_cano_j3 must have J={J} joints, got {int(joints_cano_j3.shape[0])}")

        device = w0.device
        dtype = w0.dtype
        Vpos = verts_cano_v3.to(device=device, dtype=dtype)
        Jpos = joints_cano_j3.to(device=device, dtype=dtype)

        Kw = int(self.topk_w0)
        Kd = int(self.topk_joint_dist)
        forced_ids = list(self.forced_joint_ids)
        K_total = int(self.k_total)

        idx_w = torch.topk(w0, k=min(Kw, J), dim=1, largest=True, sorted=False).indices  # (V,Kw)

        if Kd > 0:
            d2 = (Vpos[:, None, :] - Jpos[None, :, :]).pow(2).sum(dim=-1)  # (V,J)
            idx_d = torch.topk(d2, k=min(Kd, J), dim=1, largest=False, sorted=False).indices  # (V,Kd)
        else:
            idx_d = torch.empty((V, 0), device=device, dtype=torch.int64)

        if len(forced_ids) > 0:
            idx_f = torch.tensor(forced_ids, device=device, dtype=torch.int64)[None, :].expand(V, -1)  # (V,F)
        else:
            idx_f = torch.empty((V, 0), device=device, dtype=torch.int64)

        cand = torch.cat([idx_w.to(torch.int64), idx_d.to(torch.int64), idx_f.to(torch.int64)], dim=1)  # (V,Kc)
        Kc = int(cand.shape[1])
        # Stable dedup by "keep first occurrence" mask, vectorized.
        if Kc > 1:
            eq = cand[:, :, None] == cand[:, None, :]  # (V,Kc,Kc)
            earlier = torch.tril(torch.ones((Kc, Kc), device=device, dtype=torch.bool), diagonal=-1)  # (Kc,Kc)
            has_prev = (eq & earlier[None, :, :]).any(dim=-1)  # (V,Kc)
            keep = ~has_prev
        else:
            keep = torch.ones((V, Kc), device=device, dtype=torch.bool)

        big = Kc + 10_000
        pos = torch.arange(Kc, device=device, dtype=torch.int64)[None, :].expand(V, -1)
        pos_keep = torch.where(keep, pos, torch.full_like(pos, big))
        # Select earliest k_total positions; invalid ones will have pos_keep==big.
        k_take = min(K_total, Kc)
        pos_sel = torch.topk(pos_keep, k=k_take, dim=1, largest=False, sorted=True).indices  # (V,k_take)
        idx_sel = cand.gather(1, pos_sel)  # (V,k_take)
        pos_val = pos_keep.gather(1, pos_sel)
        valid = pos_val < big

        if K_total > k_take:
            # Pad with idx_w[:,0] but mark invalid.
            pad_n = int(K_total - k_take)
            pad_idx = idx_w[:, :1].expand(V, pad_n)
            idx_support = torch.cat([idx_sel, pad_idx], dim=1)
            valid_mask = torch.cat([valid, torch.zeros((V, pad_n), device=device, dtype=torch.bool)], dim=1)
        else:
            idx_support = idx_sel[:, :K_total]
            valid_mask = valid[:, :K_total]

        # Replace invalid slots with something gather-safe (idx_w[:,0]), but keep valid_mask false.
        safe_fill = idx_w[:, :1].expand(V, K_total)
        idx_support = torch.where(valid_mask, idx_support, safe_fill)

        w0_support = w0.gather(1, idx_support).clamp_min(float(self.eps))

        # Update buffers
        self.idx_support = idx_support.detach()
        self.valid_mask = valid_mask.detach()
        self.w0_support = w0_support.detach()

    def corrected_weights_vj(self, *, global_step: Optional[int] = None) -> torch.Tensor:
        """Return corrected weights (V,J), non-zero only on support."""
        w0s = self.w0_support  # (V,K)
        idx = self.idx_support  # (V,K)
        valid = self.valid_mask  # (V,K)

        l0 = torch.log(w0s + float(self.eps))
        if bool(self.tanh_on_delta):
            dl = torch.tanh(self.delta_l) * float(self.clamp_logit)
        else:
            dl = self.delta_l * float(self.clamp_logit)
        l = l0 + dl
        # Mask invalid slots before softmax.
        if valid is not None:
            l = torch.where(valid, l, torch.full_like(l, -1.0e9))
        w_support = torch.softmax(l, dim=1)  # (V,K)

        a = float(self.alpha_at_step(global_step))
        w_mixed = (1.0 - a) * w0s + a * w_support
        if valid is not None:
            w_mixed = torch.where(valid, w_mixed, torch.zeros_like(w_mixed))
        w_mixed = w_mixed / w_mixed.sum(dim=1, keepdim=True).clamp_min(float(self.eps))

        V, J = int(self.w0_full.shape[0]), int(self.w0_full.shape[1])
        out = torch.zeros((V, J), device=w_mixed.device, dtype=w_mixed.dtype)
        out.scatter_(1, idx, w_mixed)
        out = out / out.sum(dim=1, keepdim=True).clamp_min(float(self.eps))
        return out

    def support_diff_stats(self, *, global_step: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean_abs_diff, max_abs_diff, frac_vertices_maxdiff_gt_0p05) on support.

        Important: diff is measured against the **projected baseline** weights on the same support set
        (i.e., gathered from w0 and renormalized over valid slots). This makes the stat ~0 before
        the LBS residual actually changes anything (and avoids counting support truncation as a change).
        """
        w = self.corrected_weights_vj(global_step=global_step)
        idx = self.idx_support
        valid = self.valid_mask
        w_s = w.gather(1, idx)  # (V,K)
        # Baseline = w0 gathered on support, masked, renormalized
        w0_s = self.w0_full.gather(1, idx)
        if valid is not None:
            w0_s = torch.where(valid, w0_s, torch.zeros_like(w0_s))
        w0_s = w0_s / w0_s.sum(dim=1, keepdim=True).clamp_min(float(self.eps))
        d = (w_s - w0_s).abs()
        mean_abs = d.mean()
        max_abs = d.max()
        frac = (d.max(dim=1).values > 0.05).float().mean()
        return mean_abs, max_abs, frac

    def delta_norm_per_vertex(
        self,
        *,
        global_step: Optional[int] = None,
        metric: str = "l1",
        baseline: str = "projected",  # "projected" | "full"
    ) -> torch.Tensor:
        """Per-vertex scalar norm of (w - baseline) on full J, shape (V,).

        - baseline='projected': compare against w0 projected onto current support set and renormalized.
          This is 0 before learning when alpha==0 and delta_l unchanged.
        - baseline='full': compare against raw w0_full (counts support projection/truncation as difference).
        """
        w = self.corrected_weights_vj(global_step=global_step)
        b = str(baseline).lower()
        if b == "full":
            w0 = self.w0_full.to(w.device, w.dtype)
            d = w - w0
        else:
            # Build baseline full weights = projected+renormalized on support
            idx = self.idx_support.to(w.device)
            valid = self.valid_mask.to(w.device) if self.valid_mask is not None else None
            w0_s = self.w0_full.to(w.device, w.dtype).gather(1, idx)
            if valid is not None:
                w0_s = torch.where(valid, w0_s, torch.zeros_like(w0_s))
            w0_s = w0_s / w0_s.sum(dim=1, keepdim=True).clamp_min(float(self.eps))
            V, J = int(self.w0_full.shape[0]), int(self.w0_full.shape[1])
            w0_proj = torch.zeros((V, J), device=w.device, dtype=w.dtype)
            w0_proj.scatter_(1, idx, w0_s)
            d = w - w0_proj
        m = str(metric).lower()
        if m == "l2":
            return torch.sqrt((d * d).sum(dim=1) + 1.0e-12)
        if m == "maxabs":
            return d.abs().max(dim=1).values
        # default l1
        return d.abs().sum(dim=1)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Back-compat: allow resizing delta_l when support size changes (e.g., v1 topk -> v1.1 union).
        k = prefix + "delta_l"
        if k in state_dict:
            try:
                loaded = state_dict[k]
                cur = self.delta_l
                if isinstance(loaded, torch.Tensor) and isinstance(cur, torch.Tensor) and tuple(loaded.shape) != tuple(cur.shape):
                    new = torch.zeros_like(cur)
                    # Copy overlap
                    v = min(int(new.shape[0]), int(loaded.shape[0]))
                    kk = min(int(new.shape[1]), int(loaded.shape[1]))
                    new[:v, :kk].copy_(loaded[:v, :kk].to(new.device, new.dtype))
                    state_dict[k] = new
            except Exception:
                pass
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

