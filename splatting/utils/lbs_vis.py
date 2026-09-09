import torch


def _hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """h,s,v in [0,1], returns rgb in [0,1], shape (...,3)."""
    # Based on standard HSV->RGB conversion.
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6).to(torch.int64)
    f = h6 - i.to(h6.dtype)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    i_mod = i % 6
    r = torch.where(i_mod == 0, v, torch.where(i_mod == 1, q, torch.where(i_mod == 2, p, torch.where(i_mod == 3, p, torch.where(i_mod == 4, t, v)))))
    g = torch.where(i_mod == 0, t, torch.where(i_mod == 1, v, torch.where(i_mod == 2, v, torch.where(i_mod == 3, q, torch.where(i_mod == 4, p, p)))))
    b = torch.where(i_mod == 0, p, torch.where(i_mod == 1, p, torch.where(i_mod == 2, t, torch.where(i_mod == 3, v, torch.where(i_mod == 4, v, q)))))
    return torch.stack([r, g, b], dim=-1)


def joint_palette(num_joints: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Deterministic joint color palette, (J,3) RGB in [0,1]."""
    if num_joints <= 0:
        raise ValueError(f"num_joints must be > 0, got {num_joints}")
    h = torch.linspace(0.0, 1.0, steps=num_joints + 1, device=device, dtype=dtype)[:-1]
    s = torch.full_like(h, 0.85)
    v = torch.full_like(h, 0.95)
    return _hsv_to_rgb(h, s, v)


def lbs_weights_to_vertex_colors(
    lbs_weights: torch.Tensor,
    *,
    mode: str = "argmax",
    brightness: str = "none",
) -> torch.Tensor:
    """Convert LBS weights (V,J) to per-vertex RGB colors (V,3) in [0,1].

    - mode='argmax': discrete dominant joint color (easy to interpret).
    - mode='blend': continuous blend of joint colors (shows smooth transitions).
    - brightness='max_weight': multiply color by max weight per vertex to show confidence.
    - brightness='none': no brightness scaling.
    """
    if lbs_weights.ndim != 2:
        raise ValueError(f"lbs_weights must be (V,J), got {tuple(lbs_weights.shape)}")
    V, J = lbs_weights.shape
    device = lbs_weights.device
    dtype = lbs_weights.dtype
    pal = joint_palette(J, device=device, dtype=dtype)  # (J,3)

    mode = str(mode).lower()
    if mode == "blend":
        # Assume weights sum to 1; if not, this still yields a sensible mixture.
        colors = lbs_weights @ pal  # (V,3)
        maxw = lbs_weights.max(dim=-1).values
    else:
        # default argmax
        j = lbs_weights.argmax(dim=-1)  # (V,)
        colors = pal[j]  # (V,3)
        maxw = lbs_weights.gather(1, j[:, None]).squeeze(1)

    brightness = str(brightness).lower()
    if brightness == "max_weight":
        colors = colors * maxw.clamp(0.0, 1.0).unsqueeze(-1)

    return colors.clamp(0.0, 1.0)


def scalar_to_heatmap_vertex_colors(
    s_v: torch.Tensor,
    *,
    vmax_quantile: float = 0.99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map a per-vertex scalar (V,) to RGB heatmap (V,3) in [0,1].

    Uses HSV hue ramp from blue (low) to red (high). Intended for quick debug PLY coloring.
    """
    if s_v.ndim != 1:
        raise ValueError(f"s_v must be (V,), got {tuple(s_v.shape)}")
    s = s_v.detach()
    # Robust vmax to avoid single outlier dominating the colormap.
    try:
        vmax = torch.quantile(s, float(vmax_quantile)).clamp_min(float(eps))
    except Exception:
        vmax = s.max().clamp_min(float(eps))
    t = (s / (vmax + float(eps))).clamp(0.0, 1.0)
    # Hue: t=0 -> blue (2/3), t=1 -> red (0)
    h = (2.0 / 3.0) * (1.0 - t)
    sat = torch.ones_like(h)
    val = torch.ones_like(h)
    rgb = _hsv_to_rgb(h, sat, val)
    return rgb.clamp(0.0, 1.0)

