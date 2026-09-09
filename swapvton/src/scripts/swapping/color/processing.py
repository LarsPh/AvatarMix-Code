import torch
import numpy as np
import os
from src.utils.color.sh import SH2RGB
from src.utils.color.color_format import rgb_to_lab
from loguru import logger


def prepare_skin_gaussian_colors_for_transfer(
    skin_gaussians_canon: dict,
    opacity_threshold: float = 0.1,
    device: str = 'cpu',
    debug_save_path_prefix: str = None,
    original_property_names: list = None,
    use_all_gaussians_for_stats: bool = False,
):

    if not skin_gaussians_canon or 'opacity' not in skin_gaussians_canon or \
       'f_dc_0' not in skin_gaussians_canon or \
       'f_dc_1' not in skin_gaussians_canon or \
       'f_dc_2' not in skin_gaussians_canon:
        print("Warning: Skin Gaussians dictionary is missing opacity or f_dc color keys. Cannot prepare colors.")
        return None, None, 0

    opacities = skin_gaussians_canon['opacity']
    if not isinstance(opacities, torch.Tensor): opacities = torch.from_numpy(np.array(opacities)).to(device)
    f_dc_0 = skin_gaussians_canon['f_dc_0']
    f_dc_1 = skin_gaussians_canon['f_dc_1']
    f_dc_2 = skin_gaussians_canon['f_dc_2']
    if not isinstance(f_dc_0, torch.Tensor): f_dc_0 = torch.from_numpy(np.array(f_dc_0)).to(device)
    if not isinstance(f_dc_1, torch.Tensor): f_dc_1 = torch.from_numpy(np.array(f_dc_1)).to(device)
    if not isinstance(f_dc_2, torch.Tensor): f_dc_2 = torch.from_numpy(np.array(f_dc_2)).to(device)


    activated_opacities = torch.sigmoid(opacities)

    num_original_skin_gaussians = opacities.shape[0]
    if num_original_skin_gaussians == 0:
        print("Warning: No skin Gaussians provided to prepare_skin_gaussian_colors_for_transfer.")
        return None, None, 0

    if debug_save_path_prefix:

        from ..data.savers import save_ply_gaussians

        output_debug_dir = os.path.dirname(debug_save_path_prefix)
        if output_debug_dir and not os.path.exists(output_debug_dir):
            os.makedirs(output_debug_dir, exist_ok=True)
        all_skin_np = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else np.array(v) for k,v in skin_gaussians_canon.items()}
        prop_names = original_property_names if original_property_names is not None else list(all_skin_np.keys())
        save_ply_gaussians(f"{debug_save_path_prefix}_all_part_gaussians_canon.ply", all_skin_np, prop_names)
        print(f"Saved all ({num_original_skin_gaussians}) input part Gaussians (canonical) to {debug_save_path_prefix}_all_part_gaussians_canon.ply")

    sh_dc_components = torch.stack([f_dc_0, f_dc_1, f_dc_2], dim=-1)
    rgb_from_sh = SH2RGB(sh_dc_components)

    if use_all_gaussians_for_stats:
        selected_gaussians_mask = torch.ones_like(activated_opacities, dtype=torch.bool, device=device)
        print(f"Using all {num_original_skin_gaussians} Gaussians for color stats (ignoring opacity threshold).")
    else:
        selected_gaussians_mask = activated_opacities >= opacity_threshold
        print(f"Filtering Gaussians by opacity threshold: {opacity_threshold}.")

    num_selected_skin_gaussians = selected_gaussians_mask.sum().item()

    if debug_save_path_prefix and not use_all_gaussians_for_stats:

        from ..data.savers import save_ply_gaussians

        output_debug_dir = os.path.dirname(debug_save_path_prefix)
        if output_debug_dir and not os.path.exists(output_debug_dir):
            os.makedirs(output_debug_dir, exist_ok=True)
        if num_selected_skin_gaussians > 0:
            selected_skin_dict = {}
            for key, value_tensor in skin_gaussians_canon.items():
                if isinstance(value_tensor, torch.Tensor) and value_tensor.shape[0] == num_original_skin_gaussians:
                     selected_skin_dict[key] = value_tensor[selected_gaussians_mask].cpu().numpy()
                else:
                     selected_skin_dict[key] = value_tensor.cpu().numpy() if isinstance(value_tensor, torch.Tensor) else np.array(value_tensor)
            prop_names = original_property_names if original_property_names is not None else list(selected_skin_dict.keys())
            save_ply_gaussians(f"{debug_save_path_prefix}_selected_opacity_skin_canon.ply", selected_skin_dict, prop_names)
            print(f"Saved {num_selected_skin_gaussians} selected-opacity skin Gaussians (canonical) to {debug_save_path_prefix}_selected_opacity_skin_canon.ply")
        elif num_selected_skin_gaussians == 0:
             print(f"No selected-opacity skin Gaussians to save for prefix {debug_save_path_prefix} (all filtered out).")
    elif debug_save_path_prefix and use_all_gaussians_for_stats:
        print(f"Debug save for high-opacity skipped as all Gaussians are being used for stats.")

    if num_selected_skin_gaussians == 0:
        print(f"Warning: No skin Gaussians selected. Original count: {num_original_skin_gaussians}.")
        return None, selected_gaussians_mask, num_original_skin_gaussians

    print(f"Selected {num_selected_skin_gaussians} Gaussians for color statistics out of {num_original_skin_gaussians}.")
    rgb_colors_filtered = rgb_from_sh[selected_gaussians_mask]
    rgb_colors_reshaped = rgb_colors_filtered.permute(1, 0).unsqueeze(-1)
    rgb_colors_reshaped = torch.clamp(rgb_colors_reshaped, 0.0, 1.0)
    print(f"Prepared skin colors for transfer with shape: {rgb_colors_reshaped.shape}")
    return rgb_colors_reshaped, selected_gaussians_mask, num_original_skin_gaussians


def robust_lab_stats_from_rgb(
    rgb_colors: torch.Tensor,
    opacity: torch.Tensor | None = None,
    *,
    mad_k: float = 4.0,
    min_neff: float = 100.0,
    min_ratio: float = 0.01,
):

    if rgb_colors is None:
        return None, None, None, 0.0

    x = rgb_colors
    if x.ndim == 3 and x.shape[0] == 3:

        x = x.squeeze(-1).permute(1, 0)
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"rgb_colors must be (N,3) or (3,N,1), got {tuple(rgb_colors.shape)}")

    x = torch.clamp(x.float(), 0.0, 1.0)
    device = x.device
    n = int(x.shape[0])
    if n == 0:
        return None, None, torch.zeros((0,), dtype=torch.bool, device=device), 0.0

    if opacity is None:
        w = None
        neff_all = float(n)
    else:
        w = opacity.float().to(device).view(-1)
        if w.numel() != n:
            raise ValueError(f"opacity must be (N,), got {tuple(w.shape)} for N={n}")
        neff_all = float(torch.sum(w).item())


    lab = rgb_to_lab(x.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
    ab = lab[:, 1:3]

    med_ab = torch.median(ab, dim=0).values
    abs_dev = torch.abs(ab - med_ab[None, :])
    mad_ab = torch.median(abs_dev, dim=0).values
    mad_ab = torch.clamp(mad_ab, min=1e-6)

    z = torch.sqrt(torch.sum(((ab - med_ab[None, :]) / mad_ab[None, :]) ** 2, dim=1))
    inlier = z <= float(mad_k)

    if w is None:
        neff = float(inlier.sum().item())
    else:
        neff = float(torch.sum(w[inlier]).item())

    min_required = max(float(min_neff), float(min_ratio) * float(neff_all))
    if not np.isfinite(neff) or neff < min_required:
        return None, None, inlier, neff

    lab_in = lab[inlier]
    if lab_in.shape[0] == 0:
        return None, None, inlier, neff

    mean_lab = torch.mean(lab_in, dim=0)
    std_lab = torch.std(lab_in, dim=0)
    return mean_lab, std_lab, inlier, neff
