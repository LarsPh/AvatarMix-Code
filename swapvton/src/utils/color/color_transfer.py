from typing import Tuple

import torch
from .color_format import rgb_to_lab, lab_to_rgb


def get_mean_and_std(x):


    x_std, x_mean = torch.std_mean(x, dim=(-2, -1))
    x_mean = x_mean.view(-1)
    x_std = x_std.view(-1)

    return x_mean, x_std


def preprocess_input(img: torch.Tensor) -> Tuple[torch.Tensor, bool]:

    is_channel_last = False
    if img.ndim == 4 and img.shape[1] != 3:
        image = img.permute(0, 3, 1, 2)
        is_channel_last = True
    elif img.ndim == 3 and img.shape[0] != 3:
        img =  img.permute(2, 0, 1)
        is_channel_last = True

    img = img.float()
    img = (img - img.min()) / (img.max() - img.min())
    return img, is_channel_last


def color_transfer_pytorch(
    src: torch.Tensor,
    target: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:

    if src.ndim != target.ndim:
        raise ValueError((
            "`src` must have the same dimension as `target`"
            f", Got {src.ndim} and {target.ndim}"
        ))


    src, src_ch_last = preprocess_input(src)
    target, _ = preprocess_input(target)

    src = rgb_to_lab(src)
    target = rgb_to_lab(target)

    src_mean, src_std = get_mean_and_std(src)
    target_mean, target_std = get_mean_and_std(target)

    std_ratio = target_std / src_std

    if not inplace:
        image = src.clone()
    else:
        image = src

    image = image.float()

    for c in range(3):
        if image.ndim == 4:
            image[:, c, ...] = (image[:, c, ...] - src_mean[c]) * std_ratio[c] + target_mean[c]
        else:
            image[c, ...] = (image[c, ...] - src_mean[c]) * std_ratio[c] + target_mean[c]


    image = lab_to_rgb(image)

    if src_ch_last:

        image = (
            image.permute(0, 2, 3, 1)
            if image.ndim == 4
            else image.permute(1, 2, 0)
        )


    return image

def transfer_skin_color_pytorch(source_rgb_prepared, target_rgb_prepared, debug_source_stats=None,
                              source_lab_stats=None):

    if target_rgb_prepared is None or target_rgb_prepared.shape[1] == 0:
        print("Warning: Target RGB for color transfer is None or empty. Skipping transfer.")
        return None

    current_device = target_rgb_prepared.device
    source_mean_lab, source_std_lab = None, None

    if debug_source_stats:
        if not all(k in debug_source_stats for k in ['mean_lab', 'std_lab']):
            print("Warning: debug_source_stats provided but missing 'mean_lab' or 'std_lab'. Ignoring debug stats.")
            debug_source_stats = None
        else:
            source_mean_lab = torch.tensor(debug_source_stats['mean_lab'], device=current_device, dtype=torch.float32).reshape(3,1,1)
            source_std_lab = torch.tensor(debug_source_stats['std_lab'], device=current_device, dtype=torch.float32).reshape(3,1,1)
            source_std_lab[source_std_lab < 1e-5] = 1e-5
            print(f"Using DEBUG source Lab stats: Mean={source_mean_lab.squeeze().tolist()}, Std={source_std_lab.squeeze().tolist()}")

    elif source_lab_stats:
        mean, std = source_lab_stats
        source_mean_lab = mean.to(current_device).reshape(3,1,1)
        source_std_lab = std.to(current_device).reshape(3,1,1)
        source_std_lab[source_std_lab < 1e-5] = 1e-5
        print(f"Using pre-calculated source Lab stats: Mean={source_mean_lab.squeeze().tolist()}, Std={source_std_lab.squeeze().tolist()}")

    else:
        if source_rgb_prepared is None or source_rgb_prepared.shape[1] == 0:
            print("Warning: Source RGB for color transfer is None or empty (and not in debug/pre-calc mode). Skipping transfer.")
            return target_rgb_prepared
        print(f"Source RGB shape: {source_rgb_prepared.shape}, Target RGB shape: {target_rgb_prepared.shape}")
        source_lab = rgb_to_lab(source_rgb_prepared.to(current_device))
        source_mean_lab = torch.mean(source_lab, dim=(1,2), keepdim=True)
        source_std_lab = torch.std(source_lab, dim=(1,2), keepdim=True)
        source_std_lab[source_std_lab < 1e-5] = 1e-5

    target_lab = rgb_to_lab(target_rgb_prepared.to(current_device))
    target_mean_lab = torch.mean(target_lab, dim=(1,2), keepdim=True)
    target_std_lab = torch.std(target_lab, dim=(1,2), keepdim=True)
    target_std_lab[target_std_lab < 1e-5] = 1e-5

    target_lab_normalized = (target_lab - target_mean_lab) / target_std_lab
    target_lab_transferred = target_lab_normalized * source_std_lab + source_mean_lab
    target_rgb_transferred = lab_to_rgb(target_lab_transferred, clip=True)

    print(f"Color transferred target RGB shape: {target_rgb_transferred.shape}")
    return target_rgb_transferred


def calculate_weighted_lab_stats(rgb_colors, weights):

    if rgb_colors.shape[0] == 0:
        return torch.zeros(3, device=rgb_colors.device), torch.zeros(3, device=rgb_colors.device)

    total_weight = torch.sum(weights)
    if total_weight < 1e-6:

        lab_colors = rgb_to_lab(rgb_colors.permute(1,0).unsqueeze(-1)).squeeze(-1).permute(1,0)
        return torch.mean(lab_colors, dim=0), torch.std(lab_colors, dim=0)


    lab_colors = rgb_to_lab(rgb_colors.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)


    w = weights.unsqueeze(1)


    weighted_mean_lab = torch.sum(lab_colors * w, dim=0) / total_weight


    weighted_variance_lab = torch.sum(w * (lab_colors - weighted_mean_lab.unsqueeze(0))**2, dim=0) / total_weight
    weighted_std_lab = torch.sqrt(weighted_variance_lab)

    return weighted_mean_lab, weighted_std_lab
