import argparse
import copy
import os
from typing import Dict, Tuple

import numpy as np
import torch

from scripts.ply import read_ply


def load_3dgs_ply(path: str) -> Dict[str, torch.Tensor]:
    pointcloud = read_ply(path)
    fields = pointcloud.dtype.names

    num_points = pointcloud.shape[0]


    means = np.stack([pointcloud['x'], pointcloud['y'], pointcloud['z']], axis=1).astype(np.float32)


    scales = np.stack([
        pointcloud['scale_0'],
        pointcloud['scale_1'],
        pointcloud['scale_2'],
    ], axis=1).astype(np.float32)


    quats = np.stack([
        pointcloud['rot_0'],
        pointcloud['rot_1'],
        pointcloud['rot_2'],
        pointcloud['rot_3'],
    ], axis=1).astype(np.float32)


    features_dc = np.stack([
        pointcloud['f_dc_0'],
        pointcloud['f_dc_1'],
        pointcloud['f_dc_2'],
    ], axis=1).astype(np.float32)


    rest_field_names = [name for name in fields if name.startswith('f_rest_')]
    rest_field_names_sorted = sorted(rest_field_names, key=lambda s: int(s.split('_')[-1]))
    total_rest = len(rest_field_names_sorted)
    if total_rest == 0:

        rest_per_channel = 15
        features_rest = np.zeros((num_points, rest_per_channel, 3), dtype=np.float32)
    else:

        rest_per_channel = total_rest // 3
        features_rest = np.zeros((num_points, rest_per_channel, 3), dtype=np.float32)
        for channel in range(3):
            for i in range(rest_per_channel):
                idx = i + rest_per_channel * channel
                field_name = f'f_rest_{idx}'
                if field_name in fields:
                    features_rest[:, i, channel] = pointcloud[field_name].astype(np.float32)


    opacities = pointcloud['opacity'].astype(np.float32).reshape(num_points, 1)


    return {
        'means': torch.from_numpy(means),
        'scales': torch.from_numpy(scales),
        'quats': torch.from_numpy(quats),
        'features_dc': torch.from_numpy(features_dc),
        'features_rest': torch.from_numpy(features_rest),
        'opacities': torch.from_numpy(opacities),
    }


def align_features_rest_shape(
    features_rest: torch.Tensor,
    expected_rest_dim: int,
) -> torch.Tensor:

    n, current_rest_dim, c = features_rest.shape
    if current_rest_dim == expected_rest_dim:
        return features_rest
    if current_rest_dim > expected_rest_dim:
        return features_rest[:, :expected_rest_dim, :]

    pad = torch.zeros((n, expected_rest_dim - current_rest_dim, c), dtype=features_rest.dtype)
    return torch.cat([features_rest, pad], dim=1)


def get_expected_rest_from_ckpt(ckpt: Dict) -> int:
    pipeline = ckpt.get('pipeline', {})
    key = '_model.gauss_params.features_rest'
    fr = pipeline.get(key)
    if isinstance(fr, torch.Tensor) and fr.ndim == 3:
        return fr.shape[1]

    return 15


def main():
    parser = argparse.ArgumentParser(description='Convert 3DGS PLY to Nerfstudio .ckpt (gaussian params).')
    parser.add_argument('-input_3dgs', required=True, help='Path to input 3DGS .ply file')
    parser.add_argument('-input_nerfstudio', required=True, help='Path to template Nerfstudio .ckpt file')
    parser.add_argument('-output', required=True, help='Path to output .ckpt file')
    args = parser.parse_args()

    print(f"Loading 3DGS PLY from {args.input_3dgs}")
    gauss_from_ply = load_3dgs_ply(args.input_3dgs)

    print(f"Loading Nerfstudio template from {args.input_nerfstudio}")
    template_ckpt = torch.load(args.input_nerfstudio)
    new_ckpt = copy.deepcopy(template_ckpt)


    expected_rest = get_expected_rest_from_ckpt(new_ckpt)
    gauss_from_ply['features_rest'] = align_features_rest_shape(gauss_from_ply['features_rest'], expected_rest)


    print("Updating gaussian parameters in checkpoint")
    for name, tensor in gauss_from_ply.items():
        key = f"_model.gauss_params.{name}"
        new_ckpt.setdefault('pipeline', {})
        new_ckpt['pipeline'][key] = tensor


    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving converted checkpoint to {args.output}")
    torch.save(new_ckpt, args.output)
    print("Conversion complete!")


if __name__ == '__main__':
    main()
