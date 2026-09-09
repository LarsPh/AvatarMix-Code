import numpy as np
import plyfile
import os
from loguru import logger


def save_ply_gaussians(ply_path, gaussians, property_names):

    num_gaussians = gaussians['x'].shape[0]

    vertex_data = np.empty(num_gaussians, dtype=[(name, gaussians[name].dtype) for name in property_names])
    for name in property_names:
        vertex_data[name] = gaussians[name]

    el = plyfile.PlyElement.describe(vertex_data, 'vertex')

    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    plyfile.PlyData([el]).write(ply_path)
    logger.info(f"Saved {num_gaussians} Gaussians to {ply_path}")
