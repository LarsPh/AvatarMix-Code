import torch
import pytorch3d.transforms


def transform_to_canonical_space(verts_world, src_global_orient, src_transl, src_scale, j0_src):


    verts_centered = verts_world - src_transl

    verts_scaled = verts_centered / src_scale

    global_orient_matrix = pytorch3d.transforms.axis_angle_to_matrix(src_global_orient)
    global_orient_inverse = global_orient_matrix.transpose(1, 2)

    verts_canonical = (global_orient_inverse @ (verts_scaled.unsqueeze(-1) - j0_src.unsqueeze(-1))).squeeze(-1)
    verts_canonical = verts_canonical + j0_src
    return verts_canonical


def transform_to_world_space(verts_canonical, src_global_orient, src_transl, src_scale, j0_src):


    global_orient_matrix = pytorch3d.transforms.axis_angle_to_matrix(src_global_orient)
    verts_rotated = (global_orient_matrix @ (verts_canonical.unsqueeze(-1) - j0_src.unsqueeze(-1))).squeeze(-1)
    verts_rotated = verts_rotated + j0_src

    verts_scaled = verts_rotated * src_scale

    verts_world = verts_scaled + src_transl
    return verts_world
