import torch
import pytorch3d.ops

from model.reshaping.reshape_utils import knn_gather, nearest_face_pytorch3d
try:
    import polyscope as ps
    PS_IMPORTED = True
except:
    PS_IMPORTED = False

DEBUG_FACE_AREA = False
DEBUG_BARYCENTRIC_NN = True


def calc_nearest_neighbor(query_pts, ref_v, ref_f, weights, ret_surface_pts=False, method = 'vertex', nn_scale_factor=1.0):

    assert (query_pts.shape[0] == ref_v.shape[0] == ref_f.shape[0])
    surface_pts = None
    if method == 'vertex':
        dists_to_smpl, indices, _ = pytorch3d.ops.knn_points(query_pts, ref_v, K = 1)


        pts_w = pytorch3d.ops.knn_gather(weights, indices)
        pts_w = pts_w[:, :, 0]
    else:
        dists_to_smpl, face_indices, bc_coords = nearest_face_pytorch3d(query_pts, ref_v, ref_f[0], scale_factor=nn_scale_factor)

        face_vertex_ids = torch.gather(ref_f.long(), 1, face_indices[:, :, None].long().expand(-1, -1, 3))

        face_lbs = knn_gather(weights, face_vertex_ids)
        pts_w = (bc_coords[..., None] * face_lbs).sum(2)


        if ret_surface_pts:
            face_vertex_flat = knn_gather(ref_v, face_vertex_ids)
            surface_pts = (bc_coords[..., None] * face_vertex_flat).sum(2)


            dist_ = torch.norm(surface_pts - query_pts, dim=-1)


            if DEBUG_BARYCENTRIC_NN:
                with torch.no_grad():
                    print("--- DEBUGGING PointTriangle3DistanceCoordsForward ---")
                    scale_factor = 1000.0


                    p_unscaled = query_pts[0, 0]
                    face_verts_unscaled = face_vertex_flat[0, 0]

                    p = p_unscaled * scale_factor
                    v0, v1, v2 = face_verts_unscaled * scale_factor

                    print(f"Query point (scaled): {p.cpu().numpy()}")
                    print(f"Triangle vertices (scaled): v0={v0.cpu().numpy()}, v1={v1.cpu().numpy()}, v2={v2.cpu().numpy()}")


                    normal = torch.cross(v2 - v0, v1 - v0)
                    norm_normal = torch.norm(normal)
                    normal = normal / (norm_normal + 1e-8)
                    print(f"triangle normal (scaled): {normal.cpu().numpy()}, norm: {norm_normal.item()}")


                    t = torch.dot(v0 - p, normal)
                    p0 = p + t * normal
                    print(f"Projection distance 't' (scaled): {t.item()}, Projected point p0 (scaled): {p0.cpu().numpy()}")


                    dist_manual_recalc_unscaled = torch.norm(surface_pts[0,0] - p_unscaled)
                    cuda_dist_unscaled = dists_to_smpl[0, 0]
                    print(f"--- DISTANCE COMPARISON ---")
                    print("Distances may be different since we force inside triangle case, but they should be close.")
                    print(f"CUDA kernel returned distance:    {cuda_dist_unscaled.item():.8f}")
                    print(f"Manual recalc from surface pts: {dist_manual_recalc_unscaled.item():.8f}")


                    cuda_dist_sq_scaled = (cuda_dist_unscaled * scale_factor)**2
                    dist_recalc_sq_scaled = t**2

                    print(f"CUDA kernel (d*s)^2:      {cuda_dist_sq_scaled.item():.7f}")
                    print(f"PyTorch recalc (t^2 or edge): {dist_recalc_sq_scaled.item():.7f}")
                    print("----------------------------------------------------")


    return pts_w, surface_pts

def calc_smoothed_nearest_neighbor(query_pts, ref_v, ref_f, weights, n_samples, ret_surface_pts=False, method='barycentric', nn_scale_factor=1.0,
                                    sample_std_scale=1.0, use_distance_weighting=False):

    if method != 'barycentric':
        raise NotImplementedError("Smoothing is only implemented for 'barycentric' method.")

    B, N_query, _ = query_pts.shape


    dists_to_smpl, face_indices_, bc_coords_ = nearest_face_pytorch3d(query_pts, ref_v, ref_f[0], scale_factor=nn_scale_factor)

    std_devs = torch.clamp(dists_to_smpl, min=1e-6).unsqueeze(-1)


    expanded_query_pts = query_pts.unsqueeze(2).expand(B, N_query, n_samples, 3)
    expanded_std_devs = std_devs.unsqueeze(2).expand(B, N_query, n_samples, 3)


    sampled_pts = torch.randn_like(expanded_query_pts) * expanded_std_devs * sample_std_scale + expanded_query_pts
    sampled_pts_flat = sampled_pts.view(B, N_query * n_samples, 3)


    _, face_indices_flat, bc_coords_flat = nearest_face_pytorch3d(sampled_pts_flat, ref_v, ref_f[0], scale_factor=nn_scale_factor)


    face_vertex_ids_flat = torch.gather(ref_f.long(), 1, face_indices_flat[:, :, None].long().expand(-1, -1, 3))
    face_weight_flat = knn_gather(weights, face_vertex_ids_flat)


    sampled_offsets_flat = (bc_coords_flat[..., None] * face_weight_flat).sum(2)
    if ret_surface_pts:

        face_vertex_flat = knn_gather(ref_v, face_vertex_ids_flat)
        surface_pts = (bc_coords_flat[..., None] * face_vertex_flat).sum(2)


        surface_pts = surface_pts.view(B, N_query, n_samples, 3)
    else:
        surface_pts = None


    sampled_offsets = sampled_offsets_flat.view(B, N_query, n_samples, -1)

    if use_distance_weighting:

        distances = torch.norm(sampled_pts - expanded_query_pts, dim=-1)


        normalized_distances = distances


        epsilon = 1e-6
        samples_weights = 1.0 / (normalized_distances + epsilon)
        samples_weights = samples_weights / samples_weights.sum(dim=-1, keepdim=True)

        smoothed_offsets = (sampled_offsets * samples_weights.unsqueeze(-1)).sum(dim=2)

    else:

        smoothed_offsets = torch.mean(sampled_offsets, dim=2)


    if DEBUG_FACE_AREA:
        import trimesh

        ref_faces = ref_f[0].long().cpu().numpy()
        ref_vertices = ref_v[0].cpu().numpy() * 1000.


        mesh = trimesh.trimesh(vertices=ref_vertices, faces=ref_faces)


        triangle_areas = mesh.area_faces

        print(f"reference mesh triangle areas (using trimesh):")
        print(f"  min area: {triangle_areas.min():.6f}")
        print(f"  max area: {triangle_areas.max():.6f}")
        print(f"  mean area: {triangle_areas.mean():.6f}")
        print(f"  std area: {triangle_areas.std():.6f}")
        print(f"  number of very small triangles (< 5e-3): {(triangle_areas < 5e-3).sum()}")
        print(f"  number of zero area triangles: {(triangle_areas < 1e-8).sum()}")
        print(f"  total mesh area: {mesh.area:.6f}")
        print(f"  mesh is watertight: {mesh.is_watertight}")

    return smoothed_offsets, sampled_pts, surface_pts

class NeighborVisialzier():
    def __init__(self):
        if not PS_IMPORTED:
            print("cannot import polyscope, skip nn visualization")
            return None
        ps.init()

    def cleanup(self):
        ps.remove_all_structures()
        ps.remove_all_groups()

    def vis(self, v_ref, f_ref, v_query, f_query, pts_sampled, surface_pts):

        ps.register_surface_mesh("RefMesh", v_ref, f_ref, smooth_shade=True)
        ps.register_surface_mesh("QueryMesh", v_query, f_query, smooth_shade=True)
        n_samples_vis = 2
        random_indices = torch.randint(0, pts_sampled.shape[0], (n_samples_vis,))
        src_pts = pts_sampled if pts_sampled is not None else v_query
        src_verts = v_query if pts_sampled is not None else None
        for i in random_indices:
            color_val = 0.5 + i/n_samples_vis * 0.5
            cur_surface_pts = surface_pts[i] if surface_pts.ndim == 3 else surface_pts[i][None, :]
            cur_src_pts = src_pts[i] if src_pts.ndim == 3 else src_pts[i][None, :]
            edge = cur_surface_pts - cur_src_pts
            ps.register_point_cloud(f"SampledPoints_{i}", cur_src_pts, radius=0.00005, color=[color_val, 0, 0])
            ps.register_point_cloud(f"SurfacePoints_{i}", cur_surface_pts, radius=0.00005, color=[0, color_val, 0])
            if src_verts is not None:
                cur_src_verts = src_verts[i] if src_verts.ndim == 3 else src_verts[i][None, :]
                ps.register_point_cloud(f"SrcPoints_{i}", cur_src_verts, radius=0.00005, color=[color_val, color_val, 0])
            ps.get_point_cloud(f"SampledPoints_{i}").add_vector_quantity(
                f"Edge_{i}",
                edge,
                vectortype="standard",
                color=[0, 0, color_val],

                radius=0.00015,
                )
        ps.show()

        self.cleanup()
