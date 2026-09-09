import torch
from loguru import logger
try:
    import reshape_ops
except Exception as e:
    logger.warning(f"reshape_ops not found with error: {e}, nearest face related operations will not be available")
    reshape_ops = None

def knn_gather(x, idx):

    C = x.shape[-1]
    B, N, K = idx.shape
    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, C)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    return x_out

def nearest_face(vertices, faces, query_points):
    vertices = vertices.contiguous().to(torch.float32)
    faces = faces.contiguous().to(torch.int32)
    query_points = query_points.contiguous().to(torch.float32)
    query_num = query_points.size(0)
    dist = torch.cuda.FloatTensor(query_num).fill_(0.0).contiguous()
    face_ids = torch.cuda.IntTensor(query_num).fill_(-1).contiguous()
    nearest_pts = torch.cuda.FloatTensor(query_num, 3).fill_(0.0).contiguous()
    reshape_ops.nearest_face(vertices, faces, query_points, dist, face_ids, nearest_pts)
    return dist, face_ids, nearest_pts


def nearest_face_pytorch3d(points, vertices, faces, scale_factor=1.0):


    B, N = points.shape[:2]
    F = faces.shape[0]
    dists, indices, bc_coords = [], [], []


    points_scaled = (points * scale_factor).contiguous()
    vertices_scaled = (vertices * scale_factor).contiguous()

    for b in range(B):
        triangles = vertices_scaled[b, faces.reshape(-1).to(torch.long)].reshape(F, 3, 3)
        triangles = triangles.contiguous()

        l_idx = torch.tensor([0, ]).to(torch.long).to(points.device)
        dist, index, w0, w1, w2 = reshape_ops.nearest_face_pytorch3d(
            points_scaled[b],
            l_idx,
            triangles,
            l_idx,
            N,
            5e-3
        )

        dists.append(torch.sqrt(dist) / scale_factor)
        indices.append(index)
        bc_coords.append(torch.stack([w0, w1, w2], 1))

    dists = torch.stack(dists, 0)
    indices = torch.stack(indices, 0)
    bc_coords = torch.stack(bc_coords, 0)

    return dists, indices, bc_coords
