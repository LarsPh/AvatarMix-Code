from __future__ import division, print_function
import time
import torch

import reshape_ops


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


def nearest_face_pytorch3d(points, vertices, faces):

    B, N = points.shape[:2]
    F = faces.shape[0]
    dists, indices, bc_coords = [], [], []
    points = points.contiguous()
    for b in range(B):
        triangles = vertices[b, faces.reshape(-1).to(torch.long)].reshape(F, 3, 3)
        triangles = triangles.contiguous()

        l_idx = torch.tensor([0, ]).to(torch.long).to(points.device)
        dist, index, w0, w1, w2 = reshape_ops.nearest_face_pytorch3d(
            points[b],
            l_idx,
            triangles,
            l_idx,
            N
        )
        dists.append(torch.sqrt(dist))
        indices.append(index)
        bc_coords.append(torch.stack([w0, w1, w2], 1))

    dists = torch.stack(dists, 0)
    indices = torch.stack(indices, 0)
    bc_coords = torch.stack(bc_coords, 0)

    return dists, indices, bc_coords


if __name__ == '__main__':
    import trimesh
    import igl
    mesh = trimesh.load('../../debug/smpl_torch_0.obj', process = False)
    vertices = torch.from_numpy(mesh.vertices).to(torch.float32).cuda()
    faces = torch.from_numpy(mesh.faces).to(torch.int32).cuda()
    query = torch.randn((1024*32, 3)).to(vertices)
    time0 = time.time()
    dist1, face_ids1, closest_pts1 = nearest_face(vertices, faces, query)
    print('Time cost: %f' % (time.time() - time0))


    time1 = time.time()
    dist2, face_ids2, closest_pts2 = igl.signed_distance(query.cpu().numpy(), vertices.cpu().numpy(), faces.cpu().numpy())
    print('Time cost: %f' % (time.time() - time1))


    time2 = time.time()
    dist3, face_ids3, bc_coords = nearest_face_pytorch3d(query[None], vertices, faces)
    print('Time cost: %f' % (time.time() - time2))
