import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import shutil
import math

import numpy as np
import pymeshlab
import trimesh
from loguru import logger


_THIS_FILE = Path(__file__).resolve()
_SWAPVTON_ROOT = _THIS_FILE.parents[3]
sys.path.append(str(_SWAPVTON_ROOT))
sys.path.append(str(_SWAPVTON_ROOT / "src"))

from utils.smplx_utils.smplx_models.smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES
from scripts.mesh_transplant.wrist_ring_align import align_hands_mesh_to_body_wrist, write_report_json
from scripts.swapping.data.loaders import load_ply_gaussians, load_embedding_full
from scripts.swapping.data.savers import save_ply_gaussians


DETAILED_SURFACE_LABELS = [
    "torso_skin",
    "head",
    "left_arm",
    "right_arm",
    "left_leg",
    "right_leg",
    "clothes",
    "hands",
    "shoes",
]


def json_sanitize(x, *, max_list_elems: int = 80):
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):
        if x.ndim == 1 and x.size > max_list_elems:
            return {"shape": list(x.shape), "dtype": str(x.dtype), "head": x[:max_list_elems].tolist()}
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_sanitize(v, max_list_elems=max_list_elems) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) > max_list_elems:
            return [json_sanitize(v, max_list_elems=max_list_elems) for v in x[:max_list_elems]] + ["..."]
        return [json_sanitize(v, max_list_elems=max_list_elems) for v in x]
    return str(x)


def _as_np_f32(x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim == 2 and a.shape[0] == 1:
        a = a[0]
    return a.astype(np.float32)


def normalize(v: np.ndarray) -> np.ndarray:
    v = _as_np_f32(v).reshape(-1)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.zeros((3,), dtype=np.float32)
    return (v / n).astype(np.float32)


def _plane_basis_from_axis(axis_dir: np.ndarray, ref_right: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:

    axis_dir = normalize(axis_dir)
    if float(np.linalg.norm(axis_dir)) < 1e-8:
        raise ValueError("axis_dir near zero")

    if ref_right is None:
        ref_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    ref_right = _as_np_f32(ref_right)

    u = ref_right - float(np.dot(ref_right, axis_dir)) * axis_dir
    if float(np.linalg.norm(u)) < 1e-6:
        ref_right2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u = ref_right2 - float(np.dot(ref_right2, axis_dir)) * axis_dir
    u = normalize(u)
    v = normalize(np.cross(axis_dir, u))
    return u.astype(np.float32), v.astype(np.float32)


def axis_cylinder_band_mask(
    V: np.ndarray,
    *,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    radius_m: float,
    band_center_t: float,
    band_halfwidth_m: float,
) -> np.ndarray:

    V = _as_np_f32(V)
    axis_point = _as_np_f32(axis_point)
    axis_dir = normalize(axis_dir)
    if float(np.linalg.norm(axis_dir)) < 1e-8:
        return np.zeros((V.shape[0],), dtype=bool)

    d = V - axis_point[None, :]
    t = (d @ axis_dir).astype(np.float32)
    radial = d - t[:, None] * axis_dir[None, :]
    radial2 = np.sum(radial * radial, axis=1)
    mask = radial2 <= float(radius_m) ** 2
    mask = mask & (np.abs(t - float(band_center_t)) <= float(band_halfwidth_m))
    return mask.astype(bool)


def _point_in_poly_2d(point_uv: np.ndarray, poly_uv: np.ndarray) -> bool:

    p = _as_np_f32(point_uv).reshape(2)
    P = _as_np_f32(poly_uv)
    if P.ndim != 2 or P.shape[1] != 2 or P.shape[0] < 3:
        return False
    x = P[:, 0]
    y = P[:, 1]
    px = float(p[0])
    py = float(p[1])
    inside = False
    j = int(P.shape[0]) - 1
    for i in range(int(P.shape[0])):
        xi = float(x[i])
        yi = float(y[i])
        xj = float(x[j])
        yj = float(y[j])
        intersect = ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return bool(inside)


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(str(path), process=False, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Expected mesh at {path}, got {type(mesh)}")
    return mesh


def save_point_cloud(points: np.ndarray, path: Path) -> None:
    pc = trimesh.points.PointCloud(vertices=_as_np_f32(points))
    pc.export(str(path))


def load_seg_labels_pkl(path: Path) -> np.ndarray:

    if str(path).endswith(".npy"):
        return np.asarray(np.load(str(path))).astype(np.int64)
    if str(path).endswith(".npz"):
        data = np.load(str(path))
        if "labels" in data:
            return np.asarray(data["labels"]).astype(np.int64)
        if "scan_labels" in data:
            return np.asarray(data["scan_labels"]).astype(np.int64)

        for k in data.files:
            return np.asarray(data[k]).astype(np.int64)
        raise ValueError(f"No arrays in npz: {path}")
    if str(path).endswith(".pkl"):
        obj = pickle.loads(Path(path).read_bytes())
        if isinstance(obj, dict):
            if "scan_labels" in obj:
                return np.asarray(obj["scan_labels"]).astype(np.int64)
            if "labels" in obj:
                return np.asarray(obj["labels"]).astype(np.int64)
        arr = np.asarray(obj)
        if arr.ndim == 1:
            return arr.astype(np.int64)
        raise ValueError(f"Unsupported pkl seg format: {path}")
    raise ValueError(f"Unsupported seg format: {path}")


def load_lbs_weights_npy(path: Path) -> np.ndarray:
    w = np.asarray(np.load(str(path))).astype(np.float32)
    if w.ndim != 2:
        raise ValueError(f"weights must be (Nv,J), got {w.shape} from {path}")
    return w


def _opt_path(s: str) -> Optional[Path]:
    s = str(s or "").strip()
    if not s:
        return None
    return Path(s)


def _infer_from_swap_dir(swap_dir: Path) -> Dict[str, Path | None]:


    def pick(*names: str) -> Optional[Path]:
        for n in names:
            p = swap_dir / n
            if p.exists():
                return p
        return None

    inferred: Dict[str, Path | None] = {}
    inferred["hands_mesh_A"] = pick("hands_world_A.obj")
    inferred["body_mesh_A"] = pick(
        "body_no_hands_world_A.obj",
        "body_no_hands_world_A.ply",
        "body_only_world_A.obj",
        "body_only_world_A.ply",
    )
    inferred["head_mesh_A"] = pick("head_plus_neck_world_A.obj", "head_plus_neck_world_A.ply")
    inferred["joints_npz_A"] = pick("smpl_joints_B_world_A.npz", "smpl_joints_A_world_A.npz")
    inferred["body_seg_A"] = pick("label_B_body_no_hands.pkl", "label_B_body_only.pkl")
    inferred["head_seg_A"] = pick("label_A_head_plus_neck.pkl")
    inferred["hands_seg_A"] = pick("label_A_hands.pkl")
    inferred["body_lbs_A"] = pick("smoothed_inpainted_weights_B_body_no_hands.npy", "smoothed_inpainted_weights_B_body_only.npy")
    inferred["head_lbs_A"] = pick("smoothed_inpainted_weights_A_head_plus_neck.npy")
    inferred["hands_lbs_A"] = pick("smoothed_inpainted_weights_A_hands.npy")
    inferred["embedding_head"] = pick("embedding_head_plus_neck.json")
    inferred["embedding_hands"] = pick("embedding_hands.json")
    inferred["embedding_body"] = pick("embedding_body_no_hands.json", "embedding_body_only.json")

    inferred["hands_mesh_B"] = pick("hands_world_B_body_donor.obj") or inferred["hands_mesh_A"]
    inferred["body_mesh_B"] = pick(
        "body_no_hands_world_B_body_donor.obj",
        "body_no_hands_world_B_body_donor.ply",
        "body_only_world_B_body_donor.obj",
        "body_only_world_B_body_donor.ply",
    )
    inferred["head_mesh_B"] = pick("head_plus_neck_world_B_body_donor.obj", "head_plus_neck_world_B_body_donor.ply")
    inferred["joints_npz_B"] = pick("smpl_joints_B_world_B.npz")
    inferred["body_seg_B"] = inferred["body_seg_A"]
    inferred["head_seg_B"] = inferred["head_seg_A"]
    inferred["body_lbs_B"] = inferred["body_lbs_A"]
    inferred["head_lbs_B"] = inferred["head_lbs_A"]
    inferred["embedding_head_B"] = inferred["embedding_head"]
    inferred["embedding_body_B"] = inferred["embedding_body"]
    inferred["hands_seg_B"] = inferred["hands_seg_A"]
    inferred["hands_lbs_B"] = inferred["hands_lbs_A"]
    inferred["embedding_hands_B"] = inferred["embedding_hands"]

    inferred["out_dir_default"] = swap_dir / "neck_bridge"
    return inferred


def load_joints_npz(path: Path, key: str = "joints") -> np.ndarray:
    data = np.load(str(path))
    if key not in data:
        raise KeyError(f"Key '{key}' not in {path}. keys={list(data.keys())}")
    j = _as_np_f32(data[key])
    if j.ndim != 2 or j.shape[1] != 3:
        raise ValueError(f"joints must be (J,3), got {j.shape} from {path}")
    return j


def joint_index(name: str) -> int:
    try:
        return int(SMPLX_JOINT_NAMES.index(name))
    except Exception as e:
        raise KeyError(f"Joint name '{name}' not found in SMPLX_JOINT_NAMES") from e


def build_vertex_adjacency(n_verts: int, faces: np.ndarray) -> List[List[int]]:
    faces = np.asarray(faces).astype(np.int64)
    adj: List[List[int]] = [[] for _ in range(int(n_verts))]
    for a, b, c in faces:
        a = int(a)
        b = int(b)
        c = int(c)
        if 0 <= a < n_verts and 0 <= b < n_verts:
            adj[a].append(b)
            adj[b].append(a)
        if 0 <= b < n_verts and 0 <= c < n_verts:
            adj[b].append(c)
            adj[c].append(b)
        if 0 <= c < n_verts and 0 <= a < n_verts:
            adj[c].append(a)
            adj[a].append(c)
    return adj


def dilate_mask_by_adjacency(mask: np.ndarray, adj: List[List[int]], steps: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    out = mask.copy()
    for _ in range(int(steps)):
        cur = np.where(out)[0].astype(np.int64)
        for v in cur.tolist():
            for nb in adj[int(v)]:
                out[int(nb)] = True
    return out


def submesh_from_face_mask(V: np.ndarray, F: np.ndarray, face_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = _as_np_f32(V)
    F = np.asarray(F).astype(np.int64)
    face_mask = np.asarray(face_mask).astype(bool)
    F2 = F[face_mask]
    used = np.unique(F2.reshape(-1)).astype(np.int64)
    remap = -np.ones((V.shape[0],), dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    V2 = V[used]
    F2 = remap[F2]
    return V2, F2, used


def _ordered_cycles_from_edges(n_verts: int, edges: np.ndarray) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(int(n_verts))]
    for a, b in edges.astype(np.int64):
        if a == b:
            continue
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    visited = np.zeros(int(n_verts), dtype=bool)
    cycles: List[List[int]] = []

    for v0 in range(int(n_verts)):
        if visited[v0] or len(adj[v0]) == 0:
            continue
        stack = [v0]
        comp = []
        visited[v0] = True
        while stack:
            v = stack.pop()
            comp.append(v)
            for nb in adj[v]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)

        start = None
        for v in comp:
            if len(adj[v]) != 2:
                start = v
                break
        if start is None:
            start = comp[0]

        order = [start]
        prev = -1
        cur = start
        for _ in range(len(comp) + 5):
            nbs = adj[cur]
            if len(nbs) == 0:
                break
            if prev < 0:
                nxt = nbs[0]
            else:
                if len(nbs) == 1:
                    nxt = nbs[0]
                else:
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
            if nxt == start or nxt in order:
                break
            order.append(nxt)
            prev, cur = cur, nxt

        if len(order) >= 3:
            cycles.append(order)

    return cycles


def resample_cycle_arclength(P: np.ndarray, N: int) -> np.ndarray:
    P = _as_np_f32(P)
    M = int(P.shape[0])
    if M < 3:
        raise ValueError("Need at least 3 points to resample.")
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if not np.isfinite(total) or total <= 1e-9:
        raise ValueError("Degenerate ring (zero perimeter).")
    t = np.linspace(0.0, total, num=int(N) + 1, endpoint=True)[:-1]
    out = np.zeros((int(N), 3), dtype=np.float32)
    j = 0
    for i, ti in enumerate(t):
        while j + 1 < cum.shape[0] and cum[j + 1] < ti:
            j += 1
        j2 = (j + 1) % M
        t0 = float(cum[j])
        t1 = float(cum[j + 1]) if (j + 1) < cum.shape[0] else total
        if t1 <= t0:
            a = 0.0
        else:
            a = (float(ti) - t0) / (t1 - t0)
        out[i] = (1.0 - a) * P[j % M] + a * P[j2]
    return out


def signed_area_2d(uv: np.ndarray) -> float:
    x = uv[:, 0]
    y = uv[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def ensure_same_winding_uv(Ra: np.ndarray, Rb: np.ndarray, right: np.ndarray, fwd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Ca = Ra.mean(axis=0)
    Cb = Rb.mean(axis=0)
    uva = np.stack([np.dot(Ra - Ca, right), np.dot(Ra - Ca, fwd)], axis=1)
    uvb = np.stack([np.dot(Rb - Cb, right), np.dot(Rb - Cb, fwd)], axis=1)
    if signed_area_2d(uva) * signed_area_2d(uvb) < 0.0:
        Rb = Rb[::-1].copy()
    return Ra, Rb


def best_cyclic_shift(R_src: np.ndarray, R_tgt: np.ndarray) -> int:
    R_src = _as_np_f32(R_src)
    R_tgt = _as_np_f32(R_tgt)
    N = int(R_src.shape[0])
    best_k = 0
    best = float("inf")
    for k in range(N):
        Rt = np.roll(R_tgt, -k, axis=0)
        err = float(np.mean(np.sum((R_src - Rt) ** 2, axis=1)))
        if err < best:
            best = err
            best_k = k
    return int(best_k)


def planar_section_best_ring(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    n: np.ndarray,
    ref_point: np.ndarray,
    *,
    N_resample: int,
    repair_iters: int = 2,
) -> np.ndarray:
    n = normalize(n)
    if float(np.linalg.norm(n)) < 1e-8:
        raise ValueError("Plane normal is near zero; cannot extract ring.")

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))

    for _ in range(max(1, int(repair_iters))):
        try:
            ms.meshing_repair_non_manifold_edges()
        except Exception:
            pass
        try:
            ms.meshing_repair_non_manifold_vertices()
        except Exception:
            pass

    before = int(ms.mesh_number())
    planeoffset = float(np.dot(p0, n))
    ms.generate_polyline_from_planar_section(
        planeaxis="Custom Axis",
        customaxis=n.tolist(),
        planeoffset=planeoffset,
        relativeto="Origin",
        createsectionsurface=False,
    )
    after = int(ms.mesh_number())
    if after <= before:
        raise RuntimeError("PyMeshLab planar section produced no polyline mesh.")

    best_ring = None
    best_score = -float("inf")
    for mi in range(before, after):
        m = ms.mesh(mi)
        v = np.asarray(m.vertex_matrix(), dtype=np.float32)
        e = np.asarray(m.edge_matrix(), dtype=np.int64)
        if v.size == 0 or e.size == 0:
            continue
        cycles = _ordered_cycles_from_edges(v.shape[0], e)
        for cyc in cycles:
            P = v[np.asarray(cyc, dtype=np.int64)]
            if P.shape[0] < 10:
                continue
            per = float(np.sum(np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)))
            center = P.mean(axis=0)
            d = float(np.linalg.norm(center - ref_point))
            score = per - 0.25 * d
            if score > best_score:
                best_score = score
                best_ring = P
    if best_ring is None:
        raise RuntimeError("Failed to extract a usable ring from planar section.")
    best_ring = _orient_ring_to_normal(best_ring, n)
    return resample_cycle_arclength(best_ring, int(N_resample))


def planar_section_extract_loops(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    n: np.ndarray,
    *,
    repair_iters: int = 2,
) -> List[np.ndarray]:

    n = normalize(n)
    if float(np.linalg.norm(n)) < 1e-8:
        return []

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))

    for _ in range(max(1, int(repair_iters))):
        try:
            ms.meshing_repair_non_manifold_edges()
        except Exception:
            pass
        try:
            ms.meshing_repair_non_manifold_vertices()
        except Exception:
            pass

    before = int(ms.mesh_number())
    planeoffset = float(np.dot(_as_np_f32(p0), n))
    ms.generate_polyline_from_planar_section(
        planeaxis="Custom Axis",
        customaxis=n.tolist(),
        planeoffset=planeoffset,
        relativeto="Origin",
        createsectionsurface=False,
    )
    after = int(ms.mesh_number())
    if after <= before:
        return []

    loops: List[np.ndarray] = []
    for mi in range(before, after):
        m = ms.mesh(mi)
        v = np.asarray(m.vertex_matrix(), dtype=np.float32)
        e = np.asarray(m.edge_matrix(), dtype=np.int64)
        if v.size == 0 or e.size == 0:
            continue
        cycles = _ordered_cycles_from_edges(v.shape[0], e)
        for cyc in cycles:
            P = v[np.asarray(cyc, dtype=np.int64)]
            if P.shape[0] >= 10:
                loops.append(P.astype(np.float32))
    return loops


def _select_best_loop(
    loops: List[np.ndarray],
    *,
    axis_point: np.ndarray,
    axis_dir: np.ndarray,
    plane_u: np.ndarray,
    plane_v: np.ndarray,
    prefer_enclose_point: np.ndarray,
) -> Tuple[np.ndarray | None, Dict]:

    axis_point = _as_np_f32(axis_point)
    axis_dir = normalize(axis_dir)
    plane_u = normalize(plane_u)
    plane_v = normalize(plane_v)
    prefer_enclose_point = _as_np_f32(prefer_enclose_point)

    dbg = {"num_loops": int(len(loops)), "candidates": []}
    if not loops:
        return None, dbg

    p_uv = np.array([float(np.dot(prefer_enclose_point, plane_u)), float(np.dot(prefer_enclose_point, plane_v))], dtype=np.float32)

    enriched = []
    for i, P in enumerate(loops):
        P = _as_np_f32(P)
        per = _ring_perimeter(P)
        center = P.mean(axis=0)

        d = center - axis_point
        t = float(np.dot(d, axis_dir))
        radial = d - t * axis_dir
        radial_dist = float(np.linalg.norm(radial))
        uv = np.stack([P @ plane_u, P @ plane_v], axis=1).astype(np.float32)
        encl = _point_in_poly_2d(p_uv, uv)

        score = float(per - 2.0 * radial_dist)
        enriched.append((encl, score, per, radial_dist, i, P))
        dbg["candidates"].append(
            {"i": int(i), "perimeter": float(per), "radial_center_dist": float(radial_dist), "encloses_neck": bool(encl)}
        )

    any_enclose = any(bool(x[0]) for x in enriched)
    if any_enclose:
        enriched = [x for x in enriched if bool(x[0])]

    enriched.sort(key=lambda x: float(x[1]), reverse=True)
    best = enriched[0]
    best_idx = int(best[4])
    dbg["any_enclose"] = bool(any_enclose)
    dbg["chosen_i"] = int(best_idx)
    dbg["chosen_score"] = float(best[1])
    return loops[best_idx], dbg


def _ring_perimeter(P: np.ndarray) -> float:
    P = _as_np_f32(P)
    return float(np.sum(np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)))


def _orient_ring_to_normal(P: np.ndarray, n: np.ndarray) -> np.ndarray:
    P = _as_np_f32(P)
    n = normalize(n)
    C = P.mean(axis=0)
    acc = np.zeros((3,), dtype=np.float32)
    for i in range(int(P.shape[0])):
        a = P[i] - C
        b = P[(i + 1) % int(P.shape[0])] - C
        acc += np.cross(a, b).astype(np.float32)
    if float(np.dot(acc, n)) < 0.0:
        return P[::-1].copy()
    return P


def planar_section_best_ring_search(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    n: np.ndarray,
    ref_point: np.ndarray,
    *,
    N_resample: int,
    repair_iters: int = 2,
    search_range_m: float = 0.03,
    search_steps: int = 6,
    axis_point: np.ndarray | None = None,
    axis_dir: np.ndarray | None = None,
    plane_u: np.ndarray | None = None,
    plane_v: np.ndarray | None = None,
    debug_candidates_ply_path: Path | None = None,
) -> Tuple[np.ndarray, float, Dict]:

    n = normalize(n)
    offsets = np.linspace(-float(search_range_m), float(search_range_m), num=int(2 * search_steps + 1), endpoint=True)
    best = None
    best_score = -float("inf")
    best_t = 0.0
    best_dbg: Dict = {}
    best_loops: List[np.ndarray] = []
    report: Dict = {"offsets": [], "chosen": None}
    for t in offsets.tolist():
        p = (_as_np_f32(p0) + float(t) * n).astype(np.float32)
        try:
            loops = planar_section_extract_loops(mesh, p, n, repair_iters=int(repair_iters))
            if axis_point is None or axis_dir is None or plane_u is None or plane_v is None:

                R = planar_section_best_ring(
                    mesh, p, n, ref_point, N_resample=int(N_resample), repair_iters=int(repair_iters)
                )
                loops_dbg = {"num_loops": 1, "any_enclose": False, "chosen_i": 0, "chosen_score": 0.0}
            else:
                chosen, loops_dbg = _select_best_loop(
                    loops,
                    axis_point=_as_np_f32(axis_point),
                    axis_dir=_as_np_f32(axis_dir),
                    plane_u=_as_np_f32(plane_u),
                    plane_v=_as_np_f32(plane_v),
                    prefer_enclose_point=_as_np_f32(ref_point),
                )
                if chosen is None:
                    raise RuntimeError("no usable loop")
                chosen = _orient_ring_to_normal(chosen, n)
                R = resample_cycle_arclength(chosen, int(N_resample))
        except Exception:
            report["offsets"].append({"t": float(t), "ok": False})
            continue
        per = _ring_perimeter(R)
        center = R.mean(axis=0)
        d = float(np.linalg.norm(center - _as_np_f32(ref_point)))

        score = float(per - 0.25 * d - 1.0 * abs(float(t)))
        report["offsets"].append({"t": float(t), "ok": True, "score": float(score), **(loops_dbg or {})})
        if score > best_score:
            best_score = score
            best = R
            best_t = float(t)
            best_dbg = loops_dbg or {}
            best_loops = loops if "loops" in locals() else []
    if best is None:
        raise RuntimeError("Failed to extract ring even after offset search.")
    report["chosen"] = {"t": float(best_t), "score": float(best_score), **(best_dbg or {})}

    if debug_candidates_ply_path is not None and axis_point is not None and axis_dir is not None:

        try:
            p_best = (_as_np_f32(p0) + float(best_t) * n).astype(np.float32)
            loops_best = planar_section_extract_loops(mesh, p_best, n, repair_iters=int(repair_iters))
            if loops_best:
                pts = np.concatenate([_as_np_f32(P) for P in loops_best], axis=0)
                save_point_cloud(pts, Path(debug_candidates_ply_path))
        except Exception:
            pass

    return best, float(best_t), report


def _maybe_flip_strip_faces_outward_axis(
    V: np.ndarray, F: np.ndarray, *, axis_point: np.ndarray, axis_dir: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    V = _as_np_f32(V)
    F = np.asarray(F, dtype=np.int64)
    axis_point = _as_np_f32(axis_point)
    axis_dir = normalize(axis_dir)
    if float(np.linalg.norm(axis_dir)) < 1e-8:
        return V, F
    tri = V[F]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    fn = np.cross(e1, e2)
    fn_norm = np.linalg.norm(fn, axis=1, keepdims=True)
    fn = fn / np.maximum(fn_norm, 1e-12)
    fc = tri.mean(axis=1)

    v = fc - axis_point[None, :]
    proj = (v @ axis_dir).reshape(-1, 1) * axis_dir.reshape(1, 3)
    radial = v - proj
    rnorm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.maximum(rnorm, 1e-12)
    score = float(np.mean(np.sum(fn * radial, axis=1)))
    if score < 0.0:
        return V, F[:, [0, 2, 1]].copy()
    return V, F


def _weights_to_rgb(w: np.ndarray) -> np.ndarray:

    w = np.asarray(w, dtype=np.float32).reshape(-1)
    w = np.clip(w, 0.0, 1.0)
    r = (255.0 * w).astype(np.uint8)
    g = np.zeros_like(r, dtype=np.uint8)
    b = (255.0 * (1.0 - w)).astype(np.uint8)
    a = np.full_like(r, 255, dtype=np.uint8)
    return np.stack([r, g, b, a], axis=1)


def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:

    h = np.asarray(h, dtype=np.float32)
    s = np.asarray(s, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32)
    f = (h6 - i).astype(np.float32)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = (i % 6).astype(np.int32)
    r = np.zeros_like(v, dtype=np.float32)
    g = np.zeros_like(v, dtype=np.float32)
    b = np.zeros_like(v, dtype=np.float32)
    m0 = i_mod == 0
    m1 = i_mod == 1
    m2 = i_mod == 2
    m3 = i_mod == 3
    m4 = i_mod == 4
    m5 = i_mod == 5
    r[m0], g[m0], b[m0] = v[m0], t[m0], p[m0]
    r[m1], g[m1], b[m1] = q[m1], v[m1], p[m1]
    r[m2], g[m2], b[m2] = p[m2], v[m2], t[m2]
    r[m3], g[m3], b[m3] = p[m3], q[m3], v[m3]
    r[m4], g[m4], b[m4] = t[m4], p[m4], v[m4]
    r[m5], g[m5], b[m5] = v[m5], p[m5], q[m5]
    return np.stack([r, g, b], axis=1)


def _make_joint_palette(K: int) -> np.ndarray:

    K = int(K)
    if K <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    hues = (np.arange(K, dtype=np.float32) * np.float32(0.61803398875)) % np.float32(1.0)
    s = np.full((K,), 0.85, dtype=np.float32)
    v = np.full((K,), 0.95, dtype=np.float32)
    return _hsv_to_rgb(hues, s, v).astype(np.float32)


def _lbs_to_vertex_colors_topk_blend(W: np.ndarray, k: int = 4) -> np.ndarray:

    W = np.asarray(W, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError(f"W must be (N,K), got {W.shape}")
    N, K = int(W.shape[0]), int(W.shape[1])
    if N == 0:
        return np.zeros((0, 4), dtype=np.uint8)
    if K == 0:
        return np.zeros((N, 4), dtype=np.uint8)

    palette = _make_joint_palette(K)
    row_sum = np.maximum(W.sum(axis=1, keepdims=True), 1e-12).astype(np.float32)
    P = (W / row_sum).astype(np.float32)
    k_eff = int(min(max(int(k), 1), K))
    idx = np.argpartition(-P, kth=(k_eff - 1), axis=1)[:, :k_eff]
    w_top = np.take_along_axis(P, idx, axis=1)
    w_top = w_top / np.maximum(w_top.sum(axis=1, keepdims=True), 1e-12)
    rgb = (palette[idx] * w_top[:, :, None]).sum(axis=1)
    rgba = np.concatenate([rgb, np.ones((N, 1), dtype=np.float32)], axis=1)
    rgba_u8 = np.clip(np.round(rgba * 255.0), 0, 255).astype(np.uint8)
    return rgba_u8


def save_lbs_visualizations(
    *,
    combined_mesh: trimesh.Trimesh,
    W_combined: np.ndarray,
    joint_names: List[str],
    out_dir: Path,
    mode: str,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)
    W_combined = np.asarray(W_combined, dtype=np.float32)
    if int(W_combined.shape[0]) != int(combined_mesh.vertices.shape[0]):
        raise ValueError(f"LBS viz Nv mismatch: W {W_combined.shape[0]} vs mesh {len(combined_mesh.vertices)}")


    try:
        C_topk = _lbs_to_vertex_colors_topk_blend(W_combined, k=4)
        m_topk = trimesh.Trimesh(vertices=combined_mesh.vertices, faces=combined_mesh.faces, process=False)
        m_topk.visual.vertex_colors = C_topk
        m_topk.export(str(out_dir / "combined_lbs_topk4_blend_all_joints.ply"))
    except Exception as e:
        logger.warning(f"Failed to write top-k blended LBS viz: {e}")

    jids = []
    kept = []
    for nm in joint_names:
        try:
            jids.append(int(joint_index(str(nm))))
            kept.append(str(nm))
        except Exception:
            continue
    if not jids:
        logger.warning("No valid joints for LBS viz; skipping.")
        return

    if mode in {"rgb", "both"}:
        C = np.zeros((W_combined.shape[0], 4), dtype=np.uint8)
        Wsel = W_combined[:, np.asarray(jids, dtype=np.int64)]
        denom = np.sum(Wsel, axis=1, keepdims=True)
        denom = np.maximum(denom, 1e-12)
        Wrel = Wsel / denom

        rgb = np.zeros((Wrel.shape[0], 3), dtype=np.float32)
        for k in range(min(3, Wrel.shape[1])):
            rgb[:, k] = Wrel[:, k]
        C[:, 0:3] = (255.0 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
        C[:, 3] = 255
        m = trimesh.Trimesh(vertices=combined_mesh.vertices, faces=combined_mesh.faces, process=False)
        m.visual.vertex_colors = C
        m.export(str(out_dir / f"combined_lbs_rgb__{'_'.join(kept[:3])}.ply"))

    if mode in {"per_joint", "both"}:
        for jid, nm in zip(jids, kept):
            w = W_combined[:, int(jid)]
            C = _weights_to_rgb(w)
            m = trimesh.Trimesh(vertices=combined_mesh.vertices, faces=combined_mesh.faces, process=False)
            m.visual.vertex_colors = C
            m.export(str(out_dir / f"combined_lbs_{nm}.ply"))


def _load_embedding_json(path: Path) -> Dict:
    obj = json.loads(Path(path).read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"Embedding json must be dict: {path}")

    for k in ["cano_mesh", "sample_fidxs", "sample_bary", "_xyz", "_rotation"]:
        if k not in obj:
            raise KeyError(f"Missing '{k}' in embedding json: {path}")
    return obj


def merge_split_embeddings(
    *,
    head_embed_path: Path,
    body_embed_path: Path,
    out_path: Path,
    composed_cano_mesh: str,
    body_face_offset: int,
    head_face_offset: int,
    head_first: bool = True,
) -> None:

    body = _load_embedding_json(body_embed_path)
    head = _load_embedding_json(head_embed_path)

    def _as_list(x):
        return x if isinstance(x, list) else list(x)

    body_f = [int(f) + int(body_face_offset) for f in _as_list(body["sample_fidxs"])]
    head_f = [int(f) + int(head_face_offset) for f in _as_list(head["sample_fidxs"])]

    if head_first:
        fidxs = head_f + body_f
        bary = _as_list(head["sample_bary"]) + _as_list(body["sample_bary"])
        xyz = _as_list(head["_xyz"]) + _as_list(body["_xyz"])
        rot = _as_list(head["_rotation"]) + _as_list(body["_rotation"])
        part = (["head_plus_neck"] * len(head_f)) + (["body_only"] * len(body_f))
    else:
        fidxs = body_f + head_f
        bary = _as_list(body["sample_bary"]) + _as_list(head["sample_bary"])
        xyz = _as_list(body["_xyz"]) + _as_list(head["_xyz"])
        rot = _as_list(body["_rotation"]) + _as_list(head["_rotation"])
        part = (["body_only"] * len(body_f)) + (["head_plus_neck"] * len(head_f))

    out: Dict = {
        "cano_mesh": str(composed_cano_mesh),
        "sample_fidxs": fidxs,
        "sample_bary": bary,
        "_xyz": xyz,
        "_rotation": rot,
        "sample_part": part,
        "ordering": "head_then_body" if head_first else "body_then_head",
        "parts": {
            "body_only": {k: v for k, v in body.items() if k not in {"sample_fidxs", "sample_bary", "_xyz", "_rotation"}},
            "head_plus_neck": {k: v for k, v in head.items() if k not in {"sample_fidxs", "sample_bary", "_xyz", "_rotation"}},
        },
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")


def merge_three_embeddings(
    *,
    head_embed_path: Path,
    hands_embed_path: Path,
    body_embed_path: Path,
    out_path: Path,
    composed_cano_mesh: str,
    body_face_offset: int,
    head_face_offset: int,
    hands_face_offset: int,
    ordering: str = "head_then_hands_then_body",
) -> None:

    head = _load_embedding_json(head_embed_path)
    hands = _load_embedding_json(hands_embed_path)
    body = _load_embedding_json(body_embed_path)

    def _as_list(x):
        return x if isinstance(x, list) else list(x)

    body_f = [int(f) + int(body_face_offset) for f in _as_list(body["sample_fidxs"])]
    head_f = [int(f) + int(head_face_offset) for f in _as_list(head["sample_fidxs"])]
    hands_f = [int(f) + int(hands_face_offset) for f in _as_list(hands["sample_fidxs"])]


    fidxs = head_f + hands_f + body_f
    bary = _as_list(head["sample_bary"]) + _as_list(hands["sample_bary"]) + _as_list(body["sample_bary"])
    xyz = _as_list(head["_xyz"]) + _as_list(hands["_xyz"]) + _as_list(body["_xyz"])
    rot = _as_list(head["_rotation"]) + _as_list(hands["_rotation"]) + _as_list(body["_rotation"])
    part = (["head_plus_neck"] * len(head_f)) + (["hands"] * len(hands_f)) + (["body_no_hands"] * len(body_f))

    out: Dict = {
        "cano_mesh": str(composed_cano_mesh),
        "sample_fidxs": fidxs,
        "sample_bary": bary,
        "_xyz": xyz,
        "_rotation": rot,
        "sample_part": part,
        "ordering": str(ordering),
        "parts": {
            "body": {k: v for k, v in body.items() if k not in {"sample_fidxs", "sample_bary", "_xyz", "_rotation"}},
            "head": {k: v for k, v in head.items() if k not in {"sample_fidxs", "sample_bary", "_xyz", "_rotation"}},
            "hands": {k: v for k, v in hands.items() if k not in {"sample_fidxs", "sample_bary", "_xyz", "_rotation"}},
        },
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n")


def build_bridge_strip(
    R0: np.ndarray,
    R1: np.ndarray,
    *,
    target_edge_len: float,
    K_min: int,
    K_max: int,
    smooth_iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    R0 = _as_np_f32(R0)
    R1 = _as_np_f32(R1)
    N = int(R0.shape[0])
    d = float(np.mean(np.linalg.norm(R0 - R1, axis=1)))
    K = int(np.ceil(d / float(target_edge_len)))
    K = int(np.clip(K, int(K_min), int(K_max)))
    rings: List[np.ndarray] = []
    for s in range(K + 2):
        alpha = float(s) / float(K + 1)
        rings.append(((1.0 - alpha) * R0 + alpha * R1).astype(np.float32))

    def smooth_ring(R: np.ndarray, iters: int) -> np.ndarray:
        X = R.copy()
        for _ in range(int(iters)):
            X = 0.5 * X + 0.25 * (np.roll(X, 1, axis=0) + np.roll(X, -1, axis=0))
        return X.astype(np.float32)

    if int(smooth_iters) > 0 and K >= 1:
        for s in range(1, K + 1):
            rings[s] = smooth_ring(rings[s], int(smooth_iters))

    V = np.concatenate(rings, axis=0)
    F = []
    for s in range(K + 1):
        base0 = s * N
        base1 = (s + 1) * N
        for i in range(N):
            i2 = (i + 1) % N
            F.append([base0 + i, base0 + i2, base1 + i])
            F.append([base0 + i2, base1 + i2, base1 + i])
    return V, np.asarray(F, dtype=np.int64), rings


@dataclass(frozen=True)
class FixedLbsSpec:
    joints: Tuple[str, ...] = ("neck", "head", "spine3")
    head_end: Dict[str, float] = None  # type: ignore
    body_end: Dict[str, float] = None  # type: ignore


def fixed_neck_head_patch_weights(
    *,
    J: int,
    rings: List[np.ndarray],
    spec: FixedLbsSpec,
) -> np.ndarray:

    head_end = spec.head_end or {"neck": 0.2, "head": 0.7, "spine3": 0.1}
    body_end = spec.body_end or {"neck": 0.7, "head": 0.1, "spine3": 0.2}
    joint_ids = {name: joint_index(name) for name in spec.joints}
    K = int(len(rings) - 2)
    N = int(rings[0].shape[0])
    W = np.zeros(((K + 2) * N, int(J)), dtype=np.float32)

    def vec_from_dict(d: Dict[str, float]) -> np.ndarray:
        v = np.zeros((int(J),), dtype=np.float32)
        for nm, wt in d.items():
            if nm in joint_ids:
                v[int(joint_ids[nm])] = float(wt)
        s = float(np.sum(np.maximum(v, 0.0)))
        if s <= 1e-12:
            v[int(joint_ids["neck"])] = 1.0
            s = 1.0
        v = v / s
        return v.astype(np.float32)

    w_head = vec_from_dict(head_end)
    w_body = vec_from_dict(body_end)
    for s in range(K + 2):
        alpha = float(s) / float(K + 1)
        w_layer = (1.0 - alpha) * w_head + alpha * w_body
        W[s * N : (s + 1) * N, :] = w_layer[None, :]
    return W


def compute_neck_frame(joints: np.ndarray) -> Dict[str, np.ndarray]:
    j = _as_np_f32(joints)
    J_neck = j[joint_index("neck")]
    J_head = j[joint_index("head")]
    J_spine3 = j[joint_index("spine3")]
    J_lcol = j[joint_index("left_collar")]
    J_rcol = j[joint_index("right_collar")]

    up0 = normalize(J_head - J_neck)
    up1 = normalize(J_neck - J_spine3)
    up = normalize(0.7 * up0 + 0.3 * up1)
    right = normalize(J_rcol - J_lcol)
    right = normalize(right - float(np.dot(right, up)) * up)
    fwd = normalize(np.cross(right, up))

    try:
        J_nose = j[joint_index("nose")]
        if float(np.dot(fwd, J_nose - J_head)) < 0.0:
            fwd = -fwd
            right = -right
    except Exception:
        pass
    return {"up": up, "right": right, "fwd": fwd, "neck": J_neck, "head": J_head}


def build_region_mask(
    mesh: trimesh.Trimesh,
    seg: np.ndarray | None,
    *,
    ref_point: np.ndarray,
    radius_m: float,
    allowed_label_names: List[str] | None,
    dilate_steps: int,
    face_keep_policy: str,
) -> np.ndarray | None:
    V = _as_np_f32(mesh.vertices)
    mask = np.sum((V - ref_point[None, :]) ** 2, axis=1) <= float(radius_m) ** 2
    if seg is not None and allowed_label_names:
        allowed_ids = []
        for nm in allowed_label_names:
            if nm in DETAILED_SURFACE_LABELS:
                allowed_ids.append(int(DETAILED_SURFACE_LABELS.index(nm)))
        if allowed_ids:
            mask = mask & np.isin(np.asarray(seg).astype(np.int64), np.asarray(allowed_ids, dtype=np.int64))
    if int(dilate_steps) > 0:
        adj = build_vertex_adjacency(int(V.shape[0]), np.asarray(mesh.faces, dtype=np.int64))
        mask = dilate_mask_by_adjacency(mask, adj, int(dilate_steps))
    if face_keep_policy not in {"any", "all"}:
        raise ValueError("face_keep_policy must be any|all")
    return mask.astype(bool)


def cropped_mesh_for_ring(mesh: trimesh.Trimesh, vmask: np.ndarray | None, face_keep_policy: str) -> trimesh.Trimesh:
    if vmask is None:
        return mesh
    vmask = np.asarray(vmask).astype(bool)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if face_keep_policy == "any":
        fmask = np.any(vmask[F], axis=1)
    else:
        fmask = np.all(vmask[F], axis=1)
    if not np.any(fmask):
        return mesh
    Vsub, Fsub, _ = submesh_from_face_mask(_as_np_f32(mesh.vertices), F, fmask)
    return trimesh.Trimesh(vertices=Vsub, faces=Fsub, process=False)


def concat_meshes(meshes: List[trimesh.Trimesh]) -> trimesh.Trimesh:
    v_all = []
    f_all = []
    off = 0
    for m in meshes:
        v = np.asarray(m.vertices, dtype=np.float32)
        f = np.asarray(m.faces, dtype=np.int64)
        v_all.append(v)
        f_all.append(f + int(off))
        off += int(v.shape[0])
    V = np.concatenate(v_all, axis=0)
    F = np.concatenate(f_all, axis=0)
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:

    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(np.trace(R))
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / S
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / S
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / S
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n > 1e-12:
        q /= n
    return q.astype(np.float32)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:

    q1 = np.asarray(q1, dtype=np.float32)
    q2 = np.asarray(q2, dtype=np.float32)
    w1, x1, y1, z1 = [q1[..., i] for i in range(4)]
    w2, x2, y2, z2 = [q2[..., i] for i in range(4)]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    out = np.stack([w, x, y, z], axis=-1).astype(np.float32)
    n = np.linalg.norm(out, axis=-1, keepdims=True)
    out = out / np.maximum(n, 1e-12)
    return out.astype(np.float32)


def _find_unique_swapped_ply(swap_dir: Path, *, space_name: str) -> Path | None:
    swap_dir = Path(swap_dir)
    space_name = str(space_name).strip().upper()
    if space_name == "B":
        cands = sorted([p for p in swap_dir.glob("swapped_*direct_B.ply") if p.is_file()])
    else:

        cands = sorted([p for p in swap_dir.glob("swapped_*direct.ply") if p.is_file() and (not p.name.endswith("direct_B.ply"))])
    if len(cands) != 1:
        preview = ", ".join([p.name for p in cands[:5]])
        logger.warning(f"[{space_name}] wrist_align_gaussians: expected 1 swapped PLY, found {len(cands)}. {preview}")
        return None
    return cands[0]


def _maybe_apply_wrist_alignment_to_swapped_gaussians(
    *,
    space_name: str,
    swap_dir: Path | None,
    out_dir: Path,
    hands_mesh: trimesh.Trimesh | None,
    wrist_report: Dict | None,
    embedding_head_json: Path | None,
    embedding_hands_json: Path | None,
    embedding_body_json: Path | None,
    enabled: bool,
    overwrite: bool,
    scale_mode: str,
) -> Dict:

    info: Dict = {"enabled": bool(enabled), "applied": False}
    if not enabled:
        return info
    if swap_dir is None:
        info["status"] = "no_swap_dir"
        return info
    if hands_mesh is None or wrist_report is None:
        info["status"] = "no_hands_or_report"
        return info
    if embedding_head_json is None or embedding_hands_json is None or embedding_body_json is None:
        info["status"] = "missing_embeddings"
        return info

    ply_path = _find_unique_swapped_ply(Path(swap_dir), space_name=space_name)
    if ply_path is None:
        info["status"] = "no_unique_swapped_ply"
        return info


    pairs = wrist_report.get("pairs", []) if isinstance(wrist_report, dict) else []
    comp_T: Dict[int, Dict[str, np.ndarray]] = {}
    for p in pairs:
        try:
            ci = int(p.get("hand_component"))
            s = float(p.get("scale"))
            R = np.asarray(p.get("R"), dtype=np.float32).reshape(3, 3)
            t = np.asarray(p.get("t"), dtype=np.float32).reshape(3,)
            comp_T[ci] = {"scale": np.float32(s), "R": R.astype(np.float32), "t": t.astype(np.float32)}
        except Exception:
            continue
    if not comp_T:
        info["status"] = "no_transforms"
        return info


    head_e = load_embedding_full(str(embedding_head_json))
    hands_e = load_embedding_full(str(embedding_hands_json))
    body_e = load_embedding_full(str(embedding_body_json))
    head_n = int(len(head_e.get("sample_fidxs", [])))
    hands_n = int(len(hands_e.get("sample_fidxs", [])))
    body_n = int(len(body_e.get("sample_fidxs", [])))
    info.update({"head_n": head_n, "hands_n": hands_n, "body_n": body_n})

    gauss, prop_names = load_ply_gaussians(str(ply_path))
    N = int(gauss["x"].shape[0])
    if (head_n + hands_n + body_n) != N:
        info["status"] = "count_mismatch"
        info["gaussians_N"] = N
        return info
    if hands_n <= 0:
        info["status"] = "no_hands_samples"
        return info


    Fh = np.asarray(hands_mesh.faces, dtype=np.int64)
    Nv = int(hands_mesh.vertices.shape[0])
    comps = trimesh.graph.connected_components(hands_mesh.edges, nodes=np.arange(Nv))
    comps = [np.asarray(sorted(list(c)), dtype=np.int64) for c in comps if len(c) > 0]
    comps.sort(key=lambda a: int(a.size), reverse=True)
    vlabel = -np.ones((Nv,), dtype=np.int32)
    for cid, vids in enumerate(comps):
        vlabel[vids] = int(cid)
    face_c = vlabel[Fh]
    face_comp = np.where((face_c[:, 0] == face_c[:, 1]) & (face_c[:, 0] == face_c[:, 2]), face_c[:, 0], -1).astype(np.int32)

    hands_fidxs = np.asarray(hands_e.get("sample_fidxs", []), dtype=np.int64).reshape(-1)
    if hands_fidxs.size != hands_n:
        hands_fidxs = hands_fidxs[:hands_n]
    hands_fidxs = np.clip(hands_fidxs, 0, max(0, int(face_comp.shape[0]) - 1))
    hands_comp = face_comp[hands_fidxs]

    i0 = head_n
    i1 = head_n + hands_n


    if overwrite:
        backup = ply_path.with_name(ply_path.stem + "_old" + ply_path.suffix)
        if not backup.exists():
            shutil.copy2(str(ply_path), str(backup))
            info["backup_ply"] = str(backup)


    x = np.asarray(gauss["x"]).astype(np.float32)
    y = np.asarray(gauss["y"]).astype(np.float32)
    z = np.asarray(gauss["z"]).astype(np.float32)


    mode = str(scale_mode).strip().lower()
    if mode not in {"auto", "log", "linear"}:
        mode = "auto"
    if mode == "auto":
        try:
            s0 = float(np.median(np.asarray(gauss.get("scale_0", np.zeros((N,), dtype=np.float32)))))
            mode = "log" if s0 < 0.0 else "linear"
        except Exception:
            mode = "log"
    info["scale_mode_used"] = mode


    has_rot = all(k in gauss for k in ["rot_0", "rot_1", "rot_2", "rot_3"])
    has_scale = all(k in gauss for k in ["scale_0", "scale_1", "scale_2"])
    if has_rot:
        q = np.stack(
            [
                np.asarray(gauss["rot_0"]).astype(np.float32),
                np.asarray(gauss["rot_1"]).astype(np.float32),
                np.asarray(gauss["rot_2"]).astype(np.float32),
                np.asarray(gauss["rot_3"]).astype(np.float32),
            ],
            axis=1,
        )
    else:
        q = None

    if has_scale:
        sc = np.stack(
            [
                np.asarray(gauss["scale_0"]).astype(np.float32),
                np.asarray(gauss["scale_1"]).astype(np.float32),
                np.asarray(gauss["scale_2"]).astype(np.float32),
            ],
            axis=1,
        )
    else:
        sc = None

    applied_counts: Dict[str, int] = {}
    for cid, T in comp_T.items():
        m = (hands_comp == int(cid))
        idx_local = np.where(m)[0]
        if idx_local.size == 0:
            continue
        idx = (i0 + idx_local).astype(np.int64)
        P = np.stack([x[idx], y[idx], z[idx]], axis=1).astype(np.float32)
        s = float(T["scale"])
        R = np.asarray(T["R"], dtype=np.float32)
        t = np.asarray(T["t"], dtype=np.float32).reshape(1, 3)
        P2 = (s * (P @ R.T) + t).astype(np.float32)
        x[idx], y[idx], z[idx] = P2[:, 0], P2[:, 1], P2[:, 2]

        if sc is not None:
            if mode == "log":
                sc[idx, :] = sc[idx, :] + float(np.log(max(s, 1e-12)))
            else:
                sc[idx, :] = sc[idx, :] * float(s)
        if q is not None:
            qR = _matrix_to_quat_wxyz(R)
            q[idx, :] = _quat_mul_wxyz(np.broadcast_to(qR[None, :], q[idx, :].shape), q[idx, :])
        applied_counts[str(cid)] = int(idx_local.size)


    gauss["x"] = x.astype(np.float32)
    gauss["y"] = y.astype(np.float32)
    gauss["z"] = z.astype(np.float32)
    if sc is not None:
        gauss["scale_0"] = sc[:, 0].astype(np.float32)
        gauss["scale_1"] = sc[:, 1].astype(np.float32)
        gauss["scale_2"] = sc[:, 2].astype(np.float32)
    if q is not None:
        gauss["rot_0"] = q[:, 0].astype(np.float32)
        gauss["rot_1"] = q[:, 1].astype(np.float32)
        gauss["rot_2"] = q[:, 2].astype(np.float32)
        gauss["rot_3"] = q[:, 3].astype(np.float32)

    out_path = ply_path if overwrite else ply_path.with_name(ply_path.stem + "_wrist_aligned" + ply_path.suffix)
    save_ply_gaussians(str(out_path), gauss, prop_names)
    info.update({"applied": True, "swapped_ply_in": str(ply_path), "swapped_ply_out": str(out_path), "hands_comp_counts": applied_counts})


    try:
        emb_path = out_dir / "embedding_composed.json"
        if emb_path.exists():
            emb = json.loads(emb_path.read_text())
            if isinstance(emb, dict) and ("_xyz" in emb) and ("_rotation" in emb) and isinstance(emb["_xyz"], list) and isinstance(emb["_rotation"], list):
                if overwrite:
                    emb_bak = emb_path.with_name(emb_path.stem + "_old" + emb_path.suffix)
                    if not emb_bak.exists():
                        shutil.copy2(str(emb_path), str(emb_bak))
                        info["backup_embedding"] = str(emb_bak)
                P_h = np.stack([x[i0:i1], y[i0:i1], z[i0:i1]], axis=1).astype(np.float32)
                emb["_xyz"][i0:i1] = P_h.tolist()
                if q is not None:
                    emb["_rotation"][i0:i1] = q[i0:i1, :].astype(np.float32).tolist()
                emb_path.write_text(json.dumps(emb, indent=2) + "\n")
                info["embedding_composed_updated"] = str(emb_path)
    except Exception as e:
        info["embedding_update_err"] = str(e)


    try:
        (out_dir / "debug" / "wrist_align_gaussians_report.json").write_text(json.dumps(json_sanitize(info), indent=2) + "\n")
    except Exception:
        pass

    return info

def run_one_space(
    *,
    space_name: str,
    body_mesh_path: Path,
    head_mesh_path: Path,
    joints_npz_path: Path,
    body_seg_path: Path | None,
    head_seg_path: Path | None,
    body_lbs_path: Path | None,
    head_lbs_path: Path | None,
    out_dir: Path,
    swap_dir_for_gaussians: Path | None,
    n_resample: int,
    body_offset_delta: float,
    head_offset_delta: float,
    radius_m: float,
    dilate_steps: int,
    face_keep_policy: str,
    body_allowed_labels: List[str],
    head_allowed_labels: List[str],
    target_edge_len_m: float,
    min_layers: int,
    max_layers: int,
    ring_smooth_iters: int,
    repair_iters: int,
    viz_lbs: bool = False,
    viz_lbs_mode: str = "both",
    viz_lbs_joints: List[str] | None = None,
    axis_mode: str = "hybrid",
    axis_refine_alpha: float = 0.5,
    ring_axis_radius_m: float = 0.14,
    ring_band_halfwidth_m: float = 0.06,
    offset_search_range_m: float = 0.05,
    offset_search_steps: int = 10,
    bridge_mode: str = "tube",
    embedding_head_json: Path | None = None,
    embedding_body_json: Path | None = None,
    hands_mesh_path: Path | None = None,
    hands_seg_path: Path | None = None,
    hands_lbs_path: Path | None = None,
    embedding_hands_json: Path | None = None,
    compose_embedding: bool = True,
    wrist_align_hands: bool = False,
    wrist_align_mode: str = "full",
    wrist_align_resample_N: int = 128,
    wrist_align_scale_min: float = 0.85,
    wrist_align_scale_max: float = 1.18,
    wrist_align_twist_reg_weight: float = 0.02,
    wrist_align_flip_penalty: float = 0.01,
    wrist_align_write_debug: bool = True,
    wrist_align_gaussians: bool = False,
    wrist_align_gaussians_overwrite: bool = True,
    wrist_align_gaussians_scale_mode: str = "auto",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    body_mesh = load_trimesh(body_mesh_path)
    head_mesh = load_trimesh(head_mesh_path)
    hands_mesh = load_trimesh(hands_mesh_path) if hands_mesh_path is not None else None
    joints = load_joints_npz(joints_npz_path, key="joints")
    frame = compute_neck_frame(joints)
    up = frame["up"]
    right = frame["right"]
    fwd = frame["fwd"]
    p_neck = frame["neck"]


    head_center = _as_np_f32(head_mesh.vertices).mean(axis=0)
    if float(np.dot(up, head_center - p_neck)) < 0.0:
        up = -up
        right = -right
        fwd = -fwd


    def _pca_axis_from_vertices(V_local: np.ndarray, ref_axis: np.ndarray) -> np.ndarray:
        V_local = _as_np_f32(V_local)
        if V_local.shape[0] < 20:
            return normalize(ref_axis)
        X = V_local - V_local.mean(axis=0, keepdims=True)
        C = (X.T @ X) / max(1, int(X.shape[0]))
        w, vecs = np.linalg.eigh(C.astype(np.float64))
        axis = vecs[:, int(np.argmax(w))].astype(np.float32)
        axis = normalize(axis)
        if float(np.dot(axis, ref_axis)) < 0.0:
            axis = -axis
        return axis.astype(np.float32)

    axis_dir = normalize(up)
    if str(axis_mode) != "joints":

        Vb = _as_np_f32(body_mesh.vertices)
        Vh = _as_np_f32(head_mesh.vertices)
        wide_mask_b = axis_cylinder_band_mask(
            Vb,
            axis_point=p_neck,
            axis_dir=axis_dir,
            radius_m=float(ring_axis_radius_m),
            band_center_t=0.0,
            band_halfwidth_m=float(ring_band_halfwidth_m) + float(offset_search_range_m) + 0.05,
        )
        wide_mask_h = axis_cylinder_band_mask(
            Vh,
            axis_point=p_neck,
            axis_dir=axis_dir,
            radius_m=float(ring_axis_radius_m),
            band_center_t=0.0,
            band_halfwidth_m=float(ring_band_halfwidth_m) + float(offset_search_range_m) + 0.05,
        )
        if str(axis_mode) == "pca_body":
            V_local = Vb[wide_mask_b]
        elif str(axis_mode) == "pca_head":
            V_local = Vh[wide_mask_h]
        else:
            V_local = np.concatenate([Vb[wide_mask_b], Vh[wide_mask_h]], axis=0) if (wide_mask_b.any() or wide_mask_h.any()) else Vb
        axis_pca = _pca_axis_from_vertices(V_local, axis_dir)
        a = float(np.clip(float(axis_refine_alpha), 0.0, 1.0))
        axis_dir = normalize((1.0 - a) * axis_dir + a * axis_pca)

    plane_u, plane_v = _plane_basis_from_axis(axis_dir, right)

    body_seg = load_seg_labels_pkl(body_seg_path) if body_seg_path else None
    head_seg = load_seg_labels_pkl(head_seg_path) if head_seg_path else None
    hands_seg = load_seg_labels_pkl(hands_seg_path) if hands_seg_path else None
    if body_seg is not None and body_seg.shape[0] != body_mesh.vertices.shape[0]:
        raise ValueError(f"[{space_name}] body_seg Nv mismatch: {body_seg.shape[0]} vs {body_mesh.vertices.shape[0]}")
    if head_seg is not None and head_seg.shape[0] != head_mesh.vertices.shape[0]:
        raise ValueError(f"[{space_name}] head_seg Nv mismatch: {head_seg.shape[0]} vs {head_mesh.vertices.shape[0]}")

    body_w = load_lbs_weights_npy(body_lbs_path)
    head_w = load_lbs_weights_npy(head_lbs_path)
    hands_w = load_lbs_weights_npy(hands_lbs_path) if hands_lbs_path is not None else None
    if body_w.shape[0] != body_mesh.vertices.shape[0]:
        raise ValueError(f"[{space_name}] body_lbs Nv mismatch: {body_w.shape[0]} vs {body_mesh.vertices.shape[0]}")
    if head_w.shape[0] != head_mesh.vertices.shape[0]:
        raise ValueError(f"[{space_name}] head_lbs Nv mismatch: {head_w.shape[0]} vs {head_mesh.vertices.shape[0]}")
    if body_w.shape[1] != head_w.shape[1]:
        raise ValueError(f"[{space_name}] body/head LBS J mismatch: {body_w.shape[1]} vs {head_w.shape[1]}")
    J = int(body_w.shape[1])
    if hands_mesh is not None:
        if hands_w is None:
            raise ValueError(f"[{space_name}] hands_mesh provided but hands_lbs_path is missing.")
        if int(hands_w.shape[0]) != int(hands_mesh.vertices.shape[0]):
            raise ValueError(f"[{space_name}] hands_lbs Nv mismatch: {hands_w.shape[0]} vs {hands_mesh.vertices.shape[0]}")
        if int(hands_w.shape[1]) != int(J):
            raise ValueError(f"[{space_name}] hands LBS J mismatch: {hands_w.shape[1]} vs {J}")
        if hands_seg is not None and int(hands_seg.shape[0]) != int(hands_mesh.vertices.shape[0]):
            raise ValueError(f"[{space_name}] hands_seg Nv mismatch: {hands_seg.shape[0]} vs {hands_mesh.vertices.shape[0]}")


    if str(bridge_mode) == "none":
        wrist_align_applied = False
        wrist_align_report = None
        if hands_mesh is not None and bool(wrist_align_hands):
            try:
                hands_aligned, report = align_hands_mesh_to_body_wrist(
                    hands_mesh=hands_mesh,
                    body_mesh=body_mesh,
                    mode=str(wrist_align_mode),
                    resample_N=int(wrist_align_resample_N),
                    scale_min=float(wrist_align_scale_min),
                    scale_max=float(wrist_align_scale_max),
                    twist_reg_weight=float(wrist_align_twist_reg_weight),
                    flip_penalty=float(wrist_align_flip_penalty),
                )
                hands_mesh = hands_aligned
                wrist_align_applied = True
                wrist_align_report = report
                if bool(wrist_align_write_debug):
                    hands_mesh.export(str(debug_dir / f"hands_world_{space_name}_wrist_aligned.obj"))
                    write_report_json(report, debug_dir / "wrist_align_report.json")
                logger.info(
                    f"[{space_name}] wrist_align_hands: aligned hands mesh using {len(report.get('pairs', []))} pair(s)."
                )
            except Exception as e:
                logger.warning(f"[{space_name}] wrist_align_hands failed; continuing without alignment. err={e}")

        if hands_mesh is not None:
            combined = concat_meshes([body_mesh, head_mesh, hands_mesh])
            combined_name = "combined_body_head_hands_no_bridge.ply"
        else:
            combined = concat_meshes([body_mesh, head_mesh])
            combined_name = "combined_body_head_no_bridge.ply"
        combined.export(str(out_dir / combined_name))

        if body_seg is not None and head_seg is not None and (hands_mesh is None or hands_seg is not None):
            parts = [body_seg.astype(np.int64), head_seg.astype(np.int64)]
            if hands_mesh is not None and hands_seg is not None:
                parts.append(hands_seg.astype(np.int64))
            seg_combined = np.concatenate(parts, axis=0)
            np.save(str(out_dir / "combined_seg.npy"), seg_combined)

        W_parts = [body_w.astype(np.float32), head_w.astype(np.float32)]
        if hands_mesh is not None and hands_w is not None:
            W_parts.append(hands_w.astype(np.float32))
        W_combined = np.concatenate(W_parts, axis=0)
        np.savez(str(out_dir / "combined_lbs_weights.npz"), weights=W_combined.astype(np.float32))

        Nv_b = int(body_mesh.vertices.shape[0])
        Nv_h = int(head_mesh.vertices.shape[0])
        Nv_x = int(hands_mesh.vertices.shape[0]) if hands_mesh is not None else 0
        is_body = np.zeros((Nv_b + Nv_h + Nv_x,), dtype=bool)
        is_head = np.zeros((Nv_b + Nv_h + Nv_x,), dtype=bool)
        is_tube = np.zeros((Nv_b + Nv_h + Nv_x,), dtype=bool)
        is_body[:Nv_b] = True
        is_head[Nv_b : Nv_b + Nv_h] = True
        prov = {"is_body_donor": is_body, "is_head_donor": is_head, "is_neck_tube": is_tube}
        if hands_mesh is not None:
            is_hands = np.zeros((Nv_b + Nv_h + Nv_x,), dtype=bool)
            is_hands[Nv_b + Nv_h :] = True
            prov["is_hands_donor"] = is_hands
        np.savez(str(out_dir / "provenance_onehot.npz"), **prov)

        if bool(viz_lbs):
            joints_for_viz = list(viz_lbs_joints) if viz_lbs_joints is not None else ["head", "neck", "spine3"]
            save_lbs_visualizations(
                combined_mesh=combined,
                W_combined=W_combined,
                joint_names=joints_for_viz,
                out_dir=debug_dir,
                mode=str(viz_lbs_mode),
            )

        if bool(compose_embedding):
            if hands_mesh is not None:
                if (embedding_head_json is not None) and (embedding_body_json is not None) and (embedding_hands_json is not None):
                    merge_three_embeddings(
                        head_embed_path=Path(embedding_head_json),
                        hands_embed_path=Path(embedding_hands_json),
                        body_embed_path=Path(embedding_body_json),
                        out_path=out_dir / "embedding_composed.json",
                        composed_cano_mesh=combined_name,
                        body_face_offset=0,
                        head_face_offset=int(np.asarray(body_mesh.faces).shape[0]),
                        hands_face_offset=int(np.asarray(body_mesh.faces).shape[0]) + int(np.asarray(head_mesh.faces).shape[0]),
                    )
            else:
                if (embedding_head_json is not None) and (embedding_body_json is not None):
                    merge_split_embeddings(
                        head_embed_path=Path(embedding_head_json),
                        body_embed_path=Path(embedding_body_json),
                        out_path=out_dir / "embedding_composed.json",
                        composed_cano_mesh=combined_name,
                        body_face_offset=0,
                        head_face_offset=int(np.asarray(body_mesh.faces).shape[0]),
                        head_first=True,
                    )

        wrist_align_gaussians_info = _maybe_apply_wrist_alignment_to_swapped_gaussians(
            space_name=space_name,
            swap_dir=swap_dir_for_gaussians,
            out_dir=out_dir,
            hands_mesh=hands_mesh,
            wrist_report=wrist_align_report,
            embedding_head_json=Path(embedding_head_json) if embedding_head_json is not None else None,
            embedding_hands_json=Path(embedding_hands_json) if embedding_hands_json is not None else None,
            embedding_body_json=Path(embedding_body_json) if embedding_body_json is not None else None,
            enabled=bool(wrist_align_gaussians),
            overwrite=bool(wrist_align_gaussians_overwrite),
            scale_mode=str(wrist_align_gaussians_scale_mode),
        )

        (out_dir / "run_info.json").write_text(
            json.dumps(
                json_sanitize(
                    {
                        "space": space_name,
                        "bridge_mode": "none",
                        "wrist_align": {
                            "enabled": bool(wrist_align_hands),
                            "applied": bool(wrist_align_applied),
                            "mode": str(wrist_align_mode),
                            "resample_N": int(wrist_align_resample_N),
                            "scale_min": float(wrist_align_scale_min),
                            "scale_max": float(wrist_align_scale_max),
                            "twist_reg_weight": float(wrist_align_twist_reg_weight),
                            "flip_penalty": float(wrist_align_flip_penalty),
                        },
                        "wrist_align_gaussians": wrist_align_gaussians_info,
                        "body_mesh": str(body_mesh_path),
                        "head_mesh": str(head_mesh_path),
                        "hands_mesh": str(hands_mesh_path) if hands_mesh_path is not None else "",
                        "joints_npz": str(joints_npz_path),
                        "neck_frame": {k: v for k, v in frame.items()},
                        "axis_dir": axis_dir,
                    }
                ),
                indent=2,
            )
            + "\n"
        )
        logger.success(f"[{space_name}] Wrote outputs to {out_dir}")
        return


    body_vmask = build_region_mask(
        body_mesh,
        body_seg,
        ref_point=p_neck,
        radius_m=float(radius_m),
        allowed_label_names=body_allowed_labels,
        dilate_steps=int(dilate_steps),
        face_keep_policy=str(face_keep_policy),
    )
    head_vmask = build_region_mask(
        head_mesh,
        head_seg,
        ref_point=p_neck,
        radius_m=float(radius_m),
        allowed_label_names=head_allowed_labels,
        dilate_steps=int(dilate_steps),
        face_keep_policy=str(face_keep_policy),
    )


    band_halfwidth_crop = float(ring_band_halfwidth_m) + float(offset_search_range_m) + 0.01
    Vb = _as_np_f32(body_mesh.vertices)
    Vh = _as_np_f32(head_mesh.vertices)
    cyl_body = axis_cylinder_band_mask(
        Vb,
        axis_point=p_neck,
        axis_dir=axis_dir,
        radius_m=float(ring_axis_radius_m),
        band_center_t=float(body_offset_delta),
        band_halfwidth_m=float(band_halfwidth_crop),
    )
    cyl_head = axis_cylinder_band_mask(
        Vh,
        axis_point=p_neck,
        axis_dir=axis_dir,
        radius_m=float(ring_axis_radius_m),
        band_center_t=float(head_offset_delta),
        band_halfwidth_m=float(band_halfwidth_crop),
    )
    body_vmask2 = cyl_body if body_vmask is None else (np.asarray(body_vmask).astype(bool) & cyl_body)
    head_vmask2 = cyl_head if head_vmask is None else (np.asarray(head_vmask).astype(bool) & cyl_head)

    body_mesh_ring = cropped_mesh_for_ring(body_mesh, body_vmask2, face_keep_policy=str(face_keep_policy))
    head_mesh_ring = cropped_mesh_for_ring(head_mesh, head_vmask2, face_keep_policy=str(face_keep_policy))

    p0_body = (p_neck + float(body_offset_delta) * axis_dir).astype(np.float32)
    p0_head = (p_neck + float(head_offset_delta) * axis_dir).astype(np.float32)


    R_body, t_body, report_body = planar_section_best_ring_search(
        body_mesh_ring,
        p0_body,
        axis_dir,
        p_neck,
        N_resample=int(n_resample),
        repair_iters=int(repair_iters),
        search_range_m=float(offset_search_range_m),
        search_steps=int(offset_search_steps),
        axis_point=p_neck,
        axis_dir=axis_dir,
        plane_u=plane_u,
        plane_v=plane_v,
        debug_candidates_ply_path=debug_dir / "candidates_body.ply",
    )
    R_head, t_head, report_head = planar_section_best_ring_search(
        head_mesh_ring,
        p0_head,
        axis_dir,
        p_neck,
        N_resample=int(n_resample),
        repair_iters=int(repair_iters),
        search_range_m=float(offset_search_range_m),
        search_steps=int(offset_search_steps),
        axis_point=p_neck,
        axis_dir=axis_dir,
        plane_u=plane_u,
        plane_v=plane_v,
        debug_candidates_ply_path=debug_dir / "candidates_head.ply",
    )
    logger.info(
        f"[{space_name}] ring offsets used (along axis): body {float(body_offset_delta)+t_body:+.4f}m, head {float(head_offset_delta)+t_head:+.4f}m"
    )
    (debug_dir / "ring_selection.json").write_text(
        json.dumps(
            json_sanitize(
                {
                    "space": space_name,
                    "axis_mode": str(axis_mode),
                    "axis_refine_alpha": float(axis_refine_alpha),
                    "axis_dir": axis_dir,
                    "ring_axis_radius_m": float(ring_axis_radius_m),
                    "ring_band_halfwidth_m": float(ring_band_halfwidth_m),
                    "offset_search_range_m": float(offset_search_range_m),
                    "offset_search_steps": int(offset_search_steps),
                    "body": report_body,
                    "head": report_head,
                }
            ),
            indent=2,
        )
        + "\n"
    )


    R_head, R_body = ensure_same_winding_uv(R_head, R_body, plane_u, plane_v)
    k = best_cyclic_shift(R_head, R_body)
    R_body_shifted = np.roll(R_body, -int(k), axis=0)

    save_point_cloud(R_body_shifted, debug_dir / "neck_ring_body.ply")
    save_point_cloud(R_head, debug_dir / "neck_ring_head.ply")


    V_patch, F_patch, rings = build_bridge_strip(
        R_head,
        R_body_shifted,
        target_edge_len=float(target_edge_len_m),
        K_min=int(min_layers),
        K_max=int(max_layers),
        smooth_iters=int(ring_smooth_iters),
    )
    V_patch, F_patch = _maybe_flip_strip_faces_outward_axis(V_patch, F_patch, axis_point=p_neck, axis_dir=axis_dir)
    patch_mesh = trimesh.Trimesh(vertices=V_patch, faces=F_patch, process=False)
    patch_mesh.export(str(out_dir / "neck_bridge_patch.ply"))


    combined = concat_meshes([body_mesh, head_mesh, patch_mesh])
    combined_name = "combined_body_head_bridge.ply"
    combined.export(str(out_dir / combined_name))


    if body_seg is not None and head_seg is not None:
        seg_patch = -np.ones((V_patch.shape[0],), dtype=np.int64)
        seg_combined = np.concatenate([body_seg.astype(np.int64), head_seg.astype(np.int64), seg_patch], axis=0)
        np.save(str(out_dir / "combined_seg.npy"), seg_combined)


    if body_lbs_path is not None and head_lbs_path is not None:
        body_w = load_lbs_weights_npy(body_lbs_path)
        head_w = load_lbs_weights_npy(head_lbs_path)
        if int(body_w.shape[0]) != int(body_mesh.vertices.shape[0]):
            raise ValueError(f"[{space_name}] body_lbs Nv mismatch: {body_w.shape[0]} vs body_mesh Nv {len(body_mesh.vertices)}")
        if int(head_w.shape[0]) != int(head_mesh.vertices.shape[0]):
            raise ValueError(f"[{space_name}] head_lbs Nv mismatch: {head_w.shape[0]} vs head_mesh Nv {len(head_mesh.vertices)}")

        W_patch = fixed_neck_head_patch_weights(J=J, rings=rings, spec=FixedLbsSpec())
        W_combined = np.concatenate(
            [body_w.astype(np.float32), head_w.astype(np.float32), W_patch.astype(np.float32)], axis=0
        )
        np.savez(str(out_dir / "combined_lbs_weights.npz"), weights=W_combined.astype(np.float32))

        if bool(viz_lbs):
            joints_for_viz = list(viz_lbs_joints) if viz_lbs_joints is not None else ["head", "neck", "spine3"]
            save_lbs_visualizations(
                combined_mesh=combined,
                W_combined=W_combined,
                joint_names=joints_for_viz,
                out_dir=debug_dir,
                mode=str(viz_lbs_mode),
            )

        if bool(compose_embedding) and (embedding_head_json is not None) and (embedding_body_json is not None):
            merge_split_embeddings(
                head_embed_path=Path(embedding_head_json),
                body_embed_path=Path(embedding_body_json),
                out_path=out_dir / "embedding_composed.json",
                composed_cano_mesh=combined_name,
                body_face_offset=0,
                head_face_offset=int(np.asarray(body_mesh.faces).shape[0]),
                head_first=True,
            )


    Nv_b = int(body_mesh.vertices.shape[0])
    Nv_h = int(head_mesh.vertices.shape[0])
    Nv_p = int(V_patch.shape[0])
    is_body = np.zeros((Nv_b + Nv_h + Nv_p,), dtype=bool)
    is_head = np.zeros((Nv_b + Nv_h + Nv_p,), dtype=bool)
    is_tube = np.zeros((Nv_b + Nv_h + Nv_p,), dtype=bool)
    is_body[:Nv_b] = True
    is_head[Nv_b : Nv_b + Nv_h] = True
    is_tube[Nv_b + Nv_h :] = True
    np.savez(str(out_dir / "provenance_onehot.npz"), is_body_donor=is_body, is_head_donor=is_head, is_neck_tube=is_tube)


    save_point_cloud(np.concatenate([_as_np_f32(r) for r in rings], axis=0), debug_dir / "patch_rings_layers.ply")

    (out_dir / "run_info.json").write_text(
        json.dumps(
            json_sanitize(
                {
                    "space": space_name,
                    "body_mesh": str(body_mesh_path),
                    "head_mesh": str(head_mesh_path),
                    "joints_npz": str(joints_npz_path),
                    "neck_frame": {k: v for k, v in frame.items()},
                    "ring_shift_k": int(k),
                    "patch": {"Nv_patch": int(Nv_p), "K": int(len(rings) - 2), "N": int(n_resample)},
                }
            ),
            indent=2,
        )
        + "\n"
    )

    logger.success(f"[{space_name}] Wrote outputs to {out_dir}")


def main():
    parser = argparse.ArgumentParser("SMPL-X neck bridge patch (no welding; append-only seg/LBS)")
    parser.add_argument("--space", type=str, default="both", choices=["A", "B", "both"])
    parser.add_argument("--swap_dir", type=str, default="", help="If set, infer standard inputs from this directory.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output directory root. If omitted and --swap_dir is set, defaults to <swap_dir>/neck_bridge.",
    )


    parser.add_argument("--body_mesh_A", type=str, default="")
    parser.add_argument("--head_mesh_A", type=str, default="")
    parser.add_argument("--hands_mesh_A", type=str, default="")
    parser.add_argument("--joints_npz_A", type=str, default="")
    parser.add_argument("--body_seg_A", type=str, default="")
    parser.add_argument("--head_seg_A", type=str, default="")
    parser.add_argument("--hands_seg_A", type=str, default="")
    parser.add_argument("--body_lbs_A", type=str, default="")
    parser.add_argument("--head_lbs_A", type=str, default="")
    parser.add_argument("--hands_lbs_A", type=str, default="")
    parser.add_argument("--embedding_head_A", type=str, default="")
    parser.add_argument("--embedding_body_A", type=str, default="")
    parser.add_argument("--embedding_hands_A", type=str, default="")


    parser.add_argument("--body_mesh_B", type=str, default="")
    parser.add_argument("--head_mesh_B", type=str, default="")
    parser.add_argument("--hands_mesh_B", type=str, default="")
    parser.add_argument("--joints_npz_B", type=str, default="")
    parser.add_argument("--body_seg_B", type=str, default="")
    parser.add_argument("--head_seg_B", type=str, default="")
    parser.add_argument("--hands_seg_B", type=str, default="")
    parser.add_argument("--body_lbs_B", type=str, default="")
    parser.add_argument("--head_lbs_B", type=str, default="")
    parser.add_argument("--hands_lbs_B", type=str, default="")
    parser.add_argument("--embedding_head_B", type=str, default="")
    parser.add_argument("--embedding_body_B", type=str, default="")
    parser.add_argument("--embedding_hands_B", type=str, default="")


    parser.add_argument("--n_resample", type=int, default=256)
    parser.add_argument("--body_offset_delta", type=float, default=0.1)
    parser.add_argument("--head_offset_delta", type=float, default=0.05)
    parser.add_argument("--radius_m", type=float, default=0.12)
    parser.add_argument("--dilate_steps", type=int, default=2)
    parser.add_argument("--face_keep_policy", type=str, default="any", choices=["any", "all"])
    parser.add_argument("--body_allowed_labels", nargs="+", default=["torso_skin", "clothes"])

    parser.add_argument("--head_allowed_labels", nargs="+", default=["head", "torso_skin"])
    parser.add_argument("--target_edge_len_m", type=float, default=0.01)
    parser.add_argument("--min_layers", type=int, default=1)
    parser.add_argument("--max_layers", type=int, default=8)
    parser.add_argument("--ring_smooth_iters", type=int, default=5)
    parser.add_argument("--manifold_repair_iters", type=int, default=2)
    parser.add_argument("--axis_mode", type=str, default="hybrid", choices=["joints", "hybrid", "pca_body", "pca_head"])
    parser.add_argument("--axis_refine_alpha", type=float, default=0.5)
    parser.add_argument("--ring_axis_radius_m", type=float, default=0.14)
    parser.add_argument("--ring_band_halfwidth_m", type=float, default=0.06)
    parser.add_argument("--offset_search_range_m", type=float, default=0.05)
    parser.add_argument("--offset_search_steps", type=int, default=10)
    parser.add_argument("--bridge_mode", type=str, default="none", choices=["tube", "none"])
    parser.add_argument("--compose_embedding", action="store_true", default=True, help="If embedding jsons are provided/inferred, write embedding_composed.json.")
    parser.add_argument("--viz_lbs", action="store_true", help="Write debug PLYs with vertex colors for LBS weights.")
    parser.add_argument(
        "--viz_lbs_mode",
        type=str,
        default="both",
        choices=["rgb", "per_joint", "both"],
        help="Which LBS visualizations to write (requires LBS inputs).",
    )
    parser.add_argument(
        "--viz_lbs_joints",
        nargs="+",
        default=["head", "neck", "spine3"],
        help="Joint names to visualize (rgb uses first 3).",
    )


    parser.add_argument(
        "--wrist_align_hands",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When hands mesh is provided and bridge_mode=none, align hands wrist boundary loops to body wrist boundary loops.",
    )
    parser.add_argument(
        "--wrist_align_mode",
        type=str,
        default="full",
        choices=["full", "scale_only"],
        help="Wrist alignment mode: full similarity (s,R,t) or scale_only (match ring size only; forbid rotation optimization).",
    )
    parser.add_argument("--wrist_align_resample_N", type=int, default=128, help="Resample N points per wrist ring for matching.")
    parser.add_argument("--wrist_align_scale_min", type=float, default=0.85, help="Minimum uniform scale allowed during wrist alignment.")
    parser.add_argument("--wrist_align_scale_max", type=float, default=1.18, help="Maximum uniform scale allowed during wrist alignment.")
    parser.add_argument(
        "--wrist_align_twist_reg_weight",
        type=float,
        default=0.02,
        help="Regularize against large wrist twist angles (helps avoid 180-degree palm/back flips on near-circular rings).",
    )
    parser.add_argument(
        "--wrist_align_flip_penalty",
        type=float,
        default=0.01,
        help="Penalty for reversed ring correspondences (discourages mirrored loop matches).",
    )
    parser.add_argument(
        "--wrist_align_write_debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write debug outputs for wrist alignment (aligned hands mesh + report json).",
    )
    parser.add_argument(
        "--wrist_align_gaussians",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also apply the per-hand wrist-alignment similarity transforms to the swapped Gaussian PLY "
            "(hands subset only). Requires --swap_dir and inferred embedding jsons."
        ),
    )
    parser.add_argument(
        "--wrist_align_gaussians_overwrite",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When wrist_align_gaussians is enabled, overwrite the original swapped PLY (after writing a *_old.ply backup). "
            "If false, write a new *_wrist_aligned.ply instead."
        ),
    )
    parser.add_argument(
        "--wrist_align_gaussians_scale_mode",
        type=str,
        default="auto",
        choices=["auto", "log", "linear"],
        help="How to update Gaussian scale_0/1/2 under similarity scale. 'auto' treats negative values as log-scales (default).",
    )

    args = parser.parse_args()
    swap_dir = _opt_path(args.swap_dir)
    inferred: Dict[str, Path | None] = {}
    if swap_dir is not None:
        inferred = _infer_from_swap_dir(swap_dir)

    out_root = Path(args.out_dir) if str(args.out_dir).strip() else (inferred.get("out_dir_default") if inferred else None)
    if out_root is None:
        raise ValueError("Must provide --out_dir, or provide --swap_dir to use the default <swap_dir>/neck_bridge.")

    def _run(space: str):
        if space == "A":
            body_mesh_A = _opt_path(args.body_mesh_A) or inferred.get("body_mesh_A")
            head_mesh_A = _opt_path(args.head_mesh_A) or inferred.get("head_mesh_A")
            hands_mesh_A = _opt_path(args.hands_mesh_A) or inferred.get("hands_mesh_A")
            joints_npz_A = _opt_path(args.joints_npz_A) or inferred.get("joints_npz_A")
            body_seg_A = _opt_path(args.body_seg_A) or inferred.get("body_seg_A")
            head_seg_A = _opt_path(args.head_seg_A) or inferred.get("head_seg_A")
            hands_seg_A = _opt_path(args.hands_seg_A) or inferred.get("hands_seg_A")
            body_lbs_A = _opt_path(args.body_lbs_A) or inferred.get("body_lbs_A")
            head_lbs_A = _opt_path(args.head_lbs_A) or inferred.get("head_lbs_A")
            hands_lbs_A = _opt_path(args.hands_lbs_A) or inferred.get("hands_lbs_A")
            emb_head_A = _opt_path(args.embedding_head_A) or inferred.get("embedding_head")
            emb_body_A = _opt_path(args.embedding_body_A) or inferred.get("embedding_body")
            emb_hands_A = _opt_path(args.embedding_hands_A) or inferred.get("embedding_hands")
            if body_mesh_A is None or head_mesh_A is None or joints_npz_A is None:
                raise ValueError(
                    "Missing required A-space inputs. Provide --swap_dir with expected artifacts, or explicitly set "
                    "--body_mesh_A/--head_mesh_A/--joints_npz_A."
                )
            run_one_space(
                space_name="A",
                body_mesh_path=body_mesh_A,
                head_mesh_path=head_mesh_A,
                joints_npz_path=joints_npz_A,
                body_seg_path=body_seg_A,
                head_seg_path=head_seg_A,
                body_lbs_path=body_lbs_A,
                head_lbs_path=head_lbs_A,
                out_dir=Path(out_root) / "A",
                swap_dir_for_gaussians=swap_dir,
                n_resample=int(args.n_resample),
                body_offset_delta=float(args.body_offset_delta),
                head_offset_delta=float(args.head_offset_delta),
                radius_m=float(args.radius_m),
                dilate_steps=int(args.dilate_steps),
                face_keep_policy=str(args.face_keep_policy),
                body_allowed_labels=list(args.body_allowed_labels),
                head_allowed_labels=list(args.head_allowed_labels),
                target_edge_len_m=float(args.target_edge_len_m),
                min_layers=int(args.min_layers),
                max_layers=int(args.max_layers),
                ring_smooth_iters=int(args.ring_smooth_iters),
                repair_iters=int(args.manifold_repair_iters),
                viz_lbs=bool(args.viz_lbs),
                viz_lbs_mode=str(args.viz_lbs_mode),
                viz_lbs_joints=list(args.viz_lbs_joints),
                axis_mode=str(args.axis_mode),
                axis_refine_alpha=float(args.axis_refine_alpha),
                ring_axis_radius_m=float(args.ring_axis_radius_m),
                ring_band_halfwidth_m=float(args.ring_band_halfwidth_m),
                offset_search_range_m=float(args.offset_search_range_m),
                offset_search_steps=int(args.offset_search_steps),
                bridge_mode=str(args.bridge_mode),
                embedding_head_json=emb_head_A,
                embedding_body_json=emb_body_A,
                hands_mesh_path=hands_mesh_A,
                hands_seg_path=hands_seg_A,
                hands_lbs_path=hands_lbs_A,
                embedding_hands_json=emb_hands_A,
                compose_embedding=bool(args.compose_embedding),
                wrist_align_hands=bool(args.wrist_align_hands),
                wrist_align_mode=str(args.wrist_align_mode),
                wrist_align_resample_N=int(args.wrist_align_resample_N),
                wrist_align_scale_min=float(args.wrist_align_scale_min),
                wrist_align_scale_max=float(args.wrist_align_scale_max),
                wrist_align_twist_reg_weight=float(args.wrist_align_twist_reg_weight),
                wrist_align_flip_penalty=float(args.wrist_align_flip_penalty),
                wrist_align_write_debug=bool(args.wrist_align_write_debug),
                wrist_align_gaussians=bool(args.wrist_align_gaussians),
                wrist_align_gaussians_overwrite=bool(args.wrist_align_gaussians_overwrite),
                wrist_align_gaussians_scale_mode=str(args.wrist_align_gaussians_scale_mode),
            )
        else:
            body_mesh_B = _opt_path(args.body_mesh_B) or inferred.get("body_mesh_B")
            head_mesh_B = _opt_path(args.head_mesh_B) or inferred.get("head_mesh_B")
            hands_mesh_B = _opt_path(args.hands_mesh_B) or inferred.get("hands_mesh_B")
            joints_npz_B = _opt_path(args.joints_npz_B) or inferred.get("joints_npz_B")
            body_seg_B = _opt_path(args.body_seg_B) or inferred.get("body_seg_B")
            head_seg_B = _opt_path(args.head_seg_B) or inferred.get("head_seg_B")
            hands_seg_B = _opt_path(args.hands_seg_B) or inferred.get("hands_seg_B")
            body_lbs_B = _opt_path(args.body_lbs_B) or inferred.get("body_lbs_B")
            head_lbs_B = _opt_path(args.head_lbs_B) or inferred.get("head_lbs_B")
            hands_lbs_B = _opt_path(args.hands_lbs_B) or inferred.get("hands_lbs_B")
            emb_head_B = _opt_path(args.embedding_head_B) or inferred.get("embedding_head_B") or inferred.get("embedding_head")
            emb_body_B = _opt_path(args.embedding_body_B) or inferred.get("embedding_body_B") or inferred.get("embedding_body")
            emb_hands_B = _opt_path(args.embedding_hands_B) or inferred.get("embedding_hands_B") or inferred.get("embedding_hands")
            if body_mesh_B is None or head_mesh_B is None or joints_npz_B is None:
                raise ValueError(
                    "Missing required B-space inputs. Provide --swap_dir with expected artifacts, or explicitly set "
                    "--body_mesh_B/--head_mesh_B/--joints_npz_B."
                )
            run_one_space(
                space_name="B",
                body_mesh_path=body_mesh_B,
                head_mesh_path=head_mesh_B,
                joints_npz_path=joints_npz_B,
                body_seg_path=body_seg_B,
                head_seg_path=head_seg_B,
                body_lbs_path=body_lbs_B,
                head_lbs_path=head_lbs_B,
                out_dir=Path(out_root) / "B",
                swap_dir_for_gaussians=swap_dir,
                n_resample=int(args.n_resample),
                body_offset_delta=float(args.body_offset_delta),
                head_offset_delta=float(args.head_offset_delta),
                radius_m=float(args.radius_m),
                dilate_steps=int(args.dilate_steps),
                face_keep_policy=str(args.face_keep_policy),
                body_allowed_labels=list(args.body_allowed_labels),
                head_allowed_labels=list(args.head_allowed_labels),
                target_edge_len_m=float(args.target_edge_len_m),
                min_layers=int(args.min_layers),
                max_layers=int(args.max_layers),
                ring_smooth_iters=int(args.ring_smooth_iters),
                repair_iters=int(args.manifold_repair_iters),
                viz_lbs=bool(args.viz_lbs),
                viz_lbs_mode=str(args.viz_lbs_mode),
                viz_lbs_joints=list(args.viz_lbs_joints),
                axis_mode=str(args.axis_mode),
                axis_refine_alpha=float(args.axis_refine_alpha),
                ring_axis_radius_m=float(args.ring_axis_radius_m),
                ring_band_halfwidth_m=float(args.ring_band_halfwidth_m),
                offset_search_range_m=float(args.offset_search_range_m),
                offset_search_steps=int(args.offset_search_steps),
                bridge_mode=str(args.bridge_mode),
                embedding_head_json=emb_head_B,
                embedding_body_json=emb_body_B,
                hands_mesh_path=hands_mesh_B,
                hands_seg_path=hands_seg_B,
                hands_lbs_path=hands_lbs_B,
                embedding_hands_json=emb_hands_B,
                compose_embedding=bool(args.compose_embedding),
                wrist_align_hands=bool(args.wrist_align_hands),
                wrist_align_mode=str(args.wrist_align_mode),
                wrist_align_resample_N=int(args.wrist_align_resample_N),
                wrist_align_scale_min=float(args.wrist_align_scale_min),
                wrist_align_scale_max=float(args.wrist_align_scale_max),
                wrist_align_twist_reg_weight=float(args.wrist_align_twist_reg_weight),
                wrist_align_flip_penalty=float(args.wrist_align_flip_penalty),
                wrist_align_write_debug=bool(args.wrist_align_write_debug),
                wrist_align_gaussians=bool(args.wrist_align_gaussians),
                wrist_align_gaussians_overwrite=bool(args.wrist_align_gaussians_overwrite),
                wrist_align_gaussians_scale_mode=str(args.wrist_align_gaussians_scale_mode),
            )

    if args.space in {"A", "both"}:
        _run("A")
    if args.space in {"B", "both"}:
        _run("B")


if __name__ == "__main__":
    main()
