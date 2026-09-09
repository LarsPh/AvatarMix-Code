import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pymeshlab
import trimesh
from loguru import logger


_THIS_FILE = Path(__file__).resolve()
_SWAPVTON_ROOT = _THIS_FILE.parents[3]
sys.path.append(str(_SWAPVTON_ROOT))
sys.path.append(str(_SWAPVTON_ROOT / "src"))

from utils.smplx_utils.smplx_models.smplx.joint_names import JOINT_NAMES as SMPLX_JOINT_NAMES


SURFACE_LABELS = ["skin", "hair", "shoe", "upper", "lower", "outer"]

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


def load_smplx_vert_segmentation(json_path: Path) -> Dict[str, List[int]]:
    seg = json.loads(Path(json_path).read_text())
    if not isinstance(seg, dict):
        raise ValueError(f"Invalid SMPL-X vert segmentation JSON: {json_path}")
    out: Dict[str, List[int]] = {}
    for k, v in seg.items():
        if isinstance(v, list):
            out[str(k)] = [int(x) for x in v]
    return out


def json_sanitize(x, *, max_list_elems: int = 50):

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.ndarray,)):

        if x.ndim == 1 and x.size > max_list_elems:
            return {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "head": x[:max_list_elems].tolist(),
            }
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): json_sanitize(v, max_list_elems=max_list_elems) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        if len(x) > max_list_elems:
            return [json_sanitize(v, max_list_elems=max_list_elems) for v in x[:max_list_elems]] + ["..."]
        return [json_sanitize(v, max_list_elems=max_list_elems) for v in x]
    return str(x)


@dataclass(frozen=True)
class RingAlignGrid:
    enabled: bool = True
    yaw_deg_max: float = 5.0
    yaw_samples: int = 11
    trans_xy_max: float = 0.01
    trans_samples: int = 7
    scale_enabled: bool = True
    scale_min: float = 0.98
    scale_max: float = 1.02
    scale_samples: int = 5
    reg_lambda: float = 1e-2


def _as_np_f32(x) -> np.ndarray:
    a = np.asarray(x)
    return a.astype(np.float32)


def build_vertex_adjacency(n_verts: int, faces: np.ndarray) -> List[List[int]]:
    faces = np.asarray(faces).astype(np.int64)
    adj: List[set] = [set() for _ in range(int(n_verts))]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].add(b)
        adj[a].add(c)
        adj[b].add(a)
        adj[b].add(c)
        adj[c].add(a)
        adj[c].add(b)
    return [list(s) for s in adj]


def dilate_mask_by_adjacency(mask: np.ndarray, adj: List[List[int]], steps: int) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if steps <= 0:
        return mask
    cur = mask.copy()
    frontier = np.where(cur)[0].astype(np.int64)
    for _ in range(int(steps)):
        nxt = cur.copy()
        for v in frontier:
            for nb in adj[int(v)]:
                nxt[int(nb)] = True
        new_frontier = np.where(nxt & (~cur))[0].astype(np.int64)
        cur = nxt
        frontier = new_frontier
        if frontier.size == 0:
            break
    return cur


def split_hands_into_sides_by_wrist_distance(
    hand_mask: np.ndarray, vertices: np.ndarray, jw_left: np.ndarray, jw_right: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    hand_mask = np.asarray(hand_mask).astype(bool)
    V = _as_np_f32(vertices)
    idx = np.where(hand_mask)[0]
    if idx.size == 0:
        return np.zeros_like(hand_mask, dtype=bool), np.zeros_like(hand_mask, dtype=bool)
    P = V[idx]
    dl = np.sum((P - jw_left[None, :]) ** 2, axis=1)
    dr = np.sum((P - jw_right[None, :]) ** 2, axis=1)
    left_idx = idx[dl <= dr]
    right_idx = idx[dl > dr]
    mL = np.zeros_like(hand_mask, dtype=bool)
    mR = np.zeros_like(hand_mask, dtype=bool)
    mL[left_idx] = True
    mR[right_idx] = True
    return mL, mR


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = _as_np_f32(v)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return np.zeros_like(v, dtype=np.float32)
    return (v / n).astype(np.float32)


def plane_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = normalize(n)
    if float(np.linalg.norm(n)) < 1e-8:
        n = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    e1 = normalize(np.cross(n, a))
    e2 = normalize(np.cross(n, e1))
    return e1, e2, n


def signed_area_2d(uv: np.ndarray) -> float:
    u = uv[:, 0]
    v = uv[:, 1]
    u2 = np.roll(u, -1)
    v2 = np.roll(v, -1)
    return float(0.5 * np.sum(u * v2 - v * u2))


def load_trimesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load_mesh(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):

        if hasattr(mesh, "dump"):
            parts = mesh.dump()
            if not parts:
                raise ValueError(f"Failed loading mesh (empty scene): {path}")
            mesh = trimesh.util.concatenate(parts)
        else:
            raise ValueError(f"Unsupported mesh type loaded from {path}: {type(mesh)}")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"Empty mesh loaded from {path}")
    return mesh


def load_body_part_labels(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        labels = np.load(str(path))
        labels = np.asarray(labels).astype(np.int64)
        return labels
    if path.suffix.lower() == ".pkl":
        import pickle as pkl

        with open(path, "rb") as f:
            data = pkl.load(f)
        if "scan_labels" not in data:
            raise KeyError(f"'scan_labels' not found in {path}")
        labels = np.asarray(data["scan_labels"]).astype(np.int64)
        return labels
    raise ValueError(f"Unsupported labels format: {path}")


def load_lbs_weights(path: Path) -> np.ndarray:
    w = np.load(str(path))
    w = np.asarray(w).astype(np.float32)
    if w.ndim != 2:
        raise ValueError(f"lbs weights must be 2D (Nv,J), got {w.shape} from {path}")
    return w


def load_smplx_joints(path: Path) -> np.ndarray:
    j = np.load(str(path))
    j = np.asarray(j).astype(np.float32)
    if j.ndim == 3 and j.shape[0] == 1:
        j = j[0]
    if j.ndim != 2 or j.shape[1] != 3:
        raise ValueError(f"smpl joints must be (J,3) or (1,J,3), got {j.shape} from {path}")
    return j


def joint_index(name: str) -> int:
    try:
        return int(SMPLX_JOINT_NAMES.index(name))
    except Exception as e:
        raise KeyError(f"Joint name '{name}' not found in SMPLX_JOINT_NAMES") from e


def compute_wrist_plane(joints: np.ndarray, side: str, wrist_plane_offset: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    side = str(side).lower()
    if side not in {"left", "right"}:
        raise ValueError(f"side must be 'left' or 'right', got {side}")
    wrist_name = f"{side}_wrist"
    elbow_name = f"{side}_elbow"
    jw = joints[joint_index(wrist_name)]
    je = joints[joint_index(elbow_name)]
    axis_forearm = normalize(jw - je)
    n = axis_forearm
    p0 = jw + float(wrist_plane_offset) * axis_forearm
    return p0.astype(np.float32), n.astype(np.float32), jw.astype(np.float32)


def _ordered_cycles_from_edges(n_verts: int, edges: np.ndarray) -> List[List[int]]:

    adj: List[List[int]] = [[] for _ in range(int(n_verts))]
    for a, b in edges.astype(np.int64):
        if a == b:
            continue
        adj[a].append(int(b))
        adj[b].append(int(a))

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
            nxt = None
            if len(nbs) == 0:
                break
            if prev < 0:
                nxt = nbs[0]
            else:

                if len(nbs) == 1:
                    nxt = nbs[0]
                else:
                    nxt = nbs[0] if nbs[1] == prev else nbs[1]
            if nxt == start:
                break
            if nxt in order:
                break
            order.append(nxt)
            prev, cur = cur, nxt

        if len(order) >= 3:
            cycles.append(order)

    return cycles


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


    def _log_ms(msg: str) -> None:
        try:
            m = ms.current_mesh()
            logger.info(
                f"[planar_section repair] {msg}: v={int(m.vertex_number())} f={int(m.face_number())}"
            )
        except Exception:
            logger.info(f"[planar_section repair] {msg}")

    _log_ms("initial")


    for it in range(max(1, int(repair_iters))):
        try:
            ms.meshing_repair_non_manifold_edges()
            _log_ms(f"iter={it} after repair_non_manifold_edges")
        except Exception as e:
            logger.warning(f"[planar_section repair] iter={it} repair_non_manifold_edges failed: {e}")

        try:

            ms.meshing_repair_non_manifold_vertices()
            _log_ms(f"iter={it} after repair_non_manifold_vertices")
        except Exception as e:
            logger.warning(
                f"[planar_section repair] iter={it} repair_non_manifold_vertices failed: {e}"
            )

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
        raise RuntimeError("Failed to extract a usable wrist ring from planar section.")

    return resample_cycle_arclength(best_ring, int(N_resample))


def _boundary_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces).astype(np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must be (F,3), got {faces.shape}")
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    uniq, counts = np.unique(e, axis=0, return_counts=True)
    b = uniq[counts == 1]
    return b.astype(np.int64)


def boundary_ring_best_from_cut(
    vertices: np.ndarray,
    faces: np.ndarray,
    p0: np.ndarray,
    n: np.ndarray,
    ref_point: np.ndarray,
    *,
    N_resample: int,
) -> np.ndarray:
    V = _as_np_f32(vertices)
    F = np.asarray(faces).astype(np.int64)
    n = normalize(n)
    be = _boundary_edges_from_faces(F)
    if be.size == 0:
        raise RuntimeError("No boundary edges found on cut mesh (cannot infer ring).")
    cycles = _ordered_cycles_from_edges(V.shape[0], be)
    if not cycles:
        raise RuntimeError("Failed to trace any boundary cycles on cut mesh.")

    best_ring = None
    best_score = -float("inf")
    for cyc in cycles:
        P = V[np.asarray(cyc, dtype=np.int64)]
        if P.shape[0] < 10:
            continue
        per = float(np.sum(np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)))
        center = P.mean(axis=0)
        d = float(np.linalg.norm(center - ref_point))
        plane_mean_abs = float(np.mean(np.abs((P - p0[None, :]) @ n)))

        score = per - 0.25 * d - 10.0 * plane_mean_abs
        if score > best_score:
            best_score = score
            best_ring = P

    if best_ring is None:
        raise RuntimeError("No suitable boundary ring found (all cycles too small).")
    return resample_cycle_arclength(best_ring, int(N_resample))


def robust_wrist_ring(
    mesh: trimesh.Trimesh,
    p0: np.ndarray,
    n: np.ndarray,
    ref_point: np.ndarray,
    *,
    N_resample: int,
    cut_keep_negative_side: bool,
    repair_iters: int = 2,
    region_vertex_mask: np.ndarray | None = None,
) -> np.ndarray:

    mesh_for_ring = mesh
    if region_vertex_mask is not None:
        vmask = np.asarray(region_vertex_mask).astype(bool)
        if vmask.shape[0] != mesh.vertices.shape[0]:
            raise ValueError("region_vertex_mask shape mismatch with mesh vertices")
        face_mask = np.all(vmask[np.asarray(mesh.faces, dtype=np.int64)], axis=1)
        if not np.any(face_mask):
            raise RuntimeError("region_vertex_mask selects no faces; cannot extract ring")
        Vsub, Fsub, _ = submesh_from_face_mask(_as_np_f32(mesh.vertices), np.asarray(mesh.faces, dtype=np.int64), face_mask)
        mesh_for_ring = trimesh.Trimesh(vertices=Vsub, faces=Fsub, process=False)


    try:
        return planar_section_best_ring(mesh_for_ring, p0, n, ref_point, N_resample=int(N_resample), repair_iters=int(repair_iters))
    except Exception as e:
        msg = str(e)
        logger.warning(f"Planar section failed ({msg}). Falling back to boundary-ring-from-cut.")


    Vc, Fc, _ = cut_mesh_by_plane(
        _as_np_f32(mesh_for_ring.vertices),
        np.asarray(mesh_for_ring.faces, dtype=np.int64),
        p0=p0,
        n=n,
        keep_negative_side=bool(cut_keep_negative_side),
        face_mode="any",
    )
    return boundary_ring_best_from_cut(Vc, Fc, p0, n, ref_point, N_resample=int(N_resample))


def resample_cycle_arclength(P: np.ndarray, N: int) -> np.ndarray:
    P = _as_np_f32(P)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"P must be (M,3), got {P.shape}")
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

    for i, ti in enumerate(t):

        idx = int(np.searchsorted(cum, ti, side="right") - 1)
        idx = max(0, min(idx, M - 1))
        t0 = float(cum[idx])
        t1 = float(cum[idx + 1]) if idx + 1 < cum.shape[0] else total
        a = 0.0 if t1 <= t0 else float((ti - t0) / (t1 - t0))
        p0 = P[idx]
        p1 = P[(idx + 1) % M]
        out[i] = (1.0 - a) * p0 + a * p1
    return out


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


def ensure_same_winding(Ra: np.ndarray, Rb: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    e1, e2, _ = plane_basis_from_normal(n)
    Ca = Ra.mean(axis=0)
    Cb = Rb.mean(axis=0)
    uva = np.stack([np.dot(Ra - Ca, e1), np.dot(Ra - Ca, e2)], axis=1)
    uvb = np.stack([np.dot(Rb - Cb, e1), np.dot(Rb - Cb, e2)], axis=1)
    sa = signed_area_2d(uva)
    sb = signed_area_2d(uvb)
    if sa * sb < 0.0:

        Rb = Rb[::-1].copy()
    return Ra, Rb


def apply_rigid_in_plane_transform(
    P: np.ndarray,
    *,
    origin: np.ndarray,
    n: np.ndarray,
    trans_e1: float,
    trans_e2: float,
    yaw_rad: float,
    scale: float,
) -> np.ndarray:
    e1, e2, n = plane_basis_from_normal(n)

    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))

    K = np.array(
        [[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]],
        dtype=np.float32,
    )
    R = (np.eye(3, dtype=np.float32) * c) + (1.0 - c) * np.outer(n, n).astype(np.float32) + s * K

    T = float(trans_e1) * e1 + float(trans_e2) * e2
    P0 = P - origin[None, :]
    P1 = (scale * (P0 @ R.T)).astype(np.float32) + origin[None, :] + T[None, :]
    return P1


def align_hand_ring_grid(
    Rh: np.ndarray,
    Rb: np.ndarray,
    n: np.ndarray,
    *,
    origin: np.ndarray,
    grid: RingAlignGrid,
) -> Tuple[np.ndarray, int, Dict[str, float]]:
    if not grid.enabled:
        k0 = best_cyclic_shift(Rh, Rb)
        return Rh, k0, {"yaw_deg": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0, "E": float("nan")}

    yaw_list = np.linspace(-grid.yaw_deg_max, grid.yaw_deg_max, int(grid.yaw_samples), dtype=np.float32)
    t_list = np.linspace(-grid.trans_xy_max, grid.trans_xy_max, int(grid.trans_samples), dtype=np.float32)
    if grid.scale_enabled:
        s_list = np.linspace(grid.scale_min, grid.scale_max, int(grid.scale_samples), dtype=np.float32)
    else:
        s_list = np.array([1.0], dtype=np.float32)

    best_E = float("inf")
    best_params = {"yaw_deg": 0.0, "tx": 0.0, "ty": 0.0, "scale": 1.0}
    best_Rh = Rh
    best_k = 0

    for yaw_deg in yaw_list:
        yaw_rad = float(np.deg2rad(float(yaw_deg)))
        for tx in t_list:
            for ty in t_list:
                for sc in s_list:
                    Rh_t = apply_rigid_in_plane_transform(
                        Rh,
                        origin=origin,
                        n=n,
                        trans_e1=float(tx),
                        trans_e2=float(ty),
                        yaw_rad=yaw_rad,
                        scale=float(sc),
                    )
                    k = best_cyclic_shift(Rh_t, Rb)
                    Rb_k = np.roll(Rb, -k, axis=0)
                    data = Rh_t - Rb_k
                    E = float(np.mean(np.sum(data * data, axis=1)))
                    reg = float(grid.reg_lambda) * (float(tx) ** 2 + float(ty) ** 2 + yaw_rad**2 + float(np.log(float(sc))) ** 2)
                    E2 = E + reg
                    if E2 < best_E:
                        best_E = E2
                        best_params = {"yaw_deg": float(yaw_deg), "tx": float(tx), "ty": float(ty), "scale": float(sc)}
                        best_Rh = Rh_t
                        best_k = int(k)

    best_params["E"] = float(best_E)
    return best_Rh, best_k, best_params


def cut_mesh_by_plane(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    p0: np.ndarray,
    n: np.ndarray,
    keep_negative_side: bool,
    face_mode: str,
    region_vertex_mask: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    vertices = _as_np_f32(vertices)
    faces = np.asarray(faces).astype(np.int64)
    n = normalize(n)
    s = (vertices - p0[None, :]) @ n
    keep_v = s <= 0.0 if keep_negative_side else s >= 0.0
    if region_vertex_mask is not None:
        region_vertex_mask = np.asarray(region_vertex_mask).astype(bool)
        if region_vertex_mask.shape[0] != vertices.shape[0]:
            raise ValueError("region_vertex_mask shape mismatch with vertices")
        region_face = np.all(region_vertex_mask[faces], axis=1)
    else:
        region_face = np.ones((faces.shape[0],), dtype=bool)
    if face_mode == "all":
        remove_f = region_face & np.all(~keep_v[faces], axis=1)
    elif face_mode == "any":
        remove_f = region_face & np.any(~keep_v[faces], axis=1)
    else:
        raise ValueError(f"face_mode must be 'all' or 'any', got {face_mode}")
    keep_f = ~remove_f
    faces_keep = faces[keep_f]
    if faces_keep.size == 0:
        raise ValueError("Plane cut removed all faces; adjust offset/sign.")
    kept_vids = np.unique(faces_keep.reshape(-1))
    remap = -np.ones((vertices.shape[0],), dtype=np.int64)
    remap[kept_vids] = np.arange(kept_vids.shape[0], dtype=np.int64)
    new_vertices = vertices[kept_vids]
    new_faces = remap[faces_keep]
    return new_vertices, new_faces, kept_vids


def submesh_from_face_mask(vertices: np.ndarray, faces: np.ndarray, face_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    V = _as_np_f32(vertices)
    F = np.asarray(faces).astype(np.int64)
    face_mask = np.asarray(face_mask).astype(bool)
    Fk = F[face_mask]
    if Fk.size == 0:
        raise ValueError("submesh_from_face_mask: empty face selection")
    old_vids = np.unique(Fk.reshape(-1))
    remap = -np.ones((V.shape[0],), dtype=np.int64)
    remap[old_vids] = np.arange(old_vids.shape[0], dtype=np.int64)
    Vsub = V[old_vids]
    Fsub = remap[Fk]
    return Vsub, Fsub, old_vids


def nearest_vertex_indices(points: np.ndarray, vertices: np.ndarray, chunk: int = 8192) -> np.ndarray:
    points = _as_np_f32(points)
    vertices = _as_np_f32(vertices)
    Nv = int(vertices.shape[0])
    out = np.zeros((int(points.shape[0]),), dtype=np.int64)
    for i, p in enumerate(points):
        best_d = float("inf")
        best_j = 0
        for j0 in range(0, Nv, int(chunk)):
            v = vertices[j0 : j0 + int(chunk)]
            d = v - p[None, :]
            d2 = np.sum(d * d, axis=1)
            k = int(np.argmin(d2))
            val = float(d2[k])
            if val < best_d:
                best_d = val
                best_j = int(j0 + k)
        out[i] = best_j
    return out


def topk_renorm(w: np.ndarray, k: int = 6) -> np.ndarray:
    w = np.asarray(w).astype(np.float32)
    if w.ndim != 1:
        raise ValueError("w must be 1D")
    if k <= 0 or k >= w.shape[0]:
        s = float(np.sum(np.maximum(w, 0.0)))
        if s <= 0:

            return np.full_like(w, 1.0 / float(w.shape[0]))
        ww = np.maximum(w, 0.0) / s
        return ww.astype(np.float32)
    idx = np.argpartition(-w, kth=int(k - 1))[:k]
    out = np.zeros_like(w, dtype=np.float32)
    vals = np.maximum(w[idx], 0.0)
    s = float(np.sum(vals))
    if s <= 0:
        out[idx] = 1.0 / float(k)
    else:
        out[idx] = vals / s
    return out


def build_bridge_strip(
    Rh: np.ndarray,
    Rb_shifted: np.ndarray,
    *,
    target_edge_len: float,
    K_min: int,
    K_max: int,
    smooth_iters: int,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    Rh = _as_np_f32(Rh)
    Rb_shifted = _as_np_f32(Rb_shifted)
    N = int(Rh.shape[0])
    d = float(np.mean(np.linalg.norm(Rh - Rb_shifted, axis=1)))
    K = int(np.ceil(d / float(target_edge_len)))
    K = int(np.clip(K, int(K_min), int(K_max)))

    rings: List[np.ndarray] = []
    for s in range(K + 2):
        alpha = float(s) / float(K + 1)
        rings.append(((1.0 - alpha) * Rh + alpha * Rb_shifted).astype(np.float32))


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
    F = np.asarray(F, dtype=np.int64)
    return V, F, rings


def concat_meshes(meshes: List[trimesh.Trimesh]) -> Tuple[trimesh.Trimesh, Dict]:
    v_all = []
    f_all = []
    v_off = 0
    component_map = {"components": []}
    for m_idx, m in enumerate(meshes):
        v = np.asarray(m.vertices, dtype=np.float32)
        f = np.asarray(m.faces, dtype=np.int64)
        v_range = [int(v_off), int(v_off + v.shape[0])]
        f_range = [int(len(f_all)), int(len(f_all) + f.shape[0])]
        component_map["components"].append(
            {
                "name": getattr(m, "metadata", {}).get("name", f"component_{m_idx}"),
                "vertex_range": v_range,
                "face_range": f_range,
            }
        )
        v_all.append(v)
        f_all.append(f + int(v_off))
        v_off += int(v.shape[0])
    V = np.concatenate(v_all, axis=0)
    F = np.concatenate(f_all, axis=0)
    out = trimesh.Trimesh(vertices=V, faces=F, process=False)
    return out, component_map


def component_range(component_map: Dict, name_prefix: str) -> Tuple[int, int] | None:
    for c in component_map.get("components", []):
        nm = str(c.get("name", ""))
        if nm.startswith(name_prefix):
            vr = c.get("vertex_range", None)
            if isinstance(vr, list) and len(vr) == 2:
                return int(vr[0]), int(vr[1])
    return None


def vertices_within_dist_to_points(
    V: np.ndarray, P: np.ndarray, dist: float, *, chunk: int = 20000
) -> np.ndarray:

    V = _as_np_f32(V)
    P = _as_np_f32(P)
    d2_thr = float(dist) ** 2
    out = np.zeros((V.shape[0],), dtype=bool)
    if P.size == 0:
        return out
    for i0 in range(0, V.shape[0], int(chunk)):
        X = V[i0 : i0 + int(chunk)]


        d2 = np.min(np.sum((X[:, None, :] - P[None, :, :]) ** 2, axis=2), axis=1)
        out[i0 : i0 + int(chunk)] = d2 <= d2_thr
    return out


def seam_mask_from_components_and_rings(
    unified_pre: trimesh.Trimesh,
    component_map: Dict,
    *,
    rings_per_side: Dict[str, Dict[str, np.ndarray]],
    N_ring: int,
    strip_end_rings: int,
    seam_select_dist: float,
    seam_select_dilate_steps: int,
    include_strip: bool,
    include_band: bool,
) -> np.ndarray:
    V = _as_np_f32(unified_pre.vertices)
    seam = np.zeros((V.shape[0],), dtype=bool)


    if include_strip:
        for side in ("left", "right"):
            rng = component_range(component_map, f"wrist_bridge_{side}")
            if rng is not None:
                v0, v1 = rng
                N = int(N_ring)
                e = max(1, int(strip_end_rings))
                a0, a1 = int(v0), int(min(v0 + e * N, v1))
                b0, b1 = int(max(v1 - e * N, v0)), int(v1)
                seam[a0:a1] = True
                seam[b0:b1] = True


    if include_band:
        for side in ("left", "right"):
            info = rings_per_side.get(side, {})
            Rb = info.get("Rb_shifted", None)
            Rh = info.get("Rh_aligned", None)
            if Rb is None or Rh is None:
                continue
            pts = np.concatenate([_as_np_f32(Rb), _as_np_f32(Rh)], axis=0)

            nn = nearest_vertex_indices(pts, V)
            seam[nn] = True
            seam |= vertices_within_dist_to_points(V, pts, float(seam_select_dist))


    if int(seam_select_dilate_steps) > 0:
        adj = build_vertex_adjacency(int(V.shape[0]), np.asarray(unified_pre.faces, dtype=np.int64))
        seam = dilate_mask_by_adjacency(seam, adj, int(seam_select_dilate_steps))


    if include_strip:
        N = int(N_ring)
        e = max(1, int(strip_end_rings))
        for side in ("left", "right"):
            rng = component_range(component_map, f"wrist_bridge_{side}")
            if rng is None:
                continue
            v0, v1 = rng
            keep = np.zeros((int(v1 - v0),), dtype=bool)
            keep[: min(int(e * N), int(v1 - v0))] = True
            keep[max(0, int((v1 - v0) - e * N)) :] = True
            seam[int(v0) : int(v1)] &= keep

    return seam


def component_ids_from_map(component_map: Dict, n_verts: int) -> np.ndarray:
    out = -np.ones((int(n_verts),), dtype=np.int64)
    for ci, c in enumerate(component_map.get("components", [])):
        vr = c.get("vertex_range", None)
        if isinstance(vr, list) and len(vr) == 2:
            v0, v1 = int(vr[0]), int(vr[1])
            if 0 <= v0 < v1 <= int(n_verts):
                out[v0:v1] = int(ci)
    return out


def maybe_flip_strip_faces_outward(V_strip: np.ndarray, F_strip: np.ndarray) -> np.ndarray:

    V = _as_np_f32(V_strip)
    F = np.asarray(F_strip, dtype=np.int64)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return F
    tri = V[F]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    c = np.mean(tri, axis=1)
    C = np.mean(V, axis=0, keepdims=True)
    v = c - C
    score = float(np.mean(np.sum(nrm * v, axis=1)))
    if score < 0.0:
        F = F[:, [0, 2, 1]]
    return F


def pymeshlab_repair_manifold(mesh: trimesh.Trimesh, repair_iters: int = 2) -> trimesh.Trimesh:

    import pymeshlab
    V = _as_np_f32(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)))
    iters = max(1, int(repair_iters))
    for it in range(iters):
        try:
            ms.meshing_repair_non_manifold_edges()
        except Exception as e:
            logger.warning(f"[manifold_repair] iter={it} repair_non_manifold_edges failed: {e}")
        try:
            ms.meshing_repair_non_manifold_vertices()
        except Exception as e:
            logger.warning(f"[manifold_repair] iter={it} repair_non_manifold_vertices failed: {e}")
    m = ms.current_mesh()
    V2 = np.asarray(m.vertex_matrix(), dtype=np.float32)
    F2 = np.asarray(m.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=V2, faces=F2, process=False)


def pymeshlab_close_holes(mesh: trimesh.Trimesh, maxholesize: int = 30) -> trimesh.Trimesh:

    import pymeshlab
    V = _as_np_f32(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)))
    val = int(maxholesize)
    try:
        ms.meshing_close_holes(maxholesize=val, selfintersection=False, refinehole=True)
    except Exception as e:
        if hasattr(pymeshlab, "AbsoluteValue"):
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)))
            ms.meshing_close_holes(maxholesize=pymeshlab.AbsoluteValue(val))
        elif hasattr(pymeshlab, "PureValue"):
            ms = pymeshlab.MeshSet()
            ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)))
            ms.meshing_close_holes(maxholesize=pymeshlab.PureValue(val))
        else:
            raise e
    m = ms.current_mesh()
    V2 = np.asarray(m.vertex_matrix(), dtype=np.float32)
    F2 = np.asarray(m.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=V2, faces=F2, process=False)


def pymeshlab_merge_close_vertices(mesh: trimesh.Trimesh, threshold: float) -> trimesh.Trimesh:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces))

    thr = float(threshold)
    if hasattr(pymeshlab, "PureValue"):
        thr_param = pymeshlab.PureValue(thr)
    elif hasattr(pymeshlab, "AbsoluteValue"):
        thr_param = pymeshlab.AbsoluteValue(thr)
    else:
        thr_param = thr
    ms.meshing_merge_close_vertices(threshold=thr_param)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_unreferenced_vertices()
    m = ms.current_mesh()
    V = np.asarray(m.vertex_matrix(), dtype=np.float32)
    F = np.asarray(m.face_matrix(), dtype=np.int64)
    if V.size == 0 or F.size == 0:
        raise RuntimeError("Merge-close produced empty mesh")
    return trimesh.Trimesh(vertices=V, faces=F, process=False)


def pymeshlab_standard_cleanup(mesh: trimesh.Trimesh) -> trimesh.Trimesh:

    import pymeshlab

    V = _as_np_f32(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=F.astype(np.int32)))
    try:
        ms.meshing_remove_duplicate_faces()
    except Exception:
        pass
    try:
        ms.meshing_remove_unreferenced_vertices()
    except Exception:
        pass
    m = ms.current_mesh()
    V2 = np.asarray(m.vertex_matrix(), dtype=np.float32)
    F2 = np.asarray(m.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=V2, faces=F2, process=False)


def infer_hand_label_id(body_part_labels: np.ndarray) -> int:
    labels = np.asarray(body_part_labels).astype(np.int64)
    mx = int(np.max(labels)) if labels.size else 0
    if mx >= 6 and "hands" in DETAILED_SURFACE_LABELS:
        return int(DETAILED_SURFACE_LABELS.index("hands"))

    return int(SURFACE_LABELS.index("skin"))


def save_point_cloud(points: np.ndarray, path: Path) -> None:
    pc = trimesh.points.PointCloud(vertices=_as_np_f32(points))
    pc.export(str(path))


def save_vertex_id_list(ids: np.ndarray, path: Path) -> None:
    ids = np.asarray(ids).astype(np.int64)
    path.write_text(json.dumps({"vertex_ids": ids.tolist()}, indent=2) + "\n")


def compact_mesh(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    V = _as_np_f32(V)
    F = np.asarray(F, dtype=np.int64)
    if F.size == 0:
        return V, F
    used = np.zeros((V.shape[0],), dtype=bool)
    used[F.reshape(-1)] = True
    remap = -np.ones((V.shape[0],), dtype=np.int64)
    remap[np.where(used)[0]] = np.arange(int(np.sum(used)), dtype=np.int64)
    V2 = V[used]
    F2 = remap[F]
    return V2, F2


class _UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.rank = np.zeros((n,), dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent[x]
        while p != self.parent[p]:
            p = self.parent[p]
        while x != p:
            nx = self.parent[x]
            self.parent[x] = p
            x = nx
        return p

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] = self.rank[ra] + 1


def custom_seam_weld(
    mesh: trimesh.Trimesh,
    seam_mask: np.ndarray,
    thresh: float,
    *,
    component_ids: np.ndarray | None = None,
    pairing: str = "mutual_nn",
) -> trimesh.Trimesh:

    V = _as_np_f32(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    seam_mask = np.asarray(seam_mask).astype(bool)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)
    if float(thresh) <= 0 or not np.any(seam_mask):
        return trimesh.Trimesh(vertices=V, faces=F, process=False)

    seam_ids = np.where(seam_mask)[0].astype(np.int64)
    k = int(seam_ids.shape[0])
    uf = _UnionFind(k)

    cell = float(thresh)
    grid: Dict[Tuple[int, int, int], List[int]] = {}
    thr2 = float(thresh) ** 2


    for li, vi in enumerate(seam_ids.tolist()):
        p = V[int(vi)]
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)), int(np.floor(p[2] / cell)))
        grid.setdefault(key, []).append(li)

    def _neighbors(li: int) -> List[int]:
        vi = int(seam_ids[li])
        p = V[vi]
        key = (int(np.floor(p[0] / cell)), int(np.floor(p[1] / cell)), int(np.floor(p[2] / cell)))
        out: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    out.extend(grid.get((key[0] + dx, key[1] + dy, key[2] + dz), []))
        return out

    pairing = str(pairing)
    if pairing not in {"all_within_thresh", "mutual_nn"}:
        raise ValueError(f"custom_seam_weld pairing must be all_within_thresh or mutual_nn, got {pairing}")

    if pairing == "all_within_thresh":

        for li in range(k):
            vi = int(seam_ids[li])
            p = V[vi]
            for lj in _neighbors(li):
                if lj >= li:
                    continue
                vj = int(seam_ids[lj])
                if component_ids is not None and int(component_ids[vi]) == int(component_ids[vj]):
                    continue
                if float(np.sum((p - V[vj]) ** 2)) <= thr2:
                    uf.union(li, lj)
    else:

        best = -np.ones((k,), dtype=np.int64)
        best_d2 = np.full((k,), np.inf, dtype=np.float64)
        for li in range(k):
            vi = int(seam_ids[li])
            p = V[vi]
            for lj in _neighbors(li):
                if lj == li:
                    continue
                vj = int(seam_ids[lj])
                if component_ids is not None and int(component_ids[vi]) == int(component_ids[vj]):
                    continue
                d2 = float(np.sum((p - V[vj]) ** 2))
                if d2 <= thr2 and d2 < float(best_d2[li]):
                    best_d2[li] = d2
                    best[li] = int(lj)
        for li in range(k):
            lj = int(best[li])
            if lj < 0:
                continue
            if int(best[lj]) == int(li):
                uf.union(li, lj)

    root_to_members: Dict[int, List[int]] = {}
    for li in range(k):
        r = uf.find(li)
        root_to_members.setdefault(r, []).append(li)

    old_to_new = -np.ones((V.shape[0],), dtype=np.int64)
    new_vertices: List[np.ndarray] = []

    for i in range(V.shape[0]):
        if not seam_mask[i]:
            old_to_new[i] = len(new_vertices)
            new_vertices.append(V[i])

    for _, members in root_to_members.items():
        vids = seam_ids[np.asarray(members, dtype=np.int64)]
        mean_pos = np.mean(V[vids], axis=0)
        new_idx = len(new_vertices)
        new_vertices.append(_as_np_f32(mean_pos))
        old_to_new[vids] = new_idx

    Vn = np.stack(new_vertices, axis=0).astype(np.float32)
    Fn = old_to_new[F]
    keep = (Fn[:, 0] != Fn[:, 1]) & (Fn[:, 1] != Fn[:, 2]) & (Fn[:, 0] != Fn[:, 2])
    Fn = Fn[keep]
    Vn, Fn = compact_mesh(Vn, Fn)
    return trimesh.Trimesh(vertices=Vn, faces=Fn, process=False)


def pymeshlab_seam_only_remesh(
    mesh: trimesh.Trimesh,
    seam_points: np.ndarray,
    *,
    select_radius: float,
    iters: int,
    targetlen: float,
    maxsurfdist: float,
    checksurfdist: bool,
    reproject: bool,
    smooth_iters: int,
) -> trimesh.Trimesh:
    import pymeshlab

    V = _as_np_f32(mesh.vertices)
    F = np.asarray(mesh.faces, dtype=np.int64)
    if V.shape[0] == 0 or F.shape[0] == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)
    seam_points = _as_np_f32(seam_points)
    if seam_points.size == 0:
        return trimesh.Trimesh(vertices=V, faces=F, process=False)

    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(V, F.astype(np.int32)), "mesh")
    ms.add_mesh(pymeshlab.Mesh(seam_points, np.zeros((0, 3), dtype=np.int32)), "seam_points")
    ms.set_current_mesh(0)

    ms.compute_scalar_by_distance_from_point_cloud_per_vertex(vertexmesh=1)
    ms.compute_selection_by_scalar_per_vertex(minq=0.0, maxq=float(select_radius), inclusive=True)


    def _abs_value(x: float):
        if hasattr(pymeshlab, "PureValue"):
            return pymeshlab.PureValue(x)
        if hasattr(pymeshlab, "AbsoluteValue"):
            return pymeshlab.AbsoluteValue(x)
        return x

    ms.meshing_isotropic_explicit_remeshing(
        selectedonly=True,
        iterations=int(iters),
        targetlen=_abs_value(float(targetlen)),
        maxsurfdist=_abs_value(float(maxsurfdist)),
        checksurfdist=bool(checksurfdist),
        reprojectflag=bool(reproject),
    )

    if int(smooth_iters) > 0:
        ms.apply_coord_laplacian_smoothing(selected=True, stepsmoothnum=int(smooth_iters))

    out = ms.current_mesh()
    V2 = np.asarray(out.vertex_matrix(), dtype=np.float32)
    F2 = np.asarray(out.face_matrix(), dtype=np.int64)
    return trimesh.Trimesh(vertices=V2, faces=F2, process=False)


def main():
    parser = argparse.ArgumentParser("SMPL-X hand transplant (wrist ring + bridge strip)")
    parser.add_argument(
        "--body_mesh_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--smplx_mesh_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--smplx_joints_path",
        type=str,
        default="",
        help="Path to smpl_joints.npy saved by process_meshes.py. If empty, defaults next to smplx_mesh_path.",
    )
    parser.add_argument(
        "--body_part_labels_path",
        type=str,
        default="",
        help="Optional NeuS/body per-vertex labels (PKL/NPY). If empty/missing, labels are not used.",
    )
    parser.add_argument(
        "--hand_label_id",
        type=int,
        default=-1,
        help="Hand label id to use when body labels exist (detailed labels: hands=7). -1 means infer.",
    )
    parser.add_argument(
        "--use_body_segmentation",
        action="store_true",
        default=False,
        help="Use body segmentation labels for local region selection (legacy). Default is joint-based regions.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory. If empty, writes next to body_mesh_path under mesh/hand_transplant_debug/.",
    )
    parser.add_argument(
        "--smplx_vert_segmentation_json",
        type=str,
        default=str(_SWAPVTON_ROOT / "src/utils/smplx_utils/smplx_models/smplx/smplx_vert_segmentation.json"),
        help="SMPL-X vertex segmentation json (for local hand masking).",
    )
    parser.add_argument(
        "--use_smplx_segmentation",
        action="store_true",
        default=True,
        help="Use SMPL-X segmentation json to refine the joint-based hand region (default: True).",
    )
    parser.add_argument(
        "--no_use_smplx_segmentation",
        dest="use_smplx_segmentation",
        action="store_false",
        help="Disable using SMPL-X segmentation json; use joint-based region only.",
    )


    parser.add_argument("--N_ring", type=int, default=16)


    parser.add_argument(
        "--cut_plane_offset",
        type=float,
        default=0.0,
        help="Offset (m) along forearm axis (n) to place the base wrist cut plane p0 from the wrist joint (jw).",
    )


    parser.add_argument(
        "--body_cut_plane_offset",
        type=float,
        default=0.01,
        help="Offset (m) from base cut plane for BODY cutting plane. Positive moves toward elbow (i.e., -n).",
    )
    parser.add_argument(
        "--hand_cut_plane_offset",
        type=float,
        default=0.005,
        help="Offset (m) from base cut plane for HAND cutting plane. Positive moves toward fingers (i.e., +n).",
    )
    parser.add_argument(
        "--bridge_top_plane_offset",
        type=float,
        default= 0.015,
        help="Offset (m) from base cut plane for BRIDGE TOP ring plane (body-side ring). Positive moves toward elbow (i.e., -n).",
    )
    parser.add_argument(
        "--bridge_bottom_plane_offset",
        type=float,
        default=0.007,
        help="Offset (m) from base cut plane for BRIDGE BOTTOM ring plane (hand-side ring). Positive moves toward fingers (i.e., +n).",
    )
    parser.add_argument("--keep_negative_side", action="store_true", default=True)
    parser.add_argument(
        "--keep_positive_side",
        dest="keep_negative_side",
        action="store_false",
        help="Use s>=0 as the kept side for the body cut (inverts keep_negative_side).",
    )

    parser.add_argument("--face_mode_body_cut", type=str, default="any", choices=["all", "any"])
    parser.add_argument("--face_mode_hand_patch", type=str, default="any", choices=["all", "any"])


    parser.add_argument("--hand_align", action="store_true", default=True)
    parser.add_argument("--no_hand_align", dest="hand_align", action="store_false")
    parser.add_argument("--yaw_deg_max", type=float, default=5.0)
    parser.add_argument("--trans_xy_max", type=float, default=0.01)
    parser.add_argument("--scale_align", action="store_true", default=True)
    parser.add_argument("--no_scale_align", dest="scale_align", action="store_false")
    parser.add_argument("--scale_min", type=float, default=0.98)
    parser.add_argument("--scale_max", type=float, default=1.02)
    parser.add_argument("--reg_lambda", type=float, default=1e-2)


    parser.add_argument("--target_edge_len", type=float, default=0.02)
    parser.add_argument("--K_min", type=int, default=1)
    parser.add_argument("--K_max", type=int, default=4)
    parser.add_argument("--ring_smooth_iters", type=int, default=3)
    parser.add_argument("--weld", action="store_true", default=True)
    parser.add_argument("--no_weld", dest="weld", action="store_false")
    parser.add_argument("--merge_thresh", type=float, default=0.01)
    parser.add_argument(
        "--custom_weld_pairing",
        type=str,
        default="mutual_nn",
        choices=["mutual_nn", "all_within_thresh"],
        help="For --weld_mode custom_seam: weld pairing strategy. mutual_nn is robust; all_within_thresh is legacy aggressive.",
    )
    parser.add_argument("--post_weld_nn_chunk", type=int, default=8192)
    parser.add_argument("--manifold_repair_iters", type=int, default=2)
    parser.add_argument("--weld_mode", type=str, default="custom_seam", choices=["custom_seam", "pymeshlab_global", "off"])
    parser.add_argument("--seam_select_dist", type=float, default=0.03)
    parser.add_argument("--seam_select_dilate_steps", type=int, default=2)
    parser.add_argument("--seam_select_include_strip", action="store_true", default=True)
    parser.add_argument("--no_seam_select_include_strip", dest="seam_select_include_strip", action="store_false")
    parser.add_argument(
        "--seam_strip_end_rings",
        type=int,
        default=1,
        help="When including strip in seam selection, keep only this many endpoint rings (1 = top/bottom rings only).",
    )
    parser.add_argument("--seam_select_include_band", action="store_true", default=True)
    parser.add_argument("--no_seam_select_include_band", dest="seam_select_include_band", action="store_false")
    parser.add_argument("--weld_debug_dump_mask", action="store_true", default=True)
    parser.add_argument("--no_weld_debug_dump_mask", dest="weld_debug_dump_mask", action="store_false")
    parser.add_argument("--close_holes", action="store_true", default=True, help="Close holes after welding, before seam remesh.")
    parser.add_argument("--no_close_holes", dest="close_holes", action="store_false")
    parser.add_argument("--close_holes_maxsize", type=int, default=30, help="Max hole size (edge count) for meshing_close_holes.")
    parser.add_argument("--close_holes_iters", type=int, default=1, help="Number of meshing_close_holes passes.")
    parser.add_argument(
        "--post_weld_cleanup_loops",
        type=int,
        default=1,
        help="Repeat (manifold repair -> close holes -> seam remesh) this many times after welding (default: 2).",
    )


    parser.add_argument("--seam_remesh", action="store_true", default=True)
    parser.add_argument("--no_seam_remesh", dest="seam_remesh", action="store_false")
    parser.add_argument("--seam_remesh_iters", type=int, default=8)
    parser.add_argument("--seam_remesh_targetlen", type=float, default=0.004)
    parser.add_argument("--seam_remesh_maxsurfdist", type=float, default=0.002)
    parser.add_argument("--seam_remesh_reproject", action="store_true", default=True)
    parser.add_argument("--no_seam_remesh_reproject", dest="seam_remesh_reproject", action="store_false")
    parser.add_argument("--seam_remesh_checksurfdist", action="store_true", default=True)
    parser.add_argument("--no_seam_remesh_checksurfdist", dest="seam_remesh_checksurfdist", action="store_false")
    parser.add_argument("--seam_smooth_iters", type=int, default=0)


    parser.add_argument("--local_cut", action="store_true", default=True)
    parser.add_argument("--no_local_cut", dest="local_cut", action="store_false")
    parser.add_argument("--local_dilate_steps_body", type=int, default=18)
    parser.add_argument("--local_dilate_steps_smplx", type=int, default=8)
    parser.add_argument("--local_radius_body", type=float, default=0.20)
    parser.add_argument("--local_radius_smplx", type=float, default=0.18)
    parser.add_argument("--local_axis_slab_halfwidth", type=float, default=0.12)
    parser.add_argument("--region_mode_body", type=str, default="sphere+slab", choices=["sphere", "sphere+slab"])
    parser.add_argument("--region_mode_smplx", type=str, default="sphere+slab", choices=["sphere", "sphere+slab"])

    args = parser.parse_args()

    body_mesh_path = Path(args.body_mesh_path)
    smplx_mesh_path = Path(args.smplx_mesh_path)
    if not args.smplx_joints_path:
        smplx_joints_path = smplx_mesh_path.with_name("smpl_joints.npy")
    else:
        smplx_joints_path = Path(args.smplx_joints_path)
    body_lbl_path = Path(args.body_part_labels_path)

    if not args.output_dir:

        out_dir = body_mesh_path.parent.parent / "hand_transplant_debug"
    else:
        out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = out_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading body mesh: {body_mesh_path}")
    body_mesh = load_trimesh(body_mesh_path)
    logger.info(f"Loading SMPL-X mesh: {smplx_mesh_path}")
    smplx_mesh = load_trimesh(smplx_mesh_path)
    logger.info(f"Loading SMPL joints: {smplx_joints_path}")
    joints = load_smplx_joints(smplx_joints_path)


    body_lbl = None
    if args.body_part_labels_path:
        body_lbl_path = Path(args.body_part_labels_path)
        if body_lbl_path.exists():
            logger.info(f"Loading body part labels: {body_lbl_path}")
            body_lbl = load_body_part_labels(body_lbl_path)
            if body_lbl.shape[0] != body_mesh.vertices.shape[0]:
                raise ValueError(
                    f"body_part_labels Nv mismatch: labels={body_lbl.shape[0]} vs mesh={body_mesh.vertices.shape[0]}"
                )
        else:
            logger.warning(f"body_part_labels_path does not exist: {body_lbl_path} -> proceeding without labels")

    hand_label_id = None
    if body_lbl is not None and bool(args.use_body_segmentation):
        if int(args.hand_label_id) >= 0:
            hand_label_id = int(args.hand_label_id)
            logger.info(f"Using hand_label_id from args: {hand_label_id}")
        else:
            hand_label_id = int(infer_hand_label_id(body_lbl))
            logger.info(f"Hand label id inferred: {hand_label_id}")


    body_adj = build_vertex_adjacency(int(body_mesh.vertices.shape[0]), np.asarray(body_mesh.faces, dtype=np.int64))
    smplx_adj = build_vertex_adjacency(int(smplx_mesh.vertices.shape[0]), np.asarray(smplx_mesh.faces, dtype=np.int64))


    smplx_seg_mask_L = None
    smplx_seg_mask_R = None
    if bool(args.use_smplx_segmentation):
        seg_path = Path(args.smplx_vert_segmentation_json)
        if seg_path.exists():
            try:
                smplx_seg = load_smplx_vert_segmentation(seg_path)
                mL = np.zeros((smplx_mesh.vertices.shape[0],), dtype=bool)
                mR = np.zeros((smplx_mesh.vertices.shape[0],), dtype=bool)
                for key, ids in smplx_seg.items():
                    if not isinstance(ids, list):
                        continue
                    if str(key).startswith("leftHand"):
                        for idx in ids:
                            ii = int(idx)
                            if 0 <= ii < mL.shape[0]:
                                mL[ii] = True
                    if str(key).startswith("rightHand"):
                        for idx in ids:
                            ii = int(idx)
                            if 0 <= ii < mR.shape[0]:
                                mR[ii] = True
                smplx_seg_mask_L = mL
                smplx_seg_mask_R = mR
                logger.info(
                    f"Loaded SMPL-X segmentation masks: left={int(np.sum(mL))} right={int(np.sum(mR))}"
                )
            except Exception as e:
                logger.warning(f"Failed loading SMPL-X segmentation json {seg_path}: {e}. Proceeding without it.")
        else:
            logger.warning(f"--use_smplx_segmentation set but file missing: {seg_path}. Proceeding without it.")


    pieces_meshes: List[trimesh.Trimesh] = []

    body_cur_V = _as_np_f32(body_mesh.vertices)
    body_cur_F = np.asarray(body_mesh.faces, dtype=np.int64)
    body_cur_old_vids = np.arange(body_cur_V.shape[0], dtype=np.int64)

    side_results = {}
    rings_per_side: Dict[str, Dict[str, np.ndarray]] = {}
    for side in ("left", "right"):

        p0_base, n, jw = compute_wrist_plane(joints, side, float(args.cut_plane_offset))
        logger.info(f"[{side}] base cut plane p0={p0_base.tolist()} n={n.tolist()}")

        keep_negative = bool(args.keep_negative_side)


        body_cut_off = float(args.body_cut_plane_offset)
        hand_cut_off = float(args.hand_cut_plane_offset)
        bridge_top_off = float(args.bridge_top_plane_offset)
        bridge_bottom_off = float(args.bridge_bottom_plane_offset)


        p0_body_cut = (p0_base - body_cut_off * n).astype(np.float32)
        p0_hand_cut = (p0_base + hand_cut_off * n).astype(np.float32)
        p0_bridge_top = (p0_base - bridge_top_off * n).astype(np.float32)
        p0_bridge_bottom = (p0_base + bridge_bottom_off * n).astype(np.float32)


        if bool(args.local_cut):
            Vr_body = _as_np_f32(body_mesh.vertices)
            Vr_smplx = _as_np_f32(smplx_mesh.vertices)
            jw_other = joints[joint_index("right_wrist" if side == "left" else "left_wrist")]
            je = joints[joint_index(f"{side}_elbow")]
            axis = normalize(jw - je)
            if float(np.linalg.norm(axis)) < 1e-8:
                axis = normalize(n)


            r2_body = float(args.local_radius_body) ** 2
            r2_smplx = float(args.local_radius_smplx) ** 2
            body_joint = (np.sum((Vr_body - jw[None, :]) ** 2, axis=1) <= r2_body)
            smplx_joint = (np.sum((Vr_smplx - jw[None, :]) ** 2, axis=1) <= r2_smplx)

            if str(args.region_mode_body) == "sphere+slab":
                slab = float(args.local_axis_slab_halfwidth)
                body_joint &= (np.abs((Vr_body - jw[None, :]) @ axis) <= slab)
            if str(args.region_mode_smplx) == "sphere+slab":
                slab = float(args.local_axis_slab_halfwidth)
                smplx_joint &= (np.abs((Vr_smplx - jw[None, :]) @ axis) <= slab)


            body_joint &= (np.sum((Vr_body - jw[None, :]) ** 2, axis=1) <= np.sum((Vr_body - jw_other[None, :]) ** 2, axis=1))
            smplx_joint &= (np.sum((Vr_smplx - jw[None, :]) ** 2, axis=1) <= np.sum((Vr_smplx - jw_other[None, :]) ** 2, axis=1))


            if bool(args.use_body_segmentation) and body_lbl is not None and hand_label_id is not None:
                hand_mask = (np.asarray(body_lbl).astype(np.int64) == int(hand_label_id))
                seed_L, seed_R = split_hands_into_sides_by_wrist_distance(
                    hand_mask,
                    Vr_body,
                    joints[joint_index("left_wrist")],
                    joints[joint_index("right_wrist")],
                )
                body_region_full = (seed_L if side == "left" else seed_R) & body_joint
            else:
                body_region_full = body_joint


            if side == "left" and smplx_seg_mask_L is not None:
                smplx_region = smplx_joint & smplx_seg_mask_L
            elif side == "right" and smplx_seg_mask_R is not None:
                smplx_region = smplx_joint & smplx_seg_mask_R
            else:
                smplx_region = smplx_joint


            body_region_full = dilate_mask_by_adjacency(body_region_full, body_adj, int(args.local_dilate_steps_body))
            smplx_region = dilate_mask_by_adjacency(smplx_region, smplx_adj, int(args.local_dilate_steps_smplx))
        else:
            body_region_full = None
            smplx_region = None


        Rb = robust_wrist_ring(
            body_mesh,
            p0_bridge_top,
            n,
            jw,
            N_resample=int(args.N_ring),
            cut_keep_negative_side=keep_negative,
            repair_iters=int(args.manifold_repair_iters),
            region_vertex_mask=body_region_full,
        )
        Rh = robust_wrist_ring(
            smplx_mesh,
            p0_bridge_bottom,
            n,
            jw,
            N_resample=int(args.N_ring),
            cut_keep_negative_side=not keep_negative,
            repair_iters=int(args.manifold_repair_iters),
            region_vertex_mask=smplx_region,
        )
        Rb, Rh = ensure_same_winding(Rb, Rh, n)


        grid = RingAlignGrid(
            enabled=bool(args.hand_align),
            yaw_deg_max=float(args.yaw_deg_max),
            yaw_samples=11,
            trans_xy_max=float(args.trans_xy_max),
            trans_samples=7,
            scale_enabled=bool(args.scale_align),
            scale_min=float(args.scale_min),
            scale_max=float(args.scale_max),
            scale_samples=5,
            reg_lambda=float(args.reg_lambda),
        )
        Rh_aligned, k, align_info = align_hand_ring_grid(Rh, Rb, n, origin=p0_base, grid=grid)
        Rb_shifted = np.roll(Rb, -int(k), axis=0)

        save_point_cloud(Rb, debug_dir / f"ring_body_{side}.ply")
        save_point_cloud(Rh_aligned, debug_dir / f"ring_smplx_{side}.ply")

        logger.info(f"[{side}] ring shift k={k}, align={align_info}")

        rings_per_side[side] = {
            "Rb_shifted": Rb_shifted,
            "Rh_aligned": Rh_aligned,
        }


        if body_region_full is not None:
            body_region_cur = np.asarray(body_region_full, dtype=bool)[body_cur_old_vids]
        else:
            body_region_cur = None
        body_cur_V, body_cur_F, kept_vids_local = cut_mesh_by_plane(
            body_cur_V,
            body_cur_F,
            p0=p0_body_cut,
            n=n,
            keep_negative_side=keep_negative,
            face_mode=str(args.face_mode_body_cut),
            region_vertex_mask=body_region_cur,
        )
        body_cur_old_vids = body_cur_old_vids[kept_vids_local]


        if smplx_region is not None:


            smplx_faces = np.asarray(smplx_mesh.faces, dtype=np.int64)
            smplx_region = np.asarray(smplx_region, dtype=bool)
            face_mask = np.all(smplx_region[smplx_faces], axis=1)
            if not np.any(face_mask):
                raise RuntimeError(f"[{side}] SMPL-X local region selects no faces; cannot extract hand patch.")
            Vsub, Fsub, old_vids_sub = submesh_from_face_mask(
                _as_np_f32(smplx_mesh.vertices),
                smplx_faces,
                face_mask,
            )
            hand_V, hand_F, kept_vids_sub = cut_mesh_by_plane(
                Vsub,
                Fsub,
                p0=p0_hand_cut,
                n=n,
                keep_negative_side=not keep_negative,
                face_mode=str(args.face_mode_hand_patch),
                region_vertex_mask=None,
            )
            hand_old_vids = old_vids_sub[kept_vids_sub]
        else:
            hand_V, hand_F, hand_old_vids = cut_mesh_by_plane(
                _as_np_f32(smplx_mesh.vertices),
                np.asarray(smplx_mesh.faces, dtype=np.int64),
                p0=p0_hand_cut,
                n=n,
                keep_negative_side=not keep_negative,
                face_mode=str(args.face_mode_hand_patch),
                region_vertex_mask=None,
            )

        hand_mesh = trimesh.Trimesh(vertices=hand_V, faces=hand_F, process=False)
        hand_mesh.metadata = {"name": f"smplx_hand_{side}"}
        hand_mesh.export(str(debug_dir / f"smplx_hand_{side}.obj"))


        V_strip, F_strip, rings = build_bridge_strip(
            Rh_aligned,
            Rb_shifted,
            target_edge_len=float(args.target_edge_len),
            K_min=int(args.K_min),
            K_max=int(args.K_max),
            smooth_iters=int(args.ring_smooth_iters),
        )
        F_strip = maybe_flip_strip_faces_outward(V_strip, F_strip)
        strip_mesh = trimesh.Trimesh(vertices=V_strip, faces=F_strip, process=False)
        strip_mesh.metadata = {"name": f"wrist_bridge_{side}"}
        strip_mesh.export(str(debug_dir / f"bridge_{side}.obj"))


        idx_body_ring = nearest_vertex_indices(Rb_shifted, _as_np_f32(body_mesh.vertices))
        idx_hand_ring = nearest_vertex_indices(Rh_aligned, _as_np_f32(smplx_mesh.vertices))
        dist_body_ring = np.linalg.norm(
            _as_np_f32(body_mesh.vertices)[idx_body_ring] - _as_np_f32(Rb_shifted), axis=1
        )
        dist_hand_ring = np.linalg.norm(
            _as_np_f32(smplx_mesh.vertices)[idx_hand_ring] - _as_np_f32(Rh_aligned), axis=1
        )

        side_results[side] = {
            "p0": p0_base.tolist(),
            "p0_body_cut": p0_body_cut.tolist(),
            "p0_hand_cut": p0_hand_cut.tolist(),
            "p0_bridge_top": p0_bridge_top.tolist(),
            "p0_bridge_bottom": p0_bridge_bottom.tolist(),
            "n": n.tolist(),
            "k": int(k),
            "align": align_info,
            "hand_old_vids_count": int(hand_old_vids.shape[0]),
            "idx_body_ring_count": int(idx_body_ring.shape[0]),
            "idx_hand_ring_count": int(idx_hand_ring.shape[0]),
            "ring_to_body_nn_dist_mean": float(np.mean(dist_body_ring)) if dist_body_ring.size else 0.0,
            "ring_to_body_nn_dist_max": float(np.max(dist_body_ring)) if dist_body_ring.size else 0.0,
            "ring_to_hand_nn_dist_mean": float(np.mean(dist_hand_ring)) if dist_hand_ring.size else 0.0,
            "ring_to_hand_nn_dist_max": float(np.max(dist_hand_ring)) if dist_hand_ring.size else 0.0,
            "local_cut_enabled": bool(args.local_cut),
        }


        pieces_meshes.append(hand_mesh)


        pieces_meshes.append(strip_mesh)


    body_cut_mesh = trimesh.Trimesh(vertices=body_cur_V, faces=body_cur_F, process=False)
    body_cut_mesh.metadata = {"name": "body_cut"}
    body_cut_mesh.export(str(debug_dir / "body_cut.obj"))
    pieces_meshes.insert(0, body_cut_mesh)


    unified_pre, component_map = concat_meshes(pieces_meshes)
    component_map_path = out_dir / "component_map.json"
    component_map_path.write_text(json.dumps(component_map, indent=2) + "\n")
    comp_ids_pre = component_ids_from_map(component_map, n_verts=len(unified_pre.vertices))


    seam_mask_pre = seam_mask_from_components_and_rings(
        unified_pre,
        component_map,
        rings_per_side=rings_per_side,
        N_ring=int(args.N_ring),
        strip_end_rings=int(args.seam_strip_end_rings),
        seam_select_dist=float(args.seam_select_dist),
        seam_select_dilate_steps=int(args.seam_select_dilate_steps),
        include_strip=bool(args.seam_select_include_strip),
        include_band=bool(args.seam_select_include_band),
    )
    if bool(args.weld_debug_dump_mask):
        seam_ids = np.where(seam_mask_pre)[0].astype(np.int64)
        save_vertex_id_list(seam_ids, debug_dir / "seam_mask_pre_weld.json")
        save_point_cloud(_as_np_f32(unified_pre.vertices)[seam_mask_pre], debug_dir / "seam_mask_pre_weld.ply")
        logger.info(f"Seam mask (pre-weld): {int(seam_ids.size)} vertices selected")

    unified_ref = unified_pre


    weld_mode = str(args.weld_mode)
    if not bool(args.weld):
        weld_mode = "off"

    unified_weld = unified_pre
    if weld_mode == "custom_seam":
        logger.info(f"Weld(custom_seam): merge_thresh={float(args.merge_thresh)} on seam-only vertices")
        unified_weld = custom_seam_weld(
            unified_pre,
            seam_mask_pre,
            thresh=float(args.merge_thresh),
            component_ids=comp_ids_pre,
            pairing=str(args.custom_weld_pairing),
        )
    elif weld_mode == "pymeshlab_global":
        logger.info(f"Weld(pymeshlab_global): merge_thresh={float(args.merge_thresh)} (LEGACY GLOBAL)")
        unified_weld = pymeshlab_merge_close_vertices(unified_pre, threshold=float(args.merge_thresh))
    elif weld_mode == "off":
        logger.info("Weld(off): skipping welding")
    else:
        raise ValueError(f"Unknown weld_mode: {weld_mode}")

    if weld_mode != "off":
        unified_weld.export(str(debug_dir / "merged_after_weld.obj"))
        logger.info(
            f"Weld result: {len(unified_pre.vertices)}v->{len(unified_weld.vertices)}v, "
            f"{len(unified_pre.faces)}f->{len(unified_weld.faces)}f"
        )


    if weld_mode == "off":
        seam_mask_post = seam_mask_pre
    else:
        nn_w2pre = nearest_vertex_indices(
            _as_np_f32(unified_weld.vertices), _as_np_f32(unified_pre.vertices), chunk=int(args.post_weld_nn_chunk)
        )
        seam_mask_post = seam_mask_pre[nn_w2pre]


    cleanup_loops = max(1, int(getattr(args, "post_weld_cleanup_loops", 2)))
    mesh_loop = unified_weld
    seam_mask_loop = seam_mask_post
    for loop_idx in range(cleanup_loops):
        logger.info(f"=== Post-weld cleanup loop {loop_idx + 1}/{cleanup_loops} ===")


        logger.info(f"Manifold repair: {int(args.manifold_repair_iters)} iters before hole filling")
        mesh_after_repair = pymeshlab_repair_manifold(mesh_loop, repair_iters=int(args.manifold_repair_iters))
        mesh_after_repair.export(str(debug_dir / f"merged_after_manifold_repair_loop{loop_idx:02d}.obj"))
        nn_repair_to_prev = nearest_vertex_indices(
            _as_np_f32(mesh_after_repair.vertices),
            _as_np_f32(mesh_loop.vertices),
            chunk=int(args.post_weld_nn_chunk),
        )
        seam_mask_after_repair = seam_mask_loop[nn_repair_to_prev]


        mesh_before_remesh = mesh_after_repair
        seam_mask_before_remesh = seam_mask_after_repair
        if bool(args.close_holes):
            iters = max(1, int(args.close_holes_iters))
            logger.info(
                f"Closing holes: meshing_close_holes(maxholesize={int(args.close_holes_maxsize)}) x {iters} iters"
            )
            unified_after_holes = mesh_after_repair
            for it in range(iters):
                unified_after_holes = pymeshlab_close_holes(unified_after_holes, maxholesize=int(args.close_holes_maxsize))
            unified_after_holes.export(str(debug_dir / f"merged_after_close_holes_loop{loop_idx:02d}.obj"))
            nn_holes_to_repair = nearest_vertex_indices(
                _as_np_f32(unified_after_holes.vertices),
                _as_np_f32(mesh_after_repair.vertices),
                chunk=int(args.post_weld_nn_chunk),
            )
            seam_mask_before_remesh = seam_mask_after_repair[nn_holes_to_repair]
            mesh_before_remesh = unified_after_holes


        unified_after_remesh = mesh_before_remesh
        seam_mask_after_remesh = seam_mask_before_remesh
        if bool(args.seam_remesh):
            seam_pts = _as_np_f32(mesh_before_remesh.vertices)[seam_mask_before_remesh]
            tlen = float(args.seam_remesh_targetlen)
            if tlen <= 0:
                tlen = float(args.target_edge_len)
            logger.info(
                f"Seam remesh: selected-only explicit remesh iters={int(args.seam_remesh_iters)} "
                f"targetlen={tlen} maxsurfdist={float(args.seam_remesh_maxsurfdist)}"
            )
            unified_after_remesh = pymeshlab_seam_only_remesh(
                mesh_before_remesh,
                seam_pts,
                select_radius=float(args.seam_select_dist),
                iters=int(args.seam_remesh_iters),
                targetlen=float(tlen),
                maxsurfdist=float(args.seam_remesh_maxsurfdist),
                checksurfdist=bool(args.seam_remesh_checksurfdist),
                reproject=bool(args.seam_remesh_reproject),
                smooth_iters=int(args.seam_smooth_iters),
            )
            unified_after_remesh.export(str(debug_dir / f"merged_after_seam_remesh_loop{loop_idx:02d}.obj"))


            nn_remesh_to_before = nearest_vertex_indices(
                _as_np_f32(unified_after_remesh.vertices),
                _as_np_f32(mesh_before_remesh.vertices),
                chunk=int(args.post_weld_nn_chunk),
            )
            seam_mask_after_remesh = seam_mask_before_remesh[nn_remesh_to_before]
        else:
            logger.info("Seam remesh disabled")

        mesh_loop = unified_after_remesh
        seam_mask_loop = seam_mask_after_remesh

    unified_final = mesh_loop


    logger.info("Final cleanup: remove unreferenced/dups -> merge-close -> manifold repair")
    unified_final = pymeshlab_standard_cleanup(unified_final)


    unified_final = pymeshlab_repair_manifold(unified_final, repair_iters=1)
    unified_final = pymeshlab_standard_cleanup(unified_final)
    unified_final.export(str(debug_dir / "merged_after_final_cleanup.obj"))


    mesh_out_path = out_dir / "mesh_unified_with_smplx_hands.obj"
    unified_final.export(str(mesh_out_path))

    logger.info(f"Saved unified mesh: {mesh_out_path}")

    (out_dir / "run_info.json").write_text(
        json.dumps(json_sanitize({"side_results": side_results}), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
