from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


def _as_np_f32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = _as_np_f32(v).reshape(-1)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v
    return (v / n).astype(np.float32)


def ring_perimeter(P: np.ndarray) -> float:
    P = _as_np_f32(P)
    if P.shape[0] < 3:
        return 0.0
    return float(np.sum(np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)))


def _pca_plane_normal(P: np.ndarray) -> np.ndarray:

    P = _as_np_f32(P)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] != 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    C = P - P.mean(axis=0, keepdims=True)

    try:
        _U, _S, Vt = np.linalg.svd(C.astype(np.float64), full_matrices=False)
        n = Vt[-1].astype(np.float32)
    except Exception:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = _normalize(n)
    if float(np.linalg.norm(n)) < 1e-8:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return n.astype(np.float32)


def _plane_basis_from_normal(
    n: np.ndarray,
    *,
    ref_up: np.ndarray,
    ref_right: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    n = _normalize(n)
    ref_up = _as_np_f32(ref_up).reshape(3)
    ref_right = _as_np_f32(ref_right).reshape(3)

    u = ref_up - float(np.dot(ref_up, n)) * n
    if float(np.linalg.norm(u)) < 1e-6:
        u = ref_right - float(np.dot(ref_right, n)) * n
    if float(np.linalg.norm(u)) < 1e-6:

        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(a, n))) > 0.9:
            a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        u = a - float(np.dot(a, n)) * n
    u = _normalize(u)
    v = _normalize(np.cross(n, u))
    if float(np.linalg.norm(v)) < 1e-6:

        v = _normalize(np.cross(u, n))
    return u.astype(np.float32), v.astype(np.float32)


def _canonicalize_ring_cycle(
    P: np.ndarray,
    *,
    desired_normal: Optional[np.ndarray],
    ref_up: np.ndarray,
    ref_right: np.ndarray,
) -> np.ndarray:

    P = _as_np_f32(P)
    if P.shape[0] < 3:
        return P
    c = P.mean(axis=0)
    n = _pca_plane_normal(P)
    if desired_normal is not None:
        dn = _normalize(desired_normal)
        if float(np.linalg.norm(dn)) > 1e-6 and float(np.dot(n, dn)) < 0.0:
            n = (-n).astype(np.float32)
    u, v = _plane_basis_from_normal(n, ref_up=ref_up, ref_right=ref_right)
    Q = P - c[None, :]
    x = (Q @ u).astype(np.float32)
    y = (Q @ v).astype(np.float32)
    theta = np.arctan2(y, x).astype(np.float32)
    order = np.argsort(theta, axis=0).astype(np.int64)
    P2 = P[order]

    Q2 = P2 - c[None, :]
    x2 = (Q2 @ u).astype(np.float32)
    k0 = int(np.argmax(x2))
    P2 = np.roll(P2, -k0, axis=0)
    return P2.astype(np.float32)


def boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:

    if mesh.faces is None or len(mesh.faces) == 0:
        return np.zeros((0, 2), dtype=np.int64)


    F = np.asarray(mesh.faces, dtype=np.int64)
    edges = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    edges = np.sort(edges, axis=1)


    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    bnd = uniq[counts == 1]
    return bnd.astype(np.int64)


def boundary_loops_vertex_ids(mesh: trimesh.Trimesh) -> List[np.ndarray]:

    E = boundary_edges(mesh)
    if E.shape[0] == 0:
        return []


    adj: Dict[int, List[int]] = {}
    for a, b in E.tolist():
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))

    visited_e = set()
    loops: List[np.ndarray] = []

    def edge_key(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u <= v else (v, u)

    for start in list(adj.keys()):

        nbrs = adj.get(start, [])
        if not nbrs:
            continue
        if all(edge_key(start, n) in visited_e for n in nbrs):
            continue


        nxt = None
        for n in nbrs:
            if edge_key(start, n) not in visited_e:
                nxt = n
                break
        if nxt is None:
            continue

        path = [int(start), int(nxt)]
        visited_e.add(edge_key(start, nxt))

        prev = int(start)
        cur = int(nxt)

        for _ in range(int(E.shape[0]) + 5):
            nbrs2 = adj.get(cur, [])
            if not nbrs2:
                break

            cand = None
            if len(nbrs2) == 1:
                cand = nbrs2[0]
            else:

                choices = [x for x in nbrs2 if x != prev]
                if not choices:
                    choices = nbrs2

                for x in choices:
                    if edge_key(cur, x) not in visited_e:
                        cand = x
                        break
                if cand is None:
                    cand = choices[0]

            cand = int(cand)
            if cand == path[0]:

                loops.append(np.asarray(path[:-1], dtype=np.int64))
                break

            if edge_key(cur, cand) in visited_e:

                break

            path.append(cand)
            visited_e.add(edge_key(cur, cand))
            prev, cur = cur, cand


    out: List[np.ndarray] = []
    seen = set()
    for loop in loops:
        if loop.size < 3:
            continue
        key = tuple(np.sort(loop).tolist())
        if key in seen:
            continue
        seen.add(key)
        out.append(loop)
    return out


def connected_components_vertex_ids(mesh: trimesh.Trimesh) -> List[np.ndarray]:

    if mesh.faces is None or len(mesh.faces) == 0:
        return [np.arange(int(mesh.vertices.shape[0]), dtype=np.int64)]
    comps = trimesh.graph.connected_components(mesh.edges, nodes=np.arange(len(mesh.vertices)))

    out = []
    for c in comps:
        arr = np.asarray(sorted(list(c)), dtype=np.int64)
        if arr.size > 0:
            out.append(arr)

    out.sort(key=lambda a: int(a.size), reverse=True)
    return out


def submesh_from_vertex_ids(mesh: trimesh.Trimesh, vids: np.ndarray) -> Tuple[trimesh.Trimesh, np.ndarray]:

    vids = np.asarray(vids, dtype=np.int64).reshape(-1)
    Nv = int(mesh.vertices.shape[0])
    keep = np.zeros((Nv,), dtype=bool)
    keep[vids[(vids >= 0) & (vids < Nv)]] = True
    F = np.asarray(mesh.faces, dtype=np.int64)
    face_keep = np.all(keep[F], axis=1)
    Fk = F[face_keep]
    if Fk.size == 0:

        return trimesh.Trimesh(vertices=np.zeros((0, 3), dtype=np.float32), faces=np.zeros((0, 3), dtype=np.int64), process=False), np.zeros((0,), dtype=np.int64)
    used = np.unique(Fk.reshape(-1))
    used.sort()
    vmap = -np.ones((Nv,), dtype=np.int64)
    vmap[used] = np.arange(int(used.shape[0]), dtype=np.int64)
    V2 = _as_np_f32(mesh.vertices)[used]
    F2 = vmap[Fk].astype(np.int64)
    return trimesh.Trimesh(vertices=V2, faces=F2, process=False), used.astype(np.int64)


def _umeyama_similarity(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:

    X = _as_np_f32(X)
    Y = _as_np_f32(Y)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3:
        raise ValueError(f"Bad shapes for Umeyama: X {X.shape} Y {Y.shape}")
    N = int(X.shape[0])
    if N < 3:
        raise ValueError("Need at least 3 points for similarity fit.")

    mu_x = X.mean(axis=0)
    mu_y = Y.mean(axis=0)
    Xc = X - mu_x[None, :]
    Yc = Y - mu_y[None, :]
    cov = (Yc.T @ Xc) / float(N)
    U, S, Vt = np.linalg.svd(cov.astype(np.float64), full_matrices=True)
    U = U.astype(np.float64)
    Vt = Vt.astype(np.float64)
    R = (U @ Vt).astype(np.float64)
    if np.linalg.det(R) < 0:

        U[:, -1] *= -1.0
        R = (U @ Vt).astype(np.float64)

    var_x = float(np.mean(np.sum(Xc * Xc, axis=1)))
    if var_x < 1e-20:
        s = 1.0
    else:
        s = float(np.sum(S) / var_x)
    t = (mu_y.astype(np.float64) - s * (R @ mu_x.astype(np.float64))).astype(np.float64)
    return float(s), R.astype(np.float32), t.astype(np.float32)


@dataclass
class RingMatchResult:
    cost: float
    rmse: float
    twist_angle_deg: float
    scale: float
    R: np.ndarray
    t: np.ndarray
    shift: int
    flipped: bool


def best_similarity_ring_match(
    P_hand: np.ndarray,
    P_body: np.ndarray,
    *,
    resample_N: int,
    scale_min: float,
    scale_max: float,
    desired_normal: Optional[np.ndarray] = None,
    ref_up: np.ndarray = (0.0, 1.0, 0.0),
    ref_right: np.ndarray = (1.0, 0.0, 0.0),
    twist_reg_weight: float = 0.02,
    flip_penalty: float = 0.01,
) -> RingMatchResult:

    P_hand = _as_np_f32(P_hand)
    P_body = _as_np_f32(P_body)
    if P_hand.shape[0] < 3 or P_body.shape[0] < 3:
        raise ValueError("Rings too small.")


    X_in = _canonicalize_ring_cycle(P_hand, desired_normal=desired_normal, ref_up=_as_np_f32(ref_up), ref_right=_as_np_f32(ref_right))
    Y_in = _canonicalize_ring_cycle(P_body, desired_normal=desired_normal, ref_up=_as_np_f32(ref_up), ref_right=_as_np_f32(ref_right))


    def resample_cycle(P: np.ndarray, N: int) -> np.ndarray:
        P = _as_np_f32(P)
        if int(P.shape[0]) == int(N):
            return P
        d = np.linalg.norm(P - np.roll(P, -1, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(d)], axis=0)
        total = float(s[-1])
        if total < 1e-12:
            return np.repeat(P[:1], repeats=int(N), axis=0)
        t = np.linspace(0.0, total, num=int(N) + 1, endpoint=True)[:-1]
        out = np.zeros((int(N), 3), dtype=np.float32)
        for i in range(int(N)):
            ti = float(t[i])

            k = int(np.searchsorted(s, ti, side="right") - 1)
            k = max(0, min(k, int(P.shape[0]) - 1))
            t0 = float(s[k])
            t1 = float(s[k + 1]) if (k + 1) < int(s.shape[0]) else total
            a = 0.0 if abs(t1 - t0) < 1e-12 else (ti - t0) / (t1 - t0)
            p0 = P[k]
            p1 = P[(k + 1) % int(P.shape[0])]
            out[i] = (1.0 - float(a)) * p0 + float(a) * p1
        return out

    N = int(max(16, int(resample_N)))
    X = resample_cycle(X_in, N)
    Y0 = resample_cycle(Y_in, N)

    best: Optional[RingMatchResult] = None

    for flipped in (False, True):
        Y = Y0[::-1].copy() if flipped else Y0
        for shift in range(int(N)):
            Ysh = np.roll(Y, shift, axis=0)
            s, R, t = _umeyama_similarity(X, Ysh)
            s_clamped = float(np.clip(s, float(scale_min), float(scale_max)))
            if abs(s_clamped - s) > 1e-6:

                mu_x = X.mean(axis=0)
                mu_y = Ysh.mean(axis=0)
                t = (mu_y - s_clamped * (R @ mu_x)).astype(np.float32)
                s = s_clamped
            X2 = (s * (X @ R.T) + t.reshape(1, 3)).astype(np.float32)
            rmse = float(np.sqrt(np.mean(np.sum((X2 - Ysh) ** 2, axis=1))))


            min_shift = min(int(shift), int(N) - int(shift))
            twist_angle = (2.0 * np.pi) * (float(min_shift) / float(N))
            twist_deg = float(twist_angle * (180.0 / np.pi))
            twist_pen = float(max(0.0, float(twist_reg_weight))) * float((twist_angle / np.pi) ** 2)
            flip_pen = float(max(0.0, float(flip_penalty))) if bool(flipped) else 0.0
            cost = float(rmse + twist_pen + flip_pen)

            cand = RingMatchResult(
                cost=cost,
                rmse=rmse,
                twist_angle_deg=twist_deg,
                scale=float(s),
                R=R,
                t=t,
                shift=int(shift),
                flipped=bool(flipped),
            )
            if (best is None) or (cand.cost < best.cost):
                best = cand

    assert best is not None
    return best


def _pick_main_loop(mesh: trimesh.Trimesh, loops_vids: List[np.ndarray]) -> Optional[np.ndarray]:
    if not loops_vids:
        return None
    V = _as_np_f32(mesh.vertices)
    best = None
    best_per = -1.0
    for lv in loops_vids:
        lv = np.asarray(lv, dtype=np.int64).reshape(-1)
        if lv.size < 3:
            continue
        P = V[lv]
        per = ring_perimeter(P)
        if per > best_per:
            best_per = per
            best = lv
    return best


def align_hands_mesh_to_body_wrist(
    *,
    hands_mesh: trimesh.Trimesh,
    body_mesh: trimesh.Trimesh,
    mode: str = "full",
    resample_N: int = 128,
    scale_min: float = 0.85,
    scale_max: float = 1.18,
    twist_reg_weight: float = 0.02,
    flip_penalty: float = 0.01,
) -> Tuple[trimesh.Trimesh, Dict]:

    mode = str(mode).strip().lower()
    if mode not in {"full", "scale_only"}:
        mode = "full"
    report: Dict = {"status": "ok", "mode": mode, "pairs": []}


    body_loops_vids = boundary_loops_vertex_ids(body_mesh)
    if not body_loops_vids:
        report["status"] = "no_body_boundary"
        return hands_mesh, report
    Vb = _as_np_f32(body_mesh.vertices)
    body_loops = []
    for i, lv in enumerate(body_loops_vids):
        P = Vb[np.asarray(lv, dtype=np.int64)]
        body_loops.append({"i": int(i), "vids": lv, "P": P, "center": P.mean(axis=0), "perimeter": ring_perimeter(P)})


    comps = connected_components_vertex_ids(hands_mesh)
    if not comps:
        report["status"] = "no_hands_components"
        return hands_mesh, report

    Vh_full = _as_np_f32(hands_mesh.vertices).copy()
    Fh_full = np.asarray(hands_mesh.faces, dtype=np.int64).copy() if hands_mesh.faces is not None else None


    comp_infos = []
    for ci, vids in enumerate(comps):
        sub, used = submesh_from_vertex_ids(hands_mesh, vids)
        loops_vids = boundary_loops_vertex_ids(sub)
        main_lv = _pick_main_loop(sub, loops_vids)
        if main_lv is None:
            continue
        P = _as_np_f32(sub.vertices)[np.asarray(main_lv, dtype=np.int64)]
        comp_infos.append(
            {
                "ci": int(ci),
                "used_vids_full": used,
                "submesh": sub,
                "ring_vids_sub": np.asarray(main_lv, dtype=np.int64),
                "ring_pts": P,
                "center": P.mean(axis=0),
                "perimeter": ring_perimeter(P),
            }
        )

    if not comp_infos:
        report["status"] = "no_hands_boundary"
        return hands_mesh, report

    def _perim_ok(hand_per: float, body_per: float) -> bool:


        if hand_per <= 1e-8 or body_per <= 1e-8:
            return False
        r = float(body_per) / float(hand_per)
        return (r >= 0.6) and (r <= 3.0)

    def _pair_cost(c: dict, b: dict) -> float:
        d = float(np.linalg.norm(_as_np_f32(c["center"]) - _as_np_f32(b["center"])))

        r = float(b["perimeter"]) / max(1e-8, float(c["perimeter"]))
        per_pen = float(abs(np.log(max(r, 1e-8))))
        return float(d + 0.05 * per_pen)


    assignments: List[Tuple[int, int]] = []
    body_all = list(body_loops)

    def _solve_assign(filter_by_perim: bool) -> List[Tuple[int, int]]:
        if len(comp_infos) == 1:
            c0 = comp_infos[0]
            best = None
            best_cost = float("inf")
            for j, b in enumerate(body_all):
                if filter_by_perim and (not _perim_ok(float(c0["perimeter"]), float(b["perimeter"]))):
                    continue
                cost = _pair_cost(c0, b)
                if cost < best_cost:
                    best_cost = cost
                    best = j
            return [(0, int(best))] if best is not None else []

        if len(comp_infos) == 2 and len(body_all) >= 2:
            c0, c1 = comp_infos[0], comp_infos[1]
            best_pair = None
            best_cost = float("inf")
            for j0, b0 in enumerate(body_all):
                if filter_by_perim and (not _perim_ok(float(c0["perimeter"]), float(b0["perimeter"]))):
                    continue
                for j1, b1 in enumerate(body_all):
                    if j1 == j0:
                        continue
                    if filter_by_perim and (not _perim_ok(float(c1["perimeter"]), float(b1["perimeter"]))):
                        continue
                    cost = _pair_cost(c0, b0) + _pair_cost(c1, b1)
                    if cost < best_cost:
                        best_cost = cost
                        best_pair = (j0, j1)
            if best_pair is None:
                return []
            return [(0, int(best_pair[0])), (1, int(best_pair[1]))]


        used = set()
        out = []
        for i, c in enumerate(comp_infos):
            best = None
            best_cost = float("inf")
            for j, b in enumerate(body_all):
                if j in used:
                    continue
                if filter_by_perim and (not _perim_ok(float(c["perimeter"]), float(b["perimeter"]))):
                    continue
                cost = _pair_cost(c, b)
                if cost < best_cost:
                    best_cost = cost
                    best = j
            if best is None:
                break
            used.add(best)
            out.append((i, int(best)))
        return out

    assignments = _solve_assign(filter_by_perim=True)
    if not assignments:
        assignments = _solve_assign(filter_by_perim=False)
        report["note"] = "perimeter_filter_failed_fallback_to_all_loops"


    for comp_i, body_idx in assignments:
        comp = comp_infos[int(comp_i)]
        body = body_all[int(body_idx)]
        P_hand = comp["ring_pts"]
        P_body = body["P"]
        if mode == "scale_only":

            per_h = float(ring_perimeter(P_hand))
            per_b = float(ring_perimeter(P_body))
            if per_h <= 1e-12 or per_b <= 1e-12:
                s = 1.0
            else:
                s = float(per_b / per_h)
            s = float(np.clip(s, float(scale_min), float(scale_max)))
            R = np.eye(3, dtype=np.float32)

            c = _as_np_f32(comp["center"]).reshape(1, 3)
            t = ((1.0 - float(s)) * c.reshape(3,)).astype(np.float32)
            rmse = float(abs(per_b - (float(s) * per_h)))
            match_cost = rmse
            twist_deg = 0.0
            shift = 0
            flipped = False
        else:
            desired_n = _normalize(_as_np_f32(comp["center"]) - _as_np_f32(body["center"]))
            match = best_similarity_ring_match(
                P_hand,
                P_body,
                resample_N=int(resample_N),
                scale_min=float(scale_min),
                scale_max=float(scale_max),
                desired_normal=desired_n,
                twist_reg_weight=float(twist_reg_weight),
                flip_penalty=float(flip_penalty),
            )
            s = float(match.scale)
            R = _as_np_f32(match.R).reshape(3, 3)
            t = _as_np_f32(match.t).reshape(3,)
            rmse = float(match.rmse)
            match_cost = float(match.cost)
            twist_deg = float(match.twist_angle_deg)
            shift = int(match.shift)
            flipped = bool(match.flipped)

        used_full = np.asarray(comp["used_vids_full"], dtype=np.int64).reshape(-1)
        V_sel = Vh_full[used_full]
        V_sel2 = (float(s) * (V_sel @ R.T) + t.reshape(1, 3)).astype(np.float32)
        Vh_full[used_full] = V_sel2

        report["pairs"].append(
            {
                "hand_component": int(comp["ci"]),
                "body_loop_i": int(body["i"]),
                "mode": mode,
                "cost": float(match_cost),
                "rmse": float(rmse),
                "twist_angle_deg": float(twist_deg),
                "scale": float(s),
                "R": _as_np_f32(R).tolist(),
                "t": _as_np_f32(t).reshape(-1).tolist(),
                "shift": int(shift),
                "flipped": bool(flipped),
                "hand_perimeter": float(comp["perimeter"]),
                "body_perimeter": float(body["perimeter"]),
                "twist_reg_weight": float(twist_reg_weight),
                "flip_penalty": float(flip_penalty),
            }
        )

    out_mesh = trimesh.Trimesh(vertices=Vh_full, faces=Fh_full, process=False)
    return out_mesh, report


def write_report_json(report: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
