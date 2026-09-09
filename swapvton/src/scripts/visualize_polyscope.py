from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import polyscope as ps
import trimesh
from plyfile import PlyData


ArrayF = np.ndarray
ArrayI = np.ndarray


def rotate_x_90(points: ArrayF) -> ArrayF:


    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return points @ R.T


def load_mesh(path: Path) -> Tuple[ArrayF, ArrayI]:

    mesh = trimesh.load(path, force="mesh")  # type: ignore[no-untyped-call]
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"File does not contain a single mesh: {path}")

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertices = rotate_x_90(vertices)
    faces = np.asarray(mesh.faces, dtype=np.int32)

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"Unexpected vertex shape {vertices.shape} in {path}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"Unexpected face shape {faces.shape} in {path}")

    return vertices, faces


def load_skeleton_ply(
    path: Path,
) -> Tuple[ArrayF, Optional[ArrayI], Optional[ArrayF]]:

    ply = PlyData.read(str(path))

    if "vertex" not in ply:
        raise ValueError(f"Skeleton PLY has no 'vertex' element: {path}")

    v = ply["vertex"].data
    points = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)
    points = rotate_x_90(points)

    edges: Optional[ArrayI] = None
    if "edge" in ply:
        e = ply["edge"].data
        if "vertex1" in e.dtype.names and "vertex2" in e.dtype.names:
            edges = np.stack([e["vertex1"], e["vertex2"]], axis=-1).astype(np.int32)

    colors: Optional[ArrayF] = None
    color_keys = [k for k in ("red", "green", "blue", "r", "g", "b") if k in v.dtype.names]
    if all(k in v.dtype.names for k in ("red", "green", "blue")):
        r = v["red"].astype(np.float32)
        g = v["green"].astype(np.float32)
        b = v["blue"].astype(np.float32)
        colors = np.stack([r, g, b], axis=-1) / 255.0
    elif all(k in v.dtype.names for k in ("r", "g", "b")):
        r = v["r"].astype(np.float32)
        g = v["g"].astype(np.float32)
        b = v["b"].astype(np.float32)
        colors = np.stack([r, g, b], axis=-1)

        colors = np.clip(colors, 0.0, 1.0)

    return points, edges, colors


def register_meshes(
    mesh_paths: Sequence[Path],
    alpha: float,
) -> None:

    for idx, path in enumerate(mesh_paths):
        vertices, faces = load_mesh(path)
        name = f"mesh_{idx}: {path.name}"
        handle = ps.register_surface_mesh(name, vertices, faces)
        handle.set_transparency(alpha)


def register_skeletons(
    skeleton_paths: Sequence[Path],
    alpha: float,
) -> None:

    for idx, skeleton_path in enumerate(skeleton_paths):
        points, edges, colors = load_skeleton_ply(skeleton_path)

        pc_name = f"skeleton_points_{idx}: {skeleton_path.name}"
        pc = ps.register_point_cloud(pc_name, points)
        pc.set_transparency(alpha)
        if colors is not None:
            pc.add_color_quantity("vertex_color", colors, enabled=True)

        if edges is not None and len(edges) > 0:
            cn_name = f"skeleton_edges_{idx}: {skeleton_path.name}"
            cn = ps.register_curve_network(cn_name, points, edges)

            cn.set_radius(0.003, relative=True)
            cn.set_transparency(alpha)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize meshes and skeletons with Polyscope.\n\n"
            "Examples:\n"
            "  Two transparent meshes:\n"
            "    python -m src.scripts.visualize_polyscope "
            "--mesh outer.obj --mesh inner.ply --mesh-alpha 0.4\n\n"
            "  Mesh + skeleton:\n"
            "    python -m src.scripts.visualize_polyscope "
            "--mesh mesh.obj --skeleton skeleton.ply --mesh-alpha 0.4\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--mesh",
        action="append",
        default=[],
        help="Path to a mesh file (.obj or .ply). Can be given multiple times.",
    )
    parser.add_argument(
        "-s",
        "--skeleton",
        action="append",
        default=[],
        help=(
            "Path to a skeleton PLY file (vertex + optional edge elements). "
            "Can be given multiple times."
        ),
    )
    parser.add_argument(
        "--mesh-alpha",
        type=float,
        default=0.4,
        help="Transparency for all meshes in [0,1]. Default: 0.4.",
    )
    parser.add_argument(
        "--skeleton-alpha",
        type=float,
        default=0.7,
        help="Transparency for skeleton geometry in [0,1]. Default: 0.7.",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    mesh_paths: List[Path] = [Path(p) for p in args.mesh]
    skeleton_paths: List[Path] = [Path(p) for p in args.skeleton]

    for p in mesh_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Mesh file not found: {p}")
    for p in skeleton_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Skeleton file not found: {p}")

    if not mesh_paths and not skeleton_paths:
        parser.error("Provide at least one --mesh or at least one --skeleton file.")

    if not (0.0 <= args.mesh_alpha <= 1.0):
        parser.error("--mesh-alpha must be in [0, 1].")
    if not (0.0 <= args.skeleton_alpha <= 1.0):
        parser.error("--skeleton-alpha must be in [0, 1].")

    args.mesh_paths = mesh_paths
    args.skeleton_paths = skeleton_paths
    return args


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)


    ps.init()

    ps.set_ground_plane_mode("none")
    ps.set_up_dir("z_up")

    if args.mesh_paths:
        register_meshes(args.mesh_paths, alpha=args.mesh_alpha)

    if args.skeleton_paths:
        register_skeletons(args.skeleton_paths, alpha=args.skeleton_alpha)


    ps.show()


if __name__ == "__main__":
    main()
