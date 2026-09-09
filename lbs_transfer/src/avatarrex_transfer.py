from utils import find_matches_closest_surface, inpaint, smooth
import igl
import numpy as np
import os
import argparse
import sys
from pathlib import Path


def _init_polyscope(args):

    if getattr(args, "_ps_inited", False):
        return getattr(args, "_ps", None)

    args._ps_inited = True
    args._ps = None


    os.environ.setdefault("EGL_PLATFORM", "surfaceless")

    try:
        import polyscope as ps
    except Exception as e:
        print(f"Warning: Polyscope import failed ({e}). Visualization outputs disabled.")
        return None

    try:

        if hasattr(ps, "set_allow_headless_backends"):
            ps.set_allow_headless_backends(True)

        backend = getattr(args, "viz_backend", None)
        if backend:
            ps.init(backend)
        else:
            ps.init()


        if hasattr(ps, "set_window_size"):
            ps.set_window_size(int(args.viz_width), int(args.viz_height))

        args._ps = ps
        return ps
    except Exception as e:
        print(f"Warning: Polyscope initialization failed ({e}). Visualization outputs disabled.")
        return None


def _viz_dir_for_frame(args, output_base_dir: Path, frame_id_str: str) -> Path:

    if getattr(args, "viz_dir", None):
        return Path(args.viz_dir) / frame_id_str

    return output_base_dir


def _axis_to_idx(axis: str | None) -> int | None:
    if axis is None:
        return None
    a = str(axis).lower()
    if a == "x":
        return 0
    if a == "y":
        return 1
    if a == "z":
        return 2
    return None


def _setup_camera_to_meshes(ps, V_list, args=None):


    V_all = np.concatenate(V_list, axis=0)
    bb_min = V_all.min(axis=0)
    bb_max = V_all.max(axis=0)
    center = 0.5 * (bb_min + bb_max)
    extents = (bb_max - bb_min).astype(float)
    diag = float(np.linalg.norm(extents))
    if not np.isfinite(diag) or diag <= 0:
        diag = 1.0

    up_idx = int(np.argmax(extents))
    view_idx = int(np.argmin(extents))


    if args is not None:
        forced_up = _axis_to_idx(getattr(args, "viz_up_axis", None))
        forced_view = _axis_to_idx(getattr(args, "viz_view_axis", None))
        if forced_up is not None:
            up_idx = forced_up
        if forced_view is not None:
            view_idx = forced_view
        if view_idx == up_idx:

            view_idx = int([i for i in (0, 1, 2) if i != up_idx][0])

    up = np.zeros(3, dtype=float)
    up[up_idx] = 1.0
    view = np.zeros(3, dtype=float)
    view[view_idx] = 1.0

    height = float(extents[up_idx]) if np.isfinite(extents[up_idx]) else 0.0
    if height <= 0:
        height = diag

    target = center + up * (0.20 * height)
    cam = target + view * (2.25 * diag) + up * (0.10 * height)
    try:
        if hasattr(ps, "reset_camera_to_home_view"):
            ps.reset_camera_to_home_view()
        if hasattr(ps, "look_at"):
            ps.look_at(cam, target)
    except Exception:
        pass


def process_frame(args, frame_id_str):

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    explicit_mode = bool(
        getattr(args, "source_mesh_path", None)
        or getattr(args, "source_lbs_weights_path", None)
        or getattr(args, "target_mesh_path", None)
    )


    single_frame_layout = (data_dir / "smpl_body.obj").exists() and (data_dir / "smpl_lbs_weights.npy").exists()

    if explicit_mode:

        smplx_path = Path(str(args.source_mesh_path))
        lbsw_path = Path(str(args.source_lbs_weights_path))
        target_mesh_base_dir = output_dir
        output_base_dir = output_dir
    elif args.dataset_type in {'thuman2', 'mvhumannet', 'talkbody4d'} or single_frame_layout:
        smplx_path = data_dir / "smpl_body.obj"
        lbsw_path = data_dir / "smpl_lbs_weights.npy"
        target_mesh_base_dir = data_dir
        output_base_dir = output_dir
    else:
        smplx_path = data_dir / frame_id_str / "smpl_body.obj"
        lbsw_path = data_dir / frame_id_str / "smpl_lbs_weights.npy"
        target_mesh_base_dir = data_dir / frame_id_str
        output_base_dir = output_dir / frame_id_str

        output_base_dir.mkdir(parents=True, exist_ok=True)

    if explicit_mode:
        target_mesh_path = Path(str(args.target_mesh_path))
        print(f"Target mesh: Explicit ({target_mesh_path})")
    elif args.target_mesh == "garment":
        target_mesh_path = target_mesh_base_dir / "garments.ply"
        print(f"Target mesh: Garment ({target_mesh_path})")
    elif args.target_mesh == "full_nerf":


        target_mesh_path = Path(args.full_nerf_mesh_path_template.format(frame_id=frame_id_str))
        print(f"Target mesh: Full NeRF ({target_mesh_path})")
    else:
        print(f"Error: Unknown target mesh type '{args.target_mesh}'. Choose 'garment' or 'full_nerf'.")
        return


    if not smplx_path.exists():
        print(f"Error: SMPL mesh not found at {smplx_path}. Skipping frame {frame_id_str}.")
        return
    if not lbsw_path.exists():
        print(f"Error: SMPL LBS weights not found at {lbsw_path}. Skipping frame {frame_id_str}.")
        return
    if not target_mesh_path.exists():
        print(f"Error: Target mesh not found at {target_mesh_path}. Skipping frame {frame_id_str}.")
        return

    print(f"Processing frame {frame_id_str}...")


    try:
        V1, F1 = igl.read_triangle_mesh(str(smplx_path))
        V1, F1, _, _ = igl.remove_unreferenced(V1, F1)


        N1 = igl.per_vertex_normals(V1, F1)
        print(f"Loaded source mesh (SMPL): {V1.shape[0]} vertices, {F1.shape[0]} faces.")
    except Exception as e:
        print(f"Error loading source mesh {smplx_path}: {e}. Skipping frame {frame_id_str}.")
        return


    try:

        V2, F2 = igl.read_triangle_mesh(str(target_mesh_path))
        V2, F2, _, _ = igl.remove_unreferenced(V2, F2)


        N2 = igl.per_vertex_normals(V2, F2)
        print(f"Loaded target mesh: {V2.shape[0]} vertices, {F2.shape[0]} faces.")

    except Exception as e:
        print(f"Error loading target mesh {target_mesh_path}: {e}. Skipping frame {frame_id_str}.")
        return


    try:
        W = np.load(str(lbsw_path))
        if W.shape[0] != V1.shape[0]:
             print(f"Error: Mismatch in vertex count for LBS weights ({W.shape[0]}) and SMPL mesh ({V1.shape[0]}) for frame {frame_id_str}.")
             return
        num_bones = W.shape[1]
        print(f"Loaded LBS weights: {W.shape[0]} vertices, {num_bones} bones.")

    except Exception as e:
        print(f"Error loading LBS weights {lbsw_path}: {e}. Skipping frame {frame_id_str}.")
        return


    save_viz = bool(getattr(args, "save_viz", True))
    do_ps = save_viz or bool(args.visualize)
    ps = None
    if do_ps:
        ps = _init_polyscope(args)
        if ps is not None:

            try:
                ps.remove_all_structures()
                ps.remove_all_groups()
            except Exception:
                pass


    if ps is not None:
        ps.register_surface_mesh(f"SourceMesh_{frame_id_str}", V1, F1, smooth_shade=True)
        ps.register_surface_mesh(f"TargetMesh_{frame_id_str}", V2, F2, smooth_shade=True)
        try:
            src = ps.get_surface_mesh(f"SourceMesh_{frame_id_str}")
            tgt = ps.get_surface_mesh(f"TargetMesh_{frame_id_str}")
            src.set_color((0.75, 0.75, 0.75))
            src.set_transparency(0.55)
            tgt.set_color((0.90, 0.55, 0.25))
            tgt.set_transparency(0.0)
        except Exception:
            pass
        _setup_camera_to_meshes(ps, [V1, V2], args=args)

        if save_viz:
            viz_dir = _viz_dir_for_frame(args, output_base_dir, frame_id_str)
            viz_dir.mkdir(parents=True, exist_ok=True)
            try:

                ps.screenshot(str(viz_dir / f"{frame_id_str}_overlay.png"))
            except Exception as e:
                print(f"Warning: Failed to save polyscope screenshot (overlay) for {frame_id_str}: {e}")


    dDISTANCE_THRESHOLD = float(getattr(args, "distance_threshold_m", 0.01))
    dDISTANCE_THRESHOLD_SQRD = dDISTANCE_THRESHOLD *dDISTANCE_THRESHOLD
    dANGLE_THRESHOLD_DEGREES = float(getattr(args, "angle_threshold_deg", 30.0))

    print(f"Starting closest point matching with distance threshold {dDISTANCE_THRESHOLD:.4f} and angle threshold {dANGLE_THRESHOLD_DEGREES} degrees.")


    try:
        Matched, SkinWeights_interpolated = find_matches_closest_surface(V1,F1,N1,V2,F2,N2,W,dDISTANCE_THRESHOLD_SQRD,dANGLE_THRESHOLD_DEGREES)
        print(f"Closest point matching complete. Found {np.sum(Matched)} matches.")
    except Exception as e:
        print(f"Error during closest point matching for frame {frame_id_str}: {e}. Skipping inpainting.")
        return


    if ps is not None:

        if args.visualize:
            try:
                ps.get_surface_mesh(f"TargetMesh_{frame_id_str}").add_scalar_quantity("Matched", Matched, defined_on='vertices', cmap='blues')
            except Exception:
                pass


    print("Starting skinning weights inpainting.")
    try:
        InpaintedWeights, success = inpaint(V2, F2, SkinWeights_interpolated, Matched)
        if success:
            print("Inpainting successful.")

            print("Starting smoothing.")
            try:

                 SmoothedInpaintedWeights, VIDs_to_smooth = smooth(V2, F2, InpaintedWeights, Matched, dDISTANCE_THRESHOLD, num_smooth_iter_steps=10, smooth_alpha=0.2)
                 print(f"Smoothing complete. {np.sum(VIDs_to_smooth)} vertices smoothed.")
                 if ps is not None:
                     if args.visualize:
                         try:
                             ps.get_surface_mesh(f"TargetMesh_{frame_id_str}").add_scalar_quantity("VIDs_to_smooth", VIDs_to_smooth, defined_on='vertices', cmap='blues')
                         except Exception:
                             pass


                 def _resolve_out_path(arg_name: str, default_name: str) -> Path:
                     p = getattr(args, arg_name, None)
                     if p:
                         out_p = Path(str(p))
                         if not out_p.is_absolute():
                             out_p = output_base_dir / out_p
                         out_p.parent.mkdir(parents=True, exist_ok=True)
                         return out_p
                     out_p = output_base_dir / default_name
                     out_p.parent.mkdir(parents=True, exist_ok=True)
                     return out_p

                 inpainted_output_path = _resolve_out_path("out_inpainted_weights_path", "inpainted_weights.npy")
                 smoothed_output_path = _resolve_out_path("out_smoothed_weights_path", "smoothed_inpainted_weights.npy")

                 np.save(str(inpainted_output_path), InpaintedWeights)
                 np.save(str(smoothed_output_path), SmoothedInpaintedWeights)
                 print(f"Saved inpainted weights to {inpainted_output_path}")
                 print(f"Saved smoothed inpainted weights to {smoothed_output_path}")


                 if args.visualize and ps is not None:
                      num_bones = InpaintedWeights.shape[1]
                      for i in range(num_bones):

                         ps.get_surface_mesh(f"TargetMesh_{frame_id_str}").add_scalar_quantity(f"Bone{i}_Inpainted_{frame_id_str}", InpaintedWeights[:,i], defined_on='vertices', cmap='viridis')

                         ps.get_surface_mesh(f"TargetMesh_{frame_id_str}").add_scalar_quantity(f"Bone{i}_Smoothed_{frame_id_str}", SmoothedInpaintedWeights[:,i], defined_on='vertices', cmap='viridis')


            except Exception as e:
                 print(f"Error during smoothing for frame {frame_id_str}: {e}. Saving inpainted weights only.")

                 inpainted_output_path = (
                     Path(str(args.out_inpainted_weights_path))
                     if getattr(args, "out_inpainted_weights_path", None)
                     else (output_base_dir / "inpainted_weights.npy")
                 )
                 if not inpainted_output_path.is_absolute():
                     inpainted_output_path = output_base_dir / inpainted_output_path
                 inpainted_output_path.parent.mkdir(parents=True, exist_ok=True)
                 np.save(str(inpainted_output_path), InpaintedWeights)
                 print(f"Saved inpainted weights to {inpainted_output_path}")

        else:
            print(f"[Error] Inpainting failed for frame {frame_id_str}.")

    except Exception as e:
        print(f"Error during inpainting for frame {frame_id_str}: {e}.")

    print(f"Finished processing frame {frame_id_str}.")

def main():
    parser = argparse.ArgumentParser(description="Transfer SMPL LBS weights to target mesh.")


    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the directory containing processed mesh subdirectories per frame (output of process_avatarrex_meshes.py).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the transferred weights (NPY files).")
    parser.add_argument("--target_mesh", type=str, default="full_nerf", choices=["garment", "full_nerf"],
                        help="Type of target mesh for weight transfer ('garment' or 'full_nerf').")
    parser.add_argument("--frame_start", type=int, default=0,
                        help="Start index of the frame range (inclusive).")
    parser.add_argument("--frame_end", type=int, default=10,
                        help="End index of the frame range (inclusive).")
    parser.add_argument("--frame_step", type=int, default=1,
                        help="Step for the frame range.")
    parser.add_argument("--frame_list", nargs='+', type=str, default=None,
                        help="Explicit list of frame IDs (e.g., 00000000 00000010). Overrides frame_start/end/step.")
    parser.add_argument("--frame_format_digits", type=int, default=8,
                        help="Number of digits in frame ID strings (e.g., 8 for '00000000').")
    parser.add_argument("--visualize", action="store_true",
                        help="Enable visualization using Polyscope.")
    parser.add_argument("--save_viz", dest="save_viz", action="store_true", default=True,
                        help="Save a Polyscope screenshot per frame (enabled by default; headless-friendly).")
    parser.add_argument("--no_save_viz", dest="save_viz", action="store_false",
                        help="Disable saving Polyscope screenshots.")
    parser.add_argument("--viz_dir", type=str, default=None,
                        help="Directory to save visualization screenshots. If set, screenshots go to viz_dir/<frame_id>/. "
                             "If unset, screenshots are saved alongside output weights (same directory as .npy outputs).")
    parser.add_argument("--viz_width", type=int, default=1280,
                        help="Width of saved visualization screenshots in pixels.")
    parser.add_argument("--viz_height", type=int, default=720,
                        help="Height of saved visualization screenshots in pixels.")
    parser.add_argument("--viz_up_axis", type=str, default="auto", choices=["auto", "x", "y", "z"],
                        help="Up axis for screenshot camera framing. 'auto' chooses largest bbox extent.")
    parser.add_argument("--viz_view_axis", type=str, default="auto", choices=["auto", "x", "y", "z"],
                        help="View axis for screenshot camera framing. 'auto' chooses smallest bbox extent.")
    parser.add_argument("--viz_backend", type=str, default=None,
                        help="Optional polyscope backend override (e.g. 'openGL3_egl'). If unset, polyscope auto-selects.")
    parser.add_argument("--full_nerf_mesh_path_template", type=str, default=None,
                        help="Template string for the full NeRF mesh path when target_mesh is 'full_nerf'. " \
                             "Must contain '{frame_id}', e.g. 'data/0365/mesh/labeled/vis-labeled-mesh-f{frame_id}.ply'.")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix after frame ID.")
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=['thuman2', 'avatarrex', 'generic', 'mvhumannet', 'talkbody4d'],
        default='generic',
        help="Type of the dataset being processed. Affects input/output file path construction."
    )


    parser.add_argument("--source_mesh_path", type=str, default=None,
                        help="Explicit source mesh path (OBJ/PLY). If set, bypasses dataset layout inference.")
    parser.add_argument("--source_lbs_weights_path", type=str, default=None,
                        help="Explicit source LBS weights path (NPY) aligned with source mesh vertices.")
    parser.add_argument("--target_mesh_path", type=str, default=None,
                        help="Explicit target mesh path (OBJ/PLY) to receive transferred weights.")
    parser.add_argument("--out_inpainted_weights_path", type=str, default=None,
                        help="Optional explicit output path for inpainted weights (NPY). If relative, resolved under output_dir.")
    parser.add_argument("--out_smoothed_weights_path", type=str, default=None,
                        help="Optional explicit output path for smoothed weights (NPY). If relative, resolved under output_dir.")
    parser.add_argument("--distance_threshold_m", type=float, default=0.01,
                        help="Closest-surface match distance threshold in meters.")
    parser.add_argument("--angle_threshold_deg", type=float, default=30.0,
                        help="Closest-surface match normal angle threshold in degrees.")
    parser.add_argument("--explicit_frame_id", type=str, default="0000",
                        help="Frame id label used for visualization subdir naming in explicit IO mode.")

    args = parser.parse_args()


    if args.target_mesh == "full_nerf" and args.full_nerf_mesh_path_template is None:
        print("Error: --full_nerf_mesh_path_template must be provided when --target_mesh is 'full_nerf'.")
        sys.exit(1)

    explicit_mode = bool(args.source_mesh_path or args.source_lbs_weights_path or args.target_mesh_path)
    if explicit_mode:
        missing = []
        if not args.source_mesh_path:
            missing.append("--source_mesh_path")
        if not args.source_lbs_weights_path:
            missing.append("--source_lbs_weights_path")
        if not args.target_mesh_path:
            missing.append("--target_mesh_path")
        if missing:
            print(
                "Error: explicit IO mode requires all of: "
                "--source_mesh_path --source_lbs_weights_path --target_mesh_path. "
                f"Missing: {missing}"
            )
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    data_dir = Path(args.data_dir)
    single_frame_layout = (data_dir / "smpl_body.obj").exists() and (data_dir / "smpl_lbs_weights.npy").exists()
    if explicit_mode:
        frame_id_list = [str(args.explicit_frame_id)]
        if args.frame_list or args.frame_start != 0 or args.frame_end != 10 or args.frame_step != 1:
            print("[Info] explicit IO mode; ignoring --frame_* args and processing one frame.")
    elif args.dataset_type in {'thuman2', 'mvhumannet', 'talkbody4d'} or single_frame_layout:
        frame_id_list = ["0000"]
        if args.frame_list or args.frame_start != 0 or args.frame_end != 10 or args.frame_step != 1:
            print(f"[Info] dataset_type={args.dataset_type} (or detected single-frame layout); ignoring --frame_* args and processing one frame.")
    else:
        frame_id_list = []
        if args.frame_list:
            frame_id_list = args.frame_list
            print(f"Processing explicit frame list: {frame_id_list}")
        else:
            if args.frame_start is not None and args.frame_end is not None:
                for i in range(args.frame_start, args.frame_end, args.frame_step):
                    frame_id_list.append(f"{i:0{args.frame_format_digits}d}{args.suffix}")
                print(f"Processing frame range: {args.frame_start}-{args.frame_end} step {args.frame_step} ({len(frame_id_list)} frames)")
            else:
                print("No frames specified via --frame_list or --frame_start/--frame_end. Exiting.")
                return

    if not frame_id_list:
        print("No frames to process. Exiting.")
        return


    if args.save_viz or args.visualize:
        _init_polyscope(args)


    for frame_id_str in frame_id_list:
        process_frame(args, frame_id_str)


        ps = getattr(args, "_ps", None)
        if args.visualize and ps is not None and (not hasattr(ps, "is_headless") or not ps.is_headless()):
            try:
                ps.show()
            except Exception as e:
                print(f"Error showing polyscope: {e}")


if __name__ == "__main__":
    main()
