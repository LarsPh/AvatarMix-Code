import argparse
import os, sys
import cv2
import trimesh
import numpy as np
import random
import math
import random
from tqdm import tqdm
import gc


from functools import partial
from multiprocessing import Pool, Queue
import multiprocessing as mp


import numba

numba.config.THREADING_LAYER = "workqueue"

sys.path.append(os.path.join(os.getcwd()))


def render_sides(
    render_types, rndr, rndr_smpl, view_id, save_folder, subject, smpl_type, side
):

    if "normal" in render_types:
        opengl_util.render_result(
            rndr,
            1,
            os.path.join(save_folder, subject, f"normal_{side}", f"{view_id}.png"),
        )

    if "depth" in render_types:
        opengl_util.render_result(
            rndr, 2, os.path.join(save_folder, subject, f"depth_{side}", f"{view_id}.png")
        )

    if smpl_type != "none" and rndr_smpl is not None:

        opengl_util.render_result(
            rndr_smpl,
            1,
            os.path.join(save_folder, subject, f"T_normal_{side}", f"{view_id}.png"),
        )

        if "depth" in render_types:
            opengl_util.render_result(
                rndr_smpl,
                2,
                os.path.join(save_folder, subject, f"T_depth_{side}", f"{view_id}.png"),
            )


def are_all_rendered(save_folder, subject, view_id, render_types, render_smpl, render_back):
    all_rendered = True
    sides = ["F"] if not render_back else ["F", "B"]
    for side in sides:
        if "normal" in render_types:
            path = os.path.join(save_folder, subject, f"normal_{side}", f"{view_id}.png")
            all_rendered = all_rendered and os.path.exists(path)
        if "depth" in render_types:
            path = os.path.join(save_folder, subject, f"depth_{side}", f"{view_id}.png")
            all_rendered = all_rendered and os.path.exists(path)
        if render_smpl:
            path = os.path.join(
                save_folder, subject, f"T_normal_{side}", f"{view_id}.png"
            )
            all_rendered = all_rendered and os.path.exists(path)

            if "depth" in render_types:
                path = os.path.join(
                    save_folder, subject, f"T_depth_{side}", f"{view_id}.png"
                )
                all_rendered = all_rendered and os.path.exists(path)
    return all_rendered


def render_subject(
    subject,
    dataset,
    save_folder,
    rotations_params,
    size,
    render_types,
    egl,
    use_perspective=False,
    save_calib_only=False,
    save_joint_only=False,
    disable_overwrite=False,
    render_smpl=False,
    render_back=False,
    save_smpl_obj=False,
    multi_process=False,
):

    if multi_process:
        gpu_id = queue.get()

    try:


        if multi_process:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        initialize_GL_context(width=size, height=size, egl=egl)

        if not use_perspective:
            scale = 100.0
        else:
            scale = 1.0
        up_axis = 1
        smpl_type = "smplx"

        mesh_file = os.path.join(f"./data/{dataset}/scans/{subject}", f"{subject}.obj")
        smplx_file = f"./data/{dataset}/{smpl_type}/{subject}.obj"
        tex_file = f"./data/{dataset}/scans/{subject}/material0.jpeg"
        fit_file = f"./data/{dataset}/{smpl_type}/{subject}.pkl"

        vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
            mesh_file, with_normal=True, with_texture=True
        )


        if not use_perspective:
            scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
        else:
            scan_scale = 1.0


        rescale_fitted_body, joints = load_fit_body(
            fit_file, scale, smpl_type=smpl_type, smpl_gender="male"
        )
        if save_joint_only:
            joint_file = os.path.join(
                f"./data/{dataset}/joints/{subject}", f"{subject}.npy"
            )
            os.makedirs(os.path.dirname(joint_file), exist_ok=True)
            np.save(joint_file, joints)
            return

        if save_smpl_obj:
            os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
            trimesh.Trimesh(
                rescale_fitted_body.vertices / scale, rescale_fitted_body.faces
            ).export(smplx_file)

        vertices *= scale
        vmin = vertices.min(0)
        vmax = vertices.max(0)
        vmed = joints[0]
        vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
        height = vmax[up_axis] - vmin[up_axis]

        if not save_calib_only and render_smpl:
            rndr_smpl = ColorRender(width=size, height=size, egl=egl)
            rndr_smpl.set_mesh(
                rescale_fitted_body.vertices,
                rescale_fitted_body.faces,
                rescale_fitted_body.vertices,
                rescale_fitted_body.vertex_normals,
            )
            if not use_perspective:
                rndr_smpl.set_norm_mat(scan_scale, vmed)
            else:
                rndr_smpl.set_norm_mat(scan_scale, np.array([0, 0, 0]))
        else:
            rndr_smpl = None


        if not use_perspective:
            cam = Camera(width=size, height=size)
            cam.ortho_ratio = 0.4 * (512 / size)
        if not save_calib_only:
            prt, face_prt = prt_util.computePRT(mesh_file, scale, 10, 2)
            rndr = PRTRender(width=size, height=size, ms_rate=4, egl=egl)


            texture_image = cv2.cvtColor(cv2.imread(tex_file), cv2.COLOR_BGR2RGB)

            tan, bitan = compute_tangent(normals)
            if not use_perspective:
                rndr.set_norm_mat(scan_scale, vmed)
            else:
                rndr.set_norm_mat(scan_scale, np.array([0, 0, 0]))
            rndr.set_mesh(
                vertices,
                faces,
                normals,
                faces_normals,
                textures,
                face_textures,
                prt,
                face_prt,
                tan,
                bitan,
                np.zeros((vertices.shape[0], 3)),
            )
            rndr.set_albedo(texture_image)
            del prt, face_prt, texture_image
            gc.collect()

        for yaw_deg, pitch_deg in rotations_params:
            view_id = f"{yaw_deg:03d}_p{pitch_deg:+03d}"

            if disable_overwrite:


                if are_all_rendered(save_folder, subject, view_id, render_types, render_smpl, render_back):
                    continue


            R_combined = opengl_util.make_rotate(math.radians(pitch_deg), math.radians(yaw_deg), 0)

            if use_perspective:

                cam = Camera(width=size, height=size)
                cam_center = cam.center
                height = vertices.max(0)[up_axis] - vertices.min(0)[up_axis]

                cam_center = vmed.numpy() + height * (cam_center - vmed.numpy())


                cam_center_normed = cam_center - vmed.numpy()

                cam_center_rotated = R_combined @ cam_center_normed
                cam.center = cam_center_rotated + vmed.numpy()

                cam_dir = vmed.numpy() - cam.center
                cam_dir = cam_dir / np.linalg.norm(cam_dir)
                cam.direction = cam_dir


                cam_up_rotated = R_combined @ cam.up
                cam_up_rotated = cam_up_rotated / np.linalg.norm(cam_up_rotated)


                cam_right = np.cross(cam_dir, cam_up_rotated)
                cam_right = cam_right / np.linalg.norm(cam_right)
                cam.right = cam_right


                cam.up = np.cross(cam_right, cam_dir)

                human_radius_xy = (
                    np.linalg.norm(vertices.max(0)[:2] - vertices.min(0)[:2]) / 2
                )
                cam.near = np.linalg.norm(cam_center - vmed.numpy()) - human_radius_xy
                cam.near = max(0.001, cam.near)
                cam.far = np.linalg.norm(cam_center - vmed.numpy()) + human_radius_xy
            else:
                cam.near = -100
                cam.far = 100

            cam.sanity_check()


            if not use_perspective:
                R_model_transform = R_combined
            else:
                R_model_transform = np.eye(3)

            if not save_calib_only and smpl_type != "none":
                rndr.rot_matrix = R_model_transform
                rndr.set_camera(cam)
                if render_smpl:
                    rndr_smpl.rot_matrix = R_model_transform
                    rndr_smpl.set_camera(cam)

            dic = {
                "scale": scan_scale,
                "center": vmed,
                "R": R_model_transform,
            }

            if not save_calib_only and "light" in render_types:


                shs = np.load(".env_sh.npy")
                sh_id = random.randint(0, shs.shape[0] - 1)
                sh = shs[sh_id]
                sh_angle = 0.2 * np.pi * (random.random() - 0.5)
                sh = opengl_util.rotateSH(sh, opengl_util.make_rotate(0, sh_angle, 0).T)
                dic.update({"sh": sh})

                rndr.set_sh(sh)
                rndr.analytic = False
                rndr.use_inverse_depth = False


            if not use_perspective:
                dic["ortho_ratio"] = cam.ortho_ratio
                calib = opengl_util.load_calib(dic, render_size=size)
            else:
                intrinsic = cam.get_intrinsic_matrix()
                intrinsic_44 = np.eye(4)
                intrinsic_44[:3, :3] = intrinsic
                cam_extr = cam.get_extrinsic_matrix()
                cam_extr_44 = np.eye(4)
                cam_extr_44[:3, :] = cam_extr
                extrinsic = cam_extr_44
                intrinsic_44_new = intrinsic_44
                calib = np.concatenate([extrinsic, intrinsic_44_new], axis=0)
                near_far = np.array([[cam.near, cam.far]])

            export_calib_file = os.path.join(
                save_folder, subject, "calib", f"{view_id}.txt"
            )
            os.makedirs(os.path.dirname(export_calib_file), exist_ok=True)
            np.savetxt(export_calib_file, calib)
            if use_perspective:

                with open(export_calib_file, "ab") as f:
                    np.savetxt(f, near_far)
            if save_calib_only:
                continue


            rndr.display()
            if render_smpl:
                rndr_smpl.display()

            if 'rgb' in render_types:
                opengl_util.render_result(
                    rndr, 0, os.path.join(save_folder, subject, "render", f"{view_id}.png")
                )

            render_sides(
                render_types, rndr, rndr_smpl, view_id, save_folder, subject, smpl_type, "F"
            )


            if render_back:
                if not use_perspective:
                    cam.near = 100
                    cam.far = -100
                else:
                    tmp = cam.near
                    cam.near = cam.far
                    cam.far = tmp

                cam.sanity_check()
                rndr.set_camera(cam)
                rndr_smpl.set_camera(cam)

                rndr.display()
                rndr_smpl.display()

                render_sides(
                    render_types, rndr, rndr_smpl, view_id, save_folder, subject, smpl_type, "B"
                )


        if render_smpl:
            rndr_smpl.cleanup()
            del rndr_smpl
        rndr.cleanup()
        del rndr
        gc.collect()


        del vertices, faces, normals, faces_normals, textures, face_textures
        del rescale_fitted_body, joints
        gc.collect()

    finally:
        if multi_process:
            queue.put(gpu_id)

        glFlush()
        glFinish()
        release_GL_context()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dataset", "--dataset", type=str, default="thuman2", help="dataset name"
    )
    parser.add_argument(
        "-out_dir", "--out_dir", type=str, default="./debug", help="output dir"
    )
    parser.add_argument(
        "-num_views", "--num_views", type=int, default=36, help="number of views"
    )
    parser.add_argument("-size", "--size", type=int, default=512, help="render size")
    parser.add_argument(
        "-debug",
        "--debug",
        action="store_true",
        help="debug mode, only render one subject",
    )
    parser.add_argument(
        "-headless",
        "--headless",
        type=bool,
        default=False,
        help="headless rendering with EGL",
    )
    parser.add_argument(
        "-perspective",
        "--perspective",
        type=bool,
        default=True,
        help="perspective rendering",
    )
    parser.add_argument(
        "-save_calib_only",
        "--save_calib_only",
        action="store_true",
        help="only save calibration",
    )
    parser.add_argument(
        "-use_multi_process",
        "--use_multi_process",
        type=bool,
        default=False,
        help="disable multi-process rendering",
    )
    parser.add_argument(
        "-save_joint_only",
        "--save_joint_only",
        action="store_true",
        help="only save joint",
    )
    parser.add_argument(
        "-disable_overwrite",
        "--disable_overwrite",
        action="store_true",
        help="disable overwrite",
    )
    parser.add_argument(
        "-rotation_offset",
        "--rotation_offset",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-render_types",
        "--render_types",
        type=str,

        default="rgb,depth",
        help="render types",
    )
    parser.add_argument(
        "-render_smpl",
        "--render_smpl",
        action="store_true",
        help="render smpl",
    )
    parser.add_argument(
        "-render_back",
        "--render_back",
        action="store_true",
        help="render back",
    )
    parser.add_argument(
        "-save_smpl_obj",
        "--save_smpl_obj",
        action="store_true",
        help="save smpl obj",
    )
    parser.add_argument(
        "-start_subject",
        "--start_subject",
        type=int,
        default=0,
        help="start subject",
    )
    parser.add_argument(
        "-end_subject",
        "--end_subject",
        type=int,
        default=-1,
        help="end subject",
    )
    parser.add_argument(
        "-out_dir_subfix",
        "--out_dir_subfix",
        type=str,
        default="_fixed",
        help="output dir subfix",
    )
    parser.add_argument(
        "-additional_angles",
        "--additional_angles",
        action="store_true",
        help="render additional angles",
    )
    parser.add_argument(
        "-additional_pitch_views",
        "--additional_pitch_views",
        action="store_true",
        help="render additional views with pitch rotations (18 views at +20deg, 18 views at -20deg)",
    )
    parser.add_argument(
        "-no_parent_dir",
        "--no_parent_dir",
        action="store_true",
        help="do not create parent directory",
    )
    args = parser.parse_args()


    base_yaw_step = 360 // args.num_views
    yaw_rotations_deg_only = list(range(0, 360, base_yaw_step))
    yaw_rotations_deg_only = [(r + args.rotation_offset) % 360 for r in yaw_rotations_deg_only]

    if args.additional_angles:
        assert args.disable_overwrite, "you might need to skip original angles when rendering additional angle"

        temp_additional_yaws = []
        for base_yaw in yaw_rotations_deg_only:
            for delta in [-5, -3, -1, 1, 3, 5]:
                angle = (base_yaw + delta + 360) % 360
                temp_additional_yaws.append(angle)

        yaw_rotations_deg_only = sorted(list(set(yaw_rotations_deg_only + temp_additional_yaws)))
    else:
        yaw_rotations_deg_only = sorted(list(set(yaw_rotations_deg_only)))


    rotations_to_render = []


    for yaw_deg in yaw_rotations_deg_only:
        rotations_to_render.append((yaw_deg, 0))

    if args.additional_pitch_views:


        pitch_up_num_views = 18
        pitch_up_yaw_step = 360 // pitch_up_num_views
        pitch_up_yaw_offset = 5
        pitch_up_angle_deg = 20
        for i in range(pitch_up_num_views):
            yaw_deg = (i * pitch_up_yaw_step + pitch_up_yaw_offset + 360) % 360
            rotations_to_render.append((yaw_deg, pitch_up_angle_deg))


        pitch_down_num_views = 18
        pitch_down_yaw_step = 360 // pitch_down_num_views
        pitch_down_yaw_offset = -5
        pitch_down_angle_deg = -20
        for i in range(pitch_down_num_views):
            yaw_deg = (i * pitch_down_yaw_step + pitch_down_yaw_offset + 360) % 360
            rotations_to_render.append((yaw_deg, pitch_down_angle_deg))


        rotations_to_render = sorted(list(set(rotations_to_render)), key=lambda rp: (rp[1], rp[0]))


    if args.headless:
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
    else:
        os.environ["PYOPENGL_PLATFORM"] = ""
    if args.perspective:
        print("Perspective rendering.")
    else:
        print("Default: orthographic rendering.")


    import lib.renderer.opengl_util as opengl_util
    from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent
    import lib.renderer.prt_util as prt_util
    from lib.renderer.gl.init_gl import initialize_GL_context, release_GL_context
    from lib.renderer.gl.prt_render import PRTRender
    from lib.renderer.gl.color_render import ColorRender
    from lib.renderer.camera import Camera
    from OpenGL.GL import glFlush, glFinish


    if not args.headless:
        import OpenGL.GLUT as GLUT
        GLUT.glutInit([])

    print(
        f"Start Rendering {args.dataset} with {args.num_views} views, {args.size}x{args.size} size."
    )

    current_out_dir = args.out_dir
    if not args.no_parent_dir:
        current_out_dir = os.path.join(current_out_dir, f"{args.dataset}_{args.num_views}views")
        if args.rotation_offset > 0:
            current_out_dir += f"_{args.rotation_offset}offset"
        if args.perspective:
            current_out_dir += "_perspective"
        if args.out_dir_subfix:
            current_out_dir += f"_{args.out_dir_subfix}"
    os.makedirs(current_out_dir, exist_ok=True)
    print(f"Output dir: {current_out_dir}")

    subjects = np.loadtxt(f"./data/{args.dataset}/all.txt", dtype=str)
    render_types = args.render_types.split(",") if args.render_types else []

    if args.debug:

        subjects = ["0070"]
    else:

        start_idx = args.start_subject
        end_idx = args.end_subject
        if end_idx == -1:
            end_idx = len(subjects)
        subjects = subjects[start_idx:end_idx]
        print(f"Rendering from subject {start_idx} to {end_idx}")

    print(f"Rendering types: {render_types}")

    NUM_GPUS = 1
    PROC_PER_GPU = mp.cpu_count() // NUM_GPUS

    queue = Queue()


    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    if (not args.debug) and args.use_multi_process:
        with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
            for _ in tqdm(
                pool.imap_unordered(
                    partial(
                        render_subject,
                        dataset=args.dataset,
                        save_folder=current_out_dir,
                        rotations_params=rotations_to_render,
                        size=args.size,
                        egl=args.headless,
                        render_types=render_types,
                        use_perspective=args.perspective,
                        save_calib_only=args.save_calib_only,
                        save_joint_only=args.save_joint_only,
                        render_smpl=args.render_smpl,
                        render_back=args.render_back,
                        save_smpl_obj=args.save_smpl_obj,
                        multi_process=True,
                    ),
                    subjects,
                ),
                total=len(subjects),
            ):
                pass

        pool.close()
        pool.join()
    else:
        print("Render with single process.")
        for subject in tqdm(subjects):
            render_subject(
                subject,
                dataset=args.dataset,
                save_folder=current_out_dir,
                rotations_params=rotations_to_render,
                size=args.size,
                egl=args.headless,
                render_types=render_types,
                use_perspective=args.perspective,
                save_calib_only=args.save_calib_only,
                save_joint_only=args.save_joint_only,
                disable_overwrite=args.disable_overwrite,
                render_smpl=args.render_smpl,
                render_back=args.render_back,
                save_smpl_obj=args.save_smpl_obj,
                multi_process=False,
            )

    print("Finish Rendering.")
