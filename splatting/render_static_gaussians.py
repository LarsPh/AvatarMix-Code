#!/usr/bin/env python3


import os
import cv2
import json
import math
import torch
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm
from loguru import logger

from model.splatting_avatar_model import SplattingAvatarModel
from dataset.dataset_helper import make_frameset_data
from utils.graphics_utils import getWorld2View2
from gaussian_renderer import network_gui
from model import libcore
from scene.cameras import Camera as SceneCamera
import numpy.linalg as npl


def write_tensor_image(fn, tensor, rgb2bgr=False):

    if len(tensor.shape) == 3:
        if tensor.shape[0] == 3 or tensor.shape[0] == 4:
            tensor = tensor.permute([1, 2, 0])

    if rgb2bgr:
        if tensor.shape[2] == 3:
            tensor = tensor[:, :, [2, 1, 0]]
        else:
            tensor = tensor[:, :, [2, 1, 0, 3]]

    cv2.imwrite(fn, (tensor.clamp(0, 1) * 255).detach().cpu().numpy().astype(np.uint8))


def generate_mask_from_rendered_image(rendered_image, bg_color, white_threshold=0.95, black_threshold=0.05):


    if isinstance(rendered_image, torch.Tensor):
        img_np = rendered_image.detach().cpu().numpy()
        if img_np.shape[0] == 3:
            img_np = img_np.transpose(1, 2, 0)
    else:
        img_np = rendered_image


    if bg_color == 'white':
        bg_mask = np.all(img_np >= white_threshold, axis=-1)
    elif bg_color == 'black':
        bg_mask = np.all(img_np <= black_threshold, axis=-1)
    else:
        raise ValueError(f"Invalid background color: {bg_color}")


    mask = (~bg_mask).astype(np.uint8) * 255
    return mask


def extract_subject_name_from_ply(ply_path):

    ply_name = Path(ply_path).stem


    if ply_name.startswith('gt_aligned_head_'):
        ply_name = ply_name[len('gt_aligned_head_'):]
    if ply_name.startswith('body_donator_full'):
        ply_name = ply_name[len('body_donator_full_'):]

    if '_rigidhead' in str(ply_name):
        ply_name = str(ply_name).replace('_rigidhead', '')
        ply_name = Path(ply_name)


    return ply_name


def create_output_directories(base_output_dir, ply_path, camera_ids, explicit_subject_name=None):

    if explicit_subject_name:
        subject_name = explicit_subject_name
    else:
        subject_name = extract_subject_name_from_ply(ply_path)

    subject_output_dir = Path(base_output_dir) / subject_name
    subject_output_dir.mkdir(parents=True, exist_ok=True)

    for cam_id in camera_ids:

        cam_dir = subject_output_dir / cam_id
        cam_dir.mkdir(parents=True, exist_ok=True)

        mask_dir = cam_dir / "mask" / "pha"
        mask_dir.mkdir(parents=True, exist_ok=True)

    return subject_output_dir


def get_camera_ids_from_calibration(calib_file):

    with open(calib_file, 'r') as f:
        calib_data = json.load(f)

    camera_ids = list(calib_data.keys())
    camera_ids.sort()
    return camera_ids


def apply_camera_transformation(scene_camera, transform_matrix):

    if transform_matrix is None:
        return scene_camera


    w2c_matrix = scene_camera.world_view_transform.cpu().numpy().transpose()


    w2c_new = w2c_matrix @ transform_matrix.cpu().numpy()
    R_w2c_new = w2c_new[:3, :3]
    T_w2c_new = w2c_new[:3, 3]


    scene_camera.world_view_transform = torch.tensor(getWorld2View2(R_w2c_new.T, T_w2c_new, np.array([0.0, 0.0, 0.0]), 1.0), dtype=torch.float32, device=scene_camera.world_view_transform.device).transpose(0, 1)


    scene_camera.full_proj_transform = (scene_camera.world_view_transform.unsqueeze(0).bmm(scene_camera.projection_matrix.unsqueeze(0))).squeeze(0)


    scene_camera.camera_center = scene_camera.world_view_transform.inverse()[3, :3]

    return scene_camera


def _look_at_R_c2w(pos: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:

    forward = (target - pos).astype(np.float32)
    fn = float(npl.norm(forward))
    if fn < 1e-8:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / fn

    up = up.astype(np.float32)
    upn = float(npl.norm(up))
    if upn < 1e-8:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = up / upn

    right = np.cross(forward, up)
    rn = float(npl.norm(right))
    if rn < 1e-8:

        up_alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up_alt)
        rn = float(npl.norm(right))
    right = right / max(rn, 1e-8)
    true_up = np.cross(right, forward)
    return np.stack([right, true_up, forward], axis=1).astype(np.float32)


def _render_canonical_front(args, gs_model, pipe, device: torch.device, frame_id: str) -> None:

    cam_id = args.canonical_cam_id
    subject_output_dir = create_output_directories(
        args.output_dir,
        args.input_gs_ply,
        camera_ids=[cam_id],
        explicit_subject_name=args.output_subject_name,
    )


    if args.canonical_width is not None and args.canonical_height is not None:
        width = int(args.canonical_width)
        height = int(args.canonical_height)
    else:
        calib_file = Path(args.calib_file) if getattr(args, "calib_file", None) else (Path(args.dat_dir) / "calibration_full.json")
        with open(calib_file, "r") as f:
            calib_data = json.load(f)
        if not calib_data:
            raise ValueError(f"Empty calibration file: {calib_file}")
        first_cam = calib_data[next(iter(calib_data.keys()))]
        width = int(first_cam["imgSize"][0])
        height = int(first_cam["imgSize"][1])


    fov_deg = float(args.canonical_fov_deg)
    fov_rad = math.radians(fov_deg)
    fx = 0.5 * float(width) / math.tan(0.5 * fov_rad)
    fy = fx
    cx = 0.5 * float(width)
    cy = 0.5 * float(height)

    target_y = float(args.canonical_target_y)
    dist = float(args.canonical_distance)
    z_sign = int(getattr(args, "canonical_z_sign", 1))
    if z_sign not in (-1, 1):
        raise ValueError(f"--canonical_z_sign must be -1 or 1, got {z_sign}")
    y_sign = int(getattr(args, "canonical_y_sign", 1))
    if y_sign not in (-1, 1):
        raise ValueError(f"--canonical_y_sign must be -1 or 1, got {y_sign}")


    yaw_deg = float(getattr(args, "canonical_yaw_deg", 0.0))
    yaw_rad = math.radians(yaw_deg)
    pitch_deg = float(getattr(args, "canonical_pitch_deg", 0.0))
    pitch_rad = math.radians(pitch_deg)

    base = np.array([0.0, 0.0, float(z_sign) * dist], dtype=np.float32)
    c_yaw = float(math.cos(yaw_rad))
    s_yaw = float(math.sin(yaw_rad))
    Ry = np.array([[c_yaw, 0.0, s_yaw], [0.0, 1.0, 0.0], [-s_yaw, 0.0, c_yaw]], dtype=np.float32)
    target = np.array([0.0, target_y, 0.0], dtype=np.float32)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)


    vec = (Ry @ base).astype(np.float32)


    if abs(pitch_rad) > 1e-10:
        forward_tmp = (-vec).astype(np.float32)
        fn = float(npl.norm(forward_tmp))
        if fn < 1e-8:
            forward_tmp = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward_tmp = forward_tmp / fn
        right_axis = np.cross(forward_tmp, up).astype(np.float32)
        rn = float(npl.norm(right_axis))
        if rn < 1e-8:
            right_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right_axis = right_axis / rn

        c_p = float(math.cos(pitch_rad))
        s_p = float(math.sin(pitch_rad))

        vec = (vec * c_p) + (np.cross(right_axis, vec) * s_p) + (right_axis * (float(np.dot(right_axis, vec)) * (1.0 - c_p)))
        vec = vec.astype(np.float32)

    pos = (target + vec).astype(np.float32)


    flip_yz = bool(args.canonical_flip_yz)
    if flip_yz:
        pos[1] *= -1.0
        pos[2] *= -1.0
        target[1] *= -1.0
        target[2] *= -1.0
        up[1] *= -1.0
        up[2] *= -1.0


    if y_sign == -1:
        pos[1] *= -1.0
        target[1] *= -1.0
        up[1] *= -1.0

    logger.info(
        f"[CanonicalFront] cam_id={cam_id} W×H={width}×{height} fov_deg={fov_deg} "
        f"fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}"
    )
    logger.info(
        f"[CanonicalFront] yaw_deg={yaw_deg} pitch_deg={pitch_deg} pos={pos.tolist()} target={target.tolist()} up={up.tolist()} flip_yz={flip_yz} y_sign={y_sign}"
    )

    R_c2w = _look_at_R_c2w(pos=pos, target=target, up=up)
    R_w2c = R_c2w.T
    t_w2c = -R_w2c @ pos

    dummy_image = torch.zeros(4, height, width, dtype=torch.float32)
    scene_camera = SceneCamera(
        colmap_id=0,
        R=R_c2w,
        T=t_w2c,
        image=dummy_image,
        gt_alpha_mask=None,
        image_name=f"{cam_id}_{frame_id}.jpg",
        uid=0,
        w=width,
        h=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        data_device=("cuda" if device.type == "cuda" else "cpu"),
    ).to(device)

    bg_color_list = [0.0, 0.0, 0.0] if args.bg_color == "black" else [1.0, 1.0, 1.0]
    render_pkg = gs_model.render_to_camera(
        scene_camera,
        pipe,
        background=torch.tensor(bg_color_list, dtype=torch.float32, device=device),
    )
    rendered_image = render_pkg["render"]
    mask = generate_mask_from_rendered_image(rendered_image, bg_color=args.bg_color, white_threshold=0.95, black_threshold=0.05)

    image_path = Path(subject_output_dir) / cam_id / f"{frame_id}.jpg"
    mask_path = Path(subject_output_dir) / cam_id / "mask" / "pha" / f"{frame_id}.png"
    write_tensor_image(str(image_path), rendered_image, rgb2bgr=True)
    cv2.imwrite(str(mask_path), mask)
    logger.info(f"[CanonicalFront] Wrote {image_path}")


    try:
        meta = {
            "canonical_cam_id": cam_id,
            "width": width,
            "height": height,
            "fov_deg": fov_deg,
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "canonical_distance": float(dist),
            "canonical_target_y": float(target_y),
            "canonical_flip_yz": bool(flip_yz),
            "canonical_y_sign": int(y_sign),
            "canonical_pitch_deg": float(pitch_deg),
            "canonical_yaw_deg": float(yaw_deg),
            "pos": [float(x) for x in pos.tolist()],
            "target": [float(x) for x in target.tolist()],
            "up": [float(x) for x in up.tolist()],
            "R_c2w": R_c2w.tolist(),
            "t_w2c": [float(x) for x in t_w2c.tolist()],
        }
        meta_path = Path(subject_output_dir) / cam_id / "canonical_camera.json"
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        logger.warning(f"[CanonicalFront] Failed to write camera metadata: {e}")


def _axes_for_up(up_idx: int):

    if up_idx == 0:
        return (1, 2), np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if up_idx == 1:
        return (0, 2), np.array([0.0, 1.0, 0.0], dtype=np.float32)

    return (0, 1), np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _parse_up_axis_str(up_axis: str):

    ua = (up_axis or "y").lower()
    if ua == "x":
        return np.array([1.0, 0.0, 0.0], dtype=np.float32), (1, 2)
    if ua == "-x":
        return np.array([-1.0, 0.0, 0.0], dtype=np.float32), (1, 2)
    if ua == "z":
        return np.array([0.0, 0.0, 1.0], dtype=np.float32), (0, 1)
    if ua == "-z":
        return np.array([0.0, 0.0, -1.0], dtype=np.float32), (0, 1)
    if ua == "-y":
        return np.array([0.0, -1.0, 0.0], dtype=np.float32), (0, 2)

    return np.array([0.0, 1.0, 0.0], dtype=np.float32), (0, 2)


def compute_yaw_orbit_from_calibration(calib_file, pitch_tag="p+00"):

    with open(calib_file, 'r') as f:
        calib_data = json.load(f)

    all_cam_ids = list(calib_data.keys())
    if pitch_tag is not None:
        yaw_cam_ids = [cid for cid in all_cam_ids if pitch_tag in cid]
    else:
        yaw_cam_ids = all_cam_ids


    if not yaw_cam_ids:
        yaw_cam_ids = all_cam_ids
    if not yaw_cam_ids:
        raise ValueError(f"No cameras found in calibration file: {calib_file}")

    centers = []
    for cam_id in yaw_cam_ids:
        cam = calib_data[cam_id]
        R = np.array(cam['R'], dtype=np.float32)
        T = np.array(cam['T'], dtype=np.float32).reshape(3)

        w2c = getWorld2View2(R, T, translate=np.array([0.0, 0.0, 0.0]), scale=1.0)
        c2w = np.linalg.inv(w2c)
        C = c2w[:3, 3]
        centers.append(C)

    centers = np.stack(centers, axis=0)


    plane_axes = (0, 2)
    vertical_idx = 1
    plane_coords = centers[:, list(plane_axes)]
    if plane_coords.shape[0] >= 3:

        A_mat = np.concatenate([2 * plane_coords, np.ones((plane_coords.shape[0], 1))], axis=1)
        b_vec = (plane_coords ** 2).sum(axis=1)
        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        c0, c1, c_const = sol[0], sol[1], sol[2]
        radius = math.sqrt(max(c_const + c0 * c0 + c1 * c1, 1e-8))
        center_plane = np.array([c0, c1], dtype=np.float32)
        center = centers.mean(axis=0)
        center[list(plane_axes)] = center_plane
        center[vertical_idx] = centers[:, vertical_idx].mean()
    else:

        center = centers.mean(axis=0)
        plane_center_mean = center[list(plane_axes)]
        radii = np.linalg.norm(plane_coords - plane_center_mean[None, :], axis=1)
        radius = float(radii.mean())


    first_cam = calib_data[yaw_cam_ids[0]]
    K = np.array(first_cam['K'], dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    width, height = int(first_cam['imgSize'][0]), int(first_cam['imgSize'][1])

    return {
        "center": center,
        "radius": radius,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
    }


def build_orbit_cameras(calib_file, num_frames, data_device="cuda",
                        pitch_tag="p+00", up_axis='-y', orbit_direction='cw',
                        start_deg=0.0, radius_scale=1.0):

    info = compute_yaw_orbit_from_calibration(calib_file, pitch_tag=pitch_tag)
    center = info["center"]

    radius = float(info["radius"] * float(radius_scale))
    fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
    width, height = info["width"], info["height"]

    if radius <= 0:
        raise ValueError(f"Computed orbit radius is non-positive ({radius}). Check calibration.")

    cameras = []

    up, plane_axes = _parse_up_axis_str(up_axis)
    up = up.astype(np.float32)
    start_rad = math.radians(float(start_deg))
    for idx in range(num_frames):
        base_angle = 2.0 * np.pi * float(idx) / float(num_frames) + start_rad
        theta = -base_angle if orbit_direction == "cw" else base_angle

        pos = center.copy()

        pos[plane_axes[0]] = center[plane_axes[0]] + radius * np.cos(theta)
        pos[plane_axes[1]] = center[plane_axes[1]] + radius * np.sin(theta)


        forward = center - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:

            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / forward_norm

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:

            up_alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, up_alt)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        true_up = np.cross(right, forward)


        R_c2w = np.stack([right, true_up, forward], axis=1)

        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ pos


        dummy_image = torch.zeros(4, height, width, dtype=torch.float32)

        cam = SceneCamera(
            colmap_id=idx,
            R=R_c2w,
            T=t_w2c,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name=f"orbit_{idx:04d}.png",
            uid=idx,
            w=width,
            h=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            data_device=data_device,
        )
        cameras.append(cam)

    return cameras


def compute_orbit_from_gaussians(gs_model, calib_file, radius_scale=1.2):


    with open(calib_file, 'r') as f:
        calib_data = json.load(f)
    if not calib_data:
        raise ValueError(f"No cameras found in calibration file: {calib_file}")
    first_cam = calib_data[next(iter(calib_data.keys()))]
    K = np.array(first_cam['K'], dtype=np.float32)
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    width, height = int(first_cam['imgSize'][0]), int(first_cam['imgSize'][1])


    with torch.no_grad():
        pts = gs_model.get_xyz.detach().cpu().numpy()
    if pts.shape[0] == 0:
        raise ValueError("Gaussian model has no points; cannot compute orbit.")

    xyz_min = pts.min(axis=0)
    xyz_max = pts.max(axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    extents = xyz_max - xyz_min
    up_idx = int(np.argmax(extents))
    plane_axes, up_vec = _axes_for_up(up_idx)


    plane_pts = pts[:, list(plane_axes)]
    center_plane = center[list(plane_axes)]
    radii = np.linalg.norm(plane_pts - center_plane[None, :], axis=1)
    radius = float(radii.max() * float(radius_scale))

    return {
        "center": center.astype(np.float32),
        "radius": radius,
        "up": up_vec,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "plane_axes": plane_axes,
    }


def build_orbit_cameras_from_gaussians(gs_model, calib_file, num_frames, data_device="cuda",
                                       radius_scale=1.2, orbit_direction="cw",
                                       start_deg=0.0, up_axis_str=None):

    info = compute_orbit_from_gaussians(gs_model, calib_file, radius_scale=radius_scale)
    center = info["center"]
    radius = info["radius"]
    fx, fy, cx, cy = info["fx"], info["fy"], info["cx"], info["cy"]
    width, height = info["width"], info["height"]
    up = info["up"]
    plane_axes = info["plane_axes"]

    if radius <= 0:
        raise ValueError(f"Computed orbit radius is non-positive ({radius}). Check Gaussian cloud.")

    cameras = []

    if isinstance(up_axis_str, str) and up_axis_str.strip().startswith("-"):
        up = -up

    start_rad = math.radians(float(start_deg))
    for idx in range(num_frames):
        base_angle = 2.0 * np.pi * float(idx) / float(num_frames) + start_rad
        theta = -base_angle if orbit_direction == "cw" else base_angle


        pos = center.copy()

        pos[plane_axes[0]] = center[plane_axes[0]] + radius * np.cos(theta)
        pos[plane_axes[1]] = center[plane_axes[1]] + radius * np.sin(theta)


        forward = center - pos
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            forward = forward / forward_norm

        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            up_alt = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            right = np.cross(forward, up_alt)
            right_norm = np.linalg.norm(right)
        right = right / right_norm
        true_up = np.cross(right, forward)

        R_c2w = np.stack([right, true_up, forward], axis=1)
        R_w2c = R_c2w.T
        t_w2c = -R_w2c @ pos

        dummy_image = torch.zeros(4, height, width, dtype=torch.float32)

        cam = SceneCamera(
            colmap_id=idx,
            R=R_c2w,
            T=t_w2c,
            image=dummy_image,
            gt_alpha_mask=None,
            image_name=f"orbit_{idx:04d}.png",
            uid=idx,
            w=width,
            h=height,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            data_device=data_device,
        )
        cameras.append(cam)

    return cameras


def render_static_gaussians(args, config):

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")


    dat_dir = Path(args.dat_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


    if getattr(args, "calib_file", None):
        calib_file = Path(args.calib_file)
    else:
        calib_file = dat_dir / 'calibration_full.json'
    if not calib_file.exists():
        logger.error(f"Calibration file not found: {calib_file}")
        return


    if getattr(args, "render_canonical_front", False):
        if not getattr(args, "render_360_video", None):
            cam_id = getattr(args, "canonical_cam_id", "front")
            _ = create_output_directories(
                args.output_dir,
                args.input_gs_ply,
                camera_ids=[cam_id],
                explicit_subject_name=args.output_subject_name,
            )
        if args.output_subject_name:
            logger.info(f"[CanonicalFront] Using explicit subject name: {args.output_subject_name}")
    else:

        logger.info("Loading dataset for camera information...")
        config.dataset.dat_dir = args.dat_dir

        if getattr(args, "custom_rgb_relpath", None) is not None:
            config.dataset.custom_rgb_relpath = args.custom_rgb_relpath
        if getattr(args, "custom_mask_relpath", None) is not None:
            config.dataset.custom_mask_relpath = args.custom_mask_relpath
        config.dataset.free_gaussians = True
        try:
            dataset = make_frameset_data(config.dataset, split='train')
            logger.info(f"Loaded {len(dataset)} samples from train split")
        except Exception as e:
            logger.error(f"Could not load dataset: {e}")
            return

        if len(dataset) == 0:
            logger.error("No valid samples found in dataset. Check dataset configuration.")
            return

        camera_ids = get_camera_ids_from_calibration(calib_file)
        logger.info(f"Found {len(camera_ids)} cameras: {camera_ids[:5]}...")

        if not getattr(args, "render_360_video", None):

            output_dir = create_output_directories(
                args.output_dir,
                args.input_gs_ply,
                camera_ids,
                explicit_subject_name=args.output_subject_name
            )
            if args.output_subject_name:
                logger.info(f"Using explicit subject name: {args.output_subject_name}")
            logger.info(f"Created output directory structure at: {output_dir}")


    logger.info("Loading Gaussian model in static rendering mode...")
    gs_model = SplattingAvatarModel(
        config.model,
        verbose=True,
        gaussians_are_frozen=True,
        static_rendering=True
    )


    if not args.input_gs_ply or not Path(args.input_gs_ply).exists():
        logger.error(f"Head-swapped PLY file not found: {args.input_gs_ply}")
        return

    gs_model.load_ply(args.input_gs_ply)
    logger.info(f"Loaded head-swapped Gaussians from: {args.input_gs_ply}")


    if args.input_gs_embed and Path(args.input_gs_embed).exists():
        gs_model.load_from_embedding(args.input_gs_embed)
        logger.info(f"Loaded embedding from: {args.input_gs_embed}")


    camera_transform = None
    if args.camera_transform_npz and Path(args.camera_transform_npz).exists():
        try:
            transform_data = np.load(args.camera_transform_npz)
            camera_transform = torch.tensor(transform_data['avg_head_transform'], dtype=torch.float32, device=device)
            logger.info(f"Loaded camera transformation matrix from: {args.camera_transform_npz}")
            logger.info(f"Transformation matrix shape: {camera_transform.shape}")
        except Exception as e:
            logger.warning(f"Failed to load camera transformation matrix: {e}")
            camera_transform = None
    elif args.camera_transform_npz:
        logger.warning(f"Camera transformation matrix file not found: {args.camera_transform_npz}")


    pipe = config.pipe
    if args.gui_ip != 'none':
        network_gui.init(args.gui_ip, args.gui_port)


    frame_format_digits = config.dataset.get('avatarrex_config', {}).get('frame_format_digits', 4)
    frame_id = f"{args.frame_id:0{frame_format_digits}d}"

    if getattr(args, "render_canonical_front", False):
        _render_canonical_front(args, gs_model, pipe, device=device, frame_id=frame_id)
        return


    if not getattr(args, "render_360_video", False):
        logger.info(f"Starting rendering for frame {frame_id}...")


        rendered_cameras = set()


        for sample_idx in tqdm(range(len(dataset)), desc="Rendering cameras"):
            sample = dataset[sample_idx]
            cam_id = sample['cam_id_str']


            if cam_id in rendered_cameras and not args.update_all:
                continue

            try:
                rendered_cameras.add(cam_id)


                scene_camera = sample['scene_cameras'][0].to(device)


                if camera_transform is not None:
                    scene_camera = apply_camera_transformation(scene_camera, camera_transform)


                bg_color_list = [0.0, 0.0, 0.0] if args.bg_color == 'black' else [1.0, 1.0, 1.0]
                render_pkg = gs_model.render_to_camera(
                    scene_camera,
                    pipe,
                    background=torch.tensor(bg_color_list, dtype=torch.float32, device=device)
                )

                rendered_image = render_pkg['render']


                mask = generate_mask_from_rendered_image(rendered_image, bg_color=args.bg_color, white_threshold=0.95, black_threshold=0.05)


                if camera_transform is not None:

                    if args.output_subdir:
                        image_base_dir = args.output_subdir
                    else:

                        image_base_dir = "head_aligned_rigidhead" if "_rigidhead" in args.input_gs_ply else "head_aligned"

                    image_dir = output_dir / cam_id / image_base_dir
                    image_dir.mkdir(parents=True, exist_ok=True)
                    image_path = image_dir / f"{frame_id}.png"
                    mask_dir = output_dir / cam_id / image_base_dir / "mask" / "pha"
                    mask_dir.mkdir(parents=True, exist_ok=True)
                    mask_path = mask_dir / f"{frame_id}.png"
                else:

                    image_path = output_dir / cam_id / f"{frame_id}.jpg"
                    mask_path = output_dir / cam_id / "mask" / "pha" / f"{frame_id}.png"

                write_tensor_image(str(image_path), rendered_image, rgb2bgr=True)
                cv2.imwrite(str(mask_path), mask)


                if args.gui_ip != 'none':
                    network_gui.send_image_to_network(rendered_image, f"train_{cam_id}")

            except Exception as e:
                logger.error(f"Error rendering camera {cam_id}: {e}")
                continue

        logger.info(f"Rendering completed! Rendered {len(rendered_cameras)} unique cameras. Output saved to: {output_dir}")


    if getattr(args, "render_360_video", False):
        try:

            video_out_dir = Path(args.video_out_dir) if args.video_out_dir is not None else (dat_dir / "video_360")
            video_out_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Rendering 360° orbit video with {args.video_num_frames} frames to {video_out_dir}")
            if args.orbit_source == "gaussians":
                orbit_cameras = build_orbit_cameras_from_gaussians(
                    gs_model,
                    calib_file,
                    num_frames=args.video_num_frames,
                    data_device=("cuda" if device.type == "cuda" else "cpu"),
                    radius_scale=args.orbit_radius_scale,
                    orbit_direction=args.orbit_direction,
                    start_deg=args.orbit_start_deg,
                    up_axis_str=args.orbit_up_axis,
                )
            else:

                orbit_cameras = build_orbit_cameras(
                    calib_file,
                    num_frames=args.video_num_frames,
                    data_device=("cuda" if device.type == "cuda" else "cpu"),
                    pitch_tag="p+00",
                    up_axis=args.orbit_up_axis,
                    orbit_direction=args.orbit_direction,
                    start_deg=args.orbit_start_deg,
                    radius_scale=args.orbit_radius_scale,
                )

            bg_color_list = [0.0, 0.0, 0.0] if args.bg_color == 'black' else [1.0, 1.0, 1.0]

            for idx, cam in enumerate(tqdm(orbit_cameras, desc="Rendering 360 orbit")):
                scene_camera = cam
                render_pkg = gs_model.render_to_camera(
                    scene_camera,
                    pipe,
                    background=torch.tensor(bg_color_list, dtype=torch.float32, device=device),
                )
                rendered_image = render_pkg['render']


                frame_name = f"orbit_{idx:04d}.png"
                frame_path = video_out_dir / frame_name
                write_tensor_image(str(frame_path), rendered_image, rgb2bgr=True)


            video_path = video_out_dir / "video_360.mp4"
            video_path = Path(args.video_mp4) if args.video_mp4 is not None else video_path
            try:
                import subprocess
                ffmpeg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate", str(args.video_fps),
                    "-i", str(video_out_dir / "orbit_%04d.png"),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(video_path),
                ]
                logger.info(f"Running ffmpeg to create video: {' '.join(ffmpeg_cmd)}")
                subprocess.run(ffmpeg_cmd, check=True)
                logger.info(f"Wrote 360° orbit video to {video_path} using ffmpeg")
            except Exception as e_ffmpeg:
                logger.warning(f"ffmpeg failed ({e_ffmpeg}), falling back to OpenCV VideoWriter.")
                try:
                    import cv2 as _cv2

                    frame_files = sorted(video_out_dir.glob("orbit_*.png"))
                    if not frame_files:
                        raise RuntimeError("No orbit_*.png frames found for video creation.")
                    first_frame = _cv2.imread(str(frame_files[0]))
                    h, w = first_frame.shape[:2]
                    fourcc = _cv2.VideoWriter_fourcc(*'mp4v')
                    video_path.parent.mkdir(parents=True, exist_ok=True)
                    video_writer = _cv2.VideoWriter(str(video_path), fourcc, args.video_fps, (w, h))
                    for fpath in frame_files:
                        frame = _cv2.imread(str(fpath))
                        if frame is None:
                            continue
                        video_writer.write(frame)
                    video_writer.release()
                    logger.info(f"Wrote 360° orbit video to {video_path} using OpenCV")
                except Exception as e_cv2:
                    logger.error(f"Both ffmpeg and OpenCV video writing failed: {e_cv2}")

        except Exception as e:
            logger.error(f"Error while rendering 360° orbit video: {e}")


def main():
    parser = ArgumentParser(description="Render head-swapped Gaussians in GT dataset format")


    parser.add_argument('--input_gs_ply', type=str, required=True, help='Path to head-swapped Gaussian PLY file')
    parser.add_argument('--dat_dir', type=str, required=True, help='Path to dataset directory')
    parser.add_argument('--configs', type=lambda s: [i for i in s.split(';')], required=True, help='Config files (semicolon-separated)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for rendered images')


    parser.add_argument('--input_gs_embed', type=str, default=None, help='Path to Gaussian embedding file')
    parser.add_argument('--frame_id', type=int, default=0, help='Frame ID to render (default: 0)')
    parser.add_argument('--cpu', action='store_true', help='Use CPU instead of GPU')
    parser.add_argument('--bg_color', type=str, default='black', choices=['black', 'white'], help='Background color')
    parser.add_argument('--camera_transform_npz', type=str, help='Path to NPZ file containing head transformation matrix for camera-based GT alignment')
    parser.add_argument('--output_subdir', type=str, default=None, help='Custom subdirectory name for camera-transformed renders (default: head_aligned or head_aligned_rigidhead)')
    parser.add_argument('--output_subject_name', type=str, default=None, help='Explicit subject directory name (overrides PLY-based extraction, useful for generic PLY names like point_cloud.ply)')
    parser.add_argument('--update_all', action='store_true', help='Update all cameras')
    parser.add_argument('--custom_rgb_relpath', type=str, default=None, help='Custom relative RGB path under each camera root. Supports {frame} and {cam}.')
    parser.add_argument('--custom_mask_relpath', type=str, default=None, help='Custom relative mask path under each camera root. Supports {frame} and {cam}.')

    parser.add_argument('--render_canonical_front', action='store_true', help='Render only a single canonical camera (no dataset cameras required).')

    parser.add_argument('--render_canonical', action='store_true', dest='render_canonical_front', help='Alias for --render_canonical_front.')
    parser.add_argument('--canonical_cam_id', type=str, default='front', help='Camera directory name for canonical view (default: front).')
    parser.add_argument('--canonical_fov_deg', type=float, default=10.0, help='Horizontal FOV in degrees for canonical camera (default: 10).')
    parser.add_argument('--canonical_distance', type=float, default=5.0, help='Camera distance along +Z for canonical camera.')
    parser.add_argument('--canonical_target_y', type=float, default=1.0, help='Look-at target Y (world units) for canonical camera.')
    parser.add_argument('--canonical_z_sign', type=int, default=1, choices=[-1, 1], help='Camera placed at z = canonical_z_sign * distance (default: +1).')
    parser.add_argument('--canonical_y_sign', type=int, default=1, choices=[-1, 1], help='Flip Y for canonical camera (default: +1). Use -1 for datasets with inverted world-Y.')
    parser.add_argument('--canonical_yaw_deg', type=float, default=0.0, help='Yaw rotation around +Y in degrees (default: 0).')
    parser.add_argument('--canonical_pitch_deg', type=float, default=0.0, help='Pitch rotation in degrees around camera right axis after yaw (default: 0).')
    parser.add_argument('--canonical_flip_yz', action='store_true', help='Flip Y and Z for canonical camera (ActorsHQ vs TalkBody4D sign convention).')
    parser.add_argument('--canonical_width', type=int, default=None, help='Output image width (default: read from calibration_full.json).')
    parser.add_argument('--canonical_height', type=int, default=None, help='Output image height (default: read from calibration_full.json).')

    parser.add_argument('--render_360_video', action='store_true', help='Render a 360-degree yaw orbit video around the subject.')
    parser.add_argument('--video_num_frames', type=int, default=240, help='Number of frames in the 360-degree orbit video.')
    parser.add_argument('--video_fps', type=int, default=30, help='FPS for the output video.')
    parser.add_argument('--video_out_dir', type=str, default=None, help='Directory to save orbit frames (defaults to <output_dir>/<subject>/video_360).')
    parser.add_argument('--video_mp4', type=str, default=None, help='Optional path to MP4 file to write the 360-degree orbit video.')
    parser.add_argument('--calib_file', type=str, default=None, help='Optional explicit path to calibration_full.json (if different from dat_dir).')
    parser.add_argument('--orbit_up_axis', type=str, default='-y', choices=['y','-y','z','-z','x','-x'], help='Up axis used to define orbit plane (default: -y).')
    parser.add_argument('--orbit_radius_scale', type=float, default=1.2, help='Scale factor on Gaussian bbox-based orbit radius (default: 1.2).')
    parser.add_argument('--orbit_direction', type=str, default='cw', choices=['cw', 'ccw'], help='Orbit direction around up-axis: cw or ccw (default: cw).')
    parser.add_argument('--orbit_source', type=str, default='calib', choices=['calib', 'gaussians'],
                        help='Source for orbit path: calibration cameras (calib) or Gaussian point cloud (gaussians).')
    parser.add_argument('--orbit_start_deg', type=float, default=0.0,
                        help='Starting angle offset (in degrees) for the orbit path.')


    parser.add_argument('--gui_ip', type=str, default='none', help='GUI IP address (none to disable)')
    parser.add_argument('--gui_port', type=int, default=6009, help='GUI port')

    args, extras = parser.parse_known_args()


    config = libcore.load_from_config(args.configs, cli_args=extras)


    render_static_gaussians(args, config)


if __name__ == '__main__':
    main()
