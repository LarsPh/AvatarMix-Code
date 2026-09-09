import json
import math
import numpy as np
from pathlib import Path
from scene.dataset_readers import make_scene_camera
from model import libcore


def load_camera_definitions_from_nerfstudio_path(json_path):

    with open(json_path, 'r') as f:
        data = json.load(f)

    render_w = int(data['render_width'])
    render_h = int(data['render_height'])
    fps = data.get('fps', 30)

    if 'camera_path' not in data or not data['camera_path']:
        raise ValueError("Camera path JSON does not contain 'camera_path' or it's empty.")

    camera_definitions = data['camera_path']

    return camera_definitions, fps, render_w, render_h


def create_scene_camera_from_definition(cam_definition, render_w, render_h, cam_idx, config_dataset_for_make_scene_camera, device):

    c2w_matrix_flat = cam_definition['camera_to_world']
    c2w_matrix = np.array(c2w_matrix_flat).reshape((4, 4))[:3, :]

    fov_degrees = cam_definition['fov']

    cam_obj = libcore.Camera()
    cam_obj.w = render_w
    cam_obj.h = render_h


    cam_obj.cx = render_w / 2.0
    cam_obj.cy = render_h / 2.0
    fov_rad = math.radians(fov_degrees)
    cam_obj.fy = render_h / (2 * math.tan(fov_rad / 2))
    cam_obj.fx = cam_obj.fy

    cam_obj.set_c2w(c2w_matrix)

    dummy_img_for_cam = np.zeros((cam_obj.h, cam_obj.w, 3), dtype=np.uint8)
    scene_cam = make_scene_camera(cam_idx, cam_obj, dummy_img_for_cam, config_dataset_for_make_scene_camera)
    return scene_cam.to(device)
