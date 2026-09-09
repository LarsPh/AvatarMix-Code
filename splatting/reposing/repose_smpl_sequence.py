from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from loguru import logger

from gaussian_renderer import network_gui
from scene.dataset_readers import make_scene_camera
from dataset.dataset_helper import make_frameset_data
from model.splatting_avatar_model import SplattingAvatarModel
from model import libcore

from reposing.dataset import AnimateDataset
from reposing.utils.video_utils import make_video


def repose_gs_on_smpl_mesh(args, config):


    config.dataset.dat_dir = args.dat_dir
    frameset_train = make_frameset_data(config.dataset, split='train')

    smpl_model = frameset_train.smpl_model
    cam = frameset_train.cam
    empty_img = np.zeros((cam.h, cam.w, 3), dtype=np.uint8)

    viewpoint_cam = make_scene_camera(0, cam, empty_img, config.dataset)
    mesh_py3d = frameset_train.mesh_py3d


    betas = frameset_train.smpl_params['betas']
    anim_data = AnimateDataset(args.anim_fn, betas)


    subject = Path(args.dat_dir).stem
    out_dir = os.path.join(Path(args.anim_fn).parent, f'anim_{subject}')
    os.makedirs(out_dir, exist_ok=True)

    out_dir_reposed_gs_ply = None
    if args.save_reposed_gs_ply:
        out_dir_reposed_gs_ply = Path(out_dir) / "reposed_gaussians_ply"
        out_dir_reposed_gs_ply.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving reposed GS PLY files to: {out_dir_reposed_gs_ply}")


    pipe = config.pipe

    gs_model: SplattingAvatarModel = SplattingAvatarModel(config.model, verbose=True)

    ply_fn = args.input_gs_ply if args.input_gs_ply else os.path.join(args.pc_dir, 'point_cloud.ply')
    gs_model.load_ply(ply_fn)
    embed_fn = args.input_gs_embed if args.input_gs_embed else os.path.join(args.pc_dir, 'embedding.json')
    if embed_fn and os.path.exists(embed_fn):
        gs_model.load_from_embedding(embed_fn)
    else:
        logger.warning(f"Warning: Embedding file {embed_fn} not found or not specified.")


    if args.gui_ip != 'none':
        network_gui.init(args.gui_ip, args.gui_port)
        verify = args.dat_dir
    else:
        verify = None


    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    for idx in tqdm(range(len(anim_data))):
        pose_params = anim_data.__getitem__(idx)

        pose_params = {k: v.to(device) for k,v in pose_params.items()}

        out = smpl_model(**pose_params)
        frame_mesh = mesh_py3d.update_padded(out['vertices'])
        mesh_info = {
            'mesh_verts': frame_mesh.verts_packed(),
            'mesh_norms': frame_mesh.verts_normals_packed(),
            'mesh_faces': frame_mesh.faces_packed(),
        }

        gs_model.update_to_posed_mesh(mesh_info)

        if args.save_reposed_gs_ply and out_dir_reposed_gs_ply is not None:
            from reposing.utils.rigid_head_reposing import get_rigid_head_suffix
            rigid_suffix = get_rigid_head_suffix(args)
            reposed_gs_ply_filename = f"reposed_gs_animframe{idx:04d}{rigid_suffix}.ply"
            reposed_gs_ply_path = out_dir_reposed_gs_ply / reposed_gs_ply_filename
            if args.force_update_all or args.update_reposed_gs_ply or not reposed_gs_ply_path.exists():
                gs_model.save_ply(str(reposed_gs_ply_path))


        render_pkg = gs_model.render_to_camera(viewpoint_cam, pipe, background=torch.tensor(args.render_bg_color, dtype=torch.float32, device=device))
        image = render_pkg['render']

        if verify is not None:
            network_gui.send_image_to_network(image, verify)

        rendered_image_path = os.path.join(out_dir, f'{idx:04d}.jpg')
        if args.force_update_all or args.update_renders or not os.path.exists(rendered_image_path):
            libcore.write_tensor_image(rendered_image_path, image, rgb2bgr=True)


    video_path = os.path.join(out_dir, f'{subject}.mp4')
    if args.force_update_all or args.update_video or not os.path.exists(video_path):
        make_video(out_dir, subject)
    else:
        logger.info(f"Video {video_path} already exists. Skipping generation.")
    return
