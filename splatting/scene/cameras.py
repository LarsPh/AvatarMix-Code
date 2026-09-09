#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
import copy
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrix2

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, image, gt_alpha_mask,
                 image_name, uid,
                 FoVx=None, FoVy=None,
                 w=None, h=None, fx=None, fy=None, cx=None, cy=None,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda"):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R_init = R # Store original R, T for state if needed
        self.T_init = T
        self.image_name = image_name

        try:
            self.data_device_init_arg = data_device # Store the initial target device string
            _device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device_init_arg = "cuda"
            _device = torch.device("cuda")

        # Register tensors as buffers
        # original_image is already a tensor when passed in
        self.register_buffer('original_image', image.clamp(0.0, 1.0).to(_device))
        
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # gt_alpha_mask is already a tensor when passed in
            self.register_buffer('gt_alpha_mask', gt_alpha_mask.to(_device))
        else:
            # Create a dummy buffer if it's None, or handle its absence
            # For simplicity, let's ensure it's always a buffer if used later
             self.register_buffer('gt_alpha_mask', torch.empty(0, device=_device))


        self.zfar = 5.0 # Typically float, not tensor unless varied
        self.znear = 0.5 # Typically float, not tensor unless varied

        self.trans_init = trans # Store original trans, scale
        self.scale_init = scale

        # These become buffers and will be moved by .to()
        self.register_buffer('world_view_transform', torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).to(_device))

        if FoVx is not None:
            self.FoVy = FoVy # Store original FoV
            self.FoVx = FoVx
            self.register_buffer('projection_matrix', getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=FoVx, fovY=FoVy).transpose(0,1).to(_device))
        else:
            self.FoVy = 2 * np.arctan(h / (2.0 * fy)) # Store calculated FoV
            self.FoVx = 2 * np.arctan(w / (2.0 * fx))
            # Ensure w, h, fx, fy, cx, cy are floats for getProjectionMatrix2 if they are not tensors
            _w, _h, _fx, _fy, _cx, _cy = float(w), float(h), float(fx), float(fy), float(cx), float(cy)
            self.register_buffer('projection_matrix', getProjectionMatrix2(_w, _h, _fx, _fy, _cx, _cy, self.znear, self.zfar).transpose(0,1).to(_device))
        
        # Ensure inputs to bmm are on the same device, which they will be if _device is consistent
        self.register_buffer('full_proj_transform', (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0))
        self.register_buffer('camera_center', self.world_view_transform.inverse()[3, :3])

    # The custom .cuda() method might need adjustment or can be removed if .to(device) is used consistently.
    # For now, let's keep it, as nn.Module.cuda() will correctly handle buffers.
    def cuda(self): # This method is often called as cam.cuda()
        # The default nn.Module.cuda() will move registered buffers and parameters.
        # If you deepcopy, ensure the copy also gets its buffers on cuda.
        # A simpler cam.cuda() would be just `return self.to(torch.device('cuda'))`
        # For compatibility with existing code that might expect a new object from your old .cuda():
        # However, deepcopying nn.Module with buffers and then moving can be stateful.
        # The most straightforward is to use the module's own .cuda() method which handles buffers.
        return super().cuda()

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

4