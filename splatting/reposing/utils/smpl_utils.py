import numpy as np
import math
import torch
from loguru import logger


def get_canonical_smpl_params():


    cano_smpl_pose = np.zeros(75, dtype = np.float32)
    cano_smpl_pose[3+3*1+2] = math.radians(25)
    cano_smpl_pose[3+3*2+2] = math.radians(-25)
    cano_smpl_pose_pt = torch.from_numpy(cano_smpl_pose)


    cano_smpl_transl = cano_smpl_pose_pt[:3].unsqueeze(0)
    cano_smpl_global_orient = cano_smpl_pose_pt[3:6].unsqueeze(0)
    cano_smpl_body_pose = cano_smpl_pose_pt[6:69].unsqueeze(0)
    return cano_smpl_transl, cano_smpl_global_orient, cano_smpl_body_pose


def get_betas_for_reposing(args, smpl_data_src, smpl_data_tar, context='target'):

    if args.reposing_without_beta_change:

        return smpl_data_src['betas'][0:1]
    else:

        if context == 'source':
            return smpl_data_src['betas'][0:1]
        else:
            return smpl_data_tar['betas'][0:1]


def get_v_shape_for_reposing(args, smpl_data_src, smpl_data_tar, context: str = 'target'):

    if ('v_shape' not in smpl_data_src) and ('v_shape' not in smpl_data_tar):
        return None
    if getattr(args, 'reposing_without_beta_change', False):
        return smpl_data_src.get('v_shape', None)[0:1] if smpl_data_src.get('v_shape', None) is not None else None
    if context == 'source':
        return smpl_data_src.get('v_shape', None)[0:1] if smpl_data_src.get('v_shape', None) is not None else None
    return smpl_data_tar.get('v_shape', None)[0:1] if smpl_data_tar.get('v_shape', None) is not None else None


def get_v_pose_for_reposing(smpl_data_tar, frame_idx: int):

    v_pose = smpl_data_tar.get('v_pose', None)
    if v_pose is None:
        return None
    return v_pose[frame_idx:frame_idx+1]
