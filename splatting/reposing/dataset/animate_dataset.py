import torch
import numpy as np


class AnimateDataset(torch.utils.data.Dataset):
    def __init__(self, pose_sequence, betas):
        smpl_params = dict(np.load(pose_sequence))

        thetas = smpl_params["poses"][..., :72]
        transl = smpl_params["trans"] - smpl_params["trans"][0:1]
        transl += (0, 0.15, 5)

        self.betas = betas
        self.thetas = torch.tensor(thetas).float()
        self.transl = torch.tensor(transl).float()

    def __len__(self):
        return len(self.transl)

    def __getitem__(self, idx):
        datum = {

            "betas": self.betas,
            "global_orient": self.thetas[idx:idx+1, :3],
            "body_pose": self.thetas[idx:idx+1, 3:],
            "transl": self.transl[idx:idx+1],
        }
        return datum
