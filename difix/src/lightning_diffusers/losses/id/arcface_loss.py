import torch
import torch.nn.functional as F
from torch import nn
import os
from .arcface_model import Backbone

# pretrained model url: https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn
class ArcFaceIDLoss(nn.Module):
    def __init__(self, ir_se50_path="pretrained/model_ir_se50.pth"):
        super(ArcFaceIDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # download pretrained model if not exists
        if not os.path.exists(ir_se50_path):
            raise FileNotFoundError(f"Pretrained model {ir_se50_path} not found, download it from https://drive.google.com/file/d/1KW7bjndL3QG3sxBbZxreGHigcCCpsDgn")
        self.facenet.load_state_dict(torch.load(ir_se50_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        for param in self.facenet.parameters():
            param.requires_grad = False

    def extract_feats(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        # if x.shape[2] != 112 or x.shape[3] != 112:
        #     x = F.interpolate(x, (112, 112), mode='bilinear')
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, gt, pred):
        n_samples = pred.shape[0]
        y_feats = self.extract_feats(gt)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(pred)
        y_feats = y_feats.detach()
        loss = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return loss / count