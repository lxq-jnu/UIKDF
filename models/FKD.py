import torch
from torch import nn
import torch.nn.functional as F
from models.common import gradient, RGB2YCrCb6, gradient3
import cv2
import numpy as np

class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, w=4, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)

    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class Change(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Change, self).__init__()
        
        self.Rep2 = nn.ReflectionPad2d(1)
        self.conv2d2 = nn.Conv2d(dim_in, dim_in // 2, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu2 = nn.ReLU()
       
        self.conv2d3 = nn.Conv2d(dim_in // 2, dim_out, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu3 = nn.ReLU()

    def forward(self, x):
       

        x = self.Rep2(x)
        x = self.conv2d2(x)
        x = self.relu2(x)
       
        x = self.conv2d3(x)
        x = self.relu3(x)

        return x

class GWLoss(nn.Module):
    """Gradient Weighted Loss"""
    def __init__(self, w=4, reduction='mean'):
        super(GWLoss, self).__init__()
        self.w = w
        self.reduction = reduction
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)
        self.weight_x = nn.Parameter(data=sobel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=sobel_y, requires_grad=False)

    def forward(self, x1, x2):
        b, c, w, h = x1.shape
        weight_x = self.weight_x.expand(c, 1, 3, 3).type_as(x1)
        weight_y = self.weight_y.expand(c, 1, 3, 3).type_as(x1)
        Ix1 = F.conv2d(x1, weight_x, stride=1, padding=1, groups=c)
        Ix2 = F.conv2d(x2, weight_x, stride=1, padding=1, groups=c)
        Iy1 = F.conv2d(x1, weight_y, stride=1, padding=1, groups=c)
        Iy2 = F.conv2d(x2, weight_y, stride=1, padding=1, groups=c)
        dx = torch.abs(Ix1 - Ix2)
        dy = torch.abs(Iy1 - Iy2)
        # loss = torch.exp(2*(dx + dy)) * torch.abs(x1 - x2)
        loss = (1 + self.w * dx) * (1 + self.w * dy) * torch.abs(x1 - x2)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return torch.sum(loss)


class IllumKD(nn.Module):
    def __init__(self):
        super(IllumKD, self).__init__()

        self.levels = ["f2", "f3", "f4", "f5"]
        self.in_channels = [16, 32, 64, 128]

        self.proj_layers = nn.ModuleDict({
            level: Change(c, 3)
            for level, c in zip(self.levels, self.in_channels)
        })

        self.gwloss = GWLoss()

    def forward(
        self,
        fused_y, fused_cb, fused_cr,
        vi_f2, vi_f3, vi_f4, vi_f5,
        vis_en_image, vis_en_y_image,
        visen_cb, visen_cr,
        ir_mask_image
    ):
        # -------- feature projection --------
        vi_feats_list = [vi_f2, vi_f3, vi_f4, vi_f5]

        vi_proj = {
            level: self.proj_layers[level](feat)
            for level, feat in zip(self.levels, vi_feats_list)
        }

        # -------- mask --------
        vis_mask = 1 - ir_mask_image
        vis_target = vis_en_image * vis_mask

        
        vis_grad = gradient3(vis_target)

        # -------- multi-scale loss --------
        loss_grad_list = []
        loss_int_list = []

        for k in ["f2", "f3", "f4", "f5"]:
            vi_masked = vi_proj[k] * vis_mask

            # gradient loss
            loss_grad_list.append(
                F.l1_loss(gradient3(vi_masked), vis_grad)
            )

            # intensity loss
            loss_int_list.append(
                F.l1_loss(vi_masked, vis_target)
            )

        # -------- Y --------
        losskd1 = F.l1_loss(fused_y, vis_en_y_image)
        lossg = F.l1_loss(
            gradient(fused_y),
            gradient(vis_en_y_image)
        )

        # -------- color --------
        losscolor_in = (
            F.l1_loss(fused_cb, visen_cb) +
            F.l1_loss(fused_cr, visen_cr)
        )

        losscolor_grad = (
            self.gwloss(fused_cb, visen_cb) +
            self.gwloss(fused_cr, visen_cr)
        )

        
        return (
            loss_int_list[0], loss_int_list[1], loss_int_list[2], loss_int_list[3],
            loss_grad_list[0], loss_grad_list[1], loss_grad_list[2], loss_grad_list[3],
            losskd1, lossg,
            losscolor_in, losscolor_grad
        )
