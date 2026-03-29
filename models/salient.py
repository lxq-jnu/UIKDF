import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# class operator(nn.Module):
#     def __init__(self):
#         super(operator, self).__init__()
#
#     def forward(self, ir1):
#         return
#
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SpatialAttention(nn.Module):# spatial attention
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2,1,kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
       

class operator(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes):
        super(operator, self).__init__()
        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight  # 
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
        # explicitly padding with bias
        y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
        b0_pad = self.b0.view(1, -1, 1, 1)
        y0[:, :, 0:1, :] = b0_pad
        y0[:, :, -1:, :] = b0_pad
        y0[:, :, :, 0:1] = b0_pad
        y0[:, :, :, -1:] = b0_pad
        # conv-3x3
        y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    

class IFSD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IFSD, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.rbr_conv1x1_sbx_branch = operator('conv1x1-sobelx', self.in_channels, self.out_channels)
        self.rbr_conv1x1_sby_branch = operator('conv1x1-sobely', self.in_channels, self.out_channels)
        self.rbr_conv1x1_lpl_branch = operator('conv1x1-laplacian', self.in_channels, self.out_channels)

    def forward(self, ir1):
        ir1sx = self.rbr_conv1x1_sbx_branch(ir1)
        ir1sy = self.rbr_conv1x1_sby_branch(ir1)
        ir1l = self.rbr_conv1x1_lpl_branch(ir1)

        return ir1sx + ir1sy + ir1l


class DFC(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, stride=1):
        super(DFC, self).__init__()
        self.oup = oup

        self.gate_fn = nn.Sigmoid()

        self.short_conv = nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
            nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        res = self.short_conv(x)
        out = x
        return out[:, :self.oup, :, :] * self.gate_fn(res)


class SAIF(nn.Module):
    def __init__(self):
        super(SAIF, self).__init__()
        nb_filter = [16, 32, 64, 128]
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=16, kernel_size=1, out_channels=16, stride=1, padding=0),
            nn.LeakyReLU()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, kernel_size=1, out_channels=32, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, kernel_size=1, out_channels=64, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, kernel_size=1, out_channels=128, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conves1 = nn.Sequential(
            nn.Conv2d(in_channels=16, kernel_size=1, out_channels=64, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conves2 = nn.Sequential(
            nn.Conv2d(in_channels=64, kernel_size=1, out_channels=128, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.conve = nn.Sequential(
            nn.Conv2d(in_channels=240, kernel_size=1, out_channels=128, stride=1, padding=0),
            nn.LeakyReLU()
        )

        self.ifsd = IFSD(128, 128)

        self.dfc0 = DFC(16, 16)
        self.dfc1 = DFC(32, 32)
        self.dfc2 = DFC(64, 64)
        self.dfc3 = DFC(128, 128)
        self.senet = SELayer(240)
        self.cbamsa = SpatialAttention()

    def forward(self, ir0, ir1, ir2, ir3):  # 16 | 32 64 128

        
        ir01 = self.conves1(ir0)
        ir01 = self.conves2(ir01)
        ir02 = self.ifsd(ir01)
        ir00 = self.dfc0(ir0) 
        ir11 = self.dfc1(ir1)
        ir21 = self.dfc2(ir2)
        ir31 = self.dfc3(ir3)
        irall = torch.cat((ir00,ir11, ir21, ir31), dim=1)
        irall = self.senet(irall)
        end = self.conve(irall)
        ir3g = self.cbamsa(ir3)

        return ir3g *end + end + ir02 


 
