import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from models.salient import SAIF
class Spatial_Attention_Module(nn.Module):
    def __init__(self, k: int):
        super(Spatial_Attention_Module, self).__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        # In order to keep the size of the front and rear images consistent
        # with calculate, k = 1 + 2p, k denote kernel_size, and p denote padding number
        # so, when p = 1 -> k = 3; p = 2 -> k = 5; p = 3 -> k = 7, it works. when p = 4 -> k = 9, it is too big to use in network
        assert k in [3, 5, 7], "kernel size = 1 + 2 * padding, so kernel size must be 3, 5, 7"
        self.conv = nn.Conv2d(2, 1, kernel_size = (k, k), stride = (1, 1), padding = ((k - 1) // 2, (k - 1) // 2),
                              bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # compress the C channel to 1 and keep the dimensions
        avg_x = self.avg_pooling(x, dim = 1, keepdim = True)
        max_x, _ = self.max_pooling(x, dim = 1, keepdim = True)
        v = self.conv(torch.cat((max_x, avg_x), dim = 1))
        v = self.sigmoid(v)
        return x * v
    
 

class AFF(nn.Module):
    '''
    多特征融合 AFF
    '''
    def __init__(self, channels=128, r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        self.sSE= nn.Sequential(
            nn.Conv2d(in_channels=128, kernel_size=1, out_channels=128, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )
        self.nfe= nn.Sequential(
            nn.Conv2d(in_channels=128, kernel_size=1, out_channels=128, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()
        )


        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
        )

        self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Softmax(dim=1)

    def forward(self, x, residual):
        x = self.sSE(x)
        residual = self.nfe(residual)
        #xa = x + residual

        # xa = self.sSE(xa)
        xa = x + residual

        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)

        xo = x * wei + residual * (1 - wei)
        return xo


class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.AFF = AFF(channels=128)

    def forward(self, vi_out, ir_out):

        features_con = self.AFF(vi_out, ir_out)
        
        return features_con



class S_Encoder(nn.Module):

    def make_conv_block(self, in_c, out_c, kernel_size, pad):
        return nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_c, out_c, kernel_size, 1, 0),
            nn.LeakyReLU()
        )

    def __init__(self):
        super(S_Encoder, self).__init__()

        self.viconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=1, out_channels=16, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.irconv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=1, out_channels=16, stride=1, padding=0),
            nn.LeakyReLU()
        )
        self.channels = [16, 32, 64]

        self.viconv_2 = self.make_conv_block(self.channels[0], self.channels[0], 3, 1)
        self.viconv_3 = self.make_conv_block(self.channels[0], self.channels[1], 3, 1)
        self.viconv_4 = self.make_conv_block(self.channels[1], self.channels[2], 3, 1)

        self.irconv_2 = self.make_conv_block(self.channels[0], self.channels[0], 5, 2)
        self.irconv_3 = self.make_conv_block(self.channels[0], self.channels[1], 5, 2)
        self.irconv_4 = self.make_conv_block(self.channels[1], self.channels[2], 5, 2)

        self.conv_5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3, 1, 0),
            nn.LeakyReLU()
        )


    def forward(self, y_vi_image, ir_image):
        # ----------encoder--------------
        # print(y_vi_image.shape)
        vi_out1 = self.viconv_1(y_vi_image)
        vi_out2 = self.viconv_2(vi_out1)
        vi_out3 = self.viconv_3(vi_out2)
        vi_out4 = self.viconv_4(vi_out3)   
        vi_out5 = self.conv_5(vi_out4)   

        ir_out1 = self.irconv_1(ir_image)
        ir_out2 = self.irconv_2(ir_out1)
        ir_out3 = self.irconv_3(ir_out2)
        ir_out4 = self.irconv_4(ir_out3)
        ir_out5 = self.conv_5(ir_out4)
        
        return vi_out2, ir_out2, vi_out3, ir_out3, vi_out4, ir_out4, vi_out5, ir_out5



class S_Decoder(nn.Module):

    def make_conv_block(self, in_c, out_c, k=3, pad=1):
        return nn.Sequential(
            nn.ReflectionPad2d(pad),
            nn.Conv2d(in_c, out_c, k, 1, 0),
            nn.LeakyReLU()
        )
    def __init__(self):
        super(S_Decoder, self).__init__()

        self.decoder_channels = [128, 128, 64, 32]

        self.conv2 = self.make_conv_block(self.decoder_channels[0], self.decoder_channels[1])
        self.conv3 = self.make_conv_block(self.decoder_channels[1], self.decoder_channels[2])
        self.conv4 = self.make_conv_block(self.decoder_channels[2], self.decoder_channels[3])

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, kernel_size=1, out_channels=3, stride=1, padding=0)
        )

    def forward(self, x):
        #x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x



class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.encoder = S_Encoder()
        self.saif = SAIF()
        self.fusion = Fusion()
        self.decoder = S_Decoder()

    def forward(self, vi_image, ir_image):
        vi_f2, ir_f2, vi_f3, ir_f3, vi_f4, ir_f4, vi_f5, ir_f5 = self.encoder(vi_image, ir_image)

        ir = self.saif(ir_f2, ir_f3, ir_f4, ir_f5)

        features_out = self.fusion(vi_f5, ir)
        fused_image = self.decoder(features_out)

        return vi_f2, ir_f2, vi_f3, ir_f3, vi_f4, ir_f4, vi_f5, ir_f5, fused_image