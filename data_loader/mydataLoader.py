import os
import random

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms

from models.common import RGB2YCrCb

to_tensor = transforms.Compose([transforms.ToTensor()])

patch_size = 256

class mydataLoader(data.Dataset):#加载数据集
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得LLVIP数据集的子目录

        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'vi':
                self.vis_path = temp_path  # 获得可见光路径
            elif sub_dir == 'evi':
                self.en_vis_path = temp_path  # 获得增强后的可见光路径
            else:
                self.mask_path = temp_path  # 获得掩码


        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        inf_image = Image.open(os.path.join(self.inf_path, name.replace(".jpg", ".png"))).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vis_path, name.replace(".jpg", ".png")))#获取可见光图像
        vis_en_image = Image.open(os.path.join(self.en_vis_path, name.replace(".jpg", ".png")))#获取增强后的可见光图像
        mask_image = Image.open(os.path.join(self.mask_path, name.replace(".jpg", ".png"))).convert('L') #获取增强后的可见光图像


        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_en_image = self.transform(vis_en_image)
        mask_image = self.transform(mask_image)

        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)#将可见光图像转换为 YCbCr 色彩空间，返回 Y、Cb、Cr 三个通道
        vis_en_y_image, vis_en_cb_image, vis_en_cr_image = RGB2YCrCb(vis_en_image)#将增强可见光图像转换为 YCbCr 色彩空间，返回 Y、Cb、Cr 三个通道

        #随机裁剪
        _,h, w = inf_image.shape

        y = random.randint(0, h - patch_size - 1)
        x = random.randint(0, w - patch_size - 1)

        inf_image = inf_image[:, y: y + patch_size, x: x + patch_size]
        vis_image = vis_image[:, y: y + patch_size, x: x + patch_size]
        vis_en_image = vis_en_image[:, y: y + patch_size, x: x + patch_size]
        mask_image = mask_image[:, y: y + patch_size, x: x + patch_size]

        
        vis_y_image = vis_y_image[:, y: y + patch_size, x: x + patch_size]
        vis_cb_image = vis_cb_image[:, y: y + patch_size, x: x + patch_size]
        vis_cr_image = vis_cr_image[:, y: y + patch_size, x: x + patch_size]

        vis_en_y_image = vis_en_y_image[:, y: y + patch_size, x: x + patch_size]
        vis_en_cb_image = vis_en_cb_image[:, y: y + patch_size, x: x + patch_size]
        vis_en_cr_image = vis_en_cr_image[:, y: y + patch_size, x: x + patch_size]

        
        inf3_image = torch.cat([inf_image, inf_image, inf_image], dim=0)  # 单通道变三通道

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, vis_en_image, vis_en_y_image, vis_en_cb_image, vis_en_cr_image,inf_image,inf3_image, mask_image,name

    def __len__(self):
        return len(self.name_list)
 

