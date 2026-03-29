# -*- coding: utf-8 -*-
import argparse
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from data_loader.mydataLoader import mydataLoader
from models.common import gradient, clamp, RGB2YCrCb6, gradient3
from models.FKD import  IllumKD
from models.Smodel import Student
from utils.util import padding,unpadding
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 随机向量，使得每次运行算法出来的结果不一致，基于随机种子来实现代码中的随机方法，能够 保证多次运行此段代码能够得到完全一样的结果，即保证结果的 可复现性
def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def to_device(device, *tensors):
    return [t.to(device) for t in tensors]

def compute_loss(outputs, inputs, criterion_kd, args):
    fused_y = outputs["fused_y"]
    fused_cb = outputs["fused_cb"]
    fused_cr = outputs["fused_cr"]

    vi_f2 = outputs["vi_f2"]
    vi_f3 = outputs["vi_f3"]
    vi_f4 = outputs["vi_f4"]
    vi_f5 = outputs["vi_f5"]

    vis_y = inputs["vis_y"]
    inf = inputs["inf"]
    mask = inputs["mask"]

    vis_en = inputs["vis_en"]
    vis_en_y = inputs["vis_en_y"]
    visen_cb = inputs["visen_cb"]
    visen_cr = inputs["visen_cr"]

    # -------- KD --------
    ll21,ll31,ll41,ll51,loss21,loss31,loss41,loss51,losstl1,losstg,losscolor_in,losscolor_grad = \
        criterion_kd(
            fused_y, fused_cb, fused_cr,
            vi_f2, vi_f3, vi_f4, vi_f5,
            vis_en, vis_en_y, visen_cb, visen_cr,
            mask
        )

    # -------- base loss --------
    grad_loss = 2 * F.l1_loss(gradient(fused_y * mask), gradient(inf * mask))
    illum_loss = 2 * F.l1_loss(fused_y * mask, inf * mask)

    s_loss_in = F.l1_loss(fused_y, inf) + F.l1_loss(fused_y, vis_y)
    s_grad_loss = 1.1 * F.l1_loss(
        gradient(fused_y),
        torch.max(gradient(inf), gradient(vis_y))
    )

    # -------- total --------
    loss = (
        args.w_illum * illum_loss +
        args.w_grad * grad_loss +
        args.w_tl1 * losstl1 +
        args.w_tg * losstg +
        args.w_s_in * s_loss_in +
        args.w_s_grad * s_grad_loss +
        150 * loss21 + 140 * loss31 + 130 * loss41 + 120 * loss51 +
        10 * (ll21 + ll31 + ll41 + ll51) +
        120 * losscolor_grad
    )

    loss_dict = {
        "total": loss.item(),
        "illum": illum_loss.item(),
        "grad": grad_loss.item(),
        "s_in": s_loss_in.item(),
        "s_grad": s_grad_loss.item()
    }

    return loss, loss_dict

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser = argparse.ArgumentParser(description='PyTorch myFusion')  # 创建一个名为parser的参数解析器对象，该对象带有程序的描述
    parser.add_argument('--dataset_path', metavar='DIR', default="datasets/train",
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='Smodel',
                        choices=['Smodel'])
    parser.add_argument('--save_path', default='pretrained')  # 模型存储路径

    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--epochs', default=298, type=int, metavar='N',
                        help='number of total epochs to run')  # ....................................epoch..
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=9, type=int,# .....batchsize.....................................
                        metavar='N',
                        help='mini-batch size (default: 1), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')  
    
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')  # 学习率
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
      
    parser.add_argument('--w_illum', type=float, default=20)
    parser.add_argument('--w_grad', type=float, default=200)
    parser.add_argument('--w_tl1', type=float, default=20)
    parser.add_argument('--w_tg', type=float, default=200)
    parser.add_argument('--w_s_in', type=float, default=8)
    parser.add_argument('--w_s_grad', type=float, default=50)

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=1, help='temperature for KD distillation')

    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth',
                        help='use cls pre-trained model')  # 上一步训练好的模型
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training.')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU.')
    args = parser.parse_args()

    init_seeds(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

# 加载训练集
    train_dataset = mydataLoader(args.dataset_path)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
# 如果是融合网络
    if args.arch == 'Smodel':

        model_s = Student().cuda()  
        criterion_kd = IllumKD().cuda() 


        model_list = nn.ModuleList([
                        model_s,
                        criterion_kd
                    ])
        trainable_list = nn.ModuleList([
                        model_s,
                        criterion_kd
                    ])

        model_list.cuda()


        optimizer = optim.Adam(trainable_list.parameters(), lr=args.lr)
        
        for epoch in range(args.start_epoch, args.epochs):
            # 动态调整学习率
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)
            # 修改学习率 通过迭代优化器中的参数组，并将学习率设置为先前计算得到的 lr。这是实现动态学习率调整的标准方式
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for tmodule in model_list:
                tmodule.train()


            model_list_s = model_list[0]
        


            train_tqdm = tqdm(train_loader, total=len(train_loader))  # 显示训练进度 对train_loader包装
            for vis_image, vis_y_image, vis_cb, vis_cr, vis_en_image, vis_en_y_image, visen_cb, visen_cr, inf_image,inf3_image,mask_image, _ in train_tqdm:  #


                vis_image, vis_y_image, vis_cb, vis_cr, \
                vis_en_image, vis_en_y_image, visen_cb, visen_cr, \
                inf_image, inf3_image, mask_image = to_device(device,  
                    vis_image, vis_y_image, vis_cb, vis_cr,
                    vis_en_image, vis_en_y_image, visen_cb, visen_cr,
                    inf_image, inf3_image, mask_image
                )

                vi_f2, ir_f2, vi_f3, ir_f3, vi_f4, ir_f4, vi_f5, ir_f5, fused_s_image = model_list_s(vis_image,inf3_image)
                fused_s_image = clamp(fused_s_image)
                fused_y, fused_cb, fused_cr = RGB2YCrCb6(fused_s_image)

                outputs = {
                    "fused": fused_s_image,
                    "fused_y": fused_y,
                    "fused_cb": fused_cb,
                    "fused_cr": fused_cr,
                    "vi_f2": vi_f2,
                    "vi_f3": vi_f3,
                    "vi_f4": vi_f4,
                    "vi_f5": vi_f5
                }
                inputs = {
                    "vis_y": vis_y_image,
                    "inf": inf_image,
                    "mask": mask_image,
                    "vis_en": vis_en_image,
                    "vis_en_y": vis_en_y_image,
                    "visen_cb": visen_cb,
                    "visen_cr": visen_cr
                }
               
                
                loss, loss_dict = compute_loss(outputs, inputs, criterion_kd, args)
                train_tqdm.set_postfix(**loss_dict)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (epoch == args.epochs):
                torch.save(model_list_s.state_dict(),f'{args.save_path}/UIKDF-{epoch}.pth')