import argparse
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader.test_dataloader import TestdataLoader
import torch.nn.functional as F   

from models.common import YCrCb2RGB, RGB2YCrCb, clamp
import time
from models.Smodel import Student
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')

def to_device(device, *tensors):
    return [t.to(device, non_blocking=True) for t in tensors]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch myFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/test',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='test_results/LLVIP')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--fusion_pretrained', default='pretrained/uikdf.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,help='GPU to use.')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_dataset = TestdataLoader(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

        model = Student().to(device)
        model.load_state_dict(torch.load(args.fusion_pretrained, map_location=device))
        model.eval()
        
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        start = time.time()
        with torch.no_grad():
          
            for vis_image, vis_y_image,vis_cb, vis_cr,inf_image,inf3_image, name in test_tqdm:  #
            
                vis_image, vis_y_image, vis_cb, vis_cr, inf_image, inf3_image = \
                to_device(device, vis_image, vis_y_image, vis_cb, vis_cr, inf_image, inf3_image)


                
                vi_f2, ir_f2, vi_f3, ir_f3, vi_f4, ir_f4, vi_f5, ir_f5, fused_image = model(vis_image,inf3_image)

                fused_image = clamp(fused_image)
        
                rgb_tensor = fused_image[0].detach().cpu()
                rgb_tensor_permuted = rgb_tensor.permute(1, 2, 0)
                rgb_image = Image.fromarray((rgb_tensor_permuted * 255).cpu().numpy().astype('uint8'))
                rgb_image.save(f'{args.save_path}/{name[0]}')
            endtim = time.time()
            avg_time = (endtim - start) / len(test_loader)
            print(f"Avg inference time per image: {avg_time:.4f}s")
