import torch.nn.functional as F
def padding(image, divide_size=4):
    #1   3   1024  1280
    n, c, h, w = image.shape
    padding_h = divide_size - h % divide_size  # 填充 以确保都是divide_size的倍数
    padding_w = divide_size - w % divide_size  #
    image = F.pad(image, (0, padding_w, 0, padding_h), "reflect") # 使用F.pad函数在图像的右侧和底部进行反射填充
    return image, h, w

def unpadding(image, h, w):
    return image[:, :, :int(h), :int(w)]