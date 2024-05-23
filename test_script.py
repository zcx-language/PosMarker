import cv2
import torch
import numpy as np
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import pytorch_msssim.ssim as ssim2

import torch.nn.functional as F
from math import exp


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


img1 = torch.rand(1, 3, 256, 256).cuda()
img2 = torch.rand(1, 3, 256, 256).cuda()

print(ssim(img1, img2))

ssim_loss = SSIM(window_size=11)
print(ssim_loss(img1, img2))


# ssim_loss = 1 - ssim(img1, img2, data_range=255, size_average=True)
# print(ssim_loss)

# exit()





























img_path1 = "/data_ssd2/hzh/paperworks/dataset/screenshots_jpg/00036.jpg"
img_path2 = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210408/epoch20.jpg"

img_np1 = cv2.imread(img_path1)
img_np2 = cv2.imread(img_path2)

img_tensor1 = torch.from_numpy(img_np1).permute(2, 0, 1).unsqueeze(0).float()
img_tensor2 = torch.from_numpy(img_np2).permute(2, 0, 1).unsqueeze(0).float()


print(1-ssim(img_tensor1, img_tensor2, False))
print("next")



ssim_loss = 1 - ssim2(img_tensor1, img_tensor2, data_range=255, size_average=True)
print(ssim_loss)
exit()

def mod_tensor(a):
    print(id(a))


a = torch.rand(2, 3)
print(id(a))
mod_tensor(a/1.)
exit()

mat = np.array([[1, 0, 100],
                [0, 1, 50]])
boxes = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
# 转为坐标对的格式
boxes = boxes.reshape(-1, 2)

# 增广
new_axis = np.ones((boxes.shape[0], 1))
boxes = np.concatenate((boxes, new_axis), 1).transpose()

# 矩阵乘法
boxes = np.dot(mat, boxes).transpose().reshape(-1, 4)

import cv2
import math
import torch
from torch.nn import functional as F
from torchvision import transforms

img_path = '/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png'
img_torch = transforms.ToTensor()(cv2.imread(img_path))
# 这两句等价
img_torch2 = torch.from_numpy(cv2.imread(img_path).transpose(2, 0, 1)).float()

angle = -1*math.pi/180
theta = torch.tensor([[math.cos(angle), math.sin(-angle), 0],
                      [math.sin(angle), math.cos(angle), 0]],
                     dtype=torch.float)

grid = F.affine_grid(theta.unsqueeze(0), img_torch.unsqueeze(0).size())
output = F.grid_sample(img_torch.unsqueeze(0), grid, padding_mode='reflection')
new_img_torch = output[0]/255
cv2.imshow('', new_img_torch.numpy().transpose(1, 2, 0))
cv2.waitKey(5000)