import cv2
import torch
import pytorch_msssim.ssim as ssim2

import torch.nn.functional as F
from math import exp

img_path1 = "/data_ssd2/hzh/paperworks/dataset/screenshots_jpg/00036.jpg"
# img_path2 = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210408/epoch20.jpg"

img_path2 = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210411/epoch1.jpg"
img_path3 = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210411/epoch2.jpg"

img_np1 = cv2.imread(img_path1)
img_np2 = cv2.imread(img_path2)
img_np3 = cv2.imread(img_path3)

img_tensor1 = torch.from_numpy(img_np1).permute(2, 0, 1).unsqueeze(0).float()
img_tensor2 = torch.from_numpy(img_np2).permute(2, 0, 1).unsqueeze(0).float()
img_tensor3 = torch.from_numpy(img_np3).permute(2, 0, 1).unsqueeze(0).float()

# pip的ssim
ssim_loss = 1 - ssim2(img_tensor1, img_tensor2, data_range=255, size_average=True)
print(ssim_loss)

ssim_loss = 1 - ssim2(img_tensor1, img_tensor3, data_range=255, size_average=True)
print(ssim_loss)

