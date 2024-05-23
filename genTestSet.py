import os
import cv2

import numpy as np
import torch

from models.DotsGenerator_yuv import DotsGenerator50
from utils.tools import getAnnotBoxLoc_XML

MARKroot_adv = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE'

# 构造编码器
dg = DotsGenerator50()
dg.load_state_dict(torch.load('/data_ssd2/hzh/paperworks/gen/gen_epoch6.pth'))

dg.eval()
test_anno_path = os.path.join(MARKroot_adv, 'VOC2007/Annotations')
test_img_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages')
test_mark_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages')
test_txt_path = os.path.join(MARKroot_adv, 'VOC2007/ImageSets/Main/trainval.txt')

with open(test_txt_path, 'r') as test_file:
    for idx, line in enumerate(test_file.readlines()):
        line = line.strip()
        if line:
            img_full_name = os.path.join(test_img_path, line + '.jpg')

            test_image_ = cv2.imread(img_full_name)
            test_image_ = torch.from_numpy(test_image_).permute(2, 0, 1).float().cuda()

            test_targets_ = getAnnotBoxLoc_XML(os.path.join(test_anno_path, line + '.xml'))
            test_targets_ = torch.from_numpy(np.array(test_targets_)).cuda()

            with torch.no_grad():
                gt_crops = dg.get_crops(test_image_, test_targets_)
                out = dg(gt_crops)
                image_ = dg.rewrite_image(out, test_image_, test_targets_)
                image_ = image_.permute(1, 2, 0).detach().cpu().numpy()
                cv2.imwrite(os.path.join(test_mark_path, line + '.jpg'), image_,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 100])

                if (idx+1) % 100 == 0:
                    print(idx)

print("rewrite test dataset done.")