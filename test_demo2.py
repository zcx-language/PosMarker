import os
import cv2
import torch
import numpy as np

from data import MARKroot_adv
from models.DotsGenerator_new import DotsGenerator50

def getAnnotBoxLoc_XML(AnotPath):
    try:
        import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
    except ImportError:
        import xml.etree.ElementTree as ET
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet = root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ret = [] #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        BndBox = Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        BndBoxLoc = [x1, y1, x2, y2, 1]
        ret.append(BndBoxLoc)
    return ret


dg = DotsGenerator50()
dg.load_state_dict(torch.load('weights/SSD_vgg_bn_MARK_512/20210419_fix_gen/DotGenerator_MARK_epoches_5.pth'))
dg = dg.to('cuda')
dg.eval()

test_anno_path = os.path.join(MARKroot_adv, 'VOC2007/Annotations')
test_img_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages_bak')
test_mark_path = os.path.join(MARKroot_adv, 'VOC2007/JPEGImages')
test_txt_path = os.path.join(MARKroot_adv, 'VOC2007/ImageSets/Main/test.txt')
with open(test_txt_path, 'r') as test_file:
    for line in test_file.readlines():
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
                print('rewrite ' + os.path.join(test_mark_path, line + '.jpg'))
# dg.train()