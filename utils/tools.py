import numpy as np


# 将(c,h,w)的bgr通道tensor转为yuv格式
def bgr2yuv_tensor(tensor):
    res = tensor.clone()
    res[0, :, :] = 0.299 * tensor[2, :, :] + 0.587 * tensor[1, :, :] + 0.114 * tensor[0, :, :]
    res[0, :, :] = -0.169 * tensor[2, :, :] - 0.3313 * tensor[1, :, :] + 0.5 * tensor[0, :, :] + 128
    res[0, :, :] = 0.5 * tensor[2, :, :] - 0.4187 * tensor[1, :, :] - 0.0813 * tensor[0, :, :] + 128
    res = res.clamp(0, 255)
    return res


# str转bool值
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


# 将一个batch的数据可视化
def tensor_to_np(tensor):

    rgb_means = (0, 0, 0)

    def tensor_to_np_single(tensor):
        img = tensor.cpu().numpy().transpose((1, 2, 0)).astype(np.uint8)
        img += np.array(rgb_means[::-1]).astype(np.uint8)
        return img

    img_list = []
    for i in range(tensor.shape[0]):
        img_list.append(tensor_to_np_single(tensor[i]))
    return img_list


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