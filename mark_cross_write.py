import os
import sys
import math
import argparse
import cv2
import numpy as np
# from screeninfo import get_monitors
from skimage.feature import greycomatrix, greycoprops


def bgr2yuv(B, G, R):
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.169 * R - 0.3313 * G + 0.5 * B + 128
    V = 0.5 * R - 0.4187 * G - 0.0813 * B + 128
    Y = np.clip(Y, 0, 255)
    U = np.clip(U, 0, 255)
    V = np.clip(V, 0, 255)
    return Y, U, V


def yuv2bgr(Y, U, V):
    R = Y - 0.00093 * (U - 128) + 1.401687 * (V - 128)
    G = Y - 0.3437 * (U - 128) - 0.71417 * (V - 128)
    B = Y + 1.77216 * (U - 128) + 0.00099 * (V - 128)
    R = np.clip(R, 0, 255)
    G = np.clip(G, 0, 255)
    B = np.clip(B, 0, 255)
    return B, G, R


def getNewColor(B, G, R, comp, nvf=None):
    '''
        根据底色调整定位点的颜色
    '''
    Y, U, V = bgr2yuv(B, G, R)
    # 根据复杂度计算改变值

    # weak
    # ca = 15 + comp * 30 if nvf == 0 else 20 + comp * 30

    # normal
    # ca = 25 + comp * 30 if nvf == 0 else 30 + comp * 30

    # strong
    ca = 65 + comp * 30 if nvf == 0 else 70 + comp * 30

    # ca = 20 + comp * 15 if nvf == 0 else 30 + comp * 15
    # ca = 20 + comp * 15 if nvf <= 25 else 30 + comp * 15
    # print(ca)
    Y = Y - ca if Y >= ca else Y + ca
    return yuv2bgr(Y, U, V)


def getAnnotBoxLoc_XML(AnotPath):
    '''
        返回图片“标注”信息
    '''
    try:
        import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
    except ImportError:
        import xml.etree.ElementTree as ET
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    ret=[] #以目标类别为关键字，目标框为值组成的字典结构
    for Object in ObjectSet:
        # ObjName=Object.find('name').text
        BndBox=Object.find('bndbox')
        x1 = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
        y1 = int(BndBox.find('ymin').text)#-1
        x2 = int(BndBox.find('xmax').text)#-1
        y2 = int(BndBox.find('ymax').text)#-1
        # x1 += 5
        # y1 += 5
        # x2 -= 5
        # y2 -= 5
        BndBoxLoc=[x1,y1,x2,y2]
        ret.append(BndBoxLoc)
    return ret


def getDotListFix(num_dot_inner=4, num_dot_outter=8, scale=40, anno=''):
    '''
        根据标注信息得到散点的坐标
    '''
    data = getAnnotBoxLoc_XML(anno)
    l = [(35, 25), (25, 35), (15, 25), (25, 15), (45, 25), (39, 39), (25, 45),
         (11, 39), (5, 25), (11, 11), (25, 5), (39, 11), (22, 25), (28, 25),
         (25, 22), (25, 28), (25, 25)]
    ret = []
    for coord in data:
        pt = coord[0], coord[1]
        tmp = [(x + pt[0], y + pt[1]) for x, y in l]
        ret += tmp
    return ret

# def getDotList(num_dot_inner=4, num_dot_outter=8, scale=40, anno=''):
#     '''
#         根据标注信息得到散点的坐标
#     '''
#     data = getAnnotBoxLoc_XML(anno)
#     ret = []
#     for coord in data:
#         pt1 = coord[0], coord[1]
#         pt2 = coord[2], coord[3]
#         # print(pt1, pt2)
#         r = min(coord[2] - coord[0], coord[3] - coord[1]) // 2
#         center = (coord[2] + coord[0]) // 2, (coord[3] + coord[1]) // 2

#         # 内圈4个点
#         dot_list1 = [(center[0] + r / 2 * math.cos(2 * math.pi / num_dot_inner * i),
#                       center[1] + r / 2 * math.sin(2 * math.pi / num_dot_inner * i)) for i in range(num_dot_inner)]

#         # 外圈8个点
#         dot_list2 = [(center[0] + r * math.cos(2 * math.pi / num_dot_outter * i),
#                       center[1] + r * math.sin(2 * math.pi / num_dot_outter * i)) for i in range(num_dot_outter)]

#         # # 中间十字9个点
#         # dirs= [(-2,0), (-4,0), (2,0), (4,0), (0,-2), (0,-4), (0,2), (0,4)]
#         # dot_list3 = [(center[0] + x, center[1] + y) for x,y in dirs] + [center]

#         # 中间十字5个点
#         dirs= [(-3,0), (3,0), (0,-3), (0,3)]
#         dot_list3 = [(center[0] + x, center[1] + y) for x,y in dirs] + [center]

#         ret += dot_list1
#         ret += dot_list2
#         ret += dot_list3
#     return ret


# def drawDot(img, dotlist, comps, dot_size=2, num_dot=12+9, show_comp=False):
def drawDot(img, dotlist, comps, dot_size=2, num_dot=12+5, show_comp=False):
    '''
        绘制散点
    '''
    n = 1
    ret = []
    offset = dot_size // 2
    for dot in dotlist:

        x, y = int(dot[0]), int(dot[1])
        # print(n//num_dot)
        nvf_value = NVF(img, x, y)

        comp_value = comps[(n - 1) // num_dot]

        # ret.append(-np.log(nvf_value))
        ret.append(nvf_value)

        new_color = getNewColor(img[y][x][0], img[y][x][1], img[y][x][2], comp_value, nvf_value)
        # print(new_color)
        cv2.rectangle(img, (x-offset, y-offset), (x+offset, y+offset), new_color, -1)

        # 标出复杂度
        if n % num_dot == 0 and show_comp:
            # 图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
            # cv2.putText(img, 'nvf:'+str(round(abs(min(ret)), 2)), (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
            cv2.putText(img, 'nvf:' + str(round(comp_value, 2)), (x + 15, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            if n != 4 * num_dot:
                ret.clear()
            # pass
        n += 1

    return img, ret


def glcm(arr, d=[2], gray_level=8, normed=False):
    '''
    生成灰度共生矩阵
    '''
    arr = arr.astype(np.float64)
    arr = arr * (gray_level - 1) // 255
    arr = arr.astype(np.int)
    glcm0 = greycomatrix(arr, d, [0, np.pi / 4, np.pi / 2], levels=gray_level, normed=normed)
    return glcm0


def entropy(img, data):
    '''
    根据灰度共生矩阵计算信息熵
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = []
    for coord in data:
        arr = img_gray[coord[1]:coord[3], coord[0]:coord[2]]

        # print(coord[1], coord[3], coord[0], coord[2])

        glcm0 = glcm(arr, gray_level=8, normed=True)
        glcm1 = glcm0.sum(axis=-1).squeeze()/3

        res = 0
        height, width = glcm1.shape
        # 4800为arr区域像素个数1600*3
        for j in range(height):
            for i in range(width):
                if glcm1[j, i] != 0:
                    res -= glcm1[j, i] * math.log(glcm1[j, i])
        comps.append(res)
    # print(comps)
    # exit()
    return comps


def calComplx(img, anno):
    '''
    计算定位点覆盖区域的复杂度
    '''
    data = getAnnotBoxLoc_XML(anno)
    ret = entropy(img, data)
    return ret


def NVF(img, x, y, stride=5):
    '''
    基于像素的噪声可见性函数
    '''
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
    arr = np.zeros([stride, stride])
    for i in range(stride):
        for j in range(stride):
            # print(y - (stride // 2) + i, x - (stride // 2) + j)
            arr[i, j] = img_gray[y - (stride // 2) + i, x - (stride // 2) + j]
    # res = 1/(1+np.var(arr))
    res = np.var(arr)
    # print(res)
    return res


if __name__ == "__main__":

    path = '/data_ssd/hzh/dogs-vs-cats/data/tmp'
    path = '/home/hzh/paperworks/screenshots'
    path = '/data_ssd2/hzh/paperworks/dataset/images_aug/val'
    path = '/data_ssd2/hzh/paperworks/dataset/MarkedDATA_AUG/VOC2007/JPEGImages'
    path = '/data_ssd2/hzh/paperworks/dataset/ScreenshotsDATA_AUG/VOC2007/JPEGImages_bak'
    path = '/data_ssd2/hzh/paperworks/dataset/screenshots_jpg'
    path = '/data_ssd2/hzh/paperworks/dataset/augment/screenshots_aug/train'
    path = '/data_ssd2/hzh/paperworks/dataset/augment/screenshots_aug/val'
    path = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/JPEGImages'
    path = '/data_ssd2/hzh/paperworks/dataset/augment/ScreenshotsDATA/VOC2007/JPEGImages_bak'
    # path = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/Photos/images'
    path = '/data_ssd2/hzh/paperworks/dataset/augment/WEAK/VOC2007/Photos/images'

    out = '/data_ssd/hzh/dogs-vs-cats/data/eg'
    out = '/home/hzh/paperworks/images'
    out = '/data_ssd2/hzh/paperworks/dataset/images_aug/val_marked'
    out = '/data_ssd2/hzh/paperworks/dataset/MarkedDATA_AUG/VOC2007/JPEGImages_marked'
    out = '/data_ssd2/hzh/paperworks/dataset/ScreenshotsDATA_AUG/VOC2007/JPEGImages_marked'
    out = '/data_ssd2/hzh/paperworks/dataset/images_annos'
    out = '/data_ssd2/hzh/paperworks/dataset/augment/images_aug'
    out = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/JPEGImages'
    out = '/data_ssd2/hzh/paperworks/dataset/augment/MarkedDATA/VOC2007/JPEGImages'
    # out = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/Photos/marks'
    out = '/data_ssd2/hzh/paperworks/dataset/augment/WEAK/VOC2007/Photos/marks'

    anno_path = '/home/hzh/paperworks/annotations'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/images_aug/val'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/MarkedDATA_AUG/VOC2007/Annotations'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/ScreenshotsDATA_AUG/VOC2007/Annotations'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/annotations'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/augment/annotations_aug'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/augment/ScreenshotsDATA/VOC2007/Annotations'
    # anno_path = '/data_ssd2/hzh/paperworks/dataset/augment/BASELINE/VOC2007/Photos/annos'
    anno_path = '/data_ssd2/hzh/paperworks/dataset/augment/WEAK/VOC2007/Photos/annos'

    parser=argparse.ArgumentParser()
    parser.add_argument('--scale', type=int, default=40, help="scale of every mark")
    parser.add_argument('--dot_size', type=int, default=2, help="size of every dot")
    parser.add_argument('--num_dot_outter', type=int, default=8, help="number of dots in the outter circle")
    parser.add_argument('--num_dot_inner', type=int, default=4, help="number of dots in the inner circle")
    args=parser.parse_args()

    dir = os.listdir(path)

    k=1
    for file in dir:
        if file.endswith('jpg'):
            img = cv2.imread(os.path.join(path,file))
            # img_resize = cv2.resize(img,(new_W,new_H),interpolation=cv2.INTER_CUBIC)
            img_resize = img

            comps = calComplx(img_resize, os.path.join(anno_path, file.split('.')[0]+'.xml'))

            # 全屏显示
            # out=''
            # cv2.namedWindow(out, cv2.WINDOW_NORMAL)
            # cv2.setWindowProperty(out, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # cv2.resizeWindow(out, 1920, 1080)



            # cv2.imshow(out, drawDotNVF(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner,True))
            dotlist=getDotListFix(args.num_dot_inner,args.num_dot_outter,args.scale,os.path.join(anno_path, file.split('.')[0]+'.xml'))
            # print(file)

            img_,ret=drawDot(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner+5,False)
            # img_=drawDot(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner,True)
            # min_v=min(min_v, min(ret))
            # max_v=max(max_v, max(ret))
            # cv2.imshow(out, img_)

            # cv2.imshow("", img_)
            # if cv2.waitKey(0) == 27:
            #    break

            cv2.imwrite(os.path.join(out, file.split('.')[0]+'.jpg'), img_, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if k % 100 == 0:
                print(k)
            # print(file)
            k += 1


            # cv2.imwrite(os.path.join('/data_ssd2/hzh/paperworks/gen',  'weak.jpg'), img_, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # print(file)
            # exit()


            # exit()
            # cv2.imwrite(os.path.join(out, str(k)+'.jpg'), img_, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

            # print(k, '==================================================================')
            # k+=1

            # if cv2.waitKey(3000)==27:
            #    break


# if __name__ == "__main__":
#     path = '/data_ssd/hzh/dogs-vs-cats/data/tmp3'
#     out = '/data_ssd/hzh/dogs-vs-cats/data/eg'
#     # path = "/data_ssd2/hzh/github/BaiduImageSpider/raw_data/vis"
#     # path = 'tmp2'
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--scale', type=int, default=40, help="scale of every mark")
#     parser.add_argument('--dot_size', type=int, default=2, help="size of every dot")
#     parser.add_argument('--num_dot_outter', type=int, default=8, help="number of dots in the outter circle")
#     parser.add_argument('--num_dot_inner', type=int, default=4, help="number of dots in the inner circle")
#     args = parser.parse_args()
#
#     # 获取屏幕尺寸
#     new_W, new_H = get_monitors()[0].width, get_monitors()[0].height
#     dotlist = getDotList(args.num_dot_inner, args.num_dot_outter, args.scale)
#
#     dir = os.listdir(path)
#
#     # min_v=100
#     # max_v=0
#
#     img = cv2.imread("/data_ssd/hzh/dogs-vs-cats/data/tmp/6000.jpg")
#     img_resize = cv2.resize(img, (new_W, new_H), interpolation=cv2.INTER_CUBIC)
#
#     comps = calComplx(img_resize)
#
#     # 全屏显示
#     out = ''
#     cv2.namedWindow(out, cv2.WINDOW_NORMAL)
#     cv2.setWindowProperty(out, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#     # cv2.imshow(out, drawDotNVF(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner,True))
#     img_, ret = drawDot(img_resize, dotlist, comps, args.dot_size, args.num_dot_outter + args.num_dot_inner, False)
#     # img_=drawDot(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner,True)
#     # min_v=min(min_v, min(ret))
#     # max_v=max(max_v, max(ret))
#     cv2.imshow(out, img_)
#     if cv2.waitKey(0) == 27:
#         exit()
#
#         # out='out_tmp0'
#         # cv2.imwrite(os.path.join(out,file),drawDot(img_resize,dotlist,comps,args.dot_size,args.num_dot_outter+args.num_dot_inner,True))
#         # cv2.imwrite(os.path.join(out, file),drawDotNVF(img_resize, dotlist, args.dot_size, args.num_dot_outter + args.num_dot_inner))
#     # print(min_v,max_v)