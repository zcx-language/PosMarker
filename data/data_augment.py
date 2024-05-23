"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325

TODO: implement data_augment for training

Ellis Brown, Max deGroot
"""

import math

import cv2
import numpy as np
import random
import torch

from aug_test import _mirror, _noise, _rotate, _persp
from utils.box_utils import matrix_iou

# ----------------------------------------------------------
#                   original augmentation code
# ----------------------------------------------------------

def _crop_o(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            # iou = matrix_iou(boxes, roi[np.newaxis])
            #
            # if not (min_iou <= iou.min() and iou.max() <= max_iou):
            #     continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t

def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels

    for _ in range(50):
        # scale = random.uniform(0.3, 1.)
        # # 尽量保持原图宽高比
        # min_ratio = max(0.8, scale * scale)
        # max_ratio = min(1.25, 1. / scale / scale)
        # ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        # w = int(scale * ratio * width)
        # h = int((scale / ratio) * height)

        w = h = 512

        l = random.randrange(width - w)
        t = random.randrange(height - h)
        roi = np.array((l, t, l + w, t + h))

        image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

        # centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        # bbox中心大于xmin和ymin，且小于xmax和ymax
        # 保留完整的gt
        # mask = np.logical_and(roi[:2] < boxes[:, :2], boxes[:, 2:] < roi[2:]).all(axis=1)

        centers1 = boxes[:, :2] * 0.8 + boxes[:, 2:] * 0.2
        centers2 = boxes[:, :2] * 0.2 + boxes[:, 2:] * 0.8
        mask = np.logical_and(roi[:2] < centers1, centers2 < roi[2:]).all(axis=1)
        boxes_t = boxes[mask].copy()
        labels_t = labels[mask].copy()
        if len(boxes_t) == 0:
            continue

        boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
        boxes_t[:, :2] -= roi[:2]
        boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:] - 1)
        boxes_t[:, 2:] -= roi[:2]

        return image_t, boxes_t, labels_t

    return image, boxes, labels


# 不要使用distort, 可能会使图片过曝, 造成标记消失
def _distort(image, boxes):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.8, 1.2))

    return image, boxes


def preproc_for_test(image, insize, mean, std=(1, 1, 1)):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, insize, interpolation=interp_method)

    image = image.astype(np.float32)
    image -= mean[::-1]
    image /= std
    return image.transpose(2, 0, 1)  # 换成(C, H, W)


class preproc_new(object):

    def __init__(self, resize, rgb_means=(0, 0, 0), rgb_std=(1, 1, 1), p=0.2):
        self.means = rgb_means
        self.std = rgb_std
        self.resize = resize
        self.p = p

    def __call__(self, image, targets):
        '''
        :param image:  numpy bgr格式 shape(1080,1920,3)
        :param targets: numpy 格式 shape (gt数, 5)，未归一化
        :return:
        '''

        # boxes和labels都是targets的深拷贝
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # 如果没有gt的话，不做增广，返回一个全0的gt
        if len(boxes) == 0:
            targets = np.zeros((1, 5))
            image = preproc_for_test(image, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets

        # 保存原图的大小
        height_o, width_o, _ = image.shape

        if np.random.randint(0, 2):
            image, boxes = _mirror(image, boxes)

        if np.random.randint(0, 2):
            image, boxes = _rotate(image, boxes)

        image, boxes = _persp(image, boxes)
        # crop
        image, boxes, labels = _crop(image, boxes, labels)
        height_c, width_c, _ = image.shape

        # test_flag = False
        # if test_flag:
        #     num_gts = boxes.shape[0]
        #     image_ = image.copy()
        #     boxes = boxes.astype(np.int)
        #     for i in range(num_gts):
        #         image_ = cv2.rectangle(image_, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 1)
        #     # cv2.rectangle(image_, (10, 10), (16, 16), (0, 0, 255), 3)
        #     cv2.imshow("", image_)
        #     if cv2.waitKey(0) == 27:
        #         exit(0)

        # 对图片和标注进行缩放
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, self.resize, interpolation=interp_method)

        new_h, new_w, _ = image.shape
        boxes[:, ::2] /= (width_c / new_w)
        boxes[:, 1::2] /= (height_c / new_h)

        # test_flag = False
        # if test_flag:
        #     num_gts = boxes.shape[0]
        #     image_ = image.copy()
        #     boxes = boxes.astype(np.int)
        #     for i in range(num_gts):
        #         image_ = cv2.rectangle(image_, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 255), 1)
        #     # cv2.rectangle(image_, (10, 10), (16, 16), (0, 0, 255), 3)
        #     cv2.imshow("", image_)
        #     if cv2.waitKey(0) == 27:
        #         exit(0)

        # 增广
        # image, boxes = _distort(image, boxes)
        image, boxes = _noise(image, boxes)

        image = image.astype(np.float32)
        # image -= self.means[::-1]
        # image /= self.std
        image_t = image.transpose(2, 0, 1)
        # image_t = image

        boxes[:, 0::2] /= new_w
        boxes[:, 1::2] /= new_h

        labels_t = np.expand_dims(labels, 1)
        targets_t = np.hstack((boxes, labels_t))

        return torch.from_numpy(image_t), targets_t
        # return image_t.astype(np.uint8), targets_t


def _distort_o(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand_o(image, boxes, fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1, 4)

        min_ratio = max(0.5, 1. / scale / scale)
        max_ratio = min(2, scale * scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale * ratio
        hs = scale / ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)

        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror_o(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


class preproc(object):

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), p=0.2):
        self.means = rgb_means
        self.std = rgb_std
        self.resize = resize
        self.p = p

    def __call__(self, image, targets):
        '''
        :param image: numpy bgr格式 shape(1080,1920,3)
        :param targets: numpy 格式 shape (gt数, 5)，未归一化
        :return:
        '''
        boxes = targets[:, :-1].copy()  # shape:(gt数, 4)
        labels = targets[:, -1].copy()  # shape:(gt数,)

        # 如果没有gt的话，不做增广，返回一个全0的gt
        if len(boxes) == 0:
            # boxes = np.empty((0, 4))
            targets = np.zeros((1, 5))
            image = preproc_for_test(image, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]

        # boxes长宽归一化
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)      # 增加一个维度，shape变为(gt数, 1)
        targets_o = np.hstack((boxes_o, labels_o))  # 将boxes_o 和 labels_o 合在一起，shape变为(gt数, 5)

        # 数据增广
        image_t, boxes, labels = _crop_o(image, boxes, labels)
        # cv2.imwrite('/data_ssd2/hzh/PytorchSSD1215/vis/test_crop.jpg', image_t)
        # print()
        image_t = _distort_o(image_t)
        image_t, boxes = _expand_o(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror_o(image_t, boxes)
        # cv2.imwrite('/data_ssd2/hzh/PytorchSSD1215/vis/test_mirror.jpg', image_t)
        # print()
        # image_t, boxes = _mirror(image, boxes)

        image_t, boxes = image, boxes

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means, self.std)  # 将图片resize为网络训练的大小
        # cv2.imwrite('/data_ssd2/hzh/PytorchSSD1215/vis/test_final.jpg', image_t)
        # print()

        # 取boxes原来的大小，针对增广后的大小重新归一化
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        # 如果gt太小的话就舍去
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.  # shape (gt数,)
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.  # shape (gt数,)
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        # 如果增广后没有符合要求的gt， 将增广前的图像和gt resize 后返回
        if len(boxes_t) == 0:
            image = preproc_for_test(image_o, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return torch.from_numpy(image_t), targets_t


class preproc_adv(object):

    '''
    preprocessing for adversarial training
    return origin image in CHW format
    '''

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1)):
        self.means = rgb_means
        self.std = rgb_std
        self.resize = resize

    def __call__(self, image, targets):
        '''
        :param image: numpy bgr格式 shape(1080,1920,3)
        :param targets: numpy 格式 shape (gt数, 5)
        :return:
        '''

        # 如果没有gt的话，不做增广，返回一个全0的gt
        if len(targets) == 0:
            targets = np.zeros((1, 5))
            image = preproc_for_test(image, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets

        # boxes = targets[:, :-1]
        # height, width = image.shape[:2]
        # boxes[:, 0::2] /= width
        # boxes[:, 1::2] /= height

        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)

        # 训练的时候没有对数据做归一化
        return torch.from_numpy(image), targets


class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        rgb_std: std of the dataset
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.std = rgb_std
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), self.resize, interpolation=interp_method).astype(np.float32)
        # TODO 暂时去掉了rgb的归一化，因为训练时也没有预处理
        # img -= self.means
        # img /= self.std
        img = img.transpose(self.swap)
        return torch.from_numpy(img)


if __name__ == "__main__":
    # _test_rotate()
    # pass
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/original/screenshots/00036.png')

    # 测试标记
    targets = np.array([[1000, 400, 1050, 450, 1],
                        [800, 700, 850, 750, 1]]).astype(np.float)

    targets = targets.astype(np.int)
    image = cv2.rectangle(image, (targets[0][0], targets[0][1]), (targets[0][2], targets[0][3]), (0, 0, 255), 2)
    image = cv2.rectangle(image, (targets[1][0], targets[1][1]), (targets[1][2], targets[1][3]), (0, 0, 255), 2)
    targets = targets.astype(np.float)

    image, targets = preproc_new((512, 512))(image, targets)

    targets = targets.astype(np.int)
    image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    image = cv2.rectangle(image, (targets[0][0], targets[0][1]), (targets[0][2], targets[0][3]), (255, 0, 0), 2)
    image = cv2.rectangle(image, (targets[1][0], targets[1][1]), (targets[1][2], targets[1][3]), (255, 0, 0), 2)
    targets = targets.astype(np.float)

    cv2.imshow("", image.copy().astype(np.uint8))
    cv2.waitKey(0)
