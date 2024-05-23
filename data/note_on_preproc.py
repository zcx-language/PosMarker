import numpy as np

class preproc(object):

    def __init__(self, resize, rgb_means, rgb_std=(1, 1, 1), p=0.2):
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
        # boxes和labels是targets的深拷贝
        boxes = targets[:, :-1].copy()
        labels = targets[:, -1].copy()

        # 不存在gt的情况，一般不会出现
        if len(boxes) == 0:
            # boxes = np.empty((0, 4))
            targets = np.zeros((1, 5))
            image = preproc_for_test(image, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets

        # image_o和targets_o是image和targets的深拷贝
        image_o = image.copy()
        targets_o = targets.copy()

        # 保存原图长宽
        height_o, width_o, _ = image_o.shape

        # 这段代码主要是用来归一化bbox
        boxes_o = targets_o[:, :-1]
        labels_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o, 1)
        targets_o = np.hstack((boxes_o, labels_o))

        # 增广，这里的bbox都是归一化之后的值
        # 最小iou裁剪
        image_t, boxes, labels = _crop(image, boxes, labels)
        # 颜色扭曲
        image_t = _distort(image_t)
        # 扩充
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        # 镜像
        image_t, boxes = _mirror(image_t, boxes)
        # image_t, boxes = _mirror(image, boxes)

        # 增广之后的大小
        height, width, _ = image_t.shape
        # 缩放，换成CHW通道
        image_t = preproc_for_test(image_t, self.resize, self.means, self.std)

        # boxes是原图的尺寸的标注，根据resize后的尺寸再进行归一化
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height

        # 筛选掉小的框
        b_w = (boxes[:, 2] - boxes[:, 0]) * 1.
        b_h = (boxes[:, 3] - boxes[:, 1]) * 1.
        mask_b = np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        # 增广完没有适合的样本，增广白做
        if len(boxes_t) == 0:
            image = preproc_for_test(image_o, self.resize, self.means, self.std)
            return torch.from_numpy(image), targets_o

        labels_t = np.expand_dims(labels_t, 1)
        targets_t = np.hstack((boxes_t, labels_t))

        return torch.from_numpy(image_t), targets_t