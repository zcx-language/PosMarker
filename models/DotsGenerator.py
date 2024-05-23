import torch
import torch.nn as nn
from itertools import product
from layers.weight_init import kaiming_init, constant_init, normal_init


class CBR(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True):
        super(CBR, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, dilation=dilation, bias=False),
                nn.BatchNorm2d(oup, eps=1e-03),
            )

    def forward(self, x):
        out = self.convs(x)
        return out


class DotsGenerator(nn.Module):

    def __init__(self):
        super(DotsGenerator, self).__init__()

        # 每个散点的坐标，相对gt左上位置计算，每个散点由3x3个像素构成
        self.dot_list = torch.tensor([(30, 20), (20, 30), (10, 20), (20, 10), (40, 20), (34, 34), (20, 40), (6, 34), (0, 20), (6, 6), (20, 0),
                        (34, 6), (17, 20), (23, 20), (20, 17), (20, 23), (20, 20)])
        self.dirs = list(product([-1, 0, 1], [-1, 0, 1]))
        # TODO：修改为卷积下采样、全连接层、sigmoid
        self.conv1 = CBR(3, 51)
        self.conv2 = CBR(51, 51)
        self.conv3 = CBR(51, 51, 40, 1, 0)
        self._weight_init()

    def forward(self, image, targets):
        '''
        :param image: 原图，shape为[3, 1080, 1920]
        :param targets: 原图的gt标注框，shape为[num_gts, 5]
        :return: 打上标记点之后的图片，shape为[3, 1080, 1920]
        '''

        num_gts = targets.shape[0]

        # generate gt_crops
        gt_crops_list = []
        image = image.permute(1, 2, 0)
        for i in range(num_gts):
            gt_crop = image[targets[i][0]: targets[i][2], targets[i][1]: targets[i][3]].permute(2, 0, 1).unsqueeze(0)
            gt_crops_list.append(gt_crop)
        gt_crops = torch.cat(gt_crops_list, 0)

        out = self.conv1(gt_crops)
        out = self.conv2(out)
        out = self.conv3(out).clamp(0, 255)

        # out为强度数据，shape为[num_gts, 51, 1, 1]
        out = out.squeeze(3).view(-1, 17, 3)  # [num_gts, 17, 3]

        # 所有gt左上角的坐标
        gt_left_tops = targets[:, 0:2]  # [num_gts, 2]

        gt_left_tops_format = gt_left_tops.unsqueeze(1).repeat(1, 17, 1)  # [num_gts, 17, 2]

        # 散点在每张图上的绝对坐标，[num_gts, 17, 2]
        dot_list_format = self.dot_list.unsqueeze(0).repeat(num_gts, 1, 1) + gt_left_tops_format

        # TODO: 没找到批量赋值的函数
        # image[dot_list_format] = out
        for i in range(num_gts):
            for j in range(17):
                for dir in self.dirs:
                    coord = tuple((dot_list_format[i][j] + torch.tensor(dir)))
                    image[coord] = out[i][j]
        return image.permute(2, 0, 1)

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)


if __name__ == "__main__":

    dg = DotsGenerator()
    # print(list(dg.parameters()))
    dg.eval()
    # exit()

    # image = torch.rand(3, 1080, 1920)
    # targets = torch.tensor([[10, 20, 50, 60, 1], [15, 30, 55, 70, 1], [22, 146, 62, 186, 1]])
    # out = dg(image, targets)
    # print(dg)
    # print(out.shape)

    import cv2
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')
    image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
    targets = torch.tensor([[300, 400, 340, 440, 1], [600, 800, 640, 840, 1], [500, 1000, 540, 1040, 1]])
    out = dg(image, targets)
    out = out.permute(1, 2, 0).detach().cpu().numpy()
    cv2.imwrite('/data_ssd2/hzh/PytorchSSD1215/vis/001.jpg', out, [int(cv2.IMWRITE_JPEG_QUALITY), 100])




