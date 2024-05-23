import torch
import torch.nn as nn
from itertools import product
# from layers.weight_init import kaiming_init, constant_init, normal_init


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


class DotsGenerator50(nn.Module):

    def __init__(self):
        super(DotsGenerator50, self).__init__()

        # 每个散点的坐标，相对gt左上位置计算，每个散点由3x3个像素构成
        self.dot_list = torch.tensor([(35, 25), (25, 35), (15, 25), (25, 15), (45, 25), (39, 39), (25, 45),
                                      (11, 39), (5, 25), (11, 11), (25, 5), (39, 11), (22, 25), (28, 25),
                                      (25, 22), (25, 28), (25, 25)]).cuda()
        self.dirs = torch.tensor(list(product([-1, 0, 1], [-1, 0, 1]))).cuda()
        # self.conv1 = CBR(3, 17, 3, 2)
        # self.conv2 = CBR(17, 51, 3, 2)
        # self.conv3 = CBR(51, 51, 3, 2)
        self.conv1 = CBR(3, 17, 3, 2)
        self.conv2 = CBR(17, 51, 3, 1)
        self.conv3 = CBR(51, 51, 3, 1)
        self.fc = nn.Linear(51 * 25 * 25, 51)
        # self._weight_init()

    def get_crops(self, image, targets):
        '''
        根据标注得到一张图上所有的gt块
        :param image: 原图，shape为[3, 1080, 1920]
        :param targets: 原图的gt标注框，shape为[num_gts, 5]
        :return: gt_crop，shape为[num_gts, 3, 50, 50]
        '''
        # generate gt_crops, shape: [num_gts, 3, 40, 40]
        num_gts = targets.shape[0]
        gt_crops_list = []
        image = image.permute(1, 2, 0)  # [1080, 1920, 3]
        for i in range(num_gts):
            gt_crop = image[int(targets[i][1]): int(targets[i][3]),
                      int(targets[i][0]): int(targets[i][2])].permute(2, 0, 1).unsqueeze(0)
            gt_crops_list.append(gt_crop)
        gt_crops = torch.cat(gt_crops_list, 0)
        return gt_crops

    def forward(self, gt_crops):
        '''
        :param gt_crop，shape为[num_gts, 3, 50, 50]
        :return: out，强度数据，shape为[num_gts, 17, 3]
        '''

        out = self.conv1(gt_crops)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = out.view(-1, 7 * 7 * 51)
        out = out.view(-1, 25 * 25 * 51)
        out = self.fc(out)
        # print(torch.sum(out < 0), out.shape)
        out = out.view(-1, 17, 3)
        out = torch.sigmoid(out) * 255
        # out = out.to(torch.uint8)
        return out  # [num_gts, 17, 3]

    def rewrite_image(self, out, image, targets):
        '''
        根据卷积得到的结果在原图上打上标记
        :param out: 强度数据，shape为[num_gts, 17, 3]
        :param image: 原图，shape为[3, 1080, 1920]
        :param targets: 原图的gt标注框，shape为[num_gts, 5]
        :return: image: 打上标记之后的图，shape为[3, 1080, 1920]
        '''
        # out为强度数据，shape为[num_gts, 51, 1, 1]
        # out = out.squeeze(3).view(-1, 17, 3)  # [num_gts, 17, 3]
        num_gts = out.shape[0]
        image = image.permute(1, 2, 0)

        # 所有gt左上角的坐标
        gt_left_tops = targets[:, 0:2].to(torch.int64)  # [num_gts, 2]

        gt_left_tops_format = gt_left_tops.unsqueeze(1).repeat(1, 17, 1)  # [num_gts, 17, 2]

        # 散点在每张图上的绝对坐标，[num_gts, 17, 2]
        # FIXME FloatTensor & LongTensor conflict
        dot_list_format = self.dot_list.unsqueeze(0).repeat(num_gts, 1, 1) + gt_left_tops_format

        # 每个点重复9次再加上offset，得到每个标记中心位置所有像素点的坐标
        # [num_gts, 17, 9, 2]
        dot_list_format = dot_list_format.unsqueeze(2).repeat(1, 1, self.dirs.shape[0], 1) + self.dirs
        # dots position: [num_gts * 17 * 9, 2]
        dot_list_format = dot_list_format.view(-1, 2)

        # out: [num_gts, 17, 9, 3] -> [num_gts * 17 * 9, 3]
        out = out.unsqueeze(2).repeat(1, 1, self.dirs.shape[0], 1).view(-1, 3)

        # 筛选出所有散点
        # FIXME in-place操作
        image_r = image.clone()
        image_r[dot_list_format[:, 1], dot_list_format[:, 0]] = out[:]
        # image_r[dot_list_format[:, 1], dot_list_format[:, 0]] = out[:] * 0.5 + image_r[dot_list_format[:, 1], dot_list_format[:, 0]] * 0.5

        return image_r.permute(2, 0, 1)

    # def _weight_init(self):
    #     # TODO 之前是怎么初始化的
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             # kaiming_init(m)
    #             # constant_init(m, 0.01)
    #             normal_init(m, std=0.01)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             # constant_init(m, 0.1)
    #             constant_init(m, 1)
    #         elif isinstance(m, nn.Linear):
    #             normal_init(m, std=0.01)
    #             # constant_init(m, 0.01)


def run():
    import numpy as np
    import cv2
    from PIL import Image
    from torchvision import transforms
    from matplotlib import pyplot as plt
    import pdb
    # img = Image.open('/sda1/Datasets/Hzh/mark/original/screenshots/00006.png').convert('RGB')
    image = cv2.imread('/sda1/Datasets/Hzh/mark/original/screenshots/00006.png')
    image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
    targets = torch.tensor([[300, 400, 350, 450, 1], [300, 210, 350, 260, 1]]).cuda()
                            # , [500, 900, 550, 950, 1], [800, 600, 850, 650, 1], [1000, 500, 1050, 550, 1]]).cuda()
    # targets = torch.tensor([[200, 200, 250, 250, 1]]).cuda()
    dg = DotsGenerator50().to('cuda')
    ckpt_path = '/home/chengxin/Project/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/online/gen_weak.pth'
    dg.load_state_dict(torch.load(ckpt_path))
    dg.eval()
    gt_crops = dg.get_crops(image, targets)
    out = dg(gt_crops)
    dot_image = dg.rewrite_image(out, image, targets)
    # pdb.set_trace()
    dot_image = dot_image.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    dot_image = dot_image[:, :, ::-1]
    plt.imshow(dot_image)
    plt.show()
    # cv2.imwrite((out_path, 'test.jpg'), dot_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    pass


if __name__ == "__main__":
    run()
    exit()
    import cv2
    import os
    import numpy as np
    image = cv2.imread('/data_ssd2/hzh/paperworks/dataset/screenshots/00036.png')
    image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
    targets = torch.tensor([[300, 400, 340, 440, 1], [500, 900, 540, 940, 1], [800, 600, 840, 640, 1], [1000, 500, 1040, 540, 1]])
    print(targets.is_cuda)
    exit()
    targets50 = torch.tensor(
        [[300, 400, 350, 450, 1], [500, 900, 550, 950, 1], [800, 600, 850, 650, 1], [1000, 500, 1050, 550, 1]])

    dg = DotsGenerator50()
    out_path = "/data_ssd2/hzh/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/20210407"

    # for i in range(5, 65, 5):
    #     dg.load_state_dict(torch.load(os.path.join(out_path, 'DotGenerator_MARK_epoches_' + str(i) + '.pth')))
    #     dg.eval()
    #     gt_crops = dg.get_crops(image, targets)
    #     out = dg(gt_crops)
    #     _, image_ = dg.rewrite_image(out, image, targets)
    #     image_ = image_.permute(1, 2, 0).detach().cpu().numpy()
    #     cv2.imwrite(os.path.join(out_path, 'epoch' + str(i) + '.jpg'), image_, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    dg.eval()
    gt_crops = dg.get_crops(image, targets50)
    out = dg(gt_crops)
    image_ = dg.rewrite_image(out, image, targets50)
    image_ = image_.permute(1, 2, 0).detach().cpu().numpy()
    # cv2.imwrite(os.path.join(out_path, 'test.jpg'), image_, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    targets50 = targets50.detach().cpu().numpy()
    for i in range(targets50.shape[0]):
        image_ = cv2.rectangle(image_, (targets50[i][0], targets50[i][1]), (targets50[i][2], targets50[i][3]), (0,0,255), 1)


    cv2.imshow("", image_.astype(np.uint8))
    cv2.waitKey(0)



