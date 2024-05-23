import torch
import torch.nn as nn
# from pytorch_msssim import ssim
from torchmetrics.functional.image.ssim import structural_similarity_index_measure as ssim

# TODO 改成 mse
# TODO 增加 ssim loss 和 nvf 进行对比？


class CropSSIMLoss(nn.Module):
    '''
    将单张图片的gt抠出来计算ssim loss
    '''
    def __init__(self):
        super(CropSSIMLoss, self).__init__()

    def forward(self, image, image_rewrite, target):
        '''
        :param image: 原图 [3,1080,1920]
        :param image_rewrite: 打上标记之后的图片 [3,1080,1920]
        :param target: gt标注框 [图片gt数, 5] (xmin, ymin, xmax, ymax, c)
        :return: ssim loss
        '''
        # TODO 怎么用targets把标记抠出来？
        image_p = image.permute(1, 2, 0)
        image_rewrite_p = image_rewrite.permute(1, 2, 0)
        loss_acm = 0
        num_gts = target.shape[0]
        for idx in range(num_gts):
            xmin, ymin, xmax, ymax, _ = [t.int() for t in target[idx]]
            crop = image_p[ymin: ymax, xmin: xmax].permute(2, 0, 1).unsqueeze(0)
            crop_rewrite = image_rewrite_p[ymin: ymax, xmin: xmax].permute(2, 0, 1).unsqueeze(0)
            loss_acm = loss_acm + 1 - ssim(crop, crop_rewrite, data_range=255, size_average=True)
        loss_mean = loss_acm / num_gts
        return loss_mean * 5
        # return torch.sum(image)
        # return 1 - ssim2(image.unsqueeze(0), image_rewrite.unsqueeze(0))


class CropMSELoss(nn.Module):
    '''
    将单张图片的gt抠出来计算mse loss
    '''
    def __init__(self):
        super(CropMSELoss, self).__init__()

    def forward(self, image, image_rewrite, dot_list_format):
        '''
        计算mse loss
        :param image: 原图
        :param image_rewrite: 打上标记之后的图片
        :param dot_list_format: 所有散点的坐标
        :return: loss
        '''
        image_p = image.permute(1, 2, 0)
        image_rewrite_p = image_rewrite.permute(1, 2, 0)
        num_dots = dot_list_format.shape[0]
        image_dots = image_p[dot_list_format[:, 1], dot_list_format[:, 0]]
        image_rewrite_dots = image_rewrite_p[dot_list_format[:, 1], dot_list_format[:, 0]]
        loss = torch.sum((image_dots - image_rewrite_dots)**2) / num_dots
        return loss

# class NVFLoss(nn.Module):
#     '''
#     计算单张图片的可见性损失
#     返回图片中所有标记点局部方差的平均值（每个标记点的像素为3*3，计算以标记点为中心的9*9像素块的方差，再除以点的个数）
#     '''
#
#     def __int__(self):
#         super(NVFLoss, self).__init__()
#
#     def forward(self, image_rewrite, dot_list_format):
#         '''
#         计算图片的标记点可见性损失
#         :param image_rewrite: 打上标记点之后的图片，[3,1080,1920]
#         :param dot_list_format: 图片上所有标记散点的绝对坐标，[-1,2]
#         :return: 可见性损失
#         '''
#         image = image_rewrite.permute(1, 2, 0)
#         dot_local_lt = dot_list_format - torch.tensor([4, 4])
#         dot_local_rb = dot_list_format + torch.tensor([5, 5])
#
#         num_dots = dot_list_format.shape[0]
#         num_gts = num_dots / 17
#
#         # max(vars)=17*8/9，对归一化的图片输入
#         vars = 0
#         # 对每一个散点
#         for idx in range(num_dots):
#             crop = image[dot_local_lt[idx][1]: dot_local_rb[idx][1],
#                    dot_local_lt[idx][0]: dot_local_rb[idx][0]]
#             center = torch.empty_like(crop)
#             # center[:, :] = image[dot_list_format[idx][1], dot_list_format[idx][0]]
#             center[:, :] = 1
#             var = torch.sum((crop-center)**2).item() / (81 * 3)
#             vars += var
#         return vars / num_gts


if __name__ == "__main__":
    # image_r = torch.zeros(3, 1080, 1920)
    dot_list_format = torch.tensor([[100, 200], [300, 403], [12, 323], [39, 21],
                                    [123, 243], [323, 405], [125, 32], [38, 321],
                                    [143, 223], [304, 400], [126, 335], [98, 31],
                                    [131, 254], [370, 407], [129, 332], [85, 123],
                                    [155, 222]])
    # criterion = NVFLoss()
    # loss = criterion(image_r, dot_list_format)
    # print(loss)

    image = torch.rand(3, 1080, 1920)
    image_r = torch.rand(3, 1080, 1920)
    criterion = CropMSELoss()
    loss = criterion(image, image_r, dot_list_format)
    print(loss)
