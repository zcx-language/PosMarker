import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import *

from models.VGG16_bn import get_vgg16_fms_ib
from utils.nms_wrapper import nms


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=True, up_size=0):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
        self.up_size = up_size

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.up_size > 0:
            x = F.interpolate(x, size=(self.up_size, self.up_size), mode='bilinear', align_corners=True)
        return x

def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1],kernel_size=(1, 3)[flag], stride=2, padding=1),
                           nn.BatchNorm2d(cfg[k + 1])]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag]),
                           nn.BatchNorm2d(v)]
            flag = not flag
        in_channels = v
    return layers

def multibox(fea_channels, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    assert len(fea_channels) == len(cfg)
    for i, fea_channel in enumerate(fea_channels):
        # Here use IB_Conv2d
        loc_layers += [nn.Conv2d(fea_channel, cfg[i] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(fea_channel, cfg[i] * num_classes, kernel_size=3, padding=1)]
    return (loc_layers, conf_layers)


# extras = {
#     # Conv(i,256),stride_Conv(256,512)
#     '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
#     '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
# }
# mbox = {
#     '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
#     '512': [6, 6, 6, 6, 6, 4, 4],
# }
# fea_channels = {
#     '300': [512, 1024, 512, 256, 256, 256],
#     '512': [512, 1024, 512, 256, 256, 256, 256]}

extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
    '1024': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256]
}
mbox = {
    '300': [6, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [6, 6, 6, 6, 6, 4, 4],
    '1024': [6, 6, 6, 6, 6, 4, 4]
}
fea_channels = {
    '300': [512, 1024, 512, 256, 256, 256],
    '512': [512, 1024, 512, 256, 256, 256, 256],
    '1024': [512, 1024, 512, 256, 256, 256, 256]}


class SSD(nn.Module):

    def __init__(self, num_classes, size):

        super(SSD, self).__init__()
        self.num_classes = num_classes
        self.size = size[0]
        # SSD network
        self.base =  get_vgg16_fms_ib()
        self.extras = nn.ModuleList(add_extras(extras[str(self.size)], 1024, batch_norm=True))
        #self.L2Norm = L2Norm(512, 20)
        head = multibox(fea_channels[str(self.size)], mbox[str(self.size)], self.num_classes)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax()

    def get_pyramid_feature_ib(self, x):
        source_fms = list()
        fms = self.base(x)
        source_fms += fms
        #source_fms[0] = self.L2Norm(source_fms[0])  #detecion 1
        x = source_fms[-1]
        for k, f in enumerate(self.extras):
            x = f(x)
            if k % 4 == 1 or k % 4 == 3:
                x = F.relu(x, inplace=True)
            if k % 4 == 3:
                source_fms.append(x)

        return source_fms

    def forward(self, x, test=False):
        loc = list()
        conf = list()

        pyramid_fea = self.get_pyramid_feature_ib(x)

        # apply multibox head to source layers
        # use ib after the multibox output
        for (x, l, c) in zip(pyramid_fea, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if test:
            output = (
                loc.view(loc.size(0), -1, 4),  # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
            )
        return output

    def init_model(self, base_model_path):
        # print('Not Pretrain Initialized')
        # return

        # init the base with pretrain model
        base_weights = torch.load(base_model_path)
        print('Loading base network...')
        try:
            self.base.layers.load_state_dict(base_weights)  #3/14
        except: #add batchnorm layers
            weights_map = {0: 0, 2: 3, 5: 7, 7: 10, 10: 14, 12: 17, 14: 20, 17: 24, 19: 27, 21: 30, 24: 34, 26: 37,
                           28: 40, 31: 44, 33: 46}
            model_dict = self.base.layers.state_dict()
            for k, v in base_weights.items():
                k_list = k.split('.')
                key_id = int(k_list[0])
                if key_id in weights_map:    #if need map(Conv2d) #k is str
                    k_list[0] = str(weights_map[key_id])
                    k = '.'.join(k_list)
                    if k in model_dict:
                        model_dict[k] = v
                # else:   #bn
                #     model_dict[k][...] = 1
            self.base.layers.load_state_dict(model_dict)

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        def weights_init2(m):
            for module in m.modules():
                if isinstance(m, nn.Conv2d):
                    init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    m.bias.data.fill_(0)

        def weights_init_bn(m):
            for module in m.modules():
                if isinstance(module, nn.BatchNorm2d):
                    init.constant_(module.weight, 1)
                    init.constant_(module.bias, 0)

        def weights_init_IB(m):
            for key in m.state_dict():
                if key.split('.')[0] in ['1','4','7']: #bn
                    m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'weight': #conv and deconv
                    if m.state_dict()[key].dim()>2:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0


        print('Initializing weights...')
        #FIXME:IB can't init
        self.base.layers.apply(weights_init_bn)
        #self.base.layers.apply(weights_init)
        #self.base.IB.base.apply(weights_init_IB)
        self.extras.apply(weights_init2)
        self.loc.apply(weights_init2)
        self.conf.apply(weights_init2)


    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_net(size=(300,300), num_classes=21):
    # if size != 300 and size != 512:
    #     print("Error: Sorry only FSSD300 and FSSD512 is supported currently!")
    #     return

    return SSD(num_classes=num_classes, size=size)


def run():
    import cv2
    import numpy as np
    import pdb
    from data.config import MARK_512
    from layers.functions import Detect, PriorBox
    from layers.modules import MultiBoxLoss

    priorbox = PriorBox(MARK_512)
    with torch.no_grad():
        priors = priorbox.forward()

    net = build_net(size=(512, 512), num_classes=2)
    net.load_state_dict(torch.load('/home/chengxin/Project/PytorchSSD1215/weights/SSD_vgg_bn_MARK_512/online/SSD_vgg_bn_MARK_weak.pth'))
    net.eval()
    net = net.to('cuda')

    detector = Detect(num_classes=2, bkg_label=0, cfg=MARK_512)
    img_path = '/home/chengxin/Project/PytorchSSD1215/images/Figure_1.png'
    img = cv2.resize(cv2.imread(img_path), (512, 512))
    img_tsr = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    out = net(x=img_tsr.cuda(), test=True)  # forward pass
    boxes, scores = detector.forward(out, priors)
    boxes = boxes[0].detach().cpu().numpy()
    scores = scores[0].detach().cpu().numpy()

    scale = torch.Tensor([img.shape[1], img.shape[0],
                            img.shape[1], img.shape[0]]).cpu().numpy()
    boxes *= scale

    all_boxes = [[[] for _ in range(1)]
                 for _ in range(2)]

    for j in range(1, 2):
        inds = np.where(scores[:, j] > 0.01)[0]
        if len(inds) == 0:
            all_boxes[j][0] = np.empty([0, 5], dtype=np.float32)
            continue
        c_bboxes = boxes[inds]
        c_scores = scores[inds, j]
        c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
            np.float32, copy=False)

        keep = nms(c_dets, 0.45)
        keep = keep[:50]
        c_dets = c_dets[keep, :]
        all_boxes[j][0] = c_dets
    

        detect_bboxes = all_boxes[j][i]

        if True:
            img_ = img_copy.copy()

            for class_id, class_collection in enumerate(detect_bboxes):
                if len(class_collection) > 0:
                    if class_collection[-1] > 0.2:
                        pt = class_collection

                        img_ = cv2.rectangle(img_.astype(np.uint8), (int(pt[0]), int(pt[1])), (int(pt[2]),
                                                                        int(pt[3])), (0, 0, 255), 1)
                        # cv2.putText(image, 'mark', (int(pt[0]), int(pt[1])), FONT,
                        #             0.3, (255, 255, 255), 1)
            cv2.imshow('result', img_.astype(np.uint8))
            cv2.waitKey(0)

    pdb.set_trace()
    print(boxes, scores)

    pass

if __name__ == '__main__':
    run()
    pass
    # print('SSD_IB_VGG...')
    # x = torch.rand(4, 3, 300, 300)
    # net = build_net(size=(300, 300), num_classes=21)
    # y = net(x)
