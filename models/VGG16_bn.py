
import torch
import torch.nn as nn
import torch.nn.functional as F

def vgg(cfg, batch_norm=False, pool5 = False, conv6_dilation=6):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    if pool5:
        pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    else:
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=conv6_dilation, dilation=conv6_dilation)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,nn.BatchNorm2d(1024),
               nn.ReLU(), conv7, nn.BatchNorm2d(1024), nn.ReLU()]
    return layers

vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

class Conv_bn_relu(nn.Module):
    def __init__(self, inp, oup, kernel_size=3, stride=1, padding=1, dilation=1, use_relu = True):
        super(Conv_bn_relu, self).__init__()
        self.use_relu = use_relu
        if self.use_relu:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, dilation=dilation,bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        else:
            self.convs = nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size, stride, padding, dilation=dilation,bias=False),
                nn.BatchNorm2d(oup,eps=1e-03),
            )

    def forward(self, x):
        out = self.convs(x)
        return out

class VGG16(nn.Module):

    def __init__(self, fm_ids, pool5, conv6_dilation):
        super(VGG16, self).__init__()
        self.layers = nn.ModuleList(vgg(vgg_config, batch_norm=True, pool5=pool5, conv6_dilation=conv6_dilation))
        #Common VGG Backbone
        self.fm_ids = fm_ids   #id of Feature Maps to detect

    def forward(self, x):
        fms = []
        for i, f in enumerate(self.layers):
            x = f(x)
            if i in self.fm_ids:
                fms.append(x)
        return fms

#FIXME: bn effect fm_ids
def  get_vgg16_fms_ib(fm_ids = [32,49], pool5=False, conv6_dilation=6):
    print('VGG16_bn  baseline2')
    return VGG16(fm_ids, pool5, conv6_dilation)
