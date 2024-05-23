import torch.nn as nn
import torch.nn.functional as F
from layers.weight_init import xavier_init


class FPN_DW(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
        super(FPN_DW, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            # l_conv = ConvModule(
            #     in_channels[i],
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
            #     act_cfg=act_cfg,
            #     inplace=False)

            l_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 1, 1, bias=False),
                nn.GroupNorm(32, out_channels, eps=1e-5, affine=True)
            )

            # fpn_conv = ConvModule(
            #     out_channels,
            #     out_channels,
            #     3,
            #     padding=1,
            #     groups=out_channels,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     inplace=False)

            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=256, bias=False),
                nn.GroupNorm(32, out_channels, eps=1e-5, affine=True),
                nn.ReLU()
            )

            # fpn_conv1x1 = ConvModule(
            #     out_channels,
            #     out_channels,
            #     1,
            #     conv_cfg=conv_cfg,
            #     norm_cfg=norm_cfg,
            #     act_cfg=act_cfg,
            # )

            fpn_conv1x1 = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels, eps=1e-5, affine=True)
            )

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_convs.append(fpn_conv1x1)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                # extra_fpn_conv = ConvModule(
                #     in_channels,
                #     out_channels,
                #     3,
                #     stride=2,
                #     padding=1,
                #     conv_cfg=conv_cfg,
                #     norm_cfg=norm_cfg,
                #     act_cfg=act_cfg,
                #     inplace=False)
                extra_fpn_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False),
                    nn.GroupNorm(32, out_channels, eps=1e-5, affine=True),
                )
                self.fpn_convs.append(extra_fpn_conv)
        # print(self.fpn_convs)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[2*i+1](self.fpn_convs[2*i](laterals[i])) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    # print("=================", used_backbone_levels)
                    outs.append(self.fpn_convs[used_backbone_levels*2](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        # print([out.shape for out in outs])
        return tuple(outs)


if __name__ == "__main__":

    from models.shufflenetv2 import ShuffleNetV2_Plus
    import torch
    # norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
    head = ShuffleNetV2_Plus()
    head.eval()

    neck = FPN_DW(
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=2,
        add_extra_convs=False,
        num_outs=3)
    neck.eval()

    data = torch.rand(1, 3, 1024, 576)
    outputs = head(data)

    outputs = neck(outputs)
    for o in outputs:
        print(o.shape)