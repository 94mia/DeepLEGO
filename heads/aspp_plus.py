'''
Re-implementation of ASPP+ module introduced in paper [1]

Reference:
[1] Rethinking Atrous Convolution for Semantic Image Segmentation:
    https://arxiv.org/abs/1706.05587
'''
import torch.nn as nn
import torch
import torch.nn.functional as F


class ASPP_plus(nn.Module):
    def __init__(self, params, in_encoder=None):
        """
        If ASPP_plus module is in encoder network, then set in_encoder=True,
            thus ASPP_plus won't perform the classification convolution
            during forward prop, the output will be a 256 channel feature map
            with spatial size (image_height//output_stride, image_width//output_stride,)
        else ASPP_plus will output classification feature map with
            spatial size (image_height, image_width)
        """
        super(ASPP_plus, self).__init__()

        if hasattr(params, 'dilation'):
            if len(params.dilation) == 3:
                dilation = params.dilation
            else:
                dilation = (6, 12, 18)
        else:
            dilation = (6, 12, 18)

        if in_encoder is None:
            if hasattr(params, 'in_encoder'):
                self.in_encoder = params.in_encoder
            else:
                self.in_encoder = False
        else:
            self.in_encoder = in_encoder

        # add output_channels parameter
        if self.in_encoder:
            self.output_channels = 256

        self.conv11_1 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 1, bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.conv11_2 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 1, bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 3,
                                                padding=dilation[0], dilation=dilation[0], bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 3,
                                                padding=dilation[1], dilation=dilation[1], bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 3,
                                                padding=dilation[2], dilation=dilation[2], bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.concate_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                      nn.BatchNorm2d(256)).cuda()
        self.class_conv = nn.Sequential(nn.Conv2d(256, params.num_class, 1),
                                        nn.Upsample(scale_factor=params.output_stride, mode='bilinear',
                                                    align_corners=False)).cuda()

    def forward(self, logits):
        x = logits[-1]
        conv11 = self.conv11_1(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsample
        image_pool = F.avg_pool2d(x, x.shape[2:])
        image_pool = self.conv11_2(image_pool)
        upsample = F.upsample(image_pool, size=x.shape[2:], mode='bilinear', align_corners=False)

        # concatenate
        concatenate = torch.cat((conv11, conv33_1, conv33_2, conv33_3, upsample), dim=1)
        concatenate = self.concate_conv(concatenate)

        if self.in_encoder:
            logits[-1] = concatenate
            return logits
        else:
            return self.class_conv(concatenate)