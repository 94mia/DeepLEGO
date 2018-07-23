'''
Re-implementation of head module of PSPNet introduced in paper [1]
The structure of this module refers to the Caffe implementation from [2]

Reference:
[1] Pyramid Scene Parsing Network
    https://arxiv.org/abs/1612.01105
[2] hszhao/PSPNet/evaluation/prototxt/pspnet101_cityscapes_713.prototxt
    https://github.com/hszhao/PSPNet/blob/4b53f1c97a5921a99a965a60c0940eec2d46bb06/evaluation/prototxt/pspnet101_cityscapes_713.prototxt
'''
import torch.nn.functional as F
import torch.nn as nn
import torch


def conv_bn_relu(in_channels, out_channels, kernel_size=1):
    """ 1x1 Convolution with batch norm and relu """
    pad = (kernel_size-1) // 2
    return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=pad, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU()).cuda()


class PSP(nn.Module):
    def __init__(self, params):
        super(PSP, self).__init__()

        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.pool4 = nn.AdaptiveAvgPool2d((6, 6))

        self.conv1 = conv_bn_relu(params.output_channels, 512)
        self.conv2 = conv_bn_relu(params.output_channels, 512)
        self.conv3 = conv_bn_relu(params.output_channels, 512)
        self.conv4 = conv_bn_relu(params.output_channels, 512)

        self.conv5 = conv_bn_relu(512*4+params.output_channels, 512, 3)
        self.class_conv = nn.Conv2d(512, params.num_class, 1)
        self.output_stride = params.output_stride

    def forward(self, logits):
        x = logits[-1]
        input_size = x.shape[2:]

        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)
        x4 = self.pool4(x)

        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)

        x1 = F.upsample(x1, size=input_size, mode='bilinear', align_corners=False)
        x2 = F.upsample(x2, size=input_size, mode='bilinear', align_corners=False)
        x3 = F.upsample(x3, size=input_size, mode='bilinear', align_corners=False)
        x4 = F.upsample(x4, size=input_size, mode='bilinear', align_corners=False)

        x = torch.cat((x, x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        x = self.class_conv(x)

        x = F.upsample(x, scale_factor=self.output_stride)
        return x
