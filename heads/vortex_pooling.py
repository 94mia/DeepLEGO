'''
Re-implementation of Vortex Pooling introduced in paper [1]

Reference:
[1] Vortex Pooling: Improving Context Representation in Semantic Segmentation
    https://arxiv.org/abs/1804.06242
'''
import torch.nn as nn
from layers.resnet import conv3x3_bn_relu
import torch.nn.functional as F
import torch


class VortexPooling(nn.Module):
    """
    Only Module B is implemented because AvgPool of PyTorch do not support dilation pooling
    """
    def __init__(self, params):
        super(VortexPooling, self).__init__()

        if hasattr(params, 'dilation'):
            if len(params.dilation) == 3:
                dilation = params.dilation
            else:
                dilation = (3, 9, 27)
        else:
            dilation = (3, 9, 27)

        if hasattr(params, 'in_encoder'):
            self.in_encoder = params.in_encoder
        else:
            self.in_encoder = False

        self.avg_pool1 = nn.AvgPool2d(dilation[0], padding=(dilation[0]-1)//2, stride=1)
        self.avg_pool2 = nn.AvgPool2d(dilation[1], padding=(dilation[1]-1)//2, stride=1)
        self.avg_pool3 = nn.AvgPool2d(dilation[2], padding=(dilation[2]-1)//2, stride=1)
        self.conv1 = conv3x3_bn_relu(params.output_channels, 256)
        self.conv2 = conv3x3_bn_relu(params.output_channels, 256, dilation=self.dilation[0])
        self.conv3 = conv3x3_bn_relu(params.output_channels, 256, dilation=self.dilation[1])
        self.conv4 = conv3x3_bn_relu(params.output_channels, 256, dilation=self.dilation[2])
        self.conv33 = nn.Sequential(nn.Conv2d(params.output_channels, 256, 3, bias=False),
                                    nn.BatchNorm2d(256))
        self.dilation = dilation
        self.in_encoder = self.in_encoder

    def forward(self, logits):
        x = logits[-1]

        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        image_pool = F.avg_pool2d(x.shape[2:])
        image_pool = self.conv33(image_pool)
        upsample = F.upsample(image_pool, size=x.shape[2:], mode='bilinear', align_corners=False)

        concatenate = torch.cat((x1, x2, x3, x4, upsample), dim=1)
        concatenate = self.concate_conv(concatenate)

        if self.in_encoder:
            logits[-1] = concatenate
            return logits
        else:
            return self.class_conv(concatenate)