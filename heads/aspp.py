'''
Re-implementation of ASPP+ module introduced in paper [1]
The structure of this implementation also refers to the Caffe implementation in [2]

Reference:
[1] DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs:
    https://arxiv.org/abs/1606.00915
[2] SoonminHwang/caffe-segmentation/prototxt:
    https://github.com/SoonminHwang/caffe-segmentation/tree/master/prototxt
'''
import torch.nn as nn


class ASPP_branch(nn.Module):
    def __init__(self, in_channels, dilation, num_class):
        """
        One branch in ASPP module
        :param in_channels: number of input channels
        :param dilation: rate of dilation
        :param num_class: number of classes
        """
        super(ASPP_branch, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 1024, 3,
                               padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(1024, 1024, 1)
        self.conv3 = nn.Conv2d(1024, num_class, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)

        return x


class ASPP(nn.Module):
    def __init__(self, params):
        super(ASPP, self).__init__()

        if hasattr(params, 'dilation'):
            if len(params.dilation) == 4:
                dilation = params.dilation
            else:
                dilation = (6, 12, 18, 24)
        else:
            dilation = (6, 12, 18, 24)
        self.branch1 = ASPP_branch(params.output_channels, dilation[0], params.num_class)
        self.branch2 = ASPP_branch(params.output_channels, dilation[1], params.num_class)
        self.branch3 = ASPP_branch(params.output_channels, dilation[2], params.num_class)
        self.branch4 = ASPP_branch(params.output_channels, dilation[3], params.num_class)

        self.upsample = nn.Upsample(scale_factor=params.output_stride, mode='bilinear',
                                    align_corners=False)

    def forward(self, logits):
        x = logits[-1]

        x = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)
        x = self.upsample(x)

        return x
