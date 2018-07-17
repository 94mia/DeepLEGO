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
    def __init__(self, in_channels, params):
        super(ASPP, self).__init__()

        self.branch1 = ASPP_branch(in_channels, 6, params.num_class)
        self.branch2 = ASPP_branch(in_channels, 12, params.num_class)
        self.branch3 = ASPP_branch(in_channels, 18, params.num_class)
        self.branch4 = ASPP_branch(in_channels, 24, params.num_class)

        self.upsample = nn.Upsample(scale_factor=params.output_stride, mode='bilinear',
                                    align_corners=False)

    def forward(self, logits):
        x = logits[-1]

        x = self.branch1(x) + self.branch2(x) + self.branch3(x) + self.branch4(x)
        x = self.upsample(x)

        return x
