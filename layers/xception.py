import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, 3,
                                   stride=stride, padding=dilation, dilation=dilation, groups=in_channels, bias=False)
        self.separable = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.separable(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x



class ResidualBlock(nn.Module):
    def __init__(self, in_channels, base_channels, stride, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, base_channels, dilation=dilation)

        self.conv2 = DepthwiseSeparableConv(base_channels, base_channels, dilation=dilation)

        self.conv3 = DepthwiseSeparableConv(base_channels, base_channels,
                                            stride=stride, dilation=dilation)

        if stride > 1:
            self.shortcut = nn.Conv2d(in_channels, base_channels, 1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x):
        orig = x

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.shortcut is not None:
            orig = self.shortcut(orig)

        x += orig

        return x