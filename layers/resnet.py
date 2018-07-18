import torch.nn as nn


def conv3x3(in_channels, out_channels, stride=1, dilation=1):
    """ 3x3 Convolution with padding and stride """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)

def conv1x1(in_channels, out_channels, stride=1, dilation=1):
    """ 1x1 Convolution with padding and stride """
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=1, stride=stride, padding=0, dilation=dilation, bias=False)

def conv3x3_bn_relu(in_channels, out_channels, stride=1, dilation=1):
    """ 3x3 Convolution with batch norm and relu """
    return nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                   kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())


class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, base_channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.stride = stride

        self.conv1 = conv3x3(in_channels, base_channels, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = conv3x3(base_channels, base_channels, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.relu = nn.ReLU()

        if in_channels != base_channels*self.expansion or stride != 1:
            self.shortcut = conv1x1(in_channels, base_channels*self.expansion, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        orig = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut is not None:
            orig = self.shortcut(orig)

        assert orig.shape == x.shape

        x += orig
        x = self.relu(x)

        return x


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, base_channels, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        self.stride = stride

        self.conv1 = conv1x1(in_channels, base_channels)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = conv3x3(base_channels, base_channels, stride=stride, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.conv3 = conv1x1(base_channels, base_channels*4)
        self.bn3 = nn.BatchNorm2d(base_channels*4)

        self.relu = nn.ReLU()

        if in_channels != base_channels*self.expansion:
            self.shortcut = conv1x1(in_channels, base_channels*self.expansion, stride=stride)
        else:
            self.shortcut = None

    def forward(self, x):
        orig = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.shortcut is not None:
            orig = self.shortcut(orig)

        x += orig
        x = self.relu(x)

        return x

