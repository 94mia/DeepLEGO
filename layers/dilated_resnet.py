import torch.nn as nn
from layers.resnet import conv3x3


class PlainBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channels, base_channels, stride=1, dilation=1):
        super(PlainBlock, self).__init__()
        self.stride = stride

        self.conv1 = conv3x3(in_channels, base_channels, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(base_channels)

        self.conv2 = conv3x3(base_channels, base_channels, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(base_channels)

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
