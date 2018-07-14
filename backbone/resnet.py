import torch.nn as nn
from layers.resnet import Bottleneck, BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride=32):
        super(ResNet, self).__init__()

        assert len(layers) == 4
        assert output_stride in [2, 4, 8, 16, 32]

        s1 = 2 if output_stride % 2 == 0 else 1
        d1 = 2 if s1 == 1 else 1
        s2 = 2 if output_stride % 4 == 0 else 1
        d2 = 2 * d1 if s2 == 1 else 1
        s3 = 2 if output_stride % 8 == 0 else 1
        d3 = 2 * d2 if s3 == 1 else 1
        s4 = 2 if output_stride % 16 == 0 else 1
        d4 = 2 * d3 if s4 == 1 else 1
        s5 = 2 if output_stride % 32 == 0 else 1
        d5 = 2 * d4 if s5 == 1 else 1

        # little definition
        conv1 = nn.Conv2d(3, 64,
                          kernel_size=7, stride=s1, padding=3, dilation=d1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        max_pool = nn.MaxPool2d(kernel_size=3, stride=s2, padding=1, dilation=d2)
        self.in_channels = 64

        self.stage1 = nn.Sequential(conv1, bn1, relu, max_pool)
        self.stage2 = self.conv_stage(block, 64,  layers[0])
        self.stage3 = self.conv_stage(block, 128, layers[1], s3, d3)
        self.stage4 = self.conv_stage(block, 256, layers[2], s4, d4)
        self.stage5 = self.conv_stage(block, 512, layers[3], s5, d5)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def conv_stage(self, block, base_channels, n=1, stride=1, dilation=1):
        """Conv stage set up function

        Args:
            block: block function of current conv stage
            base_channels: number of basic channels in current conv stage
            n: number of repeat time
            stride: no explanation
        """
        layers = []
        layers.append(block(self.in_channels, base_channels, stride=stride, dilation=dilation))
        self.in_channels = base_channels * block.expansion
        for _ in range(n-1):
            layers.append(block(self.in_channels, base_channels, dilation=dilation))
            self.in_channels = base_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Store all the intermediate feature maps
        """

        logits = []

        logits.append(self.stage1(x))
        logits.append(self.stage2(logits[-1]))
        logits.append(self.stage3(logits[-1]))
        logits.append(self.stage4(logits[-1]))
        logits.append(self.stage5(logits[-1]))

        return logits


def ResNet18():
    """
    Construct a ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    """
    Construct a ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    """
    Construct a ResNet-50 model
    """
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    """
    Construct a ResNet-101 model
    """
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    """
    Construct a ResNet-152 model
    """
    return ResNet(Bottleneck, [3, 8, 36, 3])


if __name__ == '__main__':
    print(ResNet18())