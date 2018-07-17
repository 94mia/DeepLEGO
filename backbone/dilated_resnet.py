import torch.nn as nn
from layers.resnet import BasicBlock
from layers.dilated_resnet import PlainBlock
"""
"""

class DilatedResNet(nn.Module):
    def __init__(self, layers, output_stride=8):
        super(DilatedResNet, self).__init__()

        assert len(layers) == 4
        assert output_stride in [2, 4, 8, 16, 32]

        s1 = 2 if output_stride % 2 == 0 else 1
        d1 = 2 if s1 == 1 else 1
        s2 = 2 if output_stride % 4 == 0 else 1
        d2 = 2*d1 if s2 == 1 else 1
        s3 = 2 if output_stride % 8 == 0 else 1
        d3 = 2*d2 if s3 == 1 else 1
        s4 = 2 if output_stride % 16 == 0 else 1
        d4 = 2 * d3 if s4 == 1 else 1
        s5 = 2 if output_stride % 32 == 0 else 1
        d5 = 2 * d4 if s5 == 1 else 1

        # little definition
        conv1 = nn.Conv2d(3, 16,
                          kernel_size=7, padding=3, bias=False)
        bn1 = nn.BatchNorm2d(16)
        relu = nn.ReLU()
        self.in_channels = 32

        self.stage1 = nn.Sequential(conv1, bn1, relu, BasicBlock(self.in_channels, 16))
        self.stage2 = BasicBlock(16, 32, stride=s1, dilation=d1)
        self.stage3 = self.conv_stage(BasicBlock, 64,  layers[0], s2, d2)
        self.stage4 = self.conv_stage(BasicBlock, 128, layers[1], s3, d3)
        self.stage5 = self.conv_stage(BasicBlock, 256, layers[2], s4, d4)
        self.stage6 = self.conv_stage(BasicBlock, 512, layers[3], s5, d5)
        self.stage7 = PlainBlock(self.in_channels, 512, dilation=2)
        self.stage8 = PlainBlock(self.in_channels, 512)

        self.output_channels = 512
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
        Store intermediate feature maps
        """

        logits = []

        x = self.stage1(x)
        x = self.stage2(x)
        logits.append(x)
        x = self.stage3(x)
        logits.append(x)
        x = self.stage4(x)
        logits.append(x)
        x = self.stage5(x)
        logits.append(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        logits.append(x)

        return logits


def DilatedResNet26():
    """
    Construct a Dilated ResNet-26 model
    """
    return DilatedResNet([2, 2, 2, 2])


def DilatedResNet42():
    """
    Construct a Dilated ResNet-42 model
    """
    return DilatedResNet([3, 4, 6, 3])


if __name__ == '__main__':
    print(DilatedResNet26())