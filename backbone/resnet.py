'''
Re-implementation of ResNet introduced in paper [1]
The structure of ResNet class is influenced by [2]
The has_max_pool parameter is a flag for max pooling layer after the first 7x7 convolution.
This max pooling is removed in paper [3]
When using [3] as a decoder, please don't forget to set params.has_max_pool=False

Reference:
[1] Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[3] Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
'''
import torch.nn as nn
from layers.resnet import Bottleneck, BasicBlock


class ResNet(nn.Module):
    def __init__(self, block, layers, params):
        super(ResNet, self).__init__()

        assert len(layers) == 4
        assert params.output_stride in [2, 4, 8, 16, 32]

        s1 = 2 if params.output_stride % 2 == 0 else 1
        d1 = 2 if s1 == 1 else 1
        s2 = 2 if params.output_stride % 4 == 0 else 1
        d2 = 2 * d1 if s2 == 1 else 1
        s3 = 2 if params.output_stride % 8 == 0 else 1
        d3 = 2 * d2 if s3 == 1 else 1
        s4 = 2 if params.output_stride % 16 == 0 else 1
        d4 = 2 * d3 if s4 == 1 else 1
        s5 = 2 if params.output_stride % 32 == 0 else 1
        d5 = 2 * d4 if s5 == 1 else 1

        if hasattr(params, 'HDC'):
            self.hdc = params.HDC

        # little definition
        conv1 = nn.Conv2d(3, 64,
                          kernel_size=7, stride=s1, padding=3, dilation=d1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu = nn.ReLU()
        max_pool = nn.MaxPool2d(kernel_size=3, stride=s2, padding=d2, dilation=d2)
        self.in_channels = 64

        if params.has_max_pool:
            self.stage1 = nn.Sequential(conv1, bn1, relu, max_pool).cuda()
            self.stage2 = self.conv_stage(block, 64, layers[0]).cuda()
        else:
            self.stage1 = nn.Sequential(conv1, bn1, relu).cuda()
            self.stage2 = self.conv_stage(block, 64, layers[0], s2, d2).cuda()

        self.stage3 = self.conv_stage(block, 128, layers[1], s3, d3).cuda()
        self.stage4 = self.conv_stage(block, 256, layers[2], s4, d4).cuda()
        self.stage5 = self.conv_stage(block, 512, layers[3], s5, d5).cuda()

        self.output_channels = 512 if block == BasicBlock else 2048

        # weight initialization
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def conv_stage(self, block, base_channels, n=1, stride=1, dilation=1):
        """Conv stage set up function

        Args:
            block: block function of current conv stage
            base_channels: number of basic channels in current conv stage
            n: number of repeat time
            stride: no explanation
        """
        layers = []
        if self.hdc and dilation>1:
            layers.append(block(self.in_channels, base_channels, stride=stride, dilation=dilation-1))
            self.in_channels = base_channels * block.expansion
            for i in range(1, n):
                layers.append(block(self.in_channels, base_channels, dilation=i%3+dilation-1))
                self.in_channels = base_channels * block.expansion
        else:
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

        x = self.stage1(x)
        x = self.stage2(x)
        logits.append(x)
        x = self.stage3(x)
        logits.append(x)
        x = self.stage4(x)
        logits.append(x)
        x = self.stage5(x)
        logits.append(x)

        return logits


def ResNet18(params, **kwargs):
    """
    Construct a ResNet-18 model
    """
    if not hasattr(params, 'has_max_pool'):
        params.has_max_pool = True

    return ResNet(BasicBlock, [2, 2, 2, 2], params)


def ResNet34(params, **kwargs):
    """
    Construct a ResNet-34 model
    """
    if not hasattr(params, 'has_max_pool'):
        params.has_max_pool = True

    return ResNet(BasicBlock, [3, 4, 6, 3], params)


def ResNet50(params, **kwargs):
    """
    Construct a ResNet-50 model
    """
    if not hasattr(params, 'has_max_pool'):
        params.has_max_pool = True

    return ResNet(Bottleneck, [3, 4, 6, 3], params)


def ResNet101(params, **kwargs):
    """
    Construct a ResNet-101 model
    """
    if not hasattr(params, 'has_max_pool'):
        params.has_max_pool = True

    return ResNet(Bottleneck, [3, 4, 23, 3], params)


def ResNet152(params, **kwargs):
    """
    Construct a ResNet-152 model
    """
    if not hasattr(params, 'has_max_pool'):
        params.has_max_pool = True

    return ResNet(Bottleneck, [3, 8, 36, 3], params)


# if __name__ == '__main__':
    # from config import Params
    # print(ResNet101(Params()))
    # print(__dict__)