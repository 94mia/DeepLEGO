import torch.nn as nn
from layers.xception import DepthwiseSeparableConv, ResidualBlock
from layers.resnet import conv3x3_bn_relu


class Xception(nn.Module):
    def __init__(self, output_stride=16):
        super(Xception, self).__init__()

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

        # because stride conv in Residual Block is the last conv
        # thus dilation will be set in the next block
        self.entry1 = nn.Sequential(conv3x3_bn_relu(3, 32, s1, 1), conv3x3_bn_relu(32, 64),
                                    ResidualBlock(64, 128, s2, d1))
        self.entry2 = ResidualBlock(128, 256, s3, d2)
        self.entry3 = ResidualBlock(256, 728, s4, d3)
        self.middle = ResidualBlock(728, 728, 1, d4)
        self.exit = nn.Sequential(ResidualBlock(728, 1024, s5, d4),
                                  DepthwiseSeparableConv(1024, 1536, 1, d5),
                                  DepthwiseSeparableConv(1536, 1536, 1, d5),
                                  DepthwiseSeparableConv(1536, 2048, 1, d5))

        self.output_channels = 2048

    def forward(self, x):
        """
        Store all intermediate logits when network meets stride
        """

        logits = []

        x = self.entry1(x)
        logits.append(x)
        x = self.entry2(x)
        logits.append(x)
        x = self.entry3(x)
        logits.append(x)

        x = logits[-1]
        for _ in range(16):
            x = self.middle(x)
        logits.append(self.exit(x))

        return logits


def Xception65(output_stride=16):
    """
    Construct Xception network modified in DeepLabv3+
    """
    return Xception(output_stride)


if __name__ == '__main__':
    print(Xception65())

