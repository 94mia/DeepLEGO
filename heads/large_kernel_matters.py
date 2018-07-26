'''
Re-implementation of decoder of Large Kernel Matters introduced in paper [1]

Reference:
[1] Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network
    https://arxiv.org/abs/1703.02719
'''
import torch.nn as nn


class GCN(nn.Module):
    def __init__(self, in_channels, num_class, k=15):
        super(GCN, self).__init__()

        pad = (k-1) // 2

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(1, k), padding=(0, pad), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False))

        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, num_class, kernel_size=(k, 1), padding=(pad, 0), bias=False),
                                   nn.Conv2d(num_class, num_class, kernel_size=(1, k), padding=(0, pad), bias=False))

    def forward(self, x):

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        assert x1.shape == x2.shape

        return x1 + x2


class BR(nn.Module):
    def __init__(self, num_class):
        super(BR, self).__init__()

        self.shortcut = nn.Sequential(nn.Conv2d(num_class, num_class, 3, padding=1, bias=False),
                                      nn.ReLU(),
                                      nn.Conv2d(num_class, num_class, 3, padding=1, bias=False))
    def forward(self, x):
        return x+self.shortcut(x)


class GCN_BR_BR_Deconv(nn.Module):
    def __init__(self, in_channels, num_class, k=15):
        super(GCN_BR_BR_Deconv, self).__init__()

        self.gcn = GCN(in_channels, num_class, k)
        self.br = BR(num_class)

        self.deconv = nn.ConvTranspose2d(num_class, num_class, 3, 2, 1, bias=False)

    def forward(self, x1, x2=None):
        output_size = [x1.shape[2]*2, x1.shape[3]*2]

        x1 = self.gcn(x1)
        x1 = self.br(x1)

        if x2 is None:
            x = self.deconv(x1, output_size=output_size)
        else:
            x = x1+x2
            x = self.br(x)
            x = self.deconv(x, output_size=output_size)

        return x


class LargeKernelDecoder(nn.Module):
    """
    Decoder of Large_Kernel_Matters, consists of GCN module and BR module

        WARNING: Encoder must have output_stride=32
    """
    def __init__(self, params, k=15):
        super(LargeKernelDecoder, self).__init__()

        assert params.output_stride == 32

        self.num_class = params.num_class
        self.k = k

        self.br = BR(params.num_class)
        self.deconv = nn.ConvTranspose2d(params.num_class, params.num_class, 3, 2, 1, bias=False)

    def forward(self, logits):
        assert len(logits) >= 4

        x1 = logits[-1]
        x2 = logits[-2]
        x3 = logits[-3]
        x4 = logits[-4]
        output_size = [x4.shape[2]*4, x4.shape[3]*4]

        branch1 = GCN_BR_BR_Deconv(x1.shape[1], self.num_class, self.k).cuda()
        branch2 = GCN_BR_BR_Deconv(x2.shape[1], self.num_class, self.k).cuda()
        branch3 = GCN_BR_BR_Deconv(x3.shape[1], self.num_class, self.k).cuda()
        branch4 = GCN_BR_BR_Deconv(x4.shape[1], self.num_class, self.k).cuda()

        x = branch1(x1)
        x = branch2(x2, x)
        x = branch3(x3, x)
        x = branch4(x4, x)

        x = self.br(x)
        x = self.deconv(x, output_size=output_size)
        x = self.br(x)

        return x
