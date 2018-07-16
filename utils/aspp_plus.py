import torch.nn as nn
import torch


class ASPP_plus(nn.Module):
    def __init__(self, params):
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 1, bias=False),
                                     nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[0], dilation=params.aspp[0], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[1], dilation=params.aspp[1], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(params.c[-1], 256, 3,
                                                padding=params.aspp[2], dilation=params.aspp[2], bias=False),
                                      nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                      nn.BatchNorm2d(256))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

    def forward(self, x):
        conv11 = self.conv11(x)
        conv33_1 = self.conv33_1(x)
        conv33_2 = self.conv33_2(x)
        conv33_3 = self.conv33_3(x)

        # image pool and upsample
        image_pool = nn.AvgPool2d(kernel_size=x.size()[2:])
        image_pool = image_pool(x)
        image_pool = self.conv11(image_pool)
        upsample = nn.Upsample(size=x.size()[2:], mode='bilinear', align_corners=True)
        upsample = upsample(image_pool)

        # concate
        concate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)

        return self.concate_conv(concate)