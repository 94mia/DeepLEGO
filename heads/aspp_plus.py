import torch.nn as nn
import torch


class ASPP_plus(nn.Module):
    def __init__(self, in_channels, dilation=(6, 12, 18), in_encoder=False):
        """
        If ASPP_plus module is in encoder network, then set in_encoder=True,
            thus ASPP_plus won't perform the classification convolution
            during forward prop, the output will be a 256 channel feature map
            with spatial size (input_height//output_stride, input_width//output_stride,)
        else ASPP_plus will output classification feature map with
            spatial size (input_height, input_width)
        """
        super(ASPP_plus, self).__init__()
        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, 256, 1, bias=False),
                                     nn.BatchNorm2d(256))
        self.conv33_1 = nn.Sequential(nn.Conv2d(in_channels, 256, 3,
                                                padding=dilation[0], dilation=dilation[0], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_2 = nn.Sequential(nn.Conv2d(in_channels, 256, 3,
                                                padding=dilation[1], dilation=dilation[1], bias=False),
                                      nn.BatchNorm2d(256))
        self.conv33_3 = nn.Sequential(nn.Conv2d(in_channels, 256, 3,
                                                padding=dilation[2], dilation=dilation[2], bias=False),
                                      nn.BatchNorm2d(256))
        self.concate_conv = nn.Sequential(nn.Conv2d(256*5, 256, 1, bias=False),
                                      nn.BatchNorm2d(256))
        self.class_conv = nn.Sequential(nn.Conv2d(256, self.params.num_class, 1),
                                        nn.Upsample(scale_factor=self.params.output_stride, mode='bilinear',
                                                    align_corners=False))
        # self.upsample = nn.Upsample(mode='bilinear', align_corners=True)

        self.in_encoder = in_encoder

    def forward(self, logits):
        x = logits[-1]
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

        # concatenate
        concatenate = torch.cat([conv11, conv33_1, conv33_2, conv33_3, upsample], dim=1)
        concatenate = self.concate_conv(concatenate)

        if self.in_encoder:
            return concatenate
        else:
            return self.class_conv(concatenate)