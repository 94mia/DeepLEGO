'''
Re-implementation of the decoder part of DeepLabv3+ introduced in paper [1]
This module can be concatenated with both common backbone or backbone with special module like ASPP+
In this case, set params.in_encoder=True for ASPP+ module

Reference:
[1] DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation
    https://arxiv.org/abs/1802.02611v2
'''
import torch.nn as nn
import torch


class DeepLabv3_plus_decoder(nn.Module):
    """
    Decoder of DeepLabv3+
        Note that the input logits can consists either 2 logits or 4/5 logits respectively
    """
    def __init__(self, params, **kwargs):
        super(DeepLabv3_plus_decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.conv11 = nn.Sequential(nn.Conv2d(params.output_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU()).cuda()
        self.conv33 = nn.Sequential(nn.Conv2d(params.output_channels+48, params.num_class, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(params.num_class),
                                    nn.ReLU()).cuda()
    def forward(self, logits):
        if len(logits) == 2:
            low = logits[0]
            high = logits[1]
        else:
            low = logits[-4]
            high = logits[-1]

        low = self.conv11(low)
        high = self.upsample(high)

        assert low.shape[2:] == high.shape[2:]

        x = torch.cat((low, high), dim=1)
        x = self.conv33(x)
        x = self.upsample(x)

        return x