import torch.nn as nn
import torch


class DeepLabv3_plus_decoder(nn.Module):
    """
    Decoder of DeepLabv3+
        Note that the input logits can consists either 2 logits or 4/5 logits respectively

        WARNING: the number of channel in higher layer logits must be 256.
    """
    def __init__(self, in_channels, num_class):
        super(DeepLabv3_plus_decoder, self).__init__()

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.conv11 = nn.Sequential(nn.Conv2d(in_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU())
        self.conv33 = nn.Sequential(nn.Conv2d(256+48, num_class, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(num_class),
                                    nn.ReLU())
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
        assert high.shape[1] == 256

        x = torch.cat([low, high], dim=1)
        x = self.conv33(x)
        x = self.upsample(x)

        return x