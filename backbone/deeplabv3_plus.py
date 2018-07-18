import torch.nn as nn
from heads.aspp_plus import ASPP_plus
from backbone.xception import Xception65


class DeepLabv3_plus(nn.Module):
    """
    Backbone of DeepLabv3+
        Encoder part consists of both Xception and ASPP+
        TODO: support mobilenet or other backbone in future
    """
    def __init__(self, output_stride=16):
        super(DeepLabv3_plus, self).__init__()

        self.dcnn = Xception65(output_stride)
        self.aspp = ASPP_plus(self.dcnn.output_channels, in_encoder=True)

    def forward(self, x):

        logits = self.dcnn(x)
        assert len(logits) > 3, 'Length of logits must be larger than 3'
        low = logits[-4]
        high = self.aspp(logits[-1])

        return [low, high]
