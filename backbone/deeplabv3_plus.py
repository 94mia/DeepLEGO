'''
Re-implementation of encoder of DeepLabv3+ introduced in paper [1]

Reference:
[1] DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation
    https://arxiv.org/abs/1802.02611v2
'''
import torch.nn as nn
from heads.aspp_plus import ASPP_plus
from backbone.xception import Xception65


class DeepLabv3_plus(nn.Module):
    """
    Backbone of DeepLabv3+
        Encoder part consists of both Xception and ASPP+
        TODO: support mobilenet or other backbone in future
    """
    def __init__(self, params):
        super(DeepLabv3_plus, self).__init__()

        self.dcnn = Xception65(params)
        self.aspp = ASPP_plus(params, in_encoder=True)

    def forward(self, x):

        logits = self.dcnn(x)
        logits[-1] = self.aspp(logits[-1])

        return logits
