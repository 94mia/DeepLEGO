from config import *
from network import MyNetwork, LOG, WARNING


def parse_network(net_str):
    net_str = net_str.split('+')
    net_backbone = net_str[:-1]
    net_head = net_str[-1]
    module = []
    for b in net_backbone:
        part = name2net[b](params, in_encoder=True)
        if hasattr(part, 'output_channels'):
            params.output_channels = part.output_channels
        module.append(part)
    module.append(name2net[net_head](params))

    return module


network = 'resnet101+aspp_plus+deeplabv3_plus_decoder'

params = Params()
params.model = network
params.dataset_root = '/media/ubuntu/disk/cityscapes'
params.has_max_pool = False
params.output_stride = 16

net = MyNetwork(params, module=parse_network(network))

net.Train()