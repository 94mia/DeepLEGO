from config import Params
from network import MyNetwork, LOG, WARNING

params = Params()
params.dataset_root = '/media/ubuntu/disk/cityscapes'
net = MyNetwork(params)

net.Train()