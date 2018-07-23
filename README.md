# DeepLEGO

This is an on-going repository aims to separate state-of-the-art Semantic Segmentation Networks into several parts so that they can be concatenated freely like LEGO toys.

Current Available Networks and Structures are:
```
ResNet, Xception, Dilated ResNet, DeepLabv3+

ASPP, ASPP+, DeepLabv3+ decoder, Large Kernel Matters Decoder, SPP in PSPNet, Vortex Pooling, HDC
```

------

# Folder Definition

Folder `backbone` stores backbone structures, which will output a list of logits where the network stride is 2.
                         No fc layer in backbone cause we do not use them in image classification task.

Folder `datasets` stores transformations for different datasets like cityscapes or so.

Folder `layers` stores different layers or blocks used in backbone networks like residual block or bottleneck.

Folder `heads` stores useful LEGO toys like common decoder, ASPP and so on.

Folder `utils` stores some useful functions like a hand-made progressbar

------

# Backbone Introduction

Backbone networks are built depending on each semantic segmentation paper, all of them have no fc layer or global average pooling layer thus classification task is not available.
    
Usually, CNNs have output_stride=32 from pooling or strided convolutions, and these continuous stridings make CNNs have the ability to against object shifts in spatial domain.
    
Each backbone network can remove stridings top-down by setting output_stride in parameters as 16, 8, 4 or 2. When this  modification happens, the dilation in convolution blocks after removed striding blocks will be doubled, you can set output_stride=16 in class Params defined in config.py and use print(ResNet18(params)) to see what has been changed

Considering the slight difference of backbone network in different segmentation model, I am trying to make all networks have the ability to be modified a little bit with flexibility, but usually papers in semantic segmentation uses ResNet as backbone, thus many modification can only be applied in ResNet. For example, you can set params.has_max_pool=False to disable the max pooling layer in ResNet while not in other backbones, also HDC(Hybrid Dilated Convolution) and scSE are available on ResNet, if you have other modifications, feel free to change the source code in the corresponding folder.

Backbone can be a list or just a torch.nn.module, when backbone is a list that means some decoder part is in backbone like ASPP+, please see head introduction below for more detail.

------

# Head Introduction

Head modules are often the decoder of segmentation networks, they can be concatenate with different backbones, I am working on adding more kinds of head module in the near future including SCSE, U-net decoder or even simple FCN decoder.

Some head modules have the ability to be embedded in backbone part, eg. ASPP+ module. This is a key method in DeepLabv3+ where they combine dilated convolution network with encoder-decoder module and achieved SOTA performance. Not all heads support this configuration, when using such module, please remember to set in_incoder=True.

------

# Dependencies

This implementation is written under Python 3.5 with following libs:

>torch 0.4.0</br>
torchvision 0.2.1</br>
numpy 1.14.5</br>
opencv-python 3.4.1.15</br>
tensorflow 1.8.0 (necessary for tensorboardX)</br>
tensorboardX 1.2</br>

------

# How to use?

I'm sorry to place this part so below cause I thought you may need to finish reading useful tips above first.
Currently the interface of this repo is still not stable and some combination of backbones and heads may not converge well, I am trying to solve them in my spare time.

So if you want to try this repo, make sure you install all dependencies and then you can try the following steps:

1. Customize `config.py` depending on the paper or just your wish, be sure to set `dataset_root` to the right path
2. Make a new python file as any name you like and write down `from network import MyNetwork` and `from config import Params`
3. Write down `MyNetwork(parmas=Params).Train()`
4. Run this python file, this will train a default network with ResNet18+ASPP module.

After the training, tensorboard is also available to visualize training procedure and the result on test set

------

# Logs

| Date | Accomplishment |
|------|----------------|
| 7.13 | Repository Setup |
| 7.14 | add ResNet backbone, add Dilated ResNet backbone |
| 7.16 | add Xception backbone, ASPP, ASPP+, cityscapes dataset |
| 7.20 | add DeepLabv3+, PSPNet, Large Kernel Matters, config.py. network.py |
| 7.23 | add vortex pooling, scSE in ResNet, fix many bugs, rearrange many many interfaces and so on |

-------

# TO-DOs

- [ ] add SOTA network structures
- [ ] make a GettingStart tutorial
- [x] add more flexibility in progressbar
- [x] merge all network interface
- [ ] debug all structures
- [ ] add multi-grid and HDC to solve gridding
- [ ] make lr_mult and decay_mult changeable for backbone and head

------

# References

1. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)
2. [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611v2)
3. [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579v1)
4. [Vortex Pooling: Improving Context Representation in Semantic Segmentation](https://arxiv.org/abs/1804.06242v1)
5. [DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Segmentation](https://arxiv.org/abs/1802.02611v2)
6. [Dilated Residual Networks](https://arxiv.org/abs/1705.09914)
7. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
8. [Large Kernel Matters -- Improve Semantic Segmentation by Global Convolutional Network](https://arxiv.org/abs/1703.02719)
9. [Understanding Convolutions for Semantic Segmentation](https://arxiv.org/abs/1702.08502v1)
