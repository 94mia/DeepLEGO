# DeepLEGO

This is an on-going repository aims to separate state-of-the-art Semantic Segmentation Networks into several parts so that they can be concatenated freely like LEGO toys.

Current Available Networks and Structures are:
```
ResNet, Xception, Dilated ResNet, DeepLabv3+

ASPP, ASPP+, DeepLabv3+ decoder, Large Kernel Matters Decoder, SPP in PSPNet
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

Backbone networks are built depending on each semantic segmentation paper, all of them have no fc layer or global average
pooling layer thus classification task is not available.
Usually, CNNs have output_stride=32 from pooling or strided convolutions, and these continuous stridings make CNNs have


------

# Logs

| Date | Accomplishment |
|------|----------------|
| 7.13 | Repository Setup |
| 7.14 | add ResNet backbone, add Dilated ResNet backbone |
| 7.16 | add Xception backbone, ASPP, ASPP+, cityscapes dataset |
| 7.20 | add DeepLabv3+, PSPNet, Large Kernel Matters, config.py. network.py |

-------

# TO-DOs

- [ ] add SOTA network structures
- [ ] make a GettingStart tutorial
- [x] add more flexibility in progressbar
- [x] merge all network interface
- [ ] debug all structures
- [ ] add multi-grid and HDC to solve gridding
- [ ] make lr_mult and decay_mult changeable for backbone and head
