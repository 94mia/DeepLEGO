# DeepLEGO
This repository aims to separate state-of-the-art Semantic Segmentation Networks into several parts so that they can be concatenated like LEGO toys. 

Networks and Structures are:
```
DeepLabv3, DeepLabv3+, Vortex Pooling and so on

ResNet, Xception, MobileNetv1, MobileNetv2 and so on
```

------

# Folder Definition

Folder `backbone` stores backbone structures, which will output both logits and feature maps where output_stride=4. No fc layer in backbone cause we do not use them in image classification task.

Folder `datasets` stores transformations for different datasets like cityscapes or so.

Folder `layers` stores different layers or blocks used in backbone networks like residual block or bottleneck.

Folder `utils` stores useful LEGO toys like common decoder, ASPP, Vortex Pooling.

------

# Logs

| Date | Accomplishment |
|------|----------------|
| 7.13 | Repository Setup |
| 7.14 | add ResNet backbone, add Dilated ResNet backbone |
