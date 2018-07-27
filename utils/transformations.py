import cv2
import numpy as np
import torch
import random
import math
import torchvision.transforms.functional as F
from PIL import Image


class Rescale(object):
    """
    Rescale the image in a sample to a given size.

    :param output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        image = cv2.resize(image, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        sample['image'], sample['label'] = image, label

        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors with normalization.
    """
    def __init__(self, output_stride=16):
        self.output_stride = output_stride

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1)).astype(np.float32)

        # reset label shape
        # w, h = label.shape[0]//self.output_stride, label.shape[1]//self.output_stride
        # label = cv2.resize(label, (h, w), interpolation=cv2.INTER_NEAREST).astype(np.int64)
        label = label.astype(np.int64)

        # normalize image
        image /= 255

        sample['image'], sample['label'] = torch.from_numpy(image), torch.from_numpy(label)

        return sample


class RandomHorizontalFlip(object):
    """
    Random flip image and label horizontally
    """
    def __call__(self, sample, p=0.5):
        image, label = sample['image'], sample['label']
        if np.random.uniform(0, 1) < p:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        sample['image'], sample['label'] = image, label

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample.

    :param output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w, :]

        label = label[top: top + new_h, left: left + new_w]

        sample['image'], sample['label'] = image, label

        return sample


class RandomResizedCrop(object):
    """
    Randomly crop image and label with random scale (default: of 0.5 to 2.0) and
        given size with the original aspect ratio.
    For memory saving and computation efficiency, an input is not rescaled into a
        random size and crop the random part of input with size (size, size).
    First, the cropped area with be calculated and the crop_size will be set
        as size/scale.
    Then randomly crop a piece of input with size (crop_size, crop_size).
    Finally, cropped piece will be resize into (size, size)

    :param size: expected output size of each edge
    :param scale: range of size of the origin size cropped
    :param interpolation: Default: BILINEAR, the definition of interpolation is:
            NEAREST = NONE = 0
            LANCZOS = ANTIALIAS = 1
            BILINEAR = LINEAR = 2
            BICUBIC = CUBIC = 3
            BOX = 4
            HAMMING = 5
    """
    def __init__(self, size, scale=(0.5, 2.0), interpolation=2):
        self.size = size
        self.interpolation = interpolation
        self.scale = scale

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = Image.fromarray(image)
        label  =Image.fromarray(label)

        # get crop size
        scale = random.uniform(*self.scale)
        crop_size = int(self.size/scale)

        # get crop parameters
        i = random.randint(0, image.size[1] - crop_size)
        j = random.randint(0, image.size[0] - crop_size)

        # crop and resize
        image = F.crop(image, i, j, crop_size, crop_size)
        label = F.crop(label, i, j, crop_size, crop_size)
        image = F.resize(image, self.size, self.interpolation)
        label = F.resize(label, self.size, self.interpolation)

        sample['image'], sample['label'] = np.array(image), np.array(label)

        return sample