import os
import numpy as np
import cv2
import torch
import zipfile
from torchvision import transforms
from datasets.cityscapes import Cityscapes


WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')


def create_train_dir(params):
    """
    Create folder used in training, folder hierarchy:
        current folder--exp_folder
                       |
                       --summaries
                       --checkpoints
        the exp_folder is named by model_name + dataset_name
    """
    experiment = 'models/' + params.model + '_' + params.dataset
    exp_dir = os.path.join(os.getcwd(), experiment)
    summary_dir = os.path.join(exp_dir, 'summaries/')
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints/')

    dir = [exp_dir, summary_dir, checkpoint_dir]
    for dir_ in dir:
        if not os.path.exists(dir_):
            os.mkdir(dir_)

    return summary_dir, checkpoint_dir


def create_datasets(params):
    """
    Create datasets for training, testing and validating

    :return datasets: a python dictionary includes three datasets
    """
    phase = ['train', 'val', 'test']
    # if params.dataset_root is not None and not os.path.exists(params.dataset_root):
    #     raise ValueError('Dataset not exists!')

    transform = {'train': transforms.Compose([Rescale(params.rescale_size),
                                              RandomCrop(params.image_size),
                                              RandomHorizontalFlip(),
                                              ToTensor()
                                              ]),
                 'val'  : transforms.Compose([Rescale(params.image_size),
                                              ToTensor()
                                              ]),
                 'test' : transforms.Compose([Rescale(params.rescale_size),
                                              ToTensor()
                                              ])}

    # file_dir = {p: os.path.join(params.dataset_root, p) for p in phase}

    # datasets = {Cityscapes(file_dir[p], mode=p, transforms=transform[p]) for p in phase}
    datasets = {p: Cityscapes(params.dataset_root, mode=p, transforms=transform[p]) for p in phase}

    return datasets


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


def print_config(params):
    for name, value in sorted(vars(params).items()):
        print('\t%-20s:%s' % (name, str(value)))
    print('')


def generate_zip(dataset_root):
    """
    Generate zip files for submit in cityscapes-dataset.com
    :param dataset_root:
    :return:
    """
    azip = zipfile.ZipFile('submit.zip', 'w')
    txt = os.path.join(dataset_root, 'testLabels.txt')
    if os.path.exists(txt):
        for line in open(txt):
            line = line.strip()
            line = line.replace('labelTrainIds', 'labelIds')
            azip.write(os.path.join(dataset_root, line), arcname=line)
        azip.close()
    else:
        generate_txt(dataset_root, 'testLabels.txt')


def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    """
    Calculate mean and std of dataset
    """
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset + ep, axis=axis) / 255.0).tolist()


# if __name__ == '__main__':
    # trainId = cv2.imread('/media/ubuntu/disk/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png')
    # trainId2color(trainId[:, :, 0])