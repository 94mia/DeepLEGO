import cv2
import torch
from collections import namedtuple
import numpy as np
import os
from torch.utils.data import Dataset

"""###############"""
"""# Definitions #"""
"""###############"""
# following definition are copied from github repository:
#   mcordts/cityscapesScripts/cityscapesscripts/helpers/labels.py
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

"""###################"""
"""# Transformations #"""
"""###################"""

def logits2trainId(logits):
    """
    Transform output of network into trainId map
    :param logits: output tensor of network, before softmax, should be in shape (#classes, h, w)
    """
    # squeeze logits
    # num_classes = logits.size[1]
    upsample = torch.nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=False)
    logits = upsample(logits.unsqueeze_(0))
    logits.squeeze_(0)
    logits = torch.argmax(logits, dim=0)

    return logits


def trainId2color(dataset_root, id_map, name):
    """
    Transform trainId map into color map
    :param dataset_root: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
    :param id_map: torch tensor
    :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
    """
    # transform = {label.trainId: label.color for label in labels}
    assert len(id_map.shape) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = id_map.shape
    color_map = np.zeros((h, w, 3))
    id_map = id_map.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            color_map[id_map == label.trainId] = np.array(label.color)
    color_map = color_map.astype(np.uint8)
    # color_map = cv2.resize(color_map, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)

    # save trainIds and color
    cv2.imwrite(dataset_root + '/' + name, id_map)
    name = name.replace('labelTrainIds', 'color')
    cv2.imwrite(dataset_root + '/' + name, color_map)

    return color_map


def trainId2LabelId(dataset_root, train_id, name):
    """
        Transform trainId map into labelId map
        :param dataset_root: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param id_map: torch tensor
        :param name: name of image, eg. 'gtFine/test/leverkusen/leverkusen_000027_000019_gtFine_labelTrainIds.png'
        """
    assert len(train_id.shape) == 2, 'Id_map must be a 2-D tensor of shape (h, w) where h, w = H, W / output_stride'
    h, w = train_id.shape
    label_id = np.zeros((h, w, 3))
    train_id = train_id.cpu().numpy()
    for label in labels:
        if not label.ignoreInEval:
            label_id[train_id == label.trainId] = np.array([label.id]*3)
    label_id = label_id.astype(np.uint8)
    # label_id = cv2.resize(label_id, dsize=(2048, 1024), interpolation=cv2.INTER_NEAREST)

    name = name.replace('labelTrainIds', 'labelIds')
    cv2.imwrite(dataset_root + '/' + name, label_id)


class Cityscapes(Dataset):
    def __init__(self, dataset_dir, mode='train', transforms=None):
        """
        Create Dataset subclass on cityscapes dataset
        :param dataset_dir: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
        :param mode: phase, 'train', 'test' or 'eval'
        :param transforms: transformation
        """
        self.dataset = dataset_dir
        self.transforms = transforms
        require_file = ['trainImages.txt', 'trainLabels.txt',
                        'valImages.txt',   'valLabels.txt',
                        'testImages.txt',  'testLabels.txt']

        # check requirement
        if mode not in ['train', 'test', 'val']:
            raise ValueError('Unsupported mode %s' % mode)

        if not os.path.exists(self.dataset):
            raise ValueError('Dataset not exists at %s' % self.dataset)

        for file in require_file:
            if file not in os.listdir(self.dataset):
                # raise ValueError('Cannot find file %s in dataset root folder!' % file)
                generate_txt(self.dataset, file)

        # create image and label list
        self.image_list = []
        self.label_list = []
        if mode == 'train':
            for line in open(os.path.join(self.dataset, 'trainImages.txt')):
                self.image_list.append(line.strip())
            for line in open(os.path.join(self.dataset, 'trainLabels.txt')):
                self.label_list.append(line.strip())
        elif mode == 'val':
            for line in open(os.path.join(self.dataset, 'valImages.txt')):
                self.image_list.append(line.strip())
            for line in open(os.path.join(self.dataset, 'valLabels.txt')):
                self.label_list.append(line.strip())
        else:
            for line in open(os.path.join(self.dataset, 'testImages.txt')):
                self.image_list.append(line.strip())
            for line in open(os.path.join(self.dataset, 'testLabels.txt')):
                self.label_list.append(line.strip())

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Overrides default method
        tips: 3 channels of label image are the same
        """
        image = cv2.imread(os.path.join(self.dataset, self.image_list[index]))
        label = cv2.imread(os.path.join(self.dataset, self.label_list[index]))  # label.size (1024, 2048, 3)
        image_name = self.image_list[index]
        label_name = self.label_list[index]

        sample = {'image': image, 'label': label[:, :, 0],
                  'image_name': image_name, 'label_name': label_name}

        if self.transforms:
            sample = self.transforms(sample)

        return sample


def generate_txt(dataset_root, file):
    """
    Generate txt files that not exists but required in both training and testing

    :param dataset_root: the path to dataset root, eg. '/media/ubuntu/disk/cityscapes'
    :param file: txt file need to generate
    """
    with open(os.path.join(dataset_root, file), 'w') as f:
        # get mode and folder
        if 'train' in file:
            mode = 'train'
        elif 'test' in file:
            mode = 'test'
        else:
            mode = 'val'
        folder = 'leftImg8bit' if 'Image' in file else 'gtFine'

        path = os.path.join(os.path.join(dataset_root, folder), mode)

        assert os.path.exists(path), 'Cannot find %s set in folder %s' % (mode, folder)

        # collect images or labels
        if 'Images' in file:
            cities = os.listdir(path)
            for city in cities:
                # write them into txt
                for image in os.listdir(os.path.join(path, city)):
                    print(folder + '/' + mode + '/' + city + '/' + image, file=f)
        else:
            image_txt = mode+'Images.txt'
            if image_txt in os.listdir(dataset_root):
                for line in open(os.path.join(dataset_root, image_txt)):
                    line = line.strip()
                    line = line.replace('leftImg8bit/', 'gtFine/')
                    line = line.replace('_leftImg8bit', '_gtFine_labelTrainIds')
                    print(line, file=f)
            else:
                generate_txt(dataset_root, image_txt)


if __name__ == '__main__':
    from config import Params
    import utils.functions as fn
    pp = Params()
    pp.dataset_root = '/media/ubuntu/disk/cityscapes'
    datasets = fn.create_datasets(pp)
