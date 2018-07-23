import os
from utils.transformations import *
import zipfile


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


def calc_dataset_stats(dataset, axis=0, ep=1e-7):
    """
    Calculate mean and std of dataset
    """
    return (np.mean(dataset, axis=axis) / 255.0).tolist(), (np.std(dataset + ep, axis=axis) / 255.0).tolist()


# if __name__ == '__main__':
    # trainId = cv2.imread('/media/ubuntu/disk/cityscapes/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png')
    # trainId2color(trainId[:, :, 0])