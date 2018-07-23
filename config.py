from utils.functions import create_train_dir, print_config


""" Dataset parameters """
class Params():
    def __init__(self):
        # network structure parameters
        self.model = 'ResNet34_DeepLabv3'  # define your model
        self.dataset = 'cityscapes'
        self.output_stride = 16
        self.down_sample_rate = 32  # classic down sample rate, DO NOT CHANGE!
        self.se_mode = None  # Squeeze and Excitation mode, 0->cSE, 1-> sSE, 2->scSE
        self.HDC = None  # Hybrid Dilated Convolution, type bool

        # dataset parameters
        self.rescale_size = 600  # rescale image when training
        self.image_size = 512  # the final image size after crop
        self.num_class = 20  # 20 classes for training
        self.dataset_root = '/path/to/your/dataset'
        self.dataloader_workers = 12
        self.shuffle = True  # if shuffle data when training
        self.train_batch = 10
        self.val_batch = 2
        self.test_batch = 1

        # train parameters
        self.num_epoch = 150
        self.base_lr = 0.00025
        self.power = 0.9  # lr decay power
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.should_val = True
        self.val_every = 2  # how often will we evaluate model on val set
        self.display = 1  # how often will we show train result

        # model restore parameters
        self.resume_from = None  # None for train from scratch
        self.pre_trained_from = None  # None for train from scratch
        self.should_save = True
        self.save_every = 10  # how often will we save checkpoint

        # create training dir
        self.summary_dir, self.ckpt_dir = create_train_dir(self)

if __name__ == '__main__':
    aa = Params()
    print_config(aa)