import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt

from utils.functions import *
from utils.progressbar import bar
from datasets.cityscapes import logits2trainId, trainId2color, trainId2LabelId, create_datasets
from heads.aspp import ASPP
from backbone.resnet import ResNet18


WARNING = lambda x: print('\033[1;31;2mWARNING: ' + x + '\033[0m')
LOG = lambda x: print('\033[0;31;2m' + x + '\033[0m')


# create model
class MyNetwork(nn.Module):
    """
    A Convolutional Neural Network used for Semantic Segmentation
    """

    """######################"""
    """# Model Construction #"""
    """######################"""

    def __init__(self, params, module=None):
        super(MyNetwork, self).__init__()

        # initializing network parameters
        self.params = params
        LOG('Network Configurations:\n')
        print_config(params)

        # creating network datasets
        LOG('Creating Datasets and Transformations......')
        self.datasets = create_datasets(params)
        LOG('Creation Succeed.\n')

        self.pb = bar()  # hand-made progressbar
        self.epoch = 0
        self.init_epoch = 0
        self.ckpt_flag = False
        self.train_loss = []
        self.val_loss = []
        if self.params.summary:
            self.summary_writer = SummaryWriter(log_dir=self.params.summary_dir)

        # build network structure
        self.backbone = module[:-1]
        self.head = module[-1]
        LOG('Building and Initializing Model......')
        self.build_network()
        LOG('Model Built.\n')

        # set default loss
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=255)

        # set default optimizer
        self.opt = torch.optim.RMSprop([{'params': self.head.parameters(), 'lr_mult': self.params.head_lr_mult},
                                        {'params': self.backbone_params, 'lr_mult': self.params.backbone_lr_mult}],
                                       lr=self.params.base_lr,
                                       momentum=self.params.momentum,
                                       weight_decay=self.params.weight_decay)

        # initialize
        self.build_dataloader()

        # load data
        self.load_checkpoint()
        self.load_model()

    def forward(self, x):
        if isinstance(self.backbone, list):
            logits = x
            for net in self.backbone:
                logits = net(logits)
        else:
            logits = self.backbone(x)

        seg = self.head(logits)

        return seg

    """######################"""
    """# Train and Validate #"""
    """######################"""

    def train_one_epoch(self):
        """
        Train network in one epoch
        """
        print('Training......')

        # set mode train
        self.train()

        # prepare data
        train_loss = 0

        train_size = len(self.datasets['train'])
        if train_size % self.params.train_batch != 0:
            total_batch = train_size // self.params.train_batch + 1
        else:
            total_batch = train_size // self.params.train_batch

        # train through dataset
        for batch_idx, batch in enumerate(self.train_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            # if self.params.should_split:
            #     image_cuda.requires_grad_()
            #     out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            # else:
            out = self(image_cuda)
            loss = self.loss_fn(out, label_cuda)

            # optimize
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # accumulate
            train_loss += loss.item()

            # record first loss
            if self.train_loss == []:
                self.train_loss.append(train_loss)
                if self.params.summary:
                    self.summary_writer.add_scalar('loss/train_loss', train_loss, 0)

        self.pb.close()
        train_loss /= total_batch
        self.train_loss.append(train_loss)

        # add to summary
        if self.params.summary:
            self.summary_writer.add_scalar('loss/train_loss', train_loss, self.epoch)

    def val_one_epoch(self):
        """
        Validate network in one epoch every m training epochs,
            m is defined in params.val_every
        """
        print('Validating......')

        # set mode eval
        self.eval()

        # prepare data
        val_loss = 0
        val_size = len(self.datasets['val'])
        if val_size % self.params.val_batch != 0:
            total_batch = val_size // self.params.val_batch + 1
        else:
            total_batch = val_size // self.params.val_batch

        # validate through dataset
        for batch_idx, batch in enumerate(self.val_loader):
            self.pb.click(batch_idx, total_batch)
            image, label = batch['image'], batch['label']
            image_cuda, label_cuda = image.cuda(), label.cuda()

            # checkpoint split
            # if self.params.should_split:
            #     image_cuda.requires_grad_()
            #     out = checkpoint_sequential(self.network, self.params.split, image_cuda)
            # else:
            out = self(image_cuda)

            loss = self.loss_fn(out, label_cuda)

            val_loss += loss.item()

            # record first loss
            if self.val_loss == []:
                self.val_loss.append(val_loss)
                if self.params.summary:
                    self.summary_writer.add_scalar('loss/val_loss', val_loss, 0)

        self.pb.close()
        val_loss /= total_batch
        self.val_loss.append(val_loss)

        # add to summary
        if self.params.summary:
            self.summary_writer.add_scalar('loss/val_loss', val_loss, self.epoch)


    def Train(self):
        """
        Train network in n epochs, n is defined in params.num_epoch
        """
        self.init_epoch = self.epoch
        if self.epoch >= self.params.num_epoch:
            WARNING('Num_epoch should be smaller than current epoch. Skip training......\n')
        else:
            for _ in range(self.epoch, self.params.num_epoch):
                self.epoch += 1
                print('-' * 20 + 'Epoch.' + str(self.epoch) + '-' * 20)

                # train one epoch
                self.train_one_epoch()

                # should display
                if self.epoch % self.params.display == 0:
                    print('\tTrain loss: %.4f' % self.train_loss[-1])

                # should save
                if self.params.should_save:
                    if self.epoch % self.params.save_every == 0:
                        self.save_checkpoint()

                # test every params.test_every epoch
                if self.params.should_val:
                    if self.epoch % self.params.val_every == 0:
                        self.val_one_epoch()
                        print('\tVal loss: %.4f' % self.val_loss[-1])

                # adjust learning rate
                self.adjust_lr()

            # save the last network state
            if self.params.should_save:
                self.save_checkpoint()

            # train visualization
            self.plot_curve()

    def Test(self):
        """
        Test network on test set
        """
        print('Testing......')
        # set mode eval
        torch.cuda.empty_cache()
        self.eval()

        # prepare test data
        test_size = len(self.datasets['test'])
        if test_size % self.params.test_batch != 0:
            total_batch = test_size // self.params.test_batch + 1
        else:
            total_batch = test_size // self.params.test_batch

        # test for one epoch
        for batch_idx, batch in enumerate(self.test_loader):
            self.pb.click(batch_idx, total_batch)
            image, label, name = batch['image'], batch['label'], batch['label_name']
            image_cuda, label_cuda = image.cuda(), label.cuda()
            out = self(image_cuda)

            for i in range(self.params.test_batch):
                idx = batch_idx*self.params.test_batch+i
                id_map = logits2trainId(out[i, ...])
                color_map = trainId2color(self.params.dataset_root, id_map, name=name[i])
                trainId2LabelId(self.params.dataset_root, id_map, name=name[i])
                image_orig = image[i].numpy().transpose(1, 2, 0)
                image_orig = image_orig*255
                image_orig = image_orig.astype(np.uint8)
                if self.params.summary:
                    self.summary_writer.add_image('test/img_%d/orig' % idx, image_orig, idx)
                    self.summary_writer.add_image('test/img_%d/seg' % idx, color_map, idx)

    """##########################"""
    """# Model Save and Restore #"""
    """##########################"""

    def save_checkpoint(self):
        save_dict = {'epoch'        :  self.epoch,
                     'train_loss'   :  self.train_loss,
                     'val_loss'     :  self.val_loss,
                     'state_dict'   :  self.state_dict(),
                     'optimizer'    :  self.opt.state_dict()}
        torch.save(save_dict, self.params.ckpt_dir+'Checkpoint_epoch_%d.pth.tar' % self.epoch)
        LOG('Checkpoint saved')

    def load_checkpoint(self):
        """
        Load checkpoint from given path
        """
        if self.params.resume_from is not None and os.path.exists(self.params.resume_from):
            try:
                LOG('Loading Checkpoint at %s' % self.params.resume_from)
                ckpt = torch.load(self.params.resume_from)
                self.epoch = ckpt['epoch']
                try:
                    self.train_loss = ckpt['train_loss']
                    self.val_loss = ckpt['val_loss']
                except:
                    self.train_loss = []
                    self.val_loss = []
                self.load_state_dict(ckpt['state_dict'])
                self.opt.load_state_dict(ckpt['optimizer'])
                LOG('Checkpoint Loaded!')
                LOG('Current Epoch: %d' % self.epoch)
                self.ckpt_flag = True
            except:
                WARNING('Cannot load checkpoint from %s. Start loading pre-trained model......' % self.params.resume_from)
        else:
            WARNING('Checkpoint do not exists. Start loading pre-trained model......')

    def load_model(self):
        """
        Load ImageNet pre-trained model into MobileNetv2 backbone, only happen when
            no checkpoint is loaded
        """
        if self.ckpt_flag:
            LOG('Skip Loading Pre-trained Model......')
        else:
            if self.params.pre_trained_from is not None and os.path.exists(self.params.pre_trained_from):
                try:
                    LOG('Loading Pre-trained Model at %s' % self.params.pre_trained_from)
                    pretrain = torch.load(self.params.pre_trained_from)
                    self.load_state_dict(pretrain)
                    LOG('Pre-trained Model Loaded!\n')
                except:
                    WARNING('Cannot load pre-trained model. Start training......\n')
            else:
                WARNING('Pre-trained model do not exits. Start training......\n')

    """#############"""
    """# Utilities #"""
    """#############"""

    def initialize(self):
        """
        Initializes the model parameters
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if isinstance(self.backbone, list):
            self.backbone_params = []
            for m in self.backbone:
                self.backbone_params.extend(list(m.parameters()))
        else:
            self.backbone_params = self.backbone.parameters

    def adjust_lr(self):
        """
        Adjust learning rate at each epoch
        """
        learning_rate = self.params.base_lr * (1 - float(self.epoch) / self.params.num_epoch) ** self.params.power
        for param_group in self.opt.param_groups:
            param_group['lr'] = learning_rate
        print('Change learning rate into %f' % (learning_rate))
        if self.params.summary:
            self.summary_writer.add_scalar('learning_rate', learning_rate, self.epoch)

    def plot_curve(self):
        """
        Plot train/val loss curve
        """
        x1 = np.arange(self.init_epoch, self.params.num_epoch+1, dtype=np.int).tolist()
        x2 = np.linspace(self.init_epoch, self.epoch,
                         num=(self.epoch-self.init_epoch)//self.params.val_every+1, dtype=np.int64)
        plt.plot(x1, self.train_loss, label='train_loss')
        plt.plot(x2, self.val_loss, label='val_loss')
        plt.legend(loc='best')
        plt.title('Train/Val loss')
        plt.grid()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    def build_network(self, backbone=None, head=None):
        """
        Build up network depend on backbone and head, default model is ResNet18+ASPP
        """
        if backbone is None:
            if self.backbone is None:
                self.backbone = ResNet18(self.params)
        else:
            self.backbone = backbone

        if head is None:
            if self.head is None:
                self.head = ASPP(self.params)
        else:
            self.head = head
        self.cuda()

        self.initialize()

    def build_dataloader(self):
        self.train_loader = DataLoader(self.datasets['train'],
                                       batch_size=self.params.train_batch,
                                       shuffle=self.params.shuffle,
                                       num_workers=self.params.dataloader_workers)
        self.test_loader = DataLoader(self.datasets['test'],
                                      batch_size=self.params.test_batch,
                                      shuffle=False, num_workers=self.params.dataloader_workers)
        self.val_loader = DataLoader(self.datasets['val'],
                                     batch_size=self.params.val_batch,
                                     shuffle=self.params.shuffle,
                                     num_workers=self.params.dataloader_workers)

""" TEST """
if __name__ == '__main__':
    import utils.functions as fn
    from config import Params
    params = Params()
    params.dataset_root = '/media/ubuntu/disk/cityscapes'
    LOG('Creating Dataset and Transformation......')
    datasets = fn.create_datasets(params)
    LOG('Creation Succeed.\n')
    net = MyNetwork(params, datasets)
    net.Train()