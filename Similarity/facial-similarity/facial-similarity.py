import random
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import argparse


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        self.parser.add_argument('-train_dir', required=True, default='./',
                                 help='path to the training data directory')
        self.parser.add_argument('-test_dir', required=True, default='./',
                                 help='path to the test data directory')
        self.parser.add_argument('--load_size', type=int, default=144,
                                 help='scale image to the size prepared for croping')
        self.parser.add_argument('--input_size', type=int, default=128,
                                 help='then crop image to the size as network input')
        self.parser.add_argument('--ratio', type=str, default='[0.95, 0.025, 0.025]',
                                 help='ratio of whole dataset for Train, Validate, Test resperctively')
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='batch size of network input. Note that batch_size sfpnhould only set to 1 in '
                                      'Test mode')
        self.parser.add_argument('--shuffle', action='store_true',
                                 help='default false. If true, data will be shuffled when split dataset and in batch')
        self.parser.add_argument('--gray', action='store_true',
                                 help='defalut false. If true, image will be converted to gray_scale')
        self.parser.add_argument('-gpu', type=int, default=-1,
                                 help='gpu: e.g. 0  1. use -1 for CPU')
        self.parser.add_argument('--box_ratio', type=float, default=-1,
                                 help='modify box ratio of width and height to specified ratio')
        self.parser.add_argument('--box_scale', type=float, default=1.0,
                                 help='scale box to specified ratio. Default 1.0 means no change')
        self.parser.add_argument('--input_channel', type=int, default=3,
                                 help='set input image channel, 1 for gray and 3 for color')
        self.parser.add_argument('--mean', type=str, default='(0,0,0)',
                                 help='sequence of means for each channel used for normization')
        self.parser.add_argument('--std', type=str, default='(1,1,1)',
                                 help='sequence standard deviations for each channel used for normization')
        self.parser.add_argument('--padding', action='store_true',
                                 help='default false. If true, image will be padded if scaled box is out of image '
                                      'boundary')
        self.parser.add_argument('--checkpoint_name', type=str, default='',
                                 help='path to pretrained model or model to deploy')
        self.parser.add_argument('--pretrain', action='store_true',
                                 help='default false. If true, load pretrained model to initizaize model state_dict')
        ## for train
        self.parser.add_argument('--validate_ratio', type=float, default=1,
                                 help='ratio of validate set when validate model')
        self.parser.add_argument('-epochs', type=int, default=20, help='sum epoches for training')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='save snapshot every $save_epoch_freq epoches training')
        self.parser.add_argument('--save_batch_iter_freq', type=int, default=100,
                                 help='save snapshot every $save_batch_iter_freq training')
        self.parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1,
                                 help='multiplicative factor of learning rate decay.')
        self.parser.add_argument('--lr_mult_w', type=float, default=20,
                                 help='learning rate of W of last layer parameter will be lr*lr_mult_w')
        self.parser.add_argument('--lr_mult_b', type=float, default=20,
                                 help='learning rate of b of last layer parameter will be lr*lr_mult_b')
        self.parser.add_argument('--lr_policy', type=str, default='step',
                                 help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_in_epoch', type=int, default=50,
                                 help='multiply by a gamma every lr_decay_in_epoch iterations')
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum of SGD')
        self.parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay of SGD')
        self.parser.add_argument('--loss_weight', type=str, default='',
                                 help='list. Loss weight for cross entropy loss.For example set $loss_weight to [1, '
                                      '0.8, 0.8] for a 3 labels classification')

        ## for test
        self.parser.add_argument('--classify_dir', type=str, default="",
                                 help='directory where data.txt to be classified exists')

    def parse(self):
        opt = self.parser.parse_args()
        return opt


op = Options()
opt = op.parse()
device = 'cuda:{}'.format(opt.gpu) if opt.gpu > -1 else 'cpu'


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        # img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0_tuple = self.imageFolderDataset.imgs[index]
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5)
        )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def train(model, train_dataloader, opt):
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0, opt.epochs):
        els = []
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            output1, output2 = model(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % 100 == 0:
                print("{} Epoch [{}/{}] << Current loss {}".format(epoch, i, len(train_dataloader), loss_contrastive.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
                els += [loss_contrastive.item()]
        print("\n {} Epoch's mean loss : {}".format(epoch, np.mean(els)))
    # show_plot(counter, loss_history)
    # print("LOGGG  ")
    # print(len(counter))
    # print(len(loss_history))


def evaluate(model, test_dataloader, opt):
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        _, x1, label2 = next(dataiter)
        # concatenated = torch.cat((x0, x1), 0)
        x0, x1 = x0.to(device), x1.to(device)
        output1, output2 = model(x0, x1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        # imshow(torchvision.utils.make_grid(concatenated),
        #       'Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))
        print('Dissimilarity: {:.2f}'.format(euclidean_distance.cpu().data.numpy()[0][0]))


if __name__ == "__main__":
    folder_dataset_train = dset.ImageFolder(root=opt.train_dir)
    siamese_dataset_train = SiameseNetworkDataset(imageFolderDataset=folder_dataset_train,
                                                  transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                                transforms.ToTensor()
                                                                                ])
                                                  , should_invert=False)

    train_dataloader = DataLoader(siamese_dataset_train, batch_size=1, shuffle=True)

    folder_dataset_test = dset.ImageFolder(root=opt.test_dir)
    siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                                 transform=transforms.Compose([transforms.Scale((100, 100)),
                                                                               transforms.ToTensor()
                                                                               ])
                                                 , should_invert=False)

    test_dataloader = DataLoader(siamese_dataset_test, batch_size=1, shuffle=True)

    model = SiameseNetwork().to(device)
    train(model, train_dataloader, opt)
    evaluate(model, test_dataloader, opt)
