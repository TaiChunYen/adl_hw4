import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

#import torchvision.datasets as dset
from mydataset import customDataset

os.makedirs("fid_pic", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=144, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--testlabel", type=str, default='sample_test/sample_fid_testing_labels.txt', help="test label")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.latent_dim)

        self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

generator = Generator()
generator.load_state_dict(torch.load('./generator.pkl'))
if cuda:
    generator.cuda()
generator.eval()

test_lab = []
ftest = open(opt.testlabel, "r")
testlines = ftest.readlines()
test_len = int(testlines[0])
for tcount in range(test_len):
    onehots = list(map(lambda x:int(x),testlines[tcount+2].split()[0:]))            
    test_lab.append(24*np.argmax(onehots[0:6])+6*np.argmax(onehots[6:10])+2*np.argmax(onehots[10:13])+np.argmax(onehots[13:15]))
test_lab = torch.cuda.LongTensor(test_lab)

for i in range(len(test_lab)):
# Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))))
# Get labels ranging from 0 to n_classes for n rows
#labels = np.array([num for _ in range(n_row) for num in range(n_row)])#
    labels = test_lab[i]
    #labels = Variable(FloatTensor(labels).view(1,-1))#
    gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, "fid_pic/%d.png" % i, normalize=True)#nrow=n_row
