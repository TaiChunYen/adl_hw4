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
parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
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

        #self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)#

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((labels, noise), -1)#self.label_emb(labels)
        #pdb.set_trace()
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
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
    test_lab.append(list(map(lambda x:int(x),testlines[tcount+2].split()[0:])))
test_lab = torch.cuda.FloatTensor(test_lab)

for i in range(len(test_lab)):
# Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))))#n_row ** 2
# Get labels ranging from 0 to n_classes for n rows
#labels = np.array([num for _ in range(n_row) for num in range(n_row)])#
    labels = test_lab[i]
    labels = Variable(FloatTensor(labels).view(1,-1))#
    gen_imgs = generator(z, labels)

    save_image(gen_imgs.data, "fid_pic/%d.png" % i, normalize=True)#nrow=n_row
