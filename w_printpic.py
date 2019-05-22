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
from wgan_gp import *

os.makedirs("fid_pic", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=144, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--testlabel", type=str, default='sample_test/sample_fid_testing_labels.txt', help="test label")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda_available = torch.cuda.is_available()
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda" if cuda_available else "cpu")


FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def gen_rand_noise_with_label(label=None,batch_size=1):
    if label is None:
        label = np.random.randint(0, opt.n_classes, batch_size)
    #attach label into noise
    noise = np.random.normal(0, 1, (batch_size, 256))
    prefix = np.zeros((batch_size, opt.n_classes))
    prefix[np.arange(batch_size), label] = 1
    noise[np.arange(batch_size), :opt.n_classes] = prefix[np.arange(batch_size)]

    noise = torch.from_numpy(noise).float()
    noise = noise.to(device)

    return noise


generator = GoodGenerator()
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
#test_lab = torch.cuda.LongTensor(test_lab)

for i in range(len(test_lab)):
# Sample noise
    #z = Variable(FloatTensor(np.random.normal(0, 1, (1, opt.latent_dim))))
# Get labels ranging from 0 to n_classes for n rows
#labels = np.array([num for _ in range(n_row) for num in range(n_row)])#
    labels = test_lab[i]
    z=gen_rand_noise_with_label(labels,1)
    #labels = Variable(FloatTensor(labels).view(1,-1))#
    gen_imgs = generator(z)
    gen_imgs = gen_imgs.view(-1, 3, opt.img_size, opt.img_size)
    save_image(gen_imgs.data, "fid_pic/%d.png" % i, normalize=True)#nrow=n_row
