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

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=15, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--dataroot", type=str, default='selected_cartoonset100k/images/', help="images")
parser.add_argument("--labelroot", type=str, default='selected_cartoonset100k/cartoon_attr.txt', help="label")
parser.add_argument("--testlabel", type=str, default='sample_test/sample_human_testing_labels.txt', help="test label")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


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
        img = self.model(gen_input)        
        img = img.view(img.size(0), *img_shape)
        #pdb.set_trace()
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        #self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)#

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes+int(np.prod(img_shape)), 512),#
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 1),
        )
        self.adv = nn.Linear(512, 1)#
        #self.aux = nn.Linear(512, opt.n_classes)#

    def forward(self, img, labels):#
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), labels), -1)#
        out = self.model(d_in)#validity = ,d_in
        validity = self.adv(out)#
        #label = self.aux(out)#
        return validity#, label


# Loss functions
adversarial_loss = torch.nn.MSELoss()
#auxiliary_loss = torch.nn.MSELoss()#

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if os.path.isfile('./generator.pkl'):
    generator.load_state_dict(torch.load('./generator.pkl'))
if os.path.isfile('./discriminator.pkl'):
    discriminator.load_state_dict(torch.load('./discriminator.pkl'))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    #auxiliary_loss.cuda()#

# Configure data loader
'''os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)
'''

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = customDataset(root=opt.dataroot,lab_root=opt.labelroot,
                           transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.CenterCrop(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.n_cpu)
#----------------------------------------------------------------------------------------------

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done,test_lab,test_len):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (720, opt.latent_dim))))#n_row ** 2
    # Get labels ranging from 0 to n_classes for n rows
    #labels = np.array([num for _ in range(n_row) for num in range(n_row)])#
    labels = test_lab
    labels = Variable(FloatTensor(labels))
    #pdb.set_trace()
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=27, normalize=True)#nrow=n_row

test_lab = []
ftest = open(opt.testlabel, "r")
testlines = ftest.readlines()
test_len = int(testlines[0])
for tcount in range(test_len):
    test_lab.append(list(map(lambda x:int(x),testlines[tcount+2].split()[0:])))
test_lab = torch.cuda.FloatTensor(test_lab)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(FloatTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        #gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))#
        gen_labels=[]
        for gl in range(batch_size):
            gen_labels.append(np.random.randint(0, 2, opt.n_classes))
        gen_labels = Variable(FloatTensor(gen_labels))
        
        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs,gen_labels)#, pred_label
        g_loss = adversarial_loss(validity, valid)# (+ auxiliary_loss(pred_label, gen_labels)) / 2

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)#, real_aux
        d_real_loss = adversarial_loss(validity_real, valid)#( + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)#, fake_aux 
        d_fake_loss = adversarial_loss(validity_fake, fake)#( + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done, test_lab=test_lab, test_len=test_len)
            torch.save(generator.state_dict(),'./generator.pkl')
            torch.save(discriminator.state_dict(),'./discriminator.pkl')
