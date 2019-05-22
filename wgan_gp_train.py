import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from torch import autograd

import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb

from mydataset2 import customDataset
from wgan_gp import *

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=9, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=144, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--dataroot", type=str, default='selected_cartoonset100k/images/', help="images")
parser.add_argument("--labelroot", type=str, default='selected_cartoonset100k/cartoon_attr.txt', help="label")
parser.add_argument("--testlabel", type=str, default='sample_test/sample_human_testing_labels.txt', help="test label")
parser.add_argument("--n_row", type=int, default=3, help="n_row")

opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

def weights_init(m):
    if isinstance(m, MyConvo2d): 
        if m.conv.weight is not None:
            if m.he_init:
                torch.nn.init.kaiming_uniform_(m.conv.weight)
            else:
                torch.nn.init.xavier_uniform_(m.conv.weight)
        if m.conv.bias is not None:
            torch.nn.init.constant_(m.conv.bias, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.0)

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

def calc_gradient_penalty(netD, real_data, fake_data,bs):
    alpha = torch.rand(bs, 1)
    alpha = alpha.expand(bs, int(real_data.nelement()/bs)).contiguous()
    alpha = alpha.view(bs, 3, opt.img_size, opt.img_size)
    alpha = alpha.to(device)

    fake_data = fake_data.view(bs, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)   

    disc_interpolates, _ = aD(interpolates)####################

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
    return gradient_penalty

def sample_image(n_row, batches_done,batch_size=opt.batch_size):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    #z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for num in range(n_row*n_row)])
    #labels = Variable(LongTensor(labels))#
    noise = gen_rand_noise_with_label(labels,batch_size)#
    with torch.no_grad():
            noisev = noise
    gen_imgs = aG(noisev)
    gen_imgs = gen_imgs.view(-1, 3, opt.img_size, opt.img_size)
    save_image(gen_imgs.data, "images/%d.png" % batches_done, nrow=n_row, normalize=True)

test_lab = []
ftest = open(opt.testlabel, "r")
testlines = ftest.readlines()
test_len = int(testlines[0])
for tcount in range(test_len):
    onehots = list(map(lambda x:int(x),testlines[tcount+2].split()[0:]))            
    test_lab.append(24*np.argmax(onehots[0:6])+6*np.argmax(onehots[6:10])+2*np.argmax(onehots[10:13])+np.argmax(onehots[13:15]))
#test_lab = torch.cuda.LongTensor(test_lab)

def gen_rand_noise_with_label(label=None,batch_size=opt.batch_size):
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

fixed_label = []
for c in range(opt.batch_size):
    fixed_label.append(c%opt.n_classes)
fixed_noise = gen_rand_noise_with_label(fixed_label,opt.batch_size)


aG = GoodGenerator(128,128*128*3)
aD = GoodDiscriminator(128, opt.n_classes)

aG.apply(weights_init)
aD.apply(weights_init)

if os.path.isfile('./generator.pkl'):
    aG.load_state_dict(torch.load('./generator.pkl'))
if os.path.isfile('./discriminator.pkl'):
    aD.load_state_dict(torch.load('./discriminator.pkl'))

optimizer_G = torch.optim.Adam(aG.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(aD.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

auxiliary_loss = torch.nn.CrossEntropyLoss()

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

one = torch.FloatTensor([1])
mone = one * -1
aG = aG.to(device)
aD = aD.to(device)
one = one.to(device)
mone = mone.to(device)


for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]
    
        #---------------------TRAIN G------------------------
        for p in aD.parameters():
            p.requires_grad_(False)  # freeze D

        optimizer_G.zero_grad()
        
        gen_labels = np.random.randint(0, opt.n_classes, batch_size)
        noise = gen_rand_noise_with_label(gen_labels,batch_size)
        noise.requires_grad_(True)
        fake_data = aG(noise)
        gen_cost, gen_aux_output = aD(fake_data)
        #pdb.set_trace()
        aux_label = torch.from_numpy(gen_labels).long()
        aux_label = aux_label.to(device)
        aux_errG = auxiliary_loss(gen_aux_output, aux_label).mean()
        gen_cost = -gen_cost.mean()
        g_cost = aux_errG + gen_cost
        g_cost.backward()
        optimizer_G.step()

        #---------------------TRAIN D------------------------
        for p in aD.parameters():  # reset requires_grad
            p.requires_grad_(True)  # they are set to False below in training G

        aD.zero_grad()
        gen_labels = np.random.randint(0, opt.n_classes, batch_size)
        noise = gen_rand_noise_with_label(gen_labels,batch_size)
        with torch.no_grad():
                noisev = noise  # totally freeze G, training D
        fake_data = aG(noisev).detach()
        real_data = Variable(imgs.type(FloatTensor))
        real_label = Variable(labels.type(LongTensor))
        real_data = real_data.to(device)
        real_label = real_label.to(device)

        disc_real, aux_output = aD(real_data)
        #pdb.set_trace()
        aux_errD_real = auxiliary_loss(aux_output, real_label)
        errD_real = aux_errD_real.mean()
        disc_real = disc_real.mean()

        disc_fake, fake_aux = aD(fake_data)
        disc_fake = disc_fake.mean()

        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data,batch_size)

        disc_cost = disc_fake - disc_real + gradient_penalty
        disc_acgan = errD_real

        d_loss = disc_cost + disc_acgan
        d_loss.backward()

        optimizer_D.step()

        # Calculate discriminator accuracy
        pred = np.concatenate([aux_output.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([real_label.data.cpu().numpy(), gen_labels], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %d%%] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), 100 * d_acc, g_cost.item())
        )


        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=opt.n_row, batches_done=batches_done,batch_size=batch_size)
            torch.save(aG.state_dict(),'./generator.pkl')
            torch.save(aD.state_dict(),'./discriminator.pkl')










