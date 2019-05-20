from torch.utils.data.dataset import Dataset
import os
#import cv2
from PIL import Image
import torch
import numpy as np
import pdb

class customDataset(Dataset):
    def __init__(self,root,lab_root,transform=None):
        ##############################################
        ### Initialize paths, transforms, and so on
        ##############################################
        self.root = root
        self.lab_root = lab_root
        self.transform = transform
        self.fig = []
        self.label = []
        self.len = 0
        
        fp = open(self.lab_root, "r")
        lines = fp.readlines()
        self.len = int(lines[0])
        for i in range(self.len):
            self.fig.append(lines[i+2].split()[0])
            onehots = list(map(lambda x:int(x),lines[i+2].split()[1:]))            
            self.label.append(24*np.argmax(onehots[0:6])+6*np.argmax(onehots[6:10])+2*np.argmax(onehots[10:13])+np.argmax(onehots[13:15]))
        self.label = torch.tensor(self.label)
        
        
    def __getitem__(self, index):
        ##############################################
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform).
        # 3. Return the data (e.g. image and label)
        ##############################################
        img = Image.open(self.root + "/" + self.fig[index]).convert('RGB')
        lab = self.label[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, lab
        
    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return self.len

