#!/usr/bin/env python3

import argparse
import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np

#trainset = dset.ImageFolder(os.path.join('/datadisk/A/liubaoen/data/TINYIMAGENET_1000_80/', 'train'),
#                    transform=transforms.ToTensor())
#data = torch.utils.data.DataLoader(trainset, 128,shuffle=True, num_workers=16).dataset
data = dset.CIFAR100(root='/datadisk/A/liubaoen/data/cifar', train=True, download=True,
                                 transform=transforms.ToTensor()).train_data

data = data.astype(np.float32)/255.

means = []
stdevs = []
for i in range(3):
    pixels = data[:,i,:,:].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
