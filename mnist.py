from __future__ import print_function

from dsec import c_pp

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from utils import transform

# Load the MNIST training and test datasets using torchvision
trainset = torchvision.datasets.MNIST(root='./mnistdata', train=True, download=True, transform=transform)

# Load test set using torchvision
testset = torchvision.datasets.MNIST(root='./mnistdata', train=False,download=True, transform=transform)

### NN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO Gaussian noise layer
        self.gn = 
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.pooling1 = nn.Conv2d(32, 64, 3, 1)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        self.conv5 = nn.Conv2d(32, 64, 3, 1)
        self.conv6 = nn.Conv2d(32, 64, 3, 1)
        self.pooling2 = nn.Conv2d(32, 64, 3, 1)
        self.conv7 = nn.Conv2d(32, 64, 3, 1)
        self.global_averaging_bn = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Conv2d(32, 64, 3, 1)
        self.fc2 = nn.Conv2d(32, 64, 3, 1)
        self.constraint_layer = c_p

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output