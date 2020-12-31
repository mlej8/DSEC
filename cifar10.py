from dsec import dsec, cp_constraint
from unsupervised_metrics import cluster

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from custom_dataset import Dataset

# Load the CIFAR10 training and test datasets using torchvision
trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True, download=True, transform=None)

# Load test set using torchvision
testset = torchvision.datasets.CIFAR10(root='./cifar10data', train=False,download=True, transform=None)

# computing the mean and variance of each channel
data = np.concatenate((trainset.data, testset.data), axis=0)
labels = np.concatenate((trainset.targets, testset.targets), axis=0)
means = (np.mean(data[:,:,:,0]),np.mean(data[:,:,:,1]),np.mean(data[:,:,:,2]))
variances = (np.var(data[:,:,:,0]),np.var(data[:,:,:,1]),np.var(data[:,:,:,2]))

"""
The output of torchvision datasets are PILImage images of range [0, 1]. 
We transform them to Tensors and normalize range to [-1, 1].
However, we concatenated both train and test sets into one single dataset. 
Therefore, we find the mean and variance for each channel and normalize. 
"""
transform = transforms.Compose([transforms.ToTensor(), # transform to tensor
                                transforms.Normalize(means,variances) # normalize range to [-1, 1]
                                ])

cifar10 = Dataset(data, labels, trainset.classes, transform)

###########
### DNN ###
###########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.pooling1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 3) 
        self.bn2 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.pooling2 = nn.MaxPool2d(2)
        self.conv7 = nn.Conv2d(128, 10, 1) 
        self.bn3 = nn.BatchNorm2d(10)
        self.global_averaging = nn.AvgPool2d(3) # global averaging
        self.fc1 = nn.Linear(in_features=10, out_features=10)
        self.bn4 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(in_features=10, out_features=len(trainset.classes))
        self.constraint_layer = cp_constraint

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = self.bn1(self.pooling1(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn2(self.conv5(x)))
        x = F.relu(self.bn2(self.conv6(x)))
        x = self.bn2(self.pooling2(x))
        x = F.relu(self.bn3(self.conv7(x)))
        x = self.bn3(self.global_averaging(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn4(self.fc2(x)))
        output = self.constraint_layer(x)
        return output

model_name = "cifar10"
model_path = "models/cifar10-Nov-28-03-06-27.pth"
model_path = dsec(cifar10, Net(), model_name=model_name)
cluster(cifar10, Net(), model_path , model_name=model_name)