from __future__ import print_function
import os
import numpy as np

from dsec import dsec, cp_constraint
from pretrain import pretrain
from unsupervised_metrics import cluster, pretrain_cluster

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from data_augmentation import GaussianNoise

from custom_dataset import Dataset

# Code to download MNST, uncomment the following code if the dataset is unable to download
# from six.moves import urllib
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)

# Load the MNIST training and test datasets using torchvision
trainset = torchvision.datasets.MNIST(root='./mnistdata', train=True, download=True, transform=None)

# Load test set using torchvision
testset = torchvision.datasets.MNIST(root='./mnistdata', train=False, download=True, transform=None)

# computing the mean and variance of each channel
data = np.concatenate((trainset.data, testset.data), axis=0)
labels = np.concatenate((trainset.targets, testset.targets), axis=0)
# data = data.astype('float32')
d = data/255.0
means = (np.mean(d[:,:,:]),)
stds = (np.std(d[:,:,:]),)

# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors and normalize using mean and std of the entire dataset. 
transform = transforms.Compose([transforms.ToTensor(), # transform to tensor
                                # transforms.Normalize(means, stds),

                                # data augmentation
                                GaussianNoise(mean=0.0,std=0.001), # gaussian noise
                                transforms.RandomRotation(5), # Random rotation from [-30,30]
                                transforms.RandomAffine(degrees=0, scale=(0.9,1.1), translate=(0.18,0.18)),
                                transforms.RandomHorizontalFlip(p=0.5)
                                # TODO: Randomly shift the channel in [0.9, 1.1]
                                # TODO: Randomly zoom the image in [0.85, 1.15]
                                ])

DA_transform = transforms.Compose([
                                # data augmentation
                                transforms.ToPILImage(mode=None),
                                transforms.RandomAffine(degrees=20, scale=(0.85,1.15), translate=(0.18,0.18), fillcolor=0),
                                transforms.ToTensor(), # transform to tensor
                                # transforms.Normalize(means, stds)
                                ])

simple_transform = transforms.Compose([transforms.ToTensor()])

mnist = Dataset(data, labels, trainset.classes, DA_transform)

non_augmented_mnist = Dataset(data, labels, trainset.classes, simple_transform)


###########
### DNN ###
###########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)

        self.pooling1 = nn.MaxPool2d(2)
        
        self.conv4 = nn.Conv2d(64, 128, 3) 
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 128, 3)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.bn6 = nn.BatchNorm2d(128)

        self.pooling2 = nn.MaxPool2d(2)

        self.conv7 = nn.Conv2d(128, 10, 1) 
        self.bn7 = nn.BatchNorm2d(10)

        self.global_averaging = nn.AvgPool2d(2) # global averaging

        self.fc1 = nn.Linear(in_features=10, out_features=10)
        self.bn8 = nn.BatchNorm1d(10)
        self.fc2 = nn.Linear(in_features=10, out_features=len(trainset.classes))
        self.bn9 = nn.BatchNorm1d(10)

        self.constraint_layer = cp_constraint

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pooling1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))

        x = self.pooling2(x)

        x = F.relu(self.bn7(self.conv7(x)))
        
        x = self.global_averaging(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.bn8(self.fc1(x)))
        x = F.relu(self.bn9(self.fc2(x)))

        output = self.constraint_layer(x)
        return output


"""
Simple Net
"""
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(in_features=(1*28*28), out_features=512)
        self.linear2 = nn.Linear(in_features=512, out_features=256)
        self.batchNorm = nn.BatchNorm1d(256,eps=0.001, momentum=0.99, affine=True, track_running_stats=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        output = self.batchNorm(x)
        return output


model_name = "mnist_dsec"
# model_path = './models/mnist_dsec/Mar-29-21-28-37/pretrain/epoch10.pth'
model_path = pretrain(mnist, non_augmented_mnist, Net(), model_name=model_name, initialized=False, pretrained_model=None)
pretrain_cluster(non_augmented_mnist, Net(), model_path, model_name=model_name)
model_path = dsec(mnist, Net(), model_name=model_name, initialized=True, pretrained_model=model_path)
cluster(non_augmented_mnist, Net(), model_path, model_name=model_name)

