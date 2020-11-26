from dsec import *

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors and normalize range to [-1, 1]. QUESTION: why?
transform = transforms.Compose([transforms.ToTensor(), # transform to tensor
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize range to [-1, 1]
                                ])

# Load the CIFAR10 training and test datasets using torchvision
trainset = torchvision.datasets.CIFAR100(root='./cifar100data', train=True, download=True, transform=transform)

# Load test set using torchvision
testset = torchvision.datasets.CIFAR100(root='./cifar100data', train=False,download=True, transform=transform)

###########
### DNN ###
###########
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # TODO add batch norm to each layer
        # TODO Gaussian noise layer
        # self.gaussian_noise = 
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
        self.conv7 = nn.Conv2d(128, 20, 1) 
        self.bn3 = nn.BatchNorm2d(20)
        self.global_averaging = nn.AvgPool2d(3) # global averaging
        self.fc1 = nn.Linear(in_features=20, out_features=20)
        self.bn4 = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(in_features=20, out_features=len(trainset.classes))
        self.bn5 = nn.BatchNorm1d(100)
        self.constraint_layer = cp_constraint

    def forward(self, x, p):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn1(x))
        x = self.conv3(x)
        x = F.relu(self.bn1(x))
        x = self.bn1(self.pooling1(x))
        x = F.relu(self.bn2(self.conv4(x)))
        x = F.relu(self.bn2(self.conv5(x)))
        x = F.relu(self.bn2(self.conv6(x)))
        x = self.pooling2(x)
        x = F.relu(self.bn3(self.conv7(x)))
        x = self.bn3(self.global_averaging(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        output = self.constraint_layer(x,p)
        return output

model_path = dsec(trainset, Net(), "cifar100")
nmi(trainset, Net(), model_path)
# nmi(trainset, Net(), "models/cifar100-2020-Nov-25-03-48-11.pth")