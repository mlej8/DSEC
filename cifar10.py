import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime

import torchvision
import torchvision.transforms as transforms

# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors and normalize range to [-1, 1]. QUESTION: why?
transform = transforms.Compose([transforms.ToTensor(), # transform to tensor
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize range to [-1, 1]
                                ])

# 1. Load the CIFAR10 training and test datasets using torchvision
trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

# Load test set using torchvision
testset = torchvision.datasets.CIFAR10(root='./cifar10data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

# Classes of CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()

# 2. Define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# create the DCNN
net = Net()

# 3. Define a loss function
criterion = nn.CrossEntryLoss() # TODO: replace this with custom loss

# 4. Define an optimizer used to update the weights
optimizer = optim.SGD(net.parameters, lr=0.001, momentum)

# 5. Train the network
num_epochs = 1
for epoch_i in range(num_epochs):
    
    # tracking total loss
    total_loss = 0.0
    
    for iteration, data in (trainloader): # loop over data iterator and feed the inputs to the netwrok and optimize
        # get the inputs 
        inputs, labels = data

        # clear the parameter gradients
        optimizer.zero_grad()

        # forward pass 
        outputs = net(inputs)

        # get loss 
        loss = criterion(outputs, labels)

        # backward pass to get all gradients
        loss.backward()

        # update params
        optimizer.step()

        # total loss
        total_loss += loss.item()

        # at every 2000 mini-batch, print the loss
        if iteration % 500 == 499: 
            print('Epoch: {}\tLoss: {}\tIteration: {}'.format(epoch_i, total_loss/500, iteration))
            
            total_loss = 0.0

# done
print("Finished Training")