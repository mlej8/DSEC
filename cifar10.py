import torch
import torchvision
import torchvision.transforms as transforms

"""
2. Define a Convolutional Neural Network

3. Define a loss function

4. Train the network on the training data

5. Test the network on the test data

6. Loading and normalizing CIFAR10
""" 


# The output of torchvision datasets are PILImage images of range [0, 1]. We transform them to Tensors of normalized range [-1, 1].
transform = transforms.Compose([transforms.ToTensor(), # transform to tensor
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalizing the CIFAR10 training and test datasets using torchvision
                                ])

# Load the CIFAR10 training and test datasets using torchvision
trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True, download=True)
# trainset = torchvision.datasets.CIFAR10(root='./cifar10data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)

# Load test set using torchvision
testset = torchvision.datasets.CIFAR10(root='./cifar10data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=2)

# Classes of CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
