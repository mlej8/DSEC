import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # optimizers

"""
@authors: Michael Li and William Zhang
@email: er.li@mail.mcgill.ca 
@email: william.zhang2@mail.mcgill.ca 
@description: This file contains an implementation of Deep Self-Evolution Clustering algorithm in PyTorch
"""

### Helper functions

def labeled_pairwise_patterns_selection(indicator_feature1, indicator_feature2, u, l):
    """ Implementation of the pairwise labelling algorithm described in the paper. """
    if similarity_estimation(indicator_feature1, indicator_feature2) > u:
        return 1
    elif similarity_estimation(indicator_feature1, indicator_feature2) <= l:
        return 0 
    else:
        return None # similarity between x_i and x_j is ambiguous, thus this pair will be omitted during training

def similarity_estimation(indicator_feature1, indicator_feature2):
    return distance.cosine(indicator_feature1, indicator_feature2)

def dot_product(indicator_feature1, indicator_feature2):
    """ 
    DSEC measures the similarities via dot product between two indicator features  
    This is basically g(x_i, x_j; w) (5)
    """
    return np.dot(indicator_feature1, indicator_feature2)

def loss(data):
    """ Objective function of DSEC """
    # n = len(data)
    # for i in range(n):
    #     for j in range(n):
    #         v_ij = 1 if == 1 or == 0 else 0 
    # loss between the binary variable r_ij and the estimated similarity g(x_i, x_j;w). Squared loss is employed
    return torch.pow(torch.linalg.norm(torch.Tensor([y_true - y_pred]), 2), 2)


### DCNN

class Net(nn.Module):
    """ Random DNN to learn indicator features """
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


### Weights update (optimizer)
optimizer = optim.SGD(net.parmaeters(), lr=0.01)

# in my training loop
optimizer.zero_grad() # clear gradient buffers (set to zero), because gradients are accumulated
output = net(input)
loss = loss(output, target)
loss.backward() 
optimizer.step() # does the update

# NOTE: how can we apply constraint at (3) to DNN ? 
# Step 1: Use a random DNN to find indicator features for each pattern
# Step 2: Define labels for each indicator feature pair
# NOTE: DNN is shared 
# Last Step: Use true label to report predicted labels to get ACC.