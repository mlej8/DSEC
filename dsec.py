import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os
from params import *
"""
@authors: Michael Li and William Zhang
@email: er.li@mail.mcgill.ca 
@email: william.zhang2@mail.mcgill.ca 
@description: This file contains an implementation of Deep Self-Evolution Clustering algorithm in PyTorch
"""

########################
### Helper functions ###
########################

def compute_loss(patterns, D):
    """ Objective function of DSEC """
    loss = 0
    for key in D:
        # minimizing loss using equation (12)
        loss += torch.pow(torch.linalg.norm(D[key] - torch.dot(patterns[key[0]], patterns[key[1]]), ord=2, dim=0), 2)
    return loss 

def construct_dataset(patterns):
    """ 
    Given an unlabeled dataset and the predefined number of clusters k, where x_i indicates the ith pattern, DSEC manages the clustering task by investigating similarities between pairwise patterns.
    To consider similarities, we transform the dataset into a binary pairwise classification problem.
    We turn a dataset into D = {{x_i, x_j, r_ij}} where x_i and x_j are unlabeled patterns (inputs) and r_ij is a binary variable which says x_i and x_j belong to the same cluster or not. 
    """
    size = len(patterns)
    D = dict()
    for i in range(size):
        for j in range(size):
            label = labeled_pairwise_patterns_selection(patterns[i], patterns[j])
            if label is not None: 
                D[(i,j)] =  label
    return D

def labeled_pairwise_patterns_selection(indicator_feature1, indicator_feature2):
    """ Implementation of the pairwise labelling algorithm described in the paper. """
    # determine similarity between two labels to create label
    similarity = torch.nn.CosineSimilarity(dim=0)(indicator_feature1, indicator_feature2)
    if similarity > u:
        return 1
    elif similarity <= l:
        return 0
    else:
        return None # similarity between x_i and x_j is ambiguous, thus this pair will be omitted during training

def cp_constraint(indicator_features):
    """ Implementation of equation (11): c_p constraint """
    
    # get the max of each indicator feature
    maximum = torch.max(indicator_features, 1)[0].reshape(len(indicator_features), -1)

    # expand the dimensions so that we can substract the indicator features
    maximum = maximum.expand(-1, len(indicator_features[0]))
    
    # compute intermediate 
    intermediate = torch.exp(indicator_features - maximum)

    # compute ||I^tem||_p 
    intermediate_norm = torch.linalg.norm(intermediate, ord=p, dim=1).reshape(len(indicator_features), -1)

    return torch.div(intermediate, intermediate_norm)

def weights_init(layer):
    """ Initialize weights using normal initialization strategy """
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.normal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)

    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.normal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)


########################
###  DSEC algorithm  ###
########################

def dsec(dataset, dnn, model_name, initialized=False):
    """ Takes as input a PyTorch dataset and a DNN model """
    
    # see if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # initial num of clusters
    num_clusters = len(dataset.classes) 
    
    # we are modifying global variable
    global u
    global l

    if not initialized:
        # initialize weights using normalized Gaussian initialization strategy 
        for key in dnn._modules:
            weights_init(dnn._modules[key])

    # look if running multiple GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Using {} GPUs".format(num_gpus))
        dnn = nn.DataParallel(dnn)
    
    # move network to appropriate device
    dnn.to(device)

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    # define optimizer
    optimizer = torch.optim.RMSprop(dnn.parameters(),lr=optimizer_lr)

    # time tracker
    start_time = datetime.now()

    # set epoch count 
    epoch = 1

    while u >= l:

        # tracking total loss
        total_loss = 0.0

        for iteration, (data,labels) in enumerate(dataloader):

            # clear the parameter gradients
            optimizer.zero_grad()

            # send data to correct device
            data = data.to(device)

            # initialize indicator features (forward pass)
            output = dnn(data)

            # construct pairwise similarities dataset, select training data from batch using formula (8), e.g. define labels for each indicator feature pair
            D = construct_dataset(output) # NOTE: DNN is shared 

            # compute loss
            loss = compute_loss(output, D)

            # accumulate loss
            total_loss += loss

            # backward pass to get all gradients
            loss.backward()

            # weights are updated on each batch that is gradually sampled from the original dataset
            optimizer.step()

            # at every 500 mini-batch, print the loss
            if iteration % 500 == 499: 
                print('Average batch loss: {}\tIteration: {}'.format(total_loss/500, iteration+1))
                total_loss = 0.0

        end_time = datetime.now()
        print("Epoch {}: u ({}) and l ({}) in {}".format(epoch,u,l, end_time - start_time))
        start_time = end_time
        epoch += 1
        
        # update u and l: s(u,l) = u - l
        u = u - lr
        l = l + lr

    # save model and create the models directory if not exist
    PATH =  './models/{0}-{1}.pth'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists('./models'):
        os.makedirs("models")
    
    # save model state dict 
    try:
        state_dict = dnn.module.state_dict()
    except AttributeError:
        state_dict = dnn.state_dict()
    torch.save(state_dict, PATH)

    # returning the model path
    return PATH