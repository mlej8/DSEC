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

def compute_loss(label, prediction):
    """ Compute loss for a pair of patterns. """
    # implementation of equation (2)
    return torch.pow(torch.linalg.norm(label - prediction, ord=2, dim=0), 2)

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
    
    # log which model we are running
    print(f"Running DSEC on {model_name}")

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

    # set dnn in training mode
    dnn.train()

    # create folder for storing weights if it does not exist
    weights_folder = './models/{}/{}'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S")) 
    if not os.path.exists(weights_folder):
        os.makedirs(weights_folder)

    while u >= l:

        # tracking total loss
        total_loss = 0.0
        batch_num = 1

        for data,labels in dataloader:

            # clear the parameter gradients
            optimizer.zero_grad()

            # send data to correct device
            data = data.to(device)

            # initialize indicator features (forward pass)
            output = dnn(data)

            # create a matrix of dot products
            predictions = torch.mm(output, torch.transpose(output, 0, 1))

            # select training data from batch using eq. 8: construct pairwise similarities matrix
            norms = torch.linalg.norm(output, ord=2, dim=1).reshape(len(output), -1)
            norms_mm = torch.mm(norms, torch.transpose(norms, 0, 1))
            similarity_matrix = torch.div(predictions,norms_mm)
            
            # minimizing loss using equation (12)
            for i in range(len(similarity_matrix)):
                for j in range(len(similarity_matrix)):
                    
                    # get similarity for pattern pair
                    similarity = similarity_matrix[i][j]

                    # pairwise labelling 
                    if similarity > u:
                        total_loss += compute_loss(1, predictions[i][j])
                    elif similarity <= l:
                        total_loss += compute_loss(0, predictions[i][j])
                    
            # backward pass to get all gradients
            total_loss.backward()

            # update weights
            optimizer.step()
            
            print('Batch: {}\tLoss: {}'.format(batch_num, total_loss/batch_size))
            total_loss = 0.0
            batch_num += 1

        end_time = datetime.now()
        print("Epoch {}: u ({}) and l ({}) done in {}".format(epoch,u,l, end_time - start_time))
        start_time = end_time

        # save weights
        PATH = weights_folder + "/epoch" + str(epoch) +  ".pth"
        try:
            state_dict = dnn.module.state_dict()
        except AttributeError:
            state_dict = dnn.state_dict()
        torch.save(state_dict, PATH)
        
        # update u and l: s(u,l) = u - l
        epoch += 1
        u = u - lr
        l = l + lr

    # save model and create the models directory if not exist
    PATH =  weights_folder + '/{0}.pth'.format(model_name)
    
    # save model state dict 
    try:
        state_dict = dnn.module.state_dict()
    except AttributeError:
        state_dict = dnn.state_dict()
    torch.save(state_dict, PATH)

    # returning the model path
    return PATH