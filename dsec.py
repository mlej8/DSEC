import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import pickle
import os
from params import *
from utils import weights_init
"""
@authors: Michael Li and William Zhang
@email: er.li@mail.mcgill.ca 
@email: william.zhang2@mail.mcgill.ca 
@description: This file contains an implementation of Deep Self-Evolution Clustering algorithm in PyTorch
"""

########################
### Helper functions ###
########################

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

########################
###  DSEC algorithm  ###
########################

def dsec(dataset, dnn, model_name, initialized=False, pretrained_model=None):
    """ Takes as input a PyTorch dataset and a DNN model """
    
    # log which model we are running
    print(f"Running DSEC on {model_name}.\nStarted experiment on {datetime.now().strftime('%b-%d-%H-%M-%S')}")

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

        # create folder for storing weights if it does not exist
        weights_folder = './models/{}/{}'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S")) 
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)


    else:
        dnn.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
        path = pretrained_model.split("/")
        new_path = path[:-2] + ['train']
        weights_folder = "/".join(new_path)
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

    # look if running multiple GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print("Using {} GPUs".format(num_gpus))
        dnn = nn.DataParallel(dnn)
    
    # move network to appropriate device
    dnn.to(device)

    # labels
    one = torch.tensor(1.).to(device)
    zero = torch.tensor(0.).to(device)

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

    # create BCE loss (does not follow paper but follows their code)
    criterion = nn.BCELoss()

    while u >= l:

        # track number of batch
        batch_num = 1
        epoch_loss = 0

        for data,labels in dataloader:

            # clear the parameter gradients
            optimizer.zero_grad()

            # send data to correct device
            data = data.to(device)

            # initialize indicator features (forward pass)
            output = dnn(data)

            # create a matrix of dot products which represents the predictions
            predictions = torch.sigmoid(torch.mm(output, torch.transpose(output, 0, 1)))

            # select training data from batch using eq. 8: construct pairwise similarities matrix
            with torch.no_grad():
                norms = torch.linalg.norm(output, ord=2, dim=1).reshape(len(output), -1)
                norms_mm = torch.mm(norms, torch.transpose(norms, 0, 1))
                similarity_matrix = torch.div(predictions.detach(),norms_mm)
            
            # minimizing loss using equation (12)
            u_loss = criterion(predictions[torch.nonzero((similarity_matrix > u).int(), as_tuple = True)], one.expand_as(predictions[torch.nonzero((similarity_matrix > u).int(), as_tuple = True)])) if len(predictions[torch.nonzero((similarity_matrix > u).int(), as_tuple = True)]) > 0 else 0
            l_loss = criterion(predictions[torch.nonzero((similarity_matrix <= l).int(), as_tuple = True)], zero.expand_as(predictions[torch.nonzero((similarity_matrix <= l).int(), as_tuple = True)])) if len(predictions[torch.nonzero((similarity_matrix <= l).int(), as_tuple = True)]) > 0 else 0
            batch_loss = u_loss + l_loss
                    
            # backward pass to get all gradients
            batch_loss.backward()

            # update weights
            optimizer.step()
            
            # print('Batch: {}\tLoss: {}'.format(batch_num, batch_loss))
            batch_num += 1
            epoch_loss += batch_loss

        end_time = datetime.now()
        print("Epoch {}: Loss: {} u ({}) and l ({}) done in {}".format(epoch,epoch_loss/batch_num, u, l, end_time - start_time))
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