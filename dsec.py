import numpy as np
from datetime import datetime
import torch
import pickle
"""
@authors: Michael Li and William Zhang
@email: er.li@mail.mcgill.ca 
@email: william.zhang2@mail.mcgill.ca 
@description: This file contains an implementation of Deep Self-Evolution Clustering algorithm in PyTorch
"""

########################
### Helper functions ###
########################

def compute_loss(patterns, D, u, l):
    """ Objective function of DSEC """
    # loss = torch.tensor(0., requires_grad=True)
    loss = 0
    for key in D:
        # minimizing loss using equation (12)
        loss += torch.pow(torch.linalg.norm(D[key] - torch.dot(patterns[key[0]], patterns[key[1]]), ord=2, dim=0), 2)
    return loss 

def construct_dataset(patterns, u, l):
    """ 
    Given an unlabeled dataset and the predefined number of clusters k, where x_i indicates the ith pattern, DSEC manages the clustering task by investigating similarities between pairwise patterns.
    To consider similarities, we transform the dataset into a binary pairwise classification problem.
    We turn a dataset into D = {{x_i, x_j, r_ij}} where x_i and x_j are unlabeled patterns (inputs) and r_ij is a binary variable which says x_i and x_j belong to the same cluster or not. 
    """
    size = len(patterns)
    D = dict()
    for i in range(size):
        for j in range(size):
            label = labeled_pairwise_patterns_selection(patterns[i], patterns[j], u, l)
            if label is not None: 
                D[(i,j)] =  label
    return D

def labeled_pairwise_patterns_selection(indicator_feature1, indicator_feature2, u, l):
    """ Implementation of the pairwise labelling algorithm described in the paper. """
    # determine similarity between two labels to create label
    similarity = torch.nn.CosineSimilarity(dim=0)(indicator_feature1, indicator_feature2)
    if similarity > u:
        return 1
    elif similarity <= l:
        return 0
    else:
        return None # similarity between x_i and x_j is ambiguous, thus this pair will be omitted during training

def cp_constraint(indicator_features, p):
    """ Implementation of equation (11): c_p constraint """
    output = []

    for indicator_feature in indicator_features: 
        # Find max value of indicator feature
        maximum = torch.max(indicator_feature)

        # Equation 11a: substract max value to all element of indicator feature and take the exponent 
        intermediate = torch.exp(indicator_feature - maximum)
        
        #  equation 11b: divide every intermediate element by the p-norm of the intermediate feature vector
        output.append(torch.div(intermediate, torch.linalg.norm(intermediate, ord=p)))

    return output

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

def dsec(dataset, dnn, model_name, p=1, initialized=False):
    """ Takes as input a PyTorch dataset and a DNN model """
    
    use_cuda = torch.cuda.is_available()

    # initial variables
    num_clusters = len(dataset.classes) 
    u = 0.95
    l = 0.80
    batch_size = 32

    if not initialized:
        # initialize weights using normalized Gaussian initialization strategy 
        for key in dnn._modules:
            weights_init(dnn._modules[key])

    if use_cuda:
        dnn.to("cuda")
        print("Training on GPU!")

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=4)

    # learning rate
    lr = 0.001
    # define optimizer
    optimizer = torch.optim.RMSprop(dnn.parameters(),lr=lr)

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

            if use_cuda:
                # send data to gpu
                data = data.to('cuda')

            # initialize indicator features (forward pass)
            output = dnn(data, p)

            # construct pairwise similarities dataset, select training data from batch using formula (8), e.g. define labels for each indicator feature pair
            D = construct_dataset(output, u, l) # NOTE: DNN is shared 

            # compute loss
            loss = compute_loss(output, D, u, l)

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

    # save model
    PATH =  './models/{0}-{1}.pth'.format(model_name, datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
    torch.save(dnn.state_dict(), PATH)

    # returning the model path
    return PATH

# TODO Investigate the difference between using each batch to create indicator features or create at beginning.
# TODO call evaluation funciton here ACC, NMI, etc. create a helper funciton to use true label to report predicted labels to get ACC.
# TODO : parallelize some parts.. move training to GPU
# TODO use logging for logs ?