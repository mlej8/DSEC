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

def compute_loss(patterns, D, ind_coef_matrix, u, l):
    """ Objective function of DSEC """
    # loss = torch.tensor(0., requires_grad=True)
    loss = 0
    for key in D:
        if key in ind_coef_matrix:
            # implementation of equation (12): L(r_ij, f(x_i;w) dot f(x_j;w))
            estimated_similarity = torch.dot(patterns[key[0]], patterns[key[1]])
            diff = D[key] - estimated_similarity
            loss += torch.pow(torch.linalg.norm(diff, ord=2, dim=0), 2) + (u - l)
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
        for j in range(i+1, size):
            # TODO modify this to only store 1/0 to save memory
            D[(i,j)] =  labeled_pairwise_patterns_selection(patterns[i], patterns[j], u, l)
    return D

def labeled_pairwise_patterns_selection(indicator_feature1, indicator_feature2, u, l):
    """ Implementation of the pairwise labelling algorithm described in the paper. """
    similarity = similarity_estimation(indicator_feature1, indicator_feature2)
    if similarity > u:
        return 1
    elif similarity <= l:
        return 0
    else:
        return None # similarity between x_i and x_j is ambiguous, thus this pair will be omitted during training

def similarity_estimation(indicator_feature1, indicator_feature2):
    return torch.nn.CosineSimilarity(dim=0)(indicator_feature1, indicator_feature2)

def construct_indicator_coefficient_matrix(dataset):
    ind_coef_matrix = dict()
    for key in dataset:
        if dataset[key] is not None:    
            ind_coef_matrix[key] = 1
    return ind_coef_matrix

def cp_constraint(indicator_features, p):
    """ Implmentation of equation (11): c_p constraint """
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
    """ initialize weights using normalized Gaussian initialization strategy """
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

def dsec(dataset, dnn):
    """ Takes as input a PyTorch dataset and a DNN model """
    
    # initial variables
    num_clusters = len(dataset.classes) 
    u = 0.95
    l = 0.05
    batch_size = 32
    p = 1

    # initialize weights using normalized Gaussian initialization strategy 
    for key in dnn._modules:
        weights_init(dnn._modules[key])

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)

    # learning rate
    lr = 0.001
    # define optimizer
    optimizer = torch.optim.RMSprop(dnn.parameters(),lr=lr)

    while u >= l:

        # tracking total loss
        total_loss = 0.0

        for iteration, (data,labels) in enumerate(dataloader):

            # initialize indicator features (forward pass)
            output = dnn(data, p)

            # clear the parameter gradients
            optimizer.zero_grad()

            # construct pairwise similarities dataset, select training data from batch using formula (8), e.g. define labels for each indicator feature pair
            D = construct_dataset(output, u, l) # NOTE: DNN is shared 

            # compute indicator coefficient matrix for batch 
            ind_coef_matrix = construct_indicator_coefficient_matrix(D)

            # compute loss
            loss = compute_loss(output, D, ind_coef_matrix, u, l)

            # accumulate loss
            total_loss += loss

            # backward pass to get all gradients
            loss.backward()

            # weights are updated on each batch that is gradually sampled from the original dataset
            optimizer.step()

            # at every 500 mini-batch, print the loss
            if iteration % 500 == 499: 
                print('Loss: {}\tIteration: {}'.format(total_loss/500, iteration))
                total_loss = 0.0

        print("Done with u ({}) and l ({})".format(u,l))
        # update u and l: s(u,l) = u - l
        u = u - lr
        l = l + lr

    # save model
    PATH =  './models/{}.pth'.format(datetime.now().strftime("%Y-%b-%d-%H-%M-%S"))
    torch.save(dnn.state_dict(), PATH)

    # output clusters
    labels = []
    for pattern in dataset.data:
        
        # clustering labels can be inferred via the learned indicator features purely, which are k-dimensional one-hot vectors ideally
        indicator_feature = dnn(pattern)
        
        # patterns are clustered by locating the largest response of indicator feature
        index = torch.argmax(indicator_feature)

        labels.append(index)

    # save predicted labels
    with open("labels.pickle", "wb") as f:
        pickle.dump(labels, f)

    correct = 0
    outof = len(dataset.targets)
    for i, label in enumerate(labels):
        if label == dataset.targets[i]:
            correct += 1

    print("{} out of {}".format(correct, outof))

    return dnn

# TODO Investigate the difference between using each batch to create indicator features or create at beginning.
# TODO call evaluation funciton here ACC, NMI, etc. create a helper funciton to use true label to report predicted labels to get ACC.
# TODO: optimizations of network's weights and u and l are alternating iterative performed. Question: What does this mean....???
