import numpy as np
import torch

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
    loss = 0 
    for key in D:
        if key in ind_coef_matrix:
            # implementation of equation (12): L(r_ij, f(x_i;w) dot f(x_j;w))
            estimated_similarity = torch.dot(patterns[key[0]], patterns[key[1]])
            loss += torch.pow(torch.linalg.norm(torch.Tensor([D[key] - estimated_similarity]), 2), 2) + (u - l)
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
        for j in range(i, size):
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

def cp_constraint(indicator_feature, p):
    """ Implmentation of equation (11): c_p constraint """
    # TODO: should I put this layer in network declaration or in algo with sequential?
    # Find max value of indicator feature
    maximum = torch.max(indicator_feature)

    # Equation 11a: substract max value to all element of indicator feature and take the exponent 
    intermediate = torch.exp(indicator_feature - maximum)
    
    #  equation 11b: divide every intermediate element by the p-norm of the intermediate feature vector
    output = torch.div(intermediate, torch.linalg.norm(intermediate, ord=p))

    return output

def weights_init(layer):
    """ initialize weights using normalized Gaussian initialization strategy """
    # TODO implement this initialization strategy
    if isinstance(layer, nn.Conv2d):
        torch.nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu;))
        if m.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)

    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
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
    for layer in dnn:
        weights_init(layer)
    
    # In DSEC, the devised constraint layer is always applied to DNNs 
    model = torch.nn.Sequential(dnn, cp_constraint)

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)

    # learning rate
    lr = 0.001
    # define optimizer
    optimizer = torch.optim.RMSprop(lr=lr)

    while u >= l:

        # tracking total loss
        total_loss = 0.0

        for batch in dataloader:

            # initialize indicator features (forward pass)
            output = net(training_set)

            # clear the parameter gradients
            optimizer.zero_grad()

            # construct pairwise similarities dataset, select training data from batch using formula (8), e.g. define labels for each indicator feature pair
            D = construct_dataset(output) # NOTE: DNN is shared 

            # compute indicator coefficient matrix for batch
            ind_coef_matrix = construct_indicator_coefficient_matrix(D)

            # QUESTION: update weights by minimizing (12) how?? optimizer.step()?
            loss = compute_loss(D, ind_coef_matrix)

            # accumulate loss
            total_loss += loss

            # backward pass to get all gradients
            loss.backward()

            # weights are updated on each batch that is gradually sampled from the original dataset
            optimizer.step()

        # update u and l: s(u,l) = u - l
        u = u - lr
        l = l + lr

    # output clusters
    labels = []
    for pattern in dataset.data:
        
        # clustering labels can be inferred via the learned indicator features purely, which are k-dimensional one-hot vectors ideally
        indicator_feature = net(pattern)
        
        # patterns are clustered by locating the largest response of indicator feature
        index = torch.argmax(indicator_feature)

        labels.append(index)

    return labels

# TODO Implement the constraint function
# TODO Implement the neural network for MNIST and CIFAR10
# TODO Investigate the difference between using each batch to create indicator features or create at beginning.
# TODO call evaluation funciton here ACC, NMI, etc.
# TODO create a helper funciton to use true label to report predicted labels to get ACC.
# TODO: optimizations of network's weights and u and l are alternating iterative performed. QUestion: What does this mean...
# turn = 0
