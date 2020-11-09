import numpy as np
from numpy import linalg as LA
from scipy.spatial import distance

"""
@authors: Michael Li and William Zhang
@email: er.li@mail.mcgill.ca 
@email: william.zhang2@mail.mcgill.ca 
@description: This file contains an implementation of Deep Self-Evolution Clustering algorithm in PyTorch
"""

########################
### Helper functions ###
########################
def compute_loss(D, ind_coef_matrix):
    """ Objective function of DSEC """
    loss = 0 
    for key in D:
        if key in ind_coef_matrix:
            # implementation of equation (12): L(r_ij, f(x_i;w) dot f(x_j;w))
            torch.pow(torch.linalg.norm(torch.Tensor([D[key] - torch.dot(patterns[key[0]], patterns[key[1]])]), 2), 2)
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
    if similarity_estimation(indicator_feature1, indicator_feature2) > u:
        return 1
    elif similarity_estimation(indicator_feature1, indicator_feature2) <= l:
        return 0 
    else:
        return None # similarity between x_i and x_j is ambiguous, thus this pair will be omitted during training

def similarity_estimation(indicator_feature1, indicator_feature2):
    return distance.cosine(indicator_feature1, indicator_feature2)

def construct_indicator_coefficient_matrix(dataset):
    ind_coef_matrix = dict()
    for key in dataset:
        if dataset[key] is not None:    
            ind_coef_matrix[key] = 1
    return ind_coef_matrix

def dsec(patterns, dataloader, net):
    
    # initialize variables
    net = 0 # TODO: randomly initialize network's weights
    num_clusters = patterns.labels # k the number of clusters
    u = 0.95
    l = 0.05
    lr = 0.01

    while not l > u:
        for batch in dataloader:

            # QUESTION: should we get find all indicator features for each pattern here first? It would be something like a forward pass 
            # output = net(batch)

            # select training data from batch using formula (8), e.g. define labels for each indicator feature pair
            D = construct_dataset(batch) # NOTE: DNN is shared 

            # compute indicator coefficient matrix for batch
            ind_coef_matrix = construct_indicator_coefficient_matrix(D)

            # QUESTION: update weights by minimizing (12) how?? optimizer.step()?
            loss = 

    # Last Step: Use true label to report predicted labels to get ACC.
    # output clusters 
    labels = []
    for pattern in patterns:
        ind_feature = net(pattern)
        index = torch.max(ind_feature, 0)
        labels.append(index)


# QUESTION: how can we apply constraint at (3) to DNN ? 

