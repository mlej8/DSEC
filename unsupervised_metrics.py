from sklearn import metrics
import torch
from datetime import datetime
import os
from params import *
from collections import OrderedDict
import numpy as np
from scipy.optimize import linear_sum_assignment

""" 
File containing evaluation metrics for DSEC.
"""
def cluster(dataset, dnn, PATH, model_name):
    
    # see if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load params 
    dnn.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    
    # move model to appropriate device
    dnn.to(device)

    # set dnn in evaluation mode
    dnn.eval()

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)

    # placeholders for labels and predictions
    predictions = []
    labels = []

    with torch.no_grad():
        for data, batch_labels in dataloader:
            
            # move data to correct device
            data = data.to(device)
                    
            # clustering labels can be inferred via the learned indicator features purely, which are k-dimensional one-hot vectors ideally
            indicator_features = dnn(data)

            # find clusters
            maximums, indices = torch.max(indicator_features, 1)

            labels.extend(batch_labels.numpy())
            predictions.extend(indices.to("cpu").numpy())
    
    # transform to numpy array
    predictions = np.array(predictions)
    labels = np.array(labels)

    # save model and create the models directory if not exist
    PATH =  './results/{0}-{1}'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists('./results'):
        os.makedirs("results")
    with open(PATH, "w") as f:
        f.write(f'NMI of DSEC {model_name}: {metrics.normalized_mutual_info_score(labels_true=labels, labels_pred=predictions)}\n')
        f.write(f'ARI of DSEC {model_name}: {metrics.adjusted_rand_score(labels_true=labels, labels_pred=predictions)}\n')  
        f.write(f'ACC of DSEC {model_name}: {acc(labels, predictions)}')  

def acc(labels, predictions):
    """ Implementation of clustering accuracy """
    assert len(predictions) == len(labels)

    # build the confusion matrix
    cm = metrics.confusion_matrix(labels, predictions)

    # transform confusion matrix into a cost matrix, because the linear_assignment minizes the cost
    s = np.max(cm)
    cost_matrix = -cm + s
    
    # we run the linear sum assignment (finding minimum weight matching in bipartite graphs) problem on the cost matrix (graph)
    indexes = linear_sum_assignment(cost_matrix)

    # get the reordered labels
    true_labels, cluster_labels = sorted(indexes, key=lambda x: x[0])

    # using best match to reorder mapping between predictions -> labels 
    reordered_cm = cm[:, cluster_labels]

    # divide summation of diagonal by whole matrix sum
    return np.trace(reordered_cm) / np.sum(reordered_cm)