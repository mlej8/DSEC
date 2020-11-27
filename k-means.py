import torch
import torch.nn as nn
import torch.nn.functional as F
from params import *
from datetime import datetime
import os

import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics

""" Experiment with K-Means on MNIST """

# Load the MNIST training and test datasets using torchvision
trainset = torchvision.datasets.MNIST(root='./mnistdata', train=True, download=True)
X = trainset.data.numpy()
Y = trainset.targets.numpy()

# convert each image to 1 dim array
X = X.reshape(len(X), -1)

# normalize the data from 0 to 1
X = X.astype(float) / 255.

# create model
mini_batch_kmeans = MiniBatchKMeans(n_clusters = len(trainset.classes))
kmeans = KMeans(n_clusters = len(trainset.classes))

# fit the model 
kmeans.fit(X)
mini_batch_kmeans.fit(X)

# output labels
labels = kmeans.labels_
mini_batch_labels = mini_batch_kmeans.labels_

# validate results
nmi = metrics.normalized_mutual_info_score(Y, labels)
ari = metrics.adjusted_rand_score(Y, labels)
mb_nmi = metrics.normalized_mutual_info_score(Y, mini_batch_labels)
mb_ari = metrics.adjusted_rand_score(Y, mini_batch_labels)

# write results
model_name = "kmeans"
PATH =  './results/{0}-{1}'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S"))
if not os.path.exists('./results'):
    os.makedirs("results")
with open(PATH, "w") as f:
    f.write("KMeans\n")
    f.write(f'NMI: {nmi}\n')
    f.write(f'ARI: {ari}\n')   
    f.write("MiniBatchKMeans\n")
    f.write(f'NMI: {mb_nmi}\n')
    f.write(f'ARI: {mb_ari}')   