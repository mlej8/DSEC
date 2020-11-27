from sklearn import metrics
import torch
from datetime import datetime
import os
from params import *

""" 
File containing evaluation metrics for DSEC.
"""
   
def cluster(dataset, dnn, PATH, model_name):

    # see if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load model
    dnn.load_state_dict(torch.load(PATH, map_location=torch.device("cpu")))
    dnn.to(device)

    # load all the images
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(),shuffle=True, num_workers=num_workers)

    nmi = 0
    ari = 0

    with torch.no_grad():
        data,labels = iter(dataloader).next()

        # move data to correct device
        data = data.to(device)
                
        # clustering labels can be inferred via the learned indicator features purely, which are k-dimensional one-hot vectors ideally
        indicator_features = dnn(data)
        
        # Take predictions
        predicted = [torch.argmax(indicator_feature) for indicator_feature in indicator_features]

        nmi = metrics.normalized_mutual_info_score(labels_true=labels, labels_pred=predicted)
        ari = metrics.adjusted_rand_score(labels_true=labels, labels_pred=predicted)

    # save model and create the models directory if not exist
    PATH =  './results/{0}-{1}'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S"))
    if not os.path.exists('./results'):
        os.makedirs("results")
    with open(PATH, "w") as f:
        f.write(f'NMI of the network on the {len(dataset)} test images: {nmi}\n')
        f.write(f'ARI of the network on the {len(dataset)} test images: {ari}')   