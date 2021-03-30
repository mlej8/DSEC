import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
from params import *
from custom_dataset import Dataset

def weights_init(layer):
    """ Initialize weights using normal initialization strategy """
    if isinstance(layer, torch.nn.Conv2d):
        torch.nn.init.normal_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias.data)

    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def pretrain(augmented_dataset,dataset, dnn, model_name, initialized=False, pretrained_model=None):
    """ Takes as input a PyTorch dataset and a DNN model """
    
    # log which model we are running
    print(f"Pretraining DSEC on {model_name}.\nStarted experiment on {datetime.now().strftime('%b-%d-%H-%M-%S')}")

    # see if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # set epoch count 
    epoch = 1
    start_epoch = 1

    if not initialized:
        # initialize weights using normalized Gaussian initialization strategy 
        for key in dnn._modules:
            weights_init(dnn._modules[key])

        # create folder for storing weights if it does not exist
        weights_folder = './models/{}/{}/pretrain'.format(model_name, datetime.now().strftime("%b-%d-%H-%M-%S")) 
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)

    else:
        path = pretrained_model.split("/")
        start_epoch = int(path[-1].split(".")[0][5:])+1 #Starts at 5, example: epoch10 we want to grab 10
        dnn.load_state_dict(torch.load(pretrained_model, map_location=torch.device('cpu')))
        weights_folder = "/".join(path[:-1])

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
    original_dataloader = torch.utils.data.DataLoader(dataset, batch_size=pretrain_batch_size, shuffle=False, num_workers=num_workers)

    # define optimizer
    optimizer = torch.optim.RMSprop(dnn.parameters(),lr=optimizer_lr)

    # time tracker
    start_time = datetime.now()

    # set dnn in training mode
    dnn.train()

    # create BCE loss (does not follow paper but follows their code)
    loss = nn.MSELoss()

    for epoch in range(start_epoch, pretrain_epoch+start_epoch):

        # track number of batch
        batch_num = 1
        epoch_loss = 0
        y_train = []
        for original_data, original_label in original_dataloader:
            original_data = original_data.to(device)
            original_output = dnn(original_data).detach()
            y_train.extend(original_output.to("cpu").numpy())

        augmented_dataset = Dataset(augmented_dataset.data, y_train , augmented_dataset.classes, transform=augmented_dataset.transform)
        augmented_dataloader = torch.utils.data.DataLoader(augmented_dataset, batch_size=256, shuffle=True, num_workers=num_workers)
        for sub_epoch in range(1, 5):
            sub_epoch_loss = 0
            sub_epoch_batch_num = 1
            for augmented_data, augmented_label in augmented_dataloader:

                # clear the parameter gradients
                optimizer.zero_grad()

                # send data to correct device
                augmented_data = augmented_data.to(device)
                augmented_label = augmented_label.to(device)

                # initialize indicator features (forward pass)
                augmented_output = dnn(augmented_data)
                
                batch_loss = loss(augmented_output, augmented_label)

                # backward pass to get all gradients
                batch_loss.backward()

                # update weights
                optimizer.step()
                
                # print('Batch: {}\tLoss: {}'.format(batch_num, batch_loss))
                sub_epoch_batch_num += 1
                sub_epoch_loss += batch_loss

            end_time = datetime.now()
            print("Epoch: {} \t Sub Epoch {}: Loss: {} \t done in {}".format(epoch, sub_epoch, sub_epoch_loss/(sub_epoch_batch_num), end_time - start_time))
            start_time = end_time
        
        # save weights
        PATH = weights_folder + "/epoch" + str(epoch) +  ".pth"
        try:
            state_dict = dnn.module.state_dict()
        except AttributeError:
            state_dict = dnn.state_dict()
        torch.save(state_dict, PATH)
        
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