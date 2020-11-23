from desc import *
from datetime import datetime
import torch
import pickle
########################
###  DSEC algorithm  ###
########################

def dsec_gpu(dataset, dnn):
    """ Takes as input a PyTorch dataset and a DNN model """
    if torch.cuda.is_available()   
        # initial variables
        num_clusters = len(dataset.classes) 
        u = 0.95
        l = 0.80
        batch_size = 32
        p = 1

        # initialize weights using normalized Gaussian initialization strategy 
        for key in dnn._modules:
            weights_init(dnn._modules[key])

        # send network on gpu
        dnn.to("cuda")
        
        # load all the images
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=2)

        # learning rate
        lr = 0.001
        # define optimizer
        optimizer = torch.optim.RMSprop(dnn.parameters(),lr=lr)

        # time tracker
        start_time = datetime.now()

        while u >= l:

            # tracking total loss
            total_loss = 0.0

            for iteration, (data,labels) in enumerate(dataloader):

                # send data to gpu
                data = data.to('cuda')

                # initialize indicator features (forward pass)
                output = dnn(data, p)

                # clear the parameter gradients
                optimizer.zero_grad()

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
                    print('Loss: {}\tIteration: {}'.format(total_loss/500, iteration))
                    total_loss = 0.0

            end_time = datetime.now()
            print("Done with u ({}) and l ({}) in {}".format(u,l, end_time - start_time))
            start_time = end_time
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
# TODO optimizations of network's weights and u and l are alternating iterative performed. Question: What does this mean....???
# TODO move training to GPU
# TODO add something to keep track of time during training .. 