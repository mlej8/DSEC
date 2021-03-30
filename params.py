""" File storing configurations variables """

# num of workers
num_workers = 2

# p coefficient
p = 2

# batch size
batch_size = 1000

# upper threshold
u = 0.99

# lower threshold
l = 0.80 

# learning rate for optimizer
optimizer_lr = 0.001

# number of epochs
e = 50

# learning rate for DSEC
lr = (u-l)/(2*e)

#number of epochs for pretraining
pretrain_epoch = 15

pretrain_batch_size = 100