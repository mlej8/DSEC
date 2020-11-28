""" File storing configurations variables """

# num of workers
num_workers = 4

# p coefficient
p = 1

# batch size
batch_size = 32

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