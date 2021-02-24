import torch

class GaussianNoise(object):
    """ 
    Gaussian noise implementation for PyTorch transforms.
    
    Resources:
    https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694/2
    https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    https://discuss.pytorch.org/t/where-is-the-noise-layer-in-pytorch/2887
    """
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor.add_(torch.randn(tensor.size()) * self.std + self.mean)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# Data augmentation: https://www.sciencedirect.com/science/article/pii/S0031320317304284