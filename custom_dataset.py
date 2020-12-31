import torch

# custom class
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, classes, transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.classes = classes

    def __len__(self):
        """ Denotes the total number of samples """
        return len(self.labels)
    
    def __getitem__(self, index):
        'Generates one sample of data'
        x = self.data[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return (x, y)