import torch
from torch.utils import data


class PoisonedDataset(data.Dataset):
    def __init__(self, dataset, source_class=None, target_class=None):
        self.dataset = dataset
        self.source_class = source_class
        self.target_class = target_class

    def __getitem__(self, index):
        x, y = self.dataset[index][0], self.dataset[index][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.dataset)
