from torch.utils import data


class CustomDataset(data.Dataset):
    def __init__(self, dataset, indices, source_class=None, target_class=None):
        self.dataset = dataset
        self.indices = indices
        self.source_class = source_class
        self.target_class = target_class
        self.contains_source_class = False

    def __getitem__(self, index):
        x, y = self.dataset[int(self.indices[index])][0], self.dataset[int(self.indices[index])][1]
        if y == self.source_class:
            y = self.target_class
        return x, y

    def __len__(self):
        return len(self.indices)