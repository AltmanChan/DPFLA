from federated_learning.datasets import PoisonedDataset


def label_flipping(data, source_class, target_class):
    poisoned_data = PoisonedDataset(data, source_class, target_class)
    return poisoned_data


