from random import choice


def replace_1_with_7():
    return {'source_class': 1,
            'target_class': 7}


def replace_6_with_9():
    return {'source_class': 6,
            'target_class': 9}


def replace_dog_with_cat():
    return {'source_class': 5,
            'target_class': 3}


def fed_svd():
    return 'fed_svd'

def DPFLA():
    return 'DPFLA'


def fed_avg():
    return 'fedavg'


def simple_median():
    return 'median'


def trimmed_mean():
    return 'tmean'


def multi_krum():
    return 'mkrum'


def fools_gold():
    return 'foolsgold'


def run_mnist():
    return {'dataset_name': 'MNIST', 'model_name': 'CNNMNIST'}


def get_mnist_labels_dict():
    return {'Zero': 0,
            'One': 1,
            'Two': 2,
            'Three': 3,
            'Four': 4,
            'Five': 5,
            'Six': 6,
            'Seven': 7,
            'Eight': 8,
            'Nine': 9}


def run_cifar10():
    return {'dataset_name': 'CIFAR10', 'model_name': 'CNNCifar10'}


def get_cifar10_labels_dict():
    return {'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9}
