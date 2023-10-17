import json
import random
import numpy as np
import torch
import torch.nn as nn

# Setting the seed for Torch
SEED = 7
random.seed(SEED)


class Arguments:

    def __init__(self, logger):
        self.logger = logger

        self.batch_size = 64
        self.test_batch_size = 500
        self.lr = 0.01
        self.momentum = 0.9
        self.device = "cuda"  # cpu\cuda
        self.log_interval = 100
        self.seed = SEED

        self.global_rounds = 100
        self.local_epochs = 2

        self.source_class = 1
        self.target_class = 7

        self.dataset_name = "MNIST"
        self.model_name = "CNNMNIST"
        self.labels_dict = {'Zero': 0,
                            'One': 1,
                            'Two': 2,
                            'Three': 3,
                            'Four': 4,
                            'Five': 5,
                            'Six': 6,
                            'Seven': 7,
                            'Eight': 8,
                            'Nine': 9}
        # self.dataset_name = "CIFAR10"
        # self.model_name = "CNNCifar10"
        # self.labels_dict = {'airplane': 0,
        #                     'automobile': 1,
        #                     'bird': 2,
        #                     'cat': 3,
        #                     'deer': 4,
        #                     'dog': 5,
        #                     'frog': 6,
        #                     'horse': 7,
        #                     'ship': 8,
        #                     'truck': 9}

        self.dd_type = "IID"  # IID\NON_IID
        self.alpha = 1

        self.num_workers = 50
        self.frac_workers = 1
        self.malicious_rate = 0

        self.loss_function = nn.CrossEntropyLoss()

        self.num_classes = 10  # number of classes in an experiment

        self.class_per_workers = 10
        self.samples_per_class = 582
        self.RATE_UNBALANCE = 1

        self.rule = "fedavg"
        self.attack_type = "no_attack"
        self.malicious_behavior_rate = 1

    def get_rule(self):
        return self.rule

    def set_rule(self, rule):
        self.rule = rule

    def get_attack_type(self):
        return self.attack_type

    def set_attack_type(self, attack_type):
        self.attack_type = attack_type

    def get_malicious_behavior_rate(self):
        return self.malicious_behavior_rate

    def set_malicious_behavior_rate(self, malicious_behavior_rate):
        self.malicious_behavior_rate = malicious_behavior_rate

    # batch_size的get方法和set方法
    def get_batch_size(self):
        return self.batch_size

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    # test_batch_size的get方法和set方法
    def get_test_batch_size(self):
        return self.test_batch_size

    def set_test_batch_size(self, test_batch_size):
        self.test_batch_size = test_batch_size

    # lr的get方法和set方法
    def get_lr(self):
        return self.lr

    def set_lr(self, lr):
        self.lr = lr

    # momentum的get方法和set方法
    def get_momentum(self):
        return self.momentum

    def set_momentum(self, momentum):
        self.momentum = momentum

    # cuda的get方法和set方法
    def get_cuda(self):
        return self.cuda

    def set_cuda(self, cuda):
        self.cuda = cuda

    # log_interval的get方法和set方法
    def get_log_interval(self):
        return self.log_interval

    def set_log_interval(self, log_interval):
        self.log_interval = log_interval

    def get_seed(self):
        return self.seed

    def set_seed(self, seed):
        self.seed = seed

    # global_rounds的get方法和set方法
    def get_global_rounds(self):
        return self.global_rounds

    def set_global_rounds(self, global_rounds):
        self.global_rounds = global_rounds

    # local_epochs的get方法和set方法
    def get_local_epochs(self):
        return self.local_epochs

    def set_local_epochs(self, local_epochs):
        self.local_epochs = local_epochs

    # dataset_name的get方法和set方法
    def get_dataset_name(self):
        return self.dataset_name

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    # model_name的get方法和set方法
    def get_model_name(self):
        return self.model_name

    def set_model_name(self, model_name):
        self.model_name = model_name

    # dd_type的get方法和set方法
    def get_dd_type(self):
        return self.dd_type

    def set_dd_type(self, dd_type):
        self.dd_type = dd_type

    # alpha的get方法和set方法
    def get_alpha(self):
        return self.alpha

    # num_workers
    def get_num_workers(self):
        return self.num_workers

    def set_num_workers(self, value):
        self.num_workers = value

    # num_workers_per_round
    def get_frac_workers(self):
        return self.frac_workers

    def set_frac_workers(self, value):
        self.frac_workers = value

    # loss_function
    def get_loss_function(self):
        return self.loss_function

    def set_loss_function(self, value):
        self.loss_function = value

    # num_classes
    def get_num_classes(self):
        return self.num_classes

    def set_num_classes(self, value):
        self.num_classes = value

    # labels_dict
    def get_labels_dict(self):
        return self.labels_dict

    def get_class_per_workers(self):
        return self.class_per_workers

    def get_samples_per_class(self):
        return self.samples_per_class

    def get_rate_unbalance(self):
        return self.RATE_UNBALANCE

    def get_malicious_rate(self):
        return self.malicious_rate

    def set_malicious_rate(self, malicious_rate):
        self.malicious_rate = malicious_rate

    def get_device(self):
        return self.device

    def get_source_class(self):
        return self.source_class

    def set_source_class(self, source_class):
        self.source_class = source_class

    def get_target_class(self):
        return self.target_class

    def set_target_class(self, target_class):
        self.target_class = target_class

    def set_labels_dict(self, labels_dict):
        self.labels_dict = labels_dict

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Global Rounds: {}\n".format(self.global_rounds) + \
               "Local Epochs: {}\n".format(self.local_epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "Device: {}\n".format(self.device) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Dataset: {}\n".format(self.dataset_name) + \
               "Model Name: {}\n".format(self.model_name) + \
               "Data distribution: {}\n".format(self.dd_type) + \
               "Aggregation rule: {}\n".format(self.rule) + \
               "Attack Type: {}\n".format(self.attack_type) + \
               "Malicious Rate: {}%\n".format(np.round(self.malicious_rate * 100, 2)) + \
               "Malicious Behavior Rate: {}\n".format(self.malicious_behavior_rate)
