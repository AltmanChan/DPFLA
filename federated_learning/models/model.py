import torch
from torch import nn
import torch.nn.functional as F
import torchvision as tv
from loguru import logger

from federated_learning.models.bilstm import BiLSTM
from federated_learning.models.cnn_cifar_10 import Cifar10CNN
from federated_learning.models.cnn_mnist import CNNMNIST


def setup_model(model_architecture, num_classes=None, tokenizer=None, embedding_dim=None):
    available_models = {
        "CNNMNIST": CNNMNIST,
        "BiLSTM": BiLSTM,
        "CNNCifar10": Cifar10CNN,
        "ResNet18": tv.models.resnet18,
        "VGG16": tv.models.vgg16,
        "DN121": tv.models.densenet121,
        "SHUFFLENET": tv.models.shufflenet_v2_x1_0
    }
    logger.info('--> Creating {} model.....'.format(model_architecture))
    # variables in pre-trained ImageNet models are model-specific.
    if "ResNet18" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, num_classes)
    elif "VGG16" in model_architecture:
        model = available_models[model_architecture]()
        n_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(n_features, num_classes)
    elif "SHUFFLENET" in model_architecture:
        model = available_models[model_architecture]()
        model.fc = nn.Linear(1024, num_classes)
    elif 'BiLSTM' in model_architecture:
        model = available_models[model_architecture](num_words=len(tokenizer.word_index), embedding_dim=embedding_dim)
    else:
        model = available_models[model_architecture]()

    if model is None:
        logger.error("Incorrect model architecture specified or architecture not available.")
        raise ValueError(model_architecture)
    logger.info('--> Model has been created!')
    return model
