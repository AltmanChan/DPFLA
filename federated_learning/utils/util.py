import torch
import collections
import numpy as np


def contains_class(dataset, source_class):
    for i in range(len(dataset)):
        x, y = dataset[i]
        if y == source_class:
            return True
    return False


def reshape_parameter_layer(parameters):
    parameters['conv1.weight'] = parameters['conv1.weight'].view([25, 10])
    parameters['conv1.bias'] = parameters['conv1.bias'].view([1, 10])
    parameters['conv2.weight'] = parameters['conv2.weight'].view([500, 10])
    parameters['conv2.bias'] = parameters['conv2.bias'].view([2, 10])
    parameters['fc1.weight'] = parameters['fc1.weight'].view([1600, 10])
    parameters['fc1.bias'] = parameters['fc1.bias'].view([5, 10])
    parameters['fc2.weight'] = parameters['fc2.weight'].view([50, 10])
    parameters['fc2.bias'] = parameters['fc2.bias'].view([1, 10])
    return parameters


def recover_parameters_shape(parameters):
    parameters['conv1.weight'] = parameters['conv1.weight'].view([10, 1, 5, 5])
    parameters['conv1.bias'] = parameters['conv1.bias'].view([10])
    parameters['conv2.weight'] = parameters['conv2.weight'].view([20, 10, 5, 5])
    parameters['conv2.bias'] = parameters['conv2.bias'].view([20])
    parameters['fc1.weight'] = parameters['fc1.weight'].view([50, 320])
    parameters['fc1.bias'] = parameters['fc1.bias'].view([50])
    parameters['fc2.weight'] = parameters['fc2.weight'].view([10, 50])
    parameters['fc2.bias'] = parameters['fc2.bias'].view([10])
    return parameters


def array_to_parameters(arr):
    parameters = collections.OrderedDict()
    parameters['conv1.weight'] = torch.tensor(arr[0:25, :])
    parameters['conv1.bias'] = torch.tensor(arr[25, :])
    parameters['conv2.weight'] = torch.tensor(arr[26:526, :])
    parameters['conv2.bias'] = torch.tensor(arr[526:528, :])
    parameters['fc1.weight'] = torch.tensor(arr[528:2128, :])
    parameters['fc1.bias'] = torch.tensor(arr[2128:2133, :])
    parameters['fc2.weight'] = torch.tensor(arr[2133:2183, :])
    parameters['fc2.bias'] = torch.tensor(arr[2183, :])
    return parameters


def reshape_parameter_and_to_array(parameter):
    param_array = []
    par = reshape_parameter_layer(parameter)
    for key in par.keys():
        data = par[key].cpu().numpy()
        param_array.append(data)

    res = param_array[0]
    for item in param_array[1:]:
        res = np.concatenate([res, item], axis=0)
    return res
