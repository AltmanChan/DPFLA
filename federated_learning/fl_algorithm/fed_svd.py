import torch
import numpy as np

from federated_learning.utils.util import array_to_parameters, reshape_parameter_and_to_array, recover_parameters_shape


class FedSVD():
    def __init__(self):
        pass
        # self.global_model = global_model
        # self.local_models = local_models

    def aggregation(self, global_weight, local_weights_mask, P):
        global_weight_arr = reshape_parameter_and_to_array(global_weight)

        local_weights_arr_mask = []
        for item in local_weights_mask:
            local_weights_arr_mask.append(reshape_parameter_and_to_array(item) - global_weight_arr)

        X_mask = secure_aggregation(local_weights_arr_mask)

        new_global_weight = array_to_parameters(X_mask + global_weight_arr)
        new_global_weight = recover_parameters_shape(new_global_weight)

        return new_global_weight


def secure_aggregation(xs):
    return sum(xs) / len(xs)


def my_random(size):
    return np.random.randint(low=-10 ** 5, high=10 ** 5, size=size) + np.random.random(size)


def svd(x):
    m, n = x.shape
    if m >= n:
        return np.linalg.svd(x)
    else:
        u, s, v = np.linalg.svd(x.T)
        return v.T, s, u.T


def mask(global_weight, local_weights):
    global_weight_arr = reshape_parameter_and_to_array(global_weight)

    local_weights_arr_mask = []
    for item in local_weights:
        local_weights_arr_mask.append(reshape_parameter_and_to_array(item) - global_weight_arr)