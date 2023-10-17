import numpy as np
import torch

eps = np.finfo(float).eps


def gaussian_attack(update, client_pseudonym, malicious_behavior_rate=0,
                    device='cpu', attack=False, mean=0.0, std=0.5):
    flag = 0
    for key in update.keys():
        r = np.random.random()
        if r <= malicious_behavior_rate:
            # print('Gausiian noise attack launched by ', client_pseudonym, ' targeting ', key, i+1)
            noise = torch.cuda.FloatTensor(update[key].shape).normal_(mean=mean, std=std)
            flag = 1
            update[key] += noise
    return update, flag
