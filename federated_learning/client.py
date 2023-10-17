import copy
import time

import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger
import torch

from federated_learning.attack_alg import label_flipping, gaussian_attack
from federated_learning.optimizer import PerturbedGradientDescent


class Client():
    # Class variable shared among all the instances
    _performed_attacks = 0

    @property
    def performed_attacks(self):
        return type(self)._performed_attacks

    @performed_attacks.setter
    def performed_attacks(self, val):
        type(self)._performed_attacks = val

    def __init__(self, client_id, client_pseudonym, local_data, labels, criterion,
                 device, local_epochs, local_bs, local_lr,
                 local_momentum, client_type='honest'):

        self.client_id = client_id
        self.client_pseudonym = client_pseudonym
        self.local_data = local_data
        self.labels = labels
        self.criterion = criterion
        self.device = device
        self.local_epochs = local_epochs
        self.local_bs = local_bs
        self.local_lr = local_lr
        self.local_momentum = local_momentum
        self.client_type = client_type

    # ======================================= Start of training function ===========================================================#
    def participant_update(self, global_epoch, model, attack_type='no_attack', malicious_behavior_rate=0,
                           source_class=None, target_class=None, dataset_name=None, untarget=False):


        if dataset_name == 'MNIST':
            backdoor_pattern = torch.tensor([[2.8238, 2.8238, 2.8238],
                                             [2.8238, 2.8238, 2.8238],
                                             [2.8238, 2.8238, 2.8238]])
        elif dataset_name == 'CIFAR10':
            backdoor_pattern = torch.tensor([[[2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141],
                                              [2.5141, 2.5141, 2.5141]],

                                             [[2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968],
                                              [2.5968, 2.5968, 2.5968]],

                                             [[2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537],
                                              [2.7537, 2.7537, 2.7537]]])

        x_offset, y_offset = backdoor_pattern.shape[0], backdoor_pattern.shape[1]


        if untarget:
            timestamp = int(time.time())
            target_class = timestamp % 10
            if target_class == source_class and target_class != 9:
                target_class += 1
            if target_class == source_class and target_class == 9:
                target_class -= 1

        epochs = self.local_epochs
        train_loader = DataLoader(self.local_data, self.local_bs, shuffle=True, drop_last=True)
        attacked = 0
        # Get the poisoned training data of the client in case of label-flipping or backdoor attacks
        if (attack_type == 'label_flipping') and (self.client_type == 'attacker'):
            r = np.random.random()
            if r <= malicious_behavior_rate:
                if dataset_name != 'IMDB':
                    poisoned_data = label_flipping(self.local_data, source_class, target_class)
                    train_loader = DataLoader(poisoned_data, self.local_bs, shuffle=True, drop_last=True)
                self.performed_attacks += 1
                attacked = 1
                logger.info('Label flipping attack launched by ' + str(self.client_pseudonym) + ' to flip class '
                            + str(source_class) + ' to class ' + str(target_class))
        lr = self.local_lr

        server_model = copy.deepcopy(model)

        if dataset_name == 'IMDB':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr,
                                  momentum=self.local_momentum, weight_decay=5e-4)

        optimizer = PerturbedGradientDescent(model.parameters(), lr=lr, mu=0.001)

        model.train()
        epoch_loss = []
        client_grad = []
        t = 0

        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                if dataset_name == 'IMDB':
                    target = target.view(-1, 1) * (1 - attacked)

                if (attack_type == 'backdoor') and (self.client_type == 'attacker') and (
                        np.random.random() <= malicious_behavior_rate):
                    pdata = data.clone()
                    ptarget = target.clone()
                    keep_idxs = (target == source_class)
                    pdata = pdata[keep_idxs]
                    ptarget = ptarget[keep_idxs]
                    pdata[:, :, -x_offset:, -y_offset:] = backdoor_pattern
                    ptarget[:] = target_class
                    data = torch.cat([data, pdata], dim=0)
                    target = torch.cat([target, ptarget], dim=0)

                output, features = model(data, return_features=True)
                loss = self.criterion(output, target)

                loss.backward()
                epoch_loss.append(loss.item())
                # get gradients
                cur_time = time.time()
                for i, (name, params) in enumerate(model.named_parameters()):
                    if params.requires_grad:
                        if epoch == 0 and batch_idx == 0:
                            client_grad.append(params.grad.clone())
                        else:
                            client_grad[i] += params.grad.clone()
                t += time.time() - cur_time
                optimizer.step(model.parameters(), self.device)
                model.zero_grad()
                optimizer.zero_grad()

            # print('Train epoch: {} \tLoss: {:.6f}'.format((epochs+1), np.mean(epoch_loss)))

        if (attack_type == 'gaussian' and self.client_type == 'attacker'):
            update, flag = gaussian_attack(model.state_dict(), self.client_pseudonym,
                                           malicious_behavior_rate=malicious_behavior_rate, device=self.device)
            if flag == 1:
                self.performed_attacks += 1
                attacked = 1
            model.load_state_dict(update)

        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        # print("Number of Attacks:{}".format(self.performed_attacks))
        # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        model = model.cpu()
        return model.state_dict(), client_grad, model, np.mean(epoch_loss), attacked, t
